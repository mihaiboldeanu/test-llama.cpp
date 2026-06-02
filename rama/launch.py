import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import Config
from .core import (
    ModelInfo,
    calc_kv_cache,
    find_free_port,
    get_model_pid,
    get_pid_dir,
    get_preferred_ctx,
    is_port_used,
)
from .log import setup_logging

logger = setup_logging()


def resolve(opts: dict, family: dict, key: str, default, config=None):
    """Resolve config value: opts > family > [config] > default."""
    if key in opts:
        return opts[key]
    if family and key in family:
        return family[key]
    if config and key in config:
        return config[key]
    return default


@dataclass
class LaunchResult:
    """Result of starting a model."""

    name: str
    port: int
    pid: int
    backend: str
    ctx_size: int
    seed: int | None = None


def _pid_exists(pid: int) -> bool:
    """Return True if a process appears to be alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, grace_seconds: float = 8.0) -> bool:
    """Best-effort terminate a process, escalating to SIGKILL if needed."""
    if not pid:
        return False

    # Kill the process first, escalate to the whole group if needed.
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.2)

    # Process didn't die — try the whole group (start_new_session=True means we own it).
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.2)

    # Force kill — process first, then group.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)

    return not _pid_exists(pid)


def _wait_for_port_free(port: int, timeout: float = 5.0) -> bool:
    """Wait for a TCP port to become free."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_port_used(port):
            return True
        time.sleep(0.2)
    return not is_port_used(port)


def get_server_bin(backend: str, config: Config) -> Path:
    """Get path to llama-server binary."""
    if backend == "turboquant":
        return config.turbo_dir / "build" / "bin" / "llama-server"
    return config.llama_cpp_dir / "build" / "bin" / "llama-server"


def build_backend(
    backend: str,
    config: Config,
    force: bool = False,
    cpu_only: bool = False,
) -> None:
    """Build llama.cpp or turboquant."""
    repo_dir = config.turbo_dir if backend == "turboquant" else config.llama_cpp_dir
    nvcc = config.cuda_root / "bin" / "nvcc"

    if not cpu_only and not nvcc.exists():
        raise RuntimeError(f"nvcc not found: {nvcc}")

    server_bin = get_server_bin(backend, config)

    logger.info("Building %s%s...", backend, " (CPU only)" if cpu_only else "")

    # Git pull
    fetch_result = subprocess.run(
        ["git", "fetch", "--all", "--prune"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if fetch_result.returncode != 0:
        raise RuntimeError(f"git fetch failed: {fetch_result.stderr.strip()}")

    pull_result = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if pull_result.returncode != 0:
        raise RuntimeError(f"git pull failed: {pull_result.stderr.strip()}")

    # Check if anything actually changed
    source_changed = "Already up to date" not in pull_result.stdout

    # If nothing changed and binary exists, skip build
    if not source_changed and server_bin.exists() and not force:
        logger.info("%s already built, no new code, skipping", backend)
        return

    # Build
    build_dir = repo_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_HOME"] = str(config.cuda_root)
    env["PATH"] = f"{config.cuda_root}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{config.cuda_root}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    )

    if cpu_only:
        cmake_args = [
            "-S",
            str(repo_dir),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    else:
        cmake_args = [
            "-S",
            str(repo_dir),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES={config.get('cuda_arch', '89')}",
            "-DGGML_NATIVE=ON",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCUDAToolkit_ROOT=" + str(config.cuda_root),
            "-DCMAKE_CUDA_COMPILER=" + str(nvcc),
        ]

    cmake_result = subprocess.run(
        ["cmake"] + cmake_args, cwd=build_dir, env=env, capture_output=True, text=True
    )
    if cmake_result.returncode != 0:
        raise RuntimeError(f"cmake configure failed:\n{cmake_result.stderr}")

    build_result = subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", str(config["build_jobs"])],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if build_result.returncode != 0:
        raise RuntimeError(f"cmake build failed:\n{build_result.stderr[-500:]}")

    logger.info("Built %s", backend)


def start_model(
    model: ModelInfo,
    backend: str = "llama.cpp",
    port: int | None = None,
    ctx_size: int | None = None,
    ctk: str | None = None,
    ctv: str | None = None,
    seed: int | None = None,
    reasoning: bool | None = None,
    reasoning_budget: int | None = None,
    draft_model: str | None = None,
    draft_max: int = 16,
    draft_min: int = 8,
    spec_ngram: bool = False,
    mtp: bool = False,
    mtp_n_max: int = 3,
    mtp_n_min: int = 0,
    family_params: dict | None = None,
    config: Config | None = None,
    config_item: dict | None = None,
    foreground: bool = False,
    log_disable: bool = False,
) -> LaunchResult | None:
    """Start a llama.cpp model server."""
    config = config or Config()
    opts = config_item or {}
    server_bin = get_server_bin(backend, config)

    if not server_bin.exists():
        logger.warning("Server not found: %s, building...", server_bin)
        build_backend(backend, config)

    # Family-specific options can steer launch behavior for hard-to-fit models.
    family = model.family
    if family_params is None:
        family_params = config.families.get(family, {})

    # Per-model overrides merge on top of family config
    model_overrides = (config.get("models", {}) or {}).get(model.name, {})
    merged_opts = {**family_params, **opts}
    if model_overrides:
        merged_opts = {**merged_opts, **model_overrides}

    # Get chat_template and chat_template_file from family or config
    chat_template = merged_opts.get("chat_template", config.get("chat_template"))
    chat_template_file = merged_opts.get(
        "chat_template_file", config.get("chat_template_file")
    )

    # Get template directory from config
    template_dir = config.get("template_dir")
    if template_dir and chat_template_file:
        chat_template_file = str(Path(template_dir) / chat_template_file)

    preferred_ctx = merged_opts.get("preferred_ctx")
    if preferred_ctx is None:
        preferred_ctx = get_preferred_ctx(model.size_gb, family, config)

    # Calculate KV cache
    if ctx_size is None:
        ctx_size = preferred_ctx or get_preferred_ctx(
            model.size_gb, model.family, config
        )
        qk_type = ctk or "q8_0"
        qkv_type = ctv or "q8_0"
    else:
        qk_type = ctk or "q8_0"
        qkv_type = ctv or "q8_0"

    # Reduce ctx for MTP (extra VRAM for draft heads + verification state)
    if mtp:
        mtp_ctx_ratio = config.get("mtp_ctx_ratio", 0.8)
        ctx_size = int(ctx_size * mtp_ctx_ratio)

    # Port resolution — if explicit port given and busy, clear it first.
    # Final availability check happens right before Popen (below).
    if port is None:
        port = find_free_port(config)
    elif is_port_used(port):
        logger.info("Port %d is already in use, stopping existing model first...", port)
        stop_model(port, config)
        if not _wait_for_port_free(port):
            try:
                subprocess.run(
                    ["fuser", "-k", "-9", f"{port}/tcp"],
                    capture_output=True,
                )
            except Exception as e:
                logger.debug(f"fuser failed on port {port}: {e}")
                try:
                    pids = subprocess.run(
                        ["pgrep", "-f", f"llama-server.*{port}"],
                        capture_output=True,
                        text=True,
                    ).stdout.strip().split()
                    for p in pids:
                        try:
                            os.kill(int(p), 9)
                        except Exception as e:
                            logger.debug(f"failed to kill PID {p}: {e}")
                except Exception as e:
                    logger.debug(f"pgrep cleanup failed on port {port}: {e}")
            time.sleep(2)
            _wait_for_port_free(port)
            time.sleep(1)

        if is_port_used(port):
            logger.warning("Port %d still busy, picking another free port...", port)
            port = find_free_port(config, port + 1)

    fit_target = opts.get("fit_target")
    if fit_target is None and mtp:
        fit_target = 2560
    if fit_target is None:
        fit_target = int(
            config.get("fit_target_mib", config.get("vram_headroom_gb", 2) * 1024),
        )

    cpu_moe = resolve(opts, family_params, "cpu_moe", False)
    n_cpu_moe = resolve(opts, family_params, "n_cpu_moe", None)
    alias = resolve(opts, family_params, "alias", model.name)
    batch_size = int(resolve(opts, family_params, "batch_size", 2048, config))
    ubatch_size = int(resolve(opts, family_params, "ubatch_size", 2048, config))
    # MTP needs VRAM headroom: reduce batch sizes to minimize compute buffers
    if mtp:
        batch_size = min(batch_size, 2048)
        ubatch_size = min(ubatch_size, 512)
    threads = int(resolve(opts, family_params, "threads", config["build_jobs"], config))
    threads_batch = int(resolve(opts, family_params, "threads_batch", max(threads, 16), config))
    n_predict = int(resolve(opts, family_params, "n_predict", 2048, config))
    temp = str(resolve(opts, family_params, "temp", 0.6, config))
    top_k = str(resolve(opts, family_params, "top_k", 20, config))
    top_p = str(resolve(opts, family_params, "top_p", 0.95, config))
    min_p = str(resolve(opts, family_params, "min_p", 0.0, config))
    repeat_penalty = str(resolve(opts, family_params, "repeat_penalty", 1.0, config))
    presence_penalty = str(resolve(opts, family_params, "presence_penalty", 0.0, config))
    samplers = resolve(opts, family_params, "samplers", None)
    no_mmproj = resolve(opts, family_params, "no_mmproj", False)
    cache_ram = resolve(opts, family_params, "cache_ram", None, config)
    seed = resolve(opts, family_params, "seed", seed, config)
    if seed is not None:
        seed = int(seed)
    chat_template_kwargs = resolve(opts, family_params, "chat_template_kwargs", None, config)

    # Speculative decoding parameters
    spec_ngram_mod_size_n = resolve(opts, family_params, "spec_ngram_mod_size_n", None, config)
    spec_ngram_mod_n_min = resolve(opts, family_params, "spec_ngram_mod_n_min", None, config)
    spec_ngram_mod_n_max = resolve(opts, family_params, "spec_ngram_mod_n_max", None, config)
    spec_type = resolve(opts, family_params, "spec_type", None, config)

    # Build args
    args = [
        str(server_bin),
        "-m",
        str(model.path),
        "--host",
        str(config["host"]),
        "--port",
        str(port),
        "--alias",
        str(alias),
        "-np",
        "1",
        "--ctx-size",
        str(ctx_size),
        "-ctk",
        qk_type,
        "-ctv",
        qkv_type,
        "--flash-attn",
        "on",
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads_batch),
        "--temp",
        temp,
        "--top-k",
        top_k,
        "--top-p",
        top_p,
        "--min-p",
        min_p,
        "--repeat-penalty",
        repeat_penalty,
        "--presence-penalty",
        presence_penalty,
        "--perf",
    ]
    if log_disable:
        args.append("--log-disable")
    # MoE detection
    is_moe = any(
        kw in model.name.lower() for kw in ("a4b", "a3b", "a18b", "moe", "mixture")
    )
    if is_moe:
        args.append("--no-mmap")
    is_gemma_moe = is_moe and family == "gemma"
    # Gemma 4 MoE: avoid chat-template-file (can crash with some llama.cpp builds).
    use_chat_template = not is_gemma_moe
    if use_chat_template and chat_template:
        args.extend(["--chat-template", str(chat_template)])
    if use_chat_template and chat_template_file:
        args.extend(["--chat-template-file", str(chat_template_file)])

    if cache_ram is not None:
        args.extend(["--cache-ram", str(cache_ram)])
    if seed is not None:
        args.extend(["-s", str(seed)])
    if not config.get("context_shift", False):
        args.append("--no-context-shift")

    # Fit mode
    # Gemma 4 MoE: avoid --fit-ctx. Use --fit on + --fit-target only.
    # MTP: if model fits GPU entirely, use -ngl -1 + --fit off to leave room
    # for MTP compute buffers. If model is too large (MoE 35B+), use fit with
    # reduced context to leave VRAM for MTP draft context.
    fit_enabled = opts.get("fit", True)
    fit_ctx_enabled = opts.get("fit_ctx", True)

    mtp_vram_headroom_gb = opts.get("mtp_vram_headroom_gb", 4)  # GB to reserve for MTP
    mtp_ngl_threshold_gb = opts.get("mtp_ngl_threshold_gb", 20)  # max model size for -ngl -1

    if mtp:
        model_fits_gpu = model.size_gb < mtp_ngl_threshold_gb
        if model_fits_gpu:
            # Model fits GPU entirely: put all layers on GPU, disable fit
            # This leaves VRAM headroom for MTP compute buffers
            args.append("-ngl")
            args.append("-1")
            fit_args = [
                "--fit",
                "off",
            ]
        else:
            # Model too large for GPU (e.g. 35B MoE): use fit to balance
            # model/CPU offload, leaving room for MTP draft context
            fit_args = [
                "--fit",
                "on",
                "--fit-target",
                str(fit_target),
            ]
    elif is_gemma_moe:
        fit_args = [
            "--fit",
            "on",
            "--fit-target",
            str(fit_target),
        ]
        # Intentionally no --fit-ctx for Gemma 4 MoE
    else:
        fit_args = [
            "--fit",
            "on" if fit_enabled else "off",
            "--fit-target",
            str(fit_target),
        ]
        if fit_ctx_enabled:
            fit_args.extend(["--fit-ctx", str(ctx_size)])
    args.extend(fit_args)

    # Draft model (speculative decoding)
    if draft_model:
        args.extend(
            [
                "--model-draft",
                draft_model,
                "--spec-draft-n-max",
                str(draft_max),
                "--spec-draft-n-min",
                str(draft_min),
            ]
        )

    # MTP (Multi-Token Prediction) - requires model with MTP heads
    if mtp:
        mtp_args = [
            "--cache-type-k-draft",
            "q8_0",
            "--cache-type-v-draft",
            "q8_0",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-p-min",
            "0.75",
            "--spec-draft-n-max",
            str(mtp_n_max),
        ]
        if mtp_n_min > 0:
            mtp_args.extend(["--spec-draft-n-min", str(mtp_n_min)])
        args.extend(mtp_args)

    # Memory options
    if mtp:
        args.append("--mlock")
    if cpu_moe:
        args.append("--cpu-moe")
    elif n_cpu_moe is not None:
        args.extend(["--n-cpu-moe", str(n_cpu_moe)])

    # Speculative decoding: only for qwen/gemma or when explicitly set per-model.
    small_threshold = opts.get("small_threshold_gb", 10)
    is_small = model.size_gb < small_threshold
    spec_allowed_family = family in ("qwen", "gemma")
    spec_explicit = opts.get("spec_type") is not None
    if (
        spec_ngram
        and spec_type is not None
        and not draft_model
        and not is_small
        and (spec_allowed_family or spec_explicit)
    ):
        spec_args = [
            "--spec-type",
            str(spec_type),
        ]

        if spec_type == "ngram-mod":
            # Turboquant uses old ngram flags, llama.cpp uses new ones
            if backend == "turboquant":
                if spec_ngram_mod_size_n is not None:
                    spec_args.extend(
                        ["--spec-ngram-size-n", str(spec_ngram_mod_size_n)]
                    )
                # Use draft_min/draft_max from opts or family_params
                draft_min_val = opts.get(
                    "draft_min",
                    family_params.get("draft_min", config.get("spec_ngram_mod_n_min")),
                )
                draft_max_val = opts.get(
                    "draft_max",
                    family_params.get("draft_max", config.get("spec_ngram_mod_n_max")),
                )
                if draft_min_val is not None:
                    spec_args.extend(["--draft-min", str(draft_min_val)])
                if draft_max_val is not None:
                    spec_args.extend(["--draft-max", str(draft_max_val)])
            else:
                if spec_ngram_mod_size_n is not None:
                    spec_args.extend(
                        ["--spec-ngram-mod-n-match", str(spec_ngram_mod_size_n)]
                    )
                if spec_ngram_mod_n_min is not None:
                    spec_args.extend(
                        ["--spec-ngram-mod-n-min", str(spec_ngram_mod_n_min)]
                    )
                if spec_ngram_mod_n_max is not None:
                    spec_args.extend(
                        ["--spec-ngram-mod-n-max", str(spec_ngram_mod_n_max)]
                    )

        args.extend(spec_args)

    # Family-specific args
    # Only Qwen uses chat-template-kwargs; others rely on reasoning flags + system prompts.
    if family == "qwen":
        reasoning_budget = reasoning_budget or opts.get(
            "reasoning_budget",
            family_params.get("reasoning_budget", config.get("reasoning_budget", 1024)),
        )
        reasoning_msg = opts.get(
            "reasoning_budget_message",
            family_params.get(
                "reasoning_budget_message",
                config.get(
                    "reasoning_budget_message",
                    "I have thought enough, let's move to answering.",
                ),
            ),
        )
        enable_reasoning = reasoning if reasoning is not None else (not is_small)
        args.extend(
            [
                "-n",
                str(n_predict),
                "--reasoning",
                "on" if enable_reasoning else "off",
                "--reasoning-budget",
                str(reasoning_budget),
                "--reasoning-budget-message",
                reasoning_msg,
            ],
        )
        ctk_args = {"preserve_thinking": True}
        if chat_template_kwargs:
            try:
                existing = json.loads(chat_template_kwargs)
                ctk_args.update(existing)
            except (json.JSONDecodeError, TypeError):
                pass
        args.extend([
            "--chat-template-kwargs",
            json.dumps(ctk_args),
        ])
    elif family == "gemma" or family == "nemotron":
        args.extend(["-n", str(n_predict)])
        if no_mmproj:
            args.append("--no-mmproj")
        if samplers:
            args.extend(["--samplers", samplers])
    else:
        args.extend(["-n", str(n_predict)])

    # Final port check — TOCTOU guard right before launch
    if is_port_used(port):
        logger.warning(f"Port {port} got taken, finding replacement...")
        port = find_free_port(config)
        for i, a in enumerate(args):
            if a == "--port":
                args[i + 1] = str(port)
                break

    # Start process
    if not log_disable:
        logger.info("Starting %s on port %d...", model.name, port)

    if foreground:
        subprocess.run(args)
        return None

    proc = subprocess.Popen(
        args,
        start_new_session=True,
        stdout=subprocess.DEVNULL if log_disable else None,
        stderr=subprocess.DEVNULL if log_disable else None,
    )

    # Save PID (JSON format with backend info)
    pid_dir = get_pid_dir()
    pid_dir.mkdir(exist_ok=True)
    pid_file = pid_dir / f"model_{model.name}_{port}.pid"
    pid_data = {
        "pid": proc.pid,
        "backend": backend,
        "name": model.name,
        "port": port,
    }
    pid_file.write_text(json.dumps(pid_data))

    # Wait for ready
    for _ in range(60):
        time.sleep(2)

        # Check if process crashed
        exit_code = proc.poll()
        if exit_code is not None:
            _terminate_pid(proc.pid)
            for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
                pid_file.unlink(missing_ok=True)
            cmd_str = " ".join(args)
            raise RuntimeError(
                f"llama-server crashed with exit code {exit_code}. Command: {cmd_str}"
            )

        try:
            urllib.request.urlopen(f"http://{config['host']}:{port}/health")
            logger.info("Model loaded! (PID: %d)", proc.pid)
            return LaunchResult(
                model.name,
                port,
                proc.pid,
                backend,
                ctx_size,
                seed,
            )
        except Exception as e:
            logger.debug(f"health check failed, retrying: {e}")
            continue

    _terminate_pid(proc.pid)
    for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
        pid_file.unlink(missing_ok=True)
    raise RuntimeError("Model failed to start within 120 seconds")


def stop_model(port: int, config: Config = None) -> bool:
    """Stop model on port."""
    config = config or Config()
    pid = get_model_pid(config, port)

    if pid is None:
        # Try pgrep
        result = subprocess.run(
            ["pgrep", "-f", f"llama-server.*{port}"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            pid = int(result.stdout.strip().split()[0])

    if pid:
        try:
            stopped = _terminate_pid(pid)
            # Give process time to shut down and release port
            time.sleep(1.0)
            # Remove PID file
            pid_dir = get_pid_dir()
            for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
                pid_file.unlink(missing_ok=True)
            return stopped
        except OSError:
            pass

    return False
