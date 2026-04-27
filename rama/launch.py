import json
import os
import shutil
import signal
import subprocess
import sys
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
    is_port_used,
)
from .log import setup_logging

logger = setup_logging()


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

    # Prefer killing the whole process group when possible so child processes
    # don't survive a failed launch.
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)

    return not _pid_exists(pid)


def _wait_for_port_free(port: int, timeout: float = 10.0) -> bool:
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
    if server_bin.exists() and not force:
        logger.info("%s already built, skipping (use --force to rebuild)", backend)
        return

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

    # Build
    build_dir = repo_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

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
    family_params: dict | None = None,
    config: Config | None = None,
    config_item: dict | None = None,
    foreground: bool = False,
) -> LaunchResult:
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

    preferred_ctx = opts.get("preferred_ctx", family_params.get("preferred_ctx"))

    # Calculate KV cache
    if ctx_size is None:
        ctx_size, qk_type, qkv_type = calc_kv_cache(model.size_gb, model.family, config)
        if preferred_ctx:
            ctx_size = min(ctx_size, int(preferred_ctx))
    else:
        qk_type = ctk if ctk else "q8_0"
        qkv_type = ctv if ctv else "q8_0"

    # Override if explicitly set
    if ctk:
        qk_type = ctk
    if ctv:
        qkv_type = ctv

    # Find port
    if port is None:
        port = find_free_port(config)
    elif is_port_used(port):
        logger.info("Port %d is already in use, stopping existing model first...", port)
        stop_model(port, config)
        if not _wait_for_port_free(port):
            raise RuntimeError(
                f"Port {port} is still in use after stopping existing model"
            )

    fit_ctx = opts.get("fit_ctx", ctx_size or 4096)
    fit_target = opts.get("fit_target")
    if fit_target is None:
        fit_target = int(
            config.get("fit_target_mib", config.get("vram_headroom_gb", 2) * 1024),
        )

    cpu_moe = opts.get("cpu_moe", family_params.get("cpu_moe", False))
    n_cpu_moe = opts.get("n_cpu_moe", family_params.get("n_cpu_moe"))
    alias = opts.get("alias", family_params.get("alias", model.name))
    batch_size = int(
        opts.get(
            "batch_size",
            family_params.get("batch_size", config.get("batch_size", 2048)),
        ),
    )
    ubatch_size = int(
        opts.get(
            "ubatch_size",
            family_params.get("ubatch_size", config.get("ubatch_size", 2048)),
        ),
    )
    threads = int(
        opts.get(
            "threads",
            family_params.get("threads", config.get("threads", config["build_jobs"])),
        ),
    )
    threads_batch = int(
        opts.get(
            "threads_batch",
            family_params.get(
                "threads_batch",
                config.get("threads_batch", max(threads, 16)),
            ),
        ),
    )
    n_predict = int(
        opts.get(
            "n_predict",
            family_params.get("n_predict", 4096 if family == "qwen" else 2048),
        ),
    )
    temp = str(opts.get("temp", family_params.get("temp", config.get("temp", 0.6))))
    top_k = str(opts.get("top_k", family_params.get("top_k", config.get("top_k", 20))))
    top_p = str(
        opts.get("top_p", family_params.get("top_p", config.get("top_p", 0.95))),
    )
    min_p = str(opts.get("min_p", family_params.get("min_p", config.get("min_p", 0.0))))
    repeat_penalty = str(
        opts.get(
            "repeat_penalty",
            family_params.get("repeat_penalty", config.get("repeat_penalty", 1.0)),
        ),
    )
    presence_penalty = str(
        opts.get(
            "presence_penalty",
            family_params.get("presence_penalty", config.get("presence_penalty", 0.0)),
        ),
    )
    cache_ram = opts.get(
        "cache_ram",
        family_params.get("cache_ram", config.get("cache_ram")),
    )
    seed = opts.get("seed", family_params.get("seed", config.get("seed", seed)))
    if seed is not None:
        seed = int(seed)
    chat_template_kwargs = opts.get(
        "chat_template_kwargs",
        family_params.get("chat_template_kwargs", config.get("chat_template_kwargs")),
    )

    # Speculative decoding parameters
    spec_ngram_size_n = opts.get(
        "spec_ngram_size_n",
        family_params.get("spec_ngram_size_n", config.get("spec_ngram_size_n")),
    )
    draft_min = opts.get(
        "draft_min", family_params.get("draft_min", config.get("draft_min"))
    )
    draft_max = opts.get(
        "draft_max", family_params.get("draft_max", config.get("draft_max"))
    )
    spec_type = opts.get(
        "spec_type", family_params.get("spec_type", config.get("spec_type"))
    )

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
        "--jinja",
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
    if cache_ram is not None:
        args.extend(["--cache-ram", str(cache_ram)])
    if seed is not None:
        args.extend(["-s", str(seed)])
    if not config.get("context_shift", False):
        args.append("--no-context-shift")

    # Fit mode
    fit_enabled = opts.get("fit", True)
    args.extend(
        [
            "--fit",
            "on" if fit_enabled else "off",
            "--fit-ctx",
            str(fit_ctx),
            "--fit-target",
            str(fit_target),
        ]
    )

    # Memory options
    if cpu_moe:
        args.append("--cpu-moe")
    elif n_cpu_moe is not None:
        args.extend(["--n-cpu-moe", str(n_cpu_moe)])

    # Speculative decoding (only if configured)
    if spec_type is not None:
        if spec_ngram_size_n is None:
            spec_ngram_size_n = 16
        if draft_min is None:
            draft_min = 8
        if draft_max is None:
            draft_max = 24

        args.extend(
            [
                "--spec-type",
                str(spec_type),
                "--spec-ngram-size-n",
                str(spec_ngram_size_n),
                "--draft-min",
                str(draft_min),
                "--draft-max",
                str(draft_max),
            ]
        )

    # Family-specific args
    if family == "qwen":
        reasoning_budget = opts.get(
            "reasoning_budget", config.get("reasoning_budget", 512)
        )
        reasoning_msg = opts.get(
            "reasoning_budget_message",
            config.get(
                "reasoning_budget_message",
                "I have thought enough, let's move to answering.",
            ),
        )
        args.extend(
            [
                "-n",
                str(n_predict),
                "--reasoning",
                "on",
                "--reasoning-budget",
                str(reasoning_budget),
                "--reasoning-budget-message",
                reasoning_msg,
            ],
        )
        if chat_template_kwargs:
            args.extend(["--chat-template-kwargs", str(chat_template_kwargs)])
    elif family == "gemma":
        args.extend(["-n", str(n_predict), "--no-mmproj"])
    elif family == "bonsai":
        args.extend(["-n", str(n_predict), "--reasoning", "on"])

    # Start process
    logger.info("Starting %s on port %d...", model.name, port)

    if foreground:
        subprocess.run(args)
        return None

    proc = subprocess.Popen(
        args,
        start_new_session=True,
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
            raise RuntimeError(f"llama-server crashed with exit code {exit_code}")

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
        except Exception:
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
            # Remove PID file
            pid_dir = get_pid_dir()
            for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
                pid_file.unlink()
            return stopped
        except OSError:
            pass

    return False
