import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import Config
from .core import ModelInfo, calc_kv_cache


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
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)

    return not _pid_exists(pid)


def get_server_bin(backend: str, config: Config) -> Path:
    """Get path to llama-server binary."""
    if backend == "turboquant":
        return config.turbo_dir / "build" / "bin" / "llama-server"
    return config.llama_cpp_dir / "build" / "bin" / "llama-server"


def build_backend(
    backend: str, config: Config, force: bool = False, cpu_only: bool = False
) -> None:
    """Build llama.cpp or turboquant."""
    repo_dir = config.turbo_dir if backend == "turboquant" else config.llama_cpp_dir
    nvcc = config.cuda_root / "bin" / "nvcc"

    if not cpu_only and not nvcc.exists():
        raise RuntimeError(f"nvcc not found: {nvcc}")

    server_bin = get_server_bin(backend, config)
    if server_bin.exists() and not force:
        print(f"{backend} already built, skipping (use --force to rebuild)")
        return

    print(f"Building {backend}{' (CPU only)' if cpu_only else ''}...")

    # Git pull
    subprocess.run(
        ["git", "fetch", "--all", "--prune"], cwd=repo_dir, capture_output=True
    )
    subprocess.run(["git", "pull", "--ff-only"], cwd=repo_dir, capture_output=True)

    # Build
    build_dir = repo_dir / "build"
    if build_dir.exists():
        subprocess.run(["rm", "-rf", str(build_dir)])

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

    subprocess.run(["cmake"] + cmake_args, cwd=build_dir, env=env)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", str(config["build_jobs"])],
        cwd=build_dir,
    )

    print(f"Built {backend}")


def start_model(
    model: ModelInfo,
    backend: str = "llama.cpp",
    port: int = None,
    ctx_size: int = None,
    ctk: str = None,
    ctv: str = None,
    seed: int | None = None,
    family_params: dict = None,
    config: Config = None,
    config_item: dict = None,
) -> LaunchResult:
    """Start a llama.cpp model server."""
    config = config or Config()
    opts = config_item or {}
    server_bin = get_server_bin(backend, config)

    if not server_bin.exists():
        print(f"Server not found: {server_bin}, building...")
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
        from .core import find_free_port

        port = find_free_port(config)

    fit_enabled = opts.get("fit", True)
    fit_ctx = opts.get("fit_ctx", ctx_size or 4096)
    fit_target = opts.get("fit_target")
    if fit_target is None:
        fit_target = int(
            config.get("fit_target_mib", config.get("vram_headroom_gb", 2) * 1024)
        )

    cpu_moe = opts.get("cpu_moe", family_params.get("cpu_moe", False))
    n_cpu_moe = opts.get("n_cpu_moe", family_params.get("n_cpu_moe"))
    n_gpu_layers = opts.get("n_gpu_layers", family_params.get("n_gpu_layers"))
    alias = opts.get("alias", family_params.get("alias", model.name))
    batch_size = int(
        opts.get(
            "batch_size",
            family_params.get("batch_size", config.get("batch_size", 2048)),
        )
    )
    ubatch_size = int(
        opts.get(
            "ubatch_size",
            family_params.get("ubatch_size", config.get("ubatch_size", 2048)),
        )
    )
    threads = int(
        opts.get(
            "threads",
            family_params.get("threads", config.get("threads", config["build_jobs"])),
        )
    )
    threads_batch = int(
        opts.get(
            "threads_batch",
            family_params.get(
                "threads_batch", config.get("threads_batch", max(threads, 16))
            ),
        )
    )
    n_predict = int(
        opts.get(
            "n_predict",
            family_params.get("n_predict", 4096 if family == "qwen" else 2048),
        )
    )
    temp = str(opts.get("temp", family_params.get("temp", config.get("temp", 0.6))))
    top_k = str(opts.get("top_k", family_params.get("top_k", config.get("top_k", 20))))
    top_p = str(
        opts.get("top_p", family_params.get("top_p", config.get("top_p", 0.95)))
    )
    min_p = str(opts.get("min_p", family_params.get("min_p", config.get("min_p", 0.0))))
    repeat_penalty = str(
        opts.get(
            "repeat_penalty",
            family_params.get("repeat_penalty", config.get("repeat_penalty", 1.0)),
        )
    )
    presence_penalty = str(
        opts.get(
            "presence_penalty",
            family_params.get("presence_penalty", config.get("presence_penalty", 0.0)),
        )
    )
    cache_ram = opts.get(
        "cache_ram", family_params.get("cache_ram", config.get("cache_ram"))
    )
    log_verbosity = int(
        opts.get(
            "log_verbosity",
            family_params.get("log_verbosity", config.get("log_verbosity", 1)),
        )
    )
    seed = opts.get("seed", family_params.get("seed", config.get("seed", seed)))
    if seed is not None:
        seed = int(seed)
    use_jinja = opts.get("jinja", family_params.get("jinja", True))
    no_context_shift = opts.get(
        "no_context_shift", family_params.get("no_context_shift", True)
    )
    enable_perf = opts.get("perf", family_params.get("perf", True))
    no_warmup = opts.get("no_warmup", family_params.get("no_warmup", False))
    chat_template_kwargs = opts.get(
        "chat_template_kwargs",
        family_params.get("chat_template_kwargs", config.get("chat_template_kwargs")),
    )

    # Build args
    args = [
        str(server_bin),
        "-m",
        str(model.path),
        "--host",
        "127.0.0.1",
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
        "--fit",
        "on" if fit_enabled else "off",
        "--fit-ctx",
        str(fit_ctx),
        "--fit-target",
        str(fit_target),
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
        "--log-verbosity",
        str(log_verbosity),
    ]
    if cache_ram is not None:
        args.extend(["--cache-ram", str(cache_ram)])
    if seed is not None:
        args.extend(["-s", str(seed)])

    # Leave gpu-layers unset by default so llama.cpp can use its built-in auto/fit logic.
    if n_gpu_layers is not None:
        args.extend(["-ngl", str(n_gpu_layers)])

    # Memory options
    if opts.get("offload_kv", True) is False or opts.get("no_offload_kv", False):
        args.append("--no-offload-kv")
    if cpu_moe:
        args.append("--cpu-moe")
    elif n_cpu_moe is not None:
        args.extend(["--n-cpu-moe", str(n_cpu_moe)])

    if use_jinja:
        args.append("--jinja")
    if no_context_shift:
        args.append("--no-context-shift")
    if no_warmup:
        args.append("--no-warmup")
    if enable_perf:
        args.append("--perf")
    if chat_template_kwargs:
        args.extend(["--chat-template-kwargs", str(chat_template_kwargs)])

    # Family-specific args
    if family == "qwen":
        args.extend(
            [
                "-n",
                str(n_predict),
                "--reasoning",
                "on",
                "--reasoning-budget",
                "512",
                "--reasoning-budget-message",
                "I have thought enough, let's move to answering.",
            ]
        )
    elif family == "gemma":
        args.extend(
            ["-n", str(n_predict), "--no-mmproj", "-ctk", "q8_0", "-ctv", "q8_0"]
        )
        # Skip adding family params for gemma too
    elif family == "bonsai":
        args.extend(["-n", str(n_predict), "--reasoning", "on"])

    # Start process
    print(f"Starting {model.name} on port {port}...")
    proc = subprocess.Popen(args)

    # Save PID
    pid_dir = Path.cwd() / ".pids"
    pid_dir.mkdir(exist_ok=True)
    pid_file = pid_dir / f"model_{model.name}_{port}.pid"
    pid_file.write_text(str(proc.pid))

    # Wait for ready
    import urllib.request

    try:
        for i in range(60):
            time.sleep(2)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
                print(f"Model loaded! (PID: {proc.pid})")
                return LaunchResult(model.name, port, proc.pid, backend, ctx_size, seed)
            except Exception:
                continue
    except Exception:
        _terminate_pid(proc.pid)
        for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
            pid_file.unlink(missing_ok=True)
        raise

    _terminate_pid(proc.pid)
    for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
        pid_file.unlink(missing_ok=True)
    raise RuntimeError("Model failed to start within 2 minutes")


def stop_model(port: int, config: Config = None) -> bool:
    """Stop model on port."""
    from .core import get_model_pid

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
            pid_dir = Path.cwd() / ".pids"
            for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
                pid_file.unlink()
            return stopped
        except OSError:
            pass

    return False
