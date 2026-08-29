import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import Config
from .core import (
    ModelInfo,
    find_free_port,
    get_model_pid,
    get_pid_dir,
    is_port_used,
)
from .domain.planner import PlannerInput, plan
from .domain.resolver import ProfileResolver, ResolvedModelProfile
from .domain.runner import RunnerInput
from .domain.runner import run as run_server
from .families import get_registry
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
    if default is not None:
        logger.debug("Using default for '%s': %s", key, default)
    return default


def _resolve_draft_model_path(config: Config, reference: str) -> str:
    """Resolve a configured draft-model name to one unambiguous GGUF path."""
    supplied_path = Path(reference).expanduser()
    if supplied_path.is_file():
        return str(supplied_path)

    model_dir = config.model_dir
    direct_path = model_dir / reference
    if direct_path.is_file():
        return str(direct_path)
    if direct_path.suffix.lower() != ".gguf":
        direct_gguf = direct_path.with_suffix(".gguf")
        if direct_gguf.is_file():
            return str(direct_gguf)

    normalized = reference.lower()
    matches = sorted(
        path
        for path in model_dir.rglob("*.gguf")
        if path.is_file() and normalized in path.stem.lower()
    )
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(
            f"Draft model {reference!r} was not found under {model_dir}"
        )
    options = ", ".join(str(path) for path in matches)
    raise ValueError(
        f"Draft model {reference!r} matches multiple GGUF files: {options}"
    )


@dataclass
class LaunchResult:
    """Result of starting a model."""

    name: str
    port: int
    pid: int
    backend: str
    ctx_size: int
    seed: int | None = None
    profile: ResolvedModelProfile | None = None
    argv: list[str] | None = None  # Set during dry_run


@dataclass
class ModelStartConfig:
    """Configuration for starting a model server."""

    # Model & backend
    model: ModelInfo
    backend: str = "llama.cpp"

    # Server settings
    port: int | None = None
    ctx_size: int | None = None
    ctk: str | None = None
    ctv: str | None = None
    seed: int | None = None
    foreground: bool = False
    log_disable: bool = False

    # Reasoning
    reasoning: bool | None = None
    reasoning_budget: int | None = None

    # Speculative decoding
    draft_model: str | None = None
    draft_max: int = 16
    draft_min: int = 8
    spec_ngram: bool = False

    # MTP
    mtp: bool = False
    mtp_n_max: int = 3
    mtp_n_min: int = 0

    # Config overrides
    family_params: dict | None = None
    config: Config | None = None
    config_item: dict | None = None

    # Dry run — return argv without launching
    dry_run: bool = False


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
            f"-DCMAKE_CUDA_ARCHITECTURES={config.get('cuda_arch', '89')}",
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


def start_model(cfg: ModelStartConfig) -> LaunchResult | None:
    """Start a llama.cpp model server."""
    config = cfg.config or Config()
    explicit_overrides = cfg.config_item or {}
    server_bin = get_server_bin(cfg.backend, config)

    if not server_bin.exists():
        logger.warning("Server not found: %s, building...", server_bin)
        build_backend(cfg.backend, config)

    model = cfg.model
    if not Path(model.path).exists():
        raise FileNotFoundError(f"Model file not found: {model.path}")
    family = model.family
    family_def = get_registry().get(family)
    family_params = cfg.family_params or config.families.get(family, {})

    model_overrides = (config.get("models", {}) or {}).get(model.name, {})
    # Explicit batch/CLI options win over model-specific overrides, which win
    # over family defaults.  All launch settings use this same precedence.
    opts = {**family_params, **model_overrides, **explicit_overrides}

    # Resolve typed profile with provenance tracking
    domain_model = model.to_domain()
    resolver = ProfileResolver(family_definition=family_def)
    profile = resolver.resolve(
        domain_model,
        global_config=config._data if hasattr(config, "_data") else None,
        family_override=family_params or None,
        model_override=model_overrides or None,
        batch_override=explicit_overrides or None,
    )

    chat_template = opts.get("chat_template", config.get("chat_template"))
    chat_template_file = opts.get(
        "chat_template_file", config.get("chat_template_file")
    )

    template_dir = config.get("template_dir")
    if template_dir and chat_template_file:
        chat_template_file = str(Path(template_dir) / chat_template_file)

    preferred_ctx = opts.get("preferred_ctx")
    if preferred_ctx is None:
        preferred_ctx = (
            profile.context.small_ctx
            if model.size_gb < 10
            else profile.context.large_ctx
        )

    ctx_size = cfg.ctx_size
    if ctx_size is None:
        ctx_size = preferred_ctx
    qk_type = cfg.ctk or profile.launch.ctk or "q8_0"
    qkv_type = cfg.ctv or profile.launch.ctv or "q8_0"

    # The planner applies the MTP ctx ratio exactly once (see planner.py);
    # only resolve it here so family/model overrides reach the planner.
    mtp_ctx_ratio = resolve(opts, family_params, "mtp_ctx_ratio", 0.8, config)

    port = cfg.port
    if port is None:
        port = find_free_port(config)
    elif is_port_used(port):
        logger.info("Port %d is already in use, stopping existing model first...", port)
        stop_model(port, config)
        if not _wait_for_port_free(port):
            # Only kill Rama-owned PIDs, not arbitrary processes on this port
            owned_pid = get_model_pid(config, port)
            if owned_pid and _pid_exists(owned_pid):
                logger.info("Terminating stuck Rama process PID %d on port %d", owned_pid, port)
                _terminate_pid(owned_pid)
                time.sleep(2)
                _wait_for_port_free(port)
                time.sleep(1)

        if is_port_used(port):
            logger.warning("Port %d still busy, picking another free port...", port)
            port = find_free_port(config, port + 1)

    # Resolve launch settings from config chain
    fit_target = profile.memory.fit_target_mib
    n_gpu_layers = profile.memory.n_gpu_layers
    use_mmap = profile.memory.use_mmap
    skip_chat_parsing = profile.capabilities.skip_chat_parsing or opts.get(
        "skip_chat_parsing", config.get("skip_chat_parsing", False)
    )
    seed_val = profile.launch.seed
    if seed_val is not None:
        seed_val = int(seed_val)

    cpu_moe = resolve(opts, family_params, "cpu_moe", False)
    n_cpu_moe = resolve(opts, family_params, "n_cpu_moe", None)
    alias = resolve(opts, family_params, "alias", model.name)
    batch_size = int(resolve(opts, family_params, "batch_size", 2048, config))
    ubatch_size = int(resolve(opts, family_params, "ubatch_size", 2048, config))
    threads = int(resolve(opts, family_params, "threads", config["build_jobs"], config))
    threads_batch = int(
        resolve(opts, family_params, "threads_batch", max(threads, 16), config)
    )
    n_predict = int(resolve(opts, family_params, "n_predict", 2048, config))
    samplers_val = resolve(opts, family_params, "samplers", None)
    no_mmproj = resolve(opts, family_params, "no_mmproj", False)
    cache_ram = resolve(opts, family_params, "cache_ram", None, config)
    chat_template_kwargs = resolve(
        opts, family_params, "chat_template_kwargs", None, config
    )
    spec_ngram_mod_size_n = resolve(
        opts, family_params, "spec_ngram_mod_size_n", None, config
    )
    spec_ngram_mod_n_min = resolve(
        opts, family_params, "spec_ngram_mod_n_min", None, config
    )
    spec_ngram_mod_n_max = resolve(
        opts, family_params, "spec_ngram_mod_n_max", None, config
    )
    spec_type = resolve(opts, family_params, "spec_type", None, config)

    # Reasoning settings
    is_small = model.size_gb < opts.get("small_threshold_gb", 10)
    reasoning_budget = cfg.reasoning_budget or opts.get(
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
    enable_reasoning = (
        cfg.reasoning
        if cfg.reasoning is not None
        else opts.get("default_reasoning", not is_small)
    )
    reasoning_template_key = resolve(
        opts, family_params, "reasoning_template_key", "enable_thinking", config
    )
    reasoning_template_value = resolve(
        opts,
        family_params,
        "reasoning_enabled_value" if enable_reasoning else "reasoning_disabled_value",
        enable_reasoning,
        config,
    )

    draft_reference = cfg.draft_model or resolve(
        opts, family_params, "draft_model", None, config
    )
    draft_model = (
        _resolve_draft_model_path(config, draft_reference)
        if draft_reference is not None
        else None
    )
    draft_max = (
        cfg.draft_max
        if cfg.draft_model is not None
        else int(resolve(opts, family_params, "draft_max", 16, config))
    )
    draft_min = (
        cfg.draft_min
        if cfg.draft_model is not None
        else int(resolve(opts, family_params, "draft_min", 0, config))
    )

    # Build pure planner input — argv construction is a pure function
    planner_inp = PlannerInput(
        model=domain_model,
        backend=cfg.backend,
        server_bin=server_bin,
        port=port,
        host=str(config["host"]),
        ctx_size=ctx_size,
        preferred_ctx=preferred_ctx,
        ctk=qk_type,
        ctv=qkv_type,
        sampler=profile.sampler,
        fit_target_mib=fit_target,
        n_gpu_layers=n_gpu_layers,
        use_mmap=use_mmap,
        capabilities=profile.capabilities,
        seed=seed_val,
        skip_chat_parsing=skip_chat_parsing,
        log_disable=cfg.log_disable,
        context_shift=config.get("context_shift", False),
        chat_template=chat_template,
        chat_template_file=chat_template_file,
        reasoning=enable_reasoning,
        reasoning_budget=reasoning_budget,
        reasoning_msg=reasoning_msg,
        reasoning_template_key=reasoning_template_key,
        reasoning_template_value=reasoning_template_value,
        draft_model=draft_model,
        draft_max=draft_max,
        draft_min=draft_min,
        spec_ngram=cfg.spec_ngram,
        spec_type=spec_type,
        spec_ngram_mod_size_n=spec_ngram_mod_size_n,
        spec_ngram_mod_n_min=spec_ngram_mod_n_min,
        spec_ngram_mod_n_max=spec_ngram_mod_n_max,
        mtp=cfg.mtp,
        mtp_n_max=cfg.mtp_n_max,
        mtp_n_min=cfg.mtp_n_min,
        mtp_ctx_ratio=mtp_ctx_ratio,
        mtp_ngl_threshold_gb=opts.get("mtp_ngl_threshold_gb", 20),
        cpu_moe=cpu_moe,
        n_cpu_moe=n_cpu_moe,
        alias=alias,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        threads=threads,
        threads_batch=threads_batch,
        n_predict=n_predict,
        samplers=samplers_val,
        no_mmproj=no_mmproj,
        cache_ram=cache_ram,
        chat_template_kwargs=chat_template_kwargs,
        fit_enabled=opts.get("fit", True),
        fit_ctx_enabled=opts.get("fit_ctx", True),
        small_threshold_gb=opts.get("small_threshold_gb", 10),
        draft_min_val=opts.get(
            "draft_min", family_params.get("draft_min", config.get("spec_ngram_mod_n_min"))
        ),
        draft_max_val=opts.get(
            "draft_max", family_params.get("draft_max", config.get("spec_ngram_mod_n_max"))
        ),
    )

    args = plan(planner_inp)

    # Final port check — TOCTOU guard right before launch
    if is_port_used(port):
        logger.warning("Port %s got taken, finding replacement...", port)
        port = find_free_port(config)
        for i, a in enumerate(args):
            if a == "--port":
                args[i + 1] = str(port)
                break

    if cfg.dry_run:
        return LaunchResult(
            model.name,
            port,
            pid=-1,
            backend=cfg.backend,
            ctx_size=ctx_size,
            seed=seed_val,
            profile=profile,
            argv=args,
        )

    # Start process
    if not cfg.log_disable:
        logger.info("Starting %s on port %d...", model.name, port)

    if cfg.foreground:
        subprocess.run(args)
        return None

    runner_inp = RunnerInput(
        args=tuple(args),
        model_name=model.name,
        port=port,
        host=str(config["host"]),
        pid_dir=get_pid_dir(),
    )
    result = run_server(runner_inp, log_disable=cfg.log_disable)
    if not cfg.log_disable:
        logger.info("Model loaded! (PID: %d)", result.pid)

    return LaunchResult(
        model.name,
        port,
        result.pid,
        cfg.backend,
        ctx_size,
        seed_val,
        profile,
    )


def stop_model(port: int, config: Config = None) -> bool:
    """Stop model on port. Only terminates Rama-owned PIDs."""
    config = config or Config()
    pid = get_model_pid(config, port)

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
