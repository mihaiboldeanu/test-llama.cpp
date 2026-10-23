import os
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
from .domain.planner import PlanRequest, plan
from .domain.resolver import ProfileResolver, ResolvedProfile
from .domain.runner import RunnerInput, pid_exists, terminate_process
from .domain.runner import run as run_server
from .families import get_registry
from .log import setup_logging

logger = setup_logging()


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
    profile: ResolvedProfile | None = None
    argv: list[str] | None = None  # Set during dry_run


@dataclass
class ModelStartConfig:
    """Configuration for starting a model server."""

    # Model & backend
    model: ModelInfo
    backend: str = "llama.cpp"

    # Server settings
    port: int | None = None
    parallel: int | None = None
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
    mtp_n_max: int | None = None
    mtp_n_min: int | None = None

    # Config overrides
    family_params: dict | None = None
    config: Config | None = None
    config_item: dict | None = None

    # Dry run — return argv without launching
    dry_run: bool = False





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


def _ensure_available_port(port: int | None, config: Config) -> int:
    """Resolve a launch port, stopping a stale Rama server if needed."""
    if port is None:
        return find_free_port(config)

    if is_port_used(port):
        logger.info("Port %d is already in use, stopping existing model first...", port)
        stop_model(port, config)
        if not _wait_for_port_free(port):
            # Only kill Rama-owned PIDs, not arbitrary processes on this port
            owned_pid = get_model_pid(config, port)
            if owned_pid and pid_exists(owned_pid):
                logger.info(
                    "Terminating stuck Rama process PID %d on port %d", owned_pid, port
                )
                terminate_process(owned_pid)
                time.sleep(2)
                _wait_for_port_free(port)
                time.sleep(1)

        if is_port_used(port):
            logger.warning("Port %d still busy, picking another free port...", port)
            port = find_free_port(config, port + 1)

    return port


def _cli_flag_overrides(cfg: "ModelStartConfig") -> dict:
    """Translate direct launch fields into literal-flag CLI overrides."""
    overrides: dict = {}
    if cfg.parallel is not None:
        overrides["-np"] = cfg.parallel
    if cfg.ctx_size is not None:
        overrides["--ctx-size"] = cfg.ctx_size
    if cfg.ctk is not None:
        overrides["-ctk"] = cfg.ctk
    if cfg.ctv is not None:
        overrides["-ctv"] = cfg.ctv
    if cfg.reasoning is not None:
        overrides["--reasoning"] = cfg.reasoning
    if cfg.reasoning_budget is not None:
        overrides["--reasoning-budget"] = cfg.reasoning_budget
    if cfg.seed is not None:
        overrides["-s"] = cfg.seed
    if cfg.draft_model is not None:
        overrides["--model-draft"] = cfg.draft_model
    if cfg.draft_model is not None:
        # Explicit draft bounds only apply to an explicitly paired draft.
        overrides["--spec-draft-n-max"] = cfg.draft_max
        overrides["--spec-draft-n-min"] = cfg.draft_min
    if cfg.spec_ngram:
        overrides["--spec-ngram"] = True
    if cfg.mtp:
        overrides["mtp"] = True
    if cfg.mtp_n_max is not None:
        overrides["mtp_n_max"] = cfg.mtp_n_max
    if cfg.mtp_n_min is not None:
        overrides["mtp_n_min"] = cfg.mtp_n_min
    return overrides


def _resolve_profile(cfg: "ModelStartConfig", config: Config) -> ResolvedProfile:
    family_def = get_registry().get(cfg.model.family)
    family_override = cfg.family_params or config.families.get(cfg.model.family, {})
    model_override = (config.get("models", {}) or {}).get(cfg.model.name, {})
    resolver = ProfileResolver(family_definition=family_def)
    return resolver.resolve(
        cfg.model,
        global_config=config._data if hasattr(config, "_data") else None,
        family_override=family_override or None,
        model_override=model_override or None,
        batch_override=cfg.config_item or None,
        cli_override=_cli_flag_overrides(cfg) or None,
    )


def _with_resolved_draft_paths(
    config: Config, profile: ResolvedProfile, *, mtp_enabled: bool = False
) -> tuple[ResolvedProfile, str | None]:
    """Resolve draft references against the configured model directory."""
    from dataclasses import replace

    args = dict(profile.args)
    provenance = dict(profile.provenance)
    draft_reference = args.get("--model-draft")
    if draft_reference is not None:
        resolved = _resolve_draft_model_path(config, str(draft_reference))
        if resolved != str(draft_reference):
            args["--model-draft"] = resolved
            provenance["args.--model-draft"] = "launch draft resolution"

    mtp_draft = profile.mtp.get("draft_model") if isinstance(profile.mtp, dict) else None
    resolved_mtp_draft = None
    if mtp_draft is not None and mtp_enabled:
        # External MTP heads are only needed for MTP launches; do not fail
        # plain launches when the sidecar file is absent.
        resolved_mtp_draft = _resolve_draft_model_path(config, str(mtp_draft))
    return replace(profile, args=args, provenance=provenance), resolved_mtp_draft


def _prepare_launch_context(
    cfg: "ModelStartConfig",
) -> tuple[Config, Path, ModelInfo, ResolvedProfile]:
    config = cfg.config or Config()
    server_bin = get_server_bin(cfg.backend, config)

    if not server_bin.exists():
        logger.warning("Server not found: %s, building...", server_bin)
        build_backend(cfg.backend, config)

    model = cfg.model
    if not Path(model.path).exists():
        raise FileNotFoundError(f"Model file not found: {model.path}")

    profile = _resolve_profile(cfg, config)
    mtp_enabled = cfg.mtp or bool(profile.mtp.get("enabled", False))
    profile, _ = _with_resolved_draft_paths(
        config, profile, mtp_enabled=mtp_enabled
    )
    return config, server_bin, model, profile


def _build_launch_args(
    cfg: "ModelStartConfig",
    config: Config,
    server_bin: Path,
    model: ModelInfo,
    profile: ResolvedProfile,
) -> tuple[list[str], int, int, int | None, str]:
    port = _ensure_available_port(cfg.port, config)
    host = str(profile.args.get("--host", config.get("host", "127.0.0.1")))
    mtp_enabled = cfg.mtp or bool(profile.mtp.get("enabled", False))
    _, resolved_mtp_draft = _with_resolved_draft_paths(
        config, profile, mtp_enabled=mtp_enabled
    )
    rules = config.get("rules", {}) if isinstance(config.get("rules", {}), dict) else {}

    request = PlanRequest(
        server_bin=server_bin,
        model=model,
        profile=profile,
        host=host,
        port=port,
        backend=cfg.backend,
        mtp=cfg.mtp,
        mtp_n_max=cfg.mtp_n_max,
        mtp_n_min=cfg.mtp_n_min,
        mtp_draft_model=resolved_mtp_draft,
        template_dir=config.get("template_dir"),
        small_threshold_gb=float(rules.get("small_threshold_gb", 10)),
    )
    args = plan(request)

    # Final port check — TOCTOU guard right before launch
    if is_port_used(port):
        logger.warning("Port %s got taken, finding replacement...", port)
        port = find_free_port(config)
        for i, a in enumerate(args):
            if a == "--port":
                args[i + 1] = str(port)
                break

    seed_value = profile.args.get("-s")
    seed = None if seed_value is None else int(seed_value)
    return args, port, profile.ctx, seed, host


def start_model(cfg: ModelStartConfig) -> LaunchResult | None:
    """Start a llama.cpp model server."""
    config, server_bin, model, profile = _prepare_launch_context(cfg)
    args, port, ctx_size, seed_val, host = _build_launch_args(
        cfg,
        config,
        server_bin,
        model,
        profile,
    )

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

    launch_env = os.environ.copy()
    if profile.env:
        launch_env.update({str(k): str(v) for k, v in profile.env.items()})

    if cfg.foreground:
        completed = subprocess.run(args, env=launch_env)
        if completed.returncode:
            raise RuntimeError(
                f"llama-server exited during startup with code {completed.returncode}"
            )
        return None

    runner_inp = RunnerInput(
        args=tuple(args),
        model_name=model.name,
        port=port,
        host=host,
        pid_dir=get_pid_dir(),
        env=launch_env,
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
            stopped = terminate_process(pid)
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
