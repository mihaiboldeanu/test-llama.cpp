import os
import socket
from dataclasses import dataclass
from pathlib import Path

from . import Config


@dataclass
class ModelInfo:
    """Discovered GGUF model."""

    name: str
    path: Path
    size_gb: float
    family: str
    quant: str
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class RunningModel:
    """A model currently running."""

    name: str
    port: int
    pid: int
    backend: str


def get_pid_dir() -> Path:
    """Get the PID file directory."""
    pid_dir = Path.home() / ".local" / "state" / "rama" / "pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir


def discover_models(config: Config) -> list[ModelInfo]:
    """Discover GGUF files in model_dir."""
    model_dir = config.model_dir
    if not model_dir.exists():
        return []

    models = []
    for gguf in model_dir.rglob("*.gguf"):
        if not gguf.is_file():
            continue

        name = gguf.stem
        size_gb = gguf.stat().st_size / (1024**3)
        quant = detect_quant(name)
        family = detect_family(gguf)
        tags = detect_tags(name)

        models.append(
            ModelInfo(
                name=name,
                path=gguf,
                size_gb=round(size_gb, 1),
                family=family,
                quant=quant,
                tags=tags,
            ),
        )

    return sorted(models, key=lambda m: m.path)


def detect_quant(filename: str) -> str:
    """Detect quantization from filename."""
    filename = filename.lower()
    if "iq4" in filename:
        return "IQ4"
    if "q8" in filename:
        return "Q8"
    if "q6" in filename:
        return "Q6"
    if "q5" in filename:
        return "Q5"
    if "q4" in filename:
        return "Q4"
    if "iq3" in filename:
        return "IQ3"
    return "unknown"


def detect_family(path: Path) -> str:
    """Detect model family from path."""
    name = path.name.lower()
    parent = path.parent.name.lower()

    if "gemma" in parent or "gemma" in name:
        # Gemma 31B gets special treatment for 256K context
        if "31b" in name or "31b" in parent:
            return "gemma31b"
        return "gemma"
    if "bonsai" in parent or "bonsai" in name:
        return "bonsai"
    if "qwen" in parent or "qwen" in name:
        return "qwen"
    return "unknown"


def detect_tags(filename: str) -> list[str]:
    """Detect special tags from filename."""
    tags = []
    lower = filename.lower()

    tags += ["uncensored"] if "uncensored" in lower else []
    tags += ["crow"] if "crow" in lower else []
    tags += ["fernflower"] if "fernflower" in lower else []
    tags += ["qwopus"] if "qwopus" in lower else []
    tags += ["heretic"] if "heretic" in lower else []

    return tags


def calc_kv_cache(
    model_size_gb: float,
    family: str,
    config: Config,
) -> tuple[int, str, str]:
    """Calculate max KV cache that fits in VRAM.
    Default: q8_0 for both k and v

    Returns: (ctx_size, qk_type, qkv_type)
    """
    vram_available = config.vram_available_gb
    model_vram = model_size_gb
    kv_vram = vram_available - model_vram

    if kv_vram < 0.5:
        raise RuntimeError(
            f"Not enough VRAM for KV cache: {kv_vram:.1f}GB available. "
            f"Model needs {model_vram:.1f}GB, only {vram_available:.1f}GB total."
        )

    # Very tight VRAM - minimal context
    if kv_vram < 1:
        kv_per_token_kb = {
            "qwen": _get_qwen_kv_bytes_per_token(model_size_gb),
            "gemma": _get_gemma_kv_bytes_per_token(model_size_gb),
            "bonsai": 8000,
            "gemma31b": _get_gemma_kv_bytes_per_token(model_size_gb),
        }.get(family, 15000)
        tight_ctx = int((kv_vram * 1024**3) / kv_per_token_kb)
        return (max(2048, min(tight_ctx, 8192)), "q8_0", "q8_0")

    # Get max_ctx from family config
    family_config = config.families.get(family, {})
    max_ctx_limit = family_config.get("max_ctx", 262144)

    # Default: q8_0 for both k and v
    kv_per_token_bytes = {
        "qwen": _get_qwen_kv_bytes_per_token(model_size_gb),
        "gemma": _get_gemma_kv_bytes_per_token(model_size_gb),
        "bonsai": 8000,
        "gemma31b": _get_gemma_kv_bytes_per_token(model_size_gb),
    }.get(family, 15000)

    # Calculate max ctx with q8_0
    max_ctx = int((kv_vram * 1024**3) / kv_per_token_bytes)
    max_ctx = min(max_ctx, max_ctx_limit)
    max_ctx = min(max_ctx, config.get("default_ctx", 131072))

    if max_ctx >= 2048:
        return (max_ctx, "q8_0", "q8_0")

    # If not enough VRAM for q8_0, try q4_k_m
    max_ctx_q4 = int((kv_vram * 1024**3) / (kv_per_token_bytes / 3.7))
    max_ctx_q4 = min(max_ctx_q4, max_ctx_limit)
    if max_ctx_q4 >= 2048:
        return (max_ctx_q4, "q4_k_m", "q4_k_m")

    # Fallback
    return (min(1024, max_ctx_limit), "q8_0", "q8_0")


def _get_qwen_kv_bytes_per_token(size_gb: float) -> int:
    if size_gb < 10:
        return 11000  # 9B
    if size_gb < 20:
        return 22000  # 27B
    if size_gb < 35:
        return 45000  # 35B
    return 60000


def _get_gemma_kv_bytes_per_token(size_gb: float) -> int:
    if size_gb < 10:
        return 5000  # 4B
    if size_gb < 25:
        return 25000  # 31B
    return 40000


def find_free_port(config: Config) -> int:
    """Find a free port in range."""
    start, end = config.port_range
    for port in range(start, end + 1):
        if not is_port_used(port):
            return port
    raise RuntimeError(f"No free ports in range {start}-{end}")


def is_port_used(port: int) -> bool:
    """Check if port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def get_running_models(config: Config) -> list[RunningModel]:
    """Get currently running models from PID dir."""
    pid_dir = get_pid_dir()
    if not pid_dir.exists():
        return []

    running = []
    for pid_file in pid_dir.glob("model_*.pid"):
        try:
            content = pid_file.read_text().strip()
            if not content:
                pid_file.unlink()
                continue

            # Try JSON format first (new format with backend)
            import json

            try:
                data = json.loads(content)
                pid = data["pid"]
                backend = data.get("backend", "llama.cpp")
                name = data.get(
                    "name", pid_file.stem.replace("model_", "").rsplit("_", 1)[0]
                )
                port = data.get("port", int(pid_file.stem.rsplit("_", 1)[-1]))
            except (json.JSONDecodeError, KeyError):
                # Fall back to plain PID (old format)
                pid = int(content)
                backend = "llama.cpp"
                name = pid_file.stem.replace("model_", "").rsplit("_", 1)[0]
                port = int(pid_file.stem.rsplit("_", 1)[-1])

            os.kill(pid, 0)  # Check if process exists
            running.append(RunningModel(name, port, pid, backend))
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)  # Stale PID file

    return running


def get_model_pid(config: Config, port: int) -> int | None:
    """Get PID for a model on given port."""
    pid_dir = get_pid_dir()
    for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
        return int(pid_file.read_text().strip())
    return None
