import os
import socket
from dataclasses import dataclass
from pathlib import Path

from . import Config


def _sweep_stale_pids() -> None:
    """Remove PID files for dead processes. Called once on import."""
    pid_dir = get_pid_dir()
    if not pid_dir.exists():
        return

    for pid_file in pid_dir.glob("model_*.pid"):
        try:
            import json

            content = pid_file.read_text().strip()
            if not content:
                pid_file.unlink()
                continue

            try:
                data = json.loads(content)
                pid = data["pid"]
            except (json.JSONDecodeError, KeyError):
                pid = int(content)

            os.kill(pid, 0)  # Process alive, keep file
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)  # Dead or corrupt, remove



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

    if "glm" in parent or "glm" in name:
        return "glm"
    if "devstral" in parent or "devstral" in name:
        return "devstral"
    if "mistral-small" in name or "mistral-small" in parent:
        return "mistral-small"
    if "gemma" in parent or "gemma" in name:
        return "gemma"
    if "bonsai" in parent or "bonsai" in name:
        return "bonsai"
    if "qwen" in parent or "qwen" in name:
        return "qwen"
    if "nemotron" in parent or "nemotron" in name:
        return "nemotron"
    if "phi-4" in name or "phi-4" in parent or "phi4" in name or "phi4" in parent:
        return "phi4"
    if "granite" in parent or "granite" in name:
        return "granite"
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

    kv_vram = max(
        kv_vram, 0.1
    )  # Allow launch even if model exceeds VRAM (llama.cpp will offload)

    # Very tight VRAM - minimal context
    if kv_vram < 1:
        kv_per_token_kb = {
            "qwen": _get_qwen_kv_bytes_per_token(model_size_gb),
            "gemma": _get_gemma_kv_bytes_per_token(model_size_gb),
            "bonsai": 8000,
            "nemotron": 6000,
            "phi4": 5000,
            "granite": _get_granite_kv_bytes_per_token(model_size_gb),
            "glm": 25000,
            "mistral-small": 25000,
            "devstral": 25000,
        }.get(family, 15000)
        tight_ctx = int((kv_vram * 1024**3) / kv_per_token_kb)
        return (max(2048, min(tight_ctx, 8192)), "q8_0", "q8_0")

    # Use a global safe maximum; individual families no longer override.
    max_ctx_limit = 262144

    # Default: q8_0 for both k and v
    kv_per_token_bytes = {
        "qwen": _get_qwen_kv_bytes_per_token(model_size_gb),
        "gemma": _get_gemma_kv_bytes_per_token(model_size_gb),
        "bonsai": 8000,
        "nemotron": _get_nemotron_kv_bytes_per_token(model_size_gb),
        "phi4": _get_phi4_kv_bytes_per_token(model_size_gb),
        "granite": _get_granite_kv_bytes_per_token(model_size_gb),
        "glm": _get_glm_kv_bytes_per_token(model_size_gb),
        "mistral-small": _get_mistral_small_kv_bytes_per_token(model_size_gb),
        "devstral": _get_devstral_kv_bytes_per_token(model_size_gb),
    }.get(family, 15000)

    # Calculate max ctx with q8_0
    max_ctx = int((kv_vram * 1024**3) / kv_per_token_bytes)
    max_ctx = min(max_ctx, max_ctx_limit)
    preferred = get_preferred_ctx(model_size_gb, family, config)
    max_ctx = min(max_ctx, preferred)

    if max_ctx >= 2048:
        return (max_ctx, "q8_0", "q8_0")

    # If not enough VRAM for q8_0, try q4_k_m
    max_ctx_q4 = int((kv_vram * 1024**3) / (kv_per_token_bytes / 3.7))
    max_ctx_q4 = min(max_ctx_q4, max_ctx_limit)
    if max_ctx_q4 >= 2048:
        return (max_ctx_q4, "q4_k_m", "q4_k_m")

    # Fallback
    return (min(1024, max_ctx_limit), "q8_0", "q8_0")


def get_preferred_ctx(
    model_size_gb: float,
    family: str,
    config: Config,
) -> int:
    """Get preferred context size based on model size and family config.

    Uses small_ctx for models below 10GB, large_ctx otherwise.
    """
    family_config = config.families.get(family, {})
    threshold = 10  # uniform threshold; per-family small_threshold_gb removed

    if model_size_gb < threshold:
        return family_config.get("small_ctx", 0)
    return family_config.get("large_ctx", 131072)


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
        return 5000  # E4B
    if size_gb < 25:
        return 25000  # 31B
    return 40000


def _get_nemotron_kv_bytes_per_token(size_gb: float) -> int:
    if size_gb < 10:
        return 6000  # 4B
    return 25000  # 30B-A3B


def _get_phi4_kv_bytes_per_token(size_gb: float) -> int:
    if size_gb < 10:
        return 5000  # mini 4B
    return 15000  # 14B


def _get_granite_kv_bytes_per_token(size_gb: float) -> int:
    if size_gb < 5:
        return 40960  # 3B: 40 layers * 8 kv_heads * 64 head_dim * 2 bytes
    if size_gb < 15:
        return 131072  # 8B: 40 layers * 8 kv_heads * 128 head_dim * 2 bytes
    return 131072  # 30B: 64 layers * 8 kv_heads * 128 head_dim * 2 bytes


def _get_glm_kv_bytes_per_token(size_gb: float) -> int:
    if size_gb < 10:
        return 8000  # MoE small variants
    return 48000  # GLM-4.7-Flash 30B MoE, kv_lora_rank=512 compression


def _get_mistral_small_kv_bytes_per_token(size_gb: float) -> int:
    return 82000  # Mistral Small 3.1: 40 layers × 8 KV heads × 128 head_dim × 2


def _get_devstral_kv_bytes_per_token(size_gb: float) -> int:
    return 82000  # Devstral Small 2: same arch as Mistral Small (40×8×128×2)


def find_free_port(config: Config, start_port: int | None = None) -> int:
    """Find a free port in range.

    ``start_port`` is optional for backward compatibility with older callers
    that want to continue scanning from a specific port.
    """
    start, end = config.port_range

    if start_port is None:
        scan_start = start
    else:
        scan_start = min(max(start_port, start), end + 1)
        if scan_start > end:
            scan_start = start

    for port in range(scan_start, end + 1):
        if not is_port_used(port):
            return port

    for port in range(start, scan_start):
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


def detect_backend(
    turbo3: bool = False,
    turbo4: bool = False,
    ctk: str | None = None,
    ctv: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Detect backend and KV cache types from CLI flags.

    Returns: (backend, ctk, ctv)
    """
    if turbo3:
        return "turboquant", ctk or "turbo3", ctv or "turbo3"
    if turbo4:
        return "turboquant", ctk or "turbo4", ctv or "turbo4"
    if ctk and "turbo" in (ctk + (ctv or "")):
        return "turboquant", ctk, ctv
    return "llama.cpp", ctk, ctv


def get_model_pid(config: Config, port: int) -> int | None:
    """Get PID for a model on given port."""
    import json

    pid_dir = get_pid_dir()
    for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
        try:
            content = pid_file.read_text().strip()
            try:
                data = json.loads(content)
                return data.get("pid")
            except (json.JSONDecodeError, KeyError):
                return int(content)
        except (ValueError, OSError):
            continue
    return None


_sweep_stale_pids()
