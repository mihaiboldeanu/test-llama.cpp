import json
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path

from . import Config
from .domain.gguf import read_gguf_metadata
from .families import (
    detect_family as _families_detect_family,
)
from .families import (
    detect_quant,
    detect_tags,
    kv_bytes_per_token,
)


def _sweep_stale_pids() -> None:
    """Remove PID files for dead processes. Called once on import."""
    pid_dir = get_pid_dir()
    if not pid_dir.exists():
        return

    for pid_file in pid_dir.glob("model_*.pid"):
        try:
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
    metadata: dict | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    @property
    def is_moe(self) -> bool:
        """Detect MoE architecture from GGUF metadata, falling back to name."""
        from .domain.gguf import is_moe_from_metadata

        if self.metadata is not None:
            return is_moe_from_metadata(self.metadata)
        lower = self.name.lower()
        return any(kw in lower for kw in ("a4b", "a3b", "a18b", "moe", "mixture"))

    def to_domain(self) -> "DomainModelInfo":  # noqa: F821
        """Convert to domain.ModelInfo for ProfileResolver."""
        from .domain.models import ModelInfo as DomainModelInfo

        return DomainModelInfo(
            name=self.name,
            path=self.path,
            size_bytes=int(self.size_gb * 1024**3),
            family_key=self.family,
            quantization=self.quant if self.quant != "unknown" else None,
            metadata=self.metadata,
        )


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
    """Discover GGUF models, combining complete multi-part downloads."""
    model_dir = config.model_dir
    if not model_dir.exists():
        return []

    models = []
    split_pattern = re.compile(
        r"^(?P<base>.+)-(?P<index>\d+)-of-(?P<count>\d+)\.gguf$", re.IGNORECASE
    )
    split_groups: dict[tuple[Path, str, int], dict[int, Path]] = {}

    for gguf in model_dir.rglob("*.gguf"):
        if not gguf.is_file():
            continue

        split_match = split_pattern.match(gguf.name)
        if split_match:
            base = split_match.group("base")
            count = int(split_match.group("count"))
            index = int(split_match.group("index"))
            split_groups.setdefault((gguf.parent, base, count), {})[index] = gguf
            continue

        name = gguf.stem
        size_gb = gguf.stat().st_size / (1024**3)
        quant = detect_quant(name)
        family = detect_family(gguf)
        tags = detect_tags(name)
        metadata = read_gguf_metadata(gguf)

        models.append(
            ModelInfo(
                name=name,
                path=gguf,
                size_gb=round(size_gb, 1),
                family=family,
                quant=quant,
                tags=tags,
                metadata=metadata,
            ),
        )

    for (_, base, count), parts in split_groups.items():
        expected_parts = set(range(1, count + 1))
        if set(parts) != expected_parts:
            # A split GGUF cannot be loaded until every shard is present. Do
            # not expose individual shards as misleading runnable models.
            continue

        first_part = parts[1]
        size_gb = sum(path.stat().st_size for path in parts.values()) / (1024**3)
        models.append(
            ModelInfo(
                name=base,
                path=first_part,
                size_gb=round(size_gb, 1),
                family=detect_family(first_part),
                quant=detect_quant(base),
                tags=detect_tags(base),
                metadata=read_gguf_metadata(first_part),
            )
        )

    return sorted(models, key=lambda m: m.path)


def detect_family(path: Path | str, parent_name: str = "") -> str:
    """Detect model family.

    Accepts either a Path or a string (filename). For backward compatibility
    with existing tests that pass Path objects.
    """
    if isinstance(path, Path):
        parent = path.parent.name
        if parent in (".", ".."):
            parent = ""
        return _families_detect_family(path.name, parent)
    return _families_detect_family(path, parent_name)


def calc_kv_cache(
    model_size_gb: float,
    family: str,
    config: Config,
    metadata: dict | None = None,
    ctk: str | None = None,
    ctv: str | None = None,
) -> tuple[int, str, str]:
    """Calculate max KV cache that fits in VRAM.

    Uses the families registry for per-family KV bytes-per-token estimates.
    If GGUF metadata is provided, uses exact architecture fields for calculation.
    Returns: (ctx_size, qk_type, qkv_type)
    """
    if model_size_gb < 0:
        raise ValueError(f"Model size cannot be negative: {model_size_gb}")

    vram_available = max(config.vram_available_gb, 0)
    kv_vram = max(vram_available - max(model_size_gb, 0), 0.1)

    family_params = config.families.get(family, {})
    cache_type_k = ctk or family_params.get("ctk")
    cache_type_v = ctv or family_params.get("ctv")
    kv_per_token = kv_bytes_per_token(
        family,
        model_size_gb,
        metadata=metadata,
        ctk=cache_type_k,
        ctv=cache_type_v,
    )
    resolved_ctk = cache_type_k or "q8_0"
    resolved_ctv = cache_type_v or cache_type_k or "q8_0"

    # Very tight VRAM — minimal context
    if kv_vram < 1:
        tight_ctx = int((kv_vram * 1024**3) / kv_per_token)
        return (max(2048, min(tight_ctx, 8192)), resolved_ctk, resolved_ctv)

    max_ctx_limit = 262144
    max_ctx = int((kv_vram * 1024**3) / kv_per_token)
    max_ctx = min(max_ctx, max_ctx_limit)
    preferred = get_preferred_ctx(model_size_gb, family, config)
    max_ctx = min(max_ctx, preferred)

    if max_ctx >= 2048:
        return (max_ctx, resolved_ctk, resolved_ctv)

    # An explicitly selected cache format is a quality/compatibility choice.
    # Do not silently replace it with q4 when it cannot reach 2K context.
    if cache_type_k or cache_type_v:
        return (max(1024, max_ctx), resolved_ctk, resolved_ctv)

    # If not enough VRAM for q8_0, try q4_k_m
    max_ctx_q4 = int((kv_vram * 1024**3) / (kv_per_token / 3.7))
    max_ctx_q4 = min(max_ctx_q4, max_ctx_limit)
    if max_ctx_q4 >= 2048:
        return (max_ctx_q4, "q4_k_m", "q4_k_m")

    return (min(1024, max_ctx_limit), resolved_ctk, resolved_ctv)


def get_preferred_ctx(
    model_size_gb: float,
    family: str,
    config: Config,
) -> int:
    """Get preferred context size based on model size and family.

    Uses small_ctx for models below 10GB, large_ctx otherwise.
    Applies config overrides via ProfileResolver for consistency with launch.
    """
    from .domain.resolver import ProfileResolver
    from .families import get_registry

    reg_fam = get_registry().get(family)
    threshold = 10

    # Use ProfileResolver to apply config.families and global config overrides
    if reg_fam is not None:
        resolver = ProfileResolver(family_definition=reg_fam)
        family_override = (config.get("families") or {}).get(family) or None
        profile = resolver.resolve(
            None,  # No model info needed for context defaults
            global_config=config._data if hasattr(config, "_data") else None,
            family_override=family_override,
        )
        return profile.context.small_ctx if model_size_gb < threshold else profile.context.large_ctx

    # Fallback when family not in registry
    if model_size_gb < threshold:
        return 0
    return 131072


def find_free_port(config: Config | None, start_port: int | None = None) -> int:
    """Find a free port in range.

    ``start_port`` is optional for backward compatibility with older callers
    that want to continue scanning from a specific port.
    """
    # The domain batch executor has no mutable Config object. Keep its default
    # range explicit rather than failing with an AttributeError.
    start, end = config.port_range if config is not None else (11435, 12434)

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
