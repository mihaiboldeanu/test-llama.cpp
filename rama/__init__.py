import json
import os
import warnings
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_LOCATIONS = [
    Path("rama.yaml"),
    Path.home() / ".config" / "rama.yaml",
]

DEFAULT_CONFIG = {
    "model_dir": str(Path.home() / "ollama"),
    "llama_cpp_dir": str(Path.home() / "Projects" / "llama.cpp"),
    "turbo_dir": str(Path.home() / "Projects" / "llama-cpp-turboquant"),
    "llama_cpp_repo": "https://github.com/ggml-org/llama.cpp.git",
    "turboquant_repo": "https://github.com/TheTom/llama-cpp-turboquant.git",
    "template_dir": str(Path(__file__).parent.parent / "templates"),
    "reasoning_budget": 1024,
    "reasoning_budget_message": "I have thought enough, let's move to answering.",
    "cuda_root": "~/cuda",
    "build_jobs": 8,
    "threads": 8,
    "threads_batch": 16,
    "vram_total_gb": 24,
    "vram_headroom_gb": 2,
    "fit_target_mib": 1024,
    "batch_size": 4096,
    "ubatch_size": 2048,
    "temp": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "seed": None,
    "chat_template_kwargs": '{"preserve_thinking": true}',
    "chat_template": None,
    "chat_template_file": None,
    "spec_type": "ngram-mod",
    "spec_ngram_mod_size_n": 16,
    "spec_ngram_mod_n_min": 24,
    "spec_ngram_mod_n_max": 48,
    "draft_min": 8,
    "draft_max": 24,
    "host": "127.0.0.1",
    "context_shift": False,
    "default_port": 11435,
    "port_range_end": 11500,
    "families": {},
    "models": {},
}

class ConfigError(ValueError):
    """Raised when a Rama configuration cannot be used safely."""

def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for a resolved configuration."""
    errors = []

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    for key in ("build_jobs", "threads", "threads_batch", "batch_size", "ubatch_size"):
        if key in cfg and (not is_int(cfg[key]) or cfg[key] <= 0):
            errors.append(f"{key} must be a positive integer")

    for key in ("vram_total_gb", "vram_headroom_gb"):
        if key in cfg and (
            not isinstance(cfg[key], (int, float))
            or isinstance(cfg[key], bool)
            or cfg[key] < 0
        ):
            errors.append(f"{key} must be a non-negative number")

    for key in ("temp", "top_p", "min_p"):
        if key in cfg and (
            not isinstance(cfg[key], (int, float))
            or isinstance(cfg[key], bool)
            or not 0 <= cfg[key] <= (2 if key == "temp" else 1)
        ):
            maximum = 2 if key == "temp" else 1
            errors.append(f"{key} must be a number between 0 and {maximum}")

    if "top_k" in cfg and (not is_int(cfg["top_k"]) or cfg["top_k"] < -1):
        errors.append("top_k must be an integer greater than or equal to -1")

    for key in ("default_port", "port_range_end"):
        if key in cfg and (not is_int(cfg[key]) or not 1024 <= cfg[key] <= 65535):
            errors.append(f"{key} must be a TCP port between 1024 and 65535")
    if cfg.get("default_port", 0) > cfg.get("port_range_end", 0):
        errors.append("default_port must not be greater than port_range_end")

    families = cfg.get("families", {})
    if not isinstance(families, dict):
        errors.append("families must be a mapping of family names to settings")
    elif any(not isinstance(settings, dict) for settings in families.values()):
        errors.append("every family setting must be a mapping")

    models = cfg.get("models", {})
    if not isinstance(models, dict):
        errors.append("models must be a mapping of model names to settings")
    elif any(not isinstance(settings, dict) for settings in models.values()):
        errors.append("every model setting must be a mapping")

    template_kwargs = cfg.get("chat_template_kwargs")
    if template_kwargs is not None:
        if not isinstance(template_kwargs, str):
            errors.append("chat_template_kwargs must be a JSON object string")
        else:
            try:
                parsed = json.loads(template_kwargs)
                if not isinstance(parsed, dict):
                    errors.append("chat_template_kwargs must decode to a JSON object")
            except json.JSONDecodeError:
                errors.append("chat_template_kwargs must contain valid JSON")

    return errors

def _detect_vram_gb() -> float | None:
    """Auto-detect VRAM via nvidia-smi."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Sum all GPUs' VRAM
            total_mb = sum(
                float(line.strip())
                for line in result.stdout.strip().split("\n")
                if line.strip()
            )
            return total_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None

def _detect_cuda_arch() -> str | None:
    """Auto-detect CUDA architecture from GPU name."""
    # GPU model suffix -> compute capability
    arch_map = {
        "4090": "89",
        "4080": "89",
        "4070": "89",
        "4060": "89",
        "3090": "86",
        "3080": "86",
        "3070": "86",
        "3060": "86",
        "2080": "75",
        "2070": "75",
        "2060": "75",
        "1080": "61",
        "1070": "61",
        "1060": "61",
    }
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            name = result.stdout.strip().split("\n")[0].strip()
            for suffix, arch in arch_map.items():
                if suffix in name:
                    return arch
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        pass
    return None

def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load config from file or use defaults."""
    cfg = DEFAULT_CONFIG.copy()

    # Find config file
    if config_path:
        path = Path(config_path)
    else:
        path = None
        for loc in DEFAULT_CONFIG_LOCATIONS:
            if loc.exists():
                path = loc
                break

    if path and path.exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        if not isinstance(user_cfg, dict):
            raise ConfigError(f"Configuration file must contain a mapping: {path}")

        # Handle repetition_penalty translation with deprecation warning
        _repetition_warned = False

        def translate_repetition(d: dict):
            nonlocal _repetition_warned
            if "repetition_penalty" in d:
                d["repeat_penalty"] = d.pop("repetition_penalty")
                if not _repetition_warned:
                    warnings.warn(
                        "Config key 'repetition_penalty' is deprecated; "
                        "use 'repeat_penalty' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    _repetition_warned = True
            for section in ("families", "models"):
                if section in d and isinstance(d[section], dict):
                    for name, settings in d[section].items():
                        if isinstance(settings, dict):
                            translate_repetition(settings)

        translate_repetition(user_cfg)
        cfg.update(user_cfg)

    # Auto-detect VRAM if not set by user
    if cfg.get("vram_total_gb") == DEFAULT_CONFIG["vram_total_gb"]:
        detected = _detect_vram_gb()
        if detected:
            cfg["vram_total_gb"] = detected

    # Auto-detect CUDA arch if not set by user
    if "cuda_arch" not in cfg:
        detected = _detect_cuda_arch()
        if detected:
            cfg["cuda_arch"] = detected

    # Expand paths
    for key in ["model_dir", "llama_cpp_dir", "turbo_dir", "cuda_root"]:
        if key in cfg:
            cfg[key] = os.path.expanduser(str(cfg[key]))

    errors = validate_config(cfg)
    if errors:
        raise ConfigError("Invalid Rama configuration:\n- " + "\n- ".join(errors))

    return cfg

class Config:
    """Config object with nice accessors."""

    def __init__(self, config_path: str | None = None):
        self._data = load_config(config_path)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def model_dir(self) -> Path:
        return Path(self._data["model_dir"])

    @property
    def llama_cpp_dir(self) -> Path:
        return Path(self._data["llama_cpp_dir"])

    @property
    def turbo_dir(self) -> Path:
        return Path(self._data["turbo_dir"])

    @property
    def cuda_root(self) -> Path:
        return Path(self._data["cuda_root"])

    @property
    def vram_available_gb(self) -> float:
        return self._data["vram_total_gb"] - self._data["vram_headroom_gb"]

    @property
    def port_range(self) -> tuple[int, int]:
        return (self._data["default_port"], self._data["port_range_end"])

    @property
    def families(self) -> dict:
        return self._data.get("families", {})
