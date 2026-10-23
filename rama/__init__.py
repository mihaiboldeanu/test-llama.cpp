import copy
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
    "model_dir": str(Path.home() / "models"),
    "llama_cpp_dir": str(Path.home() / "Projects" / "llama.cpp"),
    "turbo_dir": str(Path.home() / "Projects" / "llama-cpp-turboquant"),
    "llama_cpp_repo": "https://github.com/ggml-org/llama.cpp.git",
    "turboquant_repo": "https://github.com/TheTom/llama-cpp-turboquant.git",
    "template_dir": "templates",
    "cuda_root": "~/cuda",
    "build_jobs": 8,
    "host": "127.0.0.1",
    "vram_total_gb": 24,
    "vram_headroom_gb": 2,
    "default_port": 11435,
    "port_range_end": 11500,
    "rules": {
        "small_ctx": 32768,
        "large_ctx": 65536,
        "small_threshold_gb": 10,
        "mtp_ngl_threshold_gb": 20,
        "mtp_ctx_ratio": 0.8,
    },
    "args": {
        "-np": 1,
        "--ctx-size": "auto",
        "-ctk": "q8_0",
        "-ctv": "q8_0",
        "--flash-attn": "on",
        "-b": 4096,
        "-ub": 2048,
        "--threads": 8,
        "--threads-batch": 16,
        "--temp": 0.6,
        "--top-k": 20,
        "--top-p": 0.95,
        "--min-p": 0.0,
        "--repeat-penalty": 1.0,
        "--presence-penalty": 0.0,
        "--perf": True,
        "--kv-unified": True,
        "--no-context-shift": True,
        "--fit": "on",
        "--fit-target": 1024,
        "--fit-ctx": "auto",
        "-n": 2048,
        "--reasoning": "auto",
    },
    "families": {},
    "models": {},
}


class ConfigError(ValueError):
    """Raised when a Rama configuration cannot be used safely."""


_VALID_LOAD_MODES = {"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"}
_VALID_LAZY_MODES = {"auto", "on", "off"}


def _translate_repetition_penalty(node: Any, warned: list[bool]) -> None:
    if isinstance(node, dict):
        if "repetition_penalty" in node and "--repeat-penalty" not in node:
            node["--repeat-penalty"] = node.pop("repetition_penalty")
            if not warned[0]:
                warnings.warn(
                    "Config key 'repetition_penalty' is deprecated; "
                    "use '--repeat-penalty' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                warned[0] = True
        for value in node.values():
            _translate_repetition_penalty(value, warned)


def _normalize_mode_value(value: Any, allowed: set[str], path: str, key: str) -> str:
    if isinstance(value, bool):
        # Preserve historical boolean forms before validating the word form.
        text = "mmap" if key == "--load-mode" and value else "none"
        if key == "--lazy-mode":
            text = "on" if value else "off"
    elif isinstance(value, str):
        text = value.strip().lower()
    else:
        text = str(value).strip().lower()
    if text not in allowed:
        raise ConfigError(f"{path}.{key} must be one of: {', '.join(sorted(allowed))}")
    return text


def _normalize_section_modes(section: dict[str, Any], path: str) -> None:
    args = section.get("args")
    if not isinstance(args, dict):
        args = {}
        section["args"] = args
    legacy = {
        "load_mode": "--load-mode",
        "tensor_read_lazy": "--lazy-mode",
    }
    for old_key, flag in legacy.items():
        if old_key in section:
            value = section.pop(old_key)
            if flag not in args:
                allowed = _VALID_LOAD_MODES if flag == "--load-mode" else _VALID_LAZY_MODES
                args[flag] = _normalize_mode_value(value, allowed, path, flag)
    for flag, allowed in (
        ("--load-mode", _VALID_LOAD_MODES),
        ("--lazy-mode", _VALID_LAZY_MODES),
    ):
        if args.get(flag) is not None:
            args[flag] = _normalize_mode_value(args[flag], allowed, path, flag)


def _normalize_llama_mode_fields(cfg: dict[str, Any]) -> None:
    _normalize_section_modes(cfg, "config")
    for section_name in ("families", "models"):
        section = cfg.get(section_name, {})
        if isinstance(section, dict):
            for name, settings in section.items():
                if isinstance(settings, dict):
                    _normalize_section_modes(settings, f"{section_name}.{name}")


def _check_args_map(args: Any, path: str, errors: list[str]) -> None:
    if not isinstance(args, dict):
        errors.append(f"{path}.args must be a mapping of llama-server flags")
        return
    for flag, value in args.items():
        if not isinstance(flag, str) or not flag.startswith("-"):
            errors.append(f"{path}.args keys must be llama-server flags, got {flag!r}")
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            if flag == "--chat-template-kwargs" and isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    errors.append(f"{path}.args.{flag} must contain valid JSON")
                else:
                    if not isinstance(parsed, dict):
                        errors.append(
                            f"{path}.args.{flag} must decode to a JSON object"
                        )
            continue
        errors.append(
            f"{path}.args.{flag} must be a string, number, boolean, or null"
        )


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for a resolved configuration."""
    errors: list[str] = []

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    for key in ("build_jobs",):
        if key in cfg and (not is_int(cfg[key]) or cfg[key] <= 0):
            errors.append(f"{key} must be a positive integer")

    for key in ("vram_total_gb", "vram_headroom_gb"):
        if key in cfg and (
            not isinstance(cfg[key], (int, float))
            or isinstance(cfg[key], bool)
            or cfg[key] < 0
        ):
            errors.append(f"{key} must be a non-negative number")

    for key in ("default_port", "port_range_end"):
        if key in cfg and (not is_int(cfg[key]) or not 1024 <= cfg[key] <= 65535):
            errors.append(f"{key} must be a TCP port between 1024 and 65535")
    if cfg.get("default_port", 0) > cfg.get("port_range_end", 0):
        errors.append("default_port must not be greater than port_range_end")

    rules = cfg.get("rules", {})
    if not isinstance(rules, dict):
        errors.append("rules must be a mapping of runtime thresholds")
    else:
        for key in (
            "small_ctx",
            "large_ctx",
            "small_threshold_gb",
            "mtp_ngl_threshold_gb",
            "mtp_ctx_ratio",
        ):
            value = rules.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"rules.{key} must be a number")
            elif value <= 0:
                errors.append(f"rules.{key} must be a positive number")

    _check_args_map(cfg.get("args", {}), "config", errors)

    for section_name in ("families", "models"):
        section = cfg.get(section_name, {})
        if not isinstance(section, dict):
            errors.append(f"{section_name} must be a mapping of names to settings")
            continue
        for name, settings in section.items():
            if not isinstance(settings, dict):
                errors.append(f"every {section_name.rstrip('s')} setting must be a mapping")
                continue
            path = f"{section_name}.{name}"
            if "args" in settings:
                _check_args_map(settings["args"], path, errors)
            for key in ("small_ctx", "large_ctx"):
                if key in settings and (
                    not is_int(settings[key]) or settings[key] <= 0
                ):
                    errors.append(f"{path}.{key} must be a positive integer")
            env = settings.get("env")
            if env is not None and not isinstance(env, dict):
                errors.append(f"{path}.env must be a mapping")
            mtp = settings.get("mtp")
            if mtp is not None and not isinstance(mtp, (bool, dict)):
                errors.append(f"{path}.mtp must be a boolean or mapping")

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


def _deep_merge(base: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Merge user settings over defaults; a partial section keeps the rest."""
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load config from file or use defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)

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

        _translate_repetition_penalty(user_cfg, [False])
        _deep_merge(cfg, user_cfg)

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

    _normalize_llama_mode_fields(cfg)
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

    @property
    def models(self) -> dict:
        return self._data.get("models", {})

    @property
    def args(self) -> dict:
        return self._data.get("args", {})

    @property
    def rules(self) -> dict:
        return self._data.get("rules", {})
