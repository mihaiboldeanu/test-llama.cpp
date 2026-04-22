import os
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_LOCATIONS = [
    Path("test_llama.yaml"),
    Path.home() / ".config" / "test_llama.yaml",
]

DEFAULT_CONFIG = {
    "model_dir": str(Path.home() / "ollama"),
    "llama_cpp_dir": str(Path.home() / "Projects" / "llama.cpp"),
    "turbo_dir": str(Path.home() / "Projects" / "llama-cpp-turboquant"),
    "cuda_root": "/opt/cuda-13.1",
    "build_jobs": 8,
    "threads": 8,
    "threads_batch": 16,
    "vram_total_gb": 24,
    "vram_headroom_gb": 2,
    "fit_target_mib": 1024,
    "batch_size": 2048,
    "ubatch_size": 2048,
    "temp": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "seed": None,
    "chat_template_kwargs": '{"preserver_thinking": true}',
    "default_port": 11435,
    "port_range_end": 11500,
    "families": {},
}


def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
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
        cfg.update(user_cfg)

    # Expand paths
    for key in ["model_dir", "llama_cpp_dir", "turbo_dir", "cuda_root"]:
        if key in cfg:
            cfg[key] = os.path.expanduser(str(cfg[key]))

    return cfg


class Config:
    """Config object with nice accessors."""

    def __init__(self, config_path: Optional[str] = None):
        self._data = load_config(config_path)

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
