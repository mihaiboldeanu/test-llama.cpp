import os
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
}


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
