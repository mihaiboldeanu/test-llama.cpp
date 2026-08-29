"""Shared domain dataclasses and protocols for Rama.

This module contains neutral types that must not import application services
or CLI code to prevent circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import re
    from pathlib import Path


@dataclass(frozen=True)
class FamilyCapabilities:
    """Declarative capabilities of a model family."""

    supports_reasoning: bool = False
    reasoning_template_key: str | None = None  # e.g., "enable_thinking"
    reasoning_disabled_value: bool | str | None = None
    supports_ngram_speculation: bool = True
    skip_chat_parsing: bool = False
    # Memory defaults
    default_no_mmap: bool = False  # Disable mmap by default (known loading issues)
    # MoE workarounds
    avoid_fit_ctx_for_moe: bool = False  # Skip --fit-ctx for MoE models of this family
    # Reasoning launch behavior (facts, not decisions)
    reasoning_preserve_thinking_ctk: bool = (
        False  # Add preserve_thinking to chat-template-kwargs
    )
    reasoning_use_no_mmproj: bool = False  # Add --no-mmproj in reasoning mode
    reasoning_use_samplers: bool = False  # Add --samplers in reasoning mode
    reasoning_use_preserve_flag: bool = False  # Add --reasoning-preserve flag


@dataclass(frozen=True)
class SamplerSettings:
    """Sampling parameters for generation."""

    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0


@dataclass(frozen=True)
class ContextPolicy:
    """Context size and KV cache policy."""

    small_ctx: int = 32768
    large_ctx: int = 65536
    preferred_ctx: int | None = None


@dataclass(frozen=True)
class MemorySettings:
    """Memory and VRAM allocation policy."""

    fit_target_mib: int = 1024  # Safety margin in MiB
    use_mmap: bool = True
    n_gpu_layers: int | None = None  # None means auto


@dataclass(frozen=True)
class LaunchSettings:
    """Backend launch arguments and environment."""

    backend: str = "llama.cpp"
    ctk: str | None = None  # KV cache key type
    ctv: str | None = None  # KV cache value type
    mtp: bool = False
    mtp_n_max: int = 3
    mtp_n_min: int = 0
    seed: int | None = None


@dataclass(frozen=True)
class FamilyDefaults:
    """Portable defaults for a model family."""

    sampler: SamplerSettings = field(default_factory=SamplerSettings)
    context: ContextPolicy = field(default_factory=ContextPolicy)


@dataclass(frozen=True)
class ModelInfo:
    """Discovered model metadata."""

    name: str
    path: Path
    size_bytes: int
    family_key: str
    quantization: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    @property
    def is_moe(self) -> bool:
        """Detect MoE architecture from model name."""
        lower = self.name.lower()
        return any(kw in lower for kw in ("a4b", "a3b", "a18b", "moe", "mixture"))


class KvEstimator(Protocol):
    """Strategy for estimating KV cache bytes per token."""

    def bytes_per_token(
        self, model: ModelInfo, ctk: str | None, ctv: str | None
    ) -> int: ...


@dataclass(frozen=True)
class FamilyDefinition:
    """Complete definition of a model family."""

    key: str
    matchers: tuple[re.Pattern[str], ...]
    capabilities: FamilyCapabilities = field(default_factory=FamilyCapabilities)
    defaults: FamilyDefaults = field(default_factory=FamilyDefaults)
    kv_estimator: KvEstimator | None = None


@dataclass(frozen=True)
class ResolvedModelProfile:
    """Fully resolved model profile with provenance tracking.

    This is the single source of truth passed to launchers and request planners.
    """

    model: ModelInfo
    context: ContextPolicy
    sampler: SamplerSettings
    memory: MemorySettings
    launch: LaunchSettings
    capabilities: FamilyCapabilities
    provenance: dict[str, str] = field(default_factory=dict)
