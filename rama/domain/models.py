"""Shared domain dataclasses and protocols for Rama.

This module contains neutral types that must not import application services
or CLI code to prevent circular dependencies.

After the args-map refactor, all launch settings live in rama.yaml as literal
llama-server flags (``args:`` blocks). The resolver merges them (global ->
family -> model -> CLI) and only computes what is genuinely runtime: identity
flags, ctx size, fit-ctx, the reasoning toggle, and the MTP block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .gguf import is_moe_from_metadata

if TYPE_CHECKING:
    import re
    from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    """Discovered GGUF model."""

    name: str
    path: Path
    size_bytes: int
    family_key: str
    quantization: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] | None = None

    @property
    def family(self) -> str:
        return self.family_key

    @property
    def quant(self) -> str:
        return self.quantization or "unknown"

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024**3), 1)

    @property
    def is_moe(self) -> bool:
        """Detect MoE architecture from GGUF metadata, falling back to name."""
        if self.metadata is not None:
            return is_moe_from_metadata(self.metadata)
        lower = self.name.lower()
        return any(kw in lower for kw in ("a4b", "a3b", "a18b", "moe", "mixture"))


class KvEstimator(Protocol):
    """Strategy for estimating KV cache bytes per token."""

    def bytes_per_token(
        self, model: ModelInfo, ctk: str | None, ctv: str | None
    ) -> int: ...


@dataclass(frozen=True)
class FamilyDefinition:
    """Complete definition of a model family.

    Matchers and KV estimators are code-side knowledge (regex + architecture
    heuristics). Every launch value lives in rama.yaml.
    """

    key: str
    matchers: tuple[re.Pattern[str], ...]
    kv_estimator: KvEstimator | None = None
    moe_fit_ctx_skip: bool = False
    # Hybrid SWA/full-attention architecture: (sliding window, fraction of
    # layers with full attention). Only set for families where the recipe is
    # confirmed (measured VRAM or documented); the GGUF headers carry no
    # window info, so unlisted families keep the dense worst-case estimate.
    hybrid: tuple[int, float] | None = None


@dataclass(frozen=True)
class ResolvedProfile:
    """Fully resolved launch profile: merged llama-server args + provenance.

    ``args`` contains only actual llama-server flags (values as strings/numbers;
    ``True`` means a bare presence-flag; keys removed by an override are absent).
    The remaining fields are the documented runtime computations: ctx size,
    fit-ctx, the reasoning toggle, MTP rule data, env and identity.
    """

    model: ModelInfo
    args: dict[str, Any]
    ctx: int
    fit_ctx: int | None
    reasoning: bool
    mtp: dict[str, Any]
    env: dict[str, Any]
    alias: str
    provenance: dict[str, str] = field(default_factory=dict)
