"""Declarative model family registry.

Replaces switch-statement detection in core.py with a data-driven registry.
Each family declares its matchers, capabilities, defaults, and KV cache estimator.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .domain.gguf import kv_bytes_per_token_exact
from .domain.models import (
    ContextPolicy,
    FamilyCapabilities,
    FamilyDefaults,
    FamilyDefinition,
    KvEstimator,
    ModelInfo,
    SamplerSettings,
)

# ---------------------------------------------------------------------------
# KV Cache Estimators (strategy pattern)
# ---------------------------------------------------------------------------


class _SizeBandKvEstimator:
    """Declarative size-to-KV estimate with one final catch-all band."""

    def __init__(self, bands: tuple[tuple[float | None, int], ...]):
        self._bands = bands

    def bytes_per_token(
        self, model: ModelInfo, _ctk: str | None, _ctv: str | None
    ) -> int:
        for maximum_size_gb, bytes_per_token in self._bands:
            if maximum_size_gb is None or model.size_gb < maximum_size_gb:
                return bytes_per_token
        raise RuntimeError("KV estimator requires a final catch-all band")


_QWEN_KV = _SizeBandKvEstimator(((10, 11000), (20, 22000), (35, 45000), (None, 60000)))
_GEMMA_KV = _SizeBandKvEstimator(((10, 5000), (25, 25000), (None, 40000)))
_BONSAI_KV = _SizeBandKvEstimator(((2, 57344), (None, 65536)))
_NEX_N2_KV = _SizeBandKvEstimator(((None, 10240),))
_NEMOTRON_KV = _SizeBandKvEstimator(((10, 6000), (None, 25000)))
_PHI4_KV = _SizeBandKvEstimator(((10, 5000), (None, 15000)))
_LFM_KV = _SizeBandKvEstimator(((None, 10240),))
_LING_KV = _SizeBandKvEstimator(((None, 20480),))
_GRANITE_KV = _SizeBandKvEstimator(((5, 40960), (None, 131072)))
_GLM_KV = _SizeBandKvEstimator(((10, 8000), (None, 48000)))
_MISTRAL_KV = _SizeBandKvEstimator(((None, 82000),))
_ORNITH_KV = _SizeBandKvEstimator(((10, 11000), (None, 45000)))
_LAGUNA_KV = _SizeBandKvEstimator(((None, 45000),))
_NORTH_CODE_KV = _SizeBandKvEstimator(((None, 45000),))
_MELLUM_KV = _SizeBandKvEstimator(((None, 25000),))
_GPT_OSS_KV = _SizeBandKvEstimator(((30, 24576), (None, 36864)))
_MUSE_KV = _SizeBandKvEstimator(((None, 53248),))


class _MetadataKvEstimator:
    """Exact KV cache from GGUF metadata; falls back to size-based heuristic.

    Uses the actual architecture fields (layers, KV heads, head dim) read from
    the GGUF header to compute exact bytes-per-token for the KV cache.
    If metadata is missing or incomplete, delegates to a fallback estimator and
    logs the explicit fallback with safety margin information.
    """

    def __init__(self, fallback: KvEstimator):
        self._fallback = fallback
        self._log_once = True

    def bytes_per_token(
        self, model: ModelInfo, ctk: str | None, ctv: str | None
    ) -> int:
        if model.metadata is not None:
            exact = kv_bytes_per_token_exact(model.metadata, ctk, ctv)
            if exact is not None:
                return exact
        # Metadata missing or incomplete — fall back to size-based estimate
        fallback_val = self._fallback.bytes_per_token(model, ctk, ctv)
        if self._log_once:
            self._log_once = False
            logging.debug(
                "KV cache for %s (%.1fGB): metadata unavailable, "
                "falling back to size heuristic -> %d B/token",
                model.name, model.size_gb, fallback_val,
            )
        return fallback_val


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

_P = re.compile(r"(?P<base>.+)-(?P<index>\d+)-of-(?P<count>\d+)\.gguf$", re.IGNORECASE)

# Family matchers — order matters; more specific families first.
_GLM = [re.compile(r"glm", re.IGNORECASE)]
_DEVSTRAL = [re.compile(r"devstral", re.IGNORECASE)]
_MISTRAL_SMALL = [re.compile(r"mistral-small", re.IGNORECASE)]
_GEMMA = [re.compile(r"gemma", re.IGNORECASE)]
_BONSAI_27 = [re.compile(r"bonsai", re.IGNORECASE), re.compile(r"27b", re.IGNORECASE)]
_BONSAI = [re.compile(r"bonsai", re.IGNORECASE)]
_QWEN_38 = [re.compile(r"qwen[._-]?3[._-]?8", re.IGNORECASE)]
_QWEN_CODER = [re.compile(r"qwen3-coder", re.IGNORECASE)]
_GRUG = [re.compile(r"grug", re.IGNORECASE)]
_NEX_N2 = [re.compile(r"nex-n2", re.IGNORECASE)]
_QWEN = [re.compile(r"qwen", re.IGNORECASE)]
_GPT_OSS = [re.compile(r"gpt-oss", re.IGNORECASE)]
_LFM = [re.compile(r"lfm", re.IGNORECASE)]
_LING = [re.compile(r"ling", re.IGNORECASE)]
_NEMOTRON = [re.compile(r"nemotron", re.IGNORECASE)]
_PHI4 = [re.compile(r"phi-4|phi4", re.IGNORECASE)]
_GRANITE = [re.compile(r"granite", re.IGNORECASE)]
_ORNITH = [re.compile(r"ornith", re.IGNORECASE)]
_LAGUNA = [re.compile(r"laguna", re.IGNORECASE)]
_NORTH_CODE = [re.compile(r"north", re.IGNORECASE)]
_MELLUM = [re.compile(r"mellum", re.IGNORECASE)]
_MUSE = [re.compile(r"muse[-_]glimmer", re.IGNORECASE)]


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def _make(
    matchers: list[re.Pattern[str]], **kwargs: Any
) -> FamilyDefinition:
    """Helper to build a FamilyDefinition with sensible defaults.

    If kv_estimator is provided, it's wrapped with _MetadataKvEstimator so that
    GGUF metadata is tried first before falling back to the size-based heuristic.
    """
    est = kwargs.get("kv_estimator")
    # Wrap with metadata-aware estimator (falls back to original if metadata missing)
    if est is not None:
        kwargs["kv_estimator"] = _MetadataKvEstimator(est)
    return FamilyDefinition(
        key=kwargs.pop("key", ""),
        matchers=tuple(matchers),
        capabilities=kwargs.get("capabilities", FamilyCapabilities()),
        defaults=kwargs.get("defaults", FamilyDefaults()),
        kv_estimator=kwargs["kv_estimator"],
    )


def _match_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(text) for p in patterns)


def _all_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """All patterns must match (used for bonsai-27 which needs both 'bonsai' AND '27b')."""
    return all(p.search(text) for p in patterns)


class FamilyRegistry:
    """Declarative registry of model families.

    Families are checked in registration order; first match wins.
    """

    def __init__(self):
        self._families: list[FamilyDefinition] = []
        self._by_key: dict[str, FamilyDefinition] = {}

    def register(self, family: FamilyDefinition) -> None:
        self._families.append(family)
        self._by_key[family.key] = family

    def detect(self, filename: str, parent_name: str = "") -> str:
        """Detect family key from filename and optional parent directory name."""
        combined = f"{filename} {parent_name}"
        for fam in self._families:
            if _match_any(combined, fam.matchers):
                return fam.key
        return "unknown"

    def detect_multi_match(self, filename: str, parent_name: str = "") -> str:
        """Detect family where all matchers must match (e.g., bonsai-27)."""
        combined = f"{filename} {parent_name}"
        for fam in self._families:
            if _all_match(combined, fam.matchers):
                return fam.key
        return "unknown"

    def get(self, key: str) -> FamilyDefinition | None:
        return self._by_key.get(key)

    @property
    def all_families(self) -> list[FamilyDefinition]:
        return list(self._families)


# ---------------------------------------------------------------------------
# Default registry instance
# ---------------------------------------------------------------------------


def _build_registry() -> FamilyRegistry:
    reg = FamilyRegistry()

    # GLM
    reg.register(
        _make(
            key="glm",
            matchers=_GLM,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_GLM_KV,
        )
    )

    # Devstral (before mistral-small to avoid false match)
    reg.register(
        _make(
            key="devstral",
            matchers=_DEVSTRAL,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_MISTRAL_KV,
        )
    )

    # Mistral Small
    reg.register(
        _make(
            key="mistral-small",
            matchers=_MISTRAL_SMALL,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_MISTRAL_KV,
        )
    )

    # Gemma
    reg.register(
        _make(
            key="gemma",
            matchers=_GEMMA,
            capabilities=FamilyCapabilities(
                supports_ngram_speculation=True,
                avoid_fit_ctx_for_moe=True,
                reasoning_use_no_mmproj=True,
                reasoning_use_samplers=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=32768),
            ),
            kv_estimator=_GEMMA_KV,
        )
    )

    # Bonsai-27 (must come before bonsai — all-match pattern)
    reg.register(
        _make(
            key="bonsai-27",
            matchers=_BONSAI_27,
            capabilities=FamilyCapabilities(
                supports_reasoning=True,
                reasoning_template_key="enable_thinking",
                reasoning_disabled_value=False,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=32768),
            ),
            kv_estimator=_BONSAI_KV,
        )
    )

    # Bonsai (generic)
    reg.register(
        _make(
            key="bonsai",
            matchers=_BONSAI,
            capabilities=FamilyCapabilities(
                supports_reasoning=True,
                reasoning_template_key="enable_thinking",
                reasoning_disabled_value=False,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=32768),
            ),
            kv_estimator=_BONSAI_KV,
        )
    )

    # Qwen3.8 (before generic Qwen)
    reg.register(
        _make(
            key="qwen3.8",
            matchers=_QWEN_38,
            capabilities=FamilyCapabilities(
                supports_ngram_speculation=True,
                reasoning_preserve_thinking_ctk=True,
            ),
            defaults=FamilyDefaults(
                sampler=SamplerSettings(
                    temperature=1.0,
                    top_k=20,
                    top_p=0.95,
                    min_p=0.0,
                    repeat_penalty=1.0,
                    presence_penalty=0.0,
                ),
                context=ContextPolicy(small_ctx=32768, large_ctx=131072),
            ),
            kv_estimator=_QWEN_KV,
        )
    )

    # Qwen Coder
    reg.register(
        _make(
            key="qwen-coder",
            matchers=_QWEN_CODER,
            capabilities=FamilyCapabilities(
                supports_ngram_speculation=True,
                reasoning_preserve_thinking_ctk=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=32768, large_ctx=131072),
            ),
            kv_estimator=_QWEN_KV,
        )
    )

    # Grug
    reg.register(
        _make(
            key="grug",
            matchers=_GRUG,
            capabilities=FamilyCapabilities(
                supports_ngram_speculation=True,
                reasoning_preserve_thinking_ctk=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=32768, large_ctx=131072),
            ),
            kv_estimator=_QWEN_KV,
        )
    )

    # Nex-N2
    reg.register(
        _make(
            key="nex-n2",
            matchers=_NEX_N2,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_NEX_N2_KV,
        )
    )

    # Qwen (generic — after coder)
    reg.register(
        _make(
            key="qwen",
            matchers=_QWEN,
            capabilities=FamilyCapabilities(
                supports_ngram_speculation=True,
                reasoning_preserve_thinking_ctk=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=32768, large_ctx=131072),
            ),
            kv_estimator=_QWEN_KV,
        )
    )

    # GPT-OSS
    reg.register(
        _make(
            key="gpt-oss",
            matchers=_GPT_OSS,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_GPT_OSS_KV,
        )
    )

    # LFM
    reg.register(
        _make(
            key="lfm",
            matchers=_LFM,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_LFM_KV,
        )
    )

    # Ling
    reg.register(
        _make(
            key="ling",
            matchers=_LING,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_LING_KV,
        )
    )

    # Nemotron
    reg.register(
        _make(
            key="nemotron",
            matchers=_NEMOTRON,
            capabilities=FamilyCapabilities(
                reasoning_use_no_mmproj=True,
                reasoning_use_samplers=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=32768),
            ),
            kv_estimator=_NEMOTRON_KV,
        )
    )

    # Phi-4
    reg.register(
        _make(
            key="phi4",
            matchers=_PHI4,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=32768),
            ),
            kv_estimator=_PHI4_KV,
        )
    )

    # Granite
    reg.register(
        _make(
            key="granite",
            matchers=_GRANITE,
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=8192, large_ctx=16384),
            ),
            kv_estimator=_GRANITE_KV,
        )
    )

    # Ornith
    reg.register(
        _make(
            key="ornith",
            matchers=_ORNITH,
            capabilities=FamilyCapabilities(
                default_no_mmap=True,
                reasoning_use_preserve_flag=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_ORNITH_KV,
        )
    )

    # Laguna
    reg.register(
        _make(
            key="laguna",
            matchers=_LAGUNA,
            capabilities=FamilyCapabilities(
                default_no_mmap=True,
                reasoning_use_preserve_flag=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_LAGUNA_KV,
        )
    )

    # North Code
    reg.register(
        _make(
            key="north-code",
            matchers=_NORTH_CODE,
            capabilities=FamilyCapabilities(
                default_no_mmap=True,
                reasoning_use_preserve_flag=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_NORTH_CODE_KV,
        )
    )

    # Mellum
    reg.register(
        _make(
            key="mellum",
            matchers=_MELLUM,
            capabilities=FamilyCapabilities(
                default_no_mmap=True,
                reasoning_use_preserve_flag=True,
            ),
            defaults=FamilyDefaults(
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_MELLUM_KV,
        )
    )

    # Muse Glimmer
    reg.register(
        _make(
            key="muse-glimmer",
            matchers=_MUSE,
            capabilities=FamilyCapabilities(
                supports_reasoning=True,
                reasoning_template_key="reasoning_strength",
                reasoning_use_preserve_flag=True,
                supports_ngram_speculation=True,
            ),
            defaults=FamilyDefaults(
                sampler=SamplerSettings(temperature=1.0, top_p=0.95, top_k=64),
                context=ContextPolicy(small_ctx=16384, large_ctx=131072),
            ),
            kv_estimator=_MUSE_KV,
        )
    )

    return reg


@lru_cache(maxsize=1)
def get_registry() -> FamilyRegistry:
    """Return the immutable, process-wide family registry."""
    return _build_registry()


# Convenience functions for core.py migration
def detect_family(filename: str, parent_name: str = "") -> str:
    """Detect model family key from filename and parent directory."""
    reg = get_registry()
    # Try multi-match first (for bonsai-27 which requires both patterns)
    key = reg.detect_multi_match(filename, parent_name)
    if key != "unknown":
        return key
    return reg.detect(filename, parent_name)


def detect_quant(filename: str) -> str:
    """Detect quantization level from filename."""
    lower = filename.lower()
    for tag in ("iq4", "q8", "q6", "q5", "q4", "iq3", "iq2", "iq1", "q2", "q1"):
        if tag in lower:
            return tag.upper()
    return "unknown"


def detect_tags(filename: str) -> list[str]:
    """Detect special tags from filename."""
    lower = filename.lower()
    tags = []
    for tag in ("uncensored", "crow", "fernflower", "qwopus", "heretic"):
        if tag in lower:
            tags.append(tag)
    return tags


def kv_bytes_per_token(
    family_key: str,
    size_gb: float,
    metadata: dict[str, Any] | None = None,
    ctk: str | None = None,
    ctv: str | None = None,
) -> int:
    """Get KV cache bytes-per-token estimate for a family.

    If metadata is provided and complete, uses exact GGUF architecture fields.
    Otherwise falls back to size-based heuristics. Returns 15000 for unknown families.
    """
    reg = get_registry()
    fam = reg.get(family_key)
    if fam is not None and fam.kv_estimator is not None:
        dummy = ModelInfo(
            name="dummy",
            path=Path(__file__),
            size_bytes=int(size_gb * 1024**3),
            family_key=family_key,
            metadata=metadata,
        )
        return fam.kv_estimator.bytes_per_token(dummy, ctk, ctv)
    return 15000
