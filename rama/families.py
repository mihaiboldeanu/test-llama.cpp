"""Declarative model family registry.

Replaces switch-statement detection in core.py with a data-driven registry.
Each family declares its matchers, KV cache estimator, and MoE fit-context
rule. Every launch value lives in rama.yaml.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .domain.gguf import get_block_count, kv_bytes_per_token_exact
from .domain.models import FamilyDefinition, KvEstimator, ModelInfo

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
_QWEN_FLASH_NEXT_KV = _SizeBandKvEstimator(((None, 49152),))
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
# Flint: 49 layers, 4 KV heads, 128-dim heads -> 50,176 B/token (q8) on the
# dense worst case. The hybrid SWA helper below cuts that to the real cache
# shape (35 SWA layers at the 513-token window + 14 full-attention layers).
_FLINT_KV = _SizeBandKvEstimator(((None, 50176),))
_FLINT_HYBRID: tuple[int, float] = (513, 14 / 49)
_GPT_OSS_KV = _SizeBandKvEstimator(((30, 24576), (None, 36864)))
_MUSE_KV = _SizeBandKvEstimator(((None, 53248),))
# Spark-X2.5 is a hybrid SWA/full-attention dense model (1 full layer per 3
# SWA layers, window 512), so its KV cache is much smaller than a dense model
# of the same size. The hybrid recipe below amortizes the exact metadata dims;
# the size band only applies when GGUF metadata is unreadable.
_SPARK_KV = _SizeBandKvEstimator(((None, 16384),))
_SPARK_HYBRID: tuple[int, float] = (512, 0.25)
# MiniCPM5-2B: 42 layers, GQA with 2 KV heads and 128-dim heads -> about
# 21.5KB per token with q8 KV; exact GGUF metadata wins when available.
_MINICPM5_KV = _SizeBandKvEstimator(((None, 22000),))
# Nanbeige4.2-3B: custom Looped Transformer, so the size-band fallback is
# sized like a dense small Qwen; exact GGUF metadata wins when readable.
_NANBEIGE_KV = _SizeBandKvEstimator(((10, 11000), (None, 22000)))
# NeoHorse-1-4B: Qwen3.5-4B architecture (text-only repackage), so KV cache
# sits in the same size bands as a small dense Qwen.
_NEOHORSE_KV = _SizeBandKvEstimator(((10, 11000), (None, 22000)))


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
_HUMANIZER = [re.compile(r"humanizer", re.IGNORECASE)]
_BONSAI_27 = [re.compile(r"bonsai", re.IGNORECASE), re.compile(r"27b", re.IGNORECASE)]
_BONSAI = [re.compile(r"bonsai", re.IGNORECASE)]
_QWEN_38_FLASH_NEXT = [
    re.compile(r"qwen[._-]?3[._-]?8.*(?:flash.*next|next.*flash)", re.IGNORECASE)
]
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
_FLINT = [re.compile(r"flint", re.IGNORECASE)]
_MUSE = [re.compile(r"muse[-_]glimmer", re.IGNORECASE)]
_K2HORIZON = [re.compile(r"horizon|mova", re.IGNORECASE)]
_SPARK = [re.compile(r"spark", re.IGNORECASE)]
_MINICPM5 = [re.compile(r"minicpm", re.IGNORECASE)]
_NANBEIGE = [re.compile(r"nanbeige", re.IGNORECASE)]
_NEOHORSE = [re.compile(r"neohorse", re.IGNORECASE)]


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def _make(
    matchers: list[re.Pattern[str]],
    *,
    key: str,
    kv_estimator: KvEstimator | None = None,
    moe_fit_ctx_skip: bool = False,
    hybrid: tuple[int, float] | None = None,
) -> FamilyDefinition:
    """Build a slim family definition.

    Only matchers, KV estimators, and the MoE fit-context rule live in code.
    Every launch value lives in rama.yaml.
    """
    estimator = (
        _MetadataKvEstimator(kv_estimator) if kv_estimator is not None else None
    )
    return FamilyDefinition(
        key=key,
        matchers=tuple(matchers),
        kv_estimator=estimator,
        moe_fit_ctx_skip=moe_fit_ctx_skip,
        hybrid=hybrid,
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
            kv_estimator=_GLM_KV,
        )
    )

    # Devstral (before mistral-small to avoid false match)
    reg.register(
        _make(
            key="devstral",
            matchers=_DEVSTRAL,
            kv_estimator=_MISTRAL_KV,
        )
    )

    # Mistral Small
    reg.register(
        _make(
            key="mistral-small",
            matchers=_MISTRAL_SMALL,
            kv_estimator=_MISTRAL_KV,
        )
    )

    # Gemma
    reg.register(
        _make(
            key="gemma",
            matchers=_GEMMA,
            kv_estimator=_GEMMA_KV,
            moe_fit_ctx_skip=True,
        )
    )

    # Humanizer (jialinyyzz 12B, gemma-4 arch) — filename carries no
    # gemma marker, so it needs its own matcher. Same KV estimator.
    reg.register(
        _make(
            key="humanizer",
            matchers=_HUMANIZER,
            kv_estimator=_GEMMA_KV,
        )
    )

    # Bonsai-27 (must come before bonsai — all-match pattern)
    reg.register(
        _make(
            key="bonsai-27",
            matchers=_BONSAI_27,
            kv_estimator=_BONSAI_KV,
        )
    )

    # Bonsai (generic)
    reg.register(
        _make(
            key="bonsai",
            matchers=_BONSAI,
            kv_estimator=_BONSAI_KV,
        )
    )

    # Qwen3.8 Flash Next (before qwen3.8 — its name also contains qwen3.8)
    # Same Qwen hybrid lineage as qwen3.8 (1 full per 3 SWA, 16K window).
    reg.register(
        _make(
            key="qwen3.8-flash-next",
            matchers=_QWEN_38_FLASH_NEXT,
            kv_estimator=_QWEN_FLASH_NEXT_KV,
            hybrid=(16384, 0.25),
        )
    )

    # Qwen3.8 (before generic Qwen)
    # Hybrid attention verified against a live 27B server: the dense formula
    # predicts ~71K but the box runs 154K in ~5.4GB KV, matching 1-full-per-4
    # with a 16K sliding window.
    reg.register(
        _make(
            key="qwen3.8",
            matchers=_QWEN_38,
            kv_estimator=_QWEN_KV,
            hybrid=(16384, 0.25),
        )
    )

    # Qwen Coder
    reg.register(
        _make(
            key="qwen-coder",
            matchers=_QWEN_CODER,
            kv_estimator=_QWEN_KV,
        )
    )

    # Grug
    reg.register(
        _make(
            key="grug",
            matchers=_GRUG,
            kv_estimator=_QWEN_KV,
        )
    )

    # Nex-N2
    reg.register(
        _make(
            key="nex-n2",
            matchers=_NEX_N2,
            kv_estimator=_NEX_N2_KV,
        )
    )

    # Qwen (generic — after coder)
    reg.register(
        _make(
            key="qwen",
            matchers=_QWEN,
            kv_estimator=_QWEN_KV,
        )
    )

    # GPT-OSS
    reg.register(
        _make(
            key="gpt-oss",
            matchers=_GPT_OSS,
            kv_estimator=_GPT_OSS_KV,
        )
    )

    # LFM
    reg.register(
        _make(
            key="lfm",
            matchers=_LFM,
            kv_estimator=_LFM_KV,
        )
    )

    # Ling
    reg.register(
        _make(
            key="ling",
            matchers=_LING,
            kv_estimator=_LING_KV,
        )
    )

    # Nemotron
    reg.register(
        _make(
            key="nemotron",
            matchers=_NEMOTRON,
            kv_estimator=_NEMOTRON_KV,
        )
    )

    # Phi-4
    reg.register(
        _make(
            key="phi4",
            matchers=_PHI4,
            kv_estimator=_PHI4_KV,
        )
    )

    # Granite
    reg.register(
        _make(
            key="granite",
            matchers=_GRANITE,
            kv_estimator=_GRANITE_KV,
        )
    )

    # Ornith
    reg.register(
        _make(
            key="ornith",
            matchers=_ORNITH,
            kv_estimator=_ORNITH_KV,
        )
    )

    # Laguna
    reg.register(
        _make(
            key="laguna",
            matchers=_LAGUNA,
            kv_estimator=_LAGUNA_KV,
        )
    )

    # North Code
    reg.register(
        _make(
            key="north-code",
            matchers=_NORTH_CODE,
            kv_estimator=_NORTH_CODE_KV,
        )
    )

    # Mellum
    reg.register(
        _make(
            key="mellum",
            matchers=_MELLUM,
            kv_estimator=_MELLUM_KV,
        )
    )

    # Flint (Recursive). 39.8B-A2.5B coding-agent MoE: 2 dense + 47 MoE layers,
    # 35 sliding-window (513) + 14 full-attention layers, native 262K context.
    # Hybrid KV amortization mirrors the metadata's layer pattern exactly.
    reg.register(
        _make(
            key="flint",
            matchers=_FLINT,
            kv_estimator=_FLINT_KV,
            hybrid=_FLINT_HYBRID,
        )
    )

    # Muse Glimmer
    reg.register(
        _make(
            key="muse-glimmer",
            matchers=_MUSE,
            kv_estimator=_MUSE_KV,
        )
    )

    # Spark-X2.5 (XHToken). Hybrid SWA/full-attention dense model, native 1M
    # context, thinking on by default (enable_thinking, Qwen3-style). Official
    # sampling: temp 1.0, top_p 0.95, top_k disabled (0 in llama.cpp).
    # Total ctx covers n_parallel slots: 98304 = 32768 x 3.
    reg.register(
        _make(
            key="spark",
            matchers=_SPARK,
            kv_estimator=_SPARK_KV,
            hybrid=_SPARK_HYBRID,
        )
    )

    # MiniCPM5 (OpenBMB). Dense 2B LlamaForCausalLM, native 128K context,
    # deep-thinking SFT (enable_thinking). Official sampling lives in
    # rama.yaml: temp 1.0, top_p 0.95, min_p 0.0 (llama.cpp's default
    # min_p 0.05 causes repetitive output on this model).
    reg.register(
        _make(
            key="minicpm5",
            matchers=_MINICPM5,
            kv_estimator=_MINICPM5_KV,
        )
    )

    # Nanbeige4.2 (Nanbeige LLM Lab). Looped Transformer 3B, 256K context,
    # thinking on by default (enable_thinking). Official chat/reasoning
    # sampling (rama.yaml): temp 0.6, top_p 0.95, top_k 20; agentic
    # workloads use temp 1.0.
    reg.register(
        _make(
            key="nanbeige",
            matchers=_NANBEIGE,
            kv_estimator=_NANBEIGE_KV,
        )
    )

    # NeoHorse-1 (TokenRhythm). Qwen3.5-4B post-trained for agentic
    # harnesses; 256K native context, Qwen3 thinking template. Official
    # eval sampling (rama.yaml): temp 1.0, top_p 0.95, top_k 20,
    # min_p 0.0, presence_penalty 1.5.
    reg.register(
        _make(
            key="neohorse",
            matchers=_NEOHORSE,
            kv_estimator=_NEOHORSE_KV,
        )
    )

    # K2-Horizon (IFM). Sparse MoE with MoVA attention (MoVA-36B-A4B: 36B
    # total, 4B active; also covers the dense K2-Horizon-32B). Native 512K
    # context, thinking on by default at high reasoning effort. Official
    # sampling (rama.yaml): temp 1.0, top_p 0.95. Requires llama.cpp with
    # K2-Horizon arch support (MBZUAI-IFM fork, branch model/K2Horizon)
    # until upstream merges it. Reuses the Qwen size bands for the KV
    # fallback; exact GGUF metadata wins when readable.
    reg.register(
        _make(
            key="k2-horizon",
            matchers=_K2HORIZON,
            kv_estimator=_QWEN_KV,
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


def hybrid_kv_ctx(
    budget_bytes: int,
    family_key: str,
    metadata: dict[str, Any] | None,
    ctk: str | None = None,
    ctv: str | None = None,
) -> int | None:
    """Solve max context for a hybrid SWA/full-attention family.

    Dense layers scale KV with ctx, but SWA layers saturate at the sliding
    window, so bytes(ctx) = bpt_layer * (n_full*ctx + n_swa*min(ctx, window)).
    Returns None when the family is not registered hybrid or the GGUF
    metadata lacks a layer count (caller keeps the dense estimate).
    """
    fam = get_registry().get(family_key)
    if fam is None or fam.hybrid is None or metadata is None:
        return None
    window, full_ratio = fam.hybrid
    exact = kv_bytes_per_token_exact(metadata, ctk, ctv)
    n_layers = get_block_count(metadata)
    if exact is None or type(n_layers) is not int or n_layers < 1:
        return None
    ratio = min(max(full_ratio, 0.0), 1.0)
    n_full = max(1, min(n_layers - 1 if n_layers > 1 else 1, int(n_layers * ratio + 0.5)))
    n_swa = n_layers - n_full
    bpt_layer = exact / n_layers
    layer_tokens = budget_bytes / bpt_layer
    if layer_tokens / n_layers < window:
        return int(layer_tokens / n_layers)
    return int((layer_tokens - n_swa * window) / n_full)
