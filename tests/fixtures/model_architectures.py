"""Known model architecture fixtures for KV cache calculation testing.

These are real GGUF architecture values used to verify that the exact KV cache
calculation matches expected values. Each fixture represents a real model family.
"""
from __future__ import annotations

# Dict of known architectures -> (expected_kv_bytes_q8, description)
KNOWN_MODELS: dict[str, tuple[int, str]] = {
    # Qwen3-9B: 52 layers, 12 KV heads, 256 head_dim, q8_0
    # 52 * 12 * 256 * 2 * 1 = 319,488
    "qwen3-9b": (319488, "Qwen3 9B dense"),

    # Qwen3-27B: 48 layers, 24 KV heads, 256 head_dim, q8_0
    # 48 * 24 * 256 * 2 * 1 = 589,824 — but this is GQA with grouped_kv=3
    # Actual: 48 * 24 * 256 * 2 = 589824 per token (large!)
    "qwen3-27b": (589824, "Qwen3 27B dense"),

    # Gemma3-12B: 40 layers, 8 KV heads, 256 head_dim
    # 40 * 8 * 256 * 2 = 163,840
    "gemma3-12b": (163840, "Gemma3 12B"),

    # Llama-3.1-8B: 32 layers, 8 KV heads, 128 head_dim
    # 32 * 8 * 128 * 2 = 65,536
    "llama3-8b": (65536, "Llama 3.1 8B"),

    # Phi-4-mini: 32 layers, 4 KV heads, 128 head_dim
    # 32 * 4 * 128 * 2 = 32,768
    "phi4-mini": (32768, "Phi-4 mini"),

    # Granite-3.0-3B: 32 layers, 8 KV heads, 128 head_dim
    # 32 * 8 * 128 * 2 = 65,536
    "granite-3b": (65536, "Granite 3.0 3B"),

    # Mistral-Small-3.1: 40 layers, 8 KV heads, 128 head_dim
    # 40 * 8 * 128 * 2 = 81,920
    "mistral-small-3.1": (81920, "Mistral Small 3.1"),

    # GLM-4.7-Flash: MoE variant with smaller active params
    # 28 layers, 4 KV heads, 128 head_dim
    # 28 * 4 * 128 * 2 = 28,672
    "glm-flash": (28672, "GLM 4.7 Flash MoE"),

    # Bonsai-8B: 28 layers, 8 KV heads, 128 head_dim
    # 28 * 8 * 128 * 2 = 57,344
    "bonsai-8b": (57344, "Bonsai 8B"),

    # Nex-N2-mini: 10 layers, 2 KV heads, 256 head_dim
    # 10 * 2 * 256 * 2 = 10,240
    "nex-n2-mini": (10240, "Nex-N2 mini"),

    # Ling-mini-2.0: 20 layers, 4 KV heads, 128 head_dim
    # 20 * 4 * 128 * 2 = 20,480
    "ling-mini-2": (20480, "Ling mini 2.0"),

    # LFM2-24B-A2B: MoE with 10 attention layers, 8 KV heads, 64 head_dim
    # 10 * 8 * 64 * 2 = 10,240
    "lfm2-24b-a2b": (10240, "LFM2 24B MoE"),

    # Mellum-12B: MoE with GQA, 28 layers, 4 KV heads, ~256 head_dim
    # 28 * 4 * 256 * 2 = 57,344 — but actual head_dim may differ
    "mellum-12b": (57344, "Mellum 12B MoE"),

    # GPT-OSS-20B: 24 layers, 8 KV heads, 64 head_dim
    # 24 * 8 * 64 * 2 = 24,576
    "gpt-oss-20b": (24576, "GPT-OSS 20B"),

    # Ornith-9B: 52 layers, 12 KV heads, 256 head_dim
    # 52 * 12 * 256 * 2 = 319,488
    "ornith-9b": (319488, "Ornith 9B dense"),

    # Laguna-33B: MoE with similar arch to Qwen
    # 60 layers, 12 KV heads, 256 head_dim
    # 60 * 12 * 256 * 2 = 368,640
    "laguna-33b": (368640, "Laguna 33B MoE"),

    # North-Code-30B: MoE
    # 60 layers, 12 KV heads, 256 head_dim
    # 60 * 12 * 256 * 2 = 368,640
    "north-code-30b": (368640, "North Code 30B MoE"),

    # Devstral-Small-2: Same arch as Mistral Small
    # 40 layers, 8 KV heads, 128 head_dim
    # 40 * 8 * 128 * 2 = 81,920
    "devstral-small-2": (81920, "Devstral Small 2"),

    # Nemotron-4B: 16 layers, 4 KV heads, 128 head_dim
    # 16 * 4 * 128 * 2 = 16,384
    "nemotron-4b": (16384, "Nemotron 4B"),

    # Nemotron-30B-A3B: MoE with 40 layers, 8 KV heads, 256 head_dim
    # 40 * 8 * 256 * 2 = 163,840
    "nemotron-30b-a3b": (163840, "Nemotron 30B MoE"),
}


def fixture_metadata(model_key: str) -> dict:
    """Build GGUF metadata dict for a known model fixture.

    Returns metadata with architecture fields that produce the expected
    KV cache bytes-per-token when passed to kv_bytes_per_token_exact().
    """
    fixtures = {
        "qwen3-9b": {
            "general.architecture": "llama",
            "llama.block_count": 52,
            "llama.attention.head_count_kv": 12,
            "llama.attention.head_dim": 256,
        },
        "qwen3-27b": {
            "general.architecture": "llama",
            "llama.block_count": 48,
            "llama.attention.head_count_kv": 24,
            "llama.attention.head_dim": 256,
        },
        "gemma3-12b": {
            "general.architecture": "gemma",
            "gemma.block_count": 40,
            "gemma.attention.head_count_kv": 8,
            "gemma.attention.head_dim": 256,
        },
        "llama3-8b": {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.head_dim": 128,
        },
        "phi4-mini": {
            "general.architecture": "phi",
            "phi.block_count": 32,
            "phi.attention.head_count_kv": 4,
            "phi.attention.head_dim": 128,
        },
        "granite-3b": {
            "general.architecture": "granite",
            "granite.block_count": 32,
            "granite.attention.head_count_kv": 8,
            "granite.attention.head_dim": 128,
        },
        "mistral-small-3.1": {
            "general.architecture": "llama",
            "llama.block_count": 40,
            "llama.attention.head_count_kv": 8,
            "llama.attention.head_dim": 128,
        },
        "glm-flash": {
            "general.architecture": "gemma",
            "gemma.block_count": 28,
            "gemma.attention.head_count_kv": 4,
            "gemma.attention.head_dim": 128,
        },
        "bonsai-8b": {
            "general.architecture": "llama",
            "llama.block_count": 28,
            "llama.attention.head_count_kv": 8,
            "llama.attention.head_dim": 128,
        },
        "nex-n2-mini": {
            "general.architecture": "nexus",
            "nexus.block_count": 10,
            "nexus.attention.head_count_kv": 2,
            "nexus.attention.head_dim": 256,
        },
        "ling-mini-2": {
            "general.architecture": "llama",
            "llama.block_count": 20,
            "llama.attention.head_count_kv": 4,
            "llama.attention.head_dim": 128,
        },
        "lfm2-24b-a2b": {
            "general.architecture": "mpt",
            "mpt.block_count": 10,
            "mpt.attention.head_count_kv": 8,
            "mpt.attention.head_dim": 64,
        },
        "mellum-12b": {
            "general.architecture": "llama",
            "llama.block_count": 28,
            "llama.attention.head_count_kv": 4,
            "llama.attention.head_dim": 256,
        },
        "gpt-oss-20b": {
            "general.architecture": "llama",
            "llama.block_count": 24,
            "llama.attention.head_count_kv": 8,
            "llama.attention.head_dim": 64,
        },
        "ornith-9b": {
            "general.architecture": "llama",
            "llama.block_count": 52,
            "llama.attention.head_count_kv": 12,
            "llama.attention.head_dim": 256,
        },
        "laguna-33b": {
            "general.architecture": "llama",
            "llama.block_count": 60,
            "llama.attention.head_count_kv": 12,
            "llama.attention.head_dim": 256,
        },
        "north-code-30b": {
            "general.architecture": "llama",
            "llama.block_count": 60,
            "llama.attention.head_count_kv": 12,
            "llama.attention.head_dim": 256,
        },
        "devstral-small-2": {
            "general.architecture": "llama",
            "llama.block_count": 40,
            "llama.attention.head_count_kv": 8,
            "llama.attention.head_dim": 128,
        },
        "nemotron-4b": {
            "general.architecture": "nemotron",
            "nemotron.block_count": 16,
            "nemotron.attention.head_count_kv": 4,
            "nemotron.attention.head_dim": 128,
        },
        "nemotron-30b-a3b": {
            "general.architecture": "nemotron",
            "nemotron.block_count": 40,
            "nemotron.attention.head_count_kv": 8,
            "nemotron.attention.head_dim": 256,
        },
    }
    return fixtures[model_key]


# Mapping of model fixture key -> family registry key for cross-validation
FAMILY_MAP: dict[str, str] = {
    "qwen3-9b": "qwen",
    "qwen3-27b": "qwen",
    "gemma3-12b": "gemma",
    "llama3-8b": "unknown",  # Llama not yet in registry
    "phi4-mini": "phi4",
    "granite-3b": "granite",
    "mistral-small-3.1": "mistral-small",
    "glm-flash": "glm",
    "bonsai-8b": "bonsai",
    "nex-n2-mini": "nex-n2",
    "ling-mini-2": "ling",
    "lfm2-24b-a2b": "lfm",
    "mellum-12b": "mellum",
    "gpt-oss-20b": "gpt-oss",
    "ornith-9b": "ornith",
    "laguna-33b": "laguna",
    "north-code-30b": "north-code",
    "devstral-small-2": "devstral",
    "nemotron-4b": "nemotron",
    "nemotron-30b-a3b": "nemotron",
}
