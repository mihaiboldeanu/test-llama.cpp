"""GGUF metadata reader — pure domain adapter.

Reads architecture fields from GGUF binary headers without external dependencies.
No side effects: no subprocess, network, or file system writes.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# GGUF value types (https://github.com/ggerganov/gguf/pull/30)
_GGUF_TYPE = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "DOUBLE",
}

_GGUF_MAGIC = b"GGUF"
_GGUF_HEADER_FORMAT = "<IQQ"  # version, tensor count, KV count
_GGUF_HEADER_SIZE = struct.calcsize(_GGUF_HEADER_FORMAT)
_GGUF_FIXED_VALUE_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


def read_gguf_metadata(path: Path) -> dict[str, Any] | None:
    """Read KV metadata from a GGUF file header.

    Returns a dict of key-value pairs, or None if the file is not a valid GGUF.
    Only reads the header + KV section (skips tensor data).
    """
    try:
        with open(path, "rb") as f:
            # Read magic number
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return None

            # GGUF uses a uint32 version followed by uint64 tensor and KV
            # counts.  Reading the version as uint64 shifts the following
            # fields by four bytes and can turn a small KV count into a huge
            # string length (and consequently a MemoryError).
            _, _, num_kv = struct.unpack(
                _GGUF_HEADER_FORMAT, _read_exact(f, _GGUF_HEADER_SIZE)
            )

            metadata: dict[str, Any] = {}

            for _ in range(num_kv):
                key = _read_string(f)
                vtype_id = struct.unpack("<I", _read_exact(f, 4))[0]
                # The tokenizer section commonly contains hundreds of
                # thousands of strings.  Architecture keys are written before
                # it; if exact KV sizing is unavailable by this point, retain
                # what we found and use Rama's existing size heuristic.
                if key.startswith("tokenizer."):
                    return metadata
                if _is_architecture_field(key):
                    metadata[key] = _read_value(f, vtype_id)
                    # Standard GGUF writers place the small architecture
                    # section before large tokenizer arrays.  Stop as soon as
                    # Rama has the fields it needs; parsing the remainder is
                    # both unnecessary and noticeably slows model discovery.
                    if kv_bytes_per_token_exact(metadata) is not None:
                        return metadata
                else:
                    _skip_value(f, vtype_id)

            return metadata
    except (MemoryError, OSError, struct.error, UnicodeDecodeError, ValueError):
        return None


def _read_string(f: Any) -> str:
    """Read a GGUF string (uint64 length + UTF-8 bytes)."""
    length = struct.unpack("<Q", _read_exact(f, 8))[0]
    return _read_exact(f, length).decode("utf-8")


def _read_exact(f: Any, length: int) -> bytes:
    """Read exactly *length* bytes without accepting a truncated file."""
    if length < 0:
        raise ValueError("negative GGUF field length")

    remaining = _remaining_bytes(f)
    if remaining is not None and length > remaining:
        raise ValueError("GGUF field exceeds remaining file size")

    data = f.read(length)
    if len(data) != length:
        raise ValueError("truncated GGUF field")
    return data


def _skip_exact(f: Any, length: int) -> None:
    """Advance over a validated field without allocating its contents."""
    if length < 0:
        raise ValueError("negative GGUF field length")

    remaining = _remaining_bytes(f)
    if remaining is not None and length > remaining:
        raise ValueError("GGUF field exceeds remaining file size")

    f.seek(length, 1)


def _is_architecture_field(key: str) -> bool:
    """Whether a metadata key is needed for model classification or KV sizing."""
    return key == "general.architecture" or key.endswith(
        (
            ".block_count",
            ".attention.head_count",
            ".attention.head_count_kv",
            ".attention.key_head_count",
            ".attention.head_dim",
            ".embedding_length",
            ".expert_count",
            ".expert_used_count",
        )
    )


def _remaining_bytes(f: Any) -> int | None:
    """Return bytes remaining in a seekable binary stream, when available."""
    try:
        position = f.tell()
        f.seek(0, 2)
        end = f.tell()
        f.seek(position)
        return end - position
    except (AttributeError, OSError):
        return None


def _read_value(f: Any, vtype_id: int) -> Any:
    """Read a GGUF value based on its type."""
    if vtype_id == 7:  # BOOL
        return struct.unpack("<?", _read_exact(f, 1))[0]
    if vtype_id == 0:  # UINT8
        return struct.unpack("<B", _read_exact(f, 1))[0]
    if vtype_id == 1:  # INT8
        return struct.unpack("<b", _read_exact(f, 1))[0]
    if vtype_id == 2:  # UINT16
        return struct.unpack("<H", _read_exact(f, 2))[0]
    if vtype_id == 3:  # INT16
        return struct.unpack("<h", _read_exact(f, 2))[0]
    if vtype_id == 4:  # UINT32
        return struct.unpack("<I", _read_exact(f, 4))[0]
    if vtype_id == 5:  # INT32
        return struct.unpack("<i", _read_exact(f, 4))[0]
    if vtype_id == 6:  # FLOAT32
        return struct.unpack("<f", _read_exact(f, 4))[0]
    if vtype_id == 10:  # UINT64
        return struct.unpack("<Q", _read_exact(f, 8))[0]
    if vtype_id == 11:  # INT64
        return struct.unpack("<q", _read_exact(f, 8))[0]
    if vtype_id == 12:  # DOUBLE
        return struct.unpack("<d", _read_exact(f, 8))[0]
    if vtype_id == 8:  # STRING
        return _read_string(f)
    if vtype_id == 9:  # ARRAY
        return _read_array(f)
    # Unknown type — skip (shouldn't happen with valid GGUF)
    return None


def _read_array(f: Any) -> list[Any]:
    """Read a GGUF array (type uint32 + element count uint64 + elements)."""
    elem_type = struct.unpack("<I", _read_exact(f, 4))[0]
    elem_count = struct.unpack("<Q", _read_exact(f, 8))[0]
    return [_read_value(f, elem_type) for _ in range(elem_count)]


def _skip_value(f: Any, vtype_id: int) -> None:
    """Skip a GGUF value that Rama does not need, without loading it into RAM."""
    fixed_size = _GGUF_FIXED_VALUE_SIZES.get(vtype_id)
    if fixed_size is not None:
        _skip_exact(f, fixed_size)
        return
    if vtype_id == 8:  # STRING
        _skip_exact(f, struct.unpack("<Q", _read_exact(f, 8))[0])
        return
    if vtype_id == 9:  # ARRAY
        elem_type = struct.unpack("<I", _read_exact(f, 4))[0]
        elem_count = struct.unpack("<Q", _read_exact(f, 8))[0]
        fixed_size = _GGUF_FIXED_VALUE_SIZES.get(elem_type)
        if fixed_size is not None:
            _skip_exact(f, fixed_size * elem_count)
            return
        if elem_type == 8:  # array of strings (for example tokenizer.tokens)
            for _ in range(elem_count):
                _skip_exact(f, struct.unpack("<Q", _read_exact(f, 8))[0])
            return
    raise ValueError(f"unsupported GGUF value type: {vtype_id}")


# ---------------------------------------------------------------------------
# Architecture field accessors
# ---------------------------------------------------------------------------


def get_architecture(metadata: dict[str, Any]) -> str | None:
    """Get the architecture name (e.g., 'llama', 'mpt', 'gemma')."""
    return metadata.get("general.architecture")


def get_block_count(metadata: dict[str, Any], arch: str | None = None) -> int | None:
    """Get number of transformer blocks/layers."""
    if arch is None:
        arch = get_architecture(metadata)
    if arch:
        return metadata.get(f"{arch}.block_count")
    # Fallback: search common architectures
    for key, val in metadata.items():
        if key.endswith(".block_count"):
            return val
    return None


def get_head_count(metadata: dict[str, Any], arch: str | None = None) -> int | None:
    """Get number of attention heads (query heads)."""
    if arch is None:
        arch = get_architecture(metadata)
    if arch:
        return metadata.get(f"{arch}.attention.head_count")
    return None


def get_head_count_kv(metadata: dict[str, Any], arch: str | None = None) -> int | None:
    """Get number of KV heads (for GQA/MQA)."""
    if arch is None:
        arch = get_architecture(metadata)
    if arch:
        # Some models use key_head_count instead of head_count_kv
        val = metadata.get(f"{arch}.attention.head_count_kv")
        if val is not None:
            return val
        val = metadata.get(f"{arch}.attention.key_head_count")
        if val is not None:
            return val
    return None


def get_head_dim(metadata: dict[str, Any], arch: str | None = None) -> int | None:
    """Get attention head dimension."""
    if arch is None:
        arch = get_architecture(metadata)
    if arch:
        val = metadata.get(f"{arch}.attention.head_dim")
        if val is not None:
            return val
        # Some older GGUFs use embedding_length / head_count as proxy
        emb_len = metadata.get(f"{arch}.embedding_length")
        if emb_len is not None:
            heads = get_head_count(metadata, arch)
            if type(emb_len) is int and type(heads) is int and heads > 0:
                return int(emb_len / heads)
    return None


def get_expert_count(metadata: dict[str, Any], arch: str | None = None) -> int | None:
    """Get total number of experts (MoE)."""
    if arch is None:
        arch = get_architecture(metadata)
    if arch:
        return metadata.get(f"{arch}.expert_count")
    return None





def is_moe_from_metadata(metadata: dict[str, Any]) -> bool:
    """Detect MoE architecture from GGUF metadata."""
    arch = get_architecture(metadata)
    expert_count = get_expert_count(metadata, arch)
    return expert_count is not None and expert_count > 1


def kv_bytes_per_token_exact(
    metadata: dict[str, Any],
    ctk: str | None = None,
    ctv: str | None = None,
) -> int | None:
    """Calculate exact KV cache bytes per token from GGUF metadata.

    Returns None if required fields are missing (should fall back to heuristics).

    Formula: n_layers * n_kv_heads * head_dim * 2 * dtype_bytes
    The *2 is for both K and V caches.
    """
    arch = get_architecture(metadata)
    n_layers = get_block_count(metadata, arch)
    n_kv_heads = get_head_count_kv(metadata, arch)
    head_dim = get_head_dim(metadata, arch)

    if not all(type(value) is int and value > 0 for value in (n_layers, n_kv_heads, head_dim)):
        return None

    # Determine dtype bytes from KV cache type
    dtype_bytes = _kv_dtype_bytes(ctk)  # Key type determines K cache dtype

    return int(n_layers * n_kv_heads * head_dim * 2 * dtype_bytes)


def _kv_dtype_bytes(kv_type: str | None) -> float:
    """Get bytes per element for a KV cache type.

    llama.cpp uses the same type for both K and V by default.
    Some backends support mixed precision (e.g., f16 keys, q8 values).
    """
    if kv_type is None:
        return 1.0  # Default: q8_0 (1 byte per element)

    kv_lower = kv_type.lower()

    # Standard llama.cpp types
    if kv_lower in ("f16", "fp16"):
        return 2.0
    if kv_lower in ("f32", "fp32"):
        return 4.0
    if kv_lower in ("q8_0", "q8"):
        return 1.0
    if kv_lower in ("q4_k_m", "q4"):
        return 0.5  # 4 bits per element
    if kv_lower in ("iq4_xs", "iq4_nl"):
        return 0.5

    # Turboquant types
    if "turbo" in kv_lower:
        return 1.0  # Conservative estimate for turboquant

    return 1.0  # Default to q8_0


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def kv_provenance_string(
    metadata: dict[str, Any] | None,
    family_key: str,
    size_gb: float,
) -> str:
    """Generate a provenance string explaining KV cache calculation source.

    Examples:
        "GGUF metadata (exact): 16 layers x 4 KV heads x 128 head_dim = 16384 B/token"
        "size heuristic (qwen, 15GB): 22000 B/token"
    """
    if metadata is not None:
        exact = kv_bytes_per_token_exact(metadata)
        if exact is not None:
            arch = get_architecture(metadata) or "unknown"
            n_layers = get_block_count(metadata, arch)
            n_kv_heads = get_head_count_kv(metadata, arch)
            head_dim = get_head_dim(metadata, arch)
            parts = []
            if n_layers is not None:
                parts.append(f"{n_layers} layers")
            if n_kv_heads is not None:
                parts.append(f"{n_kv_heads} KV heads")
            if head_dim is not None:
                parts.append(f"{head_dim} head_dim")
            detail = " x ".join(parts) if parts else "GGUF"
            return f"GGUF metadata (exact): {detail} = {exact} B/token"

    return f"size heuristic ({family_key}, {size_gb:.1f}GB)"
