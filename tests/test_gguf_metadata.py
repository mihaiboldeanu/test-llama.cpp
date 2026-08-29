"""Tests for GGUF metadata reading and metadata-backed KV estimation."""
from __future__ import annotations

import struct
from pathlib import Path

from rama.domain.gguf import (
    get_architecture,
    get_block_count,
    get_head_count,
    get_head_count_kv,
    get_head_dim,
    is_moe_from_metadata,
    kv_bytes_per_token_exact,
    read_gguf_metadata,
)
from rama.domain.models import ModelInfo
from rama.families import _QWEN_KV, _MetadataKvEstimator
from tests.fixtures.model_architectures import (
    FAMILY_MAP,
    KNOWN_MODELS,
    fixture_metadata,
)


def _build_gguf(tmp_path: Path, kv_pairs: dict[str, object]) -> Path:
    """Build a minimal GGUF file with given KV metadata (no tensors).

    Simplified format matching the reader's expectations:
    - Key as GGUF string (uint64 length + UTF-8 bytes)
    - Value type (uint32) + value data
    """
    buf = bytearray()
    # Header
    buf += b"GGUF"
    buf += struct.pack("<I", 3)  # version
    buf += struct.pack("<Q", 0)  # num_tensors
    buf += struct.pack("<Q", len(kv_pairs))  # num_kv

    for key, value in kv_pairs.items():
        # Key as GGUF string (uint64 length + UTF-8 bytes)
        key_bytes = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_bytes))
        buf += key_bytes
        # Value type (uint32) + data
        if isinstance(value, str):
            buf += struct.pack("<I", 8)  # STRING type
            val_bytes = value.encode("utf-8")
            buf += struct.pack("<Q", len(val_bytes))
            buf += val_bytes
        elif isinstance(value, bool):
            buf += struct.pack("<I", 7)  # BOOL type
            buf += struct.pack("<?", value)
        elif isinstance(value, int):
            buf += struct.pack("<I", 10)  # UINT64 type
            buf += struct.pack("<Q", value)
        elif isinstance(value, float):
            buf += struct.pack("<I", 6)  # FLOAT32 type
            buf += struct.pack("<f", value)

    gguf = tmp_path / "test.gguf"
    gguf.write_bytes(bytes(buf))
    return gguf


class TestReadGgufMetadata:
    def test_valid_file(self, tmp_path: Path) -> None:
        path = _build_gguf(
            tmp_path,
            {
                "general.architecture": "llama",
                "llama.block_count": 32,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.attention.head_dim": 128,
            },
        )
        meta = read_gguf_metadata(path)
        assert meta is not None
        assert meta["general.architecture"] == "llama"
        assert meta["llama.block_count"] == 32
        assert meta["llama.attention.head_count"] == 32
        assert meta["llama.attention.head_count_kv"] == 8
        assert meta["llama.attention.head_dim"] == 128

    def test_invalid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"NOTGGUF" + b"\x00" * 100)
        assert read_gguf_metadata(path) is None

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.gguf"
        path.write_bytes(b"")
        assert read_gguf_metadata(path) is None

    def test_rejects_string_length_beyond_file_size(self, tmp_path: Path) -> None:
        """A corrupt length field must not allocate memory or crash discovery."""
        path = tmp_path / "oversized-string.gguf"
        path.write_bytes(
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 1)
            + struct.pack("<Q", 2**63)
        )
        assert read_gguf_metadata(path) is None

    def test_skips_unneeded_large_metadata(self, tmp_path: Path) -> None:
        """Tokenizer-like metadata is skipped instead of being retained in RAM."""
        path = _build_gguf(
            tmp_path,
            {
                "general.architecture": "llama",
                "llama.block_count": 32,
                "general.description": "x" * 1024,
            },
        )
        assert read_gguf_metadata(path) == {
            "general.architecture": "llama",
            "llama.block_count": 32,
        }

    def test_stops_after_collecting_kv_architecture_fields(self, tmp_path: Path) -> None:
        """Discovery does not scan trailing tokenizer-scale metadata."""
        path = _build_gguf(
            tmp_path,
            {
                "general.architecture": "llama",
                "llama.block_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.attention.head_dim": 128,
                "general.description": "x" * 1024,
            },
        )
        assert read_gguf_metadata(path) == {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.head_dim": 128,
        }

    def test_stops_at_tokenizer_section_when_exact_fields_are_unavailable(
        self, tmp_path: Path
    ) -> None:
        path = _build_gguf(
            tmp_path,
            {
                "general.architecture": "llama",
                "llama.block_count": 32,
                "tokenizer.description": "x" * 1024,
            },
        )
        assert read_gguf_metadata(path) == {
            "general.architecture": "llama",
            "llama.block_count": 32,
        }

    def test_array_head_count_falls_back_cleanly(self) -> None:
        meta = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.attention.head_count": [32] * 32,
            "llama.embedding_length": 4096,
        }
        assert kv_bytes_per_token_exact(meta) is None


class TestArchAccessors:
    def _meta(self, **overrides: object) -> dict:
        base = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.head_dim": 128,
        }
        base.update(overrides)
        return base

    def test_get_architecture(self) -> None:
        assert get_architecture(self._meta()) == "llama"

    def test_get_block_count(self) -> None:
        assert get_block_count(self._meta()) == 32

    def test_get_head_count(self) -> None:
        assert get_head_count(self._meta()) == 32

    def test_get_head_count_kv(self) -> None:
        assert get_head_count_kv(self._meta()) == 8

    def test_get_head_dim(self) -> None:
        assert get_head_dim(self._meta()) == 128

    def test_is_moe_false(self) -> None:
        assert not is_moe_from_metadata(self._meta())

    def test_is_moe_true(self) -> None:
        meta = dict(self._meta())
        meta["llama.expert_count"] = 64
        meta["llama.expert_used_count"] = 8
        assert is_moe_from_metadata(meta)


class TestKvBytesExact:
    def _meta(self, **overrides: object) -> dict:
        base = {
            "general.architecture": "llama",
            "llama.block_count": 16,
            "llama.attention.head_count_kv": 4,
            "llama.attention.head_dim": 128,
        }
        base.update(overrides)
        return base

    def test_q8_0(self) -> None:
        # 16 layers * 4 KV heads * 128 head_dim * 2 (K+V) * 1 byte = 16384
        result = kv_bytes_per_token_exact(self._meta(), ctk="q8_0", ctv="q8_0")
        assert result == 16384

    def test_f16(self) -> None:
        # 16 * 4 * 128 * 2 * 2 bytes = 32768
        result = kv_bytes_per_token_exact(self._meta(), ctk="f16", ctv="f16")
        assert result == 32768

    def test_q4_k_m(self) -> None:
        # 16 * 4 * 128 * 2 * 0.5 bytes = 8192
        result = kv_bytes_per_token_exact(self._meta(), ctk="q4_k_m", ctv="q4_k_m")
        assert result == 8192

    def test_none_default(self) -> None:
        # Default to q8_0 (1 byte)
        result = kv_bytes_per_token_exact(self._meta())
        assert result == 16384

    def test_missing_fields_returns_none(self) -> None:
        meta = {"general.architecture": "llama"}
        assert kv_bytes_per_token_exact(meta) is None


class TestMetadataKvEstimator:
    def _model(self, metadata: dict | None = None, size_gb: float = 10.0) -> ModelInfo:
        return ModelInfo(
            name="test",
            path=Path("/fake/test.gguf"),
            size_bytes=int(size_gb * 1024**3),
            family_key="qwen",
            metadata=metadata,
        )

    def test_uses_metadata_when_available(self) -> None:
        """Exact calculation from GGUF metadata."""
        meta = {
            "general.architecture": "llama",
            "llama.block_count": 28,
            "llama.attention.head_count_kv": 4,
            "llama.attention.head_dim": 128,
        }
        est = _MetadataKvEstimator(_QWEN_KV)
        # 28 * 4 * 128 * 2 * 1 = 28672
        result = est.bytes_per_token(self._model(metadata=meta), None, None)
        assert result == 28672

    def test_falls_back_to_heuristic(self) -> None:
        """Size-based heuristic when metadata is absent."""
        est = _MetadataKvEstimator(_QWEN_KV)
        # Qwen estimator for <10GB returns 11000 (9B tier)
        result = est.bytes_per_token(self._model(metadata=None, size_gb=5.0), None, None)
        assert result == 11000

    def test_falls_back_when_incomplete(self) -> None:
        """Size-based heuristic when metadata is incomplete."""
        meta = {"general.architecture": "llama"}  # Missing block_count etc.
        est = _MetadataKvEstimator(_QWEN_KV)
        result = est.bytes_per_token(
            self._model(metadata=meta, size_gb=15.0), None, None
        )
        # Qwen estimator for <20GB returns 22000 (27B tier)
        assert result == 22000


class TestProvenance:
    def test_provenance_exact(self) -> None:
        from rama.domain.gguf import kv_provenance_string

        meta = {
            "general.architecture": "llama",
            "llama.block_count": 16,
            "llama.attention.head_count_kv": 4,
            "llama.attention.head_dim": 128,
        }
        prov = kv_provenance_string(meta, "llama", 5.0)
        assert "GGUF metadata (exact)" in prov
        assert "16 layers" in prov
        assert "4 KV heads" in prov
        assert "128 head_dim" in prov

    def test_provenance_heuristic(self) -> None:
        from rama.domain.gguf import kv_provenance_string

        prov = kv_provenance_string(None, "qwen", 15.0)
        assert "size heuristic" in prov
        assert "qwen" in prov
        assert "15.0GB" in prov

    def test_provenance_incomplete_metadata(self) -> None:
        from rama.domain.gguf import kv_provenance_string

        meta = {"general.architecture": "llama"}  # Missing required fields
        prov = kv_provenance_string(meta, "gemma", 8.5)
        assert "size heuristic" in prov
        assert "gemma" in prov


class TestKnownModelFixtures:
    """Verify exact KV calculation matches expected values for real models."""

    def test_all_known_models(self) -> None:
        """Each known model fixture produces the expected KV bytes-per-token."""
        for model_key, (expected, description) in KNOWN_MODELS.items():
            meta = fixture_metadata(model_key)
            result = kv_bytes_per_token_exact(meta, ctk="q8_0", ctv="q8_0")
            assert result == expected, (
                f"{model_key} ({description}): expected {expected}, got {result}"
            )

    def test_fixture_metadata_is_readable(self) -> None:
        """Each fixture has required GGUF fields."""
        for model_key in KNOWN_MODELS:
            meta = fixture_metadata(model_key)
            arch = get_architecture(meta)
            assert arch is not None, f"{model_key}: missing general.architecture"
            assert get_block_count(meta, arch) is not None, (
                f"{model_key}: missing block_count"
            )
            assert get_head_count_kv(meta, arch) is not None, (
                f"{model_key}: missing head_count_kv"
            )
            assert get_head_dim(meta, arch) is not None, (
                f"{model_key}: missing head_dim"
            )

    def test_metadata_beats_heuristic_for_known_family(self) -> None:
        """Metadata-based estimator prefers exact over size heuristic."""
        # Qwen3-9B fixture metadata
        meta = fixture_metadata("qwen3-9b")
        model = ModelInfo(
            name="qwen3-9b",
            path=Path("/fake/qwen3.gguf"),
            size_bytes=int(6.0 * 1024**3),  # Q4 quant ~6GB
            family_key="qwen",
            metadata=meta,
        )
        est = _MetadataKvEstimator(_QWEN_KV)
        result = est.bytes_per_token(model, None, None)
        # Without metadata, size heuristic would return 11000 (9B tier)
        # With metadata, exact calculation returns 327680
        assert result == KNOWN_MODELS["qwen3-9b"][0]
