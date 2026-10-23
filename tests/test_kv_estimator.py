"""KV estimator fixes: MoE weight charge + hybrid SWA amortization."""

import tempfile

import yaml

from rama.__init__ import Config
from rama.core import calc_kv_cache
from rama.families import get_registry, hybrid_kv_ctx


def _make_cfg(data: dict) -> Config:
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as f:
        yaml.dump(data, f)
        f.flush()
        return Config(f.name)


def _cfg() -> Config:
    # vram_total_gb != 24 skips the nvidia-smi auto-detect (keeps tests
    # hermetic); available = 23 - 2 = 21 GiB.
    return _make_cfg({
        "vram_total_gb": 23,
        "vram_headroom_gb": 2,
        "rules": {
            "small_ctx": 32768,
            "large_ctx": 65536,
            "small_threshold_gb": 10,
        },
        "families": {
            "qwen3.8": {"large_ctx": 171552},
            "qwen3.8-flash-next": {"large_ctx": 170240},
            "qwen": {"large_ctx": 151552},
            "nemotron": {"large_ctx": 65536},
        },
    })


def _meta(layers: int = 64, kv_heads: int = 4, head_dim: int = 128) -> dict:
    return {
        "general.architecture": "llama",
        "llama.block_count": layers,
        "llama.attention.head_count_kv": kv_heads,
        "llama.attention.head_dim": head_dim,
    }


class TestHybridRegistry:
    def test_hybrid_families_registered(self) -> None:
        reg = get_registry()
        assert reg.get("qwen3.8").hybrid == (16384, 0.25)
        assert reg.get("qwen3.8-flash-next").hybrid == (16384, 0.25)
        assert reg.get("spark").hybrid == (512, 0.25)
        assert reg.get("flint").hybrid == (513, 14 / 49)

    def test_dense_families_not_hybrid(self) -> None:
        reg = get_registry()
        assert reg.get("qwen").hybrid is None
        assert reg.get("nemotron").hybrid is None


class TestHybridSolver:
    def test_unknown_family_returns_none(self) -> None:
        assert hybrid_kv_ctx(10**9, "nope", _meta()) is None

    def test_missing_metadata_returns_none(self) -> None:
        assert hybrid_kv_ctx(10**9, "qwen3.8", None) is None

    def test_incomplete_metadata_returns_none(self) -> None:
        assert hybrid_kv_ctx(10**9, "qwen3.8", {"general.architecture": "llama"}) is None

    def test_large_budget_uses_windowed_branch(self) -> None:
        # 64 layers * 4 kv * 128 dim * 2 = 65536 B/token, 1024/layer.
        # qwen3.8 hybrid: 16 full + 48 SWA @ 16K window.
        # budget 8 GiB -> (8388608 - 48*16384) / 16 = 475136.
        ctx = hybrid_kv_ctx(8 * 1024**3, "qwen3.8", _meta())
        assert ctx == 475136

    def test_tiny_budget_stays_below_window(self) -> None:
        # budget 0.1 GiB -> 104857 layer-tokens / 64 layers = 1638 < window.
        ctx = hybrid_kv_ctx(int(0.1 * 1024**3), "qwen3.8", _meta())
        assert ctx == 1638

    def test_single_layer_degrades_to_dense(self) -> None:
        meta = _meta(layers=1)
        # exact = 1*4*128*2 = 1024; budget 1 GiB -> 1048576 tokens.
        assert hybrid_kv_ctx(1024**3, "qwen3.8", meta) == 1048576

    def test_flint_uses_14_full_attention_layers(self) -> None:
        meta = {
            "general.architecture": "flint",
            "flint.block_count": 49,
            "flint.attention.head_count_kv": 4,
            "flint.attention.head_dim": 128,
        }
        # 50,176 B/token dense -> 1024 B/token/layer.
        # 8 GiB: (8388608 - 35*513) / 14 = 597903 tokens.
        assert hybrid_kv_ctx(8 * 1024**3, "flint", meta) == 597903


class TestCalcKvCache:
    def test_hybrid_beats_dense_estimate(self) -> None:
        cfg = _cfg()
        hybrid_ctx, _, _ = calc_kv_cache(14.0, "qwen3.8", cfg, metadata=_meta())
        dense_ctx, _, _ = calc_kv_cache(14.0, "qwen", cfg, metadata=_meta())
        assert hybrid_ctx == 171552
        # 7 GiB budget / 65536 B/token, under the 151552 preferred cap.
        assert dense_ctx == 114688
        assert hybrid_ctx > dense_ctx

    def test_moe_host_experts_not_floor_pinned(self) -> None:
        cfg = _cfg()
        meta = _meta()
        offloaded, _, _ = calc_kv_cache(
            90.8, "nemotron", cfg, metadata=meta,
            extra_args={"--moe-cache-mib": 6024},
        )
        pinned, _, _ = calc_kv_cache(90.8, "nemotron", cfg, metadata=meta)
        assert offloaded == 65536
        assert pinned == 2048

    def test_small_dense_unchanged(self) -> None:
        cfg = _cfg()
        ctx, qk, qv = calc_kv_cache(4.0, "qwen", cfg, metadata=_meta())
        assert (qk, qv) == ("q8_0", "q8_0")
        # 4 GB is below the 10 GB rule, so the small_ctx preferred cap wins.
        assert ctx == 32768

    def test_negative_size_rejected(self) -> None:
        cfg = _cfg()
        try:
            calc_kv_cache(-1.0, "qwen", cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
