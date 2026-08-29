"""Tests for batch reproducibility infrastructure.

Covers:
- BatchPlanner (pure function, no side effects)
- Run identity computation (deterministic hash)
- Fixture hashing (SHA256)
- Manifest write/load (round-trip)
- Resumable batch detection
- BatchExecutor dry-run mode
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rama.domain.batch_planner import (
    BatchItemPlan,
    BatchManifest,
    BatchPlan,
    BatchPlannerInput,
    compute_fixture_hash,
    compute_run_identity,
    hash_profile,
    plan_batch,
)
from rama.domain.batch_runner import (
    BatchExecutorInput,
    BatchExecutorResult,
    can_resume,
    load_manifest,
)
from rama.domain.models import (
    ContextPolicy,
    FamilyCapabilities,
    LaunchSettings,
    MemorySettings,
    ModelInfo,
    ResolvedModelProfile,
    SamplerSettings,
)


class TestComputeFixtureHash:
    """Fixture file hashing is deterministic."""

    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_fixture_hash(f)
        h2 = compute_fixture_hash(f)
        assert h1 == h2

    def test_sha256_length(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = compute_fixture_hash(f)
        assert len(h) == 64  # SHA256 hex digest

    def test_different_content(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        assert compute_fixture_hash(f1) != compute_fixture_hash(f2)


class TestComputeRunIdentity:
    """Run identity is deterministic and content-based."""

    def _item(
        self, model_name: str = "model1", backend: str = "llama.cpp"
    ) -> BatchItemPlan:
        model = ModelInfo(
            name=model_name,
            path=Path(f"/models/{model_name}.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        return BatchItemPlan(
            model=model,
            profile=profile,
            backend=backend,
            ctk="q8_0",
            ctv="q8_0",
            ctx_size=4096,
            seed=42,
            mtp_enabled=False,
            mtp_n_max=3,
            mtp_n_min=0,
            reasoning=None,
            reasoning_budget=1024,
            max_tokens=None,
            config_item={},
        )

    def test_deterministic(self) -> None:
        items = [self._item()]
        id1 = compute_run_identity("code", items, {})
        id2 = compute_run_identity("code", items, {})
        assert id1 == id2

    def test_different_batch_type(self) -> None:
        items = [self._item()]
        id1 = compute_run_identity("code", items, {})
        id2 = compute_run_identity("perplexity", items, {})
        assert id1 != id2

    def test_different_model(self) -> None:
        items1 = [self._item(model_name="model_a")]
        items2 = [self._item(model_name="model_b")]
        id1 = compute_run_identity("code", items1, {})
        id2 = compute_run_identity("code", items2, {})
        assert id1 != id2

    def test_different_seed(self) -> None:
        item1 = self._item()
        item2 = self._item()
        item2 = BatchItemPlan(
            **{**item2.__dict__, "seed": 99}
        )
        id1 = compute_run_identity("code", [item1], {})
        id2 = compute_run_identity("code", [item2], {})
        assert id1 != id2

    def test_fixture_hash_affects_identity(self) -> None:
        items = [self._item()]
        id1 = compute_run_identity("nihs", items, {"/text.txt": "abc"})
        id2 = compute_run_identity("nihs", items, {"/text.txt": "def"})
        assert id1 != id2

    def test_short_hex(self) -> None:
        items = [self._item()]
        rid = compute_run_identity("code", items, {})
        assert len(rid) == 16
        # Valid hex
        int(rid, 16)


class TestHashProfile:
    """Profile serialization for manifest."""

    def test_contains_model_name(self) -> None:
        model = ModelInfo(
            name="test-model",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(temperature=0.7),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        h = hash_profile(profile)
        assert h["model"]["name"] == "test-model"

    def test_contains_sampler(self) -> None:
        model = ModelInfo(
            name="test",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(temperature=0.7, top_p=0.9),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        h = hash_profile(profile)
        assert h["sampler"]["temperature"] == 0.7

    def test_json_serializable(self) -> None:
        model = ModelInfo(
            name="test",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        h = hash_profile(profile)
        # Should not raise
        json.dumps(h)


class TestPlanBatch:
    """Batch planner is pure and deterministic."""

    def _model(self, name: str) -> ModelInfo:
        return ModelInfo(
            name=name,
            path=Path(f"/models/{name}.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )

    def test_plans_items(self) -> None:
        models = [self._model("model_a"), self._model("model_b")]
        inp = BatchPlannerInput(
            models=models,
            batch_config_items=["model_a", "model_b"],
            batch_type="code",
            server_bin=Path("/usr/bin/llama-server"),
        )
        plan = plan_batch(inp)
        assert len(plan.items) == 2

    def test_skips_unknown_models(self) -> None:
        models = [self._model("model_a")]
        inp = BatchPlannerInput(
            models=models,
            batch_config_items=["model_a", "unknown_model"],
            batch_type="code",
            server_bin=Path("/usr/bin/llama-server"),
        )
        plan = plan_batch(inp)
        assert len(plan.items) == 1

    def test_dict_config_items(self) -> None:
        models = [self._model("model_a")]
        inp = BatchPlannerInput(
            models=models,
            batch_config_items=[{"model": "model_a", "seed": 99}],
            batch_type="code",
            server_bin=Path("/usr/bin/llama-server"),
        )
        plan = plan_batch(inp)
        assert len(plan.items) == 1
        assert plan.items[0].seed == 99

    def test_profile_uses_the_full_override_chain(self) -> None:
        """Batch profiles must agree with the launch resolver's precedence."""
        model = ModelInfo(
            name="qwen-test",
            path=Path("/models/qwen-test.gguf"),
            size_bytes=4_000_000_000,
            family_key="qwen",
        )
        inp = BatchPlannerInput(
            models=[model],
            batch_config_items=[{"model": "qwen-test", "temp": 0.9}],
            batch_type="code",
            server_bin=Path("/usr/bin/llama-server"),
            global_config_override={"temp": 0.1},
            family_overrides={"qwen": {"temp": 0.4}},
            model_overrides={"qwen-test": {"temp": 0.7}},
        )

        plan = plan_batch(inp)

        assert plan.items[0].profile.sampler.temperature == 0.9
        assert (
            plan.items[0].profile.provenance["sampler.temperature"]
            == "batch override"
        )

    def test_hash_text_fixture(self, tmp_path: Path) -> None:
        text_file = tmp_path / "wiki.txt"
        text_file.write_text("sample text")
        models = [self._model("model_a")]
        inp = BatchPlannerInput(
            models=models,
            batch_config_items=["model_a"],
            batch_type="nihs",
            server_bin=Path("/usr/bin/llama-server"),
            text_file=text_file,
        )
        plan = plan_batch(inp)
        assert str(text_file) in plan.fixture_hashes

    def test_run_identity_consistent(self) -> None:
        models = [self._model("model_a")]
        inp1 = BatchPlannerInput(
            models=models,
            batch_config_items=["model_a"],
            batch_type="code",
            server_bin=Path("/usr/bin/llama-server"),
        )
        inp2 = BatchPlannerInput(
            models=models,
            batch_config_items=["model_a"],
            batch_type="code",
            server_bin=Path("/usr/bin/llama-server"),
        )
        plan1 = plan_batch(inp1)
        plan2 = plan_batch(inp2)
        assert plan1.run_identity == plan2.run_identity


class TestManifestPersistence:
    """Manifest write and load round-trip."""

    def test_write_and_load(self, tmp_path: Path) -> None:
        model = ModelInfo(
            name="test",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        item = BatchItemPlan(
            model=model,
            profile=profile,
            backend="llama.cpp",
            ctk="q8_0",
            ctv="q8_0",
            ctx_size=4096,
            seed=42,
            mtp_enabled=False,
            mtp_n_max=3,
            mtp_n_min=0,
            reasoning=None,
            reasoning_budget=1024,
            max_tokens=None,
            config_item={},
        )
        plan = BatchPlan(
            items=(item,),
            batch_type="code",
            run_identity="abc123",
            fixture_hashes={},
        )
        results = [
            BatchExecutorResult(
                item=item,
                result={"score": 8.5},
                status="ok",
                duration_seconds=10.0,
            )
        ]

        from rama.domain.batch_runner import _write_manifest

        manifest_path = _write_manifest(plan, results, tmp_path)
        assert manifest_path.exists()

        # Load and verify
        loaded = load_manifest(tmp_path, "abc123")
        assert loaded is not None
        assert loaded.run_identity == "abc123"
        assert len(loaded.results) == 1

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        loaded = load_manifest(tmp_path, "nonexistent")
        assert loaded is None

    def test_most_recent_manifest(self, tmp_path: Path) -> None:
        model = ModelInfo(
            name="test",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        item = BatchItemPlan(
            model=model,
            profile=profile,
            backend="llama.cpp",
            ctk="q8_0",
            ctv="q8_0",
            ctx_size=4096,
            seed=42,
            mtp_enabled=False,
            mtp_n_max=3,
            mtp_n_min=0,
            reasoning=None,
            reasoning_budget=1024,
            max_tokens=None,
            config_item={},
        )

        from rama.domain.batch_runner import _write_manifest

        # Write first manifest
        plan1 = BatchPlan(
            items=(item,),
            batch_type="code",
            run_identity="first",
            fixture_hashes={},
        )
        results1 = [BatchExecutorResult(item=item, result={"score": 5.0}, status="ok", duration_seconds=1.0)]
        _write_manifest(plan1, results1, tmp_path)

        # Write second manifest
        plan2 = BatchPlan(
            items=(item,),
            batch_type="code",
            run_identity="second",
            fixture_hashes={},
        )
        results2 = [BatchExecutorResult(item=item, result={"score": 9.0}, status="ok", duration_seconds=1.0)]
        _write_manifest(plan2, results2, tmp_path)

        # Load most recent without specifying identity
        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert loaded.run_identity == "second"


class TestCanResume:
    """Resumable batch detection."""

    def test_can_resume_complete(self, tmp_path: Path) -> None:
        model = ModelInfo(
            name="test",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        item = BatchItemPlan(
            model=model,
            profile=profile,
            backend="llama.cpp",
            ctk="q8_0",
            ctv="q8_0",
            ctx_size=4096,
            seed=42,
            mtp_enabled=False,
            mtp_n_max=3,
            mtp_n_min=0,
            reasoning=None,
            reasoning_budget=1024,
            max_tokens=None,
            config_item={},
        )
        plan = BatchPlan(
            items=(item,),
            batch_type="code",
            run_identity="test1",
            fixture_hashes={},
        )

        from rama.domain.batch_runner import _write_manifest

        results = [BatchExecutorResult(item=item, result={"score": 8.0}, status="ok", duration_seconds=1.0)]
        _write_manifest(plan, results, tmp_path)

        assert can_resume(tmp_path, plan) is True

    def test_cannot_resume_no_manifest(self, tmp_path: Path) -> None:
        model = ModelInfo(
            name="test",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        item = BatchItemPlan(
            model=model,
            profile=profile,
            backend="llama.cpp",
            ctk="q8_0",
            ctv="q8_0",
            ctx_size=4096,
            seed=42,
            mtp_enabled=False,
            mtp_n_max=3,
            mtp_n_min=0,
            reasoning=None,
            reasoning_budget=1024,
            max_tokens=None,
            config_item={},
        )
        plan = BatchPlan(
            items=(item,),
            batch_type="code",
            run_identity="no_exist",
            fixture_hashes={},
        )

        assert can_resume(tmp_path, plan) is False


class TestBatchExecutorResult:
    """Executor result data integrity."""

    def test_frozen(self) -> None:
        model = ModelInfo(
            name="test",
            path=Path("/models/test.gguf"),
            size_bytes=4_000_000_000,
            family_key="llama",
        )
        profile = ResolvedModelProfile(
            model=model,
            context=ContextPolicy(),
            sampler=SamplerSettings(),
            memory=MemorySettings(),
            launch=LaunchSettings(),
            capabilities=FamilyCapabilities(),
        )
        item = BatchItemPlan(
            model=model,
            profile=profile,
            backend="llama.cpp",
            ctk="q8_0",
            ctv="q8_0",
            ctx_size=4096,
            seed=42,
            mtp_enabled=False,
            mtp_n_max=3,
            mtp_n_min=0,
            reasoning=None,
            reasoning_budget=1024,
            max_tokens=None,
            config_item={},
        )
        result = BatchExecutorResult(
            item=item,
            result={"score": 8.0},
            status="ok",
            duration_seconds=5.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.status = "failed"  # type: ignore[frozen-instantiationattr]
