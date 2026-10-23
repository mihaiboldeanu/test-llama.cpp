"""Tests for ProfileResolver precedence and provenance."""
import re
import unittest
from pathlib import Path

from rama.domain.models import FamilyDefinition, ModelInfo
from rama.domain.resolver import ProfileResolver


class TestProfileResolver(unittest.TestCase):
    """Verify args-map merging precedence and provenance tracking."""

    def setUp(self):
        self.model = ModelInfo(
            name="test-model",
            path=Path("test.gguf"),
            size_bytes=4 * 1024**3,
            family_key="qwen",
        )
        self.family = FamilyDefinition(
            key="qwen",
            matchers=(re.compile(r"qwen", re.IGNORECASE),),
        )

    def test_global_args_provide_defaults(self):
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            global_config={"args": {"--temp": 0.6, "--top-k": 20}},
        )

        self.assertEqual(profile.args["--temp"], 0.6)
        self.assertEqual(profile.provenance["args.--temp"], "global config")

    def test_family_overrides_global(self):
        """Family config beats global config."""
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"args": {"--temp": 0.9, "--top-k": 50}},
            family_override={"args": {"--temp": 0.6}},
        )

        self.assertEqual(profile.args["--temp"], 0.6)
        self.assertEqual(profile.provenance["args.--temp"], "family config")
        self.assertEqual(profile.args["--top-k"], 50)
        self.assertEqual(profile.provenance["args.--top-k"], "global config")

    def test_cli_overrides_everything(self):
        """CLI overrides beat family and global."""
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"args": {"--temp": 0.9}},
            family_override={"args": {"--temp": 0.6}},
            cli_override={"args": {"--temp": 0.1, "--top-p": 0.9}},
        )

        self.assertEqual(profile.args["--temp"], 0.1)
        self.assertEqual(profile.provenance["args.--temp"], "CLI override")
        self.assertEqual(profile.provenance["args.--top-p"], "CLI override")

    def test_full_precedence_chain(self):
        """Verify full precedence: CLI > batch > model > family > global."""
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"args": {"--temp": 0.9, "--top-k": 50}},
            family_override={"args": {"--temp": 0.8}},
            model_override={"args": {"--temp": 0.7}},
            batch_override={"args": {"--temp": 0.2}},
            cli_override={"args": {"--temp": 0.1}},
        )

        self.assertEqual(profile.args["--temp"], 0.1)
        self.assertEqual(profile.provenance["args.--temp"], "CLI override")
        self.assertEqual(profile.args["--top-k"], 50)
        self.assertEqual(profile.provenance["args.--top-k"], "global config")

    def test_null_drops_an_inherited_flag(self):
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"args": {"--kv-unified": True}},
            model_override={"args": {"--kv-unified": None}},
        )

        self.assertNotIn("--kv-unified", profile.args)
        self.assertEqual(
            profile.provenance["args.--kv-unified"], "model override (removed)"
        )

    def test_legacy_semantic_keys_are_translated(self):
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            cli_override={"temp": 0.5, "ctx": 4096, "reasoning": False},
        )

        self.assertEqual(profile.args["--temp"], 0.5)
        self.assertEqual(profile.args["--ctx-size"], 4096)
        self.assertEqual(profile.args["--reasoning"], "off")
        self.assertFalse(profile.reasoning)

    def test_ctx_auto_uses_family_thresholds(self):
        resolver = ProfileResolver(self.family)
        small = resolver.resolve(
            self.model,
            global_config={"rules": {"small_threshold_gb": 10}},
            family_override={"small_ctx": 24576, "large_ctx": 131072},
        )
        large_model = ModelInfo(
            name="large-model",
            path=Path("large.gguf"),
            size_bytes=20 * 1024**3,
            family_key="qwen",
        )
        large = resolver.resolve(
            large_model,
            global_config={"rules": {"small_threshold_gb": 10}},
            family_override={"small_ctx": 24576, "large_ctx": 131072},
        )

        self.assertEqual(small.ctx, 24576)
        self.assertEqual(small.args["--ctx-size"], 24576)
        self.assertEqual(large.ctx, 131072)

    def test_fit_ctx_auto_tracks_ctx(self):
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"args": {"--fit-ctx": "auto"}},
            family_override={"small_ctx": 24576, "large_ctx": 131072},
        )

        self.assertEqual(profile.fit_ctx, profile.ctx)
        self.assertEqual(profile.args["--fit-ctx"], profile.ctx)

    def test_moe_fit_ctx_skip_removes_fit_ctx(self):
        family = FamilyDefinition(
            key="gemma",
            matchers=(re.compile(r"gemma", re.IGNORECASE),),
            moe_fit_ctx_skip=True,
        )
        model = ModelInfo(
            name="gemma-4-26B-A4B-it-UD-Q8_K_XL",
            path=Path("gemma.gguf"),
            size_bytes=26 * 1024**3,
            family_key="gemma",
        )
        profile = ProfileResolver(family).resolve(model)

        self.assertIsNone(profile.fit_ctx)
        self.assertNotIn("--fit-ctx", profile.args)

    def test_reasoning_auto_follows_size_threshold(self):
        resolver = ProfileResolver(self.family)
        small = resolver.resolve(self.model)
        large_model = ModelInfo(
            name="large-model",
            path=Path("large.gguf"),
            size_bytes=20 * 1024**3,
            family_key="qwen",
        )
        large = resolver.resolve(large_model)

        self.assertFalse(small.reasoning)
        self.assertEqual(small.args["--reasoning"], "off")
        self.assertTrue(large.reasoning)
        self.assertEqual(large.args["--reasoning"], "on")

    def test_reasoning_preserve_requires_reasoning_on(self):
        resolver = ProfileResolver(self.family)
        on = resolver.resolve(
            self.model,
            family_override={"reasoning_preserve": True, "small_ctx": 1, "large_ctx": 2},
            cli_override={"--ctx-size": 2, "--reasoning": True},
        )
        off = resolver.resolve(
            self.model,
            family_override={"reasoning_preserve": True, "small_ctx": 1, "large_ctx": 2},
            cli_override={"--ctx-size": 2, "--reasoning": False},
        )

        self.assertIn("--reasoning-preserve", on.args)
        self.assertNotIn("--reasoning-preserve", off.args)

    def test_mtp_and_env_merge_by_level(self):
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"rules": {"mtp_ctx_ratio": 0.8}, "env": {"A": "1"}},
            family_override={"mtp": {"n_max": 4}, "env": {"B": "2"}},
            model_override={"mtp": {"p_min": 0}},
        )

        self.assertEqual(profile.mtp["ctx_ratio"], 0.8)
        self.assertEqual(profile.mtp["n_max"], 4)
        self.assertEqual(profile.mtp["p_min"], 0)
        self.assertEqual(profile.env, {"A": "1", "B": "2"})


if __name__ == "__main__":
    unittest.main()
