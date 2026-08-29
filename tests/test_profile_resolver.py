"""Tests for ProfileResolver precedence and provenance."""
import unittest
from pathlib import Path

from rama.domain.models import (
    ContextPolicy,
    FamilyCapabilities,
    FamilyDefaults,
    FamilyDefinition,
    LaunchSettings,
    MemorySettings,
    ModelInfo,
    SamplerSettings,
)
from rama.domain.resolver import ProfileResolver


class TestProfileResolver(unittest.TestCase):
    """Verify resolver precedence and provenance tracking."""

    def setUp(self):
        self.model = ModelInfo(
            name="test-model",
            path=Path("test.gguf"),
            size_bytes=4 * 1024**3,
            family_key="qwen",
        )
        self.family = FamilyDefinition(
            key="qwen",
            matchers=(r"qwen",),
            capabilities=FamilyCapabilities(supports_reasoning=True, reasoning_template_key="enable_thinking"),
            defaults=FamilyDefaults(
                sampler=SamplerSettings(temperature=0.6, top_k=20),
                context=ContextPolicy(small_ctx=24576, large_ctx=131072),
            ),
        )

    def test_builtin_defaults(self):
        resolver = ProfileResolver()
        profile = resolver.resolve(self.model)

        self.assertEqual(profile.sampler.temperature, 0.6)  # Built-in default
        self.assertEqual(
            profile.provenance["sampler.temperature"], "built-in default"
        )
        self.assertEqual(profile.provenance["memory.use_mmap"], "built-in default")

    def test_family_overrides_global(self):
        """Family defaults beat global config."""
        global_cfg = {"temperature": 0.9, "top_k": 50}
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(self.model, global_config=global_cfg)

        self.assertEqual(profile.sampler.temperature, 0.6)  # Family wins
        self.assertIn("sampler.temperature", profile.provenance)
        self.assertEqual(profile.provenance["sampler.temperature"], "family default")

    def test_cli_overrides_everything(self):
        """CLI overrides beat family and global."""
        cli = {"temperature": 0.1, "top_p": 0.9}
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(self.model, cli_override=cli)

        self.assertEqual(profile.sampler.temperature, 0.1)
        self.assertEqual(profile.provenance["sampler.temperature"], "CLI override")
        self.assertEqual(profile.provenance["sampler.top_p"], "CLI override")

    def test_full_precedence_chain(self):
        """Verify full precedence: CLI > batch > model > family > global > builtin."""
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"temperature": 0.9, "top_k": 50},
            model_override={"temperature": 0.8},
            batch_override={"temperature": 0.7},
            cli_override={"temperature": 0.1},
        )

        # CLI wins for temperature
        self.assertEqual(profile.sampler.temperature, 0.1)
        self.assertEqual(profile.provenance["sampler.temperature"], "CLI override")

        # Family wins for top_k (since global is weaker)
        self.assertEqual(profile.sampler.top_k, 20)
        self.assertEqual(profile.provenance["sampler.top_k"], "family default")

    def test_capabilities_from_family(self):
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(self.model)

        self.assertTrue(profile.capabilities.supports_reasoning)
        self.assertEqual(profile.capabilities.reasoning_template_key, "enable_thinking")
        self.assertEqual(
            profile.provenance["capabilities.supports_reasoning"],
            "family default",
        )

    # ---- Multi-class merging tests (Phase 1 completion) ----

    def test_context_merged_through_precedence(self):
        """Context settings go through full precedence chain."""
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"small_ctx": 8192},
            model_override={"small_ctx": 16384},
            cli_override={"large_ctx": 262144},
        )

        # Family default (24576) beats global (8192) for small_ctx since no model/cli override
        # But model_override (16384) beats family (24576)
        self.assertEqual(profile.context.small_ctx, 16384)
        self.assertEqual(profile.provenance["context.small_ctx"], "model override")

        # CLI wins for large_ctx
        self.assertEqual(profile.context.large_ctx, 262144)
        self.assertEqual(profile.provenance["context.large_ctx"], "CLI override")

    def test_memory_merged_through_precedence(self):
        """Memory settings go through full precedence chain."""
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            global_config={"fit_target_mib": 2048},
            model_override={"use_mmap": False},
            cli_override={"n_gpu_layers": -1},
        )

        self.assertEqual(profile.memory.fit_target_mib, 2048)
        self.assertEqual(profile.provenance["memory.fit_target_mib"], "global config")
        self.assertFalse(profile.memory.use_mmap)
        self.assertEqual(profile.provenance["memory.use_mmap"], "model override")
        self.assertEqual(profile.memory.n_gpu_layers, -1)
        self.assertEqual(profile.provenance["memory.n_gpu_layers"], "CLI override")

    def test_launch_merged_through_precedence(self):
        """Launch settings go through full precedence chain."""
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            global_config={"backend": "turboquant"},
            cli_override={"seed": 42},
        )

        self.assertEqual(profile.launch.backend, "turboquant")
        self.assertEqual(profile.provenance["launch.backend"], "global config")
        self.assertEqual(profile.launch.seed, 42)
        self.assertEqual(profile.provenance["launch.seed"], "CLI override")

    def test_nested_override_format(self):
        """Nested override format works: {"sampler": {...}, "context": {...}}."""
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            cli_override={
                "sampler": {"temperature": 0.1},
                "context": {"small_ctx": 4096},
                "memory": {"fit_target_mib": 512},
                "launch": {"seed": 99},
            },
        )

        self.assertEqual(profile.sampler.temperature, 0.1)
        self.assertEqual(profile.provenance["sampler.temperature"], "CLI override")
        self.assertEqual(profile.context.small_ctx, 4096)
        self.assertEqual(profile.provenance["context.small_ctx"], "CLI override")
        self.assertEqual(profile.memory.fit_target_mib, 512)
        self.assertEqual(profile.provenance["memory.fit_target_mib"], "CLI override")
        self.assertEqual(profile.launch.seed, 99)
        self.assertEqual(profile.provenance["launch.seed"], "CLI override")

    def test_mixed_override_format(self):
        """Flat and nested keys can be mixed in same override dict."""
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            cli_override={
                "temperature": 0.2,  # flat -> sampler
                "context": {"large_ctx": 512000},  # nested
                "fit_target_mib": 256,  # flat -> memory
            },
        )

        self.assertEqual(profile.sampler.temperature, 0.2)
        self.assertEqual(profile.context.large_ctx, 512000)
        self.assertEqual(profile.memory.fit_target_mib, 256)

    def test_zero_and_false_are_valid_overrides(self):
        """False, 0, and empty collections are legitimate overrides."""
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            cli_override={
                "use_mmap": False,
                "top_k": 0,
                "min_p": 0.0,
            },
        )

        self.assertFalse(profile.memory.use_mmap)
        self.assertEqual(profile.provenance["memory.use_mmap"], "CLI override")
        self.assertEqual(profile.sampler.top_k, 0)
        self.assertEqual(profile.provenance["sampler.top_k"], "CLI override")
        self.assertEqual(profile.sampler.min_p, 0.0)
        self.assertEqual(profile.provenance["sampler.min_p"], "CLI override")

    def test_family_context_beats_global(self):
        """Family context defaults beat global config."""
        resolver = ProfileResolver(self.family)
        profile = resolver.resolve(
            self.model,
            global_config={"small_ctx": 8192, "large_ctx": 4096},
        )

        # Family (24576) beats global (8192)
        self.assertEqual(profile.context.small_ctx, 24576)
        self.assertEqual(profile.provenance["context.small_ctx"], "family default")
        # Family (131072) beats global (4096)
        self.assertEqual(profile.context.large_ctx, 131072)
        self.assertEqual(profile.provenance["context.large_ctx"], "family default")

    def test_temp_alias_routes_to_temperature(self):
        """Historical 'temp' key in config is aliased to 'temperature'."""
        resolver = ProfileResolver()
        profile = resolver.resolve(
            self.model,
            cli_override={"temp": 0.5},
        )

        self.assertEqual(profile.sampler.temperature, 0.5)
        self.assertEqual(profile.provenance["sampler.temperature"], "CLI override")


if __name__ == "__main__":
    unittest.main()
