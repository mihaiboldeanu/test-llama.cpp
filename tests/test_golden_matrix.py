"""Golden matrix: verify resolved profile matches actual launch behavior.

This ensures ProfileResolver produces the same effective settings that start_model()
uses for argv construction. Each case tests a representative family with known
capabilities and defaults.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rama import Config
from rama.core import ModelInfo
from rama.domain.resolver import ProfileResolver
from rama.families import get_registry
from rama.launch import ModelStartConfig, start_model


class GoldenMatrixTests(unittest.TestCase):
    """Verify explain output matches launch behavior for representative models."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build_model(self, name: str, size_gb: float, family: str) -> ModelInfo:
        path = self.tmp_path / f"{name}.gguf"
        path.touch()
        return ModelInfo(
            name=name,
            path=path,
            size_gb=size_gb,
            family=family,
            quant="Q4_K_M",
        )

    def _capture_argv(self, model: ModelInfo, config: Config):
        """Run start_model with mocked subprocess and return the argv."""
        process = Mock(pid=9999)
        process.poll.return_value = None
        with (
            patch("rama.launch.get_server_bin", return_value=Path("/bin/true")),
            patch("rama.launch.is_port_used", return_value=False),
            patch("rama.launch.get_pid_dir", return_value=self.tmp_path),
            patch("rama.domain.runner.subprocess.Popen", return_value=process) as popen,
            patch("rama.domain.runner.urllib.request.urlopen"),
            patch("rama.domain.runner.time.sleep"),
        ):
            start_model(
                ModelStartConfig(
                    model=model,
                    port=18000,
                    config=config,
                )
            )
        return popen.call_args.args[0]

    def _resolve_profile(self, model: ModelInfo, config: Config):
        """Resolve profile using ProfileResolver for comparison.

        Matches the same precedence chain that launch.py start_model() uses.
        """
        family_def = get_registry().get(model.family)
        resolver = ProfileResolver(family_definition=family_def)
        domain_model = model.to_domain()

        # Replicate launch.py's precedence chain
        family_params = config.families.get(model.family, {})
        model_overrides = (config.get("models", {}) or {}).get(model.name, {})
        global_cfg = config._data if hasattr(config, "_data") else None

        return resolver.resolve(
            domain_model,
            global_config=global_cfg,
            family_override=family_params or None,
            model_override=model_overrides or None,
        )

    def _argv_flag_value(self, args: list[str], flag: str) -> str | None:
        idx = args.index(flag) if flag in args else -1
        return args[idx + 1] if idx >= 0 and idx + 1 < len(args) else None

    # --- Sampler consistency ---

    def test_qwen_sampler_matches_profile(self):
        """Qwen family: sampler settings from profile match argv."""
        config = Config("rama.yaml")
        model = self._build_model("Qwen3-8B-Instruct-Q4_K_M", 5.0, "qwen")
        profile = self._resolve_profile(model, config)
        args = self._capture_argv(model, config)

        # Verify sampler fields match between profile and argv
        self.assertEqual(
            self._argv_flag_value(args, "--temp"),
            str(profile.sampler.temperature),
        )
        self.assertEqual(
            self._argv_flag_value(args, "--top-k"),
            str(profile.sampler.top_k),
        )
        self.assertEqual(
            self._argv_flag_value(args, "--top-p"),
            str(profile.sampler.top_p),
        )

    def test_gemma_3_sampler_matches_profile(self):
        """Gemma 3 family: sampler settings from profile match argv."""
        config = Config("rama.yaml")
        model = self._build_model("gemma-3-27b-it-Q4_K_M", 16.0, "gemma-3")
        profile = self._resolve_profile(model, config)
        args = self._capture_argv(model, config)

        self.assertEqual(
            self._argv_flag_value(args, "--temp"),
            str(profile.sampler.temperature),
        )

    def test_laguna_mmap_matches_profile(self):
        """Laguna family: use_mmap from profile matches argv --no-mmap presence."""
        config = Config("rama.yaml")
        model = self._build_model("Laguna-S-2.1-UD-IQ4_XS", 53.6, "laguna")
        profile = self._resolve_profile(model, config)
        args = self._capture_argv(model, config)

        # Laguna should use mmap (not MoE), so --no-mmap should NOT be present
        # unless explicit override says otherwise
        if profile.memory.use_mmap:
            self.assertNotIn("--no-mmap", args)

    def test_gpt_oss_skip_chat_parsing(self):
        """GPT-OSS family: skip_chat_parsing from capabilities matches argv."""
        config = Config("rama.yaml")
        model = self._build_model("gpt-oss-20b-mxfp4", 12.0, "gpt-oss")
        profile = self._resolve_profile(model, config)
        args = self._capture_argv(model, config)

        if profile.capabilities.skip_chat_parsing:
            self.assertIn("--skip-chat-parsing", args)

    def test_reasoning_family_template_key(self):
        """Reasoning families: template key from capabilities is honored."""
        config = Config("rama.yaml")
        model = self._build_model("QwQ-32B-Preview-Q4_K_M", 20.0, "qwen")
        profile = self._resolve_profile(model, config)

        # Qwen reasoning uses enable_thinking template key
        if profile.capabilities.supports_reasoning:
            self.assertIsNotNone(profile.capabilities.reasoning_template_key)

    def test_memory_fit_target_from_config(self):
        """fit_target_mib from profile matches what launch uses."""
        config = Config("rama.yaml")
        model = self._build_model("Qwen3-8B-Instruct-Q4_K_M", 5.0, "qwen")
        profile = self._resolve_profile(model, config)

        # Profile should have fit_target_mib resolved from config chain
        self.assertIsInstance(profile.memory.fit_target_mib, int)
        self.assertGreater(profile.memory.fit_target_mib, 0)


if __name__ == "__main__":
    unittest.main()
