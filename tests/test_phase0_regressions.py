"""Regression tests for Phase 0 stabilization fixes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rama.__init__ import Config


class TestRepeatPenaltyTranslation(unittest.TestCase):
    """Ensure historical 'repetition_penalty' is translated to 'repeat_penalty'."""

    def _make_cfg(self, data: dict) -> Config:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as f:
            yaml.dump(data, f)
            f.flush()
            return Config(f.name)

    def test_top_level_translation(self):
        cfg = self._make_cfg({"repetition_penalty": 1.05})
        self.assertEqual(cfg["--repeat-penalty"], 1.05)
        self.assertNotIn("repetition_penalty", cfg)

    def test_family_translation(self):
        cfg = self._make_cfg({
            "families": {
                "qwen": {"repetition_penalty": 1.1}
            }
        })
        self.assertEqual(cfg["families"]["qwen"]["--repeat-penalty"], 1.1)

    def test_model_translation(self):
        cfg = self._make_cfg({
            "models": {
                "my_model": {"repetition_penalty": 1.2}
            }
        })
        self.assertEqual(
            cfg["models"]["my_model"]["--repeat-penalty"], 1.2
        )

    def test_translation_emits_deprecation_warning(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_cfg({"repetition_penalty": 1.05})
            rep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(rep_warnings), 1)
            self.assertIn("repeat-penalty", str(rep_warnings[0].message))


class TestFitTargetMibResolution(unittest.TestCase):
    """Ensure fit_target_mib resolves from config, not hardcoded."""

    def test_fit_target_mib_comes_from_config(self):
        from pathlib import Path

        from rama.domain.models import ModelInfo
        from rama.domain.resolver import ProfileResolver

        model = ModelInfo(
            name="test-model",
            path=Path("test.gguf"),
            size_bytes=4 * 1024**3,
            family_key="qwen",
        )
        cfg = {"args": {"--fit-target": 512}}
        # The ProfileResolver is the single precedence path for launch settings.
        profile = ProfileResolver().resolve(model, global_config=cfg)
        self.assertEqual(profile.args["--fit-target"], 512)


class TestCLIConfigPassing(unittest.TestCase):
    """Ensure CLI commands accept and use --config."""

    def test_test_command_accepts_config(self):
        from typer.testing import CliRunner

        from rama.__main__ import app

        runner = CliRunner()
        # Just verify it doesn't crash on help and recognizes the flag
        result = runner.invoke(app, ["test", "--help"])
        self.assertIn("--config", result.output)
        self.assertIn("-c", result.output)

    def test_context_command_accepts_config(self):
        from typer.testing import CliRunner

        from rama.__main__ import app

        runner = CliRunner()
        result = runner.invoke(app, ["context", "--help"])
        self.assertIn("--config", result.output)

    def test_tests_command_accepts_config(self):
        from typer.testing import CliRunner

        from rama.__main__ import app

        runner = CliRunner()
        result = runner.invoke(app, ["tests", "--help"])
        self.assertIn("--config", result.output)

    def test_nihs_batch_forwards_custom_config_path(self):
        from rama.__main__ import _run_batch_nihs

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_config = Path(temp_dir) / "batch.yaml"
            batch_config.write_text("[]\n")

            with patch("rama.__main__.BatchRunner") as batch_runner:
                batch_runner.return_value.run_nihs_batch.return_value = {}
                _run_batch_nihs(
                    Config(),
                    str(batch_config),
                    "beginning",
                    None,
                    "custom-rama.yaml",
                )

        runner_config = batch_runner.call_args.args[0]
        self.assertEqual(runner_config.config_path, "custom-rama.yaml")


if __name__ == "__main__":
    unittest.main()
