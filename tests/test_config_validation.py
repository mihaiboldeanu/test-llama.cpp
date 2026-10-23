"""Tests for early configuration validation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rama import DEFAULT_CONFIG, load_config, validate_config


class ConfigValidationTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        self.assertEqual(validate_config(DEFAULT_CONFIG), [])

    def test_reports_multiple_invalid_values(self) -> None:
        config = {
            **DEFAULT_CONFIG,
            "args": {**DEFAULT_CONFIG["args"], "-b": [1], "--temp": [1]},
            "default_port": 70000,
            "families": [],
        }

        errors = validate_config(config)

        self.assertIn(
            "config.args.-b must be a string, number, boolean, or null", errors
        )
        self.assertIn(
            "config.args.--temp must be a string, number, boolean, or null", errors
        )
        self.assertIn("default_port must be a TCP port between 1024 and 65535", errors)
        self.assertIn("families must be a mapping of names to settings", errors)


    def test_yaml_bool_tensor_read_lazy_is_normalized(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rama.yaml"
            path.write_text(
                "families:\n"
                "  qwen3.8-flash-next:\n"
                "    tensor_read_lazy: on\n"
            )

            config = load_config(str(path))

        self.assertEqual(
            config["families"]["qwen3.8-flash-next"]["args"]["--lazy-mode"], "on"
        )
        self.assertNotIn(
            "tensor_read_lazy", config["families"]["qwen3.8-flash-next"]
        )

    def test_partial_sections_merge_instead_of_replacing_defaults(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rama.yaml"
            path.write_text("rules:\n  small_ctx: 8192\nargs:\n  --temp: 0.8\n")

            config = load_config(str(path))

        self.assertEqual(config["rules"]["small_ctx"], 8192)
        self.assertEqual(
            config["rules"]["large_ctx"], DEFAULT_CONFIG["rules"]["large_ctx"]
        )
        self.assertEqual(config["args"]["--temp"], 0.8)
        self.assertEqual(
            config["args"]["-ctk"], DEFAULT_CONFIG["args"]["-ctk"]
        )


if __name__ == "__main__":
    unittest.main()
