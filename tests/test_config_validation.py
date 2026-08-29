"""Tests for early configuration validation."""

import unittest

from rama import DEFAULT_CONFIG, validate_config


class ConfigValidationTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        self.assertEqual(validate_config(DEFAULT_CONFIG), [])

    def test_reports_multiple_invalid_values(self) -> None:
        config = {
            **DEFAULT_CONFIG,
            "batch_size": 0,
            "temp": 3,
            "default_port": 70000,
            "families": [],
        }

        errors = validate_config(config)

        self.assertIn("batch_size must be a positive integer", errors)
        self.assertIn("temp must be a number between 0 and 2", errors)
        self.assertIn("default_port must be a TCP port between 1024 and 65535", errors)
        self.assertIn("families must be a mapping of family names to settings", errors)


if __name__ == "__main__":
    unittest.main()
