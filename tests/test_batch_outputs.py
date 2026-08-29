"""Regression tests for the compact code-batch result contract."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from rama import Config
from rama.testing_modules.batch_runner import BatchRunner, BatchRunnerConfig


class BatchOutputTests(unittest.TestCase):
    def test_code_batch_contains_only_test_and_kv_settings(self) -> None:
        batch_path = Path(__file__).parent.parent / "config" / "batch_code_test.yaml"
        batch = yaml.safe_load(batch_path.read_text())
        allowed_keys = {
            "model",
            "ctx",
            "ctk",
            "ctv",
            "max_tokens",
            "reasoning",
            "mtp",
        }

        self.assertTrue(batch)
        for item in batch:
            self.assertEqual(set(item) - allowed_keys, set(), item["model"])
            self.assertEqual(item["ctx"], 16384, item["model"])
            self.assertEqual(item["ctk"], "f16", item["model"])
            self.assertEqual(item["ctv"], "f16", item["model"])
            self.assertEqual(item["max_tokens"], 1024, item["model"])
            self.assertFalse(item["reasoning"], item["model"])

    def test_code_batch_saves_only_the_aggregate_result_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = BatchRunner(BatchRunnerConfig(config_file="unused.yaml"))
            runner.results_file = output_dir / "batch_20260722_120000.json"
            runner.csv_file = output_dir / "batch_20260722_120000.csv"
            runner.all_results = [
                {
                    "model": "example-model",
                    "backend": "llama.cpp",
                    "ctk": "q8_0",
                    "ctv": "q8_0",
                    "ctx": 16384,
                    "reasoning": False,
                    "score": 8.0,
                    "test_time_seconds": 12.5,
                    "runs": [
                        {"run_type": "family", "seed": 1, "score": 7.0},
                        {"run_type": "family", "seed": 2, "score": 9.0},
                    ],
                    "tests": [
                        {"category": "code", "score": 10},
                        {"category": "debugging", "score": 6},
                    ],
                },
            ]

            runner._save_code_results()

            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["batch_20260722_120000.csv", "batch_20260722_120000.json"],
            )
            saved = json.loads(runner.results_file.read_text())
            self.assertEqual(saved[0]["runs"][0]["run_type"], "family")

            with runner.csv_file.open(newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["score"], "8.0000")
            self.assertEqual(row["family_score"], "8.0000")
            self.assertEqual(row["family_score_stddev"], "1.4142")

    def test_code_batch_records_a_launch_failure_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = BatchRunner(BatchRunnerConfig(config_file="unused.yaml", quiet=True))
            runner.cfg = Config("rama.yaml")
            runner.results_file = output_dir / "batch.json"
            runner.csv_file = output_dir / "batch.csv"
            model = SimpleNamespace(name="too-big", family="laguna")

            with patch(
                "rama.testing_modules.batch_runner.start_model",
                side_effect=RuntimeError("llama-server crashed with exit code -9"),
            ):
                runner._run_single_code_model(model, {"ctx": 16384})

            self.assertEqual(len(runner.all_results), 1)
            failure = runner.all_results[0]
            self.assertEqual(failure["status"], "launch_failed")
            self.assertIsNone(failure["score"])
            self.assertIn("exit code -9", failure["error"])
            self.assertTrue(runner.results_file.exists())


if __name__ == "__main__":
    unittest.main()
