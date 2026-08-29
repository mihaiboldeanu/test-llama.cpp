"""Regression tests for fixture execution and output comparison."""

import json
import re
import unittest
from pathlib import Path

from rama.testing_modules.code_runner import run_class_task, run_task

ROOT = Path(__file__).resolve().parent


class CodeRunnerTests(unittest.TestCase):
    def test_ordered_list_output_must_remain_ordered(self) -> None:
        task = {
            "test_script": "print(result())",
            "expected_output": ["[1, 2]"],
        }

        result = run_class_task("def result():\n    return [2, 1]", task)

        self.assertFalse(result["passed"])

    def test_order_independent_list_output_is_explicit(self) -> None:
        task = {
            "test_script": "print(result())",
            "expected_output": ["['a', 'b']"],
            "order_independent": True,
        }

        result = run_class_task("def result():\n    return ['b', 'a']", task)

        self.assertTrue(result["passed"])

    def test_unexpected_extra_output_fails(self) -> None:
        task = {
            "test_script": "print(1)\nprint(2)",
            "expected_output": ["1"],
        }

        result = run_class_task("", task)

        self.assertFalse(result["passed"])
        self.assertIn("output line count mismatch", result["details"][-1]["error"])

    def test_final_empty_string_output_is_preserved(self) -> None:
        task = {
            "test_script": "print(result())",
            "expected_output": [""],
        }

        result = run_class_task("def result():\n    return ''", task)

        self.assertTrue(result["passed"])

    def test_dual_mode_fixture_runs_cases_and_script(self) -> None:
        task = {
            "function_name": "result",
            "test_cases": [{"inputs": [0], "expected": 0}],
            "test_script": "print(result(1))",
            "expected_output": ["2"],
        }

        result = run_task("def result(value):\n    return value", task)

        self.assertFalse(result["passed"])
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 2)

    def test_two_sum_validator_accepts_an_alternative_valid_pair(self) -> None:
        task = {
            "function_name": "two_sum",
            "validator": "two_sum_indices",
            "test_cases": [
                {
                    "inputs": [[-1, -2, -3, -4, -5], -6],
                    "expected": [1, 3],
                }
            ],
        }

        result = run_task("def two_sum(nums, target):\n    return [0, 4]", task)

        self.assertTrue(result["passed"])

    def test_two_sum_validator_rejects_an_invalid_pair(self) -> None:
        task = {
            "function_name": "two_sum",
            "validator": "two_sum_indices",
            "test_cases": [
                {"inputs": [[2, 7], 9], "expected": [0, 1]},
            ],
        }

        result = run_task("def two_sum(nums, target):\n    return [0, 0]", task)

        self.assertFalse(result["passed"])

    def test_cycle_debug_fixture_rejects_its_broken_code(self) -> None:
        self._assert_fixture_rejects_prompt_code("cycle.json")

    def test_min_window_debug_fixture_rejects_its_broken_code(self) -> None:
        self._assert_fixture_rejects_prompt_code("min_window_debug.json")

    def _assert_fixture_rejects_prompt_code(self, filename: str) -> None:
        task = json.loads((ROOT / "debugging" / filename).read_text())
        match = re.search(r"```python\n(.*?)```", task["prompt"], re.DOTALL)
        self.assertIsNotNone(match)

        result = run_task(match.group(1), task)  # type: ignore[union-attr]

        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
