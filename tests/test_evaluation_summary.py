"""Tests for the decision-oriented code-test evaluation summary."""

import unittest

from rama.testing import summarize_evaluation
from rama.testing_modules.base import TestResult


def result(
    name: str,
    *,
    score: float,
    category: str = "code",
    response: str = "answer",
    ran: bool = True,
    correct: bool = True,
) -> TestResult:
    return TestResult(
        name=name,
        category=category,
        score=score,
        response=response,
        ran=ran,
        correct=correct,
    )


class EvaluationSummaryTests(unittest.TestCase):
    def test_excellent_requires_score_and_reliability(self) -> None:
        summary = summarize_evaluation(
            [result("code", score=10), result("debug", score=8, category="debugging")]
        )

        self.assertEqual(summary["verdict"], "excellent")
        self.assertEqual(summary["completion_rate"], 1.0)
        self.assertEqual(summary["category_scores"], {"code": 10.0, "debugging": 8.0})

    def test_empty_or_unexecutable_outputs_are_broken(self) -> None:
        summary = summarize_evaluation(
            [
                result("works", score=10),
                result("empty", score=0, response="", ran=False, correct=False),
                result("empty", score=0, response="", ran=False, correct=False),
            ]
        )

        self.assertEqual(summary["verdict"], "broken")
        self.assertEqual(summary["recurring_failures"][0]["name"], "empty")
        self.assertEqual(summary["recurring_failures"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
