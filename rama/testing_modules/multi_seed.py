"""Multi-seed evaluation module.

Tests model with multiple seeds for robustness:
- 5 seeds (1-5) for family-specific runs with family params
- 1 deterministic run (temp=0, seed=42)

Uses setup()/teardown() lifecycle to start model once and
reset context between seeds (no server restarts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rama.core import find_free_port
from typing import Any

from rich.console import Console

from .base import TestModule, TestResult, TestSuiteResult

console = Console()


@dataclass
class MultiSeedConfig:
    """Configuration for multi-seed evaluation."""

    num_seeds: int = 5  # Number of seeds for family-specific runs
    deterministic_seed: int = 42  # Seed for deterministic run
    deterministic_temp: float = 0.0  # Temperature for deterministic run
    deterministic_top_k: int = 1  # Greedy decoding for deterministic run
    deterministic_min_p: float = 0.0  # Disable min_p pruning for deterministic run
    family_params_key: str = "family_params"  # Key for family params in config


class MultiSeedModule(TestModule):
    """Run model tests with multiple seeds for statistical robustness.

    Uses setup()/teardown() lifecycle:
    - setup(): Start model ONCE
    - run(): Loop through seeds with context resets (no restarts)
    - teardown(): Stop model ONCE

    This reduces batch execution time by ~80% vs restarting per seed.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any,
        multi_seed_config: MultiSeedConfig | None = None,
        host: str = "127.0.0.1",
    ):
        super().__init__(config, model, host)
        self.multi_seed_config = multi_seed_config or MultiSeedConfig()
        self._system_prompt: str = "You are a helpful assistant."

    def setup(self) -> None:
        """Start model ONCE on a free port."""
        self._log("Setting up: starting model once for all seeds")
        port = find_free_port(self.config)
        family_params = self._get_family_params()

        self._port = self._start_model(
            port=port,
            ctx_size=self.config.get("batch_size", 4096),
            ctk="f16",
            ctv="f16",
            **family_params,
        )
        self._log(f"Model started on port {self._port}")

    def teardown(self) -> None:
        """Stop model ONCE."""
        if self._port:
            self._stop_model(self._port)
            self._log(f"Model stopped on port {self._port}")
            self._port = None

    def run(self) -> TestSuiteResult:
        """Run multi-seed evaluation with context resets between seeds."""
        start_time = time.perf_counter()
        self._log("Starting multi-seed evaluation")

        if self._port is None:
            return TestSuiteResult(
                module_name="multi_seed",
                error="Model not started. Call setup() first.",
            )

        family_params = self._get_family_params()
        seed_results = []

        # Run with multiple seeds
        for seed in range(1, self.multi_seed_config.num_seeds + 1):
            self._log(f"  Running seed {seed}/{self.multi_seed_config.num_seeds}")

            seed_temp = self.config.get("temp", 0.6)
            seed_top_p = self.config.get("top_p", 1.0)
            seed_top_k = self.config.get("top_k", 20)
            seed_min_p = self.config.get("min_p", 0.0)

            # Reset context before each seed (no server restart)
            reset_data = self._reset_context_for_seed(seed)

            # Run tests with this seed
            test_result = self._run_tests_for_seed(
                seed=seed,
                temp=seed_temp,
                top_p=seed_top_p,
                top_k=seed_top_k,
                min_p=seed_min_p,
                conversation=reset_data.get("conversation") if reset_data else None,
            )

            seed_results.append({
                "seed": seed,
                "score": test_result.overall_score,
                "error": test_result.error if test_result.error else None,
            })

            if test_result.overall_score is not None:
                self._log(f"    Score: {test_result.overall_score:.1f}")

        # Run deterministic test
        self._log(
            f"  Running deterministic test (seed={self.multi_seed_config.deterministic_seed}, temp={self.multi_seed_config.deterministic_temp})"
        )
        det_reset = self._reset_context_for_seed(self.multi_seed_config.deterministic_seed)
        det_result = self._run_tests_for_seed(
            seed=self.multi_seed_config.deterministic_seed,
            temp=self.multi_seed_config.deterministic_temp,
            top_k=self.multi_seed_config.deterministic_top_k,
            min_p=self.multi_seed_config.deterministic_min_p,
            conversation=det_reset.get("conversation") if det_reset else None,
        )

        # Calculate statistics
        scores = [r["score"] for r in seed_results if r["score"] is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            std_dev = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
        else:
            avg_score = min_score = max_score = std_dev = 0

        duration = time.perf_counter() - start_time

        suite_result = TestSuiteResult(
            module_name="multi_seed",
            overall_score=avg_score,
            metadata={
                "seed_results": seed_results,
                "deterministic_result": {
                    "seed": self.multi_seed_config.deterministic_seed,
                    "temp": self.multi_seed_config.deterministic_temp,
                    "top_k": self.multi_seed_config.deterministic_top_k,
                    "min_p": self.multi_seed_config.deterministic_min_p,
                    "score": det_result.overall_score,
                    "error": det_result.error,
                },
                "statistics": {
                    "avg_score": avg_score,
                    "min_score": min_score,
                    "max_score": max_score,
                    "std_dev": std_dev,
                    "num_seeds": len(scores),
                },
                "duration_seconds": duration,
            },
        )

        self._log(f"Multi-seed evaluation complete in {duration:.1f}s")
        self._log(
            f"  Stats: avg={avg_score:.1f}, min={min_score:.1f}, max={max_score:.1f}, std={std_dev:.2f}"
        )
        return suite_result

    def _get_family_params(self) -> dict[str, Any]:
        """Get family-specific parameters from config."""
        family_config = self.config.get("families", {}).get(self.model.family, {})
        return {k: v for k, v in family_config.items() if k not in ("name", "path")}

    def _reset_context_for_seed(self, seed: int) -> dict | None:
        """Reset context for a new seed using warmup messages.
        
        Returns conversation dict or None on failure.
        """
        from rama.testing import reset_context

        try:
            return reset_context(
                self._port,
                system_prompt=self._system_prompt,
                host=self.host,
            )
        except Exception as e:
            self._log(f"    Context reset failed for seed {seed}: {e}", "yellow")
            return None

    def _run_tests_for_seed(
        self,
        seed: int,
        temp: float,
        top_p: float,
        top_k: int,
        min_p: float,
        conversation: list[dict] | None = None,
    ) -> TestSuiteResult:
        """Run code/debug tests for a single seed on the running model."""
        from rama.testing import run_tests

        try:
            test_result = run_tests(
                self._port,
                categories=["code", "debugging"],
                config=self.config,
                temperature=temp,
                seed=seed,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                conversation=conversation,
            )
            return test_result
        except Exception as e:
            self._log(f"    Test execution error for seed {seed}: {e}", "red")
            return TestSuiteResult(
                module_name=f"multi_seed_seed_{seed}",
                error=str(e),
            )

