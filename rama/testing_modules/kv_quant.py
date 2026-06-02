"""KV quantization comparison module.

Tests model quality across different KV cache quantization types:
- f16 (full precision baseline)
- q8_0, q4_0 (standard quantization)
- turbo3, turbo4 (TurboQuant variants)

Measures code/debug scores, perplexity, and KV cache memory usage.

Uses setup()/teardown() lifecycle for each quant type.
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
class KVQuantConfig:
    """Configuration for KV quant comparison."""

    quants: list[str] = field(
        default_factory=lambda: ["f16", "q8_0", "q4_0", "turbo3", "turbo4"]
    )
    context_sizes: list[int] = field(
        default_factory=lambda: [8192, 32768, 65536, 131072]
    )
    baseline_quant: str = "f16"
    perplexity_file: str = "wikitext-2/wiki.valid.tokens"
    perplexity_ctx: int = 4096
    degradation_threshold: float = 0.10


class KVQuantModule(TestModule):
    """Compare model quality across KV quantization types.

    Uses setup()/teardown() lifecycle for each quant type.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any,
        kv_config: KVQuantConfig | None = None,
        host: str = "127.0.0.1",
    ):
        super().__init__(config, model, host)
        self.kv_config = kv_config or KVQuantConfig()
        self._baseline_scores: dict[str, float] = {}

    def setup(self) -> None:
        """No global setup needed - each quant is handled in run()."""
        self._log("Setting up KV quant comparison")

    def teardown(self) -> None:
        """No global teardown needed."""
        self._log("KV quant comparison teardown")

    def run(self) -> TestSuiteResult:
        """Run KV quant comparison across all quant types."""
        start_time = time.perf_counter()
        self._log("Starting KV quant comparison")

        all_results = {}
        baseline_perplexity = None

        for quant in self.kv_config.quants:
            self._log(f"Testing KV quant: {quant}")
            quant_start = time.perf_counter()

            # Setup for this quant
            self._setup_quant(quant)
            
            quant_result = None
            try:
                score_result = self._run_code_debug_tests(self._port)
                perplexity = self._run_perplexity(
                    self._port,
                    self.kv_config.perplexity_file,
                    self.kv_config.perplexity_ctx,
                )
                kv_size_mb = self._estimate_kv_size(
                    self.model.size_gb,
                    self.model.family,
                    quant,
                    self.kv_config.context_sizes[-1],
                )

                quant_result = {
                    "score": score_result.get("score") if score_result else 0.0,
                    "code_score": score_result.get("code_score") if score_result else 0.0,
                    "debug_score": score_result.get("debug_score") if score_result else 0.0,
                    "perplexity": perplexity,
                    "kv_size_mb": kv_size_mb,
                    "context_size": self.kv_config.context_sizes[-1],
                    "backend": "turboquant" if quant.startswith("turbo") else "llama.cpp",
                    "duration_seconds": time.perf_counter() - quant_start,
                }

                all_results[quant] = quant_result

                if quant == self.kv_config.baseline_quant:
                    self._baseline_scores = {
                        "score": quant_result["score"],
                        "perplexity": perplexity,
                    }
                    baseline_perplexity = perplexity
                    self._log(f"Baseline established: {quant}")

                self._log(
                    f"  Score: {quant_result['score']:.1f}, "
                    f"PPL: {perplexity:.2f}" if perplexity else f"  Score: {quant_result['score']:.1f}"
                )

            except Exception as e:
                self._log(f"  Error with {quant}: {e}", "red")
                all_results[quant] = {
                    "score": 0.0,
                    "error": str(e),
                    "duration_seconds": time.perf_counter() - quant_start,
                }
            finally:
                # Teardown for this quant
                self._teardown_quant()

        degradation_data = self._calculate_degradation(all_results)

        duration = time.perf_counter() - start_time
        result = TestResult(
            name="kv_quant_comparison",
            score=all_results.get(self.kv_config.baseline_quant, {}).get("score", 0.0),
            metadata={
                "results": all_results,
                "baseline": self._baseline_scores,
                "degradation": degradation_data,
                "baseline_quant": self.kv_config.baseline_quant,
                "quants_tested": self.kv_config.quants,
                "duration_seconds": duration,
            },
        )

        self._log(f"KV quant comparison complete in {duration:.1f}s")
        return TestSuiteResult(
            module_name="kv_quant",
            results=[result],
            overall_score=result.score,
            metadata=result.metadata,
        )

    def _setup_quant(self, quant: str) -> None:
        """Start model with specific KV quant settings."""
        port = find_free_port(self.config)
        ctx_size = self.kv_config.context_sizes[-1]

        self._port = self._start_model(
            port=port,
            ctx_size=ctx_size,
            ctk=quant,
            ctv=quant,
        )
        self._log(f"Model started on port {self._port} with {quant} KV")

    def _teardown_quant(self) -> None:
        """Stop model."""
        if self._port:
            try:
                self._stop_model(self._port)
                self._log(f"Stopped model on port {self._port}")
            except Exception as e:
                self._log(f"Failed to stop model on port {self._port}: {e}", style="red")
            finally:
                self._port = None

    def _run_code_debug_tests(self, port: int) -> dict[str, float] | None:
        """Run code and debugging tests on running model."""
        from rama.testing import run_tests

        try:
            test_result = run_tests(
                port,
                categories=["code", "debugging"],
                config=self.config,
            )

            return {
                "score": test_result.overall_score,
                "code_score": self._extract_category_score(
                    test_result.results, "code"
                ),
                "debug_score": self._extract_category_score(
                    test_result.results, "debugging"
                ),
            }
        except Exception as e:
            self._log(f"  Test execution error: {e}", "yellow")
            return None

    def _extract_category_score(
        self, results: list[Any], category: str
    ) -> float:
        """Extract score for a specific test category."""
        scores = [
            r.score
            for r in results
            if r.category == category and r.score is not None
        ]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _run_perplexity(
        self, port: int, text_file: str, ctx_size: int
    ) -> float | None:
        """Run perplexity test (placeholder for llama-perplexity integration)."""
        self._log("  Perplexity: (to be implemented via llama-perplexity)", "dim")
        return None

    def _estimate_kv_size(
        self, model_size_gb: float, family: str, kv_quant: str, ctx_size: int
    ) -> float:
        """Estimate KV cache size in MB."""
        from rama.core import calc_kv_cache

        family_config = self.config.get("families", {}).get(family, {})
        kv_bytes = family_config.get("kv_bytes_per_token", 15000)

        quant_multiplier = {
            "f16": 1.0,
            "q8_0": 0.5,
            "q4_0": 0.25,
            "turbo3": 0.3,
            "turbo4": 0.2,
        }.get(kv_quant, 0.5)

        kv_bytes *= quant_multiplier
        total_bytes = kv_bytes * ctx_size
        return total_bytes / (1024 * 1024)

    def _calculate_degradation(
        self, all_results: dict[str, dict]
    ) -> dict[str, dict]:
        """Calculate degradation metrics vs baseline."""
        baseline = self._baseline_scores

        degradation = {}
        for quant, result in all_results.items():
            if quant == self.kv_config.baseline_quant:
                continue

            score_degradation = 0.0
            if baseline.get("score") and result.get("score"):
                score_degradation = (
                    baseline["score"] - result["score"]
                ) / baseline["score"] if baseline["score"] > 0 else 0.0

            ppl_degradation = 0.0
            if baseline.get("perplexity") and result.get("perplexity"):
                ppl_degradation = (
                    result["perplexity"] - baseline["perplexity"]
                ) / baseline["perplexity"] if baseline["perplexity"] > 0 else 0.0

            flagged = (
                score_degradation > self.kv_config.degradation_threshold
                or ppl_degradation > self.kv_config.degradation_threshold
            )

            degradation[quant] = {
                "score_degradation": round(score_degradation * 100, 2),
                "perplexity_degradation": round(ppl_degradation * 100, 2),
                "flagged": flagged,
            }

        return degradation

