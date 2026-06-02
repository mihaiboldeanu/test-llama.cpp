"""Unified test runner orchestrator.

Coordinates all testing modules in a pipeline:
1. Warm-up
2. KV quant comparison (f16 baseline)
3. Context difficulty
4. Multi-seed evaluation
5. Enhanced NIHS
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .base import TestModule, TestResult, TestSuiteResult
from .kv_quant import KVQuantConfig, KVQuantModule
from .context_difficulty import ContextDifficultyConfig, ContextDifficultyModule
from .warmup import WarmupConfig, WarmupModule
from .enhanced_nihs import EnhancedNIHSConfig, EnhancedNIHSModule
from .multi_seed import MultiSeedConfig, MultiSeedModule

console = Console()


@dataclass
class UnifiedTestConfig:
    """Configuration for unified testing."""

    # Enable/disable individual modules
    enable_warmup: bool = True
    enable_kv_quant: bool = True
    enable_context_difficulty: bool = True
    enable_multi_seed: bool = True
    enable_enhanced_nihs: bool = True

    # Module-specific configs
    kv_quant_config: KVQuantConfig | None = None
    context_difficulty_config: ContextDifficultyConfig | None = None
    warmup_config: WarmupConfig | None = None
    nihs_config: EnhancedNIHSConfig | None = None
    multi_seed_config: MultiSeedConfig | None = None

    # Output
    output_dir: str = "results"
    output_format: str = "json"  # json, markdown, table


class UnifiedRunner:
    """Orchestrator for all testing modules.

    Runs tests in a pipeline and collects results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any,
        test_config: UnifiedTestConfig | None = None,
        host: str = "127.0.0.1",
    ):
        self.config = config
        self.model = model
        self.host = host
        self.test_config = test_config or UnifiedTestConfig()
        self.results: dict[str, TestResult] = {}

    def run_all(self) -> TestSuiteResult:
        """Run all enabled test modules."""
        start_time = time.perf_counter()
        console.print(f"\n[cyan]=== Unified Testing: {self.model.name} ===[/cyan]")

        # Run warm-up
        if self.test_config.enable_warmup:
            warmup = WarmupModule(
                self.config,
                self.model,
                self.test_config.warmup_config,
                self.host,
            )
            warmup_result = warmup.run()
            self.results["warmup"] = warmup_result

        # Run KV quant comparison
        if self.test_config.enable_kv_quant:
            kv_quant = KVQuantModule(
                self.config,
                self.model,
                self.test_config.kv_quant_config,
                self.host,
            )
            kv_result = kv_quant.run()
            self.results["kv_quant"] = kv_result

        # Run context difficulty
        if self.test_config.enable_context_difficulty:
            ctx_diff = ContextDifficultyModule(
                self.config,
                self.model,
                self.test_config.context_difficulty_config,
                self.host,
            )
            ctx_result = ctx_diff.run()
            self.results["context_difficulty"] = ctx_result

        # Run multi-seed evaluation
        if self.test_config.enable_multi_seed:
            multi_seed = MultiSeedModule(
                self.config,
                self.model,
                self.test_config.multi_seed_config,
                self.host,
            )
            seed_result = multi_seed.run()
            self.results["multi_seed"] = seed_result

        # Run enhanced NIHS
        if self.test_config.enable_enhanced_nihs:
            nihs = EnhancedNIHSModule(
                self.config,
                self.model,
                self.test_config.nihs_config,
                self.host,
            )
            nihs_result = nihs.run()
            self.results["enhanced_nihs"] = nihs_result

        # Save results
        duration = time.perf_counter() - start_time
        self._save_results(duration)

        # Print summary
        self._print_summary(duration)

        return TestSuiteResult(
            module_name="unified",
            metadata={"results": self.results, "duration_seconds": duration},
        )

    def _save_results(self, duration: float) -> None:
        """Save results to file."""
        output_dir = Path(self.test_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if self.test_config.output_format == "json":
            results_file = output_dir / f"unified_{timestamp}.json"
            results_data = {
                "model": self.model.name,
                "duration_seconds": duration,
                "results": {
                    name: {
                        "name": result.name,
                        "score": result.score,
                        "metadata": result.metadata,
                    }
                    for name, result in self.results.items()
                },
            }
            results_file.write_text(json.dumps(results_data, indent=2))
            console.print(f"\n[green]Results saved to {results_file}[/green]")

        elif self.test_config.output_format == "markdown":
            results_file = output_dir / f"unified_{timestamp}.md"
            md_content = self._format_as_markdown()
            results_file.write_text(md_content)
            console.print(f"\n[green]Results saved to {results_file}[/green]")

    def _format_as_markdown(self) -> str:
        """Format results as markdown."""
        lines = [f"# Unified Test Results: {self.model.name}\n"]
        lines.append(f"| Metric | Value |\n|--------|-------|")

        for name, result in self.results.items():
            score_str = f"{result.score:.1f}" if result.score is not None else "N/A"
            lines.append(f"| {name} | {score_str} |")

        return "\n".join(lines)

    def _print_summary(self, duration: float) -> None:
        """Print results summary."""
        console.print(f"\n[cyan]=== Results Summary ===[/cyan]")

        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Duration", justify="right")

        for name, result in self.results.items():
            score_str = f"{result.score:.1f}" if result.score is not None else "N/A"

            # Get duration from metadata if available
            dur = result.metadata.get("duration_seconds", 0)
            dur_str = f"{dur:.1f}s" if dur else "-"

            table.add_row(name, score_str, dur_str)

        console.print(table)
        console.print(f"\n[green]Total duration: {duration:.1f}s[/green]")
