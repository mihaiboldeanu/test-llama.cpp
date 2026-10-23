"""Batch runner orchestrator.

Encapsulates batch testing logic for:
- code: code/debugging tests with multi-seed support
- perplexity: llama-perplexity evaluation
- nihs: needle-in-haystack batch
- kv: KV quant comparison

Designed to be used by the CLI and/or programmatically.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from rama.context_test import run_needle_suite
from rama.core import (
    detect_backend,
    discover_models,
    find_free_port,
)
from rama.domain.batch_planner import BatchPlannerInput, load_batch_items, plan_batch
from rama.launch import (
    ModelStartConfig,
    _wait_for_port_free,
    start_model,
    stop_model,
)
from rama.log import setup_logging
from rama.testing import (
    TestRunConfig,
    reset_context,
    run_tests,
    save_results_csv,
    summarize_evaluation,
    warmup_model,
)

from .system_prompt import (
    get_code_system_prompt,
    get_debug_system_prompt,
    get_system_prompt,
)

console = Console()
logger = setup_logging()


def _optional_int(value: Any) -> int | None:
    """Return an int for an explicit batch override, else None.

    None lets the resolved family/model profile supply the default instead
    of a hard-coded fallback masking it.
    """
    if value is None:
        return None
    return int(value)


@dataclass
class BatchRunnerConfig:
    """Configuration for batch testing."""

    config_file: str
    batch_type: str = "code"  # code, perplexity, nihs, kv
    start_port: int = 11435
    categories: str = "all"
    output_dir: str = "results"
    seed: int = 42
    system_prompt: str | None = None
    warmup: int = 0
    difficulty: str = "easy"
    placement: str = "before"
    ctx_size: int = 0
    config_path: str | None = None
    quiet: bool = False
    dry_run: bool = False
    multi_seed: bool = True
    seeds: int = 10
    text_file: str | None = None
    needle_loc: str = "beginning,middle,end"
    ctx_sizes: str = "8192,0,65536,131072,262144"
    kv_quants: str = "f16,q8_0,q4_0"
    max_tokens: int | None = None
    reasoning: bool | None = None
    max_tests_per_category: int | None = None


class BatchRunner:
    """Orchestrates batch testing across multiple models.

    Usage:
        runner = BatchRunner(BatchRunnerConfig(...))
        results = runner.run()
    """

    def __init__(self, config: BatchRunnerConfig):
        self.config = config
        self.cfg = None
        self.models = []
        self.batch_config = []
        self.plan = None  # BatchPlan from domain layer
        self.server_bin = None  # llama-server binary path
        self.output_path = None
        self.timestamp = None
        self.results_file = None
        self.csv_file = None
        self.all_results = []
        self._was_resumed = False

    def run(self) -> dict[str, Any]:
        """Run the batch test based on batch_type."""
        batch_type = self.config.batch_type

        if batch_type == "code":
            return self._run_code_batch()
        elif batch_type == "perplexity":
            return self._run_perplexity_batch()
        elif batch_type == "nihs":
            return self._run_nihs_batch()
        elif batch_type == "kv":
            return self._run_kv_batch()
        else:
            console.print(f"[red]Unknown batch type: {batch_type}[/red]")
            return {}

    def run_nihs_batch(
        self,
        config_dir: Path,
        locations: list[str],
        seed: int | None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run NIHS tests across multiple models from batch config.

        Returns nested dict: {model_name: {location: result}}
        """
        self._setup()

        text_path = config_dir / "wiki.train.tokens"
        if not text_path.exists():
            console.print(f"[red]Text file not found: {text_path}[/red]")
            return {}

        all_results: dict[str, dict[str, dict[str, Any]]] = {}

        for item in self.batch_config:
            model_name, config_item = self._parse_batch_item(item)
            if not model_name:
                continue

            model = self._find_model(model_name)
            if model is None:
                console.print(f"[red]Model not found: {model_name}[/red]")
                continue

            use_port = find_free_port(self.cfg)

            console.print(f"[cyan]=== {model.name} (port {use_port}) ===[/cyan]")

            start_ctx = 65536
            backend = config_item.get("backend", "llama.cpp")

            try:
                start_model(
                    ModelStartConfig(
                        model=model,
                        backend=backend,
                        port=use_port,
                        ctx_size=start_ctx,
                        seed=seed,
                        config=self.cfg,
                        config_item={**config_item, "ctk": "f16", "ctv": "f16"},
                        log_disable=True,
                    )
                )

                for location in locations:
                    nihs_results = run_needle_suite(
                        use_port,
                        [text_path],
                        [location],
                        seed=seed,
                        host=self.cfg.get("host", "127.0.0.1"),
                        quiet=True,
                    )

                    if nihs_results:
                        first_result = nihs_results[0]
                        found = first_result.get("found", 0)
                        total = first_result.get("total", 0)
                        score = first_result.get("score", 0)

                        if model_name not in all_results:
                            all_results[model_name] = {}

                        all_results[model_name][location] = {
                            "found": found,
                            "total": total,
                            "score": score,
                        }

                        console.print(
                            f"  {location}: {found}/{total} found (score: {score})"
                        )

            except Exception as e:
                console.print(f"[red]Error testing {model.name}: {e}[/red]")
            finally:
                try:
                    stop_model(use_port)
                except Exception as e:
                    logger.debug("failed to stop model on port %s: %s", use_port, e)

        return all_results

    def _setup(self) -> None:
        """Initialize batch runner: load config, models, output paths, domain plan."""
        from rama import Config

        self.cfg = Config(self.config.config_path)
        self.models = discover_models(self.cfg)

        self.batch_config = load_batch_items(self.config.config_file)

        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_path / f"batch_{self.timestamp}.json"
        self.csv_file = self.output_path / f"batch_{self.timestamp}.csv"
        self.all_results = []

        # Determine server binary
        backend_bin_name = "llama-server"
        if hasattr(self.cfg, "llama_cpp_dir") and self.cfg.llama_cpp_dir:
            candidate = self.cfg.llama_cpp_dir / "build" / "bin" / backend_bin_name
            if candidate.exists():
                self.server_bin = candidate
            else:
                self.server_bin = Path(backend_bin_name)
        else:
            self.server_bin = Path(backend_bin_name)

        # Build immutable domain plan with run identity and fixture hashes
        text_file_path = None
        if self.config.text_file:
            text_file_path = Path(self.config.text_file)

        planner_inp = BatchPlannerInput(
            models=self.models,
            batch_config_items=self.batch_config,
            batch_type=self.config.batch_type,
            server_bin=self.server_bin,
            host=self.cfg.get("host", "127.0.0.1"),
            start_port=self.config.start_port,
            seed=self.config.seed,
            config_path=self.config.config_path,
            text_file=text_file_path,
            max_tokens=self.config.max_tokens,
            reasoning=self.config.reasoning,
            seeds=self.config.seeds,
            multi_seed=self.config.multi_seed,
            global_config_override=self.cfg._data,
            family_overrides=self.cfg.families,
            model_overrides=self.cfg.get("models", {}),
        )
        self.plan = plan_batch(planner_inp)

        # Check for resumable batch
        from rama.domain.batch_runner import can_resume, load_manifest

        existing = load_manifest(self.output_path, self.plan.run_identity)
        if existing and can_resume(self.output_path, self.plan):
            self._was_resumed = True
            if not self.config.quiet:
                console.print(
                    f"[yellow]Batch {self.plan.run_identity} already completed "
                    f"({len(existing.results)} results). Loading cached results.[/yellow]"
                )
            self.all_results = existing.results

    def _save_code_results(self) -> None:
        """Persist the single aggregate result pair for a code batch.

        Results are checkpointed after each model so an interrupted long batch
        remains useful.  Seed-level results stay embedded in this aggregate
        file; separate files per seed make result directories hard to use.
        """
        assert self.results_file is not None
        assert self.csv_file is not None

        temp_file = self.results_file.with_suffix(f"{self.results_file.suffix}.tmp")
        temp_file.write_text(json.dumps(self.all_results, indent=2))
        temp_file.replace(self.results_file)
        save_results_csv(self.all_results, self.csv_file)

    def _run_code_batch(self) -> dict[str, Any]:
        """Code/debugging test batch with multi-seed support."""
        self._setup()

        if self._was_resumed:
            return {"results": self.all_results, "files": str(self.results_file)}

        if not self.config.quiet:
            console.print(
                f"[cyan]Batch running {len(self.batch_config)} models (code tests)...[/cyan]"
            )
            console.print(f"Results will be saved to {self.output_path}/\n")

        # Kill any leftover llama-server processes
        self._kill_leftover_servers()

        for item in self.batch_config:
            model_name, config_item = self._parse_batch_item(item)
            if not model_name:
                continue

            model = self._find_model(model_name)
            if model is None:
                console.print(f"[red]Model not found: {model_name}[/red]")
                continue

            self._run_single_code_model(model, config_item)

        # Final save + manifest
        self._save_code_results()
        self._write_manifest()

        if not self.config.quiet:
            console.print("[green]Batch complete![/green]")
            console.print(f"  JSON: {self.results_file}")
            console.print(f"  CSV:  {self.csv_file}")
            if self.plan:
                console.print(f"  Run ID: {self.plan.run_identity}")
            console.print(f"\n  Summary ({len(self.all_results)} models):")
            for r in self.all_results:
                score = r.get("score")
                verdict = r.get("evaluation", {}).get("verdict", "no data")
                console.print(
                    f"  {r['model'][:45]:<45} score={score:.1f}"
                    if score is not None
                    else f"  {r['model'][:45]:<45} score=N/A"
                )
                console.print(f"    verdict={verdict} ({r['test_time_seconds']:.0f}s)")

        return {"results": self.all_results, "files": str(self.results_file)}

    def _run_perplexity_batch(self) -> dict[str, Any]:
        """Perplexity batch (llama-perplexity)."""
        self._setup()

        if self._was_resumed:
            return {"results": self.all_results, "files": str(self.results_file)}

        text_path = Path(self.config.text_file)
        if not text_path.exists():
            console.print(f"[red]Text file not found: {self.config.text_file}[/red]")
            raise FileNotFoundError(self.config.text_file)

        backend_bin = self.cfg.llama_cpp_dir / "build" / "bin" / "llama-perplexity"
        if not backend_bin.exists():
            console.print(
                "[yellow]llama-perplexity not found. Build llama.cpp first.[/yellow]"
            )
            return {}

        ppl_results_file = self.output_path / f"pbatch_{self.timestamp}.json"
        ppl_csv_file = self.output_path / f"pbatch_{self.timestamp}.csv"
        ppl_all_results = []

        if not self.config.quiet:
            console.print(
                f"[cyan]Perplexity batch: {len(self.batch_config)} models, text: {text_path.name}[/cyan]"
            )
            console.print(f"Results: {ppl_results_file}\n")

        threads = int(self.cfg.get("threads", self.cfg.get("build_jobs", 8)))
        batch_size = int(self.cfg.get("batch_size", 2048))

        for item in self.batch_config:
            model_name, config_item = self._parse_batch_item(item)
            if not model_name:
                continue

            model = self._find_model(model_name)
            if model is None:
                console.print(f"[red]Model not found: {model_name}[/red]")
                continue

            use_ctx = config_item.get("ctx", self.config.ctx_size or 4096)
            use_batch = config_item.get("batch", batch_size)

            if not self.config.quiet:
                console.print(f"[cyan]=== {model.name} ===[/cyan]")
                console.print(f"  ctx={use_ctx}, batch={use_batch}")

            args = [
                str(backend_bin),
                "-m",
                str(model.path),
                "-f",
                str(text_path),
                "-c",
                str(use_ctx),
                "-t",
                str(threads),
                "-b",
                str(use_batch),
            ]
            if self.config.quiet:
                args.append("--log-disable")

            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=600, errors="replace"
                )
                ppl = self._parse_perplexity(result)

                if ppl is not None:
                    if not self.config.quiet:
                        console.print(f"  perplexity: {ppl:.4f}")
                    ppl_all_results.append(
                        {
                            "model": model.name,
                            "path": str(model.path),
                            "ctx": use_ctx,
                            "batch": use_batch,
                            "perplexity": round(ppl, 4),
                            "status": "ok",
                        }
                    )
                else:
                    console.print("  [yellow]No perplexity value parsed[/yellow]")
                    ppl_all_results.append(
                        {
                            "model": model.name,
                            "path": str(model.path),
                            "ctx": use_ctx,
                            "batch": use_batch,
                            "perplexity": None,
                            "status": "parse_fail",
                            "stderr": result.stderr[-1000:] + result.stdout[-1000:],
                        }
                    )
            except subprocess.TimeoutExpired:
                console.print("  [red]Timeout after 600s[/red]")
                ppl_all_results.append(
                    {
                        "model": model.name,
                        "ctx": use_ctx,
                        "perplexity": None,
                        "status": "timeout",
                    }
                )
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                ppl_all_results.append(
                    {
                        "model": model.name,
                        "ctx": use_ctx,
                        "perplexity": None,
                        "status": "error",
                        "error": str(e),
                    }
                )

            if not self.config.quiet:
                console.print("")

        ppl_results_file.write_text(json.dumps(ppl_all_results, indent=2))

        with open(ppl_csv_file, "w") as f:
            f.write("model,ctx,batch,perplexity,status\n")
            for r in ppl_all_results:
                ppl = r.get("perplexity")
                f.write(
                    f"{r['model']},{r['ctx']},{r.get('batch', '')},"
                    f"{ppl if ppl is not None else ''},{r['status']}\n"
                )

        if not self.config.quiet:
            console.print("[green]Perplexity batch complete![/green]")
            console.print(f"  JSON: {ppl_results_file}")
            console.print(f"  CSV:  {ppl_csv_file}")

            ok_results = [r for r in ppl_all_results if r["status"] == "ok"]
            if ok_results:
                console.print(f"\n  Summary ({len(ok_results)} models):")
                for r in ok_results:
                    console.print(f"  {r['model']}: {r['perplexity']:.4f}")

        # Write manifest with fixture hash (text file)
        if self.plan:
            self.all_results = ppl_all_results
            self._write_manifest()

        return {"results": ppl_all_results, "files": str(ppl_results_file)}

    def _run_nihs_batch(self) -> dict[str, Any]:
        """NIHS batch (multiple models, KV quants, context sizes)."""
        self._setup()

        if self._was_resumed:
            return {"results": self.all_results, "files": str(self.results_file)}

        text_path = Path(self.config.text_file)
        if not text_path.exists():
            console.print(f"[red]Text file not found: {self.config.text_file}[/red]")
            raise FileNotFoundError(self.config.text_file)

        ctx_size_list = sorted(int(s.strip()) for s in self.config.ctx_sizes.split(","))
        needle_locs = [loc.strip() for loc in self.config.needle_loc.split(",")]

        kv_quant_list = [
            q.strip()
            for q in (
                self.config.kv_quants.split(",")
                if self.config.kv_quants
                else ["f16", "q8_0", "q4_0", "turbo3", "turbo4"]
            )
        ]

        nihs_results_file = self.output_path / f"nihs_{self.timestamp}.json"
        nihs_csv_file = self.output_path / f"nihs_{self.timestamp}.csv"
        nihs_all_results = []

        start_ctx = max(ctx_size_list) + 15000

        if not self.config.quiet:
            console.print(
                f"[cyan]NIHS Batch: {len(self.batch_config)} models x {len(kv_quant_list)} kv-quants "
                f"x {len(ctx_size_list)} ctx sizes, text: {text_path.name}[/cyan]"
            )
            console.print(f"Ctx sizes: {ctx_size_list}")
            console.print(f"KV quants: {kv_quant_list}")
            console.print(f"Needle locations: {needle_locs}")
            console.print(f"Model start ctx: {start_ctx:,}")
            console.print(f"Results: {nihs_results_file}\n")

        for item in self.batch_config:
            model_name, config_item = self._parse_batch_item(item)
            if not model_name:
                continue

            model = self._find_model(model_name)
            if model is None:
                if not self.config.quiet:
                    console.print(f"[red]Model not found: {model_name}[/red]")
                continue

            use_port = find_free_port(self.cfg)

            if not self.config.quiet:
                console.print(f"[cyan]=== {model.name} (port {use_port}) ===[/cyan]")

            for kv_quant in kv_quant_list:
                result = None
                is_turbo = kv_quant in ("turbo3", "turbo4")
                backend = "turboquant" if is_turbo else "llama.cpp"

                try:
                    result = start_model(
                        ModelStartConfig(
                            model=model,
                            backend=backend,
                            port=use_port,
                            ctx_size=start_ctx,
                            seed=self.config.seed,
                            config=self.cfg,
                            config_item={
                                **config_item,
                                "ctk": kv_quant,
                                "ctv": kv_quant,
                            },
                            log_disable=self.config.quiet,
                        )
                    )

                    for ctx_size in ctx_size_list:
                        test_start = time.perf_counter()
                        nihs_results = run_needle_suite(
                            use_port,
                            [text_path],
                            needle_locs,
                            seed=self.config.seed,
                            host=self.cfg["host"],
                            max_tokens=ctx_size,
                            quiet=not self.config.quiet,
                        )
                        test_duration = time.perf_counter() - test_start

                        found_count = sum(1 for r in nihs_results if r.found)
                        total = len(nihs_results)
                        score = round(found_count / total, 2) if total > 0 else 0

                        result_data = {
                            "model": model.name,
                            "kv_quant": kv_quant,
                            "ctx_size": ctx_size,
                            "test_time_seconds": round(test_duration, 3),
                            "nihs_score": score,
                            "found_count": found_count,
                            "total_tests": total,
                            "results": [
                                {
                                    "location": r.needle_location,
                                    "found": r.found,
                                    "has_repetition": r.has_repetition,
                                    "is_looping": r.is_looping,
                                }
                                for r in nihs_results
                            ],
                        }
                        nihs_all_results.append(result_data)
                        nihs_results_file.write_text(
                            json.dumps(nihs_all_results, indent=2)
                        )

                        status = (
                            "[green]PASS[/green]" if score > 0.5 else "[red]FAIL[/red]"
                        )
                        locs = " ".join(
                            "[green]✓[/green]" if r.found else "[red]✗[/red]"
                            for r in nihs_results
                        )
                        if not self.config.quiet:
                            console.print(
                                f"  {model.name[:45]:<45} kv={kv_quant:<7} "
                                f"ctx={ctx_size:<7} {status} {locs} {test_duration:.0f}s"
                            )

                        if score <= 0.5:
                            if not self.config.quiet:
                                console.print(
                                    f"  [yellow]Failed at {ctx_size:,}, skipping larger sizes[/yellow]"
                                )
                            break

                except Exception as e:
                    if not self.config.quiet:
                        console.print(f"[red]Error: {e}[/red]")
                finally:
                    if result is not None:
                        stop_model(result.port, self.cfg)
                        if not _wait_for_port_free(result.port, timeout=10.0):
                            stop_model(result.port, self.cfg)
                            _wait_for_port_free(result.port, timeout=10.0)
                        if not self.config.quiet:
                            console.print("  Stopped")

        nihs_results_file.write_text(json.dumps(nihs_all_results, indent=2))

        with open(nihs_csv_file, "w") as f:
            f.write(
                "model,kv_quant,ctx_size,test_time_seconds,nihs_score,"
                "found_count,total_tests,beginning,middle,end\n"
            )
            for r in nihs_all_results:
                loc_results = {res["location"]: res["found"] for res in r["results"]}
                b = str(loc_results.get("beginning", ""))
                m = str(loc_results.get("middle", ""))
                e = str(loc_results.get("end", ""))
                f.write(
                    f"{r['model']},{r['kv_quant']},{r['ctx_size']},"
                    f"{r['test_time_seconds']},{r['nihs_score']},"
                    f"{r['found_count']},{r['total_tests']},{b},{m},{e}\n"
                )

        if not self.config.quiet:
            console.print("[green]NIHS Batch complete![/green]")
            console.print(f"  JSON: {nihs_results_file}")
            console.print(f"  CSV:  {nihs_csv_file}")

            console.print(f"\n  Summary ({len(nihs_all_results)} results):")
            for r in nihs_all_results:
                ctx_str = f"{r['ctx_size']:,}"
                console.print(
                    f"  {r['model'][:45]:<45} kv={r['kv_quant']:<7} "
                    f"ctx={ctx_str:<9} score={r['nihs_score']} "
                    f"({r['found_count']}/{r['total_tests']}) time={r['test_time_seconds']:.0f}s"
                )

        # Write manifest with fixture hash (text file)
        if self.plan:
            self.all_results = nihs_all_results
            self._write_manifest()

        return {"results": nihs_all_results, "files": str(nihs_results_file)}

    def _run_kv_batch(self) -> dict[str, Any]:
        """KV quant comparison batch."""
        self._setup()

        if self._was_resumed:
            return {"results": self.all_results, "files": str(self.results_file)}

        kv_quants_list = [q.strip() for q in self.config.kv_quants.split(",")]

        kv_results_file = self.output_path / f"batch_kv_{self.timestamp}.json"
        kv_csv_file = self.output_path / f"batch_kv_{self.timestamp}.csv"

        if not self.config.quiet:
            console.print(
                f"[cyan]Batch KV comparison: {len(self.batch_config)} models x {len(kv_quants_list)} quants[/cyan]\n"
            )

        for item in self.batch_config:
            model_name, config_item = self._parse_batch_item(item)
            if not model_name:
                continue

            model = self._find_model(model_name)
            if model is None:
                console.print(f"[red]Model not found: {model_name}[/red]")
                continue

            ctx_size = config_item.get("ctx", 0)

            for quant in kv_quants_list:
                use_port = find_free_port(self.cfg)

                if not self.config.quiet:
                    console.print(f"[cyan]{model.name} [{quant}] KV cache ===[/cyan]")

                try:
                    result = start_model(
                        ModelStartConfig(
                            model=model,
                            backend="llama.cpp",
                            port=use_port,
                            ctx_size=ctx_size,
                            ctk=quant,
                            ctv=quant,
                            config=self.cfg,
                            config_item=config_item,
                            log_disable=self.config.quiet,
                        )
                    )
                    if not self.config.quiet:
                        console.print(
                            f"  Started on port {result.port} with {quant} KV"
                        )

                    cats = (
                        self.config.categories.split(",")
                        if self.config.categories != "all"
                        else ["code", "debugging"]
                    )
                    test_start = time.perf_counter()
                    test_result = run_tests(
                        TestRunConfig(
                            port=result.port,
                            categories=cats,
                            config=self.cfg,
                            max_tests_per_category=self.config.max_tests_per_category,
                        )
                    )
                    test_duration = time.perf_counter() - test_start

                    result_data = {
                        "model": model.name,
                        "kv_quant": quant,
                        "ctx": ctx_size,
                        "test_time_seconds": round(test_duration, 3),
                        "score": test_result.overall_score,
                        "evaluation": summarize_evaluation(test_result.results),
                        "code_score": 0.0,
                        "debug_score": 0.0,
                    }
                    for r in test_result.results:
                        if r.category == "code" and r.score is not None:
                            result_data["code_score"] = r.score
                        elif r.category == "debugging" and r.score is not None:
                            result_data["debug_score"] = r.score

                    self.all_results.append(result_data)
                    kv_results_file.write_text(json.dumps(self.all_results, indent=2))

                    if not self.config.quiet:
                        console.print(
                            f"  Score: {test_result.overall_score:.1f}/10 "
                            f"(code={result_data['code_score']:.1f}, "
                            f"debug={result_data['debug_score']:.1f})"
                        )
                        evaluation = result_data["evaluation"]
                        console.print(
                            f"  Verdict: {evaluation['verdict']} "
                            f"(completion {evaluation['completion_rate']:.0%}, "
                            f"execution {evaluation['execution_rate']:.0%})"
                        )
                        console.print(f"  Test time: {test_duration:.1f}s")

                    stop_model(result.port, self.cfg)

                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

                if not self.config.quiet:
                    console.print("")

        kv_results_file.write_text(json.dumps(self.all_results, indent=2))
        save_results_csv(self.all_results, kv_csv_file)

        # Write manifest with resolved profiles
        if self.plan:
            self._write_manifest()

        if not self.config.quiet:
            console.print("[green]Batch KV comparison complete![/green]")
            console.print(f"  JSON: {kv_results_file}")
            console.print(f"  CSV:  {kv_csv_file}")
            if self.plan:
                console.print(f"  Run ID: {self.plan.run_identity}")

        return {"results": self.all_results, "files": str(kv_results_file)}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _write_manifest(self) -> None:
        """Write batch manifest with resolved profiles and fixture hashes."""
        if self.plan is None:
            return

        from rama.domain.batch_runner import BatchExecutorResult
        from rama.domain.batch_runner import (
            _write_manifest as domain_write_manifest,
        )

        results = [
            BatchExecutorResult(
                item=self.plan.items[i],
                result=r,
                status="ok" if "error" not in r else "failed",
                duration_seconds=r.get("test_time_seconds", 0.0),
            )
            for i, r in enumerate(self.all_results)
            if i < len(self.plan.items)
        ]
        domain_write_manifest(self.plan, results, self.output_path)

    def _parse_batch_item(self, item: Any) -> tuple[str, dict]:
        """Parse a batch config item into (model_name, config_item)."""
        if isinstance(item, str):
            return item, {}
        else:
            return item.get("model", ""), {
                k: v for k, v in item.items() if k != "model"
            }

    def _find_model(self, model_name: str) -> Any | None:
        """Find model by name with fuzzy matching."""
        # Exact match
        for m in self.models:
            if model_name in (m.name, m.path.name):
                return m

        # Case-insensitive
        for m in self.models:
            if (
                m.name.lower() == model_name.lower()
                or m.path.name.lower() == model_name.lower()
            ):
                return m

        # Fuzzy substring
        for m in self.models:
            if (
                model_name.lower() in m.name.lower()
                or model_name.lower() in m.path.name.lower()
            ):
                if not self.config.quiet:
                    console.print(
                        f"[yellow]Warning: fuzzy match '{model_name}' -> '{m.name}'[/yellow]"
                    )
                return m

        return None

    def _kill_leftover_servers(self) -> None:
        """Kill any leftover llama-server processes."""
        try:
            pids = (
                subprocess.run(
                    ["pgrep", "-f", "llama-server"],
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .split()
            )
            for p in pids:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(int(p), 9)
            if pids:
                time.sleep(1)
        except (subprocess.SubprocessError, PermissionError):
            pass

    def _parse_perplexity(self, result: subprocess.CompletedProcess) -> float | None:
        """Parse perplexity value from llama-perplexity output."""
        output = result.stderr + result.stdout

        for line in output.splitlines():
            ll = line.lower().strip()
            if "final estimate" in ll and "ppl" in ll:
                if "PPL =" in line:
                    idx = line.index("PPL =")
                    rest = line[idx + len("PPL =") :].strip()
                    try:
                        return float(rest.split()[0])
                    except (ValueError, IndexError):
                        pass
            elif "final estimate" in ll and "=" in ll:
                parts = line.split("=")
                for p in parts:
                    p = p.strip()
                    try:
                        val = float(p)
                        if 0 < val < 1000:
                            return val
                    except ValueError:
                        pass
        return None

    def _build_system_prompts(self, model: Any, cats: list[str]) -> dict[str, str]:
        cat_prompts = {}
        if self.config.system_prompt:
            for cat in cats:
                cat_prompts[cat] = self.config.system_prompt
        else:
            for cat in cats:
                if cat == "code":
                    cat_prompts[cat] = get_code_system_prompt(model.name)
                elif cat == "debugging":
                    cat_prompts[cat] = get_debug_system_prompt(model.name)
                else:
                    cat_prompts[cat] = get_system_prompt(model.name, cat)
        return cat_prompts

    def _build_context_payload(self) -> tuple[str | None, list[str] | None]:
        if self.config.difficulty == "easy":
            return None, None

        diff_config = {
            "easy": {"wiki_tokens": 0, "code_files": []},
            "medium": {"wiki_tokens": 4000, "code_files": []},
            "hard": {
                "wiki_tokens": 8000,
                "code_files": ["tests/code/*.txt", "count: 3"],
            },
            "expert": {
                "wiki_tokens": 16000,
                "code_files": [
                    "tests/code/*.txt",
                    "tests/debugging/*.txt",
                ],
            },
        }
        level_cfg = diff_config.get(self.config.difficulty, diff_config["easy"])

        context_text = None
        wiki_path = Path("wikitext-2/wiki.valid.tokens")
        if level_cfg["wiki_tokens"] > 0 and wiki_path.exists():
            full = wiki_path.read_text()
            max_chars = level_cfg["wiki_tokens"] * 4
            if len(full) > max_chars:
                truncated = full[:max_chars]
                last_space = truncated.rfind(" ")
                if last_space > max_chars // 2:
                    truncated = truncated[:last_space]
                context_text = truncated
            else:
                context_text = full

        code_contents = []
        for pattern_spec in level_cfg.get("code_files", []):
            if pattern_spec.startswith("tests/"):
                parts = [p.strip() for p in pattern_spec.split(",")]
                glob_pattern = parts[0]
                count = None
                for part in parts[1:]:
                    if part.startswith("count:"):
                        count = int(part.split(":")[1].strip())
                        break
                files = sorted(
                    Path(glob_pattern).parent.glob(glob_pattern.split("/")[-1])
                )
                files = [f for f in files if f.suffix == ".txt"]
                if count is not None and len(files) > count:
                    random.shuffle(files)
                    files = files[:count]
                for f in files:
                    code_contents.append(f.read_text())

        return context_text, (code_contents if code_contents else None)

    def _seed_runs_for_config(self, cats: list[str]) -> list[dict[str, Any]]:
        seeds = range(1, self.config.seeds + 1) if self.config.multi_seed else [self.config.seed]
        return [
            {
                "run_type": "family",
                "seed": seed,
                "categories": cats,
            }
            for seed in seeds
        ]

    def _single_code_run_summary(
        self,
        run_type: str,
        run_seed: int,
        sampler_args: dict | None,
        run_results: list[Any],
        run_start: float,
    ) -> dict[str, Any]:
        from rama.testing import sampler_fields_from_args

        scored_results = [r.score for r in run_results if r.score is not None]
        sampling = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repeat_penalty": None,
        }
        sampling.update(sampler_fields_from_args(sampler_args))
        return {
            "run_type": run_type,
            "seed": run_seed,
            "sampling": sampling,
            "test_time_seconds": round(time.perf_counter() - run_start, 3),
            "score": (
                sum(scored_results) / len(scored_results) if scored_results else None
            ),
            "tests": len(run_results),
            "passed": sum(r.correct for r in run_results),
            "executed": sum(r.ran for r in run_results),
            "empty": sum(not r.response.strip() for r in run_results),
        }

    def _print_single_seed_results(
        self, test_result: Any, run_type: str, run_seed: int
    ) -> None:
        for r in test_result.results:
            label = f"{r.name} ({run_type} s{run_seed})"
            if r.score is None:
                status = "N/A"
            elif r.correct:
                status = f"PASS {r.score:.1f}/10"
            elif not r.response.strip():
                status = "EMPTY 0/10"
            elif not r.ran:
                status = f"NO-RUN {r.score:.1f}/10"
            else:
                status = f"FAIL {r.score:.1f}/10"
            console.print(f"    {label}: {status}")

    def _run_single_seed_tests(
        self,
        port: int,
        run_type: str,
        run_seed: int,
        run_categories: list[str],
        cat_prompts: dict[str, str],
        context_text: str | None,
        context_files: list[str] | None,
        sampler_args: dict | None,
        effective_reasoning: bool | None,
        reasoning_template_key: str,
        reasoning_template_value: Any,
        max_tokens: int,
        run_start: float,
    ) -> tuple[list[Any], dict[str, Any]]:
        try:
            test_result = run_tests(
                TestRunConfig(
                    port=port,
                    categories=run_categories,
                    system_prompt=cat_prompts,
                    warmup_rounds=0,
                    context_text=context_text,
                    context_files=context_files,
                    context_placement=self.config.placement,
                    sampler_args=sampler_args,
                    seed=run_seed,
                    max_tokens=max_tokens,
                    reasoning=effective_reasoning,
                    reasoning_template_key=reasoning_template_key,
                    reasoning_template_value=reasoning_template_value,
                    max_tests_per_category=self.config.max_tests_per_category,
                )
            )
            run_results = []
            for r in test_result.results:
                r.metadata = {
                    "run_type": run_type,
                    "seed": run_seed,
                    "test_time": time.perf_counter() - run_start,
                }
                run_results.append(r)

            summary = self._single_code_run_summary(
                run_type, run_seed, sampler_args, run_results, run_start
            )
            if not self.config.quiet:
                self._print_single_seed_results(test_result, run_type, run_seed)
            return run_results, summary
        except Exception as e:
            if not self.config.quiet:
                console.print(
                    f"    [red]Run seed={run_seed} ({run_type}) failed: {e}[/red]"
                )
            return [], {
                "run_type": run_type,
                "seed": run_seed,
                "status": "failed",
                "error": str(e),
            }

    def _single_code_failure_data(
        self,
        model: Any,
        backend: str,
        ctk: str | None,
        ctv: str | None,
        ctx_size: Any,
        effective_reasoning: bool | None,
        mtp_enabled: bool,
        error: Exception,
    ) -> dict[str, Any]:
        return {
            "model": model.name,
            "backend": backend,
            "ctk": ctk or "q8_0",
            "ctv": ctv or "q8_0",
            "ctx": ctx_size or "auto",
            "reasoning": effective_reasoning,
            "mtp": mtp_enabled,
            "status": "launch_failed",
            "error": str(error),
            "runs": [],
            "sampled_runs": 0,
            "score": None,
            "score_mean": None,
            "score_stddev": 0.0,
            "test_time_seconds": 0.0,
            "evaluation": summarize_evaluation([]),
            "tests": [],
        }

    def _build_single_code_result_data(
        self,
        model: Any,
        config_item: dict,
        backend: str,
        ctk: str | None,
        ctv: str | None,
        ctx_size: Any,
        effective_reasoning: bool | None,
        mtp_enabled: bool,
        run_summaries: list[dict[str, Any]],
        all_test_results: list[Any],
        test_duration: float,
    ) -> tuple[dict[str, Any], Any, float | None, float]:
        from .base import TestSuiteResult

        sampled_results = all_test_results
        sampled_run_scores = [
            float(run["score"])
            for run in run_summaries
            if run.get("run_type") == "family" and run.get("score") is not None
        ]
        score_mean = (
            sum(sampled_run_scores) / len(sampled_run_scores)
            if sampled_run_scores
            else None
        )
        score_stddev = (
            statistics.stdev(sampled_run_scores)
            if len(sampled_run_scores) > 1
            else 0.0
        )
        test_result = TestSuiteResult(
            model_name=model.name,
            backend=backend,
            results=sampled_results,
            overall_score=score_mean,
        )
        result_data = {
            "model": model.name,
            "backend": backend,
            "ctk": ctk or "q8_0",
            "ctv": ctv or "q8_0",
            "ctx": ctx_size or "auto",
            "reasoning": effective_reasoning,
            "mtp": mtp_enabled,
            "runs": run_summaries,
            "test_time_seconds": round(test_duration, 3),
            "score": round(score_mean, 4) if score_mean is not None else None,
            "score_mean": (
                round(score_mean, 4) if score_mean is not None else None
            ),
            "score_stddev": round(score_stddev, 4),
            "sampled_runs": len(sampled_run_scores),
            "evaluation": summarize_evaluation(sampled_results),
            "tests": [
                {
                    "name": r.name,
                    "category": r.category,
                    "prompt": r.prompt,
                    "response": r.response,
                    "ran": r.ran,
                    "correct": r.correct,
                    "score": r.score,
                    "run_type": r.metadata.get("run_type", "unknown"),
                    "seed": r.metadata.get("seed", config_item.get("seed", 42)),
                }
                for r in all_test_results
            ],
        }
        return result_data, test_result, score_mean, score_stddev

    def _run_code_tests_on_started_model(
        self,
        model: Any,
        result: Any,
        config_item: dict,
        backend: str,
        ctk: str | None,
        ctv: str | None,
        ctx_size: Any,
        effective_reasoning: bool | None,
        mtp_enabled: bool,
    ) -> None:
        if self.config.warmup > 0:
            if not self.config.quiet:
                console.print(f"  Warming up ({self.config.warmup} rounds)...")
            warmup_model(
                result.port, host=self.cfg["host"], messages=self.config.warmup
            )

        cats = (
            self.config.categories.split(",")
            if self.config.categories != "all"
            else ["code", "debugging"]
        )
        cat_prompts = self._build_system_prompts(model, cats)
        context_text, context_files = self._build_context_payload()

        profile = result.profile
        sampler_args = dict(profile.args) if profile is not None else None
        reasoning_template_key = config_item.get(
            "reasoning_template_key", "enable_thinking"
        )
        seed_runs = self._seed_runs_for_config(cats)
        reasoning_template_value = config_item.get("reasoning_template_value")

        all_test_results = []
        run_summaries = []
        host = self.cfg.get("host", "127.0.0.1")
        test_start = time.perf_counter()
        model_port = result.port

        for i, run_cfg in enumerate(seed_runs):
            run_seed = run_cfg["seed"]
            run_type = run_cfg["run_type"]
            run_categories = run_cfg["categories"]
            run_start = time.perf_counter()

            if i > 0:
                reset_context(
                    model_port,
                    system_prompt=cat_prompts.get(
                        run_categories[0], "You are a helpful assistant."
                    ),
                    host=host,
                )
                if self.config.warmup > 0:
                    warmup_model(model_port, host=host, messages=self.config.warmup)

            if not self.config.quiet:
                console.print(f"  Running seed={run_seed} ({run_type})")

            run_results, summary = self._run_single_seed_tests(
                port=model_port,
                run_type=run_type,
                run_seed=run_seed,
                run_categories=run_categories,
                cat_prompts=cat_prompts,
                context_text=context_text,
                context_files=context_files,
                sampler_args=sampler_args,
                effective_reasoning=effective_reasoning,
                reasoning_template_key=reasoning_template_key,
                reasoning_template_value=reasoning_template_value,
                max_tokens=config_item.get("max_tokens") or self.config.max_tokens or 1024,
                run_start=run_start,
            )
            all_test_results.extend(run_results)
            run_summaries.append(summary)

        test_duration = time.perf_counter() - test_start
        result_data, test_result, _, score_stddev = self._build_single_code_result_data(
            model,
            config_item,
            backend,
            ctk,
            ctv,
            ctx_size,
            effective_reasoning,
            mtp_enabled,
            run_summaries,
            all_test_results,
            test_duration,
        )
        self.all_results.append(result_data)
        self._save_code_results()

        if not self.config.quiet:
            score = test_result.overall_score
            console.print(
                f"  Score: {score:.2f} ± {score_stddev:.2f}/10"
                if score is not None
                else "  Score: N/A"
            )
            evaluation = result_data["evaluation"]
            console.print(
                "  Verdict: "
                f"{evaluation['verdict']} "
                f"(completion {evaluation['completion_rate']:.0%}, "
                f"execution {evaluation['execution_rate']:.0%})"
            )
            console.print(f"  Test time: {test_duration:.2f}s")

    def _run_single_code_model(self, model: Any, config_item: dict) -> None:
        """Run code tests on a single model with multi-seed support."""
        # Determine backend
        ctk = config_item.get("ctk")
        ctv = config_item.get("ctv")
        ctx_size = config_item.get("ctx")
        turbo3 = config_item.get("turbo3", False)
        turbo4 = config_item.get("turbo4", False)
        model_reasoning = config_item.get("reasoning")
        effective_reasoning = (
            model_reasoning if model_reasoning is not None else self.config.reasoning
        )

        backend, ctk, ctv = detect_backend(turbo3, turbo4, ctk, ctv)

        use_port = find_free_port(self.cfg)
        # MTP must be explicitly enabled in the batch config. Filename suffixes
        # are not reliable evidence that the GGUF actually contains draft heads.
        mtp_enabled = bool(config_item.get("mtp", False))

        if not self.config.quiet:
            reasoning_label = (
                "auto"
                if effective_reasoning is None
                else "on"
                if effective_reasoning
                else "off"
            )
            console.print(f"[cyan]=== {model.name} ===[/cyan]")
            console.print(
                f"  Backend: {backend}, ctk={ctk or 'q8_0'}, "
                f"ctv={ctv or 'q8_0'}, ctx={ctx_size}, "
                f"reasoning={reasoning_label}, mtp={'on' if mtp_enabled else 'off'}",
            )

        result = None
        try:
            # Start model
            result = start_model(
                ModelStartConfig(
                    model=model,
                    backend=backend,
                    port=use_port,
                    ctx_size=ctx_size,
                    ctk=ctk,
                    ctv=ctv,
                    seed=config_item.get("seed"),
                    config=self.cfg,
                    config_item=config_item,
                    log_disable=self.config.quiet,
                    reasoning=model_reasoning,
                    reasoning_budget=config_item.get("reasoning_budget"),
                    mtp=mtp_enabled,
                    mtp_n_max=_optional_int(config_item.get("mtp_n_max")),
                    mtp_n_min=_optional_int(config_item.get("mtp_n_min")),
                )
            )
            if not self.config.quiet:
                console.print(f"  Started on port {result.port}")

            if self.config.dry_run:
                result_data = {
                    "model": model.name,
                    "backend": backend,
                    "ctk": ctk or "q8_0",
                    "ctv": ctv or "q8_0",
                    "ctx": ctx_size or "auto",
                    "mtp": mtp_enabled,
                    "seed": result.seed,
                    "status": "ok",
                    "message": "dry_run - model loaded and stopped",
                }
                self.all_results.append(result_data)
                self._save_code_results()
                if not self.config.quiet:
                    console.print("  OK (loaded & stopped)")
            else:
                self._run_code_tests_on_started_model(
                    model,
                    result,
                    config_item,
                    backend,
                    ctk,
                    ctv,
                    ctx_size,
                    effective_reasoning,
                    mtp_enabled,
                )

        except Exception as e:
            # A single model can be unavailable (or be killed by the OS) in a
            # long batch.  Record that failure and continue to the next model
            # instead of discarding the rest of the run.
            self.all_results.append(
                self._single_code_failure_data(
                    model, backend, ctk, ctv, ctx_size, effective_reasoning, mtp_enabled, e
                )
            )
            self._save_code_results()
            logger.error("Model %s could not be tested: %s", model.name, e)
            if not self.config.quiet:
                console.print(f"  [red]Model failed to start: {e}[/red]")
        finally:
            # Ensure stopped
            if result is not None:
                try:
                    stop_model(result.port, self.cfg)
                    _wait_for_port_free(result.port, timeout=10.0)
                except Exception as e:
                    logger.debug("failed to stop model on port %s: %s", result.port, e)
