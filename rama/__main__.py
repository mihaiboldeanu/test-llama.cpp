import json
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import DEFAULT_CONFIG, Config
from .context_test import format_context_results, run_needle_suite
from .core import calc_kv_cache, detect_backend, discover_models, get_running_models, is_port_used
from .launch import build_backend, start_model, stop_model
from .testing import (
    discover_tests,
    format_markdown,
    generate_online_eval_prompt,
    run_tests,
    save_results_csv,
)
from .testing_modules.batch_runner import BatchRunner, BatchRunnerConfig

console = Console()

# Default test categories for "all"
DEFAULT_TEST_CATEGORIES = ["code", "debugging"]


def get_config(config_path: str | None = None) -> Config:
    return Config(config_path)


app = typer.Typer(name="rama", help="CLI to test and run llama.cpp models")


@app.command()
def list(
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """List available models."""
    cfg = get_config(config_path)
    models = discover_models(cfg)

    if not models:
        console.print("[yellow]No models found in[/yellow]", cfg.model_dir)
        return

    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Quant")
    table.add_column("Family")
    table.add_column("Tags")

    for m in models:
        table.add_row(
            m.name[:40],
            f"{m.size_gb}GB",
            m.quant,
            m.family,
            ", ".join(m.tags) if m.tags else "-",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models[/dim]")


@app.command()
def running(
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """Show currently running models."""
    cfg = get_config(config_path)
    models = get_running_models(cfg)

    if not models:
        console.print("[yellow]No running models[/yellow]")
        return

    table = Table(title="Running Models")
    table.add_column("Name")
    table.add_column("Port", justify="right")
    table.add_column("PID", justify="right")
    table.add_column("Backend")

    for m in models:
        table.add_row(m.name, str(m.port), str(m.pid), m.backend)

    console.print(table)


@app.command()
def start(
    model_name: str = typer.Argument(..., help="Model name or alias"),
    port: int | None = typer.Option(None, "--port", "-p", help="Port to use"),
    turbo3: bool = typer.Option(
        False,
        "--turbo3",
        help="Use TurboQuant with turbo3 KV cache",
    ),
    turbo4: bool = typer.Option(
        False,
        "--turbo4",
        help="Use TurboQuant with turbo4 KV cache",
    ),
    ctx_size: int | None = typer.Option(
        None,
        "--ctx",
        help="Context size (auto calculated if not set)",
    ),
    fit: bool = typer.Option(
        True,
        "--fit/--no-fit",
        help="Auto-fit model to VRAM (default: on)",
    ),
    foreground: bool = typer.Option(
        False,
        "--foreground",
        "-f",
        help="Run server in foreground (Ctrl+C to stop)",
    ),
    ctk: str | None = typer.Option(
        None,
        "--ctk",
        help="KV cache type: turbo3, turbo4, q8_0, q4_k_m, etc.",
    ),
    ctv: str | None = typer.Option(
        None,
        "--ctv",
        help="KV cache v type: turbo3, turbo4, q8_0, q4_k_m, etc.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Random seed for generation",
    ),
    mtp: bool = typer.Option(
        False,
        "--mtp",
        help="Enable Multi-Token Prediction (requires model with MTP heads, e.g. Qwen3.5)",
    ),
    mtp_n_max: int = typer.Option(
        3,
        "--mtp-n-max",
        help="Max draft tokens for MTP (default: 3, recommended 2-3 for Qwen)",
    ),
    mtp_n_min: int = typer.Option(
        0,
        "--mtp-n-min",
        help="Min draft tokens for MTP (default: 0)",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """Start a model."""
    cfg = get_config(config_path)
    models = discover_models(cfg)

    # Find model
    model = None
    for m in models:
        if model_name in m.name or model_name in m.path.name:
            model = m
            break

    if model is None:
        console.print(f"[red]Model not found: {model_name}[/red]")
        console.print("\nAvailable models:")
        for m in models:
            console.print(f"  {m.name}")
        raise typer.Exit(1)

    # Determine backend and KV types
    backend, ctk, ctv = detect_backend(turbo3, turbo4, ctk, ctv)

    # Fit mode options
    config_item = {}
    config_item["fit"] = fit
    if seed is not None:
        config_item["seed"] = seed

    # Show what context we'll get
    if ctx_size is None:
        ctx, qk, qv = calc_kv_cache(model.size_gb, model.family, cfg)
        console.print(
            f"[dim]Model: {model.name} ({model.size_gb}GB, {model.family})[/dim]",
        )
        console.print(f"[dim]Auto ctx={ctx}, ctk={qk}, ctv={qv}[/dim]")
        if fit:
            console.print("[dim]--fit: auto-fit model to VRAM[/dim]")

    try:
        result = start_model(
            model,
            backend=backend,
            port=port,
            ctx_size=ctx_size,
            ctk=ctk,
            ctv=ctv,
            seed=seed,
            config=cfg,
            config_item=config_item,
            foreground=foreground,
            mtp=mtp,
            mtp_n_max=mtp_n_max,
            mtp_n_min=mtp_n_min,
        )
        if foreground:
            console.print("\n[cyan]Server stopped.[/cyan]")
            return
        console.print("\n[green]Model started![/green]")
        console.print(f"  Name: {result.name}")
        console.print(f"  Port: {result.port}")
        console.print(f"  PID: {result.pid}")
        console.print(f"  Backend: {result.backend}")
        console.print(f"  Context: {result.ctx_size}")
        console.print(
            f"\n  Chat: http://{cfg['host']}:{result.port}/v1/chat/completions"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stop(
    port: int = typer.Argument(..., help="Port to stop"),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """Stop a model."""
    cfg = get_config(config_path)

    if stop_model(port, cfg):
        console.print(f"[green]Stopped model on port {port}[/green]")
    else:
        console.print(f"[red]No model on port {port}[/red]")
        raise typer.Exit(1)


@app.command()
def build(
    backend: str = typer.Argument(..., help="Backend: llama.cpp, turboquant, or all"),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force rebuild even if up to date",
    ),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="CPU-only build (no CUDA)"),
) -> None:
    """Build backend."""
    cfg = get_config(config_path)

    if backend == "all":
        backends = ["llama.cpp", "turboquant"]
    elif backend in ["llama.cpp", "turboquant"]:
        backends = [backend]
    else:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        console.print("Use: llama.cpp, turboquant, or all")
        raise typer.Exit(1)

    for b in backends:
        try:
            build_backend(b, cfg, force=force, cpu_only=cpu_only)
        except Exception as e:
            console.print(f"[red]Build failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def status(
    port: int = typer.Argument(..., help="Check if port is in use"),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """Check if port is in use."""
    if is_port_used(port):
        console.print(f"[red]Port {port} is in use[/red]")
    else:
        console.print(f"[green]Port {port} is free[/green]")


@app.command()
def init(
    config_path: str = typer.Option(
        "rama.yaml",
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """Create a config file and clone missing backend repos."""
    src = Path(__file__).parent / "templates" / "rama.yaml"
    dst = Path(config_path).resolve()

    if dst.exists():
        console.print(f"[yellow]Config already exists: {dst}[/yellow]")
        return

    # Load config from template or defaults
    if src.exists():
        with open(src) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = DEFAULT_CONFIG.copy()

    # Clone llama.cpp if missing (use config defaults for paths)
    llama_cpp_dir = Path(
        DEFAULT_CONFIG.get("llama_cpp_dir", str(Path.home() / "Projects" / "llama.cpp"))
    )
    if not llama_cpp_dir.exists():
        console.print(f"[cyan]Cloning llama.cpp to {llama_cpp_dir}...[/cyan]")
        subprocess.run(
            [
                "git",
                "clone",
                DEFAULT_CONFIG.get(
                    "llama_cpp_repo", "https://github.com/ggml-org/llama.cpp.git"
                ),
                str(llama_cpp_dir),
            ],
            check=True,
        )
        console.print("[green]Cloned llama.cpp[/green]")
    else:
        console.print(f"[dim]llama.cpp already exists at {llama_cpp_dir}[/dim]")

    # Clone turboquant if missing
    turbo_dir = Path(
        DEFAULT_CONFIG.get(
            "turbo_dir", str(Path.home() / "Projects" / "llama-cpp-turboquant")
        )
    )
    if not turbo_dir.exists():
        console.print(f"[cyan]Cloning turboquant to {turbo_dir}...[/cyan]")
        subprocess.run(
            [
                "git",
                "clone",
                DEFAULT_CONFIG.get(
                    "turboquant_repo",
                    "https://github.com/TheTom/llama-cpp-turboquant.git",
                ),
                str(turbo_dir),
            ],
            check=True,
        )
        console.print("[green]Cloned turboquant[/green]")
    else:
        console.print(f"[dim]turboquant already exists at {turbo_dir}[/dim]")

    # Update config with actual paths and write
    config["llama_cpp_dir"] = str(llama_cpp_dir)
    config["turbo_dir"] = str(turbo_dir)

    with open(dst, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[green]Created config: {dst}[/green]")
    console.print("[dim]Edit it to customize paths![/dim]")


@app.command()
def test(
    port: int = typer.Argument(..., help="Port of running model"),
    categories: str = typer.Option(
        "all",
        "--categories",
        "-c",
        help="Comma-separated: code,debugging,creative",
    ),
    judge_port: int | None = typer.Option(
        None,
        "--judge",
        "-j",
        help="Port of judge model for scoring",
    ),
    no_judge: bool = typer.Option(
        False,
        "--no-judge",
        help="Skip LLM-as-judge scoring",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to file",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, markdown, json, eval",
    ),
) -> None:
    """Run tests on a running model."""
    # Parse categories
    cats = DEFAULT_TEST_CATEGORIES if categories == "all" else categories.split(",")

    console.print(f"[cyan]Running tests on port {port}...[/cyan]")
    if not no_judge and judge_port:
        console.print(f"[cyan]Using judge on port {judge_port}[/cyan]")

    # Run tests
    result = run_tests(
        port,
        judge_port=judge_port,
        categories=cats,
        use_llm_judge=not no_judge,
    )

    if output:
        output_path = Path(output)
        output_path.write_text(format_markdown(result))
        console.print(f"[green]Results saved to {output_path}[/green]")

    if format == "eval":
        # Generate copy/paste for online LLM eval
        console.print("\n[yellow]Copy the following for online evaluation:[/yellow]\n")
        console.print(generate_online_eval_prompt(result.results))
    elif format == "json":
        console.print(
            json.dumps(
                {
                    "overall_score": result.overall_score,
                    "tests": [(r.name, r.category, r.score) for r in result.results],
                },
                indent=2,
            ),
        )
    elif format == "markdown":
        console.print(format_markdown(result))
    else:
        # Table
        table = Table(title=f"Test Results (port {port})")
        table.add_column("Test")
        table.add_column("Category")
        table.add_column("Score", justify="right")

        for r in result.results:
            score_str = f"{r.score}/10" if r.score is not None else "N/A"
            table.add_row(r.name, r.category, score_str)

        console.print(table)
        console.print(f"\n[green]Overall: {result.overall_score:.1f}/10[/green]")


@app.command()
def tests(
    show: bool = typer.Option(False, "--show", "-s", help="Show available tests"),
) -> None:
    """List available tests."""
    tests_dict = discover_tests()

    for category, test_files in tests_dict.items():
        console.print(f"\n[cyan]{category.upper()}:[/cyan]")
        for f in test_files:
            console.print(f"  {f.stem}")


@app.command()
def ctxinfo(
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """Show context info for all available models."""
    cfg = get_config(config_path)
    models = discover_models(cfg)

    if not models:
        console.print("[yellow]No models found[/yellow]")
        return

    table = Table(title="Model Context Info")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Family")
    table.add_column("Max Ctx")
    table.add_column("KV Type")

    for m in models:
        ctx, qk, qv = calc_kv_cache(m.size_gb, m.family, cfg)
        kv = f"{qk}/{qv}"
        table.add_row(m.name[:35], f"{m.size_gb}GB", m.family, f"{ctx:,}", kv)

    console.print(table)
    console.print(
        f"\n[dim]Based on {cfg.vram_available_gb}GB available VRAM ({cfg['vram_total_gb']}GB - {cfg['vram_headroom_gb']}GB headroom)[/dim]",
    )


@app.command()
def context(
    port: int = typer.Argument(..., help="Port of running model"),
    files: str = typer.Argument(..., help="Comma-separated paths to text files"),
    needle_loc: str = typer.Option(
        "beginning,middle,end",
        "--needle-loc",
        "-n",
        help="Where to hide the needle: beginning, middle, end, or percentage (e.g., 25%, 75%). Comma-separated for multiple locations.",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for needle generation"
    ),
) -> None:
    """Needle-in-a-haystack test - hide a token in text and see if the model finds it.

    Examples:
      rama context 11435 book.txt -n beginning
      rama context 11435 book.txt -n beginning,25%,middle,75%,end
    """
    # Parse files
    file_paths = [Path(f.strip()) for f in files.split(",")]
    for f in file_paths:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)

    # Parse needle locations
    locations = [loc.strip() for loc in needle_loc.split(",")]

    console.print("[cyan]Running needle-in-haystack tests...[/cyan]")
    console.print(f"Files: {', '.join(f.name for f in file_paths)}")
    console.print(f"Needle locations: {locations}")

    results = run_needle_suite(port, file_paths, locations, seed=seed)

    console.print("\n" + format_context_results(results))


@app.command()
def nihs(
    port: int | None = typer.Argument(None, help="Port of running model (context mode)"),
    file_or_config: str | None = typer.Argument(None, help="Text file (context mode) or batch config (batch mode)"),
    needle_loc: str = typer.Option(
        "beginning,middle,end",
        "--needle-loc",
        "-n",
        help="Where to hide the needle: beginning, middle, end, or percentage (e.g., 25%, 75%). Comma-separated for multiple locations.",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for needle generation"
    ),
    enhanced: bool = typer.Option(
        False, "--enhanced", help="Run enhanced NIHS with multi-needle and distractors"
    ),
    difficulty: str = typer.Option(
        "easy", "--difficulty", "-d", help="Difficulty level: easy, medium, hard, expert, or all"
    ),
    model_name: str | None = typer.Option(
        None, "-m", "--model", help="Model name for enhanced/difficulty mode"
    ),
    ctx_size: int | None = typer.Option(
        None, "--ctx", help="Context size for enhanced/difficulty mode"
    ),
    num_needles: int = typer.Option(
        8, "--num-needles", help="Number of needles for enhanced mode"
    ),
    distractor_types: str = typer.Option(
        "lexical,topical,irrelevant", "--distractors", help="Comma-separated distractor types"
    ),
    config_path: str | None = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Needle-in-haystack test - retrieve information from long contexts.

    Supports four modes:
    - Context mode: test a running model on a text file
    - Batch mode: test multiple models from a batch config
    - Enhanced mode: multi-needle with distractors (auto-starts model)
    - Difficulty mode: inject context noise at various levels (auto-starts model)
    """
    cfg = get_config(config_path)

    # Enhanced mode
    if enhanced:
        _run_enhanced_nihs(cfg, model_name, ctx_size, num_needles, distractor_types, seed)
        return

    # Difficulty mode
    if difficulty != "easy" or (model_name and not port):
        _run_difficulty_nihs(cfg, model_name, ctx_size, difficulty, seed)
        return

    # Batch mode
    if port is None and file_or_config and Path(file_or_config).suffix == ".yaml":
        _run_batch_nihs(cfg, file_or_config, needle_loc, seed)
        return

    # Context mode (default)
    if port is None or file_or_config is None:
        console.print("[red]Context mode requires: rama nihs <port> <file>[/red]")
        raise typer.Exit(1)

    file_paths = [Path(f.strip()) for f in file_or_config.split(",")]
    for f in file_paths:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)

    locations = [loc.strip() for loc in needle_loc.split(",")]
    console.print("[cyan]Running needle-in-haystack tests...[/cyan]")
    console.print(f"Files: {', '.join(f.name for f in file_paths)}")
    console.print(f"Needle locations: {locations}")

    results = run_needle_suite(port, file_paths, locations, seed=seed)
    console.print("\n" + format_context_results(results))


def _run_enhanced_nihs(
    cfg: dict,
    model_name: str | None,
    ctx_size: int | None,
    num_needles: int,
    distractor_types: str,
    seed: int | None,
) -> None:
    """Run enhanced NIHS with multi-needle and distractors."""
    from rama.testing_modules.enhanced_nihs import (
        Distractor,
        EnhancedNIHSConfig,
        EnhancedNIHSModule,
        Needle,
    )

    if not model_name:
        console.print("[red]Enhanced mode requires -m <model_name>[/red]")
        raise typer.Exit(1)

    cfg_obj = Config.load(cfg)
    models = discover_models(cfg)
    model = next((m for m in models if m.name == model_name or model_name in m.aliases), None)
    if model is None:
        console.print(f"[red]Model not found: {model_name}[/red]")
        console.print("[dim]Available models:[/dim]")
        for m in models:
            console.print(f"  - {m.name} ({', '.join(m.aliases)})")
        raise typer.Exit(1)

    context_size = ctx_size or cfg_obj.context_size or 65536
    dist_types = [t.strip() for t in distractor_types.split(",")]

    nihs_config = EnhancedNIHSConfig(
        num_needles=num_needles,
        distractor_types=dist_types,
        context_size=context_size,
        seed=seed,
    )

    module = EnhancedNIHSModule(cfg, model, nihs_config)
    console.print(f"[cyan]Running enhanced NIHS on {model.name} (ctx={context_size})...[/cyan]")
    result = module.run()

    if result.metadata:
        acc = result.metadata.get("accuracy", 0)
        found = result.metadata.get("found", 0)
        total = result.metadata.get("total", 0)
        console.print(f"\n[green]Enhanced NIHS complete: {found}/{total} found ({acc:.0%})[/green]")
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")


def _run_difficulty_nihs(
    cfg: dict,
    model_name: str | None,
    ctx_size: int | None,
    difficulty: str,
    seed: int | None,
) -> None:
    """Run context difficulty injection tests."""
    from rama.testing_modules.context_difficulty import (
        ContextDifficultyConfig,
        ContextDifficultyModule,
    )

    if not model_name:
        console.print("[red]Difficulty mode requires -m <model_name>[/red]")
        raise typer.Exit(1)

    cfg_obj = Config.load(cfg)
    models = discover_models(cfg)
    model = next((m for m in models if m.name == model_name or model_name in m.aliases), None)
    if model is None:
        console.print(f"[red]Model not found: {model_name}[/red]")
        console.print("[dim]Available models:[/dim]")
        for m in models:
            console.print(f"  - {m.name} ({', '.join(m.aliases)})")
        raise typer.Exit(1)

    difficulty_levels = ["easy"] if difficulty == "easy" else difficulty.split(",")
    context_size = ctx_size or cfg_obj.context_size or 65536

    difficulty_config = ContextDifficultyConfig(
        context_size=context_size,
        difficulty_levels={
            k: v
            for k, v in ContextDifficultyConfig().difficulty_levels.items()
            if k in difficulty_levels or difficulty == "all"
        },
    )

    module = ContextDifficultyModule(cfg, model, difficulty_config)
    console.print(f"[cyan]Running context difficulty tests on {model.name}...[/cyan]")
    result = module.run()

    if result.metadata:
        results = result.metadata.get("results", {})
        for level, data in results.items():
            if "error" in data:
                console.print(f"[red]{level}: {data['error']}[/red]")
            else:
                scores = [r.get("score", 0) for r in data.get("results", [])]
                avg = sum(scores) / len(scores) if scores else 0
                wiki_tokens = data.get("wiki_tokens_injected", 0)
                console.print(f"[green]{level}: avg score={avg:.1f}/10 (wiki tokens injected: {wiki_tokens})[/green]")
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")


def _run_batch_nihs(
    cfg: dict,
    config_path: str,
    needle_loc: str,
    seed: int | None,
) -> None:
    """Run NIHS tests across multiple models from batch config."""
    from rama.testing_modules.batch_runner import BatchRunner

    if not Path(config_path).exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        raise typer.Exit(1)

    locations = [loc.strip() for loc in needle_loc.split(",")]
    cfg_obj = Config.load(cfg)

    with open(config_path) as f:
        batch_config = yaml.safe_load(f) or {}

    console.print(f"[cyan]Running batch NIHS from {config_path}...[/cyan]")
    console.print(f"Needle locations: {locations}")

    runner = BatchRunner(cfg_obj, batch_config, test_type="nihs")
    results = runner.run_nihs_batch(Path(config_path).parent, locations, seed)

    if results:
        console.print("\n[bold]=== Batch NIHS Results ===[/bold]")
        for model_name, model_results in results.items():
            for loc, result in model_results.items():
                score = result.get("score", "N/A")
                found = result.get("found", "N/A")
                total = result.get("total", "N/A")
                console.print(
                    f"  [cyan]{model_name}[/cyan] @ {loc}: {found}/{total} found (score: {score})"
                )


@app.command()
def run(
    model_name: str = typer.Argument(..., help="Model name or alias"),
    port: int | None = typer.Option(None, "--port", "-p", help="Port to use"),
    turbo3: bool = typer.Option(
        False,
        "--turbo3",
        help="Use TurboQuant with turbo3 KV cache",
    ),
    turbo4: bool = typer.Option(
        False,
        "--turbo4",
        help="Use TurboQuant with turbo4 KV cache",
    ),
    ctx_size: int | None = typer.Option(None, "--ctx", help="Context size"),
    ctk: str | None = typer.Option(None, "--ctk", help="KV cache type"),
    ctv: str | None = typer.Option(None, "--ctv", help="KV cache v type"),
    fit: bool = typer.Option(
        True, "--fit/--no-fit", help="Auto-fit model to VRAM (default: on)"
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for generation (default: random)"
    ),
    mtp: bool = typer.Option(
        False,
        "--mtp",
        help="Enable Multi-Token Prediction (requires model with MTP heads, e.g. Qwen3.5)",
    ),
    mtp_n_max: int = typer.Option(
        3,
        "--mtp-n-max",
        help="Max draft tokens for MTP (default: 3, recommended 2-3 for Qwen)",
    ),
    mtp_n_min: int = typer.Option(
        0,
        "--mtp-n-min",
        help="Min draft tokens for MTP (default: 0)",
    ),
    categories: str = typer.Option("all", "--categories", "-c", help="Test categories"),
    judge_port: int | None = typer.Option(
        None,
        "--judge",
        "-j",
        help="Judge model port",
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Results file"),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
) -> None:
    """Start model, run tests, then stop model."""
    cfg = get_config(config_path)
    models = discover_models(cfg)

    # Find model
    model = None
    for m in models:
        if model_name in m.name or model_name in m.path.name:
            model = m
            break

    if model is None:
        console.print(f"[red]Model not found: {model_name}[/red]")
        raise typer.Exit(1)

    # Determine backend
    backend, ctk, ctv = detect_backend(turbo3, turbo4, ctk, ctv)

    config_item = {}
    config_item["fit"] = fit
    if seed is not None:
        config_item["seed"] = seed

    result = None
    try:
        # Start model
        console.print(f"[cyan]Starting {model.name}...[/cyan]")
        result = start_model(
            model,
            backend=backend,
            port=port,
            ctx_size=ctx_size,
            ctk=ctk,
            ctv=ctv,
            seed=seed,
            config=cfg,
            config_item=config_item,
            mtp=mtp,
            mtp_n_max=mtp_n_max,
            mtp_n_min=mtp_n_min,
        )
        console.print(f"[green]Model started on port {result.port}[/green]")

        try:
            # Run tests
            console.print("[cyan]Running tests...[/cyan]")
            cats = (
                categories.split(",")
                if categories != "all"
                else DEFAULT_TEST_CATEGORIES
            )
            result_test = run_tests(
                result.port,
                judge_port=judge_port,
                categories=cats,
            )

            if output:
                Path(output).write_text(format_markdown(result_test))
                console.print(f"[green]Results saved to {output}[/green]")
            else:
                table = Table(title="Test Results")
                table.add_column("Test")
                table.add_column("Category")
                table.add_column("Score", justify="right")
                for r in result_test.results:
                    score_str = f"{r.score}/10" if r.score is not None else "N/A"
                    table.add_row(r.name, r.category, score_str)
                console.print(table)
                console.print(
                    f"\n[green]Overall: {result_test.overall_score:.1f}/10[/green]",
                )
        finally:
            # Stop model even if tests fail.
            console.print("[cyan]Stopping model...[/cyan]")
            stop_model(result.port, cfg)
            console.print("[green]Done![/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def bench(
    port: int = typer.Argument(..., help="Port of running model"),
    n_prompt: int = typer.Option(512, "-p", help="Prompt tokens"),
    n_gen: int = typer.Option(128, "-n", help="Tokens to generate"),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file",
    ),
) -> None:
    """Run benchmark on running model via API."""
    cfg = Config(config_path)
    url = f"http://{cfg['host']}:{port}/v1/completions"
    data = {
        "prompt": "A" * n_prompt,
        "max_tokens": n_gen,
        "stream": False,
    }

    console.print("[cyan]Benchmarking...[/cyan]")

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
        elapsed = time.time() - start

        tokens = result.get("usage", {}).get("completion_tokens", n_gen)
        prompt_tokens = result.get("usage", {}).get("prompt_tokens", n_prompt)

        console.print("[green]Done![/green]")
        console.print(f"  Prompt: {prompt_tokens} tokens")
        console.print(f"  Generated: {tokens} tokens")
        console.print(f"  Time: {elapsed:.2f}s")
        console.print(f"  Speed: {tokens / elapsed:.1f} tokens/s")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def perplexity(
    model_name: str = typer.Argument(..., help="Model name"),
    text_file: str = typer.Argument(..., help="Text file to evaluate"),
    ctx_size: int = typer.Option(4096, "--ctx", help="Context size"),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file",
    ),
) -> None:
    """Run perplexity benchmark on text file."""
    cfg = get_config(config_path)
    models = discover_models(cfg)

    model = None
    for m in models:
        if model_name in m.name:
            model = m
            break

    if model is None:
        console.print(f"[red]Model not found: {model_name}[/red]")
        raise typer.Exit(1)

    text_path = Path(text_file)
    if not text_path.exists():
        console.print(f"[red]File not found: {text_file}[/red]")
        raise typer.Exit(1)

    backend = cfg.llama_cpp_dir / "build" / "bin" / "llama-perplexity"

    if not backend.exists():
        console.print(
            "[yellow]llama-perplexity not found. Build llama.cpp first.[/yellow]",
        )
        raise typer.Exit(1)

    args = [
        str(backend),
        "-m",
        str(model.path),
        "-f",
        str(text_path),
        "--ctx-size",
        str(ctx_size),
    ]

    console.print("[cyan]Running perplexity...[/cyan]")
    result = subprocess.run(args, capture_output=True, text=True)
    console.print(result.stdout)
    if result.stderr:
        console.print(result.stderr, style="red")


@app.command()
def batch(
    config_file: str = typer.Argument(..., help="Batch config YAML file"),
    batch_type: str = typer.Option(
        "code",
        "--type",
        "-t",
        help="Batch type: code, perplexity, nihs, kv",
    ),
    start_port: int = typer.Option(11435, "--start-port", help="Starting port"),
    categories: str = typer.Option("all", "--categories", "-c", help="Test categories"),
    output_dir: str = typer.Option(
        "results",
        "--output-dir",
        "-o",
        help="Results directory",
    ),
    seed: int = typer.Option(
        42, "--seed", help="Default random seed for all batch items"
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        help="Config file path",
    ),
    text_file: str | None = typer.Option(
        None, "--text", help="Text file for NIHS/perplexity batch"
    ),
    ctx_size: int = typer.Option(0, "--ctx", help="Context size (for perplexity/nihs)"),
    kv_quants: str = typer.Option(
        "f16,q8_0,q4_0", "--kvs", "-k", help="KV quant types for kv batch"
    ),
    ctx_sizes: str = typer.Option(
        "8192,32768,65536,131072,262144",
        "--ctx-sizes",
        "-s",
        help="Context sizes for NIHS",
    ),
    needle_loc: str = typer.Option(
        "beginning,middle,end", "--needle-loc", "-n", help="Needle locations"
    ),
) -> None:
    """Run batch of models from config file.

    Batch types:
      code       - code/debugging tests (default)
      perplexity - perplexity evaluation
      nihs       - needle-in-haystack batch
      kv         - KV quant comparison batch

    Examples:
       rama batch config/batch_large.yaml --type code
       rama batch config/batch_ppl.yaml --type perplexity --text wikitext-2/wiki.valid.tokens
       rama batch config/batch_large.yaml --type nihs --text book.txt
       rama batch config/batch_large.yaml --type kv --kvs f16,q8_0
    """
    batch_config = BatchRunnerConfig(
        config_file=config_file,
        batch_type=batch_type,
        start_port=start_port,
        categories=categories,
        output_dir=output_dir,
        seed=seed,
        config_path=config_path,
        text_file=text_file,
        ctx_size=ctx_size,
        kv_quants=kv_quants,
        ctx_sizes=ctx_sizes,
        needle_loc=needle_loc,
    )

    runner = BatchRunner(batch_config)
    runner.run()


if __name__ == "__main__":
    app()
