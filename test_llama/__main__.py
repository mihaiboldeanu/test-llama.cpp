from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import Config
from .context_test import format_context_results, run_context_suite
from .core import discover_models, get_running_models, is_port_used
from .launch import build_backend, start_model, stop_model
from .testing import (
    discover_tests,
    format_markdown,
    generate_online_eval_prompt,
    run_tests,
)

console = Console()


def get_config(config_path: Optional[str] = None) -> Config:
    return Config(config_path)


app = typer.Typer(name="test-llama", help="CLI to test and run llama.cpp models")


@app.command()
def list(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
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
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
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
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to use"),
    turbo3: bool = typer.Option(
        False, "--turbo3", help="Use TurboQuant with turbo3 KV cache"
    ),
    turbo4: bool = typer.Option(
        False, "--turbo4", help="Use TurboQuant with turbo4 KV cache"
    ),
    ctx_size: Optional[int] = typer.Option(
        None, "--ctx", help="Context size (auto calculated if not set)"
    ),
    fit: bool = typer.Option(
        False, "--fit", help="Fit model to VRAM (splits MoE layers between RAM/VRAM)"
    ),
    no_offload_kv: bool = typer.Option(
        False, "--no-offload-kv", help="Don't offload KV to VRAM"
    ),
    ctk: Optional[str] = typer.Option(
        None, "--ctk", help="KV cache type: turbo3, turbo4, q8_0, q4_k_m, etc."
    ),
    ctv: Optional[str] = typer.Option(
        None, "--ctv", help="KV cache v type: turbo3, turbo4, q8_0, q4_k_m, etc."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for generation"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
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
    if turbo3:
        backend = "turboquant"
        ctk = ctk or "turbo3"
        ctv = ctv or "turbo3"
    elif turbo4:
        backend = "turboquant"
        ctk = ctk or "turbo4"
        ctv = ctv or "turbo4"
    elif ctk and "turbo" in (ctk + (ctv or "")):
        backend = "turboquant"
    else:
        backend = "llama.cpp"

    # Fit mode options
    config_item = {}
    if fit:
        config_item["fit"] = True
    if no_offload_kv:
        config_item["no_offload_kv"] = True
    if seed is not None:
        config_item["seed"] = seed
    config_item["no_warmup"] = True
    config_item["cache_ram"] = 0
    config_item["log_verbosity"] = 1
    config_item["no_context_shift"] = False

    # Show what context we'll get
    if ctx_size is None:
        from .core import calc_kv_cache

        ctx, qk, qv = calc_kv_cache(model.size_gb, model.family, cfg)
        console.print(
            f"[dim]Model: {model.name} ({model.size_gb}GB, {model.family})[/dim]"
        )
        console.print(f"[dim]Auto ctx={ctx}, ctk={qk}, ctv={qv}[/dim]")
        if fit:
            console.print("[dim]--fit: split MoE layers between RAM/VRAM[/dim]")

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
        )
        console.print("\n[green]Model started![/green]")
        console.print(f"  Name: {result.name}")
        console.print(f"  Port: {result.port}")
        console.print(f"  PID: {result.pid}")
        console.print(f"  Backend: {result.backend}")
        console.print(f"  Context: {result.ctx_size}")
        print(f"\n  Chat: http://127.0.0.1:{result.port}/v1/chat/completions")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stop(
    port: int = typer.Argument(..., help="Port to stop"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
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
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild even if up to date"
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
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
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
        "test_llama.yaml", "--config", "-c", help="Config file path"
    ),
) -> None:
    """Create a config file."""
    import shutil

    src = Path(__file__).parent.parent / "test_llama.yaml"
    dst = Path(config_path).resolve()

    if dst.exists():
        console.print(f"[yellow]Config already exists: {dst}[/yellow]")
        return

    shutil.copy(src, dst)
    console.print(f"[green]Created config: {dst}[/green]")
    console.print("[dim]Edit it to customize paths![/dim]")


@app.command()
def test(
    port: int = typer.Argument(..., help="Port of running model"),
    categories: str = typer.Option(
        "all", "--categories", "-c", help="Comma-separated: code,debugging,creative"
    ),
    judge_port: Optional[int] = typer.Option(
        None, "--judge", "-j", help="Port of judge model for scoring"
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="Skip LLM-as-judge scoring"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save results to file"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, markdown, json, eval"
    ),
) -> None:
    """Run tests on a running model."""
    # Parse categories
    if categories == "all":
        cats = ["code", "debugging", "creative"]
    else:
        cats = categories.split(",")

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
        import json

        console.print(
            json.dumps(
                {
                    "overall_score": result.overall_score,
                    "tests": [(r.name, r.category, r.score) for r in result.results],
                },
                indent=2,
            )
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
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Show context info for all available models."""
    from .core import calc_kv_cache

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
        f"\n[dim]Based on {cfg.vram_available_gb}GB available VRAM ({cfg['vram_total_gb']}GB - {cfg['vram_headroom_gb']}GB headroom)[/dim]"
    )


@app.command()
def context(
    port: int = typer.Argument(..., help="Port of running model"),
    files: str = typer.Argument(..., help="Comma-separated paths to text files"),
    ctx_sizes: str = typer.Option(
        "10240,32768,65536,131072,262144",
        "--ctx-sizes",
        help="Comma-separated context sizes",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for the context prompt generator"),
) -> None:
    """Test long context handling - stress test at various context sizes.

    Tests if models can handle large context without:
    - Looping/repetition
    - Losing track of information
    - Degrading output quality
    """

    # Parse files
    file_paths = [Path(f.strip()) for f in files.split(",")]
    for f in file_paths:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)

    # Parse ctx sizes
    sizes = [int(s.strip()) for s in ctx_sizes.split(",")]

    console.print("[cyan]Running context stress tests...[/cyan]")
    console.print(f"Files: {', '.join(f.name for f in file_paths)}")
    console.print(f"Context sizes: {sizes}")

    results = run_context_suite(port, file_paths, sizes, seed=seed)

    console.print("\n" + format_context_results(results))


@app.command()
def run(
    model_name: str = typer.Argument(..., help="Model name or alias"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to use"),
    turbo3: bool = typer.Option(
        False, "--turbo3", help="Use TurboQuant with turbo3 KV cache"
    ),
    turbo4: bool = typer.Option(
        False, "--turbo4", help="Use TurboQuant with turbo4 KV cache"
    ),
    ctx_size: Optional[int] = typer.Option(None, "--ctx", help="Context size"),
    ctk: Optional[str] = typer.Option(None, "--ctk", help="KV cache type"),
    ctv: Optional[str] = typer.Option(None, "--ctv", help="KV cache v type"),
    fit: bool = typer.Option(False, "--fit", help="Use launcher fit policy"),
    no_offload_kv: bool = typer.Option(
        False, "--no-offload-kv", help="Don't offload KV to VRAM"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for generation"),
    categories: str = typer.Option("all", "--categories", "-c", help="Test categories"),
    judge_port: Optional[int] = typer.Option(
        None, "--judge", "-j", help="Judge model port"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Results file"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
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
    if turbo3:
        backend = "turboquant"
        ctk = ctk or "turbo3"
        ctv = ctv or "turbo3"
    elif turbo4:
        backend = "turboquant"
        ctk = ctk or "turbo4"
        ctv = ctv or "turbo4"
    elif ctk and "turbo" in (ctk + (ctv or "")):
        backend = "turboquant"
    else:
        backend = "llama.cpp"

    config_item = {}
    if fit:
        config_item["fit"] = True
    if no_offload_kv:
        config_item["no_offload_kv"] = True
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
        )
        console.print(f"[green]Model started on port {result.port}[/green]")

        try:
            # Run tests
            console.print("[cyan]Running tests...[/cyan]")
            result_test = run_tests(
                result.port,
                judge_port=judge_port,
                categories=categories.split(",") if categories != "all" else None,
            )

            if output:
                from .testing import format_markdown

                Path(output).write_text(format_markdown(result_test))
                console.print(f"[green]Results saved to {output}[/green]")
            else:
                from rich.table import Table

                table = Table(title="Test Results")
                table.add_column("Test")
                table.add_column("Category")
                table.add_column("Score", justify="right")
                for r in result_test.results:
                    score_str = f"{r.score}/10" if r.score is not None else "N/A"
                    table.add_row(r.name, r.category, score_str)
                console.print(table)
                console.print(
                    f"\n[green]Overall: {result_test.overall_score:.1f}/10[/green]"
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
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file"
    ),
) -> None:
    """Run benchmark on running model via API."""
    import json
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/completions"
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

    import time

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
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file"
    ),
) -> None:
    """Run perplexity benchmark on text file."""
    import subprocess

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
            "[yellow]llama-perplexity not found. Build llama.cpp first.[/yellow]"
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
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


@app.command()
def batch(
    config_file: str = typer.Argument(..., help="Batch config YAML file"),
    start_port: int = typer.Option(11435, "--start-port", help="Starting port"),
    categories: str = typer.Option("all", "--categories", "-c", help="Test categories"),
    output_dir: str = typer.Option(
        "results", "--output-dir", "-o", help="Results directory"
    ),
    seed: int = typer.Option(42, "--seed", help="Default random seed for all batch items"),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
) -> None:
    """Run batch of models from config file.

    Config file format (YAML):
    - model: qwen27b              # just model name (uses defaults)
    - model: qwen27b, ctk: turbo3, ctv: turbo3  # with options
    - model: qwen27b, ctx: 65536  # custom context
    """
    import time
    from datetime import datetime

    import yaml

    from .testing import run_tests, save_results_csv

    cfg = get_config(config_path)
    models = discover_models(cfg)

    # Load batch config
    with open(config_file) as f:
        batch_config = yaml.safe_load(f)

    # Create output dir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"batch_{timestamp}.json"

    console.print(f"[cyan]Batch running {len(batch_config)} models...[/cyan]")
    console.print(f"Results will be saved to {output_path}/")
    console.print("")

    all_results = []
    port = start_port

    for item in batch_config:
        if isinstance(item, str):
            model_name = item
            config_item = {}
        else:
            model_name = item.get("model", "")
            config_item = {k: v for k, v in item.items() if k != "model"}

        if seed is not None and "seed" not in config_item:
            config_item["seed"] = seed
        config_item.setdefault("no_warmup", True)
        config_item.setdefault("cache_ram", 0)
        config_item.setdefault("log_verbosity", 1)
        config_item.setdefault("no_context_shift", False)

        if not model_name:
            continue

        # Find model
        model = None
        for m in models:
            if model_name in m.name or model_name in m.path.name:
                model = m
                break

        if model is None:
            console.print(f"[red]Model not found: {model_name}[/red]")
            continue

        # Determine options
        ctk = config_item.get("ctk")
        ctv = config_item.get("ctv")
        ctx_size = config_item.get("ctx")
        turbo3 = config_item.get("turbo3", False)
        turbo4 = config_item.get("turbo4", False)

        if turbo3:
            backend = "turboquant"
            ctk = ctk or "turbo3"
            ctv = ctv or "turbo3"
        elif turbo4:
            backend = "turboquant"
            ctk = ctk or "turbo4"
            ctv = ctv or "turbo4"
        elif ctk and "turbo" in (ctk + (ctv or "")):
            backend = "turboquant"
        else:
            backend = "llama.cpp"

        use_port = port
        port += 1

        console.print(f"[cyan]=== {model.name} ===[/cyan]")
        console.print(
            f"  Backend: {backend}, ctk={ctk or 'q8_0'}, ctv={ctv or 'q8_0'}, ctx={ctx_size}"
        )

        result = None
        try:
            # Start model
            result = start_model(
                model,
                backend=backend,
                port=use_port,
                ctx_size=ctx_size,
                ctk=ctk,
                ctv=ctv,
                seed=config_item.get("seed"),
                config=cfg,
                config_item=config_item,
            )
            console.print(f"  Started on port {result.port}")

            try:
                # Run tests
                cats = categories.split(",") if categories != "all" else None
                test_start = time.perf_counter()
                test_result = run_tests(result.port, categories=cats)
                test_duration = time.perf_counter() - test_start

                # Save result
                import json

                result_data = {
                    "model": model.name,
                    "backend": backend,
                    "ctk": ctk or "q8_0",
                    "ctv": ctv or "q8_0",
                    "ctx": ctx_size or "auto",
                    "seed": result.seed,
                    "test_time_seconds": round(test_duration, 3),
                    "score": test_result.overall_score,
                    "tests": [
                        {
                            "name": r.name,
                            "category": r.category,
                            "prompt": r.prompt,
                            "response": r.response,
                            "ran": r.ran,
                            "correct": r.correct,
                            "score": r.score,
                        }
                        for r in test_result.results
                    ],
                }
                all_results.append(result_data)

                # Save after each model (in case of crash)
                results_file.write_text(json.dumps(all_results, indent=2))
                csv_file = output_path / f"batch_{timestamp}.csv"
                save_results_csv(all_results, csv_file)

                console.print(f"  Score: {test_result.overall_score}/10")
                console.print(f"  Test time: {test_duration:.2f}s")
            finally:
                if result is not None:
                    stop_model(result.port, cfg)
                    console.print("  Stopped")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        console.print("")

    # Save all results
    import json

    results_file.write_text(json.dumps(all_results, indent=2))
    csv_file = output_path / f"batch_{timestamp}.csv"
    save_results_csv(all_results, csv_file)

    console.print("[green]Batch complete![/green]")
    console.print(f"  JSON results: {results_file}")
    console.print(f"  CSV results: {csv_file}")


if __name__ == "__main__":
    app()
