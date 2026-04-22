"""Long context testing - stress test models at various context sizes."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

CONTEXT_SIZES = [10240, 32768, 65536, 131072, 262144]


@dataclass
class ContextTestResult:
    """Result of a context test."""

    ctx_size: int
    prompt_tokens: int
    response: str
    has_repetition: bool
    is_looping: bool
    completed: bool
    seed: int | None = None
    error: str = ""


def generate_test_prompt(
    text_file: Path, question_about: str, seed: int | None = 42
) -> tuple[str, str]:
    """Generate a prompt that tests if model read the text.

    Returns (full_prompt, answer_key)
    """
    content = text_file.read_text()
    rng = random.Random(seed) if seed is not None else random

    # Find a unique string to ask about
    # Look for a distinctive pattern like "TOKEN: XYZ"
    import re

    tokens = re.findall(r"TOKEN:\s*(\w+)", content)

    if tokens:
        answer = rng.choice(tokens)
    else:
        # Generate a random unique token and insert it
        import uuid

        answer = f"{rng.getrandbits(32):08x}" if seed is not None else str(uuid.uuid4())[:8]
        content += f"\n\nUNIQUE_TOKEN: {answer}\n"

    prompt = f"""Read the following text carefully. Then answer the question at the end.

TEXT:
{content}

QUESTION: What is the UNIQUE_TOKEN mentioned in the text above? Only respond with the token itself, nothing else."""

    return prompt, answer


def check_for_repetition(text: str, min_repeats: int = 5) -> bool:
    """Check if text has excessive repetition."""
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        # Check same line repeated
        count = text.count(line)
        if count >= min_repeats:
            return True

    # Check for character repetition patterns
    # Look for same 20-char sequence repeated 3+ times
    for i in range(len(text) - 20):
        chunk = text[i : i + 20]
        if text.count(chunk) >= 3:
            return True

    return False


def check_for_looping(text: str, window: int = 100) -> bool:
    """Check if response is stuck in a loop."""
    text = text.strip()
    if len(text) < window * 2:
        return False

    # Check if last N chars repeat
    last_chunk = text[-window:]
    if text[:-window].rfind(last_chunk) != -1:
        return True

    # Check for cycling patterns
    import re

    # Look for repeated sentence structures
    sentences = re.split(r"[.!?]+", text)
    if len(sentences) > 3:
        unique = set(s.strip() for s in sentences if s.strip())
        if len(unique) < len(sentences) / 2:
            return True

    return False


def run_context_test(
    port: int,
    text_file: Path,
    ctx_size: int,
    seed: int | None = 42,
) -> ContextTestResult:
    """Run a single context test."""
    from .testing import run_chat_completion

    prompt, answer_key = generate_test_prompt(text_file, "UNIQUE_TOKEN", seed=seed)

    response = run_chat_completion(
        port,
        prompt,
        temperature=0.0,
        max_tokens=256,
        seed=seed if seed is not None else 42,
    )

    # Analyze response
    has_repetition = check_for_repetition(response)
    is_looping = check_for_looping(response)

    # Check if model answered correctly
    response_clean = response.strip().lower()
    answer_clean = answer_key.strip().lower()
    answered_correctly = answer_clean in response_clean

    # Count tokens (rough estimate: 4 chars per token)
    prompt_tokens = len(prompt) // 4

    return ContextTestResult(
        ctx_size=ctx_size,
        prompt_tokens=prompt_tokens,
        response=response[:500],
        has_repetition=has_repetition,
        is_looping=is_looping,
        completed=answered_correctly,
        seed=seed,
    )


def run_context_suite(
    port: int,
    text_files: list[Path],
    ctx_sizes: Optional[list[int]] = None,
    seed: int | None = 42,
) -> list[ContextTestResult]:
    """Run context tests at various sizes."""
    if ctx_sizes is None:
        ctx_sizes = CONTEXT_SIZES

    results = []

    for text_file in text_files:
        console.print(f"\n[cyan]Testing with {text_file.name}...[/cyan]")

        for ctx_size in ctx_sizes:
            console.print(f"  Context: {ctx_size:,}...", end=" ")

            result = run_context_test(port, text_file, ctx_size, seed=seed)
            results.append(result)

            if result.is_looping:
                console.print("[red]LOOPING[/red]")
            elif result.has_repetition:
                console.print("[yellow]REPETITION[/yellow]")
            elif result.completed:
                console.print("[green]OK[/green]")
            else:
                console.print("[yellow]FAILED[/yellow]")

    return results


def format_context_results(results: list[ContextTestResult]) -> str:
    """Format context test results."""
    lines = ["# Context Stress Test Results", ""]
    lines.append("| Context Size | Tokens | Repetition | Looping | Completed |")
    lines.append("|--------------|--------|-------------|---------|-----------|")

    for r in results:
        rep = "✓" if r.has_repetition else "-"
        loop = "✓" if r.is_looping else "-"
        done = "✓" if r.completed else "-"
        lines.append(
            f"| {r.ctx_size:>12,} | {r.prompt_tokens:>6,} | {rep:>11} | {loop:>7} | {done:>9} |"
        )

    return "\n".join(lines)
