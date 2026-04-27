"""Needle-in-a-haystack context testing."""

import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class ContextTestResult:
    """Result of a needle-in-haystack test."""

    needle_location: str
    needle_token: str
    prompt_tokens: int
    response: str
    found: bool
    has_repetition: bool
    is_looping: bool
    error: str = ""


def generate_needle(seed: int | None = None) -> str:
    """Generate a unique needle token."""
    if seed is not None:
        rng = random.Random(seed)
        return f"NEEDLE-{rng.getrandbits(32):08x}"
    return f"NEEDLE-{uuid.uuid4().hex[:8]}"


def insert_needle(content: str, needle: str, location: str) -> str:
    """Insert needle at the specified location.

    Location can be:
    - 'beginning' - insert at start
    - 'middle' - insert at 50%
    - 'end' - insert at end
    - '25%' - insert at 25% of the way through
    """
    needle_line = f"\n\nSPECIAL_CODE: {needle}\n\n"

    if location == "beginning":
        return needle_line + content
    if location in {"middle", "50%"}:
        mid = len(content) // 2
        return content[:mid] + needle_line + content[mid:]
    if location == "end":
        return content + needle_line

    # Parse percentage
    if location.endswith("%"):
        pct = int(location[:-1]) / 100.0
        pos = int(len(content) * pct)
        return content[:pos] + needle_line + content[pos:]

    raise ValueError(f"Invalid needle location: {location}")


def build_prompt(content: str, needle: str) -> str:
    """Build the full test prompt."""
    return (
        f"Read the following text carefully. Then answer the question at the end.\n\n"
        f"TEXT:\n{content}\n\n"
        f"QUESTION: What is the SPECIAL_CODE mentioned in the text above? "
        f"Only respond with the code itself, nothing else."
    )


def check_for_repetition(text: str, min_repeats: int = 5) -> bool:
    """Check if text has excessive repetition."""
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        if text.count(line) >= min_repeats:
            return True

    seen = {}
    window = 20
    for i in range(min(len(text) - window, 1000)):
        chunk = text[i : i + window]
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= 3:
            return True

    return False


def check_for_looping(text: str, window: int = 100) -> bool:
    """Check if response is stuck in a loop."""
    text = text.strip()
    if len(text) < window * 2:
        return False

    last_chunk = text[-window:]
    if text[:-window].rfind(last_chunk) != -1:
        return True

    sentences = re.split(r"[.!?]+", text)
    if len(sentences) > 3:
        unique = set(s.strip() for s in sentences if s.strip())
        if len(unique) < len(sentences) / 2:
            return True

    return False


def run_needle_test(
    port: int,
    text_file: Path,
    needle_loc: str,
    seed: int | None = None,
) -> ContextTestResult:
    """Run a single needle-in-haystack test."""
    from .testing import run_chat_completion

    content = text_file.read_text()
    needle = generate_needle(seed)
    haystack = insert_needle(content, needle, needle_loc)
    prompt = build_prompt(haystack, needle)

    response = run_chat_completion(
        port,
        prompt,
        temperature=0.0,
        max_tokens=256,
        seed=seed if seed is not None else 42,
    )

    has_repetition = check_for_repetition(response)
    is_looping = check_for_looping(response)

    response_clean = response.strip().lower()
    needle_clean = needle.lower()
    found = needle_clean in response_clean

    prompt_tokens = len(prompt) // 4

    return ContextTestResult(
        needle_location=needle_loc,
        needle_token=needle,
        prompt_tokens=prompt_tokens,
        response=response[:500],
        found=found,
        has_repetition=has_repetition,
        is_looping=is_looping,
    )


def run_needle_suite(
    port: int,
    text_files: list[Path],
    needle_locs: list[str],
    seed: int | None = None,
) -> list[ContextTestResult]:
    """Run needle tests at various locations."""
    results = []

    for text_file in text_files:
        console.print(
            f"\n[cyan]Testing with {text_file.name} ({len(text_file.read_text()) // 4:,} est. tokens)...[/cyan]"
        )

        for loc in needle_locs:
            console.print(f"  Needle at {loc}...", end=" ")

            result = run_needle_test(port, text_file, loc, seed=seed)
            results.append(result)

            if result.is_looping:
                console.print("[red]LOOPING[/red]")
            elif result.has_repetition:
                console.print("[yellow]REPETITION[/yellow]")
            elif result.found:
                console.print("[green]FOUND[/green]")
            else:
                console.print(f"[red]MISSED (got: {result.response[:60]})[/red]")

    return results


def format_context_results(results: list[ContextTestResult]) -> str:
    """Format needle test results."""
    lines = ["# Needle-in-Haystack Results", ""]
    lines.append("| Location | Tokens | Found | Repetition | Looping |")
    lines.append("|----------|--------|-------|------------|---------|")

    for r in results:
        found = "[green]YES[/green]" if r.found else "[red]NO[/red]"
        rep = "Y" if r.has_repetition else "-"
        loop = "Y" if r.is_looping else "-"
        lines.append(
            f"| {r.needle_location:>12} | {r.prompt_tokens:>6,} | {found:>7} | {rep:>10} | {loop:>7} |",
        )

    return "\n".join(lines)
