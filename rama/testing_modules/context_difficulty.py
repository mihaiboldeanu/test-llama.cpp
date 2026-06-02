"""Context difficulty injection module.

Injects wiki text and code files before/after/interleaved with prompts to test model robustness.
Supports 4 difficulty levels: easy, medium, hard, expert.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from rama.core import find_free_port
from pathlib import Path
from typing import Any

from rich.console import Console

from .base import TestModule, TestResult

console = Console()


@dataclass
class ContextDifficultyConfig:
    """Configuration for context difficulty testing."""

    # Wiki tokens to inject at each difficulty level
    difficulty_levels: dict[str, dict] = field(
        default_factory=lambda: {
            "easy": {"wiki_tokens": 0, "code_files": [], "placement": "none"},
            "medium": {"wiki_tokens": 4000, "code_files": [], "placement": "before"},
            "hard": {
                "wiki_tokens": 8000,
                "code_files": ["tests/code/*.txt", "count: 3"],
                "placement": "before_after",
            },
            "expert": {
                "wiki_tokens": 16000,
                "code_files": ["tests/code/*.txt", "tests/debugging/*.txt"],
                "placement": "interleaved",
            },
        }
    )

    placement: str = "before"  # "before", "after", "interleaved", "before_after"
    wiki_file: str = "wikitext-2/wiki.valid.tokens"
    context_size: int = 65536


class ContextDifficultyModule(TestModule):
    """Test model performance with injected context noise.

    Difficulty levels inject varying amounts of wiki text and code files:
    - easy: No injection (baseline)
    - medium: 4000 wiki tokens
    - hard: 8000 wiki tokens + 3 code files
    - expert: 16000 wiki tokens + all code + debugging files
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any,
        difficulty_config: ContextDifficultyConfig | None = None,
        host: str = "127.0.0.1",
    ):
        super().__init__(config, model, host)
        self.difficulty_config = difficulty_config or ContextDifficultyConfig()
        self._wiki_cache: dict[str, str] = {}

    def run(self) -> TestResult:
        """Run context difficulty tests across all levels."""
        start_time = time.perf_counter()
        self._log("Starting context difficulty tests")

        all_results = {}

        for difficulty, config in self.difficulty_config.difficulty_levels.items():
            self._log(f"Testing difficulty: {difficulty}")

            # Load wiki text if needed
            wiki_text = self._load_wiki_text(
                config["wiki_tokens"] if config["wiki_tokens"] > 0 else None
            )

            # Load code files if needed
            code_contents = self._load_code_files(config.get("code_files", []))

            # Get a port for this test
            port = find_free_port(self.config)

            try:
                # Start model with large enough context
                actual_port = self._start_model(
                    port=port,
                    ctx_size=self.difficulty_config.context_size,
                    ctk="f16",
                    ctv="f16",
                )
                self._log(f"Model started on port {actual_port}")

                # Run tests with injected context
                test_results = self._run_tests_with_context(
                    actual_port, wiki_text, code_contents, difficulty
                )

                all_results[difficulty] = {
                    "results": test_results,
                    "wiki_tokens_injected": len(wiki_text) // 4 if wiki_text else 0,
                    "code_files_injected": len(code_contents),
                    "placement": self.difficulty_config.placement,
                }

                avg_score = sum(r.get("score", 0) for r in test_results) / len(
                    test_results
                ) if test_results else 0
                self._log(f"  Average score: {avg_score:.1f}/10")

            except Exception as e:
                self._log(f"  Error with {difficulty}: {e}", "red")
                all_results[difficulty] = {"error": str(e), "results": []}
            finally:
                try:
                    self._stop_model(actual_port)
                    self._log(f"Stopped model on port {actual_port}")
                except Exception as e:
                    self._log(f"Failed to stop model on port {actual_port}: {e}", style="red")

        duration = time.perf_counter() - start_time

        result = TestResult(
            name="context_difficulty",
            score=None,
            metadata={
                "results": all_results,
                "config": {
                    "placement": self.difficulty_config.placement,
                    "difficulty_levels": {
                        k: {"wiki_tokens": v["wiki_tokens"], "code_files": v["code_files"]}
                        for k, v in self.difficulty_config.difficulty_levels.items()
                    },
                },
                "duration_seconds": duration,
            },
        )

        self._log(f"Context difficulty tests complete in {duration:.1f}s")
        return result

    def _load_wiki_text(self, max_tokens: int | None = None) -> str | None:
        """Load wiki text file, optionally truncated to max_tokens."""
        wiki_path = Path(self.difficulty_config.wiki_file)
        if not wiki_path.exists():
            self._log(f"  Wiki file not found: {wiki_path}", "yellow")
            return None

        if max_tokens is None or max_tokens == 0:
            return None

        # Cache the full file
        if str(wiki_path) not in self._wiki_cache:
            self._wiki_cache[str(wiki_path)] = wiki_path.read_text()

        full_text = self._wiki_cache[str(wiki_path)]

        # Truncate to max_tokens (roughly 4 chars per token)
        max_chars = max_tokens * 4
        if len(full_text) > max_chars:
            # Truncate at word boundary
            truncated = full_text[:max_chars]
            last_space = truncated.rfind(" ")
            if last_space > max_chars // 2:
                truncated = truncated[:last_space]
            return truncated

        return full_text

    def _load_code_files(
        self, file_patterns: list[str] | None = None
    ) -> list[str]:
        """Load code files matching the given patterns."""
        if not file_patterns:
            return []

        contents = []
        for pattern_spec in file_patterns:
            if pattern_spec.startswith("tests/"):
                # Parse pattern like "tests/code/*.txt, count: 3"
                parts = [p.strip() for p in pattern_spec.split(",")]
                glob_pattern = parts[0]
                count = None
                for part in parts[1:]:
                    if part.startswith("count:"):
                        count = int(part.split(":")[1].strip())
                        break

                files = sorted(Path(glob_pattern).parent.glob(glob_pattern.split("/")[-1]))
                files = [f for f in files if f.suffix == ".txt"]

                if count is not None:
                    # Randomly select N files
                    if len(files) > count:
                        random.shuffle(files)
                        files = files[:count]

                for f in files:
                    contents.append(f.read_text())
            else:
                self._log(f"  Unknown pattern: {pattern_spec}", "yellow")

        return contents

    def _run_tests_with_context(
        self,
        port: int,
        wiki_text: str | None,
        code_contents: list[str],
        difficulty: str = "easy",
    ) -> list[dict[str, Any]]:
        """Run code/debug tests with injected context based on difficulty level."""
        from rama.testing import run_tests

        difficulty_config = self.difficulty_config.difficulty_levels.get(difficulty, {})
        placement = difficulty_config.get("placement", self.difficulty_config.placement)

        try:
            test_result = run_tests(
                port,
                categories=["code", "debugging"],
                config=self.config,
                context_text=wiki_text,
                context_files=code_contents,
                context_placement=placement,
            )

            results = []
            for r in test_result.results:
                results.append({
                    "name": r.name,
                    "category": r.category,
                    "score": r.score,
                    "correct": r.correct,
                })

            return results
        except Exception as e:
            self._log(f"  Test execution error: {e}", "yellow")
            return []

