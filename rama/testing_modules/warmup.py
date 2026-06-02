"""Warm-up module.

Implements a 3-message warm-up protocol before testing:
1. System prompt (auto-generated based on model + category, or custom)
2. "Hello" message
3. Simple response to let the model settle

This follows llama.cpp bench best practices of discarding the first N iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rama.core import find_free_port
from typing import Any

from rich.console import Console

from .base import TestModule, TestResult
from .system_prompt import get_system_prompt

console = Console()


@dataclass
class WarmupConfig:
    """Configuration for warm-up protocol."""

    messages: int = 3  # Number of warm-up messages
    system_prompt: str | None = None  # Optional custom system prompt
    category: str = "all"  # Test category for auto-generated prompts
    hello_message: str = "Hello, how are you?"  # Default hello message
    final_message: str = "Thank you, have a great day!"  # Final warm-up message
    max_tokens: int = 64  # Max tokens per warm-up message


class WarmupModule(TestModule):
    """Run warm-up protocol before testing.

    Follows llama.cpp bench best practices:
    - Discard first N iterations
    - Use 3-message protocol
    - Let model settle before actual testing
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any,
        warmup_config: WarmupConfig | None = None,
        host: str = "127.0.0.1",
        category: str = "all",
    ):
        super().__init__(config, model, host)
        if warmup_config is None:
            warmup_config = WarmupConfig()
        warmup_config.category = category
        self.warmup_config = warmup_config

    def run(self) -> TestResult:
        """Run warm-up protocol and return timing results."""
        start_time = time.perf_counter()
        self._log("Starting warm-up protocol")

        port = find_free_port(self.config)

        try:
            # Start model
            actual_port = self._start_model(
                port=port,
                ctx_size=4096,  # Small context for warm-up
                ctk="f16",
                ctv="f16",
            )
            self._log(f"Model started on port {actual_port}")

            messages = self._build_messages()
            timings = []

            for i, msg in enumerate(messages, 1):
                msg_start = time.perf_counter()
                response = self._run_chat(
                    actual_port,
                    msg,
                    temperature=0.0,
                    max_tokens=self.warmup_config.max_tokens,
                )
                elapsed = time.perf_counter() - msg_start
                timings.append({
                    "message": i,
                    "prompt": msg[:50],
                    "response_length": len(response),
                    "tokens_per_second": (
                        len(response) / 4 / elapsed if elapsed > 0 else 0
                    ),
                    "duration_seconds": elapsed,
                })

                self._log(f"  Warm-up message {i}/{len(messages)}: {elapsed:.2f}s")

            # Stop model
            self._stop_model(actual_port)
            self._log(f"Stopped model on port {actual_port}")

            duration = time.perf_counter() - start_time
            result = TestResult(
                name="warmup",
                score=None,
                metadata={
                    "timings": timings,
                    "total_duration_seconds": duration,
                    "messages_sent": len(messages),
                },
            )

            self._log(f"Warm-up complete in {duration:.1f}s")
            return result

        except Exception as e:
            self._log(f"Warm-up failed: {e}", "red")
            return TestResult(
                name="warmup",
                score=None,
                metadata={"error": str(e), "timings": []},
            )

    def _build_messages(self) -> list[str]:
        """Build warm-up messages."""
        messages = []

        system_prompt = self.warmup_config.system_prompt or get_system_prompt(
            self.model.name,
            self.warmup_config.category,
            self.warmup_config.system_prompt,
        )
        if system_prompt:
            messages.append(system_prompt)

        messages.append(self.warmup_config.hello_message)
        messages.append(self.warmup_config.final_message)

        return messages

