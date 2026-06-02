"""Base class for all test modules."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class TestResult:
    """Single test result with standardized fields.

    Core fields:
    - name: Test identifier
    - score: Numeric score (0-10 or 0.0-1.0)
    - is_correct: Boolean correctness flag
    - raw_response: Raw model output
    - latency_ms: Execution time in milliseconds
    - metadata: Additional context

    Testing.py fields (optional):
    - category: Test category (code, debugging)
    - prompt: Original test prompt
    - response: Model response text
    - ran: Whether test code executed
    - correct: Whether test passed assertions
    """

    name: str
    score: float | None = None
    is_correct: bool = True
    raw_response: str = ""
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    # Testing.py specific fields
    category: str = ""
    prompt: str = ""
    response: str = ""
    ran: bool = False
    correct: bool = False


@dataclass
class TestSuiteResult:
    """Collection of test results with metadata."""

    module_name: str = ""
    results: list[TestResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: str = ""

    # Testing.py specific fields
    model_name: str = ""
    backend: str = ""
    overall_score: float | None = None

    @property
    def avg_score(self) -> float | None:
        """Average score across all results."""
        scores = [r.score for r in self.results if r.score is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def add_result(self, name: str, score: float | None = None, **kwargs) -> None:
        """Add a single result.
        
        Args:
            name: Test name/identifier
            score: Numeric score (optional)
            **kwargs: Additional fields passed to TestResult (is_correct, raw_response, etc.)
        """
        self.results.append(TestResult(name=name, score=score, **kwargs))

    def summary(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "module": self.module_name,
            "model_name": self.model_name,
            "backend": self.backend,
            "results": [
                {
                    "name": r.name,
                    "score": r.score,
                    "is_correct": r.is_correct,
                    "raw_response": r.raw_response[:500] if r.raw_response else "",
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                    "category": r.category,
                    "prompt": r.prompt[:200] if r.prompt else "",
                    "response": r.response[:500] if r.response else "",
                    "ran": r.ran,
                    "correct": r.correct,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
            "duration_seconds": round(self.duration_seconds, 2),
            "overall_score": self.overall_score,
            "avg_score": self.avg_score,
            "error": self.error,
        }


class TestModule(abc.ABC):
    """Base class for all test modules.

    Provides common infrastructure for:
    - Configuration loading
    - Model lifecycle (setup/teardown)
    - Metrics tracking
    - Result collection
    """

    def __init__(self, config: dict[str, Any], model: Any, host: str = "127.0.0.1"):
        """Initialize module.

        Args:
            config: Full rama config dict
            model: ModelInfo object with name, path, size_gb, family, quant
            host: Host for API calls
        """
        self.config = config
        self.model = model
        self.host = host
        self.results: list[TestResult] = []
        self._port: int | None = None

    @abc.abstractmethod
    def setup(self) -> None:
        """Prepare environment (start model, load context, etc.)."""
        pass

    @abc.abstractmethod
    def run(self) -> TestSuiteResult:
        """Execute the core test logic."""
        ...

    @abc.abstractmethod
    def teardown(self) -> None:
        """Clean up (stop model, clear ports, etc.)."""
        pass

    def _start_model(
        self,
        port: int,
        ctx_size: int | None = None,
        ctk: str | None = None,
        ctv: str | None = None,
        **kwargs,
    ) -> int:
        """Start model on given port. Returns the port."""
        from rama.launch import start_model

        result = start_model(
            self.model,
            port=port,
            ctx_size=ctx_size,
            ctk=ctk,
            ctv=ctv,
            config=self.config,
            config_item=kwargs,
        )
        return result.port

    def _stop_model(self, port: int) -> None:
        """Stop model on given port."""
        from rama.launch import stop_model
        from rama.launch import _wait_for_port_free

        stop_model(port, self.config)
        _wait_for_port_free(port, timeout=10.0)

    def _run_chat(self, port: int, prompt: str, **kwargs) -> str:
        """Send chat completion request."""
        from rama.testing import run_chat_completion

        return run_chat_completion(port, prompt, host=self.host, **kwargs)

    def _log(self, msg: str, style: str = "cyan") -> None:
        """Print styled log message."""
        console.print(f"[{style}][{self.model.name[:30]:<30}][/style] {msg}")

    def _track_duration(self, label: str) -> "DurationTracker":
        """Measure and log duration as a context manager."""

        class DurationTracker:
            def __init__(self, lbl: str, mod: TestModule) -> None:
                self.lbl = lbl
                self.mod = mod
                self.start = time.perf_counter()

            def __enter__(self) -> "DurationTracker":
                return self

            def __exit__(self, *args: Any) -> None:
                elapsed = time.perf_counter() - self.start
                self.mod._log(f"{self.lbl}: {elapsed:.1f}s", "dim")

        return DurationTracker(label, self)
