"""Enhanced needle-in-haystack module.

Tests retrieval capability with:
- Multi-needle design (5-10 needles)
- Distractors with similar but misleading values (lexical, topical, irrelevant)
- Multiple locations (beginning, middle, end, percentage)
- Realistic scenarios (API keys, config values, secrets)
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field

from rama.core import find_free_port
from pathlib import Path
from typing import Any

from rich.console import Console

from .base import TestModule, TestResult

console = Console()


@dataclass
class Distractor:
    """A distractor to insert into the context."""

    type: str  # "lexical", "topical", "irrelevant"
    text: str
    relates_to: str | None = None  # Which needle this distractor mimics


@dataclass
class Needle:
    """A needle to hide in the context."""

    token: str
    location: str  # "beginning", "middle", "end", or percentage like "25%"
    label: str = ""  # Label for the needle (e.g., "product_api_key")


@dataclass
class EnhancedNIHSConfig:
    """Configuration for enhanced NIHS testing."""

    num_needles: int = 8  # Number of needles per test
    distractors_per_needle: int = 2  # 1-3 distractors per needle
    distractor_types: list[str] = field(
        default_factory=lambda: ["lexical", "topical", "irrelevant"]
    )
    needle_locations: list[str] = field(
        default_factory=lambda: ["beginning", "middle", "end", "25%", "50%", "75%"]
    )
    context_files: list[str] = field(
        default_factory=lambda: ["wikitext-2/wiki.valid.tokens"]
    )
    context_size: int = 65536
    seed: int | None = None


class EnhancedNIHSModule(TestModule):
    """Enhanced needle-in-haystack with multi-needle and distractors.

    Inspired by RULER, ContextRot, NoLiMa research:
    - Multiple needles (5-10) per test
    - Distractors of different types
    - Variable needle locations
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Any,
        nihs_config: EnhancedNIHSConfig | None = None,
        host: str = "127.0.0.1",
    ):
        super().__init__(config, model, host)
        self.nihs_config = nihs_config or EnhancedNIHSConfig()
        self._wiki_cache: dict[str, str] = {}

    def run(self) -> TestResult:
        """Run enhanced NIHS tests."""
        start_time = time.perf_counter()
        self._log("Starting enhanced NIHS tests")

        # Load context text
        context_text = self._load_context()
        if not context_text:
            return TestResult(
                name="enhanced_nihs",
                score=None,
                metadata={"error": "No context text available"},
            )

        # Generate needles and distractors
        needles, distractors = self._generate_test_elements()

        # Get a port
        port = find_free_port(self.config)

        try:
            # Start model
            actual_port = self._start_model(
                port=port,
                ctx_size=self.nihs_config.context_size,
                ctk="f16",
                ctv="f16",
            )
            self._log(f"Model started on port {actual_port}")

            # Run tests for each needle
            results = []
            for i, needle in enumerate(needles):
                self._log(f"  Testing needle {i+1}/{len(needles)} at {needle.location}")

                # Select distractors for this needle
                needle_distractors = random.sample(
                    distractors,
                    min(self.nihs_config.distractors_per_needle, len(distractors)),
                )

                # Build context with needle and distractors
                context_with_needle = self._build_context(
                    context_text, needle, needle_distractors
                )

                # Run test
                test_result = self._run_single_needle_test(
                    actual_port, needle, context_with_needle
                )
                results.append(test_result)

            # Stop model
            self._stop_model(actual_port)
            self._log(f"Stopped model on port {actual_port}")

            # Calculate metrics
            found_count = sum(1 for r in results if r.get("found"))
            accuracy = found_count / len(results) if results else 0

            duration = time.perf_counter() - start_time
            result = TestResult(
                name="enhanced_nihs",
                score=accuracy * 10,
                metadata={
                    "results": results,
                    "accuracy": accuracy,
                    "found": found_count,
                    "total": len(results),
                    "duration_seconds": duration,
                },
            )

            self._log(
                f"NIHS complete: {found_count}/{len(results)} found ({accuracy:.0%})"
            )
            return result

        except Exception as e:
            self._log(f"NIHS test failed: {e}", "red")
            return TestResult(
                name="enhanced_nihs",
                score=None,
                metadata={"error": str(e), "results": []},
            )

    def _load_context(self) -> str | None:
        """Load context text from configured files."""
        for file_pattern in self.nihs_config.context_files:
            ctx_path = Path(file_pattern)
            if ctx_path.exists():
                if str(ctx_path) not in self._wiki_cache:
                    self._wiki_cache[str(ctx_path)] = ctx_path.read_text()
                return self._wiki_cache[str(ctx_path)]

        self._log("  No context files found", "yellow")
        return None

    def _generate_test_elements(
        self,
    ) -> tuple[list[Needle], list[Distractor]]:
        """Generate needles and distractors for testing.
        
        Creates realistic scenarios like:
        - needle: "product_api_key is BLUE-123"
        - distractor: "deprecated_api_key is RED-123" (lexical, similar format)
        """
        needles = []
        distractors = []

        rng = random.Random(self.nihs_config.seed)

        # Realistic needle templates with labels
        needle_templates = [
            ("product_api_key", "is {}"),
            ("secret_token", "equals {}"),
            ("database_password", "is {}"),
            ("api_secret", "set to {}"),
            ("auth_key", "is {}"),
            ("encryption_key", "is {}"),
            ("admin_password", "is {}"),
            ("jwt_secret", "equals {}"),
            ("stripe_key", "is {}"),
            ("aws_secret", "is {}"),
        ]

        # Realistic distractor templates (similar but misleading)
        distractor_templates = {
            "lexical": [
                ("deprecated_api_key", "is {}"),
                ("old_product_key", "was {}"),
                ("backup_token", "is {}"),
                ("test_secret", "is {}"),
                ("legacy_auth_key", "was {}"),
                ("temporary_password", "is {}"),
                ("expired_key", "was {}"),
                ("staging_secret", "is {}"),
                ("development_token", "is {}"),
                ("secondary_secret", "is {}"),
            ],
            "topical": [
                "The system uses multi-factor authentication for all API calls.",
                "API keys should be rotated every 90 days for security.",
                "The authentication module handles token validation.",
                "Secrets are stored in environment variables.",
                "The API gateway validates all incoming requests.",
                "Rate limiting is applied to prevent abuse.",
                "All keys must be at least 32 characters long.",
                "The service uses HMAC-SHA256 for signature verification.",
            ],
            "irrelevant": [
                f"SECTION-{rng.getrandbits(64):016x}",
                f"MARKER-{rng.getrandbits(32):08x}",
                f"PLACEHOLDER-{rng.getrandbits(48):012x}",
            ],
        }

        color_codes = ["BLUE", "RED", "GREEN", "GOLD", "SILVER", "BRONZE", "IVORY", "EBONY"]
        hex_suffixes = [rng.getrandbits(16) for _ in range(20)]

        # Generate needles with labels
        for i in range(self.nihs_config.num_needles):
            template_idx = i % len(needle_templates)
            label, format_str = needle_templates[template_idx]
            color = color_codes[i % len(color_codes)]
            hex_val = f"{hex_suffixes[i]:04X}"
            token = f"{color}-{hex_val}"
            full_needle = f"{label} {format_str.format(token)}"
            location = rng.choice(self.nihs_config.needle_locations)
            needles.append(Needle(token=token, location=location, label=label))

        # Generate distractors for each needle
        for needle in needles:
            for dtype in self.nihs_config.distractor_types:
                num_distractors = rng.randint(1, self.nihs_config.distractors_per_needle)
                templates = distractor_templates[dtype]

                for _ in range(num_distractors):
                    if dtype == "lexical":
                        distractor_template_idx = rng.randint(0, len(templates) - 1)
                        dist_label, dist_format = templates[distractor_template_idx]
                        # Use different color/value to make it clearly different but similar format
                        color = color_codes[rng.randint(0, len(color_codes) - 1)]
                        hex_val = f"{rng.getrandbits(16):04X}"
                        text = f"{dist_label} {dist_format.format(f'{color}-{hex_val}')}"
                    elif dtype == "topical":
                        text = templates[rng.randint(0, len(templates) - 1)]
                    else:
                        text = templates[rng.randint(0, len(templates) - 1)]

                    distractors.append(Distractor(
                        type=dtype,
                        text=text,
                        relates_to=needle.label,
                    ))

        return needles, distractors

    def _build_context(
        self,
        context: str,
        needle: Needle,
        distractors: list[Distractor],
    ) -> str:
        """Build context with needle and distractors inserted."""
        # Insert distractors first
        context_with_distractors = context
        for distractor in distractors:
            # Insert distractors at random positions
            pos = min(
                int(len(context_with_distractors) * random.random()),
                len(context_with_distractors),
            )
            context_with_distractors = (
                context_with_distractors[:pos]
                + f"\n\n[DISTRACTORY:{distractor.type}] {distractor.text}\n\n"
                + context_with_distractors[pos:]
            )

        # Insert needle at specified location
        context_with_needle = self._insert_at_location(
            context_with_distractors,
            needle,
            needle.location,
        )

        return context_with_needle

    def _insert_at_location(
        self, text: str, needle: Needle, location: str
    ) -> str:
        """Insert needle at specified location in text."""
        needle_line = f"\n\n{needle.label} is {needle.token}\n\n"

        if location == "beginning":
            return needle_line + text
        if location in {"middle", "50%"}:
            mid = len(text) // 2
            return text[:mid] + needle_line + text[mid:]
        if location == "end":
            return text + needle_line

        # Parse percentage
        if location.endswith("%"):
            pct = int(location[:-1]) / 100.0
            pos = int(len(text) * pct)
            return text[:pos] + needle_line + text[pos:]

        return text + needle_line

    def _run_single_needle_test(
        self,
        port: int,
        needle: Needle,
        context: str,
    ) -> dict[str, Any]:
        """Run a single needle-in-haystack test."""
        prompt = self._build_prompt(context, needle)

        response = self._run_chat(
            port,
            prompt,
            temperature=0.0,
            max_tokens=256,
            seed=self.nihs_config.seed if self.nihs_config.seed is not None else 42,
        )

        # Check if needle was found (case-insensitive)
        found = needle.token.lower() in response.strip().lower()

        return {
            "needle_label": needle.label,
            "needle_token": needle.token,
            "location": needle.location,
            "found": found,
            "response": response[:200],
            "response_length": len(response),
        }

    def _build_prompt(self, context: str, needle: Needle) -> str:
        """Build the full test prompt for a specific needle.
        
        Asks about the specific label (e.g., "product_api_key") to test
        if the model can find the right value among distractors.
        """
        return (
            f"Read the following text carefully. Then answer the question at the end.\n\n"
            f"TEXT:\n{context}\n\n"
            f"QUESTION: What is the value of {needle.label} mentioned in the text? "
            f"Only respond with the value itself, nothing else."
        )
