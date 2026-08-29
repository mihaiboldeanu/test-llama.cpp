"""Batch planner — pure function: batch config + models → list of launch plans.

This module is intentionally free of subprocess, PID files, and network calls.
It produces BatchPlan objects that BatchExecutor executes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .models import ModelInfo, ResolvedModelProfile


@dataclass(frozen=True)
class BatchItemPlan:
    """Immutable plan for a single model within a batch.

    Produced by BatchPlanner (pure). Executed by BatchExecutor (side-effects).
    """

    model: ModelInfo
    profile: ResolvedModelProfile
    backend: str
    ctk: str | None
    ctv: str | None
    ctx_size: int | None
    seed: int | None
    mtp_enabled: bool
    mtp_n_max: int
    mtp_n_min: int
    reasoning: bool | None
    reasoning_budget: int
    max_tokens: int | None
    config_item: dict  # Original batch config item for reference


@dataclass(frozen=True)
class BatchPlan:
    """Immutable plan for an entire batch run.

    Produced by BatchPlanner (pure). Executed by BatchExecutor (side-effects).
    Contains all information needed to reproduce the batch.
    """

    items: tuple[BatchItemPlan, ...]
    batch_type: str
    run_identity: str  # Immutable hash identifying this exact batch configuration
    fixture_hashes: dict[str, str]  # File paths → SHA256 hashes for reproducibility


@dataclass(frozen=True)
class BatchManifest:
    """Serializable manifest recording a completed batch run.

    Written to disk after batch completion for reproducibility and resumption.
    """

    run_identity: str
    batch_type: str
    fixture_hashes: dict[str, str]
    profiles: list[dict]  # ResolvedModelProfile serialized per model
    results: list[dict]  # Test results per model


def compute_run_identity(
    batch_type: str,
    items: list[BatchItemPlan],
    fixture_hashes: dict[str, str],
) -> str:
    """Compute immutable hash identifying a batch configuration.

    Two batches with the same models, profiles, and fixtures will have
    the same run_identity regardless of when they were executed.
    """
    hasher = hashlib.sha256()

    # Batch type
    hasher.update(batch_type.encode())

    # Items in order — model name, backend, config
    for item in items:
        hasher.update(item.model.name.encode())
        hasher.update(item.model.path.name.encode())
        hasher.update(str(item.model.size_bytes).encode())
        hasher.update(item.backend.encode())
        hasher.update(
            json.dumps(
                {
                    "ctk": item.ctk,
                    "ctv": item.ctv,
                    "ctx_size": item.ctx_size,
                    "seed": item.seed,
                    "mtp": item.mtp_enabled,
                    "reasoning": item.reasoning,
                },
                sort_keys=True,
            ).encode()
        )

    # Fixture hashes in sorted order
    for path, h in sorted(fixture_hashes.items()):
        hasher.update(f"{path}:{h}".encode())

    return hasher.hexdigest()[:16]


def compute_fixture_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for reproducibility tracking."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_profile(profile: ResolvedModelProfile) -> dict:
    """Serialize resolved profile to a JSON-serializable dict for manifest."""
    result = asdict(profile)
    # Model path needs serialization
    if profile.model:
        result["model"] = {
            "name": profile.model.name,
            "path": str(profile.model.path),
            "size_bytes": profile.model.size_bytes,
            "family_key": profile.model.family_key,
            "quantization": profile.model.quantization,
        }
    return result


@dataclass
class BatchPlannerInput:
    """All inputs needed to build a BatchPlan."""

    models: list[ModelInfo]  # Discovered models matching batch config
    batch_config_items: list[
        dict | str
    ]  # Parsed YAML items: [{"model": "x", ...}, "y"]
    batch_type: str  # code, perplexity, nihs, kv
    server_bin: Path

    # Global settings
    host: str = "127.0.0.1"
    start_port: int = 11435
    seed: int = 42
    config_path: str | None = None

    # Batch-type specific
    text_file: Path | None = None
    ctx_sizes: list[int] | None = None
    kv_quants: list[str] | None = None
    needle_locations: list[str] | None = None
    max_tokens: int | None = None
    reasoning: bool | None = None
    seeds: int = 10
    multi_seed: bool = True

    # Profile overrides applied to all items
    global_config_override: dict | None = None
    family_overrides: dict[str, dict] | None = None
    model_overrides: dict[str, dict] | None = None
    cli_override: dict | None = None


def plan_batch(inp: BatchPlannerInput) -> BatchPlan:
    """Build immutable batch plan from inputs. Pure function — no side effects."""
    items: list[BatchItemPlan] = []
    fixture_hashes: dict[str, str] = {}

    # Hash text fixture if present
    if inp.text_file and inp.text_file.exists():
        fixture_hashes[str(inp.text_file)] = compute_fixture_hash(inp.text_file)

    for config_item_raw in inp.batch_config_items:
        if isinstance(config_item_raw, str):
            model_name = config_item_raw
            config_item: dict = {}
        else:
            model_name = config_item_raw.get("model", "")
            config_item = {k: v for k, v in config_item_raw.items() if k != "model"}

        # Find matching model
        model = _find_model(inp.models, model_name)
        if model is None:
            continue  # Skip unknown models during planning

        items.append(
            BatchItemPlan(
                model=model,
                profile=_resolve_item_profile(model, config_item, inp),
                backend=config_item.get("backend", "llama.cpp"),
                ctk=config_item.get("ctk"),
                ctv=config_item.get("ctv"),
                ctx_size=config_item.get("ctx"),
                seed=config_item.get("seed", inp.seed),
                mtp_enabled=bool(config_item.get("mtp", False)),
                mtp_n_max=int(config_item.get("mtp_n_max", 3)),
                mtp_n_min=int(config_item.get("mtp_n_min", 0)),
                reasoning=config_item.get("reasoning", inp.reasoning),
                reasoning_budget=int(config_item.get("reasoning_budget", 1024)),
                max_tokens=config_item.get("max_tokens", inp.max_tokens),
                config_item=config_item,
            )
        )

    run_identity = compute_run_identity(inp.batch_type, items, fixture_hashes)

    return BatchPlan(
        items=tuple(items),
        batch_type=inp.batch_type,
        run_identity=run_identity,
        fixture_hashes=fixture_hashes,
    )


def _find_model(models: list[ModelInfo], model_name: str) -> ModelInfo | None:
    """Find model by name with progressive fuzzy matching."""
    # Exact match
    for m in models:
        if model_name in (m.name, m.path.name):
            return m

    # Case-insensitive
    for m in models:
        if (
            m.name.lower() == model_name.lower()
            or m.path.name.lower() == model_name.lower()
        ):
            return m

    # Fuzzy substring
    for m in models:
        if (
            model_name.lower() in m.name.lower()
            or model_name.lower() in m.path.name.lower()
        ):
            return m

    return None


def _resolve_item_profile(
    model: ModelInfo,
    config_item: dict,
    inp: BatchPlannerInput,
) -> ResolvedModelProfile:
    """Resolve profile for a single batch item.

    Uses the same precedence chain as ProfileResolver:
    CLI > Batch Override > Global Config > Family Defaults > Built-in.
    """
    from rama.families import get_registry

    from .resolver import ProfileResolver

    resolver = ProfileResolver(get_registry().get(model.family_key))
    return resolver.resolve(
        model=model,
        global_config=inp.global_config_override,
        family_override=(inp.family_overrides or {}).get(model.family_key),
        model_override=(inp.model_overrides or {}).get(model.name),
        batch_override=config_item or None,
        cli_override=inp.cli_override,
    )
