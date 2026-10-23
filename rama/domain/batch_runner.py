"""Batch executor — runs BatchPlan using domain.runner for process lifecycle.

Handles launching models via the planner/runner interface, collecting results,
writing manifests with resolved profiles and fixture hashes, and supporting
resumable batches.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .batch_planner import (
    BatchItemPlan,
    BatchManifest,
    BatchPlan,
    hash_profile,
)
from .planner import PlanRequest, plan
from .runner import RunnerInput, RunnerResult, run

if TYPE_CHECKING:
    from pathlib import Path

    from .models import ResolvedProfile

logger = logging.getLogger(__name__)


class BatchTestExecutor(Protocol):
    """Protocol for executing tests against a running model.

    Implementations handle the actual testing (code, perplexity, NIHS, KV).
    The executor manages process lifecycle; this handles test logic.
    """

    def execute(
        self,
        item: BatchItemPlan,
        port: int,
        host: str,
        profile: ResolvedProfile,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BatchExecutorResult:
    """Result of executing a single batch item."""

    item: BatchItemPlan
    result: dict[str, Any]
    status: str  # ok, failed, skipped
    duration_seconds: float
    port: int | None = None


@dataclass
class BatchExecutorInput:
    """Inputs for executing a BatchPlan."""

    plan: BatchPlan
    executor: BatchTestExecutor
    server_bin: Path
    host: str = "127.0.0.1"
    start_port: int = 11435
    output_dir: Path | None = None
    pid_dir: Path | None = None
    dry_run: bool = False
    quiet: bool = False

    # Profile resolution context
    global_config_override: dict | None = None
    cli_override: dict | None = None





def _launch_model(
    item: BatchItemPlan,
    server_bin: Path,
    port: int,
    host: str,
) -> RunnerResult:
    """Launch a model using planner + runner."""
    profile = item.profile
    mtp = profile.mtp if isinstance(profile.mtp, dict) else {}

    argv = plan(
        PlanRequest(
            server_bin=server_bin,
            model=item.model,
            profile=profile,
            host=host,
            port=port,
            backend=item.backend,
            mtp=bool(mtp.get("enabled", False)),
        )
    )

    run_env = os.environ.copy()
    if profile.env:
        run_env.update({str(k): str(v) for k, v in profile.env.items()})

    # Execute (side effects: Popen, health check, PID file)
    return run(
        RunnerInput(
            args=tuple(argv),
            model_name=item.model.name,
            port=port,
            host=host,
            pid_dir=None,  # Use default PID dir
            env=run_env,
        )
    )


def _stop_model(port: int, model_name: str) -> None:
    """Stop a model server by port."""
    from rama.launch import stop_model

    stop_model(port, None)


def _dry_run_item(
    item: BatchItemPlan,
    server_bin: Path,
    host: str,
) -> BatchExecutorResult:
    """Dry-run a batch item: plan argv but don't launch."""
    profile = item.profile
    mtp = profile.mtp if isinstance(profile.mtp, dict) else {}
    argv = plan(
        PlanRequest(
            server_bin=server_bin,
            model=item.model,
            profile=profile,
            host=host,
            port=None,  # No port for dry run
            backend=item.backend,
            mtp=bool(mtp.get("enabled", False)),
        )
    )

    return BatchExecutorResult(
        item=item,
        result={
            "argv": argv,
            "profile": hash_profile(profile),
            "status": "dry_run",
        },
        status="ok",
        duration_seconds=0.0,
    )


def _write_manifest(
    plan_obj: BatchPlan,
    results: list[BatchExecutorResult],
    output_dir: Path,
) -> Path:
    """Write batch manifest with resolved profiles and fixture hashes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"manifest_{plan_obj.run_identity}.json"
    temp_path = manifest_path.with_suffix(".tmp")

    # Serialize manifest
    manifest_dict = {
        "run_identity": plan_obj.run_identity,
        "batch_type": plan_obj.batch_type,
        "fixture_hashes": plan_obj.fixture_hashes,
        "profiles": [hash_profile(r.item.profile) for r in results],
        "results": [r.result for r in results],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_items": len(results),
        "ok_count": sum(1 for r in results if r.status == "ok"),
        "failed_count": sum(1 for r in results if r.status == "failed"),
    }

    temp_path.write_text(json.dumps(manifest_dict, indent=2, default=str))
    temp_path.replace(manifest_path)

    return manifest_path


def load_manifest(
    output_dir: Path, run_identity: str | None = None
) -> BatchManifest | None:
    """Load a previous batch manifest for resumption.

    If run_identity is provided, loads that specific manifest.
    Otherwise, loads the most recent manifest in the output directory.
    """
    if run_identity:
        manifest_path = output_dir / f"manifest_{run_identity}.json"
        if manifest_path.exists():
            return _parse_manifest(manifest_path)
        return None

    # Find most recent manifest
    manifests = sorted(
        output_dir.glob("manifest_*.json"), key=lambda p: p.stat().st_mtime
    )
    if manifests:
        return _parse_manifest(manifests[-1])
    return None


def _parse_manifest(path: Path) -> BatchManifest | None:
    """Parse manifest JSON file."""
    try:
        data = json.loads(path.read_text())
        return BatchManifest(
            run_identity=data["run_identity"],
            batch_type=data["batch_type"],
            fixture_hashes=data.get("fixture_hashes", {}),
            profiles=data.get("profiles", []),
            results=data.get("results", []),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse manifest %s: %s", path, e)
        return None


def can_resume(output_dir: Path, plan_obj: BatchPlan) -> bool:
    """Check if a batch can be resumed from existing manifest."""
    manifest = load_manifest(output_dir, plan_obj.run_identity)
    if manifest is None:
        return False

    return len(manifest.results) >= len(plan_obj.items)
