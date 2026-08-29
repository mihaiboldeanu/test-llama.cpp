"""Process runner for model server subprocess management.

Handles launching llama-server, PID persistence, health checking with
exponential backoff, and cleanup on failure. Pure side-effect module —
no config resolution or argv construction logic.
"""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerInput:
    """Immutable input for launching and monitoring a model server."""

    args: tuple[str, ...]
    model_name: str
    port: int
    host: str
    pid_dir: Path | None = None  # None -> use default ~/.local/state/rama/pids


@dataclass(frozen=True)
class RunnerResult:
    """Successful launch outcome."""

    pid: int


def _get_default_pid_dir() -> Path:
    return Path.home() / ".local" / "state" / "rama" / "pids"


def _pid_exists(pid: int) -> bool:
    """Check if a process exists."""
    try:
        import os

        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _terminate(pid: int, grace_seconds: float = 8.0) -> bool:
    """Best-effort terminate a process, escalating to SIGKILL if needed."""
    import os

    if not pid:
        return False

    # Kill the process first, escalate to the whole group if needed.
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.2)

    # Process didn't die — try the whole group (start_new_session=True means we own it).
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.2)

    # Force kill — process first, then group.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)

    return not _pid_exists(pid)


def _cleanup_pid_files(pid_dir: Path, port: int, model_name: str) -> None:
    """Remove PID files for a given model/port combination."""
    patterns = [
        f"model_{model_name}_{port}.pid",
    ]
    # Also try wildcard for any leftover files on this port
    for pattern in patterns:
        pid_file = pid_dir / pattern
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)

    # Clean up any stale model_*_<port>.pid files
    for pid_file in pid_dir.glob(f"model_*_{port}.pid"):
        pid_file.unlink(missing_ok=True)


def run(
    inp: RunnerInput,
    max_wait: int = 120,
    log_disable: bool = False,
) -> RunnerResult:
    """Launch model server and wait for health check to pass.

    Raises:
        RuntimeError: if process crashes or health check times out.
    """
    pid_dir = inp.pid_dir or _get_default_pid_dir()
    pid_dir.mkdir(parents=True, exist_ok=True)

    # Launch subprocess
    proc = subprocess.Popen(
        inp.args,
        start_new_session=True,
        stdout=subprocess.DEVNULL if log_disable else None,
        stderr=subprocess.DEVNULL if log_disable else None,
    )

    # Persist PID file
    pid_file = pid_dir / f"model_{inp.model_name}_{inp.port}.pid"
    pid_data = {
        "pid": proc.pid,
        "name": inp.model_name,
        "port": inp.port,
    }
    pid_file.write_text(json.dumps(pid_data))

    # Health check loop with exponential backoff
    total_wait = 0
    attempt = 0

    while total_wait < max_wait:
        attempt += 1
        if attempt <= 5:
            sleep_time = 1
        elif attempt <= 10:
            sleep_time = 2
        elif attempt <= 15:
            sleep_time = 4
        else:
            sleep_time = min(1 << (attempt - 10), 32)
        sleep_time = min(sleep_time, max_wait - total_wait)

        time.sleep(sleep_time)
        total_wait += sleep_time

        # Check if process crashed
        exit_code = proc.poll()
        if exit_code is not None:
            _terminate(proc.pid)
            _cleanup_pid_files(pid_dir, inp.port, inp.model_name)
            cmd_str = " ".join(inp.args)
            raise RuntimeError(
                f"llama-server crashed with exit code {exit_code}. Command: {cmd_str}"
            )

        # Health check via HTTP
        try:
            urllib.request.urlopen(f"http://{inp.host}:{inp.port}/health")
            return RunnerResult(pid=proc.pid)
        except Exception as e:
            logger.debug("health check failed (attempt %d), retrying: %s", attempt, e)
            continue

    # Timeout — clean up and raise
    _terminate(proc.pid)
    _cleanup_pid_files(pid_dir, inp.port, inp.model_name)
    raise RuntimeError(f"Model failed to start within {max_wait} seconds")
