"""Contract tests for the model-process boundary using a tiny local HTTP server."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from rama.domain.runner import RunnerInput, _terminate, run

if TYPE_CHECKING:
    from pathlib import Path


_HEALTH_SERVER = """
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass

HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
"""


def test_runner_waits_for_health_and_persists_pid(
    tmp_path: Path, free_tcp_port: int
) -> None:
    """A healthy server produces a PID record that can be managed by Rama."""
    args = (sys.executable, "-c", _HEALTH_SERVER.replace("PORT", str(free_tcp_port)))
    runner_input = RunnerInput(
        args=args,
        model_name="contract-model",
        port=free_tcp_port,
        host="127.0.0.1",
        pid_dir=tmp_path,
    )

    result = run(runner_input, max_wait=5, log_disable=True)
    pid_file = tmp_path / f"model_contract-model_{free_tcp_port}.pid"
    try:
        assert result.pid > 0
        assert pid_file.exists()
    finally:
        _terminate(result.pid)
        pid_file.unlink(missing_ok=True)


def test_runner_removes_pid_record_when_process_crashes(
    tmp_path: Path, free_tcp_port: int
) -> None:
    """A pre-health crash reports context and leaves no stale PID record."""
    runner_input = RunnerInput(
        args=(sys.executable, "-c", "raise SystemExit(23)"),
        model_name="crashing-model",
        port=free_tcp_port,
        host="127.0.0.1",
        pid_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="exit code 23"):
        run(runner_input, max_wait=2, log_disable=True)

    assert not (tmp_path / f"model_crashing-model_{free_tcp_port}.pid").exists()
