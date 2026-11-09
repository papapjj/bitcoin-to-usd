from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_tensor_rule_v186_candidate import (
    _ensure_utf8_stdio,
    _install_graceful_shutdown_handlers,
    allocate_candidate_runner,
    build_parser,
)


def test_candidate_runner_configures_utf8_stdout_and_stderr(monkeypatch) -> None:
    configured: list[tuple[str, dict[str, str]]] = []

    class FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def reconfigure(self, **kwargs) -> None:
            configured.append((self.name, kwargs))

    monkeypatch.setattr(sys, "stdout", FakeStream("stdout"))
    monkeypatch.setattr(sys, "stderr", FakeStream("stderr"))

    _ensure_utf8_stdio()

    assert configured == [
        ("stdout", {"encoding": "utf-8", "errors": "replace"}),
        ("stderr", {"encoding": "utf-8", "errors": "replace"}),
    ]


def test_candidate_runner_translates_sigterm_to_keyboard_interrupt_and_restores(
    monkeypatch,
) -> None:
    installed: dict[int, object] = {}
    calls: list[tuple[int, object]] = []

    def fake_getsignal(sig: int) -> object:
        return installed.get(sig, "previous")

    def fake_signal(sig: int, handler: object) -> object:
        calls.append((sig, handler))
        installed[sig] = handler
        return handler

    monkeypatch.setattr(signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(signal, "signal", fake_signal)

    restore = _install_graceful_shutdown_handlers()
    try:
        sigterm_handler = installed[signal.SIGTERM]
        assert callable(sigterm_handler)
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            sigterm_handler(signal.SIGTERM, None)
    finally:
        restore()

    assert installed[signal.SIGTERM] == "previous"
    assert installed[signal.SIGINT] == "previous"
    assert len(calls) == 4


def test_graph_mt5_smoke_script_fails_cleanly_before_launch_without_runner(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/run_graph_mt5_smoke.py", "--runner-dir", str(tmp_path / "missing")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["ok"] is False
    assert payload["phase"] == "preflight"
    assert "terminal64.exe" in payload["failed"]


def test_candidate_runner_namespace_is_unique_under_short_workspace_root(tmp_path) -> None:
    first_parent, first = allocate_candidate_runner(tmp_path / "candidate-a")
    second_parent, second = allocate_candidate_runner(tmp_path / "candidate-b")

    expected_parent = (Path.cwd() / ".mt5_runners").resolve()
    assert first_parent == expected_parent
    assert second_parent == expected_parent
    assert first.parent == first_parent
    assert second.parent == second_parent
    assert first != second
    assert first.name.startswith("mt5_v186_")
    assert second.name.startswith("mt5_v186_")


def test_candidate_parser_accepts_real_ticks_model() -> None:
    args = build_parser().parse_args(["--model", "4"])

    assert args.model == 4
