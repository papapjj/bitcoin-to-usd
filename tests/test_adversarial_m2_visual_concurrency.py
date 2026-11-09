"""
tests/test_adversarial_m2_visual_concurrency.py

Empirical Challenger Adversarial Test Suite for Milestone 2:
Adversarial Single-Instance Visual Guard & Concurrency Stress Challenge.

Focus Areas:
1. Single-Instance Visual Guard & Clamping:
   - CLI flags: --mt5-visual with --max-workers 4 and --max-workers 8.
   - Functional/kwarg invocations: mt5_visual=True with max_workers 4, 8, and string variations.
   - ThreadPoolExecutor suppression verification: ensure execution is strictly sequential (effective_max_workers == 1).
   - Headless contrast: mt5_visual=False does NOT clamp max_workers.
2. Concurrent Visual Launches & MT5StrategyTesterGlobalLock Named Mutex Stress:
   - 4-process and 8-process concurrent contention: verifies zero critical-section overlap and no deadlocks.
   - Process abandonment recovery: verifies WAIT_ABANDONED (0x00000080) handling when a process abruptly terminates holding the mutex.
   - Rapid acquire-release cycling under high concurrency.
3. Timeout Scaling Probing:
   - Default timeout resolution: 600s when visual=True vs 300s when visual=False.
   - Explicit overrides: custom timeout preserved in both modes.
   - Deep propagation into evaluate_window and MT5BacktestConfig.
4. Strict tester.ini Validation:
   - Exact line formatting for Visual=1 and ShutdownTerminal=1 in visual mode.
   - Exact line formatting for Visual=0 in non-visual mode.
   - INI parser integrity: check for section consistency, no duplicate or conflicting keys.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import run_frozen_temporal_generalization as runner
from self_growing_trader.mt5_runner import (
    MT5BacktestConfig,
    MT5StrategyTesterGlobalLock,
    write_tester_ini,
)


# ==============================================================================
# Helper worker functions for multiprocessing concurrency stress
# ==============================================================================

def _concurrency_stress_worker(
    lock_file: str,
    active_counter: Any,
    violation_counter: Any,
    completed_counter: Any,
    hold_time: float,
) -> None:
    """Worker process that acquires MT5StrategyTesterGlobalLock and verifies mutual exclusion."""
    try:
        with MT5StrategyTesterGlobalLock(lock_file):
            # Critical section entry: increment active processes
            with active_counter.get_lock():
                active_counter.value += 1
                current_active = active_counter.value

            # Concurrency invariant: only 1 process may be inside the critical section
            if current_active != 1:
                with violation_counter.get_lock():
                    violation_counter.value += 1

            time.sleep(hold_time)

            # Critical section exit: decrement active processes
            with active_counter.get_lock():
                if active_counter.value != 1:
                    with violation_counter.get_lock():
                        violation_counter.value += 1
                active_counter.value -= 1

            with completed_counter.get_lock():
                completed_counter.value += 1
    except Exception:
        with violation_counter.get_lock():
            violation_counter.value += 1


def _rapid_cycling_worker(
    lock_file: str,
    cycles: int,
    violation_counter: Any,
    completed_counter: Any,
) -> None:
    """Worker process that repeatedly acquires and releases the mutex in quick succession."""
    try:
        for _ in range(cycles):
            with MT5StrategyTesterGlobalLock(lock_file):
                time.sleep(0.01)
        with completed_counter.get_lock():
            completed_counter.value += 1
    except Exception:
        with violation_counter.get_lock():
            violation_counter.value += 1


# ==============================================================================
# 1. ADVERSARIAL SINGLE-INSTANCE VISUAL GUARD CLAMPING PROBES
# ==============================================================================

class TestAdversarialVisualWorkerClamping:
    """Stress tests single-instance guard when visual mode is enabled."""

    @pytest.fixture
    def mock_environment(self, tmp_path: Path):
        candidate = tmp_path / "candidate.pkl"
        candidate.write_bytes(b"dummy_candidate_payload")
        runner_dir = tmp_path / "runner"
        runner_dir.mkdir()
        out_json = tmp_path / "summary.json"
        config_path = ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml"
        return {
            "candidate": candidate,
            "runner_dir": runner_dir,
            "out_json": out_json,
            "config": config_path,
            "tmp_path": tmp_path,
        }

    @pytest.mark.parametrize("requested_workers", [4, 8, 16, 32])
    def test_cli_parsing_and_run_clamps_workers_to_one_in_visual_mode(
        self, mock_environment: dict[str, Any], requested_workers: int, monkeypatch
    ) -> None:
        """Adversarially probe passing --mt5-visual with --max-workers 4, 8, 16, 32.

        Verify that:
        1. CLI parses the requested workers into args.max_workers.
        2. run_temporal_generalization strictly clamps max_workers = 1.
        3. Parallel windows flag is False.
        4. evaluate_window receives visual=True.
        """
        eval_calls: list[dict[str, Any]] = []

        def mock_eval(**kwargs: Any) -> dict[str, Any]:
            eval_calls.append(kwargs)
            return {
                "window": kwargs["index"],
                "completed": True,
                "metrics": {"trades": 5},
                "runner_clone": str(mock_environment["tmp_path"] / f"clone_{kwargs['index']}"),
            }

        monkeypatch.setattr(runner, "evaluate_window", mock_eval)

        # Ensure ThreadPoolExecutor is NEVER instantiated when clamped to 1
        with patch("scripts.run_frozen_temporal_generalization.ThreadPoolExecutor") as mock_tpe:
            cli_args = [
                "--candidate", str(mock_environment["candidate"]),
                "--config", str(mock_environment["config"]),
                "--runner-source", str(mock_environment["runner_dir"]),
                "--windows", "2025.02.20:2025.02.24,2025.02.24:2025.02.28,2025.03.01:2025.03.05",
                "--max-workers", str(requested_workers),
                "--output", str(mock_environment["out_json"]),
                "--mt5-visual",
            ]
            parser = runner.build_parser()
            parsed = parser.parse_args(cli_args)
            assert parsed.max_workers == requested_workers
            assert parsed.mt5_visual is True

            summary = runner.run_temporal_generalization(parsed)

            # Assert strict clamping
            assert summary["visual"] is True
            assert summary["max_workers"] == 1
            assert summary["parallel_windows"] is False
            assert summary["timeout_sec"] == 600  # Default visual timeout
            assert len(summary["windows"]) == 3
            assert len(eval_calls) == 3

            # ThreadPoolExecutor must NOT have been called
            mock_tpe.assert_not_called()

            # Every window evaluation must have received visual=True
            for call_kw in eval_calls:
                assert call_kw["visual"] is True
                assert call_kw["timeout_sec"] == 600

    @pytest.mark.parametrize("kwarg_workers", [4, 8])
    def test_direct_kwarg_invocation_clamps_workers_to_one(
        self, mock_environment: dict[str, Any], kwarg_workers: int, monkeypatch
    ) -> None:
        """Probe passing max_workers via python kwargs to run_temporal_generalization with mt5_visual=True."""
        eval_calls: list[dict[str, Any]] = []

        def mock_eval(**kwargs: Any) -> dict[str, Any]:
            eval_calls.append(kwargs)
            return {
                "window": kwargs["index"],
                "completed": True,
                "metrics": {},
                "runner_clone": str(mock_environment["tmp_path"] / f"clone_{kwargs['index']}"),
            }

        monkeypatch.setattr(runner, "evaluate_window", mock_eval)

        summary = runner.run_temporal_generalization(
            candidate=mock_environment["candidate"],
            config=mock_environment["config"],
            runner_source=mock_environment["runner_dir"],
            windows="2025.02.20:2025.02.24,2025.02.24:2025.02.28",
            max_workers=kwarg_workers,
            output=mock_environment["out_json"],
            mt5_visual=True,
        )

        assert summary["visual"] is True
        assert summary["max_workers"] == 1
        assert summary["parallel_windows"] is False

    def test_contrast_headless_mode_does_not_clamp_workers(
        self, mock_environment: dict[str, Any], monkeypatch
    ) -> None:
        """When mt5_visual=False, max_workers=4 must NOT be clamped to 1."""
        def mock_eval(**kwargs: Any) -> dict[str, Any]:
            return {
                "window": kwargs["index"],
                "completed": True,
                "metrics": {},
                "runner_clone": str(mock_environment["tmp_path"] / f"clone_{kwargs['index']}"),
            }

        monkeypatch.setattr(runner, "evaluate_window", mock_eval)

        summary = runner.run_temporal_generalization(
            candidate=mock_environment["candidate"],
            config=mock_environment["config"],
            runner_source=mock_environment["runner_dir"],
            windows="2025.02.20:2025.02.24,2025.02.24:2025.02.28,2025.03.01:2025.03.05,2025.03.05:2025.03.09",
            max_workers=4,
            output=mock_environment["out_json"],
            mt5_visual=False,
        )

        assert summary["visual"] is False
        assert summary["max_workers"] == 4
        assert summary["parallel_windows"] is True
        assert summary["timeout_sec"] == 300  # Default headless timeout


# ==============================================================================
# 2. ADVERSARIAL CONCURRENCY & MT5StrategyTesterGlobalLock NAMED MUTEX STRESS
# ==============================================================================

class TestAdversarialNamedMutexConcurrency:
    """Stress tests MT5StrategyTesterGlobalLock across concurrent processes on Windows."""

    def test_concurrent_visual_launches_serialization_4_processes(self, tmp_path: Path) -> None:
        """Simulate 4 concurrent processes competing for MT5StrategyTesterGlobalLock.

        Verify that:
        1. All 4 processes enter the critical section sequentially without overlapping.
        2. No deadlocks occur.
        3. active_counter never exceeds 1.
        4. All 4 processes finish successfully.
        """
        lock_file = str(tmp_path / "global.lock")
        ctx = mp.get_context("spawn")

        active_counter = ctx.Value("i", 0)
        violation_counter = ctx.Value("i", 0)
        completed_counter = ctx.Value("i", 0)

        num_processes = 4
        processes = [
            ctx.Process(
                target=_concurrency_stress_worker,
                args=(lock_file, active_counter, violation_counter, completed_counter, 0.1),
            )
            for _ in range(num_processes)
        ]

        start_time = time.monotonic()
        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=20.0)
            assert not p.is_alive(), "Process deadlocked or timed out waiting for global mutex!"

        elapsed = time.monotonic() - start_time

        assert violation_counter.value == 0, f"Concurrency violation detected! Overlapping critical sections: {violation_counter.value}"
        assert completed_counter.value == num_processes, f"Expected {num_processes} processes to complete, got {completed_counter.value}"
        # With 4 processes each holding 0.1s sequentially, elapsed time must be >= 0.4s
        assert elapsed >= 0.35, f"Execution was too fast ({elapsed:.2f}s) to be properly serialized!"

    def test_concurrent_visual_launches_serialization_8_processes(self, tmp_path: Path) -> None:
        """Simulate 8 concurrent processes competing for MT5StrategyTesterGlobalLock.

        Verify proper scaling and serialization under higher contention.
        """
        lock_file = str(tmp_path / "global_8.lock")
        ctx = mp.get_context("spawn")

        active_counter = ctx.Value("i", 0)
        violation_counter = ctx.Value("i", 0)
        completed_counter = ctx.Value("i", 0)

        num_processes = 8
        processes = [
            ctx.Process(
                target=_concurrency_stress_worker,
                args=(lock_file, active_counter, violation_counter, completed_counter, 0.05),
            )
            for _ in range(num_processes)
        ]

        start_time = time.monotonic()
        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=30.0)
            assert not p.is_alive(), "Process deadlocked in 8-process stress test!"

        elapsed = time.monotonic() - start_time

        assert violation_counter.value == 0, "Mutual exclusion violated in 8-process stress test"
        assert completed_counter.value == num_processes
        assert elapsed >= 0.35

    def test_rapid_acquire_release_cycling_stress(self, tmp_path: Path) -> None:
        """Multiple processes executing rapid acquire/release cycles to stress mutex handles."""
        lock_file = str(tmp_path / "rapid_global.lock")
        ctx = mp.get_context("spawn")

        violation_counter = ctx.Value("i", 0)
        completed_counter = ctx.Value("i", 0)

        num_processes = 4
        cycles_per_process = 5

        processes = [
            ctx.Process(
                target=_rapid_cycling_worker,
                args=(lock_file, cycles_per_process, violation_counter, completed_counter),
            )
            for _ in range(num_processes)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=25.0)
            assert not p.is_alive()

        assert violation_counter.value == 0
        assert completed_counter.value == num_processes

    def test_abandoned_mutex_recovery_on_process_crash(self, tmp_path: Path) -> None:
        """Verify that when a process holding MT5StrategyTesterGlobalLock crashes / exits abruptly,
        the Windows kernel marks the mutex as abandoned and a subsequent process cleanly acquires it
        via WAIT_ABANDONED (0x00000080).
        """
        if os.name != "nt":
            pytest.skip("Windows-specific named mutex test")

        lock_file = str(tmp_path / "abandoned.lock")

        # Code for a crashing process that acquires the lock and immediately calls os._exit(42)
        crash_code = f"""
import sys, os
from pathlib import Path
ROOT = Path(r'{ROOT}')
sys.path.insert(0, str(ROOT / 'src'))
from self_growing_trader.mt5_runner import MT5StrategyTesterGlobalLock

lock = MT5StrategyTesterGlobalLock(r'{lock_file}')
lock.__enter__()
# Force abrupt exit without clean __exit__ release
os._exit(42)
"""
        proc = subprocess.run([sys.executable, "-c", crash_code], capture_output=True)
        assert proc.returncode == 42

        # Now acquire the lock in this process.
        # It must handle WAIT_ABANDONED (0x00000080) without raising RuntimeError!
        with MT5StrategyTesterGlobalLock(lock_file) as lock:
            assert lock.path.is_file()
            # Successfully acquired despite prior holder's abrupt crash!


# ==============================================================================
# 3. ADVERSARIAL TIMEOUT SCALING PROBES
# ==============================================================================

class TestAdversarialTimeoutScaling:
    """Stress tests timeout scaling logic when --mt5-visual is passed vs omitted."""

    @pytest.fixture
    def mock_environment(self, tmp_path: Path):
        candidate = tmp_path / "candidate.pkl"
        candidate.write_bytes(b"dummy")
        runner_dir = tmp_path / "runner"
        runner_dir.mkdir()
        out_json = tmp_path / "summary.json"
        config_path = ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml"
        return {
            "candidate": candidate,
            "runner_dir": runner_dir,
            "out_json": out_json,
            "config": config_path,
            "tmp_path": tmp_path,
        }

    def test_timeout_defaults_visual_vs_headless(self, mock_environment: dict[str, Any], monkeypatch) -> None:
        """Visual mode defaults to 600s; headless mode defaults to 300s."""
        captured_timeouts: list[int] = []

        def mock_eval(**kwargs: Any) -> dict[str, Any]:
            captured_timeouts.append(kwargs["timeout_sec"])
            return {
                "window": kwargs["index"],
                "completed": True,
                "metrics": {},
                "runner_clone": str(mock_environment["tmp_path"] / f"clone_{kwargs['index']}"),
            }

        monkeypatch.setattr(runner, "evaluate_window", mock_eval)

        # 1. Headless (visual omitted)
        summary_headless = runner.run_temporal_generalization(
            candidate=mock_environment["candidate"],
            config=mock_environment["config"],
            runner_source=mock_environment["runner_dir"],
            windows="2025.02.20:2025.02.24",
            output=mock_environment["out_json"],
            mt5_visual=False,
            timeout_sec=None,
        )
        assert summary_headless["timeout_sec"] == 300
        assert captured_timeouts[-1] == 300

        # 2. Visual (visual passed)
        summary_visual = runner.run_temporal_generalization(
            candidate=mock_environment["candidate"],
            config=mock_environment["config"],
            runner_source=mock_environment["runner_dir"],
            windows="2025.02.20:2025.02.24",
            output=mock_environment["out_json"],
            mt5_visual=True,
            timeout_sec=None,
        )
        assert summary_visual["timeout_sec"] == 600
        assert captured_timeouts[-1] == 600

    @pytest.mark.parametrize("custom_timeout", [120, 450, 900, 1800])
    def test_custom_timeout_overrides_defaults_in_both_modes(
        self, mock_environment: dict[str, Any], custom_timeout: int, monkeypatch
    ) -> None:
        """Custom --timeout-sec value must strictly override defaults regardless of visual mode."""
        captured_timeouts: list[int] = []

        def mock_eval(**kwargs: Any) -> dict[str, Any]:
            captured_timeouts.append(kwargs["timeout_sec"])
            return {
                "window": kwargs["index"],
                "completed": True,
                "metrics": {},
                "runner_clone": str(mock_environment["tmp_path"] / f"clone_{kwargs['index']}"),
            }

        monkeypatch.setattr(runner, "evaluate_window", mock_eval)

        # Custom in visual mode
        summary_vis = runner.run_temporal_generalization(
            candidate=mock_environment["candidate"],
            config=mock_environment["config"],
            runner_source=mock_environment["runner_dir"],
            windows="2025.02.20:2025.02.24",
            output=mock_environment["out_json"],
            mt5_visual=True,
            timeout_sec=custom_timeout,
        )
        assert summary_vis["timeout_sec"] == custom_timeout
        assert captured_timeouts[-1] == custom_timeout

        # Custom in headless mode
        summary_head = runner.run_temporal_generalization(
            candidate=mock_environment["candidate"],
            config=mock_environment["config"],
            runner_source=mock_environment["runner_dir"],
            windows="2025.02.20:2025.02.24",
            output=mock_environment["out_json"],
            mt5_visual=False,
            timeout_sec=custom_timeout,
        )
        assert summary_head["timeout_sec"] == custom_timeout
        assert captured_timeouts[-1] == custom_timeout


# ==============================================================================
# 4. STRICT TESTER.INI CONTENT & FORMATTING PROBES
# ==============================================================================

class TestAdversarialTesterIniStrictFormat:
    """Verifies that tester.ini strictly and unambiguously satisfies MT5 formatting rules."""

    def test_visual_mode_ini_strictly_contains_visual_1_and_shutdown_1(self, tmp_path: Path) -> None:
        """In visual mode, tester.ini must contain Visual=1 and ShutdownTerminal=1 and NOT Visual=0."""
        ini_file = tmp_path / "tester_vis.ini"
        config = MT5BacktestConfig(
            visual=True,
            shutdown_terminal=True,
            from_date="2025.02.20",
            to_date="2025.02.24",
        )
        write_tester_ini(config, "report_visual", ini_file)

        raw_text = ini_file.read_text(encoding="utf-8")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # Strict line-by-line checks
        assert "Visual=1" in lines, "tester.ini must contain exact line 'Visual=1'"
        assert "ShutdownTerminal=1" in lines, "tester.ini must contain exact line 'ShutdownTerminal=1'"
        assert "Visual=0" not in lines, "tester.ini must NOT contain 'Visual=0' when visual=True"
        assert "ShutdownTerminal=0" not in lines, "tester.ini must NOT contain 'ShutdownTerminal=0' when shutdown=True"

        # INI parser validation
        parser = configparser.ConfigParser()
        parser.read_string(raw_text)

        assert "Tester" in parser.sections()
        assert parser.get("Tester", "Visual") == "1"
        assert parser.get("Tester", "ShutdownTerminal") == "1"

    def test_headless_mode_ini_strictly_contains_visual_0_and_shutdown_1(self, tmp_path: Path) -> None:
        """In non-visual mode, tester.ini must contain Visual=0 and ShutdownTerminal=1 and NOT Visual=1."""
        ini_file = tmp_path / "tester_head.ini"
        config = MT5BacktestConfig(
            visual=False,
            shutdown_terminal=True,
            from_date="2025.02.20",
            to_date="2025.02.24",
        )
        write_tester_ini(config, "report_headless", ini_file)

        raw_text = ini_file.read_text(encoding="utf-8")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        assert "Visual=0" in lines, "tester.ini must contain exact line 'Visual=0' in headless mode"
        assert "ShutdownTerminal=1" in lines
        assert "Visual=1" not in lines, "tester.ini must NOT contain 'Visual=1' in headless mode"

        parser = configparser.ConfigParser()
        parser.read_string(raw_text)

        assert parser.get("Tester", "Visual") == "0"
        assert parser.get("Tester", "ShutdownTerminal") == "1"

    def test_shutdown_terminal_toggle_invariant(self, tmp_path: Path) -> None:
        """Verify shutdown_terminal=False writes ShutdownTerminal=0 cleanly in both visual and headless modes."""
        # Visual + Shutdown False
        ini_vis_no_shut = tmp_path / "vis_no_shut.ini"
        write_tester_ini(MT5BacktestConfig(visual=True, shutdown_terminal=False), "rep", ini_vis_no_shut)
        parser_vis = configparser.ConfigParser()
        parser_vis.read_string(ini_vis_no_shut.read_text(encoding="utf-8"))
        assert parser_vis.get("Tester", "Visual") == "1"
        assert parser_vis.get("Tester", "ShutdownTerminal") == "0"

        # Headless + Shutdown False
        ini_head_no_shut = tmp_path / "head_no_shut.ini"
        write_tester_ini(MT5BacktestConfig(visual=False, shutdown_terminal=False), "rep", ini_head_no_shut)
        parser_head = configparser.ConfigParser()
        parser_head.read_string(ini_head_no_shut.read_text(encoding="utf-8"))
        assert parser_head.get("Tester", "Visual") == "0"
        assert parser_head.get("Tester", "ShutdownTerminal") == "0"
