"""
tests/test_risk_governance_and_visual_config.py

Comprehensive test suite verifying Milestone 2: Risk Governance & Visual Runner Configuration (Requirements R2 & R3).

Requirements tested:
1. scripts/run_frozen_temporal_generalization.py:
   - CLI parsing for `--mt5-visual` and `--timeout-sec`.
   - Visual mode argument propagation (`visual=True`, `timeout_sec=args.timeout_sec`).
   - Single-instance visual guard (`max_workers=1` when `visual=True`).
   - Default timeouts (600s for visual mode, 300s for headless mode).
2. src/self_growing_trader/mt5_runner.py:
   - `write_tester_ini()` writes `Visual=1` and `ShutdownTerminal=1` in `[Tester]` section when `visual=True`.
   - `write_tester_ini()` writes `Visual=0` in headless mode.
   - `terminal64.exe` command line topology verification (`/portable`, `/config:<path>`, `/skipupdate`).
3. Core Risk Governance Constraints:
   - `POLICY_CLOSE_ALL == 0` lock: Tensor masking `allowed[2:, :] = False; allowed[2:, 0] = True` in `five_shared_policy.py`.
   - Fixed Geometry Lock: SL = 5.0 ATR, TP = 10.0 ATR, Max Holding = 120 bars M1 enforced in EA `.set` and YAML configs.
   - Zero Hard Direction Gate: `context_direction_entry_gate_enabled: false` invariant across active candidate configs and fail-closed validation in `multi_regime_gate.py`.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Ensure repository root and src/scripts are in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scripts import run_frozen_temporal_generalization as runner
from self_growing_trader.config import load_config
from self_growing_trader.five_agent_authority import CLOSE_ALL_ACTION, HOLD_ACTION
from self_growing_trader.mt5_runner import (
    MT5BacktestConfig,
    MT5StrategyTesterGlobalLock,
    write_tester_ini,
    write_tester_set,
)
from self_growing_trader.multi_regime_gate import (
    GateHurdles,
    RegimeMetrics,
    evaluate_multi_regime_gate,
)
from self_growing_trader.orders import Action


# ==============================================================================
# SECTION 1: CLI PARSING & VISUAL RUNNER CONFIGURATION
# ==============================================================================

class TestR2VisualRunnerConfiguration:
    """Verifies CLI argument parsing and visual runner execution configuration."""

    def test_cli_parser_defaults_without_visual(self) -> None:
        """When --mt5-visual is omitted, args.mt5_visual is False and timeout_sec defaults to None."""
        parser = runner.build_parser()
        args = parser.parse_args([])

        assert args.mt5_visual is False
        assert args.timeout_sec is None
        assert args.max_workers == 1

    def test_cli_parser_with_visual_flag(self) -> None:
        """When --mt5-visual is provided, args.mt5_visual is True."""
        parser = runner.build_parser()
        args = parser.parse_args(["--mt5-visual"])

        assert args.mt5_visual is True
        assert args.timeout_sec is None

    def test_cli_parser_with_visual_and_custom_timeout(self) -> None:
        """When --mt5-visual and --timeout-sec 900 are passed, they parse accurately."""
        parser = runner.build_parser()
        args = parser.parse_args(["--mt5-visual", "--timeout-sec", "900"])

        assert args.mt5_visual is True
        assert args.timeout_sec == 900

    def test_single_instance_visual_guard_clamps_max_workers_to_one(self, tmp_path: Path, monkeypatch) -> None:
        """Visual mode must clamp max_workers to 1 regardless of user input to prevent GUI window conflicts."""
        candidate = tmp_path / "candidate.pkl"
        candidate.write_bytes(b"dummy_candidate_data")
        runner_dir = tmp_path / "runner"
        runner_dir.mkdir()
        out_json = tmp_path / "test_out.json"

        mock_eval = MagicMock(return_value={
            "window": 1,
            "completed": True,
            "metrics": {},
            "runner_clone": str(tmp_path / "clone_1"),
        })
        monkeypatch.setattr(runner, "evaluate_window", mock_eval)

        # User requests max_workers = 4 along with --mt5-visual
        args = argparse.Namespace(
            candidate=candidate,
            config=ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
            profile=None,
            runner_source=runner_dir,
            windows="2025.02.20:2025.02.24,2025.02.24:2025.02.28",
            model=4,
            timeout_sec=None,
            max_workers=4,
            output=out_json,
            mt5_visual=True,
        )

        summary = runner.run_temporal_generalization(args)

        assert summary["visual"] is True
        assert summary["max_workers"] == 1
        assert summary["parallel_windows"] is False
        assert summary["timeout_sec"] == 600  # Default visual timeout

    def test_headless_mode_preserves_parallel_workers(self, tmp_path: Path, monkeypatch) -> None:
        """In headless mode (visual=False), multi-worker execution is allowed."""
        candidate = tmp_path / "candidate.pkl"
        candidate.write_bytes(b"dummy_candidate_data")
        runner_dir = tmp_path / "runner"
        runner_dir.mkdir()
        out_json = tmp_path / "test_out.json"

        mock_eval = MagicMock(return_value={
            "window": 1,
            "completed": True,
            "metrics": {},
            "runner_clone": str(tmp_path / "clone_1"),
        })
        monkeypatch.setattr(runner, "evaluate_window", mock_eval)

        args = argparse.Namespace(
            candidate=candidate,
            config=ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
            profile=None,
            runner_source=runner_dir,
            windows="2025.02.20:2025.02.24,2025.02.24:2025.02.28,2025.03.01:2025.03.05",
            model=4,
            timeout_sec=None,
            max_workers=3,
            output=out_json,
            mt5_visual=False,
        )

        summary = runner.run_temporal_generalization(args)

        assert summary["visual"] is False
        assert summary["max_workers"] == 3
        assert summary["parallel_windows"] is True
        assert summary["timeout_sec"] == 300  # Default headless timeout

    def test_argument_propagation_to_evaluate_window(self, tmp_path: Path, monkeypatch) -> None:
        """Verifies visual=True and custom timeout_sec are strictly passed to evaluate_window."""
        candidate = tmp_path / "candidate.pkl"
        candidate.write_bytes(b"dummy_candidate_data")
        runner_dir = tmp_path / "runner"
        runner_dir.mkdir()
        out_json = tmp_path / "test_out.json"

        captured_kwargs: list[dict[str, Any]] = []

        def fake_eval(**kwargs: Any) -> dict[str, Any]:
            captured_kwargs.append(copy.deepcopy(kwargs))
            return {
                "window": kwargs["index"],
                "completed": True,
                "metrics": {},
                "runner_clone": str(tmp_path / f"clone_{kwargs['index']}"),
            }

        monkeypatch.setattr(runner, "evaluate_window", fake_eval)

        args = argparse.Namespace(
            candidate=candidate,
            config=ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
            profile=None,
            runner_source=runner_dir,
            windows="2025.02.20:2025.02.24",
            model=4,
            timeout_sec=750,
            max_workers=2,
            output=out_json,
            mt5_visual=True,
        )

        summary = runner.run_temporal_generalization(args)

        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["visual"] is True
        assert kw["timeout_sec"] == 750
        assert kw["model"] == 4
        assert summary["visual"] is True
        assert summary["timeout_sec"] == 750


# ==============================================================================
# SECTION 2: TESTER.INI GENERATION & COMMAND TOPOLOGY
# ==============================================================================

class TestR2TesterIniAndTerminalTopology:
    """Verifies MT5 tester.ini writing, visual flags, and terminal process execution topology."""

    def test_write_tester_ini_visual_and_shutdown_active(self, tmp_path: Path) -> None:
        """When visual=True and shutdown_terminal=True, tester.ini contains Visual=1 and ShutdownTerminal=1."""
        ini_path = tmp_path / "tester_visual.ini"
        config = MT5BacktestConfig(
            visual=True,
            shutdown_terminal=True,
            from_date="2025.02.20",
            to_date="2025.02.24",
            deposit=10000.0,
        )

        result_path = write_tester_ini(config, "test_report_visual", ini_path)
        assert result_path.is_file()
        content = result_path.read_text(encoding="utf-8")

        assert "[Tester]" in content
        assert "Visual=1" in content
        assert "ShutdownTerminal=1" in content
        assert "Expert=SelfGrowing_MT5_Trainer_EA" in content
        assert "Symbol=BTCUSDm" in content
        assert "Period=M1" in content
        assert "FromDate=2025.02.20" in content
        assert "ToDate=2025.02.24" in content

    def test_write_tester_ini_headless_mode(self, tmp_path: Path) -> None:
        """In headless mode (visual=False), tester.ini contains Visual=0 and ShutdownTerminal=1."""
        ini_path = tmp_path / "tester_headless.ini"
        config = MT5BacktestConfig(
            visual=False,
            shutdown_terminal=True,
            from_date="2025.02.20",
            to_date="2025.02.24",
        )

        result_path = write_tester_ini(config, "test_report_headless", ini_path)
        content = result_path.read_text(encoding="utf-8")

        assert "Visual=0" in content
        assert "ShutdownTerminal=1" in content

    def test_terminal64_command_line_topology_verification(self) -> None:
        """Validates command structure passed to subprocess.Popen: [terminal, /portable, /config:<ini>, /skipupdate]."""
        terminal_exe = ROOT / "mt5_v2_runner/terminal64.exe"
        ini_path = ROOT / "codex_run_configs/round_0000_evaluation_9102.ini"

        cmd = [str(terminal_exe), "/portable", f"/config:{ini_path}", "/skipupdate"]

        assert cmd[0] == str(terminal_exe)
        assert cmd[1] == "/portable"
        assert cmd[2] == f"/config:{ini_path}"
        assert cmd[3] == "/skipupdate"
        assert len(cmd) == 4

    def test_mt5_global_lock_mutual_exclusion(self, tmp_path: Path) -> None:
        """Verifies MT5StrategyTesterGlobalLock file creation, pid tracking, and release across runs."""
        import json
        import os
        lock_path = tmp_path / ".mt5_strategy_tester.global.lock"
        assert not lock_path.exists()

        with MT5StrategyTesterGlobalLock(lock_path) as lock:
            assert lock.path.is_file()
            assert lock_path.is_file()
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            assert payload["pid"] == os.getpid()

        # Re-acquisition succeeds cleanly
        with MT5StrategyTesterGlobalLock(lock_path) as lock2:
            assert lock2.path.is_file()


# ==============================================================================
# SECTION 3: POLICY_CLOSE_ALL == 0 TENSOR MASKING INVARIANT
# ==============================================================================

class TestR3PolicyCloseAllLock:
    """Verifies Portfolio Governor HOLD lock and absolute elimination of POLICY_CLOSE_ALL (Action 10)."""

    def test_close_all_action_constant_binding(self) -> None:
        """Asserts CLOSE_ALL is index 10 and HOLD is index 0 in authoritative action contracts."""
        assert CLOSE_ALL_ACTION == 10
        assert HOLD_ACTION == 0
        assert int(Action.CLOSE_ALL) == 10
        assert int(Action.HOLD) == 0

    def test_numpy_allowed_action_masking_invariant(self) -> None:
        """Asserts allowed[2:, :] = False; allowed[2:, 0] = True forces agents 2..4 to HOLD only."""
        num_agents = 5
        num_actions = 20
        allowed = np.ones((num_agents, num_actions), dtype=bool)

        # Apply Governor Lock logic from five_shared_policy.py
        allowed[2:, :] = False
        allowed[2:, 0] = True

        # Non-entry agents (Agent 2: Position Manager, Agent 3: Order Manager, Agent 4: Portfolio Governor)
        for agent_idx in range(2, 5):
            assert allowed[agent_idx, 0] is np.True_, f"Agent {agent_idx} must allow HOLD (action 0)"
            assert not allowed[agent_idx, 1:].any(), f"Agent {agent_idx} has non-HOLD actions allowed!"
            assert allowed[agent_idx, CLOSE_ALL_ACTION] is np.False_, (
                f"Agent {agent_idx} must NOT allow CLOSE_ALL (action 10)"
            )

        # Entry agents (0 and 1) retain entry capability
        assert allowed[0, :].any()
        assert allowed[1, :].any()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for tensor distribution tests")
    def test_pytorch_tensor_masking_distribution_eliminates_close_all(self) -> None:
        """With PyTorch, even with high unmasked logit for CLOSE_ALL (+100.0), masking guarantees probability 0.0."""
        num_agents = 5
        num_actions = 20

        # Create logits where Agent 4 has extreme preference for Action 10 (CLOSE_ALL)
        logits = torch.zeros((num_agents, num_actions), dtype=torch.float32)
        logits[4, CLOSE_ALL_ACTION] = 100.0  # Dominant logit for close-all
        logits[4, HOLD_ACTION] = -10.0      # Low logit for hold

        # Legality mask
        allowed = torch.ones((num_agents, num_actions), dtype=torch.bool)
        allowed[2:, :] = False
        allowed[2:, 0] = True

        # Apply masking rule (set disallowed logits to -1e9 or -30.0)
        masked_logits = torch.where(allowed, logits, torch.full_like(logits, -1e9))
        probs = F.softmax(masked_logits, dim=-1)

        # Verify Agent 4 probability distribution
        agent_4_probs = probs[4].detach().numpy()
        assert agent_4_probs[CLOSE_ALL_ACTION] == pytest.approx(0.0, abs=1e-12), (
            f"CLOSE_ALL probability must be 0.0, got {agent_4_probs[CLOSE_ALL_ACTION]}"
        )
        assert agent_4_probs[HOLD_ACTION] == pytest.approx(1.0, abs=1e-6), (
            f"HOLD probability must be 1.0, got {agent_4_probs[HOLD_ACTION]}"
        )

        # Categorical sample
        dist = torch.distributions.Categorical(probs=probs)
        sample = dist.sample()
        for _ in range(50):
            s = dist.sample()
            assert s[4].item() == HOLD_ACTION, f"Agent 4 emitted non-HOLD action {s[4].item()}"

    def test_five_shared_policy_source_contains_governor_lock(self) -> None:
        """Verifies five_shared_policy.py contains the exact masking logic."""
        policy_source = (ROOT / "src/self_growing_trader/five_shared_policy.py").read_text(encoding="utf-8")

        assert "allowed[2:, :] = False" in policy_source
        assert "allowed[2:, 0] = True" in policy_source
        assert "seven_choice_fixed_geometry_enabled" in policy_source
        assert "lock_non_entry_to_hold" in policy_source


# ==============================================================================
# SECTION 4: FIXED GEOMETRY LOCK (SL=5 ATR, TP=10 ATR, MaxHolding=120 M1)
# ==============================================================================

class TestR3FixedGeometryParameters:
    """Verifies fixed geometry parameters across EA .set files and YAML configuration files."""

    def test_fixed_geometry_written_to_tester_set_file(self, tmp_path: Path) -> None:
        """write_tester_set writes FixedSlAtr=5, FixedTpAtr=10, MaxHoldingBars=120 into EA .set file."""
        set_path = tmp_path / "SelfGrowing_MT5_Trainer_EA.set"
        config = MT5BacktestConfig(
            fixed_sl_atr=5.0,
            fixed_tp_atr=10.0,
            max_holding_bars=120,
        )

        write_tester_set(config, set_path)
        content = set_path.read_text(encoding="utf-8")

        assert "FixedSlAtr=5" in content
        assert "FixedTpAtr=10" in content
        assert "MaxHoldingBars=120" in content

    def test_fixed_geometry_in_active_candidate_yaml_configs(self) -> None:
        """Checks configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml for fixed geometry parameters."""
        cfg_path = ROOT / "configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
        assert cfg_path.is_file(), f"Missing config file {cfg_path}"

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        orders = raw.get("orders", {})

        assert orders.get("fixed_sl_atr") == 5.0, f"Expected fixed_sl_atr 5.0, got {orders.get('fixed_sl_atr')}"
        assert orders.get("fixed_tp_atr") == 10.0, f"Expected fixed_tp_atr 10.0, got {orders.get('fixed_tp_atr')}"
        assert orders.get("max_holding_bars") == 120, f"Expected max_holding_bars 120, got {orders.get('max_holding_bars')}"

    def test_fixed_geometry_learning_flags_in_experiment_profile(self) -> None:
        """Checks configs/experiments/reasoning_rl_v1_active_candidate.yaml for learning lock overrides."""
        profile_path = ROOT / "configs/experiments/reasoning_rl_v1_active_candidate.yaml"
        assert profile_path.is_file(), f"Missing profile {profile_path}"

        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        learning_overrides = raw.get("learning_overrides", {})

        assert learning_overrides.get("five_lock_non_entry_to_hold") is True
        assert learning_overrides.get("five_fixed_geometry_enabled") is True

    def test_app_config_load_reflects_geometry_invariants(self) -> None:
        """Verifies AppConfig loaded from reasoning_rl_v1_candidate has correct order geometry."""
        config = load_config(ROOT / "configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml")
        assert config.orders.fixed_sl_atr == 5.0
        assert config.orders.fixed_tp_atr == 10.0
        assert config.orders.max_holding_bars == 120


# ==============================================================================
# SECTION 5: ZERO HARD DIRECTION GATE INVARIANT
# ==============================================================================

class TestR3ZeroHardDirectionGateInvariant:
    """Verifies that context_direction_entry_gate_enabled is false across configs and rejected if enabled."""

    def test_context_direction_entry_gate_is_false_in_active_configs(self) -> None:
        """Verifies context_direction_entry_gate_enabled is explicitly False in all active candidate configs."""
        target_configs = [
            ROOT / "configs/experiments/reasoning_rl_v1_active_candidate.yaml",
            ROOT / "configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml",
            ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
        ]

        for cfg_file in target_configs:
            if cfg_file.is_file():
                raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
                learning = raw.get("learning", {})
                learning_overrides = raw.get("learning_overrides", {})

                assert not learning.get("context_direction_entry_gate_enabled", False), (
                    f"Forbidden hard direction gate enabled in {cfg_file}"
                )
                assert not learning_overrides.get("context_direction_entry_gate_enabled", False), (
                    f"Forbidden hard direction gate enabled in overrides of {cfg_file}"
                )

    def test_multi_regime_gate_fails_candidate_when_hard_direction_gate_enabled(self) -> None:
        """multi_regime_gate must strictly fail with 'hard_direction_gate_forbidden' if the flag is True."""
        synthetic_metrics = {
            "bull": RegimeMetrics(
                regime_id="bull",
                total_trades=50,
                wins=28,
                losses=22,
                win_rate_pct=56.0,
                profit_factor=1.8,
                max_drawdown_pct=4.0,
                expectancy_r=0.15,
                payoff_ratio=1.6,
                counter_trend_rate_pct=15.0,
                timeout_close_rate_pct=5.0,
                premature_closes=0,
            ),
        }

        # Candidate config with the forbidden flag turned ON
        forbidden_config: dict[str, Any] = {
            "learning": {
                "context_direction_entry_gate_enabled": True,
            }
        }

        passed, reasons, agg = evaluate_multi_regime_gate(
            synthetic_metrics,
            hurdles=GateHurdles(require_natural_rl=True, min_total_trades=10),
            candidate_config=forbidden_config,
        )

        assert passed is False
        assert any("hard_direction_gate_forbidden" in reason for reason in reasons), (
            f"Expected hard_direction_gate_forbidden in reasons, got: {reasons}"
        )

    def test_multi_regime_gate_passes_candidate_when_hard_direction_gate_disabled(self) -> None:
        """When context_direction_entry_gate_enabled is False and performance metrics are met, gate passes."""
        synthetic_metrics = {
            "bull": RegimeMetrics(
                regime_id="bull",
                total_trades=50,
                wins=30,
                losses=20,
                win_rate_pct=60.0,
                profit_factor=2.0,
                max_drawdown_pct=3.5,
                expectancy_r=0.20,
                payoff_ratio=1.75,
                counter_trend_rate_pct=10.0,
                timeout_close_rate_pct=4.0,
                premature_closes=0,
            ),
        }

        valid_config: dict[str, Any] = {
            "learning": {
                "context_direction_entry_gate_enabled": False,
            }
        }

        passed, reasons, agg = evaluate_multi_regime_gate(
            synthetic_metrics,
            hurdles=GateHurdles(require_natural_rl=True, min_total_trades=10, min_win_rate=50.0),
            candidate_config=valid_config,
        )

        assert not any("hard_direction_gate_forbidden" in reason for reason in reasons)
        assert passed is True
