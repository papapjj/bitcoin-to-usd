"""Comprehensive Unit & Integration Test Suite for Auto-RL Engine (Milestone 3).

Verifies AutoRLEngine, AutoRLEngineConfig, TimeframeAdaptationStateMachine,
CumulativeDrawdownMonitor, AutoRLManifestManager, atomic checkpoint persistence,
automated rollback, disk pruning, OOS gate evaluation, and multi-round supervisor loops.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping
import unittest

import pytest

from src.self_growing_trader.auto_rl_engine import (
    AutoRLEngine,
    AutoRLEngineConfig,
    AutoRLManifestManager,
    CumulativeDrawdownMonitor,
    OOSValidationResult,
    TimeframeAdaptationStateMachine,
    atomic_save_checkpoint,
    evaluate_candidate_oos_gate,
    pre_execution_backup_checkpoint,
    prune_unpromoted_checkpoints,
    rollback_checkpoint,
)


# ==============================================================================
# 1. AUTORL ENGINE CONFIG TESTS
# ==============================================================================


class TestAutoRLEngineConfig:
    """Tests for AutoRLEngineConfig parameters and integrity constraints."""

    def test_default_config_parameters(self) -> None:
        """Verify default configuration values match PROJECT.md specifications."""
        cfg = AutoRLEngineConfig()
        assert cfg.max_rounds == 15
        assert cfg.initial_timeframe == "M1"
        assert cfg.max_cumulative_drawdown == 0.15
        assert cfg.m1_failure_threshold == 3
        assert cfg.m5_failure_threshold == 3
        assert cfg.initial_equity == 10000.0
        assert cfg.fixed_sl_atr == 5.0
        assert cfg.fixed_tp_atr == 10.0
        assert cfg.oos_from_date == "2025.01.01"
        assert cfg.oos_to_date == "2025.03.31"
        assert cfg.min_oos_expectancy == 0.05
        assert cfg.min_oos_win_rate_pct == 40.0
        assert cfg.min_oos_risk_reward == 2.0
        assert cfg.min_oos_trades == 30
        assert cfg.max_oos_drawdown_pct == 10.0
        assert cfg.max_hold_share == 0.80
        assert cfg.max_direction_imbalance == 0.60
        assert cfg.min_entropy == 0.35
        assert cfg.five_codex_teacher_enabled is False
        assert cfg.visual is False
        assert cfg.shutdown_terminal is True
        assert cfg.prune_non_promoted_checkpoints is True

    def test_enforce_zero_codex_teacher_native_rl(self) -> None:
        """Config rejects five_codex_teacher_enabled=True to guarantee 100% native RL."""
        with pytest.raises(ValueError, match="strictly enforces five_codex_teacher_enabled=False"):
            AutoRLEngineConfig(five_codex_teacher_enabled=True)

    def test_enforce_max_15_rounds_budget_cap(self) -> None:
        """Supervisor budget cap automatically restricts max_rounds to at most 15."""
        cfg = AutoRLEngineConfig(max_rounds=25)
        assert cfg.max_rounds == 15

    def test_invalid_timeframe_rejected(self) -> None:
        """Config rejects unsupported timeframe strings."""
        with pytest.raises(ValueError, match="Invalid initial_timeframe"):
            AutoRLEngineConfig(initial_timeframe="D1")

        with pytest.raises(ValueError, match="Invalid initial_timeframe"):
            AutoRLEngineConfig(initial_timeframe="M2")

    def test_custom_config_values(self, tmp_path: Path) -> None:
        """Custom configuration parameters are accurately preserved."""
        cfg = AutoRLEngineConfig(
            runs_dir=tmp_path / "custom_runs",
            max_rounds=10,
            initial_timeframe="M5",
            max_cumulative_drawdown=0.12,
            initial_equity=50000.0,
            seed_checkpoint=tmp_path / "custom_seed.pkl",
        )
        assert cfg.runs_dir == tmp_path / "custom_runs"
        assert cfg.max_rounds == 10
        assert cfg.initial_timeframe == "M5"
        assert cfg.max_cumulative_drawdown == 0.12
        assert cfg.initial_equity == 50000.0
        assert cfg.seed_checkpoint == tmp_path / "custom_seed.pkl"


# ==============================================================================
# 2. TIMEFRAME ADAPTATION STATE MACHINE TESTS
# ==============================================================================


class TestTimeframeAdaptationStateMachine:
    """Tests for automated timeframe adaptation and friction scaling."""

    def test_initial_state(self) -> None:
        """Initial state starts at M1 with 0 consecutive losses and empty history."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1")
        assert sm.current_timeframe == "M1"
        assert sm.consecutive_m1_losses == 0
        assert sm.consecutive_m5_losses == 0
        assert len(sm.migration_history) == 0

    def test_single_and_double_loss_remains_m1(self) -> None:
        """Rounds 1 and 2 failing remain on M1 with incremented loss counter."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1")
        tf1 = sm.record_round_result(round_num=1, expectancy=-0.05, trades=20)
        assert tf1 == "M1"
        assert sm.consecutive_m1_losses == 1

        tf2 = sm.record_round_result(round_num=2, expectancy=-0.02, trades=18)
        assert tf2 == "M1"
        assert sm.consecutive_m1_losses == 2
        assert len(sm.migration_history) == 0

    def test_three_consecutive_losses_triggers_m5(self) -> None:
        """Third consecutive non-improving round triggers automated migration to M5."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1")
        sm.record_round_result(round_num=1, expectancy=-0.05, trades=20)
        sm.record_round_result(round_num=2, expectancy=-0.02, trades=18)
        tf3 = sm.record_round_result(round_num=3, expectancy=0.0, trades=15)

        assert tf3 == "M5"
        assert sm.current_timeframe == "M5"
        assert sm.consecutive_m1_losses == 0  # Reset upon migration
        assert len(sm.migration_history) == 1
        assert sm.migration_history[0]["from"] == "M1"
        assert sm.migration_history[0]["to"] == "M5"

    def test_improving_round_resets_loss_counter(self) -> None:
        """An improving round (E[R] > 0 and trades > 0) resets consecutive loss counter."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1")
        sm.record_round_result(round_num=1, expectancy=-0.05, trades=20)
        sm.record_round_result(round_num=2, expectancy=-0.02, trades=18)
        assert sm.consecutive_m1_losses == 2

        # Improving round
        tf3 = sm.record_round_result(round_num=3, expectancy=0.03, trades=32)
        assert tf3 == "M1"
        assert sm.consecutive_m1_losses == 0

    def test_zero_trades_counted_as_loss(self) -> None:
        """Zero trades (paralyzed policy) is treated as a loss even if expectancy is reported 0."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1")
        tf = sm.record_round_result(round_num=1, expectancy=0.0, trades=0)
        assert tf == "M1"
        assert sm.consecutive_m1_losses == 1

    def test_three_consecutive_m5_losses_triggers_m15(self) -> None:
        """3 non-improving rounds on M5 triggers migration to M15."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M5")
        sm.record_round_result(round_num=1, expectancy=-0.03, trades=15)
        sm.record_round_result(round_num=2, expectancy=-0.01, trades=12)
        tf3 = sm.record_round_result(round_num=3, expectancy=-0.02, trades=10)

        assert tf3 == "M15"
        assert sm.current_timeframe == "M15"
        assert sm.consecutive_m5_losses == 0
        assert len(sm.migration_history) == 1
        assert sm.migration_history[0]["from"] == "M5"
        assert sm.migration_history[0]["to"] == "M15"

    def test_custom_failure_thresholds(self) -> None:
        """State machine respects custom failure thresholds (e.g. 2)."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1", m1_failure_threshold=2)
        sm.record_round_result(round_num=1, expectancy=-0.05, trades=20)
        tf2 = sm.record_round_result(round_num=2, expectancy=-0.01, trades=18)
        assert tf2 == "M5"
        assert len(sm.migration_history) == 1

    def test_get_timeframe_parameters_m1(self) -> None:
        """M1 geometry enforces SL=5.0, TP=10.0, ratio <= 13%, and 60s bar cadence."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1")
        params = sm.get_timeframe_parameters()
        assert params["timeframe"] == "M1"
        assert params["fixed_sl_atr"] == 5.0
        assert params["fixed_tp_atr"] == 10.0
        assert params["bar_duration_sec"] == 60
        assert params["spread_sl_ratio_limit"] == 0.13

    def test_get_timeframe_parameters_m5_and_m15(self) -> None:
        """M5 and M15 scale bar duration and friction limits accordingly."""
        sm_m5 = TimeframeAdaptationStateMachine(initial_timeframe="M5")
        p5 = sm_m5.get_timeframe_parameters()
        assert p5["bar_duration_sec"] == 300
        assert p5["spread_sl_ratio_limit"] == 0.06

        sm_m15 = TimeframeAdaptationStateMachine(initial_timeframe="M15")
        p15 = sm_m15.get_timeframe_parameters()
        assert p15["bar_duration_sec"] == 900
        assert p15["spread_sl_ratio_limit"] == 0.035


# ==============================================================================
# 3. CUMULATIVE DRAWDOWN MONITOR TESTS
# ==============================================================================


class TestCumulativeDrawdownMonitor:
    """Tests for account equity tracking and emergency halt logic."""

    def test_initial_state(self) -> None:
        """Initial equity is 10,000, DD is 0%, halt is False."""
        mon = CumulativeDrawdownMonitor(initial_equity=10000.0, halt_pct=15.0)
        assert mon.initial_equity == 10000.0
        assert mon.current_equity == 10000.0
        assert mon.peak_equity == 10000.0
        assert mon.cumulative_drawdown_pct == 0.0
        assert mon.emergency_halt_triggered is False

    def test_profitable_rounds_increase_equity_and_peak(self) -> None:
        """Profits increase current and peak equity; cumulative drawdown remains 0%."""
        mon = CumulativeDrawdownMonitor(initial_equity=10000.0, halt_pct=15.0)
        halt = mon.update(500.0)
        assert halt is False
        assert mon.current_equity == 10500.0
        assert mon.peak_equity == 10500.0
        assert mon.cumulative_drawdown_pct == 0.0

    def test_loss_increases_cumulative_drawdown(self) -> None:
        """Losses relative to initial capital base increase cumulative drawdown percentage."""
        mon = CumulativeDrawdownMonitor(initial_equity=10000.0, halt_pct=15.0)
        halt1 = mon.update(-500.0)  # Equity: 9500 (DD: 5.0%)
        assert halt1 is False
        assert mon.cumulative_drawdown_pct == pytest.approx(5.0)

        halt2 = mon.update(-500.0)  # Equity: 9000 (DD: 10.0%)
        assert halt2 is False
        assert mon.cumulative_drawdown_pct == pytest.approx(10.0)

    def test_emergency_halt_boundary_14_99_continues(self) -> None:
        """Cumulative DD of 14.99% (<= 15.0%) does NOT trigger emergency halt."""
        mon = CumulativeDrawdownMonitor(initial_equity=10000.0, halt_pct=15.0)
        halt = mon.update(-1499.0)
        assert halt is False
        assert mon.cumulative_drawdown_pct == pytest.approx(14.99)
        assert mon.emergency_halt_triggered is False

    def test_emergency_halt_boundary_15_01_halts(self) -> None:
        """Cumulative DD of 15.01% (> 15.0%) immediately triggers emergency halt."""
        mon = CumulativeDrawdownMonitor(initial_equity=10000.0, halt_pct=15.0)
        halt = mon.update(-1501.0)
        assert halt is True
        assert mon.cumulative_drawdown_pct == pytest.approx(15.01)
        assert mon.emergency_halt_triggered is True

    def test_history_logging(self) -> None:
        """Monitor logs history entries for each round update."""
        mon = CumulativeDrawdownMonitor(initial_equity=10000.0, halt_pct=15.0)
        mon.update(-200.0)
        mon.update(300.0)
        assert len(mon.history) == 2
        assert mon.history[0]["round"] == 1
        assert mon.history[0]["pnl"] == -200.0
        assert mon.history[1]["round"] == 2
        assert mon.history[1]["pnl"] == 300.0


# ==============================================================================
# 4. MANIFEST & EVOLUTION CHART PERSISTENCE TESTS
# ==============================================================================


class TestAutoRLManifestManager:
    """Tests for auto_rl_manifest.json and evolution_chart.json persistence."""

    def test_load_nonexistent_manifest_returns_defaults(self, tmp_path: Path) -> None:
        """Non-existent manifest initializes with required default keys."""
        mgr = AutoRLManifestManager(tmp_path)
        data = mgr.load_manifest()
        required_keys = {
            "active_seed",
            "current_round",
            "current_timeframe",
            "consecutive_m1_losses",
            "cumulative_drawdown_pct",
            "rounds",
        }
        assert required_keys.issubset(data.keys())
        assert data["current_round"] == 0
        assert data["current_timeframe"] == "M1"
        assert len(data["rounds"]) == 0

    def test_atomic_manifest_saving_and_reloading(self, tmp_path: Path) -> None:
        """Manifest saves atomically and reloads identically across process restarts."""
        mgr = AutoRLManifestManager(tmp_path)
        payload = {
            "active_seed": str(tmp_path / "best_v1.pkl"),
            "current_round": 2,
            "current_timeframe": "M1",
            "consecutive_m1_losses": 1,
            "cumulative_drawdown_pct": 2.5,
            "rounds": [{"round": 1, "promoted": False}, {"round": 2, "promoted": True}],
        }
        mgr.save_manifest(payload)

        reloaded = mgr.load_manifest()
        assert reloaded == payload

    def test_record_round_appends_to_manifest_and_chart(self, tmp_path: Path) -> None:
        """record_round populates both manifest and evolution chart ledgers."""
        mgr = AutoRLManifestManager(tmp_path)
        metrics = {
            "expectancy": 0.075,
            "win_rate": 43.0,
            "total_trades": 36,
            "max_equity_drawdown_pct": 6.2,
            "ppo_loss": 0.18,
            "entropy": 0.55,
        }
        mgr.record_round(
            round_num=1,
            timeframe="M1",
            metrics=metrics,
            promoted=True,
            active_seed="champion_r1.pkl",
            cumulative_dd_pct=1.2,
        )

        manifest = mgr.load_manifest()
        assert manifest["current_round"] == 1
        assert manifest["active_seed"] == "champion_r1.pkl"
        assert len(manifest["rounds"]) == 1
        assert manifest["rounds"][0]["expectancy"] == 0.075

        chart = mgr.load_evolution_chart()
        assert len(chart) == 1
        entry = chart[0]
        expected_keys = {
            "round",
            "timeframe",
            "loss",
            "entropy",
            "trades",
            "wr",
            "expectancy",
            "promoted",
        }
        assert expected_keys.issubset(entry.keys())
        assert entry["round"] == 1
        assert entry["timeframe"] == "M1"
        assert entry["expectancy"] == 0.075
        assert entry["promoted"] is True


# ==============================================================================
# 5. CHECKPOINT MANAGEMENT & DISK PRUNING TESTS
# ==============================================================================


class TestCheckpointManagement:
    """Tests for atomic saves, pre-execution backups, rollbacks, and disk pruning."""

    def test_atomic_save_checkpoint_bytes(self, tmp_path: Path) -> None:
        """Atomic persistence writes bytes to destination without temporary leftovers."""
        dest = tmp_path / "model.pkl"
        payload = b"SUPERNETWORK_BYTES_12345"
        res = atomic_save_checkpoint(payload, dest)
        assert res == dest
        assert dest.exists()
        assert not (tmp_path / "model.pkl.tmp").exists()
        assert dest.read_bytes() == payload

    def test_atomic_save_checkpoint_with_registry(self, tmp_path: Path) -> None:
        """Atomic save creates matching sidecar .registry.json."""
        dest = tmp_path / "model.pkl"
        reg_data = {"che_do": "nam_tac_nhan", "version": "v193"}
        atomic_save_checkpoint(b"WEIGHTS", dest, registry_data=reg_data)

        assert dest.exists()
        reg_file = tmp_path / "model.registry.json"
        assert reg_file.exists()
        loaded_reg = json.loads(reg_file.read_text(encoding="utf-8"))
        assert loaded_reg["che_do"] == "nam_tac_nhan"

    def test_pre_execution_backup_creates_rollback_copy(self, tmp_path: Path) -> None:
        """pre_execution_backup_checkpoint copies policy and registry with timestamp."""
        policy_file = tmp_path / "seed.pkl"
        reg_file = tmp_path / "seed.registry.json"
        policy_file.write_bytes(b"INITIAL_SEED")
        reg_file.write_text('{"name": "seed"}', encoding="utf-8")

        backups = pre_execution_backup_checkpoint(policy_file)
        assert "policy" in backups
        assert "registry" in backups
        assert Path(backups["policy"]).exists()
        assert Path(backups["registry"]).exists()
        assert ".rollback." in backups["policy"]

    def test_rollback_checkpoint_restores_active_seed(self, tmp_path: Path) -> None:
        """rollback_checkpoint safely reverts active pointer to champion."""
        champion = tmp_path / "champion.pkl"
        failed = tmp_path / "failed.pkl"
        champion.write_bytes(b"CHAMP")
        failed.write_bytes(b"FAILED")

        active = rollback_checkpoint(failed, champion)
        assert active == str(champion)

    def test_prune_unpromoted_checkpoints(self, tmp_path: Path) -> None:
        """Pruning deletes unpromoted candidate .pkl while preserving telemetry JSON manifests."""
        round_dir = tmp_path / "round_01"
        round_dir.mkdir()
        candidate = round_dir / "candidate.pkl"
        manifest = round_dir / "manifest.json"
        events = round_dir / "events.jsonl"
        diagnostics = round_dir / "mt5_diagnostics.json"

        candidate.write_bytes(b"\x00" * 1024 * 1024)  # 1MB
        manifest.write_text('{"status": "rejected"}')
        events.write_text('{"event": "start"}')
        diagnostics.write_text('{"win_rate": 30.0}')

        pruned = prune_unpromoted_checkpoints(round_dir)
        assert len(pruned) == 1
        assert candidate in pruned
        assert not candidate.exists()
        assert manifest.exists()
        assert events.exists()
        assert diagnostics.exists()


# ==============================================================================
# 6. OOS GATE & ANTI-COLLAPSE EVALUATION TESTS
# ==============================================================================


class TestOOSGateAndGuards:
    """Tests for quantitative edge criteria and anti-collapse gates."""

    def test_candidate_passing_all_criteria(self) -> None:
        """Candidate satisfying all hurdles passes gate with zero rejection reasons."""
        metrics = {
            "expectancy": 0.08,
            "win_rate": 42.5,
            "risk_reward_ratio": 2.1,
            "total_trades": 35,
            "max_equity_drawdown_pct": 7.0,
            "hold_share": 0.50,
            "entry_direction_imbalance": 0.10,
            "entropy": 0.55,
        }
        res = evaluate_candidate_oos_gate(metrics)
        assert res.passed_gate is True
        assert len(res.reasons) == 0

    def test_trade_count_boundary_29_vs_30(self) -> None:
        """Trade count 29 fails gate; exactly 30 passes gate."""
        base = {
            "expectancy": 0.08,
            "win_rate": 45.0,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.40,
            "entry_direction_imbalance": 0.0,
            "entropy": 0.50,
        }
        res29 = evaluate_candidate_oos_gate({**base, "total_trades": 29})
        assert res29.passed_gate is False
        assert any("oos_trades_insufficient:29<30" in r for r in res29.reasons)

        res30 = evaluate_candidate_oos_gate({**base, "total_trades": 30})
        assert res30.passed_gate is True

    def test_expectancy_boundary_00499_vs_00500(self) -> None:
        """Expectancy +0.0499R fails; exactly +0.0500R passes."""
        base = {
            "win_rate": 40.0,
            "risk_reward_ratio": 2.0,
            "total_trades": 35,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.40,
            "entry_direction_imbalance": 0.0,
            "entropy": 0.50,
        }
        res_fail = evaluate_candidate_oos_gate({**base, "expectancy": 0.0499})
        assert res_fail.passed_gate is False
        assert any("expectancy_below_hurdle" in r for r in res_fail.reasons)

        res_pass = evaluate_candidate_oos_gate({**base, "expectancy": 0.0500})
        assert res_pass.passed_gate is True

    def test_win_rate_fractional_input_supported(self) -> None:
        """Fractional win rate 0.42 is converted to 42.0% and passes."""
        metrics = {
            "expectancy": 0.06,
            "win_rate": 0.42,  # Fractional
            "risk_reward_ratio": 2.0,
            "total_trades": 35,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.40,
            "entry_direction_imbalance": 0.0,
            "entropy": 0.50,
        }
        res = evaluate_candidate_oos_gate(metrics)
        assert res.passed_gate is True
        assert res.win_rate == pytest.approx(42.0)

    def test_anti_hold_collapse_guard(self) -> None:
        """Hold share >= 0.80 or entropy <= 0.35 fails gate with collapse reason."""
        base = {
            "expectancy": 0.08,
            "win_rate": 45.0,
            "risk_reward_ratio": 2.0,
            "total_trades": 35,
            "max_equity_drawdown_pct": 5.0,
            "entry_direction_imbalance": 0.0,
        }
        res_hold = evaluate_candidate_oos_gate({**base, "hold_share": 0.85, "entropy": 0.50})
        assert res_hold.passed_gate is False
        assert any("hold_collapse_detected" in r for r in res_hold.reasons)

        res_entropy = evaluate_candidate_oos_gate({**base, "hold_share": 0.50, "entropy": 0.20})
        assert res_entropy.passed_gate is False
        assert any("entropy_collapse_detected" in r for r in res_entropy.reasons)

    def test_anti_directional_collapse_guard(self) -> None:
        """Directional imbalance exceeding abs(0.60) is rejected."""
        base = {
            "expectancy": 0.08,
            "win_rate": 45.0,
            "risk_reward_ratio": 2.0,
            "total_trades": 35,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.50,
            "entropy": 0.50,
        }
        # 100% BUY
        res_long = evaluate_candidate_oos_gate({**base, "entry_direction_imbalance": 1.0})
        assert res_long.passed_gate is False
        assert any("directional_imbalance_exceeded" in r for r in res_long.reasons)

        # 100% SELL
        res_short = evaluate_candidate_oos_gate({**base, "entry_direction_imbalance": -0.80})
        assert res_short.passed_gate is False
        assert any("directional_imbalance_exceeded" in r for r in res_short.reasons)

        # Balanced (imbalance = 0.20 <= 0.60)
        res_bal = evaluate_candidate_oos_gate({**base, "entry_direction_imbalance": 0.20})
        assert res_bal.passed_gate is True

    def test_empty_metrics_fails_safely(self) -> None:
        """Empty or None metrics fail closed with zero trades reason and no exception."""
        res = evaluate_candidate_oos_gate({})
        assert res.passed_gate is False
        assert any("oos_trades_insufficient:0<30" in r for r in res.reasons)


# ==============================================================================
# 7. AUTORL ENGINE SUPERVISOR LOOP INTEGRATION TESTS
# ==============================================================================


class TestAutoRLEngineSupervisor:
    """Integration tests for the complete AutoRLEngine multi-round loop."""

    def test_engine_initialization(self, tmp_path: Path) -> None:
        """Engine initializes runs_dir, manifest manager, state machine, and DD monitor."""
        runs_dir = tmp_path / "v193_runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir)
        engine = AutoRLEngine(cfg)

        assert engine.runs_dir.exists()
        assert engine.current_round == 0
        assert engine.state_machine.current_timeframe == "M1"
        assert engine.dd_monitor.cumulative_drawdown_pct == 0.0
        assert engine.terminated is False

    def test_step_round_execution(self, tmp_path: Path) -> None:
        """Single step_round advances round counter, saves artifacts, and records manifest."""
        runs_dir = tmp_path / "v193_runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir)
        engine = AutoRLEngine(cfg)

        def mock_executor(
            r: int, tf: str, seed: str, r_dir: Path
        ) -> tuple[bytes, dict[str, Any]]:
            return b"CANDIDATE_R1", {"ppo_loss": 0.25, "entropy": 0.55}

        def mock_oos(cand: Any, tf: str) -> dict[str, Any]:
            return {
                "expectancy": 0.065,
                "win_rate": 42.0,
                "risk_reward_ratio": 2.05,
                "total_trades": 35,
                "max_equity_drawdown_pct": 6.0,
                "hold_share": 0.45,
                "entry_direction_imbalance": 0.10,
                "entropy": 0.55,
                "pnl": 450.0,
            }

        res = engine.step_round(round_executor=mock_executor, oos_evaluator=mock_oos)
        assert res["round"] == 1
        assert res["timeframe"] == "M1"
        assert res["promoted"] is True
        assert engine.current_round == 1
        assert engine.champion_checkpoint is not None
        assert "champion_round_1.pkl" in engine.active_seed

    def test_rollback_and_pruning_on_candidate_failure(self, tmp_path: Path) -> None:
        """Rejected candidate rolls back active seed to champion and prunes unpromoted pkl."""
        runs_dir = tmp_path / "v193_runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir)
        engine = AutoRLEngine(cfg)

        # Round 1: Champion
        r1_champ = runs_dir / "champion_r1.pkl"
        r1_champ.write_bytes(b"CHAMP_R1")
        engine.champion_checkpoint = str(r1_champ)
        engine.active_seed = str(r1_champ)
        engine.best_expectancy = 0.06

        # Round 2: Degraded candidate
        def mock_executor(
            r: int, tf: str, seed: str, r_dir: Path
        ) -> tuple[bytes, dict[str, Any]]:
            return b"DEGRADED_R2", {"ppo_loss": 1.25, "entropy": 0.20}

        def mock_oos(cand: Any, tf: str) -> dict[str, Any]:
            return {
                "expectancy": -0.15,
                "win_rate": 20.0,
                "risk_reward_ratio": 1.5,
                "total_trades": 12,
                "max_equity_drawdown_pct": 4.0,
                "hold_share": 0.90,
                "entry_direction_imbalance": 0.0,
                "entropy": 0.20,
                "pnl": -300.0,
            }

        res = engine.step_round(round_executor=mock_executor, oos_evaluator=mock_oos)
        assert res["promoted"] is False
        assert engine.active_seed == str(r1_champ), "Must roll back to champion seed"

        # Check pruning: round_01 candidate.pkl must be unlinked
        r2_dir = runs_dir / "round_01_m1"
        assert not (r2_dir / "candidate.pkl").exists()

    def test_three_m1_failures_trigger_m5_migration_in_loop(
        self, tmp_path: Path
    ) -> None:
        """3 consecutive non-improving rounds on M1 automatically migrate engine to M5."""
        runs_dir = tmp_path / "v193_runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir)
        engine = AutoRLEngine(cfg)

        def failing_executor(
            r: int, tf: str, seed: str, r_dir: Path
        ) -> tuple[bytes, dict[str, Any]]:
            return b"SEED", {"ppo_loss": 0.5}

        def failing_oos(cand: Any, tf: str) -> dict[str, Any]:
            return {
                "expectancy": -0.05,
                "win_rate": 30.0,
                "risk_reward_ratio": 2.0,
                "total_trades": 20,
                "max_equity_drawdown_pct": 3.0,
                "hold_share": 0.50,
                "entry_direction_imbalance": 0.0,
                "entropy": 0.50,
                "pnl": -200.0,
            }

        # Rounds 1, 2, 3 fail on M1
        r1 = engine.step_round(round_executor=failing_executor, oos_evaluator=failing_oos)
        assert r1["timeframe"] == "M1"
        assert engine.state_machine.current_timeframe == "M1"

        r2 = engine.step_round(round_executor=failing_executor, oos_evaluator=failing_oos)
        assert r2["timeframe"] == "M1"
        assert engine.state_machine.current_timeframe == "M1"

        r3 = engine.step_round(round_executor=failing_executor, oos_evaluator=failing_oos)
        assert r3["timeframe"] == "M1"
        assert r3["next_timeframe"] == "M5"
        assert engine.state_machine.current_timeframe == "M5"

        # Round 4 now executes on M5!
        r4 = engine.step_round(round_executor=failing_executor, oos_evaluator=failing_oos)
        assert r4["timeframe"] == "M5"

    def test_emergency_drawdown_halt_in_loop(self, tmp_path: Path) -> None:
        """Cumulative account drawdown exceeding 15% immediately halts the supervisor."""
        runs_dir = tmp_path / "v193_runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir, initial_equity=10000.0)
        engine = AutoRLEngine(cfg)

        def crash_executor(
            r: int, tf: str, seed: str, r_dir: Path
        ) -> tuple[bytes, dict[str, Any]]:
            return b"CRASH", {"ppo_loss": 0.5}

        # Round 1: -$800 (8% DD)
        engine.step_round(
            round_executor=crash_executor,
            oos_evaluator=lambda c, tf: {
                "expectancy": -0.10,
                "total_trades": 15,
                "pnl": -800.0,
            },
        )
        assert engine.terminated is False

        # Round 2: -$800 (Total -$1600 = 16% DD > 15% -> Emergency Halt!)
        r2 = engine.step_round(
            round_executor=crash_executor,
            oos_evaluator=lambda c, tf: {
                "expectancy": -0.10,
                "total_trades": 15,
                "pnl": -800.0,
            },
        )
        assert r2["emergency_halt"] is True
        assert engine.terminated is True
        assert engine.termination_reason == "emergency_drawdown_halt"

        # Round 3 refuses to execute
        r3 = engine.step_round(round_executor=crash_executor)
        assert r3["status"] == "emergency_halt"

    def test_full_autotuning_workflow_simulation(self, tmp_path: Path) -> None:
        """Full multi-round workflow: M1 fails -> M5 migration -> promotion -> manifest check."""
        runs_dir = tmp_path / "v193_runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir)
        engine = AutoRLEngine(cfg)

        sim_rounds = [
            # (exp, wr, trades, dd, pnl, tf_expected)
            (-0.06, 32.0, 20, 3.0, -300.0, "M1"),
            (-0.04, 35.0, 22, 2.5, -250.0, "M1"),
            (-0.02, 37.0, 25, 2.0, -150.0, "M1"),  # 3rd M1 loss -> switches to M5!
            (0.01, 39.0, 28, 3.0, 100.0, "M5"),
            (0.075, 43.0, 38, 5.0, 600.0, "M5"),  # Passes OOS gate -> Champion!
        ]

        def dynamic_executor(
            r: int, tf: str, seed: str, r_dir: Path
        ) -> tuple[bytes, dict[str, Any]]:
            return f"MODEL_ROUND_{r}".encode(), {"ppo_loss": 0.4 / r}

        def dynamic_oos(cand: Any, tf: str) -> dict[str, Any]:
            r_idx = engine.current_round  # 0-indexed before advance
            exp, wr, trades, dd, pnl, _ = sim_rounds[r_idx]
            return {
                "expectancy": exp,
                "win_rate": wr,
                "risk_reward_ratio": 2.05,
                "total_trades": trades,
                "max_equity_drawdown_pct": dd,
                "hold_share": 0.45,
                "entry_direction_imbalance": 0.05,
                "entropy": 0.55,
                "pnl": pnl,
            }

        # Run 5 rounds
        summary = engine.run_loop(
            max_rounds=5,
            round_executor=dynamic_executor,
            oos_evaluator=dynamic_oos,
        )

        assert summary["total_rounds_executed"] == 5
        assert summary["current_timeframe"] == "M5"
        assert summary["champion_checkpoint"] is not None
        assert "champion_round_5.pkl" in summary["active_seed"]

        # Check manifests on disk
        manifest = engine.manifest_mgr.load_manifest()
        chart = engine.manifest_mgr.load_evolution_chart()
        assert len(manifest["rounds"]) == 5
        assert len(chart) == 5
        assert manifest["rounds"][-1]["promoted"] is True
        assert manifest["rounds"][-1]["expectancy"] == 0.075
        assert manifest["rounds"][-1]["total_trades"] == 38
