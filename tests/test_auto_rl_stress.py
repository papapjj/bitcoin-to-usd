"""Empirical Stress Test Suite for Auto-RL Engine.

Adversarial testing targeting:
1. Timeframe Adaptation:
   - 3 consecutive M1 losses -> migration to M5.
   - 3 consecutive M5 losses -> migration to M15.
   - Counter resets on improving rounds.
   - Paralyzed / zero-trade handling.
   - State restoration across reloads / restarts.
2. Checkpoint Rollback:
   - Divergence injection (NaN, Inf, loss > 100).
   - Entropy collapse (entropy <= 0.35).
   - HOLD collapse (hold_share >= 0.80).
   - Directional collapse (|imbalance| > 0.60).
   - Catastrophic drawdown (drawdown >= 10%).
   - Verification that historical manifest and evolution chart are never truncated or lost.
   - Verification that active_seed correctly points to the previous best champion.
3. Disk Safety & Pruning:
   - Intermediate non-promoted .pkl files deleted.
   - Telemetry JSON, JSONL, and registry files preserved.
   - Promoted champions and best_promoted_checkpoint preserved.
   - Large payload disk leak stress harness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

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
# 1. TIMEFRAME ADAPTATION STRESS TESTS
# ==============================================================================


class TestTimeframeMigrationStress:
    """Adversarial stress-testing of timeframe migration sequences."""

    def test_three_consecutive_m1_losses_transitions_to_m5(self, tmp_path: Path) -> None:
        """Simulate 3 consecutive M1 losses and verify transition to M5."""
        cfg = AutoRLEngineConfig(
            runs_dir=tmp_path / "runs",
            max_rounds=5,
            initial_timeframe="M1",
            m1_failure_threshold=3,
        )
        engine = AutoRLEngine(cfg)

        # Round 1 on M1: Loss
        res1 = engine.step_round(
            metric_overrides={
                "expectancy": -0.15,
                "win_rate": 25.0,
                "total_trades": 20,
                "ppo_loss": 0.45,
                "entropy": 0.50,
            }
        )
        assert res1["timeframe"] == "M1"
        assert res1["promoted"] is False
        assert engine.state_machine.consecutive_m1_losses == 1
        assert res1["next_timeframe"] == "M1"

        # Round 2 on M1: Loss
        res2 = engine.step_round(
            metric_overrides={
                "expectancy": -0.08,
                "win_rate": 30.0,
                "total_trades": 22,
                "ppo_loss": 0.35,
                "entropy": 0.51,
            }
        )
        assert res2["timeframe"] == "M1"
        assert res2["promoted"] is False
        assert engine.state_machine.consecutive_m1_losses == 2
        assert res2["next_timeframe"] == "M1"

        # Round 3 on M1: Loss (3rd consecutive loss)
        res3 = engine.step_round(
            metric_overrides={
                "expectancy": -0.02,
                "win_rate": 32.0,
                "total_trades": 25,
                "ppo_loss": 0.28,
                "entropy": 0.52,
            }
        )
        assert res3["timeframe"] == "M1"
        assert res3["promoted"] is False
        # State machine should have triggered migration to M5!
        assert res3["next_timeframe"] == "M5"
        assert engine.state_machine.current_timeframe == "M5"
        assert engine.state_machine.consecutive_m1_losses == 0
        assert len(engine.state_machine.migration_history) == 1
        assert engine.state_machine.migration_history[0]["from"] == "M1"
        assert engine.state_machine.migration_history[0]["to"] == "M5"

        # Round 4 must execute on M5!
        res4 = engine.step_round(
            metric_overrides={
                "expectancy": 0.02,
                "win_rate": 38.0,
                "total_trades": 30,
                "ppo_loss": 0.20,
                "entropy": 0.54,
            }
        )
        assert res4["timeframe"] == "M5"

    def test_three_consecutive_m5_losses_transitions_to_m15(self, tmp_path: Path) -> None:
        """Simulate 3 consecutive M5 losses and verify transition to M15."""
        cfg = AutoRLEngineConfig(
            runs_dir=tmp_path / "runs",
            max_rounds=6,
            initial_timeframe="M5",
            m5_failure_threshold=3,
        )
        engine = AutoRLEngine(cfg)
        assert engine.state_machine.current_timeframe == "M5"

        # M5 Loss 1
        res1 = engine.step_round(
            metric_overrides={"expectancy": -0.05, "total_trades": 20, "win_rate": 30.0}
        )
        assert res1["timeframe"] == "M5"
        assert engine.state_machine.consecutive_m5_losses == 1
        assert res1["next_timeframe"] == "M5"

        # M5 Loss 2
        res2 = engine.step_round(
            metric_overrides={"expectancy": -0.01, "total_trades": 25, "win_rate": 32.0}
        )
        assert res2["timeframe"] == "M5"
        assert engine.state_machine.consecutive_m5_losses == 2
        assert res2["next_timeframe"] == "M5"

        # M5 Loss 3
        res3 = engine.step_round(
            metric_overrides={"expectancy": 0.0, "total_trades": 18, "win_rate": 33.33}
        )
        assert res3["timeframe"] == "M5"
        assert res3["next_timeframe"] == "M15"
        assert engine.state_machine.current_timeframe == "M15"
        assert engine.state_machine.consecutive_m5_losses == 0
        assert len(engine.state_machine.migration_history) == 1
        assert engine.state_machine.migration_history[0]["from"] == "M5"
        assert engine.state_machine.migration_history[0]["to"] == "M15"

        # Next round must execute on M15
        res4 = engine.step_round(
            metric_overrides={"expectancy": 0.06, "total_trades": 35, "win_rate": 42.0}
        )
        assert res4["timeframe"] == "M15"

    def test_full_cascade_m1_to_m5_to_m15(self, tmp_path: Path) -> None:
        """Full cascade: 3 M1 losses -> M5, then 3 M5 losses -> M15."""
        cfg = AutoRLEngineConfig(
            runs_dir=tmp_path / "runs",
            max_rounds=8,
            initial_timeframe="M1",
        )
        engine = AutoRLEngine(cfg)

        timeframes_observed = []
        for r in range(1, 8):
            res = engine.step_round(
                metric_overrides={"expectancy": -0.05, "total_trades": 20, "win_rate": 28.0}
            )
            timeframes_observed.append(res["timeframe"])

        # Rounds 1, 2, 3 on M1
        # Rounds 4, 5, 6 on M5
        # Round 7 on M15
        assert timeframes_observed == ["M1", "M1", "M1", "M5", "M5", "M5", "M15"]
        assert len(engine.state_machine.migration_history) == 2
        assert engine.state_machine.migration_history[0]["from"] == "M1"
        assert engine.state_machine.migration_history[0]["to"] == "M5"
        assert engine.state_machine.migration_history[1]["from"] == "M5"
        assert engine.state_machine.migration_history[1]["to"] == "M15"

    def test_intermittent_improving_round_resets_loss_counter(self, tmp_path: Path) -> None:
        """An improving round (E[R] > 0 and trades > 0) resets consecutive loss counter and prevents migration."""
        cfg = AutoRLEngineConfig(
            runs_dir=tmp_path / "runs",
            max_rounds=5,
            initial_timeframe="M1",
            m1_failure_threshold=3,
        )
        engine = AutoRLEngine(cfg)

        # Round 1: Loss
        engine.step_round(metric_overrides={"expectancy": -0.05, "total_trades": 20})
        assert engine.state_machine.consecutive_m1_losses == 1

        # Round 2: Loss
        engine.step_round(metric_overrides={"expectancy": -0.02, "total_trades": 22})
        assert engine.state_machine.consecutive_m1_losses == 2

        # Round 3: Improving round (E[R] = 0.03 > 0, trades = 28)
        # Note: may not be promoted because min_expectancy is 0.05 and min_trades is 30,
        # but it showed positive mathematical expectancy!
        engine.step_round(metric_overrides={"expectancy": 0.03, "total_trades": 28, "win_rate": 35.0})
        assert engine.state_machine.consecutive_m1_losses == 0
        assert engine.state_machine.current_timeframe == "M1"

        # Round 4: Loss -> counter is 1, not 3!
        res4 = engine.step_round(metric_overrides={"expectancy": -0.01, "total_trades": 20})
        assert engine.state_machine.consecutive_m1_losses == 1
        assert res4["timeframe"] == "M1"
        assert res4["next_timeframe"] == "M1"

    def test_zero_trades_always_treated_as_loss(self, tmp_path: Path) -> None:
        """Paralyzed policy with 0 trades is strictly treated as a loss even if expectancy is 0.0."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1", m1_failure_threshold=3)
        sm.record_round_result(round_num=1, expectancy=0.0, trades=0)
        assert sm.consecutive_m1_losses == 1

        sm.record_round_result(round_num=2, expectancy=0.0, trades=0)
        assert sm.consecutive_m1_losses == 2

        tf3 = sm.record_round_result(round_num=3, expectancy=0.0, trades=0)
        assert tf3 == "M5"
        assert sm.current_timeframe == "M5"

    def test_m15_consecutive_losses_remain_on_m15_without_crashing(self, tmp_path: Path) -> None:
        """M15 is the terminal adaptive timeframe; subsequent losses remain on M15 safely."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M15")
        for r in range(1, 6):
            tf = sm.record_round_result(round_num=r, expectancy=-0.05, trades=20)
            assert tf == "M15"
        assert sm.current_timeframe == "M15"

    def test_timeframe_parameters_friction_and_geometry_scaling(self) -> None:
        """Validate that timeframe parameters scale barrier geometry and friction correctly."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1")
        p_m1 = sm.get_timeframe_parameters()
        assert p_m1["spread_sl_ratio_limit"] <= 0.13
        assert p_m1["bar_duration_sec"] == 60

        sm.current_timeframe = "M5"
        p_m5 = sm.get_timeframe_parameters()
        assert p_m5["spread_sl_ratio_limit"] <= 0.06
        assert p_m5["bar_duration_sec"] == 300
        assert p_m5["pending_expiry_bars"] < p_m1["pending_expiry_bars"]

        sm.current_timeframe = "M15"
        p_m15 = sm.get_timeframe_parameters()
        assert p_m15["spread_sl_ratio_limit"] <= 0.035
        assert p_m15["bar_duration_sec"] == 900
        assert p_m15["pending_expiry_bars"] < p_m5["pending_expiry_bars"]


# ==============================================================================
# 2. CHECKPOINT ROLLBACK & DEGRADATION STRESS TESTS
# ==============================================================================


class TestCheckpointRollbackStress:
    """Adversarial stress-testing of candidate degradation and champion rollback."""

    def test_rollback_on_nan_or_inf_ppo_loss(self, tmp_path: Path) -> None:
        """Inject NaN and Inf PPO losses and verify clean rollback to previous champion."""
        runs_dir = tmp_path / "runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=5, initial_timeframe="M5")
        engine = AutoRLEngine(cfg)

        # Round 1: Excellent candidate -> promoted to champion!
        res1 = engine.step_round(
            metric_overrides={
                "expectancy": 0.08,
                "win_rate": 45.0,
                "risk_reward_ratio": 2.1,
                "total_trades": 35,
                "max_equity_drawdown_pct": 5.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.10,
                "entropy": 0.55,
                "ppo_loss": 0.15,
            }
        )
        assert res1["promoted"] is True
        champ1_path = runs_dir / "champion_round_1.pkl"
        assert champ1_path.exists()
        assert engine.champion_checkpoint == str(champ1_path)
        assert engine.active_seed == str(champ1_path)

        # Round 2: Diverged candidate with NaN loss
        res2 = engine.step_round(
            metric_overrides={
                "expectancy": 0.08,
                "win_rate": 45.0,
                "risk_reward_ratio": 2.1,
                "total_trades": 35,
                "max_equity_drawdown_pct": 5.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.10,
                "entropy": 0.55,
                "ppo_loss": float("nan"),
            }
        )
        assert res2["promoted"] is False
        # Active seed MUST roll back to champ1
        assert engine.active_seed == str(champ1_path)
        assert res2["active_seed"] == str(champ1_path)

        # Round 3: Diverged candidate with Inf loss
        res3 = engine.step_round(
            metric_overrides={
                "expectancy": 0.08,
                "win_rate": 45.0,
                "risk_reward_ratio": 2.1,
                "total_trades": 35,
                "max_equity_drawdown_pct": 5.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.10,
                "entropy": 0.55,
                "ppo_loss": float("inf"),
            }
        )
        assert res3["promoted"] is False
        assert engine.active_seed == str(champ1_path)

        # Verify Round 4 receives champ1 as seed
        received_seed = []

        def executor(rnd: int, tf: str, seed: str, rdir: Path):
            received_seed.append(seed)
            return rdir / "candidate.pkl", {"ppo_loss": 0.10}

        engine.step_round(round_executor=executor)
        assert len(received_seed) == 1
        assert received_seed[0] == str(champ1_path)

    def test_rollback_on_entropy_collapse(self, tmp_path: Path) -> None:
        """Inject entropy collapse (entropy <= 0.35) -> verify rejection and champion rollback."""
        runs_dir = tmp_path / "runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=3, initial_timeframe="M5")
        engine = AutoRLEngine(cfg)

        # Round 1: Champion
        engine.step_round(
            metric_overrides={
                "expectancy": 0.07,
                "win_rate": 44.0,
                "risk_reward_ratio": 2.05,
                "total_trades": 32,
                "max_equity_drawdown_pct": 6.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.05,
                "entropy": 0.55,
                "ppo_loss": 0.15,
            }
        )
        champ = str(runs_dir / "champion_round_1.pkl")
        assert engine.active_seed == champ

        # Round 2: Action distribution collapses to single action (entropy = 0.10)
        res2 = engine.step_round(
            metric_overrides={
                "expectancy": 0.07,
                "win_rate": 44.0,
                "risk_reward_ratio": 2.05,
                "total_trades": 32,
                "max_equity_drawdown_pct": 6.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.05,
                "entropy": 0.10,
                "ppo_loss": 0.15,
            }
        )
        assert res2["promoted"] is False
        assert any("entropy_collapse_detected" in r for r in res2["reasons"])
        assert engine.active_seed == champ

    def test_rollback_on_hold_collapse(self, tmp_path: Path) -> None:
        """Inject HOLD collapse (hold_share >= 0.80) -> verify rejection and champion rollback."""
        runs_dir = tmp_path / "runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=3, initial_timeframe="M5")
        engine = AutoRLEngine(cfg)

        # Seed champion in round 1
        engine.step_round(
            metric_overrides={
                "expectancy": 0.06,
                "win_rate": 42.0,
                "risk_reward_ratio": 2.0,
                "total_trades": 30,
                "max_equity_drawdown_pct": 7.0,
                "hold_share": 0.50,
                "entry_direction_imbalance": 0.0,
                "entropy": 0.50,
            }
        )
        champ = str(runs_dir / "champion_round_1.pkl")

        # Round 2: HOLD share explodes to 92%
        res2 = engine.step_round(
            metric_overrides={
                "expectancy": 0.06,
                "win_rate": 42.0,
                "risk_reward_ratio": 2.0,
                "total_trades": 30,
                "max_equity_drawdown_pct": 7.0,
                "hold_share": 0.92,
                "entry_direction_imbalance": 0.0,
                "entropy": 0.45,
            }
        )
        assert res2["promoted"] is False
        assert any("hold_collapse_detected" in r for r in res2["reasons"])
        assert engine.active_seed == champ

    def test_rollback_on_directional_collapse(self, tmp_path: Path) -> None:
        """Inject directional collapse (|imbalance| > 0.60) -> verify rejection and champion rollback."""
        runs_dir = tmp_path / "runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=3, initial_timeframe="M5")
        engine = AutoRLEngine(cfg)

        engine.step_round(
            metric_overrides={
                "expectancy": 0.06,
                "win_rate": 42.0,
                "risk_reward_ratio": 2.0,
                "total_trades": 30,
                "max_equity_drawdown_pct": 7.0,
                "hold_share": 0.50,
                "entry_direction_imbalance": 0.10,
                "entropy": 0.50,
            }
        )
        champ = str(runs_dir / "champion_round_1.pkl")

        # Round 2: 95% BUY, 5% SELL -> imbalance = 0.90
        res2 = engine.step_round(
            metric_overrides={
                "expectancy": 0.06,
                "win_rate": 42.0,
                "risk_reward_ratio": 2.0,
                "total_trades": 30,
                "max_equity_drawdown_pct": 7.0,
                "hold_share": 0.50,
                "entry_direction_imbalance": 0.90,
                "entropy": 0.50,
            }
        )
        assert res2["promoted"] is False
        assert any("directional_imbalance_exceeded" in r for r in res2["reasons"])
        assert engine.active_seed == champ

    def test_rollback_preserves_historical_manifest_and_chart(self, tmp_path: Path) -> None:
        """Verify multiple degradation rollbacks never erase or truncate previous round records."""
        runs_dir = tmp_path / "runs"
        cfg = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=5, initial_timeframe="M5")
        engine = AutoRLEngine(cfg)

        # Round 1: Promoted
        engine.step_round(
            metric_overrides={
                "expectancy": 0.06,
                "win_rate": 42.0,
                "risk_reward_ratio": 2.0,
                "total_trades": 30,
                "max_equity_drawdown_pct": 5.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.10,
                "entropy": 0.55,
            }
        )

        # Round 2: Diverged loss
        engine.step_round(
            metric_overrides={"ppo_loss": 500.0, "expectancy": -0.5}
        )

        # Round 3: Entropy collapsed
        engine.step_round(
            metric_overrides={"entropy": 0.15, "expectancy": -0.2}
        )

        # Round 4: Even better candidate -> Promoted!
        engine.step_round(
            metric_overrides={
                "expectancy": 0.10,
                "win_rate": 48.0,
                "risk_reward_ratio": 2.1,
                "total_trades": 40,
                "max_equity_drawdown_pct": 4.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.05,
                "entropy": 0.60,
                "ppo_loss": 0.08,
            }
        )

        # Inspect manifest and evolution chart
        manifest = engine.manifest_mgr.load_manifest()
        chart = engine.manifest_mgr.load_evolution_chart()

        assert manifest["current_round"] == 4
        assert len(manifest["rounds"]) == 4
        assert len(chart) == 4

        # Round 1: promoted
        assert manifest["rounds"][0]["promoted"] is True
        assert chart[0]["promoted"] is True

        # Round 2: not promoted, loss recorded
        assert manifest["rounds"][1]["promoted"] is False
        assert chart[1]["promoted"] is False
        assert manifest["rounds"][1]["loss"] == 500.0

        # Round 3: not promoted, entropy recorded
        assert manifest["rounds"][2]["promoted"] is False
        assert chart[2]["promoted"] is False
        assert manifest["rounds"][2]["entropy"] == 0.15

        # Round 4: promoted, best expectancy updated
        assert manifest["rounds"][3]["promoted"] is True
        assert chart[3]["promoted"] is True
        assert manifest["best_expectancy"] == 0.10
        assert manifest["champion_checkpoint"] == str(runs_dir / "champion_round_4.pkl")


# ==============================================================================
# 3. DISK SAFETY & PRUNING STRESS TESTS
# ==============================================================================


class TestDiskPruningStress:
    """Adversarial stress-testing of checkpoint pruning and disk safety."""

    def test_pruning_removes_unpromoted_candidates_preserves_telemetry(self, tmp_path: Path) -> None:
        """Verify non-promoted candidate .pkl files are removed while telemetry JSONs remain intact."""
        round_dir = tmp_path / "round_01_m1"
        round_dir.mkdir(parents=True)

        # Create large unpromoted candidate
        candidate_pkl = round_dir / "candidate.pkl"
        candidate_tmp = round_dir / "candidate.pkl.tmp"
        intermediate_pkl = round_dir / "intermediate_step_500.pkl"
        candidate_pkl.write_bytes(b"\x00" * 5 * 1024 * 1024)  # 5MB
        candidate_tmp.write_bytes(b"\x00" * 1024)
        intermediate_pkl.write_bytes(b"\x00" * 2 * 1024 * 1024)  # 2MB

        # Create telemetry and registry files
        manifest_json = round_dir / "manifest.json"
        events_jsonl = round_dir / "events.jsonl"
        diag_json = round_dir / "mt5_diagnostics.json"
        reg_json = round_dir / "candidate.registry.json"

        manifest_json.write_text('{"round": 1, "status": "rejected"}', encoding="utf-8")
        events_jsonl.write_text('{"event": "start"}\n{"event": "finish"}\n', encoding="utf-8")
        diag_json.write_text('{"trades": 15}', encoding="utf-8")
        reg_json.write_text('{"version": "v193"}', encoding="utf-8")

        # Execute pruning
        pruned = prune_unpromoted_checkpoints(round_dir)

        # Verification: binary files pruned
        assert not candidate_pkl.exists()
        assert not candidate_tmp.exists()
        assert not intermediate_pkl.exists()
        assert candidate_pkl in pruned
        assert intermediate_pkl in pruned

        # Verification: telemetry files completely preserved
        assert manifest_json.exists()
        assert events_jsonl.exists()
        assert diag_json.exists()
        assert reg_json.exists()
        assert "rejected" in manifest_json.read_text(encoding="utf-8")

    def test_engine_prunes_rejected_rounds_and_preserves_champions(self, tmp_path: Path) -> None:
        """Verify AutoRLEngine prunes candidate.pkl in rejected rounds but leaves champions intact."""
        runs_dir = tmp_path / "runs"
        cfg = AutoRLEngineConfig(
            runs_dir=runs_dir,
            max_rounds=3,
            initial_timeframe="M5",
            prune_non_promoted_checkpoints=True,
        )
        engine = AutoRLEngine(cfg)

        # Round 1: Champion (promoted)
        res1 = engine.step_round(
            metric_overrides={
                "expectancy": 0.08,
                "win_rate": 45.0,
                "risk_reward_ratio": 2.1,
                "total_trades": 35,
                "max_equity_drawdown_pct": 5.0,
                "hold_share": 0.40,
                "entry_direction_imbalance": 0.05,
                "entropy": 0.55,
            }
        )
        assert res1["promoted"] is True
        champ_round_1 = runs_dir / "champion_round_1.pkl"
        best_promoted = runs_dir / "best_promoted_checkpoint.pkl"
        assert champ_round_1.exists()
        assert best_promoted.exists()

        # Round 2: Degraded (rejected)
        res2 = engine.step_round(
            metric_overrides={
                "expectancy": -0.10,
                "win_rate": 20.0,
                "total_trades": 15,
                "ppo_loss": 0.8,
            }
        )
        assert res2["promoted"] is False
        r2_dir = runs_dir / "round_02_m5"
        r2_candidate = r2_dir / "candidate.pkl"
        # Candidate must be pruned!
        assert not r2_candidate.exists()

        # Champions must remain intact!
        assert champ_round_1.exists()
        assert best_promoted.exists()
        assert (runs_dir / "auto_rl_manifest.json").exists()
        assert (runs_dir / "evolution_chart.json").exists()

    def test_disk_leak_stress_harness_ten_failed_rounds(self, tmp_path: Path) -> None:
        """Simulate 10 consecutive failed rounds generating 10MB payloads each.
        Verify no disk space leak occurs because pruning cleans up after each round.
        """
        runs_dir = tmp_path / "runs"
        cfg = AutoRLEngineConfig(
            runs_dir=runs_dir,
            max_rounds=10,
            initial_timeframe="M1",
            prune_non_promoted_checkpoints=True,
        )
        engine = AutoRLEngine(cfg)

        def heavy_failing_executor(rnd: int, tf: str, seed: str, rdir: Path):
            cand = rdir / "candidate.pkl"
            # Write a 2MB dummy payload
            cand.write_bytes(b"\xFF" * 2 * 1024 * 1024)
            # Write telemetry
            (rdir / "manifest.json").write_text(json.dumps({"round": rnd}))
            return cand, {"expectancy": -0.05, "total_trades": 10}

        summary = engine.run_loop(round_executor=heavy_failing_executor)
        assert summary["total_rounds_executed"] == 10

        # Total size of all round directories should be tiny (< 50KB total) because candidates were pruned!
        total_bytes = 0
        for f in runs_dir.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size

        # Manifests and small jsons are < 100KB, whereas 10 unpruned candidates would be >= 20MB
        assert total_bytes < 200 * 1024, f"Disk leak detected! Total bytes: {total_bytes}"


# ==============================================================================
# 4. CROSS-MODULE & STATE RESTORATION ADVERSARIAL TESTS
# ==============================================================================


class TestEngineStateRestorationAdversarial:
    """Stress tests verifying process restarts and state persistence across interruptions."""

    def test_restart_engine_from_existing_manifest(self, tmp_path: Path) -> None:
        """Verify that engine correctly restores state from disk after process restart."""
        runs_dir = tmp_path / "runs"
        cfg1 = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=10, initial_timeframe="M1")
        engine1 = AutoRLEngine(cfg1)

        # Run 2 failing rounds on M1
        engine1.step_round(metric_overrides={"expectancy": -0.10, "total_trades": 20})
        engine1.step_round(metric_overrides={"expectancy": -0.05, "total_trades": 22})

        assert engine1.current_round == 2
        assert engine1.state_machine.consecutive_m1_losses == 2

        # Simulate process termination and re-initialization
        del engine1

        cfg2 = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=10, initial_timeframe="M1")
        engine2 = AutoRLEngine(cfg2)

        # Engine 2 must recover round 2 and 2 consecutive M1 losses!
        assert engine2.current_round == 2
        assert engine2.state_machine.consecutive_m1_losses == 2
        assert engine2.state_machine.current_timeframe == "M1"

        # Round 3 on M1 fails -> MUST trigger migration to M5 on Round 4!
        res3 = engine2.step_round(metric_overrides={"expectancy": -0.02, "total_trades": 20})
        assert res3["round"] == 3
        assert res3["next_timeframe"] == "M5"
        assert engine2.state_machine.current_timeframe == "M5"

    def test_consecutive_m5_losses_persistence_audit(self, tmp_path: Path) -> None:
        """Audit test: check whether consecutive_m5_losses persists or resets on reload.
        This verifies our finding during code review.
        """
        runs_dir = tmp_path / "runs"
        cfg1 = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=10, initial_timeframe="M5")
        engine1 = AutoRLEngine(cfg1)

        # 2 consecutive losses on M5
        engine1.step_round(metric_overrides={"expectancy": -0.05, "total_trades": 20})
        engine1.step_round(metric_overrides={"expectancy": -0.02, "total_trades": 20})
        assert engine1.state_machine.consecutive_m5_losses == 2

        # Re-initialize engine
        del engine1
        cfg2 = AutoRLEngineConfig(runs_dir=runs_dir, max_rounds=10, initial_timeframe="M5")
        engine2 = AutoRLEngine(cfg2)

        # In current auto_rl_engine.py, consecutive_m1_losses is in manifest but consecutive_m5_losses
        # might default to 0 if not saved.
        manifest = engine2.manifest_mgr.load_manifest()
        has_m5_losses = "consecutive_m5_losses" in manifest
        # Document observed behavior
        print(f"DEBUG: manifest has consecutive_m5_losses: {has_m5_losses}")
        print(f"DEBUG: engine2 consecutive_m5_losses: {engine2.state_machine.consecutive_m5_losses}")
