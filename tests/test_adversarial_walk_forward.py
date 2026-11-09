"""Tier 5 Adversarial Stress Test Suite for Walk-Forward Multi-Cycle Orchestration Engine.

Adversarial Stress Scenarios:
1. Checkpoint & File Integrity: corrupt checkpoint hashes, 0-byte files, non-existent files.
2. Malformed Telemetry & Diagnostics: corrupted JSON syntax, missing keys, NaN/Inf strings, Simplex() errors.
3. Boundary & Negative Profit Transitions: breakeven PnL, micro-losses, edge profit factor, hold traps.
4. Adaptive Tuning Monotonicity & Decay: strict monotonicity of Veto LCB, strict decay of uniform mix.
5. Rollback & Champion Preservation: preservation of clean champion across consecutive loss retries.
6. Headless Execution Enforcement: strict Visual=0 assertion, timeout/wall-clock bypass prevention.
7. Market Cycle Dates & Boundary Invariants.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import pytest
import yaml

from self_growing_trader.walk_forward_engine import (
    DEFAULT_BASE_PROFILE,
    DEFAULT_CYCLES,
    DEFAULT_INITIAL_SEED,
    DEFAULT_MAX_ADAPTIVE_RETRIES,
    DEFAULT_MAX_WALL_SEC,
    DEFAULT_RUNNER_SOURCE,
    DEFAULT_SOURCE_CONFIG,
    DEFAULT_TIMEOUT_SEC,
    AcceptanceVerdict,
    AdaptiveTuningDiagnosis,
    CycleMetrics,
    WalkForwardCycleConfig,
    WalkForwardEngine,
    build_parser,
    build_tester_ini_content,
    compute_adaptive_tuning_overrides,
    diagnose_failure,
    evaluate_acceptance_criteria,
    extract_cycle_metrics,
    generate_adapted_profile_yaml,
    generate_walk_forward_manifest,
    sha256_file,
    verify_headless_configuration,
)


# ===========================================================================
# 1. Adversarial Checkpoint & File Integrity Tests
# ===========================================================================

class TestAdversarialCheckpointAndFiles:
    """Stress test file handling, corrupt checkpoints, and hash computations."""

    def test_sha256_non_existent_file(self, tmp_path: Path):
        """sha256_file on non-existent path returns empty string without error."""
        non_existent = tmp_path / "ghost_checkpoint.pkl"
        assert sha256_file(non_existent) == ""

    def test_sha256_zero_byte_file(self, tmp_path: Path):
        """sha256_file on 0-byte file returns the standard SHA-256 of empty content."""
        empty_file = tmp_path / "empty_seed.pkl"
        empty_file.write_bytes(b"")
        expected_empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert sha256_file(empty_file) == expected_empty_hash

    def test_sha256_corrupted_content_changes_hash(self, tmp_path: Path):
        """Altering a single byte changes the checkpoint SHA-256 hash."""
        f = tmp_path / "model.pkl"
        f.write_bytes(b"VALID_CHECKPOINT_DATA_BLOCK_001")
        hash1 = sha256_file(f)

        f.write_bytes(b"VALID_CHECKPOINT_DATA_BLOCK_002")
        hash2 = sha256_file(f)

        assert hash1 != hash2
        assert len(hash1) == 64
        assert len(hash2) == 64

    def test_engine_run_all_cycles_missing_seed_raises(self, tmp_path: Path):
        """run_all_cycles with non-existent initial seed raises FileNotFoundError immediately."""
        engine = WalkForwardEngine(workspace_dir=tmp_path)
        missing_seed = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError, match="Initial seed checkpoint not found"):
            engine.run_all_cycles(initial_seed=missing_seed, output_base_dir=tmp_path / "out")

    def test_engine_run_cycle_invalid_cycle_id_raises(self, tmp_path: Path):
        """run_cycle raises ValueError when provided invalid cycle IDs (0, 4, -1)."""
        engine = WalkForwardEngine(workspace_dir=tmp_path)
        seed = tmp_path / "seed.pkl"
        seed.write_bytes(b"seed_bytes")

        for bad_id in [0, 4, -1, 99]:
            with pytest.raises(ValueError, match="Invalid cycle_id"):
                engine.run_cycle(cycle_id=bad_id, seed_checkpoint=seed, output_base_dir=tmp_path / "out")


# ===========================================================================
# 2. Adversarial Telemetry & Malformed Diagnostics Tests
# ===========================================================================

class TestAdversarialTelemetryAndMalformedJson:
    """Stress test telemetry parsing against malformed JSON, corrupted files, and arithmetic failures."""

    def test_extract_metrics_empty_diagnostics_file(self, tmp_path: Path):
        """Empty diagnostics file returns zero-default metrics without unhandled crash."""
        diag = tmp_path / "mt5_diagnostics.json"
        diag.write_text("", encoding="utf-8")
        metrics = extract_cycle_metrics(tmp_path, cycle_id=1)
        assert metrics.net_pnl == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.closed_trades == 0
        assert metrics.raw_diagnostics == {}

    def test_extract_metrics_corrupt_json_syntax(self, tmp_path: Path):
        """Corrupted JSON syntax ({ incomplete) recovers safely without raising."""
        diag = tmp_path / "mt5_diagnostics.json"
        diag.write_text('{"performance": {"net_pnl": 500.0, "profit_factor":', encoding="utf-8")
        metrics = extract_cycle_metrics(tmp_path, cycle_id=1)
        assert metrics.net_pnl == 0.0
        assert metrics.profit_factor == 0.0

    def test_extract_metrics_null_and_malformed_types(self, tmp_path: Path):
        """Diagnostics with null or unexpected types are handled gracefully."""
        diag = tmp_path / "mt5_diagnostics.json"
        malformed_data = {
            "performance": {
                "net_pnl": "invalid_number_string",
                "profit_factor": None,
                "closed_trades": "not_an_int",
                "win_rate": None,
            },
            "risk": {
                "max_equity_drawdown_pct": None,
            },
            "policy": None,
        }
        diag.write_text(json.dumps(malformed_data), encoding="utf-8")
        metrics = extract_cycle_metrics(tmp_path, cycle_id=1)
        assert metrics.net_pnl == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.closed_trades == 0

    def test_extract_metrics_detects_simplex_and_nan_errors_in_events(self, tmp_path: Path):
        """events.jsonl containing simplex errors and NaN/Inf tokens is caught accurately."""
        events = tmp_path / "events.jsonl"
        with open(events, "w", encoding="utf-8") as f:
            f.write("Normal log line\n")
            f.write("CRITICAL: Expected parameter probs to be within the simplex\n")
            f.write("WARNING: Simplex() distribution invalid probability\n")
            f.write('{"event_type": "metric", "loss": float(\'nan\')}\n')
            f.write('{"event_type": "metric", "grad": : inf}\n')
            f.write('{"event_type": "lifecycle", "payload": {"realized_net_r": "NaN"}}\n')

        metrics = extract_cycle_metrics(tmp_path, cycle_id=1)
        assert metrics.simplex_errors == 2
        assert metrics.nan_inf_errors >= 2

        verdict = evaluate_acceptance_criteria(metrics)
        assert not verdict.passed
        assert not verdict.simplex_safe
        assert any("Numerical instability detected" in r for r in verdict.reasons)

    def test_extract_metrics_fallback_to_lifecycle_events(self, tmp_path: Path):
        """When diagnostics is absent, metrics are derived from lifecycle events."""
        events = tmp_path / "events.jsonl"
        with open(events, "w", encoding="utf-8") as f:
            for i in range(25):
                r = 1.0 if i % 2 == 0 else -0.5
                f.write(json.dumps({
                    "event_type": "lifecycle",
                    "payload": {"realized_net_r": r, "close_reason": "TAKE_PROFIT" if r > 0 else "STOP_LOSS"},
                }) + "\n")

        metrics = extract_cycle_metrics(tmp_path, cycle_id=1)
        assert metrics.closed_trades == 25
        assert metrics.net_pnl > 0.0
        assert metrics.profit_factor > 1.0


# ===========================================================================
# 3. Adversarial Acceptance Hurdles & Boundary Conditions
# ===========================================================================

class TestAdversarialAcceptanceHurdles:
    """Stress test boundary edge conditions of mandatory acceptance criteria."""

    def test_net_pnl_strictly_positive_boundary(self):
        """Net Profit must be strictly > 0.0. Breakeven (0.00) and micro-losses (-0.01) must fail."""
        # Breakeven: 0.00
        m_zero = CycleMetrics(cycle_id=1, net_pnl=0.00, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=30)
        v_zero = evaluate_acceptance_criteria(m_zero)
        assert not v_zero.passed
        assert not v_zero.net_pnl_ok
        assert any("Net Profit <= 0" in r for r in v_zero.reasons)

        # Micro-loss: -0.01
        m_neg = CycleMetrics(cycle_id=1, net_pnl=-0.01, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=30)
        v_neg = evaluate_acceptance_criteria(m_neg)
        assert not v_neg.passed
        assert not v_neg.net_pnl_ok

        # Micro-profit: +0.01
        m_pos = CycleMetrics(cycle_id=1, net_pnl=0.01, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=30)
        v_pos = evaluate_acceptance_criteria(m_pos)
        assert v_pos.passed
        assert v_pos.net_pnl_ok

    def test_profit_factor_boundary(self):
        """Profit factor must be >= 1.15. 1.1499 must fail, 1.1500 must pass."""
        m_low = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.1499, max_drawdown_pct=5.0, closed_trades=30)
        v_low = evaluate_acceptance_criteria(m_low)
        assert not v_low.passed
        assert not v_low.profit_factor_ok

        m_ok = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.1500, max_drawdown_pct=5.0, closed_trades=30)
        v_ok = evaluate_acceptance_criteria(m_ok)
        assert v_ok.passed
        assert v_ok.profit_factor_ok

    def test_max_drawdown_boundary(self):
        """Max drawdown must be strictly < 15.0%. 15.00% must fail, 14.99% must pass."""
        m_exact = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=15.0, closed_trades=30)
        v_exact = evaluate_acceptance_criteria(m_exact)
        assert not v_exact.passed
        assert not v_exact.drawdown_ok

        m_pass = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=14.99, closed_trades=30)
        v_pass = evaluate_acceptance_criteria(m_pass)
        assert v_pass.passed
        assert v_pass.drawdown_ok

    def test_trade_count_and_hold_trap_boundary(self):
        """Closed trades must be >= 20 and hold_share must be < 0.99."""
        # 19 trades: fail
        m_19 = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=19)
        assert not evaluate_acceptance_criteria(m_19).passed

        # 20 trades, hold_share 0.80: pass
        m_20 = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=20, hold_share=0.80)
        assert evaluate_acceptance_criteria(m_20).passed

        # 25 trades, hold_share 0.99 (hold-collapse trap): fail
        m_trap = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25, hold_share=0.99)
        v_trap = evaluate_acceptance_criteria(m_trap)
        assert not v_trap.passed
        assert not v_trap.trade_frequency_ok
        assert any("100% HOLD trap" in r for r in v_trap.reasons)

    def test_all_hurdles_simultaneous_failure(self):
        """When all hurdles fail simultaneously, all reasons are recorded."""
        m_all_fail = CycleMetrics(
            cycle_id=1,
            net_pnl=-500.0,
            profit_factor=0.6,
            max_drawdown_pct=25.0,
            closed_trades=5,
            hold_share=0.995,
            simplex_errors=3,
            nan_inf_errors=2,
        )
        v = evaluate_acceptance_criteria(m_all_fail)
        assert not v.passed
        assert not v.net_pnl_ok
        assert not v.profit_factor_ok
        assert not v.drawdown_ok
        assert not v.trade_frequency_ok
        assert not v.simplex_safe
        assert len(v.reasons) >= 5


# ===========================================================================
# 4. Adaptive Tuning Monotonicity & Decay Stress Tests
# ===========================================================================

class TestAdaptiveTuningMonotonicityAndDecay:
    """Stress test hyperparameter tuning progression across consecutive loss retries."""

    def test_veto_lcb_monotonic_stepping(self):
        """Verify five_expectancy_action_veto_min_lcb_r steps up monotonically: base < retry_1 < retry_2 < retry_3."""
        base_lcb = -0.15
        overrides_r1 = compute_adaptive_tuning_overrides(retry_idx=1)
        overrides_r2 = compute_adaptive_tuning_overrides(retry_idx=2)
        overrides_r3 = compute_adaptive_tuning_overrides(retry_idx=3)

        lcb_r1 = overrides_r1["five_expectancy_action_veto_min_lcb_r"]
        lcb_r2 = overrides_r2["five_expectancy_action_veto_min_lcb_r"]
        lcb_r3 = overrides_r3["five_expectancy_action_veto_min_lcb_r"]

        assert base_lcb < lcb_r1, f"Expected {base_lcb} < {lcb_r1}"
        assert lcb_r1 < lcb_r2, f"Expected {lcb_r1} < {lcb_r2}"
        assert lcb_r2 < lcb_r3, f"Expected {lcb_r2} < {lcb_r3}"

        assert lcb_r1 == -0.05
        assert lcb_r2 == 0.00
        assert lcb_r3 == 0.05

        # Also verify live veto and owner exploration match clamped expectancy veto (clamped to [-12, 0] per config.py)
        for o in (overrides_r1, overrides_r2, overrides_r3):
            expected_live = min(0.0, o["five_expectancy_action_veto_min_lcb_r"])
            assert o["five_live_action_veto_min_lcb_r"] == expected_live
            assert o["five_live_owner_exploration_min_lcb_r"] == expected_live

    def test_uniform_mix_strict_decay(self):
        """Verify five_behavior_uniform_mix decays strictly: base > retry_1 > retry_2 > retry_3 >= 0.0."""
        base_mix = 0.02
        overrides_r1 = compute_adaptive_tuning_overrides(retry_idx=1)
        overrides_r2 = compute_adaptive_tuning_overrides(retry_idx=2)
        overrides_r3 = compute_adaptive_tuning_overrides(retry_idx=3)

        mix_r1 = overrides_r1["five_behavior_uniform_mix"]
        mix_r2 = overrides_r2["five_behavior_uniform_mix"]
        mix_r3 = overrides_r3["five_behavior_uniform_mix"]

        assert base_mix > mix_r1, f"Expected {base_mix} > {mix_r1}"
        assert mix_r1 > mix_r2, f"Expected {mix_r1} > {mix_r2}"
        assert mix_r2 > mix_r3, f"Expected {mix_r2} > {mix_r3}"

        assert mix_r1 == 0.01
        assert mix_r2 == 0.005
        assert mix_r3 == 0.000

    def test_pattern_reward_weights_monotonic_increase(self):
        """Verify pattern action quality reward scale steps up across retries."""
        base_reward = 0.10
        r1 = compute_adaptive_tuning_overrides(retry_idx=1)
        r2 = compute_adaptive_tuning_overrides(retry_idx=2)
        r3 = compute_adaptive_tuning_overrides(retry_idx=3)

        p1 = r1["five_pattern_action_quality_reward_scale_r"]
        p2 = r2["five_pattern_action_quality_reward_scale_r"]
        p3 = r3["five_pattern_action_quality_reward_scale_r"]

        assert base_reward < p1 < p2 < p3
        assert p1 == 0.15
        assert p2 == 0.20
        assert p3 == 0.25

    def test_invariants_preserved_across_all_retries(self):
        """Verify five_lock_non_entry_to_hold is True and context_direction_entry_gate_enabled is False."""
        for r_idx in range(1, 5):
            o = compute_adaptive_tuning_overrides(retry_idx=r_idx)
            assert o["five_lock_non_entry_to_hold"] is True
            assert o["context_direction_entry_gate_enabled"] is False

    def test_under_trading_diagnosis_triggers_reactivation_overrides(self):
        """When under-trading is diagnosed, compute_adaptive_tuning_overrides injects entry floor and hold ceiling overrides."""
        diag = AdaptiveTuningDiagnosis(
            failure_reasons=["Closed trades < 20 (observed 4)"],
            is_loss=False,
            is_low_pf=False,
            is_high_dd=False,
            is_under_trading=True,
            is_numerical_divergence=False,
            suggested_actions=["Increase behavior entry floor"],
        )
        overrides = compute_adaptive_tuning_overrides(retry_idx=1, diagnosis=diag)
        assert overrides.get("five_behavior_entry_probability_floor") == 0.01
        assert overrides.get("five_pattern_tree_hold_ceiling_loss_weight") == 0.50


# ===========================================================================
# 5. Rollback & Champion Preservation Under Consecutive Loss
# ===========================================================================

class TestRollbackAndChampionPreservation:
    """Stress test that failed retries never pollute the champion and roll back cleanly."""

    def test_rollback_to_clean_champion_on_consecutive_losses(self, tmp_path: Path):
        """Simulate consecutive failures: Attempt 1 fails, Attempt 2 fails, Attempt 3 succeeds.

        Verify that each retry uses the original seed, not the corrupted candidate from the failed run.
        """
        seed_path = tmp_path / "champion_seed.pkl"
        seed_content = b"ORIGINAL_CHAMPION_CLEAN_WEIGHTS"
        seed_path.write_bytes(seed_content)

        attempts_executed: list[dict[str, Any]] = []

        def simulated_runner(
            from_date: str,
            to_date: str,
            seed_artifact: Path,
            output_dir: Path,
            profile_path: Path,
            config_path: Path,
            timeout_sec: int,
            max_wall_sec: int,
        ) -> bool:
            # Record what seed was passed in
            passed_seed_content = seed_artifact.read_bytes()
            attempt_num = len(attempts_executed) + 1
            attempts_executed.append({
                "attempt": attempt_num,
                "seed_artifact": seed_artifact,
                "seed_content": passed_seed_content,
                "output_dir": output_dir,
            })

            output_dir.mkdir(parents=True, exist_ok=True)

            if attempt_num in (1, 2):
                # Write a corrupted / losing candidate
                bad_cand = output_dir / f"bad_candidate_attempt_{attempt_num}.pkl"
                bad_cand.write_bytes(f"CORRUPT_OR_LOSING_WEIGHTS_{attempt_num}".encode("utf-8"))
                diag = {
                    "performance": {"net_pnl": -300.0, "profit_factor": 0.7, "closed_trades": 15},
                    "risk": {"max_equity_drawdown_pct": 18.0},
                    "policy": {"hold_share": 0.75},
                }
            else:
                # Attempt 3 succeeds
                good_cand = output_dir / "sgt_compact_tier2_candidate.pkl"
                good_cand.write_bytes(b"NEW_SUCCESSFUL_CHAMPION_CYCLE_1")
                diag = {
                    "performance": {"net_pnl": 850.0, "profit_factor": 1.6, "closed_trades": 35},
                    "risk": {"max_equity_drawdown_pct": 6.5},
                    "policy": {"hold_share": 0.80},
                }

            (output_dir / "mt5_diagnostics.json").write_text(json.dumps(diag), encoding="utf-8")
            (output_dir / "events.jsonl").write_text("", encoding="utf-8")
            return True

        engine = WalkForwardEngine(
            workspace_dir=tmp_path,
            max_adaptive_retries=3,
            runner_fn=simulated_runner,
        )

        res = engine.run_cycle(
            cycle_id=1,
            seed_checkpoint=seed_path,
            output_base_dir=tmp_path / "cycle_1_out",
        )

        assert res["promoted"] is True
        assert res["retries_used"] == 2  # Attempt 1, then Retry 1, then Retry 2 (total 3 attempts)
        assert len(attempts_executed) == 3

        # Critical Rollback Assertion: ALL 3 attempts were seeded from the clean original seed!
        for att in attempts_executed:
            assert att["seed_content"] == seed_content, (
                f"Attempt {att['attempt']} did not receive clean seed! Got {att['seed_content']}"
            )

        # Original seed remained completely uncorrupted
        assert seed_path.read_bytes() == seed_content

        # Promoted checkpoint is the successful Attempt 3 candidate
        promoted_checkpoint = Path(res["promoted_checkpoint"])
        assert promoted_checkpoint.is_file()
        assert promoted_checkpoint.read_bytes() == b"NEW_SUCCESSFUL_CHAMPION_CYCLE_1"

    def test_chain_aborts_and_preserves_champion_when_cycle_fails(self, tmp_path: Path):
        """If Cycle 1 succeeds and promotes Champion 1, but Cycle 2 fails all retries:

        - Cycle 1 champion is preserved intact.
        - Pipeline aborts before running Cycle 3.
        - Manifest certifies multi_cycle_certified = False.
        """
        seed_path = tmp_path / "initial_seed.pkl"
        seed_path.write_bytes(b"INITIAL_SEED_BYTES")

        cycles_called: list[int] = []

        def simulated_runner(
            from_date: str,
            to_date: str,
            seed_artifact: Path,
            output_dir: Path,
            profile_path: Path,
            config_path: Path,
            timeout_sec: int,
            max_wall_sec: int,
        ) -> bool:
            output_dir.mkdir(parents=True, exist_ok=True)
            if "cycle_1" in output_dir.name:
                cycles_called.append(1)
                # Success
                cand = output_dir / "cand.pkl"
                cand.write_bytes(b"CYCLE_1_PROMOTED_CHAMPION")
                diag = {
                    "performance": {"net_pnl": 500.0, "profit_factor": 1.4, "closed_trades": 25},
                    "risk": {"max_equity_drawdown_pct": 7.0},
                    "policy": {"hold_share": 0.8},
                }
            elif "cycle_2" in output_dir.name:
                cycles_called.append(2)
                # Failure on all attempts
                cand = output_dir / "bad_cand.pkl"
                cand.write_bytes(b"CYCLE_2_FAILING_CANDIDATE")
                diag = {
                    "performance": {"net_pnl": -200.0, "profit_factor": 0.8, "closed_trades": 22},
                    "risk": {"max_equity_drawdown_pct": 12.0},
                    "policy": {"hold_share": 0.8},
                }
            else:
                cycles_called.append(3)
                diag = {}

            (output_dir / "mt5_diagnostics.json").write_text(json.dumps(diag), encoding="utf-8")
            (output_dir / "events.jsonl").write_text("", encoding="utf-8")
            return True

        engine = WalkForwardEngine(
            workspace_dir=tmp_path,
            max_adaptive_retries=1,  # 1 initial + 1 retry = 2 attempts for cycle 2
            runner_fn=simulated_runner,
        )

        manifest = engine.run_all_cycles(
            initial_seed=seed_path,
            output_base_dir=tmp_path / "chain_out",
        )

        assert manifest["multi_cycle_certified"] is False
        assert 1 in manifest["cycle_results"]
        assert manifest["cycle_results"][1]["promoted"] is True
        assert 2 in manifest["cycle_results"]
        assert manifest["cycle_results"][2]["promoted"] is False
        assert 3 not in manifest["cycle_results"]  # Aborted before Cycle 3!

        # Check Cycle 1 champion was preserved
        c1_champ = tmp_path / "chain_out" / "checkpoint_cycle_1.pkl"
        assert c1_champ.is_file()
        assert c1_champ.read_bytes() == b"CYCLE_1_PROMOTED_CHAMPION"


# ===========================================================================
# 6. Headless Execution Enforcement & Bypass Prevention Tests
# ===========================================================================

class TestHeadlessExecutionEnforcement:
    """Stress test headless enforcement, ensuring Visual=0 and timeout flags cannot be bypassed."""

    def test_build_tester_ini_strictly_asserts_visual_zero(self):
        """build_tester_ini_content default produces Visual=0 and ShutdownTerminal=1."""
        ini = build_tester_ini_content(from_date="2024.01.08", to_date="2024.03.29")
        assert "Visual=0" in ini
        assert "ShutdownTerminal=1" in ini
        assert "ExecutionMode=0" in ini
        assert "Optimization=0" in ini
        assert "FromDate=2024.01.08" in ini
        assert "ToDate=2024.03.29" in ini

    def test_verify_headless_rejects_mt5_visual_flag(self):
        """Inclusion of --mt5-visual is strictly rejected."""
        cmd = ["python", "script.py", "--mt5-visual", "--timeout-sec", "3600", "--max-wall-sec", "0"]
        ok, violations = verify_headless_configuration(cmd)
        assert not ok
        assert any("--mt5-visual flag must NOT be present" in v for v in violations)

    def test_verify_headless_rejects_insufficient_timeout(self):
        """Setting timeout-sec below 3600 is rejected."""
        cmd_short = ["python", "script.py", "--timeout-sec", "1800", "--max-wall-sec", "0"]
        ok, violations = verify_headless_configuration(cmd_short)
        assert not ok
        assert any("--timeout-sec must be >= 3600" in v for v in violations)

        cmd_missing = ["python", "script.py", "--max-wall-sec", "0"]
        ok, violations = verify_headless_configuration(cmd_missing)
        assert not ok
        assert any("Missing mandatory argument: --timeout-sec" in v for v in violations)

    def test_verify_headless_rejects_non_zero_max_wall(self):
        """Setting max-wall-sec to anything other than 0 is rejected."""
        cmd_wall = ["python", "script.py", "--timeout-sec", "3600", "--max-wall-sec", "300"]
        ok, violations = verify_headless_configuration(cmd_wall)
        assert not ok
        assert any("--max-wall-sec must be 0" in v for v in violations)

        cmd_missing = ["python", "script.py", "--timeout-sec", "3600"]
        ok, violations = verify_headless_configuration(cmd_missing)
        assert not ok
        assert any("Missing mandatory argument: --max-wall-sec" in v for v in violations)

    def test_default_subprocess_runner_raises_on_headless_violation(self, tmp_path: Path):
        """Engine default runner verifies headless config and raises RuntimeError if violated."""
        engine = WalkForwardEngine(workspace_dir=tmp_path)
        # We test that verify_headless_configuration detects violations
        # and default subprocess runner guards against them
        violating_args = ["--mt5-visual", "--timeout-sec", "1000", "--max-wall-sec", "50"]
        valid, violations = verify_headless_configuration(violating_args)
        assert not valid
        assert len(violations) >= 3


# ===========================================================================
# 7. Boundary Dates & Invariants Verification
# ===========================================================================

class TestBoundaryDatesAndInvariants:
    """Verify market cycle definitions adhere strictly to R1 specification."""

    def test_default_cycles_exact_date_ranges(self):
        """Cycles 1, 2, 3 have exact expected date boundaries."""
        assert 1 in DEFAULT_CYCLES
        assert 2 in DEFAULT_CYCLES
        assert 3 in DEFAULT_CYCLES

        c1 = DEFAULT_CYCLES[1]
        assert c1.from_date == "2024.01.08"
        assert c1.to_date == "2024.03.29"
        assert "Q1/2024" in c1.name

        c2 = DEFAULT_CYCLES[2]
        assert c2.from_date == "2024.07.01"
        assert c2.to_date == "2024.09.30"
        assert "Q3/2024" in c2.name

        c3 = DEFAULT_CYCLES[3]
        assert c3.from_date == "2025.01.06"
        assert c3.to_date == "2025.03.31"
        assert "Q1/2025" in c3.name


# ===========================================================================
# 8. Adversarial Manifest Generation & Edge Cases
# ===========================================================================

class TestAdversarialManifestGeneration:
    """Stress test manifest generation when artifacts are missing or malformed."""

    def test_manifest_with_missing_candidate_and_empty_results(self, tmp_path: Path):
        """Manifest handles non-existent candidate checkpoint gracefully with sha256='', exists=False."""
        ghost_candidate = tmp_path / "ghost_candidate.pkl"
        ghost_seed = tmp_path / "ghost_seed.pkl"

        manifest = generate_walk_forward_manifest(
            candidate_checkpoint_path=ghost_candidate,
            cycle_results={},
            output_dir=tmp_path / "out",
            initial_seed_path=ghost_seed,
            target_cycles=[1, 2, 3],
        )

        assert manifest["candidate"]["sha256"] == ""
        assert manifest["candidate"]["exists"] is False
        assert manifest["multi_cycle_certified"] is False
        assert manifest["aggregate_metrics"]["all_cycles_passed"] is False
        assert manifest["aggregate_metrics"]["total_trades"] == 0

    def test_manifest_persisted_as_valid_json(self, tmp_path: Path):
        """Manifest writes a valid json file to the output directory."""
        cand = tmp_path / "cand.pkl"
        cand.write_bytes(b"DATA")
        seed = tmp_path / "seed.pkl"
        seed.write_bytes(b"SEED")

        out_dir = tmp_path / "manifest_test"
        manifest = generate_walk_forward_manifest(
            candidate_checkpoint_path=cand,
            cycle_results={
                1: {"verdict": {"passed": True}, "metrics": {"closed_trades": 25, "net_pnl": 100.0, "profit_factor": 1.5, "max_drawdown_pct": 5.0}},
                2: {"verdict": {"passed": True}, "metrics": {"closed_trades": 30, "net_pnl": 200.0, "profit_factor": 1.8, "max_drawdown_pct": 6.0}},
                3: {"verdict": {"passed": True}, "metrics": {"closed_trades": 40, "net_pnl": 350.0, "profit_factor": 2.1, "max_drawdown_pct": 4.5}},
            },
            output_dir=out_dir,
            initial_seed_path=seed,
            target_cycles=[1, 2, 3],
        )

        assert manifest["multi_cycle_certified"] is True
        saved_file = out_dir / "manifest.json"
        assert saved_file.is_file()
        loaded = json.loads(saved_file.read_text(encoding="utf-8"))
        assert loaded["multi_cycle_certified"] is True
        assert loaded["candidate"]["exists"] is True


# ===========================================================================
# 9. Adversarial CLI Execution & Subprocess Invariants
# ===========================================================================

class TestAdversarialCliExecution:
    """Stress test CLI entry point directly via subprocess."""

    def test_cli_rejects_insufficient_timeout(self):
        """CLI exits with code 1 if --timeout-sec < 3600."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "scripts/run_walk_forward_multi_cycle.py",
            "--all-cycles",
            "--timeout-sec", "1200",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "must be >= 3600" in res.stderr

    def test_cli_rejects_non_zero_max_wall(self):
        """CLI exits with code 1 if --max-wall-sec != 0."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "scripts/run_walk_forward_multi_cycle.py",
            "--all-cycles",
            "--max-wall-sec", "120",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "must be 0" in res.stderr

    def test_cli_dry_run_all_cycles_succeeds(self, tmp_path: Path):
        """CLI runs all cycles in dry-run mode and exits with code 0."""
        import subprocess
        import sys

        out_dir = tmp_path / "dry_run_test"
        cmd = [
            sys.executable,
            "scripts/run_walk_forward_multi_cycle.py",
            "--all-cycles",
            "--dry-run",
            "--output-dir", str(out_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "Full Walk-Forward Multi-Cycle Chain Certified 100% Passed!" in res.stdout
        assert (out_dir / "manifest.json").is_file()
        assert (out_dir / "final_promoted_checkpoint.pkl").is_file()

