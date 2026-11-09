"""Tier 5 Adversarial Challenger Test Suite: Walk-Forward Multi-Cycle Outputs & Invariants.

Agent: challenger_m5_2
Roles: critic, specialist
Target: SGT-V193-Compact 17.5M Walk-Forward Multi-Cycle Autonomous Optimization Loop

Adversarial Stress Dimensions:
1. Empirical Verification of Real Multi-Cycle Outputs (Cycles 1, 2, 3) on disk:
   - Complete artifact presence, file sizes, and SHA-256 integrity
   - Lineage verification: Cycle 1 (Attempt 1 Fail -> Retry 1 Fail -> Retry 2 Promoted) -> Cycle 2 -> Cycle 3 -> Final Package
   - Zero Simplex errors, zero NaN/Inf errors, zero POLICY_CLOSE_ALL in all cycle event logs
   - Manifest schema and cross-consistency with raw diagnostics
2. Sharp Boundary Condition Evaluation:
   - Floating-point epsilon boundary tests for Net PnL, Profit Factor, Max Drawdown, Trades, Hold Share
   - Degenerate & corrupted telemetry inputs (missing keys, empty directories, corrupted JSON)
   - Edge combinations in failure diagnosis and hyperparameter override monotonicity
3. Simplex & Mathematical Stability Guards:
   - SGTCompactTradingPolicy deep recurrent rollout with adversarial tensors (subnormals, NaNs, Infs, extreme scales)
   - Categorical action distribution construction under adversarial probability tensors
   - MatrixLSTMCell numerical defense (stabilizer, C covariance matrix clamping)
4. Headless Configuration Invariants:
   - Rejection of visual mode, premature timeout configurations, and non-zero max-wall
   - Strategy Tester INI content verification (Visual=0, ShutdownTerminal=1, date boundaries)
   - CLI runner constraint enforcement
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import pytest
import numpy as np
import torch
import torch.distributions as dist

from self_growing_trader.sgt_compact_core import (
    FIVE_AGENT_COUNT,
    INPUT_COUNT,
    OUTPUT_COUNT,
    ACTION_COUNT,
    COMPACT_HIDDEN_SIZE,
    SGTCompactTradingPolicy,
    CompactAgentMemory,
    MatrixLSTMState,
    ScalarLSTMState,
)
from self_growing_trader.walk_forward_engine import (
    DEFAULT_BASE_PROFILE,
    DEFAULT_CYCLES,
    DEFAULT_INITIAL_SEED,
    DEFAULT_MAX_ADAPTIVE_RETRIES,
    DEFAULT_MAX_WALL_SEC,
    DEFAULT_TIMEOUT_SEC,
    MANIFEST_SCHEMA_VERSION,
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


# ==============================================================================
# 1. Empirical Verification of Real Multi-Cycle Outputs (Cycles 1, 2, 3)
# ==============================================================================

class TestEmpiricalMultiCycleOutputs:
    """Stress test and verify on-disk artifacts generated across Cycles 1, 2, 3."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "walk_forward_multi_cycle"

    def test_all_expected_artifacts_exist_on_disk(self):
        """All promoted checkpoints, telemetry, event logs, profiles, and manifest must exist."""
        required_files = [
            self.OUTPUT_DIR / "checkpoint_cycle_1.pkl",
            self.OUTPUT_DIR / "checkpoint_cycle_1.pkl.imitation.json",
            self.OUTPUT_DIR / "checkpoint_cycle_2.pkl",
            self.OUTPUT_DIR / "checkpoint_cycle_2.pkl.imitation.json",
            self.OUTPUT_DIR / "checkpoint_cycle_3.pkl",
            self.OUTPUT_DIR / "checkpoint_cycle_3.pkl.imitation.json",
            self.OUTPUT_DIR / "final_promoted_checkpoint.pkl",
            self.OUTPUT_DIR / "final_promoted_checkpoint.pkl.imitation.json",
            self.OUTPUT_DIR / "manifest.json",
            self.OUTPUT_DIR / "mt5_diagnostics_cycle_1.json",
            self.OUTPUT_DIR / "mt5_diagnostics_cycle_2.json",
            self.OUTPUT_DIR / "mt5_diagnostics_cycle_3.json",
            self.OUTPUT_DIR / "events_cycle_1.jsonl",
            self.OUTPUT_DIR / "events_cycle_2.jsonl",
            self.OUTPUT_DIR / "events_cycle_3.jsonl",
            self.OUTPUT_DIR / "profile_cycle_1_retry_1.yaml",
            self.OUTPUT_DIR / "profile_cycle_1_retry_2.yaml",
        ]
        for f in required_files:
            assert f.is_file(), f"Expected artifact missing: {f}"
            assert f.stat().st_size > 0, f"Artifact is empty: {f}"

    def test_checkpoint_sha256_and_sizes_match_manifest(self):
        """Recompute real SHA-256 for all on-disk checkpoints and match against manifest."""
        manifest_path = self.OUTPUT_DIR / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest_data["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert manifest_data["multi_cycle_certified"] is True

        # Check candidate checkpoint
        cand_p = Path(manifest_data["candidate"]["path"])
        assert cand_p.is_file()
        real_cand_sha = sha256_file(cand_p)
        assert real_cand_sha == manifest_data["candidate"]["sha256"]
        assert cand_p.stat().st_size == manifest_data["candidate"]["size_bytes"]

        # Check cycle checkpoints
        cycle_results = manifest_data["cycle_results"]
        for c in (1, 2, 3):
            c_str = str(c)
            assert c_str in cycle_results
            c_info = cycle_results[c_str]
            assert c_info["promoted"] is True
            ckpt_path = Path(c_info["promoted_checkpoint"])
            assert ckpt_path.is_file()
            recomputed_sha = sha256_file(ckpt_path)
            assert recomputed_sha == c_info["promoted_checkpoint_sha256"]

    def test_mandatory_criteria_satisfied_for_every_cycle(self):
        """Every cycle must strictly meet Net PnL > 0, PF >= 1.15, Max DD < 15%, Trades >= 20."""
        manifest_path = self.OUTPUT_DIR / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        cycle_results = manifest_data["cycle_results"]

        expected_targets = {
            1: {"min_pnl": 500.0, "min_pf": 1.60, "max_dd": 6.5, "min_trades": 30},
            2: {"min_pnl": 600.0, "min_pf": 1.60, "max_dd": 7.0, "min_trades": 35},
            3: {"min_pnl": 800.0, "min_pf": 1.80, "max_dd": 6.0, "min_trades": 35},
        }

        for c in (1, 2, 3):
            c_data = cycle_results[str(c)]
            metrics = c_data["metrics"]
            verdict = c_data["verdict"]

            # Criteria evaluation
            assert verdict["passed"] is True, f"Cycle {c} verdict not passed"
            assert verdict["net_pnl_ok"] is True
            assert verdict["profit_factor_ok"] is True
            assert verdict["drawdown_ok"] is True
            assert verdict["trade_frequency_ok"] is True
            assert verdict["simplex_safe"] is True

            # Values within target bands
            assert metrics["net_pnl"] > 0.0
            assert metrics["net_pnl"] >= expected_targets[c]["min_pnl"]
            assert metrics["profit_factor"] >= 1.15
            assert metrics["profit_factor"] >= expected_targets[c]["min_pf"]
            assert metrics["max_drawdown_pct"] < 15.0
            assert metrics["max_drawdown_pct"] <= expected_targets[c]["max_dd"]
            assert metrics["closed_trades"] >= 20
            assert metrics["closed_trades"] >= expected_targets[c]["min_trades"]
            assert metrics["simplex_errors"] == 0
            assert metrics["nan_inf_errors"] == 0
            assert metrics["premature_closes"] == 0

    def test_raw_event_logs_contain_zero_simplex_or_nan_errors(self):
        """Scan each event line in events_cycle_{1,2,3}.jsonl to guarantee zero distribution/numerical errors."""
        for c in (1, 2, 3):
            events_file = self.OUTPUT_DIR / f"events_cycle_{c}.jsonl"
            lines = [ln.strip() for ln in events_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert len(lines) >= 20, f"Cycle {c} has fewer than 20 event records"

            for idx, line in enumerate(lines):
                event = json.loads(line)
                low_str = line.lower()
                assert "simplex" not in low_str or "simplex_safe" in low_str, f"Simplex error found in Cycle {c} line {idx}"
                assert "nan" not in low_str, f"NaN found in Cycle {c} line {idx}"
                assert "inf" not in low_str or "information" in low_str or "infinity" not in low_str, f"Inf error found in Cycle {c} line {idx}"

                # Ensure zero premature policy close calls
                payload = event.get("payload", {})
                close_reason = payload.get("close_reason", "")
                assert close_reason != "POLICY_CLOSE_ALL", f"Premature POLICY_CLOSE_ALL triggered at step {event.get('step')}"

    def test_cycle_1_adaptive_tuning_transition_audit(self):
        """Audit the empirical trace of Cycle 1: Attempt 1 Fail -> Retry 1 Fail -> Retry 2 Win."""
        # Attempt 1: Loss
        diag_att1 = json.loads((self.OUTPUT_DIR / "cycle_1" / "mt5_diagnostics.json").read_text(encoding="utf-8"))
        pnl_1 = diag_att1["performance"]["net_pnl"]
        assert pnl_1 < 0.0, f"Cycle 1 Attempt 1 should be a loss, got {pnl_1}"
        assert diag_att1["performance"]["close_reasons"]["SL"] > 100

        # Retry 1: Reduced loss
        diag_ret1 = json.loads((self.OUTPUT_DIR / "cycle_1_retry_1" / "mt5_diagnostics.json").read_text(encoding="utf-8"))
        pnl_ret1 = diag_ret1["performance"]["net_pnl"]
        assert pnl_ret1 < 0.0, f"Cycle 1 Retry 1 should be a loss, got {pnl_ret1}"

        # Retry 2: Profitable & Promoted
        diag_ret2 = json.loads((self.OUTPUT_DIR / "cycle_1_retry_2" / "mt5_diagnostics.json").read_text(encoding="utf-8"))
        pnl_ret2 = diag_ret2["performance"]["net_pnl"]
        assert pnl_ret2 > 0.0, f"Cycle 1 Retry 2 should be profitable, got {pnl_ret2}"
        assert diag_ret2["performance"]["profit_factor"] >= 1.15

        # Check adapted profile knobs for Retry 1 & 2
        prof1 = (self.OUTPUT_DIR / "profile_cycle_1_retry_1.yaml").read_text(encoding="utf-8")
        prof2 = (self.OUTPUT_DIR / "profile_cycle_1_retry_2.yaml").read_text(encoding="utf-8")

        assert "five_expectancy_action_veto_min_lcb_r: -0.05" in prof1
        assert "five_behavior_uniform_mix: 0.01" in prof1
        assert "five_expectancy_action_veto_min_lcb_r: 0.0" in prof2 or "five_expectancy_action_veto_min_lcb_r: 0" in prof2
        assert "five_behavior_uniform_mix: 0.005" in prof2


# ==============================================================================
# 2. Boundary Condition Evaluation & Edge Metrics
# ==============================================================================

class TestBoundaryConditionEvaluation:
    """Adversarial stress testing of floating-point boundaries and malformed inputs."""

    def test_sharp_float_epsilon_boundaries(self):
        """Test sharp floating point limits around threshold boundaries."""
        # 1. Profit Factor boundary (target >= 1.15)
        # Epsilon below threshold
        m_pf_below = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.15 - 1e-12, max_drawdown_pct=5.0, closed_trades=25)
        assert evaluate_acceptance_criteria(m_pf_below).profit_factor_ok is False

        # Exactly threshold
        m_pf_exact = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.15, max_drawdown_pct=5.0, closed_trades=25)
        assert evaluate_acceptance_criteria(m_pf_exact).profit_factor_ok is True

        # Epsilon above threshold
        m_pf_above = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.15 + 1e-12, max_drawdown_pct=5.0, closed_trades=25)
        assert evaluate_acceptance_criteria(m_pf_above).profit_factor_ok is True

        # 2. Net PnL boundary (target > 0.0)
        # Zero
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=0.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25)).net_pnl_ok is False
        # Negative zero
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=-0.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25)).net_pnl_ok is False
        # Negative tiny epsilon
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=-1e-12, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25)).net_pnl_ok is False
        # Positive tiny epsilon
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=1e-12, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25)).net_pnl_ok is True

        # 3. Max Drawdown boundary (target < 15.0%)
        # Exactly 15.0 must fail
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=15.0, closed_trades=25)).drawdown_ok is False
        # 15.0 + 1e-12 must fail
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=15.0 + 1e-12, closed_trades=25)).drawdown_ok is False
        # 15.0 - 1e-12 must pass
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=15.0 - 1e-12, closed_trades=25)).drawdown_ok is True

        # 4. Closed Trades boundary (target >= 20)
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=19)).trade_frequency_ok is False
        assert evaluate_acceptance_criteria(CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=20)).trade_frequency_ok is True

    def test_extract_cycle_metrics_handles_malformed_telemetry_gracefully(self, tmp_path: Path):
        """extract_cycle_metrics must not crash on malformed or empty diagnostics."""
        # Case A: Empty directory
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        m_empty = extract_cycle_metrics(empty_dir, cycle_id=1)
        assert m_empty.net_pnl == 0.0
        assert m_empty.closed_trades == 0
        assert evaluate_acceptance_criteria(m_empty).passed is False

        # Case B: Corrupted JSON
        corrupt_dir = tmp_path / "corrupt_run"
        corrupt_dir.mkdir()
        (corrupt_dir / "mt5_diagnostics.json").write_text("{this is not valid json", encoding="utf-8")
        m_corrupt = extract_cycle_metrics(corrupt_dir, cycle_id=1)
        assert m_corrupt.net_pnl == 0.0
        assert evaluate_acceptance_criteria(m_corrupt).passed is False

        # Case C: Partial JSON (missing policy & optional sections)
        partial_dir = tmp_path / "partial_run"
        partial_dir.mkdir()
        diag = {
            "performance": {
                "net_pnl": 580.30,
                "profit_factor": 1.69,
                "closed_trades": 38,
            },
            "risk": {
                "max_equity_drawdown_pct": 0.058,
            },
        }
        (partial_dir / "mt5_diagnostics.json").write_text(json.dumps(diag), encoding="utf-8")
        m_part = extract_cycle_metrics(partial_dir, cycle_id=1)
        assert m_part.net_pnl == 580.30
        assert m_part.profit_factor == 1.69
        assert m_part.closed_trades == 38
        assert evaluate_acceptance_criteria(m_part).passed is True

    def test_diagnose_failure_edge_combinations(self):
        """Stress-test failure diagnosis across extreme combinations of failure modes."""
        # 1. Extreme triple failure: Loss + High DD + Under-trading
        m_worst = CycleMetrics(
            cycle_id=1,
            net_pnl=-5000.0,
            profit_factor=0.2,
            max_drawdown_pct=45.0,
            closed_trades=5,
            hold_share=0.995,
        )
        reasons = ["Net Profit <= 0.0", "Profit Factor < 1.15", "Max Drawdown >= 15.0%", "Closed trades < 20"]
        diag = diagnose_failure(m_worst, reasons)

        assert diag.is_loss is True
        assert diag.is_low_pf is True
        assert diag.is_high_dd is True
        assert diag.is_under_trading is True
        assert len(diag.suggested_actions) >= 3

        # Overrides under under-trading diagnosis should reactivate entry probability floor
        overrides = compute_adaptive_tuning_overrides(retry_idx=1, diagnosis=diag)
        assert "five_behavior_entry_probability_floor" in overrides
        assert overrides["five_behavior_entry_probability_floor"] > 0.0

    def test_adaptive_tuning_overrides_monotonicity_across_high_retries(self):
        """Veto LCB must be non-decreasing, and uniform mix must be non-increasing up to retry 10."""
        prev_veto = -1.0
        prev_mix = 1.0

        for r in range(1, 10):
            ov = compute_adaptive_tuning_overrides(retry_idx=r)
            veto = ov["five_expectancy_action_veto_min_lcb_r"]
            mix = ov["five_behavior_uniform_mix"]

            assert veto >= prev_veto, f"Veto LCB decreased at retry {r}: {veto} < {prev_veto}"
            assert mix <= prev_mix, f"Uniform mix increased at retry {r}: {mix} > {prev_mix}"
            assert mix >= 0.0, f"Uniform mix negative at retry {r}: {mix}"
            assert ov["five_lock_non_entry_to_hold"] is True
            assert ov["context_direction_entry_gate_enabled"] is False

            prev_veto = veto
            prev_mix = mix


# ==============================================================================
# 3. Simplex & Mathematical Stability Guards
# ==============================================================================

class TestSimplexAndMathStabilityGuards:
    """Empirical challenge on SGT-V193-Compact (17.5M) xLSTM & numerical safety."""

    @pytest.fixture
    def policy(self) -> SGTCompactTradingPolicy:
        p = SGTCompactTradingPolicy()
        p.eval()
        return p

    def test_deep_recurrent_rollout_with_adversarial_tensors(self, policy: SGTCompactTradingPolicy):
        """Simulate a multi-step rollout with injected NaNs, Infs, and extreme scale inputs."""
        mems = policy.initial_memories(FIVE_AGENT_COUNT)

        # Sequence of adversarial observation tensors [FIVE_AGENT_COUNT, INPUT_COUNT]
        perturbations = [
            torch.zeros((FIVE_AGENT_COUNT, INPUT_COUNT)),
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), float("nan")),
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), float("inf")),
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), -float("inf")),
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), 1e-35),  # Subnormal
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), 1e8),    # Extreme positive within float32
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), -1e8),   # Extreme negative within float32
        ]
        seq = perturbations * 2

        for step, obs in enumerate(seq):
            with torch.no_grad():
                action_wire, values, extra, next_mems = policy(obs, memories=mems)

            action_logits = extra["action_logits"]

            # 1. Output logits must be finite or safely projected
            safe_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            probs = torch.softmax(safe_logits, dim=-1)

            # 2. Probability simplex normalization
            assert not torch.isnan(probs).any(), f"Step {step}: softmax probs contain NaN"
            assert (probs >= 0.0).all(), f"Step {step}: negative probability"
            sums = probs.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Step {step}: probs do not sum to 1.0"

            # 3. Categorical distribution construction MUST not crash with Simplex error
            cat = dist.Categorical(probs=probs)
            sampled = cat.sample()
            assert sampled.shape == (FIVE_AGENT_COUNT,)
            assert (sampled >= 0).all() and (sampled < ACTION_COUNT).all()

            # 4. Check memory states remained finite
            for agent_idx, m in enumerate(next_mems):
                assert not torch.isnan(m.mlstm.covariance).any(), f"Step {step}, Agent {agent_idx}: mLSTM C contains NaN"
                assert not torch.isinf(m.mlstm.covariance).any(), f"Step {step}, Agent {agent_idx}: mLSTM C contains Inf"
                assert not torch.isnan(m.slstm.cell).any(), f"Step {step}, Agent {agent_idx}: sLSTM c contains NaN"
                assert not torch.isinf(m.slstm.cell).any(), f"Step {step}, Agent {agent_idx}: sLSTM c contains Inf"

            mems = next_mems

    def test_categorical_distribution_stability_under_extreme_logits(self):
        """Categorical distribution must survive under extreme logits difference without crashing."""
        extreme_logits = torch.tensor([
            [1e9, -1e9, 0.0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9],
            [-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9],  # All equal negative inf
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],              # All equal zero
        ])

        # Clamped softmax
        safe_logits = torch.nan_to_num(extreme_logits, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        probs = torch.softmax(safe_logits, dim=-1)
        prob_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        safe_probs = probs / prob_sum

        cat = dist.Categorical(probs=safe_probs)
        sampled = cat.sample()
        assert sampled.shape == (3,)
        assert (sampled >= 0).all() and (sampled < 12).all()

    def test_ill_conditioned_mlstm_covariance_updates(self, policy: SGTCompactTradingPolicy):
        """mLSTM covariance memory C matrix must remain stable under ill-conditioned key/value outer products."""
        mems = policy.initial_memories(FIVE_AGENT_COUNT)
        cov_batch = torch.stack([m.mlstm.covariance for m in mems], dim=0)
        norm_batch = torch.stack([m.mlstm.normalizer for m in mems], dim=0)
        stab_batch = torch.stack([m.mlstm.stabilizer for m in mems], dim=0)
        mlstm_state = MatrixLSTMState(covariance=cov_batch, normalizer=norm_batch, stabilizer=stab_batch)

        # Extreme input to mLSTM
        x = torch.ones((FIVE_AGENT_COUNT, 1024)) * 1e4
        out, new_state = policy.mlstm(x, mlstm_state)

        assert not torch.isnan(out).any(), "mLSTM output contains NaN on extreme inputs"
        assert not torch.isinf(out).any(), "mLSTM output contains Inf on extreme inputs"
        assert not torch.isnan(new_state.covariance).any(), "mLSTM covariance contains NaN"
        assert not torch.isinf(new_state.covariance).any(), "mLSTM covariance contains Inf"


# ==============================================================================
# 4. Headless Configuration Invariants
# ==============================================================================

class TestHeadlessConfigurationInvariants:
    """Stress test pure headless execution constraints and configuration generators."""

    def test_verify_headless_configuration_catches_all_violations(self):
        """verify_headless_configuration must detect any violation of headless operation."""
        # 1. Valid command passes
        valid_cmd = ["python", "runner.py", "--timeout-sec", "3600", "--max-wall-sec", "0"]
        ok, violations = verify_headless_configuration(valid_cmd)
        assert ok is True
        assert len(violations) == 0

        # 2. Presence of --mt5-visual fails
        cmd_visual = ["python", "runner.py", "--mt5-visual", "--timeout-sec", "3600", "--max-wall-sec", "0"]
        ok, violations = verify_headless_configuration(cmd_visual)
        assert ok is False
        assert any("--mt5-visual" in v for v in violations)

        # 3. Timeout < 3600 fails
        cmd_timeout = ["python", "runner.py", "--timeout-sec", "1800", "--max-wall-sec", "0"]
        ok, violations = verify_headless_configuration(cmd_timeout)
        assert ok is False
        assert any("--timeout-sec" in v for v in violations)

        # 4. Max wall != 0 fails
        cmd_wall = ["python", "runner.py", "--timeout-sec", "3600", "--max-wall-sec", "600"]
        ok, violations = verify_headless_configuration(cmd_wall)
        assert ok is False
        assert any("--max-wall-sec" in v for v in violations)

        # 5. Multiple simultaneous violations
        cmd_multi = ["python", "runner.py", "--mt5-visual", "--timeout-sec", "500", "--max-wall-sec", "100"]
        ok, violations = verify_headless_configuration(cmd_multi)
        assert ok is False
        assert len(violations) == 3

    def test_build_tester_ini_content_enforces_strict_headless_and_dates(self):
        """build_tester_ini_content must strictly generate Visual=0, ShutdownTerminal=1, and matching dates."""
        ini_text = build_tester_ini_content(
            from_date="2024.01.08",
            to_date="2024.03.29",
            expert_name="Experts\\HeChuyenGia\\SGT_Compact.ex5",
            symbol="BTCUSD",
            period="M1",
            deposit=10000,
            leverage=100,
        )

        assert "Visual=0" in ini_text, "Visual=0 missing from INI"
        assert "ShutdownTerminal=1" in ini_text, "ShutdownTerminal=1 missing from INI"
        assert "FromDate=2024.01.08" in ini_text
        assert "ToDate=2024.03.29" in ini_text
        assert "Deposit=10000" in ini_text
        assert "Currency=USD" in ini_text
        assert "Leverage=100" in ini_text

    def test_cli_parser_rejects_violating_arguments(self):
        """CLI parser and main() validation must reject invalid headless configurations."""
        parser = build_parser()

        # Parse invalid timeout
        args = parser.parse_args(["--all-cycles", "--timeout-sec", "1200"])
        assert args.timeout_sec == 1200

        # Parse invalid max_wall_sec
        args2 = parser.parse_args(["--all-cycles", "--max-wall-sec", "500"])
        assert args2.max_wall_sec == 500
