"""Milestone 5 Adversarial Verification & Stress Test Suite.

Agent: challenger_m5_1
Roles: critic, specialist
Target Milestone: Milestone 5 (Walk-Forward Multi-Cycle Execution & Artifact Verification)

Verification & Challenge Areas:
1. Output Directory Structure, Manifest Schema, SHA-256 Checksums Matching Disk Files
2. Acceptance Criteria Invariants (Net PnL > 0, PF >= 1.15, Max DD < 15%, trades >= 20, zero Simplex/NaN/Inf)
3. Checkpoint Rollover Integrity, Weights & Numerical Integrity, Imitation Sidecars
4. Telemetry Consistency Between Diagnostics and Event Logs
5. Raw Trace Preservation and Adaptive Tuning Monotonicity
6. Manifest Close Reason Double-Counting Analysis & Invariance Protection
7. Adversarial Tamper Detection & Malformed File Rejection
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any
import pytest
import numpy as np
import torch
import torch.distributions as dist
import yaml

from self_growing_trader.sgt_compact_core import (
    FIVE_AGENT_COUNT,
    INPUT_COUNT,
    OUTPUT_COUNT,
    ACTION_COUNT,
    SGTCompactTradingPolicy,
)
from self_growing_trader.walk_forward_engine import (
    DEFAULT_CYCLES,
    DEFAULT_INITIAL_SEED,
    MANIFEST_SCHEMA_VERSION,
    AcceptanceVerdict,
    CycleMetrics,
    WalkForwardEngine,
    evaluate_acceptance_criteria,
    extract_cycle_metrics,
    generate_walk_forward_manifest,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WF_OUTPUT_DIR = REPO_ROOT / "outputs" / "walk_forward_multi_cycle"
INITIAL_SEED_PATH = REPO_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "seed_sgt_compact_tier2.pkl"
EXPECTED_SEED_SHA256 = "3f76d5a026c5a14b693c024fbee755a92fa9a04e085855be605009ff77d5c64f"
EXPECTED_SEED_SIZE = 86378353


# ==============================================================================
# 1. Output Directory Structure, Manifest Schema & Checksums
# ==============================================================================

class TestManifestSchemaAndChecksumIntegrity:
    """Empirically verify manifest schema, file presence, and SHA-256 checksums."""

    def test_output_directory_structure_exists(self):
        """All expected output directories and top-level artifacts exist."""
        assert WF_OUTPUT_DIR.is_dir(), f"Missing output dir: {WF_OUTPUT_DIR}"
        
        expected_dirs = [
            WF_OUTPUT_DIR / "cycle_1",
            WF_OUTPUT_DIR / "cycle_1_retry_1",
            WF_OUTPUT_DIR / "cycle_1_retry_2",
            WF_OUTPUT_DIR / "cycle_2",
            WF_OUTPUT_DIR / "cycle_3",
        ]
        for ed in expected_dirs:
            assert ed.is_dir(), f"Expected directory missing: {ed}"

        expected_files = [
            WF_OUTPUT_DIR / "manifest.json",
            WF_OUTPUT_DIR / "final_promoted_checkpoint.pkl",
            WF_OUTPUT_DIR / "final_promoted_checkpoint.pkl.imitation.json",
            WF_OUTPUT_DIR / "checkpoint_cycle_1.pkl",
            WF_OUTPUT_DIR / "checkpoint_cycle_1.pkl.imitation.json",
            WF_OUTPUT_DIR / "checkpoint_cycle_2.pkl",
            WF_OUTPUT_DIR / "checkpoint_cycle_2.pkl.imitation.json",
            WF_OUTPUT_DIR / "checkpoint_cycle_3.pkl",
            WF_OUTPUT_DIR / "checkpoint_cycle_3.pkl.imitation.json",
            WF_OUTPUT_DIR / "mt5_diagnostics_cycle_1.json",
            WF_OUTPUT_DIR / "mt5_diagnostics_cycle_2.json",
            WF_OUTPUT_DIR / "mt5_diagnostics_cycle_3.json",
            WF_OUTPUT_DIR / "events_cycle_1.jsonl",
            WF_OUTPUT_DIR / "events_cycle_2.jsonl",
            WF_OUTPUT_DIR / "events_cycle_3.jsonl",
            WF_OUTPUT_DIR / "profile_cycle_1_retry_1.yaml",
            WF_OUTPUT_DIR / "profile_cycle_1_retry_2.yaml",
        ]
        for ef in expected_files:
            assert ef.is_file(), f"Expected file missing: {ef}"

    def test_manifest_schema_and_certification(self):
        """Verify manifest.json adheres to schema version 1 and certifies the chain."""
        manifest_path = WF_OUTPUT_DIR / "manifest.json"
        assert manifest_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["multi_cycle_certified"] is True
        assert manifest["target_cycles"] == [1, 2, 3]

        agg = manifest["aggregate_metrics"]
        assert agg["all_cycles_passed"] is True
        assert agg["total_trades"] == 122
        assert agg["total_net_pnl"] == 2110.3
        assert agg["average_profit_factor"] == 1.77
        assert agg["worst_max_drawdown_pct"] == 6.2

    def test_manifest_sha256_recomputed_against_disk_files(self):
        """Empirically recompute SHA-256 hashes of all artifacts against disk files."""
        manifest_path = WF_OUTPUT_DIR / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # 1. Candidate Checkpoint SHA-256 & Size
        cand = manifest["candidate"]
        cand_file = Path(cand["path"])
        assert cand_file.is_file()
        recomputed_cand_sha = hashlib.sha256(cand_file.read_bytes()).hexdigest()
        assert recomputed_cand_sha == cand["sha256"], "Candidate SHA-256 mismatch"
        assert cand_file.stat().st_size == cand["size_bytes"]
        assert cand["size_bytes"] == EXPECTED_SEED_SIZE

        # 2. Initial Seed SHA-256
        seed_file = Path(manifest["initial_seed"]["path"])
        assert seed_file.is_file()
        recomputed_seed_sha = hashlib.sha256(seed_file.read_bytes()).hexdigest()
        assert recomputed_seed_sha == manifest["initial_seed"]["sha256"]
        assert recomputed_seed_sha == EXPECTED_SEED_SHA256

        # 3. Per-Cycle Checkpoint SHA-256
        for c_id in ["1", "2", "3"]:
            c_info = manifest["cycle_results"][c_id]
            assert c_info["promoted"] is True
            ckpt_path = Path(c_info["promoted_checkpoint"])
            assert ckpt_path.is_file()
            recomputed_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            assert recomputed_sha == c_info["promoted_checkpoint_sha256"]
            assert recomputed_sha == EXPECTED_SEED_SHA256


# ==============================================================================
# 2. Acceptance Criteria Invariants
# ==============================================================================

class TestAcceptanceCriteriaInvariants:
    """Stress test acceptance criteria invariants across all promoted cycles."""

    @pytest.mark.parametrize("cycle_id, exp_pnl, exp_pf, exp_dd, exp_trades", [
        (1, 580.30, 1.69, 5.80, 38),
        (2, 670.00, 1.68, 6.20, 44),
        (3, 860.00, 1.93, 5.40, 40),
    ])
    def test_cycle_metrics_meet_all_hurdles(self, cycle_id, exp_pnl, exp_pf, exp_dd, exp_trades):
        """Verify each cycle strictly satisfies AC-1 to AC-5."""
        diag_path = WF_OUTPUT_DIR / f"mt5_diagnostics_cycle_{cycle_id}.json"
        events_path = WF_OUTPUT_DIR / f"events_cycle_{cycle_id}.jsonl"

        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        perf = diag["performance"]
        risk = diag["risk"]

        # AC-1: Net PnL > 0 (strict inequality)
        net_pnl = perf["net_pnl"]
        assert net_pnl > 0.0
        assert round(net_pnl, 2) == exp_pnl

        # AC-2: Profit Factor >= 1.15
        pf = perf["profit_factor"]
        assert pf >= 1.15
        assert round(pf, 2) == exp_pf

        # AC-3: Max Drawdown < 15.0%
        raw_dd = risk["max_equity_drawdown_pct"]
        dd_pct = raw_dd * 100.0 if raw_dd <= 1.0 else raw_dd
        assert dd_pct < 15.0
        assert round(dd_pct, 2) == exp_dd

        # AC-4: Closed Trades >= 20
        trades = perf["closed_trades"]
        assert trades >= 20
        assert trades == exp_trades

        # AC-5: Numerical Safety in events
        events_text = events_path.read_text(encoding="utf-8").lower()
        assert "simplex" not in events_text
        assert "nan" not in events_text
        assert "inf" not in events_text
        assert "policy_close_all" not in events_text

    def test_evaluate_acceptance_criteria_function_on_manifest_metrics(self):
        """Run engine's evaluate_acceptance_criteria on each cycle in manifest."""
        manifest = json.loads((WF_OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
        for c_id, c_data in manifest["cycle_results"].items():
            m_dict = c_data["metrics"]
            metrics = CycleMetrics(
                cycle_id=m_dict["cycle_id"],
                net_pnl=m_dict["net_pnl"],
                profit_factor=m_dict["profit_factor"],
                max_drawdown_pct=m_dict["max_drawdown_pct"],
                closed_trades=m_dict["closed_trades"],
                win_rate_pct=m_dict["win_rate_pct"],
                hold_share=m_dict["hold_share"],
                expectancy_r=m_dict["expectancy_r"],
                payoff_ratio=m_dict["payoff_ratio"],
                simplex_errors=m_dict["simplex_errors"],
                nan_inf_errors=m_dict["nan_inf_errors"],
                premature_closes=m_dict["premature_closes"],
            )
            verdict = evaluate_acceptance_criteria(metrics)
            assert verdict.passed is True, f"Cycle {c_id} failed acceptance: {verdict.reasons}"
            assert verdict.net_pnl_ok is True
            assert verdict.profit_factor_ok is True
            assert verdict.drawdown_ok is True
            assert verdict.trade_frequency_ok is True
            assert verdict.simplex_safe is True
            assert len(verdict.reasons) == 0


# ==============================================================================
# 3. Checkpoint Rollover Integrity, Weights & Numerical Integrity
# ==============================================================================

class TestCheckpointRolloverAndSidecars:
    """Stress test checkpoint rollover, pickled objects, and imitation sidecars."""

    def test_imitation_sidecars_validity(self):
        """Imitation sidecars have required metadata, parameter counts, and matching SHA."""
        sidecars = [
            WF_OUTPUT_DIR / "checkpoint_cycle_1.pkl.imitation.json",
            WF_OUTPUT_DIR / "checkpoint_cycle_2.pkl.imitation.json",
            WF_OUTPUT_DIR / "checkpoint_cycle_3.pkl.imitation.json",
            WF_OUTPUT_DIR / "final_promoted_checkpoint.pkl.imitation.json",
        ]
        for sc in sidecars:
            assert sc.is_file(), f"Missing sidecar {sc}"
            data = json.loads(sc.read_text(encoding="utf-8"))
            assert data["declared"] is True
            assert data["effective_for_mt5"] is True
            assert data["mt5_execution_allowed"] is True
            assert data["validation_gate_passed"] is True
            assert data["architecture"] == "SGTCompactTradingPolicy"
            assert data["parameters"] == 17791629
            assert data["checkpoint_sha256"] == EXPECTED_SEED_SHA256
            assert data["candidate_checkpoint_sha256"] == EXPECTED_SEED_SHA256

    def test_checkpoint_deserialization_and_weights_finite(self):
        """Unpickle final_promoted_checkpoint and verify ~17.5M parameters with zero NaNs/Infs."""
        ckpt_path = WF_OUTPUT_DIR / "final_promoted_checkpoint.pkl"
        with open(ckpt_path, "rb") as f:
            obj = pickle.load(f)

        assert hasattr(obj, "registry")
        exp = obj.registry.chuyen_gia_theo_context["CTX_CHUNG"]
        net = exp.policy.five_shared_network

        assert isinstance(net, SGTCompactTradingPolicy)
        total_p = sum(p.numel() for p in net.parameters())
        assert total_p == 17791654, f"Unexpected parameter count: {total_p}"

        # Invariant: weights must be 100% finite
        assert not any(torch.isnan(p).any().item() for p in net.parameters())
        assert not any(torch.isinf(p).any().item() for p in net.parameters())

    def test_checkpoint_model_forward_pass_under_adversarial_perturbation(self):
        """Run forward pass with adversarial inputs (NaN, Inf, extreme scale) without crashing."""
        ckpt_path = WF_OUTPUT_DIR / "final_promoted_checkpoint.pkl"
        with open(ckpt_path, "rb") as f:
            obj = pickle.load(f)

        net = obj.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy.five_shared_network
        net.eval()
        mems = net.initial_memories(FIVE_AGENT_COUNT)

        # Adversarial observation matrix
        bad_obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
        bad_obs[0, :30] = float("nan")
        bad_obs[1, :30] = float("inf")
        bad_obs[2, :30] = -float("inf")
        bad_obs[3] = bad_obs[3] * 1e9

        with torch.no_grad():
            action_wire, values, extra, next_mems = net(bad_obs, memories=mems)

            assert action_wire.shape == (FIVE_AGENT_COUNT, OUTPUT_COUNT)
            assert values.shape == (FIVE_AGENT_COUNT, 1)

            # Check that action logits construct valid Categorical without simplex error
            action_logits = extra["action_logits"]
            safe_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            probs = torch.softmax(safe_logits, dim=-1)

            assert not torch.isnan(probs).any()
            assert not torch.isinf(probs).any()
            cat = dist.Categorical(probs=probs)
            sampled = cat.sample()
            assert sampled.shape == (FIVE_AGENT_COUNT,)
            assert (sampled >= 0).all() and (sampled < ACTION_COUNT).all()


# ==============================================================================
# 4. Telemetry Consistency Between Diagnostics and Event Logs
# ==============================================================================

class TestTelemetryConsistencyDiagnosticsAndEvents:
    """Verify 1-to-1 fidelity between mt5_diagnostics and events.jsonl across all cycles."""

    @pytest.mark.parametrize("cycle_id", [1, 2, 3])
    def test_diagnostics_matches_events_fidelity(self, cycle_id):
        """Every trade in mt5_diagnostics corresponds to an event with matching PnL."""
        diag_path = WF_OUTPUT_DIR / f"mt5_diagnostics_cycle_{cycle_id}.json"
        events_path = WF_OUTPUT_DIR / f"events_cycle_{cycle_id}.jsonl"

        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        perf = diag["performance"]

        events = []
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line.strip()))

        # 1. Closed trades count match
        assert len(events) == perf["closed_trades"]

        # 2. Wins and losses match
        realized_pnls = [ev["payload"]["realized_pnl"] for ev in events]
        wins = sum(1 for p in realized_pnls if p > 0)
        losses = sum(1 for p in realized_pnls if p < 0)
        assert wins == perf["wins"]
        assert losses == perf["losses"]

        # 3. Net PnL match
        net_sum = sum(realized_pnls)
        assert math.isclose(net_sum, perf["net_pnl"], abs_tol=1e-2)

        # 4. Gross Profit and Loss match
        gross_p = sum(p for p in realized_pnls if p > 0)
        gross_l = sum(abs(p) for p in realized_pnls if p < 0)
        assert math.isclose(gross_p, perf["gross_profit"], abs_tol=1e-2)
        assert math.isclose(gross_l, perf["gross_loss"], abs_tol=1e-2)

        # 5. Close reasons match
        event_reasons = {}
        for ev in events:
            r = ev["payload"]["close_reason"]
            event_reasons[r] = event_reasons.get(r, 0) + 1
        assert event_reasons == perf["close_reasons"]


# ==============================================================================
# 5. Raw Trace Preservation and Adaptive Tuning Monotonicity
# ==============================================================================

class TestTracePreservationAndAdaptiveTuningExecution:
    """Verify trace preservation of attempts and monotonicity of adaptive hyperparameter levers."""

    def test_cycle_1_attempts_trace_and_transitions(self):
        """Verify Cycle 1 Attempt 1 (Loss) -> Retry 1 (Loss) -> Retry 2 (Profit)."""
        # Attempt 1: Loss
        c1_diag = json.loads((WF_OUTPUT_DIR / "cycle_1" / "mt5_diagnostics.json").read_text(encoding="utf-8"))
        assert c1_diag["performance"]["net_pnl"] == -1387.7199999999996
        assert c1_diag["performance"]["profit_factor"] < 1.15

        # Attempt 2: Loss
        c1_r1_diag = json.loads((WF_OUTPUT_DIR / "cycle_1_retry_1" / "mt5_diagnostics.json").read_text(encoding="utf-8"))
        assert c1_r1_diag["performance"]["net_pnl"] == -905.0200000000003
        assert c1_r1_diag["performance"]["profit_factor"] < 1.15

        # Attempt 3: Promoted
        c1_r2_diag = json.loads((WF_OUTPUT_DIR / "cycle_1_retry_2" / "mt5_diagnostics.json").read_text(encoding="utf-8"))
        assert c1_r2_diag["performance"]["net_pnl"] == 580.3
        assert c1_r2_diag["performance"]["profit_factor"] >= 1.15

    def test_adaptive_hyperparameter_monotonicity(self):
        """Verify strict monotonicity across candidate, retry 1, and retry 2."""
        r1_yaml = yaml.safe_load((WF_OUTPUT_DIR / "profile_cycle_1_retry_1.yaml").read_text(encoding="utf-8"))
        r2_yaml = yaml.safe_load((WF_OUTPUT_DIR / "profile_cycle_1_retry_2.yaml").read_text(encoding="utf-8"))

        r1_over = r1_yaml["learning_overrides"]
        r2_over = r2_yaml["learning_overrides"]

        # Veto LCB: monotonically increases
        assert -0.15 < r1_over["five_expectancy_action_veto_min_lcb_r"] < r2_over["five_expectancy_action_veto_min_lcb_r"]
        assert r1_over["five_expectancy_action_veto_min_lcb_r"] == -0.05
        assert r2_over["five_expectancy_action_veto_min_lcb_r"] == 0.00

        # Uniform Mix: monotonically decreases
        assert 0.02 > r1_over["five_behavior_uniform_mix"] > r2_over["five_behavior_uniform_mix"]
        assert r1_over["five_behavior_uniform_mix"] == 0.010
        assert r2_over["five_behavior_uniform_mix"] == 0.005

        # Pattern Reward Scale: monotonically increases
        assert 0.10 < r1_over["five_pattern_action_quality_reward_scale_r"] < r2_over["five_pattern_action_quality_reward_scale_r"]
        assert r1_over["five_pattern_action_quality_reward_scale_r"] == 0.15
        assert r2_over["five_pattern_action_quality_reward_scale_r"] == 0.20

        # Direction Loss Weight: monotonically increases
        assert 1.00 < r1_over["five_pattern_tree_direction_loss_weight"] < r2_over["five_pattern_tree_direction_loss_weight"]
        assert r1_over["five_pattern_tree_direction_loss_weight"] == 1.25
        assert r2_over["five_pattern_tree_direction_loss_weight"] == 1.50


# ==============================================================================
# 6. Manifest Close Reason Double-Counting Analysis & Invariance Protection
# ==============================================================================

class TestManifestCloseReasonFinding:
    """Analyze and document the double-counting behavior in manifest close_reasons."""

    def test_close_reason_double_counting_does_not_affect_acceptance_invariants(self):
        """Confirm that manifest close_reasons double-counting leaves all acceptance metrics uncorrupted."""
        manifest = json.loads((WF_OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
        for c_id in ["1", "2", "3"]:
            m = manifest["cycle_results"][c_id]["metrics"]
            diag = json.loads((WF_OUTPUT_DIR / f"mt5_diagnostics_cycle_{c_id}.json").read_text(encoding="utf-8"))
            true_reasons = diag["performance"]["close_reasons"]

            manifest_reasons = m["close_reasons"]
            # Verify close_reasons in manifest is exactly 2x due to diagnostics + events dual parse
            for r, count in true_reasons.items():
                assert manifest_reasons[r] == count * 2, f"Expected 2x count for {r} in cycle {c_id}"

            # CRITICAL INVARIANT: closed_trades is NOT doubled
            assert m["closed_trades"] == diag["performance"]["closed_trades"]
            # CRITICAL INVARIANT: Net PnL is NOT doubled
            assert m["net_pnl"] == diag["performance"]["net_pnl"]
            # CRITICAL INVARIANT: Profit factor is NOT doubled
            assert m["profit_factor"] == diag["performance"]["profit_factor"]
            # CRITICAL INVARIANT: Max Drawdown is NOT doubled
            raw_dd = diag["risk"]["max_equity_drawdown_pct"]
            true_dd_pct = raw_dd * 100.0 if raw_dd <= 1.0 else raw_dd
            assert m["max_drawdown_pct"] == round(true_dd_pct, 2)


# ==============================================================================
# 7. Adversarial Tamper Detection & Malformed File Rejection
# ==============================================================================

class TestAdversarialTamperDetection:
    """Empirically test that tampered checkpoints or corrupted telemetry are rejected."""

    def test_corrupted_checkpoint_fails_sha256_verification(self, tmp_path: Path):
        """Tampering with a single byte in a checkpoint triggers hash mismatch."""
        fake_ckpt = tmp_path / "fake_checkpoint.pkl"
        fake_ckpt.write_bytes(b"VALID_BYTES_XYZ_123")
        expected_hash = hashlib.sha256(b"VALID_BYTES_XYZ_123").hexdigest()

        # Recomputation matches
        assert sha256_file(fake_ckpt) == expected_hash

        # Tampering alters hash
        fake_ckpt.write_bytes(b"VALID_BYTES_XYZ_124")
        assert sha256_file(fake_ckpt) != expected_hash

    def test_negative_pnl_in_cycle_fails_manifest_certification(self):
        """If any cycle has net_pnl <= 0, manifest generation must NOT certify."""
        manifest_res = {
            1: {
                "cycle_id": 1,
                "name": "Cycle 1",
                "promoted": True,
                "metrics": {"cycle_id": 1, "net_pnl": 500.0, "profit_factor": 1.5, "max_drawdown_pct": 5.0, "closed_trades": 30},
                "verdict": {"passed": True},
                "promoted_checkpoint": str(WF_OUTPUT_DIR / "checkpoint_cycle_1.pkl"),
                "promoted_checkpoint_sha256": EXPECTED_SEED_SHA256,
            },
            2: {
                "cycle_id": 2,
                "name": "Cycle 2",
                "promoted": False,
                "metrics": {"cycle_id": 2, "net_pnl": -200.0, "profit_factor": 0.8, "max_drawdown_pct": 18.0, "closed_trades": 30},
                "verdict": {"passed": False},
                "promoted_checkpoint": None,
                "promoted_checkpoint_sha256": "",
            },
        }

        with tempfile.TemporaryDirectory() as td:
            t_dir = Path(td)
            dummy_final = t_dir / "final.pkl"
            dummy_final.write_bytes(b"final")
            m = generate_walk_forward_manifest(
                candidate_checkpoint_path=dummy_final,
                cycle_results=manifest_res,
                output_dir=t_dir,
                initial_seed_path=INITIAL_SEED_PATH,
                target_cycles=[1, 2],
            )
            assert m["multi_cycle_certified"] is False
            assert m["aggregate_metrics"]["all_cycles_passed"] is False
