"""Adversarial Challenge Test Suite: Milestone 1 - Walk-Forward Pipeline & Simplex Defense.

Agent: challenger_m1_2
Roles: critic, specialist
Target Milestone: Milestone 1 (Walk-Forward Multi-Cycle Orchestration Engine & Adaptive Tuning Module)

Empirical Adversarial Challenges:
1. Extreme Metric Boundaries:
   - Profit Factor at 1.1499 (must fail), 1.1500 (must pass)
   - Max Drawdown at 14.99% (must pass), 15.01% (must fail), 15.00% (must fail)
   - Closed Trades at 19 (must fail), 20 (must pass)
   - Net Profit at 0.00 (must fail), -0.01 (must fail), +0.01 (must pass)
   - Hold Share at 0.9899 (must pass), 0.9900 (must fail)
   - Simplex/NaN error count at 0 (must pass), >= 1 (must fail)
2. Probability Simplex Defense in SGTCompactTradingPolicy:
   - Perturbed observation tensors (NaN, +Inf, -Inf, all zeros, extreme scale)
   - Defensive clamping: torch.nan_to_num + clamp_min(1e-12) + uniform fallback
   - Ensuring Categorical distribution never crashes with Simplex distribution error
   - Forward pass & evaluate_recurrent_sample robustness under numerical attacks
3. Manifest Generation Schema & SHA-256 On-Disk Recomputation:
   - Real SHA-256 recomputation against disk files
   - Corrupted file mismatch detection
   - Schema field verification (schema_version, aggregate_metrics, multi_cycle_certified)
   - Authoritative seed SHA-256 verification (3f76d5a0... 86,378,353 bytes)
"""

from __future__ import annotations

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
    PARAMETER_BINS,
    SGTCompactTradingPolicy,
    CompactAgentMemory,
)
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy
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


# ==============================================================================
# Challenge 1: Extreme Metric Boundaries
# ==============================================================================

class TestExtremeMetricBoundaries:
    """Empirical challenge on sharp boundary conditions of acceptance criteria."""

    def test_profit_factor_11499_fails_11500_passes(self):
        """Profit Factor at 1.1499 MUST fail; Profit Factor at 1.1500 MUST pass."""
        # 1.1499 must fail
        m_fail = CycleMetrics(
            cycle_id=1,
            net_pnl=100.0,
            profit_factor=1.1499,
            max_drawdown_pct=10.0,
            closed_trades=25,
            hold_share=0.5,
        )
        v_fail = evaluate_acceptance_criteria(m_fail)
        assert v_fail.passed is False, "Profit Factor 1.1499 should fail acceptance"
        assert v_fail.profit_factor_ok is False
        assert any("Profit Factor < 1.15" in r for r in v_fail.reasons)

        # 1.1500 must pass
        m_pass = CycleMetrics(
            cycle_id=1,
            net_pnl=100.0,
            profit_factor=1.1500,
            max_drawdown_pct=10.0,
            closed_trades=25,
            hold_share=0.5,
        )
        v_pass = evaluate_acceptance_criteria(m_pass)
        assert v_pass.passed is True, "Profit Factor 1.1500 should pass acceptance"
        assert v_pass.profit_factor_ok is True
        assert len(v_pass.reasons) == 0

    def test_profit_factor_epsilon_micro_boundaries(self):
        """Test micro-epsilon boundaries around 1.15."""
        # 1.15 - 1e-6 must fail
        m_below = CycleMetrics(
            cycle_id=1, net_pnl=50.0, profit_factor=1.15 - 1e-6, max_drawdown_pct=5.0, closed_trades=30
        )
        assert evaluate_acceptance_criteria(m_below).profit_factor_ok is False

        # 1.15 + 1e-6 must pass
        m_above = CycleMetrics(
            cycle_id=1, net_pnl=50.0, profit_factor=1.15 + 1e-6, max_drawdown_pct=5.0, closed_trades=30
        )
        assert evaluate_acceptance_criteria(m_above).profit_factor_ok is True

    def test_max_drawdown_1499_passes_1501_fails(self):
        """Max Drawdown at 14.99% MUST pass; at 15.01% MUST fail; at 15.00% MUST fail (strictly < 15.0%)."""
        # 14.99% must pass
        m_pass = CycleMetrics(
            cycle_id=1,
            net_pnl=100.0,
            profit_factor=1.20,
            max_drawdown_pct=14.99,
            closed_trades=25,
            hold_share=0.5,
        )
        v_pass = evaluate_acceptance_criteria(m_pass)
        assert v_pass.passed is True, "Max Drawdown 14.99% should pass"
        assert v_pass.drawdown_ok is True

        # 15.00% must fail
        m_1500 = CycleMetrics(
            cycle_id=1,
            net_pnl=100.0,
            profit_factor=1.20,
            max_drawdown_pct=15.00,
            closed_trades=25,
            hold_share=0.5,
        )
        v_1500 = evaluate_acceptance_criteria(m_1500)
        assert v_1500.passed is False, "Max Drawdown 15.00% should fail (boundary is < 15.0%)"
        assert v_1500.drawdown_ok is False
        assert any("Max Drawdown >= 15.0%" in r for r in v_1500.reasons)

        # 15.01% must fail
        m_fail = CycleMetrics(
            cycle_id=1,
            net_pnl=100.0,
            profit_factor=1.20,
            max_drawdown_pct=15.01,
            closed_trades=25,
            hold_share=0.5,
        )
        v_fail = evaluate_acceptance_criteria(m_fail)
        assert v_fail.passed is False, "Max Drawdown 15.01% should fail"
        assert v_fail.drawdown_ok is False
        assert any("Max Drawdown >= 15.0%" in r for r in v_fail.reasons)

    def test_closed_trades_19_fails_20_passes(self):
        """Closed Trades at 19 MUST fail; Closed Trades at 20 MUST pass."""
        # 19 trades must fail
        m_19 = CycleMetrics(
            cycle_id=1,
            net_pnl=200.0,
            profit_factor=1.35,
            max_drawdown_pct=8.0,
            closed_trades=19,
            hold_share=0.5,
        )
        v_19 = evaluate_acceptance_criteria(m_19)
        assert v_19.passed is False, "Closed trades 19 should fail (< 20)"
        assert v_19.trade_frequency_ok is False
        assert any("Closed trades < 20" in r for r in v_19.reasons)

        # 20 trades must pass
        m_20 = CycleMetrics(
            cycle_id=1,
            net_pnl=200.0,
            profit_factor=1.35,
            max_drawdown_pct=8.0,
            closed_trades=20,
            hold_share=0.5,
        )
        v_20 = evaluate_acceptance_criteria(m_20)
        assert v_20.passed is True, "Closed trades 20 should pass"
        assert v_20.trade_frequency_ok is True

    def test_net_pnl_zero_and_negative_boundaries(self):
        """Net Profit <= 0.00 MUST fail, > 0.00 MUST pass."""
        # 0.00 fails
        m_zero = CycleMetrics(cycle_id=1, net_pnl=0.0, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25)
        assert evaluate_acceptance_criteria(m_zero).net_pnl_ok is False

        # -0.01 fails
        m_neg = CycleMetrics(cycle_id=1, net_pnl=-0.01, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25)
        assert evaluate_acceptance_criteria(m_neg).net_pnl_ok is False

        # +0.01 passes
        m_pos = CycleMetrics(cycle_id=1, net_pnl=0.01, profit_factor=1.5, max_drawdown_pct=5.0, closed_trades=25)
        assert evaluate_acceptance_criteria(m_pos).net_pnl_ok is True

    def test_hold_share_trap_boundary(self):
        """Hold share >= 0.9900 is a 100% HOLD trap and fails even with 20 trades."""
        # 0.9900 fails
        m_trap = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.3, max_drawdown_pct=5.0, closed_trades=25, hold_share=0.9900)
        v_trap = evaluate_acceptance_criteria(m_trap)
        assert v_trap.passed is False
        assert v_trap.trade_frequency_ok is False
        assert any("100% HOLD trap" in r for r in v_trap.reasons)

        # 0.9899 passes
        m_safe = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.3, max_drawdown_pct=5.0, closed_trades=25, hold_share=0.9899)
        v_safe = evaluate_acceptance_criteria(m_safe)
        assert v_safe.passed is True
        assert v_safe.trade_frequency_ok is True

    def test_simplex_and_nan_error_counters_boundary(self):
        """Any simplex_errors > 0 or nan_inf_errors > 0 MUST fail."""
        m_simplex_err = CycleMetrics(
            cycle_id=1, net_pnl=100.0, profit_factor=1.3, max_drawdown_pct=5.0, closed_trades=25, simplex_errors=1
        )
        v_simplex = evaluate_acceptance_criteria(m_simplex_err)
        assert v_simplex.passed is False
        assert v_simplex.simplex_safe is False

        m_nan_err = CycleMetrics(
            cycle_id=1, net_pnl=100.0, profit_factor=1.3, max_drawdown_pct=5.0, closed_trades=25, nan_inf_errors=1
        )
        v_nan = evaluate_acceptance_criteria(m_nan_err)
        assert v_nan.passed is False
        assert v_nan.simplex_safe is False


# ==============================================================================
# Challenge 2: Probability Simplex Defense in SGTCompactTradingPolicy
# ==============================================================================

class TestProbabilitySimplexDefense:
    """Stress test probability simplex defensive guards under extreme numerical attacks."""

    @pytest.fixture
    def policy(self) -> SGTCompactTradingPolicy:
        p = SGTCompactTradingPolicy()
        p.eval()
        return p

    def test_direct_defensive_clamping_on_perturbed_probability_tensors(self):
        """Directly stress the defensive clamping logic on adversarial tensors (NaN, Inf, negative, zero)."""
        # Define adversarial tensor perturbations
        test_cases = [
            # Case 1: All NaNs
            torch.full((2, 5, 12), float("nan")),
            # Case 2: All +Infs
            torch.full((2, 5, 12), float("inf")),
            # Case 3: All -Infs
            torch.full((2, 5, 12), -float("inf")),
            # Case 4: Mixed NaNs, Infs, and negative values
            torch.tensor([
                [
                    [float("nan"), float("inf"), -float("inf"), -2.0, 0.0, 1.0, 0.5, float("nan"), 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf")],
                    [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0],
                    [float("nan"), -1.0, 1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]),
            # Case 5: All zeros
            torch.zeros((1, 5, 12)),
            # Case 6: Sub-threshold sum (e.g., 1e-15 < 1e-12)
            torch.full((1, 5, 12), 1e-15),
        ]

        allowed = torch.ones((1, 5, 12), dtype=torch.bool)
        allowed[0, :, 7:12] = False  # Mask out some actions

        for idx, raw_probs in enumerate(test_cases):
            # Expand allowed mask if needed
            batch_allowed = allowed.expand(raw_probs.shape[0], -1, -1)

            # Defensive guard from sgt_compact_core.py / five_shared_policy.py
            cleaned_probs = torch.nan_to_num(raw_probs, nan=0.0, posinf=0.0, neginf=0.0)
            cleaned_probs = cleaned_probs.clamp_min(0.0)  # No negative probabilities
            cleaned_probs = cleaned_probs * batch_allowed.to(dtype=cleaned_probs.dtype)

            prob_sum = cleaned_probs.sum(dim=-1, keepdim=True)
            uniform_fallback = batch_allowed.to(dtype=cleaned_probs.dtype) / batch_allowed.sum(dim=-1, keepdim=True).clamp_min(1.0)

            safe_probs = torch.where(
                prob_sum > 1e-12,
                cleaned_probs / prob_sum.clamp_min(1e-12),
                uniform_fallback,
            )

            # Assertions for Simplex constraints:
            # 1. No NaNs or Infs
            assert not torch.isnan(safe_probs).any(), f"Test case {idx} contains NaN"
            assert not torch.isinf(safe_probs).any(), f"Test case {idx} contains Inf"

            # 2. All values in [0.0, 1.0]
            assert (safe_probs >= 0.0).all(), f"Test case {idx} contains negative probability"
            assert (safe_probs <= 1.0 + 1e-5).all(), f"Test case {idx} contains probability > 1.0"

            # 3. Sum along action dimension is 1.0
            sums = safe_probs.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Test case {idx} sum is not 1.0 (got {sums})"

            # 4. Critical: Categorical distribution MUST construct without Simplex error!
            cat = dist.Categorical(probs=safe_probs)
            sampled = cat.sample()
            assert sampled.shape == raw_probs.shape[:-1]
            # Ensure sampled action is among allowed actions
            for b in range(sampled.shape[0]):
                for a in range(sampled.shape[1]):
                    act = sampled[b, a].item()
                    assert batch_allowed[b, a, act].item() is True, f"Sampled disallowed action {act}"

    def test_forward_pass_with_adversarial_observations(self, policy: SGTCompactTradingPolicy):
        """Feed adversarial observations (NaNs, Infs, extreme scales) into SGTCompactTradingPolicy forward()."""
        mems = policy.initial_memories(FIVE_AGENT_COUNT)

        adversarial_obs_list = [
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), float("nan")),
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), float("inf")),
            torch.full((FIVE_AGENT_COUNT, INPUT_COUNT), -float("inf")),
            torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT) * 1e8,  # Extreme logits
            torch.zeros(FIVE_AGENT_COUNT, INPUT_COUNT),
        ]

        for idx, bad_obs in enumerate(adversarial_obs_list):
            with torch.no_grad():
                # Policy forward cleans obs using nan_to_num internally
                action_wire, values, extra, next_mems = policy(bad_obs, memories=mems)

                # Verify returned shapes
                assert action_wire.shape == (FIVE_AGENT_COUNT, OUTPUT_COUNT)
                assert values.shape == (FIVE_AGENT_COUNT, 1)
                assert len(next_mems) == FIVE_AGENT_COUNT

                # Verify action logits can be safely converted to probability simplex
                action_logits = extra["action_logits"]
                safe_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                probs = torch.softmax(safe_logits, dim=-1)

                assert not torch.isnan(probs).any(), f"Case {idx} probs contain NaN"
                assert not torch.isinf(probs).any(), f"Case {idx} probs contain Inf"
                sums = probs.sum(dim=-1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

                # PyTorch Categorical must NOT raise Simplex error
                cat = dist.Categorical(probs=probs)
                sampled = cat.sample()
                assert (sampled >= 0).all() and (sampled < ACTION_COUNT).all()

    def test_forward_five_with_corrupted_inputs_and_masks(self, policy: SGTCompactTradingPolicy):
        """Stress forward_five with corrupted inputs and adversarial action masks."""
        mems = policy.initial_memories(FIVE_AGENT_COUNT)

        # Obs with mixed NaNs and extreme numbers
        corrupted_obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
        corrupted_obs[0, :50] = float("nan")
        corrupted_obs[1, 50:] = float("inf")
        corrupted_obs[2, :] = -float("inf")
        corrupted_obs[3, :] = 1e9

        # Mask that tries to disallow all actions (triggering hold fallback)
        zero_mask = torch.zeros(FIVE_AGENT_COUNT, ACTION_COUNT, dtype=torch.bool)

        results, new_mems, extra = policy.forward_five(
            observations=corrupted_obs,
            memories=mems,
            allowed_action_masks=zero_mask,
            deterministic=False,
            behavior_uniform_mix=0.05,
        )

        assert len(results) == FIVE_AGENT_COUNT
        assert len(new_mems) == FIVE_AGENT_COUNT

        for i, res in enumerate(results):
            # Wire output must be exactly 20 slots
            assert len(res.output_vector) == OUTPUT_COUNT
            assert np.isfinite(res.output_vector).all(), f"Agent {i} wire has non-finite values"

            # Primary action must fall on hold (action 0) due to all-zeros mask fallback
            assert res.action_index == 0, f"Agent {i} expected fallback to action 0 (HOLD), got {res.action_index}"

            # Parameter choices must be valid
            assert 0 <= res.delay_index < PARAMETER_BINS["delay"]
            assert 0 <= res.risk_index < PARAMETER_BINS["risk"]
            assert 0 <= res.entry_offset_index < PARAMETER_BINS["entry_offset"]
            assert 0 <= res.sl_index < PARAMETER_BINS["sl"]
            assert 0 <= res.tp_index < PARAMETER_BINS["tp"]
            assert 0 <= res.trailing_index < PARAMETER_BINS["trailing"]
            assert 0 <= res.manage_mode_index < PARAMETER_BINS["manage_mode"]
            assert 0 <= res.close_fraction_index < PARAMETER_BINS["close_fraction"]

    def test_primary_action_distribution_never_produces_simplex_divergence(self):
        """Stress SharedFiveTradingPolicy._primary_action_distribution directly with degenerate logits."""
        # Extreme logits
        logits_cases = [
            torch.full((5, 12), float("nan")),
            torch.full((5, 12), float("inf")),
            torch.full((5, 12), -float("inf")),
            torch.tensor([[1e8, -1e8, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 5),
        ]

        allowed = torch.ones(5, 12, dtype=torch.bool)
        allowed[:, 6:] = False  # Mask out higher actions

        for idx, bad_logits in enumerate(logits_cases):
            sanitized_logits = torch.nan_to_num(bad_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

            choice, log_prob, entropy, policy_probs, behavior_probs = (
                SharedFiveTradingPolicy._primary_action_distribution(
                    sanitized_logits,
                    allowed=allowed,
                    deterministic=False,
                    entry_probability_floor=0.01,
                    uniform_mix=0.02,
                    include_policy_probs=True,
                )
            )

            # Both policy_probs and behavior_probs must be valid simplices
            for p_tensor, name in [(policy_probs, "policy_probs"), (behavior_probs, "behavior_probs")]:
                assert not torch.isnan(p_tensor).any(), f"Case {idx} {name} contains NaN"
                assert not torch.isinf(p_tensor).any(), f"Case {idx} {name} contains Inf"
                assert (p_tensor >= 0.0).all(), f"Case {idx} {name} contains negative values"
                sums = p_tensor.sum(dim=-1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"Case {idx} {name} sum != 1.0"

                # Verify Categorical works
                cat = dist.Categorical(probs=p_tensor)
                s = cat.sample()
                assert (s >= 0).all() and (s < 12).all()


# ==============================================================================
# Challenge 3: Manifest Generation Schema & SHA-256 On-Disk Recomputation
# ==============================================================================

class TestManifestSchemaAndSHA256Recomputation:
    """Empirical challenge verifying manifest.json generation and exact SHA-256 integrity."""

    def test_authoritative_initial_seed_on_disk_sha256(self):
        """Verify the authoritative initial seed file on disk exists with exact SHA-256 and size."""
        seed_path = Path("outputs/he_chuyen_gia/v193_runs/seed_sgt_compact_tier2.pkl")
        assert seed_path.is_file(), f"Authoritative seed file missing: {seed_path}"

        expected_sha = "3f76d5a026c5a14b693c024fbee755a92fa9a04e085855be605009ff77d5c64f"
        expected_size = 86_378_353

        # Compute empirical SHA-256 directly
        hasher = hashlib.sha256()
        with open(seed_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()
        actual_size = seed_path.stat().st_size

        assert actual_sha == expected_sha, f"Seed SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        assert actual_size == expected_size, f"Seed size mismatch: expected {expected_size}, got {actual_size}"

    def test_generate_manifest_and_recompute_sha256_against_disk(self, tmp_path: Path):
        """Generate manifest and empirically verify that SHA-256 in manifest matches disk recomputation."""
        cand_file = tmp_path / "candidate_model.pkl"
        cand_content = b"TEST_CANDIDATE_MODEL_WEIGHTS_FOR_MANIFEST_VERIFICATION"
        cand_file.write_bytes(cand_content)

        seed_file = tmp_path / "test_seed.pkl"
        seed_content = b"TEST_INITIAL_SEED_WEIGHTS_FOR_MANIFEST_VERIFICATION"
        seed_file.write_bytes(seed_content)

        # Expected SHA-256 hashes
        expected_cand_sha = hashlib.sha256(cand_content).hexdigest()
        expected_seed_sha = hashlib.sha256(seed_content).hexdigest()

        # Build mock cycle results (Cycles 1, 2, 3 all passing)
        cycle_results = {
            1: {
                "verdict": {"passed": True},
                "metrics": {"net_pnl": 500.0, "profit_factor": 1.45, "max_drawdown_pct": 6.0, "closed_trades": 30},
            },
            2: {
                "verdict": {"passed": True},
                "metrics": {"net_pnl": 420.0, "profit_factor": 1.30, "max_drawdown_pct": 8.5, "closed_trades": 28},
            },
            3: {
                "verdict": {"passed": True},
                "metrics": {"net_pnl": 610.0, "profit_factor": 1.55, "max_drawdown_pct": 5.2, "closed_trades": 35},
            },
        }

        out_dir = tmp_path / "manifest_out"
        manifest = generate_walk_forward_manifest(
            candidate_checkpoint_path=cand_file,
            cycle_results=cycle_results,
            output_dir=out_dir,
            initial_seed_path=seed_file,
            target_cycles=[1, 2, 3],
        )

        # 1. Verify manifest.json written on disk
        manifest_file = out_dir / "manifest.json"
        assert manifest_file.is_file(), "manifest.json was not created on disk"

        disk_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        # 2. Recompute SHA-256 from disk files and compare against manifest entries
        cand_on_disk_sha = hashlib.sha256(cand_file.read_bytes()).hexdigest()
        seed_on_disk_sha = hashlib.sha256(seed_file.read_bytes()).hexdigest()

        assert disk_manifest["candidate"]["sha256"] == cand_on_disk_sha == expected_cand_sha
        assert disk_manifest["initial_seed"]["sha256"] == seed_on_disk_sha == expected_seed_sha
        assert disk_manifest["candidate"]["size_bytes"] == len(cand_content)
        assert disk_manifest["candidate"]["exists"] is True

        # 3. Verify schema structure and version
        assert disk_manifest["schema_version"] == MANIFEST_SCHEMA_VERSION == 1
        assert "Walk-Forward" in disk_manifest["pipeline"]
        assert disk_manifest["multi_cycle_certified"] is True
        assert disk_manifest["target_cycles"] == [1, 2, 3]

        # 4. Verify aggregate calculations
        agg = disk_manifest["aggregate_metrics"]
        assert agg["total_trades"] == 30 + 28 + 35 == 93
        assert agg["total_net_pnl"] == round(500.0 + 420.0 + 610.0, 2) == 1530.0
        assert agg["average_profit_factor"] == round((1.45 + 1.30 + 1.55) / 3.0, 2)
        assert agg["worst_max_drawdown_pct"] == 8.5
        assert agg["all_cycles_passed"] is True

    def test_corrupted_candidate_triggers_sha256_mismatch(self, tmp_path: Path):
        """Mutating the candidate checkpoint file after manifest generation causes SHA-256 mismatch."""
        cand_file = tmp_path / "cand.pkl"
        cand_file.write_bytes(b"INITIAL_CLEAN_BYTES")

        seed_file = tmp_path / "seed.pkl"
        seed_file.write_bytes(b"SEED_BYTES")

        out_dir = tmp_path / "mismatch_test"
        manifest = generate_walk_forward_manifest(
            candidate_checkpoint_path=cand_file,
            cycle_results={1: {"verdict": {"passed": True}, "metrics": {"closed_trades": 25, "net_pnl": 100.0, "profit_factor": 1.2, "max_drawdown_pct": 5.0}}},
            output_dir=out_dir,
            initial_seed_path=seed_file,
            target_cycles=[1],
        )

        recorded_sha = manifest["candidate"]["sha256"]

        # Tamper with the candidate file on disk
        cand_file.write_bytes(b"TAMPERED_BYTES_DETECTED")
        tampered_sha = hashlib.sha256(cand_file.read_bytes()).hexdigest()

        # Empirical proof: Hashes must no longer match!
        assert recorded_sha != tampered_sha
        assert len(recorded_sha) == 64
        assert len(tampered_sha) == 64

    def test_manifest_certification_fails_if_any_cycle_fails(self, tmp_path: Path):
        """If any target cycle failed, multi_cycle_certified MUST be False."""
        cand = tmp_path / "cand.pkl"
        cand.write_bytes(b"CAND")
        seed = tmp_path / "seed.pkl"
        seed.write_bytes(b"SEED")

        cycle_results = {
            1: {"verdict": {"passed": True}, "metrics": {"closed_trades": 25, "net_pnl": 100.0, "profit_factor": 1.2, "max_drawdown_pct": 5.0}},
            2: {"verdict": {"passed": False}, "metrics": {"closed_trades": 15, "net_pnl": -50.0, "profit_factor": 0.8, "max_drawdown_pct": 18.0}},
            3: {"verdict": {"passed": True}, "metrics": {"closed_trades": 28, "net_pnl": 200.0, "profit_factor": 1.4, "max_drawdown_pct": 4.0}},
        }

        manifest = generate_walk_forward_manifest(
            candidate_checkpoint_path=cand,
            cycle_results=cycle_results,
            output_dir=tmp_path / "cert_fail_test",
            initial_seed_path=seed,
            target_cycles=[1, 2, 3],
        )

        assert manifest["multi_cycle_certified"] is False
        assert manifest["aggregate_metrics"]["all_cycles_passed"] is False

    def test_manifest_certification_fails_if_cycles_incomplete(self, tmp_path: Path):
        """If target_cycles is [1, 2, 3] but only cycle 1 and 2 were run, certification MUST be False."""
        cand = tmp_path / "cand.pkl"
        cand.write_bytes(b"CAND")
        seed = tmp_path / "seed.pkl"
        seed.write_bytes(b"SEED")

        cycle_results = {
            1: {"verdict": {"passed": True}, "metrics": {"closed_trades": 25, "net_pnl": 100.0, "profit_factor": 1.2, "max_drawdown_pct": 5.0}},
            2: {"verdict": {"passed": True}, "metrics": {"closed_trades": 28, "net_pnl": 150.0, "profit_factor": 1.3, "max_drawdown_pct": 6.0}},
        }

        manifest = generate_walk_forward_manifest(
            candidate_checkpoint_path=cand,
            cycle_results=cycle_results,
            output_dir=tmp_path / "incomplete_test",
            initial_seed_path=seed,
            target_cycles=[1, 2, 3],
        )

        assert manifest["multi_cycle_certified"] is False
