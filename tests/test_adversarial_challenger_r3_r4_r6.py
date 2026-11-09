"""Empirical and Adversarial Challenger Test Suite for R3, R4, and R6 Remediations.

This suite executes rigorous mathematical gradient checks, adversarial stress harnesses,
and boundary verification for:
1. R3: SL Penalty Redirection to HOLD (dL/dz0 < 0, dL/dz_chosen > 0, dL/dz_other == 0)
2. R4: Bilinear State-Action Reward Model intra-state gradient preservation and dataset balance gate
3. R6: Teacher LR config validation and policy step_from_loss gradient norm clipping
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import numpy as np
import pytest
import torch
from torch import Tensor

from src.self_growing_trader.five_shared_policy import (
    FivePPOTrainer,
    SharedFiveTradingPolicy,
    ACTION_COUNT,
)
from src.self_growing_trader.rlhf_reward_model import (
    RLHFBilinearRewardModel,
    RLHFRewardPair,
    RLHFPreferencePair,
    fit_rlhf_bilinear_reward_model,
)
from src.self_growing_trader.rlhf_preference import rlhf_dataset_balance_gate
from src.self_growing_trader.config import LearningIntegrityConfig


# ==============================================================================
# PART 1: R3 SL PENALTY REDIRECTION & GRADIENT VERIFICATION
# ==============================================================================


class TestR3SLPenaltyGradientProof:
    """Mathematical gradient proofs and empirical checks for initial_sl_conditional_entry_loss."""

    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_r3_mathematical_gradient_proof_analytical_vs_autograd(
        self, batch_size: int
    ) -> None:
        """Verify element-by-element exact match between analytical derivatives and autograd.

        Loss definition:
            per_sample = softplus(z_chosen - z_0)
            L = mean(per_sample[eligible] * weight[eligible])

        Derivatives for eligible entry (b, a_idx):
            dL / d(z_chosen) = (w_b / N) * sigmoid(z_chosen - z_0) > 0
            dL / d(z_0)      = -(w_b / N) * sigmoid(z_chosen - z_0) < 0
            dL / d(z_other)  = 0.0 (zero gradient leakage)
        """
        torch.manual_seed(42 + batch_size)
        B = batch_size
        num_agents = 5
        logits = torch.randn((B, num_agents, ACTION_COUNT), dtype=torch.float32, requires_grad=True)

        # Actions in entry range [1, 6]
        actions = torch.randint(1, 7, (B, num_agents), dtype=torch.long)
        # Random masks with at least one true
        masks = torch.rand((B, num_agents)) > 0.3
        masks[0, 0] = True  # Guarantee at least one eligible
        weights = torch.rand(B, dtype=torch.float32) * 2.0 + 0.1  # strictly positive weights

        loss, count = FivePPOTrainer.initial_sl_conditional_entry_loss(
            logits, actions, masks, weights
        )
        assert count > 0
        loss.backward()

        autograd_grad = logits.grad.detach()

        # Compute analytical gradient manually
        expected_grad = torch.zeros_like(logits)
        eligible_mask = masks & (actions >= 1) & (actions <= 6)
        N = float(count)

        for b in range(B):
            w_b = float(weights[b])
            for a_idx in range(num_agents):
                if eligible_mask[b, a_idx]:
                    chosen_act = int(actions[b, a_idx])
                    z_chosen = float(logits[b, a_idx, chosen_act].detach())
                    z_0 = float(logits[b, a_idx, 0].detach())
                    diff = z_chosen - z_0
                    sig = 1.0 / (1.0 + math.exp(-diff))

                    dL_dz_chosen = (w_b / N) * sig
                    dL_dz_0 = -(w_b / N) * sig

                    expected_grad[b, a_idx, chosen_act] = dL_dz_chosen
                    expected_grad[b, a_idx, 0] = dL_dz_0

        # 1. Exact numerical match between PyTorch autograd and analytical proof
        torch.testing.assert_close(autograd_grad, expected_grad, rtol=1e-5, atol=1e-6)

        # 2. Strict inequality checks:
        for b in range(B):
            for a_idx in range(num_agents):
                if eligible_mask[b, a_idx]:
                    chosen_act = int(actions[b, a_idx])
                    # HOLD always boosted (gradient is negative in descent loss)
                    assert autograd_grad[b, a_idx, 0] < 0.0, "dL/dz0 must be strictly negative"
                    # Chosen entry penalized (gradient is positive)
                    assert autograd_grad[b, a_idx, chosen_act] > 0.0, "dL/dz_chosen must be strictly positive"
                    # Zero gradient leakage to all other actions
                    for act in range(ACTION_COUNT):
                        if act != 0 and act != chosen_act:
                            assert autograd_grad[b, a_idx, act] == 0.0, (
                                f"Gradient leakage detected at batch {b}, agent {a_idx}, action {act}: "
                                f"{autograd_grad[b, a_idx, act]}"
                            )
                else:
                    # Ineligible agents must have exactly 0 gradient across all 12 actions
                    assert torch.all(autograd_grad[b, a_idx] == 0.0), (
                        f"Ineligible agent received non-zero gradient at batch {b}, agent {a_idx}"
                    )

    def test_r3_extreme_logits_numerical_stability(self) -> None:
        """Adversarial stress test with extreme logits (+1000, -1000) ensuring no NaN/Inf."""
        B = 4
        num_agents = 5
        logits = torch.zeros((B, num_agents, ACTION_COUNT), dtype=torch.float32, requires_grad=True)

        with torch.no_grad():
            # Case 1: huge positive difference (chosen = 1000, hold = -1000) -> diff = 2000
            logits[0, 0, 1] = 1000.0
            logits[0, 0, 0] = -1000.0
            # Case 2: huge negative difference (chosen = -1000, hold = 1000) -> diff = -2000
            logits[1, 0, 2] = -1000.0
            logits[1, 0, 0] = 1000.0
            # Case 3: moderate large positive (500)
            logits[2, 0, 3] = 500.0
            # Case 4: moderate large negative (-500)
            logits[3, 0, 4] = -500.0

        actions = torch.tensor([[1, 0, 0, 0, 0],
                                [2, 0, 0, 0, 0],
                                [3, 0, 0, 0, 0],
                                [4, 0, 0, 0, 0]], dtype=torch.long)
        masks = torch.tensor([[True, False, False, False, False]] * B)
        weights = torch.ones(B)

        loss, count = FivePPOTrainer.initial_sl_conditional_entry_loss(
            logits, actions, masks, weights
        )
        assert count == 4
        assert torch.isfinite(loss), f"Loss must be finite under extreme logits, got {loss}"
        loss.backward()

        assert torch.all(torch.isfinite(logits.grad)), "Gradients must remain finite under extreme logits"

        # Case 1: sigmoid(2000) = 1.0 -> grad_chosen = 1/4 = 0.25, grad_0 = -0.25
        assert np.isclose(logits.grad[0, 0, 1].item(), 0.25, atol=1e-5)
        assert np.isclose(logits.grad[0, 0, 0].item(), -0.25, atol=1e-5)

        # Case 2: sigmoid(-2000) = 0.0 -> grad_chosen = 0.0, grad_0 = 0.0
        assert np.isclose(logits.grad[1, 0, 2].item(), 0.0, atol=1e-5)
        assert np.isclose(logits.grad[1, 0, 0].item(), 0.0, atol=1e-5)

        # Leakage check under extreme values
        for b in range(B):
            chosen = int(actions[b, 0])
            for act in range(ACTION_COUNT):
                if act != 0 and act != chosen:
                    assert logits.grad[b, 0, act] == 0.0

    def test_r3_all_false_masks_and_zero_weights(self) -> None:
        """Verify fail-safe behavior when masks are all False or weights are zero."""
        B = 10
        logits = torch.randn((B, 5, ACTION_COUNT), dtype=torch.float32, requires_grad=True)
        actions = torch.randint(1, 7, (B, 5), dtype=torch.long)

        # Subtest A: All-false masks
        masks_false = torch.zeros((B, 5), dtype=torch.bool)
        weights = torch.ones(B)

        loss_a, count_a = FivePPOTrainer.initial_sl_conditional_entry_loss(
            logits, actions, masks_false, weights
        )
        assert count_a == 0
        assert loss_a.item() == 0.0
        loss_a.backward()
        assert torch.all(logits.grad == 0.0), "All-false masks must yield strictly 0 gradients"

        # Subtest B: Zero weights with True masks
        logits.grad.zero_()
        masks_true = torch.ones((B, 5), dtype=torch.bool)
        weights_zero = torch.zeros(B)

        loss_b, count_b = FivePPOTrainer.initial_sl_conditional_entry_loss(
            logits, actions, masks_true, weights_zero
        )
        assert count_b == B * 5
        assert loss_b.item() == 0.0
        loss_b.backward()
        assert torch.all(logits.grad == 0.0), "Zero weights must yield strictly 0 gradients"

    def test_r3_ineligible_action_indices(self) -> None:
        """Verify that action 0 (HOLD) and actions > 6 are excluded from SL penalty."""
        B = 6
        logits = torch.randn((B, 5, ACTION_COUNT), dtype=torch.float32, requires_grad=True)
        # Actions: 0 (HOLD), 7 (CLOSE), 8, 9, 10, 11 (management/auxiliary)
        ineligible_actions = torch.tensor([
            [0, 0, 0, 0, 0],
            [7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9],
            [10, 10, 10, 10, 10],
            [11, 11, 11, 11, 11],
        ], dtype=torch.long)
        masks = torch.ones((B, 5), dtype=torch.bool)
        weights = torch.ones(B)

        loss, count = FivePPOTrainer.initial_sl_conditional_entry_loss(
            logits, ineligible_actions, masks, weights
        )
        assert count == 0, f"Expected 0 eligible actions, got {count}"
        assert loss.item() == 0.0
        loss.backward()
        assert torch.all(logits.grad == 0.0)


# ==============================================================================
# PART 2: R4 BILINEAR STATE-ACTION REWARD HEAD & DATASET BALANCE GATE
# ==============================================================================


class TestR4BilinearRewardAndBalanceGate:
    """Adversarial stress test for RLHFBilinearRewardModel and rlhf_dataset_balance_gate."""

    def test_r4_bilinear_vs_linear_intra_state_weight_update_proof(self) -> None:
        """Empirical proof: Bilinear model breaks intra-state cancellation bug.

        Legacy Bradley-Terry Linear Model:
            r(s, a) = w^T s + b_a
            On intra-state pairs where s_c = s_r = s:
            r_c - r_r = (w^T s + b_c) - (w^T s + b_r) = b_c - b_r
            d(r_c - r_r) / dw = s - s = 0
            All 137 market feature weights w remain 0.0! (Root Cause 3)

        Remediated Bilinear Model:
            R(s, a) = w_a^T s + b_a
            On intra-state pairs where s_c = s_r = s:
            r_c - r_r = w_c^T s - w_r^T s + b_c - b_r
            d(r_c - r_r) / dw_c = s != 0
            d(r_c - r_r) / dw_r = -s != 0
            Weights W[chosen] and W[rejected] receive non-zero gradients and update away from 0.0!
        """
        np.random.seed(42)
        n_pairs = 60
        pairs: list[RLHFRewardPair] = []

        # Construct 100% intra-state pairs (s_c = s_r = s)
        for i in range(n_pairs):
            state = np.random.randn(137).astype(np.float64)
            # Cycle through action pairs (BUY vs HOLD, SELL vs HOLD, BUY vs SELL)
            if i % 3 == 0:
                c_act, r_act = 1, 0
            elif i % 3 == 1:
                c_act, r_act = 2, 0
            else:
                c_act, r_act = 1, 2

            c_feat = np.zeros(144, dtype=np.float64)
            c_feat[:137] = state
            c_feat[137 + c_act] = 1.0

            r_feat = np.zeros(144, dtype=np.float64)
            r_feat[:137] = state  # EXACT SAME MARKET STATE!
            r_feat[137 + r_act] = 1.0

            pref = RLHFPreferencePair(
                pair_id=f"intra-{i}",
                context_id=f"ctx-{i}",
                chosen_trace_id=f"c-{i}",
                rejected_trace_id=f"r-{i}",
                chosen_action=c_act,
                rejected_action=r_act,
                decision_time="2026-10-17T12:00:00Z",
                feedback_time="2026-10-17T12:01:00Z",
                pre_decision_state_hash=f"h-{i}",
                risk_geometry_hash="g1",
                label_source="teacher",
                rater_id="challenger_2",
                rater_version="v1",
                confidence=1.0,
            )
            pairs.append(
                RLHFRewardPair(
                    pair=pref,
                    chosen_features=tuple(float(x) for x in c_feat),
                    rejected_features=tuple(float(x) for x in r_feat),
                    feature_schema="causal_state_137_v1",
                )
            )

        # 1. Fit Bilinear Reward Model
        bilinear_model, report = fit_rlhf_bilinear_reward_model(
            pairs, epochs=50, learning_rate=0.05, l2=1e-3
        )
        W_bilinear = bilinear_model.weights_array
        non_zero_market_weights = report["market_weights_non_zero_count"]

        # Market feature weights must be non-zero and updated
        assert non_zero_market_weights > 0, "Bilinear model market weights must not collapse to 0"
        assert np.count_nonzero(W_bilinear[1] != 0.0) == 137, "BUY weights W[1] must receive non-zero updates"
        assert np.count_nonzero(W_bilinear[2] != 0.0) == 137, "SELL weights W[2] must receive non-zero updates"
        assert np.count_nonzero(W_bilinear[0] != 0.0) == 137, "HOLD weights W[0] must receive non-zero updates"

        # 2. Simulate Legacy Linear Model under identical intra-state pairs
        # In legacy linear model: grad_w = signal * (s_c - s_r) = signal * 0 = 0
        w_legacy = np.zeros(137, dtype=np.float64)
        for _ in range(50):
            grad_w_legacy = np.zeros(137, dtype=np.float64)
            for pair in pairs:
                c_s = np.asarray(pair.chosen_features[:137])
                r_s = np.asarray(pair.rejected_features[:137])
                grad_w_legacy += (c_s - r_s)  # s_c - s_r is identically 0
            w_legacy += 0.05 * grad_w_legacy

        # Prove that all 137 legacy weights remained at 0.0
        assert np.all(w_legacy == 0.0), "Legacy linear model must exhibit 100% weight collapse (137/137 weights = 0.0)"

    def test_r4_dataset_balance_gate_distribution_matrix(self) -> None:
        """Adversarial testing of rlhf_dataset_balance_gate across distribution extremes."""
        # Helper to build pair mock
        def make_pair(chosen_act: int, rejected_act: int) -> dict[str, Any]:
            return {"chosen_action": chosen_act, "rejected_action": rejected_act}

        # Case 1: 100% BUY
        ds_100_buy = [make_pair(1, 0) for _ in range(100)]
        res = rlhf_dataset_balance_gate(ds_100_buy)
        assert res["passed"] is False
        assert "action_distribution_asymmetric" in res["reasons"]
        assert res["action_probabilities"]["BUY"] == 1.0
        assert res["action_probabilities"]["SELL"] == 0.0
        assert res["action_probabilities"]["HOLD"] == 0.0
        assert res["min_action_probability"] == 0.0

        # Case 2: 100% SELL
        ds_100_sell = [make_pair(2, 0) for _ in range(100)]
        res = rlhf_dataset_balance_gate(ds_100_sell)
        assert res["passed"] is False
        assert "action_distribution_asymmetric" in res["reasons"]
        assert res["action_probabilities"]["SELL"] == 1.0
        assert res["min_action_probability"] == 0.0

        # Case 3: 100% HOLD
        ds_100_hold = [make_pair(0, 1) for _ in range(100)]
        res = rlhf_dataset_balance_gate(ds_100_hold)
        assert res["passed"] is False
        assert "action_distribution_asymmetric" in res["reasons"]
        assert res["action_probabilities"]["HOLD"] == 1.0
        assert res["min_action_probability"] == 0.0

        # Case 4: 50% BUY / 50% SELL (0% HOLD)
        ds_50_50 = [make_pair(1, 0) for _ in range(50)] + [make_pair(2, 0) for _ in range(50)]
        res = rlhf_dataset_balance_gate(ds_50_50)
        assert res["passed"] is False
        assert "action_distribution_asymmetric" in res["reasons"]
        assert res["action_probabilities"]["BUY"] == 0.50
        assert res["action_probabilities"]["SELL"] == 0.50
        assert res["action_probabilities"]["HOLD"] == 0.0
        assert res["min_action_probability"] == 0.0

        # Case 5: 33% BUY / 33% SELL / 34% HOLD
        ds_balanced = (
            [make_pair(1, 0) for _ in range(33)]
            + [make_pair(2, 0) for _ in range(33)]
            + [make_pair(0, 1) for _ in range(34)]
        )
        res = rlhf_dataset_balance_gate(ds_balanced)
        assert res["passed"] is True
        assert res["reasons"] == []
        assert res["action_probabilities"]["BUY"] == 0.33
        assert res["action_probabilities"]["SELL"] == 0.33
        assert res["action_probabilities"]["HOLD"] == 0.34
        assert res["min_action_probability"] == 0.33 >= 0.20

        # Case 6: Exact 20% boundary acceptance (20% BUY, 20% SELL, 60% HOLD)
        ds_boundary_pass = (
            [make_pair(1, 0) for _ in range(20)]
            + [make_pair(2, 0) for _ in range(20)]
            + [make_pair(0, 1) for _ in range(60)]
        )
        res = rlhf_dataset_balance_gate(ds_boundary_pass)
        assert res["passed"] is True
        assert res["min_action_probability"] == 0.20

        # Case 7: Strict boundary rejection (19.9% BUY, 40% SELL, 40.1% HOLD -> 199/400/401 out of 1000)
        ds_boundary_fail = (
            [make_pair(1, 0) for _ in range(199)]
            + [make_pair(2, 0) for _ in range(400)]
            + [make_pair(0, 1) for _ in range(401)]
        )
        res = rlhf_dataset_balance_gate(ds_boundary_fail)
        assert res["passed"] is False
        assert "action_distribution_asymmetric" in res["reasons"]
        assert res["min_action_probability"] == 0.199 < 0.20


# ==============================================================================
# PART 3: R6 TEACHER LR ALIGNMENT & GRADIENT BOUNDS
# ==============================================================================


class TestR6TeacherLRAndGradientBounds:
    """Adversarial stress testing of Teacher LR validator and step_from_loss grad clipping."""

    def test_r6_config_validator_teacher_lr_boundaries(self) -> None:
        """Verify LearningIntegrityConfig enforces 0.0 < five_codex_teacher_learning_rate <= 0.001."""
        # 1. Negative LR rejected
        with pytest.raises(ValueError, match="learning.five_codex_teacher_learning_rate must be within"):
            LearningIntegrityConfig(five_codex_teacher_learning_rate=-0.0001)

        # 2. Zero LR rejected
        with pytest.raises(ValueError, match="learning.five_codex_teacher_learning_rate must be within"):
            LearningIntegrityConfig(five_codex_teacher_learning_rate=0.0)

        # 3. Excessive LR > 0.001 rejected (including legacy unaligned 0.02)
        with pytest.raises(ValueError, match="learning.five_codex_teacher_learning_rate must be within"):
            LearningIntegrityConfig(five_codex_teacher_learning_rate=0.02)

        with pytest.raises(ValueError, match="learning.five_codex_teacher_learning_rate must be within"):
            LearningIntegrityConfig(five_codex_teacher_learning_rate=0.0010001)

        # 4. Valid defaults and boundary values accepted
        cfg_default = LearningIntegrityConfig()
        assert cfg_default.five_codex_teacher_learning_rate == 0.0003

        cfg_boundary = LearningIntegrityConfig(five_codex_teacher_learning_rate=0.001)
        assert cfg_boundary.five_codex_teacher_learning_rate == 0.001

        cfg_lower = LearningIntegrityConfig(five_codex_teacher_learning_rate=1e-6)
        assert cfg_lower.five_codex_teacher_learning_rate == 1e-6

    def test_r6_step_from_loss_grad_norm_clipping_under_loss_explosion(self) -> None:
        """Verify policy step_from_loss enforces grad norm clipping <= 1.0 under 1e8 loss explosion."""
        torch.manual_seed(42)
        net = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
        trainer = FivePPOTrainer(net)

        # Adversarial loss explosion: multiply raw parameter sum by 10^8
        # Without gradient clipping, gradient norms would explode to > 10^8
        dummy_loss = sum(p.sum() for p in net.parameters()) * 1e8

        stats = trainer.step_from_loss(dummy_loss)
        assert isinstance(stats, dict)

        # Compute total L2 norm of clipped gradients across all network parameters
        total_norm_sq = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm_sq += float(p.grad.detach().norm(2).item() ** 2)
        total_norm = math.sqrt(total_norm_sq)

        # Must be strictly bounded by max_norm = 1.0 (with minimal floating point margin)
        assert total_norm <= 1.0001, (
            f"Gradient norm clipping failed under loss explosion! Observed total norm: {total_norm:.4f} > 1.0"
        )
        assert total_norm > 0.99, (
            f"Under extreme explosion, clipped norm should saturate near 1.0, got {total_norm:.4f}"
        )

    def test_r6_step_from_loss_under_zero_loss(self) -> None:
        """Verify policy step_from_loss safely handles 0.0 loss without NaN or division by zero."""
        net = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
        trainer = FivePPOTrainer(net)

        zero_loss = sum(p.sum() for p in net.parameters()) * 0.0
        stats = trainer.step_from_loss(zero_loss)
        assert isinstance(stats, dict)

        for p in net.parameters():
            if p.grad is not None:
                assert torch.all(p.grad == 0.0)
                assert not torch.any(torch.isnan(p.grad))
