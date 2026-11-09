"""
Empirical Challenger Test Suite for Milestone M1 (Reward Reform & Loss Implementation).
Author: challenger_m1_reward_1

Adversarial stress-testing of:
1. Trade Reward Mathematical Invariants (TP, SL, Wick, Breakeven WR, Expectancy).
2. HOLD Opportunity Cost & Discipline Bounds across Diverse Market Regimes.
3. Policy Entropy Loss Differentiability, Extreme Limits, and Gradient Flow.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from research_reasoning_world_class.src.latent_dynamics_world_model import LatentDynamicsWorldModel
from research_reasoning_world_class.src.muzero_latent_coevolution import ChampionRole
from scripts.train_mythos_muzero_specialized_massive import (
    compute_trade_reward,
    compute_hold_reward,
    compute_policy_entropy_loss,
)


# =============================================================================
# MOCK / HELPER OBJECTS
# =============================================================================
@dataclass
class SyntheticTradeResult:
    return_atr: float
    exit_type: str
    holding_bars: int
    exit_reason: str = ""


class SyntheticMarketEnvironment:
    """Mock environment allowing controlled candle price paths for HOLD stress-testing."""
    def __init__(self, close_prices: List[float], atr_values: Optional[List[float]] = None):
        self.close_arr = np.array(close_prices, dtype=np.float32)
        if atr_values is None:
            self.atr_arr = np.ones_like(self.close_arr, dtype=np.float32) * 100.0  # ATR = $100
        else:
            self.atr_arr = np.array(atr_values, dtype=np.float32)
        self.total_bars = len(self.close_arr)


class MockChampion:
    """Mock Champion containing LatentDynamicsWorldModel for policy entropy tests."""
    def __init__(self, role: ChampionRole, obs_dim: int = 137, latent_dim: int = 128, action_dim: int = 64):
        self.role = role
        self.world_model = LatentDynamicsWorldModel(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )


# =============================================================================
# TEST SUITE 1: TRADE REWARD MATHEMATICAL INVARIANTS
# =============================================================================
class TestTradeRewardInvariants:
    """Empirical verification of Pillar 1 & Pillar 3 mathematical properties."""

    def test_pillar1_tp_reward_values(self):
        """Verify TP (1:2 RR) yields exactly tanh(+1.0) = +0.76159... (+0.76)."""
        expected_exact = math.tanh(1.0)
        assert round(expected_exact, 2) == 0.76

        sl_test_cases = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0]
        for sl in sl_test_cases:
            tp_return_atr = 2.0 * sl  # 1:2 RR
            res = SyntheticTradeResult(return_atr=tp_return_atr, exit_type="tp", holding_bars=10)
            reward = compute_trade_reward(res, sl_atr=sl)
            assert math.isclose(reward, expected_exact, rel_tol=1e-5), f"Failed at sl={sl}: {reward}"
            assert round(reward, 2) == 0.76

    def test_pillar1_normal_sl_reward_values(self):
        """Verify normal SL (holding > 2 bars) yields exactly tanh(-0.5) = -0.46211... (-0.46)."""
        expected_exact = math.tanh(-0.5)
        assert round(expected_exact, 2) == -0.46

        holding_bars_cases = [3, 4, 5, 10, 50, 120]
        sl_test_cases = [0.5, 1.0, 2.0, 3.0, 5.0]
        for bars in holding_bars_cases:
            for sl in sl_test_cases:
                sl_return_atr = -1.0 * sl
                res = SyntheticTradeResult(return_atr=sl_return_atr, exit_type="sl", holding_bars=bars)
                reward = compute_trade_reward(res, sl_atr=sl)
                assert math.isclose(reward, expected_exact, rel_tol=1e-5), f"Failed at bars={bars}, sl={sl}: {reward}"
                assert round(reward, 2) == -0.46

    def test_pillar3_wick_fragility_penalty_values(self):
        """Verify rapid SL within <= 2 bars receives -0.15 penalty -> tanh(-0.5) - 0.15 = -0.61211... (-0.61)."""
        expected_exact = math.tanh(-0.5) - 0.15
        assert round(expected_exact, 2) == -0.61

        wick_bars = [0, 1, 2]
        exit_types = ["sl", "SL", "sl_hit", "SL_HIT"]
        for bars in wick_bars:
            for exit_t in exit_types:
                res = SyntheticTradeResult(return_atr=-3.0, exit_type=exit_t, holding_bars=bars)
                reward = compute_trade_reward(res, sl_atr=3.0)
                assert math.isclose(reward, expected_exact, rel_tol=1e-5), f"Failed for bars={bars}, exit={exit_t}"
                assert round(reward, 2) == -0.61

        # Test exit_reason trigger
        res_reason = SyntheticTradeResult(return_atr=-2.0, exit_type="timeout", holding_bars=2, exit_reason="sl_triggered")
        reward_reason = compute_trade_reward(res_reason, sl_atr=2.0)
        assert math.isclose(reward_reason, expected_exact, rel_tol=1e-5)

        # Negative test: TP in <= 2 bars must NOT receive wick penalty
        res_quick_tp = SyntheticTradeResult(return_atr=6.0, exit_type="tp", holding_bars=1)
        reward_quick_tp = compute_trade_reward(res_quick_tp, sl_atr=3.0)
        assert math.isclose(reward_quick_tp, math.tanh(1.0), rel_tol=1e-5)
        assert round(reward_quick_tp, 2) == 0.76

        # Negative test: SL at holding_bars = 3 must NOT receive wick penalty
        res_bar3 = SyntheticTradeResult(return_atr=-3.0, exit_type="sl", holding_bars=3)
        reward_bar3 = compute_trade_reward(res_bar3, sl_atr=3.0)
        assert math.isclose(reward_bar3, math.tanh(-0.5), rel_tol=1e-5)
        assert round(reward_bar3, 2) == -0.46

    def test_breakeven_win_rate_and_expectation_curve(self):
        """Verify breakeven win rate is 37.8% and E > 0 for WR >= 40%, E >> 0 for WR >= 50%."""
        r_tp = math.tanh(1.0)      # +0.761594
        r_sl = math.tanh(-0.5)     # -0.462117

        # Breakeven equation: W * r_tp + (1 - W) * r_sl = 0 => W * (r_tp - r_sl) = -r_sl
        w_be = -r_sl / (r_tp - r_sl)
        assert round(w_be * 100, 1) == 37.8, f"Breakeven WR should be 37.8%, got {w_be * 100}%"

        # Mathematical expectation at key win rates
        e_35 = 0.35 * r_tp + 0.65 * r_sl
        e_37_8 = w_be * r_tp + (1 - w_be) * r_sl
        e_40 = 0.40 * r_tp + 0.60 * r_sl
        e_50 = 0.50 * r_tp + 0.50 * r_sl
        e_60 = 0.60 * r_tp + 0.40 * r_sl

        assert e_35 < 0.0, f"Expectancy at 35% should be negative: {e_35}"
        assert math.isclose(e_37_8, 0.0, abs_tol=1e-7), f"Expectancy at breakeven should be 0.0: {e_37_8}"
        assert e_40 > 0.02, f"Expectancy at 40% should be positive: {e_40}"
        assert e_50 > 0.14, f"Expectancy at 50% should be strongly positive: {e_50}"
        assert e_60 > 0.27, f"Expectancy at 60% should be highly positive: {e_60}"

    def test_adversarial_trade_reward_inputs(self):
        """Stress-test trade reward with extreme, tiny, or malformed inputs."""
        # Non-positive or tiny SL should be clamped to sl_atr_safe >= 0.5
        res = SyntheticTradeResult(return_atr=1.0, exit_type="tp", holding_bars=5)
        for bad_sl in [0.0, -1.0, 0.0001, 0.499]:
            rew = compute_trade_reward(res, sl_atr=bad_sl)
            # R = 1.0 / 0.5 = 2.0 -> tanh(2.0 / 2.0) = tanh(1.0)
            assert math.isclose(rew, math.tanh(1.0), rel_tol=1e-5), f"Failed for bad_sl={bad_sl}"

        # Extreme huge positive return
        res_huge_pos = SyntheticTradeResult(return_atr=10000.0, exit_type="tp", holding_bars=5)
        rew_huge_pos = compute_trade_reward(res_huge_pos, sl_atr=2.0)
        assert 0.999 < rew_huge_pos <= 1.0

        # Extreme huge negative return
        res_huge_neg = SyntheticTradeResult(return_atr=-10000.0, exit_type="sl", holding_bars=5)
        rew_huge_neg = compute_trade_reward(res_huge_neg, sl_atr=2.0)
        assert -1.0 <= rew_huge_neg < -0.999


# =============================================================================
# TEST SUITE 2: HOLD REWARD BOUNDS ACROSS DIVERSE SCENARIOS
# =============================================================================
class TestHoldRewardScenarios:
    """Empirical verification of Pillar 2 HOLD opportunity cost & discipline bounds."""

    def test_breakout_extreme_penalty_bounded(self):
        """Verify extreme breakouts produce bounded opportunity penalty (-0.15)."""
        # Bull: Price moves up 10 ATR over 30 bars (now=1000, fut=2000, atr=100)
        prices = [1000.0] * 31
        prices[30] = 2000.0  # +10 ATR
        env = SyntheticMarketEnvironment(prices, atr_values=[100.0] * 31)

        bull_rew = compute_hold_reward(env, bar_idx=0, role=ChampionRole.BULL, lookahead_bars=30)
        assert math.isclose(bull_rew, -0.15, rel_tol=1e-5), f"Bull penalty should be -0.15, got {bull_rew}"

        # Bear: Price crashes 10 ATR over 30 bars (now=2000, fut=1000, atr=100)
        # For Bear, a crash is in favored direction, so HOLD misses opportunity -> penalty -0.15
        prices_bear = [2000.0] * 31
        prices_bear[30] = 1000.0  # -10 ATR
        env_bear = SyntheticMarketEnvironment(prices_bear, atr_values=[100.0] * 31)

        bear_rew = compute_hold_reward(env_bear, bar_idx=0, role=ChampionRole.BEAR, lookahead_bars=30)
        assert math.isclose(bear_rew, -0.15, rel_tol=1e-5), f"Bear penalty should be -0.15, got {bear_rew}"

    def test_crash_and_trap_avoidance_rewards(self):
        """Verify trap avoidance awards positive reward (+0.15)."""
        # Bull dodges a -5 ATR dump
        prices = [1500.0] * 31
        prices[30] = 1000.0  # -5 ATR
        env = SyntheticMarketEnvironment(prices, atr_values=[100.0] * 31)

        bull_rew = compute_hold_reward(env, bar_idx=0, role=ChampionRole.BULL, lookahead_bars=30)
        assert math.isclose(bull_rew, +0.15, rel_tol=1e-5), f"Bull trap avoidance should be +0.15, got {bull_rew}"

        # Bear dodges a +5 ATR pump
        prices_bear = [1000.0] * 31
        prices_bear[30] = 1500.0  # +5 ATR
        env_bear = SyntheticMarketEnvironment(prices_bear, atr_values=[100.0] * 31)

        bear_rew = compute_hold_reward(env_bear, bar_idx=0, role=ChampionRole.BEAR, lookahead_bars=30)
        assert math.isclose(bear_rew, +0.15, rel_tol=1e-5), f"Bear trap avoidance should be +0.15, got {bear_rew}"

    def test_chop_discipline_patience_reward(self):
        """Verify zero and minimal movement (< 0.5 ATR) awards patience reward (+0.05)."""
        # Exact zero movement
        prices_zero = [1000.0] * 31
        env_zero = SyntheticMarketEnvironment(prices_zero, atr_values=[100.0] * 31)

        bull_zero = compute_hold_reward(env_zero, bar_idx=0, role=ChampionRole.BULL, lookahead_bars=30)
        bear_zero = compute_hold_reward(env_zero, bar_idx=0, role=ChampionRole.BEAR, lookahead_bars=30)
        assert math.isclose(bull_zero, +0.05, rel_tol=1e-5)
        assert math.isclose(bear_zero, +0.05, rel_tol=1e-5)

        # Small movement: +0.3 ATR
        prices_chop = [1000.0] * 31
        prices_chop[30] = 1030.0  # +0.3 ATR
        env_chop = SyntheticMarketEnvironment(prices_chop, atr_values=[100.0] * 31)

        bull_chop = compute_hold_reward(env_chop, bar_idx=0, role=ChampionRole.BULL, lookahead_bars=30)
        bear_chop = compute_hold_reward(env_chop, bar_idx=0, role=ChampionRole.BEAR, lookahead_bars=30)
        assert math.isclose(bull_chop, +0.05, rel_tol=1e-5)
        assert math.isclose(bear_chop, +0.05, rel_tol=1e-5)

    def test_boundary_transitions_0_5_and_1_5_atr(self):
        """Verify behavior across transition thresholds: 0.5 ATR and 1.5 ATR."""
        atr = 100.0
        # 0.49 ATR -> patience +0.05
        p = [1000.0] * 31; p[30] = 1000.0 + 0.49 * atr
        rew_049 = compute_hold_reward(SyntheticMarketEnvironment(p, [atr]*31), 0, ChampionRole.BULL, 30)
        assert math.isclose(rew_049, +0.05, rel_tol=1e-5)

        # 0.50 ATR -> neutral drift 0.0
        p = [1000.0] * 31; p[30] = 1000.0 + 0.50 * atr
        rew_050 = compute_hold_reward(SyntheticMarketEnvironment(p, [atr]*31), 0, ChampionRole.BULL, 30)
        assert math.isclose(rew_050, 0.0, abs_tol=1e-5)

        # 1.00 ATR -> neutral drift 0.0
        p = [1000.0] * 31; p[30] = 1000.0 + 1.00 * atr
        rew_100 = compute_hold_reward(SyntheticMarketEnvironment(p, [atr]*31), 0, ChampionRole.BULL, 30)
        assert math.isclose(rew_100, 0.0, abs_tol=1e-5)

        # 1.50 ATR -> boundary (not > 1.5) -> 0.0
        p = [1000.0] * 31; p[30] = 1000.0 + 1.50 * atr
        rew_150 = compute_hold_reward(SyntheticMarketEnvironment(p, [atr]*31), 0, ChampionRole.BULL, 30)
        assert math.isclose(rew_150, 0.0, abs_tol=1e-5)

        # 1.80 ATR -> penalty -0.15 * (1.8 / 2.0) = -0.135
        p = [1000.0] * 31; p[30] = 1000.0 + 1.80 * atr
        rew_180 = compute_hold_reward(SyntheticMarketEnvironment(p, [atr]*31), 0, ChampionRole.BULL, 30)
        assert math.isclose(rew_180, -0.135, rel_tol=1e-5)

        # -0.50 ATR -> boundary -> 0.0
        p = [1000.0] * 31; p[30] = 1000.0 - 0.50 * atr
        rew_neg050 = compute_hold_reward(SyntheticMarketEnvironment(p, [atr]*31), 0, ChampionRole.BULL, 30)
        assert math.isclose(rew_neg050, 0.0, abs_tol=1e-5)

        # -0.51 ATR -> trap avoidance -> +0.15
        p = [1000.0] * 31; p[30] = 1000.0 - 0.51 * atr
        rew_neg051 = compute_hold_reward(SyntheticMarketEnvironment(p, [atr]*31), 0, ChampionRole.BULL, 30)
        assert math.isclose(rew_neg051, +0.15, rel_tol=1e-5)

    def test_hold_reward_edge_cases_and_end_of_dataset(self):
        """Verify graceful 0.0 return at end of dataset, empty env, or missing bars."""
        # Empty environment
        env_empty = SyntheticMarketEnvironment([], [])
        assert compute_hold_reward(env_empty, 0, ChampionRole.BULL) == 0.0

        # Exactly at last bar
        prices = [100.0, 101.0, 102.0]
        env_short = SyntheticMarketEnvironment(prices, [1.0, 1.0, 1.0])
        assert compute_hold_reward(env_short, bar_idx=2, role=ChampionRole.BULL) == 0.0
        assert compute_hold_reward(env_short, bar_idx=5, role=ChampionRole.BULL) == 0.0

        # Near end (bar_idx = 1, lookahead = 30 -> clamps to bar 2)
        # bar 1 price = 101, bar 2 price = 102 -> delta = +1.0 ATR (neutral drift 0.0)
        rew_near_end = compute_hold_reward(env_short, bar_idx=1, role=ChampionRole.BULL, lookahead_bars=30)
        assert isinstance(rew_near_end, float)


# =============================================================================
# TEST SUITE 3: POLICY ENTROPY DIFFERENTIABILITY & EXTREME LIMITS
# =============================================================================
class TestPolicyEntropyLoss:
    """Empirical verification of Policy Entropy Loss mathematical and gradient properties."""

    def test_policy_entropy_at_maximum_entropy(self):
        """When Q-values are equal, pi=[0.5, 0.5], H=ln(2), entropy_loss = 0.0."""
        # Construct synthetic Q tensor where Q(0) == Q(1)
        q = torch.tensor([[1.0, 1.0]], dtype=torch.float32, requires_grad=True)
        pi_soft = F.softmax(q, dim=-1)
        entropy = -torch.sum(pi_soft * torch.log(pi_soft + 1e-8), dim=-1).mean()
        max_ent = math.log(2.0)
        entropy_loss = 0.01 * (max_ent - entropy)

        assert math.isclose(float(entropy.item()), max_ent, rel_tol=1e-5)
        assert math.isclose(float(entropy_loss.item()), 0.0, abs_tol=1e-7)

        # Gradient at maximum entropy should be zero
        entropy_loss.backward()
        assert torch.allclose(q.grad, torch.zeros_like(q), atol=1e-6)

    def test_policy_entropy_at_zero_entropy_limit(self):
        """When Q-values collapse to deterministic, pi=[1, 0], H=0, entropy_loss = 0.01 * ln(2)."""
        q = torch.tensor([[50.0, -50.0]], dtype=torch.float32, requires_grad=True)
        pi_soft = F.softmax(q, dim=-1)
        entropy = -torch.sum(pi_soft * torch.log(pi_soft + 1e-8), dim=-1).mean()
        max_ent = math.log(2.0)
        entropy_loss = 0.01 * (max_ent - entropy)

        expected_max_loss = 0.01 * max_ent  # ~0.0069315
        assert math.isclose(float(entropy.item()), 0.0, abs_tol=1e-4)
        assert math.isclose(float(entropy_loss.item()), expected_max_loss, rel_tol=1e-3)
        assert float(entropy_loss.item()) > 0.0

    def test_policy_entropy_gradient_flow_to_world_model(self):
        """Verify gradient flows from policy entropy loss into World Model parameters."""
        bull_champ = MockChampion(ChampionRole.BULL)
        bear_champ = MockChampion(ChampionRole.BEAR)

        # Bull Champion test
        z_0 = torch.randn(1, 128, requires_grad=True)
        ent_loss, ent_val = compute_policy_entropy_loss(bull_champ, z_0, sl_val=3.0, tp_val=6.0)

        assert isinstance(ent_loss, torch.Tensor)
        assert ent_loss.requires_grad
        assert 0.0 <= ent_val <= math.log(2.0) + 1e-5

        # Check gradients backpropagate into world model
        bull_champ.world_model.zero_grad()
        ent_loss.backward()

        wm_params_with_grad = [p for p in bull_champ.world_model.parameters() if p.grad is not None]
        assert len(wm_params_with_grad) > 0, "No World Model parameters received gradients from entropy loss!"

        # Specific submodule gradient checks:
        # 1. value_head must receive grad
        val_head_grads = [p.grad for p in bull_champ.world_model.value_head.parameters() if p.grad is not None]
        assert len(val_head_grads) > 0, "value_head did not receive gradient!"

        # 2. reward_head must receive grad
        rew_head_grads = [p.grad for p in bull_champ.world_model.reward_head.parameters() if p.grad is not None]
        assert len(rew_head_grads) > 0, "reward_head did not receive gradient!"

        # 3. g_net (transition) must receive grad
        g_net_grads = [p.grad for p in bull_champ.world_model.g_net.parameters() if p.grad is not None]
        assert len(g_net_grads) > 0, "g_net transition did not receive gradient!"

        # Bear Champion test
        z_bear = torch.randn(1, 128, requires_grad=True)
        bear_champ.world_model.zero_grad()
        bear_loss, bear_val = compute_policy_entropy_loss(bear_champ, z_bear, sl_val=2.5, tp_val=5.0)
        bear_loss.backward()

        bear_params_with_grad = [p for p in bear_champ.world_model.parameters() if p.grad is not None]
        assert len(bear_params_with_grad) > 0, "Bear World Model did not receive gradients!"

    def test_policy_entropy_gradient_direction_anti_collapse(self):
        """Verify gradient direction actively pushes against probability collapse."""
        # Suppose action 0 has dominant Q (e.g. Q0=5.0, Q1=-5.0 -> collapsed into HOLD)
        q = torch.tensor([[5.0, -5.0]], dtype=torch.float32, requires_grad=True)
        pi_soft = F.softmax(q, dim=-1)
        entropy = -torch.sum(pi_soft * torch.log(pi_soft + 1e-8), dim=-1).mean()
        loss = 0.01 * (math.log(2.0) - entropy)

        loss.backward()
        # Gradient descent updates: q_new = q - lr * grad
        # To restore entropy, q[0] should DECREASE (grad > 0) and q[1] should INCREASE (grad < 0)
        grad_0 = q.grad[0, 0].item()
        grad_1 = q.grad[0, 1].item()

        assert grad_0 > 0.0, f"Expected grad_0 > 0 to penalize dominant action, got {grad_0}"
        assert grad_1 < 0.0, f"Expected grad_1 < 0 to encourage suppressed action, got {grad_1}"
        assert math.isclose(grad_0 + grad_1, 0.0, abs_tol=1e-6), "Softmax gradient sum must equal 0"

    def test_policy_entropy_input_shapes_robustness(self):
        """Verify compute_policy_entropy_loss handles [D], [1, D], [B, D] robustly."""
        champ = MockChampion(ChampionRole.BULL)

        # 1D tensor [128]
        z_1d = torch.randn(128)
        loss_1d, val_1d = compute_policy_entropy_loss(champ, z_1d, 3.0, 6.0)
        assert loss_1d.ndim == 0

        # 2D batch [1, 128]
        z_2d = torch.randn(1, 128)
        loss_2d, val_2d = compute_policy_entropy_loss(champ, z_2d, 3.0, 6.0)
        assert loss_2d.ndim == 0

        # Multi-batch [4, 128]
        z_multi = torch.randn(4, 128)
        loss_multi, val_multi = compute_policy_entropy_loss(champ, z_multi, 3.0, 6.0)
        assert loss_multi.ndim == 0
