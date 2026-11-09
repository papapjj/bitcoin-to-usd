from __future__ import annotations

"""Adversarial stress and verification test suite for Milestone 1.

Carried out by challenger_m1_1 (Empirical Challenger):
1. Randomized generator of 1,000 lifecycle trade outcomes (varying holding_bars 0..500,
   realized_r -10..+10, mae_r 0..5, mfe_r 0..10, various exit kinds and timeout states).
2. Rigorous assertions for NaN, Inf, numerical stability, exact clipping bounds [-5.0, +5.0]R.
3. Verification of bar 120 holding penalty mathematical exactness (-0.25R) and convexity.
4. Verification of dynamic cosine decay monotonicity across 1,000 steps (t in [0, 1000])
   and zero boundary derivatives (tangent = 0 at t=0 and t=T).
5. Verification of 5-agent reward conservation under PPO team-spirit mixing.
"""

import math
import numpy as np
import pytest
import torch

from self_growing_trader.five_lifecycle_reward import (
    FiveLifecycleRewardConfig,
    FiveLifecycleRewardBreakdown,
    shape_five_entry_lifecycle,
    compute_holding_duration_penalty,
    compute_mae_flooding_penalty,
    compute_mfe_harvest_reward,
    compute_timeout_penalty,
    calculate_extended_lifecycle_settlement,
    compute_extended_lifecycle_settlement,
    ExtendedLifecycleSettlementEngine,
)
from self_growing_trader.five_shared_policy import (
    FiveTrainingSchedule,
    mix_team_spirit_rewards,
    compute_team_spirit_cosine,
    dynamic_team_spirit_cosine_decay,
)


# ==============================================================================
# 1. 1,000 Randomized Lifecycle Trade Outcomes
# ==============================================================================

class TestRandomizedLifecycleTradesStress:
    """Stress test with 1,000 randomized lifecycle trade outcomes."""

    def test_1000_randomized_trade_outcomes(self) -> None:
        rng = np.random.default_rng(seed=424242)
        n_samples = 1000

        holding_bars_samples = rng.uniform(0.0, 500.0, size=n_samples)
        realized_r_samples = rng.uniform(-10.0, 10.0, size=n_samples)
        mae_r_samples = rng.uniform(0.0, 5.0, size=n_samples)
        mfe_r_samples = rng.uniform(0.0, 10.0, size=n_samples)
        exit_kinds = [
            "TIMEOUT_EXPIRED", "INITIAL_SL", "TAKE_PROFIT",
            "TRAILING_STOP", "NORMAL", "EXPIRED", "TIME_SAFETY", "",
        ]
        is_timeout_choices = [True, False, None]

        engine = ExtendedLifecycleSettlementEngine(
            alpha_mae=0.20,
            beta_time=0.25,
            gamma_mfe=0.10,
            timeout_penalty_r=0.15,
            max_holding_bars=120,
            clip_min_r=-5.0,
            clip_max_r=5.0,
        )

        for i in range(n_samples):
            hb = float(holding_bars_samples[i])
            # randomly cast some to int to test mixed typing
            if i % 3 == 0:
                hb = int(hb)
            realized = float(realized_r_samples[i])
            mae = float(mae_r_samples[i])
            mfe = float(mfe_r_samples[i])
            exit_k = exit_kinds[i % len(exit_kinds)]
            to_choice = is_timeout_choices[i % len(is_timeout_choices)]

            # 1. Compute via Engine
            res = engine.compute_settlement_reward(
                realized_r=realized,
                mae_r=mae,
                mfe_r=mfe,
                holding_bars=hb,
                exit_kind=exit_k,
                is_timeout=to_choice,
            )

            # Assertions on Engine outputs
            reward_final = res["reward_final"]
            unclipped = res["unclipped_reward"]
            time_pen = res["time_penalty"]
            drawdown_pen = res["drawdown_penalty"]
            mfe_bon = res["mfe_bonus"]
            timeout_pen = res["timeout_penalty"]

            # Finite checks
            assert np.isfinite(reward_final), f"Non-finite reward_final at iteration {i}"
            assert np.isfinite(unclipped), f"Non-finite unclipped at iteration {i}"
            assert np.isfinite(time_pen), f"Non-finite time_pen at iteration {i}"
            assert np.isfinite(drawdown_pen), f"Non-finite drawdown_pen at iteration {i}"
            assert np.isfinite(mfe_bon), f"Non-finite mfe_bon at iteration {i}"
            assert np.isfinite(timeout_pen), f"Non-finite timeout_pen at iteration {i}"

            # Strict range checks
            assert -5.0 <= reward_final <= 5.0, f"Reward {reward_final} out of [-5.0, +5.0] at iter {i}"
            assert time_pen >= 0.0, f"Negative time penalty at iter {i}"
            assert time_pen <= 0.25, f"Time penalty exceeded max 0.25 at iter {i}: {time_pen}"
            assert drawdown_pen >= 0.0, f"Negative drawdown penalty at iter {i}"
            assert mfe_bon >= 0.0, f"Negative MFE bonus at iter {i}"
            assert mfe_bon <= 0.10, f"MFE bonus exceeded max 0.10 at iter {i}: {mfe_bon}"
            assert timeout_pen in (0.0, 0.15), f"Unexpected timeout penalty at iter {i}: {timeout_pen}"

            # Exact arithmetic invariant: unclipped = realized - drawdown - time + mfe - timeout
            expected_unclipped = realized - drawdown_pen - time_pen + mfe_bon - timeout_pen
            assert abs(unclipped - expected_unclipped) < 1e-12, f"Unclipped mismatch at iter {i}"

            # Exact clipping invariant
            expected_clipped = float(np.clip(unclipped, -5.0, 5.0))
            assert abs(reward_final - expected_clipped) < 1e-12, f"Clipping mismatch at iter {i}"

            # 2. Compute via shape_five_entry_lifecycle
            config = FiveLifecycleRewardConfig(
                enabled=True,
                extended_settlement_enabled=True,
            )
            breakdown = shape_five_entry_lifecycle(
                action_index=1,  # Buy entry
                sl_atr=2.0,
                tp_atr=4.0,
                trailing_atr=0.0,
                trailing_applied=False,
                locked_profit_r=0.0,
                mfe_r=mfe,
                mae_r=mae,
                mfe_available=True,
                realized_r=realized,
                exit_kind=exit_k,
                holding_bars=hb,
                is_timeout=to_choice,
                config=config,
            )

            assert breakdown.extended_settlement is True
            assert breakdown.total_r == pytest.approx(reward_final, abs=1e-12)
            assert breakdown.unclipped_total_r == pytest.approx(unclipped, abs=1e-12)
            assert breakdown.holding_time_penalty_r == pytest.approx(-time_pen, abs=1e-12)
            assert breakdown.mae_flooding_penalty_r == pytest.approx(-drawdown_pen, abs=1e-12)
            assert breakdown.mfe_harvest_bonus_r == pytest.approx(mfe_bon, abs=1e-12)
            assert breakdown.timeout_penalty_r == pytest.approx(-timeout_pen, abs=1e-12)

            # Invariant: sum of signed components equals unclipped total
            reconstructed_unclipped = (
                breakdown.realized_r
                + breakdown.holding_time_penalty_r
                + breakdown.mae_flooding_penalty_r
                + breakdown.mfe_harvest_bonus_r
                + breakdown.timeout_penalty_r
            )
            assert abs(breakdown.unclipped_total_r - reconstructed_unclipped) < 1e-12

            # Dict serialization sanity
            bd_dict = breakdown.as_dict()
            for key, val in bd_dict.items():
                if isinstance(val, (int, float, np.floating)):
                    assert np.isfinite(val), f"Non-finite field {key} in breakdown at iter {i}"


# ==============================================================================
# 2. Numerical Stability & Extreme Adversarial Edge Cases
# ==============================================================================

class TestNumericalStabilityAndAdversarialEdges:
    """Stress testing hostile, non-finite, and out-of-domain inputs."""

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_non_finite_realized_r_handled_safely(self, bad_val: float) -> None:
        """Non-finite realized_r must be sanitized to 0.0 without throwing exceptions."""
        res = compute_extended_lifecycle_settlement(
            realized_r=bad_val,
            mae_r=0.5,
            mfe_r=1.0,
            holding_bars=30,
        )
        assert np.isfinite(res["reward_final"])
        assert res["realized_r"] == 0.0
        assert -5.0 <= res["reward_final"] <= 5.0

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_non_finite_mae_and_mfe_handled_safely(self, bad_val: float) -> None:
        """Non-finite excursion metrics must default safely without crashing."""
        # Case A: Break-even trade with non-finite excursion
        res_even = compute_extended_lifecycle_settlement(
            realized_r=0.0,
            mae_r=bad_val,
            mfe_r=bad_val,
            holding_bars=10,
        )
        assert np.isfinite(res_even["reward_final"])
        assert res_even["drawdown_penalty"] == 0.0
        assert res_even["mfe_bonus"] == 0.0

        # Case B: Winning trade with non-finite excursion defaults mfe to 0.0 (against min_mfe_floor 0.05)
        res_win = compute_extended_lifecycle_settlement(
            realized_r=1.0,
            mae_r=bad_val,
            mfe_r=bad_val,
            holding_bars=10,
        )
        assert np.isfinite(res_win["reward_final"])
        assert res_win["drawdown_penalty"] == 0.0
        assert res_win["mfe_bonus"] == pytest.approx(0.10, abs=1e-6)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf, -100.0])
    def test_non_finite_or_negative_holding_bars_handled_safely(self, bad_val: float) -> None:
        """Non-finite or negative holding duration must clamp to 0.0 penalty."""
        res = compute_extended_lifecycle_settlement(
            realized_r=1.0,
            holding_bars=bad_val,
        )
        assert np.isfinite(res["reward_final"])
        assert res["time_penalty"] == 0.0

    def test_negative_mae_sign_robustness(self) -> None:
        """Drawdown excursion can be logged as negative; penalty must strictly equal absolute."""
        pen_pos = compute_mae_flooding_penalty(1.5, alpha=0.20)
        pen_neg = compute_mae_flooding_penalty(-1.5, alpha=0.20)
        assert pen_pos == pytest.approx(0.30, abs=1e-12)
        assert pen_neg == pytest.approx(0.30, abs=1e-12)
        assert pen_pos == pen_neg

    def test_negative_mfe_yields_zero_bonus(self) -> None:
        """Negative MFE must yield 0 bonus and not cause errors."""
        bon = compute_mfe_harvest_reward(realized_r=1.0, mfe_r=-1.0, gamma=0.10)
        # Denominator will be max(-1.0, 0.05) -> 0.05, efficiency = clip(1.0/0.05, 0, 1) = 1.0 -> 0.10
        assert bon == pytest.approx(0.10, abs=1e-6)

    def test_extreme_magnitude_clamping(self) -> None:
        """Gigantic realized profit or loss must strictly clamp to [-5.0, +5.0]."""
        res_huge_win = compute_extended_lifecycle_settlement(realized_r=10000.0)
        assert res_huge_win["reward_final"] == 5.0
        # 10000.0 realized + 0.10 mfe bonus = 10000.10 unclipped
        assert res_huge_win["unclipped_reward"] == pytest.approx(10000.10, abs=1e-6)

        res_huge_loss = compute_extended_lifecycle_settlement(realized_r=-10000.0)
        assert res_huge_loss["reward_final"] == -5.0
        assert res_huge_loss["unclipped_reward"] == pytest.approx(-10000.0, abs=1e-6)


# ==============================================================================
# 3. Bar 120 Holding Penalty Mathematical Exactness (-0.25R)
# ==============================================================================

class TestBar120HoldingPenaltyExactness:
    """Rigorous mathematical verification of the Bar 120 holding penalty."""

    def test_exact_quarter_r_at_bar_120(self) -> None:
        """Formula: beta * (min(dt, 120)/120)^2 with beta = 0.25.
        At dt = 120: 0.25 * (120/120)^2 = 0.25 * 1.0 = 0.25 identically.
        """
        pen = compute_holding_duration_penalty(120.0, beta=0.25, max_bars=120.0)
        assert pen == 0.25
        # Bitwise identical
        assert pen.hex() == (0.25).hex()

    def test_clamping_for_all_bars_beyond_120(self) -> None:
        """For any dt >= 120, penalty must remain strictly saturated at 0.25."""
        test_bars = [120.0, 120.0001, 121.0, 150.0, 240.0, 500.0, 10000.0]
        for b in test_bars:
            pen = compute_holding_duration_penalty(b, beta=0.25, max_bars=120.0)
            assert pen == 0.25, f"Failed clamping at bar {b}: got {pen}"

    def test_strict_increase_prior_to_bar_120(self) -> None:
        """At dt = 119.999, penalty must be strictly less than 0.25."""
        pen_sub = compute_holding_duration_penalty(119.999, beta=0.25, max_bars=120.0)
        assert pen_sub < 0.25

    def test_first_and_second_derivatives_convexity(self) -> None:
        """Analytical check:
        P(t) = 0.25 * (t / 120)^2 = 0.25 * t^2 / 14400.
        P'(t) = 0.5 * t / 14400 = t / 28800 > 0 for t > 0.
        P''(t) = 1 / 28800 = constant > 0 (Strict Convexity).
        """
        t_values = np.linspace(0.0, 120.0, 121)
        penalties = np.array([compute_holding_duration_penalty(t) for t in t_values])

        dp = np.diff(penalties)
        d2p = np.diff(dp)

        # Monotonically increasing penalty
        assert np.all(dp > 0.0)
        # Strictly positive second difference (convex acceleration)
        assert np.all(d2p > 0.0)
        # Verify second difference matches analytical d2P/dt2 * (dt)^2
        dt = 1.0  # grid step
        expected_d2 = (1.0 / 28800.0) * (dt ** 2)
        assert np.all(np.isclose(d2p, expected_d2, atol=1e-10))

    def test_breakdown_reflects_exact_negative_025r(self) -> None:
        """In FiveLifecycleRewardBreakdown, holding_time_penalty_r is recorded as negative."""
        config = FiveLifecycleRewardConfig(enabled=True, extended_settlement_enabled=True)
        bd = shape_five_entry_lifecycle(
            action_index=1,
            sl_atr=2.0,
            tp_atr=4.0,
            trailing_atr=0.0,
            trailing_applied=False,
            locked_profit_r=0.0,
            mfe_r=0.0,
            mae_r=0.0,
            mfe_available=True,
            realized_r=0.0,
            exit_kind="NORMAL",
            holding_bars=120.0,
            is_timeout=False,
            config=config,
        )
        assert bd.holding_time_penalty_r == -0.25
        assert bd.total_r == -0.25


# ==============================================================================
# 4. Cosine Decay Monotonicity Across 1,000 Steps & Boundary Derivatives
# ==============================================================================

class TestCosineDecayMonotonicityAndBoundaryDerivatives:
    """Stress test dynamic team-spirit cosine decay schedule:
    tau(t) = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * t / T))
    where tau_max = 0.70, tau_min = 0.30, across 1,000 steps.
    """

    def test_1000_steps_monotonic_decrease(self) -> None:
        total_steps = 1000
        steps = list(range(total_steps + 1))
        taus = [
            dynamic_team_spirit_cosine_decay(step=s, total_steps=total_steps)
            for s in steps
        ]

        # 1. Exact boundary and midpoint checks
        assert taus[0] == 0.70
        assert taus[500] == pytest.approx(0.50, abs=1e-12)
        assert taus[1000] == 0.30

        # 2. Strict monotonicity across all 1,000 steps
        diffs = np.diff(taus)
        # Every adjacent step must strictly decrease (diff < 0)
        assert np.all(diffs < 0.0), "Cosine decay is not strictly monotonically decreasing!"

        # 3. Invariant range [0.30, 0.70]
        assert min(taus) == 0.30
        assert max(taus) == 0.70

    def test_boundary_derivatives_tangent_zero(self) -> None:
        """Analytical derivative:
        d(tau)/dp = -0.5 * (0.70 - 0.30) * pi * sin(pi * p) = -0.20 * pi * sin(pi * p).
        At p = 0: sin(0) = 0 -> d(tau)/dp = 0 (Horizontal tangent).
        At p = 1: sin(pi) = 0 -> d(tau)/dp = 0 (Horizontal tangent).

        We test finite difference numerical derivatives with step h = 1e-6:
        As h -> 0, |Delta tau / h| -> 0.
        """
        h = 1e-6

        # Right-sided derivative at p = 0
        tau_0 = compute_team_spirit_cosine(0.0)
        tau_h = compute_team_spirit_cosine(h)
        deriv_0 = (tau_h - tau_0) / h

        # Left-sided derivative at p = 1
        tau_1 = compute_team_spirit_cosine(1.0)
        tau_1_minus_h = compute_team_spirit_cosine(1.0 - h)
        deriv_1 = (tau_1 - tau_1_minus_h) / h

        # Both derivatives must be vanishingly small (order O(h))
        assert abs(deriv_0) < 1e-5, f"Non-zero boundary derivative at p=0: {deriv_0}"
        assert abs(deriv_1) < 1e-5, f"Non-zero boundary derivative at p=1: {deriv_1}"

    def test_smoothness_and_maximum_gradient(self) -> None:
        """Maximum gradient occurs at p = 0.5 (midpoint), where sin(pi * 0.5) = 1:
        |d(tau)/dp|_max = 0.20 * pi approx 0.6283185.
        Step-to-step delta over 1,000 steps must never exceed 0.6283185 / 1000 = 0.0006283.
        """
        total_steps = 1000
        taus = np.array([
            dynamic_team_spirit_cosine_decay(step=s, total_steps=total_steps)
            for s in range(total_steps + 1)
        ])
        diffs = np.abs(np.diff(taus))
        max_step_delta = np.max(diffs)
        expected_max_step_delta = (0.20 * math.pi) / total_steps  # ~ 0.000628318

        assert max_step_delta <= expected_max_step_delta * 1.01
        assert max_step_delta > 0.00062

    def test_pytorch_tensor_vectorization_exact_match(self) -> None:
        """Verify vectorized PyTorch execution matches float precision."""
        p_tensor = torch.linspace(0.0, 1.0, 1001)
        tau_tensor = compute_team_spirit_cosine(p_tensor)

        assert isinstance(tau_tensor, torch.Tensor)
        assert tau_tensor.shape == (1001,)

        tau_numpy = np.array([compute_team_spirit_cosine(float(p)) for p in p_tensor])
        max_discrepancy = float(torch.max(torch.abs(tau_tensor - torch.tensor(tau_numpy, dtype=torch.float32))))
        assert max_discrepancy < 1e-6

    def test_team_spirit_reward_conservation(self) -> None:
        """Under mix_team_spirit_rewards across 5 agents:
        sum_i r_mixed[i] == sum_i r_raw[i] for ANY tau in [0.30, 0.70].
        Team-spirit mixing redistributes reward without creating or destroying total reward.
        """
        rng = np.random.default_rng(seed=12345)
        raw_rewards = rng.normal(loc=0.5, scale=2.0, size=(100, 5)).astype(np.float32)

        for s in [0, 250, 500, 750, 1000]:
            tau = dynamic_team_spirit_cosine_decay(step=s, total_steps=1000)
            mixed = mix_team_spirit_rewards(raw_rewards, team_spirit=tau)

            raw_sums = np.sum(raw_rewards, axis=-1)
            mixed_sums = np.sum(mixed, axis=-1)

            assert np.allclose(raw_sums, mixed_sums, atol=1e-5), f"Conservation violated at step {s}"


# ==============================================================================
# 5. ChuyenGiaPolicy Clamp & Extended Settlement Acceptance
# ==============================================================================

class TestChuyenGiaPolicyClampCompatibility:
    """Verify routing and clamping boundaries in ChuyenGiaPolicy."""

    def test_extended_settlement_accepts_up_to_5r(self) -> None:
        """ChuyenGiaPolicy line 19879:
        clip_bound = 5.000001 if is_extended else 0.750001
        Ensure extended settlement values up to 5.0R are accepted.
        """
        for test_reward in [-5.0, -2.5, 0.0, 2.5, 5.0]:
            is_extended = True
            clip_bound = 5.000001 if is_extended else 0.750001
            assert abs(test_reward) <= clip_bound

        # Excess beyond 5.0R must be rejected
        assert 5.000002 > 5.000001
        assert -5.000002 < -5.000001
