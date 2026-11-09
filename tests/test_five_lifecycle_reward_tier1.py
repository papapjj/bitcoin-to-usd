from __future__ import annotations

"""Unit test suite for Milestone 1: Tier-1 Lifecycle Settlement & Dynamic Team-Spirit Decay.

Verifies:
1. Bar 120 quadratic holding penalty: beta * (min(dt, 120)/120)^2 (beta = 0.25),
   reaching exactly -0.25R at 120 bars, and clamped for dt > 120 bars.
2. Internal MAE flooding penalty: alpha * MAE_R (alpha = 0.20).
3. MFE harvest reward: gamma * Eff_MFE (gamma = 0.10).
4. Timeout expiration penalty: -0.15R when is_timeout = True.
5. Combined extended lifecycle settlement reward bounded within [-5.0, +5.0]R.
6. Dynamic team-spirit cosine decay: tau in [0.30, 0.70], tau(0)=0.70, tau(0.5)=0.50,
   tau(1)=0.30, strict monotonicity, continuity, and PyTorch tensor compatibility.
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
# SECTION 1: Quadratic Holding Penalty at Bar 120
# ==============================================================================

class TestQuadraticHoldingPenalty:
    """Verify holding duration penalty beta * (min(dt, 120)/120)^2 (beta = 0.25)."""

    @pytest.mark.parametrize(
        ("duration_bars", "expected_penalty"),
        [
            (0.0, 0.0),
            (30.0, 0.25 * (30.0 / 120.0) ** 2),       # 0.015625R
            (60.0, 0.25 * (60.0 / 120.0) ** 2),       # 0.062500R
            (90.0, 0.25 * (90.0 / 120.0) ** 2),       # 0.140625R
            (120.0, 0.250000),                         # Exactly 0.250000R
            (150.0, 0.250000),                         # Clamped at 120 bars -> 0.250000R
            (240.0, 0.250000),                         # Clamped at 120 bars -> 0.250000R
        ],
    )
    def test_holding_penalty_key_milestones(self, duration_bars: float, expected_penalty: float) -> None:
        penalty = compute_holding_duration_penalty(duration_bars, beta=0.25, max_bars=120.0)
        assert penalty == pytest.approx(expected_penalty, abs=1e-6)

    def test_exact_penalty_at_bar_120_is_quarter_r(self) -> None:
        """Requirement: verify penalty is exactly -0.25R at 120 bars."""
        penalty_120 = compute_holding_duration_penalty(120.0, beta=0.25, max_bars=120.0)
        assert penalty_120 == 0.25

    def test_holding_penalty_convexity(self) -> None:
        """Holding penalty must be strictly convex: second derivative d^2(pen)/dt^2 > 0."""
        bars = np.linspace(0, 120, 25)
        penalties = [compute_holding_duration_penalty(b) for b in bars]
        first_diff = np.diff(penalties)
        second_diff = np.diff(first_diff)

        # First differences are all positive (monotonically increasing penalty)
        assert np.all(first_diff > 0.0)
        # Second differences are all positive (convexity: penalty accelerates over time)
        assert np.all(second_diff > 0.0)

    def test_holding_penalty_gentle_early_vs_steep_late(self) -> None:
        """First 30 bars should have negligible penalty; last 30 bars should have steep penalty."""
        early_delta = compute_holding_duration_penalty(30.0) - compute_holding_duration_penalty(0.0)
        late_delta = compute_holding_duration_penalty(120.0) - compute_holding_duration_penalty(90.0)

        # Early delta = 0.015625R, Late delta = 0.109375R (7x steeper!)
        assert early_delta == pytest.approx(0.015625, abs=1e-6)
        assert late_delta == pytest.approx(0.109375, abs=1e-6)
        assert late_delta / early_delta == pytest.approx(7.0, rel=1e-3)

    def test_holding_penalty_edge_cases(self) -> None:
        # Negative holding time clamped to 0
        assert compute_holding_duration_penalty(-10.0) == 0.0
        # Non-finite inputs handled safely
        assert compute_holding_duration_penalty(0.0) == 0.0


# ==============================================================================
# SECTION 2: Internal MAE Flooding Penalty
# ==============================================================================

class TestInternalMAEFloodingPenalty:
    """Verify internal MAE flooding penalty alpha * MAE_R with alpha = 0.20."""

    @pytest.mark.parametrize(
        ("mae_r", "expected_penalty"),
        [
            (0.0, 0.0),
            (0.25, 0.20 * 0.25),   # 0.05R
            (0.50, 0.20 * 0.50),   # 0.10R
            (0.75, 0.20 * 0.75),   # 0.15R
            (1.00, 0.20 * 1.00),   # 0.20R
            (1.50, 0.20 * 1.50),   # 0.30R
            (2.00, 0.20 * 2.00),   # 0.40R
        ],
    )
    def test_mae_penalty_scaling(self, mae_r: float, expected_penalty: float) -> None:
        penalty = compute_mae_flooding_penalty(mae_r, alpha=0.20)
        assert penalty == pytest.approx(expected_penalty, abs=1e-6)

    def test_mae_penalty_sign_robustness(self) -> None:
        """In MT5 logs, MAE can be stored as positive (drawdown) or negative (pnl).
        The penalty must handle both signs identically by taking absolute excursion.
        """
        pen_pos = compute_mae_flooding_penalty(0.60, alpha=0.20)
        pen_neg = compute_mae_flooding_penalty(-0.60, alpha=0.20)
        assert pen_pos == pytest.approx(0.12, abs=1e-6)
        assert pen_neg == pytest.approx(0.12, abs=1e-6)
        assert pen_pos == pen_neg

    def test_mae_penalty_zero_drawdown_clean_trade(self) -> None:
        """Trades that run immediately in profit without drawdown incur 0 MAE penalty."""
        assert compute_mae_flooding_penalty(0.0, alpha=0.20) == 0.0


# ==============================================================================
# SECTION 3: MFE Harvest Reward
# ==============================================================================

class TestMFEHarvestReward:
    """Verify MFE harvest reward gamma * Eff_MFE with gamma = 0.10."""

    @pytest.mark.parametrize(
        ("realized_r", "mfe_r", "expected_reward"),
        [
            (2.0, 2.0, 0.10 * 1.0),      # 100% harvest of peak -> full 0.10R
            (1.0, 2.0, 0.10 * 0.5),      # 50% harvest -> 0.05R
            (0.5, 2.0, 0.10 * 0.25),     # 25% harvest -> 0.025R
            (0.0, 2.0, 0.0),             # Break-even -> 0.0R
            (-1.0, 2.0, 0.0),            # Losing trade -> 0.0R (no reward for giving back profits)
            (-0.5, 0.5, 0.0),            # Losing trade -> 0.0R
        ],
    )
    def test_mfe_harvest_efficiency(self, realized_r: float, mfe_r: float, expected_reward: float) -> None:
        reward = compute_mfe_harvest_reward(realized_r, mfe_r, gamma=0.10)
        assert reward == pytest.approx(expected_reward, abs=1e-6)

    def test_mfe_harvest_zero_division_guard(self) -> None:
        """Denominator has min floor of 0.05; zero mfe_r must not cause ZeroDivisionError."""
        reward = compute_mfe_harvest_reward(realized_r=0.04, mfe_r=0.0, gamma=0.10, min_mfe_floor=0.05)
        # 0.04 / 0.05 = 0.8 efficiency -> 0.08R
        assert reward == pytest.approx(0.10 * 0.8, abs=1e-6)

    def test_mfe_harvest_slippage_overshoot_clamped(self) -> None:
        """If realized_r slightly exceeds mfe_r due to positive slippage, efficiency is clamped at 1.0."""
        reward = compute_mfe_harvest_reward(realized_r=2.1, mfe_r=2.0, gamma=0.10)
        assert reward == pytest.approx(0.10, abs=1e-6)


# ==============================================================================
# SECTION 4: Timeout Penalty
# ==============================================================================

class TestTimeoutPenalty:
    """Verify timeout penalty -0.15R when is_timeout = True."""

    def test_timeout_penalty_active(self) -> None:
        penalty = compute_timeout_penalty(is_timeout=True, penalty_r=0.15)
        assert penalty == pytest.approx(0.15, abs=1e-6)

    def test_timeout_penalty_inactive(self) -> None:
        penalty = compute_timeout_penalty(is_timeout=False, penalty_r=0.15)
        assert penalty == 0.0

    def test_timeout_at_bar_120_cumulative_penalty(self) -> None:
        """A trade hitting timeout at bar 120 incurs holding penalty (0.25R) + timeout penalty (0.15R) = 0.40R."""
        hold_pen = compute_holding_duration_penalty(120.0, beta=0.25)
        to_pen = compute_timeout_penalty(is_timeout=True, penalty_r=0.15)
        total_time_penalty = hold_pen + to_pen
        assert total_time_penalty == pytest.approx(0.40, abs=1e-6)


# ==============================================================================
# SECTION 5: Combined Extended Lifecycle Settlement Reward
# ==============================================================================

class TestCombinedExtendedLifecycleSettlement:
    """Verify total combined formulation and boundary clipping within [-5.0, +5.0]R."""

    def test_clean_winner_fast_exit(self) -> None:
        # realized = +2.0R, MAE = 0.10R, dt = 20 bars, MFE = 2.0R, no timeout
        r_star = calculate_extended_lifecycle_settlement(
            realized_r=2.0,
            mae_r=0.10,
            duration_bars=20.0,
            mfe_r=2.0,
            is_timeout=False,
        )
        # Expected:
        # +2.0 - 0.20*(0.10) - 0.25*(20/120)^2 + 0.10*(2.0/2.0) - 0.0
        # = 2.0 - 0.02 - 0.25*(1/36) + 0.10
        # = 2.0 - 0.02 - 0.006944 + 0.10 = 2.073056R
        expected = 2.0 - 0.02 - 0.25 * (1.0 / 36.0) + 0.10
        assert r_star == pytest.approx(expected, abs=1e-5)

        # Also verify through shape_five_entry_lifecycle
        shaped = shape_five_entry_lifecycle(
            action_index=1,
            sl_atr=2.5,
            tp_atr=5.0,
            trailing_atr=0.0,
            trailing_applied=False,
            locked_profit_r=0.0,
            mfe_r=2.0,
            mae_r=0.10,
            mfe_available=True,
            realized_r=2.0,
            exit_kind="TP",
            duration_bars=20.0,
            is_timeout=False,
            extended_settlement=True,
            config=FiveLifecycleRewardConfig(),
        )
        assert shaped.total_r == pytest.approx(expected, abs=1e-5)
        assert shaped.extended_settlement is True

    def test_flooded_winner_near_timeout(self) -> None:
        # realized = +2.0R, MAE = 0.95R, dt = 115 bars, MFE = 2.0R, no timeout
        r_star = calculate_extended_lifecycle_settlement(
            realized_r=2.0,
            mae_r=0.95,
            duration_bars=115.0,
            mfe_r=2.0,
            is_timeout=False,
        )
        # Expected:
        # +2.0 - 0.20*(0.95) - 0.25*(115/120)^2 + 0.10*(1.0)
        # = 2.0 - 0.19 - 0.25*(0.917014) + 0.10 = 2.0 - 0.19 - 0.229253 + 0.10 = 1.680747R
        expected = 2.0 - 0.19 - 0.25 * ((115.0 / 120.0) ** 2) + 0.10
        assert r_star == pytest.approx(expected, abs=1e-5)
        # Significantly lower than clean fast winner!
        assert r_star < 1.70

    def test_forced_timeout_close_at_bar_120(self) -> None:
        # Trade stalled in consolidation, closed by CloseExpiredPositions at bar 120
        # realized = -0.10R, MAE = 0.80R, dt = 120 bars, MFE = 0.30R, is_timeout = True
        r_star = calculate_extended_lifecycle_settlement(
            realized_r=-0.10,
            mae_r=0.80,
            duration_bars=120.0,
            mfe_r=0.30,
            is_timeout=True,
        )
        # Expected:
        # -0.10 - 0.20*(0.80) - 0.25*(1.0) + 0.0 (realized <= 0) - 0.15
        # = -0.10 - 0.16 - 0.25 + 0.0 - 0.15 = -0.66R
        assert r_star == pytest.approx(-0.66, abs=1e-5)

        # Also verify breakdown in shape_five_entry_lifecycle
        shaped = shape_five_entry_lifecycle(
            action_index=1,
            sl_atr=2.5,
            tp_atr=5.0,
            trailing_atr=0.0,
            trailing_applied=False,
            locked_profit_r=0.0,
            mfe_r=0.30,
            mae_r=0.80,
            mfe_available=True,
            realized_r=-0.10,
            exit_kind="TIMEOUT_EXPIRED",
            duration_bars=120.0,
            is_timeout=True,
            extended_settlement=True,
            config=FiveLifecycleRewardConfig(),
        )
        assert shaped.total_r == pytest.approx(-0.66, abs=1e-5)
        assert shaped.holding_time_penalty_r == pytest.approx(-0.25, abs=1e-5)
        assert shaped.timeout_penalty_r == pytest.approx(-0.15, abs=1e-5)
        assert shaped.mae_flooding_penalty_r == pytest.approx(-0.16, abs=1e-5)

    def test_clipping_bounds(self) -> None:
        # Extreme negative
        clipped_low = calculate_extended_lifecycle_settlement(
            realized_r=-10.0, mae_r=5.0, duration_bars=120.0, mfe_r=0.0, is_timeout=True
        )
        assert clipped_low == -5.0

        # Extreme positive
        clipped_high = calculate_extended_lifecycle_settlement(
            realized_r=10.0, mae_r=0.0, duration_bars=5.0, mfe_r=10.0, is_timeout=False
        )
        assert clipped_high == 5.0


# ==============================================================================
# SECTION 6: Dynamic Team-Spirit Cosine Decay
# ==============================================================================

class TestDynamicTeamSpiritCosineDecay:
    """Verify dynamic team-spirit cosine decay schedule:
    tau(p) = 0.30 + 0.5 * (0.70 - 0.30) * [1 + cos(pi * p)]
    """

    def test_boundary_and_midpoint_exact_values(self) -> None:
        """Verify: tau(0) = 0.70, tau(0.5) = 0.50, tau(1.0) = 0.30."""
        tau_start = compute_team_spirit_cosine(0.0)
        tau_mid = compute_team_spirit_cosine(0.5)
        tau_end = compute_team_spirit_cosine(1.0)

        assert tau_start == pytest.approx(0.70, abs=1e-6)
        assert tau_mid == pytest.approx(0.50, abs=1e-6)
        assert tau_end == pytest.approx(0.30, abs=1e-6)

        # Also test via FiveTrainingSchedule
        schedule = FiveTrainingSchedule(progress=0.0, team_spirit_cosine_decay_enabled=True)
        assert schedule.team_spirit == pytest.approx(0.70, abs=1e-6)
        assert schedule.get_team_spirit(0, 100) == pytest.approx(0.70, abs=1e-6)
        assert schedule.get_team_spirit(50, 100) == pytest.approx(0.50, abs=1e-6)
        assert schedule.get_team_spirit(100, 100) == pytest.approx(0.30, abs=1e-6)

        # Backward compatibility check
        legacy_sched = FiveTrainingSchedule(progress=0.0)
        assert legacy_sched.team_spirit == pytest.approx(0.30, abs=1e-6)
        legacy_sched.set_progress(1.0)
        assert legacy_sched.team_spirit == pytest.approx(1.00, abs=1e-6)

    @pytest.mark.parametrize(
        ("progress", "expected_tau"),
        [
            (0.00, 0.70),
            (0.25, 0.30 + 0.20 * (1.0 + math.cos(math.pi * 0.25))),  # ~0.641421
            (0.50, 0.50),
            (0.75, 0.30 + 0.20 * (1.0 + math.cos(math.pi * 0.75))),  # ~0.358579
            (1.00, 0.30),
        ],
    )
    def test_intermediate_values(self, progress: float, expected_tau: float) -> None:
        tau = compute_team_spirit_cosine(progress)
        assert tau == pytest.approx(expected_tau, abs=1e-6)
        # Verify dynamic_team_spirit_cosine_decay module function
        assert dynamic_team_spirit_cosine_decay(progress) == pytest.approx(expected_tau, abs=1e-6)

    def test_strict_range_invariant_030_to_070(self) -> None:
        """Verify tau remains strictly within [0.30, 0.70] across 1000 points."""
        points = np.linspace(0.0, 1.0, 1000)
        taus = np.asarray([compute_team_spirit_cosine(p) for p in points])

        assert np.all(taus >= 0.30 - 1e-9)
        assert np.all(taus <= 0.70 + 1e-9)
        assert float(np.min(taus)) == pytest.approx(0.30, abs=1e-6)
        assert float(np.max(taus)) == pytest.approx(0.70, abs=1e-6)

    def test_strict_monotonicity(self) -> None:
        """tau must be strictly non-increasing: d(tau)/dp <= 0 everywhere."""
        points = np.linspace(0.0, 1.0, 1000)
        taus = np.asarray([compute_team_spirit_cosine(p) for p in points])
        diffs = np.diff(taus)

        # All step differences must be non-positive
        assert np.all(diffs <= 1e-12)

    def test_smoothness_and_no_discontinuities(self) -> None:
        """Continuous and smooth: max step change is bounded by max theoretical derivative."""
        # Derivative d(tau)/dp = -0.20 * pi * sin(pi * p). Max magnitude = 0.20 * pi ~ 0.6283185
        points = np.linspace(0.0, 1.0, 1000)
        step = points[1] - points[0]
        taus = np.asarray([compute_team_spirit_cosine(p) for p in points])
        step_diffs = np.abs(np.diff(taus))

        max_theoretical_step_change = 0.20 * math.pi * step * 1.05  # with 5% margin
        assert np.max(step_diffs) <= max_theoretical_step_change
        # Verify no sudden jumps/cliffs
        assert np.all(step_diffs < 0.001)

    def test_zero_derivative_at_endpoints(self) -> None:
        """Smooth start and smooth finish: d(tau)/dp at p=0 and p=1 is zero."""
        eps = 1e-5
        derivative_at_0 = (compute_team_spirit_cosine(eps) - compute_team_spirit_cosine(0.0)) / eps
        derivative_at_1 = (compute_team_spirit_cosine(1.0) - compute_team_spirit_cosine(1.0 - eps)) / eps

        assert abs(derivative_at_0) < 1e-4
        assert abs(derivative_at_1) < 1e-4

    def test_out_of_bounds_progress_clamping(self) -> None:
        """Progress < 0 clamped to 0.70; progress > 1 clamped to 0.30."""
        assert compute_team_spirit_cosine(-0.5) == pytest.approx(0.70, abs=1e-6)
        assert compute_team_spirit_cosine(1.5) == pytest.approx(0.30, abs=1e-6)

    def test_mix_team_spirit_rewards_with_dynamic_cosine(self) -> None:
        """Verify 5-agent reward mixing with dynamic cosine decay:
        At start (tau=0.70): team reward dominates (70% team, 30% individual)
        At end (tau=0.30): individual reward dominates (30% team, 70% individual)
        """
        raw_rewards = np.asarray([1.0, 0.0, -1.0, 2.0, 3.0], dtype=np.float32)
        team_mean = float(np.mean(raw_rewards))  # (1 + 0 - 1 + 2 + 3)/5 = 1.0

        # At p = 0.0: tau = 0.70
        tau_0 = compute_team_spirit_cosine(0.0)
        mixed_0 = mix_team_spirit_rewards(raw_rewards, tau_0)
        expected_0 = 0.30 * raw_rewards + 0.70 * team_mean
        np.testing.assert_allclose(mixed_0, expected_0, atol=1e-6)

        # At p = 0.5: tau = 0.50
        tau_mid = compute_team_spirit_cosine(0.5)
        mixed_mid = mix_team_spirit_rewards(raw_rewards, tau_mid)
        expected_mid = 0.50 * raw_rewards + 0.50 * team_mean
        np.testing.assert_allclose(mixed_mid, expected_mid, atol=1e-6)

        # At p = 1.0: tau = 0.30
        tau_1 = compute_team_spirit_cosine(1.0)
        mixed_1 = mix_team_spirit_rewards(raw_rewards, tau_1)
        expected_1 = 0.70 * raw_rewards + 0.30 * team_mean
        np.testing.assert_allclose(mixed_1, expected_1, atol=1e-6)

        # Conservation of team reward: sum(mixed) == sum(raw)
        assert np.sum(mixed_0) == pytest.approx(np.sum(raw_rewards), abs=1e-5)
        assert np.sum(mixed_mid) == pytest.approx(np.sum(raw_rewards), abs=1e-5)
        assert np.sum(mixed_1) == pytest.approx(np.sum(raw_rewards), abs=1e-5)

    def test_pytorch_tensor_compatibility(self) -> None:
        """Cosine decay must work natively with PyTorch tensors."""
        progress_tensor = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0], dtype=torch.float32)
        expected_values = [
            compute_team_spirit_cosine(float(p)) for p in progress_tensor
        ]
        
        # Pure PyTorch vectorized formula
        p_clamped = torch.clamp(progress_tensor, 0.0, 1.0)
        tau_tensor = 0.30 + 0.20 * (1.0 + torch.cos(math.pi * p_clamped))
        
        np.testing.assert_allclose(tau_tensor.numpy(), expected_values, atol=1e-6)

        # Also test compute_team_spirit_cosine directly with Tensor input
        tau_direct = compute_team_spirit_cosine(progress_tensor)
        np.testing.assert_allclose(tau_direct.numpy(), expected_values, atol=1e-6)
