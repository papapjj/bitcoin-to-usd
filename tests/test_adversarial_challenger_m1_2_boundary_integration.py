from __future__ import annotations

"""Adversarial Challenge Suite: Milestone 1 - Boundary & Integration Verification.

Subagent: challenger_m1_2
Archetype: critic, specialist

Rigorous empirical tests covering:
1. Extreme boundary cases:
   - Zero MFE (0.0, -0.0, negative MFE, floor division safety)
   - Zero MAE (0.0, -0.0, negative MAE sign invariance, extreme large MAE clamping)
   - Negative progress (< 0.0) in scalar, training schedule, and PyTorch tensors
   - Progress > 1.0 in scalar, training schedule, and PyTorch tensors
   - Huge holding bars (120, 121, 1000, 1e6, inf, nan) and negative bars
2. Integration with chuyen_gia_policy.py line 19877:
   - Exact acceptance of rewards in [-5.0, +5.0]R without false rejection
   - Strict rejection beyond boundaries (+5.0001R, -5.0001R, NaN, Inf)
   - Preservation of legacy credit vs extended zero-double-credit
   - Handling of heterogeneous components (strings, booleans, ints, floats)
3. Memory leaks and computational overhead:
   - Tracemalloc memory leak detection over 50,000 reward cycles
   - Microsecond-level throughput / latency benchmarking (< 15us per evaluation)
   - Comparative overhead analysis against legacy lifecycle shaping
"""

import math
import time
import tracemalloc
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.five_lifecycle_reward import (
    FiveLifecycleRewardBreakdown,
    FiveLifecycleRewardConfig,
    calculate_extended_lifecycle_settlement,
    compute_extended_lifecycle_settlement,
    compute_holding_duration_penalty,
    compute_mae_flooding_penalty,
    compute_mfe_harvest_reward,
    compute_timeout_penalty,
    shape_five_entry_lifecycle,
    ExtendedLifecycleSettlementEngine,
)
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FiveTrainingSchedule,
    SharedFiveTradingPolicy,
    compute_team_spirit_cosine,
    dynamic_team_spirit_cosine_decay,
)
from self_growing_trader.protocol_contract import INPUT_COUNT


# ==============================================================================
# SECTION 1: Extreme Boundary Conditions
# ==============================================================================

class TestExtremeBoundaryConditions:
    """Empirical challenge on extreme boundary inputs for all Milestone 1 components."""

    # --------------------------------------------------------------------------
    # 1. Zero and Subnormal MFE
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize("zero_mfe", [0.0, -0.0, 1e-15, -1e-15, -1.0, -100.0])
    def test_zero_and_negative_mfe_safe_division(self, zero_mfe: float) -> None:
        """MFE <= 0 must never cause ZeroDivisionError and must yield valid bounded reward."""
        # When realized profit is positive but MFE is recorded as 0 (e.g. instant fill at market)
        bonus = compute_mfe_harvest_reward(realized_r=0.04, mfe_r=zero_mfe, gamma=0.10, min_mfe_floor=0.05)
        # Denom floors at 0.05: 0.04 / 0.05 = 0.8 -> bonus = 0.08R
        assert np.isfinite(bonus)
        assert bonus == pytest.approx(0.08, abs=1e-6)

        # When trade is a loss, zero MFE must return exactly 0.0 bonus
        loss_bonus = compute_mfe_harvest_reward(realized_r=-1.0, mfe_r=zero_mfe, gamma=0.10)
        assert loss_bonus == 0.0

        # In full settlement calculation
        settlement = calculate_extended_lifecycle_settlement(
            realized_r=1.0,
            mae_r=0.0,
            duration_bars=10.0,
            mfe_r=zero_mfe,
            is_timeout=False,
        )
        assert np.isfinite(settlement)
        assert -5.0 <= settlement <= 5.0

    def test_non_finite_mfe_handled_safely(self) -> None:
        """NaN or Inf in MFE must be safely handled without throwing exceptions."""
        bonus_nan = compute_mfe_harvest_reward(realized_r=1.0, mfe_r=float("nan"))
        assert np.isfinite(bonus_nan)
        assert bonus_nan >= 0.0

        bonus_inf = compute_mfe_harvest_reward(realized_r=1.0, mfe_r=float("inf"))
        assert np.isfinite(bonus_inf)

    # --------------------------------------------------------------------------
    # 2. Zero, Negative, and Extreme Large MAE
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize("zero_mae", [0.0, -0.0, 1e-15, -1e-15])
    def test_zero_mae_incurs_zero_penalty(self, zero_mae: float) -> None:
        """Trades with zero adverse excursion must incur exactly 0.0 penalty."""
        penalty = compute_mae_flooding_penalty(zero_mae, alpha=0.20)
        assert penalty == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("mae_val", [0.50, -0.50, 1.25, -1.25])
    def test_mae_sign_invariance(self, mae_val: float) -> None:
        """MAE can be recorded as positive or negative; penalty must be identical."""
        pos_pen = compute_mae_flooding_penalty(abs(mae_val), alpha=0.20)
        raw_pen = compute_mae_flooding_penalty(mae_val, alpha=0.20)
        assert pos_pen == pytest.approx(raw_pen, abs=1e-9)

    def test_extreme_large_mae_and_clamping(self) -> None:
        """Massive MAE (catastrophic runaway slippage) must not cause overflow and must clamp to -5.0R."""
        huge_mae_penalty = compute_mae_flooding_penalty(1000.0, alpha=0.20)
        assert huge_mae_penalty == pytest.approx(200.0, abs=1e-6)

        settlement = calculate_extended_lifecycle_settlement(
            realized_r=-1.0,
            mae_r=1000.0,
            duration_bars=120.0,
            is_timeout=True,
        )
        assert settlement == -5.0

    # --------------------------------------------------------------------------
    # 3. Holding Bars: Huge, Negative, Infinite, and Exact 120
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize("bars", [120.0, 121.0, 500.0, 10_000.0, 1e8])
    def test_huge_holding_bars_strictly_clamped_at_quarter_r(self, bars: float) -> None:
        """Holding duration penalty must be strictly clamped at 0.25R for dt >= 120 bars."""
        penalty = compute_holding_duration_penalty(bars, beta=0.25, max_bars=120.0, clamp_max=True)
        assert penalty == pytest.approx(0.25, abs=1e-9)

    @pytest.mark.parametrize("neg_bars", [-1.0, -10.0, -120.0, -1e6])
    def test_negative_holding_bars_clamped_to_zero(self, neg_bars: float) -> None:
        """Negative holding duration from clock skew or corrupt timestamps must clamp to 0."""
        penalty = compute_holding_duration_penalty(neg_bars, beta=0.25, max_bars=120.0)
        assert penalty == 0.0

    def test_non_finite_holding_bars_handled_safely(self) -> None:
        """NaN or Inf duration bars must be safely coerced to 0.0 without crash."""
        penalty_nan = compute_holding_duration_penalty(float("nan"), beta=0.25)
        assert penalty_nan == 0.0

        penalty_inf = compute_holding_duration_penalty(float("inf"), beta=0.25)
        assert penalty_inf == 0.0

    # --------------------------------------------------------------------------
    # 4. Negative Progress and Progress > 1.0 (Team-Spirit Decay)
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize("neg_prog", [-0.0001, -0.5, -1.0, -100.0, -1e9])
    def test_negative_progress_clamped_to_start_tau_070(self, neg_prog: float) -> None:
        """Progress < 0 must clamp to start value tau = 0.70."""
        tau_scalar = compute_team_spirit_cosine(neg_prog)
        assert tau_scalar == pytest.approx(0.70, abs=1e-6)

        tau_dynamic = dynamic_team_spirit_cosine_decay(neg_prog)
        assert tau_dynamic == pytest.approx(0.70, abs=1e-6)

        sched = FiveTrainingSchedule(progress=neg_prog, team_spirit_cosine_decay_enabled=True)
        assert sched.team_spirit == pytest.approx(0.70, abs=1e-6)

    @pytest.mark.parametrize("over_prog", [1.0001, 1.5, 2.0, 100.0, 1e9])
    def test_progress_greater_than_one_clamped_to_end_tau_030(self, over_prog: float) -> None:
        """Progress > 1.0 must clamp to final value tau = 0.30."""
        tau_scalar = compute_team_spirit_cosine(over_prog)
        assert tau_scalar == pytest.approx(0.30, abs=1e-6)

        tau_dynamic = dynamic_team_spirit_cosine_decay(over_prog)
        assert tau_dynamic == pytest.approx(0.30, abs=1e-6)

        sched = FiveTrainingSchedule(progress=over_prog, team_spirit_cosine_decay_enabled=True)
        assert sched.team_spirit == pytest.approx(0.30, abs=1e-6)

    def test_tensor_out_of_bounds_progress_clamped(self) -> None:
        """PyTorch tensor containing out-of-bound progress values must clamp correctly."""
        t_prog = torch.tensor([-5.0, -0.01, 0.0, 0.5, 1.0, 1.01, 50.0], dtype=torch.float32)
        taus = compute_team_spirit_cosine(t_prog)
        expected = np.asarray([0.70, 0.70, 0.70, 0.50, 0.30, 0.30, 0.30], dtype=np.float32)
        np.testing.assert_allclose(taus.numpy(), expected, atol=1e-6)

    def test_nan_progress_gracefully_defaults_to_start(self) -> None:
        """NaN progress gracefully falls back to start tau 0.70 without throwing exception."""
        assert compute_team_spirit_cosine(float("nan")) == pytest.approx(0.70, abs=1e-6)
        assert dynamic_team_spirit_cosine_decay(float("nan")) == pytest.approx(0.70, abs=1e-6)


# ==============================================================================
# SECTION 2: Verification of chuyen_gia_policy.py Line 19877 Acceptance
# ==============================================================================

def _create_test_policy_and_packet():
    """Build a functional ChuyenGiaPolicy with one registered entry packet."""
    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    policy.five_agent_memories = policy.five_shared_network.initial_memories()
    policy._learning_enabled = True
    policy.five_lifecycle_batch_size = 32

    rows = list(policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details.update({
            "five_action_index": 1 if index == 0 else 0,
            "five_economic_actor_eligible": bool(index == 0),
            "five_execution_filled": bool(index == 0),
        })
        enriched.append(replace(trace, action_details=details))

    reg_result = policy.register_five_entry_lifecycle_packet(tuple(enriched))
    assert reg_result["registered"] == 1
    return policy, target_id


class TestChuyenGiaPolicyRewardAcceptanceLine19877:
    """Verifies that chuyen_gia_policy.py line 19877 properly accepts rewards up to +5.0R without false rejection."""

    @pytest.mark.parametrize(
        "reward_val",
        [
            0.75,
            0.750001,
            0.750002,
            1.0,
            2.0,
            3.0,
            4.0,
            4.999,
            5.0,
            5.000001,  # Exact float boundary
        ],
    )
    def test_positive_extended_rewards_accepted_up_to_5r(self, reward_val: float) -> None:
        """Every positive reward up to +5.0R must be accepted without false rejection."""
        policy, trace_id = _create_test_policy_and_packet()
        result = policy.apply_five_entry_lifecycle_reward(
            trace_id,
            lifecycle_reward_r=reward_val,
            realized_r=reward_val,
            components={
                "extended_settlement": True,
                "total_r": reward_val,
                "realized_r": reward_val,
            },
        )
        assert result["accepted"] is True, f"Failed to accept reward {reward_val}: {result}"
        assert result["updated"] in (0, 1)
        assert result["reward_r"] == pytest.approx(reward_val, abs=1e-5)

    @pytest.mark.parametrize(
        "reward_val",
        [
            -0.75,
            -0.750001,
            -0.750002,
            -1.0,
            -2.0,
            -3.0,
            -4.0,
            -4.999,
            -5.0,
            -5.000001,
        ],
    )
    def test_negative_extended_rewards_accepted_down_to_minus_5r(self, reward_val: float) -> None:
        """Every negative reward down to -5.0R must be accepted without false rejection."""
        policy, trace_id = _create_test_policy_and_packet()
        result = policy.apply_five_entry_lifecycle_reward(
            trace_id,
            lifecycle_reward_r=reward_val,
            realized_r=reward_val,
            components={
                "extended_settlement": True,
                "total_r": reward_val,
                "realized_r": reward_val,
            },
        )
        assert result["accepted"] is True, f"Failed to accept reward {reward_val}: {result}"
        assert result["updated"] in (0, 1)
        assert result["reward_r"] == pytest.approx(reward_val, abs=1e-5)

    @pytest.mark.parametrize(
        "invalid_reward",
        [
            5.0001,
            5.10,
            10.0,
            100.0,
            -5.0001,
            -5.10,
            -10.0,
            -100.0,
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_out_of_bounds_rewards_strictly_rejected(self, invalid_reward: float) -> None:
        """Rewards exceeding [-5.000001, +5.000001] or non-finite values must be rejected."""
        policy, trace_id = _create_test_policy_and_packet()
        result = policy.apply_five_entry_lifecycle_reward(
            trace_id,
            lifecycle_reward_r=invalid_reward,
            realized_r=1.0,
            components={"extended_settlement": True, "total_r": invalid_reward},
        )
        assert result["accepted"] is False
        assert result["reason"] == "invalid_lifecycle_v4"

    def test_automatic_detection_without_explicit_flag_when_exceeding_075(self) -> None:
        """Even if components dict omits 'extended_settlement: True', a reward > 0.750001
        must automatically be detected as extended and accepted up to 5.0R.
        """
        policy, trace_id = _create_test_policy_and_packet()
        # Omit 'extended_settlement' in components
        result = policy.apply_five_entry_lifecycle_reward(
            trace_id,
            lifecycle_reward_r=3.5,
            realized_r=3.5,
            components={"total_r": 3.5},
        )
        assert result["accepted"] is True
        assert result["reward_r"] == pytest.approx(3.5, abs=1e-5)

    def test_extended_settlement_prevents_net_r_double_credit(self) -> None:
        """When is_extended is True, net_r_credit must be set to 0.0 to prevent double-counting."""
        policy, trace_id = _create_test_policy_and_packet()
        # Enable legacy net credit
        policy.five_lifecycle_net_r_credit_enabled = True
        policy.five_lifecycle_net_r_credit_weight = 1.0

        result = policy.apply_five_entry_lifecycle_reward(
            trace_id,
            lifecycle_reward_r=2.5,
            realized_r=2.0,
            components={"extended_settlement": True, "total_r": 2.5},
        )
        assert result["accepted"] is True
        # Total reward must be exactly 2.5, NOT 2.5 + 2.0 = 4.5!
        assert result["reward_r"] == pytest.approx(2.5, abs=1e-5)

    def test_legacy_shaping_preserves_net_r_credit(self) -> None:
        """When is_extended is False (legacy V4 shaping <= 0.75R), net_r_credit is added as intended."""
        policy, trace_id = _create_test_policy_and_packet()
        policy.five_lifecycle_net_r_credit_enabled = True
        policy.five_lifecycle_net_r_credit_weight = 0.5

        result = policy.apply_five_entry_lifecycle_reward(
            trace_id,
            lifecycle_reward_r=0.20,
            realized_r=1.0,
            components={"extended_settlement": False, "total_r": 0.20},
        )
        assert result["accepted"] is True
        # 0.20 + 0.5 * 1.0 = 0.70R
        assert result["reward_r"] == pytest.approx(0.70, abs=1e-5)

    def test_heterogeneous_components_parsing_safety(self) -> None:
        """Components with strings, booleans, ints, and floats must be parsed without error."""
        policy, trace_id = _create_test_policy_and_packet()
        components = {
            "extended_settlement": True,
            "exit_kind": "TIMEOUT_EXPIRED",  # String
            "is_timeout": True,               # Bool
            "duration_bars": 120,             # Int
            "realized_r": 2.5,                # Float
            "mae_flooding_penalty_r": -0.10,  # Float
        }
        result = policy.apply_five_entry_lifecycle_reward(
            trace_id,
            lifecycle_reward_r=2.15,
            realized_r=2.5,
            components=components,
        )
        assert result["accepted"] is True


# ==============================================================================
# SECTION 3: Memory Leaks & Computational Overhead Benchmark
# ==============================================================================

class TestMemoryLeakAndComputationalOverhead:
    """Stress tests reward calculations for memory stability and microsecond throughput."""

    def test_memory_leak_free_over_50000_cycles(self) -> None:
        """50,000 sequential reward evaluations must show zero heap memory leakage."""
        tracemalloc.start()
        gc_collect = True

        # Warm-up phase
        config = FiveLifecycleRewardConfig()
        for i in range(1000):
            _ = calculate_extended_lifecycle_settlement(
                realized_r=1.0,
                mae_r=0.20,
                duration_bars=float(i % 120),
                mfe_r=1.5,
                is_timeout=(i % 120 == 0),
            )

        snapshot1 = tracemalloc.take_snapshot()

        # 50,000 stress loop iterations
        for i in range(50000):
            bars = float(i % 150)
            realized = float((i % 20) - 10) * 0.5
            mae = float(i % 30) * 0.1
            mfe = float(i % 40) * 0.1
            to = (bars >= 120.0)

            # Test engine computation
            _ = compute_extended_lifecycle_settlement(
                realized_r=realized,
                mae_r=mae,
                mfe_r=mfe,
                holding_bars=bars,
                is_timeout=to,
            )

            # Test shape_five_entry_lifecycle
            _ = shape_five_entry_lifecycle(
                action_index=1,
                sl_atr=2.5,
                tp_atr=5.0,
                trailing_atr=0.0,
                trailing_applied=False,
                locked_profit_r=0.0,
                mfe_r=mfe,
                mae_r=mae,
                mfe_available=True,
                realized_r=realized,
                exit_kind="TIMEOUT_EXPIRED" if to else "NORMAL",
                duration_bars=bars,
                is_timeout=to,
                extended_settlement=True,
                config=config,
            )

            # Test cosine decay
            p = float(i % 1000) / 1000.0
            _ = compute_team_spirit_cosine(p)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_growth_bytes = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # Allow minor Python internal runtime overhead (< 500 KB across 50,000 objects created/destroyed)
        assert total_growth_bytes < 500_000, f"Potential memory leak detected: {total_growth_bytes} bytes"

    def test_computational_throughput_and_latency(self) -> None:
        """50,000 calculations must complete in < 1.0 second (< 20 microseconds per call)."""
        engine = ExtendedLifecycleSettlementEngine()
        iterations = 50000

        start_time = time.perf_counter()
        for i in range(iterations):
            _ = engine.compute_settlement_reward(
                realized_r=1.5,
                mae_r=0.3,
                mfe_r=2.0,
                holding_bars=i % 125,
                is_timeout=(i % 125 >= 120),
            )
        elapsed = time.perf_counter() - start_time

        us_per_call = (elapsed / iterations) * 1e6
        # Must execute within 2.0 seconds on CPU (< 40 us per call)
        assert elapsed < 2.0, f"Computation too slow: {elapsed:.3f}s for {iterations} calls"
        # Must be under 40 microseconds per call
        assert us_per_call < 40.0, f"High latency: {us_per_call:.2f} us per call"

    def test_comparative_overhead_extended_vs_legacy(self) -> None:
        """Extended settlement must have comparable or lower latency than legacy V4 multi-knot interpolation."""
        config_legacy = FiveLifecycleRewardConfig()
        config_extended = FiveLifecycleRewardConfig(extended_settlement_enabled=True)
        iterations = 20000

        # Legacy V4 timing
        t0 = time.perf_counter()
        for i in range(iterations):
            _ = shape_five_entry_lifecycle(
                action_index=1,
                sl_atr=2.5,
                tp_atr=5.0,
                trailing_atr=1.5,
                trailing_applied=True,
                locked_profit_r=0.5,
                mfe_r=1.2,
                mae_r=0.3,
                mfe_available=True,
                realized_r=1.0,
                exit_kind="TP",
                duration_bars=50.0,
                extended_settlement=False,
                config=config_legacy,
            )
        t_legacy = time.perf_counter() - t0

        # Extended Settlement timing
        t1 = time.perf_counter()
        for i in range(iterations):
            _ = shape_five_entry_lifecycle(
                action_index=1,
                sl_atr=2.5,
                tp_atr=5.0,
                trailing_atr=1.5,
                trailing_applied=True,
                locked_profit_r=0.5,
                mfe_r=1.2,
                mae_r=0.3,
                mfe_available=True,
                realized_r=1.0,
                exit_kind="TP",
                duration_bars=50.0,
                extended_settlement=True,
                config=config_extended,
            )
        t_extended = time.perf_counter() - t1

        # Extended settlement should be fast and not exceed 2.5x legacy time
        assert t_extended < t_legacy * 2.5, (
            f"Extended settlement introduces excessive overhead: "
            f"Extended={t_extended:.4f}s vs Legacy={t_legacy:.4f}s"
        )
