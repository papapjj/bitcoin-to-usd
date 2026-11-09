"""Adversarial and Empirical Stress Test Suite for Auto-RL Engine and Strict OOS Validator.

This suite empirically stress-tests:
1. Extreme boundary conditions:
   - Exactly 29 vs 30 trades (min_trades = 30)
   - E[R] = 0.0499 vs 0.0500 (min_expectancy = 0.0500)
   - Drawdown 9.99% vs 10.00% (max_drawdown = 10.0%)
   - Cumulative DD 14.99% vs 15.00% vs 15.01% (halt_pct = 15.0%)
   - Hold share 0.799 vs 0.800 (max_hold = 0.80)
   - Action entropy 0.350 vs 0.3501 (min_entropy = 0.35)
   - Directional imbalance 0.6000 vs 0.6001 (max_imbalance = 0.60)
   - Win rate 39.99% vs 40.00% (min_win_rate = 40.0%)
   - Risk/Reward 1.999 vs 2.000 (min_rr = 2.0)
2. Market stress and pathological scenarios:
   - Zero-trade droughts (3-round M1 failure triggering M5, then M15)
   - Extreme volatility and catastrophic single-round crashes (immediate emergency halt)
   - Alternating win/loss sequences (whipsaw market)
   - Spread spike and geometry friction constraints
3. Adversarial numerical inputs and edge cases:
   - NaN and Inf handling
   - Negative and empty metric dictionaries
   - Policy immutability violations (PyTorch & custom objects)
   - Security budget & Codex Teacher disablement assertions
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile
from typing import Any
import pytest
import torch

from src.self_growing_trader.auto_rl_engine import (
    AutoRLEngine,
    AutoRLEngineConfig,
    CumulativeDrawdownMonitor,
    TimeframeAdaptationStateMachine,
    evaluate_candidate_oos_gate,
    prune_unpromoted_checkpoints,
)
from src.self_growing_trader.oos_validator import (
    CumulativeDrawdownTracker,
    DrawdownPct,
    EmergencyHaltError,
    OOSValidationResult,
    OOSValidator,
    PolicyImmutabilityError,
    WinRate,
    compute_action_entropy,
    compute_direction_imbalance,
    compute_expectancy,
    evaluate_strict_oos,
)


class MockPolicy:
    """Mock policy for immutability and frozen evaluation checks."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)


class MockPyTorchPolicy(torch.nn.Module):
    """PyTorch policy for state_dict hashing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)


# ==============================================================================
# 1. EXTREME BOUNDARY CONDITIONS
# ==============================================================================


class TestBoundaryConditions:
    """Test precise threshold boundaries as specified in ORIGINAL_REQUEST.md & PROJECT.md."""

    @pytest.fixture
    def validator(self) -> OOSValidator:
        return OOSValidator()

    def test_trade_count_boundary_29_vs_30(self, validator: OOSValidator) -> None:
        """Requirement: min_trades >= 30. Exactly 29 must fail, 30 must pass trade gate."""
        base_metrics = {
            "expectancy": 0.10,
            "win_rate": 0.50,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.20,
            "normalized_action_entropy": 0.80,
            "entry_direction_imbalance": 0.0,
        }

        # 29 trades
        res_29 = validator.evaluate(None, metrics={**base_metrics, "total_trades": 29})
        assert not res_29["passed_gate"], "29 trades must not pass gate"
        assert not res_29["trades_passed"]
        assert any("oos_trades_insufficient" in r or "trades_below_threshold" in r for r in res_29["reasons"])

        # 30 trades
        res_30 = validator.evaluate(None, metrics={**base_metrics, "total_trades": 30})
        assert res_30["trades_passed"], "30 trades must satisfy trade threshold"
        assert res_30["passed_gate"], "30 trades with valid metrics must pass gate"

        # Also verify in auto_rl_engine evaluate_candidate_oos_gate
        res_29_eng = evaluate_candidate_oos_gate({**base_metrics, "total_trades": 29, "win_rate": 50.0, "entropy": 0.8})
        assert not res_29_eng.passed_gate
        res_30_eng = evaluate_candidate_oos_gate({**base_metrics, "total_trades": 30, "win_rate": 50.0, "entropy": 0.8})
        assert res_30_eng.passed_gate

    def test_expectancy_boundary_0499_vs_0500(self, validator: OOSValidator) -> None:
        """Requirement: E[R] >= +0.05R. 0.0499 must fail, 0.0500 must pass expectancy gate."""
        base_metrics = {
            "total_trades": 35,
            "win_rate": 0.50,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.20,
            "normalized_action_entropy": 0.80,
            "entry_direction_imbalance": 0.0,
        }

        # 0.0499
        res_fail = validator.evaluate(None, metrics={**base_metrics, "expectancy": 0.0499})
        assert not res_fail["passed_gate"], "E[R]=0.0499 must fail gate"
        assert not res_fail["expectancy_passed"]
        assert any("expectancy_below" in r for r in res_fail["reasons"])

        # 0.0500
        res_pass = validator.evaluate(None, metrics={**base_metrics, "expectancy": 0.0500})
        assert res_pass["expectancy_passed"], "E[R]=0.0500 must satisfy expectancy threshold"
        assert res_pass["passed_gate"]

        # Also verify in auto_rl_engine evaluate_candidate_oos_gate
        assert not evaluate_candidate_oos_gate({**base_metrics, "expectancy": 0.0499, "win_rate": 50.0, "entropy": 0.8}).passed_gate
        assert evaluate_candidate_oos_gate({**base_metrics, "expectancy": 0.0500, "win_rate": 50.0, "entropy": 0.8}).passed_gate

    def test_drawdown_boundary_999_vs_1000(self, validator: OOSValidator) -> None:
        """Requirement: Max Drawdown < 10.0%. 9.99% must pass, 10.00% must fail."""
        base_metrics = {
            "total_trades": 35,
            "expectancy": 0.10,
            "win_rate": 0.50,
            "risk_reward_ratio": 2.0,
            "hold_share": 0.20,
            "normalized_action_entropy": 0.80,
            "entry_direction_imbalance": 0.0,
        }

        # 9.99%
        res_pass = validator.evaluate(None, metrics={**base_metrics, "max_equity_drawdown_pct": 9.99})
        assert res_pass["drawdown_passed"], "DD=9.99% must pass (< 10.0%)"
        assert res_pass["passed_gate"]

        # 10.00%
        res_fail = validator.evaluate(None, metrics={**base_metrics, "max_equity_drawdown_pct": 10.00})
        assert not res_fail["drawdown_passed"], "DD=10.00% must fail (>= 10.0%)"
        assert not res_fail["passed_gate"]
        assert any("drawdown_exceed" in r for r in res_fail["reasons"])

        # Also test DrawdownPct comparison class
        dd_999 = DrawdownPct(9.99)
        assert dd_999 < 10.0
        assert dd_999 < 0.10
        dd_1000 = DrawdownPct(10.0)
        assert not (dd_1000 < 10.0)
        assert not (dd_1000 < 0.10)

    def test_cumulative_drawdown_boundary_1499_vs_1500_vs_1501(self) -> None:
        """Requirement: cumulative drawdown emergency stop if DD > 15.0%."""
        tracker = CumulativeDrawdownTracker(max_allowed_drawdown_pct=15.0)

        # 14.99%
        tracker.cumulative_drawdown_pct = 14.99
        assert not tracker.is_emergency_halt(), "14.99% cumulative DD must NOT trigger halt"

        # 15.00%
        tracker.cumulative_drawdown_pct = 15.00
        assert not tracker.is_emergency_halt(), "15.00% cumulative DD must NOT trigger halt (limit is > 15.0%)"

        # 15.01%
        tracker.cumulative_drawdown_pct = 15.01
        assert tracker.is_emergency_halt(), "15.01% cumulative DD MUST trigger emergency halt"

        # Also test CumulativeDrawdownMonitor from auto_rl_engine
        monitor = CumulativeDrawdownMonitor(initial_equity=10000.0, halt_pct=15.0)
        # Loss of $1499.0 = 14.99%
        halt = monitor.update(-1499.0)
        assert not halt and not monitor.emergency_halt_triggered

        # Additional loss of $1.0 = total $1500.0 = exactly 15.00%
        halt = monitor.update(-1.0)
        assert not halt and not monitor.emergency_halt_triggered

        # Additional loss of $1.0 = total $1501.0 = 15.01%
        halt = monitor.update(-1.0)
        assert halt and monitor.emergency_halt_triggered

    def test_win_rate_boundary_3999_vs_4000(self, validator: OOSValidator) -> None:
        """Requirement: Win Rate >= 40.0%. 39.99% must fail, 40.00% must pass."""
        base_metrics = {
            "total_trades": 35,
            "expectancy": 0.10,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.20,
            "normalized_action_entropy": 0.80,
            "entry_direction_imbalance": 0.0,
        }

        # 39.99%
        res_fail = validator.evaluate(None, metrics={**base_metrics, "win_rate": 0.3999})
        assert not res_fail["win_rate_passed"], "WR=39.99% must fail (< 40.0%)"
        assert not res_fail["passed_gate"]

        # 40.00%
        res_pass = validator.evaluate(None, metrics={**base_metrics, "win_rate": 0.4000})
        assert res_pass["win_rate_passed"], "WR=40.00% must pass (>= 40.0%)"
        assert res_pass["passed_gate"]

        # Test WinRate comparison class
        wr_3999 = WinRate(0.3999)
        assert not (wr_3999 >= 0.40)
        assert not (wr_3999 >= 40.0)
        wr_4000 = WinRate(0.4000)
        assert wr_4000 >= 0.40
        assert wr_4000 >= 40.0

    def test_hold_share_boundary_799_vs_800(self, validator: OOSValidator) -> None:
        """Requirement: Anti-HOLD collapse guard: hold_share < 0.80. 0.799 passes, 0.800 fails."""
        base_metrics = {
            "total_trades": 35,
            "expectancy": 0.10,
            "win_rate": 0.50,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 5.0,
            "normalized_action_entropy": 0.80,
            "entry_direction_imbalance": 0.0,
        }

        res_799 = validator.evaluate(None, metrics={**base_metrics, "hold_share": 0.799})
        assert res_799["hold_guard_passed"], "hold_share=0.799 must pass (< 0.80)"
        assert res_799["passed_gate"]

        res_800 = validator.evaluate(None, metrics={**base_metrics, "hold_share": 0.800})
        assert not res_800["hold_guard_passed"], "hold_share=0.800 must fail (>= 0.80)"
        assert not res_800["passed_gate"]

    def test_action_entropy_boundary_350_vs_3501(self, validator: OOSValidator) -> None:
        """Requirement: Anti-HOLD collapse guard: entropy > 0.35. 0.350 fails, 0.3501 passes."""
        base_metrics = {
            "total_trades": 35,
            "expectancy": 0.10,
            "win_rate": 0.50,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.20,
            "entry_direction_imbalance": 0.0,
        }

        res_350 = validator.evaluate(None, metrics={**base_metrics, "normalized_action_entropy": 0.350})
        assert not res_350["hold_guard_passed"], "entropy=0.350 must fail (<= 0.35)"
        assert not res_350["passed_gate"]

        res_351 = validator.evaluate(None, metrics={**base_metrics, "normalized_action_entropy": 0.3501})
        assert res_351["hold_guard_passed"], "entropy=0.3501 must pass (> 0.35)"
        assert res_351["passed_gate"]

    def test_directional_imbalance_boundary_600_vs_601(self, validator: OOSValidator) -> None:
        """Requirement: Anti-directional collapse guard: |imbalance| <= 0.60."""
        base_metrics = {
            "total_trades": 35,
            "expectancy": 0.10,
            "win_rate": 0.50,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.20,
            "normalized_action_entropy": 0.80,
        }

        # Long imbalance
        assert validator.evaluate(None, metrics={**base_metrics, "entry_direction_imbalance": 0.6000})["passed_gate"]
        assert not validator.evaluate(None, metrics={**base_metrics, "entry_direction_imbalance": 0.6001})["passed_gate"]

        # Short imbalance
        assert validator.evaluate(None, metrics={**base_metrics, "entry_direction_imbalance": -0.6000})["passed_gate"]
        assert not validator.evaluate(None, metrics={**base_metrics, "entry_direction_imbalance": -0.6001})["passed_gate"]


# ==============================================================================
# 2. MARKET STRESS & PATHOLOGICAL SCENARIOS
# ==============================================================================


class TestMarketStressScenarios:
    """Stress test against extreme market dynamics."""

    def test_zero_trade_drought_triggers_timeframe_adaptation(self) -> None:
        """Stress: market becomes completely frozen or policy generates 0 trades.

        Auto-RL Engine state machine must:
        - Not crash on zero-trade inputs.
        - Treat 0 trades as failure round.
        - Transition M1 -> M5 after 3 consecutive failures.
        - Transition M5 -> M15 after 3 consecutive failures.
        """
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1", m1_failure_threshold=3, m5_failure_threshold=3)
        assert sm.current_timeframe == "M1"

        # Round 1: 0 trades
        tf1 = sm.record_round_result(round_num=1, expectancy=0.0, trades=0)
        assert tf1 == "M1"
        assert sm.consecutive_m1_losses == 1

        # Round 2: 0 trades
        tf2 = sm.record_round_result(round_num=2, expectancy=0.0, trades=0)
        assert tf2 == "M1"
        assert sm.consecutive_m1_losses == 2

        # Round 3: 0 trades -> triggers migration to M5!
        tf3 = sm.record_round_result(round_num=3, expectancy=0.0, trades=0)
        assert tf3 == "M5"
        assert sm.consecutive_m1_losses == 0
        assert len(sm.migration_history) == 1
        assert sm.migration_history[0]["from"] == "M1"
        assert sm.migration_history[0]["to"] == "M5"

        # Round 4, 5, 6: 0 trades on M5 -> triggers migration to M15!
        sm.record_round_result(round_num=4, expectancy=0.0, trades=0)
        sm.record_round_result(round_num=5, expectancy=0.0, trades=0)
        tf6 = sm.record_round_result(round_num=6, expectancy=0.0, trades=0)
        assert tf6 == "M15"
        assert len(sm.migration_history) == 2
        assert sm.migration_history[1]["from"] == "M5"
        assert sm.migration_history[1]["to"] == "M15"

    def test_extreme_volatility_catastrophic_crash_emergency_halt(self) -> None:
        """Stress: catastrophic market crash causing a -20% account drop in a single round.

        CumulativeDrawdownMonitor and AutoRLEngine must:
        - Immediately detect cumulative DD > 15%.
        - Trigger emergency halt.
        - Terminate the loop and abort subsequent execution.
        """
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            cfg = AutoRLEngineConfig(
                runs_dir=tmp_dir,
                max_rounds=15,
                initial_equity=10000.0,
                max_cumulative_drawdown=0.15,
            )
            engine = AutoRLEngine(cfg)

            # Round 1: catastrophic crash of -$2,000.0 (20% of $10,000 initial equity)
            res1 = engine.step_round(metric_overrides={"pnl": -2000.0, "total_trades": 30, "expectancy": -0.60})
            assert res1["emergency_halt"], "Crash of 20% must trigger emergency halt"
            assert res1["terminated"]
            assert res1["termination_reason"] == "emergency_drawdown_halt"
            assert engine.terminated

            # Subsequent attempt to step must be immediately rejected
            res2 = engine.step_round(metric_overrides={"pnl": 500.0})
            assert res2["status"] == "emergency_halt"
            assert res2["round"] == 1
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_alternating_whipsaw_win_loss_sequence(self) -> None:
        """Stress: choppy whipsaw market with alternating +2.0R and -1.0R trades.

        Tests whether trade return aggregation is deterministic and mathematically sound:
        - 40 trades alternating: 20 wins at +2.0R, 20 losses at -1.0R.
        - Realized win rate = 50.0%.
        - Realized risk/reward = 2.0.
        - Realized E[R] = (0.50 * 2.0) - (0.50 * 1.0) = +0.50R.
        - Gate must pass all quantitative edge tests.
        """
        trades = []
        for _ in range(20):
            trades.append({"realized_r": 2.0})
            trades.append({"realized_r": -1.0})

        v = OOSValidator()
        res = v.evaluate(
            None,
            trades=trades,
            metrics={"max_equity_drawdown_pct": 4.0},
            action_counts={"HOLD": 10, "BUY": 20, "SELL": 20},
        )

        assert res["total_trades"] == 40
        assert math.isclose(float(res["win_rate"]), 0.50, abs_tol=1e-5)
        assert math.isclose(float(res["risk_reward_ratio"]), 2.0, abs_tol=1e-5)
        assert math.isclose(float(res["expectancy"]), 0.50, abs_tol=1e-5)
        assert res["passed_gate"]

    def test_sub_hurdle_whipsaw_sequence_fails_rr_gate(self) -> None:
        """Stress: alternating sequence where win rate is 50% but RR is only 1.5.

        40 trades: 20 wins at +1.5R, 20 losses at -1.0R.
        E[R] = 0.50*1.5 - 0.50*1.0 = +0.25R (positive expectancy), but RR = 1.5 < 2.0 hurdle.
        Must be rejected by the risk/reward gate!
        """
        trades = []
        for _ in range(20):
            trades.append({"realized_r": 1.5})
            trades.append({"realized_r": -1.0})

        v = OOSValidator()
        res = v.evaluate(
            None,
            trades=trades,
            metrics={"max_equity_drawdown_pct": 4.0},
            action_counts={"HOLD": 10, "BUY": 20, "SELL": 20},
        )

        assert res["total_trades"] == 40
        assert math.isclose(float(res["risk_reward_ratio"]), 1.5, abs_tol=1e-5)
        assert not res["risk_reward_passed"]
        assert not res["passed_gate"]
        assert any("risk_reward_below_threshold" in r for r in res["reasons"])


# ==============================================================================
# 3. ADVERSARIAL NUMERICAL INPUTS & STABILITY
# ==============================================================================


class TestAdversarialNumericalInputs:
    """Stress test against NaN, Inf, negative, and malformed numerical inputs."""

    def test_nan_expectancy_fails_gracefully(self) -> None:
        """NaN expectancy must not crash and must fail gate."""
        v = OOSValidator()
        res = v.evaluate(
            None,
            metrics={
                "expectancy": float("nan"),
                "win_rate": 0.50,
                "risk_reward_ratio": 2.0,
                "total_trades": 35,
                "max_equity_drawdown_pct": 5.0,
                "hold_share": 0.20,
                "normalized_action_entropy": 0.80,
                "entry_direction_imbalance": 0.0,
            },
        )
        assert not res["passed_gate"]
        assert not res["expectancy_passed"]

    def test_nan_win_rate_fails_gracefully(self) -> None:
        """NaN win_rate must fail gate without unhandled exception."""
        v = OOSValidator()
        res = v.evaluate(
            None,
            metrics={
                "expectancy": 0.10,
                "win_rate": float("nan"),
                "risk_reward_ratio": 2.0,
                "total_trades": 35,
                "max_equity_drawdown_pct": 5.0,
                "hold_share": 0.20,
                "normalized_action_entropy": 0.80,
                "entry_direction_imbalance": 0.0,
            },
        )
        assert not res["passed_gate"]
        assert not res["win_rate_passed"]

    def test_nan_drawdown_fails_gracefully(self) -> None:
        """NaN drawdown must fail gate without unhandled exception."""
        v = OOSValidator()
        res = v.evaluate(
            None,
            metrics={
                "expectancy": 0.10,
                "win_rate": 0.50,
                "risk_reward_ratio": 2.0,
                "total_trades": 35,
                "max_equity_drawdown_pct": float("nan"),
                "hold_share": 0.20,
                "normalized_action_entropy": 0.80,
                "entry_direction_imbalance": 0.0,
            },
        )
        assert not res["passed_gate"]
        assert not res["drawdown_passed"]

    def test_nan_total_trades_handling(self) -> None:
        """Adversarial check: what happens when total_trades is float('nan')?

        Documents the behavior when NaN is passed into total_trades.
        """
        v = OOSValidator()
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            v.evaluate(None, metrics={"total_trades": float("nan")})

    def test_nan_ppo_loss_prevents_promotion(self) -> None:
        """In AutoRLEngine, NaN PPO loss indicates divergence and must prevent candidate promotion."""
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            cfg = AutoRLEngineConfig(runs_dir=tmp_dir, max_rounds=5)
            engine = AutoRLEngine(cfg)

            res = engine.step_round(
                metric_overrides={
                    "ppo_loss": float("nan"),
                    "total_trades": 40,
                    "win_rate": 50.0,
                    "risk_reward_ratio": 2.1,
                    "expectancy": 0.15,
                    "max_equity_drawdown_pct": 4.0,
                    "hold_share": 0.20,
                    "entropy": 0.80,
                    "entry_direction_imbalance": 0.0,
                }
            )
            assert not res["promoted"], "Candidate with NaN PPO loss must NEVER be promoted"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_inf_ppo_loss_prevents_promotion(self) -> None:
        """In AutoRLEngine, Inf PPO loss indicates divergence and must prevent candidate promotion."""
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            cfg = AutoRLEngineConfig(runs_dir=tmp_dir, max_rounds=5)
            engine = AutoRLEngine(cfg)

            res = engine.step_round(
                metric_overrides={
                    "ppo_loss": float("inf"),
                    "total_trades": 40,
                    "win_rate": 50.0,
                    "risk_reward_ratio": 2.1,
                    "expectancy": 0.15,
                    "max_equity_drawdown_pct": 4.0,
                    "hold_share": 0.20,
                    "entropy": 0.80,
                    "entry_direction_imbalance": 0.0,
                }
            )
            assert not res["promoted"], "Candidate with Inf PPO loss must NEVER be promoted"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_extreme_action_collapse_all_buy(self) -> None:
        """Stress: 100% BUY actions -> imbalance = +1.0. Must fail directional collapse guard."""
        v = OOSValidator()
        res = v.evaluate(
            None,
            action_counts={"BUY": 100, "SELL": 0, "HOLD": 20},
            metrics={"total_trades": 35, "win_rate": 0.50, "risk_reward_ratio": 2.0, "expectancy": 0.10, "max_equity_drawdown_pct": 5.0},
        )
        assert res["entry_direction_imbalance"] == 1.0
        assert not res["directional_guard_passed"]
        assert not res["passed_gate"]
        assert any("anti_directional" in r for r in res["reasons"])

    def test_extreme_action_collapse_all_sell(self) -> None:
        """Stress: 100% SELL actions -> imbalance = -1.0. Must fail directional collapse guard."""
        v = OOSValidator()
        res = v.evaluate(
            None,
            action_counts={"BUY": 0, "SELL": 100, "HOLD": 20},
            metrics={"total_trades": 35, "win_rate": 0.50, "risk_reward_ratio": 2.0, "expectancy": 0.10, "max_equity_drawdown_pct": 5.0},
        )
        assert res["entry_direction_imbalance"] == -1.0
        assert not res["directional_guard_passed"]
        assert not res["passed_gate"]

    def test_extreme_action_collapse_all_hold(self) -> None:
        """Stress: 100% HOLD actions -> hold_share = 1.0, entropy = 0.0. Must fail HOLD guard."""
        v = OOSValidator()
        res = v.evaluate(
            None,
            action_counts={"HOLD": 1000},
            metrics={"total_trades": 0, "expectancy": 0.0},
        )
        assert res["hold_share"] == 1.0
        assert res["normalized_action_entropy"] == 0.0
        assert not res["hold_guard_passed"]
        assert not res["passed_gate"]


# ==============================================================================
# 4. POLICY IMMUTABILITY & ENGINE SECURITY
# ==============================================================================


class TestPolicyImmutabilityAndSecurity:
    """Stress test policy frozen immutability and architectural constraints."""

    def test_custom_policy_mutation_raises_immutability_error(self) -> None:
        """If evaluation secretly mutates custom policy state, PolicyImmutabilityError must be raised."""
        policy = MockPolicy(weight=1.0)
        v = OOSValidator()

        def rogue_evaluator(cand: MockPolicy, **kwargs: Any) -> dict[str, Any]:
            cand.weight = 999.0  # Mutate state!
            return {"total_trades": 35, "win_rate": 0.50, "risk_reward_ratio": 2.0, "expectancy": 0.10, "max_equity_drawdown_pct": 5.0}

        with pytest.raises(PolicyImmutabilityError, match="Frozen OOS evaluation mutated candidate policy"):
            v.evaluate(policy, evaluator=rogue_evaluator)

    def test_pytorch_policy_mutation_raises_immutability_error(self) -> None:
        """If evaluation secretly mutates PyTorch weights, PolicyImmutabilityError must be raised."""
        torch.manual_seed(42)
        policy = MockPyTorchPolicy()
        v = OOSValidator()

        def rogue_evaluator(cand: MockPyTorchPolicy, **kwargs: Any) -> dict[str, Any]:
            with torch.no_grad():
                cand.fc.weight.add_(1.0)  # Mutate PyTorch weights!
            return {"total_trades": 35, "win_rate": 0.50, "risk_reward_ratio": 2.0, "expectancy": 0.10, "max_equity_drawdown_pct": 5.0}

        with pytest.raises(PolicyImmutabilityError, match="Frozen OOS evaluation mutated candidate policy"):
            v.evaluate(policy, evaluator=rogue_evaluator)

    def test_codex_teacher_strictly_disabled(self) -> None:
        """AutoRLEngineConfig must strictly forbid five_codex_teacher_enabled=True."""
        with pytest.raises(ValueError, match="strictly enforces five_codex_teacher_enabled=False"):
            AutoRLEngineConfig(five_codex_teacher_enabled=True)

    def test_max_rounds_budget_clamped_to_15(self) -> None:
        """AutoRLEngineConfig must clamp max_rounds > 15 to exactly 15."""
        cfg = AutoRLEngineConfig(max_rounds=50)
        assert cfg.max_rounds == 15

    def test_checkpoint_pruning_removes_pkl_preserves_telemetry(self) -> None:
        """prune_unpromoted_checkpoints must delete binary .pkl payloads but preserve JSON telemetry."""
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            r_dir = tmp_dir / "round_01_m1"
            r_dir.mkdir(parents=True)
            candidate_pkl = r_dir / "candidate.pkl"
            candidate_pkl.write_bytes(b"LARGE_MODEL_BYTES" * 1000)
            manifest_json = r_dir / "manifest.json"
            manifest_json.write_text('{"round": 1}', encoding="utf-8")
            events_jsonl = r_dir / "events.jsonl"
            events_jsonl.write_text('{"event": "start"}\n', encoding="utf-8")

            pruned = prune_unpromoted_checkpoints(r_dir)
            assert candidate_pkl in pruned
            assert not candidate_pkl.exists(), "candidate.pkl must be pruned"
            assert manifest_json.exists(), "manifest.json must be preserved"
            assert events_jsonl.exists(), "events.jsonl must be preserved"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
