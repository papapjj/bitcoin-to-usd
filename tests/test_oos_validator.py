"""Unit tests for Strict Out-of-Sample (OOS) Validator & Anti-Collapse Guards.

Verifies:
1. Dedicated Q1 2025 OOS window defaults ("2025.01.01" to "2025.03.31").
2. Frozen policy immutability assertion (before and after evaluation).
3. Formulas: Expectancy E[R], Win Rate, Risk/Reward (TP/SL), Action Entropy, Directional Imbalance.
4. Anti-HOLD collapse guard (hold_share < 0.80, entropy > 0.35, valid trades >= 30).
5. Anti-directional collapse guard (|imbalance| <= 0.60).
6. Quantitative gate criteria (E[R] >= +0.05R, WR >= 40%, RR >= 2.0, trades >= 30, DD < 10.0%).
7. Cumulative Drawdown Tracker (sequential tracking across rounds, emergency halt if DD > 15%).
8. Interface contracts and serialization (PROJECT.md contract).
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from self_growing_trader.oos_validator import (
    DEFAULT_MAX_CUMULATIVE_DRAWDOWN_PCT,
    DEFAULT_MAX_DIRECTION_IMBALANCE,
    DEFAULT_MAX_DRAWDOWN_PCT,
    DEFAULT_MAX_HOLD_SHARE,
    DEFAULT_MIN_EXPECTANCY,
    DEFAULT_MIN_NORMALIZED_ENTROPY,
    DEFAULT_MIN_RISK_REWARD,
    DEFAULT_MIN_TRADES,
    DEFAULT_MIN_WIN_RATE,
    DEFAULT_OOS_FROM_DATE,
    DEFAULT_OOS_TO_DATE,
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
    """Mock policy that maintains real internal state."""

    def __init__(self, weight: float = 1.0, name: str = "CandidatePolicy") -> None:
        self.weight = float(weight)
        self.name = str(name)
        self.step_count = 0

    def predict(self, features: Any) -> int:
        return 1


class TestWindowAndDefaults:
    """Verify OOS window defaults and constants."""

    def test_default_dates(self) -> None:
        assert DEFAULT_OOS_FROM_DATE == "2025.01.01"
        assert DEFAULT_OOS_TO_DATE == "2025.03.31"

    def test_validator_default_window(self) -> None:
        validator = OOSValidator()
        assert validator.from_date == "2025.01.01"
        assert validator.to_date == "2025.03.31"

    def test_evaluate_returns_default_window(self) -> None:
        policy = MockPolicy()
        result = evaluate_strict_oos(
            policy,
            metrics={"total_trades": 35, "win_rate": 0.50, "risk_reward_ratio": 2.1, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["from_date"] == "2025.01.01"
        assert result["to_date"] == "2025.03.31"

    def test_evaluate_custom_window(self) -> None:
        policy = MockPolicy()
        result = evaluate_strict_oos(
            policy,
            from_date="2025.02.01",
            to_date="2025.02.28",
            metrics={"total_trades": 35, "win_rate": 0.50, "risk_reward_ratio": 2.1, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["from_date"] == "2025.02.01"
        assert result["to_date"] == "2025.02.28"


class TestPolicyImmutabilityAssertion:
    """Verify frozen hash immutability assertion on candidate policy."""

    def test_immutable_policy_passes_hash_check(self) -> None:
        policy = MockPolicy(weight=2.5)
        validator = OOSValidator()

        def safe_evaluator(pol: MockPolicy, **kwargs: Any) -> dict[str, Any]:
            # Read-only evaluation
            _ = pol.weight * 2.0
            return {
                "metrics": {
                    "total_trades": 40,
                    "win_rate": 0.50,
                    "risk_reward_ratio": 2.2,
                    "equity_drawdown_pct": 6.0,
                }
            }

        result = validator.evaluate(
            policy,
            evaluator=safe_evaluator,
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["policy_hash_before"] == result["policy_hash_after"]
        assert len(result["policy_hash_before"]) == 64

    def test_mutated_policy_raises_immutability_error(self) -> None:
        policy = MockPolicy(weight=1.0)
        validator = OOSValidator()

        def mutating_evaluator(pol: MockPolicy, **kwargs: Any) -> dict[str, Any]:
            # Mutates policy state during evaluation!
            pol.weight += 0.5
            return {"metrics": {"total_trades": 50}}

        with pytest.raises(PolicyImmutabilityError) as exc_info:
            validator.evaluate(policy, evaluator=mutating_evaluator)
        assert "mutated candidate policy" in str(exc_info.value)
        assert isinstance(exc_info.value, RuntimeError)

    def test_attribute_mutation_detected(self) -> None:
        policy = MockPolicy(weight=1.0)
        validator = OOSValidator()

        def step_incrementing_evaluator(pol: MockPolicy, **kwargs: Any) -> dict[str, Any]:
            pol.step_count += 1
            return {"metrics": {"total_trades": 50}}

        with pytest.raises(RuntimeError):
            validator.evaluate(policy, evaluator=step_incrementing_evaluator)


class TestMathematicalFormulas:
    """Verify Expectancy, WinRate, Risk/Reward, Entropy, and Directional Imbalance."""

    def test_expectancy_standard_formula(self) -> None:
        # E[R] = (w * RR) - (1 - w)
        # 40% WR, 2.0 RR -> (0.40 * 2.0) - 0.60 = +0.20R
        assert math.isclose(compute_expectancy(win_rate=0.40, risk_reward=2.0), 0.20, rel_tol=1e-5)
        # 35% WR, 2.0 RR -> (0.35 * 2.0) - 0.65 = +0.05R (exact gate boundary!)
        assert math.isclose(compute_expectancy(win_rate=0.35, risk_reward=2.0), 0.05, rel_tol=1e-5)
        # 30% WR, 2.0 RR -> (0.30 * 2.0) - 0.70 = -0.10R (negative drift)
        assert math.isclose(compute_expectancy(win_rate=0.30, risk_reward=2.0), -0.10, rel_tol=1e-5)
        # 50% WR, 2.0 RR -> (0.50 * 2.0) - 0.50 = +0.50R
        assert math.isclose(compute_expectancy(win_rate=0.50, risk_reward=2.0), 0.50, rel_tol=1e-5)

    def test_expectancy_with_percentage_input(self) -> None:
        # 45% passed as 45.0
        exp = compute_expectancy(win_rate=45.0, risk_reward=2.0)
        expected = (0.45 * 2.0) - 0.55  # 0.90 - 0.55 = 0.35R
        assert math.isclose(exp, expected, rel_tol=1e-5)

    def test_expectancy_from_trade_returns(self) -> None:
        # 3 trades: +2.0R, -1.0R, +2.0R -> mean = 3.0 / 3 = +1.0R
        trades = [2.0, -1.0, 2.0]
        assert math.isclose(compute_expectancy(0.0, 0.0, trade_returns=trades), 1.0, rel_tol=1e-5)

    def test_win_rate_flexible_comparisons(self) -> None:
        wr = WinRate(0.45)
        assert wr >= 0.40
        assert wr >= 40.0
        assert wr > 0.40
        assert wr > 40.0
        assert wr <= 0.50
        assert wr <= 50.0
        assert wr < 0.50
        assert wr < 50.0
        assert wr == 0.45
        assert wr == 45.0
        assert isinstance(wr, float)
        # Reflected comparisons
        assert 0.40 <= wr
        assert 40.0 <= wr
        assert 0.50 > wr
        assert 50.0 > wr

    def test_drawdown_pct_flexible_comparisons(self) -> None:
        dd = DrawdownPct(8.5)
        assert dd < 10.0
        assert dd < 0.10
        assert dd <= 10.0
        assert dd <= 0.10
        assert dd > 5.0
        assert dd > 0.05
        assert dd == 8.5
        assert dd == 0.085
        assert isinstance(dd, float)

    def test_action_entropy_uniform_distribution(self) -> None:
        # Equal distribution across 3 actions -> maximum entropy ln(3) -> normalized = 1.0
        counts = {"HOLD": 50, "BUY": 50, "SELL": 50}
        raw_ent, norm_ent = compute_action_entropy(counts)
        assert math.isclose(norm_ent, 1.0, rel_tol=1e-4)
        assert norm_ent > DEFAULT_MIN_NORMALIZED_ENTROPY

    def test_action_entropy_collapsed_distribution(self) -> None:
        # 98 HOLD, 1 BUY, 1 SELL -> entropic collapse
        counts = {"HOLD": 98, "BUY": 1, "SELL": 1}
        raw_ent, norm_ent = compute_action_entropy(counts)
        assert norm_ent < 0.35

    def test_action_entropy_single_action(self) -> None:
        # 100% HOLD -> entropy 0.0
        counts = {"HOLD": 100}
        raw_ent, norm_ent = compute_action_entropy(counts)
        assert norm_ent == 0.0

    def test_direction_imbalance_balanced(self) -> None:
        # 50 BUY, 50 SELL -> imbalance = 0.0
        assert math.isclose(compute_direction_imbalance(long_count=50, short_count=50), 0.0)

    def test_direction_imbalance_exact_boundary(self) -> None:
        # 80 BUY, 20 SELL -> (80 - 20) / 100 = 0.60
        imb = compute_direction_imbalance(long_count=80, short_count=20)
        assert math.isclose(imb, 0.60, rel_tol=1e-5)
        assert abs(imb) <= DEFAULT_MAX_DIRECTION_IMBALANCE

    def test_direction_imbalance_exceeded(self) -> None:
        # 90 BUY, 10 SELL -> 0.80 > 0.60
        imb = compute_direction_imbalance(long_count=90, short_count=10)
        assert math.isclose(imb, 0.80, rel_tol=1e-5)
        assert abs(imb) > DEFAULT_MAX_DIRECTION_IMBALANCE

    def test_direction_imbalance_from_action_counts(self) -> None:
        counts = {"HOLD": 50, "MARKET_BUY": 30, "MARKET_SELL": 30}
        imb = compute_direction_imbalance(counts)
        assert math.isclose(imb, 0.0, rel_tol=1e-5)

    def test_direction_imbalance_zero_trades(self) -> None:
        counts = {"HOLD": 100}
        assert compute_direction_imbalance(counts) == 0.0


class TestAntiHoldCollapseGuard:
    """Verify anti-HOLD collapse guard: hold_share < 0.80, entropy > 0.35, valid trades >= 30."""

    def test_valid_active_policy_passes(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 40, "win_rate": 0.45, "risk_reward_ratio": 2.2, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 40, "BUY": 30, "SELL": 30},  # hold_share = 40/100 = 0.40 < 0.80
        )
        assert result["hold_guard_passed"] is True
        assert result["hold_share"] == 0.40
        assert result["normalized_action_entropy"] > 0.35
        assert result["total_trades"] >= 30

    def test_excessive_hold_share_fails_guard(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 40, "win_rate": 0.50, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 850, "BUY": 75, "SELL": 75},  # hold_share = 850/1000 = 0.85 >= 0.80
        )
        assert result["hold_guard_passed"] is False
        assert result["passed_gate"] is False
        assert any("hold_share_too_high" in r for r in result["reasons"])

    def test_low_entropy_fails_guard(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        # Even with trades, if distribution is entropic collapse
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 35, "win_rate": 0.50, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            hold_share=0.75,
            normalized_action_entropy=0.20,  # <= 0.35
            entry_direction_imbalance=0.10,
        )
        assert result["hold_guard_passed"] is False
        assert result["passed_gate"] is False
        assert any("entropy_too_low" in r for r in result["reasons"])

    def test_insufficient_trades_fails_guard(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 25, "win_rate": 0.50, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            hold_share=0.50,
            normalized_action_entropy=0.80,
            entry_direction_imbalance=0.10,
        )
        assert result["hold_guard_passed"] is False
        assert result["passed_gate"] is False
        assert any("insufficient_trades" in r for r in result["reasons"])


class TestAntiDirectionalCollapseGuard:
    """Verify anti-directional collapse guard: |entry_direction_imbalance| <= 0.60."""

    def test_balanced_directional_trading_passes(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 35, "win_rate": 0.45, "risk_reward_ratio": 2.1, "equity_drawdown_pct": 6.0},
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["directional_guard_passed"] is True
        assert abs(result["entry_direction_imbalance"]) <= 0.60

    def test_buy_directional_collapse_fails(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        # 95% BUY, 5% SELL -> imbalance = (95 - 5) / 100 = 0.90 > 0.60
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 40, "win_rate": 0.50, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 30, "BUY": 95, "SELL": 5},
        )
        assert result["directional_guard_passed"] is False
        assert result["passed_gate"] is False
        assert any("imbalance_exceeded" in r for r in result["reasons"])

    def test_sell_directional_collapse_fails(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        # 10% BUY, 90% SELL -> imbalance = (10 - 90) / 100 = -0.80 (|imb| = 0.80 > 0.60)
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 40, "win_rate": 0.50, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 30, "BUY": 10, "SELL": 90},
        )
        assert result["directional_guard_passed"] is False
        assert result["passed_gate"] is False
        assert any("imbalance_exceeded" in r for r in result["reasons"])


class TestGateCriteriaVerification:
    """Verify full Gate criteria: E[R] >= 0.05R, WR >= 40%, RR >= 2.0, trades >= 30, DD < 10.0%."""

    def test_champion_candidate_passes_gate(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={
                "total_trades": 45,
                "win_rate": 44.0,
                "risk_reward_ratio": 2.2,
                "equity_drawdown_pct": 6.8,
            },
            action_counts={"HOLD": 40, "BUY": 35, "SELL": 35},
        )
        assert result["passed_gate"] is True
        assert result["expectancy"] >= 0.05
        assert result["win_rate"] >= 0.40
        assert result["risk_reward_ratio"] >= 2.0
        assert result["total_trades"] >= 30
        assert result["max_equity_drawdown_pct"] < 10.0
        assert result["reasons"] == []

    def test_low_expectancy_fails_gate(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={
                "total_trades": 40,
                "win_rate": 0.30,  # 30% WR with 2.0 RR -> E[R] = -0.10R
                "risk_reward_ratio": 2.0,
                "equity_drawdown_pct": 6.0,
            },
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["expectancy_passed"] is False
        assert result["passed_gate"] is False
        assert any("expectancy_below_threshold" in r for r in result["reasons"])

    def test_low_win_rate_fails_gate(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        # E[R] = (0.35 * 3.5) - 0.65 = 1.225 - 0.65 = +0.575R > 0.05R, but WR = 35% < 40%
        result = validator.evaluate(
            policy,
            metrics={
                "total_trades": 40,
                "win_rate": 0.35,
                "risk_reward_ratio": 3.5,
                "equity_drawdown_pct": 6.0,
            },
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["win_rate_passed"] is False
        assert result["passed_gate"] is False
        assert any("win_rate_below_threshold" in r for r in result["reasons"])

    def test_low_risk_reward_fails_gate(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={
                "total_trades": 40,
                "win_rate": 0.55,
                "risk_reward_ratio": 1.7,  # < 2.0
                "equity_drawdown_pct": 6.0,
            },
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["risk_reward_passed"] is False
        assert result["passed_gate"] is False
        assert any("risk_reward_below_threshold" in r for r in result["reasons"])

    def test_high_drawdown_fails_gate(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={
                "total_trades": 45,
                "win_rate": 0.50,
                "risk_reward_ratio": 2.2,
                "equity_drawdown_pct": 11.2,  # >= 10.0%
            },
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["drawdown_passed"] is False
        assert result["passed_gate"] is False
        assert any("drawdown_exceeds_threshold" in r for r in result["reasons"])


class TestCumulativeDrawdownTracker:
    """Verify Cumulative Drawdown Tracker across sequential evaluations."""

    def test_safe_sequential_evaluations_below_limit(self) -> None:
        tracker = CumulativeDrawdownTracker(max_allowed_drawdown_pct=15.0, initial_equity=10000.0)
        # Round 1: 4% DD
        cum1 = tracker.update(drawdown_pct=4.0)
        assert math.isclose(cum1, 4.0, rel_tol=1e-3)
        assert tracker.is_emergency_halt() is False

        # Round 2: 5% DD
        cum2 = tracker.update(drawdown_pct=5.0)
        # Compounded: 0.96 * 0.95 = 0.912 -> DD = 8.8%
        assert math.isclose(cum2, 8.8, rel_tol=1e-3)
        assert tracker.is_emergency_halt() is False

    def test_accumulating_losses_trigger_emergency_halt(self) -> None:
        tracker = CumulativeDrawdownTracker(max_allowed_drawdown_pct=15.0, initial_equity=10000.0)
        # Round 1: 8.0% loss -> equity = 9200
        tracker.update(drawdown_pct=8.0)
        assert tracker.is_emergency_halt() is False

        # Round 2: 8.5% loss -> equity = 9200 * 0.915 = 8418 -> DD = (10000 - 8418) / 10000 = 15.82% > 15.0%
        cum2 = tracker.update(drawdown_pct=8.5)
        assert cum2 > 15.0
        assert tracker.is_emergency_halt() is True

    def test_validator_sequential_tracking_triggers_emergency_halt(self) -> None:
        validator = OOSValidator(max_cumulative_drawdown_pct=15.0)
        policy = MockPolicy()

        # Call 1: 8% DD -> cumulative DD = 8% <= 15%
        res1 = validator.evaluate(
            policy,
            metrics={"total_trades": 35, "win_rate": 0.45, "risk_reward_ratio": 2.1, "equity_drawdown_pct": 8.0},
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert res1["emergency_halt"] is False
        assert res1["cumulative_drawdown_pct"] <= 15.0

        # Call 2: 8.5% DD -> cumulative DD = 15.82% > 15% -> emergency halt!
        res2 = validator.evaluate(
            policy,
            metrics={"total_trades": 35, "win_rate": 0.45, "risk_reward_ratio": 2.1, "equity_drawdown_pct": 8.5},
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert res2["emergency_halt"] is True
        assert res2["passed_gate"] is False
        assert any("emergency_halt" in r for r in res2["reasons"])

    def test_tracker_reset_restores_state(self) -> None:
        validator = OOSValidator()
        validator.drawdown_tracker.update(drawdown_pct=20.0)
        assert validator.drawdown_tracker.is_emergency_halt() is True

        validator.reset_tracker()
        assert validator.drawdown_tracker.is_emergency_halt() is False
        assert validator.drawdown_tracker.cumulative_drawdown_pct == 0.0

    def test_raise_on_emergency_halt_option(self) -> None:
        validator = OOSValidator(max_cumulative_drawdown_pct=15.0, raise_on_emergency_halt=True)
        policy = MockPolicy()
        with pytest.raises(EmergencyHaltError):
            validator.evaluate(
                policy,
                metrics={"total_trades": 35, "win_rate": 0.45, "risk_reward_ratio": 2.1, "equity_drawdown_pct": 16.0},
                action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
            )


class TestInterfaceContractsAndSerialization:
    """Verify dictionary contract, attribute access, and JSON serialization."""

    def test_result_adheres_to_project_contract(self) -> None:
        policy = MockPolicy()
        result = evaluate_strict_oos(
            policy,
            metrics={
                "total_trades": 35,
                "win_rate": 0.45,
                "risk_reward_ratio": 2.2,
                "equity_drawdown_pct": 5.5,
            },
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
            reset_tracker=True,
        )

        required_keys = [
            "expectancy",
            "win_rate",
            "risk_reward_ratio",
            "total_trades",
            "max_equity_drawdown_pct",
            "hold_share",
            "entry_direction_imbalance",
            "passed_gate",
        ]
        for key in required_keys:
            assert key in result, f"Missing required interface key: {key}"

        # Dict indexing
        assert isinstance(result["expectancy"], float)
        assert isinstance(result["win_rate"], float)
        assert isinstance(result["total_trades"], int)
        assert isinstance(result["passed_gate"], bool)

        # Attribute access
        assert result.expectancy == result["expectancy"]
        assert result.passed_gate == result["passed_gate"]
        assert result.total_trades == result["total_trades"]

    def test_result_is_json_serializable(self) -> None:
        policy = MockPolicy()
        result = evaluate_strict_oos(
            policy,
            metrics={"total_trades": 35, "win_rate": 0.45, "risk_reward_ratio": 2.2, "equity_drawdown_pct": 5.5},
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
            reset_tracker=True,
        )
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["passed_gate"] is True
        assert deserialized["total_trades"] == 35


class TestPyTorchPolicyImmutability:
    """Verify frozen immutability with genuine torch.nn.Module policies."""

    def test_torch_policy_frozen_passes(self) -> None:
        import torch
        import torch.nn as nn

        class NeuralPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = NeuralPolicy()
        validator = OOSValidator()

        def inference_evaluator(candidate: nn.Module, **kwargs: Any) -> dict[str, Any]:
            # Evaluation without mutating weights
            with torch.no_grad():
                dummy = torch.randn(2, 4)
                _ = candidate(dummy)
            return {
                "metrics": {
                    "total_trades": 35,
                    "win_rate": 0.45,
                    "risk_reward_ratio": 2.1,
                    "equity_drawdown_pct": 5.0,
                }
            }

        result = validator.evaluate(
            model,
            evaluator=inference_evaluator,
            action_counts={"HOLD": 30, "BUY": 35, "SELL": 35},
        )
        assert result["passed_gate"] is True
        assert result["policy_hash_before"] == result["policy_hash_after"]

    def test_torch_policy_weight_mutation_caught(self) -> None:
        import torch
        import torch.nn as nn

        class NeuralPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)

        model = NeuralPolicy()
        validator = OOSValidator()

        def mutating_evaluator(candidate: nn.Module, **kwargs: Any) -> dict[str, Any]:
            # Sneaky mutation during evaluation
            with torch.no_grad():
                candidate.fc.weight.add_(0.01)
            return {"metrics": {"total_trades": 35}}

        with pytest.raises(PolicyImmutabilityError) as exc_info:
            validator.evaluate(model, evaluator=mutating_evaluator)
        assert "Frozen OOS evaluation mutated candidate policy" in str(exc_info.value)


class TestEdgeCaseBoundaries:
    """Verify exact numerical boundaries for all gate criteria."""

    def test_exact_005_expectancy_passes(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        # E[R] = (0.35 * 2.0) - 0.65 = +0.0500R
        # For win_rate test, let's set WR = 0.40, RR = 1.625 -> E[R] = 0.40 * 1.625 - 0.60 = 0.65 - 0.60 = 0.05
        # Wait, if RR must be >= 2.0 and WR >= 0.40, then min possible (0.40 * 2.0) - 0.60 = 0.20R.
        # But if custom expectancy is exactly 0.05:
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.40, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 9.99},
            expectancy=0.05,
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["expectancy_passed"] is True
        assert result["expectancy"] == 0.05
        assert result["passed_gate"] is True

    def test_just_below_005_expectancy_fails(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.40, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            expectancy=0.0499,
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["expectancy_passed"] is False
        assert result["passed_gate"] is False

    def test_exact_40_percent_win_rate_passes(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.40, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["win_rate_passed"] is True
        assert result["passed_gate"] is True

    def test_39_9_percent_win_rate_fails(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.399, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["win_rate_passed"] is False
        assert result["passed_gate"] is False

    def test_exact_2_0_risk_reward_passes(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.40, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["risk_reward_passed"] is True
        assert result["passed_gate"] is True

    def test_1_99_risk_reward_fails(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.40, "risk_reward_ratio": 1.99, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["risk_reward_passed"] is False
        assert result["passed_gate"] is False

    def test_exact_30_trades_passes(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.45, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["trades_passed"] is True
        assert result["passed_gate"] is True

    def test_29_trades_fails(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 29, "win_rate": 0.45, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["trades_passed"] is False
        assert result["passed_gate"] is False

    def test_exact_10_percent_drawdown_fails(self) -> None:
        # Condition is max_drawdown < 10.0%, so exactly 10.0% fails
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.45, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 10.0},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["drawdown_passed"] is False
        assert result["passed_gate"] is False

    def test_9_99_percent_drawdown_passes(self) -> None:
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 30, "win_rate": 0.45, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 9.99},
            action_counts={"HOLD": 20, "BUY": 40, "SELL": 40},
        )
        assert result["drawdown_passed"] is True
        assert result["passed_gate"] is True


class TestMultiRoundCumulativeDrawdownRecovery:
    """Verify equity recovery and multi-round tracking in CumulativeDrawdownTracker."""

    def test_equity_recovery_resets_drawdown_from_new_peak(self) -> None:
        tracker = CumulativeDrawdownTracker(max_allowed_drawdown_pct=15.0, initial_equity=10000.0)

        # Round 1: Loss of $600 -> equity $9,400 (DD = 6%)
        tracker.update(net_profit=-600.0)
        assert math.isclose(tracker.cumulative_drawdown_pct, 6.0)
        assert tracker.is_emergency_halt() is False

        # Round 2: Massive gain of $1,600 -> equity $11,000 (New Peak $11,000!)
        tracker.update(net_profit=1600.0)
        assert tracker.peak_equity == 11000.0
        assert tracker.current_equity == 11000.0
        assert math.isclose(tracker.cumulative_drawdown_pct, 0.0)
        assert tracker.is_emergency_halt() is False

        # Round 3: Loss of $1,100 -> equity $9,900 (DD from new peak $11,000 = 10%)
        tracker.update(net_profit=-1100.0)
        assert math.isclose(tracker.cumulative_drawdown_pct, 10.0)
        assert tracker.is_emergency_halt() is False


class TestActionSpacesAndFormats:
    """Verify action counts in different formats and action spaces."""

    def test_action_counts_as_sequence_of_strings(self) -> None:
        actions = ["HOLD"] * 30 + ["BUY"] * 35 + ["SELL"] * 35
        validator = OOSValidator()
        policy = MockPolicy()
        result = validator.evaluate(
            policy,
            metrics={"total_trades": 35, "win_rate": 0.45, "risk_reward_ratio": 2.0, "equity_drawdown_pct": 5.0},
            action_counts=actions,
        )
        assert result["hold_share"] == 0.30
        assert result["entry_direction_imbalance"] == 0.0
        assert result["passed_gate"] is True

    def test_seven_action_space_integers(self) -> None:
        # 0: HOLD, 1,2,3: BUY/LONG, 4,5,6: SELL/SHORT
        counts = {
            0: 20,  # HOLD
            1: 15,  # MARKET_BUY
            2: 15,  # BUY_LIMIT
            3: 10,  # BUY_STOP -> total long = 40
            4: 15,  # MARKET_SELL
            5: 15,  # SELL_LIMIT
            6: 10,  # SELL_STOP -> total short = 40
        }
        imb = compute_direction_imbalance(counts)
        assert imb == 0.0

        ent, norm_ent = compute_action_entropy(counts)
        assert norm_ent > 0.80

    def test_three_action_space_integers(self) -> None:
        # 0: HOLD, 1: BUY, 2: SELL
        counts = {0: 30, 1: 35, 2: 35}
        imb = compute_direction_imbalance(counts)
        assert imb == 0.0


class TestMetricsObjectCompatibility:
    """Verify compatibility when metrics are passed as custom objects or dataclasses."""

    def test_mt5_backtest_metrics_object_compatibility(self) -> None:
        from dataclasses import dataclass
        from pathlib import Path

        @dataclass
        class FakeMT5Metrics:
            total_trades: int = 42
            win_rate: float = 46.5
            avg_win: float = 2.2
            avg_loss: float = -1.0
            equity_drawdown_pct: float = 7.2
            net_profit: float = 1250.0
            report_path: Path = Path("fake_report.htm")

        fake = FakeMT5Metrics()
        validator = OOSValidator()
        policy = MockPolicy()

        result = validator.evaluate(
            policy,
            metrics=fake,
            action_counts={"HOLD": 25, "BUY": 40, "SELL": 35},
        )
        assert result["total_trades"] == 42
        assert result["win_rate"] >= 0.40
        assert result["risk_reward_ratio"] == 2.2
        assert result["max_equity_drawdown_pct"] == 7.2
        assert result["passed_gate"] is True

