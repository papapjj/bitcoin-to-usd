"""
tests/test_dual_champion_router.py

Comprehensive Unit & Interface Tests for DualChampionRouter Engine:
- Legacy Tests: test_dual_champion_router_regime_switching, test_dual_champion_router_context_synchronization
- TestGroup 1: Duck-Typing Bridge Compliance (input_size, output_size, ensure_compatible,
               reset_runtime_state, experience_buffer, _expert_from_trace, publish_policy)
- TestGroup 2: Hysteresis Thresholds & Confirmation Counter Buffer
- TestGroup 3: Recurrent State Freezing & Specialist Memory Isolation (4096-D LSTM states)
- TestGroup 4: Synchronous Account, Position, Decision & Telemetry Context Updates
- TestGroup 5: Checkpoint Loading & Fallback Logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    ConsecutiveLossCircuitBreaker,
    load_dual_champion_router,
)


# ==============================================================================
# HELPERS & MOCKS
# ==============================================================================

class PicklablePolicyMock:
    """Picklable mock policy for checkpoint loading tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.input_size = 137
        self.output_size = 20
        self.experts: dict[str, Any] = {"dummy_expert": 1}
        self.nhan_dien = None
        self.registry = None
        self.cau_hinh = None

    def ensure_compatible(self) -> None:
        pass

    def reset_runtime_state(self) -> None:
        pass


class StatefulRecurrentPolicyMock:
    """Mock policy that tracks simulated 4096-D recurrent LSTM memory states."""

    def __init__(self, name: str, input_size: int = 137, output_size: int = 20) -> None:
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_state = np.zeros((5, 4096), dtype=np.float32)
        self.cell_state = np.zeros((5, 4096), dtype=np.float32)
        self.step_count = 0
        self.experience_buffer: list[dict[str, Any]] = []
        self.account_context: dict[str, float] = {}
        self.decision_context: dict[str, Any] = {}
        self.position_context: Any = None
        self.raw_inputs: np.ndarray | None = None
        self.soft_probabilities: tuple[float, ...] | None = None
        self.experts = {"mock_expert": MagicMock()}
        self.nhan_dien = MagicMock()
        self.registry = MagicMock()
        self.cau_hinh = MagicMock()

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[object, ...]:
        self.step_count += 1
        # Advance recurrent state when stepped
        self.hidden_state += 1.0
        self.cell_state += 2.0
        return (f"{self.name}_trace_{self.step_count}",)

    def ensure_compatible(self) -> None:
        pass

    def reset_runtime_state(self) -> None:
        self.step_count = 0
        self.hidden_state.fill(0.0)
        self.cell_state.fill(0.0)

    def set_account_context(self, account_context: dict[str, float], positions_pipe: str = "") -> None:
        self.account_context = dict(account_context)
        self.account_context["positions_pipe"] = positions_pipe

    def set_decision_context(self, **kwargs: Any) -> None:
        self.decision_context = dict(kwargs)

    def set_position_context(self, *args: Any, **kwargs: Any) -> None:
        self.position_context = (args, kwargs)

    def set_regime_raw_inputs(self, raw_inputs: Any) -> None:
        self.raw_inputs = np.asarray(raw_inputs)

    def set_regime_soft_probabilities(
        self,
        probs: Any,
        regime_name: str | None = None,
        regime_idx: int | None = None,
    ) -> None:
        self.soft_probabilities = tuple(probs)


# ==============================================================================
# LEGACY TESTS
# ==============================================================================

def test_dual_champion_router_regime_switching() -> None:
    bull_mock = MagicMock()
    bear_mock = MagicMock()
    bull_mock.act_nam_tac_nhan.return_value = ("bull_trace_0", "bull_trace_1")
    bear_mock.act_nam_tac_nhan.return_value = ("bear_trace_0", "bear_trace_1")

    router = DualChampionRouter(
        bull_policy=bull_mock,
        bear_policy=bear_mock,
        bull_threshold=0.55,
        bear_threshold=0.55,
        chop_threshold=0.40,
        confirmation_bars=1,
    )

    # Initial state should be CHOP / Bear fallback
    assert router.current_regime == "CHOP"
    assert router.active_policy is bear_mock

    # 1. Soft Bull probabilities: [0.70, 0.15, 0.15]
    router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
    assert router.current_regime == "BULL"
    assert router.active_policy is bull_mock

    dummy_obs = np.zeros(137, dtype=float)
    res = router.act_nam_tac_nhan(dummy_obs)
    assert res == ("bull_trace_0", "bull_trace_1")
    bull_mock.act_nam_tac_nhan.assert_called_once()

    # 2. Soft Bear probabilities: [0.10, 0.75, 0.15]
    router.set_regime_soft_probabilities([0.10, 0.75, 0.15], "BEAR", 1)
    assert router.current_regime == "BEAR"
    assert router.active_policy is bear_mock

    res_bear = router.act_nam_tac_nhan(dummy_obs)
    assert res_bear == ("bear_trace_0", "bear_trace_1")
    bear_mock.act_nam_tac_nhan.assert_called_once()

    # 3. Soft Chop probabilities: [0.25, 0.25, 0.50]
    router.set_regime_soft_probabilities([0.25, 0.25, 0.50], "CHOP", 2)
    assert router.current_regime == "CHOP"
    assert router.active_policy is bear_mock


def test_dual_champion_router_context_synchronization() -> None:
    bull_mock = MagicMock()
    bear_mock = MagicMock()

    router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

    acc_context = {"balance": 10000.0, "equity": 10050.0}
    router.set_account_context(acc_context, "positions_test")

    bull_mock.set_account_context.assert_called_once_with(acc_context, "positions_test")
    bear_mock.set_account_context.assert_called_once_with(acc_context, "positions_test")

    router.set_decision_context(frame_id="test_frame")
    bull_mock.set_decision_context.assert_called_once_with(frame_id="test_frame")
    bear_mock.set_decision_context.assert_called_once_with(frame_id="test_frame")


# ==============================================================================
# TEST GROUP 1: DUCK-TYPING BRIDGE CONTRACT COMPLIANCE
# ==============================================================================

class TestDuckTypingBridgeContract:
    """Verifies all duck-typing attributes required by MT5StrategyTesterEvaluator."""

    def test_duck_typing_dimension_properties(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        bull_mock.input_size = 137
        bull_mock.output_size = 20
        bear_mock.input_size = 137
        bear_mock.output_size = 20

        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        assert router.input_size == 137
        assert isinstance(router.input_size, int)
        assert router.output_size == 20
        assert isinstance(router.output_size, int)

    def test_ensure_compatible_zero_args_callable_returns_true(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        # Must be callable with 0 arguments as required by publish_policy and PROJECT.md
        res = router.ensure_compatible()
        assert res is True
        bull_mock.ensure_compatible.assert_called_once()
        bear_mock.ensure_compatible.assert_called_once()

    def test_ensure_compatible_optional_args_backward_compatibility(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        # Must not raise when called with target_input_size or extra kwargs
        res = router.ensure_compatible(137, 20)
        assert res is True

    def test_ensure_compatible_mismatched_dimensions_raises(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        with pytest.raises(ValueError, match="DualChampionRouter requires 137x20"):
            router.ensure_compatible(100, 20)

        with pytest.raises(ValueError, match="DualChampionRouter requires 137x20"):
            router.ensure_compatible(137, 10)

    def test_reset_runtime_state_clears_counters_and_resets_policies(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            confirmation_bars=2,
        )

        # Simulate ongoing run state
        router.current_regime = "BULL"
        router.current_regime_index = 0
        router.current_probabilities = (0.75, 0.15, 0.10)
        router._pending_bull_count = 3
        router._pending_bear_count = 1

        router.reset_runtime_state()

        assert router.current_regime == "CHOP"
        assert router.current_regime_index == 2
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0
        assert router.current_probabilities == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        bull_mock.reset_runtime_state.assert_called_once()
        bear_mock.reset_runtime_state.assert_called_once()

    def test_experience_buffer_getter_and_setter(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        bull_mock.experience_buffer = [{"item": 1}]
        bear_mock.experience_buffer = [{"item": 2}]

        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        # Getter should return a list or duck-typed container
        buf = router.experience_buffer
        assert isinstance(buf, list)

        # Setter should clear/update sub-policies without crashing (as in mt5_runner.py:10488)
        router.experience_buffer = []
        assert bull_mock.experience_buffer == []
        assert bear_mock.experience_buffer == []

    def test_reset_five_economic_reward_propagation(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        hook = getattr(router, "reset_five_economic_reward", None)
        assert callable(hook), "reset_five_economic_reward must be exposed for bridge compatibility"
        router.reset_five_economic_reward()
        bull_mock.reset_five_economic_reward.assert_called_once()
        bear_mock.reset_five_economic_reward.assert_called_once()

    def test_expert_from_trace_delegation(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        dummy_trace = MagicMock()
        bull_mock._expert_from_trace.return_value = "bull_expert"
        bear_mock._expert_from_trace.return_value = "bear_expert"

        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)
        router.current_regime = "BULL"
        assert router._expert_from_trace(dummy_trace) == "bull_expert"

        router.current_regime = "BEAR"
        assert router._expert_from_trace(dummy_trace) == "bear_expert"

    def test_expert_from_trace_cross_specialist_fallback(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        dummy_trace = MagicMock()
        # Active policy (bull) doesn't own this trace, alternate (bear) does
        bull_mock._expert_from_trace.return_value = None
        bear_mock._expert_from_trace.return_value = "bear_expert"

        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)
        router.current_regime = "BULL"
        assert router.active_policy is bull_mock
        assert router._expert_from_trace(dummy_trace) == "bear_expert"

    def test_act_delegates_to_active_or_falls_back_to_act_nam_tac_nhan(self) -> None:
        # Case A: active policy exposes act()
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        bull_mock.act.return_value = "act_result"
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)
        router.current_regime = "BULL"
        obs = np.zeros(137, dtype=float)
        assert router.act(obs) == "act_result"
        bull_mock.act.assert_called_once()

        # Case B: active policy does NOT have act(), falls back to act_nam_tac_nhan()
        bull_mock_no_act = MagicMock(spec=["act_nam_tac_nhan", "ensure_compatible", "reset_runtime_state"])
        bull_mock_no_act.act_nam_tac_nhan.return_value = ("five_trace",)
        router2 = DualChampionRouter(bull_policy=bull_mock_no_act, bear_policy=bear_mock)
        router2.current_regime = "BULL"
        assert router2.act(obs) == ("five_trace",)
        bull_mock_no_act.act_nam_tac_nhan.assert_called_once()

    def test_observe_five_economic_transition_delegation_and_fallback(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        bull_mock.observe_five_economic_transition.return_value = {"reward": 1.5}
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)
        router.current_regime = "BULL"
        assert router.observe_five_economic_transition("batch") == {"reward": 1.5}

        # Fallback when active policy lacks the method
        bull_no_hook = MagicMock(spec=["act_nam_tac_nhan"])
        bear_mock.observe_five_economic_transition.return_value = {"bear_reward": 2.0}
        router_fallback = DualChampionRouter(bull_policy=bull_no_hook, bear_policy=bear_mock)
        router_fallback.current_regime = "BULL"
        assert router_fallback.observe_five_economic_transition("batch") == {"bear_reward": 2.0}

    def test_observe_transition_delegation(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        bull_mock.observe_transition.return_value = {"trans": 1}
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)
        router.current_regime = "BULL"
        assert router.observe_transition("trace", np.zeros(137), 1.0, 0.0) == {"trans": 1}
        bull_mock.observe_transition.assert_called_once()

    def test_set_experience_starvation_propagation(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)
        router.set_experience_starvation(True)
        bull_mock.set_experience_starvation.assert_called_once_with(True)
        bear_mock.set_experience_starvation.assert_called_once_with(True)

    def test_evaluator_publish_policy_compliance(self) -> None:
        from self_growing_trader.mt5_runner import MT5StrategyTesterEvaluator, MT5BacktestConfig

        bull = MagicMock(input_size=137, output_size=20, experts={"e0": MagicMock()}, nhan_dien=MagicMock(), registry=MagicMock(), experience_buffer=[])
        bear = MagicMock(input_size=137, output_size=20, experts={"e0": MagicMock()}, nhan_dien=MagicMock(), registry=MagicMock(), experience_buffer=[])
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)
        evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(offline_replay=True))
        res = evaluator.publish_policy(router)
        assert res["bridge_mode"] == "python_action"
        assert res["policy_runtime"] == "mot_context_mot_chuyen_gia"
        assert res["so_chuyen_gia"] == 1


# ==============================================================================
# TEST GROUP 2: HYSTERESIS & CONFIRMATION BUFFER
# ==============================================================================

class TestHysteresisAndConfirmationBuffer:
    """Verifies confirmation delay, margin separation, contest zone, and noise rejection."""

    def test_initial_state_defaults_to_chop_and_bear(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock
        assert router.current_probabilities == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    def test_confirmation_bars_delay_switching(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            bull_threshold=0.55,
            bear_threshold=0.55,
            chop_threshold=0.40,
            confirmation_bars=3,
        )

        # Bar 1: Bull candidate arrives
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock
        assert router._pending_bull_count == 1

        # Bar 2: Bull candidate continues
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock
        assert router._pending_bull_count == 2

        # Bar 3: Reaches confirmation_bars=3 -> switches to BULL
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock
        assert router._pending_bull_count == 3

    def test_whipsaw_noise_spike_rejected_and_counter_cleared(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            confirmation_bars=2,
        )

        # Bar 1: Bull spike
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router._pending_bull_count == 1
        assert router.current_regime == "CHOP"

        # Bar 2: Sudden Bear spike (glitch/noise)
        router.set_regime_soft_probabilities([0.10, 0.80, 0.10], "BEAR", 1)
        # Previous bull counter must be cleared immediately
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 1
        assert router.current_regime == "CHOP"  # Still CHOP, noise rejected!

    def test_margin_separation_threshold_boundary(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            bull_threshold=0.55,
            bear_threshold=0.55,
            confirmation_bars=1,
        )

        # P_Bull = 0.55, but margin (P_Bull - P_Bear) = 0.55 - 0.45 = 0.10 (not > 0.10)
        router.set_regime_soft_probabilities([0.55, 0.45, 0.00])
        assert router.current_regime == "CHOP", "Must not switch when margin <= 0.10"

        # Margin exceeds 0.10: 0.56 - 0.44 = 0.12 > 0.10
        router.set_regime_soft_probabilities([0.56, 0.44, 0.00])
        assert router.current_regime == "BULL"

    def test_contest_zone_state_retention(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock, confirmation_bars=1)

        # Manually put into BULL
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router.current_regime == "BULL"

        # Contest zone: P_Bull=0.48, P_Bear=0.48, P_Chop=0.04
        # Neither reaches 0.55, P_Chop < 0.40 -> Hold previous state (S_t = S_{t-1})
        router.set_regime_soft_probabilities([0.48, 0.48, 0.04])
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock

    def test_contest_zone_resets_pending_counters(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock, confirmation_bars=2)

        # Bar 1: Bull signal
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15])
        assert router._pending_bull_count == 1
        assert router.current_regime == "CHOP"

        # Bar 2: Contest bar [0.48, 0.48, 0.04] resets pending counters
        router.set_regime_soft_probabilities([0.48, 0.48, 0.04])
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0
        assert router.current_regime == "CHOP"

        # Bar 3: Another single bull bar -> count is 1, not 2!
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15])
        assert router._pending_bull_count == 1
        assert router.current_regime == "CHOP"

    def test_chop_threshold_switches_to_defensive_bear(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock, confirmation_bars=1)

        # First switch to BULL
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router.current_regime == "BULL"

        # P_Chop >= 0.40 triggers defensive CHOP mode
        router.set_regime_soft_probabilities([0.30, 0.25, 0.45], "CHOP", 2)
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0


# ==============================================================================
# TEST GROUP 3: RECURRENT STATE FREEZING & MEMORY ISOLATION
# ==============================================================================

class TestRecurrentStateFreezing:
    """Verifies that inactive specialist recurrent memory remains strictly frozen bit-for-bit."""

    def test_inactive_specialist_recurrent_memory_strictly_frozen(self) -> None:
        bull_policy = StatefulRecurrentPolicyMock("bull")
        bear_policy = StatefulRecurrentPolicyMock("bear")

        router = DualChampionRouter(
            bull_policy=bull_policy,
            bear_policy=bear_policy,
            confirmation_bars=1,
        )

        dummy_obs = np.zeros(137, dtype=np.float32)

        # Phase 1: Switch to BULL Specialist
        router.set_regime_soft_probabilities([0.75, 0.15, 0.10], "BULL", 0)
        assert router.active_policy is bull_policy

        # Snapshot initial Bear states
        bear_h0 = bear_policy.hidden_state.copy()
        bear_c0 = bear_policy.cell_state.copy()
        bear_steps_0 = bear_policy.step_count

        # Step 10 times in Bull regime
        for _ in range(10):
            router.act_nam_tac_nhan(dummy_obs)

        # Bull states advanced
        assert bull_policy.step_count == 10
        assert not np.array_equal(bull_policy.hidden_state, np.zeros((5, 4096)))

        # Bear states MUST REMAIN STRICTLY FROZEN
        assert bear_policy.step_count == bear_steps_0 == 0
        assert np.array_equal(bear_policy.hidden_state, bear_h0)
        assert np.array_equal(bear_policy.cell_state, bear_c0)

        # Phase 2: Switch to BEAR Specialist
        router.set_regime_soft_probabilities([0.10, 0.75, 0.15], "BEAR", 1)
        assert router.active_policy is bear_policy

        # Snapshot Bull states
        bull_h10 = bull_policy.hidden_state.copy()
        bull_c10 = bull_policy.cell_state.copy()
        bull_steps_10 = bull_policy.step_count

        # Step 5 times in Bear regime
        for _ in range(5):
            router.act_nam_tac_nhan(dummy_obs)

        # Bear states advanced
        assert bear_policy.step_count == 5
        assert not np.array_equal(bear_policy.hidden_state, bear_h0)

        # Bull states MUST REMAIN STRICTLY FROZEN
        assert bull_policy.step_count == bull_steps_10 == 10
        assert np.array_equal(bull_policy.hidden_state, bull_h10)
        assert np.array_equal(bull_policy.cell_state, bull_c10)

    def test_bidirectional_recurrent_isolation(self) -> None:
        bull_policy = StatefulRecurrentPolicyMock("bull")
        bear_policy = StatefulRecurrentPolicyMock("bear")

        router = DualChampionRouter(
            bull_policy=bull_policy,
            bear_policy=bear_policy,
            confirmation_bars=1,
        )

        dummy_obs = np.zeros(137, dtype=np.float32)

        # Starts at CHOP -> routes to bear
        router.act_nam_tac_nhan(dummy_obs)
        assert bear_policy.step_count == 1
        assert bull_policy.step_count == 0

        # Bull activated
        router.set_regime_soft_probabilities([0.70, 0.10, 0.20], "BULL", 0)
        router.act_nam_tac_nhan(dummy_obs)
        assert bull_policy.step_count == 1
        assert bear_policy.step_count == 1  # bear frozen

        # Bear activated
        router.set_regime_soft_probabilities([0.10, 0.70, 0.20], "BEAR", 1)
        router.act_nam_tac_nhan(dummy_obs)
        assert bear_policy.step_count == 2
        assert bull_policy.step_count == 1  # bull frozen


# ==============================================================================
# TEST GROUP 4: SYNCHRONOUS CONTEXT UPDATES
# ==============================================================================

class TestContextSynchronization:
    """Verifies that non-recurrent account and decision contexts synchronize across both policies."""

    def test_account_context_and_decision_context_synchronized(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        acc = {"balance": 10000.0, "equity": 10150.0, "drawdown_pct": 2.1}
        router.set_account_context(acc, positions_pipe="BTC_BUY_1")
        bull_mock.set_account_context.assert_called_once_with(acc, "BTC_BUY_1")
        bear_mock.set_account_context.assert_called_once_with(acc, "BTC_BUY_1")

        router.set_decision_context(frame_id="FR_999", timestamp="2025.02.24 12:00")
        bull_mock.set_decision_context.assert_called_once_with(frame_id="FR_999", timestamp="2025.02.24 12:00")
        bear_mock.set_decision_context.assert_called_once_with(frame_id="FR_999", timestamp="2025.02.24 12:00")

    def test_position_context_synchronized(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        router.set_position_context("pos_data_1", mode="active")
        bull_mock.set_position_context.assert_called_once_with("pos_data_1", mode="active")
        bear_mock.set_position_context.assert_called_once_with("pos_data_1", mode="active")

    def test_set_regime_raw_inputs_propagation(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        raw = np.linspace(-1.0, 1.0, 137)
        router.set_regime_raw_inputs(raw)
        bull_mock.set_regime_raw_inputs.assert_called_once_with(raw)
        bear_mock.set_regime_raw_inputs.assert_called_once_with(raw)

    def test_soft_probabilities_propagated_to_both_specialists(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        probs = [0.80, 0.10, 0.10]
        router.set_regime_soft_probabilities(probs, "BULL", 0)

        bull_mock.set_regime_soft_probabilities.assert_called_once_with(
            probs, regime_name="BULL", regime_idx=0
        )
        bear_mock.set_regime_soft_probabilities.assert_called_once_with(
            probs, regime_name="BULL", regime_idx=0
        )

    def test_extrema_and_live_outcome_records_synchronized(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        router.record_position_context_extrema(order_ticket=12345, mae=-0.5, mfe=1.2)
        bull_mock.record_position_context_extrema.assert_called_once_with(order_ticket=12345, mae=-0.5, mfe=1.2)
        bear_mock.record_position_context_extrema.assert_called_once_with(order_ticket=12345, mae=-0.5, mfe=1.2)

        router.record_five_live_owner_outcome("ticket_123", outcome="WIN", profit=150.0)
        bull_mock.record_five_live_owner_outcome.assert_called_once_with("ticket_123", outcome="WIN", profit=150.0)
        bear_mock.record_five_live_owner_outcome.assert_called_once_with("ticket_123", outcome="WIN", profit=150.0)

    def test_context_sync_fault_tolerance(self) -> None:
        bull_mock = MagicMock()
        bear_mock = MagicMock()
        bull_mock.set_account_context.side_effect = RuntimeError("Transient network issue")

        router = DualChampionRouter(bull_policy=bull_mock, bear_policy=bear_mock)

        # Should not raise; bear_mock must still be called
        router.set_account_context({"balance": 10000.0})
        bear_mock.set_account_context.assert_called_once_with({"balance": 10000.0}, "")


# ==============================================================================
# TEST GROUP 5: CHECKPOINT LOADING & FALLBACK LOGIC
# ==============================================================================

class TestCheckpointLoadingAndFallback:
    """Verifies factory loader, missing file handling, and fallback behavior."""

    def test_load_missing_bear_champion_raises_file_not_found(self, tmp_path: Path) -> None:
        missing_bear = tmp_path / "missing_bear.pkl"
        missing_bull = tmp_path / "missing_bull.pkl"

        with pytest.raises(FileNotFoundError, match="Bear Champion not found"):
            load_dual_champion_router(bear_path=missing_bear, bull_path=missing_bull)

    def test_load_missing_bull_champion_raises_file_not_found(self, tmp_path: Path) -> None:
        bear_file = tmp_path / "bear.pkl"
        with open(bear_file, "wb") as f:
            pickle.dump({"dummy": "bear"}, f)

        missing_bull = tmp_path / "missing_bull.pkl"
        with pytest.raises(FileNotFoundError, match="Bull Champion not found"):
            load_dual_champion_router(bear_path=bear_file, bull_path=missing_bull)

    def test_load_missing_regime_detector_graceful_fallback(self, tmp_path: Path) -> None:
        bear_file = tmp_path / "bear.pkl"
        bull_file = tmp_path / "bull.pkl"
        with open(bear_file, "wb") as f:
            pickle.dump(PicklablePolicyMock("bear"), f)
        with open(bull_file, "wb") as f:
            pickle.dump(PicklablePolicyMock("bull"), f)

        missing_detector = tmp_path / "missing_detector.pkl"

        # Should NOT raise; fallback detector = None
        router = load_dual_champion_router(
            bear_path=bear_file,
            bull_path=bull_file,
            detector_path=missing_detector,
        )

        assert isinstance(router, DualChampionRouter)
        assert router.regime_detector is None

    def test_load_happy_path_with_detector(self, tmp_path: Path) -> None:
        bear_file = tmp_path / "bear.pkl"
        bull_file = tmp_path / "bull.pkl"
        detector_file = tmp_path / "detector.pkl"

        with open(bear_file, "wb") as f:
            pickle.dump(PicklablePolicyMock("bear"), f)
        with open(bull_file, "wb") as f:
            pickle.dump(PicklablePolicyMock("bull"), f)
        with open(detector_file, "wb") as f:
            pickle.dump({"model": "lgbm_mock"}, f)

        router = load_dual_champion_router(
            bear_path=bear_file,
            bull_path=bull_file,
            detector_path=detector_file,
            bull_threshold=0.60,
            bear_threshold=0.60,
            chop_threshold=0.45,
            confirmation_bars=2,
        )

        assert isinstance(router, DualChampionRouter)
        assert router.regime_detector == {"model": "lgbm_mock"}
        assert router.bull_threshold == 0.60
        assert router.bear_threshold == 0.60
        assert router.chop_threshold == 0.45
        assert router.confirmation_bars == 2


# ==============================================================================
# TEST GROUP 6: TIER 1 HIGHER-TIMEFRAME (HTF) MACRO TREND FILTER
# ==============================================================================

class TestTier1HTFMacroTrendFilter:
    """Verifies Candidate E / Tier 1 HTF Macro Trend Invariance Filter."""

    def test_tier1_htf_macro_bull_routes_to_bull(self) -> None:
        bull_mock = StatefulRecurrentPolicyMock("bull")
        bear_mock = StatefulRecurrentPolicyMock("bear")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            htf_filter_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[88] = 0.12  # h1_ema_diff_atr > 0.05
        obs[89] = 0.04  # h1_ema_slope_atr >= 0.0

        res = router.evaluate_htf_macro_filter(obs)
        assert res is not None
        regime, probs = res
        assert regime == "BULL"
        assert probs == (0.80, 0.10, 0.10)
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock

    def test_tier1_htf_macro_bear_routes_to_bear(self) -> None:
        bull_mock = StatefulRecurrentPolicyMock("bull")
        bear_mock = StatefulRecurrentPolicyMock("bear")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            htf_filter_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[88] = -0.15  # h1_ema_diff_atr < -0.05
        obs[89] = -0.02  # h1_ema_slope_atr <= 0.0

        res = router.evaluate_htf_macro_filter(obs)
        assert res is not None
        regime, probs = res
        assert regime == "BEAR"
        assert probs == (0.10, 0.80, 0.10)
        assert router.current_regime == "BEAR"
        assert router.active_policy is bear_mock

    def test_tier1_htf_macro_contest_maintains_previous_regime(self) -> None:
        bull_mock = StatefulRecurrentPolicyMock("bull")
        bear_mock = StatefulRecurrentPolicyMock("bear")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            htf_filter_enabled=True,
        )
        router.current_regime = "BULL"

        # In-between values: diff=0.02 is between -0.05 and +0.05
        obs = np.zeros(137, dtype=float)
        obs[88] = 0.02
        obs[89] = 0.01

        res = router.evaluate_htf_macro_filter(obs)
        assert res is not None
        regime, probs = res
        assert regime == "BULL"
        assert router.current_regime == "BULL"

    def test_tier1_htf_macro_zero_features_fallback_to_detector(self) -> None:
        bull_mock = StatefulRecurrentPolicyMock("bull")
        bear_mock = StatefulRecurrentPolicyMock("bear")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            htf_filter_enabled=True,
        )

        # Zero H1 features (synthetic unit test vector)
        obs = np.zeros(137, dtype=float)
        res = router.evaluate_htf_macro_filter(obs)
        assert res is None

    def test_tier1_htf_macro_set_regime_raw_inputs_updates_regime(self) -> None:
        bull_mock = StatefulRecurrentPolicyMock("bull")
        bear_mock = StatefulRecurrentPolicyMock("bear")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            htf_filter_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[88] = 0.20
        obs[89] = 0.05
        router.set_regime_raw_inputs(obs)

        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock


# ==============================================================================
# TEST GROUP 7: TIER 2 MICROSTRUCTURE MOMENTUM GATE
# ==============================================================================

@dataclass
class CustomTrace:
    action_index: int = 0
    selected_idx: int = 0
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))
    action_details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.90


class FiveAgentControlledMockPolicy:
    """Mock policy allowing explicit action assignment per agent."""

    def __init__(self, agent_actions: dict[int, int]) -> None:
        self.agent_actions = dict(agent_actions)
        self.input_size = 137
        self.output_size = 20
        self.hidden_state = np.zeros((5, 4096), dtype=np.float32)
        self.cell_state = np.zeros((5, 4096), dtype=np.float32)

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[CustomTrace, ...]:
        traces = []
        for agent_idx in range(5):
            act = self.agent_actions.get(agent_idx, 0)
            outputs = np.zeros(20, dtype=float)
            outputs[act] = 1.0
            tr = CustomTrace(
                action_index=act,
                selected_idx=act,
                selected_action=outputs,
                action_details={
                    "five_agent_index": agent_idx,
                    "five_action_index": act,
                    "wire_action": act,
                },
            )
            traces.append(tr)
        return tuple(traces)

    def reset_runtime_state(self) -> None:
        self.hidden_state.fill(0.0)
        self.cell_state.fill(0.0)

    def ensure_compatible(self) -> None:
        pass


class TestTier2MicrostructureMomentumGate:
    """Verifies Candidate C / Tier 2 Microstructure Momentum Gate."""

    def test_tier2_momentum_gate_vetoes_sell_limit_in_bull_momentum(self) -> None:
        # Agent 1 proposes SELL_LIMIT (Action 4)
        policy = FiveAgentControlledMockPolicy({0: 0, 1: 4, 2: 0, 3: 0, 4: 0})
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            momentum_gate_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[80] = 0.20  # m5_ema_diff_atr > 0
        obs[81] = 0.05  # m5_ema_slope_atr > 0

        batch = router.act_nam_tac_nhan(obs)
        assert len(batch) == 5
        agent1_trace = batch[1]

        assert agent1_trace.action_details.get("five_action_index") == 0
        assert agent1_trace.action_details.get("regime_veto_applied") is True
        assert agent1_trace.action_details.get("regime_veto_reason") == "veto_sell_against_upward_momentum"

    def test_tier2_momentum_gate_vetoes_market_sell_in_bull_momentum(self) -> None:
        # Agent 1 proposes MARKET_SELL (Action 2)
        policy = FiveAgentControlledMockPolicy({0: 0, 1: 2, 2: 0, 3: 0, 4: 0})
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            momentum_gate_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[80] = 0.15
        obs[81] = 0.08

        batch = router.act_nam_tac_nhan(obs)
        agent1_trace = batch[1]

        assert agent1_trace.action_details.get("five_action_index") == 0
        assert agent1_trace.action_details.get("regime_veto_applied") is True
        assert agent1_trace.action_details.get("regime_veto_reason") == "veto_sell_against_upward_momentum"

    def test_tier2_momentum_gate_vetoes_buy_limit_in_bear_momentum(self) -> None:
        # Agent 0 proposes BUY_LIMIT (Action 3)
        policy = FiveAgentControlledMockPolicy({0: 3, 1: 0, 2: 0, 3: 0, 4: 0})
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            momentum_gate_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[80] = -0.25  # m5_ema_diff_atr < 0
        obs[81] = -0.06  # m5_ema_slope_atr < 0

        batch = router.act_nam_tac_nhan(obs)
        agent0_trace = batch[0]

        assert agent0_trace.action_details.get("five_action_index") == 0
        assert agent0_trace.action_details.get("regime_veto_applied") is True
        assert agent0_trace.action_details.get("regime_veto_reason") == "veto_buy_against_downward_momentum"

    def test_tier2_momentum_gate_allows_trade_when_momentum_aligned(self) -> None:
        # Agent 1 proposes SELL_LIMIT (Action 4) into DOWNWARD momentum (selling with trend)
        policy = FiveAgentControlledMockPolicy({0: 0, 1: 4, 2: 0, 3: 0, 4: 0})
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            momentum_gate_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[80] = -0.10  # Downward M5 momentum
        obs[81] = -0.04

        batch = router.act_nam_tac_nhan(obs)
        agent1_trace = batch[1]

        assert agent1_trace.action_details.get("five_action_index") == 4
        assert agent1_trace.action_details.get("regime_veto_applied") is not True


# ==============================================================================
# TEST GROUP 8: TIER 3 CONSECUTIVE LOSS CIRCUIT BREAKER
# ==============================================================================

class TestTier3ConsecutiveLossCircuitBreaker:
    """Verifies Candidate B / Tier 3 Consecutive Loss Circuit Breaker."""

    def test_tier3_circuit_breaker_trips_after_2_sls_within_60m(self) -> None:
        policy = FiveAgentControlledMockPolicy({0: 0, 1: 4, 2: 0, 3: 0, 4: 0})
        cb = ConsecutiveLossCircuitBreaker(
            max_consecutive_sl=2,
            cooldown_minutes=60.0,
            window_minutes=60.0,
            enabled=True,
        )
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            circuit_breaker=cb,
        )

        t0 = datetime(2025, 2, 20, 10, 0, 0)
        router.record_trade_outcome(direction="SELL", outcome="SL", profit=-108.5, timestamp=t0)
        assert router.circuit_breaker_active is False
        assert router.is_in_cooldown("SELL", t0) is False

        t1 = t0 + timedelta(minutes=15)
        router.record_trade_outcome(direction="SELL", outcome="SL", profit=-108.5, timestamp=t1)

        assert router.circuit_breaker_active is True
        assert router.is_in_cooldown("SELL", t1 + timedelta(minutes=5)) is True
        assert router.is_in_cooldown("BUY", t1 + timedelta(minutes=5)) is False

        obs = np.zeros(137, dtype=float)
        batch = router.act_nam_tac_nhan(obs, time=t1 + timedelta(minutes=20))
        agent1_trace = batch[1]
        assert agent1_trace.action_details.get("five_action_index") == 0
        assert agent1_trace.action_details.get("regime_veto_applied") is True
        assert agent1_trace.action_details.get("regime_veto_reason") == "circuit_breaker_consecutive_sl_cooldown"

    def test_tier3_circuit_breaker_symmetric_for_buy(self) -> None:
        policy = FiveAgentControlledMockPolicy({0: 3, 1: 0, 2: 0, 3: 0, 4: 0})
        cb = ConsecutiveLossCircuitBreaker(
            max_consecutive_sl=2,
            cooldown_minutes=60.0,
            window_minutes=60.0,
            enabled=True,
        )
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            circuit_breaker=cb,
        )

        t0 = datetime(2025, 2, 20, 10, 0, 0)
        router.record_trade_outcome(direction="BUY", outcome="SL", profit=-100.0, timestamp=t0)
        router.record_trade_outcome(direction="BUY", outcome="SL", profit=-100.0, timestamp=t0 + timedelta(minutes=10))

        assert router.circuit_breaker_active is True
        assert router.is_in_cooldown("BUY", t0 + timedelta(minutes=15)) is True
        assert router.is_in_cooldown("SELL", t0 + timedelta(minutes=15)) is False

        obs = np.zeros(137, dtype=float)
        batch = router.act_nam_tac_nhan(obs, time=t0 + timedelta(minutes=15))
        agent0_trace = batch[0]
        assert agent0_trace.action_details.get("five_action_index") == 0
        assert agent0_trace.action_details.get("regime_veto_applied") is True
        assert agent0_trace.action_details.get("regime_veto_reason") == "circuit_breaker_consecutive_sl_cooldown"

    def test_tier3_circuit_breaker_win_resets_loss_streak(self) -> None:
        cb = ConsecutiveLossCircuitBreaker(max_consecutive_sl=2, cooldown_minutes=60.0)
        router = DualChampionRouter(
            bull_policy=StatefulRecurrentPolicyMock("bull"),
            bear_policy=StatefulRecurrentPolicyMock("bear"),
            circuit_breaker=cb,
        )

        t0 = datetime(2025, 2, 20, 10, 0, 0)
        router.record_trade_outcome(direction="SELL", outcome="SL", profit=-100.0, timestamp=t0)
        assert router.circuit_breaker_active is False

        router.record_trade_outcome(direction="SELL", outcome="TP", profit=+217.0, timestamp=t0 + timedelta(minutes=20))

        router.record_trade_outcome(direction="SELL", outcome="SL", profit=-100.0, timestamp=t0 + timedelta(minutes=30))
        assert router.circuit_breaker_active is False
        assert router.is_in_cooldown("SELL", t0 + timedelta(minutes=35)) is False

    def test_tier3_circuit_breaker_cooldown_expires(self) -> None:
        cb = ConsecutiveLossCircuitBreaker(max_consecutive_sl=2, cooldown_minutes=60.0)
        router = DualChampionRouter(
            bull_policy=StatefulRecurrentPolicyMock("bull"),
            bear_policy=StatefulRecurrentPolicyMock("bear"),
            circuit_breaker=cb,
        )

        t0 = datetime(2025, 2, 20, 10, 0, 0)
        router.record_trade_outcome(direction="SELL", outcome="SL", timestamp=t0)
        router.record_trade_outcome(direction="SELL", outcome="SL", timestamp=t0 + timedelta(minutes=10))
        assert router.circuit_breaker_active is True

        t_after = t0 + timedelta(minutes=75)
        assert router.is_in_cooldown("SELL", t_after) is False
        assert router.circuit_breaker_active is False

    def test_tier3_reset_circuit_breaker_clears_state(self) -> None:
        cb = ConsecutiveLossCircuitBreaker(max_consecutive_sl=2, cooldown_minutes=60.0)
        router = DualChampionRouter(
            bull_policy=StatefulRecurrentPolicyMock("bull"),
            bear_policy=StatefulRecurrentPolicyMock("bear"),
            circuit_breaker=cb,
        )

        t0 = datetime(2025, 2, 20, 10, 0, 0)
        router.record_trade_outcome(direction="SELL", outcome="SL", timestamp=t0)
        router.record_trade_outcome(direction="SELL", outcome="SL", timestamp=t0 + timedelta(minutes=5))
        assert router.circuit_breaker_active is True

        router.reset_circuit_breaker()
        assert router.circuit_breaker_active is False
        assert router.is_in_cooldown("SELL", t0 + timedelta(minutes=10)) is False

        router.record_trade_outcome(direction="BUY", outcome="SL", timestamp=t0)
        router.record_trade_outcome(direction="BUY", outcome="SL", timestamp=t0 + timedelta(minutes=5))
        assert router.circuit_breaker_active is True
        router.reset_runtime_state()
        assert router.circuit_breaker_active is False


# ==============================================================================
# TEST GROUP 9: MULTI-TIER INVARIANTS & LATENCY SLA
# ==============================================================================

class TestMultiTierInvariantsAndLatency:
    """Verifies strict adherence to system invariants and sub-5ms CPU latency SLA."""

    def test_policy_close_all_strictly_zero_invariance(self) -> None:
        policy = FiveAgentControlledMockPolicy({0: 0, 1: 0, 2: 0, 3: 0, 4: 10})
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        batch = router.act_nam_tac_nhan(obs)
        agent4_trace = batch[4]

        assert agent4_trace.action_details.get("five_action_index") == 0
        assert agent4_trace.action_details.get("wire_action") == 0
        assert int(batch.actions[4]) == 0

    def test_recurrent_memory_isolation_linf_zero_under_all_tiers(self) -> None:
        bull = StatefulRecurrentPolicyMock("bull")
        bear = StatefulRecurrentPolicyMock("bear")
        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
        )

        bear_h0 = bear.hidden_state.copy()
        bear_c0 = bear.cell_state.copy()

        obs_bull = np.zeros(137, dtype=float)
        obs_bull[88] = 0.20
        obs_bull[89] = 0.05

        for _ in range(50):
            router.act_nam_tac_nhan(obs_bull)

        diff_h = np.max(np.abs(bear.hidden_state - bear_h0))
        diff_c = np.max(np.abs(bear.cell_state - bear_c0))
        assert diff_h == 0.0, f"Bear hidden state mutated: L_inf = {diff_h}"
        assert diff_c == 0.0, f"Bear cell state mutated: L_inf = {diff_c}"

    def test_multi_tier_latency_sla_under_5ms(self) -> None:
        policy = FiveAgentControlledMockPolicy({0: 3, 1: 4, 2: 0, 3: 0, 4: 0})
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[88] = 0.10
        obs[89] = 0.02
        obs[80] = 0.15
        obs[81] = 0.05

        latencies = []
        for _ in range(50):
            router.act_nam_tac_nhan(obs)

        for _ in range(1000):
            t_start = time.perf_counter()
            router.act_nam_tac_nhan(obs)
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            latencies.append(t_elapsed_ms)

        mean_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99)
        max_lat = np.max(latencies)

        assert mean_lat < 1.0, f"Mean latency too high: {mean_lat:.3f} ms"
        assert p99_lat < 5.0, f"P99 latency SLA breached: {p99_lat:.3f} ms"
        assert max_lat < 25.0, f"Max latency spike too high: {max_lat:.3f} ms"

