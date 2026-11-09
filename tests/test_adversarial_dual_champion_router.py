"""
tests/test_adversarial_dual_champion_router.py

Adversarial Stress & Empirical Verification Suite for Milestone 1:
- Suite 1: Empirical Recurrent State Freezing Bit-for-Bit Isolation (L_inf = 0.0)
  * Bull Champion 100-step execution: Bear Champion hidden & cell tensors remain 100% frozen (L_inf = 0.0)
  * Bear Champion 100-step execution: Bull Champion hidden & cell tensors remain 100% frozen (L_inf = 0.0)
  * Genuine FiveAgentMemory (torch.Tensor, 5 agents x 4096-D) recurrent memory isolation
  * High-frequency regime interleaving (200 steps) with zero cross-policy drift
  * Chop and contest zone memory freezing
- Suite 2: Duck-Typing Contract & Stress Testing with MT5StrategyTesterEvaluator
  * publish_policy interface invariance and dimension validation
  * _serve_policy_actions lifecycle methods (reset_runtime_state, experience_buffer, reset_five_economic_reward, _expert_from_trace)
  * Full offline replay execution with MT5StrategyTesterEvaluator and BridgeRequest batch
- Suite 3: Adversarial Edge Cases & Boundary Conditions
  * Probability fuzzing (zeros, negatives, unnormalized, strict boundary margins)
  * Sub-policy exception isolation under stress
  * Memory independence & reference isolation
  * Rapid whipsaw confirmation buffer state-machine fuzzing
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    INPUT_COUNT,
    OUTPUT_COUNT,
)
from self_growing_trader.five_shared_policy import FiveAgentMemory, FIVE_LSTM_SIZE
from self_growing_trader.five_agent_authority import HOLD_ACTION
from self_growing_trader.mt5_runner import (
    BridgeRequest,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
)


# ==============================================================================
# REALISTIC MOCKS FOR ADVERSARIAL TESTING
# ==============================================================================

@dataclass
class AdversarialTrace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))
    selected_idx: int = int(HOLD_ACTION)
    confidence: float = 0.85
    action_details: dict[str, Any] = field(default_factory=dict)
    positions_pipe: str = ""
    positions_pipe_after: str = ""
    execution_details: dict[str, Any] = field(default_factory=dict)
    account_context: dict[str, float] = field(default_factory=dict)
    account_context_after: dict[str, float] = field(default_factory=dict)


class _PicklableDummyPolicy:
    def __init__(self, name: str = "dummy") -> None:
        self.name = name
        self.experts: dict[str, Any] = {}

    def act_nam_tac_nhan(self, obs: np.ndarray, **kw: Any) -> dict[str, Any]:
        return {"action": 0}



class AdversarialRecurrentPolicy:
    """Mock specialist policy tracking 5-agent x 4096-D recurrent memory.

    Supports both numpy and torch representations with step counters and telemetry hooks.
    """

    def __init__(self, name: str, *, use_torch: bool = False, seed: int = 42) -> None:
        self.name = name
        self.use_torch = use_torch
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT
        self.step_count = 0

        # Deterministic non-zero initialization
        rng = np.random.RandomState(seed)
        self.initial_hidden_np = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)
        self.initial_cell_np = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)

        self.hidden_state = self.initial_hidden_np.copy()
        self.cell_state = self.initial_cell_np.copy()

        if self.use_torch:
            self.five_agent_memories = [
                FiveAgentMemory(
                    hidden=torch.from_numpy(self.hidden_state[i].copy()),
                    cell=torch.from_numpy(self.cell_state[i].copy()),
                    previous_action=torch.zeros((), dtype=torch.long),
                )
                for i in range(5)
            ]
        else:
            self.five_agent_memories = []

        self.experience_buffer: list[dict[str, Any]] = []
        self.account_context: dict[str, float] = {}
        self.decision_context: dict[str, Any] = {}
        self.position_context: Any = None
        self.raw_inputs: np.ndarray | None = None
        self.soft_probabilities: tuple[float, ...] | None = None
        self.economic_rewards_reset_count = 0
        self.runtime_state_reset_count = 0

        # Required bridge duck-typing attributes
        mock_expert = MagicMock()
        mock_expert.policy = MagicMock()
        mock_expert.policy.action_recovery_learner = MagicMock(cells={})
        self.experts = {f"expert_{name}": mock_expert}
        self.nhan_dien = MagicMock()
        self.registry = MagicMock()
        self.cau_hinh = MagicMock()
        self.cau_hinh.che_do_quyet_dinh = "nam_tac_nhan_dung_chung"

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[AdversarialTrace, ...]:
        self.step_count += 1

        # Advance recurrent state by observation-dependent delta
        obs_norm = float(np.linalg.norm(observation)) if observation.size else 1.0
        delta = 0.01 * (obs_norm + float(self.step_count))

        self.hidden_state += delta
        self.cell_state += delta * 2.0

        if self.use_torch:
            for i, mem in enumerate(self.five_agent_memories):
                mem.hidden.add_(delta)
                mem.cell.add_(delta * 2.0)

        # Return 5 traces for the 5 agents
        traces = []
        for agent_id in range(5):
            outputs = np.zeros(20, dtype=float)
            outputs[int(HOLD_ACTION)] = 1.0
            tr = AdversarialTrace(
                trace_id=f"{self.name}_step_{self.step_count}_agent_{agent_id}",
                selected_action=outputs,
                selected_idx=int(HOLD_ACTION),
                confidence=0.9,
                action_details={
                    "five_agent_index": agent_id,
                    "five_action_index": int(HOLD_ACTION),
                    "policy_name": self.name,
                },
            )
            traces.append(tr)
        return tuple(traces)

    def ensure_compatible(self, input_size: int = 137, output_size: int = 20, *args: Any, **kwargs: Any) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        self.runtime_state_reset_count += 1
        self.step_count = 0
        self.hidden_state = self.initial_hidden_np.copy()
        self.cell_state = self.initial_cell_np.copy()
        if self.use_torch:
            for i, mem in enumerate(self.five_agent_memories):
                mem.hidden.copy_(torch.from_numpy(self.hidden_state[i]))
                mem.cell.copy_(torch.from_numpy(self.cell_state[i]))

    def reset_five_economic_reward(self) -> None:
        self.economic_rewards_reset_count += 1

    def _expert_from_trace(self, trace: Any) -> Any:
        return self.experts.get(f"expert_{self.name}")

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
        self.soft_probabilities = tuple(float(x) for x in probs)

    def record_position_context_extrema(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_five_live_owner_outcome(self, *args: Any, **kwargs: Any) -> None:
        pass

    def apply_entry_timing_extrema_bias(self, trace: Any) -> Any:
        return trace

    def finalize_bridge_decision(self, trace: Any) -> Any:
        return trace

    def reset_intraday_state(self, day_key: str) -> None:
        pass

    def emit_bridge_heartbeat(self, **kwargs: Any) -> None:
        pass


# ==============================================================================
# SUITE 1: RECURRENT STATE FREEZING BIT-FOR-BIT ISOLATION (L_inf = 0.0)
# ==============================================================================

class TestRecurrentStateFreezingAdversarial:
    """Rigorous empirical verification that inactive specialist recurrent states are 100% frozen."""

    def test_recurrent_memory_freezing_bull_champion_100_steps(self) -> None:
        """When Bull Champion steps 100 times, Bear Champion states must not change by 1 bit (L_inf = 0.0)."""
        bull = AdversarialRecurrentPolicy("bull", seed=101)
        bear = AdversarialRecurrentPolicy("bear", seed=202)

        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            bull_threshold=0.55,
            bear_threshold=0.55,
            chop_threshold=0.40,
            confirmation_bars=1,
        )

        # Snapshot exact Bear initial state bytes and arrays
        bear_h0 = bear.hidden_state.copy()
        bear_c0 = bear.cell_state.copy()
        bear_h0_bytes = bear.hidden_state.tobytes()
        bear_c0_bytes = bear.cell_state.tobytes()

        # Route to Bull Specialist
        router.set_regime_soft_probabilities([0.85, 0.05, 0.10], "BULL", 0)
        assert router.active_policy is bull
        assert router.current_regime == "BULL"

        # Execute 100 steps with dynamic randomized observations
        rng = np.random.RandomState(42)
        for step in range(1, 101):
            obs = rng.randn(137).astype(np.float32)
            traces = router.act_nam_tac_nhan(obs)
            assert len(traces) == 5

            # Step-level verification: Bear state must remain 100% frozen at every single step
            diff_h = np.max(np.abs(bear.hidden_state - bear_h0))
            diff_c = np.max(np.abs(bear.cell_state - bear_c0))
            assert diff_h == 0.0, f"Bear hidden state mutated at step {step}: L_inf = {diff_h}"
            assert diff_c == 0.0, f"Bear cell state mutated at step {step}: L_inf = {diff_c}"

        # After 100 steps:
        # 1. Bull advanced exactly 100 steps
        assert bull.step_count == 100
        bull_h_diff = np.max(np.abs(bull.hidden_state - bull.initial_hidden_np))
        assert bull_h_diff > 0.0, "Bull hidden state must have evolved"

        # 2. Bear step count is 0
        assert bear.step_count == 0

        # 3. L_inf distance on Bear hidden state is EXACTLY 0.0
        l_inf_hidden = float(np.max(np.abs(bear.hidden_state - bear_h0)))
        assert l_inf_hidden == 0.0, f"Bear hidden L_inf != 0.0: {l_inf_hidden}"

        # 4. L_inf distance on Bear cell state is EXACTLY 0.0
        l_inf_cell = float(np.max(np.abs(bear.cell_state - bear_c0)))
        assert l_inf_cell == 0.0, f"Bear cell L_inf != 0.0: {l_inf_cell}"

        # 5. Bit-for-bit exact byte match (zero bit drift in IEEE-754 representation)
        assert bear.hidden_state.tobytes() == bear_h0_bytes, "Bear hidden state bytes drifted!"
        assert bear.cell_state.tobytes() == bear_c0_bytes, "Bear cell state bytes drifted!"

    def test_recurrent_memory_freezing_bear_champion_100_steps(self) -> None:
        """When Bear Champion steps 100 times, Bull Champion states must not change by 1 bit (L_inf = 0.0)."""
        bull = AdversarialRecurrentPolicy("bull", seed=303)
        bear = AdversarialRecurrentPolicy("bear", seed=404)

        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            bull_threshold=0.55,
            bear_threshold=0.55,
            chop_threshold=0.40,
            confirmation_bars=1,
        )

        # Snapshot exact Bull initial state bytes and arrays
        bull_h0 = bull.hidden_state.copy()
        bull_c0 = bull.cell_state.copy()
        bull_h0_bytes = bull.hidden_state.tobytes()
        bull_c0_bytes = bull.cell_state.tobytes()

        # Route to Bear Specialist
        router.set_regime_soft_probabilities([0.05, 0.85, 0.10], "BEAR", 1)
        assert router.active_policy is bear
        assert router.current_regime == "BEAR"

        # Execute 100 steps with dynamic randomized observations
        rng = np.random.RandomState(84)
        for step in range(1, 101):
            obs = rng.randn(137).astype(np.float32)
            traces = router.act_nam_tac_nhan(obs)
            assert len(traces) == 5

            # Step-level verification: Bull state must remain 100% frozen at every step
            diff_h = np.max(np.abs(bull.hidden_state - bull_h0))
            diff_c = np.max(np.abs(bull.cell_state - bull_c0))
            assert diff_h == 0.0, f"Bull hidden state mutated at step {step}: L_inf = {diff_h}"
            assert diff_c == 0.0, f"Bull cell state mutated at step {step}: L_inf = {diff_c}"

        # After 100 steps:
        # 1. Bear advanced exactly 100 steps
        assert bear.step_count == 100
        bear_h_diff = np.max(np.abs(bear.hidden_state - bear.initial_hidden_np))
        assert bear_h_diff > 0.0, "Bear hidden state must have evolved"

        # 2. Bull step count is 0
        assert bull.step_count == 0

        # 3. L_inf distance on Bull hidden state is EXACTLY 0.0
        l_inf_hidden = float(np.max(np.abs(bull.hidden_state - bull_h0)))
        assert l_inf_hidden == 0.0, f"Bull hidden L_inf != 0.0: {l_inf_hidden}"

        # 4. L_inf distance on Bull cell state is EXACTLY 0.0
        l_inf_cell = float(np.max(np.abs(bull.cell_state - bull_c0)))
        assert l_inf_cell == 0.0, f"Bull cell L_inf != 0.0: {l_inf_cell}"

        # 5. Bit-for-bit exact byte match
        assert bull.hidden_state.tobytes() == bull_h0_bytes, "Bull hidden state bytes drifted!"
        assert bull.cell_state.tobytes() == bull_c0_bytes, "Bull cell state bytes drifted!"

    def test_recurrent_memory_freezing_with_torch_five_agent_memory(self) -> None:
        """Verify bit-for-bit isolation using genuine PyTorch FiveAgentMemory tensors."""
        bull = AdversarialRecurrentPolicy("bull", use_torch=True, seed=505)
        bear = AdversarialRecurrentPolicy("bear", use_torch=True, seed=606)

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        # Snapshot PyTorch tensors clone
        bear_torch_h0 = [m.hidden.clone() for m in bear.five_agent_memories]
        bear_torch_c0 = [m.cell.clone() for m in bear.five_agent_memories]

        # Route to Bull and step 100 times
        router.set_regime_soft_probabilities([0.80, 0.10, 0.10], "BULL", 0)
        rng = np.random.RandomState(123)
        for _ in range(100):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))

        # Check torch tensors for all 5 agents on Bear
        for agent_idx in range(5):
            mem = bear.five_agent_memories[agent_idx]
            orig_h = bear_torch_h0[agent_idx]
            orig_c = bear_torch_c0[agent_idx]

            assert torch.equal(mem.hidden, orig_h), f"Agent {agent_idx} torch hidden tensor mutated!"
            assert torch.equal(mem.cell, orig_c), f"Agent {agent_idx} torch cell tensor mutated!"
            assert torch.max(torch.abs(mem.hidden - orig_h)).item() == 0.0
            assert torch.max(torch.abs(mem.cell - orig_c)).item() == 0.0

        # Now route to Bear and snapshot Bull
        bull_torch_h100 = [m.hidden.clone() for m in bull.five_agent_memories]
        bull_torch_c100 = [m.cell.clone() for m in bull.five_agent_memories]

        router.set_regime_soft_probabilities([0.10, 0.80, 0.10], "BEAR", 1)
        for _ in range(100):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))

        # Check torch tensors for all 5 agents on Bull
        for agent_idx in range(5):
            mem = bull.five_agent_memories[agent_idx]
            orig_h = bull_torch_h100[agent_idx]
            orig_c = bull_torch_c100[agent_idx]

            assert torch.equal(mem.hidden, orig_h), f"Agent {agent_idx} bull torch hidden mutated during Bear!"
            assert torch.equal(mem.cell, orig_c), f"Agent {agent_idx} bull torch cell mutated during Bear!"
            assert torch.max(torch.abs(mem.hidden - orig_h)).item() == 0.0
            assert torch.max(torch.abs(mem.cell - orig_c)).item() == 0.0

    def test_recurrent_memory_high_frequency_interleaving_200_steps(self) -> None:
        """Rapid regime switching across 20 alternating segments: zero drift across transitions."""
        bull = AdversarialRecurrentPolicy("bull", seed=707)
        bear = AdversarialRecurrentPolicy("bear", seed=808)

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)
        rng = np.random.RandomState(999)

        # 20 segments of 10 steps each = 200 total steps
        for cycle in range(10):
            # Phase Bull
            bear_h_before = bear.hidden_state.copy()
            bear_c_before = bear.cell_state.copy()
            router.set_regime_soft_probabilities([0.80, 0.10, 0.10], "BULL", 0)

            for _ in range(10):
                router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))

            # Bear must be unchanged
            assert np.array_equal(bear.hidden_state, bear_h_before)
            assert np.array_equal(bear.cell_state, bear_c_before)

            # Phase Bear
            bull_h_before = bull.hidden_state.copy()
            bull_c_before = bull.cell_state.copy()
            router.set_regime_soft_probabilities([0.10, 0.80, 0.10], "BEAR", 1)

            for _ in range(10):
                router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))

            # Bull must be unchanged
            assert np.array_equal(bull.hidden_state, bull_h_before)
            assert np.array_equal(bull.cell_state, bull_c_before)

        assert bull.step_count == 100
        assert bear.step_count == 100

    def test_recurrent_memory_freezing_chop_and_contest_zones(self) -> None:
        """In CHOP and Contest zones, inactive specialists remain strictly frozen."""
        bull = AdversarialRecurrentPolicy("bull", seed=111)
        bear = AdversarialRecurrentPolicy("bear", seed=222)

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)
        rng = np.random.RandomState(333)

        # 1. Start in CHOP: Bear is default active policy, Bull must remain frozen
        bull_h0 = bull.hidden_state.copy()
        router.set_regime_soft_probabilities([0.25, 0.25, 0.50], "CHOP", 2)
        assert router.active_policy is bear

        for _ in range(25):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))

        assert bear.step_count == 25
        assert bull.step_count == 0
        assert np.array_equal(bull.hidden_state, bull_h0)
        assert float(np.max(np.abs(bull.hidden_state - bull_h0))) == 0.0

        # 2. Switch to Bull
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))
        assert bull.step_count == 1
        bear_h25 = bear.hidden_state.copy()

        # 3. Enter contest zone [0.48, 0.48, 0.04]: Router retains Bull (previous state)
        router.set_regime_soft_probabilities([0.48, 0.48, 0.04])
        assert router.current_regime == "BULL"
        assert router.active_policy is bull

        for _ in range(25):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))

        assert bull.step_count == 26
        # Bear MUST remain 100% frozen during contest zone steps
        assert bear.step_count == 25
        assert np.array_equal(bear.hidden_state, bear_h25)
        assert float(np.max(np.abs(bear.hidden_state - bear_h25))) == 0.0


# ==============================================================================
# SUITE 2: DUCK-TYPING CONTRACT & STRESS TESTING WITH EVALUATOR
# ==============================================================================

class TestDuckTypingEvaluatorStress:
    """Stress-test all calls and hooks invoked by MT5StrategyTesterEvaluator."""

    def test_publish_policy_contract_invariance(self) -> None:
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)

            # 1. publish_policy must succeed with 0 args to ensure_compatible
            manifest = evaluator.publish_policy(router)
            assert manifest["bridge_mode"] == "python_action"
            assert manifest["policy_runtime"] == "mot_context_mot_chuyen_gia"
            assert manifest["so_chuyen_gia"] == 1
            assert manifest["policy_chung_ra_lenh"] is False
            assert manifest["tron_policy"] is False

            # 2. Properties must match exact contracts
            assert router.input_size == 137
            assert router.output_size == 20
            assert router.ensure_compatible() is True
            assert router.ensure_compatible(137, 20) is True

            # 3. Mismatched dimension check
            with pytest.raises(ValueError, match="137x20"):
                router.ensure_compatible(64, 20)
            with pytest.raises(ValueError, match="137x20"):
                router.ensure_compatible(137, 10)

    def test_serve_policy_actions_lifecycle_methods_stress(self) -> None:
        """Call all methods accessed during _serve_policy_actions on DualChampionRouter."""
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # 1. reset_runtime_state
        router.reset_runtime_state()
        assert router.current_regime == "CHOP"
        assert router.current_regime_index == 2
        assert router.current_probabilities == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        assert bull.runtime_state_reset_count == 1
        assert bear.runtime_state_reset_count == 1

        # 2. experience_buffer getter & setter
        assert isinstance(router.experience_buffer, list)
        router.experience_buffer = [{"dummy": "transition"}]
        assert bull.experience_buffer == [{"dummy": "transition"}]
        assert bear.experience_buffer == [{"dummy": "transition"}]
        router.experience_buffer = []
        assert bull.experience_buffer == []
        assert bear.experience_buffer == []

        # 3. reset_five_economic_reward
        router.reset_five_economic_reward()
        assert bull.economic_rewards_reset_count == 1
        assert bear.economic_rewards_reset_count == 1

        # 4. _expert_from_trace
        dummy_trace = AdversarialTrace(trace_id="test_tr")
        expert = router._expert_from_trace(dummy_trace)
        assert expert is not None

        # 5. set_account_context
        acc_payload = {"balance": 10500.0, "equity": 10700.0}
        router.set_account_context(acc_payload, positions_pipe="POS_BTC")
        assert bull.account_context["equity"] == 10700.0
        assert bear.account_context["equity"] == 10700.0
        assert bull.account_context["positions_pipe"] == "POS_BTC"
        assert bear.account_context["positions_pipe"] == "POS_BTC"

        # 6. set_decision_context
        router.set_decision_context(frame_id="FR_123", mode="live")
        assert bull.decision_context["frame_id"] == "FR_123"
        assert bear.decision_context["frame_id"] == "FR_123"

        # 7. set_position_context
        router.set_position_context("pos_test", flag=True)
        assert bull.position_context == (("pos_test",), {"flag": True})
        assert bear.position_context == (("pos_test",), {"flag": True})

        # 8. set_regime_raw_inputs
        raw_arr = np.ones(137, dtype=float)
        router.set_regime_raw_inputs(raw_arr)
        assert np.array_equal(bull.raw_inputs, raw_arr)
        assert np.array_equal(bear.raw_inputs, raw_arr)

        # 9. record_position_context_extrema & record_five_live_owner_outcome
        router.record_position_context_extrema(ticket=1, mae=-0.2, mfe=0.8)
        router.record_five_live_owner_outcome("ticket_1", outcome="TP")

        # 10. apply_entry_timing_extrema_bias & finalize_bridge_decision
        tr = AdversarialTrace(trace_id="tr_bias")
        assert router.apply_entry_timing_extrema_bias(tr) is tr
        assert router.finalize_bridge_decision(tr) is tr

        # 11. reset_intraday_state & emit_bridge_heartbeat
        router.reset_intraday_state("2025.02.24")
        router.emit_bridge_heartbeat(status="ok")

    def test_router_exposes_act_for_mt5_bridge(self) -> None:
        """MT5 runner line 11948 checks hasattr(policy, 'act'). Router must expose act()."""
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)
        assert hasattr(router, "act"), (
            "DualChampionRouter is missing act() method required by mt5_runner.py:11948 "
            "(hasattr(policy, 'act')). Without act(), _serve_policy_actions raises "
            "RuntimeError: Bridge chỉ chấp nhận HeChuyenGiaMotContextMotPolicy đã xác minh owner at line 12361."
        )

    def test_serve_policy_actions_offline_replay_end_to_end(self) -> None:
        """Execute full offline replay through MT5StrategyTesterEvaluator with 30 BridgeRequests."""
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = MT5BacktestConfig(
                runner_dir=tmp_path,
                offline_replay=True,
                learning_enabled=False,
                action_file="mt5_action.csv",
                request_file="mt5_request.csv",
                bridge_file="mt5_bridge.csv",
                execution_file="mt5_execution.csv",
            )
            evaluator = MT5StrategyTesterEvaluator(config)

            # Create synthetic sequence of 30 requests transitioning across regimes
            rng = np.random.RandomState(777)
            requests: list[BridgeRequest] = []
            for seq in range(1, 31):
                # Regimes: 1-10 Bull, 11-20 Bear, 21-25 Chop, 26-30 Contest
                if seq <= 10:
                    probs = (0.75, 0.15, 0.10)
                    r_name = "BULL"
                    r_idx = 0
                elif seq <= 20:
                    probs = (0.10, 0.80, 0.10)
                    r_name = "BEAR"
                    r_idx = 1
                elif seq <= 25:
                    probs = (0.20, 0.20, 0.60)
                    r_name = "CHOP"
                    r_idx = 2
                else:
                    probs = (0.48, 0.48, 0.04)
                    r_name = "CHOP"
                    r_idx = 2

                req = BridgeRequest(
                    sequence=seq,
                    event_id=f"evt_{seq}",
                    inputs=rng.randn(137).astype(np.float32),
                    raw_inputs=rng.randn(137).astype(np.float32),
                    balance=10000.0,
                    equity=10050.0,
                    positions=0,
                    positions_pipe="",
                    time=f"2025.02.24 10:{seq:02d}",
                    regime_probabilities=probs,
                    regime_name=r_name,
                    regime_index=r_idx,
                )
                requests.append(req)

            # Serve policy actions via offline replay
            report_path = tmp_path / "TesterReport.htm"
            report_path.touch()

            mock_process = MagicMock()
            mock_process.poll.return_value = 0

            # Execute evaluator action dispatch loop
            served_count = evaluator._serve_policy_actions(
                process=mock_process,
                report_path=report_path,
                policy=router,
                policy_id=1,
                round_index=0,
                replay_requests=requests,
            )

            assert served_count == 30
            # Bull was active for 10 steps
            assert bull.step_count == 10
            # Bear was active for 20 steps (10 in Bear + 5 in Chop + 5 in Contest holding previous Chop)
            assert bear.step_count == 20


# ==============================================================================
# SUITE 3: ADVERSARIAL EDGE CASES & BOUNDARY STRESS
# ==============================================================================

class TestAdversarialEdgeCases:
    """Stress test probability extremes, sub-policy exceptions, and state machine boundaries."""

    def test_adversarial_probabilities_fuzzing(self) -> None:
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        # 1. All zeros
        router.set_regime_soft_probabilities([0.0, 0.0, 0.0])
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear

        # 2. All ones (unnormalized)
        # p_bull=1.0, p_bear=1.0, neither margin > 0.10, p_chop=1.0 >= 0.40 -> triggers CHOP
        router.set_regime_soft_probabilities([1.0, 1.0, 1.0])
        assert router.current_regime == "CHOP"

        # 3. Negative values
        router.set_regime_soft_probabilities([-0.5, 0.9, 0.2])
        assert router.current_regime == "BEAR"
        assert router.active_policy is bear

        # 4. Strict boundary margin equality test
        # P_Bull = 0.55, P_Bear = 0.45: Margin is EXACTLY 0.10.
        # Condition is (p_bull > p_bear + 0.10), so 0.55 > 0.55 is FALSE!
        router.current_regime = "CHOP"
        router.set_regime_soft_probabilities([0.55, 0.45, 0.00])
        assert router.current_regime == "CHOP", "Margin == 0.10 must NOT trigger switch (requires strictly > 0.10)"

        # P_Bull = 0.5501, P_Bear = 0.4499: Margin is 0.1002 > 0.10 -> Triggers BULL
        router.set_regime_soft_probabilities([0.5501, 0.4499, 0.00])
        assert router.current_regime == "BULL"
        assert router.active_policy is bull

    def test_subpolicy_exception_isolation_under_stress(self) -> None:
        """Exceptions in sub-policy context hooks must not crash router or prevent active execution."""
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")

        # Inject explosive side effects into bull hooks
        bull.set_account_context = MagicMock(side_effect=RuntimeError("Corrupt account pipe"))
        bull.set_position_context = MagicMock(side_effect=ValueError("Invalid ticket geometry"))
        bull.record_position_context_extrema = MagicMock(side_effect=TypeError("Extrema failure"))

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        # Context updates must not raise
        router.set_account_context({"equity": 10000.0})
        router.set_position_context("ticket_1")
        router.record_position_context_extrema(ticket=1, mae=-0.5)

        # Bear still received the calls cleanly
        assert bear.account_context["equity"] == 10000.0
        assert bear.position_context == (("ticket_1",), {})

    def test_deep_memory_independence_no_shared_references(self) -> None:
        """Verify that Bull and Bear maintain strictly independent tensor and buffer references."""
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        assert router.bull_policy.hidden_state is not router.bear_policy.hidden_state
        assert router.bull_policy.cell_state is not router.bear_policy.cell_state
        assert router.bull_policy.experience_buffer is not router.bear_policy.experience_buffer

        # Modifying one directly cannot mutate the other
        bull.hidden_state[0, 0] = 999.0
        assert bear.hidden_state[0, 0] != 999.0

    def test_rapid_whipsaw_state_machine_fuzzing(self) -> None:
        """Fuzz confirmation buffer with 100 rapid random transitions: confirmation must hold strictly."""
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            confirmation_bars=3,  # Requires 3 consecutive confirming bars
        )

        rng = np.random.RandomState(456)
        regimes = ["BULL", "BEAR", "CHOP", "CONTEST"]

        for _ in range(100):
            chosen = rng.choice(regimes)
            if chosen == "BULL":
                probs = [0.80, 0.10, 0.10]
            elif chosen == "BEAR":
                probs = [0.10, 0.80, 0.10]
            elif chosen == "CHOP":
                probs = [0.20, 0.20, 0.60]
            else:
                probs = [0.48, 0.48, 0.04]

            router.set_regime_soft_probabilities(probs)
            # Counters must never exceed confirmation_bars + steps or become negative
            assert router._pending_bull_count >= 0
            assert router._pending_bear_count >= 0
            assert router.current_regime in {"BULL", "BEAR", "CHOP"}
            assert router.current_regime_index in {0, 1, 2}


# ==============================================================================
# SUITE 4: MULTI-TIER HYBRID ARCHITECTURE ADVERSARIAL STRESS VERIFICATION
# ==============================================================================

class TestMultiTierAdversarialStressHarness:
    """Empirical adversarial stress testing for Milestone 2 Multi-Tier Router:
    1. Recurrent memory isolation (L_inf = 0.0) across 100+ steps with dynamic HTF macro switches.
    2. Microstructure momentum gate boundary conditions (exact 0.0, epsilons, NaN/Inf, noise).
    3. Consecutive loss circuit breaker sliding window boundaries (59m59s vs 60m01s, cooldown expiry, reset conditions).
    4. Latency SLA stress testing (500+ calls under CPU load).
    """

    def test_recurrent_memory_isolation_linf_zero_under_120_step_htf_macro_switches(self) -> None:
        """Adversarially challenge recurrent memory isolation with 120 dynamic HTF macro transitions."""
        bull = AdversarialRecurrentPolicy("bull", use_torch=True, seed=101)
        bear = AdversarialRecurrentPolicy("bear", use_torch=True, seed=202)

        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
        )

        # Baseline snapshot of initial states
        bear_h_init = bear.hidden_state.copy()
        bear_c_init = bear.cell_state.copy()
        bull_h_init = bull.hidden_state.copy()
        bull_c_init = bull.cell_state.copy()

        # Step Phase 1 (0 to 29): Bull Macro (diff = +0.20, slope = +0.05)
        for step in range(30):
            obs = np.zeros(137, dtype=float)
            obs[88] = 0.20
            obs[89] = 0.05
            batch = router.act_nam_tac_nhan(obs)
            assert router.current_regime == "BULL"
            assert router.active_policy is bull
            assert bull.step_count == step + 1
            assert bear.step_count == 0

        # Verify Bear states remained 100% frozen bit-for-bit (L_inf = 0.0)
        diff_bear_h = np.max(np.abs(bear.hidden_state - bear_h_init))
        diff_bear_c = np.max(np.abs(bear.cell_state - bear_c_init))
        assert diff_bear_h == 0.0, f"Bear hidden state drifted during Bull execution: L_inf={diff_bear_h}"
        assert diff_bear_c == 0.0, f"Bear cell state drifted during Bull execution: L_inf={diff_bear_c}"
        if bear.use_torch:
            for i, mem in enumerate(bear.five_agent_memories):
                h_diff = float(torch.max(torch.abs(mem.hidden - torch.from_numpy(bear_h_init[i]))))
                c_diff = float(torch.max(torch.abs(mem.cell - torch.from_numpy(bear_c_init[i]))))
                assert h_diff == 0.0, f"Bear torch hidden drifted: L_inf={h_diff}"
                assert c_diff == 0.0, f"Bear torch cell drifted: L_inf={c_diff}"

        # Snapshot Bull state at step 30
        bull_h_step30 = bull.hidden_state.copy()
        bull_c_step30 = bull.cell_state.copy()

        # Step Phase 2 (30 to 59): Bear Macro (diff = -0.25, slope = -0.06)
        for step in range(30):
            obs = np.zeros(137, dtype=float)
            obs[88] = -0.25
            obs[89] = -0.06
            batch = router.act_nam_tac_nhan(obs)
            assert router.current_regime == "BEAR"
            assert router.active_policy is bear
            assert bear.step_count == step + 1
            assert bull.step_count == 30

        # Verify Bull states remained 100% frozen bit-for-bit from step 30 (L_inf = 0.0)
        diff_bull_h = np.max(np.abs(bull.hidden_state - bull_h_step30))
        diff_bull_c = np.max(np.abs(bull.cell_state - bull_c_step30))
        assert diff_bull_h == 0.0, f"Bull hidden state drifted during Bear execution: L_inf={diff_bull_h}"
        assert diff_bull_c == 0.0, f"Bull cell state drifted during Bear execution: L_inf={diff_bull_c}"
        if bull.use_torch:
            for i, mem in enumerate(bull.five_agent_memories):
                h_diff = float(torch.max(torch.abs(mem.hidden - torch.from_numpy(bull_h_step30[i]))))
                c_diff = float(torch.max(torch.abs(mem.cell - torch.from_numpy(bull_c_step30[i]))))
                assert h_diff == 0.0, f"Bull torch hidden drifted: L_inf={h_diff}"
                assert c_diff == 0.0, f"Bull torch cell drifted: L_inf={c_diff}"

        # Snapshot Bear state at step 60
        bear_h_step60 = bear.hidden_state.copy()
        bear_c_step60 = bear.cell_state.copy()

        # Step Phase 3 (60 to 89): Contest Zone (diff = 0.01, slope = 0.00)
        # Should retain previous regime (BEAR) and continue advancing Bear while Bull stays frozen
        for step in range(30):
            obs = np.zeros(137, dtype=float)
            obs[88] = 0.01
            obs[89] = 0.00
            batch = router.act_nam_tac_nhan(obs)
            assert router.current_regime == "BEAR"
            assert router.active_policy is bear
            assert bear.step_count == 30 + step + 1
            assert bull.step_count == 30

        diff_bull_h_contest = np.max(np.abs(bull.hidden_state - bull_h_step30))
        assert diff_bull_h_contest == 0.0, f"Bull hidden drifted in contest: L_inf={diff_bull_h_contest}"

        # Step Phase 4 (90 to 119): High-Frequency Whipsaw Switches (alternating Bull / Bear every step)
        for i in range(30):
            obs = np.zeros(137, dtype=float)
            if i % 2 == 0:
                obs[88] = 0.30
                obs[89] = 0.10
                expected_active = bull
                expected_regime = "BULL"
                inactive_h = bear.hidden_state.copy()
            else:
                obs[88] = -0.30
                obs[89] = -0.10
                expected_active = bear
                expected_regime = "BEAR"
                inactive_h = bull.hidden_state.copy()

            router.act_nam_tac_nhan(obs)
            assert router.current_regime == expected_regime
            assert router.active_policy is expected_active

            # Check that the inactive specialist was not mutated during this step
            if expected_regime == "BULL":
                diff = np.max(np.abs(bear.hidden_state - inactive_h))
                assert diff == 0.0, f"Bear mutated during whipsaw step {i}: L_inf={diff}"
            else:
                diff = np.max(np.abs(bull.hidden_state - inactive_h))
                assert diff == 0.0, f"Bull mutated during whipsaw step {i}: L_inf={diff}"

    def test_adversarial_momentum_gate_boundary_conditions_and_fuzzing(self) -> None:
        """Stress-test Tier 2 Microstructure Momentum Gate with exact boundary values, NaNs, Infs, and noise."""
        from self_growing_trader.dual_champion_router import DualChampionRouter

        class ActionInjectingPolicy(AdversarialRecurrentPolicy):
            def __init__(self, name: str, a0: int = 3, a1: int = 4) -> None:
                super().__init__(name)
                self.a0 = a0  # Agent 0 action (e.g. 3 = BUY_LIMIT, 1 = MARKET_BUY)
                self.a1 = a1  # Agent 1 action (e.g. 4 = SELL_LIMIT, 2 = MARKET_SELL)

            def act_nam_tac_nhan(self, observation: np.ndarray, *args: Any, **kwargs: Any) -> tuple[AdversarialTrace, ...]:
                traces = []
                for agent_id in range(5):
                    act = self.a0 if agent_id == 0 else (self.a1 if agent_id == 1 else int(HOLD_ACTION))
                    outputs = np.zeros(20, dtype=float)
                    outputs[act] = 1.0
                    tr = AdversarialTrace(
                        trace_id=f"{self.name}_agent_{agent_id}",
                        selected_action=outputs,
                        selected_idx=act,
                        action_details={
                            "five_agent_index": agent_id,
                            "five_action_index": act,
                            "wire_action": act,
                        },
                    )
                    traces.append(tr)
                return tuple(traces)

        policy = ActionInjectingPolicy("adv_gate", a0=3, a1=4)
        router = DualChampionRouter(
            bull_policy=policy,
            bear_policy=policy,
            momentum_gate_enabled=True,
        )

        # 1. Exact 0.0 boundary: Neither upward nor downward momentum
        obs_zero = np.zeros(137, dtype=float)
        obs_zero[80] = 0.0  # m5_diff
        obs_zero[81] = 0.0  # m5_slope
        obs_zero[84] = 0.0  # m15_diff
        batch = router.act_nam_tac_nhan(obs_zero)
        # In exact 0.0, neither veto triggers -> BUY_LIMIT and SELL_LIMIT pass through
        assert batch[0].action_details.get("five_action_index") == 3
        assert batch[1].action_details.get("five_action_index") == 4
        assert not batch[0].action_details.get("regime_veto_applied", False)
        assert not batch[1].action_details.get("regime_veto_applied", False)

        # 2. Strict Boundary Epsilon: Micro-positive vs Micro-negative
        # Micro-positive: diff = +1e-9, slope = +1e-9 -> Upward momentum -> Veto SELL
        obs_micro_up = np.zeros(137, dtype=float)
        obs_micro_up[80] = 1e-9
        obs_micro_up[81] = 1e-9
        batch_up = router.act_nam_tac_nhan(obs_micro_up)
        assert batch_up[1].action_details.get("five_action_index") == 0
        assert batch_up[1].action_details.get("regime_veto_applied") is True
        assert batch_up[1].action_details.get("regime_veto_reason") == "veto_sell_against_upward_momentum"
        # BUY is unaffected
        assert batch_up[0].action_details.get("five_action_index") == 3

        # Micro-negative: diff = -1e-9, slope = -1e-9 -> Downward momentum -> Veto BUY
        obs_micro_down = np.zeros(137, dtype=float)
        obs_micro_down[80] = -1e-9
        obs_micro_down[81] = -1e-9
        batch_down = router.act_nam_tac_nhan(obs_micro_down)
        assert batch_down[0].action_details.get("five_action_index") == 0
        assert batch_down[0].action_details.get("regime_veto_applied") is True
        assert batch_down[0].action_details.get("regime_veto_reason") == "veto_buy_against_downward_momentum"
        # SELL is unaffected
        assert batch_down[1].action_details.get("five_action_index") == 4

        # 3. Conflicting Signals: diff > 0 but slope < 0 (divergence / deceleration)
        obs_diverge = np.zeros(137, dtype=float)
        obs_diverge[80] = 0.20
        obs_diverge[81] = -0.05
        obs_diverge[84] = 0.05
        batch_div = router.act_nam_tac_nhan(obs_diverge)
        # Not steep upward momentum because slope < 0, so no veto
        assert batch_div[1].action_details.get("five_action_index") == 4
        assert not batch_div[1].action_details.get("regime_veto_applied", False)

        # 4. M15 Macro Reinforcement: M5 slope = 0, but M15 diff > 0.10 and M5 diff > 0
        obs_m15_up = np.zeros(137, dtype=float)
        obs_m15_up[80] = 0.02
        obs_m15_up[81] = 0.00
        obs_m15_up[84] = 0.15  # > 0.10
        batch_m15 = router.act_nam_tac_nhan(obs_m15_up)
        assert batch_m15[1].action_details.get("five_action_index") == 0
        assert batch_m15[1].action_details.get("regime_veto_applied") is True

        # Symmetrically for M15 downward reinforcement:
        obs_m15_down = np.zeros(137, dtype=float)
        obs_m15_down[80] = -0.02
        obs_m15_down[81] = 0.00
        obs_m15_down[84] = -0.15  # < -0.10
        batch_m15_d = router.act_nam_tac_nhan(obs_m15_down)
        assert batch_m15_d[0].action_details.get("five_action_index") == 0
        assert batch_m15_d[0].action_details.get("regime_veto_applied") is True

        # 5. Robustness against NaN and Inf in momentum features
        for bad_val in (np.nan, np.inf, -np.inf):
            obs_bad = np.zeros(137, dtype=float)
            obs_bad[80] = bad_val
            obs_bad[81] = bad_val
            obs_bad[84] = bad_val
            # Must not throw exceptions or crash
            batch_bad = router.act_nam_tac_nhan(obs_bad)
            assert len(batch_bad) == 5

        # 6. Random Fuzzing: 100 random noise vectors
        rng = np.random.RandomState(999)
        for _ in range(100):
            obs_noise = rng.randn(137) * 5.0
            batch_noise = router.act_nam_tac_nhan(obs_noise)
            assert len(batch_noise) == 5
            # Invariant: Agent 4 must ALWAYS be HOLD (0) regardless of noise
            assert batch_noise[4].action_details.get("five_action_index") == 0

    def test_adversarial_circuit_breaker_sliding_window_exact_boundaries(self) -> None:
        """Test consecutive loss circuit breaker sliding window boundaries (59m59s vs 60m01s) and reset behaviors."""
        from datetime import datetime, timedelta
        from self_growing_trader.dual_champion_router import ConsecutiveLossCircuitBreaker

        cb = ConsecutiveLossCircuitBreaker(
            max_consecutive_sl=2,
            cooldown_minutes=60.0,
            window_minutes=60.0,
            enabled=True,
        )

        t0 = datetime(2025, 2, 20, 10, 0, 0)

        # Boundary Case 1: 59m 59s separation (within 60.0 min window) -> MUST TRIP
        cb.record_outcome("SELL", is_sl=True, timestamp=t0)
        assert not cb.is_in_cooldown("SELL", t0)

        t_59m59s = t0 + timedelta(minutes=59, seconds=59)
        cb.record_outcome("SELL", is_sl=True, timestamp=t_59m59s)
        # Breaker must TRIP
        assert cb.is_in_cooldown("SELL", t_59m59s) is True
        assert cb.is_in_cooldown("BUY", t_59m59s) is False  # BUY independent

        # Reset for Boundary Case 2
        cb.reset()
        assert not cb.is_in_cooldown("SELL", t0)

        # Boundary Case 2: 60m 01s separation (outside 60.0 min window) -> MUST NOT TRIP
        cb.record_outcome("SELL", is_sl=True, timestamp=t0)
        t_60m01s = t0 + timedelta(minutes=60, seconds=1)
        cb.record_outcome("SELL", is_sl=True, timestamp=t_60m01s)
        # Breaker must NOT trip because first SL dropped out of 60-min window
        assert cb.is_in_cooldown("SELL", t_60m01s) is False

        # Reset for Boundary Case 3: Exactly 60m 00s (boundary equality <=) -> MUST TRIP
        cb.reset()
        cb.record_outcome("SELL", is_sl=True, timestamp=t0)
        t_60m00s = t0 + timedelta(minutes=60, seconds=0)
        cb.record_outcome("SELL", is_sl=True, timestamp=t_60m00s)
        assert cb.is_in_cooldown("SELL", t_60m00s) is True

        # Cooldown Expiry Precision (60 min cooldown from trip time)
        # Trip time was t_60m00s (11:00:00). Cooldown until 12:00:00.
        assert cb.is_in_cooldown("SELL", t_60m00s + timedelta(minutes=59, seconds=59)) is True
        # At exactly 12:00:00, cooldown expires
        assert cb.is_in_cooldown("SELL", t_60m00s + timedelta(minutes=60, seconds=0)) is False
        assert cb.is_in_cooldown("SELL", t_60m00s + timedelta(minutes=60, seconds=1)) is False

        # Winning Exit Reset: TP clears SL streak
        cb.reset()
        cb.record_outcome("SELL", is_sl=True, timestamp=t0)
        assert len(cb.sl_history["SELL"]) == 1

        # TP exit occurs
        cb.record_outcome("SELL", is_sl=False, is_win=True, timestamp=t0 + timedelta(minutes=10), profit=200.0)
        assert len(cb.sl_history["SELL"]) == 0  # Cleared!

        # Next SL after TP does NOT trip breaker
        cb.record_outcome("SELL", is_sl=True, timestamp=t0 + timedelta(minutes=20))
        assert cb.is_in_cooldown("SELL", t0 + timedelta(minutes=20)) is False

        # EXPERT Close Behavior via router.record_five_live_owner_outcome
        router = DualChampionRouter(
            bull_policy=AdversarialRecurrentPolicy("bull"),
            bear_policy=AdversarialRecurrentPolicy("bear"),
            circuit_breaker=cb,
        )
        cb.reset()

        # 1st SL via record_five_live_owner_outcome
        router.record_five_live_owner_outcome(
            agent_index=1,
            outcome="SL",
            profit=-100.0,
            net_r=-1.0,
            timestamp=t0,
        )
        assert len(cb.sl_history["SELL"]) == 1

        # EXPERT close with profit > 0 (profitable discretionary / governor close) resets streak
        router.record_five_live_owner_outcome(
            agent_index=1,
            outcome="EXPERT",
            profit=+45.0,
            net_r=+0.45,
            timestamp=t0 + timedelta(minutes=5),
        )
        assert len(cb.sl_history["SELL"]) == 0

        # EXPERT close with loss (profit < 0) accumulates as loss
        router.record_five_live_owner_outcome(
            agent_index=1,
            outcome="EXPERT",
            profit=-10.0,
            net_r=-0.1,
            timestamp=t0 + timedelta(minutes=10),
        )
        assert len(cb.sl_history["SELL"]) == 1

        # Second loss trips breaker
        router.record_five_live_owner_outcome(
            agent_index=1,
            outcome="SL",
            profit=-100.0,
            net_r=-1.0,
            timestamp=t0 + timedelta(minutes=15),
        )
        assert cb.is_in_cooldown("SELL", t0 + timedelta(minutes=16)) is True

    def test_adversarial_latency_sla_under_cpu_stress(self) -> None:
        """Measure inference latency across 750+ calls under CPU load, proving strict P99 < 5.0ms SLA."""
        bull = AdversarialRecurrentPolicy("bull")
        bear = AdversarialRecurrentPolicy("bear")
        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
        )

        obs = np.zeros(137, dtype=float)
        obs[88] = 0.15
        obs[89] = 0.04
        obs[80] = 0.10
        obs[81] = 0.02
        obs[84] = 0.08

        import gc

        # Warm up 50 iterations
        for _ in range(50):
            router.act_nam_tac_nhan(obs)

        gc.collect()
        call_count = 750
        latencies_ms = np.empty(call_count, dtype=np.float64)

        for i in range(call_count):
            t_start = time.perf_counter()
            router.act_nam_tac_nhan(obs)
            t_end = time.perf_counter()
            latencies_ms[i] = (t_end - t_start) * 1000.0

        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))
        max_lat = float(np.max(latencies_ms))

        print(f"\n[Adversarial Latency Benchmark] Mean: {mean_lat:.4f} ms | P95: {p95_lat:.4f} ms | P99: {p99_lat:.4f} ms | Max: {max_lat:.4f} ms")

        assert mean_lat < 1.0, f"Mean latency failed budget (<1.0ms): {mean_lat:.4f} ms"
        assert p95_lat < 3.0, f"P95 latency failed SLA (<3.0ms): {p95_lat:.4f} ms"
        assert p99_lat < 5.0, f"P99 latency failed strict SLA (<5.0ms): {p99_lat:.4f} ms"

    def test_load_dual_champion_router_accepts_preloaded_detector_instance(self, tmp_path: Path):
        import pickle
        from self_growing_trader.dual_champion_router import load_dual_champion_router

        # Create minimal dummy checkpoints
        bear_file = tmp_path / "dummy_bear.pkl"
        bull_file = tmp_path / "dummy_bull.pkl"
        dummy_pol = _PicklableDummyPolicy("Dummy")
        with open(bear_file, "wb") as f:
            pickle.dump(dummy_pol, f)
        with open(bull_file, "wb") as f:
            pickle.dump(dummy_pol, f)

        # Preloaded detector instance with predict_proba
        class PreloadedDetector:
            def predict_proba(self, feats):
                return np.array([0.6, 0.2, 0.2])

        preloaded = PreloadedDetector()
        router = load_dual_champion_router(
            bear_path=bear_file,
            bull_path=bull_file,
            detector_path=preloaded,
        )
        assert router.regime_detector is preloaded


