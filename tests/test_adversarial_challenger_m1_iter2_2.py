"""
tests/test_adversarial_challenger_m1_iter2_2.py

Adversarial Stress & Empirical Verification Suite — Milestone 1 Iteration 2:
Empirical Challenger: challenger_m1_iter2_2

Verification Targets:
1. Recurrent State Freezing under Continuous Stepping (L_inf = 0.0, Bit-for-Bit exact IEEE-754):
   - Stepping Bull specialist 150+ times leaves Bear specialist hidden and cell states completely identical (L_inf = 0.0).
   - Stepping Bear specialist 150+ times leaves Bull specialist hidden and cell states completely identical (L_inf = 0.0).
   - PyTorch FiveAgentMemory (5 agents x 4096-D) strict tensor equality under continuous stepping.
2. Synchronous Context Updates Invariance:
   - set_account_context, set_decision_context, set_position_context correctly update metadata across both specialists.
   - Synchronous context updates do NOT modify, mutate, or leak into recurrent memory (L_inf = 0.0, zero bit drift).
3. End-to-End Offline Replay Verification in MT5StrategyTesterEvaluator._serve_policy_actions:
   - Full 50-packet offline replay across Bull, Bear, Chop, and Contest regimes.
   - Pre-annotated regime probabilities are strictly preserved and not overwritten.
   - Active specialist routing and step counts match regime distributions exactly.
   - Both learning_enabled=False and learning_enabled=True execute without error.
4. Adversarial Multi-Cycle & Boundary Stress:
   - Multi-round alternating stress (400 steps across 4 continuous segments) with interleaved context updates.
   - Duck-typing delegation for act(), observe_five_economic_transition(), set_experience_starvation().
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
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


@dataclass
class Iter2Trace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))
    selected_idx: int = int(HOLD_ACTION)
    confidence: float = 0.90
    action_details: dict[str, Any] = field(default_factory=dict)
    positions_pipe: str = ""
    positions_pipe_after: str = ""
    execution_details: dict[str, Any] = field(default_factory=dict)
    account_context: dict[str, float] = field(default_factory=dict)
    account_context_after: dict[str, float] = field(default_factory=dict)


class RealisticSpecialistModel:
    """Mock specialist simulating a 117M-param 5-agent shared policy.
    
    Maintains isolated 5 x 4096 recurrent state (numpy & torch) and metadata contexts.
    """

    def __init__(self, name: str, *, use_torch: bool = False, seed: int = 100) -> None:
        self.name = name
        self.use_torch = use_torch
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT
        self.step_count = 0

        rng = np.random.RandomState(seed)
        self.initial_hidden = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)
        self.initial_cell = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)

        self.hidden_state = self.initial_hidden.copy()
        self.cell_state = self.initial_cell.copy()

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
        self.positions_pipe: str = ""
        self.decision_context: dict[str, Any] = {}
        self.position_context_args: tuple[Any, ...] = ()
        self.position_context_kwargs: dict[str, Any] = {}
        self.raw_inputs: np.ndarray | None = None
        self.soft_probabilities: tuple[float, ...] | None = None
        self.starvation_signals: list[Any] = []
        self.transitions_observed: list[Any] = []

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
    ) -> tuple[Iter2Trace, ...]:
        self.step_count += 1
        obs_mag = float(np.mean(np.abs(observation))) if observation.size else 1.0
        delta = 0.05 * (obs_mag + float(self.step_count))

        self.hidden_state += delta
        self.cell_state += delta * 1.5

        if self.use_torch:
            for mem in self.five_agent_memories:
                mem.hidden.add_(delta)
                mem.cell.add_(delta * 1.5)

        traces = []
        for i in range(5):
            act_vec = np.zeros(20, dtype=float)
            act_vec[int(HOLD_ACTION)] = 1.0
            tr = Iter2Trace(
                trace_id=f"{self.name}_{self.step_count}_{i}",
                selected_action=act_vec,
                selected_idx=int(HOLD_ACTION),
                confidence=0.88,
                action_details={
                    "five_agent_index": i,
                    "five_action_index": int(HOLD_ACTION),
                    "policy": self.name,
                },
                positions_pipe=self.positions_pipe,
                account_context=dict(self.account_context),
            )
            traces.append(tr)
        return tuple(traces)

    def ensure_compatible(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        self.step_count = 0
        self.hidden_state = self.initial_hidden.copy()
        self.cell_state = self.initial_cell.copy()
        if self.use_torch:
            for i, mem in enumerate(self.five_agent_memories):
                mem.hidden.copy_(torch.from_numpy(self.hidden_state[i]))
                mem.cell.copy_(torch.from_numpy(self.cell_state[i]))

    def reset_five_economic_reward(self) -> None:
        pass

    def _expert_from_trace(self, trace: Any) -> Any:
        details = getattr(trace, "action_details", {}) or {}
        policy_name = details.get("policy", "")
        if policy_name == self.name:
            return self.experts.get(f"expert_{self.name}")
        return None

    def set_account_context(self, account_context: Mapping[str, float], positions_pipe: str = "") -> None:
        self.account_context = dict(account_context)
        self.positions_pipe = positions_pipe

    def set_decision_context(self, **kwargs: Any) -> None:
        self.decision_context = dict(kwargs)

    def set_position_context(self, *args: Any, **kwargs: Any) -> None:
        self.position_context_args = args
        self.position_context_kwargs = kwargs

    def set_regime_raw_inputs(self, raw_inputs: Any) -> None:
        self.raw_inputs = np.asarray(raw_inputs)

    def set_regime_soft_probabilities(self, probs: Any, **kwargs: Any) -> None:
        self.soft_probabilities = tuple(float(x) for x in probs)

    def set_experience_starvation(self, *args: Any, **kwargs: Any) -> None:
        self.starvation_signals.append((args, kwargs))

    def observe_five_economic_transition(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.transitions_observed.append(("five", args, kwargs))
        return {"status": "ok", "policy": self.name}

    def observe_transition(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.transitions_observed.append(("single", args, kwargs))
        return {"status": "ok", "policy": self.name}

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
# SECTION 1: CONTINUOUS STEPPING BIT-FOR-BIT STATE FREEZING (L_inf = 0.0)
# ==============================================================================

class TestContinuousSteppingBitForBitFreezing:
    """Rigorous verification of 100+ continuous steps state freezing with zero bit drift."""

    def test_bull_specialist_150_steps_bear_stays_frozen(self) -> None:
        """Continuous stepping Bull specialist 150 times leaves Bear completely untouched."""
        bull = RealisticSpecialistModel("bull", seed=1001)
        bear = RealisticSpecialistModel("bear", seed=2002)

        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            bull_threshold=0.55,
            bear_threshold=0.55,
            chop_threshold=0.40,
            confirmation_bars=1,
        )

        bear_h_snapshot = bear.hidden_state.copy()
        bear_c_snapshot = bear.cell_state.copy()
        bear_h_bytes = bear.hidden_state.tobytes()
        bear_c_bytes = bear.cell_state.tobytes()

        # Route to Bull
        router.set_regime_soft_probabilities([0.80, 0.10, 0.10], "BULL", 0)
        assert router.active_policy is bull

        rng = np.random.RandomState(42)
        for step in range(1, 151):
            obs = rng.randn(137).astype(np.float32)
            traces = router.act_nam_tac_nhan(obs)
            assert len(traces) == 5

            # Step-level verification
            assert np.max(np.abs(bear.hidden_state - bear_h_snapshot)) == 0.0
            assert np.max(np.abs(bear.cell_state - bear_c_snapshot)) == 0.0

        assert bull.step_count == 150
        assert bear.step_count == 0

        # L_inf distance must be mathematically exactly 0.0
        l_inf_hidden = float(np.max(np.abs(bear.hidden_state - bear_h_snapshot)))
        l_inf_cell = float(np.max(np.abs(bear.cell_state - bear_c_snapshot)))
        assert l_inf_hidden == 0.0, f"Bear hidden L_inf: {l_inf_hidden}"
        assert l_inf_cell == 0.0, f"Bear cell L_inf: {l_inf_cell}"

        # Byte representation exactness
        assert bear.hidden_state.tobytes() == bear_h_bytes
        assert bear.cell_state.tobytes() == bear_c_bytes

    def test_bear_specialist_150_steps_bull_stays_frozen(self) -> None:
        """Continuous stepping Bear specialist 150 times leaves Bull completely untouched."""
        bull = RealisticSpecialistModel("bull", seed=3003)
        bear = RealisticSpecialistModel("bear", seed=4004)

        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            bull_threshold=0.55,
            bear_threshold=0.55,
            chop_threshold=0.40,
            confirmation_bars=1,
        )

        bull_h_snapshot = bull.hidden_state.copy()
        bull_c_snapshot = bull.cell_state.copy()
        bull_h_bytes = bull.hidden_state.tobytes()
        bull_c_bytes = bull.cell_state.tobytes()

        # Route to Bear
        router.set_regime_soft_probabilities([0.10, 0.80, 0.10], "BEAR", 1)
        assert router.active_policy is bear

        rng = np.random.RandomState(84)
        for step in range(1, 151):
            obs = rng.randn(137).astype(np.float32)
            traces = router.act_nam_tac_nhan(obs)
            assert len(traces) == 5

            # Step-level verification
            assert np.max(np.abs(bull.hidden_state - bull_h_snapshot)) == 0.0
            assert np.max(np.abs(bull.cell_state - bull_c_snapshot)) == 0.0

        assert bear.step_count == 150
        assert bull.step_count == 0

        l_inf_hidden = float(np.max(np.abs(bull.hidden_state - bull_h_snapshot)))
        l_inf_cell = float(np.max(np.abs(bull.cell_state - bull_c_snapshot)))
        assert l_inf_hidden == 0.0, f"Bull hidden L_inf: {l_inf_hidden}"
        assert l_inf_cell == 0.0, f"Bull cell L_inf: {l_inf_cell}"

        assert bull.hidden_state.tobytes() == bull_h_bytes
        assert bull.cell_state.tobytes() == bull_c_bytes

    def test_torch_five_agent_memory_continuous_100_steps_isolation(self) -> None:
        """PyTorch FiveAgentMemory tensors stay strictly equal under continuous stepping."""
        bull = RealisticSpecialistModel("bull", use_torch=True, seed=5005)
        bear = RealisticSpecialistModel("bear", use_torch=True, seed=6006)

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        bear_torch_h0 = [m.hidden.clone() for m in bear.five_agent_memories]
        bear_torch_c0 = [m.cell.clone() for m in bear.five_agent_memories]

        # Route to Bull and step 120 times
        router.set_regime_soft_probabilities([0.85, 0.05, 0.10], "BULL", 0)
        rng = np.random.RandomState(999)
        for _ in range(120):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))

        for idx in range(5):
            mem = bear.five_agent_memories[idx]
            assert torch.equal(mem.hidden, bear_torch_h0[idx])
            assert torch.equal(mem.cell, bear_torch_c0[idx])
            assert torch.max(torch.abs(mem.hidden - bear_torch_h0[idx])).item() == 0.0
            assert torch.max(torch.abs(mem.cell - bear_torch_c0[idx])).item() == 0.0


# ==============================================================================
# SECTION 2: SYNCHRONOUS CONTEXT UPDATES INVARIANCE
# ==============================================================================

class TestSynchronousContextUpdatesInvariance:
    """Verify that metadata context updates sync to both models without touching recurrent state."""

    def test_synchronous_context_updates_metadata_and_preserves_recurrent_memory(self) -> None:
        bull = RealisticSpecialistModel("bull", seed=7007)
        bear = RealisticSpecialistModel("bear", seed=8008)

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        bull_h_before = bull.hidden_state.copy()
        bull_c_before = bull.cell_state.copy()
        bear_h_before = bear.hidden_state.copy()
        bear_c_before = bear.cell_state.copy()

        # Repeatedly update contexts with changing data
        for i in range(50):
            acc_data = {"balance": 10000.0 + i * 10, "equity": 10050.0 + i * 5}
            pipe_data = f"TICKET={100+i};TYPE=BUY"
            router.set_account_context(acc_data, pipe_data)

            router.set_decision_context(epoch=i, step=i * 2, mode="stress")
            router.set_position_context(f"pos_{i}", flag=(i % 2 == 0))
            router.set_regime_raw_inputs(np.full(137, fill_value=float(i)))

            # Verify both models received the metadata update
            assert bull.account_context["equity"] == 10050.0 + i * 5
            assert bear.account_context["equity"] == 10050.0 + i * 5
            assert bull.positions_pipe == pipe_data
            assert bear.positions_pipe == pipe_data
            assert bull.decision_context["epoch"] == i
            assert bear.decision_context["epoch"] == i
            assert bull.position_context_args == (f"pos_{i}",)
            assert bear.position_context_args == (f"pos_{i}",)
            assert np.all(bull.raw_inputs == float(i))
            assert np.all(bear.raw_inputs == float(i))

        # Check recurrent memory invariance: L_inf MUST BE EXACTLY 0.0 on BOTH models
        assert np.max(np.abs(bull.hidden_state - bull_h_before)) == 0.0
        assert np.max(np.abs(bull.cell_state - bull_c_before)) == 0.0
        assert np.max(np.abs(bear.hidden_state - bear_h_before)) == 0.0
        assert np.max(np.abs(bear.cell_state - bear_c_before)) == 0.0

        assert bull.hidden_state.tobytes() == bull_h_before.tobytes()
        assert bull.cell_state.tobytes() == bull_c_before.tobytes()
        assert bear.hidden_state.tobytes() == bear_h_before.tobytes()
        assert bear.cell_state.tobytes() == bear_c_before.tobytes()


# ==============================================================================
# SECTION 3: OFFLINE REPLAY IN _serve_policy_actions
# ==============================================================================

class TestOfflineReplayInServePolicyActions:
    """Verify offline replay in MT5StrategyTesterEvaluator._serve_policy_actions."""

    def test_offline_replay_50_packets_with_preserved_probabilities(self) -> None:
        bull = RealisticSpecialistModel("bull", seed=9009)
        bear = RealisticSpecialistModel("bear", seed=1010)
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

            # Generate 50 synthetic requests:
            # 1-15: Bull (P_Bull=0.80)
            # 16-30: Bear (P_Bear=0.80)
            # 31-40: Chop (P_Chop=0.70)
            # 41-50: Contest (P_Bull=0.48, P_Bear=0.48, P_Chop=0.04) -> holds Chop (Bear)
            rng = np.random.RandomState(321)
            requests: list[BridgeRequest] = []
            for seq in range(1, 51):
                if seq <= 15:
                    probs = (0.80, 0.10, 0.10)
                    r_name = "BULL"
                    r_idx = 0
                elif seq <= 30:
                    probs = (0.10, 0.80, 0.10)
                    r_name = "BEAR"
                    r_idx = 1
                elif seq <= 40:
                    probs = (0.15, 0.15, 0.70)
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
                    equity=10000.0 + seq * 2.0,
                    positions=0,
                    positions_pipe="",
                    time=f"2025.02.24 12:{seq:02d}",
                    regime_probabilities=probs,
                    regime_name=r_name,
                    regime_index=r_idx,
                )
                requests.append(req)

            report_path = tmp_path / "TesterReport.htm"
            report_path.touch()

            mock_process = MagicMock()
            mock_process.poll.return_value = 0

            served_count = evaluator._serve_policy_actions(
                process=mock_process,
                report_path=report_path,
                policy=router,
                policy_id=1,
                round_index=0,
                replay_requests=requests,
            )

            assert served_count == 50
            # Bull was active for requests 1-15 (15 steps)
            assert bull.step_count == 15
            # Bear was active for requests 16-50 (15 Bear + 10 Chop + 10 Contest = 35 steps)
            assert bear.step_count == 35

    def test_offline_replay_learning_enabled_observes_transitions(self) -> None:
        """With learning_enabled=True, transitions are observed without breaking."""
        bull = RealisticSpecialistModel("bull", seed=1111)
        bear = RealisticSpecialistModel("bear", seed=2222)
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = MT5BacktestConfig(
                runner_dir=tmp_path,
                offline_replay=True,
                learning_enabled=True,  # Non-frozen mode
                action_file="mt5_action.csv",
                request_file="mt5_request.csv",
                bridge_file="mt5_bridge.csv",
                execution_file="mt5_execution.csv",
            )
            evaluator = MT5StrategyTesterEvaluator(config)

            rng = np.random.RandomState(555)
            requests = [
                BridgeRequest(
                    sequence=i,
                    event_id=f"evt_{i}",
                    inputs=rng.randn(137).astype(np.float32),
                    raw_inputs=rng.randn(137).astype(np.float32),
                    balance=10000.0,
                    equity=10000.0,
                    positions=0,
                    positions_pipe="",
                    time=f"2025.02.24 14:{i:02d}",
                    regime_probabilities=(0.80, 0.10, 0.10) if i <= 5 else (0.10, 0.80, 0.10),
                    regime_name="BULL" if i <= 5 else "BEAR",
                    regime_index=0 if i <= 5 else 1,
                )
                for i in range(1, 11)
            ]

            report_path = tmp_path / "TesterReport.htm"
            report_path.touch()

            mock_process = MagicMock()
            mock_process.poll.return_value = 0

            served_count = evaluator._serve_policy_actions(
                process=mock_process,
                report_path=report_path,
                policy=router,
                policy_id=1,
                round_index=0,
                replay_requests=requests,
            )

            assert served_count == 10
            assert bull.step_count == 5
            assert bear.step_count == 5
            # Transitions were propagated
            assert len(bull.transitions_observed) + len(bear.transitions_observed) >= 1


# ==============================================================================
# SECTION 4: MULTI-CYCLE ALTERNATING STRESS & DUCK-TYPING DELEGATION
# ==============================================================================

class TestMultiCycleAlternatingStress:
    """Stress testing 400 continuous steps across 4 alternating segments with interleaved context."""

    def test_400_step_alternating_execution_with_interleaved_context(self) -> None:
        bull = RealisticSpecialistModel("bull", seed=3333)
        bear = RealisticSpecialistModel("bear", seed=4444)
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        rng = np.random.RandomState(777)

        # Segment 1: Bull 100 steps
        router.set_regime_soft_probabilities([0.85, 0.05, 0.10], "BULL", 0)
        bear_h_seg1 = bear.hidden_state.copy()
        bear_c_seg1 = bear.cell_state.copy()
        for _ in range(100):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))
        assert bull.step_count == 100
        assert bear.step_count == 0
        assert np.max(np.abs(bear.hidden_state - bear_h_seg1)) == 0.0
        assert np.max(np.abs(bear.cell_state - bear_c_seg1)) == 0.0

        # Interleaved context update
        router.set_account_context({"equity": 10500.0}, "TICKET=1")
        router.set_decision_context(phase="seg1_complete")
        assert np.max(np.abs(bear.hidden_state - bear_h_seg1)) == 0.0
        assert np.max(np.abs(bear.cell_state - bear_c_seg1)) == 0.0

        # Segment 2: Bear 100 steps
        router.set_regime_soft_probabilities([0.05, 0.85, 0.10], "BEAR", 1)
        bull_h_seg2 = bull.hidden_state.copy()
        bull_c_seg2 = bull.cell_state.copy()
        for _ in range(100):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))
        assert bear.step_count == 100
        assert bull.step_count == 100  # unchanged
        assert np.max(np.abs(bull.hidden_state - bull_h_seg2)) == 0.0
        assert np.max(np.abs(bull.cell_state - bull_c_seg2)) == 0.0

        # Interleaved context update
        router.set_account_context({"equity": 11000.0}, "TICKET=2")
        assert np.max(np.abs(bull.hidden_state - bull_h_seg2)) == 0.0
        assert np.max(np.abs(bull.cell_state - bull_c_seg2)) == 0.0

        # Segment 3: Bull 100 steps
        router.set_regime_soft_probabilities([0.85, 0.05, 0.10], "BULL", 0)
        bear_h_seg3 = bear.hidden_state.copy()
        bear_c_seg3 = bear.cell_state.copy()
        for _ in range(100):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))
        assert bull.step_count == 200
        assert bear.step_count == 100  # unchanged
        assert np.max(np.abs(bear.hidden_state - bear_h_seg3)) == 0.0
        assert np.max(np.abs(bear.cell_state - bear_c_seg3)) == 0.0

        # Segment 4: Bear 100 steps
        router.set_regime_soft_probabilities([0.05, 0.85, 0.10], "BEAR", 1)
        bull_h_seg4 = bull.hidden_state.copy()
        bull_c_seg4 = bull.cell_state.copy()
        for _ in range(100):
            router.act_nam_tac_nhan(rng.randn(137).astype(np.float32))
        assert bear.step_count == 200
        assert bull.step_count == 200  # unchanged
        assert np.max(np.abs(bull.hidden_state - bull_h_seg4)) == 0.0
        assert np.max(np.abs(bull.cell_state - bull_c_seg4)) == 0.0

    def test_act_duck_typing_delegation_fallback(self) -> None:
        """router.act() works when sub-policy only implements act_nam_tac_nhan()."""
        bull = RealisticSpecialistModel("bull")
        bear = RealisticSpecialistModel("bear")
        # Ensure sub-policy has NO act() method
        assert not hasattr(bull, "act")
        assert not hasattr(bear, "act")

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)
        # Calling router.act() should fall back to self.act_nam_tac_nhan()
        res = router.act(np.zeros(137, dtype=np.float32))
        assert isinstance(res, tuple)
        assert len(res) == 5

    def test_set_experience_starvation_propagates_to_both(self) -> None:
        bull = RealisticSpecialistModel("bull")
        bear = RealisticSpecialistModel("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        router.set_experience_starvation(idle_bars=10, reason="low_variance")
        assert len(bull.starvation_signals) == 1
        assert len(bear.starvation_signals) == 1
        assert bull.starvation_signals[0] == ((), {"idle_bars": 10, "reason": "low_variance"})
        assert bear.starvation_signals[0] == ((), {"idle_bars": 10, "reason": "low_variance"})


# ==============================================================================
# SECTION 5: CROSS-REGIME TRACE OWNERSHIP & FAULT ISOLATION
# ==============================================================================

class TestCrossRegimeLifecycleTraceOwnership:
    """Verify trace ownership resolution across regime boundaries and fault isolation."""

    def test_expert_from_trace_cross_regime_resolution(self) -> None:
        """Positions opened in Bull mode must be resolvable when settling in Bear mode."""
        bull = RealisticSpecialistModel("bull")
        bear = RealisticSpecialistModel("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        # 1. Step in Bull mode -> generate Bull trace
        router.set_regime_soft_probabilities([0.80, 0.10, 0.10], "BULL", 0)
        assert router.active_policy is bull
        traces = router.act_nam_tac_nhan(np.zeros(137, dtype=np.float32))
        bull_trace = traces[0]

        # 2. Switch router to Bear mode
        router.set_regime_soft_probabilities([0.10, 0.80, 0.10], "BEAR", 1)
        assert router.active_policy is bear

        # 3. Resolve expert for Bull trace while active policy is Bear
        expert = router._expert_from_trace(bull_trace)
        assert expert is not None
        assert expert is bull.experts["expert_bull"]

        # 4. Switch router to defensive Chop mode
        router.set_regime_soft_probabilities([0.15, 0.15, 0.70], "CHOP", 2)
        assert router.active_policy is bear
        expert_chop = router._expert_from_trace(bull_trace)
        assert expert_chop is bull.experts["expert_bull"]

    def test_context_update_exception_containment(self) -> None:
        """Exceptions raised in one sub-policy must not crash context synchronization."""
        bull = RealisticSpecialistModel("bull")
        bear = RealisticSpecialistModel("bear")

        # Inject exploding hooks
        bull.set_account_context = MagicMock(side_effect=RuntimeError("MQL pipe corrupt"))
        bull.set_decision_context = MagicMock(side_effect=KeyError("Missing key"))
        bull.set_position_context = MagicMock(side_effect=ValueError("Bad size"))

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # Must execute cleanly without unhandled exception
        router.set_account_context({"equity": 10200.0}, "TICKET=99")
        router.set_decision_context(frame=42)
        router.set_position_context("ticket_99")

        # Bear received the calls intact
        assert bear.account_context["equity"] == 10200.0
        assert bear.decision_context["frame"] == 42
        assert bear.position_context_args == ("ticket_99",)

    def test_offline_replay_action_file_wire_contract(self) -> None:
        """Verify that offline replay generates a valid action file obeying the Governor HOLD lock."""
        bull = RealisticSpecialistModel("bull", seed=777)
        bear = RealisticSpecialistModel("bear", seed=888)
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            action_filename = "test_action.csv"
            config = MT5BacktestConfig(
                runner_dir=tmp_path,
                offline_replay=True,
                learning_enabled=False,
                action_file=action_filename,
                request_file="test_request.csv",
                bridge_file="test_bridge.csv",
                execution_file="test_execution.csv",
            )
            evaluator = MT5StrategyTesterEvaluator(config)

            req = BridgeRequest(
                sequence=1,
                event_id="evt_action_test",
                inputs=np.zeros(137, dtype=np.float32),
                raw_inputs=np.zeros(137, dtype=np.float32),
                balance=10000.0,
                equity=10000.0,
                positions=0,
                positions_pipe="",
                time="2025.02.24 15:00",
                regime_probabilities=(0.80, 0.10, 0.10),
                regime_name="BULL",
                regime_index=0,
            )

            report_path = tmp_path / "TesterReport.htm"
            report_path.touch()

            mock_process = MagicMock()
            mock_process.poll.return_value = 0

            evaluator._serve_policy_actions(
                process=mock_process,
                report_path=report_path,
                policy=router,
                policy_id=1,
                round_index=0,
                replay_requests=[req],
            )

            action_file_path = evaluator.mql5_files / action_filename
            assert action_file_path.exists(), f"Action file {action_file_path} was not created!"
            lines = action_file_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) >= 2, f"Expected header and at least one row, got {len(lines)}"

            # Inspect data row (MT5 action file is tab-delimited)
            data_row = lines[1].split("\t")
            assert data_row[1] == "evt_action_test"
            assert "bull_1_0" in data_row[2]

