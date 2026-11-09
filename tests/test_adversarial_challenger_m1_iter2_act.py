"""
tests/test_adversarial_challenger_m1_iter2_act.py

Adversarial Stress Test Suite for Milestone 1 Iteration 2:
Empirical challenge of DualChampionRouter.act() method under:
1. Varied kwargs and signature compatibility (strict signatures, extra kwargs, kwargs fallback)
2. Missing goals, None goals, positional vs keyword goals, goal_context resolution
3. Observation shapes (1D, 2D, non-contiguous, read-only, lists, NaN/Inf, empty arrays)
4. Recurrent memory isolation under repeated act() invocations (L_inf = 0.0)
5. Underlying policy variations (act only, act_nam_tac_nhan only, both, neither, exceptions)
6. Regime distribution switching via act() kwargs
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    INPUT_COUNT,
    OUTPUT_COUNT,
)
from self_growing_trader.mt5_runner import MarketRegime
from self_growing_trader.five_agent_authority import HOLD_ACTION
from self_growing_trader.five_shared_policy import FiveAgentMemory, FIVE_LSTM_SIZE


class StrictSignatureActPolicy:
    """Specialist policy that ONLY accepts (observation, goal) strictly - NO *args, NO **kwargs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0
        self.last_observation: Any = None
        self.last_goal: Any = None
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT
        self.hidden_state = np.zeros((5, FIVE_LSTM_SIZE), dtype=np.float32)
        self.cell_state = np.zeros((5, FIVE_LSTM_SIZE), dtype=np.float32)

    def act(self, observation: np.ndarray, goal: np.ndarray | None = None) -> str:
        self.call_count += 1
        self.last_observation = observation
        self.last_goal = goal
        return f"{self.name}_acted"


class KwargsAcceptingActPolicy:
    """Specialist policy that accepts *args and **kwargs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0
        self.last_observation: Any = None
        self.last_goal: Any = None
        self.last_args: tuple[Any, ...] = ()
        self.last_kwargs: dict[str, Any] = {}
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT
        self.hidden_state = np.zeros((5, FIVE_LSTM_SIZE), dtype=np.float32)
        self.cell_state = np.zeros((5, FIVE_LSTM_SIZE), dtype=np.float32)

    def act(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_observation = observation
        self.last_goal = goal
        self.last_args = args
        self.last_kwargs = kwargs
        return {"policy": self.name, "args": args, "kwargs": kwargs}


class FiveAgentOnlyPolicy:
    """Specialist policy that ONLY provides act_nam_tac_nhan (no act method)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0
        self.last_observation: Any = None
        self.last_goal: Any = None
        self.last_kwargs: dict[str, Any] = {}
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT
        self.hidden_state = np.ones((5, FIVE_LSTM_SIZE), dtype=np.float32) * 3.14
        self.cell_state = np.ones((5, FIVE_LSTM_SIZE), dtype=np.float32) * 2.71

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, ...]:
        self.call_count += 1
        self.last_observation = observation
        self.last_goal = goal
        self.last_kwargs = kwargs
        # Mutate hidden state to prove execution occurred
        self.hidden_state += 0.05
        self.cell_state += 0.10
        return tuple(f"{self.name}_agent_{i}" for i in range(5))


class StrictFiveAgentOnlyPolicy:
    """Specialist policy with strict act_nam_tac_nhan(observation, goal) - NO **kwargs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0
        self.last_observation: Any = None
        self.last_goal: Any = None
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
    ) -> tuple[str, ...]:
        self.call_count += 1
        self.last_observation = observation
        self.last_goal = goal
        return tuple(f"{self.name}_strict_{i}" for i in range(5))


class DualExposingPolicy:
    """Specialist policy exposing BOTH act and act_nam_tac_nhan."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.act_called = 0
        self.five_called = 0
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT

    def act(self, observation: np.ndarray, goal: np.ndarray | None = None, *args: Any, **kwargs: Any) -> str:
        self.act_called += 1
        return f"{self.name}_act"

    def act_nam_tac_nhan(self, observation: np.ndarray, goal: np.ndarray | None = None, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        self.five_called += 1
        return tuple(f"{self.name}_five_{i}" for i in range(5))


class NoActPolicy:
    """Specialist policy exposing NEITHER act nor act_nam_tac_nhan."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT


class InternalTypeErrorPolicy:
    """Specialist policy whose act() raises a genuine TypeError from inside its body."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT

    def act(self, observation: np.ndarray, goal: np.ndarray | None = None, *args: Any, **kwargs: Any) -> Any:
        # Deliberate internal TypeError unrelated to arguments:
        return "string" + 12345


# ==============================================================================
# TEST SUITE
# ==============================================================================

class TestActMethodKwargsAndSignatures:
    """Stress test DualChampionRouter.act() with varied signatures and kwargs."""

    def test_act_delegates_to_kwargs_accepting_policy(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)
        # Default active is BEAR
        res = router.act(
            np.zeros(137),
            np.ones(10),
            "pos_arg1",
            custom_kw="hello",
            arbitrary_num=999,
        )
        assert bear.call_count == 1
        assert bull.call_count == 0
        assert res["policy"] == "bear"
        assert res["args"] == ("pos_arg1",)
        assert res["kwargs"] == {"custom_kw": "hello", "arbitrary_num": 999}

    def test_act_gracefully_degrades_for_strict_act_signature(self) -> None:
        """When subpolicy act() has strict (obs, goal) signature, extra kwargs should be caught and dropped."""
        bull = StrictSignatureActPolicy("bull")
        bear = StrictSignatureActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        # Call with extra kwargs that strict act() does not accept
        res = router.act(np.zeros(137), goal=np.ones(10), extra_kwarg="should_be_dropped")
        assert bear.call_count == 1
        assert res == "bear_acted"
        assert np.array_equal(bear.last_goal, np.ones(10))

    def test_act_delegates_to_five_agent_only_policy(self) -> None:
        """When subpolicy lacks act() but has act_nam_tac_nhan(), router.act() delegates smoothly."""
        bull = FiveAgentOnlyPolicy("bull")
        bear = FiveAgentOnlyPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        res = router.act(np.zeros(137))
        assert bear.call_count == 1
        assert len(res) == 5
        assert res[0] == "bear_agent_0"

    def test_act_gracefully_degrades_for_strict_five_agent_signature(self) -> None:
        """When subpolicy act_nam_tac_nhan() has strict signature, extra kwargs are dropped."""
        bull = StrictFiveAgentOnlyPolicy("bull")
        bear = StrictFiveAgentOnlyPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        res = router.act(np.zeros(137), goal=np.ones(5), extra_kw="discarded")
        assert bear.call_count == 1
        assert len(res) == 5
        assert res[0] == "bear_strict_0"

    def test_act_prefers_act_over_act_nam_tac_nhan_if_both_present(self) -> None:
        """When active specialist exposes both, act() calls act()."""
        bull = DualExposingPolicy("bull")
        bear = DualExposingPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        res = router.act(np.zeros(137))
        assert bear.act_called == 1
        assert bear.five_called == 0
        assert res == "bear_act"

    def test_act_raises_runtime_error_when_neither_present(self) -> None:
        bull = NoActPolicy("bull")
        bear = NoActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        with pytest.raises(RuntimeError, match="does not expose act or act_nam_tac_nhan"):
            router.act(np.zeros(137))

    def test_act_regime_distribution_switching_via_kwargs(self) -> None:
        """Passing current_regime_distribution to act() switches the active specialist."""
        bull = FiveAgentOnlyPolicy("bull")
        bear = FiveAgentOnlyPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        assert router.active_policy is bear

        # Switch to BULL regime via act kwargs
        res1 = router.act(
            np.zeros(137),
            current_regime_distribution=(0.85, 0.05, 0.10),
        )
        assert router.active_policy is bull
        assert bull.call_count == 1
        assert bear.call_count == 0
        assert res1[0] == "bull_agent_0"

        # Switch back to BEAR regime via act kwargs
        res2 = router.act(
            np.zeros(137),
            current_regime_distribution=(0.05, 0.85, 0.10),
        )
        assert router.active_policy is bear
        assert bear.call_count == 1
        assert res2[0] == "bear_agent_0"


class TestActMissingAndVariedGoals:
    """Stress test goal parameter variations."""

    def test_act_no_goal_passed(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        router.act(np.zeros(137))
        assert bear.last_goal is None

    def test_act_positional_goal_passed(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        goal_arr = np.array([1.0, 2.0, 3.0])
        router.act(np.zeros(137), goal_arr)
        assert np.array_equal(bear.last_goal, goal_arr)

    def test_act_keyword_goal_passed(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        goal_arr = np.array([4.0, 5.0, 6.0])
        router.act(np.zeros(137), goal=goal_arr)
        assert np.array_equal(bear.last_goal, goal_arr)

    def test_act_goal_context_kwarg_resolution(self) -> None:
        """When five-agent fallback occurs, goal_context in kwargs is resolved as effective goal."""
        bull = FiveAgentOnlyPolicy("bull")
        bear = FiveAgentOnlyPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        goal_ctx = np.array([7.0, 8.0, 9.0])
        router.act(np.zeros(137), goal_context=goal_ctx)
        assert np.array_equal(bear.last_goal, goal_ctx)

    def test_act_goal_overrides_goal_context(self) -> None:
        """When both goal and goal_context are provided, goal takes precedence."""
        bull = FiveAgentOnlyPolicy("bull")
        bear = FiveAgentOnlyPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        goal_explicit = np.array([10.0, 20.0])
        goal_ctx = np.array([99.0, 99.0])
        router.act(np.zeros(137), goal=goal_explicit, goal_context=goal_ctx)
        assert np.array_equal(bear.last_goal, goal_explicit)


class TestActObservationShapesAndTypes:
    """Stress test diverse observation shapes, memory layouts, and numerical anomalies."""

    def test_act_1d_float32_and_float64(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        router.act(np.ones(137, dtype=np.float32))
        assert bear.last_observation.dtype == np.float32
        router.act(np.ones(137, dtype=np.float64))
        assert bear.last_observation.dtype == np.float64

    def test_act_2d_shape_observation(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        obs_2d = np.ones((5, 137), dtype=np.float32)
        router.act(obs_2d)
        assert bear.last_observation.shape == (5, 137)

    def test_act_non_contiguous_and_readonly_observation(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        raw = np.arange(274, dtype=np.float32)
        non_contig = raw[::2]
        assert not non_contig.flags.c_contiguous

        router.act(non_contig)
        assert bear.last_observation.shape == (137,)

        # Read-only
        read_only = np.ones(137, dtype=np.float32)
        read_only.setflags(write=False)
        router.act(read_only)
        assert not bear.last_observation.flags.writeable

    def test_act_nan_and_inf_observations(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        obs_nan = np.full(137, np.nan, dtype=np.float32)
        router.act(obs_nan)
        assert np.isnan(bear.last_observation).all()

        obs_inf = np.full(137, np.inf, dtype=np.float32)
        router.act(obs_inf)
        assert np.isinf(bear.last_observation).all()

    def test_act_empty_observation(self) -> None:
        bull = KwargsAcceptingActPolicy("bull")
        bear = KwargsAcceptingActPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        empty_obs = np.zeros(0, dtype=np.float32)
        router.act(empty_obs)
        assert bear.last_observation.size == 0


class TestActRecurrentMemoryIsolationStress:
    """Verify that repeated router.act() calls strictly preserve L_inf = 0.0 on inactive policy."""

    def test_act_100_steps_memory_isolation(self) -> None:
        bull = FiveAgentOnlyPolicy("bull")
        bear = FiveAgentOnlyPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        # Baseline states
        orig_bull_h = bull.hidden_state.copy()
        orig_bull_c = bull.cell_state.copy()
        orig_bear_h = bear.hidden_state.copy()
        orig_bear_c = bear.cell_state.copy()

        # Step 1: Execute 100 steps under Bear regime (default)
        for i in range(100):
            obs = np.random.randn(137).astype(np.float32)
            router.act(obs)

        # Bear should have stepped 100 times, modifying its state
        assert bear.call_count == 100
        assert not np.array_equal(bear.hidden_state, orig_bear_h)
        assert not np.array_equal(bear.cell_state, orig_bear_c)

        # Bull MUST remain 100% frozen: L_inf = 0.0
        assert bull.call_count == 0
        diff_h = np.max(np.abs(bull.hidden_state - orig_bull_h))
        diff_c = np.max(np.abs(bull.cell_state - orig_bull_c))
        assert diff_h == 0.0
        assert diff_c == 0.0
        assert bull.hidden_state.tobytes() == orig_bull_h.tobytes()
        assert bull.cell_state.tobytes() == orig_bull_c.tobytes()

        # Step 2: Switch to Bull regime and execute 100 steps
        router.set_regime_soft_probabilities((0.85, 0.05, 0.10))
        assert router.active_policy is bull
        bear_snapshot_h = bear.hidden_state.copy()
        bear_snapshot_c = bear.cell_state.copy()

        for i in range(100):
            obs = np.random.randn(137).astype(np.float32)
            router.act(obs)

        # Bull should now have stepped 100 times
        assert bull.call_count == 100
        # Bear MUST remain 100% frozen: L_inf = 0.0
        assert bear.call_count == 100  # unchanged
        diff_bear_h = np.max(np.abs(bear.hidden_state - bear_snapshot_h))
        diff_bear_c = np.max(np.abs(bear.cell_state - bear_snapshot_c))
        assert diff_bear_h == 0.0
        assert diff_bear_c == 0.0
        assert bear.hidden_state.tobytes() == bear_snapshot_h.tobytes()
        assert bear.cell_state.tobytes() == bear_snapshot_c.tobytes()


class TestActInternalTypeErrorPropagation:
    """Verify that an internal TypeError inside subpolicy act() is NOT swallowed."""

    def test_internal_type_error_propagates(self) -> None:
        bull = InternalTypeErrorPolicy("bull")
        bear = InternalTypeErrorPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        with pytest.raises(TypeError, match="can only concatenate str"):
            router.act(np.zeros(137))


class TestCrossRegimeTraceResolutionAndEconomicTransitions:
    """Verify cross-regime trace resolution and economic transition fallback."""

    def test_expert_from_trace_fallback_when_regime_changes(self) -> None:
        """Trace produced under Bull regime must be resolved even after router switches to Bear."""
        class MockSpecialistWithTraces:
            def __init__(self, name: str) -> None:
                self.name = name
                self.input_size = INPUT_COUNT
                self.output_size = OUTPUT_COUNT
                self.known_traces: dict[str, Any] = {}
                self.transitions: list[Any] = []

            def _expert_from_trace(self, trace: Any) -> Any:
                t_id = getattr(trace, "trace_id", str(trace))
                return self.known_traces.get(t_id)

            def observe_five_economic_transition(self, *args: Any, **kwargs: Any) -> Any:
                self.transitions.append((args, kwargs))
                return {"received_by": self.name}

        bull = MockSpecialistWithTraces("bull")
        bear = MockSpecialistWithTraces("bear")
        bull.known_traces["trace_opened_in_bull"] = "expert_bull_owner"
        bear.known_traces["trace_opened_in_bear"] = "expert_bear_owner"

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        # Active policy is bear by default
        assert router.active_policy is bear

        # Query bear trace -> resolved directly by active
        dummy_bear_trace = MagicMock(trace_id="trace_opened_in_bear")
        assert router._expert_from_trace(dummy_bear_trace) == "expert_bear_owner"

        # Query bull trace while bear is active -> resolved by fallback to other specialist (bull)!
        dummy_bull_trace = MagicMock(trace_id="trace_opened_in_bull")
        assert router._expert_from_trace(dummy_bull_trace) == "expert_bull_owner"

        # Query completely unknown trace -> returns None without exception
        unknown_trace = MagicMock(trace_id="trace_unknown")
        assert router._expert_from_trace(unknown_trace) is None

        # Economic transition fallback
        res = router.observe_five_economic_transition(ticket=123, reward=0.5)
        # Active is bear, so bear receives it
        assert len(bear.transitions) == 1
        assert res == {"received_by": "bear"}


class TestEvaluatorOfflineReplayAdversarialStress:
    """Stress test MT5StrategyTesterEvaluator._serve_policy_actions with DualChampionRouter."""

    def test_evaluator_offline_replay_with_malformed_and_missing_regimes(self, tmp_path: Any) -> None:
        """Run offline replay where requests include missing regimes, NaN vectors, and regime changes."""
        from self_growing_trader.mt5_runner import (
            BridgeRequest,
            MT5BacktestConfig,
            MT5StrategyTesterEvaluator,
        )

        class ReplayMockSpecialist:
            def __init__(self, name: str) -> None:
                self.name = name
                self.input_size = INPUT_COUNT
                self.output_size = OUTPUT_COUNT
                self.step_count = 0
                self.cau_hinh = MagicMock()
                self.cau_hinh.che_do_quyet_dinh = "nam_tac_nhan_dung_chung"
                mock_expert = MagicMock()
                mock_expert.policy = MagicMock()
                mock_expert.policy.action_recovery_learner = MagicMock(cells={})
                self.experts = {f"exp_{name}": mock_expert}
                self.nhan_dien = MagicMock()
                self.registry = MagicMock()

            def act_nam_tac_nhan(
                self,
                observation: np.ndarray,
                goal: np.ndarray | None = None,
                *args: Any,
                **kwargs: Any,
            ) -> tuple[Any, ...]:
                self.step_count += 1
                traces = []
                for a in range(5):
                    tr = MagicMock()
                    tr.trace_id = f"{self.name}_step_{self.step_count}_agent_{a}"
                    outputs = np.zeros(20, dtype=float)
                    outputs[int(HOLD_ACTION)] = 1.0
                    tr.selected_action = outputs
                    tr.selected_idx = int(HOLD_ACTION)
                    tr.allowed_action_mask = ()
                    tr.action_details = {
                        "five_agent_index": a,
                        "selected_action": int(HOLD_ACTION),
                        "selected_action_name": "HOLD",
                    }
                    tr.positions_pipe = ""
                    tr.positions_pipe_after = ""
                    traces.append(tr)
                return tuple(traces)

        bull = ReplayMockSpecialist("bull")
        bear = ReplayMockSpecialist("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=1)

        config = MT5BacktestConfig(
            runner_dir=tmp_path,
            offline_replay=True,
            learning_enabled=False,
            action_file="action.csv",
            request_file="request.csv",
            bridge_file="bridge.csv",
            execution_file="execution.csv",
        )
        evaluator = MT5StrategyTesterEvaluator(config)

        # Build 40 requests:
        # 1-10: Bull regime explicit
        # 11-20: Bear regime explicit
        # 21-30: No regime probabilities (raw inputs fallback to neutral prior CHOP)
        # 31-40: NaN / Inf vectors (fail closed to neutral prior CHOP)
        rng = np.random.RandomState(42)
        requests: list[BridgeRequest] = []
        for seq in range(1, 41):
            if seq <= 10:
                probs = (0.80, 0.10, 0.10)
                r_name = "BULL"
                r_idx = 0
                raw = rng.randn(137).astype(np.float32)
            elif seq <= 20:
                probs = (0.10, 0.80, 0.10)
                r_name = "BEAR"
                r_idx = 1
                raw = rng.randn(137).astype(np.float32)
            elif seq <= 30:
                probs = None
                r_name = None
                r_idx = None
                raw = rng.randn(137).astype(np.float32)
            else:
                probs = None
                r_name = None
                r_idx = None
                raw = np.full(137, np.nan, dtype=np.float32) if seq % 2 == 0 else np.full(137, np.inf, dtype=np.float32)

            req = BridgeRequest(
                sequence=seq,
                event_id=f"evt_{seq}",
                inputs=raw if not np.isnan(raw).any() and not np.isinf(raw).any() else np.zeros(137, dtype=np.float32),
                raw_inputs=raw,
                balance=10000.0,
                equity=10000.0,
                positions=0,
                positions_pipe="",
                time=f"2025.02.24 10:{seq:02d}",
                regime_probabilities=probs,
                regime_name=r_name,
                regime_index=r_idx,
            )
            requests.append(req)

        report_path = tmp_path / "report.htm"
        report_path.touch()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0

        served = evaluator._serve_policy_actions(
            process=mock_proc,
            report_path=report_path,
            policy=router,
            policy_id=1,
            round_index=0,
            replay_requests=requests,
        )

        assert served == 40
        # Bull was active during steps 1-10
        assert bull.step_count == 10
        # Bear was active for the rest (steps 11-40)
        assert bear.step_count == 30

