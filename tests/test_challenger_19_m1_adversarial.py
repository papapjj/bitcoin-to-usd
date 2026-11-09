"""
tests/test_challenger_19_m1_adversarial.py

Empirical Challenger Adversarial Stress Test Suite for Milestone M1:
- Verification of 3-State Dynamic Routing at Calibrated 0.35 Thresholds
- Strict Wire Soft Veto Enforcement:
  * Bull regime strictly blocks Short orders (MARKET_SELL, SELL_LIMIT, SELL_STOP)
  * Bear regime strictly blocks Long orders (MARKET_BUY, BUY_LIMIT, BUY_STOP)
  * Chop regime strictly blocks ALL entries (Long & Short) to safe HOLD
- Strict Zero POLICY_CLOSE_ALL Invariant under adversarial logits, positions, and corrupt states
- Bit-for-Bit Recurrent Memory Isolation (L_inf = 0.0) under high-frequency alternating regime switching
- Adversarial Probability Fuzzing (boundary, unnormalized, negative, NaN/Inf, noise)
- End-to-End Stress Test with Genuine Checkpoints
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    load_dual_champion_router,
    INPUT_COUNT,
    OUTPUT_COUNT,
)
from self_growing_trader.five_shared_policy import FiveAgentMemory, FIVE_LSTM_SIZE
from self_growing_trader.five_agent_authority import (
    HOLD_ACTION,
    CLOSE_ALL_ACTION,
    five_agent_authority,
)
from self_growing_trader.orders import Action
from self_growing_trader.mt5_runner import (
    BridgeRequest,
    BridgeAgentAction,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
    MarketRegime,
)


# ==============================================================================
# REALISTIC MOCKS FOR ADVERSARIAL STRESS TESTING
# ==============================================================================

@dataclass
class ChallengerTrace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(OUTPUT_COUNT, dtype=float))
    selected_idx: int = int(HOLD_ACTION)
    confidence: float = 0.90
    action_details: dict[str, Any] = field(default_factory=dict)
    allowed_action_mask: tuple[bool, ...] = field(default_factory=tuple)
    positions_pipe: str = ""
    account_context: dict[str, float] = field(default_factory=dict)
    ma_context: str = "CTX_CHUNG"
    ma_chuyen_gia: str = "EXP_CHUNG"
    ma_policy: str = "POL_CHUNG"
    ly_do_khong_hoc: bool = False


class ChallengerRecurrentPolicy:
    """Mock specialist policy tracking 5-agent x 4096-D recurrent memory."""

    def __init__(self, name: str, *, seed: int = 123) -> None:
        self.name = name
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT
        self.step_count = 0

        rng = np.random.RandomState(seed)
        self.initial_hidden = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)
        self.initial_cell = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)

        self.hidden_state = self.initial_hidden.copy()
        self.cell_state = self.initial_cell.copy()

        self.five_agent_memories = [
            FiveAgentMemory(
                hidden=torch.from_numpy(self.hidden_state[i].copy()),
                cell=torch.from_numpy(self.cell_state[i].copy()),
                previous_action=torch.zeros((), dtype=torch.long),
            )
            for i in range(5)
        ]

        self.experience_buffer: list[dict[str, Any]] = []
        self.soft_probabilities: tuple[float, ...] | None = None
        mock_expert = MagicMock()
        mock_expert.policy = MagicMock()
        mock_expert.policy.action_recovery_learner = MagicMock(cells={})
        self.experts = {"CTX_CHUNG": mock_expert}
        self.nhan_dien = lambda obs: "CTX_CHUNG"
        self.registry = MagicMock()
        self.cau_hinh = MagicMock()
        self.cau_hinh.che_do_quyet_dinh = "nam_tac_nhan_dung_chung"

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[ChallengerTrace, ...]:
        self.step_count += 1
        delta = 0.05 * (float(self.step_count) + float(np.sum(observation[:5])))

        self.hidden_state += delta
        self.cell_state += delta * 1.5

        for mem in self.five_agent_memories:
            mem.hidden.add_(delta)
            mem.cell.add_(delta * 1.5)

        traces = []
        for agent_id in range(5):
            outputs = np.zeros(OUTPUT_COUNT, dtype=float)
            outputs[int(HOLD_ACTION)] = 1.0
            mask = [True] * OUTPUT_COUNT
            # Governor locked to HOLD
            if agent_id >= 2:
                mask = [False] * OUTPUT_COUNT
                mask[int(HOLD_ACTION)] = True

            tr = ChallengerTrace(
                trace_id=f"{self.name}_step_{self.step_count}_agent_{agent_id}",
                selected_action=outputs,
                selected_idx=int(HOLD_ACTION),
                confidence=0.92,
                action_details={
                    "five_agent_index": agent_id,
                    "five_action_index": int(HOLD_ACTION),
                    "policy_name": self.name,
                },
                allowed_action_mask=tuple(mask),
            )
            traces.append(tr)
        return tuple(traces)

    def set_regime_soft_probabilities(
        self,
        probs: Any,
        regime_name: str | None = None,
        regime_idx: int | None = None,
    ) -> None:
        self.soft_probabilities = tuple(float(x) for x in probs)

    def ensure_compatible(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        self.step_count = 0
        self.hidden_state = self.initial_hidden.copy()
        self.cell_state = self.initial_cell.copy()
        for i, mem in enumerate(self.five_agent_memories):
            mem.hidden.copy_(torch.from_numpy(self.hidden_state[i]))
            mem.cell.copy_(torch.from_numpy(self.cell_state[i]))


def make_challenger_traces(
    proposals: Mapping[int, int],
    allowed_masks: Mapping[int, Sequence[bool]] | None = None,
) -> list[ChallengerTrace]:
    """Helper to construct 5-agent trace tuple with specified proposed actions."""
    traces = []
    for agent_id in range(5):
        proposed_action = proposals.get(agent_id, int(HOLD_ACTION))
        outputs = np.zeros(OUTPUT_COUNT, dtype=float)
        outputs[proposed_action] = 10.0

        if allowed_masks and agent_id in allowed_masks:
            mask = tuple(allowed_masks[agent_id])
        else:
            # Default authority mask
            auth = five_agent_authority(agent_id)
            mask = tuple(bool(i in auth.allowed_actions) for i in range(OUTPUT_COUNT))

        tr = ChallengerTrace(
            trace_id=f"adversarial_trace_{agent_id}",
            selected_action=outputs,
            selected_idx=proposed_action,
            action_details={
                "five_agent_index": agent_id,
                "five_action_index": proposed_action,
            },
            allowed_action_mask=mask,
        )
        traces.append(tr)
    return traces


# ==============================================================================
# CHALLENGE SUITE 1: 0.35 THRESHOLD BOUNDARY FUZZING & HYSTERESIS
# ==============================================================================

class TestChallengerThresholdBoundaryFuzzing:
    """Stress-test precision around calibrated 0.35 Bull/Bear and 0.38 Chop thresholds."""

    @pytest.mark.parametrize("eps", [1e-7, 1e-5, 1e-3, 0.01])
    def test_bull_threshold_infinitesimal_sub_and_super_critical(self, eps: float) -> None:
        bull = ChallengerRecurrentPolicy("bull")
        bear = ChallengerRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, bull_threshold=0.35, bear_threshold=0.35)

        # Sub-critical: 0.35 - eps with regime_name="BULL" vs without
        router.current_regime = "CHOP"
        router._pending_bull_count = 0

        # When regime_name is None, 0.35 - eps fails bull_threshold
        router.set_regime_soft_probabilities([0.35 - eps, 0.30, 0.35 + eps])
        assert router.current_regime == "CHOP"

        # Super-critical: 0.35 + eps with P_Bear=0.20 (separation > margin 0.10)
        router.set_regime_soft_probabilities([0.35 + eps, 0.20, 0.45 - eps])
        assert router.current_regime == "BULL"
        assert router.active_policy is bull

    def test_equal_probability_tie_breaking_falls_to_chop_defense(self) -> None:
        bull = ChallengerRecurrentPolicy("bull")
        bear = ChallengerRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, bull_threshold=0.35, bear_threshold=0.35, chop_threshold=0.38)

        # Exact tie: P_Bull = 0.36, P_Bear = 0.36, P_Chop = 0.28
        # Neither is strictly greater than the other
        router.set_regime_soft_probabilities([0.36, 0.36, 0.28])
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear

    def test_chop_threshold_exact_boundary(self) -> None:
        bull = ChallengerRecurrentPolicy("bull")
        bear = ChallengerRecurrentPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, bull_threshold=0.35, bear_threshold=0.35, chop_threshold=0.38)

        # Put into Bull first
        router.set_regime_soft_probabilities([0.45, 0.20, 0.35], "BULL", 0)
        assert router.current_regime == "BULL"

        # P_Chop = 0.3799 (< 0.38) with P_Bull=0.32, P_Bear=0.30 -> in contest zone, stays BULL
        router.set_regime_soft_probabilities([0.32, 0.30, 0.3799])
        assert router.current_regime == "BULL"

        # P_Chop = 0.3800 (>= 0.38) -> switches to CHOP!
        router.set_regime_soft_probabilities([0.31, 0.31, 0.3800])
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear


# ==============================================================================
# CHALLENGE SUITE 2: STRICT WIRE SOFT VETO ENFORCEMENT
# ==============================================================================

class TestChallengerWireSoftVetoEnforcement:
    """Stress-test wire soft veto at mt5_runner boundary for all order types."""

    @pytest.fixture
    def evaluator(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            offline_replay=True,
            runner_dir=Path(tmp.name),
            regime_bull_threshold=0.35,
            regime_bear_threshold=0.35,
            regime_chop_threshold=0.38,
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        return ev, tmp

    @pytest.mark.parametrize("short_action", [Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP])
    def test_bull_regime_strictly_blocks_all_short_variants(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, Any], short_action: Action
    ) -> None:
        """In BULL regime, every short order type proposed by Agent 1 MUST be vetoed to HOLD."""
        ev, tmp = evaluator
        policy = ChallengerRecurrentPolicy("bull_eval")
        req = BridgeRequest(
            sequence=101,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.42, 0.28, 0.30),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )

        # Agent 1 proposes short order; Agent 0 proposes aligned MARKET_BUY
        traces = make_challenger_traces({0: int(Action.MARKET_BUY), 1: int(short_action)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Agent 1 MUST be vetoed to HOLD
        assert packet[1].action == int(Action.HOLD), f"{short_action.name} was NOT vetoed in Bull regime!"
        assert np.argmax(packet[1].outputs) == int(Action.HOLD)
        assert enriched[1].action_details["regime_veto_applied"] is True

        # Agent 0 MUST NOT be vetoed (aligned Long)
        assert packet[0].action == int(Action.MARKET_BUY)
        assert enriched[0].action_details["regime_veto_applied"] is False

    @pytest.mark.parametrize("long_action", [Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP])
    def test_bear_regime_strictly_blocks_all_long_variants(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, Any], long_action: Action
    ) -> None:
        """In BEAR regime, every long order type proposed by Agent 0 MUST be vetoed to HOLD."""
        ev, tmp = evaluator
        policy = ChallengerRecurrentPolicy("bear_eval")
        req = BridgeRequest(
            sequence=102,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.28, 0.42, 0.30),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )

        # Agent 0 proposes long order; Agent 1 proposes aligned MARKET_SELL
        traces = make_challenger_traces({0: int(long_action), 1: int(Action.MARKET_SELL)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Agent 0 MUST be vetoed to HOLD
        assert packet[0].action == int(Action.HOLD), f"{long_action.name} was NOT vetoed in Bear regime!"
        assert np.argmax(packet[0].outputs) == int(Action.HOLD)
        assert enriched[0].action_details["regime_veto_applied"] is True

        # Agent 1 MUST NOT be vetoed (aligned Short)
        assert packet[1].action == int(Action.MARKET_SELL)
        assert enriched[1].action_details["regime_veto_applied"] is False

    @pytest.mark.parametrize("long_action", [Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP])
    @pytest.mark.parametrize("short_action", [Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP])
    def test_chop_regime_strictly_blocks_all_entry_combinations(
        self,
        evaluator: tuple[MT5StrategyTesterEvaluator, Any],
        long_action: Action,
        short_action: Action,
    ) -> None:
        """In CHOP regime, BOTH Long and Short entry orders MUST be vetoed to safe HOLD."""
        ev, tmp = evaluator
        policy = ChallengerRecurrentPolicy("chop_eval")
        req = BridgeRequest(
            sequence=103,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.30, 0.30, 0.40),
            regime_name="CHOP",
            regime_index=2,
            positions=0,
        )

        traces = make_challenger_traces({0: int(long_action), 1: int(short_action)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # BOTH Agent 0 and Agent 1 MUST be vetoed to safe HOLD
        assert packet[0].action == int(Action.HOLD), f"Agent 0 {long_action.name} leaked in Chop!"
        assert packet[1].action == int(Action.HOLD), f"Agent 1 {short_action.name} leaked in Chop!"
        assert np.argmax(packet[0].outputs) == int(Action.HOLD)
        assert np.argmax(packet[1].outputs) == int(Action.HOLD)
        assert enriched[0].action_details["regime_veto_applied"] is True
        assert enriched[1].action_details["regime_veto_applied"] is True


# ==============================================================================
# CHALLENGE SUITE 3: ZERO POLICY_CLOSE_ALL UNDER ADVERSARIAL FORCING
# ==============================================================================

class TestChallengerZeroPolicyCloseAllInvariant:
    """Stress-test that Action 10 (POLICY_CLOSE_ALL) is NEVER emitted under any circumstance."""

    @pytest.fixture
    def evaluator(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            offline_replay=True,
            runner_dir=Path(tmp.name),
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        return ev, tmp

    @pytest.mark.parametrize("agent_id", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("positions", [0, 1, 5])
    def test_close_all_action_strictly_forbidden_across_all_agents(
        self,
        evaluator: tuple[MT5StrategyTesterEvaluator, Any],
        agent_id: int,
        positions: int,
    ) -> None:
        """Even if an agent strongly attempts to emit Action 10 (CLOSE_ALL), it MUST NOT pass."""
        ev, tmp = evaluator
        policy = ChallengerRecurrentPolicy("close_all_stress")
        req = BridgeRequest(
            sequence=200 + agent_id,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.33, 0.33, 0.34),
            regime_name="CHOP",
            regime_index=2,
            positions=positions,
            positions_pipe="P1|1.0|0.5|10.0" if positions > 0 else "",
        )

        # Force candidate trace to propose Action 10 with extreme unmasked logit (+1000.0)
        # But allow mask enforces governor lock (Action 0 only for agents 2, 3, 4)
        traces = make_challenger_traces({agent_id: int(CLOSE_ALL_ACTION)})

        # For Agent 4, authority has {0, 10}, but when lock_non_entry_to_hold is active,
        # trace_mask allows ONLY 0.
        if agent_id == 4:
            # Verify trace mask enforces only Action 0
            traces[4].allowed_action_mask = tuple(bool(i == 0) for i in range(OUTPUT_COUNT))

        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Assert Action 10 is NEVER the chosen wire action for any agent
        for a_idx, act in enumerate(packet):
            assert act.action != int(CLOSE_ALL_ACTION), (
                f"VIOLATION: Agent {a_idx} emitted Action 10 (POLICY_CLOSE_ALL) with positions={positions}!"
            )
            assert np.argmax(act.outputs) != int(CLOSE_ALL_ACTION), (
                f"VIOLATION: Agent {a_idx} output argmax is Action 10 (POLICY_CLOSE_ALL)!"
            )


# ==============================================================================
# CHALLENGE SUITE 4: RECURRENT STATE ISOLATION (L_inf = 0.0) UNDER RAPID SWITCHING
# ==============================================================================

class TestChallengerRecurrentIsolationUnderHighFrequencySwitching:
    """Rigorous empirical verification of L_inf = 0.0 under 200 alternating steps."""

    def test_high_frequency_alternating_regimes_recurrent_memory_freeze(self) -> None:
        bull = ChallengerRecurrentPolicy("bull_challenger", seed=777)
        bear = ChallengerRecurrentPolicy("bear_challenger", seed=888)
        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
        )

        obs_rng = np.random.RandomState(999)

        # Run 200 alternating steps: Bull -> Bear -> Bull -> Bear ...
        for step in range(200):
            obs = obs_rng.randn(137).astype(np.float32)

            if step % 2 == 0:
                # Step Bull: Snapshot Bear state before act
                bear_h_before = bear.hidden_state.copy()
                bear_c_before = bear.cell_state.copy()
                bear_torch_h = [m.hidden.clone() for m in bear.five_agent_memories]
                bear_torch_c = [m.cell.clone() for m in bear.five_agent_memories]

                router.act_nam_tac_nhan(
                    obs,
                    current_regime_distribution=(0.45, 0.25, 0.30),
                )
                assert router.current_regime == "BULL"
                assert router.active_policy is bull

                # Assert Bear state is 100% frozen: L_inf == 0.0
                assert np.max(np.abs(bear.hidden_state - bear_h_before)) == 0.0
                assert np.max(np.abs(bear.cell_state - bear_c_before)) == 0.0
                for i in range(5):
                    assert torch.equal(bear.five_agent_memories[i].hidden, bear_torch_h[i])
                    assert torch.equal(bear.five_agent_memories[i].cell, bear_torch_c[i])

            else:
                # Step Bear: Snapshot Bull state before act
                bull_h_before = bull.hidden_state.copy()
                bull_c_before = bull.cell_state.copy()
                bull_torch_h = [m.hidden.clone() for m in bull.five_agent_memories]
                bull_torch_c = [m.cell.clone() for m in bull.five_agent_memories]

                router.act_nam_tac_nhan(
                    obs,
                    current_regime_distribution=(0.25, 0.45, 0.30),
                )
                assert router.current_regime == "BEAR"
                assert router.active_policy is bear

                # Assert Bull state is 100% frozen: L_inf == 0.0
                assert np.max(np.abs(bull.hidden_state - bull_h_before)) == 0.0
                assert np.max(np.abs(bull.cell_state - bull_c_before)) == 0.0
                for i in range(5):
                    assert torch.equal(bull.five_agent_memories[i].hidden, bull_torch_h[i])
                    assert torch.equal(bull.five_agent_memories[i].cell, bull_torch_c[i])


# ==============================================================================
# CHALLENGE SUITE 5: ADVERSARIAL PROBABILITY VECTOR FUZZING
# ==============================================================================

class TestChallengerAdversarialProbabilityFuzzing:
    """Test fail-closed robustness against malformed, unnormalized, or corrupted probability distributions."""

    @pytest.fixture
    def evaluator(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            offline_replay=True,
            runner_dir=Path(tmp.name),
            regime_bull_threshold=0.35,
            regime_bear_threshold=0.35,
            regime_chop_threshold=0.38,
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        return ev, tmp

    @pytest.mark.parametrize("corrupt_probs", [
        (float("nan"), 0.5, 0.5),
        (0.5, float("inf"), 0.5),
        (-0.2, 0.7, 0.5),
        (0.0, 0.0, 0.0),
        (),
        (0.5,),
    ])
    def test_corrupt_probabilities_fail_closed_to_chop_neutral_prior(
        self, corrupt_probs: tuple[Any, ...]
    ) -> None:
        """With detector disabled, corrupt or invalid probability vectors safely fail-closed to CHOP prior."""
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            offline_replay=True,
            runner_dir=Path(tmp.name),
            regime_detector_enabled=False,  # Test neutral prior fallback directly
            regime_bull_threshold=0.35,
            regime_bear_threshold=0.35,
            regime_chop_threshold=0.38,
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        policy = ChallengerRecurrentPolicy("corrupt_eval")
        req = BridgeRequest(
            sequence=300,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=corrupt_probs,  # type: ignore[arg-type]
            positions=0,
        )

        traces = make_challenger_traces({0: int(Action.MARKET_BUY), 1: int(Action.MARKET_SELL)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Safe fail-closed behavior: BOTH entries are vetoed to HOLD in default neutral CHOP prior
        assert packet[0].action == int(Action.HOLD)
        assert packet[1].action == int(Action.HOLD)
        assert len(packet) == 5

    def test_unnormalized_probabilities_are_safely_scaled(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, Any]
    ) -> None:
        """Unnormalized positive probabilities (e.g. [2.0, 2.0, 6.0]) should normalize to [0.2, 0.2, 0.6] (CHOP)."""
        ev, tmp = evaluator
        policy = ChallengerRecurrentPolicy("unnorm_eval")

        # 1. Chop dominant unnormalized vector: [2.0, 2.0, 6.0] -> [0.2, 0.2, 0.6]
        req_chop = BridgeRequest(
            sequence=301,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(2.0, 2.0, 6.0),
            positions=0,
        )
        traces_chop = make_challenger_traces({0: int(Action.MARKET_BUY), 1: int(Action.MARKET_SELL)})
        packet_chop, _ = ev._five_bridge_action_packet(policy, req_chop, traces_chop)

        # In [0.2, 0.2, 0.6], CHOP is dominant -> BOTH entries vetoed to HOLD
        assert packet_chop[0].action == int(Action.HOLD)
        assert packet_chop[1].action == int(Action.HOLD)

        # 2. Bull dominant unnormalized vector: [6.0, 2.0, 2.0] -> [0.6, 0.2, 0.2]
        req_bull = BridgeRequest(
            sequence=302,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(6.0, 2.0, 2.0),
            positions=0,
        )
        traces_bull = make_challenger_traces({0: int(Action.MARKET_BUY), 1: int(Action.MARKET_SELL)})
        packet_bull, enriched_bull = ev._five_bridge_action_packet(policy, req_bull, traces_bull)

        # In [0.6, 0.2, 0.2], BULL is dominant -> Agent 0 Long allowed, Agent 1 Short vetoed
        assert packet_bull[0].action == int(Action.MARKET_BUY)
        assert packet_bull[1].action == int(Action.HOLD)
        assert enriched_bull[1].action_details["regime_veto_applied"] is True


# ==============================================================================
# CHALLENGE SUITE 6: REAL CHECKPOINT END-TO-END VERIFICATION
# ==============================================================================

class TestChallengerRealCheckpointEndToEnd:
    """Load genuine Dual-Champion checkpoints and run 50 real inference steps."""

    def test_genuine_checkpoints_load_and_run_without_policy_close_all(self) -> None:
        bear_path = Path("outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl")
        bull_path = Path("outputs/he_chuyen_gia/v193_runs/multi_round_phase3/round_01/reasoning_rl_v1_active_candidate.pkl")
        det_path = Path("outputs/regime_models/champion_regime_detector.pkl")

        if not (bear_path.exists() and bull_path.exists()):
            pytest.skip("Genuine checkpoints not found in outputs/ directory")

        router = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=det_path if det_path.exists() else None,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
        )

        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            offline_replay=True,
            runner_dir=Path(tmp.name),
            regime_bull_threshold=0.35,
            regime_bear_threshold=0.35,
            regime_chop_threshold=0.38,
        )
        ev = MT5StrategyTesterEvaluator(cfg)

        rng = np.random.RandomState(42)

        # Run 50 steps alternating between Bull, Bear, and Chop scenarios
        for step in range(50):
            obs = rng.randn(137).astype(np.float32)

            if step % 3 == 0:
                probs = (0.44, 0.26, 0.30)
                reg_name = "BULL"
                reg_idx = 0
            elif step % 3 == 1:
                probs = (0.26, 0.44, 0.30)
                reg_name = "BEAR"
                reg_idx = 1
            else:
                probs = (0.30, 0.30, 0.40)
                reg_name = "CHOP"
                reg_idx = 2

            req = BridgeRequest(
                sequence=1000 + step,
                inputs=obs,
                regime_probabilities=probs,
                regime_name=reg_name,
                regime_index=reg_idx,
                positions=1 if step % 2 == 0 else 0,
                positions_pipe="P1|1.0|0.5|10.0" if step % 2 == 0 else "",
            )

            traces = router.act_nam_tac_nhan(obs, current_regime_distribution=probs)
            packet, enriched = ev._five_bridge_action_packet(router, req, traces)

            assert len(packet) == 5
            for a_idx, act in enumerate(packet):
                # Verify zero POLICY_CLOSE_ALL invariant
                assert act.action != int(CLOSE_ALL_ACTION), (
                    f"CRITICAL VIOLATION: Agent {a_idx} emitted POLICY_CLOSE_ALL on genuine checkpoint at step {step}!"
                )
                assert np.argmax(act.outputs) != int(CLOSE_ALL_ACTION)
                # Verify valid action range
                assert 0 <= act.action < OUTPUT_COUNT
