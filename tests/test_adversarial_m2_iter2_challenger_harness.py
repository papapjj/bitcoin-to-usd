"""
tests/test_adversarial_m2_iter2_challenger_harness.py

Milestone 2 Iteration 2 Adversarial Challenge Suite:
Empirical Stress Harness for Soft Veto Wire Output Forcing & Boundary Fuzzing.

Verifications:
1. Bull Regime (P_Bull >= 0.55):
   - Agent 1 Short orders (MARKET_SELL, SELL_LIMIT, SELL_STOP) forced to HOLD (0).
   - Wire output tensor argmax[:12] == 0, argmax[:] == 0.
   - decode_outputs() produces Action.HOLD.
   - Telemetry confirms regime_veto_applied == True.
2. Bear Regime (P_Bear >= 0.55):
   - Agent 0 Long orders (MARKET_BUY, BUY_LIMIT, BUY_STOP) forced to HOLD (0).
   - Wire output tensor argmax[:12] == 0, argmax[:] == 0.
   - decode_outputs() produces Action.HOLD.
   - Telemetry confirms regime_veto_applied == True.
3. Trend-Aligned Actions Strictly Preserved:
   - Bull: Agent 0 Long entries unvetoed (MARKET_BUY, BUY_LIMIT, BUY_STOP), argmax preserved.
   - Bear: Agent 1 Short entries unvetoed (MARKET_SELL, SELL_LIMIT, SELL_STOP), argmax preserved.
4. Exact Boundary Transitions at P = 0.5500:
   - P_Bull = 0.5499 (no veto) vs P_Bull = 0.5501 (veto applied).
   - P_Bear = 0.5499 (no veto) vs P_Bear = 0.5501 (veto applied).
   - P_Bull = 0.5500 exactly (veto applied).
   - P_Bear = 0.5500 exactly (veto applied).
5. Chop Regime (P_Chop >= 0.40):
   - DualChampionRouter transitions to CHOP and activates defensive Bear specialist.
   - Confirmation counters reset to 0.
   - P_Chop = 0.399999 in contest zone holds previous state vs P_Chop = 0.400000 triggers CHOP.
6. Extreme Adversarial Stress:
   - Massive negative logits on HOLD (-1e9) + massive positive logits on Short (+1e9).
   - Random fuzzing of 500 boundary probability vectors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import numpy as np
import pytest

from self_growing_trader.orders import (
    Action,
    ACTION_COUNT,
    OUTPUT_COUNT,
    RiskConfig,
    OrderConfig,
    decode_outputs,
)
from self_growing_trader.mt5_runner import (
    BridgeRequest,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
)
from self_growing_trader.dual_champion_router import DualChampionRouter
from self_growing_trader.five_agent_authority import HOLD_ACTION


@dataclass
class SimpleTrace:
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
    ma_context: str = "CTX_CHUNG"
    ma_chuyen_gia: str = "EXP_CHUNG"
    ma_policy: str = "POL_CHUNG"
    ly_do_khong_hoc: bool = False
    context_audit: dict[str, Any] = field(default_factory=dict)
    allowed_action_mask: tuple[int, ...] = ()


class MockExpertPolicy:
    def __init__(self, name: str = "adversarial_mock") -> None:
        self.name = name
        self.input_size = 137
        self.output_size = 20
        self.experts: dict[str, Any] = {"CTX_CHUNG": 1}
        self.nhan_dien = lambda obs: "CTX_CHUNG"
        self.regime_probs: list[float] | None = None
        self.regime_name: str | None = None
        self.regime_idx: int | None = None

    def set_regime_soft_probabilities(
        self, soft_probs: Any, regime_name: str | None = None, regime_idx: int | None = None
    ) -> None:
        self.regime_probs = [float(p) for p in soft_probs]
        self.regime_name = regime_name
        self.regime_idx = regime_idx

    def act_nam_tac_nhan(self, observation: np.ndarray, goal: np.ndarray | None = None) -> tuple[SimpleTrace, ...]:
        traces = []
        for i in range(5):
            arr = np.zeros(20, dtype=float)
            arr[int(HOLD_ACTION)] = 1.0
            traces.append(SimpleTrace(
                trace_id=f"{self.name}_{i}",
                selected_action=arr,
                selected_idx=int(HOLD_ACTION),
                action_details={"five_agent_index": i, "five_action_index": int(HOLD_ACTION)},
            ))
        return tuple(traces)


@pytest.fixture
def evaluator_env() -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
    tmp_dir = tempfile.TemporaryDirectory()
    config = MT5BacktestConfig(
        runner_dir=Path(tmp_dir.name),
        regime_detector_enabled=True,
        offline_replay=True,
    )
    ev = MT5StrategyTesterEvaluator(config)
    return ev, tmp_dir


def make_five_traces(
    agent_actions: dict[int, int] | None = None,
    raw_logits_map: dict[int, np.ndarray] | None = None,
    owner_valid: bool = True,
) -> list[SimpleTrace]:
    actions = agent_actions or {}
    logits_map = raw_logits_map or {}
    traces = []
    for agent_idx in range(5):
        chosen_action = int(actions.get(agent_idx, int(HOLD_ACTION)))
        if agent_idx in logits_map:
            outputs = logits_map[agent_idx].copy()
        else:
            outputs = np.zeros(20, dtype=float)
            outputs[chosen_action] = 1.0
        tr = SimpleTrace(
            trace_id=f"trace_agent_{agent_idx}",
            selected_action=outputs,
            selected_idx=chosen_action,
            confidence=0.88,
            action_details={
                "five_agent_index": agent_idx,
                "five_action_index": chosen_action,
            },
            ma_context="CTX_CHUNG" if owner_valid else "",
            ma_chuyen_gia="EXP_CHUNG" if owner_valid else "",
            ma_policy="POL_CHUNG" if owner_valid else "",
            ly_do_khong_hoc=not owner_valid,
        )
        traces.append(tr)
    return traces


# ==============================================================================
# 1. BULL REGIME: AGENT 1 SHORT FORCING & DECODER CONFORMANCE
# ==============================================================================

class TestEmpiricalBullForcing:
    @pytest.mark.parametrize("short_act", [
        Action.MARKET_SELL,
        Action.SELL_LIMIT,
        Action.SELL_STOP,
    ])
    def test_agent1_short_veto_wire_and_decoder_verification(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], short_act: Action
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("bull_test")
        req = BridgeRequest(
            sequence=101,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.65, 0.20, 0.15),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces = make_five_traces({1: int(short_act)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        agent1_action = packet[1]
        agent1_trace = enriched[1]

        # Wire action must be HOLD
        assert agent1_action.action == int(Action.HOLD)
        # 12-action argmax must be HOLD
        assert np.argmax(agent1_action.outputs[:ACTION_COUNT]) == int(Action.HOLD)
        # Full 20-output argmax must be HOLD
        assert np.argmax(agent1_action.outputs) == int(Action.HOLD)

        # decode_outputs directly on the wire outputs
        decoded = decode_outputs(agent1_action.outputs, RiskConfig(), OrderConfig())
        assert decoded.action == Action.HOLD

        # Telemetry
        assert agent1_trace.action_details["regime_veto_applied"] is True
        assert agent1_trace.action_details["wire_action"] == int(Action.HOLD)
        assert agent1_trace.action_details["policy_action"] == int(short_act)


# ==============================================================================
# 2. BEAR REGIME: AGENT 0 LONG FORCING & DECODER CONFORMANCE
# ==============================================================================

class TestEmpiricalBearForcing:
    @pytest.mark.parametrize("long_act", [
        Action.MARKET_BUY,
        Action.BUY_LIMIT,
        Action.BUY_STOP,
    ])
    def test_agent0_long_veto_wire_and_decoder_verification(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], long_act: Action
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("bear_test")
        req = BridgeRequest(
            sequence=102,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.15, 0.70, 0.15),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces = make_five_traces({0: int(long_act)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        agent0_action = packet[0]
        agent0_trace = enriched[0]

        # Wire action must be HOLD
        assert agent0_action.action == int(Action.HOLD)
        # 12-action argmax must be HOLD
        assert np.argmax(agent0_action.outputs[:ACTION_COUNT]) == int(Action.HOLD)
        # Full 20-output argmax must be HOLD
        assert np.argmax(agent0_action.outputs) == int(Action.HOLD)

        # decode_outputs directly on wire outputs
        decoded = decode_outputs(agent0_action.outputs, RiskConfig(), OrderConfig())
        assert decoded.action == Action.HOLD

        # Telemetry
        assert agent0_trace.action_details["regime_veto_applied"] is True
        assert agent0_trace.action_details["wire_action"] == int(Action.HOLD)
        assert agent0_trace.action_details["policy_action"] == int(long_act)


# ==============================================================================
# 3. TREND-ALIGNED ACTIONS STRICTLY PRESERVED
# ==============================================================================

class TestEmpiricalTrendAlignedPreservation:
    @pytest.mark.parametrize("long_act", [
        Action.MARKET_BUY,
        Action.BUY_LIMIT,
        Action.BUY_STOP,
    ])
    def test_bull_trend_aligned_long_preserved(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], long_act: Action
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("bull_aligned")
        req = BridgeRequest(
            sequence=103,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.80, 0.10, 0.10),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces = make_five_traces({0: int(long_act)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        assert packet[0].action == int(long_act)
        assert np.argmax(packet[0].outputs[:ACTION_COUNT]) == int(long_act)
        assert enriched[0].action_details["regime_veto_applied"] is False
        assert enriched[0].action_details["action_masked"] is False

    @pytest.mark.parametrize("short_act", [
        Action.MARKET_SELL,
        Action.SELL_LIMIT,
        Action.SELL_STOP,
    ])
    def test_bear_trend_aligned_short_preserved(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], short_act: Action
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("bear_aligned")
        req = BridgeRequest(
            sequence=104,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.10, 0.80, 0.10),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces = make_five_traces({1: int(short_act)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        assert packet[1].action == int(short_act)
        assert np.argmax(packet[1].outputs[:ACTION_COUNT]) == int(short_act)
        assert enriched[1].action_details["regime_veto_applied"] is False
        assert enriched[1].action_details["action_masked"] is False


# ==============================================================================
# 4. EXACT BOUNDARY TRANSITIONS AT P = 0.5500
# ==============================================================================

class TestEmpiricalBoundaryTransitions:
    def test_bull_exact_boundary_thresholds(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("boundary")

        # 0.5499: no veto (margin = 0.1099 < 0.15)
        req_5499 = BridgeRequest(
            sequence=201,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5499, 0.44, 0.0101),
            positions=0,
        )
        p_5499, e_5499 = ev._five_bridge_action_packet(policy, req_5499, make_five_traces({1: int(Action.MARKET_SELL)}))
        assert p_5499[1].action == int(Action.MARKET_SELL)
        assert e_5499[1].action_details["regime_veto_applied"] is False

        # 0.5500: veto applied (P_Bull >= 0.55)
        req_5500 = BridgeRequest(
            sequence=202,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5500, 0.44, 0.0100),
            positions=0,
        )
        p_5500, e_5500 = ev._five_bridge_action_packet(policy, req_5500, make_five_traces({1: int(Action.MARKET_SELL)}))
        assert p_5500[1].action == int(Action.HOLD)
        assert e_5500[1].action_details["regime_veto_applied"] is True

        # 0.5501: veto applied
        req_5501 = BridgeRequest(
            sequence=203,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5501, 0.44, 0.0099),
            positions=0,
        )
        p_5501, e_5501 = ev._five_bridge_action_packet(policy, req_5501, make_five_traces({1: int(Action.MARKET_SELL)}))
        assert p_5501[1].action == int(Action.HOLD)
        assert e_5501[1].action_details["regime_veto_applied"] is True

    def test_bear_exact_boundary_thresholds(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("boundary")

        # 0.5499: no veto (margin = 0.1099 < 0.15)
        req_5499 = BridgeRequest(
            sequence=204,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5499, 0.0101),
            positions=0,
        )
        p_5499, e_5499 = ev._five_bridge_action_packet(policy, req_5499, make_five_traces({0: int(Action.MARKET_BUY)}))
        assert p_5499[0].action == int(Action.MARKET_BUY)
        assert e_5499[0].action_details["regime_veto_applied"] is False

        # 0.5500: veto applied (P_Bear >= 0.55)
        req_5500 = BridgeRequest(
            sequence=205,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5500, 0.0100),
            positions=0,
        )
        p_5500, e_5500 = ev._five_bridge_action_packet(policy, req_5500, make_five_traces({0: int(Action.MARKET_BUY)}))
        assert p_5500[0].action == int(Action.HOLD)
        assert e_5500[0].action_details["regime_veto_applied"] is True

        # 0.5501: veto applied
        req_5501 = BridgeRequest(
            sequence=206,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5501, 0.0099),
            positions=0,
        )
        p_5501, e_5501 = ev._five_bridge_action_packet(policy, req_5501, make_five_traces({0: int(Action.MARKET_BUY)}))
        assert p_5501[0].action == int(Action.HOLD)
        assert e_5501[0].action_details["regime_veto_applied"] is True


# ==============================================================================
# 5. CHOP REGIME & DEFENSIVE ROUTER INTEGRATION
# ==============================================================================

class TestEmpiricalChopRegimeIntegration:
    def test_chop_boundary_transition_and_defensive_routing(self) -> None:
        bull_mock = MockExpertPolicy("bull_specialist")
        bear_mock = MockExpertPolicy("bear_specialist")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            chop_threshold=0.40,
            confirmation_bars=1,
        )

        # 1. Establish Bull state first
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock

        # 2. P_Chop = 0.399999 in contest zone -> holds previous state (BULL)
        router.set_regime_soft_probabilities([0.31, 0.29, 0.399999])
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock

        # 3. P_Chop = 0.400000 -> switches immediately to CHOP (defensive Bear specialist)
        router.set_regime_soft_probabilities([0.30, 0.30, 0.400000], "CHOP", 2)
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0

    def test_chop_regime_with_runner_bridge_execution(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        bull_mock = MockExpertPolicy("bull")
        bear_mock = MockExpertPolicy("bear")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            chop_threshold=0.40,
            confirmation_bars=1,
        )

        req_chop = BridgeRequest(
            sequence=301,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.20, 0.20, 0.60),
            regime_name="CHOP",
            regime_index=2,
            positions=0,
        )
        traces = list(router.act_nam_tac_nhan(req_chop.inputs))
        packet, enriched = ev._five_bridge_action_packet(router, req_chop, traces)

        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock
        assert len(packet) == 5
        # In defensive mode, all 5 agents should have HOLD as action
        for act in packet:
            assert act.action == int(Action.HOLD)


# ==============================================================================
# 6. EXTREME ADVERSARIAL STRESS & LOGIT FORCING
# ==============================================================================

class TestEmpiricalExtremeAdversarialStress:
    def test_extreme_negative_hold_and_extreme_positive_short(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("extreme_stress")
        req = BridgeRequest(
            sequence=401,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.95, 0.03, 0.02),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )

        extreme_logits = np.zeros(20, dtype=float)
        extreme_logits[int(Action.MARKET_SELL)] = 1e9
        extreme_logits[int(Action.HOLD)] = -1e9
        extreme_logits[12:] = 1e3

        traces = make_five_traces(
            agent_actions={1: int(Action.MARKET_SELL)},
            raw_logits_map={1: extreme_logits},
        )
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        agent1_action = packet[1]
        assert agent1_action.action == int(Action.HOLD)
        # Wire output for HOLD must exceed 1e9
        assert agent1_action.outputs[0] > 1e9
        assert np.argmax(agent1_action.outputs[:ACTION_COUNT]) == 0
        assert np.argmax(agent1_action.outputs) == 0

    def test_random_fuzzing_500_regime_vectors(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("fuzz")
        rng = np.random.RandomState(42)

        for seq in range(500):
            raw = rng.exponential(scale=1.0, size=3)
            p = raw / np.sum(raw)
            pbull, pbear, pchop = float(p[0]), float(p[1]), float(p[2])

            req = BridgeRequest(
                sequence=1000 + seq,
                inputs=np.zeros(137, dtype=float),
                regime_probabilities=(pbull, pbear, pchop),
                positions=0,
            )

            # Test Agent 1 Short
            traces_a1 = make_five_traces({1: int(Action.MARKET_SELL)})
            packet_a1, enriched_a1 = ev._five_bridge_action_packet(policy, req, traces_a1)
            is_bull_veto = (pbull >= 0.55) or (pbull > pbear + 0.15 and pbull > 0.45)
            if is_bull_veto:
                assert packet_a1[1].action == int(Action.HOLD), f"Agent 1 failed veto at {pbull, pbear, pchop}"
                assert np.argmax(packet_a1[1].outputs[:ACTION_COUNT]) == int(Action.HOLD)
                assert enriched_a1[1].action_details["regime_veto_applied"] is True
            else:
                assert packet_a1[1].action == int(Action.MARKET_SELL), f"Agent 1 false veto at {pbull, pbear, pchop}"
                assert enriched_a1[1].action_details["regime_veto_applied"] is False

            # Test Agent 0 Long
            traces_a0 = make_five_traces({0: int(Action.MARKET_BUY)})
            packet_a0, enriched_a0 = ev._five_bridge_action_packet(policy, req, traces_a0)
            is_bear_veto = (pbear >= 0.55) or (pbear > pbull + 0.15 and pbear > 0.45)
            if is_bear_veto:
                assert packet_a0[0].action == int(Action.HOLD), f"Agent 0 failed veto at {pbull, pbear, pchop}"
                assert np.argmax(packet_a0[0].outputs[:ACTION_COUNT]) == int(Action.HOLD)
                assert enriched_a0[0].action_details["regime_veto_applied"] is True
            else:
                assert packet_a0[0].action == int(Action.MARKET_BUY), f"Agent 0 false veto at {pbull, pbear, pchop}"
                assert enriched_a0[0].action_details["regime_veto_applied"] is False
