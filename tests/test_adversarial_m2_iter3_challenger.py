"""
tests/test_adversarial_m2_iter3_challenger.py

Milestone 2 Iteration 3 Empirical Adversarial Challenge Suite:
Exhaustive stress verification of:
1. Wire output tensor forcing under extreme logit dominance (+1e9 vs -1e9)
2. Exact boundary transitions at P = 0.5500 (Bull/Bear) and P = 0.4000 (Chop)
3. Immunity against false directional soft vetoes from negative probabilities & malformed types
4. Strict non-interference and preservation of trend-aligned entries
5. Multi-agent isolation (Agents 2, 3, and 4 Governor non-interference)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    MarketRegime,
)
from self_growing_trader.dual_champion_router import DualChampionRouter
from self_growing_trader.five_agent_authority import HOLD_ACTION


@dataclass
class SimpleTrace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))
    selected_idx: int = int(HOLD_ACTION)
    confidence: float = 0.88
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
    def __init__(self, name: str = "adversarial_mock_iter3") -> None:
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
        regime_detector_enabled=False,  # Isolate direct probability feeding
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
# 1. EXTREME LOGIT DOMINANCE (+1e9 vs -1e9) WIRE FORCING
# ==============================================================================

class TestAdversarialExtremeLogitDominance:
    """Verifies that genuine soft veto strictly overrides extreme logit dominance (+1e9 on entry vs -1e9 on HOLD)."""

    @pytest.mark.parametrize("short_act", [
        Action.MARKET_SELL,
        Action.SELL_LIMIT,
        Action.SELL_STOP,
    ])
    def test_bull_extreme_short_logit_forced_to_hold(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], short_act: Action
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("extreme_bull")
        req = BridgeRequest(
            sequence=1001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.85, 0.10, 0.05),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )

        extreme_logits = np.zeros(20, dtype=float)
        extreme_logits[int(short_act)] = 1e9
        extreme_logits[int(Action.HOLD)] = -1e9
        extreme_logits[12:] = 500.0  # Execution geometry parameter logits

        traces = make_five_traces(
            agent_actions={1: int(short_act)},
            raw_logits_map={1: extreme_logits},
        )
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        agent1_action = packet[1]
        agent1_trace = enriched[1]

        # Invariants:
        assert agent1_action.action == int(Action.HOLD), "Action integer must be strictly HOLD (0)"
        assert np.argmax(agent1_action.outputs[:ACTION_COUNT]) == int(Action.HOLD), "12-action argmax must be 0"
        assert np.argmax(agent1_action.outputs) == int(Action.HOLD), "20-output argmax must be 0"
        assert agent1_action.outputs[0] > 1e9, f"HOLD output must dominate: {agent1_action.outputs[0]}"

        # MT5 decoder invariance:
        decoded = decode_outputs(agent1_action.outputs, RiskConfig(), OrderConfig())
        assert decoded.action == Action.HOLD, "decode_outputs must produce Action.HOLD"

        # Telemetry:
        assert agent1_trace.action_details["regime_veto_applied"] is True
        assert agent1_trace.action_details["wire_action"] == int(Action.HOLD)
        assert agent1_trace.action_details["policy_action"] == int(short_act)

    @pytest.mark.parametrize("long_act", [
        Action.MARKET_BUY,
        Action.BUY_LIMIT,
        Action.BUY_STOP,
    ])
    def test_bear_extreme_long_logit_forced_to_hold(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], long_act: Action
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("extreme_bear")
        req = BridgeRequest(
            sequence=1002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.05, 0.85, 0.10),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )

        extreme_logits = np.zeros(20, dtype=float)
        extreme_logits[int(long_act)] = 1e9
        extreme_logits[int(Action.HOLD)] = -1e9
        extreme_logits[12:] = 500.0

        traces = make_five_traces(
            agent_actions={0: int(long_act)},
            raw_logits_map={0: extreme_logits},
        )
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        agent0_action = packet[0]
        agent0_trace = enriched[0]

        # Invariants:
        assert agent0_action.action == int(Action.HOLD)
        assert np.argmax(agent0_action.outputs[:ACTION_COUNT]) == int(Action.HOLD)
        assert np.argmax(agent0_action.outputs) == int(Action.HOLD)
        assert agent0_action.outputs[0] > 1e9

        # MT5 decoder invariance:
        decoded = decode_outputs(agent0_action.outputs, RiskConfig(), OrderConfig())
        assert decoded.action == Action.HOLD

        # Telemetry:
        assert agent0_trace.action_details["regime_veto_applied"] is True
        assert agent0_trace.action_details["wire_action"] == int(Action.HOLD)
        assert agent0_trace.action_details["policy_action"] == int(long_act)


# ==============================================================================
# 2. EXACT BOUNDARY STRESS AT P = 0.5500 AND P = 0.4000
# ==============================================================================

class TestAdversarialExactBoundaries:
    """Verifies ultra-precise boundary transitions at P = 0.5500 (Bull/Bear) and P = 0.4000 (Chop)."""

    def test_bull_exact_threshold_transitions(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("boundary_bull")

        # 0.5499999 (sub-threshold, secondary = 0.44 -> delta = 0.1099999 < 0.15) -> No veto
        req_sub = BridgeRequest(
            sequence=2001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5499999, 0.44, 0.0100001),
            positions=0,
        )
        pkt_sub, tr_sub = ev._five_bridge_action_packet(policy, req_sub, make_five_traces({1: int(Action.MARKET_SELL)}))
        assert pkt_sub[1].action == int(Action.MARKET_SELL)
        assert tr_sub[1].action_details["regime_veto_applied"] is False

        # 0.5500000 exactly -> Veto applied
        req_exact = BridgeRequest(
            sequence=2002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5500000, 0.44, 0.0100000),
            positions=0,
        )
        pkt_exact, tr_exact = ev._five_bridge_action_packet(policy, req_exact, make_five_traces({1: int(Action.MARKET_SELL)}))
        assert pkt_exact[1].action == int(Action.HOLD)
        assert tr_exact[1].action_details["regime_veto_applied"] is True
        assert np.argmax(pkt_exact[1].outputs[:ACTION_COUNT]) == int(Action.HOLD)

        # 0.5500001 -> Veto applied
        req_sup = BridgeRequest(
            sequence=2003,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5500001, 0.44, 0.0099999),
            positions=0,
        )
        pkt_sup, tr_sup = ev._five_bridge_action_packet(policy, req_sup, make_five_traces({1: int(Action.MARKET_SELL)}))
        assert pkt_sup[1].action == int(Action.HOLD)
        assert tr_sup[1].action_details["regime_veto_applied"] is True

    def test_bear_exact_threshold_transitions(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("boundary_bear")

        # 0.5499999 (sub-threshold, secondary = 0.44 -> delta = 0.1099999 < 0.15) -> No veto
        req_sub = BridgeRequest(
            sequence=2004,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5499999, 0.0100001),
            positions=0,
        )
        pkt_sub, tr_sub = ev._five_bridge_action_packet(policy, req_sub, make_five_traces({0: int(Action.MARKET_BUY)}))
        assert pkt_sub[0].action == int(Action.MARKET_BUY)
        assert tr_sub[0].action_details["regime_veto_applied"] is False

        # 0.5500000 exactly -> Veto applied
        req_exact = BridgeRequest(
            sequence=2005,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5500000, 0.0100000),
            positions=0,
        )
        pkt_exact, tr_exact = ev._five_bridge_action_packet(policy, req_exact, make_five_traces({0: int(Action.MARKET_BUY)}))
        assert pkt_exact[0].action == int(Action.HOLD)
        assert tr_exact[0].action_details["regime_veto_applied"] is True
        assert np.argmax(pkt_exact[0].outputs[:ACTION_COUNT]) == int(Action.HOLD)

        # 0.5500001 -> Veto applied
        req_sup = BridgeRequest(
            sequence=2006,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5500001, 0.0099999),
            positions=0,
        )
        pkt_sup, tr_sup = ev._five_bridge_action_packet(policy, req_sup, make_five_traces({0: int(Action.MARKET_BUY)}))
        assert pkt_sup[0].action == int(Action.HOLD)
        assert tr_sup[0].action_details["regime_veto_applied"] is True

    def test_chop_exact_boundary_in_router(self) -> None:
        bull_mock = MockExpertPolicy("router_bull")
        bear_mock = MockExpertPolicy("router_bear")
        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            chop_threshold=0.40,
            confirmation_bars=1,
        )

        # Set initial Bull state
        router.set_regime_soft_probabilities([0.70, 0.15, 0.15], "BULL", 0)
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock

        # P_Chop = 0.3999999 in contest zone -> holds previous state (BULL)
        router.set_regime_soft_probabilities([0.3000001, 0.2999999, 0.3999999])
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_mock

        # P_Chop = 0.4000000 exactly -> switches to CHOP, activates defensive Bear specialist
        router.set_regime_soft_probabilities([0.30, 0.30, 0.4000000], "CHOP", 2)
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0

        # P_Chop = 0.4000001 -> remains in CHOP
        router.set_regime_soft_probabilities([0.2999999, 0.30, 0.4000001], "CHOP", 2)
        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_mock


# ==============================================================================
# 3. NEGATIVE PROBABILITIES & MALFORMED TYPES IMMUNITY
# ==============================================================================

class TestAdversarialNegativeAndMalformedImmunity:
    """Confirms that negative probabilities and malformed types CANNOT trigger false directional soft vetoes."""

    @pytest.mark.parametrize(
        "negative_bull_candidate,desc",
        [
            ((1.5, -0.7, 0.2), "p_bull=1.5 with negative p_bear, sum=1.0"),
            ((0.9, -0.2, 0.3), "p_bull=0.9 with negative p_bear, sum=1.0"),
            ((0.6, -0.1, 0.5), "p_bull=0.6 with negative p_bear, sum=1.0"),
        ],
    )
    def test_negative_probability_cannot_trigger_bull_veto_against_short(
        self,
        evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        negative_bull_candidate: tuple[float, float, float],
        desc: str,
    ) -> None:
        """Vectors with negative components violate axioms and must fail closed without vetoing Agent 1 Short."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("negative_immunity_test")
        req = BridgeRequest(
            sequence=3001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=negative_bull_candidate,
            positions=0,
        )
        traces = make_five_traces({1: int(Action.MARKET_SELL)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Agent 1 Short MUST NOT be vetoed by mathematically corrupt input!
        assert enriched[1].action_details["regime_veto_applied"] is False, (
            f"VULNERABILITY: False veto triggered by corrupt vector in {desc}"
        )
        assert packet[1].action == int(Action.MARKET_SELL), f"Action altered in {desc}"
        assert enriched[0].action_details["regime_index"] == int(MarketRegime.CHOP)
        assert np.allclose(enriched[0].action_details["regime_probabilities"], [1/3, 1/3, 1/3])

    @pytest.mark.parametrize(
        "negative_bear_candidate,desc",
        [
            ((-0.7, 1.5, 0.2), "p_bear=1.5 with negative p_bull, sum=1.0"),
            ((-0.2, 0.9, 0.3), "p_bear=0.9 with negative p_bull, sum=1.0"),
            ((-0.1, 0.6, 0.5), "p_bear=0.6 with negative p_bull, sum=1.0"),
        ],
    )
    def test_negative_probability_cannot_trigger_bear_veto_against_long(
        self,
        evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        negative_bear_candidate: tuple[float, float, float],
        desc: str,
    ) -> None:
        """Vectors with negative components violate axioms and must fail closed without vetoing Agent 0 Long."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("negative_bear_immunity_test")
        req = BridgeRequest(
            sequence=3002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=negative_bear_candidate,
            positions=0,
        )
        traces = make_five_traces({0: int(Action.MARKET_BUY)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Agent 0 Long MUST NOT be vetoed by mathematically corrupt input!
        assert enriched[0].action_details["regime_veto_applied"] is False, (
            f"VULNERABILITY: False veto triggered by corrupt vector in {desc}"
        )
        assert packet[0].action == int(Action.MARKET_BUY), f"Action altered in {desc}"
        assert enriched[0].action_details["regime_index"] == int(MarketRegime.CHOP)
        assert np.allclose(enriched[0].action_details["regime_probabilities"], [1/3, 1/3, 1/3])

    @pytest.mark.parametrize(
        "malformed_type_input,desc",
        [
            (("a", "b", "c"), "Non-numeric string tuple"),
            ((1.0 + 0.5j, 0.0, 0.0), "Complex numbers"),
            ({"bull": 0.9, "bear": 0.05, "chop": 0.05}, "Dictionary mapping"),
            ([[0.9], [0.05, 0.05]], "Ragged nested list"),
            (b"\x00\x01\x02", "Raw bytes"),
        ],
    )
    def test_malformed_container_types_fail_closed_without_false_veto(
        self,
        evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        malformed_type_input: Any,
        desc: str,
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("malformed_type_test")
        req = BridgeRequest(
            sequence=3003,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=malformed_type_input,
            positions=0,
        )
        # Test Agent 0 BUY with Agent 1 HOLD
        traces_a0 = make_five_traces({0: int(Action.MARKET_BUY), 1: int(Action.HOLD)})
        packet_a0, enriched_a0 = ev._five_bridge_action_packet(policy, req, traces_a0)
        assert enriched_a0[0].action_details["regime_index"] == int(MarketRegime.CHOP)
        assert np.allclose(enriched_a0[0].action_details["regime_probabilities"], [1/3, 1/3, 1/3])
        assert enriched_a0[0].action_details["regime_veto_applied"] is False
        assert packet_a0[0].action == int(Action.MARKET_BUY)

        # Test Agent 1 SELL with Agent 0 HOLD
        traces_a1 = make_five_traces({0: int(Action.HOLD), 1: int(Action.MARKET_SELL)})
        packet_a1, enriched_a1 = ev._five_bridge_action_packet(policy, req, traces_a1)
        assert enriched_a1[1].action_details["regime_index"] == int(MarketRegime.CHOP)
        assert np.allclose(enriched_a1[1].action_details["regime_probabilities"], [1/3, 1/3, 1/3])
        assert enriched_a1[1].action_details["regime_veto_applied"] is False
        assert packet_a1[1].action == int(Action.MARKET_SELL)


# ==============================================================================
# 4. TREND-ALIGNED ENTRY PRESERVATION & AGENT ISOLATION
# ==============================================================================

class TestAdversarialTrendAlignedPreservationAndIsolation:
    """Verifies that trend-aligned actions and non-target agents (Agents 2, 3, 4 Governor) are strictly preserved."""

    @pytest.mark.parametrize("action_idx", [
        Action.MARKET_BUY,
        Action.BUY_LIMIT,
        Action.BUY_STOP,
    ])
    def test_bull_trend_aligned_entries_preserved_intact(
        self,
        evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        action_idx: Action,
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("trend_aligned_bull")
        req = BridgeRequest(
            sequence=4001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.90, 0.05, 0.05),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces = make_five_traces({0: int(action_idx)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Agent 0 MARKET_BUY/BUY_LIMIT/BUY_STOP must remain 100% intact
        assert packet[0].action == int(action_idx)
        assert np.argmax(packet[0].outputs[:ACTION_COUNT]) == int(action_idx)
        assert enriched[0].action_details["regime_veto_applied"] is False
        assert enriched[0].action_details["action_masked"] is False
        assert enriched[0].action_details["wire_action"] == int(action_idx)

    @pytest.mark.parametrize("action_idx", [
        Action.MARKET_SELL,
        Action.SELL_LIMIT,
        Action.SELL_STOP,
    ])
    def test_bear_trend_aligned_entries_preserved_intact(
        self,
        evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        action_idx: Action,
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("trend_aligned_bear")
        req = BridgeRequest(
            sequence=4002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.05, 0.90, 0.05),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces = make_five_traces({1: int(action_idx)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Agent 1 MARKET_SELL/SELL_LIMIT/SELL_STOP must remain 100% intact
        assert packet[1].action == int(action_idx)
        assert np.argmax(packet[1].outputs[:ACTION_COUNT]) == int(action_idx)
        assert enriched[1].action_details["regime_veto_applied"] is False
        assert enriched[1].action_details["action_masked"] is False
        assert enriched[1].action_details["wire_action"] == int(action_idx)

    def test_governor_and_exit_agents_immunity_across_all_regimes(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        ev, _ = evaluator_env
        policy = MockExpertPolicy("governor_isolation")

        regimes = [
            ((0.85, 0.10, 0.05), "BULL", 0),
            ((0.10, 0.85, 0.05), "BEAR", 1),
            ((0.20, 0.20, 0.60), "CHOP", 2),
            ((0.50, 0.50, 0.00), "CONTEST", 2),
        ]

        for seq, (probs, name, idx) in enumerate(regimes, start=5000):
            req = BridgeRequest(
                sequence=seq,
                inputs=np.zeros(137, dtype=float),
                regime_probabilities=probs,
                regime_name=name,
                regime_index=idx,
                positions=0,
            )
            traces = make_five_traces({
                2: int(Action.HOLD),
                3: int(Action.HOLD),
                4: int(Action.HOLD),
            })
            packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

            # Agents 2, 3, 4 are never subject to regime soft veto
            for a_idx in (2, 3, 4):
                assert packet[a_idx].action == int(Action.HOLD), f"Agent {a_idx} action != HOLD in {name}"
                assert enriched[a_idx].action_details["regime_veto_applied"] is False, (
                    f"Agent {a_idx} falsely vetoed in {name}"
                )
                assert enriched[a_idx].action_details["wire_action"] == int(Action.HOLD)

            # Governor lock invariant: Agent 4 NEVER emits POLICY_CLOSE_ALL (Action 10)
            assert packet[4].action != int(Action.CLOSE_ALL)
            assert packet[4].action == int(Action.HOLD)
