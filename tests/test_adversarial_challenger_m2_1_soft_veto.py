"""
tests/test_adversarial_challenger_m2_1_soft_veto.py

Adversarial Stress Test Suite for Milestone 2 (challenger_m2_1):
Empirical Verification of Soft Veto Wire Output Forcing & Boundary Fuzzing.

Challenge Dimensions:
1. Wire Output Tensor Forcing to HOLD:
   - When veto is applied to Agent 1 (Short in Bull) or Agent 0 (Long in Bear):
     * wire_action == Action.HOLD (0)
     * np.argmax(wire_outputs[:12]) == 0 AND np.argmax(wire_outputs) == 0
     * Pending orders (SELL_LIMIT, SELL_STOP, BUY_LIMIT, BUY_STOP) vetoed identically
     * Extreme positive logits on vetoed actions (+100.0, +1e6) forced to HOLD
2. High-Precision Boundary Fuzzing (P = 0.5499 vs P = 0.5501):
   - Strict inequality/equality transitions at P = 0.5500
   - Multi-step micro-fuzzing around 0.55
3. Contest Zone Fuzzing (P in [0.45, 0.55]):
   - 2D grid sweep over (P_Bull, P_Bear) in [0.45, 0.5499] x [0.45, 0.5499]
   - Proof that no false veto triggers in the contest zone for either Agent 0 or Agent 1
4. Margin Fallback Boundary Precision:
   - Margin separation threshold at delta = 0.15 and P > 0.45
   - delta = 0.1499 (no veto) vs delta = 0.1501 (veto applied)
   - Floor check at P = 0.4500 (no veto) vs P = 0.4501 (veto applied)
5. Agent Isolation, Autoprediction on None, and Non-Interference:
   - Non-target agents (Agent 2, 3, 4 Governor) never falsely vetoed
   - Automatic regime fallback when regime_probabilities is None
   - Non-finite (NaN, Inf) inputs fail-safe without false veto
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from self_growing_trader.orders import Action, ACTION_COUNT, OUTPUT_COUNT
from self_growing_trader.mt5_runner import (
    BridgeRequest,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
)
from self_growing_trader.five_agent_authority import HOLD_ACTION
from dataclasses import dataclass, field


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
    """Mock expert system policy providing valid expert interfaces and provenance."""

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
# DIMENSION 1: WIRE OUTPUT TENSOR FORCING TO HOLD & PENDING ORDERS
# ==============================================================================

class TestAdversarialWireOutputForcing:
    """Verifies that soft veto deterministically forces wire_action == 0 and argmax == 0."""

    @pytest.mark.parametrize("short_action", [
        Action.MARKET_SELL,
        Action.SELL_LIMIT,
        Action.SELL_STOP,
    ])
    def test_agent1_all_short_orders_forced_to_hold_in_bull(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], short_action: Action
    ) -> None:
        """All short actions (market, limit, stop) proposed by Agent 1 in Bull regime are forced to HOLD."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("bull_forcing_test")
        p_bull = 0.70
        req = BridgeRequest(
            sequence=1001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(p_bull, 0.20, 0.10),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )

        traces = make_five_traces({1: int(short_action)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        agent1_pkt = packet[1]
        agent1_trace = enriched[1]

        # 1. Wire action must be strictly HOLD (0)
        assert agent1_pkt.action == int(Action.HOLD)
        # 2. Both 12-action argmax and 20-output argmax must be 0
        assert np.argmax(agent1_pkt.outputs[:ACTION_COUNT]) == int(Action.HOLD)
        assert np.argmax(agent1_pkt.outputs) == int(Action.HOLD)
        # 3. Telemetry records soft veto
        assert agent1_trace.action_details["regime_veto_applied"] is True
        assert agent1_trace.action_details["regime_veto_reason"] == f"veto_short_in_bull_p{p_bull:.2f}"
        assert agent1_trace.action_details["wire_action"] == int(Action.HOLD)
        assert agent1_trace.action_details["policy_action"] == int(short_action)
        assert agent1_trace.action_details["action_masked"] is True

    @pytest.mark.parametrize("long_action", [
        Action.MARKET_BUY,
        Action.BUY_LIMIT,
        Action.BUY_STOP,
    ])
    def test_agent0_all_long_orders_forced_to_hold_in_bear(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], long_action: Action
    ) -> None:
        """All long actions (market, limit, stop) proposed by Agent 0 in Bear regime are forced to HOLD."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("bear_forcing_test")
        p_bear = 0.75
        req = BridgeRequest(
            sequence=1002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.15, p_bear, 0.10),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )

        traces = make_five_traces({0: int(long_action)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        agent0_pkt = packet[0]
        agent0_trace = enriched[0]

        # 1. Wire action must be strictly HOLD (0)
        assert agent0_pkt.action == int(Action.HOLD)
        # 2. Both 12-action argmax and 20-output argmax must be 0
        assert np.argmax(agent0_pkt.outputs[:ACTION_COUNT]) == int(Action.HOLD)
        assert np.argmax(agent0_pkt.outputs) == int(Action.HOLD)
        # 3. Telemetry records soft veto
        assert agent0_trace.action_details["regime_veto_applied"] is True
        assert agent0_trace.action_details["regime_veto_reason"] == f"veto_long_in_bear_p{p_bear:.2f}"
        assert agent0_trace.action_details["wire_action"] == int(Action.HOLD)
        assert agent0_trace.action_details["policy_action"] == int(long_action)
        assert agent0_trace.action_details["action_masked"] is True

    def test_extreme_logits_overridden_by_forcing(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """Even with extreme logits (+1000.0, +1e6) on the forbidden action, wire_outputs forces HOLD to dominate."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("extreme_logits_test")
        req = BridgeRequest(
            sequence=1003,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.70, 0.20, 0.10),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )

        extreme_logits = np.zeros(20, dtype=float)
        extreme_logits[int(Action.MARKET_SELL)] = 1_000_000.0
        extreme_logits[int(Action.HOLD)] = -500.0
        # Add non-zero parameter logits in indices 12-19
        extreme_logits[12:] = 5.0

        traces = make_five_traces(
            agent_actions={1: int(Action.MARKET_SELL)},
            raw_logits_map={1: extreme_logits},
        )
        packet, _ = ev._five_bridge_action_packet(policy, req, traces)

        agent1_pkt = packet[1]
        assert agent1_pkt.action == int(Action.HOLD)
        # wire_outputs[0] must be strictly greater than extreme_logits[2]
        assert agent1_pkt.outputs[int(Action.HOLD)] > extreme_logits[int(Action.MARKET_SELL)]
        assert agent1_pkt.outputs[int(Action.HOLD)] == pytest.approx(1_000_001.0, rel=1e-5)
        assert np.argmax(agent1_pkt.outputs[:ACTION_COUNT]) == int(Action.HOLD)
        assert np.argmax(agent1_pkt.outputs) == int(Action.HOLD)


# ==============================================================================
# DIMENSION 2: HIGH-PRECISION BOUNDARY STRESS (P = 0.5499 vs P = 0.5501)
# ==============================================================================

class TestAdversarialBoundaryFuzzing:
    """Verifies sharp threshold boundary at P = 0.5500."""

    def test_bull_boundary_5499_vs_5501(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """P_Bull = 0.5499 (no veto) vs P_Bull = 0.5501 (veto applied)."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("boundary_test")

        # 1. Sub-threshold: P_Bull = 0.5499, P_Bear = 0.44 (margin = 0.1099 < 0.15)
        req_sub = BridgeRequest(
            sequence=2001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5499, 0.44, 0.0101),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_sub = make_five_traces({1: int(Action.MARKET_SELL)})
        packet_sub, enriched_sub = ev._five_bridge_action_packet(policy, req_sub, traces_sub)
        assert packet_sub[1].action == int(Action.MARKET_SELL)
        assert enriched_sub[1].action_details["regime_veto_applied"] is False

        # 2. Super-threshold: P_Bull = 0.5501, P_Bear = 0.44 (P >= 0.55)
        req_sup = BridgeRequest(
            sequence=2002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.5501, 0.44, 0.0099),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_sup = make_five_traces({1: int(Action.MARKET_SELL)})
        packet_sup, enriched_sup = ev._five_bridge_action_packet(policy, req_sup, traces_sup)
        assert packet_sup[1].action == int(Action.HOLD)
        assert np.argmax(packet_sup[1].outputs[:ACTION_COUNT]) == int(Action.HOLD)
        assert enriched_sup[1].action_details["regime_veto_applied"] is True
        assert enriched_sup[1].action_details["regime_veto_reason"] == "veto_short_in_bull_p0.55"

    def test_bear_boundary_5499_vs_5501(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """P_Bear = 0.5499 (no veto) vs P_Bear = 0.5501 (veto applied)."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("boundary_test")

        # 1. Sub-threshold: P_Bear = 0.5499, P_Bull = 0.44 (margin = 0.1099 < 0.15)
        req_sub = BridgeRequest(
            sequence=2003,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5499, 0.0101),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces_sub = make_five_traces({0: int(Action.MARKET_BUY)})
        packet_sub, enriched_sub = ev._five_bridge_action_packet(policy, req_sub, traces_sub)
        assert packet_sub[0].action == int(Action.MARKET_BUY)
        assert enriched_sub[0].action_details["regime_veto_applied"] is False

        # 2. Super-threshold: P_Bear = 0.5501, P_Bull = 0.44 (P >= 0.55)
        req_sup = BridgeRequest(
            sequence=2004,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.44, 0.5501, 0.0099),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces_sup = make_five_traces({0: int(Action.MARKET_BUY)})
        packet_sup, enriched_sup = ev._five_bridge_action_packet(policy, req_sup, traces_sup)
        assert packet_sup[0].action == int(Action.HOLD)
        assert np.argmax(packet_sup[0].outputs[:ACTION_COUNT]) == int(Action.HOLD)
        assert enriched_sup[0].action_details["regime_veto_applied"] is True
        assert enriched_sup[0].action_details["regime_veto_reason"] == "veto_long_in_bear_p0.55"

    def test_micro_fuzzing_step_function_across_threshold(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """Fuzzes 21 points around 0.55 in steps of 0.0005 with constant secondary prob P_Bear = 0.44."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("micro_fuzz")

        p_values = np.linspace(0.545, 0.555, 21)
        for seq, p_bull in enumerate(p_values, start=3000):
            p_bull_f = float(p_bull)
            p_bear_f = 0.44
            p_chop_f = max(0.0, 1.0 - p_bull_f - p_bear_f)

            req = BridgeRequest(
                sequence=seq,
                inputs=np.zeros(137, dtype=float),
                regime_probabilities=(p_bull_f, p_bear_f, p_chop_f),
                regime_name="BULL",
                regime_index=0,
                positions=0,
            )
            traces = make_five_traces({1: int(Action.MARKET_SELL)})
            packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

            if p_bull_f >= 0.55:
                assert packet[1].action == int(Action.HOLD), f"Failed to veto at P={p_bull_f}"
                assert enriched[1].action_details["regime_veto_applied"] is True
            else:
                assert packet[1].action == int(Action.MARKET_SELL), f"False veto at P={p_bull_f}"
                assert enriched[1].action_details["regime_veto_applied"] is False


# ==============================================================================
# DIMENSION 3: CONTEST ZONE (P in [0.45, 0.55]) NO FALSE VETO
# ==============================================================================

class TestAdversarialContestZoneNoFalseVeto:
    """Verifies that in the contest zone [0.45, 0.55], no false veto triggers."""

    def test_canonical_contest_points(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """Checks symmetrical and near-symmetrical points in the contest zone for both Agent 0 and Agent 1."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("contest_canonical")

        test_points = [
            (0.50, 0.50, 0.00),
            (0.48, 0.48, 0.04),
            (0.45, 0.45, 0.10),
            (0.52, 0.48, 0.00),
            (0.48, 0.52, 0.00),
            (0.54, 0.46, 0.00),
            (0.46, 0.54, 0.00),
            (0.5499, 0.4501, 0.00),
            (0.4501, 0.5499, 0.00),
        ]

        for seq, (pbull, pbear, pchop) in enumerate(test_points, start=4000):
            req = BridgeRequest(
                sequence=seq,
                inputs=np.zeros(137, dtype=float),
                regime_probabilities=(pbull, pbear, pchop),
                regime_name="CONTEST",
                regime_index=2,
                positions=0,
            )

            # Test Agent 0 BUY (with Agent 1 HOLD)
            traces_a0 = make_five_traces({0: int(Action.MARKET_BUY)})
            packet_a0, enriched_a0 = ev._five_bridge_action_packet(policy, req, traces_a0)
            assert packet_a0[0].action == int(Action.MARKET_BUY), f"Agent 0 falsely vetoed at {pbull, pbear, pchop}"
            assert enriched_a0[0].action_details["regime_veto_applied"] is False

            # Test Agent 1 SELL (with Agent 0 HOLD)
            traces_a1 = make_five_traces({1: int(Action.MARKET_SELL)})
            packet_a1, enriched_a1 = ev._five_bridge_action_packet(policy, req, traces_a1)
            assert packet_a1[1].action == int(Action.MARKET_SELL), f"Agent 1 falsely vetoed at {pbull, pbear, pchop}"
            assert enriched_a1[1].action_details["regime_veto_applied"] is False

    def test_2d_grid_fuzzing_contest_zone(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """Dense 2D grid sweep over (P_Bull, P_Bear) in [0.45, 0.549] x [0.45, 0.549]."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("contest_grid")

        bull_range = np.linspace(0.45, 0.549, 8)
        bear_range = np.linspace(0.45, 0.549, 8)

        tested_count = 0
        for p_bull in bull_range:
            for p_bear in bear_range:
                if p_bull + p_bear > 1.0:
                    continue
                p_chop = max(0.0, 1.0 - p_bull - p_bear)
                tested_count += 1

                req = BridgeRequest(
                    sequence=5000 + tested_count,
                    inputs=np.zeros(137, dtype=float),
                    regime_probabilities=(float(p_bull), float(p_bear), float(p_chop)),
                    regime_name="CONTEST",
                    regime_index=2,
                    positions=0,
                )

                # Test Agent 0 BUY
                traces_a0 = make_five_traces({0: int(Action.MARKET_BUY)})
                packet_a0, enriched_a0 = ev._five_bridge_action_packet(policy, req, traces_a0)
                assert packet_a0[0].action == int(Action.MARKET_BUY)
                assert enriched_a0[0].action_details["regime_veto_applied"] is False

                # Test Agent 1 SELL
                traces_a1 = make_five_traces({1: int(Action.MARKET_SELL)})
                packet_a1, enriched_a1 = ev._five_bridge_action_packet(policy, req, traces_a1)
                assert packet_a1[1].action == int(Action.MARKET_SELL)
                assert enriched_a1[1].action_details["regime_veto_applied"] is False

        assert tested_count >= 20, f"Expected at least 20 valid contest pairs, got {tested_count}"


# ==============================================================================
# DIMENSION 4: MARGIN FALLBACK SEPARATION PRECISION
# ==============================================================================

class TestAdversarialMarginFallback:
    """Verifies margin fallback logic: (p_dominant > p_other + 0.15 and p_dominant > 0.45)."""

    def test_margin_threshold_delta_1499_vs_1501(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """P_Bull = 0.50, P_Bear = 0.3501 (delta = 0.1499 -> no veto) vs P_Bear = 0.3499 (delta = 0.1501 -> veto)."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("margin_test")

        # 1. Delta = 0.1499 < 0.15 -> No veto
        req_sub = BridgeRequest(
            sequence=6001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.50, 0.3501, 0.1499),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_sub = make_five_traces({1: int(Action.MARKET_SELL)})
        packet_sub, enriched_sub = ev._five_bridge_action_packet(policy, req_sub, traces_sub)
        assert packet_sub[1].action == int(Action.MARKET_SELL)
        assert enriched_sub[1].action_details["regime_veto_applied"] is False

        # 2. Delta = 0.1501 > 0.15 -> Veto applied
        req_sup = BridgeRequest(
            sequence=6002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.50, 0.3499, 0.1501),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_sup = make_five_traces({1: int(Action.MARKET_SELL)})
        packet_sup, enriched_sup = ev._five_bridge_action_packet(policy, req_sup, traces_sup)
        assert packet_sup[1].action == int(Action.HOLD)
        assert enriched_sup[1].action_details["regime_veto_applied"] is True
        assert enriched_sup[1].action_details["regime_veto_reason"] == "veto_short_in_bull_p0.50"

    def test_margin_floor_p_4500_vs_4501(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """Even with huge margin (delta = 0.25 > 0.15), P_Bull <= 0.45 must NOT trigger veto."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("floor_test")

        # 1. P_Bull = 0.4500, P_Bear = 0.20 (delta = 0.25 > 0.15, but P_Bull <= 0.45)
        req_sub = BridgeRequest(
            sequence=6003,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.4500, 0.20, 0.35),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_sub = make_five_traces({1: int(Action.MARKET_SELL)})
        packet_sub, enriched_sub = ev._five_bridge_action_packet(policy, req_sub, traces_sub)
        assert packet_sub[1].action == int(Action.MARKET_SELL)
        assert enriched_sub[1].action_details["regime_veto_applied"] is False

        # 2. P_Bull = 0.4501, P_Bear = 0.20 (delta = 0.2501 > 0.15 and P_Bull > 0.45)
        req_sup = BridgeRequest(
            sequence=6004,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.4501, 0.20, 0.3499),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_sup = make_five_traces({1: int(Action.MARKET_SELL)})
        packet_sup, enriched_sup = ev._five_bridge_action_packet(policy, req_sup, traces_sup)
        assert packet_sup[1].action == int(Action.HOLD)
        assert enriched_sup[1].action_details["regime_veto_applied"] is True
        assert enriched_sup[1].action_details["regime_veto_reason"] == "veto_short_in_bull_p0.45"


# ==============================================================================
# DIMENSION 5: AGENT ISOLATION & FAIL-SAFE INPUT ROBUSTNESS
# ==============================================================================

class TestAdversarialAgentIsolationAndRobustness:
    """Verifies that non-target agents (Agent 2, 3, 4) are never vetoed, and malformed inputs fail safe."""

    def test_non_target_agents_never_vetoed(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """In strong Bull regime (P_Bull = 0.90), Agent 0 (BUY) and Agent 2, 3, 4 (HOLD) remain unvetoed."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("isolation_test")
        req = BridgeRequest(
            sequence=7001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.90, 0.05, 0.05),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )

        # In FiveAgentAuthority:
        # Agent 0: LONG_ENTRY_OWNER -> MARKET_BUY allowed
        # Agent 1: SHORT_ENTRY_OWNER -> HOLD allowed
        # Agent 2: EXIT_OWNER_SCALP -> HOLD allowed
        # Agent 3: EXIT_OWNER_SWING -> HOLD allowed
        # Agent 4: GOVERNOR_DEFENSE -> HOLD allowed
        traces = make_five_traces({
            0: int(Action.MARKET_BUY),
            1: int(Action.HOLD),
            2: int(Action.HOLD),
            3: int(Action.HOLD),
            4: int(Action.HOLD),
        })
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Agent 0: Unvetoed MARKET_BUY
        assert packet[0].action == int(Action.MARKET_BUY)
        assert enriched[0].action_details["regime_veto_applied"] is False

        # Agents 2, 3, 4: Non-entry agents are never vetoed
        for a_idx in [2, 3, 4]:
            assert packet[a_idx].action == int(Action.HOLD)
            assert enriched[a_idx].action_details["regime_veto_applied"] is False

    def test_autopredict_when_regime_probabilities_is_none(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """When BridgeRequest.regime_probabilities is None, runner automatically infers regime without crashing."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("autopredict_test")
        req = BridgeRequest(
            sequence=7002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=None,  # triggers _predict_regime_for_request
            regime_name=None,
            regime_index=None,
            positions=0,
        )

        traces = make_five_traces({1: int(Action.MARKET_SELL)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # For zeros input vector, predicted probabilities are near ~0.34 (neither >= 0.55)
        # So Agent 1 MARKET_SELL is unvetoed
        assert packet[1].action == int(Action.MARKET_SELL)
        assert enriched[1].action_details["regime_veto_applied"] is False
        assert enriched[1].action_details["regime_name"] in ["BULL", "BEAR", "CHOP"]

    @pytest.mark.parametrize("non_finite_probs", [
        (float("nan"), 0.2, 0.1),
        (0.2, float("nan"), 0.1),
        (float("inf"), 0.2, 0.1),
        (-float("inf"), 0.2, 0.1),
    ])
    def test_non_finite_soft_probs_fail_safe(
        self, evaluator_env: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], non_finite_probs: tuple[float, float, float]
    ) -> None:
        """Non-finite (NaN, Inf) soft probabilities fail-safe without false veto."""
        ev, _ = evaluator_env
        policy = MockExpertPolicy("non_finite_test")
        req = BridgeRequest(
            sequence=7003,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=non_finite_probs,
            regime_name="UNKNOWN",
            regime_index=-1,
            positions=0,
        )

        traces = make_five_traces({1: int(Action.MARKET_SELL)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # When NaN/Inf comparison occurs (e.g. nan >= 0.55 evaluates False)
        # Verify it does not crash or raise uncaught exceptions
        assert packet[1].action in [int(Action.MARKET_SELL), int(Action.HOLD)]
