"""
tests/test_adversarial_m2_soft_probs_stress.py

Adversarial Stress Test Suite for Milestone 2 (challenger_m2_2):
Empirical stress-testing of `_five_bridge_action_packet` under degraded and corrupted conditions.

1. UnboundLocalError Immunity & Owner-Invalid Fail-Closed Behavior:
   - Empty `ma_context`, `ma_chuyen_gia`, `ma_policy`
   - Policy lacking `experts` or `nhan_dien` interfaces
   - Inactive trace flags (`ly_do_khong_hoc`, `khong_co_policy_ra_lenh`)
   - Verifies that `regime_veto_applied` is NEVER unbound and all 5 agents fail closed to HOLD (0).

2. Malformed Soft Probabilities Vulnerability:
   - `regime_probabilities = ()` (empty sequence) -> Triggers IndexError at line 9320
   - `regime_probabilities = (0.0, 0.0)` (length 2) -> Triggers IndexError at line 9322
   - `regime_probabilities = (0.0, 0.0, 0.0)` (all-zero degenerate) -> Silently bypasses soft veto
   - `regime_probabilities = (nan, nan, nan)` (NaN values) -> Silently bypasses soft veto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pytest

from self_growing_trader.orders import Action
from self_growing_trader.mt5_runner import (
    BridgeRequest,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
    OUTPUT_COUNT,
)


@dataclass
class SimpleTrace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))
    selected_idx: int = int(Action.HOLD)
    confidence: float = 0.8
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
    """Mock policy that satisfies the expert system contract."""

    def __init__(self, with_experts: bool = True, with_nhan_dien: bool = True) -> None:
        self.input_size = 137
        self.output_size = 20
        if with_experts:
            self.experts = {"CTX_CHUNG": 1}
        if with_nhan_dien:
            self.nhan_dien = lambda obs: "CTX_CHUNG"


def _make_five_traces(actions: dict[int, int] | None = None, **kwargs: Any) -> list[SimpleTrace]:
    actions = actions or {}
    traces = []
    for i in range(5):
        act = actions.get(i, int(Action.HOLD))
        act_arr = np.zeros(20, dtype=float)
        act_arr[act] = 1.0
        t = SimpleTrace(
            trace_id=f"trace-{i}",
            selected_action=act_arr,
            selected_idx=act,
            action_details={"five_agent_index": i, "five_action_index": act},
            **kwargs,
        )
        traces.append(t)
    return traces


class TestAdversarialOwnerInvalidAndUnbound:
    """Empirical verification that regime_veto_applied is NEVER unbound and owner-invalid fails closed to HOLD."""

    @pytest.fixture
    def evaluator(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            runner_dir=Path(tmp.name),
            regime_detector_enabled=True,
            offline_replay=True,
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        return ev, tmp

    @pytest.mark.parametrize(
        "trace_kwargs,policy_factory,desc",
        [
            ({"ma_context": ""}, lambda: MockExpertPolicy(), "Empty ma_context in trace"),
            ({"ma_chuyen_gia": ""}, lambda: MockExpertPolicy(), "Empty ma_chuyen_gia in trace"),
            ({"ma_policy": ""}, lambda: MockExpertPolicy(), "Empty ma_policy in trace"),
            ({}, lambda: MockExpertPolicy(with_experts=False), "Policy missing experts attribute"),
            ({}, lambda: MockExpertPolicy(with_nhan_dien=False), "Policy missing nhan_dien attribute"),
            ({"ly_do_khong_hoc": True}, lambda: MockExpertPolicy(), "Trace marked ly_do_khong_hoc"),
            ({"context_audit": {"khong_co_policy_ra_lenh": True}}, lambda: MockExpertPolicy(), "Audit marked khong_co_policy_ra_lenh"),
        ],
    )
    def test_owner_invalid_immunity_to_unbound_local_error(
        self,
        evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        trace_kwargs: dict[str, Any],
        policy_factory: Any,
        desc: str,
    ) -> None:
        """Confirms that under all owner_valid=False variations, regime_veto_applied is never unbound and all actions are HOLD."""
        ev, tmp = evaluator
        policy = policy_factory()
        req = BridgeRequest(
            sequence=10,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=(0.70, 0.20, 0.10),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        # Even if Agent 0 and Agent 1 propose entry orders, owner-invalid must force 100% HOLD
        traces = _make_five_traces(
            actions={0: int(Action.MARKET_BUY), 1: int(Action.MARKET_SELL)},
            **trace_kwargs,
        )
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        assert len(packet) == 5
        assert len(enriched) == 5
        for agent_idx, (bp, ep) in enumerate(zip(packet, enriched)):
            details = ep.action_details
            # 1. UnboundLocalError verification: regime_veto_applied MUST exist and be boolean
            assert "regime_veto_applied" in details, f"regime_veto_applied missing for agent {agent_idx} ({desc})"
            assert isinstance(details["regime_veto_applied"], bool)
            assert details["regime_veto_applied"] is False
            assert details["regime_veto_reason"] == ""

            # 2. Fail-closed to HOLD
            assert bp.action == int(Action.HOLD), f"Agent {agent_idx} did not fail closed to HOLD ({desc})"
            assert np.argmax(bp.outputs) == int(Action.HOLD)
            assert details["wire_action"] == int(Action.HOLD)
            assert details["no_neural_command"] is True
            assert details["action_origin"] == "mask"


class TestAdversarialMalformedSoftProbabilities:
    """Empirically stresses _five_bridge_action_packet under malformed soft probabilities."""

    @pytest.fixture
    def evaluator(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            runner_dir=Path(tmp.name),
            regime_detector_enabled=True,
            offline_replay=True,
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        return ev, tmp

    def test_soft_probs_none_succeeds_via_neutral_prior(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """When regime_probabilities is None, fallback to neutral prior [1/3, 1/3, 1/3] occurs."""
        ev, tmp = evaluator
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=11,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=None,
            positions=0,
        )
        traces = _make_five_traces()
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)
        assert len(packet) == 5
        for ep in enriched:
            assert len(ep.action_details["regime_probabilities"]) == 3
            assert np.isclose(sum(ep.action_details["regime_probabilities"]), 1.0)

    def test_soft_probs_empty_succeeds_without_index_error(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """When regime_probabilities is empty (), runner sanitizes input and succeeds without IndexError."""
        ev, tmp = evaluator
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=12,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=(),  # Malformed: empty sequence
            positions=0,
        )
        traces = _make_five_traces()
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)
        assert len(packet) == 5
        for ep in enriched:
            assert len(ep.action_details["regime_probabilities"]) == 3
            assert np.isclose(sum(ep.action_details["regime_probabilities"]), 1.0)
            assert np.isclose(
                ep.action_details["p_bull"] + ep.action_details["p_bear"] + ep.action_details["p_chop"],
                1.0,
            )

    def test_soft_probs_length_two_succeeds_without_index_error(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """When regime_probabilities has only 2 elements, runner sanitizes input and succeeds without IndexError."""
        ev, tmp = evaluator
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=13,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=(0.55, 0.45),  # Malformed: missing P_Chop
            positions=0,
        )
        traces = _make_five_traces()
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)
        assert len(packet) == 5
        for ep in enriched:
            assert len(ep.action_details["regime_probabilities"]) == 3
            assert np.isclose(sum(ep.action_details["regime_probabilities"]), 1.0)
            assert np.isclose(
                ep.action_details["p_bull"] + ep.action_details["p_bear"] + ep.action_details["p_chop"],
                1.0,
            )

    def test_degenerate_zero_soft_probs_bypasses_soft_veto(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]
    ) -> None:
        """Empirical proof: when regime_probabilities is [0.0, 0.0, 0.0], soft veto conditions fail to engage."""
        ev, tmp = evaluator
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=14,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=(0.0, 0.0, 0.0),  # Malformed: degenerate zero probabilities
            positions=0,
        )
        traces = _make_five_traces(actions={1: int(Action.MARKET_SELL)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)
        # Because neither p_bull nor p_bear >= 0.55, Agent 1's MARKET_SELL is NOT vetoed
        assert packet[1].action == int(Action.MARKET_SELL)
        assert enriched[1].action_details["regime_veto_applied"] is False
