"""
tests/test_adversarial_m2_iter2_stress.py

Empirical Adversarial Stress Suite for Milestone 2 Iteration 2 (challenger_m2_iter2_1).
Validates edge cases for regime_probabilities:
- Negative values (partial, full, sum > 0)
- Non-numeric / malformed container types (strings, complex, dict, inhomogeneous)
- Large lists (> 3 items)
- Inf and NaN values
- Invariant fail-closed behavior to [1/3, 1/3, 1/3] and MarketRegime.CHOP
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
    MarketRegime,
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


class TestAdversarialM2Iter2Stress:
    """Stress testing suite for degenerate and adversarial regime_probabilities inputs."""

    @pytest.fixture
    def evaluator_no_detector(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        """Evaluator with regime_detector_enabled=False to isolate pure fail-closed fallback to neutral prior."""
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            runner_dir=Path(tmp.name),
            regime_detector_enabled=False,
            offline_replay=True,
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        return ev, tmp

    @pytest.fixture
    def evaluator_with_detector(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        """Evaluator with regime_detector_enabled=True to test live detector integration fallback."""
        tmp = tempfile.TemporaryDirectory()
        cfg = MT5BacktestConfig(
            runner_dir=Path(tmp.name),
            regime_detector_enabled=True,
            offline_replay=True,
        )
        ev = MT5StrategyTesterEvaluator(cfg)
        return ev, tmp

    # --- 1. Verified Working Edge Cases (IndexError Resolution Verification) ---

    @pytest.mark.parametrize(
        "malformed_probs,desc",
        [
            ((), "Empty tuple ()"),
            ((0.55, 0.45), "Length-2 tuple"),
            (None, "None value"),
            ((0.0, 0.0, 0.0), "Zero array (0, 0, 0)"),
            ((np.nan, np.nan, np.nan), "All NaNs"),
            ((np.inf, 0.0, 0.0), "Contains positive infinity"),
            ((-np.inf, 0.0, 0.0), "Contains negative infinity"),
            ((1.0, 0.0, np.nan), "Mixed float and NaN"),
        ],
    )
    def test_iteration1_indexerror_and_nonfinite_inputs_succeed_cleanly(
        self,
        evaluator_no_detector: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        malformed_probs: Any,
        desc: str,
    ) -> None:
        """Confirms that empty, truncated, None, zero, and non-finite inputs succeed without raising IndexError,
        producing valid 5-agent action packets and valid 3-element probabilities summing to 1.0.
        """
        ev, tmp = evaluator_no_detector
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=100,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=malformed_probs,
            positions=0,
        )
        traces = _make_five_traces()
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        assert len(packet) == 5, f"Failed on {desc}"
        assert len(enriched) == 5, f"Failed on {desc}"
        for agent_idx, ep in enumerate(enriched):
            probs = ep.action_details["regime_probabilities"]
            assert len(probs) == 3, f"Agent {agent_idx} probs not length 3 for {desc}"
            assert np.isclose(sum(probs), 1.0), f"Agent {agent_idx} probs sum != 1.0 for {desc}"
            assert all(p >= 0.0 for p in probs), f"Agent {agent_idx} has negative probability for {desc}"
            assert ep.action_details["regime_index"] == int(MarketRegime.CHOP), f"Regime not CHOP for {desc}"

    def test_large_list_truncated_and_normalized(
        self,
        evaluator_no_detector: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
    ) -> None:
        """When regime_probabilities has > 3 elements, it slices the first 3 and normalizes if valid."""
        ev, tmp = evaluator_no_detector
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=101,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=[0.1, 0.2, 0.7, 0.9, 0.99, 0.5],
            positions=0,
        )
        traces = _make_five_traces()
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        assert len(packet) == 5
        probs = enriched[0].action_details["regime_probabilities"]
        assert len(probs) == 3
        assert np.isclose(probs[0], 0.1)
        assert np.isclose(probs[1], 0.2)
        assert np.isclose(probs[2], 0.7)
        assert enriched[0].action_details["regime_name"] == "CHOP"

    # --- 2. Stress Test Failure Modes (Vulnerabilities Found) ---

    @pytest.mark.parametrize(
        "negative_probs,desc",
        [
            ((-0.5, 1.0, 0.5), "Negative p_bull with sum > 0"),
            ((0.8, -0.1, 0.3), "Negative p_bear with sum > 0"),
            ((0.6, 0.6, -0.2), "Negative p_chop with sum > 0"),
            ((-0.1, 0.6, 0.5), "Negative p_bull triggers false BEAR veto"),
        ],
    )
    def test_negative_probabilities_fail_closed_to_neutral_prior(
        self,
        evaluator_no_detector: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        negative_probs: tuple[float, float, float],
        desc: str,
    ) -> None:
        """Negative probabilities violate probability axioms and must fail closed to neutral prior [1/3, 1/3, 1/3]
        and MarketRegime.CHOP.

        CURRENT BUG: mt5_runner.py accepts negative probabilities as long as sum > 0, emitting negative numbers
        in action_details['regime_probabilities'] and setting corrupt regimes.
        """
        ev, tmp = evaluator_no_detector
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=102,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=negative_probs,
            positions=0,
        )
        traces = _make_five_traces()
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        probs = enriched[0].action_details["regime_probabilities"]
        # Invariant 1: No probability can be negative
        for p in probs:
            assert p >= 0.0, f"VULNERABILITY: negative probability {p} accepted in {desc} (probs={probs})"
        # Invariant 2: Must fail closed to neutral prior [1/3, 1/3, 1/3]
        assert np.allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), (
            f"VULNERABILITY: did not fail closed to neutral prior for {desc} (probs={probs})"
        )
        # Invariant 3: Regime must fail closed to CHOP
        assert enriched[0].action_details["regime_index"] == int(MarketRegime.CHOP), (
            f"VULNERABILITY: regime_index={enriched[0].action_details['regime_index']} instead of CHOP for {desc}"
        )

    def test_negative_probabilities_trigger_unintended_soft_veto(
        self,
        evaluator_no_detector: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
    ) -> None:
        """Negative probability (0.8, -0.1, 0.3) has sum=1.0. p_bull=0.8 triggers soft veto on Agent 1 Short.
        Because the distribution is mathematically corrupt, it should NOT trigger a directional veto!
        """
        ev, tmp = evaluator_no_detector
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=103,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=(0.8, -0.1, 0.3),  # Corrupt
            positions=0,
        )
        # Agent 1 proposes MARKET_SELL
        traces = _make_five_traces(actions={1: int(Action.MARKET_SELL)})
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        # Under proper fail-closed behavior to CHOP/neutral prior, neither p_bull nor p_bear >= 0.55,
        # so Agent 1's MARKET_SELL should NOT be vetoed by corrupted probabilities.
        assert enriched[1].action_details["regime_veto_applied"] is False, (
            "VULNERABILITY: Agent 1 was vetoed based on corrupt negative probabilities"
        )
        assert packet[1].action == int(Action.MARKET_SELL)

    @pytest.mark.parametrize(
        "corrupted_input,desc",
        [
            (("a", "b", "c"), "String tuple"),
            ((1 + 2j, 0, 0), "Complex numbers"),
            ({"a": 1, "b": 2, "c": 3}, "Dictionary"),
            ([[0.3, 0.3], [0.4]], "Inhomogeneous nested list"),
        ],
    )
    def test_corrupted_types_fail_closed_without_unhandled_crash(
        self,
        evaluator_no_detector: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory],
        corrupted_input: Any,
        desc: str,
    ) -> None:
        """When regime_probabilities contains non-convertible types, runner must fail closed
        instead of crashing with unhandled ValueError or TypeError.
        """
        ev, tmp = evaluator_no_detector
        policy = MockExpertPolicy()
        req = BridgeRequest(
            sequence=104,
            inputs=np.zeros(137),
            raw_inputs=np.zeros(137),
            regime_probabilities=corrupted_input,
            positions=0,
        )
        traces = _make_five_traces()
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)
        assert len(packet) == 5
        probs = enriched[0].action_details["regime_probabilities"]
        assert len(probs) == 3
        assert np.isclose(sum(probs), 1.0)
        assert enriched[0].action_details["regime_index"] == int(MarketRegime.CHOP)
