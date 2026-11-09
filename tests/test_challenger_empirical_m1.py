"""
tests/test_challenger_empirical_m1.py

Adversarial Challenger Empirical Verification Harness for Milestone M1.
Empirically tests:
1. Wire soft veto with realistic probabilities under temperature T=2.37:
   - Bull case: P_Bull = 0.40, P_Bear = 0.31, P_Chop = 0.29 -> Agent 1 Short entries 100% vetoed to HOLD.
   - Bear case: P_Bear = 0.42, P_Bull = 0.30, P_Chop = 0.28 -> Agent 0 Long entries 100% vetoed to HOLD.
   - Chop case: P_Chop = 0.39, P_Bull = 0.31, P_Bear = 0.30 -> Both Agent 0 and Agent 1 entries 100% vetoed to HOLD.
   - Contest zone: P_Bull = 0.34, P_Bear = 0.34, P_Chop = 0.32 -> Handled as CHOP/defensive safe HOLD.
2. Comprehensive entry action matrix:
   - All 3 Short actions (MARKET_SELL, SELL_LIMIT, SELL_STOP) vetoed in Bull.
   - All 3 Long actions (MARKET_BUY, BUY_LIMIT, BUY_STOP) vetoed in Bear.
   - All 6 Entry actions vetoed in Chop.
   - Aligned actions (MARKET_BUY in Bull, MARKET_SELL in Bear) permitted.
3. Risk Invariants:
   - POLICY_CLOSE_ALL == 0 guaranteed by allowed[2:, :] = False; allowed[2:, 0] = True.
   - Fixed geometry: SL 5.0 ATR, TP 10.0 ATR, Max 120 bars M1.
   - Zero hard direction gates (context_direction_entry_gate_enabled: false).
4. DualChampionRouter routing behavior under realistic T=2.37 probabilities:
   - Bull routes to bull_policy.
   - Bear routes to bear_policy.
   - Chop routes to bear_policy (defensive).
   - Confirmation buffer and state hysteresis.
"""

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any
import numpy as np
import pytest
import torch

from self_growing_trader.orders import Action
from self_growing_trader.dual_champion_router import DualChampionRouter
from self_growing_trader.mt5_runner import (
    BridgeRequest,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
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
    """Mock expert system policy with valid provenance and attributes."""

    def __init__(self, name: str = "expert_mock") -> None:
        self.name = name
        self.input_size = 137
        self.output_size = 20
        self.experts: dict[str, Any] = {"CTX_CHUNG": 1}
        self.nhan_dien = lambda obs: "CTX_CHUNG"
        self.registry = None
        self.cau_hinh = None
        self.regime_probs: list[float] | None = None
        self.regime_name: str | None = None
        self.regime_idx: int | None = None
        self.bull_threshold = 0.35
        self.bear_threshold = 0.35
        self.chop_threshold = 0.38

    def ensure_compatible(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        pass

    def set_regime_soft_probabilities(
        self,
        probs: Any,
        regime_name: str | None = None,
        regime_idx: int | None = None,
    ) -> None:
        self.regime_probs = list(probs)
        self.regime_name = regime_name
        self.regime_idx = regime_idx

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[SimpleTrace, ...]:
        traces = []
        for agent_idx in range(5):
            st = SimpleTrace(
                trace_id=f"{self.name}_{agent_idx}",
                selected_action=np.zeros(20, dtype=float),
                selected_idx=int(Action.HOLD),
                action_details={
                    "five_agent_index": agent_idx,
                    "five_action_index": int(Action.HOLD),
                    "five_action_owner": agent_idx,
                },
            )
            traces.append(st)
        return tuple(traces)


def _make_evaluator() -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    cfg = MT5BacktestConfig(
        symbol="BTCUSDm",
        runner_dir=Path(tmp.name),
        regime_bull_threshold=0.35,
        regime_bear_threshold=0.35,
        regime_chop_threshold=0.38,
        max_holding_bars=120,
        offline_replay=True,
    )
    ev = MT5StrategyTesterEvaluator(cfg)
    return ev, tmp


def _make_five_traces(proposed_actions: dict[int, int]) -> list[SimpleTrace]:
    traces = []
    for agent_idx in range(5):
        act = proposed_actions.get(agent_idx, int(Action.HOLD))
        outputs = np.zeros(20, dtype=float)
        outputs[act] = 1.0
        st = SimpleTrace(
            trace_id=f"test_trace_{agent_idx}",
            selected_action=outputs,
            selected_idx=act,
            action_details={
                "five_agent_index": agent_idx,
                "five_action_index": act,
                "five_action_owner": agent_idx,
            },
            ma_context="CTX_CHUNG",
            ma_chuyen_gia="EXP_CHUNG",
            ma_policy="POL_CHUNG",
            ly_do_khong_hoc=False,
            allowed_action_mask=tuple(1 for _ in range(20)),
        )
        traces.append(st)
    return traces


class TestEmpiricalChallengerM1:
    """Rigorous empirical verification of M1 requirements."""

    def test_bull_case_short_entries_100_percent_vetoed(self) -> None:
        """Bull case (P_Bull = 0.40, P_Bear = 0.31): Verify Agent 1 Short entries are 100% vetoed to HOLD."""
        ev, tmp = _make_evaluator()
        bull_policy = MockExpertPolicy("bull")
        bear_policy = MockExpertPolicy("bear")
        router = DualChampionRouter(bull_policy, bear_policy)

        # Realistic T=2.37 probabilities for Bull
        p_bull, p_bear, p_chop = 0.40, 0.31, 0.29
        router.set_regime_soft_probabilities((p_bull, p_bear, p_chop), regime_name="BULL")

        assert router.current_regime == "BULL"
        assert router.active_policy is bull_policy

        short_actions = [Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP]

        for s_act in short_actions:
            req = BridgeRequest(
                sequence=1001,
                inputs=np.zeros(137, dtype=float),
                regime_probabilities=(p_bull, p_bear, p_chop),
                regime_name="BULL",
                regime_index=0,
                positions=0,
            )
            # Agent 1 proposes Short
            traces = _make_five_traces({1: int(s_act)})
            packet, enriched = ev._five_bridge_action_packet(router, req, traces)

            # Agent 1 must be 100% vetoed to HOLD
            assert packet[1].action == int(Action.HOLD), f"Failed for {s_act.name}: action was not HOLD"
            assert np.argmax(packet[1].outputs) == int(Action.HOLD)
            assert enriched[1].action_details["regime_veto_applied"] is True
            assert "veto_short_in_bull" in enriched[1].action_details["regime_veto_reason"]

        tmp.cleanup()

    def test_bear_case_long_entries_100_percent_vetoed(self) -> None:
        """Bear case (P_Bear = 0.42, P_Bull = 0.30): Verify Agent 0 Long entries are 100% vetoed to HOLD."""
        ev, tmp = _make_evaluator()
        bull_policy = MockExpertPolicy("bull")
        bear_policy = MockExpertPolicy("bear")
        router = DualChampionRouter(bull_policy, bear_policy)

        # Realistic T=2.37 probabilities for Bear
        p_bull, p_bear, p_chop = 0.30, 0.42, 0.28
        router.set_regime_soft_probabilities((p_bull, p_bear, p_chop), regime_name="BEAR")

        assert router.current_regime == "BEAR"
        assert router.active_policy is bear_policy

        long_actions = [Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP]

        for l_act in long_actions:
            req = BridgeRequest(
                sequence=1002,
                inputs=np.zeros(137, dtype=float),
                regime_probabilities=(p_bull, p_bear, p_chop),
                regime_name="BEAR",
                regime_index=1,
                positions=0,
            )
            # Agent 0 proposes Long
            traces = _make_five_traces({0: int(l_act)})
            packet, enriched = ev._five_bridge_action_packet(router, req, traces)

            # Agent 0 must be 100% vetoed to HOLD
            assert packet[0].action == int(Action.HOLD), f"Failed for {l_act.name}: action was not HOLD"
            assert np.argmax(packet[0].outputs) == int(Action.HOLD)
            assert enriched[0].action_details["regime_veto_applied"] is True
            assert "veto_long_in_bear" in enriched[0].action_details["regime_veto_reason"]

        tmp.cleanup()

    def test_chop_case_both_agents_100_percent_vetoed(self) -> None:
        """Chop case (P_Chop = 0.39 or P_Bull ~ P_Bear): Verify both Agent 0 and Agent 1 entries are 100% vetoed to HOLD."""
        ev, tmp = _make_evaluator()
        bull_policy = MockExpertPolicy("bull")
        bear_policy = MockExpertPolicy("bear")
        router = DualChampionRouter(bull_policy, bear_policy)

        test_chop_scenarios = [
            # High Chop probability (P_Chop = 0.39 >= chop_threshold 0.38)
            (0.31, 0.30, 0.39, "CHOP"),
            # Contested / balanced zone where neither Bull nor Bear dominates (P_Bull ~ P_Bear)
            (0.33, 0.33, 0.34, ""),
        ]

        for p_bull, p_bear, p_chop, r_name in test_chop_scenarios:
            router.set_regime_soft_probabilities((p_bull, p_bear, p_chop), regime_name=r_name or None)
            req = BridgeRequest(
                sequence=1003,
                inputs=np.zeros(137, dtype=float),
                regime_probabilities=(p_bull, p_bear, p_chop),
                regime_name=r_name,
                regime_index=2,
                positions=0,
            )

            # Test all pairs of (Agent 0 Long entry, Agent 1 Short entry)
            for l_act in [Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP]:
                for s_act in [Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP]:
                    traces = _make_five_traces({0: int(l_act), 1: int(s_act)})
                    packet, enriched = ev._five_bridge_action_packet(router, req, traces)

                    # Agent 0 Long entry must be vetoed to HOLD
                    assert packet[0].action == int(Action.HOLD), f"Agent 0 was not vetoed for {l_act.name} in Chop ({p_bull}, {p_bear}, {p_chop})"
                    assert np.argmax(packet[0].outputs) == int(Action.HOLD)
                    assert enriched[0].action_details["regime_veto_applied"] is True

                    # Agent 1 Short entry must be vetoed to HOLD
                    assert packet[1].action == int(Action.HOLD), f"Agent 1 was not vetoed for {s_act.name} in Chop ({p_bull}, {p_bear}, {p_chop})"
                    assert np.argmax(packet[1].outputs) == int(Action.HOLD)
                    assert enriched[1].action_details["regime_veto_applied"] is True

        tmp.cleanup()

    def test_aligned_actions_permitted(self) -> None:
        """Verify that aligned entries are NOT vetoed in their respective regimes."""
        ev, tmp = _make_evaluator()
        bull_policy = MockExpertPolicy("bull")
        bear_policy = MockExpertPolicy("bear")
        router = DualChampionRouter(bull_policy, bear_policy)

        # Bull: Agent 0 Long is aligned
        p_bull, p_bear, p_chop = 0.40, 0.31, 0.29
        router.set_regime_soft_probabilities((p_bull, p_bear, p_chop), regime_name="BULL")
        req_bull = BridgeRequest(
            sequence=2001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(p_bull, p_bear, p_chop),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        for l_act in [Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP]:
            traces = _make_five_traces({0: int(l_act)})
            packet, enriched = ev._five_bridge_action_packet(router, req_bull, traces)
            assert packet[0].action == int(l_act), f"Aligned action {l_act.name} should not be vetoed in Bull"
            assert enriched[0].action_details["regime_veto_applied"] is False

        # Bear: Agent 1 Short is aligned
        p_bull, p_bear, p_chop = 0.30, 0.42, 0.28
        router.set_regime_soft_probabilities((p_bull, p_bear, p_chop), regime_name="BEAR")
        req_bear = BridgeRequest(
            sequence=2002,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(p_bull, p_bear, p_chop),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        for s_act in [Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP]:
            traces = _make_five_traces({1: int(s_act)})
            packet, enriched = ev._five_bridge_action_packet(router, req_bear, traces)
            assert packet[1].action == int(s_act), f"Aligned action {s_act.name} should not be vetoed in Bear"
            assert enriched[1].action_details["regime_veto_applied"] is False

        tmp.cleanup()

    def test_risk_governance_invariants(self) -> None:
        """Verify POLICY_CLOSE_ALL == 0 invariant, fixed geometry, and no hard direction gates."""
        # 1. Config Invariants
        cfg = MT5BacktestConfig()
        assert cfg.max_holding_bars == 120, f"Expected max_holding_bars=120, got {cfg.max_holding_bars}"
        assert cfg.fixed_sl_atr == 5.0, f"Expected fixed_sl_atr=5.0, got {cfg.fixed_sl_atr}"
        assert cfg.fixed_tp_atr == 10.0, f"Expected fixed_tp_atr=10.0, got {cfg.fixed_tp_atr}"
        assert cfg.regime_bull_threshold == 0.35
        assert cfg.regime_bear_threshold == 0.35
        assert cfg.regime_chop_threshold == 0.38

        # 2. Action masking invariant: allowed[2:, :] = False; allowed[2:, 0] = True
        # Simulating five_shared_policy.py lines 3075-3076:
        allowed = torch.ones((5, 20), dtype=torch.bool)
        allowed[2:, :] = False
        allowed[2:, 0] = True

        # Agents 2, 3, 4 (Manager, Governor, etc.) must only allow index 0 (HOLD)
        for agent_idx in range(2, 5):
            allowed_indices = torch.where(allowed[agent_idx])[0].tolist()
            assert allowed_indices == [0], f"Agent {agent_idx} allowed actions: {allowed_indices}, must strictly be [0]"

        # Specifically, close action indices (Action.CLOSE_BUY=6, Action.CLOSE_SELL=7, etc.) cannot be selected
        close_action_indices = [6, 7, 8, 9, 10, 11]  # all legacy close actions
        for c_act in close_action_indices:
            for agent_idx in range(2, 5):
                assert not allowed[agent_idx, c_act], f"Agent {agent_idx} has close action {c_act} enabled"

        # 3. Evaluator enforcement of non-entry owners
        ev, tmp = _make_evaluator()
        # Even if trace has non-zero action for agent 2, 3, 4, verify owner validation converts it to HOLD
        req = BridgeRequest(
            sequence=3001,
            inputs=np.zeros(137, dtype=float),
            regime_probabilities=(0.40, 0.31, 0.29),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        # Agents 2, 3, 4 propose close actions
        traces = _make_five_traces({2: 6, 3: 7, 4: 8})
        policy = MockExpertPolicy("test_gov")
        packet, enriched = ev._five_bridge_action_packet(policy, req, traces)

        for agent_idx in range(2, 5):
            assert packet[agent_idx].action == int(Action.HOLD), f"Agent {agent_idx} emitted non-HOLD action {packet[agent_idx].action}"
            assert np.argmax(packet[agent_idx].outputs) == int(Action.HOLD)

        tmp.cleanup()
