from __future__ import annotations

import numpy as np

from self_growing_trader.causal_ledger import CausalLedger, _execution_action_from_details
from self_growing_trader.continuous_session import ContinuousRuntimeConfig, ContinuousSessionRuntime
from self_growing_trader.mt5_runner import Action, BridgeRequest, ExperienceStarvationState, MT5StrategyTesterEvaluator


def test_v27_continuous_default_keeps_explicit_neural_hold_legal() -> None:
    cfg = ContinuousRuntimeConfig()
    assert cfg.exclude_hold_from_neural_selection is False
    outputs = np.zeros(20, dtype=float)
    outputs[0] = 0.9
    outputs[1] = 0.1
    action, _ = MT5StrategyTesterEvaluator._choose_action(outputs, {0, 1})
    assert action == int(Action.HOLD)


def test_v27_flat_idle_is_not_action_feedback() -> None:
    before = BridgeRequest(sequence=11, inputs=np.zeros(4), positions=0, orders_pipe="")
    after = BridgeRequest(sequence=12, inputs=np.zeros(4), positions=0, orders_pipe="", execution_pipe="")
    detail = MT5StrategyTesterEvaluator._execution_details(before, after, int(Action.HOLD))
    assert detail["flat_idle"] is True
    assert detail["execution_state"] == "FLAT_IDLE"
    assert detail["action_feedback_eligible"] is False
    assert detail["memory_eligible"] is False


def test_v27_flat_idle_does_not_increase_neural_starvation() -> None:
    before = BridgeRequest(sequence=1, inputs=np.zeros(100), positions=0, day_trades=0)
    after = BridgeRequest(sequence=2, inputs=np.zeros(100), positions=0, day_trades=0)
    state = ExperienceStarvationState()
    state.confirm_execution(before, after, emitted_action=int(Action.HOLD))
    preview = state.preview(after, 3, suppress_flat_idle_feedback=True)
    assert state.no_trade_streak == 0
    assert state.unfilled_streak == 0
    assert preview["cadence_deficit"] == 0.0
    assert preview["starvation"] == 0.0


def test_flat_idle_exploration_signal_can_be_recorded_without_masking_entries() -> None:
    state = ExperienceStarvationState()
    state.observe_flat_idle()
    state.observe_flat_idle()
    preview = state.preview(
        BridgeRequest(sequence=3, inputs=np.zeros(100), positions=0, day_trades=0),
        0,
        suppress_flat_idle_feedback=True,
    )
    assert state.no_trade_streak == 2
    assert preview["starvation"] > 0.0
    # The state carries pressure only; action legality is owned by the bridge
    # contract and is not changed here.
    assert preview["flat_idle_feedback_suppressed"] == 1.0


def test_v27_unfilled_real_entry_still_produces_recovery_signal() -> None:
    before = BridgeRequest(sequence=1, inputs=np.zeros(100), positions=0, day_trades=0)
    after = BridgeRequest(sequence=2, inputs=np.zeros(100), positions=0, day_trades=0)
    state = ExperienceStarvationState()
    state.confirm_execution(before, after, emitted_action=int(Action.MARKET_BUY))
    preview = state.preview(after, 0, suppress_flat_idle_feedback=True)
    assert state.no_trade_streak == 1
    assert state.unfilled_streak == 1
    assert preview["starvation"] > 0.0


def test_v27_hold_or_flat_details_cannot_enter_causal_action_channel() -> None:
    action, source = _execution_action_from_details({
        "emitted_action": 0,
        "wire_action": 0,
        "flat_idle_non_feedback": True,
        "action_attribution_source": "hold_no_execution",
    })
    assert action == -1
    assert source == "none"


def test_v27_ledger_rejects_hold_snapshot() -> None:
    class P:
        revision = 0
        _decision_context = {}
        _causal_last_memory_query = {}

    class T:
        trace_id = "flat-hold"
        timestamp = 1.0
        context_frame_id = ""
        confidence = 0.0
        selected_idx = 0
        candidate_scores = np.zeros(1)
        candidate_actions = np.zeros((1, 20))
        selected_action = np.zeros(20)
        action_details = {
            "emitted_action": 0,
            "wire_action": 0,
            "flat_idle_non_feedback": True,
            "action_attribution_source": "hold_no_execution",
        }
        account_context = {}
        context_audit = {}
        canonical_context_key = np.zeros(64)
        allowed_action_mask = tuple()
        graph_traces = {}
        novelty = 0.0
        conflict = 0.0
        prediction_error = 0.0
        risk = 0.0
        memory_match_quality = 0.0

    ledger = CausalLedger()
    snapshot = ledger.register_decision(P(), T())
    assert snapshot.emitted_action_index == -1
    assert ledger.snapshot_for("flat-hold") is None


def test_v27_record_bridge_decision_registers_only_entry() -> None:
    class Ledger:
        def __init__(self):
            self.calls = 0
        def register_decision(self, policy, trace):
            self.calls += 1

    class Policy:
        def __init__(self):
            self.causal_ledger = Ledger()
            self._causal_trace_archive = {}

    class Trace:
        trace_id = "h"
        action_details = {}

    runtime = ContinuousSessionRuntime(ContinuousRuntimeConfig(enabled=True))
    req = BridgeRequest(sequence=3, inputs=np.zeros(4), positions=0)
    policy = Policy()
    runtime.record_bridge_decision(policy, req, Trace(), policy_action=0, legal_action=0, emitted_action=0, allowed_actions={0,1,2}, execution_bootstrap=False)
    assert policy.causal_ledger.calls == 0
    runtime.record_bridge_decision(policy, req, Trace(), policy_action=1, legal_action=1, emitted_action=1, allowed_actions={0,1,2}, execution_bootstrap=False)
    assert policy.causal_ledger.calls == 1
