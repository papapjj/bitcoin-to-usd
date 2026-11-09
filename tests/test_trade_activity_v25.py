from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import numpy as np

from self_growing_trader.causal_ledger import snapshot_from_trace
from self_growing_trader.continuous_session import ContinuousModuleRegistry, ContinuousRuntimeConfig, migrate_trade_activity_v25
from self_growing_trader.mt5_runner import BridgeRequest, ExperienceStarvationState
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def _report(trace_id: str, snapshot, *, cause: str = "wrong_direction_or_regime"):
    return SimpleNamespace(
        trace_id=trace_id,
        snapshot=snapshot,
        hypotheses=(SimpleNamespace(cause_id=cause, status="validated", confidence=0.85),),
        coverage={"counterfactual_replay_validated": True},
    )


def test_v25_snapshot_uses_emitted_bridge_action_not_m1_hypothesis_index() -> None:
    trace = SimpleNamespace(
        trace_id="t-v25",
        timestamp=1.0,
        selected_idx=3,  # hypothesis id, intentionally not the action
        selected_action=np.zeros(20, dtype=float),
        candidate_scores=np.zeros(4, dtype=float),
        candidate_actions=np.zeros((4, 20), dtype=float),
        confidence=0.2,
        action_details={
            "policy_action": 1,
            "legal_action": 1,
            "emitted_action": 2,
            "bridge_action": 2,
            "action_attribution_source": "emitted",
        },
        execution_details={},
        context_audit={},
        graph_traces={},
        account_context={},
        canonical_context_key=np.ones(16, dtype=float),
        allowed_action_mask=(True,) * 12,
        novelty=0.0,
        conflict=0.0,
        prediction_error=0.0,
        risk=0.0,
        memory_match_quality=0.0,
    )
    policy = SimpleNamespace(revision=1, _decision_context={}, _causal_last_memory_query={})
    snapshot = snapshot_from_trace(policy, trace)
    assert snapshot.model_hypothesis_index == 3
    assert snapshot.selected_hypothesis == 3
    assert snapshot.selected_action_index == 2
    assert snapshot.emitted_action_index == 2
    assert snapshot.action_attribution_source == "emitted"


def test_v25_wrong_direction_inhibition_requires_real_emitted_entry_action() -> None:
    cfg = ContinuousRuntimeConfig(
        enabled=True,
        enable_persistent_specialists=False,
        min_independent_evidence_to_inhibit=1,
        min_verified_replays_to_inhibit=1,
    )
    registry = ContinuousModuleRegistry(cfg)
    ctx = np.linspace(-0.2, 0.2, 16)
    no_mapping = SimpleNamespace(
        canonical_context_key=ctx,
        context_frame_id="ctx",
        selected_action_index=1,
        action_details={},
    )
    policy = SimpleNamespace(continuous_adaptive_state={})
    events = registry.observe_report(policy, _report("no-map", no_mapping), sequence=1)
    assert any(event["type"] == "continuous_inhibition_deferred_action_mapping" for event in events)
    assert policy.continuous_adaptive_state["local_inhibitions"] == []

    mapped = SimpleNamespace(
        canonical_context_key=ctx,
        context_frame_id="ctx2",
        selected_action_index=2,
        emitted_action_index=3,
        action_attribution_source="emitted",
        action_details={"emitted_action": 3, "bridge_action": 3},
    )
    policy = SimpleNamespace(continuous_adaptive_state={})
    registry.observe_report(policy, _report("mapped", mapped), sequence=2)
    item = policy.continuous_adaptive_state["local_inhibitions"][0]
    assert item["action"] == 3
    assert item["replacement"] == 4
    assert item["action_source"] == "emitted"


def test_v25_entry_timing_never_creates_persistent_hold_or_direction_inhibition() -> None:
    cfg = ContinuousRuntimeConfig(enabled=True, enable_persistent_specialists=False)
    registry = ContinuousModuleRegistry(cfg)
    snapshot = SimpleNamespace(
        canonical_context_key=np.ones(16, dtype=float),
        context_frame_id="ctx",
        emitted_action_index=1,
        action_attribution_source="emitted",
        action_details={"emitted_action": 1},
    )
    policy = SimpleNamespace(continuous_adaptive_state={})
    events = registry.observe_report(policy, _report("timing", snapshot, cause="entry_too_early"), sequence=1)
    assert any(event["type"] == "continuous_timing_replacement_required" for event in events)
    assert policy.continuous_adaptive_state["local_inhibitions"] == []


def test_v25_migration_rebases_gate_and_quarantines_unsafe_inhibitions() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=910, m1_backend="graph")
    policy.risk_cost_gate.b_out[:] = np.array([2.0, 1.0, 1.5])
    policy.risk_cost_gate.state[:] = 0.7
    policy.continuous_adaptive_state = {
        "trade_activity_schema_version": 24,
        "local_inhibitions": [
            {"action": 1, "replacement": 0},
            {"action": 1, "replacement": 2, "action_source": "emitted"},
        ],
    }
    result = migrate_trade_activity_v25(policy, ContinuousRuntimeConfig(enabled=True))
    assert result["migrated"] is True
    assert np.allclose(policy.risk_cost_gate.b_out, np.array([0.3, 0.15, 0.225]))
    assert np.allclose(policy.risk_cost_gate.state, 0.0)
    assert len(policy.continuous_adaptive_state["local_inhibitions"]) == 1
    assert len(policy.continuous_adaptive_state["legacy_inhibitions_quarantined_v25"]) == 1


def test_v25_starvation_mutates_only_after_mt5_can_confirm_previous_intent() -> None:
    state = ExperienceStarvationState()
    before = BridgeRequest(sequence=1, inputs=np.zeros(1), positions=0, day_trades=0)
    # Emitting an intent must not itself claim an unfilled trade.
    preview = state.snapshot(before, 1, 0, 0, 1)
    assert preview["no_trade_streak"] == 0.0
    assert preview["unfilled_streak"] == 0.0
    after_no_fill = BridgeRequest(sequence=2, inputs=np.zeros(1), positions=0, day_trades=0)
    assert state.confirm_execution(before, after_no_fill, emitted_action=1) is False
    assert state.no_trade_streak == 1
    assert state.unfilled_streak == 1
    after_fill = BridgeRequest(sequence=3, inputs=np.zeros(1), positions=1, day_trades=1)
    assert state.confirm_execution(after_no_fill, after_fill, emitted_action=1) is True
    assert state.no_trade_streak == 0
    assert state.unfilled_streak == 0


def test_v25_final_credit_uses_execution_trace_cache_and_probability_targets() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=911, m1_backend="graph")
    policy._continuous_session_active = True
    trace = policy.act(np.zeros(4, dtype=float))
    trace = replace(trace, action_details={"action_origin": "policy", "emitted_action": 1, "bridge_action": 1})
    policy._execution_trace_cache = {trace.trace_id: trace}
    calls = []
    policy.risk_cost_gate.observe_probability = lambda context, probabilities, *, strength: calls.append((np.asarray(probabilities), strength))  # type: ignore[method-assign]
    lifecycle = SimpleNamespace(position_ticket="p1", slippage_r=0.0)
    policy.apply_execution_lifecycle_credit(trace.trace_id, -1.0, close_reason="SL", duration_bars=5, lifecycle=lifecycle)
    assert calls
    probabilities, strength = calls[-1]
    assert 0.50 < probabilities[0] <= 0.58
    assert 0.02 <= strength <= 0.08


def test_v25_memory_hold_sanitizer_rebases_only_during_normal_inactivity() -> None:
    from self_growing_trader.self_growing_runner import _v25_sanitize_continuous_memory_bias

    policy = SimpleNamespace(
        _continuous_session_active=True,
        _continuous_runtime_config=SimpleNamespace(inactivity_recalibration_after_decisions=8),
        _experience_starvation={"no_trade_streak": 12},
        _account_context={"current_equity_drawdown": 0.0, "open_risk_ratio": 0.0},
        _decision_context={"execution_context": {"spread_points": 10.0}},
        output_size=20,
    )
    source = {
        "replacement_action_bias": np.array([0.22, 0.00, 0.01, 0.0, 0.0, 0.0, 0.0] + [0.0] * 13),
        "action_bias": np.array([0.22, 0.00, 0.01, 0.0, 0.0, 0.0, 0.0] + [0.0] * 13),
    }
    result = _v25_sanitize_continuous_memory_bias(policy, source)
    assert result["replacement_action_bias"][0] <= 0.0450001
    assert result["trade_activity_memory_hold_suppressed_v25"] > 0.0
    assert source["replacement_action_bias"][0] == 0.22

    # Hard account pressure must preserve the memory recommendation rather than
    # fabricate activity recovery under a real safety constraint.
    policy._account_context = {"current_equity_drawdown": 0.20, "open_risk_ratio": 0.0}
    pressured = _v25_sanitize_continuous_memory_bias(policy, source)
    assert pressured["replacement_action_bias"][0] == 0.22
