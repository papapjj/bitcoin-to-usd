from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from self_growing_trader.continuous_session import ContinuousModuleRegistry, ContinuousRuntimeConfig
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def _report(trace_id: str, cause: str, context: np.ndarray, *, confidence: float = 0.82):
    snapshot = SimpleNamespace(
        canonical_context_key=np.asarray(context, dtype=float),
        context_frame_id="ctx-v24",
        selected_action_index=1,
        emitted_action_index=1,
        action_attribution_source="emitted",
        action_details={"emitted_action": 1, "bridge_action": 1},
    )
    hypothesis = SimpleNamespace(cause_id=cause, status="validated", confidence=confidence)
    return SimpleNamespace(
        trace_id=trace_id,
        snapshot=snapshot,
        hypotheses=(hypothesis,),
        coverage={"counterfactual_replay_validated": True},
    )


def test_inhibition_requires_independent_replay_verified_cases_and_is_idempotent() -> None:
    cfg = ContinuousRuntimeConfig(
        enabled=True,
        enable_persistent_specialists=False,
        min_independent_evidence_to_inhibit=3,
        min_verified_replays_to_inhibit=2,
    )
    registry = ContinuousModuleRegistry(cfg)
    policy = SimpleNamespace(continuous_adaptive_state={})
    context = np.linspace(-0.3, 0.3, 16)

    first = _report("trace-1", "wrong_direction_or_regime", context)
    events = registry.observe_report(policy, first, sequence=10)
    assert any(event["type"] == "continuous_inhibition_deferred" for event in events)
    assert policy.continuous_adaptive_state["local_inhibitions"] == []

    # Re-scanning the same report at every bridge safe point must not turn one
    # lifecycle into dozens of artificial support cases.
    assert registry.observe_report(policy, first, sequence=11) == []
    evidence = next(iter(policy.continuous_adaptive_state["module_evidence"].values()))
    assert evidence["trace_ids"] == ["trace-1"]

    registry.observe_report(policy, _report("trace-2", "wrong_direction_or_regime", context), sequence=20)
    events = registry.observe_report(policy, _report("trace-3", "wrong_direction_or_regime", context), sequence=30)
    assert any(event["type"] == "continuous_inhibition_created" for event in events)
    inhibition = policy.continuous_adaptive_state["local_inhibitions"][0]
    assert inhibition["supports"] == 3
    assert inhibition["verified_cases"] == 3
    assert inhibition["strength"] <= cfg.local_inhibition_max_strength


def test_execution_cost_and_volatility_causes_do_not_create_persistent_hold_inhibition() -> None:
    cfg = ContinuousRuntimeConfig(enabled=True, enable_persistent_specialists=False)
    registry = ContinuousModuleRegistry(cfg)
    policy = SimpleNamespace(continuous_adaptive_state={})
    context = np.linspace(-0.1, 0.1, 16)
    events = registry.observe_report(
        policy,
        _report("trace-cost", "spread_or_slippage_damage", context),
        sequence=1,
    )
    assert any(event["type"] == "continuous_inhibition_skipped_execution_context" for event in events)
    assert policy.continuous_adaptive_state["local_inhibitions"] == []


def test_v24_neutral_risk_gate_does_not_suppress_entries_or_boost_hold() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=902, m1_backend="graph")
    policy.risk_cost_gate.gate = lambda _context: {"risk": 0.5, "cost": 0.5, "patience": 0.5}  # type: ignore[method-assign]
    output = np.zeros(20, dtype=float)
    output[1] = 0.61
    output[2] = 0.37
    adjusted, info = policy._apply_biological_action_heads(
        output,
        np.zeros(24, dtype=float),
        np.zeros(4, dtype=float),
        np.zeros(len(policy.modules), dtype=float),
        {},
    )
    assert np.isclose(adjusted[1], output[1])
    assert np.isclose(adjusted[2], output[2])
    assert np.isclose(adjusted[0], output[0])
    assert info["action_dampen"] == 0.0
    assert info["trade_activity_guard_v24"] == 1.0
