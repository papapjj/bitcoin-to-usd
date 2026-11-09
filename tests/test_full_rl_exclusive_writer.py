from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np

from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy


def _full_rl_policy() -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=10,
        hidden_layers=6,
        seed=2401,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_adamw_learning_rate = 5.0e-4
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_min_abs_mean_advantage = 0.0
    policy._learning_enabled = True
    policy.ensure_compatible()
    return policy


def _core_snapshot(policy: ChuyenGiaPolicy) -> dict[str, list[np.ndarray]]:
    fields = (
        "w_in", "w_rec", "w_out", "b_hidden", "b_out", "gate_w", "value_w",
        "gate_b", "value_b", "deep_w", "deep_rec", "deep_b",
    )
    snapshot: dict[str, list[np.ndarray]] = {}
    for field in fields:
        values: list[np.ndarray] = []
        for module in policy.modules:
            value = getattr(module, field)
            if isinstance(value, tuple):
                values.extend(np.asarray(item, dtype=float).copy() for item in value)
            else:
                values.append(np.asarray(value, dtype=float).copy())
        snapshot[field] = values
    return snapshot


def _assert_core_unchanged(policy: ChuyenGiaPolicy, before: dict[str, list[np.ndarray]]) -> None:
    after = _core_snapshot(policy)
    assert set(after) == set(before)
    for field, values in before.items():
        assert len(after[field]) == len(values)
        for previous, current in zip(values, after[field], strict=True):
            assert np.array_equal(current, previous), field


def test_full_rl_blocks_every_legacy_direct_rnn_writer() -> None:
    policy = _full_rl_policy()
    module = policy.modules[0]
    observation = np.linspace(-0.8, 0.7, INPUT_COUNT, dtype=float)
    before = _core_snapshot(policy)

    rejected = module.apply_rnn_adamw_cosine(
        {"w_out": np.ones_like(module.w_out)}, source="legacy_local_credit"
    )
    assert rejected["optimizer_skipped_non_rl"] == 1.0

    module.mutate(np.random.default_rng(2402), sigma=0.2)
    module.apply_plasticity_rules(observation, np.ones(module.hidden_size), reward_modulator=1.0)
    module.apply_homeostatic_plasticity()
    policy._apply_fitness_credit(module, credit=1.0, observation=observation)
    bridge_gate_before = float(module.gate_b)
    policy.apply_bridge_feedback((0.0,), (0.0,), fitness=-1.0)
    assert module.gate_b == bridge_gate_before

    causal = policy.apply_causal_memory_plasticity()
    curiosity = policy.apply_curiosity_proposal()
    assert causal["skipped_full_rl"] is True
    assert curiosity["skipped_full_rl"] is True
    _assert_core_unchanged(policy, before)


def test_full_rl_disables_untrained_action_and_risk_overlays_and_credit() -> None:
    policy = _full_rl_policy()
    raw_output = np.linspace(-0.5, 0.5, OUTPUT_COUNT, dtype=float)
    state = np.zeros(INPUT_COUNT, dtype=float)
    side_before = (
        pickle.dumps(policy.action_parameter_head),
        pickle.dumps(policy.risk_cost_gate),
    )

    adjusted, risk = policy._apply_biological_action_heads(
        raw_output,
        state,
        state,
        np.zeros(len(policy.modules), dtype=float),
        {},
    )
    assert np.array_equal(adjusted, raw_output)
    assert risk["full_rl_auxiliary_action_overlays_disabled"] == 1.0
    assert risk["parameter_replacement_mode"] == "disabled_full_rl"

    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    signal = policy.observe_transition(
        trace,
        np.linspace(-0.3, 0.5, INPUT_COUNT, dtype=float),
        reward=0.75,
        cost=0.10,
        done=False,
    )
    assert signal["full_rl_auxiliary_credit_skipped"] == 1.0
    assert pickle.dumps(policy.action_parameter_head) == side_before[0]
    assert pickle.dumps(policy.risk_cost_gate) == side_before[1]

    policy._execution_trace_cache["full-rl-legacy-credit"] = trace
    policy.apply_execution_lifecycle_credit(
        "full-rl-legacy-credit", 0.75, close_reason="TP", duration_bars=8.0
    )
    assert any(
        event.get("type") == "direct_rnn_full_rl_legacy_credit_skipped"
        and event.get("path") == "parameter_replacement_lifecycle"
        for event in policy.growth_events
    )
    assert pickle.dumps(policy.action_parameter_head) == side_before[0]
    assert pickle.dumps(policy.risk_cost_gate) == side_before[1]


def test_full_rl_lifecycle_writes_audit_expectancy_without_decision_memory_credit() -> None:
    policy = _full_rl_policy()
    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    trace = replace(
        trace,
        action_details={
            **dict(trace.action_details),
            "emitted_action": 1,
            "wire_action": 1,
            "semantic_action": 1,
        },
        execution_details={**dict(trace.execution_details), "action_origin": "policy"},
    )
    policy._execution_trace_cache[trace.trace_id] = trace
    decision_ledger_before = pickle.dumps(policy.memory_supernet.contextual_expectancy)
    side_before = (
        pickle.dumps(policy.action_parameter_head),
        pickle.dumps(policy.risk_cost_gate),
    )

    policy.apply_execution_lifecycle_credit(
        trace.trace_id,
        0.75,
        close_reason="TP",
        duration_bars=8.0,
        trace=trace,
    )

    report = policy.contextual_expectancy_report()
    assert report["ledger_source"] == "full_rl_audit"
    assert report["telemetry_only"] is True
    assert report["policy_origin_lifecycle_count"] == 1
    assert report["conditioned_entry_expectancy"]["record_count"] == 1
    assert policy._last_lifecycle_credit_ack["components"]["expectancy_audit"] == 1
    assert policy._last_lifecycle_credit_ack["components"]["expectancy_audit_only_full_rl"] == 1
    # The audit cannot flow back into memory logits/action selection.
    assert pickle.dumps(policy.memory_supernet.contextual_expectancy) == decision_ledger_before
    assert pickle.dumps(policy.action_parameter_head) == side_before[0]
    assert pickle.dumps(policy.risk_cost_gate) == side_before[1]


def test_full_rl_mode_stays_exclusive_when_learning_is_frozen() -> None:
    policy = _full_rl_policy()
    policy._learning_enabled = False
    policy.ensure_compatible()
    module = policy.modules[0]
    before = _core_snapshot(policy)

    assert policy._direct_rnn_full_rl_mode_enabled() is True
    assert policy._direct_rnn_full_rl_active() is False
    module.mutate(np.random.default_rng(2403), sigma=0.2)
    module.apply_homeostatic_plasticity()
    policy._apply_fitness_credit(module, credit=-1.0, observation=np.ones(INPUT_COUNT))
    _assert_core_unchanged(policy, before)
