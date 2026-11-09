from __future__ import annotations

import copy

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.entry_preference import (
    counterfactual_consensus_preference_target,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.state_alias_memory import StateAliasingMemoryState


def _policy() -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=3,
        seed=19201,
    )
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_rl_delayed_reconciliation_enabled = True
    policy.direct_rnn_rl_delayed_importance_clip = 1.0
    policy.direct_rnn_rl_delayed_policy_kl_temperature = 0.25
    policy.direct_rnn_rl_delayed_max_policy_js = 0.50
    policy.direct_rnn_rl_delayed_min_effective_weight = 0.02
    policy.direct_rnn_rl_delayed_lifecycle_encoder_credit_scale = 0.0
    policy.direct_rnn_rl_delayed_counterfactual_encoder_credit_scale = 0.25
    policy._learning_enabled = True
    policy.ensure_compatible()
    module = policy.modules[0]
    module.w_out.fill(0.0)
    module.b_out.fill(0.0)
    module.b_out[1] = 1.0
    return policy


def test_counterfactual_consensus_target_does_not_reuse_behavior_action() -> None:
    # One hindsight spike is insufficient: the second-best legal entry is
    # negative, so the consensus TRADE utility remains below the margin.
    utilities = np.asarray([0.0, 2.0, -0.4, -0.6, -0.8, -1.0, -1.2])
    target, mask, trade_utility = counterfactual_consensus_preference_target(
        utilities,
        allowed_actions=range(7),
        consensus_actions=2,
        trade_margin_r=0.10,
        trade_temperature=0.25,
    )

    assert mask.all()
    assert trade_utility == 0.8
    assert target[0] < 0.5  # two independent entry alternatives support TRADE
    np.testing.assert_allclose(np.sum(target), 1.0, atol=1e-12)

    lonely_spike = np.asarray([0.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    conservative, _, lonely_utility = counterfactual_consensus_preference_target(
        lonely_spike,
        allowed_actions=range(7),
        consensus_actions=2,
        trade_margin_r=0.10,
        trade_temperature=0.25,
    )
    assert lonely_utility == 0.0
    assert conservative[0] > 0.5


def test_current_weight_reforward_is_side_effect_free_and_changes_logits() -> None:
    policy = _policy()
    module = policy.modules[0]
    observation = np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float)
    trace = policy.act(observation)
    source = trace.module_action_traces[0].forward_trace
    assert source is not None

    runtime_before = {
        name: value.copy() for name, value in policy._runtime_memory.items()
    }
    deep_before = tuple(item.copy() for item in module.deep_state)
    temporal_before = copy.deepcopy(module.rl_temporal_forward_history)
    dropout_step_before = module.hidden_dropout_step
    original_output = trace.module_action_traces[0].raw_output.copy()

    module.b_out[2] += 2.5
    _, _, _, current_output, current_forward = module.reforward_immutable(source)

    assert not np.allclose(current_output, original_output)
    assert policy._runtime_memory.keys() == runtime_before.keys()
    for name, before in runtime_before.items():
        np.testing.assert_allclose(policy._runtime_memory[name], before)
    for current, before in zip(module.deep_state, deep_before):
        np.testing.assert_allclose(current, before)
    assert module.hidden_dropout_step == dropout_step_before
    assert len(module.rl_temporal_forward_history) == len(temporal_before)
    assert current_forward.output_hidden.shape == (module.hidden_size,)


def test_frozen_memory_read_uses_current_qkv_without_mutating_runtime() -> None:
    memory = StateAliasingMemoryState.seed(
        input_size=5,
        hidden_size=4,
        memory_size=6,
        capacity=4,
        min_history=2,
        enabled=True,
        seed=33,
    )
    memory.remember(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]))
    memory.remember(np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]))
    decision = memory.read(np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]))
    counters_before = (memory.read_count, memory.ready_read_count)
    buffer_before = [item.copy() for item in memory.runtime_buffer]
    memory.value_b += 0.5

    current = memory.read_frozen(decision.observation, decision.tokens)

    assert current.ready is True
    assert not np.allclose(current.injected, decision.injected)
    assert (memory.read_count, memory.ready_read_count) == counters_before
    for actual, expected in zip(memory.runtime_buffer, buffer_before):
        np.testing.assert_allclose(actual, expected)


def test_large_policy_drift_rejects_executed_lifecycle_credit() -> None:
    policy = _policy()
    trace = policy.act(np.linspace(-0.3, 0.7, INPUT_COUNT, dtype=float))
    old_logits = np.asarray(trace.action_details["direct_rnn_rl_logits"], dtype=float)
    policy.direct_rnn_rl_delayed_max_policy_js = 0.0
    module = policy.modules[0]
    module.b_out[1] -= 6.0
    module.b_out[2] += 6.0

    current_trace, audit = policy._reforward_delayed_trace(trace, 12)

    assert audit["reforwarded"] is True
    assert audit["accepted"] is False
    assert audit["reason"] == "policy_drift_gate"
    assert audit["policy_js"] > 0.0
    assert audit["effective_weight"] == 0.0
    current_logits = np.asarray(
        current_trace.action_details["direct_rnn_rl_logits"], dtype=float
    )
    assert not np.allclose(current_logits, old_logits)


def test_executed_action_updates_head_but_not_shared_encoder_direction() -> None:
    policy = _policy()
    trace = policy.act(np.linspace(-0.2, 0.5, INPUT_COUNT, dtype=float))
    entry = trace.module_action_traces[0]
    module = policy.modules[0]
    forward = entry.forward_trace
    assert forward is not None
    action_direction = np.zeros(OUTPUT_COUNT, dtype=float)
    action_direction[1] = 1.0
    action_direction[2] = -1.0

    directions = policy._direct_rnn_full_rl_module_directions(
        module,
        forward,
        action_direction,
        0.5,
        0.0,
        entry.gate_logit,
        entry.temporal_forward_traces,
        None,
        0.0,
        0.0,
        0.0,
    )

    assert np.linalg.norm(directions["w_out"]) > 0.0
    assert np.linalg.norm(directions["value_w"]) > 0.0
    assert np.linalg.norm(directions["w_in"]) == 0.0
    assert np.linalg.norm(directions["w_rec"]) == 0.0
    for index in range(len(module.deep_w)):
        assert np.linalg.norm(directions[f"deep_w_{index}"]) == 0.0
        assert np.linalg.norm(directions[f"deep_rec_{index}"]) == 0.0


def test_delayed_lifecycle_batch_keeps_encoder_byte_stable() -> None:
    policy = _policy()
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_min_abs_mean_advantage = 0.0
    module = policy.modules[0]
    encoder_before = {
        "w_in": module.w_in.copy(),
        "w_rec": module.w_rec.copy(),
        **{f"deep_w_{index}": value.copy() for index, value in enumerate(module.deep_w)},
        **{f"deep_rec_{index}": value.copy() for index, value in enumerate(module.deep_rec)},
    }
    action_head_before = module.w_out.copy()

    for index in range(2):
        trace_id = f"delayed-{index}"
        trace = policy.act(np.linspace(-0.3 + index * 0.01, 0.7, INPUT_COUNT))
        policy._execution_trace_cache[trace_id] = trace
        policy.apply_execution_lifecycle_credit(
            trace_id,
            1.0,
            close_reason="TP",
            duration_bars=8.0,
        )

    assert module.rnn_optimizer_step == 1
    assert not np.array_equal(module.w_out, action_head_before)
    for name, before in encoder_before.items():
        if name.startswith("deep_w_"):
            current = module.deep_w[int(name.rsplit("_", 1)[1])]
        elif name.startswith("deep_rec_"):
            current = module.deep_rec[int(name.rsplit("_", 1)[1])]
        else:
            current = getattr(module, name)
        assert np.array_equal(current, before), name
    event = next(
        row for row in reversed(policy.growth_events)
        if row.get("type") == "direct_rnn_full_rl_batch"
    )
    assert event["delayed_reconciliation"]["reforwarded_count"] == 2
    assert event["delayed_reconciliation"]["lifecycle_encoder_credit_scale"] == 0.0
