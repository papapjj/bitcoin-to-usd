from __future__ import annotations

from dataclasses import replace
import pickle

import numpy as np

from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    bounded_entry_behavior_policy,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT


def _full_rl_policy() -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=6,
        seed=2201,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_adamw_learning_rate = 5.0e-4
    policy.direct_rnn_gradient_clip = 10.0
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_min_abs_mean_advantage = 0.0
    policy.direct_rnn_rl_value_loss_weight = 0.25
    policy.direct_rnn_rl_entropy_weight = 0.01
    policy._learning_enabled = True
    policy.ensure_compatible()

    module = policy.modules[0]
    # Make BUY the emitted action without saturating the final tanh surface;
    # the hidden path remains non-zero so every depth block has a gradient.
    module.w_out.fill(0.0)
    module.w_out[1, 0] = 0.25
    module.w_out[0, 0] = -0.10
    module.b_out.fill(0.0)
    module.b_out[1] = 1.0
    return policy


def _settle(policy: ChuyenGiaPolicy, trace_id: str, outcome: float) -> None:
    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    assert int(np.argmax(trace.selected_action[:12])) == 1
    assert trace.module_action_traces[0].forward_trace is not None
    policy._execution_trace_cache[trace_id] = trace
    policy.apply_execution_lifecycle_credit(
        trace_id,
        outcome,
        close_reason="TP" if outcome > 0.0 else "SL",
        duration_bars=8.0,
    )


def test_full_lifecycle_rl_updates_all_six_hidden_blocks_via_adamw() -> None:
    policy = _full_rl_policy()
    module = policy.modules[0]
    assert module.rnn_full_rl_enabled is True
    assert len(module.deep_w) == 5

    before = {
        "w_in": module.w_in.copy(),
        "w_rec": module.w_rec.copy(),
        "w_out": module.w_out.copy(),
        "value_w": module.value_w.copy(),
        **{f"deep_w_{index}": layer.copy() for index, layer in enumerate(module.deep_w)},
        **{f"deep_rec_{index}": layer.copy() for index, layer in enumerate(module.deep_rec)},
        **{f"deep_b_{index}": layer.copy() for index, layer in enumerate(module.deep_b)},
    }

    rejected = module.apply_rnn_adamw_cosine(
        {"w_in": np.ones_like(module.w_in)}, source="legacy_local_credit"
    )
    assert rejected["updated"] == 0.0
    assert rejected["optimizer_skipped_non_rl"] == 1.0

    _settle(policy, "full-rl-1", 1.25)
    assert module.rnn_optimizer_step == 0
    _settle(policy, "full-rl-2", -0.65)

    assert module.rnn_optimizer_step == 1
    routing = policy.direct_rnn_rl_runtime_snapshot()
    assert routing["queued_lifecycle"] == 2
    assert routing["queued_preference"] == 0
    assert routing["lifecycle_optimizer_batches"] == 1
    assert routing["optimizer_updated_modules"] == 1
    assert routing["optimizer_steps"] == 1
    assert routing["pending_lifecycle"] == 0
    assert not np.allclose(module.w_in, before["w_in"])
    assert not np.allclose(module.w_rec, before["w_rec"])
    assert not np.allclose(module.w_out, before["w_out"])
    assert not np.allclose(module.value_w, before["value_w"])
    for index, layer in enumerate(module.deep_w):
        assert not np.allclose(layer, before[f"deep_w_{index}"])
        assert not np.allclose(module.deep_rec[index], before[f"deep_rec_{index}"])
        assert not np.allclose(module.deep_b[index], before[f"deep_b_{index}"])
        assert f"deep_w_{index}" in module.rnn_optimizer_m
        assert f"deep_rec_{index}" in module.rnn_optimizer_m
        assert f"deep_b_{index}" in module.rnn_optimizer_m

    batch_events = [event for event in policy.growth_events if event.get("type") == "direct_rnn_full_rl_batch"]
    assert len(batch_events) == 1
    assert batch_events[0]["depth_backprop"] is True
    assert batch_events[0]["temporal_bptt_steps"] == 2
    assert batch_events[0]["action_gradient_enabled"] is True
    block_update = batch_events[0]["direct_rnn_block_updates"][module.name]
    assert block_update["schema_version"] == 1
    assert block_update["fingerprint_before"] != block_update["fingerprint_after"]
    blocks = block_update["blocks"]
    assert blocks["encoder"]["delta_l2"] > 0.0
    assert blocks["action_head"]["delta_l2"] > 0.0
    assert blocks["value_head"]["delta_l2"] > 0.0
    for index in range(5):
        assert blocks[f"hidden_layer_{index + 1}"]["delta_l2"] > 0.0
    # Fixed V13 geometry rows are still part of the fingerprint surface but
    # must remain byte-identical under the sole Full-RL AdamW writer.
    geometry = blocks["geometry_compatibility_head"]
    assert geometry["fingerprint_before"] == geometry["fingerprint_after"]
    assert geometry["delta_l2"] == 0.0
    assert geometry["changed_parameter_count"] == 0
    # The compact nested telemetry must survive candidate cloning; it cannot
    # be coerced back to a flat float-only optimizer stats dictionary.
    copied = policy.copy()
    copied_blocks = copied.modules[0].rnn_optimizer_last_stats["direct_rnn_blocks"]
    assert copied_blocks["fingerprint_after"] == block_update["fingerprint_after"]


def test_full_lifecycle_rl_retains_the_exact_forward_cache_for_dropout_and_gate() -> None:
    policy = _full_rl_policy()
    module = policy.modules[0]
    module.hidden_dropout_enabled = True
    module.hidden_dropout_rates = (0.10, 0.10, 0.10, 0.10, 0.10, 0.10)
    module.hidden_dropout_seed = 2202
    module.hidden_dropout_train_only = True

    trace = policy.act(np.linspace(-0.2, 0.5, INPUT_COUNT, dtype=float))
    forward = trace.module_action_traces[0].forward_trace

    assert forward is not None
    assert len(forward.deep_inputs) == 5
    assert len(forward.dropout_scales) == 6
    assert 0.0 <= forward.gate <= 1.0
    assert forward.output_mask.shape == (OUTPUT_COUNT,)
    assert not np.shares_memory(forward.output_hidden, module.last_output_hidden)
    # Full-RL must optimize the masked entry forward, not an inference-only
    # cache.  This also makes a disabled/zero-step dropout path fail loudly.
    assert trace.action_details["hidden_dropout_training_active"] is True
    assert trace.action_details["hidden_dropout_steps_advanced"] == {module.name: 1}
    assert module.hidden_dropout_step == 1
    assert any(not np.allclose(scale, 1.0) for scale in forward.dropout_scales)


def test_frozen_full_rl_keeps_the_same_dropout_action_surface_without_writer() -> None:
    """Frozen/online MT5 must differ only by parameter updates, not dropout."""

    online = _full_rl_policy()
    frozen = _full_rl_policy()
    frozen._learning_enabled = False
    online.reset_runtime_state()
    frozen.reset_runtime_state()
    observation = np.linspace(-0.2, 0.5, INPUT_COUNT, dtype=float)

    online_trace = online.act(observation)
    frozen_trace = frozen.act(observation)
    online_forward = online_trace.module_action_traces[0].forward_trace
    frozen_forward = frozen_trace.module_action_traces[0].forward_trace

    assert online_forward is not None and frozen_forward is not None
    assert online_trace.action_details["hidden_dropout_training_active"] is True
    assert frozen_trace.action_details["hidden_dropout_training_active"] is True
    assert online_trace.action_details["hidden_dropout_steps_advanced"] == {online.modules[0].name: 1}
    assert frozen_trace.action_details["hidden_dropout_steps_advanced"] == {frozen.modules[0].name: 1}
    assert online.modules[0].hidden_dropout_step == frozen.modules[0].hidden_dropout_step == 1
    np.testing.assert_allclose(online_trace.selected_action, frozen_trace.selected_action, atol=1e-12)
    np.testing.assert_allclose(online_forward.output_hidden, frozen_forward.output_hidden, atol=1e-12)
    assert any(not np.allclose(scale, 1.0) for scale in frozen_forward.dropout_scales)

    frozen.reset_runtime_state()
    assert frozen.modules[0].hidden_dropout_step == 0


def test_frozen_full_rl_lifecycle_consumes_only_transient_trace_state() -> None:
    """A frozen settlement cannot fall back to legacy neuron/evidence writers."""

    policy = _full_rl_policy()
    policy.ensure_compatible()
    policy._learning_enabled = False
    module = policy.modules[0]
    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    tracked = (
        "neuron_utility",
        "neuron_advantage_ema",
        "neuron_contribution_ema",
        "neuron_activation_ema",
        "neuron_redundancy_ema",
        "neuron_samples",
    )
    before = {name: np.asarray(getattr(module, name)).copy() for name in tracked}
    lifecycle_ema_before = float(getattr(policy, "_lifecycle_outcome_ema", 0.0))
    advantage_before = (
        float(getattr(policy, "_direct_rnn_rl_advantage_mean", 0.0)),
        float(getattr(policy, "_direct_rnn_rl_advantage_second_moment", 0.0)),
        int(getattr(policy, "_direct_rnn_rl_advantage_count", 0)),
    )
    policy._execution_trace_cache["frozen-lifecycle"] = trace

    credited = policy.apply_execution_lifecycle_credit(
        "frozen-lifecycle", -0.8, close_reason="SL", duration_bars=4.0
    )

    assert credited == 0
    assert "frozen-lifecycle" not in policy._execution_trace_cache
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(module, name), value)
    assert float(getattr(policy, "_lifecycle_outcome_ema", 0.0)) == lifecycle_ema_before
    assert (
        float(getattr(policy, "_direct_rnn_rl_advantage_mean", 0.0)),
        float(getattr(policy, "_direct_rnn_rl_advantage_second_moment", 0.0)),
        int(getattr(policy, "_direct_rnn_rl_advantage_count", 0)),
    ) == advantage_before
    assert not getattr(policy, "_direct_rnn_rl_batches", [])


def test_full_rl_disables_legacy_recovery_rerank_and_shadow_writers() -> None:
    """No V9 recovery state may drift beside the sole Full-RL AdamW writer."""

    policy = _full_rl_policy()
    # Deliberately turn the legacy flags on: the Full-RL contract itself, not
    # the YAML profile, must enforce the isolation boundary.
    policy.action_recovery_enabled = True
    policy.action_recovery_learning_enabled = True
    policy.ensure_compatible()
    before = pickle.dumps(policy.action_recovery_learner, protocol=pickle.HIGHEST_PROTOCOL)
    observation = np.linspace(-0.35, 0.55, INPUT_COUNT, dtype=float)

    trace = policy.act(observation)
    assert trace.action_details["learner_recovery_full_rl_disabled"] is True
    assert trace.action_details["learner_recovery_learning_enabled"] is False
    assert trace.action_details["shadow_learning_active"] is False
    assert pickle.dumps(policy.action_recovery_learner, protocol=pickle.HIGHEST_PROTOCOL) == before

    policy.observe_transition(trace, observation * 0.99, 0.0, 0.0, False)
    shadow = policy.observe_shadow_transition(trace, observation * 0.98, sequence=7)

    assert shadow["full_rl_legacy_recovery_disabled"] is True
    assert pickle.dumps(policy.action_recovery_learner, protocol=pickle.HIGHEST_PROTOCOL) == before


def test_full_rl_act_and_transition_do_not_persist_legacy_state_writers() -> None:
    """Reasoning, memory, curiosity and world model cannot drift Full-RL."""

    policy = _full_rl_policy()
    policy.ensure_compatible()
    components = (
        "reasoning_graph",
        "curiosity_system",
        "memory_supernet",
        "partitioned_memory",
        "trading_world_model",
        "action_recovery_learner",
    )
    before = {
        name: pickle.dumps(getattr(policy, name), protocol=pickle.HIGHEST_PROTOCOL)
        for name in components
    }
    module = policy.modules[0]
    usage_before = (module.gate_ema, module.activation_ema, module.lifecycle)
    observation = np.linspace(-0.30, 0.45, INPUT_COUNT, dtype=float)

    trace = policy.act(observation)
    signal = policy.observe_transition(trace, observation * 0.97, 0.0, 0.0, False)

    assert trace.action_details["reasoning_full_rl_disabled"] is True
    assert trace.action_details["causal_full_rl_disabled"] is True
    assert trace.action_details["v8_legacy_state_full_rl_disabled"] is True
    assert signal["full_rl_transition_writers_skipped"] == 1.0
    assert signal["full_rl_world_model_skipped"] == 1.0
    assert (module.gate_ema, module.activation_ema, module.lifecycle) == usage_before
    for name, state in before.items():
        assert pickle.dumps(getattr(policy, name), protocol=pickle.HIGHEST_PROTOCOL) == state


def test_full_rl_keeps_broker_confirmed_policy_close_trace_for_delayed_credit() -> None:
    """A close receipt is cache-only until its causal post-exit horizon settles."""

    policy = _full_rl_policy()
    observation = np.linspace(-0.25, 0.35, INPUT_COUNT, dtype=float)
    original = policy.act(observation)
    close_output = np.full(OUTPUT_COUNT, -1.0, dtype=float)
    close_output[8] = 1.0  # CLOSE_WORST
    trace = replace(
        original,
        selected_action=close_output,
        execution_details={
            "bridge_observed": True,
            "execution_confirmed": True,
            "memory_eligible": False,
            "action_feedback_eligible": False,
        },
    )

    signal = policy.observe_transition(trace, observation * 0.99, 0.0, 0.0, False)

    assert signal["memory_eligible"] == 0.0
    assert signal["full_rl_exit_trace_cached"] == 1.0
    assert trace.trace_id in policy._execution_trace_cache
    credited = policy.apply_execution_exit_credit(
        trace.trace_id,
        final_realized_r=-0.4,
        hold_to_horizon_r=-1.0,
        position_ticket="policy-close-cache",
        exit_kind="POLICY_CLOSE_WORST",
        horizon_bars=12,
    )
    assert credited == 0  # The first delayed manager sample is queued, not yet a batch.
    assert len(policy._direct_rnn_exit_rl_batches) == 1
    assert policy.growth_events[-1]["type"] == "exit_counterfactual_credit_ack"


def test_full_rl_skips_position_extrema_legacy_memory_writer() -> None:
    """Runner extrema telemetry cannot grow memory after a Full-RL entry."""

    policy = _full_rl_policy()
    policy.ensure_compatible()
    before_memory = pickle.dumps(policy.memory_supernet, protocol=pickle.HIGHEST_PROTOCOL)
    before_extrema = pickle.dumps(
        getattr(policy, "position_context_extrema_memory", []),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    result = policy.record_position_context_extrema({
        "position_ticket": "full-rl-extrema",
        "owner_trace_id": "entry-trace",
        "side": 1,
        "high": {"unrealized_r": 0.7, "observation": np.zeros(policy.input_size).tolist()},
        "low": {"unrealized_r": -0.6, "observation": np.zeros(policy.input_size).tolist()},
    })

    assert result["full_rl_position_extrema_memory_skipped"] is True
    assert pickle.dumps(policy.memory_supernet, protocol=pickle.HIGHEST_PROTOCOL) == before_memory
    assert pickle.dumps(getattr(policy, "position_context_extrema_memory", []), protocol=pickle.HIGHEST_PROTOCOL) == before_extrema


def test_conditioned_representation_head_uses_only_the_full_rl_adamw_batch() -> None:
    policy = _full_rl_policy()
    policy.direct_rnn_rl_conditioned_representation_enabled = True
    policy.direct_rnn_rl_conditioned_representation_loss_weight = 0.10
    policy.direct_rnn_rl_conditioned_representation_min_abs_r = 0.10
    policy.direct_rnn_rl_conditioned_representation_requires_actor_evidence = True
    policy.direct_rnn_rl_conditioned_representation_conflict_mode = "pcgrad_v1"
    policy.ensure_compatible()
    module = policy.modules[0]
    before_w = module.conditioned_quality_w.copy()
    before_b = module.conditioned_quality_b.copy()
    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))

    updated = policy._apply_direct_rnn_full_rl_batch([{
        "trace": trace,
        "outcome": 0.80,
        "advantage": 0.80,
        "update_scale": 1.0,
    }])

    event = policy.growth_events[-1]
    assert updated == 1
    assert event["conditioned_representation_enabled"] is True
    assert event["conditioned_representation_eligible_count"] == 1
    assert event["conditioned_representation_positive_count"] == 1
    assert event["conditioned_representation_loss"] > 0.0
    assert event["conditioned_representation_head_delta_norms"][module.name]["weights"] > 0.0
    assert np.linalg.norm(module.conditioned_quality_w - before_w) > 0.0
    assert np.linalg.norm(module.conditioned_quality_b - before_b) > 0.0
    assert "conditioned_quality_w" in module.rnn_optimizer_m
    # The private head is not an inference output or independent optimizer.
    rejected = module.apply_rnn_adamw_cosine(
        {"conditioned_quality_w": np.ones_like(module.conditioned_quality_w)},
        source="legacy_local_credit",
    )
    assert rejected["updated"] == 0.0


def test_full_lifecycle_rl_suppresses_non_creditable_action_overlays() -> None:
    policy = _full_rl_policy()

    def forced_prior(count: int):
        prior = np.zeros(count, dtype=float)
        prior[2] = 1.0
        return prior, {"context_direction_prior_applied": True}

    policy._context_direction_prior = forced_prior
    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    details = trace.action_details

    assert int(np.argmax(trace.selected_action[:12])) == 1
    assert details["direct_rnn_rl_policy_action_index"] == 1
    assert details["direct_rnn_rl_policy_eligible"] is True
    assert int(np.argmax(np.asarray(details["direct_rnn_rl_logits"]))) == 1
    assert details["context_direction_prior_full_rl_suppressed"] is True


def test_full_rl_asymmetric_advantage_shields_a_net_win_below_critic_expectation() -> None:
    policy = _full_rl_policy()
    policy.direct_rnn_rl_advantage_mode = "critic_asymmetric_v1"
    policy.direct_rnn_rl_critic_warmup_lifecycles = 1
    policy.direct_rnn_rl_critic_max_weight = 1.0
    policy.direct_rnn_rl_profit_floor = 0.05
    policy.direct_rnn_rl_loss_floor = 0.05
    policy.direct_rnn_rl_advantage_normalization_enabled = True
    policy.direct_rnn_rl_advantage_norm_scale_floor = 0.35
    policy._direct_rnn_rl_advantage_count = 1
    policy.ensure_compatible()

    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    trace.action_details["direct_rnn_rl_value_prediction"] = 0.20
    info = policy._direct_rnn_full_rl_advantage(trace, 0.10, ema_baseline=0.0)

    assert info["advantage_mode"] == "critic_asymmetric_v1"
    assert info["baseline"] == 0.20
    assert info["raw_advantage"] == -0.10
    assert info["profit_shielded"] is True
    assert info["actor_advantage_pre_normalization"] == 0.0
    assert info["actor_advantage"] == 0.0


def test_full_rl_normalizes_only_actor_advantage_and_keeps_raw_value_target() -> None:
    policy = _full_rl_policy()
    policy.direct_rnn_rl_advantage_mode = "critic_asymmetric_v1"
    policy.direct_rnn_rl_critic_warmup_lifecycles = 1
    policy.direct_rnn_rl_critic_max_weight = 1.0
    policy.direct_rnn_rl_advantage_normalization_enabled = True
    policy.direct_rnn_rl_advantage_norm_scale_floor = 0.35
    policy.direct_rnn_rl_advantage_norm_clip = 2.5
    policy._direct_rnn_rl_advantage_count = 8
    policy._direct_rnn_rl_advantage_mean = 0.0
    policy._direct_rnn_rl_advantage_second_moment = 0.25
    policy.ensure_compatible()

    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    trace.action_details["direct_rnn_rl_value_prediction"] = 0.0
    info = policy._direct_rnn_full_rl_advantage(trace, -1.0, ema_baseline=0.0)

    assert info["raw_advantage"] == -1.0
    assert info["advantage_scale"] == 0.5
    assert info["actor_advantage_pre_normalization"] == -1.0
    assert info["actor_advantage"] == -2.0

    # Recreate the known historical scale because the direct helper above
    # intentionally updates the causal residual statistic after it emits its
    # signal.
    policy._direct_rnn_rl_advantage_count = 8
    policy._direct_rnn_rl_advantage_mean = 0.0
    policy._direct_rnn_rl_advantage_second_moment = 0.25
    policy._execution_trace_cache["normalized-raw-target"] = trace
    policy.apply_execution_lifecycle_credit(
        "normalized-raw-target", -1.0, close_reason="SL", duration_bars=8.0
    )
    queued = [event for event in policy.growth_events if event.get("type") == "direct_rnn_full_rl_queued"][-1]
    assert queued["raw_advantage"] == -1.0
    assert queued["advantage"] == -2.0
    assert queued["advantage_normalization_enabled"] is True
    assert policy._direct_rnn_rl_batches[-1]["outcome"] == -1.0
    assert policy._direct_rnn_rl_batches[-1]["actor_credit_scale"] == 1.0


def test_full_rl_scales_entry_actor_credit_for_policy_managed_exit() -> None:
    policy = _full_rl_policy()
    policy.direct_rnn_rl_batch_size = 8
    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    policy._execution_trace_cache["managed-exit"] = trace

    policy.apply_execution_lifecycle_credit(
        "managed-exit", -0.4, close_reason="POLICY_CLOSE_WORST", duration_bars=8.0
    )

    assert policy._direct_rnn_rl_batches[-1]["actor_credit_scale"] == 0.35


def test_zero_timing_lifecycle_scale_disables_pnl_writer() -> None:
    policy = _full_rl_policy()
    policy.timing_network_enabled = True
    policy.timing_network_lifecycle_scale = 0.0
    policy.ensure_compatible()
    trace = policy.act(np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float))
    before = int(getattr(policy.timing_network, "update_count", 0) or 0)
    policy._execution_trace_cache["timing-zero"] = trace

    policy.apply_execution_lifecycle_credit(
        "timing-zero", 0.8, close_reason="TP", duration_bars=8.0
    )

    assert int(getattr(policy.timing_network, "update_count", 0) or 0) == before
    assert not [
        event for event in policy.growth_events
        if event.get("type") == "timing_network_lifecycle_credit"
        and event.get("trace_id") == trace.trace_id
    ]


def test_bounded_entry_behavior_policy_never_samples_management_or_illegal_rows() -> None:
    target = np.asarray([0.70, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01, 0.90, 0.90])
    legal = np.asarray([True, True, False, True, True, True, True, True, True])

    selected, behavior, sampled = bounded_entry_behavior_policy(
        target,
        legal,
        greedy_action=0,
        exploration_rate=1.0,
        min_action_probability=0.02,
        exploration_draw=0.0,
        action_draw=0.99,
    )

    assert sampled is True
    assert selected in {1, 3, 4, 5, 6}
    assert behavior[2] == 0.0
    assert np.all(behavior[7:] == 0.0)
    assert np.isclose(float(np.sum(behavior)), 1.0)


def test_full_rl_exploration_samples_entry_action_and_is_reproducible_after_reset() -> None:
    policy = _full_rl_policy()
    policy.direct_rnn_rl_exploration_enabled = True
    policy.direct_rnn_rl_exploration_rate = 1.0
    policy.direct_rnn_rl_exploration_min_action_probability = 0.02
    policy.direct_rnn_rl_exploration_importance_clip = 2.0
    policy.direct_rnn_rl_exploration_seed = 90210
    policy._context_expert_owner_metadata = {
        "expert_id": "exploration-test",
        "ma_context": "RANGING",
        "management_only": False,
    }
    policy.ensure_compatible()
    observation = np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float)

    first = policy.act(observation)
    first_details = first.action_details
    first_action = int(np.argmax(first.selected_action[:12]))

    assert first_details["direct_rnn_rl_greedy_action_index"] == 1
    assert first_details["direct_rnn_rl_exploration_active"] is True
    assert first_details["direct_rnn_rl_exploration_sampled"] is True
    assert first_action == first_details["direct_rnn_rl_policy_action_index"]
    assert first_action in range(7)
    assert first_action != 1
    assert first_details["direct_rnn_rl_behavior_probability"] > 0.0
    assert 0.0 <= first_details["direct_rnn_rl_exploration_importance_weight"] <= 2.0
    assert np.allclose(
        np.asarray(first_details["direct_rnn_rl_behavior_probabilities"])[7:], 0.0
    )

    policy.reset_runtime_state()
    repeated = policy.act(observation)
    assert repeated.action_details["direct_rnn_rl_exploration_step"] == 0
    assert repeated.action_details["direct_rnn_rl_policy_action_index"] == first_action
    assert repeated.action_details["direct_rnn_rl_exploration_draw"] == first_details[
        "direct_rnn_rl_exploration_draw"
    ]


def test_full_rl_exploration_is_disabled_for_frozen_and_management_routes() -> None:
    policy = _full_rl_policy()
    policy.direct_rnn_rl_exploration_enabled = True
    policy.direct_rnn_rl_exploration_rate = 1.0
    policy._context_expert_owner_metadata = {
        "expert_id": "exploration-isolation",
        "ma_context": "RANGING",
        "management_only": False,
    }
    policy.ensure_compatible()
    observation = np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float)

    policy._learning_enabled = False
    frozen = policy.act(observation)
    assert frozen.action_details["direct_rnn_rl_exploration_active"] is False
    assert frozen.action_details["direct_rnn_rl_exploration_reason"] == "frozen_or_oos"
    assert frozen.action_details["direct_rnn_rl_policy_action_index"] == frozen.action_details[
        "direct_rnn_rl_greedy_action_index"
    ]

    policy.reset_runtime_state()
    policy._learning_enabled = True
    policy._context_expert_owner_metadata["management_only"] = True
    management = policy.act(observation)
    assert management.action_details["direct_rnn_rl_exploration_active"] is False
    assert management.action_details["direct_rnn_rl_exploration_reason"] == "management_isolated"
    assert management.action_details["direct_rnn_rl_policy_action_index"] == management.action_details[
        "direct_rnn_rl_greedy_action_index"
    ]
