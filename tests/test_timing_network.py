from __future__ import annotations

import numpy as np
import pickle
from types import SimpleNamespace

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy, tao_policy_chuyen_gia_doc_lap
from self_growing_trader.config import LearningIntegrityConfig
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.timing_network import (
    TIMING_NETWORK_INPUT_SIZE,
    TimingNetwork,
    extract_timing_features,
    timing_trend_score,
)


def test_timing_network_uses_only_versioned_market_tail_and_is_normalized() -> None:
    observation = np.zeros(INPUT_COUNT, dtype=float)
    observation[0] = 100.0
    observation[29] = -100.0  # position block must not enter the timing vector
    observation[117] = 4.0

    features = extract_timing_features(observation)
    assert features.shape == (TIMING_NETWORK_INPUT_SIZE,)
    assert features[0] == 100.0
    assert features[29] == 0.0  # market starts again after the position/account block
    assert features[-20] == 4.0

    network = TimingNetwork.seed(seed=201)
    output = network.forward(observation)
    assert output.features.shape == (TIMING_NETWORK_INPUT_SIZE,)
    assert np.max(np.abs(output.features)) <= 2.0
    assert 0.0 <= output.quality <= 1.0
    assert 0.0 <= output.entry_probability <= 1.0


def test_timing_network_materializes_a_causal_32_m15_tensor_and_rolls_it() -> None:
    network = TimingNetwork.seed(seed=2011, sequence_length=32)
    latest = None
    for step in range(33):
        observation = np.zeros(INPUT_COUNT, dtype=float)
        observation[0] = float(step + 1)
        latest = network.forward(observation)

    assert latest is not None
    assert latest.sequence_tensor.shape == (32, TIMING_NETWORK_INPUT_SIZE)
    assert latest.sequence_valid_length == 32
    # The first 32 M15 values were 1..32.  At step 33, 1 has rolled out and
    # the tensor contains exactly 2..33, oldest to newest.
    assert latest.sequence_tensor[0, 0] != 0.0
    assert latest.sequence_tensor[-1, 0] != latest.sequence_tensor[0, 0]
    trace = latest.trace_payload()
    assert trace["schema_version"] == 6
    assert trace["sequence_tensor"].shape == (32, TIMING_NETWORK_INPUT_SIZE)
    assert trace["sequence_valid_length"] == 32
    copied = network.copy()
    assert np.array_equal(copied.sequence_buffer, network.sequence_buffer)
    copied.reset_runtime_state()
    assert copied.sequence_valid_length == 0
    assert not np.any(copied.sequence_buffer)


def test_old_timingnet_checkpoint_migrates_to_tensor32_without_shape_failure() -> None:
    network = TimingNetwork.seed(seed=2012)
    delattr(network, "sequence_buffer")
    delattr(network, "sequence_valid_length")
    delattr(network, "sequence_length")

    loaded = pickle.loads(pickle.dumps(network))
    output = loaded.forward(np.zeros(INPUT_COUNT, dtype=float))

    assert loaded.sequence_length == 32
    assert output.sequence_tensor.shape == (32, TIMING_NETWORK_INPUT_SIZE)
    assert output.sequence_valid_length == 1


def test_timing_network_lifecycle_credit_changes_representation_and_heads() -> None:
    network = TimingNetwork.seed(seed=202)
    output = network.forward(np.linspace(-0.8, 0.8, INPUT_COUNT))
    before_in = network.w_in.copy()
    before_head = network.w_quality.copy()

    update = network.update_lifecycle_credit(
        output.trace_payload(),
        1.25,
        selected_action=1,
        duration_bars=8,
        close_reason="TP",
        update_scale=1.0,
    )

    assert update["updated"] == 1.0
    assert network.update_count == 1
    assert not np.allclose(network.w_in, before_in)
    assert not np.allclose(network.w_quality, before_head)


def test_timing_direction_credit_is_relative_to_selected_side() -> None:
    """A profitable SELL must teach down-direction, not up-direction."""

    long_net = TimingNetwork.seed(seed=2035, learning_rate=0.004)
    short_net = TimingNetwork.seed(seed=2035, learning_rate=0.004)
    long_trace = long_net.forward(np.linspace(-0.8, 0.8, INPUT_COUNT)).trace_payload()
    short_trace = short_net.forward(np.linspace(-0.8, 0.8, INPUT_COUNT)).trace_payload()

    long_net.update_lifecycle_credit(long_trace, 1.0, selected_action=1, update_scale=1.0)
    short_net.update_lifecycle_credit(short_trace, 1.0, selected_action=2, update_scale=1.0)

    assert long_net.last_update["direction_target"] > 0.0
    assert short_net.last_update["direction_target"] < 0.0
    assert long_net.b_direction > 0.0
    assert short_net.b_direction < 0.0


def test_timing_network_uses_frozen_recurrent_eligibility_and_reports_clipping() -> None:
    network = TimingNetwork.seed(seed=2021, bptt_window=4, gradient_clip=0.05)
    for step in range(6):
        output = network.forward(np.linspace(-0.8, 0.8, INPUT_COUNT) + step * 0.01)
    trace = output.trace_payload()
    assert trace["eligibility_in"].dtype == np.float32
    assert trace["eligibility_in"].shape == (network.hidden_size, network.hidden_size, network.input_size)
    before = network.w_rec.copy()
    update = network.update_lifecycle_credit(trace, 2.0, selected_action=1, duration_bars=8, close_reason="TP")
    assert update["bptt_steps"] == 4.0
    assert update["gradient_clip_factor"] < 1.0
    assert not np.allclose(network.w_rec, before)


def test_timing_network_old_trace_falls_back_to_one_step_chain_rule() -> None:
    network = TimingNetwork.seed(seed=2022)
    output = network.forward(np.linspace(-0.8, 0.8, INPUT_COUNT))
    old_trace = {
        "features": output.features,
        "state_before": output.state_before,
        "hidden": output.hidden,
        "embedding": output.embedding,
        "action_logits": output.action_logits,
    }
    update = network.update_lifecycle_credit(old_trace, -1.0, selected_action=1, close_reason="SL")
    assert update["updated"] == 1.0
    assert update["bptt_steps"] == 1.0


def test_disabled_timing_network_preserves_legacy_trace_and_copy() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=203)
    trace = policy.act(np.ones(INPUT_COUNT, dtype=float))
    assert trace.timing_trace == {}
    copied = policy.copy()
    assert copied.timing_network is None
    assert copied.timing_network_enabled is False


def test_legacy_policy_without_timing_field_migrates_safely() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=2031)
    delattr(policy, "timing_network")
    loaded = pickle.loads(pickle.dumps(policy))
    loaded.ensure_compatible()
    assert loaded.timing_network is None
    assert loaded.act(np.zeros(INPUT_COUNT)).timing_trace == {}


def test_enabled_timing_network_receives_delayed_policy_credit() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=204)
    policy.timing_network_enabled = True
    policy.timing_network_update_scale = 1.0
    policy.timing_network_lifecycle_scale = 1.0
    policy.ensure_compatible()
    module = policy.modules[0]
    module.w_out.fill(0.0)
    module.b_out.fill(-1.0)
    module.b_out[1] = 1.0
    trace = policy.act(np.linspace(-0.8, 0.8, INPUT_COUNT))
    assert trace.action_details["timing_network_enabled"] is True
    assert trace.timing_trace["schema_version"] == 6
    before = policy.timing_network.w_in.copy()
    policy._execution_trace_cache[trace.trace_id] = trace

    policy.apply_execution_lifecycle_credit(trace.trace_id, 1.25, close_reason="TP", duration_bars=8)

    assert policy.timing_network.update_count == 1
    assert not np.allclose(policy.timing_network.w_in, before)
    assert any(event.get("type") == "timing_network_lifecycle_credit" for event in policy.growth_events)
    follow_up = policy.act(np.linspace(-0.8, 0.8, INPUT_COUNT))
    assert follow_up.action_details["timing_network_action_scale"] > 0.0
    assert np.linalg.norm(np.asarray(follow_up.action_details["timing_network_action_delta"], dtype=float)) > 0.0


def test_direct_rnn_encoder_credit_requires_coherent_lifecycle_minibatch() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=2045)
    policy.input_lifecycle_encoder_learning_enabled = True
    policy.input_lifecycle_encoder_batch_size = 2
    policy.input_lifecycle_encoder_min_abs_advantage = 0.01
    policy.input_lifecycle_encoder_min_sign_coherence = 0.50
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_adamw_learning_rate = 0.001
    policy.direct_rnn_adamw_weight_decay = 0.0001
    policy.direct_rnn_cosine_min_factor = 0.20
    policy.direct_rnn_cosine_cycle_steps = 8
    policy.ensure_compatible()
    # Force one deterministic legal entry across every specialist. This keeps
    # the test focused on delayed encoder credit rather than on a random
    # training-time dropout mask changing the selected owner action.
    for module in policy.modules:
        module.w_out.fill(0.0)
        module.b_out.fill(-20.0)
        module.b_out[1] = 20.0
    before = policy.modules[0].w_in.copy()
    for step in range(2):
        trace = policy.act(np.linspace(-0.8, 0.8, INPUT_COUNT) + step * 0.01)
        assert int(trace.action_details["semantic_action"]) == 1
        policy._execution_trace_cache[trace.trace_id] = trace
        policy.apply_execution_lifecycle_credit(trace.trace_id, 1.0, close_reason="TP", duration_bars=8)
        if step == 0:
            assert np.allclose(policy.modules[0].w_in, before)
    assert not np.allclose(policy.modules[0].w_in, before)
    events = [event for event in policy.growth_events if event.get("type") == "input_lifecycle_encoder_credit"]
    assert events
    assert events[-1]["optimizer_effective_learning_rate"] > 0.0
    assert policy.modules[0].rnn_optimizer_step >= 1


def test_direct_rnn_adamw_cosine_has_state_and_decreases_after_peak() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=2046)
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_adamw_learning_rate = 0.001
    policy.direct_rnn_cosine_min_factor = 0.20
    policy.direct_rnn_cosine_cycle_steps = 8
    policy.ensure_compatible()
    module = policy.modules[0]
    direction = np.ones_like(module.w_in)

    first = module.apply_rnn_adamw_cosine({"w_in": direction}, source="test")
    second = module.apply_rnn_adamw_cosine({"w_in": direction}, source="test")

    assert first["updated"] == 1.0
    assert second["optimizer_step"] == first["optimizer_step"] + 1.0
    assert second["optimizer_effective_learning_rate"] < first["optimizer_effective_learning_rate"]
    assert module.rnn_optimizer_m["w_in"].shape == module.w_in.shape
    assert module.rnn_optimizer_v["w_in"].shape == module.w_in.shape


def test_timing_network_config_is_candidate_opt_in() -> None:
    config = LearningIntegrityConfig()
    assert config.timing_network_enabled is False
    assert config.timing_network_lifecycle_scale == 0.0
    candidate = LearningIntegrityConfig(
        timing_network_enabled=True,
        timing_network_action_scale=0.15,
        timing_network_lifecycle_scale=0.50,
    )
    assert candidate.timing_network_lifecycle_scale == 0.50


def test_timing_network_learns_causal_deceleration_without_future_leakage() -> None:
    network = TimingNetwork.seed(seed=2041, learning_rate=0.004)
    network.configure_deceleration({
        "timing_network_deceleration_enabled": True,
        "timing_network_deceleration_horizon": 1,
        "timing_network_deceleration_min_trend_strength": 0.10,
        "timing_network_deceleration_min_drop": 0.08,
        "timing_network_deceleration_update_scale": 1.0,
    })
    strong_bear = np.zeros(INPUT_COUNT, dtype=float)
    strong_bear[[0, 1, 2, 3, 4, 5, 11, 12, 21, 22, 25, 26, 27, 32, 33, 36, 37, 40, 41, 47, 48]] = -3.0
    assert timing_trend_score(strong_bear) < -0.10
    first = network.forward(strong_bear, learn_deceleration=True)
    before_head = network.w_deceleration.copy()
    neutral = np.zeros(INPUT_COUNT, dtype=float)
    second = network.forward(neutral, learn_deceleration=True)

    assert first.deceleration_update_count == 0
    assert second.deceleration_update_count == 1
    assert network.deceleration_positive_count == 1
    assert network.deceleration_last_target == 1.0
    assert not np.allclose(network.w_deceleration, before_head)


def test_timing_network_learns_delayed_multi_horizon_regime_transition() -> None:
    network = TimingNetwork.seed(seed=20415)
    network.configure_regime_transition({
        "timing_network_regime_transition_enabled": True,
        "timing_network_regime_transition_horizons": (1, 2),
        "timing_network_regime_transition_update_scale": 1.0,
        "timing_network_regime_transition_learning_rate": 0.02,
    })
    observation = np.linspace(-0.4, 0.4, INPUT_COUNT)
    before = network.w_regime_transition.copy()
    first = network.forward(observation, regime_label="CTX_TANG_MANH", learn_regime_transition=True)
    second = network.forward(observation, regime_label="CTX_RANGING", learn_regime_transition=True)
    third = network.forward(observation, regime_label="CTX_RANGING", learn_regime_transition=True)
    assert first.regime_transition_update_count == 0
    assert second.regime_transition_update_count >= 1
    assert third.regime_transition_update_count >= 2
    assert network.regime_transition_last_target == "CTX_RANGING"
    assert not np.allclose(network.w_regime_transition, before)


def test_deceleration_bce_head_moves_probability_without_using_lifecycle_lr() -> None:
    """Dense causal labels must calibrate a cold zero head in a practical sample size."""
    network = TimingNetwork.seed(seed=20411, learning_rate=0.0005)
    network.configure_deceleration({
        "timing_network_deceleration_enabled": True,
        "timing_network_deceleration_horizon": 1,
        "timing_network_deceleration_min_trend_strength": 0.10,
        "timing_network_deceleration_min_drop": 0.08,
        "timing_network_deceleration_update_scale": 1.0,
        "timing_network_deceleration_learning_rate": 0.02,
        # Isolate the logistic head in this test; the production profile uses
        # a small encoder share and L2, checked by configuration wiring.
        "timing_network_deceleration_encoder_scale": 0.0,
        "timing_network_deceleration_l2": 1.0e-4,
    })
    strong_bear = np.zeros(INPUT_COUNT, dtype=float)
    strong_bear[[0, 1, 2, 3, 4, 5, 11, 12, 21, 22, 25, 26, 27, 32, 33, 36, 37, 40, 41, 47, 48]] = -3.0
    neutral = np.zeros(INPUT_COUNT, dtype=float)
    for _ in range(100):
        network.forward(strong_bear, learn_deceleration=True)
        network.forward(neutral, learn_deceleration=True)

    network.reset_runtime_state()
    for _ in range(3):
        output = network.forward(strong_bear, learn_deceleration=False)

    assert network.learning_rate == 0.0005
    assert network.deceleration_update_count == 100
    assert output.deceleration_probability > 0.65


def test_frozen_timing_network_does_not_settle_deceleration_labels() -> None:
    network = TimingNetwork.seed(seed=2042)
    network.configure_deceleration({
        "timing_network_deceleration_enabled": True,
        "timing_network_deceleration_horizon": 1,
    })
    network.forward(np.ones(INPUT_COUNT, dtype=float), learn_deceleration=False)
    network.forward(np.zeros(INPUT_COUNT, dtype=float), learn_deceleration=False)
    assert network.deceleration_update_count == 0
    assert network.deceleration_pending == []


def test_late_born_context_policy_inherits_timing_network_contract() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(hidden_size=8, hidden_layers=1),
        learning=LearningIntegrityConfig(timing_network_enabled=True, timing_network_lifecycle_scale=0.5),
    )
    factory = tao_policy_chuyen_gia_doc_lap(config, seed=205)
    policy = factory()
    assert policy.timing_network_enabled is True
    assert policy.timing_network_lifecycle_scale == 0.5
    assert policy.timing_network is not None
    assert policy.timing_network.bptt_window == 24


def test_late_born_context_policy_inherits_pid_swats_contract() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(hidden_size=8, hidden_layers=1),
        learning=LearningIntegrityConfig(
            timing_network_enabled=True,
            timing_network_swats_enabled=True,
            timing_network_pid_enabled=True,
            timing_network_adamw_enabled=True,
            timing_network_adamw_weight_decay=0.001,
            timing_network_cosine_annealing_enabled=True,
            timing_network_cosine_min_factor=0.40,
            timing_network_cosine_cycle_steps=32,
            timing_network_pid_r_scale_floor=0.25,
        ),
    )
    factory = tao_policy_chuyen_gia_doc_lap(config, seed=208)
    policy = factory()
    assert policy.timing_network is not None
    assert policy.timing_network.swats_enabled is True
    assert policy.timing_network.pid_enabled is True
    assert policy.timing_network.adamw_enabled is True
    assert policy.timing_network.adamw_weight_decay == 0.001
    assert policy.timing_network.cosine_annealing_enabled is True
    assert policy.timing_network.cosine_min_factor == 0.40
    assert policy.timing_network.cosine_cycle_steps == 32
    assert policy.timing_network.optimizer_phase == "adam"


def test_pid_swats_keeps_shadow_adam_and_switches_only_timingnet() -> None:
    network = TimingNetwork.seed(seed=206, learning_rate=0.001)
    network.configure_optimizer({
        "timing_network_swats_enabled": True,
        "timing_network_pid_enabled": True,
        "timing_network_min_projection_steps": 1,
        "timing_network_switch_consecutive": 1,
        "timing_network_target_alignment": -1.0,
        "timing_network_projection_tolerance": 1.0,
        "timing_network_pid_cooldown_steps": 0,
    })
    output = network.forward(np.linspace(-0.8, 0.8, INPUT_COUNT))
    before_m = network.shadow_adam_m.copy()
    result = network.update_lifecycle_credit(output.trace_payload(), 1.0, selected_action=1, update_scale=1.0)

    assert result["updated"] == 1.0
    assert network.shadow_adam_step == 1
    assert np.linalg.norm(network.shadow_adam_m - before_m) > 0.0
    assert network.optimizer_phase == "sgd"
    assert network.switch_count == 1
    copied = network.copy()
    assert np.array_equal(copied.shadow_adam_m, network.shadow_adam_m)
    assert copied.optimizer_phase == "sgd"
    assert copied.last_update["optimizer_phase_name"] == "sgd"


def test_adamw_cosine_schedule_is_lifecycle_clocked_and_excludes_deceleration_head() -> None:
    network = TimingNetwork.seed(seed=209, learning_rate=0.01)
    network.w_deceleration.fill(1.0)
    network.configure_optimizer({
        "timing_network_swats_enabled": True,
        "timing_network_adamw_enabled": True,
        "timing_network_adamw_weight_decay": 0.10,
        "timing_network_cosine_annealing_enabled": True,
        "timing_network_cosine_min_factor": 0.20,
        "timing_network_cosine_cycle_steps": 2,
        # Keep this test in Adam so cosine remains the active LR schedule.
        "timing_network_min_projection_steps": 128,
    })
    zero_gradients = {
        name: np.zeros_like(values)
        for name, values in network._parameter_arrays()
    }
    before_representation = float(np.linalg.norm(network.w_in))
    before_deceleration = network.w_deceleration.copy()
    first = network._optimizer_update(zero_gradients, learning_rate=0.01, realized_r=0.0)
    second = network._optimizer_update(zero_gradients, learning_rate=0.01, realized_r=0.0)

    assert first["optimizer_adam_cosine_factor"] == 1.0
    assert np.isclose(second["optimizer_adam_cosine_factor"], 0.60)
    assert np.isclose(second["optimizer_effective_learning_rate"], 0.006)
    assert second["optimizer_decayed_parameters"] > 0.0
    assert float(np.linalg.norm(network.w_in)) < before_representation
    # The causal BCE head has its own optimizer and must not get R-based decay.
    assert np.array_equal(network.w_deceleration, before_deceleration)


def test_pid_swats_skips_state_update_for_invalid_trace() -> None:
    network = TimingNetwork.seed(seed=207)
    network.configure_optimizer({"timing_network_swats_enabled": True, "timing_network_pid_enabled": True})
    result = network.update_lifecycle_credit({}, -1.0, selected_action=1)
    assert result == {"updated": 0.0}
    assert network.shadow_adam_step == 0
    assert network.pid_signal_count == 0
