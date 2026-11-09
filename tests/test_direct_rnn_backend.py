from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest

from self_growing_trader.config import SupernetworkConfig, load_config
from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    tao_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.m1_graph_module import M1GraphModule
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.self_growing_runner import ModuleState, SelfGrowingPolicy, load_policy


class ForwardBomb:
    def forward(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("direct_rnn must not call M1 forward")

    def copy(self):
        return self


def test_direct_rnn_does_not_call_m1_forward_and_outputs_bridge_shape() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=101)
    policy.modules[0].m1_component = ForwardBomb()

    output = policy.activate(np.zeros(INPUT_COUNT))
    trace = policy.act(np.zeros(INPUT_COUNT))

    assert policy.m1_backend == "direct_rnn"
    assert policy.decision_backend == "direct_rnn"
    assert output.shape == (OUTPUT_COUNT,)
    assert trace.selected_action.shape == (OUTPUT_COUNT,)
    assert trace.candidate_actions.shape == (1, OUTPUT_COUNT)
    assert np.isfinite(output).all()
    assert np.isfinite(trace.selected_action).all()
    assert trace.active_edge_ids == ()
    assert trace.active_neuron_ids == ()


def test_direct_rnn_ensure_does_not_recreate_m1_component() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=102)
    policy.modules[0].m1_component = None

    policy.ensure_compatible()

    assert policy.modules[0].m1_component is None
    assert policy._graph_m1_modules() == []


def test_old_graph_checkpoint_loads_as_direct_rnn_with_dormant_graph(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=103, m1_backend="m1_graph")
    assert isinstance(policy.modules[0].m1_component, M1GraphModule)
    delattr(policy, "decision_backend")
    policy.m1_backend = "graph"
    path = tmp_path / "old_graph.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    loaded = load_policy(path)

    assert loaded.m1_backend == "direct_rnn"
    assert loaded.decision_backend == "direct_rnn"
    assert isinstance(loaded.modules[0].m1_component, M1GraphModule)
    assert loaded._graph_m1_modules() == []
    assert loaded.act(np.zeros(INPUT_COUNT)).active_edge_ids == ()


def test_six_modules_activate_direct_rnn_without_bridge_schema_change() -> None:
    rng = np.random.default_rng(104)
    modules = [
        ModuleState.create(f"expert_{idx}", INPUT_COUNT, 8, OUTPUT_COUNT, rng, hidden_layers=2)
        for idx in range(6)
    ]
    policy = SelfGrowingPolicy(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, modules=modules)
    policy.ensure_compatible()

    output = policy.activate(np.linspace(-0.5, 0.5, INPUT_COUNT))

    assert len(policy.modules) == 6
    assert output.shape == (OUTPUT_COUNT,)
    assert np.isfinite(output).all()


def test_lifecycle_credit_changes_direct_rnn_head_and_output() -> None:
    obs = np.linspace(-1.0, 1.0, INPUT_COUNT)
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=10, seed=105)
    trace = policy.act(obs)
    trace = replace(trace, action_details={**trace.action_details, "action_attribution_source": "policy"})
    before_head = policy.modules[0].w_out.copy()
    policy._execution_trace_cache[trace.trace_id] = trace

    credited = policy.apply_execution_lifecycle_credit(trace.trace_id, 1.25, close_reason="TP", duration_bars=8)

    assert credited > 0
    assert not np.allclose(policy.modules[0].w_out, before_head)


def test_lifecycle_credit_respects_bounded_online_update_scale() -> None:
    obs = np.linspace(-1.0, 1.0, INPUT_COUNT)

    def credit_delta(scale: float) -> float:
        policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=10, seed=1061)
        policy._mt5_online_update_scale = scale
        trace = policy.act(obs)
        before_head = policy.modules[0].w_out.copy()
        policy._execution_trace_cache[trace.trace_id] = trace
        assert policy.apply_execution_lifecycle_credit(
            trace.trace_id, 1.25, close_reason="TP", duration_bars=8
        ) > 0
        return float(np.linalg.norm(policy.modules[0].w_out - before_head))

    full = credit_delta(1.0)
    bounded = credit_delta(0.1)
    assert full > 0.0
    assert bounded > 0.0
    assert bounded < full * 0.2


def test_direct_rnn_respects_allowed_action_validation_mask() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=106)
    policy._decision_context = {"allowed_actions": {0: True}}
    trace = policy.act(np.ones(INPUT_COUNT))

    assert int(trace.action_details["policy_output_action_index"]) == 0
    assert np.argmax(trace.selected_action[:12]) == 0


def test_direct_rnn_output_normalization_is_applied_before_emit() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=1062)
    policy.output_normalization_mode = "rms_v1"
    policy.output_normalization_target_rms = 0.05
    trace = policy.act(np.ones(INPUT_COUNT, dtype=float))
    details = trace.action_details
    assert details["output_normalization_mode"] == "rms_v1"
    assert details["output_action_rms_after"] <= details["output_action_rms_before"]
    assert details["output_action_rms_after"] <= 0.051


def test_direct_rnn_trace_keeps_dropout_out_of_live_broker_decision() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=1063)
    policy.hidden_dropout_enabled = True
    policy.hidden_dropout_rates = (0.80,)
    policy.hidden_dropout_seed = 991
    policy.hidden_dropout_train_only = True
    module = policy.modules[0]
    module.hidden_dropout_enabled = True
    module.hidden_dropout_rates = (0.80,)
    module.hidden_dropout_seed = 991
    module.hidden_dropout_train_only = True

    trace = policy.act(np.ones(INPUT_COUNT, dtype=float))

    assert trace.action_details["hidden_dropout_enabled"] is True
    assert trace.action_details["hidden_dropout_step"] == 0
    assert all(np.allclose(scale, 1.0) for scale in trace.module_action_traces[0].forward_trace.dropout_scales)
    assert np.allclose(trace.module_action_traces[0].hidden, module.last_output_hidden)


def test_direct_rnn_context_direction_prior_is_soft_and_keeps_all_entries() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=107)
    policy._context_expert_owner_metadata = {
        "context_task": "TANG_MANH",
        "timing_direction": 0,
        "timing_quality": 0.0,
        "do_tin_cay_context": 1.0,
        "context_direction_prior_strength": 0.10,
        "management_only": False,
    }

    prior, audit = policy._context_direction_prior(12)

    assert audit["context_direction_prior_applied"] is True
    assert audit["context_direction_prior"] == 1
    assert prior[1] > 0.0 and prior[3] > 0.0 and prior[5] > 0.0
    assert prior[2] < 0.0 and prior[4] < 0.0 and prior[6] < 0.0
    assert set(range(1, 7)).issubset({index for index, value in enumerate(prior) if value != 0.0})

    trace = policy.act(np.zeros(INPUT_COUNT))
    assert trace.action_details["context_direction_prior_applied"] is True
    assert trace.action_details["context_direction_prior_source"] == "context_task"


def test_direct_rnn_applies_expectancy_delta_to_discrete_action_logits() -> None:
    def run(delta: np.ndarray):
        policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=113)
        policy._query_memory = lambda _state: {
            "action_bias": np.zeros(OUTPUT_COUNT, dtype=float),
            "replacement_action_bias": np.zeros(OUTPUT_COUNT, dtype=float),
            "expectancy_action_delta": delta.copy(),
        }
        return policy._act_direct_rnn(np.zeros(INPUT_COUNT, dtype=float))

    neutral = run(np.zeros(OUTPUT_COUNT, dtype=float))
    delta = np.zeros(OUTPUT_COUNT, dtype=float)
    delta[1] = -0.75
    inhibited = run(delta)

    applied = inhibited.action_details["expectancy_action_delta_applied"]
    assert applied[1] == pytest.approx(-0.75)
    assert inhibited.action_details["expectancy_action_delta_norm"] == pytest.approx(0.75)
    assert inhibited.selected_action[1] < neutral.selected_action[1]


def test_direct_rnn_context_direction_prior_does_not_touch_management_owner() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=108)
    policy._context_expert_owner_metadata = {
        "context_task": "TANG_MANH",
        "context_direction_prior_strength": 0.10,
        "management_only": True,
    }

    prior, audit = policy._context_direction_prior(12)

    assert np.allclose(prior, 0.0)
    assert audit["context_direction_prior_applied"] is False


def test_direct_rnn_volatile_uses_nonzero_timing_direction_as_soft_evidence() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=109)
    policy._context_expert_owner_metadata = {
        "context_task": "VOLATILE",
        "timing_direction": -1,
        "timing_quality": 0.15,
        "do_tin_cay_context": 1.0,
        "context_direction_prior_strength": 0.10,
        "volatile_timing_direction_min_quality": 0.10,
        "management_only": False,
    }

    prior, audit = policy._context_direction_prior(12)

    assert audit["context_direction_prior_applied"] is True
    assert audit["context_direction_prior_source"] == "timing_direction"
    assert audit["context_direction_prior"] == -1
    assert prior[2] > 0.0 and prior[4] > 0.0 and prior[6] > 0.0
    assert prior[1] < 0.0 and prior[3] < 0.0 and prior[5] < 0.0


def test_direct_rnn_transition_candidate_direction_is_soft_evidence_when_timing_is_neutral() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=110)
    policy._context_expert_owner_metadata = {
        "context_task": "VOLATILE_EXPANSION",
        "candidate_context": "CTX_TANG_MANH",
        "context_transition_active": True,
        "context_transition_progress": 1.0 / 3.0,
        "timing_direction": 0,
        "timing_quality": 0.0,
        "do_tin_cay_context": 0.9,
        "context_direction_prior_strength": 0.25,
        "management_only": False,
    }

    prior, audit = policy._context_direction_prior(12)

    assert audit["context_direction_prior_applied"] is True
    assert audit["context_direction_prior_source"] == "candidate_context_transition"
    assert audit["context_direction_prior"] == 1
    assert audit["context_direction_prior_score"] == pytest.approx(0.125)
    assert prior[1] > 0.0 and prior[3] > 0.0 and prior[5] > 0.0
    assert prior[2] < 0.0 and prior[4] < 0.0 and prior[6] < 0.0


def test_active_profile_can_disable_noisy_transition_and_timing_priors_without_masking() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=111)
    policy._context_expert_owner_metadata = {
        "context_task": "VOLATILE_EXPANSION",
        "candidate_context": "CTX_TANG_MANH",
        "context_transition_active": True,
        "context_transition_progress": 1.0 / 3.0,
        "candidate_transition_direction_prior_bat": False,
        "volatile_timing_direction_prior_bat": False,
        "timing_direction": 1,
        "timing_quality": 0.80,
        "do_tin_cay_context": 0.9,
        "context_direction_prior_strength": 0.25,
        "management_only": False,
    }

    prior, audit = policy._context_direction_prior(12)

    assert np.allclose(prior, 0.0)
    assert audit["context_direction_prior_applied"] is False
    assert audit["context_direction_prior_source"] == "none"


def test_active_profile_does_not_apply_stable_context_prior_during_transition() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=112)
    policy._context_expert_owner_metadata = {
        "context_task": "GIAM_MANH",
        "candidate_context": "CTX_TANG_MANH",
        "context_transition_active": True,
        "context_transition_progress": 2.0 / 3.0,
        "candidate_transition_direction_prior_bat": False,
        "volatile_timing_direction_prior_bat": False,
        "stable_context_direction_prior_during_transition_bat": False,
        "timing_direction": 0,
        "timing_quality": 0.0,
        "do_tin_cay_context": 0.9,
        "context_direction_prior_strength": 0.25,
        "management_only": False,
    }

    prior, audit = policy._context_direction_prior(12)

    assert np.allclose(prior, 0.0)
    assert audit["context_direction_prior_applied"] is False
    assert audit["context_direction_prior_source"] == "none"


def test_config_defaults_keep_bridge_contract_and_direct_backend() -> None:
    config = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")

    assert config.model.input_count == INPUT_COUNT
    assert config.model.output_count == OUTPUT_COUNT
    assert config.he_chuyen_gia.context_direction_prior_strength == 0.0
    assert config.he_chuyen_gia.candidate_transition_direction_prior_bat is False
    assert config.he_chuyen_gia.volatile_timing_direction_prior_bat is False
    assert config.he_chuyen_gia.stable_context_direction_prior_during_transition_bat is False
    assert config.supernetwork.m1_backend == "direct_rnn"
    assert SupernetworkConfig().m1_backend == "direct_rnn"


def test_mt5_profile_khong_tai_duong_direct_rnn_cu_khi_dung_nam_tac_nhan() -> None:
    config = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=1091)
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    assert system.experts
    for expert in system.experts.values():
        policy = expert.policy
        assert policy.five_shared_network is not None
        assert policy.modules == []
        assert policy.direct_rnn_full_rl_enabled is False
        assert policy.timing_network is None
        assert policy.timing_network_enabled is False


def test_mt5_profile_uses_one_full_rl_batch_contract_for_every_root_owner() -> None:
    config = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=1092)
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)

    for expert in system.experts.values():
        policy = expert.policy
        assert policy.direct_rnn_rl_batch_size == 8
