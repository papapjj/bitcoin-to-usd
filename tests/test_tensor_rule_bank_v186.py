from __future__ import annotations

from dataclasses import replace
import pickle

import numpy as np
import pytest

from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    _v186_accumulate_preference_item,
    _v186_training_samples,
)
from self_growing_trader.config import LearningIntegrityConfig
from self_growing_trader.tensor_rule_bank import (
    TENSOR_RULE_ARTIFACT_VERSION,
    TENSOR_RULE_PARAMETER_PREFIX,
    TENSOR_RULE_SCHEMA_VERSION,
    TensorRuleBankState,
    TensorRuleTargets,
)


def _bank(seed: int = 7, *, effective_batch_size: int = 4) -> TensorRuleBankState:
    bank = TensorRuleBankState.seed(seed=seed, effective_batch_size=effective_batch_size)
    bank.rule_support[:2] = 64.0
    bank.rule_action_support[:2] = 32.0
    return bank


def _sequence(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.25, size=(32, 78))


def test_tensor_rule_config_is_disabled_by_default_and_validates_transformer_geometry() -> None:
    config = LearningIntegrityConfig()
    assert config.tensor_rule_enabled is False
    assert config.tensor_rule_authority == "telemetry"
    assert config.tensor_rule_sequence_length == 32
    assert config.tensor_rule_input_size == 78
    with pytest.raises(ValueError, match="divisible"):
        LearningIntegrityConfig(tensor_rule_hidden_size=62, tensor_rule_attention_heads=4)


def test_bank_uses_real_qkv_attention_and_has_expected_fixed_capacity_shape() -> None:
    bank = _bank()
    params = bank.parameters()
    assert params[f"{TENSOR_RULE_PARAMETER_PREFIX}block_0.q_w"].shape == (64, 64)
    assert params[f"{TENSOR_RULE_PARAMETER_PREFIX}block_0.k_w"].shape == (64, 64)
    assert params[f"{TENSOR_RULE_PARAMETER_PREFIX}block_0.v_w"].shape == (64, 64)
    assert bank.prototype.shape == (16, 32, 64)
    assert bank.active_mask.sum() == 2
    assert 130_000 <= bank.parameter_count() <= 140_000
    assert bank.structure_w.shape == bank.timing_w.shape == (64, 16)
    assert bank.regime_w.shape == bank.risk_w.shape == (64, 16)
    assert bank.action_adapter_w.shape == (6, 80, 32)
    assert bank.action_value_w.shape == (6, 32)


def test_component_width_is_independent_from_transformer_hidden_width() -> None:
    bank = TensorRuleBankState.seed(seed=8, component_size=12, action_latent_size=20)
    bank.rule_support[:2] = 64.0
    bank.rule_action_support[:2] = 32.0
    result = bank.forward(_sequence(8), 32, training=True)
    assert result.auxiliary_logits.shape == (131,)
    assert bank.auxiliary_w.shape == (131, 48)
    assert bank.action_adapter_w.shape == (6, 60, 20)


def test_invalid_old_rows_are_masked_and_cannot_change_current_abstraction() -> None:
    bank = _bank()
    sequence = _sequence()
    first = bank.forward(sequence, 16)
    changed = sequence.copy()
    changed[:16] += 10_000.0
    second = bank.forward(changed, 16)
    np.testing.assert_allclose(first.summary, second.summary, atol=1e-10)
    np.testing.assert_allclose(first.memberships, second.memberships, atol=1e-10)
    assert first.input_fingerprint == second.input_fingerprint


def test_causal_attention_prevents_future_rows_from_changing_encoded_prefix() -> None:
    bank = _bank()
    sequence = _sequence()
    first, *_ = bank._transformer_forward(sequence, 32)
    changed = sequence.copy()
    changed[21:] += 100.0
    second, *_ = bank._transformer_forward(changed, 32)
    np.testing.assert_allclose(first[:21], second[:21], atol=1e-10)


def test_input_contract_fingerprint_mismatch_fails_closed() -> None:
    bank = _bank()
    result = bank.forward(
        _sequence(), 32, input_schema_fingerprint="wrong-feature-order"
    )
    assert result.unknown is True
    assert result.unknown_reason == "input_fingerprint_mismatch"
    assert not np.any(result.memberships)


def test_qkv_backward_matches_finite_difference_for_summary_objective() -> None:
    bank = _bank()
    sequence = _sequence()
    direction = np.linspace(-0.2, 0.2, bank.hidden_size)
    forward = bank.forward(sequence, 32, training=True)
    gradients = bank._transformer_backward(
        np.zeros((bank.sequence_length, bank.hidden_size)), direction, forward
    )
    analytical = float(gradients["block_0.q_w"][0, 0])
    epsilon = 1e-5
    original = float(bank.blocks[0]["q_w"][0, 0])
    bank.blocks[0]["q_w"][0, 0] = original + epsilon
    plus = float(np.dot(bank.forward(sequence, 32).summary, direction))
    bank.blocks[0]["q_w"][0, 0] = original - epsilon
    minus = float(np.dot(bank.forward(sequence, 32).summary, direction))
    bank.blocks[0]["q_w"][0, 0] = original
    numerical = (plus - minus) / (2.0 * epsilon)
    assert analytical == pytest.approx(numerical, rel=2e-2, abs=2e-3)


def test_soft_membership_and_unknown_are_explicit() -> None:
    bank = _bank()
    cold = bank.forward(_sequence(), 8)
    assert cold.unknown is True
    assert cold.unknown_reason == "insufficient_history"
    warm = bank.forward(_sequence(), 32)
    assert np.sum(warm.memberships) == pytest.approx(1.0)
    assert np.count_nonzero(warm.memberships) == 2
    assert warm.action_lcb_r.shape == (6,)


def test_rule_gradients_accumulate_until_effective_batch() -> None:
    bank = _bank(effective_batch_size=4)
    forward = bank.forward(_sequence(), 32, training=True)
    targets = TensorRuleTargets(entry_utilities=np.asarray([0.0, 1.0, -1.0, 0.2, -0.2, 0.5, -0.5]))
    gradients, _ = bank.sample_gradients(forward, targets)
    assert bank.accumulate_gradients(gradients, sample_count=2) == {}
    ready = bank.accumulate_gradients(gradients, sample_count=2)
    assert ready
    assert bank.gradient_sample_count == 0
    assert bank.update_count == 1


def test_realized_lifecycle_credit_is_action_local_and_does_not_move_encoder() -> None:
    bank = _bank()
    bank.action_value_w[2] = np.linspace(-0.2, 0.2, bank.action_latent_size)
    forward = bank.forward(_sequence(61), 32, training=True)
    targets = TensorRuleTargets(selected_action=3, realized_r=-1.0)
    gradients, _ = bank.sample_gradients(forward, targets)

    assert np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_value_w"][2])
    assert np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_adapter_w"][2])
    assert not np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_value_w"][:2])
    assert not np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_value_w"][3:])
    assert not np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_adapter_w"][:2])
    assert not np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_adapter_w"][3:])
    for name in ("input_w", "block_0.q_w", "block_0.k_w", "block_0.v_w", "structure_w"):
        assert not np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}{name}"])


def test_counterfactual_utility_can_train_action_latents_and_shared_encoder() -> None:
    bank = _bank()
    bank.action_value_w[:] = np.linspace(-0.1, 0.1, bank.action_latent_size)
    forward = bank.forward(_sequence(62), 32, training=True)
    targets = TensorRuleTargets(
        entry_utilities=np.asarray([0.0, 0.8, -0.8, 0.4, -0.4, 0.2, -0.2])
    )
    gradients, _ = bank.sample_gradients(forward, targets)
    assert np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_value_w"])
    assert np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}action_adapter_w"])
    assert np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}input_w"])


def test_auxiliary_label_family_updates_only_its_semantic_component() -> None:
    bank = _bank()
    forward = bank.forward(_sequence(63), 32, training=True)
    values = np.zeros_like(TensorRuleTargets().auxiliary_values)
    mask = np.zeros_like(TensorRuleTargets().auxiliary_mask)
    first_action_fill_index = 25
    values[first_action_fill_index] = 1.0
    mask[first_action_fill_index] = True
    gradients, _ = bank.sample_gradients(
        forward, TensorRuleTargets(auxiliary_values=values, auxiliary_mask=mask)
    )
    assert np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}timing_w"])
    for name in ("structure_w", "regime_w", "risk_w"):
        assert not np.any(gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}{name}"])
    auxiliary_gradient = gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}auxiliary_w"]
    semantic_mask = bank._auxiliary_component_mask()
    assert not np.any(auxiliary_gradient[semantic_mask == 0.0])


def test_policy_bank_is_legacy_safe_independent_and_uses_same_adamw_writer() -> None:
    legacy = ChuyenGiaPolicy.seed(input_size=137, output_size=20, hidden_size=16, seed=1)
    legacy.ensure_compatible()
    assert legacy.modules[0].tensor_rule_bank is None

    settings = LearningIntegrityConfig(
        tensor_rule_enabled=True,
        tensor_rule_effective_batch_size=1,
        direct_rnn_adamw_enabled=True,
        direct_rnn_full_rl_enabled=True,
    )
    left = ChuyenGiaPolicy.seed(input_size=137, output_size=20, hidden_size=16, seed=2)
    right = ChuyenGiaPolicy.seed(input_size=137, output_size=20, hidden_size=16, seed=3)
    left._context_expert_owner_metadata = {"expert_id": "left", "context_id": "CTX_TANG_MANH"}
    right._context_expert_owner_metadata = {"expert_id": "right", "context_id": "CTX_GIAM_MANH"}
    left.configure_learning_integrity(settings)
    right.configure_learning_integrity(settings)
    left_bank = left.modules[0].tensor_rule_bank
    right_bank = right.modules[0].tensor_rule_bank
    assert isinstance(left_bank, TensorRuleBankState)
    assert isinstance(right_bank, TensorRuleBankState)
    assert left_bank is not right_bank
    assert not np.array_equal(left_bank.input_w, right_bank.input_w)
    assert left.artifact_version >= TENSOR_RULE_ARTIFACT_VERSION
    parameters = left.modules[0]._rnn_optimizer_parameters()
    assert f"{TENSOR_RULE_PARAMETER_PREFIX}prototype" in parameters
    before = left_bank.rule_action_mean.copy()
    pending = np.ones_like(left_bank.rule_action_mean) * 0.01
    left.modules[0]._tensor_rule_pending_directions = {
        f"{TENSOR_RULE_PARAMETER_PREFIX}rule_action_mean": pending
    }
    result = left.modules[0].apply_rnn_adamw_cosine(
        {}, source="full_lifecycle_rl"
    )
    assert result["updated"] == 1.0
    assert not np.array_equal(before, left_bank.rule_action_mean)
    np.testing.assert_array_equal(right_bank.rule_action_mean, np.zeros_like(right_bank.rule_action_mean))


def test_cold_tensor_rule_gate_blocks_greedy_entry_until_lcb_is_known() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=137, output_size=20, hidden_size=16, seed=21)
    policy.configure_learning_integrity(LearningIntegrityConfig(
        tensor_rule_enabled=True,
        tensor_rule_authority="residual",
        tensor_rule_entry_lcb_gate_enabled=True,
        direct_rnn_adamw_enabled=True,
        direct_rnn_full_rl_enabled=True,
    ))
    module = policy.modules[0]
    module.w_out.fill(0.0)
    module.b_out.fill(0.0)
    module.b_out[1] = 2.0

    trace = policy.act(np.linspace(-0.4, 0.6, 137, dtype=float))

    assert trace.action_details["direct_rnn_rl_greedy_action_index"] == 1
    assert int(np.argmax(trace.selected_action[:7])) == 0
    assert trace.action_details["tensor_rule_entry_lcb_gate_applied"] is True
    assert trace.action_details["tensor_rule_entry_lcb_gate_reason"] == "unknown_hold"
    assert trace.action_details["direct_rnn_rl_policy_eligible"] is False


def test_cold_tensor_rule_gate_preserves_exploration_entry_provenance() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=137, output_size=20, hidden_size=16, seed=22)
    policy.configure_learning_integrity(LearningIntegrityConfig(
        tensor_rule_enabled=True,
        tensor_rule_authority="residual",
        tensor_rule_entry_lcb_gate_enabled=True,
        direct_rnn_adamw_enabled=True,
        direct_rnn_full_rl_enabled=True,
        direct_rnn_rl_exploration_enabled=True,
        direct_rnn_rl_exploration_rate=1.0,
        direct_rnn_rl_exploration_min_action_probability=0.02,
    ))
    policy.direct_rnn_rl_exploration_enabled = True
    policy.direct_rnn_rl_exploration_rate = 1.0
    policy.direct_rnn_rl_exploration_min_action_probability = 0.02
    policy.direct_rnn_rl_exploration_seed = 90210
    policy._learning_enabled = True
    policy._context_expert_owner_metadata = {
        "expert_id": "cold-gate-exploration",
        "ma_context": "RANGING",
        "management_only": False,
    }
    module = policy.modules[0]
    module.w_out.fill(0.0)
    module.b_out.fill(0.0)
    module.b_out[1] = 2.0

    trace = policy.act(np.linspace(-0.4, 0.6, 137, dtype=float))
    selected = int(np.argmax(trace.selected_action[:7]))

    assert trace.action_details["direct_rnn_rl_exploration_sampled"] is True
    assert selected == trace.action_details["direct_rnn_rl_policy_action_index"]
    assert selected in range(1, 7)
    assert trace.action_details["tensor_rule_entry_lcb_gate_applied"] is False
    assert trace.action_details["tensor_rule_entry_lcb_gate_reason"] == "exploration_entry"
    assert trace.action_details["direct_rnn_rl_behavior_probability"] > 0.0


def test_rule_topology_is_frozen_inside_round_and_fixed_shape_after_split() -> None:
    bank = _bank()
    bank.rule_support[0] = 160.0
    shape = bank.prototype.shape
    blocked = bank.evolve_at_round_boundary(
        {"split_loss_improvement": 0.10, "split_child_min_support": 40, "split_lcb_delta": 0.0},
        round_boundary=False,
    )
    assert blocked["changed"] is False
    changed = bank.evolve_at_round_boundary(
        {"split_loss_improvement": 0.10, "split_child_min_support": 40, "split_lcb_delta": 0.0},
        round_boundary=True,
    )
    assert changed["operation"] == "split"
    assert bank.prototype.shape == shape
    assert bank.active_mask.sum() == 3


def test_bank_save_load_is_deterministic_and_keeps_source_fingerprint() -> None:
    bank = TensorRuleBankState.seed(seed=44, seed_fingerprint="source-sha256")
    sequence = _sequence(44)
    expected = bank.forward(sequence, 32)
    first = pickle.dumps(bank, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(first)
    restored.ensure_compatible()
    second = pickle.dumps(restored, protocol=pickle.HIGHEST_PROTOCOL)
    assert second == pickle.dumps(restored, protocol=pickle.HIGHEST_PROTOCOL)
    observed = restored.forward(sequence, 32)
    np.testing.assert_array_equal(expected.summary, observed.summary)
    np.testing.assert_array_equal(expected.memberships, observed.memberships)
    assert restored.seed_fingerprint == "source-sha256"


def test_schema1_migration_resets_incompatible_aux_rows_but_keeps_rule_values() -> None:
    bank = _bank(seed=45)
    bank.schema_version = 1
    bank.rule_action_mean[0, 1] = 0.75
    bank.auxiliary_w = np.full((58, 64), 7.0)
    bank.auxiliary_b = np.full(58, 7.0)
    for name in (
        "structure_w", "structure_b", "timing_w", "timing_b",
        "regime_w", "regime_b", "risk_w", "risk_b", "action_embedding",
        "action_adapter_w", "action_adapter_b", "action_value_w", "action_value_b",
    ):
        setattr(bank, name, None)
    bank.ensure_compatible()
    assert bank.schema_version == TENSOR_RULE_SCHEMA_VERSION
    assert bank.auxiliary_w.shape == (131, 64)
    assert not np.all(bank.auxiliary_w[:58] == 7.0)
    assert not np.any(bank.auxiliary_b)
    assert bank.rule_action_mean[0, 1] == pytest.approx(0.75)


def test_schema5_migration_preserves_old_auxiliary_rows_and_adds_timing_rows() -> None:
    bank = _bank(seed=46)
    bank.schema_version = 5
    old_w = np.arange(77 * 64, dtype=float).reshape(77, 64) / 1000.0
    old_b = np.arange(77, dtype=float) / 100.0
    bank.auxiliary_w = old_w.copy()
    bank.auxiliary_b = old_b.copy()
    bank.ensure_compatible()
    assert bank.auxiliary_w.shape == (131, 64)
    assert bank.auxiliary_b.shape == (131,)
    np.testing.assert_allclose(bank.auxiliary_w[:25], old_w[:25])
    np.testing.assert_allclose(bank.auxiliary_b[:25], old_b[:25])
    for action_offset in range(6):
        old_start = 25 + action_offset * 8
        new_start = 25 + action_offset * 17
        np.testing.assert_allclose(bank.auxiliary_w[new_start:new_start + 8], old_w[old_start:old_start + 8])
        np.testing.assert_allclose(bank.auxiliary_b[new_start:new_start + 8], old_b[old_start:old_start + 8])
    np.testing.assert_allclose(bank.auxiliary_w[127:131], old_w[73:77])
    np.testing.assert_allclose(bank.auxiliary_b[127:131], old_b[73:77])
    assert np.isfinite(bank.auxiliary_w).all()
    assert not np.any(bank.action_value_w)
    assert bank.utility_prototype.shape == (16, 7)
    assert not hasattr(bank, "utility_rule_assignments")


def test_soft_utility_prototypes_balance_opposite_outcomes_without_hard_rules() -> None:
    bank = _bank(seed=46)
    samples = []
    for kind in (0, 1):
        utilities = (
            np.asarray([0.0, 1.0, -0.8, 0.1, -0.2, 0.0, -0.1])
            if kind == 0 else
            np.asarray([0.0, -0.8, 1.0, 0.1, -0.2, 0.0, -0.1])
        )
        for sample in range(4):
            sequence = _sequence(4_600 + kind * 10 + sample)
            sequence[-16:, 2:12] = (1.0 if kind == 0 else -1.0) * 0.5
            samples.append((sequence, 32, TensorRuleTargets(entry_utilities=utilities)))

    gradients, metrics = bank.batch_gradients(samples)

    assert f"{TENSOR_RULE_PARAMETER_PREFIX}utility_prototype" in gradients
    assert np.linalg.norm(
        gradients[f"{TENSOR_RULE_PARAMETER_PREFIX}utility_prototype"]
    ) > 0.0
    assert metrics["utility_assignment_min_usage"] >= 0.40
    assert metrics["utility_assignment_max_usage"] <= 0.60
    assert not hasattr(bank, "utility_rule_assignments")


def test_dense_auxiliary_objective_is_mean_normalized_by_label_count() -> None:
    bank = _bank(seed=93)
    sequence = _sequence(seed=94)
    values = np.zeros_like(TensorRuleTargets().auxiliary_values)
    one_mask = np.zeros_like(TensorRuleTargets().auxiliary_mask)
    one_mask[0] = True
    many_mask = np.zeros_like(one_mask)
    many_mask[:12] = True

    _one_gradients, one_metrics = bank.batch_gradients([
        (sequence, 32, TensorRuleTargets(auxiliary_values=values, auxiliary_mask=one_mask))
    ])
    _many_gradients, many_metrics = bank.batch_gradients([
        (sequence, 32, TensorRuleTargets(auxiliary_values=values, auxiliary_mask=many_mask))
    ])

    assert one_metrics["auxiliary_target_count"] == 1.0
    assert many_metrics["auxiliary_target_count"] == 12.0
    assert np.isfinite(one_metrics["auxiliary_loss"])
    assert np.isfinite(many_metrics["auxiliary_loss"])
    assert many_metrics["auxiliary_loss"] < 4.0 * one_metrics["auxiliary_loss"]


def test_ready_tensor_gradient_has_own_budget_before_shared_adamw() -> None:
    bank = _bank(seed=95, effective_batch_size=4)
    bank.ready_gradient_norm_budget = 0.50
    raw = {
        f"{TENSOR_RULE_PARAMETER_PREFIX}prototype": np.full_like(bank.prototype, 20.0),
        f"{TENSOR_RULE_PARAMETER_PREFIX}utility_prototype": np.full_like(
            bank.utility_prototype, -10.0
        ),
    }

    ready = bank.accumulate_gradients(raw, sample_count=4)
    ready_norm = np.sqrt(sum(float(np.sum(value * value)) for value in ready.values()))

    assert bank.last_ready_gradient_raw_norm > bank.ready_gradient_norm_budget
    assert 0.0 < bank.last_ready_gradient_clip_factor < 1.0
    assert ready_norm == pytest.approx(bank.ready_gradient_norm_budget)


def test_delayed_preference_builds_tensor_direction_without_an_optimizer_write() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=137, output_size=20, hidden_size=16, seed=96)
    policy.configure_learning_integrity(LearningIntegrityConfig(
        tensor_rule_enabled=True,
        tensor_rule_effective_batch_size=64,
        direct_rnn_adamw_enabled=True,
        direct_rnn_full_rl_enabled=True,
    ))
    bank = policy.modules[0].tensor_rule_bank
    module = policy.modules[0]
    trace = policy.act(np.linspace(-0.2, 0.3, 137, dtype=float))
    trace = replace(trace, timing_trace={
        **dict(trace.timing_trace),
        "sequence_tensor": _sequence(96),
        "sequence_valid_length": 32,
    })
    items = []
    for index in range(64):
        utilities = np.asarray([0.0, 0.8, -0.6, 0.2, -0.2, 0.1, -0.1])
        if index % 2:
            utilities = utilities[[0, 2, 1, 3, 4, 5, 6]]
        item = {"trace": trace, "entry_preference_only": True,
                "entry_preference_utilities": utilities}
        items.append(item)
        result = _v186_accumulate_preference_item(policy, item)
    assert result["optimizer_directions_ready"] is True
    assert result["optimizer_write_deferred_to_lifecycle"] is True
    assert module.rnn_optimizer_step == 0
    assert module._tensor_rule_pending_directions
    assert all(item["tensor_rule_preaccumulated"] for item in items)
    assert _v186_training_samples(bank, items) == []


def test_preaccumulated_preference_is_excluded_from_tensor_training() -> None:
    bank = _bank(seed=97)
    item = {
        "tensor_rule_preaccumulated": True,
        "entry_preference_only": True,
        "entry_preference_utilities": np.asarray([0.0, 0.8, -0.6, 0.2, -0.2, 0.1, -0.1]),
    }
    assert _v186_training_samples(bank, [item], include_ready_regime=False) == []


def test_temporal_fold_lcb_is_reported_per_rule_and_action() -> None:
    bank = _bank()
    samples = []
    for fold in range(3):
        for seed in range(6):
            targets = TensorRuleTargets(
                entry_utilities=np.asarray([0.0, 0.8, -0.4, 0.2, -0.2, 0.1, -0.1]),
                temporal_fold=fold,
            )
            samples.append((_sequence(100 + fold * 10 + seed), 32, targets))
    bank.batch_gradients(samples)
    summary = bank.summary()
    assert set(summary["temporal_fold_lcb95_r"]) == {"0", "1"}
    assert len(summary["temporal_fold_lcb95_r"]["0"]) == 3
    assert summary["temporal_fold_lcb95_r"]["0"][0][0] is not None


def test_uncertainty_calibration_uses_supported_temporal_cells() -> None:
    bank = _bank()
    bank.fold_action_return_support[:2] = 25.0
    bank.fold_action_return_sum[0] = 25.0 * 0.30
    bank.fold_action_return_sum[1] = 25.0 * -0.20
    bank.fold_action_return_sq_sum[0] = 25.0 * 0.09
    bank.fold_action_return_sq_sum[1] = 25.0 * 0.04
    result = bank.calibrate_uncertainty()
    assert result["updated"] is True
    assert result["calibration_rounds"] == 1
    np.testing.assert_allclose(bank.calibration_error, np.full(6, 0.30))


def test_synthetic_motifs_abstract_across_time_and_split_opposite_utility() -> None:
    bank = _bank(seed=101, effective_batch_size=12)
    # UNKNOWN threshold is independently tested above.  This test isolates
    # whether unseen temporal folds preserve the learned utility abstraction.
    bank.unknown_membership = 0.50

    def motif(kind: int, fold: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        values = rng.normal(0.0, 0.01, size=(32, 78))
        values[-16:, 0] = np.linspace(-1.0, 1.0, 16)
        values[-16:, 1] = np.sin(np.linspace(0.0, np.pi, 16))
        # Same temporal motif, but a causal context subspace reverses its
        # utility.  A snapshot memorizer tends to merge these examples.
        values[-16:, 2:12] = (1.0 if kind == 0 else -1.0) * 0.50
        values[-16:, 12] = fold * 0.01
        return values

    def target(kind: int, fold: int) -> TensorRuleTargets:
        utilities = (
            np.asarray([0.0, 1.0, -0.8, 0.1, -0.2, 0.0, -0.1])
            if kind == 0 else
            np.asarray([0.0, -0.8, 1.0, 0.1, -0.2, 0.0, -0.1])
        )
        return TensorRuleTargets(entry_utilities=utilities, temporal_fold=fold)

    parameters = bank.parameters()
    moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    variance = {name: np.zeros_like(value) for name, value in parameters.items()}
    for step in range(90):
        samples = [
            (
                motif(kind, fold, 1_000 + step * 100 + fold * 10 + sample + kind * 40),
                32,
                target(kind, fold),
            )
            for kind in (0, 1)
            for fold in (0, 1)
            for sample in range(3)
        ]
        gradients, _ = bank.batch_gradients(samples)
        for name, direction in gradients.items():
            direction = direction / len(samples)
            moment[name] = 0.9 * moment[name] + 0.1 * direction
            variance[name] = 0.999 * variance[name] + 0.001 * direction * direction
            corrected_m = moment[name] / (1.0 - 0.9 ** (step + 1))
            corrected_v = variance[name] / (1.0 - 0.999 ** (step + 1))
            parameters[name] += 0.003 * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        bank.sanitize()

    # Fold 2 is an unseen temporal segment.  Both motifs must reuse their
    # learned abstraction, remain non-UNKNOWN, and preserve action ranking.
    first = bank.forward(motif(0, 2, 9_000), 32)
    second = bank.forward(motif(1, 2, 9_001), 32)
    assert first.rule_id != second.rule_id
    assert first.unknown is False and second.unknown is False
    assert int(np.argmax(first.action_mean_r)) + 1 == 1
    assert int(np.argmax(second.action_mean_r)) + 1 == 2
    for fold in (0, 1):
        assert bank.forward(motif(0, fold, 9_100 + fold), 32).rule_id == first.rule_id
        assert bank.forward(motif(1, fold, 9_200 + fold), 32).rule_id == second.rule_id


def test_authority_promotion_fails_closed_without_oos_proof() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=137, output_size=20, hidden_size=16, seed=5)
    policy.configure_learning_integrity(LearningIntegrityConfig(tensor_rule_enabled=True))
    bank = policy.modules[0].tensor_rule_bank
    bank.rule_support[:2] = 60.0
    bank.temporal_fold_support[:2] = 12.0
    bank.rule_action_support[:2] = 32.0
    residual = policy.set_tensor_rule_authority("residual")
    assert residual["changed"] is True
    primary = policy.set_tensor_rule_authority("primary", evidence={})
    assert primary["changed"] is False
    assert primary["reason"] == "promotion_gate_failed"


def test_regime_targets_mature_only_after_h2_h4_h6_h8() -> None:
    bank = _bank()
    sequence = _sequence()
    assert bank.observe_regime_decision(sequence, 32, "CTX_TANG_MANH", sequence_id=0) == 0
    for index in range(1, 32):
        assert bank.observe_regime_decision(sequence, 32, "CTX_TANG_MANH", sequence_id=index) == 0
    matured = bank.observe_regime_decision(sequence, 32, "CTX_RANGING", sequence_id=32)
    assert matured >= 1
    ready = bank.pop_ready_regime()
    assert ready
    target = ready[0][2]
    np.testing.assert_array_equal(target.auxiliary_mask[-4:], np.ones(4, dtype=bool))
