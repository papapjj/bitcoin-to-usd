from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.run_tensor_rule_v186_candidate import (
    build_candidate_feedback_contract,
    candidate_config,
    validate_candidate_feedback_contract,
)
from self_growing_trader.config import load_config
from self_growing_trader.execution_contract import OUTPUT_COUNT
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FIVE_REASONING_ACTION_COUNT,
    FactorizedActionHeads,
    FivePPOTrainer,
    SharedFiveTradingPolicy,
)
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.named_parameters()
    }


def test_internal_policy_state_is_shared_causal_and_changes_action_logits() -> None:
    torch.manual_seed(29)
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.require_current_architecture()
    observations = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
    memories = network.initial_memories()

    _, _, baseline = network.forward_five(
        observations, memories, deterministic=True
    )
    latent = baseline["internal_policy_state"]
    assert latent.shape == (FIVE_AGENT_COUNT, 8)
    # The internal policy state is a frame-level representation shared by the
    # five authority replicas, not five externally assigned strategy labels.
    torch.testing.assert_close(latent, latent[0:1].expand_as(latent))

    with torch.no_grad():
        network.heads.internal_policy_state_net.action_residual[-1].bias[1] = 0.75
    _, _, changed = network.forward_five(
        observations, memories, deterministic=True
    )
    assert not torch.allclose(
        changed["action_logits"], baseline["action_logits"], rtol=0.0, atol=1e-8
    )
    assert changed["internal_policy_state"].shape[-1] == 8


def test_internal_policy_state_is_trainable_by_five_adam() -> None:
    torch.manual_seed(31)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.require_current_architecture()
    trainer = FivePPOTrainer(network)
    hidden = torch.randn(FIVE_AGENT_COUNT, 8)
    logits, latent = network.heads.apply_internal_policy_state(
        torch.zeros(FIVE_AGENT_COUNT, ACTION_COUNT), hidden
    )
    before = network.heads.internal_policy_state_net.action_residual[-1].bias.detach().clone()
    # This is the same signed actor direction used by PPO: increasing the
    # selected action logit is a positive-advantage update.
    loss = -logits[:, 1].mean()
    trainer.step_from_loss(loss)
    after = network.heads.internal_policy_state_net.action_residual[-1].bias.detach()
    assert torch.isfinite(loss)
    assert not torch.equal(after, before)
    assert latent.shape == (FIVE_AGENT_COUNT, 8)


def test_reasoning_candidate_is_neutral_when_action_scale_is_zero() -> None:
    torch.manual_seed(17)
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.require_current_architecture()
    assert "reasoning_head" not in network._modules
    assert "reasoning_reference_head" not in network._modules
    observations = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
    memories = network.initial_memories()

    network.reasoning_rl_enabled = False
    baseline, _, baseline_tensors = network.forward_five(
        observations, memories, deterministic=True
    )
    network.reasoning_rl_enabled = True
    network.reasoning_steps = 3
    network.reasoning_action_scale = 0.0
    candidate, _, candidate_tensors = network.forward_five(
        observations, memories, deterministic=True
    )

    assert [item.action_index for item in candidate] == [
        item.action_index for item in baseline
    ]
    for before, after in zip(baseline, candidate):
        np.testing.assert_array_equal(after.output_vector, before.output_vector)
        np.testing.assert_array_equal(
            after.policy_action_probabilities,
            before.policy_action_probabilities,
        )
        assert len(after.reasoning_trace["steps"]) == 3
        assert after.reasoning_trace["training_labels_visible"] is False
    torch.testing.assert_close(
        candidate_tensors["action_logits"], baseline_tensors["action_logits"],
        rtol=0.0, atol=0.0,
    )


def test_reasoning_active_scale_changes_public_action_logits() -> None:
    torch.manual_seed(19)
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.require_current_architecture()
    observations = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
    memories = network.initial_memories()

    network.reasoning_rl_enabled = False
    _, _, baseline_tensors = network.forward_five(
        observations, memories, deterministic=True
    )
    network.reasoning_rl_enabled = True
    network.reasoning_steps = 3
    network.reasoning_action_scale = 0.25
    _, _, active_tensors = network.forward_five(
        observations, memories, deterministic=True
    )

    # The residual is bounded but no longer telemetry-only: it must reach the
    # public seven-choice action logits that drive the sampled action.
    assert not torch.allclose(
        active_tensors["action_logits"][:, :FIVE_REASONING_ACTION_COUNT],
        baseline_tensors["action_logits"][:, :FIVE_REASONING_ACTION_COUNT],
        rtol=0.0,
        atol=1e-8,
    )


def test_reasoning_active_candidate_profile_enables_action_and_optimizer() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/reasoning_rl_v1_active_candidate.yaml"
    ).read_text(encoding="utf-8"))
    assert profile["app_overrides"]["mt5_learning_enabled"] is True
    overrides = profile["learning_overrides"]
    assert overrides["five_reasoning_rl_enabled"] is True
    assert overrides["five_reasoning_action_scale"] == pytest.approx(0.25)
    assert overrides["five_behavior_uniform_mix"] == pytest.approx(0.15)
    assert overrides["five_expectancy_action_veto_hold_owner_on_negative"] is False
    assert overrides["five_live_action_veto_hold_owner_on_negative"] is False
    assert overrides["five_reasoning_encoder_credit_scale"] == pytest.approx(0.0)
    assert overrides["five_rlhf_recovery_snapshot_interval_updates"] == 8
    assert profile["gradient_contract"]["sole_writer"] == "five_ppo_adam"


def test_reasoning_candidate_runtime_contract_rejects_telemetry_only_wiring() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/reasoning_rl_v1_active_candidate.yaml"
    ).read_text(encoding="utf-8"))
    base = load_config("configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml")
    config = candidate_config(base, profile, Path("outputs/_contract_test"))
    contract = build_candidate_feedback_contract(config, profile)

    assert contract["expected_feedback_to_learner"] is True
    assert contract["effective"]["mt5_learning_enabled"] is True
    assert contract["effective"]["five_reasoning_action_scale"] == pytest.approx(0.25)
    assert contract["effective"]["five_codex_teacher_auto_connect"] is True
    assert contract["effective"][
        "five_rlhf_recovery_snapshot_interval_updates"
    ] == 8
    assert contract["effective"]["five_codex_teacher_online_wait_sec"] == pytest.approx(120.0)
    assert contract["effective"]["five_expectancy_action_veto_enabled"] is True
    assert contract["effective"]["five_expectancy_action_veto_min_samples"] == 6
    assert contract["effective"]["five_live_action_veto_enabled"] is True
    assert contract["effective"]["five_live_action_veto_min_samples"] == 6
    assert contract["effective"]["five_live_action_veto_probe_enabled"] is True
    assert contract["effective"]["five_live_action_veto_probe_on_negative_transition"] is True
    assert contract["effective"]["five_live_action_veto_probe_interval_decisions"] == 16
    assert contract["effective"]["five_live_owner_exploration_enabled"] is True
    validate_candidate_feedback_contract(contract)

    broken = dict(contract)
    broken["violations"] = ["profile_requires_mt5_learning_enabled=true"]
    with pytest.raises(RuntimeError, match="would not update the learner"):
        validate_candidate_feedback_contract(broken)


def test_codex_teacher_candidate_runtime_contract_enables_close_feedback() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/codex_teacher_every_close_candidate.yaml"
    ).read_text(encoding="utf-8"))
    base = load_config("configs/he_chuyen_gia_codex_teacher_candidate.yaml")
    config = candidate_config(base, profile, Path("outputs/_codex_contract_test"))
    contract = build_candidate_feedback_contract(config, profile)

    validate_candidate_feedback_contract(contract)
    assert contract["effective"]["mt5_learning_enabled"] is True
    assert contract["effective"]["five_codex_teacher_enabled"] is True
    assert contract["effective"]["five_codex_teacher_online_wait_sec"] == pytest.approx(120.0)
    assert contract["effective"]["five_expectancy_action_veto_enabled"] is True
    assert contract["effective"]["five_live_action_veto_enabled"] is True
    assert contract["effective"]["five_live_action_veto_probe_enabled"] is True
    assert contract["effective"]["five_live_action_veto_probe_on_negative_transition"] is True
    assert contract["effective"]["five_live_owner_exploration_enabled"] is True
    assert contract["negative_close_route"] == "codex_teacher_receipt_to_five_adam"


def test_codex_teacher_specialist_contract_exposes_full_rl_route() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/reasoning_rl_v1_specialist_smoke.yaml"
    ).read_text(encoding="utf-8"))
    base = load_config("configs/he_chuyen_gia_reasoning_rl_v1_specialist_smoke.yaml")
    config = candidate_config(base, profile, Path("outputs/_specialist_contract_test"))
    contract = build_candidate_feedback_contract(config, profile)

    assert contract["effective"]["codex_teacher_mode"] == "mot_context_mot_chuyen_gia"
    assert contract["effective"]["codex_direct_rnn_full_rl_enabled"] is True
    assert contract["effective"]["codex_parameter_scope"] == "direct_rnn_full_rl_adamw_v1"
    assert contract["negative_close_route"] == "codex_teacher_receipt_to_direct_rnn_full_rl_adamw"
    assert contract["optimizer"] == "direct_rnn_full_rl_adamw"


def test_reasoning_grpo_updates_only_reasoning_head_with_zero_encoder_credit() -> None:
    torch.manual_seed(23)
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.reasoning_rl_enabled = True
    network.require_current_architecture()
    trainer = FivePPOTrainer(network)
    hidden = torch.randn(2, FIVE_AGENT_COUNT, 16, requires_grad=True)
    allowed = torch.ones(2, FIVE_AGENT_COUNT, ACTION_COUNT, dtype=torch.bool)
    verifier = torch.tensor([
        [0.0, 1.0, -1.0, 0.6, -0.6, 0.3, -0.3],
        [0.0, -1.0, 1.0, -0.6, 0.6, -0.3, 0.3],
    ])
    reference_before = _parameter_snapshot(network.reasoning_reference_head)
    head_before = _parameter_snapshot(network.reasoning_head)
    trunk_before = _parameter_snapshot(network.lstm)

    loss, stats = trainer.reasoning_grpo_loss(
        hidden=hidden,
        allowed_action_masks=allowed,
        action_q_values_r=verifier,
        anchor_mask=torch.ones(2, dtype=torch.bool),
        group_size=16,
        reasoning_steps=3,
        ppo_clip=0.20,
        kl_beta=0.02,
        entropy_weight=0.0,
        loss_weight=1.0,
        encoder_credit_scale=0.0,
        min_group_std_r=0.01,
    )
    assert torch.isfinite(loss)
    assert stats["reasoning_group_eligible"] > 0
    assert stats["reasoning_optimizer_eligible"] > 0
    trainer.step_from_loss(loss)

    assert any(
        not torch.equal(value, head_before[name])
        for name, value in network.reasoning_head.named_parameters()
    )
    for name, value in network.reasoning_reference_head.named_parameters():
        torch.testing.assert_close(value, reference_before[name], rtol=0.0, atol=0.0)
    for name, value in network.lstm.named_parameters():
        torch.testing.assert_close(value, trunk_before[name], rtol=0.0, atol=0.0)
    assert hidden.grad is None or float(hidden.grad.abs().sum().item()) == 0.0


def test_reasoning_grpo_skips_groups_without_reward_variance() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.reasoning_rl_enabled = True
    network.require_current_architecture()
    trainer = FivePPOTrainer(network)
    hidden = torch.randn(1, FIVE_AGENT_COUNT, 16)
    allowed = torch.ones(1, FIVE_AGENT_COUNT, ACTION_COUNT, dtype=torch.bool)
    verifier = torch.full((1, FIVE_REASONING_ACTION_COUNT), 0.25)

    loss, stats = trainer.reasoning_grpo_loss(
        hidden=hidden,
        allowed_action_masks=allowed,
        action_q_values_r=verifier,
        anchor_mask=torch.ones(1, dtype=torch.bool),
        group_size=8,
        reasoning_steps=3,
        ppo_clip=0.20,
        kl_beta=0.02,
        entropy_weight=0.0,
        loss_weight=1.0,
        encoder_credit_scale=0.0,
        min_group_std_r=0.01,
    )
    assert float(loss.detach().cpu().item()) == pytest.approx(0.0)
    assert stats["reasoning_group_eligible"] == 0.0
    assert stats["reasoning_optimizer_eligible"] == 0.0


def test_reasoning_candidate_config_preserves_bridge_and_stays_telemetry_only() -> None:
    config = load_config("configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml")
    assert config.model.recurrent_core == "openai_five_lstm_v1"
    assert config.learning.five_reasoning_rl_enabled is True
    assert config.learning.five_rlhf_reference_capture_enabled is True
    assert config.learning.five_codex_teacher_enabled is True
    assert config.learning.five_codex_teacher_online_wait_sec == pytest.approx(120.0)
    assert config.learning.five_reasoning_steps == 3
    assert config.learning.five_reasoning_group_size == 8
    assert config.learning.five_reasoning_encoder_credit_scale == pytest.approx(0.0)
    assert config.learning.five_reasoning_action_scale == pytest.approx(0.0)
    assert INPUT_COUNT == 137
    assert OUTPUT_COUNT == 20


def test_reasoning_reference_refresh_is_explicit() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.reasoning_rl_enabled = True
    network.require_current_architecture()
    reference_before = copy.deepcopy(network.reasoning_reference_head.state_dict())
    with torch.no_grad():
        next(network.reasoning_head.parameters()).add_(0.25)
    for name, value in network.reasoning_reference_head.state_dict().items():
        torch.testing.assert_close(value, reference_before[name], rtol=0.0, atol=0.0)
    network.refresh_reasoning_reference()
    for name, value in network.reasoning_reference_head.state_dict().items():
        torch.testing.assert_close(
            value, network.reasoning_head.state_dict()[name], rtol=0.0, atol=0.0
        )


def test_legacy_checkpoint_migration_attaches_reasoning_without_rng_drift() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.reasoning_rl_enabled = True
    for name in (
        "reasoning_schema_version", "reasoning_steps",
        "reasoning_action_scale", "reasoning_action_residual_clip",
    ):
        network.__dict__.pop(name, None)
    torch.manual_seed(919)
    expected = torch.rand(4)
    torch.manual_seed(919)
    assert network.require_current_architecture() is True
    actual = torch.rand(4)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert network.reasoning_rl_enabled is True
    assert network.reasoning_action_scale == pytest.approx(0.0)


def test_legacy_checkpoint_migration_attaches_zero_codex_residual() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    hidden = torch.randn(5, 16)
    agent_ids = torch.arange(5, dtype=torch.long)
    base_before = network.heads.primary_action_logits(hidden, agent_ids).detach().clone()
    network.heads._modules.pop("codex_action_residual", None)
    network.heads._buffers.pop("codex_context_prototypes", None)
    network.__dict__.pop("codex_action_residual_enabled", None)
    network.__dict__.pop("codex_action_residual_scale", None)
    network.__dict__.pop("codex_context_prototype_capacity", None)
    network.__dict__.pop("codex_context_similarity_threshold", None)
    assert network.require_current_architecture() is True
    assert network.heads.codex_action_residual_enabled is False
    assert tuple(network.heads.codex_context_prototypes.shape) == (0, 16)
    base_after = network.heads.primary_action_logits(hidden, agent_ids).detach()
    torch.testing.assert_close(base_after, base_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        network.heads.apply_codex_action_residual(base_after, hidden),
        base_after,
        rtol=0.0,
        atol=0.0,
    )


def test_codex_residual_is_zero_outside_verified_context_support() -> None:
    heads = FactorizedActionHeads(hidden_size=4)
    heads.codex_action_residual_enabled = True
    hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        heads.codex_action_residual[-1].bias.fill_(1.0)
    assert heads.capture_codex_context_prototype(hidden[0]) is True

    logits = torch.zeros((2, ACTION_COUNT))
    support = heads.codex_context_support(hidden)
    updated = heads.apply_codex_action_residual(logits, hidden)

    torch.testing.assert_close(support, torch.tensor([1.0, 0.0]), rtol=0.0, atol=0.0)
    assert bool(torch.all(updated[0] > 0.0).item())
    torch.testing.assert_close(updated[1], logits[1], rtol=0.0, atol=0.0)


def test_codex_context_gate_keeps_near_context_and_rejects_low_tail() -> None:
    heads = FactorizedActionHeads(hidden_size=4)
    heads.codex_action_residual_enabled = True
    heads.codex_context_similarity_threshold = 0.995
    heads.codex_context_support_floor = 0.50
    prototype = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert heads.capture_codex_context_prototype(prototype) is True
    near = torch.tensor([0.9996, float((1.0 - 0.9996**2) ** 0.5), 0.0, 0.0])
    low_tail = torch.tensor([0.9970, float((1.0 - 0.9970**2) ** 0.5), 0.0, 0.0])

    support = heads.codex_context_support(torch.stack((near, low_tail)))

    assert float(support[0]) > 0.50
    assert float(support[1]) == pytest.approx(0.0)
