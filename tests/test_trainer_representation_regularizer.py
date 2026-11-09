"""Integration test for RepresentationRegularizer with FivePPOTrainer."""

import pytest
import numpy as np
import torch
from self_growing_trader.five_shared_policy import (
    SharedFiveTradingPolicy,
    FivePPOTrainer,
    FiveTrainingSchedule,
    FiveAgentMemory,
)
from self_growing_trader.representation_regularizer import RepresentationRegularizer


def test_trainer_attach_regularizer() -> None:
    policy = SharedFiveTradingPolicy()
    trainer = FivePPOTrainer(policy)

    assert trainer.regularizer is None
    reg = trainer.attach_representation_regularizer(
        hidden_dim=4096,
        var_weight=1.0,
        cov_weight=1.0,
        rank_weight=0.05,
        arc_weight=0.20,
    )
    assert trainer.regularizer is reg
    assert isinstance(trainer.regularizer, RepresentationRegularizer)

    # Verify that arc_weights was added to optimizer
    param_ids = {id(p) for p in trainer.optimizer.parameters}
    assert id(reg.arc_weights) in param_ids


def test_trainer_optimize_sequence_with_regularizer() -> None:
    torch.manual_seed(42)
    policy = SharedFiveTradingPolicy()
    trainer = FivePPOTrainer(policy)
    trainer.attach_representation_regularizer(hidden_dim=4096)

    # Create dummy rollout sequence: T=4 steps, 5 agents
    T, A, F = 4, 5, 137
    observations = np.random.randn(T, A, F).astype(np.float32)
    initial_memories = [FiveAgentMemory.zeros(device=trainer.device) for _ in range(A)]
    actions = np.random.randint(0, 7, size=(T, A), dtype=np.int64)
    targets = np.zeros((T, A), dtype=np.int64)
    delay_indices = np.zeros((T, A), dtype=np.int64)
    risk_indices = np.zeros((T, A), dtype=np.int64)
    offset_indices = np.zeros((T, A), dtype=np.int64)
    manage_indices = np.zeros((T, A), dtype=np.int64)
    close_indices = np.zeros((T, A), dtype=np.int64)
    old_log_probs = np.full((T, A), -1.5, dtype=np.float32)
    rewards = np.zeros((T, A), dtype=np.float32)
    rewards[-1] = 1.0
    dones = np.zeros((T, A), dtype=np.float32)

    stats = trainer.optimize_sequence(
        observations=observations,
        initial_memories=initial_memories,
        actions=actions,
        targets=targets,
        delay_indices=delay_indices,
        risk_indices=risk_indices,
        entry_offset_indices=offset_indices,
        manage_mode_indices=manage_indices,
        close_fraction_indices=close_indices,
        old_log_probabilities=old_log_probs,
        rewards=rewards,
        dones=dones,
        step_seconds=1.0,
        horizon_seconds=60.0,
    )

    assert "loss_var" in stats
    assert "loss_cov" in stats
    assert "loss_rank" in stats
    assert "loss_arcface" in stats
    assert "loss_reg_total" in stats
    assert "effective_rank" in stats
    assert stats["effective_rank"] > 0.0


def test_trainer_combined_regularizer_and_ewc() -> None:
    torch.manual_seed(42)
    policy = SharedFiveTradingPolicy()
    trainer = FivePPOTrainer(policy)
    reg = trainer.attach_representation_regularizer(hidden_dim=4096)
    ewc = trainer.attach_ewc(lambda_ewc=100.0)

    # Populate EWC anchor to activate it
    dummy_params = {name: p.detach().clone() for name, p in policy.named_parameters() if ewc.is_protected(name)}
    dummy_fisher = {name: torch.ones_like(p) for name, p in dummy_params.items()}
    ewc.optimal_params = dummy_params
    ewc.fisher_diag = dummy_fisher
    assert ewc.is_active

    T, A, F = 2, 5, 137
    observations = np.random.randn(T, A, F).astype(np.float32)
    initial_memories = [FiveAgentMemory.zeros(device=trainer.device) for _ in range(A)]
    actions = np.random.randint(0, 7, size=(T, A), dtype=np.int64)

    stats = trainer.optimize_sequence(
        observations=observations,
        initial_memories=initial_memories,
        actions=actions,
        targets=np.zeros((T, A), dtype=np.int64),
        delay_indices=np.zeros((T, A), dtype=np.int64),
        risk_indices=np.zeros((T, A), dtype=np.int64),
        entry_offset_indices=np.zeros((T, A), dtype=np.int64),
        manage_mode_indices=np.zeros((T, A), dtype=np.int64),
        close_fraction_indices=np.zeros((T, A), dtype=np.int64),
        old_log_probabilities=np.full((T, A), -1.5, dtype=np.float32),
        rewards=np.zeros((T, A), dtype=np.float32),
        dones=np.zeros((T, A), dtype=np.float32),
        step_seconds=1.0,
        horizon_seconds=60.0,
    )

    assert "loss_reg_total" in stats
    assert "effective_rank" in stats
    assert "ewc_loss" in stats
    assert stats["ewc_loss"] >= 0.0
