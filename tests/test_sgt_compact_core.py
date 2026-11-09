"""Comprehensive unit tests for SGT-V193-Compact Sequence Core.

Verifies:
1. Shape contract: Input [5, 137] -> Action Wire [5, 20], Values [5, 1], Hidden [5, 768].
2. Matrix memory capacity: exactly 147,456 elements per agent (4 heads x 192 x 192).
3. Parameter count: between 16.5M and 18.5M parameters.
4. Numerical stability: Log-space exponential gating with dynamic stabilizer m_t over 500 steps.
5. ReZero initialization: rezero_alpha is 0.0 initially, ensuring zero perturbation.
6. TorchScript JIT tracing compatibility.
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.self_growing_trader.sgt_compact_core import (
    COMPACT_HEAD_DIM,
    COMPACT_HIDDEN_SIZE,
    COMPACT_NUM_HEADS,
    FIVE_AGENT_COUNT,
    INPUT_COUNT,
    OUTPUT_COUNT,
    CompactAgentMemory,
    MatrixLSTMCell,
    MatrixLSTMState,
    RelationalCrossAgentGraphTransformer,
    ScalarLSTMCell,
    ScalarLSTMState,
    SGTCompactTradingPolicy,
)


def test_compact_parameter_count_acceptance_criterion():
    policy = SGTCompactTradingPolicy()
    params = policy.count_parameters()
    assert 16_500_000 <= params <= 18_500_000, f"Expected params in [16.5M, 18.5M], got {params:,}"


def test_matrix_memory_capacity_criterion():
    num_heads = COMPACT_NUM_HEADS
    head_dim = COMPACT_HEAD_DIM
    total_elements = num_heads * head_dim * head_dim
    assert total_elements == 147_456, f"Expected 147,456 matrix memory elements, got {total_elements}"


def test_compact_forward_shape_contract():
    policy = SGTCompactTradingPolicy().eval()
    obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
    memories = policy.initial_memories(FIVE_AGENT_COUNT)

    action_wire, values, extra, next_memories = policy(obs, memories=memories)

    assert action_wire.shape == (FIVE_AGENT_COUNT, OUTPUT_COUNT)
    assert values.shape == (FIVE_AGENT_COUNT, 1)
    assert extra["hidden"].shape == (FIVE_AGENT_COUNT, COMPACT_HIDDEN_SIZE)
    assert len(next_memories) == FIVE_AGENT_COUNT
    assert next_memories[0].mlstm.covariance.shape == (COMPACT_NUM_HEADS, COMPACT_HEAD_DIM, COMPACT_HEAD_DIM)
    assert next_memories[0].slstm.cell.shape == (COMPACT_HIDDEN_SIZE,)


def test_rezero_initialization_criterion():
    transformer = RelationalCrossAgentGraphTransformer()
    assert float(transformer.rezero_alpha.item()) == 0.0, "ReZero alpha must be 0.0 initially"

    # Forward pass with alpha=0 should produce identity plus layer norm
    x = torch.randn(FIVE_AGENT_COUNT, COMPACT_HIDDEN_SIZE)
    out, attn = transformer(x)
    assert out.shape == x.shape
    assert attn.shape == (COMPACT_NUM_HEADS, FIVE_AGENT_COUNT, FIVE_AGENT_COUNT)


def test_exponential_gating_numerical_stability_500_steps():
    cell = MatrixLSTMCell(in_features=1024, hidden_size=768, num_heads=4).eval()
    state = cell.initial_state(FIVE_AGENT_COUNT, torch.device("cpu"))

    torch.manual_seed(42)
    for step in range(500):
        # Generate varied amplitude input to stress-test gating
        x = torch.randn(FIVE_AGENT_COUNT, 1024) * 2.5
        h, state = cell(x, state)

        # Assert no NaN or Inf
        assert not torch.isnan(h).any(), f"NaN encountered at step {step}"
        assert not torch.isinf(h).any(), f"Inf encountered at step {step}"
        assert not torch.isnan(state.covariance).any(), f"NaN in covariance at step {step}"
        assert not torch.isinf(state.covariance).any(), f"Inf in covariance at step {step}"

        # Assert stabilizer m_t remains bounded
        assert (state.stabilizer > -100.0).all()


def test_slstm_numerical_stability_500_steps():
    cell = ScalarLSTMCell(in_features=768, hidden_size=768, num_heads=4).eval()
    state = cell.initial_state(FIVE_AGENT_COUNT, torch.device("cpu"))

    torch.manual_seed(42)
    for step in range(500):
        x = torch.randn(FIVE_AGENT_COUNT, 768) * 2.0
        h, state = cell(x, state)
        assert not torch.isnan(h).any(), f"NaN in sLSTM at step {step}"
        assert not torch.isinf(h).any(), f"Inf in sLSTM at step {step}"


def test_gradient_flow_end_to_end():
    policy = SGTCompactTradingPolicy().train()
    obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT, requires_grad=True)
    memories = policy.initial_memories(FIVE_AGENT_COUNT)

    action_wire, values, extra, _ = policy(obs, memories=memories)
    loss = action_wire.sum() + values.sum()
    loss.backward()

    assert obs.grad is not None
    assert not torch.isnan(obs.grad).any()
    assert not torch.isinf(obs.grad).any()


def test_torchscript_jit_tracing_compatibility():
    class TracedWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs, cov, norm, stab, cell, snorm, sstab):
            batch = obs.shape[0]
            memories = []
            for i in range(batch):
                memories.append(
                    CompactAgentMemory(
                        mlstm=MatrixLSTMState(covariance=cov[i], normalizer=norm[i], stabilizer=stab[i]),
                        slstm=ScalarLSTMState(cell=cell[i], normalizer=snorm[i], stabilizer=sstab[i]),
                        hidden=torch.zeros(COMPACT_HIDDEN_SIZE),
                    )
                )
            wire, val, extra, next_m = self.policy(obs, memories=memories)
            next_cov = torch.stack([m.mlstm.covariance for m in next_m])
            return wire, val, next_cov

    policy = SGTCompactTradingPolicy().eval()
    wrapper = TracedWrapper(policy)

    obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
    cov = torch.zeros(FIVE_AGENT_COUNT, COMPACT_NUM_HEADS, COMPACT_HEAD_DIM, COMPACT_HEAD_DIM)
    norm = torch.zeros(FIVE_AGENT_COUNT, COMPACT_NUM_HEADS, COMPACT_HEAD_DIM)
    stab = torch.zeros(FIVE_AGENT_COUNT, COMPACT_NUM_HEADS)
    cell = torch.zeros(FIVE_AGENT_COUNT, COMPACT_HIDDEN_SIZE)
    snorm = torch.zeros(FIVE_AGENT_COUNT, COMPACT_HIDDEN_SIZE)
    sstab = torch.zeros(FIVE_AGENT_COUNT, COMPACT_HIDDEN_SIZE)

    # Trace wrapper
    traced = torch.jit.trace(wrapper, (obs, cov, norm, stab, cell, snorm, sstab))
    out_wire, out_val, out_cov = traced(obs, cov, norm, stab, cell, snorm, sstab)
    assert out_wire.shape == (FIVE_AGENT_COUNT, OUTPUT_COUNT)
    assert out_val.shape == (FIVE_AGENT_COUNT, 1)


def test_compact_evaluate_sequence_and_ppo_step():
    """Verify that evaluate_sequence matches PPO optimizer contract and autograd backprop."""
    from src.self_growing_trader.five_shared_policy import FivePPOTrainer

    policy = SGTCompactTradingPolicy()
    trainer = FivePPOTrainer(policy, device=torch.device("cpu"))

    T = 4
    obs = np.random.randn(T, FIVE_AGENT_COUNT, INPUT_COUNT).astype(np.float32)
    memories = policy.initial_memories(FIVE_AGENT_COUNT)

    actions = np.random.randint(0, 12, (T, FIVE_AGENT_COUNT))
    targets = np.zeros((T, FIVE_AGENT_COUNT), dtype=np.int64)
    delays = np.zeros((T, FIVE_AGENT_COUNT), dtype=np.int64)
    risks = np.zeros((T, FIVE_AGENT_COUNT), dtype=np.int64)
    entry_offsets = np.zeros((T, FIVE_AGENT_COUNT), dtype=np.int64)
    manage_modes = np.zeros((T, FIVE_AGENT_COUNT), dtype=np.int64)
    close_fractions = np.zeros((T, FIVE_AGENT_COUNT), dtype=np.int64)
    old_lps = -np.random.rand(T, FIVE_AGENT_COUNT).astype(np.float32)
    rewards = np.random.randn(T, FIVE_AGENT_COUNT).astype(np.float32)
    dones = np.zeros((T, FIVE_AGENT_COUNT), dtype=bool)
    dones[1, 0] = True  # Agent 0 finished episode

    metrics = trainer.optimize_sequence(
        observations=obs,
        initial_memories=memories,
        actions=actions,
        targets=targets,
        delay_indices=delays,
        risk_indices=risks,
        entry_offset_indices=entry_offsets,
        manage_mode_indices=manage_modes,
        close_fraction_indices=close_fractions,
        old_log_probabilities=old_lps,
        rewards=rewards,
        dones=dones,
        step_seconds=1.0,
        horizon_seconds=60.0,
    )

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["total_loss"])

