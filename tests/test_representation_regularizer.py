"""Unit tests for RepresentationRegularizer (VICReg, RankMe, ArcFace)."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from self_growing_trader.representation_regularizer import (
    RepresentationRegularizer,
    DEFAULT_CLASS_WEIGHTS,
)


def test_target_derivation() -> None:
    actions = torch.tensor([[0, 1, 2, 7, 8], [3, 4, 5, 6, 11]], dtype=torch.long)
    targets = RepresentationRegularizer.derive_targets(actions)
    expected = torch.tensor([[0, 1, 2, 0, 0], [1, 2, 1, 2, 0]], dtype=torch.long)
    assert torch.equal(targets, expected)


def test_representation_regularizer_shapes_and_outputs() -> None:
    torch.manual_seed(42)
    B, A, D = 4, 5, 128  # Use smaller D for fast test
    hidden = torch.randn(B, A, D, requires_grad=True)
    actions = torch.randint(0, 12, (B, A))
    targets = RepresentationRegularizer.derive_targets(actions)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(hidden, targets)

    assert "loss_reg_total" in out
    assert "loss_var" in out
    assert "loss_cov" in out
    assert "loss_rank" in out
    assert "loss_arcface" in out
    assert "effective_rank" in out
    assert "mean_std" in out
    assert "inter_class_cos" in out

    assert out["loss_reg_total"].item() > 0.0
    assert out["loss_var"].item() >= 0.0
    assert out["loss_cov"].item() >= 0.0
    assert out["effective_rank"] > 1.0


def test_representation_regularizer_gradient_flow() -> None:
    torch.manual_seed(42)
    B, A, D = 2, 5, 64
    hidden = torch.randn(B, A, D, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 0, 1], [2, 0, 1, 2, 0]], dtype=torch.long)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(hidden, targets)
    loss = out["loss_reg_total"]

    loss.backward()

    assert hidden.grad is not None
    assert not torch.isnan(hidden.grad).any()
    assert not torch.isinf(hidden.grad).any()
    assert reg.arc_weights.grad is not None
    assert not torch.isnan(reg.arc_weights.grad).any()


def test_effective_rank_recovery() -> None:
    """Test that optimizing variance + covariance loss increases effective rank."""
    torch.manual_seed(42)
    N, D = 64, 32
    # Create collapsed representations (rank 2)
    v1 = torch.randn(D)
    v2 = torch.randn(D)
    weights = torch.randn(N, 2)
    collapsed_hidden = (weights[:, :1] @ v1.unsqueeze(0) + weights[:, 1:] @ v2.unsqueeze(0)).detach()
    collapsed_hidden.requires_grad = True

    reg = RepresentationRegularizer(hidden_dim=D, var_weight=1.0, cov_weight=1.0, rank_weight=0.1, arc_weight=0.0)
    initial_rank = reg.compute_effective_rank(collapsed_hidden)
    assert initial_rank < 3.0  # Confirm it starts collapsed

    # Run 40 optimizer steps
    optimizer = torch.optim.Adam([collapsed_hidden], lr=0.05)
    for _ in range(40):
        optimizer.zero_grad()
        out = reg(collapsed_hidden)
        out["loss_reg_total"].backward()
        optimizer.step()

    recovered_rank = reg.compute_effective_rank(collapsed_hidden)
    assert recovered_rank > initial_rank * 2.5
    assert recovered_rank > 6.0


def test_sylvester_determinant_identity() -> None:
    """Verify that dual formulation (N < D) matches primal formulation."""
    torch.manual_seed(42)
    N, D = 20, 50
    z = torch.randn(N, D)
    reg = RepresentationRegularizer(hidden_dim=D, gamma_reg=1e-3)

    # Primal
    gamma = reg.gamma_reg
    C = (z.T @ z) / N
    C_reg = C + gamma * torch.eye(D)
    sign_p, logdet_p = torch.linalg.slogdet(C_reg)
    primal_loss = -logdet_p / D + torch.trace(C) / D

    # Dual via method
    dual_loss = reg.compute_rankme_loss(z)

    # Difference should be sub-epsilon
    assert torch.isclose(primal_loss, dual_loss, atol=1e-4)


def test_warmup_scheduler() -> None:
    reg = RepresentationRegularizer(hidden_dim=128)
    reg.set_warmup_weights(1)
    assert reg.var_weight == 0.10
    assert reg.cov_weight == 0.10
    assert reg.rank_weight == 0.01
    assert reg.arc_weight == 0.05

    reg.set_warmup_weights(2)
    assert reg.var_weight == 0.50
    assert reg.cov_weight == 0.50
    assert reg.rank_weight == 0.02
    assert reg.arc_weight == 0.10

    reg.set_warmup_weights(3)
    assert reg.var_weight == 1.00
    assert reg.cov_weight == 1.00
    assert reg.rank_weight == 0.05
    assert reg.arc_weight == 0.20
