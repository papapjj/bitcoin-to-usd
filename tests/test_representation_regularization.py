"""Comprehensive Tier 1 Test Suite for Representation Regularization (Milestone 3).

Follows the 6-section specification blueprint:
Section 1: Shape Contracts & Interface Compliance
Section 2: Gradient Flow Verification
Section 3: Numerical Stability & Edge Cases
Section 4: RankMe Effective Rank Tracking & Expansion
Section 5: ArcFace Angular Separation & Class Balancing
Section 6: Zero Inference Impact & System Invariants
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from self_growing_trader.representation_regularizer import (
    RepresentationRegularizer,
    DEFAULT_CLASS_WEIGHTS,
    BUY_ACTION_SET,
    SELL_ACTION_SET,
)
from self_growing_trader.five_shared_policy import (
    SharedFiveTradingPolicy,
    FivePPOTrainer,
    FiveAgentMemory,
    ACTION_COUNT,
    INPUT_COUNT,
    FIVE_AGENT_COUNT,
)


# ============================================================================
# Section 1: Shape Contracts & Interface Compliance
# ============================================================================

def test_target_derivation_mapping() -> None:
    """Verify target derivation mapping matches specification across all action indices [0..11]."""
    # 0: HOLD, 1: BUY_MKT, 2: SELL_MKT, 3: BUY_LIM, 4: SELL_LIM, 5: BUY_STP, 6: SELL_STP, 7..11: MANAGE/CLOSE
    actions = torch.arange(0, 12, dtype=torch.long)
    targets = RepresentationRegularizer.derive_targets(actions)

    assert targets.shape == actions.shape
    assert targets.dtype == torch.long
    assert targets.min().item() >= 0
    assert targets.max().item() <= 2

    # Expected: 0, 7..11 -> 0; {1, 3, 5} -> 1; {2, 4, 6} -> 2
    expected = torch.tensor([0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 0, 0], dtype=torch.long)
    assert torch.equal(targets, expected), f"Target mapping mismatch: got {targets.tolist()}, expected {expected.tolist()}"

    # Also test multi-dimensional input [Batch, Agents]
    actions_2d = torch.tensor([[0, 1, 2, 7, 8], [3, 4, 5, 6, 11]], dtype=torch.long)
    targets_2d = RepresentationRegularizer.derive_targets(actions_2d)
    expected_2d = torch.tensor([[0, 1, 2, 0, 0], [1, 2, 1, 2, 0]], dtype=torch.long)
    assert torch.equal(targets_2d, expected_2d)


def test_regularizer_input_dimensions_2d_and_3d() -> None:
    """Verify regularizer handles both 2D [N, D] and 3D [B, A, D] inputs identically."""
    torch.manual_seed(42)
    B, A, D = 4, 5, 64
    reg = RepresentationRegularizer(hidden_dim=D)

    # 3D inputs
    hidden_3d = torch.randn(B, A, D)
    actions_3d = torch.randint(0, 12, (B, A))
    targets_3d = RepresentationRegularizer.derive_targets(actions_3d)
    mask_3d = torch.ones(B, A, dtype=torch.bool)
    mask_3d[0, 0] = False  # Mask out one element to test sample_mask

    out_3d = reg(hidden_3d, targets=targets_3d, sample_mask=mask_3d)

    # Equivalent 2D reshaped inputs
    hidden_2d = hidden_3d.reshape(B * A, D)
    targets_2d = targets_3d.reshape(B * A)
    mask_2d = mask_3d.reshape(B * A)

    out_2d = reg(hidden_2d, targets=targets_2d, sample_mask=mask_2d)

    # Outputs should match closely
    assert torch.isclose(out_3d["loss_reg_total"], out_2d["loss_reg_total"], atol=1e-5)
    assert torch.isclose(out_3d["loss_var"], out_2d["loss_var"], atol=1e-5)
    assert torch.isclose(out_3d["loss_cov"], out_2d["loss_cov"], atol=1e-5)
    assert torch.isclose(out_3d["loss_rank"], out_2d["loss_rank"], atol=1e-5)
    assert torch.isclose(out_3d["loss_arcface"], out_2d["loss_arcface"], atol=1e-5)
    assert abs(out_3d["effective_rank"] - out_2d["effective_rank"]) < 1e-4


def test_output_dictionary_contract_and_metric_types() -> None:
    """Verify output dictionary contract, tensor dimensions, and Python float types."""
    torch.manual_seed(42)
    N, D = 30, 32
    hidden = torch.randn(N, D, requires_grad=True)
    targets = torch.randint(0, 3, (N,))

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(hidden, targets=targets)

    expected_keys = {
        "loss_reg_total",
        "loss_var",
        "loss_cov",
        "loss_rank",
        "loss_arcface",
        "effective_rank",
        "mean_std",
        "inter_class_cos",
    }
    assert expected_keys.issubset(out.keys()), f"Missing keys: {expected_keys - set(out.keys())}"

    # Verify all loss items are 0-dimensional scalar tensors requiring grad
    for loss_key in ("loss_reg_total", "loss_var", "loss_cov", "loss_rank", "loss_arcface"):
        val = out[loss_key]
        assert isinstance(val, torch.Tensor), f"{loss_key} must be a torch.Tensor"
        assert val.dim() == 0, f"{loss_key} must be a 0D scalar, got dim {val.dim()}"
        assert val.requires_grad, f"{loss_key} must require grad when input requires grad"
        assert bool(torch.isfinite(val).item()), f"{loss_key} must be finite"

    # Verify diagnostic metrics are standard Python floats
    assert isinstance(out["effective_rank"], float)
    assert isinstance(out["mean_std"], float)
    assert isinstance(out["inter_class_cos"], float)
    assert np.isfinite(out["effective_rank"])
    assert np.isfinite(out["mean_std"])
    assert np.isfinite(out["inter_class_cos"])


def test_policy_forward_shape_contract_preservation() -> None:
    """Verify forward_five preserves strict action wire [5, 20] and value [5, 1] shape contracts."""
    policy = SharedFiveTradingPolicy()
    memories = [FiveAgentMemory.zeros(device=torch.device("cpu")) for _ in range(FIVE_AGENT_COUNT)]
    obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)

    results, next_memories, context_probe = policy.forward_five(obs, memories)

    assert len(results) == FIVE_AGENT_COUNT
    for r in results:
        assert r.output_vector.shape == (20,)
    values = context_probe["value"]
    assert values.shape in ((FIVE_AGENT_COUNT,), (FIVE_AGENT_COUNT, 1))
    assert len(next_memories) == FIVE_AGENT_COUNT
    for mem in next_memories:
        assert mem.hidden.shape == (4096,)
        assert mem.cell.shape == (4096,)


# ============================================================================
# Section 2: Gradient Flow Verification
# ============================================================================

def test_loss_reg_total_gradient_backprop() -> None:
    """Verify gradient backpropagation through total regularization loss to hidden activations and arc centers."""
    torch.manual_seed(42)
    N, D = 40, 64
    hidden = torch.randn(N, D, requires_grad=True)
    targets = torch.randint(0, 3, (N,))

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(hidden, targets=targets)
    total_loss = out["loss_reg_total"]

    total_loss.backward()

    # Verify hidden gradients
    assert hidden.grad is not None, "hidden.grad must not be None"
    assert hidden.grad.shape == hidden.shape
    assert not torch.isnan(hidden.grad).any(), "NaN found in hidden.grad"
    assert not torch.isinf(hidden.grad).any(), "Inf found in hidden.grad"
    assert torch.count_nonzero(hidden.grad).item() > 0, "hidden.grad must be non-zero"

    # Verify arc_weights gradients
    assert reg.arc_weights.grad is not None, "reg.arc_weights.grad must not be None"
    assert reg.arc_weights.grad.shape == reg.arc_weights.shape
    assert not torch.isnan(reg.arc_weights.grad).any(), "NaN found in arc_weights.grad"
    assert not torch.isinf(reg.arc_weights.grad).any(), "Inf found in arc_weights.grad"
    assert torch.count_nonzero(reg.arc_weights.grad).item() > 0


def test_component_gradient_flow_isolated() -> None:
    """Verify clean isolated gradient flow for each component (var, cov, rank, arcface)."""
    torch.manual_seed(42)
    N, D = 50, 32
    reg = RepresentationRegularizer(hidden_dim=D)

    # 1. Variance Loss: leaf tensor scaled to std < 1.0 -> active loss
    z_var = (torch.randn(N, D) * 0.5).requires_grad_(True)
    l_var = reg.compute_variance_loss(z_var)
    l_var.backward()
    assert z_var.grad is not None and not torch.isnan(z_var.grad).any() and not torch.isinf(z_var.grad).any()
    assert torch.count_nonzero(z_var.grad).item() > 0

    # 2. Covariance Loss
    z_cov = torch.randn(N, D, requires_grad=True)
    z_c = z_cov - z_cov.mean(dim=0, keepdim=True)
    l_cov = reg.compute_covariance_loss(z_c)
    l_cov.backward()
    assert z_cov.grad is not None and not torch.isnan(z_cov.grad).any() and not torch.isinf(z_cov.grad).any()
    assert torch.count_nonzero(z_cov.grad).item() > 0

    # 3. RankMe Loss
    z_rank = torch.randn(N, D, requires_grad=True)
    l_rank = reg.compute_rankme_loss(z_rank)
    l_rank.backward()
    assert z_rank.grad is not None and not torch.isnan(z_rank.grad).any() and not torch.isinf(z_rank.grad).any()
    assert torch.count_nonzero(z_rank.grad).item() > 0

    # 4. ArcFace Loss
    z_arc = torch.randn(N, D, requires_grad=True)
    targets = torch.randint(0, 3, (N,))
    l_arc, _ = reg.compute_arcface_loss(z_arc, targets)
    l_arc.backward()
    assert z_arc.grad is not None and not torch.isnan(z_arc.grad).any() and not torch.isinf(z_arc.grad).any()
    assert torch.count_nonzero(z_arc.grad).item() > 0


def test_policy_parameter_gradient_reachability() -> None:
    """Verify gradients propagate end-to-end to all policy submodules (transformer, lstm, encoder, heads)."""
    torch.manual_seed(42)
    policy = SharedFiveTradingPolicy()
    trainer = FivePPOTrainer(policy)
    trainer.attach_representation_regularizer(hidden_dim=4096)

    T, A, F_dim = 2, 5, INPUT_COUNT
    observations = np.random.randn(T, A, F_dim).astype(np.float32)
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

    assert stats["loss_reg_total"] > 0.0

    # Verify gradient reachability in core submodules
    assert any(p.grad is not None for p in policy.post_lstm_transformer.parameters()), "Transformer params must receive grad"
    assert any(p.grad is not None for p in policy.lstm.parameters()), "LSTM params must receive grad"
    assert any(p.grad is not None for p in policy.observation_encoder.parameters()), "Encoder params must receive grad"
    assert any(p.grad is not None for p in policy.heads.parameters()), "Head params must receive grad"


def test_regularizer_optimizer_registration() -> None:
    """Verify regularizer parameters are registered in optimizer and updated on step."""
    policy = SharedFiveTradingPolicy()
    trainer = FivePPOTrainer(policy)
    reg = trainer.attach_representation_regularizer(hidden_dim=4096)

    # Ensure arc_weights is tracked
    assert any(p is reg.arc_weights for p in trainer.optimizer.parameters)

    initial_arc_weights = reg.arc_weights.detach().clone()

    T, A, F_dim = 2, 5, INPUT_COUNT
    observations = np.random.randn(T, A, F_dim).astype(np.float32)
    initial_memories = [FiveAgentMemory.zeros(device=trainer.device) for _ in range(A)]
    actions = np.random.randint(0, 7, size=(T, A), dtype=np.int64)

    trainer.optimize_sequence(
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

    final_arc_weights = reg.arc_weights.detach()
    # Parameters must have been updated by optimizer step
    assert not torch.equal(initial_arc_weights, final_arc_weights), "arc_weights must be updated by optimizer"


# ============================================================================
# Section 3: Numerical Stability & Edge Cases
# ============================================================================

def test_batch_size_edge_cases_n1_and_n2() -> None:
    """Verify regularizer operates smoothly on boundary batch sizes N=1 and N=2."""
    D = 32
    reg = RepresentationRegularizer(hidden_dim=D)

    # N = 1 (Short-circuit edge case)
    z_n1 = torch.randn(1, D, requires_grad=True)
    out_n1 = reg(z_n1, targets=torch.tensor([0]))
    assert out_n1["loss_reg_total"].item() == 0.0
    assert out_n1["effective_rank"] == 1.0

    # Crucial autograd check: backward on N=1 must succeed with 0 grad and no exception
    out_n1["loss_reg_total"].backward()
    assert z_n1.grad is not None
    assert torch.equal(z_n1.grad, torch.zeros_like(z_n1))

    # N = 2 (Minimal valid pair for unbiased variance)
    z_n2 = torch.randn(2, D, requires_grad=True)
    out_n2 = reg(z_n2, targets=torch.tensor([0, 1]))
    assert out_n2["loss_reg_total"].item() > 0.0
    assert not np.isnan(out_n2["effective_rank"])

    out_n2["loss_reg_total"].backward()
    assert z_n2.grad is not None
    assert not torch.isnan(z_n2.grad).any()
    assert not torch.isinf(z_n2.grad).any()


def test_sylvester_determinant_identity_primal_vs_dual() -> None:
    """Verify mathematical and numerical equivalence between dual and primal RankMe formulations."""
    torch.manual_seed(42)
    # Test N < D (dual case)
    N, D = 15, 60
    z = torch.randn(N, D)
    reg = RepresentationRegularizer(hidden_dim=D, gamma_reg=1e-3)
    gamma = reg.gamma_reg

    # Primal formulation: D x D
    C = (z.T @ z) / N
    C_reg = C + gamma * torch.eye(D)
    _, logdet_p = torch.linalg.slogdet(C_reg)
    primal_loss = -logdet_p / D + torch.trace(C) / D

    # Dual formulation: computed via regularizer compute_rankme_loss
    dual_loss = reg.compute_rankme_loss(z)

    assert torch.isclose(primal_loss, dual_loss, atol=1e-4), (
        f"Primal and Dual mismatch: primal={primal_loss.item():.6f}, dual={dual_loss.item():.6f}"
    )

    # Test N == D boundary condition
    N_eq, D_eq = 25, 25
    z_eq = torch.randn(N_eq, D_eq)
    reg_eq = RepresentationRegularizer(hidden_dim=D_eq, gamma_reg=1e-3)
    loss_eq = reg_eq.compute_rankme_loss(z_eq)
    assert bool(torch.isfinite(loss_eq).item())


def test_zero_vector_and_zero_variance_stability() -> None:
    """Verify stability under complete point collapse (all zeros) and collinear constant vectors."""
    D = 32
    reg = RepresentationRegularizer(hidden_dim=D)

    # 1. Complete zero representation
    z_zeros = torch.zeros(10, D, requires_grad=True)
    targets_zeros = torch.zeros(10, dtype=torch.long)
    out_zeros = reg(z_zeros, targets=targets_zeros)
    out_zeros["loss_reg_total"].backward()

    assert z_zeros.grad is not None
    assert not torch.isnan(z_zeros.grad).any(), "Zero vector backprop produced NaNs"
    assert not torch.isinf(z_zeros.grad).any(), "Zero vector backprop produced Infs"

    # 2. Constant non-zero vectors (Zero variance collapse)
    z_const = (torch.ones(15, D) * 5.0).requires_grad_(True)
    targets_const = torch.zeros(15, dtype=torch.long)
    out_const = reg(z_const, targets=targets_const)
    out_const["loss_reg_total"].backward()

    assert z_const.grad is not None
    assert not torch.isnan(z_const.grad).any(), "Constant vector backprop produced NaNs"
    assert not torch.isinf(z_const.grad).any(), "Constant vector backprop produced Infs"


def test_extreme_norm_and_scale_robustness() -> None:
    """Verify numerical robustness under extreme magnitude scales (1e-6 and 1e6)."""
    D = 32
    reg = RepresentationRegularizer(hidden_dim=D)

    # Tiny scale: 1e-6
    z_tiny = (torch.randn(20, D) * 1e-6).requires_grad_(True)
    targets = torch.randint(0, 3, (20,))
    out_tiny = reg(z_tiny, targets)
    out_tiny["loss_reg_total"].backward()
    assert z_tiny.grad is not None
    assert not torch.isnan(z_tiny.grad).any()
    assert not torch.isinf(z_tiny.grad).any()

    # Large scale: 1e6
    z_large = (torch.randn(20, D) * 1e6).requires_grad_(True)
    out_large = reg(z_large, targets)
    out_large["loss_reg_total"].backward()
    assert z_large.grad is not None
    assert not torch.isnan(z_large.grad).any()
    assert not torch.isinf(z_large.grad).any()


def test_opposing_vector_arcface_linear_fallback() -> None:
    """Verify ArcFace linear fallback activates when cos(theta) <= cos(pi - m)."""
    D = 16
    margin = 0.35
    reg = RepresentationRegularizer(hidden_dim=D, margin=margin)

    # Set class 1 center along axis 0
    with torch.no_grad():
        reg.arc_weights.zero_()
        reg.arc_weights[1, 0] = 1.0

    # Directly opposing vector along axis 0 (cos = -1.0 < cos(pi - 0.35) = -0.9394)
    z_opposing = torch.zeros(1, D)
    z_opposing[0, 0] = -1.0
    z_opposing.requires_grad_(True)

    targets = torch.tensor([1], dtype=torch.long)
    loss, _ = reg.compute_arcface_loss(z_opposing, targets)

    assert bool(torch.isfinite(loss).item())
    loss.backward()

    assert z_opposing.grad is not None
    assert not torch.isnan(z_opposing.grad).any(), "ArcFace linear fallback produced NaNs"
    assert not torch.isinf(z_opposing.grad).any(), "ArcFace linear fallback produced Infs"


def test_monoclass_and_extreme_imbalance_handling() -> None:
    """Verify regularizer handles single-class batch (100% HOLD) with inverse frequency weights."""
    D = 32
    reg = RepresentationRegularizer(hidden_dim=D)

    # Batch with 100% HOLD (targets = all 0)
    z_hold = torch.randn(25, D, requires_grad=True)
    targets_hold = torch.zeros(25, dtype=torch.long)

    out = reg(z_hold, targets=targets_hold)
    assert bool(torch.isfinite(out["loss_arcface"]).item())
    out["loss_reg_total"].backward()

    assert z_hold.grad is not None
    assert not torch.isnan(z_hold.grad).any()
    assert not torch.isinf(z_hold.grad).any()


# ============================================================================
# Section 4: RankMe Effective Rank Tracking & Expansion
# ============================================================================

def test_effective_rank_mathematical_bounds() -> None:
    """Verify mathematical bounds of effective rank: 1.0 <= r_eff <= min(N, D)."""
    torch.manual_seed(42)
    reg = RepresentationRegularizer(hidden_dim=64)

    # 1. Rank-1 centered representation: r_eff must be 1.0
    u = torch.randn(50, 1)
    u = u - u.mean()
    v = torch.randn(1, 64)
    z_rank1 = u @ v
    r_rank1 = reg.compute_effective_rank(z_rank1)
    assert 1.0 <= r_rank1 <= 1.10, f"Rank-1 representation should have r_eff ~ 1.0, got {r_rank1}"

    # 2. High-dimensional random representation: 1.0 <= r_eff <= min(N, D)
    N, D = 40, 64
    z_rand = torch.randn(N, D)
    r_rand = reg.compute_effective_rank(z_rand)
    assert 1.0 <= r_rand <= min(N, D) + 1e-4, f"r_eff out of bounds: {r_rand} not in [1.0, {min(N, D)}]"


def test_effective_rank_active_recovery_from_collapse() -> None:
    """Verify active optimization of regularizer expands collapsed effective rank by >= 2.5x and > 6.0."""
    torch.manual_seed(42)
    N, D = 64, 32
    # Construct rank-2 collapsed representation: Z = U @ V^T
    U = torch.randn(N, 2)
    V = torch.randn(D, 2)
    Z = (U @ V.T).detach().requires_grad_(True)

    reg = RepresentationRegularizer(
        hidden_dim=D,
        var_weight=1.0,
        cov_weight=1.0,
        rank_weight=0.10,
        arc_weight=0.0,
    )

    initial_rank = reg.compute_effective_rank(Z)
    assert initial_rank < 3.0, f"Must start with collapsed effective rank, was {initial_rank}"

    # Optimize Z for 40 steps
    optimizer = torch.optim.Adam([Z], lr=0.05)
    for _ in range(40):
        optimizer.zero_grad()
        out = reg(Z)
        out["loss_reg_total"].backward()
        optimizer.step()

    final_rank = reg.compute_effective_rank(Z)
    assert final_rank >= initial_rank * 2.5, (
        f"Effective rank failed to expand >= 2.5x: initial={initial_rank:.2f}, final={final_rank:.2f}"
    )
    assert final_rank > 6.0, f"Final effective rank should exceed 6.0, got {final_rank:.2f}"


def test_rankme_gradient_expands_singular_value_spectrum() -> None:
    """Verify RankMe loss gradient actively increases singular values of collapsed dimensions."""
    torch.manual_seed(42)
    N, D = 64, 32

    U = torch.randn(N, 2)
    V = torch.randn(D, 2)
    Z = (U @ V.T).detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D, gamma_reg=1e-3)

    initial_sv = torch.linalg.svdvals(Z - Z.mean(dim=0, keepdim=True))
    assert initial_sv[2].item() < 1e-4, "Third singular value should be near zero initially"

    optimizer = torch.optim.Adam([Z], lr=0.05)
    for _ in range(50):
        optimizer.zero_grad()
        loss = reg.compute_rankme_loss(Z) + 0.5 * reg.compute_variance_loss(Z)
        loss.backward()
        optimizer.step()

    final_sv = torch.linalg.svdvals(Z - Z.mean(dim=0, keepdim=True))
    assert final_sv[2].item() > 0.10, f"Third singular value failed to expand: {final_sv[2].item()}"
    assert final_sv[5].item() > 0.05, f"Sixth singular value failed to expand: {final_sv[5].item()}"


# ============================================================================
# Section 5: ArcFace Angular Separation & Class Balancing
# ============================================================================

def test_arcface_inter_class_center_angular_margin() -> None:
    """Verify ArcFace enforces angular separation (cos < cos(0.35)) between class centers."""
    torch.manual_seed(42)
    N, D = 120, 64
    margin = 0.35
    scale = 30.0

    # Realistic financial class distribution: 90% HOLD (108), 5% BUY (6), 5% SELL (6)
    targets = torch.tensor([0] * 108 + [1] * 6 + [2] * 6, dtype=torch.long)
    Z = torch.randn(N, D, requires_grad=True)

    reg = RepresentationRegularizer(hidden_dim=D, margin=margin, scale=scale)

    optimizer = torch.optim.Adam([Z, reg.arc_weights], lr=0.03)
    for _ in range(80):
        optimizer.zero_grad()
        loss, _ = reg.compute_arcface_loss(Z, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        norm_w = F.normalize(reg.arc_weights, p=2, dim=-1)
        center_cosines = norm_w @ norm_w.T

        c01 = center_cosines[0, 1].item()
        c02 = center_cosines[0, 2].item()

        margin_cos = np.cos(margin)
        assert c01 < margin_cos, f"HOLD and BUY centers not separated: cos={c01} >= {margin_cos}"
        assert c02 < margin_cos, f"HOLD and SELL centers not separated: cos={c02} >= {margin_cos}"


def test_arcface_intra_class_compactness() -> None:
    """Verify samples cluster with positive alignment and maintain distinct angular separation."""
    torch.manual_seed(42)
    N, D = 40, 32
    targets = torch.tensor([0] * 30 + [1] * 5 + [2] * 5, dtype=torch.long)
    base = F.normalize(torch.randn(3, D), dim=-1)
    Z = (base[targets] + torch.randn(N, D) * 0.10).requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D, margin=0.35, scale=30.0)

    optimizer = torch.optim.Adam([Z, reg.arc_weights], lr=0.03)
    for _ in range(50):
        optimizer.zero_grad()
        loss, _ = reg.compute_arcface_loss(Z, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        norm_z = F.normalize(Z, p=2, dim=-1)
        norm_w = F.normalize(reg.arc_weights, p=2, dim=-1)

        for c in range(3):
            mask = (targets == c)
            target_sim = (norm_z[mask] @ norm_w[c]).mean().item()
            other_sim = max((norm_z[mask] @ norm_w[o]).mean().item() for o in range(3) if o != c)
            assert target_sim > 0.55, f"Class {c} target alignment too low: {target_sim}"
            assert target_sim > other_sim + 0.30, f"Class {c} margin separation failed: target={target_sim}, other={other_sim}"


def test_inverse_frequency_gradient_amplification() -> None:
    """Verify inverse class frequency weighting amplifies minority entry gradients ~18x over HOLD."""
    D = 16
    reg = RepresentationRegularizer(hidden_dim=D)

    with torch.no_grad():
        reg.arc_weights.copy_(torch.eye(3, D))

    # Evaluate both HOLD (center 0) and BUY (center 1) in the same batch
    # so cross_entropy with inverse class weights distributes gradients properly
    z = torch.zeros(2, D)
    z[0, 0] = 0.5
    z[0, 3] = np.sqrt(0.75)
    z[1, 1] = 0.5
    z[1, 3] = np.sqrt(0.75)
    z.requires_grad_(True)

    targets = torch.tensor([0, 1], dtype=torch.long)
    loss, _ = reg.compute_arcface_loss(z, targets)
    loss.backward()

    norm_hold = z.grad[0].norm().item()
    norm_buy = z.grad[1].norm().item()

    ratio = norm_buy / max(norm_hold, 1e-8)
    expected_ratio = DEFAULT_CLASS_WEIGHTS[1] / DEFAULT_CLASS_WEIGHTS[0]  # ~18.02

    assert 14.0 <= ratio <= 22.0, (
        f"Gradient amplification ratio was {ratio:.2f}, expected approximately {expected_ratio:.2f}"
    )


# ============================================================================
# Section 6: Zero Inference Impact & System Invariants
# ============================================================================

def test_state_dict_keys_and_parameter_count_immutability() -> None:
    """Verify policy state_dict and parameter count are 100% immutable by regularizer attachment."""
    policy_standalone = SharedFiveTradingPolicy()
    keys_standalone = set(policy_standalone.state_dict().keys())
    params_standalone = sum(p.numel() for p in policy_standalone.parameters())

    policy_trained = SharedFiveTradingPolicy()
    trainer = FivePPOTrainer(policy_trained)
    trainer.attach_representation_regularizer(hidden_dim=4096)

    keys_trained = set(policy_trained.state_dict().keys())
    params_trained = sum(p.numel() for p in policy_trained.parameters())

    # 0 new parameters invariant: keys and total parameters in policy must be identical
    assert keys_standalone == keys_trained, "Policy state_dict keys must remain identical"
    assert params_standalone == params_trained, (
        f"0 new parameters invariant violated: standalone={params_standalone}, trained={params_trained}"
    )
    assert params_standalone > 0


def test_eval_mode_forward_pass_identical_behavior() -> None:
    """Verify policy forward pass in eval mode produces identical action wire output with 0 regularizer overhead."""
    torch.manual_seed(42)
    policy = SharedFiveTradingPolicy()
    policy.eval()

    memories = [FiveAgentMemory.zeros(device=torch.device("cpu")) for _ in range(FIVE_AGENT_COUNT)]
    obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)

    results1, _, probe1 = policy.forward_five(obs, memories, deterministic=True)

    # Attach regularizer to trainer referencing policy
    trainer = FivePPOTrainer(policy)
    trainer.attach_representation_regularizer(hidden_dim=4096)

    results2, _, probe2 = policy.forward_five(obs, memories, deterministic=True)

    for r1, r2 in zip(results1, results2):
        assert np.array_equal(r1.output_vector, r2.output_vector)
    assert torch.equal(probe1["value"], probe2["value"])


def test_governor_hold_invariance_uncompromised() -> None:
    """Verify Governor (Agent 4) action distribution remains strictly locked to HOLD and mapped to superclass 0."""
    policy = SharedFiveTradingPolicy()
    policy.heads.lock_non_entry_to_hold = True
    memories = [FiveAgentMemory.zeros(device=torch.device("cpu")) for _ in range(FIVE_AGENT_COUNT)]
    obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)

    results, _, _ = policy.forward_five(obs, memories)

    # Agent 4 is Governor (index 4)
    gov_action_idx = results[4].action_index
    assert gov_action_idx == 0, f"Governor must choose HOLD (action 0), got action {gov_action_idx}"

    # Target derivation must map Governor action strictly to superclass 0
    gov_target = RepresentationRegularizer.derive_targets(torch.tensor([gov_action_idx]))
    assert gov_target.item() == 0, f"Governor target must map to superclass 0 (HOLD), got {gov_target.item()}"
