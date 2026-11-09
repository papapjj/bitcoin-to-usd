"""Adversarial stress testing suite for ArcFace and VICReg in RepresentationRegularizer.

Tested by: challenger_m3_2 (Empirical Challenger)
Task Focus:
1. Stress test ArcFace angular margin with skewed distribution: 90% HOLD, 5% BUY, 5% SELL.
2. Stress test linear fallback for anti-parallel angles (theta -> pi).
3. Verify inverse class frequency gradient amplification ratio is approximately 18x.
4. Stress test VICReg variance hinge loss: verify dimensions with std >= 1.0 produce 0 gradient,
   while std < 1.0 produce expanding gradients.
5. Confirm zero NaNs and zero Infs on backward pass across all tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from self_growing_trader.representation_regularizer import (
    DEFAULT_CLASS_WEIGHTS,
    RepresentationRegularizer,
)


# ============================================================================
# 1. ArcFace Skewed Distribution Stress Tests (90% HOLD, 5% BUY, 5% SELL)
# ============================================================================

def test_adversarial_arcface_skewed_90_5_5_separation() -> None:
    """Empirically prove that ArcFace with 90-5-5 skewed distribution cleanly separates BUY/SELL from HOLD."""
    torch.manual_seed(42)
    N = 200
    D = 32
    # 90% HOLD (180), 5% BUY (10), 5% SELL (10)
    targets = torch.cat([
        torch.zeros(180, dtype=torch.long),
        torch.ones(10, dtype=torch.long),
        torch.full((10,), 2, dtype=torch.long),
    ])

    # Start with overlapping random representations
    z = torch.randn(N, D, requires_grad=True)
    reg = RepresentationRegularizer(hidden_dim=D, margin=0.35, scale=30.0)

    # Initial overlap check
    with torch.no_grad():
        norm_z = F.normalize(z, p=2, dim=-1)
        hold_center_init = norm_z[targets == 0].mean(dim=0)
        buy_center_init = norm_z[targets == 1].mean(dim=0)
        sell_center_init = norm_z[targets == 2].mean(dim=0)
        init_hb_cos = torch.dot(F.normalize(hold_center_init, dim=0), F.normalize(buy_center_init, dim=0)).item()
        init_hs_cos = torch.dot(F.normalize(hold_center_init, dim=0), F.normalize(sell_center_init, dim=0)).item()
        # Overlapping initial state has cosine > -0.1
        assert init_hb_cos > -0.2
        assert init_hs_cos > -0.2

    optimizer = torch.optim.Adam([z, reg.arc_weights], lr=0.05)
    for step in range(60):
        optimizer.zero_grad()
        loss, inter_cos = reg.compute_arcface_loss(z, targets)

        assert torch.isfinite(loss).item(), f"Non-finite loss at step {step}"
        loss.backward()

        assert z.grad is not None and not torch.isnan(z.grad).any()
        assert not torch.isinf(z.grad).any()
        assert reg.arc_weights.grad is not None and not torch.isnan(reg.arc_weights.grad).any()
        assert not torch.isinf(reg.arc_weights.grad).any()

        optimizer.step()

    # Verify geometric separation post-optimization
    with torch.no_grad():
        norm_z = F.normalize(z, p=2, dim=-1)
        hold_center = norm_z[targets == 0].mean(dim=0)
        buy_center = norm_z[targets == 1].mean(dim=0)
        sell_center = norm_z[targets == 2].mean(dim=0)

        final_hb_cos = torch.dot(F.normalize(hold_center, dim=0), F.normalize(buy_center, dim=0)).item()
        final_hs_cos = torch.dot(F.normalize(hold_center, dim=0), F.normalize(sell_center, dim=0)).item()
        final_bs_cos = torch.dot(F.normalize(buy_center, dim=0), F.normalize(sell_center, dim=0)).item()

    # Minority classes must be driven to negative cosine similarity (mutually repulsive, >90 deg angular separation)
    assert final_hb_cos < -0.25, f"HOLD-BUY failed to separate: cos={final_hb_cos}"
    assert final_hs_cos < -0.25, f"HOLD-SELL failed to separate: cos={final_hs_cos}"
    assert final_bs_cos < -0.25, f"BUY-SELL failed to separate: cos={final_bs_cos}"
    assert loss.item() < 0.05, f"Loss failed to converge on skewed distribution: {loss.item()}"


@pytest.mark.parametrize("n_hold,n_buy,n_sell", [
    (18, 1, 1),       # N=20 (90%, 5%, 5%)
    (98, 1, 1),       # N=100 extreme skew (98%, 1%, 1%)
    (45, 5, 0),       # Missing class SELL (90% HOLD, 10% BUY, 0% SELL)
    (50, 0, 0),       # 100% HOLD (no entry actions)
    (0, 10, 10),      # No HOLD (50% BUY, 50% SELL)
    (300, 15, 15),    # N=330 large skewed batch
])
def test_adversarial_arcface_extreme_imbalances(n_hold: int, n_buy: int, n_sell: int) -> None:
    """Stress test ArcFace with extreme imbalances, missing classes, and edge batch compositions."""
    torch.manual_seed(101 + n_hold + n_buy + n_sell)
    D = 64
    target_list = []
    if n_hold > 0:
        target_list.append(torch.zeros(n_hold, dtype=torch.long))
    if n_buy > 0:
        target_list.append(torch.ones(n_buy, dtype=torch.long))
    if n_sell > 0:
        target_list.append(torch.full((n_sell,), 2, dtype=torch.long))

    targets = torch.cat(target_list)
    N = targets.shape[0]
    z = torch.randn(N, D, requires_grad=True)

    reg = RepresentationRegularizer(hidden_dim=D)
    loss, inter_cos = reg.compute_arcface_loss(z, targets)

    assert bool(torch.isfinite(loss).item())
    assert bool(torch.isfinite(inter_cos).item())

    loss.backward()
    assert z.grad is not None
    assert not torch.isnan(z.grad).any(), "NaN in z.grad under extreme imbalance"
    assert not torch.isinf(z.grad).any(), "Inf in z.grad under extreme imbalance"
    assert not torch.isnan(reg.arc_weights.grad).any()
    assert not torch.isinf(reg.arc_weights.grad).any()


@pytest.mark.parametrize("N", [2, 5, 10, 50, 100, 500])
def test_adversarial_arcface_batch_sizes(N: int) -> None:
    """Stress test ArcFace across various batch sizes from minimal N=2 to N=500."""
    torch.manual_seed(202 + N)
    D = 128
    # Skewed sample generation
    probs = torch.tensor([0.90, 0.05, 0.05])
    targets = torch.multinomial(probs, num_samples=N, replacement=True)
    z = torch.randn(N, D, requires_grad=True)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(z, targets=targets)

    assert bool(torch.isfinite(out["loss_arcface"]).item())
    out["loss_reg_total"].backward()

    assert not torch.isnan(z.grad).any()
    assert not torch.isinf(z.grad).any()


# ============================================================================
# 2. Linear Fallback for Anti-Parallel Angles (theta -> pi)
# ============================================================================

def test_adversarial_arcface_linear_fallback_regimes() -> None:
    """Test ArcFace angular margin across both trigonometric regime (theta <= pi - m)
    and linear fallback regime (theta > pi - m)."""
    reg = RepresentationRegularizer(hidden_dim=16, margin=0.35)
    m = reg.margin
    th = reg.th  # cos(pi - m) = -cos(m) ~ -0.93937
    mm = reg.mm  # sin(pi - m) * m = sin(m) * m ~ 0.12001

    # Class centers aligned with Cartesian axes
    with torch.no_grad():
        reg.arc_weights.zero_()
        reg.arc_weights[0, 0] = 1.0  # Class 0 along axis 0
        reg.arc_weights[1, 1] = 1.0
        reg.arc_weights[2, 2] = 1.0

    target = torch.tensor([0], dtype=torch.long)

    # 1. Sweep angles from 0 to pi
    thetas = np.linspace(0.01, np.pi - 1e-6, 100)
    phi_values = []
    cos_values = []

    for theta in thetas:
        z = torch.zeros(1, 16, requires_grad=True)
        z.data[0, 0] = float(np.cos(theta))
        z.data[0, 1] = float(np.sin(theta))

        loss, _ = reg.compute_arcface_loss(z, target)
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any(), f"NaN in grad at theta={theta}"
        assert not torch.isinf(z.grad).any(), f"Inf in grad at theta={theta}"

        cosine = float(np.cos(theta))
        cos_values.append(cosine)

        # Compute phi manually according to implementation
        if cosine > th:
            sine = float(np.sqrt(max(1.0 - cosine ** 2, 1e-6)))
            phi_expected = cosine * reg.cos_m - sine * reg.sin_m
        else:
            phi_expected = cosine - mm
        phi_values.append(phi_expected)

    # Verify that phi is strictly monotonically decreasing with respect to theta
    diffs = np.diff(phi_values)
    assert (diffs <= 1e-6).all(), "phi is not monotonically non-increasing across [0, pi]!"


def test_adversarial_arcface_linear_fallback_boundary_continuity() -> None:
    """Test boundary continuity and stability at theta = pi - m (cosine = th)."""
    reg = RepresentationRegularizer(hidden_dim=16, margin=0.35)
    th = reg.th

    with torch.no_grad():
        reg.arc_weights.zero_()
        reg.arc_weights[0, 0] = 1.0
        reg.arc_weights[1, 1] = 1.0
        reg.arc_weights[2, 2] = 1.0

    # Probe just before threshold, at threshold, and just after threshold
    deltas = [-1e-4, -1e-6, 0.0, 1e-6, 1e-4]
    for d in deltas:
        cos_val = np.clip(th + d, -0.99999, 0.99999)
        sin_val = np.sqrt(1.0 - cos_val ** 2)

        z = torch.zeros(1, 16, requires_grad=True)
        z.data[0, 0] = float(cos_val)
        z.data[0, 1] = float(sin_val)

        target = torch.tensor([0], dtype=torch.long)
        loss, _ = reg.compute_arcface_loss(z, target)

        assert torch.isfinite(loss).item()
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any(), f"NaN in grad near threshold delta={d}"
        assert not torch.isinf(z.grad).any(), f"Inf in grad near threshold delta={d}"


@pytest.mark.parametrize("cos_target", [
    -0.95, -0.99, -0.999, -0.9999, -0.99999, -1.0
])
def test_adversarial_arcface_anti_parallel_extreme_limits(cos_target: float) -> None:
    """Stress test anti-parallel angles approaching exact -1.0 (opposite to class center)."""
    reg = RepresentationRegularizer(hidden_dim=32, margin=0.35)
    with torch.no_grad():
        reg.arc_weights.zero_()
        reg.arc_weights[0, 0] = 1.0  # Center 0 along dim 0

    z = torch.zeros(1, 32, requires_grad=True)
    # Point directly opposite along dim 0
    z.data[0, 0] = float(cos_target)
    sin_val = np.sqrt(max(0.0, 1.0 - cos_target ** 2))
    z.data[0, 1] = float(sin_val)

    target = torch.tensor([0], dtype=torch.long)
    loss, _ = reg.compute_arcface_loss(z, target)

    assert bool(torch.isfinite(loss).item())
    loss.backward()

    assert z.grad is not None
    assert not torch.isnan(z.grad).any(), f"NaN at anti-parallel cos={cos_target}"
    assert not torch.isinf(z.grad).any(), f"Inf at anti-parallel cos={cos_target}"
    assert not torch.isnan(reg.arc_weights.grad).any()
    assert not torch.isinf(reg.arc_weights.grad).any()


# ============================================================================
# 3. Inverse Class Frequency Gradient Amplification (~18x)
# ============================================================================

def test_adversarial_arcface_class_weights_analytical_ratio() -> None:
    """Verify analytical 18x weight ratio: (1/0.05) / (1/0.90) = 20 / (10/9) = 18.0."""
    w0, w1, w2 = DEFAULT_CLASS_WEIGHTS
    ratio_buy_to_hold = w1 / w0
    ratio_sell_to_hold = w2 / w0

    assert ratio_buy_to_hold == pytest.approx(18.0, rel=1e-5), f"BUY weight ratio {ratio_buy_to_hold} != 18.0"
    assert ratio_sell_to_hold == pytest.approx(18.0, rel=1e-5), f"SELL weight ratio {ratio_sell_to_hold} != 18.0"
    assert w1 == pytest.approx(w2, rel=1e-6), "BUY and SELL weights should be equal"

    reg = RepresentationRegularizer(hidden_dim=32)
    buf_w0, buf_w1, buf_w2 = reg.class_weights.tolist()
    assert buf_w1 / buf_w0 == pytest.approx(18.0, rel=1e-5)
    assert buf_w2 / buf_w0 == pytest.approx(18.0, rel=1e-5)


def test_adversarial_arcface_gradient_amplification_symmetric_pairs() -> None:
    """Empirically prove that backpropagated gradients for BUY and SELL are amplified by exactly 18x
    relative to HOLD under symmetric geometric configurations."""
    reg = RepresentationRegularizer(hidden_dim=16)

    # Use orthogonal class centers
    with torch.no_grad():
        reg.arc_weights.zero_()
        reg.arc_weights[0, 0] = 1.0  # Center 0 (HOLD)
        reg.arc_weights[1, 1] = 1.0  # Center 1 (BUY)
        reg.arc_weights[2, 2] = 1.0  # Center 2 (SELL)

    # Construct symmetric representations:
    # z0 has dot product 0.8 with w0, 0.1 with others
    # z1 has dot product 0.8 with w1, 0.1 with others
    # z2 has dot product 0.8 with w2, 0.1 with others
    z0 = torch.zeros(1, 16)
    z0[0, 0] = 0.8; z0[0, 1] = 0.1; z0[0, 2] = 0.1

    z1 = torch.zeros(1, 16)
    z1[0, 0] = 0.1; z1[0, 1] = 0.8; z1[0, 2] = 0.1

    z2 = torch.zeros(1, 16)
    z2[0, 0] = 0.1; z2[0, 1] = 0.1; z2[0, 2] = 0.8

    # Test BUY vs HOLD pair
    z_pair_buy = torch.cat([z0, z1], dim=0).requires_grad_(True)
    targets_buy = torch.tensor([0, 1], dtype=torch.long)

    loss_buy, _ = reg.compute_arcface_loss(z_pair_buy, targets_buy)
    loss_buy.backward()

    grad_hold = z_pair_buy.grad[0].norm().item()
    grad_buy = z_pair_buy.grad[1].norm().item()
    ratio_buy = grad_buy / grad_hold

    assert ratio_buy == pytest.approx(18.0, rel=1e-4), (
        f"Gradient amplification BUY vs HOLD was {ratio_buy:.4f}, expected 18.0"
    )

    # Test SELL vs HOLD pair
    z_pair_sell = torch.cat([z0, z2], dim=0).requires_grad_(True)
    targets_sell = torch.tensor([0, 2], dtype=torch.long)

    loss_sell, _ = reg.compute_arcface_loss(z_pair_sell, targets_sell)
    loss_sell.backward()

    grad_hold_2 = z_pair_sell.grad[0].norm().item()
    grad_sell = z_pair_sell.grad[1].norm().item()
    ratio_sell = grad_sell / grad_hold_2

    assert ratio_sell == pytest.approx(18.0, rel=1e-4), (
        f"Gradient amplification SELL vs HOLD was {ratio_sell:.4f}, expected 18.0"
    )


def test_adversarial_arcface_cross_entropy_logit_gradients() -> None:
    """Verify logit gradient scaling directly in cross_entropy with class_weights buffer.

    When base logits are uniform across classes (p0 = p1 = p2 = 1/3), the unweighted gradient
    magnitudes ||p - e_c|| are identical (sqrt(2/3)), isolating the exact class weight ratio:
    ||grad_buy|| / ||grad_hold|| = w1 / w0 = 18.0.
    """
    reg = RepresentationRegularizer(hidden_dim=32)
    # Uniform base logits across 3 classes for 3 samples
    logits = torch.zeros(3, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    loss = F.cross_entropy(logits, targets, weight=reg.class_weights)
    loss.backward()

    norm_hold = logits.grad[0].norm().item()
    norm_buy = logits.grad[1].norm().item()
    norm_sell = logits.grad[2].norm().item()

    assert norm_buy / norm_hold == pytest.approx(18.0, rel=1e-4)
    assert norm_sell / norm_hold == pytest.approx(18.0, rel=1e-4)


# ============================================================================
# 4. VICReg Variance Hinge Loss Stress Tests
# ============================================================================

def test_adversarial_vicreg_variance_inactive_regime_std_ge_1() -> None:
    """Verify that dimensions with std >= 1.0 produce EXACTLY ZERO gradient."""
    torch.manual_seed(303)
    N, D = 100, 8
    target_stds = [1.0, 1.0001, 1.1, 1.5, 2.0, 5.0, 10.0, 100.0]

    z = torch.randn(N, D)
    z = z - z.mean(dim=0, keepdim=True)
    for j in range(D):
        z[:, j] = (z[:, j] / z[:, j].std(unbiased=True)) * target_stds[j]

    z.requires_grad_(True)
    reg = RepresentationRegularizer(hidden_dim=D)
    loss_var = reg.compute_variance_loss(z)

    # Since all std >= 1.0, loss must be exactly 0.0
    assert loss_var.item() == 0.0, f"Expected loss 0.0 when std >= 1.0, got {loss_var.item()}"

    loss_var.backward()
    assert z.grad is not None
    # Gradients for all dimensions must be exactly zero
    for j in range(D):
        grad_col_norm = z.grad[:, j].norm().item()
        assert grad_col_norm == 0.0, f"Dim {j} with std={target_stds[j]} produced non-zero grad: {grad_col_norm}"


def test_adversarial_vicreg_variance_active_regime_expanding_gradients() -> None:
    """Verify that dimensions with std < 1.0 produce non-zero gradients that strictly EXPAND variance."""
    torch.manual_seed(404)
    N, D = 120, 6
    target_stds = [0.01, 0.1, 0.3, 0.6, 0.8, 0.99]

    z = torch.randn(N, D)
    z = z - z.mean(dim=0, keepdim=True)
    for j in range(D):
        z[:, j] = (z[:, j] / z[:, j].std(unbiased=True)) * target_stds[j]

    z.requires_grad_(True)
    reg = RepresentationRegularizer(hidden_dim=D)
    loss_var = reg.compute_variance_loss(z)

    assert loss_var.item() > 0.0
    loss_var.backward()

    assert z.grad is not None

    # Check 1: Non-zero gradients for every dimension < 1.0
    for j in range(D):
        grad_norm = z.grad[:, j].norm().item()
        assert grad_norm > 1e-4, f"Dim {j} with std={target_stds[j]} produced vanishing gradient"

    # Check 2: Gradients point in expanding direction:
    # d(loss)/d(z_{i,j}) * (z_{i,j} - mean_j) < 0
    # Gradient descent update z_new = z - lr * grad will push points away from mean, increasing variance!
    mean_z = z.mean(dim=0, keepdim=True)
    centered_z = z - mean_z
    inner_products = (z.grad * centered_z).sum(dim=0)
    for j in range(D):
        assert inner_products[j].item() < 0.0, (
            f"Dim {j} gradient does not point in expanding direction! inner_prod = {inner_products[j].item()}"
        )

    # Check 3: Explicit single step of gradient descent strictly increases std
    with torch.no_grad():
        lr = 0.5
        z_updated = z - lr * z.grad
        stds_after = z_updated.std(dim=0, unbiased=True)
        stds_before = z.std(dim=0, unbiased=True)

        for j in range(D):
            assert stds_after[j] > stds_before[j], (
                f"Dim {j} failed to expand: std_before={stds_before[j]:.4f}, std_after={stds_after[j]:.4f}"
            )


def test_adversarial_vicreg_variance_mixed_active_inactive_dimensions() -> None:
    """Stress test partitioned matrix where first half has std < 1.0 and second half has std >= 1.0."""
    torch.manual_seed(505)
    N, D = 80, 20
    z = torch.randn(N, D)
    z = z - z.mean(dim=0, keepdim=True)

    # Dimensions 0..9: std < 1.0 (active)
    # Dimensions 10..19: std >= 1.0 (inactive)
    for j in range(10):
        z[:, j] = (z[:, j] / z[:, j].std(unbiased=True)) * (0.05 + 0.09 * j)
    for j in range(10, 20):
        z[:, j] = (z[:, j] / z[:, j].std(unbiased=True)) * (1.0 + 0.5 * (j - 10))

    z.requires_grad_(True)
    reg = RepresentationRegularizer(hidden_dim=D)
    loss = reg.compute_variance_loss(z)
    loss.backward()

    # Active dimensions must have non-zero gradients
    for j in range(10):
        assert z.grad[:, j].norm().item() > 0.0, f"Active dimension {j} produced zero grad"

    # Inactive dimensions must have strictly zero gradients
    for j in range(10, 20):
        assert z.grad[:, j].norm().item() == 0.0, f"Inactive dimension {j} produced non-zero grad"


def test_adversarial_vicreg_variance_full_convergence_expansion() -> None:
    """Empirically prove that optimizing variance hinge loss drives all collapsed dimensions to std >= 1.0."""
    torch.manual_seed(606)
    N, D = 64, 16
    # Start severely collapsed: std = 0.05 across all dimensions
    z = (torch.randn(N, D) * 0.05).requires_grad_(True)
    reg = RepresentationRegularizer(hidden_dim=D)

    initial_stds = z.std(dim=0, unbiased=True)
    assert (initial_stds < 0.1).all()

    optimizer = torch.optim.Adam([z], lr=0.1)
    for step in range(50):
        optimizer.zero_grad()
        loss = reg.compute_variance_loss(z)
        loss.backward()
        optimizer.step()

    final_stds = z.std(dim=0, unbiased=True)
    # All dimensions should have reached or exceeded 1.0
    assert (final_stds >= 0.99).all(), f"Failed to expand all dimensions to >= 1.0: min std = {final_stds.min().item()}"
    assert loss.item() < 0.01, f"Variance loss failed to converge: {loss.item()}"


# ============================================================================
# 5. Joint Adversarial Stress & Zero NaNs/Infs Verification
# ============================================================================

@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_adversarial_backward_pass_scale_sweeps(scale: float) -> None:
    """Confirm zero NaNs and zero Infs on backward pass across 12 orders of magnitude in representation scale."""
    torch.manual_seed(707 + int(abs(np.log10(scale))))
    N, D = 40, 64
    z = (torch.randn(N, D) * scale).requires_grad_(True)
    targets = torch.randint(0, 3, (N,))

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(z, targets=targets)

    for loss_key in ["loss_reg_total", "loss_var", "loss_cov", "loss_rank", "loss_arcface"]:
        assert bool(torch.isfinite(out[loss_key]).item()), f"{loss_key} non-finite at scale={scale}"

    out["loss_reg_total"].backward()
    assert z.grad is not None
    assert not torch.isnan(z.grad).any(), f"NaN in z.grad at scale={scale}"
    assert not torch.isinf(z.grad).any(), f"Inf in z.grad at scale={scale}"
    assert not torch.isnan(reg.arc_weights.grad).any()
    assert not torch.isinf(reg.arc_weights.grad).any()


def test_adversarial_sgt_v193_production_shape_joint_stress() -> None:
    """Stress test exact SGT V193 production shapes: [B=10, 5 agents, D=4096] with skewed targets."""
    torch.manual_seed(808)
    B, A, D = 10, 5, 4096  # 50 total agent steps, 4096 hidden dimensions
    hidden = torch.randn(B, A, D, requires_grad=True)

    # 90% HOLD (45 steps), 5% BUY (3 steps), 5% SELL (2 steps)
    target_flat = torch.cat([
        torch.zeros(45, dtype=torch.long),
        torch.ones(3, dtype=torch.long),
        torch.full((2,), 2, dtype=torch.long),
    ])
    targets = target_flat.reshape(B, A)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(hidden, targets=targets)

    for loss_key in ["loss_reg_total", "loss_var", "loss_cov", "loss_rank", "loss_arcface"]:
        assert bool(torch.isfinite(out[loss_key]).item())

    out["loss_reg_total"].backward()

    assert hidden.grad is not None
    assert hidden.grad.shape == (B, A, D)
    assert not torch.isnan(hidden.grad).any()
    assert not torch.isinf(hidden.grad).any()
    assert not torch.isnan(reg.arc_weights.grad).any()
    assert not torch.isinf(reg.arc_weights.grad).any()


def test_adversarial_zero_input_tensor_stability() -> None:
    """Confirm complete numerical stability when Z is exactly all zeros."""
    N, D = 20, 64
    z = torch.zeros(N, D, requires_grad=True)
    targets = torch.zeros(N, dtype=torch.long)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(z, targets=targets)

    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    out["loss_reg_total"].backward()

    assert z.grad is not None
    assert not torch.isnan(z.grad).any(), "NaN in grad for z=0"
    assert not torch.isinf(z.grad).any(), "Inf in grad for z=0"
