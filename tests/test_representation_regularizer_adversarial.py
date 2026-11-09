"""Adversarial stress testing suite for RankMe and Sylvester identity in RepresentationRegularizer.

Tested by: challenger_m3_1 (Empirical Challenger)
Focus areas:
1. Sylvester determinant identity primal vs dual equivalence across (N, D) grids and gamma scales.
2. Collapsed representations (r_eff <= 2.0, rank-1, rank-2).
3. Degenerate matrices: Zero matrices, identical row vectors, duplicate rows/cols.
4. Extreme scale representations (10^-6, 10^6, and mixed condition number 10^12).
5. Dynamic spectrum expansion and r_eff growth during optimization.
6. Production dimensions: N=5, D=4096 and N=50, D=4096.
7. Zero NaNs and zero Infs on backward pass across all stress conditions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from self_growing_trader.representation_regularizer import RepresentationRegularizer


# ============================================================================
# Helper Functions & Oracles
# ============================================================================

def compute_primal_rankme_loss(z: torch.Tensor, gamma: float) -> torch.Tensor:
    """Independent oracle for primal RankMe loss: - (1/D) * ln det(gamma*I_D + (1/N)*Z^T Z) + (1/(DN))*||Z||_F^2."""
    N, D = z.shape
    device = z.device
    dtype = z.dtype
    C = (z.T @ z) / max(N, 1)
    C_reg = C + gamma * torch.eye(D, device=device, dtype=dtype)
    sign, logdet = torch.linalg.slogdet(C_reg)
    assert (sign > 0).all(), "Primal matrix must be positive definite"
    trace_term = (z ** 2).sum() / (D * max(N, 1))
    return -logdet / D + trace_term


def compute_dual_rankme_loss(z: torch.Tensor, gamma: float) -> torch.Tensor:
    """Independent oracle for dual RankMe loss using Sylvester's identity:
    ln det(primal) = (D - N)*ln(gamma) + ln det(gamma*I_N + (1/N)*Z Z^T).
    """
    N, D = z.shape
    device = z.device
    dtype = z.dtype
    K = (z @ z.T) / max(N, 1)
    K_reg = K + gamma * torch.eye(N, device=device, dtype=dtype)
    sign, logdet_dual = torch.linalg.slogdet(K_reg)
    assert (sign > 0).all(), "Dual matrix must be positive definite"
    logdet_total = float(D - N) * np.log(gamma) + logdet_dual
    trace_term = (z ** 2).sum() / (D * max(N, 1))
    return -logdet_total / D + trace_term


# ============================================================================
# 1. Sylvester Determinant Identity Primal vs Dual Stress Sweep
# ============================================================================

@pytest.mark.parametrize("N,D", [
    (5, 20),
    (10, 50),
    (20, 100),
    (32, 128),
    (64, 256),
    (10, 10),
    (30, 30),
    (15, 60),
])
@pytest.mark.parametrize("gamma", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
def test_adversarial_sylvester_grid(N: int, D: int, gamma: float) -> None:
    """Empirically verify Sylvester identity primal vs dual loss and gradients across (N, D) and gamma."""
    torch.manual_seed(42 + N + D + int(gamma * 1000))
    z_primal = torch.randn(N, D, dtype=torch.float64, requires_grad=True)
    z_dual = z_primal.detach().clone().requires_grad_(True)

    loss_primal = compute_primal_rankme_loss(z_primal, gamma)
    loss_dual = compute_dual_rankme_loss(z_dual, gamma)

    # 1. Value equivalence
    assert torch.isclose(loss_primal, loss_dual, rtol=1e-5, atol=1e-6), (
        f"Value mismatch for (N={N}, D={D}, gamma={gamma}): primal={loss_primal.item()}, dual={loss_dual.item()}"
    )

    # 2. Gradient backprop equivalence
    loss_primal.backward()
    loss_dual.backward()

    assert z_primal.grad is not None and z_dual.grad is not None
    assert not torch.isnan(z_primal.grad).any(), "NaN in primal gradient"
    assert not torch.isinf(z_primal.grad).any(), "Inf in primal gradient"
    assert not torch.isnan(z_dual.grad).any(), "NaN in dual gradient"
    assert not torch.isinf(z_dual.grad).any(), "Inf in dual gradient"

    max_grad_diff = (z_primal.grad - z_dual.grad).abs().max().item()
    assert max_grad_diff < 1e-5, (
        f"Gradient mismatch for (N={N}, D={D}, gamma={gamma}): max diff = {max_grad_diff}"
    )


# ============================================================================
# 2. Collapsed Representations (r_eff <= 2.0)
# ============================================================================

def test_adversarial_collapsed_representations_exact_rank1() -> None:
    """Test RankMe and full regularizer on exact rank-1 matrix (all points on 1 line)."""
    torch.manual_seed(101)
    N, D = 40, 64
    u = torch.randn(N, 1)
    v = torch.randn(1, D)
    Z = (u @ v).detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    r_eff = reg.compute_effective_rank(Z)
    assert 1.0 <= r_eff <= 1.05, f"Exact rank-1 should have r_eff ~ 1.0, got {r_eff}"

    out = reg(Z)
    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any(), "NaN in rank-1 grad"
    assert not torch.isinf(Z.grad).any(), "Inf in rank-1 grad"
    assert torch.count_nonzero(Z.grad).item() > 0


def test_adversarial_collapsed_representations_rank2_dominant() -> None:
    """Test RankMe on severely collapsed rank-2 representation with r_eff <= 2.0."""
    torch.manual_seed(202)
    N, D = 50, 128
    # Construct rank-2 representation with dominant first component
    u1 = torch.randn(N, 1)
    v1 = torch.randn(1, D)
    u2 = torch.randn(N, 1)
    v2 = torch.randn(1, D)
    Z = (100.0 * (u1 @ v1) + 1.0 * (u2 @ v2)).detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    r_eff = reg.compute_effective_rank(Z)
    assert r_eff <= 2.0, f"Expected r_eff <= 2.0, got {r_eff}"

    out = reg(Z)
    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any(), "NaN in rank-2 grad"
    assert not torch.isinf(Z.grad).any(), "Inf in rank-2 grad"


def test_adversarial_exponential_spectrum_decay() -> None:
    """Test representation with exponentially decaying spectrum sigma_k = exp(-k*3)."""
    torch.manual_seed(303)
    N, D = 40, 40
    U, _ = torch.linalg.qr(torch.randn(N, N))
    V, _ = torch.linalg.qr(torch.randn(D, D))
    sigmas = torch.exp(-torch.arange(min(N, D), dtype=torch.float32) * 3.0)
    S = torch.diag(sigmas)
    Z = (U @ S @ V.T).detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    r_eff = reg.compute_effective_rank(Z)
    assert r_eff <= 2.0, f"Exponential decay representation should have r_eff <= 2.0, got {r_eff}"

    out = reg(Z)
    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any()
    assert not torch.isinf(Z.grad).any()


# ============================================================================
# 3. Degenerate Matrices: Zero Matrices & Identical Vectors
# ============================================================================

@pytest.mark.parametrize("N,D", [(2, 32), (10, 64), (64, 16), (5, 4096)])
def test_adversarial_zero_matrix_stress(N: int, D: int) -> None:
    """Verify stability of zero matrix Z=0 across both N < D and N >= D dimensions."""
    Z = torch.zeros(N, D, requires_grad=True)
    reg = RepresentationRegularizer(hidden_dim=D)

    out = reg(Z)
    loss = out["loss_reg_total"]
    assert bool(torch.isfinite(loss).item()), f"Loss must be finite for Z=0 (N={N}, D={D})"
    assert out["effective_rank"] == 1.0

    loss.backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any(), f"NaN in grad for Z=0 (N={N}, D={D})"
    assert not torch.isinf(Z.grad).any(), f"Inf in grad for Z=0 (N={N}, D={D})"


@pytest.mark.parametrize("N,D", [(5, 32), (20, 64), (40, 16), (5, 4096)])
def test_adversarial_identical_vectors_stress(N: int, D: int) -> None:
    """Verify stability when all rows of Z are identical non-zero vectors (sample covariance is 0)."""
    torch.manual_seed(404)
    v = torch.randn(1, D)
    Z = v.repeat(N, 1).detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    r_eff = reg.compute_effective_rank(Z)
    assert r_eff == pytest.approx(1.0, abs=1e-5), f"Effective rank of identical vectors must be ~1.0, got {r_eff}"

    out = reg(Z)
    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())
    assert bool(torch.isfinite(out["loss_cov"]).item())
    assert bool(torch.isfinite(out["loss_var"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any(), f"NaN in grad for identical vectors (N={N}, D={D})"
    assert not torch.isinf(Z.grad).any(), f"Inf in grad for identical vectors (N={N}, D={D})"


def test_adversarial_duplicate_and_collinear_features() -> None:
    """Verify stability when representation has duplicate rows, duplicate columns, and zero columns."""
    torch.manual_seed(505)
    N, D = 30, 40
    z_base = torch.randn(N, 10)
    # Replicate features to make 40 columns (rank at most 10)
    Z = z_base.repeat(1, 4)
    # Set 5 columns to exact zero
    Z[:, 0:5] = 0.0
    # Duplicate first 5 rows
    Z[5:10] = Z[0:5]
    Z = Z.detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(Z)
    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any()
    assert not torch.isinf(Z.grad).any()


# ============================================================================
# 4. Extreme Scale Representations (10^-6, 10^6, and Mixed)
# ============================================================================

def test_adversarial_tiny_scale_1e_minus_6() -> None:
    """Verify numerical stability and finite gradients when vectors have scale 10^-6."""
    torch.manual_seed(601)
    N, D = 25, 64
    Z = (torch.randn(N, D) * 1e-6).requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(Z)

    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any(), "NaN in tiny scale grad"
    assert not torch.isinf(Z.grad).any(), "Inf in tiny scale grad"


def test_adversarial_huge_scale_1e_plus_6() -> None:
    """Verify numerical stability and finite gradients when vectors have scale 10^6."""
    torch.manual_seed(602)
    N, D = 25, 64
    Z = (torch.randn(N, D) * 1e6).requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(Z)

    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any(), "NaN in huge scale grad"
    assert not torch.isinf(Z.grad).any(), "Inf in huge scale grad"


def test_adversarial_mixed_extreme_scale_condition_1e12() -> None:
    """Verify stability when representation has mixed feature scales spanning 10^-6 to 10^6 (condition 10^12)."""
    torch.manual_seed(603)
    N, D = 30, 32
    scales = torch.logspace(-6, 6, steps=D)
    Z = (torch.randn(N, D) * scales).requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D)
    out = reg(Z)

    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any(), "NaN in ill-conditioned scale grad"
    assert not torch.isinf(Z.grad).any(), "Inf in ill-conditioned scale grad"


# ============================================================================
# 5. Spectrum Expansion and Effective Rank Growth Dynamics
# ============================================================================

def test_adversarial_rankme_active_spectrum_expansion() -> None:
    """Empirically prove that optimizing RankMe reliably expands singular value spectrum and increases r_eff."""
    torch.manual_seed(701)
    N, D = 64, 32
    # Start with severely collapsed rank-1 representation
    u = torch.randn(N, 1)
    v = torch.randn(1, D)
    Z = (u @ v).detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D, gamma_reg=1e-3)
    initial_r_eff = reg.compute_effective_rank(Z)
    assert initial_r_eff <= 1.05

    initial_sv = torch.linalg.svdvals(Z - Z.mean(dim=0, keepdim=True))
    # Check that initial higher singular values are zero
    assert initial_sv[1].item() < 1e-4

    # Optimize using Adam with RankMe loss + small variance anchor
    optimizer = torch.optim.Adam([Z], lr=0.05)
    history_r_eff = []
    for step in range(60):
        optimizer.zero_grad()
        loss = reg.compute_rankme_loss(Z) + 0.5 * reg.compute_variance_loss(Z)
        loss.backward()

        assert Z.grad is not None
        assert not torch.isnan(Z.grad).any(), f"NaN at step {step}"
        assert not torch.isinf(Z.grad).any(), f"Inf at step {step}"

        optimizer.step()
        with torch.no_grad():
            cur_r_eff = reg.compute_effective_rank(Z)
            history_r_eff.append(cur_r_eff)

    final_r_eff = history_r_eff[-1]
    final_sv = torch.linalg.svdvals(Z - Z.mean(dim=0, keepdim=True))

    # Verification:
    # 1. Effective rank must expand substantially (> 5x initial and > 6.0)
    assert final_r_eff >= initial_r_eff * 5.0, (
        f"r_eff failed to expand >= 5x: initial={initial_r_eff:.2f}, final={final_r_eff:.2f}"
    )
    assert final_r_eff > 6.0, f"Final r_eff must exceed 6.0, got {final_r_eff:.2f}"

    # 2. Singular value spectrum expansion: non-dominant singular values must be non-trivial
    assert final_sv[2].item() > 0.05, f"3rd singular value failed to expand: {final_sv[2].item()}"
    assert final_sv[5].item() > 0.02, f"6th singular value failed to expand: {final_sv[5].item()}"
    assert final_sv[10].item() > 0.01, f"11th singular value failed to expand: {final_sv[10].item()}"

    # 3. Spectrum spread: ratio of (sv[0] / sv[10]) decreased drastically compared to collapse
    ratio_initial = initial_sv[0].item() / max(initial_sv[10].item(), 1e-12)
    ratio_final = final_sv[0].item() / max(final_sv[10].item(), 1e-12)
    assert ratio_final < ratio_initial / 100.0, "Singular value spectrum failed to flatten"


def test_adversarial_full_regularizer_recovery_from_rank2() -> None:
    """Test full multi-objective regularizer recovers collapsed rank-2 representation."""
    torch.manual_seed(702)
    N, D = 64, 32
    # Start with rank-2 collapse (normalized to unit scale like realistic policy features)
    u1, v1 = torch.randn(N, 1), torch.randn(1, D)
    u2, v2 = torch.randn(N, 1), torch.randn(1, D)
    Z_raw = 50.0 * (u1 @ v1) + (u2 @ v2)
    Z = (Z_raw / Z_raw.std()).detach().requires_grad_(True)
    targets = torch.randint(0, 3, (N,))

    reg = RepresentationRegularizer(
        hidden_dim=D,
        var_weight=1.0,
        cov_weight=1.0,
        rank_weight=0.10,
        arc_weight=0.20,
    )
    initial_r_eff = reg.compute_effective_rank(Z)
    assert initial_r_eff <= 2.0

    optimizer = torch.optim.Adam([Z, reg.arc_weights], lr=0.04)
    for step in range(50):
        optimizer.zero_grad()
        out = reg(Z, targets=targets)
        loss = out["loss_reg_total"]
        loss.backward()

        assert not torch.isnan(Z.grad).any(), f"NaN in Z.grad at step {step}"
        assert not torch.isnan(reg.arc_weights.grad).any(), f"NaN in arc_weights.grad at step {step}"

        optimizer.step()

    final_r_eff = reg.compute_effective_rank(Z)
    assert final_r_eff > initial_r_eff * 2.5
    assert final_r_eff > 15.0


def test_adversarial_rankme_float32_scale_cancellation_edge_case() -> None:
    """Empirically document float32 precision boundary: when Z is unnormalized (std > 10) and collapsed.
    
    Cancellation errors in float32 can cause negative eigenvalues in C_reg or K_reg.
    Verify that RepresentationRegularizer handles this gracefully with zero NaNs, zero Infs,
    and falls back to zero loss rather than crashing.
    """
    torch.manual_seed(703)
    N, D = 64, 32
    u1, v1 = torch.randn(N, 1), torch.randn(1, D)
    u2, v2 = torch.randn(N, 1), torch.randn(1, D)
    # Unnormalized scale: std ~ 40
    Z_unnormalized = (50.0 * (u1 @ v1) + (u2 @ v2)).detach().requires_grad_(True)

    reg = RepresentationRegularizer(hidden_dim=D, gamma_reg=1e-3)
    out = reg(Z_unnormalized)

    # Must not crash, loss must be finite
    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())

    # Backward pass must produce zero NaNs and zero Infs
    out["loss_reg_total"].backward()
    assert Z_unnormalized.grad is not None
    assert not torch.isnan(Z_unnormalized.grad).any()
    assert not torch.isinf(Z_unnormalized.grad).any()



# ============================================================================
# 6. SGT V193 Production Dimensions Stress (D=4096, N=5 and N=50)
# ============================================================================

def test_adversarial_sgt_v193_production_dimensions_single_step() -> None:
    """Verify RankMe dual formulation handles exact production shape [5, 4096] (N=5 agents, D=4096)."""
    torch.manual_seed(801)
    N, D = 5, 4096  # 5 agents, 1 time-step
    Z = torch.randn(N, D, requires_grad=True)

    reg = RepresentationRegularizer(hidden_dim=D, gamma_reg=1e-3)

    # Compute loss via dual Sylvester identity (5x5 matrix inversion/logdet instead of 4096x4096!)
    loss = reg.compute_rankme_loss(Z)
    assert bool(torch.isfinite(loss).item())

    loss.backward()
    assert Z.grad is not None
    assert Z.grad.shape == (5, 4096)
    assert not torch.isnan(Z.grad).any()
    assert not torch.isinf(Z.grad).any()


def test_adversarial_sgt_v193_production_dimensions_multi_step() -> None:
    """Verify RankMe dual formulation handles production sequence batch [50, 4096] (T=10 steps, 5 agents)."""
    torch.manual_seed(802)
    N, D = 50, 4096  # 10 steps x 5 agents
    Z = torch.randn(N, D, requires_grad=True)
    targets = torch.randint(0, 3, (N,))

    reg = RepresentationRegularizer(hidden_dim=D, gamma_reg=1e-3)
    out = reg(Z, targets=targets)

    assert bool(torch.isfinite(out["loss_reg_total"]).item())
    assert bool(torch.isfinite(out["loss_rank"]).item())
    assert out["effective_rank"] > 10.0

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert Z.grad.shape == (50, 4096)
    assert not torch.isnan(Z.grad).any()
    assert not torch.isinf(Z.grad).any()
    assert not torch.isnan(reg.arc_weights.grad).any()
    assert not torch.isinf(reg.arc_weights.grad).any()


# ============================================================================
# 7. Boundary Conditions (N=0, N=1, N=D)
# ============================================================================

def test_adversarial_boundary_n0_empty_tensor() -> None:
    """Verify regularizer handles empty tensor N=0 gracefully with zero loss and grad."""
    D = 32
    Z = torch.empty(0, D, requires_grad=True)
    reg = RepresentationRegularizer(hidden_dim=D)

    out = reg(Z)
    assert out["loss_reg_total"].item() == 0.0
    assert out["loss_rank"].item() == 0.0
    assert out["effective_rank"] == 1.0

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert Z.grad.shape == (0, D)


def test_adversarial_boundary_n1_single_sample() -> None:
    """Verify regularizer handles single sample N=1 gracefully with zero loss and zero grad."""
    D = 32
    Z = torch.randn(1, D, requires_grad=True)
    reg = RepresentationRegularizer(hidden_dim=D)

    out = reg(Z)
    assert out["loss_reg_total"].item() == 0.0
    assert out["effective_rank"] == 1.0

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any()
    assert torch.equal(Z.grad, torch.zeros_like(Z))


def test_adversarial_boundary_n_equals_d_exact_square() -> None:
    """Verify boundary N == D handles primal vs dual transition seamlessly."""
    torch.manual_seed(901)
    N, D = 32, 32
    Z = torch.randn(N, D, requires_grad=True)
    reg = RepresentationRegularizer(hidden_dim=D)

    out = reg(Z)
    assert bool(torch.isfinite(out["loss_rank"]).item())

    out["loss_reg_total"].backward()
    assert Z.grad is not None
    assert not torch.isnan(Z.grad).any()
    assert not torch.isinf(Z.grad).any()
