"""Representation Regularization Suite for SGT V193 Policy.

Combines:
1. VICReg Variance Criterion (forces std(z_j) >= 1.0)
2. VICReg Covariance Decorrelation Criterion (forces Cov_off_diag(z) -> 0)
3. Differentiable RankMe Log-Determinant Loss (maximizes singular value entropy)
4. ArcFace Additive Angular Margin Loss with Inverse Class Frequency Weighting

Zero parameter addition during MT5 inference runtime!
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Action sets following protocol_contract.py:
# 0: HOLD
# 1: BUY_MARKET, 3: BUY_LIMIT, 5: BUY_STOP
# 2: SELL_MARKET, 4: SELL_LIMIT, 6: SELL_STOP
# 7..11: CLOSE, MANAGE (mapped to HOLD superclass 0)
BUY_ACTION_SET = frozenset({1, 3, 5})
SELL_ACTION_SET = frozenset({2, 4, 6})

# Inverse class frequencies for financial actions: ~90% HOLD, ~5% BUY, ~5% SELL
DEFAULT_CLASS_WEIGHTS: tuple[float, float, float] = (
    (1.0 / 0.90) / ((1.0 / 0.90) + (1.0 / 0.05) + (1.0 / 0.05)),
    (1.0 / 0.05) / ((1.0 / 0.90) + (1.0 / 0.05) + (1.0 / 0.05)),
    (1.0 / 0.05) / ((1.0 / 0.90) + (1.0 / 0.05) + (1.0 / 0.05)),
)


class RepresentationRegularizer(nn.Module):
    """Multi-Objective Representation Regularization Suite.

    Resolves:
    - Hidden Space Collapse (Dimensional & Point collapse, low effective rank).
    - Geometric Cluster Overlap between ENTRY (BUY/SELL) and HOLD actions.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        *,
        num_classes: int = 3,
        margin: float = 0.35,
        scale: float = 30.0,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        rank_weight: float = 0.05,
        arc_weight: float = 0.20,
        gamma_reg: float = 1e-3,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.scale = float(scale)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.rank_weight = float(rank_weight)
        self.arc_weight = float(arc_weight)
        self.gamma_reg = float(gamma_reg)
        self.eps = float(eps)

        # ArcFace class center weight vectors on hypersphere: [num_classes, hidden_dim]
        # Only used during training backprop, zero inference overhead!
        self.arc_weights = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.arc_weights)

        # Precompute constants for ArcFace angular margin
        self.cos_m = float(np.cos(self.margin))
        self.sin_m = float(np.sin(self.margin))
        self.th = float(np.cos(np.pi - self.margin))
        self.mm = float(np.sin(np.pi - self.margin) * self.margin)

        # Register normalized inverse class weights
        self.register_buffer(
            "class_weights",
            torch.tensor(DEFAULT_CLASS_WEIGHTS, dtype=torch.float32),
        )

    @staticmethod
    def derive_targets(actions: Tensor) -> Tensor:
        """Derive 3-class target labels from action indices.

        0: HOLD (or manage / close)
        1: BUY_ENTRY (1, 3, 5)
        2: SELL_ENTRY (2, 4, 6)
        """
        targets = torch.zeros_like(actions, dtype=torch.long)
        for act in (1, 3, 5):
            targets = torch.where(actions == act, torch.tensor(1, device=actions.device, dtype=torch.long), targets)
        for act in (2, 4, 6):
            targets = torch.where(actions == act, torch.tensor(2, device=actions.device, dtype=torch.long), targets)
        return targets

    def set_warmup_weights(self, epoch: int) -> None:
        """Apply warmup schedule to avoid gradient shock to pre-trained weights.

        Epoch 1: var=0.10, cov=0.10, rank=0.01, arc=0.05
        Epoch 2: var=0.50, cov=0.50, rank=0.02, arc=0.10
        Epoch 3+: var=1.00, cov=1.00, rank=0.05, arc=0.20
        """
        if epoch <= 1:
            self.var_weight = 0.10
            self.cov_weight = 0.10
            self.rank_weight = 0.01
            self.arc_weight = 0.05
        elif epoch == 2:
            self.var_weight = 0.50
            self.cov_weight = 0.50
            self.rank_weight = 0.02
            self.arc_weight = 0.10
        else:
            self.var_weight = 1.00
            self.cov_weight = 1.00
            self.rank_weight = 0.05
            self.arc_weight = 0.20

    def compute_variance_loss(self, z: Tensor) -> Tensor:
        """VICReg Variance Criterion: forces std across batch for each feature >= 1.0."""
        # z: [N, D]
        if z.shape[0] < 2:
            return z.sum() * 0.0
        std = torch.sqrt(z.var(dim=0, unbiased=True) + self.eps)
        loss_var = torch.mean(F.relu(1.0 - std))
        return loss_var

    def compute_covariance_loss(self, z_centered: Tensor) -> Tensor:
        """VICReg Covariance Criterion: forces off-diagonal elements of cov matrix to 0."""
        # z_centered: [N, D]
        if z_centered.shape[0] < 2:
            return z_centered.sum() * 0.0
        N, D = z_centered.shape
        # Sample covariance matrix
        cov = (z_centered.T @ z_centered) / max(N - 1, 1)  # [D, D]
        cov_diag = torch.diag(cov)
        cov_off_diag = cov - torch.diag_embed(cov_diag)
        loss_cov = (cov_off_diag ** 2).sum() / D
        return loss_cov

    def compute_rankme_loss(self, z: Tensor) -> Tensor:
        """Differentiable RankMe Log-Determinant Loss.

        Using Sylvester's determinant identity when N < D:
        det(gamma * I_D + (1/N) * Z^T Z) = gamma^(D - N) * det(gamma * I_N + (1/N) * Z Z^T)
        ln det(...) = (D - N) * ln(gamma) + ln det(gamma * I_N + (1/N) * Z Z^T)
        This avoids computing slogdet of a [4096, 4096] matrix on CPU while remaining
        mathematically exact.
        """
        N, D = z.shape
        if N < 1:
            return z.sum() * 0.0
        gamma = self.gamma_reg
        device = z.device
        dtype = z.dtype

        # Trace term: (1/D) * Tr((1/N) * Z^T Z) = (1 / (D * N)) * ||Z||_F^2
        trace_term = (z ** 2).sum() / (D * max(N, 1))

        if N < D:
            # Dual formulation in sample space: [N, N]
            K = (z @ z.T) / max(N, 1)  # [N, N]
            K_reg = K + gamma * torch.eye(N, device=device, dtype=dtype)
            sign, logdet_dual = torch.linalg.slogdet(K_reg)
            if (sign > 0).all() and not torch.isnan(logdet_dual).any() and not torch.isinf(logdet_dual).any():
                logdet_total = float(D - N) * np.log(gamma) + logdet_dual
                loss_rank = -logdet_total / D + trace_term
                loss_rank = torch.nan_to_num(loss_rank, nan=0.0).clamp(-100.0, 100.0)
            else:
                loss_rank = z.sum() * 0.0
        else:
            # Primal formulation: [D, D]
            C = (z.T @ z) / max(N, 1)
            C_reg = C + gamma * torch.eye(D, device=device, dtype=dtype)
            sign, logdet = torch.linalg.slogdet(C_reg)
            if (sign > 0).all() and not torch.isnan(logdet).any() and not torch.isinf(logdet).any():
                loss_rank = -logdet / D + trace_term
                loss_rank = torch.nan_to_num(loss_rank, nan=0.0).clamp(-100.0, 100.0)
            else:
                loss_rank = z.sum() * 0.0

        return torch.nan_to_num(loss_rank, nan=0.0).clamp(-100.0, 100.0)

    def compute_arcface_loss(self, z: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """ArcFace Additive Angular Margin Loss with Inverse Class Frequency Weighting.

        Forces angular margin m between ENTRY and HOLD clusters on the unit hypersphere.
        """
        # z: [N, D], targets: [N]
        if z.shape[0] < 1:
            return z.sum() * 0.0, torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # L2-normalize representations and class centers
        norm_z = F.normalize(z, p=2, dim=-1, eps=self.eps)
        norm_w = F.normalize(self.arc_weights, p=2, dim=-1, eps=self.eps)

        # Cosine similarity: [N, num_classes]
        cosine = F.linear(norm_z, norm_w).clamp(-0.99999, 0.99999)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(min=1e-6))

        # cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Monotonicity check for angles theta > pi - m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for ground truth targets
        one_hot = F.one_hot(targets, num_classes=self.num_classes).to(dtype=cosine.dtype)
        output_logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output_logits = output_logits * self.scale

        # Cross-entropy with inverse class frequency weights
        loss_arcface = F.cross_entropy(output_logits, targets, weight=self.class_weights)

        # Metric: inter-class center cosine similarity
        w_cosine = norm_w @ norm_w.T
        inter_class_cos = (w_cosine.sum() - torch.trace(w_cosine)) / max(
            self.num_classes * (self.num_classes - 1), 1
        )
        return loss_arcface, inter_class_cos

    def forward(
        self,
        hidden: Tensor,
        targets: Tensor | None = None,
        sample_mask: Tensor | None = None,
    ) -> dict[str, Tensor | float]:
        """Calculate representation regularization losses.

        Args:
            hidden: [Batch, 5, D] or [N, D] policy hidden states.
            targets: Optional [Batch, 5] or [N] class targets (0: HOLD, 1: BUY, 2: SELL).
            sample_mask: Optional boolean mask of active steps.

        Returns:
            Dictionary containing combined and individual losses, and diagnostic metrics.
        """
        if hidden.dim() == 3:
            B, A, D = hidden.shape
            z = hidden.reshape(B * A, D)
            y = targets.reshape(B * A) if targets is not None else None
            mask = sample_mask.reshape(B * A) if sample_mask is not None else None
        else:
            z = hidden
            y = targets
            mask = sample_mask

        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            if mask_bool.any():
                z = z[mask_bool]
                if y is not None:
                    y = y[mask_bool]

        N, D = z.shape
        device = z.device
        dtype = z.dtype

        if N < 2:
            zero_loss = z.sum() * 0.0
            return {
                "loss_reg_total": zero_loss,
                "loss_var": zero_loss,
                "loss_cov": zero_loss,
                "loss_rank": zero_loss,
                "loss_arcface": zero_loss,
                "effective_rank": 1.0,
                "mean_std": 0.0,
                "inter_class_cos": 0.0,
            }

        # 1. VICReg: Variance loss
        loss_var = self.compute_variance_loss(z)

        # 2. VICReg: Covariance loss
        z_centered = z - z.mean(dim=0, keepdim=True)
        loss_cov = self.compute_covariance_loss(z_centered)

        # 3. RankMe: Log-determinant loss
        loss_rank = self.compute_rankme_loss(z)

        # 4. ArcFace: Additive angular margin loss
        if y is not None:
            loss_arcface, inter_class_cos = self.compute_arcface_loss(z, y)
        else:
            loss_arcface = z.sum() * 0.0
            inter_class_cos = torch.tensor(0.0, device=device, dtype=dtype)

        # Total combined regularization loss
        loss_reg_total = (
            self.var_weight * loss_var
            + self.cov_weight * loss_cov
            + self.rank_weight * loss_rank
            + self.arc_weight * loss_arcface
        )

        # Diagnostic metrics
        with torch.no_grad():
            std_mean = float(torch.sqrt(z.var(dim=0, unbiased=True) + self.eps).mean().item())
            eff_rank = self.compute_effective_rank(z)

        return {
            "loss_reg_total": loss_reg_total,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "loss_rank": loss_rank,
            "loss_arcface": loss_arcface,
            "effective_rank": eff_rank,
            "mean_std": std_mean,
            "inter_class_cos": float(inter_class_cos.detach().cpu().item()),
        }

    @torch.no_grad()
    def compute_effective_rank(self, hidden: Tensor) -> float:
        """Measure Roy & Vetterli / RankMe Effective Rank.

        Computes r_eff = exp(-sum p_k ln p_k) where p_k are singular values
        normalized into a probability distribution.
        """
        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, self.hidden_dim)
        if hidden.shape[0] < 2:
            return 1.0
        h_centered = hidden - hidden.mean(dim=0, keepdim=True)
        try:
            S = torch.linalg.svdvals(h_centered)
            S_sum = S.sum()
            if S_sum < 1e-8:
                return 1.0
            p = S / S_sum
            p_nz = p[p > 1e-8]
            entropy = -(p_nz * torch.log(p_nz)).sum().item()
            return float(np.exp(entropy))
        except Exception:
            return 1.0
