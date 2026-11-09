"""Progressive Knowledge Distillation for SGT-V193-Compact.

Transfers representation and policy capabilities from the 93.2M Teacher (V193)
to the 17.5M Student (SGT-V193-Compact) without training from scratch.

Mathematical formulations:
1. Representation Distillation:
   L_rep = (1 / B) * sum || W_proj @ h_S - h_T ||_2^2
   where W_proj in R^{4096 x 768} is an orthogonal projection adapter.

2. Logit Distillation:
   L_logit = tau^2 * D_KL( Softmax(z_T / tau) || Softmax(z_S / tau) )
   with temperature tau = 2.0.

3. Total Distillation Objective:
   L_total = lambda_rep * L_rep + lambda_logit * L_logit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DistillationMetrics:
    total_loss: float
    rep_loss: float
    logit_loss: float
    action_match_rate: float
    student_effective_rank: float


class ProgressiveDistillationAdapter(nn.Module):
    """Linear projection layer mapping Student 768D representations to Teacher 4096D."""

    def __init__(self, student_dim: int = 768, teacher_dim: int = 4096) -> None:
        super().__init__()
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.proj = nn.Linear(self.student_dim, self.teacher_dim, bias=False)
        # Initialize with semi-orthogonal projection
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, h_student: Tensor) -> Tensor:
        """Projects student latent vectors to teacher dimension."""
        return self.proj(h_student)


class ProgressiveDistillationLoss(nn.Module):
    """Joint representation and logit distillation loss module."""

    def __init__(
        self,
        student_dim: int = 768,
        teacher_dim: int = 4096,
        temperature: float = 2.0,
        lambda_rep: float = 1.0,
        lambda_logit: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_rep = float(lambda_rep)
        self.lambda_logit = float(lambda_logit)
        self.adapter = ProgressiveDistillationAdapter(student_dim, teacher_dim)

    def representation_loss(self, h_student: Tensor, h_teacher: Tensor) -> Tensor:
        """L2 loss between projected student hidden state and teacher hidden state."""
        h_proj = self.adapter(h_student)  # [B, 4096]
        return F.mse_loss(h_proj, h_teacher)

    def logit_loss(self, logits_student: Tensor, logits_teacher: Tensor) -> Tensor:
        """KL divergence loss scaled by tau^2."""
        p_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)
        log_p_student = F.log_softmax(logits_student / self.temperature, dim=-1)
        kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        return (self.temperature ** 2) * kl

    def compute_action_match_rate(self, logits_student: Tensor, logits_teacher: Tensor) -> float:
        """Fraction of actions where student's argmax matches teacher's argmax."""
        a_s = torch.argmax(logits_student, dim=-1)
        a_t = torch.argmax(logits_teacher, dim=-1)
        return float((a_s == a_t).float().mean().item())

    def compute_effective_rank(self, h_student: Tensor, eps: float = 1e-7) -> float:
        """Roy & Vetterli (2007) entropy-based effective rank."""
        if h_student.shape[0] < 2:
            return 1.0
        h_centered = h_student - h_student.mean(dim=0, keepdim=True)
        # SVD on centered representations
        try:
            _, s, _ = torch.linalg.svd(h_centered, full_matrices=False)
            p = s / (torch.sum(s) + eps)
            p = p[p > eps]
            entropy = -torch.sum(p * torch.log(p))
            return float(torch.exp(entropy).item())
        except Exception:
            return 1.0

    def forward(
        self,
        h_student: Tensor,
        h_teacher: Tensor,
        logits_student: Tensor,
        logits_teacher: Tensor,
    ) -> Tuple[Tensor, DistillationMetrics]:
        """Compute total loss and diagnostic metrics."""
        rep_l = self.representation_loss(h_student, h_teacher)
        logit_l = self.logit_loss(logits_student, logits_teacher)
        total_l = self.lambda_rep * rep_l + self.lambda_logit * logit_l

        match_rate = self.compute_action_match_rate(logits_student, logits_teacher)
        eff_rank = self.compute_effective_rank(h_student)

        metrics = DistillationMetrics(
            total_loss=float(total_l.item()),
            rep_loss=float(rep_l.item()),
            logit_loss=float(logit_l.item()),
            action_match_rate=match_rate,
            student_effective_rank=eff_rank,
        )
        return total_l, metrics
