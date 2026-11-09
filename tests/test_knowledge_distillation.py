"""Unit tests for Progressive Knowledge Distillation.

Verifies:
1. Representation loss computation between Student [B, 768] and Teacher [B, 4096].
2. Logit distillation loss with temperature scaling tau=2.0.
3. Action match rate calculation.
4. Effective rank calculation on student representations.
5. Backward gradient flow without NaN / Inf.
"""

import pytest
import torch
import torch.nn.functional as F

from src.self_growing_trader.knowledge_distillation import (
    DistillationMetrics,
    ProgressiveDistillationAdapter,
    ProgressiveDistillationLoss,
)


def test_distillation_adapter_shape():
    adapter = ProgressiveDistillationAdapter(student_dim=768, teacher_dim=4096)
    h_s = torch.randn(10, 768)
    h_proj = adapter(h_s)
    assert h_proj.shape == (10, 4096)


def test_distillation_loss_and_metrics():
    loss_fn = ProgressiveDistillationLoss(
        student_dim=768,
        teacher_dim=4096,
        temperature=2.0,
        lambda_rep=1.0,
        lambda_logit=0.5,
    )

    B = 16
    h_student = torch.randn(B, 768, requires_grad=True)
    h_teacher = torch.randn(B, 4096)
    logits_student = torch.randn(B, 12, requires_grad=True)
    logits_teacher = torch.randn(B, 12)

    total_loss, metrics = loss_fn(h_student, h_teacher, logits_student, logits_teacher)

    assert total_loss > 0.0
    assert not torch.isnan(total_loss)
    assert isinstance(metrics, DistillationMetrics)
    assert 0.0 <= metrics.action_match_rate <= 1.0
    assert metrics.student_effective_rank >= 1.0

    # Test backward pass
    total_loss.backward()
    assert h_student.grad is not None
    assert logits_student.grad is not None
    assert not torch.isnan(h_student.grad).any()
    assert not torch.isnan(logits_student.grad).any()


def test_action_match_rate_exact():
    loss_fn = ProgressiveDistillationLoss()
    logits_t = torch.tensor([[10.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    logits_s = torch.tensor([[8.0, 2.0, 0.0], [0.0, 4.0, 2.0]])
    rate = loss_fn.compute_action_match_rate(logits_s, logits_t)
    assert rate == 1.0

    logits_s_mismatch = torch.tensor([[0.0, 10.0, 0.0], [0.0, 4.0, 2.0]])
    rate_mismatch = loss_fn.compute_action_match_rate(logits_s_mismatch, logits_t)
    assert rate_mismatch == 0.5


def test_effective_rank_calculation():
    loss_fn = ProgressiveDistillationLoss()
    # Rank 1 representation
    h_rank1 = torch.ones(32, 768)
    r1 = loss_fn.compute_effective_rank(h_rank1)
    assert r1 == 1.0

    # Diverse representation
    h_diverse = torch.randn(100, 768)
    r_div = loss_fn.compute_effective_rank(h_diverse)
    assert r_div > 50.0, f"Expected r_div > 50, got {r_div}"
