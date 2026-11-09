"""Empirical verification harness for Milestone 2 deliverables by challenger_m2_3."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from self_growing_trader.balanced_replay_buffer import (
    BalancedMultiRegimeReservoirBuffer,
    normalize_regime,
)
from self_growing_trader.elastic_weight_consolidation import ElasticWeightConsolidation
from self_growing_trader.five_shared_policy import (
    SharedFiveTradingPolicy,
    FivePPOTrainer,
)


def test_reservoir_buffer_exact_balanced_distribution() -> None:
    """1. Reservoir Buffer balance test: check mini-batch sampling across regimes achieves ~33.3% balanced distribution."""
    buf = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    for i in range(200):
        buf.add({"step_id": i, "obs": np.zeros((5, 137), dtype=np.float32), "action": 0}, regime="UPTREND", realized_r=1.0)
        buf.add({"step_id": 1000 + i, "obs": np.zeros((5, 137), dtype=np.float32), "action": 1}, regime="DOWNTREND", realized_r=-1.0)
        buf.add({"step_id": 2000 + i, "obs": np.zeros((5, 137), dtype=np.float32), "action": 2}, regime="SIDEWAY", realized_r=0.0)

    assert len(buf.uptrend_pool) == 128
    assert len(buf.downtrend_pool) == 128
    assert len(buf.sideway_pool) == 128

    counts = {"UPTREND": 0, "DOWNTREND": 0, "SIDEWAY": 0}
    total_sampled = 0

    for _ in range(100):
        batch, is_weights = buf.sample_balanced_reservoir_batch(batch_size=96)
        assert len(batch) == 96
        assert is_weights.shape[0] == 96
        assert torch.all(is_weights >= 0.0) and torch.all(is_weights <= 1.0)
        assert abs(float(is_weights.max().item()) - 1.0) < 1e-5

        batch_counts = {"UPTREND": 0, "DOWNTREND": 0, "SIDEWAY": 0}
        for item in batch:
            batch_counts[item["regime"]] += 1
            counts[item["regime"]] += 1
            total_sampled += 1

        assert batch_counts["UPTREND"] == 32
        assert batch_counts["DOWNTREND"] == 32
        assert batch_counts["SIDEWAY"] == 32

    p_up = counts["UPTREND"] / total_sampled
    p_down = counts["DOWNTREND"] / total_sampled
    p_side = counts["SIDEWAY"] / total_sampled

    assert abs(p_up - 1/3) < 1e-4
    assert abs(p_down - 1/3) < 1e-4
    assert abs(p_side - 1/3) < 1e-4


def test_reservoir_skewed_streaming_resilience() -> None:
    """Verify that buffer retains equal capacity and yields 33.3% even after heavily skewed streaming."""
    buf = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    for i in range(8000):
        buf.add({"step_id": i, "obs": np.zeros((5, 137), dtype=np.float32)}, regime="DOWNTREND", realized_r=-1.0)
    for i in range(500):
        buf.add({"step_id": i, "obs": np.zeros((5, 137), dtype=np.float32)}, regime="UPTREND", realized_r=1.0)
    for i in range(500):
        buf.add({"step_id": i, "obs": np.zeros((5, 137), dtype=np.float32)}, regime="SIDEWAY", realized_r=0.0)

    assert len(buf.uptrend_pool) == 128
    assert len(buf.downtrend_pool) == 128
    assert len(buf.sideway_pool) == 128

    batch, is_weights = buf.sample_balanced_reservoir_batch(batch_size=96)
    counts = {"UPTREND": 0, "DOWNTREND": 0, "SIDEWAY": 0}
    for it in batch:
        counts[it["regime"]] += 1

    assert counts["UPTREND"] == 32
    assert counts["DOWNTREND"] == 32
    assert counts["SIDEWAY"] == 32


def test_ewc_damping_floor_enforcement() -> None:
    """2. EWC gradient flow & damping floor test: verify Fisher damping floor epsilon_fisher = 10^-4 * bar(F) is enforced."""
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0, sample_count_floor=500)

    val_data = []
    for _ in range(500):
        val_data.append({
            "obs": torch.randn(5, 137),
            "action": torch.randint(0, 12, (5,)),
            "agent_ids": torch.arange(5),
            "prev_actions": torch.zeros(5, dtype=torch.long),
        })

    ewc.register_optimal_checkpoint(val_data)
    assert ewc.is_active

    all_f = torch.cat([f.view(-1) for f in ewc.fisher_diag.values()])
    mean_f = float(all_f.mean().item())
    min_f = float(all_f.min().item())

    assert torch.isfinite(all_f).all()
    assert not torch.isnan(all_f).any()
    assert not torch.isinf(all_f).any()
    assert min_f > 0.0

    # Theoretical floor check: min_f >= 1e-4 * mean_f * 0.999
    expected_floor = 1e-4 * mean_f * 0.999
    assert min_f >= expected_floor


def test_ewc_gradient_flow_cleanliness_and_restoring_force() -> None:
    """Verify gradients propagate cleanly with zero NaNs/Infs and exact analytical restoring force."""
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0, sample_count_floor=500)

    val_data = []
    for _ in range(500):
        val_data.append({
            "obs": torch.randn(5, 137),
            "action": torch.randint(0, 12, (5,)),
            "agent_ids": torch.arange(5),
            "prev_actions": torch.zeros(5, dtype=torch.long),
        })

    ewc.register_optimal_checkpoint(val_data)

    perturbation_delta = 0.1
    for name, p in policy.named_parameters():
        if ewc.is_protected(name):
            with torch.no_grad():
                p.add_(perturbation_delta)

    policy.zero_grad()
    ewc_loss = ewc.compute_ewc_loss()
    assert torch.isfinite(ewc_loss) and not torch.isnan(ewc_loss) and not torch.isinf(ewc_loss)
    ewc_loss.backward()

    max_analytical_err = 0.0
    for name, p in policy.named_parameters():
        if ewc.is_protected(name):
            assert p.grad is not None
            assert torch.isfinite(p.grad).all(), f"{name} grad is not finite!"
            assert not torch.isnan(p.grad).any(), f"{name} grad contains NaN!"
            assert not torch.isinf(p.grad).any(), f"{name} grad contains Inf!"

            # Analytical restoring force: nabla = lambda * F * (theta - theta*)
            F_j = ewc.fisher_diag[name]
            expected_grad = 250.0 * F_j * perturbation_delta
            diff = torch.max(torch.abs(p.grad - expected_grad)).item()
            if diff > max_analytical_err:
                max_analytical_err = diff
        else:
            if p.grad is not None:
                assert torch.all(p.grad == 0.0), f"Non-protected param {name} received non-zero grad!"

    assert max_analytical_err < 1e-4


def test_trainer_step_from_loss_integration() -> None:
    """Verify FivePPOTrainer.step_from_loss() integrates EWC loss seamlessly."""
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    trainer = FivePPOTrainer(policy)
    ewc = trainer.attach_ewc(lambda_ewc=250.0)

    val_data = []
    for _ in range(500):
        val_data.append({
            "obs": torch.randn(5, 137),
            "action": torch.randint(0, 12, (5,)),
            "agent_ids": torch.arange(5),
            "prev_actions": torch.zeros(5, dtype=torch.long),
        })
    ewc.register_optimal_checkpoint(val_data)

    # Perturb weights
    with torch.no_grad():
        policy.heads.action_query.weight.add_(0.05)

    dummy_loss = torch.tensor(2.0, requires_grad=True)
    stats = trainer.step_from_loss(dummy_loss)

    assert stats["ewc_active"] == 1.0
    assert stats["ewc_loss"] > 0.0
    assert not math.isnan(stats["ewc_loss"])
    assert not math.isinf(stats["ewc_loss"])
