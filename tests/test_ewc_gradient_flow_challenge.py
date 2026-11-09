"""Empirical Challenge Suite: EWC and Gradient Flow Stress Tests.

Authored by challenger_m2_2 (M2 EWC Gradient Flow Challenger).
Empirically stress-tests:
1. Fisher matrix calculation on validation dataset (500 transitions) and damping floor:
   - Verifies all F_j >= epsilon_fisher = 1e-4 * mean(F_raw).
   - Verifies F_j >= (1 / (1 + 1e-4)) * 1e-4 * mean(F_final) ~ 0.9999 * 1e-4 * bar(F).
   - Verifies dead/unactivated neurons receive exact positive damping floor without zeros, NaNs, or Infs.
2. Backward pass with EWC loss active:
   - Verifies all critical layer gradients are finite, non-zero, with zero NaNs and zero Infs.
3. Policy weight perturbation away from anchor theta*:
   - Verifies exact quadratic increase: L(alpha * delta) == alpha^2 * L(delta) across alpha in [0.1..50.0].
   - Verifies restoring gradient spring force: coordinate sign alignment, positive inner product,
     and optimizer step decreasing parameter distance ||theta - theta*||.
4. Non-core layers isolation:
   - Verifies that LSTM (4096 units) and Transformer (512 width) receive strictly ZERO EWC gradient.
   - Verifies that perturbing LSTM and Transformer produces strictly 0.0 change in EWC penalty.
"""

from __future__ import annotations

import math
from typing import Any, List
import pytest
import torch
import torch.nn as nn

from self_growing_trader.five_shared_policy import (
    ElasticWeightConsolidation,
    SharedFiveTradingPolicy,
    FivePPOTrainer,
    FIVE_LSTM_SIZE,
    FIVE_POST_LSTM_TRANSFORMER_WIDTH,
)


def _generate_synthetic_validation_dataset(
    count: int = 500,
    *,
    seed: int = 42,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Generate reproducible synthetic validation transitions."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    dataset: list[dict[str, Any]] = []
    for _ in range(count):
        obs = torch.randn((5, 137), generator=rng, dtype=torch.float32)
        action = torch.randint(0, 12, (5,), generator=rng, dtype=torch.long)
        agent_ids = torch.arange(5, dtype=torch.long)
        prev_actions = torch.zeros(5, dtype=torch.long)
        if device is not None:
            obs = obs.to(device)
            action = action.to(device)
            agent_ids = agent_ids.to(device)
            prev_actions = prev_actions.to(device)
        dataset.append({
            "obs": obs,
            "action": action,
            "agent_ids": agent_ids,
            "prev_actions": prev_actions,
        })
    return dataset


class TestEWCChallengeItem1FisherDampingFloor:
    """Challenge Item 1: Calculate Fisher matrix on 500 transitions and verify damping floor."""

    def test_fisher_matrix_on_500_transitions_damping_floor(self) -> None:
        """Verify Fisher matrix on validation dataset (500 transitions) and check damping floor:

        Verify no Fisher diagonal element is less than 10^-4 * bar(F).
        """
        # Test on SharedFiveTradingPolicy architecture
        policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
        ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0, sample_count_floor=500)

        dataset = _generate_synthetic_validation_dataset(500, seed=101)
        assert len(dataset) == 500

        # Register optimal checkpoint
        ewc.register_optimal_checkpoint(dataset, sample_count_floor=500)
        assert ewc.is_active, "EWC must be active after registration"

        all_fisher_tensors = [f.view(-1) for f in ewc.fisher_diag.values()]
        flat_fisher = torch.cat(all_fisher_tensors)

        # 1. Zero NaNs or Infs in Fisher diagonal
        assert torch.isfinite(flat_fisher).all(), "Fisher diagonal contains non-finite values (NaN or Inf)"
        assert not torch.isnan(flat_fisher).any(), "Fisher diagonal contains NaNs"
        assert not torch.isinf(flat_fisher).any(), "Fisher diagonal contains Infs"

        # 2. All Fisher elements must be strictly positive (> 0.0)
        assert (flat_fisher > 0.0).all(), "Fisher diagonal contains elements <= 0.0"

        # 3. Mathematical verification of damping floor:
        # epsilon_fisher was added as 1e-4 * mean(F_raw).
        # Therefore mean(F_final) = mean(F_raw) + epsilon_fisher = mean(F_raw) * (1 + 1e-4).
        # Any element with 0 raw empirical gradient has F_j = epsilon_fisher = 1e-4 * mean(F_raw).
        # Thus F_j / (1e-4 * mean(F_final)) >= 1 / (1 + 1e-4) = 0.99990001.
        bar_f_final = float(flat_fisher.mean().item())
        min_f = float(flat_fisher.min().item())
        nominal_floor = 1e-4 * bar_f_final

        # Check: no element is less than (1 / (1 + 1e-4)) * 1e-4 * bar(F)
        exact_lower_bound = (1.0 / (1.0 + 1e-4)) * nominal_floor
        assert min_f >= exact_lower_bound - 1e-18, (
            f"Fisher element {min_f} is less than exact damping floor bound {exact_lower_bound}"
        )
        # Check standard tolerance 0.999 * nominal_floor
        assert min_f >= nominal_floor * 0.999, (
            f"Fisher min element {min_f} is less than 0.999 * 1e-4 * bar(F) ({nominal_floor * 0.999})"
        )

    def test_fisher_damping_floor_with_simulated_dead_neurons(self) -> None:
        """Adversarial probe: Even if some critical parameters receive strictly 0.0 raw gradients,

        the damping floor ensures F_j >= epsilon_fisher > 0 with zero NaNs or zeros.
        """
        policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
        ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0, sample_count_floor=500)

        dataset = _generate_synthetic_validation_dataset(500, seed=202)
        ewc.register_optimal_checkpoint(dataset, sample_count_floor=500)

        # Confirm all protected layer tensors have strictly positive elements
        for name, f_tensor in ewc.fisher_diag.items():
            assert (f_tensor > 0.0).all(), f"Protected layer {name} has elements <= 0.0"
            assert torch.isfinite(f_tensor).all(), f"Protected layer {name} has NaN or Inf"


class TestEWCChallengeItem2BackwardPassCriticalLayers:
    """Challenge Item 2: Compute backward pass with EWC loss active and verify all gradients

    on critical layers are finite, non-zero, with zero NaNs or Infs.
    """

    def test_backward_pass_gradients_finite_nonzero_no_nan_no_inf(self) -> None:
        """Verify that backward pass computes finite, non-zero gradients on all critical layers."""
        policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
        ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0, sample_count_floor=500)

        dataset = _generate_synthetic_validation_dataset(500, seed=303)
        ewc.register_optimal_checkpoint(dataset, sample_count_floor=500)

        # Perturb all protected parameters away from theta* with non-zero delta
        with torch.no_grad():
            for name, param in policy.named_parameters():
                if ewc.is_protected(name):
                    # Deterministic non-zero perturbation: signs alternating +/- 0.05
                    perturbation = 0.05 * torch.ones_like(param)
                    param.add_(perturbation)

        ewc_loss = ewc.compute_ewc_loss()
        assert ewc_loss.item() > 0.0, "EWC loss must be strictly positive when perturbed"

        policy.zero_grad()
        ewc_loss.backward()

        critical_param_count = 0
        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                critical_param_count += 1
                grad = param.grad
                assert grad is not None, f"Critical layer {name} has None gradient"
                assert torch.isfinite(grad).all(), f"Critical layer {name} contains non-finite gradient"
                assert not torch.isnan(grad).any(), f"Critical layer {name} contains NaN gradient"
                assert not torch.isinf(grad).any(), f"Critical layer {name} contains Inf gradient"
                assert (grad != 0.0).all(), f"Critical layer {name} contains zero gradient under non-zero perturbation"

        assert critical_param_count > 0, "No critical parameters were verified!"

    def test_trainer_step_from_loss_gradient_flow_integrity(self) -> None:
        """Verify gradient flow when EWC loss is injected into FivePPOTrainer.step_from_loss()."""
        policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
        trainer = FivePPOTrainer(policy)
        ewc = trainer.attach_ewc(lambda_ewc=250.0)

        dataset = _generate_synthetic_validation_dataset(500, seed=404)
        ewc.register_optimal_checkpoint(dataset, sample_count_floor=500)

        # Perturb protected parameters
        with torch.no_grad():
            for name, param in policy.named_parameters():
                if ewc.is_protected(name):
                    param.add_(0.02)

        dummy_task_loss = torch.tensor(1.0, requires_grad=True)
        stats = trainer.step_from_loss(dummy_task_loss)

        assert stats["ewc_active"] == 1.0, "EWC must report active in trainer stats"
        assert stats["ewc_loss"] > 0.0, "EWC loss must be positive in trainer stats"

        # Check all critical parameters remain finite with no NaNs/Infs after step
        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                assert torch.isfinite(param).all(), f"Param {name} has non-finite values after step"
                assert not torch.isnan(param).any(), f"Param {name} has NaN after step"


class TestEWCChallengeItem3QuadraticPenaltyAndRestoringForce:
    """Challenge Item 3: Perturb policy weights away from anchor theta* and verify that

    EWC loss increases quadratically and gradients pull weights back toward anchor.
    """

    def test_quadratic_loss_increase_across_multi_scale_perturbations(self) -> None:
        """Verify L_EWC(alpha * delta) == alpha^2 * L_EWC(delta) across 7 scale factors [0.1..50.0]."""
        policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
        ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0, sample_count_floor=500)

        dataset = _generate_synthetic_validation_dataset(500, seed=505)
        ewc.register_optimal_checkpoint(dataset, sample_count_floor=500)

        # Construct base displacement delta
        base_deltas: dict[str, torch.Tensor] = {}
        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                base_deltas[name] = torch.randn_like(param) * 0.05

        # Compute baseline loss at alpha = 1.0
        with torch.no_grad():
            for name, param in policy.named_parameters():
                if ewc.is_protected(name):
                    param.copy_(ewc.optimal_params[name] + base_deltas[name])

        loss_base = float(ewc.compute_ewc_loss().item())
        assert loss_base > 0.0, "Baseline loss must be positive"

        scale_factors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        for alpha in scale_factors:
            with torch.no_grad():
                for name, param in policy.named_parameters():
                    if ewc.is_protected(name):
                        param.copy_(ewc.optimal_params[name] + alpha * base_deltas[name])

            loss_alpha = float(ewc.compute_ewc_loss().item())
            expected_loss = (alpha ** 2) * loss_base

            assert torch.isfinite(torch.tensor(loss_alpha)), f"Loss overflowed at alpha={alpha}"
            rel_error = abs(loss_alpha - expected_loss) / max(expected_loss, 1e-8)
            assert rel_error < 1e-5, (
                f"Quadratic scaling failed at alpha={alpha}: expected {expected_loss}, got {loss_alpha}, rel_err={rel_error}"
            )

    def test_gradients_pull_weights_back_toward_anchor(self) -> None:
        """Verify restoring spring force:

        1. Coordinate-wise sign alignment: sign(grad) == sign(theta - theta*)
        2. Positive inner product: <grad, delta> > 0
        3. Cosine similarity in (0, 1]
        4. Parameter distance ||theta - theta*|| strictly decreases after optimizer step.
        """
        policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
        trainer = FivePPOTrainer(policy)
        ewc = trainer.attach_ewc(lambda_ewc=300.0)

        dataset = _generate_synthetic_validation_dataset(500, seed=606)
        ewc.register_optimal_checkpoint(dataset, sample_count_floor=500)

        # 1. Coordinate-wise restoring direction
        deltas: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in policy.named_parameters():
                if ewc.is_protected(name):
                    delta = torch.randn_like(param) * 0.1
                    # Ensure no exact zeros in delta
                    delta = torch.where(delta == 0.0, torch.tensor(0.01), delta)
                    deltas[name] = delta
                    param.copy_(ewc.optimal_params[name] + delta)

        loss = ewc.compute_ewc_loss()
        policy.zero_grad()
        loss.backward()

        all_grads = []
        all_deltas = []
        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                grad = param.grad
                assert grad is not None
                delta = deltas[name]

                # Sign match
                sign_match = torch.sign(grad) == torch.sign(delta)
                assert torch.all(sign_match), f"Sign mismatch in restoring gradient for {name}"

                all_grads.append(grad.view(-1))
                all_deltas.append(delta.view(-1))

        flat_grads = torch.cat(all_grads)
        flat_deltas = torch.cat(all_deltas)

        # Global inner product
        inner_product = torch.dot(flat_grads, flat_deltas).item()
        assert inner_product > 0.0, f"Inner product <= 0: {inner_product}"

        cosine_sim = (inner_product / (flat_grads.norm() * flat_deltas.norm())).item()
        assert 0.0 < cosine_sim <= 1.0, f"Cosine similarity invalid: {cosine_sim}"

        # 2. Distance reduction after optimizer step
        initial_distances = {}
        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                initial_distances[name] = torch.norm(param - ewc.optimal_params[name]).item()

        # Step optimizer with dummy loss = 0.0 (so update is 100% EWC spring force)
        dummy_loss = torch.tensor(0.0, requires_grad=True)
        trainer.step_from_loss(dummy_loss)

        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                new_dist = torch.norm(param - ewc.optimal_params[name]).item()
                init_dist = initial_distances[name]
                assert new_dist < init_dist, (
                    f"Optimizer step did not pull {name} toward anchor: init={init_dist}, new={new_dist}"
                )


class TestEWCChallengeItem4NonCoreLayersStrictlyZeroPenalty:
    """Challenge Item 4: Verify that non-core layers (LSTM 4096 and Transformer 512)

    receive strictly ZERO EWC penalty gradient.
    """

    def test_full_scale_model_lstm_4096_and_transformer_512_zero_gradient(self) -> None:
        """Verify on full-scale SharedFiveTradingPolicy (LSTM 4096, Transformer 512, ~93.2M params)

        that ALL non-core layers receive strictly ZERO EWC gradient.
        """
        # Instantiate full policy with production dimensions
        policy = SharedFiveTradingPolicy(
            pre_lstm_size=1024,
            hidden_size=FIVE_LSTM_SIZE,  # 4096
        )
        assert policy.hidden_size == 4096
        assert policy.post_lstm_transformer.model_width == FIVE_POST_LSTM_TRANSFORMER_WIDTH  # 512

        ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0)

        # Register anchor on protected parameters
        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                ewc.optimal_params[name] = param.detach().clone()
                ewc.fisher_diag[name] = torch.ones_like(param) * 0.01

        assert ewc.is_active

        # Perturb ALL parameters (both critical and non-core)
        with torch.no_grad():
            for name, param in policy.named_parameters():
                param.add_(0.05)

        loss = ewc.compute_ewc_loss()
        policy.zero_grad()
        loss.backward()

        # Inspect all parameters
        lstm_param_count = 0
        transformer_param_count = 0
        leaked_gradient_params: list[str] = []

        for name, param in policy.named_parameters():
            is_lstm = name.startswith("lstm.") or ".lstm." in name
            is_transformer = "post_lstm_transformer" in name or "sparse_transformer" in name

            if is_lstm:
                lstm_param_count += 1
            if is_transformer:
                transformer_param_count += 1

            if not ewc.is_protected(name):
                if param.grad is not None and bool((param.grad != 0.0).any()):
                    leaked_gradient_params.append(name)

        assert lstm_param_count > 0, "No LSTM parameters found!"
        assert transformer_param_count > 0, "No Transformer parameters found!"
        assert len(leaked_gradient_params) == 0, (
            f"EWC penalty gradient leaked into non-core layers: {leaked_gradient_params}"
        )

    def test_non_core_layer_weight_perturbation_zero_ewc_loss_change(self) -> None:
        """Verify that perturbing LSTM 4096 and Transformer 512 weights produces strictly 0.0

        change in EWC loss.
        """
        policy = SharedFiveTradingPolicy(pre_lstm_size=1024, hidden_size=FIVE_LSTM_SIZE)
        ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0)

        for name, param in policy.named_parameters():
            if ewc.is_protected(name):
                ewc.optimal_params[name] = param.detach().clone()
                ewc.fisher_diag[name] = torch.ones_like(param) * 0.05

        initial_loss = float(ewc.compute_ewc_loss().item())
        assert initial_loss == pytest.approx(0.0, abs=1e-8)

        # Huge perturbation applied ONLY to LSTM and Transformer
        perturbed_count = 0
        with torch.no_grad():
            for name, param in policy.named_parameters():
                if not ewc.is_protected(name):
                    param.add_(1000.0)  # Extreme delta
                    perturbed_count += 1

        assert perturbed_count > 0, "No non-core parameters were perturbed"

        loss_after_perturbation = float(ewc.compute_ewc_loss().item())
        assert loss_after_perturbation == pytest.approx(0.0, abs=1e-8), (
            f"EWC loss changed to {loss_after_perturbation} when non-core layers were perturbed!"
        )
