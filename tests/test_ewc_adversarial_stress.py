"""Adversarial stress testing suite for Elastic Weight Consolidation (EWC).

Conducts empirical verification on:
1. Damping floor behavior with zero empirical gradients (all grads = 0).
   - Verifies F_j > 0, epsilon_fisher is active, and no NaN/zero division occurs.
   - Tests mathematical zero gradients (flat loss surface).
   - Documents behavior when loss graph has requires_grad=False (fragility finding).
2. Large perturbation quadratic scaling and numerical stability.
   - Moves weights by up to 10.0 standard deviations (and scale factors up to 50x).
   - Verifies exact quadratic scaling L(alpha * delta) == alpha^2 * L(delta).
   - Verifies absence of NaN and Inf under extreme perturbations.
3. Restoring gradient force direction and sign coherence.
   - Verifies nabla_{theta_j} L_EWC strictly opposes perturbation direction.
   - Verifies coordinate-wise sign alignment and positive inner product with displacement.
4. Excluded layers invariance.
   - Verifies modifying LSTM core or Transformer produces strictly 0.0 change in EWC penalty.
   - Verifies backward pass generates zero/None gradients for excluded layers.
5. Promotion sidecar round-trip fidelity.
   - Verifies serialization and reload into fresh instances preserves parameters and Fisher values.
   - Verifies AutoRLEngine promotion artifact creation and numerical equivalence.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from self_growing_trader.five_shared_policy import (
    ElasticWeightConsolidation,
    SharedFiveTradingPolicy,
    FivePPOTrainer,
)
from self_growing_trader.auto_rl_engine import AutoRLEngine, AutoRLEngineConfig


class ZeroGradDecisionModel(nn.Module):
    """Model where backward graph exists (requires_grad=True) but all gradients evaluate strictly to 0.0."""

    def __init__(self) -> None:
        super().__init__()
        self.market_conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.market_fc = nn.Linear(4, 8)
        self.pre_lstm = nn.Linear(8, 16)
        self.action_query = nn.Linear(16, 12)
        self.action_embedding = nn.Embedding(12, 12)
        self.value_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Connect to parameters through autograd graph but scale by 0.0 so gradients are strictly 0.0
        c = self.market_conv(x.unsqueeze(1)) * 0.0
        m = self.market_fc(c.mean(dim=-1)) * 0.0
        p = self.pre_lstm(m) * 0.0
        logits = self.action_query(p) * 0.0
        # Add dummy embedding lookup to connect action_embedding
        dummy_idx = torch.zeros(1, dtype=torch.long, device=x.device)
        dummy_emb = self.action_embedding(dummy_idx).sum() * 0.0
        return logits + dummy_emb


class DetachedOutputModel(nn.Module):
    """Model where output is detached from parameter graph (requires_grad=False)."""

    def __init__(self) -> None:
        super().__init__()
        self.action_query = nn.Linear(16, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return constant detached tensor
        return torch.zeros(x.shape[0], 12)


class ToyAdversarialModel(nn.Module):
    """Toy architecture containing both protected layers and explicitly excluded layers."""

    def __init__(self) -> None:
        super().__init__()
        # Protected layers
        self.market_conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.market_fc = nn.Linear(4, 8)
        self.pre_lstm = nn.Linear(8, 16)
        self.action_query = nn.Linear(16, 12)
        self.action_embedding = nn.Embedding(12, 12)
        self.value_head = nn.Linear(16, 1)
        # Excluded layers (must never be protected or penalized)
        self.lstm = nn.LSTM(16, 16, batch_first=True)
        self.post_lstm_transformer = nn.Linear(16, 16)
        self.sparse_transformer = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.tanh(self.market_conv(x.unsqueeze(1)))
        m = torch.tanh(self.market_fc(c.mean(dim=-1)))
        p = torch.tanh(self.pre_lstm(m))
        l_out, _ = self.lstm(p.unsqueeze(1))
        t_out = torch.tanh(self.post_lstm_transformer(l_out.squeeze(1)))
        return self.action_query(t_out)


def test_adversarial_zero_empirical_gradients_damping_floor() -> None:
    """Adversarial Test 1A: Verify damping floor when empirical gradients are completely zero.

    Requirement:
    - F_j does not become zero or cause NaN.
    - epsilon_fisher = 1e-4 * max(mean_F, 1e-8) = 1e-12 prevents unconstrained drift.
    - compute_ewc_loss() and backward() produce valid finite numbers, strictly no NaN or Inf.
    """
    model = ZeroGradDecisionModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [
        {"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))}
        for _ in range(25)
    ]

    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    assert ewc.is_active, "EWC must be active after registration"

    # Verify that all empirical gradients were 0 before floor, and now equal exact damping floor
    for name, f_tensor in ewc.fisher_diag.items():
        assert torch.isfinite(f_tensor).all(), f"NaN or Inf in Fisher diagonal for {name}"
        assert (f_tensor > 0.0).all(), f"Fisher diagonal contains zero for {name}"
        # Since empirical gradients were 0, flat_fisher mean is 0.0, so epsilon_fisher = 1e-4 * 1e-8 = 1e-12
        assert torch.allclose(f_tensor, torch.tensor(1e-12), rtol=1e-5, atol=1e-15), (
            f"Fisher diagonal for {name} ({f_tensor.mean().item()}) does not match expected damping floor 1e-12"
        )

    # Verify zero penalty at anchor
    initial_loss = ewc.compute_ewc_loss()
    assert torch.isfinite(initial_loss), "Initial loss must be finite"
    assert math.isclose(float(initial_loss.item()), 0.0, abs_tol=1e-8)

    # Perturb weights and check loss and gradient under pure damping floor
    with torch.no_grad():
        model.action_query.weight.add_(0.5)

    loss = ewc.compute_ewc_loss()
    assert torch.isfinite(loss), "Loss must be finite after perturbation"
    assert loss.item() > 0.0, "Loss must be strictly positive under non-zero perturbation"

    model.zero_grad()
    loss.backward()

    grad = model.action_query.weight.grad
    assert grad is not None, "Gradient must be computed"
    assert torch.isfinite(grad).all(), "Gradient must be finite"
    assert (grad > 0.0).all(), "Gradient must be strictly positive to oppose positive perturbation"


def test_adversarial_zero_empirical_gradients_detached_graph_fragility() -> None:
    """Adversarial Test 1B (Fragility Investigation): Test behavior when loss graph has requires_grad=False.

    Finding:
    - If a model returns detached outputs or if protected parameters do not contribute
      to the loss computation graph, PyTorch raises:
      RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    - This test documents and verifies this behavior as an empirical finding.
    """
    model = DetachedOutputModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [
        {"obs": torch.randn(4, 16), "action": torch.randint(0, 12, (4,))}
        for _ in range(5)
    ]

    with pytest.raises(RuntimeError, match="does not require grad"):
        ewc.register_optimal_checkpoint(dataset, sample_count_floor=5)


def test_adversarial_large_perturbation_quadratic_scaling() -> None:
    """Adversarial Test 2: Verify quadratic scaling L(alpha * delta) == alpha^2 * L(delta).

    Requirement:
    - Scales quadratically across multiple orders of magnitude (alpha from 0.1 to 50.0).
    - No numerical overflow under extreme perturbations (up to 10 standard deviations).
    """
    model = ToyAdversarialModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [
        {"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))}
        for _ in range(20)
    ]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    # Compute parameter standard deviation for protected weights
    base_deltas: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in ewc.optimal_params:
            std = float(param.data.std().item()) if param.data.numel() > 1 else 0.1
            # Base delta is 1 standard deviation
            base_deltas[name] = torch.randn_like(param) * (std if std > 1e-4 else 0.1)

    # Measure loss at alpha = 1.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ewc.optimal_params:
                param.copy_(ewc.optimal_params[name] + base_deltas[name])

    loss_1x = float(ewc.compute_ewc_loss().item())
    assert loss_1x > 0.0, "Base loss must be positive"

    # Test scaling factors alpha in [0.1, 0.5, 2.0, 5.0, 10.0 (10 sigma!), 50.0]
    scaling_factors = [0.1, 0.5, 2.0, 5.0, 10.0, 50.0]
    for alpha in scaling_factors:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ewc.optimal_params:
                    param.copy_(ewc.optimal_params[name] + alpha * base_deltas[name])

        loss_alpha = float(ewc.compute_ewc_loss().item())
        expected_loss = (alpha ** 2) * loss_1x

        assert torch.isfinite(torch.tensor(loss_alpha)), f"Loss overflowed at alpha={alpha}"
        # Relative error should be < 1e-4
        rel_error = abs(loss_alpha - expected_loss) / max(expected_loss, 1e-8)
        assert rel_error < 1e-4, f"Quadratic scaling failed at alpha={alpha}: rel_error={rel_error}"

        # Backward test at 10-sigma perturbation
        if alpha == 10.0:
            model.zero_grad()
            loss_tensor = ewc.compute_ewc_loss()
            loss_tensor.backward()
            for name, param in model.named_parameters():
                if name in ewc.optimal_params:
                    assert param.grad is not None
                    assert torch.isfinite(param.grad).all(), f"Gradient overflowed at 10-sigma for {name}"


def test_adversarial_restoring_force_direction_and_sign_coherence() -> None:
    """Adversarial Test 3: Verify restoring gradient consistently opposes perturbation direction.

    Requirement:
    - Every coordinate j satisfies sign(nabla_{theta_j} L_EWC) == sign(theta_j - theta_j^*).
    - Inner product <nabla L_EWC, Delta> is strictly positive.
    - Cosine similarity between nabla L_EWC and Delta is in (0, 1].
    - Works for both purely positive, purely negative, and mixed random perturbations.
    """
    model = ToyAdversarialModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [
        {"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))}
        for _ in range(20)
    ]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    perturbation_scenarios = [
        ("positive", lambda shape: torch.abs(torch.randn(shape)) + 0.05),
        ("negative", lambda shape: -(torch.abs(torch.randn(shape)) + 0.05)),
        ("mixed", lambda shape: torch.randn(shape) + torch.sign(torch.randn(shape)) * 0.02),
    ]

    for scenario_name, perturb_fn in perturbation_scenarios:
        deltas: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ewc.optimal_params:
                    d = perturb_fn(param.shape)
                    deltas[name] = d
                    param.copy_(ewc.optimal_params[name] + d)

        model.zero_grad()
        loss = ewc.compute_ewc_loss()
        loss.backward()

        all_grads = []
        all_deltas = []

        for name, param in model.named_parameters():
            if name in ewc.optimal_params:
                grad = param.grad
                assert grad is not None, f"Grad is None for {name} in {scenario_name}"
                delta = deltas[name]

                # Coordinate-wise sign coherence: grad and delta must have matching signs
                sign_match = torch.sign(grad) == torch.sign(delta)
                # Where delta != 0, signs MUST strictly match
                non_zero_delta = delta != 0.0
                assert torch.all(sign_match[non_zero_delta]), (
                    f"Sign mismatch in restoring gradient for {name} in {scenario_name}"
                )

                all_grads.append(grad.view(-1))
                all_deltas.append(delta.view(-1))

        flat_grads = torch.cat(all_grads)
        flat_deltas = torch.cat(all_deltas)

        # Global restoring spring force: <grad, delta> > 0
        inner_product = torch.dot(flat_grads, flat_deltas).item()
        assert inner_product > 0.0, f"Inner product <= 0 in {scenario_name}: {inner_product}"

        cosine_sim = (inner_product / (flat_grads.norm() * flat_deltas.norm())).item()
        assert 0.0 < cosine_sim <= 1.0, f"Cosine similarity invalid in {scenario_name}: {cosine_sim}"


def test_adversarial_excluded_layers_zero_penalty_invariant() -> None:
    """Adversarial Test 4: Verify un-protected layers (LSTM, Transformer) produce strictly 0.0 penalty.

    Requirement:
    - Modifying LSTM weights by huge deltas produces EXACTLY 0.0 change in EWC loss.
    - Modifying Transformer weights produces EXACTLY 0.0 change in EWC loss.
    - Backward pass leaves LSTM and Transformer grads as None or zero.
    - Tested on both ToyAdversarialModel and full SharedFiveTradingPolicy.
    """
    # Part A: ToyAdversarialModel
    model = ToyAdversarialModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [
        {"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))}
        for _ in range(20)
    ]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    initial_loss = float(ewc.compute_ewc_loss().item())
    assert initial_loss == pytest.approx(0.0, abs=1e-8)

    # Excluded layers in ToyAdversarialModel are:
    # lstm.weight_*, lstm.bias_*, post_lstm_transformer.*, sparse_transformer.*
    # Note: pre_lstm is NOT excluded (it is in DEFAULT_PROTECTED_PATTERNS)
    excluded_params_perturbed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.startswith("lstm.") or ".lstm." in name or "transformer" in name:
                param.add_(100.0)  # Huge perturbation!
                excluded_params_perturbed += 1

    assert excluded_params_perturbed > 0, "No excluded parameters were perturbed!"

    loss_after_excluded_pert = float(ewc.compute_ewc_loss().item())
    assert loss_after_excluded_pert == pytest.approx(0.0, abs=1e-8), (
        f"EWC loss changed to {loss_after_excluded_pert} when modifying excluded layers!"
    )

    model.zero_grad()
    ewc.compute_ewc_loss().backward()
    for name, param in model.named_parameters():
        if name.startswith("lstm.") or ".lstm." in name or "transformer" in name:
            assert param.grad is None or torch.all(param.grad == 0.0), (
                f"Gradient leaked into excluded layer {name}!"
            )

    # Part B: Full SharedFiveTradingPolicy
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    policy_ewc = ElasticWeightConsolidation(policy, lambda_ewc=250.0)

    # Populate anchors and fishers for protected parameters
    for name, p in policy.named_parameters():
        if policy_ewc.is_protected(name):
            policy_ewc.optimal_params[name] = p.detach().clone()
            policy_ewc.fisher_diag[name] = torch.ones_like(p) * 0.01

    base_policy_loss = float(policy_ewc.compute_ewc_loss().item())
    assert base_policy_loss == pytest.approx(0.0, abs=1e-8)

    # Perturb all recurrent LSTM and Transformer parameters in full policy
    lstm_params_found = 0
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name.startswith("lstm.") or ".lstm." in name or "post_lstm_transformer" in name:
                p.add_(50.0)
                lstm_params_found += 1

    assert lstm_params_found > 0, "No LSTM/Transformer parameters found in SharedFiveTradingPolicy"

    policy_loss_after_lstm = float(policy_ewc.compute_ewc_loss().item())
    assert policy_loss_after_lstm == pytest.approx(0.0, abs=1e-8), (
        f"SharedFiveTradingPolicy EWC loss changed to {policy_loss_after_lstm} when altering LSTM!"
    )


def test_adversarial_promotion_sidecar_roundtrip_fidelity(tmp_path: Path) -> None:
    """Adversarial Test 5: Verify promotion sidecar serialization and bit-for-bit reload.

    Requirement:
    - Sidecar ewc_anchor_round_X.pt can be reloaded into fresh EWC instance.
    - Optimal weights, Fisher diagonals, lambda, and sample floor match 100%.
    - EWC loss under identical perturbations matches bit-for-bit down to machine precision.
    - Verifies AutoRLEngine promotion artifact generation on real SharedFiveTradingPolicy.
    """
    cfg = AutoRLEngineConfig(
        runs_dir=tmp_path / "adversarial_runs",
        ewc_enabled=True,
        ewc_lambda=350.0,
        ewc_sample_floor=15,
    )
    engine = AutoRLEngine(cfg)

    # Create dummy candidate checkpoint
    cand_path = tmp_path / "candidate_m2_test.pkl"
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    torch.save(policy, cand_path)

    # Trigger promotion hook
    engine._update_ewc_on_promotion(round_num=3, candidate_ckpt_path=cand_path)

    sidecar_path = engine.runs_dir / "ewc_anchor_round_3.pt"
    assert sidecar_path.exists(), f"Sidecar {sidecar_path} does not exist"

    # Reload into completely new EWC instance with clean policy loaded from same checkpoint
    fresh_policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    fresh_policy.load_state_dict(policy.state_dict())

    fresh_ewc = ElasticWeightConsolidation(fresh_policy)
    fresh_ewc.load(sidecar_path)

    assert fresh_ewc.is_active, "Reloaded EWC must be active"
    assert fresh_ewc.lambda_ewc == 350.0, f"Expected lambda=350.0, got {fresh_ewc.lambda_ewc}"
    assert fresh_ewc.sample_count_floor == 15

    # Check every parameter and Fisher diagonal
    original_ewc = engine.ewc
    assert original_ewc is not None

    assert set(original_ewc.optimal_params.keys()) == set(fresh_ewc.optimal_params.keys())
    assert set(original_ewc.fisher_diag.keys()) == set(fresh_ewc.fisher_diag.keys())

    for k in original_ewc.optimal_params:
        assert torch.equal(original_ewc.optimal_params[k], fresh_ewc.optimal_params[k]), (
            f"optimal_params mismatch for {k}"
        )
        assert torch.equal(original_ewc.fisher_diag[k], fresh_ewc.fisher_diag[k]), (
            f"fisher_diag mismatch for {k}"
        )

    # At identical anchor weights, both losses must be 0.0
    assert float(original_ewc.compute_ewc_loss().item()) == pytest.approx(0.0, abs=1e-7)
    assert float(fresh_ewc.compute_ewc_loss().item()) == pytest.approx(0.0, abs=1e-7)

    # Perturb the engine's model and fresh_policy identically, compare loss
    engine_model = original_ewc.model
    with torch.no_grad():
        for (name1, p1), (name2, p2) in zip(
            engine_model.named_parameters(), fresh_policy.named_parameters()
        ):
            if name1 in original_ewc.optimal_params:
                noise = torch.randn_like(p1) * 0.1
                p1.add_(noise)
                p2.add_(noise)

    orig_loss = original_ewc.compute_ewc_loss()
    fresh_loss = fresh_ewc.compute_ewc_loss()

    assert orig_loss.item() > 0.0, "Perturbed loss must be non-zero"
    assert torch.allclose(orig_loss, fresh_loss, rtol=1e-7, atol=1e-7), (
        f"Loss mismatch between original ({orig_loss.item()}) and reloaded ({fresh_loss.item()})"
    )


def test_adversarial_trainer_restoring_force_optimization() -> None:
    """Adversarial Test 6: Verify FivePPOTrainer optimizer actually steps weights toward anchor.

    Requirement:
    - Perturb protected parameter away from optimal anchor.
    - Run PPO step_from_loss() with dummy loss = 0.0 so gradient is purely from EWC.
    - Verify that after optimizer step, the parameter distance ||theta - theta*|| has DECREASED.
    """
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    trainer = FivePPOTrainer(policy)
    ewc = trainer.attach_ewc(lambda_ewc=500.0)

    # Register anchor
    for name, p in policy.named_parameters():
        if ewc.is_protected(name):
            ewc.optimal_params[name] = p.detach().clone()
            ewc.fisher_diag[name] = torch.ones_like(p) * 1.0

    target_param = policy.heads.action_query.weight
    optimal_target = ewc.optimal_params["heads.action_query.weight"]

    # Perturb parameter away by +0.2
    with torch.no_grad():
        target_param.add_(0.2)

    initial_dist = torch.norm(target_param - optimal_target).item()

    # Step with zero task loss so gradient is 100% EWC spring force
    dummy_loss = torch.tensor(0.0, requires_grad=True)
    trainer.step_from_loss(dummy_loss)

    new_dist = torch.norm(target_param - optimal_target).item()
    assert new_dist < initial_dist, (
        f"Optimizer step did not pull weights toward anchor! initial={initial_dist}, new={new_dist}"
    )
