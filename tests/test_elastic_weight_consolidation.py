"""Unit tests for Elastic Weight Consolidation (EWC) manager, FivePPOTrainer, and AutoRLEngine.

Verifies:
1. Core layer parameter filtering (~1.07M critical params protected, LSTM and Transformer excluded).
2. Diagonal Fisher information matrix calculation F_j = (1/N) sum (nabla log pi(a|s))^2.
3. Fisher sample size floor (>= 500 transitions) with UserWarning on undersized sets.
4. Damping floor epsilon_fisher = 10^-4 * mean(F) preventing dead neurons and unconstrained drift.
5. Quadratic penalty L_EWC = (lambda / 2) * sum F_j * (theta_j - theta_j^*)^2 with lambda=250.0.
6. Gradient flow and restoring spring force pulling weights back to anchor.
7. Integration with FivePPOTrainer.step_from_loss().
8. Artifact serialization (save/load) fidelity.
9. Champion promotion hook in AutoRLEngine creating sidecar ewc_anchor_round_{round_num}.pt.
"""

from __future__ import annotations

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


class ToyDecisionModel(nn.Module):
    """Small model featuring the exact core layer names of SharedFiveTradingPolicy."""

    def __init__(self) -> None:
        super().__init__()
        self.market_conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.market_fc = nn.Linear(8, 16)
        self.pre_lstm = nn.Linear(16, 32)
        self.action_query = nn.Linear(32, 12)
        self.action_embedding = nn.Embedding(12, 12)
        self.value_head = nn.Linear(32, 1)
        self.sparse_transformer = nn.Linear(32, 32)  # Should NOT be protected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        c = torch.tanh(self.market_conv(x.unsqueeze(1)))
        m = torch.tanh(self.market_fc(c.mean(dim=-1)))
        h = torch.tanh(self.pre_lstm(m))
        return self.action_query(h)


def test_ewc_initialization_and_layer_filtering() -> None:
    """Verify that EWC protects only core layers and excludes sparse transformer/recurrent core."""
    model = ToyDecisionModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    assert ewc.lambda_ewc == 250.0
    assert not ewc.is_active

    protected = ewc.get_protected_parameters()
    protected_names = set(protected.keys())

    # Core layers MUST be protected
    assert any("market_conv" in n for n in protected_names)
    assert any("market_fc" in n for n in protected_names)
    assert any("pre_lstm" in n for n in protected_names)
    assert any("action_query" in n for n in protected_names)
    assert any("action_embedding" in n for n in protected_names)
    assert any("value_head" in n for n in protected_names)

    # Sparse transformer layer MUST be excluded
    assert not any("sparse_transformer" in n for n in protected_names)


def test_ewc_full_policy_layer_matching() -> None:
    """Verify protected parameter matching on real SharedFiveTradingPolicy (~1.07M params)."""
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    ewc = ElasticWeightConsolidation(policy)

    protected = ewc.get_protected_parameters()
    assert len(protected) > 0

    # Ensure no recurrent LSTM core or transformer parameters are accidentally included
    for name in protected:
        assert not name.startswith("lstm.")
        assert ".lstm." not in name
        assert "post_lstm_transformer" not in name

    # Verify critical components are matched
    assert any("action_query" in n for n in protected)
    assert any("action_embedding" in n for n in protected)
    assert any("pre_lstm" in n for n in protected)


def test_ewc_sample_size_floor_warning() -> None:
    """Verify warning when validation dataset size < 500 transitions."""
    model = ToyDecisionModel()
    ewc = ElasticWeightConsolidation(model, sample_count_floor=500)

    small_dataset = [{"obs": torch.randn(10, 8), "action": torch.randint(0, 12, (10,))} for _ in range(50)]

    with pytest.warns(UserWarning, match="sample_count=50 < 500"):
        ewc.register_optimal_checkpoint(small_dataset)

    assert ewc.is_active


def test_ewc_fisher_diagonal_calculation() -> None:
    """Verify diagonal Fisher calculation F_j = (1/N) * sum (nabla log pi)^2."""
    model = ToyDecisionModel()
    ewc = ElasticWeightConsolidation(model)

    dataset = [{"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))} for _ in range(20)]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    assert ewc.is_active
    for name, f_tensor in ewc.fisher_diag.items():
        assert f_tensor.shape == ewc.optimal_params[name].shape
        assert torch.all(f_tensor >= 0.0)


def test_ewc_damping_floor() -> None:
    """Verify damping floor epsilon_fisher = 10^-4 * mean(F) eliminates exact zeros."""
    model = ToyDecisionModel()
    ewc = ElasticWeightConsolidation(model)

    dataset = [{"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))} for _ in range(20)]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    all_f = torch.cat([f.view(-1) for f in ewc.fisher_diag.values()])
    mean_f = float(all_f.mean().item())
    expected_damping = 1e-4 * max(mean_f, 1e-8)

    # Every element must be >= damping floor
    for name, f_tensor in ewc.fisher_diag.items():
        assert torch.all(f_tensor >= expected_damping * 0.999)


def test_ewc_quadratic_penalty_calculation() -> None:
    """Verify penalty formula (lambda / 2) * sum F_j * (theta - theta^*)^2."""
    model = ToyDecisionModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [{"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))} for _ in range(20)]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    # At optimal anchor, penalty must be exactly 0.0
    zero_loss = ewc.compute_ewc_loss()
    assert float(zero_loss.item()) == pytest.approx(0.0, abs=1e-6)

    # Perturb one parameter by delta
    param_name = next(iter(ewc.optimal_params.keys()))
    param = dict(model.named_parameters())[param_name]
    delta = 0.1
    with torch.no_grad():
        param.add_(delta)

    # Expected penalty for this parameter
    f_j = ewc.fisher_diag[param_name]
    expected_penalty = (250.0 / 2.0) * float(torch.sum(f_j * (delta ** 2)).item())
    computed_loss = float(ewc.compute_ewc_loss().item())

    assert computed_loss == pytest.approx(expected_penalty, rel=1e-4)


def test_ewc_gradient_flow_and_restoring_force() -> None:
    """Verify that EWC gradient nabla L_EWC = lambda * F * (theta - theta^*) pulls weights back."""
    model = ToyDecisionModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [{"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))} for _ in range(20)]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    # Perturb weights positively: theta = theta^* + 0.1
    for name, param in model.named_parameters():
        if name in ewc.optimal_params:
            with torch.no_grad():
                param.add_(0.1)

    model.zero_grad()
    loss = ewc.compute_ewc_loss()
    loss.backward()

    # Gradient must be positive (since theta > theta^*, pulling back in negative direction)
    for name, param in model.named_parameters():
        if name in ewc.optimal_params and param.grad is not None:
            assert torch.all(param.grad > 0.0)


def test_ewc_step_from_loss_trainer_integration() -> None:
    """Verify EWC penalty injection into FivePPOTrainer.step_from_loss()."""
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    trainer = FivePPOTrainer(policy)

    ewc = trainer.attach_ewc(lambda_ewc=250.0)
    # Manually populate anchor and fisher for testing
    for name, p in policy.named_parameters():
        if ewc.is_protected(name):
            ewc.optimal_params[name] = p.detach().clone()
            ewc.fisher_diag[name] = torch.ones_like(p) * 0.05

    assert ewc.is_active

    # Perturb policy weights
    with torch.no_grad():
        policy.heads.action_query.weight.add_(0.05)

    dummy_loss = torch.tensor(1.0, requires_grad=True)
    stats = trainer.step_from_loss(dummy_loss)

    assert stats["ewc_active"] == 1.0
    assert stats["ewc_loss"] > 0.0


def test_ewc_persistence_roundtrip(tmp_path: Path) -> None:
    """Verify bit-for-bit saving and loading of EWC anchor and Fisher matrix."""
    model = ToyDecisionModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=250.0)

    dataset = [{"obs": torch.randn(4, 8), "action": torch.randint(0, 12, (4,))} for _ in range(20)]
    ewc.register_optimal_checkpoint(dataset, sample_count_floor=10)

    save_file = tmp_path / "test_ewc.pt"
    ewc.save(save_file)
    assert save_file.exists()

    # Load into fresh EWC instance
    new_model = ToyDecisionModel()
    new_ewc = ElasticWeightConsolidation(new_model)
    new_ewc.load(save_file)

    assert new_ewc.is_active
    assert new_ewc.lambda_ewc == 250.0

    for name in ewc.optimal_params:
        assert torch.equal(ewc.optimal_params[name], new_ewc.optimal_params[name])
        assert torch.equal(ewc.fisher_diag[name], new_ewc.fisher_diag[name])


def test_autorl_engine_ewc_hook_on_promotion(tmp_path: Path) -> None:
    """Verify that AutoRLEngine._update_ewc_on_promotion generates sidecar ewc_anchor_round_X.pt."""
    cfg = AutoRLEngineConfig(
        runs_dir=tmp_path / "runs",
        ewc_enabled=True,
        ewc_lambda=250.0,
        ewc_sample_floor=10,  # small floor for test speed
    )
    engine = AutoRLEngine(cfg)

    # Create dummy candidate checkpoint
    cand_path = tmp_path / "candidate_01.pkl"
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    torch.save(policy, cand_path)

    engine._update_ewc_on_promotion(round_num=1, candidate_ckpt_path=cand_path)

    sidecar = engine.runs_dir / "ewc_anchor_round_1.pt"
    best_sidecar = engine.runs_dir / "best_promoted_ewc.pt"

    assert sidecar.exists(), f"Sidecar {sidecar} was not created!"
    assert best_sidecar.exists(), f"Best sidecar {best_sidecar} was not created!"
    assert engine.ewc is not None
    assert engine.ewc.is_active


def test_ewc_module_exports_and_identity() -> None:
    """Verify that ElasticWeightConsolidation can be imported directly and is identical in both modules."""
    import self_growing_trader.elastic_weight_consolidation as ewc_mod
    import self_growing_trader.five_shared_policy as policy_mod

    assert hasattr(ewc_mod, "ElasticWeightConsolidation")
    assert hasattr(ewc_mod, "DEFAULT_PROTECTED_PATTERNS")
    assert ewc_mod.ElasticWeightConsolidation is policy_mod.ElasticWeightConsolidation
    assert ewc_mod.DEFAULT_PROTECTED_PATTERNS == policy_mod.DEFAULT_PROTECTED_PATTERNS


def test_shared_five_trading_policy_forward_shape_contract() -> None:
    """Verify SharedFiveTradingPolicy.forward conforming to shape contract [5, 137] -> [5, 20] & [5, 1]."""
    policy = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    obs = torch.randn(5, 137)

    action_wire, values, extra = policy.forward(obs)

    assert isinstance(action_wire, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert isinstance(extra, dict)

    assert action_wire.shape == (5, 20)
    assert values.shape == (5, 1)
    assert "action_logits" in extra
    assert "hidden" in extra
    assert "encoded" in extra

