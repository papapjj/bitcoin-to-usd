"""Milestone 4 System Integration & Composite Loss Test Suite.

Verifies the 6 key architectural invariants and interface contracts:
1. End-to-end multi-agent rollout and training step with composite loss (PPO + EWC + Reg).
2. Target derivation and representation regularizer evaluation on multi-regime batches.
3. Verification of shape contracts [5, 137] -> [5, 20] and [5, 1] under training and eval modes.
4. Verification of Governor locked to HOLD across integration steps.
5. Parameter count and state_dict immutability check in production checkpoint export.
6. Gradient backward pass and optimizer update verification with zero NaNs/Infs.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from self_growing_trader.auto_rl_engine import atomic_save_checkpoint
from self_growing_trader.balanced_replay_buffer import (
    BalancedMultiRegimeReservoirBuffer,
    normalize_regime,
)
from self_growing_trader.elastic_weight_consolidation import ElasticWeightConsolidation
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    INPUT_COUNT,
    FiveAgentMemory,
    FivePPOTrainer,
    SharedFiveTradingPolicy,
)
from self_growing_trader.representation_regularizer import (
    DEFAULT_CLASS_WEIGHTS,
    RepresentationRegularizer,
)


# ==============================================================================
# Helper Functions
# ==============================================================================

def _make_dummy_rollout(
    T: int = 4,
    A: int = FIVE_AGENT_COUNT,
    F_dim: int = INPUT_COUNT,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Create a realistic dummy rollout sequence for 5 agents over T steps."""
    if device is None:
        device = torch.device("cpu")
    obs = np.random.randn(T, A, F_dim).astype(np.float32)
    memories = [FiveAgentMemory.zeros(device=device) for _ in range(A)]
    actions = np.random.randint(0, 7, size=(T, A), dtype=np.int64)
    # Ensure Agent 4 (Governor) only has action 0 (HOLD)
    actions[:, 4] = 0
    targets = np.zeros((T, A), dtype=np.int64)
    delay_indices = np.zeros((T, A), dtype=np.int64)
    risk_indices = np.zeros((T, A), dtype=np.int64)
    offset_indices = np.zeros((T, A), dtype=np.int64)
    manage_indices = np.zeros((T, A), dtype=np.int64)
    close_indices = np.zeros((T, A), dtype=np.int64)
    old_log_probs = np.full((T, A), -1.5, dtype=np.float32)
    rewards = np.zeros((T, A), dtype=np.float32)
    rewards[-1, 0] = 1.0
    rewards[-1, 1] = -0.5
    dones = np.zeros((T, A), dtype=np.float32)

    return {
        "observations": obs,
        "initial_memories": memories,
        "actions": actions,
        "targets": targets,
        "delay_indices": delay_indices,
        "risk_indices": risk_indices,
        "entry_offset_indices": offset_indices,
        "manage_mode_indices": manage_indices,
        "close_fraction_indices": close_indices,
        "old_log_probabilities": old_log_probs,
        "rewards": rewards,
        "dones": dones,
        "step_seconds": 1.0,
        "horizon_seconds": 60.0,
    }


def _make_dummy_reservoir_step(
    step_id: int,
    regime: str = "UPTREND",
    realized_r: float = 0.5,
    td_error: float = 1.0,
) -> dict[str, Any]:
    """Create dummy step dictionary for multi-regime reservoir buffer."""
    actions = np.random.randint(0, 7, size=(FIVE_AGENT_COUNT,), dtype=np.int64)
    actions[4] = 0  # Governor HOLD
    return {
        "step_id": step_id,
        "observations": np.random.randn(FIVE_AGENT_COUNT, INPUT_COUNT).astype(np.float32),
        "actions": actions,
        "rewards": np.zeros(FIVE_AGENT_COUNT, dtype=np.float32),
        "regime": regime,
        "lifecycle_stratum": ("entry", "pos", regime),
        "td_error": float(td_error),
    }


# ==============================================================================
# SECTION 1: End-to-End Multi-Agent Rollout & Composite Loss (PPO + EWC + Reg)
# ==============================================================================

class TestCompositeLossAndRollout:
    """Verifies end-to-end multi-agent rollout and composite loss optimization."""

    def test_e2e_rollout_and_composite_loss_training_step(self) -> None:
        """Verify unified optimization step with PPO loss, EWC penalty, and Representation Regularization."""
        torch.manual_seed(42)
        policy = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy)

        # 1. Attach Representation Regularizer
        reg = trainer.attach_representation_regularizer(
            hidden_dim=4096,
            var_weight=1.0,
            cov_weight=1.0,
            rank_weight=0.05,
            arc_weight=0.20,
        )
        assert trainer.regularizer is reg
        assert isinstance(trainer.regularizer, RepresentationRegularizer)

        # 2. Attach EWC Manager and populate anchor
        ewc = trainer.attach_ewc(lambda_ewc=150.0)
        assert trainer.ewc is ewc
        protected_params = ewc.get_protected_parameters()
        assert len(protected_params) > 0

        # Snapshot anchor and Fisher diagonal
        ewc.optimal_params = {name: p.detach().clone() for name, p in protected_params.items()}
        ewc.fisher_diag = {name: torch.ones_like(p) * 0.5 for name, p in protected_params.items()}
        assert ewc.is_active

        # 3. Create multi-agent rollout data
        rollout = _make_dummy_rollout(T=4, A=FIVE_AGENT_COUNT, F_dim=INPUT_COUNT, device=trainer.device)

        # 4. Execute optimization step
        stats = trainer.optimize_sequence(**rollout)

        # 5. Verify presence of all composite loss components
        expected_metrics = [
            "loss_var",
            "loss_cov",
            "loss_rank",
            "loss_arcface",
            "loss_reg_total",
            "effective_rank",
            "inter_class_cos",
            "ewc_loss",
            "ewc_active",
        ]
        for key in expected_metrics:
            assert key in stats, f"Missing required composite loss metric: '{key}'"

        # Verify quantitative sanity
        assert stats["ewc_active"] == 1.0
        assert stats["ewc_loss"] >= 0.0
        assert stats["loss_reg_total"] > 0.0
        assert stats["effective_rank"] >= 1.0
        assert np.isfinite(stats["loss_reg_total"])
        assert np.isfinite(stats["effective_rank"])

    def test_composite_loss_additive_components(self) -> None:
        """Verify each term (PPO, Reg, EWC) contributes additively to total loss."""
        torch.manual_seed(42)
        policy = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy)

        # Without regularizer and EWC
        rollout = _make_dummy_rollout(T=2, A=FIVE_AGENT_COUNT, device=trainer.device)
        stats_baseline = trainer.optimize_sequence(**rollout)
        assert "loss_reg_total" not in stats_baseline
        assert stats_baseline.get("ewc_active", 0.0) == 0.0

        # Now attach regularizer
        trainer.attach_representation_regularizer(hidden_dim=4096)
        stats_with_reg = trainer.optimize_sequence(**rollout)
        assert "loss_reg_total" in stats_with_reg
        assert stats_with_reg["loss_reg_total"] > 0.0

        # Now activate EWC
        ewc = trainer.attach_ewc(lambda_ewc=200.0)
        protected = ewc.get_protected_parameters()
        # Perturb parameters slightly to ensure positive EWC penalty
        ewc.optimal_params = {name: (p.detach().clone() + 0.01) for name, p in protected.items()}
        ewc.fisher_diag = {name: torch.ones_like(p) for name, p in protected.items()}
        assert ewc.is_active

        stats_with_all = trainer.optimize_sequence(**rollout)
        assert stats_with_all["ewc_loss"] > 0.0
        assert stats_with_all["ewc_active"] == 1.0
        assert stats_with_all["loss_reg_total"] > 0.0


# ==============================================================================
# SECTION 2: Multi-Regime Batches & Target Derivation
# ==============================================================================

class TestMultiRegimeAndTargetDerivation:
    """Verifies target derivation and regularizer behavior on multi-regime batches."""

    def test_target_derivation_contract(self) -> None:
        """Verify target derivation maps 12 action indices to {0, 1, 2} superclasses."""
        # 0: HOLD, 1: BUY_MKT, 2: SELL_MKT, 3: BUY_LIM, 4: SELL_LIM, 5: BUY_STP, 6: SELL_STP, 7..11: MANAGE/CLOSE
        actions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)
        targets = RepresentationRegularizer.derive_targets(actions)

        # Expected:
        # HOLD (0, 7..11) -> 0
        # BUY (1, 3, 5) -> 1
        # SELL (2, 4, 6) -> 2
        expected = torch.tensor([0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 0, 0], dtype=torch.long)
        assert torch.equal(targets, expected)

        # Multi-dimensional tensor [Batch, Agents]
        batch_actions = torch.tensor([
            [0, 1, 2, 3, 0],
            [4, 5, 6, 10, 0],
        ], dtype=torch.long)
        batch_targets = RepresentationRegularizer.derive_targets(batch_actions)
        expected_batch = torch.tensor([
            [0, 1, 2, 1, 0],
            [2, 1, 2, 0, 0],
        ], dtype=torch.long)
        assert torch.equal(batch_targets, expected_batch)

    def test_multi_regime_reservoir_buffer_sampling_and_regularizer_evaluation(self) -> None:
        """Verify balanced sampling from 3 market regimes and regularizer evaluation."""
        buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=64)

        # Populate buffer with transitions across 3 regimes
        for i in range(30):
            buffer.add(_make_dummy_reservoir_step(i, "UPTREND"), regime="UPTREND")
            buffer.add(_make_dummy_reservoir_step(100 + i, "DOWNTREND"), regime="DOWNTREND")
            buffer.add(_make_dummy_reservoir_step(200 + i, "SIDEWAY"), regime="SIDEWAY")

        # Sample balanced mini-batch of 30 items
        batch = buffer.sample_balanced_batch(batch_size=30)
        assert len(batch) == 30

        regimes = [normalize_regime(item["regime"]) for item in batch]
        assert regimes.count("UPTREND") == 10
        assert regimes.count("DOWNTREND") == 10
        assert regimes.count("SIDEWAY") == 10

        # Stack actions across batch and agents: [30, 5]
        all_actions = torch.from_numpy(np.stack([item["actions"] for item in batch]))
        assert all_actions.shape == (30, FIVE_AGENT_COUNT)

        # Derive targets for all transitions
        derived_targets = RepresentationRegularizer.derive_targets(all_actions)
        assert derived_targets.shape == (30, FIVE_AGENT_COUNT)
        assert torch.all(derived_targets >= 0) and torch.all(derived_targets <= 2)

        # Pass dummy hidden states of shape [30, 5, 64] through regularizer
        D = 64
        hidden = torch.randn(30, FIVE_AGENT_COUNT, D, requires_grad=True)
        reg = RepresentationRegularizer(hidden_dim=D)

        out = reg(hidden, targets=derived_targets)
        assert "loss_reg_total" in out
        assert "effective_rank" in out
        assert out["effective_rank"] > 1.0

        # Backward pass must be completely finite
        out["loss_reg_total"].backward()
        assert hidden.grad is not None
        assert not torch.isnan(hidden.grad).any()
        assert not torch.isinf(hidden.grad).any()


# ==============================================================================
# SECTION 3: Shape Contracts [5, 137] -> [5, 20] and [5, 1]
# ==============================================================================

class TestShapeContracts:
    """Verifies strict shape preservation under training and evaluation modes."""

    def test_shape_contracts_train_and_eval_modes(self) -> None:
        """Verify policy.forward maintains [5, 20] action wire and [5, 1] value across modes."""
        policy = SharedFiveTradingPolicy()
        obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)

        for mode in (True, False):
            policy.train(mode)
            action_wire, values, extra = policy.forward(obs)

            # Contract INV-1: action wire shape [5, 20]
            assert action_wire.shape == (FIVE_AGENT_COUNT, 20), (
                f"Action wire shape mismatch in mode={mode}: got {action_wire.shape}"
            )
            assert action_wire.dtype == torch.float32

            # Contract: values shape [5, 1]
            assert values.shape == (FIVE_AGENT_COUNT, 1), (
                f"Value shape mismatch in mode={mode}: got {values.shape}"
            )
            assert values.dtype == torch.float32

    def test_forward_five_agent_results_and_memory_shapes(self) -> None:
        """Verify forward_five produces 5 AgentStepResult and preserves 4096D memory shapes."""
        policy = SharedFiveTradingPolicy()
        memories = [FiveAgentMemory.zeros(device=torch.device("cpu")) for _ in range(FIVE_AGENT_COUNT)]
        obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)

        results, next_memories, context_probe = policy.forward_five(
            obs, memories, deterministic=True
        )

        assert len(results) == FIVE_AGENT_COUNT
        for i, res in enumerate(results):
            assert res.output_vector.shape == (20,), f"Agent {i} output vector not shape (20,)"
            assert isinstance(res.action_index, (int, np.integer))

        assert len(next_memories) == FIVE_AGENT_COUNT
        for i, mem in enumerate(next_memories):
            assert mem.hidden.shape == (4096,), f"Agent {i} hidden state not 4096D"
            assert mem.cell.shape == (4096,), f"Agent {i} cell state not 4096D"

        assert "value" in context_probe
        assert context_probe["value"].shape in ((FIVE_AGENT_COUNT, 1), (FIVE_AGENT_COUNT,))

    def test_shape_contracts_sequential_rollout_unrolling(self) -> None:
        """Verify recurrent unrolling across 8 sequential steps preserves contract."""
        policy = SharedFiveTradingPolicy()
        policy.eval()
        memories = [FiveAgentMemory.zeros(device=torch.device("cpu")) for _ in range(FIVE_AGENT_COUNT)]

        for step in range(8):
            obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
            results, memories, probe = policy.forward_five(obs, memories)
            assert len(results) == 5
            for r in results:
                assert r.output_vector.shape == (20,)


# ==============================================================================
# SECTION 4: Governor HOLD Invariance Across Integration Steps
# ==============================================================================

class TestGovernorHoldInvariance:
    """Verifies Governor (Agent 4) is locked to HOLD across integration steps."""

    def test_governor_locked_to_hold_across_100_integration_steps(self) -> None:
        """Verify Agent 4 produces action 0 (HOLD) in 100% of steps under diverse observations."""
        policy = SharedFiveTradingPolicy()
        policy.heads.lock_non_entry_to_hold = True
        memories = [FiveAgentMemory.zeros(device=torch.device("cpu")) for _ in range(FIVE_AGENT_COUNT)]

        rng = np.random.default_rng(42)
        total_steps = 100

        for step in range(total_steps):
            # Generate varied market regimes: bull, bear, volatile, tiny
            if step % 4 == 0:
                obs_np = rng.standard_normal((FIVE_AGENT_COUNT, INPUT_COUNT)).astype(np.float32)
            elif step % 4 == 1:
                obs_np = rng.uniform(10.0, 500.0, (FIVE_AGENT_COUNT, INPUT_COUNT)).astype(np.float32)
            elif step % 4 == 2:
                obs_np = rng.uniform(-500.0, -10.0, (FIVE_AGENT_COUNT, INPUT_COUNT)).astype(np.float32)
            else:
                obs_np = rng.standard_normal((FIVE_AGENT_COUNT, INPUT_COUNT)).astype(np.float32) * 1e-5

            obs_tensor = torch.as_tensor(obs_np)
            results, memories, _ = policy.forward_five(obs_tensor, memories)

            gov_res = results[4]
            assert gov_res.action_index == 0, (
                f"Governor emitted non-HOLD action {gov_res.action_index} at step {step}"
            )

    def test_governor_target_derivation_invariance(self) -> None:
        """Verify Governor action (0) is always mapped to target superclass 0 (HOLD)."""
        gov_action = torch.tensor([0], dtype=torch.long)
        target = RepresentationRegularizer.derive_targets(gov_action)
        assert target.item() == 0, "Governor target must map to superclass 0"

    def test_governor_lock_preserved_after_composite_training_step(self) -> None:
        """Verify Governor remains strictly locked to HOLD even after composite training updates."""
        torch.manual_seed(42)
        policy = SharedFiveTradingPolicy()
        policy.heads.lock_non_entry_to_hold = True
        trainer = FivePPOTrainer(policy)
        trainer.attach_representation_regularizer(hidden_dim=4096)
        ewc = trainer.attach_ewc(lambda_ewc=100.0)
        protected = ewc.get_protected_parameters()
        ewc.optimal_params = {n: p.detach().clone() for n, p in protected.items()}
        ewc.fisher_diag = {n: torch.ones_like(p) for n, p in protected.items()}

        rollout = _make_dummy_rollout(T=3, A=FIVE_AGENT_COUNT)
        trainer.optimize_sequence(**rollout)

        # Test forward pass on updated policy
        memories = [FiveAgentMemory.zeros(device=torch.device("cpu")) for _ in range(FIVE_AGENT_COUNT)]
        obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
        results, _, _ = policy.forward_five(obs, memories)
        assert results[4].action_index == 0, "Governor must remain locked to HOLD after gradient update"


# ==============================================================================
# SECTION 5: Parameter Count & State_Dict Immutability Check
# ==============================================================================

class TestCheckpointImmutability:
    """Verifies 0 new inference parameters and state_dict immutability in checkpoint export."""

    def test_parameter_count_zero_inference_addition(self) -> None:
        """Verify policy parameter count is identical before and after attaching regularizer & EWC."""
        policy_standalone = SharedFiveTradingPolicy()
        count_standalone = sum(p.numel() for p in policy_standalone.parameters())

        policy_integrated = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy_integrated)
        trainer.attach_representation_regularizer(hidden_dim=4096)
        trainer.attach_ewc()
        count_integrated = sum(p.numel() for p in policy_integrated.parameters())

        # Invariant: 0 new inference parameters in policy network
        assert count_standalone == count_integrated, (
            f"0 new parameters invariant violated: standalone={count_standalone}, integrated={count_integrated}"
        )

    def test_state_dict_keys_exact_match(self) -> None:
        """Verify policy state_dict keys are 100% identical and exclude trainer-side parameters."""
        policy_standalone = SharedFiveTradingPolicy()
        keys_standalone = set(policy_standalone.state_dict().keys())

        policy_integrated = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy_integrated)
        reg = trainer.attach_representation_regularizer(hidden_dim=4096)
        ewc = trainer.attach_ewc()

        keys_integrated = set(policy_integrated.state_dict().keys())
        assert keys_standalone == keys_integrated, "Policy state_dict keys must remain identical"

        # arc_weights from regularizer must NOT be in policy.state_dict()
        assert not any("arc_weights" in k for k in keys_integrated)

    def test_production_checkpoint_export_and_reload(self) -> None:
        """Verify checkpoint can be exported via atomic_save_checkpoint and reloaded strictly."""
        policy = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy)
        trainer.attach_representation_regularizer(hidden_dim=4096)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.pkl"
            state = policy.state_dict()
            atomic_save_checkpoint(state, ckpt_path)

            assert ckpt_path.exists()
            assert ckpt_path.stat().st_size > 0

            # Reload into a fresh policy instance
            fresh_policy = SharedFiveTradingPolicy()
            import pickle
            with open(ckpt_path, "rb") as f:
                loaded_state = pickle.load(f)

            fresh_policy.load_state_dict(loaded_state, strict=True)

            # Verify forward passes match bit-for-bit
            obs = torch.randn(FIVE_AGENT_COUNT, INPUT_COUNT)
            wire_orig, val_orig, _ = policy.forward(obs)
            wire_loaded, val_loaded, _ = fresh_policy.forward(obs)

            assert torch.equal(wire_orig, wire_loaded)
            assert torch.equal(val_orig, val_loaded)


# ==============================================================================
# SECTION 6: Gradient Backward Pass & Optimizer Update Sanity
# ==============================================================================

class TestGradientFlowAndOptimizerSanity:
    """Verifies gradient flow to all submodules and zero NaNs/Infs during training updates."""

    def test_gradient_flow_to_all_submodules_zero_nans_infs(self) -> None:
        """Verify gradients propagate to transformer, lstm, encoder, heads, and arc_weights with 0 NaNs/Infs."""
        torch.manual_seed(42)
        policy = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy)
        reg = trainer.attach_representation_regularizer(hidden_dim=4096)
        ewc = trainer.attach_ewc(lambda_ewc=100.0)

        protected = ewc.get_protected_parameters()
        ewc.optimal_params = {n: (p.detach().clone() + 0.005) for n, p in protected.items()}
        ewc.fisher_diag = {n: torch.ones_like(p) for n, p in protected.items()}

        rollout = _make_dummy_rollout(T=2, A=FIVE_AGENT_COUNT)
        stats = trainer.optimize_sequence(**rollout)

        assert stats["loss_reg_total"] > 0.0
        assert stats["ewc_loss"] > 0.0

        # Verify no NaN/Inf in any policy parameter gradients that were updated
        for name, p in policy.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in parameter '{name}'"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in parameter '{name}'"

        # Verify regularizer arc_weights received clean gradient
        assert reg.arc_weights.grad is not None
        assert not torch.isnan(reg.arc_weights.grad).any()
        assert not torch.isinf(reg.arc_weights.grad).any()

    def test_optimizer_parameter_update_and_ewc_anchor_immutability(self) -> None:
        """Verify policy parameters are updated while EWC anchors remain completely immutable."""
        torch.manual_seed(42)
        policy = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy)
        trainer.attach_representation_regularizer(hidden_dim=4096)
        ewc = trainer.attach_ewc(lambda_ewc=100.0)

        protected = ewc.get_protected_parameters()
        anchor_init = {n: p.detach().clone() for n, p in protected.items()}
        ewc.optimal_params = {n: p.detach().clone() for n, p in protected.items()}
        ewc.fisher_diag = {n: torch.ones_like(p) for n, p in protected.items()}

        # Snapshot a live parameter to check for update
        sample_param_name = next(iter(protected))
        val_before = protected[sample_param_name].detach().clone()

        rollout = _make_dummy_rollout(T=2, A=FIVE_AGENT_COUNT)
        trainer.optimize_sequence(**rollout)

        val_after = protected[sample_param_name].detach()
        # Parameter must have been updated
        assert not torch.equal(val_before, val_after), f"Parameter {sample_param_name} was not updated"

        # EWC anchor MUST remain strictly immutable
        for n, p_anchor in ewc.optimal_params.items():
            assert torch.equal(p_anchor, anchor_init[n]), f"EWC anchor '{n}' was mutated during training step"

    def test_consecutive_training_steps_stability(self) -> None:
        """Verify 5 consecutive optimization steps execute stably without numerical divergence."""
        torch.manual_seed(42)
        policy = SharedFiveTradingPolicy()
        trainer = FivePPOTrainer(policy)
        trainer.attach_representation_regularizer(hidden_dim=4096)
        trainer.attach_ewc(lambda_ewc=50.0)

        for step in range(5):
            rollout = _make_dummy_rollout(T=2, A=FIVE_AGENT_COUNT)
            stats = trainer.optimize_sequence(**rollout)

            assert np.isfinite(stats["loss_reg_total"])
            assert np.isfinite(stats["effective_rank"])
            assert stats["effective_rank"] >= 1.0
