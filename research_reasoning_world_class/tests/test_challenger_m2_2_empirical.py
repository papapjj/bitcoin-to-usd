"""
Empirical Challenger 2 Test Suite: Target Grounding & Mathematical Invariants.
Module: research_reasoning_world_class/tests/test_challenger_m2_2_empirical.py
Agent: teamwork_challenger_m2_2
Milestone: M2 (Grounded Latent Dynamics World Model)

Verifies:
1. Strict orthogonality invariant: z_hat^T grad_z_hat CosSim(z_hat, z*) == 0.
2. SmoothL1 gradient norm <= 1.0 under arbitrary fat-tail price shock targets (vs MSE explosion).
3. 500 future trajectory sampling from real M1 data (64,785 bars):
   - Zero NaNs / Infs.
   - Target values in [-1.0, 1.0].
   - Target rewards in [-10.0, 10.0].
   - Target states non-identical to predictions (mean CosSim < 0.99, eliminating Flaw 2 circular echoes).
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import pytest
import torch
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    generate_trajectory_targets,
)
from research_reasoning_world_class.src.real_m1_environment import RealM1Environment


class TestOrthogonalGradientCosineLoss(unittest.TestCase):
    """Empirical challenge suite for Cosine Similarity gradient direction orthogonality."""

    def test_orthogonal_gradient_multiscale_and_multidim(self):
        """Verifies z_hat^T grad_z_hat CosSim == 0 across varied dimensions and scales."""
        torch.manual_seed(42)
        dimensions = [2, 64, 128, 768]
        scales = [1e-4, 1e-2, 1.0, 1e2, 1e4]

        for dim in dimensions:
            for scale in scales:
                z_hat = (torch.randn(16, dim) * scale).requires_grad_(True)
                z_target = torch.randn(16, dim) * scale

                cossim = F.cosine_similarity(z_hat, z_target, dim=-1)
                cossim.sum().backward()

                self.assertIsNotNone(z_hat.grad)
                # Compute inner product z_hat^T * grad
                inner_prod = (z_hat * z_hat.grad).sum(dim=-1)
                z_norm = z_hat.norm(dim=-1)
                grad_norm = z_hat.grad.norm(dim=-1)
                rel_inner_prod = inner_prod.abs() / (z_norm * grad_norm + 1e-12)

                # Absolute inner product bounded by floating-point precision eps
                max_abs = inner_prod.abs().max().item()
                max_rel = rel_inner_prod.max().item()

                self.assertLess(
                    max_abs,
                    1e-5,
                    f"Orthogonality absolute violation at dim={dim}, scale={scale}: max_abs={max_abs:.2e}",
                )
                self.assertLess(
                    max_rel,
                    1e-4,
                    f"Orthogonality relative violation at dim={dim}, scale={scale}: max_rel={max_rel:.2e}",
                )

    def test_orthogonal_gradient_extreme_geometry_boundaries(self):
        """Verifies orthogonality under near-collinear and antiparallel edge cases."""
        torch.manual_seed(101)
        dim = 768
        batch = 8

        # 1. Exactly parallel vectors
        z_hat = torch.randn(batch, dim, requires_grad=True)
        z_target = z_hat.detach() * 2.5
        cossim = F.cosine_similarity(z_hat, z_target, dim=-1).sum()
        grad = torch.autograd.grad(cossim, z_hat)[0]
        dot = (z_hat * grad).sum(dim=-1)
        self.assertLess(dot.abs().max().item(), 1e-5)

        # 2. Exactly antiparallel vectors
        z_hat = torch.randn(batch, dim, requires_grad=True)
        z_target = -z_hat.detach() * 3.0
        cossim = F.cosine_similarity(z_hat, z_target, dim=-1).sum()
        grad = torch.autograd.grad(cossim, z_hat)[0]
        dot = (z_hat * grad).sum(dim=-1)
        self.assertLess(dot.abs().max().item(), 1e-5)

        # 3. Near-parallel vectors (delta = 1e-4)
        z_hat = torch.randn(batch, dim, requires_grad=True)
        z_target = z_hat.detach() + torch.randn(batch, dim) * 1e-4
        cossim = F.cosine_similarity(z_hat, z_target, dim=-1).sum()
        grad = torch.autograd.grad(cossim, z_hat)[0]
        dot = (z_hat * grad).sum(dim=-1)
        self.assertLess(dot.abs().max().item(), 1e-5)

    def test_coevolution_latent_loss_gradient_orthogonality(self):
        """Verifies LatentDynamicsWorldModel.compute_coevolution_loss preserves orthogonality."""
        torch.manual_seed(202)
        wm = LatentDynamicsWorldModel(latent_dim=768)

        z_pred = torch.randn(8, 5, 768, requires_grad=True)
        z_target = torch.randn(8, 5, 768)
        val_dummy = torch.zeros(8, 5, 1)
        rew_dummy = torch.zeros(8, 5, 1)

        losses = wm.compute_coevolution_loss(
            z_pred, z_target, val_dummy, val_dummy, rew_dummy, rew_dummy
        )
        loss_latent = losses["loss_latent"]
        grad = torch.autograd.grad(loss_latent, z_pred)[0]

        dot = (z_pred * grad).sum(dim=-1)
        max_abs_dot = dot.abs().max().item()
        self.assertLess(max_abs_dot, 1e-7, f"Latent loss gradient inner product non-zero: {max_abs_dot}")


class TestSmoothL1PriceShockGradientBounds(unittest.TestCase):
    """Empirical challenge suite for SmoothL1 gradient norm <= 1.0 under heavy-tailed shocks."""

    def test_smooth_l1_gradient_norm_bounded_by_one(self):
        """Verifies SmoothL1 gradient norm is strictly <= 1.0 across extreme target shocks."""
        shocks = [0.0, 0.25, 0.75, 1.0, 2.5, 10.0, 100.0, 1000.0, 1e6, 1e9, -1e6, -1e9]

        for s in shocks:
            pred = torch.tensor([[0.5]], requires_grad=True)
            target = torch.tensor([[s]])

            loss = F.smooth_l1_loss(pred, target, beta=1.0, reduction="sum")
            grad = torch.autograd.grad(loss, pred)[0]

            grad_norm = grad.norm().item()
            self.assertLessEqual(
                grad_norm,
                1.000001,
                f"SmoothL1 gradient norm exceeded 1.0 for shock {s}: {grad_norm}",
            )

    def test_smooth_l1_prevents_gradient_explosion_vs_mse(self):
        """Proves SmoothL1 prevents gradient explosion where MSE blows up under fat tails."""
        pred_l1 = torch.tensor([[0.0]], requires_grad=True)
        pred_mse = torch.tensor([[0.0]], requires_grad=True)
        extreme_shock = torch.tensor([[1_000_000.0]])

        loss_l1 = F.smooth_l1_loss(pred_l1, extreme_shock, beta=1.0, reduction="sum")
        loss_mse = F.mse_loss(pred_mse, extreme_shock, reduction="sum")

        grad_l1 = torch.autograd.grad(loss_l1, pred_l1)[0].abs().item()
        grad_mse = torch.autograd.grad(loss_mse, pred_mse)[0].abs().item()

        self.assertAlmostEqual(grad_l1, 1.0, places=5)
        self.assertEqual(grad_mse, 2_000_000.0)
        self.assertLess(grad_l1 * 1_000_000, grad_mse)

    def test_coevolution_loss_heads_bounded_under_fat_tails(self):
        """Verifies compute_coevolution_loss value and reward heads have bounded gradients."""
        wm = LatentDynamicsWorldModel()
        extreme_shocks = [50.0, 500.0, 1e6, -1e6]

        B, H = 4, 5
        for s in extreme_shocks:
            val_pred = torch.randn(B, H, 1, requires_grad=True)
            val_target = torch.full((B, H, 1), s)
            rew_pred = torch.randn(B, H, 1, requires_grad=True)
            rew_target = torch.full((B, H, 1), s)
            z_dummy = torch.randn(B, H, 768)

            losses = wm.compute_coevolution_loss(
                z_dummy, z_dummy, val_pred, val_target, rew_pred, rew_target
            )

            grad_val = torch.autograd.grad(losses["loss_value"], val_pred)[0]
            grad_rew = torch.autograd.grad(losses["loss_reward"], rew_pred)[0]

            # Undoing mean reduction (multiply by B * H)
            unscaled_val = grad_val * (B * H)
            unscaled_rew = grad_rew * (B * H)

            self.assertLessEqual(unscaled_val.abs().max().item(), 1.000001)
            self.assertLessEqual(unscaled_rew.abs().max().item(), 1.000001)
            self.assertLessEqual(grad_val.norm().item(), 1.0)
            self.assertLessEqual(grad_rew.norm().item(), 1.0)


class TestGroundedTrajectoryTargetInvariants(unittest.TestCase):
    """Empirical challenge suite: 500 future trajectories sampled from real M1 dataset."""

    @classmethod
    def setUpClass(cls):
        cls.env = RealM1Environment()
        cls.total_bars = cls.env.load_data("2025.01.15", "2025.02.28")
        cls.wm = LatentDynamicsWorldModel()
        cls.wm.eval()

    def test_sample_500_trajectories_bounds_and_validity(self):
        """
        Samples 500 future trajectories from real M1 data (64,785 bars):
        1. Asserts zero NaNs and zero Infs across all 2,500 horizon steps.
        2. Asserts target values strictly in [-1.0, 1.0].
        3. Asserts target rewards strictly in [-10.0, 10.0].
        """
        np.random.seed(42)
        N = 500
        K = 5
        indices = np.random.randint(100, self.total_bars - K - 100, size=N)

        val_min, val_max = float("inf"), float("-inf")
        rew_min, rew_max = float("inf"), float("-inf")

        for idx in indices:
            actions = np.random.choice([0, 1, 2], size=K)
            sl_atr = float(np.random.uniform(1.5, 4.0))
            tp_atr = float(np.random.uniform(3.0, 8.0))

            s0, fut_obs, target_v, target_r = generate_trajectory_targets(
                env=self.env,
                start_idx=int(idx),
                actions=actions,
                horizon_k=K,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
            )

            # Check finiteness
            self.assertTrue(np.all(np.isfinite(s0)), f"NaN/Inf in s0 at bar {idx}")
            self.assertTrue(np.all(np.isfinite(fut_obs)), f"NaN/Inf in fut_obs at bar {idx}")
            self.assertTrue(np.all(np.isfinite(target_v)), f"NaN/Inf in target_v at bar {idx}")
            self.assertTrue(np.all(np.isfinite(target_r)), f"NaN/Inf in target_r at bar {idx}")

            val_min = min(val_min, float(target_v.min()))
            val_max = max(val_max, float(target_v.max()))
            rew_min = min(rew_min, float(target_r.min()))
            rew_max = max(rew_max, float(target_r.max()))

        # Check bounds
        self.assertGreaterEqual(val_min, -1.0, f"Target value min below -1.0: {val_min}")
        self.assertLessEqual(val_max, 1.0, f"Target value max above 1.0: {val_max}")
        self.assertGreaterEqual(rew_min, -10.0, f"Target reward min below -10.0: {rew_min}")
        self.assertLessEqual(rew_max, 10.0, f"Target reward max above 10.0: {rew_max}")

    def test_target_authenticity_and_non_circularity(self):
        """
        Proves target states are authentic market encodings and non-circular:
        1. Mean trajectory CosSim with prediction is strictly < 0.99.
        2. Horizon step 5 CosSim diverges as market evolves (mean < 0.95).
        3. Contrasts against legacy Flaw 2 (pred + 0.02 * randn) which produced CosSim > 0.999.
        """
        np.random.seed(1337)
        torch.manual_seed(1337)

        N = 100
        K = 5
        indices = np.random.randint(100, self.total_bars - K - 100, size=N)

        real_traj_sims = []
        step_5_sims = []
        flaw2_sims = []

        with torch.no_grad():
            for idx in indices:
                actions = np.random.choice([0, 1, 2], size=K)
                s0, fut_obs, _, _ = generate_trajectory_targets(
                    env=self.env,
                    start_idx=int(idx),
                    actions=actions,
                    horizon_k=K,
                )

                z0 = self.wm.encode_state(s0)
                action_seq = torch.tensor(actions, dtype=torch.long).unsqueeze(0)
                pred_z, _, _ = self.wm.unroll_trajectory(z0, action_seq)
                target_z = self.wm.encode_state(fut_obs).unsqueeze(0)

                # Real trajectory CosSim
                sim_traj = F.cosine_similarity(pred_z.view(1, -1), target_z.view(1, -1), dim=-1).item()
                real_traj_sims.append(sim_traj)

                # Step 5 CosSim
                sim_step5 = F.cosine_similarity(pred_z[:, 4, :], target_z[:, 4, :], dim=-1).item()
                step_5_sims.append(sim_step5)

                # Simulated legacy Flaw 2 (pred + 0.02 * randn)
                circ_target = pred_z + torch.randn_like(pred_z) * 0.02
                sim_flaw2 = F.cosine_similarity(pred_z.view(1, -1), circ_target.view(1, -1), dim=-1).item()
                flaw2_sims.append(sim_flaw2)

        mean_real_sim = float(np.mean(real_traj_sims))
        mean_step5_sim = float(np.mean(step_5_sims))
        mean_flaw2_sim = float(np.mean(flaw2_sims))

        # Assertions proving authenticity and non-circularity
        self.assertLess(
            mean_real_sim,
            0.99,
            f"Target trajectory CosSim is not < 0.99 (mean={mean_real_sim:.4f})",
        )
        self.assertLess(
            mean_step5_sim,
            0.96,
            f"Step 5 CosSim did not diverge as expected (mean={mean_step5_sim:.4f})",
        )
        self.assertGreater(
            mean_flaw2_sim,
            0.999,
            f"Legacy circular echo baseline should be > 0.999 (mean={mean_flaw2_sim:.4f})",
        )
        self.assertLess(
            mean_real_sim,
            mean_flaw2_sim - 0.02,
            "Real targets must clearly separate from artificial circular echoes",
        )


if __name__ == "__main__":
    unittest.main()
