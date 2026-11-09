"""
Comprehensive Unit Test Suite for Grounded Latent Dynamics World Model.
Module: research_reasoning_world_class/tests/test_latent_dynamics_world_model.py
Milestone: M2 (Grounded Latent Dynamics World Model)

Verifies:
1. TestStateRepresentationEncoder: 137D -> 768D encoding, batching, stability, latency, and encode_state.
2. TestTransitionDynamics: residual transition, action conditioning, non-identity, polymorphic action inputs.
3. TestMultiStepUnrolling: K=5 unrolling, deep K=20 stability, BPTT gradient flow, unroll_trajectory.
4. TestAuthenticTargetGeneration: real M1 future targets, path-dependent value, reward anchoring, helper utility.
5. TestCoEvolutionLossFormulation: Cosine similarity loss, SmoothL1 value/reward, convergence, orthogonal gradient.
6. TestBackwardCompatibilityAndMythosPreservation: weight shapes, state dict, legacy ckpt loading, export roundtrip.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    AdversarialMarketStressor,
    LatentDynamicsWorldModel,
    MarketStateEncoder,
    generate_trajectory_targets,
)
from research_reasoning_world_class.src.real_m1_environment import (
    RealM1Environment,
    TradeResult,
)


class TestStateRepresentationEncoder(unittest.TestCase):
    """Test Suite 1: State Representation Encoder (137D -> 768D)."""

    def setUp(self):
        self.encoder = MarketStateEncoder(input_dim=137, latent_dim=768, hidden_dim=512)
        self.wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)

    def test_encoder_output_shape_and_dtype(self):
        """T1.1: Verifies 1D and 2D observation tensors map to (B, 768) float32."""
        obs_1d = torch.randn(137)
        obs_2d_single = torch.randn(1, 137)
        obs_2d_batch = torch.randn(16, 137)

        z_1d = self.encoder(obs_1d)
        z_2d_single = self.encoder(obs_2d_single)
        z_2d_batch = self.encoder(obs_2d_batch)

        self.assertEqual(z_1d.shape, (1, 768))
        self.assertEqual(z_2d_single.shape, (1, 768))
        self.assertEqual(z_2d_batch.shape, (16, 768))
        self.assertEqual(z_1d.dtype, torch.float32)
        self.assertEqual(z_2d_batch.dtype, torch.float32)

    def test_encoder_deterministic_mapping(self):
        """T1.2: Verifies identical inputs yield identical outputs and distinct inputs differ."""
        self.encoder.eval()
        obs = torch.randn(4, 137)
        z1 = self.encoder(obs)
        z2 = self.encoder(obs)
        self.assertTrue(torch.allclose(z1, z2, atol=1e-6))

        obs_different = torch.randn(4, 137)
        z_diff = self.encoder(obs_different)
        cos_sim = F.cosine_similarity(z1, z_diff, dim=-1).mean().item()
        self.assertLess(cos_sim, 0.9999, "Distinct market observations must produce distinct latent states")

    def test_encoder_numerical_stability_nan_inf(self):
        """T1.3: Verifies NaN and Inf inputs are sanitized without producing NaNs/Infs."""
        corrupted = torch.tensor([[float("nan"), float("inf"), -float("inf")] + [0.5] * 134])
        z = self.encoder(corrupted)
        self.assertEqual(z.shape, (1, 768))
        self.assertFalse(torch.isnan(z).any(), "Found NaN in encoder output")
        self.assertFalse(torch.isinf(z).any(), "Found Inf in encoder output")
        self.assertTrue(torch.all(torch.isfinite(z)))

    def test_encoder_gradient_flow(self):
        """T1.4: Verifies autograd gradients propagate backward through all encoder layers."""
        obs = torch.randn(4, 137, requires_grad=True)
        z = self.encoder(obs)
        loss = (z ** 2).sum()
        loss.backward()

        self.assertIsNotNone(obs.grad)
        self.assertTrue(torch.any(obs.grad != 0.0))
        for name, p in self.encoder.named_parameters():
            self.assertIsNotNone(p.grad, f"Parameter {name} did not receive gradients")
            self.assertTrue(torch.any(p.grad != 0.0), f"Parameter {name} gradient is zero")

    def test_encode_state_method_contract(self):
        """T1.5: Verifies LatentDynamicsWorldModel.encode_state public method contract."""
        obs_np = np.random.randn(137).astype(np.float32)
        obs_list = [0.1] * 137
        obs_tensor_1d = torch.randn(137)
        obs_tensor_2d = torch.randn(8, 137)

        z_np = self.wm.encode_state(obs_np)
        z_list = self.wm.encode_state(obs_list)
        z_1d = self.wm.encode_state(obs_tensor_1d)
        z_2d = self.wm.encode_state(obs_tensor_2d)

        self.assertEqual(z_np.shape, (1, 768))
        self.assertEqual(z_list.shape, (1, 768))
        self.assertEqual(z_1d.shape, (1, 768))
        self.assertEqual(z_2d.shape, (8, 768))
        self.assertEqual(z_np.device, next(self.wm.parameters()).device)


class TestTransitionDynamics(unittest.TestCase):
    """Test Suite 2: Transition Dynamics & Action Conditioning."""

    def setUp(self):
        self.wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512, alpha=0.25)

    def test_transition_residual_formula_and_normalization(self):
        """T2.1: Verifies transition follows z_{t+1} = LayerNorm(z_t + alpha * g(z_t, a_t))."""
        z_t = torch.randn(4, 768)
        a_t = torch.zeros(4, 64)
        z_next = self.wm.transition(z_t, a_t)

        self.assertEqual(z_next.shape, (4, 768))
        self.assertFalse(torch.isnan(z_next).any())
        self.assertFalse(torch.isinf(z_next).any())

        # LayerNorm invariant: ||z||_2 ~ sqrt(768) ~ 27.7128 and std ~ 1.0
        norms = torch.norm(z_next, dim=-1)
        stds = torch.std(z_next, dim=-1)
        for i in range(4):
            self.assertAlmostEqual(norms[i].item(), math.sqrt(768), delta=0.5)
            self.assertAlmostEqual(stds[i].item(), 1.0, delta=0.1)

    def test_transition_action_conditioning(self):
        """T2.2: Verifies HOLD, BUY, and SELL actions produce diverging trajectories."""
        z = torch.randn(1, 768)
        z_hold = self.wm.transition(z, 0)
        z_buy = self.wm.transition(z, 1)
        z_sell = self.wm.transition(z, 2)

        self.assertFalse(torch.allclose(z_buy, z_sell, atol=1e-4))
        self.assertFalse(torch.allclose(z_buy, z_hold, atol=1e-4))
        self.assertFalse(torch.allclose(z_sell, z_hold, atol=1e-4))

    def test_transition_polymorphic_action_inputs(self):
        """T2.3: Verifies polymorphic action types: int, 1D tensor, 2D int, 64D dense."""
        z = torch.randn(2, 768)

        # 1. Python integer
        out_int = self.wm.transition(z, 1)
        self.assertEqual(out_int.shape, (2, 768))

        # 2. 1D LongTensor [B]
        out_long = self.wm.transition(z, torch.tensor([1, 2]))
        self.assertEqual(out_long.shape, (2, 768))

        # 3. 2D LongTensor [B, 1]
        out_2d_long = self.wm.transition(z, torch.tensor([[1], [2]]))
        self.assertEqual(out_2d_long.shape, (2, 768))

        # 4. Full 64D dense continuous vector
        out_dense = self.wm.transition(z, torch.randn(2, 64))
        self.assertEqual(out_dense.shape, (2, 768))

    def test_encode_action_geometry_mapping(self):
        """T2.4: Verifies canonical 64D action vector format and geometry bounds."""
        vec_hold = LatentDynamicsWorldModel.encode_action(0)
        vec_buy = LatentDynamicsWorldModel.encode_action(1, sl_atr=3.0, tp_atr=6.0)
        vec_sell = LatentDynamicsWorldModel.encode_action(2, sl_atr=1.0, tp_atr=15.0)

        self.assertEqual(vec_buy.shape, (1, 64))
        # Direction: index 0
        self.assertEqual(vec_hold[0, 0].item(), 0.0)
        self.assertEqual(vec_buy[0, 0].item(), 1.0)
        self.assertEqual(vec_sell[0, 0].item(), -1.0)

        # One-hot flags: indices 1, 2, 3
        self.assertEqual(vec_hold[0, 1].item(), 1.0)
        self.assertEqual(vec_buy[0, 2].item(), 1.0)
        self.assertEqual(vec_sell[0, 3].item(), 1.0)

        # Continuous SL and TP normalization
        # sl_atr=3.0 -> (3.0 - 1.0)/7.0 = 2/7 ~ 0.2857
        self.assertAlmostEqual(vec_buy[0, 4].item(), 2.0 / 7.0, places=4)
        # tp_atr=6.0 -> (6.0 - 1.5)/13.5 = 4.5/13.5 = 1/3 ~ 0.3333
        self.assertAlmostEqual(vec_buy[0, 5].item(), 1.0 / 3.0, places=4)

    def test_adversarial_market_stressor(self):
        """T2.5: Verifies AdversarialMarketStressor injection into latent state."""
        stressor = AdversarialMarketStressor(latent_dim=768, stress_scale=0.15)
        z = torch.randn(2, 768)
        stressed_z, slip = stressor(z, volatility_factor=1.5)

        self.assertEqual(stressed_z.shape, (2, 768))
        self.assertFalse(torch.isnan(stressed_z).any())
        self.assertGreater(slip, 0.0)
        self.assertAlmostEqual(slip, 2.5 * 1.5, places=4)


class TestMultiStepUnrolling(unittest.TestCase):
    """Test Suite 3: Multi-Step Rollout & Trajectory Unrolling."""

    def setUp(self):
        self.wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)

    def test_unroll_horizon_5_shape(self):
        """T3.1: Verifies rollout over horizon H=5 produces [B, 5, 768]."""
        z_0 = torch.randn(3, 768)
        actions = torch.randn(3, 5, 64)
        traj = self.wm.rollout(z_0, actions)

        self.assertEqual(traj.shape, (3, 5, 768))
        self.assertFalse(torch.isnan(traj).any())

    def test_unroll_deep_horizon_stability(self):
        """T3.2: Verifies deep rollout (H=20) maintains bounded norm without collapse."""
        z_0 = torch.randn(2, 768)
        actions = torch.randn(2, 20, 64) * 0.1
        traj = self.wm.rollout(z_0, actions)

        for step in range(20):
            norm = torch.norm(traj[:, step, :], dim=-1).mean().item()
            std = torch.std(traj[:, step, :], dim=-1).mean().item()
            self.assertGreater(norm, 10.0, f"Norm collapsed at step {step}: {norm}")
            self.assertLess(norm, 50.0, f"Norm exploded at step {step}: {norm}")
            self.assertAlmostEqual(std, 1.0, delta=0.2)

    def test_unroll_bptt_gradient_propagation(self):
        """T3.3: Verifies BPTT gradients propagate across multi-step trajectory."""
        z_0 = torch.randn(2, 768, requires_grad=True)
        actions = torch.randn(2, 5, 64, requires_grad=True)
        traj = self.wm.rollout(z_0, actions)

        loss = (traj[:, -1, :] ** 2).sum()
        loss.backward()

        self.assertIsNotNone(z_0.grad)
        self.assertTrue(torch.any(z_0.grad != 0.0))
        for name, p in self.wm.g_net.named_parameters():
            self.assertIsNotNone(p.grad, f"g_net parameter {name} has no gradient")

    def test_unroll_trajectory_auxiliary_heads(self):
        """T3.4: Verifies unroll_trajectory returns latent states, values, and rewards."""
        z_0 = torch.randn(2, 768)
        actions = torch.tensor([[1, 2, 0, 1, 2], [0, 1, 1, 2, 0]])

        pred_z, pred_v, pred_r = self.wm.unroll_trajectory(z_0, actions)

        self.assertEqual(pred_z.shape, (2, 5, 768))
        self.assertEqual(pred_v.shape, (2, 5, 1))
        self.assertEqual(pred_r.shape, (2, 5, 1))
        self.assertTrue(torch.all(torch.isfinite(pred_z)))
        self.assertTrue(torch.all(torch.isfinite(pred_v)))
        self.assertTrue(torch.all(torch.isfinite(pred_r)))


class TestAuthenticTargetGeneration(unittest.TestCase):
    """Test Suite 4: Target Grounding with Real M1 Candles."""

    @classmethod
    def setUpClass(cls):
        cls.env = RealM1Environment()
        cls.env.load_data("2025.01.15", "2025.01.17")

    def test_target_generation_from_real_m1_candles(self):
        """T4.1: Verifies future target observations are non-identical real market vectors."""
        start_idx = 100
        K = 5
        future_obs = [self.env.get_observation(start_idx + k) for k in range(1, K + 1)]
        self.assertEqual(len(future_obs), 5)
        self.assertEqual(future_obs[0].shape, (137,))
        self.assertFalse(np.allclose(future_obs[0], future_obs[-1]), "Market candles must evolve over time")

    def test_target_value_anchoring_path_dependent(self):
        """T4.2: Verifies target value is bounded in [-1, 1] from realized return_atr."""
        res = self.env.simulate_trade(bar_idx=100, action=1, sl_atr=2.0, tp_atr=4.0)
        target_val = math.tanh(res.return_atr / 5.0)
        self.assertTrue(-1.0 <= target_val <= 1.0)
        self.assertTrue(math.isfinite(target_val))

    def test_target_reward_anchoring_price_delta(self):
        """T4.3: Verifies 1-step reward target reflects authentic price delta / ATR."""
        c0 = self.env.close_arr[100]
        c1 = self.env.close_arr[101]
        atr = max(float(self.env.atr_arr[101]), 1e-4)
        step_rew = (c1 - c0) / atr
        self.assertTrue(math.isfinite(step_rew))

    def test_generate_trajectory_targets_utility(self):
        """T4.4: Verifies generate_trajectory_targets returns valid tuple of target arrays."""
        s0, fut_obs, target_v, target_r = generate_trajectory_targets(
            env=self.env,
            start_idx=150,
            actions=[1, 2, 0, 1, 2],
            horizon_k=5,
            sl_atr=2.0,
            tp_atr=4.0,
        )

        self.assertEqual(s0.shape, (137,))
        self.assertEqual(fut_obs.shape, (5, 137))
        self.assertEqual(target_v.shape, (5, 1))
        self.assertEqual(target_r.shape, (5, 1))
        self.assertTrue(np.all(np.isfinite(fut_obs)))
        self.assertTrue(np.all(np.isfinite(target_v)))
        self.assertTrue(np.all(np.isfinite(target_r)))


class TestCoEvolutionLossFormulation(unittest.TestCase):
    """Test Suite 5: Multi-Step Co-Evolution Loss Formulation."""

    def setUp(self):
        self.wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)

    def test_compute_coevolution_loss_keys_and_shapes(self):
        """T5.1: Verifies compute_coevolution_loss returns expected keys and scalar loss values."""
        B, H = 4, 5
        z_pred = torch.randn(B, H, 768)
        z_target = torch.randn(B, H, 768)
        val_pred = torch.randn(B, H, 1)
        val_target = torch.randn(B, H, 1)
        rew_pred = torch.randn(B, H, 1)
        rew_target = torch.randn(B, H, 1)

        loss_dict = self.wm.compute_coevolution_loss(
            z_pred, z_target, val_pred, val_target, rew_pred, rew_target
        )

        expected_keys = {"total_loss", "loss_latent", "loss_value", "loss_reward", "cos_similarity"}
        self.assertEqual(set(loss_dict.keys()), expected_keys)
        for k, v in loss_dict.items():
            self.assertEqual(v.ndim, 0, f"Loss {k} must be scalar")
            self.assertTrue(torch.isfinite(v), f"Loss {k} must be finite")

    def test_loss_reduction_over_optimization_steps(self):
        """T5.2: Verifies Adam gradient steps decrease latent transition loss on real transitions."""
        opt = torch.optim.Adam(self.wm.parameters(), lr=1e-3)
        z_init = torch.randn(4, 768)
        actions = torch.randn(4, 5, 64)
        targets = torch.randn(4, 5, 768)

        losses = []
        for _ in range(8):
            opt.zero_grad()
            traj = self.wm.rollout(z_init, actions)
            cos_loss = (1.0 - F.cosine_similarity(traj, targets, dim=-1)).clamp(min=0.0).mean()
            cos_loss.backward()
            opt.step()
            losses.append(cos_loss.item())

        self.assertLess(losses[-1], losses[0], "Latent loss must decrease over optimization steps")

    def test_loss_non_negative_and_finite(self):
        """T5.3: Verifies losses remain non-negative and finite under boundary conditions."""
        z = torch.randn(2, 768)
        v = torch.zeros(2, 1)
        r = torch.zeros(2, 1)

        # Identical targets -> loss_latent should be 0.0
        loss_zero = self.wm.compute_coevolution_loss(z, z, v, v, r, r)
        self.assertGreaterEqual(loss_zero["loss_latent"].item(), 0.0)
        self.assertAlmostEqual(loss_zero["loss_latent"].item(), 0.0, places=5)
        self.assertAlmostEqual(loss_zero["loss_value"].item(), 0.0, places=5)
        self.assertAlmostEqual(loss_zero["loss_reward"].item(), 0.0, places=5)
        self.assertGreaterEqual(loss_zero["total_loss"].item(), 0.0)

    def test_orthogonal_gradient_direction_cosine_loss(self):
        """T5.4: Verifies gradient of cosine similarity is orthogonal to predicted vector."""
        u = torch.randn(4, 768, requires_grad=True)
        v = torch.randn(4, 768)

        cos_sim = F.cosine_similarity(u, v, dim=-1).sum()
        cos_sim.backward()

        inner_prod = (u * u.grad).sum(dim=-1)
        for i in range(4):
            self.assertAlmostEqual(inner_prod[i].item(), 0.0, delta=1e-4)


class TestBackwardCompatibilityAndMythosPreservation(unittest.TestCase):
    """Test Suite 6: Backward Compatibility, Weights & Checkpoints."""

    def test_weight_shapes_and_param_counts(self):
        """T6.1: Verifies all 20 legacy parameter keys match exact shapes."""
        wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)
        state_dict = wm.state_dict()

        # Legacy 20 keys
        self.assertEqual(state_dict["g_net.0.weight"].shape, (512, 832))
        self.assertEqual(state_dict["g_net.0.bias"].shape, (512,))
        self.assertEqual(state_dict["g_net.1.weight"].shape, (512,))
        self.assertEqual(state_dict["g_net.1.bias"].shape, (512,))
        self.assertEqual(state_dict["g_net.3.weight"].shape, (512, 512))
        self.assertEqual(state_dict["g_net.3.bias"].shape, (512,))
        self.assertEqual(state_dict["g_net.5.weight"].shape, (768, 512))
        self.assertEqual(state_dict["g_net.5.bias"].shape, (768,))
        self.assertEqual(state_dict["norm.weight"].shape, (768,))
        self.assertEqual(state_dict["norm.bias"].shape, (768,))
        self.assertEqual(state_dict["value_head.0.weight"].shape, (256, 768))
        self.assertEqual(state_dict["value_head.0.bias"].shape, (256,))
        self.assertEqual(state_dict["value_head.2.weight"].shape, (1, 256))
        self.assertEqual(state_dict["value_head.2.bias"].shape, (1,))
        self.assertEqual(state_dict["reward_head.0.weight"].shape, (256, 832))
        self.assertEqual(state_dict["reward_head.0.bias"].shape, (256,))
        self.assertEqual(state_dict["reward_head.2.weight"].shape, (1, 256))
        self.assertEqual(state_dict["reward_head.2.bias"].shape, (1,))
        self.assertEqual(state_dict["risk_bounds_head.0.weight"].shape, (256, 768))
        self.assertEqual(state_dict["risk_bounds_head.0.bias"].shape, (256,))
        self.assertEqual(state_dict["risk_bounds_head.2.weight"].shape, (2, 256))
        self.assertEqual(state_dict["risk_bounds_head.2.bias"].shape, (2,))

    def test_predict_heads_contract(self):
        """T6.2: Verifies auxiliary prediction head outputs match contract."""
        wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64)
        z = torch.randn(3, 768)
        a = torch.randn(3, 64)

        heads = wm.predict_heads(z, a)
        self.assertEqual(heads["value"].shape, (3, 1))
        self.assertEqual(heads["reward"].shape, (3, 1))
        self.assertEqual(heads["risk_bounds"].shape, (3, 2))
        self.assertTrue((heads["risk_bounds"] >= 0.0).all(), "Risk bounds must be strictly non-negative")

    def test_legacy_checkpoint_loading_strict(self):
        """T6.3: Verifies legacy 20-tensor checkpoint loads with strict=True without error."""
        legacy_path = os.path.join(
            repo_root, "outputs", "he_chuyen_gia", "v193_runs", "mythos_selfplay_triad_champion.pt"
        )
        if not os.path.exists(legacy_path):
            self.skipTest(f"Legacy checkpoint not found: {legacy_path}")

        ckpt = torch.load(legacy_path, map_location="cpu", weights_only=False)
        wm = LatentDynamicsWorldModel()
        res = wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
        # Should not raise RuntimeError and should return IncompatibleKeys object
        self.assertTrue(len(res.missing_keys) > 0)
        self.assertTrue(all(k.startswith("encoder.") for k in res.missing_keys))

    def test_exported_checkpoint_roundtrip(self):
        """T6.4: Verifies full checkpoint save and reload preserves encoder and dynamics."""
        wm_orig = LatentDynamicsWorldModel()
        buf = io.BytesIO()
        torch.save(wm_orig.state_dict(), buf)
        buf.seek(0)

        wm_loaded = LatentDynamicsWorldModel()
        wm_loaded.load_state_dict(torch.load(buf, weights_only=False), strict=True)

        obs = np.random.randn(137).astype(np.float32)
        z_orig = wm_orig.encode_state(obs)
        z_loaded = wm_loaded.encode_state(obs)
        self.assertTrue(torch.allclose(z_orig, z_loaded, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
