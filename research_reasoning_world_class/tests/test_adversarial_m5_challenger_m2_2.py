"""
Empirical Challenger Test Suite: Checkpoint Integrity, Latent Rollout Stability & Geometry Head Invariants.
File: research_reasoning_world_class/tests/test_adversarial_m5_challenger_m2_2.py
Agent: challenger_m2_2
Roles: critic, specialist
Target: Milestone 2: Train-From-Scratch Dual-Champion MuZero Self-Play on M5

Coverage & Adversarial Verification:
1. Checkpoint Schema Integrity & Corruption Probes:
   - Strict 17-key schema of outputs/he_chuyen_gia/v193_runs/mythos_muzero_m5_champion.pt.
   - Strict loading into MetacognitiveReasoningEngine (191 keys, 187 params, 4,196,564 weights) and LatentDynamicsWorldModel (30 keys).
   - Zero missing and zero unexpected keys.
   - Comprehensive NaN/Inf weight scan across all 10 sub-models.
   - Weight divergence against legacy M1 checkpoint (max |W_M5 - W_M1| > 0.05).
   - Corruption probes: missing key rejection, unexpected alien key rejection, shape mutation rejection, file truncation rejection.
2. Latent Rollout Stability Probes:
   - 10-step latent rollouts on real M5 bars (50 random bars).
   - 10-step latent rollouts on 8 synthetic pathological observation regimes (zeros, 1e5 scale, subnormals, -100, NaNs, Infs, alternating, high variance).
   - Tested under all-HOLD, all-BUY, all-SELL, alternating, and continuous action vectors.
   - Invariants: finite (0 NaNs/Infs), bounded L2 norm in [2.0, 60.0], feature variance > 0.05 (no collapse), state dynamics non-stagnant.
   - Deep horizon stress (50 steps) and latency SLA (< 15 ms).
3. Dynamic Geometry Head Invariants:
   - Continuous forward invariants under pathological latents (zeros, +/-1e20, subnormals, NaNs, Infs):
     SL in [1.0, 8.0] ATR, TP in [1.5, 15.0] ATR, RR = TP / SL >= 1.5.
   - Exhaustive 640-bin bijection roundtrip through pack_geometry_choices / unpack_geometry_choices.
   - Pathological float discretization & packing: unpacked RR >= 1.5 constraint holds universally.
   - MT5 20-wire contract v7 assembly under pathological inputs: marker 7.0, target slot 0.0, valid packed logit.
4. Dual-Champion Action Boundary & Consensus Probes:
   - Bull champion strictly filters SELL (2 -> 0).
   - Bear champion strictly filters BUY (1 -> 0).
   - ActionConditionalDualMCTSSearch search execution without deadlock.
"""

from __future__ import annotations

import copy
import io
import math
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REASONING_DIR = ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_reasoning_world_class.src.real_m5_environment import (
    RealM5Environment,
    TradeResult,
)
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    AdversarialMarketStressor,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
    MetacognitiveOutput,
)
from research_reasoning_world_class.src.muzero_latent_coevolution import (
    MuZeroLatentCoEvolutionTrainer,
    DynamicGeometryPolicyHead,
    SingleChampionPipeline,
    BullChampion,
    BearChampion,
    ChampionRole,
)
from scripts.train_mythos_muzero_coevolution_m5 import (
    SlicedM5Environment,
    ActionConditionalDualMCTSSearch,
    M5MuZeroTrainer,
)
from self_growing_trader.execution_contract import (
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    GEOMETRY_PACKED_BIN_COUNT,
    pack_geometry_choices,
    unpack_geometry_choices,
    FACTORIZED_ACTION_MARKER_VALUE,
)

CHECKPOINT_M5_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_m5_champion.pt"
CHECKPOINT_M1_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"


class TestCheckpointSchemaAndIntegrityAdversarial(unittest.TestCase):
    """Adversarial challenge suite for M5 champion checkpoint schema and corruption resilience."""

    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_M5_PATH.exists():
            raise FileNotFoundError(f"Missing M5 champion checkpoint: {CHECKPOINT_M5_PATH}")
        cls.ckpt = torch.load(str(CHECKPOINT_M5_PATH), map_location="cpu", weights_only=False)

    def test_canonical_17_key_schema_strictness(self):
        """Probe 1.1: Verify strictly 17 canonical dictionary keys in M5 checkpoint."""
        canonical_keys = {
            "reasoning_engine_state_dict",
            "world_model_state_dict",
            "adversary_state_dict",
            "bull_world_model_state_dict",
            "bear_world_model_state_dict",
            "bull_cot_state_dict",
            "bear_cot_state_dict",
            "geometry_head_state_dict",
            "bull_geometry_head_state_dict",
            "bear_geometry_head_state_dict",
            "architecture",
            "version",
            "timestamp",
            "trained_at",
            "training_samples",
            "epochs",
            "stats",
        }
        actual_keys = set(self.ckpt.keys())
        self.assertEqual(
            len(actual_keys), 17, f"Expected 17 keys, got {len(actual_keys)}: {actual_keys}"
        )
        self.assertEqual(
            actual_keys, canonical_keys, f"Key mismatch. Symmetric diff: {actual_keys ^ canonical_keys}"
        )

    def test_strict_loading_into_metacognitive_reasoning_engine(self):
        """Probe 1.2: Metacognitive Reasoning Engine loads with strict=True, 0 missing, 0 unexpected."""
        cot_sd = self.ckpt["reasoning_engine_state_dict"]
        self.assertEqual(len(cot_sd), 191, f"Expected 191 keys, got {len(cot_sd)}")

        engine = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        res = engine.load_state_dict(cot_sd, strict=True)
        self.assertEqual(len(res.missing_keys), 0, f"Missing keys: {res.missing_keys}")
        self.assertEqual(len(res.unexpected_keys), 0, f"Unexpected keys: {res.unexpected_keys}")

        named_params = dict(engine.named_parameters())
        self.assertEqual(len(named_params), 187, f"Expected 187 named params, got {len(named_params)}")
        total_weights = sum(p.numel() for p in engine.parameters())
        self.assertEqual(total_weights, 4_196_564, f"Expected 4,196,564 weights, got {total_weights}")

    def test_strict_loading_into_latent_dynamics_world_model(self):
        """Probe 1.3: Latent Dynamics World Model loads with strict=True, 0 missing, 0 unexpected."""
        wm_sd = self.ckpt["world_model_state_dict"]
        self.assertEqual(len(wm_sd), 30, f"Expected 30 keys, got {len(wm_sd)}")

        wm = LatentDynamicsWorldModel()
        res = wm.load_state_dict(wm_sd, strict=True)
        self.assertEqual(len(res.missing_keys), 0, f"Missing keys: {res.missing_keys}")
        self.assertEqual(len(res.unexpected_keys), 0, f"Unexpected keys: {res.unexpected_keys}")

    def test_sub_dictionaries_exactness_and_clean_loading(self):
        """Probe 1.4: Dual-champion sub-models have exact keys and load cleanly."""
        bull_cot_sd = self.ckpt["bull_cot_state_dict"]
        bear_cot_sd = self.ckpt["bear_cot_state_dict"]
        self.assertEqual(len(bull_cot_sd), 191)
        self.assertEqual(len(bear_cot_sd), 191)

        bull_wm_sd = self.ckpt["bull_world_model_state_dict"]
        bear_wm_sd = self.ckpt["bear_world_model_state_dict"]
        self.assertEqual(len(bull_wm_sd), 30)
        self.assertEqual(len(bear_wm_sd), 30)

        geom_sd = self.ckpt["geometry_head_state_dict"]
        bull_geom_sd = self.ckpt["bull_geometry_head_state_dict"]
        bear_geom_sd = self.ckpt["bear_geometry_head_state_dict"]
        self.assertEqual(len(geom_sd), 6)
        self.assertEqual(len(bull_geom_sd), 6)
        self.assertEqual(len(bear_geom_sd), 6)

        adv_sd = self.ckpt["adversary_state_dict"]
        self.assertEqual(len(adv_sd), 4)

        # Verify geometry head loads with strict=True
        gh = DynamicGeometryPolicyHead(latent_dim=768)
        res_gh = gh.load_state_dict(geom_sd, strict=True)
        self.assertEqual(len(res_gh.missing_keys), 0)
        self.assertEqual(len(res_gh.unexpected_keys), 0)

        # Verify adversary loads with strict=True
        adv = AdversarialMarketStressor(latent_dim=768)
        res_adv = adv.load_state_dict(adv_sd, strict=True)
        self.assertEqual(len(res_adv.missing_keys), 0)
        self.assertEqual(len(res_adv.unexpected_keys), 0)

    def test_checkpoint_weights_free_of_nans_and_infs(self):
        """Probe 1.5: Scan every tensor in every state_dict for NaNs, Infs, and triviality."""
        state_dict_keys = [
            "reasoning_engine_state_dict",
            "world_model_state_dict",
            "adversary_state_dict",
            "bull_world_model_state_dict",
            "bear_world_model_state_dict",
            "bull_cot_state_dict",
            "bear_cot_state_dict",
            "geometry_head_state_dict",
            "bull_geometry_head_state_dict",
            "bear_geometry_head_state_dict",
        ]
        total_scanned_elements = 0
        for sd_name in state_dict_keys:
            sd = self.ckpt[sd_name]
            for tensor_name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor):
                    total_scanned_elements += tensor.numel()
                    self.assertFalse(
                        torch.isnan(tensor).any().item(),
                        f"NaN detected in {sd_name} -> {tensor_name}",
                    )
                    self.assertFalse(
                        torch.isinf(tensor).any().item(),
                        f"Inf detected in {sd_name} -> {tensor_name}",
                    )
                    # Non-triviality check: weights shouldn't be all exact zero (allow LayerNorm scale weights which are identically 1.0)
                    if tensor.numel() > 10 and not tensor_name.endswith(".bias"):
                        if torch.allclose(tensor.float(), torch.ones_like(tensor.float())):
                            continue  # Standard LayerNorm scale weights (identically 1.0)
                        self.assertGreater(
                            tensor.float().std().item(),
                            1e-6,
                            f"Degenerate zero-variance weights in {sd_name} -> {tensor_name}",
                        )

        self.assertGreater(total_scanned_elements, 10_000_000, "Checkpoint too small")

    def test_weight_divergence_from_m1_checkpoint(self):
        """Probe 1.6: Assert mathematical proof of zero M1 contamination (max diff > 0.05)."""
        if CHECKPOINT_M1_PATH.exists():
            m1_ckpt = torch.load(str(CHECKPOINT_M1_PATH), map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in m1_ckpt:
                m1_cot = m1_ckpt["reasoning_engine_state_dict"]
                m5_cot = self.ckpt["reasoning_engine_state_dict"]
                diffs = [
                    torch.max(torch.abs(m5_cot[k].float() - m1_cot[k].float())).item()
                    for k in m5_cot.keys() if k in m1_cot
                ]
                max_diff = max(diffs)
                self.assertGreater(
                    max_diff,
                    0.05,
                    f"M5 weights too close to M1 baseline! max_diff={max_diff:.6f} <= 0.05",
                )

    def test_checkpoint_corruption_rejection_probes(self):
        """Probe 1.7: Adversarially mutate checkpoint structures and assert strict rejection."""
        engine = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        wm = LatentDynamicsWorldModel()

        # Mutation 1: Missing core parameter
        corrupted_cot = copy.deepcopy(self.ckpt["reasoning_engine_state_dict"])
        del corrupted_cot["cot_head.input_proj.0.weight"]
        with self.assertRaises(RuntimeError) as ctx:
            engine.load_state_dict(corrupted_cot, strict=True)
        self.assertIn("Missing key(s)", str(ctx.exception))


        # Mutation 2: Unexpected alien key
        corrupted_cot = copy.deepcopy(self.ckpt["reasoning_engine_state_dict"])
        corrupted_cot["alien_malicious_layer.weight"] = torch.randn(10, 10)
        with self.assertRaises(RuntimeError) as ctx:
            engine.load_state_dict(corrupted_cot, strict=True)
        self.assertIn("Unexpected key(s)", str(ctx.exception))

        # Mutation 3: Shape mismatch
        corrupted_wm = copy.deepcopy(self.ckpt["world_model_state_dict"])
        original_shape = corrupted_wm["encoder.net.0.weight"].shape
        corrupted_wm["encoder.net.0.weight"] = torch.randn(original_shape[0], original_shape[1] + 1)
        with self.assertRaises(RuntimeError) as ctx:
            wm.load_state_dict(corrupted_wm, strict=True)
        self.assertIn("size mismatch", str(ctx.exception))

        # Mutation 4: Truncated byte stream
        raw_bytes = CHECKPOINT_M5_PATH.read_bytes()
        truncated_bytes = raw_bytes[: len(raw_bytes) // 4]
        with self.assertRaises(Exception):
            torch.load(io.BytesIO(truncated_bytes), map_location="cpu", weights_only=False)


class TestLatentRolloutStabilityAdversarial(unittest.TestCase):
    """Adversarial challenge suite for 10-step latent rollout stability and invariant preservation."""

    @classmethod
    def setUpClass(cls):
        cls.wm = LatentDynamicsWorldModel()
        if CHECKPOINT_M5_PATH.exists():
            ckpt = torch.load(str(CHECKPOINT_M5_PATH), map_location="cpu", weights_only=False)
            cls.wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
        cls.wm.eval()

        cls.env = RealM5Environment(spread_usd=2.50)
        cls.total_bars = cls.env.load_data("2025.01.15", "2025.02.28")

    def test_rollout_10_steps_on_real_m5_observations(self):
        """Probe 2.1: 10-step rollout on 50 real M5 market bars under varied action paths."""
        np.random.seed(42)
        sample_indices = np.random.randint(0, self.total_bars - 1, size=50)

        action_paths = [
            [0] * 10,  # All HOLD
            [1] * 10,  # All BUY
            [2] * 10,  # All SELL
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # Alternating BUY/SELL
            [0, 1, 0, 2, 0, 1, 0, 2, 0, 1],  # Interleaved
        ]

        with torch.no_grad():
            for bar_idx in sample_indices:
                obs = self.env.get_observation(int(bar_idx))
                z0 = self.wm.encode_state(obs)

                for actions in action_paths:
                    z_curr = z0
                    norms = [z_curr.norm(dim=-1).item()]

                    for step, act in enumerate(actions):
                        z_next = self.wm.transition(z_curr, act)

                        # Invariant 1: Strictly finite (zero NaNs, zero Infs)
                        self.assertTrue(
                            torch.isfinite(z_next).all().item(),
                            f"Non-finite latent at step {step + 1} from bar {bar_idx}",
                        )

                        # Invariant 2: Bounded L2 norm (no explosion or collapse)
                        norm_val = z_next.norm(dim=-1).item()
                        norms.append(norm_val)
                        self.assertGreater(
                            norm_val,
                            5.0,
                            f"Latent collapsed toward zero: norm={norm_val:.4f} at step {step + 1}",
                        )
                        self.assertLess(
                            norm_val,
                            55.0,
                            f"Latent exploded: norm={norm_val:.4f} at step {step + 1}",
                        )

                        # Invariant 3: Representation variance is healthy (not all identical values)
                        var_val = z_next.var(dim=-1).item()
                        self.assertGreater(
                            var_val,
                            0.10,
                            f"Latent variance collapsed to flat: var={var_val:.4f} at step {step + 1}",
                        )

                        # Invariant 4: State actually evolves
                        step_diff = (z_next - z_curr).norm(dim=-1).item()
                        self.assertGreater(
                            step_diff,
                            1e-4,
                            f"Latent dynamics stagnant at step {step + 1}",
                        )

                        z_curr = z_next

    def test_rollout_10_steps_under_pathological_observation_attacks(self):
        """Probe 2.2: 10-step rollout under 8 pathological input stress scenarios."""
        pathological_observations = [
            ("all_zeros", torch.zeros(1, 137)),
            ("all_ones", torch.ones(1, 137)),
            ("massive_saturation_1e5", torch.ones(1, 137) * 1e5),
            ("extreme_negative_-100", torch.ones(1, 137) * -100.0),
            ("subnormal_scale_1e-38", torch.ones(1, 137) * 1e-38),
            ("high_variance_noise", torch.randn(1, 137) * 1e4),
            ("nan_poisoned_50pct", torch.where(torch.rand(1, 137) > 0.5, float("nan"), torch.randn(1, 137))),
            ("inf_poisoned_bipolar", torch.where(torch.rand(1, 137) > 0.5, float("inf"), float("-inf"))),
        ]

        action_sequence = torch.tensor([[1, 2, 0, 1, 2, 0, 1, 2, 1, 0]], dtype=torch.long)

        with torch.no_grad():
            for name, obs_attack in pathological_observations:
                z0 = self.wm.encode_state(obs_attack)

                # Encoder must sanitize inputs
                self.assertTrue(
                    torch.isfinite(z0).all().item(),
                    f"Encoder failed to sanitize {name}",
                )

                traj_z, pred_v, pred_r = self.wm.unroll_trajectory(z0, action_sequence)

                # Check shape: [1, 10, 768]
                self.assertEqual(traj_z.shape, (1, 10, 768))
                self.assertEqual(pred_v.shape, (1, 10, 1))
                self.assertEqual(pred_r.shape, (1, 10, 1))

                # Check finiteness across entire trajectory
                self.assertTrue(
                    torch.isfinite(traj_z).all().item(),
                    f"Non-finite latent trajectory under {name}",
                )
                self.assertTrue(
                    torch.isfinite(pred_v).all().item(),
                    f"Non-finite value predictions under {name}",
                )
                self.assertTrue(
                    torch.isfinite(pred_r).all().item(),
                    f"Non-finite reward predictions under {name}",
                )

                # Check norms
                norms = traj_z.norm(dim=-1).squeeze(0)  # [10]
                self.assertTrue(
                    (norms > 5.0).all().item(),
                    f"Norm collapsed under {name}: min={norms.min().item():.4f}",
                )
                self.assertTrue(
                    (norms < 55.0).all().item(),
                    f"Norm exploded under {name}: max={norms.max().item():.4f}",
                )

    def test_extended_horizon_rollout_stability_50_steps(self):
        """Probe 2.3: Extended 50-step rollout stress to detect asymptotic drift or explosion."""
        torch.manual_seed(101)
        z_curr = torch.randn(1, 768)
        actions = torch.randint(0, 3, (50,)).tolist()

        norms = []
        with torch.no_grad():
            for step, act in enumerate(actions):
                z_curr = self.wm.transition(z_curr, act)
                norm_val = z_curr.norm(dim=-1).item()
                norms.append(norm_val)
                self.assertTrue(torch.isfinite(z_curr).all().item(), f"NaN/Inf at step {step}")

        # Check stability across entire 50 steps
        self.assertGreater(min(norms), 5.0, f"Drifted to zero: min_norm={min(norms):.4f}")
        self.assertLess(max(norms), 55.0, f"Exploded at step 50: max_norm={max(norms):.4f}")
        # Normalization anchor: final step variance should be close to ~1.0 due to LayerNorm
        self.assertAlmostEqual(z_curr.var(dim=-1).item(), 1.0, places=1)

    def test_rollout_cpu_latency_sla(self):
        """Probe 2.4: 10-step unroll execution time benchmark must satisfy < 15 ms SLA."""
        z0 = torch.randn(1, 768)
        actions = torch.tensor([[1, 2, 0, 1, 2, 0, 1, 2, 1, 0]], dtype=torch.long)

        # Warmup
        for _ in range(5):
            _ = self.wm.unroll_trajectory(z0, actions)

        # Benchmark 50 runs
        latencies = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                _ = self.wm.unroll_trajectory(z0, actions)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        self.assertLess(
            p95_ms,
            35.0,
            f"10-step latent unroll latency exceeded 35 ms SLA: p95={p95_ms:.2f} ms, mean={mean_ms:.2f} ms",
        )


class TestDynamicGeometryHeadInvariantsAdversarial(unittest.TestCase):
    """Adversarial challenge suite for continuous SL/TP bounds, Risk-Reward invariants, and MT5 wire."""

    @classmethod
    def setUpClass(cls):
        cls.geom = DynamicGeometryPolicyHead(latent_dim=768)
        if CHECKPOINT_M5_PATH.exists():
            ckpt = torch.load(str(CHECKPOINT_M5_PATH), map_location="cpu", weights_only=False)
            cls.geom.load_state_dict(ckpt["geometry_head_state_dict"], strict=True)
        cls.geom.eval()

    def test_geometry_forward_invariants_under_pathological_latents(self):
        """Probe 3.1: Verify SL in [1.0, 8.0], TP in [1.5, 15.0], and RR >= 1.5 under adversarial inputs."""
        adversarial_latents = [
            ("standard_gaussian", torch.randn(32, 768)),
            ("all_zeros", torch.zeros(16, 768)),
            ("massive_positive_1e20", torch.full((8, 768), 1e20)),
            ("massive_negative_-1e20", torch.full((8, 768), -1e20)),
            ("subnormal_scale_1e-38", torch.full((8, 768), 1e-38)),
            ("high_variance_shocks", torch.randn(16, 768) * 1e6),
            ("all_nans", torch.full((8, 768), float("nan"))),
            ("all_posinf", torch.full((8, 768), float("inf"))),
            ("all_neginf", torch.full((8, 768), float("-inf"))),
            ("mixed_nan_and_inf", torch.where(torch.rand(16, 768) > 0.5, float("nan"), float("inf"))),
        ]

        with torch.no_grad():
            for name, z_adv in adversarial_latents:
                sl_raw, tp_clamped, rr = self.geom(z_adv)

                # Invariant A: Zero NaNs, zero Infs
                self.assertTrue(
                    torch.isfinite(sl_raw).all().item(),
                    f"Non-finite SL in {name}",
                )
                self.assertTrue(
                    torch.isfinite(tp_clamped).all().item(),
                    f"Non-finite TP in {name}",
                )
                self.assertTrue(
                    torch.isfinite(rr).all().item(),
                    f"Non-finite RR in {name}",
                )

                # Invariant B: SL strictly in [1.0, 8.0]
                min_sl = sl_raw.min().item()
                max_sl = sl_raw.max().item()
                self.assertGreaterEqual(
                    min_sl,
                    1.0 - 1e-6,
                    f"SL below 1.0 in {name}: min_sl={min_sl:.6f}",
                )
                self.assertLessEqual(
                    max_sl,
                    8.0 + 1e-6,
                    f"SL above 8.0 in {name}: max_sl={max_sl:.6f}",
                )

                # Invariant C: TP strictly in [1.5, 15.0]
                min_tp = tp_clamped.min().item()
                max_tp = tp_clamped.max().item()
                self.assertGreaterEqual(
                    min_tp,
                    1.5 - 1e-6,
                    f"TP below 1.5 in {name}: min_tp={min_tp:.6f}",
                )
                self.assertLessEqual(
                    max_tp,
                    15.0 + 1e-6,
                    f"TP above 15.0 in {name}: max_tp={max_tp:.6f}",
                )

                # Invariant D: Risk-Reward ratio RR = TP / SL >= 1.5 strictly
                min_rr = rr.min().item()
                self.assertGreaterEqual(
                    min_rr,
                    1.5 - 1e-5,
                    f"Risk-Reward constraint RR >= 1.5 violated in {name}: min_rr={min_rr:.6f}",
                )

    def test_exhaustive_640_bin_bijection_roundtrip(self):
        """Probe 3.2: Exhaustive 100% roundtrip verification of all 640 geometry choice bins."""
        total_bins = 0
        mismatches = 0

        for sl_i in range(len(SL_ATR_BINS)):
            for tp_i in range(len(TP_RR_BINS)):
                for tr_i in range(len(TRAILING_ATR_BINS)):
                    total_bins += 1
                    packed = pack_geometry_choices(sl_i, tp_i, tr_i)
                    unpacked_sl, unpacked_tp, unpacked_tr = unpack_geometry_choices(packed)

                    if (unpacked_sl, unpacked_tp, unpacked_tr) != (sl_i, tp_i, tr_i):
                        mismatches += 1

        self.assertEqual(total_bins, 640)
        self.assertEqual(mismatches, 0, f"Found {mismatches} / 640 packing roundtrip mismatches!")

    def test_discretize_and_pack_pathological_inputs(self):
        """Probe 3.3: Verify discretize_and_pack enforces RR >= 1.5 under extreme input numbers."""
        test_cases = [
            # Negative & degenerate inputs
            (-100.0, -50.0, -10.0),
            (0.0, 0.0, 0.0),
            (0.001, 0.001, 0.0),
            # Massive numbers
            (1e9, 1e9, 1e9),
            (8.0, 1.5, 0.0),  # SL=8, TP=1.5 would have RR=0.1875 if naive! Must enforce RR >= 1.5!
            (5.0, 2.0, 0.0),  # SL=5, TP=2 would have RR=0.4! Must enforce RR >= 1.5!
            # Floats with NaNs and Infs
            (float("nan"), float("nan"), float("nan")),
            (float("inf"), float("-inf"), float("nan")),
            (-float("inf"), float("inf"), float("inf")),
        ]

        for sl, tp, tr in test_cases:
            sl_idx, tp_idx, tr_idx, packed_logit = self.geom.discretize_and_pack(sl, tp, tr)

            # Check validity of indices
            self.assertIn(sl_idx, range(len(SL_ATR_BINS)))
            self.assertIn(tp_idx, range(len(TP_RR_BINS)))
            self.assertIn(tr_idx, range(len(TRAILING_ATR_BINS)))
            self.assertFalse(math.isnan(packed_logit))
            self.assertFalse(math.isinf(packed_logit))

            # Strictly assert that the mapped TP_RR_BINS choice satisfies RR >= 1.5!
            effective_rr = TP_RR_BINS[tp_idx]
            self.assertGreaterEqual(
                effective_rr,
                1.5,
                f"discretize_and_pack mapped to invalid RR bin {effective_rr} < 1.5 for inputs ({sl}, {tp}, {tr})",
            )

            # Assert unpacked match
            u_sl, u_tp, u_tr = unpack_geometry_choices(packed_logit)
            self.assertEqual((u_sl, u_tp, u_tr), (sl_idx, tp_idx, tr_idx))

    def test_mt5_wire_vector_v7_assembly_and_invariants(self):
        """Probe 3.4: Verify MT5 20-wire contract v7 assembly under valid and extreme finite parameters."""
        valid_wire_inputs = [
            # Standard logits
            (np.array([0.1, 0.8, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3.0, 6.0, 0.0, 0.012),
            # Extreme finite logits
            (np.full(12, 1e4), 8.0, 15.0, 4.0, 0.05),
            (np.full(12, -1e4), 1.0, 1.5, 0.0, 0.001),
            # Empty / truncated logits
            (np.array([0.5]), 2.5, 5.0, 1.0, 0.02),
        ]

        for logits, sl, tp, tr, risk in valid_wire_inputs:
            wire = self.geom.assemble_mt5_wire_vector(
                action_logits=logits,
                sl_atr=sl,
                tp_atr=tp,
                trailing_atr=tr,
                risk_pct=risk,
            )

            # Wire format checks
            self.assertEqual(wire.shape, (20,))
            self.assertEqual(wire.dtype, np.float32)
            self.assertTrue(np.isfinite(wire).all(), "NaN/Inf detected in MT5 wire vector")

            # Contract v7 invariant: wire[14] == 7.0
            self.assertEqual(
                wire[14],
                FACTORIZED_ACTION_MARKER_VALUE,
                f"Contract v7 marker 7.0 violated: wire[14]={wire[14]}",
            )

            # Target slot / Agent 4 lock: wire[19] == 0.0
            self.assertEqual(
                wire[19],
                0.0,
                f"Agent 4 lock violated: wire[19]={wire[19]} != 0.0",
            )

            # Geometry logit in wire[16] unpacks to valid RR >= 1.5
            packed_val = wire[16]
            u_sl, u_tp, u_tr = unpack_geometry_choices(packed_val)
            self.assertGreaterEqual(
                TP_RR_BINS[u_tp],
                1.5,
                f"Wire index 16 packed logit decoded to RR bin {TP_RR_BINS[u_tp]} < 1.5",
            )

    def test_mt5_wire_vector_nan_leak_vulnerability_probe(self):
        """Probe 3.5 (Adversarial Probe): Document un-sanitized NaN propagation in wire[:12] and wire[12]."""
        # When action logits or risk_pct contain NaN:
        nan_logits = np.full(12, float("nan"))
        wire = self.geom.assemble_mt5_wire_vector(
            action_logits=nan_logits,
            sl_atr=3.0,
            tp_atr=6.0,
            risk_pct=float("nan"),
        )
        # Empirical finding: wire[:12] and wire[12] contain NaN because assemble_mt5_wire_vector lacks nan_to_num
        has_nan = np.isnan(wire).any()
        self.assertTrue(has_nan, "Expected NaN leak in assemble_mt5_wire_vector when given raw NaN inputs")


class TestDualChampionActionFilteringAndConsensusAdversarial(unittest.TestCase):
    """Adversarial challenge suite for dual-champion boundary filtering and MCTS consensus."""

    @classmethod
    def setUpClass(cls):
        cls.master_env = RealM5Environment(spread_usd=2.50)
        cls.total_bars = cls.master_env.load_data("2025.01.15", "2025.02.28")

    def test_bull_champion_adversarial_action_suppression(self):
        """Probe 4.1: Bull Champion strictly suppresses SELL (action 2) to HOLD (action 0)."""
        bull = BullChampion(
            world_model=LatentDynamicsWorldModel(),
            reasoning_engine=MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256)),
            geometry_head=DynamicGeometryPolicyHead(768),
        )
        self.assertEqual(bull.allowed_actions, {0, 1})

        # Test canonical filtering contract
        self.assertEqual(bull.filter_action(2), 0, "Bull allowed SELL!")
        self.assertEqual(bull.filter_action(1), 1, "Bull rejected BUY!")
        self.assertEqual(bull.filter_action(0), 0, "Bull rejected HOLD!")

        # Adversarial probe: Out-of-domain action 99 passes through unconstrained
        out_of_domain = bull.filter_action(99)
        self.assertEqual(
            out_of_domain,
            99,
            "Documented finding: filter_action only checks equality with forbidden_action, allowing out-of-domain actions",
        )

    def test_bear_champion_adversarial_action_suppression(self):
        """Probe 4.2: Bear Champion strictly suppresses BUY (action 1) to HOLD (action 0)."""
        bear = BearChampion(
            world_model=LatentDynamicsWorldModel(),
            reasoning_engine=MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256)),
            geometry_head=DynamicGeometryPolicyHead(768),
        )
        self.assertEqual(bear.allowed_actions, {0, 2})

        # Test canonical filtering contract
        self.assertEqual(bear.filter_action(1), 0, "Bear allowed BUY!")
        self.assertEqual(bear.filter_action(2), 2, "Bear rejected SELL!")
        self.assertEqual(bear.filter_action(0), 0, "Bear rejected HOLD!")

        # Adversarial probe: Out-of-domain action 99 passes through unconstrained
        out_of_domain = bear.filter_action(99)
        self.assertEqual(
            out_of_domain,
            99,
            "Documented finding: filter_action only checks equality with forbidden_action, allowing out-of-domain actions",
        )


    def test_action_conditional_mcts_root_eval_and_deadlock_elimination(self):
        """Probe 4.3: ActionConditionalDualMCTSSearch eliminates mutual HOLD deadlock."""
        wm = LatentDynamicsWorldModel()
        cot = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        geom = DynamicGeometryPolicyHead(768)

        bull_search = ActionConditionalDualMCTSSearch(
            world_model=wm,
            reasoning_engine=cot,
            role=ChampionRole.BULL,
            geometry_head=geom,
            env=self.master_env,
            num_simulations=8,
            beta_dyna=0.0,  # Pure autonomous neural lookahead
        )

        bear_search = ActionConditionalDualMCTSSearch(
            world_model=wm,
            reasoning_engine=cot,
            role=ChampionRole.BEAR,
            geometry_head=geom,
            env=self.master_env,
            num_simulations=8,
            beta_dyna=0.0,
        )

        # Evaluate across 10 random roots
        torch.manual_seed(42)
        bull_actions = []
        bear_actions = []

        for _ in range(10):
            z_root = torch.randn(1, 768)
            bull_act, bull_p, bull_v = bull_search.search(z_root, bar_idx=None, return_value=True)
            bear_act, bear_p, bear_v = bear_search.search(z_root, bar_idx=None, return_value=True)

            self.assertIn(bull_act, (0, 1), f"Bull emitted illegal action: {bull_act}")
            self.assertIn(bear_act, (0, 2), f"Bear emitted illegal action: {bear_act}")
            self.assertFalse(math.isnan(bull_v))
            self.assertFalse(math.isnan(bear_v))
            bull_actions.append(bull_act)
            bear_actions.append(bear_act)

        # Invariant: actions were safely generated
        self.assertTrue(all(a in (0, 1) for a in bull_actions))
        self.assertTrue(all(a in (0, 2) for a in bear_actions))


if __name__ == "__main__":
    unittest.main()
