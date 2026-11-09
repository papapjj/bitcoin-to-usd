"""
Comprehensive Unit & Integration Test Suite for Train-From-Scratch Dual-Champion MuZero Self-Play on M5.
Module: research_reasoning_world_class/tests/test_muzero_m5_training.py
Author: worker_m2_1
Milestone: Milestone 2 (Train-From-Scratch Dual-Champion MuZero Self-Play on M5)

Coverage:
1. Fresh Initialization & Zero M1 Leakage:
   - from_scratch = True skips loading outputs/he_chuyen_gia/v193_runs/mythos_muzero_coevolved_champion.pt
   - max |W_M5 - W_M1| > 0.05 strictly verified
   - Exact model architectures (191 cot keys, 187 params, 4,196,564 weights, 30 wm keys, 4 adv keys, 6 geom keys)
2. M5 Environment & Training Steps:
   - RealM5Environment loading 12,960 bars (2025.01.15 to 2025.02.28)
   - SlicedM5Environment Walk-Forward partitions: C1 Train [0:8064], C1 OOS [8064:12096], C2 Train [2016:10080], C2 OOS [10080:12960]
   - Grounded self-play rollouts and loss convergence logging
   - ActionConditionalDualMCTSSearch and action filtering (Bull only BUY/HOLD, Bear only SELL/HOLD)
3. Canonical Checkpoint Schema (Strict 17 Keys):
   - mythos_muzero_m5_champion.pt has exactly 17 keys
   - reasoning_engine_state_dict has 191 tensor keys, loads under strict=True with 0 missing/unexpected keys
   - world_model_state_dict has 30 keys, loads under strict=True
   - Dual-champion sub-dictionaries (bull_cot, bear_cot, bull_wm, bear_wm)
   - Lineage metadata: architecture="MythosMuZeroDualChampion", timeframe="M5", from_scratch=True
4. Training Log Artifact:
   - outputs/m5_training_log.md existence, multi-task convergence records, and audit proof
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REASONING_DIR = ROOT / "research_reasoning_world_class"

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
    OOSConfigM5,
    OOSEvaluationEngineM5,
    verify_m5_weight_divergence,
)

CHECKPOINT_M5_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_m5_champion.pt"
CHECKPOINT_M1_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
TRAINING_LOG_PATH = ROOT / "outputs" / "m5_training_log.md"


class TestMuZeroM5FreshInitialization(unittest.TestCase):
    """Verifies Train-From-Scratch Guarantee and Zero M1 Contamination."""

    def test_from_scratch_flag_defaults_to_true(self):
        """M5.1: MuZeroLatentCoEvolutionTrainer defaults to from_scratch=True."""
        trainer = MuZeroLatentCoEvolutionTrainer()
        self.assertTrue(trainer.from_scratch)

    def test_fresh_initialization_does_not_load_m1_weights(self):
        """M5.2: from_scratch=True freshly initializes parameters with zero M1 weight contamination."""
        trainer = MuZeroLatentCoEvolutionTrainer(from_scratch=True)

        if CHECKPOINT_M1_PATH.exists():
            m1_ckpt = torch.load(str(CHECKPOINT_M1_PATH), map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in m1_ckpt:
                m1_cot = m1_ckpt["reasoning_engine_state_dict"]
                fresh_cot = trainer.reasoning_engine.state_dict()
                diffs = [
                    torch.max(torch.abs(fresh_cot[k].float() - m1_cot[k].float())).item()
                    for k in fresh_cot.keys() if k in m1_cot
                ]
                max_diff = max(diffs)
                self.assertGreater(max_diff, 0.05, f"Expected weight divergence > 0.05, got {max_diff:.6f}")

    def test_reasoning_engine_architectural_invariants(self):
        """M5.3: Metacognitive Reasoning Engine conforms to 191 keys, 187 parameters, 4,196,564 weights."""
        cot = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        sd = cot.state_dict()
        self.assertEqual(len(sd), 191)
        self.assertEqual(len(list(cot.named_parameters())), 187)
        self.assertEqual(sum(p.numel() for p in cot.parameters()), 4_196_564)

    def test_world_model_30_tensor_keys_exactness(self):
        """M5.4: Latent Dynamics World Model strictly contains 30 tensor keys."""
        wm = LatentDynamicsWorldModel()
        self.assertEqual(len(wm.state_dict()), 30)

    def test_adversary_and_geometry_head_tensor_keys(self):
        """M5.5: Adversary has 4 keys and Geometry Head has 6 keys."""
        adv = AdversarialMarketStressor(latent_dim=768)
        self.assertEqual(len(adv.state_dict()), 4)

        geom = DynamicGeometryPolicyHead(latent_dim=768)
        self.assertEqual(len(geom.state_dict()), 6)


class TestMuZeroM5TrainingStepAndEnvironment(unittest.TestCase):
    """Verifies M5 environment ingestion, Walk-Forward scheduling, and training mechanics."""

    @classmethod
    def setUpClass(cls):
        cls.master_env = RealM5Environment(spread_usd=2.50)
        cls.total_bars = cls.master_env.load_data("2025.01.15", "2025.02.28")

    def test_m5_environment_total_bars_and_observation_matrix(self):
        """M5.6: Environment loads 12,960 bars and produces contiguous 137D observation matrix."""
        self.assertEqual(self.total_bars, 12960)
        self.assertEqual(self.master_env.obs_matrix.shape, (12960, 137))
        self.assertEqual(self.master_env.obs_matrix.dtype, np.float32)
        self.assertFalse(np.isnan(self.master_env.obs_matrix).any())
        self.assertFalse(np.isinf(self.master_env.obs_matrix).any())

    def test_walk_forward_m5_schedule_partitions(self):
        """M5.7: Walk-Forward partitions match exact M5 calendar specifications."""
        c1_train = SlicedM5Environment(self.master_env, 0, 8064, name="C1 Train")
        c1_oos = SlicedM5Environment(self.master_env, 8064, 12096, name="C1 OOS")
        c2_train = SlicedM5Environment(self.master_env, 2016, 10080, name="C2 Train")
        c2_oos = SlicedM5Environment(self.master_env, 10080, 12960, name="C2 OOS")

        self.assertEqual(len(c1_train), 8064)
        self.assertEqual(len(c1_oos), 4032)
        self.assertEqual(len(c2_train), 8064)
        self.assertEqual(len(c2_oos), 2880)

        # Slice edge clamping verification (zero future peeking across edge)
        res = c1_train.simulate_trade(
            bar_idx=len(c1_train) - 2,
            action=1,
            sl_atr=3.0,
            tp_atr=6.0,
            max_holding_bars=12,
        )
        self.assertLessEqual(res.holding_bars, 1)

    def test_dual_champion_action_masking_invariants(self):
        """M5.8: Bull Champion strictly suppresses SELL; Bear Champion strictly suppresses BUY."""
        trainer = M5MuZeroTrainer(from_scratch=True, env=self.master_env)

        bull = trainer.bull_champion
        self.assertEqual(bull.allowed_actions, {0, 1})
        self.assertEqual(bull.filter_action(2), 0)
        self.assertEqual(bull.filter_action(1), 1)
        self.assertEqual(bull.filter_action(0), 0)

        bear = trainer.bear_champion
        self.assertEqual(bear.allowed_actions, {0, 2})
        self.assertEqual(bear.filter_action(1), 0)
        self.assertEqual(bear.filter_action(2), 2)
        self.assertEqual(bear.filter_action(0), 0)

    def test_m5_training_step_executes_and_logs_convergence(self):
        """M5.9: M5 training step executes self-play rollouts and logs multi-task loss metrics."""
        c1_train = SlicedM5Environment(self.master_env, 0, 500, name="Mini C1 Train")
        trainer = M5MuZeroTrainer(from_scratch=True, env=c1_train, mcts_simulations=2)

        results = trainer.run_self_play_mcts(num_rounds=2, log_interval=1, env=c1_train)
        self.assertIn("metrics", results)
        metrics = results["metrics"]
        self.assertIn("total_loss", metrics)
        self.assertIn("loss_dynamics", metrics)
        self.assertIn("cos_similarity", metrics)
        self.assertIn("loss_value", metrics)
        self.assertIn("loss_prm", metrics)

        self.assertEqual(len(metrics["total_loss"]), 2)
        for loss_val in metrics["total_loss"]:
            self.assertFalse(math.isnan(loss_val))
            self.assertFalse(math.isinf(loss_val))

    def test_action_conditional_mcts_search(self):
        """M5.10: ActionConditionalDualMCTSSearch searches root and computes Bellman returns."""
        wm = LatentDynamicsWorldModel()
        cot = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        mcts = ActionConditionalDualMCTSSearch(
            world_model=wm,
            reasoning_engine=cot,
            role=ChampionRole.BULL,
            geometry_head=DynamicGeometryPolicyHead(768),
            env=self.master_env,
            num_simulations=4,
            beta_dyna=0.5,
        )

        z_root = torch.randn(1, 768)
        action, policy, val = mcts.search(z_root, bar_idx=None, return_value=True)
        self.assertIn(action, (0, 1))
        self.assertIn(0, policy)
        self.assertIn(1, policy)
        self.assertFalse(math.isnan(val))


class TestCanonicalM5CheckpointAndTrainingLog(unittest.TestCase):
    """Verifies strict 17-key checkpoint schema, weights, and training log artifact."""

    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_M5_PATH.exists():
            raise FileNotFoundError(f"Canonical M5 checkpoint not found: {CHECKPOINT_M5_PATH}")
        cls.ckpt = torch.load(str(CHECKPOINT_M5_PATH), map_location="cpu", weights_only=False)

    def test_canonical_17_key_schema_exactness(self):
        """M5.11: Checkpoint dictionary strictly contains the 17 canonical schema keys."""
        expected_keys = [
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
        ]
        self.assertEqual(len(self.ckpt), 17)
        for k in expected_keys:
            self.assertIn(k, self.ckpt, f"Missing required canonical key: {k}")

    def test_reasoning_engine_strict_load_and_parameter_count(self):
        """M5.12: reasoning_engine_state_dict has 191 keys and loads with 0 missing/unexpected keys."""
        cot_sd = self.ckpt["reasoning_engine_state_dict"]
        self.assertEqual(len(cot_sd), 191)

        engine = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        res = engine.load_state_dict(cot_sd, strict=True)
        self.assertEqual(len(res.missing_keys), 0)
        self.assertEqual(len(res.unexpected_keys), 0)

        named_params = dict(engine.named_parameters())
        self.assertEqual(len(named_params), 187)
        self.assertEqual(sum(p.numel() for p in engine.parameters()), 4_196_564)

    def test_world_model_30_keys_strict_load(self):
        """M5.13: world_model_state_dict has strictly 30 tensor keys and loads with strict=True."""
        wm_sd = self.ckpt["world_model_state_dict"]
        self.assertEqual(len(wm_sd), 30)

        wm = LatentDynamicsWorldModel()
        res = wm.load_state_dict(wm_sd, strict=True)
        self.assertEqual(len(res.missing_keys), 0)
        self.assertEqual(len(res.unexpected_keys), 0)

    def test_dual_champion_sub_dictionaries_exactness(self):
        """M5.14: Dual-champion sub-dictionaries have exact tensor key counts."""
        self.assertEqual(len(self.ckpt["bull_cot_state_dict"]), 191)
        self.assertEqual(len(self.ckpt["bear_cot_state_dict"]), 191)
        self.assertEqual(len(self.ckpt["bull_world_model_state_dict"]), 30)
        self.assertEqual(len(self.ckpt["bear_world_model_state_dict"]), 30)
        self.assertEqual(len(self.ckpt["geometry_head_state_dict"]), 6)
        self.assertEqual(len(self.ckpt["bull_geometry_head_state_dict"]), 6)
        self.assertEqual(len(self.ckpt["bear_geometry_head_state_dict"]), 6)
        self.assertEqual(len(self.ckpt["adversary_state_dict"]), 4)

    def test_m5_lineage_metadata(self):
        """M5.15: Lineage metadata confirms M5 timeframe, architecture, and from_scratch=True."""
        self.assertEqual(self.ckpt["architecture"], "MythosMuZeroDualChampion")
        self.assertEqual(self.ckpt["version"], "v193-real-m5")
        self.assertEqual(self.ckpt["training_samples"], 12960)
        self.assertEqual(self.ckpt["epochs"], 2)

        stats = self.ckpt["stats"]
        self.assertEqual(stats.get("timeframe"), "M5")
        self.assertTrue(stats.get("from_scratch"))

    def test_weight_divergence_from_m1_checkpoint(self):
        """M5.16: M5 champion parameters strictly diverge from M1 baseline with max |W_M5 - W_M1| > 0.05."""
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
                self.assertGreater(max_diff, 0.05, f"Expected weight divergence > 0.05, got {max_diff:.6f}")

    def test_training_log_artifact_content(self):
        """M5.17: outputs/m5_training_log.md exists and documents World Model convergence and rollouts."""
        self.assertTrue(TRAINING_LOG_PATH.exists())
        content = TRAINING_LOG_PATH.read_text(encoding="utf-8")
        self.assertIn("M5 Dual-Champion MuZero Train-From-Scratch Training Log", content)
        self.assertIn("Zero M1 Weight Contamination Audit", content)
        self.assertIn("Walk-Forward Schedule & Dataset Partitions", content)
        self.assertIn("World Model Multi-Task Loss Convergence", content)
        self.assertIn("Dual-Champion Policy Rollouts", content)
        self.assertIn("Canonical M5 Checkpoint Export Verification", content)


if __name__ == "__main__":
    unittest.main()
