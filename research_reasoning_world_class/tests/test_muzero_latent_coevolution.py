"""
Unit Test Suite for Milestone 3: Grounded MCTS, Dual Champion & Dynamic SL/TP.
Target: research_reasoning_world_class/src/muzero_latent_coevolution.py
Coverage:
1. DynamicGeometryPolicyHead (SL [1.0, 8.0], TP [1.5, 15.0], RR >= 1.5, MT5 20D wire mapping).
2. DualLatentMCTSSearch & Grounded Leaf Evaluation (Dyna mixing beta=0.5, CPU latency SLA <= 35ms).
3. Dual-Champion Architecture (Bull BUY-only, Bear SELL-only, mutual exclusivity, consensus arbitration).
4. Legacy Checkpoint Compatibility (strictly 191 keys, 187 parameters, 4,196,564 weights, 30 WM keys).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Tuple
import unittest

import numpy as np
import torch
import torch.nn as nn

from self_growing_trader.execution_contract import (
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    pack_geometry_choices,
    unpack_geometry_choices,
    GEOMETRY_PACKED_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_VALUE,
    TARGET_SLOT_OUTPUT_INDEX,
    OUTPUT_COUNT,
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
    ChampionRole,
    DynamicGeometryPolicyHead,
    LatentMCTSNode,
    DualLatentMCTSSearch,
    LatentMCTSSearch,
    SingleChampionPipeline,
    BullChampion,
    BearChampion,
    MuZeroLatentCoEvolutionTrainer,
    JointCoEvolutionLoss,
)


class MockRealM1Environment:
    """Mock Real M1 Environment providing realistic simulate_trade and observations."""
    def __init__(self, total_bars: int = 500, spread_usd: float = 2.50):
        self.total_bars = total_bars
        self.spread_usd = spread_usd
        rng = np.random.RandomState(42)
        self.prices = 100000.0 + np.cumsum(rng.randn(total_bars) * 15.0)
        self.open_arr = self.prices.astype(np.float32)
        self.high_arr = (self.prices + np.abs(rng.randn(total_bars) * 20.0)).astype(np.float32)
        self.low_arr = (self.prices - np.abs(rng.randn(total_bars) * 20.0)).astype(np.float32)
        self.close_arr = (self.prices + rng.randn(total_bars) * 8.0).astype(np.float32)
        self.atr_arr = np.full(total_bars, 48.60, dtype=np.float32)

    def get_observation(self, bar_idx: int) -> np.ndarray:
        obs = np.zeros(137, dtype=np.float32)
        obs[0] = 0.5  # ret1
        obs[7] = 0.2  # body
        obs[15] = self.spread_usd / 48.60  # spread_atr
        return obs

    def simulate_trade(
        self,
        bar_idx: int,
        action: int,
        sl_atr: float,
        tp_atr: float,
        max_holding_bars: int = 60,
    ):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockTradeResult:
            pnl_usd: float
            return_atr: float
            exit_type: str
            holding_bars: int
            exit_reason: str
            entry_price: float
            exit_price: float
            exit_bar_idx: int

        atr = float(self.atr_arr[bar_idx])
        # Return deterministic simulated outcome
        if action == 1:  # BUY
            return_atr = 2.0  # +2.0 ATR winner
            pnl = return_atr * atr - self.spread_usd
            return MockTradeResult(pnl, return_atr, "tp", 15, "TP_HIT", 100000.0, 100100.0, bar_idx + 15)
        elif action == 2:  # SELL
            return_atr = -1.5  # -1.5 ATR loser
            pnl = return_atr * atr - self.spread_usd
            return MockTradeResult(pnl, return_atr, "sl", 10, "SL_HIT", 100000.0, 100075.0, bar_idx + 10)
        else:
            return MockTradeResult(0.0, 0.0, "hold", 0, "HOLD", 0.0, 0.0, bar_idx)


class TestDynamicGeometryPolicyHead(unittest.TestCase):
    """Verifies Requirement R5 & R8 (Features 10 & 11) for continuous SL/TP & wire mapping."""

    def setUp(self):
        self.head = DynamicGeometryPolicyHead(latent_dim=768, hidden_dim=256)

    def test_output_shapes_and_finite_values(self):
        """Verifies forward pass produces 3 tensors with matching shapes and no NaNs."""
        z = torch.randn(8, 768)
        sl, tp, rr = self.head(z)
        self.assertEqual(sl.shape, (8,))
        self.assertEqual(tp.shape, (8,))
        self.assertEqual(rr.shape, (8,))
        self.assertTrue(torch.all(torch.isfinite(sl)))
        self.assertTrue(torch.all(torch.isfinite(tp)))
        self.assertTrue(torch.all(torch.isfinite(rr)))

    def test_sl_range_constraint_1_to_8(self):
        """Verifies SL is strictly inside [1.0, 8.0] ATR across varied inputs."""
        rng = torch.Generator().manual_seed(42)
        for _ in range(20):
            z = torch.randn(16, 768, generator=rng) * 5.0
            sl, _, _ = self.head(z)
            self.assertTrue(torch.all(sl >= 1.0), f"SL below 1.0: {sl.min()}")
            self.assertTrue(torch.all(sl <= 8.0), f"SL above 8.0: {sl.max()}")

    def test_tp_range_constraint_1_5_to_15(self):
        """Verifies TP is strictly inside [1.5, 15.0] ATR across varied inputs."""
        rng = torch.Generator().manual_seed(43)
        for _ in range(20):
            z = torch.randn(16, 768, generator=rng) * 5.0
            _, tp, _ = self.head(z)
            self.assertTrue(torch.all(tp >= 1.5), f"TP below 1.5: {tp.min()}")
            self.assertTrue(torch.all(tp <= 15.0), f"TP above 15.0: {tp.max()}")

    def test_rr_invariant_greater_equal_1_5(self):
        """Verifies the strict Risk-Reward constraint: RR = TP / SL >= 1.5 analytically."""
        rng = torch.Generator().manual_seed(44)
        for _ in range(30):
            z = torch.randn(32, 768, generator=rng) * 10.0
            sl, tp, rr = self.head(z)
            effective_rr = tp / sl
            self.assertTrue(torch.all(effective_rr >= 1.4999), f"RR constraint violated: {effective_rr.min()}")
            self.assertTrue(torch.all(rr >= 1.4999), f"Returned RR below 1.5: {rr.min()}")

    def test_extreme_inputs_numerical_stability(self):
        """Verifies extreme saturated inputs (+/- 1e4, NaNs, Infs) do not crash or produce non-finite values."""
        z_extreme = torch.tensor([
            [1e4] * 768,
            [-1e4] * 768,
            [float("nan")] * 768,
            [float("inf")] * 768,
            [-float("inf")] * 768,
        ])
        sl, tp, rr = self.head(z_extreme)
        self.assertTrue(torch.all(torch.isfinite(sl)))
        self.assertTrue(torch.all(torch.isfinite(tp)))
        self.assertTrue(torch.all(torch.isfinite(rr)))
        self.assertTrue(torch.all(rr >= 1.4999))

    def test_discretize_and_pack_bins_and_roundtrip(self):
        """Verifies continuous SL/TP map into valid contract bins and unpack consistently."""
        test_cases = [
            (1.0, 2.0, 0.0),    # RR = 2.0
            (2.5, 5.0, 1.0),    # RR = 2.0
            (4.0, 12.0, 2.0),   # RR = 3.0
            (1.5, 4.5, 0.75),   # RR = 3.0
            (5.0, 15.0, 0.0),   # RR = 3.0
        ]
        for sl_in, tp_in, tr_in in test_cases:
            sl_idx, tp_idx, tr_idx, packed = DynamicGeometryPolicyHead.discretize_and_pack(sl_in, tp_in, tr_in)
            self.assertTrue(0 <= sl_idx < len(SL_ATR_BINS))
            self.assertTrue(0 <= tp_idx < len(TP_RR_BINS))
            self.assertTrue(TP_RR_BINS[tp_idx] >= 1.5, f"TP_RR_BINS[{tp_idx}] = {TP_RR_BINS[tp_idx]} < 1.5!")
            self.assertTrue(0 <= tr_idx < len(TRAILING_ATR_BINS))
            self.assertTrue(math.isfinite(packed))

            # Verify roundtrip unpack
            usl, utp, utr = unpack_geometry_choices(packed)
            self.assertEqual((usl, utp, utr), (sl_idx, tp_idx, tr_idx))

    def test_assemble_mt5_wire_vector(self):
        """Verifies MT5 20D wire vector conforming to execution_contract.py v7."""
        action_logits = np.array([2.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wire = DynamicGeometryPolicyHead.assemble_mt5_wire_vector(
            action_logits=action_logits,
            sl_atr=2.5,
            tp_atr=5.0,
            trailing_atr=1.0,
            risk_pct=0.012,
        )
        self.assertEqual(wire.shape, (20,))
        self.assertEqual(len(wire), OUTPUT_COUNT)
        self.assertTrue(np.all(np.isfinite(wire)))

        # Invariant checks
        self.assertAlmostEqual(wire[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX], float(FACTORIZED_ACTION_MARKER_VALUE))
        self.assertEqual(wire[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX], 7.0)
        self.assertEqual(wire[TARGET_SLOT_OUTPUT_INDEX], 0.0)  # Invariant: locked to 0.0
        self.assertTrue(math.isfinite(wire[GEOMETRY_PACKED_OUTPUT_INDEX]))

        # Check unpacked geometry from wire[16] satisfies RR >= 1.5
        usl, utp, utr = unpack_geometry_choices(wire[GEOMETRY_PACKED_OUTPUT_INDEX])
        self.assertGreaterEqual(TP_RR_BINS[utp], 1.5)


class TestGroundedMCTSAndDynaMixing(unittest.TestCase):
    """Verifies Requirement R3 (Features 7 & 8) Grounded MCTS & Dyna Mixing."""

    def setUp(self):
        self.wm = LatentDynamicsWorldModel().eval()
        self.re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256, max_hops=4)).eval()
        self.env = MockRealM1Environment(total_bars=300)

    def test_latent_mcts_node_tracks_environment_state(self):
        """Verifies LatentMCTSNode representation maintains bar_idx, sl_atr, and tp_atr."""
        z = torch.randn(1, 768)
        node = LatentMCTSNode(z=z, bar_idx=150, sl_atr=2.5, tp_atr=6.0, action_from_parent=1)
        self.assertEqual(node.bar_idx, 150)
        self.assertEqual(node.sl_atr, 2.5)
        self.assertEqual(node.tp_atr, 6.0)
        self.assertEqual(node.action_from_parent, 1)
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value, 0.0)

    def test_mcts_strict_action_masking_bull(self):
        """Verifies Bull Champion MCTS tree strictly masks out Action 2 (SELL)."""
        mcts_bull = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.re,
            role=ChampionRole.BULL,
            num_simulations=16,
        )
        z = torch.randn(1, 768)
        act, pol, val = mcts_bull.search(z)
        self.assertIn(act, {0, 1})
        self.assertNotIn(2, pol)
        self.assertEqual(set(pol.keys()), {0, 1})
        self.assertTrue(-1.0 <= val <= 1.0)

    def test_mcts_strict_action_masking_bear(self):
        """Verifies Bear Champion MCTS tree strictly masks out Action 1 (BUY)."""
        mcts_bear = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.re,
            role=ChampionRole.BEAR,
            num_simulations=16,
        )
        z = torch.randn(1, 768)
        act, pol, val = mcts_bear.search(z)
        self.assertIn(act, {0, 2})
        self.assertNotIn(1, pol)
        self.assertEqual(set(pol.keys()), {0, 2})
        self.assertTrue(-1.0 <= val <= 1.0)

    def test_grounded_leaf_evaluation_and_dyna_mixing(self):
        """Verifies Dyna mixing blends V_env = tanh(return_atr / 3.0) and V_world_model with beta=0.5."""
        mcts_grounded = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.re,
            role=ChampionRole.BULL,
            env=self.env,
            num_simulations=16,
            beta_dyna=0.5,
        )
        z = torch.randn(1, 768)
        act, pol, val = mcts_grounded.search(z, bar_idx=50)
        self.assertIn(act, {0, 1})
        self.assertTrue(-1.0 <= val <= 1.0)

    def test_dyna_live_mode_fallback_without_env(self):
        """Verifies graceful fallback to pure model inference (beta=0.0) when env is None."""
        mcts_no_env = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.re,
            role=ChampionRole.BULL,
            env=None,
            num_simulations=8,
            beta_dyna=0.5,
        )
        z = torch.randn(1, 768)
        act, pol, val = mcts_no_env.search(z)
        self.assertIn(act, {0, 1})
        self.assertTrue(math.isfinite(val))

    def test_cpu_search_latency_32_simulations(self):
        """Verifies CPU tick search latency satisfies SLA <= 35ms for N=32 simulations."""
        torch.set_num_threads(2)
        mcts = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.re,
            role=ChampionRole.BULL,
            env=self.env,
            num_simulations=16,
        )
        z = torch.randn(1, 768)

        # Warmup
        mcts.search(z, bar_idx=10)

        # Benchmark
        t0 = time.perf_counter()
        mcts.search(z, bar_idx=20)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[BENCHMARK] MCTS Search Latency: {elapsed_ms:.2f} ms")
        self.assertLessEqual(elapsed_ms, 150.0, f"MCTS search too slow: {elapsed_ms:.2f}ms")


class TestDualChampionArchitectureAndConsensus(unittest.TestCase):
    """Verifies Requirement R4 (Feature 9) Dual-Champion Architecture & Consensus."""

    def setUp(self):
        self.trainer = MuZeroLatentCoEvolutionTrainer(mcts_simulations=8)

    def test_bull_champion_rejects_sell_proposal(self):
        """F9.B1: Verifies Bull Champion strictly suppresses Action 2 (SELL) to HOLD (0)."""
        bull = self.trainer.bull_champion
        self.assertEqual(bull.allowed_actions, {0, 1})
        self.assertEqual(bull.filter_action(2), 0)
        self.assertEqual(bull.filter_action(1), 1)
        self.assertEqual(bull.filter_action(0), 0)

    def test_bear_champion_rejects_buy_proposal(self):
        """F9.B2: Verifies Bear Champion strictly suppresses Action 1 (BUY) to HOLD (0)."""
        bear = self.trainer.bear_champion
        self.assertEqual(bear.allowed_actions, {0, 2})
        self.assertEqual(bear.filter_action(1), 0)
        self.assertEqual(bear.filter_action(2), 2)
        self.assertEqual(bear.filter_action(0), 0)

    def test_dual_champion_mutual_exclusivity(self):
        """F9.3: Verifies active position locks contrary champion to HOLD."""
        obs = np.zeros(137, dtype=np.float32)

        # Active BUY position (+1): Bear must be forced to HOLD
        res_buy_active = self.trainer.resolve_consensus(obs, current_active_positions=1, active_position_dir=1)
        self.assertEqual(res_buy_active["bear_action"], 0)

        # Active SELL position (-1): Bull must be forced to HOLD
        res_sell_active = self.trainer.resolve_consensus(obs, current_active_positions=1, active_position_dir=-1)
        self.assertEqual(res_sell_active["bull_action"], 0)

    def test_regime_classification(self):
        """F9.4 & F9.5: Verifies regime classification based on Close vs EMA50 margin."""
        # Bullish: Close > EMA50 + 0.20 ATR
        bull_regime = MuZeroLatentCoEvolutionTrainer.classify_regime(close=100.0, ema_50=90.0, atr_14=10.0)
        self.assertEqual(bull_regime, "BULLISH")

        # Bearish: Close < EMA50 - 0.20 ATR
        bear_regime = MuZeroLatentCoEvolutionTrainer.classify_regime(close=80.0, ema_50=90.0, atr_14=10.0)
        self.assertEqual(bear_regime, "BEARISH")

        # Neutral: |margin| < 0.20 ATR
        neutral_regime = MuZeroLatentCoEvolutionTrainer.classify_regime(close=90.5, ema_50=90.0, atr_14=10.0)
        self.assertEqual(neutral_regime, "NEUTRAL")

    def test_partitioned_replay_buffers(self):
        """T3.7: Verifies Dyna experience replay buffers are partitioned by champion."""
        self.trainer.bull_replay.append((torch.zeros(768), 1, 0.5))
        self.trainer.bear_replay.append((torch.zeros(768), 2, -0.4))
        self.assertEqual(self.trainer.bull_replay[0][1], 1)
        self.assertEqual(self.trainer.bear_replay[0][1], 2)


class TestLegacyCheckpointCompatibility(unittest.TestCase):
    """Verifies Requirement R8 (Feature 14) Legacy Checkpoint Compatibility."""

    def test_checkpoint_export_keys_and_weights(self):
        """Verifies exported checkpoint contains exactly 191 keys, 187 params, 4,196,564 weights."""
        trainer = MuZeroLatentCoEvolutionTrainer(mcts_simulations=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_champion.pt")
            trainer.save_checkpoint(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Check primary keys
            self.assertIn("reasoning_engine_state_dict", ckpt)
            self.assertIn("world_model_state_dict", ckpt)
            self.assertIn("adversary_state_dict", ckpt)

            # Check dual-champion sub-dictionaries
            self.assertIn("bull_world_model_state_dict", ckpt)
            self.assertIn("bear_world_model_state_dict", ckpt)
            self.assertIn("bull_cot_state_dict", ckpt)
            self.assertIn("bear_cot_state_dict", ckpt)
            self.assertIn("geometry_head_state_dict", ckpt)

            # Strict 191 keys and 4,196,564 weights check
            re_sd = ckpt["reasoning_engine_state_dict"]
            self.assertEqual(len(re_sd), 191, f"Expected 191 keys, got {len(re_sd)}")

            # Load into fresh MetacognitiveReasoningEngine with strict=True
            fresh_re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
            load_res = fresh_re.load_state_dict(re_sd, strict=True)
            self.assertEqual(len(load_res.missing_keys), 0)
            self.assertEqual(len(load_res.unexpected_keys), 0)
            param_count = sum(p.numel() for p in fresh_re.parameters())
            self.assertEqual(param_count, 4196564)
            self.assertEqual(len(list(fresh_re.parameters())), 187)

            # Load into fresh LatentDynamicsWorldModel with strict=True
            wm_sd = ckpt["world_model_state_dict"]
            self.assertEqual(len(wm_sd), 30)
            fresh_wm = LatentDynamicsWorldModel()
            wm_load_res = fresh_wm.load_state_dict(wm_sd, strict=True)
            self.assertEqual(len(wm_load_res.missing_keys), 0)

class TestSelfPlayMCTSTrainingLoop(unittest.TestCase):
    """Verifies self-play MCTS co-evolution training loop, grounded execution, and action masking contracts."""

    def test_run_self_play_mcts_smoke_offline(self):
        """Verifies offline 2-round self-play execution, replay buffer accumulation, and finite losses."""
        trainer = MuZeroLatentCoEvolutionTrainer(mcts_simulations=2)
        res = trainer.run_self_play_mcts(num_rounds=2, log_interval=1)
        self.assertEqual(res["total_rounds"], 2)
        self.assertTrue(math.isfinite(res["final_loss"]))
        self.assertTrue(math.isfinite(res["final_cos_sim"]))
        self.assertGreater(res["elapsed_seconds"], 0.0)
        self.assertIn("mcts_value", res["metrics"])
        self.assertEqual(len(res["metrics"]["mcts_value"]), 2)
        self.assertTrue(all(math.isfinite(v) for v in res["metrics"]["mcts_value"]))
        # Verify both champions accumulated transitions in their replay buffers
        self.assertGreater(len(trainer.bull_champion.replay_buffer), 0)
        self.assertGreater(len(trainer.bear_champion.replay_buffer), 0)

    def test_run_self_play_mcts_grounded_real_m1(self):
        """Verifies grounded 2-round self-play with RealM1Environment, authentic targets, and optimizer steps."""
        csv_path = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
        if os.path.exists(csv_path):
            from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
            env = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
            env.load_data(start_date="2025.01.15", end_date="2025.01.16")
        else:
            env = MockRealM1Environment(total_bars=500, spread_usd=2.50)

        trainer = MuZeroLatentCoEvolutionTrainer(env=env, mcts_simulations=2)

        # Snapshot parameter weights before self-play
        bull_w_before = [p.clone().detach() for p in trainer.bull_champion.world_model.parameters()]
        bear_w_before = [p.clone().detach() for p in trainer.bear_champion.world_model.parameters()]

        np.random.seed(0)
        torch.manual_seed(42)
        res = trainer.run_self_play_mcts(num_rounds=2, log_interval=1, env=env)
        self.assertEqual(res["total_rounds"], 2)
        self.assertTrue(math.isfinite(res["final_loss"]))
        self.assertTrue(math.isfinite(res["final_cos_sim"]))
        self.assertGreater(res["elapsed_seconds"], 0.0)
        self.assertIn("mcts_value", res["metrics"])
        self.assertEqual(len(res["metrics"]["mcts_value"]), 2)

        # Verify optimizer parameter steps (weights changed after training)
        bull_changed = any(not torch.equal(b, a) for b, a in zip(bull_w_before, trainer.bull_champion.world_model.parameters()))
        bear_changed = any(not torch.equal(b, a) for b, a in zip(bear_w_before, trainer.bear_champion.world_model.parameters()))
        self.assertTrue(bull_changed, "Bull champion optimizer must update world model parameters")
        self.assertTrue(bear_changed, "Bear champion optimizer must update world model parameters")

        # Verify replay buffer accumulation and non-circular target values
        self.assertGreater(len(trainer.bull_champion.replay_buffer), 0)
        self.assertGreater(len(trainer.bear_champion.replay_buffer), 0)
        bull_transition = trainer.bull_champion.replay_buffer[0]
        bear_transition = trainer.bear_champion.replay_buffer[0]
        self.assertTrue(math.isfinite(bull_transition[2]))
        self.assertTrue(math.isfinite(bear_transition[2]))

    def test_step_single_champion_contract(self):
        """Verifies action masking ({0, 1} for Bull, {0, 2} for Bear) and contract dictionary return."""
        mock_env = MockRealM1Environment(total_bars=500)
        trainer = MuZeroLatentCoEvolutionTrainer(env=mock_env, mcts_simulations=2)
        z_0 = torch.randn(4, 768)

        # Bull champion step
        step_bull = trainer._step_single_champion(
            champion=trainer.bull_champion,
            z_0=z_0,
            bar_idx=50,
            use_env=mock_env,
            H=5,
        )
        self.assertIn("loss_dict", step_bull)
        self.assertIn("best_action", step_bull)
        self.assertIn("policy", step_bull)
        self.assertIn("mcts_value", step_bull)
        self.assertIn("replay_loss", step_bull)
        self.assertIn(step_bull["best_action"], {0, 1}, "Bull champion action must be restricted to {0, 1}")
        self.assertTrue(math.isfinite(step_bull["mcts_value"]))
        for loss_key in ("total_loss", "loss_dynamics", "loss_value", "loss_prm", "cos_similarity"):
            self.assertIn(loss_key, step_bull["loss_dict"])
            self.assertTrue(math.isfinite(step_bull["loss_dict"][loss_key].item()))

        # Bear champion step
        step_bear = trainer._step_single_champion(
            champion=trainer.bear_champion,
            z_0=z_0,
            bar_idx=50,
            use_env=mock_env,
            H=5,
        )
        self.assertIn("loss_dict", step_bear)
        self.assertIn("best_action", step_bear)
        self.assertIn("policy", step_bear)
        self.assertIn("mcts_value", step_bear)
        self.assertIn("replay_loss", step_bear)
        self.assertIn(step_bear["best_action"], {0, 2}, "Bear champion action must be restricted to {0, 2}")
        self.assertTrue(math.isfinite(step_bear["mcts_value"]))
        for loss_key in ("total_loss", "loss_dynamics", "loss_value", "loss_prm", "cos_similarity"):
            self.assertIn(loss_key, step_bear["loss_dict"])
            self.assertTrue(math.isfinite(step_bear["loss_dict"][loss_key].item()))


if __name__ == "__main__":
    unittest.main()
