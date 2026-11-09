"""
Adversarial Stress-Testing, Fuzzing & Boundary Probing Suite for Milestone 5 Deliverables.
Target: Milestone 5 Quality Gate Challenge Battery
Author: teamwork_challenger_m5_1 (Role: critic, specialist)

Probes:
- Part 1: Real M1 Environment (boundary clamping, out-of-bounds rejection, simultaneous breach SL priority, spread linearity)
- Part 2: Grounded World Model (NaN/Inf adversarial injection, deep BPTT gradient norms, non-circular target validation)
- Part 3: Grounded MCTS & Dual Champions (action masking strictness, 1000-sample SL/TP/RR fuzzing, CPU search SLA)
- Part 4: Walk-Forward Isolation (sliced environment terminal boundary clamping, future-peeking immunity, TradeResult immutability)
- Part 5: Checkpoint & MT5 Integration (exact 191 keys, 187 parameters, 4,196,564 weights under strict=True)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import torch
import torch.nn.functional as F

# Constrain thread count for CPU execution
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(min(4, os.cpu_count() or 4))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from self_growing_trader.execution_contract import (
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    unpack_geometry_choices,
    GEOMETRY_PACKED_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    TARGET_SLOT_OUTPUT_INDEX,
)
from research_reasoning_world_class.src.real_m1_environment import (
    RealM1Environment,
    TradeResult,
    ExitType,
)
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    AdversarialMarketStressor,
    generate_trajectory_targets,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.muzero_latent_coevolution import (
    ChampionRole,
    DualLatentMCTSSearch,
    DynamicGeometryPolicyHead,
    MuZeroLatentCoEvolutionTrainer,
)
from scripts.train_mythos_muzero_coevolution import SlicedM1Environment

DEFAULT_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "he_chuyen_gia"
    / "v193_runs"
    / "mythos_muzero_coevolved_champion.pt"
)


class SharedM1Fixture:
    _env: RealM1Environment | None = None

    @classmethod
    def get_env(cls) -> RealM1Environment:
        if cls._env is None:
            if not os.path.exists(DEFAULT_CSV_PATH):
                raise FileNotFoundError(f"Canonical candle file not found: {DEFAULT_CSV_PATH}")
            env = RealM1Environment(csv_path=DEFAULT_CSV_PATH, spread_usd=2.50)
            env.load_data(start_date="2025.01.15", end_date="2025.02.28")
            cls._env = env
        return cls._env


class TestAdversarialPart1RealM1(unittest.TestCase):
    """Adversarial stress-testing of RealM1Environment boundaries, exceptions, and execution physics."""

    @classmethod
    def setUpClass(cls):
        cls.env = SharedM1Fixture.get_env()

    def test_probe_1_1_boundary_indices_and_out_of_bounds_rejection(self):
        """Probes edge indices [0, total_bars-1] and confirms strict rejection of out-of-bounds indices."""
        total = self.env.total_bars
        self.assertEqual(total, 64785)

        # Valid boundaries
        obs_first = self.env.get_observation(0)
        self.assertEqual(obs_first.shape, (137,))
        obs_last = self.env.get_observation(total - 1)
        self.assertEqual(obs_last.shape, (137,))

        # Strict rejection for get_observation
        with self.assertRaises(IndexError):
            self.env.get_observation(-1)
        with self.assertRaises(IndexError):
            self.env.get_observation(total)
        with self.assertRaises(IndexError):
            self.env.get_observation(total + 100)

        # Strict rejection for simulate_trade
        with self.assertRaises(IndexError):
            self.env.simulate_trade(-1, action=1, sl_atr=2.0, tp_atr=4.0)
        with self.assertRaises(IndexError):
            self.env.simulate_trade(total, action=1, sl_atr=2.0, tp_atr=4.0)

        # Invalid action and parameters rejection
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=0, sl_atr=2.0, tp_atr=4.0)
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=3, sl_atr=2.0, tp_atr=4.0)
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=1, sl_atr=0.0, tp_atr=4.0)
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=1, sl_atr=-1.0, tp_atr=4.0)
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=1, sl_atr=2.0, tp_atr=0.0)

        # Terminal bar execution: avail = 0 -> DATASET_END with 0 holding bars
        res_terminal = self.env.simulate_trade(total - 1, action=1, sl_atr=2.0, tp_atr=4.0)
        self.assertEqual(res_terminal.holding_bars, 0)
        self.assertEqual(res_terminal.exit_reason, "DATASET_END")
        self.assertEqual(res_terminal.exit_bar_idx, total - 1)

    def test_probe_1_2_simultaneous_breach_exhaustive_fuzzing(self):
        """
        Adversarially probes:
        1. Simultaneous collision: when both SL and TP are breached on the SAME bar (sl_idx == tp_idx),
           institutional conservative SL-first priority MUST ALWAYS trigger (exit_type='sl', exit_reason='SIMULTANEOUS_SL_FIRST').
        2. Non-spanning gap bars: when a bar gaps up or down past entry, only the breached side triggers first.
        """
        # Find 20 bars where the very next bar spans across ref_close (low < close < high)
        spanning_bars = []
        gap_bars = []
        for b in range(100, 2000):
            c = float(self.env.close_arr[b])
            next_low = float(self.env.low_arr[b + 1])
            next_high = float(self.env.high_arr[b + 1])
            if next_low < c - 1.0 and next_high > c + 1.0:
                if len(spanning_bars) < 20:
                    spanning_bars.append(b)
            elif next_low > c:
                if len(gap_bars) < 5:
                    gap_bars.append(b)
            if len(spanning_bars) >= 20 and len(gap_bars) >= 5:
                break

        self.assertEqual(len(spanning_bars), 20)
        self.assertGreaterEqual(len(gap_bars), 1)

        # 1. Simultaneous collision on spanning bars
        for bar in spanning_bars:
            # BUY side
            res_buy = self.env.simulate_trade(bar, action=1, sl_atr=0.0001, tp_atr=0.0001, max_holding_bars=60)
            self.assertEqual(res_buy.exit_type, "sl", f"BUY failed SL priority at spanning bar {bar}")
            self.assertEqual(res_buy.exit_reason, "SIMULTANEOUS_SL_FIRST", f"BUY wrong reason at spanning bar {bar}")
            self.assertEqual(res_buy.holding_bars, 1, f"BUY holding != 1 at spanning bar {bar}")
            self.assertLess(res_buy.pnl_usd, 0.0, f"BUY PnL >= 0 at spanning bar {bar}")

            # SELL side
            res_sell = self.env.simulate_trade(bar, action=2, sl_atr=0.0001, tp_atr=0.0001, max_holding_bars=60)
            self.assertEqual(res_sell.exit_type, "sl", f"SELL failed SL priority at spanning bar {bar}")
            self.assertEqual(res_sell.exit_reason, "SIMULTANEOUS_SL_FIRST", f"SELL wrong reason at spanning bar {bar}")
            self.assertEqual(res_sell.holding_bars, 1, f"SELL holding != 1 at spanning bar {bar}")
            self.assertLess(res_sell.pnl_usd, 0.0, f"SELL PnL >= 0 at spanning bar {bar}")

        # 2. Path-dependent gap bar (e.g. gap up: next_low > close)
        gap_bar = gap_bars[0]
        # For BUY: gapped up, so TP is touched before SL -> exit_type='tp', exit_reason='TP_HIT'
        res_gap_buy = self.env.simulate_trade(gap_bar, action=1, sl_atr=0.0001, tp_atr=0.0001, max_holding_bars=60)
        self.assertEqual(res_gap_buy.exit_type, "tp")
        self.assertEqual(res_gap_buy.exit_reason, "TP_HIT")

    def test_probe_1_3_spread_accounting_linearity_across_regimes(self):
        """Probes spread friction across multiple spreads ($0, $2.50, $10.00) to confirm exact linear subtraction."""
        test_bars = [100, 500, 1200, 3000, 8000]
        for bar in test_bars:
            for act in (1, 2):
                res_0 = self.env.simulate_trade(bar, action=act, sl_atr=2.0, tp_atr=4.0, spread_usd=0.0)
                res_25 = self.env.simulate_trade(bar, action=act, sl_atr=2.0, tp_atr=4.0, spread_usd=2.50)
                res_10 = self.env.simulate_trade(bar, action=act, sl_atr=2.0, tp_atr=4.0, spread_usd=10.00)

                diff_25 = res_0.pnl_usd - res_25.pnl_usd
                diff_10 = res_0.pnl_usd - res_10.pnl_usd

                self.assertAlmostEqual(diff_25, 2.50, delta=1e-3, msg=f"Spread 2.50 delta invalid at bar {bar}, act {act}")
                self.assertAlmostEqual(diff_10, 10.00, delta=1e-3, msg=f"Spread 10.00 delta invalid at bar {bar}, act {act}")


class TestAdversarialPart2GroundedWorldModel(unittest.TestCase):
    """Adversarial stress-testing of LatentDynamicsWorldModel numerical health, gradients, and targets."""

    @classmethod
    def setUpClass(cls):
        cls.env = SharedM1Fixture.get_env()
        cls.wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, obs_dim=137)

    def test_probe_2_1_nan_inf_adversarial_injection(self):
        """Injects NaNs, positive/negative infinities, and extreme magnitudes into encoder and stressor."""
        # 1. NaN and Inf in observation
        corrupted_obs = np.full((137,), np.nan, dtype=np.float32)
        corrupted_obs[10] = np.inf
        corrupted_obs[20] = -np.inf
        corrupted_obs[30] = 1e20

        t_obs = torch.from_numpy(corrupted_obs).unsqueeze(0)
        z_out = self.wm.encode_state(t_obs)
        self.assertEqual(z_out.shape, (1, 768))
        self.assertTrue(torch.all(torch.isfinite(z_out)), "Encoder output contains NaN or Inf on adversarial input")

        # 2. Extreme latent input to transition
        z_corrupted = torch.full((1, 768), float("nan"))
        z_trans = self.wm.transition(z_corrupted, 1)
        self.assertEqual(z_trans.shape, (1, 768))
        self.assertTrue(torch.all(torch.isfinite(z_trans)), "Transition output contains NaN on NaN latent input")

        # 3. AdversarialMarketStressor with extreme volatility factor
        stressor = AdversarialMarketStressor(latent_dim=768)
        stressed_z, slip = stressor(z_out, volatility_factor=100.0)
        self.assertTrue(torch.all(torch.isfinite(stressed_z)))
        self.assertGreater(slip, 0.0)

    def test_probe_2_2_bptt_gradient_norms_and_layernorm_health_deep_rollout(self):
        """Verifies deep 15-step unrolling gradients are non-zero, finite, and non-exploding."""
        wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, obs_dim=137)
        z0 = torch.randn(1, 768, requires_grad=True)
        actions = torch.randint(0, 3, (1, 15), dtype=torch.long)

        pred_z, pred_v, pred_r = wm.unroll_trajectory(z0, actions)
        self.assertEqual(pred_z.shape, (1, 15, 768))

        # Check LayerNorm norm bounds across all rollout steps
        for step in range(15):
            norm_step = torch.norm(pred_z[0, step], p=2).item()
            self.assertGreaterEqual(norm_step, 5.0, f"Representation collapsed at step {step}")
            self.assertLessEqual(norm_step, 60.0, f"Representation exploded at step {step}")

        # Compute loss and verify backward pass
        loss = pred_z.sum() + pred_v.sum() + pred_r.sum()
        loss.backward()

        self.assertIsNotNone(z0.grad)
        grad_norm_z0 = torch.norm(z0.grad, p=2).item()
        self.assertGreater(grad_norm_z0, 1e-6, "Vanishing gradient at z0")
        self.assertLess(grad_norm_z0, 1e4, "Exploding gradient at z0")

        # Check network parameter gradient health
        for name, param in wm.named_parameters():
            if param.grad is not None:
                p_grad_norm = torch.norm(param.grad, p=2).item()
                self.assertTrue(np.isfinite(p_grad_norm), f"Parameter {name} has non-finite gradient")
                self.assertLess(p_grad_norm, 1e4, f"Parameter {name} has exploding gradient: {p_grad_norm}")

    def test_probe_2_3_non_circular_trajectory_target_integrity(self):
        """Verifies generate_trajectory_targets produces authentic future data with zero circular self-echo."""
        start_idx = 3000
        horizon = 8
        actions = [1, 1, 2, 0, 1, 2, 0, 1]

        s0, future_obs, target_v, target_r = generate_trajectory_targets(
            self.env, start_idx=start_idx, actions=actions, horizon_k=horizon, sl_atr=2.0, tp_atr=4.0
        )

        self.assertEqual(s0.shape, (137,))
        self.assertEqual(future_obs.shape, (horizon, 137))
        self.assertEqual(target_v.shape, (horizon, 1))
        self.assertEqual(target_r.shape, (horizon, 1))

        # 1. Non-identity: future_obs is not a repetition of s0
        for k in range(horizon):
            obs_k = future_obs[k]
            diff = np.linalg.norm(obs_k - s0)
            self.assertGreater(diff, 1e-3, f"Step {k} observation is identical to s0 (circular repeat)")

        # 2. Future observations strictly match actual environment at start_idx + k + 1
        for k in range(horizon):
            actual_fut = self.env.get_observation(start_idx + k + 1)
            np.testing.assert_allclose(future_obs[k], actual_fut, atol=1e-6)

        # 3. Target values and rewards are strictly finite and bounded
        self.assertTrue(np.all(np.isfinite(target_v)))
        self.assertTrue(np.all(np.isfinite(target_r)))
        self.assertTrue(np.all(np.abs(target_v) <= 1.0))


class TestAdversarialPart3GroundedMCTS(unittest.TestCase):
    """Adversarial stress-testing of Dual Champion action masking, continuous SL/TP bounds, and SLA."""

    @classmethod
    def setUpClass(cls):
        cls.env = SharedM1Fixture.get_env()
        cot_cfg = ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=8)
        cls.cot = MetacognitiveReasoningEngine(cot_cfg)
        cls.wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, obs_dim=137)
        cls.head = DynamicGeometryPolicyHead(latent_dim=768)

    def test_probe_3_1_strict_action_masking_adversarial_probe(self):
        """Probes action masking to verify Bull NEVER evaluates SELL and Bear NEVER evaluates BUY."""
        bull_searcher = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.cot,
            role=ChampionRole.BULL,
            num_simulations=4,
            beta_dyna=0.0,
        )
        bear_searcher = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.cot,
            role=ChampionRole.BEAR,
            num_simulations=4,
            beta_dyna=0.0,
        )

        test_bars = [100, 500, 1000, 2000, 4000]
        for bar in test_bars:
            z = self.wm.encode_state(self.env.get_observation(bar))

            # Bull Champion search
            bull_act, bull_policy = bull_searcher.search(z, return_value=False)
            self.assertIn(bull_act, (0, 1), f"Bull selected forbidden action {bull_act}")
            self.assertNotIn(2, bull_policy, f"Bull policy expanded forbidden action 2: {bull_policy}")

            # Bear Champion search
            bear_act, bear_policy = bear_searcher.search(z, return_value=False)
            self.assertIn(bear_act, (0, 2), f"Bear selected forbidden action {bear_act}")
            self.assertNotIn(1, bear_policy, f"Bear policy expanded forbidden action 1: {bear_policy}")

    def test_probe_3_2_continuous_geometry_head_fuzzing_1000_samples(self):
        """Fuzzes DynamicGeometryPolicyHead with 1,000 diverse latent vectors to verify strict RR >= 1.5."""
        torch.manual_seed(2026)
        # Mix of standard, high-variance, and extreme inputs
        z_norm = torch.randn(500, 768)
        z_high = torch.randn(300, 768) * 10.0
        z_extreme = torch.empty(200, 768).uniform_(-100.0, 100.0)
        z_batch = torch.cat([z_norm, z_high, z_extreme], dim=0)

        sl, tp, rr = self.head(z_batch)

        self.assertEqual(sl.shape, (1000,))
        self.assertEqual(tp.shape, (1000,))
        self.assertEqual(rr.shape, (1000,))

        # Strict continuous bounds
        self.assertTrue(torch.all(sl >= 1.0), "SL violates min bound 1.0")
        self.assertTrue(torch.all(sl <= 8.0), "SL violates max bound 8.0")
        self.assertTrue(torch.all(tp >= 1.5), "TP violates min bound 1.5")
        self.assertTrue(torch.all(tp <= 15.0), "TP violates max bound 15.0")
        self.assertTrue(torch.all(rr >= 1.4999), "RR violates strict invariant RR >= 1.5")

        # Discretization and wire packing roundtrip on 20 random samples
        for idx in range(20):
            sl_val = float(sl[idx].item())
            tp_val = float(tp[idx].item())
            wire = DynamicGeometryPolicyHead.assemble_mt5_wire_vector(
                action_logits=[1.0, 0.0, 0.0],
                sl_atr=sl_val,
                tp_atr=tp_val,
                trailing_atr=1.0,
            )
            self.assertEqual(wire.shape, (20,))
            self.assertEqual(wire[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX], 7.0)
            self.assertEqual(wire[TARGET_SLOT_OUTPUT_INDEX], 0.0)

            packed_logit = float(wire[GEOMETRY_PACKED_OUTPUT_INDEX])
            sl_bin_idx, tp_bin_idx, tr_bin_idx = unpack_geometry_choices(packed_logit)

            self.assertIn(sl_bin_idx, range(len(SL_ATR_BINS)))
            self.assertIn(tp_bin_idx, range(len(TP_RR_BINS)))
            self.assertIn(tr_bin_idx, range(len(TRAILING_ATR_BINS)))
            self.assertGreaterEqual(TP_RR_BINS[tp_bin_idx], 1.4999)

    def test_probe_3_3_cpu_search_latency_stress(self):
        """Verifies MCTS latency over 10 consecutive searches meets <= 50.0 ms CPU SLA."""
        searcher = DualLatentMCTSSearch(
            world_model=self.wm,
            reasoning_engine=self.cot,
            num_simulations=16,
            beta_dyna=0.0,
        )
        z = self.wm.encode_state(self.env.get_observation(500))

        # Warmup pass
        searcher.search(z)

        # 10 timed iterations
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            searcher.search(z)
            times.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(times))
        p95_ms = float(np.percentile(times, 95))
        self.assertLess(mean_ms, 50.0, f"Mean MCTS search latency {mean_ms:.2f}ms exceeds 50ms SLA")
        self.assertLess(p95_ms, 50.0, f"P95 MCTS search latency {p95_ms:.2f}ms exceeds 50ms SLA")


class TestAdversarialPart4WalkForwardIsolation(unittest.TestCase):
    """Adversarial stress-testing of SlicedM1Environment boundary clamping and TradeResult immutability."""

    @classmethod
    def setUpClass(cls):
        cls.master = SharedM1Fixture.get_env()
        cls.slice = SlicedM1Environment(cls.master, start_idx=10000, end_idx=15000, name="AdversarialSlice")

    def test_probe_4_1_sliced_boundary_clamping_and_peeking_immunity(self):
        """Probes edge bars of sliced window to verify safe_max_holding prevents reading outside slice."""
        self.assertEqual(len(self.slice), 5000)

        # 1. Terminal bar of slice
        terminal_idx = len(self.slice) - 1
        res_terminal = self.slice.simulate_trade(terminal_idx, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=60)
        self.assertEqual(res_terminal.holding_bars, 0, "Terminal bar allowed holding > 0")
        self.assertEqual(res_terminal.exit_reason, "DATASET_END")
        self.assertEqual(res_terminal.exit_bar_idx, terminal_idx)

        # 2. Near boundary bars (total_bars - 5)
        near_idx = len(self.slice) - 5
        res_near = self.slice.simulate_trade(near_idx, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=60)
        self.assertLessEqual(res_near.holding_bars, 4, "Holding bars exceeded slice boundary")
        self.assertLessEqual(res_near.exit_bar_idx, terminal_idx)

        # 3. Out of bounds index rejection
        with self.assertRaises(IndexError):
            self.slice.get_observation(5000)
        with self.assertRaises(IndexError):
            self.slice.simulate_trade(5000, action=1, sl_atr=2.0, tp_atr=4.0)

    def test_probe_4_2_frozen_traderesult_immutability(self):
        """Confirms TradeResult raises FrozenInstanceError on attempted mutation and supports replace."""
        res = self.slice.simulate_trade(100, action=1, sl_atr=2.0, tp_atr=4.0)

        # Mutation attempts must fail
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            res.pnl_usd = 9999.0  # type: ignore
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            res.exit_reason = "FORGED"  # type: ignore

        # Replace creates clean new object without altering original
        res_replaced = replace(res, pnl_usd=-42.0, exit_reason="REPLACED")
        self.assertEqual(res_replaced.pnl_usd, -42.0)
        self.assertEqual(res_replaced.exit_reason, "REPLACED")
        self.assertNotEqual(res.pnl_usd, -42.0)
        self.assertNotEqual(res.exit_reason, "REPLACED")


class TestAdversarialPart5CheckpointExactness(unittest.TestCase):
    """Adversarial stress-testing of Checkpoint format, strict tensor loading, and weight exactness."""

    def test_probe_5_1_checkpoint_strict_191_187_4196564_and_wm_30(self):
        """Verifies checkpoint contains exactly 191 keys, 187 parameters, 4,196,564 weights and 30 WM keys."""
        self.assertTrue(CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

        # 1. Reasoning Engine State Dict
        cot_sd = ckpt["reasoning_engine_state_dict"]
        self.assertEqual(len(cot_sd), 191, f"Expected 191 state_dict keys, found {len(cot_sd)}")

        cot_cfg = ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=8)
        engine = MetacognitiveReasoningEngine(cot_cfg)
        load_res = engine.load_state_dict(cot_sd, strict=True)
        self.assertEqual(len(load_res.missing_keys), 0, f"Missing keys: {load_res.missing_keys}")
        self.assertEqual(len(load_res.unexpected_keys), 0, f"Unexpected keys: {load_res.unexpected_keys}")

        named_params = dict(engine.named_parameters())
        self.assertEqual(len(named_params), 187, f"Expected 187 parameters, found {len(named_params)}")

        total_weights = sum(p.numel() for p in engine.parameters())
        self.assertEqual(total_weights, 4_196_564, f"Expected 4,196,564 weights, found {total_weights}")

        # 2. World Model State Dict
        wm_sd = ckpt["world_model_state_dict"]
        self.assertEqual(len(wm_sd), 30, f"Expected 30 WM keys, found {len(wm_sd)}")

        wm = LatentDynamicsWorldModel()
        wm_load = wm.load_state_dict(wm_sd, strict=True)
        self.assertEqual(len(wm_load.missing_keys), 0)
        self.assertEqual(len(wm_load.unexpected_keys), 0)


if __name__ == "__main__":
    unittest.main()
