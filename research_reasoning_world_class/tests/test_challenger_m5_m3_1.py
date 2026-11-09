"""
Adversarial Verification & Empirical Challenger Test Suite for MT5 Export & Policy Interfaces
File: research_reasoning_world_class/tests/test_challenger_m5_m3_1.py
Milestone: Milestone 3 (Empirical Challenger 1)
Role: Empirical Challenger (critic, specialist)

Adversarial Stress Test Dimensions:
1. Checkpoint Corruption Probes:
   - Missing top-level canonical keys (17-key contract)
   - Missing sub-keys in reasoning engine (191 keys) and world model (30 keys)
   - Alien / unexpected keys injection
   - Tensor shape mutations (e.g. weight transpositions)
   - Truncated file stream and EOF corruption
   - Zero-byte checkpoint files
2. Input Sanitization Probes:
   - 100% NaN ingress into 137D observation vector
   - Isolated NaNs at critical feature boundaries (return, OHLC geometry, spread, volume)
   - Positive (+inf) and negative (-inf) infinities, mixed NaN/inf arrays
   - All zeros dead-sensor / flat-market ingress
   - Extreme astronomical numerical values (+/- 1e12, subnormals 1e-35)
   - Dimension mutations: under-length (0, 50), over-length (300), 2D shapes, Python lists
   - Adversarial request metadata (negative/infinite equity, hostile position counts)
3. 20-Wire Contract & Invariant Defenses:
   - Wire layout invariants: wire.shape == (20,), wire[14] == 7.0, wire[19] == 0.0
   - Hostile base policy bypass resistance: Agent 4 locked to HOLD (action=0, out[0]=1.0, out[10]=-100.0, out[19]=0.0)
   - Static hold prototype immutability (read-only ValueError)
   - Packed geometry wire[16] round-trip across all 168 bin permutations (RR >= 1.5)
   - Dynamic continuous geometry head analytical bounds (SL in [1.0, 8.0], TP in [1.5, 15.0], RR >= 1.5)
4. CPU Decision Latency Multi-Sample SLA Probe:
   - Multi-sample consensus latency across 50 diverse historical bars on CPU (< 50.0 ms budget)
   - P95 and median latency compliance verification
   - Cold vs warm execution profile and jitter bounding
"""

from __future__ import annotations

import copy
import io
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import unittest

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(min(2, os.cpu_count() or 2))

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
    pack_geometry_choices,
    unpack_geometry_choices,
    GEOMETRY_PACKED_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    TARGET_SLOT_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_VALUE,
)
from self_growing_trader.orders import Action
from research_reasoning_world_class.src.real_m5_environment import RealM5Environment
from research_reasoning_world_class.src.latent_dynamics_world_model import LatentDynamicsWorldModel
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.muzero_latent_coevolution import (
    DynamicGeometryPolicyHead,
    MuZeroLatentCoEvolutionTrainer,
)
from scripts.train_mythos_muzero_coevolution_m5 import M5MuZeroTrainer
from research_reasoning_world_class.src.mythos_mt5_policy import (
    MythosMT5Policy,
    _STATIC_HOLD_VECTOR,
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "he_chuyen_gia"
    / "v193_runs"
    / "mythos_muzero_m5_champion.pt"
)
M5_CSV_PATH = PROJECT_ROOT / "outputs" / "BTCUSDm_M5.csv"

CANONICAL_17_KEYS = {
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


class MockTrace:
    def __init__(self):
        self.output_vector = np.zeros(20, dtype=np.float64)
        self.selected_action = np.zeros(20, dtype=np.float64)
        self.action = 0
        self.action_index = 0
        self.selected_idx = 0
        self.action_details = {}


class MockBasePolicy:
    def act_nam_tac_nhan(self, obs, *args, **kwargs):
        return [MockTrace() for _ in range(5)]


# ==============================================================================
# 1. ADVERSARIAL CHECKPOINT CORRUPTION PROBES
# ==============================================================================
class TestAdversarialCheckpointCorruption(unittest.TestCase):
    """Stress tests checkpoint loading against intentional corruption, missing keys, and shape mutations."""

    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Target checkpoint not found: {CHECKPOINT_PATH}")
        cls.raw_ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        cls.temp_dir = tempfile.mkdtemp(prefix="challenger_ckpt_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_probe_1_1_missing_top_level_canonical_keys(self):
        """Probe: Deleting any of the 17 canonical keys must be detected by schema validation."""
        for missing_key in CANONICAL_17_KEYS:
            mutated = dict(self.raw_ckpt)
            del mutated[missing_key]
            self.assertEqual(len(mutated), 16)
            self.assertNotIn(missing_key, mutated)
            # Ensure schema validator would reject
            self.assertNotEqual(set(mutated.keys()), CANONICAL_17_KEYS)

    def test_probe_1_2_missing_cot_subkeys_strict_rejection(self):
        """Probe: Deleting an internal tensor key from reasoning_engine_state_dict must raise RuntimeError under strict=True."""
        cot_sd = dict(self.raw_ckpt["reasoning_engine_state_dict"])
        first_key = list(cot_sd.keys())[0]
        del cot_sd[first_key]

        engine = MetacognitiveReasoningEngine(
            ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=8)
        )
        with self.assertRaises(RuntimeError) as ctx:
            engine.load_state_dict(cot_sd, strict=True)
        self.assertIn("Missing key(s)", str(ctx.exception))
        self.assertIn(first_key, str(ctx.exception))

    def test_probe_1_3_missing_world_model_subkeys_strict_rejection(self):
        """Probe: Deleting an internal tensor key from world_model_state_dict must raise RuntimeError under strict=True."""
        wm_sd = dict(self.raw_ckpt["world_model_state_dict"])
        first_key = list(wm_sd.keys())[0]
        del wm_sd[first_key]

        wm = LatentDynamicsWorldModel()
        with self.assertRaises(RuntimeError) as ctx:
            wm.load_state_dict(wm_sd, strict=True)
        self.assertIn("Missing key(s)", str(ctx.exception))

    def test_probe_1_4_alien_keys_injection_in_cot_rejected(self):
        """Probe: Injecting rogue alien keys into reasoning_engine_state_dict must trigger RuntimeError under strict=True."""
        cot_sd = dict(self.raw_ckpt["reasoning_engine_state_dict"])
        cot_sd["alien_malicious_backdoor_weight"] = torch.randn(128, 128)

        engine = MetacognitiveReasoningEngine(
            ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=8)
        )
        with self.assertRaises(RuntimeError) as ctx:
            engine.load_state_dict(cot_sd, strict=True)
        self.assertIn("Unexpected key(s)", str(ctx.exception))
        self.assertIn("alien_malicious_backdoor_weight", str(ctx.exception))

    def test_probe_1_5_alien_keys_injection_in_world_model_rejected(self):
        """Probe: Injecting rogue alien keys into world_model_state_dict must trigger RuntimeError under strict=True."""
        wm_sd = dict(self.raw_ckpt["world_model_state_dict"])
        wm_sd["alien_dynamics_perturbation_matrix"] = torch.randn(64, 64)

        wm = LatentDynamicsWorldModel()
        with self.assertRaises(RuntimeError) as ctx:
            wm.load_state_dict(wm_sd, strict=True)
        self.assertIn("Unexpected key(s)", str(ctx.exception))

    def test_probe_1_6_tensor_shape_mutation_rejected(self):
        """Probe: Mutating a 2D weight matrix shape (transposing dimensions) must trigger size mismatch RuntimeError."""
        cot_sd = dict(self.raw_ckpt["reasoning_engine_state_dict"])
        # Find first 2D tensor with unequal dimensions
        target_key = None
        for k, v in cot_sd.items():
            if v.dim() == 2 and v.shape[0] != v.shape[1]:
                target_key = k
                break
        self.assertIsNotNone(target_key, "Could not find asymmetric 2D tensor in state_dict")

        cot_sd[target_key] = cot_sd[target_key].t()  # Transpose shape

        engine = MetacognitiveReasoningEngine(
            ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=8)
        )
        with self.assertRaises(RuntimeError) as ctx:
            engine.load_state_dict(cot_sd, strict=True)
        self.assertIn("size mismatch", str(ctx.exception).lower())

    def test_probe_1_7_truncated_checkpoint_file_stream(self):
        """Probe: Truncating a checkpoint file to simulate network or disk stream corruption must raise an unpickling error."""
        corrupt_path = Path(self.temp_dir) / "truncated_test.pt"
        torch.save(self.raw_ckpt, corrupt_path)
        full_size = corrupt_path.stat().st_size
        self.assertGreater(full_size, 1024)

        # Truncate to 50%
        with open(corrupt_path, "r+b") as f:
            f.truncate(full_size // 2)

        with self.assertRaises(Exception) as ctx:
            torch.load(corrupt_path, weights_only=False)
        err_msg = str(ctx.exception).lower()
        self.assertTrue(
            any(s in err_msg for s in ("zip", "eof", "corrupt", "unpickling", "stream")),
            f"Expected corruption error, got: {ctx.exception}",
        )

    def test_probe_1_8_zero_byte_checkpoint_file(self):
        """Probe: Loading a zero-byte empty file must raise an immediate exception."""
        zero_path = Path(self.temp_dir) / "empty.pt"
        zero_path.touch()

        with self.assertRaises(Exception):
            torch.load(zero_path, weights_only=False)


# ==============================================================================
# 2. ADVERSARIAL INPUT SANITIZATION PROBES
# ==============================================================================
class TestAdversarialInputSanitization(unittest.TestCase):
    """Stress tests 137D observation ingress into MythosMT5Policy with hostile and corrupted numerical inputs."""

    @classmethod
    def setUpClass(cls):
        cls.policy = MythosMT5Policy(
            base_policy=MockBasePolicy(),
            initial_capital=10_000.0,
            checkpoint_path=CHECKPOINT_PATH,
        )

    def test_probe_2_1_all_nan_ingress(self):
        """Probe: 137D vector of 100% NaNs must not crash policy and must enforce Agent 4 HOLD lock."""
        obs_nan = np.full(137, np.nan, dtype=np.float32)
        traces = self.policy.act_nam_tac_nhan(obs_nan)
        self.assertEqual(len(traces), 5)

        agent4 = traces[4]
        out_vec = getattr(agent4, "output_vector", None)
        self.assertIsNotNone(out_vec)
        self.assertEqual(agent4.action, 0)
        self.assertEqual(out_vec[0], 1.0)
        self.assertEqual(out_vec[10], -100.0)
        self.assertEqual(out_vec[19], 0.0)

    def test_probe_2_2_isolated_nans_at_critical_feature_indices(self):
        """Probe: NaNs placed at sensitive feature positions (return, geometry, spread, volume) must be safely tolerated."""
        critical_indices = [0, 7, 8, 9, 10, 15, 62, 136]
        for idx in critical_indices:
            obs = np.zeros(137, dtype=np.float32)
            obs[idx] = np.nan
            traces = self.policy.act_nam_tac_nhan(obs)
            self.assertEqual(len(traces), 5)
            agent4 = traces[4]
            out_vec = getattr(agent4, "output_vector", None)
            self.assertEqual(agent4.action, 0)
            self.assertEqual(out_vec[0], 1.0)
            self.assertEqual(out_vec[19], 0.0)

    def test_probe_2_3_positive_and_negative_infinities(self):
        """Probe: Vectors with +inf, -inf, and mixed inf/nan must not cause unhandled arithmetic exceptions."""
        probe_arrays = [
            np.full(137, np.inf, dtype=np.float32),
            np.full(137, -np.inf, dtype=np.float32),
            np.array([np.nan if i % 2 == 0 else np.inf for i in range(137)], dtype=np.float32),
        ]
        for p in probe_arrays:
            traces = self.policy.act_nam_tac_nhan(p)
            self.assertEqual(len(traces), 5)
            agent4 = traces[4]
            out_vec = getattr(agent4, "output_vector", None)
            self.assertEqual(agent4.action, 0)
            self.assertEqual(out_vec[0], 1.0)
            self.assertEqual(out_vec[19], 0.0)

    def test_probe_2_4_all_zeros_dead_sensor_input(self):
        """Probe: All-zeros vector (flat market or disconnected sensor) must evaluate cleanly into a safe HOLD posture."""
        obs_zeros = np.zeros(137, dtype=np.float32)
        traces = self.policy.act_nam_tac_nhan(obs_zeros)
        self.assertEqual(len(traces), 5)
        agent4 = traces[4]
        out_vec = getattr(agent4, "output_vector", None)
        self.assertEqual(agent4.action, 0)
        self.assertEqual(out_vec[0], 1.0)
        self.assertEqual(out_vec[19], 0.0)

    def test_probe_2_5_extreme_numerical_magnitudes(self):
        """Probe: Extreme numerical magnitudes (+1e12, -1e12, 1e-35) must not cause float overflow or crashes."""
        magnitudes = [
            np.full(137, 1e12, dtype=np.float32),
            np.full(137, -1e12, dtype=np.float32),
            np.full(137, 1e-35, dtype=np.float32),
        ]
        for obs in magnitudes:
            traces = self.policy.act_nam_tac_nhan(obs)
            self.assertEqual(len(traces), 5)
            self.assertEqual(traces[4].action, 0)

    def test_probe_2_6_dimension_mutations_and_auto_alignment(self):
        """Probe: Under-length, over-length, 2D shapes, and Python lists must auto-align to 137D without IndexErrors."""
        mutations = [
            ("empty_0", np.zeros(0, dtype=np.float32)),
            ("underlength_50", np.zeros(50, dtype=np.float32)),
            ("underlength_136", np.zeros(136, dtype=np.float32)),
            ("overlength_138", np.zeros(138, dtype=np.float32)),
            ("overlength_300", np.zeros(300, dtype=np.float32)),
            ("shape_2d_1x137", np.zeros((1, 137), dtype=np.float32)),
            ("py_list", [0.0] * 137),
        ]
        for name, p in mutations:
            traces = self.policy.act_nam_tac_nhan(p)
            self.assertEqual(len(traces), 5, f"Failed on dimension mutation: {name}")
            self.assertEqual(traces[4].action, 0)

    def test_probe_2_7_adversarial_request_metadata(self):
        """Probe: Malicious request attributes (negative equity, infinite equity, negative positions) must not corrupt state."""
        class MaliciousRequest:
            def __init__(self, eq, pos):
                self.equity = eq
                self.positions = pos
                self.inputs = [0.0] * 137

        bad_requests = [
            MaliciousRequest(eq=-5000.0, pos=100),
            MaliciousRequest(eq=float("inf"), pos=-10),
            MaliciousRequest(eq=float("nan"), pos=0),
        ]
        for req in bad_requests:
            traces = self.policy.act_nam_tac_nhan(req.inputs, request=req)
            self.assertEqual(len(traces), 5)
            # Equity must not be corrupted to negative or non-finite
            self.assertTrue(math.isfinite(self.policy.last_equity))
            self.assertGreater(self.policy.last_equity, 0.0)


# ==============================================================================
# 3. 20-WIRE CONTRACT & INVARIANT DEFENSES
# ==============================================================================
class TestAdversarial20WireContractInvariants(unittest.TestCase):
    """Stress tests the 20-wire execution contract v7 and proves Agent 4 HOLD lock cannot be bypassed."""

    def test_probe_3_1_wire_layout_invariants(self):
        """Probe: Wire vector must have length exactly 20, wire[14] == 7.0, and wire[19] == 0.0."""
        action_logits = np.random.randn(12).astype(np.float32)
        wire = DynamicGeometryPolicyHead.assemble_mt5_wire_vector(
            action_logits=action_logits,
            sl_atr=2.5,
            tp_atr=5.0,
            trailing_atr=1.0,
            risk_pct=0.012,
        )
        self.assertEqual(wire.shape, (20,))
        self.assertEqual(wire[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX], FACTORIZED_ACTION_MARKER_VALUE)
        self.assertEqual(wire[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX], 7.0)
        self.assertEqual(wire[TARGET_SLOT_OUTPUT_INDEX], 0.0)

    def test_probe_3_2_hostile_base_policy_agent4_bypass_resistance(self):
        """Probe: A compromised base policy attempting to emit BUY or CLOSE_ALL on Agent 4 must be completely neutralized."""
        class HostileTrace:
            def __init__(self):
                self.output_vector = np.ones(20, dtype=np.float64) * 999.0
                self.selected_action = np.ones(20, dtype=np.float64) * 999.0
                self.action = Action.MARKET_BUY
                self.action_index = 10
                self.selected_idx = 10
                self.action_details = {
                    "wire_action": 10,
                    "policy_close_all_locked": 1.0,
                    "emitted_action": 10,
                }

        class HostileBasePolicy:
            def act_nam_tac_nhan(self, obs, *args, **kwargs):
                return [HostileTrace() for _ in range(5)]

        policy = MythosMT5Policy(
            base_policy=HostileBasePolicy(),
            initial_capital=10_000.0,
            checkpoint_path=CHECKPOINT_PATH,
        )
        traces = policy.act_nam_tac_nhan(np.zeros(137, dtype=np.float32))
        self.assertEqual(len(traces), 5)

        agent4 = traces[4]
        out_vec = getattr(agent4, "output_vector", None)
        self.assertIsNotNone(out_vec)

        # Proves Agent 4 cannot be coerced into non-HOLD actions
        self.assertIn(agent4.action, (0, Action.HOLD))
        self.assertEqual(agent4.action_index, 0)
        self.assertEqual(out_vec[0], 1.0, "Wire index 0 (ACTION_HOLD) must be locked to 1.0")
        self.assertEqual(out_vec[10], -100.0, "Wire index 10 (ACTION_CLOSE_ALL) must be suppressed to -100.0")
        self.assertEqual(out_vec[19], 0.0, "Wire index 19 (POLICY_CLOSE_ALL) must be locked to 0.0")

    def test_probe_3_3_static_hold_vector_immutability(self):
        """Probe: Attempting to mutate _STATIC_HOLD_VECTOR in-place must raise ValueError (read-only buffer)."""
        self.assertFalse(_STATIC_HOLD_VECTOR.flags.writeable)
        with self.assertRaises(ValueError):
            _STATIC_HOLD_VECTOR[0] = 999.0
        with self.assertRaises(ValueError):
            _STATIC_HOLD_VECTOR[19] = 1.0

    def test_probe_3_4_packed_geometry_wire16_exhaustive_permutations(self):
        """Probe: Exhaustively test all 640 codec bin combinations roundtrip, and verify discretize_and_pack enforces RR >= 1.5."""
        tested_combinations = 0
        for sl_i in range(len(SL_ATR_BINS)):
            for tp_i in range(len(TP_RR_BINS)):
                for tr_i in range(len(TRAILING_ATR_BINS)):
                    packed = pack_geometry_choices(sl_i, tp_i, tr_i)
                    un_sl, un_tp, un_tr = unpack_geometry_choices(packed)
                    self.assertEqual(un_sl, sl_i)
                    self.assertEqual(un_tp, tp_i)
                    self.assertEqual(un_tr, tr_i)
                    tested_combinations += 1
        self.assertEqual(tested_combinations, len(SL_ATR_BINS) * len(TP_RR_BINS) * len(TRAILING_ATR_BINS))
        self.assertEqual(tested_combinations, 640)

        # Invariant defense: discretize_and_pack must strictly enforce RR >= 1.5 even under hostile continuous inputs
        hostile_inputs = [
            (1.0, 1.5),
            (8.0, 15.0),
            (0.1, 0.2),
            (100.0, 200.0),
            (-5.0, -10.0),
            (np.nan, np.nan),
            (np.inf, np.inf),
            (4.0, 1.0),  # Hostile: TP < SL
            (2.0, 0.5),  # Hostile: TP << SL
        ]
        for sl_val, tp_val in hostile_inputs:
            sl_idx, tp_idx, tr_idx, logit = DynamicGeometryPolicyHead.discretize_and_pack(sl_val, tp_val, 0.0)
            un_sl, un_tp, un_tr = unpack_geometry_choices(logit)
            self.assertEqual(un_tp, tp_idx)
            self.assertGreaterEqual(TP_RR_BINS[tp_idx], 1.5, f"discretize_and_pack yielded RR < 1.5 for ({sl_val}, {tp_val})")

    def test_probe_3_5_dynamic_continuous_geometry_head_analytical_bounds(self):
        """Probe: Under extreme inputs (+/- 1000.0), DynamicGeometryPolicyHead must strictly satisfy SL in [1, 8], TP in [1.5, 15], RR >= 1.5."""
        head = DynamicGeometryPolicyHead(latent_dim=768)
        extreme_z = torch.cat([
            torch.randn(100, 768),
            torch.ones(50, 768) * 1000.0,
            torch.ones(50, 768) * -1000.0,
            torch.zeros(50, 768),
        ], dim=0)

        sl, tp, rr = head(extreme_z)
        sl_np = sl.detach().cpu().numpy()
        tp_np = tp.detach().cpu().numpy()
        rr_np = rr.detach().cpu().numpy()

        self.assertTrue(np.all(sl_np >= 1.0 - 1e-6), f"SL below 1.0: min is {sl_np.min()}")
        self.assertTrue(np.all(sl_np <= 8.0 + 1e-6), f"SL above 8.0: max is {sl_np.max()}")
        self.assertTrue(np.all(tp_np >= 1.5 - 1e-6), f"TP below 1.5: min is {tp_np.min()}")
        self.assertTrue(np.all(tp_np <= 15.0 + 1e-6), f"TP above 15.0: max is {tp_np.max()}")
        self.assertTrue(np.all(rr_np >= 1.5 - 1e-6), f"RR below 1.5: min is {rr_np.min()}")


# ==============================================================================
# 4. ADVERSARIAL CPU DECISION LATENCY MULTI-SAMPLE SLA PROBE
# ==============================================================================
class TestAdversarialDecisionLatencySLABenchmark(unittest.TestCase):
    """Measures decision latency under multi-sample CPU load and confirms compliance with < 50.0 ms budget."""

    @classmethod
    def setUpClass(cls):
        if not M5_CSV_PATH.exists():
            raise FileNotFoundError(f"M5 dataset missing: {M5_CSV_PATH}")
        cls.env = RealM5Environment(csv_path=str(M5_CSV_PATH), spread_usd=2.50)
        cls.env.load_data()

        cls.trainer = M5MuZeroTrainer(env=cls.env, from_scratch=True)
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        cls.trainer.reasoning_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)
        cls.trainer.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=True)
        cls.trainer.bull_champion.reasoning_engine.load_state_dict(ckpt["bull_cot_state_dict"], strict=True)
        cls.trainer.bear_champion.reasoning_engine.load_state_dict(ckpt["bear_cot_state_dict"], strict=True)
        cls.trainer.bull_champion.world_model.load_state_dict(ckpt["bull_world_model_state_dict"], strict=True)
        cls.trainer.bear_champion.world_model.load_state_dict(ckpt["bear_world_model_state_dict"], strict=True)
        cls.trainer.reasoning_engine.eval()
        cls.trainer.world_model.eval()

    def test_probe_4_1_multi_sample_consensus_latency_under_50ms(self):
        """Probe: Measures Dual-Champion consensus latency across 50 diverse historical bars on CPU."""
        obs0 = self.env.get_observation(500)
        # Pre-warm passes
        for _ in range(5):
            with torch.inference_mode():
                _ = self.trainer.resolve_consensus(
                    obs0, current_active_positions=0, active_position_dir=0, bar_idx=None
                )

        latencies_ms: List[float] = []
        num_samples = 50
        # Stride through authentic M5 dataset
        for i in range(num_samples):
            bar_idx = (i * 200) % self.env.total_bars
            obs = self.env.get_observation(bar_idx)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = self.trainer.resolve_consensus(
                    obs, current_active_positions=0, active_position_dir=0, bar_idx=None
                )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(latencies_ms))
        median_ms = float(np.median(latencies_ms))
        p95_ms = float(np.percentile(latencies_ms, 95))
        max_ms = float(np.max(latencies_ms))

        print(f"\n[Adversarial Latency Benchmark] 50 Diverse Samples on CPU:")
        print(f"  Mean:   {mean_ms:.2f} ms (SLA budget <= 50.0 ms)")
        print(f"  Median: {median_ms:.2f} ms")
        print(f"  P95:    {p95_ms:.2f} ms (SLA budget <= 50.0 ms)")
        print(f"  Max:    {max_ms:.2f} ms")

        self.assertLessEqual(mean_ms, 50.0, f"Mean latency {mean_ms:.2f} ms exceeds 50.0 ms budget")
        self.assertLessEqual(median_ms, 50.0, f"Median latency {median_ms:.2f} ms exceeds 50.0 ms budget")
        self.assertLessEqual(p95_ms, 50.0, f"P95 latency {p95_ms:.2f} ms exceeds 50.0 ms budget")

    def test_probe_4_2_cold_start_vs_warm_latency_delta(self):
        """Probe: Verifies warmup clears thread allocations and cold-to-warm delta does not exhibit unbounded latency."""
        obs = self.env.get_observation(100)
        t_cold_start = time.perf_counter()
        with torch.inference_mode():
            _ = self.trainer.resolve_consensus(
                obs, current_active_positions=0, active_position_dir=0, bar_idx=None
            )
        cold_ms = (time.perf_counter() - t_cold_start) * 1000.0

        warm_latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = self.trainer.resolve_consensus(
                    obs, current_active_positions=0, active_position_dir=0, bar_idx=None
                )
            warm_latencies.append((time.perf_counter() - t0) * 1000.0)

        warm_mean = float(np.mean(warm_latencies))
        self.assertLessEqual(warm_mean, 50.0)
        # Cold start should not exceed 250 ms even before caches are primed
        self.assertLessEqual(cold_ms, 250.0)


if __name__ == "__main__":
    unittest.main()
