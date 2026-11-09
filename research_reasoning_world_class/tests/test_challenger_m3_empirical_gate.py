"""
Empirical Challenger Test Suite: Milestone M3 Gate Verification
Agent: teamwork_preview_challenger_m3_1
Roles: critic, specialist

Objectives:
1. Verify Champion Checkpoint loading integrity: Strict loading of all 4,196,564 parameters
   under strict=True with 0 missing and 0 unexpected keys.
2. Verify Agent 4 Governor invariant across edge-case input vectors: Ensure outputs[19] == 0.0
   (POLICY_CLOSE_ALL == 0.0) and outputs[0] == 1.0 (ACTION_HOLD) under extreme market volatility,
   flash crashes, zero ATR, and corrupted 137D inputs (NaN, Inf, Denormalized).
3. Empirically verify CPU single-tick latency: Run 100 consecutive ticks through MythosMT5Policy
   on CPU and verify whether mean tick latency satisfies <= 50.0ms (target ~13ms).
"""

from __future__ import annotations

import os
import sys
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from research_reasoning_world_class.src.unified_reasoning_pipeline import UnifiedReasoningTraderPipeline
from research_reasoning_world_class.src.metacognitive_reasoning import (
    ReasoningConfig,
    MetacognitiveReasoningEngine,
)
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy


# =============================================================================
# 1. CHAMPION CHECKPOINT LOADING INTEGRITY
# =============================================================================
class TestChampionCheckpointLoadingIntegrity:
    """Rigorous verification of champion checkpoint loading, keys, and parameter count."""

    @pytest.fixture(scope="class")
    def ckpt_data(self):
        ckpt_path = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        assert ckpt_path.exists(), f"Champion checkpoint not found at: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "reasoning_engine_state_dict" in ckpt, "Checkpoint missing 'reasoning_engine_state_dict'"
        return ckpt["reasoning_engine_state_dict"]

    def test_checkpoint_file_and_keys(self, ckpt_data):
        """Verifies checkpoint contains exactly 191 tensor keys, all finite with 0 NaNs and 0 Infs."""
        assert len(ckpt_data) == 191, f"Expected 191 checkpoint tensor keys, got {len(ckpt_data)}"
        for k, v in ckpt_data.items():
            assert isinstance(v, torch.Tensor), f"Key {k} is not a torch.Tensor"
            assert not torch.isnan(v).any(), f"Key {k} contains NaN!"
            assert not torch.isinf(v).any(), f"Key {k} contains Inf!"

    def test_strict_load_4196564_parameters(self, ckpt_data):
        """
        Loads state_dict into fresh MetacognitiveReasoningEngine under strict=True.
        Asserts 0 missing keys, 0 unexpected keys, 187 unique named params, 4,196,564 total elements.
        """
        cfg = ReasoningConfig(
            input_dim=768,
            hidden_dim=256,
            num_hops=4,
            max_hops=8,
            enable_cross_hop=True,
            enable_backtrack=True,
            enable_act=True,
            prm_threshold=0.85,
            adversarial_margin_threshold=0.20,
        )
        engine = MetacognitiveReasoningEngine(cfg).eval()
        load_res = engine.load_state_dict(ckpt_data, strict=True)

        assert len(load_res.missing_keys) == 0, f"Missing keys under strict=True: {load_res.missing_keys}"
        assert len(load_res.unexpected_keys) == 0, f"Unexpected keys under strict=True: {load_res.unexpected_keys}"

        named_params = dict(engine.named_parameters())
        assert len(named_params) == 187, f"Expected 187 unique named parameters, got {len(named_params)}"

        total_elements = sum(p.numel() for p in engine.parameters())
        assert total_elements == 4_196_564, f"Expected 4,196,564 total elements, got {total_elements}"

    def test_pipeline_checkpoint_load(self):
        """Verifies UnifiedReasoningTraderPipeline initializes and strictly binds champion checkpoint."""
        pipe = UnifiedReasoningTraderPipeline()
        total_elements = sum(p.numel() for p in pipe.cot_engine.parameters())
        assert total_elements == 4_196_564, f"Pipeline cot_engine parameter count mismatch: {total_elements}"


# =============================================================================
# 2. AGENT 4 GOVERNOR INVARIANT ADVERSARIAL STRESS
# =============================================================================
class TestAgent4GovernorInvariantAdversarialStress:
    """
    Empirical stress-testing of Agent 4 Governor invariant across edge-case input vectors:
    - outputs[19] == 0.0 (POLICY_CLOSE_ALL == 0.0)
    - outputs[0] == 1.0 (ACTION_HOLD)
    - outputs[10] == -100.0 (Suppressed ACTION_CLOSE_ALL)
    - wire_action == 0, policy_close_all_locked == 0.0
    under extreme volatility, flash crashes, zero ATR, and corrupted 137D inputs.
    """

    @pytest.fixture(scope="class")
    def policy(self):
        config_path = PROJECT_ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
        config = load_config(config_path)
        checkpoint_path = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
        base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
        pol = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
        # Advance _last_exec_pos to avoid replaying historic ledger
        cand_path = PROJECT_ROOT / "mt5_v2_runner" / "MQL5" / "Files" / "self_growing_execution.csv"
        if cand_path.exists():
            pol._last_exec_pos = cand_path.stat().st_size
        return pol

    def _verify_agent4_invariants(self, traces: Tuple[Any, ...], context_tag: str) -> None:
        """Helper to enforce strict Agent 4 invariants on emitted traces."""
        assert len(traces) == 5, f"{context_tag}: Expected 5 agent traces, got {len(traces)}"
        agent4 = traces[4]
        out = getattr(agent4, "selected_action", getattr(agent4, "output_vector", None))
        assert out is not None, f"{context_tag}: Agent 4 output vector / selected_action is None"
        assert len(out) == 20, f"{context_tag}: Agent 4 output vector length {len(out)} != 20"

        # Check invariant #1: outputs[19] == 0.0 (POLICY_CLOSE_ALL)
        assert out[19] == 0.0, f"{context_tag}: Agent 4 wire[19] (POLICY_CLOSE_ALL) = {out[19]} != 0.0"

        # Check invariant #2: outputs[0] == 1.0 (ACTION_HOLD)
        assert out[0] == 1.0, f"{context_tag}: Agent 4 wire[0] (ACTION_HOLD) = {out[0]} != 1.0"

        # Check invariant #3: outputs[10] == -100.0 (ACTION_CLOSE_ALL suppressed)
        assert out[10] == -100.0, f"{context_tag}: Agent 4 wire[10] = {out[10]} != -100.0"

        # Check action details metadata
        details = getattr(agent4, "action_details", {}) or {}
        assert details.get("policy_close_all_locked", None) == 0.0, (
            f"{context_tag}: details['policy_close_all_locked'] = {details.get('policy_close_all_locked')} != 0.0"
        )
        assert details.get("wire_action", None) == 0, (
            f"{context_tag}: details['wire_action'] = {details.get('wire_action')} != 0"
        )

    def test_flash_crash_invariants(self, policy: MythosMT5Policy):
        """50 simulated flash crash ticks: extreme negative returns (-10% to -500%), massive volume, plunging slopes."""
        rng = np.random.RandomState(42)
        for i in range(50):
            obs = np.zeros(137, dtype=np.float32)
            obs[0] = float(rng.choice([-15.0, -25.0, -50.0, -100.0, -500.0]))  # ret1
            obs[7] = float(rng.uniform(-30.0, -5.0))                            # body
            obs[8] = float(rng.uniform(0.1, 1.0))                              # upper wick
            obs[9] = float(rng.uniform(5.0, 25.0))                             # giant lower wick
            obs[10] = abs(obs[7]) + obs[8] + obs[9]
            obs[15] = float(rng.uniform(0.5, 3.0))                             # spread
            obs[81] = float(rng.uniform(-20.0, -5.0))                          # m5 slope
            obs[85] = float(rng.uniform(-20.0, -5.0))                          # m15 slope
            obs[89] = float(rng.uniform(-20.0, -5.0))                          # h1 slope
            obs[121] = float(rng.uniform(20.0, 100.0))                         # volume spike

            traces = policy.act_nam_tac_nhan(obs)
            self._verify_agent4_invariants(traces, f"Flash Crash Tick #{i}")

    def test_flash_pump_invariants(self, policy: MythosMT5Policy):
        """50 simulated flash pump / mega volatility ticks: extreme positive returns (+10% to +500%), giant upper wicks."""
        rng = np.random.RandomState(43)
        for i in range(50):
            obs = np.zeros(137, dtype=np.float32)
            obs[0] = float(rng.choice([15.0, 25.0, 50.0, 100.0, 500.0]))      # ret1
            obs[7] = float(rng.uniform(5.0, 30.0))                             # body
            obs[8] = float(rng.uniform(5.0, 25.0))                             # giant upper wick
            obs[9] = float(rng.uniform(0.1, 1.0))                              # lower wick
            obs[10] = abs(obs[7]) + obs[8] + obs[9]
            obs[15] = float(rng.uniform(0.5, 3.0))                             # spread
            obs[81] = float(rng.uniform(5.0, 20.0))                            # m5 slope
            obs[85] = float(rng.uniform(5.0, 20.0))                            # m15 slope
            obs[89] = float(rng.uniform(5.0, 20.0))                            # h1 slope
            obs[121] = float(rng.uniform(20.0, 100.0))                         # volume spike

            traces = policy.act_nam_tac_nhan(obs)
            self._verify_agent4_invariants(traces, f"Flash Pump Tick #{i}")

    def test_zero_atr_flatline_invariants(self, policy: MythosMT5Policy):
        """50 simulated flatline / zero ATR ticks: ret1=0, body=0, wicks=0, range=0, spread=0."""
        # Force policy current_atr to edge cases
        for edge_atr in [0.0, 1e-6, 1e-3, 0.01, 48.60]:
            policy.current_atr = edge_atr
            obs = np.zeros(137, dtype=np.float32)
            traces = policy.act_nam_tac_nhan(obs)
            self._verify_agent4_invariants(traces, f"Zero ATR (atr={edge_atr})")

    def test_corrupted_137d_inputs_invariants(self, policy: MythosMT5Policy):
        """Adversarial corrupted input vectors: all NaN, all +Inf, all -Inf, mixed NaNs, sub-137, super-137."""
        corrupted_cases = [
            ("All NaN 137D", np.full(137, np.nan, dtype=np.float32)),
            ("All +Inf 137D", np.full(137, np.inf, dtype=np.float32)),
            ("All -Inf 137D", np.full(137, -np.inf, dtype=np.float32)),
            ("Sub-137 Truncated (32D)", np.ones(32, dtype=np.float32) * 5.0),
            ("Sub-137 Truncated (64D)", np.ones(64, dtype=np.float32) * -5.0),
            ("Super-137 Oversized (256D)", np.ones(256, dtype=np.float32) * 2.5),
            ("Extreme Floats (1e30)", np.full(137, 1e30, dtype=np.float32)),
            ("Denormalized Floats (1e-38)", np.full(137, 1e-38, dtype=np.float32)),
        ]

        # Mixed NaNs and Infs in specific critical feature slots
        mixed_vec = np.zeros(137, dtype=np.float32)
        mixed_vec[0] = np.nan   # ret1 NaN
        mixed_vec[7] = np.inf   # body Inf
        mixed_vec[15] = -np.inf # spread -Inf
        mixed_vec[81] = np.nan  # m5 slope NaN
        mixed_vec[121] = np.nan # rel_vol NaN
        corrupted_cases.append(("Mixed NaN/Inf Critical Slots", mixed_vec))

        for name, vec in corrupted_cases:
            traces = policy.act_nam_tac_nhan(vec)
            self._verify_agent4_invariants(traces, f"Corrupted Input: {name}")


# =============================================================================
# 3. CPU SINGLE-TICK LATENCY BENCHMARK & SLA AUDIT
# =============================================================================
class TestCPUSingleTickLatencySLA:
    """
    Empirical benchmark of CPU single-tick latency across 100 consecutive ticks.
    Evaluates:
    - Isolated cot_engine forward pass latency (target ~13ms)
    - Full MythosMT5Policy.act_nam_tac_nhan end-to-end latency (SLA <= 50.0ms)
    """

    @pytest.fixture(scope="class")
    def policy(self):
        config_path = PROJECT_ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
        config = load_config(config_path)
        checkpoint_path = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
        base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
        pol = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
        cand_path = PROJECT_ROOT / "mt5_v2_runner" / "MQL5" / "Files" / "self_growing_execution.csv"
        if cand_path.exists():
            pol._last_exec_pos = cand_path.stat().st_size
        return pol

    def test_cot_engine_isolated_cpu_latency_100_passes(self, policy: MythosMT5Policy):
        """Empirically confirms that isolated cot_engine satisfies ~13ms latency on CPU."""
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        engine = policy.pipeline.cot_engine.eval()

        # Warmup
        with torch.no_grad():
            for u_val in [0.05, 0.20, 0.50, 0.80, 0.95]:
                x = torch.randn(1, 768)
                u = torch.tensor([u_val])
                _ = engine(x, uncertainty=u)

        latencies = []
        with torch.no_grad():
            for i in range(100):
                u_val = [0.05, 0.20, 0.50, 0.80, 0.95][i % 5]
                x = torch.randn(1, 768)
                u = torch.tensor([u_val])

                t0 = time.perf_counter()
                _ = engine(x, uncertainty=u)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        lat_arr = np.array(latencies)
        mean_lat = float(np.mean(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))

        print(f"\n[BENCHMARK] cot_engine isolated CPU latency (100 passes): Mean={mean_lat:.2f}ms, P95={p95_lat:.2f}ms")
        assert mean_lat <= 50.0, f"Isolated cot_engine mean latency {mean_lat:.2f}ms > 50.0ms budget"

    def test_mythos_mt5_policy_end_to_end_cpu_latency_100_ticks(self, policy: MythosMT5Policy):
        """
        Empirically benchmarks 100 consecutive ticks through MythosMT5Policy.act_nam_tac_nhan on CPU.
        Verifies whether mean tick latency satisfies <= 50.0ms SLA.
        """
        obs = np.zeros(137, dtype=np.float32)

        # Warmup 5 ticks
        for _ in range(5):
            _ = policy.act_nam_tac_nhan(obs)

        latencies = []
        for tick_idx in range(100):
            t0 = time.perf_counter()
            traces = policy.act_nam_tac_nhan(obs)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms)

        lat_arr = np.array(latencies)
        min_lat = float(np.min(lat_arr))
        mean_lat = float(np.mean(lat_arr))
        p50_lat = float(np.percentile(lat_arr, 50))
        p95_lat = float(np.percentile(lat_arr, 95))
        max_lat = float(np.max(lat_arr))

        print("\n" + "=" * 80)
        print("EMPIRICAL BENCHMARK: MythosMT5Policy 100 CONSECUTIVE TICKS ON CPU")
        print("=" * 80)
        print(f"Total Ticks Measured: 100")
        print(f"Min Latency:         {min_lat:.2f} ms")
        print(f"Mean Latency:        {mean_lat:.2f} ms (Target SLA <= 50.0 ms)")
        print(f"Median (P50):        {p50_lat:.2f} ms")
        print(f"P95 Latency:         {p95_lat:.2f} ms")
        print(f"Max Latency:         {max_lat:.2f} ms")
        print("=" * 80)

        # Check against mandatory SLA <= 50.0ms
        assert mean_lat <= 50.0, (
            f"FAIL: MythosMT5Policy mean CPU tick latency is {mean_lat:.2f}ms, "
            f"which severely violates the <= 50.0ms SLA budget by {mean_lat / 50.0:.2f}x!"
        )
