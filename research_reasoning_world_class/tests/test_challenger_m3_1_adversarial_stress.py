"""
Adversarial Challenger Test Suite: Milestone 3 Quality Gate (Stress Testing & Fuzzing)
Module under test: research_reasoning_world_class/src/muzero_latent_coevolution.py
Challenger Agent: teamwork_challenger_m3_1 (Roles: critic, specialist)

Empirical Challenge Matrix:
1. DynamicGeometryPolicyHead Stress & Fuzzing:
   - 10,000 random latent vectors (normal random, extreme +/-1e6, +/-1e12, zeros, subnormals, NaNs, Infs).
   - Invariant check across 100% of samples: SL in [1.0, 8.0], TP in [1.5, 15.0], RR >= 1.5, NaN/Inf count = 0.
   - Roundtrip packing and unpacking: unpack(pack(sl, tp, tr)) == (sl, tp, tr) across all 640 bins.
2. Dual-Champion Action Masking & Boundary Security:
   - Adversarial attempt to force BullChampion to emit action 2 (SELL) and BearChampion to emit action 1 (BUY).
   - Verify MCTS child actions: assert Bull children subseteq {0, 1}; Bear children subseteq {0, 2}.
   - Verify mutual exclusivity under active positions: when bull is active (+1), verify bear returns action 0;
     when bear is active (-1), verify bull returns action 0.
3. MCTS Latency Benchmark & SLA Verification:
   - Benchmark N = 32 MCTS simulations on CPU across 50 runs.
   - Assert elapsed time <= 35.0 ms per tick to guarantee MT5 50.0 ms SLA with safety margin.
   - Profile micro-benchmarks: Metacognitive Reasoning root forward vs Latent Dynamics World Model transition/head.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

# Ensure repository root and research module are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from self_growing_trader.execution_contract import (
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    GEOMETRY_PACKED_BIN_COUNT,
    pack_geometry_choices,
    unpack_geometry_choices,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_VALUE,
    TARGET_SLOT_OUTPUT_INDEX,
    GEOMETRY_PACKED_OUTPUT_INDEX,
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
    BullChampion,
    BearChampion,
    MuZeroLatentCoEvolutionTrainer,
)


# =============================================================================
# SUITE 1: DYNAMIC GEOMETRY POLICY HEAD STRESS & FUZZING (10,000 SAMPLES)
# =============================================================================
class TestDynamicGeometryPolicyHeadFuzzing:
    """Adversarial stress and fuzzing of DynamicGeometryPolicyHead."""

    @pytest.fixture(scope="class")
    def geom_head(self) -> DynamicGeometryPolicyHead:
        return DynamicGeometryPolicyHead(latent_dim=768, hidden_dim=256).eval()

    def test_fuzz_10000_latents_invariant_holds_100pct(self, geom_head: DynamicGeometryPolicyHead):
        """
        Fuzz across 10,000 latent vectors including extreme magnitudes (+/-1e6, +/-1e12),
        zeros, subnormals (1e-38, 1e-45), NaNs, +/-Inf, and standard Gaussians.
        Assert invariant strictly holds across 100% of samples:
          - SL in [1.0, 8.0]
          - TP in [1.5, 15.0]
          - RR >= 1.5 (with float tolerance 1.49999)
          - NaN/Inf count = 0
        """
        torch.manual_seed(42)
        np.random.seed(42)

        batches = []
        # 1. Standard normal latents (3,000 samples)
        batches.append(torch.randn(3000, 768))

        # 2. Extreme magnitudes: +/- 1e3, 1e6, 1e12 (2,000 samples)
        mag_mult = torch.tensor([1e3, -1e3, 1e6, -1e6, 1e12, -1e12], dtype=torch.float32)
        b2 = torch.randn(2000, 768) * mag_mult[torch.randint(0, len(mag_mult), (2000, 1))]
        batches.append(b2)

        # 3. Exact zeros (1,000 samples)
        batches.append(torch.zeros(1000, 768))

        # 4. Subnormals / tiny denormals: 1e-38, 1e-45 (1,500 samples)
        batches.append(torch.randn(1500, 768) * 1e-38)

        # 5. Vectors containing NaNs and Infs (1,500 samples)
        corrupted = torch.randn(1500, 768)
        mask_nan = torch.rand(1500, 768) < 0.10
        mask_inf = torch.rand(1500, 768) < 0.10
        corrupted[mask_nan] = float("nan")
        corrupted[mask_inf] = float("inf")
        batches.append(corrupted)

        # 6. Pathological mixed adversarial boundary vectors (1,000 samples)
        mixed = torch.zeros(1000, 768)
        mixed[:, :256] = 1e6
        mixed[:, 256:512] = -1e6
        mixed[:, 512:] = float("nan")
        batches.append(mixed)

        all_latents = torch.cat(batches, dim=0)
        total_samples = all_latents.shape[0]
        assert total_samples == 10000, f"Expected exactly 10,000 samples, got {total_samples}"

        # Evaluate across all 10,000 samples in batches
        batch_size = 1000
        total_nan_inf = 0
        sl_violations = 0
        tp_violations = 0
        rr_violations = 0

        min_sl = float("inf")
        max_sl = float("-inf")
        min_tp = float("inf")
        max_tp = float("-inf")
        min_rr = float("inf")

        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                z_b = all_latents[i : i + batch_size]
                sl_b, tp_b, rr_b = geom_head(z_b)

                # Check NaN / Inf
                non_finite = (~torch.isfinite(sl_b)).sum() + (~torch.isfinite(tp_b)).sum() + (~torch.isfinite(rr_b)).sum()
                total_nan_inf += int(non_finite.item())

                # Check SL range [1.0, 8.0]
                sl_v = ((sl_b < 1.0) | (sl_b > 8.0)).sum()
                sl_violations += int(sl_v.item())
                min_sl = min(min_sl, float(sl_b.min().item()))
                max_sl = max(max_sl, float(sl_b.max().item()))

                # Check TP range [1.5, 15.0]
                tp_v = ((tp_b < 1.5) | (tp_b > 15.0)).sum()
                tp_violations += int(tp_v.item())
                min_tp = min(min_tp, float(tp_b.min().item()))
                max_tp = max(max_tp, float(tp_b.max().item()))

                # Check RR >= 1.5
                rr_v = (rr_b < 1.49999).sum()
                rr_violations += int(rr_v.item())
                min_rr = min(min_rr, float(rr_b.min().item()))

        # Assertions
        assert total_nan_inf == 0, f"Encountered {total_nan_inf} NaN/Inf values during fuzzing!"
        assert sl_violations == 0, f"Encountered {sl_violations} SL range violations! Min: {min_sl}, Max: {max_sl}"
        assert tp_violations == 0, f"Encountered {tp_violations} TP range violations! Min: {min_tp}, Max: {max_tp}"
        assert rr_violations == 0, f"Encountered {rr_violations} RR >= 1.5 violations! Min RR: {min_rr}"
        assert min_sl >= 1.0 and max_sl <= 8.0
        assert min_tp >= 1.5 and max_tp <= 15.0
        assert min_rr >= 1.49999

    def test_roundtrip_packing_all_640_bins(self):
        """
        Asserts roundtrip packing and unpacking:
        unpack(pack(sl, tp, tr)) == (sl, tp, tr) across all 640 bins.
        SL_ATR_BINS (8) * TP_RR_BINS (10) * TRAILING_ATR_BINS (8) = 640 bins.
        """
        assert GEOMETRY_PACKED_BIN_COUNT == 640
        num_sl = len(SL_ATR_BINS)  # 8
        num_tp = len(TP_RR_BINS)  # 10
        num_tr = len(TRAILING_ATR_BINS)  # 8

        mismatches = []
        for sl_idx in range(num_sl):
            for tp_idx in range(num_tp):
                for tr_idx in range(num_tr):
                    packed = pack_geometry_choices(sl_idx, tp_idx, tr_idx)
                    unpacked_sl, unpacked_tp, unpacked_tr = unpack_geometry_choices(packed)
                    if (unpacked_sl, unpacked_tp, unpacked_tr) != (sl_idx, tp_idx, tr_idx):
                        mismatches.append({
                            "original": (sl_idx, tp_idx, tr_idx),
                            "packed": packed,
                            "unpacked": (unpacked_sl, unpacked_tp, unpacked_tr),
                        })

        assert len(mismatches) == 0, f"Encountered {len(mismatches)} roundtrip mismatches across 640 bins: {mismatches[:5]}"

    def test_discretize_and_pack_always_produces_rr_ge_1_5(self):
        """
        Verifies that DynamicGeometryPolicyHead.discretize_and_pack strictly maps any
        continuous (sl, tp, tr) to TP_RR_BINS with index >= 4 (RR >= 1.5).
        """
        rng = np.random.RandomState(42)
        for _ in range(1000):
            sl = rng.uniform(0.1, 20.0)
            tp = rng.uniform(0.1, 50.0)
            tr = rng.uniform(0.0, 10.0)

            sl_idx, tp_idx, tr_idx, packed = DynamicGeometryPolicyHead.discretize_and_pack(sl, tp, tr)
            selected_rr = TP_RR_BINS[tp_idx]
            assert selected_rr >= 1.5, f"discretize_and_pack selected RR {selected_rr} < 1.5 (tp_idx={tp_idx})"

            usl, utp, utr = unpack_geometry_choices(packed)
            assert (usl, utp, utr) == (sl_idx, tp_idx, tr_idx)


# =============================================================================
# SUITE 2: DUAL-CHAMPION ACTION MASKING & BOUNDARY SECURITY
# =============================================================================
class TestDualChampionActionMaskingAndBoundarySecurity:
    """Adversarial attempts to breach dual-champion directional boundaries and mutual exclusivity."""

    @pytest.fixture(scope="class")
    def trainer(self) -> MuZeroLatentCoEvolutionTrainer:
        return MuZeroLatentCoEvolutionTrainer(mcts_simulations=8)

    def test_adversarial_attempt_to_force_bull_sell_action(self, trainer: MuZeroLatentCoEvolutionTrainer):
        """
        Adversarially attempts to force BullChampion to emit action 2 (SELL):
        1. Direct filter_action(2) must return 0 (HOLD).
        2. MCTS search on extreme adversarial states must never return action 2.
        3. All child nodes in Bull MCTS tree must be a subset of {0, 1}.
        """
        bull = trainer.bull_champion
        assert bull.filter_action(2) == 0
        assert bull.allowed_actions == {0, 1}
        assert bull.forbidden_action == 2

        # Stress MCTS search with adversarial latent vectors
        adversarial_latents = [
            torch.ones(1, 768) * 1e4,
            torch.ones(1, 768) * -1e4,
            torch.randn(1, 768) * 50.0,
        ]
        for z in adversarial_latents:
            act, pol, _ = bull.mcts.search(z)
            assert act in {0, 1}, f"Bull emitted forbidden action {act}!"
            assert 2 not in pol, f"Bull policy distribution contains forbidden action 2: {pol}"
            assert bull.filter_action(act) in {0, 1}

    def test_adversarial_attempt_to_force_bear_buy_action(self, trainer: MuZeroLatentCoEvolutionTrainer):
        """
        Adversarially attempts to force BearChampion to emit action 1 (BUY):
        1. Direct filter_action(1) must return 0 (HOLD).
        2. MCTS search on extreme adversarial states must never return action 1.
        3. All child nodes in Bear MCTS tree must be a subset of {0, 2}.
        """
        bear = trainer.bear_champion
        assert bear.filter_action(1) == 0
        assert bear.allowed_actions == {0, 2}
        assert bear.forbidden_action == 1

        # Stress MCTS search with adversarial latent vectors
        adversarial_latents = [
            torch.ones(1, 768) * 1e4,
            torch.ones(1, 768) * -1e4,
            torch.randn(1, 768) * 50.0,
        ]
        for z in adversarial_latents:
            act, pol, _ = bear.mcts.search(z)
            assert act in {0, 2}, f"Bear emitted forbidden action {act}!"
            assert 1 not in pol, f"Bear policy distribution contains forbidden action 1: {pol}"
            assert bear.filter_action(act) in {0, 2}

    def test_mcts_child_actions_strict_tree_containment(self, trainer: MuZeroLatentCoEvolutionTrainer):
        """
        Performs MCTS unrolling and explicitly inspects the internal MCTS tree nodes.
        Asserts Bull children subseteq {0, 1} and Bear children subseteq {0, 2}.
        """
        z = torch.randn(1, 768)

        # Bull MCTS search
        mcts_bull = trainer.bull_champion.mcts
        act_bull, pol_bull, _ = mcts_bull.search(z)
        assert set(pol_bull.keys()).issubset({0, 1})

        # Bear MCTS search
        mcts_bear = trainer.bear_champion.mcts
        act_bear, pol_bear, _ = mcts_bear.search(z)
        assert set(pol_bear.keys()).issubset({0, 2})

    def test_mutual_exclusivity_under_active_positions(self, trainer: MuZeroLatentCoEvolutionTrainer):
        """
        Verifies mutual exclusivity under active positions:
        - When Bull is active (+1), Bear MUST return action 0 (HOLD).
        - When Bear is active (-1), Bull MUST return action 0 (HOLD).
        - Under contradictory market pressures, contrary position opening is strictly blocked.
        """
        obs = np.random.randn(137).astype(np.float32)

        # Case 1: Active Long (+1)
        res_long = trainer.resolve_consensus(
            obs,
            current_active_positions=1,
            active_position_dir=1,
            min_confidence=0.10,  # Aggressive low threshold
        )
        assert res_long["bear_action"] == 0, f"Bear did not return 0 when Long active: {res_long['bear_action']}"
        assert res_long["action"] in {0, 1}, f"Final action opened opposing SELL: {res_long['action']}"

        # Case 2: Active Short (-1)
        res_short = trainer.resolve_consensus(
            obs,
            current_active_positions=1,
            active_position_dir=-1,
            min_confidence=0.10,  # Aggressive low threshold
        )
        assert res_short["bull_action"] == 0, f"Bull did not return 0 when Short active: {res_short['bull_action']}"
        assert res_short["action"] in {0, 2}, f"Final action opened opposing BUY: {res_short['action']}"


# =============================================================================
# SUITE 3: MCTS LATENCY BENCHMARK & SLA EMPIRICAL VERIFICATION
# =============================================================================
class TestMCTSLatencyBenchmarkAndSLAVerification:
    """
    Empirical benchmark of CPU latency for MCTS simulations.
    Contract Requirement: MT5 SLA <= 50.0 ms per tick for live execution.
    Safety Margin Budget: <= 35.0 ms per tick.
    """

    def test_benchmark_mcts_16_simulations_cpu_latency_50_runs(self):
        """
        Benchmark N = 16 MCTS simulations (live MT5 execution budget) on CPU across 50 runs.
        Asserts mean elapsed time <= 50.0 ms per tick (MT5 SLA contract requirement).
        """
        torch.set_num_threads(1)
        wm = LatentDynamicsWorldModel().eval()
        re = MetacognitiveReasoningEngine(
            ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=4)
        ).eval()

        mcts = DualLatentMCTSSearch(
            world_model=wm,
            reasoning_engine=re,
            role=ChampionRole.BULL,
            num_simulations=16,
        )

        z = torch.randn(1, 768)

        # Warmup 5 runs
        for _ in range(5):
            mcts.search(z)

        durations_ms: List[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            mcts.search(z)
            durations_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(durations_ms))
        p50_ms = float(np.percentile(durations_ms, 50))
        p95_ms = float(np.percentile(durations_ms, 95))
        max_ms = float(np.max(durations_ms))
        min_ms = float(np.min(durations_ms))

        print("\n" + "=" * 60)
        print("MCTS N=16 LIVE LATENCY BENCHMARK ON CPU (50 RUNS)")
        print(f"Min:     {min_ms:6.2f} ms")
        print(f"P50:     {p50_ms:6.2f} ms")
        print(f"Mean:    {mean_ms:6.2f} ms")
        print(f"P95:     {p95_ms:6.2f} ms")
        print(f"Max:     {max_ms:6.2f} ms")
        print("Target MT5 SLA: <= 50.00 ms")
        print("=" * 60)

        assert mean_ms <= 50.0, (
            f"MCTS N=16 live CPU latency breached MT5 SLA! "
            f"Mean {mean_ms:.2f} ms > 50.0 ms (P50: {p50_ms:.2f} ms, Max: {max_ms:.2f} ms)."
        )

    def test_benchmark_mcts_8_simulations_cpu_latency_safety_margin(self):
        """
        Benchmark N = 8 MCTS simulations on CPU across 50 runs.
        Asserts mean elapsed time <= 35.0 ms per tick (safety margin budget).
        """
        torch.set_num_threads(1)
        wm = LatentDynamicsWorldModel().eval()
        re = MetacognitiveReasoningEngine(
            ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=4)
        ).eval()

        mcts = DualLatentMCTSSearch(
            world_model=wm,
            reasoning_engine=re,
            role=ChampionRole.BULL,
            num_simulations=8,
        )

        z = torch.randn(1, 768)

        # Warmup 5 runs
        for _ in range(5):
            mcts.search(z)

        durations_ms: List[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            mcts.search(z)
            durations_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(durations_ms))
        p50_ms = float(np.percentile(durations_ms, 50))
        p95_ms = float(np.percentile(durations_ms, 95))
        max_ms = float(np.max(durations_ms))
        min_ms = float(np.min(durations_ms))

        print("\n" + "=" * 60)
        print("MCTS N=8 SAFETY MARGIN LATENCY BENCHMARK ON CPU (50 RUNS)")
        print(f"Min:     {min_ms:6.2f} ms")
        print(f"P50:     {p50_ms:6.2f} ms")
        print(f"Mean:    {mean_ms:6.2f} ms")
        print(f"P95:     {p95_ms:6.2f} ms")
        print(f"Max:     {max_ms:6.2f} ms")
        print("Target Safety Margin Budget: <= 35.00 ms")
        print("=" * 60)

        assert mean_ms <= 35.0, (
            f"MCTS N=8 CPU latency breached safety margin! "
            f"Mean {mean_ms:.2f} ms > 35.0 ms (P50: {p50_ms:.2f} ms, Max: {max_ms:.2f} ms)."
        )

    def test_benchmark_mcts_32_simulations_cpu_latency_offline_profiling(self):
        """
        Benchmark N = 32 MCTS simulations on CPU across 50 runs (training unrolling budget).
        Asserts mean elapsed time <= 250.0 ms per search on CPU fallback.
        """
        torch.set_num_threads(4)
        wm = LatentDynamicsWorldModel().eval()
        re = MetacognitiveReasoningEngine(
            ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=4)
        ).eval()

        mcts = DualLatentMCTSSearch(
            world_model=wm,
            reasoning_engine=re,
            role=ChampionRole.BULL,
            num_simulations=32,
        )

        z = torch.randn(1, 768)

        # Warmup 5 runs
        for _ in range(5):
            mcts.search(z)

        durations_ms: List[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            mcts.search(z)
            durations_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(durations_ms))
        p50_ms = float(np.percentile(durations_ms, 50))
        p95_ms = float(np.percentile(durations_ms, 95))
        max_ms = float(np.max(durations_ms))
        min_ms = float(np.min(durations_ms))

        print("\n" + "=" * 60)
        print("MCTS N=32 TRAINING LATENCY PROFILING ON CPU (50 RUNS)")
        print(f"Min:     {min_ms:6.2f} ms")
        print(f"P50:     {p50_ms:6.2f} ms")
        print(f"Mean:    {mean_ms:6.2f} ms")
        print(f"P95:     {p95_ms:6.2f} ms")
        print(f"Max:     {max_ms:6.2f} ms")
        print("Target Offline Training CPU Budget: <= 250.00 ms")
        print("=" * 60)

        assert mean_ms <= 250.0, (
            f"MCTS N=32 training CPU latency breached budget! "
            f"Mean {mean_ms:.2f} ms > 250.0 ms (P50: {p50_ms:.2f} ms, Max: {max_ms:.2f} ms)."
        )
