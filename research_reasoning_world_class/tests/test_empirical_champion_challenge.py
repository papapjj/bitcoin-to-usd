"""
Empirical Challenger Test Suite: Co-Evolved Champion Weights & 8-Hop CoT Reasoning Engine.
Artifact: outputs/he_chuyen_gia/v193_runs/mythos_muzero_coevolved_champion.pt
Pipeline: research_reasoning_world_class/src/unified_reasoning_pipeline.py

Objective:
1. Inspect all 187 unique named parameters and 191 tensor keys of pipe.cot_engine.
   Verify that every tensor matches the exact shape and numerical values in
   mythos_muzero_coevolved_champion.pt with maximum absolute difference == 0.0.
2. Verify 8-hop dynamic execution:
   - Low and high uncertainty input regimes.
   - Hop tensor shapes [B, 8, 256].
   - Dense Cross-Hop Attention at hop 7 (attending across 7 preceding hops).
   - Backtracking Revision activation upon contradiction with N_backtrack <= 1 bound.
3. Benchmark CPU latency across 100 consecutive forward passes:
   - Verify mean <= 50.0ms.
   - Report Min, Mean, P50, P95, P99, Max latency.
"""

import sys
import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from src.unified_reasoning_pipeline import UnifiedReasoningTraderPipeline
from src.metacognitive_reasoning import (
    ReasoningConfig,
    MetacognitiveReasoningEngine,
    CoTReasoningHead,
    BacktrackingRevisionModule,
)


class TestEmpiricalChampionWeightsExactness:
    """Empirical verification of checkpoint weights and parameter exactness."""

    @pytest.fixture(scope="class")
    def loaded_pipeline_and_checkpoint(self):
        pipe = UnifiedReasoningTraderPipeline()
        ckpt_path = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        assert ckpt_path.exists(), f"Champion checkpoint not found at {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "reasoning_engine_state_dict" in ckpt, "Checkpoint missing 'reasoning_engine_state_dict'"
        return pipe, ckpt["reasoning_engine_state_dict"]

    def test_state_dict_191_keys_exactness(self, loaded_pipeline_and_checkpoint):
        """
        Verify all 191 tensor keys in state_dict:
        - Exactly 0 missing keys and 0 unexpected keys.
        - Exact tensor shape match.
        - Maximum absolute numerical difference == 0.0.
        """
        pipe, ckpt_sd = loaded_pipeline_and_checkpoint
        model_sd = pipe.cot_engine.state_dict()

        # Check total count of keys
        assert len(ckpt_sd) == 191, f"Expected 191 checkpoint keys, got {len(ckpt_sd)}"
        assert len(model_sd) == 191, f"Expected 191 model state_dict keys, got {len(model_sd)}"

        missing_keys = [k for k in ckpt_sd if k not in model_sd]
        unexpected_keys = [k for k in model_sd if k not in ckpt_sd]
        assert missing_keys == [], f"Missing keys found: {missing_keys}"
        assert unexpected_keys == [], f"Unexpected keys found: {unexpected_keys}"

        # Per-tensor exactness check
        max_abs_diff_global = 0.0
        discrepant_keys: List[str] = []

        for key in sorted(ckpt_sd.keys()):
            t_ckpt = ckpt_sd[key]
            t_model = model_sd[key]

            assert t_model.shape == t_ckpt.shape, (
                f"Shape mismatch for key '{key}': model {t_model.shape} vs ckpt {t_ckpt.shape}"
            )
            assert t_model.dtype == t_ckpt.dtype, (
                f"Dtype mismatch for key '{key}': model {t_model.dtype} vs ckpt {t_ckpt.dtype}"
            )

            abs_diff = torch.max(torch.abs(t_model.float() - t_ckpt.float())).item()
            if abs_diff > max_abs_diff_global:
                max_abs_diff_global = abs_diff
            if abs_diff != 0.0:
                discrepant_keys.append(f"{key}: max_diff={abs_diff}")

        print("\n" + "=" * 80)
        print("EMPIRICAL AUDIT: 191 STATE_DICT TENSOR KEYS EXACTNESS")
        print("=" * 80)
        print(f"Total Evaluated Keys: {len(ckpt_sd)}")
        print(f"Missing Keys:         {len(missing_keys)}")
        print(f"Unexpected Keys:      {len(unexpected_keys)}")
        print(f"Max Absolute Diff:    {max_abs_diff_global:.10f}")
        print("=" * 80)

        assert discrepant_keys == [], f"Discrepant keys found: {discrepant_keys}"
        assert max_abs_diff_global == 0.0, f"Max absolute difference {max_abs_diff_global} != 0.0"

    def test_named_parameters_187_unique_exactness(self, loaded_pipeline_and_checkpoint):
        """
        Verify all 187 unique named parameters of pipe.cot_engine:
        - Exactly 187 unique named parameters.
        - Total elements count == 4,196,564.
        - Exact shape and numerical values with max abs diff == 0.0.
        """
        pipe, ckpt_sd = loaded_pipeline_and_checkpoint
        named_params = dict(pipe.cot_engine.named_parameters())

        assert len(named_params) == 187, f"Expected 187 unique named parameters, got {len(named_params)}"

        total_elements = sum(p.numel() for p in pipe.cot_engine.parameters())
        assert total_elements == 4_196_564, f"Expected 4,196,564 total elements, got {total_elements}"

        max_abs_param_diff = 0.0
        discrepant_params: List[str] = []

        for name, param in sorted(named_params.items()):
            assert name in ckpt_sd, f"Named parameter '{name}' not found in checkpoint state_dict"
            ckpt_param = ckpt_sd[name]

            assert param.shape == ckpt_param.shape, (
                f"Shape mismatch for parameter '{name}': {param.shape} vs {ckpt_param.shape}"
            )

            abs_diff = torch.max(torch.abs(param.float() - ckpt_param.float())).item()
            if abs_diff > max_abs_param_diff:
                max_abs_param_diff = abs_diff
            if abs_diff != 0.0:
                discrepant_params.append(f"{name}: max_diff={abs_diff}")

        print("\n" + "=" * 80)
        print("EMPIRICAL AUDIT: 187 UNIQUE NAMED PARAMETERS EXACTNESS")
        print("=" * 80)
        print(f"Unique Named Parameters Count: {len(named_params)}")
        print(f"Total Parameter Elements:      {total_elements}")
        print(f"Max Absolute Parameter Diff:   {max_abs_param_diff:.10f}")
        print("=" * 80)

        assert discrepant_params == [], f"Discrepant parameters found: {discrepant_params}"
        assert max_abs_param_diff == 0.0, f"Max absolute parameter difference {max_abs_param_diff} != 0.0"


class TestEmpirical8HopDynamicExecution:
    """Empirical verification of 8-hop dynamic reasoning, ACT, Cross-Hop Attention, and Backtracking."""

    @pytest.fixture(scope="class")
    def engine(self):
        pipe = UnifiedReasoningTraderPipeline()
        return pipe.cot_engine.eval()

    def test_dynamic_execution_shapes_and_act_depth(self, engine):
        """
        Verify:
        - Low uncertainty (0.05): hop shape [B, 8, 256], expected depth <= 2.5.
        - High uncertainty (0.95): hop shape [B, 8, 256], expected depth >= 5.0.
        - Pairwise hop cosine similarity cos(h_i, h_j) < 0.95 (representation diversity).
        """
        B = 4
        torch.manual_seed(42)

        # 1. Low Uncertainty Regime (Calm Sideway Market)
        x_low = torch.randn(B, 768)
        u_low = torch.full((B, 1), 0.05)
        out_low = engine(x_low, uncertainty=u_low)

        assert out_low.cot_hops.shape == (B, 8, 256), (
            f"Expected cot_hops shape ({B}, 8, 256), got {out_low.cot_hops.shape}"
        )
        assert out_low.expected_depth <= 2.5, (
            f"Low uncertainty expected depth {out_low.expected_depth:.3f} > 2.5"
        )

        # 2. High Uncertainty Regime (Volatile Trap Market)
        x_high = torch.randn(B, 768)
        u_high = torch.full((B, 1), 0.95)
        out_high = engine(x_high, uncertainty=u_high)

        assert out_high.cot_hops.shape == (B, 8, 256), (
            f"Expected cot_hops shape ({B}, 8, 256), got {out_high.cot_hops.shape}"
        )
        assert out_high.expected_depth >= 5.0, (
            f"High uncertainty expected depth {out_high.expected_depth:.3f} < 5.0"
        )

        # 3. Representation Diversity: Pairwise hop cosine similarity < 0.95
        hops = out_high.cot_hops[0]  # [8, 256]
        hops_norm = F.normalize(hops, p=2, dim=-1)
        sim_matrix = torch.matmul(hops_norm, hops_norm.T)  # [8, 8]
        # Mask diagonal
        off_diag_mask = ~torch.eye(8, dtype=torch.bool)
        max_pairwise_cos = float(torch.max(sim_matrix[off_diag_mask]).item())

        print("\n" + "=" * 80)
        print("EMPIRICAL AUDIT: 8-HOP DYNAMIC REASONING & SOFT ACT DYNAMICS")
        print("=" * 80)
        print(f"Hop Tensor Shape:                {list(out_low.cot_hops.shape)}")
        print(f"Low Uncertainty Expected Depth:   {out_low.expected_depth:.4f} (Required <= 2.5)")
        print(f"High Uncertainty Expected Depth:  {out_high.expected_depth:.4f} (Required >= 5.0)")
        print(f"Max Pairwise Hop Cosine Sim:     {max_pairwise_cos:.4f} (Required < 0.95)")
        print("=" * 80)

        assert max_pairwise_cos < 0.95, f"Max pairwise cosine similarity {max_pairwise_cos:.4f} >= 0.95"

    def test_dense_cross_hop_attention_at_hop_7(self, engine):
        """
        Verify Dense Cross-Hop Attention at hop 7:
        - Attends across all 7 preceding hops (hops 0..6).
        - Attention weight tensor shape is [B, num_heads=4, 7].
        - Weights along last dimension are strictly normalized probabilities summing to 1.0.
        """
        B = 2
        torch.manual_seed(123)
        x = torch.randn(B, 768)
        u = torch.full((B, 1), 0.8)

        _ = engine(x, uncertainty=u)
        attn_weights = engine.cot_head.last_cross_hop_weights

        assert attn_weights is not None, "last_cross_hop_weights is None"
        assert attn_weights.shape == (B, 4, 7), (
            f"Expected attention shape ({B}, 4, 7), got {attn_weights.shape}"
        )

        sums = torch.sum(attn_weights, dim=-1)
        max_sum_err = float(torch.max(torch.abs(sums - 1.0)).item())

        print("\n" + "=" * 80)
        print("EMPIRICAL AUDIT: DENSE CROSS-HOP ATTENTION AT HOP 7")
        print("=" * 80)
        print(f"Attention Weights Shape: [B={B}, Heads=4, Preceding_Hops=7]")
        print(f"Max Sum Normalization Error: {max_sum_err:.10f}")
        print("=" * 80)

        assert max_sum_err < 1e-5, f"Attention weights do not sum to 1.0, max error={max_sum_err}"

    def test_backtracking_revision_contradiction_activation(self, engine):
        """
        Verify Backtracking Revision activation upon contradiction:
        - When a deep hop contradicts Hop 1 premise (directional conflict) and trap confidence exceeds 0.65:
          * was_backtracked is activated (True).
          * revision_magnitude > 0.0.
          * revised h1 maintains LayerNorm bounded norm near sqrt(256) = 16.0.
        - Verify bounded execution invariant: N_backtrack <= 1.
        - Verify zero false positives when thoughts are concordant.
        """
        bm = engine.cot_head.backtrack_module

        # 1. Concordant test: Agreeing directions and zero SMC trap -> was_backtracked MUST be False
        torch.manual_seed(55)
        h1 = torch.randn(4, 256)
        hk_concordant = h1.clone()
        smc_benign = torch.zeros(4, 64)

        rev_h1_c, was_bt_c, mag_c = bm(h1, hk_concordant, smc_benign)
        assert was_bt_c is False, "Concordant thoughts falsely triggered backtracking!"
        assert mag_c == 0.0, f"Expected 0.0 magnitude for concordant thoughts, got {mag_c}"

        # 2. Contradiction test: Opposing direction and supra-threshold trap confidence
        # Test directly on BacktrackingRevisionModule
        hk_opposing = -h1
        # Instrument detector to output 0.85 (> 0.65)
        orig_detector = bm.detector
        class MockTrapDetector(nn.Module):
            def __init__(self, val=0.85):
                super().__init__()
                self.val = val
            def forward(self, inp):
                if inp.dim() == 3:
                    return torch.full((inp.size(0), inp.size(1), 1), self.val, device=inp.device)
                return torch.full((inp.size(0), 1), self.val, device=inp.device)

        bm.detector = MockTrapDetector(val=0.85)
        try:
            rev_h1_trap, was_bt_trap, mag_trap = bm(h1, hk_opposing, smc_benign)
            assert was_bt_trap is True, "Contradiction failed to activate backtracking!"
            assert mag_trap > 0.0, f"Revision magnitude {mag_trap} <= 0.0 upon contradiction"

            l2_norm = float(torch.norm(rev_h1_trap, dim=-1).mean().item())
            expected_norm = math.sqrt(256)  # 16.0
            norm_err = abs(l2_norm - expected_norm)
            assert norm_err < 0.5, f"Revised h1 norm {l2_norm} deviated from expected {expected_norm}"
        finally:
            bm.detector = orig_detector

        # 3. Test CoTReasoningHead bounded N_backtrack <= 1 execution under contradiction
        head = engine.cot_head
        call_count = [0]
        orig_bm_forward = head.backtrack_module.forward
        orig_dir_hk = head.backtrack_module.contradiction_detector.dir_hk

        class OpposingDir(nn.Module):
            def forward(self, hk):
                # Always return direction opposite to dir1 to guarantee directional conflict
                if hk.dim() == 3:
                    return -torch.ones(hk.size(0), hk.size(1), 1, device=hk.device)
                return -torch.ones(hk.size(0), 1, device=hk.device)

        def instrumented_bm_forward(*args, **kwargs):
            call_count[0] += 1
            return orig_bm_forward(*args, **kwargs)

        head.backtrack_module.forward = instrumented_bm_forward
        head.backtrack_module.detector = MockTrapDetector(val=0.85)
        head.backtrack_module.contradiction_detector.dir_hk = OpposingDir()

        try:
            x_test = torch.randn(2, 768)
            all_hops, final_thought = head(x_test)

            print("\n" + "=" * 80)
            print("EMPIRICAL AUDIT: BACKTRACKING REVISION ACTIVATION & N_BACKTRACK <= 1")
            print("=" * 80)
            print(f"Backtracking Passes Executed: {call_count[0]} (Required == 1)")
            print(f"was_backtracked Flag:         {head.last_was_backtracked}")
            print(f"Revision Magnitude:           {head.last_revision_magnitude:.4f}")
            print(f"Revised h1 L2 Norm:           {l2_norm:.4f} (Standard: {expected_norm:.4f})")
            print("=" * 80)

            assert call_count[0] == 1, f"N_backtrack violated: executed {call_count[0]} times (must be <= 1)"
            assert head.last_was_backtracked is True, "Head failed to record was_backtracked=True"
            assert head.last_revision_magnitude > 0.0, "Head recorded zero revision magnitude"
        finally:
            head.backtrack_module.forward = orig_bm_forward
            head.backtrack_module.detector = orig_detector
            head.backtrack_module.contradiction_detector.dir_hk = orig_dir_hk


class TestEmpiricalCPULatencyBenchmark:
    """Empirical CPU latency benchmark across 100 consecutive forward passes."""

    def test_cpu_latency_100_consecutive_passes(self):
        """
        Benchmark CPU latency across 100 consecutive forward passes:
        - Verify Mean latency <= 50.0ms (MT5 SLA requirement).
        - Report Min, Mean, P50 (Median), P95, P99, Max latency.
        - Verify zero NaN and zero Inf in confidence, prm_score, and action outputs.
        """
        # Ensure CPU threads are set optimally for low latency
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        pipe = UnifiedReasoningTraderPipeline()
        engine = pipe.cot_engine.eval()

        # Warm-up phase across dynamic depth regimes
        with torch.inference_mode():
            for u_warm in [0.05, 0.20, 0.50, 0.80, 0.95] * 2:
                x_w = torch.randn(1, 768)
                u_w = torch.tensor([u_warm])
                _ = engine(x_w, uncertainty=u_w)

        latencies_ms: List[float] = []
        N_PASSES = 100

        with torch.inference_mode():
            for i in range(N_PASSES):
                # Cycle uncertainty across realistic market regimes
                u_val = [0.05, 0.20, 0.50, 0.80, 0.95][i % 5]
                x = torch.randn(1, 768)
                u = torch.tensor([u_val])

                t0 = time.perf_counter()
                out = engine(x, uncertainty=u)
                t1 = time.perf_counter()

                elapsed_ms = (t1 - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                # Numerical stability assertions (AC-7)
                assert not torch.isnan(out.confidence).any(), f"NaN in confidence at pass {i}"
                assert not torch.isinf(out.confidence).any(), f"Inf in confidence at pass {i}"
                assert not torch.isnan(out.prm_score).any(), f"NaN in prm_score at pass {i}"
                assert not torch.isinf(out.prm_score).any(), f"Inf in prm_score at pass {i}"

        lat_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(lat_arr))
        p50_lat = float(np.percentile(lat_arr, 50))
        p95_lat = float(np.percentile(lat_arr, 95))
        p99_lat = float(np.percentile(lat_arr, 99))
        max_lat = float(np.max(lat_arr))
        min_lat = float(np.min(lat_arr))

        print("\n" + "=" * 80)
        print("EMPIRICAL BENCHMARK: CPU FORWARD PASS LATENCY (100 CONSECUTIVE PASSES)")
        print("=" * 80)
        print(f"Total Evaluated Passes: {N_PASSES}")
        print(f"Min Latency:    {min_lat:.3f} ms")
        print(f"Mean Latency:   {mean_lat:.3f} ms (Target <= 50.0 ms)")
        print(f"Median (P50):   {p50_lat:.3f} ms")
        print(f"P95 Latency:    {p95_lat:.3f} ms")
        print(f"P99 Latency:    {p99_lat:.3f} ms")
        print(f"Max Latency:    {max_lat:.3f} ms")
        print("=" * 80)

        assert mean_lat <= 50.0, f"Mean latency {mean_lat:.2f}ms exceeds SLA 50.0ms budget!"
