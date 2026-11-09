"""
Adversarial Stress Test Suite for Solution 3: Temporal Cross-Attention (TCA) & KV-Cache.

Stress testing dimensions:
1. KV-cache equivalence under stress: run across multiple random batches, varying sequence lengths (240, 120, 10, 1), and verify error < 1e-5 across all outputs.
2. Numerical stability: test extreme values (+1e8, -1e8), infs, nans, zero volume, zero spread, zero body. Verify that no NaNs or Infs leak into z_out or attn_weights.
3. Checkpoint load stress: verify loading outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt with strict=True across repeated and concurrent instances.
"""

import math
import concurrent.futures
import pytest
import numpy as np
import torch

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    TemporalKVCache,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.real_m1_environment import RealM1Environment


class TestKVCacheStressEquivalence:
    """Stress tests verifying KV-cache mathematical equivalence and immutability."""

    @pytest.mark.parametrize("L", [240, 120, 64, 10, 1, 300])
    @pytest.mark.parametrize("B", [1, 2, 7, 16])
    def test_kv_cache_equivalence_across_variable_sequence_lengths(self, L, B):
        """Verify KV-cache forward pass matches full sequence forward pass within 1e-5 across diverse shapes."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        torch.manual_seed(42 + B * 1000 + L)
        z_feat = torch.randn(B, 768)
        candle_seq = torch.randn(B, L, 6)

        with torch.no_grad():
            z_full, attn_full = tca(z_feat, candle_seq=candle_seq)
            cache = tca.precompute_kv_cache(candle_seq)
            z_cached, attn_cached = tca.forward_with_kv_cache(z_feat, cache)
            z_delegated, attn_delegated = tca(z_feat, kv_cache=cache)

        diff_z = (z_full - z_cached).abs().max().item()
        diff_attn = (attn_full - attn_cached).abs().max().item()
        diff_del = (z_cached - z_delegated).abs().max().item()

        assert diff_z < 1e-5, f"B={B}, L={L}: z_out difference too high: {diff_z:.6e}"
        assert diff_attn < 1e-5, f"B={B}, L={L}: attn difference too high: {diff_attn:.6e}"
        assert diff_del == 0.0, f"B={B}, L={L}: delegation difference: {diff_del:.6e}"

    def test_kv_cache_broadcasting_stress(self):
        """Verify precomputed cache for B=1 correctly broadcasts to batched queries B in [1, 4, 16, 64]."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        torch.manual_seed(101)
        candle_seq_1 = torch.randn(1, 240, 6)
        cache = tca.precompute_kv_cache(candle_seq_1)

        for B in [1, 4, 16, 64]:
            z_feat = torch.randn(B, 768)
            with torch.no_grad():
                z_cached, attn_cached = tca.forward_with_kv_cache(z_feat, cache)
                candle_expanded = candle_seq_1.expand(B, -1, -1)
                z_full, attn_full = tca(z_feat, candle_seq=candle_expanded)

            diff_z = (z_full - z_cached).abs().max().item()
            diff_attn = (attn_full - attn_cached).abs().max().item()

            assert diff_z < 1e-5, f"Broadcast B={B} z_out diff: {diff_z:.6e}"
            assert diff_attn < 1e-5, f"Broadcast B={B} attn diff: {diff_attn:.6e}"

    def test_kv_cache_repeated_query_immutability(self):
        """Verify that precomputed TemporalKVCache is immutable under 100 sequential queries."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        torch.manual_seed(202)
        cache = tca.precompute_kv_cache(torch.randn(1, 240, 6))

        k_orig = cache.key.clone()
        v_orig = cache.value.clone()

        for _ in range(100):
            z = torch.randn(8, 768)
            with torch.no_grad():
                z_out, attn = tca.forward_with_kv_cache(z, cache)
            assert torch.isfinite(z_out).all()
            assert torch.isfinite(attn).all()

        assert torch.equal(cache.key, k_orig), "TemporalKVCache.key was mutated!"
        assert torch.equal(cache.value, v_orig), "TemporalKVCache.value was mutated!"

    def test_kv_cache_unbatched_and_dimension_variations(self):
        """Verify handling of unbatched 2D candle sequences [240, 6] and 1D z_feat [768]."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        z_1d = torch.randn(768)
        candle_2d = torch.randn(240, 6)

        with torch.no_grad():
            z_out, attn = tca(z_1d, candle_seq=candle_2d)
            cache = tca.precompute_kv_cache(candle_2d)
            z_cached, attn_cached = tca.forward_with_kv_cache(z_1d, cache)

        assert z_out.shape == (1, 768)
        assert attn.shape == (1, 4, 240)
        assert (z_out - z_cached).abs().max().item() == 0.0


class TestNumericalStabilityAdversarialStress:
    """Stress tests verifying numerical stability under extreme values, NaNs, Infs, and zero distributions."""

    @pytest.mark.parametrize("val", [1e8, -1e8, 1e12, -1e12, 1e18, -1e18])
    def test_extreme_finite_values_positive_and_negative(self, val):
        """Verify that extreme inputs up to +/-1e18 produce strictly finite outputs and valid probabilities."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        z = torch.full((2, 768), val)
        c = torch.full((2, 240, 6), val)

        with torch.no_grad():
            z_out, attn = tca(z, c)

        assert torch.isfinite(z_out).all(), f"Non-finite z_out for val {val}"
        assert torch.isfinite(attn).all(), f"Non-finite attn for val {val}"
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4), atol=1e-5), f"Attn sum mismatch for val {val}"
        assert (attn >= 0.0).all() and (attn <= 1.0).all()

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_inputs_infs_and_nans(self, bad_val):
        """Verify that injected NaNs and Infs are sanitized and do not leak into outputs."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        z = torch.full((2, 768), bad_val)
        c = torch.full((2, 240, 6), bad_val)

        with torch.no_grad():
            z_out, attn = tca(z, c)

        assert torch.isfinite(z_out).all(), f"Non-finite z_out for bad_val {bad_val}"
        assert torch.isfinite(attn).all(), f"Non-finite attn for bad_val {bad_val}"
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4), atol=1e-5)

    def test_degenerate_zero_inputs_volume_spread_body(self):
        """Verify that completely zeroed candle sequences (zero volume, zero spread, zero body) are stable."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        z_zero = torch.zeros(2, 768)
        c_zero = torch.zeros(2, 240, 6)

        with torch.no_grad():
            z_out, attn = tca(z_zero, c_zero)

        assert torch.isfinite(z_out).all(), "Non-finite z_out on all zeros"
        assert torch.isfinite(attn).all(), "Non-finite attn on all zeros"
        assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4), atol=1e-5)

    def test_attention_telemetry_under_adversarial_distributions(self):
        """Verify get_attention_summary under extreme and zero distributions."""
        tca = TemporalCrossAttentionEncoder(candle_len=240).eval()

        # Run on zeros
        with torch.no_grad():
            tca(torch.zeros(1, 768), torch.zeros(1, 240, 6))

        summary = tca.get_attention_summary()
        assert summary is not None
        assert np.isfinite(summary["mean_attention"]).all()
        assert np.isfinite(summary["recency_weight_ratio"])
        assert summary["recency_weight_ratio"] >= 0.0
        assert all(np.isfinite(e) for e in summary["head_entropy"])
        assert all(e >= 0.0 for e in summary["head_entropy"])
        assert 0.0 <= summary["last_candle_weight"] <= 1.0

    def test_environment_candle_sequence_extreme_market_conditions(self):
        """Verify RealM1Environment.get_candle_sequence under flat market, zero ATR/volume/spread, and corrupted data."""
        N = 300
        env = RealM1Environment()
        env.total_bars = N
        env.open_arr = np.full(N, 50000.0, dtype=np.float32)
        env.high_arr = np.full(N, 50010.0, dtype=np.float32)
        env.low_arr = np.full(N, 49990.0, dtype=np.float32)
        env.close_arr = np.full(N, 50005.0, dtype=np.float32)
        env.volume_arr = np.zeros(N, dtype=np.float32)  # Zero volume
        env.spread_arr = np.zeros(N, dtype=np.float32)  # Zero spread
        env.atr_arr = np.zeros(N, dtype=np.float32)     # Zero ATR

        # Corrupt specific indices with NaNs and Infs
        env.open_arr[10] = np.nan
        env.high_arr[20] = np.inf
        env.low_arr[30] = -np.inf
        env.atr_arr[50] = -5.0  # Negative ATR

        # Test boundary and middle indices
        for idx in [0, 1, 15, 25, 35, 50, 238, 239, 240, 299]:
            seq = env.get_candle_sequence(idx)
            assert seq.shape == (240, 6)
            assert np.isfinite(seq).all(), f"Non-finite at idx {idx}"
            # Values should be bounded within normalization ranges
            assert np.all((seq[:, 0] >= -10.0) & (seq[:, 0] <= 10.0))
            assert np.all((seq[:, 1] >= 0.0) & (seq[:, 1] <= 10.0))
            assert np.all((seq[:, 2] >= 0.0) & (seq[:, 2] <= 10.0))
            assert np.all((seq[:, 3] >= -10.0) & (seq[:, 3] <= 10.0))
            assert np.all((seq[:, 4] >= -1.0) & (seq[:, 4] <= 15.0))
            assert np.all((seq[:, 5] >= 0.0) & (seq[:, 5] <= 10.0))


class TestCheckpointStrictLoadingStress:
    """Stress tests verifying checkpoint loading under strict=True across repeated and concurrent calls."""

    CKPT_PATH = "outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt"

    def test_repeated_strict_checkpoint_loading_50x(self):
        """Verify 50 consecutive checkpoint loads into fresh LatentDynamicsWorldModel instances."""
        for _ in range(50):
            wm = LatentDynamicsWorldModel()
            assert len(wm.state_dict()) == 30
            ckpt = torch.load(self.CKPT_PATH, map_location="cpu", weights_only=False)
            res = wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
            assert len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0

    def test_concurrent_checkpoint_loading_multithreaded(self):
        """Verify concurrent multi-threaded loading across 8 worker threads for WM and Reasoning Engine."""
        def load_worker(worker_id):
            ckpt = torch.load(self.CKPT_PATH, map_location="cpu", weights_only=False)

            # World Model test
            wm = LatentDynamicsWorldModel()
            assert len(wm.state_dict()) == 30
            res_wm = wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
            assert len(res_wm.missing_keys) == 0 and len(res_wm.unexpected_keys) == 0

            # Attach temporal stream and run forward pass
            wm.enable_temporal_stream()
            z = wm.encode_state(torch.randn(1, 137), candle_seq=torch.randn(1, 240, 6))
            assert z.shape == (1, 768)

            # Reasoning Engine test (191 keys, 4.2M parameters)
            engine = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
            assert len(engine.state_dict()) == 191
            res_re = engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)
            assert len(res_re.missing_keys) == 0 and len(res_re.unexpected_keys) == 0

            return worker_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(load_worker, i) for i in range(16)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 16

    def test_strictness_enforcement_rejects_incompatible_state_dict(self):
        """Verify that strict=True strictly enforces the key contract and rejects mismatched weights."""
        wm_temporal = LatentDynamicsWorldModel()
        wm_temporal.enable_temporal_stream()
        temporal_state_dict = wm_temporal.state_dict()
        assert len(temporal_state_dict) > 30

        # Attempt to load temporal state_dict into default fresh WM with strict=True
        fresh_wm = LatentDynamicsWorldModel()
        with pytest.raises(RuntimeError) as exc_info:
            fresh_wm.load_state_dict(temporal_state_dict, strict=True)
        assert "Unexpected key(s) in state_dict" in str(exc_info.value)
