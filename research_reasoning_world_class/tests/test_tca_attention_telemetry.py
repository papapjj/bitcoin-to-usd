"""
Unit and Integration Tests for Feature 15: 240-Candle Cross-Attention Heatmap Extraction and Telemetry.
Verifies:
1. encode_state backward compatibility (return_attention=False returns Tensor, return_attention=True returns Tuple)
2. Mathematical telemetry invariants (Shannon entropy, uniform & Dirac oracles, recency ratios)
3. Cross-head coherence, Gini sparsity, peak candle detection
4. Zero-drift and 30-key baseline state dict preservation
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest
import torch

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    compute_attention_telemetry,
    extract_attention_weights,
)


class TestAttentionExtractionAPI:
    """Tests encode_state and extraction helper APIs."""

    def test_encode_state_default_returns_tensor(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        obs = torch.randn(2, 137)
        candles = torch.randn(2, 240, 6)

        z = wm.encode_state(obs, candle_seq=candles)
        assert isinstance(z, torch.Tensor)
        assert z.shape == (2, 768)

    def test_encode_state_return_attention_true(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        obs = torch.randn(2, 137)
        candles = torch.randn(2, 240, 6)

        out = wm.encode_state(obs, candle_seq=candles, return_attention=True)
        assert isinstance(out, tuple)
        assert len(out) == 2
        z, attn = out
        assert isinstance(z, torch.Tensor) and z.shape == (2, 768)
        assert isinstance(attn, torch.Tensor) and attn.shape == (2, 4, 240)

    def test_encode_state_unaugmented_stream_return_attention(self):
        wm = LatentDynamicsWorldModel()  # temporal stream disabled
        obs = torch.randn(2, 137)

        # Default
        z = wm.encode_state(obs)
        assert isinstance(z, torch.Tensor) and z.shape == (2, 768)

        # return_attention=True when temporal stream is disabled
        z_out, attn = wm.encode_state(obs, return_attention=True)
        assert isinstance(z_out, torch.Tensor) and z_out.shape == (2, 768)
        assert attn is None

    def test_extract_attention_weights_helpers(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        obs = torch.randn(1, 137)
        candles = torch.randn(1, 240, 6)

        # 1. Method on model
        attn1 = wm.extract_attention_weights(obs, candle_seq=candles)
        assert attn1 is not None and attn1.shape == (1, 4, 240)

        # Unbatched
        attn_unbatch = wm.extract_attention_weights(unbatch=True)
        assert attn_unbatch.shape == (4, 240)

        # 2. Standalone function
        attn2 = extract_attention_weights(wm)
        assert attn2 is not None and attn2.shape == (1, 4, 240)

        # 3. Direct tensor passing
        raw_tensor = torch.randn(1, 4, 240)
        assert extract_attention_weights(raw_tensor, unbatch=True).shape == (4, 240)


class TestMathematicalTelemetryOracles:
    """Verifies mathematical correctness of entropy, recency, and distribution metrics against analytical oracles."""

    def test_uniform_distribution_analytical_oracles(self):
        # Under uniform distribution across 240 candles:
        # p_i = 1/240
        # H = - sum(1/240 * ln(1/240)) = ln(240) = 5.480639 nats
        # recency_ratio_30_210 = (30/240) / (210/240) = 30 / 210 = 1/7 = 0.142857
        # gini_sparsity = 0.0
        # cross_head_coherence = 1.0 (identical uniform heads)
        p_uniform = torch.full((1, 4, 240), 1.0 / 240.0)
        t = compute_attention_telemetry(p_uniform, bar_idx=55000)

        expected_entropy = math.log(240.0)
        assert abs(t["entropy"] - expected_entropy) < 1e-4
        assert abs(t["mean_entropy"] - expected_entropy) < 1e-4
        assert abs(t["normalized_entropy"] - 1.0) < 1e-4
        assert abs(t["recency_ratio_30_210"] - (1.0 / 7.0)) < 1e-4
        assert abs(t["recency_ratio"] - (1.0 / 7.0)) < 1e-4
        assert abs(t["recency_weight_ratio"] - (15.0 / 225.0)) < 1e-4
        assert t["gini_sparsity"] == 0.0
        assert abs(t["cross_head_coherence"] - 1.0) < 1e-4

    def test_dirac_delta_distribution_analytical_oracles(self):
        # Under Dirac delta at last candle (index 239):
        # p_239 = 1.0, others = 0.0
        # H = 0.0 nats
        # peak_candle_index = 239, peak_candle_lag = 0
        # recency_ratio_30_210 = 1.0 / 1e-6 = 1,000,000.0
        # gini_sparsity = (239 * 1.0) / 240 = 0.995833
        p_dirac = torch.zeros((1, 4, 240))
        p_dirac[:, :, -1] = 1.0
        t = compute_attention_telemetry(p_dirac, bar_idx=64784)

        assert t["peak_candle_index"] == 239
        assert t["peak_candle_lag"] == 0
        assert t["entropy"] == 0.0
        assert t["mean_entropy"] == 0.0
        assert t["normalized_entropy"] == 0.0
        assert t["recency_ratio_30_210"] == 1000000.0
        assert t["recency_ratio"] == 1000000.0
        assert abs(t["gini_sparsity"] - (239.0 / 240.0)) < 1e-4
        assert t["last_candle_weight"] == 1.0

    def test_head_entropy_and_coherence_under_orthogonal_heads(self):
        # 4 heads attending to completely disjoint subsets of candles
        p_disjoint = torch.zeros((1, 4, 240))
        p_disjoint[0, 0, 0:60] = 1.0 / 60.0
        p_disjoint[0, 1, 60:120] = 1.0 / 60.0
        p_disjoint[0, 2, 120:180] = 1.0 / 60.0
        p_disjoint[0, 3, 180:240] = 1.0 / 60.0

        t = compute_attention_telemetry(p_disjoint)
        # Each head has entropy ln(60)
        expected_head_entropy = math.log(60.0)
        for h in range(4):
            assert abs(t["head_entropy"][h] - expected_head_entropy) < 1e-4

        # Disjoint heads have negative pairwise correlation
        assert t["cross_head_coherence"] < 0.0
        assert t["cross_head_variance"] > 0.0

    def test_json_serialization_safety(self):
        # All float/int fields must be serializable without TypeError
        p_rand = torch.softmax(torch.randn(1, 4, 240), dim=-1)
        t = compute_attention_telemetry(p_rand, bar_idx=100, timestamp="2025-02-28T12:00:00Z")

        # mean_attention is numpy array for backwards compatibility; remove for pure JSON
        serializable = {k: v for k, v in t.items() if k != "mean_attention"}
        json_str = json.dumps(serializable)
        loaded = json.loads(json_str)

        assert loaded["bar_idx"] == 100
        assert len(loaded["mean_weights_240"]) == 240
        assert len(loaded["head_weights"]) == 4
        assert len(loaded["head_weights"][0]) == 240
        assert "recency_ratio" in loaded
        assert "entropy" in loaded


class TestWorldModelAttentionIntegration:
    """Verifies integration with LatentDynamicsWorldModel."""

    def test_world_model_extract_telemetry_workflow(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        obs = torch.randn(1, 137)
        candles = torch.randn(1, 240, 6)

        # Forward pass and extract telemetry
        _ = wm.encode_state(obs, candle_seq=candles)
        summary = wm.extract_attention_telemetry(bar_idx=55000, timestamp="2025.02.22 04:48:00")

        assert summary is not None
        assert summary["bar_idx"] == 55000
        assert summary["timestamp"] == "2025.02.22 04:48:00"
        assert len(summary["top_k_indices"]) == 5
        assert len(summary["mean_weights_240"]) == 240
        assert summary["recency_ratio"] > 0.0
        assert summary["entropy"] > 0.0

    def test_world_model_30_keys_isolation_intact(self):
        # Enforce that adding Feature 15 did NOT alter the 30-key baseline state dict
        wm = LatentDynamicsWorldModel()
        clean_keys = [k for k in wm.state_dict().keys() if not k.startswith("temporal_encoder.")]
        assert len(clean_keys) == 30
        assert sum(p.numel() for p in wm.parameters()) == 2160900
