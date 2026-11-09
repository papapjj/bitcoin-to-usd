"""
Comprehensive unit, regression, and streaming tests for Solution 3:
Temporal Cross-Attention (Option C) with 240 M1 Candles Lookback & KV-Cache Logging.

Verifies:
1. RealM1Environment default lookback=240, 6 relative-normalized attributes, and causal padding.
2. TemporalCrossAttentionEncoder default candle_len=240, pos_emb [240, 128], causal_mask [240, 240].
3. KV-cache precomputation and mathematically equivalent incremental evaluation (< 1e-5 error, 0.0 bitwise).
4. Attention weights extraction across 240 candles [B, 4, 240] and telemetry logging interface.
5. Soft logit clamping [-50.0, 50.0] and extreme value / NaN sanitization.
6. Initial gated residual bias -2.0 preserving > 0.95 cosine similarity.
7. Strict checkpoint compatibility: default LatentDynamicsWorldModel retains exactly 30 keys,
   strict loading of mythos_muzero_specialized_challenger.pt, and 191 keys strict for reasoning engine.
8. SlicedM1Environment delegation to master environment.
9. End-to-end environment -> KV-cache -> world model -> telemetry pipeline.
"""

from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MarketStateEncoder,
    TemporalCrossAttentionEncoder,
    TemporalKVCache,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
from scripts.train_mythos_muzero_coevolution import SlicedM1Environment


def test_temporal_cross_attention_240_shapes_and_nan():
    """Verify forward pass with default 240 candles produces exact [B, 768] shape and [B, 4, 240] attention weights with zero NaNs."""
    B, L, D_candle = 4, 240, 6
    latent_dim = 768

    tca = TemporalCrossAttentionEncoder(
        candle_dim=D_candle,
        candle_len=L,
        latent_dim=latent_dim,
        temp_dim=128,
        num_heads=4,
        num_layers=2,
    )
    assert tca.candle_len == 240
    assert tca.pos_emb.shape == (240, 128)
    assert tca.causal_mask.shape == (240, 240)

    z_feat = torch.randn(B, latent_dim)
    candle_seq = torch.randn(B, L, D_candle)

    z_out, attn_weights = tca(z_feat, candle_seq)

    assert z_out.shape == (B, latent_dim), f"Expected ({B}, {latent_dim}), got {z_out.shape}"
    assert attn_weights.shape == (B, 4, L), f"Expected ({B}, 4, {L}), got {attn_weights.shape}"
    assert not torch.isnan(z_out).any(), "NaN found in z_out"
    assert not torch.isinf(z_out).any(), "Inf found in z_out"
    assert not torch.isnan(attn_weights).any(), "NaN found in attn_weights"

    # Attention weights must sum to 1.0 along the candle sequence dimension
    sum_weights = attn_weights.sum(dim=-1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5), "Attention weights do not sum to 1"


def test_temporal_cross_attention_arbitrary_length_padding_and_slicing():
    """Verify encoder automatically left-pads shorter sequences (< 240) and slices longer sequences (> 240)."""
    tca = TemporalCrossAttentionEncoder()  # Defaults to 240
    z_feat = torch.randn(2, 768)

    # 1. Shorter sequence (64 candles)
    short_seq = torch.randn(2, 64, 6)
    z_short, attn_short = tca(z_feat, short_seq)
    assert z_short.shape == (2, 768)
    assert attn_short.shape == (2, 4, 240)
    assert not torch.isnan(z_short).any()
    assert torch.allclose(attn_short.sum(dim=-1), torch.ones(2, 4), atol=1e-5)

    # 2. Longer sequence (300 candles)
    long_seq = torch.randn(2, 300, 6)
    z_long, attn_long = tca(z_feat, long_seq)
    assert z_long.shape == (2, 768)
    assert attn_long.shape == (2, 4, 240)
    assert not torch.isnan(z_long).any()
    assert torch.allclose(attn_long.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_gated_bias_preservation_240():
    """
    Verify that with initial gate bias = -2.0, z_out strongly preserves z_feat
    (Cosine similarity > 0.95) over 240 candles, preventing distribution shock.
    """
    B, L, D_candle = 8, 240, 6
    latent_dim = 768

    tca = TemporalCrossAttentionEncoder(
        candle_dim=D_candle,
        candle_len=L,
        latent_dim=latent_dim,
        init_gate_bias=-2.0,
    )

    z_feat = torch.randn(B, latent_dim)
    candle_seq = torch.randn(B, L, D_candle)

    z_out, _ = tca(z_feat, candle_seq)

    cos_sim = F.cosine_similarity(z_out, F.layer_norm(z_feat, (latent_dim,)), dim=-1).mean().item()
    assert cos_sim > 0.95, f"Cosine similarity too low ({cos_sim:.4f}), risk of distribution shock"


def test_kv_cache_precomputation_and_mathematical_equivalence():
    """
    Verify that incremental evaluation using precomputed KV-cache produces
    mathematically equivalent output to full-sequence forward pass (< 1e-5 error, 0.0 bitwise in eval mode).
    """
    tca = TemporalCrossAttentionEncoder(candle_len=240)
    tca.eval()  # Eval mode disables dropout to ensure exact deterministic math

    B = 3
    z_feat = torch.randn(B, 768)
    candle_seq = torch.randn(B, 240, 6)

    with torch.no_grad():
        # Full forward pass
        z_full, attn_full = tca(z_feat, candle_seq)

        # Precompute KV-cache
        cache = tca.precompute_kv_cache(candle_seq)
        assert isinstance(cache, TemporalKVCache)
        assert cache.key.shape == (B, 4, 240, 32)
        assert cache.value.shape == (B, 4, 240, 32)
        assert cache.candle_len == 240
        assert cache.batch_size == B

        # Tuple unpacking test
        k_unpacked, v_unpacked = cache
        assert torch.equal(k_unpacked, cache.key)
        assert torch.equal(v_unpacked, cache.value)

        # Dict indexing test
        assert torch.equal(cache["key"], cache.key)
        assert torch.equal(cache["value"], cache.value)

        # Cached forward pass via forward_with_kv_cache
        z_cached, attn_cached = tca.forward_with_kv_cache(z_feat, cache)

        # Cached forward pass via forward(..., kv_cache=cache)
        z_cached_fwd, attn_cached_fwd = tca(z_feat, kv_cache=cache)

    diff_z = (z_full - z_cached).abs().max().item()
    diff_attn = (attn_full - attn_cached).abs().max().item()
    assert diff_z < 1e-5, f"KV-cache z_out difference too high: {diff_z:.6e}"
    assert diff_attn < 1e-5, f"KV-cache attn_weights difference too high: {diff_attn:.6e}"

    diff_fwd = (z_cached - z_cached_fwd).abs().max().item()
    assert diff_fwd == 0.0, f"Forward delegation discrepancy: {diff_fwd}"


def test_attention_weights_extraction_and_telemetry_240():
    """Verify extraction and logging of attention weights across all 240 candles and summary telemetry."""
    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream()  # Defaults to 240

    obs_137 = np.random.randn(137).astype(np.float32)
    candle_seq = np.random.randn(240, 6).astype(np.float32)

    # Encode with candles
    z_out = wm.encode_state(obs_137, candle_seq=candle_seq)
    assert z_out.shape == (1, 768)

    # Test get_temporal_attention_weights
    attn_batched = wm.get_temporal_attention_weights(unbatch=False)
    assert attn_batched is not None
    assert attn_batched.shape == (1, 4, 240)

    attn_unbatched = wm.get_temporal_attention_weights(unbatch=True)
    assert attn_unbatched is not None
    assert attn_unbatched.shape == (4, 240)

    # Test get_attention_summary
    summary = wm.get_attention_summary()
    assert summary is not None
    assert "mean_attention" in summary
    assert "top_k_indices" in summary
    assert "recency_weight_ratio" in summary
    assert "head_entropy" in summary
    assert "last_candle_weight" in summary

    assert summary["mean_attention"].shape == (240,)
    assert not np.isnan(summary["mean_attention"]).any()
    assert len(summary["top_k_indices"]) == 5
    for idx in summary["top_k_indices"]:
        assert 0 <= idx < 240

    assert summary["recency_weight_ratio"] > 0.0
    assert math.isfinite(summary["recency_weight_ratio"])
    assert len(summary["head_entropy"]) == 4
    for ent in summary["head_entropy"]:
        assert ent > 0.0 and math.isfinite(ent)
    assert 0.0 <= summary["last_candle_weight"] <= 1.0


def test_numerical_stability_soft_clamping_and_extreme_inputs():
    """Verify that soft logit clamping [-50.0, 50.0] and input sanitization prevent overflow on extreme inputs."""
    tca = TemporalCrossAttentionEncoder(candle_len=240)

    # Inject extreme values, infs, and nans
    z_feat = torch.tensor([[float("nan"), 1e6, -1e6] + [0.0] * 765])
    candle_seq = torch.full((1, 240, 6), 1e5)
    candle_seq[0, 50, :] = float("nan")
    candle_seq[0, 100, :] = float("inf")
    candle_seq[0, 150, :] = float("-inf")

    z_out, attn_weights = tca(z_feat, candle_seq)

    assert not torch.isnan(z_out).any(), "NaN found in extreme input z_out"
    assert not torch.isinf(z_out).any(), "Inf found in extreme input z_out"
    assert not torch.isnan(attn_weights).any(), "NaN found in extreme input attn_weights"
    assert not torch.isinf(attn_weights).any(), "Inf found in extreme input attn_weights"
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_checkpoint_strict_load_30_keys():
    """Verify that default LatentDynamicsWorldModel retains exactly 30 keys and loads challenger checkpoint under strict=True."""
    ckpt_path = "outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    wm = LatentDynamicsWorldModel()
    # Before enabling temporal stream, temporal_encoder is None
    assert wm.temporal_encoder is None
    state_dict = wm.state_dict()
    assert len(state_dict) == 30, f"Expected exactly 30 keys, got {len(state_dict)}"

    # Strict load must pass with 0 missing and 0 unexpected keys
    res = wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
    assert len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0


def test_reasoning_engine_191_keys_strict_intact():
    """Verify that MetacognitiveReasoningEngine retains exactly 191 keys and 4,196,564 weights under strict=True."""
    cfg = ReasoningConfig(input_dim=768)
    engine = MetacognitiveReasoningEngine(cfg)

    assert len(engine.state_dict()) == 191, f"Expected 191 keys, got {len(engine.state_dict())}"
    param_count = sum(p.numel() for p in engine.parameters())
    assert param_count == 4196564, f"Expected 4196564 parameters, got {param_count}"

    ckpt_path = "outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    res = engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)
    assert len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0


def test_real_m1_environment_default_lookback_240_and_attributes():
    """Verify RealM1Environment.get_candle_sequence defaults to lookback=240 with all 6 relative-normalized attributes."""
    N = 500
    env = RealM1Environment()
    env.total_bars = N
    env.open_arr = np.linspace(50000, 50100, N, dtype=np.float32)
    env.high_arr = env.open_arr + 12.0
    env.low_arr = env.open_arr - 8.0
    env.close_arr = env.open_arr + 3.0
    env.volume_arr = np.ones(N, dtype=np.float32) * 20.0
    env.spread_arr = np.ones(N, dtype=np.float32) * 2.5
    env.atr_arr = np.ones(N, dtype=np.float32) * 25.0

    # 1. Default call (lookback omitted)
    seq_default = env.get_candle_sequence(350)
    assert seq_default.shape == (240, 6), f"Expected (240, 6), got {seq_default.shape}"
    assert not np.isnan(seq_default).any()
    assert not np.isinf(seq_default).any()

    # Attribute bounds check:
    # 0: Body ATR ratio [-10, 10]
    assert np.all((seq_default[:, 0] >= -10.0) & (seq_default[:, 0] <= 10.0))
    # 1: Upper Wick ATR ratio [0, 10]
    assert np.all((seq_default[:, 1] >= 0.0) & (seq_default[:, 1] <= 10.0))
    # 2: Lower Wick ATR ratio [0, 10]
    assert np.all((seq_default[:, 2] >= 0.0) & (seq_default[:, 2] <= 10.0))
    # 3: Price Velocity ATR ratio [-10, 10]
    assert np.all((seq_default[:, 3] >= -10.0) & (seq_default[:, 3] <= 10.0))
    # 4: Relative Volume [-1, 15]
    assert np.all((seq_default[:, 4] >= -1.0) & (seq_default[:, 4] <= 15.0))
    # 5: Relative Spread [0, 10]
    assert np.all((seq_default[:, 5] >= 0.0) & (seq_default[:, 5] <= 10.0))

    # 2. Early boundary condition (bar_idx = 50 < 240)
    seq_early = env.get_candle_sequence(50)
    assert seq_early.shape == (240, 6)
    assert not np.isnan(seq_early).any()
    # Left-padding: 240 - 51 = 189 leading zero rows
    assert np.all(seq_early[:189, :] == 0.0)
    # The populated 51 rows should have non-zero or valid values
    assert not np.all(seq_early[189:, :] == 0.0)


def test_sliced_m1_environment_delegation_240():
    """Verify SlicedM1Environment exposes get_candle_sequence delegating to master environment."""
    N = 600
    master_env = RealM1Environment()
    master_env.total_bars = N
    master_env.open_arr = np.linspace(50000, 50200, N, dtype=np.float32)
    master_env.high_arr = master_env.open_arr + 15.0
    master_env.low_arr = master_env.open_arr - 15.0
    master_env.close_arr = master_env.open_arr + 4.0
    master_env.volume_arr = np.ones(N, dtype=np.float32) * 25.0
    master_env.spread_arr = np.ones(N, dtype=np.float32) * 2.5
    master_env.atr_arr = np.ones(N, dtype=np.float32) * 30.0
    master_env.day_progress_arr = np.zeros(N, dtype=np.float32)
    master_env.timestamps = ["2025-01-15 00:00:00"] * N
    master_env.obs_matrix = np.zeros((N, 137), dtype=np.float32)

    sliced = SlicedM1Environment(master_env, start_idx=150, end_idx=550, name="test_slice")
    assert hasattr(sliced, "get_candle_sequence"), "SlicedM1Environment missing get_candle_sequence"

    # Default lookback = 240
    seq_sliced = sliced.get_candle_sequence(100)
    seq_master = master_env.get_candle_sequence(250, lookback=240)

    assert seq_sliced.shape == (240, 6)
    assert np.allclose(seq_sliced, seq_master)


def test_end_to_end_environment_world_model_with_kv_cache_240():
    """Verify full end-to-end integration: RealM1Environment -> KV-Cache -> WorldModel encode_state -> Telemetry."""
    N = 400
    env = RealM1Environment()
    env.total_bars = N
    env.open_arr = np.linspace(50000, 50150, N, dtype=np.float32)
    env.high_arr = env.open_arr + 10.0
    env.low_arr = env.open_arr - 10.0
    env.close_arr = env.open_arr + 2.0
    env.volume_arr = np.ones(N, dtype=np.float32) * 18.0
    env.spread_arr = np.ones(N, dtype=np.float32) * 2.5
    env.atr_arr = np.ones(N, dtype=np.float32) * 28.0

    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream()  # 240 candles

    obs_137 = np.random.randn(137).astype(np.float32)
    candle_seq = env.get_candle_sequence(300)  # Default 240 candles

    # 1. Direct encode_state with candle_seq
    z_direct = wm.encode_state(obs_137, candle_seq=candle_seq)
    assert z_direct.shape == (1, 768)

    # 2. Encode with precomputed KV-cache
    kv_cache = wm.precompute_temporal_kv_cache(candle_seq)
    assert kv_cache is not None
    z_cached = wm.encode_state(obs_137, kv_cache=kv_cache)
    assert z_cached.shape == (1, 768)

    # 3. Verify telemetry is populated
    summary = wm.get_attention_summary()
    assert summary is not None
    assert summary["candle_len"] == 240
    assert summary["num_heads"] == 4
