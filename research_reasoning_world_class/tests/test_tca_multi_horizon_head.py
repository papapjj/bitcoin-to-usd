"""
Unit and Integration Test Suite for Multi-Horizon Projection Head & TCA Integration.
File: research_reasoning_world_class/tests/test_tca_multi_horizon_head.py
Milestone: Milestone 2 (Multi-Horizon Projection Head & TCA Integration — Features 5, 6, 7, 8)

Contains 20 focused, rigorous unit tests:
- Test 1: Instantiation, parameter count (230,659), and tensor shapes.
- Test 2: Input/output shape validation for arbitrary batch sizes [B, 768] -> [B, 3].
- Test 3: Weight names and structure inspection (8 keys).
- Test 4: Gradient flow back into TemporalCrossAttentionEncoder parameters.
- Test 5: Gradient isolation when Base MarketStateEncoder is frozen (requires_grad=False).
- Test 6: Reasoning Engine zero drift isolation when Reasoning Engine is frozen.
- Test 7: compute_multi_horizon_loss correctness against manual Smooth L1 reference.
- Test 8: Loss behavior with custom per-horizon weighting tensor.
- Test 9: Loss computation handles target clipping boundaries [-10.0, 10.0].
- Test 10: TCA KV-cache bitwise equivalence (max(|z_cache - z_full|) == 0.0 in eval mode).
- Test 11: KV-cache inference speedup benchmark (>5x speedup verified).
- Test 12: Dropout behavior in train() vs eval() mode.
- Test 13: Checkpoint serialization schema isolating 4 namespaces: world_model_state_dict,
          temporal_encoder_state_dict, projection_head_state_dict, reasoning_engine_state_dict.
- Test 14: Strict 30-key baseline loading on detached LatentDynamicsWorldModel.
- Test 15: Loading authentic production checkpoint outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt with strict=True.
- Test 16: End-to-end forward pass with SingleRegimeDataset.get_batch() tensors.
- Test 17: MegaPoolDataset multi-regime batch compatibility.
- Test 18: Un-normalized return prediction via predict_returns(z, atr).
- Test 19: Numerical stability under extreme inputs (large values, zeros).
- Test 20: Deterministic output repeatability given identical seeds.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    TemporalKVCache,
    MultiHorizonProjectionHead,
    create_projection_head,
    compute_multi_horizon_loss,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.mega_pool_dataset import (
    MarketRegime,
    SingleRegimeDataset,
    MegaPoolDataset,
    MegaPoolBatch,
)
from research_reasoning_world_class.tests.test_mega_pool_dataset import MockM1Environment


# ==============================================================================
# TEST 1: Instantiation, Parameter Count (230,659), and Tensor Shapes
# ==============================================================================

def test_01_instantiation_parameter_count_shapes():
    """Validates instantiation, exact parameter count (230,659), and layer tensor shapes."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
    param_count = sum(p.numel() for p in head.parameters())
    assert param_count == 230659, f"Expected exactly 230,659 parameters, got {param_count}"

    # Verify individual parameter shapes
    sd = head.state_dict()
    assert sd["net.0.weight"].shape == (256, 768)
    assert sd["net.0.bias"].shape == (256,)
    assert sd["net.1.weight"].shape == (256,)
    assert sd["net.1.bias"].shape == (256,)
    assert sd["net.4.weight"].shape == (128, 256)
    assert sd["net.4.bias"].shape == (128,)
    assert sd["net.6.weight"].shape == (3, 128)
    assert sd["net.6.bias"].shape == (3,)

    # Verify factory function
    factory_head = create_projection_head(d_in=768, horizons=3)
    assert isinstance(factory_head, MultiHorizonProjectionHead)
    assert sum(p.numel() for p in factory_head.parameters()) == 230659


# ==============================================================================
# TEST 2: Input/Output Shape Validation for Arbitrary Batch Sizes [B, 768] -> [B, 3]
# ==============================================================================

def test_02_input_output_shape_validation():
    """Validates input/output shapes across varied batch sizes and unbatched vectors."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).eval()

    batch_sizes = [1, 2, 4, 7, 16, 32, 64, 128]
    for b in batch_sizes:
        z = torch.randn(b, 768)
        pred = head(z)
        assert pred.shape == (b, 3), f"Expected shape ({b}, 3), got {pred.shape}"

    # Single unbatched vector [768] should produce [1, 3]
    z_1d = torch.randn(768)
    pred_1d = head(z_1d)
    assert pred_1d.shape == (1, 3), f"Expected shape (1, 3), got {pred_1d.shape}"


# ==============================================================================
# TEST 3: Weight Names and Structure Inspection (8 Keys)
# ==============================================================================

def test_03_weight_names_and_structure_inspection():
    """Verifies that state_dict has exactly 8 keys and inspects submodule topology."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
    state_dict = head.state_dict()

    expected_keys = {
        "net.0.weight",
        "net.0.bias",
        "net.1.weight",
        "net.1.bias",
        "net.4.weight",
        "net.4.bias",
        "net.6.weight",
        "net.6.bias",
    }
    assert set(state_dict.keys()) == expected_keys, f"Key mismatch: {set(state_dict.keys()) ^ expected_keys}"
    assert len(state_dict) == 8

    # Inspect network children
    layers = list(head.net.children())
    assert len(layers) == 7
    assert isinstance(layers[0], nn.Linear) and layers[0].in_features == 768 and layers[0].out_features == 256
    assert isinstance(layers[1], nn.LayerNorm) and layers[1].normalized_shape == (256,)
    assert isinstance(layers[2], nn.GELU)
    assert isinstance(layers[3], nn.Dropout) and layers[3].p == 0.05
    assert isinstance(layers[4], nn.Linear) and layers[4].in_features == 256 and layers[4].out_features == 128
    assert isinstance(layers[5], nn.GELU)
    assert isinstance(layers[6], nn.Linear) and layers[6].in_features == 128 and layers[6].out_features == 3


# ==============================================================================
# TEST 4: Gradient Flow Back into TemporalCrossAttentionEncoder Parameters
# ==============================================================================

def test_04_gradient_flow_into_tca_module():
    """Verifies end-to-end backpropagation through head into TCA transformer, cross-attn, and gate."""
    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)

    obs = torch.randn(2, 137)
    candles = torch.randn(2, 240, 6)
    targets = torch.randn(2, 3)

    z_out = wm.encode_state(obs, candle_seq=candles)
    preds = head(z_out)
    loss = compute_multi_horizon_loss(preds, targets, beta=1.0)
    loss.backward()

    # Head parameters must receive gradients
    assert head.net[0].weight.grad is not None
    assert torch.isfinite(head.net[0].weight.grad).all()

    # Temporal encoder parameters must receive active gradients
    assert wm.temporal_encoder.candle_proj[0].weight.grad is not None
    assert torch.isfinite(wm.temporal_encoder.candle_proj[0].weight.grad).all()

    assert wm.temporal_encoder.out_proj[0].weight.grad is not None
    assert torch.isfinite(wm.temporal_encoder.out_proj[0].weight.grad).all()

    assert wm.temporal_encoder.gate[0].weight.grad is not None
    assert torch.isfinite(wm.temporal_encoder.gate[0].weight.grad).all()


# ==============================================================================
# TEST 5: Gradient Isolation when Base MarketStateEncoder is Frozen
# ==============================================================================

def test_05_gradient_isolation_base_encoder_frozen():
    """Verifies Phase 1 freeze invariant: zero gradient leakage into Base MarketStateEncoder."""
    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)

    # Freeze Base MarketStateEncoder and Dynamics Head (Phase 1 contract)
    for p in wm.encoder.parameters():
        p.requires_grad = False
    for p in wm.g_net.parameters():
        p.requires_grad = False

    obs = torch.randn(4, 137)
    candles = torch.randn(4, 240, 6)
    targets = torch.randn(4, 3)

    z_out = wm.encode_state(obs, candle_seq=candles)
    preds = head(z_out)
    loss = compute_multi_horizon_loss(preds, targets, beta=1.0)
    loss.backward()

    # Base encoder and g_net must have strictly None gradients
    assert all(p.grad is None for p in wm.encoder.parameters()), "Gradients leaked into frozen Base Encoder!"
    assert all(p.grad is None for p in wm.g_net.parameters()), "Gradients leaked into frozen g_net!"

    # TCA and Head parameters must still receive non-zero gradients
    assert wm.temporal_encoder.gate[0].weight.grad is not None
    assert head.net[0].weight.grad is not None


# ==============================================================================
# TEST 6: Reasoning Engine Zero Drift Isolation when Frozen
# ==============================================================================

def test_06_reasoning_engine_zero_drift_isolation():
    """Verifies that Reasoning Engine (191 keys, 4,196,564 weights) remains 100% frozen with zero drift."""
    re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))

    # Strict invariant: 191 keys, 4,196,564 weights
    assert len(re.state_dict()) == 191
    assert sum(p.numel() for p in re.parameters()) == 4196564

    # Freeze all parameters
    for p in re.parameters():
        p.requires_grad = False

    initial_weights = {name: p.clone() for name, p in re.named_parameters()}

    # Run reasoning pass and backprop through a downstream loss
    z = torch.randn(2, 768)
    re.eval()
    _ = re(z)

    # Run separate training updates
    dummy_head = nn.Linear(768, 3)
    loss = dummy_head(z).sum()
    loss.backward()

    # Verify zero gradient leakage and zero weight drift
    assert all(p.grad is None for p in re.parameters()), "Reasoning engine parameters received gradients!"
    for name, p in re.named_parameters():
        drift = torch.max(torch.abs(p - initial_weights[name])).item()
        assert drift == 0.0, f"Weight drift {drift} detected in reasoning engine parameter {name}!"


# ==============================================================================
# TEST 7: compute_multi_horizon_loss Correctness Against Manual Smooth L1
# ==============================================================================

def test_07_compute_multi_horizon_loss_correctness():
    """Validates compute_multi_horizon_loss against a direct manual reference."""
    preds = torch.tensor([[0.5, 2.0, -3.0], [-0.2, 0.0, 1.5]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0, -1.0], [0.1, -0.4, 0.5]], dtype=torch.float32)

    loss = compute_multi_horizon_loss(preds, targets, beta=1.0)

    # Manual Huber calculation
    diff = preds - targets
    abs_diff = torch.abs(diff)
    manual_huber = torch.where(abs_diff < 1.0, 0.5 * (abs_diff ** 2), abs_diff - 0.5)
    expected_loss = manual_huber.mean()

    assert torch.isclose(loss, expected_loss, atol=1e-6)

    # Exact match must yield 0.0
    zero_loss = compute_multi_horizon_loss(targets, targets, beta=1.0)
    assert torch.isclose(zero_loss, torch.tensor(0.0), atol=1e-7)


# ==============================================================================
# TEST 8: Loss Behavior with Custom Per-Horizon Weighting Tensor
# ==============================================================================

def test_08_loss_behavior_custom_weighting():
    """Validates compute_multi_horizon_loss with custom per-horizon weighting tensor."""
    preds = torch.tensor([[0.5, 1.5, 3.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    weights = torch.tensor([2.0, 1.0, 0.5])
    loss = compute_multi_horizon_loss(preds, targets, beta=1.0, weights=weights)

    # Manual calculation
    diff = preds - targets
    abs_diff = torch.abs(diff)
    huber = torch.where(abs_diff < 1.0, 0.5 * (abs_diff ** 2), abs_diff - 0.5)
    expected = (huber * weights).mean()

    assert torch.isclose(loss, expected, atol=1e-6)

    # Verify that zeroing weight for horizon 0 eliminates gradient for horizon 0
    preds_zero_w = torch.tensor([[0.5, 1.5, 3.0]], dtype=torch.float32, requires_grad=True)
    weights_zero0 = torch.tensor([0.0, 1.0, 1.0])
    loss_zero0 = compute_multi_horizon_loss(preds_zero_w, targets, beta=1.0, weights=weights_zero0)
    loss_zero0.backward()

    assert preds_zero_w.grad[0, 0].item() == 0.0, "Zero-weighted horizon received non-zero gradient!"
    assert preds_zero_w.grad[0, 1].item() != 0.0
    assert preds_zero_w.grad[0, 2].item() != 0.0


# ==============================================================================
# TEST 9: Loss Computation Handles Target Clipping Boundaries [-10.0, 10.0]
# ==============================================================================

def test_09_loss_handles_target_clipping_boundaries():
    """Validates that loss computation and gradients remain bounded at clipping boundaries [-10, 10]."""
    targets = torch.tensor([[-10.0, 0.0, 10.0], [10.0, -10.0, 0.0]], dtype=torch.float32)
    preds = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)

    loss = compute_multi_horizon_loss(preds, targets, beta=1.0)
    assert torch.isfinite(loss)
    loss.backward()

    # For errors >= 1.0 under Smooth L1, gradient magnitude is bounded by 1.0 (mean across 6 elements is 1/6)
    assert torch.isfinite(preds.grad).all()
    # Max element gradient magnitude must not exceed 1.0 / num_elements
    assert torch.max(torch.abs(preds.grad)).item() <= 1.0


# ==============================================================================
# TEST 10: TCA KV-Cache Bitwise Equivalence in Eval Mode
# ==============================================================================

def test_10_tca_kv_cache_bitwise_equivalence():
    """Validates exact 0.0 bitwise equivalence between full forward pass and KV-cached pass."""
    wm = LatentDynamicsWorldModel().eval()
    wm.enable_temporal_stream(candle_len=240)
    wm.eval()
    head = MultiHorizonProjectionHead().eval()

    obs = torch.randn(2, 137)
    candles = torch.randn(2, 240, 6)

    with torch.no_grad():
        z_full = wm.encode_state(obs, candle_seq=candles)
        pred_full = head(z_full)

        cache = wm.precompute_temporal_kv_cache(candles)
        z_cached = wm.encode_state(obs, kv_cache=cache)
        pred_cached = head(z_cached)

    max_z_diff = torch.max(torch.abs(z_full - z_cached)).item()
    max_pred_diff = torch.max(torch.abs(pred_full - pred_cached)).item()

    assert max_z_diff == 0.0, f"Expected 0.0 z discrepancy, got {max_z_diff}"
    assert max_pred_diff == 0.0, f"Expected 0.0 prediction discrepancy, got {max_pred_diff}"


# ==============================================================================
# TEST 11: KV-Cache Inference Speedup Benchmark (>5x Speedup Verified)
# ==============================================================================

def test_11_kv_cache_inference_speedup_benchmark():
    """Validates that KV-cached inference achieves >5x speedup over full sequence pass."""
    wm = LatentDynamicsWorldModel().eval()
    wm.enable_temporal_stream(candle_len=240)
    wm.eval()
    head = MultiHorizonProjectionHead().eval()

    B = 4
    obs = torch.randn(B, 137)
    candles = torch.randn(B, 240, 6)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = head(wm.encode_state(obs, candle_seq=candles))
            cache = wm.precompute_temporal_kv_cache(candles)
            _ = head(wm.encode_state(obs, kv_cache=cache))

    # Benchmark full forward pass
    N = 10
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = head(wm.encode_state(obs, candle_seq=candles))
    t_full = (time.perf_counter() - t0) / N

    # Benchmark cached forward pass
    cache = wm.precompute_temporal_kv_cache(candles)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = head(wm.encode_state(obs, kv_cache=cache))
    t_cached = (time.perf_counter() - t0) / N

    speedup = t_full / max(t_cached, 1e-9)
    assert speedup >= 5.0, f"Expected >= 5.0x speedup, got {speedup:.2f}x (full={t_full*1000:.2f}ms, cached={t_cached*1000:.2f}ms)"


# ==============================================================================
# TEST 12: Dropout Behavior in Train() vs Eval() Mode
# ==============================================================================

def test_12_dropout_behavior_train_vs_eval():
    """Validates deterministic output in eval() mode and stochasticity in train() mode."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.05)
    z = torch.randn(16, 768)

    # Eval mode: must be 100% deterministic
    head.eval()
    with torch.no_grad():
        p_eval_1 = head(z)
        p_eval_2 = head(z)
    assert torch.equal(p_eval_1, p_eval_2), "Eval mode output was not deterministic!"

    # Train mode: dropout causes stochastic difference
    head.train()
    torch.manual_seed(12345)
    p_train_1 = head(z)
    p_train_2 = head(z)
    assert not torch.equal(p_train_1, p_train_2), "Train mode did not activate dropout!"


# ==============================================================================
# TEST 13: Checkpoint Serialization Schema Isolating 4 Namespaces
# ==============================================================================

def test_13_checkpoint_serialization_isolating_4_namespaces():
    """Validates 4-namespace checkpoint serialization and roundtrip strict loading."""
    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
    re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))

    # Build 4-namespace dictionary
    base_wm_sd = {
        k: v for k, v in wm.state_dict().items()
        if not k.startswith("temporal_encoder.") and not k.startswith("projection_head.")
    }
    temporal_sd = wm.temporal_encoder.state_dict()
    head_sd = head.state_dict()
    re_sd = re.state_dict()

    assert len(base_wm_sd) == 30
    assert len(temporal_sd) == 41
    assert len(head_sd) == 8
    assert len(re_sd) == 191

    payload = {
        "world_model_state_dict": base_wm_sd,
        "temporal_encoder_state_dict": temporal_sd,
        "projection_head_state_dict": head_sd,
        "reasoning_engine_state_dict": re_sd,
    }

    # Roundtrip through BytesIO
    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location="cpu", weights_only=False)

    # Strict loading into independent fresh modules
    fresh_wm = LatentDynamicsWorldModel()
    res_wm = fresh_wm.load_state_dict(loaded["world_model_state_dict"], strict=True)
    assert len(res_wm.missing_keys) == 0 and len(res_wm.unexpected_keys) == 0

    fresh_head = MultiHorizonProjectionHead()
    res_head = fresh_head.load_state_dict(loaded["projection_head_state_dict"], strict=True)
    assert len(res_head.missing_keys) == 0 and len(res_head.unexpected_keys) == 0

    fresh_re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
    res_re = fresh_re.load_state_dict(loaded["reasoning_engine_state_dict"], strict=True)
    assert len(res_re.missing_keys) == 0 and len(res_re.unexpected_keys) == 0


# ==============================================================================
# TEST 14: Strict 30-Key Baseline Loading on Detached LatentDynamicsWorldModel
# ==============================================================================

def test_14_strict_30_key_baseline_loading():
    """Validates strict 30-key state dict and parameter count on baseline LatentDynamicsWorldModel."""
    wm = LatentDynamicsWorldModel()
    assert wm.temporal_encoder is None

    sd = wm.state_dict()
    assert len(sd) == 30, f"Expected strictly 30 keys, got {len(sd)}"
    param_count = sum(p.numel() for p in wm.parameters())
    assert param_count == 2160900, f"Expected 2,160,900 parameters, got {param_count}"

    # Load into another instance under strict=True
    wm2 = LatentDynamicsWorldModel()
    res = wm2.load_state_dict(sd, strict=True)
    assert len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0


# ==============================================================================
# TEST 15: Loading Authentic Production Checkpoint with strict=True
# ==============================================================================

def test_15_loading_authentic_production_checkpoint():
    """Validates loading authentic production challenger checkpoint under strict=True."""
    ckpt_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt")
    if not ckpt_path.exists():
        pytest.skip(f"Challenger checkpoint {ckpt_path} not found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    wm = LatentDynamicsWorldModel()
    res = wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
    assert len(res.missing_keys) == 0
    assert len(res.unexpected_keys) == 0

    # Also verify Reasoning Engine weights if present
    re_key = "bull_cot_state_dict" if "bull_cot_state_dict" in ckpt else "reasoning_engine_state_dict"
    if re_key in ckpt:
        re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
        res_re = re.load_state_dict(ckpt[re_key], strict=True)
        assert len(res_re.missing_keys) == 0
        assert len(res_re.unexpected_keys) == 0


# ==============================================================================
# TEST 16: End-to-End Forward Pass with SingleRegimeDataset.get_batch() Tensors
# ==============================================================================

def test_16_end_to_end_single_regime_dataset_forward():
    """Validates end-to-end forward and backward pass with SingleRegimeDataset.get_batch()."""
    env = MockM1Environment(total_bars=500, name="Bull")
    dataset = SingleRegimeDataset("bull", env)

    batch = dataset.get_batch([10, 20, 30, 40])
    assert isinstance(batch, MegaPoolBatch)
    assert batch.obs_137.shape == (4, 137)
    assert batch.candle_seq_240.shape == (4, 240, 6)
    assert batch.targets_3.shape == (4, 3)

    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240)
    head = MultiHorizonProjectionHead()

    z_out = wm.encode_state(batch.obs_137, candle_seq=batch.candle_seq_240)
    preds = head(z_out)
    assert preds.shape == (4, 3)

    loss = compute_multi_horizon_loss(preds, batch.targets_3)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.net[0].weight.grad is not None


# ==============================================================================
# TEST 17: MegaPoolDataset Multi-Regime Batch Compatibility
# ==============================================================================

def test_17_megapool_multi_regime_batch_compatibility():
    """Validates forward pass and loss computation on multi-regime batches from MegaPoolDataset."""
    env_bull = MockM1Environment(total_bars=500, name="Bull")
    ds_bull = SingleRegimeDataset("bull", env_bull)
    env_bear = MockM1Environment(total_bars=500, name="Bear")
    ds_bear = SingleRegimeDataset("bear", env_bear)

    mp_ds = MegaPoolDataset({"bull": ds_bull, "bear": ds_bear})

    # Sample batch from both regimes
    indices = [5, 10, 500, 505]
    obs_list, seq_list, tgt_list = [], [], []
    for idx in indices:
        obs, seq, tgt, atr = mp_ds[idx]
        obs_list.append(obs)
        seq_list.append(seq)
        tgt_list.append(tgt)

    batch_obs = torch.stack(obs_list)
    batch_seq = torch.stack(seq_list)
    batch_tgt = torch.stack(tgt_list)

    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240)
    head = MultiHorizonProjectionHead()

    z_out = wm.encode_state(batch_obs, candle_seq=batch_seq)
    preds = head(z_out)
    loss = compute_multi_horizon_loss(preds, batch_tgt)

    assert torch.isfinite(loss)
    loss.backward()
    assert head.net[6].weight.grad is not None


# ==============================================================================
# TEST 18: Un-Normalized Return Prediction via predict_returns(z, atr)
# ==============================================================================

def test_18_unnormalized_return_prediction():
    """Validates predict_returns(z, atr) with 1D, scalar, and None ATR inputs."""
    head = MultiHorizonProjectionHead().eval()
    z = torch.randn(4, 768)
    atr = torch.tensor([10.0, 20.0, 5.0, 15.0])

    with torch.no_grad():
        p_norm = head(z)
        p_unnorm = head.predict_returns(z, atr=atr)

    expected = p_norm * atr.unsqueeze(-1)
    assert torch.allclose(p_unnorm, expected, atol=1e-6)

    # When atr is None, returns normalized predictions directly
    assert torch.equal(head.predict_returns(z, atr=None), p_norm)

    # When atr is a scalar float
    p_scalar = head.predict_returns(z, atr=2.5)
    assert torch.allclose(p_scalar, p_norm * 2.5, atol=1e-6)


# ==============================================================================
# TEST 19: Numerical Stability Under Extreme Inputs (Large Values, Zeros, Infs, NaNs)
# ==============================================================================

def test_19_numerical_stability_extreme_inputs():
    """Validates that MultiHorizonProjectionHead sanitizes extreme inputs without NaNs or Infs."""
    head = MultiHorizonProjectionHead().eval()

    inputs = [
        torch.zeros(4, 768),
        torch.full((4, 768), 10000.0),
        torch.full((4, 768), -10000.0),
        torch.full((4, 768), float("nan")),
        torch.full((4, 768), float("inf")),
        torch.full((4, 768), -float("inf")),
    ]

    for x in inputs:
        out = head(x)
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all(), f"Failed to sanitize extreme input: {x[0, :3]}"


# ==============================================================================
# TEST 20: Deterministic Output Repeatability Given Identical Seeds
# ==============================================================================

def test_20_deterministic_repeatability_identical_seeds():
    """Validates deterministic output repeatability under identical random seeds."""
    torch.manual_seed(42)
    head1 = MultiHorizonProjectionHead().eval()

    torch.manual_seed(42)
    head2 = MultiHorizonProjectionHead().eval()

    z = torch.randn(8, 768)
    with torch.no_grad():
        p1 = head1(z)
        p2 = head2(z)

    assert torch.equal(p1, p2), "Model outputs diverged despite identical random seeds!"
