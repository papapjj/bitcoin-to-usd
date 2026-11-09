# -*- coding: utf-8 -*-
"""
Empirical Challenger Suite: Milestone 2 (Multi-Horizon Projection Head & TCA Integration)
Author: teamwork_challenger_tca_m2_2 (Empirical Challenger 2)
Scope:
  1. Independent Naive Reference Oracle comparison for MultiHorizonProjectionHead & Huber Loss (1,000 vectors).
  2. KV-Cache bitwise equivalence (50 distinct sequences) & CPU throughput benchmark (>5x speedup).
  3. 4-namespace checkpoint serialization zero-drift & authentic production checkpoint loading.
  4. Exact parameter and topological structural invariants.
"""

import io
import math
import time
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import pytest
from scipy.special import erf
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    MultiHorizonProjectionHead,
    create_projection_head,
    compute_multi_horizon_loss,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)


# ==============================================================================
# NAIVE REFERENCE ORACLE IMPLEMENTATIONS (Pure NumPy & Standard Library)
# ==============================================================================

def naive_oracle_projection_head_forward(
    z: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    ln_w: np.ndarray,
    ln_b: np.ndarray,
    w4: np.ndarray,
    b4: np.ndarray,
    w6: np.ndarray,
    b6: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Pure NumPy implementation of MultiHorizonProjectionHead in eval() mode.
    Exact equation:
      clean_z = nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
      h0 = clean_z @ W0.T + b0
      norm0 = (h0 - mean(h0)) / sqrt(var(h0) + eps) * ln_w + ln_b
      gelu0 = norm0 * 0.5 * (1.0 + erf(norm0 / sqrt(2.0)))
      h4 = gelu0 @ W4.T + b4
      gelu4 = h4 * 0.5 * (1.0 + erf(h4 / sqrt(2.0)))
      out = gelu4 @ W6.T + b6
    """
    # 1. Input sanitization
    clean_z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
    original_1d = False
    if clean_z.ndim == 1:
        clean_z = clean_z[None, :]
        original_1d = True

    # 2. Linear layer 0: [B, 768] -> [B, 256]
    h0 = clean_z @ w0.T + b0

    # 3. LayerNorm: eps=1e-5, biased sample variance matching PyTorch
    mean = np.mean(h0, axis=-1, keepdims=True)
    var = np.var(h0, axis=-1, keepdims=True)
    norm0 = (h0 - mean) / np.sqrt(var + eps) * ln_w + ln_b

    # 4. GELU (exact standard formulation, approximate='none')
    gelu0 = norm0 * 0.5 * (1.0 + erf(norm0 / np.sqrt(2.0)))

    # 5. Dropout is identity in eval() mode

    # 6. Linear layer 4: [B, 256] -> [B, 128]
    h4 = gelu0 @ w4.T + b4

    # 7. GELU
    gelu4 = h4 * 0.5 * (1.0 + erf(h4 / np.sqrt(2.0)))

    # 8. Linear layer 6: [B, 128] -> [B, 3]
    out = gelu4 @ w6.T + b6

    if original_1d:
        return out
    return out


def naive_oracle_huber_loss(
    pred: np.ndarray,
    tgt: np.ndarray,
    beta: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Pure NumPy Huber / Smooth L1 loss formulation:
      L_beta(x, y) = 0.5 * (x - y)^2 / beta  if |x - y| < beta
                     else |x - y| - 0.5 * beta
      loss = mean(L_beta * weights)
    """
    clean_p = np.nan_to_num(pred, nan=0.0, posinf=50.0, neginf=-50.0)
    clean_t = np.nan_to_num(tgt, nan=0.0, posinf=50.0, neginf=-50.0)

    if clean_p.ndim == 1:
        clean_p = clean_p[None, :]
    if clean_t.ndim == 1:
        clean_t = clean_t[None, :]

    diff = np.abs(clean_p - clean_t)
    loss = np.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)

    if weights is not None:
        if weights.ndim == 1:
            weights = weights[None, :]
        loss = loss * weights

    return float(np.mean(loss))


# ==============================================================================
# TEST 1: Naive Reference Oracle vs MultiHorizonProjectionHead (1,000 Vectors)
# ==============================================================================

def test_01_naive_oracle_projection_head_1000_vectors():
    """
    Stress-tests MultiHorizonProjectionHead forward pass against the pure NumPy naive oracle
    across 1,000 randomized test vectors including standard, extreme, and corrupted values.
    Asserts max numerical divergence Delta <= 1e-6.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.05).eval()
    sd = head.state_dict()

    w0 = sd["net.0.weight"].numpy()
    b0 = sd["net.0.bias"].numpy()
    ln_w = sd["net.1.weight"].numpy()
    ln_b = sd["net.1.bias"].numpy()
    w4 = sd["net.4.weight"].numpy()
    b4 = sd["net.4.bias"].numpy()
    w6 = sd["net.6.weight"].numpy()
    b6 = sd["net.6.bias"].numpy()

    # Generate 1,000 randomized vectors across diverse distributions
    # 1. 400 normal vectors
    v_normal = np.random.randn(400, 768).astype(np.float32)
    # 2. 200 uniform vectors [-5, 5]
    v_uniform = np.random.uniform(-5.0, 5.0, (200, 768)).astype(np.float32)
    # 3. 200 high-magnitude vectors [-50, 50]
    v_high = np.random.uniform(-50.0, 50.0, (200, 768)).astype(np.float32)
    # 4. 100 near-zero vectors
    v_near_zero = (np.random.randn(100, 768) * 1e-4).astype(np.float32)
    # 5. 100 vectors containing NaNs, +Infs, -Infs to stress test sanitization
    v_corrupt = np.random.randn(100, 768).astype(np.float32)
    for row in range(100):
        nan_idx = np.random.choice(768, 5, replace=False)
        posinf_idx = np.random.choice(768, 5, replace=False)
        neginf_idx = np.random.choice(768, 5, replace=False)
        v_corrupt[row, nan_idx] = np.nan
        v_corrupt[row, posinf_idx] = np.inf
        v_corrupt[row, neginf_idx] = -np.inf

    all_vectors = np.concatenate([v_normal, v_uniform, v_high, v_near_zero, v_corrupt], axis=0)
    assert all_vectors.shape == (1000, 768)

    max_delta = 0.0

    # Test individual vectors (batch_size=1) and small batches
    for i in range(1000):
        vec = all_vectors[i]  # shape [768]
        with torch.no_grad():
            prod_out = head(torch.from_numpy(vec)).numpy()  # shape [1, 3]

        oracle_out = naive_oracle_projection_head_forward(
            vec, w0, b0, ln_w, ln_b, w4, b4, w6, b6
        )

        delta = float(np.max(np.abs(prod_out - oracle_out)))
        if delta > max_delta:
            max_delta = delta

        assert delta <= 1e-6, f"Vector {i} exceeded tolerance: Delta = {delta:.8e} > 1e-6"

    # Also test batch forward pass (1000 x 768)
    with torch.no_grad():
        batch_prod = head(torch.from_numpy(all_vectors)).numpy()
    batch_oracle = naive_oracle_projection_head_forward(
        all_vectors, w0, b0, ln_w, ln_b, w4, b4, w6, b6
    )
    batch_delta = float(np.max(np.abs(batch_prod - batch_oracle)))

    assert batch_delta <= 1e-6, f"Batch Delta = {batch_delta:.8e} > 1e-6"
    print(f"\n[Test 1] 1,000 Vectors Oracle Max Delta: {max_delta:.8e} <= 1e-6 (PASSED)")


# ==============================================================================
# TEST 2: Naive Reference Oracle vs compute_multi_horizon_loss (1,000 Pairs)
# ==============================================================================

def test_02_naive_oracle_huber_loss_1000_pairs():
    """
    Stress-tests compute_multi_horizon_loss against pure NumPy naive Huber formulation
    across 1,000 randomized pairs of (pred, tgt) spanning:
      - Quadratic regime (|d| < beta)
      - Linear regime (|d| >= beta)
      - Transition boundary (|d| == beta)
      - Diverse weight configurations
    Asserts max numerical divergence Delta <= 1e-6 across all 1,000 pairs,
    and exact 0.0 elementwise mathematical identity before reduction.
    """
    torch.manual_seed(101)
    np.random.seed(101)

    max_delta = 0.0
    max_elem_delta = 0.0

    for i in range(1000):
        b_size = np.random.randint(1, 16)
        beta = float(np.random.choice([0.5, 1.0, 1.5, 2.0]))

        # Returns normalized by ATR are bounded in [-2.5, 2.5]
        p = np.random.uniform(-2.5, 2.5, (b_size, 3)).astype(np.float32)
        t = np.random.uniform(-2.5, 2.5, (b_size, 3)).astype(np.float32)

        # Inject specific conditions
        if i % 4 == 0:
            # Quadratic regime: small diffs
            t = p + np.random.uniform(-0.4 * beta, 0.4 * beta, (b_size, 3)).astype(np.float32)
        elif i % 4 == 1:
            # Linear regime: larger diffs
            t = p + np.random.uniform(1.2 * beta, 2.5 * beta, (b_size, 3)).astype(np.float32)
        elif i % 4 == 2:
            # Exact boundary: diff == beta
            t = p + beta

        # Weights
        use_weights = (i % 2 == 0)
        if use_weights:
            weights = np.random.uniform(0.5, 1.5, (3,)).astype(np.float32)
            torch_weights = torch.from_numpy(weights)
        else:
            weights = None
            torch_weights = None

        # Verify elementwise exactness
        with torch.no_grad():
            cp = torch.nan_to_num(torch.from_numpy(p), nan=0.0, posinf=50.0, neginf=-50.0)
            ct = torch.nan_to_num(torch.from_numpy(t), nan=0.0, posinf=50.0, neginf=-50.0)
            t_elem = F.smooth_l1_loss(cp, ct, beta=beta, reduction="none").numpy()

        np_cp = np.nan_to_num(p, nan=0.0, posinf=50.0, neginf=-50.0)
        np_ct = np.nan_to_num(t, nan=0.0, posinf=50.0, neginf=-50.0)
        np_diff = np.abs(np_cp - np_ct)
        np_elem = np.where(np_diff < beta, 0.5 * (np_diff ** 2) / beta, np_diff - 0.5 * beta)

        elem_diff = float(np.max(np.abs(t_elem - np_elem)))
        if elem_diff > max_elem_delta:
            max_elem_delta = elem_diff
        assert elem_diff == 0.0, f"Elementwise discrepancy at pair {i}: {elem_diff}"

        # Verify reduced scalar loss
        torch_loss = compute_multi_horizon_loss(
            torch.from_numpy(p),
            torch.from_numpy(t),
            beta=beta,
            weights=torch_weights,
        ).item()

        oracle_loss = naive_oracle_huber_loss(
            p,
            t,
            beta=beta,
            weights=weights,
        )

        delta = abs(torch_loss - oracle_loss)
        if delta > max_delta:
            max_delta = delta

        assert delta <= 1e-6, f"Pair {i} exceeded tolerance: Delta = {delta:.8e} > 1e-6 (torch={torch_loss}, oracle={oracle_loss})"

    # Separate verification for NaN/Inf sanitization consistency
    p_corrupt = np.array([[np.nan, 1.0, -np.inf]], dtype=np.float32)
    t_corrupt = np.array([[0.5, np.inf, 2.0]], dtype=np.float32)
    t_loss_c = compute_multi_horizon_loss(torch.from_numpy(p_corrupt), torch.from_numpy(t_corrupt), beta=1.0).item()
    o_loss_c = naive_oracle_huber_loss(p_corrupt, t_corrupt, beta=1.0)
    assert abs(t_loss_c - o_loss_c) <= 1e-6

    print(f"[Test 2] 1,000 Huber Pairs Oracle: Max Delta = {max_delta:.8e} <= 1e-6, Elementwise Delta = {max_elem_delta} (PASSED)")


# ==============================================================================
# TEST 3: KV-Cache Bitwise Equivalence (50 Distinct Sequences)
# ==============================================================================

def test_03_kv_cache_bitwise_equivalence_50_sequences():
    """
    Empirically validates exact 0.0 bitwise equivalence between full forward pass
    and KV-cached forward pass across 50 distinct randomized 240-candle sequences.
    Asserts max(|z_cache - z_full|) == 0.0 and max(|pred_cache - pred_full|) == 0.0.
    """
    torch.manual_seed(202)

    wm = LatentDynamicsWorldModel().eval()
    wm.enable_temporal_stream(candle_len=240)
    wm.eval()

    head = MultiHorizonProjectionHead().eval()

    max_z_diff = 0.0
    max_pred_diff = 0.0

    for seq_idx in range(50):
        b_size = 1 if seq_idx % 2 == 0 else 2
        obs = torch.randn(b_size, 137)
        candles = torch.randn(b_size, 240, 6)

        with torch.no_grad():
            # 1. Full sequence pass
            z_full = wm.encode_state(obs, candle_seq=candles)
            pred_full = head(z_full)

            # 2. KV-cached pass
            cache = wm.precompute_temporal_kv_cache(candles)
            assert cache is not None
            z_cached = wm.encode_state(obs, kv_cache=cache)
            pred_cached = head(z_cached)

        z_diff = torch.max(torch.abs(z_full - z_cached)).item()
        pred_diff = torch.max(torch.abs(pred_full - pred_cached)).item()

        if z_diff > max_z_diff:
            max_z_diff = z_diff
        if pred_diff > max_pred_diff:
            max_pred_diff = pred_diff

        assert z_diff == 0.0, f"Sequence {seq_idx} had non-zero z diff: {z_diff}"
        assert pred_diff == 0.0, f"Sequence {seq_idx} had non-zero pred diff: {pred_diff}"

    assert max_z_diff == 0.0
    assert max_pred_diff == 0.0
    print(f"[Test 3] 50 Sequences KV-Cache Bitwise Equivalence: max_diff = {max_z_diff} (PASSED)")


# ==============================================================================
# TEST 4: KV-Cache Throughput Benchmark (>5x Speedup on CPU)
# ==============================================================================

def test_04_kv_cache_throughput_benchmark_speedup():
    """
    Empirically benchmarks execution latency of full sequence forward pass vs
    KV-cached forward pass on CPU, asserting acceleration factor > 5.0x.
    """
    torch.manual_seed(303)

    wm = LatentDynamicsWorldModel().eval()
    wm.enable_temporal_stream(candle_len=240)
    wm.eval()

    head = MultiHorizonProjectionHead().eval()

    B = 2
    obs = torch.randn(B, 137)
    candles = torch.randn(B, 240, 6)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = head(wm.encode_state(obs, candle_seq=candles))
            cache = wm.precompute_temporal_kv_cache(candles)
            _ = head(wm.encode_state(obs, kv_cache=cache))

    # Benchmark full sequence forward pass
    N = 25
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = head(wm.encode_state(obs, candle_seq=candles))
    t_full = (time.perf_counter() - t0) / N

    # Benchmark KV-cached forward pass
    cache = wm.precompute_temporal_kv_cache(candles)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = head(wm.encode_state(obs, kv_cache=cache))
    t_cached = (time.perf_counter() - t0) / N

    speedup = t_full / max(t_cached, 1e-9)
    print(f"\n[Test 4] Benchmark: Full = {t_full*1000:.3f} ms | Cached = {t_cached*1000:.3f} ms | Speedup = {speedup:.2f}x")

    assert speedup > 5.0, f"Expected > 5.0x speedup, got {speedup:.2f}x"


# ==============================================================================
# TEST 5: 4-Namespace Checkpoint Serialization & 0.0 Parameter Drift
# ==============================================================================

def test_05_checkpoint_serialization_4_namespaces_zero_drift():
    """
    Empirically validates that serializing and deserializing a 4-namespace checkpoint
    (world_model, temporal_encoder, projection_head, reasoning_engine) produces
    exact 0.0 parameter drift across 100% of weights.
    """
    torch.manual_seed(404)

    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
    re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))

    # Snapshot baseline state dicts
    base_wm_sd = {
        k: v.clone() for k, v in wm.state_dict().items()
        if not k.startswith("temporal_encoder.") and not k.startswith("projection_head.")
    }
    te_sd = {k: v.clone() for k, v in wm.temporal_encoder.state_dict().items()}
    head_sd = {k: v.clone() for k, v in head.state_dict().items()}
    re_sd = {k: v.clone() for k, v in re.state_dict().items()}

    assert len(base_wm_sd) == 30
    assert len(te_sd) == 41
    assert len(head_sd) == 8
    assert len(re_sd) == 191

    payload = {
        "world_model_state_dict": base_wm_sd,
        "temporal_encoder_state_dict": te_sd,
        "projection_head_state_dict": head_sd,
        "reasoning_engine_state_dict": re_sd,
    }

    # Roundtrip through BytesIO stream
    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location="cpu", weights_only=False)

    # 1. Verify World Model drift
    fresh_wm = LatentDynamicsWorldModel()
    res_wm = fresh_wm.load_state_dict(loaded["world_model_state_dict"], strict=True)
    assert len(res_wm.missing_keys) == 0 and len(res_wm.unexpected_keys) == 0
    wm_drift = max(torch.max(torch.abs(v - base_wm_sd[k])).item() for k, v in fresh_wm.state_dict().items())
    assert wm_drift == 0.0, f"World Model had drift: {wm_drift}"

    # 2. Verify Temporal Cross Attention Encoder drift
    fresh_te = TemporalCrossAttentionEncoder()
    res_te = fresh_te.load_state_dict(loaded["temporal_encoder_state_dict"], strict=True)
    assert len(res_te.missing_keys) == 0 and len(res_te.unexpected_keys) == 0
    te_drift = max(torch.max(torch.abs(v - te_sd[k])).item() for k, v in fresh_te.state_dict().items())
    assert te_drift == 0.0, f"Temporal Encoder had drift: {te_drift}"

    # 3. Verify Multi-Horizon Projection Head drift
    fresh_head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
    res_head = fresh_head.load_state_dict(loaded["projection_head_state_dict"], strict=True)
    assert len(res_head.missing_keys) == 0 and len(res_head.unexpected_keys) == 0
    head_drift = max(torch.max(torch.abs(v - head_sd[k])).item() for k, v in fresh_head.state_dict().items())
    assert head_drift == 0.0, f"Projection Head had drift: {head_drift}"

    # 4. Verify Metacognitive Reasoning Engine drift
    fresh_re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
    res_re = fresh_re.load_state_dict(loaded["reasoning_engine_state_dict"], strict=True)
    assert len(res_re.missing_keys) == 0 and len(res_re.unexpected_keys) == 0
    re_drift = max(torch.max(torch.abs(v - re_sd[k])).item() for k, v in fresh_re.state_dict().items())
    assert re_drift == 0.0, f"Reasoning Engine had drift: {re_drift}"

    print(f"[Test 5] 4-Namespace Drift: WM={wm_drift}, TE={te_drift}, Head={head_drift}, RE={re_drift} (PASSED)")


# ==============================================================================
# TEST 6: Authentic Production Checkpoint Compatibility (strict=True)
# ==============================================================================

def test_06_authentic_checkpoint_backward_compatibility():
    """
    Validates loading authentic production challenger checkpoint into baseline
    LatentDynamicsWorldModel with strict=True, asserting 0 missing and 0 unexpected keys.
    """
    ckpt_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt")
    if not ckpt_path.exists():
        pytest.skip(f"Challenger checkpoint {ckpt_path} not found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "world_model_state_dict" in ckpt

    wm = LatentDynamicsWorldModel()
    res = wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
    assert len(res.missing_keys) == 0, f"Missing keys: {res.missing_keys}"
    assert len(res.unexpected_keys) == 0, f"Unexpected keys: {res.unexpected_keys}"

    # Forward pass sanity check
    obs = torch.randn(2, 137)
    with torch.no_grad():
        z_init = wm.encode_state(obs)
    assert z_init.shape == (2, 768)

    print(f"[Test 6] Authentic Production Checkpoint loaded with strict=True: 0 missing, 0 unexpected (PASSED)")


# ==============================================================================
# TEST 7: Structural Parameter & Invariance Auditing
# ==============================================================================

def test_07_structural_parameter_invariants():
    """
    Audits exact parameter counts, layer types, and topological invariants:
      - LatentDynamicsWorldModel detached: exactly 30 keys, 2,160,900 parameters
      - MultiHorizonProjectionHead: exactly 8 tensors, 230,659 parameters
      - MetacognitiveReasoningEngine: exactly 191 keys, 4,196,564 parameters
    """
    # 1. Baseline WM
    wm = LatentDynamicsWorldModel()
    assert wm.temporal_encoder is None
    wm_sd = wm.state_dict()
    assert len(wm_sd) == 30, f"Expected 30 keys, got {len(wm_sd)}"
    wm_params = sum(p.numel() for p in wm.parameters())
    assert wm_params == 2160900, f"Expected 2,160,900 params, got {wm_params}"

    # 2. Projection Head
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
    head_sd = head.state_dict()
    assert len(head_sd) == 8, f"Expected 8 tensors, got {len(head_sd)}"
    head_params = sum(p.numel() for p in head.parameters())
    assert head_params == 230659, f"Expected 230,659 params, got {head_params}"

    # Expected shape verification
    assert head_sd["net.0.weight"].shape == (256, 768)
    assert head_sd["net.0.bias"].shape == (256,)
    assert head_sd["net.1.weight"].shape == (256,)
    assert head_sd["net.1.bias"].shape == (256,)
    assert head_sd["net.4.weight"].shape == (128, 256)
    assert head_sd["net.4.bias"].shape == (128,)
    assert head_sd["net.6.weight"].shape == (3, 128)
    assert head_sd["net.6.bias"].shape == (3,)

    # 3. Reasoning Engine
    re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
    re_sd = re.state_dict()
    assert len(re_sd) == 191, f"Expected 191 keys, got {len(re_sd)}"
    re_params = sum(p.numel() for p in re.parameters())
    assert re_params == 4196564, f"Expected 4,196,564 params, got {re_params}"

    print(f"[Test 7] Structural Invariants Verified: WM={wm_params}, Head={head_params}, RE={re_params} (PASSED)")


# ==============================================================================
# TEST 8: predict_returns Un-normalization Oracle & ATR Scaling
# ==============================================================================

def test_08_predict_returns_unnormalization_oracle():
    """
    Validates that predict_returns correctly un-normalizes multi-horizon return
    predictions via unnorm_pred = pred * atr across scalars, 1D arrays, and tensors.
    """
    head = MultiHorizonProjectionHead().eval()
    z = torch.randn(4, 768)

    with torch.no_grad():
        raw_pred = head(z)

        # 1. Scalar float ATR
        atr_scalar = 2.5
        unnorm_scalar = head.predict_returns(z, atr=atr_scalar)
        expected_scalar = raw_pred * atr_scalar
        assert torch.allclose(unnorm_scalar, expected_scalar, atol=1e-6)

        # 2. NumPy array ATR of shape [4]
        atr_np = np.array([1.2, 2.4, 0.8, 3.1], dtype=np.float32)
        unnorm_np = head.predict_returns(z, atr=atr_np)
        expected_np = raw_pred * torch.tensor(atr_np).unsqueeze(-1)
        assert torch.allclose(unnorm_np, expected_np, atol=1e-6)

        # 3. Torch tensor ATR of shape [4, 1]
        atr_t = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
        unnorm_t = head.predict_returns(z, atr=atr_t)
        expected_t = raw_pred * atr_t
        assert torch.allclose(unnorm_t, expected_t, atol=1e-6)

    print(f"[Test 8] Return Un-normalization via ATR Oracle Verified (PASSED)")
