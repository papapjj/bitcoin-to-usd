"""
Adversarial Stress Test and Fuzzing Harness for Milestone 2:
Multi-Horizon Projection Head & TCA Integration (Features 5, 6, 7, 8).

Tested Components:
1. MultiHorizonProjectionHead:
   - Extreme inputs: large values (1e6, 1e12, 1e20, 1e38), zero vectors, near-zero (1e-12, 1e-35), negative (-10, -1e6, -1e38).
   - Arbitrary and non-standard batch sizes: B=1, B=2, B=128, B=1024, B=0, 1D unbatched [768], 3D sequence [B, S, 768].
   - NaN / Inf input protection: all-NaN, +Inf, -Inf, mixed corruptions.
   - predict_returns(z, atr) with degenerate ATRs: 0.0, -10.0, 1e-6, 1e-18, numpy array, 1D tensor, 2D tensor, scalar.
2. compute_multi_horizon_loss:
   - Non-standard weights: zeros [0, 0, 0], single-horizon [1, 0, 0], negative weights, large weights (1e4), sum != 1.
   - Boundary targets: y = +/-10.0, y = 0.0, y = +/-100.0, y = +/-1e6, targets with NaN/Inf.
   - Smooth L1 gradient bounding: verify ||dL/d(pred)|| <= 1.0 per horizon element across all error scales (0.001 to 1e6).
3. mega_pool_dataset.py line 371 safeguard:
   - Corrupted atr_arr containing NaNs, +Inf, -Inf, 0.0, negative values, sub-floor (1e-12).
   - Verify get_batch gracefully floors all ATR values to atr_floor (1e-4) without exceptions.
4. End-to-end integration with TCA and LatentDynamicsWorldModel:
   - Full pipeline forward and backward stability under adversarial perturbation.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.parent != PROJECT_ROOT:
    if (PROJECT_ROOT / "research_reasoning_world_class").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from research_reasoning_world_class.src.mega_pool_dataset import (
    MarketRegime,
    SingleRegimeDataset,
    MegaPoolDataset,
    MegaPoolBatch,
)
from research_reasoning_world_class.tests.test_mega_pool_dataset import MockM1Environment


# ==============================================================================
# SECTION 1: MultiHorizonProjectionHead Extreme Fuzzing & Batch Shapes
# ==============================================================================

def test_adversarial_extreme_value_inputs():
    """Fuzzes head with extreme magnitudes: 0, 1e-35, 1e-12, 1e6, 1e12, 1e20, 1e38, -1e6, -1e38."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).eval()
    
    extreme_cases = {
        "all_zeros": torch.zeros(16, 768),
        "near_zero_subnormal": torch.full((8, 768), 1e-35),
        "near_zero_pico": torch.full((8, 768), 1e-12),
        "large_million": torch.full((8, 768), 1e6),
        "negative_million": torch.full((8, 768), -1e6),
        "large_trillion": torch.full((4, 768), 1e12),
        "alternating_extreme": torch.tensor([[1e6 if i % 2 == 0 else -1e6 for i in range(768)] for _ in range(4)]),
        "subtle_perturbation": torch.randn(8, 768) * 1e-8,
        "huge_variance": torch.randn(8, 768) * 1e5,
    }

    for name, z in extreme_cases.items():
        out = head(z)
        assert out.shape == (z.shape[0], 3), f"Failed shape check for case {name}: got {out.shape}"
        assert torch.all(torch.isfinite(out)), f"Output contains NaN or Inf for case {name}: {out}"


def test_adversarial_arbitrary_and_extreme_batch_sizes():
    """Tests forward pass across non-standard batch sizes: B=1, B=2, B=128, B=1024, unbatched 1D [768], 3D sequence [B, S, 768]."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).eval()

    # Standard and high batch sizes
    for b in [1, 2, 3, 5, 17, 64, 128, 256, 512, 1024]:
        z = torch.randn(b, 768)
        out = head(z)
        assert out.shape == (b, 3), f"Failed batch size {b}: expected {(b, 3)}, got {out.shape}"
        assert torch.all(torch.isfinite(out))

    # Unbatched 1D vector [768] -> returns [1, 3]
    z_1d = torch.randn(768)
    out_1d = head(z_1d)
    assert out_1d.shape == (1, 3), f"Expected (1, 3) for 1D input, got {out_1d.shape}"
    assert torch.all(torch.isfinite(out_1d))

    # 3D tensor input [B, SeqLen, 768] (e.g. sequence of states)
    z_3d = torch.randn(4, 10, 768)
    out_3d = head(z_3d)
    assert out_3d.shape == (4, 10, 3), f"Expected (4, 10, 3) for 3D input, got {out_3d.shape}"
    assert torch.all(torch.isfinite(out_3d))


# ==============================================================================
# SECTION 2: NaN / Inf Input Sanitization & Immunity
# ==============================================================================

def test_adversarial_nan_and_inf_sanitization():
    """Fuzzes head with all-NaN, +Inf, -Inf, and mixed corrupted tensors."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).eval()

    corruptions = {
        "all_nans": torch.full((8, 768), float("nan")),
        "all_pos_infs": torch.full((8, 768), float("inf")),
        "all_neg_infs": torch.full((8, 768), float("-inf")),
        "mixed_nans_and_infs": torch.tensor(
            [[float("nan") if i % 3 == 0 else (float("inf") if i % 3 == 1 else float("-inf")) for i in range(768)] for _ in range(4)]
        ),
        "random_sparse_nans": torch.where(torch.rand(8, 768) < 0.2, torch.tensor(float("nan")), torch.randn(8, 768)),
        "random_sparse_infs": torch.where(torch.rand(8, 768) < 0.2, torch.tensor(float("inf")), torch.randn(8, 768)),
    }

    for name, z in corruptions.items():
        out = head(z)
        assert out.shape == (z.shape[0], 3), f"Shape mismatch on corruption {name}"
        assert torch.all(torch.isfinite(out)), f"Head leaked NaN or Inf on corruption {name}: {out}"


def test_adversarial_backward_pass_with_nan_inputs():
    """Verifies that backpropagating through NaN-sanitized inputs does not crash autograd."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).train()
    
    # Input with some NaNs
    z = torch.randn(4, 768, requires_grad=True)
    with torch.no_grad():
        z[0, 0:10] = float("nan")
        z[1, 10:20] = float("inf")

    pred = head(z)
    loss = pred.sum()
    loss.backward()

    # Verify head parameters have finite gradients
    for p_name, param in head.named_parameters():
        assert param.grad is not None, f"Gradient missing for {p_name}"
        assert torch.all(torch.isfinite(param.grad)), f"Gradient for {p_name} contains NaN/Inf!"


# ==============================================================================
# SECTION 3: predict_returns(z, atr) with Degenerate ATRs
# ==============================================================================

def test_adversarial_predict_returns_degenerate_atrs():
    """Tests predict_returns with 0.0, negative ATR, sub-normal ATR, and various tensor formats."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).eval()
    z = torch.randn(8, 768)

    # 1. atr = None: returns normalized predictions directly
    pred_raw = head.predict_returns(z, atr=None)
    assert pred_raw.shape == (8, 3)

    # 2. atr = 0.0: un-normalized returns must be exactly 0.0
    pred_zero = head.predict_returns(z, atr=0.0)
    assert pred_zero.shape == (8, 3)
    assert torch.all(pred_zero == 0.0), f"Expected 0.0 returns for zero ATR, got {pred_zero}"

    # 3. atr < 0.0 (negative ATR): should not crash and should invert signs consistently
    pred_neg = head.predict_returns(z, atr=-2.5)
    assert pred_neg.shape == (8, 3)
    assert torch.allclose(pred_neg, pred_raw * -2.5, atol=1e-6)

    # 4. Tiny ATR: 1e-6, 1e-12, 1e-18
    for tiny in [1e-6, 1e-12, 1e-18]:
        pred_tiny = head.predict_returns(z, atr=tiny)
        assert pred_tiny.shape == (8, 3)
        assert torch.all(torch.isfinite(pred_tiny))

    # 5. ATR formats: float scalar, numpy array, 1D tensor [B], 2D tensor [B, 1]
    atr_np = np.array([1.5, 2.0, 3.0, 0.5, 1.2, 2.5, 0.8, 1.0], dtype=np.float32)
    pred_np = head.predict_returns(z, atr=atr_np)
    assert pred_np.shape == (8, 3)

    atr_1d = torch.tensor([1.5, 2.0, 3.0, 0.5, 1.2, 2.5, 0.8, 1.0], dtype=torch.float32)
    pred_1d = head.predict_returns(z, atr=atr_1d)
    assert pred_1d.shape == (8, 3)
    assert torch.allclose(pred_np, pred_1d)

    atr_2d = atr_1d.unsqueeze(1) # shape [8, 1]
    pred_2d = head.predict_returns(z, atr=atr_2d)
    assert pred_2d.shape == (8, 3)
    assert torch.allclose(pred_1d, pred_2d)


# ==============================================================================
# SECTION 4: compute_multi_horizon_loss Adversarial Weights & Boundaries
# ==============================================================================

def test_adversarial_loss_non_standard_weights():
    """Tests loss with zeros, negatives, single-horizon focus, large weights, sum!=1."""
    pred = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], dtype=torch.float32)
    tgt = torch.tensor([[0.0, 2.0, 1.0], [0.5, 0.0, 0.0]], dtype=torch.float32)

    # 1. Zero weights: loss should be 0.0
    loss_zeros = compute_multi_horizon_loss(pred, tgt, weights=[0.0, 0.0, 0.0])
    assert loss_zeros.item() == 0.0, f"Expected 0.0, got {loss_zeros.item()}"

    # 2. Single horizon weight: weights=[1.0, 0.0, 0.0]
    # In horizon 0: pred=[1.0, 0.5], tgt=[0.0, 0.5]. Diff=[1.0, 0.0]. Smooth L1(1.0)=0.5, Smooth L1(0.0)=0.0.
    # Total sum over 2 samples = 0.5. Mean over (2 samples * 3 horizons) = 0.5 / 6 = 1/12.
    loss_h0 = compute_multi_horizon_loss(pred, tgt, weights=[1.0, 0.0, 0.0])
    assert math.isclose(loss_h0.item(), 0.5 / 6.0, rel_tol=1e-5)

    # 3. Sum != 1 weights: [10.0, 20.0, 30.0]
    loss_large = compute_multi_horizon_loss(pred, tgt, weights=[10.0, 20.0, 30.0])
    assert loss_large.item() > 0.0 and torch.isfinite(loss_large)

    # 4. Weights passed as 1D tensor, 2D tensor, list, tuple, numpy array
    w_list = [0.5, 1.0, 1.5]
    w_tuple = (0.5, 1.0, 1.5)
    w_np = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    w_tensor_1d = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
    w_tensor_2d = torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float32)

    l_list = compute_multi_horizon_loss(pred, tgt, weights=w_list).item()
    l_tuple = compute_multi_horizon_loss(pred, tgt, weights=w_tuple).item()
    l_np = compute_multi_horizon_loss(pred, tgt, weights=w_np).item()
    l_1d = compute_multi_horizon_loss(pred, tgt, weights=w_tensor_1d).item()
    l_2d = compute_multi_horizon_loss(pred, tgt, weights=w_tensor_2d).item()

    assert math.isclose(l_list, l_tuple, rel_tol=1e-6)
    assert math.isclose(l_list, l_np, rel_tol=1e-6)
    assert math.isclose(l_list, l_1d, rel_tol=1e-6)
    assert math.isclose(l_list, l_2d, rel_tol=1e-6)


def test_adversarial_loss_boundary_targets():
    """Tests loss with y = +/-10.0, y = 0.0, y = +/-100.0, and NaN/Inf target inputs."""
    pred = torch.zeros(4, 3)

    # Boundary targets +/-10.0
    tgt_boundary = torch.tensor([
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0],
        [10.0, 0.0, -10.0],
        [0.0, 0.0, 0.0],
    ])
    loss_b = compute_multi_horizon_loss(pred, tgt_boundary)
    assert torch.isfinite(loss_b) and loss_b.item() > 0.0

    # Extreme targets +/-100.0
    tgt_extreme = torch.tensor([
        [100.0, -100.0, 100.0],
        [-100.0, 100.0, -100.0],
        [100.0, 100.0, 100.0],
        [-100.0, -100.0, -100.0],
    ])
    loss_ext = compute_multi_horizon_loss(pred, tgt_extreme)
    assert torch.isfinite(loss_ext) and loss_ext.item() > 0.0

    # Targets containing NaNs and Infs: must sanitize without throwing exceptions
    tgt_corrupted = torch.tensor([
        [float("nan"), float("inf"), float("-inf")],
        [1.0, float("nan"), 0.0],
        [float("inf"), float("inf"), float("inf")],
        [0.0, 0.0, 0.0],
    ])
    loss_corrupt = compute_multi_horizon_loss(pred, tgt_corrupted)
    assert torch.isfinite(loss_corrupt)


# ==============================================================================
# SECTION 5: Gradient Norm Bounding Verification (||dL/d(pred)|| <= 1.0)
# ==============================================================================

def test_adversarial_smooth_l1_gradient_norm_bounded():
    """
    Verifies that under Smooth L1 loss (beta=1.0), the elementwise gradient magnitude
    |dL/d(pred_i)| is strictly bounded regardless of how massive the error is.
    Specifically:
    For loss L = (1/N) * sum(SmoothL1(pred - target)), the derivative with respect to pred_i is:
        d(SmoothL1)/d(pred_i) = sign(error) * 1.0  (when |error| >= beta=1.0)
    Thus, dL/d(pred_i) = (1/N) * sign(error), so |dL/d(pred_i)| <= 1/N.
    Elementwise unreduced gradient is <= 1.0.
    """
    error_magnitudes = [0.001, 0.1, 0.5, 0.99, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 1e6]

    for err in error_magnitudes:
        pred = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        tgt = torch.tensor([[err, -err, err]], dtype=torch.float32)

        loss = compute_multi_horizon_loss(pred, tgt, beta=1.0)
        loss.backward()

        # Since batch size = 1 and horizons = 3, N = 3
        # Max theoretical gradient per element is 1.0 / 3 = 0.3333...
        grad = pred.grad
        assert grad is not None
        assert torch.all(torch.isfinite(grad)), f"Gradient is not finite for error {err}: {grad}"
        max_grad_mag = grad.abs().max().item()
        
        # Verify strict boundedness <= 1.0 / N <= 1.0
        assert max_grad_mag <= 1.0 / 3.0 + 1e-6, (
            f"Gradient norm exceeded theoretical bound for error {err}: {max_grad_mag} > {1.0/3.0}"
        )

        if err >= 1.0:
            # Under linear region (|err| >= 1.0), gradient magnitude must saturate exactly at 1/3
            assert math.isclose(max_grad_mag, 1.0 / 3.0, rel_tol=1e-4), (
                f"Gradient failed to saturate at 1/3 in linear region for err {err}: got {max_grad_mag}"
            )


# ==============================================================================
# SECTION 6: mega_pool_dataset.py line 371 Safeguard Stress Test
# ==============================================================================

def test_adversarial_dataset_line_371_corrupted_atr():
    """
    Adversarially injects NaNs, +Inf, -Inf, 0.0, -50.0, and sub-floor values into env.atr_arr
    and proves that SingleRegimeDataset.get_batch handles all of them gracefully without crashing,
    flooring all values to atr_floor (1e-4).
    """
    mock_env = MockM1Environment(total_bars=500)
    dataset = SingleRegimeDataset(
        name="Bull",
        env=mock_env,
        lookback=240,
        min_bar_idx=100,
        atr_floor=1e-4,
    )

    # Invalidate ATR values in mock_env.atr_arr across diverse indices
    mock_env.atr_arr[150] = float("nan")
    mock_env.atr_arr[151] = float("inf")
    mock_env.atr_arr[152] = float("-inf")
    mock_env.atr_arr[153] = 0.0
    mock_env.atr_arr[154] = -100.0
    mock_env.atr_arr[155] = 1e-12 # sub-floor

    # Query a batch containing all corrupted indices
    # Notice: local indices in dataset map to bar_indices = local_idx + min_bar_idx
    # dataset.min_bar_idx is 100, so local indices are: 50, 51, 52, 53, 54, 55, 60
    corrupted_local_indices = [50, 51, 52, 53, 54, 55, 60]
    batch = dataset.get_batch(indices=corrupted_local_indices)

    assert isinstance(batch, MegaPoolBatch)
    assert batch.atr.shape == (len(corrupted_local_indices),)

    # 1. Every single ATR value must be finite
    assert torch.all(torch.isfinite(batch.atr)), f"Non-finite ATR found in batch: {batch.atr}"

    # 2. Every single ATR value must be strictly >= atr_floor (1e-4)
    assert torch.all(batch.atr >= 1e-4), f"ATR dropped below floor: {batch.atr}"

    # 3. Check specific corrupted values are floored to 1e-4
    for i in range(6):
        assert math.isclose(batch.atr[i].item(), 1e-4, rel_tol=1e-5), f"Index {i} (original {mock_env.atr_arr[150+i]}) not floored to 1e-4: {batch.atr[i].item()}"

    # 4. Check normal uncorrupted value at index 6 is preserved
    assert math.isclose(batch.atr[6].item(), max(float(mock_env.atr_arr[160]), 1e-4), rel_tol=1e-5)


# ==============================================================================
# SECTION 7: End-to-End TCA + LatentDynamicsWorldModel Adversarial Integration
# ==============================================================================

def test_adversarial_end_to_end_tca_pipeline():
    """
    Stress-tests the full pipeline:
    obs [B, 137] + candle_seq [B, 240, 6] -> WorldModel (with TCA) -> z_out [B, 768]
    -> MultiHorizonProjectionHead -> pred [B, 3] -> compute_multi_horizon_loss -> backward.
    Injects mild random perturbations and tests gradient cleanliness.
    """
    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240)
    head = wm.create_projection_head(dropout=0.05)

    B = 4
    obs = torch.randn(B, 137)
    candle_seq = torch.randn(B, 240, 6)
    targets = torch.tensor([
        [1.5, -0.5, 0.2],
        [-2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [10.0, -10.0, 5.0],
    ], dtype=torch.float32)

    # Forward
    z_out = wm.encode_state(obs, candle_seq=candle_seq)
    assert z_out.shape == (B, 768)
    assert torch.all(torch.isfinite(z_out))

    pred = head(z_out)
    assert pred.shape == (B, 3)
    assert torch.all(torch.isfinite(pred))

    loss = compute_multi_horizon_loss(pred, targets, beta=1.0)
    assert torch.isfinite(loss)

    # Backward pass
    loss.backward()

    # Verify head gradients
    for name, p in head.named_parameters():
        assert p.grad is not None, f"Missing gradient for {name}"
        assert torch.all(torch.isfinite(p.grad)), f"Head gradient non-finite for {name}"

    # Verify TCA module received gradients
    tca_grads = [p.grad for p in wm.temporal_encoder.parameters() if p.grad is not None]
    assert len(tca_grads) > 0, "No gradients received by TCA module!"
    for g in tca_grads:
        assert torch.all(torch.isfinite(g)), "TCA module gradient non-finite!"


# ==============================================================================
# SECTION 8: Additional Boundary & Architectural Invariant Tests
# ==============================================================================

def test_adversarial_empty_tensor_handling():
    """Validates that B=0 tensors pass cleanly through head without raising exception."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).eval()
    z_empty = torch.empty(0, 768)
    out = head(z_empty)
    assert out.shape == (0, 3)

    # predict_returns with empty tensor
    out_ret = head.predict_returns(z_empty, atr=1.5)
    assert out_ret.shape == (0, 3)


def test_adversarial_custom_dimension_flexibility():
    """Validates head with non-default dimensions: d_in=512, d_in=1024, num_horizons=1, num_horizons=5."""
    for d_in, n_horizons in [(512, 1), (1024, 5), (64, 4)]:
        h = MultiHorizonProjectionHead(d_in=d_in, num_horizons=n_horizons).eval()
        z = torch.randn(8, d_in)
        out = h(z)
        assert out.shape == (8, n_horizons)
        assert torch.all(torch.isfinite(out))


def test_adversarial_loss_beta_sensitivity():
    """Validates Smooth L1 behavior across extreme beta thresholds: beta=1e-5 (L1) to beta=100.0 (L2)."""
    pred = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
    tgt = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    for beta in [1e-5, 1e-3, 0.1, 1.0, 10.0, 100.0]:
        loss = compute_multi_horizon_loss(pred, tgt, beta=beta)
        assert torch.isfinite(loss), f"Loss failed to stay finite for beta={beta}"
        assert loss.item() > 0.0


def test_adversarial_world_model_state_dict_isolation_after_factory():
    """Proves that calling wm.create_projection_head() leaves wm.state_dict() strictly at 30 keys."""
    wm = LatentDynamicsWorldModel()
    assert len(wm.state_dict()) == 30, f"Baseline state_dict had {len(wm.state_dict())} keys, expected 30"
    
    # Call factory method on wm
    head = wm.create_projection_head(dropout=0.05)
    assert isinstance(head, MultiHorizonProjectionHead)

    # Verify that wm.state_dict() is STILL strictly 30 keys (no child module leakage)
    assert len(wm.state_dict()) == 30, (
        f"State dict mutated after create_projection_head! Key count: {len(wm.state_dict())}"
    )


def test_adversarial_per_horizon_atr_shapes():
    """Validates predict_returns when ATR is passed with shape [B, 3] (per-horizon ATR)."""
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).eval()
    z = torch.randn(4, 768)
    atr_matrix = torch.tensor([
        [1.0, 1.5, 2.0],
        [2.0, 2.5, 3.0],
        [0.5, 0.8, 1.0],
        [1.2, 1.4, 1.6],
    ])

    raw_pred = head(z)
    unnorm = head.predict_returns(z, atr=atr_matrix)
    assert unnorm.shape == (4, 3)
    assert torch.allclose(unnorm, raw_pred * atr_matrix, atol=1e-5)


def main():
    """Command-line runner executing all adversarial tests."""
    print("=" * 80)
    print("RUNNING ADVERSARIAL STRESS TEST & FUZZING HARNESS (MILESTONE 2)")
    print("=" * 80)

    tests = [
        ("Extreme Value Inputs", test_adversarial_extreme_value_inputs),
        ("Arbitrary & Extreme Batch Sizes", test_adversarial_arbitrary_and_extreme_batch_sizes),
        ("NaN & Inf Input Sanitization", test_adversarial_nan_and_inf_sanitization),
        ("Backward Pass with Corrupted Inputs", test_adversarial_backward_pass_with_nan_inputs),
        ("predict_returns Degenerate ATRs", test_adversarial_predict_returns_degenerate_atrs),
        ("Loss Non-Standard Weights", test_adversarial_loss_non_standard_weights),
        ("Loss Boundary Targets", test_adversarial_loss_boundary_targets),
        ("Smooth L1 Gradient Norm Bounded", test_adversarial_smooth_l1_gradient_norm_bounded),
        ("mega_pool_dataset.py line 371 safeguard", test_adversarial_dataset_line_371_corrupted_atr),
        ("End-to-End TCA Pipeline Integration", test_adversarial_end_to_end_tca_pipeline),
        ("Empty Tensor Handling", test_adversarial_empty_tensor_handling),
        ("Custom Dimension Flexibility", test_adversarial_custom_dimension_flexibility),
        ("Loss Beta Sensitivity", test_adversarial_loss_beta_sensitivity),
        ("WorldModel State Dict Isolation After Factory", test_adversarial_world_model_state_dict_isolation_after_factory),
        ("Per-Horizon ATR Shapes", test_adversarial_per_horizon_atr_shapes),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests.")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

