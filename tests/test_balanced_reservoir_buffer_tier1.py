"""Tier 1 Unit Tests for Dynamic Multi-Regime Reservoir Buffer (Milestone 2 R1).

Verifies:
1. Genuine Algorithm R Reservoir Replacement (C/k uniform survival).
2. Markov 3x3 state transition matrix estimation with Laplace smoothing.
3. TD-error and Markov transition prioritized sampling probabilities.
4. Balanced 33.3% sampling across Uptrend, Downtrend, and Sideway pools.
5. Importance Sampling (IS) weights computation, annealing beta [0.4 -> 1.0], and max-1.0 normalization.
6. Deficit redistribution across asymmetric or starving pools.
7. Post-step TD-error priority updates.
8. Serialization, deserialization, and pickle fidelity.
9. Backward compatibility for legacy callers.
"""

from __future__ import annotations

import copy
import math
import pickle
from typing import Any

import numpy as np
import pytest
import torch

from self_growing_trader.balanced_replay_buffer import (
    BalancedMultiRegimeReplayBuffer,
    BalancedMultiRegimeReservoirBuffer,
    RegimeExperience,
    normalize_regime,
)


def _make_dummy_step(
    step_id: int,
    regime: str = "UPTREND",
    realized_r: float = 0.5,
    td_error: float = 1.0,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "observations": np.full((5, 137), float(step_id), dtype=np.float32),
        "actions": np.zeros(5, dtype=np.int64),
        "rewards": np.zeros(5, dtype=np.float32),
        "regime": regime,
        "lifecycle_stratum": ("entry", "pos", regime),
    }


class DeterministicRNG:
    """Mock RNG to simulate exact integer sequences for Algorithm R."""

    def __init__(self, sequence: list[int]) -> None:
        self.sequence = list(sequence)
        self.idx = 0

    def randint(self, a: int, b: int) -> int:
        if self.idx < len(self.sequence):
            val = self.sequence[self.idx]
            self.idx += 1
            return val
        return a


def test_reservoir_uniform_retention_over_long_stream() -> None:
    """Verify that Algorithm R retains items uniformly from early, middle, and late stream."""
    capacity = 128
    stream_len = 1280  # 10x capacity
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=capacity)

    np.random.seed(42)
    for i in range(stream_len):
        step = _make_dummy_step(step_id=i, regime="UPTREND", realized_r=1.0)
        buffer.add(step, regime="UPTREND", realized_r=1.0)

    pool = buffer.get_pool("UPTREND")
    assert len(pool) == capacity

    surviving_ids = [e.step_dict["step_id"] for e in pool]
    # Stratify stream into 3 equal thirds: early (0..426), mid (427..853), late (854..1279)
    early_count = sum(1 for sid in surviving_ids if sid < 427)
    mid_count = sum(1 for sid in surviving_ids if 427 <= sid < 854)
    late_count = sum(1 for sid in surviving_ids if sid >= 854)

    # In uniform sampling of 128 out of 1280, each third should have ~42.6 items.
    # We verify each third has significant presence (> 15 items), proving no catastrophic forgetting.
    assert early_count > 15, f"Early stream forgotten: only {early_count} items survived"
    assert mid_count > 15, f"Mid stream underrepresented: {mid_count} items"
    assert late_count > 15, f"Late stream underrepresented: {late_count} items"


def test_reservoir_step_by_step_exact_replacement() -> None:
    """Verify exact Algorithm R replacement logic: replace at j if j < C, else discard."""
    capacity = 4
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=capacity)

    # Step 1: fill buffer to capacity (k=1..4)
    for i in range(4):
        buffer.add(_make_dummy_step(i), regime="UPTREND", realized_r=1.0)
    assert [e.step_dict["step_id"] for e in buffer.uptrend_pool] == [0, 1, 2, 3]

    # Step 2: 5th item (k=5). RNG returns j=1 (< 4) -> replaces position 1
    mock_rng_replace = DeterministicRNG([1])
    buffer.add(_make_dummy_step(4), regime="UPTREND", realized_r=1.0, rng=mock_rng_replace)
    assert [e.step_dict["step_id"] for e in buffer.uptrend_pool] == [0, 4, 2, 3]

    # Step 3: 6th item (k=6). RNG returns j=5 (>= 4) -> discarded, pool untouched
    mock_rng_discard = DeterministicRNG([5])
    buffer.add(_make_dummy_step(5), regime="UPTREND", realized_r=1.0, rng=mock_rng_discard)
    assert [e.step_dict["step_id"] for e in buffer.uptrend_pool] == [0, 4, 2, 3]


def test_markov_transition_matrix_estimation() -> None:
    """Verify 3x3 Markov transition matrix with Laplace smoothing."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    # Initial probabilities under Laplace smoothing should be uniform 1/3
    init_P = buffer.get_markov_probs()
    assert init_P.shape == (3, 3)
    np.testing.assert_allclose(init_P, np.ones((3, 3)) / 3.0, atol=1e-5)

    # Stream: UP -> UP -> DOWN -> SIDEWAY -> SIDEWAY -> UP
    # last_regime_idx starts at 2 (SIDEWAY)
    transitions = ["UPTREND", "UPTREND", "DOWNTREND", "SIDEWAY", "SIDEWAY", "UPTREND"]
    for reg in transitions:
        buffer.add(_make_dummy_step(0, regime=reg), regime=reg)

    P = buffer.get_markov_probs()
    # Each row must sum to 1.0
    row_sums = P.sum(axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-5)

    # Verify SIDEWAY -> UPTREND occurred twice (init last=2 -> UP, then SIDEWAY -> UP)
    # while SIDEWAY -> DOWNTREND did not occur
    p_side_up = P[2, 0]
    p_side_down = P[2, 1]
    assert p_side_up > p_side_down


def test_td_error_and_markov_prioritization_probabilities() -> None:
    r"""Verify PER probabilities p \propto (|delta| + 1e-5)^0.6 * (1 + 0.5 * P_trans)."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128, alpha=0.6)

    # Add 3 samples with stark TD-errors to UPTREND pool
    buffer.add(_make_dummy_step(0, "UPTREND"), regime="UPTREND", td_error=0.01)
    buffer.add(_make_dummy_step(1, "UPTREND"), regime="UPTREND", td_error=1.0)
    buffer.add(_make_dummy_step(2, "UPTREND"), regime="UPTREND", td_error=100.0)

    # Populate DOWNTREND and SIDEWAY so there is no deficit redistribution
    for i in range(10):
        buffer.add(_make_dummy_step(100 + i, "DOWNTREND"), regime="DOWNTREND", td_error=1.0)
        buffer.add(_make_dummy_step(200 + i, "SIDEWAY"), regime="SIDEWAY", td_error=1.0)

    # Sample batch_size=3 (1 UP, 1 DOWN, 1 SIDEWAY) 500 times and tally selections from UPTREND
    np.random.seed(123)
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(500):
        batch = buffer.sample_balanced_batch(batch_size=3)
        for item in batch:
            sid = item.get("step_id")
            if sid in counts:
                counts[sid] += 1

    # Higher TD error MUST receive strictly more selections
    assert counts[2] > counts[1] > counts[0], f"Expected counts[2] > counts[1] > counts[0], got {counts}"



def test_exact_balanced_sampling_96_and_32() -> None:
    """Verify exact 33.3% balanced regime allocation for batch sizes 96 and 32."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    # Populate each pool with 64 items
    for i in range(64):
        buffer.add(_make_dummy_step(i, "UPTREND"), regime="UPTREND")
        buffer.add(_make_dummy_step(100 + i, "DOWNTREND"), regime="DOWNTREND")
        buffer.add(_make_dummy_step(200 + i, "SIDEWAY"), regime="SIDEWAY")

    # 1. Test Batch 96: exact 32 UP, 32 DOWN, 32 SIDEWAY
    batch_96 = buffer.sample_balanced_batch(batch_size=96)
    assert len(batch_96) == 96
    up_cnt = sum(1 for item in batch_96 if normalize_regime(item["regime"]) == "UPTREND")
    down_cnt = sum(1 for item in batch_96 if normalize_regime(item["regime"]) == "DOWNTREND")
    side_cnt = sum(1 for item in batch_96 if normalize_regime(item["regime"]) == "SIDEWAY")
    assert up_cnt == 32
    assert down_cnt == 32
    assert side_cnt == 32

    # 2. Test Batch 32: 10 UP, 10 DOWN, 12 SIDEWAY (10 + 10 + 12 = 32)
    batch_32 = buffer.sample_balanced_batch(batch_size=32)
    assert len(batch_32) == 32
    up_32 = sum(1 for item in batch_32 if normalize_regime(item["regime"]) == "UPTREND")
    down_32 = sum(1 for item in batch_32 if normalize_regime(item["regime"]) == "DOWNTREND")
    side_32 = sum(1 for item in batch_32 if normalize_regime(item["regime"]) == "SIDEWAY")
    assert up_32 == 10
    assert down_32 == 10
    assert side_32 == 12


def test_importance_sampling_weights_normalization() -> None:
    """Verify IS weights: normalized to max 1.0, tensor output, and dict embedding."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    for i in range(30):
        buffer.add(_make_dummy_step(i, "UPTREND"), regime="UPTREND", td_error=float(i + 1))
        buffer.add(_make_dummy_step(100 + i, "DOWNTREND"), regime="DOWNTREND", td_error=1.0)
        buffer.add(_make_dummy_step(200 + i, "SIDEWAY"), regime="SIDEWAY", td_error=2.0)

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=30)
    assert len(batch) == 30
    assert isinstance(is_weights, torch.Tensor)
    assert is_weights.shape == (30,)
    assert is_weights.dtype == torch.float32

    # Maximum IS weight MUST be normalized to 1.0
    assert torch.max(is_weights).item() == pytest.approx(1.0, abs=1e-5)
    # All weights must be within (0, 1]
    assert torch.all(is_weights > 0.0)
    assert torch.all(is_weights <= 1.0 + 1e-6)

    # IS weights must also be embedded into step dicts
    for item in batch:
        assert "importance_sampling_weight" in item
        assert "is_weight" in item
        assert 0.0 < item["importance_sampling_weight"] <= 1.0 + 1e-6


def test_importance_sampling_beta_annealing() -> None:
    """Verify beta annealing schedule: beta_start=0.4 -> beta_end=1.0 over beta_steps=1000."""
    buffer = BalancedMultiRegimeReservoirBuffer(
        capacity_per_regime=128,
        beta_start=0.4,
        beta_end=1.0,
        beta_steps=1000,
    )
    for i in range(10):
        buffer.add(_make_dummy_step(i, "UPTREND"), regime="UPTREND")

    assert buffer.beta == 0.4
    assert buffer.step_count == 0

    # Simulate 500 steps
    buffer.step_count = 499  # next sample will make it 500
    buffer.sample_balanced_batch(batch_size=3)
    assert buffer.step_count == 500
    assert buffer.beta == pytest.approx(0.70, abs=1e-4)

    # Simulate 1000 steps
    buffer.step_count = 999
    buffer.sample_balanced_batch(batch_size=3)
    assert buffer.step_count == 1000
    assert buffer.beta == pytest.approx(1.0, abs=1e-4)

    # Beyond 1000 steps, beta must remain clamped at 1.0
    buffer.step_count = 2000
    buffer.sample_balanced_batch(batch_size=3)
    assert buffer.beta == pytest.approx(1.0, abs=1e-4)


def test_starvation_and_deficit_redistribution() -> None:
    """Verify that when some pools are empty or sparse, deficit is redistributed cleanly."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    # Only populate UPTREND pool with 50 items; DOWNTREND has 0; SIDEWAY has 0
    for i in range(50):
        buffer.add(_make_dummy_step(i, "UPTREND"), regime="UPTREND")

    # Request batch of 30: quotas are 10 UP, 10 DOWN, 10 SIDEWAY
    # Since DOWN and SIDEWAY are empty, all 30 should be served by UPTREND without error
    batch = buffer.sample_balanced_batch(batch_size=30)
    assert len(batch) == 30
    assert all(normalize_regime(item["regime"]) == "UPTREND" for item in batch)


def test_update_priorities_post_step() -> None:
    """Verify priority updates via update_priorities."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    for i in range(5):
        buffer.add(_make_dummy_step(i, "UPTREND"), regime="UPTREND", td_error=1.0)

    # Update indices 1 and 3 with new TD errors
    buffer.update_priorities("UPTREND", indices=[1, 3], td_errors=[12.5, 45.0])

    pool = buffer.get_pool("UPTREND")
    assert pool[0].td_error == pytest.approx(1.0)
    assert pool[1].td_error == pytest.approx(12.5)
    assert pool[2].td_error == pytest.approx(1.0)
    assert pool[3].td_error == pytest.approx(45.0)


def test_reservoir_serialization_fidelity() -> None:
    """Verify to_dict/from_dict and pickle roundtrip fidelity with reservoir sampling."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=64, alpha=0.55, beta_start=0.35)
    for i in range(5):
        buffer.add(_make_dummy_step(i, "UPTREND"), regime="UPTREND", td_error=float(i + 1))
        buffer.add(_make_dummy_step(i + 10, "DOWNTREND"), regime="DOWNTREND", td_error=2.0)
        buffer.add(_make_dummy_step(i + 20, "SIDEWAY"), regime="SIDEWAY", td_error=3.0)

    buffer.sample_balanced_batch(batch_size=9)

    # 1. to_dict / from_dict
    d = buffer.to_dict()
    assert d["reservoir_sampling"] is True
    assert d["alpha"] == 0.55
    assert d["step_count"] == 1
    restored = BalancedMultiRegimeReservoirBuffer.from_dict(d)
    assert restored.reservoir_sampling is True
    assert restored.alpha == 0.55
    assert restored.step_count == 1
    assert len(restored.uptrend_pool) == 5
    assert restored.uptrend_pool[0].td_error == pytest.approx(1.0)
    assert restored.uptrend_pool[4].td_error == pytest.approx(5.0)

    # 2. Pickle roundtrip
    p_bytes = pickle.dumps(buffer)
    p_loaded = pickle.loads(p_bytes)
    assert p_loaded.reservoir_sampling is True
    assert p_loaded.alpha == 0.55
    assert len(p_loaded.uptrend_pool) == 5
    assert p_loaded.uptrend_pool[2].td_error == pytest.approx(3.0)


def test_backward_compatibility_defaults() -> None:
    """Verify default flags ensure 100% backward compatibility for legacy callers."""
    legacy = BalancedMultiRegimeReplayBuffer()
    assert legacy.reservoir_sampling is False

    reservoir = BalancedMultiRegimeReservoirBuffer()
    assert reservoir.reservoir_sampling is True

    # Legacy sample returns list[dict], not tuple
    legacy.add(_make_dummy_step(1, "UPTREND"), regime="UPTREND", realized_r=1.0)
    res = legacy.sample_balanced_batch(batch_size=1)
    assert isinstance(res, list)
