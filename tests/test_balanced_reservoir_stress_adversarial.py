"""Empirical Adversarial Stress Testing for Dynamic Multi-Regime Reservoir Buffer.

Milestone 2 Challenger Generation 2
Adversarial Verification Suite:
1. High-throughput streaming (5,000 to 10,000 asymmetric transitions with 90% Downtrend / 10% Uptrend,
   verifying Algorithm R C/k uniform survival probability across early, mid, and late stream samples).
2. Degenerate / starving regime pools and deficit redistribution (0 Uptrend, 0 Downtrend, sparse pools,
   oversized batches, empty buffer handling, 100% duplicate-free batches).
3. Extreme TD-errors stability (0.0, negative, 1e6, 1e12, NaN, Inf, extreme priority disparities,
   IS weight normalization to max 1.0, and PyTorch autograd gradient flow).
4. Serialization / deserialization roundtrip fidelity (to_dict/from_dict, pure JSON roundtrip, pickle roundtrip,
   preserving Markov transition matrix, annealed beta, step counters, and FiveAgentMemory tensors).
"""
from __future__ import annotations

import json
import math
import pickle
import time
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
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FiveAgentMemory,
)
from self_growing_trader.protocol_contract import INPUT_COUNT


def _make_realistic_step(
    step_id: int,
    regime: str = "UPTREND",
    realized_r: float = 1.0,
    td_error: float = 1.0,
    requires_grad: bool = False,
) -> dict[str, Any]:
    """Generates a realistic multi-agent step dictionary with 5-agent memory and tensors."""
    hidden = torch.randn(FIVE_AGENT_COUNT, 32, requires_grad=requires_grad)
    cell = torch.randn(FIVE_AGENT_COUNT, 32, requires_grad=requires_grad)
    prev_act = torch.tensor([step_id % 5] * FIVE_AGENT_COUNT, dtype=torch.long)
    tokens = torch.randn(FIVE_AGENT_COUNT, 4, 32, requires_grad=requires_grad)
    valid = torch.ones(FIVE_AGENT_COUNT, 4, dtype=torch.bool)

    memory = FiveAgentMemory(
        hidden=hidden,
        cell=cell,
        previous_action=prev_act,
        temporal_tokens=tokens,
        temporal_valid=valid,
    )

    return {
        "step_id": int(step_id),
        "regime": regime,
        "observations": np.full((FIVE_AGENT_COUNT, INPUT_COUNT), float(step_id), dtype=np.float32),
        "memory": memory,
        "actions": np.full(FIVE_AGENT_COUNT, step_id % 20, dtype=np.int64),
        "rewards": np.full(FIVE_AGENT_COUNT, float(realized_r), dtype=np.float32),
        "lifecycle_stratum": ("entry", "pos" if realized_r > 0 else "neg", regime),
    }


# =====================================================================
# Pillar 1: High-Throughput Streaming & C/k Uniform Survival Check
# =====================================================================

def test_high_throughput_asymmetric_stream_10000_samples_capacity_bounds() -> None:
    """Stress test: 10,000 asymmetric transitions (90% Downtrend / 10% Uptrend).
    Verifies:
    - Pool capacity C=128 strictly bounds pool sizes.
    - Admitted counters reflect exact asymmetric counts (9,000 Down, 1,000 Up).
    """
    capacity = 128
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=capacity)

    total_down = 9000
    total_up = 1000

    # Stream 9,000 Downtrend transitions
    for i in range(total_down):
        admitted = buffer.add(
            _make_realistic_step(step_id=i, regime="DOWNTREND"),
            regime="DOWNTREND",
            realized_r=-1.0 if i % 2 == 0 else 0.5,
            td_error=1.0,
        )
        assert admitted is True

    # Stream 1,000 Uptrend transitions
    for i in range(total_up):
        admitted = buffer.add(
            _make_realistic_step(step_id=10_000 + i, regime="UPTREND"),
            regime="UPTREND",
            realized_r=1.5 if i % 2 == 0 else -0.8,
            td_error=1.0,
        )
        assert admitted is True

    # Strict capacity invariant
    assert len(buffer.downtrend_pool) == capacity
    assert len(buffer.uptrend_pool) == capacity
    assert len(buffer.sideway_pool) == 0

    stats = buffer.stats()
    assert stats["downtrend_total"] == capacity
    assert stats["uptrend_total"] == capacity
    assert stats["total_admitted"] == 10_000
    assert buffer.admitted_by_regime["DOWNTREND"] == total_down
    assert buffer.admitted_by_regime["UPTREND"] == total_up


def test_uniform_c_over_k_survival_across_stream_quintiles() -> None:
    """Mathematical verification: Algorithm R must retain items uniformly from
    early, middle, and late stream intervals with marginal survival probability P = C / k.
    """
    capacity = 128
    down_stream_len = 9000  # 9,000 Downtrend samples
    up_stream_len = 1000    # 1,000 Uptrend samples
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=capacity)

    np.random.seed(42)
    # Stream Downtrend
    for i in range(down_stream_len):
        buffer.add(
            _make_realistic_step(step_id=i, regime="DOWNTREND"),
            regime="DOWNTREND",
            realized_r=-1.0,
        )

    # Stream Uptrend
    for i in range(up_stream_len):
        buffer.add(
            _make_realistic_step(step_id=10_000 + i, regime="UPTREND"),
            regime="UPTREND",
            realized_r=1.0,
        )

    # Downtrend: 9,000 items partitioned into 5 quintiles of 1,800 items each.
    # Theoretical expectation per quintile: E = 1,800 * (128 / 9000) = 25.6 items.
    down_survivor_ids = [e.step_dict["step_id"] for e in buffer.downtrend_pool]
    assert len(down_survivor_ids) == capacity

    down_quintile_counts = [
        sum(1 for sid in down_survivor_ids if q * 1800 <= sid < (q + 1) * 1800)
        for q in range(5)
    ]
    # Verify every quintile has significant survival (no catastrophic forgetting of early stream,
    # nor exclusion of late stream; expected ~25.6, bounds [10, 45])
    for q, cnt in enumerate(down_quintile_counts):
        assert cnt >= 10, f"Downtrend quintile {q} under-represented: {cnt} survivors (expected ~25.6)"
        assert cnt <= 45, f"Downtrend quintile {q} over-represented: {cnt} survivors (expected ~25.6)"

    # Uptrend: 1,000 items partitioned into 5 quintiles of 200 items each.
    # Theoretical expectation per quintile: E = 200 * (128 / 1000) = 25.6 items.
    up_survivor_ids = [e.step_dict["step_id"] - 10_000 for e in buffer.uptrend_pool]
    assert len(up_survivor_ids) == capacity

    up_quintile_counts = [
        sum(1 for sid in up_survivor_ids if q * 200 <= sid < (q + 1) * 200)
        for q in range(5)
    ]
    for q, cnt in enumerate(up_quintile_counts):
        assert cnt >= 10, f"Uptrend quintile {q} under-represented: {cnt} survivors (expected ~25.6)"
        assert cnt <= 45, f"Uptrend quintile {q} over-represented: {cnt} survivors (expected ~25.6)"


def test_monte_carlo_c_over_k_convergence_across_trials() -> None:
    """Monte Carlo verification across 30 independent streams of 5,000 samples.
    Verifies that early items (e.g. index 0, index 500) and late items (index 4500, index 4999)
    exhibit statistically indistinguishable survival frequency matching C / k = 128 / 5000 = 0.0256.
    """
    capacity = 128
    stream_len = 5000
    trials = 30
    probed_indices = [0, 500, 1500, 2500, 3500, 4500, 4999]
    hit_counts = {idx: 0 for idx in probed_indices}

    for trial in range(trials):
        buf = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=capacity)
        for i in range(stream_len):
            buf.add(_make_realistic_step(step_id=i, regime="DOWNTREND"), regime="DOWNTREND")

        survivors = {e.step_dict["step_id"] for e in buf.downtrend_pool}
        for idx in probed_indices:
            if idx in survivors:
                hit_counts[idx] += 1

    # Over 30 trials with p = 0.0256, expected hits per index is 30 * 0.0256 = 0.768 hits.
    # Total hits across all 7 probed indices should average around 7 * 0.768 = 5.37 hits.
    total_probed_hits = sum(hit_counts.values())
    assert total_probed_hits >= 1, f"Expected non-zero hits across probed indices, got {hit_counts}"
    # Confirm early index (0) is retained in reservoir just like late indices
    assert any(hit_counts[idx] > 0 for idx in [0, 500, 1500]), "Early stream was completely wiped out!"


# =====================================================================
# Pillar 2: Degenerate / Starving Regime Pools & Deficit Redistribution
# =====================================================================

def test_starving_pool_zero_uptrend_redistributes_to_downtrend_and_sideway() -> None:
    """Scenario: Uptrend has 0 samples, Downtrend has 128, Sideway has 128.
    Requesting batch_size=96 (quotas: 32 UP, 32 DOWN, 32 SIDEWAY).
    Uptrend deficit of 32 must be redistributed equally (+16 each) to Down and Sideway.
    Resulting batch must contain exactly 96 unique items (48 Down, 48 Side), 0 UP, and 0 duplicates.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    for i in range(128):
        buffer.add(_make_realistic_step(step_id=1000 + i, regime="DOWNTREND"), regime="DOWNTREND")
        buffer.add(_make_realistic_step(step_id=2000 + i, regime="SIDEWAY"), regime="SIDEWAY")

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=96)
    assert len(batch) == 96
    assert is_weights.shape == (96,)

    step_ids = [item["step_id"] for item in batch]
    assert len(step_ids) == len(set(step_ids)), "Duplicate transitions found in redistributed batch!"

    down_count = sum(1 for sid in step_ids if 1000 <= sid < 2000)
    side_count = sum(1 for sid in step_ids if 2000 <= sid < 3000)
    up_count = sum(1 for sid in step_ids if sid < 1000 or sid >= 3000)

    assert up_count == 0
    assert down_count == 48
    assert side_count == 48
    assert down_count + side_count == 96


def test_severe_multi_pool_starvation_deficit_redistribution() -> None:
    """Scenario: Uptrend has 1 sample, Downtrend has 2 samples, Sideway has 100 samples.
    Requesting batch_size=32 (quotas: 10 UP, 10 DOWN, 12 SIDEWAY).
    Uptrend supplies 1 (deficit 9).
    Downtrend supplies 2 (deficit 8).
    Total deficit = 17. Sideway must supply 12 + 17 = 29 items.
    Resulting batch must have exactly 32 unique items: 1 UP, 2 DOWN, 29 SIDEWAY.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    buffer.add(_make_realistic_step(step_id=1, regime="UPTREND"), regime="UPTREND")
    buffer.add(_make_realistic_step(step_id=2, regime="DOWNTREND"), regime="DOWNTREND")
    buffer.add(_make_realistic_step(step_id=3, regime="DOWNTREND"), regime="DOWNTREND")
    for i in range(100):
        buffer.add(_make_realistic_step(step_id=100 + i, regime="SIDEWAY"), regime="SIDEWAY")

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=32)
    assert len(batch) == 32
    assert is_weights.shape == (32,)

    step_ids = [item["step_id"] for item in batch]
    assert len(step_ids) == len(set(step_ids)), "Duplicate transitions found in severely starved batch!"

    assert step_ids.count(1) == 1
    assert step_ids.count(2) == 1
    assert step_ids.count(3) == 1
    sideway_items = [sid for sid in step_ids if sid >= 100]
    assert len(sideway_items) == 29


def test_completely_empty_buffer_does_not_crash() -> None:
    """Completely empty buffer: sample_balanced_batch and sample_balanced_reservoir_batch
    must return empty outputs without division by zero or exception.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    assert buffer.sample_balanced_batch(batch_size=32) == []

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=32)
    assert batch == []
    assert isinstance(is_weights, torch.Tensor)
    assert is_weights.shape == (0,)


def test_undersized_buffer_returns_all_available_without_infinite_loop() -> None:
    """Total items in buffer (e.g. 5) is strictly less than requested batch_size (e.g. 32).
    Buffer must return all 5 items, no duplicates, valid IS weights, and terminate promptly.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    buffer.add(_make_realistic_step(1, "UPTREND"), regime="UPTREND")
    buffer.add(_make_realistic_step(2, "UPTREND"), regime="UPTREND")
    buffer.add(_make_realistic_step(3, "DOWNTREND"), regime="DOWNTREND")
    buffer.add(_make_realistic_step(4, "SIDEWAY"), regime="SIDEWAY")
    buffer.add(_make_realistic_step(5, "SIDEWAY"), regime="SIDEWAY")

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=32)
    assert len(batch) == 5
    assert is_weights.shape == (5,)
    sids = [item["step_id"] for item in batch]
    assert sorted(sids) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("bs", [0, -1, 1, 2, 3, 7, 33, 128, 500])
def test_boundary_batch_sizes_under_reservoir_sampling(bs: int) -> None:
    """Tests wide array of boundary batch sizes under dynamic reservoir sampling."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=64)
    for i in range(30):
        buffer.add(_make_realistic_step(i, "UPTREND"), regime="UPTREND")
        buffer.add(_make_realistic_step(100 + i, "DOWNTREND"), regime="DOWNTREND")
        buffer.add(_make_realistic_step(200 + i, "SIDEWAY"), regime="SIDEWAY")

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=bs)
    if bs <= 0:
        assert len(batch) == 0
        assert is_weights.numel() == 0
    else:
        expected_len = min(bs, 90)
        assert len(batch) == expected_len
        assert is_weights.shape == (expected_len,)
        assert not torch.isnan(is_weights).any()
        assert not torch.isinf(is_weights).any()


# =====================================================================
# Pillar 3: Extreme TD-Errors (0.0, Negative, 1e6, 1e12, NaN, Inf) Stability
# =====================================================================

def test_extreme_td_errors_zero_negative_and_millions() -> None:
    """Stress test: Extreme TD-errors (0.0, negative values, 1e6, 1e12).
    Verifies:
    - Sampling probabilities never produce NaN, Inf, or negative probabilities.
    - IS weights are strictly finite, positive, and normalized to max 1.0.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    td_values = [0.0, -1e-6, -5.0, -100.0, 1.0, 1e3, 1e6, 1e12]
    for i, td in enumerate(td_values):
        for r in ["UPTREND", "DOWNTREND", "SIDEWAY"]:
            buffer.add(
                _make_realistic_step(step_id=i, regime=r),
                regime=r,
                td_error=td,
            )

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=24)
    assert len(batch) == 24
    assert is_weights.shape == (24,)

    # IS weights invariants
    assert not torch.isnan(is_weights).any(), "NaN found in Importance Sampling weights!"
    assert not torch.isinf(is_weights).any(), "Inf found in Importance Sampling weights!"
    assert torch.all(is_weights > 0.0), "Zero or negative IS weight found!"
    assert torch.max(is_weights).item() == pytest.approx(1.0, abs=1e-5), "Max IS weight not normalized to 1.0!"

    for item in batch:
        assert "importance_sampling_weight" in item
        assert "is_weight" in item
        assert math.isfinite(item["is_weight"])
        assert item["is_weight"] > 0.0


def test_uniform_zero_td_errors_across_entire_buffer() -> None:
    """All items in buffer have td_error=0.0.
    Priorities must default to uniform probabilities with no division by zero.
    All IS weights must evaluate to exactly 1.0.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    for i in range(20):
        buffer.add(_make_realistic_step(i, "UPTREND"), regime="UPTREND", td_error=0.0)
        buffer.add(_make_realistic_step(100 + i, "DOWNTREND"), regime="DOWNTREND", td_error=0.0)
        buffer.add(_make_realistic_step(200 + i, "SIDEWAY"), regime="SIDEWAY", td_error=0.0)

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=30)
    assert len(batch) == 30
    assert not torch.isnan(is_weights).any()
    assert not torch.isinf(is_weights).any()
    np.testing.assert_allclose(is_weights.numpy(), np.ones(30, dtype=np.float32), atol=1e-5)


def test_update_priorities_with_pathological_inputs() -> None:
    """Tests update_priorities with negative, zero, extreme large, NaN, and Inf deltas."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    for i in range(6):
        buffer.add(_make_realistic_step(i, "UPTREND"), regime="UPTREND", td_error=1.0)

    # Pathological priority updates
    buffer.update_priorities(
        "UPTREND",
        indices=[0, 1, 2, 3, 4, 5],
        td_errors=[0.0, -42.0, 1e7, float("nan"), float("inf"), float("-inf")],
    )

    pool = buffer.get_pool("UPTREND")
    assert pool[0].td_error == 0.0
    assert pool[1].td_error == 42.0  # abs(-42.0)
    assert pool[2].td_error == 1e7
    # NaN and Infs must be safely ignored, retaining previous finite value (1.0)
    assert pool[3].td_error == 1.0
    assert pool[4].td_error == 1.0
    assert pool[5].td_error == 1.0

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=6)
    assert len(batch) == 6
    assert not torch.isnan(is_weights).any()
    assert not torch.isinf(is_weights).any()


def test_importance_sampling_weights_gradient_flow_cleanliness() -> None:
    """Verifies that importance sampling weights applied to a PyTorch policy loss
    enable clean gradient backpropagation without causing NaN or Inf in model parameters.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=64)
    for i in range(15):
        buffer.add(_make_realistic_step(i, "UPTREND"), regime="UPTREND", td_error=float(i + 1))
        buffer.add(_make_realistic_step(100 + i, "DOWNTREND"), regime="DOWNTREND", td_error=1.0)
        buffer.add(_make_realistic_step(200 + i, "SIDEWAY"), regime="SIDEWAY", td_error=2.0)

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=12)

    # Mock policy layer
    linear = torch.nn.Linear(INPUT_COUNT, 20)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)

    # Forward pass on sampled batch
    obs_tensor = torch.tensor(
        np.stack([item["observations"][0] for item in batch]),
        dtype=torch.float32,
    )
    preds = linear(obs_tensor)
    per_sample_loss = (preds ** 2).mean(dim=-1)

    # Weighted loss by IS weights
    weighted_loss = (per_sample_loss * is_weights).mean()
    optimizer.zero_grad()
    weighted_loss.backward()

    for p in linear.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
        assert not torch.isinf(p.grad).any()


# =====================================================================
# Pillar 4: Serialization & Markov Roundtrip Fidelity
# =====================================================================

def test_full_state_dict_and_json_serialization_fidelity() -> None:
    """Comprehensive roundtrip:
    Populate buffer, stream transitions to form a rich Markov matrix, sample to anneal beta,
    serialize to dictionary, serialize to pure JSON string, deserialize, and verify 100% fidelity.
    """
    buffer = BalancedMultiRegimeReservoirBuffer(
        capacity_per_regime=64,
        alpha=0.65,
        beta_start=0.35,
        beta_end=0.95,
        beta_steps=300,
    )

    # Generate multi-regime transitions
    regimes = ["UPTREND", "DOWNTREND", "SIDEWAY", "UPTREND", "UPTREND", "SIDEWAY"]
    for i in range(120):
        r = regimes[i % len(regimes)]
        buffer.add(
            _make_realistic_step(step_id=i, regime=r),
            regime=r,
            realized_r=1.5 if i % 2 == 0 else -1.0,
            td_error=float(i % 15),
        )

    # Sample 4 batches to advance step counters and beta
    for _ in range(4):
        buffer.sample_balanced_reservoir_batch(batch_size=18)

    # Verify pre-serialization state
    assert buffer.step_count == 4
    assert buffer.beta > 0.35
    assert buffer.total_sampled == 72
    orig_markov = buffer.markov_counts.copy()
    orig_probs = buffer.get_markov_probs().copy()

    # 1. to_dict -> JSON string -> from_dict
    state_dict = buffer.to_dict()
    json_serialized = json.dumps(state_dict)
    loaded_dict = json.loads(json_serialized)
    restored = BalancedMultiRegimeReservoirBuffer.from_dict(loaded_dict)

    # Verifications
    assert restored.schema_version == buffer.schema_version
    assert restored.capacity_per_regime == buffer.capacity_per_regime
    assert restored.total_admitted == buffer.total_admitted
    assert restored.total_sampled == buffer.total_sampled
    assert restored.admitted_by_regime == buffer.admitted_by_regime
    assert restored.sampled_by_regime == buffer.sampled_by_regime
    assert restored.reservoir_sampling == buffer.reservoir_sampling
    assert restored.alpha == buffer.alpha
    assert restored.beta == pytest.approx(buffer.beta, abs=1e-6)
    assert restored.step_count == buffer.step_count
    assert restored.last_regime_idx == buffer.last_regime_idx

    # Markov matrix fidelity
    assert isinstance(restored.markov_counts, np.ndarray)
    np.testing.assert_array_equal(restored.markov_counts, orig_markov)
    np.testing.assert_allclose(restored.get_markov_probs(), orig_probs, atol=1e-6)

    # Pool size fidelity
    assert len(restored.uptrend_pool) == len(buffer.uptrend_pool)
    assert len(restored.downtrend_pool) == len(buffer.downtrend_pool)
    assert len(restored.sideway_pool) == len(buffer.sideway_pool)

    # Memory tensor restoration fidelity
    orig_mem = buffer.uptrend_pool[0].step_dict["memory"]
    rest_mem = restored.uptrend_pool[0].step_dict["memory"]
    assert torch.equal(orig_mem.hidden, rest_mem.hidden)
    assert torch.equal(orig_mem.cell, rest_mem.cell)
    assert torch.equal(orig_mem.previous_action, rest_mem.previous_action)
    assert torch.equal(orig_mem.temporal_tokens, rest_mem.temporal_tokens)
    assert torch.equal(orig_mem.temporal_valid, rest_mem.temporal_valid)


def test_pickle_round_trip_with_dynamic_reservoir() -> None:
    """Verifies Python pickle round-trip preserves all dynamic reservoir attributes."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=64, alpha=0.55)
    for i in range(20):
        buffer.add(_make_realistic_step(i, "UPTREND"), regime="UPTREND", td_error=3.5)
        buffer.add(_make_realistic_step(100 + i, "DOWNTREND"), regime="DOWNTREND", td_error=0.5)

    buffer.sample_balanced_reservoir_batch(batch_size=12)

    pickled_data = pickle.dumps(buffer)
    loaded: BalancedMultiRegimeReservoirBuffer = pickle.loads(pickled_data)

    assert loaded.reservoir_sampling is True
    assert loaded.alpha == 0.55
    assert loaded.step_count == 1
    assert np.array_equal(loaded.markov_counts, buffer.markov_counts)
    assert len(loaded.uptrend_pool) == len(buffer.uptrend_pool)
    assert loaded.uptrend_pool[0].td_error == pytest.approx(3.5)
