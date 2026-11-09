"""Adversarial stress and edge-case testing for BalancedMultiRegimeReplayBuffer.

Milestone 1 — Challenger 1
Covers:
- Extreme asymmetric streams (10,000+ rapid inputs, 100% wins, 100% losses)
- Multi-phase regime shock and distribution shifts
- Strict invariant checks: pool size <= 128, win/loss quota <= 64
- Non-standard batch sizes (batch_size=0, 1, 2, 3, 7, 127, 384, 500, 10000)
- Severe starvation & deficit redistribution
- Rapid interleaved add/sample cycles & concurrency stress
- Autograd detachment & memory safety
- Full-capacity serialization round-trip fidelity
"""
from __future__ import annotations

import concurrent.futures
import math
import time
from typing import Any

import numpy as np
import pytest
import torch

from self_growing_trader.balanced_replay_buffer import (
    BalancedMultiRegimeReplayBuffer,
    RegimeExperience,
    normalize_regime,
)
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FiveAgentMemory,
)
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


def _make_stress_step_dict(
    step_id: int,
    regime_name: str = "UPTREND",
    *,
    requires_grad: bool = False,
) -> dict[str, Any]:
    """Creates a realistic step_dict with memory tensors, features, and stratum."""
    obs = np.random.randn(FIVE_AGENT_COUNT, INPUT_COUNT).astype(np.float32)
    t_hidden = torch.randn(FIVE_AGENT_COUNT, 32, requires_grad=requires_grad)
    t_cell = torch.randn(FIVE_AGENT_COUNT, 32, requires_grad=requires_grad)
    t_prev = torch.zeros(FIVE_AGENT_COUNT, dtype=torch.long)
    t_tokens = torch.randn(FIVE_AGENT_COUNT, 4, 32, requires_grad=requires_grad)
    t_valid = torch.ones(FIVE_AGENT_COUNT, 4, dtype=torch.bool)

    memory = FiveAgentMemory(
        hidden=t_hidden,
        cell=t_cell,
        previous_action=t_prev,
        temporal_tokens=t_tokens,
        temporal_valid=t_valid,
    )

    return {
        "step_id": int(step_id),
        "observations": obs,
        "memory": memory,
        "actions": np.ones(FIVE_AGENT_COUNT, dtype=np.int64),
        "targets": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "delay_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "risk_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "entry_offset_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "sl_indices": np.full(FIVE_AGENT_COUNT, 7, dtype=np.int64),
        "tp_indices": np.full(FIVE_AGENT_COUNT, 5, dtype=np.int64),
        "trailing_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "manage_mode_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "close_fraction_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "old_log_probabilities": np.zeros(FIVE_AGENT_COUNT, dtype=np.float32),
        "old_values": np.zeros(FIVE_AGENT_COUNT, dtype=np.float32),
        "rewards": np.zeros(FIVE_AGENT_COUNT, dtype=np.float32),
        "allowed_action_masks": np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool),
        "policy_loss_masks": np.ones(FIVE_AGENT_COUNT, dtype=bool),
        "direction_class": np.int64(0),
        "utility_target": np.zeros(2, dtype=np.float32),
        "excursion_target": np.zeros(4, dtype=np.float32),
        "no_edge_target": np.float32(0.0),
        "pattern_tree_enabled": False,
        "lifecycle_stratum": ("mua_thi_truong", "duong", regime_name),
    }


# =====================================================================
# Test Suite 1: Extreme Asymmetric Input Streams & Capacity Bounds
# =====================================================================

def test_extreme_asymmetric_stream_10000_wins() -> None:
    """Stress test: 10,000 rapid consecutive wins in UPTREND."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for i in range(10_000):
        admitted = buffer.add(
            _make_stress_step_dict(step_id=i, regime_name="UPTREND"),
            regime="UPTREND",
            realized_r=1.5,
        )
        assert admitted is True
        # Invariant: win quota is 64; pool size must NEVER exceed 64 when all inputs are wins
        assert len(buffer.uptrend_pool) <= 64

    stats = buffer.stats()
    assert stats["uptrend_wins"] == 64
    assert stats["uptrend_losses"] == 0
    assert stats["uptrend_total"] == 64
    assert stats["total_samples"] == 64
    assert stats["total_admitted"] == 10_000

    # Oldest 9,936 wins must have been evicted in strict FIFO order
    # Remaining IDs must be 9936..9999
    stored_ids = [exp.step_dict["step_id"] for exp in buffer.uptrend_pool]
    assert stored_ids[0] == 9_936
    assert stored_ids[-1] == 9_999


def test_extreme_asymmetric_stream_10000_losses() -> None:
    """Stress test: 10,000 rapid consecutive losses in DOWNTREND."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for i in range(10_000):
        admitted = buffer.add(
            _make_stress_step_dict(step_id=i, regime_name="DOWNTREND"),
            regime="DOWNTREND",
            realized_r=-1.2,
        )
        assert admitted is True
        # Invariant: loss quota is 64; pool size must NEVER exceed 64 when all inputs are losses
        assert len(buffer.downtrend_pool) <= 64

    stats = buffer.stats()
    assert stats["downtrend_losses"] == 64
    assert stats["downtrend_wins"] == 0
    assert stats["downtrend_total"] == 64
    assert stats["total_samples"] == 64
    assert stats["total_admitted"] == 10_000

    stored_ids = [exp.step_dict["step_id"] for exp in buffer.downtrend_pool]
    assert stored_ids[0] == 9_936
    assert stored_ids[-1] == 9_999


def test_extreme_multi_phase_regime_shock_20000_steps() -> None:
    """Stress test: 20,000 inputs across 5 alternating regime & win/loss phases."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Phase 1: 4,000 losses in UPTREND
    for i in range(4000):
        buffer.add(_make_stress_step_dict(i), regime="UPTREND", realized_r=-0.8)

    # Phase 2: 4,000 wins in UPTREND
    for i in range(4000, 8000):
        buffer.add(_make_stress_step_dict(i), regime="UPTREND", realized_r=2.5)

    # Phase 3: 4,000 losses in DOWNTREND
    for i in range(8000, 12000):
        buffer.add(_make_stress_step_dict(i), regime="DOWNTREND", realized_r=-1.5)

    # Phase 4: 4,000 wins in SIDEWAY
    for i in range(12000, 16000):
        buffer.add(_make_stress_step_dict(i), regime="SIDEWAY", realized_r=0.9)

    # Phase 5: 4,000 random mixed trades
    rng = np.random.default_rng(12345)
    regimes = ["UPTREND", "DOWNTREND", "SIDEWAY"]
    for i in range(16000, 20000):
        r = rng.choice(regimes)
        realized = float(rng.uniform(-2.0, 2.0))
        buffer.add(_make_stress_step_dict(i), regime=r, realized_r=realized)

    stats = buffer.stats()
    # Invariants under extreme load:
    assert stats["uptrend_total"] <= 128
    assert stats["downtrend_total"] <= 128
    assert stats["sideway_total"] <= 128
    assert stats["total_samples"] <= 384
    assert stats["uptrend_wins"] <= 64
    assert stats["uptrend_losses"] <= 64
    assert stats["downtrend_wins"] <= 64
    assert stats["downtrend_losses"] <= 64
    assert stats["sideway_wins"] <= 64
    assert stats["sideway_losses"] <= 64
    assert stats["total_admitted"] == 20_000


# =====================================================================
# Test Suite 2: Non-Standard & Boundary Batch Sizes
# =====================================================================

@pytest.mark.parametrize("invalid_batch_size", [0, -1, -50, -999])
def test_sample_balanced_batch_non_positive_batch_size(invalid_batch_size: int) -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        buffer.add(_make_stress_step_dict(1), regime=r, realized_r=1.0)
    result = buffer.sample_balanced_batch(batch_size=invalid_batch_size)
    assert result == []


def test_sample_balanced_batch_size_1() -> None:
    """Boundary test: batch_size=1 on full buffer and single-regime buffer."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(10):
            buffer.add(_make_stress_step_dict(i), regime=r, realized_r=1.0)

    # Sample batch of 1
    batch = buffer.sample_balanced_batch(batch_size=1)
    assert len(batch) == 1
    assert "step_id" in batch[0]


def test_sample_balanced_batch_size_2() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(10):
            buffer.add(_make_stress_step_dict(i), regime=r, realized_r=1.0)
    batch = buffer.sample_balanced_batch(batch_size=2)
    assert len(batch) == 2


def test_sample_balanced_batch_size_3_exact_trinity() -> None:
    """batch_size=3 should pull exactly 1 Uptrend, 1 Downtrend, 1 Sideway when available."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(5):
            buffer.add(_make_stress_step_dict(i, regime_name=r), regime=r, realized_r=1.0)

    batch = buffer.sample_balanced_batch(batch_size=3)
    assert len(batch) == 3
    regimes = {normalize_regime(item["lifecycle_stratum"][2]) for item in batch}
    assert regimes == {"UPTREND", "DOWNTREND", "SIDEWAY"}


@pytest.mark.parametrize("batch_size", [7, 13, 127, 384])
def test_sample_balanced_batch_non_standard_sizes_distribution(batch_size: int) -> None:
    """Tests non-standard batch sizes on fully saturated pools."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(64):
            buffer.add(_make_stress_step_dict(i, regime_name=r), regime=r, realized_r=1.0)
            buffer.add(_make_stress_step_dict(100 + i, regime_name=r), regime=r, realized_r=-1.0)

    batch = buffer.sample_balanced_batch(batch_size=batch_size)
    assert len(batch) == batch_size

    # Check regime distribution proportions
    regimes = [normalize_regime(item["lifecycle_stratum"][2]) for item in batch]
    up_cnt = regimes.count("UPTREND")
    down_cnt = regimes.count("DOWNTREND")
    side_cnt = regimes.count("SIDEWAY")

    expected_base = batch_size // 3
    expected_rem = batch_size % 3

    assert up_cnt == expected_base
    assert down_cnt == expected_base
    assert side_cnt == expected_base + expected_rem


def test_sample_balanced_batch_oversized_beyond_capacity() -> None:
    """Tests batch_size=10,000 when buffer only has 384 items."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(64):
            buffer.add(_make_stress_step_dict(i, regime_name=r), regime=r, realized_r=1.0)
            buffer.add(_make_stress_step_dict(100 + i, regime_name=r), regime=r, realized_r=-1.0)

    # 384 total items
    batch = buffer.sample_balanced_batch(batch_size=10_000)
    # Must not crash, loop infinitely, or duplicate items; returns all available 384
    assert len(batch) == 384


# =====================================================================
# Test Suite 3: Severe Regime Starvation & Deficit Redistribution
# =====================================================================

def test_severe_asymmetric_starvation_deficit_redistribution() -> None:
    """Uptrend has 128 items, Downtrend has 2 items, Sideway has 0 items.
    Requesting batch_size=32 requires deficit redistribution."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # 128 items in Uptrend
    for i in range(64):
        buffer.add(_make_stress_step_dict(i, regime_name="UPTREND"), regime="UPTREND", realized_r=1.0)
        buffer.add(_make_stress_step_dict(100 + i, regime_name="UPTREND"), regime="UPTREND", realized_r=-1.0)

    # 2 items in Downtrend
    buffer.add(_make_stress_step_dict(200, regime_name="DOWNTREND"), regime="DOWNTREND", realized_r=1.0)
    buffer.add(_make_stress_step_dict(201, regime_name="DOWNTREND"), regime="DOWNTREND", realized_r=-1.0)

    # Sideway is empty

    batch = buffer.sample_balanced_batch(batch_size=32)
    assert len(batch) == 32

    # Verify all returned items are unique step_ids
    step_ids = [item["step_id"] for item in batch]
    assert len(step_ids) == len(set(step_ids))

    # Downtrend provided 2, Uptrend provided 30 (10 base + 8 Down deficit + 12 Sideway deficit)
    downtrend_items = [item for item in batch if item["step_id"] in {200, 201}]
    assert len(downtrend_items) == 2


def test_intra_regime_win_loss_starvation() -> None:
    """Uptrend pool has 64 wins and 0 losses. Sampling should not crash or hang."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for i in range(64):
        buffer.add(_make_stress_step_dict(i, regime_name="UPTREND"), regime="UPTREND", realized_r=1.0)

    batch = buffer.sample_balanced_batch(batch_size=10)
    assert len(batch) == 10
    for item in batch:
        assert item["step_id"] < 64


# =====================================================================
# Test Suite 4: Rapid Interleaved Add/Sample Cycles
# =====================================================================

def test_rapid_interleaved_add_sample_1000_cycles() -> None:
    """Simulates 1,000 rapid real-time trading steps alternating add() and sample()."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    rng = np.random.default_rng(999)
    regimes = ["UPTREND", "DOWNTREND", "SIDEWAY"]

    for cycle in range(1000):
        # 1. Add 1-3 new trade transitions
        for _ in range(rng.integers(1, 4)):
            r = str(rng.choice(regimes))
            realized = float(rng.choice([-1.5, -0.5, 0.5, 1.5, 3.0]))
            buffer.add(_make_stress_step_dict(cycle, regime_name=r), regime=r, realized_r=realized)

        # 2. Sample batch of 16-64
        bs = int(rng.choice([16, 32, 48, 64]))
        batch = buffer.sample_balanced_batch(batch_size=bs)
        # Expected count is min(bs, total_samples)
        total = buffer.stats()["total_samples"]
        assert len(batch) == min(bs, total)
        # Pool size invariant check at each cycle
        assert len(buffer.uptrend_pool) <= 128
        assert len(buffer.downtrend_pool) <= 128
        assert len(buffer.sideway_pool) <= 128


# =====================================================================
# Test Suite 5: Massive Concurrency & Thread-Safety Stress
# =====================================================================

def test_concurrent_multi_threaded_add_and_sample_stress() -> None:
    """Stress test: 8 worker threads streaming 500 adds each, while 4 threads sample batches concurrently."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    errors: list[Exception] = []

    def writer_worker(thread_idx: int) -> None:
        try:
            regimes = ["UPTREND", "DOWNTREND", "SIDEWAY"]
            for step in range(500):
                step_id = thread_idx * 1000 + step
                regime = regimes[step % 3]
                realized = 1.0 if step % 2 == 0 else -1.0
                buffer.add(
                    _make_stress_step_dict(step_id, regime_name=regime),
                    regime=regime,
                    realized_r=realized,
                )
        except Exception as e:
            errors.append(e)

    def reader_worker(reader_idx: int) -> None:
        try:
            for _ in range(200):
                buffer.sample_balanced_batch(batch_size=32)
                time.sleep(0.0001)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for w in range(8):
            futures.append(executor.submit(writer_worker, w))
        for r in range(4):
            futures.append(executor.submit(reader_worker, r))
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent execution produced errors: {errors}"
    stats = buffer.stats()
    assert stats["total_admitted"] == 4000
    # Final pool sizes must remain bounded by capacity
    # Under GIL race, pool size could at most slightly fluctuate if not locked, let's verify exact behavior:
    assert stats["uptrend_total"] <= 128
    assert stats["downtrend_total"] <= 128
    assert stats["sideway_total"] <= 128


# =====================================================================
# Test Suite 6: Autograd Detachment & Memory Safety
# =====================================================================

def test_autograd_detachment_and_cpu_placement() -> None:
    """Verifies that all tensors entering the buffer have detached autograd and reside on CPU."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    step = _make_stress_step_dict(1, requires_grad=True)
    mem = step["memory"]
    assert mem.hidden.requires_grad is True

    buffer.add(step, regime="UPTREND", realized_r=1.0)
    stored = buffer.uptrend_pool[0]
    stored_mem = stored.step_dict["memory"]

    assert stored_mem.hidden.requires_grad is False
    assert stored_mem.hidden.grad_fn is None
    assert stored_mem.hidden.device.type == "cpu"
    assert stored_mem.cell.requires_grad is False
    assert stored_mem.temporal_tokens.requires_grad is False


# =====================================================================
# Test Suite 7: Full-Capacity Serialization Round-Trip Fidelity
# =====================================================================

def test_full_capacity_dict_serialization_and_deserialization() -> None:
    """Populates buffer to absolute max capacity (384 items) and verifies lossless dict round-trip."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(64):
            buffer.add(_make_stress_step_dict(i, regime_name=r), regime=r, realized_r=1.0)
            buffer.add(_make_stress_step_dict(100 + i, regime_name=r), regime=r, realized_r=-1.0)

    t0 = time.perf_counter()
    serialized = buffer.to_dict()
    t_serialize = time.perf_counter() - t0

    t1 = time.perf_counter()
    restored = BalancedMultiRegimeReplayBuffer.from_dict(serialized)
    t_deserialize = time.perf_counter() - t1

    assert t_serialize < 2.0, f"Serialization too slow: {t_serialize:.3f}s"
    assert t_deserialize < 2.0, f"Deserialization too slow: {t_deserialize:.3f}s"

    assert restored.stats() == buffer.stats()
    assert len(restored.uptrend_pool) == 128
    assert len(restored.downtrend_pool) == 128
    assert len(restored.sideway_pool) == 128

    # Verify tensor shape and numerical equality for restored memory
    orig_mem = buffer.uptrend_pool[0].step_dict["memory"]
    rest_mem = restored.uptrend_pool[0].step_dict["memory"]
    assert torch.equal(orig_mem.hidden, rest_mem.hidden)
    assert torch.equal(orig_mem.cell, rest_mem.cell)
    assert torch.equal(orig_mem.previous_action, rest_mem.previous_action)


# =====================================================================
# Test Suite 8: Malformed & Pathological Inputs
# =====================================================================

def test_malformed_and_extreme_inputs() -> None:
    """Tests buffer behavior against NaN, Inf, extreme floats, and weird regimes."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # 1. NaN and Inf realized_r should be rejected safely
    assert buffer.add(_make_stress_step_dict(1), regime="UPTREND", realized_r=float("nan")) is False
    assert buffer.add(_make_stress_step_dict(2), regime="UPTREND", realized_r=float("inf")) is False
    assert buffer.add(_make_stress_step_dict(3), regime="UPTREND", realized_r=float("-inf")) is False
    assert buffer.stats()["total_samples"] == 0

    # 2. Extreme floats
    assert buffer.add(_make_stress_step_dict(4), regime="UPTREND", realized_r=1e12) is True
    assert buffer.add(_make_stress_step_dict(5), regime="DOWNTREND", realized_r=-1e12) is True
    assert buffer.stats()["uptrend_wins"] == 1
    assert buffer.stats()["downtrend_losses"] == 1

    # 3. None / Empty regime strings fallback to SIDEWAY
    assert buffer.add(_make_stress_step_dict(6), regime=None, realized_r=0.5) is True  # type: ignore
    assert buffer.add(_make_stress_step_dict(7), regime="", realized_r=0.5) is True
    assert buffer.stats()["sideway_wins"] == 2
