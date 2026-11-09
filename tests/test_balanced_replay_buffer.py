from __future__ import annotations

import pickle
from typing import Any
import numpy as np
import pytest
import torch

from self_growing_trader.balanced_replay_buffer import (
    BalancedMultiRegimeReplayBuffer,
    RegimeExperience,
)
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FivePPOTrainer,
    SharedFiveTradingPolicy,
)
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


def _make_dummy_step_dict(
    *,
    step_id: int,
    action: int = 1,
    reward_r: float = 1.0,
    realized_r: float = 0.0,
    regime_stratum: tuple[str, str, str] = ("mua_thi_truong", "duong", "xu_huong_tang"),
    **kwargs: Any,
) -> dict[str, Any]:
    """Constructs a fully populated step_dict matching observe_five_pattern_transition."""
    obs = np.zeros((FIVE_AGENT_COUNT, INPUT_COUNT), dtype=np.float32)
    obs[:, 0] = float(step_id)
    actions = np.zeros(FIVE_AGENT_COUNT, dtype=np.int64)
    actions[0] = action
    rewards = np.zeros(FIVE_AGENT_COUNT, dtype=np.float32)
    rewards[0] = reward_r

    return {
        "step_id": step_id,
        "observations": obs,
        "initial_memories": (
            np.zeros((FIVE_AGENT_COUNT, 32), dtype=np.float32),
            np.zeros((FIVE_AGENT_COUNT, 32), dtype=np.float32),
        ),
        "actions": actions,
        "targets": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "delay_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "risk_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "entry_offset_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "sl_indices": np.full(FIVE_AGENT_COUNT, 7, dtype=np.int64),
        "tp_indices": np.full(FIVE_AGENT_COUNT, 5, dtype=np.int64),
        "trailing_indices": np.full(FIVE_AGENT_COUNT, 0, dtype=np.int64),
        "manage_mode_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "close_fraction_indices": np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        "old_log_probabilities": np.zeros(FIVE_AGENT_COUNT, dtype=np.float32),
        "behavior_uniform_mixes": np.zeros(FIVE_AGENT_COUNT, dtype=np.float32),
        "flat_account_masks": np.zeros(FIVE_AGENT_COUNT, dtype=bool),
        "old_values": np.zeros(FIVE_AGENT_COUNT, dtype=np.float32),
        "rewards": rewards,
        "allowed_action_masks": np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool),
        "policy_loss_masks": np.ones(FIVE_AGENT_COUNT, dtype=bool),
        "pattern_tree_enabled": False,
        "utility_target": np.zeros(2, dtype=np.float32),
        "excursion_target": np.zeros(4, dtype=np.float32),
        "no_edge_target": np.float32(0.0),
        "direction_class": np.int64(0),
        "lifecycle_stratum": regime_stratum,
    }


# =====================================================================
# Area 1: Data Structures & Pool Admission
# =====================================================================

def test_regime_experience_dataclass_fields() -> None:
    step = _make_dummy_step_dict(step_id=1, realized_r=1.5)
    exp = RegimeExperience(
        step_dict=step,
        regime="UPTREND",
        realized_r=1.5,
        is_win=True,
        agent_index=0,
        action_index=1,
        timestamp_ns=100_000,
    )
    assert exp.regime == "UPTREND"
    assert exp.realized_r == 1.5
    assert exp.is_win is True
    assert exp.agent_index == 0
    assert exp.action_index == 1
    assert exp.timestamp_ns == 100_000
    assert exp.step_dict["step_id"] == 1


def test_buffer_initialization_defaults() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    assert buffer.capacity_per_regime == 128
    assert len(buffer.uptrend_pool) == 0
    assert len(buffer.downtrend_pool) == 0
    assert len(buffer.sideway_pool) == 0
    stats = buffer.stats()
    assert stats["uptrend_total"] == 0
    assert stats["downtrend_total"] == 0
    assert stats["sideway_total"] == 0
    assert stats["total_samples"] == 0


def test_add_routes_correctly_to_pools() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Standard upper-case
    buffer.add(_make_dummy_step_dict(step_id=1), regime="UPTREND", realized_r=1.0)
    buffer.add(_make_dummy_step_dict(step_id=2), regime="DOWNTREND", realized_r=-1.0)
    buffer.add(_make_dummy_step_dict(step_id=3), regime="SIDEWAY", realized_r=0.5)

    assert len(buffer.uptrend_pool) == 1
    assert len(buffer.downtrend_pool) == 1
    assert len(buffer.sideway_pool) == 1

    # Case insensitivity
    buffer.add(_make_dummy_step_dict(step_id=4), regime="uptrend", realized_r=0.8)
    buffer.add(_make_dummy_step_dict(step_id=5), regime="DownTrend", realized_r=-0.5)
    buffer.add(_make_dummy_step_dict(step_id=6), regime="sideway", realized_r=-0.2)

    assert len(buffer.uptrend_pool) == 2
    assert len(buffer.downtrend_pool) == 2
    assert len(buffer.sideway_pool) == 2

    # Vietnamese aliases from five_lifecycle_reward
    buffer.add(_make_dummy_step_dict(step_id=7), regime="xu_huong_tang", realized_r=1.2)
    buffer.add(_make_dummy_step_dict(step_id=8), regime="xu_huong_giam", realized_r=-1.1)
    buffer.add(_make_dummy_step_dict(step_id=9), regime="di_ngang", realized_r=0.1)
    buffer.add(_make_dummy_step_dict(step_id=10), regime="bien_dong", realized_r=-0.1)

    assert len(buffer.uptrend_pool) == 3
    assert len(buffer.downtrend_pool) == 3
    assert len(buffer.sideway_pool) == 4  # di_ngang + bien_dong both map to sideway


def test_invalid_regime_fallback() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    # Unknown regime should safely default to sideway / consolidation pool
    buffer.add(_make_dummy_step_dict(step_id=99), regime="UNKNOWN_REGIME", realized_r=0.5)
    assert len(buffer.sideway_pool) == 1
    assert len(buffer.uptrend_pool) == 0
    assert len(buffer.downtrend_pool) == 0


# =====================================================================
# Area 2: Capacity Capping & FIFO 50/50 Win/Loss Balance
# =====================================================================

def test_fifo_eviction_of_oldest_wins_when_win_quota_exceeded() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    win_quota = 64  # capacity // 2

    # Add 70 consecutive wins to uptrend pool
    for i in range(70):
        buffer.add(
            _make_dummy_step_dict(step_id=i),
            regime="UPTREND",
            realized_r=1.0 + (i * 0.01),
        )

    # Pool should cap wins at win_quota (64)
    assert len(buffer.uptrend_pool) == 64
    stored_ids = [exp.step_dict["step_id"] for exp in buffer.uptrend_pool]
    # Oldest 6 wins (0..5) must have been evicted in FIFO order
    assert stored_ids[0] == 6
    assert stored_ids[-1] == 69


def test_fifo_eviction_of_oldest_losses_when_loss_quota_exceeded() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    loss_quota = 64  # capacity // 2

    # Add 70 consecutive losses to downtrend pool
    for i in range(70):
        buffer.add(
            _make_dummy_step_dict(step_id=i),
            regime="DOWNTREND",
            realized_r=-1.0 - (i * 0.01),
        )

    assert len(buffer.downtrend_pool) == 64
    stored_ids = [exp.step_dict["step_id"] for exp in buffer.downtrend_pool]
    # Oldest 6 losses (0..5) must have been evicted in FIFO order
    assert stored_ids[0] == 6
    assert stored_ids[-1] == 69


def test_fifty_fifty_win_loss_balance_under_asymmetric_stream() -> None:
    """Simulates real market conditions: prolonged loss streak followed by win streak."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # 1. Stream 80 losses
    for i in range(80):
        buffer.add(_make_dummy_step_dict(step_id=100 + i), regime="UPTREND", realized_r=-0.5)
    stats = buffer.stats()
    assert stats["uptrend_losses"] == 64
    assert stats["uptrend_wins"] == 0
    assert stats["uptrend_total"] == 64

    # 2. Then stream 80 wins
    for i in range(80):
        buffer.add(_make_dummy_step_dict(step_id=200 + i), regime="UPTREND", realized_r=1.2)
    stats = buffer.stats()
    assert stats["uptrend_losses"] == 64
    assert stats["uptrend_wins"] == 64
    assert stats["uptrend_total"] == 128

    # 3. Stream another 40 wins
    for i in range(40):
        buffer.add(_make_dummy_step_dict(step_id=300 + i), regime="UPTREND", realized_r=2.0)
    stats = buffer.stats()
    # Wins must remain capped at 64; losses must remain at 64; total 128
    assert stats["uptrend_wins"] == 64
    assert stats["uptrend_losses"] == 64
    assert stats["uptrend_total"] == 128

    # Ensure losses were NOT wiped out by incoming wins
    losses = [exp for exp in buffer.uptrend_pool if not exp.is_win]
    wins = [exp for exp in buffer.uptrend_pool if exp.is_win]
    assert len(losses) == 64
    assert len(wins) == 64


def test_zero_realized_r_classification() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    # Neutral trade (R = 0.0) should be treated consistently (as non-win)
    buffer.add(_make_dummy_step_dict(step_id=1), regime="SIDEWAY", realized_r=0.0)
    assert len(buffer.sideway_pool) == 1
    assert buffer.sideway_pool[0].is_win is False


# =====================================================================
# Area 3: Balanced Composite Batch Sampling & Graceful Fallback
# =====================================================================

def test_sample_balanced_batch_allocation_when_all_pools_full() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Populate each pool with 64 wins and 64 losses
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(64):
            buffer.add(_make_dummy_step_dict(step_id=i), regime=r, realized_r=1.0)
            buffer.add(_make_dummy_step_dict(step_id=100 + i), regime=r, realized_r=-1.0)

    batch = buffer.sample_balanced_batch(batch_size=32)
    assert len(batch) == 32

    # Verify ~33% representation from each regime
    uptrend_count = sum(1 for item in batch if item["lifecycle_stratum"][2] in {"UPTREND", "xu_huong_tang"})
    downtrend_count = sum(1 for item in batch if item["lifecycle_stratum"][2] in {"DOWNTREND", "xu_huong_giam"})
    sideway_count = sum(1 for item in batch if item["lifecycle_stratum"][2] in {"SIDEWAY", "di_ngang", "bien_dong"})

    # Tolerances: each regime gets 9 to 12 samples (32 // 3 = 10, remainder 2)
    assert 9 <= uptrend_count <= 12
    assert 9 <= downtrend_count <= 12
    assert 9 <= sideway_count <= 13
    assert uptrend_count + downtrend_count + sideway_count == 32


def test_sample_balanced_batch_fallback_when_one_pool_empty() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Only populate Uptrend and Downtrend (Sideway is empty)
    for i in range(50):
        buffer.add(_make_dummy_step_dict(step_id=i, regime_stratum=("m", "d", "UPTREND")), regime="UPTREND", realized_r=1.0)
        buffer.add(_make_dummy_step_dict(step_id=100 + i, regime_stratum=("b", "a", "DOWNTREND")), regime="DOWNTREND", realized_r=-1.0)

    # Request batch_size=32
    batch = buffer.sample_balanced_batch(batch_size=32)
    assert len(batch) == 32
    # Quota from empty Sideway pool should be gracefully redistributed between Uptrend and Downtrend
    uptrends = sum(1 for item in batch if item["lifecycle_stratum"][2] == "UPTREND")
    downtrends = sum(1 for item in batch if item["lifecycle_stratum"][2] == "DOWNTREND")
    assert uptrends + downtrends == 32
    assert abs(uptrends - downtrends) <= 2  # approximately equal (16 vs 16)


def test_sample_balanced_batch_fallback_when_all_pools_sparse() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Only 4 samples in total
    buffer.add(_make_dummy_step_dict(step_id=1), regime="UPTREND", realized_r=1.0)
    buffer.add(_make_dummy_step_dict(step_id=2), regime="UPTREND", realized_r=-1.0)
    buffer.add(_make_dummy_step_dict(step_id=3), regime="DOWNTREND", realized_r=1.0)
    buffer.add(_make_dummy_step_dict(step_id=4), regime="SIDEWAY", realized_r=-1.0)

    batch = buffer.sample_balanced_batch(batch_size=32)
    # Should not crash, loop infinitely, or invent invalid data; returns available samples
    assert len(batch) == 4
    ids = {item["step_id"] for item in batch}
    assert ids == {1, 2, 3, 4}


def test_sample_balanced_batch_empty_buffer_returns_empty_list() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    batch = buffer.sample_balanced_batch(batch_size=32)
    assert batch == []


def test_sample_balanced_batch_composite_with_current_batch() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Populate historical buffer
    for i in range(20):
        buffer.add(_make_dummy_step_dict(step_id=100 + i), regime="UPTREND", realized_r=1.0)
        buffer.add(_make_dummy_step_dict(step_id=200 + i), regime="DOWNTREND", realized_r=-1.0)

    # Active round emitted 8 transitions in current window (mostly Sideway)
    current_batch = [
        _make_dummy_step_dict(step_id=300 + i, regime_stratum=("m", "d", "SIDEWAY"))
        for i in range(8)
    ]

    composite = buffer.sample_balanced_batch(batch_size=32, current_batch=current_batch)
    assert len(composite) == 32
    # Ensure current batch items are integrated into composite
    composite_ids = {item["step_id"] for item in composite}
    current_ids = {item["step_id"] for item in current_batch}
    assert len(composite_ids.intersection(current_ids)) > 0


# =====================================================================
# Area 4: Serialization & Pickle Round-Trip Fidelity
# =====================================================================

def test_to_dict_and_from_dict_preserves_state() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    buffer.add(_make_dummy_step_dict(step_id=10), regime="UPTREND", realized_r=1.5)
    buffer.add(_make_dummy_step_dict(step_id=20), regime="DOWNTREND", realized_r=-0.8)
    buffer.add(_make_dummy_step_dict(step_id=30), regime="SIDEWAY", realized_r=0.2)

    data = buffer.to_dict()
    assert isinstance(data, dict)
    assert data["capacity_per_regime"] == 128
    assert len(data["uptrend_pool"]) == 1
    assert len(data["downtrend_pool"]) == 1
    assert len(data["sideway_pool"]) == 1

    restored = BalancedMultiRegimeReplayBuffer.from_dict(data)
    assert restored.capacity_per_regime == buffer.capacity_per_regime
    assert restored.stats() == buffer.stats()
    assert restored.uptrend_pool[0].realized_r == pytest.approx(1.5)
    assert restored.downtrend_pool[0].realized_r == pytest.approx(-0.8)
    assert restored.sideway_pool[0].realized_r == pytest.approx(0.2)
    np.testing.assert_array_equal(
        restored.uptrend_pool[0].step_dict["observations"],
        buffer.uptrend_pool[0].step_dict["observations"],
    )


def test_pickle_round_trip_fidelity() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=64)
    for i in range(10):
        buffer.add(_make_dummy_step_dict(step_id=i), regime="UPTREND", realized_r=1.0)
        buffer.add(_make_dummy_step_dict(step_id=i + 50), regime="DOWNTREND", realized_r=-1.0)
        buffer.add(_make_dummy_step_dict(step_id=i + 100), regime="SIDEWAY", realized_r=0.1)

    raw_bytes = pickle.dumps(buffer)
    loaded = pickle.loads(raw_bytes)

    assert isinstance(loaded, BalancedMultiRegimeReplayBuffer)
    assert loaded.capacity_per_regime == 64
    assert loaded.stats() == buffer.stats()
    assert len(loaded.uptrend_pool) == 10
    assert len(loaded.downtrend_pool) == 10
    assert len(loaded.sideway_pool) == 10


def test_empty_buffer_pickle_and_dict_fidelity() -> None:
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    raw_bytes = pickle.dumps(buffer)
    loaded = pickle.loads(raw_bytes)
    assert loaded.stats()["total_samples"] == 0

    as_dict = buffer.to_dict()
    from_dict = BalancedMultiRegimeReplayBuffer.from_dict(as_dict)
    assert from_dict.stats()["total_samples"] == 0


# =====================================================================
# Area 5: Integration with ChuyenGiaPolicy & FivePPOTrainer
# =====================================================================

def test_chuyen_gia_policy_gradient_step_on_sampled_batch() -> None:
    """Verifies that a balanced batch sampled from BalancedMultiRegimeReplayBuffer
    executes cleanly through FivePPOTrainer without shape mismatch or NaN losses."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize neural network & trainer
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.require_current_architecture()
    trainer = FivePPOTrainer(network)

    # Create and populate balanced buffer
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    for r in ("UPTREND", "DOWNTREND", "SIDEWAY"):
        for i in range(15):
            buffer.add(
                _make_dummy_step_dict(step_id=i, action=1 if i % 2 == 0 else 2, reward_r=1.0 if i % 2 == 0 else -1.0),
                regime=r,
                realized_r=1.0 if i % 2 == 0 else -1.0,
            )

    # Sample batch
    batch = buffer.sample_balanced_batch(batch_size=32)
    assert len(batch) == 32

    # Stack batch into trainer tensors
    observations = np.stack([item["observations"] for item in batch])
    actions = np.stack([item["actions"] for item in batch])
    targets = np.stack([item["targets"] for item in batch])
    delay_indices = np.stack([item["delay_indices"] for item in batch])
    risk_indices = np.stack([item["risk_indices"] for item in batch])
    entry_offset_indices = np.stack([item["entry_offset_indices"] for item in batch])
    manage_mode_indices = np.stack([item["manage_mode_indices"] for item in batch])
    close_fraction_indices = np.stack([item["close_fraction_indices"] for item in batch])
    old_log_probs = np.stack([item["old_log_probabilities"] for item in batch])
    old_values = np.stack([item["old_values"] for item in batch])
    rewards = np.stack([item["rewards"] for item in batch])
    allowed_action_masks = np.stack([item["allowed_action_masks"] for item in batch])
    policy_loss_masks = np.stack([item["policy_loss_masks"] for item in batch])

    direction_classes = np.asarray([item["direction_class"] for item in batch], dtype=np.int64)
    utility_targets = np.stack([item["utility_target"] for item in batch])
    excursion_targets = np.stack([item["excursion_target"] for item in batch])
    no_edge_targets = np.asarray([item["no_edge_target"] for item in batch], dtype=np.float32)
    pattern_tree_enabled = np.asarray([item["pattern_tree_enabled"] for item in batch], dtype=bool)

    memories = [network.initial_memories() for _ in range(32)]

    # Execute gradient step
    stats = trainer.optimize_independent_pattern_steps(
        observations=observations,
        initial_memories=memories,
        actions=actions,
        targets=targets,
        delay_indices=delay_indices,
        risk_indices=risk_indices,
        entry_offset_indices=entry_offset_indices,
        manage_mode_indices=manage_mode_indices,
        close_fraction_indices=close_fraction_indices,
        old_log_probabilities=old_log_probs,
        old_values=old_values,
        rewards=rewards,
        allowed_action_masks=allowed_action_masks,
        policy_loss_masks=policy_loss_masks,
        direction_classes=direction_classes,
        utility_targets=utility_targets,
        excursion_targets=excursion_targets,
        no_edge_targets=no_edge_targets,
        pattern_tree_enabled=pattern_tree_enabled,
    )

    # Assertions
    assert "total_loss" in stats
    assert np.isfinite(stats["total_loss"])
    assert not np.isnan(stats["total_loss"])


def test_flush_queue_preserves_balanced_replay_buffer(monkeypatch) -> None:
    """Verifies that flush_five_lifecycle_reward_queue() empties five_lifecycle_steps
    while preserving all historical experiences inside balanced_replay_buffer."""
    from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy

    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    policy.balanced_replay_buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Populate persistent buffer
    policy.balanced_replay_buffer.add(_make_dummy_step_dict(step_id=1), regime="UPTREND", realized_r=1.0)
    policy.balanced_replay_buffer.add(_make_dummy_step_dict(step_id=2), regime="DOWNTREND", realized_r=-1.0)
    assert policy.balanced_replay_buffer.stats()["total_samples"] == 2

    # Put transient items in five_lifecycle_steps
    policy.five_lifecycle_steps = [_make_dummy_step_dict(step_id=99)]
    assert len(policy.five_lifecycle_steps) == 1

    # Mock observe_five_pattern_transition to consume queue
    def fake_observe(*args, **kwargs):
        policy.five_lifecycle_steps.clear()
        return {"updated": 1.0}

    monkeypatch.setattr(policy, "observe_five_pattern_transition", fake_observe)
    policy.flush_five_lifecycle_reward_queue()

    # Transient queue is flushed
    assert len(policy.five_lifecycle_steps) == 0
    # Persistent balanced buffer remains intact!
    assert policy.balanced_replay_buffer.stats()["total_samples"] == 2
