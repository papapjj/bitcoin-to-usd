"""Adversarial Challenge Suite: Milestone 1 - Memory, Isolation & Serialization Integrity.

Adversarially validates:
1. Autograd graph detachment and CUDA/CPU tensor isolation across 1,000 add cycles.
2. Direct pickle serialization fidelity and zero type loss across all pools and containers.
3. flush_five_lifecycle_reward_queue() non-deletion and transition immutability.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import pickle
from unittest.mock import MagicMock
from typing import Any
import numpy as np
import pytest
import torch

from self_growing_trader.balanced_replay_buffer import (
    BalancedMultiRegimeReplayBuffer,
    RegimeExperience,
    _sanitize_step_dict_for_memory,
    _sanitize_value_for_memory,
)
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FiveAgentMemory,
    FivePPOTrainer,
    SharedFiveTradingPolicy,
)
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


def _make_adversarial_step_dict(
    *,
    step_id: int,
    with_autograd_tensors: bool = False,
    with_five_agent_memory: bool = False,
    regime_label: str = "xu_huong_tang",
    source_root: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Builds a rich step_dict containing diverse dtypes, nested structures, and optional autograd graphs."""
    obs = np.full((FIVE_AGENT_COUNT, INPUT_COUNT), float(step_id), dtype=np.float32)
    actions = np.arange(FIVE_AGENT_COUNT, dtype=np.int64)
    rewards = np.linspace(-1.0, 1.0, FIVE_AGENT_COUNT, dtype=np.float32)
    masks = np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool)
    masks[0, 1] = False

    step: dict[str, Any] = {
        "step_id": int(step_id),
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "allowed_action_masks": masks,
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
        "policy_loss_masks": np.ones(FIVE_AGENT_COUNT, dtype=bool),
        "direction_class": np.int64(1),
        "utility_target": np.array([0.5, -0.2], dtype=np.float32),
        "excursion_target": np.array([1.0, -0.5, 0.8, -0.1], dtype=np.float32),
        "no_edge_target": np.float32(0.25),
        "pattern_tree_enabled": False,
        "lifecycle_stratum": ("mua", "duong", regime_label),
        "metadata_nested": {
            "version": 1,
            "flags": [True, False, True],
            "tags": ("alpha", "beta"),
            "none_field": None,
        },
    }

    if with_autograd_tensors:
        root = source_root if source_root is not None else torch.randn(4, 4, requires_grad=True)
        inter = (root * 3.14).pow(2)
        leaf = inter.sum() + torch.tensor(float(step_id))
        step["autograd_loss_tensor"] = leaf
        step["autograd_matrix"] = inter

    if with_five_agent_memory:
        h = torch.randn(FIVE_AGENT_COUNT, 16, requires_grad=with_autograd_tensors)
        c = torch.randn(FIVE_AGENT_COUNT, 16, requires_grad=with_autograd_tensors)
        if with_autograd_tensors:
            h = (h * 2.0) + 1.0
            c = (c * 0.5) - 0.5
        prev_act = torch.tensor([1, 0, 0, 0, 0], dtype=torch.long)
        toks = torch.randn(FIVE_AGENT_COUNT, 4, 8, requires_grad=with_autograd_tensors)
        valid = torch.ones(FIVE_AGENT_COUNT, 4, dtype=torch.bool)
        step["initial_memories"] = FiveAgentMemory(
            hidden=h,
            cell=c,
            previous_action=prev_act,
            temporal_tokens=toks,
            temporal_valid=valid,
        )
    else:
        step["initial_memories"] = (
            np.zeros((FIVE_AGENT_COUNT, 32), dtype=np.float32),
            np.zeros((FIVE_AGENT_COUNT, 32), dtype=np.float32),
        )

    return step


# ============================================================================
# Dimension 1: Autograd Graph Detachment & CUDA/CPU Isolation Stress Tests
# ============================================================================

def test_autograd_graph_severed_on_buffer_add() -> None:
    """Verifies that all tensors added to the buffer have their autograd graph severed."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    root = torch.randn(8, 8, requires_grad=True)
    step = _make_adversarial_step_dict(
        step_id=1,
        with_autograd_tensors=True,
        source_root=root,
    )

    buffer.add(step, regime="UPTREND", realized_r=1.5)

    stored_step = buffer.uptrend_pool[0].step_dict
    stored_loss = stored_step["autograd_loss_tensor"]
    stored_mat = stored_step["autograd_matrix"]

    assert isinstance(stored_loss, torch.Tensor)
    assert stored_loss.grad_fn is None, "Autograd grad_fn was NOT severed!"
    assert not stored_loss.requires_grad, "requires_grad remains True!"
    assert stored_loss.device.type == "cpu", f"Device is {stored_loss.device.type}, not cpu!"

    assert isinstance(stored_mat, torch.Tensor)
    assert stored_mat.grad_fn is None
    assert not stored_mat.requires_grad
    assert stored_mat.device.type == "cpu"

    # Backward on original root tensor must NOT alter or raise in the stored detached tensor
    step["autograd_loss_tensor"].backward()
    assert root.grad is not None
    assert stored_loss.grad is None


def test_five_agent_memory_deep_sanitization() -> None:
    """Verifies that all 5 tensor components of FiveAgentMemory are detached and on CPU."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    step = _make_adversarial_step_dict(
        step_id=42,
        with_autograd_tensors=True,
        with_five_agent_memory=True,
    )
    buffer.add(step, regime="DOWNTREND", realized_r=-0.8)

    stored_mem = buffer.downtrend_pool[0].step_dict["initial_memories"]
    assert isinstance(stored_mem, FiveAgentMemory)

    for field_name in ("hidden", "cell", "previous_action", "temporal_tokens", "temporal_valid"):
        t = getattr(stored_mem, field_name)
        assert t is not None, f"Field {field_name} is None!"
        assert isinstance(t, torch.Tensor)
        assert t.grad_fn is None, f"{field_name} retained grad_fn: {t.grad_fn}"
        assert not t.requires_grad, f"{field_name} retained requires_grad=True"
        assert t.device.type == "cpu", f"{field_name} device is {t.device.type}, expected cpu"


def test_memory_leak_absence_across_1000_add_cycles() -> None:
    """Stress test: 1,000 add cycles with attached autograd graphs.
    
    If autograd graphs were retained, memory would grow proportionally to 1,000 graphs.
    With proper detachment and capacity capping, memory must remain strictly bounded.
    """
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    gc.collect()

    regimes = ["UPTREND", "DOWNTREND", "SIDEWAY"]
    for i in range(1000):
        # Create heavy autograd graph with multiple operations and allocations
        x = torch.randn(64, 64, requires_grad=True)
        w = torch.randn(64, 64, requires_grad=True)
        y = (x @ w).sin().sum()

        step = _make_adversarial_step_dict(
            step_id=i,
            with_autograd_tensors=True,
            with_five_agent_memory=True,
        )
        step["heavy_activation"] = y

        buffer.add(
            step,
            regime=regimes[i % 3],
            realized_r=1.0 if (i % 2 == 0) else -1.0,
        )

    gc.collect()

    # Buffer must strictly respect capacity capping
    assert len(buffer.uptrend_pool) <= 128
    assert len(buffer.downtrend_pool) <= 128
    assert len(buffer.sideway_pool) <= 128
    total_stored = len(buffer.uptrend_pool) + len(buffer.downtrend_pool) + len(buffer.sideway_pool)
    assert total_stored == 384  # Exactly 128 * 3

    # All 384 stored items must have zero grad_fn
    for pool in (buffer.uptrend_pool, buffer.downtrend_pool, buffer.sideway_pool):
        for exp in pool:
            act = exp.step_dict["heavy_activation"]
            assert act.grad_fn is None
            assert not act.requires_grad
            assert act.device.type == "cpu"
            mem = exp.step_dict["initial_memories"]
            assert mem.hidden.grad_fn is None
            assert not mem.hidden.requires_grad


def test_cuda_device_isolation_via_tensor_mocking() -> None:
    """Verifies that tensors regardless of original device are forced to CPU and detached."""
    mock_cuda_tensor = MagicMock(spec=torch.Tensor)
    mock_detached = MagicMock(spec=torch.Tensor)
    mock_cpu = MagicMock(spec=torch.Tensor)
    mock_cpu.device = torch.device("cpu")
    mock_cpu.requires_grad = False
    mock_cpu.grad_fn = None

    mock_cuda_tensor.detach.return_value = mock_detached
    mock_detached.cpu.return_value = mock_cpu

    sanitized = _sanitize_value_for_memory(mock_cuda_tensor)
    mock_cuda_tensor.detach.assert_called_once()
    mock_detached.cpu.assert_called_once()
    assert sanitized is mock_cpu


def test_deeply_nested_mixed_container_sanitization() -> None:
    """Verifies recursive sanitization inside arbitrary mixtures of tuples, dicts, lists."""
    t1 = torch.tensor([1.0], requires_grad=True)
    t2 = torch.tensor([2.0], requires_grad=True)
    arr = np.array([3.0], dtype=np.float32)

    nested = {
        "level1": [
            ({"target": t1}, (t2, arr)),
        ]
    }
    sanitized = _sanitize_value_for_memory(nested)

    stored_t1 = sanitized["level1"][0][0]["target"]
    stored_t2 = sanitized["level1"][0][1][0]
    stored_arr = sanitized["level1"][0][1][1]

    assert stored_t1.grad_fn is None
    assert not stored_t1.requires_grad
    assert stored_t2.grad_fn is None
    assert not stored_t2.requires_grad
    # numpy array must be an independent copy
    assert isinstance(stored_arr, np.ndarray)
    assert stored_arr is not arr
    np.testing.assert_array_equal(stored_arr, arr)


def test_sampled_batch_deepcopied_prevents_buffer_corruption() -> None:
    """Verifies that sample_balanced_batch returns deep copies, protecting buffer from mutation."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    step = _make_adversarial_step_dict(step_id=1, with_five_agent_memory=True)
    buffer.add(step, regime="UPTREND", realized_r=1.5)

    batch = buffer.sample_balanced_batch(batch_size=1)
    assert len(batch) == 1
    sampled_item = batch[0]

    # Mutate sampled numpy array and tensor in memory
    sampled_item["observations"][0, 0] = 9999.0
    mem: FiveAgentMemory = sampled_item["initial_memories"]
    mem.hidden[0, 0] = 8888.0

    # Stored item in buffer must remain completely unmodified
    stored_exp = buffer.uptrend_pool[0]
    stored_obs = stored_exp.step_dict["observations"]
    stored_mem: FiveAgentMemory = stored_exp.step_dict["initial_memories"]

    assert stored_obs[0, 0] == 1.0, "Buffer observation was mutated by sampled batch modification!"
    assert stored_mem.hidden[0, 0] != 8888.0, "Buffer hidden memory was mutated by sampled batch modification!"


# ============================================================================
# Dimension 2: Pickle Serialization Fidelity & Zero Type Loss Stress Tests
# ============================================================================

def test_direct_pickle_round_trip_fidelity_and_zero_type_loss() -> None:
    """Empirically validates that pickle.dumps / loads produces an identical object with zero type loss."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    for i in range(30):
        buffer.add(
            _make_adversarial_step_dict(step_id=i, with_five_agent_memory=True, regime_label="UPTREND"),
            regime="UPTREND",
            realized_r=1.2,
            agent_index=0,
            action_index=1,
            trace_id=f"up_{i}",
        )
        buffer.add(
            _make_adversarial_step_dict(step_id=100 + i, with_five_agent_memory=True, regime_label="DOWNTREND"),
            regime="DOWNTREND",
            realized_r=-1.1,
            agent_index=1,
            action_index=2,
            trace_id=f"down_{i}",
        )
        buffer.add(
            _make_adversarial_step_dict(step_id=200 + i, with_five_agent_memory=True, regime_label="SIDEWAY"),
            regime="SIDEWAY",
            realized_r=0.05,
            agent_index=4,
            action_index=0,
            trace_id=f"side_{i}",
        )

    # Sample some items so sampled_by_regime and total_sampled are populated
    _ = buffer.sample_balanced_batch(batch_size=18)

    # Serialize
    raw_payload = pickle.dumps(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    assert len(raw_payload) > 0

    # Deserialize
    restored: BalancedMultiRegimeReplayBuffer = pickle.loads(raw_payload)

    # 1. Structural assertions
    assert type(restored) is BalancedMultiRegimeReplayBuffer
    assert restored.capacity_per_regime == buffer.capacity_per_regime
    assert restored.schema_version == buffer.schema_version
    assert restored.total_admitted == buffer.total_admitted
    assert restored.total_sampled == buffer.total_sampled
    assert restored.admitted_by_regime == buffer.admitted_by_regime
    assert restored.sampled_by_regime == buffer.sampled_by_regime

    # 2. Pool length assertions
    assert len(restored.uptrend_pool) == len(buffer.uptrend_pool) == 30
    assert len(restored.downtrend_pool) == len(buffer.downtrend_pool) == 30
    assert len(restored.sideway_pool) == len(buffer.sideway_pool) == 30

    # 3. Deep item-by-item verification with zero type loss
    for pool_name in ("uptrend_pool", "downtrend_pool", "sideway_pool"):
        orig_pool: list[RegimeExperience] = getattr(buffer, pool_name)
        rest_pool: list[RegimeExperience] = getattr(restored, pool_name)

        for orig_exp, rest_exp in zip(orig_pool, rest_pool):
            assert type(rest_exp) is RegimeExperience
            assert rest_exp.regime == orig_exp.regime
            assert type(rest_exp.regime) is str
            assert rest_exp.realized_r == pytest.approx(orig_exp.realized_r)
            assert type(rest_exp.realized_r) is float
            assert rest_exp.is_win == orig_exp.is_win
            assert type(rest_exp.is_win) is bool
            assert rest_exp.agent_index == orig_exp.agent_index
            assert type(rest_exp.agent_index) is int
            assert rest_exp.action_index == orig_exp.action_index
            assert type(rest_exp.action_index) is int
            assert rest_exp.timestamp_ns == orig_exp.timestamp_ns
            assert type(rest_exp.timestamp_ns) is int
            assert rest_exp.trace_id == orig_exp.trace_id
            assert type(rest_exp.trace_id) is str

            orig_step = orig_exp.step_dict
            rest_step = rest_exp.step_dict

            # Check all arrays
            for arr_key in (
                "observations", "actions", "rewards", "allowed_action_masks",
                "targets", "sl_indices", "tp_indices", "utility_target", "excursion_target",
            ):
                o_val = orig_step[arr_key]
                r_val = rest_step[arr_key]
                assert type(r_val) is np.ndarray, f"{arr_key} type lost: {type(r_val)}"
                assert r_val.dtype == o_val.dtype, f"{arr_key} dtype mismatch: {r_val.dtype} vs {o_val.dtype}"
                assert r_val.shape == o_val.shape, f"{arr_key} shape mismatch: {r_val.shape} vs {o_val.shape}"
                np.testing.assert_array_equal(r_val, o_val)

            # Check FiveAgentMemory
            o_mem = orig_step["initial_memories"]
            r_mem = rest_step["initial_memories"]
            assert type(r_mem) is FiveAgentMemory
            for mem_field in ("hidden", "cell", "previous_action", "temporal_tokens", "temporal_valid"):
                o_t = getattr(o_mem, mem_field)
                r_t = getattr(r_mem, mem_field)
                if o_t is None:
                    assert r_t is None
                else:
                    assert type(r_t) is torch.Tensor
                    assert r_t.dtype == o_t.dtype, f"FiveAgentMemory.{mem_field} dtype altered!"
                    assert r_t.shape == o_t.shape
                    torch.testing.assert_close(r_t, o_t)


def test_dict_serialization_fidelity_and_zero_type_loss() -> None:
    """Validates to_dict() and from_dict() fidelity with zero type loss."""
    buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=64)
    buffer.add(
        _make_adversarial_step_dict(step_id=7, with_five_agent_memory=True, regime_label="UPTREND"),
        regime="UPTREND",
        realized_r=2.5,
    )

    data = buffer.to_dict()
    restored = BalancedMultiRegimeReplayBuffer.from_dict(data)

    assert restored.capacity_per_regime == buffer.capacity_per_regime
    assert len(restored.uptrend_pool) == 1
    exp = restored.uptrend_pool[0]
    assert exp.realized_r == pytest.approx(2.5)

    mem = exp.step_dict["initial_memories"]
    assert isinstance(mem, FiveAgentMemory)
    assert mem.hidden.dtype == torch.float32
    assert mem.cell.dtype == torch.float32
    assert mem.previous_action.dtype == torch.long
    assert mem.temporal_tokens.dtype == torch.float32
    assert mem.temporal_valid.dtype == torch.bool


def test_chuyen_gia_policy_candidate_pickle_integrity() -> None:
    """Verifies that an entire ChuyenGiaPolicy with a populated buffer survives pickle dump/load."""
    from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy

    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    policy.balanced_replay_buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    for i in range(10):
        policy.balanced_replay_buffer.add(
            _make_adversarial_step_dict(step_id=i, with_five_agent_memory=True),
            regime="UPTREND" if i % 2 == 0 else "DOWNTREND",
            realized_r=1.0 if i % 2 == 0 else -1.0,
        )

    assert policy.balanced_replay_buffer.stats()["total_samples"] == 10

    # Pickling the whole policy object (as candidate.pkl does)
    blob = pickle.dumps(policy, protocol=pickle.HIGHEST_PROTOCOL)
    loaded_policy: ChuyenGiaPolicy = pickle.loads(blob)

    assert loaded_policy.balanced_replay_buffer is not None
    assert loaded_policy.balanced_replay_buffer.stats()["total_samples"] == 10
    assert len(loaded_policy.balanced_replay_buffer.uptrend_pool) == 5
    assert len(loaded_policy.balanced_replay_buffer.downtrend_pool) == 5


# ============================================================================
# Dimension 3: flush_five_lifecycle_reward_queue() Integrity & Non-Mutation
# ============================================================================

def _compute_buffer_checksum(buffer: BalancedMultiRegimeReplayBuffer) -> str:
    """Hashes all bytes across all stored transitions in the buffer."""
    hasher = hashlib.sha256()
    for pool_name in ("uptrend_pool", "downtrend_pool", "sideway_pool"):
        pool: list[RegimeExperience] = getattr(buffer, pool_name)
        for exp in pool:
            hasher.update(exp.regime.encode("utf-8"))
            hasher.update(str(exp.realized_r).encode("utf-8"))
            hasher.update(str(exp.is_win).encode("utf-8"))
            hasher.update(str(exp.agent_index).encode("utf-8"))
            hasher.update(str(exp.action_index).encode("utf-8"))
            hasher.update(str(exp.timestamp_ns).encode("utf-8"))
            hasher.update(exp.trace_id.encode("utf-8"))
            step = exp.step_dict
            hasher.update(str(step.get("step_id", 0)).encode("utf-8"))
            for k in ("observations", "actions", "rewards"):
                if k in step and isinstance(step[k], np.ndarray):
                    hasher.update(step[k].tobytes())
    return hasher.hexdigest()


def test_flush_queue_preserves_buffer_and_does_not_mutate_transitions(monkeypatch) -> None:
    """Verifies that flush_five_lifecycle_reward_queue() does not delete the buffer or mutate stored transitions."""
    from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy

    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    policy.balanced_replay_buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)

    # Populate buffer with 45 transitions across 3 regimes
    for i in range(15):
        policy.balanced_replay_buffer.add(_make_adversarial_step_dict(step_id=i), regime="UPTREND", realized_r=1.0)
        policy.balanced_replay_buffer.add(_make_adversarial_step_dict(step_id=100 + i), regime="DOWNTREND", realized_r=-1.0)
        policy.balanced_replay_buffer.add(_make_adversarial_step_dict(step_id=200 + i), regime="SIDEWAY", realized_r=0.2)

    initial_stats = policy.balanced_replay_buffer.stats()
    assert initial_stats["total_samples"] == 45
    checksum_before = _compute_buffer_checksum(policy.balanced_replay_buffer)

    # Queue contains 12 pending lifecycle steps
    policy.five_lifecycle_steps = [
        _make_adversarial_step_dict(step_id=500 + i) for i in range(12)
    ]
    assert len(policy.five_lifecycle_steps) == 12

    # Mock observe_five_pattern_transition to simulate consumption of queue
    def fake_observe(traces, pattern_rewards, credit_channel="lifecycle", flush_pending=False, **kwargs):
        assert credit_channel == "lifecycle"
        assert flush_pending is True
        # consume queue items
        if policy.five_lifecycle_steps:
            del policy.five_lifecycle_steps[:min(len(policy.five_lifecycle_steps), 8)]
        return {"updated": 1.0}

    monkeypatch.setattr(policy, "observe_five_pattern_transition", fake_observe)

    # Execute flush
    result = policy.flush_five_lifecycle_reward_queue()

    # Invariants verification:
    # 1. Result telemetry matches expectations
    assert result["flushed_steps"] == 12
    assert result["pending"] == 0
    assert len(policy.five_lifecycle_steps) == 0

    # 2. Buffer was NOT deleted or reset
    assert policy.balanced_replay_buffer is not None
    assert isinstance(policy.balanced_replay_buffer, BalancedMultiRegimeReplayBuffer)
    assert policy.balanced_replay_buffer.stats()["total_samples"] == 45

    # 3. Zero mutation: Checksum across all stored transitions is perfectly identical
    checksum_after = _compute_buffer_checksum(policy.balanced_replay_buffer)
    assert checksum_before == checksum_after, "Stored transitions were mutated during flush!"


def test_flush_queue_idempotency_on_empty_queue(monkeypatch) -> None:
    """Verifies that calling flush on an empty queue is a clean no-op without side effects."""
    from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy

    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    policy.balanced_replay_buffer = BalancedMultiRegimeReplayBuffer(capacity_per_regime=128)
    policy.balanced_replay_buffer.add(_make_adversarial_step_dict(step_id=1), regime="UPTREND", realized_r=1.0)

    policy.five_lifecycle_steps = []

    res1 = policy.flush_five_lifecycle_reward_queue()
    assert res1 == {"flushed_batches": 0, "flushed_steps": 0, "pending": 0}
    assert policy.balanced_replay_buffer.stats()["total_samples"] == 1

    res2 = policy.flush_five_lifecycle_reward_queue()
    assert res2 == {"flushed_batches": 0, "flushed_steps": 0, "pending": 0}
    assert policy.balanced_replay_buffer.stats()["total_samples"] == 1
