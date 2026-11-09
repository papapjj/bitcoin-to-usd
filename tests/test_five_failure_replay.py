from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.five_failure_replay import FiveFailureReplayBank
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FivePPOTrainer,
    SharedFiveTradingPolicy,
)
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


def _small_policy() -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    policy.five_agent_memories = policy.five_shared_network.initial_memories()
    policy._learning_enabled = True
    policy.five_failure_replay_enabled = True
    return policy


def _add_case(
    bank: FiveFailureReplayBank, *, action: int = 1, realized_r: float = -1.0,
    trace_id: str = "trace-failure-1",
    condition_key: str = "CTX|MARKET_BUY|down|mid|expanding",
):
    hidden = 32
    masks = np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool)
    return bank.add(
        trace_id=trace_id,
        condition_key=condition_key,
        failed_agent_index=0,
        failed_action_index=action,
        realized_r=realized_r,
        observations=np.zeros((FIVE_AGENT_COUNT, INPUT_COUNT), dtype=np.float32),
        memories_hidden=np.zeros((FIVE_AGENT_COUNT, hidden), dtype=np.float32),
        memories_cell=np.zeros((FIVE_AGENT_COUNT, hidden), dtype=np.float32),
        previous_actions=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        actions=np.asarray([action, 0, 0, 0, 0], dtype=np.int64),
        targets=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        delay_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        risk_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        entry_offset_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        sl_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        tp_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        trailing_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        manage_mode_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        close_fraction_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        allowed_action_masks=masks,
    )


def test_failure_bank_replays_until_repaired_then_reopens_on_regression() -> None:
    bank = FiveFailureReplayBank(capacity=32)
    case_id = _add_case(bank)

    # A newly admitted failure is not replayed in the same optimizer batch.
    assert bank.active_replay(count=8, repair_probability=0.08) == []
    bank.note_batch_without_replay()
    assert [row.case_id for row in bank.active_replay(count=8, repair_probability=0.08)] == [case_id]

    # Three consecutive audits below the repair threshold put the old failure
    # to sleep; it no longer consumes training capacity.
    for _ in range(3):
        bank.observe_probabilities(
            {case_id: 0.05},
            repair_probability=0.08,
            reopen_probability=0.14,
            repair_streak_required=3,
        )
    assert bank.cases[case_id].resolved is True
    assert bank.active_replay(count=8, repair_probability=0.08) == []

    # Dormant cases are not forgotten. Once their audit interval expires, a
    # regression above the reopen threshold reactivates the old mistake.
    for _ in range(8):
        bank.note_batch_without_replay()
    due = bank.audit_candidates(audit_interval=8, max_items=8)
    assert case_id in {row.case_id for row in due}
    bank.observe_probabilities(
        {case_id: 0.20},
        repair_probability=0.08,
        reopen_probability=0.14,
        repair_streak_required=3,
    )
    assert bank.cases[case_id].resolved is False
    assert bank.cases[case_id].reopen_count == 1
    assert [row.case_id for row in bank.active_replay(count=8, repair_probability=0.08)] == [case_id]


def test_failure_replay_can_require_repeated_evidence_per_condition_bucket() -> None:
    bank = FiveFailureReplayBank(capacity=32)
    repeated_a = _add_case(bank, trace_id="trace-repeated-a")
    repeated_b = _add_case(bank, trace_id="trace-repeated-b")
    singleton = _add_case(
        bank,
        trace_id="trace-singleton",
        action=2,
        condition_key="CTX|MARKET_SELL|up|mid|expanding",
    )
    bank.note_batch_without_replay()

    selected = bank.active_replay(
        count=8, repair_probability=0.08, min_bucket_support=2
    )
    selected_ids = {row.case_id for row in selected}
    assert {repeated_a, repeated_b} <= selected_ids
    assert singleton not in selected_ids


def test_failure_replay_uses_current_policy_probability_not_stale_ppo_ratio() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    trainer = FivePPOTrainer(network)
    bank = FiveFailureReplayBank(capacity=32)
    case_id = _add_case(bank)
    bank.note_batch_without_replay()
    case = bank.cases[case_id].copy()

    before = trainer.evaluate_failure_replay_probabilities([case])[case_id]
    loss, stats = trainer.failure_replay_loss([case])
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert stats["failure_replay_samples"] == 1.0
    assert stats["failure_replay_failed_action_probability_mean"] == pytest.approx(before)

    # Make the one-step test visible. This is an auxiliary current-policy loss,
    # not a PPO update with an old behavior likelihood.
    trainer.optimizer.set_lr(1e-2)
    trainer.step_from_loss(loss)
    after = trainer.evaluate_failure_replay_probabilities([case])[case_id]
    assert after < before


def test_failure_replay_gradient_chi_nham_dung_tac_nhan_va_hanh_dong_loi(
    monkeypatch,
) -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    trainer = FivePPOTrainer(network)
    bank = FiveFailureReplayBank(capacity=32)
    case_id = _add_case(bank, action=1)
    case = bank.cases[case_id].copy()
    logits = torch.zeros((1, FIVE_AGENT_COUNT, ACTION_COUNT), requires_grad=True)

    monkeypatch.setattr(
        trainer,
        "_evaluate_failure_replay_case",
        lambda _case: {"policy_action_probabilities": torch.softmax(logits, dim=-1)},
    )
    loss, _ = trainer.failure_replay_loss([case])
    loss.backward()

    assert logits.grad is not None
    # Chỉ phân phối của agent sở hữu lỗi được sửa. Bốn agent còn lại và mọi
    # trạng thái không nằm trong case không nhận gradient cấm toàn cục.
    assert torch.count_nonzero(logits.grad[0, 1:]).item() == 0
    assert logits.grad[0, 0, 1] > 0
    assert torch.count_nonzero(logits.grad[0, 0]).item() == ACTION_COUNT


def test_material_lifecycle_loss_is_admitted_to_persistent_failure_bank() -> None:
    policy = _small_policy()
    rows = list(policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details["five_action_index"] = 1 if index == 0 else 0
        details["five_economic_actor_eligible"] = bool(index == 0)
        details["five_execution_filled"] = bool(index == 0)
        enriched.append(replace(trace, action_details=details))

    registered = policy.register_five_entry_lifecycle_packet(tuple(enriched))
    assert registered["registered"] == 1
    routed = policy.apply_five_entry_lifecycle_reward(
        target_id,
        lifecycle_reward_r=-0.30,
        realized_r=-1.0,
        components={
            "geometry_quality_r": -0.30,
            "entry_excursion_quality_r": 0.0,
            "initial_sl": True,
            "total_r": -0.30,
        },
    )

    assert routed["accepted"] is True
    assert routed["failure_replay_case_id"]
    assert policy.five_failure_replay_bank.stats()["cases"] == 1
    stored = next(iter(policy.five_failure_replay_bank.cases.values()))
    assert stored.failed_agent_index == 0
    assert stored.failed_action_index == 1
    assert stored.realized_r == pytest.approx(-1.0)
    assert stored.memories_temporal_tokens is not None
    assert stored.memories_temporal_tokens.shape[:2] == (FIVE_AGENT_COUNT, 4)
    assert stored.memories_temporal_valid is not None
    assert stored.memories_temporal_valid.shape == (FIVE_AGENT_COUNT, 4)


def test_non_material_or_profitable_lifecycle_is_not_admitted() -> None:
    policy = _small_policy()
    rows = list(policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details["five_action_index"] = 1 if index == 0 else 0
        details["five_economic_actor_eligible"] = bool(index == 0)
        details["five_execution_filled"] = bool(index == 0)
        enriched.append(replace(trace, action_details=details))
    policy.register_five_entry_lifecycle_packet(tuple(enriched))
    routed = policy.apply_five_entry_lifecycle_reward(
        target_id,
        lifecycle_reward_r=0.05,
        realized_r=0.10,
        components={"total_r": 0.05},
    )
    assert routed["accepted"] is True
    assert routed["failure_replay_case_id"] == ""
    assert policy.five_failure_replay_bank.stats()["cases"] == 0
