from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.orders import Action
from self_growing_trader.five_agent_authority import five_agent_authority_mask
from self_growing_trader.replacement_learning import action_scope
from self_growing_trader.five_shared_policy import (
    ACCOUNT_EMBED_SIZE,
    EXECUTION_EMBED_SIZE,
    FIVE_AGENT_COUNT,
    FIVE_LSTM_UNROLL,
    FIVE_OBSERVATION_ARCHITECTURE_VERSION,
    MARKET_STRUCTURE_EMBED_SIZE,
    POSITION_EMBED_SIZE,
    FiveObservationEncoder,
    FiveAgentMemory,
    SharedFiveTradingPolicy,
    FiveTrainingSchedule,
    FiveEconomicState,
    FiveTradingRewardShaper,
    ProcessSet,
    mix_team_spirit_rewards,
    time_weight_shaped_rewards,
    discount_from_horizon,
)
from self_growing_trader.protocol_contract import (
    ACCOUNT_OFFSET,
    EXECUTION_OFFSET,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    POSITION_OFFSET,
)
from self_growing_trader.he_chuyen_gia_ngu_canh import (
    CauHinhHeChuyenGia,
    HeChuyenGiaMotContextMotPolicy,
)


def _small_policy() -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy(input_size=137, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    policy.five_agent_memories = policy.five_shared_network.initial_memories()
    policy._learning_enabled = False
    return policy


def test_nam_tac_nhan_dung_chung_dung_mot_bo_tham_so() -> None:
    policy = _small_policy()
    network_id = id(policy.five_shared_network)
    count = policy.five_parameter_count()
    traces = policy.act_five_agents(np.linspace(-1.0, 1.0, 137))
    assert len(traces) == FIVE_AGENT_COUNT == 5
    assert id(policy.five_shared_network) == network_id
    assert policy.five_parameter_count() == count
    assert policy.modules == []
    assert len(policy.five_agent_memories) == 5


def test_five_expectancy_veto_giu_context_causal_va_chan_dung_entry_xau() -> None:
    policy = _small_policy()
    policy._learning_enabled = False
    policy.five_expectancy_action_veto_enabled = True
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-veto-test",
        timestamp=1.0,
    )
    first = policy.act_five_agents(observation)
    key = np.asarray(first[0].canonical_context_key, dtype=float)
    ledger = policy.memory_supernet.contextual_expectancy
    for _ in range(12):
        ledger.observe(
            key,
            action_scope(1, positions_pipe=""),
            -1.0,
            origin="policy",
        )

    second = policy.act_five_agents(observation)

    assert second[0].context_frame_id == "five-veto-test"
    assert second[0].canonical_context_key.shape == key.shape
    assert second[0].action_details["five_expectancy_query_available"] is True
    assert second[0].action_details["five_expectancy_reliable_negative_actions"] == [1]
    assert second[0].action_details["five_expectancy_vetoed_actions"] == [1]
    assert second[0].allowed_action_mask[1] is False
    assert second[0].allowed_action_mask[0] is True


def test_five_expectancy_aggregate_fallback_veto_khi_context_troi_prototype() -> None:
    policy = _small_policy()
    policy._learning_enabled = False
    policy.five_expectancy_action_veto_enabled = True
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-aggregate-veto-test",
        timestamp=1.0,
    )
    first = policy.act_five_agents(observation)
    shifted_key = np.asarray(first[0].canonical_context_key, dtype=float).copy()
    shifted_key[:8] *= -1.0
    ledger = policy.memory_supernet.contextual_expectancy
    ledger.query_min_similarity = 0.99
    for _ in range(16):
        ledger.observe(
            shifted_key,
            action_scope(1, positions_pipe=""),
            -1.0,
            origin="policy",
        )

    second = policy.act_five_agents(observation)

    assert second[0].action_details["five_expectancy_query_available"] is True
    assert second[0].action_details["five_expectancy_aggregate_negative_actions"] == [1]
    assert second[0].action_details["five_expectancy_vetoed_actions"] == [1]
    assert second[0].allowed_action_mask[1] is False


def test_five_expectancy_veto_khong_khoa_exact_context_chua_du_bang_chung() -> None:
    policy = _small_policy()
    policy.five_expectancy_action_veto_enabled = True
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-veto-threshold-test",
        timestamp=1.0,
    )
    first = policy.act_five_agents(observation)
    key = np.asarray(first[0].canonical_context_key, dtype=float)
    ledger = policy.memory_supernet.contextual_expectancy
    for _ in range(8):
        ledger.observe(key, action_scope(1, positions_pipe=""), -1.0, origin="policy")

    second = policy.act_five_agents(observation)

    assert second[0].action_details["five_expectancy_reliable_negative_actions"] == []
    assert second[0].action_details["five_expectancy_vetoed_actions"] == []
    assert second[0].allowed_action_mask[1] is True


def test_five_expectancy_veto_force_hold_owner_khi_khong_co_entry_thay_the_duong() -> None:
    policy = _small_policy()
    policy._learning_enabled = False
    policy.five_expectancy_action_veto_enabled = True
    policy.five_expectancy_action_veto_hold_owner_on_negative = True
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-veto-hold-owner-test",
        timestamp=1.0,
    )
    first = policy.act_five_agents(observation)
    key = np.asarray(first[0].canonical_context_key, dtype=float)
    ledger = policy.memory_supernet.contextual_expectancy
    for _ in range(12):
        ledger.observe(key, action_scope(1, positions_pipe=""), -1.0, origin="policy")

    second = policy.act_five_agents(observation)

    assert second[0].action_details["five_expectancy_hold_owner_actions"] == [0]
    assert second[0].allowed_action_mask[0] is True
    assert second[0].allowed_action_mask[1] is False
    assert second[0].allowed_action_mask[3] is False
    assert second[0].allowed_action_mask[5] is False


def test_five_live_action_veto_pool_evidence_across_contexts() -> None:
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_action_veto_min_samples = 12
    policy.five_live_action_veto_min_confidence = 0.20
    policy.five_live_action_veto_min_lcb_r = -0.15
    policy.five_live_action_veto_hold_owner_on_negative = True
    policy.five_expectancy_action_veto_enabled = False
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-live-action-veto-test",
        timestamp=1.0,
    )
    for index in range(12):
        assert policy.record_five_live_owner_outcome(
            agent_index=0,
            action=1,
            net_r=-1.0,
            trace_id=f"live-action-veto-{index}",
        ) is True

    traces = policy.act_five_agents(observation)

    assert traces[0].action_details["five_live_action_veto_negative_actions"] == [1]
    assert traces[0].action_details["five_live_action_veto_vetoed_actions"] == [1, 3, 5]
    assert traces[0].action_details["five_live_action_veto_hold_owner_actions"] == [0]
    assert traces[0].action_details["five_live_action_veto_samples"][1] == 12
    assert traces[0].allowed_action_mask[0] is True
    assert traces[0].allowed_action_mask[1] is False
    assert traces[0].allowed_action_mask[3] is False
    assert traces[0].allowed_action_mask[5] is False


def test_five_live_action_veto_probe_keeps_one_unproven_entry() -> None:
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_action_veto_min_samples = 12
    policy.five_live_action_veto_min_confidence = 0.20
    policy.five_live_action_veto_min_lcb_r = -0.15
    policy.five_live_action_veto_hold_owner_on_negative = True
    policy.five_live_action_veto_probe_enabled = True
    policy.five_live_action_veto_probe_interval_decisions = 2
    policy.five_expectancy_action_veto_enabled = False
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-live-action-veto-probe-test",
        timestamp=1.0,
    )
    for index in range(12):
        assert policy.record_five_live_owner_outcome(
            agent_index=0,
            action=1,
            net_r=-1.0,
            trace_id=f"live-action-veto-probe-{index}",
        ) is True

    traces = policy.act_five_agents(observation)

    assert traces[0].action_details["five_live_action_veto_probe_enabled"] is True
    assert traces[0].action_details["five_live_action_veto_probe_actions"] == [3]
    assert traces[0].action_details["five_live_action_veto_hold_owner_actions"] == []
    assert traces[0].allowed_action_mask[0] is True
    assert traces[0].allowed_action_mask[1] is False
    assert traces[0].allowed_action_mask[3] is True
    assert traces[0].allowed_action_mask[5] is False

    suppressed = policy.act_five_agents(observation)

    assert suppressed[0].action_details["five_live_action_veto_probe_due"] is False
    assert suppressed[0].action_details["five_live_action_veto_probe_actions"] == []
    assert suppressed[0].action_details["five_live_action_veto_hold_owner_actions"] == [0]
    assert suppressed[0].allowed_action_mask[0] is True
    assert suppressed[0].allowed_action_mask[3] is False


def test_five_live_owner_bootstrap_probe_opens_zero_sample_short_owner() -> None:
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_owner_bootstrap_probe_enabled = True
    policy.five_live_owner_bootstrap_probe_interval_decisions = 1
    policy.five_live_owner_bootstrap_probe_max_per_run = 1
    policy.set_decision_context(
        account_context={"position_count_norm": 0.0},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={"pending_orders": 0.0, "orders_pipe": ""},
        sequence_id=1,
        frame_id="five-owner-bootstrap-short-test",
        timestamp=1.0,
    )

    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))
    owner = traces[1]
    details = owner.action_details

    assert details["five_live_owner_bootstrap_probe_due"] is True
    assert details["five_live_owner_bootstrap_probe_owner"] == 1
    assert details["five_live_owner_bootstrap_probe_action"] == int(Action.MARKET_SELL)
    assert details["five_live_owner_bootstrap_probe_samples"] == [0, 0]
    assert owner.allowed_action_mask[int(Action.HOLD)] is True
    assert owner.allowed_action_mask[int(Action.MARKET_SELL)] is True
    assert owner.allowed_action_mask[int(Action.SELL_LIMIT)] is False
    assert owner.allowed_action_mask[int(Action.SELL_STOP)] is False

    exhausted = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))[1]
    assert exhausted.action_details["five_live_owner_bootstrap_probe_owner"] is None
    assert exhausted.action_details["five_live_owner_bootstrap_probe_count"] == 1


def test_five_live_owner_bootstrap_probe_skips_non_flat_account() -> None:
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_owner_bootstrap_probe_enabled = True
    policy.five_live_owner_bootstrap_probe_interval_decisions = 1
    policy.set_decision_context(
        account_context={"position_count_norm": 0.2},
        positions_pipe="position-ticket-1",
        allowed_actions={index: True for index in range(12)},
        execution_context={"pending_orders": 0.0, "orders_pipe": ""},
        sequence_id=1,
        frame_id="five-owner-bootstrap-open-position-test",
        timestamp=1.0,
    )

    owner = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))[1]

    assert owner.action_details["five_live_owner_bootstrap_probe_due"] is True
    assert owner.action_details["five_live_owner_bootstrap_probe_owner"] is None
    assert owner.action_details["five_live_owner_bootstrap_probe_count"] == 0


def test_five_live_owner_bootstrap_probe_force_market_is_trace_visible() -> None:
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_owner_bootstrap_probe_enabled = True
    policy.five_live_owner_bootstrap_probe_force_market = True
    policy.five_live_owner_bootstrap_probe_interval_decisions = 1
    policy.five_live_owner_bootstrap_probe_max_per_run = 1
    policy.set_decision_context(
        account_context={"position_count_norm": 0.0},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={"pending_orders": 0.0, "orders_pipe": ""},
        sequence_id=1,
        frame_id="five-owner-bootstrap-force-market-test",
        timestamp=1.0,
    )

    owner = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))[1]

    assert owner.action_details["five_live_owner_bootstrap_probe_force_market"] is True
    assert owner.action_details["five_live_owner_bootstrap_probe_owner"] == 1
    assert owner.action_details["five_live_owner_bootstrap_probe_action"] == int(Action.MARKET_SELL)
    assert owner.allowed_action_mask[int(Action.HOLD)] is False
    assert owner.allowed_action_mask[int(Action.MARKET_SELL)] is True
    assert owner.selected_idx == int(Action.MARKET_SELL)
    # Execution is forced to MARKET_SELL, but the learner must retain the
    # pre-override owner surface so this bounded probe has a non-degenerate
    # PPO distribution instead of an unavoidable 1.0 -> 1.0 no-op.
    learning_mask = owner.action_details["five_rlhf_update_action_mask"]
    assert learning_mask[int(Action.HOLD)] is True
    assert learning_mask[int(Action.MARKET_SELL)] is True
    assert owner.action_details["five_rlhf_update_action_mask_source"] == (
        "owner_bootstrap_probe_pre_force_market"
    )


def test_five_live_action_veto_probe_recovers_when_hold_owner_flag_is_false() -> None:
    """A negative pool must not permanently remove every entry action."""
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_action_veto_min_samples = 6
    policy.five_live_action_veto_min_confidence = 0.20
    policy.five_live_action_veto_min_lcb_r = -0.15
    policy.five_live_action_veto_hold_owner_on_negative = False
    policy.five_live_action_veto_probe_enabled = True
    policy.five_live_action_veto_probe_interval_decisions = 1
    policy.five_live_action_veto_probe_on_negative_transition = True
    policy.five_expectancy_action_veto_enabled = False
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-live-action-veto-probe-hold-false-test",
        timestamp=1.0,
    )
    for action in (1, 3, 5):
        for index in range(6):
            assert policy.record_five_live_owner_outcome(
                agent_index=0,
                action=action,
                net_r=-1.0,
                trace_id=f"live-action-veto-probe-hold-false-{action}-{index}",
            ) is True

    traces = policy.act_five_agents(observation)
    details = traces[0].action_details

    assert details["five_live_action_veto_negative_actions"] == [1, 3, 5]
    assert details["five_live_action_veto_probe_due"] is True
    assert len(details["five_live_action_veto_probe_actions"]) == 1
    probe_action = details["five_live_action_veto_probe_actions"][0]
    assert probe_action in (1, 3, 5)
    assert details["five_live_action_veto_hold_owner_actions"] == []
    assert traces[0].allowed_action_mask[0] is True
    assert traces[0].allowed_action_mask[probe_action] is True
    assert sum(bool(value) for value in traces[0].allowed_action_mask) == 2


def test_five_live_action_veto_exploration_keeps_entry_between_probes() -> None:
    """Enabled owner exploration must not become HOLD-only off-probe."""
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_action_veto_min_samples = 6
    policy.five_live_action_veto_min_confidence = 0.20
    policy.five_live_action_veto_min_lcb_r = -0.15
    policy.five_live_action_veto_hold_owner_on_negative = False
    policy.five_live_action_veto_probe_enabled = True
    policy.five_live_action_veto_probe_interval_decisions = 16
    policy.five_live_owner_exploration_enabled = True
    policy.five_expectancy_action_veto_enabled = False
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-live-action-veto-exploration-between-probes",
        timestamp=1.0,
    )
    # Consume the initial probe lease so the next call is an off-probe
    # decision. The negative owner evidence is then still allowed to collect
    # another lifecycle instead of being hard-routed to HOLD.
    policy.act_five_agents(observation)
    for action in (1, 3, 5):
        for index in range(6):
            assert policy.record_five_live_owner_outcome(
                agent_index=0,
                action=action,
                net_r=-1.0,
                trace_id=f"live-action-veto-exploration-{action}-{index}",
            ) is True

    traces = policy.act_five_agents(observation)
    details = traces[0].action_details

    assert details["five_live_action_veto_probe_due"] is False
    assert details["five_live_owner_exploration_escape_actions"] == [1]
    assert traces[0].allowed_action_mask[0] is True
    assert traces[0].allowed_action_mask[1] is True
    assert sum(bool(value) for value in traces[0].allowed_action_mask) == 2


def test_five_live_action_veto_probe_starts_on_negative_transition_and_survives_context_veto() -> None:
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_action_veto_min_samples = 12
    policy.five_live_action_veto_min_confidence = 0.20
    policy.five_live_action_veto_min_lcb_r = -0.15
    policy.five_live_action_veto_hold_owner_on_negative = True
    policy.five_live_action_veto_probe_enabled = True
    policy.five_live_action_veto_probe_interval_decisions = 64
    policy.five_live_action_veto_probe_on_negative_transition = True
    policy.five_expectancy_action_veto_enabled = True
    policy.five_expectancy_action_veto_hold_owner_on_negative = True
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-live-action-veto-transition-test",
        timestamp=1.0,
    )
    first = policy.act_five_agents(observation)
    key = np.asarray(first[0].canonical_context_key, dtype=float)
    ledger = policy.memory_supernet.contextual_expectancy
    for index in range(12):
        ledger.observe(key, action_scope(1, positions_pipe=""), -1.0, origin="policy")
        ledger.observe(key, action_scope(3, positions_pipe=""), -1.0, origin="policy")
        assert policy.record_five_live_owner_outcome(
            agent_index=0,
            action=1,
            net_r=-1.0,
            trace_id=f"live-action-veto-transition-{index}",
        ) is True

    second = policy.act_five_agents(observation)

    assert second[0].action_details["five_live_action_veto_probe_due"] is False
    assert second[0].action_details["five_live_action_veto_probe_transition_owners"] == [0]
    assert second[0].action_details["five_live_action_veto_probe_actions"] == [3]
    assert second[0].action_details["five_expectancy_probe_preserved_actions"] == [3]
    assert second[0].allowed_action_mask[0] is True
    assert second[0].allowed_action_mask[1] is False
    assert second[0].allowed_action_mask[3] is True


def test_context_veto_periodic_probe_survives_without_live_negative_samples() -> None:
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_action_veto_probe_enabled = True
    policy.five_live_action_veto_probe_interval_decisions = 1
    policy.five_expectancy_action_veto_enabled = True
    policy.five_expectancy_action_veto_hold_owner_on_negative = True
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-context-probe-without-live-evidence",
        timestamp=1.0,
    )
    first = policy.act_five_agents(observation)
    key = np.asarray(first[0].canonical_context_key, dtype=float)
    ledger = policy.memory_supernet.contextual_expectancy
    for index in range(12):
        ledger.observe(key, action_scope(1, positions_pipe=""), -1.0, origin="policy")
        ledger.observe(key, action_scope(3, positions_pipe=""), -1.0, origin="policy")

    second = policy.act_five_agents(observation)

    assert second[0].action_details["five_live_action_veto_probe_due"] is True
    assert second[0].action_details["five_expectancy_probe_preserved_actions"] == [5]
    assert second[0].action_details["five_expectancy_hold_owner_actions"] == []
    assert second[0].allowed_action_mask[0] is True
    assert second[0].allowed_action_mask[1] is False
    assert second[0].allowed_action_mask[3] is False
    assert second[0].allowed_action_mask[5] is True


def test_context_veto_does_not_collapse_owner_to_hold_before_live_probe() -> None:
    """A stale contextual veto must leave one bounded entry escape hatch."""
    policy = _small_policy()
    policy.five_live_action_veto_enabled = True
    policy.five_live_action_veto_probe_enabled = True
    policy.five_live_action_veto_probe_interval_decisions = 16
    policy.five_expectancy_action_veto_enabled = True
    policy.five_expectancy_action_veto_hold_owner_on_negative = False
    observation = np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=np.float32)
    policy.set_decision_context(
        account_context={},
        positions_pipe="",
        allowed_actions={index: True for index in range(12)},
        execution_context={},
        sequence_id=1,
        frame_id="five-context-escape-before-live-probe",
        timestamp=1.0,
    )
    first = policy.act_five_agents(observation)
    key = np.asarray(first[0].canonical_context_key, dtype=float)
    ledger = policy.memory_supernet.contextual_expectancy
    for action in (1, 3, 5):
        for _ in range(12):
            ledger.observe(key, action_scope(action, positions_pipe=""), -1.0, origin="policy")

    second = policy.act_five_agents(observation)
    details = second[0].action_details

    assert details["five_live_action_veto_probe_due"] is False
    assert details["five_expectancy_context_escape_actions"] == [1]
    assert second[0].allowed_action_mask[0] is True
    assert second[0].allowed_action_mask[1] is True
    assert sum(bool(value) for value in second[0].allowed_action_mask) == 2


def test_candidate_direct_action_decision_mode_overrides_only_runtime_selector() -> None:
    policy = _small_policy()
    for module in (
        policy.five_shared_network.heads.direct_action_head,
        policy.five_shared_network.heads.direct_observation_head,
        policy.five_shared_network.heads.value,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    policy.five_action_decision_mode = "direct"
    policy._ensure_public_five_runtime()

    assert policy.five_shared_network.heads.seven_choice_action_decision_mode == "direct"
    assert policy.five_shared_network.heads.seven_choice_expected_value_decision_enabled is False
    assert all(
        parameter.requires_grad
        for module in (
            policy.five_shared_network.heads.direct_action_head,
            policy.five_shared_network.heads.direct_observation_head,
            policy.five_shared_network.heads.value,
        )
        for parameter in module.parameters()
    )


def test_candidate_direct_action_mode_keeps_ppo_loss_differentiable() -> None:
    policy = _small_policy()
    policy.five_action_decision_mode = "direct"
    policy._learning_enabled = True
    policy._ensure_public_five_runtime()
    trainer = policy._ensure_five_trainer()
    memories = policy.five_shared_network.initial_memories()

    stats = trainer.optimize_sequence(
        observations=np.zeros((1, FIVE_AGENT_COUNT, INPUT_COUNT), dtype=np.float32),
        initial_memories=memories,
        actions=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64),
        targets=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64),
        delay_indices=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64),
        risk_indices=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64),
        entry_offset_indices=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64),
        manage_mode_indices=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64),
        close_fraction_indices=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64),
        old_log_probabilities=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.float32),
        rewards=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.float32),
        dones=np.ones((1, FIVE_AGENT_COUNT), dtype=np.float32),
        step_seconds=60.0,
        horizon_seconds=60.0,
        allowed_action_masks=np.ones((1, FIVE_AGENT_COUNT, 12), dtype=bool),
        policy_loss_masks=np.ones((1, FIVE_AGENT_COUNT), dtype=bool),
        sample_loss_masks=np.ones((1, FIVE_AGENT_COUNT), dtype=bool),
        behavior_uniform_mixes=np.zeros((1, FIVE_AGENT_COUNT), dtype=np.float32),
        flat_account_masks=np.zeros((1, FIVE_AGENT_COUNT), dtype=bool),
    )

    assert np.isfinite(stats["total_loss"])


def test_candidate_direct_action_mode_keeps_independent_pattern_raw_shape() -> None:
    from self_growing_trader.five_shared_policy import FivePPOTrainer

    net = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    net.heads.seven_choice_action_decision_mode = "direct"
    net.heads.seven_choice_expected_value_decision_enabled = False
    trainer = FivePPOTrainer(net)
    observations = np.zeros((1, FIVE_AGENT_COUNT, INPUT_COUNT), dtype=np.float32)
    zeros = np.zeros((1, FIVE_AGENT_COUNT), dtype=np.int64)
    float_zeros = np.zeros((1, FIVE_AGENT_COUNT), dtype=np.float32)
    stats = trainer.optimize_independent_pattern_steps(
        observations=observations,
        initial_memories=[net.initial_memories()],
        actions=zeros,
        targets=zeros,
        delay_indices=zeros,
        risk_indices=zeros,
        entry_offset_indices=zeros,
        manage_mode_indices=zeros,
        close_fraction_indices=zeros,
        old_log_probabilities=float_zeros,
        old_values=float_zeros,
        rewards=float_zeros,
        allowed_action_masks=np.ones((1, FIVE_AGENT_COUNT, 12), dtype=bool),
        policy_loss_masks=np.ones((1, FIVE_AGENT_COUNT), dtype=bool),
        pattern_tree_enabled=np.zeros(1, dtype=bool),
        utility_targets=np.zeros((1, 2), dtype=np.float32),
        no_edge_targets=np.zeros(1, dtype=np.float32),
    )

    assert np.isfinite(stats["total_loss"])


def test_do_tin_cay_five_la_xac_suat_cua_hanh_dong_da_chon() -> None:
    policy = _small_policy()
    observation = np.zeros(INPUT_COUNT, dtype=np.float32)
    results, _, _ = policy.five_shared_network.forward_five(
        torch.zeros((FIVE_AGENT_COUNT, INPUT_COUNT)),
        policy.five_shared_network.initial_memories(),
        deterministic=True,
    )
    probabilities = np.zeros(12, dtype=np.float32)
    probabilities[0] = 0.80
    probabilities[2] = 0.20
    behavior = np.full(12, 1.0 / 12.0, dtype=np.float32)
    selected_non_max = replace(
        results[0], action_index=2, action_probabilities=behavior,
        policy_action_probabilities=probabilities,
        behavior_action_probabilities=behavior,
    )
    trace = policy._five_trace_from_result(observation, 0, selected_non_max)
    assert trace.confidence == pytest.approx(0.20)
    assert trace.action_details["five_selected_action_probability"] == pytest.approx(0.20)
    assert trace.action_details["five_selected_behavior_action_probability"] == pytest.approx(1.0 / 12.0)


def test_moi_tac_nhan_co_o_nho_va_trang_thai_an_rieng() -> None:
    policy = _small_policy()
    traces = policy.act_five_agents(np.ones(137, dtype=float))
    assert all(trace.memory_after.shape == (32,) for trace in traces)
    hidden = torch.stack([m.hidden for m in policy.five_agent_memories]).cpu().numpy()
    cells = torch.stack([m.cell for m in policy.five_agent_memories]).cpu().numpy()
    assert np.unique(np.round(hidden, 7), axis=0).shape[0] > 1
    assert np.unique(np.round(cells, 7), axis=0).shape[0] > 1
    for index, trace in enumerate(traces):
        assert trace.action_details["five_agent_index"] == index
        assert trace.raw_obs.shape == (137,)
        assert trace.action_details["five_agent_shared_parameters"] is True


def test_gom_cheo_25_phan_tram_giong_nhau_giua_nam_tac_nhan() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    obs = torch.stack([torch.full((137,), float(i)) for i in range(5)])
    memories = net.initial_memories()
    ids = torch.arange(5, dtype=torch.long)
    previous = torch.stack([m.previous_action for m in memories])
    encoded, _, _ = net.observation_encoder(obs, ids, previous)
    width = 16
    for index in range(1, 5):
        torch.testing.assert_close(encoded[0, :width], encoded[index, :width])


def test_tap_vi_the_khong_phu_thuoc_thu_tu() -> None:
    processor = ProcessSet(8, 16, 12)
    values = torch.randn(1, 5, 8)
    valid = torch.tensor([[True, True, True, False, False]])
    pooled_a, _ = processor(values, valid)
    perm = torch.tensor([2, 0, 1, 3, 4])
    pooled_b, _ = processor(values[:, perm], valid[:, perm])
    torch.testing.assert_close(pooled_a, pooled_b)


def test_encoder_tach_dung_cac_nhom_dac_trung_co_cau_truc() -> None:
    torch.manual_seed(7)
    encoder = FiveObservationEncoder(pre_lstm_size=32)
    observations = torch.randn(5, INPUT_COUNT) * 0.1
    agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    previous_actions = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    components = encoder.encode_components(observations, agent_ids, previous_actions)

    assert components["market"].shape == (5, MARKET_STRUCTURE_EMBED_SIZE)
    assert components["account"].shape == (5, ACCOUNT_EMBED_SIZE)
    assert components["execution"].shape == (5, EXECUTION_EMBED_SIZE)
    assert components["positions"].shape == (5, POSITION_EMBED_SIZE)
    assert components["position_embeddings"].shape == (5, 5, POSITION_EMBED_SIZE)
    assert components["position_valid"].shape == (5, 5)
    assert components["agent"].shape == (5, 32)
    assert components["previous_action"].shape == (5, 32)
    assert encoder.market_structure.conv1.kernel_size == (3,)
    assert encoder.market_structure.conv2.kernel_size == (3,)


def test_thay_doi_tai_khoan_khong_ro_ri_sang_nhanh_gia_va_thuc_thi() -> None:
    torch.manual_seed(8)
    encoder = FiveObservationEncoder(pre_lstm_size=32).eval()
    base = torch.randn(5, INPUT_COUNT) * 0.05
    changed = base.clone()
    changed[:, ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET] += 0.7
    ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    previous = torch.zeros(FIVE_AGENT_COUNT, dtype=torch.long)
    with torch.no_grad():
        before = encoder.encode_components(base, ids, previous)
        after = encoder.encode_components(changed, ids, previous)
    assert not torch.allclose(before["account"], after["account"])
    torch.testing.assert_close(before["market"], after["market"])
    torch.testing.assert_close(before["execution"], after["execution"])
    torch.testing.assert_close(before["positions"], after["positions"])


def test_gradient_di_qua_du_cac_nhanh_dac_trung_moi() -> None:
    torch.manual_seed(9)
    encoder = FiveObservationEncoder(pre_lstm_size=32)
    observations = torch.randn(5, INPUT_COUNT) * 0.05
    ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    previous = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    encoded, _, _ = encoder(observations, ids, previous)
    encoded.square().mean().backward()

    parameters = (
        encoder.market_structure.conv1.weight,
        encoder.market_structure.conv2.weight,
        encoder.account_encoder.fc1.weight,
        encoder.execution_encoder.fc1.weight,
        encoder.position_set.fc1.weight,
        encoder.agent_embedding.weight,
        encoder.previous_action_embedding.weight,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(float(parameter.grad.abs().sum().item()) > 0.0 for parameter in parameters)


def test_encoder_khu_gia_tri_khong_huu_han_truoc_khi_tim_dac_trung() -> None:
    encoder = FiveObservationEncoder(pre_lstm_size=32)
    observations = torch.zeros(5, INPUT_COUNT)
    observations[0, 0] = torch.nan
    observations[1, ACCOUNT_OFFSET] = torch.inf
    observations[2, EXECUTION_OFFSET] = -torch.inf
    observations[3, POSITION_OFFSET] = torch.nan
    encoded, position_embeddings, _ = encoder(
        observations,
        torch.arange(FIVE_AGENT_COUNT, dtype=torch.long),
        torch.zeros(FIVE_AGENT_COUNT, dtype=torch.long),
    )
    assert torch.isfinite(encoded).all()
    assert torch.isfinite(position_embeddings).all()


def test_checkpoint_five_v2_duoc_migrate_trung_tinh_khong_doi_encoder_output() -> None:
    torch.manual_seed(10)
    encoder = FiveObservationEncoder(pre_lstm_size=32)
    observations = torch.randn(5, INPUT_COUNT) * 0.05
    agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    previous = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    with torch.no_grad():
        before = encoder(observations, agent_ids, previous)[0]
    del encoder.pattern_tree
    del encoder.pattern_residual
    encoder.architecture_version = FIVE_OBSERVATION_ARCHITECTURE_VERSION - 1
    with torch.no_grad():
        after = encoder(observations, agent_ids, previous)[0]
    assert encoder.architecture_version == FIVE_OBSERVATION_ARCHITECTURE_VERSION
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_checkpoint_five_v3_duoc_gan_head_path_v4_ma_khong_doi_encoder_output() -> None:
    torch.manual_seed(11)
    encoder = FiveObservationEncoder(pre_lstm_size=32)
    observations = torch.randn(5, INPUT_COUNT) * 0.05
    agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    previous = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long)
    with torch.no_grad():
        before = encoder(observations, agent_ids, previous)[0]
    for name in (
        "horizon_head", "path_dynamics_head", "first_hit_head",
        "action_path_head", "action_outcome_head",
    ):
        delattr(encoder.pattern_tree, name)
    encoder.architecture_version = FIVE_OBSERVATION_ARCHITECTURE_VERSION - 1
    with torch.no_grad():
        after = encoder(observations, agent_ids, previous)[0]
    assert encoder.architecture_version == FIVE_OBSERVATION_ARCHITECTURE_VERSION
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_checkpoint_five_qua_cu_van_bi_tu_choi_ro_rang() -> None:
    encoder = FiveObservationEncoder(pre_lstm_size=32)
    encoder.architecture_version = FIVE_OBSERVATION_ARCHITECTURE_VERSION - 2
    with pytest.raises(ValueError, match="--fresh-system"):
        encoder(
            torch.zeros(5, INPUT_COUNT),
            torch.arange(FIVE_AGENT_COUNT, dtype=torch.long),
            torch.zeros(FIVE_AGENT_COUNT, dtype=torch.long),
        )


def test_nam_tac_nhan_co_the_nhan_nam_quan_sat_rieng() -> None:
    policy = _small_policy()
    observations = np.vstack([np.full(137, float(i), dtype=float) for i in range(5)])
    traces = policy.act_five_agents(observations)
    for index, trace in enumerate(traces):
        np.testing.assert_allclose(trace.raw_obs, float(index))


def test_he_dieu_phoi_tra_dung_nam_quyet_dinh() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    traces = system.act_nam_tac_nhan(np.zeros(137, dtype=float))
    assert len(traces) == 5
    assert list(system.experts) == ["CTX_CHUNG"]


def test_facade_delegates_codex_live_parameter_hash_to_shared_owner() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner.get_codex_teacher_live_parameter_hash = lambda: "inner-live-hash"
    assert system.get_codex_teacher_live_parameter_hash() == "inner-live-hash"


def test_reset_episode_can_preserve_incomplete_five_rollout() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner.five_rollout_steps = [{"sentinel": True}]
    system.reset_five_economic_reward(clear_rollout=False)
    assert owner.five_rollout_steps == [{"sentinel": True}]
    system.reset_five_economic_reward(clear_rollout=True)
    assert owner.five_rollout_steps == []


def test_five_ppo_reuses_exact_decision_action_mask(monkeypatch) -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    mask = np.asarray([True, True, False, False, False, False, False, False, False, False, False, False])
    policy._decision_context = {"allowed_action_mask": mask}
    trainer = policy._ensure_five_trainer()
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {"total_loss": 0.0}

    monkeypatch.setattr(trainer, "optimize_sequence", capture)
    expected = five_agent_authority_mask(mask)
    for _ in range(FIVE_LSTM_UNROLL + 1):
        traces = policy.act_five_agents(np.zeros(137, dtype=np.float32))
        assert all(
            tuple(trace.allowed_action_mask) == tuple(expected[agent])
            for agent, trace in enumerate(traces)
        )
        stats = policy.observe_five_transition(
            traces,
            np.zeros(5, dtype=np.float32),
            dones=np.zeros(5, dtype=np.float32),
        )
    assert stats["updated"] == 1.0
    assert captured["allowed_action_masks"].shape == (FIVE_LSTM_UNROLL, 5, 12)
    assert np.all(captured["allowed_action_masks"] == expected.reshape(1, 5, 12))
    assert np.asarray(captured["bootstrap_value"]).shape == (5,)
    assert len(policy.five_rollout_steps) == 1


def test_five_mt5_allowed_actions_mapping_duoc_mask_truoc_khi_lay_mau() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    policy._decision_context = {
        "allowed_actions": {0: True, 1: True, 2: True, 11: False},
    }
    expected = five_agent_authority_mask(np.asarray([
        True, True, True, False, False, False,
        False, False, False, False, False, False,
    ]))
    for _ in range(32):
        traces = policy.act_five_agents(np.zeros(137, dtype=np.float32))
        for agent, trace in enumerate(traces):
            assert trace.selected_idx in set(np.flatnonzero(expected[agent]))
            assert tuple(trace.allowed_action_mask) == tuple(expected[agent])


def test_five_mt5_allowed_action_mask_mapping_v13_khong_bi_bo_qua() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    policy.set_decision_context(allowed_actions={0: True})
    for _ in range(8):
        traces = policy.act_five_agents(np.zeros(137, dtype=np.float32))
        assert all(trace.selected_idx == int(Action.HOLD) for trace in traces)
        assert all(tuple(trace.allowed_action_mask) == (True,) + (False,) * 11 for trace in traces)


def test_phat_rieng_action_dong_khi_truoc_quyet_dinh_khong_co_vi_the() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    rows = list(policy.act_five_agents(np.zeros(137, dtype=np.float32)))
    rows[0] = replace(rows[0], action_details={
        **rows[0].action_details,
        "five_action_index": int(Action.CLOSE_ALL),
        "five_economic_actor_eligible": True,
    })
    zeros = np.zeros(5, dtype=np.float32)
    policy.observe_five_economic_transition(
        rows,
        agent_mark_to_market_r=zeros,
        transaction_cost_r=zeros,
        agent_risk_ratio=zeros,
        conflict_cost_r=zeros,
        policy_violation_penalty_r=np.asarray([1.0, 0, 0, 0, 0], dtype=np.float32),
        team_equity_r=0.0,
        team_drawdown=0.0,
    )
    rewards = np.asarray(policy.five_rollout_steps[-1]["rewards"], dtype=np.float32)
    assert rewards[0] == pytest.approx(-1.0)
    np.testing.assert_allclose(rewards[1:], 0.0, atol=1e-7)
    assert policy.five_last_training_stats == {}


def test_giu_co_vi_the_nhan_loi_nhuan_tang_con_giu_tai_khoan_trong_bi_loai() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    rows = list(policy.act_five_agents(np.zeros(137, dtype=np.float32)))
    for index, row in enumerate(rows):
        rows[index] = replace(row, action_details={
            **row.action_details,
            "five_action_index": int(Action.HOLD),
            "five_owned_position_before": index == 0,
            "five_reward_sample_eligible": index == 0,
        })
    zeros = np.zeros(5, dtype=np.float32)
    marked = zeros.copy()
    marked[0] = 0.5
    risk = zeros.copy()
    risk[0] = 1.0
    policy.observe_five_economic_transition(
        rows,
        agent_mark_to_market_r=marked,
        transaction_cost_r=zeros,
        agent_risk_ratio=risk,
        conflict_cost_r=zeros,
        team_equity_r=0.5,
        team_drawdown=0.0,
    )
    step = policy.five_rollout_steps[-1]
    assert step["sample_loss_masks"].tolist() == [True, False, False, False, False]
    assert step["policy_loss_masks"].tolist() == [True, False, False, False, False]
    assert float(step["rewards"][0]) > 0.0
    np.testing.assert_allclose(step["rewards"][1:], 0.0, atol=1e-7)


def test_giu_tai_khoan_trong_khong_vao_actor_critic_entropy_hay_reward_scale() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    trainer = policy._ensure_five_trainer()
    rows = list(policy.act_five_agents(np.zeros(137, dtype=np.float32)))
    for index, row in enumerate(rows):
        rows[index] = replace(row, action_details={
            **row.action_details,
            "five_action_index": int(Action.HOLD),
            "five_owned_position_before": False,
            "five_reward_sample_eligible": False,
        })
    zeros = np.zeros(5, dtype=np.float32)
    policy.observe_five_economic_transition(
        rows,
        agent_mark_to_market_r=zeros,
        transaction_cost_r=zeros,
        agent_risk_ratio=zeros,
        conflict_cost_r=zeros,
        team_equity_r=0.0,
        team_drawdown=0.0,
    )
    step = policy.five_rollout_steps[-1]
    assert not np.any(step["sample_loss_masks"])
    before = trainer.reward_scale.count
    trainer.reward_scale.update_and_normalize(
        np.zeros((1, 5), dtype=np.float32),
        sample_mask=np.zeros((1, 5), dtype=bool),
    )
    assert trainer.reward_scale.count == before


def test_five_ppo_excludes_arbitration_blocked_actor_rows(monkeypatch) -> None:
    from dataclasses import replace

    policy = _small_policy()
    policy._learning_enabled = True
    trainer = policy._ensure_five_trainer()
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {"total_loss": 0.0}

    monkeypatch.setattr(trainer, "optimize_sequence", capture)
    for _ in range(FIVE_LSTM_UNROLL + 1):
        rows = list(policy.act_five_agents(np.zeros(137, dtype=np.float32)))
        rows[2] = replace(rows[2], action_details={**rows[2].action_details, "five_actor_eligible": False})
        stats = policy.observe_five_transition(
            rows, np.zeros(5, dtype=np.float32), dones=np.zeros(5, dtype=np.float32)
        )
    assert stats["updated"] == 1.0
    assert captured["policy_loss_masks"].shape == (FIVE_LSTM_UNROLL, 5)
    assert not np.any(captured["policy_loss_masks"][:, 2])
    assert np.all(captured["policy_loss_masks"][:, [0, 1, 3, 4]])


def test_policy_loss_mask_removes_blocked_advantage_from_actor_gradient() -> None:
    from self_growing_trader.five_shared_policy import FivePPOTrainer

    net = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    trainer = FivePPOTrainer(net)
    new_log_prob = torch.tensor([[0.0, 0.0]], requires_grad=True)
    old_log_prob = torch.zeros_like(new_log_prob)
    advantage = torch.tensor([[1.0, -1000.0]])
    value = torch.zeros_like(new_log_prob, requires_grad=True)
    returns = torch.zeros_like(new_log_prob)
    entropy = torch.zeros_like(new_log_prob, requires_grad=True)
    loss, stats = trainer.loss_from_tensors(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantage=advantage,
        value=value,
        returns=returns,
        entropy=entropy,
        policy_loss_mask=torch.tensor([[True, False]]),
    )
    loss.backward()
    assert new_log_prob.grad[0, 0] != 0
    assert new_log_prob.grad[0, 1] == 0
    assert stats["actor_eligible_samples"] == 1.0


def test_sample_loss_mask_removes_value_and_entropy_gradients() -> None:
    from self_growing_trader.five_shared_policy import FivePPOTrainer

    trainer = FivePPOTrainer(SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8))
    new_log_prob = torch.zeros((1, 2), requires_grad=True)
    value = torch.tensor([[1.0, 1.0]], requires_grad=True)
    entropy = torch.tensor([[1.0, 1.0]], requires_grad=True)
    loss, stats = trainer.loss_from_tensors(
        new_log_prob=new_log_prob,
        old_log_prob=torch.zeros_like(new_log_prob),
        advantage=torch.ones_like(new_log_prob),
        value=value,
        returns=torch.zeros_like(value),
        entropy=entropy,
        policy_loss_mask=torch.ones((1, 2), dtype=torch.bool),
        sample_loss_mask=torch.tensor([[True, False]]),
    )
    loss.backward()
    assert value.grad[0, 0] != 0
    assert value.grad[0, 1] == 0
    assert entropy.grad[0, 0] != 0
    assert entropy.grad[0, 1] == 0
    assert stats["reward_sample_eligible_samples"] == 1.0


def test_chia_phan_thuong_tap_the_dung_cong_thuc() -> None:
    raw = np.asarray([1.0, 0.0, -1.0, 2.0, 3.0], dtype=np.float32)
    out = mix_team_spirit_rewards(raw, 1.0)
    np.testing.assert_allclose(out, np.mean(raw))
    out = mix_team_spirit_rewards(raw, 0.3)
    expected = 0.7 * raw + 0.3 * np.mean(raw)
    np.testing.assert_allclose(out, expected)


def test_chia_phan_thuong_tap_the_chi_tinh_agent_dang_hoc() -> None:
    raw = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    active = np.asarray([True, False, False, False, False])
    out = mix_team_spirit_rewards(raw, 1.0, sample_mask=active)
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0, 1.0])


def test_reward_settlement_net_r_uu_tien_net_r_khi_dong_lenh() -> None:
    shaper = FiveTradingRewardShaper(
        competition_weight=0.0,
        reward_mode="settlement_net_r_v1",
        settlement_weight=1.0,
        mark_to_market_weight=0.10,
    )
    open_state = FiveEconomicState(
        agent_mark_to_market_r=np.asarray([0.8, 0.0, 0.0, 0.0, 0.0]),
        transaction_cost_r=np.zeros(5), agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5), team_equity_r=0.8, team_drawdown=0.0,
    )
    opened = shaper.shape(open_state)
    assert opened.profit_basis_r[0] == pytest.approx(0.08)
    close_state = replace(
        open_state,
        agent_mark_to_market_r=np.asarray([0.8, 0.0, 0.0, 0.0, 0.0]),
        agent_closed_trade_count=np.asarray([1, 0, 0, 0, 0]),
        agent_winning_trade_count=np.asarray([1, 0, 0, 0, 0]),
        agent_realized_trade_r=np.asarray([0.8, 0.0, 0.0, 0.0, 0.0]),
    )
    closed = shaper.shape(close_state)
    assert closed.settled_trade_delta_r[0] == pytest.approx(0.8)
    assert closed.profit_basis_r[0] == pytest.approx(0.8)
    assert closed.raw_rewards[0] == pytest.approx(0.8)


def test_thong_so_huan_luyen_cong_khai() -> None:
    schedule = FiveTrainingSchedule(progress=0.0)
    assert schedule.profile.ppo_clip == 0.2
    assert schedule.profile.gae_lambda == 0.95
    assert schedule.profile.adam_beta1 == 0.9
    assert schedule.profile.adam_beta2 == 0.999
    assert schedule.team_spirit == 0.3
    schedule.progress = 1.0
    assert schedule.team_spirit == 1.0
    assert np.isclose(schedule.entropy_coefficient, 0.001)
    assert np.isclose(schedule.learning_rate, 1e-6)
    assert schedule.gae_horizon_seconds == 840.0
    assert abs(discount_from_horizon(0.133, 180.0) - 0.9992611111111112) < 1e-12
    assert schedule.gae_min_feedback_steps == 16
    effective = schedule.effective_gae_horizon_seconds(1200.0)
    assert effective == pytest.approx(19200.0)
    assert discount_from_horizon(1200.0, effective) == pytest.approx(0.9375)


def test_entropy_five_chi_giam_sau_khi_du_edge_duong() -> None:
    schedule = FiveTrainingSchedule(
        progress=1.0,
        entropy_edge_gate_enabled=True,
        entropy_edge_release_count=3,
        entropy_edge_decay_count=8,
    )
    schedule.verified_edge_count = 0
    assert schedule.entropy_coefficient == pytest.approx(0.06)
    assert schedule.flat_entry_mass_floor == pytest.approx(0.75)
    schedule.verified_edge_count = 2
    assert schedule.entropy_coefficient == pytest.approx(0.06)
    schedule.verified_edge_count = 3
    assert schedule.entropy_coefficient == pytest.approx(0.06)
    assert schedule.flat_entry_mass_floor == pytest.approx(0.75)
    schedule.verified_edge_count = 5
    assert schedule.entropy_coefficient == pytest.approx(0.0364)
    assert schedule.flat_entry_mass_floor == pytest.approx(0.45)
    schedule.verified_edge_count = 8
    assert schedule.entropy_coefficient == pytest.approx(0.001)
    assert schedule.flat_entry_mass_floor == pytest.approx(0.0)


def test_directional_entry_floor_tracks_each_owner() -> None:
    schedule = FiveTrainingSchedule(
        progress=1.0,
        entropy_edge_gate_enabled=True,
        entropy_edge_release_count=3,
        entropy_edge_decay_count=8,
        flat_entry_direction_mass_floor_start=0.25,
        flat_entry_direction_mass_floor_end=0.10,
    )
    schedule.verified_edge_count = 0
    assert schedule.flat_entry_direction_mass_floor == pytest.approx(0.25)
    schedule.verified_edge_count = 5
    assert schedule.flat_entry_direction_mass_floor == pytest.approx(0.19)
    schedule.verified_edge_count = 8
    assert schedule.flat_entry_direction_mass_floor == pytest.approx(0.10)


def test_negative_expectancy_khong_lam_giam_kham_pha_toan_cuc() -> None:
    schedule = FiveTrainingSchedule(
        progress=0.0,
        entropy_edge_gate_enabled=True,
        entropy_edge_release_count=3,
        entropy_edge_decay_count=8,
    )
    schedule.verified_edge_count = 0
    schedule.policy_expectancy_eligible = True
    schedule.policy_expectancy_lcb = -0.16
    schedule.negative_critical_scope_count = 12
    assert schedule.negative_evidence_strength == pytest.approx(0.8)
    # Entropy vẫn độc lập để khám phá, nhưng không được ép policy thua lỗ giữ
    # nguyên 75% entry mass trên mọi trạng thái tài khoản trống.
    assert schedule.entropy_coefficient == pytest.approx(0.06)
    assert schedule.flat_entry_mass_floor == pytest.approx(0.15)
    schedule.negative_critical_scope_count = 0
    assert schedule.negative_evidence_strength == pytest.approx(0.0)
    assert schedule.flat_entry_mass_floor == pytest.approx(0.75)


def test_five_behavior_uniform_mix_doc_lap_voi_expectancy_am(monkeypatch) -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    policy.five_behavior_uniform_mix = 0.15
    policy.five_behavior_entry_probability_floor = 0.005
    monkeypatch.setattr(policy, "contextual_expectancy_report", lambda: {
        "positive_eligible_scope_count": 0,
        "negative_critical_scope_count": 6,
        "policy_expectancy": {
            "eligible": True,
            "sample_count": 256,
            "expectancy_lcb": -0.20,
        },
    })
    policy._account_context = {
        # Cố tình đặt rủi ro cao: rủi ro phải đi qua sizing/exposure, không
        # được làm giảm sampler khám phá toàn cục.
        "current_equity_drawdown": 0.50,
        "open_risk_ratio": 2.0,
    }
    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))
    for trace in traces:
        assert trace.action_details["five_behavior_negative_evidence_strength"] == pytest.approx(1.0)
        assert trace.action_details["five_behavior_uniform_mix_configured"] == pytest.approx(0.15)
        assert trace.action_details["five_behavior_uniform_mix"] == pytest.approx(0.15)
        assert trace.action_details["five_behavior_entry_probability_floor"] == pytest.approx(0.005)
        assert trace.action_details["five_behavior_exploration_independent"] is True
        assert "five_behavior_account_risk_strength" not in trace.action_details
        assert "five_behavior_exploration_attenuation" not in trace.action_details


def test_five_live_owner_exploration_chi_giam_phia_am_trong_cua_so_hien_tai(monkeypatch) -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    policy.five_behavior_uniform_mix = 0.15
    policy.five_live_owner_exploration_enabled = True
    policy.five_live_owner_exploration_min_samples = 12
    policy.five_live_owner_exploration_min_confidence = 0.20
    policy.five_live_owner_exploration_min_lcb_r = -0.15
    policy.five_live_owner_exploration_min_mix = 0.02
    policy._five_live_owner_outcomes = {0: [-1.0] * 12, 1: [0.50] * 12}
    captured = {}
    original = policy.five_shared_network.forward_five

    def capture(*args, **kwargs):
        captured["behavior_uniform_mix"] = np.asarray(
            kwargs["behavior_uniform_mix"], dtype=float
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(policy.five_shared_network, "forward_five", capture)
    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))

    mixes = captured["behavior_uniform_mix"]
    assert mixes.shape == (FIVE_AGENT_COUNT,)
    assert mixes[0] == pytest.approx(0.02)
    assert mixes[1] > mixes[0]
    assert traces[0].action_details["five_live_owner_exploration_negative_owners"] == [0]
    assert traces[0].action_details["five_live_owner_exploration_samples"] == [12, 12]
    assert traces[0].action_details["five_behavior_uniform_mix_owner_effective"] == pytest.approx(0.02)
    assert traces[1].action_details["five_behavior_uniform_mix_owner_effective"] == pytest.approx(float(mixes[1]))


def test_frozen_behavior_sampling_draws_seeded_actions_without_trainer(monkeypatch) -> None:
    policy = _small_policy()
    policy._learning_enabled = False
    policy.five_frozen_behavior_sampling_enabled = True
    policy.five_behavior_uniform_mix = 0.15
    policy.five_behavior_entry_probability_floor = 0.005
    policy.five_ppo_trainer = None
    captured = {}
    original = policy.five_shared_network.forward_five

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(policy.five_shared_network, "forward_five", capture)
    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))

    assert captured["deterministic"] is False
    assert captured["behavior_uniform_mix"] == pytest.approx(0.15)
    assert captured["behavior_entry_probability_floor"] == pytest.approx(0.005)
    assert policy.five_ppo_trainer is None
    for trace in traces:
        assert trace.action_details["five_frozen_behavior_sampling_enabled"] is True
        assert trace.action_details["five_optimizer_learning_enabled"] is False


def test_frozen_behavior_sampling_uses_private_reproducible_rng() -> None:
    first = _small_policy()
    second = _small_policy()
    second.five_shared_network.load_state_dict(first.five_shared_network.state_dict())
    for policy in (first, second):
        policy._learning_enabled = False
        policy.five_frozen_behavior_sampling_enabled = True
        policy.five_frozen_behavior_sampling_seed = 99173
        policy._five_frozen_behavior_sampling_generator = None

    torch.manual_seed(1)
    first_packets = [
        tuple(trace.action_details["five_action_index"] for trace in first.act_five_agents(
            np.full(INPUT_COUNT, step / 10.0, dtype=np.float32)
        ))
        for step in range(8)
    ]
    torch.manual_seed(987654321)
    second_packets = [
        tuple(trace.action_details["five_action_index"] for trace in second.act_five_agents(
            np.full(INPUT_COUNT, step / 10.0, dtype=np.float32)
        ))
        for step in range(8)
    ]

    assert first_packets == second_packets


def test_imitation_v2_warmup_hien_ro_va_giam_exploration_tren_trace() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    policy.five_behavior_uniform_mix = 0.15
    policy.five_behavior_entry_probability_floor = 0.005
    calibrator = policy.five_shared_network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator.teacher_imitation_trained = True
    policy.five_ppo_trainer = None

    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))

    for trace in traces:
        assert trace.action_details["five_imitation_prior_enabled"] is True
        assert trace.action_details["five_imitation_warmup_active"] is True
        assert trace.action_details["five_imitation_pathway_frozen"] is True
        assert trace.action_details["five_imitation_representation_frozen"] is True
        assert trace.action_details["five_behavior_uniform_mix"] == pytest.approx(0.05)


def test_checkpoint_cu_xoa_scalar_initial_sl_ma_khong_doi_tham_so() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    policy.five_lifecycle_initial_sl_penalty_r = 0.75
    before = policy.five_parameter_count()
    trainer = policy._ensure_five_trainer()
    assert not hasattr(policy, "five_lifecycle_initial_sl_penalty_r")
    assert policy.five_parameter_count() == before
    assert trainer.schedule.flat_entry_mass_floor == pytest.approx(0.75)


def test_san_entry_chi_ap_dung_khi_toan_tai_khoan_khong_co_vi_the() -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    rows = []
    for trace in policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32)):
        details = dict(trace.action_details)
        details["five_positions_before"] = 1
        details["five_owned_position_before"] = False
        details["five_reward_sample_eligible"] = True
        details["five_economic_actor_eligible"] = True
        rows.append(replace(trace, action_details=details))
    policy.observe_five_transition(tuple(rows), np.zeros(5, dtype=np.float32))
    assert not np.any(policy.five_rollout_steps[-1]["flat_account_masks"])


def test_timeout_pending_phat_dung_trace_da_dat_lenh_va_chi_mot_lan(monkeypatch) -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    rows = list(policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32)))
    for index, row in enumerate(rows):
        details = dict(row.action_details)
        details["five_action_index"] = int(Action.SELL_LIMIT if index == 2 else Action.HOLD)
        details["five_economic_actor_eligible"] = index == 2
        rows[index] = replace(row, action_details=details)
    registered = policy.register_five_pending_order_packet(rows)
    assert registered["registered"] == 1

    captured = {}

    def observe(traces, rewards, *, auxiliary_targets=None, credit_channel="pattern"):
        captured["traces"] = tuple(traces)
        captured["rewards"] = np.asarray(rewards)
        captured["channel"] = credit_channel
        return {"updated": 0.0, "pattern_rollout_steps": 1.0}

    monkeypatch.setattr(policy, "observe_five_pattern_transition", observe)
    target_trace = rows[2].trace_id
    result = policy.apply_five_unfilled_pending_timeout_penalty(target_trace, 0.25)
    assert result["accepted"] is True
    assert captured["channel"] == "pending_timeout"
    np.testing.assert_allclose(captured["rewards"], [0, 0, -0.25, 0, 0])
    assert [
        bool(item.action_details["five_pending_timeout_actor_eligible"])
        for item in captured["traces"]
    ] == [False, False, True, False, False]
    duplicate = policy.apply_five_unfilled_pending_timeout_penalty(target_trace, 0.25)
    assert duplicate["reason"] == "duplicate_trace"


def test_policy_five_doc_so_edge_tu_so_thong_ke_vong_doi(monkeypatch) -> None:
    policy = _small_policy()
    policy._learning_enabled = True
    policy.five_entropy_edge_gate_enabled = True
    policy.five_entropy_edge_release_count = 3
    policy.five_entropy_edge_decay_count = 8
    monkeypatch.setattr(
        policy,
        "contextual_expectancy_report",
        lambda: {"positive_eligible_scope_count": 5},
    )
    trainer = policy._ensure_five_trainer()
    assert trainer.schedule.verified_edge_count == 5
    assert trainer.schedule.entropy_coefficient == pytest.approx(0.0364)


def test_lo_phat_mot_chieu_khong_bi_tru_trung_binh_lam_mat_gradient() -> None:
    from self_growing_trader.five_shared_policy import FivePPOTrainer, FivePublicProfile

    net = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    trainer = FivePPOTrainer(
        net,
        schedule=FiveTrainingSchedule(
            profile=FivePublicProfile(entropy_start=0.0, entropy_end=0.0)
        ),
    )
    new_log_prob = torch.zeros(16, requires_grad=True)
    old_log_prob = torch.zeros(16)
    negative = -torch.ones(16)
    zeros = torch.zeros(16)
    loss, stats = trainer.loss_from_tensors(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantage=negative,
        value=zeros,
        returns=zeros,
        entropy=zeros,
        policy_loss_mask=torch.ones(16, dtype=torch.bool),
        value_loss_weight_override=0.0,
        center_advantages=False,
    )
    loss.backward()
    assert stats["advantage_centered"] == 0.0
    assert stats["normalized_actor_advantage_mean"] == pytest.approx(-1.0)
    assert bool(torch.all(new_log_prob.grad > 0.0))


def test_cap_nhat_16_buoc_thay_doi_tham_so() -> None:
    from self_growing_trader.five_shared_policy import FivePPOTrainer

    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    memories = net.initial_memories()
    initial = [FiveAgentMemory(m.hidden.clone(), m.cell.clone(), m.previous_action.clone()) for m in memories]
    observations = []
    actions = []
    targets = []
    delays = []
    risks = []
    offsets = []
    manage_modes = []
    close_fractions = []
    old_lp = []
    values = []
    with torch.no_grad():
        for step in range(16):
            obs = torch.randn(5, 137) * 0.1
            observations.append(obs.cpu().numpy())
            results, memories, _ = net.forward_five(obs, memories)
            actions.append([r.action_index for r in results])
            targets.append([r.target_index for r in results])
            delays.append([r.delay_index for r in results])
            risks.append([r.risk_index for r in results])
            offsets.append([r.entry_offset_index for r in results])
            manage_modes.append([r.manage_mode_index for r in results])
            close_fractions.append([r.close_fraction_index for r in results])
            old_lp.append([r.log_probability for r in results])
            values.append([r.value for r in results])
    before = next(net.parameters()).detach().clone()
    trainer = FivePPOTrainer(net)
    stats = trainer.optimize_sequence(
        observations=np.asarray(observations, dtype=np.float32),
        initial_memories=initial,
        actions=np.asarray(actions),
        targets=np.asarray(targets),
        delay_indices=np.asarray(delays),
        risk_indices=np.asarray(risks),
        entry_offset_indices=np.asarray(offsets),
        manage_mode_indices=np.asarray(manage_modes),
        close_fraction_indices=np.asarray(close_fractions),
        old_log_probabilities=np.asarray(old_lp, dtype=np.float32),
        rewards=np.random.default_rng(1).normal(size=(16, 5)).astype(np.float32),
        dones=np.zeros((16, 5), dtype=np.float32),
        step_seconds=1.0,
        horizon_seconds=180.0,
    )
    after = next(net.parameters()).detach()
    assert stats["total_loss"] == stats["total_loss"]
    assert stats["entropy"] == pytest.approx(stats["policy_action_entropy"])
    assert not torch.equal(before, after)


def test_qua_khu_khac_nhau_lam_trang_thai_hien_tai_khac_nhau() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    zeros = net.initial_memories()
    lich_su_a = torch.zeros(4, 5, 137)
    lich_su_b = torch.zeros(4, 5, 137)
    lich_su_b[:, :, 0] = 1.0
    nho_a = net.advance_history(lich_su_a, zeros)
    nho_b = net.advance_history(lich_su_b, net.initial_memories())
    hien_tai = torch.zeros(5, 137)
    with torch.no_grad():
        ket_qua_a, _, _ = net.forward_five(hien_tai, nho_a, deterministic=True)
        ket_qua_b, _, _ = net.forward_five(hien_tai, nho_b, deterministic=True)
    assert not np.allclose(ket_qua_a[0].hidden_after, ket_qua_b[0].hidden_after)
    assert not np.allclose(ket_qua_a[0].cell_after, ket_qua_b[0].cell_after)


def test_co_the_nap_lai_lich_su_de_dung_ky_uc_truoc_khi_ra_quyet_dinh() -> None:
    policy = _small_policy()
    lich_su = np.zeros((3, 137), dtype=np.float32)
    lich_su[:, 0] = np.asarray([0.1, 0.4, -0.2], dtype=np.float32)
    policy.prime_five_history(lich_su)
    assert policy.five_agent_history_steps == [3, 3, 3, 3, 3]
    assert any(float(torch.linalg.vector_norm(m.hidden).item()) > 0.0 for m in policy.five_agent_memories)
    traces = policy.act_five_agents(np.zeros(137, dtype=np.float32))
    assert all(int(t.action_details["five_history_steps_before"]) == 3 for t in traces)
    assert policy.five_agent_history_steps == [4, 4, 4, 4, 4]
    policy.reset_runtime_state()
    assert policy.five_agent_history_steps == [0, 0, 0, 0, 0]
    assert all(float(torch.linalg.vector_norm(m.hidden).item()) == 0.0 for m in policy.five_agent_memories)
    assert all(float(torch.linalg.vector_norm(m.cell).item()) == 0.0 for m in policy.five_agent_memories)


def test_ket_thuc_chi_xoa_ky_uc_cua_tac_nhan_da_ket_thuc() -> None:
    policy = _small_policy()
    traces = policy.act_five_agents(np.ones(137, dtype=np.float32))
    before = [m.hidden.clone() for m in policy.five_agent_memories]
    assert all(float(torch.linalg.vector_norm(value).item()) > 0.0 for value in before)
    stats = policy.observe_five_transition(
        traces,
        np.zeros(5, dtype=np.float32),
        dones=np.asarray([1, 0, 0, 0, 0], dtype=np.float32),
    )
    assert stats["updated"] == 0.0
    assert float(torch.linalg.vector_norm(policy.five_agent_memories[0].hidden).item()) == 0.0
    assert float(torch.linalg.vector_norm(policy.five_agent_memories[0].cell).item()) == 0.0
    assert policy.five_agent_history_steps[0] == 0
    for i in range(1, 5):
        assert torch.equal(policy.five_agent_memories[i].hidden, before[i])
        assert policy.five_agent_history_steps[i] == 1


def test_doan_hoc_cat_ky_uc_tai_ranh_gioi_ket_thuc() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=24, hidden_size=12)
    observations = torch.randn(2, 5, 137) * 0.1
    actions = torch.zeros(2, 5, dtype=torch.long)
    targets = torch.zeros(2, 5, dtype=torch.long)
    delays = torch.zeros(2, 5, dtype=torch.long)
    risks = torch.zeros(2, 5, dtype=torch.long)
    offsets = torch.zeros(2, 5, dtype=torch.long)
    manage_modes = torch.zeros(2, 5, dtype=torch.long)
    close_fractions = torch.zeros(2, 5, dtype=torch.long)
    dones = torch.zeros(2, 5, dtype=torch.bool)
    dones[0, 0] = True
    initial = net.initial_memories()
    evaluated = net.evaluate_sequence(
        observations,
        initial,
        actions=actions,
        targets=targets,
        delay_indices=delays,
        risk_indices=risks,
        entry_offset_indices=offsets,
        manage_mode_indices=manage_modes,
        close_fraction_indices=close_fractions,
        episode_dones=dones,
    )
    direct = net.advance_history(observations[1:2], net.initial_memories())
    torch.testing.assert_close(evaluated["hidden"][1, 0], direct[0].hidden)
    torch.testing.assert_close(evaluated["cell"][1, 0], direct[0].cell)
    # Tac nhan 2 khong ket thuc, nen buoc thu hai van mang ky uc tu buoc dau.
    assert not torch.allclose(evaluated["hidden"][1, 1], direct[1].hidden)


def test_he_dieu_phoi_co_the_nap_chuoi_qua_khu_vao_nam_tac_nhan() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    lich_su = np.zeros((5, 137), dtype=np.float32)
    lich_su[:, 0] = np.linspace(-0.5, 0.5, 5)
    system.nap_lich_su_nam_tac_nhan(lich_su)
    policy = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    assert policy.five_agent_history_steps == [5, 5, 5, 5, 5]
    traces = system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32))
    assert all(int(t.action_details["five_history_steps_before"]) == 5 for t in traces)



def test_nhanh_do_tre_co_dung_bon_lua_chon_0_den_3() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    assert net.heads.parameter_heads["delay"].out_features == 4
    obs = torch.zeros(5, 137)
    results, _, _ = net.forward_five(obs, net.initial_memories(), deterministic=True)
    assert all(0 <= int(item.delay_index) <= 3 for item in results)
    assert all(int(item.action_index) not in {8, 9} for item in results)


def test_ba_head_hinh_hoc_lenh_ton_tai_va_checkpoint_cu_duoc_di_chuyen_trung_tinh() -> None:
    from self_growing_trader.execution_contract import (
        DEFAULT_SL_BIN_INDEX,
        DEFAULT_TP_BIN_INDEX,
        DEFAULT_TRAILING_BIN_INDEX,
    )

    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    for name in ("sl", "tp", "trailing"):
        del net.heads.parameter_heads[name]

    assert net.require_current_architecture()
    assert set(("sl", "tp", "trailing")) <= set(net.heads.parameter_heads)
    obs = torch.zeros(5, 137)
    results, _, _ = net.forward_five(obs, net.initial_memories(), deterministic=True)
    assert all(item.sl_index == DEFAULT_SL_BIN_INDEX for item in results)
    assert all(item.tp_index == DEFAULT_TP_BIN_INDEX for item in results)
    assert all(item.trailing_index == DEFAULT_TRAILING_BIN_INDEX for item in results)


def test_ba_head_hinh_hoc_nhan_xac_suat_ppo_khi_vao_lenh() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    allowed = torch.zeros((5, 12), dtype=torch.bool)
    allowed[:, 1] = True
    results, _, _ = net.forward_five(
        torch.zeros(5, 137), net.initial_memories(),
        allowed_action_masks=allowed, deterministic=False,
    )
    for item in results:
        assert item.action_index == 1
        assert {"sl", "tp", "trailing"} <= set(item.head_log_probabilities)
        assert {"sl", "tp", "trailing"} <= set(item.head_entropies)


def test_bay_lua_chon_dung_ky_vong_sau_chi_phi_lam_bien_vao_lenh() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    with torch.no_grad():
        head = net.observation_encoder.pattern_tree.action_path_head
        head.weight.zero_()
        head.bias.zero_()
        # action_path stores normalized net-R at component 2 of each action.
        head.bias[2] = torch.atanh(torch.tensor(0.20 / 5.0))
        head.bias[7] = torch.atanh(torch.tensor(0.25 / 5.0))
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = 0.10
    net.heads.seven_choice_expected_value_logit_scale = 10.0
    allowed = torch.zeros((5, 12), dtype=torch.bool)
    allowed[0, [0, 1, 3, 5]] = True
    allowed[1, [0, 2, 4, 6]] = True
    allowed[2:, 0] = True

    results, _, tensors = net.forward_five(
        torch.zeros(5, 137), net.initial_memories(),
        allowed_action_masks=allowed, deterministic=True,
    )

    assert [item.action_index for item in results] == [1, 2, 0, 0, 0]
    assert tensors["action_logits"][0, 0].item() == 0.0
    assert tensors["action_logits"][0, 1].item() > 0.0
    assert results[0].pattern_predictions["entry_1_expected_net_r"] > 0.19
    assert results[0].pattern_predictions[
        "seven_choice_expected_value_decision_enabled"
    ] is True

    net.heads.seven_choice_expected_value_threshold_r = 0.30
    held, _, _ = net.forward_five(
        torch.zeros(5, 137), net.initial_memories(),
        allowed_action_masks=allowed, deterministic=True,
    )
    assert [item.action_index for item in held] == [0, 0, 0, 0, 0]


def test_bay_lua_chon_co_the_nhan_gia_tri_voi_xac_suat_thang() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    with torch.no_grad():
        head = net.observation_encoder.pattern_tree.action_path_head
        head.weight.zero_()
        head.bias.zero_()
        # MARKET_BUY: high value but low win probability.
        head.bias[1] = torch.logit(torch.tensor(0.20))
        head.bias[2] = torch.atanh(torch.tensor(0.40 / 5.0))
        # BUY_LIMIT: lower value but high win probability.
        head.bias[11] = torch.logit(torch.tensor(0.90))
        head.bias[12] = torch.atanh(torch.tensor(0.30 / 5.0))
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = 0.15
    net.heads.seven_choice_expected_value_logit_scale = 10.0
    net.heads.seven_choice_expected_value_score_mode = "value_times_win"
    allowed = torch.zeros((5, 12), dtype=torch.bool)
    allowed[0, [0, 1, 3, 5]] = True
    allowed[1, [0, 2, 4, 6]] = True
    allowed[2:, 0] = True

    results, _, _ = net.forward_five(
        torch.zeros(5, 137), net.initial_memories(),
        allowed_action_masks=allowed, deterministic=True,
    )

    assert results[0].action_index == 3


def test_bay_lua_chon_gia_tri_sau_khi_khop_chan_lenh_khop_thap() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = 0.0
    net.heads.seven_choice_expected_value_logit_scale = 10.0
    net.heads.seven_choice_expected_value_score_mode = "filled_value"
    path = torch.zeros((1, 6, 5))
    # MARKET_BUY has the larger conditional value but almost never fills;
    # BUY_LIMIT has the smaller value and a valid fill probability.
    path[0, 0, 0] = 0.05
    path[0, 0, 2] = 0.40 / 5.0
    path[0, 2, 0] = 0.80
    path[0, 2, 2] = 0.25 / 5.0
    adjusted = net._apply_seven_choice_expected_value_decision(
        torch.zeros((1, 12)), torch.tensor([0]), path,
    )
    assert adjusted[0, 3] > adjusted[0, 1]


def test_bay_lua_chon_gia_tri_sau_khi_khop_xep_theo_loi_nhuan_dieu_kien() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = 0.0
    net.heads.seven_choice_expected_value_logit_scale = 10.0
    net.heads.seven_choice_expected_value_score_mode = "filled_value"
    net.heads.seven_choice_expected_value_fill_probability_floor = 0.10
    path = torch.zeros((1, 6, 5))
    # Conditional profit of MARKET_BUY is lower than BUY_LIMIT, but its
    # fill probability is lower.  The selector must rank conditional profit,
    # rather than divide by fill probability and prefer the rare fill.
    path[0, 0, 0] = 0.20
    path[0, 0, 2] = 0.29 / 5.0
    path[0, 2, 0] = 0.80
    path[0, 2, 2] = 0.30 / 5.0
    adjusted = net._apply_seven_choice_expected_value_decision(
        torch.zeros((1, 12)), torch.tensor([0]), path,
    )
    assert adjusted[0, 3] > adjusted[0, 1]


def test_bay_lua_chon_bo_quyet_dinh_khi_hai_diem_qua_sat_nhau() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = 0.10
    net.heads.seven_choice_expected_value_logit_scale = 10.0
    net.heads.seven_choice_expected_value_min_score_margin = 0.01
    path = torch.zeros((1, 6, 5))
    # Hai lua chon mua cung vuot nguong nhung chenhlech chi 0,005R.
    path[0, 0, 2] = 0.30 / 5.0
    path[0, 2, 2] = 0.295 / 5.0
    adjusted = net._apply_seven_choice_expected_value_decision(
        torch.zeros((1, 12)), torch.tensor([0]), path,
    )
    # Logit HOLD = 0; khi khong du khoang cach, moi lenh vao bi khoa.
    assert adjusted[0, 0].item() == 0.0
    assert adjusted[0, 1].item() < -1e5
    assert adjusted[0, 3].item() < -1e5


def test_dau_hanh_dong_da_hoc_duoc_truyen_vao_diem_gia_tri() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = 0.10
    net.heads.seven_choice_expected_value_logit_scale = 10.0
    net.heads.seven_choice_expected_value_action_residual_scale = 1.0
    action_logits = torch.zeros((1, 12))
    action_logits[0, 3] = 10.0
    path = torch.zeros((1, 6, 5))
    path[0, 0, 2] = 0.12 / 5.0
    path[0, 2, 2] = 0.12 / 5.0

    adjusted = net._apply_seven_choice_expected_value_decision(
        action_logits, torch.tensor([0]), path,
    )

    assert adjusted[0, 3] > adjusted[0, 1]

    ineligible_path = path.clone()
    ineligible_path[0, 0, 2] = 0.05 / 5.0
    ineligible_logits = torch.zeros((1, 12))
    ineligible_logits[0, 1] = 100.0
    blocked = net._apply_seven_choice_expected_value_decision(
        ineligible_logits, torch.tensor([0]), ineligible_path,
    )
    assert blocked[0, 1] < blocked[0, 0]


def test_hieu_chinh_gia_tri_trang_thai_an_chi_tac_dong_khi_bat() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = 0.0
    net.heads.seven_choice_expected_value_logit_scale = 10.0
    path = torch.zeros((1, 6, 5))
    path[0, 0, 2] = 0.10 / 5.0
    path[0, 2, 2] = 0.10 / 5.0
    hidden = torch.zeros((1, 16))
    with torch.no_grad():
        net.heads.expected_value_hidden_residual[-1].bias.zero_()
        # Output slot 2 corresponds to BUY_LIMIT (action index 3).
        net.heads.expected_value_hidden_residual[-1].bias[2] = 1.0

    reference = net._apply_seven_choice_expected_value_decision(
        torch.zeros((1, 12)), torch.tensor([0]), path,
    )
    disabled = net._apply_seven_choice_expected_value_decision(
        torch.zeros((1, 12)), torch.tensor([0]), path, policy_hidden=hidden,
    )
    net.heads.seven_choice_expected_value_hidden_residual_enabled = True
    net.heads.seven_choice_expected_value_hidden_residual_scale_r = 0.25
    enabled = net._apply_seven_choice_expected_value_decision(
        torch.zeros((1, 12)), torch.tensor([0]), path, policy_hidden=hidden,
    )

    assert enabled[0, 3] > disabled[0, 3]
    assert torch.equal(disabled, reference)


def test_hieu_chinh_gia_tri_trang_thai_an_bat_buoc_co_hidden() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_hidden_residual_enabled = True
    net.heads.seven_choice_expected_value_hidden_residual_scale_r = 0.25
    with pytest.raises(ValueError, match="thieu policy_hidden"):
        net._apply_seven_choice_expected_value_decision(
            torch.zeros((1, 12)),
            torch.tensor([0]),
            torch.zeros((1, 6, 5)),
        )


def test_bay_lua_chon_ep_hinh_hoc_co_dinh_khop_nhan() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_fixed_geometry_enabled = True
    net.heads.seven_choice_fixed_delay_index = 0
    net.heads.seven_choice_fixed_entry_offset_index = 7
    net.heads.seven_choice_fixed_sl_index = 2
    net.heads.seven_choice_fixed_tp_index = 5
    net.heads.seven_choice_fixed_trailing_index = 0
    net.heads.seven_choice_expected_value_decision_enabled = True
    net.heads.seven_choice_expected_value_threshold_r = -1.0
    allowed = torch.zeros((5, 12), dtype=torch.bool)
    allowed[0, 1] = True
    allowed[1, 2] = True
    allowed[2:, 0] = True
    results, _, _ = net.forward_five(
        torch.zeros(5, 137), net.initial_memories(),
        allowed_action_masks=allowed, deterministic=True,
    )
    for item in results[:2]:
        assert item.delay_index == 0
        assert item.entry_offset_index == 7
        assert item.sl_index == 2
        assert item.tp_index == 5
        assert item.trailing_index == 0
        assert item.output_vector[14] == 7.0


def test_nhanh_bay_lua_chon_giu_vi_the_den_het_han_co_dinh() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    net.heads.seven_choice_fixed_geometry_enabled = True
    allowed = torch.zeros((5, 12), dtype=torch.bool)
    allowed[0, 1] = True
    allowed[1, 2] = True
    allowed[2, 11] = True
    allowed[3, 7] = True
    allowed[4, 10] = True
    results, _, _ = net.forward_five(
        torch.zeros(5, 137), net.initial_memories(),
        allowed_action_masks=allowed, deterministic=True,
    )
    assert [item.action_index for item in results[2:]] == [0, 0, 0]


def test_san_kham_pha_entry_khong_cho_action_bao_hoa_ve_hold() -> None:
    logits = torch.full((5, 12), -100.0)
    logits[:, 0] = 100.0
    allowed = torch.ones_like(logits, dtype=torch.bool)
    allowed[:, 8:10] = False
    _, _, entropy, probabilities = SharedFiveTradingPolicy._primary_action_distribution(
        logits,
        allowed=allowed,
        deterministic=False,
        entry_probability_floor=0.005,
    )
    torch.testing.assert_close(
        probabilities[:, 1:7],
        torch.full((5, 6), 0.005),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(5), rtol=0.0, atol=1e-7
    )
    assert torch.all(entropy > 0.0)
    assert torch.all(probabilities[:, 8:10] == 0.0)


def test_san_kham_pha_entry_floor_candidate_khong_phai_mask() -> None:
    logits = torch.full((5, 12), -100.0)
    logits[:, 0] = 100.0
    allowed = torch.ones_like(logits, dtype=torch.bool)
    _, _, _, probabilities = SharedFiveTradingPolicy._primary_action_distribution(
        logits,
        allowed=allowed,
        deterministic=False,
        entry_probability_floor=0.10,
    )
    # Six entries each receive the configured behavior floor while HOLD and
    # management remain legal; this is exploration pressure, not EA masking.
    torch.testing.assert_close(
        probabilities[:, 1:7],
        torch.full((5, 6), 0.10),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(5), rtol=0.0, atol=1e-7
    )
    assert torch.all(probabilities[:, 0] > 0.0)


def test_san_kham_pha_khong_thay_doi_quyet_dinh_deterministic() -> None:
    logits = torch.full((5, 12), -100.0)
    logits[:, 0] = 100.0
    allowed = torch.ones_like(logits, dtype=torch.bool)
    choices, _, _, probabilities = SharedFiveTradingPolicy._primary_action_distribution(
        logits,
        allowed=allowed,
        deterministic=True,
        entry_probability_floor=0.005,
    )
    assert torch.equal(choices, torch.zeros(5, dtype=torch.long))
    assert torch.all(probabilities[:, 1:7] < 1e-20)


def test_ppo_danh_gia_lai_dung_xac_suat_hanh_vi_da_tron() -> None:
    torch.manual_seed(91)
    net = SharedFiveTradingPolicy(pre_lstm_size=24, hidden_size=12)
    observations = torch.randn(1, 5, 137) * 0.05
    initial = net.initial_memories()
    results, _, tensors = net.forward_five(
        observations[0], initial, deterministic=False, behavior_uniform_mix=0.40
    )
    actions = tensors["actions"].reshape(1, 5)
    zeros = torch.zeros((1, 5), dtype=torch.long)
    evaluated = net.evaluate_sequence(
        observations,
        initial,
        actions=actions,
        targets=tensors["targets"].reshape(1, 5),
        delay_indices=tensors["delay_choice"].reshape(1, 5),
        risk_indices=tensors["risk_choice"].reshape(1, 5),
        entry_offset_indices=tensors["entry_offset_choice"].reshape(1, 5),
        manage_mode_indices=tensors["manage_mode_choice"].reshape(1, 5),
        close_fraction_indices=tensors["close_fraction_choice"].reshape(1, 5),
        behavior_uniform_mixes=torch.full((1, 5), 0.40),
    )
    expected = torch.as_tensor(
        [item.head_log_probabilities["action"] for item in results], dtype=torch.float32
    )
    torch.testing.assert_close(
        evaluated["primary_action_log_probability"][0], expected, rtol=1e-5, atol=1e-6
    )
    raw_probs = torch.as_tensor(
        np.stack([item.policy_action_probabilities for item in results]), dtype=torch.float32
    ).unsqueeze(0)
    raw_action_log_prob = torch.log(
        torch.gather(raw_probs, 2, actions.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(
        evaluated["policy_action_log_probability"][0],
        raw_action_log_prob[0],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.all(evaluated["primary_entry_probability_mass"] >= 0.03 - 1e-6)
    sampled = torch.as_tensor(
        np.stack([item.behavior_action_probabilities for item in results]), dtype=torch.float32
    )
    torch.testing.assert_close(
        evaluated["primary_action_probabilities"][0], sampled, rtol=1e-5, atol=1e-6
    )
    raw = torch.as_tensor(
        np.stack([item.policy_action_probabilities for item in results]), dtype=torch.float32
    )
    torch.testing.assert_close(
        evaluated["policy_action_probabilities"][0], raw, rtol=1e-5, atol=1e-6
    )


def test_phan_thuong_trung_gian_chi_thuong_thay_doi_kinh_te_moi() -> None:
    shaper = FiveTradingRewardShaper()
    base = FiveEconomicState(
        agent_mark_to_market_r=np.zeros(5),
        transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5) * 0.8,
        conflict_cost_r=np.zeros(5),
        team_equity_r=0.0,
        team_drawdown=0.0,
    )
    first = shaper.shape(base)
    np.testing.assert_allclose(first.raw_rewards, 0.0)
    gain = FiveEconomicState(
        agent_mark_to_market_r=np.asarray([0.2, 0.0, 0.0, 0.0, 0.0]),
        transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5) * 0.8,
        conflict_cost_r=np.zeros(5),
        team_equity_r=0.2,
        team_drawdown=0.0,
    )
    second = shaper.shape(gain)
    assert second.raw_rewards[0] > 0.19
    assert second.position_improvement_reward_r[0] == pytest.approx(0.2)
    third = shaper.shape(gain)
    np.testing.assert_allclose(third.raw_rewards, 0.0, atol=1e-7)
    np.testing.assert_allclose(third.position_improvement_reward_r, 0.0, atol=1e-7)


def test_feedback_dau_tien_khong_lam_mat_pnl_da_phat_sinh() -> None:
    shaper = FiveTradingRewardShaper()
    state = FiveEconomicState(
        agent_mark_to_market_r=np.asarray([-0.4, 0.2, 0.0, 0.0, 0.0]),
        transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5),
        team_equity_r=-0.2,
        team_drawdown=0.0,
    )
    out = shaper.shape(state)
    np.testing.assert_allclose(out.profit_delta_r[:2], [-0.4, 0.2], atol=1e-7)
    assert out.raw_rewards[0] < 0.0 < out.raw_rewards[1]


def test_reward_v2_chi_mo_bonus_wr_khi_expectancy_da_duong() -> None:
    shaper = FiveTradingRewardShaper(
        competition_weight=0.0,
        expectancy_wr_v2_enabled=True,
        wr_target=0.55,
        wr_balance_weight=0.05,
        expectancy_shortfall_weight=0.10,
        tail_loss_weight=0.10,
        expectancy_floor_r=0.05,
    )
    negative_expectancy = FiveEconomicState(
        agent_mark_to_market_r=np.asarray([-0.2, 0.0, 0.0, 0.0, 0.0]),
        transaction_cost_r=np.zeros(5), agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5), team_equity_r=-0.2, team_drawdown=0.0,
        agent_closed_trade_count=np.asarray([10, 0, 0, 0, 0]),
        agent_winning_trade_count=np.asarray([9, 0, 0, 0, 0]),
        agent_expectancy_shortfall_r=np.asarray([0.25, 0, 0, 0, 0]),
        agent_realized_trade_r=np.asarray([-0.2, 0, 0, 0, 0]),
    )
    blocked = shaper.shape(negative_expectancy)
    assert blocked.expectancy_gate[0] == 0.0
    assert blocked.wr_balance_bonus[0] == 0.0
    assert blocked.raw_rewards[0] < -0.2

    shaper.reset()
    positive_expectancy = FiveEconomicState(
        agent_mark_to_market_r=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
        transaction_cost_r=np.zeros(5), agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5), team_equity_r=1.0, team_drawdown=0.0,
        agent_closed_trade_count=np.asarray([10, 0, 0, 0, 0]),
        agent_winning_trade_count=np.asarray([7, 0, 0, 0, 0]),
        agent_realized_trade_r=np.asarray([1.0, 0, 0, 0, 0]),
    )
    allowed = shaper.shape(positive_expectancy)
    assert allowed.expectancy_gate[0] == 1.0
    assert allowed.wr_balance_bonus[0] > 0.0


def test_reward_v2_phat_loss_tail_va_expectancy_shortfall_chi_mot_lan() -> None:
    shaper = FiveTradingRewardShaper(
        competition_weight=0.0,
        expectancy_wr_v2_enabled=True,
        expectancy_shortfall_weight=0.10,
        tail_loss_weight=0.10,
    )
    state = FiveEconomicState(
        agent_mark_to_market_r=np.asarray([-2.0, 0.0, 0.0, 0.0, 0.0]),
        transaction_cost_r=np.zeros(5), agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5), team_equity_r=-2.0, team_drawdown=0.0,
        agent_closed_trade_count=np.asarray([1, 0, 0, 0, 0]),
        agent_expectancy_shortfall_r=np.asarray([2.05, 0, 0, 0, 0]),
        agent_tail_loss_cost_r=np.asarray([1.0, 0, 0, 0, 0]),
        agent_realized_trade_r=np.asarray([-2.0, 0, 0, 0, 0]),
    )
    first = shaper.shape(state)
    assert first.expectancy_shortfall_penalty[0] == pytest.approx(0.205)
    assert first.tail_loss_penalty[0] == pytest.approx(0.1)
    repeat = shaper.shape(state)
    np.testing.assert_allclose(repeat.raw_rewards, 0.0, atol=1e-7)


def test_nam_chien_luoc_canh_tranh_bang_cai_thien_loi_nhuan_tuong_doi() -> None:
    shaper = FiveTradingRewardShaper(competition_weight=0.50)
    shaper.reset(FiveEconomicState(
        agent_mark_to_market_r=np.zeros(5), transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5), conflict_cost_r=np.zeros(5),
        team_equity_r=0.0, team_drawdown=0.0,
    ))
    state = FiveEconomicState(
        agent_mark_to_market_r=np.asarray([1.0, 0.4, 0.1, -0.2, -0.5]),
        transaction_cost_r=np.zeros(5), agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5), team_equity_r=0.8, team_drawdown=0.0,
    )
    out = shaper.shape(state)
    assert out.raw_rewards[0] > out.raw_rewards[1] > out.raw_rewards[2]
    assert out.raw_rewards[2] > out.raw_rewards[3] > out.raw_rewards[4]
    assert np.isclose(np.sum(out.competitive_advantage), 0.0, atol=2e-7)
    assert np.isclose(np.sum(out.competitive_bonus), 0.0, atol=2e-7)
    # A static lead is not a new economic event and cannot earn reward again.
    repeat = shaper.shape(state)
    np.testing.assert_allclose(repeat.raw_rewards, 0.0, atol=1e-7)


def test_xep_hang_cuoi_episode_thuong_chien_luoc_loi_nhuan_hon_nhung_giu_team_score() -> None:
    shaper = FiveTradingRewardShaper(terminal_rank_weight=0.50)
    shaper.reset(FiveEconomicState(
        agent_mark_to_market_r=np.zeros(5), transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5), conflict_cost_r=np.zeros(5),
        team_equity_r=0.0, team_drawdown=0.0,
    ))
    out = shaper.shape(FiveEconomicState(
        agent_mark_to_market_r=np.asarray([2.0, 1.0, 0.0, -1.0, -2.0]),
        transaction_cost_r=np.zeros(5), agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5), team_equity_r=2.0, team_drawdown=0.5,
    ), terminal=True)
    assert np.isclose(out.terminal_team_score, 0.0)
    np.testing.assert_allclose(out.terminal_components, out.terminal_rank_bonus)
    assert out.terminal_components[0] > out.terminal_components[1] > out.terminal_components[2]
    assert out.terminal_components[2] > out.terminal_components[3] > out.terminal_components[4]
    assert np.max(np.abs(out.terminal_components)) <= 0.50 + 1e-7
    assert np.isclose(np.mean(out.terminal_rank_bonus), 0.0, atol=1e-7)


def test_phan_thuong_phat_sut_giam_moi_va_vuot_ngan_sach_rui_ro() -> None:
    shaper = FiveTradingRewardShaper()
    shaper.reset(FiveEconomicState(
        agent_mark_to_market_r=np.zeros(5),
        transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5),
        team_equity_r=0.0,
        team_drawdown=0.02,
    ))
    state = FiveEconomicState(
        agent_mark_to_market_r=np.zeros(5),
        transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.asarray([1.2, 1.0, 0.8, 1.0, 1.0]),
        conflict_cost_r=np.zeros(5),
        team_equity_r=0.0,
        team_drawdown=0.05,
    )
    out = shaper.shape(state)
    assert np.isclose(out.drawdown_increase, 0.03)
    assert out.raw_rewards[0] < out.raw_rewards[1] < 0.0
    repeat = shaper.shape(state)
    assert np.isclose(repeat.drawdown_increase, 0.0)
    assert repeat.raw_rewards[1] == 0.0


def test_terminal_khong_con_cong_lai_equity_tich_luy() -> None:
    shaper = FiveTradingRewardShaper()
    shaper.reset(FiveEconomicState(
        agent_mark_to_market_r=np.zeros(5), transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5), conflict_cost_r=np.zeros(5),
        team_equity_r=0.0, team_drawdown=0.0,
    ))
    out = shaper.shape(FiveEconomicState(
        agent_mark_to_market_r=np.ones(5) * 0.1, transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5), conflict_cost_r=np.zeros(5),
        team_equity_r=0.5, team_drawdown=0.1,
    ), terminal=True)
    assert np.isclose(out.terminal_team_score, 0.0)
    np.testing.assert_allclose(out.terminal_components, 0.0, atol=1e-7)


def test_p0_reward_khong_decay_theo_thoi_gian_trong_ngay() -> None:
    reward = np.asarray([1.0, -1.0, 0.25, -0.25, 0.0], dtype=np.float32)
    terminal = np.asarray([0.1, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    early = time_weight_shaped_rewards(reward, 60.0, terminal_component=terminal)
    late = time_weight_shaped_rewards(reward, 8.0 * 60.0 * 60.0, terminal_component=terminal)
    np.testing.assert_allclose(early, reward, atol=1e-7)
    np.testing.assert_allclose(late, reward, atol=1e-7)


def test_p0_terminal_equity_cuc_lon_khong_tao_reward_shock() -> None:
    common = dict(
        agent_mark_to_market_r=np.asarray([2.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float32),
        transaction_cost_r=np.zeros(5, dtype=np.float32),
        agent_risk_ratio=np.ones(5, dtype=np.float32),
        conflict_cost_r=np.zeros(5, dtype=np.float32),
        team_drawdown=100.0,
    )
    shaper = FiveTradingRewardShaper(terminal_rank_weight=0.50)
    shaper.reset(FiveEconomicState(
        agent_mark_to_market_r=np.zeros(5), transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5), conflict_cost_r=np.zeros(5),
        team_equity_r=0.0, team_drawdown=0.0,
    ))
    out = shaper.shape(FiveEconomicState(team_equity_r=-100.0, **common), terminal=True)
    assert out.terminal_team_score == pytest.approx(0.0)
    assert np.max(np.abs(out.terminal_components)) <= 0.50 + 1e-7
    assert np.max(np.abs(out.raw_rewards - (
        out.profit_delta_r + out.competitive_bonus
        - np.float32(shaper.drawdown_weight * out.drawdown_increase)
    ))) <= 0.50 + 1e-6


def test_duong_act_mot_lenh_bi_tu_choi_sau_khi_bridge_nam_agent_thay_the() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    with pytest.raises(RuntimeError, match="act_nam_tac_nhan"):
        system.act(np.zeros(137, dtype=np.float32))


def test_phan_hoi_runtime_gom_du_nam_tac_nhan_moi_goi_duong_hoc_five() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    traces = list(system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)))
    for i, trace in enumerate(traces):
        result = system.observe_transition(trace, np.zeros(137), 0.1 * (i + 1), 0.0, False)
        assert result["five_feedback_count"] == float(i + 1 if i < 4 else 5)
        if i < 4:
            assert "five_feedback_batch_submitted" not in result
        else:
            assert result["five_feedback_batch_submitted"] == 1.0


def test_bridge_economic_batch_cap_nhat_schedule_va_rollout_mot_lan() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    traces = system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32))
    result = system.observe_five_economic_transition(
        traces,
        agent_mark_to_market_r=np.asarray([0.4, 0.2, 0.0, -0.1, -0.2]),
        transaction_cost_r=np.zeros(5),
        agent_risk_ratio=np.ones(5),
        conflict_cost_r=np.zeros(5),
        team_equity_r=0.3,
        team_drawdown=0.1,
        elapsed_seconds=60.0,
        step_seconds=60.0,
        training_progress=0.5,
    )
    assert result["rollout_steps"] == 1.0
    assert result["updated"] == 0.0
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    assert owner.five_training_progress == pytest.approx(0.5)
    assert owner.five_ppo_trainer is not None
    assert owner.five_ppo_trainer.schedule.team_spirit == pytest.approx(0.65)


def test_reward_v2_mt5_cumulative_khong_bi_cong_lai_sau_terminal() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner.five_reward_expectancy_wr_v2_enabled = True
    owner.reset_five_reward_shaper()
    zeros = np.zeros(5, dtype=np.float32)
    ones = np.ones(5, dtype=np.float32)
    cumulative = {
        "agent_mark_to_market_r": np.asarray([-2.0, 0, 0, 0, 0], dtype=np.float32),
        "transaction_cost_r": zeros,
        "agent_risk_ratio": ones,
        "conflict_cost_r": zeros,
        "team_equity_r": -2.0,
        "team_drawdown": 2.0,
        "agent_closed_trade_count": np.asarray([1, 0, 0, 0, 0], dtype=np.float32),
        "agent_winning_trade_count": zeros,
        "agent_expectancy_shortfall_r": np.asarray([2.05, 0, 0, 0, 0], dtype=np.float32),
        "agent_tail_loss_cost_r": np.asarray([1.0, 0, 0, 0, 0], dtype=np.float32),
        "agent_realized_trade_r": np.asarray([-2.0, 0, 0, 0, 0], dtype=np.float32),
    }
    first = system.observe_five_economic_transition(
        system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)),
        **cumulative,
        terminal=True,
        dones=np.ones(5, dtype=bool),
    )
    assert first["reward_closed_trades"] == pytest.approx(1.0)
    assert first["reward_expectancy_shortfall_mean"] > 0.0
    assert first["reward_tail_loss_mean"] > 0.0

    repeated = system.observe_five_economic_transition(
        system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)),
        **cumulative,
        terminal=False,
        dones=np.zeros(5, dtype=bool),
    )
    assert repeated["reward_closed_trades"] == pytest.approx(0.0)
    assert repeated["reward_wins"] == pytest.approx(0.0)
    assert repeated["reward_wr_balance_mean"] == pytest.approx(0.0)
    assert repeated["reward_expectancy_shortfall_mean"] == pytest.approx(0.0)
    assert repeated["reward_tail_loss_mean"] == pytest.approx(0.0)


def test_canh_tranh_van_ton_tai_khi_team_spirit_da_dat_mot() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    zeros = np.zeros(5, dtype=np.float32)
    ones = np.ones(5, dtype=np.float32)
    system.observe_five_economic_transition(
        system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)),
        agent_mark_to_market_r=zeros,
        transaction_cost_r=zeros,
        agent_risk_ratio=ones,
        conflict_cost_r=zeros,
        team_equity_r=0.0,
        team_drawdown=0.0,
        training_progress=1.0,
    )
    system.observe_five_economic_transition(
        system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)),
        agent_mark_to_market_r=np.asarray([1.0, 0.5, 0.0, -0.5, -1.0]),
        transaction_cost_r=zeros,
        agent_risk_ratio=ones,
        conflict_cost_r=zeros,
        team_equity_r=0.0,
        team_drawdown=0.0,
        training_progress=1.0,
    )
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    assert owner.five_ppo_trainer.schedule.team_spirit == pytest.approx(1.0)
    rewards = np.asarray(owner.five_rollout_steps[-1]["rewards"], dtype=np.float32)
    assert rewards[0] > rewards[1] > rewards[2] > rewards[3] > rewards[4]


def test_reward_vong_doi_reforward_dung_trace_entry_bat_bien() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner._learning_enabled = True
    rows = list(system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details["five_action_index"] = 1 if index == 0 else 0
        details["five_economic_actor_eligible"] = bool(index == 0)
        details["five_execution_filled"] = bool(index == 0)
        if index == 0:
            details["five_selected_action_probability"] = 0.90
        enriched.append(replace(trace, action_details=details))

    registered = system.register_five_entry_lifecycle_packet(tuple(enriched))
    assert registered["registered"] == 1
    routed = system.apply_five_entry_lifecycle_reward(
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
    assert routed["reward_r"] == pytest.approx(-0.30)
    assert routed["initial_sl_local_loss"] is True
    assert len(owner.five_lifecycle_steps) == 1
    queued = np.asarray(owner.five_lifecycle_steps[0]["rewards"], dtype=np.float32)
    assert queued[0] == pytest.approx(-0.30)
    assert np.count_nonzero(queued) == 1
    local_mask = np.asarray(
        owner.five_lifecycle_steps[0]["initial_sl_local_loss_masks"], dtype=bool
    )
    assert local_mask.tolist() == [True, False, False, False, False]


def test_flush_lifecycle_queue_hoc_mau_duoi_batch_size_ma_khong_tao_mau_moi() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner._learning_enabled = True
    owner.five_lifecycle_batch_size = 16
    rows = list(system.act_nam_tac_nhan(np.zeros(INPUT_COUNT, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details.update({
            "five_action_index": 1 if index == 0 else 0,
            "five_economic_actor_eligible": bool(index == 0),
            "five_execution_filled": bool(index == 0),
        })
        enriched.append(replace(trace, action_details=details))
    assert system.register_five_entry_lifecycle_packet(tuple(enriched))["registered"] == 1
    assert system.apply_five_entry_lifecycle_reward(
        target_id,
        lifecycle_reward_r=-0.25,
        realized_r=-0.25,
        components={"total_r": -0.25},
    )["accepted"] is True
    assert len(owner.five_lifecycle_steps) == 1

    flushed = system.flush_five_lifecycle_reward_queue()

    assert flushed["flushed_batches"] == 1
    assert flushed["flushed_steps"] == 1
    assert flushed["pending"] == 0
    assert owner.five_lifecycle_steps == []


def test_lifecycle_cache_bo_qua_entry_proposal_khong_duoc_mt5_xac_nhan() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner._learning_enabled = True
    rows = list(system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)))
    unexecuted = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details["five_action_index"] = 1 if index == 0 else 0
        details["five_economic_actor_eligible"] = bool(index == 0)
        details["five_execution_filled"] = False
        details["five_execution_placed"] = False
        unexecuted.append(replace(trace, action_details=details))

    assert owner.register_five_entry_lifecycle_packet(tuple(unexecuted))["registered"] == 0

    filled = list(unexecuted)
    filled[0] = replace(
        filled[0],
        action_details={
            **dict(filled[0].action_details),
            "five_execution_filled": True,
        },
    )
    assert owner.register_five_entry_lifecycle_packet(tuple(filled))["registered"] == 1


def test_lifecycle_owner_packet_duoc_giu_qua_episode_boundary() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner._learning_enabled = True
    rows = list(system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details["five_action_index"] = 1 if index == 0 else 0
        details["five_economic_actor_eligible"] = bool(index == 0)
        details["five_execution_filled"] = bool(index == 0)
        enriched.append(replace(trace, action_details=details))

    assert owner.register_five_entry_lifecycle_packet(tuple(enriched))["registered"] == 1
    owner.reset_five_agent_memories(clear_rollout=True)
    routed = owner.apply_five_entry_lifecycle_reward(
        target_id,
        lifecycle_reward_r=0.25,
        realized_r=0.25,
        components={"total_r": 0.25},
    )

    assert routed["accepted"] is True
    assert routed["reason"] == "entry_lifecycle_reward_queued"


def test_reward_vong_doi_candidate_cong_net_r_co_dau_chi_cho_owner() -> None:
    cfg = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    owner._learning_enabled = True
    owner.five_lifecycle_net_r_credit_enabled = True
    owner.five_lifecycle_net_r_credit_weight = 1.0
    owner.five_lifecycle_net_r_credit_clip_r = 2.5
    rows = list(system.act_nam_tac_nhan(np.zeros(137, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details["five_action_index"] = 1 if index == 0 else 0
        details["five_economic_actor_eligible"] = bool(index == 0)
        details["five_execution_filled"] = bool(index == 0)
        enriched.append(replace(trace, action_details=details))
    assert system.register_five_entry_lifecycle_packet(tuple(enriched))["registered"] == 1

    routed = system.apply_five_entry_lifecycle_reward(
        target_id,
        lifecycle_reward_r=0.0,
        realized_r=-1.25,
        components={"total_r": 0.0},
    )

    assert routed["accepted"] is True
    assert routed["reward_v4_r"] == pytest.approx(0.0)
    assert routed["net_r_credit_r"] == pytest.approx(-1.25)
    assert routed["reward_r"] == pytest.approx(-1.25)
    queued = np.asarray(owner.five_lifecycle_steps[0]["rewards"], dtype=np.float32)
    assert queued.tolist() == pytest.approx([-1.25, 0.0, 0.0, 0.0, 0.0])


def test_loi_ro_rang_neu_goi_nham_duong_phan_hoi_mot_trace_tren_loi_five() -> None:
    policy = _small_policy()
    trace = policy.act_five_agents(np.zeros(137, dtype=np.float32))[0]
    with pytest.raises(RuntimeError, match="observe_five_transition"):
        policy.observe_transition(trace, np.zeros(137), 0.0, 0.0, False)
