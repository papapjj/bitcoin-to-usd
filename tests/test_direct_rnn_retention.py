from __future__ import annotations

from dataclasses import replace
import pickle

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.entry_preference import (
    COUNTERFACTUAL_CONSENSUS_TARGET_MODE,
    HIERARCHICAL_TRADE_TARGET_MODE,
    HOLD_ACTION,
    MARKET_BUY,
    MARKET_SELL,
    hierarchical_action_probabilities,
    hierarchical_preference_loss_and_ascent,
    hierarchical_preference_target,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.retention import ExpertRetentionStore


def _policy(*, rollback_tolerance: float = 0.02) -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=4,
        seed=881,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_adamw_learning_rate = 5.0e-4
    policy.direct_rnn_gradient_clip = 10.0
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_min_abs_mean_advantage = 0.0
    policy.direct_rnn_rl_entropy_weight = 0.0
    policy.direct_rnn_rl_trade_hold_separation_enabled = True
    policy.direct_rnn_rl_trade_threshold = 0.55
    policy.direct_rnn_rl_retention_enabled = True
    policy.direct_rnn_rl_retention_recent_capacity = 8
    policy.direct_rnn_rl_retention_diverse_capacity = 16
    policy.direct_rnn_rl_retention_anchor_capacity = 8
    policy.direct_rnn_rl_retention_replay_items_per_batch = 2
    policy.direct_rnn_rl_retention_replay_weight = 0.5
    policy.direct_rnn_rl_retention_distillation_weight = 0.25
    policy.direct_rnn_rl_retention_anchor_min_r = 0.25
    policy.direct_rnn_rl_retention_audit_min_anchors = 1
    policy.direct_rnn_rl_retention_max_anchor_kl_increase = rollback_tolerance
    policy.direct_rnn_rl_retention_max_update_norm = 100.0
    policy._learning_enabled = True
    policy.ensure_compatible()
    return policy


def _forced_entry_trace(policy: ChuyenGiaPolicy, action: int, offset: float):
    trace = policy.act(np.linspace(-0.4 + offset, 0.6 + offset, INPUT_COUNT, dtype=float))
    output = np.full(OUTPUT_COUNT, -1.0, dtype=float)
    output[action] = 1.0
    details = dict(trace.action_details)
    details["direct_rnn_rl_policy_action_index"] = int(action)
    details["direct_rnn_rl_policy_eligible"] = True
    return replace(
        trace,
        selected_idx=int(action),
        selected_action=output,
        action_details=details,
    )


def _queue(policy: ChuyenGiaPolicy, action: int, outcome: float, offset: float) -> int:
    trace = _forced_entry_trace(policy, action, offset)
    return policy._queue_direct_rnn_full_rl_lifecycle(
        trace,
        outcome,
        outcome,
        update_scale=1.0,
    )


def test_trade_hold_probability_is_not_biased_by_six_entry_classes() -> None:
    probabilities, trade_probability, conditional = hierarchical_action_probabilities(
        np.zeros(7, dtype=float)
    )
    assert trade_probability == 0.5
    assert probabilities[HOLD_ACTION] == 0.5
    np.testing.assert_allclose(probabilities[1:], np.full(6, 1.0 / 12.0))
    np.testing.assert_allclose(conditional[1:], np.full(6, 1.0 / 6.0))


def test_tensor_rule_retention_bucket_includes_rule_fold_action_and_outcome() -> None:
    policy = _policy()
    trace = _forced_entry_trace(policy, MARKET_BUY, 0.0)
    trace = replace(
        trace,
        tensor_rule_trace={"schema_version": 1, "rule_id": 3},
    )
    store = ExpertRetentionStore()
    episode_id = store.add(
        {"trace": trace, "outcome": 1.0, "tensor_rule_temporal_fold": 2},
        anchor_min_r=0.25,
    )
    bucket = store.episodes[episode_id].bucket
    assert bucket[0] == "lifecycle"
    assert bucket[2] == str(MARKET_BUY)
    assert bucket[3] == "win"
    assert bucket[-2:] == ("rule:3", "fold:2")


def test_hierarchical_preference_ascent_matches_finite_difference() -> None:
    logits = np.asarray([0.2, -0.3, 0.7, -0.1, 0.4, -0.8, 0.1], dtype=float)
    target, mask = hierarchical_preference_target(
        [0.0, -0.5, 1.0, -0.2, 0.4, -1.0, 0.2],
        trade_margin_r=0.10,
        trade_temperature=0.25,
    )
    loss, ascent, _probabilities, _trade = hierarchical_preference_loss_and_ascent(
        logits, target, action_mask=mask
    )
    assert loss > 0.0
    epsilon = 1.0e-6
    numerical = np.zeros_like(logits)
    for index in range(logits.size):
        plus = logits.copy()
        minus = logits.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss = hierarchical_preference_loss_and_ascent(
            plus, target, action_mask=mask
        )[0]
        minus_loss = hierarchical_preference_loss_and_ascent(
            minus, target, action_mask=mask
        )[0]
        numerical[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    np.testing.assert_allclose(ascent, -numerical, atol=2.0e-6, rtol=2.0e-6)


def test_trade_target_uses_selected_action_not_hindsight_best() -> None:
    utilities = [0.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    target, _mask = hierarchical_preference_target(
        utilities,
        selected_action=MARKET_BUY,
        trade_margin_r=0.10,
        trade_temperature=0.25,
    )
    expected_trade = 1.0 / (1.0 + np.exp(-(-1.0 - 0.10) / 0.25))
    assert np.isclose(float(np.sum(target[1:])), expected_trade)
    assert target[HOLD_ACTION] > 0.98


def test_hold_trade_target_uses_mean_legal_entry_utility() -> None:
    utilities = [0.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    target, _mask = hierarchical_preference_target(
        utilities,
        selected_action=HOLD_ACTION,
        trade_margin_r=0.0,
        trade_temperature=0.25,
    )
    assert np.isclose(float(np.sum(target[1:])), 0.5)


def test_retention_discards_replaced_preference_target_contract() -> None:
    store = ExpertRetentionStore()
    stale = {
        "credit_kind": "entry_preference_counterfactual",
        "entry_preference_only": True,
        "entry_preference_sample_scope": "stale",
        "entry_preference_target_mode": "hierarchical_trade_hold_v1",
    }
    current = {
        "credit_kind": "entry_preference_counterfactual",
        "entry_preference_only": True,
        "entry_preference_sample_scope": "current",
        "entry_preference_target_mode": HIERARCHICAL_TRADE_TARGET_MODE,
    }
    stale_id = store.add(stale, anchor_min_r=0.25)
    current_id = store.add(current, anchor_min_r=0.25)
    assert store.discard_incompatible_preference_targets(
        {HIERARCHICAL_TRADE_TARGET_MODE}
    ) == 1
    assert stale_id not in store.episodes
    assert current_id in store.episodes
    assert store.stats()["discarded_count"] == 1


def test_consensus_preference_survives_repeated_compatibility_checks() -> None:
    policy = _policy()
    policy.direct_rnn_rl_delayed_preference_target_mode = (
        COUNTERFACTUAL_CONSENSUS_TARGET_MODE
    )
    item = _hold_preference_item(policy, "consensus-pending")
    item["entry_preference_target_mode"] = COUNTERFACTUAL_CONSENSUS_TARGET_MODE
    policy._direct_rnn_rl_batches = [item]
    episode_id = policy.direct_rnn_retention_store.add(item, anchor_min_r=0.25)

    policy.ensure_compatible()
    policy.ensure_compatible()

    assert len(policy._direct_rnn_rl_batches) == 1
    assert policy._direct_rnn_rl_batches[0]["entry_preference_target_mode"] == (
        COUNTERFACTUAL_CONSENSUS_TARGET_MODE
    )
    assert episode_id in policy.direct_rnn_retention_store.episodes


def test_equal_entry_logits_choose_hold_at_the_candidate_threshold() -> None:
    policy = _policy()
    module = policy.modules[0]
    module.w_out[:7].fill(0.0)
    module.b_out[:7].fill(0.0)
    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    assert trace.action_details["direct_rnn_rl_trade_probability"] == 0.5
    assert trace.action_details["direct_rnn_rl_trade_hold_decision"] == "hold"
    assert policy._trace_action_index(trace, 12) == HOLD_ACTION


def test_retention_replays_past_batches_and_survives_pickle() -> None:
    policy = _policy()
    assert _queue(policy, MARKET_BUY, 1.0, 0.00) == 0
    assert _queue(policy, MARKET_BUY, 0.8, 0.01) == 1
    first = policy.direct_rnn_retention_store.stats()
    assert first["episodes"] == 2
    assert first["anchors"] == 2
    assert first["batch_index"] == 1
    sampled = policy.direct_rnn_retention_store.sample(
        1, replay_weight=0.5, min_lifecycle_fraction=1.0
    )
    assert len(sampled) == 1
    episode_id = str(sampled[0]["retention_episode_id"])
    stored_outcome = float(
        policy.direct_rnn_retention_store.episodes[episode_id].payload["outcome"]
    )
    sampled[0]["outcome"] = -99.0
    assert float(
        policy.direct_rnn_retention_store.episodes[episode_id].payload["outcome"]
    ) == stored_outcome

    assert _queue(policy, MARKET_SELL, -0.8, 0.02) == 0
    assert _queue(policy, MARKET_SELL, -1.0, 0.03) in {0, 1}
    event = [
        row for row in policy.growth_events
        if row.get("type") == "direct_rnn_full_rl_batch"
    ][-1]
    assert event["retention_enabled"] is True
    assert event["retention_replay_count"] in {1, 2}
    assert event["retention_anchor_replay_count"] == event["retention_replay_count"]
    assert event["retention_distillation_count"] == event["retention_replay_count"]
    assert event["retention_audit_only_count"] in {0, 1}
    assert event["retention_replay_count"] + event["retention_audit_only_count"] == 2
    expected_source = (
        "replay_held_out_store_v1"
        if event["retention_audit_only_count"] else "training_batch_fallback"
    )
    assert event["retention_anchor_audit_source"] == expected_source
    assert policy.direct_rnn_retention_store.stats()["replayed_count"] == event[
        "retention_replay_count"
    ]

    restored = pickle.loads(pickle.dumps(policy, protocol=pickle.HIGHEST_PROTOCOL))
    restored.ensure_compatible()
    assert restored.direct_rnn_retention_store.stats() == policy.direct_rnn_retention_store.stats()


def _hold_preference_item(policy: ChuyenGiaPolicy, trace_id: str) -> dict[str, object]:
    trace = replace(
        _forced_entry_trace(policy, HOLD_ACTION, 0.0),
        trace_id=trace_id,
    )
    target = np.asarray([0.90] + [0.10 / 6.0] * 6, dtype=float)
    return {
        "trace": trace,
        "outcome": 0.0,
        "advantage": 0.0,
        "update_scale": 1.0,
        "credit_kind": "entry_preference_counterfactual",
        "entry_preference_only": True,
        "entry_preference_target": target,
        "entry_preference_mask": np.ones(7, dtype=bool),
        "entry_preference_target_mode": HIERARCHICAL_TRADE_TARGET_MODE,
        "entry_preference_hard_negative": False,
        "entry_preference_retention_hold_threshold": 0.55,
        "entry_preference_sample_scope": "flat_v1",
    }


def test_confirmed_hold_is_audited_as_preference_anchor() -> None:
    policy = _policy()
    store = ExpertRetentionStore(
        recent_capacity=8, diverse_capacity=8, anchor_capacity=8
    )
    episode_id = store.add(
        _hold_preference_item(policy, "confirmed-hold"), anchor_min_r=0.25
    )
    episode = store.episodes[episode_id]
    assert episode.preference_anchor is True
    assert episode.behavior_anchor is False
    assert episode.bucket[-1] == "control_anchor"
    store.note_batch([])
    sampled = store.sample(1, replay_weight=0.5, min_lifecycle_fraction=0.0)
    assert sampled[0]["retention_preference_anchor"] is True
    assert sampled[0]["retention_behavior_anchor"] is False
    kl, count, behavior_count, preference_count, scores = (
        policy._direct_rnn_retention_anchor_kl(sampled)
    )
    assert np.isfinite(kl)
    assert (count, behavior_count, preference_count) == (1, 0, 1)
    assert len(scores) == 1


def test_anchor_capacity_balances_profitable_lifecycle_and_hold_controls() -> None:
    policy = _policy()
    store = ExpertRetentionStore(
        recent_capacity=64, diverse_capacity=64, anchor_capacity=4
    )
    for index in range(8):
        trace = replace(
            _forced_entry_trace(policy, MARKET_BUY, index * 0.001),
            trace_id=f"winner-{index}",
        )
        store.add({
            "trace": trace,
            "outcome": 1.0,
            "retention_behavior_probabilities": np.asarray(
                trace.action_details["direct_rnn_rl_action_probabilities"], dtype=float
            ),
        }, anchor_min_r=0.25)
    for index in range(12):
        store.add(
            _hold_preference_item(policy, f"hold-{index}"), anchor_min_r=0.25
        )
    anchor_rows = [store.episodes[item] for item in store.anchor_ids]
    assert len(anchor_rows) == 4
    assert any(row.behavior_anchor for row in anchor_rows)
    assert any(row.preference_anchor for row in anchor_rows)


def test_schema_v2_hold_target_migrates_to_preference_anchor() -> None:
    policy = _policy()
    store = ExpertRetentionStore()
    episode_id = store.add(
        _hold_preference_item(policy, "legacy-hold"), anchor_min_r=0.25
    )
    episode = store.episodes[episode_id]
    del episode.preference_anchor
    episode.anchor = False
    episode.payload.pop("entry_preference_retention_anchor", None)
    store.anchor_ids.clear()
    store.schema_version = 2
    store.ensure_compatible()
    migrated = store.episodes[episode_id]
    assert store.schema_version == 6
    assert migrated.preference_anchor is True
    assert migrated.anchor is True
    assert episode_id in store.anchor_ids


def test_audit_split_is_disjoint_from_gradient_replay() -> None:
    policy = _policy()
    store = ExpertRetentionStore(
        recent_capacity=128, diverse_capacity=128, anchor_capacity=128
    )
    for index in range(60):
        store.add(
            _hold_preference_item(policy, f"audit-split-{index}"),
            anchor_min_r=0.25,
        )
    store.note_batch([])
    replay = store.sample(128, replay_weight=0.5, min_lifecycle_fraction=0.0)
    audit = store.audit(128)
    replay_ids = {str(row["retention_episode_id"]) for row in replay}
    audit_ids = {str(row["retention_episode_id"]) for row in audit}
    assert replay_ids
    assert audit_ids
    assert replay_ids.isdisjoint(audit_ids)
    assert all(row.get("retention_replay") is True for row in replay)
    assert all(row.get("retention_audit_only") is True for row in audit)
    assert all(not bool(row.get("retention_replay", False)) for row in audit)
    assert store.stats()["audit_only_anchors"] == len(audit_ids)


def test_audit_split_uses_deterministic_frame_not_runtime_trace_suffix() -> None:
    policy = _policy()
    store = ExpertRetentionStore()
    items = []
    for trace_id in ("runtime-a", "runtime-b"):
        item = _hold_preference_item(policy, trace_id)
        trace = item["trace"]
        details = dict(trace.action_details)
        details["context_frame_id"] = "mt5:0:77"
        item["trace"] = replace(trace, action_details=details)
        items.append(item)
    first = store.add(items[0], anchor_min_r=0.25)
    second = store.add(items[1], anchor_min_r=0.25)
    assert first != second
    assert store.episodes[first].audit_only == store.episodes[second].audit_only


def test_held_out_preference_regression_rolls_back_candidate_step() -> None:
    policy = _policy(rollback_tolerance=0.0)
    audit_item = _hold_preference_item(policy, "held-out-hold")
    audit_item.update({
        "retention_anchor": True,
        "retention_preference_anchor": True,
        "retention_audit_only": True,
    })
    hold_trace = audit_item["trace"]
    details = dict(hold_trace.action_details)
    details["direct_rnn_rl_policy_action_index"] = MARKET_BUY
    selected_output = np.full(OUTPUT_COUNT, -1.0, dtype=float)
    selected_output[MARKET_BUY] = 1.0
    buy_trace = replace(
        hold_trace,
        selected_idx=MARKET_BUY,
        selected_action=selected_output,
        action_details=details,
    )
    training_item = {
        "trace": buy_trace,
        "outcome": 1.0,
        "advantage": 1.0,
        "raw_advantage": 1.0,
        "actor_advantage_pre_normalization": 1.0,
        "update_scale": 1.0,
        "policy_eligible": True,
    }
    before = {
        module.name: {
            key: value.copy()
            for key, value in module._rnn_optimizer_parameters().items()
        }
        for module in policy.modules
    }
    assert policy._apply_direct_rnn_full_rl_batch(
        [training_item, dict(training_item)],
        retention_audit_batch=[audit_item],
    ) == 0
    event = policy.growth_events[-1]
    assert event["retention_anchor_audit_source"] == "replay_held_out_store_v1"
    assert event["retention_preference_anchor_audit_count"] == 1
    assert event["retention_anchor_kl_max_increase"] > 0.0
    assert event["retention_rollback_reason"] == "anchor_kl_regression"
    for module in policy.modules:
        for key, value in module._rnn_optimizer_parameters().items():
            np.testing.assert_array_equal(value, before[module.name][key])


def test_anchor_audit_never_accepts_an_a_to_b_regression() -> None:
    policy = _policy(rollback_tolerance=0.0)
    _queue(policy, MARKET_BUY, 1.0, 0.00)
    _queue(policy, MARKET_BUY, 1.0, 0.01)
    before = {
        module.name: {
            key: value.copy() for key, value in module._rnn_optimizer_parameters().items()
        }
        for module in policy.modules
    }

    _queue(policy, MARKET_SELL, -1.0, 0.02)
    _queue(policy, MARKET_SELL, -1.0, 0.03)
    event = [
        row for row in policy.growth_events
        if row.get("type") == "direct_rnn_full_rl_batch"
    ][-1]
    assert event["retention_anchor_audit_count"] >= 1
    if event["retention_rollback"]:
        assert event["retention_rollback_reason"] == "anchor_kl_regression"
        for module in policy.modules:
            for key, value in module._rnn_optimizer_parameters().items():
                np.testing.assert_array_equal(value, before[module.name][key])
    else:
        assert event["retention_anchor_kl_after"] <= event["retention_anchor_kl_before"] + 1.0e-12


def test_update_norm_guard_restores_weights_and_optimizer_state() -> None:
    policy = _policy()
    policy.direct_rnn_rl_retention_max_update_norm = 1.0e-12
    policy.ensure_compatible()
    module = policy.modules[0]
    before = {key: value.copy() for key, value in module._rnn_optimizer_parameters().items()}
    _queue(policy, MARKET_BUY, 1.0, 0.00)
    assert _queue(policy, MARKET_BUY, 1.0, 0.01) == 0
    event = [
        row for row in policy.growth_events
        if row.get("type") == "direct_rnn_full_rl_batch"
    ][-1]
    assert event["retention_rollback"] is True
    assert event["retention_rollback_reason"] == "update_norm_limit"
    assert module.rnn_optimizer_step == 0
    for key, value in module._rnn_optimizer_parameters().items():
        np.testing.assert_array_equal(value, before[key])
