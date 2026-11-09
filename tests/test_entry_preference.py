from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.entry_preference import (
    BUY_LIMIT,
    BUY_STOP,
    HOLD_ACTION,
    MARKET_BUY,
    EntryPathBar,
    EntryPreferenceRecord,
    parse_entry_preference_row,
    preference_target,
    replay_entry_component_matrix,
    replay_entry_timing_comparisons,
    replay_entry_utilities,
    timing_gate_loss_and_ascent,
    timing_gate_target,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.mt5_runner import (
    _IncrementalCsvTail,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
    write_tester_set,
)


def _record(trace_id: str) -> EntryPreferenceRecord:
    # Buy market and buy-limit lose; buy-stop fills then reaches TP. HOLD is
    # still present as a valid 0R target, proving the label is action-local.
    return EntryPreferenceRecord(
        trace_id=trace_id,
        sequence=1,
        decision_time="2025.01.01 00:00",
        selected_action=MARKET_BUY,
        bid=100.0,
        ask=101.0,
        atr=1.0,
        sl_atr=1.0,
        rr=2.0,
        entry_prices=(101.0, 100.0, 100.0, 102.0, 102.0, 99.0),
        bars=(
            EntryPathBar(high=100.5, low=98.5, close=99.5, spread=1.0),
            EntryPathBar(high=104.5, low=101.5, close=104.0, spread=1.0),
        ),
    )


def _policy() -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=501,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_entry_preference_enabled = True
    policy.direct_rnn_rl_entry_preference_loss_weight = 0.25
    policy.direct_rnn_rl_entry_preference_temperature = 0.5
    policy.direct_rnn_rl_entry_utility_mode = "path_quality_v1"
    policy.direct_rnn_rl_auxiliary_enabled = True
    policy.direct_rnn_rl_auxiliary_loss_weight = 0.10
    policy.direct_rnn_rl_reconstruction_loss_weight = 0.02
    policy.direct_rnn_rl_counterfactual_representation_enabled = True
    policy.direct_rnn_rl_counterfactual_representation_loss_weight = 0.08
    policy.direct_rnn_rl_counterfactual_representation_temperature = 0.5
    policy.direct_rnn_gradient_clip = 10.0
    policy._learning_enabled = True
    policy.ensure_compatible()
    return policy


def test_counterfactual_replay_is_conservative_and_keeps_hold() -> None:
    utilities = replay_entry_utilities(_record("r"))
    assert utilities[HOLD_ACTION] == 0.0
    assert utilities[MARKET_BUY] == -1.0
    assert utilities[BUY_LIMIT] == -1.0
    assert utilities[BUY_STOP] == 2.0

    target, mask = preference_target(utilities, allowed_actions=(HOLD_ACTION, MARKET_BUY, BUY_STOP))
    assert mask.tolist() == [True, True, False, False, False, True, False]
    assert target[BUY_STOP] > target[HOLD_ACTION] > target[MARKET_BUY]
    assert target[2] == 0.0


def test_path_quality_decomposes_entry_without_management_pnl() -> None:
    record = _record("quality")
    components = replay_entry_component_matrix(record)
    buy = components[MARKET_BUY]
    stop = components[BUY_STOP]
    assert buy.terminal_r == -1.0
    assert buy.mae_r <= -1.0
    assert buy.quality_r < 0.0
    assert stop.terminal_r == 2.0
    assert stop.mfe_r >= 2.0
    assert stop.quality_r > 0.0
    terminal = replay_entry_utilities(record, mode="terminal_r_v1")
    quality = replay_entry_utilities(record, mode="path_quality_v1")
    assert terminal[BUY_STOP] == 2.0
    assert quality[BUY_STOP] != terminal[BUY_STOP]
    assert quality[BUY_STOP] > quality[HOLD_ACTION] > quality[MARKET_BUY]


def test_relative_timing_compares_same_action_without_future_bar_leakage() -> None:
    record = EntryPreferenceRecord(
        trace_id="timing", sequence=1, decision_time="2025.01.01 00:00",
        selected_action=MARKET_BUY, bid=100.0, ask=100.2, atr=1.0,
        sl_atr=1.0, rr=2.0,
        entry_prices=(100.2, 100.0, 99.2, 101.0, 101.2, 99.0),
        bars=(
            EntryPathBar(100.3, 99.8, 100.0, 0.2),
            EntryPathBar(100.2, 99.0, 99.2, 0.2),
            EntryPathBar(101.6, 99.1, 101.4, 0.2),
            EntryPathBar(102.0, 101.0, 101.8, 0.2),
        ),
    )
    rows = replay_entry_timing_comparisons(
        record, delay_bars=(1, 2), utility_mode="path_quality_v1"
    )
    buy = next(row for row in rows if row.action == MARKET_BUY)
    assert buy.valid_delays == 2
    assert buy.best_delay_bars in {1, 2}
    assert buy.timing_advantage_r < 0.0
    target, audit = timing_gate_target(
        rows, allowed_actions=(HOLD_ACTION, MARKET_BUY), consensus_actions=1
    )
    assert audit["eligible"] is True
    assert target < 0.5


def test_timing_gate_gradient_changes_only_trade_hold_group() -> None:
    logits = np.asarray([0.1, 0.8, -0.2, 0.4, -0.5, 0.2, -0.1])
    _, direction, _, _ = timing_gate_loss_and_ascent(logits, 0.0)
    # Entry logits move only as one aggregate group: dividing by their current
    # conditional probabilities yields the same scalar for all six actions.
    from self_growing_trader.entry_preference import hierarchical_action_probabilities
    _, _, conditional = hierarchical_action_probabilities(logits)
    ratios = direction[1:7] / np.maximum(conditional[1:7], 1e-12)
    assert np.max(ratios) - np.min(ratios) < 1e-10
    assert direction[HOLD_ACTION] > 0.0


def test_timing_first_batch_freezes_terminal_entry_actor_and_trains_gate() -> None:
    policy = _policy()
    policy.direct_rnn_rl_timing_first_enabled = True
    policy.direct_rnn_rl_timing_delay_bars = (1,)
    policy.direct_rnn_rl_timing_gate_loss_weight = 0.5
    policy.direct_rnn_rl_timing_freeze_entry_lifecycle_actor = True
    preference_trace = policy.act(np.linspace(-0.3, 0.4, INPUT_COUNT, dtype=float))
    preference_trace = replace(preference_trace, trace_id="timing-first-pref")
    selected = policy._trace_action_index(preference_trace, 12)
    assert policy.register_entry_preference_trace(preference_trace)["registered"] is True
    policy.apply_entry_preference_credit(
        preference_trace.trace_id,
        replace(_record(preference_trace.trace_id), selected_action=selected),
    )

    lifecycle_trace = policy.act(np.linspace(-0.2, 0.5, INPUT_COUNT, dtype=float))
    lifecycle_output = np.full(OUTPUT_COUNT, -1.0, dtype=float)
    lifecycle_output[MARKET_BUY] = 1.0
    lifecycle_details = dict(lifecycle_trace.action_details)
    lifecycle_details["direct_rnn_rl_policy_action_index"] = MARKET_BUY
    lifecycle_trace = replace(
        lifecycle_trace, selected_idx=MARKET_BUY,
        selected_action=lifecycle_output, action_details=lifecycle_details,
    )
    assert policy._queue_direct_rnn_full_rl_lifecycle(
        lifecycle_trace, 0.8, 0.8, update_scale=1.0
    ) == 1
    event = policy.growth_events[-1]
    assert event["timing_first"]["enabled"] is True
    assert event["timing_first"]["eligible_count"] == 1
    assert event["timing_first"]["gradient_l1"] > 0.0
    assert event["timing_first"]["entry_lifecycle_actor_frozen_count"] == 1
    assert event["entry_preference_eligible_count"] == 0


def test_invalid_entry_utility_mode_fails_closed() -> None:
    with np.testing.assert_raises(ValueError):
        replay_entry_utilities(_record("invalid"), mode="future_leaking_v1")


def test_preference_loss_uses_the_same_full_rl_adamw_writer() -> None:
    policy = _policy()
    module = policy.modules[0]
    before = module.w_out[:7].copy()
    before_quality_w = module.conditioned_quality_w.copy()
    before_quality_b = module.conditioned_quality_b.copy()
    before_auxiliary = module.auxiliary_w.copy()
    before_reconstruction = module.auxiliary_reconstruction_w.copy()

    for index in range(1):
        trace = policy.act(np.linspace(-0.3, 0.4, INPUT_COUNT, dtype=float))
        trace_id = f"entry-pref-{index}"
        # The actual bridge trace id is immutable provenance; a unit test may
        # replace only this identifier before routing delayed evidence.
        trace = replace(trace, trace_id=trace_id)
        registered = policy.register_entry_preference_trace(trace)
        assert registered["registered"] is True
        record = replace(
            _record(trace_id),
            selected_action=policy._trace_action_index(trace, 12),
            sample_scope="entry_marginal_v2" if index else "flat_v1",
            positions_before=2 if index else 0,
            orders_before=1 if index else 0,
        )
        result = policy.apply_entry_preference_credit(trace_id, record)
        assert result in {0, 1}

    # Preference-only evidence must not own an AdamW step.  Add one canonical
    # lifecycle item so the two-item batch has a 50/50 evidence quota.
    assert module.rnn_optimizer_step == 0
    lifecycle_trace = policy.act(np.linspace(-0.2, 0.5, INPUT_COUNT, dtype=float))
    lifecycle_output = np.full(OUTPUT_COUNT, -1.0, dtype=float)
    lifecycle_output[MARKET_BUY] = 1.0
    lifecycle_details = dict(lifecycle_trace.action_details)
    lifecycle_details["direct_rnn_rl_policy_action_index"] = MARKET_BUY
    lifecycle_trace = replace(
        lifecycle_trace,
        selected_idx=MARKET_BUY,
        selected_action=lifecycle_output,
        action_details=lifecycle_details,
    )
    assert policy._queue_direct_rnn_full_rl_lifecycle(
        lifecycle_trace, 0.8, 0.8, update_scale=1.0
    ) == 1

    event = policy.growth_events[-1]
    assert event["type"] == "direct_rnn_full_rl_batch"
    assert event["entry_preference_enabled"] is True
    assert event["entry_preference_eligible_count"] == 1
    assert event["entry_preference_loss"] > 0.0
    assert event["entry_preference_gradient_l1"] > 0.0
    assert event["counterfactual_representation_enabled"] is True
    assert event["counterfactual_representation_eligible_count"] == 6
    assert event["counterfactual_representation_loss"] > 0.0
    assert event["counterfactual_representation_gradient_l1"] > 0.0
    assert event["counterfactual_representation_conditions"]
    assert event["multi_task_auxiliary_enabled"] is True
    assert event["multi_task_auxiliary_sample_count"] == 1
    assert event["multi_task_auxiliary_eligible_targets"] >= 13
    assert event["multi_task_auxiliary_loss"] > 0.0
    assert event["multi_task_auxiliary_reconstruction_loss"] > 0.0
    assert event["entry_preference_sample_scopes"] == {"flat_v1": 1}
    queue_receipts = [
        item for item in policy.growth_events
        if item.get("type") == "direct_rnn_entry_preference_queued"
    ]
    assert len(queue_receipts) == 1
    assert queue_receipts[0]["entry_preference_sample_scope"] == "flat_v1"
    assert queue_receipts[0]["entry_preference_utility_mode"] == "path_quality_v1"
    assert queue_receipts[0]["batch_triggered"] is False
    assert module.rnn_optimizer_step == 1
    assert np.linalg.norm(module.w_out[:7] - before) > 0.0
    assert np.linalg.norm(module.conditioned_quality_w - before_quality_w) > 0.0
    assert np.linalg.norm(module.conditioned_quality_b - before_quality_b) > 0.0
    assert np.linalg.norm(module.auxiliary_w - before_auxiliary) > 0.0
    assert np.linalg.norm(module.auxiliary_reconstruction_w - before_reconstruction) > 0.0
    # The private optimizer refuses any second/manual writer.
    rejected = module.apply_rnn_adamw_cosine(
        {"w_out": np.ones_like(module.w_out)}, source="entry_preference_manual"
    )
    assert rejected["updated"] == 0.0


def test_preference_delivery_ack_is_independent_from_optimizer_step() -> None:
    policy = _policy()
    trace = policy.act(np.linspace(-0.25, 0.35, INPUT_COUNT, dtype=float))
    trace_id = "preference-explicit-ack"
    trace = replace(trace, trace_id=trace_id)
    selected = policy._trace_action_index(trace, 12)
    assert policy.register_entry_preference_trace(trace)["registered"] is True

    result = policy.apply_entry_preference_credit_detailed(
        trace_id, replace(_record(trace_id), selected_action=selected)
    )

    assert result == {
        "accepted": True,
        "queued": True,
        "updates": 0,
        "retryable": False,
        "reason": "queued",
        "selected_action": selected,
    }
    assert trace_id not in policy._entry_preference_trace_cache
    assert any(
        row.get("type") == "direct_rnn_entry_preference_queued"
        for row in policy.growth_events
    )
    duplicate = policy.apply_entry_preference_credit_detailed(
        trace_id, replace(_record(trace_id), selected_action=selected)
    )
    assert duplicate["accepted"] is True
    assert duplicate["queued"] is False
    assert duplicate["reason"] == "duplicate_receipt"


def test_preference_validation_failure_does_not_pop_immutable_trace() -> None:
    policy = _policy()
    trace = policy.act(np.linspace(-0.15, 0.45, INPUT_COUNT, dtype=float))
    trace_id = "preference-mismatch"
    trace = replace(trace, trace_id=trace_id)
    selected = policy._trace_action_index(trace, 12)
    assert policy.register_entry_preference_trace(trace)["registered"] is True
    wrong = MARKET_BUY if selected != MARKET_BUY else BUY_STOP

    result = policy.apply_entry_preference_credit_detailed(
        trace_id, replace(_record(trace_id), selected_action=wrong)
    )

    assert result["accepted"] is False
    assert result["retryable"] is False
    assert result["reason"] == "selected_action_mismatch"
    assert trace_id in policy._entry_preference_trace_cache


def test_mt5_set_exposes_the_same_fixed_counterfactual_horizon(tmp_path) -> None:
    set_path = write_tester_set(
        MT5BacktestConfig(
            entry_preference_horizon_bars=48,
            entry_abstraction_horizon_bars=600,
            entry_preference_include_hold_snapshots=True,
        ),
        tmp_path / "entry_preference.set",
    )
    content = set_path.read_text(encoding="utf-8")
    assert "EntryPreferenceFile=self_growing_entry_preference.csv" in content
    assert "EntryPreferenceHorizonBars=48" in content
    assert "EntryAbstractionHorizonBars=600" in content
    assert "EntryPreferenceIncludeHoldSnapshots=true" in content


def test_ea_releases_five_path_at_48_bars_but_keeps_legacy_abstraction_horizon() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    ).read_text(encoding="utf-8")

    assert 'StringFind(g_entryPreference[i].traceId, "five:") == 0' in source
    assert "int releaseHorizon = fivePatternSnapshot ? pathHorizon : abstractionHorizon;" in source
    assert "observedBars >= releaseHorizon" in source
    assert "observedBars >= abstractionHorizon" not in source


def test_hold_snapshot_coverage_is_explicit_and_management_stays_excluded() -> None:
    eligible = MT5StrategyTesterEvaluator._entry_preference_snapshot_eligible
    assert eligible(HOLD_ACTION, include_hold=True) is True
    assert eligible(HOLD_ACTION, include_hold=False) is False
    for action in range(1, 7):
        assert eligible(action, include_hold=False) is True
    for action in range(7, 12):
        assert eligible(action, include_hold=True) is False


def test_hold_decision_receives_counterfactual_credit_and_enters_retention() -> None:
    policy = _policy()
    policy.direct_rnn_rl_trade_hold_separation_enabled = True
    policy.direct_rnn_rl_retention_enabled = True
    trace = policy.act(np.linspace(-0.1, 0.2, INPUT_COUNT, dtype=float))
    trace_id = "hold-preference"
    selected_action = np.full(OUTPUT_COUNT, -1.0, dtype=float)
    selected_action[HOLD_ACTION] = 1.0
    details = dict(trace.action_details)
    details["direct_rnn_rl_policy_action_index"] = HOLD_ACTION
    details["bridge_action"] = HOLD_ACTION
    details["flat_idle_non_feedback"] = True
    details["action_attribution_source"] = "hold_no_execution"
    details["wire_noop"] = False
    details["no_neural_command"] = False
    trace = replace(
        trace,
        trace_id=trace_id,
        selected_idx=HOLD_ACTION,
        selected_action=selected_action,
        action_details=details,
    )

    assert policy.register_entry_preference_trace(trace)["registered"] is True
    record = replace(_record(trace_id), selected_action=HOLD_ACTION)
    assert policy.apply_entry_preference_credit(trace_id, record) == 0

    queued = policy._direct_rnn_rl_batches[-1]
    assert queued["entry_preference_trade_target_basis"] == "uniform_legal_entries"
    assert queued["entry_preference_retention_anchor"] is True
    assert queued["entry_preference_hold_target_probability"] >= 0.55
    retention_stats = policy.direct_rnn_retention_store.stats()
    assert retention_stats["preference_episodes"] == 1
    assert retention_stats["preference_anchors"] == 1
    episode = next(iter(policy.direct_rnn_retention_store.episodes.values()))
    assert episode.bucket[:3] == ("preference", "unknown", str(HOLD_ACTION))
    assert episode.preference_anchor is True
    assert episode.anchor is True


def test_transport_hold_and_safety_rewrite_do_not_become_preference_credit() -> None:
    policy = _policy()
    trace = policy.act(np.linspace(-0.1, 0.2, INPUT_COUNT, dtype=float))
    details = dict(trace.action_details)
    details.update({
        "bridge_action": HOLD_ACTION,
        "direct_rnn_rl_policy_action_index": MARKET_BUY,
        "semantic_safety_veto_applied": True,
    })
    rewritten = replace(trace, action_details=details)
    assert policy._entry_preference_trace_action_index(rewritten, 12) == -1

    details.update({
        "direct_rnn_rl_policy_action_index": HOLD_ACTION,
        "semantic_safety_veto_applied": False,
        "wire_noop": True,
        "no_neural_command": True,
    })
    transport_only = replace(trace, action_details=details)
    assert policy._entry_preference_trace_action_index(transport_only, 12) == -1


def test_incremental_counterfactual_reader_does_not_rescan_old_rows(tmp_path) -> None:
    path = tmp_path / "self_growing_entry_preference.csv"
    path.write_text("trace_id\tsequence\nfirst\t1\n", encoding="utf-8")
    reader = _IncrementalCsvTail()
    assert reader.read_rows(path) == [{"trace_id": "first", "sequence": "1"}]
    assert reader.read_rows(path) == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write("second\t2\n")
    assert reader.read_rows(path) == [{"trace_id": "second", "sequence": "2"}]


def test_entry_marginal_v2_record_preserves_portfolio_provenance() -> None:
    row = {
        "trace_id": "marginal-entry",
        "sequence": "7",
        "decision_time": "2025.01.01 00:00",
        "selected_action": str(MARKET_BUY),
        "sample_scope": "entry_marginal_v2",
        "positions_before": "2",
        "orders_before": "1",
        "bid": "100", "ask": "101", "atr": "1", "sl_atr": "1", "rr": "2",
        "entry_prices_pipe": "101|100|100|102|102|99",
        "path_pipe": "100.5|98.5|99.5|1",
    }
    record = parse_entry_preference_row(row)
    assert record is not None
    assert record.sample_scope == "entry_marginal_v2"
    assert record.positions_before == 2
    assert record.orders_before == 1


def test_entry_preference_v3_preserves_sparse_long_horizon_targets() -> None:
    row = {
        "trace_id": "long-horizon", "sequence": "8", "decision_time": "2025.01.01 00:00",
        "selected_action": "1", "bid": "100", "ask": "100.2", "atr": "1",
        "sl_atr": "1", "rr": "2", "entry_prices_pipe": "100.2|100|99.8|100.4|100.5|99.7",
        "path_pipe": "100.4|99.8|100.1|0.2", "observed_bars": "600",
        "future_close_pipe": "100.3|100.4|101|100.8|102",
        "future_volatility_pipe": "0.01|0.02|0.03|0.04|0.05",
    }
    record = parse_entry_preference_row(row)
    assert record is not None
    assert record.observed_bars == 600
    assert record.future_closes == (100.3, 100.4, 101.0, 100.8, 102.0)
    assert record.future_volatilities == (0.01, 0.02, 0.03, 0.04, 0.05)
