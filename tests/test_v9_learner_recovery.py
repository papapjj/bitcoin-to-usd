from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from self_growing_trader.learner_recovery import (
    ActionEvidenceCell,
    ActionRecoveryLearner,
    LEARNER_RECOVERY_SCHEMA_VERSION,
    MARKET_BUY,
    MARKET_SELL,
    trace_context,
)
from self_growing_trader.live_observatory import LiveEventReader, LiveObservabilityConfig, LiveTelemetryWriter
from self_growing_trader.desktop_app import summarize_live_state
from self_growing_trader.config import LearningIntegrityConfig
from self_growing_trader.chuyen_gia_policy import _v9_replacement_trace
from self_growing_trader.self_growing_runner import SelfGrowingPolicy, StepTrace


def _trace(*, trace_id: str = "trace-v9", selected: int = 0, context: np.ndarray | None = None) -> StepTrace:
    ctx = np.ones(46, dtype=float) if context is None else np.asarray(context, dtype=float)
    candidates = np.zeros((3, 20), dtype=float)
    candidates[0, 0] = 1.0  # HOLD
    candidates[1, MARKET_BUY] = 1.0
    candidates[2, MARKET_SELL] = 1.0
    selected_action = candidates[0].copy()
    selected_action[:] = 0.0
    selected_action[int(selected)] = 1.0
    return StepTrace(
        trace_id=trace_id,
        timestamp=1.0,
        raw_obs=np.zeros(119, dtype=float),
        encoded_state=np.zeros(119, dtype=float),
        memory_before=np.zeros(8, dtype=float),
        memory_after=np.zeros(8, dtype=float),
        selected_idx=0,
        selected_action=selected_action,
        candidate_actions=candidates,
        candidate_logits=np.array([0.4, 0.2, 0.1], dtype=float),
        candidate_scores=np.array([0.4, 0.2, 0.1], dtype=float),
        selected_action_log_prob=-0.5,
        predicted_next_state=np.zeros(119, dtype=float),
        critic_value=0.4,
        confidence=0.4,
        novelty=0.0,
        conflict=0.0,
        risk=0.0,
        gate_values=(),
        prediction_error=0.0,
        td_delta=0.0,
        action_details={},
        canonical_context_key=ctx.copy(),
        allowed_action_mask=tuple([True] * 12),
    )


def test_v9_negative_sell_evidence_does_not_create_global_hold_penalty_and_buy_can_replace_hold() -> None:
    learner = ActionRecoveryLearner(
        context_size=46,
        min_replacement_support=2.0,
        min_replacement_lcb=0.02,
        shadow_weight=1.0,
        lcb_z=0.25,
    )
    context = np.ones(46, dtype=float)
    # Build context-local BUY-positive and SELL-negative evidence.  The action
    # cells are separate: SELL losses cannot lower BUY or write a HOLD bias.
    for sequence in range(6):
        learner.update(MARKET_BUY, context, 0.80, weight=1.0, sequence=sequence, trace_id=f"b{sequence}", is_real=False)
        learner.update(MARKET_SELL, context, -0.80, weight=1.0, sequence=sequence, trace_id=f"s{sequence}", is_real=False)

    trace = _trace(context=context, selected=0)
    result = learner.rerank(trace, allowed_actions={0, MARKET_BUY, MARKET_SELL})

    assert result["replacement_applied"] is True
    assert result["replacement_action"] == MARKET_BUY
    assert result["hold_reason"] == "REPLACED_HOLD_WITH_VERIFIED_ACTION"
    assert result["action_values"][str(MARKET_SELL)]["lcb"] < 0.0
    assert result["action_values"][str(MARKET_BUY)]["lcb"] > 0.0
    # The recovery learner has no global HOLD or entry penalty state at all.
    assert "global_hold_penalty" not in result


def test_v9_context_negative_sell_keeps_entry_available_without_verified_replacement() -> None:
    learner = ActionRecoveryLearner(
        context_size=46,
        min_replacement_support=3.0,
        min_replacement_lcb=0.08,
        shadow_weight=1.0,
        lcb_z=0.25,
    )
    context = np.ones(46, dtype=float)
    for sequence in range(5):
        learner.update(MARKET_SELL, context, -0.80, weight=1.0, sequence=sequence, trace_id=f"s{sequence}", is_real=True)

    trace = _trace(context=context, selected=MARKET_SELL)
    result = learner.rerank(trace, allowed_actions={0, MARKET_BUY, MARKET_SELL})

    assert result["base_action_local_negative"] is True
    assert result["replacement_applied"] is False
    assert result["hold_reason"] == "NO_VERIFIED_ENTRY_REPLACEMENT"
    assert "global_hold_penalty" not in result


def test_v13_real_negative_local_action_is_vetoed_to_hold_without_global_penalty() -> None:
    learner = ActionRecoveryLearner(
        context_size=46,
        min_replacement_support=3.0,
        min_replacement_lcb=0.10,
        shadow_weight=1.0,
        lcb_z=0.25,
        negative_veto_enabled=True,
        negative_veto_min_real_support=3.0,
        negative_veto_lcb=-0.04,
    )
    context = np.ones(46, dtype=float)
    for sequence in range(5):
        learner.update(MARKET_SELL, context, -0.80, weight=1.0, sequence=sequence, trace_id=f"real-{sequence}", is_real=True)

    trace = _trace(context=context, selected=MARKET_SELL)
    recovery = learner.rerank(trace, allowed_actions={0, MARKET_BUY, MARKET_SELL})
    vetoed = _v9_replacement_trace(trace, recovery)

    assert recovery["replacement_applied"] is False
    assert recovery["negative_veto_to_hold"] is True
    assert recovery["hold_reason"] == "HOLD_LOCAL_REAL_NEGATIVE_VETO"
    assert int(np.argmax(vetoed.selected_action[:12])) == 0
    assert vetoed.action_details["learner_recovery_negative_veto_applied"] is True
    assert vetoed.action_details["learner_recovery_negative_veto_from"] == MARKET_SELL
    assert "global_hold_penalty" not in recovery


def test_v9_one_strong_real_sell_loss_marks_repeat_sell_without_hard_block() -> None:
    learner = ActionRecoveryLearner(
        context_size=46,
        min_replacement_support=3.0,
        min_replacement_lcb=0.08,
        shadow_weight=0.50,
        lcb_z=1.28,
    )
    context = np.ones(46, dtype=float)
    learner.update(MARKET_SELL, context, -0.30, weight=0.5, sequence=1, trace_id="shadow-sell", is_real=False)
    learner.update(MARKET_SELL, context, -1.00, weight=1.0, sequence=2, trace_id="real-sell-sl", is_real=True)

    trace = _trace(context=context, selected=MARKET_SELL)
    result = learner.rerank(trace, allowed_actions={0, MARKET_BUY, MARKET_SELL})

    assert result["base_action_local_negative"] is True
    assert result["replacement_applied"] is False
    assert result["hold_reason"] == "NO_VERIFIED_ENTRY_REPLACEMENT"
    assert result["action_values"][str(MARKET_SELL)]["support"] == 1.5
    assert result["action_values"][str(MARKET_SELL)]["real_support"] == 1.0


def test_v9_shadow_learning_updates_both_directional_counterfactuals_once_per_trace() -> None:
    learner = ActionRecoveryLearner(context_size=46, shadow_weight=1.0)
    trace = _trace(trace_id="shadow-1")
    next_obs = np.zeros(119, dtype=float)
    next_obs[0] = 0.65
    next_obs[15] = 0.10

    first = learner.observe_shadow_transition(trace, next_obs, sequence=11)
    duplicate = learner.observe_shadow_transition(trace, next_obs, sequence=12)
    values = learner.query(np.ones(46, dtype=float), allowed_actions={MARKET_BUY, MARKET_SELL})

    assert first["updated"] is True
    assert duplicate["updated"] is False
    assert values[MARKET_BUY]["mean_net_r"] > 0.0
    assert values[MARKET_SELL]["mean_net_r"] < 0.0


def test_v10_pending_lifecycle_keeps_exact_entry_action_separate_from_market_sell() -> None:
    sell_limit = 4
    sell_stop = 6
    learner = ActionRecoveryLearner(context_size=46, merge_similarity=0.80)
    context = np.linspace(-0.5, 0.5, 46)
    trace = _trace(context=context, selected=sell_stop)
    trace = replace(trace, action_details={"emitted_action": sell_stop, "action_name": "SELL_STOP"})

    result = learner.observe_lifecycle(trace, -1.0, close_reason="SL", sequence=7)
    values = learner.query(context, allowed_actions={sell_limit, sell_stop})

    assert result["updated"] is True
    assert result["recorded_action"] == sell_stop
    assert values[sell_stop]["real_support"] == 1.0
    assert values[sell_limit]["real_support"] == 0.0


def test_v11_position_identity_deduplicates_bridge_and_final_ledger_credit() -> None:
    learner = ActionRecoveryLearner(context_size=46)
    context = np.ones(46, dtype=float)
    trace = replace(
        _trace(context=context, selected=MARKET_BUY),
        action_details={"emitted_action": MARKET_BUY},
        execution_details={"position_ticket": "position-42"},
    )

    first = learner.observe_lifecycle(
        trace,
        0.8,
        close_reason="TP",
        sequence=10,
        signature="bridge-close-signature",
    )
    second = learner.observe_lifecycle(
        replace(trace, trace_id="management-trace"),
        0.8,
        close_reason="TP",
        sequence=11,
        signature="final-ledger-signature",
        lifecycle_identity="position-42",
    )

    assert first["updated"] is True
    assert second["updated"] is False
    assert second["reason"] == "duplicate_lifecycle_identity"
    assert learner.lifecycle_updates == 1
    assert learner.summary()["schema_version"] == LEARNER_RECOVERY_SCHEMA_VERSION


def test_v12_loading_legacy_recovery_evidence_discounts_duplicate_support() -> None:
    learner = ActionRecoveryLearner(context_size=46)
    learner.cells = {MARKET_BUY: [ActionEvidenceCell(np.ones(46), MARKET_BUY, support=4.0)]}
    learner.schema_version = 10
    learner.ensure_compatible()

    assert learner.schema_version == LEARNER_RECOVERY_SCHEMA_VERSION
    assert learner.cells[MARKET_BUY][0].support == 2.0
    assert learner.shadow_updates == 0
    assert learner.lifecycle_updates == 0


def test_v12_intraday_reset_keeps_lifecycle_identity_markers() -> None:
    learner = ActionRecoveryLearner(context_size=46)
    learner.seen_shadow_trace_ids.add("shadow-1")
    learner.seen_lifecycle_signatures.add("close-signature")
    learner.seen_lifecycle_ids.add("position:42")

    learner.reset_intraday_state()

    assert learner.seen_shadow_trace_ids == set()
    assert learner.seen_lifecycle_signatures == {"close-signature"}
    assert learner.seen_lifecycle_ids == {"position:42"}


def test_v10_negative_pending_action_exposes_soft_penalty_without_masking_entry() -> None:
    sell_stop = 6
    learner = ActionRecoveryLearner(
        context_size=46,
        min_replacement_support=3.0,
        min_replacement_lcb=0.08,
        lcb_z=0.25,
    )
    context = np.ones(46, dtype=float)
    for sequence in range(5):
        learner.update(sell_stop, context, -0.80, weight=1.0, sequence=sequence, trace_id=f"stop-{sequence}", is_real=True)

    candidates = np.zeros((2, 20), dtype=float)
    candidates[0, 0] = 1.0
    candidates[1, sell_stop] = 1.0
    trace = replace(
        _trace(context=context, selected=sell_stop),
        candidate_actions=candidates,
        candidate_scores=np.array([0.2, 0.8], dtype=float),
        selected_idx=1,
    )

    result = learner.rerank(trace, allowed_actions={0, sell_stop})

    assert result["base_action_local_negative"] is True
    assert result["replacement_applied"] is False
    assert result["soft_action_penalty"] > 0.0
    assert sell_stop in result["real_lifecycle_entry_actions"]


def test_v10_similar_cells_are_consolidated_with_sufficient_statistics() -> None:
    learner = ActionRecoveryLearner(context_size=46, merge_similarity=0.75)
    first = np.ones(46, dtype=float)
    second = first.copy()
    second[:10] = 0.0
    learner.cells = {
        MARKET_SELL: [
            ActionEvidenceCell(first, MARKET_SELL, support=2.0, mean=-0.5, m2=0.2, real_support=2.0),
            ActionEvidenceCell(second, MARKET_SELL, support=2.0, mean=0.5, m2=0.2, real_support=2.0),
        ]
    }
    learner._cells_consolidated_similarity = -1.0

    learner.ensure_compatible()

    assert len(learner.cells[MARKET_SELL]) == 1
    assert learner.cells[MARKET_SELL][0].support == 4.0
    assert learner.cells[MARKET_SELL][0].real_support == 4.0



def test_v9_zero_filled_legacy_context_does_not_merge_every_market_into_one_cell() -> None:
    trace = _trace(context=np.zeros(46, dtype=float))
    trace = replace(trace, raw_obs=np.linspace(-1.0, 1.0, 119, dtype=float))
    recovered = trace_context(trace)
    assert recovered.size > 0
    assert np.linalg.norm(recovered) > 0.0

def test_v9_policy_exposes_recovery_audit_without_breaking_v8_contract() -> None:
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, hidden_size=6, seed=9)
    trace = policy.act(np.zeros(119, dtype=float))

    assert trace.action_details["mechanism_schema_version"] == 8
    assert trace.action_details["learner_recovery_schema_version"] == LEARNER_RECOVERY_SCHEMA_VERSION
    assert trace.action_details["learner_recovery_global_hold_penalty"] == 0.0
    assert trace.action_details["hold_reason"]
    policy.observe_shadow_transition(trace, np.ones(119, dtype=float) * 0.01, sequence=1)
    summary = policy.mechanism_summary()
    assert summary["schema_version"] == 8
    assert summary["learner_recovery_schema_version"] == LEARNER_RECOVERY_SCHEMA_VERSION
    assert "action_recovery" in summary



def test_v9_recovery_thresholds_are_strict_learning_config_not_hidden_constants() -> None:
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, hidden_size=6, seed=19)
    settings = LearningIntegrityConfig(
        action_recovery_enabled=True,
        action_recovery_min_support=4.0,
        action_recovery_min_lcb=0.12,
        action_recovery_merge_similarity=0.91,
        action_recovery_shadow_weight=0.40,
        action_recovery_max_cells_per_action=64,
        action_recovery_negative_veto_enabled=True,
        action_recovery_negative_veto_min_real_support=5.0,
        action_recovery_negative_veto_lcb=-0.12,
    )
    policy.configure_learning_integrity(settings)
    learner = policy.action_recovery_learner
    assert policy.action_recovery_enabled is True
    assert learner.min_replacement_support == 4.0
    assert learner.min_replacement_lcb == 0.12
    assert learner.merge_similarity == 0.91
    assert learner.shadow_weight == 0.40
    assert learner.max_cells_per_action == 64
    assert learner.negative_veto_enabled is True
    assert learner.negative_veto_min_real_support == 5.0
    assert learner.negative_veto_lcb == -0.12


def test_v9_recovery_replacement_can_be_disabled_without_disabling_evidence_learning() -> None:
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, hidden_size=6, seed=21)
    settings = LearningIntegrityConfig(
        action_recovery_enabled=False,
        action_recovery_learning_enabled=True,
    )
    policy.configure_learning_integrity(settings)

    trace = policy.act(np.zeros(119, dtype=float))
    assert policy.action_recovery_enabled is False
    assert policy.action_recovery_learning_enabled is True
    assert trace.action_details["hold_reason"] == "RECOVERY_DISABLED"
    assert trace.action_details["learner_recovery_shadow_only"] is True
    assert trace.action_details["learner_recovery_learning_enabled"] is True
    assert trace.action_details["shadow_learning_active"] is True

    update = policy.observe_shadow_transition(
        trace,
        np.array([0.40] + [0.0] * 118, dtype=float),
        sequence=1,
    )
    assert update["shadow"]["updated"] is True
    # One shadow transition updates both counterfactual market directions.
    assert policy.action_recovery_learner.summary()["shadow_updates"] == 2


def test_v17_zero_lcb_veto_boundary_is_preserved_and_shared_by_base_classifier() -> None:
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, hidden_size=6, seed=23)
    settings = LearningIntegrityConfig(
        action_recovery_negative_veto_enabled=True,
        action_recovery_negative_veto_min_real_support=2.0,
        action_recovery_negative_veto_lcb=0.0,
    )

    policy.configure_learning_integrity(settings)
    learner = policy.action_recovery_learner
    learner.ensure_compatible()

    assert learner.negative_veto_lcb == 0.0
    assert learner.negative_lcb_threshold == 0.0

    context = np.ones(46, dtype=float)
    learner.update(MARKET_BUY, context, -0.01, weight=1.0, sequence=1, trace_id="loss-1", is_real=True)
    learner.update(MARKET_BUY, context, 0.00, weight=1.0, sequence=2, trace_id="flat-2", is_real=True)
    result = learner.rerank(_trace(context=context, selected=MARKET_BUY), allowed_actions={0, MARKET_BUY})

    assert result["base_action_local_negative"] is True
    assert result["negative_veto_to_hold"] is True
    assert result["hold_reason"] == "HOLD_LOCAL_REAL_NEGATIVE_VETO"


def test_v9_observe_transition_close_feeds_real_lifecycle_to_recovery() -> None:
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, hidden_size=6, seed=29)
    trace = _trace(trace_id="mt5-close-1", selected=MARKET_SELL)
    trace = replace(
        trace,
        raw_obs=np.linspace(-1.0, 1.0, 119, dtype=float),
        action_details={"emitted_action": MARKET_SELL, "action_name": "MARKET_SELL"},
        execution_details={
            "bridge_observed": True,
            "closed": True,
            "action_feedback_eligible": True,
            "memory_eligible": True,
            "ack_sequence": "17",
            "deal_ticket": "901",
            "position_ticket": "701",
            "profit": "-100.0",
            "commission": "0.0",
            "swap": "0.0",
            "initial_risk": "100.0",
            "close_reason": "SL",
        },
    )

    policy.observe_transition(trace, np.zeros(119, dtype=float), reward=-0.1, cost=0.0, done=False)
    policy.observe_transition(trace, np.zeros(119, dtype=float), reward=-0.1, cost=0.0, done=False)

    summary = policy.action_recovery_learner.summary()
    sell = summary["action_cells"][str(MARKET_SELL)]
    assert summary["lifecycle_updates"] == 1
    assert sell["real_support"] == 1.0
    assert any(
        event.get("type") == "learner_recovery_lifecycle_evidence"
        and event.get("source") == "observe_transition_close"
        and event.get("realized_r") == -1.0
        for event in policy.growth_events
    )

def test_v9_telemetry_status_and_bridge_heartbeat_make_stale_detection_visible(tmp_path: Path) -> None:
    path = tmp_path / "neural_live.jsonl"
    writer = LiveTelemetryWriter(path=str(path), enabled=True, run_id="v9-test", source="mt5")
    assert writer.emit("bridge_heartbeat", {"stage": "request_seen", "sequence": 17}) is True
    assert writer.emit("bridge_heartbeat", {"stage": "action_written", "sequence": 17, "action": 0}) is True

    status_path = path.with_suffix(path.suffix + ".status.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["event_sequence"] == 2
    assert status["last_event_type"] == "bridge_heartbeat"

    reader = LiveEventReader(path)
    reader.refresh()
    state = reader.state()
    assert state["latest"]["bridge_heartbeat"]["payload"]["sequence"] == 17
    assert state["writer_status"]["event_sequence"] == 2
    summary = summarize_live_state(state)
    assert summary["bridge_sequence"] == 17
    assert summary["bridge_stage"] == "action_written"


def test_v9_policy_bridge_heartbeat_exposes_execution_action_context(tmp_path: Path) -> None:
    path = tmp_path / "neural_live.jsonl"
    policy = SelfGrowingPolicy.seed(seed=91)
    policy.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(path), reset_on_process_start=True),
        run_id="v9-action-test",
        source="mt5",
    )

    policy.emit_bridge_heartbeat(
        stage="action_written",
        sequence=23,
        request_event_id="event-23",
        trace_id="trace-23",
        action=MARKET_BUY,
        metadata={
            "wire_action": MARKET_BUY,
            "policy_action": 0,
            "legal_action": 0,
            "emitted_action": MARKET_BUY,
            "execution_bootstrap": True,
            "action_masked": False,
        },
    )

    state = LiveEventReader(path).state()
    payload = state["latest"]["bridge_heartbeat"]["payload"]
    assert payload["sequence"] == 23
    assert payload["action"] == MARKET_BUY
    assert payload["action_name"] == "MARKET_BUY"
    assert payload["execution_action_event"] is True
    assert payload["metadata"]["policy_action_name"] == "HOLD"
    assert payload["metadata"]["emitted_action_name"] == "MARKET_BUY"
    assert payload["metadata"]["execution_bootstrap"] is True
    assert payload["topology_summary"]["node_count"] > 0
