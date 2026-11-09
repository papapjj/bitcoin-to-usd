from __future__ import annotations

import json
import hashlib
import io
import os
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from self_growing_trader.codex_teacher import (
    CodexGradeExecutionError,
    CodexTeacherPreflightError,
    CodexTeacherRewardModel,
    RUBRIC_VERSION,
    build_codex_teaching_packet,
    preflight_codex_teacher,
    grade_with_codex,
    resolve_codex_executable,
    validate_codex_teacher_receipt,
)
from self_growing_trader.rlhf_policy_loop import (
    RLHFPolicyRollout, causal_policy_feature_vector, prepare_rlhf_rollout_batch,
)
from self_growing_trader.mt5_runner import (
    MT5BacktestConfig,
    MT5PositionLifecycle,
    MT5StrategyTesterEvaluator,
    _legacy_codex_lifecycle_from_close_details,
)
from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    _persist_codex_recovery_snapshot,
    _persist_paired_rejected_research_candidate,
    _persist_codex_teacher_preflight_failure_status,
    _five_optimizer_state_for_parameter,
    _five_optimizer_state_remove_for_parameter,
    _five_optimizer_state_set_for_parameter,
)
from self_growing_trader.five_shared_policy import (
    FactorizedActionHeads,
    FiveAdam,
    SharedFiveTradingPolicy,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT


def test_codex_recovery_snapshot_is_atomic_and_restores_frozen_reference(tmp_path) -> None:
    state = SimpleNamespace(
        weight={"action_head": [1.0, -2.0]},
        five_rlhf_reference_network=SimpleNamespace(kind="reference"),
        five_rlhf_reference_memories=["sentinel"],
        five_rlhf_reference_policy_sha256="reference-hash",
    )
    output = tmp_path / "candidate.pkl"

    skipped = _persist_codex_recovery_snapshot(
        state,
        output,
        update_count=1,
        interval_updates=2,
    )
    assert skipped["saved"] is False
    assert not tmp_path.joinpath("candidate.recovery.pkl").exists()

    saved = _persist_codex_recovery_snapshot(
        state,
        output,
        update_count=2,
        interval_updates=2,
    )
    checkpoint = Path(str(saved["path"]))
    metadata = Path(str(saved["metadata"]))
    assert saved["saved"] is True
    assert checkpoint.is_file()
    assert metadata.is_file()
    assert state.five_rlhf_reference_policy_sha256 == "reference-hash"
    assert state.five_rlhf_reference_memories == ["sentinel"]
    assert json.loads(metadata.read_text(encoding="utf-8"))["promotable"] is False
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == saved["checkpoint_sha256"]

    with checkpoint.open("rb") as handle:
        recovered = __import__("pickle").load(handle)
    assert recovered.weight == state.weight
    assert recovered.five_rlhf_reference_network is None
    assert recovered.five_rlhf_reference_memories == []
    assert recovered.five_rlhf_reference_policy_sha256 == ""


def test_codex_recovery_snapshot_strips_optimizer_only_in_artifact(tmp_path) -> None:
    optimizer_state = {1: {"m": "moment", "v": "variance"}}
    state = SimpleNamespace(
        weight={"action_head": [1.0, -2.0]},
        five_ppo_trainer=SimpleNamespace(
            optimizer=SimpleNamespace(state=optimizer_state),
        ),
        five_rlhf_reference_network=None,
        five_rlhf_reference_memories=[],
        five_rlhf_reference_policy_sha256="",
    )
    output = tmp_path / "candidate.pkl"
    settings = SimpleNamespace(
        five_rlhf_strip_optimizer_state_for_oos=True,
        five_rlhf_allow_optimizer_state_reinitialization_for_research=True,
    )

    saved = _persist_codex_recovery_snapshot(
        state,
        output,
        update_count=8,
        interval_updates=8,
        learning_settings=settings,
    )

    assert saved["saved"] is True
    assert saved["optimizer_state_stripped"] is True
    assert state.five_ppo_trainer.optimizer.state is optimizer_state
    payload = json.loads(Path(str(saved["metadata"])).read_text(encoding="utf-8"))
    assert payload["optimizer_state_stripped"] is True
    assert payload["optimizer_resume_mode"] == "research_reinitialized"
    with Path(str(saved["path"])).open("rb") as handle:
        recovered = __import__("pickle").load(handle)
    assert recovered.five_ppo_trainer.optimizer.state == {}
    assert recovered._five_rlhf_optimizer_state_stripped is True


def test_paired_rejected_research_candidate_preserves_trained_state(tmp_path) -> None:
    state = SimpleNamespace(
        weight={"action_head": [1.0, -2.0]},
        five_rlhf_reference_network=SimpleNamespace(kind="reference"),
        five_rlhf_reference_memories=["sentinel"],
        five_rlhf_reference_policy_sha256="reference-hash",
    )
    output = tmp_path / "candidate.pkl"
    phase = {"graded": 3, "updates": 2, "hypotheses_confirmed": 3}

    saved = _persist_paired_rejected_research_candidate(
        state,
        output,
        source_checkpoint_sha256="source-hash",
        paired_validation={"accepted": False, "selected": "source_seed"},
        codex_teacher_phase=phase,
    )

    checkpoint = Path(saved["path"])
    metadata = Path(saved["metadata"])
    assert saved["saved"] is True
    assert saved["promotable"] is False
    assert saved["codex_optimizer_updates"] == 2
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == saved["checkpoint_sha256"]
    assert state.five_rlhf_reference_policy_sha256 == "reference-hash"
    assert state.five_rlhf_reference_memories == ["sentinel"]
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["status"] == "paired_gate_rejected_research_candidate"
    assert payload["source_checkpoint_sha256"] == "source-hash"
    assert payload["codex_teacher_phase"]["updates"] == 2

    with checkpoint.open("rb") as handle:
        recovered = __import__("pickle").load(handle)
    assert recovered.weight == state.weight
    assert recovered.five_rlhf_reference_network is None
    assert recovered.five_rlhf_reference_memories == []
    assert recovered.five_rlhf_reference_policy_sha256 == ""


def writable_test_dir(label: str) -> Path:
    """Keep focused tests off the ACL-restricted global pytest temp root."""
    path = Path.cwd() / "_tmp" / f"codex_teacher_{label}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def evidence(tag: str = ""):
    trace_prefix = f"five:{tag}" if tag else "five:trace"
    traces = []
    for agent in range(5):
        owner = agent == 2
        traces.append(SimpleNamespace(
            trace_id=f"{trace_prefix}:{agent}",
            selected_idx=1 if owner else 0,
            raw_obs=np.zeros(137, dtype=np.float32),
            action_details={
                "five_agent_index": agent,
                "five_action_index": 1 if owner else 0,
                "five_decision_state_sha256": "state",
                "five_risk_geometry_sha256": "geometry",
                "five_reference_policy_sha256": "reference",
                "five_reference_policy_frozen": True,
            },
            account_context={}, execution_details={}, positions_pipe="",
        ))
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": f"{trace_prefix}:2"},
        position_ticket=f"12345-{tag}" if tag else "12345",
        realized_r=0.8, mfe_r=1.1, mae_r=-0.3,
        commission=0.0, swap=0.0, close_reason="tp",
    )
    return lifecycle, traces


def receipt_for(packet, *, confidence=0.9):
    return {
        "packet_sha256": packet["packet_sha256"], "trace_id": packet["trace_id"],
        "position_ticket": packet["position_ticket"], "rubric_version": RUBRIC_VERSION,
        "rationale": "Entry hợp lệ, rủi ro giới hạn và đã đóng bởi TP.",
        "decision_quality": 0.7, "risk_quality": 0.7, "timing_quality": 0.6,
        "execution_quality": 0.8, "teacher_reward": 0.4, "confidence": confidence,
    }


def hypothesis_receipt_for(packet, *, status="confirmed", predicted_sign=1):
    raw = receipt_for(packet)
    packet_impact = packet.get("account_impact", {})
    raw.update({
        "hypothesis": "Action quality predicts a positive or negative realized outcome.",
        "prediction": f"The immutable close should have realized_r sign {predicted_sign}.",
        "experiment": "outcome_direction_v1 compares predicted sign with finalized realized_r.",
        "predicted_reward_sign": predicted_sign,
        "verification_status": status,
        "verification_observed_reward": float(packet["outcome"]["realized_r"]),
        "verification_evidence": "Final close packet contains the observed realized_r used by the test.",
        "intervention_multiplier": 1.1,
        "intervention_target_action": int(packet["action"]),
    })
    if isinstance(packet_impact, dict):
        raw["account_impact"] = {
            key: packet_impact.get(key)
            for key in (
                "schema_version", "status", "verification_ready", "evidence_scope",
                "observed_net_pnl", "observed_realized_r", "balance_delta", "equity_delta",
                "cashflow_delta", "cashflow_deal_ticket",
                "account_mark_attribution",
                "counterfactual_status", "counterfactual_hold_r", "counterfactual_delta_r",
                "counterfactual_horizon_bars", "verification_evidence",
            )
        }
    return raw


def test_packet_requires_exact_confirmed_owner():
    lifecycle, traces = evidence()
    packet = build_codex_teaching_packet(lifecycle, traces)
    assert packet["trace_id"] == "five:trace:2"
    assert packet["outcome"]["realized_r"] == 0.8
    lifecycle.final_close_confirmed = False
    with pytest.raises(ValueError, match="final close"):
        build_codex_teaching_packet(lifecycle, traces)


def test_packet_preserves_zero_exit_requested_action():
    lifecycle, traces = evidence("zero-exit-action")
    lifecycle.exit_requested_action = 0
    packet = build_codex_teaching_packet(lifecycle, traces)
    assert packet["action_outcome_context"]["exit_requested_action"] == 0


def _legacy_evidence(*, reward: float, seed: int, m1_backend: str = "direct_rnn"):
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=2,
        seed=seed,
        m1_backend=m1_backend,
    )
    reference_hash = hashlib.sha256(f"legacy-source-{seed}".encode()).hexdigest()
    installed = policy.install_codex_teacher_reference(reference_hash)
    assert installed["installed"] is True
    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": trace.trace_id},
        position_ticket=f"legacy-{seed}",
        realized_r=reward,
        mfe_r=0.1,
        mae_r=-1.0,
        commission=0.0,
        swap=0.0,
        close_reason="SL" if reward < 0.0 else "TP",
        side=1,
        entry_count=1,
        entry_price=100.0,
        initial_sl_price=99.0,
        initial_tp_price=102.0,
        initial_risk=1.0,
        initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, (trace,))
    raw = {
        **receipt_for(packet),
        "teacher_reward": reward,
        "confidence": 1.0,
    }
    receipt = validate_codex_teacher_receipt(raw, packet, "test-model")
    return policy, trace, packet, CodexTeacherRewardModel(
        packet,
        receipt,
        max_probability_shift=0.05,
    ), reference_hash


def test_legacy_codex_packet_and_negative_update_change_direct_rnn_behavior():
    policy, trace, packet, scorer, reference_hash = _legacy_evidence(
        reward=-0.8,
        seed=811,
    )
    action = int(packet["action"])
    probability_before = float(
        trace.action_details["direct_rnn_rl_action_probabilities"][action]
    )
    result = policy.optimize_codex_teacher_trace_packet(
        (trace,),
        scorer,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )
    assert packet["packet_mode"] == "legacy_direct_rnn_v1"
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_parameter_scope"] == "codex_legacy_action_bias_adamw_v1"
    assert result["rlhf_parameter_hash_before"] != result["rlhf_parameter_hash_after"]
    assert result["rlhf_direction_expected"] == "decrease"
    assert result["rlhf_target_action_probability_after"] < probability_before
    assert float(policy.codex_legacy_action_bias[action]) < 0.0


def test_legacy_codex_positive_update_and_copy_preserve_action_surface():
    policy, trace, packet, scorer, reference_hash = _legacy_evidence(
        reward=0.8,
        seed=812,
    )
    action = int(packet["action"])
    probability_before = float(
        trace.action_details["direct_rnn_rl_action_probabilities"][action]
    )
    result = policy.optimize_codex_teacher_trace_packet(
        (trace,),
        scorer,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_direction_expected"] == "increase"
    assert result["rlhf_target_action_probability_after"] > probability_before
    copied = policy.copy()
    assert copied.codex_legacy_action_bias_enabled is True
    assert copied.codex_legacy_optimizer_update_count == policy.codex_legacy_optimizer_update_count
    next_trace = copied.act(np.zeros(INPUT_COUNT, dtype=float))
    assert float(next_trace.action_details["codex_legacy_action_bias_max_abs"]) > 0.0


def test_legacy_codex_graph_update_reaches_action_head():
    policy, trace, packet, scorer, reference_hash = _legacy_evidence(
        reward=-0.8,
        seed=813,
        m1_backend="m1_graph",
    )
    action = int(packet["action"])
    result = policy.optimize_codex_teacher_trace_packet(
        (trace,),
        scorer,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )

    assert packet["packet_mode"] == "legacy_direct_rnn_v1"
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_direction_expected"] == "decrease"
    assert result["rlhf_target_action_probability_after"] < result["rlhf_target_action_probability_before"]
    assert float(result["rlhf_action_head_update_norm"]) > 0.0
    assert float(result["rlhf_action_head_target_delta"]) < 0.0
    next_trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    details = next_trace.action_details
    assert details["codex_legacy_action_bias_applied"] is True
    assert float(details["codex_legacy_action_head_delta_norm"]) > 0.0
    assert float(details["codex_legacy_action_bias_target_delta"]) < 0.0
    graph_action_surface = min(12, next_trace.selected_action.size)
    assert details["semantic_action"] == int(np.argmax(next_trace.selected_action[:graph_action_surface]))
    assert bool(next_trace.allowed_action_mask[action]) or action == 0


def test_codex_specialist_full_rl_writes_direct_rnn_tensor_scope():
    """The specialist profile must send signed Codex reward to Direct-RNN AdamW."""
    policy, trace, packet, scorer, reference_hash = _legacy_evidence(
        reward=-0.8,
        seed=800,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_min_abs_mean_advantage = 0.0
    policy.direct_rnn_rl_delayed_reconciliation_enabled = False

    def lifecycle_for(current_trace, ticket, reward=-0.8):
        return SimpleNamespace(
            final_close_confirmed=True,
            metadata={"position_owner_trace_id": current_trace.trace_id},
            position_ticket=ticket,
            realized_r=reward,
            mfe_r=0.1,
            mae_r=-1.0,
            commission=0.0,
            swap=0.0,
            close_reason="SL" if reward < 0.0 else "TP",
            side=1,
            entry_count=1,
            entry_price=100.0,
            initial_sl_price=99.0,
            initial_tp_price=102.0,
            initial_risk=1.0,
            initial_atr=0.4,
            pending_fill_bars=0,
        )

    packet_1 = packet
    scorer_1 = scorer
    trace_2 = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    packet_2 = build_codex_teaching_packet(
        lifecycle_for(trace_2, "legacy-specialist-2"),
        (trace_2,),
    )
    raw_2 = receipt_for(packet_2)
    raw_2["teacher_reward"] = -0.8
    receipt_2 = validate_codex_teacher_receipt(raw_2, packet_2, "test-model")
    scorer_2 = CodexTeacherRewardModel(packet_2, receipt_2, max_probability_shift=0.05)

    lifecycle_queue_before = list(getattr(policy, "_direct_rnn_rl_batches", ()) or ())
    first = policy.optimize_codex_teacher_trace_packet(
        (trace,),
        scorer_1,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )
    assert first["rlhf_parameter_scope"] == "direct_rnn_full_rl_adamw_v1"
    assert first["rlhf_ppo_update"] is True
    assert first["rlhf_update_pending"] is False
    assert first["rlhf_direction_expected"] == "decrease"
    assert first["rlhf_directional_update"] is True
    assert first["rlhf_target_action_probability_after"] < first[
        "rlhf_target_action_probability_before"
    ]
    assert first["rlhf_reward_mean"] < 0.0
    assert first["rlhf_parameter_hash_before"] != first["rlhf_parameter_hash_after"]
    assert list(getattr(policy, "_direct_rnn_rl_batches", ()) or ()) == lifecycle_queue_before

    second = policy.optimize_codex_teacher_trace_packet(
        (trace_2,),
        scorer_2,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )
    assert second["rlhf_parameter_scope"] == "direct_rnn_full_rl_adamw_v1"
    assert second["rlhf_ppo_update"] is True
    assert second["rlhf_parameter_changed"] is True
    assert second["rlhf_parameter_hash_before"] != second["rlhf_parameter_hash_after"]
    assert second["rlhf_reward_mean"] < 0.0
    assert second["rlhf_direct_rnn_optimizer_steps"] >= 1
    assert second["rlhf_direct_rnn_updated_modules"] >= 1
    assert second["rlhf_target_action_probability_after"] is not None
    assert second["rlhf_probability_after_source"] == (
        "frozen_forward_action_head_projection_v1"
    )
    assert second["rlhf_direction_expected"] == "decrease"
    assert second["rlhf_directional_update"] is True
    assert second["rlhf_target_action_probability_after"] < second[
        "rlhf_target_action_probability_before"
    ]
    assert policy.get_codex_teacher_live_parameter_hash() == second[
        "rlhf_parameter_hash_after"
    ]

    trace_3 = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    packet_3 = build_codex_teaching_packet(
        lifecycle_for(trace_3, "legacy-specialist-positive", reward=0.8),
        (trace_3,),
    )
    raw_3 = receipt_for(packet_3)
    raw_3["teacher_reward"] = 0.8
    receipt_3 = validate_codex_teacher_receipt(raw_3, packet_3, "test-model")
    scorer_3 = CodexTeacherRewardModel(packet_3, receipt_3, max_probability_shift=0.05)
    positive_queue_before = list(getattr(policy, "_direct_rnn_rl_batches", ()) or ())
    positive = policy.optimize_codex_teacher_trace_packet(
        (trace_3,),
        scorer_3,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )
    assert positive["rlhf_parameter_scope"] == "direct_rnn_full_rl_adamw_v1"
    assert positive["rlhf_ppo_update"] is True
    assert positive["rlhf_parameter_changed"] is True
    assert positive["rlhf_direction_expected"] == "increase"
    assert positive["rlhf_directional_update"] is True
    assert positive["rlhf_target_action_probability_after"] > positive[
        "rlhf_target_action_probability_before"
    ]
    assert positive["rlhf_packet_max_probability_shift"] <= 0.05 + 1.0e-8
    assert list(getattr(policy, "_direct_rnn_rl_batches", ()) or ()) == positive_queue_before


def test_codex_specialist_full_rl_projects_from_hidden_when_forward_cache_is_stripped():
    """Delayed bridge copies may retain hidden output but strip the large cache."""
    policy, trace, packet, scorer, reference_hash = _legacy_evidence(
        reward=-0.8,
        seed=801,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_min_abs_mean_advantage = 0.0
    policy.direct_rnn_rl_delayed_reconciliation_enabled = False
    # The runtime profile synchronizes module-level AdamW flags during the
    # next decision; mirror that lifecycle before stripping the cache.
    policy.act(np.zeros(INPUT_COUNT, dtype=float))
    stripped = replace(
        trace,
        module_action_traces=tuple(
            replace(entry, forward_trace=None)
            for entry in trace.module_action_traces
        ),
    )

    result = policy.optimize_codex_teacher_trace_packet(
        (stripped,),
        scorer,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )

    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_update_rejected"] is False
    assert result["rlhf_parameter_changed"] is True
    assert result["rlhf_parameter_hash_before"] != result["rlhf_parameter_hash_after"]
    assert result["rlhf_probability_after_source"] == (
        "module_action_trace_hidden_fallback_v1"
    )
    assert result["rlhf_direction_expected"] == "decrease"
    assert result["rlhf_target_action_probability_after"] < result[
        "rlhf_target_action_probability_before"
    ]


def test_codex_specialist_full_rl_verifies_hierarchical_trade_hold_probability():
    """Post-write verification must match the live HOLD/trade denominator."""
    policy, _initial_trace, _initial_packet, _initial_scorer, reference_hash = _legacy_evidence(
        reward=-0.8,
        seed=802,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_min_abs_mean_advantage = 0.0
    policy.direct_rnn_rl_delayed_reconciliation_enabled = False
    policy.direct_rnn_rl_trade_hold_separation_enabled = True
    policy.act(np.zeros(INPUT_COUNT, dtype=float))
    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": trace.trace_id},
        position_ticket="legacy-hierarchical-802",
        realized_r=-0.8,
        mfe_r=0.1,
        mae_r=-1.0,
        commission=0.0,
        swap=0.0,
        close_reason="SL",
        side=1,
        entry_count=1,
        entry_price=100.0,
        initial_sl_price=99.0,
        initial_tp_price=102.0,
        initial_risk=1.0,
        initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, (trace,))
    raw = receipt_for(packet)
    raw["teacher_reward"] = -0.8
    receipt = validate_codex_teacher_receipt(raw, packet, "test-model")
    scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)

    result = policy.optimize_codex_teacher_trace_packet(
        (trace,),
        scorer,
        source_checkpoint_sha256=reference_hash,
        reference_policy_sha256=reference_hash,
    )

    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_update_rejected"] is False
    assert result["rlhf_parameter_changed"] is True
    assert result["rlhf_probability_after_source"] == (
        "frozen_forward_action_head_projection_v1_hierarchical_trade_hold_v1"
    )
    assert result["rlhf_target_action_probability_after"] < result[
        "rlhf_target_action_probability_before"
    ]
    assert result["rlhf_packet_max_probability_shift"] <= 0.05 + 1.0e-8


def test_receipt_and_scorer_bind_exact_owner():
    packet = build_codex_teaching_packet(*evidence())
    receipt = validate_codex_teacher_receipt(receipt_for(packet), packet, "test-model")
    scorer = CodexTeacherRewardModel(packet, receipt, scale=0.25)
    features = causal_policy_feature_vector(packet["raw_obs137"], packet["action"])
    assert scorer.score(features) == pytest.approx(0.4 * 0.9 * 0.25)


def test_management_reward_uses_incremental_counterfactual_not_total_lifecycle_pnl():
    lifecycle, traces = evidence("management-counterfactual")
    target = traces[1]
    target.action_details["five_action_index"] = 10
    lifecycle.exit_trace_id = target.trace_id
    lifecycle.exit_requested_action = 10
    lifecycle.exit_kind = "EXPERT"
    lifecycle.exit_controller = "five"
    lifecycle.realized_r = 0.75
    lifecycle.net_pnl = 25.0
    lifecycle.post_exit_hold_r = 0.875
    lifecycle.post_exit_horizon_bars = 12

    packet = build_codex_teaching_packet(
        lifecycle, traces, target_trace_id=target.trace_id,
    )
    assert packet["target_is_entry_owner"] is False
    assert packet["account_impact"]["counterfactual_delta_r"] == pytest.approx(-0.125)
    receipt = validate_codex_teacher_receipt(
        hypothesis_receipt_for(packet, status="confirmed", predicted_sign=1),
        packet,
        "test-model",
        require_hypothesis=True,
    )

    scorer = CodexTeacherRewardModel(packet, receipt, scale=0.25)

    assert scorer.account_impact_reward == pytest.approx(-0.125)
    assert scorer.account_impact_reward_source == "management_counterfactual_delta"
    assert scorer.combined_reward < 0.0
    with pytest.raises(ValueError, match="non-owner"):
        scorer.score(causal_policy_feature_vector(packet["raw_obs137"], 2))
    bad = receipt_for(packet)
    bad["trace_id"] = "five:wrong"
    with pytest.raises(ValueError, match="trace_id mismatch"):
        validate_codex_teacher_receipt(bad, packet, "test-model")


def test_hypothesis_receipt_is_locally_verified_and_scales_intervention():
    packet = build_codex_teaching_packet(*evidence())
    receipt = validate_codex_teacher_receipt(
        hypothesis_receipt_for(packet), packet, "test-model", require_hypothesis=True,
    )
    assert receipt["hypothesis_schema_version"] == "codex_hypothesis_v2"
    assert receipt["verification_status"] == "confirmed"
    scorer = CodexTeacherRewardModel(packet, receipt, scale=0.25)
    assert scorer.hypothesis_status == "confirmed"
    assert scorer.intervention_multiplier == pytest.approx(1.1)
    assert scorer.score(causal_policy_feature_vector(packet["raw_obs137"], packet["action"])) == pytest.approx(
        0.4 * 0.9 * 0.25 * 1.1
    )


def test_hypothesis_confirmation_rejects_mismatched_outcome_sign():
    packet = build_codex_teaching_packet(*evidence())
    bad = hypothesis_receipt_for(packet, predicted_sign=-1)
    with pytest.raises(ValueError, match="contradicts immutable outcome"):
        validate_codex_teacher_receipt(
            bad, packet, "test-model", require_hypothesis=True,
        )


def test_account_impact_packet_and_negative_scorer_use_post_close_evidence():
    lifecycle, traces = evidence("account-impact")
    traces[2].account_context = {
        "balance": 10_000.0,
        "equity": 10_000.0,
        "sequence": 10.0,
    }
    lifecycle.realized_r = -0.8
    lifecycle.net_pnl = -25.0
    lifecycle.post_exit_hold_r = 0.35
    lifecycle.post_exit_horizon_bars = 5
    lifecycle.account_context_after_close = {
        "balance": 9_975.0,
        "equity": 9_975.0,
        "sequence": 15.0,
    }
    packet = build_codex_teaching_packet(lifecycle, traces)
    impact = packet["account_impact"]
    assert impact["status"] == "confirmed"
    assert impact["verification_ready"] is True
    assert impact["balance_delta"] == pytest.approx(-25.0)
    assert impact["equity_delta"] == pytest.approx(-25.0)
    assert impact["counterfactual_status"] == "validated"
    assert impact["counterfactual_delta_r"] == pytest.approx(-1.15)

    raw = hypothesis_receipt_for(packet, predicted_sign=-1)
    receipt = validate_codex_teacher_receipt(
        raw, packet, "test-model", require_hypothesis=True,
    )
    scorer = CodexTeacherRewardModel(packet, receipt)
    assert scorer.account_impact_status == "confirmed"
    assert scorer.account_impact_reward < 0.0
    assert scorer.score(causal_policy_feature_vector(packet["raw_obs137"], packet["action"])) < 0.0


def test_account_impact_confirms_unique_deal_cashflow_without_shared_snapshot():
    """A unique final deal proves cashflow impact even if MT5 has no safe snapshot."""
    lifecycle, traces = evidence("unique-deal-cashflow")
    lifecycle.realized_r = -0.8
    lifecycle.net_pnl = -25.0
    lifecycle.close_count = 1
    lifecycle.partial_close_count = 0
    lifecycle.metadata["account_cashflow_attribution"] = "unique_final_deal"
    lifecycle.account_cashflow_delta = -25.0
    lifecycle.account_cashflow_deal_ticket = "deal-9001"

    packet = build_codex_teaching_packet(lifecycle, traces)
    impact = packet["account_impact"]
    assert impact["status"] == "confirmed"
    assert impact["verification_ready"] is True
    assert impact["evidence_scope"] == "immutable_lifecycle_plus_unique_deal_cashflow"
    assert impact["balance_delta"] is None
    assert impact["equity_delta"] is None
    assert impact["cashflow_delta"] == pytest.approx(-25.0)
    assert impact["cashflow_deal_ticket"] == "deal-9001"
    assert impact["account_mark_attribution"] == "unique_deal_cashflow"

    raw = hypothesis_receipt_for(packet, predicted_sign=-1)
    receipt = validate_codex_teacher_receipt(
        raw, packet, "test-model", require_hypothesis=True,
    )
    scorer = CodexTeacherRewardModel(packet, receipt)
    assert scorer.account_impact_status == "confirmed"
    assert scorer.account_impact_reward is not None
    assert scorer.account_impact_reward < 0.0


def test_account_impact_uses_raw_legacy_decision_mark_when_context_is_normalized():
    lifecycle, traces = evidence("legacy-raw-account-mark")
    traces[2].account_context = {
        "balance_drawdown": 0.0,
        "equity_to_balance": 0.0,
        "positions": 0.0,
    }
    traces[2].action_details["codex_decision_account"] = {
        "balance": 10_000.0,
        "equity": 10_000.0,
        "sequence": 10.0,
    }
    lifecycle.net_pnl = -25.0
    lifecycle.realized_r = -0.8
    lifecycle.account_context_after_close = {
        "balance": 9_975.0,
        "equity": 9_975.0,
        "sequence": 15.0,
    }

    packet = build_codex_teaching_packet(lifecycle, traces)
    impact = packet["account_impact"]
    assert impact["status"] == "confirmed"
    assert impact["verification_ready"] is True
    assert impact["balance_delta"] == pytest.approx(-25.0)
    assert impact["equity_delta"] == pytest.approx(-25.0)


def test_legacy_close_lifecycle_keeps_delayed_counterfactual_fields():
    trace = SimpleNamespace(trace_id="legacy:trace")
    lifecycle = _legacy_codex_lifecycle_from_close_details(
        {
            "position_ticket": "17",
            "profit": "-25.0",
            "initial_risk": "31.25",
            "post_exit_horizon_bars": "48",
            "post_exit_hold_r": "0.25",
        },
        trace,
        -0.8,
    )
    assert lifecycle.post_exit_horizon_bars == 48
    assert lifecycle.post_exit_hold_r == pytest.approx(0.25)


def test_legacy_deferred_case_releases_from_current_counterfactual_lifecycle(monkeypatch):
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(),
        offline_replay=True,
        codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True,
        exit_counterfactual_horizon_bars=48,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    monkeypatch.setattr(evaluator, "_write_codex_teacher_status", lambda *args, **kwargs: None)
    original = SimpleNamespace(
        position_ticket="17",
        trace_id="legacy:trace",
        metadata={"position_owner_trace_id": "legacy:trace"},
        close_reason="TP",
        realized_r=1.0,
        final_close_confirmed=True,
        post_exit_hold_r=None,
        post_exit_horizon_bars=None,
        account_context_after_close={"balance": 9_975.0, "equity": 9_975.0},
    )
    current = SimpleNamespace(
        position_ticket=original.position_ticket,
        trace_id=original.trace_id,
        metadata=dict(original.metadata),
        close_reason=original.close_reason,
        realized_r=original.realized_r,
        final_close_confirmed=True,
        account_context_after_close=dict(original.account_context_after_close),
        post_exit_hold_r=-0.5,
        post_exit_horizon_bars=48,
    )
    case_id = "|".join((
        evaluator._codex_lifecycle_identity(original),
        "legacy",
        "legacy:trace",
    ))
    evaluator._codex_teacher_deferred_cases[case_id] = {
        "lifecycle": original,
        "rows": (SimpleNamespace(trace_id="legacy:trace"),),
        "consume_packet": None,
        "target_trace_id": "legacy:trace",
    }
    captured = []
    monkeypatch.setattr(
        evaluator,
        "_submit_codex_teacher_case",
        lambda identity, lifecycle, traces, **kwargs: (
            captured.append((identity, lifecycle, traces, kwargs)) or True
        ),
    )

    assert evaluator._flush_deferred_codex_teacher_cases((current,)) == 1
    assert len(captured) == 1
    assert captured[0][0] == case_id
    assert captured[0][1].post_exit_horizon_bars == 48
    assert captured[0][1].post_exit_hold_r == pytest.approx(-0.5)
    assert captured[0][3]["target_trace_id"] == "legacy:trace"


def test_entry_loss_cannot_become_positive_when_hold_counterfactual_is_worse():
    """Absolute account loss must dominate a favorable relative counterfactual."""
    lifecycle, traces = evidence("account-loss-hold-worse")
    lifecycle.realized_r = -0.4145915246832678
    lifecycle.net_pnl = -18.98
    lifecycle.post_exit_hold_r = -1.64394932
    lifecycle.post_exit_horizon_bars = 12
    traces[2].account_context = {
        "balance": 10_000.0,
        "equity": 10_000.0,
    }
    lifecycle.account_context_after_close = {
        "balance": 9_981.02,
        "equity": 9_981.02,
    }
    packet = build_codex_teaching_packet(lifecycle, traces)
    raw = hypothesis_receipt_for(packet, predicted_sign=-1)
    # An optimistic rationale must not be able to override the immutable loss.
    raw["teacher_reward"] = 0.90
    receipt = validate_codex_teacher_receipt(
        raw, packet, "test-model", require_hypothesis=True,
    )
    scorer = CodexTeacherRewardModel(packet, receipt)
    features = causal_policy_feature_vector(packet["raw_obs137"], packet["action"])

    assert packet["account_impact"]["counterfactual_delta_r"] == pytest.approx(1.2293577953)
    assert scorer.account_impact_reward < 0.0
    assert scorer.combined_reward < 0.0
    assert scorer.score(features) < 0.0


def test_ambiguous_account_mark_stays_partial_but_keeps_signed_reward():
    lifecycle, traces = evidence("ambiguous-account-mark")
    lifecycle.realized_r = -0.8
    lifecycle.net_pnl = -25.0
    lifecycle.metadata["account_mark_attribution"] = "ambiguous_same_close_batch"
    traces[2].account_context = {
        "balance": 10_000.0,
        "equity": 10_000.0,
    }
    lifecycle.account_context_after_close = {
        "balance": 9_975.0,
        "equity": 9_975.0,
    }

    packet = build_codex_teaching_packet(lifecycle, traces)
    impact = packet["account_impact"]
    assert impact["status"] == "partial"
    assert impact["verification_ready"] is False
    assert impact["account_mark_attribution"] == "ambiguous_same_close_batch"
    assert "ambiguous" in impact["verification_evidence"]

    raw = hypothesis_receipt_for(packet, predicted_sign=-1)
    receipt = validate_codex_teacher_receipt(
        raw, packet, "test-model", require_hypothesis=True,
    )
    scorer = CodexTeacherRewardModel(packet, receipt)
    assert scorer.account_impact_status == "partial"
    assert scorer.account_impact_reward < 0.0


def test_partial_account_impact_keeps_signed_lifecycle_reward():
    """A close without a uniquely attributable account mark still teaches its sign."""
    lifecycle, traces = evidence("account-impact-partial")
    lifecycle.realized_r = -0.8
    lifecycle.net_pnl = -25.0
    packet = build_codex_teaching_packet(lifecycle, traces)
    impact = packet["account_impact"]
    assert impact["status"] == "partial"
    assert impact["verification_ready"] is False

    raw = hypothesis_receipt_for(packet, predicted_sign=-1)
    receipt = validate_codex_teacher_receipt(
        raw, packet, "test-model", require_hypothesis=True,
    )
    scorer = CodexTeacherRewardModel(packet, receipt)
    assert scorer.account_impact_status == "partial"
    assert scorer.account_impact_reward == pytest.approx(-0.8)
    assert scorer.account_impact_reward_source == "immutable_lifecycle_partial"
    assert scorer.combined_reward < 0.0
    assert scorer.score(
        causal_policy_feature_vector(packet["raw_obs137"], packet["action"])
    ) < 0.0


def test_reward_composition_is_auditable_when_teacher_and_account_signs_conflict():
    """The learner signal exposes why a raw Codex penalty can become positive."""
    lifecycle, traces = evidence("account-impact-conflict")
    lifecycle.net_pnl = 25.0
    packet = build_codex_teaching_packet(lifecycle, traces)
    raw = hypothesis_receipt_for(packet, predicted_sign=1)
    raw["teacher_reward"] = -0.4
    receipt = validate_codex_teacher_receipt(
        raw, packet, "test-model", require_hypothesis=True,
    )
    scorer = CodexTeacherRewardModel(packet, receipt)

    assert scorer.teacher_reward == pytest.approx(-0.4)
    assert scorer.account_impact_reward == pytest.approx(0.8)
    assert scorer.combined_reward == pytest.approx(0.2)
    assert scorer.reward_composition == (
        "teacher_account_impact_50_50_account_negative_floor_v1"
    )
    assert scorer.account_impact_reward_source == "immutable_lifecycle_partial"


def test_account_impact_receipt_cannot_change_immutable_delta():
    lifecycle, traces = evidence("account-impact-mismatch")
    traces[2].account_context = {"balance": 100.0, "equity": 100.0}
    lifecycle.net_pnl = 2.0
    lifecycle.account_context_after_close = {"balance": 102.0, "equity": 102.0}
    packet = build_codex_teaching_packet(lifecycle, traces)
    raw = hypothesis_receipt_for(packet, predicted_sign=1)
    raw["account_impact"]["balance_delta"] = 200.0
    with pytest.raises(ValueError, match="account impact balance_delta mismatch"):
        validate_codex_teacher_receipt(
            raw, packet, "test-model", require_hypothesis=True,
        )


def test_account_impact_receipt_cannot_change_mark_attribution():
    lifecycle, traces = evidence("account-impact-attribution-mismatch")
    lifecycle.metadata["account_mark_attribution"] = "ambiguous_same_close_batch"
    lifecycle.account_context_after_close = {"balance": 102.0, "equity": 102.0}
    packet = build_codex_teaching_packet(lifecycle, traces)
    raw = hypothesis_receipt_for(packet, predicted_sign=1)
    raw["account_impact"]["account_mark_attribution"] = "unique_final_deal"
    with pytest.raises(ValueError, match="account impact attribution mismatch"):
        validate_codex_teacher_receipt(
            raw, packet, "test-model", require_hypothesis=True,
        )


def test_account_impact_receipt_normalizes_known_scope_alias():
    lifecycle, traces = evidence("account-impact-scope-alias")
    lifecycle.net_pnl = 2.0
    lifecycle.metadata["account_mark_attribution"] = "ambiguous_same_close_batch"
    traces[2].account_context = {"balance": 100.0, "equity": 100.0}
    lifecycle.account_context_after_close = {"balance": 102.0, "equity": 102.0}
    packet = build_codex_teaching_packet(lifecycle, traces)
    raw = hypothesis_receipt_for(packet, predicted_sign=1)
    raw["account_impact"]["evidence_scope"] = (
        "immutable_lifecycle_with_ambiguous_post_close_account"
    )
    receipt = validate_codex_teacher_receipt(
        raw, packet, "test-model", require_hypothesis=True,
    )
    assert receipt["account_impact"]["evidence_scope"] == packet["account_impact"]["evidence_scope"]


def test_active_hypothesis_must_acknowledge_account_impact_scope():
    packet = build_codex_teaching_packet(*evidence("account-impact-required"))
    raw = hypothesis_receipt_for(packet)
    raw.pop("account_impact")
    with pytest.raises(ValueError, match="account impact receipt is required"):
        validate_codex_teacher_receipt(
            raw, packet, "test-model", require_hypothesis=True,
        )


def test_negative_teacher_reward_never_flips_to_positive_advantage():
    packet = build_codex_teaching_packet(*evidence())
    raw = receipt_for(packet)
    raw["teacher_reward"] = -0.2
    receipt = validate_codex_teacher_receipt(raw, packet, "test-model")
    scorer = CodexTeacherRewardModel(packet, receipt)
    rollout = RLHFPolicyRollout(
        rollout_id=packet["trace_id"], pair_id="pair", state_hash="state",
        geometry_hash="geometry", feature_schema=scorer.feature_schema,
        features=tuple(causal_policy_feature_vector(packet["raw_obs137"], packet["action"])),
        action=packet["action"], old_logits=(0.0,) * 7, reference_logits=(0.0,) * 7,
        value=-0.5, allowed_action_mask=(True,) * 7,
    )
    batch = prepare_rlhf_rollout_batch(scorer, (rollout,))
    assert float(batch["raw_rewards"][0]) < 0
    assert float(batch["advantages"][0]) < 0


def test_optimizer_state_lookup_never_compares_tensor_keys():
    """Legacy and current FiveAdam keys must both avoid Tensor truthiness."""
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    current_optimizer = SimpleNamespace(state={id(parameter): {"step": 3}})
    legacy_optimizer = SimpleNamespace(state={parameter: {"step": 4}})

    # The current integer identity wins without ever evaluating
    # ``parameter == id(parameter)``.  The second assertion exercises the
    # legacy object-key compatibility path without mixing colliding key types.
    assert _five_optimizer_state_for_parameter(current_optimizer, parameter)["step"] == 3
    assert _five_optimizer_state_for_parameter(legacy_optimizer, parameter)["step"] == 4


def test_optimizer_state_reset_and_restore_never_compare_legacy_tensor_keys():
    """Reset/rollback must also tolerate pre-identity FiveAdam checkpoints."""
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = SimpleNamespace(state={parameter: {"step": 4}})

    removed = _five_optimizer_state_remove_for_parameter(optimizer, parameter)
    assert removed == {"step": 4}
    assert optimizer.state == {}

    _five_optimizer_state_set_for_parameter(optimizer, parameter, {"step": 5})
    assert list(optimizer.state) == [id(parameter)]
    assert _five_optimizer_state_for_parameter(optimizer, parameter)["step"] == 5


def test_five_adam_step_canonicalizes_legacy_parameter_keys():
    """The optimizer itself must tolerate pre-identity checkpoint state."""
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = FiveAdam([parameter], lr=0.01, beta1=0.9, beta2=0.999)
    optimizer.state[parameter] = {
        "step": 0,
        "m": torch.zeros_like(parameter),
        "v": torch.zeros_like(parameter),
    }
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    assert all(isinstance(key, int) for key in optimizer.state)
    assert list(optimizer.state) == [id(parameter)]
    assert torch.all(parameter.detach() < 0.0)


def test_five_adam_unpickle_canonicalizes_legacy_parameter_keys_immediately():
    """A loaded legacy checkpoint must be safe before the first optimizer step."""
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = FiveAdam([parameter], lr=0.01, beta1=0.9, beta2=0.999)
    legacy_entry = {
        "step": 4,
        "m": torch.zeros_like(parameter),
        "v": torch.zeros_like(parameter),
    }

    optimizer.__setstate__({
        "parameters": [parameter],
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "state": {parameter: legacy_entry},
    })

    assert list(optimizer.state) == [id(parameter)]
    assert optimizer.state[id(parameter)]["step"] == 4
    assert _five_optimizer_state_for_parameter(optimizer, parameter)["step"] == 4


def test_grade_uses_structured_cli_and_validates(monkeypatch):
    packet = build_codex_teaching_packet(*evidence())
    def fake_run(command, **kwargs):
        assert "--output-schema" in command
        assert "read-only" in command
        assert "-c" in command
        assert "model_reasoning_effort=medium" in command
        assert packet["packet_sha256"] in kwargs["input"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt_for(packet)), stderr="")
    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    graded = grade_with_codex(packet, model="test-model", codex_exe="codex.exe")
    assert graded["confidence"] == 0.9
    assert graded["codex_execution"]["process"] == "completed"
    assert graded["codex_execution"]["returncode"] == 0
    assert graded["codex_execution"]["json_parse"] == "ok"
    assert graded["codex_execution"]["receipt_validation"] == "ok"


def test_grade_requires_hypothesis_contract_when_requested(monkeypatch):
    packet = build_codex_teaching_packet(*evidence("cli-hypothesis"))

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(hypothesis_receipt_for(packet)),
            stderr="",
        )

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    graded = grade_with_codex(
        packet, model="test-model", codex_exe="codex.exe", require_hypothesis=True,
    )
    assert graded["verification_status"] == "confirmed"
    assert graded["intervention_target_action"] == packet["action"]


def test_grade_retries_only_packet_hash_mismatch(monkeypatch):
    packet = build_codex_teaching_packet(*evidence("hash-retry"))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(1)
        raw = receipt_for(packet)
        if len(calls) == 1:
            raw["packet_sha256"] = "0" * 64
        return SimpleNamespace(returncode=0, stdout=json.dumps(raw), stderr="")

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    graded = grade_with_codex(packet, model="test-model", codex_exe="codex.exe")
    assert len(calls) == 2
    assert graded["packet_sha256"] == packet["packet_sha256"]
    assert graded["codex_execution"]["retry_attempted"] is True
    assert graded["codex_execution"]["retry_reason"] == "packet_sha256_mismatch"
    assert graded["codex_execution"]["retry_attempt"] == 2


def test_grade_retry_prompt_repairs_truncated_packet_hash(monkeypatch):
    packet = build_codex_teaching_packet(*evidence("hash-truncation-retry"))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(dict(kwargs))
        raw = receipt_for(packet)
        if len(calls) == 1:
            expected = packet["packet_sha256"]
            raw["packet_sha256"] = expected[:53] + expected[-1]
        return SimpleNamespace(returncode=0, stdout=json.dumps(raw), stderr="")

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    graded = grade_with_codex(packet, model="test-model", codex_exe="codex.exe")

    assert len(calls) == 2
    assert packet["packet_sha256"] in calls[1]["input"]
    assert "đủ đúng 64 ký tự" in calls[1]["input"]
    assert graded["packet_sha256"] == packet["packet_sha256"]
    assert graded["codex_execution"]["retry_attempted"] is True
    assert graded["codex_execution"]["retry_reason"] == "packet_sha256_mismatch"


def test_grade_retries_packet_hash_twice_before_failing_closed(monkeypatch):
    packet = build_codex_teaching_packet(*evidence("hash-double-retry"))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(dict(kwargs))
        raw = receipt_for(packet)
        if len(calls) < 3:
            expected = packet["packet_sha256"]
            raw["packet_sha256"] = expected[:52] + "0" + expected[53:]
        return SimpleNamespace(returncode=0, stdout=json.dumps(raw), stderr="")

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    graded = grade_with_codex(packet, model="test-model", codex_exe="codex.exe")

    assert len(calls) == 3
    assert "bốn chunk chuẩn 16 ký tự" in calls[1]["input"]
    assert "bốn chunk chuẩn 16 ký tự" in calls[2]["input"]
    assert graded["packet_sha256"] == packet["packet_sha256"]
    assert graded["codex_execution"]["retry_attempted"] is True
    assert graded["codex_execution"]["retry_reason"] == "packet_sha256_mismatch"
    assert graded["codex_execution"]["retry_attempt"] == 3


def test_resolver_prefers_desktop_binary_over_path_shim(monkeypatch):
    tmp_path = writable_test_dir("resolver")
    desktop = tmp_path / "desktop" / "codex.exe"
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text("stub", encoding="utf-8")
    monkeypatch.delenv("CODEX_CLI_PATH", raising=False)
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher._desktop_codex_executables",
        lambda: [desktop],
    )
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.shutil.which",
        lambda name: "npm-codex" if name in {"codex.exe", "codex"} else None,
    )
    assert resolve_codex_executable() == str(desktop)
    monkeypatch.setenv("CODEX_CLI_PATH", "configured-codex")
    assert resolve_codex_executable() == "configured-codex"


def test_preflight_records_version_model_and_structured_probe(monkeypatch):
    tmp_path = writable_test_dir("preflight")
    executable = tmp_path / "codex.exe"
    executable.write_text("stub", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="codex-cli 0.155.0-alpha.16.4\n", stderr="")
        assert "--output-schema" in command
        assert "--ignore-user-config" in command
        assert "gpt-5.6-luna" in command
        assert "model_reasoning_effort=xhigh" in command
        probe_receipt = {
            "packet_sha256": "preflight", "trace_id": "preflight",
            "position_ticket": "preflight", "rubric_version": RUBRIC_VERSION,
            "rationale": "preflight structured output ok", "decision_quality": 0.5,
            "risk_quality": 0.5, "timing_quality": 0.5, "execution_quality": 0.5,
            "teacher_reward": 0.0, "confidence": 1.0,
            "hypothesis": "preflight schema hypothesis is structurally valid",
            "prediction": "the structured probe should parse with no training update",
            "experiment": "validate the returned object against the teacher schema",
            "predicted_reward_sign": 0, "verification_status": "inconclusive",
            "verification_observed_reward": 0.0,
            "verification_evidence": "preflight has no market outcome ledger",
            "intervention_multiplier": 1.0, "intervention_target_action": 0,
            "account_impact": {
                "schema_version": 1, "status": "inconclusive", "verification_ready": False,
                "evidence_scope": "preflight has no market account ledger",
                "observed_net_pnl": None, "observed_realized_r": None,
                "balance_delta": None, "equity_delta": None,
                "counterfactual_status": "not_available", "counterfactual_hold_r": None,
                "counterfactual_delta_r": None, "counterfactual_horizon_bars": None,
                "verification_evidence": "preflight has no market account ledger",
            },
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(probe_receipt),
            stderr="",
        )

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    report = preflight_codex_teacher(
        model="gpt-5.6-luna", reasoning_effort="xhigh",
        codex_exe=str(executable), timeout_sec=10,
    )
    assert report["status"] == "passed"
    assert report["cli_version"].startswith("codex-cli 0.155.0")
    assert report["model"] == "gpt-5.6-luna"
    assert report["probe_json"] == "ok"
    assert len(calls) == 2


def test_preflight_retries_transient_codex_state_lock(monkeypatch):
    tmp_path = writable_test_dir("preflight_state_retry")
    executable = tmp_path / "codex.exe"
    executable.write_text("stub", encoding="utf-8")
    probe_calls = 0

    probe_receipt = {
        "packet_sha256": "preflight", "trace_id": "preflight",
        "position_ticket": "preflight", "rubric_version": RUBRIC_VERSION,
        "rationale": "preflight structured output ok", "decision_quality": 0.5,
        "risk_quality": 0.5, "timing_quality": 0.5, "execution_quality": 0.5,
        "teacher_reward": 0.0, "confidence": 1.0,
        "hypothesis": "preflight schema hypothesis is structurally valid",
        "prediction": "the structured probe should parse with no training update",
        "experiment": "validate the returned object against the teacher schema",
        "predicted_reward_sign": 0, "verification_status": "inconclusive",
        "verification_observed_reward": 0.0,
        "verification_evidence": "preflight has no market outcome ledger",
        "intervention_multiplier": 1.0, "intervention_target_action": 0,
        "account_impact": {
            "schema_version": 1, "status": "inconclusive", "verification_ready": False,
            "evidence_scope": "preflight has no market account ledger",
            "observed_net_pnl": None, "observed_realized_r": None,
            "balance_delta": None, "equity_delta": None,
            "counterfactual_status": "not_available", "counterfactual_hold_r": None,
            "counterfactual_delta_r": None, "counterfactual_horizon_bars": None,
            "verification_evidence": "preflight has no market account ledger",
        },
    }

    def fake_run(command, **kwargs):
        nonlocal probe_calls
        if command[-1] == "--version":
            return SimpleNamespace(
                returncode=0, stdout="codex-cli 0.155.0-alpha.16.4\n", stderr=""
            )
        probe_calls += 1
        if probe_calls == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=(
                    "failed to initialize state runtime: failed to open state DB; "
                    "attempt to write a readonly database"
                ),
            )
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(probe_receipt), stderr=""
        )

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    monkeypatch.setattr("self_growing_trader.codex_teacher._codex_retry_delay", lambda _: None)
    report = preflight_codex_teacher(
        model="gpt-5.6-luna", reasoning_effort="xhigh",
        codex_exe=str(executable), timeout_sec=10,
    )
    assert report["status"] == "passed"
    assert report["probe_retry_attempts"] == 1
    assert probe_calls == 2


def test_grade_retries_transient_codex_state_lock_before_validating_receipt(monkeypatch):
    lifecycle, traces = evidence("grade-state-retry")
    packet = build_codex_teaching_packet(lifecycle, traces)
    receipt = receipt_for(packet)
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=(
                    "failed to initialize in-process app-server client: Access is denied"
                ),
            )
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(receipt), stderr=""
        )

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    monkeypatch.setattr("self_growing_trader.codex_teacher._codex_retry_delay", lambda _: None)
    graded = grade_with_codex(
        packet,
        model="gpt-5.6-luna",
        codex_exe="desktop-codex",
    )
    assert calls == 2
    assert graded["codex_execution"]["retry_attempts"] == 1
    assert graded["codex_execution"]["receipt_validation"] == "ok"


def test_preflight_failure_is_fail_closed_before_probe(monkeypatch):
    tmp_path = writable_test_dir("preflight_failure")
    executable = tmp_path / "codex.exe"
    executable.write_text("stub", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=1, stdout="", stderr="old cli")

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    with pytest.raises(CodexTeacherPreflightError) as error:
        preflight_codex_teacher(codex_exe=str(executable), timeout_sec=10)
    assert error.value.report["reason"] == "cli_version_failed"
    assert len(calls) == 1


def test_preflight_classifies_usage_limit_as_blocked(monkeypatch):
    tmp_path = writable_test_dir("preflight_usage_limit")
    executable = tmp_path / "codex.exe"
    executable.write_text("stub", encoding="utf-8")

    def fake_run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="codex-cli 0.155.0-alpha.16.4\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="You’ve hit your usage limit. Try again later.",
        )

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    with pytest.raises(CodexTeacherPreflightError) as error:
        preflight_codex_teacher(
            model="gpt-5.6-luna",
            reasoning_effort="xhigh",
            codex_exe=str(executable),
            timeout_sec=10,
        )
    report = error.value.report
    assert report["reason"] == "usage_limit"
    assert report["probe_process"] == "blocked"
    assert report["failure_reason"] == "usage_limit"
    assert report["probe_returncode"] == 1


def test_grade_classifies_usage_limit_as_blocked(monkeypatch):
    lifecycle, traces = evidence("grade-usage-limit")
    packet = build_codex_teaching_packet(lifecycle, traces)

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="You've hit your usage limit. Try again later.",
        )

    monkeypatch.setattr("self_growing_trader.codex_teacher.subprocess.run", fake_run)
    with pytest.raises(CodexGradeExecutionError) as error:
        grade_with_codex(
            packet,
            model="gpt-5.6-luna",
            reasoning_effort="xhigh",
            codex_exe="desktop-codex",
        )
    execution = error.value.execution
    assert execution["process"] == "blocked"
    assert execution["failure_reason"] == "usage_limit"
    assert execution["returncode"] == 1
    assert "usage limit" in str(error.value).lower()


def test_failure_record_preserves_packet_identity_and_cli_diagnostics():
    tmp_path = writable_test_dir("failure_record")
    lifecycle, traces = evidence("failure-record")
    packet = build_codex_teaching_packet(lifecycle, traces)
    # HOLD is action 0 and must not be rewritten to the missing-action
    # sentinel in an append-only failure record.
    packet["action"] = 0
    config = MT5BacktestConfig(
        runner_dir=tmp_path, offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_model="gpt-5.6-luna", codex_teacher_executable="desktop-codex",
        codex_teacher_cli_version="codex-cli 0.155.0", rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_receipts_path = tmp_path / "receipts.jsonl"
    error = CodexGradeExecutionError(
        "model rejected",
        {
            "stage": "execute", "returncode": 1, "stderr_tail": "requires newer version",
            "executable_path": "desktop-codex", "cli_version": "codex-cli 0.155.0",
            "model": "gpt-5.6-luna", "reasoning_effort": "xhigh",
            "command": ["desktop-codex", "exec"],
        },
    )
    evaluator._append_codex_teacher_failure(packet, error)
    records = [
        json.loads(line)
        for line in (tmp_path / "receipts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(item for item in records if item["record_type"] == "codex_teacher_failure")
    assert record["record_type"] == "codex_teacher_failure"
    assert record["packet_sha256"] == packet["packet_sha256"]
    assert record["trace_id"] == packet["trace_id"]
    assert record["action"] == 0
    assert record["executable_path"] == "desktop-codex"
    assert record["cli_version"] == "codex-cli 0.155.0"
    assert record["stderr_tail"] == "requires newer version"


def test_status_writer_falls_back_when_primary_is_locked(monkeypatch):
    tmp_path = writable_test_dir("status_fallback")
    config = MT5BacktestConfig(
        runner_dir=tmp_path, offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_receipts_path = tmp_path / "receipts.jsonl"
    primary_attempts = {"count": 0}

    def fake_atomic_write(path, content, **kwargs):
        if str(path).endswith(".status.json"):
            primary_attempts["count"] += 1
            raise OSError("file is locked")
        Path(path).write_text(content, encoding="utf-8")

    monkeypatch.setattr("self_growing_trader.mt5_runner._atomic_write_text", fake_atomic_write)
    evaluator._write_codex_teacher_status()
    fallback = tmp_path / "receipts.jsonl.status.json.latest.json"
    assert primary_attempts["count"] == 3
    assert fallback.is_file()
    status = json.loads(fallback.read_text(encoding="utf-8"))
    assert status["status_path"].endswith("receipts.jsonl.status.json")
    assert status["status_fallback_path"].endswith("receipts.jsonl.status.json.latest.json")
    assert status["status_written_path"].endswith("receipts.jsonl.status.json.latest.json")
    assert status["status_write_error"].startswith("primary_locked:")


def test_preflight_failure_marks_runtime_contract_fail_closed():
    tmp_path = writable_test_dir("preflight_contract")
    candidate = tmp_path / "candidate.pkl"
    contract_path = tmp_path / "runtime_learning_contract.json"
    contract_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "accepted",
        "effective": {
            "five_codex_teacher_effective": False,
            "codex_cli_found": True,
            "codex_preflight_passed": False,
            "codex_receipts_valid": False,
            "codex_hypotheses_confirmed": 0,
            "codex_optimizer_updates": 0,
        },
        "codex_preflight": {"status": "pending"},
    }), encoding="utf-8")
    report = {
        "status": "failed",
        "stage": "probe",
        "reason": "probe_process_failed",
        "cli_version": "codex-cli 0.155.0-alpha.16.4",
        "executable_path": "C:/Codex/codex.exe",
    }
    _persist_codex_teacher_preflight_failure_status(
        candidate,
        model="gpt-5.6-luna",
        reasoning_effort="xhigh",
        executable="C:/Codex/codex.exe",
        report=report,
        error=ValueError("readonly database"),
    )
    updated = json.loads(contract_path.read_text(encoding="utf-8"))
    assert updated["runtime_status"] == "preflight_failed"
    assert updated["effective"]["five_codex_teacher_effective"] is False
    assert updated["effective"]["codex_cli_found"] is True
    assert updated["effective"]["codex_preflight_passed"] is False
    assert updated["codex_preflight"]["reason"] == "probe_process_failed"
    assert "readonly database" in updated["last_error"]


def test_status_writer_syncs_live_runtime_feedback_contract():
    tmp_path = writable_test_dir("live_feedback_contract")
    observatory = tmp_path / "execution_trace.jsonl"
    contract_path = tmp_path / "runtime_learning_contract.json"
    contract_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "accepted",
        "effective": {
            "five_codex_teacher_effective": False,
            "codex_cli_found": False,
            "codex_preflight_passed": False,
            "codex_receipts_valid": False,
            "codex_hypotheses_confirmed": 0,
            "codex_optimizer_updates": 0,
        },
    }), encoding="utf-8")
    config = MT5BacktestConfig(
        runner_dir=tmp_path,
        execution_trace_observatory=str(observatory),
        offline_replay=True,
        codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._last_codex_teacher_phase.update({
        "enabled": True,
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "executable_path": "C:/Codex/codex.exe",
        "cli_version": "codex-cli 0.155.0",
        "preflight": {
            "status": "passed",
            "receipt_validation": "ok",
            "executable_path": "C:/Codex/codex.exe",
            "cli_version": "codex-cli 0.155.0",
        },
        "submitted": 4,
        "graded": 3,
        "hypotheses": 3,
        "hypotheses_confirmed": 2,
        "account_impact_records": 3,
        "account_impact_confirmed": 2,
        "account_impact_partial": 1,
        "account_impact_reward_sum": -1.25,
        "updates": 2,
        "interventions_applied": 2,
        "rlhf_parameter_changed": 2,
        "rlhf_internal_policy_state_changed": 2,
        "rlhf_probability_shift_count": 2,
        "rlhf_max_probability_shift": 0.04,
        "codex_causal_links_observed": 1,
        "codex_causal_updates_linked": 2,
        "codex_causal_hash_matches": 2,
        "codex_causal_hash_unknown": 0,
        "last_codex_causal_link": {
            "causal_link": "barrier_to_next_action",
            "sequence": 99,
            "action": 4,
        },
        "failures": 1,
        "pending": 1,
        "skips": 1,
    })
    evaluator._codex_teacher_pending["pending-case"] = (None, (), None)
    evaluator._write_codex_teacher_status()
    updated = json.loads(contract_path.read_text(encoding="utf-8"))
    effective = updated["effective"]
    assert updated["runtime_status"] == "running"
    assert effective["five_codex_teacher_effective"] is True
    assert effective["codex_cli_found"] is True
    assert effective["codex_preflight_passed"] is True
    assert effective["codex_receipts_valid"] is True
    assert effective["codex_hypotheses_confirmed"] == 2
    assert effective["codex_optimizer_updates"] == 2
    assert effective["five_codex_teacher_reasoning_effort"] == "xhigh"
    assert updated["live_codex_teacher"]["submitted"] == 4
    assert updated["live_codex_teacher"]["pending"] == 1
    assert updated["live_codex_teacher"]["account_impact_records"] == 3
    assert updated["live_codex_teacher"]["account_impact_confirmed"] == 2
    assert updated["live_codex_teacher"]["account_impact_partial"] == 1
    assert updated["live_codex_teacher"]["account_impact_reward_sum"] == pytest.approx(-1.25)
    assert updated["live_codex_teacher"]["rlhf_parameter_changed"] == 2
    assert updated["live_codex_teacher"]["rlhf_internal_policy_state_changed"] == 2
    assert updated["live_codex_teacher"]["rlhf_probability_shift_count"] == 2
    assert updated["live_codex_teacher"]["rlhf_max_probability_shift"] == pytest.approx(0.04)
    assert updated["live_codex_teacher"]["codex_causal_links_observed"] == 1
    assert updated["live_codex_teacher"]["codex_causal_updates_linked"] == 2
    assert updated["live_codex_teacher"]["codex_causal_hash_matches"] == 2
    assert updated["live_codex_teacher"]["codex_causal_hash_unknown"] == 0
    assert updated["live_codex_teacher"]["last_codex_causal_link"]["causal_link"] == (
        "barrier_to_next_action"
    )


def test_aborting_mt5_run_cancels_pending_teacher_queue_and_logs_failures():
    tmp_path = writable_test_dir("abort_teacher_queue")
    lifecycle, traces = evidence("abort")
    packet = build_codex_teaching_packet(lifecycle, traces)
    config = MT5BacktestConfig(
        runner_dir=tmp_path,
        offline_replay=True,
        codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_receipts_path = tmp_path / "receipts.jsonl"

    class PendingFuture:
        def cancel(self):
            return True

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def shutdown(self, **kwargs):
            self.calls.append(kwargs)

    executor = FakeExecutor()
    evaluator._codex_teacher_executor = executor
    evaluator._codex_teacher_pending["abort-case"] = (
        packet, tuple(traces), PendingFuture(),
    )
    evaluator._last_codex_teacher_phase.update({
        "enabled": True,
        "submitted": 1,
    })
    phase = evaluator._abort_codex_teacher_phase("MT5 report missing")
    assert phase["aborted"] is True
    assert phase["pending"] == 0
    assert phase["failures"] == 1
    assert evaluator._codex_teacher_pending == {}
    assert evaluator._codex_teacher_executor is None
    assert executor.calls == [{"wait": False, "cancel_futures": True}]
    records = [
        json.loads(line)
        for line in (tmp_path / "receipts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(item for item in records if item["record_type"] == "codex_teacher_failure")
    assert record["record_type"] == "codex_teacher_failure"
    assert record["stage"] == "run_abort_cancelled"
    assert record["packet_sha256"] == packet["packet_sha256"]


def test_phase_refuses_unreconciled_run(monkeypatch):
    workspace = Path.cwd()
    config = MT5BacktestConfig(
        runner_dir=workspace, offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    monkeypatch.setattr(evaluator, "_write_codex_teacher_status", lambda *args, **kwargs: None)
    phase = evaluator._run_codex_teacher_phase(object(), reconciled=False, report_path=workspace / "report.xml")
    assert phase["reason"] == "report_ledger_mismatch"
    assert phase["updates"] == 0


def test_codex_profile_isolates_legacy_online_reward():
    config = load_config("configs/he_chuyen_gia_codex_teacher_candidate.yaml")
    assert config.learning.five_codex_teacher_enabled is True
    assert config.learning.mt5_online_update_scale == 0.0
    assert config.learning.five_rlhf_policy_update_enabled is False
    assert config.learning.five_mt5_failure_replay_enabled is False
    assert config.learning.five_codex_teacher_max_cases_per_run == 0
    assert config.learning.five_codex_teacher_online_wait_sec == pytest.approx(120.0)
    assert config.learning.five_codex_teacher_max_workers == 2
    assert config.learning.five_codex_teacher_learning_rate == pytest.approx(0.0003)
    assert config.learning.five_codex_context_similarity_threshold == pytest.approx(0.90)
    assert config.learning.five_codex_context_support_floor == pytest.approx(0.50)


def test_codex_teacher_default_model_matches_desktop_ui():
    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    assert config.learning.five_codex_teacher_model == "gpt-5.6-luna"
    assert config.learning.five_codex_teacher_reasoning_effort == "medium"
    assert config.learning.five_codex_teacher_online_wait_sec == pytest.approx(120.0)
    # Learning configurations auto-connect to Codex when the local CLI and a
    # checkpoint are available; frozen/OOS validation disables the bridge at
    # runtime instead of silently spending a teacher request.
    assert config.learning.five_codex_teacher_auto_connect is True


def test_codex_teacher_worker_bound_is_validated():
    with pytest.raises(ValueError, match="max workers"):
        MT5BacktestConfig(
            runner_dir=Path.cwd(), offline_replay=True,
            codex_teacher_enabled=True,
            codex_teacher_max_workers=9,
            rlhf_source_checkpoint_sha256="source",
        )
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True,
        codex_teacher_enabled=True,
        codex_teacher_max_workers=2,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    assert evaluator._last_codex_teacher_phase["max_workers"] == 2


def test_codex_teacher_online_barrier_waits_for_verified_update(monkeypatch):
    """A finalized close must gate the next action on the first real update."""
    tmp_path = writable_test_dir("online_barrier")
    config = MT5BacktestConfig(
        runner_dir=tmp_path,
        offline_replay=True,
        codex_teacher_enabled=True,
        codex_teacher_online_wait_sec=1.0,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._write_codex_teacher_status = lambda *args, **kwargs: None
    evaluator._codex_teacher_pending["case"] = (None, (), object())
    calls = {"count": 0}

    def fake_drain(policy, *, wait=False, report_path=None):
        calls["count"] += 1
        if calls["count"] >= 2:
            evaluator._last_codex_teacher_phase["updates"] = 1
            evaluator._codex_teacher_pending.clear()
        return evaluator._last_codex_teacher_phase

    monkeypatch.setattr(evaluator, "_drain_codex_teacher_results", fake_drain)
    monkeypatch.setattr("self_growing_trader.mt5_runner.time.sleep", lambda _: None)
    phase = evaluator._wait_for_codex_teacher_action_barrier(
        object(), reason="test_lifecycle",
    )
    assert calls["count"] == 2
    assert phase["online_barrier_waits"] == 1
    assert phase["online_barrier_passed"] == 1
    assert phase["online_barrier_timeouts"] == 0
    assert phase["online_barrier_last"]["passed"] is True
    assert phase["online_barrier_last"]["reason"] == "test_lifecycle"


def test_codex_teacher_online_barrier_zero_is_explicit_async_mode():
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(),
        offline_replay=True,
        codex_teacher_enabled=True,
        codex_teacher_online_wait_sec=0.0,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_pending["case"] = (None, (), object())
    phase = evaluator._wait_for_codex_teacher_action_barrier(object())
    assert phase["online_barrier_waits"] == 0
    assert phase["online_barrier_last"]["enabled"] is False
    assert phase["online_barrier_last"]["skip_reason"] == "disabled"


def test_codex_teacher_barrier_update_is_bound_to_next_action():
    """Only a passed barrier may claim the next MT5 decision consumed an update."""
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(),
        offline_replay=True,
        codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_pending_next_action_links.append({
        "update_index": 1,
        "position_ticket": "ticket-1",
        "trace_id": "five:closed",
        "action": 2,
        "receipt_sha256": "receipt-1",
        "parameter_hash_after": "hash-after",
        "barrier_passed": False,
    })
    evaluator._mark_codex_teacher_barrier_links(
        passed=True, reason="five_lifecycle", waited_sec=0.25,
    )
    links = evaluator._consume_codex_teacher_next_action_links(
        policy=SimpleNamespace(
            get_codex_teacher_live_parameter_hash=lambda: "hash-after"
        ),
        request=SimpleNamespace(sequence=42, event_id="event-42"),
        trace_id="five:next",
        action=4,
        score=0.75,
    )
    assert len(links) == 1
    assert links[0]["barrier_passed"] is True
    assert links[0]["barrier_reason"] == "five_lifecycle"
    assert links[0]["next_request_sequence"] == 42
    assert links[0]["next_trace_id"] == "five:next"
    assert links[0]["next_action"] == 4
    assert links[0]["policy_parameter_hash_at_next_action"] == "hash-after"
    assert links[0]["policy_parameter_hash_matches_update"] is True
    assert evaluator._last_codex_teacher_phase["codex_causal_links_observed"] == 1
    assert evaluator._last_codex_teacher_phase["codex_causal_updates_linked"] == 1
    assert evaluator._last_codex_teacher_phase["codex_causal_hash_matches"] == 1
    assert evaluator._last_codex_teacher_phase["codex_causal_hash_unknown"] == 0
    assert evaluator._codex_teacher_pending_next_action_links == []


def test_codex_teacher_timeout_does_not_claim_next_action_link():
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(),
        offline_replay=True,
        codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_pending_next_action_links.append({
        "update_index": 1,
        "position_ticket": "ticket-1",
        "trace_id": "five:closed",
        "action": 2,
        "barrier_passed": False,
    })
    evaluator._mark_codex_teacher_barrier_links(
        passed=False, reason="five_lifecycle", waited_sec=1.0,
    )
    links = evaluator._consume_codex_teacher_next_action_links(
        request=SimpleNamespace(sequence=43, event_id="event-43"),
        trace_id="five:next",
        action=4,
        score=0.75,
    )
    assert links == []
    assert evaluator._last_codex_teacher_phase["codex_causal_links_observed"] == 0
    assert len(evaluator._codex_teacher_pending_next_action_links) == 1


def test_frozen_codex_validation_disables_teacher_before_disabling_learning():
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(),
        offline_replay=True,
        codex_teacher_enabled=True,
        codex_teacher_preflight={"status": "passed"},
        rlhf_source_checkpoint_sha256="source",
    )
    frozen = replace(
        config,
        learning_enabled=False,
        codex_teacher_enabled=False,
        codex_teacher_preflight={},
    )
    assert frozen.learning_enabled is False
    assert frozen.codex_teacher_enabled is False
    assert frozen.codex_teacher_preflight == {}


def test_teacher_collection_coexists_with_lifecycle_reward(monkeypatch):
    lifecycle, traces = evidence()
    packet = build_codex_teaching_packet(lifecycle, traces)
    recovery_calls = []
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
        codex_teacher_recovery_snapshot_hook=(
            lambda policy, update_count: (
                recovery_calls.append((policy, update_count))
                or {
                    "saved": True,
                    "path": "candidate.recovery.pkl",
                    "optimizer_state_stripped": True,
                    "optimizer_state_stripped_count": 2,
                }
            )
        ),
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    monkeypatch.setattr(evaluator, "_write_codex_teacher_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.grade_with_codex",
        lambda *args, **kwargs: validate_codex_teacher_receipt(
            receipt_for(packet), packet, "test-model",
        ),
    )
    evaluator._five_agent_by_trace["five:trace:2"] = 2
    evaluator._five_action_by_trace["five:trace:2"] = 1
    live_outcomes = []
    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"
        called = 0
        def __init__(self):
            self.lifecycle_calls = []
        def pop_five_entry_lifecycle_packet(self, trace_id):
            return tuple(traces)
        def record_five_live_owner_outcome(self, **kwargs):
            live_outcomes.append(dict(kwargs))
            return True
        def apply_five_entry_lifecycle_reward(self, trace_id, *args, **kwargs):
            self.lifecycle_calls.append((trace_id, args, kwargs))
            return {"accepted": True, "reward_v4_r": 0.1}
        def optimize_five_rlhf_trace_packet(self, rows, scorer, **kwargs):
            self.called += 1
            return {"rlhf_ppo_update": True, "rlhf_reward_mean": 0.1}
    policy = Policy()
    assert evaluator._record_five_competition_lifecycles((lifecycle,), policy) == 1
    assert len(policy.lifecycle_calls) == 1
    assert live_outcomes == [{
        "agent_index": 2,
        "action": 1,
        "net_r": pytest.approx(0.8),
        "trace_id": "five:trace:2",
    }]
    assert evaluator._last_codex_teacher_phase["live_owner_outcome_records"] == 1
    assert evaluator._last_codex_teacher_phase[
        "live_owner_outcome_samples_by_action"
    ] == {"1": 1}
    assert evaluator._last_codex_teacher_phase["live_owner_outcome_negative_count"] == 0
    assert evaluator._last_codex_teacher_phase["submitted"] == 1
    assert len(evaluator._codex_teacher_pending) == 1
    assert evaluator._codex_teacher_executor._max_workers == 2
    evaluator._drain_codex_teacher_results(policy, wait=True)
    evaluator._codex_teacher_executor.shutdown(wait=True)
    evaluator._codex_teacher_executor = None
    assert evaluator._last_codex_teacher_phase["updates"] == policy.called == 1
    assert recovery_calls == [(policy, 1)]
    assert evaluator._last_codex_teacher_phase["recovery_snapshot_updates"] == 1
    assert evaluator._last_codex_teacher_phase["recovery_snapshot_path"] == "candidate.recovery.pkl"
    assert evaluator._last_codex_teacher_phase[
        "recovery_snapshot_optimizer_state_stripped"
    ] is True
    assert evaluator._last_codex_teacher_phase[
        "recovery_snapshot_optimizer_state_stripped_count"
    ] == 2
    assert evaluator._last_codex_teacher_phase[
        "recovery_snapshot_optimizer_resume_mode"
    ] == "research_reinitialized"


test_teacher_collection_does_not_call_legacy_lifecycle_reward = test_teacher_collection_coexists_with_lifecycle_reward


def test_runtime_close_account_mark_reaches_codex_packet_only_for_unique_close(monkeypatch):
    lifecycle, traces = evidence("runtime-account-mark")
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    owner = traces[2]
    evaluator._five_agent_by_trace[owner.trace_id] = 2
    evaluator._five_action_by_trace[owner.trace_id] = 1
    captured = []

    def capture_submit(identity, lifecycle_arg, rows, *, target_trace_id=None):
        captured.append(lifecycle_arg)
        return True

    monkeypatch.setattr(evaluator, "_submit_codex_teacher_case", capture_submit)

    class Policy:
        def pop_five_entry_lifecycle_packet(self, trace_id):
            return tuple(traces)

    assert evaluator._record_five_competition_lifecycles(
        (lifecycle,),
        Policy(),
        account_context_after_close={
            "balance": 9_975.0,
            "equity": 9_975.0,
            "sequence": 15.0,
        },
    ) == 1
    assert len(captured) == 1
    traces[2].account_context = {"balance": 10_000.0, "equity": 10_000.0}
    packet = build_codex_teaching_packet(captured[0], traces)
    assert packet["account_impact"]["verification_ready"] is True
    assert packet["account_impact"]["balance_delta"] == pytest.approx(-25.0)


def test_stale_unowned_close_does_not_block_new_unique_account_mark(monkeypatch):
    """An unresolved historical row must not make the next close partial."""
    stale_lifecycle, stale_traces = evidence("stale-unowned")
    lifecycle, traces = evidence("new-owned")
    stale_owner = stale_traces[2]
    traces[2].account_context = {
        "balance": 1_000.0,
        "equity": 1_000.0,
        "sequence": 10.0,
    }
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    owner = traces[2]
    captured = []

    def capture_submit(identity, lifecycle_arg, rows, *, target_trace_id=None):
        captured.append((lifecycle_arg, target_trace_id))
        return True

    monkeypatch.setattr(evaluator, "_submit_codex_teacher_case", capture_submit)

    class Policy:
        def pop_five_entry_lifecycle_packet(self, trace_id):
            if trace_id == stale_owner.trace_id:
                return tuple(stale_traces)
            return tuple(traces) if trace_id == owner.trace_id else ()

    policy = Policy()
    # The first row has no owner.  It is intentionally left out of the
    # settled-owner set, but it must still be remembered as an old snapshot
    # row so it cannot block the next account mark.
    class NoOwnerPolicy:
        def pop_five_entry_lifecycle_packet(self, trace_id):
            return ()

    # Do not expose the stale owner on its first observation.  This mirrors a
    # delayed/provenance-missing close that must remain eligible for a later
    # retry without blocking newer account-mark attribution.
    assert evaluator._record_five_competition_lifecycles(
        (stale_lifecycle,),
        NoOwnerPolicy(),
    ) == 0
    assert stale_lifecycle.position_ticket not in {
        identity.split("|", 1)[0]
        for identity in evaluator._five_settled_lifecycle_ids
    }

    # Provenance becomes available only after the first snapshot, so both
    # owners can now be retried without making the old row the mark target.
    evaluator._five_agent_by_trace[stale_owner.trace_id] = 2
    evaluator._five_action_by_trace[stale_owner.trace_id] = 1
    evaluator._five_agent_by_trace[owner.trace_id] = 2
    evaluator._five_action_by_trace[owner.trace_id] = 1

    assert evaluator._record_five_competition_lifecycles(
        (stale_lifecycle, lifecycle),
        policy,
        account_context_after_close={"balance": 975.0, "equity": 975.0},
    ) == 2
    assert len(captured) == 2
    captured_by_ticket = {
        item[0].position_ticket: item[0]
        for item in captured
    }
    assert not getattr(captured_by_ticket[stale_lifecycle.position_ticket], "account_context_after_close", {})
    packet = build_codex_teaching_packet(
        captured_by_ticket[lifecycle.position_ticket], traces,
    )
    assert packet["account_impact"]["status"] == "confirmed"
    assert packet["account_impact"]["verification_ready"] is True
    assert packet["account_impact"]["balance_delta"] == pytest.approx(-25.0)


def test_terminal_entry_falls_back_to_action_packet_after_delayed_fill(monkeypatch):
    """A later DEAL_ADD must not lose the entry packet before Codex grading."""
    lifecycle, traces = evidence("delayed-fill")
    packet = build_codex_teaching_packet(lifecycle, traces)
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.grade_with_codex",
        lambda *args, **kwargs: validate_codex_teacher_receipt(
            receipt_for(packet), packet, "test-model",
        ),
    )
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._write_codex_teacher_status = lambda *args, **kwargs: None
    owner = traces[2]
    evaluator._five_agent_by_trace[owner.trace_id] = 2
    evaluator._five_action_by_trace[owner.trace_id] = 1
    consumed: list[str] = []

    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"

        def get_five_entry_lifecycle_packet(self, trace_id):
            return ()

        def get_five_action_packet(self, trace_id):
            return tuple(traces) if trace_id == owner.trace_id else ()

        def pop_five_action_packet(self, trace_id):
            consumed.append(str(trace_id))
            return tuple(traces)

        def optimize_five_rlhf_trace_packet(self, rows, scorer, **kwargs):
            assert len(rows) == 5
            assert scorer.trace_id == owner.trace_id
            return {"rlhf_ppo_update": True, "rlhf_reward_mean": 0.1}

    policy = Policy()
    assert evaluator._record_five_competition_lifecycles((lifecycle,), policy) == 1
    assert evaluator._last_codex_teacher_phase["submitted"] == 1
    evaluator._run_codex_teacher_phase(
        policy, reconciled=True, report_path=Path.cwd() / "delayed-fill.xml",
    )
    assert consumed == [owner.trace_id]
    assert evaluator._last_codex_teacher_phase["updates"] == 1


def test_hypothesis_teacher_entry_waits_for_counterfactual_before_submit(monkeypatch):
    lifecycle, traces = evidence("deferred-counterfactual")
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    owner = traces[2]
    evaluator._five_agent_by_trace[owner.trace_id] = 2
    evaluator._five_action_by_trace[owner.trace_id] = 1
    captured = []
    consumed = []

    def capture_submit(identity, lifecycle_arg, rows, *, target_trace_id=None):
        captured.append((lifecycle_arg, tuple(rows), target_trace_id))
        return True

    monkeypatch.setattr(evaluator, "_submit_codex_teacher_case", capture_submit)

    class Policy:
        def get_five_entry_lifecycle_packet(self, trace_id):
            return tuple(traces) if trace_id == owner.trace_id else ()

        def pop_five_entry_lifecycle_packet(self, trace_id):
            consumed.append(str(trace_id))
            return tuple(traces)

    policy = Policy()
    # The final close is visible, but the EA has not emitted its delayed
    # POST_EXIT_OBSERVATION row yet.  Entry ownership must remain pending so
    # the Codex receipt can compare the realized close with the hold baseline.
    assert evaluator._record_five_competition_lifecycles((lifecycle,), policy) == 1
    assert len(captured) == 0
    assert consumed == []
    assert evaluator._last_codex_teacher_phase["deferred"] == 1
    assert any("|entry|" in case_id for case_id in evaluator._codex_teacher_deferred_cases)

    refreshed = SimpleNamespace(**vars(lifecycle))
    refreshed.post_exit_horizon_bars = 12
    refreshed.post_exit_hold_r = 0.25
    assert evaluator._record_five_competition_lifecycles((refreshed,), policy) == 0
    assert len(captured) == 1
    assert consumed == [owner.trace_id]
    packet = build_codex_teaching_packet(captured[0][0], traces)
    assert packet["account_impact"]["counterfactual_status"] == "validated"
    assert packet["account_impact"]["counterfactual_horizon_bars"] == 12
    assert packet["account_impact"]["counterfactual_hold_r"] == pytest.approx(0.25)
    assert evaluator._last_codex_teacher_phase["deferred_released"] == 1
    assert evaluator._codex_teacher_deferred_cases == {}


def test_hypothesis_teacher_legacy_trace_waits_without_consumer(monkeypatch):
    """Legacy/context traces must not bypass the post-close hold label."""
    lifecycle, traces = evidence("legacy-no-consumer")
    owner = traces[2]
    owner.trace_id = "legacy:legacy-no-consumer:2"
    owner.action_details["codex_teacher_reference_policy_sha256"] = "reference"
    owner.action_details["codex_teacher_reference_policy_frozen"] = True
    lifecycle.metadata = {"position_owner_trace_id": owner.trace_id}
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    captured = []

    def capture_submit(identity, lifecycle_arg, rows, *, target_trace_id=None):
        if identity in evaluator._codex_teacher_submitted_ids:
            return False
        evaluator._codex_teacher_submitted_ids.add(str(identity))
        captured.append((lifecycle_arg, tuple(rows), target_trace_id))
        return True

    monkeypatch.setattr(evaluator, "_submit_codex_teacher_case", capture_submit)

    class Policy:
        def get_codex_teacher_trace(self, trace_id):
            return owner if trace_id == owner.trace_id else None

    policy = Policy()
    # Legacy traces have no packet-consumer callback.  That must not disable
    # the causal wait; otherwise MT5 submits a partial account mark before the
    # EA's delayed POST_EXIT_OBSERVATION row arrives.
    assert evaluator._record_five_competition_lifecycles((lifecycle,), policy) == 0
    assert captured == []
    assert evaluator._last_codex_teacher_phase["deferred"] == 1
    assert len(evaluator._codex_teacher_deferred_cases) == 1

    refreshed = SimpleNamespace(**vars(lifecycle))
    refreshed.post_exit_horizon_bars = 12
    refreshed.post_exit_hold_r = 0.25
    assert evaluator._record_five_competition_lifecycles((refreshed,), policy) == 0
    assert len(captured) == 1
    assert captured[0][0] is refreshed
    assert captured[0][2] == owner.trace_id
    packet = build_codex_teaching_packet(
        captured[0][0], captured[0][1], target_trace_id=captured[0][2],
    )
    assert packet["account_impact"]["counterfactual_status"] == "validated"
    assert packet["account_impact"]["counterfactual_horizon_bars"] == 12
    assert packet["account_impact"]["counterfactual_hold_r"] == pytest.approx(0.25)
    assert evaluator._codex_teacher_deferred_cases == {}


def test_hypothesis_teacher_management_still_defers_until_counterfactual(monkeypatch):
    lifecycle, traces = evidence("deferred-management")
    target = traces[1]
    target.action_details["five_action_index"] = 8
    lifecycle.exit_trace_id = target.trace_id
    lifecycle.exit_requested_action = 8
    lifecycle.exit_kind = "EXPERT"
    lifecycle.exit_controller = "five"
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    owner = traces[2]
    evaluator._five_agent_by_trace[owner.trace_id] = 2
    evaluator._five_action_by_trace[owner.trace_id] = 1
    evaluator._five_agent_by_trace[target.trace_id] = 2
    evaluator._five_action_by_trace[target.trace_id] = 8
    captured = []
    consumed = []

    def capture_submit(identity, lifecycle_arg, rows, *, target_trace_id=None):
        captured.append((lifecycle_arg, tuple(rows), target_trace_id))
        return True

    monkeypatch.setattr(evaluator, "_submit_codex_teacher_case", capture_submit)

    class Policy:
        def get_five_entry_lifecycle_packet(self, trace_id):
            return tuple(traces) if trace_id == owner.trace_id else ()

        def pop_five_entry_lifecycle_packet(self, trace_id):
            consumed.append(str(trace_id))
            return tuple(traces)

        def get_five_action_packet(self, trace_id):
            return tuple(traces) if trace_id == target.trace_id else ()

        def pop_five_action_packet(self, trace_id):
            consumed.append(str(trace_id))
            return tuple(traces)

    assert evaluator._record_five_competition_lifecycles((lifecycle,), Policy()) == 1
    assert len(captured) == 0  # both entry and management await the hold horizon
    assert consumed == []
    assert any("|entry|" in case_id for case_id in evaluator._codex_teacher_deferred_cases)
    assert any("|action|" in case_id for case_id in evaluator._codex_teacher_deferred_cases)


def test_management_deferred_case_refresh_uses_new_post_exit_lifecycle_object(monkeypatch):
    """A refreshed immutable ledger object must replace the pre-horizon case."""
    lifecycle, traces = evidence("deferred-refresh")
    target = traces[1]
    target.action_details["five_action_index"] = 8
    lifecycle.exit_trace_id = target.trace_id
    lifecycle.exit_requested_action = 8
    lifecycle.exit_kind = "EXPERT"
    lifecycle.exit_controller = "five"
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    owner = traces[2]
    evaluator._five_agent_by_trace[owner.trace_id] = 2
    evaluator._five_action_by_trace[owner.trace_id] = 1
    evaluator._five_agent_by_trace[target.trace_id] = 2
    evaluator._five_action_by_trace[target.trace_id] = 8
    captured = []
    consumed = []

    def capture_submit(identity, lifecycle_arg, rows, *, target_trace_id=None):
        captured.append((lifecycle_arg, tuple(rows), target_trace_id))
        return True

    monkeypatch.setattr(evaluator, "_submit_codex_teacher_case", capture_submit)

    class Policy:
        def get_five_entry_lifecycle_packet(self, trace_id):
            return tuple(traces) if trace_id == owner.trace_id else ()

        def pop_five_entry_lifecycle_packet(self, trace_id):
            consumed.append(str(trace_id))
            return tuple(traces)

        def get_five_action_packet(self, trace_id):
            return tuple(traces) if trace_id == target.trace_id else ()

        def pop_five_action_packet(self, trace_id):
            consumed.append(str(trace_id))
            return tuple(traces)

    policy = Policy()
    assert evaluator._record_five_competition_lifecycles((lifecycle,), policy) == 1
    assert len(captured) == 0
    assert len(evaluator._codex_teacher_deferred_cases) == 2
    assert any("|entry|" in case_id for case_id in evaluator._codex_teacher_deferred_cases)
    assert any("|action|" in case_id for case_id in evaluator._codex_teacher_deferred_cases)

    refreshed = SimpleNamespace(**vars(lifecycle))
    refreshed.post_exit_horizon_bars = 12
    refreshed.post_exit_hold_r = 0.25
    assert evaluator._record_five_competition_lifecycles((refreshed,), policy) == 0
    assert len(captured) == 2
    assert {id(item[0]) for item in captured} == {id(refreshed)}
    assert consumed == [owner.trace_id, target.trace_id]
    packets = [build_codex_teaching_packet(item[0], traces) for item in captured]
    assert all(item["account_impact"]["counterfactual_status"] == "validated" for item in packets)
    assert all(item["account_impact"]["counterfactual_horizon_bars"] == 12 for item in packets)
    assert all(item["account_impact"]["counterfactual_hold_r"] == pytest.approx(0.25) for item in packets)
    assert evaluator._last_codex_teacher_phase["deferred_forced_partial"] == 0


def test_terminal_management_trace_gets_its_own_codex_case(monkeypatch):
    lifecycle, traces = evidence("management")
    target = traces[1]
    target.action_details["five_action_index"] = 8
    lifecycle.exit_trace_id = target.trace_id
    lifecycle.exit_requested_action = 8
    lifecycle.exit_kind = "EXPERT"
    lifecycle.exit_controller = "five"
    lifecycle.realized_r = 0.75
    lifecycle.net_pnl = 25.0
    lifecycle.post_exit_hold_r = 0.875
    lifecycle.post_exit_horizon_bars = 12
    packets = {}

    def fake_grade(packet, **kwargs):
        packets[str(packet["trace_id"])] = packet
        return validate_codex_teacher_receipt(receipt_for(packet), packet, "test-model")

    monkeypatch.setattr("self_growing_trader.codex_teacher.grade_with_codex", fake_grade)
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    monkeypatch.setattr(evaluator, "_write_codex_teacher_status", lambda *args, **kwargs: None)
    evaluator._five_agent_by_trace[traces[2].trace_id] = 2
    evaluator._five_action_by_trace[traces[2].trace_id] = 1
    evaluator._five_agent_by_trace[target.trace_id] = 2
    evaluator._five_action_by_trace[target.trace_id] = 8

    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"
        called = 0
        def pop_five_entry_lifecycle_packet(self, trace_id):
            return tuple(traces)
        def pop_five_action_packet(self, trace_id):
            return tuple(traces)
        def optimize_five_rlhf_trace_packet(self, rows, scorer, **kwargs):
            self.called += 1
            target_row = next(
                row for row in rows if str(row.trace_id) == scorer.trace_id
            )
            action = int(target_row.action_details["five_action_index"])
            reward = scorer.score(
                causal_policy_feature_vector(
                    target_row.raw_obs, action, action_count=12,
                )
            )
            return {"rlhf_ppo_update": True, "rlhf_reward_mean": reward}

    policy = Policy()
    assert evaluator._record_five_competition_lifecycles((lifecycle,), policy) == 1
    assert evaluator._last_codex_teacher_phase["submitted"] == 2
    phase = evaluator._run_codex_teacher_phase(
        policy, reconciled=True, report_path=Path.cwd() / "report.xml",
    )
    assert policy.called == 2
    assert packets[target.trace_id]["action_name"] == "CLOSE_WORST"
    assert packets[target.trace_id]["action_surface_count"] == 12
    management_update = next(
        item for item in phase["update_receipts"]
        if item["trace_id"] == target.trace_id
    )
    assert management_update["account_impact_reward_source"] == (
        "management_counterfactual_delta"
    )
    assert management_update["account_impact_reward"] == pytest.approx(-0.125)
    assert management_update["scaled_reward"] < 0.0


def test_phase_grades_and_updates_exact_owner(monkeypatch):
    lifecycle, traces = evidence()
    packet = build_codex_teaching_packet(lifecycle, traces)
    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"
        called = 0
        def optimize_five_rlhf_trace_packet(self, rows, scorer, **kwargs):
            self.called += 1
            assert len(rows) == 5
            assert scorer.trace_id == "five:trace:2"
            assert kwargs["reference_policy_sha256"] == "reference"
            return {"rlhf_ppo_update": True, "rlhf_reward_mean": scorer.score(
                causal_policy_feature_vector(packet["raw_obs137"], packet["action"])
            )}
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.grade_with_codex",
        lambda *args, **kwargs: validate_codex_teacher_receipt(receipt_for(packet), packet, "test-model"),
    )
    sink = io.StringIO()
    class OpenSink:
        def __enter__(self):
            return sink
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: OpenSink())
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    monkeypatch.setattr(evaluator, "_write_codex_teacher_status", lambda *args, **kwargs: None)
    assert evaluator._submit_codex_teacher_case(
        "12345|five:trace:2", lifecycle, tuple(traces),
    ) is True
    policy = Policy()
    phase = evaluator._run_codex_teacher_phase(
        policy, reconciled=True, report_path=Path.cwd() / "report.xml",
    )
    assert phase["graded"] == phase["updates"] == policy.called == 1
    assert phase["failures"] == 0
    log_records = [json.loads(line) for line in sink.getvalue().splitlines() if line]
    assert [item["record_type"] for item in log_records] == [
        "codex_teacher_receipt",
        "codex_teacher_update",
    ]
    assert log_records[0]["trace_id"] == "five:trace:2"
    assert log_records[1]["trace_id"] == "five:trace:2"


def test_phase_skips_unconfirmed_hypothesis_before_optimizer(monkeypatch):
    lifecycle, traces = evidence("hypothesis-rejected")
    packet = build_codex_teaching_packet(lifecycle, traces)
    rejected = validate_codex_teacher_receipt(
        hypothesis_receipt_for(packet, status="rejected", predicted_sign=-1),
        packet, "test-model", require_hypothesis=True,
    )
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.grade_with_codex",
        lambda *args, **kwargs: rejected,
    )
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    monkeypatch.setattr(evaluator, "_write_codex_teacher_status", lambda *args, **kwargs: None)
    evaluator._submit_codex_teacher_case("hypothesis-rejected", lifecycle, tuple(traces))

    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"
        called = 0
        def optimize_five_rlhf_trace_packet(self, *args, **kwargs):
            self.called += 1
            return {"rlhf_ppo_update": True}

    policy = Policy()
    phase = evaluator._run_codex_teacher_phase(
        policy, reconciled=True, report_path=Path.cwd() / "hypothesis-rejected.xml",
    )
    assert phase["hypotheses"] == 1
    assert phase["hypotheses_confirmed"] == 0
    assert phase["hypothesis_rejections"] == 1
    assert phase["interventions_applied"] == 0
    assert policy.called == 0


def test_codex_stub_smoke_records_confirmed_update_and_audit_fields(monkeypatch):
    lifecycle, traces = evidence("stub-smoke")
    lifecycle.realized_r = -0.8
    lifecycle.net_pnl = -25.0
    lifecycle.post_exit_hold_r = 0.25
    lifecycle.post_exit_horizon_bars = 12
    packet = build_codex_teaching_packet(lifecycle, traces)
    receipt = validate_codex_teacher_receipt(
        hypothesis_receipt_for(packet, status="confirmed", predicted_sign=-1),
        packet, "gpt-5.6-luna", require_hypothesis=True,
    )
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.grade_with_codex",
        lambda *args, **kwargs: receipt,
    )
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True, codex_teacher_model="gpt-5.6-luna",
        codex_teacher_executable="desktop-codex", codex_teacher_cli_version="0.155.0",
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_receipts_path = writable_test_dir("stub_smoke") / "receipts.jsonl"
    evaluator._write_codex_teacher_status = lambda *args, **kwargs: None

    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"

        def optimize_five_rlhf_trace_packet(self, rows, scorer, **kwargs):
            assert scorer.score(causal_policy_feature_vector(packet["raw_obs137"], packet["action"])) < 0.5
            return {
                "rlhf_ppo_update": True,
                "rlhf_reward_mean": scorer.score(
                    causal_policy_feature_vector(packet["raw_obs137"], packet["action"])
                ),
                "rlhf_ppo_advantage_mean": -0.1,
                "rlhf_parameter_hash_before": "before",
                "rlhf_parameter_hash_after": "after",
                "rlhf_target_action_probability_before": 0.2,
                "rlhf_target_action_probability_after": 0.19,
                "rlhf_packet_max_probability_shift": 0.01,
                "rlhf_codex_learning_rate": 0.005,
                "rlhf_optimizer_learning_rate_before": 0.00005,
                "rlhf_optimizer_learning_rate_for_update": 0.005,
                "rlhf_optimizer_learning_rate_restored": True,
                "rlhf_codex_context_similarity_threshold": 0.995,
                "rlhf_codex_context_support_floor": 0.5,
                "rlhf_codex_context_support_target": 1.0,
                "rlhf_codex_context_support_max": 1.0,
                "rlhf_codex_context_prototype_captured": True,
                "rlhf_codex_context_prototype_count": 1,
                "rlhf_parameter_scope": "active_action_head_bounded_v1",
                "rlhf_trainable_parameter_tensors": 3,
            }

    evaluator._submit_codex_teacher_case("stub-smoke", lifecycle, tuple(traces))
    phase = evaluator._run_codex_teacher_phase(
        Policy(), reconciled=True, report_path=Path.cwd() / "stub-smoke-report.xml",
    )
    assert phase["submitted"] == phase["graded"] == phase["updates"] == 1
    assert phase["hypotheses_confirmed"] == 1
    assert phase["interventions_applied"] == 1
    assert phase["counterfactual_validated"] == 1
    assert phase["counterfactual_unavailable"] == 0
    update = phase["update_receipts"][0]
    assert update["teacher_reward"] > 0.0
    assert update["verification_status"] == "confirmed"
    assert update["account_impact_status"] == "partial"
    assert update["account_impact_reward"] == pytest.approx(-0.9)
    assert update["account_impact_reward_source"] == "immutable_lifecycle_partial"
    assert update["account_impact_observed_realized_r"] == pytest.approx(-0.8)
    assert update["account_impact_observed_net_pnl"] == pytest.approx(-25.0)
    assert update["account_impact_balance_delta"] is None
    assert update["account_impact_equity_delta"] is None
    assert update["counterfactual_status"] == "validated"
    assert update["counterfactual_horizon_bars"] == 12
    assert update["counterfactual_hold_r"] == pytest.approx(0.25)
    assert update["counterfactual_delta_r"] == pytest.approx(-1.05)
    assert update["scaled_reward"] < 0.0
    assert update["rlhf_ppo_update"] is True
    assert update["rlhf_packet_max_probability_shift"] == pytest.approx(0.01)
    assert update["rlhf_codex_learning_rate"] == pytest.approx(0.005)
    assert update["rlhf_optimizer_learning_rate_restored"] is True
    assert update["rlhf_codex_context_similarity_threshold"] == pytest.approx(0.995)
    assert update["rlhf_codex_context_support_floor"] == pytest.approx(0.5)
    assert update["rlhf_codex_context_prototype_count"] == 1
    assert update["rlhf_target_action_probability_after"] == pytest.approx(0.19)
    log_path = evaluator._codex_teacher_receipt_file(
        Path.cwd() / "stub-smoke-report.xml"
    )
    log_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    update_record = next(
        item for item in log_records if item.get("record_type") == "codex_teacher_update"
    )
    assert update_record["accepted"] is True
    assert update_record["trace_id"] == packet["trace_id"]
    assert update_record["position_ticket"] == packet["position_ticket"]
    assert update_record["packet_sha256"] == packet["packet_sha256"]
    assert update_record["receipt_sha256"] == receipt["receipt_sha256"]
    assert update_record["teacher_reward"] == pytest.approx(receipt["teacher_reward"])
    assert update_record["scaled_reward"] < 0.0
    assert update_record["advantage"] == pytest.approx(-0.1)
    assert update_record["rlhf_ppo_update"] is True
    assert update_record["rlhf_parameter_scope"] == "active_action_head_bounded_v1"
    assert update_record["account_impact_reward"] == pytest.approx(-0.9)
    assert update_record["account_impact_reward_source"] == "immutable_lifecycle_partial"
    assert update_record["action_probability_before"] == pytest.approx(0.2)
    assert update_record["action_probability_after"] == pytest.approx(0.19)
    assert update_record["account_impact"]["observed_net_pnl"] == pytest.approx(-25.0)


def test_phase_rejects_noop_mt5_update_contract(monkeypatch):
    """A stale learner cannot make a no-op look like a Codex intervention."""
    lifecycle, traces = evidence("noop-contract")
    packet = build_codex_teaching_packet(lifecycle, traces)
    receipt = validate_codex_teacher_receipt(
        hypothesis_receipt_for(packet, status="confirmed", predicted_sign=1),
        packet,
        "gpt-5.6-luna",
        require_hypothesis=True,
    )
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.grade_with_codex",
        lambda *args, **kwargs: receipt,
    )
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(),
        offline_replay=True,
        codex_teacher_enabled=True,
        codex_teacher_hypothesis_required=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_receipts_path = writable_test_dir("noop_contract") / "receipts.jsonl"
    evaluator._write_codex_teacher_status = lambda *args, **kwargs: None

    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"

        def optimize_five_rlhf_trace_packet(self, *args, **kwargs):
            return {
                "rlhf_mt5_trace_packet": True,
                "rlhf_ppo_update": True,
                "rlhf_reward_mean": -0.1,
                "rlhf_direction_expected": "decrease",
                "rlhf_directional_update": False,
                "rlhf_parameter_changed": False,
                "rlhf_parameter_hash_before": "same",
                "rlhf_parameter_hash_after": "same",
                "rlhf_target_action_probability_before": 0.2,
                "rlhf_target_action_probability_after": 0.2,
                "rlhf_packet_max_probability_shift": 0.0,
            }

    evaluator._submit_codex_teacher_case(
        "noop-contract", lifecycle, tuple(traces)
    )
    phase = evaluator._run_codex_teacher_phase(
        Policy(),
        reconciled=True,
        report_path=Path.cwd() / "noop-contract-report.xml",
    )
    assert phase["submitted"] == phase["graded"] == 1
    assert phase["updates"] == 0
    assert phase["interventions_applied"] == 0
    assert phase["rejected_updates"] == 1
    assert phase["rejection_records"][0]["reason"] == (
        "update_contract_parameter_hash_unchanged"
    )
    log_path = evaluator._codex_teacher_receipt_file(
        Path.cwd() / "noop-contract-report.xml"
    )
    log_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rejected_update = next(
        item for item in log_records if item.get("record_type") == "codex_teacher_update"
    )
    assert rejected_update["accepted"] is False
    assert rejected_update["rlhf_update_rejected"] is True
    assert rejected_update["rlhf_rejection_reason"] == (
        "update_contract_parameter_hash_unchanged"
    )
    assert rejected_update["trace_id"] == packet["trace_id"]
    assert rejected_update["position_ticket"] == packet["position_ticket"]


def test_real_codex_evaluator_drain_applies_bounded_update(monkeypatch):
    """Exercise queue -> receipt -> drain -> real Five optimizer end to end."""
    torch.manual_seed(705)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=6,
        seed=705,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    assert policy.install_five_rlhf_reference()["installed"] is True
    traces = tuple(
        policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
    )
    owner = next(
        trace
        for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="e2e-codex",
        realized_r=-0.5,
        mfe_r=0.1,
        mae_r=-0.6,
        commission=0.0,
        swap=0.0,
        close_reason="SL",
        side=1,
        entry_count=1,
        entry_price=100.0,
        initial_sl_price=99.0,
        initial_tp_price=102.0,
        initial_risk=1.0,
        initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(
        lifecycle, traces, target_trace_id=owner.trace_id
    )
    raw = hypothesis_receipt_for(packet, status="confirmed", predicted_sign=-1)
    raw.update(teacher_reward=-0.5, confidence=1.0)
    receipt = validate_codex_teacher_receipt(
        raw, packet, "gpt-5.6-luna", require_hypothesis=True
    )
    monkeypatch.setattr(
        "self_growing_trader.codex_teacher.grade_with_codex",
        lambda *args, **kwargs: receipt,
    )
    config = MT5BacktestConfig(
        runner_dir=Path.cwd(),
        offline_replay=True,
        codex_teacher_enabled=True,
        codex_teacher_model="gpt-5.6-luna",
        codex_teacher_hypothesis_required=True,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    evaluator._codex_teacher_receipts_path = (
        writable_test_dir("e2e_drain") / "receipts.jsonl"
    )
    evaluator._write_codex_teacher_status = lambda *args, **kwargs: None
    assert evaluator._submit_codex_teacher_case(
        "e2e-codex", lifecycle, traces, target_trace_id=owner.trace_id
    ) is True

    phase = evaluator._run_codex_teacher_phase(
        policy,
        reconciled=True,
        report_path=Path.cwd() / "e2e-codex-report.xml",
    )

    assert phase["submitted"] == phase["graded"] == 1
    assert phase["failures"] == 0
    assert phase["hypotheses_confirmed"] == 1
    assert phase["updates"] == 1
    assert phase["interventions_applied"] == 1
    assert phase.get("rejected_updates", 0) == 0
    update = phase["update_receipts"][0]
    assert update["rlhf_ppo_update"] is True
    assert update["rlhf_directional_update"] is True
    assert update["parameter_hash_before"] != update["parameter_hash_after"]
    assert update["action_probability_after"] < update["action_probability_before"]
    assert update["target_action_argmax_before"] in range(7)
    assert update["target_action_argmax_after"] in range(7)
    assert update["target_action_rank_before"] in range(1, 8)
    assert update["target_action_rank_after"] in range(1, 8)
    assert update["target_action_logit_delta"] < 0.0
    assert update["target_action_probability_delta"] < 0.0
    assert update["target_action_allowed"] is True
    assert update["rlhf_internal_policy_state_parameter_changed"] is True
    assert update["rlhf_internal_policy_state_trainable_parameter_tensors"] == 8
    assert update["rlhf_codex_global_intervention_residual_fraction"] > 0.0
    assert update["rlhf_codex_global_intervention_shift_rejections"] >= 0
    assert update["rlhf_codex_global_intervention_direction_rejections"] >= 0
    assert update["rlhf_codex_global_intervention_nonfinite_rejections"] >= 0
    assert update["rlhf_codex_global_action_bias_enabled"] is True
    assert update["rlhf_codex_global_action_bias_scale"] == pytest.approx(0.50)
    assert update["rlhf_codex_global_action_bias_max_abs"] > 0.0


def test_unlimited_profile_grades_every_final_close_beyond_two(monkeypatch):
    packets = [build_codex_teaching_packet(*evidence(str(index))) for index in range(3)]
    packet_by_hash = {packet["packet_sha256"]: packet for packet in packets}

    def fake_grade(packet, **kwargs):
        expected = packet_by_hash[packet["packet_sha256"]]
        return validate_codex_teacher_receipt(
            receipt_for(expected), expected, "test-model",
        )

    monkeypatch.setattr("self_growing_trader.codex_teacher.grade_with_codex", fake_grade)
    sink = io.StringIO()
    class OpenSink:
        def __enter__(self):
            return sink
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: OpenSink())

    config = MT5BacktestConfig(
        runner_dir=Path.cwd(), offline_replay=True, codex_teacher_enabled=True,
        codex_teacher_max_cases_per_run=0,
        rlhf_source_checkpoint_sha256="source",
    )
    evaluator = MT5StrategyTesterEvaluator(config)
    monkeypatch.setattr(evaluator, "_write_codex_teacher_status", lambda *args, **kwargs: None)

    class Policy:
        five_rlhf_reference_policy_sha256 = "reference"
        called = 0
        def optimize_five_rlhf_trace_packet(self, rows, scorer, **kwargs):
            self.called += 1
            return {
                "rlhf_ppo_update": True,
                "rlhf_reward_mean": 0.1,
                # Zero is a valid HOLD argmax and must not be replaced by the
                # receipt serializer's missing-value sentinel (-1).
                "rlhf_target_action_argmax_before": 0,
                "rlhf_target_action_argmax_after": 0,
                "rlhf_target_action_rank_before": 1,
                "rlhf_target_action_rank_after": 1,
                "rlhf_target_action_probability_delta": 0.0,
                "rlhf_target_action_logit_delta": 0.0,
                "rlhf_target_action_allowed": True,
            }

    for index in range(3):
        lifecycle, traces = evidence(str(index))
        assert evaluator._submit_codex_teacher_case(
            f"case-{index}", lifecycle, tuple(traces),
        ) is True

    policy = Policy()
    phase = evaluator._run_codex_teacher_phase(
        policy, reconciled=True, report_path=Path.cwd() / "report.xml",
    )
    assert phase["submitted"] == phase["graded"] == phase["updates"] == 3
    assert phase["pending"] == 0
    assert policy.called == 3
    log_records = [json.loads(line) for line in sink.getvalue().splitlines() if line]
    assert len(log_records) == 6
    assert sum(item["record_type"] == "codex_teacher_receipt" for item in log_records) == 3
    assert sum(item["record_type"] == "codex_teacher_update" for item in log_records) == 3
    assert all(
        receipt["target_action_argmax_before"] == 0
        and receipt["target_action_argmax_after"] == 0
        for receipt in phase["update_receipts"]
    )


def test_partial_positive_account_impact_stays_contextual():
    """Weak positive account evidence must not globalize one action."""
    torch.manual_seed(716)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=716,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(
        np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32)
    )
    owner = next(
        trace for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="partial-positive",
        realized_r=0.4, mfe_r=0.7, mae_r=-0.1,
        commission=0.0, swap=0.0, close_reason="TP", side=1,
        entry_count=1, entry_price=100.0, initial_sl_price=99.0,
        initial_tp_price=102.0, initial_risk=1.0, initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = copy.deepcopy(build_codex_teaching_packet(lifecycle, traces))
    packet["account_impact"].update({
        "status": "partial",
        "verification_ready": False,
        "observed_realized_r": 0.4,
        "observed_net_pnl": 4.0,
        "balance_delta": 4.0,
        "equity_delta": 4.0,
    })
    raw = receipt_for(packet)
    raw.update(
        teacher_reward=0.4,
        confidence=1.0,
        receipt_sha256="partial-positive-test",
    )
    scorer = CodexTeacherRewardModel(packet, raw, max_probability_shift=0.05)
    bias_before = policy.five_shared_network.heads.codex_action_bias.detach().clone()
    result = policy.optimize_five_rlhf_trace_packet(
        traces,
        scorer,
        source_checkpoint_sha256="source",
        config=MT5BacktestConfig(
            codex_teacher_learning_rate=0.005,
            codex_context_similarity_threshold=0.995,
            codex_context_support_floor=0.5,
        ),
    )
    bias_after = policy.five_shared_network.heads.codex_action_bias.detach()
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_codex_global_intervention_suppressed"] is True
    assert result["rlhf_codex_global_intervention_applied"] is False
    assert result["rlhf_codex_global_intervention_reason"] == (
        "global_positive_suppressed_partial_account_impact"
    )
    assert torch.equal(bias_after, bias_before)
    assert result["rlhf_target_action_probability_after"] > result[
        "rlhf_target_action_probability_before"
    ]


@pytest.mark.parametrize(
    ("learning_rate", "backtrack_expected"),
    ((0.005, False), (0.02, False)),
)
def test_real_codex_update_is_scoped_to_action_head_and_bounded(
    learning_rate: float,
    backtrack_expected: bool,
):
    torch.manual_seed(705)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=705,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
    # Simulate the live candidate: generic PPO has already initialized
    # FiveAdam state for the Codex contextual residual before the delayed
    # receipt is drained. This catches accidental Tensor-key lookups in the
    # Codex snapshot/rollback path.
    trainer = policy._ensure_five_trainer()
    for parameter in policy.five_shared_network.parameters():
        # Reproduce a pre-identity checkpoint after a generic PPO update:
        # every parameter may still be keyed by the Parameter object.  The
        # live Codex path must snapshot/remove this state without asking a
        # Tensor to produce a multi-element boolean during dict lookup.
        trainer.optimizer.state[parameter] = {
            "step": 0,
            "m": torch.zeros_like(parameter),
            "v": torch.zeros_like(parameter),
        }
    owner = next(
        trace for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    owner_action = int(
        owner.action_details.get("five_action_index", owner.selected_idx)
    )
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="scoped", realized_r=-0.5, mfe_r=0.1, mae_r=-0.6,
        commission=0.0, swap=0.0, close_reason="SL", side=1, entry_count=1,
        entry_price=100.0, initial_sl_price=99.0, initial_tp_price=102.0,
        initial_risk=1.0, initial_atr=0.4, pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, traces)
    raw = receipt_for(packet)
    raw.update(teacher_reward=-0.5, confidence=0.9)
    receipt = validate_codex_teacher_receipt(raw, packet, "test-model")
    scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)
    codex_bias_before = policy.five_shared_network.heads.codex_action_bias.detach().clone()
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.five_shared_network.named_parameters()
    }
    result = policy.optimize_five_rlhf_trace_packet(
        traces,
        scorer,
        source_checkpoint_sha256="source",
        # The live MT5 drain passes its runtime config so the Codex-specific
        # learning-rate/context controls are available.  The implementation
        # must keep this object out of the generic PPO helper, which accepts
        # only RLHFPPOConfig (or None).
        config=MT5BacktestConfig(
            codex_teacher_learning_rate=learning_rate,
            codex_context_similarity_threshold=0.995,
            codex_context_support_floor=0.5,
        ),
    )
    assert all(isinstance(key, int) for key in trainer.optimizer.state)
    changed = {
        name for name, parameter in policy.five_shared_network.named_parameters()
        if not torch.equal(parameter, before[name])
    }
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_teacher_reward"] == pytest.approx(-0.5)
    assert result["rlhf_account_impact_reward"] is None
    assert result["rlhf_combined_reward"] == pytest.approx(-0.5)
    assert result["rlhf_reward_composition"] == (
        "teacher_account_impact_50_50_account_negative_floor_v1"
    )
    assert result["rlhf_reward_direction_source"] == "teacher_reward"
    assert result["rlhf_packet_max_probability_shift"] <= 0.05
    assert result["rlhf_parameter_scope"] == (
        "codex_context_action_residual_v3_internal_policy_state_global_bias"
        "_factorized_parameter_heads"
    )
    assert result["rlhf_factorized_geometry_update"] is True
    assert {"delay", "risk", "entry_offset", "sl", "tp", "trailing"}.issubset(
        set(result["rlhf_factorized_geometry_heads"])
    )
    assert result["rlhf_factorized_geometry_parameter_changed"] is True
    assert result["rlhf_factorized_geometry_parameter_hash_before"] != result[
        "rlhf_factorized_geometry_parameter_hash_after"
    ]
    ratio = float(result["rlhf_factorized_geometry_ratio"])
    assert result["rlhf_factorized_geometry_clip_active"] is (
        ratio < 0.80 or ratio > 1.20
    )
    assert float(
        result["rlhf_factorized_geometry_unclipped_correction_scale"]
    ) == pytest.approx(0.05 if ratio < 0.80 or ratio > 1.20 else 0.0)
    assert result["rlhf_trainable_parameter_tensors"] > 0
    assert result["rlhf_internal_policy_state_trained"] is True
    assert result["rlhf_internal_policy_state_parameter_changed"] is True
    assert result["rlhf_internal_policy_state_trainable_parameter_tensors"] == 8
    assert result["rlhf_parameter_hash_before"] != result["rlhf_parameter_hash_after"]
    assert policy.get_codex_teacher_live_parameter_hash() == result[
        "rlhf_parameter_hash_after"
    ]
    assert result["rlhf_target_action_probability_before"] != pytest.approx(
        result["rlhf_target_action_probability_after"]
    )
    assert result["rlhf_target_action_probability_after"] < result[
        "rlhf_target_action_probability_before"
    ]
    assert float(result["rlhf_action_head_update_norm"]) > 0.0
    assert float(result["rlhf_action_head_target_delta"]) < 0.0
    assert float(result["rlhf_codex_learning_rate"]) <= learning_rate + 1.0e-12
    assert result.get("rlhf_codex_backtrack_attempts", 0) == 0
    assert result["rlhf_codex_global_intervention_applied"] is True
    assert abs(float(result["rlhf_codex_global_intervention_step"])) > 0.0
    assert 0.0 < float(result["rlhf_codex_global_intervention_residual_fraction"]) <= 1.0
    assert result["rlhf_packet_max_probability_shift"] > 1.0e-4
    assert result["rlhf_optimizer_learning_rate_restored"] is True
    assert result["rlhf_codex_context_prototype_count"] == 1
    assert result["rlhf_codex_context_prototype_captured"] is True
    assert result["rlhf_codex_context_support_target"] == pytest.approx(1.0)
    assert result["rlhf_codex_global_action_bias_enabled"] is True
    assert result["rlhf_codex_global_bias_gradient_frozen"] is True
    assert result["rlhf_codex_ppo_gradient_parameter_tensors"] == (
        result["rlhf_trainable_parameter_tensors"] - 1
    )
    assert result["rlhf_codex_global_action_bias_scale"] == pytest.approx(0.50)
    assert result["rlhf_codex_entry_bias_centering_enabled"] is True
    assert result["rlhf_codex_global_action_bias_max_abs"] <= 0.25 + 1.0e-8
    assert result["rlhf_codex_entry_mass_guard_active"] is True
    assert result["rlhf_codex_entry_probability_mass_after"] >= (
        result["rlhf_codex_entry_probability_mass_floor"] - 1.0e-6
    )
    assert result["rlhf_target_action_argmax_before"] in range(7)
    assert result["rlhf_target_action_argmax_after"] in range(7)
    assert result["rlhf_target_action_rank_before"] in range(1, 8)
    assert result["rlhf_target_action_rank_after"] in range(1, 8)
    assert result["rlhf_target_action_logit_delta"] < 0.0
    assert result["rlhf_target_action_probability_delta"] < 0.0
    assert result["rlhf_target_action_allowed"] is True
    assert result["rlhf_codex_global_action_bias_max_abs"] > 0.0
    assert bool(
        policy.five_shared_network.heads.state_dict()[
            "codex_action_surface_enabled"
        ].item()
    ) is True
    assert result["rlhf_reward_mean"] < 0.0
    codex_bias_after = policy.five_shared_network.heads.codex_action_bias.detach()
    non_target_indices = [index for index in range(codex_bias_after.numel()) if index != owner_action]
    assert torch.equal(codex_bias_after[non_target_indices], codex_bias_before[non_target_indices])
    assert codex_bias_after[owner_action].item() < codex_bias_before[owner_action].item()
    assert {
        "heads.codex_action_residual.2.weight",
        "heads.codex_action_residual.2.bias",
        "heads.codex_action_bias",
    }.issubset(changed)
    assert "heads.internal_policy_state_net.action_residual.2.weight" in changed
    assert "heads.internal_policy_state_net.action_residual.2.bias" in changed

    # Exercise the actual bridge linker with the same policy instance after
    # the real optimizer update.  This is stronger than checking the hash in
    # isolation: it proves the next-action boundary can observe the updated
    # Codex scope and record a causal link.
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=Path.cwd(), offline_replay=True)
    )
    evaluator._write_codex_teacher_status = lambda *args, **kwargs: None
    evaluator._codex_teacher_pending_next_action_links.append({
        "update_index": 1,
        "position_ticket": "scoped",
        "trace_id": owner.trace_id,
        "action": owner_action,
        "receipt_sha256": "receipt-scoped",
        "parameter_hash_after": result["rlhf_parameter_hash_after"],
        "barrier_passed": False,
    })
    evaluator._mark_codex_teacher_barrier_links(
        passed=True, reason="five_lifecycle", waited_sec=0.01,
    )
    linked = evaluator._consume_codex_teacher_next_action_links(
        policy=policy,
        request=SimpleNamespace(sequence=99, event_id="event-99"),
        trace_id="five:next-action",
        action=owner_action,
        score=0.5,
    )
    assert len(linked) == 1
    assert linked[0]["barrier_reason"] == "five_lifecycle"
    assert linked[0]["policy_parameter_hash_matches_update"] is True
    assert evaluator._last_codex_teacher_phase[
        "codex_causal_hash_matches"
    ] == 1


def test_negative_codex_update_changes_unseen_context_action_probability():
    """A negative receipt must affect a later context, not only its owner."""
    torch.manual_seed(709)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=709,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    owner_observation = np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32)
    traces = policy.act_five_agents(owner_observation)
    owner = next(
        trace for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    owner_action = int(owner.action_details.get("five_action_index", owner.selected_idx))
    unseen_observation = torch.linspace(
        0.91, -0.73, 5 * INPUT_COUNT, dtype=torch.float32
    ).reshape(5, INPUT_COUNT)
    network = policy.five_shared_network
    with torch.no_grad():
        _, _, unseen_before = network.forward_five(
            unseen_observation,
            network.initial_memories(),
            deterministic=True,
            behavior_uniform_mix=np.zeros(5, dtype=np.float32),
            behavior_entry_probability_floor=0.0,
        )
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="unseen-context", realized_r=-0.9, mfe_r=0.1, mae_r=-1.0,
        commission=0.0, swap=0.0, close_reason="SL", side=1, entry_count=1,
        entry_price=100.0, initial_sl_price=99.0, initial_tp_price=102.0,
        initial_risk=1.0, initial_atr=0.4, pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, traces)
    receipt = validate_codex_teacher_receipt(
        {**receipt_for(packet), "teacher_reward": -0.8, "confidence": 1.0},
        packet,
        "test-model",
    )
    scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)
    result = policy.optimize_five_rlhf_trace_packet(
        traces,
        scorer,
        source_checkpoint_sha256="source",
        config=MT5BacktestConfig(
            codex_teacher_learning_rate=0.005,
            codex_context_similarity_threshold=0.995,
            codex_context_support_floor=0.5,
        ),
    )
    with torch.no_grad():
        _, _, unseen_after = network.forward_five(
            unseen_observation,
            network.initial_memories(),
            deterministic=True,
            behavior_uniform_mix=np.zeros(5, dtype=np.float32),
            behavior_entry_probability_floor=0.0,
        )
        mask = torch.ones_like(unseen_before["action_logits"], dtype=torch.bool)
        before_probabilities = torch.softmax(
            unseen_before["action_logits"].masked_fill(~mask, -30.0), dim=-1
        )
        after_probabilities = torch.softmax(
            unseen_after["action_logits"].masked_fill(~mask, -30.0), dim=-1
        )
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_parameter_scope"] == (
        "codex_context_action_residual_v3_internal_policy_state_global_bias"
        "_factorized_parameter_heads"
    )
    assert result["rlhf_factorized_geometry_update"] is True
    assert {"delay", "risk", "entry_offset", "sl", "tp", "trailing"}.issubset(
        set(result["rlhf_factorized_geometry_heads"])
    )
    assert result["rlhf_factorized_geometry_parameter_changed"] is True
    assert result["rlhf_factorized_geometry_parameter_hash_before"] != result[
        "rlhf_factorized_geometry_parameter_hash_after"
    ]
    ratio = float(result["rlhf_factorized_geometry_ratio"])
    assert result["rlhf_factorized_geometry_clip_active"] is (
        ratio < 0.80 or ratio > 1.20
    )
    assert float(
        result["rlhf_factorized_geometry_unclipped_correction_scale"]
    ) == pytest.approx(0.05 if ratio < 0.80 or ratio > 1.20 else 0.0)
    assert result["rlhf_internal_policy_state_trained"] is True
    assert result["rlhf_internal_policy_state_parameter_changed"] is True
    assert float(unseen_after["codex_context_support"][0].item()) == pytest.approx(0.0)
    assert float(network.heads.codex_action_bias.detach()[owner_action].item()) < 0.0
    assert after_probabilities[0, owner_action].item() < before_probabilities[0, owner_action].item()
    assert result["rlhf_codex_global_intervention_applied"] is True
    assert float(
        before_probabilities[0, owner_action].item()
        - after_probabilities[0, owner_action].item()
    ) > 0.005


def test_codex_sequential_updates_follow_reward_sign_on_same_context():
    """A verified negative/positive receipt must move its action both ways."""
    torch.manual_seed(705)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=705,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
    lifecycle, _ = evidence("direction")
    lifecycle.metadata = {"position_owner_trace_id": next(
        trace.trace_id for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )}
    lifecycle.position_ticket = "direction"
    packet = build_codex_teaching_packet(lifecycle, traces)

    def apply_reward(value: float):
        raw = receipt_for(packet)
        raw.update(teacher_reward=value, confidence=1.0)
        receipt = validate_codex_teacher_receipt(raw, packet, "test-model")
        scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)
        return policy.optimize_five_rlhf_trace_packet(
            traces,
            scorer,
            source_checkpoint_sha256="source",
            config=MT5BacktestConfig(
                codex_teacher_learning_rate=0.005,
                codex_context_similarity_threshold=0.995,
                codex_context_support_floor=0.5,
            ),
        )

    negative = apply_reward(-0.5)
    assert negative["rlhf_ppo_update"] is True
    assert negative["rlhf_codex_context_support_target"] >= 0.5
    assert negative["rlhf_target_action_probability_after"] < negative[
        "rlhf_target_action_probability_before"
    ]

    positive = apply_reward(0.5)
    assert positive["rlhf_ppo_update"] is True
    assert positive["rlhf_codex_context_support_target"] >= 0.5
    assert positive["rlhf_target_action_probability_after"] > positive[
        "rlhf_target_action_probability_before"
    ]


def test_repeated_negative_codex_updates_do_not_accumulate_global_hold_bias():
    """A losing entry must not turn the shared Codex bias into HOLD collapse."""
    torch.manual_seed(705)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=705,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
    owner = next(
        trace for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    owner_action = int(owner.action_details.get("five_action_index", owner.selected_idx))
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="repeated-negative", realized_r=-0.9, mfe_r=0.1, mae_r=-1.0,
        commission=0.0, swap=0.0, close_reason="SL", side=1, entry_count=1,
        entry_price=100.0, initial_sl_price=99.0, initial_tp_price=102.0,
        initial_risk=1.0, initial_atr=0.4, pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, traces)
    accepted_updates = 0
    compensated_updates = 0
    for _ in range(8):
        raw = receipt_for(packet)
        raw.update(teacher_reward=-0.8, confidence=1.0)
        receipt = validate_codex_teacher_receipt(raw, packet, "test-model")
        scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)
        result = policy.optimize_five_rlhf_trace_packet(
            traces,
            scorer,
            source_checkpoint_sha256="source",
            config=MT5BacktestConfig(
                codex_teacher_learning_rate=0.0003,
                codex_context_similarity_threshold=0.995,
                codex_context_support_floor=0.5,
            ),
        )
        accepted_updates += int(bool(result["rlhf_ppo_update"]))
        compensated_updates += int(
            bool(result["rlhf_codex_entry_mass_compensation_applied"])
        )
        assert result["rlhf_codex_entry_mass_guard_mode"] == (
            "frozen_reference_cumulative"
        )
        assert result["rlhf_codex_entry_probability_mass_reference"] > 0.0
        assert result["rlhf_codex_entry_probability_mass_after"] >= (
            result["rlhf_codex_entry_probability_mass_cumulative_floor"] - 1.0e-6
        )
    assert accepted_updates == 8
    assert compensated_updates >= 0
    bias = policy.five_shared_network.heads.codex_action_bias.detach()
    assert bias[0].item() == pytest.approx(0.0, abs=1.0e-8)
    assert bias[owner_action].item() < 0.0
    other_indices = [index for index in range(bias.numel()) if index not in (0, owner_action)]
    assert torch.allclose(bias[other_indices], torch.zeros_like(bias[other_indices]))


def test_codex_entry_floor_recovery_records_unreachable_frozen_floor():
    """A pre-existing global drift uses current mass as a no-further-drop floor."""
    torch.manual_seed(7061)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=7061,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
    owner = next(
        trace for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    with torch.no_grad():
        # The reference was captured before this drift. Raising HOLD makes
        # the current entry mass materially lower than the frozen reference,
        # so restoring the old floor is impossible under a 0.05 guard.
        policy.five_shared_network.heads.codex_action_bias[0] = 2.0
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="unreachable-floor", realized_r=-0.8, mfe_r=0.1,
        mae_r=-1.0, commission=0.0, swap=0.0, close_reason="SL", side=1,
        entry_count=1, entry_price=100.0, initial_sl_price=99.0,
        initial_tp_price=102.0, initial_risk=1.0, initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, traces)
    receipt = validate_codex_teacher_receipt(
        {**receipt_for(packet), "teacher_reward": -0.6, "confidence": 1.0},
        packet,
        "test-model",
    )
    scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)
    result = policy.optimize_five_rlhf_trace_packet(
        traces,
        scorer,
        source_checkpoint_sha256="source",
        config=MT5BacktestConfig(
            codex_teacher_learning_rate=0.005,
            codex_context_similarity_threshold=0.995,
            codex_context_support_floor=0.5,
        ),
    )
    assert result["rlhf_codex_entry_mass_floor_recovery_active"] is True
    assert result["rlhf_codex_entry_probability_mass_cumulative_floor_original"] > (
        result["rlhf_codex_entry_probability_mass_cumulative_floor"]
    )
    assert result["rlhf_codex_entry_probability_mass_cumulative_floor"] == pytest.approx(
        result["rlhf_codex_entry_probability_mass_before"], abs=1.0e-6
    )
    assert result["rlhf_target_action_probability_after"] < result[
        "rlhf_target_action_probability_before"
    ]


def test_positive_codex_update_recovers_entry_floor_without_noop():
    """A positive receipt must not become a frozen-floor no-op."""
    torch.manual_seed(7061)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=7061,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
    owner = next(
        trace for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    with torch.no_grad():
        # This deterministic drift leaves enough entry mass to be recoverable,
        # but makes the positive contextual step cross the frozen-reference
        # floor before the compensation path is applied.
        policy.five_shared_network.heads.codex_action_bias[0] = 0.4
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="positive-floor", realized_r=0.4, mfe_r=0.7,
        mae_r=-0.1, commission=0.0, swap=0.0, close_reason="TP", side=1,
        entry_count=1, entry_price=100.0, initial_sl_price=99.0,
        initial_tp_price=102.0, initial_risk=1.0, initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, traces)
    packet["account_impact"].update({
        "status": "partial",
        "verification_ready": False,
        "observed_realized_r": 0.4,
        "observed_net_pnl": 4.0,
        "balance_delta": 4.0,
        "equity_delta": 4.0,
    })
    receipt = validate_codex_teacher_receipt(
        {**receipt_for(packet), "teacher_reward": 0.04, "confidence": 1.0},
        packet,
        "test-model",
    )
    result = policy.optimize_five_rlhf_trace_packet(
        traces,
        CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05),
        source_checkpoint_sha256="source",
        config=MT5BacktestConfig(
            codex_teacher_learning_rate=0.005,
            codex_context_similarity_threshold=0.995,
            codex_context_support_floor=0.5,
        ),
    )

    assert result["rlhf_ppo_update"] is True
    assert result.get("rlhf_update_rejected", False) is not True
    assert result["rlhf_parameter_changed"] is True
    assert result["rlhf_target_action_probability_after"] > result[
        "rlhf_target_action_probability_before"
    ]
    assert result["rlhf_packet_max_probability_shift"] <= 0.05 + 1.0e-8
    assert result["rlhf_codex_entry_probability_mass_after"] >= (
        result["rlhf_codex_entry_probability_mass_cumulative_floor"] - 1.0e-6
    )
    assert result["rlhf_codex_entry_mass_compensation_applied"] is True
    assert result["rlhf_codex_entry_mass_compensation_step"] < 0.0


def test_negative_codex_update_recovers_when_entry_floor_blocks_step():
    """A losing receipt uses a signed HOLD/target compensation before failing closed."""
    torch.manual_seed(7061)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=7061,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
    owner = next(
        trace for trace in traces
        if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
    )
    with torch.no_grad():
        policy.five_shared_network.heads.codex_action_bias[0] = 0.4
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="negative-floor", realized_r=-0.8, mfe_r=0.1,
        mae_r=-1.0, commission=0.0, swap=0.0, close_reason="SL", side=1,
        entry_count=1, entry_price=100.0, initial_sl_price=99.0,
        initial_tp_price=102.0, initial_risk=1.0, initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(lifecycle, traces)
    receipt = validate_codex_teacher_receipt(
        {**receipt_for(packet), "teacher_reward": -0.6, "confidence": 1.0},
        packet,
        "test-model",
    )
    result = policy.optimize_five_rlhf_trace_packet(
        traces,
        CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05),
        source_checkpoint_sha256="source",
        config=MT5BacktestConfig(
            codex_teacher_learning_rate=0.005,
            codex_context_similarity_threshold=0.995,
            codex_context_support_floor=0.5,
        ),
    )

    assert result["rlhf_ppo_update"] is True
    assert result.get("rlhf_update_rejected", False) is not True
    assert result["rlhf_parameter_changed"] is True
    assert result["rlhf_target_action_probability_after"] < result[
        "rlhf_target_action_probability_before"
    ]
    assert result["rlhf_packet_max_probability_shift"] <= 0.05 + 1.0e-8
    assert result["rlhf_codex_entry_probability_mass_after"] >= (
        result["rlhf_codex_entry_probability_mass_cumulative_floor"] - 1.0e-6
    )
    assert result["rlhf_codex_entry_mass_compensation_applied"] is True
    assert result["rlhf_codex_entry_mass_compensation_step"] < 0.0
    assert result["rlhf_codex_global_intervention_reason"].endswith(":hold_target")


def test_codex_entry_bias_centering_prevents_common_hold_drift():
    """Negative receipts for every entry keep the entry class alive."""
    heads = FactorizedActionHeads(hidden_size=8)
    with torch.no_grad():
        heads.codex_action_bias[1:7] = -1.0
        heads.codex_action_bias[7:] = 0.0
    heads.codex_action_bias_scale = 0.50
    heads.codex_entry_bias_centering_enabled = False
    raw = heads.effective_codex_action_bias(
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert float(raw[1:7].mean().item()) < 0.0

    heads.codex_entry_bias_centering_enabled = True
    centered = heads.effective_codex_action_bias(
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert centered[0].item() == pytest.approx(0.0)
    assert centered[1:7].mean().item() == pytest.approx(0.0, abs=1.0e-7)
    assert torch.allclose(centered[7:], torch.zeros_like(centered[7:]))

    base_logits = torch.zeros((1, 12), dtype=torch.float32)
    base_entry_mass = torch.softmax(base_logits, dim=-1)[0, 1:7].sum()
    centered_entry_mass = torch.softmax(
        base_logits + centered.unsqueeze(0), dim=-1
    )[0, 1:7].sum()
    assert centered_entry_mass.item() == pytest.approx(
        base_entry_mass.item(), abs=1.0e-7
    )


def test_real_codex_update_supports_traceable_management_action():
    torch.manual_seed(706)
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
        hidden_size=12, hidden_layers=6, seed=706,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    policy.install_five_rlhf_reference()
    raw = list(policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32)))
    owner = raw[0]
    target = raw[2]
    owner.action_details["five_action_index"] = 1
    owner = replace(
        owner,
        selected_idx=1,
        allowed_action_mask=tuple(True for _ in owner.allowed_action_mask),
    )
    target.action_details["five_action_index"] = 8
    target = replace(
        target,
        selected_idx=8,
        allowed_action_mask=tuple(True for _ in target.allowed_action_mask),
    )
    traces = [owner, raw[1], target, raw[3], raw[4]]
    lifecycle = SimpleNamespace(
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": owner.trace_id},
        position_ticket="managed", exit_trace_id=target.trace_id,
        exit_requested_action=8, exit_kind="EXPERT", exit_controller="five",
        realized_r=-0.4, mfe_r=0.1, mae_r=-0.6,
        commission=0.0, swap=0.0, close_reason="EXPERT", side=1,
        entry_count=1, entry_price=100.0, initial_sl_price=99.0,
        initial_tp_price=102.0, initial_risk=1.0, initial_atr=0.4,
        pending_fill_bars=0,
    )
    packet = build_codex_teaching_packet(
        lifecycle, traces, target_trace_id=target.trace_id,
    )
    assert packet["action_name"] == "CLOSE_WORST"
    assert packet["feature_schema"].endswith("action12_onehot_v1")
    receipt = validate_codex_teacher_receipt(
        {**receipt_for(packet), "teacher_reward": -0.5}, packet, "test-model",
    )
    scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)
    result = policy.optimize_five_rlhf_trace_packet(
        traces, scorer, source_checkpoint_sha256="source",
    )
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_action_surface_count"] == 12
    assert result["rlhf_management_trace_count_skipped"] == 1
    assert result["rlhf_parameter_hash_before"] != result["rlhf_parameter_hash_after"]
    assert result["rlhf_codex_context_support_target"] >= 0.5
    assert result["rlhf_target_action_probability_after"] < result[
        "rlhf_target_action_probability_before"
    ]


def test_codex_residual_is_applied_after_expected_value_surface():
    """A Codex residual must reach the final entry logits in value mode."""
    torch.manual_seed(707)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    heads = network.heads
    heads.seven_choice_expected_value_decision_enabled = True
    heads.seven_choice_action_decision_mode = "expected_value"
    heads.seven_choice_expected_value_score_mode = "value"
    heads.seven_choice_expected_value_threshold_r = 0.0
    heads.seven_choice_expected_value_logit_scale = 1.0
    heads.seven_choice_expected_value_action_residual_scale = 0.0
    heads.seven_choice_expected_value_hidden_residual_enabled = False
    heads.seven_choice_expected_value_fill_probability_floor = 0.1
    heads.seven_choice_expected_value_win_probability_floor = 0.0
    heads.seven_choice_expected_value_min_score_margin = 0.0
    observations = torch.randn(5, INPUT_COUNT)
    memories = network.initial_memories()
    with torch.no_grad():
        _, _, baseline = network.forward_five(
            observations,
            memories,
            deterministic=True,
            behavior_uniform_mix=np.zeros(5, dtype=np.float32),
            behavior_entry_probability_floor=0.0,
        )
        assert heads.capture_codex_context_prototype(
            baseline["policy_hidden"][0]
        ) is True
        heads.codex_action_residual_enabled = True
        heads.codex_action_residual[-1].weight.zero_()
        heads.codex_action_residual[-1].bias.zero_()
        heads.codex_action_residual[-1].bias[1] = 3.0
        _, _, intervened = network.forward_five(
            observations,
            memories,
            deterministic=True,
            behavior_uniform_mix=np.zeros(5, dtype=np.float32),
            behavior_entry_probability_floor=0.0,
        )
    assert intervened["codex_context_support"][0].item() == pytest.approx(1.0)
    # Action 1 is the long entry action for agent 0.  The +3 bias must survive
    # the expected-value boundary and be visible in the final logits.
    assert intervened["action_logits"][0, 1].item() - baseline[
        "action_logits"
    ][0, 1].item() == pytest.approx(3.0, abs=1e-5)


def test_codex_global_bias_reaches_unseen_context_when_prototype_support_is_zero():
    """A verified Codex update must not become a one-context-only no-op."""
    torch.manual_seed(708)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    memories = network.initial_memories()
    observation_with_prototype = torch.randn(5, INPUT_COUNT)
    unseen_observation = torch.randn(5, INPUT_COUNT)
    with torch.no_grad():
        _, _, probe = network.forward_five(
            observation_with_prototype,
            memories,
            deterministic=True,
            behavior_uniform_mix=np.zeros(5, dtype=np.float32),
            behavior_entry_probability_floor=0.0,
        )
        assert network.heads.capture_codex_context_prototype(
            probe["policy_hidden"][0]
        ) is True
        _, _, baseline = network.forward_five(
            unseen_observation,
            memories,
            deterministic=True,
            behavior_uniform_mix=np.zeros(5, dtype=np.float32),
            behavior_entry_probability_floor=0.0,
        )
        network.heads.codex_action_residual_enabled = True
        network.heads.codex_action_bias_enabled = True
        network.heads.codex_action_bias.zero_()
        network.heads.codex_action_bias[1] = 0.8
        _, _, intervened = network.forward_five(
            unseen_observation,
            memories,
            deterministic=True,
            behavior_uniform_mix=np.zeros(5, dtype=np.float32),
            behavior_entry_probability_floor=0.0,
        )
    assert intervened["codex_context_support"][0].item() == pytest.approx(0.0)
    assert intervened["action_logits"][0, 1].item() > baseline[
        "action_logits"
    ][0, 1].item()
    centered_entry_bias = (
        0.50 * float(torch.tanh(torch.tensor(0.8)).item()) * (5.0 / 6.0)
    )
    assert intervened["action_logits"][0, 1].item() - baseline[
        "action_logits"
    ][0, 1].item() == pytest.approx(
        centered_entry_bias, abs=1e-5
    )


def test_codex_context_reuses_near_regime_and_rejects_distant_state():
    """Context credit must reuse a near state without accepting a distant one."""
    heads = FactorizedActionHeads(hidden_size=8)
    heads.codex_context_similarity_threshold = 0.90
    heads.codex_context_support_floor = 0.50
    base = torch.zeros(8)
    base[0] = 1.0
    near = torch.zeros(8)
    near[0] = 0.96
    near[1] = float((1.0 - 0.96**2) ** 0.5)
    distant = torch.zeros(8)
    distant[0] = 0.94
    distant[1] = float((1.0 - 0.94**2) ** 0.5)

    assert heads.capture_codex_context_prototype(base) is True
    # Effective threshold is 0.90 + 0.50 * (1 - 0.90) = 0.95.
    assert heads.capture_codex_context_prototype(near) is False
    assert heads.codex_context_prototypes.shape[0] == 1
    assert heads.codex_context_support(near.unsqueeze(0)).item() == pytest.approx(0.60, abs=1e-5)
    assert heads.capture_codex_context_prototype(distant) is True
    assert heads.codex_context_prototypes.shape[0] == 2


def test_codex_action_surface_gate_round_trips_through_state_dict():
    """A learned Codex surface must remain active after candidate reload."""
    torch.manual_seed(709)
    source = FactorizedActionHeads(hidden_size=8)
    with torch.no_grad():
        source.codex_action_bias[1] = 0.8
    source.set_codex_action_surface_enabled(True)
    state = source.state_dict()
    assert bool(state["codex_action_surface_enabled"].item()) is True

    restored = FactorizedActionHeads(hidden_size=8)
    restored.load_state_dict(state)
    restored.require_current_architecture(8)
    assert restored.codex_action_residual_enabled is True
    assert restored.codex_action_bias_enabled is True

    hidden = torch.zeros((1, 8))
    logits = torch.zeros((1, 12))
    with torch.no_grad():
        intervened = restored.apply_codex_action_residual(logits, hidden)
    centered_entry_bias = (
        0.50 * float(torch.tanh(torch.tensor(0.8)).item()) * (5.0 / 6.0)
    )
    assert intervened[0, 1].item() == pytest.approx(
        centered_entry_bias, abs=1e-5
    )

    # Artifacts written before the persistent marker must remain loadable and
    # recover the gate from a non-zero learned Codex tensor.
    legacy_state = dict(state)
    legacy_state.pop("codex_action_surface_enabled")
    legacy = FactorizedActionHeads(hidden_size=8)
    load_result = legacy.load_state_dict(legacy_state, strict=True)
    legacy.require_current_architecture(8)
    assert load_result.missing_keys == []
    assert legacy.codex_action_residual_enabled is True
    assert legacy.codex_action_bias_enabled is True
