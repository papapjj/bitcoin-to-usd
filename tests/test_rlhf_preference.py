from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dataclasses import replace

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.five_shared_policy import (
    FivePPOTrainer,
    SharedFiveTradingPolicy,
)
from self_growing_trader.rlhf_preference import (
    RLHF_PREFERENCE_SCHEMA_VERSION,
    RLHFPreferencePair,
    load_rlhf_preference_jsonl,
    pairwise_preference_metrics,
    parse_rlhf_preference_row,
    rlhf_action_target,
    rlhf_inter_rater_agreement,
    rlhf_pairwise_loss_and_ascent,
    rlhf_preference_gate,
    rlhf_dataset_gate,
    validate_rlhf_dataset,
)
from self_growing_trader.rlhf_ppo import RLHFPPOConfig, rlhf_ppo_loss_from_logits
from self_growing_trader.rlhf_policy_loop import (
    RLHFPolicyRollout,
    RLHF_POLICY_FEATURE_SCHEMA,
    causal_policy_feature_vector,
    freeze_reference_policy,
    optimize_rlhf_rollout_batch,
)
from self_growing_trader.rlhf_feedback import append_rlhf_feedback_row
from self_growing_trader.rlhf_reward_model import (
    RLHFRewardModel,
    RLHFRewardPair,
    fit_rlhf_reward_model,
    load_rlhf_reward_model,
    parse_rlhf_reward_row,
    score_rlhf_reward_pairs,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT


def _pair(**overrides) -> RLHFPreferencePair:
    row = {
        "pair_id": "pair-1",
        "context_id": "ctx-1",
        "chosen_trace_id": "trace-chosen",
        "rejected_trace_id": "trace-rejected",
        "chosen_action": 1,
        "rejected_action": 2,
        "decision_time": "2026-09-22T08:00:00+00:00",
        "feedback_time": "2026-09-22T08:01:00+00:00",
        "pre_decision_state_hash": "state-hash-1",
        "risk_geometry_hash": "geometry-hash-1",
        "label_source": "human",
        "rater_id": "rater-1",
        "rater_version": "rubric-v1",
        "confidence": 0.9,
    }
    row.update(overrides)
    return RLHFPreferencePair(**row)


def test_pair_contract_is_causal_and_round_trippable() -> None:
    pair = _pair(reason_codes=("timing", "risk"))
    parsed = parse_rlhf_preference_row(pair.to_row())
    assert parsed == pair
    assert parsed.schema_version == RLHF_PREFERENCE_SCHEMA_VERSION
    assert parsed.candidate_only is True
    assert parsed.outcome_audit_only is True


def test_pair_rejects_same_action_or_future_feature_declaration() -> None:
    try:
        _pair(chosen_action=2)
    except ValueError as exc:
        assert "chosen_and_rejected_action_must_differ" in str(exc)
    else:
        raise AssertionError("same action pair must fail closed")
    assert parse_rlhf_preference_row(
        _pair().to_row() | {"feature_causal": False}
    ) is None


def test_pairwise_loss_pushes_chosen_up_and_rejected_down() -> None:
    pair = _pair()
    result = rlhf_pairwise_loss_and_ascent(
        np.zeros(20, dtype=float), pair, temperature=0.25
    )
    assert result.loss > 0.0
    assert result.probability_chosen == 0.5
    assert result.ascent[pair.chosen_action] > 0.0
    assert result.ascent[pair.rejected_action] < 0.0
    assert np.count_nonzero(result.ascent) == 2


def test_low_confidence_feedback_is_not_allowed_to_update() -> None:
    result = rlhf_pairwise_loss_and_ascent(
        np.zeros(7, dtype=float), _pair(confidence=0.2), min_confidence=0.7
    )
    assert result.effective_confidence == 0.0
    assert result.loss == 0.0
    assert np.all(result.ascent == 0.0)


def test_action_target_and_metrics_are_pairwise_not_lcb_claims() -> None:
    pairs = tuple(_pair(pair_id=f"pair-{index}") for index in range(3))
    target = rlhf_action_target(pairs[0])
    assert target.shape == (7,)
    assert np.isclose(target.sum(), 1.0)
    metrics = pairwise_preference_metrics(pairs, [2.0] * 3, [0.0] * 3)
    assert metrics["accuracy"] == 1.0
    assert metrics["evidence_class"] == "offline_preference_only"
    gate = rlhf_preference_gate(metrics, min_pairs=2, min_weighted_accuracy=0.9)
    assert gate["passed"] is True
    assert gate["promotion_allowed"] is False
    assert gate["requires_mt5_frozen_oos_lcb"] is True


def test_dataset_loader_rejects_invalid_lines_and_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "rlhf.jsonl"
    valid = _pair().to_row()
    invalid = valid | {"pair_id": "bad", "candidate_only": False}
    path.write_text(
        "\n".join(json.dumps(row) for row in (valid, invalid, {"broken": True})) + "\n",
        encoding="utf-8",
    )
    pairs, invalid_lines = load_rlhf_preference_jsonl(path)
    assert len(pairs) == 1
    assert invalid_lines == (2, 3)
    summary = validate_rlhf_dataset(pairs + pairs)
    assert summary["valid"] is False
    assert any("duplicate_pair_ids" in error for error in summary["errors"])


def test_dataset_gate_requires_raters_and_all_temporal_splits() -> None:
    summary = validate_rlhf_dataset((_pair(),))
    gate = rlhf_dataset_gate(summary, min_pairs=1, min_raters=1)
    assert gate["ready_for_training"] is False
    assert "missing_split:validation" in gate["reasons"]
    assert "missing_split:test" in gate["reasons"]


def test_same_context_can_be_rated_by_multiple_raters_and_agreement_is_visible() -> None:
    first = _pair(pair_id="rater-pair-1", rater_id="rater-1")
    same_preference = _pair(pair_id="rater-pair-2", rater_id="rater-2")
    summary = validate_rlhf_dataset((first, same_preference))
    assert summary["valid"] is True
    assert summary["inter_rater"]["rated_group_count"] == 1
    assert summary["inter_rater"]["agreement"] == 1.0
    disagreement = _pair(
        pair_id="rater-pair-3",
        rater_id="rater-3",
        chosen_trace_id="trace-rejected",
        rejected_trace_id="trace-chosen",
        chosen_action=2,
        rejected_action=1,
    )
    agreement = rlhf_inter_rater_agreement((first, disagreement))
    assert agreement["rated_group_count"] == 1
    assert agreement["agreement"] == 0.5


def test_causal_policy_feature_schema_contains_only_state_and_action() -> None:
    values = causal_policy_feature_vector(np.zeros(137, dtype=np.float32), 3)
    assert values.shape == (144,)
    assert values[137 + 3] == 1.0
    assert np.sum(values[137:]) == 1.0


def test_mt5_trace_bridge_requires_reference_logits_and_preserves_provenance() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=6,
        seed=704,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    reference_install = policy.install_five_rlhf_reference()
    assert reference_install["installed"] is True
    assert reference_install["reference_policy_frozen"] is True
    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))
    assert all(
        trace.action_details["five_reference_policy_frozen"] is True
        for trace in traces
    )
    assert all(
        trace.action_details["five_reference_policy_sha256"]
        == reference_install["reference_policy_sha256"]
        for trace in traces
    )
    features = {trace.trace_id: (0.1, 0.2) for trace in traces}
    states = {trace.trace_id: f"state-{index}" for index, trace in enumerate(traces)}
    geometries = {trace.trace_id: "geometry-v1" for trace in traces}
    rollouts = policy.build_rlhf_policy_rollouts(
        traces,
        feature_schema="test_features_v1",
        features_by_trace_id=features,
        state_hash_by_trace_id=states,
        geometry_hash_by_trace_id=geometries,
        reference_policy_sha256=str(reference_install["reference_policy_sha256"]),
    )
    assert len(rollouts) == 5
    assert rollouts[0].state_hash.startswith("state-")
    assert rollouts[0].geometry_hash == "geometry-v1"
    assert len(rollouts[0].old_logits) == len(rollouts[0].reference_logits)
    entry_traces = tuple(
        trace for trace in traces
        if 0 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) < 7
    )
    assert entry_traces
    auto_rollouts = policy.build_rlhf_policy_rollouts(
        entry_traces,
        action_surface_count=7,
    )
    assert auto_rollouts[0].state_hash
    assert auto_rollouts[0].geometry_hash
    assert len(auto_rollouts[0].features) == INPUT_COUNT + 7


def test_mt5_trace_bridge_rejects_tensor_reference_marker_without_ambiguous_boolean() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=6,
        seed=706,
    )
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    policy._learning_enabled = False
    reference_install = policy.install_five_rlhf_reference()
    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))
    malformed = replace(
        traces[0],
        action_details=dict(traces[0].action_details)
        | {"five_reference_policy_frozen": torch.tensor([True, True])},
    )

    try:
        policy.build_rlhf_policy_rollouts(
            (malformed,),
            action_surface_count=7,
            reference_policy_sha256=str(reference_install["reference_policy_sha256"]),
        )
    except ValueError as exc:
        assert "reference policy is not frozen" in str(exc)
    else:
        raise AssertionError("non-scalar frozen marker must fail closed")


def test_mt5_trace_packet_replays_causal_state_and_updates_candidate_policy() -> None:
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
    reference_install = policy.install_five_rlhf_reference()
    assert reference_install["installed"] is True
    traces = policy.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))

    weights = np.zeros(INPUT_COUNT + 7, dtype=np.float64)
    weights[INPUT_COUNT + 1] = 1.0
    weights[INPUT_COUNT + 2] = -0.5
    reward_model = RLHFRewardModel(
        feature_schema=RLHF_POLICY_FEATURE_SCHEMA,
        mean=(0.0,) * (INPUT_COUNT + 7),
        scale=(1.0,) * (INPUT_COUNT + 7),
        weights=tuple(weights),
        bias=0.0,
    )
    before = [parameter.detach().clone() for parameter in policy.five_shared_network.parameters()]
    reference_before = [
        parameter.detach().clone()
        for parameter in policy.five_rlhf_reference_network.parameters()
    ]
    result = policy.optimize_five_rlhf_trace_packet(
        traces,
        reward_model,
        source_checkpoint_sha256="checkpoint-before-rlhf",
    )
    assert result["rlhf_mt5_trace_packet"] is True
    assert result["rlhf_policy_optimizer_writer"] == "FiveAdam"
    assert result["rlhf_reward_model_sha256"] == reward_model.sha256
    assert result["rlhf_rollout_count"] == result["rlhf_entry_surface_trace_count"]
    assert result["rlhf_rollout_count"] >= 1
    assert result["rlhf_rollout_count"] + result["rlhf_management_trace_count_skipped"] == 5
    assert result["promotion_allowed"] is False
    assert any(
        not torch.equal(parameter, previous)
        for parameter, previous in zip(policy.five_shared_network.parameters(), before)
    )
    assert all(
        torch.equal(parameter, previous)
        for parameter, previous in zip(
            policy.five_rlhf_reference_network.parameters(), reference_before
        )
    )


def test_reference_policy_snapshot_is_frozen_and_hash_stable() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    reference, first_hash = freeze_reference_policy(network)
    _, second_hash = freeze_reference_policy(network)
    assert first_hash == second_hash
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert reference.training is False


def test_reward_model_fits_only_causal_rater_features() -> None:
    rows = tuple(
        RLHFRewardPair(
            pair=_pair(
                pair_id=f"reward-{index}",
                chosen_action=1,
                rejected_action=2,
            ),
            chosen_features=(2.0 + index, 1.0),
            rejected_features=(0.0, -1.0),
            feature_schema="entry_rater_features_v1",
        )
        for index in range(4)
    )
    model, fit = fit_rlhf_reward_model(rows, epochs=30, learning_rate=0.1)
    chosen, rejected = model.score_pair(rows[0])
    assert fit["model_sha256"] == model.sha256
    assert chosen > rejected
    metrics = score_rlhf_reward_pairs(model, rows)
    assert metrics["pair_count"] == 4
    assert metrics["evidence_class"] == "offline_reward_model_only"
    assert metrics["promotion_allowed"] is False


def test_frozen_reward_model_artifact_loader_checks_declared_hash() -> None:
    model = RLHFRewardModel(
        feature_schema=RLHF_POLICY_FEATURE_SCHEMA,
        mean=(0.0,) * (INPUT_COUNT + 7),
        scale=(1.0,) * (INPUT_COUNT + 7),
        weights=(0.0,) * (INPUT_COUNT + 7),
        bias=0.0,
    )
    artifact = Path("outputs/rlhf_test_reward_model_loader.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    try:
        artifact.write_text(json.dumps(model.to_row()), encoding="utf-8")
        loaded = load_rlhf_reward_model(artifact)
        assert loaded.sha256 == model.sha256
        assert loaded.feature_schema == RLHF_POLICY_FEATURE_SCHEMA
    finally:
        artifact.unlink(missing_ok=True)


def test_frozen_reward_model_loader_accepts_only_a_gated_training_report() -> None:
    model = RLHFRewardModel(
        feature_schema=RLHF_POLICY_FEATURE_SCHEMA,
        mean=(0.0,) * (INPUT_COUNT + 7),
        scale=(1.0,) * (INPUT_COUNT + 7),
        weights=(0.0,) * (INPUT_COUNT + 7),
        bias=0.0,
    )
    gated = Path("outputs/rlhf_test_reward_model_gated_report.json")
    rejected = Path("outputs/rlhf_test_reward_model_rejected_report.json")
    gated.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": model.to_row(),
        "test_gate": {"passed": True, "reasons": []},
    }
    try:
        gated.write_text(json.dumps(report), encoding="utf-8")
        loaded = load_rlhf_reward_model(gated)
        assert loaded.sha256 == model.sha256

        raw = Path("outputs/rlhf_test_reward_model_raw.json")
        raw.write_text(json.dumps(model.to_row()), encoding="utf-8")
        try:
            try:
                load_rlhf_reward_model(raw, require_test_gate=True)
            except ValueError as exc:
                assert "passed test_gate" in str(exc)
            else:
                raise AssertionError("MT5 loader must reject an ungated raw model row")
        finally:
            raw.unlink(missing_ok=True)

        rejected.write_text(
            json.dumps({
                "model": model.to_row(),
                "test_gate": {"passed": False, "reasons": ["log_loss_above_ceiling"]},
            }),
            encoding="utf-8",
        )
        try:
            load_rlhf_reward_model(rejected)
        except ValueError as exc:
            assert "test_gate must pass" in str(exc)
        else:
            raise AssertionError("failed reward-model gate must block MT5 loading")
    finally:
        gated.unlink(missing_ok=True)
        rejected.unlink(missing_ok=True)


def test_reward_pair_parser_fails_closed_on_nonboolean_safety_flags() -> None:
    pair = RLHFRewardPair(
        pair=_pair(pair_id="parser-reward"),
        chosen_features=(1.0, 0.0),
        rejected_features=(0.0, 1.0),
        feature_schema="entry_rater_features_v1",
    )
    row = pair.to_row()
    row["reward_feature_causal"] = "false"
    assert parse_rlhf_reward_row(row) is None


def test_human_feedback_writer_rejects_teacher_labels_before_writing() -> None:
    row = RLHFRewardPair(
        pair=_pair(pair_id="teacher-not-human", label_source="teacher"),
        chosen_features=(1.0, 0.0),
        rejected_features=(0.0, 1.0),
        feature_schema="entry_rater_features_v1",
    ).to_row()
    try:
        append_rlhf_feedback_row(
            "outputs\\rlhf\\should_not_be_written.jsonl",
            row,
            reward_pair=True,
            require_human_feedback=True,
        )
    except ValueError as exc:
        assert "human or expert" in str(exc)
    else:
        raise AssertionError("teacher label must not enter the human RLHF writer")


def test_rlhf_ppo_objective_uses_frozen_reference_kl_and_backpropagates() -> None:
    new_logits = torch.tensor(
        [[0.5, -0.2, 0.1], [-0.1, 0.4, 0.0]], dtype=torch.float32, requires_grad=True
    )
    old_logits = torch.zeros_like(new_logits)
    reference_logits = torch.zeros_like(new_logits)
    current_values = torch.tensor([0.2, -0.1], requires_grad=True)
    loss, stats = rlhf_ppo_loss_from_logits(
        new_logits=new_logits,
        old_logits=old_logits,
        reference_logits=reference_logits,
        actions=torch.tensor([0, 1]),
        advantages=torch.tensor([1.0, -1.0]),
        values=torch.zeros(2),
        returns=torch.tensor([1.0, -0.5]),
        new_values=current_values,
        allowed_action_mask=torch.tensor([
            [True, True, True],
            [True, True, False],
        ]),
        config=RLHFPPOConfig(kl_beta=0.1),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(new_logits.grad).all()
    assert torch.isfinite(current_values.grad).all()
    assert stats["rlhf_ppo_samples"] == 2.0
    assert stats["rlhf_ppo_reference_kl"] >= 0.0
    assert stats["promotion_allowed"] is False


def test_existing_five_sole_writer_can_apply_guarded_rlhf_ppo_update() -> None:
    torch.manual_seed(702)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    trainer = FivePPOTrainer(network)
    before = [parameter.detach().clone() for parameter in network.parameters()]
    source_parameter = next(parameter for parameter in network.parameters() if parameter.numel() >= 14)
    logits = source_parameter.reshape(-1)[:14].reshape(2, 7)
    result = trainer.optimize_rlhf_ppo_from_logits(
        new_logits=logits,
        old_logits=torch.zeros_like(logits),
        reference_logits=torch.zeros_like(logits),
        actions=torch.tensor([0, 1]),
        advantages=torch.tensor([1.0, -1.0]),
        values=torch.zeros(2),
        returns=torch.tensor([1.0, -1.0]),
        allowed_action_mask=torch.ones(2, 7, dtype=torch.bool),
        reward_model_sha256="reward-hash",
        reference_policy_sha256="reference-hash",
    )
    assert result["rlhf_ppo_update"] is True
    assert result["rlhf_policy_optimizer_writer"] == "FiveAdam"
    assert result["promotion_allowed"] is False
    assert any(
        not torch.equal(parameter, previous)
        for parameter, previous in zip(network.parameters(), before)
    )


def test_closed_loop_scores_rollout_then_updates_existing_five_writer() -> None:
    reward_rows = tuple(
        RLHFRewardPair(
            pair=_pair(pair_id=f"loop-reward-{index}"),
            chosen_features=(2.0 + index, 1.0),
            rejected_features=(0.0, -1.0),
            feature_schema="entry_rater_features_v1",
        )
        for index in range(4)
    )
    reward_model, _ = fit_rlhf_reward_model(
        reward_rows, epochs=30, learning_rate=0.1
    )
    zero_logits = (0.0,) * 7
    rollouts = tuple(
        RLHFPolicyRollout(
            rollout_id=f"rollout-{index}",
            pair_id=f"loop-reward-{index}",
            state_hash=f"state-{index}",
            geometry_hash="geometry-v1",
            feature_schema=reward_model.feature_schema,
            features=(2.0 + index, 1.0),
            action=index % 2,
            old_logits=zero_logits,
            reference_logits=zero_logits,
            value=0.0,
            allowed_action_mask=(True,) * 7,
        )
        for index in range(2)
    )
    torch.manual_seed(703)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    trainer = FivePPOTrainer(network)
    source_parameter = next(
        parameter for parameter in network.parameters() if parameter.numel() >= 14
    )
    result = optimize_rlhf_rollout_batch(
        trainer,
        reward_model,
        rollouts,
        new_logits=source_parameter.reshape(-1)[:14].reshape(2, 7),
        reference_policy_sha256="reference-hash",
        source_checkpoint_sha256="source-checkpoint-hash",
    )
    assert result["rlhf_rollout_count"] == 2
    assert result["rlhf_reward_model_sha256"] == reward_model.sha256
    assert result["rlhf_source_checkpoint_sha256"] == "source-checkpoint-hash"
    assert result["promotion_allowed"] is False


def test_policy_routes_rlhf_pair_into_the_existing_full_rl_queue() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=12,
        hidden_layers=6,
        seed=701,
    )
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_rl_batch_size = 2
    policy.direct_rnn_rl_rlhf_enabled = True
    policy.direct_rnn_rl_rlhf_loss_weight = 0.5
    policy.direct_rnn_rl_rlhf_min_confidence = 0.7
    policy._learning_enabled = True
    policy.ensure_compatible()
    trace = policy.act(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=float))
    trace = replace(trace, trace_id="rlhf-trace-chosen")
    registered = policy.register_entry_preference_trace(trace)
    assert registered["registered"] is True
    pair = _pair(
        pair_id="policy-pair-1",
        chosen_trace_id=trace.trace_id,
        rejected_trace_id="rlhf-trace-rejected",
        chosen_action=1,
        rejected_action=2,
    )
    result = policy.apply_rlhf_preference_credit_detailed(pair)
    assert result["accepted"] is True
    assert result["queued"] is True
    queued = policy._direct_rnn_rl_batches[-1]
    assert queued["rlhf_preference_only"] is True
    assert queued["credit_kind"] == "rlhf_pairwise_preference"
    stats = policy._apply_direct_rnn_full_rl_batch([queued])
    event = policy.growth_events[-1]
    assert stats == 1
    assert event["rlhf_preference_enabled"] is True
    assert event["rlhf_preference_item_count"] == 1
    assert event["rlhf_preference_eligible_count"] == 1
    assert event["rlhf_preference_gradient_l1"] > 0.0
