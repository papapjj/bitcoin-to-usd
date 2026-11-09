from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train_teacher_imitation import (
    CONFIRMED_ACTION_DATASET,
    _action_resume_contract,
    _current_action_imitation_report_passed,
    _effective_imitation_modes,
    _geometry_should_reuse_hidden_cache,
    _load_action_resume,
    _resolve_imitation_dataset,
    _reset_five_runtime_after_imitation,
    _verification_space,
)
from self_growing_trader.five_shared_policy import (
    ACTION_EMBED_SIZE,
    FivePPOTrainer,
    SharedFiveTradingPolicy,
)
from self_growing_trader.five_teacher_imitation import (
    TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH,
    TEACHER_IMITATION_MAX_SEQUENCE_LENGTH,
    _distance_aware_geometry_row_loss,
    _audit_role_operating_points,
    autonomous_entry_validation_gate,
    autonomous_entry_validation_metrics,
    _calibrate_role_zero_boundary_from_train,
    _fit_diagnostic,
    _imitation_mentor_step,
    _protected_logits_sha256,
    _role_logits,
    TeacherGeometryImitationConfig,
    TeacherImitationConfig,
    TeacherImitationPaused,
    extract_owner_hidden_sequences,
    geometry_invariant_sha256,
    geometry_imitation_metrics,
    geometry_trainable_sha256,
    ImitationMentorConfig,
    load_teacher_geometry_targets,
    load_teacher_imitation_data,
    network_invariant_sha256,
    recurrent_imitation_frozen_sha256,
    recurrent_imitation_trainable_sha256,
    reuse_verified_owner_hidden_cache,
    sha256_file,
    teacher_trainable_sha256,
    train_recurrent_geometry_heads,
    train_recurrent_entry_calibrator,
)


def test_imitation_sequence_length_is_ui_scalable_to_64() -> None:
    assert TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH == 64
    assert TEACHER_IMITATION_MAX_SEQUENCE_LENGTH == 64
    TeacherImitationConfig(sequence_length=64).validate()
    TeacherGeometryImitationConfig(sequence_length=64).validate()
    with pytest.raises(ValueError, match=r"\[2,64\]"):
        TeacherImitationConfig(sequence_length=65).validate()


def test_imitation_mentor_guides_penalty_role_balance_and_learning_rate() -> None:
    mentor = ImitationMentorConfig(min_epoch=2)
    mentor.validate()
    state = {
        "active_false_entry_penalty": 0.50,
        "role_scales": {"buy": 1.0, "sell": 1.0},
        "learning_rate_multiplier": 1.0,
        "stage": "warmup",
        "last_action": "warmup",
        "decision_count": 0,
    }
    record = {
        "epoch": 2,
        "validation_label_entry_rate": 0.10,
        "validation_predicted_entry_rate": 0.30,
        "validation_entry_precision": 0.20,
        "validation_entry_recall": 0.10,
        "validation_buy_agent_balanced_accuracy": 0.45,
        "validation_sell_agent_balanced_accuracy": 0.62,
        "overfit_status": "high",
        "underfit_status": "low",
    }
    updated, decision = _imitation_mentor_step(record, [], state, mentor)

    assert decision["action"]
    assert "raise_false_entry_penalty" in decision["action"]
    assert "rebalance_toward_buy" in decision["action"]
    assert "reduce_learning_rate_overfit" in decision["action"]
    assert updated["active_false_entry_penalty"] > state["active_false_entry_penalty"]
    assert updated["role_scales"]["buy"] > updated["role_scales"]["sell"]
    assert updated["learning_rate_multiplier"] < 1.0


def test_imitation_mentor_focuses_ineligible_sell_operating_point() -> None:
    mentor = ImitationMentorConfig(min_epoch=2)
    state = {
        "active_false_entry_penalty": 0.50,
        "role_scales": {"buy": 1.0, "sell": 1.0},
        "learning_rate_multiplier": 1.0,
        "stage": "warmup",
        "last_action": "warmup",
        "decision_count": 0,
    }
    record = {
        "epoch": 2,
        "validation_label_entry_rate": 0.10,
        "validation_predicted_entry_rate": 0.10,
        "validation_entry_precision": 0.35,
        "validation_entry_recall": 0.30,
        "validation_buy_agent_balanced_accuracy": 0.62,
        "validation_sell_agent_balanced_accuracy": 0.61,
        "validation_buy_agent_entry_precision": 0.44,
        "validation_sell_agent_entry_precision": 0.24,
        "validation_buy_agent_entry_recall": 0.31,
        "validation_sell_agent_entry_recall": 0.28,
        "validation_threshold_buy_eligible": 1.0,
        "validation_threshold_sell_eligible": 0.0,
        "overfit_status": "low",
        "underfit_status": "watch",
    }
    updated, decision = _imitation_mentor_step(record, [], state, mentor)

    assert "focus_sell_operating_point" in decision["action"]
    assert decision["stage"] == "focus_role_operating_point"
    assert updated["role_scales"]["sell"] > state["role_scales"]["sell"]
    assert updated["role_scales"]["sell"] > updated["role_scales"]["buy"]


def test_imitation_mentor_recovers_material_validation_regression() -> None:
    """A moderate entry-rate collapse must not be hidden by absolute floors."""
    mentor = ImitationMentorConfig(min_epoch=2)
    state = {
        "active_false_entry_penalty": 0.60,
        "role_scales": {"buy": 1.0, "sell": 1.0},
        "learning_rate_multiplier": 1.0,
        "teacher_confidence_weight": 0.50,
        "hard_negative_multiplier": 1.10,
        "active_ranking_loss_weight": 0.50,
        "stage": "balanced",
        "last_action": "hold_guidance",
        "decision_count": 10,
    }
    history = [{
        "epoch": 12,
        "validation_balanced_accuracy": 0.70591,
        "validation_entry_recall": 0.47655,
    }]
    record = {
        "epoch": 13,
        "validation_label_entry_rate": 0.12467,
        "validation_predicted_entry_rate": 0.11014,
        "validation_entry_precision": 0.52381,
        "validation_entry_recall": 0.46276,
        "validation_balanced_accuracy": 0.70142,
        "validation_buy_agent_balanced_accuracy": 0.70039,
        "validation_sell_agent_balanced_accuracy": 0.70243,
        "validation_buy_agent_entry_precision": 0.51969,
        "validation_sell_agent_entry_precision": 0.52786,
        "validation_buy_agent_entry_recall": 0.46089,
        "validation_sell_agent_entry_recall": 0.46458,
        "validation_threshold_buy_eligible": 1.0,
        "validation_threshold_sell_eligible": 1.0,
        "overfit_status": "low",
        "underfit_status": "low",
    }

    updated, decision = _imitation_mentor_step(record, history, state, mentor)

    assert decision["stage"] == "recover_entry_recall"
    assert "lower_false_entry_penalty_for_validation_recovery" in decision["action"]
    assert "relax_hard_negative_for_validation_recovery" in decision["action"]
    assert "teach_entry_recovery" in decision["action"]
    assert decision["signals"]["validation_ba_regression"] is True
    assert decision["signals"]["validation_recall_regression"] is True
    assert decision["signals"]["entry_mass_recovery"] is True
    assert updated["active_false_entry_penalty"] < state["active_false_entry_penalty"]
    assert updated["hard_negative_multiplier"] < state["hard_negative_multiplier"]
    assert updated["active_ranking_loss_weight"] > state["active_ranking_loss_weight"]


def test_imitation_mentor_prioritizes_severe_entry_collapse_over_low_precision() -> None:
    """Sparse predictions must trigger recovery instead of deeper suppression."""
    mentor = ImitationMentorConfig(min_epoch=2)
    state = {
        "active_false_entry_penalty": 0.50,
        "role_scales": {"buy": 1.0, "sell": 1.0},
        "learning_rate_multiplier": 1.0,
        "teacher_confidence_weight": 0.05,
        "hard_negative_multiplier": 1.0,
        "active_ranking_loss_weight": 0.50,
        "stage": "warmup",
        "last_action": "warmup",
        "decision_count": 1,
    }
    record = {
        "epoch": 2,
        "validation_label_entry_rate": 0.13693,
        "validation_predicted_entry_rate": 0.00692,
        "validation_entry_precision": 0.20,
        "validation_entry_recall": 0.01010,
        "validation_balanced_accuracy": 0.50185,
        "validation_buy_agent_balanced_accuracy": 0.50468,
        "validation_sell_agent_balanced_accuracy": 0.50000,
        "validation_buy_agent_entry_precision": 0.20,
        "validation_sell_agent_entry_precision": 0.0,
        "validation_buy_agent_entry_recall": 0.02222,
        "validation_sell_agent_entry_recall": 0.0,
        "validation_threshold_buy_eligible": 0.0,
        "validation_threshold_sell_eligible": 0.0,
        "overfit_status": "insufficient_data",
        "underfit_status": "insufficient_data",
    }

    updated, decision = _imitation_mentor_step(record, [], state, mentor)

    assert decision["signals"]["entry_mass_recovery"] is True
    assert decision["signals"]["low_precision"] is True
    assert decision["signals"]["validation_recovery"] is True
    assert decision["stage"] == "recover_entry_recall"
    assert "lower_false_entry_penalty_for_validation_recovery" in decision["action"]
    assert "relax_hard_negative_for_validation_recovery" in decision["action"]
    assert updated["active_false_entry_penalty"] < state["active_false_entry_penalty"]
    assert updated["hard_negative_multiplier"] == state["hard_negative_multiplier"]


def test_verify_only_space_check_does_not_reserve_training_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.train_teacher_imitation as imitation_module

    monkeypatch.setattr(
        imitation_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=123),
    )
    space = _verification_space(tmp_path / "new-candidate.pkl")
    assert space["free_bytes"] == 123
    assert space["effective_free_bytes"] == 123
    assert space["required_bytes"] == 0


def _dataset(tmp_path: Path) -> Path:
    labels = tmp_path / "labels.csv"
    labels.write_text("immutable teacher labels\n", encoding="utf-8")
    path = tmp_path / "teacher.npz"
    actions = []
    agents = []
    splits = []
    times = []
    finalized = []
    minute = 0
    for split, repeats in (("train", 4), ("validation", 2), ("test", 2)):
        for _ in range(repeats):
            for agent, action in ((0, 0), (0, 1), (1, 0), (1, 2)):
                actions.append(action)
                agents.append(agent)
                splits.append(split)
                times.append(f"2026-01-01T00:{minute:02d}:00")
                finalized.append(f"2026-01-02T00:{minute:02d}:00")
                minute += 1
    actions_array = np.asarray(actions, dtype=np.int64)
    np.savez_compressed(
        path,
        schema_version=np.asarray([7]),
        feature_count=np.asarray([137]),
        observations=np.zeros((len(actions), 137), dtype=np.float32),
        actions=actions_array,
        candidate_actions=actions_array,
        teacher_trade_accepted=actions_array != 0,
        teacher_net_r=np.where(actions_array != 0, 1.0, 0.0).astype(np.float32),
        agent_ids=np.asarray(agents, dtype=np.int64),
        quality_weights=np.ones(len(actions), dtype=np.float32),
        sample_times=np.asarray(times),
        label_finalized_times=np.asarray(finalized),
        splits=np.asarray(splits),
        geometry_target_valid=actions_array != 0,
        teacher_delay_indices=np.zeros(len(actions), dtype=np.int64),
        teacher_sl_indices=np.where(actions_array == 1, 1, np.where(actions_array == 2, 3, 0)),
        teacher_tp_indices=np.where(actions_array == 1, 2, np.where(actions_array == 2, 4, 0)),
        teacher_trailing_indices=np.zeros(len(actions), dtype=np.int64),
        teacher_sl_atr=np.where(
            actions_array == 1, 1.0, np.where(actions_array == 2, 2.0, 0.0)
        ).astype(np.float32),
        teacher_tp_rr=np.where(
            actions_array == 1, 0.75, np.where(actions_array == 2, 1.5, 0.0)
        ).astype(np.float32),
    )
    path.with_suffix(".npz.json").write_text(json.dumps({
        "historical_label_csv": str(labels),
        "historical_label_sha256": sha256_file(labels),
        "sample_count": len(actions),
    }), encoding="utf-8")
    return path


def test_dataset_preflight_checks_hash_authority_and_chronological_splits(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    data = load_teacher_imitation_data(path)
    assert data.observations.shape == (32, 137)
    Path(data.manifest["historical_label_csv"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV da thay doi"):
        load_teacher_imitation_data(path)


def test_geometry_only_builds_cache_while_combined_stage_reuses_it(tmp_path: Path) -> None:
    cache = tmp_path / "hidden.f16.mmap"
    assert _geometry_should_reuse_hidden_cache(
        cache, geometry_only=True, explicit_reuse=False
    ) is False
    assert _geometry_should_reuse_hidden_cache(
        cache, geometry_only=True, explicit_reuse=True
    ) is True
    assert _geometry_should_reuse_hidden_cache(
        cache, geometry_only=False, explicit_reuse=False
    ) is True
    assert _geometry_should_reuse_hidden_cache(
        None, geometry_only=False, explicit_reuse=True
    ) is False


def test_action_resume_contract_fails_closed_on_drift(tmp_path: Path) -> None:
    trainer_config = TeacherImitationConfig(epochs=3, sequence_length=4)
    contract = _action_resume_contract(
        source=(tmp_path / "source.pkl").resolve(), source_hash="source-hash",
        source_bytes=123, dataset=(tmp_path / "teacher.npz").resolve(),
        dataset_hash="dataset-hash", config_path=(tmp_path / "config.yaml").resolve(),
        config_hash="config-hash", output=(tmp_path / "candidate.pkl").resolve(),
        trainer_config=trainer_config,
    )
    resume = tmp_path / "candidate.pkl.resume.pkl"
    for status in ("paused", "checkpoint"):
        resume.write_bytes(pickle.dumps({
            "schema_version": 2,
            "status": status,
            "contract": contract,
            "system": object(),
            "training_state": {"epoch": 2},
        }, protocol=pickle.HIGHEST_PROTOCOL))
        assert _load_action_resume(resume, contract)["training_state"] == {"epoch": 2}
    drifted = {**contract, "dataset_sha256": "changed"}
    with pytest.raises(ValueError, match="khong khop"):
        _load_action_resume(resume, drifted)


def test_production_imitation_defaults_to_action_only_and_migrates_retired_ui_dataset() -> None:
    assert _effective_imitation_modes(action_only=False, geometry_only=False) == (True, False)
    assert _effective_imitation_modes(action_only=True, geometry_only=False) == (True, False)
    assert _effective_imitation_modes(action_only=False, geometry_only=True) == (False, True)
    with pytest.raises(ValueError, match="loai tru"):
        _effective_imitation_modes(action_only=True, geometry_only=True)

    retired = (
        Path("outputs/bottom_top_teacher_recurrent_v5_max_sl075_20260827")
        / "teacher_max_sl075_recurrent.npz"
    )
    effective, migrated = _resolve_imitation_dataset(retired, action_only=True)
    assert migrated is True
    assert effective == CONFIRMED_ACTION_DATASET
    schema8 = (
        Path("outputs/bottom_top_teacher_confirmed_action_v8_20260830")
        / "teacher_confirmed_action_v8.npz"
    )
    effective, migrated = _resolve_imitation_dataset(schema8, action_only=True)
    assert migrated is True
    assert effective == CONFIRMED_ACTION_DATASET
    unchanged, migrated = _resolve_imitation_dataset(retired, action_only=False)
    assert migrated is False
    assert unchanged == retired.resolve()


def test_geometry_only_requires_current_continuous_self_fed_action_proof() -> None:
    report = {
        "schema_version": 12,
        "mt5_integration": {
            "entry_scope": "all_entry_mass_v2",
            "representation_scope": (
                "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
            ),
            "requires_runtime_teacher_retention": True,
            "requires_autonomous_continuous_validation": True,
        },
        "verification": {"candidate_reload_passed": True},
        "training": {
            "validation_gate": {"enforced": True, "passed": True},
            "autonomous_validation_gate": {"enforced": True, "passed": True},
        },
    }

    assert _current_action_imitation_report_passed(report) is True
    legacy = copy.deepcopy(report)
    legacy["schema_version"] = 6
    assert _current_action_imitation_report_passed(legacy) is False
    missing_closed_loop = copy.deepcopy(report)
    missing_closed_loop["training"].pop("autonomous_validation_gate")
    assert _current_action_imitation_report_passed(missing_closed_loop) is False


def test_sequence_extraction_carries_state_inside_public_unroll(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    hidden, audit = extract_owner_hidden_sequences(
        network, data, batch_size=3, sequence_length=4
    )
    assert hidden.shape == (32, 16)
    assert audit["sequence_length"] == 4
    assert audit["unique_time_steps"] == 32
    assert audit["sequence_windows"] == 8
    assert not torch.allclose(hidden[0], hidden[1])


def test_sequence_extraction_supports_float16_ram_storage(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    hidden, audit = extract_owner_hidden_sequences(
        network, data, batch_size=3, sequence_length=4, output_dtype=torch.float16
    )
    assert hidden.dtype == torch.float16
    assert hidden.shape == (32, 16)
    assert torch.isfinite(hidden).all()
    assert audit["hidden_storage"] == "ram_float16"


def test_autonomous_validation_never_feeds_teacher_actions_back_into_memory(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    original = autonomous_entry_validation_metrics(network, data)
    changed_actions = np.where(
        data.agent_ids == 0,
        np.where(data.actions == 0, 1, 0),
        np.where(data.actions == 0, 2, 0),
    ).astype(np.int64)
    relabelled = autonomous_entry_validation_metrics(
        network, replace(data, actions=changed_actions)
    )

    assert original["action_sequence_sha256"] == relabelled["action_sequence_sha256"]
    assert original["teacher_action_feedback"] is False
    assert original["state_reset_count"] == 1
    assert sum(original["action_counts_all_agents"].values()) == 5 * original["time_steps"]


def test_autonomous_validation_carries_all_five_actions_beyond_step_64(tmp_path: Path) -> None:
    base = load_teacher_imitation_data(_dataset(tmp_path))
    count = 65
    owners = np.arange(count, dtype=np.int64) % 2
    actions = np.where(owners == 0, 1, 2).astype(np.int64)
    observations = np.repeat(base.observations[:1], count, axis=0)
    data = replace(
        base,
        observations=observations,
        actions=actions,
        candidate_actions=actions.copy(),
        teacher_trade_accepted=np.ones(count, dtype=np.bool_),
        teacher_net_r=np.ones(count, dtype=np.float32),
        agent_ids=owners,
        quality_weights=np.ones(count, dtype=np.float32),
        sample_times=np.asarray([f"2026-01-02T{i // 60:02d}:{i % 60:02d}:00" for i in range(count)]),
        label_finalized_times=np.asarray([f"2026-01-03T{i // 60:02d}:{i % 60:02d}:00" for i in range(count)]),
        splits=np.asarray(["validation"] * count),
    )

    class FakeNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.previous_seen: list[list[int]] = []

        def initial_memories(self, *, device):
            return [SimpleNamespace(previous_action=torch.zeros((), dtype=torch.long, device=device)) for _ in range(5)]

        def forward_five(self, observations, memories, **_kwargs):
            del observations
            self.previous_seen.append([int(item.previous_action.item()) for item in memories])
            emitted = [1, 2, 0, 0, 0]
            decisions = []
            next_memories = []
            for action in emitted:
                probabilities = np.zeros(12, dtype=np.float32)
                probabilities[action] = 1.0
                decisions.append(SimpleNamespace(
                    action_index=action,
                    policy_action_probabilities=probabilities,
                ))
                next_memories.append(SimpleNamespace(
                    previous_action=torch.tensor(action, dtype=torch.long, device=self.marker.device)
                ))
            return decisions, next_memories, {}

    network = FakeNetwork()
    result = autonomous_entry_validation_metrics(network, data)

    assert len(network.previous_seen) == 65
    assert network.previous_seen[0] == [0, 0, 0, 0, 0]
    assert network.previous_seen[64] == [1, 2, 0, 0, 0]
    assert result["state_reset_count"] == 1
    assert result["missing_owner_steps"]["buy"] > 0
    assert result["missing_owner_steps"]["sell"] > 0
    assert result["buy_agent_tp"] + result["buy_agent_fn"] == 33.0
    assert result["sell_agent_tp"] + result["sell_agent_fn"] == 32.0


def test_autonomous_validation_gate_rejects_collapsed_self_fed_entry_policy() -> None:
    metrics = {
        "buy_agent_tp": 20.0, "buy_agent_tn": 70.0,
        "buy_agent_fp": 5.0, "buy_agent_fn": 5.0,
        "sell_agent_tp": 20.0, "sell_agent_tn": 70.0,
        "sell_agent_fp": 5.0, "sell_agent_fn": 5.0,
        "balanced_accuracy": 0.85,
        "buy_agent_balanced_accuracy": 0.85,
        "sell_agent_balanced_accuracy": 0.85,
        "buy_agent_entry_recall": 0.80,
        "sell_agent_entry_recall": 0.80,
        "mcc": 0.65,
        "entry_precision": 0.80,
        "label_entry_rate": 0.25,
        "predicted_entry_rate": 0.25,
    }
    accepted = autonomous_entry_validation_gate(metrics)
    rejected = autonomous_entry_validation_gate({
        **metrics,
        "balanced_accuracy": 0.50,
        "mcc": 0.0,
        "predicted_entry_rate": 0.90,
    })

    assert accepted["enforced"] is True
    assert accepted["passed"] is True
    assert rejected["passed"] is False
    assert {"balanced_accuracy", "mcc", "predicted_entry_rate"}.issubset(
        rejected["failed_checks"]
    )


def test_geometry_training_changes_only_teacher_geometry_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    targets = load_teacher_geometry_targets(data)
    assert int(targets.valid.sum()) == 16
    network = SharedFiveTradingPolicy(hidden_size=16)
    invariant_before = geometry_invariant_sha256(network)
    trainable_before = geometry_trainable_sha256(network)
    action_before = teacher_trainable_sha256(network)

    def fake_hidden(_network, _data, *, batch_size, sequence_length, cache_path=None):
        hidden = torch.zeros((len(_data.actions), _network.hidden_size), dtype=torch.float32)
        actions = torch.as_tensor(_data.actions)
        hidden[:, 0] = torch.where(actions == 1, 2.0, torch.where(actions == 2, -2.0, 0.0))
        hidden[:, 1] = (actions != 0).float()
        return hidden, {"sequence_length": sequence_length, "sequence_windows": 2,
                        "unique_time_steps": len(hidden), "state_reset_count": 2}

    monkeypatch.setattr(
        "self_growing_trader.five_teacher_imitation.extract_owner_hidden_sequences", fake_hidden
    )
    result = train_recurrent_geometry_heads(
        network, data,
        TeacherGeometryImitationConfig(
            epochs=80, batch_size=8, learning_rate=0.05, patience=12,
            seed=3, sequence_length=4,
        ),
    )

    assert geometry_invariant_sha256(network) == invariant_before
    assert geometry_trainable_sha256(network) != trainable_before
    assert teacher_trainable_sha256(network) == action_before
    assert result["after"]["test"]["sl_accuracy"] >= 0.99
    assert result["after"]["test"]["tp_accuracy"] >= 0.99
    assert result["after"]["test"]["sl_log_ratio_mae"] < 0.01
    assert result["after"]["test"]["tp_log_ratio_mae"] < 0.01
    assert result["objective"]["sample_weighting"] == "uniform_accepted_geometry_rows"
    assert result["logit_scale_calibration"]["sl"]["selected_scale"] >= 0.5
    assert result["logit_scale_calibration"]["tp"]["selected_scale"] >= 0.5
    assert all(name.startswith("heads.parameter_heads.") for name in result["changed_parameters"])
    assert network.heads.teacher_geometry_imitation_trained is True


def test_geometry_metrics_distinguish_adjacent_tp_bin_from_far_miss(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    targets = load_teacher_geometry_targets(data)
    network = SharedFiveTradingPolicy(hidden_size=16)
    hidden = torch.zeros((len(data.actions), 16), dtype=torch.float32)
    test_mask = torch.as_tensor(data.splits == "test")
    tp_head = network.heads.parameter_heads["tp"]
    with torch.no_grad():
        tp_head.weight.zero_()
        tp_head.bias.fill_(-20.0)
        tp_head.bias[3] = 20.0  # 1R: one bin from both 0.75R and 1.5R targets.
    adjacent = geometry_imitation_metrics(hidden, targets, network, test_mask)
    with torch.no_grad():
        tp_head.bias.fill_(-20.0)
        tp_head.bias[-1] = 20.0  # 5R: exact is still false, distance is far worse.
    far = geometry_imitation_metrics(hidden, targets, network, test_mask)
    assert adjacent["tp_accuracy"] == far["tp_accuracy"] == 0.0
    assert adjacent["tp_within_one_bin_accuracy"] == 1.0
    assert far["tp_within_one_bin_accuracy"] == 0.0
    assert adjacent["tp_log_ratio_mae"] < far["tp_log_ratio_mae"]


def test_geometry_loss_penalizes_probability_on_far_rr_bins() -> None:
    adjacent_logits = torch.full((1, 10), -8.0)
    far_logits = torch.full((1, 10), -8.0)
    adjacent_logits[0, 4] = 8.0  # 1.5R, adjacent to 1R.
    far_logits[0, 9] = 8.0  # 5R, far from 1R.
    target_index = torch.tensor([3], dtype=torch.long)
    target_rr = torch.tensor([1.0], dtype=torch.float32)
    bins = (0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    adjacent = _distance_aware_geometry_row_loss(
        adjacent_logits, target_index, target_rr, bins
    )
    far = _distance_aware_geometry_row_loss(far_logits, target_index, target_rr, bins)
    assert float(adjacent.item()) < float(far.item())


def test_geometry_schema2_uses_ordinal_median_for_deterministic_sl_tp() -> None:
    network = SharedFiveTradingPolicy(hidden_size=16)
    tp_head = network.heads.parameter_heads["tp"]
    with torch.no_grad():
        tp_head.weight.zero_()
        tp_head.bias.fill_(-30.0)
        tp_head.bias[2] = float(np.log(0.40))
        tp_head.bias[4] = float(np.log(0.35))
        tp_head.bias[5] = float(np.log(0.25))
    observations = torch.zeros((5, 137), dtype=torch.float32)
    legacy, _, _ = network.forward_five(
        observations, network.initial_memories(), deterministic=True
    )
    assert all(item.tp_index == 2 for item in legacy)

    network.heads.teacher_geometry_imitation_trained = True
    network.heads.teacher_geometry_imitation_schema_version = 2
    ordinal, _, _ = network.forward_five(
        observations, network.initial_memories(), deterministic=True
    )
    assert all(item.tp_index == 4 for item in ordinal)


def test_sequence_extraction_uses_bounded_float16_memmap(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    cache = tmp_path / "owner_hidden.f16.mmap"
    hidden, audit = extract_owner_hidden_sequences(
        network, data, batch_size=3, sequence_length=4, cache_path=cache
    )
    assert isinstance(hidden, np.memmap)
    assert hidden.dtype == np.float16
    assert hidden.shape == (32, 16)
    assert np.isfinite(hidden).all()
    assert cache.stat().st_size == 32 * 16 * 2
    assert audit["hidden_storage"] == "disk_memmap_float16"
    assert audit["hidden_cache_bytes"] == cache.stat().st_size


def test_completed_hidden_cache_is_reused_only_after_semantic_replay_probe(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    cache = tmp_path / "owner_hidden.f16.mmap"
    original, _ = extract_owner_hidden_sequences(
        network, data, batch_size=3, sequence_length=4, cache_path=cache
    )

    reused, audit = reuse_verified_owner_hidden_cache(
        network, data, cache_path=cache, sequence_length=4
    )

    np.testing.assert_allclose(np.asarray(reused), np.asarray(original), rtol=0.0, atol=0.0)
    assert audit["hidden_storage"] == "disk_memmap_float16_reused_verified"
    assert audit["reuse_semantic_probe_windows"] >= 3
    assert audit["reuse_semantic_probe_max_abs_drift"] < 2e-3

    reused[0, 0] = np.float16(float(reused[0, 0]) + 0.25)
    reused.flush()
    with pytest.raises(RuntimeError, match="khong khop network/dataset"):
        reuse_verified_owner_hidden_cache(
            network, data, cache_path=cache, sequence_length=4
        )


def test_training_backpropagates_through_full_recurrent_representation(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    network = SharedFiveTradingPolicy(hidden_size=16)
    with torch.no_grad():
        network.heads.action_query.weight.zero_()
        network.heads.action_query.bias.zero_()
        network.heads.action_query.weight[0, 0] = 1.0
        network.heads.action_query.weight[1, 1] = 1.0
        network.heads.action_embedding.weight[:7].zero_()
    frozen_before = recurrent_imitation_frozen_sha256(network)
    trainable_before = recurrent_imitation_trainable_sha256(network)
    probe = torch.randn(12, network.hidden_size)
    owner_ids = torch.tensor([0, 1] * 6)
    owner_before = network.heads.primary_action_logits(probe, owner_ids).detach().clone()
    progress_rows: list[dict[str, float | int | str]] = []
    result = train_recurrent_entry_calibrator(
        network,
        data,
        TeacherImitationConfig(
            epochs=80, batch_size=16, feature_batch_size=4,
            learning_rate=0.03, recurrent_learning_rate=0.003, patience=12,
            seed=7, sequence_length=4, false_entry_penalty=0.0,
        ),
        progress=progress_rows.append,
    )
    changed = result["changed_parameters"]
    assert any(name.startswith("heads.entry_role_calibrator.") for name in changed)
    assert recurrent_imitation_trainable_sha256(network) != trainable_before
    assert recurrent_imitation_frozen_sha256(network) == frozen_before
    owner_after = network.heads.primary_action_logits(probe, owner_ids)
    assert not torch.equal(owner_after, owner_before)
    assert network.heads.entry_role_calibrator.teacher_entry_scope == "all_entry_mass_v2"
    assert network.heads.entry_role_calibrator.teacher_imitation_trained is True
    assert (
        network.heads.entry_role_calibrator.teacher_imitation_projection
        == "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
    )
    assert "hidden" not in network.heads.entry_role_calibrator._modules
    assert all(
        torch.count_nonzero(branch.query_adapter.output.weight).item() > 0
        for branch in network.heads.entry_role_calibrator.role_calibrators
    )
    assert result["after"]["test"]["balanced_accuracy"] >= 0.75
    assert result["after"]["test"]["buy_agent_balanced_accuracy"] >= 0.75
    assert result["after"]["test"]["sell_agent_balanced_accuracy"] >= 0.75
    assert all(
        value > 0.0
        for value in result["projection_contract"]["gradient_norm_max_by_group"].values()
    )
    assert any(
        "post_lstm_transformer.temporal_" in name
        for name in result["projection_contract"]["gradient_nonzero_parameters"]
    )
    assert result["validation_gate"]["enforced"] is False
    assert np.isfinite(result["after"]["test"]["loss"])
    progress_events = {str(row.get("event", "action_epoch")) for row in progress_rows}
    assert {
        "action_replay_progress", "action_batch_progress", "action_epoch"
    }.issubset(progress_events)
    epoch_rows = [row for row in progress_rows if row.get("event", "action_epoch") == "action_epoch"]
    assert epoch_rows
    latest = epoch_rows[-1]
    assert 0.0 <= float(latest["train_balanced_accuracy_online"]) <= 1.0
    assert 0.0 <= float(latest["validation_entry_recall"]) <= 1.0
    assert np.isfinite(float(latest["imitation_reward_penalty_ratio"]))
    assert latest["underfit_status"] in {"insufficient_data", "high", "watch", "low"}
    assert latest["overfit_status"] in {"insufficient_data", "high", "watch", "low"}


def test_fit_diagnostic_distinguishes_underfit_and_overfit() -> None:
    underfit = _fit_diagnostic(
        epoch=3,
        train_metrics={"balanced_accuracy": 0.55, "mcc": 0.05},
        validation={"balanced_accuracy": 0.54, "loss": 0.70},
        train_loss=0.90,
        history=[],
    )
    assert underfit["underfit_status"] == "high"
    assert underfit["overfit_status"] == "low"

    overfit = _fit_diagnostic(
        epoch=5,
        train_metrics={"balanced_accuracy": 0.82, "mcc": 0.55},
        validation={"balanced_accuracy": 0.65, "loss": 0.60},
        train_loss=0.40,
        history=[{
            "train_weighted_loss": 0.45,
            "validation_balanced_accuracy": 0.70,
            "validation_loss": 0.55,
        }],
    )
    assert overfit["underfit_status"] == "low"
    assert overfit["overfit_status"] == "high"


def test_action_imitation_pause_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    torch.manual_seed(29)
    base = SharedFiveTradingPolicy(hidden_size=16)
    with torch.no_grad():
        base.heads.action_query.weight.zero_()
        base.heads.action_query.bias.zero_()
        base.heads.action_query.weight[0, 0] = 1.0
        base.heads.action_query.weight[1, 1] = 1.0
        base.heads.action_embedding.weight[:7].zero_()
    uninterrupted = copy.deepcopy(base)
    paused = copy.deepcopy(base)
    config = TeacherImitationConfig(
        epochs=80, batch_size=16, feature_batch_size=4,
        learning_rate=0.03, recurrent_learning_rate=0.003, patience=12,
        seed=7, sequence_length=4, false_entry_penalty=0.0,
    )

    expected = train_recurrent_entry_calibrator(uninterrupted, data, config)
    with pytest.raises(TeacherImitationPaused) as captured:
        train_recurrent_entry_calibrator(
            paused, data, config, stop_requested=lambda: True,
        )
    paused, resume_state = pickle.loads(pickle.dumps(
        (paused, captured.value.resume_state), protocol=pickle.HIGHEST_PROTOCOL
    ))
    actual = train_recurrent_entry_calibrator(
        paused, data, config, resume_state=resume_state,
    )

    for name, expected_tensor in uninterrupted.state_dict().items():
        torch.testing.assert_close(
            paused.state_dict()[name], expected_tensor, rtol=0.0, atol=0.0
        )
    assert actual["best_epoch"] == expected["best_epoch"]
    assert actual["epochs_ran"] == expected["epochs_ran"]
    assert actual["after"] == expected["after"]


def test_action_imitation_emits_restartable_checkpoint_after_each_epoch(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    torch.manual_seed(31)
    network = SharedFiveTradingPolicy(hidden_size=16)
    config = TeacherImitationConfig(
        epochs=2, batch_size=16, feature_batch_size=4,
        learning_rate=0.03, recurrent_learning_rate=0.003, patience=12,
        seed=9, sequence_length=4, false_entry_penalty=0.0,
    )
    checkpoints: list[dict[str, object]] = []
    result = train_recurrent_entry_calibrator(
        network,
        data,
        config,
        checkpoint=lambda state: checkpoints.append(
            pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        ),
    )

    assert len(checkpoints) == result["epochs_ran"]
    assert checkpoints[-1]["epoch"] == result["epochs_ran"] + 1
    assert checkpoints[-1]["permutation"] == []
    assert checkpoints[-1]["next_batch_start"] == 0
    assert len(checkpoints[-1]["history"]) == result["epochs_ran"]


def test_action_imitation_can_emit_mid_epoch_checkpoint(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    torch.manual_seed(32)
    network = SharedFiveTradingPolicy(hidden_size=16)
    config = TeacherImitationConfig(
        epochs=1, batch_size=2, feature_batch_size=4,
        learning_rate=0.03, recurrent_learning_rate=0.003, patience=12,
        seed=10, sequence_length=4, false_entry_penalty=0.0,
    )
    checkpoints: list[dict[str, object]] = []
    train_recurrent_entry_calibrator(
        network,
        data,
        config,
        checkpoint=lambda state: checkpoints.append(
            pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        ),
        checkpoint_every_windows=2,
    )

    mid_epoch = [item for item in checkpoints if item["checkpoint_kind"] == "batch"]
    assert mid_epoch
    assert mid_epoch[0]["epoch"] == 1
    assert mid_epoch[0]["next_batch_start"] == 2
    assert mid_epoch[0]["permutation"]


def test_action_imitation_resumes_from_completed_epoch_checkpoint(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    torch.manual_seed(33)
    network = SharedFiveTradingPolicy(hidden_size=16)
    config = TeacherImitationConfig(
        epochs=2, batch_size=16, feature_batch_size=4,
        learning_rate=0.03, recurrent_learning_rate=0.003, patience=12,
        seed=13, sequence_length=4, false_entry_penalty=0.0,
    )

    def stop_after_first_epoch(state: dict[str, object]) -> None:
        if int(state["epoch"]) == 2:
            raise TeacherImitationPaused(state)

    with pytest.raises(TeacherImitationPaused) as captured:
        train_recurrent_entry_calibrator(
            network, data, config, checkpoint=stop_after_first_epoch,
        )
    resume_state = pickle.loads(pickle.dumps(
        captured.value.resume_state, protocol=pickle.HIGHEST_PROTOCOL
    ))
    resumed = train_recurrent_entry_calibrator(
        network, data, config, resume_state=resume_state,
    )

    assert resume_state["epoch"] == 2
    assert resume_state["permutation"] == []
    assert resume_state["next_batch_start"] == 0
    assert resumed["epochs_ran"] == 2


def test_action_imitation_persists_mentor_guidance_in_history_and_resume(
    tmp_path: Path,
) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    torch.manual_seed(37)
    network = SharedFiveTradingPolicy(hidden_size=16)
    config = TeacherImitationConfig(
        epochs=2, batch_size=16, feature_batch_size=4,
        learning_rate=0.03, recurrent_learning_rate=0.003, patience=12,
        seed=11, sequence_length=4, false_entry_penalty=0.0,
    )
    mentor = ImitationMentorConfig(min_epoch=2)
    checkpoints: list[dict[str, object]] = []
    result = train_recurrent_entry_calibrator(
        network,
        data,
        config,
        mentor_config=mentor,
        checkpoint=lambda state: checkpoints.append(
            pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        ),
    )

    assert result["mentor"]["enabled"] is True
    assert len(result["history"]) == result["epochs_ran"]
    assert result["history"][0]["mentor_action"] == "warmup"
    assert "mentor_action" in result["history"][-1]
    assert {
        "validation_buy_agent_balanced_accuracy",
        "validation_sell_agent_balanced_accuracy",
        "validation_buy_agent_entry_precision",
        "validation_sell_agent_entry_precision",
        "validation_buy_agent_entry_recall",
        "validation_sell_agent_entry_recall",
    }.issubset(result["history"][-1])
    assert checkpoints[-1]["mentor_config"] == mentor.__dict__
    assert checkpoints[-1]["mentor_state"]["decision_count"] >= 1


def test_role_specific_calibrators_keep_market_only_until_retrained() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    hidden = torch.randn(2, 8)
    agent_ids = torch.tensor([0, 1])
    calibrator = network.heads.entry_role_calibrator
    with torch.no_grad():
        calibrator.role_calibrators[0].output.bias[:] = torch.tensor([0.0, 1.5])
        calibrator.role_calibrators[1].output.bias[:] = torch.tensor([0.0, 0.75])
    raw_query = network.heads.action_query(hidden)
    raw = torch.matmul(
        raw_query, network.heads.action_embedding.weight.t()
    ) / np.sqrt(float(network.heads.action_embedding.embedding_dim))
    legacy = network.heads.primary_action_logits(hidden, agent_ids)
    torch.testing.assert_close(legacy[:, [3, 4, 5, 6]], raw[:, [3, 4, 5, 6]])

    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    expanded = network.heads.primary_action_logits(hidden, agent_ids)
    torch.testing.assert_close(expanded[0, torch.tensor([1, 3, 5])] - raw[0, torch.tensor([1, 3, 5])], torch.full((3,), 1.5))
    torch.testing.assert_close(expanded[1, torch.tensor([2, 4, 6])] - raw[1, torch.tensor([2, 4, 6])], torch.full((3,), 0.75))


def test_buy_calibrator_update_cannot_change_sell_or_non_owner_adjustment() -> None:
    torch.manual_seed(31)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    projected = torch.randn(5, ACTION_EMBED_SIZE)
    agents = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    before = calibrator.forward_projected(projected, agents).detach().clone()

    with torch.no_grad():
        calibrator.role_calibrators[0].output.weight.add_(0.5)
        calibrator.role_calibrators[0].output.bias.add_(torch.tensor([-0.3, 0.7]))
    after = calibrator.forward_projected(projected, agents).detach()

    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1:], before[1:], rtol=0.0, atol=0.0)
    torch.testing.assert_close(after[2:], torch.zeros_like(after[2:]), rtol=0.0, atol=0.0)


def test_shared_action_query_training_representation_matches_runtime_logits() -> None:
    torch.manual_seed(13)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    with torch.no_grad():
        for branch in calibrator.role_calibrators:
            branch.output.weight.normal_(0.0, 0.1)
            branch.output.bias.normal_(0.0, 0.1)
    hidden = torch.randn(17, network.hidden_size)
    agents = torch.tensor(([0, 1] * 9)[:17], dtype=torch.long)
    projected = network.heads.action_query(hidden)
    torch.testing.assert_close(
        _role_logits(projected, agents, network),
        _role_logits(hidden, agents, network),
        rtol=1e-6,
        atol=1e-6,
    )


def test_legacy_shared_calibrator_migrates_to_two_exact_role_branches() -> None:
    from self_growing_trader.five_shared_policy import EntryQueryAdapter

    torch.manual_seed(17)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator._modules.pop("role_calibrators")
    calibrator.query_adapter = EntryQueryAdapter(ACTION_EMBED_SIZE)
    calibrator.output = torch.nn.Linear(ACTION_EMBED_SIZE, 2)
    calibrator.register_parameter("role_entry_bias", torch.nn.Parameter(torch.tensor([0.2, -0.1])))
    with torch.no_grad():
        calibrator.output.weight.normal_(0.0, 0.1)
        calibrator.output.bias.normal_(0.0, 0.1)
    hidden = torch.randn(12, network.hidden_size)
    agents = torch.tensor([0, 1] * 6)
    projected = network.heads.action_query(hidden)
    expected_adjustment = calibrator.output(calibrator.query_adapter(projected, agents))
    expected_adjustment[:, 1] += torch.where(
        agents == 0, torch.tensor(0.2), torch.tensor(-0.1)
    )
    calibrator.hidden = torch.nn.Linear(network.hidden_size, ACTION_EMBED_SIZE)

    assert network.require_current_architecture() is True
    assert "hidden" not in calibrator._modules
    assert "query_adapter" not in calibrator._modules
    assert "output" not in calibrator._modules
    assert "role_entry_bias" not in calibrator._parameters
    assert len(calibrator.role_calibrators) == 2
    torch.testing.assert_close(
        calibrator.forward_projected(projected, agents),
        expected_adjustment, rtol=0.0, atol=1e-7
    )


def test_full_recurrent_imitation_learns_teacher_mapping(tmp_path: Path) -> None:
    data = load_teacher_imitation_data(_dataset(tmp_path))
    network = SharedFiveTradingPolicy(hidden_size=16)
    with torch.no_grad():
        network.heads.action_embedding.weight[:7].zero_()

    result = train_recurrent_entry_calibrator(
        network,
        data,
        TeacherImitationConfig(
            epochs=120,
            batch_size=16,
            learning_rate=0.02,
            patience=24,
            seed=19,
            sequence_length=4,
            false_entry_penalty=0.0,
            ranking_loss_weight=1.0,
        ),
    )

    assert result["after"]["validation"]["entry_precision"] >= 0.95
    assert result["after"]["validation"]["balanced_accuracy"] >= 0.95
    assert result["after"]["test"]["mcc"] >= 0.90
    assert result["projection_contract"]["representation"] == (
        "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
    )


def test_protected_hash_accepts_shared_float_residual_and_rejects_outside_surface_drift() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    hidden = torch.randn(64, 8)
    agent_ids = torch.tensor(([0, 1] * 32), dtype=torch.long)
    before = _protected_logits_sha256(hidden, agent_ids, network)

    with torch.no_grad():
        calibrator.role_calibrators[0].output.bias[1].add_(0.1234567)
        calibrator.role_calibrators[1].output.bias[1].sub_(0.2345678)

    assert _protected_logits_sha256(hidden, agent_ids, network) == before

    with torch.no_grad():
        # Action 7 is outside both entry-owner surfaces and must remain exact.
        network.heads.action_embedding.weight[7, 0].add_(0.25)

    assert _protected_logits_sha256(hidden, agent_ids, network) != before


def test_conservative_calibration_uses_validation_rate_ceiling() -> None:
    torch.manual_seed(11)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.heads.entry_role_calibrator.teacher_entry_scope = "all_entry_mass_v2"
    hidden = torch.randn(400, 8)
    agent_ids = torch.tensor(([0] * 200) + ([1] * 200), dtype=torch.long)
    with torch.no_grad():
        margins = _role_logits(hidden, agent_ids, network)
        margins = margins[:, 1] - margins[:, 0]
    actions = torch.zeros(400, dtype=torch.long)
    for role, action in ((0, 1), (1, 2)):
        selected = agent_ids == role
        threshold = torch.quantile(margins[selected], 0.80)
        actions[selected & (margins >= threshold)] = action
    mask = torch.ones(400, dtype=torch.bool)

    state_before = teacher_trainable_sha256(network)
    audit = _audit_role_operating_points(
        hidden,
        agent_ids,
        actions,
        network,
        mask,
        rate_ceiling_multiplier=1.05,
        minimum_rate_ceiling=0.0,
    )
    assert teacher_trainable_sha256(network) == state_before

    for role in ("buy", "sell"):
        assert audit[role]["predicted_entry_rate"] <= (
            1.05 * audit[role]["label_entry_rate"] + 1e-12
        )
        assert audit[role]["entry_precision"] == pytest.approx(1.0)
        assert audit[role]["rate_ceiling_multiplier"] == pytest.approx(1.05)


def test_zero_boundary_calibration_uses_train_operating_point_and_embeds_bias() -> None:
    torch.manual_seed(23)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.heads.entry_role_calibrator.teacher_entry_scope = "all_entry_mass_v2"
    hidden = torch.randn(400, 8)
    agent_ids = torch.tensor(([0] * 200) + ([1] * 200), dtype=torch.long)
    actions = torch.zeros(400, dtype=torch.long)
    with torch.no_grad():
        margins = _role_logits(hidden, agent_ids, network)
        margins = margins[:, 1] - margins[:, 0]
    for role, action in ((0, 1), (1, 2)):
        selected = agent_ids == role
        threshold = torch.quantile(margins[selected], 0.80)
        actions[selected & (margins >= threshold)] = action
    train_mask = torch.ones(400, dtype=torch.bool)
    before = teacher_trainable_sha256(network)

    audit = _calibrate_role_zero_boundary_from_train(
        hidden, agent_ids, actions, network, train_mask
    )

    assert teacher_trainable_sha256(network) != before
    assert network.heads.entry_role_calibrator.teacher_calibration_mode == (
        "role_specific_train_operating_point_zero_boundary_v6"
    )
    with torch.no_grad():
        predictions = torch.argmax(_role_logits(hidden, agent_ids, network), dim=-1)
    for role, name in ((0, "buy"), (1, "sell")):
        selected = agent_ids == role
        assert float((predictions[selected] == 1).float().mean()) == pytest.approx(0.20)
        assert audit[name]["train_zero_boundary_entry_rate"] == pytest.approx(0.20)
        assert audit[name]["train_operating_point_entry_precision"] == pytest.approx(1.0)
        assert audit[name]["train_operating_point_eligible"] == pytest.approx(1.0)


def test_zero_boundary_calibration_reserves_validation_drift_allowance() -> None:
    torch.manual_seed(29)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.heads.entry_role_calibrator.teacher_entry_scope = "all_entry_mass_v2"
    hidden = torch.randn(400, 8)
    agent_ids = torch.tensor(([0] * 200) + ([1] * 200), dtype=torch.long)
    actions = torch.zeros(400, dtype=torch.long)
    with torch.no_grad():
        margins = _role_logits(hidden, agent_ids, network)
        margins = margins[:, 1] - margins[:, 0]
    for role, action in ((0, 1), (1, 2)):
        selected = agent_ids == role
        threshold = torch.quantile(margins[selected], 0.875)
        actions[selected & (margins >= threshold)] = action

    audit = _calibrate_role_zero_boundary_from_train(
        hidden, agent_ids, actions, network, torch.ones(400, dtype=torch.bool)
    )

    with torch.no_grad():
        predictions = torch.argmax(_role_logits(hidden, agent_ids, network), dim=-1)
    for role, name in ((0, "buy"), (1, "sell")):
        selected = agent_ids == role
        label_rate = float((actions[selected] != 0).float().mean())
        predicted_rate = float((predictions[selected] == 1).float().mean())
        assert predicted_rate <= label_rate + 1e-12
        assert audit[name]["train_zero_boundary_entry_rate"] <= label_rate + 1e-12


def test_ineligible_calibration_falls_back_inside_rate_ceiling() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.heads.entry_role_calibrator.teacher_entry_scope = "all_entry_mass_v2"
    hidden = torch.zeros(200, 8)
    agent_ids = torch.tensor(([0] * 100) + ([1] * 100), dtype=torch.long)
    actions = torch.zeros(200, dtype=torch.long)
    actions[::10] = 1
    actions[101::10] = 2
    mask = torch.ones(200, dtype=torch.bool)

    audit = _audit_role_operating_points(
        hidden, agent_ids, actions, network, mask
    )

    for role in ("buy", "sell"):
        assert audit[role]["eligible"] == 0.0
        assert audit[role]["predicted_entry_rate"] <= audit[role]["rate_ceiling"]


def test_mt5_warmup_freezes_then_anchors_imitation_calibrator() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator.teacher_imitation_trained = True
    trainer = FivePPOTrainer(network)
    trainer.configure_teacher_imitation_retention(
        freeze_updates=1, anchor_updates=3, anchor_weight=0.5,
    )
    before = {name: value.detach().clone() for name, value in calibrator.named_parameters()}
    pathway_before = {
        name: value.detach().clone()
        for name, value in network.named_parameters()
        if name.startswith((
            "observation_encoder.", "lstm.", "post_lstm_transformer.",
            "heads.action_query.", "heads.action_embedding.",
        ))
    }
    hidden = torch.randn(6, 8)
    agent_ids = torch.tensor([0, 1, 0, 1, 0, 1])

    first = trainer.step_from_loss(
        network.heads.primary_action_logits(hidden, agent_ids)[:, :7].sum()
    )
    assert first["teacher_imitation_warmup_active"] == 1.0
    assert first["teacher_imitation_pathway_frozen"] == 1.0
    assert first["teacher_imitation_representation_frozen"] == 1.0
    for name, value in calibrator.named_parameters():
        torch.testing.assert_close(value, before[name])
    for name, value in network.named_parameters():
        if name in pathway_before:
            torch.testing.assert_close(value, pathway_before[name])

    second = trainer.step_from_loss(
        network.heads.primary_action_logits(hidden, agent_ids)[:, :7].sum()
    )
    assert second["teacher_imitation_warmup_active"] == 0.0
    assert second["teacher_imitation_pathway_frozen"] == 0.0
    assert second["teacher_imitation_representation_frozen"] == 1.0
    assert second["teacher_imitation_anchor_active"] == 1.0
    assert any(not torch.equal(value, before[name]) for name, value in calibrator.named_parameters())
    for name, value in network.named_parameters():
        if name in pathway_before:
            torch.testing.assert_close(value, pathway_before[name])

    third = trainer.step_from_loss(
        network.heads.primary_action_logits(hidden, agent_ids)[:, :7].sum()
    )
    assert third["teacher_imitation_representation_frozen"] == 1.0
    for name, value in network.named_parameters():
        if name in pathway_before:
            torch.testing.assert_close(value, pathway_before[name])

    fourth = trainer.step_from_loss(
        network.heads.primary_action_logits(hidden, agent_ids)[:, :7].sum()
    )
    assert fourth["teacher_imitation_representation_frozen"] == 0.0
    assert any(
        not torch.equal(value, pathway_before[name])
        for name, value in network.named_parameters()
        if name in pathway_before
    )


def test_mt5_retention_freezes_and_anchors_teacher_geometry_heads() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator.teacher_imitation_trained = True
    network.heads.teacher_geometry_imitation_trained = True
    trainer = FivePPOTrainer(network)
    trainer.configure_teacher_imitation_retention(
        freeze_updates=1, anchor_updates=3, anchor_weight=0.5,
    )
    geometry_before = {
        f"{head}.{name}": value.detach().clone()
        for head in ("delay", "sl", "tp", "trailing")
        for name, value in network.heads.parameter_heads[head].named_parameters()
    }
    hidden = torch.randn(6, 8)
    geometry_loss = sum(
        network.heads.parameter_heads[name](hidden).sum()
        for name in ("delay", "sl", "tp", "trailing")
    )
    first = trainer.step_from_loss(geometry_loss)
    assert first["teacher_imitation_geometry_enabled"] == 1.0
    for head in ("delay", "sl", "tp", "trailing"):
        for name, value in network.heads.parameter_heads[head].named_parameters():
            torch.testing.assert_close(value, geometry_before[f"{head}.{name}"])

    second_loss = sum(
        network.heads.parameter_heads[name](hidden).sum()
        for name in ("delay", "sl", "tp", "trailing")
    )
    second = trainer.step_from_loss(second_loss)
    assert second["teacher_imitation_anchor_active"] == 1.0
    assert any(
        not torch.equal(value, geometry_before[f"{head}.{name}"])
        for head in ("delay", "sl", "tp", "trailing")
        for name, value in network.heads.parameter_heads[head].named_parameters()
    )


def test_seven_choice_retention_covers_every_retrained_output_head() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator.teacher_imitation_trained = True
    network.heads.seven_choice_imitation_trained = True
    network.heads.teacher_geometry_imitation_trained = True
    trainer = FivePPOTrainer(network)
    trainer.configure_teacher_imitation_retention(
        freeze_updates=1, anchor_updates=3, anchor_weight=0.5,
    )

    retained = trainer._teacher_retained_parameters()
    expected_prefixes = (
        "seven_choice.action_query.",
        "seven_choice.action_embedding.",
        "action.",
        "geometry.delay.",
        "geometry.entry_offset.",
        "geometry.sl.",
        "geometry.tp.",
        "geometry.trailing.",
        "seven_choice.action_path.",
    )
    for prefix in expected_prefixes:
        assert any(name.startswith(prefix) for name in retained), prefix

    before = {name: value.detach().clone() for name, value in retained.items()}
    loss = sum(value.sum() for value in retained.values())
    stats = trainer.step_from_loss(loss)

    assert stats["teacher_imitation_warmup_active"] == 1.0
    for name, value in trainer._teacher_retained_parameters().items():
        torch.testing.assert_close(value, before[name])


def test_seven_choice_checkpoint_missing_runtime_metadata_fails_closed() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.heads.seven_choice_imitation_trained = True
    with pytest.raises(RuntimeError, match="thieu cau hinh quyet dinh sau chi phi"):
        network.require_current_architecture()


def test_seven_choice_checkpoint_runtime_metadata_round_trips() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    heads = network.heads
    heads.seven_choice_imitation_trained = True
    heads.seven_choice_expected_value_decision_enabled = True
    heads.seven_choice_action_decision_mode = "expected_value"
    heads.seven_choice_expected_value_threshold_r = 0.05
    heads.seven_choice_expected_value_score_mode = "filled_value"
    heads.seven_choice_expected_value_fill_probability_floor = 0.10
    heads.seven_choice_expected_value_action_residual_scale = 0.0
    heads.seven_choice_expected_value_logit_scale = 10.0
    heads.seven_choice_fixed_geometry_enabled = True
    heads.seven_choice_fixed_delay_index = 0
    heads.seven_choice_fixed_entry_offset_index = 7
    heads.seven_choice_fixed_sl_index = 2
    heads.seven_choice_fixed_tp_index = 5
    heads.seven_choice_fixed_trailing_index = 0
    network.require_current_architecture()
    assert heads.seven_choice_action_decision_mode == "expected_value"
    assert heads.seven_choice_expected_value_score_mode == "filled_value"


def test_checkpoint_save_reset_drops_stale_ppo_and_rollout_state() -> None:
    from types import SimpleNamespace

    policy = SimpleNamespace(
        five_ppo_trainer=object(),
        five_rollout_steps=[1],
        five_pattern_steps=[2],
        five_pending_timeout_steps=[3],
        five_lifecycle_steps=[4],
        five_last_training_stats={"updated": 1.0},
    )
    system = SimpleNamespace(registry=SimpleNamespace(
        chuyen_gia_theo_context={
            "A": SimpleNamespace(policy=policy),
            "B": SimpleNamespace(policy=policy),
        }
    ))

    assert _reset_five_runtime_after_imitation(system) == 1
    assert policy.five_ppo_trainer is None
    assert policy.five_rollout_steps == []
    assert policy.five_pattern_steps == []
    assert policy.five_pending_timeout_steps == []
    assert policy.five_lifecycle_steps == []
    assert policy.five_last_training_stats == {}
