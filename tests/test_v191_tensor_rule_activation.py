import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_tensor_rule_v186_candidate as candidate_runner
from self_growing_trader.config import load_config
from self_growing_trader.tensor_rule_bank import TensorRuleBankState


def test_interrupted_codex_status_is_recovered_into_feedback_contract(tmp_path) -> None:
    receipts = tmp_path / (
        "reasoning_rl_v1_active_candidate.codex_teacher_receipts.jsonl"
    )
    status_path = Path(str(receipts) + ".status.json.latest.json")
    status_path.write_text(
        json.dumps({
            "enabled": True,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "xhigh",
            "executable_path": "C:/Codex/codex.exe",
            "cli_version": "codex-cli test",
            "preflight": {"status": "passed", "stage": "complete"},
            "submitted": 35,
            "graded": 32,
            "hypotheses_confirmed": 32,
            "updates": 32,
            "interventions_applied": 32,
            "failures": 0,
            "pending": 3,
            "account_impact_records": 32,
            "account_impact_confirmed": 21,
            "account_impact_partial": 11,
            "account_impact_reward_sum": -3.18,
            "rlhf_parameter_changed": 32,
            "rlhf_internal_policy_state_changed": 20,
            "rlhf_probability_shift_count": 32,
            "rlhf_max_probability_shift": 0.0499,
            "codex_causal_links_observed": 12,
            "codex_causal_updates_linked": 11,
            "codex_causal_hash_matches": 10,
            "codex_causal_hash_unknown": 1,
            "last_codex_causal_link": {
                "causal_link": "barrier_to_next_action",
                "policy_parameter_hash_matches_update": True,
            },
            "report_ledger_reconciled": True,
        }),
        encoding="utf-8",
    )
    contract = {
        "runtime_status": "running",
        "effective": {
            "five_codex_teacher_effective": False,
            "codex_preflight_passed": False,
            "codex_optimizer_updates": 0,
        },
    }

    recovered = candidate_runner.recover_interrupted_codex_feedback(
        contract, tmp_path
    )

    assert recovered["submitted"] == 35
    assert recovered["graded"] == 32
    assert recovered["updates"] == 32
    assert recovered["account_impact_confirmed"] == 21
    assert recovered["rlhf_parameter_changed"] == 32
    assert contract["codex_teacher_phase"]["codex_causal_links_observed"] == 12
    assert contract["codex_teacher_phase"]["codex_causal_updates_linked"] == 11
    assert contract["codex_teacher_phase"]["codex_causal_hash_matches"] == 10
    assert contract["codex_teacher_phase"]["codex_causal_hash_unknown"] == 1
    assert contract["codex_teacher_phase"]["last_codex_causal_link"]["causal_link"] == (
        "barrier_to_next_action"
    )
    assert contract["codex_teacher_phase"]["report_ledger_reconciled"] is True
    assert contract["codex_teacher_phase"]["updates"] == 32
    assert contract["codex_teacher_phase"]["codex_causal_hash_matches"] == 10
    assert contract["codex_teacher_phase"]["account_impact_confirmed"] == 21
    assert contract["effective"]["five_codex_teacher_effective"] is True
    assert contract["effective"]["codex_preflight_passed"] is True
    assert contract["effective"]["codex_optimizer_updates"] == 32
    assert contract["runtime_feedback_contract"][
        "codex_rlhf_parameter_changed"
    ] == 32
    assert contract["codex_teacher_status_recovered"] is True


def test_runtime_contract_syncs_recovery_interval_into_effective() -> None:
    contract = {
        "effective": {
            "five_rlhf_recovery_snapshot_interval_updates": 8,
        }
    }

    candidate_runner._sync_runtime_recovery_snapshot_interval(
        contract,
        {"codex_teacher_recovery_snapshot_interval_updates": 12},
    )
    assert contract["effective"][
        "five_rlhf_recovery_snapshot_interval_updates"
    ] == 12

    # An older/interrupted runtime contract may omit the field; do not erase
    # the profile value that was written before MT5 started.
    candidate_runner._sync_runtime_recovery_snapshot_interval(contract, {})
    assert contract["effective"][
        "five_rlhf_recovery_snapshot_interval_updates"
    ] == 12


def test_v191_profile_requests_bounded_residual_authority(tmp_path) -> None:
    base = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    profile = yaml.safe_load(
        Path("configs/experiments/action_conditioned_abstraction_v191_candidate.yaml")
        .read_text(encoding="utf-8")
    )

    config = candidate_runner.candidate_config(base, profile, tmp_path)

    assert config.learning.tensor_rule_authority == "residual"
    assert config.learning.tensor_rule_actor_residual_clip == 0.25
    assert config.learning.tensor_rule_entry_lcb_gate_enabled is True
    assert config.learning.tensor_rule_membership_temperature == 0.04
    assert config.learning.tensor_rule_motif_distance_weight == 0.75
    assert config.learning.tensor_rule_ready_gradient_norm_budget == 0.50
    assert config.learning.direct_rnn_state_memory_temperature == 0.35
    assert config.learning.direct_rnn_state_memory_injection_scale == 0.50
    assert config.learning.direct_rnn_state_memory_path_delta_scale == 0.75
    assert config.learning.direct_rnn_rl_exploration_enabled is True
    assert config.learning.direct_rnn_rl_exploration_rate == 0.10
    assert "authority" not in profile["isolation"]
    assert "action_scale" not in profile["isolation"]


def test_imitation_frozen_profile_disables_all_mt5_learning(tmp_path) -> None:
    base = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    profile = yaml.safe_load(
        Path("configs/experiments/imitation_mt5_frozen_ab_v2.yaml")
        .read_text(encoding="utf-8")
    )

    config = candidate_runner.candidate_config(base, profile, tmp_path)

    assert base.mt5_learning_enabled is True
    assert config.mt5_learning_enabled is False
    assert config.learning.five_frozen_behavior_sampling_enabled is True
    assert config.learning.five_frozen_behavior_sampling_seed == 20260702
    assert config.learning.five_mt5_failure_replay_enabled is False
    assert config.learning.five_mt5_failure_replay_sessions_per_run == 0
    assert config.learning.execution_trace_observatory_include_recurrent_state is False
    assert config.learning.execution_trace_observatory_max_bytes == 0
    assert config.observability.event_path == str(tmp_path / "events.jsonl")


def test_v193_edge_profile_uses_exact_scope_expectancy_gate_and_bounded_replay(tmp_path) -> None:
    base = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    profile = yaml.safe_load(
        Path("configs/experiments/edge_discovery_v193_candidate.yaml")
        .read_text(encoding="utf-8")
    )

    config = candidate_runner.candidate_config(base, profile, tmp_path)

    assert config.learning.context_expectancy_entry_gate_enabled is True
    assert config.learning.context_expectancy_entry_gate_min_support == 12.0
    assert config.learning.context_expectancy_entry_gate_max_lcb == -0.15
    assert config.learning.context_expectancy_entry_gate_allowed_contexts == ["CTX_CHUNG"]
    assert config.learning.context_expectancy_entry_gate_scope_fallback_enabled is False
    assert config.learning.five_mt5_failure_replay_sessions_per_run == 4
    assert config.learning.input_normalization_mode == "raw"


def test_candidate_profile_can_control_contextual_expectancy_aggregation(tmp_path) -> None:
    base = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    profile = {
        "expectancy_overrides": {
            "merge_similarity": 0.75,
        }
    }

    config = candidate_runner.candidate_config(base, profile, tmp_path)

    assert config.expectancy.merge_similarity == pytest.approx(0.75)
    assert config.expectancy.min_samples == base.expectancy.min_samples
    assert config.expectancy.promotion_lcb_min_delta == base.expectancy.promotion_lcb_min_delta


def test_seven_choice_checkpoint_contract_is_checked_before_mt5(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pkl"
    checkpoint.write_bytes(b"placeholder")
    report = {
        "dataset_manifest": {
            "config": {
                "horizon_bars": 48,
                "pending_expiry_bars": 5,
                "entry_offset_cap_atr": 5.0,
                "fixed_sl_atr": 1.5,
                "fixed_tp_rr": 2.0,
                "holding_horizon_from_fill": True,
                "risk_denominator_from_fill": False,
                "pending_fill_price_mode": "order",
            }
        }
    }
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".seven_choice.json")
    sidecar.write_text(json.dumps(report), encoding="utf-8")

    matching = load_config("configs/he_chuyen_gia_reasoning_rl_v1_candidate_rr2.yaml")
    inspected = candidate_runner.inspect_seven_choice_runtime_contract(
        checkpoint, matching
    )
    assert inspected["declared"] is True
    assert inspected["match"] is True

    mismatching = load_config("configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml")
    inspected = candidate_runner.inspect_seven_choice_runtime_contract(
        checkpoint, mismatching
    )
    assert inspected["declared"] is True
    assert inspected["match"] is False
    assert set(inspected["mismatches"]) == {"fixed_sl_atr", "max_holding_bars"}


def test_candidate_configurator_does_not_force_telemetry(monkeypatch) -> None:
    bank = TensorRuleBankState.seed(seed=91, authority="telemetry")
    policy = SimpleNamespace(
        tensor_rule_authority="telemetry",
        modules=[SimpleNamespace(tensor_rule_bank=bank)],
    )
    system = SimpleNamespace(experts={"CTX": SimpleNamespace(policy=policy)})
    config = SimpleNamespace(
        learning=SimpleNamespace(tensor_rule_authority="residual")
    )
    monkeypatch.setattr(
        candidate_runner,
        "_cau_hinh_policy_chuyen_gia_cho_mt5",
        lambda _system, _config: None,
    )

    candidate_runner.configure_candidate(system, config)

    assert policy.tensor_rule_authority == "residual"
    assert bank.authority == "residual"


def test_candidate_parser_accepts_mt5_visual_as_opt_in() -> None:
    default_args = candidate_runner.build_parser().parse_args([])
    visual_args = candidate_runner.build_parser().parse_args(["--mt5-visual"])

    assert default_args.mt5_visual is False
    assert visual_args.mt5_visual is True


def test_imitation_seed_provenance_rejects_retired_reports(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"candidate")
    digest = candidate_runner.sha256_file(seed)
    sidecar = seed.with_suffix(".pkl.imitation.json")
    sidecar.write_text(json.dumps({
        "schema_version": 2,
        "candidate_checkpoint_sha256": digest,
        "dataset_manifest": {"sample_count": 960120},
        "verification": {"candidate_reload_passed": True},
    }), encoding="utf-8")

    legacy = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert legacy["declared"] is True
    assert legacy["effective_for_mt5"] is False
    assert legacy["entry_scope"] == "market_only_v1"

    sidecar.write_text(json.dumps({
        "schema_version": 3,
        "candidate_checkpoint_sha256": digest,
        "dataset_manifest": {"sample_count": 960120},
        "mt5_integration": {
            "entry_scope": "all_entry_mass_v2",
            "requires_runtime_teacher_retention": True,
        },
        "verification": {"candidate_reload_passed": True},
        "training": {"after": {"test": {
            "balanced_accuracy": 0.81,
            "entry_f0_5": 0.73,
        }}},
    }), encoding="utf-8")

    retired_action = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert retired_action["effective_for_mt5"] is False
    assert retired_action["autonomous_validation_certified"] is False
    assert retired_action["offline_test_metrics_available"] is True

    sidecar.write_text(json.dumps({
        "schema_version": 6,
        "candidate_checkpoint_sha256": digest,
        "dataset_manifest": {"sample_count": 960120},
        "mt5_integration": {
            "entry_scope": "all_entry_mass_v2",
            "geometry_scope": "accepted_pivot_entries_v1",
            "geometry_objective": "ordinal_log_distance_v2",
            "execution_contract_version": 7,
            "requires_runtime_teacher_retention": True,
        },
        "verification": {
            "candidate_reload_passed": True,
            "geometry_checkpoint_reload_passed": True,
        },
        "geometry_training": {"after": {"test": {
            "sl_accuracy": 0.62, "tp_accuracy": 0.31, "joint_accuracy": 0.20,
        }}},
    }), encoding="utf-8")
    geometry = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert geometry["effective_for_mt5"] is False
    assert geometry["geometry_effective_for_mt5"] is False
    assert geometry["execution_contract_version"] == 7
    assert geometry["geometry_objective"] == "ordinal_log_distance_v2"
    assert geometry["geometry_test_tp_accuracy"] == pytest.approx(0.31)
    assert geometry["reason"] == "autonomous_validation_missing_or_failed"

    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["schema_version"] = 7
    sidecar_payload["mt5_integration"]["geometry_scope"] = "accepted_pivot_entries_sl_only_v1"
    sidecar_payload["mt5_integration"]["geometry_heads"] = ["sl"]
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    sl_only = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert sl_only["geometry_effective_for_mt5"] is False
    assert sl_only["geometry_heads"] == ["sl"]
    assert sl_only["reason"] == "autonomous_validation_missing_or_failed"


def test_imitation_seed_provenance_rejects_sidecar_for_different_checkpoint(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"candidate")
    seed.with_suffix(".pkl.imitation.json").write_text(json.dumps({
        "schema_version": 3,
        "candidate_checkpoint_sha256": "0" * 64,
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="khong khop checkpoint"):
        candidate_runner.inspect_imitation_seed(
            seed, checkpoint_sha256=candidate_runner.sha256_file(seed)
        )


def test_mt5_continuation_seed_accepts_exact_reconciled_saved_candidate(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"online-candidate")
    digest = candidate_runner.sha256_file(seed)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 5,
        "mode": "mt5-smoke",
        "candidate": str(seed.resolve()),
        "candidate_sha256": digest,
        "source_config_unchanged": True,
        "source_seed_unchanged": True,
        "source_seed_imitation": {
            "effective_for_mt5": True,
            "autonomous_validation_certified": True,
            "report_path": "teacher.json",
        },
        "mt5": {
            "interrupted": False,
            "metrics": {
                "net_profit": 12.5,
                "fitness_eligible": False,
                "reconciliation": {"reconciled": True},
                "mt5_failure_replay": {"post_replay_validation": {
                    "performed": True,
                    "metrics_match_saved_weights": True,
                }},
            },
        },
    }), encoding="utf-8")

    inspected = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)

    assert inspected["effective_for_mt5"] is True
    assert inspected["reason"] == "mt5_continuation_lineage"
    assert inspected["autonomous_validation_certified"] is True
    assert inspected["previous_fitness_eligible"] is False
    assert inspected["previous_net_profit"] == pytest.approx(12.5)

    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    payload["source_seed_imitation"].pop("autonomous_validation_certified")
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    uncertified = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert uncertified["effective_for_mt5"] is False
    assert uncertified["upstream_autonomous_validation_certified"] is False


def test_mt5_continuation_seed_allows_research_without_promotion_lineage(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"research-candidate")
    digest = candidate_runner.sha256_file(seed)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 5,
        "mode": "mt5-smoke",
        "candidate": str(seed.resolve()),
        "candidate_sha256": digest,
        "source_config_unchanged": True,
        "source_seed_unchanged": True,
        "checkpoint_collapse_guard": {
            "collapsed": False,
            "status": "insufficient_data",
        },
        "source_seed_imitation": {
            "effective_for_mt5": False,
            "autonomous_validation_certified": False,
        },
        "mt5": {
            "interrupted": False,
            "metrics": {"reconciliation": {"reconciled": True}},
        },
    }), encoding="utf-8")

    inspected = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)

    assert inspected["effective_for_mt5"] is False
    assert inspected["promotion_lineage_verified"] is False
    assert inspected["research_continuation_allowed"] is True
    assert inspected["reason"] == "mt5_research_continuation"
    assert inspected["saved_weights_verified"] is False

    allowed = candidate_runner.apply_imitation_seed_gate_policy(
        inspected, allow_unverified=False,
    )
    assert allowed["mt5_execution_allowed"] is True
    assert allowed["gate_bypassed"] is False
    assert allowed["execution_scope"] == "mt5_smoke_research_continuation"
    assert allowed["promotable"] is False
    assert allowed["promotion_block_reason"] == "promotion_lineage_unverified"

    collapsed = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    collapsed["checkpoint_collapse_guard"]["collapsed"] = True
    (tmp_path / "manifest.json").write_text(
        json.dumps(collapsed), encoding="utf-8",
    )
    rejected = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert rejected["research_continuation_allowed"] is False
    assert rejected["reason"] == "invalid_mt5_continuation_lineage"


def test_mt5_continuation_seed_rejects_hash_or_saved_weight_mismatch(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"online-candidate")
    digest = candidate_runner.sha256_file(seed)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 5,
        "mode": "mt5-smoke",
        "candidate": str(seed.resolve()),
        "candidate_sha256": "0" * 64,
        "source_config_unchanged": True,
        "source_seed_unchanged": True,
        "source_seed_imitation": {
            "effective_for_mt5": True,
            "autonomous_validation_certified": True,
        },
        "mt5": {"metrics": {
            "reconciliation": {"reconciled": True},
            "mt5_failure_replay": {"post_replay_validation": {
                "performed": True,
                "metrics_match_saved_weights": False,
            }},
        }},
    }), encoding="utf-8")

    inspected = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)

    assert inspected["effective_for_mt5"] is False
    assert inspected["reason"] == "invalid_mt5_continuation_lineage"
    assert inspected["checkpoint_sha256_matches"] is False
    assert inspected["research_continuation_allowed"] is False
    assert inspected["saved_weights_verified"] is False


def test_schema4_mt5_continuation_is_retired(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"online-candidate")
    digest = candidate_runner.sha256_file(seed)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 4,
        "mode": "mt5-smoke",
        "candidate_sha256": digest,
        "source_config_unchanged": True,
        "source_seed_unchanged": True,
        "source_seed_imitation": {
            "effective_for_mt5": True,
            "autonomous_validation_certified": True,
        },
        "mt5": {"metrics": {
            "reconciliation": {"reconciled": True},
            "paired_seed_validation": {
                "performed": True,
                "selected": "source_seed",
                "metrics_match_saved_weights": True,
            },
        }},
    }), encoding="utf-8")

    inspected = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)

    assert inspected["effective_for_mt5"] is False
    assert inspected["candidate_path_matches"] is False
    assert inspected["paired_seed_validation_performed"] is True
    assert inspected["paired_seed_selected"] == "source_seed"


def test_imitation_seed_provenance_marks_resume_checkpoint_as_paused(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"stale-output")
    seed.with_suffix(".pkl.resume.pkl").write_bytes(b"resume-state")

    inspected = candidate_runner.inspect_imitation_seed(
        seed, checkpoint_sha256=candidate_runner.sha256_file(seed)
    )

    assert inspected["declared"] is True
    assert inspected["effective_for_mt5"] is False
    assert inspected["reason"] == "imitation_paused"
    assert inspected["resume_path"].endswith("candidate.pkl.resume.pkl")


def test_explicit_imitation_seed_bypass_allows_only_research_mt5_execution() -> None:
    evidence = {
        "declared": False,
        "effective_for_mt5": False,
        "reason": "no_imitation_sidecar",
    }

    rejected = candidate_runner.apply_imitation_seed_gate_policy(
        evidence, allow_unverified=False,
    )
    bypassed = candidate_runner.apply_imitation_seed_gate_policy(
        evidence, allow_unverified=True,
    )

    assert rejected["mt5_execution_allowed"] is False
    assert rejected["gate_bypassed"] is False
    assert bypassed["effective_for_mt5"] is False
    assert bypassed["mt5_execution_allowed"] is True
    assert bypassed["gate_bypassed"] is True
    assert bypassed["bypass_scope"] == "mt5_smoke_research_only"
    assert bypassed["promotable"] is False


def test_current_imitation_requires_schema11_and_both_validation_gates(tmp_path) -> None:
    seed = tmp_path / "candidate.pkl"
    seed.write_bytes(b"candidate")
    digest = candidate_runner.sha256_file(seed)
    sidecar = seed.with_suffix(".pkl.imitation.json")
    payload = {
        "schema_version": 10,
        "candidate_checkpoint_sha256": digest,
        "mt5_integration": {
            "entry_scope": "all_entry_mass_v2",
            "representation_scope": (
                "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
            ),
            "requires_runtime_teacher_retention": True,
        },
        "verification": {"candidate_reload_passed": True},
        "training": {"validation_gate": {"enforced": False, "passed": True}},
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    rejected = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert rejected["effective_for_mt5"] is False

    payload["training"]["validation_gate"]["enforced"] = True
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    legacy_rejected = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert legacy_rejected["effective_for_mt5"] is False
    assert legacy_rejected["reason"] == "autonomous_validation_missing_or_failed"

    payload["schema_version"] = 11
    payload["mt5_integration"]["requires_autonomous_continuous_validation"] = True
    payload["training"]["autonomous_validation_gate"] = {
        "enforced": True,
        "passed": False,
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    autonomous_rejected = candidate_runner.inspect_imitation_seed(
        seed, checkpoint_sha256=digest
    )
    assert autonomous_rejected["effective_for_mt5"] is False
    assert autonomous_rejected["reason"] == "autonomous_validation_missing_or_failed"
    assert autonomous_rejected["autonomous_validation_gate_enforced"] is True

    payload["training"]["autonomous_validation_gate"]["passed"] = True
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    accepted = candidate_runner.inspect_imitation_seed(seed, checkpoint_sha256=digest)
    assert accepted["effective_for_mt5"] is True
    assert accepted["validation_gate_passed"] is True
    assert accepted["autonomous_validation_gate_passed"] is True
    assert accepted["autonomous_validation_certified"] is True

    payload["mt5_integration"].update({
        "geometry_scope": "accepted_pivot_entries_v1",
        "geometry_objective": "ordinal_log_distance_v2",
        "execution_contract_version": 7,
    })
    payload["verification"]["geometry_checkpoint_reload_passed"] = True
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    geometry_accepted = candidate_runner.inspect_imitation_seed(
        seed, checkpoint_sha256=digest
    )
    assert geometry_accepted["effective_for_mt5"] is True
    assert geometry_accepted["geometry_effective_for_mt5"] is True


def test_manifest_ghi_checkpoint_incomplete_cua_loi_mt5(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.incomplete.pkl"
    metadata = tmp_path / "candidate.incomplete.json"
    checkpoint.write_bytes(b"learned-state")
    metadata.write_text('{"promotable": false}', encoding="utf-8")
    error = RuntimeError("MT5 exited before writing report")
    error.checkpoint_path = str(checkpoint)
    error.metadata_path = str(metadata)
    error.original_error_type = "RuntimeError"

    row = candidate_runner.incomplete_checkpoint_manifest(error)

    assert row is not None
    assert row["path"] == str(checkpoint)
    assert row["metadata"] == str(metadata)
    assert row["bytes"] == len(b"learned-state")
    assert row["sha256"] == candidate_runner.sha256_file(checkpoint)
    assert row["promotable"] is False
    assert row["original_error_type"] == "RuntimeError"


def test_incomplete_checkpoint_is_exposed_as_research_candidate(tmp_path) -> None:
    incomplete_path = tmp_path / "reasoning_rl_v1_active_candidate.incomplete.pkl"
    incomplete_path.write_bytes(b"partial-learned-state")
    candidate_path = tmp_path / "reasoning_rl_v1_active_candidate.pkl"
    row = {
        "path": str(incomplete_path),
        "sha256": candidate_runner.sha256_file(incomplete_path),
        "bytes": incomplete_path.stat().st_size,
        "promotable": False,
    }
    manifest = {"promotable": True}

    result = candidate_runner.expose_incomplete_checkpoint(
        manifest,
        incomplete=row,
        candidate_path=candidate_path,
        metrics_only=False,
    )

    assert result["candidate"] == str(incomplete_path)
    assert result["candidate_sha256"] == row["sha256"]
    assert result["checkpoint_persistence_decision"] == (
        "incomplete_snapshot_saved_research_only"
    )
    assert result["checkpoint_incomplete"] is True
    assert result["checkpoint_reuse_method"] == "none"
    assert result["promotable"] is False
    assert result["promotion_block_reason"] == "incomplete_run_research_only"


def test_interrupted_mt5_result_discovers_incomplete_artifact(tmp_path) -> None:
    incomplete_path = tmp_path / "candidate.incomplete.pkl"
    metadata_path = tmp_path / "candidate.incomplete.json"
    incomplete_path.write_bytes(b"partial-state")
    metadata_path.write_text('{"status":"incomplete"}', encoding="utf-8")

    row = candidate_runner.incomplete_checkpoint_manifest_from_result({
        "checkpoint": {
            "incomplete_artifact": str(incomplete_path),
            "incomplete_metadata": str(metadata_path),
        },
        "metrics": {"interrupted_reason": "max_wall_sec"},
    })

    assert row is not None
    assert row["path"] == str(incomplete_path)
    assert row["metadata"] == str(metadata_path)
    assert row["sha256"] == candidate_runner.sha256_file(incomplete_path)
    assert row["original_error_type"] == "max_wall_sec"


def test_candidate_payload_quota_excludes_protected_seed_clone(tmp_path) -> None:
    seed = tmp_path / "source_seed_clone.pkl"
    payload = tmp_path / "candidate.incomplete.pkl"
    seed.write_bytes(b"s" * 4096)
    payload.write_bytes(b"p" * 3072)

    assert candidate_runner.directory_size(tmp_path) == 7168
    assert candidate_runner.directory_size(tmp_path, exclude=(seed,)) == 3072


def test_storage_preflight_failure_is_durable_before_mt5_start(tmp_path) -> None:
    record = candidate_runner.write_storage_preflight_failure(
        tmp_path,
        mode="mt5-smoke",
        free_bytes=100,
        reserve_floor_bytes=1_000,
        required_free_bytes=2_000,
        projected_payload_bytes=1_000,
        reason="free_below_reserve_plus_projected_payload",
    )

    failure = json.loads(
        (tmp_path / "storage_gate_failure.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert failure == record
    assert failure["type"] == "storage_preflight_failure"
    assert failure["reason"] == "free_below_reserve_plus_projected_payload"
    assert failure["required_free_bytes"] == 2_000
    assert manifest["promotable"] is False
    assert manifest["promotion_block_reason"] == "storage_preflight_failed"
    assert manifest["storage_gate_failure"] == record


def test_candidate_rejects_nonexistent_calendar_date_and_reversed_window() -> None:
    with pytest.raises(ValueError, match="ngay ton tai"):
        candidate_runner.validate_date_window("2025.01.01", "2025.02.30")
    with pytest.raises(ValueError, match="som hon"):
        candidate_runner.validate_date_window("2025.03.01", "2025.03.01")
    start, end = candidate_runner.validate_date_window("2025.02.28", "2025.03.01")
    assert start < end


def test_mt5_candidate_rejects_checkpoint_without_valid_imitation_sidecar(
    tmp_path, monkeypatch,
) -> None:
    seed = tmp_path / "misleading_teacher.pkl"
    seed.write_bytes(b"random-weights")
    monkeypatch.setattr(
        candidate_runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=8 * 1024**3),
    )
    monkeypatch.setattr(
        candidate_runner.sys,
        "argv",
        [
            "run_tensor_rule_v186_candidate.py",
            "--mode", "mt5-smoke",
            "--config", str(Path("configs/he_chuyen_gia_latest_ui_v193.yaml").resolve()),
            "--profile", str(Path("configs/experiments/edge_discovery_v193_candidate.yaml").resolve()),
            "--seed-artifact", str(seed),
            "--output-dir", str(tmp_path / "run"),
        ],
    )

    with pytest.raises(RuntimeError, match="no_imitation_sidecar"):
        candidate_runner.main()


def test_projected_payload_is_not_silently_clamped_to_small_quota() -> None:
    seed_bytes = 8 * 1024**3
    telemetry_budget = 96 * 1024**2
    projected = candidate_runner.projected_candidate_payload_bytes(
        seed_bytes, telemetry_budget,
    )
    assert projected == 10 * 1024**3
    assert projected > 6 * 1024**3


def test_projected_payload_metrics_only_reserves_telemetry_not_checkpoint() -> None:
    seed_bytes = 8 * 1024**3
    telemetry_budget = 96 * 1024**2
    projected = candidate_runner.projected_candidate_payload_bytes(
        seed_bytes, telemetry_budget, metrics_only=True,
    )
    assert projected == telemetry_budget


def test_metrics_only_storage_floor_accounts_for_temporary_runner() -> None:
    configured = 1536 * 1024**2
    runner = 960 * 1024**2
    assert candidate_runner.effective_storage_reserve_floor(
        configured, runner, metrics_only=True,
    ) == 576 * 1024**2
    assert candidate_runner.effective_storage_reserve_floor(
        configured, runner, metrics_only=False,
    ) == configured
    assert candidate_runner.effective_storage_reserve_floor(
        configured, 4 * 1024**2, metrics_only=True,
    ) == configured - 4 * 1024**2


def test_completed_metrics_only_storage_pressure_is_warning_only() -> None:
    manifest = {
        "mt5": {
            "interrupted": False,
            "training_error": "",
            "metrics": {"reconciliation": {"reconciled": True}},
        },
        "feedback_contract": {"codex_teacher_phase": {"pending": 0}},
    }
    assert candidate_runner.metrics_only_result_complete_for_storage_pressure(
        manifest,
        candidate_payload_bytes=1024,
        candidate_payload_quota=2048,
    ) is True


def test_incomplete_metrics_only_storage_pressure_remains_fail_closed() -> None:
    manifest = {
        "mt5": {
            "interrupted": False,
            "training_error": "",
            "metrics": {"reconciliation": {"reconciled": True}},
        },
        "feedback_contract": {"codex_teacher_phase": {"pending": 1}},
    }
    assert candidate_runner.metrics_only_result_complete_for_storage_pressure(
        manifest,
        candidate_payload_bytes=1024,
        candidate_payload_quota=2048,
    ) is False


def test_action_collapse_guard_rejects_hold_only_continuation(tmp_path) -> None:
    status = tmp_path / "mt5_diagnostics.json"
    status.write_text(json.dumps({
        "status": {"policy": "collapsed"},
        "policy": {
            "hold_share": 1.0,
            "effective_action_count": 1.0,
            "normalized_action_entropy": 0.0,
        },
    }), encoding="utf-8")

    result = candidate_runner.inspect_action_collapse(status)

    assert result["available"] is True
    assert result["collapsed"] is True
    assert result["hold_share"] == pytest.approx(1.0)


def test_action_collapse_guard_keeps_diverse_continuation(tmp_path) -> None:
    status = tmp_path / "mt5_diagnostics.json"
    status.write_text(json.dumps({
        "status": {"policy": "healthy"},
        "policy": {
            "hold_share": 0.80,
            "effective_action_count": 2.0,
            "normalized_action_entropy": 0.30,
        },
    }), encoding="utf-8")

    result = candidate_runner.inspect_action_collapse(status)

    assert result["available"] is True
    assert result["collapsed"] is False


def test_action_collapse_guard_preserves_wire_metrics_when_economic_scope_selected(tmp_path) -> None:
    status = tmp_path / "mt5_diagnostics.json"
    status.write_text(json.dumps({
        "status": {"policy": "collapsed"},
        "policy": {
            "scope": "economic_actor",
            "hold_share": 0.25,
            "effective_action_count": 4.0,
            "normalized_action_entropy": 0.70,
            "wire_hold_share": 0.96,
            "wire_effective_action_count": 1.20,
            "wire_normalized_action_entropy": 0.04,
            "economic_actor": {
                "total": 20,
                "hold_share": 0.25,
                "effective_action_count": 4.0,
                "normalized_action_entropy": 0.70,
            },
        },
    }), encoding="utf-8")

    result = candidate_runner.inspect_action_collapse(status)

    assert result["collapse_scope"] == "economic_actor"
    assert result["collapsed"] is False
    assert result["hold_share"] == pytest.approx(0.25)
    assert result["wire_hold_share"] == pytest.approx(0.96)
    assert result["wire_effective_action_count"] == pytest.approx(1.20)
    assert result["wire_collapse_detected"] is True
    assert result["economic_actor_collapse_detected"] is False
    assert result["wire_collapse_warning"] is True


def test_action_collapse_saves_research_candidate_without_reusing_seed() -> None:
    manifest = {"promotable": True}

    result = candidate_runner.finalize_checkpoint_persistence(
        manifest,
        metrics_only=False,
        selected_champion=False,
        collapsed=True,
        candidate_exists=True,
    )

    assert result["checkpoint_persistence_decision"] == (
        "candidate_snapshot_saved_research_only_due_action_collapse"
    )
    assert result["checkpoint_reuse_method"] == "none"
    assert result["promotable"] is False
    assert result["promotion_block_reason"] == "action_collapse_research_only"




def test_continuation_economic_verdict_reuses_parent_after_regression(tmp_path) -> None:
    source_seed = tmp_path / "generation1.pkl"
    candidate = tmp_path / "generation2.pkl"
    manifest = {
        "source_seed": str(source_seed),
        "source_seed_imitation": {
            "previous_net_profit": -76.91,
            "previous_fitness_eligible": False,
        },
        "mt5": {
            "metrics": {
                "net_profit": -151.86,
                "fitness_eligible": False,
            },
        },
        "candidate": str(candidate),
    }

    result = candidate_runner.annotate_continuation_economic_verdict(manifest)

    verdict = result["continuation_economic_verdict"]
    assert verdict["available"] is True
    assert verdict["parent_net_profit"] == pytest.approx(-76.91)
    assert verdict["candidate_net_profit"] == pytest.approx(-151.86)
    assert verdict["net_profit_delta"] == pytest.approx(-74.95)
    assert verdict["economic_regression"] is True
    assert verdict["economic_comparison"] == "regressed"
    assert verdict["recommendation"] == "reuse_parent_seed_until_fitness_eligible"
    assert verdict["preferred_continuation_seed"] == str(source_seed)
    assert result["preferred_continuation_seed"] == str(source_seed)
    assert result["checkpoint_reuse_recommended"] is False


def test_continuation_economic_verdict_prefers_eligible_candidate(tmp_path) -> None:
    source_seed = tmp_path / "parent.pkl"
    candidate = tmp_path / "candidate.pkl"
    manifest = {
        "source_seed": str(source_seed),
        "source_seed_imitation": {
            "previous_net_profit": -1.0,
            "previous_fitness_eligible": False,
        },
        "mt5": {
            "metrics": {
                "net_profit": 12.0,
                "fitness_eligible": True,
            },
        },
        "candidate": str(candidate),
    }

    result = candidate_runner.annotate_continuation_economic_verdict(manifest)

    verdict = result["continuation_economic_verdict"]
    assert verdict["economic_regression"] is False
    assert verdict["economic_comparison"] == "eligible_non_regression"
    assert verdict["recommendation"] == "candidate_preferred"
    assert verdict["preferred_continuation_seed"] == str(candidate)
    assert result["preferred_continuation_seed"] == str(candidate)
    assert result["checkpoint_reuse_recommended"] is True


def test_continuation_economic_verdict_uses_paired_candidate_before_rollback(tmp_path) -> None:
    source_seed = tmp_path / "parent.pkl"
    candidate = tmp_path / "candidate.pkl"
    manifest = {
        "source_seed": str(source_seed),
        "source_seed_imitation": {
            "previous_net_profit": -76.91,
            "previous_fitness_eligible": False,
        },
        "mt5": {
            "metrics": {
                # These are the metrics of the source checkpoint selected for
                # persistence after the paired gate rejected the candidate.
                "net_profit": -47.11,
                "fitness_eligible": False,
                "paired_seed_validation": {
                    "performed": True,
                    "candidate_net_profit": -144.78,
                    "candidate_fitness_eligible": False,
                    "selected": "source_seed",
                    "rollback_applied": True,
                },
            },
        },
        "candidate": str(candidate),
    }

    result = candidate_runner.annotate_continuation_economic_verdict(manifest)

    verdict = result["continuation_economic_verdict"]
    assert verdict["candidate_net_profit"] == pytest.approx(-144.78)
    assert verdict["net_profit_delta"] == pytest.approx(-67.87)
    assert verdict["economic_regression"] is True
    assert verdict["economic_comparison"] == "regressed"
    assert verdict["candidate_metric_source"] == (
        "paired_seed_validation.candidate_net_profit"
    )
    assert verdict["paired_validation_performed"] is True
    assert verdict["recommendation"] == "reuse_parent_seed_until_fitness_eligible"
    assert verdict["preferred_continuation_seed"] == str(source_seed)


def test_continuation_economic_verdict_is_unavailable_without_parent_metric() -> None:
    manifest = {
        "source_seed": "parent.pkl",
        "source_seed_imitation": {},
        "mt5": {"metrics": {"net_profit": 3.0, "fitness_eligible": True}},
        "candidate": "candidate.pkl",
    }

    result = candidate_runner.annotate_continuation_economic_verdict(manifest)

    assert result["continuation_economic_verdict"] == {
        "available": False,
        "reason": "parent_or_candidate_net_profit_missing",
    }


def test_continuation_economic_verdict_does_not_call_ineligible_gain_an_improvement(
    tmp_path,
) -> None:
    source_seed = tmp_path / "parent.pkl"
    manifest = {
        "source_seed": str(source_seed),
        "source_seed_imitation": {"previous_net_profit": -76.91},
        "mt5": {"metrics": {"net_profit": -47.11, "fitness_eligible": False}},
        "candidate": str(tmp_path / "candidate.pkl"),
    }

    result = candidate_runner.annotate_continuation_economic_verdict(manifest)

    verdict = result["continuation_economic_verdict"]
    assert verdict["net_profit_delta"] == pytest.approx(29.80)
    assert verdict["economic_regression"] is False
    assert verdict["economic_comparison"] == "inconclusive_ineligible"
    assert verdict["recommendation"] == "reuse_parent_seed_until_fitness_eligible"


def test_failure_diagnostics_are_copied_before_runner_cleanup(tmp_path) -> None:
    runner = tmp_path / "runner"
    output = tmp_path / "candidate"
    tester_log = runner / "Tester" / "Agent-0.0.0.0-2000" / "logs" / "tester.log"
    report = runner / "Reports" / "failed.htm"
    ignored = runner / "Tester" / "cache.bin"
    tester_log.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    ignored.parent.mkdir(parents=True, exist_ok=True)
    tester_log.write_text("tester stopped", encoding="utf-8")
    report.write_text("partial report", encoding="utf-8")
    ignored.write_bytes(b"cache")

    result = candidate_runner.preserve_runner_diagnostics(runner, output)

    assert result["file_count"] == 2
    assert (output / "mt5_failure_diagnostics" / tester_log.relative_to(runner)).is_file()
    assert (output / "mt5_failure_diagnostics" / report.relative_to(runner)).is_file()
    assert not (output / "mt5_failure_diagnostics" / ignored.relative_to(runner)).exists()


def test_seed_clone_uses_same_volume_hardlink_without_losing_source(tmp_path) -> None:
    source = tmp_path / "source.pkl"
    target = tmp_path / "candidate" / "source_seed_clone.pkl"
    target.parent.mkdir()
    source.write_bytes(b"immutable-seed")

    method = candidate_runner.clone_seed_artifact(source, target)

    assert method in {"hardlink", "copy2"}
    assert target.read_bytes() == source.read_bytes()
    if method == "hardlink":
        assert source.samefile(target)
    target.unlink()
    assert source.read_bytes() == b"immutable-seed"


def test_runner_clone_uses_complete_runtime_override_and_matching_build(tmp_path) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts",
        "MQL5/Include",
        "Bases/server",
        "Tester/bases/server/history",
        "Config",
        "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    (source / "terminal64.exe").write_bytes(b"old-terminal")
    (source / "metatester64.exe").write_bytes(b"old-tester")
    (source / "runtime_overrides" / "terminal64.exe").write_bytes(b"new-terminal")
    (source / "runtime_overrides" / "metatester64.exe").write_bytes(b"new-tester")
    (source / "runtime_overrides" / "build.txt").write_text("6140\n", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    (source / "MQL5/Include/self_growing_execution_contract.mqh").write_text(
        "#define TEST_INCLUDE 1\n", encoding="utf-8"
    )
    (source / "Bases/server/history.hcc").write_bytes(b"history")
    (source / "Tester/bases/server/history/2025.hcs").write_bytes(b"tester-history")
    (source / "Config/terminal.ini").write_text(
        "[LiveUpdate]\nLastBuildDataPath=5833\nLastBuild=5833\n"
        "LastBuildIDE=5833\nLastBuildTester=5833\n[Common]\nProgramBuild=5833\n",
        encoding="utf-16",
    )

    metadata = candidate_runner.clone_runner_candidate(source, target)

    assert (target / "terminal64.exe").read_bytes() == b"new-terminal"
    assert (target / "metatester64.exe").read_bytes() == b"new-tester"
    assert (target / "build.txt").read_text(encoding="utf-8") == "6140\n"
    assert (target / "MQL5/Include/self_growing_execution_contract.mqh").read_text(
        encoding="utf-8"
    ) == "#define TEST_INCLUDE 1\n"
    assert (target / "Tester/bases/server/history/2025.hcs").read_bytes() == b"tester-history"
    assert metadata["tester_market_data_copies"] == 1
    terminal_ini = (target / "Config/terminal.ini").read_text(encoding="utf-16")
    assert "LastBuild=6140" in terminal_ini
    assert "LastBuildTester=6140" in terminal_ini
    assert "ProgramBuild=6140" in terminal_ini
    assert metadata["runtime_override"] is True
    assert metadata["runtime_build"] == "6140"


def test_runner_clone_copies_base_file_when_hardlink_is_denied(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in ("MQL5/Experts", "Bases", "Config", "runtime_overrides"):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    (source / "Bases/alerts.dat").write_bytes(b"locked-but-readable")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")

    monkeypatch.setattr(candidate_runner.os, "link", lambda *_args: (_ for _ in ()).throw(PermissionError(5, "denied")))

    metadata = candidate_runner.clone_runner_candidate(source, target)

    assert (target / "Bases/alerts.dat").read_bytes() == b"locked-but-readable"
    assert metadata["base_hardlinks"] == 0
    assert metadata["base_copies"] == 1


def test_runner_clone_skips_exclusively_locked_alerts_only(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in ("MQL5/Experts", "Bases", "Config", "runtime_overrides"):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    locked = source / "Bases/alerts.dat"
    locked.write_bytes(b"locked")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")
    real_copy2 = candidate_runner.shutil.copy2
    monkeypatch.setattr(candidate_runner.os, "link", lambda *_args: (_ for _ in ()).throw(PermissionError(5, "denied")))
    monkeypatch.setattr(
        candidate_runner.shutil, "copy2",
        lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == locked else real_copy2(src, dst),
    )

    metadata = candidate_runner.clone_runner_candidate(source, target)

    assert not (target / "Bases/alerts.dat").exists()
    assert metadata["skipped_volatile_base_files"] == ["alerts.dat"]


def test_runner_clone_never_skips_locked_server_history(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in ("MQL5/Experts", "Bases/server/history", "Config", "runtime_overrides"):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    locked = source / "Bases/server/history/BTCUSDm.hcc"
    locked.write_bytes(b"history")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")
    real_copy2 = candidate_runner.shutil.copy2
    monkeypatch.setattr(candidate_runner.os, "link", lambda *_args: (_ for _ in ()).throw(PermissionError(5, "denied")))
    monkeypatch.setattr(
        candidate_runner.shutil, "copy2",
        lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == locked else real_copy2(src, dst),
    )

    with pytest.raises(PermissionError):
        candidate_runner.clone_runner_candidate(
            source, target, required_symbol="BTCUSDm",
        )


def test_runner_clone_excludes_unrelated_symbol_history_before_copy(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts", "Bases/server/history/BTCUSDm",
        "Bases/server/history/EURUSD", "Bases/server/ticks/BTCUSDm",
        "Bases/server/ticks/EURUSD", "Config", "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    wanted_history = source / "Bases/server/history/BTCUSDm/2025.hcc"
    wanted_ticks = source / "Bases/server/ticks/BTCUSDm/202509.tkc"
    unrelated_history = source / "Bases/server/history/EURUSD/2025.hcc"
    unrelated_ticks = source / "Bases/server/ticks/EURUSD/202509.tkc"
    wanted_history.write_bytes(b"btc-history")
    wanted_ticks.write_bytes(b"btc-ticks")
    unrelated_history.write_bytes(b"eur-history")
    unrelated_ticks.write_bytes(b"eur-ticks")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")

    real_link = candidate_runner.os.link
    real_copy2 = candidate_runner.shutil.copy2

    def guarded_link(src, dst):
        if Path(src) in {unrelated_history, unrelated_ticks}:
            raise PermissionError(13, "unrelated file must not be touched")
        return real_link(src, dst)

    def guarded_copy(src, dst, *args, **kwargs):
        if Path(src) in {unrelated_history, unrelated_ticks}:
            raise PermissionError(13, "unrelated file must not be touched")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(candidate_runner.os, "link", guarded_link)
    monkeypatch.setattr(candidate_runner.shutil, "copy2", guarded_copy)

    metadata = candidate_runner.clone_runner_candidate(
        source, target, required_symbol="BTCUSDm",
    )

    assert (target / "Bases/server/history/BTCUSDm/2025.hcc").read_bytes() == b"btc-history"
    assert (target / "Bases/server/ticks/BTCUSDm/202509.tkc").read_bytes() == b"btc-ticks"
    assert not (target / "Bases/server/history/EURUSD").exists()
    assert not (target / "Bases/server/ticks/EURUSD").exists()
    assert metadata["required_symbol"] == "BTCUSDm"
    assert metadata["excluded_symbol_trees"] == [
        "server/history/EURUSD", "server/ticks/EURUSD",
    ]


def test_runner_clone_excludes_terminal_communication_state(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts", "Bases/server/history/BTCUSDm", "Bases/server/Mail",
        "Bases/server/news", "Bases/server/subscriptions", "Bases/signals",
        "Bases/server/trades", "Config", "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    wanted = source / "Bases/server/history/BTCUSDm/2025.hcc"
    locked = source / "Bases/server/Mail/welcome.dat"
    wanted.write_bytes(b"history")
    locked.write_bytes(b"mail")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")
    real_link = candidate_runner.os.link
    real_copy2 = candidate_runner.shutil.copy2
    monkeypatch.setattr(
        candidate_runner.os, "link",
        lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == locked else real_link(src, dst),
    )
    monkeypatch.setattr(
        candidate_runner.shutil, "copy2",
        lambda src, dst, *args, **kwargs: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == locked else real_copy2(src, dst, *args, **kwargs),
    )

    metadata = candidate_runner.clone_runner_candidate(
        source, target, required_symbol="BTCUSDm",
    )

    assert (target / "Bases/server/history/BTCUSDm/2025.hcc").read_bytes() == b"history"
    assert not (target / "Bases/server/Mail").exists()
    assert metadata["excluded_non_tester_trees"] == [
        "server/Mail", "server/news", "server/subscriptions",
        "server/trades", "signals",
    ]


def test_runner_clone_selects_server_from_common_config(tmp_path) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts", "Bases/Exness-MT5Real8/history/BTCUSDm",
        "Bases/Default/Symbols", "Bases/Exness-MT5Real20/Symbols",
        "Config", "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    (source / "Bases/Exness-MT5Real8/history/BTCUSDm/2025.hcc").write_bytes(b"history")
    (source / "Bases/Default/Symbols/selected-0.dat").write_bytes(b"default")
    (source / "Bases/Exness-MT5Real20/Symbols/selected-0.dat").write_bytes(b"other")
    (source / "Config/common.ini").write_text(
        "[Common]\nServer=Exness-MT5Real8\n", encoding="utf-16",
    )
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")

    server = candidate_runner.detect_runner_server(source)
    metadata = candidate_runner.clone_runner_candidate(
        source, target, required_symbol="BTCUSDm", required_server=server,
    )

    assert server == "Exness-MT5Real8"
    assert (target / "Bases/Exness-MT5Real8/history/BTCUSDm/2025.hcc").is_file()
    assert not (target / "Bases/Default").exists()
    assert not (target / "Bases/Exness-MT5Real20").exists()
    assert metadata["required_server"] == "Exness-MT5Real8"
    assert metadata["excluded_server_trees"] == ["Default", "Exness-MT5Real20"]


def test_runner_clone_excludes_regenerable_symbol_cache(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts", "Bases/server/history/BTCUSDm/cache",
        "Config", "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    history = source / "Bases/server/history/BTCUSDm/2025.hcc"
    locked_cache = source / "Bases/server/history/BTCUSDm/cache/M15.hc"
    history.write_bytes(b"history")
    locked_cache.write_bytes(b"cache")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")
    real_link = candidate_runner.os.link
    real_copy2 = candidate_runner.shutil.copy2
    monkeypatch.setattr(
        candidate_runner.os, "link",
        lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == locked_cache else real_link(src, dst),
    )
    monkeypatch.setattr(
        candidate_runner.shutil, "copy2",
        lambda src, dst, *args, **kwargs: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == locked_cache else real_copy2(src, dst, *args, **kwargs),
    )

    metadata = candidate_runner.clone_runner_candidate(
        source, target, required_symbol="BTCUSDm",
    )

    assert (target / "Bases/server/history/BTCUSDm/2025.hcc").read_bytes() == b"history"
    assert not (target / "Bases/server/history/BTCUSDm/cache").exists()
    assert metadata["excluded_regenerable_cache_trees"] == [
        "server/history/BTCUSDm/cache",
    ]


def test_runner_clone_excludes_account_symbol_selection_file(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts", "Bases/server/symbols", "Config", "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    database = source / "Bases/server/symbols/symbols.raw"
    selection = source / "Bases/server/symbols/selected-123.dat"
    database.write_bytes(b"symbol-database")
    selection.write_bytes(b"ui-state")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")
    real_link = candidate_runner.os.link
    real_copy2 = candidate_runner.shutil.copy2
    monkeypatch.setattr(
        candidate_runner.os, "link",
        lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == selection else real_link(src, dst),
    )
    monkeypatch.setattr(
        candidate_runner.shutil, "copy2",
        lambda src, dst, *args, **kwargs: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == selection else real_copy2(src, dst, *args, **kwargs),
    )

    metadata = candidate_runner.clone_runner_candidate(source, target)

    assert (target / "Bases/server/symbols/symbols.raw").read_bytes() == b"symbol-database"
    assert not (target / "Bases/server/symbols/selected-123.dat").exists()
    assert metadata["excluded_symbol_selection_files"] == [
        "server/symbols/selected-123.dat",
    ]


def test_runner_clone_limits_market_files_to_test_window(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts", "Bases/server/history/BTCUSDm",
        "Bases/server/ticks/BTCUSDm", "Config", "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    in_year = source / "Bases/server/history/BTCUSDm/2025.hcc"
    old_year = source / "Bases/server/history/BTCUSDm/2023.hcc"
    warmup_ticks = source / "Bases/server/ticks/BTCUSDm/202501.tkc"
    in_ticks = source / "Bases/server/ticks/BTCUSDm/202502.tkc"
    end_ticks = source / "Bases/server/ticks/BTCUSDm/202503.tkc"
    old_ticks = source / "Bases/server/ticks/BTCUSDm/202303.tkc"
    for path in (in_year, old_year, warmup_ticks, in_ticks, end_ticks, old_ticks):
        path.write_bytes(path.name.encode())
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")
    real_link = candidate_runner.os.link
    real_copy2 = candidate_runner.shutil.copy2
    locked = {old_year, old_ticks}
    monkeypatch.setattr(
        candidate_runner.os, "link",
        lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) in locked else real_link(src, dst),
    )
    monkeypatch.setattr(
        candidate_runner.shutil, "copy2",
        lambda src, dst, *args, **kwargs: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) in locked else real_copy2(src, dst, *args, **kwargs),
    )

    metadata = candidate_runner.clone_runner_candidate(
        source, target, required_symbol="BTCUSDm",
        from_date="2025.02.01", to_date="2025.03.01",
    )

    assert (target / "Bases/server/history/BTCUSDm/2025.hcc").is_file()
    assert not (target / "Bases/server/history/BTCUSDm/2025.hcc").samefile(in_year)
    assert not (target / "Bases/server/ticks/BTCUSDm/202501.tkc").exists()
    assert (target / "Bases/server/ticks/BTCUSDm/202502.tkc").is_file()
    assert not (target / "Bases/server/ticks/BTCUSDm/202502.tkc").samefile(in_ticks)
    assert (target / "Bases/server/ticks/BTCUSDm/202503.tkc").is_file()
    assert not (target / "Bases/server/history/BTCUSDm/2023.hcc").exists()
    assert not (target / "Bases/server/ticks/BTCUSDm/202303.tkc").exists()
    assert metadata["market_data_window"] == ["2025.02.01", "2025.03.01"]
    assert metadata["isolated_market_data_copies"] == 3
    assert metadata["excluded_out_of_window_market_files"] == [
        "server/history/BTCUSDm/2023.hcc",
        "server/ticks/BTCUSDm/202303.tkc",
        "server/ticks/BTCUSDm/202501.tkc",
    ]


def test_runner_clone_defers_locked_required_tick_file_to_terminal_download(tmp_path, monkeypatch) -> None:
    source = tmp_path / "runner_source"
    target = tmp_path / "runner_clone"
    for folder in (
        "MQL5/Experts", "Bases/server/history/BTCUSDm",
        "Bases/server/ticks/BTCUSDm", "Config", "runtime_overrides",
    ):
        (source / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        (source / "runtime_overrides" / name).write_bytes(name.encode())
    (source / "runtime_overrides/build.txt").write_text("6182", encoding="utf-8")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").write_bytes(b"ex5")
    (source / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5").write_text("mq5", encoding="utf-8")
    history = source / "Bases/server/history/BTCUSDm/2025.hcc"
    tick_file = source / "Bases/server/ticks/BTCUSDm/202502.tkc"
    history.write_bytes(b"history")
    tick_file.write_bytes(b"ticks")
    (source / "Config/terminal.ini").write_text("ProgramBuild=5833\n", encoding="utf-16")
    real_link = candidate_runner.os.link
    real_copy2 = candidate_runner.shutil.copy2
    monkeypatch.setattr(
        candidate_runner.os, "link",
        lambda src, dst: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == tick_file else real_link(src, dst),
    )
    monkeypatch.setattr(
        candidate_runner.shutil, "copy2",
        lambda src, dst, *args, **kwargs: (_ for _ in ()).throw(PermissionError(13, "locked"))
        if Path(src) == tick_file else real_copy2(src, dst, *args, **kwargs),
    )

    metadata = candidate_runner.clone_runner_candidate(
        source, target, required_symbol="BTCUSDm",
        from_date="2025.02.01", to_date="2025.03.01",
    )

    assert (target / "Bases/server/history/BTCUSDm/2025.hcc").is_file()
    assert not (target / "Bases/server/ticks/BTCUSDm/202502.tkc").exists()
    assert metadata["deferred_locked_tick_files"] == [
        "server/ticks/BTCUSDm/202502.tkc",
    ]


def test_storage_monitor_does_not_kill_tester_for_seed_clone_bytes(tmp_path, monkeypatch) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runner"
    output.mkdir()
    runner.mkdir()
    seed = output / "source_seed_clone.pkl"
    seed.write_bytes(b"s" * 4096)
    (output / "candidate.partial.pkl").write_bytes(b"p" * 3072)
    monitor = candidate_runner.StorageMonitor(
        output,
        runner,
        quota=4096,
        reserve=1024,
        excluded_paths=(seed,),
    )

    class _TwoPollStop:
        calls = 0

        def wait(self, _seconds):
            self.calls += 1
            return self.calls > 1

    monitor.stop = _TwoPollStop()
    monkeypatch.setattr(
        candidate_runner.shutil,
        "disk_usage",
        lambda _anchor: SimpleNamespace(free=8192),
    )
    monkeypatch.setattr(
        monitor,
        "_terminate_candidate_tester",
        lambda: (_ for _ in ()).throw(AssertionError("khong duoc dung tester")),
    )

    monitor._run()

    assert monitor.failure == {}


def test_storage_monitor_reports_payload_and_protected_bytes_separately(tmp_path, monkeypatch) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runner"
    output.mkdir()
    runner.mkdir()
    seed = output / "source_seed_clone.pkl"
    seed.write_bytes(b"s" * 4096)
    (output / "candidate.partial.pkl").write_bytes(b"p" * 5120)
    monitor = candidate_runner.StorageMonitor(
        output,
        runner,
        quota=4096,
        reserve=1024,
        excluded_paths=(seed,),
    )

    class _OnePollStop:
        def wait(self, _seconds):
            return False

    monitor.stop = _OnePollStop()
    monkeypatch.setattr(
        candidate_runner.shutil,
        "disk_usage",
        lambda _anchor: SimpleNamespace(free=8192),
    )
    monkeypatch.setattr(monitor, "_terminate_candidate_tester", lambda: None)

    monitor._run()

    assert monitor.failure["schema_version"] == candidate_runner.STORAGE_GATE_SCHEMA_VERSION
    assert monitor.failure["candidate_total_bytes"] >= 9216
    assert monitor.failure["candidate_payload_bytes"] >= 5120
    assert monitor.failure["excluded_protected_bytes"] >= 4096
    assert monitor.failure["candidate_payload_quota_bytes"] == 4096


def test_v193_profile_chi_override_cac_co_che_five_hien_hanh(tmp_path) -> None:
    base = load_config("configs/he_chuyen_gia_latest_ui_v191.yaml")
    profile = yaml.safe_load(
        Path("configs/experiments/edge_discovery_v193_candidate.yaml")
        .read_text(encoding="utf-8")
    )
    config = candidate_runner.candidate_config(base, profile, tmp_path)
    learning = config.learning
    overrides = dict(profile["learning_overrides"])
    assert profile["isolation"]["candidate_output_quota_bytes"] == 12 * 1024**3
    assert profile["isolation"]["reserve_floor_bytes"] == int(1.5 * 1024**3)
    assert not any(name.startswith("direct_rnn_") for name in overrides)
    assert not any(name.startswith("tensor_rule_") for name in overrides)
    assert learning.five_behavior_uniform_mix == 0.15
    assert learning.five_gae_min_feedback_steps == 16
    assert learning.five_behavior_entry_probability_floor == 0.005
    assert learning.five_flat_entry_mass_floor_start == pytest.approx(0.15)
    assert learning.five_pattern_unroll == 4
    assert learning.five_pattern_action_quality_reward_scale_r == 0.10
    assert learning.five_entropy_start == 0.06
    assert learning.five_invalid_cancel_no_pending_penalty_r == 2.0
    assert learning.five_unfilled_pending_timeout_penalty_r == 0.25
    assert learning.five_entropy_edge_gate_enabled is True
    assert learning.five_entropy_edge_release_count == 3
    assert learning.five_entropy_edge_decay_count == 8
