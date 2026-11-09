from __future__ import annotations

from scripts.run_mt5_transformer_ablation import (
    TEMPORAL_SCALE_KEY,
    _arm_profile,
    _arm_completion,
    _changed_state_keys,
    _paired_oos_gate,
    _require_schema10_seed,
    _validate_bounded_window,
)
import pytest
import json


def test_changed_state_keys_detects_only_temporal_scale() -> None:
    before = {"encoder.weight": "a", TEMPORAL_SCALE_KEY: "b"}
    after = {"encoder.weight": "a", TEMPORAL_SCALE_KEY: "c"}

    assert _changed_state_keys(before, after) == [TEMPORAL_SCALE_KEY]


def test_architecture_ablation_profile_forces_frozen_learning() -> None:
    base = {
        "experiment": {"id": "base"},
        "app_overrides": {"mt5_learning_enabled": True},
        "learning_overrides": {"five_mt5_failure_replay_enabled": False},
    }

    profile = _arm_profile(base, "v2_temporal_off")

    assert profile["experiment"]["architecture_ablation"] is True
    assert profile["experiment"]["ablation_arm"] == "v2_temporal_off"
    assert profile["app_overrides"]["mt5_learning_enabled"] is False
    assert base["app_overrides"]["mt5_learning_enabled"] is True


def test_fail_fast_window_rejects_long_run_without_explicit_override() -> None:
    assert _validate_bounded_window(
        "2025.05.01", "2025.05.02", max_span_days=1, allow_long_window=False
    ) == 1
    with pytest.raises(ValueError, match="rejects 7 days"):
        _validate_bounded_window(
            "2025.05.01", "2025.05.08", max_span_days=1, allow_long_window=False
        )
    assert _validate_bounded_window(
        "2025.05.01", "2025.05.08", max_span_days=1, allow_long_window=True
    ) == 7


def test_arm_completion_fails_closed_on_embedded_mt5_interruption() -> None:
    incomplete = {
        "mt5": {
            "policy": "",
            "interrupted": True,
            "metrics": {"incomplete": True, "interrupted_reason": "max_wall_sec"},
        }
    }
    assert _arm_completion(0, incomplete) == (False, "max_wall_sec")
    complete = {
        "mt5": {
            "policy": "candidate.pkl",
            "interrupted": False,
            "metrics": {"net_profit": 1.0},
        }
    }
    assert _arm_completion(0, complete) == (True, "completed_mt5_report")


def test_paired_temporal_oos_gate_requires_support_and_multi_metric_win() -> None:
    def metrics(net, pf, mean, lcb, dd):
        return {
            "net_profit": net,
            "profit_factor": pf,
            "equity_drawdown_pct": dd,
            "execution_summary": {"ledger_integrity": 1.0},
            "reconciliation": {"reconciled": True},
            "feature_audit": {"valid": True, "numerical_valid": True, "checked_rows": 80},
            "expectancy_audit": {"policy_expectancy": {
                "sample_count": 80,
                "mean_net_r": mean,
                "expectancy_lcb": lcb,
            }},
        }

    passed = _paired_oos_gate(
        metrics(20.0, 1.2, 0.10, 0.02, 2.0),
        metrics(10.0, 1.1, 0.05, 0.01, 2.1),
        minimum_lifecycles=50,
    )
    assert passed["passed"] is True

    failed = _paired_oos_gate(
        metrics(20.0, 1.2, 0.10, -0.03, 2.0),
        metrics(10.0, 1.1, 0.05, -0.01, 2.1),
        minimum_lifecycles=50,
    )
    assert failed["passed"] is False
    assert failed["checks"]["v3_policy_expectancy_lcb_not_worse"] is False

    on = metrics(20.0, 1.2, 0.10, 0.02, 2.0)
    off = metrics(10.0, 1.1, 0.05, 0.01, 2.1)
    off["feature_audit"]["numerical_valid"] = False
    assert not _paired_oos_gate(on, off, minimum_lifecycles=50)["passed"]
    off["feature_audit"]["numerical_valid"] = True
    off["expectancy_audit"]["policy_expectancy"]["sample_count"] = 49
    gate = _paired_oos_gate(on, off, minimum_lifecycles=1)
    assert gate["minimum_policy_lifecycles_per_arm"] == 50
    assert not gate["passed"]


def test_paired_preflight_rejects_missing_full_imitation_evidence(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="schema-10 full-validation/reload PASS"):
        _require_schema10_seed(tmp_path / "research_candidate.pkl", "source-hash")


@pytest.mark.parametrize("schema,passed,reload_passed,allowed", [
    (9, True, True, False), (10, False, True, False),
    (10, True, False, False), (10, True, True, True),
])
def test_paired_preflight_requires_current_validation_and_reload(
    tmp_path, schema, passed, reload_passed, allowed
) -> None:
    source = tmp_path / "seed.pkl"
    report = {
        "schema_version": schema,
        "candidate_checkpoint_sha256": "source-hash",
        "mt5_integration": {
            "entry_scope": "all_entry_mass_v2",
            "requires_runtime_teacher_retention": True,
            "representation_scope": "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3",
        },
        "verification": {"candidate_reload_passed": reload_passed},
        "training": {"validation_gate": {"enforced": True, "passed": passed}},
    }
    source.with_suffix(".pkl.imitation.json").write_text(json.dumps(report), encoding="utf-8")
    if allowed:
        assert _require_schema10_seed(source, "source-hash")["effective_for_mt5"]
    else:
        with pytest.raises(RuntimeError, match="schema-10 full-validation/reload PASS"):
            _require_schema10_seed(source, "source-hash")
    with pytest.raises(RuntimeError, match="khong khop checkpoint"):
        _require_schema10_seed(source, "wrong-hash")
