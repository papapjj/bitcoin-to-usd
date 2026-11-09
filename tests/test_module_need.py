from __future__ import annotations

import pytest

from self_growing_trader.module_need import (
    ROLE_MEMORY_SPECIALIST,
    ROLE_POSITION_MANAGEMENT_SPECIALIST,
    ROLE_RISK_GATE_SPECIALIST,
    ROLE_PREDICTOR_SPECIALIST,
    ModuleErrorTrace,
    ModuleNeedSignal,
    ModuleRolePrior,
    ModuleSpec,
    cluster_module_errors,
    module_spec_for_role,
)


def test_module_need_scores_select_position_manager_for_giveback_cluster() -> None:
    signal = ModuleNeedSignal(
        near_tp_missed=1.0,
        position_giveback=0.9,
        late_close_loss=0.8,
        prediction_error=0.1,
        memory_match_quality=0.9,
    )

    scores = signal.scores()

    assert scores[ROLE_POSITION_MANAGEMENT_SPECIALIST] > scores[ROLE_MEMORY_SPECIALIST]
    assert scores[ROLE_POSITION_MANAGEMENT_SPECIALIST] > scores[ROLE_RISK_GATE_SPECIALIST]


def test_module_need_scores_select_memory_for_unknown_regime() -> None:
    signal = ModuleNeedSignal(
        novelty=0.95,
        memory_match_quality=0.05,
        repeated_unknown_regime=0.8,
        prediction_error=0.7,
    )

    scores = signal.scores()

    assert scores[ROLE_MEMORY_SPECIALIST] >= max(
        value for role, value in scores.items() if role != ROLE_MEMORY_SPECIALIST
    )


def test_module_spec_for_position_manager_targets_management_outputs() -> None:
    spec = module_spec_for_role(ROLE_POSITION_MANAGEMENT_SPECIALIST, reason="near_tp_missed_cluster")

    assert spec.initial_gate_bias < 0.0
    assert spec.initial_plasticity > 1.0
    assert "position_state" in spec.input_sources
    assert "trailing_gate" in spec.output_targets
    assert "partial_close_ratio" in spec.output_targets
    assert spec.validation_goal["reduce_giveback"] is True


def test_module_spec_rejects_bridge_schema_targets() -> None:
    with pytest.raises(ValueError, match="bridge schema"):
        ModuleSpec(
            role=ROLE_RISK_GATE_SPECIALIST,
            reason="bad_target",
            output_targets=["ea_bridge_schema"],
        )


def test_role_prior_feedback_and_cooldown() -> None:
    prior = ModuleRolePrior()
    before = prior.prior(ROLE_RISK_GATE_SPECIALIST)

    prior.record_outcome(ROLE_RISK_GATE_SPECIALIST, promoted=False, cluster_id="stress_entry")
    assert prior.prior(ROLE_RISK_GATE_SPECIALIST) < before
    assert prior.cooldown(ROLE_RISK_GATE_SPECIALIST, cluster_id="stress_entry") == 1

    degraded = prior.prior(ROLE_RISK_GATE_SPECIALIST)
    prior.record_outcome(ROLE_RISK_GATE_SPECIALIST, promoted=True, cluster_id="stress_entry")

    assert prior.prior(ROLE_RISK_GATE_SPECIALIST) > degraded
    assert prior.cooldown(ROLE_RISK_GATE_SPECIALIST, cluster_id="stress_entry") == 0


def test_cluster_module_errors_groups_lifecycle_and_prediction_failures() -> None:
    clusters = cluster_module_errors(
        [
            ModuleErrorTrace("a", regime="range", lifecycle_tags=("near_tp_missed",), reward=-0.2),
            ModuleErrorTrace("b", regime="range", lifecycle_tags=("gave_back_profit",), reward=-0.1),
            ModuleErrorTrace("c", regime="trend", account_stress=0.9, reward=-1.0),
            ModuleErrorTrace("d", regime="trend", prediction_error=0.95, reward=-0.3),
        ]
    )
    by_id = {cluster.cluster_id: cluster for cluster in clusters}

    assert "range:position_giveback" in by_id
    assert by_id["range:position_giveback"].signal.scores()[ROLE_POSITION_MANAGEMENT_SPECIALIST] > 0.0
    assert "trend:stress_entry_loss" in by_id
    assert by_id["trend:stress_entry_loss"].signal.scores()[ROLE_RISK_GATE_SPECIALIST] > 0.0
    assert "trend:high_prediction_error" in by_id
    assert by_id["trend:high_prediction_error"].signal.scores()[ROLE_PREDICTOR_SPECIALIST] > 0.0


def test_every_specialist_role_has_explicit_safe_module_spec() -> None:
    from self_growing_trader.module_need import VALID_MODULE_ROLES

    forbidden = {"ea_bridge_schema", "hard_risk_limit", "OrderSend", "raw_input"}
    for role in VALID_MODULE_ROLES:
        spec = module_spec_for_role(role, reason="test")
        assert spec.input_sources
        assert spec.output_targets
        assert not (set(spec.output_targets) & forbidden)
        assert spec.initial_gate_bias < 0.0
        assert spec.initial_plasticity > 0.0
