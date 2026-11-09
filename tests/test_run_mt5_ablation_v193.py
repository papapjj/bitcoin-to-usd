from __future__ import annotations

from scripts.run_mt5_ablation_v193 import _frozen_positive_lcb_gate, _profile_for_arm


def test_ablation_profiles_change_only_intended_controls() -> None:
    base = {
        "experiment": {"id": "base"},
        "app_overrides": {"some_app_flag": True},
        "learning_overrides": {
            "five_reasoning_rl_enabled": True,
            "five_reasoning_action_scale": 0.0,
            "five_pattern_unroll": 4,
        },
    }

    frozen = _profile_for_arm(base, "frozen_policy")
    full = _profile_for_arm(base, "full_online")
    reasoning_off = _profile_for_arm(base, "reasoning_off")

    assert frozen["app_overrides"]["mt5_learning_enabled"] is False
    assert full["app_overrides"] == {"some_app_flag": True, "mt5_learning_enabled": True}
    assert reasoning_off["app_overrides"]["mt5_learning_enabled"] is True
    assert reasoning_off["learning_overrides"]["five_reasoning_rl_enabled"] is False
    assert reasoning_off["learning_overrides"]["five_reasoning_action_scale"] == 0.0
    assert reasoning_off["learning_overrides"]["five_pattern_unroll"] == 4
    assert base["experiment"] == {"id": "base"}  # source profile stays immutable
    assert {frozen["experiment"]["ablation_arm"], full["experiment"]["ablation_arm"],
            reasoning_off["experiment"]["ablation_arm"]} == {
        "frozen_policy", "full_online", "reasoning_off"
    }


def test_ppo_gate_requires_positive_frozen_lower_bound_and_support() -> None:
    diagnostics = {
        "performance": {
            "final_position_closes": 240,
            "position_expectancy_lcb_95_one_sided": 0.12,
        },
        "optimizer": {"ppo_updates": 0},
    }
    assert _frozen_positive_lcb_gate(diagnostics, minimum_positions=200)["passed"] is True
    diagnostics["performance"]["position_expectancy_lcb_95_one_sided"] = 0.0
    assert _frozen_positive_lcb_gate(diagnostics, minimum_positions=200)["passed"] is False
    diagnostics["performance"]["position_expectancy_lcb_95_one_sided"] = 0.12
    diagnostics["optimizer"]["ppo_updates"] = 1
    assert _frozen_positive_lcb_gate(diagnostics, minimum_positions=200)["passed"] is False
