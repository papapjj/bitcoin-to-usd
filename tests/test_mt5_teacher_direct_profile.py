from pathlib import Path

import yaml


PROFILE = Path(__file__).parents[1] / "configs" / "experiments" / "mt5_teacher_direct_retention_v1_low_disk.yaml"
ACTIVE_REASONING_PROFILE = (
    Path(__file__).parents[1]
    / "configs"
    / "experiments"
    / "reasoning_rl_v1_active_candidate.yaml"
)


def test_mt5_teacher_direct_profile_protects_teacher_action_path() -> None:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    learning = payload["learning_overrides"]

    assert learning["five_imitation_freeze_updates"] == 4096
    assert learning["five_imitation_anchor_updates"] == 16384
    assert learning["five_imitation_anchor_weight"] == 2.0
    assert learning["five_imitation_warmup_uniform_mix_max"] == 0.0
    assert learning["five_behavior_uniform_mix"] == 0.0
    assert learning["five_live_owner_exploration_min_mix"] == 0.0
    assert learning["five_entry_preference_supervision_enabled"] is True


def test_reasoning_candidate_credits_verified_flat_hold_counterfactuals() -> None:
    payload = yaml.safe_load(ACTIVE_REASONING_PROFILE.read_text(encoding="utf-8"))
    learning = payload["learning_overrides"]

    assert learning["five_action_ranking_bidirectional_enabled"] is True
    # Keep the two-entry qualification guard; enabling HOLD credit must not
    # turn a single noisy path into a ranking target.
    assert learning["five_action_ranking_min_qualified_entries"] == 2
