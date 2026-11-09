from pathlib import Path

import pytest
import yaml

from self_growing_trader.config import LearningIntegrityConfig, load_config


def test_five_training_schedule_covers_one_runtime_unroll() -> None:
    with pytest.raises(ValueError, match="at least 64"):
        LearningIntegrityConfig(five_training_schedule_steps=63)


def test_latest_ui_config_inherits_base_and_enables_v191_contract() -> None:
    config = load_config("configs/he_chuyen_gia_latest_ui_v191.yaml")

    assert config.model.hidden_size == 4096
    assert config.model.hidden_layers == 1
    assert config.model.recurrent_core == "openai_five_lstm_v1"
    assert config.he_chuyen_gia.che_do_quyet_dinh == "mot_context_mot_chuyen_gia"
    assert config.learning.direct_rnn_state_memory_enabled is False
    assert config.learning.tensor_rule_enabled is False
    assert config.learning.timing_network_enabled is False
    assert config.learning.direct_rnn_rl_auxiliary_enabled is False
    assert config.learning.direct_rnn_rl_entry_abstraction_horizon_bars == 600
    assert config.learning.train_entry_probe_enabled is False
    assert config.learning.direct_rnn_rl_exploration_enabled is True
    assert config.learning.direct_rnn_rl_exploration_rate == pytest.approx(0.10)
    assert config.learning.direct_rnn_rl_exploration_min_action_probability == pytest.approx(0.02)
    assert config.learning.direct_rnn_rl_entropy_weight == pytest.approx(0.002)


def test_latest_v193_config_uses_evidence_gated_edge_discovery() -> None:
    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    assert config.model.hidden_size == 4096
    assert config.model.hidden_layers == 1
    assert config.model.recurrent_core == "openai_five_lstm_v1"
    assert config.he_chuyen_gia.che_do_quyet_dinh == "nam_tac_nhan_dung_chung"
    assert config.he_chuyen_gia.so_tac_nhan == 5
    assert config.he_chuyen_gia.so_chieu_nhan_dang_tac_nhan == 5
    assert config.learning.tensor_rule_enabled is False
    assert config.learning.timing_network_enabled is False
    assert config.learning.direct_rnn_rl_auxiliary_enabled is False
    assert config.learning.five_training_schedule_steps == 10_000
    assert config.learning.five_reward_drawdown_weight == pytest.approx(0.30)
    assert config.learning.five_reward_risk_excess_weight == pytest.approx(0.10)
    assert config.learning.five_reward_conflict_weight == pytest.approx(0.05)
    assert config.learning.five_reward_competition_weight == pytest.approx(0.25)
    assert config.learning.five_reward_terminal_rank_weight == pytest.approx(0.50)
    assert config.learning.five_reward_expectancy_wr_v2_enabled is True
    assert config.learning.five_reward_wr_target == pytest.approx(0.55)
    assert config.learning.five_reward_wr_balance_weight == pytest.approx(0.05)
    assert config.learning.five_reward_expectancy_floor_r == pytest.approx(0.05)
    assert config.learning.five_reward_expectancy_shortfall_weight == pytest.approx(0.10)
    assert config.learning.five_reward_tail_loss_threshold_r == pytest.approx(1.0)
    assert config.learning.five_reward_tail_loss_weight == pytest.approx(0.10)
    assert config.learning.five_reward_tail_loss_cap_r == pytest.approx(3.0)
    assert config.learning.five_invalid_position_action_no_position_penalty_r == pytest.approx(1.0)
    assert config.learning.five_invalid_cancel_no_pending_penalty_r == pytest.approx(2.0)
    assert config.learning.five_unfilled_pending_timeout_penalty_r == pytest.approx(0.25)
    assert config.orders.pending_expiry_bars == 5
    assert config.learning.five_entropy_edge_gate_enabled is True
    assert config.learning.five_entropy_edge_release_count == 3
    assert config.learning.five_entropy_edge_decay_count == 8
    assert config.learning.five_entropy_start == pytest.approx(0.06)
    assert config.learning.five_behavior_uniform_mix == pytest.approx(0.15)
    assert config.learning.five_pattern_unroll == 64
    assert config.learning.five_pattern_action_quality_reward_scale_r == pytest.approx(0.0)
    assert config.learning.five_flat_entry_mass_floor_start == pytest.approx(0.75)
    assert config.learning.five_flat_entry_mass_floor_end == pytest.approx(0.0)
    assert config.learning.five_flat_entry_mass_loss_weight == pytest.approx(1.0)
    assert config.learning.five_lifecycle_initial_sl_local_loss_weight == pytest.approx(1.0)
    assert not hasattr(config.learning, "five_lifecycle_initial_sl_penalty_r")
    assert config.learning.five_pattern_reward_enabled is True
    assert config.learning.five_flat_hold_missed_edge_penalty_r == pytest.approx(0.25)
    assert config.learning.five_flat_hold_confirmed_no_edge_bonus_r == pytest.approx(0.01)
    assert config.learning.five_lifecycle_shape_clip_r == pytest.approx(0.75)
    assert config.learning.five_lifecycle_mfe_penalty_zero_r == pytest.approx(0.40)
    assert config.learning.five_lifecycle_mfe_penalty_010_r == pytest.approx(0.30)
    assert config.learning.five_lifecycle_mfe_penalty_025_r == pytest.approx(0.15)
    assert config.learning.five_lifecycle_mfe_penalty_neutral_at_r == pytest.approx(0.50)
    assert config.learning.five_lifecycle_trailing_activation_mfe_r == pytest.approx(0.50)
    assert config.learning.five_lifecycle_batch_size == 32
    assert config.learning.five_lifecycle_batch_max_wait == 64
    assert config.learning.five_position_improvement_reward_weight == pytest.approx(1.0)
    assert config.learning.five_pattern_tree_auxiliary_loss_weight == pytest.approx(0.25)
    assert config.learning.five_pattern_tree_min_edge_atr == pytest.approx(0.10)
    assert config.learning.five_pattern_tree_mfe_weight == pytest.approx(0.15)
    assert config.learning.five_pattern_tree_mae_weight == pytest.approx(0.25)
    assert config.learning.five_pattern_tree_utility_clip_atr == pytest.approx(3.0)
    assert config.learning.five_pattern_tree_utility_loss_weight == pytest.approx(1.0)
    assert config.learning.five_pattern_tree_excursion_loss_weight == pytest.approx(0.50)
    assert config.learning.five_pattern_tree_no_edge_loss_weight == pytest.approx(0.50)
    assert config.learning.five_pattern_tree_direction_loss_weight == pytest.approx(1.0)
    assert config.learning.five_pattern_tree_edge_hold_ceiling == pytest.approx(0.20)
    assert config.learning.five_pattern_tree_hold_ceiling_loss_weight == pytest.approx(1.0)
    assert config.learning.five_pattern_tree_horizon_loss_weight == pytest.approx(0.50)
    assert config.learning.five_pattern_tree_path_dynamics_loss_weight == pytest.approx(0.50)
    assert config.learning.five_pattern_tree_first_hit_loss_weight == pytest.approx(0.50)
    assert config.learning.five_pattern_tree_action_path_loss_weight == pytest.approx(0.50)
    assert config.learning.five_pattern_tree_action_outcome_loss_weight == pytest.approx(0.50)


def test_value_oos_profile_inherits_v193_and_matches_seven_choice_lifecycle() -> None:
    config = load_config(
        "configs/experiments/edge_discovery_v193_value_oos_fixed_rr2_low_disk.yaml"
    )

    assert config.model.hidden_size == 4096
    assert config.he_chuyen_gia.che_do_quyet_dinh == "nam_tac_nhan_dung_chung"
    assert config.orders.max_holding_bars == 48
    assert config.orders.pending_expiry_bars == 5
    assert config.orders.fixed_sl_atr == pytest.approx(1.5)
    assert config.orders.fixed_tp_atr == pytest.approx(3.0)
    assert config.orders.entry_offset_atr == pytest.approx(5.0)


def test_reasoning_rr2_candidate_profile_preserves_reasoning_and_matches_lifecycle() -> None:
    config = load_config(
        "configs/he_chuyen_gia_reasoning_rl_v1_candidate_rr2.yaml"
    )

    assert config.learning.five_reasoning_rl_enabled is True
    assert config.learning.five_reasoning_action_scale == pytest.approx(0.0)
    assert config.orders.max_holding_bars == 48
    assert config.orders.pending_expiry_bars == 5
    assert config.orders.fixed_sl_atr == pytest.approx(1.5)
    assert config.orders.fixed_tp_atr == pytest.approx(3.0)


def test_rlhf_reference_capture_profile_is_opt_in_and_candidate_only() -> None:
    config = load_config("configs/he_chuyen_gia_rlhf_reference_capture_candidate.yaml")
    assert config.learning.five_rlhf_reference_capture_enabled is True
    assert config.learning.five_rlhf_policy_update_enabled is False
    assert config.learning.five_rlhf_reward_model_path == ""
    assert config.learning.five_reasoning_rl_enabled is True
    assert config.learning.five_reasoning_action_scale == pytest.approx(0.0)


def test_pure_net_r_mt5_profile_disables_implicit_v4_lifecycle_shaping() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_direct_adaptive_owner_pure_net_r_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_reward_mode"] == "settlement_net_r_v1"
    assert overrides["five_reward_settlement_weight"] == pytest.approx(1.0)
    assert overrides["five_reward_mark_to_market_weight"] == pytest.approx(0.0)
    assert overrides["five_lifecycle_reward_enabled"] is False
    assert overrides["five_lifecycle_net_r_credit_enabled"] is True
    assert overrides["five_lifecycle_net_r_credit_weight"] == pytest.approx(1.0)
    assert overrides["five_pattern_reward_enabled"] is False
    assert overrides["five_entry_preference_supervision_enabled"] is False
    assert overrides["five_failure_replay_enabled"] is False


def test_net_r_selector_profile_separates_selector_from_v4_pattern_reward() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_direct_adaptive_owner_net_r_selector_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_entry_preference_supervision_enabled"] is True
    assert overrides["direct_rnn_rl_entry_preference_include_hold_snapshots"] is True


def test_net_r_selector_recovery_profile_increases_update_cadence_without_owner_suppression() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_direct_adaptive_owner_net_r_selector_recovery_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_pattern_unroll"] == 8
    assert overrides["five_live_owner_exploration_enabled"] is False
    assert overrides["five_reward_mode"] == "settlement_net_r_v1"
    assert overrides["five_pattern_reward_enabled"] is False
    assert overrides["five_entry_preference_supervision_enabled"] is True


def test_net_r_selector_balanced_profile_limits_exploration_without_starving_selector() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_direct_adaptive_owner_net_r_selector_balanced_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_pattern_unroll"] == 16
    assert overrides["five_behavior_uniform_mix"] == pytest.approx(0.05)
    assert overrides["five_live_owner_exploration_enabled"] is False
    assert overrides["five_entry_preference_supervision_enabled"] is True
    assert overrides["five_reward_mode"] == "settlement_net_r_v1"


def test_net_r_selector_mass_recovery_applies_flat_entry_curriculum_to_selector() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_direct_adaptive_owner_net_r_selector_mass_recovery_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_pattern_unroll"] == 8
    assert overrides["five_flat_entry_mass_floor_start"] == pytest.approx(0.25)
    assert overrides["five_flat_entry_mass_floor_end"] == pytest.approx(0.0)


def test_expected_value_causal_net_r_fast_veto_only_lowers_support_floor() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_veto_fast_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])
    assert overrides["five_expectancy_action_veto_enabled"] is True
    assert overrides["five_expectancy_action_veto_min_samples"] == 6
    assert overrides["five_expectancy_action_veto_min_confidence"] == 0.20
    assert overrides["five_expectancy_action_veto_min_lcb_r"] == -0.15
    assert overrides["five_expectancy_action_veto_hold_owner_on_negative"] is True
    assert overrides["five_flat_entry_mass_loss_weight"] == pytest.approx(1.0)
    assert overrides["five_live_owner_exploration_enabled"] is False
    assert overrides["five_entry_preference_supervision_enabled"] is True
    assert overrides["five_pattern_reward_enabled"] is False


def test_expected_value_causal_net_r_arm_only_changes_entry_decision_surface() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_action_decision_mode"] == "expected_value"


def test_expected_value_causal_net_r_veto_arm_requires_realized_negative_support() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_veto_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])
    assert overrides["five_action_decision_mode"] == "expected_value"
    assert overrides["five_expectancy_action_veto_enabled"] is True
    assert overrides["five_expectancy_action_veto_min_samples"] == 12
    assert overrides["five_expectancy_action_veto_min_confidence"] == 0.20
    assert overrides["five_expectancy_action_veto_min_lcb_r"] == -0.15
    assert overrides["five_expectancy_action_veto_hold_owner_on_negative"] is True
    assert overrides["five_reward_mode"] == "settlement_net_r_v1"
    assert overrides["five_lifecycle_net_r_credit_enabled"] is True
    assert overrides["five_live_owner_exploration_enabled"] is False
    assert overrides["five_flat_entry_mass_floor_start"] == pytest.approx(0.25)
    assert overrides["five_flat_entry_mass_floor_end"] == pytest.approx(0.0)


def test_expected_value_causal_net_r_live_veto_pools_only_current_window_actions() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])
    assert overrides["five_expectancy_action_veto_enabled"] is True
    assert overrides["five_live_action_veto_enabled"] is True
    assert overrides["five_live_action_veto_min_samples"] == 12
    assert overrides["five_live_action_veto_min_confidence"] == 0.20
    assert overrides["five_live_action_veto_min_lcb_r"] == -0.15
    assert overrides["five_live_action_veto_hold_owner_on_negative"] is True
    assert overrides["five_reward_mode"] == "settlement_net_r_v1"
    assert overrides["five_lifecycle_net_r_credit_enabled"] is True
    assert overrides["five_live_owner_exploration_enabled"] is False


def test_expected_value_causal_net_r_action_only_log_floor_wires_candidate_recovery_term() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_log_floor_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_flat_entry_mass_floor_start"] == pytest.approx(0.35)
    assert overrides["five_flat_entry_mass_floor_end"] == pytest.approx(0.10)
    assert overrides["five_flat_entry_mass_loss_weight"] == pytest.approx(1.0)
    assert overrides["five_flat_entry_mass_log_loss_weight"] == pytest.approx(0.10)
    assert overrides["five_expectancy_action_veto_hold_owner_on_negative"] is False
    assert overrides["five_live_action_veto_hold_owner_on_negative"] is False
    assert overrides["five_live_action_veto_probe_enabled"] is False


def test_expected_value_causal_net_r_behavior_compatible_followup_enables_seeded_frozen_sampling() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_log_floor_behavior_compatible_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_flat_entry_mass_log_loss_weight"] == pytest.approx(0.10)
    assert overrides["five_frozen_behavior_sampling_enabled"] is True
    assert overrides["five_frozen_behavior_sampling_seed"] == 20260702
    assert profile["app_overrides"]["mt5_learning_enabled"] is True


def test_expected_value_causal_net_r_behavior_compatible_directional_mass_is_a_single_low_pressure_delta() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_log_floor_behavior_compatible_directional_mass_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_flat_entry_mass_floor_start"] == pytest.approx(0.35)
    assert overrides["five_flat_entry_mass_floor_end"] == pytest.approx(0.10)
    assert overrides["five_flat_entry_mass_log_loss_weight"] == pytest.approx(0.10)
    assert overrides["five_flat_entry_direction_mass_floor_start"] == pytest.approx(0.06)
    assert overrides["five_flat_entry_direction_mass_floor_end"] == pytest.approx(0.02)
    assert overrides["five_flat_entry_direction_mass_loss_weight"] == pytest.approx(0.25)
    assert overrides["five_frozen_behavior_sampling_enabled"] is True
    assert overrides["five_frozen_behavior_sampling_seed"] == 20260702


def test_expected_value_causal_net_r_failure_replay_arm_enables_only_public_five_replay() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_log_floor_behavior_compatible_directional_mass_failure_replay_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_failure_replay_enabled"] is True
    assert overrides["five_failure_replay_items_per_batch"] == 8
    assert overrides["five_failure_replay_weight"] == pytest.approx(0.35)
    assert overrides["five_failure_replay_admit_threshold_r"] == pytest.approx(-0.25)
    assert overrides["five_mt5_failure_replay_enabled"] is False
    assert overrides["five_mt5_failure_replay_sessions_per_run"] == 0


def test_expected_value_causal_net_r_failure_replay_support2_arm_requires_repeated_bucket_evidence() -> None:
    profile = yaml.safe_load(Path(
        "configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_log_floor_behavior_compatible_directional_mass_failure_replay_support2_low_disk.yaml"
    ).read_text(encoding="utf-8"))
    overrides = dict(profile["learning_overrides"])

    assert overrides["five_failure_replay_enabled"] is True
    assert overrides["five_failure_replay_min_bucket_support"] == 2
    assert overrides["five_mt5_failure_replay_enabled"] is False
    assert profile["app_overrides"]["mt5_learning_enabled"] is True


def test_config_extends_resolves_relative_path_and_child_wins(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = child_dir / "child.yaml"
    parent.write_text("model:\n  hidden_size: 40\n  hidden_layers: 2\n", encoding="utf-8")
    child.write_text(
        "extends: ../parent.yaml\nmodel:\n  hidden_layers: 3\n",
        encoding="utf-8",
    )

    config = load_config(child)

    assert config.model.hidden_size == 40
    assert config.model.hidden_layers == 3


def test_config_extends_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config extends cycle"):
        load_config(first)
