from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from self_growing_trader.five_shared_policy import (
    FivePPOTrainer,
    FivePublicProfile,
    FiveTrainingSchedule,
    SharedFiveTradingPolicy,
)
from self_growing_trader.pattern_discovery_reward import (
    PATH_REWARD_COMPONENTS,
    PatternDiscoveryRewardOracle,
)
from self_growing_trader.entry_preference import EntryPathBar, EntryPreferenceRecord


def _oracle(**overrides) -> PatternDiscoveryRewardOracle:
    values = dict(
        long_utility_atr=1.0,
        short_utility_atr=-1.0,
    )
    values.update(overrides)
    return PatternDiscoveryRewardOracle(**values)


def _rich_record(*, rising: bool = True) -> EntryPreferenceRecord:
    bars = tuple(
        EntryPathBar(
            high=(100.4 + 0.10 * i) if rising else (100.2 - 0.04 * i),
            low=(99.9 + 0.05 * i) if rising else (99.6 - 0.10 * i),
            close=(100.2 + 0.08 * i) if rising else (99.9 - 0.08 * i),
            spread=0.1,
        )
        for i in range(48)
    )
    return EntryPreferenceRecord(
        trace_id="rich", sequence=7, decision_time="2025.01.01", selected_action=0,
        bid=100.0, ask=100.1, atr=1.0, sl_atr=1.0, rr=3.0,
        entry_prices=(100.1, 100.0, 99.1, 101.0, 101.1, 99.0),
        bars=bars, observed_bars=48,
    )


def test_reward_duong_gia_chi_chay_khi_path_day_du() -> None:
    rewards = _oracle().rewards([1, 3, 2, 6, 0], [True] * 5)
    np.testing.assert_allclose(rewards, 0.0)


def test_reward_scalar_entry_bi_loai_bo_va_hold_bo_lo_edge_bi_phat() -> None:
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(_rich_record())
    rewards = oracle.rewards([1, 3, 2, 6, 0], [True] * 5)
    np.testing.assert_allclose(rewards[:4], 0.0)
    assert rewards[4] < 0.0
    np.testing.assert_allclose(oracle.action_path_reward_r, 0.0)


def test_shaping_quality_path_co_tin_hieu_duong_va_am_cho_entry_duoc_lay_mau() -> None:
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(
        _rich_record(), action_quality_reward_scale_r=0.10
    )
    q = np.asarray(oracle.action_q_values_r, dtype=float)
    best = int(np.argmax(q[1:])) + 1
    worst = int(np.argmin(q[1:])) + 1
    rewards = oracle.rewards([best, worst, 0, 0, 0], [True, True, False, False, False])
    assert rewards[0] > 0.0
    assert rewards[1] < 0.0
    np.testing.assert_allclose(rewards[2:], 0.0)
    assert oracle.reward_diagnostics()["action_quality_reward_scale_r"] == pytest.approx(0.10)


def test_shaping_quality_khong_phat_dong_loat_khi_path_that_su_khong_co_edge() -> None:
    oracle = replace(
        PatternDiscoveryRewardOracle.from_entry_preference_record(
            _rich_record(), action_quality_reward_scale_r=0.10
        ),
        action_q_values_r=(0.0, -0.30, -0.20, -0.40, -0.10, -0.25, -0.15),
    )
    rewards = oracle.rewards([1, 2, 3, 4, 5], [True] * 5)
    np.testing.assert_allclose(rewards, 0.0)


def test_hold_tai_khoan_trong_chi_duoc_thuong_rat_nho_khi_moi_entry_xau() -> None:
    oracle = replace(
        PatternDiscoveryRewardOracle.from_entry_preference_record(_rich_record(rising=False)),
        flat_account_anchor=True,
        hold_reward_components_r=(0.0, 0.01),
    )
    rewards = oracle.rewards([0, 1, 2, 0, 0], [True] * 5)
    assert rewards[0] == pytest.approx(0.01)
    np.testing.assert_allclose(rewards[1:3], 0.0)


def test_q_day_du_chon_hold_khi_moi_entry_deu_am() -> None:
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(_rich_record())
    oracle = replace(oracle, action_q_values_r=(0.0, -1.0, -0.6, -1.2, -1.0, -0.8, -0.9))
    targets = oracle.auxiliary_targets()
    q_values = np.asarray(targets["action_q_values_r"], dtype=np.float32)
    assert q_values.shape == (7,)
    assert int(np.argmax(q_values)) == 0


def test_pattern_reward_khong_cap_credit_cho_proposal_bi_arbitration_chan() -> None:
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(_rich_record())
    rewards = oracle.rewards([0, 0, 0, 0, 0], [True, False, False, True, False])
    assert rewards[0] < 0.0 and rewards[3] < 0.0
    np.testing.assert_allclose(rewards[[1, 2, 4]], 0.0)


def test_pattern_oracle_tru_spread_va_dung_mfe_mae_sau_khi_mine() -> None:
    sample = {
        "long_final_return_atr": 0.8,
        "long_mfe_atr": 1.5,
        "long_mae_atr": -0.4,
        "short_final_return_atr": -0.8,
        "short_mfe_atr": 0.4,
        "short_mae_atr": -1.5,
    }
    oracle = PatternDiscoveryRewardOracle.from_sample(
        sample,
        spread_cost_atr=0.2,
        tree_min_edge_atr=0.1,
        tree_mfe_weight=0.15,
        tree_mae_weight=0.25,
        tree_utility_clip_atr=3.0,
    )
    assert oracle.long_utility_atr == pytest.approx(0.725)
    assert oracle.short_utility_atr == pytest.approx(-1.315)
    targets = oracle.auxiliary_targets()
    np.testing.assert_allclose(targets["utility"], [0.725 / 3.0, -1.315 / 3.0])
    np.testing.assert_allclose(targets["excursion"], [0.5, 0.4 / 3.0, 0.4 / 3.0, 0.5])
    assert targets["no_edge"] == 0.0
    assert targets["direction_class"] == 1
    np.testing.assert_allclose(oracle.rewards([0, 1, 2, 3, 4], [True] * 5), 0.0)


def test_pattern_oracle_mt5_chi_dung_path_da_dong_va_bo_qua_future_sparse() -> None:
    record = EntryPreferenceRecord(
        trace_id="five:anchor:0",
        sequence=42,
        decision_time="2025.01.01 00:00",
        selected_action=1,
        bid=100.0,
        ask=100.2,
        atr=1.0,
        sl_atr=1.0,
        rr=2.0,
        entry_prices=(100.2, 100.0, 99.5, 100.7, 100.7, 99.5),
        bars=(
            EntryPathBar(high=101.1, low=99.6, close=100.5, spread=0.2),
            EntryPathBar(high=102.1, low=100.0, close=101.1, spread=0.2),
        ),
        future_closes=(1_000_000.0,),
        future_volatilities=(1_000_000.0,),
        observed_bars=600,
    )
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(
        record,
        tree_min_edge_atr=0.1,
        tree_mfe_weight=0.15,
        tree_mae_weight=0.25,
        tree_utility_clip_atr=3.0,
    )
    assert oracle.long_utility_atr == pytest.approx(0.975)
    assert oracle.short_utility_atr == pytest.approx(-1.625)
    assert oracle.long_mfe_atr == pytest.approx(2.0)
    assert oracle.short_mae_atr == pytest.approx(-2.0)
    targets = oracle.auxiliary_targets()
    assert targets["rich_path_available"] is True
    assert targets["horizon_returns"].shape == (5,)
    np.testing.assert_allclose(targets["horizon_mask"], np.zeros(5))
    assert targets["path_dynamics"].shape == (10,)
    assert targets["action_path"].shape == (6, 5)
    assert targets["action_outcome_classes"].shape == (6,)
    assert int(targets["action_outcome_classes"][0]) == 1


def test_pattern_oracle_phan_biet_market_limit_stop_va_thu_tu_sl_tp() -> None:
    record = EntryPreferenceRecord(
        trace_id="rich", sequence=7, decision_time="2025.01.01", selected_action=0,
        bid=100.0, ask=100.1, atr=1.0, sl_atr=1.0, rr=2.0,
        entry_prices=(100.1, 100.0, 99.1, 101.0, 101.1, 99.0),
        bars=tuple(
            EntryPathBar(high=100.8 + 0.1 * i, low=99.8, close=100.2 + 0.1 * i, spread=0.1)
            for i in range(48)
        ), observed_bars=48,
    )
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(
        record, tree_min_edge_atr=.1, tree_mfe_weight=.15,
        tree_mae_weight=.25, tree_utility_clip_atr=3.,
    )
    targets = oracle.auxiliary_targets()
    np.testing.assert_allclose(targets["horizon_mask"], np.ones(5))
    assert targets["action_path"][0, 0] == pytest.approx(1.0)
    assert targets["action_path"][2, 0] == pytest.approx(0.0)
    assert targets["action_outcome_classes"][0] in (1, 3)
    assert not np.allclose(targets["action_path"][0], targets["action_path"][2])
    diagnostics = oracle.reward_diagnostics()
    assert diagnostics["component_names"] == list(PATH_REWARD_COMPONENTS)
    assert "sl_truoc" not in diagnostics["component_names"]
    assert "mfe_bien_do" not in diagnostics["component_names"]
    assert oracle.action_path_reward_r[2] == pytest.approx(0.0)  # khong khop khong phat lan hai


def test_mae_mfe_chi_con_o_nhan_cay_va_xep_hang_khong_con_scalar_entry() -> None:
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(_rich_record())
    targets = oracle.auxiliary_targets()
    assert targets["action_path"].shape == (6, 5)
    assert np.any(targets["action_path"][:, 3] > 0.0)  # MFE label
    assert np.any(targets["action_path"][:, 4] > 0.0)  # MAE label
    assert np.ptp(targets["action_q_values_r"][1:]) > 0.0
    np.testing.assert_allclose(oracle.action_path_reward_r, 0.0)


def test_hold_reward_ton_trong_hai_muc_cau_hinh_moi() -> None:
    oracle = PatternDiscoveryRewardOracle.from_entry_preference_record(
        _rich_record(), flat_hold_missed_edge_penalty_r=0.02,
        flat_hold_confirmed_no_edge_bonus_r=0.003,
    )
    assert -0.02 - 1e-12 <= sum(oracle.hold_reward_components_r) <= 0.003 + 1e-12


def test_initial_sl_chi_day_xac_suat_sang_hold() -> None:
    logits = torch.zeros((1, 5, 12), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits[..., 1] = 3.0
        logits[..., 0] = 2.0
        logits[..., 11] = 4.0
    actions = torch.ones((1, 5), dtype=torch.long)
    masks = torch.zeros((1, 5), dtype=torch.bool)
    masks[0, 0] = True
    loss, count = FivePPOTrainer.initial_sl_conditional_entry_loss(
        logits, actions, masks, torch.ones(1)
    )
    loss.backward()
    assert count == 1
    assert logits.grad is not None
    assert logits.grad[0, 0, 1] > 0.0
    assert logits.grad[0, 0, 0] < 0.0
    torch.testing.assert_close(logits.grad[0, 0, 2:7], torch.zeros_like(logits.grad[0, 0, 2:7]))
    torch.testing.assert_close(logits.grad[..., 7:], torch.zeros_like(logits.grad[..., 7:]))


test_initial_sl_chi_day_xac_suat_sang_nam_entry_con_lai = test_initial_sl_chi_day_xac_suat_sang_hold


def test_san_policy_entry_tai_khoan_trong_tang_xac_suat_goc() -> None:
    torch.manual_seed(120)
    net = SharedFiveTradingPolicy(pre_lstm_size=24, hidden_size=12)
    with torch.no_grad():
        net.heads.action_query.weight.zero_()
        net.heads.action_query.bias.fill_(0.50)
        net.heads.action_embedding.weight.zero_()
        net.heads.action_embedding.weight[11].fill_(3.0)
    profile = FivePublicProfile(
        entropy_start=0.0, entropy_end=0.0,
        learning_rate_start=1e-2, learning_rate_end=1e-2,
    )
    schedule = FiveTrainingSchedule(
        profile=profile,
        entropy_edge_gate_enabled=True,
        verified_edge_count=0,
        flat_entry_mass_floor_start=0.75,
        flat_entry_mass_floor_end=0.0,
        flat_entry_mass_loss_weight=1.0,
        flat_entry_direction_mass_floor_start=0.25,
        flat_entry_direction_mass_floor_end=0.10,
        flat_entry_direction_mass_loss_weight=1.0,
    )
    trainer = FivePPOTrainer(net, schedule=schedule, value_loss_weight=0.0)
    observations = np.zeros((1, 5, 137), dtype=np.float32)
    initial = net.initial_memories()
    zeros = np.zeros((1, 5), dtype=np.int64)

    def raw_entry_mass() -> float:
        with torch.no_grad():
            evaluated = net.evaluate_sequence(
                torch.as_tensor(observations), initial,
                actions=torch.zeros((1, 5), dtype=torch.long),
                targets=torch.zeros((1, 5), dtype=torch.long),
                delay_indices=torch.zeros((1, 5), dtype=torch.long),
                risk_indices=torch.zeros((1, 5), dtype=torch.long),
                entry_offset_indices=torch.zeros((1, 5), dtype=torch.long),
                manage_mode_indices=torch.zeros((1, 5), dtype=torch.long),
                close_fraction_indices=torch.zeros((1, 5), dtype=torch.long),
            )
        return float(evaluated["policy_action_probabilities"][..., 1:7].sum(dim=-1).mean())

    before = raw_entry_mass()
    stats = trainer.optimize_sequence(
        observations=observations,
        initial_memories=initial,
        actions=zeros,
        targets=zeros,
        delay_indices=zeros,
        risk_indices=zeros,
        entry_offset_indices=zeros,
        manage_mode_indices=zeros,
        close_fraction_indices=zeros,
        old_log_probabilities=np.zeros((1, 5), dtype=np.float32),
        rewards=np.zeros((1, 5), dtype=np.float32),
        dones=np.ones((1, 5), dtype=np.float32),
        step_seconds=60.0,
        horizon_seconds=840.0,
        policy_loss_masks=np.zeros((1, 5), dtype=bool),
        sample_loss_masks=np.zeros((1, 5), dtype=bool),
        flat_account_masks=np.ones((1, 5), dtype=bool),
    )
    after = raw_entry_mass()
    assert before < 0.75
    assert after > before
    assert stats["flat_entry_mass_loss"] > 0.0
    assert stats["flat_entry_mass_samples"] == 5.0


def test_directional_entry_floor_moves_both_flat_owner_masses() -> None:
    torch.manual_seed(121)
    net = SharedFiveTradingPolicy(pre_lstm_size=24, hidden_size=12)
    with torch.no_grad():
        net.heads.action_query.weight.zero_()
        net.heads.action_query.bias.fill_(0.50)
        net.heads.action_embedding.weight.zero_()
        net.heads.action_embedding.weight[0].fill_(3.0)
    profile = FivePublicProfile(
        entropy_start=0.0, entropy_end=0.0,
        learning_rate_start=1e-2, learning_rate_end=1e-2,
        behavior_entry_probability_floor=0.0,
    )
    schedule = FiveTrainingSchedule(
        profile=profile,
        entropy_edge_gate_enabled=False,
        flat_entry_mass_floor_start=0.0,
        flat_entry_mass_floor_end=0.0,
        flat_entry_direction_mass_floor_start=0.25,
        flat_entry_direction_mass_floor_end=0.0,
        flat_entry_direction_mass_loss_weight=2.0,
    )
    trainer = FivePPOTrainer(net, schedule=schedule, value_loss_weight=0.0)
    observations = np.zeros((1, 5, 137), dtype=np.float32)
    initial = [net.initial_memories()]
    zeros = np.zeros((1, 5), dtype=np.int64)
    allowed = np.ones((1, 5, 12), dtype=bool)
    flat = np.ones((1, 5), dtype=bool)

    def owner_direction_masses() -> tuple[float, float]:
        with torch.no_grad():
            evaluated = net.evaluate_sequence(
                torch.as_tensor(observations), initial[0],
                actions=torch.zeros((1, 5), dtype=torch.long),
                targets=torch.zeros((1, 5), dtype=torch.long),
                delay_indices=torch.zeros((1, 5), dtype=torch.long),
                risk_indices=torch.zeros((1, 5), dtype=torch.long),
                entry_offset_indices=torch.zeros((1, 5), dtype=torch.long),
                manage_mode_indices=torch.zeros((1, 5), dtype=torch.long),
                close_fraction_indices=torch.zeros((1, 5), dtype=torch.long),
                allowed_action_masks=torch.as_tensor(allowed),
                episode_dones=torch.ones((1, 5), dtype=torch.bool),
            )
            probs = evaluated["policy_action_probabilities"]
            return (
                float(probs[0, 0, [1, 3, 5]].sum().item()),
                float(probs[0, 1, [2, 4, 6]].sum().item()),
            )

    before = owner_direction_masses()
    stats = trainer.optimize_independent_pattern_steps(
        observations=observations,
        initial_memories=initial,
        actions=zeros,
        targets=zeros,
        delay_indices=zeros,
        risk_indices=zeros,
        entry_offset_indices=zeros,
        manage_mode_indices=zeros,
        close_fraction_indices=zeros,
        old_log_probabilities=np.zeros((1, 5), dtype=np.float32),
        old_values=np.zeros((1, 5), dtype=np.float32),
        rewards=np.zeros((1, 5), dtype=np.float32),
        allowed_action_masks=allowed,
        policy_loss_masks=np.zeros((1, 5), dtype=bool),
        pattern_tree_enabled=np.zeros(1, dtype=bool),
        utility_targets=np.zeros((1, 2), dtype=np.float32),
        no_edge_targets=np.zeros(1, dtype=np.float32),
        flat_account_masks=flat,
    )
    after = owner_direction_masses()

    assert stats["flat_entry_direction_mass_samples"] == 2.0
    assert stats["flat_entry_direction_mass_loss"] > 0.0
    assert after[0] > before[0]
    assert after[1] > before[1]


def test_log_barrier_keeps_gradient_when_entry_mass_has_collapsed() -> None:
    raw_entry_mass = torch.full((1, 5), 1e-8, dtype=torch.float32, requires_grad=True)
    flat_account_mask = torch.ones((1, 5), dtype=torch.bool)
    square_loss, log_loss = FivePPOTrainer._flat_entry_mass_regularizers(
        raw_entry_mass, flat_account_mask, 0.35
    )
    assert square_loss.item() > 0.0
    assert log_loss.item() > 0.0
    log_loss.backward()
    assert torch.isfinite(raw_entry_mass.grad).all()
    assert torch.all(raw_entry_mass.grad < 0.0)


def test_pattern_batch_dung_state_anchor_doc_lap_va_khong_hoc_value_head() -> None:
    torch.manual_seed(12)
    net = SharedFiveTradingPolicy(pre_lstm_size=24, hidden_size=12)
    trainer = FivePPOTrainer(net)
    batch = 2
    observations = np.random.default_rng(12).normal(0.0, 0.05, (batch, 5, 137)).astype(np.float32)
    memories = [net.initial_memories() for _ in range(batch)]
    actions = np.asarray([[1, 2, 0, 1, 2], [2, 1, 0, 2, 1]], dtype=np.int64)
    zeros = np.zeros((batch, 5), dtype=np.int64)
    rewards = np.asarray([[0.1, -0.1, -0.05, 0.1, -0.1], [-0.1, 0.1, -0.05, -0.1, 0.1]], dtype=np.float32)
    action_before = net.heads.action_query.weight.detach().clone()
    value_before = net.heads.value.weight.detach().clone()
    tree_before = net.observation_encoder.pattern_tree.utility_head.weight.detach().clone()
    horizon_before = net.observation_encoder.pattern_tree.horizon_head.weight.detach().clone()
    stats = trainer.optimize_independent_pattern_steps(
        observations=observations,
        initial_memories=memories,
        actions=actions,
        targets=zeros,
        delay_indices=zeros,
        risk_indices=zeros,
        entry_offset_indices=zeros,
        manage_mode_indices=zeros,
        close_fraction_indices=zeros,
        old_log_probabilities=np.zeros((batch, 5), dtype=np.float32),
        old_values=np.zeros((batch, 5), dtype=np.float32),
        rewards=rewards,
        allowed_action_masks=np.ones((batch, 5, 12), dtype=bool),
        policy_loss_masks=np.ones((batch, 5), dtype=bool),
        flat_account_masks=np.ones((batch, 5), dtype=bool),
        pattern_tree_enabled=np.ones(batch, dtype=bool),
        utility_targets=np.asarray([[0.6, -0.4], [-0.5, 0.7]], dtype=np.float32),
        excursion_targets=np.asarray([
            [0.8, 0.1, 0.2, 0.7], [0.2, 0.6, 0.9, 0.1],
        ], dtype=np.float32),
        no_edge_targets=np.asarray([0.0, 1.0], dtype=np.float32),
        auxiliary_loss_weights=np.full(batch, 0.25, dtype=np.float32),
        utility_loss_weights=np.ones(batch, dtype=np.float32),
        excursion_loss_weights=np.full(batch, 0.5, dtype=np.float32),
        no_edge_loss_weights=np.full(batch, 0.5, dtype=np.float32),
        direction_classes=np.asarray([1, 2], dtype=np.int64),
        direction_loss_weights=np.ones(batch, dtype=np.float32),
        edge_hold_ceilings=np.full(batch, 0.20, dtype=np.float32),
        hold_ceiling_loss_weights=np.ones(batch, dtype=np.float32),
        horizon_targets=np.asarray([[.1,.2,.3,.4,.5],[-.1,-.2,-.3,-.4,-.5]], dtype=np.float32),
        horizon_masks=np.ones((batch, 5), dtype=np.float32),
        path_dynamics_targets=np.full((batch, 10), .25, dtype=np.float32),
        first_hit_classes=np.asarray([[2,1],[1,2]], dtype=np.int64),
        action_path_targets=np.full((batch, 6, 5), .30, dtype=np.float32),
        action_outcome_classes=np.asarray([[3,2,1,0,3,2],[2,3,0,1,2,3]], dtype=np.int64),
        horizon_loss_weights=np.full(batch, .5, dtype=np.float32),
        path_dynamics_loss_weights=np.full(batch, .5, dtype=np.float32),
        first_hit_loss_weights=np.full(batch, .5, dtype=np.float32),
        action_path_loss_weights=np.full(batch, .5, dtype=np.float32),
        action_outcome_loss_weights=np.full(batch, .5, dtype=np.float32),
    )
    assert stats["pattern_batch_size"] == 2.0
    assert stats["pattern_tree_supervised_anchors"] == 2.0
    assert stats["pattern_tree_auxiliary_loss"] > 0.0
    assert stats["pattern_tree_direction_loss"] > 0.0
    assert stats["pattern_tree_horizon_loss"] > 0.0
    assert stats["pattern_tree_action_outcome_loss"] > 0.0
    assert stats["flat_entry_mass_samples"] == 10.0
    assert stats["flat_entry_mass_loss"] > 0.0
    assert stats["flat_entry_direction_mass_samples"] == 4.0
    assert stats["flat_entry_direction_mass_loss"] >= 0.0
    assert not torch.equal(action_before, net.heads.action_query.weight)
    assert not torch.equal(tree_before, net.observation_encoder.pattern_tree.utility_head.weight)
    assert not torch.equal(horizon_before, net.observation_encoder.pattern_tree.horizon_head.weight)
    torch.testing.assert_close(value_before, net.heads.value.weight, rtol=0.0, atol=0.0)


def test_pattern_tree_du_doan_du_nhan_va_gioi_han_hop_le() -> None:
    torch.manual_seed(13)
    net = SharedFiveTradingPolicy(pre_lstm_size=24, hidden_size=12)
    predictions = net.observation_encoder.pattern_predictions(
        torch.randn(7, 137, dtype=torch.float32) * 0.05
    )
    assert predictions["utility"].shape == (7, 2)
    assert predictions["excursion"].shape == (7, 4)
    assert predictions["horizon_returns"].shape == (7, 5)
    assert predictions["path_dynamics"].shape == (7, 10)
    assert predictions["first_hit_logits"].shape == (7, 2, 3)
    assert predictions["action_path"].shape == (7, 6, 5)
    assert predictions["action_outcome_logits"].shape == (7, 6, 4)
    assert predictions["no_edge_logit"].shape == (7,)
    assert predictions["root"].shape[0] == 7
    assert bool(torch.all(torch.abs(predictions["utility"]) <= 1.0))
    assert bool(torch.all((predictions["excursion"] >= 0.0) & (predictions["excursion"] <= 1.0)))


def test_edge_direction_loss_keo_hold_xuong_du_khong_co_ppo_actor_credit() -> None:
    torch.manual_seed(14)
    net = SharedFiveTradingPolicy(pre_lstm_size=24, hidden_size=12)
    with torch.no_grad():
        net.heads.action_query.weight.zero_()
        net.heads.action_query.bias.fill_(0.50)
        net.heads.action_embedding.weight.zero_()
        net.heads.action_embedding.weight[0].fill_(1.0)
    profile = FivePublicProfile(entropy_start=0.0, entropy_end=0.0)
    trainer = FivePPOTrainer(net, schedule=FiveTrainingSchedule(profile=profile))
    observations = np.zeros((1, 5, 137), dtype=np.float32)
    memories = [net.initial_memories()]
    zeros = np.zeros((1, 5), dtype=np.int64)

    def hold_probability() -> float:
        with torch.no_grad():
            evaluated = net.evaluate_sequence(
                torch.as_tensor(observations), memories[0],
                actions=zeros_t, targets=zeros_t, delay_indices=zeros_t,
                risk_indices=zeros_t, entry_offset_indices=zeros_t,
                manage_mode_indices=zeros_t, close_fraction_indices=zeros_t,
                allowed_action_masks=torch.ones((1, 5, 12), dtype=torch.bool),
                episode_dones=torch.ones((1, 5), dtype=torch.bool),
            )
        return float(evaluated["primary_action_probabilities"][..., 0].mean().item())

    zeros_t = torch.zeros((1, 5), dtype=torch.long)
    before = hold_probability()
    stats = trainer.optimize_independent_pattern_steps(
        observations=observations,
        initial_memories=memories,
        actions=zeros,
        targets=zeros,
        delay_indices=zeros,
        risk_indices=zeros,
        entry_offset_indices=zeros,
        manage_mode_indices=zeros,
        close_fraction_indices=zeros,
        old_log_probabilities=np.zeros((1, 5), dtype=np.float32),
        old_values=np.zeros((1, 5), dtype=np.float32),
        rewards=np.zeros((1, 5), dtype=np.float32),
        allowed_action_masks=np.ones((1, 5, 12), dtype=bool),
        policy_loss_masks=np.zeros((1, 5), dtype=bool),
        pattern_tree_enabled=np.ones(1, dtype=bool),
        utility_targets=np.asarray([[0.8, -0.8]], dtype=np.float32),
        excursion_targets=np.zeros((1, 4), dtype=np.float32),
        no_edge_targets=np.zeros(1, dtype=np.float32),
        auxiliary_loss_weights=np.ones(1, dtype=np.float32),
        utility_loss_weights=np.zeros(1, dtype=np.float32),
        excursion_loss_weights=np.zeros(1, dtype=np.float32),
        no_edge_loss_weights=np.zeros(1, dtype=np.float32),
        direction_classes=np.ones(1, dtype=np.int64),
        direction_loss_weights=np.ones(1, dtype=np.float32),
        edge_hold_ceilings=np.full(1, 0.20, dtype=np.float32),
        hold_ceiling_loss_weights=np.ones(1, dtype=np.float32),
    )
    after = hold_probability()
    assert before > 0.90
    assert after < before
    assert stats["actor_eligible_samples"] == 0.0
    assert stats["pattern_tree_hold_ceiling_loss"] > 0.0
