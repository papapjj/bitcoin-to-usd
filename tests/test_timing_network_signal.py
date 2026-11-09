from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from self_growing_trader.timing_network import TimingNetwork, TimingNetworkOutput
from self_growing_trader.he_chuyen_gia_ngu_canh import BoNhanDienNguCanh, CauHinhHeChuyenGia
from self_growing_trader.chuyen_gia_policy import (
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    tao_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config


def test_timing_network_signal_uses_quality_entry_and_uncertainty() -> None:
    output = TimingNetworkOutput(
        quality=0.8,
        direction=0.6,
        entry_probability=0.5,
        urgency=0.2,
        uncertainty=0.25,
        deceleration_probability=0.0,
        trend_score=0.0,
        deceleration_update_count=0,
        regime_transition_probability=0.0,
        regime_transition_probabilities=np.zeros(3),
        regime_transition_update_count=0,
        current_regime="RANGING",
        embedding=np.zeros(2),
        action_logits=np.zeros(12),
        features=np.zeros(4),
        sequence_tensor=np.zeros((1, 4)),
        sequence_valid_length=1,
        state_before=np.zeros(2),
        candidate=np.zeros(2),
        hidden=np.zeros(2),
        eligibility_in=np.zeros((2, 2, 4)),
        eligibility_rec=np.zeros((2, 2, 2)),
        eligibility_bias=np.zeros((2, 2)),
        eligibility_steps=1,
    )

    signal = output.signal_payload()

    assert signal["timing_quality"] == pytest.approx(0.8 * 0.5 * 0.75)
    assert signal["timing_direction"] == 1
    assert signal["timing_network_direction"] == 0.6


def test_timing_network_can_generate_a_signal_without_legacy_detector() -> None:
    network = TimingNetwork.seed(hidden_size=4, embedding_size=4, seed=7)

    output = network.forward(np.zeros(119, dtype=float))
    signal = output.signal_payload()

    assert 0.0 <= float(signal["timing_quality"]) <= 1.0
    assert signal["timing_direction"] == 0
    assert network.update_count == 0


def test_context_router_uses_timingnet_as_the_only_timing_source() -> None:
    router = BoNhanDienNguCanh(
        CauHinhHeChuyenGia(),
        timing_network=TimingNetwork.seed(seed=8),
    )

    result = router.nhan_dien(np.zeros(119, dtype=float))

    assert result.timing_source == "TimingNet"
    assert result.timing_network_update_count == 0
    assert not hasattr(router, "_timing")


def test_shared_router_timingnet_learns_confirmed_regime_transition_labels() -> None:
    router = BoNhanDienNguCanh(CauHinhHeChuyenGia())
    router.configure_timing_network(SimpleNamespace(
        timing_network_enabled=True,
        timing_network_hidden_size=4,
        timing_network_embedding_size=4,
        timing_network_learning_rate=0.01,
        timing_network_bptt_window=4,
        timing_network_eligibility_decay=0.90,
        timing_network_gradient_clip=1.0,
        timing_network_regime_transition_enabled=True,
        timing_network_regime_transition_horizons=(1,),
        timing_network_regime_transition_update_scale=1.0,
        timing_network_regime_transition_learning_rate=0.01,
        timing_network_regime_transition_encoder_scale=0.0,
        timing_network_regime_transition_l2=1e-4,
    ))
    router.set_timing_network_learning_enabled(True)

    first_input = np.zeros(137, dtype=float)
    second_input = first_input.copy()
    # A TimingNet label clock is an M15 signature, not request count.  Change
    # the closed-M15 EMA slot to advance one causal observation.
    second_input[77 + (36 - 29)] = 0.50
    first = router.nhan_dien(first_input)
    second = router.nhan_dien(second_input)

    assert first.timing_network_regime_transition_update_count == 0
    assert second.timing_network_regime_transition_update_count >= 1
    assert second.timing_network_regime_transition_calibration_count >= 1
    assert second.timing_network_regime_transition_brier <= 1.0


def test_timingnet_bootstrap_does_not_deadlock_strict_entry_gate() -> None:
    config = load_config("configs/he_chuyen_gia.yaml")
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=81)
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    policy = next(iter(system.experts.values())).policy
    policy._context_direction_entry_gate_enabled = True
    policy._context_expert_owner_metadata = {
        "context_task": "TANG_MANH",
        "timing_source": "TimingNet",
        "timing_network_update_count": 0,
        "timing_direction": 0,
        "timing_quality": 0.125,
    }

    mask, audit = policy._context_direction_entry_gate_mask(12)

    assert all(mask)
    assert audit["context_direction_entry_gate_reason"] == "timing_network_bootstrap"
    assert audit["context_direction_entry_gate_bootstrap"] is True


def test_mt5_runtime_profile_migrates_every_legacy_owner_to_pid_swats() -> None:
    """A seed without per-owner TimingNets must not retain legacy updates."""

    candidate = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    legacy = replace(
        candidate,
        learning=replace(
            candidate.learning,
            timing_network_enabled=False,
            timing_network_lifecycle_scale=0.0,
            timing_network_swats_enabled=False,
            timing_network_pid_enabled=False,
            timing_network_adamw_enabled=False,
            timing_network_cosine_annealing_enabled=False,
        ),
    )
    system = tao_he_chuyen_gia_mot_context_mot_policy(legacy, seed=810)
    assert all(expert.policy.timing_network is None for expert in system.experts.values())

    _cau_hinh_policy_chuyen_gia_cho_mt5(system, candidate)

    for expert in system.experts.values():
        # Cấu hình hiện tại dùng nhánh thời điểm nằm ngay trong mạng Five.
        # Không được dựng lại TimingNet độc lập của kiến trúc cũ.
        assert expert.policy.five_shared_network is not None
        assert expert.policy.timing_network is None
        assert expert.policy.timing_network_enabled is False


def test_mt5_runtime_profile_installs_adamw_cosine_on_every_direct_rnn() -> None:
    config = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=811)

    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)

    for expert in system.experts.values():
        policy = expert.policy
        assert policy.five_shared_network is not None
        assert policy.modules == []
        assert policy.direct_rnn_full_rl_enabled is False
        assert policy.timing_network is None
