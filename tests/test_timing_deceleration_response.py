from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from self_growing_trader.chuyen_gia_policy import tao_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.config import load_config
from self_growing_trader.he_chuyen_gia_ngu_canh import NhanDienNguCanh, NhiemVuNguCanh


def _transition_signal(
    *,
    probability: float,
    updates: int,
    active: bool = True,
    transition_probability: float = 0.0,
    transition_updates: int = 0,
    calibration_count: int = 0,
    brier: float = 1.0,
    high_support: int = 0,
    precision: float = 0.0,
) -> NhanDienNguCanh:
    return NhanDienNguCanh(
        ma_context="CTX_GIAM_MANH",
        ma_context_cha="CTX_GIAM_MANH",
        nhiem_vu=NhiemVuNguCanh.VAO_LENH,
        toa_do=(),
        do_tin_cay=0.80,
        on_dinh=not active,
        ly_do="test",
        candidate_context="CTX_GIAM_YEU",
        transition_active=active,
        transition_progress=0.5,
        timing_network_deceleration_probability=probability,
        timing_network_deceleration_update_count=updates,
        timing_network_trend_score=-0.30,
        timing_network_regime_transition_probability=transition_probability,
        timing_network_regime_transition_update_count=transition_updates,
        timing_network_regime_transition_calibration_count=calibration_count,
        timing_network_regime_transition_brier=brier,
        timing_network_regime_transition_high_confidence_support=high_support,
        timing_network_regime_transition_high_confidence_precision=precision,
    )


def test_high_deceleration_in_transition_converts_limit_to_same_side_stop_and_reduces_risk() -> None:
    config = load_config("configs/he_chuyen_gia.yaml")
    config = replace(
        config,
        he_chuyen_gia=replace(
            config.he_chuyen_gia,
            timing_deceleration_response_bat=True,
            timing_deceleration_entry_threshold=0.68,
            timing_deceleration_min_updates=24,
            timing_deceleration_min_risk_multiplier=0.35,
        ),
    )
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=901)
    outputs = np.zeros(20, dtype=float)
    outputs[4] = 0.90  # SELL_LIMIT
    outputs[12] = 1.00  # risk head
    trace = SimpleNamespace(selected_action=outputs, action_details={}, context_audit={})

    adjusted = system._apply_timing_deceleration_response(
        trace,
        _transition_signal(probability=0.90, updates=30),
    )

    assert int(np.argmax(adjusted.selected_action[:12])) == 6  # SELL_STOP
    assert adjusted.selected_action[12] < outputs[12]
    assert adjusted.action_details["timing_deceleration_entry_type_changed"] is True
    assert adjusted.action_details["timing_deceleration_risk_scaled"] is True
    assert adjusted.action_details["timing_deceleration_action_before"] == 4
    assert adjusted.action_details["timing_deceleration_action_after"] == 6


def test_deceleration_response_stays_inactive_without_transition_or_evidence() -> None:
    config = load_config("configs/he_chuyen_gia.yaml")
    config = replace(
        config,
        he_chuyen_gia=replace(config.he_chuyen_gia, timing_deceleration_response_bat=True),
    )
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=902)
    outputs = np.zeros(20, dtype=float)
    outputs[4] = 0.90
    outputs[12] = 1.00
    trace = SimpleNamespace(selected_action=outputs.copy(), action_details={}, context_audit={})

    adjusted = system._apply_timing_deceleration_response(
        trace,
        _transition_signal(probability=0.99, updates=999, active=False),
    )

    assert np.array_equal(adjusted.selected_action, outputs)
    assert adjusted.action_details["timing_deceleration_qualified"] is False


def test_calibrated_shared_timingnet_blocks_only_new_entry_before_transition() -> None:
    config = load_config("configs/he_chuyen_gia.yaml")
    config = replace(
        config,
        he_chuyen_gia=replace(
            config.he_chuyen_gia,
            timing_regime_transition_entry_block_bat=True,
            timing_regime_transition_entry_threshold=0.78,
            timing_regime_transition_min_updates=4,
            timing_regime_transition_min_calibration_samples=2,
            timing_regime_transition_min_high_confidence_samples=1,
            timing_regime_transition_min_precision=0.75,
            timing_regime_transition_max_brier=0.20,
        ),
    )
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=903)
    outputs = np.zeros(20, dtype=float)
    outputs[3] = 0.90  # BUY_LIMIT
    outputs[12] = 1.00
    trace = SimpleNamespace(selected_action=outputs, action_details={}, context_audit={})
    adjusted = system._apply_timing_regime_transition_entry_block(
        trace,
        _transition_signal(
            probability=0.0,
            updates=0,
            transition_probability=0.90,
            transition_updates=8,
            calibration_count=12,
            brier=0.08,
            high_support=4,
            precision=1.0,
        ),
    )
    assert int(np.argmax(adjusted.selected_action[:12])) == 0
    assert adjusted.action_details["timing_regime_transition_entry_blocked"] is True
    assert adjusted.action_details["timing_regime_transition_calibration_ready"] is True


def test_transition_entry_block_stays_open_when_high_probability_is_not_calibrated() -> None:
    config = load_config("configs/he_chuyen_gia.yaml")
    config = replace(
        config,
        he_chuyen_gia=replace(config.he_chuyen_gia, timing_regime_transition_entry_block_bat=True),
    )
    system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=904)
    outputs = np.zeros(20, dtype=float)
    outputs[3] = 0.90
    trace = SimpleNamespace(selected_action=outputs.copy(), action_details={}, context_audit={})

    adjusted = system._apply_timing_regime_transition_entry_block(
        trace,
        _transition_signal(
            probability=0.0,
            updates=0,
            transition_probability=0.99,
            transition_updates=999,
            calibration_count=999,
            brier=0.01,
            high_support=0,
            precision=0.0,
        ),
    )

    assert int(np.argmax(adjusted.selected_action[:12])) == 3
    assert adjusted.action_details["timing_regime_transition_entry_blocked"] is False
