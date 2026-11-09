import numpy as np
import pytest

from self_growing_trader.he_chuyen_gia_ngu_canh import BoNhanDienNguCanh, CauHinhHeChuyenGia
from self_growing_trader.market_regime_classifier import (
    MarketRegimeClassifier,
    REGIME_GIAM_MANH,
    REGIME_GIAM_YEU,
    REGIME_TANG_MANH,
    REGIME_TANG_YEU,
    REGIME_VOLATILE,
)
from self_growing_trader.input_normalization import normalize_input_vector
from self_growing_trader.protocol_contract import INPUT_COUNT


def _market(direction: float) -> np.ndarray:
    market = np.zeros(58, dtype=float)
    for index, scale in {
        0: 0.8,
        1: 0.9,
        2: 1.0,
        3: 1.2,
        4: 1.1,
        5: 1.0,
        11: 1.3,
        12: 0.7,
        17: 0.6,
        19: 0.5,
        22: 0.9,
        23: 1.0,
        26: 0.6,
        27: 0.7,
        28: 0.8,
        32: 1.0,
        36: 0.9,
        40: 0.8,
    }.items():
        market[index] = direction * scale
    market[14] = 0.35
    market[21] = 1.0
    market[45] = 0.35
    return market


def _wire_market(market: np.ndarray) -> np.ndarray:
    """Place the 58 market slots into the current 137-column EA wire."""

    wire = np.zeros(INPUT_COUNT, dtype=float)
    wire[:29] = market[:29]
    wire[77:106] = market[29:58]
    return wire


def test_regime_classifier_uses_directional_features_for_uptrend_and_downtrend() -> None:
    classifier = MarketRegimeClassifier()

    up = classifier.rule_based_label(_market(1.0))
    down = classifier.rule_based_label(_market(-1.0))

    assert up in {REGIME_TANG_MANH, REGIME_TANG_YEU}
    assert down in {REGIME_GIAM_MANH, REGIME_GIAM_YEU}


def test_m15_weak_direction_and_high_m15_volatility_route_explicitly() -> None:
    classifier = MarketRegimeClassifier()

    weak_up = _market(0.2)
    weak_up[36], weak_up[37], weak_up[38], weak_up[39], weak_up[42] = 2.50, 1.00, 3.9, 0.0, 7.8
    weak_down = _market(-0.2)
    weak_down[36], weak_down[37], weak_down[38], weak_down[39], weak_down[42] = -2.50, -1.00, 3.9, 0.0, 7.8
    neutral = weak_up.copy()
    for index in (36, 37, 39):
        neutral[index] = 0.0
    neutral[38], neutral[42] = 8.0, 8.0

    assert classifier.rule_based_label(weak_up) == REGIME_TANG_YEU
    assert classifier.rule_based_label(weak_down) == REGIME_GIAM_YEU
    assert classifier.rule_based_label(neutral) == REGIME_VOLATILE


def test_volatile_subregime_splits_shock_expansion_and_chop() -> None:
    classifier = MarketRegimeClassifier()
    neutral = np.zeros(58, dtype=float)

    spike = neutral.copy()
    spike[38], spike[42] = 3.9, 7.8
    spike[39] = 3.0
    assert classifier.volatile_subregime(spike) == "SPIKE"

    expansion = neutral.copy()
    expansion[38], expansion[42] = 8.0, 8.0
    assert classifier.volatile_subregime(expansion) == "EXPANSION"

    chop = neutral.copy()
    chop[38], chop[42] = 3.9, 7.8
    assert classifier.volatile_subregime(chop) == "CHOP"


def test_context_detector_extracts_current_ea_market_layout_without_position_contamination() -> None:
    detector = BoNhanDienNguCanh(CauHinhHeChuyenGia())
    observation = np.zeros(INPUT_COUNT, dtype=float)
    market = _market(-1.0)
    observation[:29] = market[:29]
    observation[77:106] = market[29:58]
    observation[29] = 1.0  # open-position flag; must not be treated as trend.
    observation[30] = 1.0

    context, _, _, _, reason = detector._ung_vien(observation)

    assert context.startswith("CTX_GIAM_")
    assert "regime_timeframe=M15" in reason


def test_context_detector_routes_from_raw_bridge_view_not_normalized_policy_view() -> None:
    """Native M15/H1 ratios must not be re-read after structure-aware V4."""

    market = _market(0.2)
    market[36], market[37], market[38], market[39], market[42] = 2.50, 1.00, 3.90, 0.0, 7.80
    raw_wire = _wire_market(market)
    normalized_policy_wire = normalize_input_vector(
        raw_wire,
        mode="structure_aware_v4",
        size=INPUT_COUNT,
    )

    expected = BoNhanDienNguCanh(CauHinhHeChuyenGia()).nhan_dien(raw_wire)
    actual = BoNhanDienNguCanh(CauHinhHeChuyenGia()).nhan_dien(
        normalized_policy_wire,
        raw_regime_observation=raw_wire,
    )

    assert actual.ma_context == expected.ma_context
    assert actual.regime_input_source == "bridge_raw"
    assert actual.regime_input_audit["raw_bridge"]["m15_over_h1_atr"] == pytest.approx(0.5)
    # V4 changes these policy-facing values; the router audit proves it did
    # not mistake the transformed pair for the native M15/H1 ATR pair.
    assert actual.regime_input_audit["policy_input"]["m15_over_h1_atr"] != pytest.approx(0.5)


def test_current_ea_layout_does_not_let_bad_macd_flip_clear_uptrend() -> None:
    classifier = MarketRegimeClassifier()
    market = np.asarray([
        -0.338623, 0.338945, 1.854600, 0.641108, 4.735553, 8.642537,
        0.0, -0.338623, 0.165359, 0.090181, 0.594162, 2.824133,
        0.162899, 0.885866, 0.850000, 0.467199, 0.243050, 0.334994,
        -1355.127374, 0.500209, 0.785313, 1.854600, 0.641108,
        0.569916, 0.693808, 0.153369, 1.547543, 3.757548,
        -0.100000, 0.100000, 0.737277, -0.675590, 5.288450,
        0.662801, 2.527823, 0.338623, 5.762097, 1.031766,
        3.613936, 2.193222, -1.288250, 1.012176, 8.020108,
        10.785587, 2.210005, 0.416279, 0.186376, 0.0, 0.0,
        0.005942, 0.0, 0.154167, 1.392886, 3.156175, 1.392886,
        10.811076, -1.547543, -3.757548,
    ], dtype=float)

    regime = classifier.rule_based_label(market)

    assert regime in {REGIME_TANG_MANH, REGIME_TANG_YEU}


def test_context_detector_applies_new_m15_label_without_candidate_count() -> None:
    detector = BoNhanDienNguCanh(CauHinhHeChuyenGia())

    down = np.zeros(117, dtype=float)
    down_market = _market(-1.0)
    down[:29] = down_market[:29]
    down[77:106] = down_market[29:58]
    up = np.zeros(117, dtype=float)
    up_market = _market(1.0)
    up[:29] = up_market[:29]
    up[77:106] = up_market[29:58]

    stable_down = detector.nhan_dien(down)
    switched_up = detector.nhan_dien(up)

    assert stable_down.ma_context.startswith("CTX_GIAM_")
    assert switched_up.ma_context.startswith("CTX_TANG_")
    assert switched_up.on_dinh is True
    assert switched_up.transition_active is False
    assert "confirm=" not in switched_up.ly_do


def test_context_detector_keeps_m15_regime_when_only_m1_slots_change() -> None:
    detector = BoNhanDienNguCanh(CauHinhHeChuyenGia())
    first = np.zeros(INPUT_COUNT, dtype=float)
    market = _market(1.0)
    market[36], market[37], market[38], market[39], market[42] = 2.5, 1.0, 3.9, 0.0, 7.8
    first[:29] = market[:29]
    first[77:106] = market[29:58]

    second = first.copy()
    second[0] = -5.0
    second[1] = -5.0
    second[77 + (49 - 29)] = 5.0
    # A different M1 ATR rescales all four legacy M15 wire fields. The
    # detector signature must remain invariant after its ratio projection.
    for market_index in (36, 37, 38, 39, 42):
        wire_index = market_index if market_index < 29 else 77 + (market_index - 29)
        second[wire_index] *= 2.0

    first_context = detector.nhan_dien(first)
    second_context = detector.nhan_dien(second)

    assert first_context.ma_context == second_context.ma_context
    assert detector._last_regime_signature == MarketRegimeClassifier.m15_signature(
        detector._market_features_from_observation(second)
    )


def test_native_mtf_classifier_uses_native_atr_geometry_without_m1_churn() -> None:
    classifier = MarketRegimeClassifier(mtf_scaling_mode="native_tf_atr_v2")
    market = np.zeros(58, dtype=float)
    # Native M15 EMA/gap values are already in M15 ATR units.  The two
    # volatility slots encode ATR_tf / ATR_m1 through a symmetric ratio.
    market[36], market[37], market[39] = 1.20, 0.60, 0.10
    market[38], market[42] = 0.80, 0.90
    first = classifier.m15_signature(market, mtf_scaling_mode=classifier.mtf_scaling_mode)

    # Same M15/H1 ATR relation under a different M1 ATR: 4/8.444 ~= 9/19.
    changed_m1_atr = market.copy()
    changed_m1_atr[38] = 0.60
    changed_m1_atr[42] = (8.4444444444 - 1.0) / (8.4444444444 + 1.0)
    second = classifier.m15_signature(
        changed_m1_atr,
        mtf_scaling_mode=classifier.mtf_scaling_mode,
    )

    assert first == second
    assert classifier.rule_based_label(market) == REGIME_TANG_MANH


def test_native_feature_contract_rebinds_router_and_timingnet() -> None:
    from types import SimpleNamespace

    router = BoNhanDienNguCanh(CauHinhHeChuyenGia())
    router.configure_feature_contract(SimpleNamespace(mtf_scaling_mode="native_tf_atr_v2"))
    router.configure_timing_network(SimpleNamespace(
        timing_network_enabled=True,
        timing_network_hidden_size=4,
        timing_network_embedding_size=4,
        timing_network_learning_rate=0.01,
        timing_network_bptt_window=4,
        timing_network_eligibility_decay=0.90,
        timing_network_gradient_clip=1.0,
    ))

    assert router._classifier.mtf_scaling_mode == "native_tf_atr_v2"
    assert router.timing_network is not None
    assert router.timing_network.mtf_scaling_mode == "native_tf_atr_v2"
