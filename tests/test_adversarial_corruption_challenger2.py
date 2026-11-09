"""
tests/test_adversarial_corruption_challenger2.py

Adversarial Corruption and Fail-Closed Stress Test Suite for Production RegimeDetector.
Authored and executed by Challenger 2 for Milestone 4 (critic, specialist).

Adversarial Challenge Vectors:
1. Malformed / Corrupted Inputs & Fail-Closed Behavior:
   - NaN, Inf, negative/zero prices injected into update_bar().
   - Checks graceful fail-closed behavior: returns neutral prior [1/3, 1/3, 1/3] AND MarketRegime.CHOP (2).
   - Checks buffer poisoning prevention: corrupted bar must not corrupt subsequent valid bars.
   - Checks predict() and predict_proba() on NaN/Inf feature vectors.
   - Checks predict_wire_features() with corrupted 137-dim wire vectors.
2. Flat Market Attack:
   - 200 consecutive zero-variance flat bars (O == H == L == C == 90000.0, volume == 0.0).
   - Asserts no ZeroDivisionError, valid probabilities summing to 1.0 within 1e-6.
3. Extreme Flash Crashes / Spikes:
   - +/-50% instantaneous price jumps.
   - Asserts bounded probabilities in [0.0, 1.0] summing strictly to 1.0.
4. Bit-for-Bit Serialization Invariance:
   - Save and reload detector instance (.pkl).
   - Test across 1,000 synthetic inputs and assert bit-for-bit identical probabilities.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np
import pytest

from self_growing_trader.regime_detector import (
    DEFAULT_WARMUP_BARS,
    MarketRegime,
    NUM_REGIMES,
    RegimeDetector,
)

CHAMPION_PATH = Path("outputs") / "regime_models" / "champion_regime_detector.pkl"


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def default_detector() -> RegimeDetector:
    """Returns a freshly initialized default RegimeDetector."""
    detector = RegimeDetector()
    detector.reset()
    return detector


@pytest.fixture
def warmed_default_detector(default_detector: RegimeDetector) -> RegimeDetector:
    """Returns a default detector warmed up with 70 realistic M1 bars."""
    for i in range(70):
        default_detector.update_bar(
            open_=90000.0 + i,
            high=90050.0 + i,
            low=89950.0 + i,
            close=90010.0 + i,
            volume=300.0,
        )
    return default_detector


@pytest.fixture
def champion_detector() -> RegimeDetector:
    """Returns the loaded champion RegimeDetector."""
    if not CHAMPION_PATH.exists():
        pytest.skip(f"Champion artifact {CHAMPION_PATH} does not exist.")
    detector = RegimeDetector.load(CHAMPION_PATH)
    detector.reset()
    return detector


# ==============================================================================
# PILLAR 1: MALFORMED & CORRUPTED INPUTS (FAIL-CLOSED)
# ==============================================================================

class TestAdversarialCorruptionFailClosed:
    """Empirical adversarial verification of fail-closed contracts on corrupted inputs."""

    @pytest.mark.parametrize(
        "corrupt_bar",
        [
            (float("nan"), 90050.0, 89950.0, 90000.0, 300.0),
            (90000.0, float("nan"), 89950.0, 90000.0, 300.0),
            (90000.0, 90050.0, float("nan"), 90000.0, 300.0),
            (90000.0, 90050.0, 89950.0, float("nan"), 300.0),
            (float("nan"), float("nan"), float("nan"), float("nan"), 0.0),
        ],
        ids=["nan_open", "nan_high", "nan_low", "nan_close", "all_nan"],
    )
    def test_update_bar_nan_inputs_fail_closed_to_chop_and_neutral(
        self, warmed_default_detector: RegimeDetector, corrupt_bar: tuple[float, ...]
    ) -> None:
        """NaN bar injected into update_bar must fail closed to neutral prior [1/3, 1/3, 1/3] AND CHOP."""
        regime_idx, probs = warmed_default_detector.update_bar(*corrupt_bar)

        # Invariant 1: Neutral probabilities [1/3, 1/3, 1/3]
        np.testing.assert_allclose(
            probs,
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            atol=1e-5,
            err_msg="Corrupted NaN bar did not produce neutral prior probabilities!",
        )

        # Invariant 2: Fail closed to CHOP (regime=2), NOT BULL (regime=0)
        assert regime_idx == MarketRegime.CHOP, (
            f"Fail-closed contract violation: update_bar with NaN returned {regime_idx} "
            f"instead of MarketRegime.CHOP ({int(MarketRegime.CHOP)})"
        )

    @pytest.mark.parametrize(
        "inf_bar",
        [
            (float("inf"), 90050.0, 89950.0, 90000.0, 300.0),
            (90000.0, float("inf"), 89950.0, 90000.0, 300.0),
            (90000.0, 90050.0, float("inf"), 90000.0, 300.0),
            (90000.0, 90050.0, 89950.0, float("inf"), 300.0),
            (float("-inf"), 90050.0, 89950.0, 90000.0, 300.0),
            (90000.0, 90050.0, 89950.0, float("-inf"), 300.0),
        ],
        ids=["pos_inf_open", "pos_inf_high", "pos_inf_low", "pos_inf_close", "neg_inf_open", "neg_inf_close"],
    )
    def test_update_bar_inf_inputs_do_not_crash_and_fail_closed(
        self, warmed_default_detector: RegimeDetector, inf_bar: tuple[float, ...]
    ) -> None:
        """Inf bar injected into update_bar must not raise unhandled exception and must fail closed to CHOP."""
        regime_idx, probs = warmed_default_detector.update_bar(*inf_bar)

        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()
        np.testing.assert_allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-5)
        assert regime_idx == MarketRegime.CHOP, (
            f"Fail-closed violation on Inf bar: returned {regime_idx} instead of MarketRegime.CHOP"
        )

    def test_corrupted_bar_does_not_poison_buffer_for_subsequent_bars(
        self, warmed_default_detector: RegimeDetector
    ) -> None:
        """An invalid/corrupted bar must not poison internal sliding buffer for subsequent valid bars."""
        # Inject corrupted bar
        try:
            warmed_default_detector.update_bar(float("inf"), 90050.0, 89950.0, 90000.0, 300.0)
        except Exception:
            pass  # Even if it crashed, subsequent normal bar should recover or buffer must be protected

        # Next bar is completely normal
        regime_idx, probs = warmed_default_detector.update_bar(
            open_=90000.0, high=90050.0, low=89950.0, close=90010.0, volume=300.0
        )

        assert not np.isnan(probs).any(), "Subsequent valid bar produced NaN due to buffer poisoning!"
        assert not np.isinf(probs).any(), "Subsequent valid bar produced Inf due to buffer poisoning!"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    @pytest.mark.parametrize(
        "nonpositive_bar",
        [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (-100.0, -90.0, -110.0, -100.0, 10.0),
            (-50000.0, 90000.0, -60000.0, 85000.0, 100.0),
        ],
        ids=["all_zero_bar", "negative_prices_bar", "negative_open_low"],
    )
    def test_nonpositive_prices_fail_closed_to_chop_and_neutral(
        self, warmed_default_detector: RegimeDetector, nonpositive_bar: tuple[float, ...]
    ) -> None:
        """Non-positive (zero or negative) price inputs are corrupt and must fail closed to neutral and CHOP."""
        regime_idx, probs = warmed_default_detector.update_bar(*nonpositive_bar)

        np.testing.assert_allclose(
            probs,
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            atol=1e-5,
            err_msg="Non-positive prices did not fail closed to neutral prior!",
        )
        assert regime_idx == MarketRegime.CHOP, (
            f"Non-positive prices returned {regime_idx} instead of MarketRegime.CHOP"
        )

    def test_predict_discrete_on_nan_features_emits_chop(
        self, default_detector: RegimeDetector
    ) -> None:
        """predict() on NaN/Inf feature vector must return MarketRegime.CHOP (2), not BULL (0)."""
        nan_feats = np.full(15, np.nan)
        regime = default_detector.predict(nan_feats)
        assert regime == MarketRegime.CHOP, (
            f"predict(nan) returned {regime} (BULL) instead of MarketRegime.CHOP (2)!"
        )

    def test_predict_wire_features_fail_closed(
        self, default_detector: RegimeDetector
    ) -> None:
        """predict_wire_features() with corrupted inputs fails closed to CHOP and neutral prior."""
        # 1. NaN wire
        nan_wire = np.full(137, np.nan)
        r_nan, p_nan = default_detector.predict_wire_features(nan_wire)
        assert r_nan == MarketRegime.CHOP
        np.testing.assert_allclose(p_nan, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-6)

        # 2. Inf wire
        inf_wire = np.full(137, np.inf)
        r_inf, p_inf = default_detector.predict_wire_features(inf_wire)
        assert r_inf == MarketRegime.CHOP
        np.testing.assert_allclose(p_inf, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-6)

    @pytest.mark.parametrize(
        "inverted_bar",
        [
            (90000.0, 89900.0, 90100.0, 90000.0, 300.0),
            (90000.0, 85000.0, 86000.0, 90000.0, 300.0),
        ],
        ids=["high_lt_low", "inverted_extreme"],
    )
    def test_inverted_bars_fail_closed_to_chop_and_do_not_poison_buffer(
        self, warmed_default_detector: RegimeDetector, inverted_bar: tuple[float, ...]
    ) -> None:
        """Inverted bars (high < low) must immediately return CHOP with neutral prior and not grow buffer."""
        initial_len = len(warmed_default_detector.buffer)
        regime_idx, probs = warmed_default_detector.update_bar(*inverted_bar)

        assert regime_idx == MarketRegime.CHOP
        np.testing.assert_allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-5)
        assert len(warmed_default_detector.buffer) == initial_len, "Inverted bar poisoned the buffer!"

    @pytest.mark.parametrize(
        "corrupt_vol_bar",
        [
            (90000.0, 90050.0, 89950.0, 90000.0, -5.0),
            (90000.0, 90050.0, 89950.0, 90000.0, float("nan")),
            (90000.0, 90050.0, 89950.0, 90000.0, float("inf")),
            (90000.0, 90050.0, 89950.0, 90000.0, float("-inf")),
        ],
        ids=["neg_vol", "nan_vol", "pos_inf_vol", "neg_inf_vol"],
    )
    def test_corrupt_volumes_fail_closed_to_chop_and_do_not_poison_buffer(
        self, warmed_default_detector: RegimeDetector, corrupt_vol_bar: tuple[float, ...]
    ) -> None:
        """Corrupt volumes (negative, NaN, Inf) must fail closed to CHOP and not grow buffer."""
        initial_len = len(warmed_default_detector.buffer)
        regime_idx, probs = warmed_default_detector.update_bar(*corrupt_vol_bar)

        assert regime_idx == MarketRegime.CHOP
        np.testing.assert_allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-5)
        assert len(warmed_default_detector.buffer) == initial_len, "Corrupted volume poisoned the buffer!"

    @pytest.mark.parametrize(
        "invalid_type_bar",
        [
            ("corrupt_price", 90050.0, 89950.0, 90000.0, 300.0),
            (90000.0, None, 89950.0, 90000.0, 300.0),
            (90000.0, 90050.0, [89950.0], 90000.0, 300.0),
            (90000.0, 90050.0, 89950.0, 90000.0, "bad_vol"),
        ],
        ids=["string_open", "none_high", "list_low", "string_vol"],
    )
    def test_invalid_types_fail_closed_to_chop_and_do_not_poison_buffer(
        self, warmed_default_detector: RegimeDetector, invalid_type_bar: tuple[Any, ...]
    ) -> None:
        """Non-convertible types (strings, None, list) must fail closed to CHOP and not grow buffer."""
        initial_len = len(warmed_default_detector.buffer)
        regime_idx, probs = warmed_default_detector.update_bar(*invalid_type_bar)

        assert regime_idx == MarketRegime.CHOP
        np.testing.assert_allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-5)
        assert len(warmed_default_detector.buffer) == initial_len, "Invalid type poisoned the buffer!"

    def test_fail_closed_tie_break_verification_across_all_interfaces(
        self, warmed_default_detector: RegimeDetector
    ) -> None:
        """When predict_proba evaluates to uniform neutral prior [1/3, 1/3, 1/3], all interfaces
        must return MarketRegime.CHOP (2), NEVER MarketRegime.BULL (0)."""
        # Save original predict_proba
        orig_proba = warmed_default_detector.predict_proba
        try:
            # Force uniform neutral prior
            warmed_default_detector.predict_proba = lambda x: (
                np.full((len(x), NUM_REGIMES), 1.0 / NUM_REGIMES, dtype=np.float64)
                if getattr(x, "ndim", 1) == 2
                else np.full(NUM_REGIMES, 1.0 / NUM_REGIMES, dtype=np.float64)
            )

            # Interface 1: update_bar()
            r_bar, p_bar = warmed_default_detector.update_bar(90000.0, 90050.0, 89950.0, 90010.0, 300.0)
            assert r_bar == MarketRegime.CHOP, f"update_bar tie-break returned {r_bar} instead of CHOP!"

            # Interface 2: predict() 1D
            r_pred_1d = warmed_default_detector.predict(np.zeros(15))
            assert r_pred_1d == MarketRegime.CHOP, f"predict 1D tie-break returned {r_pred_1d} instead of CHOP!"

            # Interface 3: predict() 2D
            r_pred_2d = warmed_default_detector.predict(np.zeros((5, 15)))
            assert (r_pred_2d == int(MarketRegime.CHOP)).all(), f"predict 2D tie-break returned {r_pred_2d} instead of all CHOP!"

            # Interface 4: predict_wire_features() 137-dim
            r_wire, p_wire = warmed_default_detector.predict_wire_features(np.zeros(137))
            assert r_wire == MarketRegime.CHOP, f"predict_wire_features 137-dim tie-break returned {r_wire} instead of CHOP!"

            # Interface 5: predict_wire_features() 15-dim
            r_wire_15, p_wire_15 = warmed_default_detector.predict_wire_features(np.zeros(15))
            assert r_wire_15 == MarketRegime.CHOP, f"predict_wire_features 15-dim tie-break returned {r_wire_15} instead of CHOP!"
        finally:
            warmed_default_detector.predict_proba = orig_proba


# ==============================================================================
# PILLAR 2: FLAT MARKET ATTACK (200 ZERO-VARIANCE BARS)
# ==============================================================================

class TestAdversarialFlatMarketAttack:
    """Empirical verification of stability under 200 identical flat market bars."""

    def test_200_zero_variance_bars_default_detector(
        self, default_detector: RegimeDetector
    ) -> None:
        """200 identical zero-variance bars produce valid probabilities with no ZeroDivisionError."""
        for bar_idx in range(200):
            regime_idx, probs = default_detector.update_bar(
                open_=90000.0,
                high=90000.0,
                low=90000.0,
                close=90000.0,
                volume=0.0,
            )

            assert not np.isnan(probs).any(), f"Bar {bar_idx} produced NaN!"
            assert not np.isinf(probs).any(), f"Bar {bar_idx} produced Inf!"
            assert np.all(probs >= 0.0), f"Bar {bar_idx} produced negative probability!"
            assert np.all(probs <= 1.0), f"Bar {bar_idx} produced probability > 1.0!"
            assert np.isclose(np.sum(probs), 1.0, atol=1e-5), f"Bar {bar_idx} sum != 1.0!"

        # Final state must be CHOP with high confidence
        assert regime_idx == MarketRegime.CHOP

        # Verify spread_atr is strictly clamped to <= 50.0 (preventing 280,000 explosion)
        feat_default = default_detector.extract_bar_features()
        assert feat_default[11] <= 50.0, f"Default detector spread_atr exploded to {feat_default[11]}"

    def test_200_zero_variance_bars_champion_detector(
        self, champion_detector: RegimeDetector
    ) -> None:
        """Champion detector on 200 zero-variance bars produces valid, bounded probabilities."""
        for bar_idx in range(200):
            regime_idx, probs = champion_detector.update_bar(
                open_=90000.0,
                high=90000.0,
                low=90000.0,
                close=90000.0,
                volume=0.0,
            )

            assert not np.isnan(probs).any(), f"Champion bar {bar_idx} produced NaN!"
            assert not np.isinf(probs).any(), f"Champion bar {bar_idx} produced Inf!"
            assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
            assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

        # Verify spread_atr is strictly clamped to <= 50.0
        feat_champ = champion_detector.extract_bar_features()
        assert feat_champ[11] <= 50.0, f"Champion detector spread_atr exploded to {feat_champ[11]}"


# ==============================================================================
# PILLAR 3: EXTREME PRICE FLASH CRASHES / SPIKES (+/-50% JUMPS)
# ==============================================================================

class TestAdversarialExtremeFlashJumps:
    """Empirical verification of numerical bounding during extreme market dislocations."""

    def test_extreme_flash_crash_and_spike_50_percent(
        self, warmed_default_detector: RegimeDetector
    ) -> None:
        """Instantaneous +/-50% price dislocation maintains strictly bounded probabilities summing to 1.0."""
        # 1. Flash crash: 90,000 -> 45,000 (-50%)
        r_crash, p_crash = warmed_default_detector.update_bar(
            open_=90000.0,
            high=90000.0,
            low=44000.0,
            close=45000.0,
            volume=15000.0,
        )

        assert not np.isnan(p_crash).any(), "Flash crash produced NaN!"
        assert not np.isinf(p_crash).any(), "Flash crash produced Inf!"
        assert np.all(p_crash >= 0.0) and np.all(p_crash <= 1.0)
        assert np.isclose(np.sum(p_crash), 1.0, atol=1e-5)

        # 2. Flash spike: 45,000 -> 67,500 (+50%)
        r_spike, p_spike = warmed_default_detector.update_bar(
            open_=45000.0,
            high=68000.0,
            low=45000.0,
            close=67500.0,
            volume=12000.0,
        )

        assert not np.isnan(p_spike).any(), "Flash spike produced NaN!"
        assert not np.isinf(p_spike).any(), "Flash spike produced Inf!"
        assert np.all(p_spike >= 0.0) and np.all(p_spike <= 1.0)
        assert np.isclose(np.sum(p_spike), 1.0, atol=1e-5)

    def test_extreme_flash_crash_and_spike_champion_detector(
        self, champion_detector: RegimeDetector
    ) -> None:
        """Champion detector maintains probability axioms under +/-50% flash dislocations."""
        # Prime champion with 70 bars
        for i in range(70):
            champion_detector.update_bar(90000.0 + i, 90050.0 + i, 89950.0 + i, 90010.0 + i, 300.0)

        # Flash crash -50%
        r_crash, p_crash = champion_detector.update_bar(90000.0, 90000.0, 44000.0, 45000.0, 15000.0)
        assert not np.isnan(p_crash).any()
        assert not np.isinf(p_crash).any()
        assert np.all(p_crash >= 0.0) and np.all(p_crash <= 1.0)
        assert np.isclose(np.sum(p_crash), 1.0, atol=1e-5)

        # Flash spike +50%
        r_spike, p_spike = champion_detector.update_bar(45000.0, 68000.0, 45000.0, 67500.0, 12000.0)
        assert not np.isnan(p_spike).any()
        assert not np.isinf(p_spike).any()
        assert np.all(p_spike >= 0.0) and np.all(p_spike <= 1.0)
        assert np.isclose(np.sum(p_spike), 1.0, atol=1e-5)

    def test_extreme_flash_crash_80_percent_and_spike_100_percent(
        self, warmed_default_detector: RegimeDetector
    ) -> None:
        """Massive 80% market drop (90k -> 18k) and 100% bounce (18k -> 36k) maintain probability axioms."""
        # 80% crash
        r_crash, p_crash = warmed_default_detector.update_bar(90000.0, 90000.0, 17000.0, 18000.0, 50000.0)
        assert not np.isnan(p_crash).any()
        assert not np.isinf(p_crash).any()
        assert np.all(p_crash >= 0.0) and np.all(p_crash <= 1.0)
        assert np.isclose(np.sum(p_crash), 1.0, atol=1e-5)

        # 100% spike
        r_spike, p_spike = warmed_default_detector.update_bar(18000.0, 37000.0, 18000.0, 36000.0, 40000.0)
        assert not np.isnan(p_spike).any()
        assert not np.isinf(p_spike).any()
        assert np.all(p_spike >= 0.0) and np.all(p_spike <= 1.0)
        assert np.isclose(np.sum(p_spike), 1.0, atol=1e-5)


# ==============================================================================
# PILLAR 4: BIT-FOR-BIT SERIALIZATION TEST (1,000 INPUTS)
# ==============================================================================

class TestAdversarialSerializationInvariance:
    """Empirical verification of bit-for-bit serialization invariance over 1,000 inputs."""

    def test_default_detector_bit_for_bit_serialization_1000_inputs(
        self, default_detector: RegimeDetector
    ) -> None:
        """Default detector serialization preserves bit-for-bit identical probabilities across 1,000 inputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "default_detector.pkl"
            default_detector.save(save_path)
            loaded_detector = RegimeDetector.load(save_path)

            np.random.seed(42)
            X = np.random.randn(1000, 15)

            p_orig = default_detector.predict_proba(X)
            p_loaded = loaded_detector.predict_proba(X)

            np.testing.assert_array_equal(
                p_orig,
                p_loaded,
                err_msg="Default detector serialization failed bit-for-bit invariance across 1,000 inputs!",
            )

    def test_champion_detector_bit_for_bit_serialization_1000_inputs(
        self, champion_detector: RegimeDetector
    ) -> None:
        """Champion detector serialization preserves bit-for-bit identical probabilities across 1,000 inputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "champion_detector.pkl"
            champion_detector.save(save_path)
            loaded_detector = RegimeDetector.load(save_path)

            np.random.seed(1337)
            X = np.random.randn(1000, 15)

            p_orig = champion_detector.predict_proba(X)
            p_loaded = loaded_detector.predict_proba(X)

            np.testing.assert_array_equal(
                p_orig,
                p_loaded,
                err_msg="Champion detector serialization failed bit-for-bit invariance across 1,000 inputs!",
            )

    def test_wire_features_bit_for_bit_serialization_1000_inputs(
        self, champion_detector: RegimeDetector
    ) -> None:
        """MT5 wire hook predict_wire_features preserves exact bit-for-bit output across 1,000 wire inputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "champion_wire_detector.pkl"
            champion_detector.save(save_path)
            loaded_detector = RegimeDetector.load(save_path)

            np.random.seed(999)
            wire_inputs = np.random.randn(1000, 137)

            for i in range(1000):
                r1, p1 = champion_detector.predict_wire_features(wire_inputs[i])
                r2, p2 = loaded_detector.predict_wire_features(wire_inputs[i])

                assert r1 == r2, f"Wire input {i}: regime mismatch ({r1} != {r2})"
                np.testing.assert_array_equal(
                    p1, p2, err_msg=f"Wire input {i}: probability array not bit-for-bit equal!"
                )
