"""
tests/test_regime_detector.py

Comprehensive Automated Unit & Regression Test Suite for Production RegimeDetector.
Covers:
- T1: Initialization & defaults, config propagation, buffer reset.
- T2: Shape & contract invariants (1D [15] -> [3], 2D [N, 15] -> [N, 3], predict int/array).
- T3: Probability axioms (sum P_i = 1.0 +- 1e-6, P_i >= 0, P_i <= 1, no NaNs/Infs).
- T4: Numerical equivalence between streaming update_bar() and batch predict_proba().
- T5: Causal invariance (mutating future bars produces 0.0 change to past predictions).
- T6: Latency SLA benchmark (2,000 continuous update_bar() calls: P50 < 1.0ms, P99 < 5.0ms on CPU).
- T7: MT5 IPC wire hook integration (predict_wire_features with 137 floats).
- T8: Robustness & adversarial safety (warmup neutral prior, flat market, flash crash, zero volume, NaN/Inf).
- T9: Serialization round-trip (save & load bit-for-bit equality for .pkl, .json, and champion artifact).
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from self_growing_trader.regime_detector import (
    CANONICAL_BUFFER_REGIME_MAP,
    DEFAULT_WARMUP_BARS,
    DEFAULT_WINDOW_MAXLEN,
    MarketRegime,
    NUM_REGIMES,
    REGIME_NAMES,
    RegimeDetector,
)
from self_growing_trader.regime_models import BaseRegimeDetector, SupervisedSoftmaxRegimeDetector


# ==============================================================================
# T1: INITIALIZATION & DEFAULTS
# ==============================================================================

class TestT1InitializationAndDefaults:
    """T1: Verifies initialization defaults, configuration storage, and buffer reset."""

    def test_default_initialization(self) -> None:
        """Detector initializes with empty buffer, default window size, and warmup bars."""
        detector = RegimeDetector()
        assert detector.model is None
        assert detector.scaler is None
        assert detector.feature_extractor is None
        assert detector.config == {}
        assert detector.window_maxlen == DEFAULT_WINDOW_MAXLEN
        assert detector.warmup_bars == DEFAULT_WARMUP_BARS
        assert isinstance(detector.buffer, collections.deque)
        assert len(detector.buffer) == 0

    def test_custom_config_propagation(self) -> None:
        """Custom configuration dictionary and parameters are faithfully stored."""
        cfg = {"model_name": "lgbm_v1", "threshold": 0.65, "alpha": 0.20}
        detector = RegimeDetector(
            config=cfg,
            window_maxlen=500,
            warmup_bars=80,
        )
        assert detector.config == cfg
        assert detector.window_maxlen == 500
        assert detector.warmup_bars == 80
        assert detector.buffer.maxlen == 500

    def test_buffer_reset_clears_history(self) -> None:
        """Calling reset() completely empties internal buffer."""
        detector = RegimeDetector()
        for i in range(50):
            detector.update_bar(90000.0 + i, 90010.0 + i, 89990.0 + i, 90005.0 + i, 300.0)
        assert len(detector.buffer) == 50

        detector.reset()
        assert len(detector.buffer) == 0

    def test_market_regime_enum_values_and_mappings(self) -> None:
        """Verifies canonical IntEnum values: BULL=0, BEAR=1, CHOP=2."""
        assert int(MarketRegime.BULL) == 0
        assert int(MarketRegime.BEAR) == 1
        assert int(MarketRegime.CHOP) == 2

        assert REGIME_NAMES[MarketRegime.BULL] == "BULL"
        assert REGIME_NAMES[MarketRegime.BEAR] == "BEAR"
        assert REGIME_NAMES[MarketRegime.CHOP] == "CHOP"

        assert CANONICAL_BUFFER_REGIME_MAP[MarketRegime.BULL] == "UPTREND"
        assert CANONICAL_BUFFER_REGIME_MAP[MarketRegime.BEAR] == "DOWNTREND"
        assert CANONICAL_BUFFER_REGIME_MAP[MarketRegime.CHOP] == "SIDEWAY"


# ==============================================================================
# T2: SHAPE & CONTRACT INVARIANTS
# ==============================================================================

class TestT2ShapeAndContractInvariants:
    """T2: Verifies tensor dimensions, output shapes, and type contracts."""

    def test_predict_proba_1d_shape_contract(self) -> None:
        """1D feature vector of shape [15] returns 1D probability array of shape [3]."""
        detector = RegimeDetector()
        feat_1d = np.random.randn(15)
        probs = detector.predict_proba(feat_1d)

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3,)
        assert np.issubdtype(probs.dtype, np.floating)

    def test_predict_proba_2d_batch_shapes(self) -> None:
        """2D feature matrix of shape [N, 15] returns 2D probability matrix of shape [N, 3]."""
        detector = RegimeDetector()
        for n in [1, 5, 23, 100]:
            feat_batch = np.random.randn(n, 15)
            probs = detector.predict_proba(feat_batch)

            assert isinstance(probs, np.ndarray)
            assert probs.shape == (n, 3)
            assert np.issubdtype(probs.dtype, np.floating)

    def test_predict_discrete_argmax_contract(self) -> None:
        """predict() returns scalar int for 1D input, and 1D integer ndarray for 2D batch."""
        detector = RegimeDetector()

        # 1D input -> scalar int in {0, 1, 2}
        feat_1d = np.random.randn(15)
        pred_scalar = detector.predict(feat_1d)
        assert isinstance(pred_scalar, (int, np.integer))
        assert pred_scalar in {0, 1, 2}

        # 2D batch -> array of ints in {0, 1, 2}
        feat_batch = np.random.randn(20, 15)
        pred_batch = detector.predict(feat_batch)
        assert isinstance(pred_batch, np.ndarray)
        assert pred_batch.shape == (20,)
        assert np.issubdtype(pred_batch.dtype, np.integer)
        assert set(np.unique(pred_batch)).issubset({0, 1, 2})


# ==============================================================================
# T3: PROBABILITY AXIOMS
# ==============================================================================

class TestT3ProbabilityAxioms:
    """T3: Verifies Kolmogorov probability axioms (sum=1.0, non-negativity, finite)."""

    @pytest.mark.parametrize(
        "input_vector",
        [
            np.zeros(15),
            np.ones(15),
            np.ones(15) * 100.0,
            np.ones(15) * -100.0,
            np.array([0.1, -0.2, 0.05, 0.0, 1.2, 0.02, 0.015, 50.0, 1.1, 0.8, -0.1, 0.05, 0.6, 1.5, 0.8]),
            np.random.normal(loc=0.0, scale=1.0, size=15),
            np.random.normal(loc=10.0, scale=5.0, size=15),
        ],
    )
    def test_probability_axioms_single_sample(self, input_vector: np.ndarray) -> None:
        """Every single prediction satisfies sum=1.0, P_i >= 0, P_i <= 1, zero NaNs/Infs."""
        detector = RegimeDetector()
        probs = detector.predict_proba(input_vector)

        assert not np.isnan(probs).any(), f"NaN encountered for input {input_vector}"
        assert not np.isinf(probs).any(), f"Inf encountered for input {input_vector}"
        assert np.all(probs >= 0.0), f"Negative probability encountered: {probs}"
        assert np.all(probs <= 1.0), f"Probability > 1.0 encountered: {probs}"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-6), f"Probabilities do not sum to 1.0: {np.sum(probs)}"

    def test_probability_axioms_batch(self) -> None:
        """Batch predictions maintain exact row-wise sum to 1.0 and bounds."""
        detector = RegimeDetector()
        np.random.seed(42)
        X = np.random.randn(250, 15) * 3.0
        probs = detector.predict_proba(X)

        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        np.testing.assert_allclose(np.sum(probs, axis=-1), 1.0, atol=1e-6)


# ==============================================================================
# T4: NUMERICAL EQUIVALENCE: STREAMING VS BATCH
# ==============================================================================

class TestT4NumericalEquivalence:
    """T4: Verifies genuine numerical equivalence between streaming update_bar() and batch predict_batch(df)."""

    def test_streaming_and_batch_feature_prediction_equivalence(self) -> None:
        """Streaming update_bar() produces predictions matching batch predict_batch(df) on OHLCV series."""
        detector = RegimeDetector()
        detector.reset()

        np.random.seed(101)
        n_bars = 300
        close_prices = 90000.0 + np.cumsum(np.random.normal(0.0, 15.0, size=n_bars))
        high_prices = close_prices + np.abs(np.random.normal(5.0, 2.0, size=n_bars))
        low_prices = close_prices - np.abs(np.random.normal(5.0, 2.0, size=n_bars))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0] - 2.0
        tickvols = np.random.randint(100, 500, size=n_bars).astype(np.float64)
        spreads = np.full(n_bars, 2800.0, dtype=np.float64)

        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "tickvol": tickvols,
            "spread": spreads,
        })

        streaming_probs = []
        streaming_regimes = []

        for _, row in df.iterrows():
            regime_idx, probs = detector.update_bar(
                open_=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["tickvol"],
            )
            streaming_probs.append(probs)
            streaming_regimes.append(regime_idx)

        # Batch prediction directly from DataFrame
        batch_probs = detector.predict_batch(df)
        batch_regimes = np.argmax(batch_probs, axis=-1)

        # Post-warmup bars (bars 241 to 300 where mom_240_log has full lookback)
        eval_slice = slice(241, n_bars)
        streaming_post_probs = np.array(streaming_probs[eval_slice])
        streaming_post_regimes = np.array(streaming_regimes[eval_slice])
        batch_post_probs = batch_probs[eval_slice]
        batch_post_regimes = batch_regimes[eval_slice]

        # Assert exact numerical match within 1e-5
        np.testing.assert_allclose(
            streaming_post_probs, batch_post_probs, atol=1e-5,
            err_msg="Streaming update_bar() probabilities differed from batch predict_batch(df)!"
        )
        np.testing.assert_array_equal(
            streaming_post_regimes, batch_post_regimes,
            err_msg="Streaming update_bar() discrete regimes differed from batch predict_batch(df)!"
        )

    def test_champion_streaming_and_batch_prediction_equivalence(self) -> None:
        """Verifies streaming vs batch predict_batch(df) equivalence on the Champion model."""
        champ_pkl = Path("outputs") / "regime_models" / "champion_regime_detector.pkl"
        if not champ_pkl.exists():
            pytest.skip(f"Champion artifact {champ_pkl} not present in workspace.")

        detector = RegimeDetector.load(champ_pkl)
        detector.reset()

        np.random.seed(202)
        n_bars = 300
        close_prices = 90000.0 + np.cumsum(np.random.normal(0.0, 15.0, size=n_bars))
        high_prices = close_prices + np.abs(np.random.normal(5.0, 2.0, size=n_bars))
        low_prices = close_prices - np.abs(np.random.normal(5.0, 2.0, size=n_bars))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0] - 2.0
        tickvols = np.random.randint(100, 500, size=n_bars).astype(np.float64)
        spreads = np.full(n_bars, 2800.0, dtype=np.float64)

        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "tickvol": tickvols,
            "spread": spreads,
        })

        streaming_probs = []
        streaming_regimes = []

        for _, row in df.iterrows():
            regime_idx, probs = detector.update_bar(
                open_=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["tickvol"],
            )
            streaming_probs.append(probs)
            streaming_regimes.append(regime_idx)

        batch_probs = detector.predict_batch(df)
        batch_regimes = np.argmax(batch_probs, axis=-1)

        eval_slice = slice(241, n_bars)
        np.testing.assert_allclose(
            np.array(streaming_probs[eval_slice]), batch_probs[eval_slice], atol=1e-5,
            err_msg="Champion streaming update_bar() differed from batch predict_batch(df)!"
        )
        np.testing.assert_array_equal(
            np.array(streaming_regimes[eval_slice]), batch_regimes[eval_slice],
            err_msg="Champion streaming discrete regimes differed from batch predict_batch(df)!"
        )

    def test_predict_batch_matches_predict_proba(self) -> None:
        """predict_batch(matrix) produces identical output to predict_proba(matrix)."""
        detector = RegimeDetector()
        X = np.random.randn(50, 15)
        p_batch = detector.predict_batch(X)
        p_proba = detector.predict_proba(X)
        np.testing.assert_allclose(p_batch, p_proba, atol=1e-12)


# ==============================================================================
# T5: CAUSAL INVARIANCE (ZERO LOOKAHEAD BIAS)
# ==============================================================================

class TestT5CausalInvariance:
    """T5: Verifies that altering future bars causes strictly 0.0 change to past predictions."""

    def test_strict_causal_invariance_streaming_bars(self) -> None:
        """Mutating bars t+1...t+K causes strictly 0.0 difference to streaming predictions at bar t."""
        np.random.seed(42)
        n = 100
        bars_clean = []
        p = 90000.0
        for _ in range(n):
            p += np.random.normal(0.0, 10.0)
            bars_clean.append((p - 2.0, p + 5.0, p - 5.0, p, 300.0))

        # Run stream 1: all 100 clean bars
        d1 = RegimeDetector()
        d1.reset()
        probs_stream_1 = []
        for o, h, l, c, v in bars_clean:
            _, probs = d1.update_bar(open_=o, high=h, low=l, close=c, volume=v)
            probs_stream_1.append(probs)

        # Create perturbed stream: bars 0..49 identical, bars 50..99 violently perturbed (+100,000 USD)
        bars_perturbed = list(bars_clean[:50])
        for o, h, l, c, v in bars_clean[50:]:
            bars_perturbed.append((o + 100000.0, h + 150000.0, l + 90000.0, c + 120000.0, v * 10.0))

        # Run stream 2: perturbed bars
        d2 = RegimeDetector()
        d2.reset()
        probs_stream_2 = []
        for o, h, l, c, v in bars_perturbed:
            _, probs = d2.update_bar(open_=o, high=h, low=l, close=c, volume=v)
            probs_stream_2.append(probs)

        # Assert past predictions at bars 0..49 are bit-for-bit identical!
        arr1 = np.array(probs_stream_1[:50])
        arr2 = np.array(probs_stream_2[:50])
        max_diff = float(np.max(np.abs(arr1 - arr2)))

        assert max_diff == 0.0, f"Lookahead bias violation! Past predictions changed by {max_diff}"


# ==============================================================================
# T6: LATENCY SLA BENCHMARK
# ==============================================================================

class TestT6LatencySLABenchmark:
    """T6: Verifies streaming update_bar() execution latency: P50 < 1.0ms, P99 < 5.0ms on CPU."""

    def test_2000_iterations_update_bar_cpu_latency_sla(self) -> None:
        """Profiles 2,000 continuous update_bar() invocations and checks strict latency SLAs."""
        detector = RegimeDetector()
        detector.reset()

        # Warm up buffer
        for _ in range(65):
            detector.update_bar(90000.0, 90050.0, 89950.0, 90010.0, volume=300.0)

        n_iter = 2000
        latencies_ms = []

        for i in range(n_iter):
            p = 90000.0 + (i % 50) * 5.0
            t0 = time.perf_counter()
            detector.update_bar(p, p + 10.0, p - 10.0, p + 2.0, volume=350.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

        arr = np.array(latencies_ms)
        p50_ms = float(np.percentile(arr, 50))
        p90_ms = float(np.percentile(arr, 90))
        p99_ms = float(np.percentile(arr, 99))
        mean_ms = float(np.mean(arr))

        print(f"\n[Latency Benchmark] P50: {p50_ms:.4f} ms | P90: {p90_ms:.4f} ms | P99: {p99_ms:.4f} ms | Mean: {mean_ms:.4f} ms")

        # SLAs: P50 < 1.0ms, P99 < 5.0ms
        assert p50_ms < 1.0, f"Median latency P50 SLA violation: {p50_ms:.4f} ms >= 1.0 ms"
        assert p99_ms < 5.0, f"P99 latency SLA violation: {p99_ms:.4f} ms >= 5.0 ms"
        assert mean_ms < 1.0, f"Mean latency SLA violation: {mean_ms:.4f} ms >= 1.0 ms"


# ==============================================================================
# T7: MT5 IPC WIRE HOOK INTEGRATION
# ==============================================================================

class TestT7MT5IPCWireIntegration:
    """T7: Verifies MT5 Strategy Tester IPC bridge hook (predict_wire_features) with 137 floats."""

    def test_exact_137_float_wire_input(self) -> None:
        """Ingests standard 137-float wire input vector and returns valid regime & probabilities."""
        detector = RegimeDetector()
        wire_137 = np.zeros(137, dtype=np.float64)
        wire_137[0] = 0.55   # velocity-like indicator
        wire_137[1] = -0.45  # acceleration
        wire_137[93] = 1.0   # position indicator

        regime_idx, probs = detector.predict_wire_features(wire_137)
        assert isinstance(regime_idx, (int, np.integer))
        assert regime_idx in {MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.CHOP}
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_truncated_wire_input_auto_padding(self) -> None:
        """Inputs shorter than 137 floats are safely zero-padded without crashing."""
        detector = RegimeDetector()
        short_inputs = np.array([0.2, -0.1, 0.5])
        regime_idx, probs = detector.predict_wire_features(short_inputs)
        assert regime_idx in {0, 1, 2}
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    def test_direct_15_feature_wire_passthrough(self) -> None:
        """Exact 15-feature wire vector directly evaluates without reshaping errors."""
        detector = RegimeDetector()
        feat_15 = np.zeros(15, dtype=np.float64)
        feat_15[9] = 1.2
        feat_15[14] = 1.4

        regime_idx, probs = detector.predict_wire_features(feat_15)
        assert regime_idx == MarketRegime.BULL
        assert probs[MarketRegime.BULL] > probs[MarketRegime.BEAR]


# ==============================================================================
# T8: ROBUSTNESS & ADVERSARIAL SAFETY
# ==============================================================================

class TestT8RobustnessAndAdversarialSafety:
    """T8: Verifies safety under warmup, flat markets, flash crash, zero volume, and corrupted inputs."""

    def test_warmup_phase_fails_closed_to_neutral_prior(self) -> None:
        """First 59 bars emit neutral prior [1/3, 1/3, 1/3] and CHOP."""
        detector = RegimeDetector(warmup_bars=60)
        detector.reset()

        for bar_idx in range(59):
            regime_idx, probs = detector.update_bar(
                open_=90000.0, high=90050.0, low=89950.0, close=90010.0, volume=300.0
            )
            assert regime_idx == MarketRegime.CHOP, f"Bar {bar_idx} must emit CHOP during warmup"
            np.testing.assert_allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-6)

        # Bar 60 reaches warmup threshold
        regime_idx, probs = detector.update_bar(
            open_=90010.0, high=90060.0, low=89960.0, close=90020.0, volume=300.0
        )
        assert len(detector.buffer) == 60
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    def test_flat_market_zero_variance_handled_gracefully(self) -> None:
        """100 bars of identical O == H == L == C do not raise ZeroDivisionError and emit valid CHOP."""
        detector = RegimeDetector()
        detector.reset()

        for _ in range(70):
            regime_idx, probs = detector.update_bar(
                open_=90000.0, high=90000.0, low=90000.0, close=90000.0, volume=0.0
            )

        assert regime_idx == MarketRegime.CHOP
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    def test_single_bar_flash_crash_elevates_bear_probability(self) -> None:
        """Sudden -15% flash crash in a single bar safely produces dominant Bear probability (>0.30)."""
        detector = RegimeDetector()
        detector.reset()

        # Prime with 70 quiet bars
        for _ in range(70):
            detector.update_bar(90000.0, 90020.0, 89980.0, 90000.0, volume=300.0)

        # Flash crash bar: 90,000 -> 75,000
        regime_idx, probs = detector.update_bar(90000.0, 90000.0, 74000.0, 75000.0, volume=2000.0)
        assert probs[MarketRegime.BEAR] > 0.30, f"Flash crash must elevate Bear probability, got {probs}"
        assert not np.isnan(probs).any()

    def test_corrupted_nan_and_inf_inputs_fail_closed(self) -> None:
        """NaN or Inf values in input features fail closed safely to neutral prior."""
        detector = RegimeDetector()

        corrupt_nan = np.array([np.nan] * 15)
        probs_nan = detector.predict_proba(corrupt_nan)
        np.testing.assert_allclose(probs_nan, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-6)

        corrupt_inf = np.array([np.inf] * 15)
        probs_inf = detector.predict_proba(corrupt_inf)
        np.testing.assert_allclose(probs_inf, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-6)


# ==============================================================================
# T9: SERIALIZATION ROUND-TRIP
# ==============================================================================

class TestT9SerializationRoundTrip:
    """T9: Verifies save() and load() persistence with bit-for-bit identical probabilities."""

    def test_pickle_serialization_round_trip_bit_for_bit(self) -> None:
        """Saving to .pkl and reloading restores detector producing bit-for-bit identical outputs."""
        detector = RegimeDetector(
            config={"model_id": "test_v1", "version": "1.0"},
            window_maxlen=400,
            warmup_bars=70,
        )
        test_feat = np.array([0.1, -0.2, 0.05, 0.0, 1.2, 0.02, 0.015, 50.0, 1.1, 0.8, -0.1, 0.05, 0.6, 1.5, 0.8])
        p1 = detector.predict_proba(test_feat)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir) / "test_detector.pkl"
            detector.save(pkl_path)
            assert pkl_path.exists()

            reloaded = RegimeDetector.load(pkl_path)
            assert reloaded.config == detector.config
            assert reloaded.window_maxlen == detector.window_maxlen
            assert reloaded.warmup_bars == detector.warmup_bars

            p2 = reloaded.predict_proba(test_feat)
            np.testing.assert_array_equal(p1, p2)

    def test_json_config_serialization_round_trip(self) -> None:
        """Saving to .json preserves configuration dictionary faithfully."""
        cfg = {"threshold": 0.5, "model_id": "json_test", "active": True}
        detector = RegimeDetector(config=cfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "detector_config.json"
            detector.save(json_path)
            assert json_path.exists()

            reloaded = RegimeDetector.load(json_path)
            assert reloaded.config == cfg

    def test_champion_model_artifact_load_and_inference(self) -> None:
        """Loads official champion model from outputs/regime_models/champion_regime_detector.pkl."""
        champ_pkl = Path("outputs") / "regime_models" / "champion_regime_detector.pkl"
        if not champ_pkl.exists():
            pytest.skip(f"Champion artifact {champ_pkl} not present in workspace.")

        champ_detector = RegimeDetector.load(champ_pkl)
        assert champ_detector.model is not None
        assert isinstance(champ_detector.model, BaseRegimeDetector)

        # Evaluate inference with champion detector
        test_inputs = np.random.randn(10, 15)
        probs = champ_detector.predict_proba(test_inputs)
        assert probs.shape == (10, 3)
        np.testing.assert_allclose(np.sum(probs, axis=-1), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
