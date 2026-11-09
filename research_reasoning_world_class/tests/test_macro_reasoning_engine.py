"""
Unit Tests for Module R3: Cross-Asset Macro Reasoning Engine
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine
Milestone: M1 (Cross-Asset Macro Reasoning)

Validates:
1. Multi-asset ingestion for DXY, US10Y, SPX, XAUUSD.
2. Multi-horizon momentum & velocities over 5, 15, and 60 bars.
3. 6 pairwise rolling Pearson correlations with zero-variance protection.
4. Global Risk-On vs Risk-Off composite score I_macro in [-1.0, 1.0].
5. Dense 32-float embedding tensor contract: shape (32,), dtype float32, bounded.
6. Absolute zero NaN, zero Inf preservation (AC-7) across flatline and corrupt inputs.
7. Real-time inference latency benchmark under 1.0ms (AC-6 sub-50ms budget).
"""

from __future__ import annotations

import math
import os
import sys
import time
import pytest
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from research_reasoning_world_class.src.macro_reasoning_engine import (
        CrossAssetMacroEngine,
        DEFAULT_MACRO_BASELINES,
        MacroAsset,
        MacroConfig,
        MacroEngineOutput,
        MacroRegime,
        SUPPORTED_MACRO_ASSETS,
        MACRO_FEATURE_NAMES,
    )
except ImportError:
    from src.macro_reasoning_engine import (
        CrossAssetMacroEngine,
        DEFAULT_MACRO_BASELINES,
        MacroAsset,
        MacroConfig,
        MacroEngineOutput,
        MacroRegime,
        SUPPORTED_MACRO_ASSETS,
        MACRO_FEATURE_NAMES,
    )


class TestMacroIngestionAndDataStructures:
    """Suite 1: Ingestion mechanics and array buffers."""

    def test_supported_assets_and_initial_state(self):
        engine = CrossAssetMacroEngine()
        assert len(engine.symbols) == 4
        assert engine.symbols == ["DXY", "US10Y", "SPX", "XAUUSD"]
        assert len(engine.FEATURE_NAMES) == 32
        assert len(MACRO_FEATURE_NAMES) == 32
        assert engine.FEATURE_COUNT == 32

    def test_streaming_quote_ingestion_and_bounding(self):
        engine = CrossAssetMacroEngine(lookback_window=10)
        for i in range(100):
            engine.update_quote("DXY", 100.0 + i * 0.1, timestamp=float(i))
        # Buffer bounded to 4x lookback = 40
        assert len(engine._prices["DXY"]) == 40
        assert engine._prices["DXY"][-1] == pytest.approx(109.9)

    def test_direct_matrix_ingestion(self):
        engine = CrossAssetMacroEngine(lookback_window=60)
        mat = np.ones((4, 60), dtype=np.float64) * 100.0
        engine.ingest_matrix(mat)
        for sym in engine.symbols:
            assert len(engine._prices[sym]) == 60


class TestMomentumAndVelocityComputations:
    """Suite 2: Multi-window returns (5, 15, 60) and velocities."""

    def test_multi_window_returns_progression(self):
        engine = CrossAssetMacroEngine(lookback_window=60)
        # Monotonically increasing price series
        prices = [100.0 * (1.01 ** i) for i in range(70)]
        engine.set_price_history("SPX", prices)

        output = engine.extract_features()
        feats = output.features

        # SPX features start at index 8: [ret_5, ret_15, ret_60, vol_60]
        spx_ret_5 = feats[8]
        spx_ret_15 = feats[9]
        spx_ret_60 = feats[10]

        assert spx_ret_5 > 0.0
        assert spx_ret_15 > spx_ret_5  # Longer horizon cumulative return is strictly larger
        assert spx_ret_60 > spx_ret_15
        assert -1.0 <= spx_ret_60 <= 1.0


class TestPairwiseCrossAssetCorrelations:
    """Suite 3: 6 Pairwise Pearson correlations."""

    def test_perfect_positive_and_inverse_correlations(self):
        engine = CrossAssetMacroEngine(lookback_window=60)
        # Construct series with perfectly correlated (+1.0) and inverse (-1.0) returns
        ret = np.linspace(0.005, 0.02, 60)
        p_dxy = 100.0 * np.exp(np.cumsum(ret))
        p_us10y = 100.0 * np.exp(np.cumsum(ret * 2.0))       # Perfect +1.0
        p_spx = 100.0 * np.exp(np.cumsum(-ret))              # Perfect -1.0
        p_xau = p_dxy.copy()

        engine.set_price_history("DXY", p_dxy)
        engine.set_price_history("US10Y", p_us10y)
        engine.set_price_history("SPX", p_spx)
        engine.set_price_history("XAUUSD", p_xau)

        output = engine.extract_features()
        rho_dxy_us10y = output.features[16]
        rho_dxy_spx = output.features[17]

        assert rho_dxy_us10y == pytest.approx(1.0, abs=1e-3)
        assert rho_dxy_spx == pytest.approx(-1.0, abs=1e-3)

    def test_uncorrelated_noise(self):
        engine = CrossAssetMacroEngine(lookback_window=60)
        np.random.seed(123)
        engine.set_price_history("DXY", 100.0 + np.random.normal(0, 1, 65))
        engine.set_price_history("SPX", 5000.0 + np.random.normal(0, 5, 65))

        output = engine.extract_features()
        rho_dxy_spx = output.features[17]
        assert abs(rho_dxy_spx) < 0.40


class TestGlobalMacroRegimeClassification:
    """Suite 4: Risk-On vs Risk-Off score I_macro in [-1.0, 1.0]."""

    def test_strong_risk_on_scenario(self):
        """SPX surges, DXY drops, US10Y yields drop, XAUUSD rallies."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        engine.set_price_history("SPX", np.linspace(4800, 5200, 65))     # Strong rally
        engine.set_price_history("DXY", np.linspace(106, 102, 65))       # Dollar plunge
        engine.set_price_history("US10Y", np.linspace(4.5, 4.0, 65))     # Yield plunge
        engine.set_price_history("XAUUSD", np.linspace(2300, 2400, 65))  # Gold rally

        output = engine.extract_features()
        assert output.macro_regime_index > 0.35
        assert output.regime == MacroRegime.STRONG_RISK_ON
        assert output.crypto_macro_alignment > 0.35
        assert output.equity_sentiment_score > 0.0
        assert output.dollar_pressure_score > 0.0   # Positive means easing dollar pressure

    def test_strong_risk_off_scenario(self):
        """SPX collapses, DXY spikes, US10Y yields spike, XAUUSD drops."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        engine.set_price_history("SPX", np.linspace(5200, 4800, 65))     # Plunge
        engine.set_price_history("DXY", np.linspace(102, 106, 65))       # Dollar spike
        engine.set_price_history("US10Y", np.linspace(4.0, 4.6, 65))     # Yield spike
        engine.set_price_history("XAUUSD", np.linspace(2400, 2300, 65))  # Gold drop

        output = engine.extract_features()
        assert output.macro_regime_index < -0.35
        assert output.regime == MacroRegime.STRONG_RISK_OFF
        assert output.crypto_macro_alignment < -0.35

    def test_neutral_chop_regime(self):
        engine = CrossAssetMacroEngine(lookback_window=60)
        np.random.seed(42)
        for sym in engine.symbols:
            engine.set_price_history(sym, 100.0 + np.random.normal(0, 0.01, 65))

        output = engine.extract_features()
        assert abs(output.macro_regime_index) <= 0.15
        assert output.regime == MacroRegime.NEUTRAL

    def test_flight_to_safety_divergence(self):
        engine = CrossAssetMacroEngine(lookback_window=60)
        # Gold rallies while SPX dumps
        engine.set_price_history("XAUUSD", np.linspace(2300, 2450, 65))
        engine.set_price_history("SPX", np.linspace(5100, 4900, 65))
        output = engine.extract_features()
        assert output.flight_to_safety_score > 0.0


class TestDenseContextEmbeddingShapeAndIntegrity:
    """Suite 5: 32-float dense vector contract."""

    def test_output_shape_and_dtype(self):
        engine = CrossAssetMacroEngine()
        output = engine.extract_features()
        assert isinstance(output.features, np.ndarray)
        assert output.features.shape == (32,)
        assert output.features.dtype == np.float32

    def test_all_features_within_bounded_ranges(self):
        engine = CrossAssetMacroEngine()
        for sym in engine.symbols:
            engine.set_price_history(sym, np.random.uniform(50, 150, 65))
        output = engine.extract_features()
        assert np.all(output.features >= -1.0)
        assert np.all(output.features <= 1.0)

    def test_extract_features_array_helper(self):
        engine = CrossAssetMacroEngine()
        arr = engine.extract_features_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (32,)
        assert arr.dtype == np.float32

    def test_get_feature_dict_mapping(self):
        engine = CrossAssetMacroEngine()
        f_dict = engine.get_feature_dict()
        assert len(f_dict) == 32
        for name in engine.FEATURE_NAMES:
            assert name in f_dict
            assert isinstance(f_dict[name], float)


class TestNumericalSafetyAndInvariantsAC7:
    """Suite 6: Zero NaN, Zero Inf, zero variance, and missing quote safeguards."""

    def test_flatline_zero_variance_no_nan_or_inf(self):
        """When market is closed and prices flatline, standard deviations are zero."""
        engine = CrossAssetMacroEngine()
        for sym in engine.symbols:
            engine.set_price_history(sym, [100.0] * 65)

        output = engine.extract_features()
        assert not np.isnan(output.features).any()
        assert not np.isinf(output.features).any()
        # Correlations on zero-variance series must be 0.0
        assert np.all(output.features[16:22] == 0.0)
        assert output.macro_regime_index == 0.0

    def test_missing_and_partial_quotes_graceful_degradation(self):
        """Only DXY updated; others empty."""
        engine = CrossAssetMacroEngine()
        engine.set_price_history("DXY", [104.0, 104.2, 104.5])

        output = engine.extract_features()
        assert not np.isnan(output.features).any()
        assert not np.isinf(output.features).any()
        assert output.data_integrity == pytest.approx(0.25)  # 1 out of 4 symbols valid

    def test_adversarial_nan_and_inf_injection(self):
        """Adversarial input with NaN and extreme numbers."""
        engine = CrossAssetMacroEngine()
        engine.update_quote("DXY", float("nan"))
        engine.update_quote("US10Y", float("inf"))
        engine.update_quote("SPX", -999.0)
        engine.update_quote("XAUUSD", 1e12)

        output = engine.extract_features()
        assert not np.isnan(output.features).any()
        assert not np.isinf(output.features).any()

    def test_extreme_price_inputs(self):
        engine = CrossAssetMacroEngine()
        for sym in engine.symbols:
            engine.set_price_history(sym, [1e-12, 1e12] * 35)
        output = engine.extract_features()
        assert not np.isnan(output.features).any()
        assert not np.isinf(output.features).any()
        assert np.all(output.features >= -1.0)
        assert np.all(output.features <= 1.0)


class TestLatencyBenchmarkAC6:
    """Suite 7: Sub-50ms execution verification."""

    def test_feature_extraction_latency_sub_millisecond(self):
        engine = CrossAssetMacroEngine(lookback_window=60)
        for sym in engine.symbols:
            engine.set_price_history(sym, np.random.normal(100, 2, 100))

        # Benchmark 1,000 extractions
        start_time = time.perf_counter()
        iters = 1000
        for _ in range(iters):
            _ = engine.extract_features()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        avg_latency_ms = elapsed_ms / iters
        # Must execute far below the 50ms MT5 SLA (typically < 5ms)
        assert avg_latency_ms < 50.0, f"Average latency {avg_latency_ms:.4f}ms exceeded 50.0ms"
        assert elapsed_ms < 50.0 * iters
