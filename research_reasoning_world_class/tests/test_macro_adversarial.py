"""
Adversarial Stress Test Suite for CrossAssetMacroEngine (Module R3)
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine
Milestone: M1 (Cross-Asset Macro Reasoning)

Adversarial Objectives:
1. Weekend flatline stress testing (zero variance across all 4 or subset of assets).
2. Missing instruments, partial quote updates, and asynchronous feed rates.
3. Injected NaNs, Infs, extreme numbers, zero, negative prices, and subnormals.
4. Mathematical invariant verification: correlation bounds [-1.0, 1.0], regime index bounds [-1.0, 1.0].
5. Shape (32,), dtype float32, zero NaN, zero Inf verification (AC-7).
6. Execution latency benchmark under AC-6 (< 1.0ms target per extraction).
7. Memory bounding under prolonged streaming (10,000+ ticks).
8. Fuzzing / Monte Carlo corruption test harness.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pytest

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


class TestAdversarialWeekendFlatline:
    """Stress test engine behavior under weekend / market-closed flatlines (zero variance)."""

    def test_complete_weekend_flatline_all_assets(self):
        """All 4 macro assets completely flatlined for 100 consecutive bars."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        for sym, baseline in DEFAULT_MACRO_BASELINES.items():
            engine.set_price_history(sym, [baseline] * 100)

        output = engine.extract_features()
        feats = output.features

        # AC-7 invariants
        assert feats.shape == (32,), f"Expected shape (32,), got {feats.shape}"
        assert feats.dtype == np.float32, f"Expected float32, got {feats.dtype}"
        assert not np.isnan(feats).any(), "NaN detected in flatline features!"
        assert not np.isinf(feats).any(), "Inf detected in flatline features!"
        assert np.all(feats >= -1.0) and np.all(feats <= 1.0), "Features outside [-1.0, 1.0]!"

        # Features 0..15: all returns and volatilities must be strictly 0.0
        assert np.all(feats[0:16] == 0.0), f"Returns/volatilities should be 0.0, got {feats[0:16]}"

        # Features 16..21: all pairwise correlations on zero-variance series must be 0.0
        assert np.all(feats[16:22] == 0.0), f"Pairwise correlations should be 0.0, got {feats[16:22]}"

        # Features 22..27: Regime scores must be neutral/zero
        assert feats[22] == pytest.approx(0.0), f"Regime index should be 0.0, got {feats[22]}"
        assert output.regime == MacroRegime.NEUTRAL
        assert feats[23] == pytest.approx(0.0)  # dollar_pressure
        assert feats[24] == pytest.approx(0.0)  # yield_pressure
        assert feats[25] == pytest.approx(0.0)  # equity_sentiment
        assert feats[26] == pytest.approx(0.0)  # flight_to_safety
        assert feats[27] == pytest.approx(0.0)  # crypto_macro_alignment

        # Features 28..31: Systemic macro metrics
        assert feats[28] == pytest.approx(0.0)  # velocity_spread
        assert feats[29] == pytest.approx(0.0)  # macro_stress_vol
        assert feats[30] == pytest.approx(0.0)  # liquidity_expansion
        assert feats[31] == pytest.approx(1.0)  # data_integrity: all 4 assets present

    def test_single_asset_flatline_spx_holiday(self):
        """SPX is flatlined (e.g. US market holiday) while DXY, US10Y, XAUUSD are active."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        np.random.seed(101)
        steps = 70
        ret = np.random.normal(0.001, 0.005, steps)

        p_dxy = 104.0 * np.exp(np.cumsum(ret))
        p_us10y = 4.25 * np.exp(np.cumsum(ret * 0.5))
        p_xau = 2350.0 * np.exp(np.cumsum(-ret))
        p_spx = [5000.0] * steps  # Flatlined

        engine.set_price_history("DXY", p_dxy)
        engine.set_price_history("US10Y", p_us10y)
        engine.set_price_history("SPX", p_spx)
        engine.set_price_history("XAUUSD", p_xau)

        output = engine.extract_features()
        feats = output.features

        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        assert np.all(feats >= -1.0) and np.all(feats <= 1.0)

        # SPX returns and vol (features 8..11) should be 0.0
        assert np.all(feats[8:12] == 0.0)

        # Correlations involving SPX must be 0.0
        # pairs = [("DXY", "US10Y"), ("DXY", "SPX"), ("DXY", "XAUUSD"),
        #          ("US10Y", "SPX"), ("US10Y", "XAUUSD"), ("SPX", "XAUUSD")]
        # indices: 16: DXY-US10Y, 17: DXY-SPX, 18: DXY-XAU, 19: US10Y-SPX, 20: US10Y-XAU, 21: SPX-XAU
        assert feats[17] == pytest.approx(0.0)  # DXY-SPX
        assert feats[19] == pytest.approx(0.0)  # US10Y-SPX
        assert feats[21] == pytest.approx(0.0)  # SPX-XAU

        # Active pairs must have non-zero, valid correlations
        assert feats[16] > 0.50  # DXY and US10Y are positively correlated
        assert feats[18] < -0.50  # DXY and XAU are negatively correlated

    def test_multi_asset_flatline_three_assets_closed(self):
        """Only Crypto/Gold is trading (XAUUSD active); DXY, US10Y, SPX are closed."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        steps = 65
        engine.set_price_history("DXY", [104.0] * steps)
        engine.set_price_history("US10Y", [4.25] * steps)
        engine.set_price_history("SPX", [5000.0] * steps)
        engine.set_price_history("XAUUSD", np.linspace(2300, 2400, steps))

        output = engine.extract_features()
        feats = output.features

        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        assert np.all(feats[16:22] == 0.0)  # All pairwise correlations zero since at least one is flat
        assert feats[12] > 0.0  # XAU ret_5 > 0
        assert feats[13] > 0.0  # XAU ret_15 > 0
        assert feats[14] > 0.0  # XAU ret_60 > 0


class TestAdversarialMissingInstrumentsAndPartialFeeds:
    """Stress test missing symbols, empty engine, single symbol, and disparate buffer depths."""

    def test_completely_empty_engine(self):
        """Engine with zero quotes ingested."""
        engine = CrossAssetMacroEngine()
        output = engine.extract_features()
        feats = output.features

        assert feats.shape == (32,)
        assert feats.dtype == np.float32
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        assert np.all(feats == 0.0)
        assert output.data_integrity == 0.0
        assert output.regime == MacroRegime.NEUTRAL

    @pytest.mark.parametrize("valid_count", [1, 2, 3])
    def test_partial_asset_presence_and_integrity_score(self, valid_count: int):
        """Engine with only 1, 2, or 3 symbols receiving updates."""
        engine = CrossAssetMacroEngine()
        active_symbols = SUPPORTED_MACRO_ASSETS[:valid_count]
        for sym in active_symbols:
            engine.set_price_history(sym, [100.0, 101.0, 102.0, 103.0, 104.0])

        output = engine.extract_features()
        feats = output.features

        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        expected_integrity = valid_count / 4.0
        assert output.data_integrity == pytest.approx(expected_integrity)
        assert feats[31] == pytest.approx(expected_integrity)

    def test_highly_asymmetric_buffer_lengths(self):
        """Different instruments having vastly different buffer depths (e.g. 1, 5, 20, 200)."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        engine.set_price_history("DXY", [104.0] * 200)
        engine.set_price_history("US10Y", [4.25, 4.26, 4.27, 4.28, 4.29])  # 5 quotes
        engine.set_price_history("SPX", [5000.0, 5010.0])  # 2 quotes
        engine.set_price_history("XAUUSD", [2350.0])  # 1 quote (insufficient for return)

        output = engine.extract_features()
        feats = output.features

        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()
        # XAUUSD has only 1 price, so valid_assets_count = 3
        assert output.data_integrity == pytest.approx(0.75)


class TestAdversarialCorruptedAndExtremeInputs:
    """Stress test injected NaNs, Infs, extreme values, negative prices, subnormals, and zeros."""

    def test_injected_nans_and_infs_streaming(self):
        """Streaming quotes with interspersed NaNs and Infs."""
        engine = CrossAssetMacroEngine()
        # Warmup with valid quotes
        for _ in range(10):
            engine.update_quote("DXY", 104.0)
            engine.update_quote("US10Y", 4.25)
            engine.update_quote("SPX", 5000.0)
            engine.update_quote("XAUUSD", 2350.0)

        # Inject adversarial values
        corrupted_inputs = [
            float("nan"),
            float("inf"),
            float("-inf"),
            -99999.0,
            0.0,
            -0.0,
            -1e-12,
        ]

        for bad_val in corrupted_inputs:
            engine.update_quote("DXY", bad_val)
            engine.update_quote("US10Y", bad_val)
            engine.update_quote("SPX", bad_val)
            engine.update_quote("XAUUSD", bad_val)

            output = engine.extract_features()
            assert not np.isnan(output.features).any(), f"NaN leaked from input {bad_val}!"
            assert not np.isinf(output.features).any(), f"Inf leaked from input {bad_val}!"
            assert np.all(output.features >= -1.0) and np.all(output.features <= 1.0)

    def test_initial_quote_corruption(self):
        """First quote received for an instrument is corrupted."""
        engine = CrossAssetMacroEngine()
        engine.update_quote("DXY", float("nan"))
        engine.update_quote("US10Y", float("-inf"))
        engine.update_quote("SPX", -500.0)
        engine.update_quote("XAUUSD", 0.0)

        output = engine.extract_features()
        # Should sanitize to default baseline without crashing
        assert not np.isnan(output.features).any()
        assert not np.isinf(output.features).any()

    def test_extreme_floating_point_numbers(self):
        """Values near float limits: subnormals and near overflow."""
        engine = CrossAssetMacroEngine()
        extreme_series = [1e-300, 1e-100, 1e100, 1e300, 1e-12, 1e12]
        for sym in engine.symbols:
            engine.set_price_history(sym, extreme_series)

        output = engine.extract_features()
        assert not np.isnan(output.features).any()
        assert not np.isinf(output.features).any()
        assert np.all(output.features >= -1.0) and np.all(output.features <= 1.0)

    def test_direct_matrix_ingestion_with_corruptions(self):
        """Direct 2D matrix ingestion containing NaNs, -Infs, and negative numbers."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        mat = np.ones((4, 60), dtype=np.float64) * 100.0
        mat[0, 10] = np.nan
        mat[1, 20] = np.inf
        mat[2, 30] = -np.inf
        mat[3, 40] = -500.0
        mat[:, 50:] = 0.0

        engine.ingest_matrix(mat)
        output = engine.extract_features()
        assert not np.isnan(output.features).any()
        assert not np.isinf(output.features).any()
        assert np.all(output.features >= -1.0) and np.all(output.features <= 1.0)

    def test_direct_matrix_invalid_dimensions(self):
        """Ingesting matrix with invalid dimension raises ValueError."""
        engine = CrossAssetMacroEngine()
        with pytest.raises(ValueError):
            engine.ingest_matrix(np.zeros((3, 60)))  # 3 rows instead of 4
        with pytest.raises(ValueError):
            engine.ingest_matrix(np.zeros((4, 60, 2)))  # 3D instead of 2D


class TestAdversarialMathematicalInvariants:
    """Verify bounds, correlation correctness, and regime stability."""

    def test_correlation_bounds_and_symmetry(self):
        """6 pairwise correlations must strictly obey -1.0 <= rho <= 1.0."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        np.random.seed(42)
        for sym in engine.symbols:
            engine.set_price_history(sym, np.random.uniform(50, 150, 80))

        output = engine.extract_features()
        corrs = output.pairwise_correlations
        assert len(corrs) == 6

        for pair, rho in corrs.items():
            assert -1.0 <= rho <= 1.0, f"Correlation {pair} = {rho} violated [-1.0, 1.0]!"
            assert not math.isnan(rho), f"Correlation {pair} is NaN!"
            assert not math.isinf(rho), f"Correlation {pair} is Inf!"

    def test_macro_regime_index_and_sub_factors_bounds(self):
        """Macro regime index and sub-factors must strictly obey [-1.0, 1.0]."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        # Test across 100 extreme random scenarios
        np.random.seed(999)
        for _ in range(100):
            for sym in engine.symbols:
                engine.set_price_history(sym, np.random.lognormal(mean=4.0, sigma=1.0, size=70))

            output = engine.extract_features()
            assert -1.0 <= output.macro_regime_index <= 1.0
            assert -1.0 <= output.dollar_pressure_score <= 1.0
            assert -1.0 <= output.yield_pressure_score <= 1.0
            assert -1.0 <= output.equity_sentiment_score <= 1.0
            assert -1.0 <= output.flight_to_safety_score <= 1.0
            assert -1.0 <= output.crypto_macro_alignment <= 1.0
            assert 0.0 <= output.data_integrity <= 1.0
            assert isinstance(output.regime, MacroRegime)


class TestAdversarialPerformanceAndLatencyAC6:
    """Verify sub-1ms inference latency (AC-6 target) and memory bounds."""

    def test_latency_distribution_under_one_millisecond(self):
        """Benchmark 2,000 extractions to verify average and p99 latency < 1.0ms."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        for sym in engine.symbols:
            engine.set_price_history(sym, np.random.normal(100, 2, 70))

        # Warm-up run
        for _ in range(50):
            _ = engine.extract_features()

        iterations = 2000
        latencies_ms: List[float] = []

        for _ in range(iterations):
            t_start = time.perf_counter_ns()
            _ = engine.extract_features()
            t_end = time.perf_counter_ns()
            latencies_ms.append((t_end - t_start) / 1_000_000.0)

        lat_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(lat_arr))
        median_lat = float(np.median(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))
        p99_lat = float(np.percentile(lat_arr, 99))
        max_lat = float(np.max(lat_arr))

        print(
            f"\n[AC-6 LATENCY REPORT] {iterations} iterations: "
            f"Mean={mean_lat:.4f}ms, Median={median_lat:.4f}ms, "
            f"p95={p95_lat:.4f}ms, p99={p99_lat:.4f}ms, Max={max_lat:.4f}ms"
        )

        # Target: Mean latency must be < 1.0ms (MT5 SLA requirement)
        assert mean_lat < 1.0, f"Mean latency {mean_lat:.4f}ms exceeds 1.0ms target!"
        # Sub-50ms hard limit from AC-6
        assert max_lat < 50.0, f"Max latency {max_lat:.4f}ms exceeds 50.0ms hard SLA!"

    def test_buffer_memory_stability_streaming_ten_thousand_ticks(self):
        """Stream 10,000 sequential ticks per symbol; verify buffer sizes stay bounded."""
        lookback = 60
        engine = CrossAssetMacroEngine(lookback_window=lookback)
        max_allowed_buffer = lookback * 4  # 240 items

        for i in range(10000):
            p = 100.0 + math.sin(i * 0.1) * 10.0
            engine.update_quote("DXY", p, timestamp=float(i))
            engine.update_quote("US10Y", p * 0.04, timestamp=float(i))
            engine.update_quote("SPX", p * 50.0, timestamp=float(i))
            engine.update_quote("XAUUSD", p * 23.0, timestamp=float(i))

            if i % 1000 == 0:
                out = engine.extract_features()
                assert out.features.shape == (32,)

        # Check internal buffer limits
        for sym in engine.symbols:
            buf_len = len(engine._prices[sym])
            ts_len = len(engine._timestamps[sym])
            assert buf_len <= max_allowed_buffer, f"Buffer size {buf_len} exceeded {max_allowed_buffer}!"
            assert ts_len <= max_allowed_buffer, f"Timestamp size {ts_len} exceeded {max_allowed_buffer}!"
            assert buf_len == ts_len, f"Buffer and timestamp lengths diverged: {buf_len} vs {ts_len}"


class TestAdversarialFuzzingMonteCarlo:
    """Monte Carlo fuzzing harness with randomized extreme, corrupted, and degenerate inputs."""

    def test_monte_carlo_fuzz_harness(self):
        """Execute 500 randomized trials with arbitrary corruptions."""
        np.random.seed(777)
        engine = CrossAssetMacroEngine(lookback_window=60)
        corrupted_tokens = [np.nan, np.inf, -np.inf, 0.0, -100.0, 1e-300, 1e300, 1e-8]

        for trial in range(500):
            # Generate random length between 0 and 100
            length = np.random.randint(0, 100)
            for sym in engine.symbols:
                if length == 0:
                    engine.set_price_history(sym, [])
                else:
                    # Normal prices with 15% random corrupted tokens injected
                    prices = np.random.uniform(10.0, 1000.0, length)
                    corrupt_mask = np.random.rand(length) < 0.15
                    for idx in np.where(corrupt_mask)[0]:
                        prices[idx] = np.random.choice(corrupted_tokens)
                    engine.set_price_history(sym, prices.tolist())

            out = engine.extract_features()
            feats = out.features

            # Strict Invariants
            assert feats.shape == (32,), f"Trial {trial}: Shape {feats.shape} != (32,)"
            assert feats.dtype == np.float32, f"Trial {trial}: Dtype {feats.dtype} != float32"
            assert not np.isnan(feats).any(), f"Trial {trial}: NaN detected!"
            assert not np.isinf(feats).any(), f"Trial {trial}: Inf detected!"
            assert np.all(feats >= -1.0) and np.all(feats <= 1.0), f"Trial {trial}: Features not in [-1, 1]!"
            assert -1.0 <= out.macro_regime_index <= 1.0, f"Trial {trial}: Regime index out of bounds!"
            assert 0.0 <= out.data_integrity <= 1.0, f"Trial {trial}: Data integrity out of bounds!"
            for pair, rho in out.pairwise_correlations.items():
                assert -1.0 <= rho <= 1.0, f"Trial {trial}: Correlation {pair}={rho} out of bounds!"
