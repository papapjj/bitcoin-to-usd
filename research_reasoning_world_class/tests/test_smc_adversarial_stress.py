"""
Adversarial Stress Test Suite for SMCLiquidityEngine (Milestone M1 / AC-6 / AC-7)
Empirical Challenger: challenger_m1_1

This suite aggressively probes SMCLiquidityEngine with:
1. Zero-range candles, identical prices (H=L=O=C), flatlines.
2. High-frequency volatility bursts, flash crashes, parabolic pumps.
3. Injected NaNs, Infs, -Infs across single, multiple, and all columns.
4. Extreme magnitudes: +/- 1e12, +/- 1e300, subnormal numbers (1e-300).
5. Truncated buffers: 0 bars, 1 bar, 2 bars, 3 bars, 4 bars, 5 bars.
6. Malformed input structures: inverted candles (H < L), negative volumes.
7. Random Monte Carlo stress test with 1,000 randomized pathological bar sets.
8. Rigorous latency benchmark (< 1.0ms SLA verification).
"""

import math
import time
import pytest
import numpy as np

from research_reasoning_world_class.src.smc_liquidity_engine import (
    SMCConfig,
    SMCLiquidityEngine,
    SMCOutput,
    SMC_FEATURE_NAMES,
)


@pytest.fixture
def engine() -> SMCLiquidityEngine:
    return SMCLiquidityEngine()


def assert_finite_64(feat: np.ndarray):
    """Rigorous invariant verification: exact shape (64,), float32, zero NaN, zero Inf."""
    assert isinstance(feat, np.ndarray), f"Expected np.ndarray, got {type(feat)}"
    assert feat.shape == (64,), f"Expected shape (64,), got {feat.shape}"
    assert feat.dtype == np.float32, f"Expected dtype float32, got {feat.dtype}"
    assert not np.isnan(feat).any(), f"NaN detected in features: {np.where(np.isnan(feat))[0]}"
    assert not np.isinf(feat).any(), f"Inf detected in features: {np.where(np.isinf(feat))[0]}"
    assert np.all(np.abs(feat) <= 100.0), f"Clamping exceeded: max abs={np.max(np.abs(feat))}"


class TestAdversarialTruncatedAndEmptyInputs:
    """Stress tests on buffers with < 5 bars, empty arrays, and edge lengths."""

    def test_empty_buffer_array(self, engine: SMCLiquidityEngine):
        bars = np.zeros((0, 5), dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)
        np.testing.assert_array_equal(feat, np.zeros(64, dtype=np.float32))

    def test_empty_list_input(self, engine: SMCLiquidityEngine):
        feat = engine.extract_features([])
        assert_finite_64(feat)
        np.testing.assert_array_equal(feat, np.zeros(64, dtype=np.float32))

    @pytest.mark.parametrize("n_bars", [1, 2, 3, 4])
    def test_sub_five_bar_buffers(self, engine: SMCLiquidityEngine, n_bars: int):
        bars = np.full((n_bars, 5), 100.0, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)
        np.testing.assert_array_equal(feat, np.zeros(64, dtype=np.float32))

    def test_exact_five_bars_minimal(self, engine: SMCLiquidityEngine):
        # 5 bars is the minimum threshold where extraction proceeds
        bars = np.array([
            [100.0, 105.0, 95.0, 102.0, 1000.0],
            [102.0, 106.0, 98.0, 104.0, 1100.0],
            [104.0, 108.0, 101.0, 107.0, 1200.0],
            [107.0, 109.0, 103.0, 105.0, 900.0],
            [105.0, 110.0, 102.0, 108.0, 1500.0],
        ], dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)


class TestAdversarialZeroRangeAndIdenticalPrices:
    """Stress tests on identical prices (H=L=O=C), zero ATR, zero volume."""

    def test_all_identical_positive_prices(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), 1234.56, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_all_identical_zero_prices(self, engine: SMCLiquidityEngine):
        bars = np.zeros((60, 5), dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_all_identical_negative_prices(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), -50.0, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_zero_range_candles_varying_over_time(self, engine: SMCLiquidityEngine):
        """H=L=O=C on every individual candle, but price steps upward."""
        bars = np.zeros((60, 5), dtype=np.float64)
        for i in range(60):
            p = 100.0 + i * 2.0
            bars[i] = [p, p, p, p, 50.0]
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_zero_volume_all_bars(self, engine: SMCLiquidityEngine):
        bars = np.random.uniform(90.0, 110.0, (60, 5))
        bars[:, 1] = bars[:, 0] + 5.0
        bars[:, 2] = bars[:, 0] - 5.0
        bars[:, 4] = 0.0  # Zero volume baseline
        feat = engine.extract_features(bars)
        assert_finite_64(feat)


class TestAdversarialInjectedNaNsAndInfs:
    """Stress tests on corrupted inputs containing NaN, +Inf, -Inf."""

    def test_all_nans_in_all_columns(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), np.nan, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_all_pos_infs_in_all_columns(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), np.inf, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_all_neg_infs_in_all_columns(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), -np.inf, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_randomly_scattered_nans_and_infs(self, engine: SMCLiquidityEngine):
        np.random.seed(42)
        bars = np.random.uniform(100.0, 200.0, (60, 5))
        mask_nan = np.random.rand(60, 5) < 0.20
        mask_inf = np.random.rand(60, 5) < 0.10
        mask_neginf = np.random.rand(60, 5) < 0.10
        bars[mask_nan] = np.nan
        bars[mask_inf] = np.inf
        bars[mask_neginf] = -np.inf

        feat = engine.extract_features(bars)
        assert_finite_64(feat)


class TestAdversarialExtremeMagnitudes:
    """Stress tests with astronomical and microscopic magnitudes."""

    def test_astronomical_positive_values_1e12(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), 1e12, dtype=np.float64)
        bars[:, 1] = 1e12 + 1e9
        bars[:, 2] = 1e12 - 1e9
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_astronomical_negative_values_minus_1e12(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), -1e12, dtype=np.float64)
        bars[:, 1] = -1e12 + 1e9
        bars[:, 2] = -1e12 - 1e9
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_extreme_float_limit_1e300(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), 1e300, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_subnormal_microscopic_numbers_1e_300(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), 1e-300, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert_finite_64(feat)


class TestAdversarialVolatilityBurstsAndPatterns:
    """Stress tests on flash crashes, 100x volatility explosions, and inverted bars."""

    def test_flash_crash_single_bar(self, engine: SMCLiquidityEngine):
        """Bar drops from 100,000 to 1 and back."""
        bars = np.full((60, 5), 100000.0, dtype=np.float64)
        bars[30, 2] = 1.0  # Flash crash low
        bars[30, 3] = 95000.0
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_volatility_explosion_burst(self, engine: SMCLiquidityEngine):
        """Alternating wild oscillations between 1e8 and 1e-8."""
        bars = np.zeros((60, 5), dtype=np.float64)
        for i in range(60):
            val = 1e8 if i % 2 == 0 else 1e-8
            bars[i] = [val, val * 2.0, val * 0.5, val, 1e6]
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_inverted_candles(self, engine: SMCLiquidityEngine):
        """Malformed candles where High < Low or Open > High."""
        bars = np.zeros((60, 5), dtype=np.float64)
        for i in range(60):
            # High is lower than Low
            bars[i] = [100.0, 90.0, 110.0, 95.0, 1000.0]
        feat = engine.extract_features(bars)
        assert_finite_64(feat)

    def test_negative_volume(self, engine: SMCLiquidityEngine):
        """Anomalous negative volume inputs."""
        bars = np.full((60, 5), 100.0, dtype=np.float64)
        bars[:, 4] = -500.0
        feat = engine.extract_features(bars)
        assert_finite_64(feat)


class TestFalseBreakoutTrapFlaws:
    """Demonstrates structural logic flaws in false breakout trap detection."""

    def test_trapped_buyer_on_recent_lower_high_flaw(self, engine: SMCLiquidityEngine):
        """
        In a downtrend:
        Swing high 1 at bar 15: price = 1080.0 (historical high).
        Swing high 2 at bar 45: price = 1050.0 (recent lower swing high).
        Bar 58 breaks out above 1050.0 to 1055.0, closes at 1052.0.
        Bar 59 reverses back down to 1040.0 (trapping breakout buyers above 1050.0).

        FLAW: Line 757 uses latest_sh = max(prior_highs) which returns 1080.0 instead
        of the most recent swing high (1050.0). Thus h[-2] (1055.0) > 1080.0 is False,
        and trapped_buyer_score incorrectly yields 0.0 instead of >= 0.50!
        """
        bars = np.full((60, 5), 1000.0, dtype=np.float64)
        bars[:, 1] = 1005.0
        bars[:, 2] = 995.0

        # Major High at 15
        bars[15, 1] = 1080.0
        bars[13:15, 1] = 1020.0
        bars[16:18, 1] = 1020.0

        # Recent Lower Swing High at 45
        bars[45, 1] = 1050.0
        bars[43:45, 1] = 1020.0
        bars[46:48, 1] = 1020.0

        # Breakout at 58
        bars[58, 0] = 1045.0
        bars[58, 1] = 1055.0
        bars[58, 2] = 1045.0
        bars[58, 3] = 1052.0

        # Reversal at 59
        bars[59, 0] = 1050.0
        bars[59, 1] = 1051.0
        bars[59, 2] = 1038.0
        bars[59, 3] = 1040.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        # Expected: trapped_buyer_score > 0.50.
        # Actual bug in engine: returns 0.0 because max(prior_highs) masked the recent swing high.
        assert f["trapped_buyer_score"] > 0.50, (
            f"Expected trapped_buyer_score > 0.50 for breakout above recent swing high 1050, got {f['trapped_buyer_score']}"
        )

    def test_trapped_seller_on_recent_higher_low_flaw(self, engine: SMCLiquidityEngine):
        """
        In an uptrend:
        Swing low 1 at bar 15: price = 920.0 (historical low).
        Swing low 2 at bar 45: price = 950.0 (recent higher swing low).
        Bar 58 breaks down below 950.0 to 945.0, closes at 947.0.
        Bar 59 reverses back up to 960.0 (trapping breakdown sellers below 950.0).

        FLAW: Line 769 uses latest_sl = min(prior_lows) which returns 920.0 instead
        of the most recent swing low (950.0). Thus l[-2] (945.0) < 920.0 is False,
        and trapped_seller_score incorrectly yields 0.0 instead of >= 0.50!
        """
        bars = np.full((60, 5), 1000.0, dtype=np.float64)
        bars[:, 1] = 1005.0
        bars[:, 2] = 995.0

        # Major Low at 15
        bars[15, 2] = 920.0
        bars[13:15, 2] = 980.0
        bars[16:18, 2] = 980.0

        # Recent Higher Swing Low at 45
        bars[45, 2] = 950.0
        bars[43:45, 2] = 980.0
        bars[46:48, 2] = 980.0

        # Breakdown at 58
        bars[58, 0] = 955.0
        bars[58, 1] = 956.0
        bars[58, 2] = 945.0
        bars[58, 3] = 947.0

        # Reversal at 59
        bars[59, 0] = 948.0
        bars[59, 1] = 962.0
        bars[59, 2] = 948.0
        bars[59, 3] = 960.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        # Expected: trapped_seller_score > 0.50.
        # Actual bug in engine: returns 0.0 because min(prior_lows) masked the recent swing low.
        assert f["trapped_seller_score"] > 0.50, (
            f"Expected trapped_seller_score > 0.50 for breakdown below recent swing low 950, got {f['trapped_seller_score']}"
        )




class TestMonteCarloAdversarialFuzzing:
    """1,000 randomized Monte Carlo stress iterations with pathological conditions."""

    def test_monte_carlo_1000_pathological_runs(self, engine: SMCLiquidityEngine):
        np.random.seed(999)
        num_trials = 1000

        for trial in range(num_trials):
            n_bars = np.random.choice([0, 1, 2, 4, 5, 10, 30, 60, 120])
            if n_bars == 0:
                bars = np.zeros((0, 5), dtype=np.float64)
            else:
                scale = 10.0 ** np.random.uniform(-10, 10)
                bars = np.random.uniform(-10.0, 10.0, (n_bars, 5)) * scale

                # Ensure High >= Low in 80% of cases, inverted in 20%
                if np.random.rand() > 0.20:
                    bars[:, 1] = np.maximum(bars[:, 0], bars[:, 3]) + np.abs(bars[:, 1])
                    bars[:, 2] = np.minimum(bars[:, 0], bars[:, 3]) - np.abs(bars[:, 2])

                # Inject NaNs/Infs randomly in 30% of trials
                if np.random.rand() < 0.30:
                    corrupt_mask = np.random.rand(*bars.shape) < 0.15
                    bars[corrupt_mask] = np.random.choice([np.nan, np.inf, -np.inf])

            feat = engine.extract_features(bars)
            assert_finite_64(feat)


class TestSubMillisecondPerformanceAC6:
    """Empirical verification that SMC extraction completes in < 1ms on average."""

    def test_execution_latency_sub_millisecond(self, engine: SMCLiquidityEngine):
        # Generate representative 60-bar M1 sequence
        np.random.seed(123)
        bars = np.zeros((60, 5), dtype=np.float64)
        p = 50000.0
        for i in range(60):
            o = p + np.random.uniform(-20, 20)
            c = o + np.random.uniform(-20, 20)
            h = max(o, c) + np.random.uniform(5, 30)
            l = min(o, c) - np.random.uniform(5, 30)
            v = np.random.uniform(50, 500)
            bars[i] = [o, h, l, c, v]
            p = c

        # Warmup JIT/CPU caches
        for _ in range(50):
            _ = engine.extract_features(bars)

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _ = engine.extract_features(bars)
        elapsed_sec = time.perf_counter() - start

        avg_latency_ms = (elapsed_sec / iterations) * 1000.0
        p99_budget_ms = 1.0  # Target < 1ms

        print(f"\n[BENCHMARK] SMC 60-bar extraction: {avg_latency_ms:.4f} ms/call ({iterations} iterations)")
        assert avg_latency_ms < p99_budget_ms, (
            f"Extraction latency {avg_latency_ms:.4f}ms exceeds the < 1.0ms SLA target!"
        )
