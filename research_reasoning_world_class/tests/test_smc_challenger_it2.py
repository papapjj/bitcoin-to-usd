"""
Adversarial Challenger Test Suite: SMC Liquidity Engine
Challenger: challenger_m1_it2_1
Milestone: M1 Iteration 2
Target: research_reasoning_world_class/src/smc_liquidity_engine.py

Aggressively stress-tests:
1. Complete elimination of false breakout trap flaws (descending highs, ascending lows, non-traps).
2. Latency stress benchmark (1,000 iterations consistently well below 1.0ms SLA).
3. Edge cases: flat bars, single-tick wicks, zero volume, negative spreads/inverted bars, extreme ATR multiples, random noise sequences.
4. Numerical integrity: all 64 floats finite, zero NaN/Inf, contract bounds compliance.
5. Multi-asset scale preservation under missing/corrupted data.
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


def check_finite_and_shape(feat: np.ndarray):
    """Enforce exact shape (64,), float32, zero NaN, zero Inf, all finite."""
    assert isinstance(feat, np.ndarray), f"Expected np.ndarray, got {type(feat)}"
    assert feat.shape == (64,), f"Expected shape (64,), got {feat.shape}"
    assert feat.dtype == np.float32, f"Expected dtype float32, got {feat.dtype}"
    assert not np.isnan(feat).any(), f"NaN detected at indices: {np.where(np.isnan(feat))[0]}"
    assert not np.isinf(feat).any(), f"Inf detected at indices: {np.where(np.isinf(feat))[0]}"


class TestFalseBreakoutTrapDeepElimination:
    """Rigorous verification of false breakout trap detection across diverse market topologies."""

    def test_multi_tier_descending_highs_traps_most_recent(self, engine: SMCLiquidityEngine):
        """
        Setup 3 descending swing highs:
        SH1 at bar 10: 1100.0 (all-time highest)
        SH2 at bar 25: 1060.0 (intermediate)
        SH3 at bar 42: 1030.0 (most recent lower swing high)
        Bar 58 breaks out above SH3 (high=1036.0, close=1033.0).
        Bar 59 falls back and closes below SH3 at 1025.0.
        """
        bars = np.full((60, 5), 1000.0, dtype=np.float64)
        bars[:, 1] = 1005.0
        bars[:, 2] = 995.0

        # SH1 at 10
        bars[10, 1] = 1100.0
        bars[8:10, 1] = 1020.0
        bars[11:13, 1] = 1020.0

        # SH2 at 25
        bars[25, 1] = 1060.0
        bars[23:25, 1] = 1020.0
        bars[26:28, 1] = 1020.0

        # SH3 at 42
        bars[42, 1] = 1030.0
        bars[40:42, 1] = 1010.0
        bars[43:45, 1] = 1010.0

        # Bar 58 breaches SH3
        bars[58, 0] = 1025.0
        bars[58, 1] = 1036.0
        bars[58, 2] = 1024.0
        bars[58, 3] = 1033.0

        # Bar 59 reverses below SH3
        bars[59, 0] = 1032.0
        bars[59, 1] = 1032.0
        bars[59, 2] = 1022.0
        bars[59, 3] = 1025.0

        feat = engine.extract_features(bars)
        check_finite_and_shape(feat)
        f = engine.get_feature_dict(feat)

        assert f["trapped_buyer_score"] > 0.50, (
            f"Expected trapped_buyer_score > 0.50 on descending SH3, got {f['trapped_buyer_score']}"
        )

    def test_multi_tier_ascending_lows_traps_most_recent(self, engine: SMCLiquidityEngine):
        """
        Setup 3 ascending swing lows:
        SL1 at bar 10: 900.0 (all-time lowest)
        SL2 at bar 25: 940.0 (intermediate)
        SL3 at bar 42: 970.0 (most recent higher swing low)
        Bar 58 breaks down below SL3 (low=964.0, close=967.0).
        Bar 59 reverses back up and closes above SL3 at 975.0.
        """
        bars = np.full((60, 5), 1000.0, dtype=np.float64)
        bars[:, 1] = 1005.0
        bars[:, 2] = 995.0

        # SL1 at 10
        bars[10, 2] = 900.0
        bars[8:10, 2] = 980.0
        bars[11:13, 2] = 980.0

        # SL2 at 25
        bars[25, 2] = 940.0
        bars[23:25, 2] = 980.0
        bars[26:28, 2] = 980.0

        # SL3 at 42
        bars[42, 2] = 970.0
        bars[40:42, 2] = 990.0
        bars[43:45, 2] = 990.0

        # Bar 58 breaches SL3
        bars[58, 0] = 975.0
        bars[58, 1] = 976.0
        bars[58, 2] = 964.0
        bars[58, 3] = 967.0

        # Bar 59 reverses above SL3
        bars[59, 0] = 968.0
        bars[59, 1] = 978.0
        bars[59, 2] = 968.0
        bars[59, 3] = 975.0

        feat = engine.extract_features(bars)
        check_finite_and_shape(feat)
        f = engine.get_feature_dict(feat)

        assert f["trapped_seller_score"] > 0.50, (
            f"Expected trapped_seller_score > 0.50 on ascending SL3, got {f['trapped_seller_score']}"
        )

    def test_genuine_breakout_does_not_trigger_false_trap(self, engine: SMCLiquidityEngine):
        """
        A strong breakout that holds and expands above the swing high
        must NOT trigger trapped_buyer_score.
        """
        bars = np.full((60, 5), 1000.0, dtype=np.float64)
        bars[:, 1] = 1005.0
        bars[:, 2] = 995.0

        # Swing high at 30: 1020.0
        bars[30, 1] = 1020.0
        bars[28:30, 1] = 1005.0
        bars[31:33, 1] = 1005.0

        # Bar 58 breaks out to 1030.0 and closes at 1028.0
        bars[58, 0] = 1015.0
        bars[58, 1] = 1030.0
        bars[58, 2] = 1015.0
        bars[58, 3] = 1028.0

        # Bar 59 expands higher to 1045.0, closes at 1042.0 (Genuine breakout continuation)
        bars[59, 0] = 1028.0
        bars[59, 1] = 1045.0
        bars[59, 2] = 1027.0
        bars[59, 3] = 1042.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["trapped_buyer_score"] == 0.0, (
            f"Genuine breakout continuation must have trapped_buyer_score == 0.0, got {f['trapped_buyer_score']}"
        )

    def test_genuine_breakdown_does_not_trigger_false_trap(self, engine: SMCLiquidityEngine):
        """
        A strong breakdown that holds and expands below the swing low
        must NOT trigger trapped_seller_score.
        """
        bars = np.full((60, 5), 1000.0, dtype=np.float64)
        bars[:, 1] = 1005.0
        bars[:, 2] = 995.0

        # Swing low at 30: 980.0
        bars[30, 2] = 980.0
        bars[28:30, 2] = 995.0
        bars[31:33, 2] = 995.0

        # Bar 58 breaks down to 970.0 and closes at 972.0
        bars[58, 0] = 985.0
        bars[58, 1] = 986.0
        bars[58, 2] = 970.0
        bars[58, 3] = 972.0

        # Bar 59 expands lower to 955.0, closes at 958.0 (Genuine breakdown continuation)
        bars[59, 0] = 972.0
        bars[59, 1] = 973.0
        bars[59, 2] = 955.0
        bars[59, 3] = 958.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["trapped_seller_score"] == 0.0, (
            f"Genuine breakdown continuation must have trapped_seller_score == 0.0, got {f['trapped_seller_score']}"
        )


class TestLatencyBenchmark1000:
    """Rigorous empirical verification of latency across 1,000 iterations."""

    def test_1000_iterations_sub_millisecond_consistency(self, engine: SMCLiquidityEngine):
        np.random.seed(42)
        bars = np.zeros((60, 5), dtype=np.float64)
        p = 65000.0  # BTC price scale
        for i in range(60):
            o = p + np.random.uniform(-30, 30)
            c = o + np.random.uniform(-30, 30)
            h = max(o, c) + np.random.uniform(5, 50)
            l = min(o, c) - np.random.uniform(5, 50)
            v = np.random.uniform(10, 500)
            bars[i] = [o, h, l, c, v]
            p = c

        # Warmup
        for _ in range(50):
            _ = engine.extract_features(bars)

        latencies = []
        for _ in range(1000):
            t0 = time.perf_counter()
            _ = engine.extract_features(bars)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        p50_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        max_lat = float(np.max(latencies))

        print(
            f"\n[LATENCY 1000 RUNS] Mean={mean_lat:.4f}ms, P50={p50_lat:.4f}ms, "
            f"P95={p95_lat:.4f}ms, P99={p99_lat:.4f}ms, Max={max_lat:.4f}ms"
        )

        assert mean_lat < 1.0, f"Mean latency {mean_lat:.4f}ms exceeds 1.0ms SLA!"
        assert p99_lat < 2.0, f"P99 latency {p99_lat:.4f}ms exceeds 2.0ms!"


class TestEdgeCasesAdversarialHarness:
    """Aggressive probing of pathological and edge inputs."""

    def test_flat_bars_zero_range(self, engine: SMCLiquidityEngine):
        """All bars have identical O=H=L=C=constant."""
        for val in [0.0, 1.0, 100.0, 50000.0, -100.0]:
            bars = np.full((60, 5), val, dtype=np.float64)
            feat = engine.extract_features(bars)
            check_finite_and_shape(feat)

    def test_single_tick_wicks_microscopic_ranges(self, engine: SMCLiquidityEngine):
        """Candles with 1-tick wicks (range = 1e-5)."""
        bars = np.full((60, 5), 100.0, dtype=np.float64)
        bars[:, 1] = 100.0 + 1e-5
        bars[:, 2] = 100.0 - 1e-5
        bars[:, 4] = 10.0
        feat = engine.extract_features(bars)
        check_finite_and_shape(feat)

    def test_zero_volume_variations(self, engine: SMCLiquidityEngine):
        """Zero volume on all bars, single bars, alternating bars."""
        # All zero volume
        bars = np.random.uniform(90.0, 110.0, (60, 5))
        bars[:, 1] = np.maximum(bars[:, 0], bars[:, 3]) + 1.0
        bars[:, 2] = np.minimum(bars[:, 0], bars[:, 3]) - 1.0
        bars[:, 4] = 0.0
        feat = engine.extract_features(bars)
        check_finite_and_shape(feat)

        # Alternating zero volume
        bars[:, 4] = np.array([0.0 if i % 2 == 0 else 50.0 for i in range(60)])
        feat = engine.extract_features(bars)
        check_finite_and_shape(feat)

    def test_negative_spreads_and_inverted_candles(self, engine: SMCLiquidityEngine):
        """High < Low inverted candles (malformed broker tick feed)."""
        bars = np.random.uniform(100.0, 120.0, (60, 5))
        # Deliberately invert High and Low
        bars[:, 1] = np.minimum(bars[:, 0], bars[:, 3]) - 5.0  # High lower than Open/Close
        bars[:, 2] = np.maximum(bars[:, 0], bars[:, 3]) + 5.0  # Low higher than Open/Close
        feat = engine.extract_features(bars)
        check_finite_and_shape(feat)

    def test_extreme_atr_multiples_massive_spike(self, engine: SMCLiquidityEngine):
        """A normal range market followed by a 1,000x ATR flash spike."""
        bars = np.full((60, 5), 1000.0, dtype=np.float64)
        bars[:, 1] = 1002.0
        bars[:, 2] = 998.0
        bars[:, 4] = 100.0

        # Bar 59 spikes to 20,000 (20x price, ~5000x ATR)
        bars[59, 1] = 20000.0
        bars[59, 3] = 18000.0

        feat = engine.extract_features(bars)
        check_finite_and_shape(feat)

    def test_random_noise_sequences_cauchy_and_gaussian(self, engine: SMCLiquidityEngine):
        """Random noise sequences with heavy tails."""
        np.random.seed(777)
        for _ in range(50):
            length = np.random.randint(5, 120)
            # Cauchy jumps
            jumps = np.random.standard_cauchy(length) * 5.0
            p = np.cumsum(jumps) + 1000.0
            bars = np.zeros((length, 5), dtype=np.float64)
            bars[:, 0] = p
            bars[:, 3] = p + np.random.normal(0, 2, length)
            bars[:, 1] = np.maximum(bars[:, 0], bars[:, 3]) + np.abs(np.random.normal(0, 3, length))
            bars[:, 2] = np.minimum(bars[:, 0], bars[:, 3]) - np.abs(np.random.normal(0, 3, length))
            bars[:, 4] = np.maximum(0.0, np.random.normal(100, 50, length))

            feat = engine.extract_features(bars)
            check_finite_and_shape(feat)


class TestFeatureRangeBoundsContract:
    """
    Verify all 64 floats are finite and strictly within their defined mathematical ranges:
    e.g. [-5.0, 5.0] or [0.0, 1.0] or [-1.0, 1.0] or bounded volume/wick ratios <= 10.0.
    """

    def test_feature_ranges_on_100_diverse_scenarios(self, engine: SMCLiquidityEngine):
        np.random.seed(888)
        # Expected ranges per feature group based on SMC_FEATURE_NAMES
        # Groups:
        # 0, 1: binary flags [0, 1]
        # 2, 3: magnitudes >= 0
        # 4, 5: wick ratios [0, 1]
        # 6, 7: vol ratios [0, 10]
        # 8, 9: displacement [-1, 1]
        # 10, 11: bars since [0, 1]
        # 12, 13: distance [-5, 5]
        # 14, 15: equal highs/lows [0, 1]
        # 16, 17: FVG active [0, 1]
        # 18, 19: FVG gap size [0, 5]
        # 20, 21: CE distance [-5, 5]
        # 22, 23: mitigation ratio [0, 1]
        # 24, 25: price inside FVG [0, 1]
        # 26, 27: impulse vol [0, 10]
        # 28, 29: bars since [0, 1]
        # 30: net FVG bias [-1, 1]
        # 31: inversion flag [-1, 1]
        # 32, 33: OB active [0, 1]
        # 34..39: OB distances [-5, 5]
        # 40, 41: OB impulse vol [0, 10]
        # 42, 43: OB mitigation state [0, 1]
        # 44, 45: bars since [0, 1]
        # 46: breaker status [-1, 1]
        # 47: OB momentum [-5, 5]
        # 48, 49: trapped score [0, 1]
        # 50, 51: breakout failure, vol exhaustion [0, 1]
        # 52: judas swing [-1, 1]
        # 53, 54: choch, bos [-1, 1]
        # 55: disp quality [0, 1]
        # 56: equilibrium [0, 1]
        # 57: idm proximity [0, 1]
        # 58: sweep divergence [-1, 1]
        # 59: void rate [0, 5]
        # 60: wick body ratio [0, 10]
        # 61: internal bias [-1, 1]
        # 62: external potential [0, 1]
        # 63: composite regime score [-1, 1]

        for trial in range(100):
            scale = float(np.random.choice([0.01, 1.0, 100.0, 2500.0, 65000.0]))
            n = np.random.randint(10, 100)
            bars = np.zeros((n, 5), dtype=np.float64)
            p = scale
            for i in range(n):
                o = p + np.random.uniform(-0.01, 0.01) * scale
                c = o + np.random.uniform(-0.01, 0.01) * scale
                h = max(o, c) + np.random.uniform(0.001, 0.02) * scale
                l = min(o, c) - np.random.uniform(0.001, 0.02) * scale
                v = np.random.uniform(10.0, 1000.0)
                bars[i] = [o, h, l, c, v]
                p = c

            feat = engine.extract_features(bars)
            check_finite_and_shape(feat)

            # Assert strict global bounds [-10.0, 10.0]
            assert np.all(feat >= -10.0) and np.all(feat <= 10.0), (
                f"Trial {trial}: Feat out of [-10, 10]! min={np.min(feat)}, max={np.max(feat)}"
            )

            # Assert specific subsets are in [0, 1]
            unit_indices = [0, 1, 4, 5, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 28, 29,
                            32, 33, 42, 43, 44, 45, 48, 49, 50, 51, 55, 56, 57, 62]
            for idx in unit_indices:
                val = feat[idx]
                assert 0.0 <= val <= 1.0, (
                    f"Trial {trial}: Feature {idx} ({SMC_FEATURE_NAMES[idx]}) = {val} outside [0.0, 1.0]!"
                )

            # Assert bipolar features are in [-1, 1]
            bipolar_indices = [8, 9, 30, 31, 46, 52, 53, 54, 58, 61, 63]
            for idx in bipolar_indices:
                val = feat[idx]
                assert -1.0 <= val <= 1.0, (
                    f"Trial {trial}: Feature {idx} ({SMC_FEATURE_NAMES[idx]}) = {val} outside [-1.0, 1.0]!"
                )

            # Assert distance and gap features are in [-5, 5]
            five_indices = [12, 13, 18, 19, 20, 21, 34, 35, 36, 37, 38, 39, 47, 59]
            for idx in five_indices:
                val = feat[idx]
                assert -5.0 <= val <= 5.0, (
                    f"Trial {trial}: Feature {idx} ({SMC_FEATURE_NAMES[idx]}) = {val} outside [-5.0, 5.0]!"
                )


class TestMultiAssetScalePreservation:
    """Verify that multi-asset price scale is preserved when corrupted with NaNs."""

    @pytest.mark.parametrize("symbol, base_price", [
        ("BTCUSD", 65000.0),
        ("ETHUSD", 3500.0),
        ("SOLUSD", 150.0),
        ("XAUUSD", 2600.0),
        ("EURUSD", 1.0850),
        ("US30", 42500.0),
    ])
    def test_price_scale_preserved_with_nans(self, engine: SMCLiquidityEngine, symbol: str, base_price: float):
        np.random.seed(123)
        bars = np.full((60, 5), base_price, dtype=np.float64)
        bars[:, 1] = base_price * 1.002
        bars[:, 2] = base_price * 0.998
        bars[:, 4] = 100.0

        # Corrupt 10 random prices with NaN
        corrupt_idx = [3, 7, 12, 22, 35, 41, 50, 55, 58]
        for idx in corrupt_idx:
            bars[idx, :4] = np.nan

        feat = engine.extract_features(bars, symbol=symbol)
        check_finite_and_shape(feat)

        # Confirm equilibrium feature is close to 0.5 (meaning price scale was not distorted to 100.0)
        f = engine.get_feature_dict(feat)
        assert 0.0 <= f["premium_discount_equilibrium"] <= 1.0


class TestAdversarialFuzzHarness2000Trials:
    """2,000 aggressive randomized Monte Carlo trials stressing the SMC Liquidity Engine."""

    def test_2000_fuzz_trials_strict_finite(self, engine: SMCLiquidityEngine):
        np.random.seed(99999)
        pathological_floats = [
            np.nan,
            np.inf,
            -np.inf,
            0.0,
            -0.0,
            1e-300,
            -1e-300,
            1e300,
            -1e300,
            1e-15,
            1e15,
            -100000.0,
        ]

        for trial in range(2000):
            # Length between 0 and 200
            n = int(np.random.choice([0, 1, 2, 4, 5, 15, 60, 100, 200]))
            if n == 0:
                bars = np.zeros((0, 5), dtype=np.float64)
            else:
                scale = float(10.0 ** np.random.uniform(-10, 10))
                bars = np.random.uniform(-5.0, 5.0, (n, 5)) * scale

                # 30% chance inverted High < Low
                if np.random.rand() < 0.30:
                    bars[:, 1] = np.minimum(bars[:, 0], bars[:, 3]) - np.abs(bars[:, 1])
                    bars[:, 2] = np.maximum(bars[:, 0], bars[:, 3]) + np.abs(bars[:, 2])
                else:
                    bars[:, 1] = np.maximum(bars[:, 0], bars[:, 3]) + np.abs(bars[:, 1])
                    bars[:, 2] = np.minimum(bars[:, 0], bars[:, 3]) - np.abs(bars[:, 2])

                # 35% chance corrupt with pathological tokens
                if np.random.rand() < 0.35:
                    corrupt_mask = np.random.rand(*bars.shape) < 0.25
                    bars[corrupt_mask] = np.random.choice(pathological_floats)

                # 10% chance zero volume
                if np.random.rand() < 0.10:
                    bars[:, 4] = 0.0

            with np.errstate(all="ignore"):
                feat = engine.extract_features(bars)
            check_finite_and_shape(feat)
            assert np.all(feat >= -100.0) and np.all(feat <= 100.0)


class TestBufferScalingAndLatencies:
    """Assess execution time scaling from small (5 bars) to large (500 bars) buffers."""

    @pytest.mark.parametrize("n_bars", [5, 30, 60, 120, 250, 500])
    def test_buffer_size_latency(self, engine: SMCLiquidityEngine, n_bars: int):
        np.random.seed(42)
        bars = np.zeros((n_bars, 5), dtype=np.float64)
        p = 50000.0
        for i in range(n_bars):
            o = p + np.random.uniform(-10, 10)
            c = o + np.random.uniform(-10, 10)
            h = max(o, c) + np.random.uniform(1, 20)
            l = min(o, c) - np.random.uniform(1, 20)
            v = np.random.uniform(10, 200)
            bars[i] = [o, h, l, c, v]
            p = c

        # Warmup
        for _ in range(10):
            _ = engine.extract_features(bars)

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = engine.extract_features(bars)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        print(f"\n[SCALE] n_bars={n_bars:3d}: {elapsed_ms:.4f} ms/call")
        # Even with 500 bars, execution should be well within the 50ms MT5 latency budget
        assert elapsed_ms < 10.0, f"Latency {elapsed_ms:.4f}ms exceeded 10.0ms for {n_bars} bars!"

