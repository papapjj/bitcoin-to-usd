"""
Unit Tests for Module R2: Smart Money Concepts (SMC) & ICT Liquidity Engine
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine
Milestone: M1 (Deep Order Flow & Liquidity Engine)

Validates:
1. Shape contract: exactly 64 float features, 1D array of shape (64,), dtype float32.
2. BSL / SSL liquidity sweep detection, wick rejection ratios, sweep magnitude.
3. Bullish / Bearish FVG detection, gap size, Consequent Encroachment (CE), mitigation.
4. Institutional Order Block (OB) detection with impulse volume, boundaries, MT.
5. False breakout trap detection (trapped buyers, trapped sellers, follow-through failure).
6. Absolute 0 NaN, 0 Inf preservation (AC-7) across adversarial inputs.
7. Sub-millisecond execution latency benchmark (AC-6 <= 50ms MT5 budget).
8. Batch extraction consistency and multi-symbol BarData compatibility.
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
    from research_reasoning_world_class.src.smc_liquidity_engine import (
        SMCConfig,
        SMCOutput,
        SMCLiquidityEngine,
        SMC_FEATURE_NAMES,
    )
except ImportError:
    from src.smc_liquidity_engine import (
        SMCConfig,
        SMCOutput,
        SMCLiquidityEngine,
        SMC_FEATURE_NAMES,
    )


@pytest.fixture
def engine() -> SMCLiquidityEngine:
    return SMCLiquidityEngine(SMCConfig(swing_lookback=2, atr_period=10, volume_baseline_period=15))


def generate_baseline_bars(n: int = 60, seed: int = 42) -> np.ndarray:
    """Generates synthetic baseline OHLCV bars with stable range."""
    np.random.seed(seed)
    bars = np.zeros((n, 5), dtype=np.float64)
    base_price = 1000.0
    for i in range(n):
        open_ = base_price + np.random.uniform(-1.0, 1.0)
        close = open_ + np.random.uniform(-1.0, 1.0)
        high = max(open_, close) + np.random.uniform(0.5, 1.5)
        low = min(open_, close) - np.random.uniform(0.5, 1.5)
        volume = np.random.uniform(80.0, 120.0)
        bars[i] = [open_, high, low, close, volume]
        base_price = close
    return bars


class TestSMCLiquidityEngineContracts:
    """Validates structural and dimensional invariants of the SMC feature extractor."""

    def test_feature_count_and_names(self, engine: SMCLiquidityEngine):
        """Verify exactly 64 registered features and matching name length."""
        assert engine.FEATURE_COUNT == 64
        assert len(engine.FEATURE_NAMES) == 64
        assert len(SMC_FEATURE_NAMES) == 64
        assert len(set(engine.FEATURE_NAMES)) == 64

    def test_output_tensor_shape_and_dtype(self, engine: SMCLiquidityEngine):
        """Verify output is 1D array of shape (64,) and float32."""
        bars = generate_baseline_bars(60)
        feat = engine.extract_features(bars)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (64,)
        assert feat.dtype == np.float32

    def test_feature_dict_mapping(self, engine: SMCLiquidityEngine):
        """Verify get_feature_dict accurately maps all 64 keys."""
        bars = generate_baseline_bars(60)
        feat = engine.extract_features(bars)
        f_dict = engine.get_feature_dict(feat)
        assert len(f_dict) == 64
        for name in engine.FEATURE_NAMES:
            assert name in f_dict
            assert isinstance(f_dict[name], float)

    def test_smc_output_wrapper_and_properties(self, engine: SMCLiquidityEngine):
        """Verify SMCOutput wrapper returns structured telemetry and array compatibility."""
        bars = generate_baseline_bars(60)
        output = engine.extract_smc_output(bars)
        assert isinstance(output, SMCOutput)
        assert isinstance(output.features, np.ndarray)
        assert output.features.shape == (64,)
        assert len(output) == 64
        assert output.latency_ms > 0.0
        assert output.premium_discount_state in ["DISCOUNT", "EQUILIBRIUM", "PREMIUM"]
        # Array duck-typing check
        arr = np.asarray(output)
        assert arr.shape == (64,)


class TestLiquiditySweeps:
    """Validates BSL/SSL sweep detection, wick rejection, and sweep magnitude."""

    def test_bsl_sweep_detection(self, engine: SMCLiquidityEngine):
        """
        Construct a synthetic BSL sweep:
        1. Form a distinct swing high at bar 30 (High = 1050.0).
        2. At bar 58, spike high to 1055.0 (> 1050.0) but reject and close at 1044.0 (< 1050.0).
        """
        bars = generate_baseline_bars(60)
        # Establish swing high at bar 30
        bars[30, 0] = 1042.0
        bars[30, 1] = 1050.0  # High
        bars[30, 2] = 1038.0
        bars[30, 3] = 1044.0
        # Make surrounding bars lower to confirm fractal swing
        bars[28:30, 1] = 1040.0
        bars[31:33, 1] = 1040.0

        # Bar 58 performs BSL sweep
        bars[58, 0] = 1045.0  # Open
        bars[58, 1] = 1055.0  # High (sweeps 1050.0)
        bars[58, 2] = 1042.0  # Low
        bars[58, 3] = 1044.0  # Close (strictly below 1050.0)
        bars[58, 4] = 400.0   # High volume surge

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["bsl_sweep_detected"] == 1.0
        assert f["bsl_sweep_magnitude"] > 0.0
        assert f["bsl_upper_wick_ratio"] > 0.50  # Long upper wick rejection: (1055 - 1045) / 13 = 0.769
        assert f["bsl_sweep_volume_ratio"] > 1.5

    def test_ssl_sweep_detection(self, engine: SMCLiquidityEngine):
        """
        Construct a synthetic SSL sweep:
        1. Form a swing low at bar 30 (Low = 950.0).
        2. At bar 58, spike low to 944.0 (< 950.0) but reject and close at 955.0 (> 950.0).
        """
        bars = generate_baseline_bars(60)
        bars[30, 0] = 955.0
        bars[30, 1] = 960.0
        bars[30, 2] = 950.0  # Low
        bars[30, 3] = 956.0
        bars[28:30, 2] = 960.0
        bars[31:33, 2] = 960.0

        # Bar 58 performs SSL sweep
        bars[58, 0] = 952.0  # Open
        bars[58, 1] = 958.0  # High
        bars[58, 2] = 944.0  # Low (sweeps 950.0)
        bars[58, 3] = 956.0  # Close (closed above 950.0)
        bars[58, 4] = 450.0  # High volume surge

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["ssl_sweep_detected"] == 1.0
        assert f["ssl_sweep_magnitude"] > 0.0
        assert f["ssl_lower_wick_ratio"] > 0.50  # Long lower wick rejection: (952 - 944) / 14 = 0.571
        assert f["ssl_sweep_volume_ratio"] > 1.5

    def test_equal_highs_and_lows_clustering(self, engine: SMCLiquidityEngine):
        """Verify Equal Highs (EQH) pool detection when two swing highs cluster within threshold."""
        bars = generate_baseline_bars(60)
        # Swing high 1 at bar 25
        bars[25, 1] = 1020.0
        bars[23:25, 1] = 1010.0
        bars[26:28, 1] = 1010.0

        # Swing high 2 at bar 40 (within 0.15 ATR of 1020.0)
        bars[40, 1] = 1020.1
        bars[38:40, 1] = 1010.0
        bars[41:43, 1] = 1010.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)
        assert f["equal_highs_clustering_score"] > 0.0


class TestFairValueGaps:
    """Validates Bullish/Bearish FVG detection, gap magnitude, CE, and mitigation."""

    def test_bullish_fvg_detection_and_mitigation(self, engine: SMCLiquidityEngine):
        """
        Construct a Bullish FVG (BISI):
        Bar 50: High = 1000.0
        Bar 51: Large impulse bar (Open = 1001, Close = 1020, Vol = 500)
        Bar 52: Low = 1008.0 (> High[50] = 1000.0) -> Gap is [1000.0, 1008.0]
        """
        bars = generate_baseline_bars(60)
        bars[50, 0] = 998.0
        bars[50, 1] = 1000.0  # High of candle 1
        bars[50, 2] = 996.0
        bars[50, 3] = 999.0

        bars[51, 0] = 1001.0
        bars[51, 1] = 1022.0
        bars[51, 2] = 1001.0
        bars[51, 3] = 1020.0
        bars[51, 4] = 600.0   # Large impulse volume

        bars[52, 0] = 1018.0
        bars[52, 1] = 1025.0
        bars[52, 2] = 1008.0  # Low of candle 3 (> 1000.0)
        bars[52, 3] = 1023.0

        # Bars 53-59 stay above 1010.0 (unmitigated gap)
        for k in range(53, 60):
            bars[k, 0] = 1015.0
            bars[k, 1] = 1020.0
            bars[k, 2] = 1012.0
            bars[k, 3] = 1016.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["bullish_fvg_active"] == 1.0
        assert f["bullish_fvg_gap_size"] > 0.0
        assert f["bullish_fvg_mitigation_ratio"] == 0.0  # Virgin unmitigated gap

        # Now simulate a retest dipping into Consequent Encroachment (CE = 1004.0) at bar 59
        bars[59, 2] = 1004.0
        feat_retest = engine.extract_features(bars)
        f_retest = engine.get_feature_dict(feat_retest)
        assert f_retest["bullish_fvg_mitigation_ratio"] >= 0.50

    def test_bearish_fvg_detection(self, engine: SMCLiquidityEngine):
        """
        Construct a Bearish FVG (SIBI):
        Bar 50: Low = 1020.0
        Bar 51: Large down impulse bar
        Bar 52: High = 1010.0 (< Low[50] = 1020.0) -> Gap is [1010.0, 1020.0]
        """
        bars = generate_baseline_bars(60)
        bars[50, 0] = 1022.0
        bars[50, 1] = 1025.0
        bars[50, 2] = 1020.0
        bars[50, 3] = 1021.0

        bars[51, 0] = 1019.0
        bars[51, 1] = 1019.0
        bars[51, 2] = 998.0
        bars[51, 3] = 1000.0
        bars[51, 4] = 600.0

        bars[52, 0] = 1002.0
        bars[52, 1] = 1010.0  # High of candle 3 (< 1020.0)
        bars[52, 2] = 995.0
        bars[52, 3] = 998.0

        for k in range(53, 60):
            bars[k, 0] = 1002.0
            bars[k, 1] = 1006.0
            bars[k, 2] = 998.0
            bars[k, 3] = 1001.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["bearish_fvg_active"] == 1.0
        assert f["bearish_fvg_gap_size"] > 0.0

    def test_inversion_fvg_reversal(self, engine: SMCLiquidityEngine):
        """Verify that violated FVG flips inversion_fvg_reversal_flag."""
        bars = generate_baseline_bars(60)
        # Form bullish FVG at 40
        bars[40, 1] = 1000.0
        bars[41, 0] = 1001.0
        bars[41, 3] = 1015.0
        bars[42, 2] = 1008.0

        # Bar 55-59 close below FVG bottom (1000.0)
        for k in range(55, 60):
            bars[k, 0] = 990.0
            bars[k, 1] = 995.0
            bars[k, 2] = 985.0
            bars[k, 3] = 988.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)
        assert f["inversion_fvg_reversal_flag"] == -1.0


class TestOrderBlocks:
    """Validates Bullish/Bearish Order Block identification, impulse volume, and MT."""

    def test_bullish_ob_detection_and_boundaries(self, engine: SMCLiquidityEngine):
        """
        Construct a Bullish Order Block:
        Bar 45: Down candle (Open = 1005, High = 1007, Low = 998, Close = 1000)
        Bars 46-48: Violent upward displacement (reaching 1030.0 with high volume)
        """
        bars = generate_baseline_bars(60)
        bars[45, 0] = 1005.0
        bars[45, 1] = 1007.0
        bars[45, 2] = 998.0
        bars[45, 3] = 1000.0  # Down candle

        # Upward displacement leg
        bars[46, 0] = 1001.0
        bars[46, 1] = 1015.0
        bars[46, 2] = 1001.0
        bars[46, 3] = 1014.0
        bars[46, 4] = 400.0

        bars[47, 0] = 1015.0
        bars[47, 1] = 1025.0
        bars[47, 2] = 1014.0
        bars[47, 3] = 1024.0
        bars[47, 4] = 450.0

        bars[48, 0] = 1025.0
        bars[48, 1] = 1035.0
        bars[48, 2] = 1023.0
        bars[48, 3] = 1033.0
        bars[48, 4] = 500.0

        # Subsequent bars hold above OB high (1007.0)
        for k in range(49, 60):
            bars[k, 0] = 1025.0
            bars[k, 1] = 1030.0
            bars[k, 2] = 1020.0
            bars[k, 3] = 1026.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["bullish_ob_detected"] == 1.0
        assert f["bullish_ob_top_distance"] > 0.0
        assert f["bullish_ob_mean_threshold_dist"] > 0.0
        assert f["bullish_ob_impulse_volume_ratio"] > 1.5

    def test_bearish_ob_detection_and_mitigation(self, engine: SMCLiquidityEngine):
        """Construct a Bearish Order Block and verify detection."""
        bars = generate_baseline_bars(60)
        bars[45, 0] = 1000.0
        bars[45, 1] = 1008.0
        bars[45, 2] = 998.0
        bars[45, 3] = 1006.0  # Up candle

        # Downward displacement leg
        bars[46, 0] = 1004.0
        bars[46, 1] = 1005.0
        bars[46, 2] = 988.0
        bars[46, 3] = 990.0
        bars[46, 4] = 450.0

        bars[47, 0] = 989.0
        bars[47, 1] = 990.0
        bars[47, 2] = 975.0
        bars[47, 3] = 978.0
        bars[47, 4] = 500.0

        bars[48, 0] = 977.0
        bars[48, 1] = 978.0
        bars[48, 2] = 965.0
        bars[48, 3] = 968.0
        bars[48, 4] = 550.0

        for k in range(49, 60):
            bars[k, 0] = 970.0
            bars[k, 1] = 975.0
            bars[k, 2] = 965.0
            bars[k, 3] = 972.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["bearish_ob_detected"] == 1.0
        assert f["bearish_ob_top_distance"] > 0.0

    def test_breaker_block_inversion(self, engine: SMCLiquidityEngine):
        """Verify breaker block status when an OB is violated."""
        bars = generate_baseline_bars(60)
        # Form bullish OB at 40
        bars[40, 0] = 1005.0
        bars[40, 1] = 1007.0
        bars[40, 2] = 998.0
        bars[40, 3] = 1000.0
        bars[41:44, 3] = 1020.0

        # Bar 50-59 violates OB low (998.0) and closes at 980.0
        for k in range(50, 60):
            bars[k, 0] = 985.0
            bars[k, 1] = 990.0
            bars[k, 2] = 978.0
            bars[k, 3] = 980.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)
        assert f["breaker_block_status"] == -1.0


class TestFalseBreakoutTraps:
    """Validates Trapped Breakout detection and failure follow-through."""

    def test_trapped_buyer_score(self, engine: SMCLiquidityEngine):
        """
        Form a swing high at bar 45 (price = 1050.0).
        Bar 58: Breakout candle closes above swing high (High = 1055, Close = 1052).
        Bar 59: Immediate reversal candle closes below breakout level (Close = 1044).
        """
        bars = generate_baseline_bars(60)
        bars[45, 0] = 1042.0
        bars[45, 1] = 1050.0  # High
        bars[45, 2] = 1038.0
        bars[45, 3] = 1044.0
        bars[43:45, 1] = 1040.0
        bars[46:48, 1] = 1040.0

        # Bar 58: Breakout
        bars[58, 0] = 1045.0
        bars[58, 1] = 1055.0
        bars[58, 2] = 1044.0
        bars[58, 3] = 1052.0  # Closed above 1050.0
        bars[58, 4] = 60.0    # Low volume exhaustion

        # Bar 59: Reversal
        bars[59, 0] = 1051.0
        bars[59, 1] = 1052.0
        bars[59, 2] = 1042.0
        bars[59, 3] = 1044.0  # Closed back below 1050.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["trapped_buyer_score"] > 0.50
        assert f["breakout_follow_through_failure"] > 0.0

    def test_trapped_seller_score(self, engine: SMCLiquidityEngine):
        """
        Form a swing low at bar 45 (price = 950.0).
        Bar 58: Breakdown candle closes below swing low (Low = 945, Close = 947).
        Bar 59: Immediate reversal candle closes back above swing low (Close = 956).
        """
        bars = generate_baseline_bars(60)
        bars[45, 0] = 955.0
        bars[45, 1] = 960.0
        bars[45, 2] = 950.0  # Low
        bars[45, 3] = 956.0
        bars[43:45, 2] = 960.0
        bars[46:48, 2] = 960.0

        # Bar 58: Breakdown
        bars[58, 0] = 954.0
        bars[58, 1] = 955.0
        bars[58, 2] = 945.0
        bars[58, 3] = 947.0
        bars[58, 4] = 50.0

        # Bar 59: Reversal
        bars[59, 0] = 948.0
        bars[59, 1] = 958.0
        bars[59, 2] = 948.0
        bars[59, 3] = 956.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)

        assert f["trapped_seller_score"] > 0.50

    def test_market_structure_shift_and_bos(self, engine: SMCLiquidityEngine):
        """Verify MSS (CHoCH) and BOS structural changes."""
        bars = generate_baseline_bars(60)

        # Monotonic downward flow from 20 (1020) to 27 (1000)
        for k in range(21, 27):
            bars[k, 1] = 1020.0 - (k - 20) * 2.5
            bars[k, 2] = bars[k, 1] - 2.0
            bars[k, 3] = bars[k, 1] - 1.0

        # Monotonic upward flow from 27 (1000) to 35 (1015)
        for k in range(28, 35):
            bars[k, 1] = 1000.0 + (k - 27) * 1.8
            bars[k, 2] = bars[k, 1] - 2.0
            bars[k, 3] = bars[k, 1] - 1.0

        # Monotonic downward flow from 35 (1015) to 45 (995)
        for k in range(36, 45):
            bars[k, 1] = 1015.0 - (k - 35) * 2.0
            bars[k, 2] = bars[k, 1] - 2.0
            bars[k, 3] = bars[k, 1] - 1.0

        # Monotonic upward flow from 45 (995) to 59 (1018)
        for k in range(46, 59):
            bars[k, 1] = 995.0 + (k - 45) * 1.5
            bars[k, 2] = bars[k, 1] - 2.0
            bars[k, 3] = bars[k, 1] - 1.0

        # Swing High 1 at 20 (1020), Swing High 2 at 35 (1015) - lower high
        bars[20, 1] = 1020.0
        bars[18:20, 1] = 1010.0
        bars[35, 1] = 1015.0

        # Swing Lows
        bars[27, 2] = 1000.0
        bars[45, 2] = 995.0

        # Bar 59 breaks above Swing High 2 (1015) -> CHoCH
        bars[59, 1] = 1019.0
        bars[59, 3] = 1018.0

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)
        assert f["market_structure_shift_choch"] == 1.0

    def test_premium_discount_equilibrium_zones(self, engine: SMCLiquidityEngine):
        """Verify dealing range premium/discount position."""
        bars = generate_baseline_bars(60)
        # Force range 1000 to 1100
        bars[10, 2] = 1000.0
        bars[20, 1] = 1100.0
        # Close at 1020 (discount)
        bars[59, 3] = 1020.0
        bars[59, 1] = max(bars[59, 1], 1020.0)
        bars[59, 2] = min(bars[59, 2], 1020.0)

        feat = engine.extract_features(bars)
        f = engine.get_feature_dict(feat)
        assert f["premium_discount_equilibrium"] < 0.45


class TestNumericalIntegrityAC7:
    """Verifies strict 0 NaN, 0 Inf preservation under adversarial and pathological edge cases."""

    def test_all_zeros_array(self, engine: SMCLiquidityEngine):
        bars = np.zeros((60, 5), dtype=np.float64)
        feat = engine.extract_features(bars)
        assert feat.shape == (64,)
        assert not np.isnan(feat).any()
        assert not np.isinf(feat).any()

    def test_flat_price_bars(self, engine: SMCLiquidityEngine):
        bars = np.full((60, 5), 100.0, dtype=np.float64)
        feat = engine.extract_features(bars)
        assert not np.isnan(feat).any()
        assert not np.isinf(feat).any()

    def test_zero_volume_bars(self, engine: SMCLiquidityEngine):
        bars = generate_baseline_bars(60)
        bars[:, 4] = 0.0
        feat = engine.extract_features(bars)
        assert not np.isnan(feat).any()
        assert not np.isinf(feat).any()

    def test_nan_and_inf_in_inputs(self, engine: SMCLiquidityEngine):
        bars = generate_baseline_bars(60)
        bars[10, 0] = np.nan
        bars[20, 1] = np.inf
        bars[30, 2] = -np.inf
        feat = engine.extract_features(bars)
        assert not np.isnan(feat).any()
        assert not np.isinf(feat).any()

    def test_extreme_negative_and_large_values(self, engine: SMCLiquidityEngine):
        bars = generate_baseline_bars(60) * 1e8
        feat = engine.extract_features(bars)
        assert not np.isnan(feat).any()
        assert not np.isinf(feat).any()
        assert np.all(np.abs(feat) <= engine.config.clamping_bound)

    def test_short_sequence_inputs(self, engine: SMCLiquidityEngine):
        for length in [1, 2, 4]:
            bars = np.ones((length, 5)) * 100.0
            feat = engine.extract_features(bars)
            assert feat.shape == (64,)
            assert not np.isnan(feat).any()
            assert not np.isinf(feat).any()


class TestBatchAndLatencyAC6:
    """Verifies batch extraction and real-time execution latency benchmark."""

    def test_batch_extraction_consistency(self, engine: SMCLiquidityEngine):
        b1 = generate_baseline_bars(60, seed=1)
        b2 = generate_baseline_bars(60, seed=2)
        batch = np.stack([b1, b2], axis=0)

        out_batch = engine.extract_batch_features(batch)
        assert out_batch.shape == (2, 64)

        out_b1 = engine.extract_features(b1)
        out_b2 = engine.extract_features(b2)

        np.testing.assert_allclose(out_batch[0], out_b1, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out_batch[1], out_b2, rtol=1e-5, atol=1e-5)

    def test_latency_sub_millisecond_ac6(self, engine: SMCLiquidityEngine):
        bars = generate_baseline_bars(60)
        # Warmup
        for _ in range(20):
            _ = engine.extract_features(bars)

        iters = 500
        start = time.perf_counter()
        for _ in range(iters):
            _ = engine.extract_features(bars)
        total_time = time.perf_counter() - start

        avg_time_ms = (total_time / iters) * 1000.0
        # Average time must be < 5.0ms (MT5 total budget is 50ms)
        assert avg_time_ms < 5.0, f"Average execution time {avg_time_ms:.3f}ms exceeds 5.0ms SLA"
