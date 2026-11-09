"""
Module R2: Smart Money Concepts (SMC) & ICT Liquidity Engine
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine
Milestone: M1 (Deep Order Flow & Liquidity Engine)

This module extracts institutional Smart Money Concepts (SMC) and ICT features
from market price action, identifying:
1. Liquidity Sweeps & Stop Hunts (BSL/SSL runs, wick exhaustion, sweep magnitude)
2. Fair Value Gaps (FVG) & Imbalance Zones (BISI/SIBI detection, CE retest, mitigation)
3. Institutional Order Blocks (OB) (Displacement origin, impulse volume, boundaries, MT)
4. False Breakout Traps & Structure (Trapped retail, CHoCH/MSS, equilibrium discount/premium)

Guarantees:
- Exactly 64 float features returned as a deterministic 1D array of shape (64,).
- Absolute 0 NaN, 0 Inf arithmetic preservation (AC-7) via strict epsilon clamping and nan_to_num.
- Ultra-low latency execution (< 1ms), satisfying the <= 50ms MT5 latency budget (AC-6).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class SMCConfig:
    """Configuration parameters for SMC / ICT Liquidity Engine."""
    swing_lookback: int = 2              # Window order for fractal swing high/low detection
    atr_period: int = 14                 # Period for Average True Range normalization
    volume_baseline_period: int = 20     # Lookback for rolling average volume baseline
    min_fvg_atr_ratio: float = 0.15      # Minimum gap size relative to ATR to qualify as FVG
    ob_impulse_bars: int = 3             # Number of departure bars to evaluate displacement
    eq_threshold_atr_ratio: float = 0.15 # Max distance between swing points to form Equal Highs/Lows
    eps: float = 1e-12                   # Strict floating-point safeguard for zero-division
    clamping_bound: float = 100.0        # Maximum absolute bound for normalized features
    lookback_window: int = 60            # Recommended input bar buffer length
    swing_pivot_len: int = 5             # Pivot window length
    ob_displacement_atr_mult: float = 1.50
    ob_volume_mult: float = 1.40
    trap_exhaustion_wick_ratio: float = 0.40


@dataclass
class SMCOutput:
    """Structured container for detailed SMC diagnostics and telemetry."""
    features: np.ndarray                 # Shape: (64,), dtype: np.float32
    bsl_swept: bool
    ssl_swept: bool
    active_bullish_fvgs: List[Dict[str, Any]] = field(default_factory=list)
    active_bearish_fvgs: List[Dict[str, Any]] = field(default_factory=list)
    active_bullish_obs: List[Dict[str, Any]] = field(default_factory=list)
    active_bearish_obs: List[Dict[str, Any]] = field(default_factory=list)
    trap_detected: bool = False
    premium_discount_state: str = "EQUILIBRIUM"  # 'DISCOUNT', 'EQUILIBRIUM', 'PREMIUM'
    latency_ms: float = 0.0

    def __array__(self, dtype=None):
        if dtype is None:
            return self.features
        return self.features.astype(dtype)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# Feature Registry: 64 institutional float features (4 groups x 16 features)
SMC_FEATURE_NAMES: List[str] = [
    # Group 1: Liquidity Sweeps & Stop Hunts (Features 0 - 15)
    "bsl_sweep_detected",              # 0: 1.0 if BSL sweep detected in lookback, else 0.0
    "ssl_sweep_detected",              # 1: 1.0 if SSL sweep detected in lookback, else 0.0
    "bsl_sweep_magnitude",             # 2: (High - SwingHigh) / ATR for latest BSL sweep
    "ssl_sweep_magnitude",             # 3: (SwingLow - Low) / ATR for latest SSL sweep
    "bsl_upper_wick_ratio",            # 4: Upper wick rejection ratio on BSL sweep candle [0, 1]
    "ssl_lower_wick_ratio",            # 5: Lower wick rejection ratio on SSL sweep candle [0, 1]
    "bsl_sweep_volume_ratio",          # 6: Volume on BSL sweep candle relative to 20-bar baseline
    "ssl_sweep_volume_ratio",          # 7: Volume on SSL sweep candle relative to 20-bar baseline
    "bsl_reversal_displacement",       # 8: Normalized bearish displacement following BSL sweep [-1, 1]
    "ssl_reversal_displacement",       # 9: Normalized bullish displacement following SSL sweep [-1, 1]
    "bars_since_bsl_sweep",            # 10: Normalized recency of BSL sweep in [0, 1]
    "bars_since_ssl_sweep",            # 11: Normalized recency of SSL sweep in [0, 1]
    "distance_to_nearest_bsl",         # 12: Normalized distance from current Close to active BSL pool
    "distance_to_nearest_ssl",         # 13: Normalized distance from current Close to active SSL pool
    "equal_highs_clustering_score",    # 14: Density score for resting liquidity pool above (EQH) [0, 1]
    "equal_lows_clustering_score",     # 15: Density score for resting liquidity pool below (EQL) [0, 1]

    # Group 2: Fair Value Gaps (FVG) / Imbalances (Features 16 - 31)
    "bullish_fvg_active",              # 16: 1.0 if unmitigated bullish FVG exists, else 0.0
    "bearish_fvg_active",              # 17: 1.0 if unmitigated bearish FVG exists, else 0.0
    "bullish_fvg_gap_size",            # 18: Size of most recent active bullish FVG / ATR
    "bearish_fvg_gap_size",            # 19: Size of most recent active bearish FVG / ATR
    "bullish_fvg_ce_distance",         # 20: Normalized distance to Consequent Encroachment (50% midpoint)
    "bearish_fvg_ce_distance",         # 21: Normalized distance to Consequent Encroachment (50% midpoint)
    "bullish_fvg_mitigation_ratio",    # 22: Fraction of bullish FVG filled by subsequent price action [0, 1]
    "bearish_fvg_mitigation_ratio",    # 23: Fraction of bearish FVG filled by subsequent price action [0, 1]
    "price_inside_bullish_fvg",        # 24: 1.0 if current Close is inside bullish FVG zone, else 0.0
    "price_inside_bearish_fvg",        # 25: 1.0 if current Close is inside bearish FVG zone, else 0.0
    "bullish_fvg_impulse_volume_ratio",# 26: Volume ratio of central displacement bar creating bullish FVG
    "bearish_fvg_impulse_volume_ratio",# 27: Volume ratio of central displacement bar creating bearish FVG
    "bars_since_bullish_fvg",          # 28: Normalized recency of active bullish FVG in [0, 1]
    "bars_since_bearish_fvg",          # 29: Normalized recency of active bearish FVG in [0, 1]
    "net_unmitigated_fvg_volume_bias", # 30: Aggregate bullish vs bearish unmitigated FVG volume bias [-1, 1]
    "inversion_fvg_reversal_flag",     # 31: Inversion FVG flipped support/resistance signal [-1, 1]

    # Group 3: Institutional Order Blocks (OB) & Impulse Anchors (Features 32 - 47)
    "bullish_ob_detected",             # 32: 1.0 if active Bullish Order Block identified, else 0.0
    "bearish_ob_detected",             # 33: 1.0 if active Bearish Order Block identified, else 0.0
    "bullish_ob_top_distance",         # 34: Normalized distance from current Close to Bullish OB top
    "bullish_ob_bottom_distance",      # 35: Normalized distance from current Close to Bullish OB invalidation
    "bullish_ob_mean_threshold_dist",  # 36: Normalized distance from current Close to Bullish OB MT (50%)
    "bearish_ob_top_distance",         # 37: Normalized distance from Bearish OB invalidation to current Close
    "bearish_ob_bottom_distance",      # 38: Normalized distance from Bearish OB bottom to current Close
    "bearish_ob_mean_threshold_dist",  # 39: Normalized distance from Bearish OB MT (50%) to current Close
    "bullish_ob_impulse_volume_ratio", # 40: Volume ratio of displacement leg departing Bullish OB
    "bearish_ob_impulse_volume_ratio", # 41: Volume ratio of displacement leg departing Bearish OB
    "bullish_ob_mitigation_state",     # 42: Mitigation depth of Bullish OB: 0=fresh, 0.5=MT tested, 1=broken
    "bearish_ob_mitigation_state",     # 43: Mitigation depth of Bearish OB: 0=fresh, 0.5=MT tested, 1=broken
    "bars_since_bullish_ob",           # 44: Normalized recency of active Bullish OB in [0, 1]
    "bars_since_bearish_ob",           # 45: Normalized recency of active Bearish OB in [0, 1]
    "breaker_block_status",            # 46: +1.0 for Bullish Breaker, -1.0 for Bearish Breaker, 0.0 neutral
    "ob_displacement_momentum",        # 47: Normalized velocity of displacement move originating from OB

    # Group 4: False Breakout Traps & Market Structure (Features 48 - 63)
    "trapped_buyer_score",             # 48: Quantitative score of trapped breakout buyers [0, 1]
    "trapped_seller_score",            # 49: Quantitative score of trapped breakdown sellers [0, 1]
    "breakout_follow_through_failure", # 50: Failure ratio of breakout expansion post-breakout [0, 1]
    "breakout_volume_exhaustion",      # 51: Low volume anomaly during price breakout [0, 1]
    "judas_swing_trap_indicator",      # 52: Counter-trend fakeout indicator [-1, 1]
    "market_structure_shift_choch",    # 53: Change of Character (MSS/CHoCH) signal: +1 bull, -1 bear, 0 none
    "bos_trend_alignment",             # 54: Break of Structure (BOS) continuation: +1 bull, -1 bear, 0 none
    "displacement_quality_score",      # 55: Body-to-total-range ratio of recent directional candle [0, 1]
    "premium_discount_equilibrium",    # 56: Position in internal range: 0=discount, 0.5=eq, 1=premium [0, 1]
    "inducement_trap_proximity",       # 57: Proximity to retail inducement swing point [0, 1]
    "sweep_trap_divergence",           # 58: Divergence between breakout price and sweep confirmation [-1, 1]
    "liquidity_void_expansion_rate",   # 59: Rapid one-sided expansion rate signaling mean-reversion risk [0, 5]
    "wick_to_body_exhaustion_ratio",   # 60: Total wick to body ratio on latest bar [0, 10]
    "internal_range_liquidity_bias",   # 61: Relative momentum bias within internal range [-1, 1]
    "external_range_liquidity_run_potential", # 62: Attraction potential toward external swing highs/lows [0, 1]
    "smc_composite_regime_score",      # 63: Aggregated institutional bias score [-1, 1] (+1 buy, -1 sell)
]

assert len(SMC_FEATURE_NAMES) == 64, f"Registry must have exactly 64 features, got {len(SMC_FEATURE_NAMES)}"


class SMCLiquidityEngine:
    """
    R2 Engine: Deep Order Flow & Smart Money Concepts (SMC) Liquidity Extractor.
    Extracts institutional-grade features (sweeps, FVGs, OBs, traps) from OHLCV arrays.
    """

    FEATURE_COUNT: int = 64
    FEATURE_NAMES: List[str] = SMC_FEATURE_NAMES

    def __init__(self, config: Optional[SMCConfig] = None) -> None:
        self.config: SMCConfig = config or SMCConfig()

    def extract_features(
        self,
        bars: Union[np.ndarray, List[Any]],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extracts exactly 64 float features from OHLCV bar array.

        Parameters
        ----------
        bars : np.ndarray or list of BarData
            Input bars of shape (N, 5) or (N, 6) where columns correspond to:
            [Open, High, Low, Close, Volume, (Timestamp)].
        symbol : Optional[str]
            Symbol name for logging or symbol-specific calibration.
        timeframe : Optional[str]
            Timeframe identifier (e.g. 'M1', 'M5').

        Returns
        -------
        np.ndarray
            1D array of length 64 with dtype float32.
            Guaranteed: zero NaN, zero Inf, all values finite.
        """
        arr = self._to_numpy(bars)
        if arr.shape[0] < 5:
            return np.zeros(self.FEATURE_COUNT, dtype=np.float32)

        eps = self.config.eps
        n = arr.shape[0]

        # Sanitization: forward-fill / buffer median to prevent scale distortion across symbols
        if not np.isfinite(arr).all():
            arr = arr.copy()
            price_cols = min(arr.shape[1], 4)
            finite_mask_prices = np.isfinite(arr[:, :price_cols])
            buffer_median = (
                float(np.median(arr[:, :price_cols][finite_mask_prices]))
                if finite_mask_prices.any()
                else 100.0
            )
            for col_idx in range(arr.shape[1]):
                col = arr[:, col_idx]
                finite_mask = np.isfinite(col)
                if finite_mask.all():
                    continue
                if not finite_mask.any():
                    col[:] = buffer_median if col_idx < 4 else 1.0
                else:
                    first_valid = col[finite_mask][0]
                    last_val = first_valid
                    for i in range(n):
                        if finite_mask[i]:
                            last_val = col[i]
                        else:
                            col[i] = last_val

        # Extract price vectors
        o = arr[:, 0]
        h = arr[:, 1]
        l = arr[:, 2]
        c = arr[:, 3]
        v = arr[:, 4] if arr.shape[1] > 4 else np.ones(n, dtype=np.float64)

        # Baseline metrics: ATR and Volume SMA
        atr = self._calculate_atr(h, l, c, self.config.atr_period, eps)
        vol_baseline = self._calculate_vol_baseline(v, self.config.volume_baseline_period, eps)

        # Find fractal swing highs and lows
        swing_highs, swing_lows = self._find_swings(h, l, self.config.swing_lookback)

        # Initialize feature container
        feat = np.zeros(self.FEATURE_COUNT, dtype=np.float64)

        # -------------------------------------------------------------
        # Group 1: Liquidity Sweeps & Stop Hunts (Features 0 - 15)
        # -------------------------------------------------------------
        self._extract_sweeps(
            feat=feat,
            o=o, h=h, l=l, c=c, v=v,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            atr=atr,
            vol_baseline=vol_baseline,
            eps=eps,
        )

        # -------------------------------------------------------------
        # Group 2: Fair Value Gaps (FVG) / Imbalances (Features 16 - 31)
        # -------------------------------------------------------------
        self._extract_fvgs(
            feat=feat,
            o=o, h=h, l=l, c=c, v=v,
            atr=atr,
            vol_baseline=vol_baseline,
            eps=eps,
        )

        # -------------------------------------------------------------
        # Group 3: Institutional Order Blocks (OB) (Features 32 - 47)
        # -------------------------------------------------------------
        self._extract_order_blocks(
            feat=feat,
            o=o, h=h, l=l, c=c, v=v,
            atr=atr,
            vol_baseline=vol_baseline,
            eps=eps,
        )

        # -------------------------------------------------------------
        # Group 4: False Breakout Traps & Market Structure (Features 48 - 63)
        # -------------------------------------------------------------
        self._extract_traps_and_structure(
            feat=feat,
            o=o, h=h, l=l, c=c, v=v,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            atr=atr,
            vol_baseline=vol_baseline,
            eps=eps,
        )

        # Arithmetic Preservation (AC-7): strictly zero NaN, zero Inf, clamp
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        bound = self.config.clamping_bound
        feat = np.clip(feat, -bound, bound)

        return feat.astype(np.float32)

    def extract_smc_output(
        self,
        bars: Union[np.ndarray, List[Any]],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> SMCOutput:
        """
        Extract features and wrap with detailed diagnostic metadata into SMCOutput.
        """
        t0 = time.perf_counter()
        features = self.extract_features(bars, symbol=symbol, timeframe=timeframe)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        bsl_swept = bool(features[0] > 0.5)
        ssl_swept = bool(features[1] > 0.5)
        trap_detected = bool(features[48] > 0.5 or features[49] > 0.5)

        eq_pos = float(features[56])
        if eq_pos < 0.45:
            p_state = "DISCOUNT"
        elif eq_pos > 0.55:
            p_state = "PREMIUM"
        else:
            p_state = "EQUILIBRIUM"

        return SMCOutput(
            features=features,
            bsl_swept=bsl_swept,
            ssl_swept=ssl_swept,
            trap_detected=trap_detected,
            premium_discount_state=p_state,
            latency_ms=latency_ms,
        )

    def extract_batch_features(
        self,
        bars_batch: Union[np.ndarray, List[Any]],
    ) -> np.ndarray:
        """Batch feature extraction: (B, N, 5) -> (B, 64)."""
        if isinstance(bars_batch, np.ndarray) and bars_batch.ndim == 3:
            b, n, _ = bars_batch.shape
            out = np.zeros((b, self.FEATURE_COUNT), dtype=np.float32)
            for i in range(b):
                out[i] = self.extract_features(bars_batch[i])
            return out
        elif isinstance(bars_batch, list):
            b = len(bars_batch)
            out = np.zeros((b, self.FEATURE_COUNT), dtype=np.float32)
            for i in range(b):
                out[i] = self.extract_features(bars_batch[i])
            return out
        else:
            raise ValueError(f"Unsupported batch shape or type: {type(bars_batch)}")

    def get_feature_dict(self, features: np.ndarray) -> Dict[str, float]:
        """Maps 64-D feature vector to named dictionary for telemetry and inspection."""
        if features.shape[0] != self.FEATURE_COUNT:
            raise ValueError(f"Expected array of length {self.FEATURE_COUNT}, got {features.shape[0]}")
        return {name: float(val) for name, val in zip(self.FEATURE_NAMES, features)}

    # -----------------------------------------------------------------
    # Private Computation Helpers
    # -----------------------------------------------------------------

    def _to_numpy(self, bars: Union[np.ndarray, List[Any]]) -> np.ndarray:
        if isinstance(bars, np.ndarray):
            return bars
        if isinstance(bars, list):
            if not bars:
                return np.zeros((0, 5), dtype=np.float64)
            first = bars[0]
            if hasattr(first, "open") and hasattr(first, "high"):
                arr = np.zeros((len(bars), 5), dtype=np.float64)
                for i, b in enumerate(bars):
                    arr[i, 0] = float(b.open)
                    arr[i, 1] = float(b.high)
                    arr[i, 2] = float(b.low)
                    arr[i, 3] = float(b.close)
                    arr[i, 4] = float(getattr(b, "volume", 1.0))
                return arr
            elif isinstance(first, (list, tuple)):
                return np.array(bars, dtype=np.float64)
        return np.array(bars, dtype=np.float64)

    def _calculate_atr(
        self,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        period: int,
        eps: float,
    ) -> float:
        n = len(h)
        if n < 2:
            return max(float(h[-1] - l[-1]), eps) if n == 1 else eps
        tr0 = h[1:] - l[1:]
        tr1 = np.abs(h[1:] - c[:-1])
        tr2 = np.abs(l[1:] - c[:-1])
        tr = np.maximum(tr0, np.maximum(tr1, tr2))
        window = tr[-period:] if len(tr) >= period else tr
        val = float(np.mean(window)) if len(window) > 0 else eps
        return max(val, eps)

    def _calculate_vol_baseline(
        self,
        v: np.ndarray,
        period: int,
        eps: float,
    ) -> float:
        if len(v) == 0:
            return eps
        window = v[-period:] if len(v) >= period else v
        val = float(np.mean(window)) if len(window) > 0 else eps
        return max(val, eps)

    def _find_swings(
        self,
        h: np.ndarray,
        l: np.ndarray,
        lookback: int,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        n = len(h)
        swing_highs: List[Tuple[int, float]] = []
        swing_lows: List[Tuple[int, float]] = []
        if n < 2 * lookback + 1:
            return swing_highs, swing_lows

        h_list = h.tolist() if hasattr(h, "tolist") else list(h)
        l_list = l.tolist() if hasattr(l, "tolist") else list(l)

        if lookback == 2:
            for i in range(2, n - 1):
                val_h = h_list[i]
                if val_h >= h_list[i - 1] and val_h >= h_list[i - 2] and val_h >= h_list[i + 1]:
                    if i + 2 >= n or val_h >= h_list[i + 2]:
                        swing_highs.append((i, float(val_h)))
                val_l = l_list[i]
                if val_l <= l_list[i - 1] and val_l <= l_list[i - 2] and val_l <= l_list[i + 1]:
                    if i + 2 >= n or val_l <= l_list[i + 2]:
                        swing_lows.append((i, float(val_l)))
        else:
            for i in range(lookback, n - 1):
                val_h = h_list[i]
                val_l = l_list[i]

                is_high = True
                for k in range(i - lookback, i):
                    if val_h < h_list[k]:
                        is_high = False
                        break
                if is_high:
                    for k in range(i + 1, min(n, i + lookback + 1)):
                        if val_h < h_list[k]:
                            is_high = False
                            break
                if is_high:
                    swing_highs.append((i, float(val_h)))

                is_low = True
                for k in range(i - lookback, i):
                    if val_l > l_list[k]:
                        is_low = False
                        break
                if is_low:
                    for k in range(i + 1, min(n, i + lookback + 1)):
                        if val_l > l_list[k]:
                            is_low = False
                            break
                if is_low:
                    swing_lows.append((i, float(val_l)))

        return swing_highs, swing_lows

    # -----------------------------------------------------------------
    # Group Extraction Methods
    # -----------------------------------------------------------------

    def _extract_sweeps(
        self,
        feat: np.ndarray,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
        swing_highs: List[Tuple[int, float]],
        swing_lows: List[Tuple[int, float]],
        atr: float,
        vol_baseline: float,
        eps: float,
    ) -> None:
        """Extracts 16 Sweep features into feat[0:16]."""
        n = len(c)
        curr_c = c[-1]

        sweep_lookback = min(10, n)
        bsl_detected = 0.0
        ssl_detected = 0.0
        bsl_mag = 0.0
        ssl_mag = 0.0
        bsl_wick_ratio = 0.0
        ssl_wick_ratio = 0.0
        bsl_vol_ratio = 0.0
        ssl_vol_ratio = 0.0
        bsl_displacement = 0.0
        ssl_displacement = 0.0
        bars_since_bsl = 1.0
        bars_since_ssl = 1.0

        # Scan backwards from newest bar k to find the latest confirmed sweep
        if swing_highs:
            for k in range(n - 1, n - sweep_lookback - 1, -1):
                c_range = max(h[k] - l[k], eps)
                for sh_idx, sh_price in reversed(swing_highs):
                    if sh_idx < k and h[k] > sh_price and c[k] < sh_price:
                        bsl_detected = 1.0
                        bsl_mag = max(0.0, (h[k] - sh_price) / atr)
                        bsl_wick_ratio = max(0.0, min(1.0, (h[k] - max(o[k], c[k])) / c_range))
                        bsl_vol_ratio = min(10.0, v[k] / vol_baseline)
                        bsl_displacement = max(-1.0, min(1.0, (o[k] - c[k]) / c_range))
                        bars_since_bsl = (n - 1 - k) / float(sweep_lookback)
                        break
                if bsl_detected > 0.0:
                    break

        if swing_lows:
            for k in range(n - 1, n - sweep_lookback - 1, -1):
                c_range = max(h[k] - l[k], eps)
                for sl_idx, sl_price in reversed(swing_lows):
                    if sl_idx < k and l[k] < sl_price and c[k] > sl_price:
                        ssl_detected = 1.0
                        ssl_mag = max(0.0, (sl_price - l[k]) / atr)
                        ssl_wick_ratio = max(0.0, min(1.0, (min(o[k], c[k]) - l[k]) / c_range))
                        ssl_vol_ratio = min(10.0, v[k] / vol_baseline)
                        ssl_displacement = max(-1.0, min(1.0, (c[k] - o[k]) / c_range))
                        bars_since_ssl = (n - 1 - k) / float(sweep_lookback)
                        break
                if ssl_detected > 0.0:
                    break

        # Distances to nearest un-swept swing points
        dist_bsl = 5.0
        if swing_highs:
            unbroken_highs = [p for _, p in swing_highs if p >= curr_c]
            if unbroken_highs:
                dist_bsl = max(-5.0, min(5.0, (min(unbroken_highs) - curr_c) / atr))

        dist_ssl = 5.0
        if swing_lows:
            unbroken_lows = [p for _, p in swing_lows if p <= curr_c]
            if unbroken_lows:
                dist_ssl = max(-5.0, min(5.0, (curr_c - max(unbroken_lows)) / atr))

        # Equal Highs / Lows density
        eq_thresh = self.config.eq_threshold_atr_ratio * atr
        eqh_score = 0.0
        if len(swing_highs) >= 2:
            high_prices = [p for _, p in swing_highs[-5:]]
            diffs = [abs(p1 - p2) for i, p1 in enumerate(high_prices) for p2 in high_prices[i + 1 :]]
            close_clusters = sum(1 for d in diffs if d <= eq_thresh)
            eqh_score = min(1.0, close_clusters * 0.35)

        eql_score = 0.0
        if len(swing_lows) >= 2:
            low_prices = [p for _, p in swing_lows[-5:]]
            diffs = [abs(p1 - p2) for i, p1 in enumerate(low_prices) for p2 in low_prices[i + 1 :]]
            close_clusters = sum(1 for d in diffs if d <= eq_thresh)
            eql_score = min(1.0, close_clusters * 0.35)

        feat[0] = bsl_detected
        feat[1] = ssl_detected
        feat[2] = bsl_mag
        feat[3] = ssl_mag
        feat[4] = bsl_wick_ratio
        feat[5] = ssl_wick_ratio
        feat[6] = bsl_vol_ratio
        feat[7] = ssl_vol_ratio
        feat[8] = bsl_displacement
        feat[9] = ssl_displacement
        feat[10] = bars_since_bsl
        feat[11] = bars_since_ssl
        feat[12] = dist_bsl
        feat[13] = dist_ssl
        feat[14] = eqh_score
        feat[15] = eql_score

    def _extract_fvgs(
        self,
        feat: np.ndarray,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
        atr: float,
        vol_baseline: float,
        eps: float,
    ) -> None:
        """Extracts 16 FVG features into feat[16:32]."""
        n = len(c)
        curr_c = c[-1]
        min_fvg_size = self.config.min_fvg_atr_ratio * atr

        bull_active = 0.0
        bear_active = 0.0
        bull_gap_size = 0.0
        bear_gap_size = 0.0
        bull_ce_dist = 5.0
        bear_ce_dist = 5.0
        bull_mitigation = 0.0
        bear_mitigation = 0.0
        price_in_bull_fvg = 0.0
        price_in_bear_fvg = 0.0
        bull_impulse_vol = 0.0
        bear_impulse_vol = 0.0
        bars_since_bull = 1.0
        bars_since_bear = 1.0
        total_bull_vol = 0.0
        total_bear_vol = 0.0
        inversion_flag = 0.0

        fvg_lookback = min(30, n - 2)
        for i in range(n - 1, n - fvg_lookback, -1):
            # Bullish FVG (BISI): Low[i] > High[i-2]
            if l[i] > h[i - 2]:
                gap = l[i] - h[i - 2]
                if gap >= min_fvg_size:
                    subsequent_lows = l[i + 1 : n]
                    min_sub_low = float(np.min(subsequent_lows)) if len(subsequent_lows) > 0 else l[i]
                    ce_level = (l[i] + h[i - 2]) / 2.0

                    if min_sub_low > h[i - 2]:  # Unviolated
                        if bull_active == 0.0:
                            bull_active = 1.0
                            bull_gap_size = min(5.0, gap / atr)
                            bull_ce_dist = max(-5.0, min(5.0, (curr_c - ce_level) / atr))
                            bull_mitigation = max(0.0, min(1.0, (l[i] - min_sub_low) / max(gap, eps)))
                            price_in_bull_fvg = 1.0 if (h[i - 2] <= curr_c <= l[i]) else 0.0
                            bull_impulse_vol = min(10.0, v[i - 1] / vol_baseline)
                            bars_since_bull = (n - 1 - i) / float(fvg_lookback)
                        total_bull_vol += gap * v[i - 1]
                    else:
                        if curr_c < h[i - 2] and inversion_flag == 0.0:
                            inversion_flag = -1.0

            # Bearish FVG (SIBI): High[i] < Low[i-2]
            if h[i] < l[i - 2]:
                gap = l[i - 2] - h[i]
                if gap >= min_fvg_size:
                    subsequent_highs = h[i + 1 : n]
                    max_sub_high = float(np.max(subsequent_highs)) if len(subsequent_highs) > 0 else h[i]
                    ce_level = (l[i - 2] + h[i]) / 2.0

                    if max_sub_high < l[i - 2]:  # Unviolated
                        if bear_active == 0.0:
                            bear_active = 1.0
                            bear_gap_size = min(5.0, gap / atr)
                            bear_ce_dist = max(-5.0, min(5.0, (ce_level - curr_c) / atr))
                            bear_mitigation = max(0.0, min(1.0, (max_sub_high - h[i]) / max(gap, eps)))
                            price_in_bear_fvg = 1.0 if (h[i] <= curr_c <= l[i - 2]) else 0.0
                            bear_impulse_vol = min(10.0, v[i - 1] / vol_baseline)
                            bars_since_bear = (n - 1 - i) / float(fvg_lookback)
                        total_bear_vol += gap * v[i - 1]
                    else:
                        if curr_c > l[i - 2] and inversion_flag == 0.0:
                            inversion_flag = 1.0

        vol_sum = total_bull_vol + total_bear_vol
        net_fvg_bias = (total_bull_vol - total_bear_vol) / max(vol_sum, eps) if vol_sum > 0 else 0.0
        net_fvg_bias = max(-1.0, min(1.0, net_fvg_bias))

        feat[16] = bull_active
        feat[17] = bear_active
        feat[18] = bull_gap_size
        feat[19] = bear_gap_size
        feat[20] = bull_ce_dist
        feat[21] = bear_ce_dist
        feat[22] = bull_mitigation
        feat[23] = bear_mitigation
        feat[24] = price_in_bull_fvg
        feat[25] = price_in_bear_fvg
        feat[26] = bull_impulse_vol
        feat[27] = bear_impulse_vol
        feat[28] = bars_since_bull
        feat[29] = bars_since_bear
        feat[30] = net_fvg_bias
        feat[31] = inversion_flag

    def _extract_order_blocks(
        self,
        feat: np.ndarray,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
        atr: float,
        vol_baseline: float,
        eps: float,
    ) -> None:
        """Extracts 16 Order Block features into feat[32:48]."""
        n = len(c)
        curr_c = c[-1]
        impulse_k = self.config.ob_impulse_bars

        bull_ob_detected = 0.0
        bear_ob_detected = 0.0
        bull_ob_top_dist = 5.0
        bull_ob_bot_dist = 5.0
        bull_ob_mt_dist = 5.0
        bear_ob_top_dist = 5.0
        bear_ob_bot_dist = 5.0
        bear_ob_mt_dist = 5.0
        bull_ob_vol_ratio = 0.0
        bear_ob_vol_ratio = 0.0
        bull_ob_mitigation = 0.0
        bear_ob_mitigation = 0.0
        bars_since_bull_ob = 1.0
        bars_since_bear_ob = 1.0
        breaker_status = 0.0
        ob_momentum = 0.0

        ob_lookback = min(30, n - impulse_k - 1)
        for i in range(n - impulse_k - 1, n - ob_lookback - impulse_k, -1):
            if i < 0:
                break

            # Bullish OB: Down candle before explosive upward departure
            disp_up = c[i + impulse_k] - h[i]
            if c[i] < o[i] and disp_up > 0.8 * atr:
                dep_vol = float(np.mean(v[i + 1 : i + impulse_k + 1]))
                ob_high = h[i]
                ob_low = l[i]
                ob_mt = (ob_high + ob_low) / 2.0

                sub_lows = l[i + impulse_k + 1 : n]
                min_sub = float(np.min(sub_lows)) if len(sub_lows) > 0 else ob_high + atr

                if min_sub > ob_low:
                    if bull_ob_detected == 0.0:
                        bull_ob_detected = 1.0
                        bull_ob_top_dist = max(-5.0, min(5.0, (curr_c - ob_high) / atr))
                        bull_ob_bot_dist = max(-5.0, min(5.0, (curr_c - ob_low) / atr))
                        bull_ob_mt_dist = max(-5.0, min(5.0, (curr_c - ob_mt) / atr))
                        bull_ob_vol_ratio = min(10.0, dep_vol / vol_baseline)
                        bull_ob_mitigation = 0.5 if min_sub <= ob_mt else (0.25 if min_sub <= ob_high else 0.0)
                        bars_since_bull_ob = (n - 1 - i) / float(ob_lookback)
                        ob_momentum = min(5.0, disp_up / (atr * impulse_k))
                else:
                    if curr_c < ob_low and breaker_status == 0.0:
                        breaker_status = -1.0

            # Bearish OB: Up candle before explosive downward departure
            disp_down = l[i] - c[i + impulse_k]
            if c[i] > o[i] and disp_down > 0.8 * atr:
                dep_vol = float(np.mean(v[i + 1 : i + impulse_k + 1]))
                ob_high = h[i]
                ob_low = l[i]
                ob_mt = (ob_high + ob_low) / 2.0

                sub_highs = h[i + impulse_k + 1 : n]
                max_sub = float(np.max(sub_highs)) if len(sub_highs) > 0 else ob_low - atr

                if max_sub < ob_high:
                    if bear_ob_detected == 0.0:
                        bear_ob_detected = 1.0
                        bear_ob_top_dist = max(-5.0, min(5.0, (ob_high - curr_c) / atr))
                        bear_ob_bot_dist = max(-5.0, min(5.0, (ob_low - curr_c) / atr))
                        bear_ob_mt_dist = max(-5.0, min(5.0, (ob_mt - curr_c) / atr))
                        bear_ob_vol_ratio = min(10.0, dep_vol / vol_baseline)
                        bear_ob_mitigation = 0.5 if max_sub >= ob_mt else (0.25 if max_sub >= ob_low else 0.0)
                        bars_since_bear_ob = (n - 1 - i) / float(ob_lookback)
                        ob_momentum = max(-5.0, -disp_down / (atr * impulse_k))
                else:
                    if curr_c > ob_high and breaker_status == 0.0:
                        breaker_status = 1.0

        feat[32] = bull_ob_detected
        feat[33] = bear_ob_detected
        feat[34] = bull_ob_top_dist
        feat[35] = bull_ob_bot_dist
        feat[36] = bull_ob_mt_dist
        feat[37] = bear_ob_top_dist
        feat[38] = bear_ob_bot_dist
        feat[39] = bear_ob_mt_dist
        feat[40] = bull_ob_vol_ratio
        feat[41] = bear_ob_vol_ratio
        feat[42] = bull_ob_mitigation
        feat[43] = bear_ob_mitigation
        feat[44] = bars_since_bull_ob
        feat[45] = bars_since_bear_ob
        feat[46] = breaker_status
        feat[47] = ob_momentum

    def _extract_traps_and_structure(
        self,
        feat: np.ndarray,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        v: np.ndarray,
        swing_highs: List[Tuple[int, float]],
        swing_lows: List[Tuple[int, float]],
        atr: float,
        vol_baseline: float,
        eps: float,
    ) -> None:
        """Extracts 16 False Breakout Trap and Market Structure features into feat[48:64]."""
        n = len(c)
        curr_c = c[-1]
        curr_o = o[-1]
        curr_h = h[-1]
        curr_l = l[-1]

        trapped_buyers = 0.0
        trapped_sellers = 0.0
        follow_through_failure = 0.0
        vol_exhaustion = 0.0

        if swing_highs and n >= 3:
            prior_highs = [p for idx, p in swing_highs if idx < n - 2]
            if prior_highs:
                breached = [p for p in reversed(prior_highs) if h[-2] > p]
                if breached:
                    reclaimed = [p for p in breached if c[-1] < p]
                    latest_sh = reclaimed[0] if reclaimed else breached[0]
                    if c[-1] < h[-2]:
                        follow_through_failure = max(0.0, min(1.0, (h[-2] - c[-1]) / atr))
                    if v[-2] < vol_baseline:
                        vol_exhaustion = max(0.0, min(1.0, 1.0 - (v[-2] / vol_baseline)))
                    if c[-1] < latest_sh:
                        trapped_buyers = min(1.0, 0.5 + 0.5 * follow_through_failure)

        if swing_lows and n >= 3:
            prior_lows = [p for idx, p in swing_lows if idx < n - 2]
            if prior_lows:
                breached = [p for p in reversed(prior_lows) if l[-2] < p]
                if breached:
                    reclaimed = [p for p in breached if c[-1] > p]
                    latest_sl = reclaimed[0] if reclaimed else breached[0]
                    if c[-1] > l[-2]:
                        follow_through_failure = max(0.0, min(1.0, (c[-1] - l[-2]) / atr))
                    if v[-2] < vol_baseline:
                        vol_exhaustion = max(0.0, min(1.0, 1.0 - (v[-2] / vol_baseline)))
                    if c[-1] > latest_sl:
                        trapped_sellers = min(1.0, 0.5 + 0.5 * follow_through_failure)

        # Judas Swing fakeout
        short_drift = curr_c - c[-min(10, n)]
        candle_body = curr_c - curr_o
        judas_swing = 0.0
        if short_drift > atr and candle_body < -0.5 * atr:
            judas_swing = -1.0
        elif short_drift < -atr and candle_body > 0.5 * atr:
            judas_swing = 1.0

        # Market Structure Shift (MSS / CHoCH) and BOS
        choch = 0.0
        bos = 0.0
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            sh1, sh2 = swing_highs[-2][1], swing_highs[-1][1]
            sl1, sl2 = swing_lows[-2][1], swing_lows[-1][1]
            if sh2 < sh1 and curr_c > sh2:
                choch = 1.0
            elif sl2 > sl1 and curr_c < sl2:
                choch = -1.0

            if sh2 > sh1 and curr_c > sh2:
                bos = 1.0
            elif sl2 < sl1 and curr_c < sl2:
                bos = -1.0

        # Displacement quality score
        tot_range = max(curr_h - curr_l, eps)
        disp_quality = max(0.0, min(1.0, abs(curr_c - curr_o) / tot_range))

        # Premium / Discount Equilibrium Pricing [0, 1] with zero/nan guard
        range_high = float(np.max(h))
        range_low = float(np.min(l))
        eq_range = range_high - range_low
        if not math.isfinite(eq_range) or eq_range < eps:
            equilibrium = 0.5
            internal_bias = 0.0
        else:
            equilibrium = max(0.0, min(1.0, (curr_c - range_low) / eq_range))
            internal_bias = max(-1.0, min(1.0, (curr_c - (range_high + range_low) / 2.0) / (0.5 * eq_range)))

        # Inducement (IDM) proximity
        idm_proximity = 0.0
        if swing_lows and swing_highs:
            minor_dist = min(abs(curr_c - swing_lows[-1][1]), abs(curr_c - swing_highs[-1][1]))
            idm_proximity = max(0.0, min(1.0, 1.0 - (minor_dist / (2.0 * atr))))

        # Sweep vs Trap Divergence
        sweep_divergence = max(-1.0, min(1.0, trapped_buyers - feat[0]))

        # Liquidity void expansion rate
        recent_ranges = h[-3:] - l[-3:]
        void_rate = min(5.0, float(np.mean(recent_ranges)) / atr)

        # Wick to body exhaustion ratio
        body = max(abs(curr_c - curr_o), eps)
        total_wicks = (curr_h - max(curr_o, curr_c)) + (min(curr_o, curr_c) - curr_l)
        wick_body_ratio = min(10.0, total_wicks / body)

        # External range liquidity run potential [0, 1]
        dist_to_extreme = min(range_high - curr_c, curr_c - range_low)
        external_potential = max(0.0, min(1.0, 1.0 - (dist_to_extreme / max(eq_range, eps))))

        # Composite SMC Regime Score [-1, 1]
        bull_signal = (
            feat[1] * 0.25
            + feat[16] * 0.20
            + feat[32] * 0.20
            + trapped_sellers * 0.15
            + max(0.0, choch) * 0.20
        )
        bear_signal = (
            feat[0] * 0.25
            + feat[17] * 0.20
            + feat[33] * 0.20
            + trapped_buyers * 0.15
            + max(0.0, -choch) * 0.20
        )
        composite_score = max(-1.0, min(1.0, bull_signal - bear_signal))

        feat[48] = trapped_buyers
        feat[49] = trapped_sellers
        feat[50] = follow_through_failure
        feat[51] = vol_exhaustion
        feat[52] = judas_swing
        feat[53] = choch
        feat[54] = bos
        feat[55] = disp_quality
        feat[56] = equilibrium
        feat[57] = idm_proximity
        feat[58] = sweep_divergence
        feat[59] = void_rate
        feat[60] = wick_body_ratio
        feat[61] = internal_bias
        feat[62] = external_potential
        feat[63] = composite_score
