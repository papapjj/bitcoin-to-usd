from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig, SymbolConfig

BASE_MARKET_FEATURE_COLUMNS = [
    "ret1_atr",
    "ret3_atr",
    "ret5_atr",
    "ret15_atr",
    "ret30_atr",
    "ret60_atr",
    "gap_atr",
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "range_atr",
    "ema_diff_atr",
    "ema_slope_atr",
    "range_position",
    "atr_percentile",
    "spread_atr",
    "rsi14",
    "rsi5",
    "macd_hist_atr",
    "bb_position",
    "volatility_ratio",
    "momentum_5",
    "momentum_15",
    "candle_body_ratio",
    "high_low_ratio",
    "ema50_slope_atr",
    "price_vs_ema20_atr",
    "price_vs_ema50_atr",
    "consec_up_atr",
]

EXTRA_MARKET_FEATURE_COLUMNS = [
    "consec_down_atr",
    "hour_sin",
    "hour_cos",
    "m5_ema_diff_atr",
    "m5_ema_slope_atr",
    "m5_volatility_ratio",
    "m5_gap_atr",
    "m15_ema_diff_atr",
    "m15_ema_slope_atr",
    "m15_volatility_ratio",
    "m15_gap_atr",
    "h1_ema_diff_atr",
    "h1_ema_slope_atr",
    "h1_volatility_ratio",
    "h1_gap_atr",
    "trend_strength_atr",
    "chop_index",
    "range_compression",
    "breakout_pressure_20",
    "breakout_pressure_60",
    "shock_range_atr",
    "session_asia",
    "spread_percentile",
    "dist_high20_atr",
    "dist_low20_atr",
    "dist_high60_atr",
    "dist_low60_atr",
    "pullback_ema20_atr",
    "pullback_ema50_atr",
]

MARKET_FEATURE_COLUMNS = BASE_MARKET_FEATURE_COLUMNS + EXTRA_MARKET_FEATURE_COLUMNS

TIMING_FEATURE_COLUMNS = [
    "swing_range120_position",
    "dist_swing_high120_atr",
    "dist_swing_low120_atr",
    "liquidity_sweep_rejection",
]

WYCKOFF_FEATURE_COLUMNS = [
    "relative_volume_20",
    "relative_volume_60",
    "volume_zscore_60",
    "volume_trend_20",
    "effort_result_ratio",
    "absorption_score",
    "range_width_atr",
    "range_age_normalized",
    "spring_score",
    "upthrust_score",
    "sign_of_strength_score",
    "sign_of_weakness_score",
    "spring_test_quality",
    "upthrust_test_quality",
    "bars_since_spring",
    "bars_since_upthrust",
]

TAIL_FEATURE_COLUMNS = TIMING_FEATURE_COLUMNS + WYCKOFF_FEATURE_COLUMNS


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def spread_to_price(spread: pd.Series | float, symbol: SymbolConfig) -> pd.Series | float:
    if symbol.spread_in_points:
        return spread * symbol.point_size
    return spread


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute the MT5 ``iATR`` value used by the EA feature contract.

    MetaTrader's ``iATR`` indicator is calculated using Wilder's Smoothed Moving Average
    (SMMA / RMA), which is an exponential moving average with alpha = 1 / period (matching MT5 iATR).
    """
    previous_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    window = max(int(period), 1)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def _rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    # Uses only values inside the trailing window, including the current bar.
    def percentile(values: np.ndarray) -> float:
        last = values[-1]
        return float(np.mean(values <= last))

    return series.rolling(window, min_periods=window).apply(percentile, raw=True)


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (rsi - 50.0) / 50.0


def _rolling_sum(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).sum()


def _selected_market_volume(frame: pd.DataFrame) -> pd.Series:
    """Prefer real volume, then tick volume, and stay neutral for old CSVs."""
    selected = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in ("RealVolume", "RealVol", "real_volume", "real_volume_ticks", "Vol"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            selected = selected.where(selected > 0.0, values.where(values > 0.0))
    for column in ("TickVolume", "TickVol", "tick_volume", "Volume", "volume"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            selected = selected.where(selected > 0.0, values.where(values > 0.0))
    return selected.fillna(1.0).clip(lower=1.0)


def _bars_since_rolling_extreme(series: pd.Series, window: int, *, high: bool) -> pd.Series:
    def age(values: np.ndarray) -> float:
        reversed_values = np.asarray(values, dtype=float)[::-1]
        return float(np.argmax(reversed_values) if high else np.argmin(reversed_values))

    return series.rolling(window, min_periods=window).apply(age, raw=True)


def _recent_event_context(
    score: pd.Series,
    level: pd.Series,
    volume: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return causal context for the latest non-zero event before each bar."""
    scores = score.to_numpy(dtype=float)
    levels = level.to_numpy(dtype=float)
    volumes = volume.to_numpy(dtype=float)
    age_norm = np.ones(scores.size, dtype=float)
    event_level = np.full(scores.size, np.nan, dtype=float)
    event_volume = np.full(scores.size, np.nan, dtype=float)
    found = np.zeros(scores.size, dtype=float)
    latest = -1
    for index in range(scores.size):
        if latest >= 0 and index - latest <= horizon:
            age_norm[index] = float(index - latest) / float(max(horizon, 1))
            event_level[index] = levels[latest]
            event_volume[index] = volumes[latest]
            found[index] = 1.0
        if np.isfinite(scores[index]) and scores[index] > 1e-12:
            latest = index
    return (
        pd.Series(age_norm, index=score.index),
        pd.Series(event_level, index=score.index),
        pd.Series(event_volume, index=score.index),
        pd.Series(found, index=score.index),
    )


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(int(span), 1), adjust=False).mean()


def _completed_timeframe_features(out: pd.DataFrame, rule: str, feature_config: FeatureConfig) -> pd.DataFrame:
    """Higher-timeframe features using only the candle closed at this M1 close.

    The EA runs its decision on the *previous* M1 bar (shift=1) and asks MT5
    for higher-timeframe indicators at shift=1.  For an M1 candle stamped
    ``t``, its information set closes at ``t + 1 minute``.  The newest usable
    M5/M15/H1 candle therefore has start
    ``floor(t + 1 minute, rule) - rule``.  This implementation matches that
    contract exactly; the former resample+shift approach had boundary leakage
    and was not numerically comparable to the EA around timeframe rollovers.
    """
    indexed = out.set_index("Datetime")[["Open", "High", "Low", "Close"]].sort_index()
    bars = indexed.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    neutral = pd.DataFrame(index=out.index, data={"ema_diff": 0.0, "ema_slope": 0.0, "tf_atr": np.nan, "gap": 0.0})
    if bars.empty:
        return neutral
    tf_atr = compute_atr(bars, feature_config.atr_period).replace(0.0, np.nan)
    fast = _ema(bars["Close"], feature_config.ema_fast)
    slow = _ema(bars["Close"], feature_config.ema_slow)
    htf = pd.DataFrame(
        {
            "ema_diff": fast - slow,
            "ema_slope": fast - fast.shift(1),
            "tf_atr": tf_atr,
            "gap": bars["Open"] - bars["Close"].shift(1),
        },
        index=bars.index,
    )
    frequency = pd.tseries.frequencies.to_offset(rule)
    diffs = out["Datetime"].diff().dropna()
    candle_delta = diffs.median() if not diffs.empty else pd.Timedelta(minutes=1)
    if pd.isna(candle_delta) or candle_delta <= pd.Timedelta(0):
        candle_delta = pd.Timedelta(minutes=1)
    close_times = pd.DatetimeIndex(out["Datetime"]) + candle_delta
    lookup = close_times.floor(rule) - frequency
    aligned = htf.reindex(lookup)
    aligned.index = out.index
    return aligned.fillna({"ema_diff": 0.0, "ema_slope": 0.0})


def add_features(df: pd.DataFrame, feature_config: FeatureConfig, symbol: SymbolConfig) -> pd.DataFrame:
    out = df.copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"], dayfirst=True, errors="coerce")
    out = out.sort_values("Datetime").reset_index(drop=True)
    out["spread_price"] = spread_to_price(out["Spread"].astype(float), symbol)
    out["market_volume"] = _selected_market_volume(out)
    out["atr"] = compute_atr(out, feature_config.atr_period)
    atr = out["atr"].replace(0.0, np.nan)
    true_range = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - out["Close"].shift(1)).abs(),
            (out["Low"] - out["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for n in (1, 3, 5, 15, 30, 60):
        out[f"ret{n}_atr"] = (out["Close"] - out["Close"].shift(n)) / atr

    out["gap_atr"] = (out["Open"] - out["Close"].shift(1)) / atr
    candle_high_body = out[["Open", "Close"]].max(axis=1)
    candle_low_body = out[["Open", "Close"]].min(axis=1)
    out["body_atr"] = (out["Close"] - out["Open"]) / atr
    out["upper_wick_atr"] = (out["High"] - candle_high_body) / atr
    out["lower_wick_atr"] = (candle_low_body - out["Low"]) / atr
    out["range_atr"] = (out["High"] - out["Low"]) / atr

    ema_fast = out["Close"].ewm(span=feature_config.ema_fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=feature_config.ema_slow, adjust=False).mean()
    out["ema_diff_atr"] = (ema_fast - ema_slow) / atr
    out["ema_slope_atr"] = (ema_fast - ema_fast.shift(1)) / atr

    rolling_high = out["High"].rolling(feature_config.range_window, min_periods=feature_config.range_window).max()
    rolling_low = out["Low"].rolling(feature_config.range_window, min_periods=feature_config.range_window).min()
    range_width = (rolling_high - rolling_low).replace(0.0, np.nan)
    out["range_position"] = ((out["Close"] - rolling_low) / range_width).clip(0.0, 1.0)
    out["atr_percentile"] = _rolling_last_percentile(out["atr"], feature_config.atr_percentile_window)
    out["spread_atr"] = out["spread_price"] / atr
    out["not_in_shock_regime"] = (out["range_atr"] <= feature_config.shock_range_atr).astype(float)

    out["rsi14"] = _compute_rsi(out["Close"], 14)
    out["rsi5"] = _compute_rsi(out["Close"], 5)

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    # MT5 iMACD uses a simple moving-average signal line (despite the EMA
    # fast/slow legs); using an EMA here shifts the histogram materially.
    macd_signal = macd_line.rolling(9, min_periods=9).mean()
    out["macd_hist_atr"] = (macd_line - macd_signal) / atr

    bb_mid = out["Close"].rolling(20, min_periods=20).mean()
    # ``iBands`` uses population standard deviation (N), not pandas'
    # sample standard deviation (N-1).
    bb_std = out["Close"].rolling(20, min_periods=20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0.0, np.nan)
    out["bb_position"] = ((out["Close"] - bb_lower) / bb_width).fillna(0.5).clip(0.0, 1.0)

    atr_slow = compute_atr(out, feature_config.atr_period * 3)
    out["volatility_ratio"] = (atr / atr_slow.replace(0.0, np.nan)).clip(0.1, 5.0)

    out["momentum_5"] = (out["Close"] / out["Close"].shift(5) - 1.0) / (atr / out["Close"].shift(5).replace(0.0, np.nan))
    out["momentum_15"] = (out["Close"] / out["Close"].shift(15) - 1.0) / (atr / out["Close"].shift(15).replace(0.0, np.nan))

    out["candle_body_ratio"] = (out["body_atr"].abs() / out["range_atr"].replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)

    rolling_high_20 = out["High"].rolling(20, min_periods=20).max()
    rolling_low_20 = out["Low"].rolling(20, min_periods=20).min()
    hl_range = (rolling_high_20 - rolling_low_20).replace(0.0, np.nan)
    out["high_low_ratio"] = ((out["Close"] - rolling_low_20) / hl_range).clip(0.0, 1.0)

    # The EA names this handle ``ema20`` for historical reasons, but its
    # period is the configured ``FeatureEmaFast`` (50 in the MT5 profile),
    # not a hard-coded 20.  Keep the fixed 50-period companion separate.
    ema20 = ema_fast
    ema50 = out["Close"].ewm(span=50, adjust=False).mean()
    out["ema20_slope_atr"] = (ema20 - ema20.shift(1)) / atr
    out["ema50_slope_atr"] = (ema50 - ema50.shift(1)) / atr
    out["price_vs_ema20_atr"] = (out["Close"] - ema20) / atr
    out["price_vs_ema50_atr"] = (out["Close"] - ema50) / atr

    up_bar = (out["Close"] > out["Close"].shift(1)).astype(float)
    down_bar = (out["Close"] < out["Close"].shift(1)).astype(float)
    def _consec(s: pd.Series) -> pd.Series:
        result = np.zeros(len(s), dtype=float)
        count = 0.0
        for i in range(len(s)):
            if s.iloc[i] > 0:
                count = max(count, 0.0) + 1.0
            elif s.iloc[i] < 0:
                count = min(count, 0.0) - 1.0
            else:
                count = 0.0
            result[i] = count
        return pd.Series(result, index=s.index)
    combined = up_bar - down_bar
    out["consec_up_atr"] = _consec(combined).clip(-10.0, 10.0) / 10.0
    out["consec_down_atr"] = (-_consec(combined)).clip(-10.0, 10.0) / 10.0

    dt = out["Datetime"].dt
    hour = dt.hour + dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    mtf_scaling_mode = str(getattr(feature_config, "mtf_scaling_mode", "legacy_m1_atr_v1") or "legacy_m1_atr_v1").strip().lower()
    for prefix, rule in (("m5", "5min"), ("m15", "15min"), ("h1", "1h")):
        completed = _completed_timeframe_features(out, rule, feature_config)
        if mtf_scaling_mode == "native_tf_atr_v2":
            # Candidate-only: a completed H1 bar must be expressed in H1
            # volatility units, not in a much smaller M1 ATR.  The symmetric
            # relative-ATR map is naturally bounded in (-1, 1), so soft-clip
            # cannot pin H1 volatility to a tail constant.
            # Match the EA fallback: neither an unfinished higher-timeframe
            # bar nor a malformed base ATR may create an infinite ratio.
            base_atr = atr.where(atr.abs() > 1e-10, 1e-10)
            tf_atr = completed["tf_atr"].where(completed["tf_atr"].abs() > 1e-10, base_atr)
            out[f"{prefix}_ema_diff_atr"] = (completed["ema_diff"] / tf_atr).fillna(0.0)
            out[f"{prefix}_ema_slope_atr"] = (completed["ema_slope"] / tf_atr).fillna(0.0)
            out[f"{prefix}_volatility_ratio"] = ((tf_atr - base_atr) / (tf_atr + base_atr)).fillna(0.0).clip(-1.0, 1.0)
            out[f"{prefix}_gap_atr"] = (completed["gap"] / tf_atr).fillna(0.0)
        else:
            # Historical protected-v13 definition.  Do not change the seed's
            # wire contract; normalized candidates select native_tf_atr_v2.
            out[f"{prefix}_ema_diff_atr"] = completed["ema_diff"] / atr
            out[f"{prefix}_ema_slope_atr"] = completed["ema_slope"] / atr
            out[f"{prefix}_volatility_ratio"] = (completed["tf_atr"] / atr).clip(0.1, 10.0).fillna(1.0)
            out[f"{prefix}_gap_atr"] = (completed["gap"] / atr).clip(-20.0, 20.0).fillna(0.0)

    trend_strength = (ema20 - ema50).abs() / atr
    out["trend_strength_atr"] = trend_strength.clip(0.0, 20.0)

    regime_window = max(feature_config.regime_window, 20)
    regime_high = out["High"].rolling(regime_window, min_periods=regime_window).max()
    regime_low = out["Low"].rolling(regime_window, min_periods=regime_window).min()
    regime_range = (regime_high - regime_low).replace(0.0, np.nan)
    tr_sum = _rolling_sum(true_range, regime_window, regime_window).replace(0.0, np.nan)
    out["chop_index"] = (
        100.0
        * np.log10(tr_sum / regime_range)
        / np.log10(float(regime_window))
    ).clip(0.0, 100.0) / 100.0

    range20 = (rolling_high_20 - rolling_low_20).replace(0.0, np.nan)
    range60 = (rolling_high - rolling_low).replace(0.0, np.nan)
    out["range_compression"] = (range20 / range60).clip(0.0, 2.0) / 2.0

    prev_high20 = rolling_high_20.shift(1)
    prev_low20 = rolling_low_20.shift(1)
    prev_high60 = rolling_high.shift(1)
    prev_low60 = rolling_low.shift(1)
    out["breakout_pressure_20"] = (
        (out["Close"] - prev_high20).clip(lower=0.0)
        - (prev_low20 - out["Close"]).clip(lower=0.0)
    ) / atr
    out["breakout_pressure_60"] = (
        (out["Close"] - prev_high60).clip(lower=0.0)
        - (prev_low60 - out["Close"]).clip(lower=0.0)
    ) / atr
    out["shock_range_atr"] = (out["range_atr"] / max(feature_config.shock_range_atr, 1e-12)).clip(0.0, 2.0) / 2.0

    out["session_asia"] = ((hour >= 0.0) & (hour < 8.0)).astype(float)
    out["spread_percentile"] = _rolling_last_percentile(out["spread_atr"], feature_config.atr_percentile_window)

    out["dist_high20_atr"] = (rolling_high_20 - out["Close"]) / atr
    out["dist_low20_atr"] = (out["Close"] - rolling_low_20) / atr
    out["dist_high60_atr"] = (rolling_high - out["Close"]) / atr
    out["dist_low60_atr"] = (out["Close"] - rolling_low) / atr
    out["pullback_ema20_atr"] = (ema20 - out["Close"]) / atr
    out["pullback_ema50_atr"] = (ema50 - out["Close"]) / atr

    wyckoff_range_window = int(feature_config.wyckoff_range_window)
    volume_short_window = int(feature_config.wyckoff_volume_short_window)
    volume_long_window = int(feature_config.wyckoff_volume_long_window)
    event_horizon = int(feature_config.wyckoff_event_horizon)
    swing_high_120 = out["High"].rolling(wyckoff_range_window, min_periods=wyckoff_range_window).max()
    swing_low_120 = out["Low"].rolling(wyckoff_range_window, min_periods=wyckoff_range_window).min()
    swing_width_120 = (swing_high_120 - swing_low_120).replace(0.0, np.nan)
    out["swing_range120_position"] = ((out["Close"] - swing_low_120) / swing_width_120).clip(0.0, 1.0)
    out["dist_swing_high120_atr"] = ((swing_high_120 - out["Close"]) / atr).clip(0.0, 20.0)
    out["dist_swing_low120_atr"] = ((out["Close"] - swing_low_120) / atr).clip(0.0, 20.0)

    prev_high_20 = out["High"].shift(1).rolling(20, min_periods=20).max()
    prev_low_20 = out["Low"].shift(1).rolling(20, min_periods=20).min()
    upper_rejection = ((out["High"] - candle_high_body) / atr).clip(0.0, 2.0) / 2.0
    lower_rejection = ((candle_low_body - out["Low"]) / atr).clip(0.0, 2.0) / 2.0
    swept_high = (out["High"] > prev_high_20) & (out["Close"] < prev_high_20)
    swept_low = (out["Low"] < prev_low_20) & (out["Close"] > prev_low_20)
    out["liquidity_sweep_rejection"] = (
        lower_rejection.where(swept_low, 0.0) - upper_rejection.where(swept_high, 0.0)
    ).clip(-1.0, 1.0)

    volume = out["market_volume"]
    previous_volume_short = volume.shift(1).rolling(volume_short_window, min_periods=volume_short_window).mean()
    previous_volume_long = volume.shift(1).rolling(volume_long_window, min_periods=volume_long_window).mean()
    previous_volume_std = volume.shift(1).rolling(volume_long_window, min_periods=volume_long_window).std(ddof=0)
    relative_volume_20_raw = volume / previous_volume_short.replace(0.0, np.nan)
    relative_volume_60_raw = volume / previous_volume_long.replace(0.0, np.nan)
    out["relative_volume_20"] = relative_volume_20_raw.clip(0.0, 5.0) / 5.0
    out["relative_volume_60"] = relative_volume_60_raw.clip(0.0, 5.0) / 5.0
    # Mirror MQL5 ``max(previousStd, previousMean * 0.05)`` before the
    # absolute floor.  The old ``where`` only applied the 5% floor when the
    # standard deviation was zero, overstating z-scores in quiet volume.
    volume_zscore_scale = previous_volume_std.clip(
        lower=previous_volume_long * 0.05,
    ).clip(lower=1.0)
    out["volume_zscore_60"] = (
        (volume - previous_volume_long) / volume_zscore_scale
    ).fillna(0.0).clip(-5.0, 5.0) / 5.0
    recent_volume = volume.rolling(5, min_periods=5).mean()
    prior_volume = volume.shift(5).rolling(volume_short_window, min_periods=volume_short_window).mean()
    out["volume_trend_20"] = ((recent_volume / prior_volume.replace(0.0, np.nan)) - 1.0).clip(-2.0, 2.0) / 2.0
    out["effort_result_ratio"] = (
        out["body_atr"] / relative_volume_20_raw.clip(lower=0.25)
    ).clip(-2.0, 2.0) / 2.0
    out["absorption_score"] = (
        (
            lower_rejection * (1.0 - out["swing_range120_position"])
            - upper_rejection * out["swing_range120_position"]
        )
        * (relative_volume_20_raw - 1.0).clip(lower=0.0)
    ).clip(-2.0, 2.0) / 2.0
    out["range_width_atr"] = (swing_width_120 / atr).clip(0.0, 20.0) / 20.0
    high_age = _bars_since_rolling_extreme(out["High"], wyckoff_range_window, high=True)
    low_age = _bars_since_rolling_extreme(out["Low"], wyckoff_range_window, high=False)
    out["range_age_normalized"] = np.minimum(high_age, low_age).clip(0.0, 60.0) / 60.0

    volume_support = 0.5 + 0.5 * (relative_volume_20_raw / 2.0).clip(0.0, 1.0)
    spring_structure = (
        0.5 * ((prev_low_20 - out["Low"]) / atr).clip(0.0, 1.0)
        + 0.5 * lower_rejection
    )
    upthrust_structure = (
        0.5 * ((out["High"] - prev_high_20) / atr).clip(0.0, 1.0)
        + 0.5 * upper_rejection
    )
    out["spring_score"] = (spring_structure * volume_support).where(swept_low, 0.0).clip(0.0, 1.0)
    out["upthrust_score"] = (upthrust_structure * volume_support).where(swept_high, 0.0).clip(0.0, 1.0)

    close_location = ((out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0.0, np.nan)).clip(0.0, 1.0)
    sos_breakout = ((out["Close"] - prev_high_20) / atr).clip(0.0, 1.0)
    sow_breakout = ((prev_low_20 - out["Close"]) / atr).clip(0.0, 1.0)
    sos_body = out["body_atr"].clip(0.0, 1.0)
    sow_body = (-out["body_atr"]).clip(0.0, 1.0)
    out["sign_of_strength_score"] = (
        ((sos_breakout + sos_body + close_location) / 3.0) * volume_support
    ).where((out["Close"] > prev_high_20) & (out["Close"] > out["Open"]), 0.0).clip(0.0, 1.0)
    out["sign_of_weakness_score"] = (
        ((sow_breakout + sow_body + (1.0 - close_location)) / 3.0) * volume_support
    ).where((out["Close"] < prev_low_20) & (out["Close"] < out["Open"]), 0.0).clip(0.0, 1.0)

    spring_age, spring_level, spring_volume, spring_found = _recent_event_context(
        out["spring_score"], out["Low"], volume, event_horizon
    )
    upthrust_age, upthrust_level, upthrust_volume, upthrust_found = _recent_event_context(
        out["upthrust_score"], out["High"], volume, event_horizon
    )
    spring_price_quality = (1.0 - (out["Low"] - spring_level).abs() / atr).clip(0.0, 1.0)
    spring_hold = ((out["Low"] >= spring_level - 0.25 * atr) & (out["Close"] > spring_level)).astype(float)
    spring_volume_quality = (1.0 - volume / spring_volume.replace(0.0, np.nan)).clip(0.0, 1.0)
    out["spring_test_quality"] = (
        spring_found * spring_price_quality * spring_hold * (0.5 + 0.5 * spring_volume_quality)
    ).fillna(0.0).clip(0.0, 1.0)
    upthrust_price_quality = (1.0 - (out["High"] - upthrust_level).abs() / atr).clip(0.0, 1.0)
    upthrust_hold = ((out["High"] <= upthrust_level + 0.25 * atr) & (out["Close"] < upthrust_level)).astype(float)
    upthrust_volume_quality = (1.0 - volume / upthrust_volume.replace(0.0, np.nan)).clip(0.0, 1.0)
    out["upthrust_test_quality"] = (
        upthrust_found * upthrust_price_quality * upthrust_hold * (0.5 + 0.5 * upthrust_volume_quality)
    ).fillna(0.0).clip(0.0, 1.0)
    out["bars_since_spring"] = spring_age.clip(0.0, 1.0)
    out["bars_since_upthrust"] = upthrust_age.clip(0.0, 1.0)

    all_feature_columns = MARKET_FEATURE_COLUMNS + TAIL_FEATURE_COLUMNS
    out[all_feature_columns] = out[all_feature_columns].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=all_feature_columns + ["atr", "Datetime"]).reset_index(drop=True)
