"""
src/self_growing_trader/regime_data.py

BTCUSD M1 Data Pipeline & Objective Ground Truth Labeling.
Part of Milestone 1 for the Multi-Agent Regime Detection Engine.

Key components:
- High-efficiency raw data ingestion and parsing from BTCUSDm.csv (Q1 2025).
- In-Sample (Jan 2025) vs Out-of-Sample (Feb & Mar 2025) temporal splitting.
- 15-dimensional strictly causal feature extraction (backward-looking only, zero lookahead).
- Triple Barrier Method objective ground truth labeling (H=60, k=5.0 ATR).
- Boundary embargo purge to prevent lookahead leakage into Out-of-Sample evaluation.
- Scaler isolation: standardizers fitted strictly on In-Sample January data.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

DEFAULT_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"

REGIME_FEATURE_NAMES = [
    "mom_5_log",
    "mom_15_log",
    "mom_60_log",
    "mom_240_log",
    "mom_15_atr",
    "vol_parkinson_30",
    "vol_garman_klass_30",
    "atr_14",
    "atr_ratio",
    "price_velocity",
    "price_acceleration",
    "spread_atr",
    "rel_tickvol_20",
    "tickvol_flow_15",
    "bb_kc_squeeze_ratio",
]

REGIME_LABELS = {
    0: "BULL",
    1: "BEAR",
    2: "CHOP",
}
REGIME_LABEL_BULL = 0
REGIME_LABEL_BEAR = 1
REGIME_LABEL_CHOP = 2
REGIME_LABEL_PURGED = -1


def _resolve_col_name(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Finds the first matching column name in df, ignoring case and whitespace."""
    col_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        cand_norm = candidate.strip().lower()
        if cand_norm in col_map:
            return col_map[cand_norm]
    return None


def _get_series(df: pd.DataFrame, candidates: Sequence[str], default_val: float = 0.0) -> pd.Series:
    """Extracts a numeric pd.Series from df matching candidate names, or returns a default series."""
    col_name = _resolve_col_name(df, candidates)
    if col_name is not None:
        return pd.to_numeric(df[col_name], errors="coerce")
    return pd.Series(default_val, index=df.index, dtype=float)


def load_raw_btcusd_m1(
    csv_path: str = DEFAULT_CSV_PATH,
    start_date: str = "2025-01-01",
    end_date: str = "2025-03-31",
) -> pd.DataFrame:
    """
    Loads, parses datetime, filters date range (default Q1 2025), and ensures chronological order.

    Args:
        csv_path: Absolute path to the BTCUSDm.csv file.
        start_date: Starting date in YYYY-MM-DD or YYYY.MM.DD format.
        end_date: Ending date in YYYY-MM-DD or YYYY.MM.DD format (inclusive).

    Returns:
        pd.DataFrame containing chronological M1 bars with standardized columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"BTCUSD M1 dataset not found at: {csv_path}")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    # Fast read: try pyarrow first for sub-second ingestion, fallback to standard engine='c'
    raw_df: pd.DataFrame
    try:
        raw_df = pd.read_csv(csv_path, sep="\t", engine="pyarrow")
    except Exception:
        raw_df = pd.read_csv(csv_path, sep="\t", engine="c")

    date_col = _resolve_col_name(raw_df, ["<DATE>", "date", "Date"])
    time_col = _resolve_col_name(raw_df, ["<TIME>", "time", "Time"])
    dt_col = _resolve_col_name(raw_df, ["datetime", "Datetime", "<DATETIME>"])

    if date_col is not None and time_col is not None:
        # Pre-filter by date string if possible to accelerate datetime conversion
        date_str_series = raw_df[date_col].astype(str)
        time_str_series = raw_df[time_col].astype(str)
        start_str = start_ts.strftime("%Y.%m.%d")
        end_str = end_ts.strftime("%Y.%m.%d")

        date_mask = (date_str_series >= start_str) & (date_str_series <= end_str)
        filtered = raw_df[date_mask].copy().reset_index(drop=True)

        comb = filtered[date_col].astype(str) + " " + filtered[time_col].astype(str)
        datetimes = pd.to_datetime(comb, format="%Y.%m.%d %H:%M:%S", errors="coerce")
        filtered["datetime"] = datetimes
    elif dt_col is not None:
        filtered = raw_df.copy()
        filtered["datetime"] = pd.to_datetime(filtered[dt_col], errors="coerce")
    else:
        raise ValueError("Dataset does not contain recognizable date/time columns.")

    mask = (filtered["datetime"] >= start_ts) & (filtered["datetime"] <= end_ts)
    df_slice = filtered[mask].copy()

    # Standardize column mapping while preserving original columns
    open_s = _get_series(df_slice, ["<OPEN>", "open", "Open"])
    high_s = _get_series(df_slice, ["<HIGH>", "high", "High"])
    low_s = _get_series(df_slice, ["<LOW>", "low", "Low"])
    close_s = _get_series(df_slice, ["<CLOSE>", "close", "Close"])
    tickvol_s = _get_series(df_slice, ["<TICKVOL>", "tickvol", "TickVolume", "volume", "vol"])
    vol_s = _get_series(df_slice, ["<VOL>", "vol", "Vol"])
    spread_s = _get_series(df_slice, ["<SPREAD>", "spread", "Spread"], default_val=2880.0)

    standardized = pd.DataFrame(
        {
            "datetime": df_slice["datetime"],
            "open": open_s,
            "high": high_s,
            "low": low_s,
            "close": close_s,
            "tickvol": tickvol_s,
            "vol": vol_s,
            "spread": spread_s,
            # MT5 tag aliases for direct interoperability
            "<OPEN>": open_s,
            "<HIGH>": high_s,
            "<LOW>": low_s,
            "<CLOSE>": close_s,
            "<TICKVOL>": tickvol_s,
            "<VOL>": vol_s,
            "<SPREAD>": spread_s,
        }
    )

    standardized.sort_values("datetime", inplace=True)
    standardized.reset_index(drop=True, inplace=True)
    return standardized


def split_q1_data(
    df: pd.DataFrame,
    train_end: str = "2025-01-31 23:59:59",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits continuous Q1 time series into In-Sample (Jan 2025) and Out-of-Sample (Feb & Mar 2025).

    Args:
        df: Continuous M1 bars dataframe with a 'datetime' column.
        train_end: Cutoff timestamp string separating In-Sample and Out-of-Sample.

    Returns:
        tuple (df_train, df_oos)
    """
    if "datetime" not in df.columns:
        raise KeyError("Input dataframe must contain a 'datetime' column.")

    cutoff = pd.to_datetime(train_end)
    train_mask = df["datetime"] <= cutoff
    df_train = df[train_mask].copy().reset_index(drop=True)
    df_oos = df[~train_mask].copy().reset_index(drop=True)
    return df_train, df_oos


def load_and_preprocess_q1_data(
    csv_path: str = DEFAULT_CSV_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    PROJECT.md Contract Interface:
    Loads BTCUSD M1 data for Q1 2025 and returns (df_jan_train, df_feb_mar_oos).
    """
    df_q1 = load_raw_btcusd_m1(csv_path=csv_path, start_date="2025-01-01", end_date="2025-03-31")
    return split_q1_data(df_q1, train_end="2025-01-31 23:59:59")


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts 15-dimensional stationary, causal feature vector from M1 bars.
    Zero lookahead bias guaranteed: all computations depend strictly on bar t and past bars.

    Features:
    1. mom_5_log: ln(C_t / C_{t-5})
    2. mom_15_log: ln(C_t / C_{t-15})
    3. mom_60_log: ln(C_t / C_{t-60})
    4. mom_240_log: ln(C_t / C_{t-240})
    5. mom_15_atr: (C_t - C_{t-15}) / ATR14(t)
    6. vol_parkinson_30: Parkinson volatility (W=30)
    7. vol_garman_klass_30: Garman-Klass volatility (W=30)
    8. atr_14: MT5 simple ATR(14)
    9. atr_ratio: ATR14(t) / ATR42(t)
    10. price_velocity: First derivative of EMA10(C) normalized by ATR14(t)
    11. price_acceleration: Second derivative of EMA10(C) normalized by ATR14(t)
    12. spread_atr: (Spread * 0.01) / ATR14(t)
    13. rel_tickvol_20: TickVol_t / SMA20(TickVol)
    14. tickvol_flow_15: 15-bar cumulative tick volume weighted by bar direction sign
    15. bb_kc_squeeze_ratio: Bollinger Bandwidth (20, 2.0) / Keltner Bandwidth (20, 1.5)

    Args:
        df: pd.DataFrame containing open, high, low, close, tickvol, spread.

    Returns:
        pd.DataFrame with shape [N, 15] and column names REGIME_FEATURE_NAMES.
    """
    open_s = _get_series(df, ["open", "<OPEN>", "Open"])
    high_s = _get_series(df, ["high", "<HIGH>", "High"])
    low_s = _get_series(df, ["low", "<LOW>", "Low"])
    close_s = _get_series(df, ["close", "<CLOSE>", "Close"])
    tickvol_s = _get_series(df, ["tickvol", "<TICKVOL>", "TickVolume", "volume", "vol"])
    spread_s = _get_series(df, ["spread", "<SPREAD>", "Spread"], default_val=2880.0)

    n_bars = len(df)
    if n_bars == 0:
        return pd.DataFrame(columns=REGIME_FEATURE_NAMES)

    safe_low = np.maximum(low_s.values, 1e-8)
    safe_high = np.maximum(high_s.values, safe_low)
    safe_open = np.maximum(open_s.values, 1e-8)
    safe_close = np.maximum(close_s.values, 1e-8)

    # 1-4. Multi-horizon Momentum (Log returns)
    close_vals = safe_close
    mom_5_log = np.full(n_bars, np.nan, dtype=np.float64)
    mom_15_log = np.full(n_bars, np.nan, dtype=np.float64)
    mom_60_log = np.full(n_bars, np.nan, dtype=np.float64)
    mom_240_log = np.full(n_bars, np.nan, dtype=np.float64)

    if n_bars > 5:
        mom_5_log[5:] = np.log(close_vals[5:] / close_vals[:-5])
    if n_bars > 15:
        mom_15_log[15:] = np.log(close_vals[15:] / close_vals[:-15])
    if n_bars > 60:
        mom_60_log[60:] = np.log(close_vals[60:] / close_vals[:-60])
    if n_bars > 240:
        mom_240_log[240:] = np.log(close_vals[240:] / close_vals[:-240])

    # True Range & ATR computations (MT5 iATR: simple rolling mean of TR)
    prev_close = np.roll(close_vals, 1)
    prev_close[0] = close_vals[0]
    tr = np.maximum(
        safe_high - safe_low,
        np.maximum(
            np.abs(safe_high - prev_close),
            np.abs(safe_low - prev_close),
        ),
    )
    tr_series = pd.Series(tr, index=df.index)

    atr_14_s = tr_series.rolling(window=14, min_periods=14).mean()
    atr_42_s = tr_series.rolling(window=42, min_periods=42).mean()
    atr_20_s = tr_series.rolling(window=20, min_periods=20).mean()

    atr_14 = atr_14_s.values
    atr_42 = atr_42_s.values
    atr_20 = atr_20_s.values
    safe_atr_14 = np.maximum(atr_14, 1e-8)

    # 5. mom_15_atr: (C_t - C_{t-15}) / ATR14(t)
    mom_15_atr = np.full(n_bars, np.nan, dtype=np.float64)
    if n_bars > 15:
        mom_15_atr[15:] = (close_vals[15:] - close_vals[:-15]) / safe_atr_14[15:]

    # 6. Parkinson Volatility (W=30): sqrt(1 / (4*ln(2)*W) * sum(ln(H/L)^2))
    hl_log_sq = (np.log(safe_high / safe_low)) ** 2
    hl_log_sq_s = pd.Series(hl_log_sq, index=df.index)
    parkinson_var = hl_log_sq_s.rolling(window=30, min_periods=30).mean().values / (4.0 * np.log(2.0))
    vol_parkinson_30 = np.sqrt(np.maximum(parkinson_var, 0.0))

    # 7. Garman-Klass Volatility (W=30):
    # sqrt(1/W * sum[ 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2 ])
    co_log_sq = (np.log(safe_close / safe_open)) ** 2
    gk_element = 0.5 * hl_log_sq - (2.0 * np.log(2.0) - 1.0) * co_log_sq
    gk_element_s = pd.Series(gk_element, index=df.index)
    gk_var = gk_element_s.rolling(window=30, min_periods=30).mean().values
    vol_garman_klass_30 = np.sqrt(np.maximum(gk_var, 0.0))

    # 8. atr_14
    # 9. atr_ratio: ATR14 / ATR42
    atr_ratio = atr_14 / np.maximum(atr_42, 1e-8)

    # 10. price_velocity: first derivative of EMA10(C) normalized by ATR14(t)
    close_series = pd.Series(close_vals, index=df.index)
    ema_10 = close_series.ewm(span=10, adjust=False).mean().values
    price_velocity = np.full(n_bars, np.nan, dtype=np.float64)
    if n_bars > 1:
        price_velocity[1:] = (ema_10[1:] - ema_10[:-1]) / safe_atr_14[1:]

    # 11. price_acceleration: second derivative of EMA10(C) normalized by ATR14(t)
    price_acceleration = np.full(n_bars, np.nan, dtype=np.float64)
    if n_bars > 2:
        price_acceleration[2:] = price_velocity[2:] - price_velocity[1:-1]

    # 12. spread_atr: (Spread * 0.01) / ATR14(t)
    spread_atr = (spread_s.values * 0.01) / safe_atr_14

    # 13. rel_tickvol_20: TickVol_t / SMA20(TickVol)
    tv_vals = tickvol_s.values
    tv_s = pd.Series(tv_vals, index=df.index)
    tv_sma20 = tv_s.rolling(window=20, min_periods=20).mean().values
    rel_tickvol_20 = tv_vals / np.maximum(tv_sma20, 1e-8)

    # 14. tickvol_flow_15: 15-bar cumulative tick volume weighted by bar direction sign
    bar_direction = np.sign(close_vals - safe_open)
    flow_elem = tv_vals * bar_direction
    flow_s = pd.Series(flow_elem, index=df.index)
    tickvol_flow_15 = flow_s.rolling(window=15, min_periods=15).sum().values

    # 15. bb_kc_squeeze_ratio: Bollinger Bandwidth (20, 2.0) / Keltner Bandwidth (20, 1.5)
    close_std20 = close_series.rolling(window=20, min_periods=20).std(ddof=0).values
    bb_width = 4.0 * close_std20
    kc_width = 3.0 * atr_20
    bb_kc_squeeze_ratio = bb_width / np.maximum(kc_width, 1e-8)

    features = pd.DataFrame(
        {
            "mom_5_log": mom_5_log,
            "mom_15_log": mom_15_log,
            "mom_60_log": mom_60_log,
            "mom_240_log": mom_240_log,
            "mom_15_atr": mom_15_atr,
            "vol_parkinson_30": vol_parkinson_30,
            "vol_garman_klass_30": vol_garman_klass_30,
            "atr_14": atr_14,
            "atr_ratio": atr_ratio,
            "price_velocity": price_velocity,
            "price_acceleration": price_acceleration,
            "spread_atr": spread_atr,
            "rel_tickvol_20": rel_tickvol_20,
            "tickvol_flow_15": tickvol_flow_15,
            "bb_kc_squeeze_ratio": bb_kc_squeeze_ratio,
        },
        index=df.index,
    )

    return features


def extract_regime_features(df: pd.DataFrame) -> np.ndarray:
    """
    PROJECT.md Contract Interface:
    Computes 15D causal feature matrix of shape [N, 15] as numpy ndarray.
    """
    return compute_regime_features(df).to_numpy(dtype=np.float64)


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 60,
    k_atr: float = 5.0,
    purge_tail: bool = True,
) -> np.ndarray:
    """
    Computes forward objective ground truth labels via Triple Barrier Method.
    Upper barrier = Close_t + k_atr * ATR14(t)
    Lower barrier = Close_t - k_atr * ATR14(t)
    Vertical barrier = t + horizon bars

    Labels:
    0 = BULL (Upper barrier touched strictly before lower barrier within horizon)
    1 = BEAR (Lower barrier touched strictly before upper barrier within horizon)
    2 = CHOP (Neither horizontal barrier touched before vertical horizon expiration)
    -1 = PURGED (Warmup bars with missing ATR14, or tail bars purged to avoid lookahead bleed)

    Args:
        df: pd.DataFrame containing high, low, close.
        horizon: Forward holding horizon in M1 bars (default 60 = 1 hour).
        k_atr: Multiplier for ATR14 horizontal barriers (default 5.0).
        purge_tail: If True, sets labels of the last `horizon` bars to -1 (embargo purge).

    Returns:
        np.ndarray of shape [N] with int32 class labels.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if k_atr <= 0:
        raise ValueError("k_atr must be a positive float.")

    n_bars = len(df)
    if n_bars == 0:
        return np.empty(0, dtype=np.int32)

    close_s = _get_series(df, ["close", "<CLOSE>", "Close"])
    high_s = _get_series(df, ["high", "<HIGH>", "High"])
    low_s = _get_series(df, ["low", "<LOW>", "Low"])

    close = close_s.values.astype(np.float64)
    high = high_s.values.astype(np.float64)
    low = low_s.values.astype(np.float64)

    # ATR(14)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr_14 = pd.Series(tr).rolling(14, min_periods=14).mean().values

    labels = np.full(n_bars, REGIME_LABEL_PURGED, dtype=np.int32)
    valid_lookahead_len = n_bars - horizon

    if valid_lookahead_len <= 0:
        return labels

    # Vectorized sliding window check for barriers
    # high_windows shape: [valid_lookahead_len, horizon]
    # high_windows[i, j] corresponds to high[i + 1 + j]
    high_windows = np.lib.stride_tricks.sliding_window_view(high[1:], window_shape=horizon)[:valid_lookahead_len]
    low_windows = np.lib.stride_tricks.sliding_window_view(low[1:], window_shape=horizon)[:valid_lookahead_len]

    up_barrier = (close[:valid_lookahead_len] + k_atr * atr_14[:valid_lookahead_len])[:, None]
    dn_barrier = (close[:valid_lookahead_len] - k_atr * atr_14[:valid_lookahead_len])[:, None]

    up_touches = high_windows >= up_barrier
    dn_touches = low_windows <= dn_barrier

    has_up = np.any(up_touches, axis=1)
    has_dn = np.any(dn_touches, axis=1)

    # First touch index (0 to horizon-1)
    tau_up = np.where(has_up, np.argmax(up_touches, axis=1), horizon + 999)
    tau_dn = np.where(has_dn, np.argmax(dn_touches, axis=1), horizon + 999)

    valid_labels = np.full(valid_lookahead_len, REGIME_LABEL_CHOP, dtype=np.int32)

    mask_bull = has_up & (tau_up < tau_dn)
    mask_bear = has_dn & (tau_dn < tau_up)
    mask_tie = has_up & has_dn & (tau_up == tau_dn)

    valid_labels[mask_bull] = REGIME_LABEL_BULL
    valid_labels[mask_bear] = REGIME_LABEL_BEAR

    if np.any(mask_tie):
        tie_indices = np.where(mask_tie)[0]
        tie_tau = tau_up[tie_indices]
        c_future = close[tie_indices + 1 + tie_tau]
        c_origin = close[tie_indices]
        valid_labels[tie_indices] = np.where(c_future >= c_origin, REGIME_LABEL_BULL, REGIME_LABEL_BEAR)

    # Bars with NaN ATR14 are marked PURGED (-1)
    nan_atr_mask = np.isnan(atr_14[:valid_lookahead_len])
    valid_labels[nan_atr_mask] = REGIME_LABEL_PURGED

    labels[:valid_lookahead_len] = valid_labels

    # If purge_tail is False, handle remaining tail bars with truncated horizons
    if not purge_tail and valid_lookahead_len < n_bars:
        for t in range(valid_lookahead_len, n_bars - 1):
            if np.isnan(atr_14[t]):
                continue
            rem_h = n_bars - 1 - t
            u_bar = close[t] + k_atr * atr_14[t]
            d_bar = close[t] - k_atr * atr_14[t]
            future_high = high[t + 1 : n_bars]
            future_low = low[t + 1 : n_bars]

            u_hit = np.where(future_high >= u_bar)[0]
            d_hit = np.where(future_low <= d_bar)[0]
            t_u = u_hit[0] if len(u_hit) > 0 else 9999
            t_d = d_hit[0] if len(d_hit) > 0 else 9999

            if t_u < t_d and t_u < rem_h:
                labels[t] = REGIME_LABEL_BULL
            elif t_d < t_u and t_d < rem_h:
                labels[t] = REGIME_LABEL_BEAR
            elif t_u == t_d and t_u < rem_h:
                labels[t] = REGIME_LABEL_BULL if close[t + 1 + t_u] >= close[t] else REGIME_LABEL_BEAR
            else:
                labels[t] = REGIME_LABEL_CHOP

    return labels


def generate_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 60,
    k_atr: float = 5.0,
) -> np.ndarray:
    """
    PROJECT.md Contract Interface:
    Computes ground truth labels [N] with values {0: BULL, 1: BEAR, 2: CHOP}.
    """
    return compute_triple_barrier_labels(df, horizon=horizon, k_atr=k_atr, purge_tail=True)


def prepare_dataset(
    csv_path: str = DEFAULT_CSV_PATH,
    start_date: str = "2025-01-01",
    end_date: str = "2025-03-31",
    train_end: str = "2025-01-31 23:59:59",
    horizon: int = 60,
    k_atr: float = 5.0,
    scaler_type: str = "robust",
    warmup_drop_bars: int = 240,
) -> dict[str, Any]:
    """
    Assembles complete ML dataset:
    - Ingests raw continuous Q1 2025 BTCUSD M1 data.
    - Computes 15-dimensional backward-looking causal features.
    - Computes objective Triple Barrier ground truth labels with boundary embargo.
    - Embargo purge window: purges the last `horizon` bars of January so forward paths
      never cross the January/February boundary into Out-of-Sample evaluation.
    - Scaler isolation: fits scaler exclusively on In-Sample January training data.
    - Transforms both Train and OOS features with the In-Sample fitted scaler.

    Args:
        csv_path: Path to BTCUSDm.csv.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        train_end: Cutoff between Jan In-Sample and Feb/Mar Out-of-Sample.
        horizon: Holding horizon H (default 60 M1 bars).
        k_atr: Barrier width multiplier (default 5.0 ATR).
        scaler_type: "robust" (RobustScaler) or "standard" (StandardScaler).
        warmup_drop_bars: Number of initial bars to drop for feature warmup (default 240).

    Returns:
        dict containing:
        - X_train: np.ndarray [N_train, 15] (scaled)
        - y_train: np.ndarray [N_train] (labels {0, 1, 2})
        - X_oos: np.ndarray [N_oos, 15] (scaled)
        - y_oos: np.ndarray [N_oos] (labels {0, 1, 2})
        - feature_names: list[str]
        - scaler: fitted sklearn scaler
        - X_train_raw: unscaled training features
        - X_oos_raw: unscaled OOS features
        - df_train: cleaned training DataFrame
        - df_oos: cleaned OOS DataFrame
        - train_timestamps: pd.DatetimeIndex
        - oos_timestamps: pd.DatetimeIndex
    """
    df_raw = load_raw_btcusd_m1(csv_path=csv_path, start_date=start_date, end_date=end_date)
    features_df = compute_regime_features(df_raw)

    cutoff_ts = pd.to_datetime(train_end)
    is_train = df_raw["datetime"] <= cutoff_ts
    train_indices = np.where(is_train)[0]
    oos_indices = np.where(~is_train)[0]

    if len(train_indices) == 0:
        raise ValueError(f"No training rows found before {train_end}")
    if len(oos_indices) == 0:
        raise ValueError(f"No out-of-sample rows found after {train_end}")

    df_train_sub = df_raw.iloc[train_indices].copy().reset_index(drop=True)
    df_oos_sub = df_raw.iloc[oos_indices].copy().reset_index(drop=True)
    n_train_bars = len(df_train_sub)

    # Compute continuous labels on full dataset
    labels_continuous = compute_triple_barrier_labels(
        df_raw, horizon=horizon, k_atr=k_atr, purge_tail=True
    )

    # EMBARGO PURGE: purge the last `horizon` bars of January to eliminate
    # forward boundary leakage into the February Out-of-Sample window!
    labels_continuous[n_train_bars - horizon : n_train_bars] = REGIME_LABEL_PURGED

    y_train_all = labels_continuous[:n_train_bars]
    y_oos_all = labels_continuous[n_train_bars:]

    # Extract feature subsets
    X_train_raw_all = features_df.iloc[train_indices].to_numpy(dtype=np.float64)
    X_oos_raw_all = features_df.iloc[oos_indices].to_numpy(dtype=np.float64)

    # Filter valid training rows:
    # Warmup filter: drop first `warmup_drop_bars` rows (e.g. 240 bars)
    # Embargo filter: drop purged labels (y != -1) and any remaining NaNs
    train_valid_mask = (
        (np.arange(n_train_bars) >= warmup_drop_bars)
        & (y_train_all != REGIME_LABEL_PURGED)
        & (~np.isnan(X_train_raw_all).any(axis=1))
    )

    # For OOS, continuous feature calculation across Q1 means Feb 1 starts with full warmup history!
    # Only filter out purged labels (tail of March) and any NaN rows
    oos_valid_mask = (y_oos_all != REGIME_LABEL_PURGED) & (~np.isnan(X_oos_raw_all).any(axis=1))

    X_train_raw = X_train_raw_all[train_valid_mask]
    y_train = y_train_all[train_valid_mask]
    df_train_clean = df_train_sub.iloc[train_valid_mask].reset_index(drop=True)
    train_timestamps = df_train_clean["datetime"]

    X_oos_raw = X_oos_raw_all[oos_valid_mask]
    y_oos = y_oos_all[oos_valid_mask]
    df_oos_clean = df_oos_sub.iloc[oos_valid_mask].reset_index(drop=True)
    oos_timestamps = df_oos_clean["datetime"]

    # SCALER ISOLATION: Fit strictly on In-Sample January data only!
    scaler: RobustScaler | StandardScaler
    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    scaler.fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_oos_scaled = scaler.transform(X_oos_raw)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_oos": X_oos_scaled,
        "y_oos": y_oos,
        "feature_names": list(REGIME_FEATURE_NAMES),
        "scaler": scaler,
        "X_train_raw": X_train_raw,
        "X_oos_raw": X_oos_raw,
        "df_train": df_train_clean,
        "df_oos": df_oos_clean,
        "train_timestamps": train_timestamps,
        "oos_timestamps": oos_timestamps,
    }
