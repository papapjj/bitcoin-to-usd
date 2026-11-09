from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterator, List, Tuple
import numpy as np
import pandas as pd

from .causal_zigzag import Pivot, causal_zigzag, compute_atr
from .structures import PatternSpec

REQUIRED = ("open", "high", "low", "close")


def _validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("empty dataframe")
    if not np.all(np.isfinite(df[list(REQUIRED)].to_numpy(dtype=np.float64))):
        raise ValueError("OHLC contains NaN/inf")
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bool(bad.any()):
        raise ValueError("invalid OHLC rows: high/low inconsistent with open/close")


def _leg_features(a: Pivot, b: Pivot) -> Dict[str, float]:
    delta = float(b.price - a.price)
    direction = 1 if delta > 0 else -1 if delta < 0 else 0
    bars = int(b.pivot_index - a.pivot_index)
    base_atr = max(float(a.atr_at_pivot), np.finfo(np.float64).eps)
    return {
        "delta": delta,
        "abs_atr": abs(delta) / base_atr,
        "direction": direction,
        "bars": bars,
    }


def _matches(pivs: List[Pivot], start: int, spec: PatternSpec) -> Tuple[bool, List[Dict[str, float]]]:
    feats: List[Dict[str, float]] = []
    prev_abs_atr = None
    for j, rule in enumerate(spec.legs):
        f = _leg_features(pivs[start + j], pivs[start + j + 1])
        feats.append(f)
        if rule.direction and f["direction"] != rule.direction:
            return False, feats
        if not (rule.min_atr <= f["abs_atr"] <= rule.max_atr):
            return False, feats
        if not (rule.min_bars <= f["bars"] <= rule.max_bars):
            return False, feats
        if prev_abs_atr is not None:
            ratio = f["abs_atr"] / max(prev_abs_atr, np.finfo(np.float64).eps)
            if not (rule.min_ratio_prev <= ratio <= rule.max_ratio_prev):
                return False, feats
            f["ratio_prev"] = ratio
        else:
            f["ratio_prev"] = np.nan
        prev_abs_atr = f["abs_atr"]
    return True, feats


def mine_dataframe(
    df: pd.DataFrame,
    spec: PatternSpec,
    *,
    atr_period: int = 14,
    zigzag_atr_mult: float = 1.0,
    atr_mode: str = "mt5_sma",
) -> Iterator[dict]:
    """Yield causal structure samples from an OHLC dataframe.

    Anchor = confirm_index of the final pivot, so every value used to classify
    the structure was knowable at or before the anchor.
    """
    _validate_df(df)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns else np.zeros(len(df), dtype=np.float64)
    atr = compute_atr(h, l, c, atr_period, mode=atr_mode)
    pivs = causal_zigzag(h, l, c, atr_period=atr_period, atr_mult=zigzag_atr_mult, atr_mode=atr_mode)

    need = len(spec.legs) + 1
    last_anchor = -10**18
    for start in range(0, len(pivs) - need + 1):
        ok, leg_feats = _matches(pivs, start, spec)
        if not ok:
            continue
        selected = pivs[start:start + need]
        anchor = int(selected[-1].confirm_index)
        if anchor - last_anchor < spec.min_gap_bars:
            continue
        hist_start = anchor - spec.history_bars + 1
        future_end = anchor + spec.future_bars
        if hist_start < 0 or future_end >= len(df):
            continue

        anchor_atr = max(float(atr[anchor]), np.finfo(np.float64).eps)
        anchor_close = float(c[anchor])
        history = np.column_stack([o[hist_start:anchor+1], h[hist_start:anchor+1], l[hist_start:anchor+1], c[hist_start:anchor+1], v[hist_start:anchor+1]])
        future = np.column_stack([o[anchor+1:future_end+1], h[anchor+1:future_end+1], l[anchor+1:future_end+1], c[anchor+1:future_end+1], v[anchor+1:future_end+1]])

        # Normalized history preserves local raw geometry without absolute price scale.
        history_norm = history.copy()
        history_norm[:, :4] = (history_norm[:, :4] - anchor_close) / anchor_atr
        if np.max(np.abs(history_norm[:, 4])) > 0:
            history_norm[:, 4] = history_norm[:, 4] / max(np.mean(np.abs(history_norm[:, 4])), 1.0)

        future_high = float(np.max(h[anchor+1:future_end+1]))
        future_low = float(np.min(l[anchor+1:future_end+1]))
        final_close = float(c[future_end])
        long_mfe = (future_high - anchor_close) / anchor_atr
        long_mae = (future_low - anchor_close) / anchor_atr
        long_final = (final_close - anchor_close) / anchor_atr
        short_mfe = (anchor_close - future_low) / anchor_atr
        short_mae = (anchor_close - future_high) / anchor_atr
        short_final = (anchor_close - final_close) / anchor_atr
        outcome = {
            # Ten cu duoc giu de doc tap du lieu 0.1; nghia cua chung la huong mua.
            "mfe_atr": long_mfe,
            "mae_atr": long_mae,
            "final_return_atr": long_final,
            "long_mfe_atr": long_mfe,
            "long_mae_atr": long_mae,
            "long_final_return_atr": long_final,
            "short_mfe_atr": short_mfe,
            "short_mae_atr": short_mae,
            "short_final_return_atr": short_final,
        }

        yield {
            "pattern": spec.name,
            "history_start_index": int(hist_start),
            "anchor_index": anchor,
            "future_end_index": int(future_end),
            "anchor_time": str(df.index[anchor]),
            "anchor_close": anchor_close,
            "anchor_atr": anchor_atr,
            "pivots": [asdict(p) for p in selected],
            "legs": leg_feats,
            "history": history.astype(np.float32),
            "history_norm": history_norm.astype(np.float32),
            "future": future.astype(np.float32),
            "outcome": outcome,
        }
        last_anchor = anchor
