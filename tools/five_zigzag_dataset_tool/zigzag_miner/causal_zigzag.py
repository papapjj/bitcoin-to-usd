from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import numpy as np


@dataclass(frozen=True)
class Pivot:
    pivot_index: int
    confirm_index: int
    price: float
    kind: str  # "H" or "L"
    threshold: float
    atr_at_pivot: float


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    *,
    mode: str = "mt5_sma",
) -> np.ndarray:
    """ATR nhan qua. ``mt5_sma`` khop dinh nghia dac trung cua du an."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high/low/close must have identical length")
    n = len(close)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if period < 1:
        raise ValueError("ATR period must be >= 1")

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev = close[:-1]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev),
            np.abs(low[1:] - prev),
        ])

    mode = str(mode or "mt5_sma").strip().lower()
    atr = np.empty(n, dtype=np.float64)
    if mode in {"mt5_sma", "sma", "simple"}:
        # MT5 iATR trong giao uoc hien tai cua du an dung trung binh dong don
        # cua true range. Vung khoi dong dung trung binh tich luy de ZigZag van
        # co nguong nhan qua truoc khi du period; tu period-1 tro di khop SMA.
        csum = np.cumsum(tr, dtype=np.float64)
        for i in range(n):
            start = max(0, i - period + 1)
            total = csum[i] - (csum[start - 1] if start else 0.0)
            atr[i] = total / float(i - start + 1)
        return atr
    if mode in {"wilder", "rma"}:
        seed = min(period, n)
        running = 0.0
        for i in range(seed):
            running += tr[i]
            atr[i] = running / (i + 1)
        for i in range(seed, n):
            atr[i] = ((period - 1) * atr[i - 1] + tr[i]) / period
        return atr
    raise ValueError(f"unsupported ATR mode: {mode}")


def causal_zigzag(
    high: Iterable[float],
    low: Iterable[float],
    close: Iterable[float],
    *,
    atr_period: int = 14,
    atr_mult: float = 1.0,
    min_abs_reversal: float = 0.0,
    atr_mode: str = "mt5_sma",
) -> List[Pivot]:
    """Build a causal/confirmed ZigZag.

    A candidate extreme becomes a Pivot only after a reversal large enough to
    confirm it. `pivot_index` is where the extreme occurred; `confirm_index`
    is when that information first became available. Training anchors MUST use
    confirm_index, not pivot_index.
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if not (len(h) == len(l) == len(c)):
        raise ValueError("high/low/close must have identical length")
    if len(c) < 2:
        return []
    if atr_mult <= 0:
        raise ValueError("atr_mult must be > 0")

    atr = compute_atr(h, l, c, atr_period, mode=atr_mode)
    eps = np.finfo(np.float64).eps

    def threshold(i: int) -> float:
        return max(float(atr[i]) * atr_mult, float(min_abs_reversal), eps)

    pivots: List[Pivot] = []
    direction = 0  # 0 unknown, +1 rising leg (search high), -1 falling leg (search low)

    low_i = high_i = 0
    low_p = float(l[0])
    high_p = float(h[0])
    low_thr = high_thr = threshold(0)

    for i in range(1, len(c)):
        # Bootstrap: track both extremes until one side is causally confirmed.
        if direction == 0:
            if l[i] < low_p:
                low_i, low_p, low_thr = i, float(l[i]), threshold(i)
            if h[i] > high_p:
                high_i, high_p, high_thr = i, float(h[i]), threshold(i)

            # If market has risen enough from the candidate low, confirm low.
            if float(h[i]) - low_p >= low_thr and low_i < i:
                pivots.append(Pivot(low_i, i, low_p, "L", low_thr, float(atr[low_i])))
                direction = +1
                high_i, high_p, high_thr = i, float(h[i]), threshold(i)
                continue
            # Or fallen enough from candidate high, confirm high.
            if high_p - float(l[i]) >= high_thr and high_i < i:
                pivots.append(Pivot(high_i, i, high_p, "H", high_thr, float(atr[high_i])))
                direction = -1
                low_i, low_p, low_thr = i, float(l[i]), threshold(i)
                continue

        elif direction == +1:
            # Up-leg: candidate high can repaint internally, but is not emitted.
            if h[i] >= high_p:
                high_i, high_p, high_thr = i, float(h[i]), threshold(i)
            if high_p - float(l[i]) >= high_thr and high_i < i:
                pivots.append(Pivot(high_i, i, high_p, "H", high_thr, float(atr[high_i])))
                direction = -1
                low_i, low_p, low_thr = i, float(l[i]), threshold(i)

        else:  # direction == -1
            if l[i] <= low_p:
                low_i, low_p, low_thr = i, float(l[i]), threshold(i)
            if float(h[i]) - low_p >= low_thr and low_i < i:
                pivots.append(Pivot(low_i, i, low_p, "L", low_thr, float(atr[low_i])))
                direction = +1
                high_i, high_p, high_thr = i, float(h[i]), threshold(i)

    return pivots
