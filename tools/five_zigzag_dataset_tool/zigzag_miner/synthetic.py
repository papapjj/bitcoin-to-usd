from __future__ import annotations
import numpy as np
import pandas as pd


def repeated_swing_market(repeats: int = 80, leg_bars: int = 8, seed: int = 7) -> pd.DataFrame:
    """Synthetic alternating up/down market used for deterministic smoke tests."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    # Up leg ~ +4, down leg ~ -2, repeated; tiny noise inside each leg.
    for _ in range(repeats):
        for delta in (4.0, -2.0, 3.5, -1.8):
            step = delta / leg_bars
            for _ in range(leg_bars):
                prices.append(prices[-1] + step + rng.normal(0.0, 0.015))
    close = np.asarray(prices, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    wick = 0.05
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    vol = np.full(len(close), 100.0)
    idx = pd.date_range("2020-01-01", periods=len(close), freq="min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
