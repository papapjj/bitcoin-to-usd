"""Compatibility facade for the current MT5 feature-parity workflow.

The old module claimed numerical equivalence while calculating only 34 slots
with divergent MACD/spread semantics. It is retained only for import
compatibility and delegates to the canonical 58-market plus 20-tail reference.
Use :mod:`self_growing_trader.feature_contract` with an EA FeatureAuditFile for
an actual 137-input parity check.
"""
from __future__ import annotations

import pandas as pd

from .config import FeatureConfig, SymbolConfig
from .features import MARKET_FEATURE_COLUMNS, TAIL_FEATURE_COLUMNS, add_features, compute_atr

FEATURE_COLUMNS = list(MARKET_FEATURE_COLUMNS) + list(TAIL_FEATURE_COLUMNS)


def compute_atr_mql5(close, high, low, period: int = 14):
    frame = pd.DataFrame({"Close": close, "High": high, "Low": low})
    return compute_atr(frame, period)


def compute_ema_mql5(close, period: int):
    return close.ewm(span=max(int(period), 1), adjust=False).mean()


def compute_rsi_mql5(close, period: int):
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / max(int(period), 1), min_periods=int(period), adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / max(int(period), 1), min_periods=int(period), adjust=False).mean()
    rs = gain / loss.replace(0.0, 1e-10)
    return (100.0 - 100.0 / (1.0 + rs) - 50.0) / 50.0


def compute_features_mql5(df: pd.DataFrame, feature_config: FeatureConfig | None = None, symbol: SymbolConfig | None = None) -> pd.DataFrame:
    return add_features(df, feature_config or FeatureConfig(), symbol or SymbolConfig())
