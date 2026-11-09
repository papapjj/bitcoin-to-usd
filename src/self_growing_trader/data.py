from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AppConfig
from .features import add_features


def load_market_data(config: AppConfig, rows: int | None = None) -> pd.DataFrame:
    path = Path(config.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, nrows=rows)
    required = {"Datetime", "Open", "High", "Low", "Close", "Spread"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return add_features(df, config.features, config.symbol)


def split_time_ordered(df: pd.DataFrame, train: float, validation: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError("Expected train > 0, validation > 0, and train + validation < 1.")
    n = len(df)
    train_end = int(n * train)
    validation_end = int(n * (train + validation))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:validation_end].reset_index(drop=True),
        df.iloc[validation_end:].reset_index(drop=True),
    )

