"""
Export M5 Dataset Utility
Resamples M1 OHLCV candlestick data into an authentic M5 candlestick dataset.

Usage:
    python scripts/export_m5_dataset.py --input "C:\\Users\\Admin\\Desktop\\BTCUSDm.csv" \
                                       --output "C:\\Users\\Admin\\Desktop\\BTCUSDm_M5.csv" \
                                       --start_date 2025.01.15 --end_date 2025.02.28
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from typing import Optional, Tuple

import pandas as pd


def detect_delimiter(sample_text: str) -> str:
    """Detects delimiter from first line of text."""
    first_line = sample_text.splitlines()[0] if sample_text else ""
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    if ";" in first_line:
        return ";"
    return r"[,\t]"


def load_raw_candles(
    file_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads raw candles from CSV or TSV file, optionally pre-filtering by date.
    Fast line-by-line streaming is used for large files (> 10MB) when start_date is set.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)

    if file_size > 10_000_000 and start_date is not None:
        # Pre-filter lines to minimize memory usage and maximize speed
        norm_start = start_date.replace("-", ".")[:10]
        norm_end = (end_date.replace("-", ".")[:10]) if end_date else "9999.99.99"

        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            lines.append(header)
            sep = detect_delimiter(header)
            for line in f:
                # Fast check on date prefix
                date_part = line[:10]
                if date_part >= norm_start:
                    if date_part > norm_end:
                        break
                    lines.append(line)

        raw_text = "".join(lines)
        df = pd.read_csv(io.StringIO(raw_text), sep=sep)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.readline()
        sep = detect_delimiter(sample)
        df = pd.read_csv(file_path, sep=sep)

    return df


def normalize_candle_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Normalizes column names case-insensitively and creates unified Datetime column.
    Returns normalized DataFrame and mapping of standard roles to original columns.
    """
    col_norm = {c: c.strip("<>").strip() for c in df.columns}
    df = df.rename(columns=col_norm)
    col_lower = {c.lower(): c for c in df.columns}

    # Resolve Datetime
    if "datetime" in col_lower:
        dt_col = col_lower["datetime"]
        datetime_series = pd.to_datetime(df[dt_col], errors="coerce")
    elif "time" in col_lower and "date" not in col_lower:
        dt_col = col_lower["time"]
        datetime_series = pd.to_datetime(df[dt_col], errors="coerce")
    elif "date" in col_lower and "time" in col_lower:
        d_col = col_lower["date"]
        t_col = col_lower["time"]
        datetime_series = pd.to_datetime(
            df[d_col].astype(str) + " " + df[t_col].astype(str),
            errors="coerce",
        )
    elif "timestamp" in col_lower:
        dt_col = col_lower["timestamp"]
        datetime_series = pd.to_datetime(df[dt_col], errors="coerce")
    else:
        raise ValueError(f"Unable to resolve Datetime column from headers: {list(df.columns)}")

    df["Datetime"] = datetime_series

    # Identify standard price and volume columns
    open_col = col_lower.get("open")
    high_col = col_lower.get("high")
    low_col = col_lower.get("low")
    close_col = col_lower.get("close")
    spread_col = col_lower.get("spread")
    tickvol_col = col_lower.get("tickvol", col_lower.get("tick_vol", col_lower.get("volume")))
    vol_col = col_lower.get("vol", col_lower.get("realvol", col_lower.get("real_vol")))

    missing = [name for name, c in [("Open", open_col), ("High", high_col), ("Low", low_col), ("Close", close_col)] if c is None]
    if missing:
        raise ValueError(f"Missing required price column(s): {missing}")

    role_map = {
        "Open": open_col,
        "High": high_col,
        "Low": low_col,
        "Close": close_col,
        "Spread": spread_col,
        "TickVol": tickvol_col,
        "Vol": vol_col,
    }

    return df, role_map


def resample_m1_to_m5(
    input_path: str,
    output_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rule: str = "5min",
    delimiter: str = ",",
    warmup_days: int = 2,
) -> pd.DataFrame:
    """
    Resamples M1 OHLCV candlestick data to M5 frequency:
      Open -> 'first'
      High -> 'max'
      Low -> 'min'
      Close -> 'last'
      TickVol -> 'sum'
      Vol -> 'sum'
      Spread -> 'median'

    Args:
        input_path: Path to source M1 CSV or TSV file.
        output_path: Optional destination path for exported M5 dataset.
        start_date: Optional start date filter (e.g. '2025.01.15').
        end_date: Optional end date filter (e.g. '2025.02.28').
        rule: Target resampling frequency (default: '5min').
        delimiter: Output delimiter for saved file (default: ',').
        warmup_days: Days of historical data before start_date to retain for indicator warmup (default: 2).

    Returns:
        pd.DataFrame containing resampled M5 bars.
    """
    t0 = time.perf_counter()
    load_start = start_date
    if start_date and warmup_days > 0:
        start_dt = pd.to_datetime(start_date.replace(".", "-"))
        warmup_dt = start_dt - pd.Timedelta(days=warmup_days)
        load_start = warmup_dt.strftime("%Y.%m.%d")

    raw_df = load_raw_candles(input_path, start_date=load_start, end_date=end_date)
    norm_df, role_map = normalize_candle_columns(raw_df)

    # Set Datetime index and sort
    norm_df = norm_df.dropna(subset=["Datetime"]).sort_values("Datetime").set_index("Datetime")

    # Filter date range if specified (including warmup)
    if load_start:
        filter_start_dt = pd.to_datetime(load_start.replace(".", "-") + " 00:00:00")
        norm_df = norm_df[norm_df.index >= filter_start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date.replace(".", "-") + " 23:59:59")
        norm_df = norm_df[norm_df.index <= end_dt]

    if norm_df.empty:
        raise ValueError(f"No bars found within date window [{start_date}, {end_date}].")

    # Build aggregation mapping
    agg_map = {
        role_map["Open"]: "first",
        role_map["High"]: "max",
        role_map["Low"]: "min",
        role_map["Close"]: "last",
    }
    if role_map["TickVol"]:
        agg_map[role_map["TickVol"]] = "sum"
    if role_map["Vol"]:
        agg_map[role_map["Vol"]] = "sum"
    if role_map["Spread"]:
        agg_map[role_map["Spread"]] = "median"

    # Resample left-labeled, left-closed
    resampled = norm_df.resample(rule, label="left", closed="left").agg(agg_map).dropna()

    # Rename to standard canonical columns
    rename_final = {
        role_map["Open"]: "Open",
        role_map["High"]: "High",
        role_map["Low"]: "Low",
        role_map["Close"]: "Close",
    }
    if role_map["TickVol"]:
        rename_final[role_map["TickVol"]] = "TickVol"
    else:
        resampled["TickVol"] = 100.0

    if role_map["Vol"]:
        rename_final[role_map["Vol"]] = "Vol"
    else:
        resampled["Vol"] = 0.0

    if role_map["Spread"]:
        rename_final[role_map["Spread"]] = "Spread"
    else:
        resampled["Spread"] = 250.0

    resampled = resampled.rename(columns=rename_final)
    resampled = resampled.reset_index()

    # Format Datetime
    resampled["Datetime"] = resampled["Datetime"].dt.strftime("%Y.%m.%d %H:%M:%S")

    # Column ordering
    cols = ["Datetime", "Open", "High", "Low", "Close", "TickVol", "Vol", "Spread"]
    resampled = resampled[cols]

    elapsed = time.perf_counter() - t0
    print(f"[export_m5_dataset] Resampled {len(raw_df)} M1 bars -> {len(resampled)} M5 bars in {elapsed:.2f}s")

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        resampled.to_csv(output_path, sep=delimiter, index=False)
        print(f"[export_m5_dataset] Saved M5 dataset to: {output_path}")

    return resampled


def main():
    parser = argparse.ArgumentParser(description="Resample M1 candle dataset to M5 frequency.")
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\Admin\Desktop\BTCUSDm.csv",
        help="Path to source M1 dataset (TSV or CSV).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\Admin\Desktop\BTCUSDm_M5.csv",
        help="Path to save resampled M5 dataset.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2025.01.15",
        help="Start date in YYYY.MM.DD format (default: 2025.01.15).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025.02.28",
        help="End date in YYYY.MM.DD format (default: 2025.02.28).",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="5min",
        help="Resample rule frequency (default: 5min).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Output delimiter (default: ',').",
    )
    parser.add_argument(
        "--warmup_days",
        type=int,
        default=2,
        help="Days of historical warmup before start_date (default: 2).",
    )

    args = parser.parse_args()
    resample_m1_to_m5(
        input_path=args.input,
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        rule=args.rule,
        delimiter=args.delimiter,
        warmup_days=args.warmup_days,
    )


if __name__ == "__main__":
    main()
