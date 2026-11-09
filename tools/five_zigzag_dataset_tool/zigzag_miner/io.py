from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List
import json
import numpy as np
import pandas as pd


TIMEFRAME_SECONDS = {
    "M1": 60, "M2": 120, "M3": 180, "M5": 300, "M15": 900,
    "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400,
}


def trim_sparse_timeframe_prefix(
    df: pd.DataFrame,
    timeframe: str,
    *,
    minimum_consecutive_bars: int = 256,
) -> tuple[pd.DataFrame, int]:
    """Remove broker sparse prehistory before the first true timeframe run.

    Some MT5 servers return one old M1 row per day before dense M1 history.
    Treating those rows as adjacent minutes corrupts ATR, higher-timeframe
    features, episode clocks and PPO discounting. Market-session gaps after the
    dense history starts remain intact and can be filtered per episode later.
    """
    key = str(timeframe or "").upper()
    if key not in TIMEFRAME_SECONDS or not isinstance(df.index, pd.DatetimeIndex):
        return df, 0
    required = max(2, int(minimum_consecutive_bars))
    if len(df) < required:
        return df, 0
    ns = df.index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    expected_ns = int(TIMEFRAME_SECONDS[key]) * 1_000_000_000
    valid_edge = np.diff(ns) == expected_ns
    run = 0
    start = None
    for edge_index, valid in enumerate(valid_edge):
        run = run + 1 if bool(valid) else 0
        if run >= required - 1:
            start = edge_index - run + 1
            break
    if start is None:
        raise ValueError(
            f"du lieu {key} khong co {required} thanh lien tuc dung cadence; "
            "khong the tao feature/tap huan luyen chinh xac"
        )
    removed = int(start)
    return (df.iloc[removed:].copy() if removed else df), removed


def load_ohlc(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".npz":
        z = np.load(path, allow_pickle=False)
        cols = {k: z[k] for k in z.files}
        df = pd.DataFrame(cols)
    elif ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported input format: {ext}")

    # MT5 exports commonly use time/tick_volume.
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    if "time" in df.columns:
        time_values = df["time"]
        if pd.api.types.is_numeric_dtype(time_values.dtype):
            finite = pd.to_numeric(time_values, errors="coerce").dropna()
            magnitude = float(finite.abs().median()) if not finite.empty else 0.0
            # Accept legacy MT5 NPZ epoch seconds while standardizing new files
            # on epoch nanoseconds. Pandas 3 may preserve second resolution when
            # astype("int64") is called without an explicit datetime64[ns] cast.
            unit = "ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s"
            parsed = pd.to_datetime(time_values, unit=unit, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(time_values, errors="coerce", utc=True)
        if parsed.notna().all():
            df = df.drop(columns=["time"])
            df.index = parsed
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index(kind="stable")
        if df.index.has_duplicates:
            df = df.loc[~df.index.duplicated(keep="last")].copy()
    return df


def write_npz_shards(samples: Iterable[dict], output_dir: str | Path, shard_size: int = 10000) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_rows: List[dict] = []
    shard = []
    shard_id = 0
    total = 0

    def flush(items: List[dict], sid: int) -> None:
        if not items:
            return
        np.savez_compressed(
            out / f"shard_{sid:05d}.npz",
            history=np.stack([x["history"] for x in items]),
            history_norm=np.stack([x["history_norm"] for x in items]),
            future=np.stack([x["future"] for x in items]),
            anchor_index=np.asarray([x["anchor_index"] for x in items], dtype=np.int64),
            anchor_close=np.asarray([x["anchor_close"] for x in items], dtype=np.float64),
            anchor_atr=np.asarray([x["anchor_atr"] for x in items], dtype=np.float64),
            mfe_atr=np.asarray([x["outcome"]["mfe_atr"] for x in items], dtype=np.float32),
            mae_atr=np.asarray([x["outcome"]["mae_atr"] for x in items], dtype=np.float32),
            final_return_atr=np.asarray([x["outcome"]["final_return_atr"] for x in items], dtype=np.float32),
            long_mfe_atr=np.asarray([x["outcome"]["long_mfe_atr"] for x in items], dtype=np.float32),
            long_mae_atr=np.asarray([x["outcome"]["long_mae_atr"] for x in items], dtype=np.float32),
            long_final_return_atr=np.asarray([x["outcome"]["long_final_return_atr"] for x in items], dtype=np.float32),
            short_mfe_atr=np.asarray([x["outcome"]["short_mfe_atr"] for x in items], dtype=np.float32),
            short_mae_atr=np.asarray([x["outcome"]["short_mae_atr"] for x in items], dtype=np.float32),
            short_final_return_atr=np.asarray([x["outcome"]["short_final_return_atr"] for x in items], dtype=np.float32),
        )
        with (out / f"shard_{sid:05d}.jsonl").open("w", encoding="utf-8") as f:
            for x in items:
                row = {k: x[k] for k in ("pattern", "anchor_index", "anchor_time", "anchor_close", "anchor_atr", "pivots", "legs", "outcome")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for sample in samples:
        shard.append(sample)
        total += 1
        if len(shard) >= shard_size:
            flush(shard, shard_id)
            shard_id += 1
            shard = []
    flush(shard, shard_id)
    shard_count = shard_id + (1 if shard else 0)
    summary = {"samples": total, "shards": shard_count, "format": "npz+jsonl", "shard_size": shard_size}
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def iter_dataset(output_dir: str | Path) -> Iterator[dict]:
    out = Path(output_dir)
    for path in sorted(out.glob("shard_*.npz")):
        z = np.load(path, allow_pickle=False)
        n = len(z["anchor_index"])
        for i in range(n):
            yield {k: z[k][i] for k in z.files}


def write_indexed_five_shards(
    samples: Iterable[dict],
    output_dir: str | Path,
    *,
    shard_size: int = 100000,
    dataset_metadata: dict | None = None,
) -> dict:
    """Ghi chi muc mau gon cho tap Five.

    Du lieu thi truong theo tung thanh gia duoc ghi mot lan trong ``five_market_tape``;
    moi mau chi giu cac chi so vao bang do. Cach nay tranh nhan ban history/future
    hang trieu lan va buoc bo hoc dung chuoi theo thu tu thoi gian.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard: list[dict] = []
    shard_id = 0
    total = 0

    scalar_fields = (
        "history_start_index", "anchor_index", "future_end_index",
        "tape_start_index", "tape_anchor_index", "tape_end_index",
        "context_steps", "future_steps", "anchor_close", "anchor_atr",
        "model_anchor_atr", "feature_schema_version",
    )
    outcome_fields = (
        "mfe_atr", "mae_atr", "final_return_atr",
        "long_mfe_atr", "long_mae_atr", "long_final_return_atr",
        "short_mfe_atr", "short_mae_atr", "short_final_return_atr",
    )

    def flush(items: list[dict], sid: int) -> None:
        if not items:
            return
        payload: dict[str, np.ndarray] = {}
        for field in scalar_fields:
            if all(field in item for item in items):
                values = [item[field] for item in items]
                if field.endswith("index") or field.endswith("steps") or field == "feature_schema_version":
                    payload[field] = np.asarray(values, dtype=np.int64)
                else:
                    payload[field] = np.asarray(values, dtype=np.float64)
        for field in outcome_fields:
            if all(field in item.get("outcome", {}) for item in items):
                payload[field] = np.asarray([item["outcome"][field] for item in items], dtype=np.float32)
        np.savez_compressed(out / f"shard_{sid:05d}.npz", **payload)
        with (out / f"shard_{sid:05d}.jsonl").open("w", encoding="utf-8") as f:
            for item in items:
                row = {
                    "pattern": item.get("pattern", ""),
                    "anchor_time": item.get("anchor_time", ""),
                    "pivots": item.get("pivots", []),
                    "legs": item.get("legs", []),
                    "outcome": item.get("outcome", {}),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for sample in samples:
        shard.append(sample)
        total += 1
        if len(shard) >= int(shard_size):
            flush(shard, shard_id)
            shard_id += 1
            shard = []
    if shard:
        flush(shard, shard_id)
        shard_id += 1

    summary = {
        "samples": total,
        "shards": shard_id,
        "format": "indexed-npz+jsonl+tape",
        "shard_size": int(shard_size),
        **(dataset_metadata or {}),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
