from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import json
from pathlib import Path
import sys
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


TAPE_VERSION = 1
TAPE_DIRNAME = "five_market_tape"
TAPE_VALUES_FILE = "environment_values.npy"
TAPE_TIME_FILE = "time_ns.npy"
TAPE_MANIFEST_FILE = "manifest.json"


def _datetime_ns(values) -> np.ndarray:
    return (
        pd.to_datetime(values, utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64, copy=False)
    )


def _project_src(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    src = root / "src"
    if not (src / "self_growing_trader").exists():
        raise FileNotFoundError(f"khong tim thay src/self_growing_trader trong {root}")
    return src


def _import_project(project_root: str | Path) -> dict[str, object]:
    src = _project_src(project_root)
    value = str(src)
    if value not in sys.path:
        sys.path.insert(0, value)
    return {
        "config": importlib.import_module("self_growing_trader.config"),
        "features": importlib.import_module("self_growing_trader.features"),
        "feature_contract": importlib.import_module("self_growing_trader.feature_contract"),
        "protocol": importlib.import_module("self_growing_trader.protocol_contract"),
        "environment": importlib.import_module("self_growing_trader.environment"),
        "five": importlib.import_module("self_growing_trader.five_shared_policy"),
    }


def _time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(df.index)
    elif "time" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True, errors="coerce"))
    elif "Datetime" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["Datetime"], utc=True, errors="coerce"))
    else:
        raise ValueError("du lieu can co moc thoi gian de noi vao giao uoc Five")
    if idx.isna().any():
        raise ValueError("moc thoi gian co gia tri khong hop le")
    return idx


def _find_column(df: pd.DataFrame, *names: str) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


@dataclass
class ProjectFiveContract:
    project_root: Path
    config_path: Path
    config: object
    input_count: int
    agent_count: int
    feature_schema_version: int
    feature_schema_hash: str
    environment_columns: tuple[str, ...]
    modules: dict[str, object]

    @classmethod
    def load(
        cls, project_root: str | Path, config_path: str | Path,
        symbol_contract: dict[str, object] | None = None,
    ) -> "ProjectFiveContract":
        root = Path(project_root).resolve()
        modules = _import_project(root)
        config_module = modules["config"]
        protocol = modules["protocol"]
        features = modules["features"]
        five = modules["five"]
        feature_contract = modules["feature_contract"]

        cp = Path(config_path)
        if not cp.is_absolute():
            cp = root / cp
        cp = cp.resolve()
        config = config_module.load_config(cp)
        if symbol_contract:
            point = float(symbol_contract.get("point", 0.0) or 0.0)
            digits = int(symbol_contract.get("digits", -1))
            if point <= 0.0 or digits < 0:
                raise ValueError("symbol_contract can digits va point hop le")
            config = replace(
                config,
                symbol=replace(config.symbol, price_digits=digits, point_size=point, spread_in_points=True),
            )

        input_count = int(protocol.INPUT_COUNT)
        agent_count = int(getattr(five, "FIVE_AGENT_COUNT", 5))
        if input_count != 137:
            raise ValueError(f"giao uoc dau vao hien tai la {input_count}, cong cu nay yeu cau 137")
        if agent_count != 5:
            raise ValueError(f"giao uoc tac nhan hien tai la {agent_count}, cong cu nay yeu cau 5")

        decision_mode = str(getattr(config.he_chuyen_gia, "che_do_quyet_dinh", "") or "")
        if decision_mode != "nam_tac_nhan_dung_chung":
            raise ValueError(
                "cau hinh duoc chon khong o che do nam_tac_nhan_dung_chung; "
                f"dang la {decision_mode!r}"
            )

        environment_columns = (
            "Open", "High", "Low", "Close", "spread_price", "atr", "not_in_shock_regime",
            *tuple(features.BASE_MARKET_FEATURE_COLUMNS),
            *tuple(features.EXTRA_MARKET_FEATURE_COLUMNS),
            *tuple(features.TIMING_FEATURE_COLUMNS),
            *tuple(features.WYCKOFF_FEATURE_COLUMNS),
        )
        # Keep first occurrence only because spread_atr is already part of BASE.
        environment_columns = tuple(dict.fromkeys(environment_columns))
        schema_hash = feature_contract.feature_schema_fingerprint(
            config.features,
            config.symbol,
            input_normalization_mode=str(getattr(config.learning, "input_normalization_mode", "raw") or "raw"),
        )
        return cls(
            project_root=root,
            config_path=cp,
            config=config,
            input_count=input_count,
            agent_count=agent_count,
            feature_schema_version=int(protocol.FEATURE_SCHEMA_VERSION),
            feature_schema_hash=str(schema_hash),
            environment_columns=environment_columns,
            modules=modules,
        )

    def prepare_project_raw(
        self,
        df: pd.DataFrame,
        *,
        default_spread_points: float | None = None,
        allow_neutral_volume: bool = False,
    ) -> pd.DataFrame:
        idx = _time_index(df)
        if not idx.is_monotonic_increasing:
            order = np.argsort(idx.asi8, kind="stable")
            df = df.iloc[order].copy()
            idx = idx[order]
        if idx.has_duplicates:
            keep = ~idx.duplicated(keep="last")
            df = df.loc[keep].copy()
            idx = idx[keep]

        mapping = {}
        for target, candidates in {
            "Open": ("Open", "open"),
            "High": ("High", "high"),
            "Low": ("Low", "low"),
            "Close": ("Close", "close"),
        }.items():
            source = _find_column(df, *candidates)
            if source is None:
                raise ValueError(f"thieu cot {target}")
            mapping[target] = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)

        spread_col = _find_column(df, "Spread", "spread")
        if spread_col is None:
            if default_spread_points is None:
                raise ValueError(
                    "che do Five can cot spread de tao spread_atr va trang thai thuc thi; "
                    "hay cung cap spread hoac --default-spread-points"
                )
            spread = np.full(len(df), float(default_spread_points), dtype=float)
        else:
            spread = pd.to_numeric(df[spread_col], errors="coerce").to_numpy(dtype=float)
        mapping["Spread"] = spread

        tick_col = _find_column(df, "TickVolume", "tick_volume", "volume")
        real_col = _find_column(df, "RealVolume", "real_volume")
        if tick_col is None and real_col is None and not allow_neutral_volume:
            raise ValueError(
                "che do Five can du lieu khoi luong de tao 16 dac trung Wyckoff; "
                "hay cung cap volume/tick_volume/real_volume hoac cho phep khoi luong trung tinh"
            )
        if tick_col is not None:
            mapping["TickVolume"] = pd.to_numeric(df[tick_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            mapping["TickVolume"] = np.ones(len(df), dtype=float)
        if real_col is not None:
            mapping["RealVolume"] = pd.to_numeric(df[real_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        out = pd.DataFrame(mapping)
        out.insert(0, "Datetime", pd.DatetimeIndex(idx).tz_convert("UTC") if idx.tz is not None else idx)
        numeric = [c for c in out.columns if c != "Datetime"]
        if not np.all(np.isfinite(out[numeric].to_numpy(dtype=float))):
            raise ValueError("du lieu dau vao co NaN/vo cuc sau khi chuan hoa cot")
        return out

    def build_feature_frame(self, project_raw: pd.DataFrame) -> pd.DataFrame:
        add_features = self.modules["features"].add_features
        frame = add_features(project_raw, self.config.features, self.config.symbol)
        missing = [c for c in self.environment_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"khung dac trung thieu cot bat buoc: {missing}")
        return frame

    def write_tape(self, feature_frame: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
        out = Path(output_dir)
        tape_dir = out / TAPE_DIRNAME
        tape_dir.mkdir(parents=True, exist_ok=True)
        values = feature_frame.loc[:, self.environment_columns].to_numpy(dtype=np.float32, copy=True)
        times = _datetime_ns(feature_frame["Datetime"])
        np.save(tape_dir / TAPE_VALUES_FILE, values, allow_pickle=False)
        np.save(tape_dir / TAPE_TIME_FILE, times, allow_pickle=False)
        manifest = {
            "version": TAPE_VERSION,
            "rows": int(len(feature_frame)),
            "columns": list(self.environment_columns),
            "dtype": "float32",
            "time_dtype": "int64_ns_utc",
            "input_count": self.input_count,
            "agent_count": self.agent_count,
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_hash": self.feature_schema_hash,
            "input_normalization_mode": str(getattr(self.config.learning, "input_normalization_mode", "raw") or "raw"),
            "config": str(self.config_path),
            "symbol_contract": {
                "digits": int(self.config.symbol.price_digits),
                "point": float(self.config.symbol.point_size),
                "spread_in_points": bool(self.config.symbol.spread_in_points),
            },
        }
        (tape_dir / TAPE_MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _time_to_tape_index(feature_frame: pd.DataFrame) -> dict[int, int]:
        times = _datetime_ns(feature_frame["Datetime"])
        return {int(ts): int(i) for i, ts in enumerate(times)}

    def enrich_index_samples(
        self,
        samples: Iterable[dict],
        raw_df: pd.DataFrame,
        feature_frame: pd.DataFrame,
    ) -> Iterator[dict]:
        raw_times = _time_index(raw_df)
        if raw_times.tz is None:
            raw_times = raw_times.tz_localize("UTC")
        else:
            raw_times = raw_times.tz_convert("UTC")
        raw_ns = _datetime_ns(raw_times)
        tape_index = self._time_to_tape_index(feature_frame)
        atr_by_time = {
            int(ts): float(atr)
            for ts, atr in zip(
                _datetime_ns(feature_frame["Datetime"]),
                feature_frame["atr"].to_numpy(dtype=float),
            )
        }

        for sample in samples:
            start = int(sample["history_start_index"])
            anchor = int(sample["anchor_index"])
            end = int(sample["future_end_index"])
            if start < 0 or end >= len(raw_ns):
                continue
            keys = (int(raw_ns[start]), int(raw_ns[anchor]), int(raw_ns[end]))
            if any(key not in tape_index for key in keys):
                # Feature warm-up can legitimately remove early structures.
                continue
            tape_start, tape_anchor, tape_end = (tape_index[key] for key in keys)
            if not (tape_start <= tape_anchor <= tape_end):
                continue
            # All bars in the requested episode must be contiguous on the feature tape.
            expected_len = end - start + 1
            if tape_end - tape_start + 1 != expected_len:
                continue
            enriched = dict(sample)
            enriched.update({
                "tape_start_index": int(tape_start),
                "tape_anchor_index": int(tape_anchor),
                "tape_end_index": int(tape_end),
                "context_steps": int(tape_anchor - tape_start),  # excludes anchor
                "future_steps": int(tape_end - tape_anchor),
                "model_anchor_atr": float(atr_by_time[keys[1]]),
                "feature_schema_version": int(self.feature_schema_version),
            })
            yield enriched

    def metadata(self, *, tape_manifest: dict[str, object] | None = None) -> dict[str, object]:
        return {
            "storage_mode": "indexed_environment_tape_v1",
            "input_count": self.input_count,
            "agent_count": self.agent_count,
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_hash": self.feature_schema_hash,
            "input_normalization_mode": str(getattr(self.config.learning, "input_normalization_mode", "raw") or "raw"),
            "environment_columns": list(self.environment_columns),
            "tape": tape_manifest or {},
            "causal_contract": {
                "context_excludes_anchor": True,
                "anchor_is_confirm_index": True,
                "future_never_enters_model_context": True,
                "dynamic_position_account_execution_built_by_environment": True,
            },
        }


class ProjectFiveDataset:
    """Mo tap du lieu chi muc, dung truc tiep voi TradingEnvironment cua du an."""

    def __init__(self, dataset_dir: str | Path, project_root: str | Path, config_path: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        tape_dir = self.dataset_dir / TAPE_DIRNAME
        manifest_path = tape_dir / TAPE_MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.contract = ProjectFiveContract.load(
            project_root, config_path, self.manifest.get("symbol_contract")
        )
        if int(self.manifest.get("input_count", -1)) != self.contract.input_count:
            raise ValueError("input_count cua tap du lieu khong khop du an")
        if int(self.manifest.get("feature_schema_version", -1)) != self.contract.feature_schema_version:
            raise ValueError("feature_schema_version cua tap du lieu khong khop du an")
        if str(self.manifest.get("feature_schema_hash", "")) != self.contract.feature_schema_hash:
            raise ValueError("feature_schema_hash cua tap du lieu khong khop cau hinh hien tai")
        self.values = np.load(tape_dir / TAPE_VALUES_FILE, mmap_mode="r", allow_pickle=False)
        self.times = np.load(tape_dir / TAPE_TIME_FILE, mmap_mode="r", allow_pickle=False)
        sampled_times = np.asarray(self.times[: min(len(self.times), 1024)], dtype=np.int64)
        magnitude = float(np.median(np.abs(sampled_times))) if sampled_times.size else 0.0
        self.time_unit = "ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s"
        self.columns = tuple(self.manifest["columns"])

    def close(self) -> None:
        """Dong mmap de Windows khong giu khoa cac tep tape sau khi dung."""
        for array in (getattr(self, "values", None), getattr(self, "times", None)):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "ProjectFiveDataset":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def environment_for_sample(self, sample: dict):
        start = int(sample["tape_start_index"])
        end = int(sample["tape_end_index"])
        if not (0 <= start <= end < len(self.values)):
            raise IndexError("chi so tap phim vuot khoi bang du lieu")
        values = np.asarray(self.values[start:end + 1], dtype=np.float64)
        frame = pd.DataFrame(values, columns=self.columns)
        frame.insert(0, "Datetime", pd.to_datetime(
            np.asarray(self.times[start:end + 1]), unit=self.time_unit, utc=True
        ))
        env_class = self.contract.modules["environment"].TradingEnvironment
        return env_class(frame, self.contract.config)

    def context_and_anchor(self, sample: dict) -> tuple[np.ndarray, np.ndarray]:
        """Tra ve [T,137] qua khu (khong gom neo) va observation 137 tai neo.

        Day la trang thai trung tinh truoc khi mo vi the. Tu neo tro di, phai de
        TradingEnvironment tao position/account/execution dong theo hanh dong thuc te.
        """
        env = self.environment_for_sample(sample)
        anchor_local = int(sample["tape_anchor_index"]) - int(sample["tape_start_index"])
        context = [env.observation(i) for i in range(anchor_local)]
        anchor = env.observation(anchor_local)
        if context:
            context_array = np.stack(context).astype(np.float32)
        else:
            context_array = np.zeros((0, self.contract.input_count), dtype=np.float32)
        return context_array, np.asarray(anchor, dtype=np.float32)


def iter_index_samples(dataset_dir: str | Path) -> Iterator[dict]:
    root = Path(dataset_dir)
    for path in sorted(root.glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            n = len(z["anchor_index"])
            for i in range(n):
                sample = {k: z[k][i] for k in z.files}
                yield sample
