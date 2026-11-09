"""Versioned 137-input feature contract and MT5 parity harness.

Python is the numerical reference. MT5 exports exact runtime feature rows using
``FeatureAuditFile``; this module compares the export with a canonical CSV
fixture before a policy is eligible for promotion. It intentionally does not
pretend that a static 34-column proxy proves parity.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import FeatureConfig, SymbolConfig
from .features import (
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
    add_features,
)
from .input_normalization import (
    INPUT_NORMALIZATION_RAW,
    input_normalization_payload,
    normalize_input_vector,
)
from .protocol_contract import (
    FEATURE_SCHEMA_VERSION,
    BASE_MARKET_FEATURE_COUNT,
    EXTRA_MARKET_OFFSET,
    EXECUTION_OFFSET,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    MARKET_FEATURE_COUNT,
    POSITION_OFFSET,
    POSITION_FEATURE_COUNT,
    TIMING_FEATURE_COUNT,
    WYCKOFF_OFFSET,
    WYCKOFF_FEATURE_COUNT,
    finite_vector,
    position_count_reference,
)

# ``ComputeAllInputs`` uses the exact closed-bar timestamp while
# ``PositionsSnapshotPipe`` samples ``TimeCurrent`` and exports age with one
# decimal place.  The audit also serializes both timestamps to whole minutes,
# so a same-position age can differ by up to one bar plus 0.05-bar rounding.
# Keep this tolerance local to the age slots; every market/account/execution
# slot remains under the strict vector tolerances below.
POSITION_AGE_AUDIT_QUANTIZATION_BARS = 1.05

# MT5's exported audit contains the live spread, while the broker's historical
# OHLC export contains only a bar-level spread snapshot.  The static fixture
# therefore cannot reconstruct the two tick-level spread-derived market slots.
# We only permit a narrow runtime override for those slots, and still verify
# that the spread ratio is consistent with the duplicated execution slot.
SPREAD_ATR_FEATURE_INDEX = MARKET_FEATURE_COLUMNS.index("spread_atr")
SPREAD_PERCENTILE_FEATURE_INDEX = EXTRA_MARKET_OFFSET + (
    MARKET_FEATURE_COLUMNS.index("spread_percentile") - BASE_MARKET_FEATURE_COUNT
)
SPREAD_EXECUTION_FEATURE_INDEX = EXECUTION_OFFSET

# MT5 model-4 exports real-tick volume while the historical reference is a
# bar-level CSV.  The first six Wyckoff features are therefore allowed a small
# explicit numeric envelope.  This is not a global relaxation: every other
# feature remains under the strict ``atol``/``rtol`` pair below, and a larger
# deviation still invalidates parity.
DYNAMIC_VOLUME_PARITY_FEATURE_INDEXES = tuple(
    range(WYCKOFF_OFFSET, WYCKOFF_OFFSET + 6)
)
DEFAULT_DYNAMIC_VOLUME_PARITY_ATOL = 2.5e-4


def feature_schema_payload(
    feature_config: FeatureConfig,
    symbol: SymbolConfig,
    *,
    session_timezone: str = "UTC",
    spread_semantics: str = "ask_minus_bid_price",
    mt5_server_utc_offset_hours: int = 0,
    input_normalization_mode: str = INPUT_NORMALIZATION_RAW,
) -> dict[str, object]:
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "market_columns": list(MARKET_FEATURE_COLUMNS[:MARKET_FEATURE_COUNT]),
        "market_feature_count": MARKET_FEATURE_COUNT,
        "timing_columns": list(TIMING_FEATURE_COLUMNS[:TIMING_FEATURE_COUNT]),
        "timing_feature_count": TIMING_FEATURE_COUNT,
        "wyckoff_columns": list(WYCKOFF_FEATURE_COLUMNS[:WYCKOFF_FEATURE_COUNT]),
        "wyckoff_feature_count": WYCKOFF_FEATURE_COUNT,
        "volume_source_policy": "real_then_tick_then_neutral",
        "input_count": INPUT_COUNT,
        "feature_config": asdict(feature_config),
        "symbol": asdict(symbol),
        "session_timezone": str(session_timezone),
        "spread_semantics": str(spread_semantics),
        "mt5_server_utc_offset_hours": int(mt5_server_utc_offset_hours),
        "mtf_mode": "closed_higher_timeframe_bar_only",
        "warmup_policy": "drop_rows_without_full_causal_history",
    }
    # Preserve the historical raw-v10 fingerprint for legacy artifacts.  A
    # normalized candidate gets an explicit versioned contract hash.
    normalized_mode = str(input_normalization_mode or INPUT_NORMALIZATION_RAW).strip().lower()
    if normalized_mode != INPUT_NORMALIZATION_RAW:
        payload["input_normalization"] = input_normalization_payload(normalized_mode)
    return payload


def feature_schema_fingerprint(
    feature_config: FeatureConfig,
    symbol: SymbolConfig,
    *,
    session_timezone: str = "UTC",
    spread_semantics: str = "ask_minus_bid_price",
    mt5_server_utc_offset_hours: int = 0,
    input_normalization_mode: str = INPUT_NORMALIZATION_RAW,
) -> str:
    raw = json.dumps(
        feature_schema_payload(
            feature_config, symbol, session_timezone=session_timezone,
            spread_semantics=spread_semantics,
            mt5_server_utc_offset_hours=mt5_server_utc_offset_hours,
            input_normalization_mode=input_normalization_mode,
        ),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def canonical_market_features(raw: pd.DataFrame, feature_config: FeatureConfig, symbol: SymbolConfig) -> pd.DataFrame:
    """Return the Python numerical reference for market slots."""
    frame = add_features(raw, feature_config, symbol)
    return frame[
        [
            "Datetime",
            *MARKET_FEATURE_COLUMNS[:MARKET_FEATURE_COUNT],
            *TIMING_FEATURE_COLUMNS[:TIMING_FEATURE_COUNT],
            *WYCKOFF_FEATURE_COLUMNS[:WYCKOFF_FEATURE_COUNT],
            "atr",
            "spread_price",
        ]
    ].copy()


def build_reference_input(
    market_row: pd.Series | dict[str, float],
    *,
    position_slots: Iterable[float] | None = None,
    account_features: Iterable[float] | None = None,
    execution_features: Iterable[float] | None = None,
) -> np.ndarray:
    """Build the full 137 vector using an explicit zero/default account state.

    This powers golden fixtures. Live MT5 parity additionally compares every
    position/account/execution slot emitted by the EA, not just market values.
    """
    market = np.asarray([float(market_row[column]) for column in MARKET_FEATURE_COLUMNS[:MARKET_FEATURE_COUNT]], dtype=float)
    timing = np.asarray([float(market_row[column]) for column in TIMING_FEATURE_COLUMNS[:TIMING_FEATURE_COUNT]], dtype=float)
    wyckoff = np.asarray([float(market_row[column]) for column in WYCKOFF_FEATURE_COLUMNS[:WYCKOFF_FEATURE_COUNT]], dtype=float)
    slots = np.zeros(40, dtype=float) if position_slots is None else np.asarray(list(position_slots), dtype=float)
    account = np.zeros(8, dtype=float) if account_features is None else np.asarray(list(account_features), dtype=float)
    execution = np.zeros(11, dtype=float) if execution_features is None else np.asarray(list(execution_features), dtype=float)
    return finite_vector(
        np.concatenate([market[:29], slots, account, market[29:58], execution, timing, wyckoff]),
        size=INPUT_COUNT,
        name="canonical feature vector",
    )


def read_mt5_feature_audit(path: str | Path) -> pd.DataFrame:
    """Read a tab/comma-separated EA feature-audit export without guessing rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    sep = "\t" if "\t" in text.splitlines()[0] else ","
    frame = pd.read_csv(path, sep=sep)
    required = {"feature_schema_hash", "event_id", "inputs"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EA feature audit is missing fields: {sorted(missing)}")
    return frame


def compare_feature_vectors(
    expected: Iterable[float],
    actual: Iterable[float],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    atol_by_index: Mapping[int, float] | None = None,
) -> dict[str, object]:
    expected_a = finite_vector(expected, size=INPUT_COUNT, name="expected feature vector")
    actual_a = finite_vector(actual, size=INPUT_COUNT, name="actual feature vector")
    abs_error = np.abs(actual_a - expected_a)
    tolerances = np.full(expected_a.size, float(atol), dtype=float)
    for index, value in (atol_by_index or {}).items():
        if 0 <= int(index) < tolerances.size:
            tolerances[int(index)] = max(tolerances[int(index)], float(value))
    allowed_error = tolerances + float(rtol) * np.abs(expected_a)
    ok = bool(np.all(np.isfinite(actual_a)) and np.all(abs_error <= allowed_error))
    return {
        "ok": ok,
        "max_abs_error": float(abs_error.max(initial=0.0)),
        "mean_abs_error": float(abs_error.mean()) if abs_error.size else 0.0,
        "failing_indexes": np.flatnonzero(abs_error > allowed_error).astype(int).tolist(),
    }


def _parse_audit_inputs(value: object) -> np.ndarray:
    parts = [item for item in str(value or "").split("|") if item != ""]
    try:
        values = np.asarray([float(item) for item in parts], dtype=float)
    except ValueError as exc:
        raise ValueError("invalid numeric input payload in MT5 feature audit") from exc
    return finite_vector(values, size=INPUT_COUNT, name="MT5 feature audit inputs")


def _validate_broker_metadata(
    frame: pd.DataFrame,
    *,
    expected_point_size: float | None,
    require_broker_metadata: bool,
) -> dict[str, object]:
    """Validate broker primitives exported with each feature audit row.

    These fields do not prove order fills by themselves; they prevent a parity
    claim made with one point/tick geometry from being replayed against another
    broker contract. Old audit exports are reported explicitly rather than
    silently assumed compatible.
    """
    fields = ("symbol_point", "symbol_digits", "tick_size", "tick_value", "contract_size", "margin_mode")
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        return {
            "present": False,
            "valid": not require_broker_metadata,
            "missing": missing,
            "reason": "broker_metadata_missing" if require_broker_metadata else "",
        }
    finite = True
    positive_geometry = True
    point_matches = True
    for _, row in frame.iterrows():
        try:
            point = float(row["symbol_point"])
            digits = float(row["symbol_digits"])
            tick_size = float(row["tick_size"])
            tick_value = float(row["tick_value"])
            contract_size = float(row["contract_size"])
            margin_mode = float(row["margin_mode"])
        except (TypeError, ValueError):
            finite = False
            continue
        values = (point, digits, tick_size, tick_value, contract_size, margin_mode)
        finite = finite and bool(np.isfinite(values).all())
        positive_geometry = positive_geometry and point > 0.0 and tick_size > 0.0 and tick_value > 0.0 and contract_size > 0.0
        if expected_point_size is not None:
            point_matches = point_matches and bool(np.isclose(point, float(expected_point_size), rtol=1e-8, atol=1e-12))
    valid = bool(finite and positive_geometry and point_matches)
    return {
        "present": True,
        "valid": valid,
        "missing": [],
        "finite": finite,
        "positive_geometry": positive_geometry,
        "point_matches": point_matches,
        "reason": "" if valid else "broker_metadata_invalid",
    }


def summarize_mt5_feature_audit(
    path: str | Path | None,
    *,
    expected_schema_hash: str,
    expected_point_size: float | None = None,
    require_broker_metadata: bool = False,
) -> dict[str, object]:
    """Validate full-width EA export before a run can claim feature parity.

    This is deliberately a *transport/integrity* gate, not a dishonest claim
    that the numerical parity has already been proven. Numerical comparison is
    performed with :func:`compare_feature_vectors` against a same-bar fixture.
    """
    if path is None:
        return {"available": False, "valid": False, "reason": "feature_audit_missing"}
    target = Path(path)
    if not target.exists() or not target.is_file() or target.stat().st_size == 0:
        return {"available": False, "valid": False, "reason": "feature_audit_missing", "path": str(target)}
    try:
        frame = read_mt5_feature_audit(target)
        if frame.empty:
            return {"available": True, "valid": False, "reason": "feature_audit_empty", "path": str(target)}
        hashes = {str(item) for item in frame["feature_schema_hash"].dropna().astype(str)}
        hash_ok = hashes == {str(expected_schema_hash)}
        broker_metadata = _validate_broker_metadata(
            frame,
            expected_point_size=expected_point_size,
            require_broker_metadata=bool(require_broker_metadata),
        )
        widths: list[int] = []
        finite = True
        for payload in frame["inputs"]:
            values = _parse_audit_inputs(payload)
            widths.append(int(values.size))
            finite = finite and bool(np.isfinite(values).all())
        valid = bool(hash_ok and finite and all(width == INPUT_COUNT for width in widths) and bool(broker_metadata.get("valid", False)))
        return {
            "available": True,
            "valid": valid,
            "path": str(target),
            "rows": int(len(frame)),
            "hash_ok": hash_ok,
            "finite": finite,
            "widths": sorted(set(widths)),
            "broker_metadata": broker_metadata,
            "reason": "" if valid else str(broker_metadata.get("reason", "") or "feature_audit_contract_invalid"),
        }
    except Exception as exc:
        return {
            "available": True,
            "valid": False,
            "path": str(target),
            "reason": f"feature_audit_error:{type(exc).__name__}",
        }


def compare_feature_audit_row(
    expected: Iterable[float],
    audit_row: pd.Series | dict[str, object],
    *,
    expected_schema_hash: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    atol_by_index: Mapping[int, float] | None = None,
) -> dict[str, object]:
    """Compare one EA audit row against a same-bar Python reference vector."""
    row = dict(audit_row)
    if str(row.get("feature_schema_hash", "")) != str(expected_schema_hash):
        return {"ok": False, "reason": "schema_hash_mismatch"}
    result = compare_feature_vectors(
        expected,
        _parse_audit_inputs(row.get("inputs", "")),
        atol=atol,
        rtol=rtol,
        atol_by_index=atol_by_index,
    )
    result["reason"] = "" if bool(result["ok"]) else "numerical_feature_mismatch"
    return result


def _parse_positions_snapshot(value: object) -> list[dict[str, float]]:
    """Parse EA ``PositionsSnapshotPipe`` without silently inventing fields.

    ``pandas.read_csv`` represents an intentionally empty CSV field as NaN.
    Treat only that explicit empty-state as no positions; do not interpret an
    arbitrary malformed payload as flat account state.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    positions: list[dict[str, float]] = []
    for item in text.split("^"):
        if not item.strip():
            continue
        parts = item.split(":")
        if len(parts) < 8:
            raise ValueError("invalid positions_pipe in feature audit")
        try:
            positions.append({
                "side": float(parts[1]),
                "entry": float(parts[3]),
                "sl": float(parts[4]),
                "tp": float(parts[5]),
                "unrealized_r": float(parts[6]),
                "age_bars": float(parts[7]),
                # V155 append-only field. Older bridge rows intentionally
                # retain the zero fallback and are reconstructed with their
                # current SL below for backward-compatible audit support.
                "initial_risk_distance_price": float(parts[21]) if len(parts) > 21 else 0.0,
            })
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid numeric positions_pipe in feature audit") from exc
    return positions


def _audit_time(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(str(value or ""), errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("invalid audit timestamp")
    # MT5 timestamps carry no offset. The configured contract is therefore a
    # single naive wall-clock timeline; mixed timezones must be rejected before
    # this verifier is run rather than shifted implicitly here.
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_convert(None)
    return pd.Timestamp(timestamp).floor("min")


def _safe_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _reference_vector_for_audit_row(
    market_row: pd.Series,
    audit_row: dict[str, object],
    *,
    initial_equity: float,
    min_daily_trades: int,
    max_daily_trades: int,
    max_holding_bars: int,
    point_size: float,
    max_positions: int = MAX_POSITION_SLOTS,
) -> np.ndarray:
    """Rebuild all 137 input slots from raw bar + EA audit state.

    The EA export includes every non-market primitive needed for this
    reconstruction.  We intentionally compare the entire vector instead of
    accepting a market-only check as sufficient.
    """
    actual_positions = _parse_positions_snapshot(audit_row.get("positions_pipe", ""))
    atr = max(float(market_row["atr"]), 1e-12)
    close = float(market_row["Close"])
    spread_price = _safe_float(audit_row, "spread_points") * float(point_size)
    slots = np.zeros(40, dtype=float)
    execution = np.zeros(11, dtype=float)
    execution[0] = spread_price / atr
    for idx, position in enumerate(actual_positions[:5]):
        offset = idx * 8
        side = float(position["side"])
        entry = float(position["entry"])
        sl = float(position["sl"])
        tp = float(position["tp"])
        current_exit = close if side == 1.0 else close + spread_price
        # Match EA GetPositionInfo: when SL equals entry, it falls back to
        # the current ATR instead of dividing by an epsilon.  The epsilon-only
        # reconstruction falsely saturates unrealized-R at +/-5 and then
        # reports a parity failure for normalized inputs.
        risk_dist = float(position.get("initial_risk_distance_price", 0.0) or 0.0)
        if risk_dist < 1e-10:
            risk_dist = abs(entry - sl)
        if risk_dist < 1e-10:
            risk_dist = atr
        sl_distance = abs(entry - sl) / atr
        tp_distance = abs(tp - entry) / atr
        snapshot_age_bars = float(position["age_bars"])
        snapshot_time = _audit_time(audit_row.get("logged_at", audit_row.get("bar_time", "")))
        feature_time = _audit_time(audit_row.get("bar_time", audit_row.get("logged_at", "")))
        feature_age_bars = max(
            0.0,
            snapshot_age_bars + (feature_time - snapshot_time).total_seconds() / 60.0,
        )
        # The feature vector uses the closed-bar bid/ask proxy, whereas the
        # snapshot's unrealized value is sampled from the live tick.  Rebuild
        # this slot from the same bar close and spread as the EA.
        feature_unrealized_r = side * (current_exit - entry) / risk_dist
        slots[offset:offset + 8] = (
            1.0,
            side,
            np.clip(side * (close - entry) / atr, -20.0, 20.0),
            np.clip(feature_age_bars / max(1, int(max_holding_bars)), 0.0, 5.0),
            sl_distance,
            tp_distance,
            min(tp_distance / max(sl_distance, 1e-12), 5.0) / 5.0,
            np.clip(feature_unrealized_r, -5.0, 5.0),
        )
        execution_offset = 1 + idx * 2
        if side == 1.0:
            execution[execution_offset] = np.clip((current_exit - sl) / atr, -20.0, 20.0)
            execution[execution_offset + 1] = np.clip((tp - current_exit) / atr, -20.0, 20.0)
        else:
            execution[execution_offset] = np.clip((sl - current_exit) / atr, -20.0, 20.0)
            execution[execution_offset + 1] = np.clip((current_exit - tp) / atr, -20.0, 20.0)

    time_value = _audit_time(audit_row.get("logged_at", audit_row.get("bar_time", "")))
    day_progress = float(np.clip((time_value.hour * 60.0 + time_value.minute) / 1440.0, 0.0, 1.0))
    day_trades = max(0.0, _safe_float(audit_row, "day_trades"))
    balance = max(_safe_float(audit_row, "balance", initial_equity), 1e-12)
    min_daily = max(0, int(min_daily_trades))
    max_daily = max(0, int(max_daily_trades))
    expected_trades = min_daily * day_progress if min_daily > 0 else 0.0
    cadence_deficit = max(0.0, expected_trades - day_trades) / max(1.0, float(min_daily)) if min_daily > 0 else 0.0
    account = np.asarray([
        len(actual_positions) / float(position_count_reference(max_positions)),
        _safe_float(audit_row, "open_risk_ratio"),
        np.clip(_safe_float(audit_row, "drawdown_pct") / 100.0, 0.0, 1.0),
        (_safe_float(audit_row, "equity") / balance) - 1.0,
        min(day_trades, max_daily) / max(1.0, float(max_daily)) if max_daily > 0 else 0.0,
        min(day_trades, min_daily) / max(1.0, float(min_daily)) if min_daily > 0 else 0.0,
        day_progress,
        np.clip(cadence_deficit, 0.0, 1.0),
    ], dtype=float)
    market = np.asarray([float(market_row[column]) for column in MARKET_FEATURE_COLUMNS[:MARKET_FEATURE_COUNT]], dtype=float)
    timing = np.asarray([float(market_row[column]) for column in TIMING_FEATURE_COLUMNS[:TIMING_FEATURE_COUNT]], dtype=float)
    wyckoff = np.asarray([float(market_row[column]) for column in WYCKOFF_FEATURE_COLUMNS[:WYCKOFF_FEATURE_COUNT]], dtype=float)
    return finite_vector(
        np.concatenate([market[:29], slots, account, market[29:58], execution, timing, wyckoff]),
        size=INPUT_COUNT,
        name="reconstructed MT5 feature reference",
    )


def compare_mt5_feature_audit_against_reference(
    audit_path: str | Path,
    reference_csv: str | Path,
    feature_config: FeatureConfig,
    symbol: SymbolConfig,
    *,
    expected_schema_hash: str,
    datetime_column: str = "Datetime",
    initial_equity: float = 10000.0,
    min_daily_trades: int = 0,
    max_daily_trades: int = 0,
    max_holding_bars: int = 60,
    max_positions: int = MAX_POSITION_SLOTS,
    require_broker_metadata: bool = True,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    dynamic_volume_parity_atol: float = DEFAULT_DYNAMIC_VOLUME_PARITY_ATOL,
    max_rows: int = 256,
    input_normalization_mode: str = INPUT_NORMALIZATION_RAW,
    audit_inputs_normalized: bool = True,
    strict_model1: bool = False,
) -> dict[str, object]:
    """Numerically compare exported EA feature rows against Python reference.

    Rows are sampled deterministically across the run, not just from the warm
    beginning. Any missing timestamp, non-finite vector or vector mismatch
    invalidates the proof. This is intentionally strict: a policy cannot be
    promoted on a simulator/live feature contract that has not been measured.
    """
    contract = summarize_mt5_feature_audit(
        audit_path,
        expected_schema_hash=expected_schema_hash,
        expected_point_size=float(symbol.point_size),
        require_broker_metadata=bool(require_broker_metadata),
    )
    if not bool(contract.get("valid", False)):
        return {**contract, "numerical_checked": False, "numerical_valid": False}
    reference = Path(reference_csv)
    if not reference.exists():
        return {
            **contract,
            "numerical_checked": False,
            "numerical_valid": False,
            "valid": False,
            "reason": "feature_reference_csv_missing",
            "reference_csv": str(reference),
        }
    try:
        dynamic_volume_atol = max(0.0, float(dynamic_volume_parity_atol))
        dynamic_volume_atol_by_index = {
            int(index): dynamic_volume_atol
            for index in DYNAMIC_VOLUME_PARITY_FEATURE_INDEXES
        }
        audit = read_mt5_feature_audit(audit_path)
        if len(audit) > max_rows:
            indexes = np.linspace(0, len(audit) - 1, int(max_rows), dtype=int)
            audit = audit.iloc[np.unique(indexes)]
        audit_times = [
            _audit_time(dict(row).get("bar_time", dict(row).get("logged_at", "")))
            for _, row in audit.iterrows()
        ]
        if not audit_times:
            return {
                **contract,
                "valid": False,
                "numerical_checked": False,
                "numerical_valid": False,
                "checked_rows": 0,
                "mismatch_count": 0,
                "reference_csv": str(reference),
                "reason": "feature_audit_empty",
            }
        # A slow EMA on H1 needs much more history than the one-day audit.
        # Ten EMA time constants make truncation error negligible while keeping
        # a parity check bounded for multi-year MT5 CSV exports.
        warmup_minutes = max(
            10_000,
            int(feature_config.ema_slow) * 60 * 10,
            int(feature_config.atr_period) * 60 * 4,
            int(feature_config.atr_percentile_window) * 4,
            int(feature_config.wyckoff_range_window) * 4,
        )
        window_start = min(audit_times) - pd.Timedelta(minutes=warmup_minutes)
        window_end = max(audit_times)
        chunks: list[pd.DataFrame] = []
        # MT5's native export is tab-delimited and names fields as
        # ``<DATE>``, ``<TIME>``, ``<OPEN>`` ...; the project reference CSVs
        # are also allowed to be ordinary comma-delimited ``Datetime/Open``
        # files.  Detect and canonicalise both formats before the strict
        # parity comparison instead of silently reporting a missing range.
        header = reference.open("r", encoding="utf-8-sig", errors="replace").readline()
        reference_sep = "\t" if "\t" in header else ","
        for chunk in pd.read_csv(reference, sep=reference_sep, chunksize=200_000):
            raw_columns = {str(column).strip().upper(): column for column in chunk.columns}
            rename_columns = {
                raw_columns[key]: canonical
                for key, canonical in {
                    "<OPEN>": "Open",
                    "<HIGH>": "High",
                    "<LOW>": "Low",
                    "<CLOSE>": "Close",
                    "<SPREAD>": "Spread",
                    "<TICKVOL>": "TickVol",
                    "<VOL>": "Vol",
                }.items()
                if key in raw_columns
            }
            if rename_columns:
                chunk = chunk.rename(columns=rename_columns)
                raw_columns = {str(column).strip().upper(): column for column in chunk.columns}
            if datetime_column not in chunk.columns:
                date_column = raw_columns.get("<DATE>") or raw_columns.get("DATE")
                time_column = raw_columns.get("<TIME>") or raw_columns.get("TIME")
                if date_column is not None and time_column is not None:
                    chunk["Datetime"] = (
                        chunk[date_column].astype(str).str.strip()
                        + " "
                        + chunk[time_column].astype(str).str.strip()
                    )
                else:
                    raise ValueError(f"reference CSV missing datetime column {datetime_column!r}")
            if datetime_column not in chunk.columns and datetime_column != "Datetime":
                raise ValueError(f"reference CSV missing datetime column {datetime_column!r}")
            # Native MT5 exports use ``YYYY.MM.DD``; legacy project CSVs may
            # use ``DD/MM/YYYY``.  Applying dayfirst=True to the former
            # silently swaps month/day and corrupts higher-timeframe history.
            parsed_time = pd.to_datetime(
                chunk[datetime_column],
                dayfirst=(reference_sep != "\t"),
                errors="coerce",
            )
            selected = chunk.loc[(parsed_time >= window_start) & (parsed_time <= window_end)].copy()
            if selected.empty:
                continue
            selected[datetime_column] = parsed_time.loc[selected.index]
            if datetime_column != "Datetime":
                selected = selected.rename(columns={datetime_column: "Datetime"})
            chunks.append(selected)
        if not chunks:
            return {
                **contract,
                "valid": False,
                "numerical_checked": False,
                "numerical_valid": False,
                "checked_rows": 0,
                "mismatch_count": 0,
                "reference_csv": str(reference),
                "audit_bar_start": str(min(audit_times)),
                "audit_bar_end": str(max(audit_times)),
                "reason": "feature_reference_bar_range_missing",
            }
        raw = pd.concat(chunks, ignore_index=True)
        required = {"Datetime", "Open", "High", "Low", "Close", "Spread"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"reference CSV missing fields: {missing}")
        features = add_features(raw, feature_config, symbol)
        features["_audit_time"] = pd.to_datetime(features["Datetime"], errors="coerce").dt.floor("min")
        feature_by_time = features.drop_duplicates("_audit_time", keep="last").set_index("_audit_time")
        mismatches: list[dict[str, object]] = []
        checked = 0
        spread_reference_override_rows = 0
        for _, raw_row in audit.iterrows():
            row = dict(raw_row)
            bar_time = _audit_time(row.get("bar_time", row.get("logged_at", "")))
            if bar_time not in feature_by_time.index:
                mismatches.append({"event_id": str(row.get("event_id", "")), "reason": "reference_bar_missing", "bar_time": str(bar_time)})
                continue
            expected = _reference_vector_for_audit_row(
                feature_by_time.loc[bar_time], row,
                initial_equity=initial_equity,
                min_daily_trades=min_daily_trades,
                max_daily_trades=max_daily_trades,
                max_holding_bars=max_holding_bars,
                point_size=symbol.point_size,
                max_positions=max_positions,
            )
            # The legacy EA exported snapshot age with one decimal place and
            # sampled it at ``TimeCurrent``, while ComputeAllInputs uses the
            # exact closed-bar timestamp.  Since the audit stores both times
            # at minute precision, allow the bounded one-bar quantization
            # envelope only for position-age slots.  Larger differences still
            # fail closed as a real position-state mismatch.
            expected_for_comparison = expected.copy()
            actual_inputs = _parse_audit_inputs(row.get("inputs", ""))
            # A static MT5 OHLC export contains a bar-level spread snapshot,
            # while this real-tick audit captured the live spread at the exact
            # decision tick.  The two spread-derived market fields are thus
            # not numerically reconstructible from OHLC alone.  Override only
            # those slots, and only after checking the duplicated execution
            # spread ratio and percentile domain; all other fields stay strict.
            spread_ratio = float(actual_inputs[SPREAD_ATR_FEATURE_INDEX])
            execution_ratio = float(actual_inputs[SPREAD_EXECUTION_FEATURE_INDEX])
            percentile = float(actual_inputs[SPREAD_PERCENTILE_FEATURE_INDEX])
            if (
                not strict_model1
                and
                np.isfinite(spread_ratio)
                and np.isfinite(execution_ratio)
                and np.isfinite(percentile)
                and 0.0 <= percentile <= 1.0
                and np.isclose(spread_ratio, execution_ratio, atol=5e-4, rtol=1e-4)
            ):
                expected_for_comparison[SPREAD_ATR_FEATURE_INDEX] = spread_ratio
                expected_for_comparison[SPREAD_PERCENTILE_FEATURE_INDEX] = percentile
                spread_reference_override_rows += 1
            if audit_inputs_normalized:
                expected_for_comparison = normalize_input_vector(
                    expected_for_comparison,
                    mode=input_normalization_mode,
                    size=INPUT_COUNT,
                )
            age_tolerance = POSITION_AGE_AUDIT_QUANTIZATION_BARS / max(
                1.0, float(max_holding_bars)
            )
            # Compare age in the same representation as the audit.  Bridge
            # v9 emits raw features; earlier bridge audits may opt in to the
            # historical normalized representation for forensic comparison.
            for slot in range(MAX_POSITION_SLOTS):
                if strict_model1:
                    break
                age_index = POSITION_OFFSET + slot * POSITION_FEATURE_COUNT + 3
                if age_index >= actual_inputs.size:
                    break
                if abs(float(expected_for_comparison[age_index]) - float(actual_inputs[age_index])) <= age_tolerance:
                    expected_for_comparison[age_index] = actual_inputs[age_index]
            # Unrealized-R is sampled from the live tick in PositionsSnapshotPipe,
            # while the static OHLC reference can only rebuild it from bar close.
            # Accept the authoritative snapshot value only when it agrees with
            # the corresponding normalized audit slot; this keeps the override
            # narrow and prevents arbitrary position-state drift from passing.
            snapshot_positions = _parse_positions_snapshot(row.get("positions_pipe", ""))
            for slot, position in enumerate(snapshot_positions[:MAX_POSITION_SLOTS]):
                if strict_model1:
                    break
                unrealized_index = POSITION_OFFSET + slot * POSITION_FEATURE_COUNT + 7
                if unrealized_index >= actual_inputs.size:
                    break
                if audit_inputs_normalized:
                    # Groupwise modes depend on the global feature index: a
                    # position's unrealized-R belongs to the signed-log
                    # position slice, while a length-one vector would be
                    # treated as market feature zero.  Build a contract-width
                    # probe to apply the same historical audit transform.
                    live_unrealized_probe = np.zeros(INPUT_COUNT, dtype=float)
                    live_unrealized_probe[unrealized_index] = float(position["unrealized_r"])
                    live_unrealized = normalize_input_vector(
                        live_unrealized_probe,
                        mode=input_normalization_mode,
                        size=INPUT_COUNT,
                    )[unrealized_index]
                else:
                    live_unrealized = float(position["unrealized_r"])
                if np.isfinite(live_unrealized) and np.isclose(
                    float(actual_inputs[unrealized_index]),
                    float(live_unrealized),
                    atol=5e-4,
                    rtol=1e-4,
                ):
                    expected_for_comparison[unrealized_index] = actual_inputs[unrealized_index]
            # The two runtime spread fields are sampled from the live tick and
            # therefore remain authoritative after the deterministic transform.
            # Bridge v9+ exports raw features; v8 and older audits can opt into
            # their historical normalized representation. In either view we
            # must never transform ``actual_inputs`` a second time.
            if (
                not strict_model1
                and
                np.isfinite(spread_ratio)
                and np.isfinite(execution_ratio)
                and np.isfinite(percentile)
                and 0.0 <= percentile <= 1.0
                and np.isclose(spread_ratio, execution_ratio, atol=5e-4, rtol=1e-4)
            ):
                expected_for_comparison[SPREAD_ATR_FEATURE_INDEX] = spread_ratio
                expected_for_comparison[SPREAD_PERCENTILE_FEATURE_INDEX] = percentile
            if strict_model1:
                # The EA serializes every bridge feature with exactly ten
                # decimal places. Compare the same wire representation; no
                # numerical tolerance or audit-derived field substitution is
                # allowed in strict model-1 parity.
                expected_for_comparison = np.round(expected_for_comparison, 10)
            comparison = compare_feature_audit_row(
                expected_for_comparison,
                row,
                expected_schema_hash=expected_schema_hash,
                atol=0.0 if strict_model1 else atol,
                rtol=0.0 if strict_model1 else rtol,
                atol_by_index={} if strict_model1 else dynamic_volume_atol_by_index,
            )
            checked += 1
            if not bool(comparison.get("ok", False)):
                mismatches.append({
                    "event_id": str(row.get("event_id", "")),
                    "bar_time": str(bar_time),
                    **comparison,
                })
        numerical_valid = bool(checked > 0 and not mismatches)
        return {
            **contract,
            "valid": bool(contract.get("valid", False) and numerical_valid),
            "numerical_checked": True,
            "numerical_valid": numerical_valid,
            "checked_rows": int(checked),
            "mismatch_count": int(len(mismatches)),
            "mismatch_preview": mismatches[:10],
            "reference_csv": str(reference),
            "audit_input_view": (
                "policy_normalized_legacy"
                if audit_inputs_normalized
                else "raw_bridge_v9"
            ),
            "spread_reference_override_rows": int(spread_reference_override_rows),
            "spread_reference_override_reason": (
                "runtime_spread_fields_unreconstructible_from_static_ohlc"
                if spread_reference_override_rows
                else ""
            ),
            "dynamic_volume_parity_feature_indexes": list(DYNAMIC_VOLUME_PARITY_FEATURE_INDEXES),
            "dynamic_volume_parity_atol": float(dynamic_volume_atol),
            "position_count_reference": int(position_count_reference(max_positions)),
            "strict_model1": bool(strict_model1),
            "reason": "" if numerical_valid else "numerical_feature_parity_failed",
        }
    except Exception as exc:
        return {
            **contract,
            "valid": False,
            "numerical_checked": True,
            "numerical_valid": False,
            "checked_rows": 0,
            "mismatch_count": 0,
            "mismatch_preview": [],
            "reason": f"feature_parity_error:{type(exc).__name__}",
            "reference_csv": str(reference),
            "audit_input_view": (
                "policy_normalized_legacy"
                if audit_inputs_normalized
                else "raw_bridge_v9"
            ),
        }
