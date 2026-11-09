from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator
import csv
import io

import pandas as pd
import numpy as np

from .protocol_contract import position_count_reference


MODEL1_TICK_SCHEMA_VERSION = 1
MODEL1_TESTER_MODEL = 1
MODEL1_TICK_COLUMNS = (
    "time_msc",
    "bid",
    "ask",
    "last",
    "volume",
    "flags",
)
PARITY_SCHEMA_VERSION = 1
BASE_MARKET_SLICE = slice(0, 29)
POSITION_SLICE = slice(29, 69)
ACCOUNT_SLICE = slice(69, 77)
EXTRA_MARKET_SLICE = slice(77, 106)
EXECUTION_SPREAD_INDEX = 106
EXECUTION_POSITION_SLICE = slice(107, 117)
TAIL_MARKET_SLICE = slice(117, 137)
MODEL1_ORACLE_TAPE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Model1Tick:
    time_msc: int
    bid: float
    ask: float
    last: float
    volume: float
    flags: int


@dataclass(frozen=True)
class Model1ReplayManifest:
    schema_version: int
    tester_model: int
    symbol: str
    server: str
    from_date: str
    to_date: str
    tick_count: int
    tick_sha256: str
    point: float
    digits: int
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float

    @classmethod
    def from_capture_metadata(
        cls, *, tick_path: str | Path, metadata: dict[str, object]
    ) -> "Model1ReplayManifest":
        frame = pd.read_csv(Path(tick_path), sep="\t", usecols=["time_msc"])
        return cls(
            schema_version=MODEL1_TICK_SCHEMA_VERSION,
            tester_model=int(metadata["tester_model"]),
            symbol=str(metadata["symbol"]), server=str(metadata.get("server", "")),
            from_date=str(metadata.get("from_date", "")), to_date=str(metadata.get("to_date", "")),
            tick_count=int(len(frame)), tick_sha256=sha256_file(tick_path),
            point=float(metadata["point"]), digits=int(metadata["digits"]),
            tick_size=float(metadata["tick_size"]), tick_value=float(metadata["tick_value"]),
            volume_min=float(metadata["volume_min"]), volume_max=float(metadata["volume_max"]),
            volume_step=float(metadata["volume_step"]), contract_size=float(metadata["contract_size"]),
        )

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Model1ReplayManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if int(self.schema_version) != MODEL1_TICK_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model-1 replay schema {self.schema_version}; "
                f"expected {MODEL1_TICK_SCHEMA_VERSION}"
            )
        if int(self.tester_model) != MODEL1_TESTER_MODEL:
            raise ValueError("python_replay accepts only an MT5 tester model-1 capture")
        if not self.symbol or int(self.tick_count) < 1:
            raise ValueError("model-1 replay manifest has no symbol/ticks")
        if float(self.point) <= 0.0 or int(self.digits) < 0 or float(self.tick_size) <= 0.0:
            raise ValueError("model-1 replay manifest has invalid symbol geometry")
        if not self.tick_sha256:
            raise ValueError("model-1 replay manifest must pin the tick file SHA256")


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


_MODEL1_VOLATILE_STATE_FIELDS = frozenset({
    "graph_trial_id",
    "trial_id",
    "graph_instance_uuid",
    "memory_object_identity",
    "policy_checkpoint_hash",
    "memory_hash",
    "created_at",
    "last_seen_at",
})


def model1_training_state_hash(policy: object) -> str:
    """Hash complete logical learner state while excluding process identity only.

    Numeric tensors, optimizer/RNG state, counters, memories, deterministic
    trace keys and topology all participate. UUIDs, object addresses and wall
    timestamps used only for provenance do not; they cannot affect inference
    or an optimizer update and necessarily differ between two processes.
    """
    digest = sha256()
    seen: dict[int, str] = {}

    def emit(value: object, path: str, field_name: str = "") -> None:
        if field_name in _MODEL1_VOLATILE_STATE_FIELDS or field_name.endswith("_uuid"):
            return
        if value is None or isinstance(value, (bool, int, str, bytes)):
            digest.update(f"{path}|{type(value).__name__}|".encode("utf-8"))
            digest.update(value if isinstance(value, bytes) else repr(value).encode("utf-8"))
            digest.update(b"\n")
            return
        if isinstance(value, (float, np.floating)):
            digest.update(f"{path}|float|{float(value).hex()}\n".encode("ascii"))
            return
        if isinstance(value, np.integer):
            digest.update(f"{path}|int|{int(value)}\n".encode("ascii"))
            return
        if isinstance(value, Enum):
            emit(value.value, path, field_name)
            return
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(
                f"{path}|ndarray|{array.dtype.str}|{array.shape}\n".encode("ascii")
            )
            digest.update(array.tobytes(order="C"))
            digest.update(b"\n")
            return
        if isinstance(value, np.random.Generator):
            emit(value.bit_generator.state, path + ".bit_generator_state")
            return
        if callable(value):
            # Factories/hooks are executable code, not checkpoint state. Their
            # source hash belongs in the run manifest.
            return
        identity = id(value)
        if identity in seen:
            digest.update(f"{path}|ref|{seen[identity]}\n".encode("utf-8"))
            return
        seen[identity] = path
        if isinstance(value, dict):
            keys = [
                key for key in value
                if str(key) not in _MODEL1_VOLATILE_STATE_FIELDS
                and not str(key).endswith("_uuid")
            ]
            digest.update(f"{path}|dict|{len(keys)}\n".encode("utf-8"))
            for key in sorted(keys, key=lambda item: (type(item).__name__, repr(item))):
                key_name = str(key)
                emit(key, path + ".<key>")
                emit(value[key], path + f"[{repr(key)}]", key_name)
            return
        if isinstance(value, (list, tuple)):
            digest.update(f"{path}|{type(value).__name__}|{len(value)}\n".encode("utf-8"))
            for index, item in enumerate(value):
                emit(item, f"{path}[{index}]")
            return
        if isinstance(value, (set, frozenset)):
            digest.update(f"{path}|set|{len(value)}\n".encode("utf-8"))
            for index, item in enumerate(sorted(value, key=lambda item: (type(item).__name__, repr(item)))):
                emit(item, f"{path}[{index}]")
            return
        if is_dataclass(value):
            names = sorted(item.name for item in fields(value))
            digest.update(f"{path}|{type(value).__qualname__}\n".encode("utf-8"))
            for name in names:
                emit(getattr(value, name), f"{path}.{name}", name)
            return
        if hasattr(value, "__dict__"):
            digest.update(f"{path}|{type(value).__qualname__}\n".encode("utf-8"))
            for name in sorted(vars(value)):
                emit(vars(value)[name], f"{path}.{name}", name)
            return
        slots = tuple(str(name) for name in getattr(type(value), "__slots__", ()) if isinstance(name, str))
        if slots:
            digest.update(f"{path}|{type(value).__qualname__}\n".encode("utf-8"))
            for name in sorted(slots):
                if hasattr(value, name):
                    emit(getattr(value, name), f"{path}.{name}", name)
            return
        digest.update(f"{path}|opaque|{type(value).__module__}.{type(value).__qualname__}\n".encode("utf-8"))

    emit(policy, "policy")
    return digest.hexdigest().upper()


class Model1TickDataset:
    """Immutable ticks captured from MT5 Strategy Tester model 1.

    Python never attempts to recreate MetaTrader's private tick generator.
    The capture is the oracle market path; its manifest pins both tester model
    and content hash so a replay cannot silently use OHLC or real-tick data.
    """

    def __init__(self, tick_path: str | Path, manifest_path: str | Path | None = None):
        self.tick_path = Path(tick_path)
        self.manifest_path = Path(manifest_path) if manifest_path else self.tick_path.with_suffix(
            self.tick_path.suffix + ".manifest.json"
        )
        if not self.tick_path.is_file():
            raise FileNotFoundError(f"model-1 tick capture not found: {self.tick_path}")
        self.manifest = Model1ReplayManifest.load(self.manifest_path)
        actual_hash = sha256_file(self.tick_path)
        if actual_hash != self.manifest.tick_sha256.upper():
            raise ValueError(
                "model-1 tick capture hash mismatch: "
                f"manifest={self.manifest.tick_sha256} actual={actual_hash}"
            )

    def frame(self) -> pd.DataFrame:
        frame = pd.read_csv(self.tick_path, sep="\t")
        missing = set(MODEL1_TICK_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"model-1 tick capture is missing columns: {sorted(missing)}")
        if len(frame) != int(self.manifest.tick_count):
            raise ValueError(
                f"model-1 tick count mismatch: manifest={self.manifest.tick_count} file={len(frame)}"
            )
        if frame.empty or not frame["time_msc"].is_monotonic_increasing:
            raise ValueError("model-1 ticks must be non-empty and ordered by time_msc")
        if (frame["ask"] < frame["bid"]).any():
            raise ValueError("model-1 tick capture contains ask below bid")
        return frame.loc[:, MODEL1_TICK_COLUMNS]

    def __iter__(self) -> Iterator[Model1Tick]:
        for row in self.frame().itertuples(index=False, name=None):
            yield Model1Tick(
                time_msc=int(row[0]), bid=float(row[1]), ask=float(row[2]),
                last=float(row[3]), volume=float(row[4]), flags=int(row[5]),
            )

    def minute_bars(self) -> pd.DataFrame:
        """Build the exact bid OHLC path seen by model 1, without interpolation."""
        ticks = self.frame().copy()
        ticks["Datetime"] = pd.to_datetime(ticks["time_msc"], unit="ms", utc=True).dt.tz_localize(None)
        ticks["minute"] = ticks["Datetime"].dt.floor("min")
        grouped = ticks.groupby("minute", sort=True, observed=True)
        bars = grouped.agg(
            Open=("bid", "first"),
            High=("bid", "max"),
            Low=("bid", "min"),
            Close=("bid", "last"),
            AskOpen=("ask", "first"),
            AskClose=("ask", "last"),
            TickVol=("time_msc", "size"),
            Vol=("volume", "sum"),
        ).reset_index(names="Datetime")
        bars["Spread"] = (
            (bars["AskClose"] - bars["Close"]) / float(self.manifest.point)
        ).round().astype(int)
        bars["decision_bid"] = bars["Open"]
        bars["decision_ask"] = bars["AskOpen"]
        bars["decision_spread_points"] = (
            (bars["AskOpen"] - bars["Open"]) / float(self.manifest.point)
        )
        return bars


@dataclass(frozen=True)
class ParityTolerance:
    input_abs: float = 1e-8
    output_abs: float = 1e-10
    price_ticks: float = 0.0
    money_abs: float = 0.01
    realized_r_abs: float = 1e-8


@dataclass(frozen=True)
class ParityMismatch:
    sequence: int
    field: str
    expected: object
    actual: object
    absolute_error: float | None = None


@dataclass(frozen=True)
class ParityReport:
    schema_version: int
    checked_sequences: int
    mismatches: tuple[ParityMismatch, ...]

    @property
    def exact(self) -> bool:
        return not self.mismatches

    def require_exact(self) -> None:
        if self.mismatches:
            first = self.mismatches[0]
            raise ValueError(
                "MT5 model-1 parity failed: "
                f"sequence={first.sequence} field={first.field} "
                f"expected={first.expected!r} actual={first.actual!r}"
            )


def compare_numeric_vectors(
    *, sequence: int, field: str, expected: object, actual: object, tolerance: float,
) -> list[ParityMismatch]:
    left = [float(item) for item in expected]
    right = [float(item) for item in actual]
    if len(left) != len(right):
        return [ParityMismatch(sequence, f"{field}.length", len(left), len(right))]
    mismatches: list[ParityMismatch] = []
    for index, (wanted, got) in enumerate(zip(left, right, strict=True)):
        error = abs(wanted - got)
        if error > float(tolerance):
            mismatches.append(
                ParityMismatch(sequence, f"{field}[{index}]", wanted, got, error)
            )
    return mismatches


def require_decision_parity(
    *,
    sequence: int,
    mt5_raw_inputs: object,
    python_raw_inputs: object,
    mt5_normalized_inputs: object,
    python_normalized_inputs: object,
    mt5_outputs: object,
    python_outputs: object,
    mt5_action: int,
    python_action: int,
    tolerance: ParityTolerance = ParityTolerance(),
) -> ParityReport:
    """Fail-closed comparison at the complete policy/execution boundary."""
    mismatches: list[ParityMismatch] = []
    mismatches.extend(compare_numeric_vectors(
        sequence=sequence, field="raw_inputs", expected=mt5_raw_inputs,
        actual=python_raw_inputs, tolerance=tolerance.input_abs,
    ))
    mismatches.extend(compare_numeric_vectors(
        sequence=sequence, field="normalized_inputs", expected=mt5_normalized_inputs,
        actual=python_normalized_inputs, tolerance=tolerance.input_abs,
    ))
    mismatches.extend(compare_numeric_vectors(
        sequence=sequence, field="outputs", expected=mt5_outputs,
        actual=python_outputs, tolerance=tolerance.output_abs,
    ))
    if int(mt5_action) != int(python_action):
        mismatches.append(ParityMismatch(sequence, "action", int(mt5_action), int(python_action)))
    report = ParityReport(PARITY_SCHEMA_VERSION, 1, tuple(mismatches))
    report.require_exact()
    return report


@dataclass(frozen=True)
class CanonicalOrderRequest:
    sequence: int
    action: int
    order_type: str
    side: int
    volume: float
    price: float
    sl: float
    tp: float
    deviation_points: int
    expiration_msc: int
    target_position: str = ""


@dataclass(frozen=True)
class CanonicalExecutionState:
    sequence: int
    accepted: bool
    retcode: int
    order_ticket: str
    deal_ticket: str
    position_ticket: str
    positions_pipe: str
    orders_pipe: str
    balance: float
    equity: float
    commission: float
    swap: float
    profit: float
    realized_r: float


def compare_order_request(
    mt5: CanonicalOrderRequest,
    python: CanonicalOrderRequest,
    *,
    tick_size: float,
    tolerance: ParityTolerance = ParityTolerance(),
) -> tuple[ParityMismatch, ...]:
    if mt5.sequence != python.sequence:
        return (ParityMismatch(python.sequence, "sequence", mt5.sequence, python.sequence),)
    mismatches: list[ParityMismatch] = []
    for name in ("action", "order_type", "side", "deviation_points", "expiration_msc", "target_position"):
        wanted, got = getattr(mt5, name), getattr(python, name)
        if wanted != got:
            mismatches.append(ParityMismatch(mt5.sequence, name, wanted, got))
    for name, absolute in (("volume", 1e-12),):
        wanted, got = float(getattr(mt5, name)), float(getattr(python, name))
        error = abs(wanted - got)
        if error > absolute:
            mismatches.append(ParityMismatch(mt5.sequence, name, wanted, got, error))
    price_abs = max(0.0, float(tolerance.price_ticks)) * float(tick_size)
    for name in ("price", "sl", "tp"):
        wanted, got = float(getattr(mt5, name)), float(getattr(python, name))
        error = abs(wanted - got)
        if error > price_abs:
            mismatches.append(ParityMismatch(mt5.sequence, name, wanted, got, error))
    return tuple(mismatches)


def compare_execution_state(
    mt5: CanonicalExecutionState,
    python: CanonicalExecutionState,
    *,
    tolerance: ParityTolerance = ParityTolerance(),
) -> tuple[ParityMismatch, ...]:
    mismatches: list[ParityMismatch] = []
    for name in (
        "sequence", "accepted", "retcode", "order_ticket", "deal_ticket",
        "position_ticket", "positions_pipe", "orders_pipe",
    ):
        wanted, got = getattr(mt5, name), getattr(python, name)
        if wanted != got:
            mismatches.append(ParityMismatch(mt5.sequence, name, wanted, got))
    for name in ("balance", "equity", "commission", "swap", "profit"):
        wanted, got = float(getattr(mt5, name)), float(getattr(python, name))
        error = abs(wanted - got)
        if error > float(tolerance.money_abs):
            mismatches.append(ParityMismatch(mt5.sequence, name, wanted, got, error))
    error = abs(float(mt5.realized_r) - float(python.realized_r))
    if error > float(tolerance.realized_r_abs):
        mismatches.append(ParityMismatch(
            mt5.sequence, "realized_r", mt5.realized_r, python.realized_r, error
        ))
    return tuple(mismatches)


def _pipe_vector(value: object) -> np.ndarray:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return np.empty(0, dtype=float)
    return np.asarray([float(item) for item in text.split("|")], dtype=float)


def canonical_market_vector(raw_inputs: object) -> np.ndarray:
    """Keep only fields that are independent of simulated portfolio state."""
    raw = np.asarray(raw_inputs, dtype=float).reshape(-1)
    if raw.size != 137:
        raise ValueError(f"canonical market vector requires 137 inputs, got {raw.size}")
    market = np.zeros(137, dtype=float)
    market[BASE_MARKET_SLICE] = raw[BASE_MARKET_SLICE]
    market[EXTRA_MARKET_SLICE] = raw[EXTRA_MARKET_SLICE]
    market[EXECUTION_SPREAD_INDEX] = raw[EXECUTION_SPREAD_INDEX]
    market[TAIL_MARKET_SLICE] = raw[TAIL_MARKET_SLICE]
    return market


def compose_canonical_inputs(
    market_vector: object,
    *,
    position_features: object,
    account_features: object,
    execution_position_features: object,
) -> np.ndarray:
    """Compose the one authoritative 137-value policy input in wire order."""
    result = canonical_market_vector(market_vector)
    positions = np.asarray(position_features, dtype=float).reshape(-1)
    account = np.asarray(account_features, dtype=float).reshape(-1)
    execution = np.asarray(execution_position_features, dtype=float).reshape(-1)
    if positions.size != 40 or account.size != 8 or execution.size != 10:
        raise ValueError(
            "canonical dynamic widths must be position=40 account=8 execution_position=10"
        )
    result[POSITION_SLICE] = positions
    result[ACCOUNT_SLICE] = account
    result[EXECUTION_POSITION_SLICE] = execution
    if not np.isfinite(result).all():
        raise ValueError("canonical inputs contain non-finite values")
    # EA VectorToPipe is the policy boundary and writes ten decimals. Python
    # replay uses the identical wire values, never higher-precision shadows.
    return np.round(result, 10)


@dataclass(frozen=True)
class Model1OracleDecision:
    sequence: int
    event_id: str
    raw_inputs: np.ndarray
    outputs: np.ndarray
    action: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level_pct: float
    leverage: int
    drawdown_pct: float
    spread_points: float
    positions_pipe: str
    orders_pipe: str
    execution_pipe: str
    response_status: str
    logged_at: str
    bar_time: str
    bar_close: float
    atr: float
    open_risk_ratio: float
    point: float
    day_trades: int


@dataclass(frozen=True)
class ReplayPositionSnapshot:
    ticket: str
    side: int
    volume: float
    entry: float
    sl: float
    tp: float
    initial_risk_distance: float
    open_time: int


def parse_replay_positions(value: object) -> tuple[ReplayPositionSnapshot, ...]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ()
    positions: list[ReplayPositionSnapshot] = []
    for encoded in text.split("^"):
        parts = encoded.split(":")
        if len(parts) < 23:
            raise ValueError("model-1 parity position snapshot requires append-only open_time field")
        positions.append(ReplayPositionSnapshot(
            ticket=str(parts[0]), side=int(parts[1]), volume=float(parts[2]),
            entry=float(parts[3]), sl=float(parts[4]), tp=float(parts[5]),
            initial_risk_distance=float(parts[21]), open_time=int(parts[22]),
        ))
    return tuple(positions)


def build_dynamic_input_blocks(
    *,
    positions: tuple[ReplayPositionSnapshot, ...],
    bar_time: object,
    decision_time: object,
    bar_close: float,
    atr: float,
    spread_points: float,
    point: float,
    balance: float,
    equity: float,
    open_risk_ratio: float,
    day_trades: int,
    max_positions: int,
    max_holding_bars: int,
    min_daily_trades: int,
    max_daily_trades: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror EA ComputeAllInputs dynamic slots from explicit primitives."""
    atr_value = max(float(atr), 1e-10)
    # ComputeAllInputs uses integer SYMBOL_SPREAD rather than the floating
    # ask-bid quotient exported for observability.
    spread_price = round(float(spread_points)) * float(point)
    closed_epoch = int(pd.Timestamp(bar_time).timestamp())
    pos_features = np.zeros(40, dtype=float)
    exec_features = np.zeros(10, dtype=float)
    for slot, position in enumerate(positions[:5]):
        offset = slot * 8
        side = int(position.side)
        current_exit = float(bar_close) if side == 1 else float(bar_close) + spread_price
        risk_distance = max(float(position.initial_risk_distance), 1e-12)
        sl_distance = abs(float(position.entry) - float(position.sl)) / atr_value
        tp_distance = abs(float(position.tp) - float(position.entry)) / atr_value
        pos_features[offset:offset + 8] = (
            1.0,
            float(side),
            float(np.clip(side * (float(bar_close) - float(position.entry)) / atr_value, -20.0, 20.0)),
            float(np.clip(((closed_epoch - int(position.open_time)) / 60.0) / max(1, int(max_holding_bars)), 0.0, 5.0)),
            sl_distance,
            tp_distance,
            min(tp_distance / max(sl_distance, 1e-12), 5.0) / 5.0,
            float(np.clip(side * (current_exit - float(position.entry)) / risk_distance, -5.0, 5.0)),
        )
        exec_offset = slot * 2
        if side == 1:
            exec_features[exec_offset] = np.clip((current_exit - position.sl) / atr_value, -20.0, 20.0)
            exec_features[exec_offset + 1] = np.clip((position.tp - current_exit) / atr_value, -20.0, 20.0)
        else:
            exec_features[exec_offset] = np.clip((position.sl - current_exit) / atr_value, -20.0, 20.0)
            exec_features[exec_offset + 1] = np.clip((current_exit - position.tp) / atr_value, -20.0, 20.0)

    decision = pd.Timestamp(decision_time)
    day_progress = (decision.hour * 60.0 + decision.minute) / 1440.0
    min_daily = max(0, int(min_daily_trades))
    max_daily = max(0, int(max_daily_trades))
    expected = min_daily * day_progress
    cadence = (expected - int(day_trades)) / max(min_daily, 1) if expected > int(day_trades) else 0.0
    safe_balance = float(balance)
    account = np.asarray((
        len(positions) / float(position_count_reference(max_positions)),
        float(open_risk_ratio),
        max(0.0, (safe_balance - float(equity)) / safe_balance) if safe_balance > 0.0 else 0.0,
        float(equity) / safe_balance - 1.0 if safe_balance > 0.0 else 0.0,
        min(int(day_trades), max_daily) / max(max_daily, 1),
        min(int(day_trades), min_daily) / max(min_daily, 1),
        float(np.clip(day_progress, 0.0, 1.0)),
        float(np.clip(cadence, 0.0, 1.0)),
    ), dtype=float)
    return pos_features, account, exec_features


class Model1OracleDataset:
    """Decision boundary captured from the production EA on tester model 1."""

    def __init__(
        self,
        bridge_path: str | Path,
        feature_audit_path: str | Path,
        execution_path: str | Path | None = None,
    ):
        self.bridge_path = Path(bridge_path)
        self.feature_audit_path = Path(feature_audit_path)
        self.execution_path = Path(execution_path) if execution_path else None

    def decisions(self) -> tuple[Model1OracleDecision, ...]:
        bridge = pd.read_csv(self.bridge_path, sep="\t", dtype=str, keep_default_na=False)
        audit = pd.read_csv(self.feature_audit_path, sep="\t", dtype=str, keep_default_na=False)
        required_bridge = {
            "event_id", "inputs_pipe", "outputs_pipe", "action", "balance", "equity",
            "drawdown_pct", "spread_points", "positions_pipe", "orders_pipe", "response_status", "execution_pipe",
        }
        required_audit = {
            "event_id", "inputs", "logged_at", "bar_time", "bar_close", "atr",
            "open_risk_ratio", "symbol_point", "day_trades", "positions_pipe",
            "orders_pipe", "balance", "equity", "drawdown_pct", "spread_points",
            "margin", "free_margin", "margin_level_pct", "leverage",
        }
        if missing := required_bridge.difference(bridge.columns):
            raise ValueError(f"model-1 bridge oracle missing columns: {sorted(missing)}")
        if missing := required_audit.difference(audit.columns):
            raise ValueError(f"model-1 feature oracle missing columns: {sorted(missing)}")
        if bridge["event_id"].duplicated().any() or audit["event_id"].duplicated().any():
            raise ValueError("single-agent parity requires one unique row per event_id")
        audit_rows = {str(row["event_id"]): row for _, row in audit.iterrows()}
        decisions: list[Model1OracleDecision] = []
        for index, row in bridge.iterrows():
            event_id = str(row["event_id"])
            if event_id not in audit_rows:
                raise ValueError(f"feature audit missing bridge event {event_id}")
            audit_row = audit_rows[event_id]
            bridge_inputs = _pipe_vector(row["inputs_pipe"])
            raw_inputs = _pipe_vector(audit_row["inputs"])
            if bridge_inputs.size != raw_inputs.size or not np.array_equal(bridge_inputs, raw_inputs):
                raise ValueError(f"EA bridge/audit input mismatch at event {event_id}")
            decisions.append(Model1OracleDecision(
                sequence=index + 1, event_id=event_id, raw_inputs=raw_inputs,
                outputs=_pipe_vector(row["outputs_pipe"]), action=int(row["action"]),
                # Inputs are computed before the action. Feature-audit state is
                # therefore authoritative; bridge state is appended after
                # ExecuteAction and would introduce a one-decision lookahead.
                balance=float(audit_row["balance"]), equity=float(audit_row["equity"]),
                margin=float(audit_row["margin"]), free_margin=float(audit_row["free_margin"]),
                margin_level_pct=float(audit_row["margin_level_pct"]),
                leverage=int(float(audit_row["leverage"])),
                drawdown_pct=float(audit_row["drawdown_pct"]), spread_points=float(audit_row["spread_points"]),
                positions_pipe=str(audit_row["positions_pipe"]), orders_pipe=str(audit_row["orders_pipe"]),
                execution_pipe=str(row["execution_pipe"]),
                response_status=str(row["response_status"]),
                logged_at=str(audit_row["logged_at"]), bar_time=str(audit_row["bar_time"]),
                bar_close=float(audit_row["bar_close"]), atr=float(audit_row["atr"]),
                open_risk_ratio=float(audit_row["open_risk_ratio"]),
                point=float(audit_row["symbol_point"]), day_trades=int(audit_row["day_trades"]),
            ))
        if not decisions:
            raise ValueError("model-1 oracle contains no decisions")
        return tuple(decisions)

    def market_frames(self) -> dict[str, np.ndarray]:
        return {
            decision.event_id: canonical_market_vector(decision.raw_inputs)
            for decision in self.decisions()
        }

    def reconstruct_inputs(
        self,
        *,
        max_positions: int,
        max_holding_bars: int,
        min_daily_trades: int,
        max_daily_trades: int,
    ) -> tuple[np.ndarray, ...]:
        reconstructed: list[np.ndarray] = []
        for decision in self.decisions():
            position_features, account_features, execution_features = build_dynamic_input_blocks(
                positions=parse_replay_positions(decision.positions_pipe),
                bar_time=decision.bar_time, decision_time=decision.logged_at,
                bar_close=decision.bar_close, atr=decision.atr,
                spread_points=decision.spread_points, point=decision.point,
                balance=decision.balance, equity=decision.equity,
                open_risk_ratio=decision.open_risk_ratio, day_trades=decision.day_trades,
                max_positions=max_positions, max_holding_bars=max_holding_bars,
                min_daily_trades=min_daily_trades, max_daily_trades=max_daily_trades,
            )
            reconstructed.append(compose_canonical_inputs(
                decision.raw_inputs,
                position_features=position_features,
                account_features=account_features,
                execution_position_features=execution_features,
            ))
        return tuple(reconstructed)

    def compile(self, target: str | Path) -> Path:
        """Compile CSV capture once; Python replay never needs CSV in its hot loop."""
        decisions = self.decisions()
        output = Path(target)
        output.parent.mkdir(parents=True, exist_ok=True)
        inputs = np.stack([item.raw_inputs for item in decisions]).astype(np.float64, copy=False)
        outputs = np.stack([item.outputs for item in decisions]).astype(np.float64, copy=False)
        actions = np.asarray([item.action for item in decisions], dtype=np.int16)
        metadata = np.asarray([
            json.dumps({
                "sequence": item.sequence,
                "event_id": item.event_id,
                # WriteRequest is the policy side-channel contract. Keep its
                # exact MT5 decimal widths separate from the 12-decimal audit
                # primitives used to reconstruct the 137 input vector.
                "balance": float(f"{item.balance:.2f}"),
                "equity": float(f"{item.equity:.2f}"),
                "margin": float(f"{item.margin:.2f}"),
                "free_margin": float(f"{item.free_margin:.2f}"),
                "margin_level_pct": float(f"{item.margin_level_pct:.6f}"),
                "leverage": int(item.leverage),
                "drawdown_pct": float(f"{item.drawdown_pct:.6f}"),
                "spread_points": float(f"{item.spread_points:.2f}"),
                "positions_pipe": item.positions_pipe,
                "orders_pipe": item.orders_pipe,
                "execution_pipe": item.execution_pipe,
                "response_status": item.response_status,
                "logged_at": item.logged_at,
                "bar_time": item.bar_time,
                "bar_close": item.bar_close,
                "atr": item.atr,
                "open_risk_ratio": item.open_risk_ratio,
                "point": item.point,
                "day_trades": item.day_trades,
            }, ensure_ascii=False, separators=(",", ":"))
            for item in decisions
        ], dtype=np.str_)
        execution_lines: list[str] = []
        if self.execution_path is not None:
            execution_lines = self.execution_path.read_text(
                encoding="utf-8-sig", errors="strict"
            ).splitlines()
        np.savez_compressed(
            output,
            schema_version=np.asarray([MODEL1_ORACLE_TAPE_SCHEMA_VERSION], dtype=np.int16),
            inputs=inputs,
            outputs=outputs,
            actions=actions,
            metadata=metadata,
            execution_lines=np.asarray(execution_lines, dtype=np.str_),
        )
        return output


class Model1OracleTape:
    """Fail-closed, memory-mapped-friendly decision tape for Python replay."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            version = int(np.asarray(payload["schema_version"]).reshape(-1)[0])
            if version != MODEL1_ORACLE_TAPE_SCHEMA_VERSION:
                raise ValueError(f"unsupported model-1 oracle tape schema: {version}")
            self.inputs = np.asarray(payload["inputs"], dtype=np.float64)
            self.outputs = np.asarray(payload["outputs"], dtype=np.float64)
            self.actions = np.asarray(payload["actions"], dtype=np.int16)
            self.metadata = tuple(json.loads(str(item)) for item in payload["metadata"])
            self.execution_lines = tuple(
                str(item) for item in payload["execution_lines"]
            ) if "execution_lines" in payload.files else ()
        rows = self.inputs.shape[0]
        if self.inputs.ndim != 2 or self.inputs.shape[1] != 137:
            raise ValueError(f"model-1 tape inputs must have shape (N, 137), got {self.inputs.shape}")
        if self.outputs.ndim != 2 or self.outputs.shape != (rows, 20):
            raise ValueError(f"model-1 tape outputs must have shape ({rows}, 20), got {self.outputs.shape}")
        if self.actions.shape != (rows,) or len(self.metadata) != rows:
            raise ValueError("model-1 tape arrays have inconsistent row counts")
        if not np.isfinite(self.inputs).all() or not np.isfinite(self.outputs).all():
            raise ValueError("model-1 tape contains non-finite values")
        account_keys = (
            "balance", "equity", "margin", "free_margin", "margin_level_pct", "leverage",
        )
        for index, item in enumerate(self.metadata, start=1):
            missing = [key for key in account_keys if key not in item]
            if missing:
                raise ValueError(
                    f"model-1 tape row {index} missing account/margin fields: {missing}"
                )
            values = np.asarray([float(item[key]) for key in account_keys], dtype=float)
            if not np.isfinite(values).all() or float(item["margin"]) < 0.0:
                raise ValueError(f"model-1 tape row {index} has invalid account/margin state")
            if int(item["leverage"]) <= 0:
                raise ValueError(f"model-1 tape row {index} has invalid leverage")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def content_sha256(self) -> str:
        digest = sha256()
        digest.update(self.inputs.tobytes(order="C"))
        digest.update(self.outputs.tobytes(order="C"))
        digest.update(self.actions.tobytes(order="C"))
        for item in self.metadata:
            digest.update(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for line in self.execution_lines:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest().upper()

    @property
    def decision_wire_sha256(self) -> str:
        digest = sha256()
        digest.update(self.inputs.tobytes(order="C"))
        digest.update(self.outputs.tobytes(order="C"))
        digest.update(self.actions.tobytes(order="C"))
        return digest.hexdigest().upper()

    @property
    def logical_state_sha256(self) -> str:
        digest = sha256()
        for item in self.metadata:
            state = {
                key: item.get(key)
                for key in (
                    "sequence", "event_id", "balance", "equity", "drawdown_pct",
                    "margin", "free_margin", "margin_level_pct", "leverage",
                    "spread_points", "positions_pipe", "orders_pipe", "execution_pipe",
                    "day_trades", "bar_close", "atr",
                )
            }
            digest.update(json.dumps(
                state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest().upper()

    @property
    def account_margin_state_sha256(self) -> str:
        digest = sha256()
        for item in self.metadata:
            state = {
                key: item[key]
                for key in (
                    "sequence", "event_id", "balance", "equity", "margin",
                    "free_margin", "margin_level_pct", "leverage",
                )
            }
            digest.update(json.dumps(
                state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest().upper()

    @property
    def execution_ledger_sha256(self) -> str:
        digest = sha256()
        for line in self.execution_lines:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest().upper()


@dataclass(frozen=True)
class Model1ExecutionContractReport:
    exact: bool
    checked_decisions: int
    checked_entry_geometries: int
    checked_execution_rows: int
    mismatches: tuple[str, ...]


def validate_model1_execution_contract(
    tape: Model1OracleTape,
    *,
    price_tick: float = 0.01,
    volume_step: float = 0.01,
) -> Model1ExecutionContractReport:
    """Validate command, state-pipe and requested-order geometry invariants."""
    mismatches: list[str] = []
    if not tape.execution_lines:
        return Model1ExecutionContractReport(False, 0, 0, 0, ("missing execution ledger",))
    reader = csv.DictReader(
        io.StringIO("\n".join(tape.execution_lines) + "\n"), delimiter="\t"
    )
    rows = list(reader)
    batch: dict[int, dict[str, str]] = {}
    for row in rows:
        if str(row.get("transaction_type", "")) != "BATCH_RESOLUTION":
            continue
        try:
            sequence = int(float(row.get("ack_sequence", "0") or 0))
        except ValueError:
            mismatches.append("batch row has invalid ack_sequence")
            continue
        if sequence in batch:
            mismatches.append(f"sequence {sequence}: duplicate BATCH_RESOLUTION")
        batch[sequence] = row
    geometry_count = 0
    tolerance = max(float(price_tick) * 5.0, 1e-9)
    for index in range(len(tape)):
        sequence = index + 1
        expected_action = int(tape.actions[index])
        row = batch.get(sequence)
        if row is None:
            mismatches.append(f"sequence {sequence}: missing BATCH_RESOLUTION")
            continue
        try:
            actual_action = int(float(row.get("requested_action", "-1")))
        except ValueError:
            actual_action = -1
        if actual_action != expected_action:
            mismatches.append(
                f"sequence {sequence}: ledger action {actual_action} != wire {expected_action}"
            )
        if str(row.get("trace_id", "")) != f"policy~model1-{sequence}":
            mismatches.append(f"sequence {sequence}: non-deterministic trace_id")
        positions = str(tape.metadata[index].get("positions_pipe", ""))
        try:
            parse_replay_positions(positions)
        except ValueError as exc:
            mismatches.append(f"sequence {sequence}: invalid positions_pipe: {exc}")
        orders = str(tape.metadata[index].get("orders_pipe", ""))
        for token in (item for item in orders.split("^") if item):
            parts = token.split(":")
            if len(parts) != 4:
                mismatches.append(f"sequence {sequence}: invalid orders_pipe token")
                continue
            try:
                volume = float(parts[2])
                price = float(parts[3])
            except ValueError:
                mismatches.append(f"sequence {sequence}: non-numeric pending order")
                continue
            steps = volume / max(float(volume_step), 1e-12)
            if volume <= 0.0 or price <= 0.0 or abs(steps - round(steps)) > 1e-7:
                mismatches.append(f"sequence {sequence}: illegal pending volume/price")
        if expected_action not in {1, 2, 3, 4, 5, 6}:
            continue
        parts = str(row.get("geometry_entry_pipe", "")).split("|")
        if len(parts) < 13:
            mismatches.append(f"sequence {sequence}: missing entry geometry")
            continue
        try:
            entry, sl, tp, atr = map(float, parts[5:9])
            fixed_sl, fixed_tp, fixed_rr = map(float, parts[10:13])
        except ValueError:
            mismatches.append(f"sequence {sequence}: non-numeric entry geometry")
            continue
        geometry_count += 1
        side = 1 if expected_action in {1, 3, 5} else -1
        expected_sl = entry - side * fixed_sl * atr
        expected_tp = entry + side * fixed_tp * atr
        if min(entry, sl, tp, atr) <= 0.0:
            mismatches.append(f"sequence {sequence}: non-positive entry geometry")
        if abs(sl - expected_sl) > tolerance or abs(tp - expected_tp) > tolerance:
            mismatches.append(f"sequence {sequence}: SL/TP geometry drift")
        if abs(fixed_rr - fixed_tp / max(fixed_sl, 1e-12)) > 1e-9:
            mismatches.append(f"sequence {sequence}: fixed RR drift")
    return Model1ExecutionContractReport(
        exact=not mismatches,
        checked_decisions=len(tape),
        checked_entry_geometries=geometry_count,
        checked_execution_rows=len(rows),
        mismatches=tuple(mismatches),
    )
