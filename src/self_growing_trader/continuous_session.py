"""Event-driven continuous learning runtime (V27).

This module replaces *external* train round/trial orchestration with a persistent
session runtime. It is intentionally additive: the existing graph/M1 policy,
causal ledger and MT5 bridge remain the execution source of truth.

Key invariants
--------------
* A final lifecycle is consumed at most once (signature based idempotency).
* No learning update is applied before the next bridge request safe-point.
* A post-exit wait is sequence based; lack of market-path data is recorded as
  incomplete evidence instead of fabricated as a causal conclusion.
* Replays remain externally verified. The runtime consumes verified results;
  it never invents a result from PnL sign alone.
* Continuous specialist modules are persistent policy state. They are created
  only after repeated, replay-validated evidence and evolve through
  probation -> active -> consolidated -> fading/pruned.

The canonical EA is not part of this archive. Therefore real-time post-exit
trajectory evidence requires the EA to emit the documented V24 execution
ledger columns. Legacy ledgers still permit mechanical lifecycle settlement,
but do not unlock trajectory-dependent causal conclusions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .causal_ledger import CausalOutcomeReport, ExitMechanism, classify_exit_mechanism
from .module_need import (
    ROLE_CRITIC_SPECIALIST,
    ROLE_ENTRY_QUALITY_SPECIALIST,
    ROLE_EXIT_QUALITY_SPECIALIST,
    ROLE_MEMORY_CRITIC_SPECIALIST,
    ROLE_POSITION_MANAGEMENT_SPECIALIST,
    ROLE_RISK_GATE_SPECIALIST,
    ROLE_SELECTOR_SPECIALIST,
)
from .neural_graph import NEURON_LIFECYCLE_ACTIVE, NEURON_LIFECYCLE_CONSOLIDATED, NEURON_LIFECYCLE_PRUNING
from .specialist_module_factory import SpecialistModuleFactory

CONTINUOUS_SCHEMA_VERSION = 27


def _float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace("\xa0", " ").replace(" ", "").replace(",", "")
        if text in {"", "-", "None", "none"}:
            return float(default)
        parsed = float(text)
        return parsed if math.isfinite(parsed) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok", "success"}


def _optional_bool(row: Mapping[str, object], *keys: str) -> bool | None:
    for key in keys:
        if key in row and str(row.get(key, "")).strip() != "":
            return _bool(row.get(key))
    return None


def _has_value(row: Mapping[str, object], *keys: str) -> bool:
    return any(
        key in row and str(row.get(key, "")).strip() not in {"", "-", "None", "none"}
        for key in keys
    )


def _expand_geometry_path_pipe(row: Mapping[str, object]) -> dict[str, object]:
    """Read both the five-value V15 and eight-value V16 path pipe."""
    expanded = dict(row)
    values = str(expanded.get("geometry_path_pipe", "") or "").split("|")
    keys = (
        "path_high_price", "path_low_price", "path_close_price", "path_atr",
        "path_spread_price", "unrealized_r", "mfe_r", "mae_r",
    )
    if len(values) >= 5:
        for key, value in zip(keys, values):
            if not str(expanded.get(key, "") or "").strip():
                expanded[key] = value
    entry_values = str(expanded.get("geometry_entry_pipe", "") or "").split("|")
    if len(entry_values) >= 2:
        if not str(expanded.get("initial_atr", "") or "").strip():
            expanded["initial_atr"] = entry_values[0]
        if not str(expanded.get("side", "") or "").strip():
            expanded["side"] = entry_values[1]
    return expanded


def _text(row: Mapping[str, object], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(default)


def _value(row: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() not in {"", "-", "None", "none"}:
            return _float(value, default)
    return float(default)


def _time(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in (
        "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def _vector(values: Any, limit: int = 128) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=float)
    array = array[: max(0, int(limit))]
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine(left: Any, right: Any) -> float:
    a = _vector(left)
    b = _vector(right)
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip((a @ b) / denom, -1.0, 1.0)) if denom > 1e-12 else 0.0


def _ctx_key(vector: Any, fallback: str = "") -> str:
    arr = _vector(vector, 64)
    if arr.size:
        payload = ",".join(f"{value:.3f}" for value in arr)
        return "ctx:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return str(fallback or "ctx:unknown")


def _event_id(kind: str, *values: object) -> str:
    payload = "|".join([kind, *[str(value) for value in values]])
    return f"{kind}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]}"


def _report_fingerprint(report: CausalOutcomeReport) -> str:
    """Return a stable version marker for a causal report.

    ``_harvest_reports`` runs at every bridge safe point.  A report must be
    processed again only when replay coverage or its hypotheses actually
    changed; otherwise one SL can repeatedly strengthen the same inhibition on
    every later bar.  The fingerprint intentionally includes replay validation
    and each hypothesis state, but excludes timestamps/UI-only fields.
    """
    payload = {
        "trace_id": str(getattr(report, "trace_id", "")),
        "coverage": dict(getattr(report, "coverage", {}) or {}),
        "hypotheses": [
            {
                "cause": str(getattr(item, "cause_id", "")),
                "status": str(getattr(item, "status", "")),
                "confidence": round(float(getattr(item, "confidence", 0.0) or 0.0), 6),
            }
            for item in tuple(getattr(report, "hypotheses", ()) or ())
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContinuousRuntimeConfig:
    enabled: bool = False
    session_root: str = "outputs/continuous"
    post_exit_bars: int = 20
    max_lifecycle_wait_sequences: int = 32
    checkpoint_every_closed_positions: int = 5
    checkpoint_every_minutes: int = 15
    max_event_bytes: int = 128 * 1024 * 1024
    min_independent_evidence: int = 3
    min_verified_replays_to_create_module: int = 2
    min_verified_replays_to_activate_module: int = 3
    min_verified_replays_to_consolidate_module: int = 6
    min_causal_confidence: float = 0.65
    max_probation_modules: int = 12
    max_active_modules: int = 24
    max_modules_per_context_cause: int = 1
    max_modules_per_decision: int = 3
    max_total_action_bias: float = 0.20
    # V24 trade-activity safeguards.  Inhibition is deliberately stricter than
    # specialist creation: it can suppress an entry, so it requires repeated
    # independent, replay-verified evidence and a narrower context match.
    min_independent_evidence_to_inhibit: int = 3
    min_verified_replays_to_inhibit: int = 2
    local_inhibition_min_similarity: float = 0.90
    local_inhibition_max_strength: float = 0.12
    local_inhibition_half_life_positions: int = 80
    # Trade-activity observability/recovery.  These values never force an
    # entry; they expose the decision -> legal -> emitted -> fill funnel and
    # release a stale pending order through the existing CANCEL primitive.
    trade_funnel_emit_every_decisions: int = 16
    trade_activity_warning_after_decisions: int = 48
    pending_cancel_after_decisions: int = 12
    pending_cancel_interval_decisions: int = 4
    enable_local_inhibition: bool = True
    enable_persistent_specialists: bool = True
    # V25 trade-activity hardening.  Old V15-V24 checkpoints may contain
    # global HOLD/risk bias and inhibitions built from hypothesis indices.
    # Migrate only adaptive calibration; keep topology, causal evidence and
    # specialist modules intact.
    migrate_legacy_trade_activity: bool = True
    legacy_risk_gate_rebase_factor: float = 0.15
    legacy_output_hold_rebase_factor: float = 0.20
    risk_gate_entry_dampen_max: float = 0.08
    inactivity_recalibration_after_decisions: int = 48
    inactivity_risk_gate_dampen_cap: float = 0.02
    require_execution_action_attribution: bool = True
    # V27 flat-idle firebreak:
    #
    # ``positions == 0`` / no pending order is an MT5 *execution state*, never
    # an action selected by the neural policy.  Explicit neural HOLD remains a
    # legal policy output, but a flat MT5 snapshot cannot feed HOLD, reward,
    # memory, starvation, or causal credit back into the network.
    #
    # The V26 non-HOLD selector is retained only for backwards-compatible
    # experiments.  It is OFF by default because suppressing a genuine HOLD
    # neuron was not the requested fix.
    exclude_hold_from_neural_selection: bool = False
    neural_command_min_score: float = 0.0
    flat_idle_is_non_feedback: bool = True
    suppress_flat_idle_starvation: bool = True

    def __post_init__(self) -> None:
        if self.post_exit_bars < 0:
            raise ValueError("continuous.post_exit_bars must be >= 0")
        if self.max_lifecycle_wait_sequences < 1:
            raise ValueError("continuous.max_lifecycle_wait_sequences must be >= 1")
        if self.checkpoint_every_closed_positions < 1:
            raise ValueError("continuous.checkpoint_every_closed_positions must be >= 1")
        if self.checkpoint_every_minutes < 1:
            raise ValueError("continuous.checkpoint_every_minutes must be >= 1")
        if self.min_independent_evidence < 1:
            raise ValueError("continuous.min_independent_evidence must be >= 1")
        if self.min_independent_evidence_to_inhibit < 1:
            raise ValueError("continuous.min_independent_evidence_to_inhibit must be >= 1")
        if self.min_verified_replays_to_inhibit < 1:
            raise ValueError("continuous.min_verified_replays_to_inhibit must be >= 1")
        if not 0.50 <= self.local_inhibition_min_similarity <= 1.0:
            raise ValueError("continuous.local_inhibition_min_similarity must be within [0.50, 1.00]")
        if not 0.0 <= self.local_inhibition_max_strength <= self.max_total_action_bias:
            raise ValueError("continuous.local_inhibition_max_strength must be within [0, max_total_action_bias]")
        if self.trade_funnel_emit_every_decisions < 1:
            raise ValueError("continuous.trade_funnel_emit_every_decisions must be >= 1")
        if self.trade_activity_warning_after_decisions < 1:
            raise ValueError("continuous.trade_activity_warning_after_decisions must be >= 1")
        if self.pending_cancel_after_decisions < 1:
            raise ValueError("continuous.pending_cancel_after_decisions must be >= 1")
        if self.pending_cancel_interval_decisions < 1:
            raise ValueError("continuous.pending_cancel_interval_decisions must be >= 1")
        if not 0.0 <= self.min_causal_confidence <= 1.0:
            raise ValueError("continuous.min_causal_confidence must be within [0,1]")
        if not 0.0 <= self.max_total_action_bias <= 1.0:
            raise ValueError("continuous.max_total_action_bias must be within [0,1]")
        if not 0.0 <= self.legacy_risk_gate_rebase_factor <= 1.0:
            raise ValueError("continuous.legacy_risk_gate_rebase_factor must be within [0,1]")
        if not 0.0 <= self.legacy_output_hold_rebase_factor <= 1.0:
            raise ValueError("continuous.legacy_output_hold_rebase_factor must be within [0,1]")
        if not 0.0 <= self.risk_gate_entry_dampen_max <= 0.25:
            raise ValueError("continuous.risk_gate_entry_dampen_max must be within [0,0.25]")
        if self.inactivity_recalibration_after_decisions < 1:
            raise ValueError("continuous.inactivity_recalibration_after_decisions must be >= 1")
        if not 0.0 <= self.inactivity_risk_gate_dampen_cap <= self.risk_gate_entry_dampen_max:
            raise ValueError("continuous.inactivity_risk_gate_dampen_cap must be within [0, risk_gate_entry_dampen_max]")
        if not -1.0 <= self.neural_command_min_score <= 1.0:
            raise ValueError("continuous.neural_command_min_score must be within [-1,1]")


@dataclass
class ContinuousLifecycle:
    position_ticket: str
    trace_id: str
    entry_sequence: int = 0
    close_sequence: int = 0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    initial_risk: float = 0.0
    realized_r: float = 0.0
    duration_bars: float | None = None
    close_reason: str = ""
    final_close_confirmed: bool = True
    exit_mechanism: str = ExitMechanism.UNKNOWN.value
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_sl_price: float = 0.0
    final_sl_price: float = 0.0
    initial_tp_price: float = 0.0
    final_tp_price: float = 0.0
    partial_close_count: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    excursion_available: bool = False
    giveback_r: float = 0.0
    near_tp_ratio: float = 0.0
    slippage_r: float = 0.0
    post_exit_mfe_r: float | None = None
    post_exit_mae_r: float | None = None
    returned_to_entry: bool | None = None
    original_tp_hit_after_exit: bool | None = None
    trajectory: tuple[dict[str, object], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return _event_id(
            "final_lifecycle",
            self.position_ticket,
            self.trace_id,
            self.close_sequence,
            self.exit_time.isoformat() if self.exit_time else "",
            round(float(self.net_pnl), 8),
            self.close_reason,
        )


@dataclass
class _PositionAccumulator:
    ticket: str
    trace_id: str = ""
    entry_sequence: int = 0
    close_sequence: int = 0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    initial_risk: float = 0.0
    duration_bars: float | None = None
    close_reason: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_sl_price: float = 0.0
    final_sl_price: float = 0.0
    initial_tp_price: float = 0.0
    final_tp_price: float = 0.0
    partial_close_count: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    excursion_available: bool = False
    side: int = 0
    initial_risk_distance_price: float = 0.0
    giveback_r: float = 0.0
    near_tp_ratio: float = 0.0
    slippage_r: float = 0.0
    post_exit_mfe_r: float | None = None
    post_exit_mae_r: float | None = None
    returned_to_entry: bool | None = None
    original_tp_hit_after_exit: bool | None = None
    trajectory: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    final_emitted: bool = False


class ExecutionLedgerTailer:
    """Incrementally consume append-only MT5 execution CSV/TSV rows.

    The tailer deliberately accepts only explicit final-close rows for online
    settlement. A legacy ledger without ``position_final_close`` remains valid
    for post-run fitness parsing, but cannot safely trigger mid-run learning.
    """

    def __init__(self, initial_equity: float = 10000.0) -> None:
        self.initial_equity = max(1e-9, float(initial_equity))
        self.path: Path | None = None
        self.offset = 0
        self.header: list[str] = []
        self.delimiter = ","
        self.partial = ""
        self.positions: dict[str, _PositionAccumulator] = {}
        # Final lifecycle objects remain mutable while the EA emits post-exit
        # observation rows. Pending settlements hold the same object reference.
        self.final_lifecycles: dict[str, ContinuousLifecycle] = {}
        self.emitted: set[str] = set()

    def reset(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.offset = 0
        self.header = []
        self.delimiter = ","
        self.partial = ""
        self.positions.clear()
        self.final_lifecycles.clear()
        self.emitted.clear()

    def poll(self, path: str | Path | None = None) -> list[ContinuousLifecycle]:
        if path is not None:
            candidate = Path(path)
            if self.path is None or candidate.resolve() != self.path.resolve():
                self.reset(candidate)
        if self.path is None or not self.path.exists():
            return []
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:
            self.reset(self.path)
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                raw = handle.read()
                self.offset = handle.tell()
        except OSError:
            return []
        if not raw:
            return []
        text = self.partial + raw.decode("utf-8", errors="ignore")
        if not text:
            return []
        if not text.endswith("\n") and not text.endswith("\r"):
            pieces = text.splitlines()
            self.partial = pieces.pop() if pieces else text
        else:
            pieces = text.splitlines()
            self.partial = ""
        if not pieces:
            return []
        if not self.header:
            first = pieces.pop(0)
            self.delimiter = "\t" if "\t" in first else ","
            self.header = [str(item).strip() for item in next(csv.reader([first], delimiter=self.delimiter), [])]
        if not self.header:
            return []
        output: list[ContinuousLifecycle] = []
        for line in pieces:
            if not line.strip():
                continue
            try:
                values = next(csv.reader([line], delimiter=self.delimiter))
            except (csv.Error, StopIteration):
                continue
            if len(values) < 1:
                continue
            row = _expand_geometry_path_pipe({
                self.header[index]: values[index] if index < len(values) else ""
                for index in range(len(self.header))
            })
            lifecycle = self._ingest(row)
            if lifecycle is not None:
                output.append(lifecycle)
        return output

    def _trajectory_point(self, row: Mapping[str, object], event_time: datetime | None) -> dict[str, object] | None:
        event_kind = _text(row, "transaction_type", "event_type").upper()
        has_path = all(
            _value(row, key) > 0.0
            for key in ("path_high_price", "path_low_price", "path_close_price")
        )
        explicit_excursion = _has_value(row, "unrealized_r", "mfe_r", "mae_r")
        other_markers = (
            "market_price", "last_price", "management_event", "volatility_shock",
            "spread_atr_ratio",
        )
        if not (
            event_kind == "POSITION_TRAJECTORY"
            or has_path
            or explicit_excursion
            or any(str(row.get(key, "")).strip() for key in other_markers)
        ):
            return None
        return {
            "timestamp": event_time.timestamp() if event_time is not None else 0.0,
            "age_bars": _value(row, "age_bars", "bar_index", default=-1.0),
            "price": _value(row, "market_price", "last_price", "price", "bid", "close"),
            "high": _value(row, "path_high_price", "high", "bar_high"),
            "low": _value(row, "path_low_price", "low", "bar_low"),
            "close": _value(row, "path_close_price", "close", "price"),
            "spread": _value(row, "path_spread_price", "spread_price"),
            "unrealized_r": _value(row, "unrealized_r", "floating_r"),
            "mfe_r": _value(row, "mfe_r"),
            "mae_r": _value(row, "mae_r"),
            "distance_to_sl_atr": _value(row, "distance_to_sl_atr", "sl_distance_atr"),
            "distance_to_tp_atr": _value(row, "distance_to_tp_atr", "tp_distance_atr"),
            "spread_atr_ratio": _value(row, "spread_atr_ratio"),
            "atr": _value(row, "path_atr", "atr"),
            "volatility_shock": _value(row, "volatility_shock", "volatility_shock_score"),
            "regime": _text(row, "regime", "market_regime"),
            "management_event": _text(row, "management_event", "event_type", "event"),
        }

    def _apply_excursion_evidence(
        self,
        row: Mapping[str, object],
        state: _PositionAccumulator,
        point: dict[str, object] | None,
    ) -> None:
        if point is None:
            return
        if _has_value(row, "mfe_r") and _has_value(row, "mae_r"):
            state.mfe_r = max(state.mfe_r, _value(row, "mfe_r"))
            state.mae_r = min(state.mae_r, _value(row, "mae_r"))
            point["unrealized_r"] = _value(row, "unrealized_r", "floating_r")
            point["mfe_r"] = state.mfe_r
            point["mae_r"] = state.mae_r
            state.excursion_available = True
            state.metadata["excursion_evidence_source"] = "ledger_v16"
            return
        entry = float(state.entry_price)
        risk = float(state.initial_risk_distance_price)
        if risk <= 1e-12:
            risk = abs(entry - float(state.initial_sl_price))
        high = float(point.get("high", 0.0) or 0.0)
        low = float(point.get("low", 0.0) or 0.0)
        close = float(point.get("close", 0.0) or 0.0)
        spread = max(0.0, float(point.get("spread", 0.0) or 0.0))
        if entry <= 0.0 or risk <= 1e-12 or state.side not in {-1, 1}:
            return
        if high <= 0.0 or low <= 0.0 or close <= 0.0 or high + 1e-12 < low:
            return
        point_timestamp = float(point.get("timestamp", 0.0) or 0.0)
        if state.entry_time is not None and point_timestamp > 0.0:
            closed_bar_start = float(int(point_timestamp // 60.0) * 60 - 60)
            if state.entry_time.timestamp() > closed_bar_start + 1e-6:
                return
        if state.side > 0:
            favorable = (high - entry) / risk
            adverse = (low - entry) / risk
            unrealized = (close - entry) / risk
        else:
            favorable = (entry - (low + spread)) / risk
            adverse = (entry - (high + spread)) / risk
            unrealized = (entry - (close + spread)) / risk
        state.mfe_r = max(state.mfe_r, favorable)
        state.mae_r = min(state.mae_r, adverse)
        point["unrealized_r"] = float(unrealized)
        point["mfe_r"] = float(state.mfe_r)
        point["mae_r"] = float(state.mae_r)
        state.excursion_available = True
        state.metadata["excursion_evidence_source"] = "geometry_path_v15_reconstructed"

    def _ingest(self, row: Mapping[str, object]) -> ContinuousLifecycle | None:
        ticket = _text(row, "position_ticket")
        if not ticket or ticket in {"0", "0.0"}:
            return None
        filled = _bool(row.get("filled"))
        closed = _bool(row.get("closed"))
        final_field_present = "position_final_close" in row or "final_close" in row
        final_close = _bool(row.get("position_final_close", row.get("final_close", "0")))
        point = self._trajectory_point(row, _time(row.get("logged_at", row.get("event_time", ""))))
        if not (filled or closed or point is not None):
            return None
        state = self.positions.setdefault(ticket, _PositionAccumulator(ticket=ticket))
        event_time = _time(row.get("logged_at", row.get("event_time", "")))
        sequence = int(_value(row, "event_seq", "sequence", "ack_sequence", default=0.0))
        row_trace = _text(row, "trace_id")
        if row_trace and (filled or not state.trace_id):
            state.trace_id = row_trace
            if sequence > 0:
                state.entry_sequence = sequence
        if filled:
            if state.entry_time is None or (event_time is not None and event_time < state.entry_time):
                state.entry_time = event_time
        if closed:
            state.gross_profit += _value(row, "profit")
            state.commission += _value(row, "commission")
            state.swap += _value(row, "swap")
            state.net_pnl += _value(row, "profit") + _value(row, "commission") + _value(row, "swap")
            if event_time is not None and (state.exit_time is None or event_time >= state.exit_time):
                state.exit_time = event_time
            if sequence > 0:
                state.close_sequence = max(state.close_sequence, sequence)
            reason = _text(row, "close_reason")
            if reason and reason.upper() != "OTHER":
                state.close_reason = reason
            if not final_close:
                state.partial_close_count += 1
        state.initial_risk = max(state.initial_risk, _value(row, "initial_risk"))
        age = _value(row, "age_bars", default=-1.0)
        if age >= 0.0:
            state.duration_bars = max(float(state.duration_bars or 0.0), age)
        for attribute, aliases, reducer in (
            ("entry_price", ("entry_price", "open_price", "fill_price"), "replace"),
            ("exit_price", ("exit_price", "close_price", "deal_price"), "replace"),
            ("initial_sl_price", ("initial_sl_price", "initial_sl", "sl_initial"), "replace"),
            ("initial_risk_distance_price", ("initial_risk_distance_price",), "replace"),
            ("final_sl_price", ("final_sl_price", "final_sl", "sl"), "replace"),
            ("initial_tp_price", ("initial_tp_price", "initial_tp", "tp_initial"), "replace"),
            ("final_tp_price", ("final_tp_price", "final_tp", "tp"), "replace"),
            ("mfe_r", ("mfe_r",), "max"),
            ("mae_r", ("mae_r",), "min"),
            ("giveback_r", ("giveback_r",), "replace"),
            ("near_tp_ratio", ("near_tp_ratio",), "max"),
            ("slippage_r", ("slippage_r",), "replace"),
        ):
            value = _value(row, *aliases)
            if abs(value) <= 1e-12:
                continue
            current = float(getattr(state, attribute))
            if reducer == "max":
                setattr(state, attribute, max(current, value))
            elif reducer == "min":
                setattr(state, attribute, min(current, value))
            else:
                setattr(state, attribute, value)
        side = int(_value(row, "side", "direction", default=0.0))
        if side in {-1, 1}:
            state.side = side
        self._apply_excursion_evidence(row, state, point)
        if point is not None:
            state.trajectory.append(point)
            state.trajectory = state.trajectory[-512:]
        for attribute, aliases in (
            ("post_exit_mfe_r", ("post_exit_mfe_r", "post_exit_max_favorable_r")),
            ("post_exit_mae_r", ("post_exit_mae_r", "post_exit_max_adverse_r")),
        ):
            if any(str(row.get(alias, "")).strip() for alias in aliases):
                setattr(state, attribute, _value(row, *aliases))
        for attribute, aliases in (
            ("returned_to_entry", ("returned_to_entry", "price_returned_to_entry")),
            ("original_tp_hit_after_exit", ("original_tp_hit_after_exit", "original_tp_hit")),
        ):
            value = _optional_bool(row, *aliases)
            if value is not None:
                setattr(state, attribute, value)
        state.metadata.update({
            key: value for key, value in row.items()
            if key in {"symbol", "side", "direction", "entry_type", "initial_tp_r", "target_r", "market_regime", "run_id", "schema_version"}
            and str(value).strip()
        })
        # The EA can continue writing POST_EXIT_OBSERVATION rows after the
        # position final-close. Keep the already-emitted lifecycle object in
        # sync so the scheduler can use genuine future-path evidence at its
        # configured safe point. A later row never emits a second lifecycle.
        if state.final_emitted:
            existing = self.final_lifecycles.get(ticket)
            if existing is not None:
                existing.post_exit_mfe_r = state.post_exit_mfe_r
                existing.post_exit_mae_r = state.post_exit_mae_r
                existing.returned_to_entry = state.returned_to_entry
                existing.original_tp_hit_after_exit = state.original_tp_hit_after_exit
                existing.excursion_available = state.excursion_available
                existing.trajectory = tuple(state.trajectory)
                existing.metadata = {**dict(existing.metadata), **dict(state.metadata)}
            return None
        # Online adaptation needs an explicit final-close field. This avoids
        # treating a partial close or a transient closed flag as a final result.
        if not (closed and final_field_present and final_close):
            return None
        risk_unit = state.initial_risk if state.initial_risk > 1e-12 else self.initial_equity * 0.01
        realized_r = state.net_pnl / max(1e-9, risk_unit)
        duration = state.duration_bars
        if duration is None and state.entry_time is not None and state.exit_time is not None:
            duration = max(0.0, (state.exit_time - state.entry_time).total_seconds() / 60.0)
        mechanism = classify_exit_mechanism(state.close_reason, realized_r=realized_r, partial_close_count=state.partial_close_count)
        lifecycle = ContinuousLifecycle(
            position_ticket=ticket,
            trace_id=state.trace_id,
            entry_sequence=state.entry_sequence,
            close_sequence=state.close_sequence,
            entry_time=state.entry_time,
            exit_time=state.exit_time,
            net_pnl=state.net_pnl,
            gross_profit=state.gross_profit,
            commission=state.commission,
            swap=state.swap,
            initial_risk=state.initial_risk,
            realized_r=float(realized_r),
            duration_bars=duration,
            close_reason=state.close_reason,
            final_close_confirmed=True,
            exit_mechanism=mechanism.value,
            entry_price=state.entry_price,
            exit_price=state.exit_price,
            initial_sl_price=state.initial_sl_price,
            final_sl_price=state.final_sl_price,
            initial_tp_price=state.initial_tp_price,
            final_tp_price=state.final_tp_price,
            partial_close_count=state.partial_close_count,
            mfe_r=state.mfe_r,
            mae_r=state.mae_r,
            excursion_available=state.excursion_available,
            giveback_r=state.giveback_r,
            near_tp_ratio=state.near_tp_ratio,
            slippage_r=state.slippage_r,
            post_exit_mfe_r=state.post_exit_mfe_r,
            post_exit_mae_r=state.post_exit_mae_r,
            returned_to_entry=state.returned_to_entry,
            original_tp_hit_after_exit=state.original_tp_hit_after_exit,
            trajectory=tuple(state.trajectory),
            metadata={**state.metadata, "continuous_schema_version": CONTINUOUS_SCHEMA_VERSION},
        )
        if lifecycle.signature in self.emitted:
            return None
        state.final_emitted = True
        self.final_lifecycles[ticket] = lifecycle
        self.emitted.add(lifecycle.signature)
        return lifecycle


class ContinuousEventStore:
    """Append-only JSONL audit plus a small SQLite idempotency index."""

    def __init__(self, root: str | Path, session_id: str, max_event_bytes: int = 128 * 1024 * 1024) -> None:
        self.root = Path(root)
        self.session_id = str(session_id)
        self.dir = self.root / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.dir / "events.jsonl"
        self.db_path = self.dir / "session.sqlite"
        self.max_event_bytes = max(1024 * 1024, int(max_event_bytes))
        self._connection = sqlite3.connect(str(self.db_path))
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, event_type TEXT, sequence INTEGER, trace_id TEXT, created_at REAL, payload TEXT)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints (checkpoint_id TEXT PRIMARY KEY, sequence INTEGER, created_at REAL, path TEXT, reason TEXT)"
        )
        self._connection.commit()

    def close(self) -> None:
        try:
            self._connection.close()
        except Exception:
            pass

    def emit(self, event_type: str, *, event_id: str | None = None, sequence: int = 0, trace_id: str = "", payload: Mapping[str, Any] | None = None) -> bool:
        eid = str(event_id or _event_id(event_type, self.session_id, sequence, trace_id, time.time_ns()))
        item = {
            "schema_version": CONTINUOUS_SCHEMA_VERSION,
            "event_id": eid,
            "event_type": str(event_type),
            "session_id": self.session_id,
            "sequence": int(sequence),
            "trace_id": str(trace_id),
            "created_at": time.time(),
            "payload": dict(payload or {}),
        }
        try:
            self._connection.execute(
                "INSERT INTO events(event_id,event_type,sequence,trace_id,created_at,payload) VALUES(?,?,?,?,?,?)",
                (eid, str(event_type), int(sequence), str(trace_id), float(item["created_at"]), json.dumps(item["payload"], ensure_ascii=False, default=str)),
            )
            self._connection.commit()
        except sqlite3.IntegrityError:
            return False
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
        self._compact()
        return True

    def checkpoint(self, checkpoint_id: str, *, sequence: int, path: Path, reason: str) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO checkpoints(checkpoint_id,sequence,created_at,path,reason) VALUES(?,?,?,?,?)",
            (str(checkpoint_id), int(sequence), time.time(), str(path), str(reason)),
        )
        self._connection.commit()

    def _compact(self) -> None:
        try:
            if not self.event_path.exists() or self.event_path.stat().st_size <= self.max_event_bytes:
                return
            lines = self.event_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            keep = lines[-max(256, len(lines) // 2):]
            self.event_path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        except OSError:
            return


class ContinuousCheckpointManager:
    def __init__(self, store: ContinuousEventStore, checkpoint_every_closed_positions: int, checkpoint_every_minutes: int) -> None:
        self.store = store
        self.every_positions = max(1, int(checkpoint_every_closed_positions))
        self.every_seconds = max(60.0, float(checkpoint_every_minutes) * 60.0)
        self.last_closed = 0
        self.last_at = time.monotonic()

    def due(self, closed_positions: int) -> bool:
        return (closed_positions - self.last_closed) >= self.every_positions or (time.monotonic() - self.last_at) >= self.every_seconds

    def save(self, policy: Any, *, sequence: int, closed_positions: int, reason: str) -> Path | None:
        directory = self.store.dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_id = f"adaptive_{int(sequence):08d}_{int(time.time())}"
        target = directory / f"{checkpoint_id}.pkl"
        temporary = target.with_suffix(".tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, target)
        except Exception as exc:
            self.store.emit("checkpoint_error", sequence=sequence, payload={"reason": reason, "error": str(exc)})
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        self.last_closed = int(closed_positions)
        self.last_at = time.monotonic()
        self.store.checkpoint(checkpoint_id, sequence=sequence, path=target, reason=reason)
        self.store.emit("checkpoint_saved", event_id=_event_id("checkpoint", checkpoint_id), sequence=sequence, payload={"path": str(target), "reason": reason})
        return target


class ContinuousHealthMonitor:
    def __init__(self) -> None:
        self.outcomes: list[float] = []
        self.invalid_lifecycles = 0
        self.state = "GREEN"

    def observe(self, realized_r: float | None = None, *, invalid: bool = False) -> str:
        if invalid:
            self.invalid_lifecycles += 1
            self.state = "INVALID_LEDGER"
            return self.state
        if realized_r is not None:
            self.outcomes.append(float(np.clip(realized_r, -5.0, 5.0)))
            self.outcomes = self.outcomes[-250:]
        recent = self.outcomes[-20:]
        mean = float(np.mean(recent)) if recent else 0.0
        if self.invalid_lifecycles > 0:
            self.state = "INVALID_LEDGER"
        elif len(recent) >= 10 and mean <= -0.70:
            self.state = "RED"
        elif len(recent) >= 8 and mean <= -0.35:
            self.state = "ORANGE"
        elif len(recent) >= 5 and mean <= -0.10:
            self.state = "YELLOW"
        else:
            self.state = "GREEN"
        return self.state

    def summary(self) -> dict[str, object]:
        recent = self.outcomes[-20:]
        return {
            "state": self.state,
            "closed_positions": len(self.outcomes),
            "recent_expectancy": float(np.mean(recent)) if recent else 0.0,
            "invalid_lifecycles": int(self.invalid_lifecycles),
        }


class ContinuousModuleRegistry:
    """Persistent V24 specialist lifecycle, stored directly in policy state."""

    _ROLE_BY_CAUSE = {
        "stop_loss_too_tight": ROLE_RISK_GATE_SPECIALIST,
        "take_profit_too_far": ROLE_EXIT_QUALITY_SPECIALIST,
        "take_profit_too_close": ROLE_EXIT_QUALITY_SPECIALIST,
        "entry_too_early": ROLE_ENTRY_QUALITY_SPECIALIST,
        "wrong_direction_or_regime": ROLE_CRITIC_SPECIALIST,
        "false_breakout": ROLE_ENTRY_QUALITY_SPECIALIST,
        "volatility_shock": ROLE_RISK_GATE_SPECIALIST,
        "spread_or_slippage_damage": ROLE_RISK_GATE_SPECIALIST,
        "management_failure": ROLE_POSITION_MANAGEMENT_SPECIALIST,
        "memory_mismatch": ROLE_MEMORY_CRITIC_SPECIALIST,
        "reasoning_override_error": ROLE_SELECTOR_SPECIALIST,
        "risk_size_too_large": ROLE_RISK_GATE_SPECIALIST,
    }

    def __init__(self, config: ContinuousRuntimeConfig) -> None:
        self.config = config

    @staticmethod
    def _state(policy: Any) -> dict[str, object]:
        state = getattr(policy, "continuous_adaptive_state", None)
        if not isinstance(state, dict):
            state = {}
            setattr(policy, "continuous_adaptive_state", state)
        state.setdefault("module_evidence", {})
        state.setdefault("module_ids", {})
        state.setdefault("local_inhibitions", [])
        # Report fingerprints prevent repeated safe-point scans from treating
        # the same replay-validated SL/TP as fresh evidence on every bar.
        state.setdefault("processed_report_fingerprints", {})
        state.setdefault("closed_positions", 0)
        state.setdefault("overlay_revision", 0)
        return state

    def observe_report(self, policy: Any, report: CausalOutcomeReport, *, sequence: int) -> list[dict[str, object]]:
        """Consume a causal report once per evidence version.

        V23 scanned all reports at every decision.  Because inhibition was
        created inside that scan, a single validated report could increment its
        support count dozens of times without a new independent lifecycle.
        V24 makes report processing idempotent and lets a replay upgrade the
        same report exactly once.
        """
        state = self._state(policy)
        snapshot = report.snapshot
        if snapshot is None:
            return []
        fingerprint = _report_fingerprint(report)
        fingerprints = dict(state.get("processed_report_fingerprints", {}) or {})
        trace_id = str(getattr(report, "trace_id", ""))
        if fingerprints.get(trace_id) == fingerprint:
            return []

        evidence = state["module_evidence"]
        module_ids = state["module_ids"]
        events: list[dict[str, object]] = []
        vector = _vector(snapshot.canonical_context_key, 64)
        context_key = _ctx_key(vector, snapshot.context_frame_id)
        validated = bool(report.coverage.get("counterfactual_replay_validated", False))
        closed_positions = int(state.get("closed_positions", 0) or 0)

        for hypothesis in report.hypotheses:
            if hypothesis.status not in {"replay_validated", "validated"}:
                continue
            if float(hypothesis.confidence) < self.config.min_causal_confidence:
                continue
            key = f"{context_key}|{hypothesis.cause_id}"
            record = evidence.setdefault(key, {
                "context_key": context_key,
                "context_vector": vector.tolist(),
                "cause": hypothesis.cause_id,
                "trace_ids": [],
                "verified_trace_ids": [],
                "confidence": 0.0,
                "utility": 0.0,
                "last_sequence": 0,
            })
            trace_ids = list(record.get("trace_ids", []) or [])
            new_trace = trace_id not in trace_ids
            if new_trace:
                trace_ids.append(trace_id)
            record["trace_ids"] = trace_ids[-256:]
            verified = list(record.get("verified_trace_ids", []) or [])
            new_verified = bool(validated and trace_id not in verified)
            if new_verified:
                verified.append(trace_id)
            record["verified_trace_ids"] = verified[-256:]
            record["confidence"] = max(float(record.get("confidence", 0.0)), float(hypothesis.confidence))
            # Utility moves only when a new independent report or an actual
            # replay upgrade arrives.  Re-reading the journal has no effect.
            if new_trace or new_verified:
                support = 1.0 if new_verified else 0.0
                record["utility"] = float(0.85 * float(record.get("utility", 0.0)) + 0.15 * support)
                record["last_sequence"] = int(sequence)
            independent = len(set(record["trace_ids"]))
            verified_count = len(set(record.get("verified_trace_ids", [])))

            module_id = str(module_ids.get(key, "") or "")
            if (
                self.config.enable_persistent_specialists
                and not module_id
                and independent >= self.config.min_independent_evidence
                and verified_count >= self.config.min_verified_replays_to_create_module
            ):
                module_id = self._create_module(policy, record, hypothesis, sequence)
                if module_id:
                    module_ids[key] = module_id
                    events.append({"type": "continuous_specialist_created", "module_id": module_id, "cause": hypothesis.cause_id, "context_key": context_key, "verified_cases": verified_count})
            if self.config.enable_persistent_specialists and module_id:
                update = self._advance_module(policy, module_id, record, hypothesis, sequence)
                if update:
                    events.append(update)
            if validated and (new_trace or new_verified):
                inhibition_event = self._maybe_create_inhibition(
                    policy,
                    snapshot,
                    hypothesis,
                    sequence,
                    record=record,
                    independent=independent,
                    verified_count=verified_count,
                    closed_positions=closed_positions,
                )
                if inhibition_event:
                    events.append(inhibition_event)

        fingerprints[trace_id] = fingerprint
        state["processed_report_fingerprints"] = dict(list(fingerprints.items())[-1024:])
        if events:
            state["overlay_revision"] = int(state.get("overlay_revision", 0) or 0) + 1
        return events

    def _create_module(self, policy: Any, record: Mapping[str, object], hypothesis: Any, sequence: int) -> str:
        graph_items = list(getattr(policy, "_graph_m1_modules", lambda: [])())
        if not graph_items:
            return ""
        _, m1 = graph_items[0]
        graph = getattr(m1, "graph", None)
        if graph is None:
            return ""
        registry = getattr(graph, "specialist_modules", {})
        if len(registry) >= self.config.max_probation_modules + self.config.max_active_modules:
            return ""
        cause = str(record.get("cause", "unknown"))
        role = self._ROLE_BY_CAUSE.get(cause, ROLE_CRITIC_SPECIALIST)
        digest = hashlib.sha1(f"{record.get('context_key')}|{cause}".encode("utf-8")).hexdigest()[:12]
        module_id = f"continuous:{role}:{digest}"
        if module_id in registry:
            return module_id
        cfg = dict(getattr(policy, "graph_supernetwork_config", {}) or {})
        factory = SpecialistModuleFactory(
            input_adapters=int(cfg.get("module_input_adapters", 3)),
            core_neurons=int(cfg.get("module_core_neurons", 4)),
            output_adapters=int(cfg.get("module_output_adapters", 2)),
            initial_gate=float(cfg.get("module_initial_gate", -2.15)),
            initial_weight=float(cfg.get("module_initial_weight", 0.055)),
            enable_bypass_interfaces=bool(cfg.get("module_enable_bypass_interfaces", True)),
        )
        try:
            build = factory.create(
                graph,
                module_id=module_id,
                role=role,
                reason=f"continuous_verified:{cause}",
                candidate_id="continuous",
                trial_id=f"continuous:{getattr(policy, 'revision', 0)}",
                context_key=str(record.get("context_key", "")),
                context_vector=_vector(record.get("context_vector", []), 64),
                causal_confidence=float(record.get("confidence", 0.0)),
                causal_validated=True,
                causal_pattern_key=cause,
            )
        except Exception:
            return ""
        module = graph.specialist_modules.get(build.module_id, {})
        module["status"] = "probation"
        module["continuous"] = True
        module["continuous_created_sequence"] = int(sequence)
        module["verified_cases"] = int(len(set(record.get("verified_trace_ids", []) or [])))
        module["independent_cases"] = int(len(set(record.get("trace_ids", []) or [])))
        graph.specialist_modules[build.module_id] = module
        return build.module_id

    def _advance_module(self, policy: Any, module_id: str, record: Mapping[str, object], hypothesis: Any, sequence: int) -> dict[str, object] | None:
        graph_items = list(getattr(policy, "_graph_m1_modules", lambda: [])())
        for _, m1 in graph_items:
            graph = getattr(m1, "graph", None)
            registry = getattr(graph, "specialist_modules", {}) if graph is not None else {}
            module = registry.get(module_id)
            if not isinstance(module, dict):
                continue
            verified = int(len(set(record.get("verified_trace_ids", []) or [])))
            utility = float(record.get("utility", 0.0) or 0.0)
            module["verified_cases"] = verified
            module["independent_cases"] = int(len(set(record.get("trace_ids", []) or [])))
            module["causal_confidence"] = max(float(module.get("causal_confidence", 0.0) or 0.0), float(record.get("confidence", 0.0) or 0.0))
            module["causal_validated"] = True
            module["utility"] = utility
            module["last_continuous_sequence"] = int(sequence)
            status = str(module.get("status", "probation"))
            node_ids = [str(value) for value in tuple(module.get("node_ids", ()) or ())]
            edge_ids = [str(value) for value in tuple(module.get("edge_ids", ()) or ())]
            if status == "probation" and verified >= self.config.min_verified_replays_to_activate_module:
                self._set_module_lifecycle(graph, node_ids, edge_ids, NEURON_LIFECYCLE_ACTIVE, gate_delta=0.35)
                module["status"] = "active"
                return {"type": "continuous_specialist_activated", "module_id": module_id, "verified_cases": verified, "utility": utility}
            if status in {"probation", "active"} and verified >= self.config.min_verified_replays_to_consolidate_module and utility > 0.20:
                self._set_module_lifecycle(graph, node_ids, edge_ids, NEURON_LIFECYCLE_CONSOLIDATED, gate_delta=0.20)
                module["status"] = "consolidated"
                return {"type": "continuous_specialist_consolidated", "module_id": module_id, "verified_cases": verified, "utility": utility}
            registry[module_id] = module
            return {"type": "continuous_specialist_evidence", "module_id": module_id, "status": str(module.get("status", "probation")), "verified_cases": verified, "utility": utility}
        return None

    @staticmethod
    def _set_module_lifecycle(graph: Any, node_ids: Iterable[str], edge_ids: Iterable[str], lifecycle: str, *, gate_delta: float) -> None:
        for node_id in node_ids:
            node = graph.neurons.get(node_id)
            if node is not None:
                node.lifecycle = lifecycle
        for edge_id in edge_ids:
            edge = graph.edges.get(edge_id)
            if edge is not None and bool(getattr(edge, "enabled", True)):
                edge.lifecycle = lifecycle
                edge.gate = float(np.clip(float(edge.gate) + gate_delta, -8.0, 1.5))

    def _maybe_create_inhibition(
        self,
        policy: Any,
        snapshot: Any,
        hypothesis: Any,
        sequence: int,
        *,
        record: Mapping[str, object],
        independent: int,
        verified_count: int,
        closed_positions: int,
    ) -> dict[str, object] | None:
        """Create a narrow, evidence-gated local inhibition.

        Execution-cost/shock hypotheses are *not* converted into persistent
        HOLD pressure: they describe transaction conditions, not the direction
        being wrong.  They remain available to risk/size logic and telemetry.
        """
        if not self.config.enable_local_inhibition:
            return None
        cause = str(hypothesis.cause_id)
        # V25: only a replay-validated *wrong direction* can suppress an entry
        # action.  Timing/false-breakout need delay/confirmation replacements,
        # not a durable BUY/SELL -> HOLD ratchet.  Execution and volatility
        # causes remain parameter/risk evidence only.
        if cause in {"entry_too_early", "false_breakout"}:
            return {
                "type": "continuous_timing_replacement_required",
                "cause": cause,
                "reason": "timing_causes_are_not_persistent_action_inhibition",
            }
        if cause != "wrong_direction_or_regime":
            return {
                "type": "continuous_inhibition_skipped_execution_context",
                "cause": cause,
                "reason": "only_verified_wrong_direction_creates_action_inhibition",
            }
        if (
            independent < self.config.min_independent_evidence_to_inhibit
            or verified_count < self.config.min_verified_replays_to_inhibit
        ):
            return {
                "type": "continuous_inhibition_deferred",
                "cause": cause,
                "independent_cases": independent,
                "verified_cases": verified_count,
                "required_independent": self.config.min_independent_evidence_to_inhibit,
                "required_verified": self.config.min_verified_replays_to_inhibit,
            }
        state = self._state(policy)
        details = dict(getattr(snapshot, "action_details", {}) or {})
        emitted = int(getattr(snapshot, "emitted_action_index", -1) or -1)
        source = str(getattr(snapshot, "action_attribution_source", "") or "")
        if emitted < 0:
            for key, candidate_source in (("filled_action", "filled"), ("emitted_action", "emitted"), ("bridge_action", "bridge")):
                try:
                    candidate = int(float(details.get(key, -1)))
                except (TypeError, ValueError):
                    continue
                if 0 <= candidate < 12:
                    emitted, source = candidate, candidate_source
                    break
        if self.config.require_execution_action_attribution and source not in {"filled", "emitted", "bridge"}:
            return {
                "type": "continuous_inhibition_deferred_action_mapping",
                "cause": cause,
                "reason": "missing_emitted_or_filled_mt5_action",
            }
        action = int(emitted)
        # A hypothesis index is not an MT5 action.  HOLD/MANAGE/CLOSE do not
        # create entry inhibition; only actual entry actions are eligible.
        if action not in {1, 2, 3, 4, 5, 6}:
            return {
                "type": "continuous_inhibition_skipped_non_entry_action",
                "cause": cause,
                "action": action,
                "action_source": source,
            }
        context = _vector(getattr(snapshot, "canonical_context_key", []), 64)
        if context.size == 0:
            return None
        key = _ctx_key(context, getattr(snapshot, "context_frame_id", ""))
        # Preserve the entry family when direction is wrong: market BUY↔SELL,
        # BUY_LIMIT↔SELL_LIMIT, BUY_STOP↔SELL_STOP.  Never default to HOLD.
        target = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5}[action]
        strength = float(np.clip(
            0.025 + 0.075 * float(hypothesis.confidence) + 0.005 * max(0, verified_count - self.config.min_verified_replays_to_inhibit),
            0.02,
            self.config.local_inhibition_max_strength,
        ))
        entries = list(state.get("local_inhibitions", []) or [])
        existing = next((item for item in entries if item.get("context_key") == key and int(item.get("action", -1)) == action), None)
        if existing is None:
            entries.append({
                "context_key": key,
                "context_vector": context.tolist(),
                "action": action,
                "replacement": target,
                "strength": strength,
                "confidence": float(hypothesis.confidence),
                "created_sequence": int(sequence),
                "last_sequence": int(sequence),
                "last_closed_positions": int(closed_positions),
                "supports": int(independent),
                "verified_cases": int(verified_count),
                "cause": cause,
                "action_source": source,
            })
        else:
            # A new independently verified case may strengthen an existing
            # inhibition, but it cannot grow because a report is re-scanned.
            existing["strength"] = float(np.clip(max(float(existing.get("strength", 0.0)), strength), 0.0, self.config.local_inhibition_max_strength))
            existing["confidence"] = max(float(existing.get("confidence", 0.0)), float(hypothesis.confidence))
            existing["last_sequence"] = int(sequence)
            existing["last_closed_positions"] = int(closed_positions)
            existing["supports"] = int(independent)
            existing["verified_cases"] = int(verified_count)
            existing["cause"] = cause
        state["local_inhibitions"] = entries[-512:]
        return {
            "type": "continuous_inhibition_created" if existing is None else "continuous_inhibition_strengthened",
            "cause": cause,
            "action": action,
            "replacement": target,
            "strength": strength,
            "independent_cases": independent,
            "verified_cases": verified_count,
        }

    def overlay_trace(self, policy: Any, trace: Any, *, sequence: int) -> Any:
        """Apply context-scoped replacement before bridge legality masks.

        Age is measured in *closed positions*, matching the configuration name
        ``local_inhibition_half_life_positions``.  V23 accidentally used bar/
        request sequence, making the same knob behave differently on M1 vs.
        sparse data.
        """
        if not self.config.enable_local_inhibition:
            return trace
        state = self._state(policy)
        context = _vector(getattr(trace, "canonical_context_key", []), 64)
        if context.size == 0:
            return trace
        outputs = _vector(getattr(trace, "selected_action", []), 64)
        if outputs.size < 12:
            return trace
        delta = np.zeros(12, dtype=float)
        applied: list[dict[str, object]] = []
        kept: list[dict[str, object]] = []
        current_closed = int(state.get("closed_positions", 0) or 0)
        for item in list(state.get("local_inhibitions", []) or []):
            age = max(0, current_closed - int(item.get("last_closed_positions", current_closed) or current_closed))
            half = max(1, int(self.config.local_inhibition_half_life_positions))
            freshness = float(0.5 ** (age / half))
            similarity = max(0.0, _cosine(context, item.get("context_vector", [])))
            if similarity < self.config.local_inhibition_min_similarity or freshness <= 0.01:
                kept.append(item)
                continue
            action = int(item.get("action", -1))
            replacement = int(item.get("replacement", 0))
            if not 0 <= action < 12 or not 0 <= replacement < 12:
                continue
            strength = float(item.get("strength", 0.0)) * float(item.get("confidence", 0.0)) * similarity * freshness
            strength = float(np.clip(strength, 0.0, self.config.local_inhibition_max_strength))
            delta[action] -= strength
            delta[replacement] += 0.85 * strength
            applied.append({"action": action, "replacement": replacement, "strength": strength, "similarity": similarity, "cause": item.get("cause", "")})
            kept.append(item)
        state["local_inhibitions"] = kept[-512:]
        if not applied:
            return trace
        delta = np.clip(delta, -self.config.max_total_action_bias, self.config.max_total_action_bias)
        updated = np.asarray(outputs, dtype=float).copy()
        updated[:12] = np.clip(updated[:12] + delta, -1.0, 1.0)
        details = dict(getattr(trace, "action_details", {}) or {})
        details["continuous_overlay"] = {
            "schema_version": CONTINUOUS_SCHEMA_VERSION,
            "action_score_delta": delta.tolist(),
            "applied": applied,
            "overlay_revision": int(state.get("overlay_revision", 0) or 0),
            "closed_positions_age_basis": current_closed,
            "min_similarity": self.config.local_inhibition_min_similarity,
        }
        try:
            return replace(trace, selected_action=updated, action_details=details)
        except (TypeError, ValueError):
            setattr(trace, "selected_action", updated)
            setattr(trace, "action_details", details)
            return trace


def migrate_trade_activity_v25(policy: Any, config: ContinuousRuntimeConfig) -> dict[str, object]:
    """Rebase global anti-entry state from V15-V24 without deleting learning.

    Older checkpoints stored rejection feedback directly in global risk-gate
    logits and contained local inhibitions created from ``selected_idx``
    (hypothesis index).  Those two items are not context-safe.  The migration
    scales the global gate back toward neutral, resets only its recurrent
    runtime state and quarantines unsafe inhibition records.  Specialist
    topology, causal reports, memories and verified evidence are preserved.
    """
    state = ContinuousModuleRegistry._state(policy)
    prior = int(state.get("trade_activity_schema_version", 0) or 0)
    if prior >= CONTINUOUS_SCHEMA_VERSION:
        return {"migrated": False, "schema_version": CONTINUOUS_SCHEMA_VERSION}
    result: dict[str, object] = {
        "migrated": bool(config.migrate_legacy_trade_activity),
        "from_schema": prior,
        "to_schema": CONTINUOUS_SCHEMA_VERSION,
        "quarantined_inhibitions": 0,
    }
    if config.migrate_legacy_trade_activity and prior < 25:
        gate = getattr(policy, "risk_cost_gate", None)
        if gate is not None:
            try:
                before = np.asarray(getattr(gate, "b_out", np.zeros(3)), dtype=float).copy()
                gate.b_out = np.asarray(gate.b_out, dtype=float) * float(config.legacy_risk_gate_rebase_factor)
                if hasattr(gate, "state"):
                    gate.state = np.zeros_like(np.asarray(gate.state, dtype=float))
                result["risk_gate_b_out_before"] = before.tolist()
                result["risk_gate_b_out_after"] = np.asarray(gate.b_out, dtype=float).tolist()
            except Exception as exc:
                result["risk_gate_rebase_error"] = type(exc).__name__
        # Legacy rejection feedback also wrote global HOLD bias directly into
        # module output biases and M1 graph output neurons.  Recenter only the
        # *excess* of HOLD above the median entry prior; retain all relative
        # BUY/SELL and parameter learning.
        hold_adjustments: list[dict[str, object]] = []
        for module in list(getattr(policy, "modules", []) or []):
            vector = np.asarray(getattr(module, "b_out", np.zeros(0)), dtype=float).reshape(-1)
            if vector.size >= 7:
                entry_center = float(np.median(vector[1:7]))
                excess = max(0.0, float(vector[0] - entry_center))
                if excess > 1e-9:
                    delta = float(config.legacy_output_hold_rebase_factor) * excess
                    vector[0] -= delta
                    module.b_out = vector
                    hold_adjustments.append({"module": str(getattr(module, "name", "")), "hold_delta": -delta})
            component = getattr(module, "m1_component", None)
            graph = getattr(component, "graph", None)
            neurons = getattr(graph, "neurons", {}) if graph is not None else {}
            output_nodes: dict[int, object] = {}
            for node in getattr(neurons, "values", lambda: [])():
                if str(getattr(node, "region", "")) != "output":
                    continue
                try:
                    index = int(str(getattr(node, "id", "")).rsplit(":", 1)[-1])
                except ValueError:
                    continue
                output_nodes[index] = node
            if all(index in output_nodes for index in range(7)):
                entry_center = float(np.median([float(output_nodes[index].bias) for index in range(1, 7)]))
                excess = max(0.0, float(output_nodes[0].bias) - entry_center)
                if excess > 1e-9:
                    delta = float(config.legacy_output_hold_rebase_factor) * excess
                    output_nodes[0].bias = float(output_nodes[0].bias) - delta
                    hold_adjustments.append({"module": str(getattr(module, "name", "")), "graph_hold_delta": -delta})
        result["legacy_hold_rebase"] = hold_adjustments
        old = list(state.get("local_inhibitions", []) or [])
        safe: list[dict[str, object]] = []
        quarantined: list[dict[str, object]] = []
        for item in old:
            source = str(item.get("action_source", "") or "")
            action = int(item.get("action", -1) or -1)
            replacement = int(item.get("replacement", -1) or -1)
            if source in {"filled", "emitted", "bridge"} and action in {1, 2, 3, 4, 5, 6} and replacement in {1, 2, 3, 4, 5, 6}:
                safe.append(dict(item))
            else:
                quarantined.append(dict(item))
        state["local_inhibitions"] = safe
        state["legacy_inhibitions_quarantined_v25"] = quarantined[-512:]
        result["quarantined_inhibitions"] = len(quarantined)
    elif config.migrate_legacy_trade_activity:
        result["legacy_hold_rebase"] = "already_migrated_v25"
    # V26 only changes the internal representation of no-command decisions.
    # It does not alter topology or causal evidence.
    state["no_command_internal_sentinel"] = -1
    state["hold_wire_noop_only"] = True
    state["trade_activity_schema_version"] = CONTINUOUS_SCHEMA_VERSION
    return result


@dataclass
class _PendingSettlement:
    lifecycle: ContinuousLifecycle
    due_sequence: int
    first_seen_sequence: int
    attempts: int = 0


class ContinuousSessionRuntime:
    """Coordinates in-run ledger tailing, causal safe-points and persistence."""

    def __init__(
        self,
        config: ContinuousRuntimeConfig,
        *,
        session_id: str | None = None,
        policy_output_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.session_id = session_id or f"continuous_{int(time.time())}"
        self.policy_output_path = Path(policy_output_path) if policy_output_path else None
        self.store = ContinuousEventStore(config.session_root, self.session_id, config.max_event_bytes)
        self.tailer = ExecutionLedgerTailer()
        self.health = ContinuousHealthMonitor()
        self.registry = ContinuousModuleRegistry(config)
        self.checkpoints = ContinuousCheckpointManager(
            self.store,
            config.checkpoint_every_closed_positions,
            config.checkpoint_every_minutes,
        )
        self.pending: dict[str, _PendingSettlement] = {}
        self.applied: set[str] = set()
        self.current_sequence = 0
        self.closed_positions = 0
        self.started = False
        self._last_entry_sequence = 0
        self._last_warning_sequence = -10**9
        self.trade_funnel: dict[str, int] = {
            "decisions": 0,
            "policy_entry_intents": 0,
            "legal_entry_intents": 0,
            "emitted_entry_intents": 0,
            "policy_holds": 0,
            "risk_gate_holds": 0,
            "overlay_affected_decisions": 0,
            "masked_entries": 0,
            "blocked_capacity": 0,
            "blocked_pending": 0,
            "bootstrap_entries": 0,
            "pending_recovery_cancels": 0,
            "acknowledged": 0,
            "filled": 0,
            "rejected": 0,
            "final_closes": 0,
        }

    @staticmethod
    def _emit_live(policy: Any, event_type: str, payload: Mapping[str, Any]) -> None:
        """Publish continuous state without taking ownership of the live writer.

        The normal policy lifecycle still emits canonical ``decision`` and
        ``lifecycle`` telemetry. These additional events make the V24
        session/checkpoint/replay state observable without mutating Java/UI
        assumptions about the core graph payload.
        """
        writer = getattr(policy, "_live_observer", None)
        if writer is None or not bool(getattr(writer, "enabled", False)):
            return
        emit = getattr(writer, "emit", None)
        if callable(emit):
            try:
                emit(str(event_type), dict(payload))
            except Exception:
                pass

    def start(self, policy: Any, *, execution_path: str | Path | None, metadata: Mapping[str, Any] | None = None) -> None:
        if self.started:
            return
        initial_equity = float(getattr(getattr(policy, "risk", None), "initial_equity", 10000.0) or 10000.0)
        self.tailer.initial_equity = max(1e-9, initial_equity)
        self.tailer.reset(execution_path)
        state = ContinuousModuleRegistry._state(policy)
        state.setdefault("session_id", self.session_id)
        state.setdefault("overlay_revision", 0)
        state["closed_positions"] = int(self.closed_positions)
        migration = migrate_trade_activity_v25(policy, self.config)
        # Bridge code uses this marker to defer all economic credit to the
        # final continuous lifecycle settlement (never both paths).
        setattr(policy, "_continuous_session_active", True)
        # Action heads read only bounded V25 trade-activity controls from this
        # typed config.  It is session-local and is overwritten on the next
        # continuous session, never serialized as a second policy runtime.
        setattr(policy, "_continuous_runtime_config", self.config)
        start_payload = {"session_id": self.session_id, "metadata": dict(metadata or {}), "config": asdict(self.config), "trade_activity_migration": migration}
        self.store.emit("continuous_session_started", event_id=_event_id("continuous_session_started", self.session_id), payload=start_payload)
        self._emit_live(policy, "continuous_session_started", start_payload)
        self.started = True

    def before_decision(self, policy: Any, request: Any, *, execution_path: str | Path | None) -> dict[str, object]:
        self.current_sequence = int(getattr(request, "sequence", self.current_sequence) or self.current_sequence)
        if not self.started:
            self.start(policy, execution_path=execution_path)
        ingested = self.tailer.poll(execution_path)
        for lifecycle in ingested:
            if not lifecycle.trace_id:
                self.health.observe(invalid=True)
                self.store.emit("continuous_lifecycle_unmapped", event_id=lifecycle.signature, sequence=self.current_sequence, payload={"position_ticket": lifecycle.position_ticket, "reason": "trace_id_missing"})
                continue
            due = max(self.current_sequence, int(lifecycle.close_sequence or self.current_sequence)) + int(self.config.post_exit_bars)
            self.pending[lifecycle.signature] = _PendingSettlement(lifecycle, due, self.current_sequence)
            detected_payload = {"position_ticket": lifecycle.position_ticket, "due_sequence": due, "realized_r": lifecycle.realized_r, "close_reason": lifecycle.close_reason, "post_exit_coverage": bool(lifecycle.post_exit_mfe_r is not None or lifecycle.returned_to_entry is not None)}
            self.store.emit("continuous_lifecycle_detected", event_id=lifecycle.signature, sequence=self.current_sequence, trace_id=lifecycle.trace_id, payload=detected_payload)
            self._emit_live(policy, "continuous_lifecycle_detected", {"trace_id": lifecycle.trace_id, **detected_payload})
        settled = self._advance_settlements(policy, force=False)
        ContinuousModuleRegistry._state(policy)["closed_positions"] = int(self.closed_positions)
        replay_summary = self._consume_replay_results(policy)
        self._harvest_reports(policy)
        if self.checkpoints.due(self.closed_positions):
            self.checkpoints.save(policy, sequence=self.current_sequence, closed_positions=self.closed_positions, reason="periodic_continuous_safe_point")
        return {"ingested": len(ingested), "settled": settled, "replay": replay_summary, "health": self.health.summary()}

    def decorate_trace(self, policy: Any, request: Any, trace: Any) -> Any:
        trace = self.registry.overlay_trace(policy, trace, sequence=int(getattr(request, "sequence", 0) or 0))
        # V26: do not register a provisional decision before the bridge has
        # decided whether an actual MT5 command exists.  In particular, a
        # no-command wire no-op (EA action=0) must never become an internal
        # HOLD snapshot that can later be treated as a neural action.
        self.store.emit(
            "continuous_decision",
            event_id=_event_id("continuous_decision", self.session_id, getattr(request, "sequence", 0), getattr(trace, "trace_id", "")),
            sequence=int(getattr(request, "sequence", 0) or 0),
            trace_id=str(getattr(trace, "trace_id", "")),
            payload={
                "overlay_revision": int(ContinuousModuleRegistry._state(policy).get("overlay_revision", 0) or 0),
                "health": self.health.summary(),
                "action_details": dict(getattr(trace, "action_details", {}) or {}),
            },
        )
        return trace

    @staticmethod
    def _entry_action(action: int) -> bool:
        return int(action) in {1, 2, 3, 4, 5, 6}

    def _funnel_summary(self) -> dict[str, object]:
        funnel = dict(self.trade_funnel)
        denom = max(1, int(funnel.get("policy_entry_intents", 0)))
        funnel["legal_from_policy_entry_ratio"] = float(funnel.get("legal_entry_intents", 0)) / denom
        funnel["emitted_from_policy_entry_ratio"] = float(funnel.get("emitted_entry_intents", 0)) / denom
        emitted = max(1, int(funnel.get("emitted_entry_intents", 0)))
        funnel["fill_from_emitted_entry_ratio"] = float(funnel.get("filled", 0)) / emitted
        funnel["sequences_since_last_emitted_entry"] = max(0, int(self.current_sequence) - int(self._last_entry_sequence))
        return funnel

    def record_bridge_feedback(self, policy: Any, request: Any, execution_details: Mapping[str, object]) -> None:
        """Record only acknowledged MT5 execution events.

        V27 makes the flat state a hard feedback boundary.  MT5 saying
        ``positions=0`` after a wire no-op/HOLD is merely ``FLAT_IDLE``.  It is
        observable telemetry, but it cannot be interpreted as a policy action,
        outcome, memory event, or learning transition.
        """
        if bool(execution_details.get("flat_idle", False)):
            self.trade_funnel["flat_idle_observations"] = int(self.trade_funnel.get("flat_idle_observations", 0)) + 1
            self.trade_funnel["flat_idle_feedback_blocked"] = int(self.trade_funnel.get("flat_idle_feedback_blocked", 0)) + 1
            self._emit_trade_funnel(policy, force=False)
            return
        for key in ("acknowledged", "filled", "rejected", "closed"):
            if bool(execution_details.get(key, False)):
                name = "final_closes" if key == "closed" else key
                self.trade_funnel[name] = int(self.trade_funnel.get(name, 0)) + 1
        if bool(execution_details.get("filled", False)):
            self._last_entry_sequence = int(getattr(request, "sequence", self.current_sequence) or self.current_sequence)
        self._emit_trade_funnel(policy, force=False)

    def record_bridge_decision(
        self,
        policy: Any,
        request: Any,
        trace: Any,
        *,
        policy_action: int,
        legal_action: int,
        emitted_action: int,
        allowed_actions: Iterable[int],
        execution_bootstrap: bool,
        starvation: Mapping[str, object] | None = None,
        max_positions: int = 1,
    ) -> None:
        """Expose where trade opportunities disappear without forcing entries."""
        self.current_sequence = int(getattr(request, "sequence", self.current_sequence) or self.current_sequence)
        # V27: only an entry intent can create an eventual position lifecycle.
        # HOLD/no-op, CANCEL, MANAGE and CLOSE must never create a fresh causal
        # snapshot simply because MT5 is currently flat.  Their acknowledgments
        # are execution telemetry; the entry trace remains the sole owner of
        # final PnL/causal credit.
        has_lifecycle_entry = self._entry_action(int(emitted_action))
        if has_lifecycle_entry:
            ledger = getattr(policy, "causal_ledger", None)
            register = getattr(ledger, "register_decision", None)
            if callable(register):
                try:
                    register(policy, trace)
                except Exception:
                    pass
            archive = getattr(policy, "_causal_trace_archive", None)
            if isinstance(archive, dict):
                archive[str(getattr(trace, "trace_id", ""))] = trace
        self.trade_funnel["decisions"] += 1
        pending = bool(str(getattr(request, "orders_pipe", "") or "").strip())
        positions = max(0, int(getattr(request, "positions", 0) or 0))
        if self._entry_action(policy_action):
            self.trade_funnel["policy_entry_intents"] += 1
        if self._entry_action(legal_action):
            self.trade_funnel["legal_entry_intents"] += 1
        if self._entry_action(emitted_action):
            self.trade_funnel["emitted_entry_intents"] += 1
            self._last_entry_sequence = self.current_sequence
        if int(policy_action) < 0:
            self.trade_funnel["no_neural_command"] = int(self.trade_funnel.get("no_neural_command", 0)) + 1
        elif int(policy_action) == 0:
            # An explicit neural HOLD is visible as a decision, but it does not
            # create an execution-feedback transition when MT5 stays flat.
            self.trade_funnel["explicit_neural_holds"] = int(self.trade_funnel.get("explicit_neural_holds", 0)) + 1
            self.trade_funnel["policy_holds"] = int(self.trade_funnel.get("policy_holds", 0)) + 1
        if self._entry_action(policy_action) and not self._entry_action(legal_action):
            self.trade_funnel["masked_entries"] += 1
            if int(max_positions) > 0 and positions >= int(max_positions):
                self.trade_funnel["blocked_capacity"] += 1
            elif pending:
                self.trade_funnel["blocked_pending"] += 1
        details = dict(getattr(trace, "action_details", {}) or {})
        overlay = dict(details.get("continuous_overlay", {}) or {})
        if list(overlay.get("applied", []) or []):
            self.trade_funnel["overlay_affected_decisions"] += 1
        risk_gate = dict(details.get("risk_gate", {}) or {})
        if int(policy_action) < 0 and float(risk_gate.get("action_dampen", 0.0) or 0.0) >= 0.02:
            self.trade_funnel["risk_gate_no_command"] += 1
        hold_reason = str(details.get("hold_reason", "") or "")
        learner_recovery = dict(details.get("learner_recovery", {}) or {})
        root_blocker = "none"
        if int(policy_action) < 0:
            if float(risk_gate.get("action_dampen", 0.0) or 0.0) >= 0.02:
                root_blocker = "risk_gate_no_command"
            elif list(overlay.get("applied", []) or []):
                root_blocker = "verified_local_direction_overlay"
            elif float(risk_gate.get("memory_hold_bias", details.get("memory_hold_bias", 0.0)) or 0.0) > 0.03:
                root_blocker = "replacement_memory_no_command"
            else:
                root_blocker = "no_neural_command"
        elif int(policy_action) == 0:
            root_blocker = hold_reason or "policy_hold"
        elif self._entry_action(policy_action) and not self._entry_action(legal_action):
            root_blocker = (
                "capacity"
                if int(max_positions) > 0 and positions >= int(max_positions)
                else ("pending_order" if pending else "legality_mask")
            )
        elif self._entry_action(emitted_action) and pending:
            root_blocker = "pending_order_live"
        if bool(execution_bootstrap) and self._entry_action(emitted_action):
            self.trade_funnel["bootstrap_entries"] += 1
        if bool(execution_bootstrap) and int(emitted_action) == 7 and pending:
            self.trade_funnel["pending_recovery_cancels"] += 1
        payload = {
            "sequence": self.current_sequence,
            "policy_action": int(policy_action),
            "legal_action": int(legal_action),
            "emitted_action": int(emitted_action),
            "wire_action": 0 if int(emitted_action) < 0 else int(emitted_action),
            "wire_noop": bool(int(emitted_action) < 0),
            "allowed_actions": sorted(int(item) for item in allowed_actions),
            "positions": positions,
            "pending_order": pending,
            "max_positions": int(max_positions),
            "execution_bootstrap": bool(execution_bootstrap),
            "starvation": dict(starvation or {}),
            "risk_gate": risk_gate,
            "overlay": overlay,
            "memory_hold_suppressed_v25": float(risk_gate.get("memory_hold_recenter", 0.0) or 0.0),
            "hold_reason": hold_reason,
            "learner_recovery": learner_recovery,
            "root_blocker": root_blocker,
            "funnel": self._funnel_summary(),
        }
        self.store.emit(
            "continuous_trade_decision",
            event_id=_event_id("continuous_trade_decision", self.session_id, self.current_sequence, getattr(trace, "trace_id", "")),
            sequence=self.current_sequence,
            trace_id=str(getattr(trace, "trace_id", "")),
            payload=payload,
        )
        self._emit_trade_funnel(policy, force=False)
        if (
            positions == 0
            and not pending
            and self.current_sequence - self._last_entry_sequence >= self.config.trade_activity_warning_after_decisions
            and self.current_sequence - self._last_warning_sequence >= self.config.trade_activity_warning_after_decisions
        ):
            self._last_warning_sequence = self.current_sequence
            warning = {
                "sequence": self.current_sequence,
                "reason": "no_emitted_entry_intent_with_free_capacity",
                "threshold": self.config.trade_activity_warning_after_decisions,
                "funnel": self._funnel_summary(),
            }
            self.store.emit("continuous_trade_activity_warning", sequence=self.current_sequence, payload=warning)
            self._emit_live(policy, "continuous_trade_activity_warning", warning)

    def _emit_trade_funnel(self, policy: Any, *, force: bool) -> None:
        every = max(1, int(self.config.trade_funnel_emit_every_decisions))
        if not force and self.current_sequence % every != 0:
            return
        payload = {"sequence": self.current_sequence, "funnel": self._funnel_summary()}
        self.store.emit("continuous_trade_funnel", sequence=self.current_sequence, payload=payload)
        self._emit_live(policy, "continuous_trade_funnel", payload)

    def _advance_settlements(self, policy: Any, *, force: bool) -> int:
        settled = 0
        for signature, pending in list(self.pending.items()):
            ready = force or self.current_sequence >= pending.due_sequence
            if not ready:
                continue
            lifecycle = pending.lifecycle
            if signature in self.applied:
                self.pending.pop(signature, None)
                continue
            if force and self.current_sequence < pending.due_sequence:
                lifecycle.metadata = {**dict(lifecycle.metadata), "post_exit_status": "insufficient_horizon_at_session_end"}
            elif lifecycle.post_exit_mfe_r is None and lifecycle.returned_to_entry is None:
                lifecycle.metadata = {**dict(lifecycle.metadata), "post_exit_status": "sequence_horizon_elapsed_but_ea_path_missing"}
            apply = getattr(policy, "apply_execution_lifecycle_credit", None)
            credit = 0
            if callable(apply):
                try:
                    credit = int(apply(
                        lifecycle.trace_id,
                        float(lifecycle.realized_r),
                        close_reason=lifecycle.close_reason,
                        duration_bars=lifecycle.duration_bars,
                        min_holding_bars=3,
                        lifecycle=lifecycle,
                    ))
                except TypeError:
                    credit = int(apply(
                        lifecycle.trace_id,
                        float(lifecycle.realized_r),
                        close_reason=lifecycle.close_reason,
                        duration_bars=lifecycle.duration_bars,
                        min_holding_bars=3,
                    ))
            if credit <= 0 and pending.attempts < self.config.max_lifecycle_wait_sequences and not force:
                pending.attempts += 1
                pending.due_sequence = self.current_sequence + 1
                continue
            self.applied.add(signature)
            self.pending.pop(signature, None)
            self.closed_positions += 1
            ContinuousModuleRegistry._state(policy)["closed_positions"] = int(self.closed_positions)
            health_state = self.health.observe(float(lifecycle.realized_r), invalid=not bool(lifecycle.trace_id))
            applied_payload = {
                "position_ticket": lifecycle.position_ticket,
                "credit": credit,
                "realized_r": lifecycle.realized_r,
                "exit_mechanism": lifecycle.exit_mechanism,
                "close_reason": lifecycle.close_reason,
                "post_exit_status": lifecycle.metadata.get("post_exit_status", "provided_by_ea"),
                "health": health_state,
            }
            self.store.emit(
                "continuous_lifecycle_applied",
                event_id=signature,
                sequence=self.current_sequence,
                trace_id=lifecycle.trace_id,
                payload=applied_payload,
            )
            self._emit_live(policy, "continuous_lifecycle_applied", {"trace_id": lifecycle.trace_id, **applied_payload})
            settled += 1
        return settled

    def _consume_replay_results(self, policy: Any) -> dict[str, object]:
        process = getattr(policy, "process_causal_replay_results", None)
        if not callable(process):
            return {}
        try:
            summary = dict(process() or {})
        except Exception as exc:
            self.store.emit("continuous_replay_error", sequence=self.current_sequence, payload={"error": str(exc)})
            return {"error": str(exc)}
        if any(int(value or 0) for value in summary.values() if isinstance(value, (int, float))):
            self.store.emit("continuous_replay_consumed", sequence=self.current_sequence, payload=summary)
        return summary

    def _harvest_reports(self, policy: Any) -> None:
        ledger = getattr(policy, "causal_ledger", None)
        reports = getattr(ledger, "reports", None)
        if not isinstance(reports, dict):
            return
        for report in list(reports.values()):
            if not isinstance(report, CausalOutcomeReport):
                continue
            events = self.registry.observe_report(policy, report, sequence=self.current_sequence)
            for event in events:
                event_type = str(event.get("type", "continuous_module_event"))
                self.store.emit(event_type, sequence=self.current_sequence, trace_id=report.trace_id, payload=event)
                self._emit_live(policy, event_type, {"trace_id": report.trace_id, **dict(event)})
        state = ContinuousModuleRegistry._state(policy)
        health = self.health.summary()
        state["health"] = health
        state["continuous_schema_version"] = CONTINUOUS_SCHEMA_VERSION

    def finish(self, policy: Any) -> dict[str, object]:
        # One final tail/poll ensures a terminal flush does not miss a close
        # written just before Strategy Tester exits.
        for lifecycle in self.tailer.poll():
            if lifecycle.trace_id:
                self.pending[lifecycle.signature] = _PendingSettlement(lifecycle, self.current_sequence, self.current_sequence)
        settled = self._advance_settlements(policy, force=True)
        replay = self._consume_replay_results(policy)
        self._harvest_reports(policy)
        checkpoint = self.checkpoints.save(policy, sequence=self.current_sequence, closed_positions=self.closed_positions, reason="continuous_session_finish")
        summary = {
            "schema_version": CONTINUOUS_SCHEMA_VERSION,
            "session_id": self.session_id,
            "closed_positions": self.closed_positions,
            "settled_at_finish": settled,
            "pending_unsettled": len(self.pending),
            "health": self.health.summary(),
            "replay": replay,
            "checkpoint": str(checkpoint) if checkpoint else "",
            "event_path": str(self.store.event_path),
            "sqlite_path": str(self.store.db_path),
        }
        self._emit_trade_funnel(policy, force=True)
        self.store.emit("continuous_session_finished", event_id=_event_id("continuous_session_finished", self.session_id), sequence=self.current_sequence, payload=summary)
        self._emit_live(policy, "continuous_session_finished", summary)
        setattr(policy, "_continuous_session_active", False)
        return summary
