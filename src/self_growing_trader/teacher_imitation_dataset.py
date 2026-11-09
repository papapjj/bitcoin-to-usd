"""Build imitation datasets from historical Bottom/Top labels.

The EA records a causal 137-feature snapshot at ``sample_time``.  Only after
the configured right-side/horizon bars exist does it attach a future-derived
BUY/SELL/HOLD label.  Future fields remain metadata and never enter the input.
"""

from __future__ import annotations

import csv
import hashlib
import json
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from .execution_contract import DEFAULT_TRAILING_BIN_INDEX, SL_ATR_BINS, TP_RR_BINS
from .five_agent_authority import five_agent_authority
from .orders import Action


HISTORICAL_LABEL_SCHEMA_VERSION = 4
LEGACY_TEACHER_DATASET_SCHEMA_VERSION = 5
TEACHER_DATASET_SCHEMA_VERSION = 7
CONFIRMED_ACTION_ONLY_TEACHER_DATASET_SCHEMA_VERSION = 8
CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION = 9
SUPPORTED_TEACHER_DATASET_SCHEMAS = frozenset({
    LEGACY_TEACHER_DATASET_SCHEMA_VERSION,
    6,
    TEACHER_DATASET_SCHEMA_VERSION,
    CONFIRMED_ACTION_ONLY_TEACHER_DATASET_SCHEMA_VERSION,
    CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION,
})
TEACHER_FEATURE_COUNT = 137
SUPPORTED_ACTIONS = {int(Action.HOLD), int(Action.MARKET_BUY), int(Action.MARKET_SELL)}


@dataclass(frozen=True)
class TeacherDatasetBuildConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    prominence_weight_scale: float = 0.25
    net_r_weight_scale: float = 0.25
    min_quality_weight: float = 0.5
    max_quality_weight: float = 2.0
    confirmation_bars: int = 0
    max_confirmation_wait_bars: int = 5

    def validate(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction phai nam trong (0,1)")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction phai nam trong [0,1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction phai nho hon 1")
        if self.prominence_weight_scale < 0.0 or self.net_r_weight_scale < 0.0:
            raise ValueError("quality weight scale khong duoc am")
        if self.min_quality_weight <= 0.0 or self.max_quality_weight < self.min_quality_weight:
            raise ValueError("quality weight bounds khong hop le")
        if self.confirmation_bars < 0:
            raise ValueError("confirmation_bars khong duoc am")
        if self.max_confirmation_wait_bars < 0:
            raise ValueError("max_confirmation_wait_bars khong duoc am")


@dataclass(frozen=True)
class TeacherDatasetBuildResult:
    output_path: Path
    manifest_path: Path
    sample_count: int
    split_counts: dict[str, int]
    action_counts: dict[int, int]
    skipped_counts: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_dict_reader(handle: object) -> csv.DictReader:
    first_line = handle.readline()
    if not first_line:
        return csv.DictReader(())
    delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
    handle.seek(0)
    return csv.DictReader(handle, delimiter=delimiter)


def _parse_pipe_floats(value: str) -> np.ndarray:
    parts = str(value or "").split("|")
    if len(parts) != TEACHER_FEATURE_COUNT:
        raise ValueError(f"inputs_pipe co {len(parts)} gia tri, can {TEACHER_FEATURE_COUNT}")
    result = np.asarray([float(part) for part in parts], dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("inputs_pipe chua NaN/Inf")
    return result


def _parse_time(value: str) -> datetime:
    text = str(value or "").strip()
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"timestamp khong hop le: {text!r}")


def _nearest_bin_index(value: float, bins: tuple[float, ...]) -> int:
    values = np.asarray(bins, dtype=np.float64)
    return int(np.argmin(np.abs(values - float(value))))


def _decimal_quantum(value: str) -> float:
    """Return the serialization quantum of a decimal field.

    MQL writes price columns at the symbol's decimal precision.  Gross, spread
    and net PnL are rounded independently, so the exact identity can differ by
    up to half a quantum for each serialized operand.
    """
    try:
        decimal = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return 0.0
    exponent = decimal.as_tuple().exponent
    return float(Decimal(1).scaleb(exponent)) if exponent < 0 else 1.0


def _serialized_price_identity_close(gross: str, net: str, spread: str) -> bool:
    gross_value = float(gross or 0.0)
    net_value = float(net or 0.0)
    spread_value = float(spread or 0.0)
    rounding_bound = 0.5 * sum(
        _decimal_quantum(value) for value in (gross, net, spread)
    )
    residual = abs((gross_value - net_value) - spread_value)
    floating_bound = 1e-5 + 1e-5 * abs(spread_value)
    return residual <= rounding_bound + floating_bound


def _chronological_splits(
    rows: list[dict[str, object]], train_fraction: float, validation_fraction: float
) -> np.ndarray:
    unique_times = sorted({row["sample_time"] for row in rows})
    count = len(unique_times)
    train_end = max(1, int(np.floor(count * train_fraction))) if count else 0
    validation_end = min(count, max(train_end, int(np.floor(count * (train_fraction + validation_fraction)))))
    split_by_time: dict[datetime, str] = {}
    for index, value in enumerate(unique_times):
        split_by_time[value] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return np.asarray([split_by_time[row["sample_time"]] for row in rows], dtype="U10")


def _quality_weight(
    *,
    prominence_atr: float,
    informative_r: float,
    config: TeacherDatasetBuildConfig,
) -> float:
    return float(np.clip(
        1.0 + config.prominence_weight_scale * max(0.0, float(prominence_atr))
        + config.net_r_weight_scale * max(0.0, float(informative_r)),
        config.min_quality_weight,
        config.max_quality_weight,
    ))


def _reset_pivot_row_to_regular_hold(row: dict[str, object]) -> None:
    row.update({
        "action": int(Action.HOLD),
        "candidate_action": int(Action.HOLD),
        "teacher_trade_accepted": False,
        "quality_weight": 1.0,
        "confidence": 1.0,
        "prominence_atr": 0.0,
        "opposite_pivot_found": False,
        "opposite_pivot_time": row["sample_time"],
        "opposite_pivot_kind": "none",
        "opposite_pivot_price": 0.0,
        "entry_price": 0.0,
        "stop_loss_price": 0.0,
        "opposite_exit_price": 0.0,
        "exit_price": 0.0,
        "exit_reason": "NO_TRADE",
        "exit_bar_offset": 0,
        "ambiguous_hit": 0,
        "teacher_gross_r": 0.0,
        "teacher_gross_pnl_price": 0.0,
        "teacher_spread_cost_price": 0.0,
        "teacher_net_pnl_price": 0.0,
        "teacher_spread_cost_r": 0.0,
        "teacher_net_r": 0.0,
        "teacher_mfe_r": 0.0,
        "teacher_mae_r": 0.0,
        "geometry_target_valid": False,
        "teacher_delay_index": 0,
        "teacher_sl_atr": 0.0,
        "teacher_sl_index": 0,
        "teacher_tp_rr": 0.0,
        "teacher_tp_index": 0,
        "teacher_trailing_index": DEFAULT_TRAILING_BIN_INDEX,
        "label_origin_time": row["sample_time"],
        "confirmation_delay_minutes": 0,
    })


def _assign_confirmed_pivot_label(
    target: dict[str, object],
    source: dict[str, object],
    *,
    config: TeacherDatasetBuildConfig,
) -> str:
    """Move one retrospective pivot label to its first observable confirmation row."""

    action = int(source["candidate_action"])
    target_time = target["sample_time"]
    source_time = source["sample_time"]
    delay_minutes = max(0, int((target_time - source_time).total_seconds() // 60))
    target.update({
        "candidate_action": action,
        "confidence": source["confidence"],
        "prominence_atr": source["prominence_atr"],
        "future_high_return_atr": source["future_high_return_atr"],
        "future_low_return_atr": source["future_low_return_atr"],
        "future_close_return_atr": source["future_close_return_atr"],
        "pivot_stop_buffer_atr": source["pivot_stop_buffer_atr"],
        "pivot_price": source["pivot_price"],
        "opposite_pivot_found": source["opposite_pivot_found"],
        "opposite_pivot_time": source["opposite_pivot_time"],
        "opposite_pivot_kind": source["opposite_pivot_kind"],
        "opposite_pivot_price": source["opposite_pivot_price"],
        "stop_loss_price": source["stop_loss_price"],
        "opposite_exit_price": source["opposite_exit_price"],
        "exit_price": source["exit_price"],
        "exit_reason": source["exit_reason"],
        "exit_bar_offset": max(0, int(source["exit_bar_offset"]) - config.confirmation_bars),
        "ambiguous_hit": source["ambiguous_hit"],
        "teacher_mfe_r": 0.0,
        "teacher_mae_r": 0.0,
        "geometry_target_valid": False,
        "teacher_delay_index": 0,
        "teacher_sl_atr": 0.0,
        "teacher_sl_index": 0,
        "teacher_tp_rr": 0.0,
        "teacher_tp_index": 0,
        "teacher_trailing_index": DEFAULT_TRAILING_BIN_INDEX,
        "label_finalized_time": source["label_finalized_time"],
        "label_origin_time": source_time,
        "confirmation_delay_minutes": delay_minutes,
    })

    # A rejected pivot remains a hard-negative HOLD, but it is now attached to
    # the confirmation observation instead of the unknowable pivot observation.
    if not bool(source["teacher_trade_accepted"]):
        informative_r = abs(min(0.0, float(source["teacher_net_r"])))
        target.update({
            "action": int(Action.HOLD),
            "teacher_trade_accepted": False,
            "quality_weight": _quality_weight(
                prominence_atr=float(source["prominence_atr"]),
                informative_r=informative_r,
                config=config,
            ),
            "entry_price": source["entry_price"],
            "teacher_gross_r": source["teacher_gross_r"],
            "teacher_gross_pnl_price": source["teacher_gross_pnl_price"],
            "teacher_spread_cost_price": source["teacher_spread_cost_price"],
            "teacher_net_pnl_price": source["teacher_net_pnl_price"],
            "teacher_spread_cost_r": source["teacher_spread_cost_r"],
            "teacher_net_r": source["teacher_net_r"],
        })
        return "shifted_rejected_pivot"

    entry_bid = float(target["entry_bid"])
    entry_ask = float(target["entry_ask"])
    stop_loss = float(source["stop_loss_price"])
    opposite_pivot = float(source["opposite_pivot_price"])
    opposite_exit = float(source["opposite_exit_price"])
    if action == int(Action.MARKET_BUY):
        entry_price = entry_ask
        initial_risk = entry_price - stop_loss
        gross_pnl = opposite_pivot - entry_bid
        net_pnl = opposite_exit - entry_price
    else:
        entry_price = entry_bid
        initial_risk = stop_loss - entry_price
        gross_pnl = entry_bid - opposite_pivot
        net_pnl = entry_price - opposite_exit
    spread_cost = gross_pnl - net_pnl
    finite = np.all(np.isfinite(np.asarray([
        entry_price, initial_risk, gross_pnl, spread_cost, net_pnl,
    ], dtype=np.float64)))
    accepted = bool(
        finite
        and initial_risk > 0.0
        and spread_cost >= -1e-8
        and net_pnl > 0.0
        and bool(source["opposite_pivot_found"])
        and target_time < source["opposite_pivot_time"]
    )
    gross_r = gross_pnl / initial_risk if finite and initial_risk > 0.0 else 0.0
    spread_r = max(0.0, spread_cost) / initial_risk if finite and initial_risk > 0.0 else 0.0
    net_r = net_pnl / initial_risk if finite and initial_risk > 0.0 else 0.0
    geometry_target_valid = False
    teacher_sl_atr = 0.0
    teacher_tp_rr = 0.0
    teacher_sl_index = 0
    teacher_tp_index = 0
    if accepted:
        pivot_buffer_price = abs(float(source["pivot_price"]) - stop_loss)
        pivot_stop_buffer_atr = float(source["pivot_stop_buffer_atr"])
        if pivot_stop_buffer_atr > 0.0 and pivot_buffer_price > 0.0:
            atr_price = pivot_buffer_price / pivot_stop_buffer_atr
            teacher_sl_atr = initial_risk / atr_price
            # TP is the executable opposite structural pivot.  The continuous
            # RR preserves that exact price; the categorical index is only the
            # nearest representation available in the 20-output contract.
            teacher_tp_rr = net_r
            geometry_target_valid = bool(
                np.isfinite(teacher_sl_atr)
                and np.isfinite(teacher_tp_rr)
                and teacher_sl_atr > 0.0
                and teacher_tp_rr > 0.0
            )
            if geometry_target_valid:
                teacher_sl_index = _nearest_bin_index(teacher_sl_atr, SL_ATR_BINS)
                teacher_tp_index = _nearest_bin_index(teacher_tp_rr, TP_RR_BINS)
    target.update({
        "action": action if accepted else int(Action.HOLD),
        "teacher_trade_accepted": accepted,
        "entry_price": entry_price if finite else 0.0,
        "teacher_gross_r": gross_r,
        "teacher_gross_pnl_price": gross_pnl if finite else 0.0,
        "teacher_spread_cost_price": max(0.0, spread_cost) if finite else 0.0,
        "teacher_net_pnl_price": net_pnl if finite else 0.0,
        "teacher_spread_cost_r": spread_r,
        "teacher_net_r": net_r,
        "geometry_target_valid": geometry_target_valid,
        "teacher_sl_atr": teacher_sl_atr,
        "teacher_sl_index": teacher_sl_index,
        "teacher_tp_rr": teacher_tp_rr,
        "teacher_tp_index": teacher_tp_index,
        "quality_weight": _quality_weight(
            prominence_atr=float(source["prominence_atr"]),
            informative_r=max(0.0, net_r) if accepted else abs(min(0.0, net_r)),
            config=config,
        ),
    })
    return "shifted_positive" if accepted else "shifted_nonpositive_after_spread"


def _apply_confirmation_contract(
    samples: list[dict[str, object]],
    config: TeacherDatasetBuildConfig,
) -> dict[str, object]:
    if config.confirmation_bars <= 0:
        return {"enabled": False, "confirmation_bars": 0}

    original = [dict(row) for row in samples]
    pivot_indices = [
        index for index, row in enumerate(original)
        if int(row["candidate_action"]) != int(Action.HOLD)
    ]
    per_agent: dict[int, list[tuple[datetime, int]]] = {}
    for index, row in enumerate(original):
        per_agent.setdefault(int(row["agent_id"]), []).append((row["sample_time"], index))
    per_agent_times = {
        agent: [value[0] for value in values] for agent, values in per_agent.items()
    }
    assigned_targets: set[int] = set()
    audit: Counter[str] = Counter()
    delay_histogram: Counter[str] = Counter()
    audit["original_pivots"] = len(pivot_indices)
    audit["original_positive_actions"] = sum(
        bool(original[index]["teacher_trade_accepted"]) for index in pivot_indices
    )

    for source_index in pivot_indices:
        source = original[source_index]
        agent = int(source["agent_id"])
        minimum_time = source["sample_time"] + timedelta(minutes=config.confirmation_bars)
        maximum_time = minimum_time + timedelta(minutes=config.max_confirmation_wait_bars)
        values = per_agent[agent]
        times = per_agent_times[agent]
        cursor = bisect_left(times, minimum_time)
        target_index: int | None = None
        while cursor < len(values) and values[cursor][0] <= maximum_time:
            candidate_index = values[cursor][1]
            candidate = original[candidate_index]
            if (
                candidate_index not in assigned_targets
                and int(candidate["candidate_action"]) == int(Action.HOLD)
                and int(candidate["action"]) == int(Action.HOLD)
            ):
                target_index = candidate_index
                break
            cursor += 1
        if target_index is None:
            audit["missing_confirmation_observation"] += 1
            continue
        outcome = _assign_confirmed_pivot_label(
            samples[target_index], source, config=config,
        )
        assigned_targets.add(target_index)
        audit[outcome] += 1
        delay_histogram[str(samples[target_index]["confirmation_delay_minutes"])] += 1

    for source_index in pivot_indices:
        _reset_pivot_row_to_regular_hold(samples[source_index])
    audit["assigned_confirmation_rows"] = len(assigned_targets)
    audit["final_positive_actions"] = sum(
        int(row["action"]) != int(Action.HOLD) for row in samples
    )
    return {
        "enabled": True,
        "confirmation_bars": config.confirmation_bars,
        "max_confirmation_wait_bars": config.max_confirmation_wait_bars,
        **dict(sorted(audit.items())),
        "confirmation_delay_minutes": dict(sorted(
            delay_histogram.items(), key=lambda item: int(item[0])
        )),
    }


def build_teacher_imitation_dataset(
    historical_label_csv: str | Path,
    output_npz: str | Path,
    *,
    config: TeacherDatasetBuildConfig | None = None,
) -> TeacherDatasetBuildResult:
    cfg = config or TeacherDatasetBuildConfig()
    cfg.validate()
    label_path = Path(historical_label_csv).resolve()
    output_path = Path(output_npz).resolve()
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    skipped: Counter[str] = Counter()
    samples: list[dict[str, object]] = []

    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = _csv_dict_reader(handle)
        required = {
            "schema_version", "sample_id", "sample_time", "label_finalized_time",
            "agent_id", "action", "candidate_action", "teacher_trade_accepted",
            "is_pivot", "pivot_kind", "pivot_price", "pivot_prominence_atr", "causal_input",
            "future_derived_label", "future_high_return_atr", "future_low_return_atr",
            "future_close_return_atr", "pivot_stop_buffer_atr", "opposite_pivot_found",
            "opposite_pivot_time", "opposite_pivot_kind", "opposite_pivot_price",
            "entry_bid", "entry_ask",
            "entry_price", "entry_spread_price", "initial_risk_price", "stop_loss_price",
            "opposite_exit_price", "exit_time", "exit_bar_offset", "exit_price",
            "exit_spread_price", "exit_reason", "ambiguous_hit", "teacher_gross_r",
            "teacher_gross_pnl_price", "teacher_spread_cost_price", "teacher_net_pnl_price",
            "teacher_spread_cost_r", "teacher_net_r", "teacher_mfe_r", "teacher_mae_r",
            "inputs_pipe",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"historical label CSV thieu cot: {sorted(missing)}")
        for row in reader:
            try:
                if int(float(row["schema_version"] or 0)) != HISTORICAL_LABEL_SCHEMA_VERSION:
                    skipped["schema_version"] += 1
                    continue
                if int(float(row["causal_input"] or 0)) != 1:
                    skipped["noncausal_input"] += 1
                    continue
                if int(float(row["future_derived_label"] or 0)) != 1:
                    skipped["missing_future_label_marker"] += 1
                    continue
                sample_time = _parse_time(row["sample_time"])
                finalized_time = _parse_time(row["label_finalized_time"])
                if finalized_time <= sample_time:
                    skipped["invalid_label_delay"] += 1
                    continue
                agent_id = int(row["agent_id"])
                action = int(row["action"])
                candidate_action = int(row["candidate_action"])
                if action not in SUPPORTED_ACTIONS:
                    skipped["unsupported_action"] += 1
                    continue
                if candidate_action not in SUPPORTED_ACTIONS:
                    skipped["unsupported_candidate_action"] += 1
                    continue
                if not five_agent_authority(agent_id).allows(action):
                    skipped["authority_violation"] += 1
                    continue
                if candidate_action != int(Action.HOLD) and not five_agent_authority(agent_id).allows(candidate_action):
                    skipped["candidate_authority_violation"] += 1
                    continue
                is_pivot = int(float(row["is_pivot"] or 0)) == 1
                if is_pivot != (candidate_action != int(Action.HOLD)):
                    skipped["pivot_action_mismatch"] += 1
                    continue
                trade_accepted = int(float(row["teacher_trade_accepted"] or 0)) == 1
                exit_reason = str(row["exit_reason"] or "").strip().upper()
                opposite_pivot_found = int(float(row["opposite_pivot_found"] or 0)) == 1
                teacher_net_r = float(row["teacher_net_r"] or 0.0)
                teacher_gross_r = float(row["teacher_gross_r"] or 0.0)
                spread_cost_r = float(row["teacher_spread_cost_r"] or 0.0)
                gross_pnl_price = float(row["teacher_gross_pnl_price"] or 0.0)
                spread_cost_price = float(row["teacher_spread_cost_price"] or 0.0)
                net_pnl_price = float(row["teacher_net_pnl_price"] or 0.0)
                entry_bid = float(row["entry_bid"] or 0.0)
                entry_ask = float(row["entry_ask"] or 0.0)
                entry_price = float(row["entry_price"] or 0.0)
                entry_spread_price = float(row["entry_spread_price"] or 0.0)
                initial_risk_price = float(row["initial_risk_price"] or 0.0)
                stop_loss_price = float(row["stop_loss_price"] or 0.0)
                opposite_exit_price = float(row["opposite_exit_price"] or 0.0)
                exit_price = float(row["exit_price"] or 0.0)
                exit_spread_price = float(row["exit_spread_price"] or 0.0)
                pivot_price = float(row["pivot_price"] or 0.0)
                pivot_stop_buffer_atr = float(row["pivot_stop_buffer_atr"] or 0.0)
                execution_numbers = np.asarray([
                    teacher_net_r, teacher_gross_r, spread_cost_r,
                    gross_pnl_price, spread_cost_price, net_pnl_price,
                    entry_bid, entry_ask, entry_price, entry_spread_price, initial_risk_price,
                    stop_loss_price, opposite_exit_price, exit_price, exit_spread_price,
                    pivot_price, pivot_stop_buffer_atr,
                ], dtype=np.float64)
                if not np.all(np.isfinite(execution_numbers)) or min(
                    spread_cost_r, spread_cost_price, entry_spread_price, exit_spread_price
                ) < 0.0:
                    skipped["invalid_execution_numbers"] += 1
                    continue
                if candidate_action == int(Action.HOLD):
                    if (
                        trade_accepted or action != int(Action.HOLD)
                        or exit_reason != "NO_TRADE" or opposite_pivot_found
                    ):
                        skipped["invalid_regular_hold"] += 1
                        continue
                else:
                    if initial_risk_price <= 0.0 or entry_ask < entry_bid:
                        skipped["invalid_execution_geometry"] += 1
                        continue
                    if candidate_action == int(Action.MARKET_BUY):
                        geometry_valid = (
                            np.isclose(entry_price, entry_ask, rtol=0.0, atol=1e-8)
                            and stop_loss_price < entry_price
                            and (not opposite_pivot_found or opposite_exit_price > 0.0)
                        )
                    else:
                        geometry_valid = (
                            np.isclose(entry_price, entry_bid, rtol=0.0, atol=1e-8)
                            and entry_price < stop_loss_price
                            and (not opposite_pivot_found or opposite_exit_price > 0.0)
                        )
                    if not geometry_valid or not np.isclose(
                        teacher_gross_r - teacher_net_r, spread_cost_r, rtol=1e-5, atol=1e-5
                    ) or not _serialized_price_identity_close(
                        row["teacher_gross_pnl_price"],
                        row["teacher_net_pnl_price"],
                        row["teacher_spread_cost_price"],
                    ):
                        skipped["invalid_execution_geometry"] += 1
                        continue
                    accepted_valid = (
                        trade_accepted
                        and action == candidate_action
                        and opposite_pivot_found
                        and exit_reason == "OPPOSITE_PIVOT"
                        and net_pnl_price > 0.0
                        and teacher_net_r > 0.0
                    )
                    rejected_valid = not trade_accepted and action == int(Action.HOLD)
                    if not (accepted_valid or rejected_valid):
                        skipped["invalid_teacher_acceptance"] += 1
                        continue
                geometry_target_valid = bool(
                    trade_accepted
                    and action in {int(Action.MARKET_BUY), int(Action.MARKET_SELL)}
                )
                teacher_sl_atr = 0.0
                teacher_tp_rr = 0.0
                teacher_sl_index = 0
                teacher_tp_index = 0
                if geometry_target_valid:
                    pivot_buffer_price = abs(pivot_price - stop_loss_price)
                    if pivot_stop_buffer_atr <= 0.0 or pivot_buffer_price <= 0.0:
                        skipped["invalid_geometry_target"] += 1
                        continue
                    atr_price = pivot_buffer_price / pivot_stop_buffer_atr
                    teacher_sl_atr = initial_risk_price / atr_price
                    teacher_tp_rr = teacher_net_r
                    if (
                        not np.isfinite(teacher_sl_atr)
                        or not np.isfinite(teacher_tp_rr)
                        or teacher_sl_atr <= 0.0
                        or teacher_tp_rr <= 0.0
                    ):
                        skipped["invalid_geometry_target"] += 1
                        continue
                    teacher_sl_index = _nearest_bin_index(teacher_sl_atr, SL_ATR_BINS)
                    teacher_tp_index = _nearest_bin_index(teacher_tp_rr, TP_RR_BINS)
                prominence = float(row["pivot_prominence_atr"] or 0.0)
                is_rejected_pivot = (
                    action == int(Action.HOLD)
                    and candidate_action != int(Action.HOLD)
                )
                informative_r = (
                    abs(min(0.0, teacher_net_r)) if is_rejected_pivot
                    else max(0.0, teacher_net_r)
                )
                weight = 1.0 if (
                    action == int(Action.HOLD) and not is_rejected_pivot
                ) else _quality_weight(
                    prominence_atr=prominence,
                    informative_r=informative_r,
                    config=cfg,
                )
                samples.append({
                    "sample_id": str(row["sample_id"]),
                    "sample_time": sample_time,
                    "label_finalized_time": finalized_time,
                    "agent_id": agent_id,
                    "action": action,
                    "candidate_action": candidate_action,
                    "teacher_trade_accepted": trade_accepted,
                    "observation": _parse_pipe_floats(row["inputs_pipe"]),
                    "quality_weight": weight,
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "prominence_atr": prominence,
                    "future_high_return_atr": float(row["future_high_return_atr"]),
                    "future_low_return_atr": float(row["future_low_return_atr"]),
                    "future_close_return_atr": float(row["future_close_return_atr"]),
                    "pivot_stop_buffer_atr": pivot_stop_buffer_atr,
                    "pivot_price": pivot_price,
                    "entry_bid": entry_bid,
                    "entry_ask": entry_ask,
                    "opposite_pivot_found": opposite_pivot_found,
                    "opposite_pivot_time": _parse_time(row["opposite_pivot_time"]),
                    "opposite_pivot_kind": str(row["opposite_pivot_kind"]),
                    "opposite_pivot_price": float(row["opposite_pivot_price"]),
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "opposite_exit_price": opposite_exit_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "exit_bar_offset": int(row["exit_bar_offset"]),
                    "ambiguous_hit": int(float(row["ambiguous_hit"] or 0)),
                    "teacher_gross_r": teacher_gross_r,
                    "teacher_gross_pnl_price": gross_pnl_price,
                    "teacher_spread_cost_price": spread_cost_price,
                    "teacher_net_pnl_price": net_pnl_price,
                    "teacher_spread_cost_r": spread_cost_r,
                    "teacher_net_r": teacher_net_r,
                    "teacher_mfe_r": float(row["teacher_mfe_r"]),
                    "teacher_mae_r": float(row["teacher_mae_r"]),
                    "geometry_target_valid": geometry_target_valid,
                    "teacher_delay_index": 0,
                    "teacher_sl_atr": teacher_sl_atr,
                    "teacher_sl_index": teacher_sl_index,
                    "teacher_tp_rr": teacher_tp_rr,
                    "teacher_tp_index": teacher_tp_index,
                    "teacher_trailing_index": DEFAULT_TRAILING_BIN_INDEX,
                    "label_origin_time": sample_time,
                    "confirmation_delay_minutes": 0,
                })
            except (TypeError, ValueError):
                skipped["malformed_row"] += 1

    if not samples:
        raise ValueError("khong co historical label hop le sau khi loc")
    samples = sorted(
        samples,
        key=lambda row: (row["sample_time"], int(row["agent_id"])),
    )
    confirmation_audit = _apply_confirmation_contract(samples, cfg)
    output_schema_version = (
        CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION
        if cfg.confirmation_bars > 0
        else TEACHER_DATASET_SCHEMA_VERSION
    )
    splits = _chronological_splits(samples, cfg.train_fraction, cfg.validation_fraction)
    arrays = {
        "observations": np.stack([row["observation"] for row in samples]).astype(np.float32),
        "actions": np.asarray([row["action"] for row in samples], dtype=np.int64),
        "candidate_actions": np.asarray([row["candidate_action"] for row in samples], dtype=np.int64),
        "teacher_trade_accepted": np.asarray(
            [row["teacher_trade_accepted"] for row in samples], dtype=np.bool_
        ),
        "agent_ids": np.asarray([row["agent_id"] for row in samples], dtype=np.int64),
        "quality_weights": np.asarray([row["quality_weight"] for row in samples], dtype=np.float32),
        "confidence": np.asarray([row["confidence"] for row in samples], dtype=np.float32),
        "prominence_atr": np.asarray([row["prominence_atr"] for row in samples], dtype=np.float32),
        "future_high_return_atr": np.asarray([row["future_high_return_atr"] for row in samples], dtype=np.float32),
        "future_low_return_atr": np.asarray([row["future_low_return_atr"] for row in samples], dtype=np.float32),
        "future_close_return_atr": np.asarray([row["future_close_return_atr"] for row in samples], dtype=np.float32),
        "pivot_stop_buffer_atr": np.asarray(
            [row["pivot_stop_buffer_atr"] for row in samples], dtype=np.float32
        ),
        "pivot_prices": np.asarray([row["pivot_price"] for row in samples], dtype=np.float64),
        "opposite_pivot_found": np.asarray(
            [row["opposite_pivot_found"] for row in samples], dtype=np.bool_
        ),
        "opposite_pivot_times": np.asarray(
            [row["opposite_pivot_time"].isoformat() for row in samples], dtype="U19"
        ),
        "opposite_pivot_kinds": np.asarray(
            [row["opposite_pivot_kind"] for row in samples], dtype="U32"
        ),
        "opposite_pivot_prices": np.asarray(
            [row["opposite_pivot_price"] for row in samples], dtype=np.float64
        ),
        "entry_prices": np.asarray([row["entry_price"] for row in samples], dtype=np.float64),
        "stop_loss_prices": np.asarray([row["stop_loss_price"] for row in samples], dtype=np.float64),
        "opposite_exit_prices": np.asarray(
            [row["opposite_exit_price"] for row in samples], dtype=np.float64
        ),
        "exit_prices": np.asarray([row["exit_price"] for row in samples], dtype=np.float64),
        "exit_reasons": np.asarray([row["exit_reason"] for row in samples], dtype="U24"),
        "exit_bar_offsets": np.asarray([row["exit_bar_offset"] for row in samples], dtype=np.int64),
        "ambiguous_hits": np.asarray([row["ambiguous_hit"] for row in samples], dtype=np.bool_),
        "teacher_gross_r": np.asarray([row["teacher_gross_r"] for row in samples], dtype=np.float32),
        "teacher_gross_pnl_price": np.asarray(
            [row["teacher_gross_pnl_price"] for row in samples], dtype=np.float64
        ),
        "teacher_spread_cost_price": np.asarray(
            [row["teacher_spread_cost_price"] for row in samples], dtype=np.float64
        ),
        "teacher_net_pnl_price": np.asarray(
            [row["teacher_net_pnl_price"] for row in samples], dtype=np.float64
        ),
        "teacher_spread_cost_r": np.asarray(
            [row["teacher_spread_cost_r"] for row in samples], dtype=np.float32
        ),
        "teacher_net_r": np.asarray([row["teacher_net_r"] for row in samples], dtype=np.float32),
        "teacher_mfe_r": np.asarray([row["teacher_mfe_r"] for row in samples], dtype=np.float32),
        "teacher_mae_r": np.asarray([row["teacher_mae_r"] for row in samples], dtype=np.float32),
        "geometry_target_valid": np.asarray(
            [row["geometry_target_valid"] for row in samples], dtype=np.bool_
        ),
        "teacher_delay_indices": np.asarray(
            [row["teacher_delay_index"] for row in samples], dtype=np.int64
        ),
        "teacher_sl_atr": np.asarray([row["teacher_sl_atr"] for row in samples], dtype=np.float32),
        "teacher_sl_indices": np.asarray(
            [row["teacher_sl_index"] for row in samples], dtype=np.int64
        ),
        "teacher_tp_rr": np.asarray([row["teacher_tp_rr"] for row in samples], dtype=np.float32),
        "teacher_tp_indices": np.asarray(
            [row["teacher_tp_index"] for row in samples], dtype=np.int64
        ),
        "teacher_trailing_indices": np.asarray(
            [row["teacher_trailing_index"] for row in samples], dtype=np.int64
        ),
        "sample_times": np.asarray([row["sample_time"].isoformat() for row in samples], dtype="U19"),
        "label_finalized_times": np.asarray([row["label_finalized_time"].isoformat() for row in samples], dtype="U19"),
        "label_origin_times": np.asarray(
            [row["label_origin_time"].isoformat() for row in samples], dtype="U19"
        ),
        "confirmation_delay_minutes": np.asarray(
            [row["confirmation_delay_minutes"] for row in samples], dtype=np.int64
        ),
        "sample_ids": np.asarray([row["sample_id"] for row in samples], dtype="U96"),
        "splits": splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray([output_schema_version], dtype=np.int64),
        feature_count=np.asarray([TEACHER_FEATURE_COUNT], dtype=np.int64),
        **arrays,
    )
    actions = arrays["actions"]
    action_counts = {int(k): int(v) for k, v in sorted(Counter(actions.tolist()).items())}
    split_counts = {name: int(np.count_nonzero(splits == name)) for name in ("train", "validation", "test")}
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest = {
        "schema_version": output_schema_version,
        "historical_label_schema_version": HISTORICAL_LABEL_SCHEMA_VERSION,
        "feature_count": TEACHER_FEATURE_COUNT,
        "historical_label_csv": str(label_path),
        "historical_label_sha256": _sha256(label_path),
        "output_npz": str(output_path),
        "output_npz_sha256": _sha256(output_path),
        "sample_count": len(samples),
        "unique_time_count": len({row["sample_time"] for row in samples}),
        "action_counts": action_counts,
        "split_counts": split_counts,
        "skipped_counts": dict(sorted(skipped.items())),
        "confirmation_audit": confirmation_audit,
        "config": cfg.__dict__,
        "leakage_contract": (
            "causal 137-feature input at sample_time; schema 9 moves pivot labels to the first "
            "available observation at/after right-side confirmation and recomputes positive "
            "Bid/Ask net PnL plus structural SL/opposite-pivot TP geometry from that executable "
            "row; future fields remain target metadata only"
            if output_schema_version == CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION
            else "causal 137-feature input at sample_time; future opposite-pivot label, structural "
            "SL and spread-aware Bid/Ask net PnL remain target metadata only"
        ),
        "sequence_contract": (
            "all valid chronological label rows are retained; no HOLD row is dropped before "
            "recurrent window construction"
        ),
        "geometry_target_contract": {
            "scope": "accepted BUY/SELL only",
            "sl_atr": "initial_risk_price / (abs(pivot_price-stop_loss_price) / pivot_stop_buffer_atr)",
            "tp_price": "exact executable opposite structural pivot price retained in opposite_exit_prices",
            "tp_rr": "teacher_net_r from executable confirmation entry to exact opposite pivot; no fixed-RR target",
            "tp_index": "nearest TP_RR_BINS representation only; categorical replay can approximate the exact pivot",
            "delay_index": 0,
            "trailing_index": DEFAULT_TRAILING_BIN_INDEX,
            "sl_atr_bins": list(SL_ATR_BINS),
            "tp_rr_bins": list(TP_RR_BINS),
            "valid_count": int(np.count_nonzero(arrays["geometry_target_valid"])),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return TeacherDatasetBuildResult(
        output_path=output_path,
        manifest_path=manifest_path,
        sample_count=len(samples),
        split_counts=split_counts,
        action_counts=action_counts,
        skipped_counts=dict(skipped),
    )
