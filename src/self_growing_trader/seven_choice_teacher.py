"""Build and train a frozen-trunk teacher for the seven flat-state choices.

The raw MT5 file contains only causal observations and dense completed M1 bars.
This module attaches future-derived counterfactual outcomes offline.  None of the
future columns are copied into the 137-value observation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .execution_contract import SL_ATR_BINS, TP_RR_BINS, TRAILING_ATR_BINS
from .feature_contract import feature_schema_fingerprint
from .five_shared_policy import PARAMETER_BINS, PATTERN_NODE_SIZE, SharedFiveTradingPolicy
from .five_teacher_imitation import (
    TeacherImitationData,
    extract_owner_hidden_sequences,
    shared_five_network,
)
from .orders import Action, MIN_PENDING_OFFSET_ATR


RAW_SNAPSHOT_SCHEMA_VERSION = 2
SEVEN_CHOICE_DATASET_SCHEMA_VERSION = 14
SEVEN_CHOICES = tuple(range(7))
ENTRY_CHOICES = tuple(range(1, 7))
LONG_CHOICES = (1, 3, 5)
SHORT_CHOICES = (2, 4, 6)
PENDING_CHOICES = (3, 4, 5, 6)
FEATURE_COUNT = 137


@dataclass(frozen=True)
class SevenChoiceBuildConfig:
    horizon_bars: int = 48
    pending_expiry_bars: int = 5
    # The MT5 expert starts the age limit when a pending order is filled
    # (POSITION_TIME), not when the proposal is emitted.  Keep the old mode
    # available only for inspecting legacy datasets; new live-aligned datasets
    # must opt into the fill-anchored lifecycle explicitly.
    holding_horizon_from_fill: bool = False
    # MT5 records initial risk from the actual fill price to the immutable
    # stop price.  Legacy datasets used the requested ATR distance instead.
    risk_denominator_from_fill: bool = False
    # MT5 pending orders fill at the requested order price when the quote
    # crosses it.  ``touch`` is retained only for reading legacy datasets;
    # new live-aligned datasets must use ``order``.
    pending_fill_price_mode: str = "order"
    entry_offset_cap_atr: float = 5.0
    fixed_entry_offset_atr: float = 0.75
    fixed_sl_atr: float = 1.50
    fixed_tp_rr: float = 2.00
    fixed_delay_index: int = 0
    action_history_min_count: int = 48
    # Outcome rows inside one holding horizon share the same future path.  A
    # stride of one keeps the legacy dense history; a stride equal to the
    # horizon updates the causal history only once per non-overlapping block.
    action_history_stride_bars: int = 1
    action_lcb_z: float = 1.0
    action_lcb_min_r: float = 0.05
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    max_rows: int = 0

    def validate(self) -> None:
        if self.horizon_bars < 2 or self.pending_expiry_bars < 1:
            raise ValueError("horizon_bars/pending_expiry_bars khong hop le")
        if self.pending_expiry_bars > self.horizon_bars:
            raise ValueError("pending_expiry_bars khong duoc lon hon horizon_bars")
        if self.entry_offset_cap_atr <= 0.0:
            raise ValueError("entry_offset_cap_atr phai duong")
        if not 0.0 < self.fixed_entry_offset_atr <= self.entry_offset_cap_atr:
            raise ValueError("fixed_entry_offset_atr khong hop le")
        if self.fixed_sl_atr <= 0.0 or self.fixed_tp_rr <= 0.0:
            raise ValueError("fixed_sl_atr/fixed_tp_rr phai duong")
        if self.pending_fill_price_mode not in {"order", "touch"}:
            raise ValueError("pending_fill_price_mode phai la order hoac touch")
        if not 0 <= self.fixed_delay_index < int(PARAMETER_BINS["delay"]):
            raise ValueError("fixed_delay_index khong hop le")
        if self.action_history_min_count < 2 or self.action_lcb_z < 0.0:
            raise ValueError("cau hinh lich su hanh dong khong hop le")
        if self.action_history_stride_bars < 1:
            raise ValueError("action_history_stride_bars phai >= 1")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction phai nam trong (0,1)")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction phai nam trong [0,1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("tong train/validation phai nho hon 1")
        if self.max_rows < 0:
            raise ValueError("max_rows khong duoc am")


@dataclass(frozen=True)
class SevenChoiceDataset:
    path: Path
    observations: np.ndarray
    best_actions: np.ndarray
    outcome_best_actions: np.ndarray
    agent_ids: np.ndarray
    sample_times: np.ndarray
    splits: np.ndarray
    allowed_action_mask: np.ndarray
    filled: np.ndarray
    order_prices: np.ndarray
    entry_prices: np.ndarray
    fill_wait_bars: np.ndarray
    fill_wait_ticks: np.ndarray
    placement_delay_indices: np.ndarray
    entry_offset_indices: np.ndarray
    sl_indices: np.ndarray
    tp_indices: np.ndarray
    sl_prices: np.ndarray
    tp_prices: np.ndarray
    mfe_r: np.ndarray
    mae_r: np.ndarray
    gross_r: np.ndarray
    spread_cost_r: np.ndarray
    net_r: np.ndarray
    exit_bars: np.ndarray
    exit_codes: np.ndarray
    action_scores_r: np.ndarray
    manifest: dict[str, Any]


def seven_choice_runtime_contract(
    manifest_or_data: Mapping[str, Any] | SevenChoiceDataset,
    config: Any,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare the finite counterfactual labels with the live order contract.

    The seven-choice builder closes an otherwise open position at the end of
    its finite horizon.  A live profile with another pending lifetime,
    geometry, or holding limit therefore scores a different trade.  Keep this
    check next to the dataset schema so training and replay can fail before a
    seemingly good result is interpreted as live evidence.
    """

    manifest = (
        manifest_or_data.manifest
        if isinstance(manifest_or_data, SevenChoiceDataset)
        else manifest_or_data
    )
    dataset_cfg = dict(manifest.get("config", {}) or {})
    orders = getattr(config, "orders", config)

    def _float(value: object, default: float = float("nan")) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _int(value: object, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    horizon = _int(dataset_cfg.get("horizon_bars"))
    dataset_pending = _int(dataset_cfg.get("pending_expiry_bars"))
    dataset_horizon_from_fill = bool(dataset_cfg.get("holding_horizon_from_fill", False))
    dataset_risk_from_fill = bool(dataset_cfg.get("risk_denominator_from_fill", False))
    # A pending MT5 order is filled at the requested price after the quote
    # crosses it.  The older teacher used the touching bid/ask price, which
    # gives optimistic labels and must never be treated as live-aligned.
    dataset_pending_fill_mode = str(
        dataset_cfg.get("pending_fill_price_mode", "touch")
    ).strip().lower()
    runtime_pending = _int(getattr(orders, "pending_expiry_bars", -1))
    dataset_sl = _float(dataset_cfg.get("fixed_sl_atr"))
    runtime_sl = _float(getattr(orders, "fixed_sl_atr", float("nan")))
    dataset_rr = _float(dataset_cfg.get("fixed_tp_rr"))
    runtime_tp = _float(getattr(orders, "fixed_tp_atr", float("nan")))
    runtime_rr = runtime_tp / runtime_sl if runtime_sl > 0.0 else float("nan")
    dataset_offset_cap = _float(dataset_cfg.get("entry_offset_cap_atr"))
    runtime_offset_cap = _float(getattr(orders, "entry_offset_atr", float("nan")))
    runtime_max_holding = _int(getattr(orders, "max_holding_bars", -1))
    runtime_min_rr = _float(getattr(orders, "technical_min_rr", float("nan")))

    # A dense exporter can carry the feature fingerprint of the exact values
    # used to build its 137 inputs.  When present, compare it with the live
    # configuration before allowing a head-only checkpoint to be trained.
    # Legacy manifests without this fingerprint remain inspectable, but the
    # missing fingerprint is not evidence of live parity.
    source_feature_hashes = tuple(
        str(value).strip()
        for value in (manifest.get("source_feature_schema_hashes", []) or [])
        if str(value).strip()
    )
    runtime_feature_hash = ""
    feature_config = getattr(config, "features", None)
    symbol_config = getattr(config, "symbol", None)
    if feature_config is not None and symbol_config is not None:
        try:
            runtime_feature_hash = feature_schema_fingerprint(
                feature_config,
                symbol_config,
                session_timezone=str(
                    getattr(getattr(config, "learning", None), "session_timezone", "UTC")
                ),
                spread_semantics=str(
                    getattr(
                        getattr(config, "learning", None),
                        "spread_semantics",
                        "ask_minus_bid_price",
                    )
                ),
                mt5_server_utc_offset_hours=int(
                    getattr(getattr(config, "learning", None), "mt5_server_utc_offset_hours", 0)
                ),
                input_normalization_mode=str(
                    getattr(getattr(config, "learning", None), "input_normalization_mode", "raw")
                ),
            )
        except (TypeError, ValueError):
            runtime_feature_hash = ""

    def _close(a: float, b: float) -> bool:
        return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(
            float(tolerance), float(tolerance) * max(abs(a), abs(b), 1.0)
        )

    checks: dict[str, dict[str, Any]] = {
        "pending_expiry_bars": {
            "dataset": dataset_pending,
            "runtime": runtime_pending,
            "matches": dataset_pending >= 1 and runtime_pending == dataset_pending,
        },
        "fixed_sl_atr": {
            "dataset": dataset_sl,
            "runtime": runtime_sl,
            "matches": _close(dataset_sl, runtime_sl),
        },
        "fixed_tp_rr": {
            "dataset": dataset_rr,
            "runtime": runtime_rr,
            "matches": _close(dataset_rr, runtime_rr),
        },
        "entry_offset_cap_atr": {
            "dataset": dataset_offset_cap,
            "runtime": runtime_offset_cap,
            "matches": _close(dataset_offset_cap, runtime_offset_cap),
        },
        "max_holding_bars": {
            "dataset": horizon,
            "runtime": runtime_max_holding,
            # Exit code 3 in the labels is a finite-horizon close.  Zero means
            # that live execution can keep the position open past that mark.
            "matches": (
                horizon >= 2
                and runtime_max_holding == horizon
                and dataset_horizon_from_fill
            ),
        },
        "holding_horizon_origin": {
            "dataset": "from_fill" if dataset_horizon_from_fill else "from_placement",
            # The EA uses POSITION_TIME, which is set at fill time.
            "runtime": "from_fill",
            "matches": bool(dataset_horizon_from_fill),
        },
        "risk_denominator_origin": {
            "dataset": "fill_to_sl" if dataset_risk_from_fill else "requested_at_placement",
            # The EA sizes the position from the requested order price and
            # now records that same immutable distance in its lifecycle.
            "runtime": "requested_at_placement",
            "matches": not bool(dataset_risk_from_fill),
        },
        "pending_fill_price_mode": {
            "dataset": dataset_pending_fill_mode,
            "runtime": "order",
            "matches": dataset_pending_fill_mode == "order",
        },
        "technical_min_rr": {
            "dataset": dataset_rr,
            "runtime": runtime_min_rr,
            "matches": (
                math.isfinite(dataset_rr)
                and math.isfinite(runtime_min_rr)
                and runtime_min_rr <= dataset_rr + float(tolerance)
            ),
        },
    }
    if source_feature_hashes:
        checks["feature_schema"] = {
            "dataset": list(source_feature_hashes),
            "runtime": runtime_feature_hash,
            "matches": bool(runtime_feature_hash)
            and len(source_feature_hashes) == 1
            and source_feature_hashes[0] == runtime_feature_hash,
        }
    mismatches = {
        name: values for name, values in checks.items() if not bool(values["matches"])
    }
    return {
        "schema_version": 1,
        "scope": "seven_choice_finite_horizon_fixed_geometry_v1",
        "match": not mismatches,
        "horizon_bars": horizon,
        "checks": checks,
        "mismatches": mismatches,
        "note": (
            "Nhãn đóng lệnh chưa chạm mức tại cuối số nến hữu hạn; vì vậy "
            "thời gian giữ lệnh của hồ sơ chạy phải bằng đúng độ dài này."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_time(text: str) -> datetime:
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            return datetime.strptime(str(text).strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"thoi gian khong hop le: {text!r}")


def _parse_inputs(text: str) -> np.ndarray:
    values = np.fromstring(str(text), sep="|", dtype=np.float32)
    if values.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("inputs_pipe phai co dung 137 so huu han")
    return values


def _parse_tick_pipe(text: str, *, name: str) -> np.ndarray:
    values = np.fromstring(str(text), sep="|", dtype=np.float64)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} phai co it nhat mot gia tri huu han")
    return values


def _offset_codebook(cap_atr: float) -> np.ndarray:
    bins = int(PARAMETER_BINS["entry_offset"])
    probabilities = (np.arange(bins, dtype=np.float64) + 0.5) / float(bins)
    offsets = np.maximum(
        min(MIN_PENDING_OFFSET_ATR, float(cap_atr)),
        np.abs(probabilities * 2.0 - 1.0) * float(cap_atr),
    )
    return offsets.astype(np.float32)


def _first_true(mask: np.ndarray) -> np.ndarray:
    any_hit = mask.any(axis=-1)
    return np.where(any_hit, mask.argmax(axis=-1), mask.shape[-1] + 1)


def _normalized_pending_price(
    action: int,
    raw_price: float,
    bid: float,
    ask: float,
    minimum_distance: float,
    digits: int,
) -> float:
    if action == int(Action.BUY_LIMIT):
        value = min(raw_price, ask - minimum_distance)
    elif action == int(Action.SELL_LIMIT):
        value = max(raw_price, bid + minimum_distance)
    elif action == int(Action.BUY_STOP):
        value = max(raw_price, ask + minimum_distance)
    elif action == int(Action.SELL_STOP):
        value = min(raw_price, bid - minimum_distance)
    else:
        raise ValueError("chi lenh cho moi co gia dat")
    return round(float(value), int(digits))


def _placement_on_ticks(
    action: int,
    delay: int,
    offset_index: int,
    close_reference: float,
    placement_atr: float,
    minimum_distance: float,
    digits: int,
    offsets: np.ndarray,
    tick_bid: np.ndarray,
    tick_ask: np.ndarray,
    bar_starts: np.ndarray,
    expiry_bars: int,
    pending_fill_price_mode: str = "touch",
) -> tuple[int, float, float, float, int, int] | None:
    if delay >= len(tick_bid):
        return None
    placed = int(delay)
    bid_at_place = float(tick_bid[placed])
    ask_at_place = float(tick_ask[placed])
    fill_bar_limit = min(int(expiry_bars), len(bar_starts) - 1)
    fill_stop = int(bar_starts[fill_bar_limit])
    if action == int(Action.MARKET_BUY):
        return placed, ask_at_place, ask_at_place, bid_at_place, 0, 0
    if action == int(Action.MARKET_SELL):
        return placed, bid_at_place, bid_at_place, bid_at_place, 0, 0
    offset = float(offsets[offset_index]) * float(placement_atr)
    below = action in (int(Action.BUY_LIMIT), int(Action.SELL_STOP))
    raw_price = float(close_reference) + (-offset if below else offset)
    order_price = _normalized_pending_price(
        action, raw_price, bid_at_place, ask_at_place, minimum_distance, digits,
    )
    future_bid = tick_bid[placed + 1:fill_stop]
    future_ask = tick_ask[placed + 1:fill_stop]
    if action == int(Action.BUY_LIMIT):
        hit = future_ask <= order_price
    elif action == int(Action.SELL_LIMIT):
        hit = future_bid >= order_price
    elif action == int(Action.BUY_STOP):
        hit = future_ask >= order_price
    else:
        hit = future_bid <= order_price
    if not bool(hit.any()):
        return None
    fill_pos = placed + 1 + int(hit.argmax())
    fill_bid = float(tick_bid[fill_pos])
    fill_ask = float(tick_ask[fill_pos])
    if pending_fill_price_mode == "order" and action in PENDING_CHOICES:
        # The Strategy Tester does not provide price improvement for a
        # pending order: a crossing quote fills at the requested price.
        entry = float(order_price)
        if action in LONG_CHOICES:
            entry_bid = float(order_price - (fill_ask - fill_bid))
        else:
            entry_bid = float(order_price)
    else:
        entry = fill_ask if action in LONG_CHOICES else fill_bid
        entry_bid = fill_bid
    fill_bar = int(np.searchsorted(bar_starts, fill_pos, side="right") - 1)
    return fill_pos, order_price, entry, entry_bid, fill_pos - placed, fill_bar


def _fixed_geometry_ticks(
    action: int,
    order_price: float,
    entry_price: float,
    entry_bid: float,
    geometry_atr: float,
    future_bid: np.ndarray,
    future_ask: np.ndarray,
    sl_index: int,
    tp_index: int,
    *,
    geometry_price: float | None = None,
    risk_denominator_from_fill: bool = False,
) -> dict[str, Any]:
    long_side = action in LONG_CHOICES
    # The MT5 expert sends SL/TP together with a pending order.  Keep both the
    # requested risk distance and the prices anchored to the placement ATR;
    # changing them to fill-time ATR would train a geometry that the bridge
    # never sends.  The fill price follows the selected pending-fill mode:
    # live-aligned labels use the requested order price, while legacy labels
    # retain the touching bid/ask convention.
    base_atr = max(float(geometry_atr), 1e-12)
    bids = np.asarray(future_bid, dtype=np.float64)
    asks = np.asarray(future_ask, dtype=np.float64)
    if len(bids) == 0 or bids.shape != asks.shape:
        raise ValueError("khong du duong gia sau diem khop")
    sl_values = np.asarray(SL_ATR_BINS, dtype=np.float64)
    tp_rr_values = np.asarray(TP_RR_BINS, dtype=np.float64)
    contract_distances = sl_values * base_atr
    anchor_price = float(entry_price if geometry_price is None else geometry_price)
    if long_side:
        sl_prices = anchor_price - contract_distances
        tp_prices = anchor_price + contract_distances[:, None] * tp_rr_values[None, :]
        sl_hit = bids[None, :] <= sl_prices[:, None]
        tp_hit = bids[None, None, :] >= tp_prices[:, :, None]
    else:
        sl_prices = anchor_price + contract_distances
        tp_prices = anchor_price - contract_distances[:, None] * tp_rr_values[None, :]
        sl_hit = asks[None, :] >= sl_prices[:, None]
        tp_hit = asks[None, None, :] <= tp_prices[:, :, None]
    sl_first = _first_true(sl_hit)[:, None]
    tp_first = _first_true(tp_hit)
    use_sl = sl_first <= tp_first
    exit_offsets = np.minimum(sl_first, tp_first)
    no_hit = exit_offsets > len(bids)
    exit_offsets = np.where(no_hit, len(bids) - 1, exit_offsets)
    selected_bid = bids[np.minimum(exit_offsets, len(bids) - 1)]
    selected_ask = asks[np.minimum(exit_offsets, len(asks) - 1)]
    if long_side:
        exit_prices = selected_bid
        net = exit_prices - entry_price
        gross = exit_prices - entry_bid
    else:
        exit_prices = selected_ask
        net = entry_price - exit_prices
        gross = entry_bid - selected_bid
    # R is anchored to the risk requested when the order was placed. A price
    # gap may therefore realize less than -1R, which is the intended evidence;
    # using fill-to-SL distance would approach zero after a gap through SL.
    requested_risk = contract_distances[:, None]
    if risk_denominator_from_fill:
        # The EA sizes and records the lifecycle using the distance from the
        # actual fill to the immutable SL.  A gap through a pending price can
        # therefore change R even though SL/TP remain anchored at placement.
        fill_risk = np.maximum(
            np.abs(entry_price - sl_prices)[:, None], 1e-12
        )
        denominator = fill_risk
    else:
        denominator = requested_risk
    net_r = net / denominator
    gross_r = gross / denominator
    exit_offset = int(exit_offsets[sl_index, tp_index])
    path_bid = bids[:exit_offset + 1]
    path_ask = asks[:exit_offset + 1]
    risk = float(denominator[sl_index, 0])
    if long_side:
        mfe = max(0.0, (float(path_bid.max()) - entry_price) / risk)
        mae = min(0.0, (float(path_bid.min()) - entry_price) / risk)
    else:
        mfe = max(0.0, (entry_price - float(path_ask.min())) / risk)
        mae = min(0.0, (entry_price - float(path_ask.max())) / risk)
    gross_value = float(gross_r[sl_index, tp_index])
    net_value = float(net_r[sl_index, tp_index])
    return {
        "sl_index": int(sl_index), "tp_index": int(tp_index),
        "sl_price": float(sl_prices[sl_index]), "tp_price": float(tp_prices[sl_index, tp_index]),
        "mfe_r": mfe, "mae_r": mae, "gross_r": gross_value,
        "spread_cost_r": max(0.0, gross_value - net_value), "net_r": net_value,
        "exit_ticks": exit_offset + 1,
        "exit_code": 3 if bool(no_hit[sl_index, tp_index]) else (1 if bool(use_sl[sl_index, tp_index]) else 2),
        # Kept for the causal geometry probe.  The selected fields above remain
        # the compact dataset contract; callers can inspect every SL/RR pair
        # without rerunning the tick replay.
        "all_net_r": net_r.astype(np.float32),
    }


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - float(target))))


def _causal_regimes(
    close_bid: np.ndarray,
    open_bid: np.ndarray,
    atr: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Create fixed causal regime buckets without fitted future thresholds."""
    count = len(close_bid)
    short = np.zeros(count, dtype=np.float64)
    long = np.zeros(count, dtype=np.float64)
    for index in range(count):
        short[index] = (
            close_bid[index] - close_bid[max(0, index - 8)]
        ) / max(atr[index], 1e-12)
        long[index] = (
            close_bid[index] - close_bid[max(0, index - 32)]
        ) / max(atr[index], 1e-12)
    short_bin = np.digitize(short, [-1.0, -0.25, 0.25, 1.0]).astype(np.int8)
    long_bin = np.digitize(long, [-2.0, -0.50, 0.50, 2.0]).astype(np.int8)
    candle = (close_bid - open_bid) / np.maximum(atr, 1e-12)
    candle_bin = np.digitize(candle, [-0.25, 0.25]).astype(np.int8)
    hour_bin = np.asarray([
        datetime.fromisoformat(str(value)).hour // 6 for value in times
    ], dtype=np.int8)
    return np.stack((short_bin, long_bin, candle_bin, hour_bin), axis=1)


def _causal_action_targets(
    net_r: np.ndarray,
    regimes: np.ndarray,
    *,
    horizon_bars: int,
    minimum_count: int,
    lower_bound_z: float,
    history_stride_bars: int = 1,
    minimum_lcb_r: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select actions from outcomes observable strictly before each decision.

    Statistics are updated only when an older sample's full outcome horizon has
    elapsed.  The current sample's future path can never select its own label.
    Exact regimes fall back to direction-only and then global action history.
    """
    if history_stride_bars < 1:
        raise ValueError("history_stride_bars phai >= 1")
    count = len(net_r)
    scores = np.full((count, len(SEVEN_CHOICES)), -np.inf, dtype=np.float32)
    scores[:, 0] = 0.0
    exact: defaultdict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    direction: defaultdict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    global_stats: defaultdict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    source_levels = Counter()

    def update(state: list[float], value: float) -> None:
        state[0] += 1.0
        state[1] += value
        state[2] += value * value

    def lower(state: list[float]) -> float | None:
        n, total, squared = state
        if n < minimum_count:
            return None
        mean = total / n
        variance = max(0.0, (squared - total * total / n) / max(1.0, n - 1.0))
        return mean - lower_bound_z * math.sqrt(variance) / math.sqrt(n)

    for index in range(count):
        matured = index - int(horizon_bars)
        if matured >= 0 and matured % int(history_stride_bars) == 0:
            regime = regimes[matured]
            for action in ENTRY_CHOICES:
                value = float(net_r[matured, action])
                update(exact[tuple(int(x) for x in (*regime, action))], value)
                update(direction[(int(regime[0]), int(regime[1]), action)], value)
                update(global_stats[action], value)
        regime = regimes[index]
        for action in ENTRY_CHOICES:
            candidates = (
                ("exact", exact[tuple(int(x) for x in (*regime, action))]),
                ("direction", direction[(int(regime[0]), int(regime[1]), action)]),
                ("global", global_stats[action]),
            )
            for level, state in candidates:
                value = lower(state)
                if value is not None:
                    scores[index, action] = float(value)
                    source_levels[level] += 1
                    break

    role_masks = np.zeros((2, 7), dtype=np.bool_)
    role_masks[0, [0, 1, 3, 5]] = True
    role_masks[1, [0, 2, 4, 6]] = True
    allowed = np.repeat(role_masks[None, :, :], count, axis=0).reshape(-1, 7)
    duplicated_scores = np.repeat(scores[:, None, :], 2, axis=1).reshape(-1, 7)
    legal_scores = np.where(allowed, duplicated_scores, -np.inf)
    best = legal_scores.argmax(axis=1).astype(np.int64)
    chosen = legal_scores[np.arange(len(best)), best]
    best[chosen <= float(minimum_lcb_r)] = int(Action.HOLD)
    stored_scores = np.where(np.isfinite(duplicated_scores), duplicated_scores, -5.0)
    return best, stored_scores.astype(np.float32), {
        "source_levels": dict(source_levels),
        "minimum_history_count": int(minimum_count),
        "history_stride_bars": int(history_stride_bars),
        "lower_bound_z": float(lower_bound_z),
        "minimum_lcb_r": float(minimum_lcb_r),
    }


def _outcome_best_actions(
    net_r: np.ndarray,
    *,
    positive_threshold_r: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose the best role-legal realised outcome as a training label.

    ``net_r`` is a future-derived label produced by the offline replay.  It is
    never copied into observations and must not be used by a live decision
    path.  HOLD is retained whenever every legal entry choice is at or below
    the supplied net-R threshold.  Returning a separate target prevents the
    conservative historical gate from turning every later split into HOLD.
    """
    values = np.asarray(net_r, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(SEVEN_CHOICES):
        raise ValueError("net_r phai co kich thuoc [N,7]")
    if not math.isfinite(float(positive_threshold_r)):
        raise ValueError("positive_threshold_r phai huu han")
    role_choices = {
        0: np.asarray((int(Action.HOLD), *LONG_CHOICES), dtype=np.int64),
        1: np.asarray((int(Action.HOLD), *SHORT_CHOICES), dtype=np.int64),
    }
    labels = np.zeros((len(values), 2), dtype=np.int64)
    positive_counts = {"role_0": 0, "role_1": 0}
    for role, choices in role_choices.items():
        legal = values[:, choices]
        selected_local = np.argmax(legal, axis=1)
        selected = choices[selected_local]
        selected_score = legal[np.arange(len(values)), selected_local]
        selected[selected_score <= float(positive_threshold_r)] = int(Action.HOLD)
        labels[:, role] = selected
        positive_counts[f"role_{role}"] = int(np.sum(selected > 0))
    return labels, {
        "selection_uses_current_sample_future": True,
        "used_as_model_input": False,
        "positive_threshold_r": float(positive_threshold_r),
        "positive_entry_counts_by_role": positive_counts,
        "warning": "chi dung lam nhan hoc offline; khong phai nhan nhan qua de bat PPO",
    }


def build_seven_choice_dataset(
    snapshot_file: str | Path,
    output_npz: str | Path,
    *,
    config: SevenChoiceBuildConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SevenChoiceBuildConfig()
    cfg.validate()
    source = Path(snapshot_file).resolve()
    output = Path(output_npz).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    source_feature_schema_hashes: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "schema_version", "sample_time", "open_bid", "high_bid", "low_bid",
            "close_bid", "atr", "spread_price", "point", "pending_min_distance",
            "digits", "causal_input", "decision_bar_time", "tick_count",
            "tick_time_msc_pipe", "tick_bid_pipe", "tick_ask_pipe", "inputs_pipe",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"tep snapshot thieu cot: {sorted(missing)}")
        for raw in reader:
            feature_schema_hash = str(raw.get("feature_schema_hash", "") or "").strip()
            if feature_schema_hash and len(feature_schema_hash) <= 256 and feature_schema_hash.isprintable():
                source_feature_schema_hashes.add(feature_schema_hash)
            if int(float(raw["schema_version"])) != RAW_SNAPSHOT_SCHEMA_VERSION:
                raise ValueError("schema snapshot khong duoc ho tro")
            if int(float(raw["causal_input"])) != 1:
                raise ValueError("snapshot khong duoc danh dau causal_input=1")
            tick_bid = _parse_tick_pipe(raw["tick_bid_pipe"], name="tick_bid_pipe")
            tick_ask = _parse_tick_pipe(raw["tick_ask_pipe"], name="tick_ask_pipe")
            tick_time = _parse_tick_pipe(raw["tick_time_msc_pipe"], name="tick_time_msc_pipe")
            tick_count = int(raw["tick_count"])
            if tick_bid.shape != tick_ask.shape or tick_bid.shape != tick_time.shape:
                raise ValueError("ba chuoi nhip gia phai cung kich thuoc")
            if tick_count != len(tick_bid) or np.any(tick_ask < tick_bid):
                raise ValueError("tick_count hoac gia mua/ban khong hop le")
            rows.append({
                "time": _parse_time(raw["sample_time"]),
                "decision_time": _parse_time(raw["decision_bar_time"]),
                "open": float(raw["open_bid"]), "high": float(raw["high_bid"]),
                "low": float(raw["low_bid"]), "close": float(raw["close_bid"]),
                "atr": float(raw["atr"]), "spread": float(raw["spread_price"]),
                "point": float(raw["point"]),
                "minimum_distance": float(raw["pending_min_distance"]),
                "digits": int(raw["digits"]),
                "tick_bid": tick_bid, "tick_ask": tick_ask,
                "inputs": _parse_inputs(raw["inputs_pipe"]),
            })
            if cfg.max_rows and len(rows) >= cfg.max_rows:
                break
    lookahead_bars = int(cfg.horizon_bars)
    if cfg.holding_horizon_from_fill:
        lookahead_bars += int(cfg.pending_expiry_bars)
    if len(rows) <= lookahead_bars + 32:
        raise ValueError("tep snapshot qua ngan de tao nhan phan thuc te")
    times = np.asarray([row["time"].strftime("%Y-%m-%dT%H:%M:%S") for row in rows], dtype="U19")
    if np.any(times[1:] <= times[:-1]):
        raise ValueError("snapshot phai tang nghiem ngat theo thoi gian")
    decision_times = np.asarray([
        row["decision_time"].strftime("%Y-%m-%dT%H:%M:%S") for row in rows
    ], dtype="U19")
    if np.any(decision_times[1:] <= decision_times[:-1]):
        raise ValueError("decision_bar_time phai tang nghiem ngat")
    observations = np.stack([row["inputs"] for row in rows]).astype(np.float32)
    open_bid = np.asarray([row["open"] for row in rows], dtype=np.float64)
    high_bid = np.asarray([row["high"] for row in rows], dtype=np.float64)
    low_bid = np.asarray([row["low"] for row in rows], dtype=np.float64)
    close_bid = np.asarray([row["close"] for row in rows], dtype=np.float64)
    atr = np.asarray([row["atr"] for row in rows], dtype=np.float64)
    spread = np.asarray([row["spread"] for row in rows], dtype=np.float64)
    point = np.asarray([row["point"] for row in rows], dtype=np.float64)
    minimum_distance = np.asarray([row["minimum_distance"] for row in rows], dtype=np.float64)
    digits = np.asarray([row["digits"] for row in rows], dtype=np.int64)
    if not all(np.all(np.isfinite(values)) for values in (
        open_bid, high_bid, low_bid, close_bid, atr, spread, point, minimum_distance,
    )):
        raise ValueError("OHLC/ATR/spread chua NaN/Inf")
    if (
        np.any(atr <= 0.0) or np.any(spread < 0.0) or np.any(high_bid < low_bid)
        or np.any(point <= 0.0) or np.any(minimum_distance < 2.0 * point)
        or np.any(digits < 0)
    ):
        raise ValueError("OHLC/ATR/spread khong hop le")

    tick_counts = np.asarray([len(row["tick_bid"]) for row in rows], dtype=np.int64)
    global_bar_starts = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(tick_counts)))
    all_tick_bid = np.concatenate([row["tick_bid"] for row in rows])
    all_tick_ask = np.concatenate([row["tick_ask"] for row in rows])

    usable = len(rows) - lookahead_bars
    observations = observations[:usable]
    times = times[:usable]
    shape = (usable, len(SEVEN_CHOICES))
    filled = np.zeros(shape, dtype=np.bool_)
    order_prices = np.zeros(shape, dtype=np.float32)
    entry_prices = np.zeros(shape, dtype=np.float32)
    fill_wait = np.zeros(shape, dtype=np.int16)
    fill_wait_ticks = np.zeros(shape, dtype=np.int32)
    delays = np.zeros(shape, dtype=np.int8)
    offset_indices = np.zeros(shape, dtype=np.int16)
    sl_indices = np.zeros(shape, dtype=np.int8)
    tp_indices = np.zeros(shape, dtype=np.int8)
    sl_prices = np.zeros(shape, dtype=np.float32)
    tp_prices = np.zeros(shape, dtype=np.float32)
    mfe_r = np.zeros(shape, dtype=np.float32)
    mae_r = np.zeros(shape, dtype=np.float32)
    gross_r = np.zeros(shape, dtype=np.float32)
    spread_cost_r = np.zeros(shape, dtype=np.float32)
    net_r = np.zeros(shape, dtype=np.float32)
    exit_bars = np.zeros(shape, dtype=np.int16)
    exit_codes = np.zeros(shape, dtype=np.int8)
    filled[:, 0] = True
    offsets = _offset_codebook(cfg.entry_offset_cap_atr)
    fixed_offset_index = _nearest_index(offsets, cfg.fixed_entry_offset_atr)
    fixed_sl_index = _nearest_index(np.asarray(SL_ATR_BINS), cfg.fixed_sl_atr)
    fixed_tp_index = _nearest_index(np.asarray(TP_RR_BINS), cfg.fixed_tp_rr)
    for index in range(usable):
        tick_start = int(global_bar_starts[index])
        tick_stop = int(global_bar_starts[index + lookahead_bars])
        horizon_bid = all_tick_bid[tick_start:tick_stop]
        horizon_ask = all_tick_ask[tick_start:tick_stop]
        bar_starts = global_bar_starts[index:index + lookahead_bars + 1] - tick_start
        for action in ENTRY_CHOICES:
            offset_index = fixed_offset_index if action in PENDING_CHOICES else 0
            placement = _placement_on_ticks(
                action, cfg.fixed_delay_index, offset_index,
                close_bid[index], atr[index], minimum_distance[index],
                int(digits[index]), offsets, horizon_bid, horizon_ask,
                bar_starts, cfg.pending_expiry_bars,
                cfg.pending_fill_price_mode,
            )
            if placement is None or placement[0] >= len(horizon_bid) - 1:
                continue
            fill_pos, order_price, entry_price, entry_bid, waited_ticks, fill_bar = placement
            # The placement helper returns either the side-specific touching
            # price for legacy data or the requested order price for the
            # MT5-aligned mode.
            entry_price = float(placement[2])
            entry_bid = float(placement[3])
            exit_horizon = int(cfg.horizon_bars) if cfg.holding_horizon_from_fill else len(horizon_bid) - fill_pos
            action_bid = horizon_bid[fill_pos:fill_pos + exit_horizon]
            action_ask = horizon_ask[fill_pos:fill_pos + exit_horizon]
            geometry = _fixed_geometry_ticks(
                action, order_price, entry_price, entry_bid, float(atr[index]),
                action_bid, action_ask,
                fixed_sl_index, fixed_tp_index,
                geometry_price=(order_price if action in PENDING_CHOICES else entry_price),
                risk_denominator_from_fill=cfg.risk_denominator_from_fill,
            )
            filled[index, action] = True
            order_prices[index, action] = order_price
            entry_prices[index, action] = entry_price
            fill_wait[index, action] = fill_bar
            fill_wait_ticks[index, action] = waited_ticks
            delays[index, action] = cfg.fixed_delay_index
            offset_indices[index, action] = offset_index
            sl_indices[index, action] = int(geometry["sl_index"])
            tp_indices[index, action] = int(geometry["tp_index"])
            sl_prices[index, action] = float(geometry["sl_price"])
            tp_prices[index, action] = float(geometry["tp_price"])
            mfe_r[index, action] = float(geometry["mfe_r"])
            mae_r[index, action] = float(geometry["mae_r"])
            gross_r[index, action] = float(geometry["gross_r"])
            spread_cost_r[index, action] = float(geometry["spread_cost_r"])
            net_r[index, action] = float(geometry["net_r"])
            exit_pos = fill_pos + int(geometry["exit_ticks"]) - 1
            exit_bar = int(np.searchsorted(bar_starts, exit_pos, side="right") - 1)
            exit_bars[index, action] = max(1, exit_bar - fill_bar + 1)
            exit_codes[index, action] = int(geometry["exit_code"])

    # One causal observation is replayed for the two entry-role agents. Their
    # legal choices differ, while the seven fixed-policy outcomes stay equal.
    role_masks = np.zeros((2, 7), dtype=np.bool_)
    role_masks[0, [0, 1, 3, 5]] = True
    role_masks[1, [0, 2, 4, 6]] = True
    allowed = np.repeat(role_masks[None, :, :], usable, axis=0).reshape(-1, 7)
    regimes = _causal_regimes(close_bid[:usable], open_bid[:usable], atr[:usable], times)
    best_actions, action_scores, causal_target_audit = _causal_action_targets(
        net_r, regimes,
        # A pending order is not finalized until its fill-expiry window has
        # elapsed; only then can the filled-position horizon be complete.
        # Using the shorter holding horizon here would feed an unmatured
        # pending outcome into the historical action statistics.
        horizon_bars=lookahead_bars,
        minimum_count=cfg.action_history_min_count,
        lower_bound_z=cfg.action_lcb_z,
        history_stride_bars=cfg.action_history_stride_bars,
        minimum_lcb_r=cfg.action_lcb_min_r,
    )
    outcome_best_unique, outcome_target_audit = _outcome_best_actions(net_r)
    outcome_best_actions = np.empty(usable * 2, dtype=np.int64)
    outcome_best_actions[0::2] = outcome_best_unique[:, 0]
    outcome_best_actions[1::2] = outcome_best_unique[:, 1]

    split_per_time = np.full(usable, "test", dtype="U10")
    train_end = int(math.floor(usable * cfg.train_fraction))
    validation_end = int(math.floor(usable * (cfg.train_fraction + cfg.validation_fraction)))
    # The longest future dependency is pending expiry plus the holding
    # horizon when the live contract starts ageing at fill time.
    purge = lookahead_bars if cfg.holding_horizon_from_fill else cfg.horizon_bars
    split_per_time[:max(0, train_end - purge)] = "train"
    split_per_time[train_end:max(train_end, validation_end - purge)] = "validation"
    split_per_time[validation_end:] = "test"
    # Purged boundary rows retain a dedicated name and are excluded from all training.
    split_per_time[max(0, train_end - purge):train_end] = "purged"
    split_per_time[max(train_end, validation_end - purge):validation_end] = "purged"

    def duplicate(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:, None, ...], 2, axis=1).reshape((usable * 2,) + values.shape[1:])

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray([SEVEN_CHOICE_DATASET_SCHEMA_VERSION], dtype=np.int64),
        "observations": duplicate(observations),
        "best_actions": best_actions,
        "outcome_best_actions": outcome_best_actions,
        "agent_ids": np.tile(np.asarray([0, 1], dtype=np.int64), usable),
        "sample_times": duplicate(times),
        "splits": duplicate(split_per_time),
        "allowed_action_mask": allowed,
        "counterfactual_filled": duplicate(filled),
        "counterfactual_order_prices": duplicate(order_prices),
        "counterfactual_entry_prices": duplicate(entry_prices),
        "counterfactual_fill_wait_bars": duplicate(fill_wait),
        "counterfactual_fill_wait_ticks": duplicate(fill_wait_ticks),
        "counterfactual_placement_delay_indices": duplicate(delays),
        "counterfactual_entry_offset_indices": duplicate(offset_indices),
        "counterfactual_sl_indices": duplicate(sl_indices),
        "counterfactual_tp_indices": duplicate(tp_indices),
        "counterfactual_sl_prices": duplicate(sl_prices),
        "counterfactual_tp_prices": duplicate(tp_prices),
        "counterfactual_mfe_r": duplicate(mfe_r),
        "counterfactual_mae_r": duplicate(mae_r),
        "counterfactual_gross_r": duplicate(gross_r),
        "counterfactual_spread_cost_r": duplicate(spread_cost_r),
        "counterfactual_net_r": duplicate(net_r),
        "counterfactual_exit_bars": duplicate(exit_bars),
        "counterfactual_exit_codes": duplicate(exit_codes),
        "causal_action_scores_r": action_scores,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    duplicated_net_r = duplicate(net_r)
    chosen_net_r = duplicated_net_r[np.arange(len(best_actions)), best_actions]
    chosen_scores = action_scores[np.arange(len(best_actions)), best_actions]
    entry_label_fraction = float(np.mean(best_actions > 0))
    config_manifest = dict(cfg.__dict__)
    config_manifest["fixed_entry_offset_effective_atr"] = float(offsets[fixed_offset_index])
    manifest = {
        "schema_version": SEVEN_CHOICE_DATASET_SCHEMA_VERSION,
        "source": str(source), "source_sha256": _sha256(source),
        "output": str(output), "rows": int(usable * 2), "unique_times": int(usable),
        "feature_count": FEATURE_COUNT, "choice_count": 7,
        "source_feature_schema_hashes": sorted(source_feature_schema_hashes),
        "choices": {str(value): Action(value).name for value in SEVEN_CHOICES},
        "config": config_manifest, "purge_bars": int(lookahead_bars),
        "label_horizon_bars": int(cfg.horizon_bars),
        "label_horizon_origin": (
            "from_fill" if cfg.holding_horizon_from_fill else "from_placement"
        ),
        "risk_denominator_origin": (
            "fill_to_sl" if cfg.risk_denominator_from_fill else "requested_at_placement"
        ),
        "split_counts": {
            name: int(np.sum(duplicate(split_per_time) == name))
            for name in ("train", "validation", "test", "purged")
        },
        "best_action_counts": {
            Action(value).name: int(np.sum(best_actions == value)) for value in SEVEN_CHOICES
        },
        "outcome_best_action_counts": {
            split: {
                Action(value).name: int(
                    np.sum(outcome_best_actions[duplicate(split_per_time) == split] == value)
                )
                for value in SEVEN_CHOICES
            }
            for split in ("train", "validation", "test", "purged")
        },
        "fill_rates": {
            Action(value).name: float(filled[:, value].mean()) for value in ENTRY_CHOICES
        },
        "causal_selection_audit": {
            "placement_search_uses_future_terminal_price": False,
            "geometry_search_uses_future_path": False,
            "action_search_uses_current_sample_future": False,
            "fixed_delay_index": int(cfg.fixed_delay_index),
            "fixed_entry_offset_index": int(fixed_offset_index),
            "fixed_entry_offset_atr": float(offsets[fixed_offset_index]),
            "fixed_sl_index": int(fixed_sl_index),
            "fixed_sl_atr": float(SL_ATR_BINS[fixed_sl_index]),
            "fixed_tp_index": int(fixed_tp_index),
            "fixed_tp_rr": float(TP_RR_BINS[fixed_tp_index]),
            "holding_horizon_bars": int(cfg.horizon_bars),
            "holding_horizon_origin": (
                "from_fill" if cfg.holding_horizon_from_fill else "from_placement"
            ),
            "lookahead_bars": int(lookahead_bars),
            "risk_denominator_origin": (
                "fill_to_sl" if cfg.risk_denominator_from_fill else "requested_at_placement"
            ),
            **causal_target_audit,
            "entry_label_fraction": entry_label_fraction,
            "chosen_realized_net_r_mean": float(np.mean(chosen_net_r)),
            "chosen_realized_net_r_median": float(np.median(chosen_net_r)),
            "chosen_causal_score_mean": float(np.mean(chosen_scores)),
            "production_eligible": True,
            "reason": "each decision uses only outcomes whose full horizon ended earlier",
        },
        "outcome_target_audit": {
            **outcome_target_audit,
            "counts_by_split": {
                split: int(
                    np.sum(
                        (duplicate(split_per_time) == split)
                        & (outcome_best_actions > 0)
                    )
                )
                for split in ("train", "validation", "test", "purged")
            },
            "label_only_future_fields": True,
        },
        "label_contract": {
            "inputs": "causal 137-value snapshot only",
            "labels": "future-derived seven-choice outcomes, never model input",
            "pending_fill": (
                (
                    "lenh cho khop dung gia dat nhu MT5 khi bao gia vuot qua; "
                    "SL/TP va rui ro neo theo gia cung ATR luc dat lenh; "
                    "gap-aware, bounded expiry"
                )
                if cfg.pending_fill_price_mode == "order"
                else (
                    "fill theo dung phia bid/ask tai nac khop; SL/TP va rui ro neo "
                    "theo gia cung ATR luc dat lenh; gap-aware, bounded expiry"
                )
            ),
            "placement_delay": "fixed causal index; no per-row future search",
            "intrabar": "exact bid/ask tick order",
            "cost": "entry/exit spread included in net_r",
            "selection": "best positive historical lower bound among role-legal choices",
            "outcome_training_target": (
                "best role-legal current counterfactual net-R when positive; HOLD otherwise; "
                "offline label only"
            ),
            "geometry": "fixed SL/TP/offset bins from the MT5 execution contract",
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_seven_choice_dataset(path: str | Path) -> SevenChoiceDataset:
    source = Path(path).resolve()
    manifest_path = source.with_suffix(source.suffix + ".json")
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(source if not source.is_file() else manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {
        "observations", "best_actions", "outcome_best_actions", "agent_ids", "sample_times", "splits",
        "allowed_action_mask", "counterfactual_filled", "counterfactual_order_prices",
        "counterfactual_entry_prices", "counterfactual_fill_wait_bars",
        "counterfactual_fill_wait_ticks", "counterfactual_placement_delay_indices",
        "counterfactual_entry_offset_indices", "counterfactual_sl_indices",
        "counterfactual_tp_indices", "counterfactual_sl_prices", "counterfactual_tp_prices",
        "counterfactual_mfe_r", "counterfactual_mae_r", "counterfactual_gross_r",
        "counterfactual_spread_cost_r", "counterfactual_net_r",
        "counterfactual_exit_bars", "counterfactual_exit_codes",
        "causal_action_scores_r", "schema_version",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = names - set(archive.files)
        if missing:
            raise ValueError(f"dataset bay lua chon thieu mang: {sorted(missing)}")
        if int(archive["schema_version"][0]) != SEVEN_CHOICE_DATASET_SCHEMA_VERSION:
            raise ValueError("schema dataset bay lua chon khong hop le")
        values = {name: np.array(archive[name], copy=True) for name in names - {"schema_version"}}
    count = len(values["best_actions"])
    if values["observations"].shape != (count, FEATURE_COUNT):
        raise ValueError("observations sai kich thuoc")
    for name in names:
        if name.startswith("counterfactual_") and values[name].shape != (count, 7):
            raise ValueError(f"{name} phai co kich thuoc [N,7]")
    if values["allowed_action_mask"].shape != (count, 7):
        raise ValueError("allowed_action_mask phai co kich thuoc [N,7]")
    if values["causal_action_scores_r"].shape != (count, 7):
        raise ValueError("causal_action_scores_r phai co kich thuoc [N,7]")
    if values["outcome_best_actions"].shape != (count,):
        raise ValueError("outcome_best_actions phai co kich thuoc [N]")
    if not np.all(values["allowed_action_mask"][np.arange(count), values["best_actions"]]):
        raise ValueError("best_actions vi pham quyen agent")
    if not np.all(values["allowed_action_mask"][np.arange(count), values["outcome_best_actions"]]):
        raise ValueError("outcome_best_actions vi pham quyen agent")
    return SevenChoiceDataset(
        path=source, manifest=manifest,
        observations=values["observations"].astype(np.float32),
        best_actions=values["best_actions"].astype(np.int64),
        outcome_best_actions=values["outcome_best_actions"].astype(np.int64),
        agent_ids=values["agent_ids"].astype(np.int64),
        sample_times=values["sample_times"], splits=values["splits"],
        allowed_action_mask=values["allowed_action_mask"].astype(np.bool_),
        filled=values["counterfactual_filled"].astype(np.bool_),
        order_prices=values["counterfactual_order_prices"].astype(np.float32),
        entry_prices=values["counterfactual_entry_prices"].astype(np.float32),
        fill_wait_bars=values["counterfactual_fill_wait_bars"].astype(np.int64),
        fill_wait_ticks=values["counterfactual_fill_wait_ticks"].astype(np.int64),
        placement_delay_indices=values["counterfactual_placement_delay_indices"].astype(np.int64),
        entry_offset_indices=values["counterfactual_entry_offset_indices"].astype(np.int64),
        sl_indices=values["counterfactual_sl_indices"].astype(np.int64),
        tp_indices=values["counterfactual_tp_indices"].astype(np.int64),
        sl_prices=values["counterfactual_sl_prices"].astype(np.float32),
        tp_prices=values["counterfactual_tp_prices"].astype(np.float32),
        mfe_r=values["counterfactual_mfe_r"].astype(np.float32),
        mae_r=values["counterfactual_mae_r"].astype(np.float32),
        gross_r=values["counterfactual_gross_r"].astype(np.float32),
        spread_cost_r=values["counterfactual_spread_cost_r"].astype(np.float32),
        net_r=values["counterfactual_net_r"].astype(np.float32),
        exit_bars=values["counterfactual_exit_bars"].astype(np.int64),
        exit_codes=values["counterfactual_exit_codes"].astype(np.int64),
        action_scores_r=values["causal_action_scores_r"].astype(np.float32),
    )


def _hash_except_seven_choice_heads(network: SharedFiveTradingPolicy) -> str:
    trainable = (
        "heads.action_query.", "heads.action_embedding.",
        "heads.direct_action_head.",
        "heads.direct_observation_head.",
        "heads.entry_role_calibrator.",
        "heads.parameter_heads.delay.", "heads.parameter_heads.entry_offset.",
        "heads.parameter_heads.sl.", "heads.parameter_heads.tp.",
        "heads.parameter_heads.trailing.",
        "observation_encoder.pattern_tree.action_path_head.",
        "observation_encoder.pattern_tree.action_path_residual.",
        "observation_encoder.pattern_tree.action_path_observation_adapter.",
        "heads.expected_value_hidden_residual.",
    )
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        if name.startswith(trainable):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _require_hidden_cache_metadata(
    metadata_path: Path,
    expected: dict[str, Any],
) -> None:
    if not metadata_path.is_file():
        raise RuntimeError("bo nho an tai su dung thieu dau kiem nguon")
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("dau kiem nguon bo nho an khong doc duoc") from exc
    if actual != expected:
        raise RuntimeError("bo nho an tai su dung khong khop tap du lieu hoac than mang")


def _grouped_lcb(
    values: np.ndarray,
    times: np.ndarray,
    *,
    group_width: int = 10,
) -> float | None:
    """One-sided lower bound after collapsing overlapping rows by time block."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    times = np.asarray(times).reshape(-1)
    if values.size == 0 or values.size != times.size:
        return None
    groups: defaultdict[str, list[float]] = defaultdict(list)
    for value, timestamp in zip(values.tolist(), times.tolist()):
        groups[str(timestamp)[:group_width]].append(float(value))
    means = np.asarray(
        [float(np.mean(item)) for item in groups.values() if item], dtype=np.float64
    )
    if means.size < 2:
        return None
    mean = float(means.mean())
    variance = float(means.var(ddof=1))
    return mean - 1.64 * math.sqrt(max(0.0, variance)) / math.sqrt(float(means.size))


def _entry_selection_metrics(
    data: SevenChoiceDataset,
    scores: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    indices = np.flatnonzero(mask)
    legal = np.where(data.allowed_action_mask[indices], scores[indices], -np.inf)
    legal[:, 0] = -np.inf
    selected = legal.argmax(axis=1).astype(np.int64)
    best_score = legal[np.arange(len(indices)), selected]
    selected[best_score <= float(threshold)] = int(Action.HOLD)
    active = selected > 0
    realized = data.net_r[indices, selected]
    entry_values = realized[active]
    count = len(entry_values)
    selected_filled = data.filled[indices, selected]
    selected_closed = selected_filled & (data.exit_codes[indices, selected] > 0)

    def summary(values: np.ndarray) -> tuple[float | None, float | None]:
        n = int(values.size)
        if n == 0:
            return None, None
        value = float(values.mean())
        if n < 2:
            return value, None
        lower_value = value - 1.64 * float(values.std(ddof=1)) / math.sqrt(n)
        return value, lower_value

    mean, lower = summary(entry_values)
    filled_values = entry_values[selected_filled[active]]
    closed_values = entry_values[selected_closed[active]]
    filled_mean, filled_lower = summary(filled_values)
    closed_mean, closed_lower = summary(closed_values)
    closed_block_lcb = _grouped_lcb(
        closed_values,
        data.sample_times[indices[active][selected_closed[active]]],
    )
    return selected, {
        "threshold": float(threshold),
        "entries": int(count),
        "proposed_entries": int(count),
        "filled_entries": int(filled_values.size),
        "closed_entries": int(closed_values.size),
        "pending_unfilled_entries": int(count - filled_values.size),
        "entry_rate": float(active.mean()),
        "mean_net_r": mean,
        "lcb_95_one_sided": lower,
        "filled_mean_net_r": filled_mean,
        "filled_lcb_95_one_sided_iid": filled_lower,
        "closed_mean_net_r": closed_mean,
        "closed_lcb_95_one_sided_iid": closed_lower,
        "closed_lcb_95_one_sided_block": closed_block_lcb,
        "lcb_independence_note": "chi la uoc luong IID; can bootstrap theo khoi/ngay cho cong",
        "win_rate": float((entry_values > 0.0).mean()) if count else None,
        "buy_entries": int(np.isin(selected, LONG_CHOICES).sum()),
        "sell_entries": int(np.isin(selected, SHORT_CHOICES).sum()),
        "mean_all_rows_net_r": float(realized.mean()),
    }


def _expected_value_score_matrix(
    network: SharedFiveTradingPolicy,
    data: SevenChoiceDataset,
    hidden: Any,
    roots: Any,
    *,
    selection_score_mode: str,
) -> np.ndarray:
    """Tái tạo đúng điểm sáu lựa chọn sau khi hiệu chỉnh trạng thái ẩn."""
    import torch

    if selection_score_mode not in {
        "value", "value_times_win", "filled_value", "win_probability",
    }:
        raise ValueError("selection_score_mode khong hop le")
    device = next(network.parameters()).device
    scores = np.zeros((len(data.observations), 7), dtype=np.float32)
    threshold_floor = float(
        getattr(network.heads, "seven_choice_expected_value_win_probability_floor", 0.0)
        or 0.0
    )
    residual_module = getattr(network.heads, "expected_value_hidden_residual", None)
    residual_enabled = bool(
        getattr(network.heads, "seven_choice_expected_value_hidden_residual_enabled", False)
    )
    residual_scale = float(
        getattr(network.heads, "seven_choice_expected_value_hidden_residual_scale_r", 0.0)
        or 0.0
    )
    network.eval()
    with torch.no_grad():
        for start in range(0, len(data.observations), 4096):
            stop = min(len(data.observations), start + 4096)
            root_batch = torch.as_tensor(
                np.asarray(roots[start:stop]), dtype=torch.float32, device=device
            )
            raw = network.observation_encoder.pattern_tree.action_path_head(root_batch)
            root_residual = getattr(
                network.observation_encoder.pattern_tree, "action_path_residual", None
            )
            if isinstance(root_residual, torch.nn.Module):
                raw = raw + root_residual(root_batch)
            observation_adapter = getattr(
                network.observation_encoder.pattern_tree,
                "action_path_observation_adapter", None,
            )
            if isinstance(observation_adapter, torch.nn.Module):
                observation_batch = torch.as_tensor(
                    data.observations[start:stop], dtype=torch.float32, device=device
                )
                bounded = torch.tanh(torch.nan_to_num(
                    observation_batch, nan=0.0, posinf=0.0, neginf=0.0
                ))
                raw = raw + observation_adapter(bounded)
            path = raw.reshape(-1, 6, 5)
            predicted = torch.tanh(path[..., 2]) * 5.0
            if (
                residual_enabled and isinstance(residual_module, torch.nn.Module)
                and residual_scale > 0.0
            ):
                hidden_batch = torch.as_tensor(
                    np.asarray(hidden[start:stop]), dtype=torch.float32, device=device
                )
                predicted = predicted + residual_scale * torch.tanh(
                    residual_module(hidden_batch)
                )
            predicted_win = torch.sigmoid(path[..., 1])
            predicted_fill = torch.sigmoid(path[..., 0])
            if selection_score_mode == "value":
                selection = predicted
            elif selection_score_mode == "value_times_win":
                selection = predicted * predicted_win
            elif selection_score_mode == "filled_value":
                selection = predicted
            else:
                selection = predicted_win
            selection_np = selection.cpu().numpy()
            if selection_score_mode == "filled_value":
                selection_np = np.where(
                    predicted_fill.cpu().numpy() >= 0.10, selection_np, -5.0
                )
            else:
                selection_np = np.where(
                    predicted_win.cpu().numpy() >= threshold_floor,
                    selection_np,
                    -5.0,
                )
            scores[start:stop, 1:7] = selection_np
    return scores


def _recalibrate_expected_value_threshold(
    network: SharedFiveTradingPolicy,
    data: SevenChoiceDataset,
    hidden: Any,
    roots: Any,
    *,
    selection_score_mode: str,
    validation_blocks: int,
    minimum_validation_entries: int,
    allow_research_only: bool,
) -> tuple[SevenChoiceDataset, dict[str, Any]]:
    """Đặt lại ngưỡng sau khi phần trạng thái ẩn làm thay đổi mặt giá trị.

    Ngưỡng cũ được chọn trước khi hiệu chỉnh. Giữ ngưỡng đó sẽ làm tăng số
    lệnh ngoài ý muốn khi phần hiệu chỉnh đẩy toàn bộ giá trị lên; việc đặt lại
    chỉ dùng tập kiểm định và không đọc kết quả tập thử kín.
    """
    if validation_blocks < 1 or minimum_validation_entries < 2:
        raise ValueError("tham so dat lai nguong gia tri khong hop le")
    scores = _expected_value_score_matrix(
        network, data, hidden, roots, selection_score_mode=selection_score_mode
    )
    validation_mask = data.splits == "validation"
    validation_best = np.max(
        np.where(
            data.allowed_action_mask[validation_mask, 1:7],
            scores[validation_mask, 1:7],
            -np.inf,
        ),
        axis=1,
    )
    finite = validation_best[np.isfinite(validation_best)]
    if finite.size == 0:
        raise RuntimeError("khong co diem kiem dinh huu han de dat lai nguong")
    thresholds = np.unique(np.quantile(finite, np.linspace(0.25, 0.995, 151)))
    validation_indices = np.flatnonzero(validation_mask)
    validation_parts = [
        part for part in np.array_split(validation_indices, validation_blocks)
        if len(part)
    ]
    minimum_block_entries = max(
        2, int(math.ceil(minimum_validation_entries / max(1, len(validation_parts))))
    )
    candidates: list[tuple[float, float, dict[str, Any], list[dict[str, Any]]]] = []
    for threshold in thresholds:
        _, metrics = _entry_selection_metrics(
            data, scores, validation_mask, float(threshold)
        )
        if int(metrics["closed_entries"] or 0) < minimum_validation_entries:
            continue
        block_metrics = [
            _entry_selection_metrics(data, scores, block_mask, float(threshold))[1]
            for block_mask in [
                np.isin(np.arange(len(data.observations)), part)
                for part in validation_parts
            ]
        ]
        if any(
            int(item["closed_entries"] or 0) < minimum_block_entries
            for item in block_metrics
        ):
            continue
        selection_lcb = min(
            float(item["closed_lcb_95_one_sided_block"] or -1e9)
            for item in block_metrics
        )
        candidates.append((selection_lcb, float(threshold), metrics, block_metrics))
    research_only = bool(allow_research_only)
    if not candidates:
        if not research_only:
            raise RuntimeError("khong tao duoc nguong moi sau khi hieu chinh trang thai an")
        threshold = float(np.min(finite) - 1e-6)
        _, fallback = _entry_selection_metrics(
            data, scores, validation_mask, threshold
        )
        blocks = [
            _entry_selection_metrics(
                data, scores,
                np.isin(np.arange(len(data.observations)), part),
                threshold,
            )[1]
            for part in validation_parts
        ]
        candidates.append((
            min(float(item["closed_lcb_95_one_sided_block"] or -1e9) for item in blocks),
            threshold, fallback, blocks,
        ))
    selection_lcb, threshold, selected_metrics, selected_blocks = max(
        candidates, key=lambda item: float(item[0])
    )
    targets = np.zeros(len(data.observations), dtype=np.int64)
    for split in ("train", "validation", "test"):
        selected, _ = _entry_selection_metrics(
            data, scores, data.splits == split, threshold
        )
        targets[np.flatnonzero(data.splits == split)] = selected
    recalibrated = replace(data, best_actions=targets, action_scores_r=scores)
    return recalibrated, {
        "enabled": True,
        "selected_threshold": float(threshold),
        "previous_threshold": float(
            getattr(network.heads, "seven_choice_expected_value_threshold_r", 0.0)
            or 0.0
        ),
        "selected_validation_lcb": float(selection_lcb),
        "selected_validation_metrics": selected_metrics,
        "selected_validation_block_metrics": selected_blocks,
        "validation_blocks": int(validation_blocks),
        "minimum_validation_entries": int(minimum_validation_entries),
        "test_used_for_selection": False,
        "research_only": bool(research_only),
    }


def _fit_expected_value_teacher(
    network: SharedFiveTradingPolicy,
    data: SevenChoiceDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    minimum_validation_entries: int,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None,
    allow_research_only: bool = False,
    positive_outcome_weight: float = 0.0,
    validation_blocks: int = 1,
    selection_score_mode: str = "value",
    ranking_loss_weight: float = 0.5,
    win_target_threshold_r: float = 0.0,
    path_excursion_loss_weight: float = 0.0,
) -> tuple[SevenChoiceDataset, Any, dict[str, Any]]:
    """Fit all-action outcomes, then select entries by validation lower bound."""
    import torch
    import torch.nn.functional as F

    if not math.isfinite(positive_outcome_weight) or positive_outcome_weight < 0.0:
        raise ValueError("positive_outcome_weight phai huu han va >= 0")
    if not math.isfinite(ranking_loss_weight) or ranking_loss_weight < 0.0:
        raise ValueError("ranking_loss_weight phai huu han va >= 0")
    if not math.isfinite(win_target_threshold_r):
        raise ValueError("win_target_threshold_r phai huu han")
    if not math.isfinite(path_excursion_loss_weight) or path_excursion_loss_weight < 0.0:
        raise ValueError("path_excursion_loss_weight phai huu han va >= 0")
    if validation_blocks < 1:
        raise ValueError("validation_blocks phai >= 1")
    if selection_score_mode not in {
        "value", "value_times_win", "filled_value", "win_probability",
    }:
        raise ValueError("selection_score_mode khong hop le")
    device = next(network.parameters()).device
    network.require_current_architecture()
    unique = np.flatnonzero(data.agent_ids == 0)
    if len(unique) * 2 != len(data.observations):
        raise ValueError("tap bay lua chon phai co dung hai vai vao lenh moi thoi diem")
    if not np.all(
        (unique + 1 < len(data.agent_ids))
        & (data.agent_ids[unique + 1] == 1)
        & (data.sample_times[unique + 1] == data.sample_times[unique])
    ):
        raise ValueError("hai vai vao lenh phai nam lien tiep trong tap bay lua chon")
    head = network.observation_encoder.pattern_tree.action_path_head
    residual = getattr(
        network.observation_encoder.pattern_tree, "action_path_residual", None
    )
    observation_adapter = getattr(
        network.observation_encoder.pattern_tree,
        "action_path_observation_adapter", None,
    )
    train_modules = [head]
    if isinstance(residual, torch.nn.Module):
        train_modules.append(residual)
    if isinstance(observation_adapter, torch.nn.Module):
        train_modules.append(observation_adapter)
    def action_path_raw(
        root: torch.Tensor, raw_observations: torch.Tensor | None = None
    ) -> torch.Tensor:
        value = head(root)
        if isinstance(residual, torch.nn.Module):
            value = value + residual(root)
        if isinstance(observation_adapter, torch.nn.Module) and raw_observations is not None:
            bounded = torch.tanh(torch.nan_to_num(
                raw_observations, nan=0.0, posinf=0.0, neginf=0.0
            ))
            value = value + observation_adapter(bounded)
        return value
    roots_unique = torch.empty((len(unique), int(head.in_features)), dtype=torch.float32)
    network.eval()
    with torch.no_grad():
        for start in range(0, len(unique), 4096):
            idx = unique[start:start + 4096]
            obs = torch.as_tensor(data.observations[idx], dtype=torch.float32, device=device)
            root = network.observation_encoder.pattern_predictions(obs)["root"].cpu()
            roots_unique[start:start + len(root)] = root

    for parameter in network.parameters():
        parameter.requires_grad_(False)
    train_parameters = []
    for module in train_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            train_parameters.append(parameter)
    optimizer = torch.optim.Adam(train_parameters, lr=learning_rate, weight_decay=1e-4)
    unique_splits = data.splits[unique]
    unique_net = data.net_r[unique]
    unique_mfe = data.mfe_r[unique]
    unique_mae = data.mae_r[unique]
    unique_filled = data.filled[unique, 1:7]
    unique_outcome_long = data.outcome_best_actions[unique]
    unique_outcome_short = data.outcome_best_actions[unique + 1]
    unique_observations = torch.as_tensor(
        data.observations[unique], dtype=torch.float32
    )
    train_indices = np.flatnonzero(unique_splits == "train")
    validation_indices = np.flatnonzero(unique_splits == "validation")
    rng = np.random.default_rng(int(seed))
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] = {}
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_indices)
        for module in train_modules:
            module.train()
        total = 0.0
        seen = 0
        for start in range(0, len(train_indices), batch_size):
            idx = train_indices[start:start + batch_size]
            raw = action_path_raw(
                roots_unique[idx].to(device=device),
                unique_observations[idx].to(device=device),
            ).reshape(-1, 6, 5)
            fill_target = torch.as_tensor(
                unique_filled[idx], dtype=torch.float32, device=device
            )
            net_target = torch.as_tensor(
                np.clip(unique_net[idx, 1:7] / 5.0, -1.0, 1.0),
                dtype=torch.float32, device=device,
            )
            win_target = torch.as_tensor(
                unique_net[idx, 1:7] > float(win_target_threshold_r),
                dtype=torch.float32, device=device,
            )
            excursion_target = torch.as_tensor(
                np.stack((
                    np.tanh(np.maximum(unique_mfe[idx, 1:7], 0.0) / 2.0),
                    np.tanh(np.maximum(-unique_mae[idx, 1:7], 0.0) / 2.0),
                ), axis=-1,
                dtype=np.float32,
            ),
                dtype=torch.float32, device=device,
            )
            filled_mask = fill_target >= 0.5
            value_error = F.smooth_l1_loss(
                torch.tanh(raw[..., 2]), net_target, reduction="none"
            )
            if positive_outcome_weight > 0.0:
                value_weight = 1.0 + positive_outcome_weight * (
                    net_target > 0.0
                ).to(dtype=value_error.dtype)
                value_error = value_error * value_weight
            if selection_score_mode == "filled_value":
                # Dau gia tri co dieu kien chi dung cho lenh da khop. Dau xac
                # suat thang van giu sai so tren toan bo mau de tranh thay doi
                # phan than dung chung khi chua co bang chung ben vung.
                value_loss = value_error[filled_mask].mean() if bool(
                    filled_mask.any().item()
                ) else value_error.mean() * 0.0
            else:
                value_loss = value_error.mean()
            loss = (
                2.0 * value_loss
                + 0.25 * F.binary_cross_entropy_with_logits(raw[..., 0], fill_target)
            )
            # A zero net result for an unfilled pending order is not a loss or
            # a win.  The win head therefore learns only from filled rows;
            # otherwise it settles near the fill rate and cannot express the
            # break-even probability of a fixed reward/risk geometry.
            if bool(filled_mask.any().item()):
                loss = loss + 0.25 * F.binary_cross_entropy_with_logits(
                    raw[..., 1][filled_mask], win_target[filled_mask]
                )
            if path_excursion_loss_weight > 0.0 and bool(filled_mask.any().item()):
                excursion_error = F.smooth_l1_loss(
                    torch.sigmoid(raw[..., 3:5]), excursion_target,
                    reduction="none",
                ).mean(dim=-1)
                loss = loss + float(path_excursion_loss_weight) * excursion_error[filled_mask].mean()
            if ranking_loss_weight > 0.0:
                # Train the value surface as a role-legal ranking problem as
                # well as a pointwise regression.  HOLD is the zero utility
                # baseline; the target is future-derived and remains offline.
                value_scores = torch.tanh(raw[..., 2])
                hold_score = torch.zeros(
                    (len(idx), 1), dtype=value_scores.dtype, device=device
                )
                long_scores = torch.cat(
                    (hold_score, value_scores[:, [0, 2, 4]]), dim=1
                )
                short_scores = torch.cat(
                    (hold_score, value_scores[:, [1, 3, 5]]), dim=1
                )
                long_map = {0: 0, 1: 1, 3: 2, 5: 3}
                short_map = {0: 0, 2: 1, 4: 2, 6: 3}
                long_target = torch.as_tensor(
                    [long_map[int(value)] for value in unique_outcome_long[idx]],
                    dtype=torch.long,
                    device=device,
                )
                short_target = torch.as_tensor(
                    [short_map[int(value)] for value in unique_outcome_short[idx]],
                    dtype=torch.long,
                    device=device,
                )
                ranking_loss = 0.5 * (
                    F.cross_entropy(long_scores / 0.5, long_target)
                    + F.cross_entropy(short_scores / 0.5, short_target)
                )
                loss = loss + float(ranking_loss_weight) * ranking_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        for module in train_modules:
            module.eval()
        with torch.no_grad():
            raw = action_path_raw(
                roots_unique[validation_indices].to(device=device),
                unique_observations[validation_indices].to(device=device),
            ).reshape(-1, 6, 5)
            predicted = torch.tanh(raw[..., 2]).cpu().numpy() * 5.0
        validation_mae = float(np.mean(np.abs(
            predicted - np.clip(unique_net[validation_indices, 1:7], -5.0, 5.0)
        )))
        row = {
            "epoch": float(epoch),
            "train_loss": total / max(1, seen),
            "validation_net_r_mae": validation_mae,
        }
        history.append(row)
        if progress is not None:
            progress({"stage": "expected_value", **row})
        if validation_mae < best_loss:
            best_loss = validation_mae
            best_epoch = epoch
            best_state = {}
            for module_name, module in (
                ("head", head),
                ("residual", residual),
                ("observation_adapter", observation_adapter),
            ):
                if not isinstance(module, torch.nn.Module):
                    continue
                for name, value in module.state_dict().items():
                    best_state[f"{module_name}.{name}"] = value.detach().cpu().clone()
    if not best_state:
        raise RuntimeError("khong tao duoc dau du doan ky vong")
    head.load_state_dict({
        name[len("head."):]: value for name, value in best_state.items()
        if name.startswith("head.")
    })
    if isinstance(residual, torch.nn.Module):
        residual.load_state_dict({
            name[len("residual."):]: value for name, value in best_state.items()
            if name.startswith("residual.")
        })
    if isinstance(observation_adapter, torch.nn.Module):
        observation_adapter.load_state_dict({
            name[len("observation_adapter."):]: value
            for name, value in best_state.items()
            if name.startswith("observation_adapter.")
        })
    for module in train_modules:
        module.eval()
    predicted_unique = np.zeros((len(unique), 6), dtype=np.float32)
    predicted_win_unique = np.zeros((len(unique), 6), dtype=np.float32)
    predicted_fill_unique = np.zeros((len(unique), 6), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(unique), 4096):
            raw = action_path_raw(
                roots_unique[start:start + 4096].to(device=device),
                unique_observations[start:start + 4096].to(device=device),
            ).reshape(-1, 6, 5)
            predicted_unique[start:start + len(raw)] = (
                torch.tanh(raw[..., 2]).cpu().numpy() * 5.0
            )
            predicted_fill_unique[start:start + len(raw)] = (
                torch.sigmoid(raw[..., 0]).cpu().numpy()
            )
            predicted_win_unique[start:start + len(raw)] = (
                torch.sigmoid(raw[..., 1]).cpu().numpy()
            )
    predicted = np.repeat(predicted_unique[:, None, :], 2, axis=1).reshape(-1, 6)
    predicted_win = np.repeat(
        predicted_win_unique[:, None, :], 2, axis=1
    ).reshape(-1, 6)
    predicted_fill = np.repeat(
        predicted_fill_unique[:, None, :], 2, axis=1
    ).reshape(-1, 6)
    fixed_tp_rr = float(data.manifest.get("config", {}).get("fixed_tp_rr", 2.0))
    break_even_win_probability = 1.0 / (1.0 + max(fixed_tp_rr, 1e-9))
    win_probability_floor = float(
        min(0.95, break_even_win_probability + 0.05)
    )
    safety_win_probability_floor = float(win_probability_floor)
    research_only = bool(allow_research_only)
    scores = np.zeros((len(data.observations), 7), dtype=np.float32)
    if selection_score_mode == "value":
        selection_scores = predicted
    elif selection_score_mode == "value_times_win":
        selection_scores = predicted * predicted_win
    elif selection_score_mode == "filled_value":
        fill_probability_floor = 0.10
        # ``predicted`` is trained only on rows that actually filled, so it
        # already estimates E[net_R | filled].  The target here is the mean
        # result per filled trade; dividing by fill probability would magnify
        # low-fill actions and change the objective into no meaningful
        # economic quantity.  The fill head remains an eligibility floor.
        selection_scores = predicted
    else:
        selection_scores = predicted_win
    if selection_score_mode == "filled_value":
        scores[:, 1:7] = np.where(
            predicted_fill >= 0.10,
            selection_scores,
            -5.0,
        )
    else:
        scores[:, 1:7] = np.where(
            predicted_win >= (0.0 if allow_research_only else safety_win_probability_floor),
            selection_scores,
            -5.0,
        )
    validation_mask = data.splits == "validation"
    validation_best = np.max(
        np.where(data.allowed_action_mask[validation_mask, 1:7], scores[validation_mask, 1:7], -np.inf),
        axis=1,
    )
    thresholds = np.unique(np.quantile(validation_best, np.linspace(0.25, 0.995, 151)))
    validation_indices = np.flatnonzero(validation_mask)
    validation_parts = [
        part for part in np.array_split(validation_indices, validation_blocks)
        if len(part)
    ]
    validation_block_masks = [
        np.isin(np.arange(len(data.observations)), part) for part in validation_parts
    ]
    minimum_block_entries = max(
        2, int(math.ceil(minimum_validation_entries / max(1, len(validation_parts))))
    )
    candidates: list[
        tuple[float, float, dict[str, float | int | None], list[dict[str, float | int | None]]]
    ] = []
    for threshold in thresholds:
        _, metrics = _entry_selection_metrics(
            data, scores, validation_mask, float(threshold)
        )
        if int(metrics["closed_entries"] or 0) >= minimum_validation_entries:
            block_metrics = [
                _entry_selection_metrics(data, scores, block_mask, float(threshold))[1]
                for block_mask in validation_block_masks
            ]
            if any(
                int(item["closed_entries"] or 0) < minimum_block_entries
                for item in block_metrics
            ):
                continue
            selection_lcb = min(
                float(item["closed_lcb_95_one_sided_block"] or -1e9)
                for item in block_metrics
            )
            candidates.append((selection_lcb, float(threshold), metrics, block_metrics))
    if not candidates:
        if not allow_research_only:
            raise RuntimeError("dau du doan ky vong khong tao du nguong co mau kiem dinh")
        finite = validation_best[np.isfinite(validation_best)]
        if finite.size == 0:
            raise RuntimeError("dau du doan ky vong khong co diem kiem dinh huu han")
        # Keep a checkpoint for inspection when the economic gate is negative.
        # The threshold is deliberately below the lowest score so the report
        # contains the raw learned entry surface instead of silently becoming
        # an all-HOLD artifact.  This path is never production-eligible.
        threshold = float(np.min(finite) - 1e-6)
        _, fallback_metrics = _entry_selection_metrics(
            data, scores, validation_mask, threshold
        )
        fallback_blocks = [
            _entry_selection_metrics(data, scores, block_mask, threshold)[1]
            for block_mask in validation_block_masks
        ]
        candidates.append((
            min(
                float(item["closed_lcb_95_one_sided_block"] or -1e9)
                for item in fallback_blocks
            ),
            threshold, fallback_metrics, fallback_blocks,
        ))
        research_only = True
    selection_lcb, threshold, selected_validation_metrics, selected_block_metrics = max(
        candidates,
        key=lambda item: float(item[0]),
    )
    targets = np.zeros(len(data.observations), dtype=np.int64)
    results: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        mask = data.splits == split
        selected, metrics = _entry_selection_metrics(data, scores, mask, threshold)
        targets[np.flatnonzero(mask)] = selected
        results[split] = metrics
    trained = replace(data, best_actions=targets, action_scores_r=scores)
    roots = torch.repeat_interleave(roots_unique, 2, dim=0)
    return trained, roots, {
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history),
        "history": history,
        "selected_threshold": float(threshold),
        "break_even_win_probability": float(break_even_win_probability),
        "win_probability_floor": float(
            0.0 if research_only else safety_win_probability_floor
        ),
        "fill_probability_floor": float(
            0.10 if selection_score_mode == "filled_value" else 0.0
        ),
        "safety_win_probability_floor": float(safety_win_probability_floor),
        "win_probability_floor_enforced": bool(not research_only),
        "research_only": bool(research_only),
        "positive_outcome_weight": float(positive_outcome_weight),
        "validation_blocks": int(validation_blocks),
        "selection_score_mode": selection_score_mode,
        "ranking_loss_weight": float(ranking_loss_weight),
        "win_target_threshold_r": float(win_target_threshold_r),
        "path_excursion_loss_weight": float(path_excursion_loss_weight),
        "minimum_validation_block_entries": int(minimum_block_entries),
        "selected_validation_lcb_for_threshold": float(selection_lcb),
        "selected_validation_block_metrics": selected_block_metrics,
        "minimum_validation_entries": int(minimum_validation_entries),
        "selection_uses": "validation only",
        "test_used_for_selection": False,
        "results": results,
        "validation_gate_passed": bool(
            (results["validation"]["closed_lcb_95_one_sided_block"] or -1e9) > 0.0
        ),
        "test_gate_passed": bool(
            (results["test"]["closed_lcb_95_one_sided_block"] or -1e9) > 0.0
        ),
    }


def _head_metrics(
    network: SharedFiveTradingPolicy,
    data: SevenChoiceDataset,
    hidden: Any,
    roots: Any,
    mask: np.ndarray,
    *,
    target_actions: np.ndarray | None = None,
) -> dict[str, float]:
    import torch

    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("split rong")
    if target_actions is not None:
        target_actions = np.asarray(target_actions, dtype=np.int64)
        if target_actions.shape != (len(data.observations),):
            raise ValueError("target_actions phai cung kich thuoc voi dataset")
    device = next(network.parameters()).device
    action_correct = 0
    regret_sum = 0.0
    realized_sum = 0.0
    entry_realized_sum = 0.0
    entry_realized_square_sum = 0.0
    predicted_entries = 0
    predicted_entry_wins = 0
    filled_entries = 0
    filled_realized_sum = 0.0
    filled_realized_square_sum = 0.0
    closed_entries = 0
    closed_realized_sum = 0.0
    closed_realized_square_sum = 0.0
    closed_values_for_blocks: list[float] = []
    closed_times_for_blocks: list[str] = []
    target_entries = 0
    fill_correct = fill_total = 0
    parameter_correct = {name: 0 for name in ("delay", "entry_offset", "sl", "tp", "trailing")}
    parameter_count = {name: 0 for name in parameter_correct}
    net_abs = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), 4096):
            idx = indices[start:start + 4096]
            h = torch.as_tensor(np.asarray(hidden[idx]), dtype=torch.float32, device=device)
            agent = torch.as_tensor(data.agent_ids[idx], dtype=torch.long, device=device)
            action_mode = str(
                getattr(network.heads, "seven_choice_action_decision_mode", "expected_value")
            )
            logits = (
                network.heads.direct_action_logits(
                    h,
                    agent,
                    torch.as_tensor(
                        data.observations[idx], dtype=torch.float32, device=device
                    ),
                )
                if action_mode == "direct"
                else network.heads.primary_action_logits(h, agent)
            )[:, :7]
            allowed = torch.as_tensor(data.allowed_action_mask[idx], dtype=torch.bool, device=device)
            root = torch.as_tensor(np.asarray(roots[idx]), dtype=torch.float32, device=device)
            raw = network.observation_encoder.pattern_tree.action_path_head(root)
            residual = getattr(
                network.observation_encoder.pattern_tree, "action_path_residual", None
            )
            if isinstance(residual, torch.nn.Module):
                raw = raw + residual(root)
            observation_adapter = getattr(
                network.observation_encoder.pattern_tree,
                "action_path_observation_adapter", None,
            )
            if isinstance(observation_adapter, torch.nn.Module):
                bounded = torch.tanh(torch.nan_to_num(
                    torch.as_tensor(
                        data.observations[idx], dtype=torch.float32, device=device
                    ),
                    nan=0.0, posinf=0.0, neginf=0.0,
                ))
                raw = raw + observation_adapter(bounded)
            raw = raw.reshape(-1, 6, 5)
            path = torch.cat((
                torch.sigmoid(raw[..., :2]),
                torch.tanh(raw[..., 2:3]),
                torch.sigmoid(raw[..., 3:]),
            ), dim=-1)
            if bool(getattr(network.heads, "seven_choice_expected_value_decision_enabled", False)):
                logits = network._apply_seven_choice_expected_value_decision(
                    logits, agent, path, policy_hidden=h,
                )
            predicted = logits.masked_fill(~allowed, -30.0).argmax(dim=-1).cpu().numpy()
            expected = (
                target_actions[idx]
                if target_actions is not None
                else data.best_actions[idx]
            )
            action_correct += int(np.sum(predicted == expected))
            if target_actions is None:
                chosen_utility = data.action_scores_r[idx, predicted]
                best_utility = data.action_scores_r[idx, expected]
            else:
                # With a future-derived outcome target, regret is measured on
                # realised net-R rather than the causal teacher score.
                chosen_utility = data.net_r[idx, predicted]
                best_utility = data.net_r[idx, expected]
            regret_sum += float(np.sum(best_utility - chosen_utility))
            realized = data.net_r[idx, predicted]
            realized_sum += float(np.sum(realized))
            predicted_entry = predicted > 0
            target_entries += int(np.sum(expected > 0))
            if np.any(predicted_entry):
                values = realized[predicted_entry]
                predicted_entries += int(len(values))
                entry_realized_sum += float(np.sum(values))
                entry_realized_square_sum += float(np.sum(values * values))
                predicted_entry_wins += int(np.sum(values > 0.0))
                filled_mask = predicted_entry & data.filled[idx, predicted]
                closed_mask = filled_mask & (data.exit_codes[idx, predicted] > 0)
                if np.any(filled_mask):
                    filled_values = realized[filled_mask]
                    filled_entries += int(len(filled_values))
                    filled_realized_sum += float(np.sum(filled_values))
                    filled_realized_square_sum += float(np.sum(filled_values * filled_values))
                if np.any(closed_mask):
                    closed_values = realized[closed_mask]
                    closed_entries += int(len(closed_values))
                    closed_realized_sum += float(np.sum(closed_values))
                    closed_realized_square_sum += float(np.sum(closed_values * closed_values))
                    closed_values_for_blocks.extend(float(value) for value in closed_values.tolist())
                    closed_times_for_blocks.extend(
                        str(value) for value in data.sample_times[idx[closed_mask]].tolist()
                    )
            fill_pred = (torch.sigmoid(raw[..., 0]) >= 0.5).cpu().numpy()
            fill_target = data.filled[idx, 1:7]
            fill_correct += int(np.sum(fill_pred == fill_target))
            fill_total += int(fill_target.size)
            net_pred = torch.tanh(raw[..., 2]).cpu().numpy() * 5.0
            net_abs += float(np.sum(np.abs(net_pred - np.clip(data.net_r[idx, 1:7], -5.0, 5.0))))
            active = expected > 0
            if np.any(active):
                active_idx = torch.as_tensor(np.flatnonzero(active), dtype=torch.long, device=device)
                selected = expected[active]
                targets = {
                    "delay": data.placement_delay_indices[idx[active], selected],
                    "entry_offset": data.entry_offset_indices[idx[active], selected],
                    "sl": data.sl_indices[idx[active], selected],
                    "tp": data.tp_indices[idx[active], selected],
                    "trailing": np.zeros(int(active.sum()), dtype=np.int64),
                }
                for name, target in targets.items():
                    head_active = (
                        np.isin(selected, PENDING_CHOICES)
                        if name == "entry_offset" else np.ones(len(selected), dtype=np.bool_)
                    )
                    if not np.any(head_active):
                        continue
                    local = active_idx[torch.as_tensor(head_active, dtype=torch.bool, device=device)]
                    pred = network.heads.parameter_heads[name](h[local]).argmax(dim=-1).cpu().numpy()
                    parameter_correct[name] += int(np.sum(pred == target[head_active]))
                    parameter_count[name] += int(np.sum(head_active))
    entry_mean = entry_realized_sum / max(1, predicted_entries)
    if predicted_entries >= 2:
        entry_variance = max(
            0.0,
            (
                entry_realized_square_sum
                - entry_realized_sum * entry_realized_sum / predicted_entries
            ) / (predicted_entries - 1),
        )
        entry_lcb = entry_mean - 1.64 * math.sqrt(entry_variance) / math.sqrt(
            predicted_entries
        )
    else:
        entry_lcb = None
    def _mean_and_lcb(total: float, squared: float, count: int) -> tuple[float | None, float | None]:
        if count <= 0:
            return None, None
        mean_value = total / float(count)
        if count < 2:
            return mean_value, None
        variance = max(
            0.0,
            (squared - total * total / float(count)) / float(count - 1),
        )
        return mean_value, mean_value - 1.64 * math.sqrt(variance) / math.sqrt(float(count))

    filled_mean, filled_lcb = _mean_and_lcb(
        filled_realized_sum, filled_realized_square_sum, filled_entries
    )
    closed_mean, closed_lcb = _mean_and_lcb(
        closed_realized_sum, closed_realized_square_sum, closed_entries
    )
    closed_block_lcb = _grouped_lcb(
        np.asarray(closed_values_for_blocks, dtype=np.float64),
        np.asarray(closed_times_for_blocks, dtype="U19"),
    )
    result = {
        "action_accuracy": action_correct / len(indices),
        "mean_net_r_regret": regret_sum / len(indices),
        "mean_causal_score_regret": regret_sum / len(indices),
        "mean_realized_net_r": realized_sum / len(indices),
        "entry_realized_net_r": entry_mean,
        "entry_realized_lcb_95_one_sided": entry_lcb,
        "predicted_entries": float(predicted_entries),
        "proposed_entries": float(predicted_entries),
        "predicted_entry_rate": predicted_entries / len(indices),
        "target_entry_rate": target_entries / len(indices),
        "predicted_entry_win_rate": predicted_entry_wins / max(1, predicted_entries),
        "filled_entries": float(filled_entries),
        "closed_entries": float(closed_entries),
        "pending_unfilled_entries": float(predicted_entries - filled_entries),
        "proposal_fill_rate": filled_entries / max(1, predicted_entries),
        "filled_entry_realized_net_r": filled_mean,
        "filled_entry_realized_lcb_95_one_sided_iid": filled_lcb,
        "closed_entry_realized_net_r": closed_mean,
        "closed_entry_realized_lcb_95_one_sided_iid": closed_lcb,
        "closed_entry_realized_lcb_95_one_sided_block": closed_block_lcb,
        "lcb_independence_note": "chi la uoc luong IID; can bootstrap theo khoi/ngay cho cong",
        "fill_accuracy": fill_correct / max(1, fill_total),
        "net_r_mae_clipped_5r": net_abs / max(1, fill_total),
        "rows": float(len(indices)),
        "entry_rows": float(parameter_count["delay"]),
        "pending_entry_rows": float(parameter_count["entry_offset"]),
    }
    for name, correct in parameter_correct.items():
        result[f"{name}_accuracy"] = correct / max(1, parameter_count[name])
    return result


def _fit_hidden_value_residual(
    network: SharedFiveTradingPolicy,
    data: SevenChoiceDataset,
    hidden: Any,
    roots: Any,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    scale_r: float,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None,
    ranking_loss_weight: float = 0.0,
) -> dict[str, Any]:
    """Fit a bounded value correction from the frozen recurrent state.

    The root value surface remains the economic prior.  This helper trains
    only a zero-initialized six-output residual on the owner hidden state, so
    it can expose role/temporal information without changing the encoder,
    LSTM, Transformer, or the 137-input contract.  It is deliberately opt-in
    and research-only until a paired frozen replay proves a gain.
    """
    import torch
    import torch.nn.functional as F

    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("tham so hieu chinh gia tri theo trang thai an khong hop le")
    if not math.isfinite(scale_r) or not 0.0 < scale_r <= 2.0:
        raise ValueError("scale_r cua hieu chinh gia tri theo trang thai an khong hop le")
    if not math.isfinite(ranking_loss_weight) or ranking_loss_weight < 0.0:
        raise ValueError("ranking_loss_weight cua hieu chinh gia tri theo trang thai an khong hop le")
    if len(hidden) != len(data.observations) or len(roots) != len(data.observations):
        raise ValueError("hidden/roots phai cung so dong voi tap bay lua chon")
    module = getattr(network.heads, "expected_value_hidden_residual", None)
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError("thieu dau hieu chinh gia tri theo trang thai an")
    device = next(network.parameters()).device
    network.eval()
    module.eval()

    # Freeze the existing value surface while materialising its prediction.
    base_values = torch.empty((len(data.observations), 6), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(data.observations), 4096):
            stop = min(len(data.observations), start + 4096)
            root_batch = torch.as_tensor(
                np.asarray(roots[start:stop]), dtype=torch.float32, device=device
            )
            raw = network.observation_encoder.pattern_tree.action_path_head(root_batch)
            root_residual = getattr(
                network.observation_encoder.pattern_tree, "action_path_residual", None
            )
            if isinstance(root_residual, torch.nn.Module):
                raw = raw + root_residual(root_batch)
            observation_adapter = getattr(
                network.observation_encoder.pattern_tree,
                "action_path_observation_adapter",
                None,
            )
            if isinstance(observation_adapter, torch.nn.Module):
                bounded = torch.tanh(torch.nan_to_num(
                    torch.as_tensor(
                        data.observations[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ))
                raw = raw + observation_adapter(bounded)
            base_values[start:stop] = (torch.tanh(raw.reshape(-1, 6, 5)[..., 2]) * 5.0).cpu()

    owner_rows = np.isin(data.agent_ids, (0, 1))
    legal_entries = np.asarray(data.allowed_action_mask[:, 1:7], dtype=np.bool_)
    eligible = owner_rows[:, None] & legal_entries
    target_values = np.clip(np.asarray(data.net_r[:, 1:7], dtype=np.float32), -5.0, 5.0)
    base_numpy = base_values.numpy()
    target_delta = np.clip(target_values - base_numpy, -float(scale_r), float(scale_r))
    target_delta = target_delta / float(scale_r)
    train_mask = (data.splits == "train")[:, None] & eligible
    validation_mask = (data.splits == "validation")[:, None] & eligible
    if int(train_mask.sum()) < 2 or int(validation_mask.sum()) < 2:
        raise RuntimeError("khong du dong hop le de hoc hieu chinh gia tri theo trang thai an")

    for parameter in network.parameters():
        parameter.requires_grad_(False)
    train_parameters = []
    for parameter in module.parameters():
        parameter.requires_grad_(True)
        train_parameters.append(parameter)
    optimizer = torch.optim.Adam(train_parameters, lr=float(learning_rate), weight_decay=1e-4)
    train_rows = np.flatnonzero(np.any(train_mask, axis=1))
    validation_rows = np.flatnonzero(np.any(validation_mask, axis=1))
    rng = np.random.default_rng(int(seed))

    def _ranking_loss(
        values: torch.Tensor,
        rows: np.ndarray,
    ) -> torch.Tensor:
        """Xep hang HOLD va ba lua chon dung vai cua tung chu so."""
        if ranking_loss_weight <= 0.0:
            return values.new_zeros(())
        agents = torch.as_tensor(data.agent_ids[rows], dtype=torch.long, device=device)
        targets = np.asarray(data.outcome_best_actions[rows], dtype=np.int64)
        terms: list[torch.Tensor] = []
        for role, choices, mapping in (
            (0, (1, 3, 5), {0: 0, 1: 1, 3: 2, 5: 3}),
            (1, (2, 4, 6), {0: 0, 2: 1, 4: 2, 6: 3}),
        ):
            mask = agents == int(role)
            if not bool(mask.any().item()):
                continue
            role_values = values[mask][:, [action - 1 for action in choices]]
            hold = torch.zeros((role_values.shape[0], 1), dtype=values.dtype, device=device)
            logits = torch.cat((hold, role_values), dim=1) / 0.5
            target_local = torch.as_tensor(
                [mapping.get(int(value), 0) for value in targets[mask.cpu().numpy()]],
                dtype=torch.long,
                device=device,
            )
            terms.append(F.cross_entropy(logits, target_local))
        return torch.stack(terms).mean() if terms else values.new_zeros(())

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] = {}
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        shuffled = train_rows.copy()
        rng.shuffle(shuffled)
        module.train()
        total = 0.0
        seen = 0
        for start in range(0, len(shuffled), batch_size):
            rows = shuffled[start:start + batch_size]
            h = torch.as_tensor(
                np.asarray(hidden[rows]), dtype=torch.float32, device=device
            )
            predicted = torch.tanh(module(h))
            target = torch.as_tensor(target_delta[rows], dtype=torch.float32, device=device)
            mask = torch.as_tensor(train_mask[rows], dtype=torch.bool, device=device)
            if not bool(mask.any().item()):
                continue
            loss = F.smooth_l1_loss(predicted[mask], target[mask])
            if ranking_loss_weight > 0.0:
                base = torch.as_tensor(base_values[rows].numpy(), dtype=torch.float32, device=device)
                corrected = base + float(scale_r) * predicted
                loss = loss + float(ranking_loss_weight) * _ranking_loss(corrected, rows)
            # Keep the correction near zero unless it explains a target edge.
            loss = loss + 1e-3 * torch.mean(predicted * predicted)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * int(mask.sum().item())
            seen += int(mask.sum().item())
        module.eval()
        with torch.no_grad():
            validation_loss_sum = 0.0
            validation_count = 0
            for start in range(0, len(validation_rows), batch_size):
                rows = validation_rows[start:start + batch_size]
                h = torch.as_tensor(
                    np.asarray(hidden[rows]), dtype=torch.float32, device=device
                )
                predicted = torch.tanh(module(h))
                target = torch.as_tensor(target_delta[rows], dtype=torch.float32, device=device)
                mask = torch.as_tensor(validation_mask[rows], dtype=torch.bool, device=device)
                if bool(mask.any().item()):
                    value = F.smooth_l1_loss(predicted[mask], target[mask], reduction="sum")
                    validation_loss_sum += float(value.cpu())
                    validation_count += int(mask.sum().item())
                    if ranking_loss_weight > 0.0:
                        base = torch.as_tensor(base_values[rows].numpy(), dtype=torch.float32, device=device)
                        corrected = base + float(scale_r) * predicted
                        validation_loss_sum += float(
                            ranking_loss_weight
                            * _ranking_loss(corrected, rows).detach().cpu()
                            * int(mask.sum().item())
                        )
            validation_loss = validation_loss_sum / max(1, validation_count)
        row = {
            "epoch": float(epoch),
            "train_loss": total / max(1, seen),
            "validation_loss": float(validation_loss),
        }
        history.append(row)
        if progress is not None:
            progress({"stage": "hidden_value_residual", **row})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in module.state_dict().items()
            }
    if not best_state:
        raise RuntimeError("khong tao duoc hieu chinh gia tri theo trang thai an")
    module.load_state_dict(best_state)
    module.eval()
    network.heads.seven_choice_expected_value_hidden_residual_enabled = True
    network.heads.seven_choice_expected_value_hidden_residual_scale_r = float(scale_r)
    return {
        "enabled": True,
        "scale_r": float(scale_r),
        "ranking_loss_weight": float(ranking_loss_weight),
        "epochs_ran": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "train_eligible_values": int(train_mask.sum()),
        "validation_eligible_values": int(validation_mask.sum()),
        "history": history,
        "target": "net_r sau phi, chi tren cac action hop le cua hai chu so vao lenh",
        "trunk_frozen": True,
    }


def train_seven_choice_heads(
    network: SharedFiveTradingPolicy,
    data: SevenChoiceDataset,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 8,
    sequence_length: int = 64,
    seed: int = 0,
    hidden_cache_path: str | Path | None = None,
    reuse_hidden_cache: bool = False,
    hidden_extraction_batch_size: int | None = None,
    action_learning_rate_multiplier: float = 10.0,
    expected_value_epochs: int = 20,
    expected_value_learning_rate: float = 3e-3,
    positive_outcome_weight: float = 0.0,
    minimum_validation_entries: int = 128,
    allow_research_only: bool = False,
    validation_blocks: int = 1,
    action_target_mode: str = "causal",
    selection_score_mode: str = "value",
    ranking_loss_weight: float = 0.5,
    win_target_threshold_r: float = 0.0,
    path_excursion_loss_weight: float = 0.0,
    action_decision_mode: str = "expected_value",
    hidden_value_residual_enabled: bool = False,
    hidden_value_residual_epochs: int = 8,
    hidden_value_residual_learning_rate: float = 1e-3,
    hidden_value_residual_scale_r: float = 0.25,
    hidden_value_residual_ranking_loss_weight: float = 0.0,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train action/output heads while encoder, LSTM and Transformer stay frozen."""
    import torch
    import torch.nn.functional as F

    if (
        epochs < 1 or batch_size < 1 or learning_rate <= 0.0 or patience < 1
        or action_learning_rate_multiplier <= 0.0
        or expected_value_epochs < 1 or expected_value_learning_rate <= 0.0
        or not math.isfinite(positive_outcome_weight) or positive_outcome_weight < 0.0
        or minimum_validation_entries < 2
        or (hidden_extraction_batch_size is not None and hidden_extraction_batch_size < 1)
        or validation_blocks < 1
        or action_target_mode not in {"causal", "outcome"}
        or selection_score_mode not in {
            "value", "value_times_win", "filled_value", "win_probability",
        }
        or action_decision_mode not in {"expected_value", "direct"}
        or not math.isfinite(ranking_loss_weight) or ranking_loss_weight < 0.0
        or not math.isfinite(win_target_threshold_r)
        or not math.isfinite(path_excursion_loss_weight) or path_excursion_loss_weight < 0.0
        or hidden_value_residual_epochs < 1
        or hidden_value_residual_learning_rate <= 0.0
        or not math.isfinite(hidden_value_residual_scale_r)
        or not 0.0 < hidden_value_residual_scale_r <= 2.0
        or not math.isfinite(hidden_value_residual_ranking_loss_weight)
        or hidden_value_residual_ranking_loss_weight < 0.0
    ):
        raise ValueError("cau hinh train khong hop le")
    if hidden_value_residual_enabled and not allow_research_only:
        raise RuntimeError(
            "hieu chinh gia tri theo trang thai an chi duoc phep trong ho so nghien cuu"
        )
    if hidden_value_residual_enabled and action_decision_mode != "expected_value":
        raise ValueError(
            "hieu chinh gia tri theo trang thai an chi dung voi duong expected_value"
        )
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = next(network.parameters()).device
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    calibrator = network.heads.entry_role_calibrator
    calibrator.reset_teacher_mapping()
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    # Never inherit a research residual from an earlier in-memory run.  The
    # helper below enables it only after the fresh hidden replay has completed.
    network.heads.seven_choice_expected_value_hidden_residual_enabled = False
    network.heads.seven_choice_expected_value_hidden_residual_scale_r = 0.0
    invariant_before = _hash_except_seven_choice_heads(network)
    # Preserve the causal historical action stream before the value teacher
    # replaces ``best_actions`` with its threshold-selected labels.
    causal_hidden_actions = data.best_actions.copy()
    causal_action_scores = data.action_scores_r.copy()
    data, roots, expected_value_report = _fit_expected_value_teacher(
        network, data,
        epochs=expected_value_epochs,
        batch_size=max(256, batch_size),
        learning_rate=expected_value_learning_rate,
        minimum_validation_entries=minimum_validation_entries,
        seed=seed,
        allow_research_only=allow_research_only,
        positive_outcome_weight=positive_outcome_weight,
        validation_blocks=validation_blocks,
        selection_score_mode=selection_score_mode,
        ranking_loss_weight=ranking_loss_weight,
        win_target_threshold_r=win_target_threshold_r,
        path_excursion_loss_weight=path_excursion_loss_weight,
        progress=progress,
    )
    # The value teacher returns its own threshold-selected actions.  For the
    # full output-head run we may instead supervise the action/geometry heads
    # with the dense realised-outcome label.  That label is future-derived
    # offline data only; it never enters the frozen trunk or live decision
    # path.  Keeping the choice explicit prevents silently replacing the
    # causal audit target.
    # Keep the causal action stream for recurrent teacher forcing.  The dense
    # outcome labels are future-derived targets for the frozen output heads;
    # feeding them back as previous actions would leak a 48-bar-ahead result
    # into the hidden state at the next decision.
    if action_target_mode == "outcome":
        data = replace(data, best_actions=data.outcome_best_actions.copy())
    else:
        # ``_fit_expected_value_teacher`` returns a temporary thresholded
        # action stream so it can calibrate the value surface.  That stream is
        # not the causal teacher target.  Restore the dataset's historical
        # lower-bound labels for the action/geometry heads; otherwise the
        # public ``causal`` mode silently trains on a future-derived value
        # selection and differs from the mode named by the CLI.
        data = replace(
            data,
            best_actions=causal_hidden_actions.copy(),
            action_scores_r=causal_action_scores.copy(),
        )
    # The expected-value surface is only a production decision boundary when
    # explicitly selected.  In direct mode it is still fitted and reported as
    # a diagnostic, but it must not block a candidate whose primary action head
    # is the actual runtime decision source.
    if (
        action_decision_mode == "expected_value"
        and not expected_value_report["validation_gate_passed"]
        and not allow_research_only
    ):
        raise RuntimeError("dau du doan ky vong khong co can duoi duong tren tap kiem dinh")
    if (
        action_decision_mode == "expected_value"
        and not expected_value_report["test_gate_passed"]
        and not allow_research_only
    ):
        raise RuntimeError("dau du doan ky vong khong co can duoi duong tren tap thu kin")
    # Evaluate and select action-head epochs through the same value-surface
    # decision path that the live policy will execute.  Previously metrics
    # inspected only primary_action_logits even though runtime replaced them
    # with the expected-value surface.
    network.heads.seven_choice_expected_value_decision_enabled = (
        action_decision_mode == "expected_value"
    )
    network.heads.seven_choice_action_decision_mode = action_decision_mode
    network.heads.seven_choice_expected_value_threshold_r = float(
        expected_value_report["selected_threshold"]
    )
    network.heads.seven_choice_expected_value_win_probability_floor = float(
        expected_value_report.get("win_probability_floor", 0.0)
    )
    network.heads.seven_choice_expected_value_score_mode = selection_score_mode
    network.heads.seven_choice_expected_value_fill_probability_floor = float(
        expected_value_report.get("fill_probability_floor", 0.0) or 0.0
    )
    network.heads.seven_choice_expected_value_action_residual_scale = float(
        0.5 if action_target_mode == "outcome" else 0.0
    )
    if action_decision_mode == "direct":
        # The direct experiment gets a nonlinear residual surface while the
        # legacy bilinear action head remains a frozen prior.
        action_modules = [
            network.heads.direct_action_head,
            network.heads.direct_observation_head,
        ]
    else:
        action_modules = [
            network.heads.action_query,
            network.heads.action_embedding,
            calibrator,
        ]
    auxiliary_modules = [
        network.heads.parameter_heads["delay"], network.heads.parameter_heads["entry_offset"],
        network.heads.parameter_heads["sl"], network.heads.parameter_heads["tp"],
        network.heads.parameter_heads["trailing"],
    ]
    action_parameters = []
    auxiliary_parameters = []
    for module in action_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            action_parameters.append(parameter)
    for module in auxiliary_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            auxiliary_parameters.append(parameter)
    parameters = action_parameters + auxiliary_parameters
    pseudo = TeacherImitationData(
        path=data.path, observations=data.observations, actions=causal_hidden_actions,
        candidate_actions=data.best_actions,
        teacher_trade_accepted=data.best_actions > 0,
        teacher_net_r=data.action_scores_r[
            np.arange(len(data.best_actions)), data.best_actions
        ],
        agent_ids=data.agent_ids, quality_weights=np.ones(len(data.best_actions), dtype=np.float32),
        sample_times=data.sample_times, label_finalized_times=data.sample_times,
        splits=data.splits, manifest=data.manifest,
    )
    cache = Path(hidden_cache_path).resolve() if hidden_cache_path is not None else None
    cache_metadata_path = (
        cache.with_suffix(cache.suffix + ".json") if cache is not None else None
    )
    expected_cache_bytes = len(data.observations) * int(network.hidden_size) * 2
    expected_cache_metadata = {
        "schema_version": 3,
        "dataset": str(data.path.resolve()),
        "dataset_sha256": _sha256(data.path.resolve()),
        "rows": int(len(data.observations)),
        "hidden_size": int(network.hidden_size),
        "hidden_dtype": "float16",
        "sequence_length": int(sequence_length),
        "allow_extended_sequence": bool(sequence_length > 64),
        "frozen_invariant_sha256": invariant_before,
        "teacher_forcing_action_source": "causal_best_actions_before_value_teacher",
        "teacher_forcing_action_sha256": _array_sha256(causal_hidden_actions),
        "action_target_mode": action_target_mode,
        "selection_score_mode": selection_score_mode,
    }
    if (
        reuse_hidden_cache and cache is not None and cache.is_file()
        and cache.stat().st_size == expected_cache_bytes
    ):
        if cache_metadata_path is None:
            raise RuntimeError("bo nho an tai su dung thieu duong dan dau kiem nguon")
        _require_hidden_cache_metadata(cache_metadata_path, expected_cache_metadata)
        hidden = np.memmap(
            cache, mode="r", dtype=np.float16,
            shape=(len(data.observations), int(network.hidden_size)),
        )
        if not all(
            bool(np.isfinite(np.asarray(hidden[start:start + 4096])).all())
            for start in range(0, len(data.observations), 4096)
        ):
            raise RuntimeError("bo nho an tai su dung chua NaN/Inf")
        sequence_audit = {
            "sequence_length": int(sequence_length),
            "allow_extended_sequence": bool(sequence_length > 64),
            "unique_time_steps": int(len(np.unique(data.sample_times))),
            "hidden_storage": "reused_disk_memmap_float16",
            "hidden_dtype": str(hidden.dtype),
            "representation": "hidden",
            "representation_width": int(network.hidden_size),
            "cache_provenance_verified": True,
            "cache_metadata": str(cache_metadata_path),
        }
    else:
        extraction_batch_size = (
            int(hidden_extraction_batch_size)
            if hidden_extraction_batch_size is not None
            else max(1, min(16, batch_size // 2))
        )
        hidden, sequence_audit = extract_owner_hidden_sequences(
            network, pseudo, batch_size=extraction_batch_size,
            sequence_length=sequence_length,
            cache_path=cache,
            allow_extended_sequence=bool(sequence_length > 64),
            output_dtype=None if cache is not None else torch.float16,
        )
        if cache_metadata_path is not None:
            cache_metadata_path.write_text(
                json.dumps(expected_cache_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sequence_audit["cache_provenance_verified"] = True
            sequence_audit["cache_metadata"] = str(cache_metadata_path)
    hidden_value_residual_report: dict[str, Any] = {
        "enabled": False,
        "scale_r": 0.0,
        "epochs_ran": 0,
        "best_epoch": 0,
        "target": "disabled",
        "trunk_frozen": True,
    }
    if hidden_value_residual_enabled:
        hidden_value_residual_report = _fit_hidden_value_residual(
            network,
            data,
            hidden,
            roots,
            epochs=hidden_value_residual_epochs,
            batch_size=max(256, batch_size),
            learning_rate=hidden_value_residual_learning_rate,
            scale_r=hidden_value_residual_scale_r,
            seed=seed,
            ranking_loss_weight=hidden_value_residual_ranking_loss_weight,
            progress=progress,
        )
        data, recalibration_report = _recalibrate_expected_value_threshold(
            network,
            data,
            hidden,
            roots,
            selection_score_mode=selection_score_mode,
            validation_blocks=validation_blocks,
            minimum_validation_entries=minimum_validation_entries,
            allow_research_only=allow_research_only,
        )
        hidden_value_residual_report["threshold_recalibration"] = recalibration_report
        network.heads.seven_choice_expected_value_threshold_r = float(
            recalibration_report["selected_threshold"]
        )
    network.eval()
    split_masks = {
        name: np.asarray(data.splits == name) for name in ("train", "validation", "test")
    }
    before = {name: _head_metrics(network, data, hidden, roots, mask) for name, mask in split_masks.items()}
    optimizer = torch.optim.Adam([
        {"params": action_parameters, "lr": float(learning_rate) * float(action_learning_rate_multiplier)},
        {"params": auxiliary_parameters, "lr": float(learning_rate)},
    ])
    rng = np.random.default_rng(int(seed))
    train_indices = np.flatnonzero(split_masks["train"])
    class_counts = np.bincount(data.best_actions[train_indices], minlength=7)
    action_weights = np.ones(7, dtype=np.float32)
    hold_count = max(1, int(class_counts[int(Action.HOLD)]))
    for action in ENTRY_CHOICES:
        if class_counts[action] > 0:
            action_weights[action] = min(
                12.0, math.sqrt(hold_count / float(class_counts[action]))
            )
    action_weight_tensor = torch.as_tensor(
        action_weights, dtype=torch.float32, device=device
    )

    def live_action_logits(
        batch_indices: np.ndarray,
        hidden_batch: torch.Tensor,
        agent_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Build the exact action logits used by the live policy.

        The value surface is frozen during this stage, but the primary action
        heads are trainable.  Applying the same value boundary and residual
        here prevents cross-entropy from optimising a path that execution will
        immediately replace.
        """
        if action_decision_mode == "direct":
            logits = network.heads.direct_action_logits(
                hidden_batch,
                agent_batch,
                torch.as_tensor(
                    data.observations[batch_indices],
                    dtype=torch.float32,
                    device=device,
                ),
            )
        else:
            logits = network.heads.primary_action_logits(hidden_batch, agent_batch)
        if action_decision_mode == "direct":
            return logits
        with torch.no_grad():
            root_batch = torch.as_tensor(
                np.asarray(roots[batch_indices]), dtype=torch.float32, device=device
            )
            raw = network.observation_encoder.pattern_tree.action_path_head(root_batch)
            residual = getattr(
                network.observation_encoder.pattern_tree, "action_path_residual", None
            )
            if isinstance(residual, torch.nn.Module):
                raw = raw + residual(root_batch)
            observation_adapter = getattr(
                network.observation_encoder.pattern_tree,
                "action_path_observation_adapter", None,
            )
            if isinstance(observation_adapter, torch.nn.Module):
                bounded = torch.tanh(torch.nan_to_num(
                    torch.as_tensor(
                        data.observations[batch_indices],
                        dtype=torch.float32,
                        device=device,
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ))
                raw = raw + observation_adapter(bounded)
            raw = raw.reshape(-1, 6, 5)
            path = torch.cat((
                torch.sigmoid(raw[..., :2]),
                torch.tanh(raw[..., 2:3]),
                torch.sigmoid(raw[..., 3:]),
            ), dim=-1)
        return network._apply_seven_choice_expected_value_decision(
            logits, agent_batch, path, policy_hidden=hidden_batch
        )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] = {}
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_indices)
        loss_sum = 0.0
        rows_seen = 0
        for start in range(0, len(train_indices), batch_size):
            idx = train_indices[start:start + batch_size]
            h = torch.as_tensor(np.asarray(hidden[idx]), dtype=torch.float32, device=device)
            agents = torch.as_tensor(data.agent_ids[idx], dtype=torch.long, device=device)
            target_action = torch.as_tensor(data.best_actions[idx], dtype=torch.long, device=device)
            allowed = torch.as_tensor(data.allowed_action_mask[idx], dtype=torch.bool, device=device)
            logits = live_action_logits(idx, h, agents)[:, :7].masked_fill(
                ~allowed, -30.0
            )
            action_loss = F.cross_entropy(
                logits, target_action, weight=action_weight_tensor
            )
            entry = target_action > 0
            geometry_loss = action_loss.new_zeros(())
            if bool(entry.any()):
                selected = target_action[entry].detach().cpu().numpy()
                row_idx = idx[entry.detach().cpu().numpy()]
                targets = {
                    "delay": data.placement_delay_indices[row_idx, selected],
                    "entry_offset": data.entry_offset_indices[row_idx, selected],
                    "sl": data.sl_indices[row_idx, selected],
                    "tp": data.tp_indices[row_idx, selected],
                    "trailing": np.zeros(len(row_idx), dtype=np.int64),
                }
                for name, target in targets.items():
                    head_active = (
                        np.isin(selected, PENDING_CHOICES)
                        if name == "entry_offset" else np.ones(len(selected), dtype=np.bool_)
                    )
                    if not np.any(head_active):
                        continue
                    geometry_loss = geometry_loss + F.cross_entropy(
                        network.heads.parameter_heads[name](h[entry][torch.as_tensor(
                            head_active, dtype=torch.bool, device=device
                        )]),
                        torch.as_tensor(target[head_active], dtype=torch.long, device=device),
                    )
            loss = action_loss + 0.25 * geometry_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(idx)
            rows_seen += len(idx)
        validation = _head_metrics(network, data, hidden, roots, split_masks["validation"])
        validation_lcb = validation["closed_entry_realized_lcb_95_one_sided_block"]
        enough_entries = validation["closed_entries"] >= max(
            32, minimum_validation_entries // 2
        )
        validation_loss = (
            -float(validation_lcb) + 0.05 * validation["mean_causal_score_regret"]
            if validation_lcb is not None and enough_entries
            else 1e6 + validation["mean_causal_score_regret"]
        )
        row = {"epoch": float(epoch), "train_loss": loss_sum / max(1, rows_seen), **validation}
        history.append(row)
        if progress is not None:
            progress(dict(row))
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in network.state_dict().items()
                if name.startswith((
                    "heads.action_embedding.",
                    "heads.action_query.",
                    "heads.direct_action_head.",
                    "heads.direct_observation_head.",
                    "heads.expected_value_hidden_residual.",
                    "heads.entry_role_calibrator.",
                    "heads.parameter_heads.delay.",
                    "heads.parameter_heads.entry_offset.",
                    "heads.parameter_heads.sl.",
                    "heads.parameter_heads.tp.",
                    "heads.parameter_heads.trailing.",
                ))
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if not best_state:
        raise RuntimeError("khong tao duoc trang thai head tot nhat")
    current = network.state_dict()
    current.update(best_state)
    network.load_state_dict(current)
    after = {name: _head_metrics(network, data, hidden, roots, mask) for name, mask in split_masks.items()}
    invariant_after = _hash_except_seven_choice_heads(network)
    if invariant_after != invariant_before:
        raise RuntimeError("than mang hoac dau ra ngoai pham vi da bi thay doi")
    validation_lcb = after["validation"]["closed_entry_realized_lcb_95_one_sided_block"]
    test_lcb = after["test"]["closed_entry_realized_lcb_95_one_sided_block"]
    if not allow_research_only and (validation_lcb is None or validation_lcb <= 0.0):
        raise RuntimeError("dau hanh dong khong dat can duoi duong tren tap kiem dinh")
    if not allow_research_only and (test_lcb is None or test_lcb <= 0.0):
        raise RuntimeError("dau hanh dong khong dat can duoi duong tren tap thu kin")
    network.heads.seven_choice_imitation_trained = True
    network.heads.seven_choice_imitation_schema_version = SEVEN_CHOICE_DATASET_SCHEMA_VERSION
    network.heads.seven_choice_expected_value_decision_enabled = (
        action_decision_mode == "expected_value"
    )
    network.heads.seven_choice_action_decision_mode = action_decision_mode
    # Keep the threshold selected after the optional hidden-state recalibration.
    # The recalibration runs before action-head fine tuning and stores its value
    # on the head; replacing it here with the older root-only threshold would
    # silently reopen the over-entry problem at checkpoint finalization.
    final_threshold = float(
        getattr(
            network.heads,
            "seven_choice_expected_value_threshold_r",
            expected_value_report["selected_threshold"],
        )
        or expected_value_report["selected_threshold"]
    )
    network.heads.seven_choice_expected_value_threshold_r = final_threshold
    expected_value_report["final_threshold"] = final_threshold
    network.heads.seven_choice_expected_value_win_probability_floor = float(
        expected_value_report.get("win_probability_floor", 0.0)
    )
    geometry = data.manifest.get("config", {})
    network.heads.seven_choice_fixed_geometry_enabled = True
    network.heads.seven_choice_fixed_delay_index = int(
        geometry.get("fixed_delay_index", 0)
    )
    network.heads.seven_choice_fixed_entry_offset_index = int(
        data.manifest.get("causal_selection_audit", {}).get(
            "fixed_entry_offset_index", 7
        )
    )
    network.heads.seven_choice_fixed_sl_index = int(
        np.argmin(
            np.abs(
                np.asarray(SL_ATR_BINS, dtype=np.float64)
                - float(geometry.get("fixed_sl_atr", 1.5))
            )
        )
    )
    network.heads.seven_choice_fixed_tp_index = int(
        np.argmin(
            np.abs(
                np.asarray(TP_RR_BINS, dtype=np.float64)
                - float(geometry.get("fixed_tp_rr", 2.0))
            )
        )
    )
    network.heads.seven_choice_fixed_trailing_index = int(
        TRAILING_ATR_BINS.index(0.0)
    )
    network.heads.seven_choice_expected_value_logit_scale = 10.0
    network.heads.seven_choice_expected_value_decision_mode = (
        "direct_post_cost_net_r_v1"
        if action_decision_mode == "expected_value"
        else "direct_primary_action_v1"
    )
    network.heads.seven_choice_expected_value_score_mode = selection_score_mode
    network.heads.seven_choice_expected_value_fill_probability_floor = float(
        expected_value_report.get("fill_probability_floor", 0.0) or 0.0
    )
    network.heads.seven_choice_expected_value_action_residual_scale = float(
        0.5 if action_target_mode == "outcome" else 0.0
    )
    if not hidden_value_residual_enabled:
        network.heads.seven_choice_expected_value_hidden_residual_enabled = False
        network.heads.seven_choice_expected_value_hidden_residual_scale_r = 0.0
    network.heads.teacher_geometry_imitation_trained = True
    network.heads.teacher_geometry_imitation_schema_version = 2
    network.heads.teacher_geometry_imitation_heads = (
        "delay", "entry_offset", "sl", "tp", "trailing",
    )
    calibrator.teacher_imitation_trained = True
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    return {
        "scope": "frozen encoder, LSTM, Transformer and value; action, geometry and seven-choice output heads",
        "research_only": bool(
            allow_research_only
            or (
                action_decision_mode == "expected_value"
                and expected_value_report.get("research_only", False)
            )
        ),
        "action_target_mode": action_target_mode,
        "selection_score_mode": selection_score_mode,
        "ranking_loss_weight": float(ranking_loss_weight),
        "action_decision_mode": action_decision_mode,
        "runtime_action_source": (
            "primary_head_plus_direct_hidden_and_observation_residuals"
            if action_decision_mode == "direct"
            else "seven_choice_expected_value_surface"
        ),
        "live_action_residual_scale": float(
            getattr(network.heads, "seven_choice_expected_value_action_residual_scale", 0.0)
        ),
        "hidden_value_residual": hidden_value_residual_report,
        "best_epoch": best_epoch, "epochs_ran": len(history),
        "before": before, "after": after, "history": history,
        "expected_value_teacher": expected_value_report,
        "action_class_counts": class_counts.tolist(),
        "action_class_weights": action_weights.tolist(),
        "sequence_audit": sequence_audit,
        "frozen_invariant_sha256": invariant_after,
        "changed_heads": [
            "action_query", "action_embedding", "direct_action_head",
            "direct_observation_head",
            "entry_role_calibrator", "delay",
            "entry_offset", "sl", "tp", "trailing", "action_path",
            "action_path_residual", "action_path_observation_adapter",
            "expected_value_hidden_residual",
        ],
    }
