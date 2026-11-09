from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .execution_contract import (
    ACTION_DELAY_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    MANAGEMENT_MODE_OUTPUT_INDEX,
    PARTIAL_CLOSE_OUTPUT_INDEX,
    TARGET_SLOT_OUTPUT_INDEX,
    decode_execution_geometry,
)

from .config import OrderConfig, RiskConfig
from .features import sigmoid
from .protocol_contract import position_limit_enabled


# Graph bridge schema v4: twelve discrete EA commands plus eight parameters.
ACTION_COUNT = 12
PARAMETER_OFFSET = ACTION_COUNT
OUTPUT_COUNT = PARAMETER_OFFSET + 8

# Factorized target/management decoding is enabled only when output row 14
# carries a factorized execution-contract marker. Legacy vectors remain inert.
FACTORIZED_ACTION_MARKER_THRESHOLD = 4.5

# Shared simulator/bridge/EA execution contract. Pending offsets are a
# non-negative magnitude; order type determines direction. This removes the
# historical sign ambiguity where Python, bridge and MQL each decoded a
# different value.
MIN_PENDING_OFFSET_ATR = 0.45
MIN_MANUAL_CLOSE_AGE_BARS = 2
EMERGENCY_CLOSE_R = -0.75


class Action(IntEnum):
    HOLD = 0
    MARKET_BUY = 1
    MARKET_SELL = 2
    BUY_LIMIT = 3
    SELL_LIMIT = 4
    BUY_STOP = 5
    SELL_STOP = 6
    CANCEL_ORDERS = 7
    CLOSE_WORST = 8
    CLOSE_BEST = 9
    CLOSE_ALL = 10
    # Compatibility-only row in the 20-output layout. Fixed geometry masks it
    # from executable state; HOLD is intentionally a strict no-op.
    MANAGE_POSITION = 11


# Cac action nay chi co nghia khi snapshot truoc quyet dinh co it nhat mot
# vi the. MANAGE_POSITION bao phu toan bo mode quan ly: doi SL/TP, trailing,
# dong mot phan va dong toan phan.
POSITION_REQUIRED_ACTIONS = frozenset({
    Action.CLOSE_WORST,
    Action.CLOSE_BEST,
    Action.CLOSE_ALL,
    Action.MANAGE_POSITION,
})


class PositionManagementMode(IntEnum):
    NONE = 0
    MOVE_TO_BREAKEVEN = 1
    MODIFY_SL_TP = 2
    PARTIAL_CLOSE = 3
    FULL_CLOSE = 4


@dataclass
class Position:
    side: int
    entry: float
    sl: float
    tp: float
    risk_pct: float
    risk_amount: float
    risk_distance: float
    size: float
    opened_bar: int
    trailing_stop_atr: float = 0.0
    peak_price: float = 0.0
    adverse_price: float = 0.0
    # Exact causal source that created this exposure. Management may attach a
    # separate trace, but terminal PnL is never assigned to an unrelated
    # "latest" decision.
    origin_trace_id: str = ""
    management_trace_id: str = ""
    management_anchor_r: float = 0.0
    # Immutable ATR sampled when the position was opened. Learned trailing
    # distance is expressed against this value in Python, Model 1 and MT5 so
    # volatility changes after entry cannot make the executors diverge.
    initial_atr: float = 0.0
    trailing_applied: bool = False
    lifecycle_final_close: bool = True
    pending_fill_bars: int | None = None


@dataclass
class PendingOrder:
    action: Action
    side: int
    entry: float
    sl_atr: float
    tp_atr: float
    risk_pct: float
    created_bar: int
    trailing_stop_atr: float = 0.0
    origin_trace_id: str = ""


@dataclass(frozen=True)
class DecodedDecision:
    action: Action
    risk_pct: float
    sl_atr: float
    tp_atr: float
    entry_offset_atr: float
    trailing_stop_atr: float = 0.0
    partial_close_pct: float = 0.0
    time_filter_hour_start: float = 0.0
    time_filter_hour_end: float = 24.0
    position_scale: float = 0.0
    target_slot: int = 0
    management_mode: PositionManagementMode = PositionManagementMode.NONE
    action_delay_ticks: int = 0
    factorized_contract: bool = False

    @property
    def rr(self) -> float:
        return self.tp_atr / self.sl_atr


def action_schema_for_outputs(outputs: list[float] | np.ndarray) -> tuple[int, int, bool]:
    """Return the sole graph action schema.

    The bool is kept for call-site compatibility and is always ``False``.
    Vectors shorter than 20 are padded by the bridge normalizer; policy cores
    are always initialized with the full graph schema.
    """
    size = int(np.asarray(outputs, dtype=float).reshape(-1).size)
    # The bridge normalizes production policies to 20 outputs.  Keeping the
    # action head bounded by the available vector makes small unit-test /
    # diagnostic policies safe without reintroducing an alternative action schema.
    return min(ACTION_COUNT, max(0, size)), PARAMETER_OFFSET, False


def _at(values: np.ndarray, index: int, default: float = 0.0) -> float:
    return float(values[index]) if 0 <= index < values.size else float(default)


def canonical_pending_offset_atr(raw_value: float, configured_cap_atr: float) -> float:
    """Decode the one pending-offset parameter used by all execution paths.

    The neural output expresses magnitude only. Direction is determined solely
    by the selected pending order type. A zero/negative configured cap disables
    distance expansion without inventing a sign.
    """
    cap = max(0.0, float(configured_cap_atr))
    if cap <= 0.0:
        return 0.0
    raw = float(raw_value)
    if not np.isfinite(raw):
        raw = 0.0
    magnitude = abs(sigmoid(raw) * 2.0 - 1.0) * cap
    floor = min(MIN_PENDING_OFFSET_ATR, cap)
    return float(np.clip(max(floor, magnitude), 0.0, cap))


def pending_entry_bid_basis(action: Action, reference_bid: float, offset_atr: float, atr: float) -> float:
    """Return the canonical bid-basis entry price for every pending type."""
    offset = max(0.0, float(offset_atr)) * max(0.0, float(atr))
    if action in {Action.BUY_LIMIT, Action.SELL_STOP}:
        return float(reference_bid) - offset
    if action in {Action.SELL_LIMIT, Action.BUY_STOP}:
        return float(reference_bid) + offset
    raise ValueError(f"{action!r} is not a pending action")


def manual_position_actions_allowed(
    ages_bars: list[float] | tuple[float, ...],
    unrealized_rs: list[float] | tuple[float, ...],
    *,
    min_manual_close_age_bars: int = MIN_MANUAL_CLOSE_AGE_BARS,
    emergency_close_r: float = EMERGENCY_CLOSE_R,
) -> bool:
    """Single action-mask gate shared by simulator and bridge.

    A manual close/manage is allowed only after the common age threshold or
    when the selected position is in the documented emergency-loss state.
    SL/TP/trailing are unaffected and remain active immediately.
    """
    for age, current_r in zip(ages_bars, unrealized_rs):
        if float(age) >= float(min_manual_close_age_bars) or float(current_r) <= float(emergency_close_r):
            return True
    return False


def allowed_actions_for_state(
    *,
    position_count: int,
    pending_count: int,
    max_positions: int,
    ages_bars: list[float] | tuple[float, ...] = (),
    unrealized_rs: list[float] | tuple[float, ...] = (),
    min_manual_close_age_bars: int = MIN_MANUAL_CLOSE_AGE_BARS,
    emergency_close_r: float = EMERGENCY_CLOSE_R,
) -> set[Action]:
    """Canonical action mask for optional bounded or unlimited execution.

    This deliberately has no trade-frequency, drawdown or topology quota. It
    only expresses an explicitly configured execution quota and the shared
    manual-close rule. ``max_positions=0`` leaves every entry action available.
    """
    actions: set[Action] = {Action.HOLD}
    if (
        not position_limit_enabled(max_positions)
        or int(position_count) + int(pending_count) < int(max_positions)
    ):
        actions.update({
            Action.MARKET_BUY, Action.MARKET_SELL,
            Action.BUY_LIMIT, Action.SELL_LIMIT,
            Action.BUY_STOP, Action.SELL_STOP,
        })
    if int(pending_count) > 0:
        actions.add(Action.CANCEL_ORDERS)
    if int(position_count) > 0 and manual_position_actions_allowed(
        ages_bars, unrealized_rs,
        min_manual_close_age_bars=min_manual_close_age_bars,
        emergency_close_r=emergency_close_r,
    ):
        actions.update({Action.CLOSE_WORST, Action.CLOSE_BEST, Action.CLOSE_ALL})
    return actions


def decode_outputs(
    outputs: list[float] | np.ndarray,
    risk: RiskConfig,
    orders: OrderConfig,
    allowed_actions: set[Action] | None = None,
    max_positions: int | None = None,
) -> DecodedDecision:
    values = np.asarray(outputs, dtype=float).reshape(-1)
    action_count, parameter_offset, _ = action_schema_for_outputs(values)
    if action_count <= 0:
        action = Action.HOLD
    else:
        action_scores = values[:action_count].copy()
        if allowed_actions is not None:
            allowed_indexes = {int(item) for item in allowed_actions if 0 <= int(item) < action_count}
            if not allowed_indexes:
                allowed_indexes = {int(Action.HOLD)}
            for idx in range(action_scores.size):
                if idx not in allowed_indexes:
                    action_scores[idx] = -np.inf
        action = Action(int(np.argmax(action_scores)))

    geometry = decode_execution_geometry(
        values,
        min_risk_pct=float(risk.min_risk_pct),
        max_risk_pct=float(risk.max_risk_pct),
        technical_min_rr=float(orders.technical_min_rr),
        fixed_sl_atr=float(orders.fixed_sl_atr),
        fixed_tp_atr=float(orders.fixed_tp_atr),
        fixed_risk_pct=float(risk.fixed_risk_pct) if float(risk.fixed_risk_pct) > 0.0 else None,
        parameter_offset=parameter_offset,
    )
    risk_pct = geometry.risk_pct
    sl_atr = geometry.sl_atr
    tp_atr = geometry.tp_atr
    entry_offset_atr = canonical_pending_offset_atr(_at(values, parameter_offset + 3), orders.entry_offset_atr)

    # Contract v6 decodes the three entry-geometry heads together.  Trailing
    # remains attached to the creating entry; HOLD never mutates it.
    trailing_stop_atr = geometry.trailing_stop_atr
    partial_close_pct = 0.0
    time_filter_hour_start = 0.0
    time_filter_hour_end = 24.0
    position_scale = 0.0
    management_mode = PositionManagementMode.NONE
    action_delay_ticks = 0
    target_slot = 0
    factorized_contract = bool(
        values.size > FACTORIZED_ACTION_MARKER_OUTPUT_INDEX
        and _at(values, FACTORIZED_ACTION_MARKER_OUTPUT_INDEX) >= FACTORIZED_ACTION_MARKER_THRESHOLD
    )
    if factorized_contract:
        action_delay_ticks = min(3, max(0, int(sigmoid(_at(values, ACTION_DELAY_OUTPUT_INDEX)) * 4.0)))
        if max_positions is not None and max_positions > 0:
            target_slot = min(
                max_positions - 1,
                max(0, int(sigmoid(_at(values, TARGET_SLOT_OUTPUT_INDEX)) * max_positions)),
            )
        manage_bin = min(2, max(0, int(sigmoid(_at(values, MANAGEMENT_MODE_OUTPUT_INDEX)) * 3.0)))
        if manage_bin == 1:
            management_mode = PositionManagementMode.PARTIAL_CLOSE
        elif manage_bin == 2:
            management_mode = PositionManagementMode.FULL_CLOSE
        close_bin = min(3, max(0, int(sigmoid(_at(values, PARTIAL_CLOSE_OUTPUT_INDEX)) * 4.0)))
        partial_close_pct = 0.25 * float(close_bin + 1)

    return DecodedDecision(
        action, risk_pct, sl_atr, tp_atr, entry_offset_atr,
        trailing_stop_atr, partial_close_pct,
        time_filter_hour_start, time_filter_hour_end,
        position_scale, target_slot, management_mode,
        action_delay_ticks, factorized_contract,
    )


def build_sl_tp(side: int, entry: float, atr: float, sl_atr: float, tp_atr: float) -> tuple[float, float, float]:
    risk_distance = sl_atr * atr
    reward_distance = tp_atr * atr
    if side == 1:
        return entry - risk_distance, entry + reward_distance, risk_distance
    return entry + risk_distance, entry - reward_distance, risk_distance


def conservative_exit(
    position: Position,
    high_bid: float,
    low_bid: float,
    close_bid: float,
    spread_price: float,
) -> tuple[bool, float, str]:
    if position.side == 1:
        sl_hit = low_bid <= position.sl
        tp_hit = high_bid >= position.tp
        if sl_hit:
            # Nếu SL = entry (breakeven), return reason "be"
            if abs(position.sl - position.entry) < position.risk_distance * 0.01:
                return True, position.sl, "be"
            return True, position.sl, "sl"
        if tp_hit:
            return True, position.tp, "tp"
        return False, close_bid, "open"

    high_ask = high_bid + spread_price
    low_ask = low_bid + spread_price
    close_ask = close_bid + spread_price
    sl_hit = high_ask >= position.sl
    tp_hit = low_ask <= position.tp
    if sl_hit:
        if abs(position.sl - position.entry) < position.risk_distance * 0.01:
            return True, position.sl, "be"
        return True, position.sl, "sl"
    if tp_hit:
        return True, position.tp, "tp"
    return False, close_ask, "open"


def pnl_for_exit(position: Position, exit_price: float) -> float:
    if position.side == 1:
        return (exit_price - position.entry) * position.size
    return (position.entry - exit_price) * position.size
