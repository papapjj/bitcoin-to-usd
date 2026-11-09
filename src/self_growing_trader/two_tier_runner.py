"""
src/self_growing_trader/two_tier_runner.py

Two-Tier Trailing Runner Position Manager Engine.
Implements Requirement R3:
- Tier 1: Upon reaching 10.0 ATR unrealized excursion, execute a 50% partial close
  (ACTION_MANAGE_POSITION = 11, mode = 1, close_bin = 1) to bank secure profit into realized equity.
- Tier 2: Relocate Stop Loss of remaining 50% runner to protective profit (+1.0 ATR)
  and activate Chandelier 3.0 ATR trailing runner to capture extended 25 - 40 ATR market trends.
- Ratchet Invariant: Trailing Stop Loss can only tighten in the profit direction,
  never loosen or retreat.
- Robustness: Safe non-finite/non-positive ATR handling, idempotency, multi-position isolation.
- Strictly adheres to POLICY_CLOSE_ALL == 0 (Governor locked to HOLD, natural exits only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Union


# --- Wire Protocol & Action Constants ---
ACTION_HOLD: int = 0
ACTION_MANAGE_POSITION: int = 11

MANAGEMENT_MODE_NONE: int = 0
MANAGEMENT_MODE_PARTIAL_CLOSE: int = 1
MANAGEMENT_MODE_BREAKEVEN: int = 2
MANAGEMENT_MODE_MODIFY_SLTP: int = 3

CLOSE_BIN_25_PCT: int = 0
CLOSE_BIN_50_PCT: int = 1
CLOSE_BIN_75_PCT: int = 2
CLOSE_BIN_100_PCT: int = 3

DEFAULT_PARTIAL_CLOSE_BIN: int = CLOSE_BIN_50_PCT
DEFAULT_PARTIAL_CLOSE_RATIO: float = 0.50
DEFAULT_PARTIAL_TP_ATR: float = 10.0
DEFAULT_BREAKEVEN_SL_ATR: float = 1.0
DEFAULT_TRAILING_ATR: float = 3.0
DEFAULT_BREAKEVEN_TRIGGER_ATR: float = 2.0
DEFAULT_BREAKEVEN_LOCK_ATR: float = 0.5


class PositionPhase(str, Enum):
    """Lifecycle phases for two-tier position management."""
    INITIAL = "INITIAL"
    BREAKEVEN_LOCKED = "BREAKEVEN_LOCKED"
    PARTIAL_BANKED = "PARTIAL_BANKED"
    RUNNER_TRAILING = "RUNNER_TRAILING"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class TwoTierConfig:
    """Configuration parameters for TwoTierTrailingRunnerEngine."""
    partial_tp_atr: float = DEFAULT_PARTIAL_TP_ATR          # Threshold to trigger partial close (10.0 ATR)
    partial_close_ratio: float = DEFAULT_PARTIAL_CLOSE_RATIO  # Volume fraction to close (50%)
    breakeven_sl_atr: float = DEFAULT_BREAKEVEN_SL_ATR      # Protective profit lock (+1.0 ATR)
    trailing_atr: float = DEFAULT_TRAILING_ATR              # Chandelier trailing distance (3.0 ATR)
    immediate_trailing_at_trigger: bool = False              # If True, immediately trails if trailing_sl > breakeven_sl
    initial_sl_atr: float = 5.0                             # Default initial SL distance in ATR before Tier 2
    breakeven_trigger_atr: float = DEFAULT_BREAKEVEN_TRIGGER_ATR  # Early breakeven ratchet trigger (+2.0 ATR)
    breakeven_lock_atr: float = DEFAULT_BREAKEVEN_LOCK_ATR  # Profit locked at breakeven (+0.5 ATR)
    trail_after_breakeven: bool = False                      # If True, trails Chandelier ATR after breakeven lock

    def __post_init__(self) -> None:
        """Validate configuration parameters strictly upon instantiation."""
        if not math.isfinite(self.partial_tp_atr) or self.partial_tp_atr <= 0.0:
            raise ValueError(f"partial_tp_atr must be positive and finite, got {self.partial_tp_atr}")
        if not math.isfinite(self.partial_close_ratio) or not (0.0 < self.partial_close_ratio <= 1.0):
            raise ValueError(f"partial_close_ratio must be in (0, 1], got {self.partial_close_ratio}")
        if not math.isfinite(self.breakeven_sl_atr) or self.breakeven_sl_atr < 0.0:
            raise ValueError(f"breakeven_sl_atr must be non-negative and finite, got {self.breakeven_sl_atr}")
        if not math.isfinite(self.trailing_atr) or self.trailing_atr <= 0.0:
            raise ValueError(f"trailing_atr must be positive and finite, got {self.trailing_atr}")
        if not math.isfinite(self.breakeven_trigger_atr) or self.breakeven_trigger_atr < 0.0:
            raise ValueError(f"breakeven_trigger_atr must be non-negative and finite, got {self.breakeven_trigger_atr}")
        if not math.isfinite(self.breakeven_lock_atr) or self.breakeven_lock_atr < 0.0:
            raise ValueError(f"breakeven_lock_atr must be non-negative and finite, got {self.breakeven_lock_atr}")


@dataclass(frozen=True)
class RunnerConfig:
    """Legacy configuration for TwoTierRunnerManager."""
    initial_tp_atr: float = DEFAULT_PARTIAL_TP_ATR
    initial_sl_atr: float = 5.0
    bank_fraction: float = DEFAULT_PARTIAL_CLOSE_RATIO
    breakeven_lock_atr: float = DEFAULT_BREAKEVEN_SL_ATR
    chandelier_trail_atr: float = DEFAULT_TRAILING_ATR
    max_holding_bars: int = 120

    @property
    def partial_tp_atr(self) -> float:
        return self.initial_tp_atr

    @property
    def partial_close_ratio(self) -> float:
        return self.bank_fraction

    @property
    def breakeven_sl_atr(self) -> float:
        return self.breakeven_lock_atr

    @property
    def trailing_atr(self) -> float:
        return self.chandelier_trail_atr


@dataclass
class PositionState:
    """Per-position state tracker for TwoTierTrailingRunnerEngine.

    Maintains lifecycle flags, historical excursion peaks/troughs,
    and the monotonic ratcheted trailing stop level.
    """
    # Legacy phase string constants for full backwards compatibility
    INITIAL: ClassVar[str] = "INITIAL"
    BREAKEVEN_LOCKED: ClassVar[str] = "BREAKEVEN_LOCKED"
    PARTIAL_BANKED: ClassVar[str] = "PARTIAL_BANKED"
    RUNNER_TRAILING: ClassVar[str] = "RUNNER_TRAILING"
    CLOSED: ClassVar[str] = "CLOSED"

    position_id: str
    entry_price: float
    direction: int                      # +1 for BUY, -1 for SELL
    entry_atr: float
    highest_favorable_price: float = 0.0
    lowest_favorable_price: float = 0.0
    partial_close_executed: bool = False
    breakeven_executed: bool = False
    sl_moved_to_profit: bool = False
    trailing_active: bool = False
    current_sl: Optional[float] = None

    # Auxiliary and telemetry fields
    unrealized_atr: float = 0.0
    last_price: float = 0.0
    bars_held: int = 0
    close_executed_step: int = 0

    def __post_init__(self) -> None:
        self.position_id = str(self.position_id)
        self.entry_price = float(self.entry_price)
        self.direction = 1 if int(self.direction) > 0 else -1
        self.entry_atr = float(self.entry_atr) if (math.isfinite(self.entry_atr) and self.entry_atr > 0.0) else 0.0

        if self.highest_favorable_price <= 0.0:
            self.highest_favorable_price = self.entry_price
        if self.lowest_favorable_price <= 0.0:
            self.lowest_favorable_price = self.entry_price
        if self.last_price <= 0.0:
            self.last_price = self.entry_price

    @property
    def is_long(self) -> bool:
        """True if trade direction is BUY (+1)."""
        return self.direction > 0

    @property
    def is_short(self) -> bool:
        """True if trade direction is SELL (-1)."""
        return self.direction < 0

    @property
    def ticket(self) -> str:
        """Ticket alias for position_id."""
        return self.position_id

    @property
    def atr(self) -> float:
        """Entry ATR alias."""
        return self.entry_atr

    @property
    def phase(self) -> PositionPhase:
        """Returns the current high-level position lifecycle phase."""
        if not self.partial_close_executed:
            if self.breakeven_executed:
                return PositionPhase.BREAKEVEN_LOCKED
            return PositionPhase.INITIAL
        elif self.trailing_active:
            return PositionPhase.RUNNER_TRAILING
        elif self.sl_moved_to_profit:
            return PositionPhase.PARTIAL_BANKED
        return PositionPhase.INITIAL


class TwoTierTrailingRunnerEngine:
    """Two-Tier Trailing Runner Position Management Engine.

    Evaluates open positions bar-by-bar or tick-by-tick:
    1. Computes directional excursion in units of ATR:
       - BUY: (current_price - entry_price) / entry_atr
       - SELL: (entry_price - current_price) / entry_atr
    2. Tier 1 Trigger: When unrealized_atr >= 10.0 and not partial_close_executed:
       - Emits partial close command (close_ratio = 0.50, action = 11, mode = 1, close_bin = 1)
       - Sets partial_close_executed = True
    3. Tier 2 Trigger:
       - Moves SL to +1.0 ATR profit (BUY: entry + 1.0 * atr; SELL: entry - 1.0 * atr)
       - Activates Chandelier 3.0 ATR trailing on the remaining 50% runner
       - Strictly enforces ratchet constraint: SL can only advance in profit direction, never loosen.
    """

    def __init__(self, config: Optional[TwoTierConfig] = None) -> None:
        self.config: TwoTierConfig = config if config is not None else TwoTierConfig()
        self._positions: Dict[str, PositionState] = {}
        self._step_counter: int = 0

    @property
    def positions(self) -> Mapping[str, PositionState]:
        """Read-only view of active position states."""
        return self._positions

    @property
    def active_position_count(self) -> int:
        """Count of active positions being tracked."""
        return len(self._positions)

    def get_position(self, position_id: Union[str, int]) -> Optional[PositionState]:
        """Retrieve state for a specific position ticket."""
        return self._positions.get(str(position_id))

    def get_position_state(self, position_id: Union[str, int]) -> Optional[PositionState]:
        """Retrieve state for a specific position ticket (alias)."""
        return self._positions.get(str(position_id))

    def has_position(self, position_id: Union[str, int]) -> bool:
        """Check whether position ID is actively tracked."""
        return str(position_id) in self._positions

    def register_position(
        self,
        position_id: Union[str, int],
        entry_price: float,
        direction: int,
        entry_atr: float,
        initial_sl: Optional[float] = None,
    ) -> PositionState:
        """Explicitly registers a new position into runner tracking."""
        pid = str(position_id)
        pos = PositionState(
            position_id=pid,
            entry_price=float(entry_price),
            direction=int(direction),
            entry_atr=float(entry_atr),
            current_sl=float(initial_sl) if (initial_sl is not None and math.isfinite(initial_sl)) else None,
        )
        # Calculate default initial SL if not provided
        if pos.current_sl is None and pos.entry_atr > 0.0:
            if pos.direction > 0:
                pos.current_sl = pos.entry_price - self.config.initial_sl_atr * pos.entry_atr
            else:
                pos.current_sl = pos.entry_price + self.config.initial_sl_atr * pos.entry_atr

        self._positions[pid] = pos
        return pos

    def close_position(self, position_id: Union[str, int]) -> Optional[PositionState]:
        """Cleans up and removes a closed position from internal state tracking."""
        pid = str(position_id)
        return self._positions.pop(pid, None)

    def reset(self) -> None:
        """Clears all active positions and resets internal step counter."""
        self._positions.clear()
        self._step_counter = 0

    def evaluate_position(
        self,
        position_id: Union[str, int],
        entry_price: float,
        current_price: float,
        direction: int,
        entry_atr: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        current_bar: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates position state and excursion in ATR, generating management commands.

        Parameters:
            position_id: Unique ticket or position identifier
            entry_price: Order fill / entry price
            current_price: Current market price (close/bid)
            direction: +1 for BUY, -1 for SELL
            entry_atr: ATR value recorded at trade entry
            high: Optional bar High price (for bar-based evaluation)
            low: Optional bar Low price (for bar-based evaluation)
            current_bar: Optional current bar index

        Returns:
            Dict containing:
            - position_id: str
            - partial_close: bool (True only on step when Tier 1 fires)
            - close_ratio: float (0.50 if partial_close else 0.0)
            - action: int (11 ACTION_MANAGE_POSITION if partial_close else 0)
            - mode: int (1 Partial Close if partial_close else 0)
            - close_bin: int (1 50% ratio if partial_close else 0)
            - move_sl: bool (True when SL must be modified)
            - new_sl: Optional[float] (target SL price when move_sl is True)
            - current_sl: Optional[float] (latest SL price)
            - trailing_active: bool (True once Tier 2 is active)
            - unrealized_atr: float (current directional excursion in ATR)
            - highest_favorable_price: float (peak price reached for BUY)
            - lowest_favorable_price: float (trough price reached for SELL)
            - partial_close_executed: bool (state flag)
            - sl_moved_to_profit: bool (state flag)
            - stopped_out: bool (True if market price hit or breached current_sl)
            - exit_reason: Optional[str] (e.g., 'RUNNER_TRAILED_TP' or 'SL')
        """
        self._step_counter += 1
        pid = str(position_id)

        # 1. Input sanitization & safe handling of non-finite prices
        if not math.isfinite(entry_price) or entry_price <= 0.0:
            return self._build_safe_default_response(pid, 0.0, 0.0)
        if not math.isfinite(current_price) or current_price <= 0.0:
            return self._build_safe_default_response(pid, float(entry_price), 0.0)

        entry_p = float(entry_price)
        cur_p = float(current_price)
        dir_val = 1 if int(direction) > 0 else -1

        eval_high = float(high) if (high is not None and math.isfinite(high)) else cur_p
        eval_low = float(low) if (low is not None and math.isfinite(low)) else cur_p

        # 2. Retrieve or lazily initialize position state
        state = self._positions.get(pid)
        if state is None:
            valid_atr = float(entry_atr) if (math.isfinite(entry_atr) and entry_atr > 0.0) else 0.0
            state = PositionState(
                position_id=pid,
                entry_price=entry_p,
                direction=dir_val,
                entry_atr=valid_atr,
                highest_favorable_price=max(entry_p, eval_high),
                lowest_favorable_price=min(entry_p, eval_low),
            )
            if valid_atr > 0.0:
                state.current_sl = (entry_p - self.config.initial_sl_atr * valid_atr) if dir_val > 0 else (entry_p + self.config.initial_sl_atr * valid_atr)
            self._positions[pid] = state
        else:
            # Update entry ATR if it was previously non-positive and now has valid value
            if state.entry_atr <= 0.0 and math.isfinite(entry_atr) and entry_atr > 0.0:
                state.entry_atr = float(entry_atr)

        state.last_price = cur_p
        state.bars_held += 1

        # 3. Update favorable price watermarks
        if state.direction > 0:  # BUY
            if eval_high > state.highest_favorable_price:
                state.highest_favorable_price = eval_high
        else:  # SELL
            if eval_low < state.lowest_favorable_price:
                state.lowest_favorable_price = eval_low

        # 4. Safe Unrealized ATR Excursion Calculation
        effective_atr = state.entry_atr
        if effective_atr <= 0.0 or not math.isfinite(effective_atr):
            unrealized_atr = 0.0
            peak_unrealized_atr = 0.0
        else:
            if state.direction > 0:  # BUY
                unrealized_atr = (cur_p - state.entry_price) / effective_atr
                peak_unrealized_atr = (eval_high - state.entry_price) / effective_atr
            else:  # SELL
                unrealized_atr = (state.entry_price - cur_p) / effective_atr
                peak_unrealized_atr = (state.entry_price - eval_low) / effective_atr

        state.unrealized_atr = unrealized_atr

        # 5. Transition & Trigger Logic
        partial_close = False
        close_ratio = 0.0
        action = ACTION_HOLD
        mode = MANAGEMENT_MODE_NONE
        close_bin = 0

        move_sl = False
        new_sl: Optional[float] = None

        can_trigger = (effective_atr > 0.0 and math.isfinite(effective_atr))

        # --- Tier 1 Trigger & Tier 2 Activation ---
        # When unrealized_atr >= 10.0 and not partial_close_executed
        if can_trigger and not state.partial_close_executed and peak_unrealized_atr >= self.config.partial_tp_atr:
            # Emit partial close command (50% ratio)
            partial_close = True
            close_ratio = self.config.partial_close_ratio
            action = ACTION_MANAGE_POSITION
            mode = MANAGEMENT_MODE_PARTIAL_CLOSE
            close_bin = DEFAULT_PARTIAL_CLOSE_BIN
            state.partial_close_executed = True
            state.close_executed_step = self._step_counter

            # Tier 2: Relocate SL to protective profit (+1.0 ATR) and activate Chandelier Trailing
            state.sl_moved_to_profit = True
            state.trailing_active = True

            if state.direction > 0:  # BUY
                breakeven_sl = state.entry_price + self.config.breakeven_sl_atr * effective_atr
                if self.config.immediate_trailing_at_trigger:
                    trailing_candidate = state.highest_favorable_price - self.config.trailing_atr * effective_atr
                    target_sl = max(breakeven_sl, trailing_candidate)
                else:
                    target_sl = breakeven_sl
            else:  # SELL
                breakeven_sl = state.entry_price - self.config.breakeven_sl_atr * effective_atr
                if self.config.immediate_trailing_at_trigger:
                    trailing_candidate = state.lowest_favorable_price + self.config.trailing_atr * effective_atr
                    target_sl = min(breakeven_sl, trailing_candidate)
                else:
                    target_sl = breakeven_sl

            state.current_sl = target_sl
            move_sl = True
            new_sl = target_sl

        # --- Tier 2 Trailing Runner Phase & Post-Breakeven Trailing ---
        # Subsequent steps after Tier 2 has been activated or after breakeven locked: trail Chandelier ATR
        elif can_trigger and (state.trailing_active or (self.config.trail_after_breakeven and state.breakeven_executed)):
            # Idempotency: partial close already executed, never repeat
            partial_close = False
            close_ratio = 0.0
            action = ACTION_HOLD

            lock_floor = self.config.breakeven_lock_atr if (self.config.trail_after_breakeven and not state.partial_close_executed) else self.config.breakeven_sl_atr

            if state.direction > 0:  # BUY
                trailing_sl = state.highest_favorable_price - self.config.trailing_atr * effective_atr
                # Protected profit floor
                protective_sl = state.entry_price + lock_floor * effective_atr
                candidate_sl = max(protective_sl, trailing_sl)

                # Ratchet constraint: Trailing SL can only tighten in profit direction (increase for BUY)
                if state.current_sl is None or candidate_sl > state.current_sl:
                    state.current_sl = candidate_sl
                    move_sl = True
                    new_sl = candidate_sl

            else:  # SELL
                trailing_sl = state.lowest_favorable_price + self.config.trailing_atr * effective_atr
                # Protected profit floor
                protective_sl = state.entry_price - lock_floor * effective_atr
                candidate_sl = min(protective_sl, trailing_sl)

                # Ratchet constraint: Trailing SL can only tighten in profit direction (decrease for SELL)
                if state.current_sl is None or candidate_sl < state.current_sl:
                    state.current_sl = candidate_sl
                    move_sl = True
                    new_sl = candidate_sl

        # --- Early Breakeven Ratchet Phase ---
        # When unrealized peak >= breakeven_trigger_atr and not yet breakeven_executed
        elif (
            can_trigger
            and self.config.breakeven_trigger_atr > 0.0
            and not state.breakeven_executed
            and not state.sl_moved_to_profit
            and peak_unrealized_atr >= self.config.breakeven_trigger_atr
        ):
            state.breakeven_executed = True
            state.sl_moved_to_profit = True
            action = ACTION_HOLD
            mode = MANAGEMENT_MODE_BREAKEVEN

            if state.direction > 0:  # BUY
                be_sl = state.entry_price + self.config.breakeven_lock_atr * effective_atr
                if state.current_sl is None or be_sl > state.current_sl:
                    state.current_sl = be_sl
                    move_sl = True
                    new_sl = be_sl
            else:  # SELL
                be_sl = state.entry_price - self.config.breakeven_lock_atr * effective_atr
                if state.current_sl is None or be_sl < state.current_sl:
                    state.current_sl = be_sl
                    move_sl = True
                    new_sl = be_sl

        # 6. Stopped Out Detection
        stopped_out = False
        exit_reason: Optional[str] = None
        if state.current_sl is not None:
            if state.direction > 0:  # BUY
                if eval_low <= state.current_sl:
                    stopped_out = True
                    exit_reason = "RUNNER_TRAILED_TP" if state.trailing_active else "SL"
            else:  # SELL
                if eval_high >= state.current_sl:
                    stopped_out = True
                    exit_reason = "RUNNER_TRAILED_TP" if state.trailing_active else "SL"

        return {
            "position_id": state.position_id,
            "partial_close": partial_close,
            "close_ratio": close_ratio,
            "action": action,
            "mode": mode,
            "close_bin": close_bin,
            "move_sl": move_sl,
            "new_sl": new_sl,
            "current_sl": state.current_sl,
            "trailing_active": state.trailing_active,
            "unrealized_atr": float(unrealized_atr),
            "highest_favorable_price": float(state.highest_favorable_price),
            "lowest_favorable_price": float(state.lowest_favorable_price),
            "partial_close_executed": state.partial_close_executed,
            "breakeven_executed": state.breakeven_executed,
            "sl_moved_to_profit": state.sl_moved_to_profit,
            "stopped_out": stopped_out,
            "exit_reason": exit_reason,
        }

    def _build_safe_default_response(
        self,
        position_id: str,
        entry_price: float,
        unrealized_atr: float,
    ) -> Dict[str, Any]:
        """Produces a fail-closed, safe no-action response when inputs are malformed."""
        return {
            "position_id": str(position_id),
            "partial_close": False,
            "close_ratio": 0.0,
            "action": ACTION_HOLD,
            "mode": MANAGEMENT_MODE_NONE,
            "close_bin": 0,
            "move_sl": False,
            "new_sl": None,
            "current_sl": None,
            "trailing_active": False,
            "unrealized_atr": unrealized_atr,
            "highest_favorable_price": entry_price,
            "lowest_favorable_price": entry_price,
            "partial_close_executed": False,
            "breakeven_executed": False,
            "sl_moved_to_profit": False,
            "stopped_out": False,
            "exit_reason": None,
        }


# =====================================================================
# Backward Compatibility Layer (TwoTierRunnerManager & ManagedPosition)
# =====================================================================

@dataclass
class ManagedPosition:
    """Legacy ManagedPosition wrapper for backward compatibility."""
    ticket: int
    is_long: bool
    entry_price: float
    current_volume: float
    atr: float
    entry_bar: int
    state: PositionPhase = PositionPhase.INITIAL
    realized_banked_profit: float = 0.0
    current_sl_price: float = 0.0
    best_favorable_price: float = 0.0
    closed_bar: Optional[int] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.best_favorable_price == 0.0:
            self.best_favorable_price = self.entry_price
        if self.current_sl_price == 0.0:
            dist = 5.0 * self.atr
            self.current_sl_price = self.entry_price - dist if self.is_long else self.entry_price + dist


class TwoTierRunnerManager:
    """Legacy position manager preserving backward compatibility with existing tests."""

    def __init__(self, config: Optional[RunnerConfig] = None) -> None:
        self.config: RunnerConfig = config if config is not None else RunnerConfig()
        self.active_positions: Dict[int, ManagedPosition] = {}
        self.closed_positions: List[ManagedPosition] = []

    def open_position(
        self,
        ticket: int,
        is_long: bool,
        entry_price: float,
        volume: float,
        atr: float,
        current_bar: int,
    ) -> ManagedPosition:
        """Registers a newly opened trade into runner tracking."""
        pos = ManagedPosition(
            ticket=ticket,
            is_long=is_long,
            entry_price=entry_price,
            current_volume=volume,
            atr=max(0.1, float(atr)),
            entry_bar=current_bar,
        )
        self.active_positions[ticket] = pos
        return pos

    def on_tick_or_bar(
        self,
        ticket: int,
        high: float,
        low: float,
        close: float,
        current_bar: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Updates position tracking on a bar/tick."""
        if ticket not in self.active_positions:
            return None, None, None

        pos = self.active_positions[ticket]
        atr = pos.atr
        bars_held = current_bar - pos.entry_bar

        # Check maximum holding limit (120 bars default)
        max_bars = getattr(self.config, "max_holding_bars", 120)
        if max_bars is not None and bars_held >= max_bars:
            pos.state = PositionPhase.CLOSED
            pos.closed_bar = current_bar
            pos.close_price = close
            pos.close_reason = "TIMEOUT"
            self.closed_positions.append(pos)
            del self.active_positions[ticket]
            return None, None, "TIMEOUT"

        partial_bank = None
        new_sl = None

        if pos.is_long:
            if high > pos.best_favorable_price:
                pos.best_favorable_price = high

            if pos.state == PositionPhase.INITIAL:
                if low <= pos.current_sl_price:
                    pos.state = PositionPhase.CLOSED
                    pos.closed_bar = current_bar
                    pos.close_price = pos.current_sl_price
                    pos.close_reason = "SL"
                    self.closed_positions.append(pos)
                    del self.active_positions[ticket]
                    return None, None, "SL"

                tp_price = pos.entry_price + self.config.partial_tp_atr * atr
                if high >= tp_price:
                    bank_vol = pos.current_volume * self.config.partial_close_ratio
                    pos.current_volume -= bank_vol
                    pos.realized_banked_profit = bank_vol * (self.config.partial_tp_atr * atr)
                    pos.state = PositionPhase.RUNNER_TRAILING
                    pos.current_sl_price = pos.entry_price + self.config.breakeven_sl_atr * atr
                    partial_bank = bank_vol
                    new_sl = pos.current_sl_price

            elif pos.state == PositionPhase.RUNNER_TRAILING:
                trailed_sl = pos.best_favorable_price - self.config.trailing_atr * atr
                if trailed_sl > pos.current_sl_price:
                    pos.current_sl_price = trailed_sl
                    new_sl = pos.current_sl_price

                if low <= pos.current_sl_price:
                    pos.state = PositionPhase.CLOSED
                    pos.closed_bar = current_bar
                    pos.close_price = pos.current_sl_price
                    pos.close_reason = "RUNNER_TRAILED_TP"
                    self.closed_positions.append(pos)
                    del self.active_positions[ticket]
                    return None, new_sl, "RUNNER_TRAILED_TP"

        else:  # SHORT
            if low < pos.best_favorable_price:
                pos.best_favorable_price = low

            if pos.state == PositionPhase.INITIAL:
                if high >= pos.current_sl_price:
                    pos.state = PositionPhase.CLOSED
                    pos.closed_bar = current_bar
                    pos.close_price = pos.current_sl_price
                    pos.close_reason = "SL"
                    self.closed_positions.append(pos)
                    del self.active_positions[ticket]
                    return None, None, "SL"

                tp_price = pos.entry_price - self.config.partial_tp_atr * atr
                if low <= tp_price:
                    bank_vol = pos.current_volume * self.config.partial_close_ratio
                    pos.current_volume -= bank_vol
                    pos.realized_banked_profit = bank_vol * (self.config.partial_tp_atr * atr)
                    pos.state = PositionPhase.RUNNER_TRAILING
                    pos.current_sl_price = pos.entry_price - self.config.breakeven_sl_atr * atr
                    partial_bank = bank_vol
                    new_sl = pos.current_sl_price

            elif pos.state == PositionPhase.RUNNER_TRAILING:
                trailed_sl = pos.best_favorable_price + self.config.trailing_atr * atr
                if trailed_sl < pos.current_sl_price:
                    pos.current_sl_price = trailed_sl
                    new_sl = pos.current_sl_price

                if high >= pos.current_sl_price:
                    pos.state = PositionPhase.CLOSED
                    pos.closed_bar = current_bar
                    pos.close_price = pos.current_sl_price
                    pos.close_reason = "RUNNER_TRAILED_TP"
                    self.closed_positions.append(pos)
                    del self.active_positions[ticket]
                    return None, new_sl, "RUNNER_TRAILED_TP"

        return partial_bank, new_sl, None
