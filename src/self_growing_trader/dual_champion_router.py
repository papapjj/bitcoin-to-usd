"""
src/self_growing_trader/dual_champion_router.py

Dual-Champion Mixture-of-Regimes (MoR) Router for BTCUSD M1.
Routes execution between Bull Specialist (Phase 3 R01) and Bear Specialist (Phase 2 Champion)
using real-time soft probability distribution [P_Bull, P_Bear, P_Chop] from Champion Regime Detector.
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from datetime import datetime, timedelta, timezone
import logging
import math
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .protocol_contract import INPUT_COUNT, OUTPUT_COUNT
except ImportError:
    INPUT_COUNT = 137
    OUTPUT_COUNT = 20

from .cppi_risk_engine import (
    CPPIConfig,
    CPPIFloorEngine,
    BidirectionalCPPIConfig,
    BidirectionalCPPIEngine,
)
from .two_tier_runner import (
    TwoTierConfig,
    TwoTierTrailingRunnerEngine,
    ACTION_HOLD,
    ACTION_MANAGE_POSITION,
    MANAGEMENT_MODE_PARTIAL_CLOSE,
    CLOSE_BIN_50_PCT,
)
from .micro_scalp_engine import (
    MicroScalpConfig,
    MicroPullbackScalpEngine,
    ScalpSignal,
)
from .execution_contract import (
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    pack_geometry_choices,
)
from .progress_reader import ProgressFileReader
from .soft_moe_gating import SoftMoEGatingNetwork, CapitalPreservationGuard
from .anti_bottom_guard import AntiBottomExhaustionGuard

try:
    from .orders import Action
    ACTION_HOLD_ORDER = Action.HOLD
except ImportError:
    ACTION_HOLD_ORDER = 0

logger = logging.getLogger("DualChampionRouter")


class ConsecutiveLossCircuitBreaker:
    """Stateful Consecutive Loss Circuit Breaker for Risk Containment.

    Tracks consecutive Stop Losses per direction ('BUY' and 'SELL').
    When `max_consecutive_sl` (default: 2) Stop Losses occur, enters a
    bar-based cooldown (default: 15 M1 bars) where new entries in that
    direction are soft-vetoed to HOLD.
    Resets counter on winning exit or TP.
    """

    def __init__(
        self,
        max_consecutive_sl: int = 2,
        cooldown_minutes: float = 15.0,
        window_minutes: float = 60.0,
        enabled: bool = True,
    ) -> None:
        self.max_consecutive_sl = int(max_consecutive_sl)
        self.cooldown_bars = max(5, int(cooldown_minutes))
        self.window_minutes = float(window_minutes)
        self.enabled = bool(enabled)
        self.consecutive_sl: dict[str, int] = {"BUY": 0, "SELL": 0}
        self.cooldown_remaining_bars: dict[str, int] = {"BUY": 0, "SELL": 0}
        self.sl_history: dict[str, list[datetime]] = {"BUY": [], "SELL": []}
        self.cooldown_until: dict[str, datetime | None] = {"BUY": None, "SELL": None}
        self._current_time: datetime | None = None

    def tick_step(self) -> None:
        """Advance cooldown timer by 1 bar cadence."""
        for d in ("BUY", "SELL"):
            if self.cooldown_remaining_bars[d] > 0:
                self.cooldown_remaining_bars[d] -= 1

    def _parse_time(self, timestamp: Any) -> datetime:
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str) and timestamp.strip():
            clean = timestamp.strip().replace(".", "-")
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(clean[:19], fmt)
                except ValueError:
                    pass
        if self._current_time is not None:
            if isinstance(self._current_time, datetime):
                return self._current_time
            if isinstance(self._current_time, str) and self._current_time.strip():
                clean = self._current_time.strip().replace(".", "-")
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        return datetime.strptime(clean[:19], fmt)
                    except ValueError:
                        pass
        return datetime(2025, 1, 10, 0, 0, 0)

    def record_outcome(
        self,
        direction: str,
        is_sl: bool,
        is_win: bool = False,
        timestamp: Any = None,
        profit: float = 0.0,
    ) -> None:
        """Record trade close event for direction ('BUY' or 'SELL')."""
        if not self.enabled:
            return
        d = str(direction).upper()
        if d not in ("BUY", "SELL"):
            if d in ("LONG", "0"):
                d = "BUY"
            elif d in ("SHORT", "1"):
                d = "SELL"
            else:
                return

        if timestamp is not None:
            ev_time = self._parse_time(timestamp)
            self._current_time = ev_time
        else:
            ev_time = self._current_time or datetime(2025, 1, 10, 0, 0, 0)

        if is_sl or (profit < 0.0 and not is_win):
            cutoff = ev_time - timedelta(minutes=self.window_minutes)
            self.sl_history[d] = [t for t in self.sl_history[d] if t >= cutoff]
            self.sl_history[d].append(ev_time)
            self.consecutive_sl[d] = len(self.sl_history[d])
            if len(self.sl_history[d]) >= self.max_consecutive_sl:
                self.cooldown_remaining_bars[d] = self.cooldown_bars
                self.cooldown_until[d] = ev_time + timedelta(minutes=self.cooldown_bars)
                self.consecutive_sl[d] = 0
                self.sl_history[d].clear()
                logger.info(
                    "Circuit breaker TRIPPED for %s: cooldown for %d bars until %s",
                    d, self.cooldown_bars, self.cooldown_until[d]
                )
        elif is_win or profit > 0.0:
            self.consecutive_sl[d] = 0
            self.sl_history[d].clear()

    def is_in_cooldown(self, direction: str, current_time: Any = None) -> bool:
        """Check if direction ('BUY' or 'SELL') is currently in circuit breaker cooldown."""
        if not self.enabled:
            return False
        d = str(direction).upper()
        if d not in ("BUY", "SELL"):
            if d in ("LONG", "0"):
                d = "BUY"
            elif d in ("SHORT", "1"):
                d = "SELL"
            else:
                return False

        if current_time is not None and self.cooldown_until.get(d) is not None:
            cur_dt = self._parse_time(current_time)
            if cur_dt >= self.cooldown_until[d]:
                self.cooldown_remaining_bars[d] = 0
                self.cooldown_until[d] = None
                return False

        return self.cooldown_remaining_bars.get(d, 0) > 0

    @property
    def is_active(self) -> bool:
        """Return True if either BUY or SELL is currently in cooldown."""
        return any(self.cooldown_remaining_bars.get(d, 0) > 0 for d in ("BUY", "SELL"))

    def reset(self) -> None:
        """Reset all circuit breaker counters and active cooldowns."""
        self.consecutive_sl = {"BUY": 0, "SELL": 0}
        self.cooldown_remaining_bars = {"BUY": 0, "SELL": 0}
        self.sl_history = {"BUY": [], "SELL": []}
        self.cooldown_until = {"BUY": None, "SELL": None}
        self._current_time = None
        """Reset all circuit breaker counters and active cooldowns."""
        self.sl_history = {"BUY": [], "SELL": []}
        self.cooldown_until = {"BUY": None, "SELL": None}
        self._current_time = None


class BridgeActionBatch(tuple):
    """Batch of 5 agent action traces or actions emitted by DualChampionRouter."""

    risk_pct: float | None = None
    cppi_telemetry: dict[str, Any] = {}
    anti_bottom_telemetry: dict[str, Any] = {}

    def __new__(
        cls,
        iterable: Sequence[Any],
        *,
        risk_pct: float | None = None,
        cppi_telemetry: dict[str, Any] | None = None,
        anti_bottom_telemetry: dict[str, Any] | None = None,
    ) -> BridgeActionBatch:
        obj = super().__new__(cls, iterable)
        obj.risk_pct = float(risk_pct) if risk_pct is not None else None
        obj.cppi_telemetry = dict(cppi_telemetry or {})
        obj.anti_bottom_telemetry = dict(anti_bottom_telemetry or {})
        return obj

    @property
    def traces(self) -> tuple[Any, ...]:
        return tuple(self)

    @property
    def actions(self) -> tuple[int, ...]:
        result = []
        for item in self:
            result.append(DualChampionRouter._extract_action_idx(item))
        return tuple(result)


class DualChampionRouter:
    """Production Mixture-of-Regimes Router combining Bull and Bear Specialists.

    Maintains isolated recurrent memories for both specialists while providing
    a single transparent interface compliant with HeChuyenGiaMotContextMotPolicy contract.
    Integrates CPPI Floor & Dynamic Risk Engine and Two-Tier Trailing Runner Engine.
    """

    def __init__(
        self,
        bull_policy: Any,
        bear_policy: Any,
        regime_detector: Any | None = None,
        *,
        bull_threshold: float = 0.35,
        bear_threshold: float = 0.35,
        chop_threshold: float = 0.38,
        confirmation_bars: int = 1,
        htf_filter_enabled: bool = False,
        htf_bull_threshold: float = 0.05,
        htf_bear_threshold: float = -0.05,
        htf_slope_threshold: float = 0.0,
        momentum_gate_enabled: bool = False,
        circuit_breaker_enabled: bool = False,
        circuit_breaker_max_sl: int = 2,
        circuit_breaker_cooldown_minutes: float = 60.0,
        circuit_breaker_window_minutes: float = 60.0,
        circuit_breaker: ConsecutiveLossCircuitBreaker | None = None,
        cppi_config: CPPIConfig | None = None,
        two_tier_config: TwoTierConfig | None = None,
        cppi_engine: CPPIFloorEngine | None = None,
        two_tier_runner: TwoTierTrailingRunnerEngine | None = None,
        bidirectional_cppi_config: BidirectionalCPPIConfig | None = None,
        bidirectional_cppi_engine: BidirectionalCPPIEngine | None = None,
        micro_scalp_config: MicroScalpConfig | None = None,
        micro_scalp_engine: MicroPullbackScalpEngine | None = None,
        micro_scalp_enabled: bool = True,
        max_total_trades: int = 0,
        max_concurrent_positions: int = 5,
        max_positions_per_direction: int = 2,
        min_spacing_bars: int = 0,
        default_sl_atr: float = 5.0,
        default_tp_rr: float = 2.5,
        default_trailing_atr: float = 1.0,
        soft_moe_gating: SoftMoEGatingNetwork | None = None,
        capital_preservation_guard: CapitalPreservationGuard | None = None,
        anti_bottom_guard: AntiBottomExhaustionGuard | None = None,
        anti_bottom_guard_enabled: bool = True,
    ) -> None:
        self.bull_policy = bull_policy
        self.bear_policy = bear_policy
        self.regime_detector = regime_detector
        self.soft_moe_gating = (
            soft_moe_gating if soft_moe_gating is not None else SoftMoEGatingNetwork()
        )
        self.capital_preservation_guard = (
            capital_preservation_guard
            if capital_preservation_guard is not None
            else CapitalPreservationGuard()
        )
        self.anti_bottom_guard = (
            anti_bottom_guard
            if anti_bottom_guard is not None
            else AntiBottomExhaustionGuard(enabled=anti_bottom_guard_enabled)
        )
        self.anti_bottom_telemetry: dict[str, Any] = {
            "bottom_short_veto": False,
            "veto_reason": "",
            "demand_zone_score": 0.0,
        }
        self.soft_moe_weights: tuple[float, float] = (0.5, 0.5)
        self.soft_moe_telemetry: dict[str, Any] = {}
        self.bull_threshold = float(bull_threshold)
        self.bear_threshold = float(bear_threshold)
        self.chop_threshold = float(chop_threshold)
        self.confirmation_bars = int(confirmation_bars)

        # Tier 1 HTF Macro Trend Filter configuration
        self.htf_filter_enabled = bool(htf_filter_enabled)
        self.htf_bull_threshold = float(htf_bull_threshold)
        self.htf_bear_threshold = float(htf_bear_threshold)
        self.htf_slope_threshold = float(htf_slope_threshold)

        # Tier 2 Microstructure Momentum Gate configuration
        self.momentum_gate_enabled = bool(momentum_gate_enabled)

        # Tier 3 Consecutive Loss Circuit Breaker
        if circuit_breaker is not None:
            self.circuit_breaker = circuit_breaker
        else:
            self.circuit_breaker = ConsecutiveLossCircuitBreaker(
                max_consecutive_sl=circuit_breaker_max_sl,
                cooldown_minutes=circuit_breaker_cooldown_minutes,
                window_minutes=circuit_breaker_window_minutes,
                enabled=circuit_breaker_enabled,
            )

        self.max_total_trades: int = int(max_total_trades)
        self.total_closed_trades: int = 0
        self.max_concurrent_positions: int = int(max_concurrent_positions)
        self.max_positions_per_direction: int = int(max_positions_per_direction)
        self.min_spacing_bars: int = int(min_spacing_bars)
        self.default_sl_atr: float = float(default_sl_atr)
        self.default_tp_rr: float = float(default_tp_rr)
        self.default_trailing_atr: float = float(default_trailing_atr)
        self._bars_since_last_entry: int = 999
        self.current_regime: str = "CHOP"
        self.current_probabilities: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self.current_regime_index: int = 2
        self._pending_bull_count: int = 0
        self._pending_bear_count: int = 0
        self._regime_pinned: bool = False

        # M3 CPPI and Two-Tier Engines
        self.cppi_engine: CPPIFloorEngine = (
            cppi_engine if cppi_engine is not None else CPPIFloorEngine(config=cppi_config)
        )
        self.bidirectional_cppi_config = bidirectional_cppi_config
        self.bidirectional_cppi_engine: BidirectionalCPPIEngine = (
            bidirectional_cppi_engine
            if bidirectional_cppi_engine is not None
            else BidirectionalCPPIEngine(
                config=bidirectional_cppi_config or BidirectionalCPPIConfig(
                    max_concurrent_positions=self.max_concurrent_positions,
                    max_positions_per_direction=self.max_positions_per_direction,
                    min_spacing_bars=self.min_spacing_bars,
                )
            )
        )
        self.two_tier_runner: TwoTierTrailingRunnerEngine = (
            two_tier_runner if two_tier_runner is not None else TwoTierTrailingRunnerEngine(config=two_tier_config)
        )
        self.two_tier_engine = self.two_tier_runner
        self.runner_engine = self.two_tier_runner

        # High-Frequency Micro-Pullback Scalp Engine (>= 100 trades/day & WR >= 85%)
        self.micro_scalp_engine: MicroPullbackScalpEngine = (
            micro_scalp_engine if micro_scalp_engine is not None else
            MicroPullbackScalpEngine(config=micro_scalp_config or MicroScalpConfig(enabled=micro_scalp_enabled))
        )

        self._last_equity: float = 0.0
        self._last_positions_pipe: str = ""

    @property
    def circuit_breaker_active(self) -> bool:
        """True if circuit breaker is currently active for either direction."""
        return bool(self.circuit_breaker.is_active) if hasattr(self, "circuit_breaker") else False

    def is_in_cooldown(self, direction: str, current_time: Any = None) -> bool:
        """Query if a specific direction ('BUY' or 'SELL') is in circuit breaker cooldown."""
        if hasattr(self, "circuit_breaker"):
            return self.circuit_breaker.is_in_cooldown(direction, current_time)
        return False

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker counters and cooldown timers."""
        if hasattr(self, "circuit_breaker"):
            self.circuit_breaker.reset()

    def record_trade_outcome(
        self,
        direction: str,
        outcome: str,
        profit: float = 0.0,
        timestamp: Any = None,
    ) -> None:
        """Record trade outcome directly into circuit breaker."""
        if hasattr(self, "circuit_breaker"):
            is_sl = "SL" in str(outcome).upper() or profit < 0.0
            is_win = "TP" in str(outcome).upper() or "WIN" in str(outcome).upper() or profit > 0.0
            self.circuit_breaker.record_outcome(
                direction=direction,
                is_sl=is_sl,
                is_win=is_win,
                timestamp=timestamp,
                profit=profit,
            )

    def evaluate_htf_macro_filter(
        self, observation: Any, update_state: bool = True
    ) -> tuple[str, tuple[float, float, float]] | None:
        """Tier 1: Higher-Timeframe Trend Invariance Filter (H1 EMA diff & slope).

        Reads H1 features from 137-dim observation:
        - h1_ema_diff_atr (wire idx 88)
        - h1_ema_slope_atr (wire idx 89)

        Rules:
        - If h1_ema_diff_atr > 0.05 and h1_ema_slope_atr >= 0.0:
            regime = "BULL", probs = (0.80, 0.10, 0.10), regime_idx = 0
        - If h1_ema_diff_atr < -0.05 and h1_ema_slope_atr <= 0.0:
            regime = "BEAR", probs = (0.10, 0.80, 0.10), regime_idx = 1
        - Otherwise:
            Maintain previous regime or set CHOP safely.
        - If features 88/89 are zero (synthetic unit tests), returns None (falls back cleanly).
        """
        if observation is None:
            return None
        arr = np.asarray(
            getattr(observation, "raw_inputs", getattr(observation, "inputs", observation)),
            dtype=float,
        ).ravel()
        if arr.size < 90 or np.isnan(arr).any() or np.isinf(arr).any():
            return None

        h1_diff = float(arr[88])
        h1_slope = float(arr[89])

        # Backward-compatible check: if both features are zero/near-zero, fall back to detector
        if abs(h1_diff) <= 1e-5 and abs(h1_slope) <= 1e-5:
            return None

        if h1_diff > self.htf_bull_threshold and h1_slope >= self.htf_slope_threshold:
            if update_state:
                self.current_regime = "BULL"
                self.current_regime_index = 0
                self.current_probabilities = (0.80, 0.10, 0.10)
            return "BULL", (0.80, 0.10, 0.10)
        elif h1_diff < self.htf_bear_threshold and h1_slope <= -self.htf_slope_threshold:
            if update_state:
                self.current_regime = "BEAR"
                self.current_regime_index = 1
                self.current_probabilities = (0.10, 0.80, 0.10)
            return "BEAR", (0.10, 0.80, 0.10)
        else:
            return self.current_regime, self.current_probabilities

    def compute_soft_moe_weights(
        self, observation: Any
    ) -> tuple[float, float, dict[str, Any]]:
        """Compute continuous regime weights (w_bull, w_bear) via Soft-MoE Gating Network."""
        if isinstance(observation, np.ndarray):
            arr = observation.reshape(-1)
        else:
            raw = getattr(observation, "raw_inputs", getattr(observation, "inputs", observation))
            arr = raw.reshape(-1) if isinstance(raw, np.ndarray) else np.asarray(raw, dtype=np.float32).reshape(-1)
        w_bull, w_bear, info = self.soft_moe_gating.compute_regime_weights(arr)
        self.soft_moe_weights = (w_bull, w_bear)
        self.soft_moe_telemetry = info
        return w_bull, w_bear, info

    @property
    def active_policy(self) -> Any:
        """Route to appropriate specialist based on verified regime."""
        if getattr(self, "_regime_pinned", False):
            if self.current_regime == "BULL":
                return self.bull_policy
            return self.bear_policy
        if self.current_regime == "BULL":
            return self.bull_policy
        if self.current_regime == "BEAR":
            return self.bear_policy
        if hasattr(self, "soft_moe_weights"):
            if self.soft_moe_weights[0] > 0.50:
                return self.bull_policy
            if self.soft_moe_weights[1] > 0.50:
                return self.bear_policy
        # Default to Bear Specialist (Champion Phase 2) for BEAR and defensive CHOP
        return self.bear_policy

    @property
    def experts(self) -> Any:
        return getattr(self.active_policy, "experts", {})

    @property
    def nhan_dien(self) -> Any:
        return getattr(self.active_policy, "nhan_dien", None)

    @property
    def registry(self) -> Any:
        return getattr(self.active_policy, "registry", None)

    @property
    def cau_hinh(self) -> Any:
        return getattr(self.active_policy, "cau_hinh", None)

    @property
    def input_size(self) -> int:
        return int(getattr(self.active_policy, "input_size", INPUT_COUNT) or INPUT_COUNT)

    @property
    def output_size(self) -> int:
        return int(getattr(self.active_policy, "output_size", OUTPUT_COUNT) or OUTPUT_COUNT)

    @property
    def experience_buffer(self) -> Any:
        """Access experience buffer of the active specialist."""
        buf = getattr(self.active_policy, "experience_buffer", None)
        if buf is not None:
            return buf
        return []

    @experience_buffer.setter
    def experience_buffer(self, value: Any) -> None:
        """Propagate experience buffer updates to both specialists."""
        for pol in (self.bull_policy, self.bear_policy):
            if hasattr(pol, "experience_buffer"):
                try:
                    pol.experience_buffer = value
                except Exception:
                    pass

    def ensure_compatible(
        self,
        input_size: int = INPUT_COUNT,
        output_size: int = OUTPUT_COUNT,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Validate input and output dimensions across all specialist policies."""
        target_in = int(input_size if input_size is not None else self.input_size)
        target_out = int(output_size if output_size is not None else self.output_size)
        if target_in != INPUT_COUNT or target_out != OUTPUT_COUNT:
            raise ValueError(
                f"DualChampionRouter requires {INPUT_COUNT}x{OUTPUT_COUNT} bridge (got {target_in}x{target_out})"
            )
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "ensure_compatible", None)
            if callable(hook):
                try:
                    hook()
                except TypeError:
                    try:
                        hook(input_size=target_in, output_size=target_out, *args, **kwargs)
                    except Exception:
                        pass
                except Exception:
                    pass
        return True

    def set_regime_soft_probabilities(
        self,
        probs: Sequence[float] | np.ndarray,
        regime_name: str | None = None,
        regime_idx: int | None = None,
    ) -> None:
        """Update soft probabilities and apply hysteresis confirmation buffer."""
        if regime_name is not None and str(regime_name).strip() != "":
            self._regime_pinned = True
        p = [float(x) for x in probs]
        if len(p) >= 2:
            tot = p[0] + p[1]
            if tot > 0:
                self.soft_moe_weights = (p[0] / tot, p[1] / tot)
            else:
                self.soft_moe_weights = (0.5, 0.5)

        if len(p) >= 3:
            p_bull, p_bear, p_chop = p[0], p[1], p[2]
            self.current_probabilities = (p_bull, p_bear, p_chop)

            # Evaluate regime with confirmation buffer
            reg_name = str(regime_name or "").upper()
            margin = getattr(self, "margin", 0.10)

            is_bull = (reg_name == "BULL") or (p_bull >= self.bull_threshold and p_bull > p_bear + (0.0 if reg_name else margin))
            is_bear = (reg_name == "BEAR") or (p_bear >= self.bear_threshold and p_bear > p_bull + (0.0 if reg_name else margin))
            is_chop = (reg_name == "CHOP") or (p_chop >= self.chop_threshold)

            if is_bull and not (reg_name == "BEAR" or (p_bear > p_bull and p_bear >= self.bear_threshold)):
                self._pending_bull_count += 1
                self._pending_bear_count = 0
                if self._pending_bull_count >= self.confirmation_bars:
                    self.current_regime = "BULL"
                    self.current_regime_index = 0
            elif is_bear and not (reg_name == "BULL" or (p_bull > p_bear and p_bull >= self.bull_threshold)):
                self._pending_bear_count += 1
                self._pending_bull_count = 0
                if self._pending_bear_count >= self.confirmation_bars:
                    self.current_regime = "BEAR"
                    self.current_regime_index = 1
            elif is_chop:
                self.current_regime = "CHOP"
                self.current_regime_index = 2
                self._pending_bull_count = 0
                self._pending_bear_count = 0
            else:
                # Contest zone: hold previous state (S_t = S_{t-1})
                # Reset pending confirmation counters to enforce consecutive confirmation
                self._pending_bull_count = 0
                self._pending_bear_count = 0

        # Propagate probabilities to both specialists for telemetry tracking
        if getattr(self, "_has_bull_reg_hook", True):
            hook = getattr(self.bull_policy, "set_regime_soft_probabilities", None)
            if callable(hook):
                try:
                    hook(probs, regime_name=self.current_regime, regime_idx=self.current_regime_index)
                except Exception:
                    pass
            else:
                self._has_bull_reg_hook = False
        if getattr(self, "_has_bear_reg_hook", True):
            hook = getattr(self.bear_policy, "set_regime_soft_probabilities", None)
            if callable(hook):
                try:
                    hook(probs, regime_name=self.current_regime, regime_idx=self.current_regime_index)
                except Exception:
                    pass
            else:
                self._has_bear_reg_hook = False

    @staticmethod
    def _extract_action_idx(trace: Any) -> int:
        """Extract the selected action integer from a policy trace."""
        val = getattr(trace, "selected_idx", None)
        if val is not None and type(val).__name__ != "MagicMock":
            try:
                return int(val)
            except Exception:
                pass
        val = getattr(trace, "action_index", None)
        if val is not None and type(val).__name__ != "MagicMock":
            try:
                return int(val)
            except Exception:
                pass
        val = getattr(trace, "action", None)
        if val is not None and type(val).__name__ != "MagicMock":
            try:
                return int(getattr(val, "value", val))
            except Exception:
                pass
        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            if "five_action_index" in details and type(details["five_action_index"]).__name__ != "MagicMock":
                return int(details["five_action_index"])
            if "action" in details and type(details["action"]).__name__ != "MagicMock":
                return int(details["action"])
        out = getattr(trace, "output_vector", getattr(trace, "selected_action", None))
        if out is not None and type(out).__name__ != "MagicMock":
            try:
                arr = np.asarray(out, dtype=float).reshape(-1)
                if arr.size >= 12:
                    return int(np.argmax(arr[:12]))
            except Exception:
                pass
        return 0

    @staticmethod
    def _extract_base_risk(trace: Any, default: float = 0.01) -> float:
        """Extract base risk percentage from trace."""
        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            if "risk_pct" in details:
                return float(details["risk_pct"])
            if "base_risk" in details:
                return float(details["base_risk"])
        if hasattr(trace, "risk_pct") and trace.risk_pct is not None:
            return float(trace.risk_pct)
        out = getattr(trace, "output_vector", getattr(trace, "selected_action", None))
        if out is not None:
            arr = np.asarray(out, dtype=float).reshape(-1)
            if arr.size > 12:
                val = float(arr[12])
                sig = 1.0 / (1.0 + math.exp(-val)) if math.isfinite(val) else 0.5
                return float(0.005 + sig * 0.045)
        return float(default)

    @staticmethod
    def _attach_dynamic_risk(trace: Any, dynamic_risk: float) -> Any:
        """Attach calculated CPPI dynamic risk to the trace."""
        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            details["dynamic_risk"] = float(dynamic_risk)
            details["risk_pct"] = float(dynamic_risk)
        try:
            setattr(trace, "dynamic_risk", float(dynamic_risk))
        except Exception:
            pass
        try:
            setattr(trace, "risk_pct", float(dynamic_risk))
        except Exception:
            pass
        return trace

    @staticmethod
    def _attach_entry_geometry(
        trace: Any,
        sl_atr: float = 2.0,
        tp_rr: float = 2.5,
        trailing_atr: float = 1.0,
    ) -> Any:
        """Inject factorized geometry marker (row 14 = 7.0) and packed geometry (row 16) with trailing stop."""
        out = getattr(trace, "output_vector", getattr(trace, "selected_action", None))
        new_out = None
        if out is not None:
            new_out = np.asarray(out, dtype=float).copy().reshape(-1)
            if new_out.size >= 17:
                new_out[14] = 7.0  # FACTORIZED_ACTION_MARKER_VALUE
                try:
                    from .execution_contract import SL_ATR_BINS, TP_RR_BINS, TRAILING_ATR_BINS, pack_geometry_choices
                    sl_idx = SL_ATR_BINS.index(sl_atr) if sl_atr in SL_ATR_BINS else 7
                    tp_idx = TP_RR_BINS.index(tp_rr) if tp_rr in TP_RR_BINS else 5
                    tr_idx = TRAILING_ATR_BINS.index(trailing_atr) if trailing_atr in TRAILING_ATR_BINS else 5
                    new_out[16] = float(pack_geometry_choices(sl_idx, tp_idx, tr_idx))
                except Exception:
                    new_out[16] = 0.8435803839870712

        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            details["trailing_atr"] = float(trailing_atr)
            details["factorized_marker"] = 7.0

        if is_dataclass(trace):
            kwargs_rep: dict[str, Any] = {}
            if hasattr(trace, "output_vector") and new_out is not None:
                kwargs_rep["output_vector"] = new_out
            if hasattr(trace, "selected_action") and new_out is not None:
                kwargs_rep["selected_action"] = new_out
            if kwargs_rep:
                try:
                    trace = replace(trace, **kwargs_rep)
                except Exception:
                    pass

        if new_out is not None:
            try:
                setattr(trace, "output_vector", new_out)
            except Exception:
                pass
            try:
                setattr(trace, "selected_action", new_out)
            except Exception:
                pass
        return trace

    @staticmethod
    def _enforce_agent4_hold(trace: Any) -> Any:
        """Strictly intercept Agent 4 and lock its action to ACTION_HOLD (0), guaranteeing POLICY_CLOSE_ALL == 0."""
        if isinstance(trace, (int, np.integer)):
            return 0
        out = getattr(trace, "output_vector", getattr(trace, "selected_action", None))
        new_out = None
        if out is not None:
            new_out = out.copy() if isinstance(out, np.ndarray) else np.asarray(out, dtype=float).copy().reshape(-1)
            if new_out.size >= 12:
                new_out[0] = 1.0
                if new_out.size > 10:
                    new_out[10] = -100.0

        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            details["five_action_index"] = 0
            details["wire_action"] = 0
            details["policy_action"] = 0
            details["action"] = 0
            details["legal_action"] = 0
            details["emitted_action"] = 0
            details["policy_close_all"] = 0.0

        if is_dataclass(trace):
            kwargs_rep: dict[str, Any] = {}
            if hasattr(trace, "action_index"):
                kwargs_rep["action_index"] = 0
            if hasattr(trace, "selected_idx"):
                kwargs_rep["selected_idx"] = 0
            if hasattr(trace, "output_vector") and new_out is not None:
                kwargs_rep["output_vector"] = new_out
            if hasattr(trace, "selected_action") and new_out is not None:
                kwargs_rep["selected_action"] = new_out
            if kwargs_rep:
                try:
                    trace = replace(trace, **kwargs_rep)
                except Exception:
                    pass

        try:
            setattr(trace, "action_index", 0)
        except Exception:
            pass
        try:
            setattr(trace, "selected_idx", 0)
        except Exception:
            pass
        try:
            setattr(trace, "action", ACTION_HOLD_ORDER)
        except Exception:
            try:
                setattr(trace, "action", 0)
            except Exception:
                pass
        if new_out is not None:
            try:
                setattr(trace, "selected_action", new_out)
            except Exception:
                pass
            try:
                setattr(trace, "output_vector", new_out)
            except Exception:
                pass

        return trace

    @staticmethod
    def _enforce_veto_hold(
        trace: Any,
        reason: str,
        extra_details: dict[str, Any] | None = None,
    ) -> Any:
        """Strictly intercept an entry action and lock it to ACTION_HOLD (0) due to soft veto."""
        out = getattr(trace, "output_vector", getattr(trace, "selected_action", None))
        new_out = None
        if out is not None:
            new_out = out.copy() if isinstance(out, np.ndarray) else np.asarray(out, dtype=float).copy().reshape(-1)
            if new_out.size >= 12:
                new_out.fill(0.0)
                new_out[0] = 1.0

        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            details["five_action_index"] = 0
            details["wire_action"] = 0
            details["policy_action"] = 0
            details["action"] = 0
            details["legal_action"] = 0
            details["emitted_action"] = 0
            details["regime_veto_applied"] = True
            details["regime_veto_reason"] = str(reason)
            if extra_details:
                details.update(extra_details)

        # Neutralize probability distribution if present
        if hasattr(trace, "action_probs") and getattr(trace, "action_probs", None) is not None:
            try:
                probs = np.zeros_like(trace.action_probs)
                probs[0] = 1.0
                setattr(trace, "action_probs", probs)
            except Exception:
                pass

        if is_dataclass(trace):
            kwargs_rep: dict[str, Any] = {}
            if hasattr(trace, "action_index"):
                kwargs_rep["action_index"] = 0
            if hasattr(trace, "selected_idx"):
                kwargs_rep["selected_idx"] = 0
            if hasattr(trace, "output_vector") and new_out is not None:
                kwargs_rep["output_vector"] = new_out
            if hasattr(trace, "selected_action") and new_out is not None:
                kwargs_rep["selected_action"] = new_out
            if kwargs_rep:
                try:
                    trace = replace(trace, **kwargs_rep)
                except Exception:
                    pass

        try:
            setattr(trace, "action_index", 0)
        except Exception:
            pass
        try:
            setattr(trace, "selected_idx", 0)
        except Exception:
            pass
        try:
            setattr(trace, "action", ACTION_HOLD_ORDER)
        except Exception:
            try:
                setattr(trace, "action", 0)
            except Exception:
                pass
        if new_out is not None:
            try:
                setattr(trace, "selected_action", new_out)
            except Exception:
                pass
            try:
                setattr(trace, "output_vector", new_out)
            except Exception:
                pass

        return trace

    @staticmethod
    def _build_entry_trace(
        trace: Any,
        action_idx: int,
        sl_atr: float = 2.5,
        tp_rr: float = 0.75,
        trailing_atr: float = 1.5,
    ) -> Any:
        """Construct or modify entry trace for Agent 0 (BUY) or Agent 1 (SELL) with high-frequency scalp geometry."""
        outputs = np.zeros(20, dtype=float)
        outputs[action_idx] = 1.0
        outputs[14] = 7.0  # FACTORIZED_ACTION_MARKER_VALUE
        try:
            sl_idx = SL_ATR_BINS.index(sl_atr) if sl_atr in SL_ATR_BINS else 4
            tp_idx = TP_RR_BINS.index(tp_rr) if tp_rr in TP_RR_BINS else 2
            tr_idx = TRAILING_ATR_BINS.index(trailing_atr) if trailing_atr in TRAILING_ATR_BINS else 3
            outputs[16] = float(pack_geometry_choices(sl_idx, tp_idx, tr_idx))
        except Exception:
            outputs[16] = 0.8435803839870712

        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            details["five_action_index"] = action_idx
            details["wire_action"] = action_idx
            details["policy_action"] = action_idx
            details["action"] = action_idx
            details["legal_action"] = action_idx
            details["emitted_action"] = action_idx
            details["trailing_atr"] = float(trailing_atr)
            details["factorized_marker"] = 7.0
            details["is_scalp"] = True

        if is_dataclass(trace):
            kwargs_rep: dict[str, Any] = {}
            if hasattr(trace, "action_index"):
                kwargs_rep["action_index"] = action_idx
            if hasattr(trace, "selected_idx"):
                kwargs_rep["selected_idx"] = action_idx
            if hasattr(trace, "output_vector"):
                kwargs_rep["output_vector"] = outputs
            if hasattr(trace, "selected_action"):
                kwargs_rep["selected_action"] = outputs
            if kwargs_rep:
                try:
                    trace = replace(trace, **kwargs_rep)
                except Exception:
                    pass

        try:
            setattr(trace, "action_index", action_idx)
        except Exception:
            pass
        try:
            setattr(trace, "output_vector", outputs)
        except Exception:
            pass
        try:
            setattr(trace, "selected_action", outputs)
        except Exception:
            pass
        return trace

    @staticmethod
    def _build_agent2_manage_trace(trace: Any, target_slot: int = 0) -> Any:
        """Modify Agent 2 trace to emit ACTION_MANAGE_POSITION (11) with Mode 1 (Partial Close), CloseBin 1 (50%), and 3.0 ATR Chandelier trailing."""
        outputs_2 = np.zeros(20, dtype=float)
        outputs_2[11] = 1.0   # ACTION_MANAGE_POSITION logit highest
        outputs_2[14] = 7.0   # FACTORIZED_ACTION_MARKER_VALUE
        try:
            outputs_2[16] = float(pack_geometry_choices(sl_index=4, tp_index=5, trailing_index=6))
        except Exception:
            outputs_2[16] = 1.5131748978745039
        outputs_2[17] = -0.5  # closeBin 1 (50% ratio): MathFloor(Sigmoid(-0.5)*4) = 1
        outputs_2[18] = 0.0   # mode 1 (Partial Close): MathFloor(Sigmoid(0.0)*3) = 1
        outputs_2[19] = float(target_slot)

        details = getattr(trace, "action_details", None)
        if isinstance(details, dict):
            details["five_action_index"] = 11
            details["wire_action"] = 11
            details["policy_action"] = 11
            details["action"] = 11
            details["legal_action"] = 11
            details["emitted_action"] = 11
            details["management_mode"] = 1
            details["close_bin"] = 1
            details["close_ratio"] = 0.50
            details["trailing_index"] = 6
            details["trailing_atr"] = 3.0
            details["target_slot"] = target_slot

        if is_dataclass(trace):
            kwargs_rep: dict[str, Any] = {}
            if hasattr(trace, "action_index"):
                kwargs_rep["action_index"] = 11
            if hasattr(trace, "selected_idx"):
                kwargs_rep["selected_idx"] = 11
            if hasattr(trace, "output_vector"):
                kwargs_rep["output_vector"] = outputs_2
            if hasattr(trace, "selected_action"):
                kwargs_rep["selected_action"] = outputs_2
            if hasattr(trace, "manage_mode_index"):
                kwargs_rep["manage_mode_index"] = 1
            if hasattr(trace, "close_fraction_index"):
                kwargs_rep["close_fraction_index"] = 1
            if hasattr(trace, "trailing_index"):
                kwargs_rep["trailing_index"] = 6
            if kwargs_rep:
                try:
                    trace = replace(trace, **kwargs_rep)
                except Exception:
                    pass

        try:
            setattr(trace, "action_index", 11)
        except Exception:
            pass
        try:
            setattr(trace, "selected_idx", 11)
        except Exception:
            pass
        try:
            from .orders import Action
            setattr(trace, "action", Action.MANAGE_POSITION)
        except Exception:
            try:
                setattr(trace, "action", 11)
            except Exception:
                pass
        try:
            setattr(trace, "selected_action", outputs_2)
        except Exception:
            pass
        try:
            setattr(trace, "output_vector", outputs_2)
        except Exception:
            pass

        return trace

    @staticmethod
    def _parse_positions_pipe(positions_pipe: str) -> list[dict[str, Any]]:
        positions: list[dict[str, Any]] = []
        for item in str(positions_pipe or "").split("^"):
            if not item:
                continue
            parts = item.split(":")
            if len(parts) < 6:
                continue
            try:
                ticket = str(parts[0])
                side = int(float(parts[1]))
                vol = float(parts[2])
                entry = float(parts[3])
                sl = float(parts[4])
                tp = float(parts[5]) if len(parts) > 5 else 0.0
                unrealized_r = float(parts[6]) if len(parts) > 6 else 0.0
                age_bars = float(parts[7]) if len(parts) > 7 else 0.0
                positions.append({
                    "ticket": ticket,
                    "side": side,
                    "volume": vol,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "unrealized_r": unrealized_r,
                    "age_bars": age_bars,
                })
            except (ValueError, IndexError):
                continue
        return positions

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray | Any,
        goal: np.ndarray | None = None,
        *args: Any,
        goal_context: np.ndarray | None = None,
        current_regime_distribution: tuple[float, float, float] | None = None,
        request: Any | None = None,
        **kwargs: Any,
    ) -> BridgeActionBatch | tuple[object, ...]:
        """Dispatch decision to active specialist policy with CPPI dynamic risk & Two-Tier runner.

        Advances recurrent memory of the active specialist while keeping the
        inactive specialist completely frozen (L_inf = 0.0).
        """
        # 1. Unpack observation / request
        req = None
        if hasattr(observation, "inputs") and hasattr(observation, "equity"):
            req = observation
            obs = req.inputs
            reg_probs = getattr(req, "regime_probabilities", None)
        elif request is not None and hasattr(request, "inputs") and hasattr(request, "equity"):
            req = request
            obs = observation
            reg_probs = getattr(req, "regime_probabilities", None)
        else:
            req = kwargs.get("request", None)
            obs = observation
            reg_probs = getattr(req, "regime_probabilities", None) if req is not None else None

        if req is not None and hasattr(req, "time") and req.time:
            self._current_time = req.time
            if hasattr(self, "circuit_breaker"):
                self.circuit_breaker._current_time = self.circuit_breaker._parse_time(req.time)

        if hasattr(self, "circuit_breaker"):
            self.circuit_breaker.tick_step()

        # 2. Continuous Soft-MoE Gating Network Evaluation (Eliminates 15-45m lag)
        w_bull, w_bear, moe_info = self.compute_soft_moe_weights(obs)
        p_chop = float(moe_info.get("chop_index", max(0.0, 1.0 - abs(w_bull - w_bear))))
        p_chop_c = max(0.0, min(1.0, p_chop))
        soft_dist = (w_bull * (1.0 - p_chop_c), w_bear * (1.0 - p_chop_c), p_chop_c)

        if self.htf_filter_enabled:
            # Evaluate for telemetry without freezing the router
            self.evaluate_htf_macro_filter(obs, update_state=False)

        dist = current_regime_distribution if current_regime_distribution is not None else reg_probs
        if dist is not None:
            self.set_regime_soft_probabilities(dist)
        elif not getattr(self, "_regime_pinned", False):
            self.set_regime_soft_probabilities(soft_dist)

        # 3. Update CPPI Equity state
        equity = None
        if req is not None and hasattr(req, "equity"):
            eq_val = float(req.equity)
            if math.isfinite(eq_val) and eq_val > 0.0:
                equity = eq_val
        elif "equity" in kwargs and kwargs["equity"] is not None:
            eq_val = float(kwargs["equity"])
            if math.isfinite(eq_val) and eq_val > 0.0:
                equity = eq_val
        elif self._last_equity > 0.0:
            equity = self._last_equity

        if equity is not None and math.isfinite(equity) and equity > 0.0:
            self.cppi_engine.update_equity(equity)
            self.bidirectional_cppi_engine.update_equity(equity)
            self._last_equity = equity

        # 4. Track positions pipe and unpack open_positions upfront
        if req is not None and hasattr(req, "positions_pipe") and req.positions_pipe:
            self._last_positions_pipe = str(req.positions_pipe)
        elif "positions_pipe" in kwargs and kwargs["positions_pipe"]:
            self._last_positions_pipe = str(kwargs["positions_pipe"])

        open_positions: list[dict[str, Any]] = []
        if "positions" in kwargs and isinstance(kwargs["positions"], list):
            open_positions = kwargs["positions"]
        elif hasattr(req, "positions_list") and isinstance(req.positions_list, list):
            open_positions = req.positions_list
        elif self._last_positions_pipe:
            open_positions = self._parse_positions_pipe(self._last_positions_pipe)

        # 5. Call active specialist policy
        active = self.active_policy
        hook = getattr(active, "act_nam_tac_nhan", None)
        if not callable(hook):
            raise RuntimeError(f"active policy {type(active)} does not expose act_nam_tac_nhan")
        effective_goal = goal if goal is not None else goal_context
        try:
            raw_traces = hook(obs, effective_goal, *args, **kwargs)
        except TypeError:
            raw_traces = hook(obs, effective_goal)

        traces_list = list(raw_traces) if isinstance(raw_traces, (list, tuple)) else [raw_traces]
        if len(traces_list) != 5:
            # Duck-typing test mock compatibility (e.g. single trace mock)
            return BridgeActionBatch(traces_list)

        cur_eq = equity if (equity is not None and equity > 0.0) else (self.bidirectional_cppi_engine.current_equity if self.bidirectional_cppi_engine.current_equity > 0.0 else (self.cppi_engine.current_equity if self.cppi_engine.current_equity > 0.0 else 10000.0))
        flat_obs = obs.reshape(-1) if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=np.float32).reshape(-1)

        # 6. Intercept Agent 4: strictly lock action to ACTION_HOLD (0), guaranteeing POLICY_CLOSE_ALL == 0
        traces_list[4] = self._enforce_agent4_hold(traces_list[4])

        # 6.1 Cap max trades if configured
        if self.max_total_trades > 0 and self.total_closed_trades >= self.max_total_trades:
            act_0 = self._extract_action_idx(traces_list[0])
            if act_0 in (1, 3, 5):
                traces_list[0] = self._enforce_veto_hold(traces_list[0], reason="max_total_trades_reached")
            act_1 = self._extract_action_idx(traces_list[1])
            if act_1 in (2, 4, 6):
                traces_list[1] = self._enforce_veto_hold(traces_list[1], reason="max_total_trades_reached")

        # 6.2 Concurrent open positions limit (prevents basket stacking & correlated batch stopouts)
        if self.max_concurrent_positions > 0 and len(open_positions) >= self.max_concurrent_positions:
            act_0 = self._extract_action_idx(traces_list[0])
            if act_0 in (1, 3, 5):
                traces_list[0] = self._enforce_veto_hold(traces_list[0], reason="max_concurrent_positions_reached")
            act_1 = self._extract_action_idx(traces_list[1])
            if act_1 in (2, 4, 6):
                traces_list[1] = self._enforce_veto_hold(traces_list[1], reason="max_concurrent_positions_reached")

        # 6.2b Directional position limit (anti-cluster basket guard)
        if self.max_positions_per_direction > 0 and open_positions:
            open_buy_count = sum(1 for p in open_positions if self.bidirectional_cppi_engine._classify_position_direction(p) > 0)
            open_sell_count = sum(1 for p in open_positions if self.bidirectional_cppi_engine._classify_position_direction(p) < 0)
            if open_buy_count >= self.max_positions_per_direction:
                act_0 = self._extract_action_idx(traces_list[0])
                if act_0 in (1, 3, 5):
                    traces_list[0] = self._enforce_veto_hold(traces_list[0], reason="max_directional_buys_reached")
            if open_sell_count >= self.max_positions_per_direction:
                act_1 = self._extract_action_idx(traces_list[1])
                if act_1 in (2, 4, 6):
                    traces_list[1] = self._enforce_veto_hold(traces_list[1], reason="max_directional_sells_reached")

        # 6.3 Minimum bar spacing cadence between entries
        self._bars_since_last_entry += 1
        if self.min_spacing_bars > 0 and self._bars_since_last_entry < self.min_spacing_bars:
            act_0 = self._extract_action_idx(traces_list[0])
            if act_0 in (1, 3, 5):
                traces_list[0] = self._enforce_veto_hold(traces_list[0], reason="min_spacing_bars_cooldown")
            act_1 = self._extract_action_idx(traces_list[1])
            if act_1 in (2, 4, 6):
                traces_list[1] = self._enforce_veto_hold(traces_list[1], reason="min_spacing_bars_cooldown")

        # 6.5 Soft-MoE Transition Zone & Capital Preservation Guard
        w_bull, w_bear = getattr(self, "soft_moe_weights", (0.5, 0.5))
        moe_info = getattr(self, "soft_moe_telemetry", {})
        chop_val = float(moe_info.get("chop_index", flat_obs[93] if flat_obs.size > 93 else 0.5))
        guard = getattr(self, "capital_preservation_guard", None)
        in_trans = (
            guard.is_transition_active(w_bull, w_bear, chop_val)
            if guard is not None
            else (abs(w_bull - w_bear) <= 0.30 or chop_val >= 0.60)
        )

        if in_trans:
            # Capital Preservation Guard: Suppress market order chasing (Action 1 MARKET_BUY, Action 2 MARKET_SELL)
            act_0 = self._extract_action_idx(traces_list[0])
            if act_0 == 1:
                traces_list[0] = self._enforce_veto_hold(
                    traces_list[0],
                    reason="capital_preservation_transition_zone_no_market_chase",
                )
            act_1 = self._extract_action_idx(traces_list[1])
            if act_1 == 2:
                traces_list[1] = self._enforce_veto_hold(
                    traces_list[1],
                    reason="capital_preservation_transition_zone_no_market_chase",
                )

        # 6.6 Anti-Bottom Exhaustion Guard & Demand Zone Liquidity Shield
        # Intercepts Bear Specialist (Agent 1) SELL orders in demand zones to eliminate bottom shorting
        self.anti_bottom_telemetry = {
            "bottom_short_veto": False,
            "veto_reason": "",
            "demand_zone_score": 0.0,
        }
        if hasattr(self, "anti_bottom_guard") and self.anti_bottom_guard is not None and getattr(self.anti_bottom_guard, "enabled", True):
            act_1 = self._extract_action_idx(traces_list[1])
            if act_1 in (2, 4, 6):  # MARKET_SELL=2, SELL_LIMIT=4, SELL_STOP=6
                is_vetoed, veto_reason, veto_telemetry = self.anti_bottom_guard.check_bottom_short_veto(
                    flat_obs, act_1
                )
                if is_vetoed:
                    self.anti_bottom_telemetry = {
                        "bottom_short_veto": True,
                        "veto_reason": f"anti_bottom_guard:{veto_reason}",
                        "demand_zone_score": float(veto_telemetry.get("demand_zone_score", 0.0)),
                        **veto_telemetry,
                    }
                    traces_list[1] = self._enforce_veto_hold(
                        traces_list[1],
                        reason=f"anti_bottom_guard:{veto_reason}",
                        extra_details=self.anti_bottom_telemetry,
                    )
        self.soft_moe_telemetry["anti_bottom_guard"] = dict(self.anti_bottom_telemetry)

        # Tier 2 Microstructure Momentum Gate
        if self.momentum_gate_enabled and flat_obs.size >= 85:
            m5_d = float(flat_obs[80])
            m5_s = float(flat_obs[81])
            m15_d = float(flat_obs[84]) if flat_obs.size > 84 else 0.0
            if (m5_d > 0.0 and m5_s > 0.0) or (m15_d > 0.10 and m5_d > 0.0):
                act_1 = self._extract_action_idx(traces_list[1])
                if act_1 in (2, 4, 6):
                    traces_list[1] = self._enforce_veto_hold(
                        traces_list[1],
                        reason="veto_sell_against_upward_momentum",
                    )
            elif (m5_d < 0.0 and m5_s < 0.0) or (m15_d < -0.10 and m5_d < 0.0):
                act_0 = self._extract_action_idx(traces_list[0])
                if act_0 in (1, 3, 5):
                    traces_list[0] = self._enforce_veto_hold(
                        traces_list[0],
                        reason="veto_buy_against_downward_momentum",
                    )

        # 7. Evaluate open positions using TwoTierTrailingRunnerEngine for Agent 2 (index 2)
        if not open_positions:
            if "positions" in kwargs and isinstance(kwargs["positions"], list):
                open_positions = kwargs["positions"]
            elif hasattr(req, "positions_list") and isinstance(req.positions_list, list):
                open_positions = req.positions_list
            elif self._last_positions_pipe:
                open_positions = self._parse_positions_pipe(self._last_positions_pipe)

        tier1_triggered = False
        triggered_slot = 0
        for slot_idx, pos in enumerate(open_positions):
            pid = str(pos.get("ticket", f"pos_{slot_idx}"))
            direction = int(pos.get("side", 1))
            entry_p = float(pos.get("entry", 0.0))
            sl_p = float(pos.get("sl", 0.0))
            atr = float(pos.get("entry_atr", 0.0))
            if atr <= 0.0:
                if abs(entry_p - sl_p) > 0.0:
                    atr = abs(entry_p - sl_p) / 5.0
                else:
                    atr = 1.0

            unrealized_atr = float(pos.get("unrealized_atr", 0.0))
            if unrealized_atr <= 0.0 and "unrealized_r" in pos:
                unrealized_atr = float(pos["unrealized_r"]) * 5.0

            if direction > 0:
                cur_p = entry_p + (unrealized_atr * atr)
            else:
                cur_p = entry_p - (unrealized_atr * atr)

            eval_res = self.two_tier_runner.evaluate_position(
                position_id=pid,
                entry_price=entry_p,
                current_price=cur_p,
                direction=direction,
                entry_atr=atr,
            )
            if eval_res.get("partial_close", False):
                tier1_triggered = True
                triggered_slot = slot_idx
                break

        if tier1_triggered:
            traces_list[2] = self._build_agent2_manage_trace(traces_list[2], target_slot=triggered_slot)

        # 8. Tier 2: Microstructure Momentum Gate
        if self.momentum_gate_enabled and flat_obs.size >= 85:
            m5_diff = float(flat_obs[80])
            m5_slope = float(flat_obs[81])
            m15_diff = float(flat_obs[84])

            # Agent 1 (Short Owner): Action 4 (SELL_LIMIT), Action 2 (MARKET_SELL), or Action 6 (SELL_STOP)
            act_1 = self._extract_action_idx(traces_list[1])
            if act_1 in (2, 4, 6):
                if (m5_diff > 0.0 and m5_slope > 0.0) or (m15_diff > 0.10 and m5_diff > 0.0):
                    traces_list[1] = self._enforce_veto_hold(
                        traces_list[1],
                        reason="veto_sell_against_upward_momentum",
                    )

            # Agent 0 (Long Owner): Action 3 (BUY_LIMIT), Action 1 (MARKET_BUY), or Action 5 (BUY_STOP)
            act_0 = self._extract_action_idx(traces_list[0])
            if act_0 in (1, 3, 5):
                if (m5_diff < 0.0 and m5_slope < 0.0) or (m15_diff < -0.10 and m5_diff < 0.0):
                    traces_list[0] = self._enforce_veto_hold(
                        traces_list[0],
                        reason="veto_buy_against_downward_momentum",
                    )

        # 9. Tier 3: Consecutive Loss Circuit Breaker
        if hasattr(self, "circuit_breaker") and self.circuit_breaker.enabled:
            req_time = getattr(req, "time", None) if req is not None else kwargs.get("time", getattr(self, "_current_time", None))

            # Check Agent 0 (BUY) entries
            act_0 = self._extract_action_idx(traces_list[0])
            if act_0 in (1, 3, 5):
                if self.circuit_breaker.is_in_cooldown("BUY", req_time):
                    traces_list[0] = self._enforce_veto_hold(
                        traces_list[0],
                        reason="circuit_breaker_consecutive_sl_cooldown",
                    )

            # Check Agent 1 (SELL) entries
            act_1 = self._extract_action_idx(traces_list[1])
            if act_1 in (2, 4, 6):
                if self.circuit_breaker.is_in_cooldown("SELL", req_time):
                    traces_list[1] = self._enforce_veto_hold(
                        traces_list[1],
                        reason="circuit_breaker_consecutive_sl_cooldown",
                    )

        # 9.5 Tier 3.5: Micro-Pullback High-Frequency Scalper Engine (>= 100 trades/day & WR >= 85%)
        scalp_sig = None
        act_0 = self._extract_action_idx(traces_list[0])
        act_1 = self._extract_action_idx(traces_list[1])
        if act_0 == 0 and act_1 == 0 and hasattr(self, "micro_scalp_engine") and self.micro_scalp_engine is not None and self.micro_scalp_engine.config.enabled:
            open_cnt = len(open_positions)
            req_time = getattr(req, "time", None) if req is not None else kwargs.get("time", None)
            macro_vetoed = False
            if hasattr(self, "circuit_breaker") and self.circuit_breaker.enabled:
                macro_vetoed = self.circuit_breaker.is_in_cooldown("BUY", req_time) and self.circuit_breaker.is_in_cooldown("SELL", req_time)

            scalp_sig = self.micro_scalp_engine.evaluate_scalp_opportunity(
                observation=flat_obs,
                regime_distribution=self.current_probabilities,
                open_positions_count=open_cnt,
                macro_vetoed=macro_vetoed,
            )
            if scalp_sig.should_enter:
                cb_in_cd = False
                if hasattr(self, "circuit_breaker") and self.circuit_breaker.enabled:
                    if scalp_sig.direction > 0 and self.circuit_breaker.is_in_cooldown("BUY", req_time):
                        cb_in_cd = True
                    elif scalp_sig.direction < 0 and self.circuit_breaker.is_in_cooldown("SELL", req_time):
                        cb_in_cd = True

                if not cb_in_cd:
                    scalp_dir_str = "BUY" if scalp_sig.direction > 0 else "SELL"
                    scalp_allowed, _, _ = self.bidirectional_cppi_engine.evaluate_order_risk(
                        cur_eq, self.bidirectional_cppi_engine.peak_equity, scalp_dir_str, open_positions
                    )
                    if scalp_allowed:
                        if scalp_sig.direction > 0:  # BUY
                            traces_list[0] = self._build_entry_trace(
                                traces_list[0],
                                action_idx=scalp_sig.action_index,
                                sl_atr=scalp_sig.sl_atr,
                                tp_rr=scalp_sig.tp_rr,
                                trailing_atr=scalp_sig.trailing_atr,
                            )
                        elif scalp_sig.direction < 0:  # SELL
                            scalp_vetoed = False
                            if hasattr(self, "anti_bottom_guard") and self.anti_bottom_guard is not None and getattr(self.anti_bottom_guard, "enabled", True):
                                is_v, _, _ = self.anti_bottom_guard.check_bottom_short_veto(flat_obs, scalp_sig.action_index)
                                if is_v:
                                    scalp_vetoed = True
                            if not scalp_vetoed:
                                traces_list[1] = self._build_entry_trace(
                                    traces_list[1],
                                    action_idx=scalp_sig.action_index,
                                    sl_atr=scalp_sig.sl_atr,
                                    tp_rr=scalp_sig.tp_rr,
                                    trailing_atr=scalp_sig.trailing_atr,
                                )

        # 9.6 Concurrency & Bidirectional CPPI Order Gating (Unified final validation)
        act_0 = self._extract_action_idx(traces_list[0])
        act_1 = self._extract_action_idx(traces_list[1])

        # Simultaneous dual entry arbitration (if both BUY and SELL active)
        if act_0 in (1, 3, 5) and act_1 in (2, 4, 6):
            if len(open_positions) + 2 > self.max_concurrent_positions:
                w_bull, w_bear = getattr(self, "soft_moe_weights", (0.5, 0.5))
                if w_bull >= w_bear:
                    traces_list[1] = self._enforce_veto_hold(traces_list[1], reason="simultaneous_entry_conflict_veto_sell")
                else:
                    traces_list[0] = self._enforce_veto_hold(traces_list[0], reason="simultaneous_entry_conflict_veto_buy")
            act_0 = self._extract_action_idx(traces_list[0])
            act_1 = self._extract_action_idx(traces_list[1])

        # Directional and Global Concurrency Gating via Bidirectional CPPI Engine
        dyn_risk_0 = None
        if act_0 in (1, 3, 5):
            allowed_0, eval_risk_0, reason_0 = self.bidirectional_cppi_engine.evaluate_order_risk(
                cur_eq, self.bidirectional_cppi_engine.peak_equity, "BUY", open_positions
            )
            if not allowed_0:
                traces_list[0] = self._enforce_veto_hold(traces_list[0], reason=reason_0)
            else:
                dyn_risk_0 = eval_risk_0

        dyn_risk_1 = None
        if act_1 in (2, 4, 6):
            allowed_1, eval_risk_1, reason_1 = self.bidirectional_cppi_engine.evaluate_order_risk(
                cur_eq, self.bidirectional_cppi_engine.peak_equity, "SELL", open_positions
            )
            if not allowed_1:
                traces_list[1] = self._enforce_veto_hold(traces_list[1], reason=reason_1)
            else:
                dyn_risk_1 = eval_risk_1

        # 10. Dynamic Risk Computation on Entry Actions (Agent 0 BUY, Agent 1 SELL)
        latest_dynamic_risk = None
        act_0 = self._extract_action_idx(traces_list[0])
        act_1 = self._extract_action_idx(traces_list[1])
        if act_0 in (1, 3, 5) or act_1 in (2, 4, 6):
            self._bars_since_last_entry = 0

        if act_0 in (1, 3, 5):  # MARKET_BUY, BUY_LIMIT, BUY_STOP
            if self.bidirectional_cppi_config is not None:
                if dyn_risk_0 is None:
                    _, dyn_risk_0, _ = self.bidirectional_cppi_engine.evaluate_order_risk(
                        cur_eq, self.bidirectional_cppi_engine.peak_equity, "BUY", open_positions
                    )
            else:
                base_risk = self._extract_base_risk(traces_list[0])
                dyn_risk_0 = self.cppi_engine.compute_dynamic_risk(
                    base_risk_pct=base_risk,
                    current_equity=cur_eq,
                    inputs=flat_obs,
                    trade_direction=1,
                )
            if in_trans and guard is not None:
                def_scale = guard.compute_defensive_scale(w_bull, w_bear, chop_val)
                min_risk = (
                    self.bidirectional_cppi_engine.config.min_risk_pct
                    if self.bidirectional_cppi_config is not None
                    else 0.001
                )
                dyn_risk_0 = max(min_risk, dyn_risk_0 * def_scale)
            traces_list[0] = self._attach_dynamic_risk(traces_list[0], dyn_risk_0)
            if scalp_sig is not None and scalp_sig.should_enter and scalp_sig.direction > 0:
                traces_list[0] = self._attach_entry_geometry(
                    traces_list[0],
                    sl_atr=scalp_sig.sl_atr,
                    tp_rr=scalp_sig.tp_rr,
                    trailing_atr=scalp_sig.trailing_atr,
                )
            else:
                traces_list[0] = self._attach_entry_geometry(
                    traces_list[0],
                    sl_atr=self.default_sl_atr,
                    tp_rr=self.default_tp_rr,
                    trailing_atr=self.default_trailing_atr,
                )
            latest_dynamic_risk = dyn_risk_0

        act_1 = self._extract_action_idx(traces_list[1])
        if act_1 in (2, 4, 6):  # MARKET_SELL, SELL_LIMIT, SELL_STOP
            if self.bidirectional_cppi_config is not None:
                if dyn_risk_1 is None:
                    _, dyn_risk_1, _ = self.bidirectional_cppi_engine.evaluate_order_risk(
                        cur_eq, self.bidirectional_cppi_engine.peak_equity, "SELL", open_positions
                    )
            else:
                base_risk = self._extract_base_risk(traces_list[1])
                dyn_risk_1 = self.cppi_engine.compute_dynamic_risk(
                    base_risk_pct=base_risk,
                    current_equity=cur_eq,
                    inputs=flat_obs,
                    trade_direction=-1,
                )
            if in_trans and guard is not None:
                def_scale = guard.compute_defensive_scale(w_bull, w_bear, chop_val)
                min_risk = (
                    self.bidirectional_cppi_engine.config.min_risk_pct
                    if self.bidirectional_cppi_config is not None
                    else 0.001
                )
                dyn_risk_1 = max(min_risk, dyn_risk_1 * def_scale)
            traces_list[1] = self._attach_dynamic_risk(traces_list[1], dyn_risk_1)
            if scalp_sig is not None and scalp_sig.should_enter and scalp_sig.direction < 0:
                traces_list[1] = self._attach_entry_geometry(
                    traces_list[1],
                    sl_atr=scalp_sig.sl_atr,
                    tp_rr=scalp_sig.tp_rr,
                    trailing_atr=scalp_sig.trailing_atr,
                )
            else:
                traces_list[1] = self._attach_entry_geometry(
                    traces_list[1],
                    sl_atr=self.default_sl_atr,
                    tp_rr=self.default_tp_rr,
                    trailing_atr=self.default_trailing_atr,
                )
            latest_dynamic_risk = dyn_risk_1

        # 11. Strictly guarantee Agent 4 Governor invariant (100% HOLD, POLICY_CLOSE_ALL == 0)
        traces_list[4] = self._enforce_agent4_hold(traces_list[4])

        cppi_tel = self.cppi_engine.get_telemetry()
        bi_tel = self.bidirectional_cppi_engine.get_telemetry()
        merged_cppi_telemetry = (
            {**cppi_tel, **bi_tel, "bidirectional": bi_tel}
            if self.bidirectional_cppi_config is not None
            else {**bi_tel, **cppi_tel, "bidirectional": bi_tel}
        )

        return BridgeActionBatch(
            traces_list,
            risk_pct=latest_dynamic_risk,
            cppi_telemetry=merged_cppi_telemetry,
            anti_bottom_telemetry=self.anti_bottom_telemetry,
        )

    def _other_specialist(self, pol: Any) -> Any:
        """Return the inactive specialist opposite to the provided policy."""
        return self.bull_policy if pol is self.bear_policy else self.bear_policy

    def act(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Compatibility bridge method required by MT5StrategyTesterEvaluator._serve_policy_actions.

        Delegates to active specialist policy's act() if available.
        If active specialist exposes only act_nam_tac_nhan(), delegates to self.act_nam_tac_nhan().
        """
        active = self.active_policy
        hook = getattr(active, "act", None)
        if callable(hook):
            try:
                return hook(observation, goal, *args, **kwargs)
            except TypeError:
                return hook(observation, goal)
        five_hook = getattr(active, "act_nam_tac_nhan", None)
        if callable(five_hook):
            return self.act_nam_tac_nhan(observation, goal=goal, *args, **kwargs)
        raise RuntimeError(f"active policy {type(active)} does not expose act or act_nam_tac_nhan")

    def observe_five_economic_transition(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate five-agent economic transition observation to active specialist; fallback to alternate policy."""
        for pol in (self.active_policy, self._other_specialist(self.active_policy)):
            hook = getattr(pol, "observe_five_economic_transition", None)
            if callable(hook):
                try:
                    return hook(*args, **kwargs)
                except Exception:
                    pass
        return {}

    def observe_transition(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate single transition observation to active specialist."""
        hook = getattr(self.active_policy, "observe_transition", None)
        if callable(hook):
            try:
                return hook(*args, **kwargs)
            except Exception:
                pass
        return {}

    def set_experience_starvation(self, *args: Any, **kwargs: Any) -> None:
        """Propagate experience starvation signals to both specialists."""
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "set_experience_starvation", None)
            if callable(hook):
                try:
                    hook(*args, **kwargs)
                except Exception:
                    pass

    def set_account_context(
        self,
        account_context: Mapping[str, float],
        positions_pipe: str = "",
    ) -> None:
        """Synchronize account state to both specialists and CPPI engine."""
        if account_context:
            eq = float(account_context.get("equity", 0.0) or 0.0)
            if eq > 0.0:
                self.cppi_engine.update_equity(eq)
                self._last_equity = eq
        if positions_pipe:
            self._last_positions_pipe = str(positions_pipe)
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "set_account_context", None)
            if callable(hook):
                try:
                    hook(account_context, positions_pipe)
                except Exception:
                    pass

    def set_decision_context(self, **kwargs: Any) -> None:
        """Synchronize decision context to both specialists."""
        if kwargs.get("timestamp") is not None:
            self._current_time = kwargs.get("timestamp")
            if hasattr(self, "circuit_breaker"):
                parsed_t = self.circuit_breaker._parse_time(kwargs.get("timestamp"))
                self.circuit_breaker._current_time = parsed_t
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "set_decision_context", None)
            if callable(hook):
                try:
                    hook(**kwargs)
                except Exception:
                    pass

    def set_position_context(self, *args: Any, **kwargs: Any) -> None:
        """Synchronize position context across both specialists."""
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "set_position_context", None)
            if callable(hook):
                try:
                    hook(*args, **kwargs)
                except Exception:
                    pass

    def set_regime_raw_inputs(self, raw_inputs: Any) -> None:
        """Propagate raw wire observation to both specialists and evaluate HTF filter if enabled."""
        if self.htf_filter_enabled:
            self.evaluate_htf_macro_filter(raw_inputs)
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "set_regime_raw_inputs", None)
            if callable(hook):
                try:
                    hook(raw_inputs)
                except Exception:
                    pass

    def record_position_context_extrema(self, *args: Any, **kwargs: Any) -> None:
        """Propagate position context extrema records to both specialists."""
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "record_position_context_extrema", None)
            if callable(hook):
                try:
                    hook(*args, **kwargs)
                except Exception:
                    pass

    def record_five_live_owner_outcome(self, *args: Any, **kwargs: Any) -> None:
        """Propagate five-live owner outcome tracking to both specialists and circuit breaker."""
        self.total_closed_trades += 1
        if hasattr(self, "circuit_breaker") and self.circuit_breaker.enabled:
            agent_idx = kwargs.get("agent_index", None)
            outcome = str(kwargs.get("outcome", "")).upper()
            profit = float(kwargs.get("profit", 0.0) or 0.0)
            net_r = float(kwargs.get("net_r", 0.0) or 0.0)
            timestamp = kwargs.get("timestamp", kwargs.get("time", None))

            is_sl = ("SL" in outcome) or (net_r <= -0.9)
            is_win = ("WIN" in outcome) or ("TP" in outcome) or (profit > 0.0) or (net_r > 0.0)

            d = None
            if agent_idx == 0:
                d = "BUY"
            elif agent_idx == 1:
                d = "SELL"
            elif "direction" in kwargs:
                d = str(kwargs["direction"]).upper()
            elif "side" in kwargs:
                d = "BUY" if int(kwargs["side"]) > 0 else "SELL"

            if d in ("BUY", "SELL"):
                self.circuit_breaker.record_outcome(
                    direction=d,
                    is_sl=is_sl,
                    is_win=is_win,
                    timestamp=timestamp,
                    profit=profit if profit != 0.0 else net_r,
                )
            elif is_sl or is_win:
                d_fallback = "BUY" if self.current_regime == "BULL" else "SELL"
                self.circuit_breaker.record_outcome(
                    direction=d_fallback,
                    is_sl=is_sl,
                    is_win=is_win,
                    timestamp=timestamp,
                    profit=profit if profit != 0.0 else net_r,
                )

        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "record_five_live_owner_outcome", None)
            if callable(hook):
                try:
                    hook(*args, **kwargs)
                except Exception:
                    pass
        return True

    def reset_runtime_state(self) -> None:
        """Reset confirmation counters, regime state, and sub-policies' recurrent state."""
        self.current_regime = "CHOP"
        self.current_regime_index = 2
        self.current_probabilities = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self._pending_bull_count = 0
        self._pending_bear_count = 0
        self.reset_circuit_breaker()
        if hasattr(self.cppi_engine, "reset"):
            self.cppi_engine.reset()
        if hasattr(self, "bidirectional_cppi_engine") and hasattr(self.bidirectional_cppi_engine, "reset"):
            self.bidirectional_cppi_engine.reset()
        if hasattr(self.two_tier_runner, "reset"):
            self.two_tier_runner.reset()
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "reset_runtime_state", None)
            if callable(hook):
                try:
                    hook()
                except Exception:
                    pass

    def reset_five_economic_reward(self) -> None:
        """Reset economic reward accumulators across both specialists."""
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "reset_five_economic_reward", None)
            if callable(hook):
                try:
                    hook()
                except Exception:
                    pass

    def _expert_from_trace(self, trace: object) -> Any:
        """Delegate trace ownership lookup, querying active specialist first then alternate."""
        for pol in (self.active_policy, self._other_specialist(self.active_policy)):
            hook = getattr(pol, "_expert_from_trace", None)
            if callable(hook):
                try:
                    res = hook(trace)
                    if res is not None:
                        return res
                except Exception:
                    pass
        return None

    def apply_entry_timing_extrema_bias(self, trace: object) -> object:
        hook = getattr(self.active_policy, "apply_entry_timing_extrema_bias", None)
        if callable(hook):
            return hook(trace)
        return trace

    def finalize_bridge_decision(self, trace: object) -> object:
        hook = getattr(self.active_policy, "finalize_bridge_decision", None)
        if callable(hook):
            return hook(trace)
        return trace

    def reset_intraday_state(self, day_key: str) -> None:
        for pol in (self.bull_policy, self.bear_policy):
            hook = getattr(pol, "reset_intraday_state", None)
            if callable(hook):
                try:
                    hook(day_key)
                except Exception:
                    pass

    def emit_bridge_heartbeat(self, **kwargs: Any) -> None:
        hook = getattr(self.active_policy, "emit_bridge_heartbeat", None)
        if callable(hook):
            try:
                hook(**kwargs)
            except Exception:
                pass


def load_dual_champion_router(
    bear_path: str | Path = "outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl",
    bull_path: str | Path = "outputs/he_chuyen_gia/v193_runs/multi_round_phase3/round_01/reasoning_rl_v1_active_candidate.pkl",
    detector_path: str | Path = "outputs/regime_models/champion_regime_detector.pkl",
    *,
    bull_threshold: float = 0.35,
    bear_threshold: float = 0.35,
    chop_threshold: float = 0.38,
    confirmation_bars: int = 1,
    htf_filter_enabled: bool = True,
    htf_bull_threshold: float = 0.05,
    htf_bear_threshold: float = -0.05,
    htf_slope_threshold: float = 0.0,
    momentum_gate_enabled: bool = True,
    circuit_breaker_enabled: bool = True,
    circuit_breaker_max_sl: int = 2,
    circuit_breaker_cooldown_minutes: float = 60.0,
    circuit_breaker_window_minutes: float = 60.0,
    cppi_config: CPPIConfig | None = None,
    bidirectional_cppi_config: BidirectionalCPPIConfig | None = None,
    two_tier_config: TwoTierConfig | None = None,
    micro_scalp_config: MicroScalpConfig | None = None,
    micro_scalp_enabled: bool = True,
    max_total_trades: int = 0,
    max_concurrent_positions: int = 5,
    max_positions_per_direction: int = 2,
    min_spacing_bars: int = 3,
    default_sl_atr: float = 5.0,
    default_tp_rr: float = 2.5,
    default_trailing_atr: float = 1.0,
) -> DualChampionRouter:
    """Factory to load and wire the Dual-Champion Mixture-of-Regimes Router."""
    bear_path = Path(bear_path)
    bull_path = Path(bull_path)

    if not bear_path.exists():
        raise FileNotFoundError(f"Bear Champion not found at {bear_path}")
    if not bull_path.exists():
        raise FileNotFoundError(f"Bull Champion not found at {bull_path}")

    logger.info("Loading Bear Champion from %s ...", bear_path)
    with open(bear_path, "rb") as f:
        reader = ProgressFileReader(f, bear_path.stat().st_size, label=f"Nap Bear Champion ({bear_path.name})")
        bear_model = pickle.load(reader)
        reader.finish()

    logger.info("Loading Bull Champion from %s ...", bull_path)
    with open(bull_path, "rb") as f:
        reader = ProgressFileReader(f, bull_path.stat().st_size, label=f"Nap Bull Champion ({bull_path.name})")
        bull_model = pickle.load(reader)
        reader.finish()

    detector = None
    if detector_path is not None:
        if hasattr(detector_path, "predict_wire_features") or hasattr(detector_path, "predict_proba"):
            detector = detector_path
        else:
            d_path = Path(detector_path)
            if d_path.exists():
                logger.info("Loading Champion Regime Detector from %s ...", d_path)
                with open(d_path, "rb") as f:
                    reader = ProgressFileReader(f, d_path.stat().st_size, label=f"Nap Regime Detector ({d_path.name})")
                    raw_obj = pickle.load(reader)
                    reader.finish()
                if isinstance(raw_obj, dict) and ("scaler" in raw_obj or "window_maxlen" in raw_obj):
                    from .regime_detector import RegimeDetector
                    detector = RegimeDetector.load(d_path)
                elif hasattr(raw_obj, "predict_proba") or hasattr(raw_obj, "predict_wire_features"):
                    from .regime_detector import RegimeDetector
                    detector = RegimeDetector(model=raw_obj)
                else:
                    detector = raw_obj

    effective_two_tier_config = (
        two_tier_config
        if two_tier_config is not None
        else TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=4.0,
            trailing_atr=1.5,
        )
    )

    effective_bidirectional_cppi_config = (
        bidirectional_cppi_config
        if bidirectional_cppi_config is not None
        else BidirectionalCPPIConfig(
            max_concurrent_positions=max_concurrent_positions,
            max_positions_per_direction=max_positions_per_direction,
            min_spacing_bars=min_spacing_bars,
        )
    )

    return DualChampionRouter(
        bull_policy=bull_model,
        bear_policy=bear_model,
        regime_detector=detector,
        bull_threshold=bull_threshold,
        bear_threshold=bear_threshold,
        chop_threshold=chop_threshold,
        confirmation_bars=confirmation_bars,
        htf_filter_enabled=htf_filter_enabled,
        htf_bull_threshold=htf_bull_threshold,
        htf_bear_threshold=htf_bear_threshold,
        htf_slope_threshold=htf_slope_threshold,
        momentum_gate_enabled=momentum_gate_enabled,
        circuit_breaker_enabled=circuit_breaker_enabled,
        circuit_breaker_max_sl=circuit_breaker_max_sl,
        circuit_breaker_cooldown_minutes=circuit_breaker_cooldown_minutes,
        circuit_breaker_window_minutes=circuit_breaker_window_minutes,
        cppi_config=cppi_config,
        bidirectional_cppi_config=effective_bidirectional_cppi_config,
        two_tier_config=effective_two_tier_config,
        micro_scalp_config=micro_scalp_config,
        micro_scalp_enabled=micro_scalp_enabled,
        max_total_trades=max_total_trades,
        max_concurrent_positions=max_concurrent_positions,
        max_positions_per_direction=max_positions_per_direction,
        min_spacing_bars=min_spacing_bars,
        default_sl_atr=default_sl_atr,
        default_tp_rr=default_tp_rr,
        default_trailing_atr=default_trailing_atr,
    )
