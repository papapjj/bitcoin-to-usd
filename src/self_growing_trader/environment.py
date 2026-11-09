from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .config import AppConfig
from .features import (
    BASE_MARKET_FEATURE_COLUMNS,
    EXTRA_MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)
from .protocol_contract import (
    BASE_MARKET_FEATURE_COUNT,
    EXTRA_MARKET_FEATURE_COUNT,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    TIMING_FEATURE_COUNT,
    WYCKOFF_FEATURE_COUNT,
    finite_vector,
    position_limit_enabled,
    require_supported_position_limit,
)
from .input_normalization import normalize_input_vector
from .execution_contract import FACTORIZED_ACTION_MARKER_OUTPUT_INDEX
from .five_lifecycle_reward import (
    FiveLifecycleRewardConfig,
    shape_five_entry_lifecycle,
)
from .pattern_discovery_reward import PatternDiscoveryRewardOracle
from .five_agent_authority import (
    FIVE_AGENT_AUTHORITY_VERSION,
    resolve_opposite_entry_conflict,
    five_agent_authority,
    five_agent_responsibility_active,
)
from .orders import (
    Action,
    DecodedDecision,
    PendingOrder,
    Position,
    PositionManagementMode,
    FACTORIZED_ACTION_MARKER_THRESHOLD,
    allowed_actions_for_state,
    build_sl_tp,
    conservative_exit,
    decode_outputs,
    pending_entry_bid_basis,
    pnl_for_exit,
    EMERGENCY_CLOSE_R,
    MIN_MANUAL_CLOSE_AGE_BARS,
    POSITION_REQUIRED_ACTIONS,
)


@dataclass
class Trade:
    side: int
    entry: float
    exit: float
    pnl: float
    r: float
    opened_bar: int
    closed_bar: int
    reason: str
    risk_pct: float
    # Causal provenance is carried from the decision which opened exposure and
    # any later explicit management decision. This prevents a close on bar t
    # from being rewarded to an unrelated latest action.
    origin_trace_id: str = ""
    management_trace_id: str = ""
    entry_credit_r: float = 0.0
    management_credit_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    locked_profit_r: float = 0.0
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    trailing_stop_atr: float = 0.0
    trailing_applied: bool = False
    exit_kind: str = ""
    lifecycle_final_close: bool = True
    pending_fill_bars: int | None = None


@dataclass
class BacktestResult:
    fitness: float
    equity: float
    equity_curve: list[float]
    trades: list[Trade]
    daily_returns: list[float]
    daily_trade_counts: list[int]
    daily_bar_counts: list[int]
    min_daily_trades: int
    max_daily_trades_limit: int
    max_drawdown_pct: float
    max_drawdown_r: float
    total_r: float
    expectancy: float
    winrate: float
    ruin: bool
    risk_abuse_penalty: float
    causal_credit_coverage: float = 0.0
    unattributed_trade_count: int = 0
    recorded_actions: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ScheduledDecision:
    decision: DecodedDecision
    source_idx: int
    trace_id: str = ""
    remaining_delay: int = 0


@dataclass
class TradingEnvironment:
    """Causal, close-marked offline execution reference.

    It deliberately does *not* claim tick fidelity. A decision observed after
    bar t executes at open(t+1); a pending order created at t cannot trigger
    until a later bar; a position opened on a bar cannot be stopped/targeted by
    that bar's already-known high/low. This removes the old intrabar
    look-through bias. Promotion must still use MT5 every-tick evidence.
    """

    data: pd.DataFrame
    config: AppConfig
    equity: float = field(init=False)
    realized_equity: float = field(init=False)
    peak_equity: float = field(init=False)
    positions: list[Position] = field(default_factory=list, init=False)
    pending_orders: list[PendingOrder] = field(default_factory=list, init=False)
    trades: list[Trade] = field(default_factory=list, init=False)
    equity_curve: list[float] = field(default_factory=list, init=False)
    day_start_equity: float = field(init=False)
    current_day: object = field(default=None, init=False)
    current_day_trade_count: int = field(default=0, init=False)
    current_day_bar_count: int = field(default=0, init=False)
    daily_trade_counts: list[int] = field(default_factory=list, init=False)
    daily_bar_counts: list[int] = field(default_factory=list, init=False)
    daily_returns: list[float] = field(default_factory=list, init=False)
    risk_samples: list[float] = field(default_factory=list, init=False)
    ruin: bool = field(default=False, init=False)
    simulated_leverage: float = 100.0
    enforce_simulated_margin: bool = False

    # Pre-prepared arrays
    open_values: np.ndarray = field(default=None, repr=False)
    high_values: np.ndarray = field(default=None, repr=False)
    low_values: np.ndarray = field(default=None, repr=False)
    close_values: np.ndarray = field(default=None, repr=False)
    spread_values: np.ndarray = field(default=None, repr=False)
    atr_values: np.ndarray = field(default=None, repr=False)
    spread_atr_values: np.ndarray = field(default=None, repr=False)
    not_shock_values: np.ndarray = field(default=None, repr=False)
    feature_values: np.ndarray = field(default=None, repr=False)
    extra_feature_values: np.ndarray = field(default=None, repr=False)
    timing_feature_values: np.ndarray = field(default=None, repr=False)
    wyckoff_feature_values: np.ndarray = field(default=None, repr=False)
    day_values: np.ndarray = field(default=None, repr=False)
    day_progress_values: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        require_supported_position_limit(self.config.risk.max_positions)
        if self.open_values is None:
            self._prepare_arrays()
        self.reset()

    def _prepare_arrays(self) -> None:
        self.open_values = self.data["Open"].to_numpy(dtype=float, copy=True)
        self.high_values = self.data["High"].to_numpy(dtype=float, copy=True)
        self.low_values = self.data["Low"].to_numpy(dtype=float, copy=True)
        self.close_values = self.data["Close"].to_numpy(dtype=float, copy=True)
        self.spread_values = self.data["spread_price"].to_numpy(dtype=float, copy=True)
        self.atr_values = self.data["atr"].to_numpy(dtype=float, copy=True)
        self.spread_atr_values = self.data["spread_atr"].to_numpy(dtype=float, copy=True)
        self.not_shock_values = self.data["not_in_shock_regime"].to_numpy(dtype=float, copy=True)
        self.feature_values = self.data[BASE_MARKET_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        self.extra_feature_values = self.data[EXTRA_MARKET_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        self.timing_feature_values = self.data[TIMING_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        self.wyckoff_feature_values = self.data[WYCKOFF_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        datetimes = pd.to_datetime(self.data["Datetime"])
        self.day_values = datetimes.dt.date.to_numpy()
        minute_of_day = datetimes.dt.hour.to_numpy(dtype=float) * 60.0 + datetimes.dt.minute.to_numpy(dtype=float)
        self.day_progress_values = np.clip(minute_of_day / 1440.0, 0.0, 1.0)

    def reset(self) -> None:
        self.realized_equity = float(self.config.risk.initial_equity)
        self.equity = float(self.config.risk.initial_equity)
        self.peak_equity = self.equity
        self.positions = []
        self.pending_orders = []
        self.trades = []
        self.equity_curve = []
        self.day_start_equity = self.equity
        self.current_day = None
        self.current_day_trade_count = 0
        self.current_day_bar_count = 0
        self.daily_trade_counts = []
        self.daily_bar_counts = []
        self.daily_returns = []
        self.risk_samples = []
        self.ruin = False
        self.recorded_actions: list[int] = []

    def observation_size(self) -> int:
        return INPUT_COUNT

    def used_margin(self, idx: int | None = None) -> float:
        """Ước lượng ký quỹ đang dùng từ giá trị danh nghĩa và đòn bẩy mô phỏng.

        ``Position.size`` của môi trường là số đơn vị giá sao cho PnL = biến động
        giá × size. Vì vậy giá trị danh nghĩa tại giá đánh dấu là ``size × price``.
        Đòn bẩy chỉ phục vụ mô phỏng ký quỹ/quan sát; nó không thay đổi PnL hay
        ngân sách rủi ro đã có của môi trường.
        """
        leverage = max(float(self.simulated_leverage), 1e-9)
        if not self.positions:
            return 0.0
        if idx is None:
            idx = max(0, min(len(self.close_values) - 1, len(self.equity_curve) - 1))
        idx = max(0, min(int(idx), len(self.close_values) - 1))
        close_bid = float(self.close_values[idx])
        spread = float(self.spread_values[idx])
        total = 0.0
        for position in self.positions:
            mark = close_bid if position.side > 0 else close_bid + spread
            total += abs(float(position.size)) * max(abs(mark), 1e-12) / leverage
        return float(max(0.0, total))

    def free_margin(self, idx: int | None = None) -> float:
        return float(self.equity - self.used_margin(idx))

    def margin_level_pct(self, idx: int | None = None) -> float:
        used = self.used_margin(idx)
        if used <= 1e-12:
            return float('inf')
        return float(100.0 * self.equity / used)

    def account_snapshot(self, idx: int, *, episode: int = 0, mode: str = '') -> dict[str, object]:
        """Ảnh tài khoản nhỏ gọn để giao diện vẽ trực tiếp theo từng thanh giá."""
        used = self.used_margin(idx)
        free = float(self.equity - used)
        level = float(100.0 * self.equity / used) if used > 1e-12 else None
        return {
            'bar_index': int(idx),
            'time': str(self.data.iloc[int(idx)]['Datetime']),
            'episode': int(episode),
            'mode': str(mode),
            'balance': float(self.realized_equity),
            'equity': float(self.equity),
            'used_margin': float(used),
            'free_margin': float(free),
            'margin_level_pct': level,
            'drawdown_pct': float(100.0 * self.drawdown_pct()),
            'open_positions': int(len(self.positions)),
            'pending_orders': int(len(self.pending_orders)),
            'closed_trades': int(len(self.trades)),
            'total_open_risk_pct': float(self.total_open_risk()),
            'ruin': bool(self.ruin),
        }

    def observation(self, idx: int) -> np.ndarray:
        atr = max(float(self.atr_values[idx]), 1e-12)
        close_bid = float(self.close_values[idx])
        spread_price = float(self.spread_values[idx])
        market_features = self.feature_values[idx][:BASE_MARKET_FEATURE_COUNT]

        slots = np.zeros(MAX_POSITION_SLOTS * 8, dtype=np.float32)
        for i, position in enumerate(self.positions[:MAX_POSITION_SLOTS]):
            current_exit = close_bid if position.side == 1 else close_bid + spread_price
            price_move_atr = position.side * (close_bid - position.entry) / atr
            age_norm = (idx - position.opened_bar) / max(self.config.orders.max_holding_bars, 1)
            sl_distance_atr = abs(position.entry - position.sl) / atr
            tp_distance_atr = abs(position.tp - position.entry) / atr
            rr_norm = min(tp_distance_atr / max(sl_distance_atr, 1e-12), 5.0) / 5.0
            unrealized_r = position.side * (current_exit - position.entry) / max(position.risk_distance, 1e-12)
            offset = i * 8
            slots[offset:offset + 8] = (
                1.0,
                float(position.side),
                float(np.clip(price_move_atr, -20.0, 20.0)),
                float(np.clip(age_norm, 0.0, 5.0)),
                float(sl_distance_atr),
                float(tp_distance_atr),
                float(rr_norm),
                float(np.clip(unrealized_r, -5.0, 5.0)),
            )

        # The old quota/cadence features remain in the versioned bridge schema
        # schema but are neutral when no explicit quota profile is enabled.
        day_progress = float(self.day_progress_values[idx])
        min_daily = int(self.config.orders.min_daily_trades)
        max_daily = int(self.config.orders.max_daily_trades)
        expected_trades_now = min_daily * day_progress if min_daily > 0 else 0.0
        cadence_deficit = (
            max(0.0, expected_trades_now - self.current_day_trade_count) / max(min_daily, 1)
            if min_daily > 0 else 0.0
        )
        account = np.array([
            len(self.positions) / MAX_POSITION_SLOTS,
            self.total_open_risk(),
            self.drawdown_pct(),
            (self.equity / self.config.risk.initial_equity) - 1.0,
            min(self.current_day_trade_count, max_daily) / max(max_daily, 1) if max_daily > 0 else 0.0,
            min(self.current_day_trade_count, min_daily) / max(min_daily, 1) if min_daily > 0 else 0.0,
            day_progress,
            float(np.clip(cadence_deficit, 0.0, 1.0)),
        ], dtype=np.float32)

        execution = np.zeros(1 + MAX_POSITION_SLOTS * 2, dtype=np.float32)
        execution[0] = float(self.spread_atr_values[idx])
        for i, position in enumerate(self.positions[:MAX_POSITION_SLOTS]):
            exit_price = close_bid if position.side == 1 else close_bid + spread_price
            if position.side == 1:
                dist_to_sl = (exit_price - position.sl) / atr
                dist_to_tp = (position.tp - exit_price) / atr
            else:
                dist_to_sl = (position.sl - exit_price) / atr
                dist_to_tp = (exit_price - position.tp) / atr
            offset = 1 + i * 2
            execution[offset] = float(np.clip(dist_to_sl, -20.0, 20.0))
            execution[offset + 1] = float(np.clip(dist_to_tp, -20.0, 20.0))

        extra_market = self.extra_feature_values[idx][:EXTRA_MARKET_FEATURE_COUNT]
        timing = self.timing_feature_values[idx][:TIMING_FEATURE_COUNT]
        wyckoff = self.wyckoff_feature_values[idx][:WYCKOFF_FEATURE_COUNT]
        observation = np.concatenate([market_features, slots, account, extra_market, execution, timing, wyckoff])
        raw_observation = finite_vector(observation, size=INPUT_COUNT, name="environment observation")
        normalization_mode = str(
            getattr(getattr(self.config, "learning", None), "input_normalization_mode", "raw")
            or "raw"
        )
        return normalize_input_vector(raw_observation, mode=normalization_mode, size=INPUT_COUNT)

    @staticmethod
    def _trace_id(trace: object | None, source_idx: int) -> str:
        if trace is None:
            return ""
        value = str(getattr(trace, "trace_id", "") or "")
        if not value:
            value = f"offline-{source_idx}"
            try:
                setattr(trace, "trace_id", value)
            except Exception:
                pass
        return value

    def _observe_trade_credits(self, policy, traces: dict[str, object], idx: int, closed: list[Trade], *, done: bool) -> None:
        """Deliver final rewards only to their originating/management traces."""
        if not closed or not hasattr(policy, "observe_transition"):
            return
        credits: dict[str, float] = {}
        for trade in closed:
            if trade.origin_trace_id:
                credits[trade.origin_trace_id] = credits.get(trade.origin_trace_id, 0.0) + float(trade.entry_credit_r)
            if trade.management_trace_id:
                credits[trade.management_trace_id] = credits.get(trade.management_trace_id, 0.0) + float(trade.management_credit_r)
        obs = self.observation(idx)
        for trace_id, reward in credits.items():
            trace = traces.get(trace_id)
            if trace is None:
                continue
            confidence = float(getattr(trace, "confidence", 1.0) or 1.0)
            signal = policy.observe_transition(trace, obs, float(reward), 0.001 * max(0.0, 1.0 - confidence), bool(done))

    def run(self, policy, *, learning_enabled: bool = True) -> BacktestResult:
        """Run a causally ordered offline reference backtest.

        It is intentionally a low-fidelity diagnostic engine, not promotion
        evidence. Validation/test callers set ``learning_enabled=False``; no
        policy mutation occurs in that mode.
        """
        self.reset()
        scheduled: ScheduledDecision | None = None
        has_act = hasattr(policy, "act")
        trace_registry: dict[str, object] = {}
        step_count = 0

        for idx in range(len(self.data)):
            self._roll_day(idx)
            if scheduled is not None:
                if scheduled.remaining_delay <= 0:
                    self._apply_decision(idx, scheduled.decision, scheduled.source_idx, scheduled.trace_id)
                    scheduled = None
                else:
                    scheduled = ScheduledDecision(
                        scheduled.decision,
                        scheduled.source_idx,
                        scheduled.trace_id,
                        scheduled.remaining_delay - 1,
                    )

            # Pending orders only become eligible on a later bar; orders and
            # positions created on this bar cannot use this bar's known high/low.
            self._trigger_pending(idx)
            self._update_open_positions(idx)
            self._mark_to_market_equity(idx)
            self.equity_curve.append(self.equity)
            self.peak_equity = max(self.peak_equity, self.equity)

            if self.equity <= 0.0:
                self.ruin = True

            closed_this_bar = [trade for trade in self.trades if trade.closed_bar == idx]
            if learning_enabled:
                self._observe_trade_credits(policy, trace_registry, idx, closed_this_bar, done=self.ruin)

            if self.ruin:
                break

            if idx < len(self.data) - 1 and scheduled is None:
                obs = self.observation(idx)
                trace = None
                if has_act:
                    trace = policy.act(obs)
                    outputs = getattr(trace, "selected_action", trace)
                else:
                    outputs = policy(obs)
                trace_id = self._trace_id(trace, idx)
                if trace_id:
                    trace_registry[trace_id] = trace
                allowed_actions = self._allowed_actions(idx)
                values = np.asarray(outputs, dtype=float).reshape(-1)
                factorized = bool(
                    values.size > FACTORIZED_ACTION_MARKER_OUTPUT_INDEX
                    and float(values[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX]) >= FACTORIZED_ACTION_MARKER_THRESHOLD
                )
                if factorized and any(
                    item in allowed_actions
                    for item in (Action.CLOSE_WORST, Action.CLOSE_BEST, Action.CLOSE_ALL)
                ):
                    allowed_actions.add(Action.MANAGE_POSITION)
                decision = decode_outputs(
                    outputs,
                    self.config.risk,
                    self.config.orders,
                    allowed_actions,
                    max_positions=max(1, len(self.positions)),
                )
                # The offline engine is bar-based. Contract-v5 delay 0 means
                # execute on the next causal bar (the existing simulator rule);
                # delay 1..3 adds that many whole bars before execution. While
                # an action is pending, the policy is not queried again.
                extra_delay = int(decision.action_delay_ticks) if decision.factorized_contract else 0
                scheduled = ScheduledDecision(decision, idx, trace_id, max(0, min(3, extra_delay)))
                step_count += 1
            elif idx >= len(self.data) - 1:
                scheduled = None

        final_idx = max(0, min(len(self.data) - 1, len(self.data) - 1))
        before_close = len(self.trades)
        self._close_all(final_idx, "end")
        self._mark_to_market_equity(final_idx)
        if learning_enabled and len(self.trades) > before_close:
            self._observe_trade_credits(policy, trace_registry, final_idx, self.trades[before_close:], done=True)
        if self.current_day is not None and self.day_start_equity > 0:
            self.daily_returns.append((self.equity / self.day_start_equity) - 1.0)
            self.daily_trade_counts.append(self.current_day_trade_count)
            self.daily_bar_counts.append(self.current_day_bar_count)
        return self._result()

    @staticmethod
    def _five_trace_id(trace: object | None, source_idx: int, agent_index: int) -> str:
        value = '' if trace is None else str(getattr(trace, 'trace_id', '') or '')
        if not value:
            value = f'offline-five-{source_idx}-a{agent_index}'
            if trace is not None:
                try:
                    setattr(trace, 'trace_id', value)
                except Exception:
                    pass
        return value

    def _five_owner_state(
        self,
        idx: int,
        trace_owner: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trả trạng thái reward tích lũy theo đúng tác nhân tạo vị thế."""
        marked = np.zeros(5, dtype=np.float32)
        risk = np.zeros(5, dtype=np.float32)
        closed = np.zeros(5, dtype=np.float32)
        wins = np.zeros(5, dtype=np.float32)
        expectancy_shortfall = np.zeros(5, dtype=np.float32)
        tail_loss_cost = np.zeros(5, dtype=np.float32)
        realized_trade_r = np.zeros(5, dtype=np.float32)
        learning = self.config.learning
        expectancy_floor = float(getattr(learning, "five_reward_expectancy_floor_r", 0.05))
        tail_threshold = float(getattr(learning, "five_reward_tail_loss_threshold_r", 1.0))
        tail_cap = float(getattr(learning, "five_reward_tail_loss_cap_r", 3.0))
        for trade in self.trades:
            agent = trace_owner.get(str(trade.origin_trace_id or ''))
            if agent is not None and 0 <= int(agent) < 5:
                owner = int(agent)
                trade_r = float(trade.r)
                marked[owner] += np.float32(trade_r)
                realized_trade_r[owner] += np.float32(trade_r)
                closed[owner] += np.float32(1.0)
                wins[owner] += np.float32(trade_r > 0.0)
                expectancy_shortfall[owner] += np.float32(max(0.0, expectancy_floor - trade_r))
                tail_excess = min(max(0.0, -trade_r - tail_threshold), tail_cap)
                tail_loss_cost[owner] += np.float32(tail_excess * tail_excess)
        for position in self.positions:
            agent = trace_owner.get(str(position.origin_trace_id or ''))
            if agent is None or not 0 <= int(agent) < 5:
                continue
            marked[int(agent)] += np.float32(self._unrealized_r(position, idx, 'Close'))
            # Khớp đường MT5 hiện tại: mỗi vị thế đang mở tương đương một đơn vị
            # rủi ro cạnh tranh; >1 trên cùng tác nhân mới phát sinh phạt vượt mức.
            risk[int(agent)] += np.float32(1.0)
        return marked, risk, closed, wins, expectancy_shortfall, tail_loss_cost, realized_trade_r

    def _route_five_lifecycle_shaping(
        self,
        policy: object,
        trades: list[Trade],
        trace_actions: dict[str, int],
    ) -> dict[str, int]:
        """Route lifecycle V4 ve dung packet entry; khong tron vao economic step."""

        hook = getattr(policy, "apply_five_entry_lifecycle_reward", None)
        if not callable(hook):
            return {"routed": 0, "missed": len(trades)}
        reward_config = FiveLifecycleRewardConfig.from_source(self.config.learning)
        routed = missed = 0
        for trade in trades:
            if not bool(trade.lifecycle_final_close):
                continue
            trace_id = str(trade.origin_trace_id or "")
            action = trace_actions.get(trace_id)
            if not trace_id or action not in {1, 2, 3, 4, 5, 6}:
                missed += 1
                continue
            shaped = shape_five_entry_lifecycle(
                action_index=int(action),
                sl_atr=float(trade.sl_atr),
                tp_atr=float(trade.tp_atr),
                trailing_atr=float(trade.trailing_stop_atr),
                trailing_applied=bool(trade.trailing_applied),
                locked_profit_r=float(trade.locked_profit_r),
                mfe_r=float(trade.mfe_r),
                mae_r=float(trade.mae_r),
                mfe_available=True,
                realized_r=float(trade.r),
                exit_kind=str(trade.exit_kind),
                pending_fill_bars=trade.pending_fill_bars,
                config=reward_config,
            )
            result = hook(
                trace_id,
                lifecycle_reward_r=float(shaped.total_r),
                realized_r=float(trade.r),
                components=shaped.as_dict(),
            )
            if isinstance(result, dict) and bool(result.get("accepted", False)):
                routed += 1
            else:
                missed += 1
        return {"routed": routed, "missed": missed}

    def _apply_five_decision_with_conflict(
        self, idx: int, scheduled: ScheduledDecision
    ) -> float:
        """Thực thi một quyết định đã đóng băng và trả chi phí xung đột nếu bị từ chối."""
        decision = scheduled.decision
        action = decision.action
        if action == Action.HOLD:
            return 0.0
        before_positions = len(self.positions)
        before_pending = len(self.pending_orders)
        before_trades = len(self.trades)
        before_signature = tuple(
            (p.origin_trace_id, p.size, p.risk_amount, p.sl, p.tp) for p in self.positions
        )
        self._apply_decision(idx, decision, scheduled.source_idx, scheduled.trace_id)
        if action == Action.CANCEL_ORDERS:
            return 0.0
        if action in {
            Action.MARKET_BUY, Action.MARKET_SELL, Action.BUY_LIMIT, Action.SELL_LIMIT,
            Action.BUY_STOP, Action.SELL_STOP,
        }:
            accepted = len(self.positions) > before_positions or len(self.pending_orders) > before_pending
            return 0.0 if accepted else 0.5
        if action == Action.MANAGE_POSITION:
            if decision.management_mode == PositionManagementMode.NONE:
                return 0.0
            after_signature = tuple(
                (p.origin_trace_id, p.size, p.risk_amount, p.sl, p.tp) for p in self.positions
            )
            changed = (len(self.trades) > before_trades) or (after_signature != before_signature)
            return 0.0 if changed else 0.5
        if action in {Action.CLOSE_WORST, Action.CLOSE_BEST, Action.CLOSE_ALL}:
            changed = len(self.trades) > before_trades or len(self.positions) < before_positions
            return 0.0 if changed else 0.5
        return 0.0

    def run_five(
        self,
        policy,
        *,
        learning_enabled: bool = True,
        start_idx: int = 0,
        end_idx: int | None = None,
        episode: int = 0,
        mode: str = 'time',
        training_progress_offset: float = 0.0,
        training_progress_scale: float = 1.0,
        telemetry_callback: Callable[[dict[str, object]], None] | None = None,
        pattern_reward_oracle: PatternDiscoveryRewardOracle | None = None,
    ) -> BacktestResult:
        """Chạy đúng đường năm tác nhân trên cùng môi trường tài khoản.

        Năm tác nhân dùng chung quan sát tài khoản nhưng có ký ức riêng. Mỗi gói
        quyết định được đóng băng đồng thời; độ trễ lớn nhất trong gói khóa lần
        quyết định kế tiếp, vì hành động đang chờ không được phép bị ghi đè.
        Lợi nhuận đang mở/đã đóng được quy về tác nhân tạo vị thế bằng trace_id.
        """
        self.reset()
        if len(self.data) <= 0:
            return self._result()
        start = max(0, min(int(start_idx), len(self.data) - 1))
        stop = len(self.data) - 1 if end_idx is None else max(start, min(int(end_idx), len(self.data) - 1))
        act_five = getattr(policy, 'act_nam_tac_nhan', None)
        if not callable(act_five):
            act_five = getattr(policy, 'act_five_agents', None)
        if not callable(act_five):
            raise RuntimeError('moi truong nam tac nhan can act_nam_tac_nhan() hoac act_five_agents()')
        economic_hook = getattr(policy, 'observe_five_economic_transition', None)
        if learning_enabled and not callable(economic_hook):
            raise RuntimeError('moi truong nam tac nhan can observe_five_economic_transition()')
        pattern_hook = getattr(policy, 'observe_five_pattern_transition', None)
        if learning_enabled and pattern_reward_oracle is not None and not callable(pattern_hook):
            raise RuntimeError('moi truong pattern reward can observe_five_pattern_transition()')
        try:
            setattr(policy, '_learning_enabled', bool(learning_enabled))
        except Exception:
            pass
        for exp in getattr(policy, "experts", {}).values():
            try:
                setattr(getattr(exp, "policy", None), "_learning_enabled", bool(learning_enabled))
            except Exception:
                pass
        registry = getattr(policy, "registry", None)
        if registry is not None:
            for exp in getattr(registry, "chuyen_gia_theo_context", {}).values():
                try:
                    setattr(getattr(exp, "policy", None), "_learning_enabled", bool(learning_enabled))
                except Exception:
                    pass

        trace_owner: dict[str, int] = {}
        trace_actions: dict[str, int] = {}
        scheduled: list[ScheduledDecision | None] = [None] * 5
        feedback_traces: tuple[object, ...] | None = None
        last_conflict = np.zeros(5, dtype=np.float32)
        last_policy_violation_penalty = np.zeros(5, dtype=np.float32)
        episode_start_time = pd.to_datetime(self.data.iloc[start]['Datetime'])
        last_feedback_time = episode_start_time
        total_span = max(1, stop - start + 1)
        terminal_sent = False
        pattern_reward_pending = pattern_reward_oracle is not None
        pattern_anchor_traces: tuple[object, ...] | None = None
        pattern_anchor_rewards = np.zeros(5, dtype=np.float32)
        lifecycle_trade_cursor = 0

        for idx in range(start, stop + 1):
            self._roll_day(idx)
            conflict = np.zeros(5, dtype=np.float32)
            for agent_index, item in enumerate(scheduled):
                if item is None:
                    continue
                if item.remaining_delay <= 0:
                    conflict[agent_index] = np.float32(self._apply_five_decision_with_conflict(idx, item))
                    scheduled[agent_index] = None
                else:
                    scheduled[agent_index] = ScheduledDecision(
                        item.decision, item.source_idx, item.trace_id, item.remaining_delay - 1
                    )

            self._trigger_pending(idx)
            self._update_open_positions(idx)
            self._mark_to_market_equity(idx)
            self.equity_curve.append(self.equity)
            self.peak_equity = max(self.peak_equity, self.equity)
            if self.equity <= 0.0:
                self.ruin = True

            if telemetry_callback is not None:
                snapshot = self.account_snapshot(idx, episode=episode, mode=mode)
                snapshot['training_progress'] = float(np.clip(
                    training_progress_offset + training_progress_scale * ((idx - start + 1) / total_span), 0.0, 1.0
                ))
                telemetry_callback(snapshot)

            last_conflict = (last_conflict + conflict).astype(np.float32)
            packet_finished = feedback_traces is not None and all(item is None for item in scheduled)
            # Nếu gói đã thực thi nhưng không còn đủ bốn thanh để phát một gói
            # mới an toàn với delay=3, giữ chính trace đó tới cuối tập phim.
            # Nhờ vậy terminal reward vẫn gắn vào một hành động thật, không
            # nhân đôi mẫu PPO và không cắt bỏ phần cuối của quỹ đạo thị trường.
            defer_terminal_feedback = bool(packet_finished and not self.ruin and idx < stop and idx + 4 > stop)
            if packet_finished and learning_enabled and not defer_terminal_feedback:
                marked, risk, closed, wins, expectancy_shortfall, tail_loss_cost, realized_trade_r = self._five_owner_state(idx, trace_owner)
                new_lifecycle_trades = self.trades[lifecycle_trade_cursor:]
                lifecycle_trade_cursor = len(self.trades)
                lifecycle_route_stats = self._route_five_lifecycle_shaping(
                    policy, new_lifecycle_trades, trace_actions,
                )
                now_time = pd.to_datetime(self.data.iloc[idx]['Datetime'])
                elapsed_seconds = max(0.0, float((now_time - episode_start_time).total_seconds()))
                step_seconds = max(1.0, float((now_time - last_feedback_time).total_seconds()))
                terminal = bool(self.ruin or idx >= stop)
                progress = float(np.clip(
                    training_progress_offset + training_progress_scale * ((idx - start + 1) / total_span), 0.0, 1.0
                ))
                kwargs = dict(
                    agent_mark_to_market_r=marked,
                    # PnL mô phỏng đã chứa spread/slippage trong giá khớp; trừ
                    # transaction_cost lần nữa sẽ tính chi phí hai lần.
                    transaction_cost_r=np.zeros(5, dtype=np.float32),
                    agent_risk_ratio=risk,
                    agent_closed_trade_count=closed,
                    agent_winning_trade_count=wins,
                    agent_expectancy_shortfall_r=expectancy_shortfall,
                    agent_tail_loss_cost_r=tail_loss_cost,
                    agent_realized_trade_r=realized_trade_r,
                    conflict_cost_r=last_conflict.astype(np.float32),
                    policy_violation_penalty_r=last_policy_violation_penalty.astype(np.float32),
                    team_equity_r=float((self.equity - self.config.risk.initial_equity) / max(self.config.risk.initial_equity * 0.01, 1e-9)),
                    team_drawdown=float(100.0 * self.drawdown_pct()),
                    terminal=terminal,
                    dones=np.ones(5, dtype=bool) if terminal else np.zeros(5, dtype=bool),
                    elapsed_seconds=elapsed_seconds,
                    step_seconds=step_seconds,
                    training_progress=progress,
                )
                try:
                    stats = economic_hook(feedback_traces, **kwargs)
                except TypeError:
                    kwargs.pop('training_progress', None)
                    stats = economic_hook(feedback_traces, **kwargs)
                if telemetry_callback is not None and isinstance(stats, dict):
                    telemetry_callback({
                        'event': 'learning', 'bar_index': int(idx), 'episode': int(episode), 'mode': str(mode),
                        'five_lifecycle_routed': float(lifecycle_route_stats['routed']),
                        'five_lifecycle_route_missed': float(lifecycle_route_stats['missed']),
                        **{
                            str(k): float(v)
                            for k, v in stats.items()
                            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                        },
                    })
                feedback_traces = None
                last_feedback_time = now_time
                last_conflict.fill(0.0)
                last_policy_violation_penalty.fill(0.0)
                if terminal:
                    terminal_sent = True
            elif packet_finished and not learning_enabled:
                feedback_traces = None
                last_conflict.fill(0.0)
                last_policy_violation_penalty.fill(0.0)

            if self.ruin or terminal_sent:
                break

            if feedback_traces is None and idx < stop and idx + 4 <= stop:
                obs = self.observation(idx)
                allowed_actions = self._allowed_actions(idx)
                mask = np.asarray([Action(i) in allowed_actions for i in range(12)], dtype=bool)
                set_context = getattr(policy, 'set_decision_context', None)
                if callable(set_context):
                    set_context(allowed_action_mask=mask)
                set_account = getattr(policy, 'set_account_context', None)
                if callable(set_account):
                    set_account({
                        'balance': float(self.realized_equity),
                        'equity': float(self.equity),
                        'drawdown_pct': float(100.0 * self.drawdown_pct()),
                        'position_count': float(len(self.positions)),
                        'used_margin': float(self.used_margin(idx)),
                        'free_margin': float(self.free_margin(idx)),
                    }, '')
                traces = tuple(act_five(obs))
                if len(traces) != 5:
                    raise RuntimeError('nam tac nhan phai tra ve dung 5 trace')
                decoded_rows: list[tuple[DecodedDecision, str, float]] = []
                for agent_index, trace in enumerate(traces):
                    authority = five_agent_authority(agent_index)
                    trace_id = self._five_trace_id(trace, idx, agent_index)
                    trace_owner[trace_id] = int(agent_index)
                    outputs = np.asarray(getattr(trace, 'selected_action', trace), dtype=float).reshape(-1)
                    local_allowed = {
                        action for action in allowed_actions
                        if int(action) in authority.allowed_actions
                    }
                    if not local_allowed:
                        local_allowed = {Action.HOLD}
                    factorized = bool(
                        outputs.size > FACTORIZED_ACTION_MARKER_OUTPUT_INDEX
                        and float(outputs[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX]) >= FACTORIZED_ACTION_MARKER_THRESHOLD
                    )
                    if (
                        int(Action.MANAGE_POSITION) in authority.allowed_actions
                        and factorized and any(
                        item in local_allowed for item in (Action.CLOSE_WORST, Action.CLOSE_BEST, Action.CLOSE_ALL)
                        )
                    ):
                        local_allowed.add(Action.MANAGE_POSITION)
                    decision = decode_outputs(
                        outputs, self.config.risk, self.config.orders, local_allowed,
                        max_positions=max(1, len(self.positions)),
                    )
                    if int(decision.action) not in authority.allowed_actions:
                        raise RuntimeError(
                            f'action {int(decision.action)} vuot quyen agent {agent_index}'
                        )
                    action_index = int(decision.action)
                    self.recorded_actions.append(action_index)
                    score = float(outputs[action_index]) if 0 <= action_index < outputs.size else 0.0
                    decoded_rows.append((decision, trace_id, score))

                # Mỗi action khác HOLD chỉ có đúng một owner. Hai owner entry
                # vẫn là hai giả thuyết loại trừ trên cùng snapshot, nên nếu cả
                # hai cùng phát thì chỉ xác suất policy cao hơn được thực thi.
                # CLOSE_ALL của portfolio governor vẫn chặn toàn bộ packet.
                packet: list[ScheduledDecision | None] = [None] * 5
                governor_close_all = decoded_rows[4][0].action == Action.CLOSE_ALL
                entry_indices = [
                    i for i, (decision, _, _) in enumerate(decoded_rows)
                    if decision.action in {
                        Action.MARKET_BUY, Action.MARKET_SELL,
                        Action.BUY_LIMIT, Action.SELL_LIMIT,
                        Action.BUY_STOP, Action.SELL_STOP,
                    }
                ]
                opposite_winner, opposite_blocked = resolve_opposite_entry_conflict(
                    [int(decision.action) for decision, _, _ in decoded_rows],
                    [
                        float(dict(getattr(trace, 'action_details', {}) or {}).get(
                            'five_selected_action_probability', getattr(trace, 'confidence', 0.0)
                        ) or 0.0)
                        for trace in traces
                    ],
                )
                if opposite_blocked is not None:
                    entry_indices = [
                        index for index in entry_indices if index != int(opposite_blocked)
                    ]
                if position_limit_enabled(self.config.risk.max_positions):
                    capacity = max(
                        0,
                        self.config.risk.max_positions
                        - len(self.positions)
                        - len(self.pending_orders),
                    )
                    entry_indices = entry_indices[:capacity]
                risk_scale = 1.0 / len(entry_indices) if entry_indices else 1.0
                for agent_index, (decision, trace_id, _) in enumerate(decoded_rows):
                    if decision.action == Action.HOLD:
                        continue
                    if governor_close_all and agent_index != 4:
                        continue
                    if decision.action in {
                        Action.MARKET_BUY, Action.MARKET_SELL,
                        Action.BUY_LIMIT, Action.SELL_LIMIT,
                        Action.BUY_STOP, Action.SELL_STOP,
                    } and agent_index not in entry_indices:
                        continue
                    scaled = (
                        replace(decision, risk_pct=float(decision.risk_pct) * risk_scale)
                        if agent_index in entry_indices
                        else decision
                    )
                    delay = int(scaled.action_delay_ticks) if scaled.factorized_contract else 0
                    packet[agent_index] = ScheduledDecision(
                        scaled, idx, trace_id, max(0, min(3, delay))
                    )
                scheduled = packet
                annotated_traces = []
                positions_before = int(len(self.positions))
                pending_orders_before = int(len(self.pending_orders))
                invalid_position_action_penalty_r = float(getattr(
                    getattr(self.config, 'learning', None),
                    'five_invalid_position_action_no_position_penalty_r',
                    1.0,
                ))
                invalid_cancel_no_pending_penalty_r = float(getattr(
                    getattr(self.config, 'learning', None),
                    'five_invalid_cancel_no_pending_penalty_r',
                    2.0,
                ))
                owned_agents_before = {
                    int(trace_owner[origin])
                    for origin in (str(position.origin_trace_id or "") for position in self.positions)
                    if origin in trace_owner
                }
                for agent_index, trace in enumerate(traces):
                    authority = five_agent_authority(agent_index)
                    decision = decoded_rows[agent_index][0]
                    sent = packet[agent_index] is not None
                    is_hold = decision.action == Action.HOLD
                    blocked_status = (
                        'BATCH_AUTHORITY_GOVERNOR'
                        if governor_close_all and agent_index != 4 and not is_hold
                        else 'BATCH_OPPOSITE_ENTRY_CONFLICT'
                        if opposite_blocked == agent_index and not is_hold
                        else 'BATCH_ACCOUNT_CAPACITY'
                    )
                    details = dict(getattr(trace, 'action_details', {}) or {})
                    policy_action = int(details.get(
                        'five_action_index', getattr(trace, 'selected_idx', int(decision.action))
                    ) or 0)
                    if policy_action in {1, 2, 3, 4, 5, 6}:
                        trace_actions[self._five_trace_id(trace, idx, agent_index)] = policy_action
                    invalid_position_action_without_position = bool(
                        policy_action in POSITION_REQUIRED_ACTIONS
                        and positions_before == 0
                    )
                    invalid_cancel_without_pending = bool(
                        policy_action == int(Action.CANCEL_ORDERS)
                        and pending_orders_before == 0
                    )
                    owns_position_before = agent_index in owned_agents_before
                    authority_responsibility_active = five_agent_responsibility_active(
                        agent_index,
                        position_count=positions_before,
                        pending_order_count=pending_orders_before,
                        owns_position=owns_position_before,
                    )
                    if invalid_position_action_without_position:
                        last_policy_violation_penalty[agent_index] = np.float32(
                            invalid_position_action_penalty_r
                        )
                    elif invalid_cancel_without_pending:
                        last_policy_violation_penalty[agent_index] = np.float32(
                            invalid_cancel_no_pending_penalty_r
                        )
                    details.update({
                        # PPO hoc dung action da lay mau. Decoded action chi la
                        # lenh sau mask/arbitration va duoc luu rieng de audit.
                        'five_action_index': int(policy_action),
                        'five_decoded_action_index': int(decision.action),
                        # Economic PnL can credit only an action that was sent.
                        # HOLD must never consume another agent's execution
                        # reward. Dataset pattern credit is a separate channel:
                        # it can supervise HOLD when the terminal oracle proves
                        # no-edge, or penalize it when an edge existed.
                        'five_economic_actor_eligible': bool(
                            sent or invalid_position_action_without_position
                            or invalid_cancel_without_pending
                            or (is_hold and authority_responsibility_active)
                        ),
                        'five_pattern_actor_eligible': bool(policy_action == int(Action.HOLD) or sent),
                        # Backward-compatible alias now means economic credit.
                        'five_actor_eligible': bool(
                            sent or invalid_position_action_without_position
                            or invalid_cancel_without_pending
                            or (is_hold and authority_responsibility_active)
                        ),
                        'five_execution_status': (
                            'HOLD' if is_hold else 'SCHEDULED' if sent
                            else blocked_status
                        ),
                        'five_arbitration_blocked': bool(not is_hold and not sent),
                        'five_opposite_entry_conflict': bool(opposite_blocked is not None),
                        'five_opposite_entry_winner_agent': (
                            -1 if opposite_winner is None else int(opposite_winner)
                        ),
                        'five_opposite_entry_blocked_agent': (
                            -1 if opposite_blocked is None else int(opposite_blocked)
                        ),
                        'five_agent_authority_version': FIVE_AGENT_AUTHORITY_VERSION,
                        'five_agent_authority_role': authority.role,
                        'five_agent_authority_actions': sorted(authority.allowed_actions),
                        'five_agent_authority_responsibility': authority.responsibility,
                        'five_authority_responsibility_active': (
                            authority_responsibility_active
                        ),
                        'five_owned_position_before': owns_position_before,
                        'five_flat_hold_no_reward': bool(
                            is_hold and not authority_responsibility_active
                        ),
                        'five_positions_before': int(positions_before),
                        'five_pending_orders_before': int(pending_orders_before),
                        'five_invalid_position_action_no_position': (
                            invalid_position_action_without_position
                        ),
                        'five_invalid_cancel_no_pending': invalid_cancel_without_pending,
                        'five_policy_violation_penalty_r': float(
                            last_policy_violation_penalty[agent_index]
                        ),
                    })
                    try:
                        annotated = replace(trace, action_details=details)
                    except TypeError:
                        # Lightweight external/test policies may return a
                        # mutable trace object instead of the frozen StepTrace.
                        annotated = copy(trace)
                        setattr(annotated, 'action_details', details)
                    annotated_traces.append(annotated)
                feedback_traces = tuple(annotated_traces)
                if learning_enabled:
                    register_lifecycle = getattr(policy, 'register_five_entry_lifecycle_packet', None)
                    if callable(register_lifecycle):
                        register_lifecycle(feedback_traces)
                if pattern_reward_pending and pattern_reward_oracle is not None:
                    pattern_anchor_traces = feedback_traces
                    pattern_anchor_rewards = pattern_reward_oracle.rewards(
                        [int(dict(getattr(t, 'action_details', {}) or {})['five_action_index']) for t in feedback_traces],
                        [bool(dict(getattr(t, 'action_details', {}) or {}).get('five_pattern_actor_eligible', False)) for t in feedback_traces],
                    )
                    pattern_reward_pending = False


        final_idx = max(start, min(stop, len(self.data) - 1))
        self._close_all(final_idx, 'end')
        final_lifecycle_route_stats = {'routed': 0, 'missed': 0}
        if learning_enabled:
            final_lifecycle_route_stats = self._route_five_lifecycle_shaping(
                policy, self.trades[lifecycle_trade_cursor:], trace_actions,
            )
        self._mark_to_market_equity(final_idx)
        if learning_enabled and pattern_anchor_traces is not None and callable(pattern_hook):
            auxiliary_targets = pattern_reward_oracle.auxiliary_targets()
            try:
                pattern_stats = pattern_hook(
                    pattern_anchor_traces,
                    pattern_anchor_rewards,
                    auxiliary_targets=auxiliary_targets,
                )
            except TypeError as exc:
                if 'auxiliary_targets' not in str(exc):
                    raise
                # Tuong thich voi policy ngoai chi co hook hai doi so. Policy
                # van hoc reward nhung khong nhan supervision cua cay quan sat.
                pattern_stats = pattern_hook(pattern_anchor_traces, pattern_anchor_rewards)
            if telemetry_callback is not None and isinstance(pattern_stats, dict):
                path_diagnostics = pattern_reward_oracle.reward_diagnostics()
                component_means = dict(path_diagnostics['component_mean_r'])
                telemetry_callback({
                    'event': 'pattern_learning',
                    'bar_index': int(final_idx),
                    'episode': int(episode),
                    'mode': str(mode),
                    'pattern_reward_applied': 1.0,
                    'pattern_long_utility_atr': float(pattern_reward_oracle.long_utility_atr),
                    'pattern_short_utility_atr': float(pattern_reward_oracle.short_utility_atr),
                    'pattern_best_utility_atr': float(max(
                        pattern_reward_oracle.long_utility_atr,
                        pattern_reward_oracle.short_utility_atr,
                    )),
                    'pattern_long_mfe_atr': float(pattern_reward_oracle.long_mfe_atr),
                    'pattern_long_mae_atr': float(pattern_reward_oracle.long_mae_atr),
                    'pattern_short_mfe_atr': float(pattern_reward_oracle.short_mfe_atr),
                    'pattern_short_mae_atr': float(pattern_reward_oracle.short_mae_atr),
                    'pattern_no_edge_target': float(auxiliary_targets['no_edge']),
                    'path_reward_best_action_r': float(path_diagnostics['best_action_path_reward_r']),
                    **{
                        f'path_reward_{name}_mean_r': float(value)
                        for name, value in component_means.items()
                    },
                    **{
                        str(k): float(v)
                        for k, v in pattern_stats.items()
                        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                    },
                })
        if self.current_day is not None and self.day_start_equity > 0:
            self.daily_returns.append((self.equity / self.day_start_equity) - 1.0)
            self.daily_trade_counts.append(self.current_day_trade_count)
            self.daily_bar_counts.append(self.current_day_bar_count)
        if telemetry_callback is not None:
            snapshot = self.account_snapshot(final_idx, episode=episode, mode=mode)
            snapshot['event'] = 'episode_end'
            snapshot['five_lifecycle_routed'] = float(final_lifecycle_route_stats['routed'])
            snapshot['five_lifecycle_route_missed'] = float(final_lifecycle_route_stats['missed'])
            telemetry_callback(snapshot)
        return self._result()

    def _roll_day(self, idx: int) -> None:
        day = self.day_values[idx]
        if self.current_day is None:
            self.current_day = day
            self.current_day_trade_count = 0
            self.current_day_bar_count = 1
            self.day_start_equity = self.equity
        elif day != self.current_day:
            self.daily_returns.append((self.equity / self.day_start_equity) - 1.0)
            self.daily_trade_counts.append(self.current_day_trade_count)
            self.daily_bar_counts.append(self.current_day_bar_count)
            self.current_day = day
            self.current_day_trade_count = 0
            self.current_day_bar_count = 1
            self.day_start_equity = self.equity
        else:
            self.current_day_bar_count += 1

    def _apply_decision(self, idx: int, decision: DecodedDecision, source_idx: int | None = None, trace_id: str = "") -> None:
        action = decision.action
        if action == Action.HOLD:
            return
        if action == Action.MANAGE_POSITION:
            # Legacy vectors stay execution-inert. Contract-v5 factorized
            # decisions explicitly select a visible position slot and whether
            # to close part or all of that exposure.
            if not decision.factorized_contract or not self.positions:
                return
            slot = min(max(int(decision.target_slot), 0), len(self.positions) - 1)
            position = self.positions[slot]
            if decision.management_mode == PositionManagementMode.NONE:
                return
            if decision.management_mode == PositionManagementMode.PARTIAL_CLOSE:
                self._partial_close_position(
                    position,
                    idx,
                    float(decision.partial_close_pct),
                    management_trace_id=trace_id,
                )
                return
            if decision.management_mode == PositionManagementMode.FULL_CLOSE:
                self.positions.pop(slot)
                self._set_management_trace(position, idx, trace_id)
                self._close_position(position, self._exit_price(position, idx, "Open"), idx, "manage_full_close")
                return
            return
        if action == Action.CANCEL_ORDERS:
            self.pending_orders.clear()
            return
        if action == Action.CLOSE_ALL:
            self._close_all(idx, "close_all", price_column="Open", management_trace_id=trace_id)
            return
        if action == Action.CLOSE_BEST:
            self._close_ranked(idx, best=True, price_column="Open", management_trace_id=trace_id)
            return
        if action == Action.CLOSE_WORST:
            self._close_ranked(idx, best=False, price_column="Open", management_trace_id=trace_id)
            return
        if action in {Action.MARKET_BUY, Action.MARKET_SELL, Action.BUY_LIMIT, Action.SELL_LIMIT, Action.BUY_STOP, Action.SELL_STOP}:
            self._open_or_place(idx, decision, source_idx if source_idx is not None else idx, trace_id)


    def _partial_close_position(
        self,
        position: Position,
        idx: int,
        fraction: float,
        *,
        management_trace_id: str = "",
    ) -> None:
        """Close only the selected fraction without changing SL/TP of the remainder."""
        if position not in self.positions:
            return
        ratio = float(np.clip(fraction, 0.0, 1.0))
        if ratio <= 0.0:
            return
        if ratio >= 1.0 - 1e-9:
            self.positions.remove(position)
            self._set_management_trace(position, idx, management_trace_id)
            self._close_position(position, self._exit_price(position, idx, "Open"), idx, "manage_partial_100")
            return

        # Attribute only the fraction actually closed to this management
        # decision. The untouched remainder keeps its original causal trace.
        closed = Position(
            side=position.side,
            entry=position.entry,
            sl=position.sl,
            tp=position.tp,
            risk_pct=position.risk_pct * ratio,
            risk_amount=position.risk_amount * ratio,
            risk_distance=position.risk_distance,
            size=position.size * ratio,
            opened_bar=position.opened_bar,
            trailing_stop_atr=position.trailing_stop_atr,
            peak_price=position.peak_price,
            adverse_price=position.adverse_price,
            origin_trace_id=position.origin_trace_id,
            initial_atr=position.initial_atr,
            trailing_applied=position.trailing_applied,
            lifecycle_final_close=False,
        )
        self._set_management_trace(closed, idx, management_trace_id)
        self._close_position(closed, self._exit_price(closed, idx, "Open"), idx, "manage_partial_close")

        remain = 1.0 - ratio
        position.size *= remain
        position.risk_amount *= remain
        position.risk_pct *= remain

    def _position_ages_and_r(self, idx: int) -> tuple[list[float], list[float]]:
        return (
            [float(idx - position.opened_bar) for position in self.positions],
            [self._unrealized_r(position, idx, "Close") for position in self.positions],
        )

    def _allowed_actions(self, idx: int = 0) -> set[Action]:
        ages, rs = self._position_ages_and_r(idx) if self.positions else ([], [])
        actions = allowed_actions_for_state(
            position_count=len(self.positions),
            pending_count=len(self.pending_orders),
            max_positions=self.config.risk.max_positions,
            ages_bars=ages,
            unrealized_rs=rs,
            min_manual_close_age_bars=MIN_MANUAL_CLOSE_AGE_BARS,
            emergency_close_r=EMERGENCY_CLOSE_R,
        )
        # Five authority v1 gives agent 2 the learned target/partial-close head.
        # The compatibility helper does not know whether the active policy uses
        # factorized contract v6, so expose MANAGE only on this Five path when
        # a real position exists.
        if self.positions:
            actions.add(Action.MANAGE_POSITION)
        # Frequency limits are opt-in operational controls. When explicitly
        # enabled, mask entries before the policy selects so simulator and MT5
        # bridge agree on what can execute; otherwise they are neutral.
        max_daily = int(self.config.orders.max_daily_trades)
        if self._entry_limits_enabled() and max_daily > 0 and self.current_day_trade_count >= max_daily:
            actions.difference_update({
                Action.MARKET_BUY, Action.MARKET_SELL,
                Action.BUY_LIMIT, Action.SELL_LIMIT,
                Action.BUY_STOP, Action.SELL_STOP,
            })
        return actions

    def _set_management_trace(self, position: Position, idx: int, trace_id: str) -> None:
        """Attach credit only to an explicit CLOSE_* decision, never SL/TP management."""
        if not trace_id or trace_id == position.origin_trace_id:
            return
        position.management_trace_id = trace_id
        position.management_anchor_r = self._unrealized_r(position, idx, "Open")

    def _entry_limits_enabled(self) -> bool:
        return bool(getattr(self.config.risk, "enforce_entry_limits", False))

    def _has_open_capacity(self) -> bool:
        return bool(
            not position_limit_enabled(self.config.risk.max_positions)
            or len(self.positions) + len(self.pending_orders) < self.config.risk.max_positions
        )

    def _open_or_place(self, idx: int, decision: DecodedDecision, source_idx: int, trace_id: str = "") -> None:
        side = 1 if decision.action in {Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP} else -1
        open_bid = float(self.open_values[idx])
        spread_price = float(self.spread_values[idx])
        reference_close = float(self.close_values[source_idx])
        if decision.action == Action.MARKET_BUY:
            self._try_open(idx, side, open_bid + spread_price, decision.sl_atr, decision.tp_atr, decision.risk_pct, decision.trailing_stop_atr, trace_id=trace_id)
        elif decision.action == Action.MARKET_SELL:
            self._try_open(idx, side, open_bid, decision.sl_atr, decision.tp_atr, decision.risk_pct, decision.trailing_stop_atr, trace_id=trace_id)
        else:
            entry_bid_basis = pending_entry_bid_basis(decision.action, reference_close, decision.entry_offset_atr, float(self.atr_values[source_idx]))
            self._place_pending(idx, decision.action, side, entry_bid_basis, decision, trace_id=trace_id)

    def _place_pending(self, idx: int, action: Action, side: int, entry: float, decision: DecodedDecision, trace_id: str = "") -> None:
        if not self._can_open(decision.risk_pct, decision.sl_atr, decision.tp_atr, idx):
            return
        self.pending_orders.append(PendingOrder(action, side, entry, decision.sl_atr, decision.tp_atr, decision.risk_pct, idx, decision.trailing_stop_atr, trace_id))

    def _trigger_pending(self, idx: int) -> None:
        high_bid = float(self.high_values[idx])
        low_bid = float(self.low_values[idx])
        spread_price = float(self.spread_values[idx])
        active: list[PendingOrder] = []
        for order in self.pending_orders:
            if idx - order.created_bar >= self.config.orders.pending_expiry_bars:
                continue
            # A pending order is not allowed to use the high/low of its own
            # creation bar, which was the former OHLC look-ahead bug.
            if order.created_bar >= idx:
                active.append(order)
                continue
            triggered = False
            entry = order.entry
            if order.action == Action.BUY_LIMIT and low_bid <= order.entry:
                triggered, entry = True, order.entry + spread_price
            elif order.action == Action.BUY_STOP and high_bid >= order.entry:
                triggered, entry = True, order.entry + spread_price
            elif order.action == Action.SELL_LIMIT and high_bid >= order.entry:
                triggered = True
            elif order.action == Action.SELL_STOP and low_bid <= order.entry:
                triggered = True
            if triggered:
                self._try_open(
                    idx, order.side, entry, order.sl_atr, order.tp_atr, order.risk_pct,
                    order.trailing_stop_atr, trace_id=order.origin_trace_id,
                    pending_slots_to_ignore=1,
                    pending_fill_bars=max(0, idx - order.created_bar),
                )
            else:
                active.append(order)
        self.pending_orders = active

    def _apply_slippage(self, side: int, price: float, atr: float) -> float:
        total_slip = self.config.risk.effective_slippage_atr_multiple * atr + self.config.risk.spread_slippage_points * self.config.symbol.point_size
        return float(price) + int(side) * total_slip

    def _try_open(
        self,
        idx: int,
        side: int,
        entry: float,
        sl_atr: float,
        tp_atr: float,
        risk_pct: float,
        trailing_stop_atr: float = 0.0,
        *,
        trace_id: str = "",
        pending_slots_to_ignore: int = 0,
        pending_fill_bars: int | None = None,
    ) -> bool:
        if not self._can_open(risk_pct, sl_atr, tp_atr, idx, pending_slots_to_ignore=pending_slots_to_ignore):
            return False
        atr = float(self.atr_values[idx])
        slipped_entry = self._apply_slippage(side, entry, atr)
        sl, tp, risk_distance = build_sl_tp(side, slipped_entry, atr, sl_atr, tp_atr)
        if self.config.risk.fixed_risk_pct > 0:
            risk_pct = float(self.config.risk.fixed_risk_pct)
        risk_amount = self.equity * risk_pct
        size = risk_amount / max(risk_distance, 1e-12)
        self.positions.append(Position(
            side=side,
            entry=slipped_entry,
            sl=sl,
            tp=tp,
            risk_pct=risk_pct,
            risk_amount=risk_amount,
            risk_distance=risk_distance,
            size=size,
            opened_bar=idx,
            trailing_stop_atr=trailing_stop_atr,
            origin_trace_id=trace_id,
            initial_atr=atr,
            pending_fill_bars=pending_fill_bars,
        ))
        self.risk_samples.append(risk_pct)
        self.current_day_trade_count += 1
        return True

    def _can_open(self, risk_pct: float, sl_atr: float, tp_atr: float, idx: int, *, pending_slots_to_ignore: int = 0) -> bool:
        atr = float(self.atr_values[idx])
        spread_atr = float(self.spread_atr_values[idx])
        rr = tp_atr / max(sl_atr, 1e-12)
        spread_cost_r = float(self.spread_values[idx]) / max(sl_atr * atr, 1e-12)
        occupied = len(self.positions) + max(0, len(self.pending_orders) - int(pending_slots_to_ignore))
        capacity_valid = bool(
            not position_limit_enabled(self.config.risk.max_positions)
            or occupied < self.config.risk.max_positions
        )
        technical_valid = bool(
            capacity_valid
            and np.isfinite(risk_pct) and np.isfinite(sl_atr) and np.isfinite(tp_atr)
            and self.config.risk.min_risk_pct <= risk_pct <= self.config.risk.max_risk_pct
            and atr > 0.0 and np.isfinite(atr) and rr >= self.config.orders.technical_min_rr
        )
        if not technical_valid:
            return False
        if bool(self.enforce_simulated_margin):
            risk_distance = max(float(sl_atr) * atr, 1e-12)
            risk_amount = max(0.0, float(self.equity) * float(risk_pct))
            estimated_size = risk_amount / risk_distance
            mark = max(abs(float(self.close_values[idx])), 1e-12)
            required_margin = estimated_size * mark / max(float(self.simulated_leverage), 1e-9)
            if required_margin > max(0.0, self.free_margin(idx)) + 1e-9:
                return False
        if not self._entry_limits_enabled():
            return True
        max_daily = int(self.config.orders.max_daily_trades)
        daily_ok = max_daily <= 0 or self.current_day_trade_count < max_daily
        return bool(
            daily_ok
            and self.total_open_risk() + risk_pct <= self.config.risk.max_open_risk_pct
            and spread_cost_r < self.config.orders.max_spread_cost_r
            and spread_atr <= self.config.features.max_spread_atr
            and float(self.not_shock_values[idx]) >= 1.0
        )

    def _update_open_positions(self, idx: int) -> None:
        high_bid = float(self.high_values[idx])
        low_bid = float(self.low_values[idx])
        close_bid = float(self.close_values[idx])
        spread_price = float(self.spread_values[idx])
        remaining: list[Position] = []
        for position in self.positions:
            # Avoid using the current bar's extrema against a fill made at this
            # bar's open (or later). That would be a future-information exit.
            if position.opened_bar >= idx:
                remaining.append(position)
                continue
            if position.side == 1:
                position.peak_price = max(position.peak_price, high_bid)
                position.adverse_price = (
                    low_bid if position.adverse_price == 0.0
                    else min(position.adverse_price, low_bid)
                )
            else:
                current_low = low_bid + spread_price
                position.peak_price = current_low if position.peak_price == 0.0 else min(position.peak_price, current_low)
                current_high = high_bid + spread_price
                position.adverse_price = (
                    current_high if position.adverse_price == 0.0
                    else max(position.adverse_price, current_high)
                )

            # Tính unrealized R
            if position.side == 1:
                unrealized_r = (close_bid - position.entry) / max(position.risk_distance, 1e-12)
            else:
                unrealized_r = (position.entry - close_bid - spread_price) / max(position.risk_distance, 1e-12)

            should_close, exit_price, reason = conservative_exit(position, high_bid, low_bid, close_bid, spread_price)
            if not should_close and idx - position.opened_bar >= self.config.orders.max_holding_bars:
                should_close = True
                exit_price = close_bid if position.side == 1 else close_bid + spread_price
                reason = "max_holding"
            if should_close:
                self._close_position(position, exit_price, idx, reason)
            else:
                # The learned trailing choice is evaluated from the completed
                # bar close and becomes active only for the next bar.  This
                # avoids using one bar's high to move a stop against that same
                # bar's low (an impossible intrabar ordering assumption).
                entry_atr = float(position.initial_atr)
                if entry_atr <= 0.0:
                    # Compatibility for manually constructed legacy Position
                    # fixtures that predate the immutable entry ATR field.
                    entry_atr = float(self.atr_values[position.opened_bar])
                if position.trailing_stop_atr > 0.0 and entry_atr > 0.0:
                    if position.side == 1:
                        candidate_sl = close_bid - position.trailing_stop_atr * entry_atr
                        if candidate_sl < close_bid:
                            improved_sl = max(position.sl, candidate_sl)
                            if improved_sl > position.sl + 1e-12:
                                position.trailing_applied = True
                                position.sl = improved_sl
                    else:
                        close_ask = close_bid + spread_price
                        candidate_sl = close_ask + position.trailing_stop_atr * entry_atr
                        if candidate_sl > close_ask:
                            improved_sl = min(position.sl, candidate_sl)
                            if improved_sl < position.sl - 1e-12:
                                position.trailing_applied = True
                                position.sl = improved_sl
                remaining.append(position)
        self.positions = remaining

    def _exit_price(self, position: Position, idx: int, price_column: str) -> float:
        if price_column == "Open":
            bid_price = float(self.open_values[idx])
        elif price_column == "Close":
            bid_price = float(self.close_values[idx])
        else:
            bid_price = float(self.data.iloc[idx][price_column])
        return bid_price if position.side == 1 else bid_price + float(self.spread_values[idx])

    def _unrealized_r(self, position: Position, idx: int, price_column: str = "Close") -> float:
        exit_price = self._exit_price(position, idx, price_column)
        return position.side * (exit_price - position.entry) / max(position.risk_distance, 1e-12)

    def _close_all(self, idx: int, reason: str, price_column: str = "Close", management_trace_id: str = "") -> None:
        if not self.positions:
            return
        positions = list(self.positions)
        self.positions.clear()
        for position in positions:
            self._set_management_trace(position, idx, management_trace_id)
            self._close_position(position, self._exit_price(position, idx, price_column), idx, reason)

    def _close_ranked(self, idx: int, best: bool, price_column: str = "Close", management_trace_id: str = "") -> None:
        if not self.positions:
            return
        ranked = [(self._unrealized_r(position, idx, price_column), position, self._exit_price(position, idx, price_column)) for position in self.positions]
        ranked.sort(key=lambda item: item[0], reverse=best)
        _, position, exit_price = ranked[0]
        self.positions.remove(position)
        self._set_management_trace(position, idx, management_trace_id)
        self._close_position(position, exit_price, idx, "close_best" if best else "close_worst")

    def _close_position(self, position: Position, exit_price: float, idx: int, reason: str) -> None:
        slipped_exit = self._apply_slippage(-position.side, exit_price, float(self.atr_values[idx]))
        pnl = pnl_for_exit(position, slipped_exit)
        self.realized_equity += pnl
        self.equity = self.realized_equity
        trade_r = pnl / max(position.risk_amount, 1e-12)
        entry_credit = trade_r
        management_credit = 0.0
        if position.management_trace_id and position.management_trace_id != position.origin_trace_id:
            entry_credit = float(position.management_anchor_r)
            management_credit = float(trade_r - entry_credit)
        if position.side == 1:
            favorable_price = max(float(position.entry), float(position.peak_price or position.entry))
            mfe_r = (favorable_price - float(position.entry)) / max(position.risk_distance, 1e-12)
            adverse_price = min(float(position.entry), float(position.adverse_price or position.entry))
            mae_r = (adverse_price - float(position.entry)) / max(position.risk_distance, 1e-12)
            locked_profit_r = max(
                0.0, (float(position.sl) - float(position.entry)) / max(position.risk_distance, 1e-12)
            )
        else:
            favorable_price = min(float(position.entry), float(position.peak_price or position.entry))
            mfe_r = (float(position.entry) - favorable_price) / max(position.risk_distance, 1e-12)
            adverse_price = max(float(position.entry), float(position.adverse_price or position.entry))
            mae_r = (float(position.entry) - adverse_price) / max(position.risk_distance, 1e-12)
            locked_profit_r = max(
                0.0, (float(position.entry) - float(position.sl)) / max(position.risk_distance, 1e-12)
            )
        initial_atr = max(float(position.initial_atr), 1e-12)
        sl_atr = float(position.risk_distance) / initial_atr
        tp_atr = abs(float(position.tp) - float(position.entry)) / initial_atr
        if str(reason).lower() == "sl":
            exit_kind = "TRAILING_SL" if position.trailing_applied else "INITIAL_SL"
        elif str(reason).lower() == "tp":
            exit_kind = "TP"
        else:
            exit_kind = str(reason).upper()
        self.trades.append(Trade(
            side=position.side,
            entry=position.entry,
            exit=slipped_exit,
            pnl=pnl,
            r=trade_r,
            opened_bar=position.opened_bar,
            closed_bar=idx,
            reason=reason,
            risk_pct=position.risk_pct,
            origin_trace_id=position.origin_trace_id,
            management_trace_id=position.management_trace_id,
            entry_credit_r=entry_credit,
            management_credit_r=management_credit,
            mfe_r=float(max(0.0, mfe_r)),
            mae_r=float(min(0.0, mae_r)),
            locked_profit_r=float(max(0.0, locked_profit_r)),
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            trailing_stop_atr=float(position.trailing_stop_atr),
            trailing_applied=bool(position.trailing_applied),
            exit_kind=exit_kind,
            lifecycle_final_close=bool(position.lifecycle_final_close),
            pending_fill_bars=position.pending_fill_bars,
        ))

    def _mark_to_market_equity(self, idx: int) -> float:
        unrealized = 0.0
        for position in self.positions:
            unrealized += pnl_for_exit(position, self._exit_price(position, idx, "Close"))
        self.equity = float(self.realized_equity + unrealized)
        return self.equity

    def total_open_risk(self) -> float:
        return float(sum(position.risk_pct for position in self.positions))

    def drawdown_pct(self) -> float:
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity) if self.peak_equity > 0 else 0.0

    def _result(self) -> BacktestResult:
        from .fitness import score_result

        equity_curve = np.asarray(self.equity_curve or [self.equity], dtype=float)
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = np.where(peaks > 0, (peaks - equity_curve) / peaks, 0.0)
        max_drawdown_pct = float(np.max(drawdowns)) if drawdowns.size else 0.0
        trade_rs = np.asarray([trade.r for trade in self.trades], dtype=float)
        total_r = float(np.sum(trade_rs)) if trade_rs.size else 0.0
        expectancy = float(np.mean(trade_rs)) if trade_rs.size else 0.0
        winrate = float(np.mean(trade_rs > 0)) if trade_rs.size else 0.0
        risk_abuse = float(max(0.0, np.mean(self.risk_samples) - 0.01) * 100.0) if self.risk_samples else 0.0
        attributable = sum(1 for trade in self.trades if trade.origin_trace_id)
        result = BacktestResult(
            fitness=0.0, equity=self.equity, equity_curve=list(equity_curve), trades=self.trades,
            daily_returns=self.daily_returns, daily_trade_counts=self.daily_trade_counts,
            daily_bar_counts=self.daily_bar_counts, min_daily_trades=self.config.orders.min_daily_trades,
            max_daily_trades_limit=self.config.orders.max_daily_trades,
            max_drawdown_pct=max_drawdown_pct, max_drawdown_r=max_drawdown_pct * 100.0,
            total_r=total_r, expectancy=expectancy, winrate=winrate, ruin=self.ruin,
            risk_abuse_penalty=risk_abuse,
            causal_credit_coverage=(attributable / len(self.trades)) if self.trades else 1.0,
            unattributed_trade_count=len(self.trades) - attributable,
            recorded_actions=list(self.recorded_actions),
        )
        result.fitness = score_result(result)
        return result
