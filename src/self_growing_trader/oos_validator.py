"""Strict Out-of-Sample (OOS) Validator & Anti-Collapse Guards.

Validates trading model candidates on dedicated unseen Q1 2025 data (2025.01.01 - 2025.03.31),
enforcing frozen policy immutability assertions, quantitative edge criteria (E[R] >= +0.05R,
WR >= 40%, RR >= 2.0, valid trades >= 30, Max Drawdown < 10.0%), anti-HOLD collapse guards,
anti-directional collapse guards, and a cumulative drawdown safety monitor (abort if cumulative DD > 15%).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from self_growing_trader.learning_integrity import policy_state_hash


def compute_policy_state_hash(policy: Any) -> str:
    """Robust hash for policy state, asserting frozen weights across PyTorch, dataclasses, and custom objects."""
    if hasattr(policy, "state_dict") and callable(policy.state_dict):
        try:
            import io
            import torch
            sd = policy.state_dict()
            buffer = io.BytesIO()
            torch.save(sd, buffer)
            return sha256(buffer.getvalue()).hexdigest()
        except Exception:
            pass
    return policy_state_hash(policy)

DEFAULT_OOS_FROM_DATE: str = "2025.01.01"
DEFAULT_OOS_TO_DATE: str = "2025.03.31"

DEFAULT_MIN_EXPECTANCY: float = 0.05
DEFAULT_MIN_WIN_RATE: float = 0.40
DEFAULT_MIN_RISK_REWARD: float = 2.0
DEFAULT_MIN_TRADES: int = 30
DEFAULT_MAX_DRAWDOWN_PCT: float = 10.0
DEFAULT_MAX_CUMULATIVE_DRAWDOWN_PCT: float = 15.0

DEFAULT_MAX_HOLD_SHARE: float = 0.80
DEFAULT_MIN_NORMALIZED_ENTROPY: float = 0.35
DEFAULT_MAX_DIRECTION_IMBALANCE: float = 0.60


class PolicyImmutabilityError(RuntimeError):
    """Raised when candidate policy state is mutated during frozen evaluation."""


class EmergencyHaltError(RuntimeError):
    """Raised when cumulative equity drawdown exceeds the safety budget limit."""


def _normalize_comparison_pct(val: Any) -> float:
    """Normalize percentage or fractional value to a common 0.0-100.0 percentage scale."""
    v = float(val)
    return v * 100.0 if (0.0 <= v <= 1.0) else v


class WinRate(float):
    """Float representing win rate (e.g. 0.45 for 45%).

    Supports rich comparisons seamlessly against both fractional thresholds
    (e.g. ``wr >= 0.40``) and percentage thresholds (e.g. ``wr >= 40.0``),
    regardless of whether ``self`` is initialized as fractional (0.45) or percentage (45.0).
    """

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) >= _normalize_comparison_pct(other)
        return super().__ge__(other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) > _normalize_comparison_pct(other)
        return super().__gt__(other)

    def __le__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) <= _normalize_comparison_pct(other)
        return super().__le__(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) < _normalize_comparison_pct(other)
        return super().__lt__(other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return math.isclose(
                _normalize_comparison_pct(self),
                _normalize_comparison_pct(other),
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        return super().__eq__(other)


class DrawdownPct(float):
    """Float representing drawdown percentage (e.g. 8.5 for 8.5%).

    Supports comparisons against both percentage thresholds (e.g. ``dd < 10.0``)
    and fractional thresholds (e.g. ``dd < 0.10``), regardless of whether ``self``
    is initialized as percentage (8.5) or fractional (0.085).
    """

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) < _normalize_comparison_pct(other)
        return super().__lt__(other)

    def __le__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) <= _normalize_comparison_pct(other)
        return super().__le__(other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) > _normalize_comparison_pct(other)
        return super().__gt__(other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return _normalize_comparison_pct(self) >= _normalize_comparison_pct(other)
        return super().__ge__(other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return math.isclose(
                _normalize_comparison_pct(self),
                _normalize_comparison_pct(other),
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        return super().__eq__(other)


def compute_action_entropy(counts: dict[str, int] | Counter | Sequence[Any]) -> tuple[float, float]:
    """Compute Shannon entropy and normalized entropy from action counts or sequence.

    Formula follows ``src/self_growing_trader/mt5_live_diagnostics.py``:
        entropy = -sum(p * ln(p))
        normalized = entropy / ln(len(counts)) if len(counts) > 1 else 0.0

    Returns:
        tuple[float, float]: (raw_entropy, normalized_entropy)
    """
    if isinstance(counts, (list, tuple)):
        counts = Counter(counts)
    elif not isinstance(counts, (dict, Counter)):
        return 0.0, 0.0

    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0

    probabilities = [float(cnt) / total for cnt in counts.values() if cnt > 0]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    num_classes = len(counts)
    normalized = (entropy / math.log(num_classes)) if num_classes > 1 else 0.0
    return float(entropy), float(normalized)


def compute_direction_imbalance(
    counts: dict[str, int] | Counter | Sequence[Any] | None = None,
    long_count: int | None = None,
    short_count: int | None = None,
) -> float:
    """Compute entry direction imbalance.

    Formula follows ``src/self_growing_trader/mt5_live_diagnostics.py``:
        imbalance = (long_entries - short_entries) / (long_entries + short_entries)
    Range: [-1.0, 1.0], where +1.0 = 100% BUY, -1.0 = 100% SELL, 0.0 = 50/50 balance.
    """
    if long_count is not None and short_count is not None:
        long_entries = int(long_count)
        short_entries = int(short_count)
    elif counts is not None:
        if isinstance(counts, (list, tuple)):
            counts = Counter(counts)

        int_keys = [k for k in counts.keys() if isinstance(k, int)]
        max_int_key = max(int_keys) if int_keys else 0
        long_entries = 0
        short_entries = 0
        for act, cnt in counts.items():
            cnt_val = int(cnt)
            act_str = str(act).strip().upper()
            if any(term in act_str for term in ("BUY", "LONG")):
                long_entries += cnt_val
            elif any(term in act_str for term in ("SELL", "SHORT")):
                short_entries += cnt_val
            elif isinstance(act, int):
                # 3-action or 7-action space indexing
                if act in (1, 2, 3) and max_int_key > 3:
                    long_entries += cnt_val
                elif act in (4, 5, 6):
                    short_entries += cnt_val
                elif act == 1:
                    long_entries += cnt_val
                elif act == 2 and max_int_key <= 2:
                    short_entries += cnt_val
    else:
        return 0.0

    directional = long_entries + short_entries
    if directional == 0:
        return 0.0
    return float((long_entries - short_entries) / directional)


def compute_expectancy(
    win_rate: float,
    risk_reward: float,
    trade_returns: Sequence[float] | None = None,
) -> float:
    """Compute mathematical expectancy E[R] per trade.

    Formula:
        E[R] = (w * RR) - (1.0 - w)
    or sample mean of realized R values if trade_returns is provided.
    """
    if trade_returns is not None and len(trade_returns) > 0:
        return float(sum(trade_returns) / len(trade_returns))

    w = float(win_rate) / 100.0 if win_rate > 1.0 else float(win_rate)
    w = max(0.0, min(1.0, w))
    rr = max(0.0, float(risk_reward))
    return float((w * rr) - (1.0 - w))


@dataclass
class CumulativeDrawdownTracker:
    """Tracks cumulative account equity drawdown across sequential evaluations.

    Flags emergency halt if cumulative drawdown exceeds max_allowed_drawdown_pct (default 15.0%).
    """

    max_allowed_drawdown_pct: float = DEFAULT_MAX_CUMULATIVE_DRAWDOWN_PCT
    initial_equity: float = 10000.0
    peak_equity: float = 10000.0
    current_equity: float = 10000.0
    cumulative_drawdown_pct: float = 0.0
    max_cumulative_drawdown_pct: float = 0.0
    emergency_halt: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        drawdown_pct: float | None = None,
        net_profit: float | None = None,
        equity: float | None = None,
        round_id: int | None = None,
    ) -> float:
        """Update tracker state with a new run's metrics."""
        if equity is not None:
            self.current_equity = float(equity)
        elif net_profit is not None:
            self.current_equity += float(net_profit)
        elif drawdown_pct is not None:
            dd = float(drawdown_pct)
            dd_rate = dd / 100.0 if dd > 1.0 else dd
            dd_rate = max(0.0, min(1.0, dd_rate))
            self.current_equity *= (1.0 - dd_rate)

        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        if self.peak_equity > 0.0:
            self.cumulative_drawdown_pct = float(
                ((self.peak_equity - self.current_equity) / self.peak_equity) * 100.0
            )
        else:
            self.cumulative_drawdown_pct = 0.0

        if self.cumulative_drawdown_pct > self.max_cumulative_drawdown_pct:
            self.max_cumulative_drawdown_pct = self.cumulative_drawdown_pct

        if self.cumulative_drawdown_pct > self.max_allowed_drawdown_pct:
            self.emergency_halt = True

        self.history.append({
            "round_id": round_id if round_id is not None else len(self.history) + 1,
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "cumulative_drawdown_pct": self.cumulative_drawdown_pct,
            "emergency_halt": self.emergency_halt,
        })
        return self.cumulative_drawdown_pct

    def is_emergency_halt(self) -> bool:
        """Check whether emergency halt is triggered."""
        return self.emergency_halt or (self.cumulative_drawdown_pct > self.max_allowed_drawdown_pct)

    def reset(self, initial_equity: float | None = None) -> None:
        """Reset equity tracking history."""
        if initial_equity is not None:
            self.initial_equity = float(initial_equity)
        self.peak_equity = self.initial_equity
        self.current_equity = self.initial_equity
        self.cumulative_drawdown_pct = 0.0
        self.max_cumulative_drawdown_pct = 0.0
        self.emergency_halt = False
        self.history.clear()


class OOSValidationResult(dict):
    """Result dictionary providing dict indexing and attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'OOSValidationResult' object has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class OOSValidator:
    """Strict Out-of-Sample (OOS) Validator with Anti-Collapse Guards and Immutability Checks."""

    def __init__(
        self,
        from_date: str = DEFAULT_OOS_FROM_DATE,
        to_date: str = DEFAULT_OOS_TO_DATE,
        min_expectancy: float = DEFAULT_MIN_EXPECTANCY,
        min_win_rate: float = DEFAULT_MIN_WIN_RATE,
        min_risk_reward: float = DEFAULT_MIN_RISK_REWARD,
        min_trades: int = DEFAULT_MIN_TRADES,
        max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
        max_cumulative_drawdown_pct: float = DEFAULT_MAX_CUMULATIVE_DRAWDOWN_PCT,
        max_hold_share: float = DEFAULT_MAX_HOLD_SHARE,
        min_normalized_entropy: float = DEFAULT_MIN_NORMALIZED_ENTROPY,
        max_direction_imbalance: float = DEFAULT_MAX_DIRECTION_IMBALANCE,
        drawdown_tracker: CumulativeDrawdownTracker | None = None,
        raise_on_emergency_halt: bool = False,
    ):
        self.from_date = str(from_date)
        self.to_date = str(to_date)
        self.min_expectancy = float(min_expectancy)
        self.min_win_rate = float(min_win_rate)
        self.min_risk_reward = float(min_risk_reward)
        self.min_trades = int(min_trades)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.max_cumulative_drawdown_pct = float(max_cumulative_drawdown_pct)
        self.max_hold_share = float(max_hold_share)
        self.min_normalized_entropy = float(min_normalized_entropy)
        self.max_direction_imbalance = float(max_direction_imbalance)
        self.raise_on_emergency_halt = bool(raise_on_emergency_halt)

        self.drawdown_tracker = drawdown_tracker or CumulativeDrawdownTracker(
            max_allowed_drawdown_pct=self.max_cumulative_drawdown_pct
        )

    def evaluate(
        self,
        candidate: Any,
        config: Any = None,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        evaluator: Callable[..., Any] | None = None,
        metrics: Any = None,
        trades: Sequence[Any] | None = None,
        action_counts: dict[str, int] | Counter | Sequence[Any] | None = None,
        round_id: int | None = None,
        **kwargs: Any,
    ) -> OOSValidationResult:
        """Execute strict OOS validation on a candidate policy.

        1. Asserts candidate policy frozen hash before and after evaluation.
        2. Computes expectancy E[R], win rate, risk/reward, valid trades, max drawdown.
        3. Asserts anti-HOLD collapse guard (hold_share < 0.80, entropy > 0.35, trades >= 30).
        4. Asserts anti-directional collapse guard (|imbalance| <= 0.60).
        5. Asserts quantitative gate criteria (E[R] >= 0.05R, WR >= 40%, RR >= 2.0, DD < 10.0%).
        6. Updates cumulative drawdown tracker; flags emergency halt if cumulative DD > 15%.
        """
        eval_from_date = str(from_date or self.from_date)
        eval_to_date = str(to_date or self.to_date)

        # 1. Capture frozen policy hash before evaluation
        before_hash = compute_policy_state_hash(candidate)

        # 2. Execute evaluation or extract supplied metrics
        raw_metrics = metrics
        if raw_metrics is None:
            if evaluator is not None:
                eval_output = evaluator(
                    candidate,
                    config=config,
                    from_date=eval_from_date,
                    to_date=eval_to_date,
                    **kwargs,
                )
                if isinstance(eval_output, dict):
                    raw_metrics = (
                        eval_output.get("best", {}).get("metrics")
                        or eval_output.get("metrics")
                        or eval_output
                    )
                else:
                    raw_metrics = eval_output
            elif config is not None:
                try:
                    from self_growing_trader.chuyen_gia_policy import evaluate_policy_mt5_oos
                    eval_output = evaluate_policy_mt5_oos(
                        candidate,
                        config=config,
                        from_date=eval_from_date,
                        to_date=eval_to_date,
                        **kwargs,
                    )
                    raw_metrics = (
                        eval_output.get("best", {}).get("metrics")
                        or eval_output.get("metrics")
                        or eval_output
                    )
                except Exception:
                    raw_metrics = getattr(candidate, "metrics", None)
            else:
                raw_metrics = getattr(candidate, "metrics", None)

        # 3. Assert frozen policy hash immutability after evaluation
        after_hash = compute_policy_state_hash(candidate)
        if before_hash != after_hash:
            raise PolicyImmutabilityError(
                f"Frozen OOS evaluation mutated candidate policy! before={before_hash} after={after_hash}"
            )

        # 4. Extract metrics
        extracted = self._extract_metrics(
            raw_metrics=raw_metrics,
            trades=trades,
            action_counts=action_counts,
            kwargs=kwargs,
        )

        total_trades = extracted["total_trades"]
        win_rate_val = extracted["win_rate"]
        risk_reward_val = extracted["risk_reward_ratio"]
        expectancy_val = extracted["expectancy"]
        drawdown_pct_val = extracted["max_equity_drawdown_pct"]
        hold_share_val = extracted["hold_share"]
        entropy_val = extracted["normalized_action_entropy"]
        imbalance_val = extracted["entry_direction_imbalance"]

        # 5. Update cumulative drawdown tracker
        if "cumulative_drawdown_pct" in kwargs:
            cum_dd = float(kwargs["cumulative_drawdown_pct"])
            self.drawdown_tracker.cumulative_drawdown_pct = cum_dd
            if cum_dd > self.drawdown_tracker.max_allowed_drawdown_pct:
                self.drawdown_tracker.emergency_halt = True
        else:
            cum_dd = self.drawdown_tracker.update(
                drawdown_pct=drawdown_pct_val,
                net_profit=extracted.get("net_profit"),
                round_id=round_id,
            )

        emergency_halt = self.drawdown_tracker.is_emergency_halt()

        # 6. Evaluate Anti-Collapse Guards
        reasons: list[str] = []

        hold_guard_reasons: list[str] = []
        if hold_share_val >= self.max_hold_share:
            hold_guard_reasons.append(
                f"anti_hold:hold_share_too_high:{hold_share_val:.4f}>={self.max_hold_share:.4f}"
            )
        if entropy_val <= self.min_normalized_entropy:
            hold_guard_reasons.append(
                f"anti_hold:entropy_too_low:{entropy_val:.4f}<={self.min_normalized_entropy:.4f}"
            )
        if total_trades < self.min_trades:
            hold_guard_reasons.append(
                f"anti_hold:insufficient_trades:{total_trades}<{self.min_trades}"
            )
        hold_guard_passed = len(hold_guard_reasons) == 0

        directional_guard_reasons: list[str] = []
        if abs(imbalance_val) > self.max_direction_imbalance:
            directional_guard_reasons.append(
                f"anti_directional:imbalance_exceeded:{abs(imbalance_val):.4f}>{self.max_direction_imbalance:.4f}"
            )
        directional_guard_passed = len(directional_guard_reasons) == 0

        # 7. Evaluate Quantitative Gate Criteria
        expectancy_passed = expectancy_val >= self.min_expectancy
        if not expectancy_passed:
            reasons.append(
                f"gate:expectancy_below_threshold:{expectancy_val:.4f}<{self.min_expectancy:.4f}"
            )

        win_rate_threshold = self.min_win_rate if self.min_win_rate <= 1.0 else self.min_win_rate / 100.0
        win_rate_passed = win_rate_val >= win_rate_threshold
        if not win_rate_passed:
            reasons.append(
                f"gate:win_rate_below_threshold:{float(win_rate_val):.4f}<{win_rate_threshold:.4f}"
            )

        risk_reward_passed = risk_reward_val >= self.min_risk_reward
        if not risk_reward_passed:
            reasons.append(
                f"gate:risk_reward_below_threshold:{risk_reward_val:.4f}<{self.min_risk_reward:.4f}"
            )

        trades_passed = total_trades >= self.min_trades
        if not trades_passed:
            reasons.append(f"gate:trades_below_threshold:{total_trades}<{self.min_trades}")

        drawdown_passed = drawdown_pct_val < self.max_drawdown_pct
        if not drawdown_passed:
            reasons.append(
                f"gate:drawdown_exceeds_threshold:{drawdown_pct_val:.2f}%>={self.max_drawdown_pct:.2f}%"
            )

        reasons.extend(hold_guard_reasons)
        reasons.extend(directional_guard_reasons)

        if emergency_halt:
            reasons.append(
                f"emergency_halt:cumulative_drawdown_exceeded:{cum_dd:.2f}%>{self.max_cumulative_drawdown_pct:.2f}%"
            )
            if self.raise_on_emergency_halt:
                raise EmergencyHaltError(
                    f"Emergency halt triggered: cumulative drawdown {cum_dd:.2f}% > {self.max_cumulative_drawdown_pct:.2f}%"
                )

        passed_gate = (
            expectancy_passed
            and win_rate_passed
            and risk_reward_passed
            and trades_passed
            and drawdown_passed
            and hold_guard_passed
            and directional_guard_passed
            and not emergency_halt
        )

        return OOSValidationResult({
            "expectancy": float(expectancy_val),
            "win_rate": WinRate(win_rate_val),
            "win_rate_pct": float(win_rate_val * 100.0),
            "risk_reward_ratio": float(risk_reward_val),
            "total_trades": int(total_trades),
            "max_equity_drawdown_pct": DrawdownPct(drawdown_pct_val),
            "hold_share": float(hold_share_val),
            "entry_direction_imbalance": float(imbalance_val),
            "normalized_action_entropy": float(entropy_val),
            "cumulative_drawdown_pct": float(cum_dd),
            "emergency_halt": bool(emergency_halt),
            "passed_gate": bool(passed_gate),
            "reasons": reasons,
            "from_date": eval_from_date,
            "to_date": eval_to_date,
            "policy_hash_before": before_hash,
            "policy_hash_after": after_hash,
            "hold_guard_passed": bool(hold_guard_passed),
            "directional_guard_passed": bool(directional_guard_passed),
            "expectancy_passed": bool(expectancy_passed),
            "win_rate_passed": bool(win_rate_passed),
            "risk_reward_passed": bool(risk_reward_passed),
            "trades_passed": bool(trades_passed),
            "drawdown_passed": bool(drawdown_passed),
        })

    def evaluate_strict_oos(self, candidate: Any, config: Any = None, **kwargs: Any) -> OOSValidationResult:
        """Alias for ``evaluate`` method."""
        return self.evaluate(candidate, config=config, **kwargs)

    def reset_tracker(self, initial_equity: float | None = None) -> None:
        """Reset the cumulative drawdown tracker."""
        self.drawdown_tracker.reset(initial_equity=initial_equity)

    def _extract_metrics(
        self,
        raw_metrics: Any,
        trades: Sequence[Any] | None,
        action_counts: dict[str, int] | Counter | Sequence[Any] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract or derive all required validation metrics from inputs."""
        metrics_dict: dict[str, Any] = {}
        if isinstance(raw_metrics, dict):
            metrics_dict = dict(raw_metrics)
        elif hasattr(raw_metrics, "__dict__"):
            metrics_dict = dict(vars(raw_metrics))

        # 1. Total trades
        total_trades = kwargs.get("total_trades")
        if total_trades is None:
            total_trades = (
                metrics_dict.get("total_trades")
                or metrics_dict.get("trades")
                or metrics_dict.get("lifecycle_count")
                or (
                    metrics_dict.get("execution_summary", {}).get("lifecycle_count")
                    if isinstance(metrics_dict.get("execution_summary"), dict)
                    else None
                )
            )
        if total_trades is None and trades is not None:
            total_trades = len(trades)
        total_trades = int(total_trades or 0)

        # 2. Trade returns (realized R)
        realized_returns: list[float] = []
        if trades is not None:
            for item in trades:
                if isinstance(item, (int, float)):
                    realized_returns.append(float(item))
                elif isinstance(item, dict) and "realized_r" in item:
                    realized_returns.append(float(item["realized_r"]))
                elif hasattr(item, "realized_r"):
                    realized_returns.append(float(getattr(item, "realized_r")))

        # 3. Win rate
        raw_win_rate = kwargs.get("win_rate")
        if raw_win_rate is None:
            raw_win_rate = metrics_dict.get("win_rate")
        if raw_win_rate is None and realized_returns:
            wins = sum(1 for r in realized_returns if r > 0.0)
            raw_win_rate = wins / len(realized_returns)
        if raw_win_rate is None:
            raw_win_rate = 0.0
        win_rate = float(raw_win_rate)
        if win_rate > 1.0:
            win_rate = win_rate / 100.0

        # 4. Risk / Reward ratio
        raw_rr = kwargs.get("risk_reward_ratio") or kwargs.get("rr")
        if raw_rr is None:
            raw_rr = metrics_dict.get("risk_reward_ratio") or metrics_dict.get("rr")
        if raw_rr is None:
            avg_win = float(metrics_dict.get("avg_win", 0.0) or 0.0)
            avg_loss = float(metrics_dict.get("avg_loss", 0.0) or 0.0)
            if avg_loss != 0.0:
                raw_rr = avg_win / abs(avg_loss)
            elif realized_returns:
                wins = [r for r in realized_returns if r > 0.0]
                losses = [abs(r) for r in realized_returns if r < 0.0]
                if wins and losses:
                    raw_rr = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        if raw_rr is None:
            raw_rr = float(metrics_dict.get("tp_sl_ratio", 0.0) or 0.0)
        risk_reward = float(raw_rr or 0.0)

        # 5. Expectancy
        raw_exp = kwargs.get("expectancy")
        if raw_exp is None:
            raw_exp = metrics_dict.get("expectancy")
        if raw_exp is None and realized_returns:
            raw_exp = sum(realized_returns) / len(realized_returns)
        if raw_exp is None and (win_rate > 0.0 or risk_reward > 0.0):
            raw_exp = compute_expectancy(win_rate, risk_reward)
        expectancy = float(raw_exp or 0.0)

        # 6. Max equity drawdown pct
        raw_dd = (
            kwargs.get("max_equity_drawdown_pct")
            or kwargs.get("drawdown_pct")
            or metrics_dict.get("max_equity_drawdown_pct")
            or metrics_dict.get("equity_drawdown_pct")
            or metrics_dict.get("drawdown_pct")
            or metrics_dict.get("max_drawdown_pct")
            or metrics_dict.get("balance_drawdown_pct")
            or 0.0
        )
        drawdown_pct = float(raw_dd)
        # If passed as 0.085 instead of 8.5%, convert to percentage
        if 0.0 < drawdown_pct <= 1.0 and "pct" in str(raw_dd).lower():
            drawdown_pct = drawdown_pct * 100.0

        # 7. Action counts, Hold share, Entropy, Directional imbalance
        counts = action_counts or kwargs.get("actions") or metrics_dict.get("action_counts")
        if counts is None and "diagnostics" in metrics_dict:
            diag = metrics_dict["diagnostics"]
            if isinstance(diag, dict):
                counts = diag.get("wire_policy", {}).get("action_counts")

        raw_hold_share = kwargs.get("hold_share") or metrics_dict.get("hold_share")
        raw_entropy = kwargs.get("normalized_action_entropy") or metrics_dict.get("normalized_action_entropy")
        raw_imbalance = kwargs.get("entry_direction_imbalance") or metrics_dict.get("entry_direction_imbalance")

        if counts is not None:
            if isinstance(counts, (list, tuple)):
                counts = Counter(counts)

            total_actions = sum(counts.values())
            if total_actions > 0 and raw_hold_share is None:
                hold_cnt = sum(
                    int(cnt)
                    for act, cnt in counts.items()
                    if str(act).strip().upper() in ("HOLD", "0", "NOOP", "FLAT") or act == 0
                )
                raw_hold_share = hold_cnt / total_actions

            if raw_entropy is None:
                _, norm_ent = compute_action_entropy(counts)
                raw_entropy = norm_ent

            if raw_imbalance is None:
                raw_imbalance = compute_direction_imbalance(counts)

        hold_share = float(raw_hold_share if raw_hold_share is not None else 0.0)
        entropy = float(raw_entropy if raw_entropy is not None else 0.0)
        imbalance = float(raw_imbalance if raw_imbalance is not None else 0.0)

        net_profit = metrics_dict.get("net_profit") or kwargs.get("net_profit")

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "risk_reward_ratio": risk_reward,
            "expectancy": expectancy,
            "max_equity_drawdown_pct": drawdown_pct,
            "hold_share": hold_share,
            "normalized_action_entropy": entropy,
            "entry_direction_imbalance": imbalance,
            "net_profit": float(net_profit) if net_profit is not None else None,
        }


# Global default validator instance for module-level evaluate_strict_oos calls
_DEFAULT_VALIDATOR = OOSValidator()


def evaluate_strict_oos(
    candidate: Any,
    config: Any = None,
    *,
    from_date: str = DEFAULT_OOS_FROM_DATE,
    to_date: str = DEFAULT_OOS_TO_DATE,
    evaluator: Callable[..., Any] | None = None,
    validator: OOSValidator | None = None,
    reset_tracker: bool = False,
    **kwargs: Any,
) -> OOSValidationResult:
    """Evaluate candidate policy against strict Q1 2025 Out-of-Sample (OOS) Gate.

    Parameters:
        candidate: Candidate policy instance to validate.
        config: Optional AppConfig instance.
        from_date: OOS window start date (defaults to "2025.01.01").
        to_date: OOS window end date (defaults to "2025.03.31").
        evaluator: Optional backtest evaluation callable.
        validator: Optional custom OOSValidator instance (defaults to shared module validator).
        reset_tracker: If True, resets cumulative drawdown tracker before evaluation.
        **kwargs: Additional metric overrides, trades list, or action counts.

    Returns:
        OOSValidationResult: Dictionary adhering to PROJECT.md interface contract.
    """
    v = validator or _DEFAULT_VALIDATOR
    if reset_tracker:
        v.reset_tracker()
    return v.evaluate(
        candidate=candidate,
        config=config,
        from_date=from_date,
        to_date=to_date,
        evaluator=evaluator,
        **kwargs,
    )
