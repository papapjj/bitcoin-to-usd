from __future__ import annotations

import numpy as np


def daily_trade_metrics(result) -> dict[str, float]:
    raw_counts = getattr(result, "daily_trade_counts", [])
    counts = np.asarray(raw_counts if raw_counts is not None else [], dtype=float)
    min_daily = float(getattr(result, "min_daily_trades", 0))
    max_daily = float(getattr(result, "max_daily_trades_limit", 0))
    raw_bar_counts = getattr(result, "daily_bar_counts", None)
    if not counts.size:
        return {
            "avg_daily_trades": 0.0,
            "max_daily_trades": 0.0,
            "min_daily_trades": min_daily,
            "max_daily_trades_limit": max_daily,
            "daily_trade_under_rate": 1.0 if min_daily > 0 else 0.0,
            "daily_trade_avg_shortfall": min_daily,
            "daily_trade_completion_rate": 0.0 if min_daily > 0 else 1.0,
            "daily_trade_day_hit_rate": 0.0 if min_daily > 0 else 1.0,
        }
    if raw_bar_counts is not None:
        bar_counts = np.asarray(raw_bar_counts, dtype=float)
        if bar_counts.size == counts.size and min_daily > 0:
            expected_counts = np.ceil(min_daily * np.clip(bar_counts, 1.0, 1440.0) / 1440.0)
            expected_counts = np.clip(expected_counts, 1.0, min_daily)
        else:
            expected_counts = np.full_like(counts, min_daily, dtype=float)
    else:
        expected_counts = np.full_like(counts, min_daily, dtype=float)
    shortfalls = np.maximum(0.0, expected_counts - counts)
    if min_daily > 0:
        completion_rate = float(np.mean(np.minimum(counts / np.maximum(expected_counts, 1.0), 1.0)))
        day_hit_rate = float(np.mean(counts >= expected_counts))
    else:
        completion_rate = 1.0
        day_hit_rate = 1.0
    return {
        "avg_daily_trades": float(np.mean(counts)),
        "max_daily_trades": float(np.max(counts)),
        "min_daily_trades": min_daily,
        "max_daily_trades_limit": max_daily,
        "daily_trade_under_rate": float(np.mean(counts < expected_counts)) if min_daily > 0 else 0.0,
        "daily_trade_avg_shortfall": float(np.mean(shortfalls)) if min_daily > 0 else 0.0,
        "daily_trade_completion_rate": completion_rate,
        "daily_trade_day_hit_rate": day_hit_rate,
    }


def goal_progress_metrics(result) -> dict[str, float | bool]:
    daily = np.asarray(result.daily_returns, dtype=float)
    trade_metrics = daily_trade_metrics(result)
    target_day_rate = float(np.mean(daily >= 0.05)) if daily.size else 0.0
    avg_daily_return = float(np.mean(daily)) if daily.size else 0.0
    max_daily_return = float(np.max(daily)) if daily.size else 0.0
    daily_stability = float(np.std(daily)) if daily.size else 0.0
    days_above_3pct = float(np.mean(daily >= 0.03)) if daily.size else 0.0
    days_above_5pct = float(np.mean(daily >= 0.05)) if daily.size else 0.0
    days_above_8pct = float(np.mean(daily >= 0.08)) if daily.size else 0.0
    dd_penalty = 0.0
    if result.max_drawdown_pct > 0.08:
        dd_penalty = (result.max_drawdown_pct - 0.08) * 500.0
    elif result.max_drawdown_pct > 0.05:
        dd_penalty = (result.max_drawdown_pct - 0.05) * 100.0
    stable_goal_score = float(
        150.0 * target_day_rate
        + 80.0 * days_above_5pct
        + 40.0 * days_above_8pct
        + 200.0 * max(0.0, avg_daily_return - 0.05)
        + 30.0 * np.tanh(avg_daily_return / 0.05)
        + result.total_r
        + 8.0 * result.expectancy
        + 50.0 * trade_metrics["daily_trade_completion_rate"]
        + 60.0 * trade_metrics["daily_trade_day_hit_rate"]
        - 200.0 * result.max_drawdown_pct
        - dd_penalty
        - 5.0 * daily_stability
        - 8.0 * trade_metrics["daily_trade_avg_shortfall"]
        - 50.0 * trade_metrics["daily_trade_under_rate"]
        - result.risk_abuse_penalty
        - (100.0 if result.ruin else 0.0)
    )
    return {
        "target_day_rate": target_day_rate,
        "avg_daily_return": avg_daily_return,
        "max_daily_return": max_daily_return,
        "daily_stability": daily_stability,
        "stable_goal_score": stable_goal_score,
        "min_daily_trades": trade_metrics["min_daily_trades"],
        "max_daily_trades": trade_metrics["max_daily_trades"],
        "avg_daily_trades": trade_metrics["avg_daily_trades"],
        "daily_trade_under_rate": trade_metrics["daily_trade_under_rate"],
        "daily_trade_avg_shortfall": trade_metrics["daily_trade_avg_shortfall"],
        "daily_trade_completion_rate": trade_metrics["daily_trade_completion_rate"],
        "daily_trade_day_hit_rate": trade_metrics["daily_trade_day_hit_rate"],
        "days_above_3pct": days_above_3pct,
        "days_above_5pct": days_above_5pct,
        "days_above_8pct": days_above_8pct,
        "goal_hit": bool(
            target_day_rate >= 0.5
            and avg_daily_return > 0.05
            and result.total_r > 0.0
            and result.max_drawdown_pct <= 0.08
            and trade_metrics["daily_trade_day_hit_rate"] >= 0.5
            and not result.ruin
        ),
        "strict_goal_hit": bool(
            target_day_rate >= 0.5
            and avg_daily_return > 0.05
            and result.total_r > 0.0
            and result.max_drawdown_pct <= 0.05
            and trade_metrics["daily_trade_day_hit_rate"] >= 0.5
            and not result.ruin
        ),
    }


def edge_confidence_lcb(result, z: float = 1.96) -> float:
    """Lower confidence bound of realised R, used for selection not cadence."""
    raw_trades = list(getattr(result, "trades", []) or [])
    values = np.asarray([
        float(getattr(trade, "r", 0.0) or 0.0)
        for trade in raw_trades
    ], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        # Legacy/test summaries may expose only total_r. Treat that as one
        # conservative observation rather than crashing selection.
        total_r = float(getattr(result, "total_r", 0.0) or 0.0)
        return total_r - 1.0
    mean = float(values.mean())
    if values.size == 1:
        return mean - 1.0
    return float(mean - float(z) * values.std(ddof=1) / np.sqrt(values.size))


def tight_risk_score(result) -> float:
    """Edge-first robust score with no return target or trade-frequency quota.

    A sample floor remains evidence discipline: it prevents a one-trade lucky
    result from becoming a champion, but never demands that the policy trade.
    """
    trades = len(getattr(result, "trades", []))
    if trades == 0:
        return -500.0
    daily = np.asarray(getattr(result, "daily_returns", []), dtype=float)
    lcb = edge_confidence_lcb(result)
    avg_daily = float(np.mean(daily)) if daily.size else 0.0
    downside = float(np.mean(np.minimum(daily, 0.0))) if daily.size else 0.0
    stability = float(np.std(daily)) if daily.size else 0.0
    dd = float(getattr(result, "max_drawdown_pct", 0.0))
    evidence = min(1.0, trades / 12.0)
    return float(
        120.0 * lcb
        + 25.0 * float(getattr(result, "expectancy", 0.0))
        + 8.0 * float(getattr(result, "total_r", 0.0)) * evidence
        + 30.0 * np.tanh(avg_daily / 0.02)
        - 450.0 * dd
        - 80.0 * abs(downside)
        - 12.0 * stability
        - float(getattr(result, "risk_abuse_penalty", 0.0))
        - (500.0 if bool(getattr(result, "ruin", False)) else 0.0)
        - max(0, 5 - trades) * 20.0
    )


def policy_complexity_penalty(policy) -> float:
    modules = getattr(policy, "modules", [])
    hidden_units = sum(int(getattr(module, "hidden_size", 0)) for module in modules)
    return float(1.5 * max(0, len(modules) - 1) + 0.05 * hidden_units)


def compute_l_reg(policy) -> float:
    total = 0.0
    for module in getattr(policy, "modules", []):
        w_in = getattr(module, "w_in", None)
        if w_in is not None:
            total += float(np.sum(w_in * w_in))
        w_rec = getattr(module, "w_rec", None)
        if w_rec is not None:
            total += float(np.sum(w_rec * w_rec))
        w_out = getattr(module, "w_out", None)
        if w_out is not None:
            total += float(np.sum(w_out * w_out))
    return total


def compute_l_sparse(policy) -> float:
    gates = np.asarray([module.gate_ema for module in getattr(policy, "modules", [])], dtype=float)
    if gates.size == 0:
        return 0.0
    gate_mean = float(np.mean(gates))
    if gate_mean <= 0.0 or gate_mean >= 1.0:
        return 0.0
    return -(gate_mean * np.log(gate_mean + 1e-10) + (1.0 - gate_mean) * np.log(1.0 - gate_mean + 1e-10))


def compute_l_homeo(policy, a_target: float = 0.35) -> float:
    total = 0.0
    for module in getattr(policy, "modules", []):
        act_ema = getattr(module, "activation_ema", 0.0)
        total += (float(act_ema) - a_target) ** 2
    return total


def compute_l_plastic(policy) -> float:
    total = 0.0
    for module in getattr(policy, "modules", []):
        w_in = getattr(module, "w_in", None)
        if w_in is not None:
            total += float(np.sum(np.abs(w_in)))
        w_rec = getattr(module, "w_rec", None)
        if w_rec is not None:
            total += float(np.sum(np.abs(w_rec)))
    return total


def compute_policy_total_loss(
    policy,
    l_task: float,
    lambda_reg: float = 0.0001,
    lambda_sparse: float = 0.001,
    lambda_homeo: float = 0.01,
    lambda_plastic: float = 0.001,
    a_target: float = 0.35,
) -> tuple[float, dict[str, float]]:
    l_reg = compute_l_reg(policy)
    l_sparse = compute_l_sparse(policy)
    l_homeo = compute_l_homeo(policy, a_target)
    l_plastic = compute_l_plastic(policy)
    total = l_task + lambda_reg * l_reg + lambda_sparse * l_sparse + lambda_homeo * l_homeo + lambda_plastic * l_plastic
    components = {
        "l_task": l_task,
        "l_reg": lambda_reg * l_reg,
        "l_sparse": lambda_sparse * l_sparse,
        "l_homeo": lambda_homeo * l_homeo,
        "l_plastic": lambda_plastic * l_plastic,
    }
    return total, components


def score_result(result) -> float:
    """Offline diagnostic score aligned with the V7 edge-selection contract."""
    trades = len(getattr(result, "trades", []))
    if trades == 0:
        return -1000.0
    daily = np.asarray(getattr(result, "daily_returns", []), dtype=float)
    lcb = edge_confidence_lcb(result)
    downside = float(np.mean(np.minimum(daily, 0.0))) if daily.size else 0.0
    stability = float(np.std(daily)) if daily.size else 0.0
    evidence = min(1.0, trades / 12.0)
    return float(
        100.0 * lcb
        + 20.0 * float(getattr(result, "expectancy", 0.0))
        + 7.5 * float(getattr(result, "total_r", 0.0)) * evidence
        - 400.0 * float(getattr(result, "max_drawdown_pct", 0.0))
        - 70.0 * abs(downside)
        - 10.0 * stability
        - float(getattr(result, "risk_abuse_penalty", 0.0))
        - (100.0 if bool(getattr(result, "ruin", False)) else 0.0)
        - max(0, 5 - trades) * 20.0
    )

