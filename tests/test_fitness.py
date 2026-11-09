from __future__ import annotations

from dataclasses import dataclass

from self_growing_trader.fitness import goal_progress_metrics


@dataclass
class DummyResult:
    daily_returns: list[float]
    total_r: float
    expectancy: float
    max_drawdown_pct: float
    risk_abuse_penalty: float
    ruin: bool
    daily_trade_counts: list[int] | None = None
    daily_bar_counts: list[int] | None = None
    min_daily_trades: int = 0
    max_daily_trades_limit: int = 60


def test_goal_progress_rewards_capped_daily_trade_cadence() -> None:
    sparse = DummyResult(
        [0.06, 0.02],
        total_r=5.0,
        expectancy=0.5,
        max_drawdown_pct=0.0,
        risk_abuse_penalty=0.0,
        ruin=False,
        daily_trade_counts=[3, 4],
        min_daily_trades=12,
    )
    active = DummyResult(
        [0.06, 0.02],
        total_r=5.0,
        expectancy=0.5,
        max_drawdown_pct=0.0,
        risk_abuse_penalty=0.0,
        ruin=False,
        daily_trade_counts=[12, 20],
        min_daily_trades=12,
    )
    active_metrics = goal_progress_metrics(active)
    assert active_metrics["stable_goal_score"] > goal_progress_metrics(sparse)["stable_goal_score"]
    assert active_metrics["daily_trade_completion_rate"] == 1.0
    assert active_metrics["daily_trade_day_hit_rate"] == 1.0


def test_goal_progress_prorates_min_trades_for_partial_days() -> None:
    partial = DummyResult(
        [0.02],
        total_r=1.0,
        expectancy=0.5,
        max_drawdown_pct=0.0,
        risk_abuse_penalty=0.0,
        ruin=False,
        daily_trade_counts=[6],
        daily_bar_counts=[720],
        min_daily_trades=12,
    )
    metrics = goal_progress_metrics(partial)
    assert metrics["daily_trade_completion_rate"] == 1.0
    assert metrics["daily_trade_day_hit_rate"] == 1.0


def test_goal_progress_rewards_target_days_and_total_r() -> None:
    weak = DummyResult([0.01, 0.0], total_r=1.0, expectancy=0.1, max_drawdown_pct=0.0, risk_abuse_penalty=0.0, ruin=False)
    strong = DummyResult([0.06, 0.05], total_r=5.0, expectancy=0.5, max_drawdown_pct=0.0, risk_abuse_penalty=0.0, ruin=False)
    assert goal_progress_metrics(strong)["stable_goal_score"] > goal_progress_metrics(weak)["stable_goal_score"]
    assert goal_progress_metrics(strong)["goal_hit"] is True


def test_goal_progress_penalizes_drawdown_and_ruin() -> None:
    clean = DummyResult([0.06], total_r=5.0, expectancy=0.5, max_drawdown_pct=0.0, risk_abuse_penalty=0.0, ruin=False)
    risky = DummyResult([0.06], total_r=5.0, expectancy=0.5, max_drawdown_pct=0.2, risk_abuse_penalty=10.0, ruin=True)
    assert goal_progress_metrics(risky)["stable_goal_score"] < goal_progress_metrics(clean)["stable_goal_score"]
    assert goal_progress_metrics(risky)["goal_hit"] is False

