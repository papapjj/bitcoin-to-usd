from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from self_growing_trader.config import AppConfig, FeatureConfig, OrderConfig, RiskConfig, SymbolConfig
from self_growing_trader.environment import TradingEnvironment
from self_growing_trader.features import add_features
from self_growing_trader.fitness import tight_risk_score
from self_growing_trader.mt5_runner import MT5BacktestConfig, write_tester_set
from self_growing_trader.orders import Action, DecodedDecision, Position
from self_growing_trader.protocol_contract import INPUT_COUNT


def _config(max_positions: int = 5, *, slippage: float = 0.0) -> AppConfig:
    return AppConfig(
        symbol=SymbolConfig(point_size=0.01, spread_in_points=True),
        features=FeatureConfig(atr_period=3, range_window=5, atr_percentile_window=5, max_spread_atr=10.0, shock_range_atr=20.0),
        risk=RiskConfig(
            initial_equity=10000.0,
            min_risk_pct=0.001,
            max_risk_pct=0.02,
            max_open_risk_pct=0.05,
            max_positions=max_positions,
            enforce_entry_limits=True,
            slippage_atr_multiple=slippage,
        ),
        orders=OrderConfig(max_holding_bars=10, pending_expiry_bars=5, max_spread_cost_r=0.2, entry_offset_atr=2.0),
    )


def _data(rows: int = 240) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=rows, freq="min").strftime("%d/%m/%Y %H:%M"),
            "Open": np.linspace(100.0, 104.0, rows),
            "High": np.linspace(101.0, 105.0, rows),
            "Low": np.linspace(99.0, 103.0, rows),
            "Close": np.linspace(100.5, 104.5, rows),
            "Spread": [10] * rows,
        }
    )
    cfg = _config()
    return add_features(raw, cfg.features, cfg.symbol)


@pytest.mark.parametrize(
    ("action", "expected_sign"),
    [
        (Action.BUY_LIMIT, -1),
        (Action.SELL_STOP, -1),
        (Action.SELL_LIMIT, 1),
        (Action.BUY_STOP, 1),
    ],
)
def test_pending_order_direction_is_execution_contract(action: Action, expected_sign: int) -> None:
    env = TradingEnvironment(_data(), _config())
    source_idx = 100
    decision = DecodedDecision(action, 0.01, 1.0, 2.0, 1.5)
    env._open_or_place(source_idx + 1, decision, source_idx)
    assert len(env.pending_orders) == 1
    order = env.pending_orders[0]
    reference = float(env.close_values[source_idx])
    offset = 1.5 * float(env.atr_values[source_idx])
    assert np.isclose(order.entry, reference + expected_sign * offset)


def test_schema_remains_canonical_when_trade_cap_is_lowered() -> None:
    env = TradingEnvironment(_data(), _config(max_positions=2))
    assert env.observation_size() == INPUT_COUNT
    assert env.observation(100).shape == (INPUT_COUNT,)


def test_slippage_is_atr_multiple_not_price_percentage() -> None:
    env = TradingEnvironment(_data(), _config(slippage=0.5))
    assert env._apply_slippage(1, 1000.0, 4.0) == pytest.approx(1002.0)
    assert env._apply_slippage(-1, 1000.0, 4.0) == pytest.approx(998.0)


def test_fitness_never_rewards_a_worse_than_minus_five_percent_day() -> None:
    base = dict(
        trades=[SimpleNamespace()],
        max_drawdown_pct=0.0,
        total_r=0.0,
        expectancy=0.0,
        risk_abuse_penalty=0.0,
        ruin=False,
        daily_trade_counts=[1, 1],
        daily_bar_counts=[1, 1],
        min_daily_trades=0,
        max_daily_trades_limit=10,
    )
    mild = SimpleNamespace(**base, daily_returns=[-0.05, 0.0])
    severe = SimpleNamespace(**base, daily_returns=[-0.10, 0.0])
    assert tight_risk_score(severe) < tight_risk_score(mild)


def test_frozen_evaluation_does_not_call_learning_methods() -> None:
    class Policy:
        def __init__(self) -> None:
            self.observe_calls = 0
            self.develop_calls = 0

        def __call__(self, _obs):
            return np.zeros(20)

        def observe_transition(self, *_args):
            self.observe_calls += 1
            return {}

        def develop(self, *_args):
            self.develop_calls += 1

    policy = Policy()
    env = TradingEnvironment(_data(), _config())
    env.run(policy, learning_enabled=False)
    assert policy.observe_calls == 0
    assert policy.develop_calls == 0


def test_tester_set_does_not_serialize_python_only_risk_contract(tmp_path) -> None:
    config = MT5BacktestConfig(
        runner_dir=tmp_path,
        max_positions=5,
        min_risk_pct=0.003,
        max_risk_pct=0.015,
        max_open_risk_pct=0.04,
        pending_expiry_bars=19,
        max_spread_atr=0.7,
        max_spread_cost_r=0.2,
        shock_range_atr=3.0,
    )
    path = write_tester_set(config, tmp_path / "bridge.set")
    content = path.read_text(encoding="utf-8")
    for removed_input in (
        "MinRiskPct=",
        "MaxRiskPct=",
        "MaxOpenRiskPct=",
        "MaxSpreadAtr=",
        "MaxSpreadCostR=",
        "ShockRangeAtr=",
        "EnforceEntryLimits=",
    ):
        assert removed_input not in content
    assert "PendingExpiryBars=19" in content

def test_mql5_v15_execution_and_credit_contract_is_integrated_in_canonical_ea() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ea_source = project_root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    source = ea_source.read_text(encoding="utf-8")

    for marker in (
        "#define EXECUTION_LEDGER_SCHEMA_VERSION 17",
        "#define BRIDGE_SCHEMA_VERSION 10",
        "position_owner_trace_id",
        "intent_context",
        "ownerTraceId",
        "positionFinalClose",
        "OnTradeTransaction",
        "ExitCounterfactualHorizonBars",
    ):
        assert marker in source



def test_unrestricted_entry_mode_ignores_strategy_level_caps() -> None:
    """Default V6 mode keeps opening trades after loss/cadence/spread shocks.

    The five-slot position capacity and numerical/risk-output validity remain
    intentionally active; they are execution-contract constraints, not
    discretionary risk gates.
    """
    config = AppConfig(
        symbol=SymbolConfig(point_size=0.01, spread_in_points=True),
        features=FeatureConfig(atr_period=3, range_window=5, atr_percentile_window=5, max_spread_atr=0.01, shock_range_atr=0.01),
        risk=RiskConfig(
            initial_equity=10_000.0,
            min_risk_pct=0.001,
            max_risk_pct=0.02,
            max_open_risk_pct=0.02,
            max_positions=5,
            ruin_equity_pct=0.9,
            enforce_entry_limits=False,
        ),
        orders=OrderConfig(timeframe="M5", max_holding_bars=10, pending_expiry_bars=5, min_daily_trades=0, max_daily_trades=1, max_spread_cost_r=0.0001),
    )
    env = TradingEnvironment(_data(), config)
    env.current_day_trade_count = 1
    env.equity = 1_000.0  # Below the configured ruin threshold: still allowed.
    env.positions = [
        # Existing risk exceeds max_open_risk_pct once the next order is added.
        # Position capacity remains below the fixed five-slot protocol cap.
        Position(1, 100.0, 99.0, 102.0, 0.02, 200.0, 1.0, 200.0, 0)
    ]
    env.spread_atr_values[0] = 999.0
    env.not_shock_values[0] = 0.0

    assert env._has_open_capacity()
    assert env._can_open(0.02, 1.0, 2.0, 0)


def test_tester_set_defaults_to_unrestricted_entry_mode(tmp_path) -> None:
    path = write_tester_set(MT5BacktestConfig(runner_dir=tmp_path), tmp_path / "bridge.set")
    assert "EnforceEntryLimits=" not in path.read_text(encoding="utf-8")
