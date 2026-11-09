from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from self_growing_trader.config import AppConfig, FeatureConfig, OrderConfig, RiskConfig, SymbolConfig
from self_growing_trader.environment import TradingEnvironment
from self_growing_trader.features import TIMING_FEATURE_COLUMNS, WYCKOFF_FEATURE_COLUMNS, add_features
from self_growing_trader.protocol_contract import (
    BASE_MARKET_FEATURE_COUNT,
    EXTRA_MARKET_FEATURE_COUNT,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    TIMING_OFFSET,
    WYCKOFF_OFFSET,
)
from self_growing_trader.orders import Action, DecodedDecision, Position, PositionManagementMode


def _config() -> AppConfig:
    return AppConfig(
        symbol=SymbolConfig(point_size=0.01, spread_in_points=True),
        features=FeatureConfig(atr_period=3, range_window=5, atr_percentile_window=5, max_spread_atr=10.0, shock_range_atr=20.0),
        risk=RiskConfig(initial_equity=10000.0, min_risk_pct=0.001, max_risk_pct=0.02, max_open_risk_pct=0.05, max_positions=5, enforce_entry_limits=True),
        orders=OrderConfig(max_holding_bars=3, pending_expiry_bars=3, min_daily_trades=6, max_daily_trades=12, max_spread_cost_r=0.2),
    )


def _data(rows: int = 200) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=rows, freq="min").strftime("%d/%m/%Y %H:%M"),
            "Open": np.linspace(100, 110, rows),
            "High": np.linspace(101, 111, rows),
            "Low": np.linspace(99, 109, rows),
            "Close": np.linspace(100.5, 110.5, rows),
            "Spread": [10] * rows,
        }
    )
    return add_features(raw, _config().features, _config().symbol)


def test_can_open_uses_risk_gate_not_age_norm() -> None:
    config = _config()
    env = TradingEnvironment(_data(), config)
    idx = 0
    assert env._can_open(0.01, 1.0, 2.0, idx)


def test_zero_position_limit_keeps_simulator_capacity_open() -> None:
    base = _config()
    config = replace(
        base,
        risk=RiskConfig(
            initial_equity=10000.0,
            min_risk_pct=0.001,
            max_risk_pct=0.02,
            max_open_risk_pct=0.05,
            max_positions=0,
            enforce_entry_limits=False,
        ),
    )
    env = TradingEnvironment(_data(), config)
    env.positions = [
        Position(1, 100.0, 99.0, 102.0, 0.001, 10.0, 1.0, 10.0, 0)
        for _ in range(8)
    ]
    assert env._has_open_capacity()
    assert env._can_open(0.001, 1.0, 2.0, 0)


def test_risk_cap_blocks_total_open_risk_above_limit() -> None:
    config = _config()
    env = TradingEnvironment(_data(), config)
    env.positions = [
        Position(1, 100.0, 99.0, 102.0, 0.02, 200.0, 1.0, 200.0, 0),
        Position(1, 100.0, 99.0, 102.0, 0.02, 200.0, 1.0, 200.0, 0),
    ]
    assert not env._can_open(0.02, 1.0, 2.0, 0)


def test_daily_trade_cap_resets_on_new_day() -> None:
    config = AppConfig(
        symbol=SymbolConfig(point_size=0.01, spread_in_points=True),
        features=FeatureConfig(atr_period=3, range_window=5, atr_percentile_window=5, max_spread_atr=10.0, shock_range_atr=20.0),
        risk=RiskConfig(initial_equity=10000.0, min_risk_pct=0.001, max_risk_pct=0.02, max_open_risk_pct=0.05, max_positions=5, enforce_entry_limits=True),
        orders=OrderConfig(max_holding_bars=3, pending_expiry_bars=3, max_daily_trades=1, max_spread_cost_r=0.2),
    )
    env = TradingEnvironment(_data(1600), config)
    env._roll_day(0)
    close = float(env.data.iloc[0]["Close"])
    assert env._try_open(0, 1, close + float(env.data.iloc[0]["spread_price"]), 5.0, 10.0, 0.01)
    assert not env._try_open(1, 1, close + float(env.data.iloc[1]["spread_price"]), 5.0, 10.0, 0.01)
    next_day_idx = int(np.flatnonzero(env.day_values != env.day_values[0])[0])
    env._roll_day(next_day_idx)
    assert env._try_open(next_day_idx, 1, float(env.data.iloc[next_day_idx]["Close"]) + float(env.data.iloc[next_day_idx]["spread_price"]), 5.0, 10.0, 0.01)
    result = env._result()
    assert result.daily_trade_counts == [1]
    assert result.daily_bar_counts == [1]


def test_pending_order_expires() -> None:
    config = _config()
    env = TradingEnvironment(_data(), config)
    decision = DecodedDecision(Action.BUY_LIMIT, 0.01, 1.0, 2.0, -5.0)
    env._place_pending(0, Action.BUY_LIMIT, 1, 1.0, decision)
    assert len(env.pending_orders) == 1
    env._trigger_pending(config.orders.pending_expiry_bars)
    assert len(env.pending_orders) == 0


def test_force_close_after_max_holding_bars() -> None:
    config = _config()
    env = TradingEnvironment(_data(), config)
    assert env._try_open(0, 1, float(env.data.iloc[0]["Close"]) + float(env.data.iloc[0]["spread_price"]), 5.0, 10.0, 0.01)
    env._update_open_positions(config.orders.max_holding_bars)
    assert len(env.positions) == 0
    assert env.trades[-1].reason == "max_holding"


def test_market_decision_at_t_executes_at_next_bar_open() -> None:
    config = _config()
    env = TradingEnvironment(_data(), config)
    calls = 0

    def policy(_obs):
        nonlocal calls
        out = np.zeros(15)
        out[1 if calls == 0 else 0] = 10.0
        calls += 1
        return out

    result = env.run(policy)
    assert result.trades
    first_trade = result.trades[0]
    expected_entry = float(env.data.iloc[1]["Open"]) + float(env.data.iloc[1]["spread_price"])
    assert first_trade.opened_bar == 1
    assert first_trade.entry == expected_entry


def test_pending_fill_defers_intrabar_exit_until_next_bar() -> None:
    config = _config()
    env = TradingEnvironment(_data(), config)
    # Pending created on bar t cannot consume bar t's OHLC. It becomes
    # eligible on the following observed bar.
    idx = 2
    row = env.data.iloc[idx]
    entry_bid = float(row["Low"]) + 1.2
    decision = DecodedDecision(Action.BUY_LIMIT, 0.01, 0.5, 1.0, 0.0)
    env._place_pending(idx - 1, Action.BUY_LIMIT, 1, entry_bid, decision)
    env._trigger_pending(idx)
    # A decision/fill cannot inspect the same bar high/low.  Exit is deferred
    # to the next observed bar to prevent intrabar look-through leakage.
    env._update_open_positions(idx)
    assert len(env.positions) == 1
    assert env.trades == []
    env._update_open_positions(idx + 1)
    assert len(env.positions) == 0
    assert env.trades[-1].reason == "sl"
    assert env.trades[-1].r < 0


def test_close_best_and_worst_use_unrealized_r() -> None:
    env = TradingEnvironment(_data(), _config())
    close = float(env.data.iloc[0]["Close"])
    best_by_r = Position(1, close - 1.0, close - 2.0, close + 4.0, 0.01, 100.0, 1.0, 1.0, 0)
    worse_by_r = Position(1, close - 5.0, close - 15.0, close + 10.0, 0.01, 100.0, 10.0, 100.0, 0)
    env.positions = [best_by_r, worse_by_r]
    env._close_ranked(0, best=True)
    assert env.trades[-1].entry == best_by_r.entry

    env = TradingEnvironment(_data(), _config())
    env.positions = [best_by_r, worse_by_r]
    env._close_ranked(0, best=False)
    assert env.trades[-1].entry == worse_by_r.entry


def test_allowed_actions_exclude_close_when_no_positions() -> None:
    env = TradingEnvironment(_data(), _config())
    assert Action.CLOSE_BEST not in env._allowed_actions()
    assert Action.CLOSE_WORST not in env._allowed_actions()
    assert Action.CLOSE_ALL not in env._allowed_actions()
    assert Action.MARKET_BUY in env._allowed_actions()


def test_allowed_actions_exclude_open_when_daily_cap_reached() -> None:
    config = _config()
    env = TradingEnvironment(_data(), replace(config, orders=replace(config.orders, min_daily_trades=0, max_daily_trades=1)))
    env.current_day_trade_count = 1
    assert Action.MARKET_BUY not in env._allowed_actions()
    assert Action.BUY_STOP not in env._allowed_actions()
    assert Action.HOLD in env._allowed_actions()


def test_environment_observation_size() -> None:
    env = TradingEnvironment(_data(), _config())
    assert env.observation_size() == INPUT_COUNT


def test_observation_includes_daily_trade_progress() -> None:
    env = TradingEnvironment(_data(800), _config())
    idx = 360
    env._roll_day(idx)
    env.current_day_trade_count = 6
    obs = env.observation(idx)
    day_progress = float(env.day_progress_values[idx])
    account_offset = BASE_MARKET_FEATURE_COUNT + MAX_POSITION_SLOTS * 8
    assert np.isclose(obs[account_offset + 4], 6 / env.config.orders.max_daily_trades)
    assert np.isclose(obs[account_offset + 5], 6 / env.config.orders.min_daily_trades)
    assert np.isclose(obs[account_offset + 6], day_progress)
    assert np.isclose(obs[account_offset + 7], 0.0)


def test_observation_includes_cadence_deficit_when_behind_schedule() -> None:
    env = TradingEnvironment(_data(800), _config())
    idx = 600
    env._roll_day(idx)
    env.current_day_trade_count = 1
    obs = env.observation(idx)
    day_progress = float(env.day_progress_values[idx])
    expected_deficit = ((env.config.orders.min_daily_trades * day_progress) - 1) / env.config.orders.min_daily_trades
    account_offset = BASE_MARKET_FEATURE_COUNT + MAX_POSITION_SLOTS * 8
    assert np.isclose(obs[account_offset + 6], day_progress)
    assert np.isclose(obs[account_offset + 7], expected_deficit)


def test_observation_appends_live_spread_and_distance_to_sl_tp() -> None:
    env = TradingEnvironment(_data(), _config())
    idx = 20
    close_bid = float(env.close_values[idx])
    atr = float(env.atr_values[idx])
    spread_atr = float(env.spread_atr_values[idx])
    env.positions = [
        Position(
            side=1,
            entry=close_bid,
            sl=close_bid - 2.0 * atr,
            tp=close_bid + 3.0 * atr,
            risk_pct=0.01,
            risk_amount=100.0,
            risk_distance=2.0 * atr,
            size=1.0,
            opened_bar=idx - 2,
        )
    ]

    obs = env.observation(idx)
    execution_offset = BASE_MARKET_FEATURE_COUNT + MAX_POSITION_SLOTS * 8 + 8 + EXTRA_MARKET_FEATURE_COUNT

    assert np.isclose(obs[execution_offset], spread_atr)
    assert np.isclose(obs[execution_offset + 1], 2.0)
    assert np.isclose(obs[execution_offset + 2], 3.0)


def test_observation_appends_timing_and_wyckoff_features_in_tail_slots() -> None:
    env = TradingEnvironment(_data(260), _config())
    idx = 120
    obs = env.observation(idx)
    expected_timing = env.data.iloc[idx][TIMING_FEATURE_COLUMNS].to_numpy(dtype=float)
    expected_wyckoff = env.data.iloc[idx][WYCKOFF_FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.allclose(obs[TIMING_OFFSET:WYCKOFF_OFFSET], expected_timing)
    assert np.allclose(obs[WYCKOFF_OFFSET:], expected_wyckoff)
    assert obs.size == INPUT_COUNT




def test_hold_and_compatibility_manage_position_are_noops_in_fixed_geometry_mode() -> None:
    env = TradingEnvironment(_data(), _config())
    idx = 20
    close = float(env.data.iloc[idx]["Close"])
    atr = float(env.atr_values[idx])
    original = Position(
        side=1,
        entry=close - atr,
        sl=close - 2.0 * atr,
        tp=close + 3.0 * atr,
        risk_pct=0.01,
        risk_amount=100.0,
        risk_distance=atr,
        size=1.0,
        opened_bar=idx - 2,
    )
    env.positions = [original]
    management = dict(
        risk_pct=0.01,
        sl_atr=1.0,
        tp_atr=2.0,
        entry_offset_atr=0.0,
        partial_close_pct=1.0,
        target_slot=0,
        management_mode=PositionManagementMode.PARTIAL_CLOSE,
    )

    env._apply_decision(idx, DecodedDecision(Action.HOLD, **management))

    assert env.positions == [original]
    assert env.trades == []

    env._apply_decision(idx, DecodedDecision(Action.MANAGE_POSITION, **management))

    assert env.positions == [original]
    assert env.trades == []


def test_factorized_manage_position_can_partially_close_selected_slot() -> None:
    env = TradingEnvironment(_data(), _config())
    idx = 20
    close = float(env.data.iloc[idx]["Close"])
    atr = float(env.atr_values[idx])
    first = Position(1, close - atr, close - 2 * atr, close + 3 * atr, 0.01, 100.0, atr, 10.0, idx - 3, origin_trace_id="entry-a")
    second = Position(1, close - atr, close - 2 * atr, close + 3 * atr, 0.02, 200.0, atr, 20.0, idx - 3, origin_trace_id="entry-b")
    env.positions = [first, second]

    decision = DecodedDecision(
        Action.MANAGE_POSITION,
        risk_pct=0.01,
        sl_atr=1.0,
        tp_atr=2.0,
        entry_offset_atr=0.0,
        partial_close_pct=0.5,
        target_slot=1,
        management_mode=PositionManagementMode.PARTIAL_CLOSE,
        factorized_contract=True,
    )
    env._apply_decision(idx, decision, trace_id="manage-b")

    assert len(env.positions) == 2
    assert np.isclose(env.positions[0].size, 10.0)
    assert np.isclose(env.positions[1].size, 10.0)
    assert np.isclose(env.positions[1].risk_amount, 100.0)
    assert np.isclose(env.positions[1].risk_pct, 0.01)
    assert len(env.trades) == 1
    assert env.trades[0].origin_trace_id == "entry-b"
    assert env.trades[0].management_trace_id == "manage-b"
    assert env.trades[0].reason == "manage_partial_close"


def test_factorized_manage_position_can_fully_close_selected_slot() -> None:
    env = TradingEnvironment(_data(), _config())
    idx = 20
    close = float(env.data.iloc[idx]["Close"])
    atr = float(env.atr_values[idx])
    first = Position(1, close - atr, close - 2 * atr, close + 3 * atr, 0.01, 100.0, atr, 10.0, idx - 3, origin_trace_id="entry-a")
    second = Position(-1, close + atr, close + 2 * atr, close - 3 * atr, 0.01, 100.0, atr, 10.0, idx - 3, origin_trace_id="entry-b")
    env.positions = [first, second]

    decision = DecodedDecision(
        Action.MANAGE_POSITION,
        risk_pct=0.01,
        sl_atr=1.0,
        tp_atr=2.0,
        entry_offset_atr=0.0,
        target_slot=0,
        management_mode=PositionManagementMode.FULL_CLOSE,
        factorized_contract=True,
    )
    env._apply_decision(idx, decision, trace_id="manage-a")

    assert env.positions == [second]
    assert len(env.trades) == 1
    assert env.trades[0].origin_trace_id == "entry-a"
    assert env.trades[0].management_trace_id == "manage-a"
    assert env.trades[0].reason == "manage_full_close"


def test_factorized_action_delay_waits_extra_bars_without_policy_overwrite() -> None:
    env = TradingEnvironment(_data(), _config())
    calls = 0

    def policy(_obs):
        nonlocal calls
        calls += 1
        out = np.zeros(20, dtype=float)
        if calls == 1:
            out[int(Action.MARKET_BUY)] = 10.0
            out[13] = 0.0  # sigmoid(0)*4 -> delay 2
            out[14] = 5.0  # legacy factorized marker (fixed geometry)
        else:
            out[int(Action.HOLD)] = 10.0
            out[14] = 5.0
            out[13] = -10.0
        return out

    result = env.run(policy, learning_enabled=False)
    assert result.trades
    assert result.trades[0].opened_bar == 3
    # The policy is called at bar 0, then not again on bars 1 and 2 while the
    # delayed action is pending. It may be called again only after execution.
    assert calls < len(env.data) - 1


def test_learned_trailing_uses_immutable_entry_atr_when_volatility_changes() -> None:
    config = replace(_config(), orders=replace(_config().orders, max_holding_bars=100))
    env = TradingEnvironment(_data(), config)
    idx = 20
    close_bid = float(env.close_values[idx])
    env.atr_values[idx] = 100.0
    position = Position(
        side=1,
        entry=close_bid,
        sl=close_bid - 50.0,
        tp=close_bid + 1000.0,
        risk_pct=0.01,
        risk_amount=100.0,
        risk_distance=2.0,
        size=50.0,
        opened_bar=idx - 1,
        trailing_stop_atr=1.5,
        initial_atr=2.0,
    )
    env.positions = [position]

    env._update_open_positions(idx)

    assert env.positions == [position]
    assert np.isclose(position.sl, close_bid - 3.0)
