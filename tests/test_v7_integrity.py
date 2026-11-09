from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from self_growing_trader.config import AppConfig, FeatureConfig, LearningIntegrityConfig, OrderConfig, RiskConfig, SymbolConfig, load_config
from self_growing_trader.feature_contract import (
    _reference_vector_for_audit_row,
    build_reference_input,
    compare_mt5_feature_audit_against_reference,
    feature_schema_fingerprint,
)
from self_growing_trader.features import (
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
    add_features,
)
from self_growing_trader.learning_integrity import (
    clone_policy,
    frozen_oos_promotion_gate,
    paired_oos_non_regression_gate,
    policy_state_hash,
    promotion_stability_report,
)
from self_growing_trader.orders import (
    Action,
    MIN_MANUAL_CLOSE_AGE_BARS,
    allowed_actions_for_state,
    pending_entry_bid_basis,
)


def _raw(rows: int = 700) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=rows, freq="min")
    drift = np.linspace(100.0, 111.0, rows)
    wave = np.sin(np.arange(rows) / 9.0) * 0.35
    close = drift + wave
    return pd.DataFrame(
        {
            "Datetime": t.strftime("%Y-%m-%d %H:%M"),
            "Open": close - 0.03,
            "High": close + 0.20,
            "Low": close - 0.21,
            "Close": close,
            "Spread": 10,
            "TickVolume": 100.0 + 25.0 * np.sin(np.arange(rows) / 7.0),
        }
    )


def _config() -> AppConfig:
    return AppConfig(
        symbol=SymbolConfig(point_size=0.01, spread_in_points=True),
        features=FeatureConfig(atr_period=14, range_window=60, atr_percentile_window=240),
        risk=RiskConfig(max_positions=5),
        orders=OrderConfig(min_daily_trades=0, max_daily_trades=0),
    )


def test_v10_full_137_feature_audit_accepts_exact_reference(tmp_path: Path) -> None:
    config = _config()
    raw = _raw()
    reference_path = tmp_path / "reference.csv"
    raw.to_csv(reference_path, index=False)
    # Parity must use the exact on-disk reference CSV consumed by the verifier.
    # CSV serialization can change near-tie percentile ranks in synthetic data.
    frame = add_features(pd.read_csv(reference_path), config.features, config.symbol)
    row = frame.iloc[-1]
    timestamp = pd.Timestamp(row["Datetime"]).strftime("%Y-%m-%d %H:%M:%S")
    day_progress = (pd.Timestamp(row["Datetime"]).hour * 60 + pd.Timestamp(row["Datetime"]).minute) / 1440.0
    expected = build_reference_input(
        row,
        account_features=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, day_progress, 0.0],
        execution_features=[float(row["spread_price"]) / float(row["atr"])] + [0.0] * 10,
    )
    expected_tail = row[TIMING_FEATURE_COLUMNS + WYCKOFF_FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.allclose(expected[-20:], expected_tail)
    assert np.any(np.abs(expected[-20:]) > 1e-12)
    schema = feature_schema_fingerprint(config.features, config.symbol)
    audit_path = tmp_path / "audit.csv"
    audit_path.write_text(
        "feature_schema_hash,event_id,logged_at,bar_time,inputs,positions_pipe,orders_pipe,balance,equity,drawdown_pct,spread_points,day_trades,open_risk_ratio,symbol_point,symbol_digits,tick_size,tick_value,contract_size,margin_mode\n"
        + ",".join(
            [
                schema,
                "evt-1",
                timestamp,
                timestamp,
                '"' + "|".join(f"{x:.12g}" for x in expected) + '"',
                "",
                "",
                "10000",
                "10000",
                "0",
                "10",
                "0",
                "0",
                "0.01",
                "2",
                "0.01",
                "1",
                "1",
                "2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = compare_mt5_feature_audit_against_reference(
        audit_path,
        reference_path,
        config.features,
        config.symbol,
        expected_schema_hash=schema,
        initial_equity=10000.0,
    )
    assert result["numerical_checked"] is True
    assert result["numerical_valid"] is True
    assert result["checked_rows"] == 1


def test_v10_full_137_feature_audit_rejects_single_slot_drift(tmp_path: Path) -> None:
    config = _config()
    raw = _raw()
    reference_path = tmp_path / "reference.csv"
    raw.to_csv(reference_path, index=False)
    # Parity must use the exact on-disk reference CSV consumed by the verifier.
    # CSV serialization can change near-tie percentile ranks in synthetic data.
    frame = add_features(pd.read_csv(reference_path), config.features, config.symbol)
    row = frame.iloc[-1]
    timestamp = pd.Timestamp(row["Datetime"]).strftime("%Y-%m-%d %H:%M:%S")
    day_progress = (pd.Timestamp(row["Datetime"]).hour * 60 + pd.Timestamp(row["Datetime"]).minute) / 1440.0
    expected = build_reference_input(
        row,
        account_features=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, day_progress, 0.0],
        execution_features=[float(row["spread_price"]) / float(row["atr"])] + [0.0] * 10,
    )
    expected[-1] += 0.25
    schema = feature_schema_fingerprint(config.features, config.symbol)
    audit_path = tmp_path / "audit.csv"
    audit_path.write_text(
        "feature_schema_hash,event_id,logged_at,bar_time,inputs,positions_pipe,orders_pipe,balance,equity,drawdown_pct,spread_points,day_trades,open_risk_ratio,symbol_point,symbol_digits,tick_size,tick_value,contract_size,margin_mode\n"
        + ",".join([schema, "evt-1", timestamp, timestamp, '"' + "|".join(f"{x:.12g}" for x in expected) + '"', "", "", "10000", "10000", "0", "10", "0", "0", "0.01", "2", "0.01", "1", "1", "2"])
        + "\n",
        encoding="utf-8",
    )
    result = compare_mt5_feature_audit_against_reference(
        audit_path,
        reference_path,
        config.features,
        config.symbol,
        expected_schema_hash=schema,
        initial_equity=10000.0,
    )
    assert result["numerical_valid"] is False
    assert result["mismatch_count"] == 1


def test_v10_feature_parity_uses_narrow_real_tick_volume_tolerance(tmp_path: Path) -> None:
    config = _config()
    raw = _raw()
    reference_path = tmp_path / "reference.csv"
    raw.to_csv(reference_path, index=False)
    frame = add_features(pd.read_csv(reference_path), config.features, config.symbol)
    row = frame.iloc[-1]
    bar_time = pd.Timestamp(row["Datetime"])
    timestamp = bar_time.strftime("%Y-%m-%d %H:%M:%S")
    day_progress = (bar_time.hour * 60 + bar_time.minute) / 1440.0
    expected = build_reference_input(
        row,
        account_features=[0.0] * 6 + [day_progress, 0.0],
        execution_features=[float(row["spread_price"]) / float(row["atr"])] + [0.0] * 10,
    )
    schema = feature_schema_fingerprint(config.features, config.symbol)
    audit_path = tmp_path / "audit.csv"

    def write_audit(values: np.ndarray) -> None:
        audit_path.write_text(
            "feature_schema_hash,event_id,logged_at,bar_time,inputs,positions_pipe,orders_pipe,balance,equity,drawdown_pct,spread_points,day_trades,open_risk_ratio,symbol_point,symbol_digits,tick_size,tick_value,contract_size,margin_mode\n"
            + ",".join([
                schema, "evt-volume", timestamp, timestamp,
                '"' + "|".join(f"{x:.12g}" for x in values) + '"',
                "", "", "10000", "10000", "0", "10", "0", "0", "0.01", "2", "0.01", "1", "1", "2",
            ])
            + "\n",
            encoding="utf-8",
        )

    # Index 121 is relative_volume_20.  A real-tick/bar-volume difference
    # inside the explicit 2.5e-4 envelope is accepted.
    within = expected.copy()
    within[121] += 2.0e-4
    write_audit(within)
    result = compare_mt5_feature_audit_against_reference(
        audit_path,
        reference_path,
        config.features,
        config.symbol,
        expected_schema_hash=schema,
        initial_equity=10000.0,
    )
    assert result["numerical_valid"] is True
    assert result["dynamic_volume_parity_atol"] == 2.5e-4

    # A materially larger drift remains fail-closed.
    outside = expected.copy()
    outside[121] += 4.0e-4
    write_audit(outside)
    result = compare_mt5_feature_audit_against_reference(
        audit_path,
        reference_path,
        config.features,
        config.symbol,
        expected_schema_hash=schema,
        initial_equity=10000.0,
    )
    assert result["numerical_valid"] is False
    assert result["mismatch_count"] == 1


def test_v10_feature_parity_uses_runtime_spread_for_tick_only_slots(tmp_path: Path) -> None:
    config = _config()
    raw = _raw()
    reference_path = tmp_path / "reference.csv"
    raw.to_csv(reference_path, index=False)
    frame = add_features(pd.read_csv(reference_path), config.features, config.symbol)
    row = frame.iloc[-1]
    timestamp = pd.Timestamp(row["Datetime"]).strftime("%Y-%m-%d %H:%M:%S")
    schema = feature_schema_fingerprint(config.features, config.symbol)
    audit_row = {
        "feature_schema_hash": schema,
        "event_id": "evt-runtime-spread",
        "logged_at": timestamp,
        "bar_time": timestamp,
        "positions_pipe": "",
        "orders_pipe": "",
        "balance": "10000",
        "equity": "10000",
        "drawdown_pct": "0",
        # Deliberately differ from the static bar spread (10 points).
        "spread_points": "25",
        "day_trades": "0",
        "open_risk_ratio": "0",
    }
    actual = _reference_vector_for_audit_row(
        row, audit_row, initial_equity=10000.0, min_daily_trades=0,
        max_daily_trades=0, max_holding_bars=60, point_size=config.symbol.point_size,
    )
    # Spread/percentile are tick-level fields; execution[0] is their strict
    # duplicate used to validate the runtime ratio.
    actual[15] = actual[106]
    actual[99] = 0.9
    audit_path = tmp_path / "audit.csv"
    audit_path.write_text(
        "feature_schema_hash,event_id,logged_at,bar_time,inputs,positions_pipe,orders_pipe,balance,equity,drawdown_pct,spread_points,day_trades,open_risk_ratio,symbol_point,symbol_digits,tick_size,tick_value,contract_size,margin_mode\n"
        + ",".join([
            schema, audit_row["event_id"], timestamp, timestamp,
            '"' + "|".join(f"{x:.12g}" for x in actual) + '"',
            "", "", "10000", "10000", "0", "25", "0", "0", "0.01", "2", "0.01", "1", "1", "2",
        ])
        + "\n",
        encoding="utf-8",
    )
    result = compare_mt5_feature_audit_against_reference(
        audit_path, reference_path, config.features, config.symbol,
        expected_schema_hash=schema, initial_equity=10000.0,
    )
    assert result["numerical_valid"] is True
    assert result["mismatch_count"] == 0
    assert result["spread_reference_override_rows"] == 1


def test_v10_feature_parity_allows_bounded_closed_bar_age_quantization(tmp_path: Path) -> None:
    config = _config()
    config = replace(config, orders=replace(config.orders, max_holding_bars=0))
    raw = _raw()
    reference_path = tmp_path / "reference.csv"
    raw.to_csv(reference_path, index=False)
    frame = add_features(pd.read_csv(reference_path), config.features, config.symbol)
    row = frame.iloc[-1]
    bar_time = pd.Timestamp(row["Datetime"])
    logged_at = bar_time + pd.Timedelta(minutes=1)
    positions_pipe = "1:-1:0.10:100.0:99.0:102.0:0.0:4.8"
    audit_row = {
        "feature_schema_hash": feature_schema_fingerprint(config.features, config.symbol),
        "event_id": "evt-age-quantized",
        "logged_at": logged_at.strftime("%Y-%m-%d %H:%M:%S"),
        "bar_time": bar_time.strftime("%Y-%m-%d %H:%M:%S"),
        "positions_pipe": positions_pipe,
        "orders_pipe": "",
        "balance": "10000",
        "equity": "10000",
        "drawdown_pct": "0",
        "spread_points": "10",
        "day_trades": "0",
        "open_risk_ratio": "0",
    }
    expected = _reference_vector_for_audit_row(
        row,
        audit_row,
        initial_equity=10000.0,
        min_daily_trades=0,
        max_daily_trades=0,
        max_holding_bars=0,
        point_size=config.symbol.point_size,
    )
    # The EA's closed-bar age can be almost one bar away from the rounded
    # snapshot age; this is inside the explicit quantization envelope.
    actual = expected.copy()
    actual[32] += 0.8
    schema = audit_row["feature_schema_hash"]
    audit_path = tmp_path / "audit.csv"
    audit_path.write_text(
        "feature_schema_hash,event_id,logged_at,bar_time,inputs,positions_pipe,orders_pipe,balance,equity,drawdown_pct,spread_points,day_trades,open_risk_ratio,symbol_point,symbol_digits,tick_size,tick_value,contract_size,margin_mode\n"
        + ",".join([
            schema, audit_row["event_id"], audit_row["logged_at"], audit_row["bar_time"],
            '"' + "|".join(f"{x:.12g}" for x in actual) + '"', positions_pipe, "", "10000", "10000", "0", "10", "0", "0", "0.01", "2", "0.01", "1", "1", "2",
        ])
        + "\n",
        encoding="utf-8",
    )
    result = compare_mt5_feature_audit_against_reference(
        audit_path, reference_path, config.features, config.symbol,
        expected_schema_hash=schema, initial_equity=10000.0, max_holding_bars=0,
    )
    assert result["numerical_valid"] is True
    assert result["mismatch_count"] == 0


def test_v10_feature_parity_rejects_large_position_age_drift(tmp_path: Path) -> None:
    config = _config()
    config = replace(config, orders=replace(config.orders, max_holding_bars=0))
    raw = _raw()
    reference_path = tmp_path / "reference.csv"
    raw.to_csv(reference_path, index=False)
    frame = add_features(pd.read_csv(reference_path), config.features, config.symbol)
    row = frame.iloc[-1]
    bar_time = pd.Timestamp(row["Datetime"])
    logged_at = bar_time + pd.Timedelta(minutes=1)
    positions_pipe = "1:-1:0.10:100.0:99.0:102.0:0.0:4.8"
    schema = feature_schema_fingerprint(config.features, config.symbol)
    audit_row = {
        "feature_schema_hash": schema,
        "event_id": "evt-age-drift",
        "logged_at": logged_at.strftime("%Y-%m-%d %H:%M:%S"),
        "bar_time": bar_time.strftime("%Y-%m-%d %H:%M:%S"),
        "positions_pipe": positions_pipe,
        "orders_pipe": "",
        "balance": "10000",
        "equity": "10000",
        "drawdown_pct": "0",
        "spread_points": "10",
        "day_trades": "0",
        "open_risk_ratio": "0",
    }
    expected = _reference_vector_for_audit_row(
        row, audit_row, initial_equity=10000.0, min_daily_trades=0,
        max_daily_trades=0, max_holding_bars=0, point_size=config.symbol.point_size,
    )
    actual = expected.copy()
    actual[32] += 1.2
    audit_path = tmp_path / "audit.csv"
    audit_path.write_text(
        "feature_schema_hash,event_id,logged_at,bar_time,inputs,positions_pipe,orders_pipe,balance,equity,drawdown_pct,spread_points,day_trades,open_risk_ratio,symbol_point,symbol_digits,tick_size,tick_value,contract_size,margin_mode\n"
        + ",".join([
            schema, audit_row["event_id"], audit_row["logged_at"], audit_row["bar_time"],
            '"' + "|".join(f"{x:.12g}" for x in actual) + '"', positions_pipe, "", "10000", "10000", "0", "10", "0", "0", "0.01", "2", "0.01", "1", "1", "2",
        ])
        + "\n",
        encoding="utf-8",
    )
    result = compare_mt5_feature_audit_against_reference(
        audit_path, reference_path, config.features, config.symbol,
        expected_schema_hash=schema, initial_equity=10000.0, max_holding_bars=0,
    )
    assert result["numerical_valid"] is False
    assert result["mismatch_count"] == 1


def test_v10_position_count_feature_keeps_positions_beyond_detail_slots() -> None:
    config = _config()
    frame = add_features(_raw(), config.features, config.symbol)
    row = frame.iloc[-1]
    timestamp = pd.Timestamp(row["Datetime"]).strftime("%Y-%m-%d %H:%M:%S")
    position = "1:1:0.10:105.0:104.0:107.0:0.0:1.0"
    audit_row = {
        "logged_at": timestamp,
        "bar_time": timestamp,
        "positions_pipe": "^".join(position for _ in range(6)),
        "balance": "10000",
        "equity": "10000",
        "drawdown_pct": "0",
        "spread_points": "10",
        "day_trades": "0",
        "open_risk_ratio": "0",
    }
    actual = _reference_vector_for_audit_row(
        row,
        audit_row,
        initial_equity=10000.0,
        min_daily_trades=0,
        max_daily_trades=0,
        max_holding_bars=60,
        point_size=config.symbol.point_size,
        max_positions=0,
    )
    # The wire has only five detailed position slots, but account feature 69
    # reports the complete count divided by the EA's reference capacity (5).
    assert actual[69] == 1.2
    assert np.count_nonzero(actual[29:69:8]) == 5


def test_v10_feature_parity_accepts_native_mt5_tab_export(tmp_path: Path) -> None:
    config = _config()
    raw = _raw()
    reference_path = tmp_path / "BTCUSDm.csv"
    mt5_export = pd.DataFrame(
        {
            "<DATE>": pd.to_datetime(raw["Datetime"]).dt.strftime("%Y.%m.%d"),
            "<TIME>": pd.to_datetime(raw["Datetime"]).dt.strftime("%H:%M:%S"),
            "<OPEN>": raw["Open"],
            "<HIGH>": raw["High"],
            "<LOW>": raw["Low"],
            "<CLOSE>": raw["Close"],
            "<TICKVOL>": raw["TickVolume"],
            "<VOL>": 0,
            "<SPREAD>": raw["Spread"],
        }
    )
    mt5_export.to_csv(reference_path, sep="\t", index=False)
    canonical = pd.read_csv(reference_path, sep="\t")
    canonical["Datetime"] = canonical["<DATE>"].astype(str) + " " + canonical["<TIME>"].astype(str)
    canonical = canonical.rename(
        columns={
            "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close",
            "<TICKVOL>": "TickVol", "<VOL>": "Vol", "<SPREAD>": "Spread",
        }
    )
    frame = add_features(canonical, config.features, config.symbol)
    row = frame.iloc[-1]
    timestamp = pd.Timestamp(row["Datetime"]).strftime("%Y-%m-%d %H:%M:%S")
    day_progress = (pd.Timestamp(row["Datetime"]).hour * 60 + pd.Timestamp(row["Datetime"]).minute) / 1440.0
    expected = build_reference_input(
        row,
        account_features=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, day_progress, 0.0],
        execution_features=[float(row["spread_price"]) / float(row["atr"])] + [0.0] * 10,
    )
    schema = feature_schema_fingerprint(config.features, config.symbol)
    audit_path = tmp_path / "audit.csv"
    audit_path.write_text(
        "feature_schema_hash,event_id,logged_at,bar_time,inputs,positions_pipe,orders_pipe,balance,equity,drawdown_pct,spread_points,day_trades,open_risk_ratio,symbol_point,symbol_digits,tick_size,tick_value,contract_size,margin_mode\n"
        + ",".join([
            schema, "evt-mt5", timestamp, timestamp,
            '"' + "|".join(f"{x:.12g}" for x in expected) + '"',
            "", "", "10000", "10000", "0", "10", "0", "0", "0.01", "2", "0.01", "1", "1", "2",
        ])
        + "\n",
        encoding="utf-8",
    )
    result = compare_mt5_feature_audit_against_reference(
        audit_path,
        reference_path,
        config.features,
        config.symbol,
        expected_schema_hash=schema,
        initial_equity=10000.0,
    )
    assert result["numerical_checked"] is True
    assert result["numerical_valid"] is True
    assert result["checked_rows"] == 1


def test_v7_pending_direction_is_canonical_for_all_four_order_types() -> None:
    price = 100.0
    atr = 2.0
    offset = 1.0
    assert pending_entry_bid_basis(Action.BUY_LIMIT, price, offset, atr) == 98.0
    assert pending_entry_bid_basis(Action.SELL_STOP, price, offset, atr) == 98.0
    assert pending_entry_bid_basis(Action.SELL_LIMIT, price, offset, atr) == 102.0
    assert pending_entry_bid_basis(Action.BUY_STOP, price, offset, atr) == 102.0


def test_v7_mask_blocks_young_management_but_keeps_entry_and_emergency_close() -> None:
    young = allowed_actions_for_state(
        position_count=1, pending_count=0, max_positions=5,
        ages_bars=[MIN_MANUAL_CLOSE_AGE_BARS - 1], unrealized_rs=[-0.10],
    )
    assert Action.MANAGE_POSITION not in young
    assert Action.MARKET_BUY in young
    emergency = allowed_actions_for_state(
        position_count=1, pending_count=0, max_positions=5,
        ages_bars=[0], unrealized_rs=[-1.0],
    )
    assert Action.CLOSE_WORST in emergency
    assert Action.MANAGE_POSITION not in emergency


def test_v7_load_config_resolves_relative_data_path_and_rejects_non_utc_timezone(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("Datetime,Open,High,Low,Close,Spread\n", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "data:\n  csv_path: data.csv\nlearning:\n  session_timezone: UTC\n  mt5_server_utc_offset_hours: 2\n",
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert Path(loaded.csv_path) == csv.resolve()
    bad = tmp_path / "bad.yaml"
    bad.write_text("learning:\n  session_timezone: Europe/Warsaw\n", encoding="utf-8")
    try:
        load_config(bad)
    except ValueError as exc:
        assert "UTC" in str(exc)
    else:
        raise AssertionError("non-UTC session contract must fail closed")


def test_v7_clone_isolated_from_source_policy_state() -> None:
    class Toy:
        def __init__(self) -> None:
            self.values = [1, 2, 3]

    source = Toy()
    before = policy_state_hash(source)
    clone = clone_policy(source)
    clone.values.append(4)
    assert policy_state_hash(source) == before


def test_v7_promotion_requires_complete_oos_seed_matrix() -> None:
    incomplete = promotion_stability_report(
        [
            {"validation_window": "w1", "seed": 1, "val_fitness": 1.0},
            {"validation_window": "w2", "seed": 2, "val_fitness": 2.0},
            {"validation_window": "w3", "seed": 3, "val_fitness": 3.0},
        ],
        min_windows=3,
        min_seeds=3,
    )
    assert incomplete["eligible"] is False
    assert "oos_seed_matrix_incomplete:3<9" in incomplete["reasons"]

    complete = promotion_stability_report(
        [
            {"validation_window": f"w{window}", "seed": seed, "val_fitness": float(window + seed)}
            for window in range(1, 4)
            for seed in range(1, 4)
        ],
        min_windows=3,
        min_seeds=3,
    )
    assert complete["eligible"] is True
    assert complete["distinct_pairs"] == 9


def test_v13_frozen_oos_gate_rejects_finite_but_ineligible_result() -> None:
    rejected = frozen_oos_promotion_gate(
        {
            "fitness_eligible": False,
            "fitness_reconciled": True,
            "total_trades": 80,
            "equity_drawdown_pct": 18.0,
            "fitness_policy_expectancy_lcb": -0.10,
            "fitness_context_negative_critical": 4,
            "feature_audit_valid": True,
            "real_tick_evidence_verified": True,
        }
    )
    assert rejected["eligible"] is False
    assert "fitness_ineligible" in rejected["reasons"]
    assert any(reason.startswith("oos_drawdown_above_limit") for reason in rejected["reasons"])


def test_v13_frozen_oos_gate_accepts_only_complete_evidence() -> None:
    accepted = frozen_oos_promotion_gate(
        {
            "fitness_eligible": True,
            "fitness_reconciled": True,
            "total_trades": 80,
            "equity_drawdown_pct": 8.0,
            "balance_drawdown_pct": 7.0,
            "fitness_policy_expectancy_lcb": 0.04,
            "fitness_context_negative_critical": 0,
            "feature_audit_valid": True,
            "real_tick_evidence_verified": True,
        }
    )
    assert accepted["eligible"] is True
    assert accepted["reasons"] == []


def test_paired_oos_gate_rejects_candidate_regression_against_seed() -> None:
    baseline = {
        "net_profit": 100.0,
        "equity_drawdown_pct": 6.0,
        "execution_summary": {"expectancy": 0.04, "expectancy_lcb": 0.02},
    }
    candidate = {
        "net_profit": 90.0,
        "equity_drawdown_pct": 8.0,
        "execution_summary": {"expectancy": -0.01, "expectancy_lcb": -0.03},
    }
    result = paired_oos_non_regression_gate(candidate, baseline)
    assert result["eligible"] is False
    assert "candidate_expectancy_below_floor:-0.01<0" in result["reasons"]
    assert any(reason.startswith("paired_profit_regression:") for reason in result["reasons"])


def test_paired_oos_gate_accepts_positive_candidate_with_small_variance() -> None:
    baseline = {
        "net_profit": 100.0,
        "equity_drawdown_pct": 6.0,
        "execution_summary": {"expectancy": 0.04, "expectancy_lcb": 0.02},
    }
    candidate = {
        "net_profit": 101.0,
        "equity_drawdown_pct": 6.5,
        "execution_summary": {"expectancy": 0.03, "expectancy_lcb": 0.01},
    }
    result = paired_oos_non_regression_gate(candidate, baseline)
    assert result["eligible"] is True
    assert result["reasons"] == []


def test_paired_oos_gate_prefers_current_window_and_fails_closed_on_missing_baseline() -> None:
    candidate = {
        "net_profit": 10.0,
        "equity_drawdown_pct": 2.0,
        "fitness_policy_expectancy_lcb": -0.9,
        "execution_summary": {"expectancy": 0.10, "expectancy_lcb": 0.05},
    }
    baseline = {
        "net_profit": 9.0,
        "equity_drawdown_pct": 2.0,
        "fitness_policy_expectancy_lcb": 0.9,
        "execution_summary": {"expectancy": 0.08, "expectancy_lcb": 0.04},
    }

    result = paired_oos_non_regression_gate(candidate, baseline)
    missing = paired_oos_non_regression_gate(candidate, {"net_profit": 9.0})

    assert result["eligible"] is True
    assert np.isclose(result["candidate_expectancy_lcb"], 0.05)
    assert "baseline_expectancy_lcb_missing" in missing["reasons"]
    assert "baseline_drawdown_missing" in missing["reasons"]


def test_v7_learning_integrity_keeps_memory_caps_after_compatibility() -> None:
    from self_growing_trader.self_growing_runner import SelfGrowingPolicy

    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=71)
    policy.configure_learning_integrity(
        LearningIntegrityConfig(
            experience_buffer_max=9,
            rejection_memory_max=7,
            execution_trace_cache_max=11,
            memory_neurons_max=13,
            memory_connections_max=17,
            mistake_episodes_max=19,
        )
    )
    policy.ensure_compatible()
    assert policy.experience_buffer_max == 9
    assert policy.rejection_memory_max == 7
    assert policy._execution_trace_cache_max == 11
    assert policy.memory_supernet.max_neurons == 13
    assert policy.memory_supernet.max_connections == 17
    assert policy.mistake_episode_store.max_episodes == 19
