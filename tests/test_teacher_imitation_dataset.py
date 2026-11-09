from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from self_growing_trader.teacher_imitation_dataset import (
    CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION,
    TEACHER_FEATURE_COUNT,
    TeacherDatasetBuildConfig,
    build_teacher_imitation_dataset,
)


FIELDS = [
    "schema_version", "sample_id", "sample_time", "label_finalized_time", "agent_id",
    "action", "candidate_action", "teacher_trade_accepted", "is_pivot",
    "pivot_kind", "pivot_price", "pivot_prominence_atr", "causal_input",
    "future_derived_label", "future_high_return_atr", "future_low_return_atr",
    "future_close_return_atr", "pivot_stop_buffer_atr", "opposite_pivot_found",
    "opposite_pivot_time", "opposite_pivot_kind", "opposite_pivot_price",
    "entry_bid", "entry_ask",
    "entry_price", "entry_spread_price", "initial_risk_price", "stop_loss_price",
    "opposite_exit_price", "exit_time", "exit_bar_offset", "exit_price",
    "exit_spread_price", "exit_reason", "ambiguous_hit", "teacher_gross_pnl_price",
    "teacher_spread_cost_price", "teacher_net_pnl_price", "teacher_gross_r",
    "teacher_spread_cost_r", "teacher_net_r", "teacher_mfe_r", "teacher_mae_r",
    "confidence", "inputs_pipe",
]


def _features(seed: float) -> str:
    return "|".join(str(seed + index / 1000.0) for index in range(TEACHER_FEATURE_COUNT))


def _row(sample_id: str, minute: int, agent: int, action: int, prominence: float = 0.0) -> dict[str, object]:
    candidate_action = action
    is_trade = action != 0
    entry_bid = 100.0
    entry_ask = 101.0
    entry_price = entry_ask if action == 1 else entry_bid
    stop_loss = 98.0 if action == 1 else 103.0 if action == 2 else 0.0
    pivot_price = 98.5 if action == 1 else 102.5 if action == 2 else 0.0
    opposite_price = 110.0 if action == 1 else 90.0 if action == 2 else 0.0
    take_profit = 110.0 if action == 1 else 91.0 if action == 2 else 0.0
    exit_price = take_profit if is_trade else entry_price
    return {
        "schema_version": 4,
        "sample_id": sample_id,
        "sample_time": f"2026.01.01 00:{minute:02d}:00",
        "label_finalized_time": f"2026.01.01 00:{minute + 48:02d}:00",
        "agent_id": agent,
        "action": action,
        "candidate_action": candidate_action,
        "teacher_trade_accepted": int(is_trade),
        "is_pivot": int(is_trade),
        "pivot_kind": "bottom" if action == 1 else "top" if action == 2 else "none",
        "pivot_price": pivot_price,
        "pivot_prominence_atr": prominence,
        "causal_input": 1,
        "future_derived_label": 1,
        "future_high_return_atr": 2.0,
        "future_low_return_atr": -1.0,
        "future_close_return_atr": 0.5,
        "pivot_stop_buffer_atr": 0.25,
        "opposite_pivot_found": int(is_trade),
        "opposite_pivot_time": f"2026.01.01 00:{minute + 12:02d}:00",
        "opposite_pivot_kind": (
            "historical_swing_high" if action == 1
            else "historical_swing_low" if action == 2 else "none"
        ),
        "opposite_pivot_price": opposite_price,
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "entry_price": entry_price,
        "entry_spread_price": 1.0,
        "initial_risk_price": 3.0 if is_trade else 0.0,
        "stop_loss_price": stop_loss,
        "opposite_exit_price": take_profit,
        "exit_time": f"2026.01.01 00:{minute + 12:02d}:00",
        "exit_bar_offset": 12,
        "exit_price": exit_price,
        "exit_spread_price": 1.0,
        "exit_reason": "OPPOSITE_PIVOT" if is_trade else "NO_TRADE",
        "ambiguous_hit": 0,
        "teacher_gross_pnl_price": 10.0 if is_trade else 0.0,
        "teacher_spread_cost_price": 1.0 if is_trade else 0.0,
        "teacher_net_pnl_price": 9.0 if is_trade else 0.0,
        "teacher_gross_r": 10.0 / 3.0 if is_trade else 0.0,
        "teacher_spread_cost_r": 1.0 / 3.0 if is_trade else 0.0,
        "teacher_net_r": 3.0 if is_trade else 0.0,
        "teacher_mfe_r": 3.0 if is_trade else 0.0,
        "teacher_mae_r": -0.5 if is_trade else 0.0,
        "confidence": 0.8,
        "inputs_pipe": _features(float(minute)),
    }


def _write(path: Path, rows: list[dict[str, object]], delimiter: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_builder_reads_future_labels_but_keeps_only_causal_observation(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher.npz"
    _write(labels, [_row("buy", 1, 0, 1, 1.5), _row("sell", 2, 1, 2, 2.0), _row("hold", 3, 0, 0)])
    result = build_teacher_imitation_dataset(labels, output)
    assert result.action_counts == {0: 1, 1: 1, 2: 1}
    with np.load(output) as data:
        assert data["observations"].shape == (3, TEACHER_FEATURE_COUNT)
        assert data["actions"].tolist() == [1, 2, 0]
        assert data["candidate_actions"].tolist() == [1, 2, 0]
        assert data["exit_reasons"].tolist() == ["OPPOSITE_PIVOT", "OPPOSITE_PIVOT", "NO_TRADE"]
        assert np.allclose(data["teacher_spread_cost_price"], [1.0, 1.0, 0.0])
        assert np.allclose(data["teacher_net_pnl_price"], [9.0, 9.0, 0.0])
        assert np.all(data["label_finalized_times"] > data["sample_times"])
        assert np.allclose(data["quality_weights"], [2.0, 2.0, 1.0])
        assert data["geometry_target_valid"].tolist() == [True, True, False]
        assert np.allclose(data["teacher_sl_atr"], [1.5, 1.5, 0.0])
        assert data["teacher_sl_indices"].tolist() == [2, 2, 0]
        assert data["teacher_tp_indices"].tolist() == [7, 7, 0]
        assert data["teacher_delay_indices"].tolist() == [0, 0, 0]
        assert data["teacher_trailing_indices"].tolist() == [0, 0, 0]


def test_builder_rejects_wrong_authority_and_non_delayed_label(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher.npz"
    wrong_owner = _row("wrong", 1, 1, 1, 1.5)
    invalid_delay = _row("delay", 2, 0, 1, 1.5)
    invalid_delay["label_finalized_time"] = invalid_delay["sample_time"]
    _write(labels, [wrong_owner, invalid_delay, _row("good", 3, 0, 1, 1.2)])
    result = build_teacher_imitation_dataset(labels, output)
    assert result.sample_count == 1
    assert result.skipped_counts == {"authority_violation": 1, "invalid_label_delay": 1}


def test_builder_retains_all_holds_in_chronological_stream_and_reads_mql_tabs(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher.npz"
    rows = [_row(f"hold-{index}", index + 1, index % 2, 0) for index in range(8)]
    rows.append(_row("buy", 10, 0, 1, 1.0))
    _write(labels, rows, delimiter="\t")
    result = build_teacher_imitation_dataset(labels, output)
    assert result.action_counts == {0: 8, 1: 1}
    with np.load(output) as data:
        assert data["schema_version"].tolist() == [7]
        assert data["sample_times"].tolist() == sorted(data["sample_times"].tolist())


def test_builder_moves_pivot_action_to_causal_confirmation_and_reprices_spread(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher-confirmed.npz"
    rows = [
        _row("pivot-buy", 1, 0, 1, 1.5),
        _row("hold-2", 2, 0, 0),
        _row("hold-3", 3, 0, 0),
        _row("confirm-4", 4, 0, 0),
        _row("hold-5", 5, 0, 0),
    ]
    rows[3]["entry_bid"] = 103.0
    rows[3]["entry_ask"] = 104.0
    _write(labels, rows)
    result = build_teacher_imitation_dataset(
        labels,
        output,
        config=TeacherDatasetBuildConfig(confirmation_bars=3),
    )
    assert result.action_counts == {0: 4, 1: 1}
    with np.load(output) as data:
        assert data["schema_version"].tolist() == [CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION]
        assert data["actions"].tolist() == [0, 0, 0, 1, 0]
        assert data["candidate_actions"].tolist() == [0, 0, 0, 1, 0]
        assert data["teacher_trade_accepted"].tolist() == [False, False, False, True, False]
        assert data["label_origin_times"][3] == data["sample_times"][0]
        assert data["confirmation_delay_minutes"].tolist() == [0, 0, 0, 3, 0]
        assert np.isclose(data["teacher_gross_pnl_price"][3], 7.0)
        assert np.isclose(data["teacher_spread_cost_price"][3], 1.0)
        assert np.isclose(data["teacher_net_pnl_price"][3], 6.0)
        assert data["geometry_target_valid"].tolist() == [False, False, False, True, False]
        assert np.isclose(data["teacher_sl_atr"][3], 3.0)
        assert data["teacher_sl_indices"][3] == 5
        assert np.isclose(data["teacher_tp_rr"][3], 1.0)
        assert data["teacher_tp_indices"][3] == 3
        assert np.isclose(data["opposite_exit_prices"][3], 110.0)
    manifest = __import__("json").loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["confirmation_audit"]["shifted_positive"] == 1


def test_builder_turns_confirmation_with_nonpositive_net_pnl_into_hard_negative(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher-confirmed-negative.npz"
    rows = [
        _row("pivot-buy", 1, 0, 1, 1.5),
        _row("confirm-4", 4, 0, 0),
        _row("hold-5", 5, 0, 0),
    ]
    rows[1]["entry_bid"] = 110.0
    rows[1]["entry_ask"] = 111.0
    _write(labels, rows)
    result = build_teacher_imitation_dataset(
        labels,
        output,
        config=TeacherDatasetBuildConfig(confirmation_bars=3),
    )
    assert result.action_counts == {0: 3}
    with np.load(output) as data:
        assert data["actions"].tolist() == [0, 0, 0]
        assert data["candidate_actions"].tolist() == [0, 1, 0]
        assert not np.any(data["teacher_trade_accepted"])
        assert data["teacher_net_pnl_price"][1] < 0.0
    manifest = __import__("json").loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["confirmation_audit"]["shifted_nonpositive_after_spread"] == 1


def test_builder_keeps_rejected_pivot_as_hard_negative_hold(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher.npz"
    rejected = _row("rejected", 1, 0, 1, 1.5)
    rejected.update({
        "action": 0,
        "teacher_trade_accepted": 0,
        "exit_reason": "SL",
        "exit_price": rejected["stop_loss_price"],
        "teacher_gross_pnl_price": -2.0,
        "teacher_spread_cost_price": 1.0,
        "teacher_net_pnl_price": -3.0,
        "teacher_gross_r": -2.0 / 3.0,
        "teacher_spread_cost_r": 1.0 / 3.0,
        "teacher_net_r": -1.0,
    })
    _write(labels, [rejected, _row("buy", 2, 0, 1, 1.0)])
    result = build_teacher_imitation_dataset(labels, output)
    assert result.action_counts == {0: 1, 1: 1}
    with np.load(output) as data:
        assert data["candidate_actions"].tolist() == [1, 1]
        assert data["teacher_trade_accepted"].tolist() == [False, True]
        assert data["quality_weights"][0] > 1.0


def test_builder_accepts_symbol_precision_rounding_in_price_identity(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher.npz"
    rejected = _row("rounded-sl", 1, 1, 2, 1.5)
    rejected.update({
        "action": 0,
        "teacher_trade_accepted": 0,
        "exit_reason": "SL",
        "exit_price": rejected["stop_loss_price"],
        "teacher_gross_pnl_price": "-39.32",
        "teacher_spread_cost_price": "28.80",
        "teacher_net_pnl_price": "-68.13",
        "teacher_gross_r": "-0.57724771",
        "teacher_spread_cost_r": "0.42275229",
        "teacher_net_r": "-1.00000000",
    })
    _write(labels, [rejected, _row("sell", 2, 1, 2, 1.0)])
    result = build_teacher_imitation_dataset(labels, output)
    assert result.action_counts == {0: 1, 2: 1}
    assert result.skipped_counts == {}


def test_builder_rejects_fake_accepted_trade_without_opposite_pivot_or_positive_net_pnl(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    output = tmp_path / "teacher.npz"
    no_opposite = _row("no-opposite", 1, 0, 1, 1.5)
    no_opposite["opposite_pivot_found"] = 0
    no_opposite["exit_reason"] = "NO_OPPOSITE_PIVOT"
    negative_net = _row("negative-net", 2, 1, 2, 1.5)
    negative_net.update({
        "teacher_gross_pnl_price": 0.0,
        "teacher_spread_cost_price": 1.0,
        "teacher_net_pnl_price": -1.0,
        "teacher_gross_r": 0.0,
        "teacher_spread_cost_r": 1.0 / 3.0,
        "teacher_net_r": -1.0 / 3.0,
    })
    _write(labels, [no_opposite, negative_net, _row("good", 3, 0, 1, 1.2)])
    result = build_teacher_imitation_dataset(labels, output)
    assert result.sample_count == 1
    assert result.skipped_counts == {"invalid_teacher_acceptance": 2}


def test_ea_is_historical_labeler_without_order_or_lifecycle_path() -> None:
    source = Path("mt5_v2_runner/MQL5/Experts/BottomTop_Teacher_EA.mq5").read_text(encoding="utf-8")
    assert "historical_opposite_pivot_net_pnl_v4" in source
    assert "future_derived_label" in source
    assert "ComputeFlatHistoricalInputs" in source
    assert "EvaluateHistoricalPivot" in source
    assert "SimulateTeacherOutcome" in source
    assert "entryAsk" in source
    assert "stopLossPrice" in source
    assert "oppositeExitPrice" in source
    assert "oppositePivotFound" in source
    assert "netPnlPrice" in source
    assert "spreadCostR" in source
    assert "bar pivot doi dien cung cham SL" in source
    assert "TeacherTakeProfitAtr" not in source
    assert "MaxTeacherSpreadCostR" not in source
    assert "ACTION_MARKET_BUY 1" in source
    assert "ACTION_MARKET_SELL 2" in source
    assert "CTrade" not in source
    assert "OrderSend" not in source
    assert "PositionClose" not in source
    assert "LifecycleDatasetFile" not in source
    assert "MQL_VISUAL_MODE" in source
    assert "OBJ_ARROW" in source
    assert "OBJ_TREND" in source
    assert "BUY ENTRY" in source
    assert "SELL ENTRY" in source
    assert "TP/EXIT at opposite pivot (not fixed RR TP)" in source
    assert "ShowTradeGeometry" in source
    assert "DrawChartLegend();" in source
    assert "Teacher visual objects accepted_trades=" in source
    assert "input double PivotStopBufferAtr = 1.50;" in source
    assert "TP tai pivot doi dien" in source
    assert "DrawHistoricalLabel(target, action, candidateAction, prominenceAtr, outcome)" in source
    assert "if(!FeatureBuffersReady())" in source
    assert "WaitForFeatureBuffers(10000)" not in source
    assert "NormalizeDouble(outcome.netPnlPrice, _Digits) > 0.0" in source


def test_counterfactual_exporter_uses_live_decision_input_parity() -> None:
    source = Path(
        "mt5_v2_runner/MQL5/Experts/BottomTop_Counterfactual_Exporter_EA.mq5"
    ).read_text(encoding="utf-8")
    assert "TimeToStruct(TimeCurrent(), barParts)" in source
    assert "double liveSpreadPrice = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;" in source
    assert "iSpread(_Symbol, PERIOD_M1, 1)" not in source
    assert source.count("SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)") >= 2
