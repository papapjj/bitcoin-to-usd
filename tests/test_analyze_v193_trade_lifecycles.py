from __future__ import annotations

import pandas as pd

from scripts.analyze_v193_trade_lifecycles import analyze


def test_duplicate_final_close_rows_form_one_lifecycle_and_keep_deal_reasons(tmp_path) -> None:
    ledger = tmp_path / "execution.csv"
    geometry = "1.0|1|2.5|2.0|5.0|0|0|0|0|learned_factorized_atr_v2|2.5|5.0|2.0|0|0"
    rows = [
        {
            "position_ticket": 22,
            "filled": 1,
            "closed": 0,
            "position_final_close": 0,
            "action_name": "MARKET_BUY",
            "logged_at": "2026.09.15 10:00:00",
            "geometry_entry_pipe": geometry,
            "profit": 0.0,
            "commission": 0.0,
            "swap": 0.0,
            "close_reason": "",
            "exit_kind": "",
            "age_bars": 0,
            "initial_risk": 100.0,
        },
        {
            "position_ticket": 22,
            "filled": 0,
            "closed": 1,
            "position_final_close": 1,
            "action_name": "SYSTEM",
            "logged_at": "2026.09.15 10:05:00",
            "geometry_entry_pipe": "",
            "profit": 20.0,
            "commission": -1.0,
            "swap": 0.0,
            "close_reason": "EXPERT",
            "exit_kind": "POLICY_CLOSE",
            "age_bars": 5,
            "initial_risk": 100.0,
        },
        {
            "position_ticket": 22,
            "filled": 0,
            "closed": 1,
            "position_final_close": 1,
            "action_name": "SYSTEM",
            "logged_at": "2026.09.15 10:05:00",
            "geometry_entry_pipe": "",
            "profit": -40.0,
            "commission": -1.0,
            "swap": 0.0,
            "close_reason": "SL",
            "exit_kind": "HARD_STOP_LOSS",
            "age_bars": 5,
            "initial_risk": 100.0,
        },
        {
            "position_ticket": 23,
            "filled": 1,
            "closed": 0,
            "position_final_close": 0,
            "action_name": "SELL_LIMIT",
            "logged_at": "2026.09.15 11:00:00",
            "geometry_entry_pipe": geometry.replace("|1|", "|-1|", 1),
            "profit": 0.0,
            "commission": 0.0,
            "swap": 0.0,
            "close_reason": "",
            "exit_kind": "",
            "age_bars": 0,
            "initial_risk": 100.0,
        },
        {
            "position_ticket": 23,
            "filled": 0,
            "closed": 1,
            "position_final_close": 1,
            "action_name": "SYSTEM",
            "logged_at": "2026.09.15 11:08:00",
            "geometry_entry_pipe": "",
            "profit": 30.0,
            "commission": -1.0,
            "swap": 0.0,
            "close_reason": "TP",
            "exit_kind": "TAKE_PROFIT",
            "age_bars": 8,
            "initial_risk": 100.0,
        },
    ]
    pd.DataFrame(rows).to_csv(ledger, sep="\t", index=False)

    result = analyze(ledger)

    assert result["schema_version"] == 2
    assert result["closed_deal_rows"] == 3
    assert result["final_close_rows"] == 3
    assert result["unique_completed_positions"] == 2
    assert result["duplicate_final_close_rows"] == 1
    assert result["overall"]["positions"] == 2
    assert result["overall"]["net_pnl"] == 7.0

    position = next(row for row in result["positions"] if row["position_ticket"] == "22")
    assert position["close_reason"] == "EXPERT+SL"
    assert position["close_reasons"] == ["EXPERT", "SL"]
    assert position["final_close_reason"] == "EXPERT+SL"
    assert position["final_close_rows"] == 2
    assert position["close_deals"] == 2
    assert position["net_pnl"] == -22.0

    assert result["by_lifecycle_close_reason"]["EXPERT+SL"]["positions"] == 1
    assert result["by_lifecycle_close_reason"]["EXPERT+SL"]["net_pnl"] == -22.0
    assert result["by_deal_close_reason"]["EXPERT"]["deals"] == 1
    assert result["by_deal_close_reason"]["EXPERT"]["net_pnl"] == 19.0
    assert result["by_deal_close_reason"]["SL"]["net_pnl"] == -41.0
    assert result["by_deal_close_reason"]["TP"]["net_pnl"] == 29.0
