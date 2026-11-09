from __future__ import annotations

import pandas as pd

from scripts.finalize_seven_choice_oos_gate import _analyse_ledger, _first_existing


def test_oos_analysis_aggregates_partial_close_costs_and_r_by_position(tmp_path) -> None:
    ledger = tmp_path / "execution.tsv"
    rows = [
        # The entry establishes the action and immutable risk denominator.
        {
            "transaction_type": "DEAL_ADD",
            "source": "policy",
            "action_name": "MARKET_BUY",
            "filled": "1",
            "closed": "0",
            "position_final_close": "0",
            "position_ticket": "11",
            "initial_risk": "100",
        },
        # Two closing deals must form one lifecycle and include every cost.
        {
            "transaction_type": "DEAL_ADD",
            "source": "policy",
            "action_name": "MANAGE_POSITION",
            "filled": "0",
            "closed": "1",
            "position_final_close": "0",
            "position_ticket": "11",
            "profit": "30",
            "commission": "-2",
            "swap": "-1",
            "initial_risk": "100",
        },
        {
            "transaction_type": "DEAL_ADD",
            "source": "unknown",
            "action_name": "SYSTEM",
            "filled": "0",
            "closed": "1",
            "position_final_close": "1",
            "position_ticket": "11",
            "profit": "-10",
            "commission": "-1",
            "swap": "0",
            "initial_risk": "100",
        },
        {
            "transaction_type": "DEAL_ADD",
            "source": "policy",
            "action_name": "SELL_LIMIT",
            "filled": "1",
            "closed": "0",
            "position_final_close": "0",
            "position_ticket": "12",
            "initial_risk": "50",
        },
        {
            "transaction_type": "DEAL_ADD",
            "source": "unknown",
            "action_name": "SYSTEM",
            "filled": "0",
            "closed": "1",
            "position_final_close": "1",
            "position_ticket": "12",
            "profit": "5",
            "commission": "0",
            "swap": "0",
            "initial_risk": "50",
        },
    ]
    pd.DataFrame(rows).to_csv(ledger, sep="\t", index=False)

    result = _analyse_ledger(ledger)

    # Position 11 is (30-2-1) + (-10-1) = 16, or 0.16R.
    assert result["position_net_pnl"] == 21.0
    assert result["position_mean_pnl"] == 10.5
    assert result["position_mean_r"] == 0.13
    assert result["risk_coverage"] == 1.0
    assert result["position_r_denominator"] == "initial_risk_max_per_position"
    assert result["position_results_by_entry_action"]["MARKET_BUY"]["net_pnl"] == 16.0
    assert result["position_results_by_entry_action"]["MARKET_BUY"]["mean_r"] == 0.16
    assert result["position_results_by_entry_action"]["SELL_LIMIT"]["mean_r"] == 0.10


def test_oos_analysis_does_not_overwrite_entry_action_on_filled_non_entry_row(tmp_path) -> None:
    ledger = tmp_path / "execution.tsv"
    rows = [
        {
            "transaction_type": "DEAL_ADD",
            "source": "policy",
            "action_name": "BUY_LIMIT",
            "filled": "1",
            "closed": "0",
            "position_final_close": "0",
            "position_ticket": "7",
            "initial_risk": "10",
        },
        {
            "transaction_type": "DEAL_ADD",
            "source": "policy",
            "action_name": "MARKET_SELL",
            "filled": "1",
            "closed": "1",
            "position_final_close": "1",
            "deal_entry": "1",
            "position_ticket": "7",
            "profit": "2",
            "commission": "0",
            "swap": "0",
            "initial_risk": "10",
        },
    ]
    pd.DataFrame(rows).to_csv(ledger, sep="\t", index=False)

    result = _analyse_ledger(ledger)

    assert result["filled_entry_counts"] == {"BUY_LIMIT": 1}
    assert result["position_results_by_entry_action"]["BUY_LIMIT"]["mean_r"] == 0.2
    assert "MARKET_SELL" not in result["position_results_by_entry_action"]


def test_gate_accepts_new_frozen_root_storage_layout(tmp_path) -> None:
    runtime = tmp_path / "frozen_policy" / "ablation_frozen_policy.runtime"
    frozen = runtime.parent
    runtime.mkdir(parents=True)
    retained = frozen / "mt5_execution_ledger.tsv"
    retained.write_text("evidence", encoding="utf-8")
    legacy = runtime / "execution_ledger.csv"
    assert _first_existing((legacy, retained)) == retained
