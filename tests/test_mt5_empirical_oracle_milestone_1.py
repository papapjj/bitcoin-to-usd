"""
tests/test_mt5_empirical_oracle_milestone_1.py

Independent Naive Oracle and Stress Harness for Milestone 1 Acceptance Telemetry.
Author: teamwork_challenger (Empirical Oracle Verifier 2)
Parent Orchestrator ID: e7f1f25a-4346-4772-8586-51fec9b0aef4

Verifies:
1. Re-computes all telemetry metrics directly from ground-truth CSV data:
   - Consolidated trades: outputs/acceptance_artifacts/milestone_1/consolidated/consolidated_trades.csv
   - 5 individual regime trade logs: outputs/acceptance_artifacts/milestone_1/regime_*/trades.csv
   - 5 individual regime equity curves: outputs/acceptance_artifacts/milestone_1/regime_*/equity_curve.csv
2. Independent naive mathematical oracle calculations for:
   - Total Trades, Wins, Losses, Win Rate %
   - Buy Trades, Buy Wins, Buy Win Rate %
   - Sell Trades, Sell Wins, Sell Win Rate %
   - Gross Profit, Gross Loss, Profit Factor (both raw contract execution and rounded net PnL)
   - Net PnL USD
   - Max Drawdown USD from high-resolution bar-level equity curves
   - Average MAE in ATR
3. 100% Agreement against manifest.json within precision/rounding bounds
4. Strict evaluation against Milestone 1 Acceptance Criteria:
   - Win Rate: 96.25% >= 70.0%
   - Profit Factor: 34.78 >= 2.0+
   - Stop Loss: 1.50 ATR within [1.2, 1.5] ATR
   - Average MAE: 0.4634 ATR <= 0.60 ATR
   - Strict Zero-Filter Invariant enforced
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "outputs" / "acceptance_artifacts" / "milestone_1"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

REGIME_KEYS = [
    "regime_1_bull_expansion",
    "regime_2_drift_bear",
    "regime_3_range_chop",
    "regime_4_waterfall_crash",
    "regime_5_high_volatility",
]


def naive_oracle_trade_metrics(trades_df: pd.DataFrame, use_raw_price_pnl: bool = True) -> Dict[str, Any]:
    """
    Independent naive mathematical oracle calculating trade metrics directly
    from raw trade records without relying on worker helper functions.

    If use_raw_price_pnl is True, computes PnL from (exit_price - entry_price) * lot_size
    which eliminates per-trade rounding distortion.
    """
    n_trades = len(trades_df)
    if n_trades == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "buy_trades": 0,
            "buy_wins": 0,
            "buy_win_rate_pct": 0.0,
            "sell_trades": 0,
            "sell_wins": 0,
            "sell_win_rate_pct": 0.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
            "net_pnl_usd": 0.0,
            "profit_factor": 0.0,
            "average_mae_atr": 0.0,
        }

    dir_series = trades_df["direction"].to_numpy()
    mae_series = trades_df["mae_atr"].to_numpy(dtype=np.float64)

    if use_raw_price_pnl:
        entry_prices = trades_df["entry_price"].to_numpy(dtype=np.float64)
        exit_prices = trades_df["exit_price"].to_numpy(dtype=np.float64)
        lot_sizes = trades_df["lot_size"].to_numpy(dtype=np.float64)
        pnl_series = np.where(
            dir_series == "BUY",
            (exit_prices - entry_prices) * lot_sizes,
            (entry_prices - exit_prices) * lot_sizes,
        )
    else:
        pnl_series = trades_df["net_pnl_usd"].to_numpy(dtype=np.float64)

    wins_mask = pnl_series > 0
    losses_mask = pnl_series < 0

    wins = int(np.sum(wins_mask))
    losses = int(np.sum(losses_mask))
    wr = (wins / n_trades) * 100.0

    buy_mask = dir_series == "BUY"
    sell_mask = dir_series == "SELL"

    buy_trades = int(np.sum(buy_mask))
    buy_wins = int(np.sum(buy_mask & wins_mask))
    buy_wr = (buy_wins / buy_trades * 100.0) if buy_trades > 0 else 0.0

    sell_trades = int(np.sum(sell_mask))
    sell_wins = int(np.sum(sell_mask & wins_mask))
    sell_wr = (sell_wins / sell_trades * 100.0) if sell_trades > 0 else 0.0

    gross_profit = float(np.sum(pnl_series[wins_mask]))
    gross_loss = float(np.sum(np.abs(pnl_series[losses_mask])))
    net_pnl = float(np.sum(pnl_series))

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = gross_profit / 1e-4

    avg_mae = float(np.mean(mae_series))

    return {
        "total_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wr, 2),
        "buy_trades": buy_trades,
        "buy_wins": buy_wins,
        "buy_win_rate_pct": round(buy_wr, 2),
        "sell_trades": sell_trades,
        "sell_wins": sell_wins,
        "sell_win_rate_pct": round(sell_wr, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "profit_factor": round(pf, 2),
        "average_mae_atr": round(avg_mae, 4),
    }


def naive_oracle_max_drawdown(equity_df: pd.DataFrame) -> float:
    """
    Independent naive mathematical oracle calculating peak-to-trough max drawdown
    from equity time series.
    """
    if len(equity_df) == 0:
        return 0.0
    equity = equity_df["equity"].to_numpy(dtype=np.float64)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = float(np.max(drawdowns))
    return round(max_dd, 2)


# =============================================================================
# TEST CASES
# =============================================================================


def test_manifest_existence_and_schema():
    """Verify manifest.json exists, is valid JSON, and adheres to the schema."""
    assert MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["milestone"] == "Milestone 1"
    assert manifest["overall_verdict"] == "PASSED"
    assert "consolidated_metrics" in manifest
    assert "regime_metrics" in manifest
    assert "acceptance_criteria" in manifest
    assert "zero_filter_invariant" in manifest


def test_independent_oracle_on_consolidated_trades():
    """
    Read consolidated_trades.csv and compare independent oracle recomputation
    against manifest.json consolidated metrics.
    """
    consolidated_csv = ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv"
    assert consolidated_csv.is_file(), f"Consolidated trades CSV missing: {consolidated_csv}"

    trades_df = pd.read_csv(consolidated_csv)
    
    # 1. Oracle recomputation with unrounded raw contract price PnL
    oracle_raw = naive_oracle_trade_metrics(trades_df, use_raw_price_pnl=True)
    # 2. Oracle recomputation from recorded net_pnl_usd column
    oracle_col = naive_oracle_trade_metrics(trades_df, use_raw_price_pnl=False)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    cons = manifest["consolidated_metrics"]

    # Total trades: exactly 480
    assert oracle_raw["total_trades"] == 480
    assert oracle_raw["total_trades"] == cons["total_trades"]

    # Wins and losses: exactly 462 wins, 18 losses
    assert oracle_raw["wins"] == 462
    assert oracle_raw["wins"] == cons["wins"]
    assert oracle_raw["losses"] == 18
    assert oracle_raw["losses"] == cons["losses"]

    # Win Rate: exactly 96.25%
    assert oracle_raw["win_rate_pct"] == 96.25
    assert oracle_raw["win_rate_pct"] == cons["win_rate_pct"]

    # Directional breakdown
    assert oracle_raw["buy_trades"] == 228
    assert oracle_raw["buy_wins"] == 221
    assert oracle_raw["buy_win_rate_pct"] == 96.93
    assert oracle_raw["sell_trades"] == 252
    assert oracle_raw["sell_wins"] == 241
    assert oracle_raw["sell_win_rate_pct"] == 95.63

    # Profit Factor: 34.78 in manifest and raw oracle, 34.77 in column oracle (due to cent rounding)
    assert oracle_raw["profit_factor"] == 34.78
    assert oracle_raw["profit_factor"] == cons["profit_factor"]
    assert abs(oracle_col["profit_factor"] - cons["profit_factor"]) <= 0.02

    # Financial sums
    assert oracle_raw["gross_profit_usd"] == 599.78
    assert oracle_raw["gross_profit_usd"] == cons["gross_profit_usd"]
    assert oracle_raw["gross_loss_usd"] == 17.24
    assert oracle_raw["gross_loss_usd"] == cons["gross_loss_usd"]
    assert abs(oracle_raw["net_pnl_usd"] - cons["net_pnl_usd"]) <= 0.01

    # Average MAE: exactly 0.4634 ATR
    assert oracle_raw["average_mae_atr"] == 0.4634
    assert oracle_raw["average_mae_atr"] == cons["average_mae_atr"]


def test_independent_oracle_on_all_5_regimes():
    """
    Independently re-compute metrics for each of the 5 regimes from raw CSVs
    and assert exact agreement with manifest.json regime records.
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for regime_key in REGIME_KEYS:
        reg_dir = ARTIFACTS_DIR / regime_key
        trades_csv = reg_dir / "trades.csv"
        equity_csv = reg_dir / "equity_curve.csv"

        assert trades_csv.is_file(), f"Regime trades CSV missing: {trades_csv}"
        assert equity_csv.is_file(), f"Regime equity CSV missing: {equity_csv}"

        trades_df = pd.read_csv(trades_csv)
        equity_df = pd.read_csv(equity_csv)

        oracle_metrics = naive_oracle_trade_metrics(trades_df, use_raw_price_pnl=True)
        oracle_max_dd = naive_oracle_max_drawdown(equity_df)

        expected = manifest["regime_metrics"][regime_key]

        assert oracle_metrics["total_trades"] == expected["total_trades"], f"{regime_key} total trades mismatch"
        assert oracle_metrics["wins"] == expected["wins"], f"{regime_key} wins mismatch"
        assert oracle_metrics["losses"] == expected["losses"], f"{regime_key} losses mismatch"
        assert abs(oracle_metrics["win_rate_pct"] - expected["win_rate_pct"]) < 0.05, f"{regime_key} WR mismatch"
        assert abs(oracle_metrics["buy_win_rate_pct"] - expected["buy_win_rate_pct"]) < 0.1, f"{regime_key} Buy WR mismatch"
        assert abs(oracle_metrics["sell_win_rate_pct"] - expected["sell_win_rate_pct"]) < 0.1, f"{regime_key} Sell WR mismatch"
        assert abs(oracle_metrics["average_mae_atr"] - expected["average_mae_atr"]) < 0.001, f"{regime_key} MAE mismatch"
        assert abs(oracle_max_dd - expected["max_drawdown_usd"]) < 0.05, f"{regime_key} Max DD mismatch"

        # Check all trades have sl_atr == 1.5, tp_rr == 1.5, trailing_atr == 0.5
        assert (trades_df["sl_atr"] == 1.5).all(), f"{regime_key} SL ATR non-1.5 detected"
        assert (trades_df["tp_rr"] == 1.5).all(), f"{regime_key} TP RR non-1.5 detected"
        assert (trades_df["trailing_atr"] == 0.5).all(), f"{regime_key} Trailing ATR non-0.5 detected"


def test_acceptance_criteria_thresholds_independent_check():
    """
    Rigorously compare verified metrics against project acceptance thresholds.
    """
    consolidated_csv = ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv"
    trades_df = pd.read_csv(consolidated_csv)
    metrics = naive_oracle_trade_metrics(trades_df, use_raw_price_pnl=True)

    # 1. Win Rate >= 70%
    assert metrics["win_rate_pct"] >= 70.0, f"Win rate {metrics['win_rate_pct']}% < 70.0%"

    # 2. Profit Factor >= 2.0+
    assert metrics["profit_factor"] >= 2.0, f"Profit factor {metrics['profit_factor']} < 2.0"

    # 3. Stop Loss bounded tightly within [1.2, 1.5] ATR
    sl_values = trades_df["sl_atr"].unique()
    assert len(sl_values) == 1 and sl_values[0] == 1.50
    assert 1.2 <= sl_values[0] <= 1.5

    # 4. Average MAE <= 0.60 ATR
    assert metrics["average_mae_atr"] <= 0.60, f"Average MAE {metrics['average_mae_atr']} > 0.60 ATR"


def test_zero_filter_invariant_in_artifacts():
    """
    Verify all trades in artifacts were triggered by pure neural actions with zero heuristic overrides.
    """
    consolidated_csv = ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv"
    trades_df = pd.read_csv(consolidated_csv)

    # All trades must have valid direction and PRM >= 0.58
    assert (trades_df["direction"].isin(["BUY", "SELL"])).all()
    assert (trades_df["prm_score"] >= 0.58).all(), "Found trade with PRM < 0.58"

    # Verify exit reasons are strictly Stop Loss or Take Profit
    assert (trades_df["exit_reason"].isin(["TP_HIT", "SL_HIT"])).all()
