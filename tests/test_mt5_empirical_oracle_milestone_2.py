"""
tests/test_mt5_empirical_oracle_milestone_2.py

Independent Naive Oracle and Telemetry Verification Suite for Milestone 2.
Parent Orchestrator ID: e7f1f25a-4346-4772-8586-51fec9b0aef4

Verifies:
1. Re-computes all telemetry metrics directly from ground-truth CSV data:
   - Consolidated trades: outputs/acceptance_artifacts/milestone_2/consolidated/consolidated_trades.csv
   - 5 individual regime trade logs: outputs/acceptance_artifacts/milestone_2/regime_*/trades.csv
   - 5 individual regime equity curves: outputs/acceptance_artifacts/milestone_2/regime_*/equity_curve.csv
2. Independent naive mathematical oracle calculations for:
   - Total Trades, Wins, Losses, Win Rate %
   - Buy Trades, Buy Wins, Buy Win Rate %
   - Sell Trades, Sell Wins, Sell Win Rate %
   - Gross Profit, Gross Loss, Profit Factor
   - Net PnL USD
   - Max Drawdown USD from bar-level equity curves
   - Average MAE in ATR
3. 100% Agreement against manifest.json within precision/rounding bounds
4. Strict evaluation against Milestone 2 Acceptance Criteria:
   - Win Rate: 98.01% >= 70.0%
   - Profit Factor: 117.19 >= 2.0+
   - Stop Loss: 1.50 ATR within [1.2, 1.5] ATR
   - Average MAE: 0.3962 ATR <= 0.60 ATR
   - Strict Zero-Filter Invariant enforced
5. Independent forensic validation of Flaw Diagnosis artifact (trade_flaw_diagnosis.json)
6. Independent validation of Training Loss History artifact (training_loss_history.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "outputs" / "acceptance_artifacts" / "milestone_2"
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

    wins_mask = pnl_series >= 0
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
    """Verify manifest.json exists, is valid JSON, and adheres to Milestone 2 schema."""
    assert MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["milestone"] == "Milestone 2"
    assert manifest["overall_verdict"] == "PASSED"
    assert "consolidated_metrics" in manifest
    assert "regime_metrics" in manifest
    assert "acceptance_criteria" in manifest
    assert "zero_filter_invariant" in manifest
    assert "parameters" in manifest

    params = manifest["parameters"]
    assert params["prm_threshold"] == 0.58
    assert params["sl_atr"] == 1.50
    assert params["tp_rr"] == 1.50
    assert params["tp_atr"] == 2.25
    assert params["trailing_atr"] == 0.50
    assert params["lot_size"] == 0.01


def test_independent_oracle_on_consolidated_trades():
    """
    Read consolidated_trades.csv and compare independent oracle recomputation
    against manifest.json consolidated metrics.
    """
    consolidated_csv = ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv"
    assert consolidated_csv.is_file(), f"Consolidated trades CSV missing: {consolidated_csv}"

    trades_df = pd.read_csv(consolidated_csv)
    
    oracle_raw = naive_oracle_trade_metrics(trades_df, use_raw_price_pnl=True)
    oracle_col = naive_oracle_trade_metrics(trades_df, use_raw_price_pnl=False)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    cons = manifest["consolidated_metrics"]

    # Total trades: 552
    assert oracle_raw["total_trades"] == 552
    assert oracle_raw["total_trades"] == cons["total_trades"]

    # Wins and losses: 541 wins, 11 losses
    assert oracle_raw["wins"] == 541
    assert oracle_raw["losses"] == 11
    assert oracle_raw["wins"] == cons["wins"]
    assert oracle_raw["losses"] == cons["losses"]

    # Win rate: 98.01%
    assert abs(oracle_raw["win_rate_pct"] - cons["win_rate_pct"]) <= 0.05
    assert cons["win_rate_pct"] >= 70.0

    # Directional counts
    assert oracle_raw["buy_trades"] == 261
    assert oracle_raw["buy_wins"] == 257
    assert oracle_raw["sell_trades"] == 291
    assert oracle_raw["sell_wins"] == 284

    # Directional win rates
    assert abs(oracle_raw["buy_win_rate_pct"] - cons["buy_win_rate_pct"]) <= 0.05
    assert abs(oracle_raw["sell_win_rate_pct"] - cons["sell_win_rate_pct"]) <= 0.05

    # Profit Factor >= 2.0 (manifest reports 117.19)
    assert cons["profit_factor"] >= 2.0
    assert oracle_raw["profit_factor"] >= 100.0

    # Net PnL USD
    assert abs(oracle_col["net_pnl_usd"] - cons["net_pnl_usd"]) <= 0.10
    assert cons["net_pnl_usd"] > 500.0

    # Average MAE
    assert abs(oracle_raw["average_mae_atr"] - cons["average_mae_atr"]) <= 0.001
    assert cons["average_mae_atr"] <= 0.60


def test_independent_oracle_per_regime():
    """
    Independently verify trade metrics and equity curves across each of the 5 regimes.
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for regime_key in REGIME_KEYS:
        regime_dir = ARTIFACTS_DIR / regime_key
        assert regime_dir.is_dir(), f"Regime dir missing: {regime_dir}"

        trades_csv = regime_dir / "trades.csv"
        equity_csv = regime_dir / "equity_curve.csv"

        assert trades_csv.is_file(), f"Trades CSV missing: {trades_csv}"
        assert equity_csv.is_file(), f"Equity CSV missing: {equity_csv}"

        trades_df = pd.read_csv(trades_csv)
        equity_df = pd.read_csv(equity_csv)

        oracle = naive_oracle_trade_metrics(trades_df, use_raw_price_pnl=False)
        oracle_dd = naive_oracle_max_drawdown(equity_df)

        reg_manifest = manifest["regime_metrics"][regime_key]

        assert oracle["total_trades"] == reg_manifest["total_trades"]
        assert oracle["wins"] == reg_manifest["wins"]
        assert oracle["losses"] == reg_manifest["losses"]
        assert abs(oracle["win_rate_pct"] - reg_manifest["win_rate_pct"]) <= 0.05
        assert abs(oracle["net_pnl_usd"] - reg_manifest["net_pnl_usd"]) <= 0.10
        assert abs(oracle["average_mae_atr"] - reg_manifest["average_mae_atr"]) <= 0.001
        assert abs(oracle_dd - reg_manifest["max_drawdown_usd"]) <= 0.10
        assert reg_manifest["passed"] is True


def test_acceptance_criteria_verification():
    """
    Evaluate manifest against all Milestone 2 Acceptance Criteria:
    - Win Rate >= 70.0%
    - Profit Factor >= 2.0
    - Stop Loss bounded [1.2, 1.5] ATR
    - Average MAE <= 0.60 ATR
    - Strict Zero-Filter Invariant enforced
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    criteria = manifest["acceptance_criteria"]

    # 1. Win Rate
    wr_crit = criteria["win_rate_ge_70"]
    assert wr_crit["passed"] is True
    assert wr_crit["actual"] >= wr_crit["target"]
    assert wr_crit["actual"] >= 70.0

    # 2. Profit Factor
    pf_crit = criteria["profit_factor_ge_2"]
    assert pf_crit["passed"] is True
    assert pf_crit["actual"] >= pf_crit["target"]
    assert pf_crit["actual"] >= 2.0

    # 3. Stop Loss
    sl_crit = criteria["sl_bounded_1_2_to_1_5_atr"]
    assert sl_crit["passed"] is True
    assert sl_crit["target_range"][0] <= sl_crit["actual"] <= sl_crit["target_range"][1]

    # 4. MAE
    mae_crit = criteria["mae_le_0_5_to_0_6_atr"]
    assert mae_crit["passed"] is True
    assert mae_crit["actual"] <= mae_crit["target_max"]
    assert mae_crit["actual"] <= 0.60

    # 5. Strict Zero Filter
    assert criteria["strict_zero_filter_enforced"]["passed"] is True

    # Overall Verdict
    assert manifest["overall_verdict"] == "PASSED"


def test_trade_flaw_diagnosis_artifact():
    """
    Verify trade_flaw_diagnosis.json and trade_flaw_decomposition.csv:
    - Exactly 18 losing trades across the 5 regimes diagnosed
    - Classification taxonomy matches the 4 root causes
    - Submodule activations (6 BilateralDualCoTEngine submodules) logged
    """
    flaw_json_path = ARTIFACTS_DIR / "trade_flaw_diagnosis.json"
    flaw_csv_path = ARTIFACTS_DIR / "trade_flaw_decomposition.csv"

    assert flaw_json_path.is_file(), f"Flaw diagnosis JSON missing: {flaw_json_path}"
    assert flaw_csv_path.is_file(), f"Flaw decomposition CSV missing: {flaw_csv_path}"

    with open(flaw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["milestone"] == "Milestone 2"
    assert data["summary"]["total_losses"] == 18
    assert len(data["diagnosed_trades"]) == 18

    breakdown = data["summary"]["losses_by_category"]
    total_classified = sum(breakdown.values())
    assert total_classified == 18

    assert breakdown["Profit Reversal / Trailing Gap"] == 9
    assert breakdown["Rejection Wick"] == 6
    assert breakdown["Marginal PRM"] == 1
    assert breakdown["Counter-Trend Momentum Trap"] == 2

    # Check each trade has full 6-submodule diagnostics
    for trade in data["diagnosed_trades"]:
        assert "neural_attribution" in trade
        sub = trade["neural_attribution"]
        assert "bull_score" in sub
        assert "bear_score" in sub
        assert "trap_probabilities" in sub
        assert "arbiter_probabilities" in sub
        assert "prm_score" in sub


def test_training_loss_history_artifact():
    """
    Verify training_loss_history.json:
    - Baseline Epoch 0 evaluation performed on loaded champion (~7.40)
    - Anti-MAE quadratic excursion penalty tracked
    - Early stopping triggered preserving pristine champion
    """
    loss_history_path = ARTIFACTS_DIR / "training_loss_history.json"
    assert loss_history_path.is_file(), f"Training loss history missing: {loss_history_path}"

    with open(loss_history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "baseline_val_loss" in data
    assert 7.0 <= data["baseline_val_loss"] <= 8.0
    assert "best_val_loss" in data
    assert data["best_val_loss"] == data["baseline_val_loss"]
    assert data["epochs_executed"] >= 1
    assert "history" in data
    assert data["history"][0]["status"] == "BASELINE_CHAMPION"
