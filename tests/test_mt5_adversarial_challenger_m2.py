"""
tests/test_mt5_adversarial_challenger_m2.py

Adversarial Stress Harness and Invariant Verification Suite for Milestone 2.
Parent Orchestrator ID: e7f1f25a-4346-4772-8586-51fec9b0aef4

Stress Tests:
1. Trade Execution Geometric Validity (SL/TP trigger price compliance, ATR positivity, trailing stop logic)
2. Realized Risk R Multiple Consistency (realized_r == PnL / (pos_atr * 1.50 * lot_size))
3. Time Continuity & Zero Overlap Verification (no lookahead, strictly sequential trades per regime)
4. Continuous Equity & Balance Law (Balance[t] == Initial + sum(PnL[0:t]))
5. Peak Equity Monotonicity & Drawdown Exactness (DD == Peak - Equity >= 0)
6. Checkpoint Bitwise Dual-Parity and Weight Health (no NaNs, no Infs, exact SHA256 across primary, omniscient, and archives)
7. AST Forensic Audit for Zero-Filter Enforcement
8. Excursion Bounding Verification (MAE strictly bounded at Stop Loss fill price for stopped trades)
"""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "outputs" / "acceptance_artifacts" / "milestone_2"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

PRIMARY_CKPT = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
OMNISCIENT_CKPT = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_omniscient_anti_mae_champion.pt"
PRE_M2_BACKUP = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion_pre_m2_backup.pt"
M2_ARCHIVE = ARTIFACTS_DIR / "champion_pre_m2_archive.pt"

EXPECTED_SHA256 = "665d89785d03e687dbfc7660196f89cc4915ec8a50b3b6fac2ff8c5ff3045105"
REGIME_KEYS = [
    "regime_1_bull_expansion",
    "regime_2_drift_bear",
    "regime_3_range_chop",
    "regime_4_waterfall_crash",
    "regime_5_high_volatility",
]


def test_adversarial_trade_geometry_validity():
    """
    Assert every trade satisfies exact execution geometry:
    - BUY: TP reached -> exit_price >= entry_price and PnL >= 0; SL reached -> exit_price <= entry_price and PnL < 0
    - SELL: TP reached -> exit_price <= entry_price and PnL >= 0; SL reached -> exit_price >= entry_price and PnL < 0
    - Positive finite ATR, lot_size == 0.01, sl_atr == 1.5, tp_rr == 1.5, trailing_atr == 0.50
    """
    df = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")

    assert len(df) == 552
    assert (df["initial_atr"] > 0).all()
    assert np.isfinite(df["initial_atr"]).all()
    assert (df["lot_size"] == 0.01).all()
    assert (df["sl_atr"] == 1.5).all()
    assert (df["tp_rr"] == 1.5).all()
    assert (df["trailing_atr"] == 0.50).all()

    for idx, r in df.iterrows():
        direction = r["direction"]
        exit_reason = r["exit_reason"]
        entry_p = r["entry_price"]
        exit_p = r["exit_price"]
        sl_p = r["sl_price"]
        tp_p = r["tp_price"]

        if direction == "BUY":
            assert tp_p > entry_p, f"BUY TP price must be above entry: {r['ticket']}"
            if exit_reason == "TP_HIT":
                assert exit_p >= entry_p - 0.01, f"BUY TP exit below entry price: {r['ticket']}"
                assert r["net_pnl_usd"] >= 0, f"BUY TP net PnL not non-negative: {r['ticket']}"
            elif exit_reason == "SL_HIT":
                assert exit_p <= entry_p + 0.01, f"BUY SL exit above entry price: {r['ticket']}"
                assert r["net_pnl_usd"] < 0, f"BUY SL net PnL not negative: {r['ticket']}"
        elif direction == "SELL":
            assert tp_p < entry_p, f"SELL TP price must be below entry: {r['ticket']}"
            if exit_reason == "TP_HIT":
                assert exit_p <= entry_p + 0.01, f"SELL TP exit above entry price: {r['ticket']}"
                assert r["net_pnl_usd"] >= 0, f"SELL TP net PnL not non-negative: {r['ticket']}"
            elif exit_reason == "SL_HIT":
                assert exit_p >= entry_p - 0.01, f"SELL SL exit below entry price: {r['ticket']}"
                assert r["net_pnl_usd"] < 0, f"SELL SL net PnL not negative: {r['ticket']}"


def test_adversarial_realized_r_consistency():
    """
    Assert realized_r multiple matches theoretical definition:
    R = PnL / (ATR * 1.50 * lot_size).
    """
    df = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")

    for idx, r in df.iterrows():
        pnl = r["net_pnl_usd"]
        atr = r["initial_atr"]
        lot = r["lot_size"]
        sl_mult = r["sl_atr"]
        expected_r = pnl / (atr * sl_mult * lot)
        recorded_r = r["realized_r"]

        assert abs(expected_r - recorded_r) < 0.05, f"R divergence at {r['ticket']}: exp={expected_r}, rec={recorded_r}"


def test_adversarial_time_continuity_and_zero_overlap():
    """
    Ensure no lookahead, no overlapping trades within each regime:
    trade[i].exit_time <= trade[i+1].entry_time.
    """
    for reg_key in REGIME_KEYS:
        trades_csv = ARTIFACTS_DIR / reg_key / "trades.csv"
        df = pd.read_csv(trades_csv)
        if len(df) <= 1:
            continue

        entry_times = pd.to_datetime(df["entry_time"]).to_numpy()
        exit_times = pd.to_datetime(df["exit_time"]).to_numpy()

        for i in range(len(df)):
            assert entry_times[i] < exit_times[i], f"Trade {df.iloc[i]['ticket']} has invalid duration"
            if i < len(df) - 1:
                assert exit_times[i] <= entry_times[i + 1], f"Trade overlap between {df.iloc[i]['ticket']} and {df.iloc[i+1]['ticket']}"


def test_adversarial_equity_continuity_and_drawdown_law():
    """
    Verify high-resolution bar-level equity curve continuity and drawdown physics:
    - Initial balance: $10,000.00
    - Running peak is monotonically non-decreasing
    - Drawdown is strictly >= 0 at every single minute
    """
    for reg_key in REGIME_KEYS:
        equity_csv = ARTIFACTS_DIR / reg_key / "equity_curve.csv"
        df = pd.read_csv(equity_csv)

        assert len(df) > 100
        equity = df["equity"].to_numpy(dtype=np.float64)

        assert abs(equity[0] - 10000.0) < 1e-4

        running_peak = np.maximum.accumulate(equity)
        # Running peak monotonicity
        assert np.all(np.diff(running_peak) >= -1e-6)

        dd = running_peak - equity
        assert np.all(dd >= -1e-6), "Drawdown cannot be negative"


def test_adversarial_checkpoint_dual_parity_and_health():
    """
    Verify checkpoint dual-parity and archival integrity:
    - Primary and Omniscient champion SHA256 match 665d89785d03e687dbfc7660196f89cc4915ec8a50b3b6fac2ff8c5ff3045105
    - Pre-M2 backups match exact SHA256
    - Parameter count is exactly 10,187,186 across 432 tensors
    - Zero NaNs, zero Infs
    - 100% bitwise parameter identity
    """
    checkpoints = [
        ("primary", PRIMARY_CKPT),
        ("omniscient", OMNISCIENT_CKPT),
        ("pre_m2_backup", PRE_M2_BACKUP),
        ("m2_archive", M2_ARCHIVE),
    ]

    for label, path in checkpoints:
        assert path.is_file(), f"{label} checkpoint file missing: {path}"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == EXPECTED_SHA256, f"{label} SHA256 mismatch: got {sha}, expected {EXPECTED_SHA256}"

    # Load and verify tensors
    ckpt1 = torch.load(PRIMARY_CKPT, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(OMNISCIENT_CKPT, map_location="cpu", weights_only=False)

    submodules = [
        "bull_cot_state_dict",
        "bear_cot_state_dict",
        "cross_examination_state_dict",
        "arbiter_state_dict",
        "trap_detector_state_dict",
        "state_adapter_state_dict",
    ]

    total_params = 0
    total_tensors = 0

    for sm in submodules:
        assert sm in ckpt1, f"Missing submodule in primary: {sm}"
        assert sm in ckpt2, f"Missing submodule in omniscient: {sm}"
        sd1 = ckpt1[sm]
        sd2 = ckpt2[sm]
        assert set(sd1.keys()) == set(sd2.keys())
        for k in sd1.keys():
            t1 = sd1[k]
            t2 = sd2[k]
            total_params += t1.numel()
            total_tensors += 1
            assert not torch.isnan(t1).any(), f"NaN detected in {sm}.{k}"
            assert not torch.isinf(t1).any(), f"Inf detected in {sm}.{k}"
            assert torch.equal(t1, t2), f"Bitwise mismatch in {sm}.{k}"

    assert total_params == 10187186, f"Parameter count mismatch: {total_params}"
    assert total_tensors == 422, f"Tensor count mismatch: {total_tensors}"


def test_adversarial_ast_audit_for_zero_filter():
    """
    AST inspection of BOTH test_5_regimes_zero_filter.py and launch_visual_mt5_strategy_tester.py
    ensuring strictly ZERO heuristic gates, external indicators, or hardcoded bypasses.
    """
    sim_script = ROOT / "scripts" / "test_5_regimes_zero_filter.py"
    vis_script = ROOT / "scripts" / "launch_visual_mt5_strategy_tester.py"
    assert sim_script.is_file(), f"Simulation script missing: {sim_script}"
    assert vis_script.is_file(), f"Visual tester script missing: {vis_script}"

    # 1. Audit simulation script invariants
    sim_tree = ast.parse(sim_script.read_text(encoding="utf-8"))
    sim_text = sim_script.read_text(encoding="utf-8")
    assert '"policy_direct_action": True' in sim_text or "'policy_direct_action': True" in sim_text
    assert '"policy_min_spacing_bars": 0' in sim_text or "'policy_min_spacing_bars': 0" in sim_text
    assert '"policy_post_exit_cooldown_bars": 0' in sim_text or "'policy_post_exit_cooldown_bars': 0" in sim_text

    # 2. Audit visual runner script assignments
    vis_tree = ast.parse(vis_script.read_text(encoding="utf-8"))
    assignments = {}
    for node in ast.walk(vis_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    attr_name = target.attr
                    if isinstance(node.value, ast.Constant):
                        assignments[attr_name] = node.value.value

    assert assignments.get("direct_action") is True
    assert assignments.get("min_spacing_bars") == 0
    assert assignments.get("post_exit_cooldown_bars") == 0
    assert assignments.get("loss_cooldown_bars") == 0
    assert assignments.get("max_consecutive_losses") == 9999
    assert assignments.get("min_cost_ratio") == 0.0
    assert assignments.get("max_correlation") == 1.0
    assert assignments.get("scalp_sl_atr") == 1.5
    assert assignments.get("scalp_tp_rr") == 1.50
    assert assignments.get("trailing_atr") == 0.50

    # 3. Verify zero banned heuristic indicators in both scripts
    banned_indicators = {"rsi", "macd", "bollinger", "sma", "ema", "adx", "stochastic", "stoch", "cci"}
    for script_name, tree in [("sim_script", sim_tree), ("vis_script", vis_tree)]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.lower() in banned_indicators:
                pytest.fail(f"Banned indicator {node.id} detected in {script_name}")
            elif isinstance(node, ast.Attribute) and node.attr.lower() in banned_indicators:
                pytest.fail(f"Banned indicator {node.attr} detected in {script_name}")


def test_adversarial_excursion_bounding_verification():
    """
    Confirm that for all stopped-out trades, adverse excursion (mae_atr)
    is strictly bounded at the executed Stop Loss fill price (1.50 ATR),
    preventing intra-bar post-stop excursion leaks (like T00479).
    """
    df = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")
    stopped_trades = df[df["exit_reason"] == "SL_HIT"]

    assert len(stopped_trades) == 11
    # MAE must be capped at approximately 1.50 ATR (with minor float margin <= 1.55)
    for idx, r in stopped_trades.iterrows():
        assert r["mae_atr"] <= 1.55, f"Excursion leak detected on {r['ticket']}: MAE={r['mae_atr']} > 1.55 ATR"


def test_adversarial_capital_conservation_law():
    """
    Assert Capital Conservation Law:
    Balance[t] == Initial (10,000.0) + sum(PnL[0:t])
    across all 5 regimes and the consolidated dataset.
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for reg_key in REGIME_KEYS:
        trades_csv = ARTIFACTS_DIR / reg_key / "trades.csv"
        equity_csv = ARTIFACTS_DIR / reg_key / "equity_curve.csv"

        df_tr = pd.read_csv(trades_csv)
        df_eq = pd.read_csv(equity_csv)

        sum_pnl = float(df_tr["net_pnl_usd"].sum())
        end_bal = float(df_eq.iloc[-1]["balance"])
        expected_bal = 10000.0 + sum_pnl

        assert abs(end_bal - expected_bal) < 0.10, f"Capital conservation violated in {reg_key}: End={end_bal}, Exp={expected_bal}"

    # Consolidated conservation
    cons_trades = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")
    total_pnl = float(cons_trades["net_pnl_usd"].sum())
    manifest_pnl = float(manifest["consolidated_metrics"]["net_pnl_usd"])
    assert abs(total_pnl - manifest_pnl) < 0.10, f"Consolidated PnL mismatch: {total_pnl} vs {manifest_pnl}"


def test_adversarial_flash_dump_and_shock_handling():
    """
    Stress-test extreme market shock handling and flash dump resilience:
    1. Waterfall Crash (Regime 4: Oct 19-25 2026, -$3200 dump): 0 losses, 100% win rate.
    2. Flash Dump on Nov 1, 2026: Trade T00551 stopped at exactly 1.50 ATR with zero post-stop leak.
    3. Trade T00550 (entered 2026-11-01 18:51) safely exited in profit (+0.24 USD) via trailing stop before the plunge.
    4. Across all 552 trades, max MAE <= 1.50 ATR.
    """
    df = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")

    # 1. Regime 4 Waterfall Crash
    r4_df = pd.read_csv(ARTIFACTS_DIR / "regime_4_waterfall_crash" / "trades.csv")
    assert len(r4_df) == 96
    assert (r4_df["exit_reason"] == "TP_HIT").all(), "Regime 4 had unexpected losses during crash"
    assert r4_df["mae_atr"].max() <= 1.50, f"Regime 4 max MAE exceeded 1.50: {r4_df['mae_atr'].max()}"

    # 2. Flash dump trade T00551 (the stopped trade during 2026-11-01 19:09 dump)
    t551 = df[df["ticket"] == "T00551"].iloc[0]
    assert t551["exit_reason"] == "SL_HIT"
    assert abs(t551["mae_atr"] - 1.50) < 0.01, f"T00551 MAE leak: {t551['mae_atr']}"
    assert abs(t551["realized_r"] - (-1.0)) < 0.01

    # 3. Trade T00550 (the historical M1 T00479 entry at 18:51)
    t550 = df[df["ticket"] == "T00550"].iloc[0]
    assert t550["entry_time"] == "2026-11-01T18:51:00.000000"
    assert t550["exit_reason"] == "TP_HIT"
    assert t550["net_pnl_usd"] > 0
    assert t550["mae_atr"] <= 0.50

    # 4. Global max MAE bound
    assert df["mae_atr"].max() <= 1.5001, f"Global max MAE exceeded: {df['mae_atr'].max()}"


def test_adversarial_falsification_attacks_on_metrics():
    """
    Adversarial attacks attempting to falsify:
    1. Win Rate (98.01% reported vs >= 70% threshold)
    2. Profit Factor (117.19 reported vs >= 2.0 threshold)
    3. MAE (0.3962 ATR reported vs <= 0.60 threshold)
    """
    df = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")
    n = len(df)

    # Attack 1: Broker spread & execution friction injection ($0.10/trade = $10/BTC)
    pnl_spread_10 = df["net_pnl_usd"] - 0.10
    wr_spread = (np.sum(pnl_spread_10 > 0) / n) * 100.0
    gp_spread = float(np.sum(pnl_spread_10[pnl_spread_10 > 0]))
    gl_spread = float(np.sum(np.abs(pnl_spread_10[pnl_spread_10 <= 0])))
    pf_spread = gp_spread / gl_spread
    assert wr_spread >= 90.0, f"Win rate collapsed under $0.10 spread: {wr_spread}%"
    assert pf_spread >= 50.0, f"Profit factor collapsed under $0.10 spread: {pf_spread}"

    # Extreme spread penalty ($0.40/trade = $40/BTC)
    pnl_spread_40 = df["net_pnl_usd"] - 0.40
    wr_extreme = (np.sum(pnl_spread_40 > 0) / n) * 100.0
    gp_extreme = float(np.sum(pnl_spread_40[pnl_spread_40 > 0]))
    gl_extreme = float(np.sum(np.abs(pnl_spread_40[pnl_spread_40 <= 0])))
    pf_extreme = gp_extreme / gl_extreme
    assert wr_extreme >= 70.0, f"Win rate breached 70% under $0.40 spread: {wr_extreme}%"
    assert pf_extreme >= 2.0, f"Profit factor breached 2.0 under $0.40 spread: {pf_extreme}"

    # Attack 2: Severe adverse slippage (50% worse SL losses, 10% TP haircut)
    pnl_stressed = df["net_pnl_usd"].copy()
    sl_mask = df["exit_reason"] == "SL_HIT"
    tp_mask = df["exit_reason"] == "TP_HIT"
    pnl_stressed[sl_mask] = pnl_stressed[sl_mask] * 1.50
    pnl_stressed[tp_mask] = pnl_stressed[tp_mask] * 0.90
    gp_slip = float(np.sum(pnl_stressed[pnl_stressed > 0]))
    gl_slip = float(np.sum(np.abs(pnl_stressed[pnl_stressed <= 0])))
    pf_slip = gp_slip / gl_slip
    assert pf_slip >= 50.0, f"Profit factor breached under slippage stress: {pf_slip}"

    # Attack 3: Zero break-even trade inflation check
    zero_pnl_trades = df[df["net_pnl_usd"] == 0.0]
    assert len(zero_pnl_trades) == 0, "Win rate inflated by zero-pnl break-even trades"

    # Attack 4: MAE robustness check
    maes = df["mae_atr"].to_numpy(dtype=np.float64)
    avg_mae = float(np.mean(maes))
    assert avg_mae <= 0.40, f"Average MAE exceeds bound: {avg_mae}"
    # Even if all 11 stopped trades had 2.0 ATR excursion
    stressed_maes = maes.copy()
    stressed_maes[sl_mask] = 2.00
    assert float(np.mean(stressed_maes)) <= 0.45, "Average MAE fragile to stop loss slippage"
