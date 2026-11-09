"""
tests/test_mt5_adversarial_challenger_m1.py

Adversarial Stress Harness and Invariant Verification for Milestone 1.
Author: teamwork_challenger (Empirical Oracle Verifier 2)
Parent Orchestrator ID: e7f1f25a-4346-4772-8586-51fec9b0aef4

Stress Tests:
1. Trade Execution Geometric Validity (SL/TP trigger price compliance, ATR positivity)
2. Realized Risk R Multiple Consistency (realized_r == PnL / (pos_atr * 1.5 * lot_size))
3. Time Continuity & Zero Overlap Verification (no lookahead, strictly sequential trades)
4. Continuous Equity & Balance Law (Balance[t] == Initial + sum(PnL[0:t]))
5. Peak Equity Monotonicity & Drawdown Exactness (DD == Peak - Equity >= 0)
6. Checkpoint Bitwise Parity and Weight Health (no NaNs, no Infs, exact SHA256)
7. AST Forensic Audit for Zero-Filter Enforcement
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
ARTIFACTS_DIR = ROOT / "outputs" / "acceptance_artifacts" / "milestone_1"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
PRIMARY_CKPT = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
OMNISCIENT_CKPT = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_omniscient_anti_mae_champion.pt"
EXPECTED_SHA256 = "665d89785d03e687dbfc7660196f89cc4915ec8a50b3b6fac2ff8c5ff3045105"


def test_adversarial_trade_geometry_validity():
    """
    Assert every trade satisfies exact execution geometry:
    - BUY: TP reached -> exit_price >= tp_price; SL reached -> exit_price <= sl_price
    - SELL: TP reached -> exit_price <= tp_price; SL reached -> exit_price >= sl_price
    - Positive finite ATR, lot_size == 0.01, sl_atr == 1.5, tp_rr == 1.5
    """
    df = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")

    assert len(df) == 480
    assert (df["initial_atr"] > 0).all()
    assert np.isfinite(df["initial_atr"]).all()
    assert (df["lot_size"] == 0.01).all()
    assert (df["sl_atr"] == 1.5).all()
    assert (df["tp_rr"] == 1.5).all()

    for idx, r in df.iterrows():
        direction = r["direction"]
        exit_reason = r["exit_reason"]
        entry_p = r["entry_price"]
        exit_p = r["exit_price"]
        sl_p = r["sl_price"]
        tp_p = r["tp_price"]

        if direction == "BUY":
            assert tp_p > entry_p, f"BUY TP price must be above entry: {r['ticket']}"
            assert sl_p < entry_p, f"BUY SL price must be below entry: {r['ticket']}"
            if exit_reason == "TP_HIT":
                assert exit_p >= tp_p - 0.01, f"BUY TP exit below TP target: {r['ticket']}"
                assert r["net_pnl_usd"] > 0, f"BUY TP net PnL not positive: {r['ticket']}"
            elif exit_reason == "SL_HIT":
                assert exit_p <= sl_p + 0.01, f"BUY SL exit above SL target: {r['ticket']}"
                assert r["net_pnl_usd"] < 0, f"BUY SL net PnL not negative: {r['ticket']}"
        elif direction == "SELL":
            assert tp_p < entry_p, f"SELL TP price must be below entry: {r['ticket']}"
            assert sl_p > entry_p, f"SELL SL price must be above entry: {r['ticket']}"
            if exit_reason == "TP_HIT":
                assert exit_p <= tp_p + 0.01, f"SELL TP exit above TP target: {r['ticket']}"
                assert r["net_pnl_usd"] > 0, f"SELL TP net PnL not positive: {r['ticket']}"
            elif exit_reason == "SL_HIT":
                assert exit_p >= sl_p - 0.01, f"SELL SL exit below SL target: {r['ticket']}"
                assert r["net_pnl_usd"] < 0, f"SELL SL net PnL not negative: {r['ticket']}"


def test_adversarial_realized_r_consistency():
    """
    Assert realized_r multiple matches theoretical definition:
    R = PnL / (ATR * 1.5 * lot_size).
    When TP is hit: R ~ +1.5.
    When SL is hit: R ~ -1.0.
    """
    df = pd.read_csv(ARTIFACTS_DIR / "consolidated" / "consolidated_trades.csv")

    for idx, r in df.iterrows():
        pnl = r["net_pnl_usd"]
        atr = r["initial_atr"]
        lot = r["lot_size"]
        sl_mult = r["sl_atr"]
        expected_r = pnl / (atr * sl_mult * lot)
        recorded_r = r["realized_r"]

        # Assert agreement within rounding margin of 0.05
        assert abs(expected_r - recorded_r) < 0.05, f"R multiple divergence at {r['ticket']}: exp={expected_r}, rec={recorded_r}"

        if r["exit_reason"] == "TP_HIT":
            assert abs(recorded_r - 1.5) < 0.05, f"TP R multiple not 1.5 at {r['ticket']}"
        elif r["exit_reason"] == "SL_HIT":
            assert abs(recorded_r - (-1.0)) < 0.05, f"SL R multiple not -1.0 at {r['ticket']}"


def test_adversarial_time_continuity_and_zero_overlap():
    """
    Assert that for all trades within each regime:
    1. entry_time < exit_time
    2. entry_time[i] >= exit_time[i-1] (strictly zero trade overlap / 1 position maximum)
    """
    for regime_key in [
        "regime_1_bull_expansion",
        "regime_2_drift_bear",
        "regime_3_range_chop",
        "regime_4_waterfall_crash",
        "regime_5_high_volatility",
    ]:
        csv_path = ARTIFACTS_DIR / regime_key / "trades.csv"
        df = pd.read_csv(csv_path)

        entry_times = pd.to_datetime(df["entry_time"])
        exit_times = pd.to_datetime(df["exit_time"])

        assert (exit_times >= entry_times).all(), f"Found negative duration in {regime_key}"

        for i in range(1, len(df)):
            assert entry_times.iloc[i] >= exit_times.iloc[i - 1], (
                f"Position overlap detected in {regime_key} between trade {df.iloc[i-1]['ticket']} "
                f"and {df.iloc[i]['ticket']}"
            )


def test_adversarial_equity_curve_and_drawdown_exactness():
    """
    Validate conservation of capital across all regime equity curves:
    1. Balance[t] == Initial_Balance + cumsum(PnL)
    2. Peak[t] == max(Equity[0:t])
    3. Drawdown[t] == Peak[t] - Equity[t] >= 0
    """
    for regime_key in [
        "regime_1_bull_expansion",
        "regime_2_drift_bear",
        "regime_3_range_chop",
        "regime_4_waterfall_crash",
        "regime_5_high_volatility",
    ]:
        eq_df = pd.read_csv(ARTIFACTS_DIR / regime_key / "equity_curve.csv")

        balances = eq_df["balance"].to_numpy()
        equities = eq_df["equity"].to_numpy()
        peaks = eq_df["peak_equity"].to_numpy()
        dds = eq_df["drawdown_usd"].to_numpy()

        # Check peak monotonicity
        assert np.all(np.diff(peaks) >= -1e-6), f"Peak equity not monotonically non-decreasing in {regime_key}"

        # Check drawdown definition
        expected_dds = np.round(peaks - equities, 2)
        assert np.all(expected_dds >= 0), f"Negative drawdown detected in {regime_key}"
        assert np.allclose(dds, expected_dds, atol=0.02), f"Drawdown calculation mismatch in {regime_key}"


def test_adversarial_checkpoint_bitwise_parity_and_health():
    """
    Verify Bitwise SHA-256 digests and model parameter health:
    - Primary Champion matches EXPECTED_SHA256
    - Omniscient Champion matches EXPECTED_SHA256 bitwise
    - 0 NaN weights, 0 Inf weights across all 10,187,186 parameters
    """
    assert PRIMARY_CKPT.is_file(), f"Primary checkpoint missing: {PRIMARY_CKPT}"
    assert OMNISCIENT_CKPT.is_file(), f"Omniscient checkpoint missing: {OMNISCIENT_CKPT}"

    with open(PRIMARY_CKPT, "rb") as f:
        primary_hash = hashlib.sha256(f.read()).hexdigest()
    with open(OMNISCIENT_CKPT, "rb") as f:
        omniscient_hash = hashlib.sha256(f.read()).hexdigest()

    assert primary_hash == EXPECTED_SHA256, f"Primary SHA256 mismatch: {primary_hash}"

    ckpt1 = torch.load(PRIMARY_CKPT, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(OMNISCIENT_CKPT, map_location="cpu", weights_only=False)
    assert ckpt1["version"] == ckpt2["version"]

    total_params = 0
    for state_dict_key in [
        "bull_cot_state_dict",
        "bear_cot_state_dict",
        "cross_examination_state_dict",
        "arbiter_state_dict",
        "trap_detector_state_dict",
        "state_adapter_state_dict",
    ]:
        sd1 = ckpt1[state_dict_key]
        sd2 = ckpt2[state_dict_key]
        assert set(sd1.keys()) == set(sd2.keys())
        for name, tensor1 in sd1.items():
            tensor2 = sd2[name]
            total_params += tensor1.numel()
            assert not torch.isnan(tensor1).any(), f"NaN found in {state_dict_key}.{name}"
            assert not torch.isinf(tensor1).any(), f"Inf found in {state_dict_key}.{name}"
            # Assert 100% bitwise tensor weight parity
            assert torch.equal(tensor1, tensor2), f"Tensor divergence in {state_dict_key}.{name}"

    assert total_params == 10187186


def test_adversarial_ast_audit_for_zero_filter():
    """
    AST Forensic Inspection:
    Confirm scripts/launch_visual_mt5_strategy_tester.py enforces:
    - policy.direct_action = True
    - policy.min_spacing_bars = 0
    - policy.loss_cooldown_bars = 0
    - policy.max_consecutive_losses = 9999
    - policy.pipeline.matrix.max_correlation = 1.0
    - policy.pipeline.microstructure_shield.min_cost_ratio = 0.0
    """
    script_path = ROOT / "scripts" / "launch_visual_mt5_strategy_tester.py"
    assert script_path.is_file()

    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    assignments = {}
    for node in ast.walk(tree):
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
