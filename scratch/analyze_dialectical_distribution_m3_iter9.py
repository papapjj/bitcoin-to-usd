import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd
import json
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from scripts.test_5_regimes_zero_filter import process_regime_csv

WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

print("=" * 80)
print("PHASE 1: EXTRACTING WEEK 3 DATA & FORWARD PASS OF BILATERAL DUAL-COT")
print("=" * 80)

data = extract_features_from_csv(WEEK3_CSV)
ingress = torch.from_numpy(data["ingress"]).float()
w3_data = process_regime_csv(WEEK3_CSV)
closes = w3_data["closes"]
highs = w3_data["highs"]
lows = w3_data["lows"]
atrs = w3_data["atrs"]
N = min(len(closes), len(ingress))
print(f"Loaded {N} aligned M1 bars for 2026.10.19 - 2026.10.25 BTCUSDm.")

cfg = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.32,
    trap_threshold=0.60,
    margin_threshold=0.12,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

with torch.no_grad():
    v = engine.forward_batch(ingress[:N], batch_size=256)

b_scores = v.bull_scores.cpu().numpy()
r_scores = v.bear_scores.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
trap_probs = v.trap_probs.cpu().numpy()
# Empirical median centering as in adversarial_dialectical_reasoning.py
margins = (b_scores - 0.0218) - (r_scores - 0.2275)

print("\n" + "=" * 80)
print("PHASE 2: EMPIRICAL DISTRIBUTION OF DIALECTICAL METRICS")
print("=" * 80)

def print_stats(name, arr):
    p = np.percentile(arr, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    print(f"--- {name} (N={len(arr)}) ---")
    print(f"Mean: {np.mean(arr):.4f} | Std: {np.std(arr):.4f}")
    print(f"Min: {p[0]:.4f} | 1%: {p[1]:.4f} | 5%: {p[2]:.4f} | 10%: {p[3]:.4f} | 25%: {p[4]:.4f}")
    print(f"Median (50%): {p[5]:.4f}")
    print(f"75%: {p[6]:.4f} | 90%: {p[7]:.4f} | 95%: {p[8]:.4f} | 99%: {p[9]:.4f} | Max: {p[10]:.4f}")

print_stats("Bull Score (b_score)", b_scores)
print_stats("Bear Score (r_score)", r_scores)
print_stats("Dialectical Margin: (b - 0.0218) - (r - 0.2275)", margins)
print_stats("PRM Score", prm_scores)

print("\n" + "=" * 80)
print("PHASE 3: RAW SIGNAL COUNTS ACROSS MARGIN & PRM THRESHOLDS")
print("=" * 80)

thresholds = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.15]
prm_gates = [0.32, 0.35, 0.40]

print(f"{'PRM':<6} | {'Margin Thresh':<14} | {'BUY signals':<12} | {'SELL signals':<12} | {'Total Signals':<14} | {'BUY/SELL Ratio':<14}")
print("-" * 80)
for p_g in prm_gates:
    prm_mask = prm_scores >= p_g
    for th in thresholds:
        n_buy = np.sum(prm_mask & (margins >= th))
        n_sell = np.sum(prm_mask & (margins <= -th))
        tot = n_buy + n_sell
        ratio = (n_buy / max(1, n_sell))
        print(f"{p_g:<6.2f} | {th:<14.2f} | {n_buy:<12d} | {n_sell:<12d} | {tot:<14d} | {ratio:<14.2f}")
    print("-" * 80)

print("\n" + "=" * 80)
print("PHASE 4: HIGH-FIDELITY MT5 STRATEGY TESTER EXECUTION SIMULATION")
print("=" * 80)

# Simulate MT5 execution engine with exact parameters
# Spread: 640 points ($6.40 USD)
# SL: 2.50 ATR
# TP: 2.00 RR (5.00 ATR)
# Trailing: Activation 2.20 ATR, Lock 1.20 ATR (EA logic: once fav >= 2.20 ATR, SL trails at highest - 1.20 ATR)
# Max holding: 120 bars

def simulate_mt5(
    margin_thresh: float,
    prm_gate: float,
    min_spacing: int,
    loss_cd: int,
    max_pos: int,
    spread_usd: float = 6.40,
    sl_atr_mult: float = 2.50,
    tp_rr: float = 2.00,
    trail_act_atr: float = 2.20,
    trail_lock_atr: float = 1.20,
):
    trades = []
    open_positions = []
    last_entry_bar = -999
    last_exit_bar = -999
    cooldown_until = 0
    consec_loss = 0
    post_exit_cd = 2
    
    initial_balance = 10000.0
    equity = initial_balance
    peak_equity = initial_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    
    # Lot size calculation: ~0.01 lot per position, risk bounded at ~$16 per trade
    lot_size = 0.01
    
    for t in range(N - 60):
        cur_p = closes[t]
        cur_h = highs[t]
        cur_l = lows[t]
        cur_atr = max(atrs[t], 10.0)
        
        # 1. Update open positions & exits
        still_open = []
        for pos in open_positions:
            age = t - pos["entry_bar"]
            
            if pos["dir"] == 1:
                best_h = max(pos["best_fav"], cur_h)
                pos["best_fav"] = best_h
                fav_atr = (best_h - pos["entry_p"]) / cur_atr
                
                # Trailing stop update (Iteration 8 EA logic)
                if fav_atr >= trail_act_atr:
                    tr_sl = best_h - trail_lock_atr * cur_atr
                    pos["sl"] = max(pos["sl"], tr_sl)
                    pos["trailing_active"] = True
                
                sl_hit = cur_l <= pos["sl"]
                tp_hit = cur_h >= pos["tp"]
                time_exit = age >= 120
                
                if sl_hit or tp_hit or time_exit:
                    if tp_hit:
                        exit_p = pos["tp"]
                        exit_kind = "TP"
                    elif sl_hit:
                        exit_p = pos["sl"]
                        exit_kind = "TRAILING_SL" if pos.get("trailing_active") else "INITIAL_SL"
                    else:
                        exit_p = cur_p
                        exit_kind = "TIME_SAFETY"
                    
                    price_diff = exit_p - pos["entry_p"]
                    pnl_usd = price_diff * lot_size - (spread_usd * lot_size)
                    is_win = pnl_usd > 0
                    
                    equity += pnl_usd
                    peak_equity = max(peak_equity, equity)
                    dd_usd = peak_equity - equity
                    dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                    max_dd_usd = max(max_dd_usd, dd_usd)
                    max_dd_pct = max(max_dd_pct, dd_pct)
                    
                    trades.append({
                        "entry_bar": pos["entry_bar"],
                        "exit_bar": t,
                        "dir": 1,
                        "pnl_usd": pnl_usd,
                        "is_win": is_win,
                        "exit_kind": exit_kind,
                        "age": age,
                    })
                    last_exit_bar = t
                    if not is_win:
                        consec_loss += 1
                        if consec_loss >= 2:
                            cooldown_until = t + loss_cd
                    else:
                        consec_loss = 0
                else:
                    still_open.append(pos)
            else:
                best_l = min(pos["best_fav"], cur_l)
                pos["best_fav"] = best_l
                fav_atr = (pos["entry_p"] - best_l) / cur_atr
                
                # Trailing stop update (Iteration 8 EA logic)
                if fav_atr >= trail_act_atr:
                    tr_sl = best_l + trail_lock_atr * cur_atr
                    pos["sl"] = min(pos["sl"], tr_sl)
                    pos["trailing_active"] = True
                
                sl_hit = cur_h >= pos["sl"]
                tp_hit = cur_l <= pos["tp"]
                time_exit = age >= 120
                
                if sl_hit or tp_hit or time_exit:
                    if tp_hit:
                        exit_p = pos["tp"]
                        exit_kind = "TP"
                    elif sl_hit:
                        exit_p = pos["sl"]
                        exit_kind = "TRAILING_SL" if pos.get("trailing_active") else "INITIAL_SL"
                    else:
                        exit_p = cur_p
                        exit_kind = "TIME_SAFETY"
                        
                    price_diff = pos["entry_p"] - exit_p
                    pnl_usd = price_diff * lot_size - (spread_usd * lot_size)
                    is_win = pnl_usd > 0
                    
                    equity += pnl_usd
                    peak_equity = max(peak_equity, equity)
                    dd_usd = peak_equity - equity
                    dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                    max_dd_usd = max(max_dd_usd, dd_usd)
                    max_dd_pct = max(max_dd_pct, dd_pct)
                    
                    trades.append({
                        "entry_bar": pos["entry_bar"],
                        "exit_bar": t,
                        "dir": -1,
                        "pnl_usd": pnl_usd,
                        "is_win": is_win,
                        "exit_kind": exit_kind,
                        "age": age,
                    })
                    last_exit_bar = t
                    if not is_win:
                        consec_loss += 1
                        if consec_loss >= 2:
                            cooldown_until = t + loss_cd
                    else:
                        consec_loss = 0
                else:
                    still_open.append(pos)
                    
        open_positions = still_open
        
        # 2. Check Entry Conditions
        if t < cooldown_until:
            continue
        if (t - last_exit_bar) < post_exit_cd:
            continue
        if (t - last_entry_bar) < min_spacing:
            continue
        if len(open_positions) >= max_pos:
            continue
            
        cur_prm = prm_scores[t]
        cur_margin = margins[t]
        
        if cur_prm < prm_gate:
            continue
            
        act_dir = 0
        if cur_margin >= margin_thresh:
            act_dir = 1
        elif cur_margin <= -margin_thresh:
            act_dir = -1
            
        if act_dir == 0:
            continue
            
        # Spatial pricing check if active positions > 0
        if len(open_positions) > 0:
            last_p = open_positions[-1]["entry_p"]
            if abs(cur_p - last_p) < 0.20 * cur_atr:
                continue
            if act_dir == 1 and cur_p < last_p - 0.25 * cur_atr:
                continue
            if act_dir == -1 and cur_p > last_p + 0.25 * cur_atr:
                continue
                
        # Enter position
        ep = cur_p
        sl_dist = sl_atr_mult * cur_atr
        tp_dist = sl_dist * tp_rr
        sl = (ep - sl_dist) if act_dir == 1 else (ep + sl_dist)
        tp = (ep + tp_dist) if act_dir == 1 else (ep - tp_dist)
        
        open_positions.append({
            "entry_bar": t,
            "entry_p": ep,
            "dir": act_dir,
            "sl": sl,
            "tp": tp,
            "best_fav": ep,
            "trailing_active": False,
        })
        last_entry_bar = t

    tot = len(trades)
    if tot == 0:
        return {"trades": 0, "trades_per_day": 0.0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "max_dd_pct": 0.0, "buy_n": 0, "sell_n": 0}
        
    wins = [tr for tr in trades if tr["is_win"]]
    losses = [tr for tr in trades if not tr["is_win"]]
    w_cnt = len(wins)
    wr = (w_cnt / tot) * 100.0
    gp = sum(tr["pnl_usd"] for tr in wins)
    gl = abs(sum(tr["pnl_usd"] for tr in losses))
    pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 1.0)
    net_pnl = gp - gl
    trades_per_day = tot / 6.0  # 6 active trading days in Oct 19-25 week
    
    buys = [tr for tr in trades if tr["dir"] == 1]
    sells = [tr for tr in trades if tr["dir"] == -1]
    buy_wr = (sum(1 for tr in buys if tr["is_win"]) / max(1, len(buys))) * 100.0
    sell_wr = (sum(1 for tr in sells if tr["is_win"]) / max(1, len(sells))) * 100.0
    
    tp_cnt = sum(1 for tr in trades if tr["exit_kind"] == "TP")
    trail_cnt = sum(1 for tr in trades if tr["exit_kind"] == "TRAILING_SL")
    sl_cnt = sum(1 for tr in trades if tr["exit_kind"] == "INITIAL_SL")
    
    return {
        "trades": tot,
        "trades_per_day": trades_per_day,
        "wr": wr,
        "pf": pf,
        "net_pnl": net_pnl,
        "max_dd_pct": max_dd_pct,
        "buy_n": len(buys),
        "buy_wr": buy_wr,
        "sell_n": len(sells),
        "sell_wr": sell_wr,
        "tp_cnt": tp_cnt,
        "trail_cnt": trail_cnt,
        "sl_cnt": sl_cnt,
    }

# 1. First test: Replicate Iteration 8 baseline
baseline_res = simulate_mt5(
    margin_thresh=0.12,
    prm_gate=0.40,
    min_spacing=2,
    loss_cd=2,
    max_pos=3,
)
print("ITERATION 8 BASELINE REPLICATION CHECK:")
print(f"Simulated: Trades={baseline_res['trades']}, Trades/Day={baseline_res['trades_per_day']:.2f}, WR={baseline_res['wr']:.2f}%, PF={baseline_res['pf']:.2f}, NetPnL=${baseline_res['net_pnl']:.2f}, MaxDD={baseline_res['max_dd_pct']:.2f}% (Buys: {baseline_res['buy_n']}, Sells: {baseline_res['sell_n']})")
print("Actual MT5 Iter 8: Trades=89, Trades/Day=14.83, WR=57.30%, PF=1.38, NetPnL=$230.07, MaxDD=1.05% (Buys: 78, Sells: 11)")

print("\n" + "=" * 80)
print("PHASE 5: SYSTEMATIC PARAMETER CALIBRATION SWEEP")
print("=" * 80)
print(f"{'Margin':<8} | {'PRM':<6} | {'Space':<6} | {'MaxP':<6} | {'LossCD':<6} | {'Trades':<7} | {'Tr/Day':<8} | {'WR%':<7} | {'PF':<6} | {'NetPnL':<9} | {'MaxDD%':<7} | {'B:S':<9}")
print("-" * 105)

results = []
for m_th in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]:
    for p_g in [0.32, 0.35, 0.40]:
        for sp in [1, 2]:
            for mp in [2, 3]:
                for lcd in [2, 3]:
                    r = simulate_mt5(
                        margin_thresh=m_th,
                        prm_gate=p_g,
                        min_spacing=sp,
                        loss_cd=lcd,
                        max_pos=mp,
                    )
                    r.update({
                        "margin_thresh": m_th,
                        "prm_gate": p_g,
                        "min_spacing": sp,
                        "max_pos": mp,
                        "loss_cd": lcd,
                    })
                    results.append(r)
                    
                    # Highlight if trades/day >= 35.0
                    tag = " *** TARGET ***" if r["trades_per_day"] >= 40.0 and r["wr"] >= 55.0 and r["pf"] >= 1.20 else ""
                    if r["trades_per_day"] >= 20.0 or m_th == 0.12:
                        print(f"{m_th:<8.2f} | {p_g:<6.2f} | {sp:<6d} | {mp:<6d} | {lcd:<6d} | {r['trades']:<7d} | {r['trades_per_day']:<8.2f} | {r['wr']:<7.2f} | {r['pf']:<6.2f} | ${r['net_pnl']:<8.2f} | {r['max_dd_pct']:<7.2f} | {r['buy_n']}:{r['sell_n']:<4d}{tag}")

out_path = ROOT / "scratch" / "calibration_sweep_results.json"
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved {len(results)} sweep configurations to {out_path}")
