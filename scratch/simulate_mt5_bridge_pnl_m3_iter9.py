import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd
import json

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)
from scripts.test_5_regimes_zero_filter import process_regime_csv

WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
BRIDGE_CSV = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_bridge.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

# Load M1 price dataset
w3_data = process_regime_csv(WEEK3_CSV)
closes_m1 = w3_data["closes"]
highs_m1 = w3_data["highs"]
lows_m1 = w3_data["lows"]
atrs_m1 = w3_data["atrs"]

# Load bridge csv
df_bridge = pd.read_csv(BRIDGE_CSV, sep="\t", low_memory=False)
df_agent0 = df_bridge[df_bridge["agent_id"] == 0].reset_index(drop=True)
N = len(df_agent0)

# Extract inputs and forward pass
base_obs_list = []
times_list = df_agent0["time"].tolist()

for i in range(N):
    raw_str = str(df_agent0.iloc[i, 18])
    parts = [float(x) for x in raw_str.strip("|").split("|") if x]
    if len(parts) < 137:
        parts = parts + [0.0] * (137 - len(parts))
    base_obs_list.append(parts[:137])
base_obs_matrix = np.array(base_obs_list, dtype=np.float32)

ingress_matrix = np.array([
    extract_canonical_bilateral_ingress_233(base_obs_matrix[i], None, 0.0, 45.0)
    for i in range(N)
], dtype=np.float32)

cfg = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256,
    num_hops=4, num_debate_turns=3, prm_threshold=0.32,
    trap_threshold=0.60, margin_threshold=0.12,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

with torch.no_grad():
    v = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=256)

b_scores = v.bull_scores.cpu().numpy()
r_scores = v.bear_scores.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
margins = (b_scores - 0.0218) - (r_scores - 0.2275)

# Map bridge times to M1 bar index in w3_data
# w3_data dataframe dates
df_m1 = pd.read_csv(WEEK3_CSV, sep="\t")
col_date = "<DATE>" if "<DATE>" in df_m1.columns else "DATE"
col_time = "<TIME>" if "<TIME>" in df_m1.columns else "TIME"
m1_times = (df_m1[col_date].astype(str) + " " + df_m1[col_time].astype(str).str[:5]).values
time_to_m1_idx = {str(t): idx for idx, t in enumerate(m1_times)}

packet_to_m1_idx = []
for t_str in times_list:
    # formatted like '2026.10.19 00:03'
    idx = time_to_m1_idx.get(t_str, -1)
    packet_to_m1_idx.append(idx)

# High-fidelity simulation across the 1,770 decision packets
def simulate_mt5_packets(
    margin_thresh: float,
    prm_gate: float,
    min_spacing_packets: int = 2,
    post_exit_cd_packets: int = 2,
    loss_cd_packets: int = 2,
    max_pos: int = 3,
    sl_atr_mult: float = 2.50,
    tp_rr: float = 2.00,
    trail_act_atr: float = 2.20,
    trail_lock_atr: float = 1.20,
    spread_usd: float = 6.40,
):
    trades = []
    open_positions = []
    last_entry_pkt = -999
    last_exit_pkt = -999
    cooldown_until_pkt = 0
    consec_losses = 0
    
    initial_balance = 10000.0
    equity = initial_balance
    peak_equity = initial_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    
    # Track daily counts
    daily_trades = {}
    
    for pkt in range(N):
        pkt_time = times_list[pkt]
        day_str = pkt_time.split()[0]
        m1_idx = packet_to_m1_idx[pkt]
        if m1_idx < 0:
            continue
            
        cur_p = closes_m1[m1_idx]
        cur_h = highs_m1[m1_idx]
        cur_l = lows_m1[m1_idx]
        cur_atr = max(atrs_m1[m1_idx], 10.0)
        
        # 1. Update exits for open positions
        still_open = []
        for pos in open_positions:
            entry_m1 = pos["entry_m1"]
            m1_age = m1_idx - entry_m1
            
            # Check price trajectory between last check and current m1_idx
            # For realism, scan highs and lows during hold
            window_h = np.max(highs_m1[pos["last_scanned_m1"]:m1_idx+1]) if m1_idx >= pos["last_scanned_m1"] else cur_h
            window_l = np.min(lows_m1[pos["last_scanned_m1"]:m1_idx+1]) if m1_idx >= pos["last_scanned_m1"] else cur_l
            pos["last_scanned_m1"] = m1_idx + 1
            
            if pos["dir"] == 1:
                best_h = max(pos["best_fav"], window_h)
                pos["best_fav"] = best_h
                fav_atr = (best_h - pos["entry_p"]) / pos["entry_atr"]
                
                if fav_atr >= trail_act_atr:
                    tr_sl = best_h - trail_lock_atr * pos["entry_atr"]
                    pos["sl"] = max(pos["sl"], tr_sl)
                    pos["trailing_active"] = True
                    
                sl_hit = window_l <= pos["sl"]
                tp_hit = window_h >= pos["tp"]
                time_exit = m1_age >= 120
                
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
                        
                    pnl_usd = (exit_p - pos["entry_p"] - spread_usd) * pos["lot_size"]
                    is_win = pnl_usd > 0
                    
                    equity += pnl_usd
                    peak_equity = max(peak_equity, equity)
                    dd_usd = peak_equity - equity
                    dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                    max_dd_usd = max(max_dd_usd, dd_usd)
                    max_dd_pct = max(max_dd_pct, dd_pct)
                    
                    trades.append({
                        "entry_time": pos["entry_time"],
                        "exit_time": pkt_time,
                        "day": day_str,
                        "dir": 1,
                        "pnl_usd": pnl_usd,
                        "is_win": is_win,
                        "exit_kind": exit_kind,
                    })
                    daily_trades[day_str] = daily_trades.get(day_str, 0) + 1
                    last_exit_pkt = pkt
                    if not is_win:
                        consec_losses += 1
                        if consec_losses >= 2:
                            cooldown_until_pkt = pkt + loss_cd_packets
                    else:
                        consec_losses = 0
                else:
                    still_open.append(pos)
            else:
                best_l = min(pos["best_fav"], window_l)
                pos["best_fav"] = best_l
                fav_atr = (pos["entry_p"] - best_l) / pos["entry_atr"]
                
                if fav_atr >= trail_act_atr:
                    tr_sl = best_l + trail_lock_atr * pos["entry_atr"]
                    pos["sl"] = min(pos["sl"], tr_sl)
                    pos["trailing_active"] = True
                    
                sl_hit = window_h >= pos["sl"]
                tp_hit = window_l <= pos["tp"]
                time_exit = m1_age >= 120
                
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
                        
                    pnl_usd = (pos["entry_p"] - exit_p - spread_usd) * pos["lot_size"]
                    is_win = pnl_usd > 0
                    
                    equity += pnl_usd
                    peak_equity = max(peak_equity, equity)
                    dd_usd = peak_equity - equity
                    dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                    max_dd_usd = max(max_dd_usd, dd_usd)
                    max_dd_pct = max(max_dd_pct, dd_pct)
                    
                    trades.append({
                        "entry_time": pos["entry_time"],
                        "exit_time": pkt_time,
                        "day": day_str,
                        "dir": -1,
                        "pnl_usd": pnl_usd,
                        "is_win": is_win,
                        "exit_kind": exit_kind,
                    })
                    daily_trades[day_str] = daily_trades.get(day_str, 0) + 1
                    last_exit_pkt = pkt
                    if not is_win:
                        consec_losses += 1
                        if consec_losses >= 2:
                            cooldown_until_pkt = pkt + loss_cd_packets
                    else:
                        consec_losses = 0
                else:
                    still_open.append(pos)
                    
        open_positions = still_open
        
        # 2. Check entry eligibility
        if pkt < cooldown_until_pkt:
            continue
        if (pkt - last_exit_pkt) < post_exit_cd_packets:
            continue
        if (pkt - last_entry_pkt) < min_spacing_packets:
            continue
        if len(open_positions) >= max_pos:
            continue
            
        cur_prm = prm_scores[pkt]
        cur_margin = margins[pkt]
        
        if cur_prm < prm_gate:
            continue
            
        act_dir = 0
        if cur_margin >= margin_thresh:
            act_dir = 1
        elif cur_margin <= -margin_thresh:
            act_dir = -1
        else:
            continue
            
        # Spatial pricing check
        if len(open_positions) > 0:
            last_p = open_positions[-1]["entry_p"]
            if abs(cur_p - last_p) < 0.20 * cur_atr:
                continue
            if act_dir == 1 and cur_p < last_p - 0.25 * cur_atr:
                continue
            if act_dir == -1 and cur_p > last_p + 0.25 * cur_atr:
                continue
                
        # Risk sizing: dynamic risk scaled to ~0.01-0.02 lot ($16 risk per trade)
        sl_dist = sl_atr_mult * cur_atr
        tp_dist = sl_dist * tp_rr
        # ~0.01 lot for BTCUSDm: point value = 1.0, 1 lot = 1 BTC.
        # $16.00 risk / sl_dist = lot_size
        lot_sz = round(16.0 / max(sl_dist, 10.0), 2)
        lot_sz = max(0.01, min(0.30, lot_sz))
        
        sl = (cur_p - sl_dist) if act_dir == 1 else (cur_p + sl_dist)
        tp = (cur_p + tp_dist) if act_dir == 1 else (cur_p - tp_dist)
        
        open_positions.append({
            "entry_pkt": pkt,
            "entry_m1": m1_idx,
            "last_scanned_m1": m1_idx,
            "entry_time": pkt_time,
            "entry_p": cur_p,
            "entry_atr": cur_atr,
            "lot_size": lot_sz,
            "dir": act_dir,
            "sl": sl,
            "tp": tp,
            "best_fav": cur_p,
            "trailing_active": False,
        })
        last_entry_pkt = pkt
        
    tot = len(trades)
    if tot == 0:
        return {"trades": 0, "trades_per_day": 0.0, "wr": 0.0, "pf": 0.0, "net_pnl": 0.0, "max_dd_pct": 0.0}
        
    wins = [tr for tr in trades if tr["is_win"]]
    losses = [tr for tr in trades if not tr["is_win"]]
    wr = (len(wins) / tot) * 100.0
    gp = sum(tr["pnl_usd"] for tr in wins)
    gl = abs(sum(tr["pnl_usd"] for tr in losses))
    pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 1.0)
    net_pnl = gp - gl
    
    buys = [tr for tr in trades if tr["dir"] == 1]
    sells = [tr for tr in trades if tr["dir"] == -1]
    buy_wr = (sum(1 for tr in buys if tr["is_win"]) / max(1, len(buys))) * 100.0
    sell_wr = (sum(1 for tr in sells if tr["is_win"]) / max(1, len(sells))) * 100.0
    
    tp_cnt = sum(1 for tr in trades if tr["exit_kind"] == "TP")
    trail_cnt = sum(1 for tr in trades if tr["exit_kind"] == "TRAILING_SL")
    sl_cnt = sum(1 for tr in trades if tr["exit_kind"] == "INITIAL_SL")
    
    # Active trading days: 6 days
    trades_per_day = tot / 6.0
    
    return {
        "trades": tot,
        "trades_per_day": trades_per_day,
        "wr": wr,
        "pf": pf,
        "net_pnl": net_pnl,
        "max_dd_pct": max_dd_pct,
        "buy_cnt": len(buys),
        "buy_wr": buy_wr,
        "sell_cnt": len(sells),
        "sell_wr": sell_wr,
        "tp_cnt": tp_cnt,
        "trail_cnt": trail_cnt,
        "sl_cnt": sl_cnt,
        "daily_breakdown": daily_trades,
    }

print("=" * 80)
print("ACCURATE MT5 STRATEGY TESTER PACKET SIMULATION RESULTS")
print("=" * 80)
print(f"{'Config':<28} | {'Trades':<7} | {'Tr/Day':<7} | {'WR%':<6} | {'PF':<5} | {'Net PnL':<9} | {'Max DD':<6} | {'B:S Ratio':<9} | {'Verdict'}")
print("-" * 105)

configs = [
    ("Iter 8 Baseline (0.12/0.40/2/3)", 0.12, 0.40, 2, 2, 2, 3),
    ("Candidate 1 (0.08/0.35/2/3)", 0.08, 0.35, 2, 2, 2, 3),
    ("Candidate 2 (0.07/0.35/2/3)", 0.07, 0.35, 2, 2, 2, 3),
    ("Candidate 3 (0.06/0.35/2/3)", 0.06, 0.35, 2, 2, 2, 3),
    ("Candidate 4 (0.07/0.35/1/3)", 0.07, 0.35, 1, 2, 2, 3),
    ("Candidate 5 (0.08/0.32/2/3)", 0.08, 0.32, 2, 2, 2, 3),
    ("Candidate 6 (0.07/0.32/2/3)", 0.07, 0.32, 2, 2, 2, 3),
    ("Candidate 7 (0.06/0.32/2/3)", 0.06, 0.32, 2, 2, 2, 3),
]

for name, m_th, p_g, sp, pcd, lcd, mp in configs:
    res = simulate_mt5_packets(
        margin_thresh=m_th,
        prm_gate=p_g,
        min_spacing_packets=sp,
        post_exit_cd_packets=pcd,
        loss_cd_packets=lcd,
        max_pos=mp,
    )
    ratio_str = f"{res['buy_cnt']}:{res['sell_cnt']}"
    meets_target = res["trades_per_day"] >= 40.0 and res["net_pnl"] > 0
    verd = "PASSED TRADES" if meets_target else "UNDER TARGET"
    print(f"{name:<28} | {res['trades']:<7d} | {res['trades_per_day']:<7.2f} | {res['wr']:<6.2f} | {res['pf']:<5.2f} | ${res['net_pnl']:<8.2f} | {res['max_dd_pct']:<5.2f}% | {ratio_str:<9} | {verd}")
    print(f"   -> Daily: {res['daily_breakdown']}")
    print(f"   -> Exits: TP={res['tp_cnt']}, Trailing={res['trail_cnt']}, SL={res['sl_cnt']} | BuyWR={res['buy_wr']:.1f}%, SellWR={res['sell_wr']:.1f}%\n")
