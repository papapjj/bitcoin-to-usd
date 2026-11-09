import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from scripts.test_5_regimes_zero_filter import process_regime_csv

WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

cfg = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.50,
    trap_threshold=0.60,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

data = extract_features_from_csv(WEEK3_CSV)
ingress = torch.from_numpy(data["ingress"]).float()
with torch.no_grad():
    v = engine.forward_batch(ingress, batch_size=256)

actions = v.actions.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
confidences = v.confidences.cpu().numpy()

w3_data = process_regime_csv(WEEK3_CSV)
closes = w3_data["closes"]
highs = w3_data["highs"]
lows = w3_data["lows"]
atrs = w3_data["atrs"]
N = min(len(closes), len(actions))
spread_usd = 2.50
sl_atr_mult = 2.50
trail_atr_mult = 0.75

tp_rr = 2.00

for prm_thresh in [0.45, 0.48, 0.50, 0.52]:
    for max_pos in [2, 3]:
        for min_spacing in [2, 3]:
            for cd_bars in [3, 5]:
                trades = []
                open_positions = []
                last_entry_bar = -999
                cooldown_until = 0
                consec_loss = 0
                
                for t in range(N - 60):
                    # Check exits for open positions
                    cur_p = closes[t]
                    cur_h = highs[t]
                    cur_l = lows[t]
                    cur_atr = max(atrs[t], 10.0)
                    
                    still_open = []
                    for pos in open_positions:
                        age = t - pos["entry_bar"]
                        # Trailing stop
                        if pos["dir"] == 1:
                            best_high = max(pos["best_fav"], cur_h)
                            pos["best_fav"] = best_high
                            fav_atr = (best_high - pos["entry_p"]) / cur_atr
                            if trail_atr_mult > 0.0 and fav_atr >= 1.50:
                                tr_sl = best_high - trail_atr_mult * cur_atr
                                pos["sl"] = max(pos["sl"], tr_sl)
                            
                            sl_hit = cur_l <= pos["sl"]
                            tp_hit = cur_h >= pos["tp"]
                            time_exit = age >= 120
                            
                            if sl_hit or tp_hit or time_exit:
                                exit_p = pos["sl"] if sl_hit else (pos["tp"] if tp_hit else cur_p)
                                pnl = exit_p - pos["entry_p"] - spread_usd
                                is_win = pnl > 0
                                trades.append({"pnl": pnl, "is_win": is_win, "dir": 1, "age": age})
                                if not is_win:
                                    consec_loss += 1
                                    if consec_loss >= 2:
                                        cooldown_until = t + cd_bars
                                else:
                                    consec_loss = 0
                            else:
                                still_open.append(pos)
                        else:
                            best_low = min(pos["best_fav"], cur_l)
                            pos["best_fav"] = best_low
                            fav_atr = (pos["entry_p"] - best_low) / cur_atr
                            if trail_atr_mult > 0.0 and fav_atr >= 1.50:
                                tr_sl = best_low + trail_atr_mult * cur_atr
                                pos["sl"] = min(pos["sl"], tr_sl)
                            
                            sl_hit = cur_h >= pos["sl"]
                            tp_hit = cur_l <= pos["tp"]
                            time_exit = age >= 120
                            
                            if sl_hit or tp_hit or time_exit:
                                exit_p = pos["sl"] if sl_hit else (pos["tp"] if tp_hit else cur_p)
                                pnl = pos["entry_p"] - exit_p - spread_usd
                                is_win = pnl > 0
                                trades.append({"pnl": pnl, "is_win": is_win, "dir": -1, "age": age})
                                if not is_win:
                                    consec_loss += 1
                                    if consec_loss >= 2:
                                        cooldown_until = t + cd_bars
                                else:
                                    consec_loss = 0
                            else:
                                still_open.append(pos)
                    open_positions = still_open
                    
                    # Entry logic
                    if t < cooldown_until:
                        continue
                    if t - last_entry_bar < min_spacing:
                        continue
                    if len(open_positions) >= max_pos:
                        continue
                        
                    raw_act = actions[t]
                    prm = prm_scores[t]
                    
                    if prm < prm_thresh:
                        continue
                    if raw_act == 0:
                        continue
                        
                    act_dir = 1 if raw_act == 1 else (-1 if raw_act in (-1, 2) else 0)
                    if act_dir == 0:
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
                    })
                    last_entry_bar = t
                    
                tot = len(trades)
                if tot > 0:
                    w = sum(1 for tr in trades if tr["is_win"])
                    wr = (w / tot) * 100.0
                    gp = sum(tr["pnl"] for tr in trades if tr["pnl"] > 0)
                    gl = abs(sum(tr["pnl"] for tr in trades if tr["pnl"] < 0))
                    pf = (gp / gl) if gl > 0 else 999.0
                    trades_per_day = tot / 7.0
                    print(f"PRM: {prm_thresh:.2f} | MaxPos: {max_pos} | Spacing: {min_spacing} | CD: {cd_bars} | Trades: {tot:4d} ({trades_per_day:5.1f}/d) | WR: {wr:5.1f}% | PF: {pf:5.2f}")
