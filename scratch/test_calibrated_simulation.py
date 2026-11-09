import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

clean_csv = ROOT / "outputs" / "btcusd_m1_7day_clean.csv"
df = pd.read_csv(clean_csv, sep="\t")
ingress, targets, traps, prices, atrs, s_highs, s_lows = extract_features_and_targets_from_m1(df)

ckpt_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
cfg = BilateralDualCoTConfig(prm_threshold=0.65, trap_threshold=0.65)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(ckpt_path)
engine.eval()

device = torch.device("cpu")
with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress).float(), batch_size=512)
    actions = verdict.actions.numpy()
    prms = verdict.prm_scores.numpy()
    cps = verdict.counter_punches.numpy()

N = len(prices)
for min_cadence in [5, 8, 12]:
    for prm_gate in [0.55, 0.65, 0.70]:
        trades = []
        last_bar = -999
        for t in range(N):
            act = int(actions[t])
            prm = float(prms[t])
            if act != 0 and prm >= prm_gate and (t - last_bar >= min_cadence):
                px = prices[t]
                atr = atrs[t]
                
                # Check 60-bar range guard
                rec_l = np.min(prices[max(0, t - 60):t + 1])
                rec_h = np.max(prices[max(0, t - 60):t + 1])
                rng = max(1.0, rec_h - rec_l)
                rel_pos = (px - rec_l) / rng
                if act == 1 and rel_pos >= 0.75:
                    continue
                if act == -1 and rel_pos <= 0.25:
                    continue
                
                # Path-dependent simulation
                horizon = min(30, N - t - 1)
                if horizon < 5:
                    continue
                
                fwd_c = prices[t+1 : t+1+horizon]
                fwd_h = s_highs[t+1 : t+1+horizon]
                fwd_l = s_lows[t+1 : t+1+horizon]
                
                spread_usd = 2.50
                sl_dist = 1.50 * atr
                tp_dist = 3.00 * atr # 1:2 RR
                be_trig = 1.00 * atr
                be_lock = 0.25 * atr
                cur_sl = sl_dist
                is_win = False
                pnl_r = 0.0
                
                for k in range(len(fwd_c)):
                    h_b = fwd_h[k]
                    l_b = fwd_l[k]
                    
                    if act == 1:
                        fav = h_b - px
                        adv = px - l_b
                        if fav >= be_trig:
                            cur_sl = -be_lock
                        sl_hit = (adv >= cur_sl)
                        tp_hit = (fav >= tp_dist)
                        if sl_hit and tp_hit:
                            is_win = (cur_sl < 0)
                            pnl_r = (be_lock / atr) if is_win else -1.0
                            break
                        elif sl_hit:
                            is_win = (cur_sl < 0)
                            pnl_r = (be_lock / atr) if is_win else -1.0
                            break
                        elif tp_hit:
                            is_win = True
                            pnl_r = (tp_dist - spread_usd) / sl_dist
                            break
                    else:
                        fav = px - l_b
                        adv = h_b - px
                        if fav >= be_trig:
                            cur_sl = -be_lock
                        sl_hit = (adv >= cur_sl)
                        tp_hit = (fav >= tp_dist)
                        if sl_hit and tp_hit:
                            is_win = (cur_sl < 0)
                            pnl_r = (be_lock / atr) if is_win else -1.0
                            break
                        elif sl_hit:
                            is_win = (cur_sl < 0)
                            pnl_r = (be_lock / atr) if is_win else -1.0
                            break
                        elif tp_hit:
                            is_win = True
                            pnl_r = (tp_dist - spread_usd) / sl_dist
                            break
                
                if pnl_r == 0.0:
                    net_diff = (fwd_c[-1] - px - spread_usd) if act == 1 else (px - fwd_c[-1] - spread_usd)
                    is_win = (net_diff > 0)
                    pnl_r = net_diff / max(1.0, sl_dist)
                    
                trades.append({"act": act, "is_win": is_win, "pnl_r": pnl_r})
                last_bar = t
                
        tot = len(trades)
        if tot > 0:
            w = sum(1 for tr in trades if tr["is_win"])
            wr = w / tot * 100
            gp = sum(tr["pnl_r"] for tr in trades if tr["pnl_r"] > 0)
            gl = abs(sum(tr["pnl_r"] for tr in trades if tr["pnl_r"] < 0))
            pf = (gp / gl) if gl > 0 else 999.0
            net_r = sum(tr["pnl_r"] for tr in trades)
            print(f"Cadence: {min_cadence:2d} | PRM: {prm_gate:.2f} | Trades: {tot:4d} | WR: {wr:5.1f}% | PF: {pf:5.2f} | Net R: {net_r:+6.1f}R")
