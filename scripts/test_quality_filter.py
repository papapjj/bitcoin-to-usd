import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

print("Loading data...")
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
ingress, targets, traps, prices, atrs = extract_features_and_targets_from_m1(df)

config = BilateralDualCoTConfig()
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint('outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt')
engine.eval()

with torch.no_grad():
    batch_verdict = engine.forward_batch(torch.from_numpy(ingress).float(), batch_size=512)
    actions = batch_verdict.actions.numpy()
    prms = batch_verdict.prm_scores.numpy()
    confs = batch_verdict.confidences.numpy()
    b_scores = batch_verdict.bull_scores.numpy()
    r_scores = batch_verdict.bear_scores.numpy()

N = len(prices)
highs = df['<HIGH>'].values[-N:] if '<HIGH>' in df.columns else df['high'].values[-N:]
lows = df['<LOW>'].values[-N:] if '<LOW>' in df.columns else df['low'].values[-N:]
closes = df['<CLOSE>'].values[-N:] if '<CLOSE>' in df.columns else df['close'].values[-N:]
opens = df['<OPEN>'].values[-N:] if '<OPEN>' in df.columns else df['open'].values[-N:]

margins = b_scores - r_scores

header = f"{'Min_Margin':<11} {'Min_PRM':<8} {'Min_Conf':<9} | {'Trades':<7} {'Buys':<6} {'Sells':<6} {'WR%':<7} {'PF':<6} {'DD%':<6}"
print(header)
print('-' * len(header))

results = []
for min_margin in [0.05, 0.08, 0.10, 0.12, 0.15]:
    for min_prm in [0.30, 0.35, 0.40]:
        for min_conf in [0.45, 0.48, 0.50]:
            trades = []
            buys_cnt = 0
            sells_cnt = 0
            last_t = -999
            for t in range(50, N - 30):
                if t - last_t < 2:
                    continue
                prm = prms[t]
                conf = confs[t]
                m = margins[t]
                
                if prm < min_prm or conf < min_conf:
                    continue
                
                act = 0
                if m >= min_margin and b_scores[t] >= 0.45:
                    act = 1
                elif -m >= min_margin and r_scores[t] >= 0.35:
                    act = -1
                if act == 0:
                    continue
                
                # Trend Guard
                ma20 = np.mean(prices[t-20:t])
                ma50 = np.mean(prices[t-50:t])
                if ma20 > ma50 and prices[t] > ma50 and act == -1:
                    continue
                if ma20 < ma50 and prices[t] < ma50 and act == 1:
                    continue
                
                # Sniper confirmation: green candle or lower wick absorption for buy, red candle or upper wick rejection for sell
                if act == 1:
                    if not (closes[t] >= opens[t] or (closes[t] - lows[t] >= 0.20 * atrs[t])):
                        continue
                else:
                    if not (closes[t] <= opens[t] or (highs[t] - closes[t] >= 0.20 * atrs[t])):
                        continue
                        
                last_t = t
                if act == 1: buys_cnt += 1
                else: sells_cnt += 1
                
                px = prices[t]
                atr = atrs[t]
                fwd_h = highs[t+1 : t+31]
                fwd_l = lows[t+1 : t+31]
                fwd_c = closes[t+1 : t+31]
                
                sl_dist = 2.0 * atr
                be_trig = 0.8 * atr
                be_lock = 0.35 * atr
                tp_dist = 2.2 * atr
                trail = 0.8 * atr
                
                cur_sl = sl_dist
                peak_fav = 0.0
                pnl_r = 0.0
                done = False
                for k in range(len(fwd_c)):
                    fav = (fwd_h[k] - px) if act == 1 else (px - fwd_l[k])
                    adv = (px - fwd_l[k]) if act == 1 else (fwd_h[k] - px)
                    c_fav = (fwd_c[k] - px) if act == 1 else (px - fwd_c[k])
                    peak_fav = max(peak_fav, fav)
                    if peak_fav >= be_trig:
                        cur_sl = -max(be_lock, peak_fav - trail)
                    if adv >= cur_sl:
                        pnl_r = -cur_sl / sl_dist
                        done = True
                        break
                    elif fav >= tp_dist:
                        pnl_r = tp_dist / sl_dist
                        done = True
                        break
                if not done:
                    pnl_r = c_fav / sl_dist
                trades.append(pnl_r)
                
            if len(trades) >= 100:
                tr = np.array(trades)
                wins = tr[tr > 0]
                losses = tr[tr <= 0]
                wr = len(wins) / len(tr) * 100
                total_w = wins.sum()
                total_l = abs(losses.sum()) if len(losses) > 0 else 0.001
                pf = total_w / total_l
                cum = np.cumsum(tr)
                dd = (np.maximum.accumulate(cum) - cum).max() * 0.25
                results.append((wr, pf, len(tr), buys_cnt, sells_cnt, dd, min_margin, min_prm, min_conf))

results.sort(key=lambda x: (x[0], x[1]), reverse=True)
for wr, pf, n, b, s, dd, mm, mp, mc in results[:20]:
    print(f"{mm:<11.2f} {mp:<8.2f} {mc:<9.2f} | {n:<7} {b:<6} {s:<6} {wr:<7.2f} {pf:<6.2f} {dd:<6.2f}")
