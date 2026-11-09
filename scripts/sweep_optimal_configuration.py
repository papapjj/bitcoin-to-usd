import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from collections import deque
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

print("Loading data...")
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
ingress, targets, traps, prices, atrs = extract_features_and_targets_from_m1(df)
N = len(ingress)
opens = df['<OPEN>'].values[-N:] if '<OPEN>' in df.columns else df['open'].values[-N:]
highs = df['<HIGH>'].values[-N:] if '<HIGH>' in df.columns else df['high'].values[-N:]
lows = df['<LOW>'].values[-N:] if '<LOW>' in df.columns else df['low'].values[-N:]
closes = df['<CLOSE>'].values[-N:] if '<CLOSE>' in df.columns else df['close'].values[-N:]

config = BilateralDualCoTConfig()
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint('outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt')
engine.eval()

with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress).float(), batch_size=512)
    b_s = verdict.bull_scores.numpy()
    r_s = verdict.bear_scores.numpy()
    prms = verdict.prm_scores.numpy()
    cps = verdict.counter_punches.numpy()

s_prices = pd.Series(prices)
ema20 = s_prices.ewm(span=20).mean().values
ema50 = s_prices.ewm(span=50).mean().values

results = []
for spacing in [1, 2, 3]:
    for sl_atr in [1.5, 2.0, 2.5]:
        for tp_atr in [1.5, 2.0, 3.0]:
            for be_trig in [0.50, 0.60, 0.80]:
                for be_lock in [0.25, 0.35, 0.50]:
                    b_hist, r_hist = deque(maxlen=120), deque(maxlen=120)
                    trades = []
                    b_cnt, s_cnt = 0, 0
                    last_t = -999
                    for t in range(120, N - 35):
                        bs, rs = float(b_s[t]), float(r_s[t])
                        b_hist.append(bs); r_hist.append(rs)
                        if len(b_hist) < 20: continue
                        mb, sb = np.mean(b_hist), max(0.01, np.std(b_hist))
                        mr, sr = np.mean(r_hist), max(0.01, np.std(r_hist))
                        bz, rz = (bs - mb) / sb, (rs - mr) / sr
                        
                        up = prices[t] > ema50[t] and ema20[t] > ema50[t]
                        down = prices[t] < ema50[t] and ema20[t] < ema50[t]
                        
                        act = 0
                        if cps[t]: act = int(verdict.actions[t])
                        elif bz >= 0.20 and (bz - rz) >= 0.15 and up: act = 1
                        elif rz >= 0.20 and (rz - bz) >= 0.15 and down: act = -1
                        if act == 0: continue
                        
                        atr = atrs[t]
                        cur_p = prices[t]
                        if not cps[t]:
                            if act == 1 and not (closes[t] >= opens[t] or (closes[t] - lows[t]) >= 0.20 * atr): continue
                            elif act == -1 and not (closes[t] <= opens[t] or (highs[t] - closes[t]) >= 0.20 * atr): continue
                            
                        if t - last_t < spacing: continue
                        last_t = t
                        if act == 1: b_cnt += 1
                        else: s_cnt += 1
                        
                        sl_d = sl_atr * atr
                        tp_d = tp_atr * atr
                        bet_d = be_trig * atr
                        bel_d = be_lock * atr
                        trail_d = 0.60 * atr
                        
                        if act == 1:
                            dyn_sl = cur_p - sl_d
                            tp_p = cur_p + tp_d
                            be_arm = False
                            pnl = 0.0
                            for k in range(t + 1, min(t + 35, N)):
                                bh, bl, bc = highs[k], lows[k], closes[k]
                                if not be_arm and bh >= cur_p + bet_d:
                                    be_arm = True
                                    dyn_sl = max(dyn_sl, cur_p + bel_d)
                                if be_arm: dyn_sl = max(dyn_sl, bh - trail_d)
                                if bl <= dyn_sl: pnl = dyn_sl - cur_p; break
                                elif bh >= tp_p: pnl = tp_d; break
                            else: pnl = bc - cur_p
                        else:
                            dyn_sl = cur_p + sl_d
                            tp_p = cur_p - tp_d
                            be_arm = False
                            pnl = 0.0
                            for k in range(t + 1, min(t + 35, N)):
                                bh, bl, bc = highs[k], lows[k], closes[k]
                                if not be_arm and bl <= cur_p - bet_d:
                                    be_arm = True
                                    dyn_sl = min(dyn_sl, cur_p - bel_d)
                                if be_arm: dyn_sl = min(dyn_sl, bl + trail_d)
                                if bh >= dyn_sl: pnl = cur_p - dyn_sl; break
                                elif bl <= tp_p: pnl = tp_d; break
                            else: pnl = cur_p - bc
                        trades.append(pnl / sl_d)
                        
                    tr = np.array(trades)
                    n = len(tr)
                    if n < 100: continue
                    w = np.sum(tr > 0)
                    wr = w / n * 100.0
                    gp = np.sum(tr[tr > 0])
                    gl = abs(np.sum(tr[tr < 0]))
                    pf = gp / gl if gl > 0 else 999.0
                    cum = np.cumsum(tr)
                    dd = (np.maximum.accumulate(cum) - cum).max() * 0.20
                    results.append((wr, pf, n, b_cnt, s_cnt, dd, spacing, sl_atr, tp_atr, be_trig, be_lock))

results.sort(key=lambda x: (x[1] * x[0], x[1]), reverse=True)
print("Top 20 configurations ranked by Quality (WR * PF):")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Trades/d':<10} {'B/S':<12} {'DD':<6} {'Sp':<4} {'SL':<5} {'TP':<5} {'BE_trig':<8} {'BE_lock':<8}")
print("-" * 90)
for wr, pf, n, b, s, dd, sp, sl, tp, bet, bel in results[:20]:
    bs_str = f"{b}/{s}"
    print(f"{wr:5.2f}%   {pf:5.2f} {n:<8} {n/7.0:<10.1f} {bs_str:<12} {dd:4.2f}% {sp:<4} {sl:<5.1f} {tp:<5.1f} {bet:<8.2f} {bel:<8.2f}")
