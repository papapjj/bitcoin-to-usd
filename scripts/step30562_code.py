python -c "
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from collections import deque
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

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
    cps = verdict.counter_punches.numpy()

s_prices = pd.Series(prices)
ema20 = s_prices.ewm(span=20).mean().values
ema50 = s_prices.ewm(span=50).mean().values

# Pre-extract signals
b_hist, r_hist = deque(maxlen=120), deque(maxlen=120)
raw_signals = []
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
    if not cps[t]:
        if act == 1 and not (closes[t] >= opens[t] or (closes[t] - lows[t]) >= 0.20 * atr): continue
        elif act == -1 and not (closes[t] <= opens[t] or (highs[t] - closes[t]) >= 0.20 * atr): continue
    raw_signals.append((t, act, prices[t], atr))

print(f'Total candidate signals: {len(raw_signals)}')

results = []
for spacing in [2, 3]:
    for sl_atr in [1.5, 2.0, 2.5, 3.0]:
        for tp_atr in [1.5, 2.0, 2.5, 3.0]:
            for be_trig in [0.50, 0.60, 0.70, 0.80]:
                for be_lock in [0.25, 0.35, 0.45]:
                    if be_lock >= be_trig: continue
                    for trail_d in [0.50, 0.60, 0.75, 1.00]:
                        trades = []
                        last_t = -999
                        b_cnt, s_cnt = 0, 0
                        for t, act, cur_p, atr in raw_signals:
                            if t - last_t < spacing: continue
                            last_t = t
                            if act == 1: b_cnt += 1
                            else: s_cnt += 1
                            sl_d = sl_atr * atr
                            tp_d = tp_atr * atr
                            bet_d = be_trig * atr
                            bel_d = be_lock * atr
                            tr_d = trail_d * atr
                            
                            pnl = 0.0
                            if act == 1:
                                dyn_sl = cur_p - sl_d
                                tp_p = cur_p + tp_d
                                be_arm = False
                                for k in range(t + 1, min(t + 35, N)):
                                    bh, bl, bc = highs[k], lows[k], closes[k]
                                    if not be_arm and bh >= cur_p + bet_d:
                                        be_arm = True
                                        dyn_sl = max(dyn_sl, cur_p + bel_d)
                                    if be_arm: dyn_sl = max(dyn_sl, bh - tr_d)
                                    if bl <= dyn_sl: pnl = dyn_sl - cur_p; break
                                    elif bh >= tp_p: pnl = tp_d; break
                                else: pnl = bc - cur_p
                            else:
                                dyn_sl = cur_p + sl_d
                                tp_p = cur_p - tp_d
                                be_arm = False
                                for k in range(t + 1, min(t + 35, N)):
                                    bh, bl, bc = highs[k], lows[k], closes[k]
                                    if not be_arm and bl <= cur_p - bet_d:
                                        be_arm = True
                                        dyn_sl = min(dyn_sl, cur_p - bel_d)
                                    if be_arm: dyn_sl = min(dyn_sl, bl + tr_d)
                                    if bh >= dyn_sl: pnl = cur_p - dyn_sl; break
                                    elif bl <= tp_p: pnl = tp_d; break
                                else: pnl = cur_p - bc
                            trades.append(pnl / sl_d)
                        tr = np.array(trades)
                        wins = np.sum(tr > 0)
                        wr = wins / len(tr) * 100.0
                        gp = np.sum(tr[tr > 0])
                        gl = abs(np.sum(tr[tr < 0]))
                        pf = gp / gl if gl > 0 else 999.0
                        cum = np.cumsum(tr)
                        dd = (np.maximum.accumulate(cum) - cum).max() / (cum.max() + 1e-4) * 100.0 if cum.max() > 0 else 99.0
                        results.append((wr, pf, len(tr), b_cnt, s_cnt, dd, gp-gl, spacing, sl_atr, tp_atr, be_trig, be_lock, trail_d))

results.sort(key=lambda x: (x[0] >= 80.0, x[1] >= 2.5, x[1], x[0]), reverse=True)
print('\nTop 25 Configurations:')
print('WR       PF     Trades   Tr/d     B/S          DD     Net(R)   Sp   SL    TP    BE_t   BE_l   Trail')
print('-' * 95)
for wr, pf, n, b, s, dd, net, sp, sl, tp, bet, bel, trl in results[:25]:
    bs_str = f'{b}/{s}'
    print(f'{wr:5.2f}%   {pf:5.2f}  {n:<8} {n/7.0:<8.1f} {bs_str:<12} {dd:4.1f}% {net:+8.1f} {sp:<4} {sl:<5.2f} {tp:<5.2f} {bet:<6.2f} {bel:<6.2f} {trl:<5.2f}')
"