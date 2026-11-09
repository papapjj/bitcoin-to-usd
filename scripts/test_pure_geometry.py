import sys, os
sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
opens = df['<OPEN>'].values if '<OPEN>' in df.columns else df['open'].values
highs = df['<HIGH>'].values if '<HIGH>' in df.columns else df['high'].values
lows = df['<LOW>'].values if '<LOW>' in df.columns else df['low'].values
closes = df['<CLOSE>'].values if '<CLOSE>' in df.columns else df['close'].values
N = len(closes)

tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
atr = np.zeros(N)
atr[0] = tr[0]
for i in range(1, len(tr)):
    atr[i] = 0.95 * atr[i-1] + 0.05 * tr[i]
atr = np.maximum(atr, 5.0)

s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

s_high = pd.Series(highs)
s_low = pd.Series(lows)
h60 = s_high.rolling(60, min_periods=1).max().values
l60 = s_low.rolling(60, min_periods=1).min().values
rng60 = np.maximum(1.0, h60 - l60)
rel_pos_60 = (closes - l60) / rng60

# Candlestick pattern
# Bull: Trend up (ema20 > ema50 and cp > ema50), low reached ema20, close <= ema20 + 0.35 ATR, green candle, rel_pos <= 0.85
# Bear: Trend down (ema20 < ema50 and cp < ema50), high reached ema20, close >= ema20 - 0.35 ATR, red candle, rel_pos >= 0.15
cand_signals = []
for t in range(60, N - 35):
    cur_atr = atr[t]
    cp = closes[t]
    e20 = ema20[t]
    e50 = ema50[t]
    rp = rel_pos_60[t]
    
    up = e20 > e50 and cp > e50
    down = e20 < e50 and cp < e50
    
    act = 0
    if up and lows[t] <= e20 + 0.35 * cur_atr and cp <= e20 + 0.50 * cur_atr and rp <= 0.85:
        if closes[t] >= opens[t] or (closes[t] - lows[t]) >= 0.20 * cur_atr:
            act = 1
    elif down and highs[t] >= e20 - 0.35 * cur_atr and cp >= e20 - 0.50 * cur_atr and rp >= 0.15:
        if closes[t] <= opens[t] or (highs[t] - closes[t]) >= 0.20 * cur_atr:
            act = -1
            
    if act != 0:
        cand_signals.append((t, act, cp, cur_atr))

print(f"Total candidate pullback signals: {len(cand_signals)}", flush=True)

# Grid search pure geometry
results = []
for sl_m in [0.75, 1.0, 1.25, 1.5, 2.0]:
    for tp_m in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
        trades = []
        last_entry = -999
        b_cnt, s_cnt = 0, 0
        tp_cnt, sl_cnt = 0, 0
        
        for t, act, cp, cur_atr in cand_signals:
            if t - last_entry < 2: continue
            last_entry = t
            if act == 1: b_cnt += 1
            else: s_cnt += 1
            
            sl_d = sl_m * cur_atr
            tp_d = tp_m * cur_atr
            
            pnl = 0.0
            if act == 1:
                sl_p = cp - sl_d
                tp_p = cp + tp_d
                for k in range(t + 1, min(t + 35, N)):
                    if highs[k] >= tp_p:
                        pnl = tp_d
                        tp_cnt += 1
                        break
                    elif lows[k] <= sl_p:
                        pnl = -sl_d
                        sl_cnt += 1
                        break
                else:
                    pnl = closes[min(t + 34, N - 1)] - cp
            else:
                sl_p = cp + sl_d
                tp_p = cp - tp_d
                for k in range(t + 1, min(t + 35, N)):
                    if lows[k] <= tp_p:
                        pnl = tp_d
                        tp_cnt += 1
                        break
                    elif highs[k] >= sl_p:
                        pnl = -sl_d
                        sl_cnt += 1
                        break
                else:
                    pnl = cp - closes[min(t + 34, N - 1)]
            trades.append(pnl / cur_atr)
            
        tr = np.array(trades)
        wins = np.sum(tr > 0)
        wr = wins / len(tr) * 100.0
        gp = np.sum(tr[tr > 0])
        gl = abs(np.sum(tr[tr < 0]))
        pf = gp / gl if gl > 0 else 999.0
        cum = np.cumsum(tr)
        dd = (np.maximum.accumulate(cum) - cum).max() / max(1.0, cum.max()) * 100.0 if cum.max() > 0 else 99.0
        results.append((wr, pf, len(tr), b_cnt, s_cnt, dd, tp_cnt, sl_cnt, cum[-1], sl_m, tp_m))

results.sort(key=lambda x: (x[1] >= 1.5, x[0] >= 60.0, x[1]), reverse=True)

print(f"\nTop Pure Geometry Configurations (Without Choking Breakeven):")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/SL':<12} {'DD':<6} {'Net_R':<8} {'SL':<5} {'TP':<5}")
print("-" * 80)
for wr, pf, n, b, s, dd, tp, sl, net_r, sl_m, tp_m in results[:20]:
    bs = f"{b}/{s}"
    hits = f"{tp}/{sl}"
    print(f"{wr:5.2f}%   {pf:5.2f}  {n:<8} {n/7.0:<8.1f} {bs:<12} {hits:<12} {dd:4.1f}% {net_r:+8.1f} {sl_m:<5.2f} {tp_m:<5.2f}")
