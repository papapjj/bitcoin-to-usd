import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd, numpy as np

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

# RSI 14
delta = s_c.diff()
gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
rs = gain / (loss + 1e-8)
rsi = 100 - (100 / (1 + rs)).values

print(f"Loaded {N} bars. Extracting candidate entry bars...", flush=True)

# Precalculate signals for different trend requirements
cand_signals = []
last_t = -999
for t in range(60, N - 40):
    cur_atr = atr[t]
    cp = closes[t]
    
    h60 = np.max(highs[t-60:t])
    l60 = np.min(lows[t-60:t])
    rng = max(1.0, h60 - l60)
    rel_pos = (cp - l60) / rng
    trend_diff = (ema20[t] - ema50[t]) / cur_atr
    
    # Uptrend Pullback
    up = trend_diff >= 0.15 and cp > ema50[t]
    buy_pb = lows[t] <= ema20[t] + 0.35 * cur_atr and cp >= ema20[t] - 0.50 * cur_atr
    buy_trig = (closes[t] >= opens[t]) or ((closes[t] - lows[t]) >= 0.20 * cur_atr)
    anti_top = rel_pos <= 0.85
    buy_rsi = rsi[t] <= 65
    
    # Downtrend Rally
    down = trend_diff <= -0.15 and cp < ema50[t]
    sell_rally = highs[t] >= ema20[t] - 0.35 * cur_atr and cp <= ema20[t] + 0.50 * cur_atr
    sell_trig = (closes[t] <= opens[t]) or ((highs[t] - closes[t]) >= 0.20 * cur_atr)
    anti_bot = rel_pos >= 0.15
    sell_rsi = rsi[t] >= 35
    
    act = 0
    if up and buy_pb and buy_trig and anti_top and buy_rsi:
        act = 1
    elif down and sell_rally and sell_trig and anti_bot and sell_rsi:
        act = -1
        
    if act != 0:
        cand_signals.append((t, act, cp, cur_atr))

print(f"Found {len(cand_signals)} raw candidate signals. Simulating execution geometry...", flush=True)

results = []
for spacing in [2, 3]:
    for sl_m in [1.20, 1.50, 1.80, 2.00]:
        for tp_m in [1.50, 2.00, 2.50]:
            for be_trig_m in [0.60, 0.70, 0.80]:
                for be_lock_m in [0.35, 0.45, 0.55]:
                    if be_lock_m >= be_trig_m: continue
                    trades = []
                    last_bar = -999
                    b_cnt, s_cnt = 0, 0
                    
                    for t, act, cp, cur_atr in cand_signals:
                        if t - last_bar < spacing: continue
                        last_bar = t
                        if act == 1: b_cnt += 1
                        else: s_cnt += 1
                        
                        sl_d = sl_m * cur_atr
                        tp_d = tp_m * cur_atr
                        bet_d = be_trig_m * cur_atr
                        bel_d = be_lock_m * cur_atr
                        
                        pnl = 0.0
                        if act == 1:
                            cur_sl = cp - sl_d
                            tp_p = cp + tp_d
                            be_active = False
                            for k in range(t + 1, min(t + 35, N)):
                                if not be_active and highs[k] >= cp + bet_d:
                                    be_active = True
                                    cur_sl = max(cur_sl, cp + bel_d)
                                if be_active:
                                    cur_sl = max(cur_sl, highs[k] - 0.50 * cur_atr)
                                if lows[k] <= cur_sl:
                                    pnl = cur_sl - cp
                                    break
                                elif highs[k] >= tp_p:
                                    pnl = tp_d
                                    break
                            else:
                                pnl = closes[min(t + 34, N - 1)] - cp
                        else:
                            cur_sl = cp + sl_d
                            tp_p = cp - tp_d
                            be_active = False
                            for k in range(t + 1, min(t + 35, N)):
                                if not be_active and lows[k] <= cp - bet_d:
                                    be_active = True
                                    cur_sl = min(cur_sl, cp - bel_d)
                                if be_active:
                                    cur_sl = min(cur_sl, lows[k] + 0.50 * cur_atr)
                                if highs[k] >= cur_sl:
                                    pnl = cp - cur_sl
                                    break
                                elif lows[k] <= tp_p:
                                    pnl = tp_d
                                    break
                            else:
                                pnl = cp - closes[min(t + 34, N - 1)]
                        trades.append(pnl)
                        
                    tr = np.array(trades)
                    n = len(tr)
                    if n < 400: continue
                    wins = np.sum(tr > 0)
                    wr = wins / n * 100.0
                    gp = np.sum(tr[tr > 0])
                    gl = abs(np.sum(tr[tr < 0]))
                    pf = gp / gl if gl > 0 else 999.0
                    cum = np.cumsum(tr)
                    dd = (np.maximum.accumulate(cum) - cum).max() / (cum.max() + 1e-4) * 100.0 if cum.max() > 0 else 99.0
                    results.append((wr, pf, n, b_cnt, s_cnt, dd, spacing, sl_m, tp_m, be_trig_m, be_lock_m))

results.sort(key=lambda x: (x[0] >= 80.0, x[1]), reverse=True)
print(f"\nTop 20 Configurations:")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'DD':<6} {'Sp':<4} {'SL':<5} {'TP':<5} {'BE_trig':<8} {'BE_lock':<8}")
print("-" * 85)
for wr, pf, n, b, s, dd, sp, sl, tp, bet, bel in results[:20]:
    bs_str = f"{b}/{s}"
    print(f"{wr:5.2f}%   {pf:5.2f}  {n:<8} {n/7.0:<8.1f} {bs_str:<12} {dd:4.1f}% {sp:<4} {sl:<5.2f} {tp:<5.2f} {bet:<8.2f} {bel:<8.2f}")
