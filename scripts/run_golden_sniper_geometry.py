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
for i in range(1, len(tr)): atr[i] = 0.95 * atr[i-1] + 0.05 * tr[i]
atr = np.maximum(atr, 5.0)

s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

# Candidate entries
entries = []
last_t = -999
for t in range(60, N - 40):
    if t - last_t < 2: continue
    cur_atr = atr[t]
    cp = closes[t]
    h60, l60 = np.max(highs[t-60:t]), np.min(lows[t-60:t])
    rel_pos = (cp - l60) / max(1.0, h60 - l60)
    
    up = ema20[t] > ema50[t] and cp > ema50[t]
    buy_pb = lows[t] <= ema20[t] + 0.35 * cur_atr and cp >= ema20[t] - 0.50 * cur_atr
    buy_trig = (closes[t] >= opens[t]) or ((closes[t] - lows[t]) >= 0.20 * cur_atr)
    anti_top = rel_pos <= 0.85
    
    down = ema20[t] < ema50[t] and cp < ema50[t]
    sell_rally = highs[t] >= ema20[t] - 0.35 * cur_atr and cp <= ema20[t] + 0.50 * cur_atr
    sell_trig = (closes[t] <= opens[t]) or ((highs[t] - closes[t]) >= 0.20 * cur_atr)
    anti_bot = rel_pos >= 0.15
    
    act = 0
    if up and buy_pb and buy_trig and anti_top: act = 1
    elif down and sell_rally and sell_trig and anti_bot: act = -1
    if act == 0: continue
    last_t = t
    entries.append((t, act, cp, cur_atr))

print(f"Precalculated {len(entries)} candidate entries. Evaluating geometry matrix...", flush=True)

results = []
for sl_m in [1.50, 2.00, 2.50, 3.00, 3.50]:
    for tp_m in [1.00, 1.20, 1.50, 1.80, 2.00, 2.50]:
        for bet_m in [0.50, 0.60, 0.70, 0.80]:
            for bel_m in [0.25, 0.35, 0.45]:
                if bel_m >= bet_m: continue
                trades = []
                tp_hits = 0
                be_hits = 0
                sl_hits = 0
                b_cnt = 0
                s_cnt = 0
                
                for t, act, cp, cur_atr in entries:
                    if act == 1: b_cnt += 1
                    else: s_cnt += 1
                    sl_d = sl_m * cur_atr
                    tp_d = tp_m * cur_atr
                    bet_d = bet_m * cur_atr
                    bel_d = bel_m * cur_atr
                    
                    pnl = 0.0
                    if act == 1:
                        cur_sl = cp - sl_d
                        tp_p = cp + tp_d
                        be_active = False
                        for k in range(t + 1, min(t + 40, N)):
                            if highs[k] >= tp_p:
                                pnl = tp_d
                                tp_hits += 1
                                break
                            if not be_active and highs[k] >= cp + bet_d:
                                be_active = True
                                cur_sl = max(cur_sl, cp + bel_d)
                            if lows[k] <= cur_sl:
                                pnl = cur_sl - cp
                                if be_active: be_hits += 1
                                else: sl_hits += 1
                                break
                        else:
                            pnl = closes[min(t + 39, N - 1)] - cp
                    else:
                        cur_sl = cp + sl_d
                        tp_p = cp - tp_d
                        be_active = False
                        for k in range(t + 1, min(t + 40, N)):
                            if lows[k] <= tp_p:
                                pnl = tp_d
                                tp_hits += 1
                                break
                            if not be_active and lows[k] <= cp - bet_d:
                                be_active = True
                                cur_sl = min(cur_sl, cp - bel_d)
                            if highs[k] >= cur_sl:
                                pnl = cp - cur_sl
                                if be_active: be_hits += 1
                                else: sl_hits += 1
                                break
                        else:
                            pnl = cp - closes[min(t + 39, N - 1)]
                    trades.append(pnl)
                    
                tr = np.array(trades)
                wins = np.sum(tr > 0)
                wr = wins / len(tr) * 100.0
                gp = np.sum(tr[tr > 0])
                gl = abs(np.sum(tr[tr < 0]))
                pf = gp / gl if gl > 0 else 999.0
                cum = np.cumsum(tr)
                dd = (np.maximum.accumulate(cum) - cum).max() / (cum.max() + 1e-4) * 100.0 if cum.max() > 0 else 99.0
                results.append((wr, pf, len(tr), b_cnt, s_cnt, tp_hits, be_hits, sl_hits, dd, sl_m, tp_m, bet_m, bel_m))

results.sort(key=lambda x: (x[0] >= 80.0, x[1] >= 2.0, x[1]), reverse=True)

print(f"\nTop Configurations:")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/BE/SL':<16} {'DD':<6} {'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6}")
print("-" * 90)
for wr, pf, n, b, s, tp_h, be_h, sl_h, dd, sl, tp, bet, bel in results[:25]:
    bs_str = f"{b}/{s}"
    hits_str = f"{tp_h}/{be_h}/{sl_h}"
    print(f"{wr:5.2f}%   {pf:5.2f}  {n:<8} {n/7.0:<8.1f} {bs_str:<12} {hits_str:<16} {dd:4.1f}% {sl:<5.2f} {tp:<5.2f} {bet:<6.2f} {bel:<6.2f}")
