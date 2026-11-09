import sys, os
import numpy as np
import pandas as pd

df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
closes = df['<CLOSE>'].values.astype(np.float64)
highs = df['<HIGH>'].values.astype(np.float64)
lows = df['<LOW>'].values.astype(np.float64)
opens = df['<OPEN>'].values.astype(np.float64)
N = len(closes)

tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
tr = np.insert(tr, 0, highs[0] - lows[0])
atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

REAL_SPREAD = 28.98

print(f"Total M1 bars: {N}. Testing empirical win rates across horizons and geometries...")
print(f"{'SL_ATR':<8} {'TP_ATR':<8} {'Horizon':<8} | {'BUY WR':<10} {'SELL WR':<10} {'Total Bars'}")
print("-" * 55)

for horizon in [30, 60, 120]:
    for sl_mult in [3.0, 4.0, 5.0]:
        for tp_mult in [1.5, 2.0, 3.0, 4.0]:
            b_wins, b_total = 0, 0
            s_wins, s_total = 0, 0
            
            for t in range(50, N - horizon):
                cur_atr = max(atr[t], 10.0)
                cp = closes[t]
                
                # BUY: entry Ask = cp + SPREAD/2, exit at Bid
                entry_buy = cp + REAL_SPREAD * 0.5
                sl_buy = cp - sl_mult * cur_atr
                tp_buy = cp + tp_mult * cur_atr
                
                # Check BUY outcome
                fwd_h = highs[t+1 : t+1+horizon]
                fwd_l = lows[t+1 : t+1+horizon]
                
                for k in range(len(fwd_h)):
                    if fwd_l[k] <= sl_buy:
                        b_total += 1
                        break
                    elif fwd_h[k] >= tp_buy:
                        b_wins += 1
                        b_total += 1
                        break
                else:
                    b_total += 1
                    if closes[t+horizon] >= entry_buy:
                        b_wins += 1
                        
                # SELL: entry Bid = cp - SPREAD/2, exit at Ask
                entry_sell = cp - REAL_SPREAD * 0.5
                sl_sell = cp + sl_mult * cur_atr
                tp_sell = cp - tp_mult * cur_atr
                
                for k in range(len(fwd_l)):
                    if fwd_h[k] + REAL_SPREAD >= sl_sell:
                        s_total += 1
                        break
                    elif fwd_l[k] + REAL_SPREAD <= tp_sell:
                        s_wins += 1
                        s_total += 1
                        break
                else:
                    s_total += 1
                    if entry_sell >= (closes[t+horizon] + REAL_SPREAD):
                        s_wins += 1
                        
            b_wr = b_wins / b_total * 100.0 if b_total > 0 else 0
            s_wr = s_wins / s_total * 100.0 if s_total > 0 else 0
            print(f"{sl_mult:<8.1f} {tp_mult:<8.1f} {horizon:<8d} | {b_wr:6.2f}%    {s_wr:6.2f}%    {b_total}")
