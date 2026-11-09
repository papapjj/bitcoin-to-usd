import pandas as pd, numpy as np

df = pd.read_csv("outputs/btcusd_m1_7day_with_warmup.csv", sep="\t")
closes = df["<CLOSE>"].values
highs = df["<HIGH>"].values
lows = df["<LOW>"].values
opens = df["<OPEN>"].values
N = len(closes)

tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
atrs = np.zeros(N)
atrs[14:] = pd.Series(tr).rolling(14).mean().values[13:]
atrs[:14] = atrs[14]

s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

def test_alpha_geometry(sl_m, tp_m, be_trig=None, be_lock=None, trailing_m=None):
    trades = []
    last_exit = -999
    
    for t in range(50, min(N-50, 5000)):
        if t <= last_exit: continue
        
        atr = atrs[t]
        cp = closes[t]
        
        side = 0
        if ema20[t] > ema50[t] and cp > ema50[t]:
            if lows[t] <= ema20[t] + 0.5 * atr and closes[t] >= opens[t]:
                side = 1
        elif ema20[t] < ema50[t] and cp < ema50[t]:
            if highs[t] >= ema20[t] - 0.5 * atr and closes[t] <= opens[t]:
                side = -1
                
        if side == 0: continue
        
        sl_d = sl_m * atr
        tp_d = tp_m * atr
        cur_sl = cp - sl_d if side == 1 else cp + sl_d
        tp_p = cp + tp_d if side == 1 else cp - tp_d
        be_active = False
        
        for k in range(t+1, min(t+60, N)):
            if be_trig and not be_active:
                if side == 1 and highs[k] >= cp + be_trig * atr:
                    be_active = True
                    cur_sl = max(cur_sl, cp + be_lock * atr)
                elif side == -1 and lows[k] <= cp - be_trig * atr:
                    be_active = True
                    cur_sl = min(cur_sl, cp - be_lock * atr)
                    
            if trailing_m and be_active:
                if side == 1 and highs[k] - cp > trailing_m * atr:
                    cur_sl = max(cur_sl, highs[k] - trailing_m * atr)
                elif side == -1 and cp - lows[k] > trailing_m * atr:
                    cur_sl = min(cur_sl, lows[k] + trailing_m * atr)
            
            if side == 1:
                if lows[k] <= cur_sl:
                    pnl = cur_sl - cp
                    last_exit = k
                    break
                elif highs[k] >= tp_p:
                    pnl = tp_d
                    last_exit = k
                    break
            else:
                if highs[k] >= cur_sl:
                    pnl = cp - cur_sl
                    last_exit = k
                    break
                elif lows[k] <= tp_p:
                    pnl = tp_d
                    last_exit = k
                    break
        else:
            pnl = (closes[min(t+59, N-1)] - cp) * side
            last_exit = min(t+59, N-1)
            
        trades.append(pnl)
        
    tr = np.array(trades)
    if len(tr) == 0: return 0, 0, 0, 0
    n = len(tr)
    wins = np.sum(tr > 0)
    wr = wins / n * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    return n, round(wr, 2), round(pf, 2), round(np.sum(tr), 2)

print(f"{'Configuration':<52} | {'Trades':<6} | {'WR':<7} | {'PF':<6} | {'Net PnL'}")
print("-" * 85)
geos = [
    ("Old Geometry: SL 2.0, TP 5.0, No BE", 2.0, 5.0, None, None, None),
    ("Geometry 1: SL 1.5, TP 1.5, No BE (1:1 RR)", 1.5, 1.5, None, None, None),
    ("Geometry 2: SL 1.5, TP 2.0, No BE (1:1.33 RR)", 1.5, 2.0, None, None, None),
    ("Geometry 3: SL 1.8, TP 1.5, BE (0.8 -> 0.3)", 1.8, 1.5, 0.8, 0.3, None),
    ("Geometry 4: SL 1.8, TP 2.2, BE (1.0 -> 0.5)", 1.8, 2.2, 1.0, 0.5, None),
    ("Geometry 5: SL 2.0, TP 2.5, BE (1.2 -> 0.6) + Trail 1.0", 2.0, 2.5, 1.2, 0.6, 1.0),
    ("Geometry 6: SL 2.5, TP 3.0, BE (1.5 -> 0.8) + Trail 1.2", 2.5, 3.0, 1.5, 0.8, 1.2),
    ("Geometry 7: SL 2.0, TP 1.8, BE (0.9 -> 0.45) + Trail 0.8", 2.0, 1.8, 0.9, 0.45, 0.8),
]

for name, sl, tp, bet, bel, trl in geos:
    n, wr, pf, tot = test_alpha_geometry(sl, tp, bet, bel, trl)
    print(f"{name:<52} | {n:<6} | {wr:5.2f}% | {pf:5.2f} | {tot:+,.0f} USD")
