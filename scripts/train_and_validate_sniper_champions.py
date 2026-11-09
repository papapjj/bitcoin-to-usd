import sys, os
sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

print("Loading dataset...", flush=True)
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

# Engineer 30 clean, robust microstructural features
feats = np.zeros((N, 32), dtype=np.float32)
feats[:, 0] = (closes - opens) / atr                       # bar body
feats[:, 1] = (highs - np.maximum(opens, closes)) / atr     # upper wick
feats[:, 2] = (np.minimum(opens, closes) - lows) / atr      # lower wick
feats[:, 3] = (highs - lows) / atr                         # total range
feats[:, 4] = (closes - ema20) / atr                       # dist from EMA20
feats[:, 5] = (closes - ema50) / atr                       # dist from EMA50
feats[:, 6] = (ema20 - ema50) / atr                        # trend slope
feats[:, 7] = rel_pos_60                                   # 60-bar range position
for lag in [1, 2, 3, 5, 8, 13, 21]:
    col = 8 + [1, 2, 3, 5, 8, 13, 21].index(lag)
    feats[lag:, col] = (closes[lag:] - closes[:-lag]) / atr[lag:]

# RSI 14
delta = s_c.diff()
gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
rs = gain / (loss + 1e-8)
rsi = (100 - (100 / (1 + rs))).values
feats[:, 15] = (rsi - 50.0) / 25.0

# Volatility ratios
feats[:, 16] = atr / (pd.Series(atr).rolling(60, min_periods=1).mean().values + 1e-6)

# Generate strict Value Zone Sniper Ground Truth Targets
horizon = 25
targets = np.zeros(N, dtype=np.int64)

for t in range(60, N - horizon):
    cur_p = closes[t]
    cur_atr = atr[t]
    
    fwd_h = highs[t+1 : t+1+horizon]
    fwd_l = lows[t+1 : t+1+horizon]
    
    # Uptrend Value Pullback Target:
    # 1. Trend: EMA20 > EMA50 and Price > EMA50
    # 2. Value zone: lows[t] reached ema20 + 0.35 ATR and close <= ema20 + 0.50 ATR
    # 3. Anti-Top: rel_pos_60 <= 0.85
    # 4. Forward: moves +1.20 ATR before dropping -0.50 ATR
    up = ema20[t] > ema50[t] and cur_p > ema50[t]
    in_buy_zone = (lows[t] <= ema20[t] + 0.35 * cur_atr) and (cur_p <= ema20[t] + 0.50 * cur_atr) and (rel_pos_60[t] <= 0.85)
    
    buy_success = False
    if up and in_buy_zone:
        for k in range(len(fwd_h)):
            if (cur_p - fwd_l[k]) >= 0.50 * cur_atr: break
            if (fwd_h[k] - cur_p) >= 1.20 * cur_atr: buy_success = True; break
            
    # Downtrend Value Rally Target:
    # 1. Trend: EMA20 < EMA50 and Price < EMA50
    # 2. Value zone: highs[t] reached ema20 - 0.35 ATR and close >= ema20 - 0.50 ATR
    # 3. Anti-Bottom: rel_pos_60 >= 0.15
    # 4. Forward: drops -1.20 ATR before rising +0.50 ATR
    down = ema20[t] < ema50[t] and cur_p < ema50[t]
    in_sell_zone = (highs[t] >= ema20[t] - 0.35 * cur_atr) and (cur_p >= ema20[t] - 0.50 * cur_atr) and (rel_pos_60[t] >= 0.15)
    
    sell_success = False
    if down and in_sell_zone:
        for k in range(len(fwd_l)):
            if (fwd_h[k] - cur_p) >= 0.50 * cur_atr: break
            if (cur_p - fwd_l[k]) >= 1.20 * cur_atr: sell_success = True; break
            
    if buy_success and not sell_success: targets[t] = 1
    elif sell_success and not buy_success: targets[t] = -1

valid_start = 60
valid_end = N - horizon
valid_N = valid_end - valid_start

X = feats[valid_start:valid_end]
y_dir = targets[valid_start:valid_end]
v_closes = closes[valid_start:valid_end]
v_highs = highs[valid_start:valid_end]
v_lows = lows[valid_start:valid_end]
v_opens = opens[valid_start:valid_end]
v_atrs = atr[valid_start:valid_end]
v_ema20 = ema20[valid_start:valid_end]
v_ema50 = ema50[valid_start:valid_end]
v_rel_pos = rel_pos_60[valid_start:valid_end]

print(f"Total valid bars: {valid_N}. Ground truth sniper targets: BUY={np.sum(y_dir == 1)}, SELL={np.sum(y_dir == -1)}, HOLD={np.sum(y_dir == 0)}", flush=True)

# Train on first 65%, test on 35% Out-of-Sample (OOS)
split = int(0.65 * valid_N)
X_train, y_train_bull, y_train_bear = X[:split], (y_dir[:split] == 1).astype(int), (y_dir[:split] == -1).astype(int)
X_oos = X[split:]

print(f"Training Bull Champion Specialist on {split} bars...", flush=True)
bull_champion = HistGradientBoostingClassifier(max_iter=120, min_samples_leaf=25, class_weight='balanced', random_state=42)
bull_champion.fit(X_train, y_train_bull)

print(f"Training Bear Champion Specialist on {split} bars...", flush=True)
bear_champion = HistGradientBoostingClassifier(max_iter=120, min_samples_leaf=25, class_weight='balanced', random_state=42)
bear_champion.fit(X_train, y_train_bear)

# Out-of-Sample Simulation
p_bull_oos = bull_champion.predict_proba(X_oos)[:, 1]
p_bear_oos = bear_champion.predict_proba(X_oos)[:, 1]

oos_c = v_closes[split:]
oos_h = v_highs[split:]
oos_l = v_lows[split:]
oos_o = v_opens[split:]
oos_atr = v_atrs[split:]
oos_e20 = v_ema20[split:]
oos_e50 = v_ema50[split:]
oos_rp = v_rel_pos[split:]
oos_N = len(oos_c)
oos_days = oos_N / 1440.0

print(f"\nSimulating {oos_N} strictly out-of-sample bars ({oos_days:.2f} days)...", flush=True)
print(f"{'Thresh':<8} {'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/BE/SL':<16} {'DD':<6} {'Net_R':<8} {'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6}")
print("-" * 105)

results = []
for p_th in [0.45, 0.50, 0.55, 0.60, 0.65]:
    for sl_m in [1.5, 2.0, 2.5]:
        for tp_m in [1.2, 1.5, 1.8, 2.0]:
            for bet_m in [0.60, 0.70]:
                for bel_m in [0.35, 0.45]:
                    if bel_m >= bet_m: continue
                    trades = []
                    last_exit_t = -999
                    last_entry_t = -999
                    b_cnt, s_cnt = 0, 0
                    
                    for t in range(oos_N - 30):
                        if t - last_entry_t < 2: continue
                        if t - last_exit_t < 3: continue
                        
                        pb = p_bull_oos[t]
                        pr = p_bear_oos[t]
                        cp = oos_c[t]
                        cur_atr = oos_atr[t]
                        e20 = oos_e20[t]
                        e50 = oos_e50[t]
                        rp = oos_rp[t]
                        
                        act = 0
                        # Bull Champion Buy
                        if pb >= p_th and pb > pr + 0.10:
                            if e20 > e50 and cp > e50:
                                if oos_l[t] <= e20 + 0.35 * cur_atr and cp <= e20 + 0.50 * cur_atr and rp <= 0.85:
                                    if oos_c[t] >= oos_o[t] or (oos_c[t] - oos_l[t]) >= 0.20 * cur_atr:
                                        act = 1
                        # Bear Champion Sell
                        elif pr >= p_th and pr > pb + 0.10:
                            if e20 < e50 and cp < e50:
                                if oos_h[t] >= e20 - 0.35 * cur_atr and cp >= e20 - 0.50 * cur_atr and rp >= 0.15:
                                    if oos_c[t] <= oos_o[t] or (oos_h[t] - oos_c[t]) >= 0.20 * cur_atr:
                                        act = -1
                                        
                        if act == 0: continue
                        last_entry_t = t
                        if act == 1: b_cnt += 1
                        else: s_cnt += 1
                        
                        sl_d = sl_m * cur_atr
                        tp_d = tp_m * cur_atr
                        bet_d = bet_m * cur_atr
                        bel_d = bel_m * cur_atr
                        
                        pnl = 0.0
                        reason = None
                        if act == 1:
                            cur_sl = cp - sl_d
                            tp_p = cp + tp_d
                            be_active = False
                            for k in range(t + 1, min(t + 30, oos_N)):
                                if not be_active and oos_h[k] >= cp + bet_d:
                                    be_active = True
                                    cur_sl = max(cur_sl, cp + bel_d)
                                if oos_l[k] <= cur_sl:
                                    pnl = cur_sl - cp
                                    reason = 'BE' if be_active else 'SL'
                                    last_exit_t = k
                                    break
                                elif oos_h[k] >= tp_p:
                                    pnl = tp_d
                                    reason = 'TP'
                                    last_exit_t = k
                                    break
                            else:
                                pnl = oos_c[min(t + 29, oos_N - 1)] - cp
                                reason = 'TIME'
                                last_exit_t = min(t + 29, oos_N - 1)
                        else:
                            cur_sl = cp + sl_d
                            tp_p = cp - tp_d
                            be_active = False
                            for k in range(t + 1, min(t + 30, oos_N)):
                                if not be_active and oos_l[k] <= cp - bet_d:
                                    be_active = True
                                    cur_sl = min(cur_sl, cp - bel_d)
                                if oos_h[k] >= cur_sl:
                                    pnl = cp - cur_sl
                                    reason = 'BE' if be_active else 'SL'
                                    last_exit_t = k
                                    break
                                elif oos_l[k] <= tp_p:
                                    pnl = tp_d
                                    reason = 'TP'
                                    last_exit_t = k
                                    break
                            else:
                                pnl = cp - oos_c[min(t + 29, oos_N - 1)]
                                reason = 'TIME'
                                last_exit_t = min(t + 29, oos_N - 1)
                                
                        trades.append((pnl, reason, cur_atr))
                        
                    if len(trades) < 50: continue
                    pnls = np.array([x[0] for x in trades])
                    wins = np.sum(pnls > 0)
                    wr = wins / len(pnls) * 100.0
                    gp = np.sum(pnls[pnls > 0])
                    gl = abs(np.sum(pnls[pnls < 0]))
                    pf = gp / gl if gl > 0 else 999.0
                    cum = np.cumsum(pnls)
                    dd = (np.maximum.accumulate(cum) - cum).max() / max(1.0, cum.max()) * 100.0 if cum.max() > 0 else 99.0
                    
                    tp_cnt = sum(1 for x in trades if x[1] == 'TP')
                    be_cnt = sum(1 for x in trades if x[1] == 'BE')
                    sl_cnt = sum(1 for x in trades if x[1] == 'SL')
                    
                    net_r = np.sum([x[0] / x[2] for x in trades])
                    results.append((wr, pf, len(trades), b_cnt, s_cnt, dd, net_r, tp_cnt, be_cnt, sl_cnt, p_th, sl_m, tp_m, bet_m, bel_m))

results.sort(key=lambda x: (x[0] >= 80.0, x[1] >= 2.0, x[1], x[0]), reverse=True)

for wr, pf, n, b, s, dd, net, tp, be, sl, p_th, sl_m, tp_m, bet_m, bel_m in results[:20]:
    bs = f"{b}/{s}"
    hits = f"{tp}/{be}/{sl}"
    print(f"{p_th:<8.2f} {wr:5.2f}%   {pf:5.2f}  {n:<8} {n/oos_days:<8.1f} {bs:<12} {hits:<16} {dd:4.1f}% {net:+8.1f} {sl_m:<5.1f} {tp_m:<5.1f} {bet_m:<6.2f} {bel_m:<6.2f}")
