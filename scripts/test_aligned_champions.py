import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd, numpy as np
from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)
from research_reasoning_world_class.src.bilateral_dual_cot import extract_canonical_bilateral_ingress_233
from sklearn.ensemble import HistGradientBoostingClassifier

print("Loading dataset...", flush=True)
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df_renamed = df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume', '<VOL>': 'RealVolume', '<SPREAD>': 'Spread'
})
df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']

feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
N_total = len(feat_df)
print(f"feat_df has {N_total} bars.", flush=True)

# 137D Canonical features
m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
slots = np.zeros((N_total, 40), dtype=np.float32)
account = np.zeros((N_total, 8), dtype=np.float32)
m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
execution = np.zeros((N_total, 11), dtype=np.float32)
timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

closes = feat_df['Close'].values.astype(np.float64)
highs = feat_df['High'].values.astype(np.float64)
lows = feat_df['Low'].values.astype(np.float64)
opens = feat_df['Open'].values.astype(np.float64)
atrs = feat_df['atr'].values.astype(np.float64)

lookback_bars = 60
horizon_bars = 20
start_idx = lookback_bars
end_idx = N_total - horizon_bars
valid_count = end_idx - start_idx

ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
targets = np.zeros(valid_count, dtype=np.int64)

for i, t in enumerate(range(start_idx, end_idx)):
    cur_p = closes[t]
    cur_atr = max(atrs[t], 1.0)
    m1_bars_60 = np.column_stack([
        opens[t-lookback_bars:t],
        highs[t-lookback_bars:t],
        lows[t-lookback_bars:t],
        closes[t-lookback_bars:t],
    ])
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(base_137[t], m1_bars_60, cur_p, cur_atr)
    
    # Ground truth targets
    fwd_h = highs[t+1 : t+1+horizon_bars]
    fwd_l = lows[t+1 : t+1+horizon_bars]
    
    # Anti-MAE sniper condition
    buy_ok = False
    for k in range(len(fwd_h)):
        if (cur_p - fwd_l[k]) >= 0.35 * cur_atr: break
        if (fwd_h[k] - cur_p) >= 1.00 * cur_atr: buy_ok = True; break
        
    sell_ok = False
    for k in range(len(fwd_h)):
        if (fwd_h[k] - cur_p) >= 0.35 * cur_atr: break
        if (cur_p - fwd_l[k]) >= 1.00 * cur_atr: sell_ok = True; break
        
    recent_h = np.max(highs[t-lookback_bars:t])
    recent_l = np.min(lows[t-lookback_bars:t])
    rng = max(1.0, recent_h - recent_l)
    rel_pos = (cur_p - recent_l) / rng
    if rel_pos <= 0.15: sell_ok = False
    if rel_pos >= 0.85: buy_ok = False
    
    if buy_ok and not sell_ok: targets[i] = 1
    elif sell_ok and not buy_ok: targets[i] = -1

# Arrays for simulation: exactly aligned with ingress_matrix index i!
sim_opens = opens[start_idx:end_idx]
sim_highs = highs[start_idx:end_idx]
sim_lows = lows[start_idx:end_idx]
sim_closes = closes[start_idx:end_idx]
sim_atrs = atrs[start_idx:end_idx]

print(f"Extracted {valid_count} aligned samples. BUY: {np.sum(targets==1)}, SELL: {np.sum(targets==-1)}", flush=True)

# Train/Val split: 70% train, 30% out-of-sample test
split = int(0.70 * valid_count)
X_train, y_bull_train, y_bear_train = ingress_matrix[:split], (targets[:split] == 1).astype(int), (targets[:split] == -1).astype(int)
X_val = ingress_matrix[split:]

print("Training Bull Champion GBDT on 70% train data...", flush=True)
clf_bull = HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=20, random_state=42)
clf_bull.fit(X_train, y_bull_train)

print("Training Bear Champion GBDT on 70% train data...", flush=True)
clf_bear = HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=20, random_state=42)
clf_bear.fit(X_train, y_bear_train)

p_bull_val = clf_bull.predict_proba(X_val)[:, 1]
p_bear_val = clf_bear.predict_proba(X_val)[:, 1]

# Out-of-sample simulation
val_o = sim_opens[split:]
val_h = sim_highs[split:]
val_l = sim_lows[split:]
val_c = sim_closes[split:]
val_atr = sim_atrs[split:]
N_val = len(val_c)
val_days = N_val / 1440.0

print(f"\nSimulating {N_val} out-of-sample bars ({val_days:.1f} days)...", flush=True)
print(f"{'Prob':<6} {'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'GrossWin':<10} {'GrossLoss':<10} {'NetPnL':<10}")
print("-" * 80)

for pb_thresh in [0.28, 0.30, 0.32, 0.35, 0.40]:
    trades = []
    last_t = -999
    b_cnt, s_cnt = 0, 0
    
    for t in range(N_val - 35):
        if t - last_t < 2: continue
        pb = p_bull_val[t]
        pr = p_bear_val[t]
        
        act = 0
        if pb >= pb_thresh and pb > pr + 0.05: act = 1
        elif pr >= pb_thresh and pr > pb + 0.05: act = -1
        
        if act == 0: continue
        last_t = t
        if act == 1: b_cnt += 1
        else: s_cnt += 1
        
        cp = val_c[t]
        cur_atr = val_atr[t]
        sl_d = 1.50 * cur_atr
        tp_d = 2.00 * cur_atr
        bet_d = 0.80 * cur_atr
        bel_d = 0.40 * cur_atr
        
        pnl = 0.0
        if act == 1:
            cur_sl = cp - sl_d
            tp_p = cp + tp_d
            be_active = False
            for k in range(t + 1, min(t + 35, N_val)):
                if not be_active and val_h[k] >= cp + bet_d:
                    be_active = True
                    cur_sl = max(cur_sl, cp + bel_d)
                if be_active:
                    cur_sl = max(cur_sl, val_h[k] - 0.70 * cur_atr)
                if val_l[k] <= cur_sl: pnl = cur_sl - cp; break
                elif val_h[k] >= tp_p: pnl = tp_d; break
            else: pnl = val_c[min(t + 34, N_val - 1)] - cp
        else:
            cur_sl = cp + sl_d
            tp_p = cp - tp_d
            be_active = False
            for k in range(t + 1, min(t + 35, N_val)):
                if not be_active and val_l[k] <= cp - bet_d:
                    be_active = True
                    cur_sl = min(cur_sl, cp - bel_d)
                if be_active:
                    cur_sl = min(cur_sl, val_l[k] + 0.70 * cur_atr)
                if val_h[k] >= cur_sl: pnl = cp - cur_sl; break
                elif val_l[k] <= tp_p: pnl = tp_d; break
            else: pnl = cp - val_c[min(t + 34, N_val - 1)]
        trades.append(pnl)
        
    tr = np.array(trades)
    n = len(tr)
    if n == 0: continue
    wins = np.sum(tr > 0)
    wr = wins / n * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    net = gp - gl
    bs_str = f"{b_cnt}/{s_cnt}"
    print(f"{pb_thresh:<6.2f} {wr:5.2f}%   {pf:5.2f}  {n:<8} {n/val_days:<8.1f} {bs_str:<12} {gp:<10.1f} {gl:<10.1f} {net:+10.1f}")
