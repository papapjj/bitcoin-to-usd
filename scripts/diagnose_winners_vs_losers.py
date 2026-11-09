import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import pandas as pd
import torch

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)
from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)

# 1. Ingest Data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df_renamed = df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume', '<VOL>': 'RealVolume', '<SPREAD>': 'Spread'
})
if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
    df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']

feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
N_total = len(feat_df)

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

s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values
ema200 = s_c.ewm(span=200).mean().values

s_h = pd.Series(highs)
s_l = pd.Series(lows)
rolling_h60 = s_h.rolling(60, min_periods=1).max().values
rolling_l60 = s_l.rolling(60, min_periods=1).min().values
rng60 = np.maximum(1.0, rolling_h60 - rolling_l60)
rel_pos_60 = (closes - rolling_l60) / rng60

lookback = 60
horizon = 25
start_idx = lookback
end_idx = N_total - horizon
valid_count = end_idx - start_idx

ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
for i, t in enumerate(range(start_idx, end_idx)):
    cur_p = closes[t]
    cur_atr = max(atrs[t], 1.0)
    m1_bars_60 = np.column_stack([
        opens[t-lookback:t],
        highs[t-lookback:t],
        lows[t-lookback:t],
        closes[t-lookback:t],
    ])
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(base_137[t], m1_bars_60, cur_p, cur_atr)

config = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256, num_hops=4, num_debate_turns=3,
    prm_threshold=0.32, trap_threshold=0.60
)
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
engine.eval()

with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=512)
    actions = verdict.actions.numpy()
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()
    confidences = verdict.confidences.numpy()

val_closes = closes[start_idx:end_idx]
val_highs = highs[start_idx:end_idx]
val_lows = lows[start_idx:end_idx]
val_opens = opens[start_idx:end_idx]
val_atrs = atrs[start_idx:end_idx]
val_ema20 = ema20[start_idx:end_idx]
val_ema50 = ema50[start_idx:end_idx]
val_ema200 = ema200[start_idx:end_idx]
val_rp = rel_pos_60[start_idx:end_idx]

# Let's collect trade details
sl_atr = 1.5
tp_atr = 1.2
max_bars = 30

trade_records = []
last_entry = -999
last_exit = -999

for t in range(valid_count - max_bars):
    if t - last_entry < 2: continue
    if t - last_exit < 3: continue
    
    act = int(actions[t])
    if act == 0: continue
    
    cp = val_closes[t]
    cur_atr = val_atrs[t]
    e20 = val_ema20[t]
    e50 = val_ema50[t]
    e200 = val_ema200[t]
    rp = val_rp[t]
    bs = b_scores[t]
    rs = r_scores[t]
    prm = prm_scores[t]
    conf = confidences[t]
    
    if act == 1:
        if not (e20 > e50 and cp > e50): continue
        if not (val_lows[t] <= e20 + 0.35 * cur_atr and cp <= e20 + 0.50 * cur_atr and rp <= 0.85): continue
        if not (val_closes[t] >= val_opens[t] or (val_closes[t] - val_lows[t]) >= 0.20 * cur_atr): continue
    elif act == -1:
        if not (e20 < e50 and cp < e50): continue
        if not (val_highs[t] >= e20 - 0.35 * cur_atr and cp >= e20 - 0.50 * cur_atr and rp >= 0.15): continue
        if not (val_closes[t] <= val_opens[t] or (val_highs[t] - val_closes[t]) >= 0.20 * cur_atr): continue
        
    last_entry = t
    sl_d = sl_atr * cur_atr
    tp_d = tp_atr * cur_atr
    
    is_win = False
    exit_bar = t + max_bars
    if act == 1:
        cur_sl = cp - sl_d
        tp_p = cp + tp_d
        for k in range(t + 1, min(t + max_bars, valid_count)):
            if val_lows[k] <= cur_sl:
                is_win = False
                exit_bar = k
                break
            elif val_highs[k] >= tp_p:
                is_win = True
                exit_bar = k
                break
    else:
        cur_sl = cp + sl_d
        tp_p = cp - tp_d
        for k in range(t + 1, min(t + max_bars, valid_count)):
            if val_highs[k] >= cur_sl:
                is_win = False
                exit_bar = k
                break
            elif val_lows[k] <= tp_p:
                is_win = True
                exit_bar = k
                break
                
    last_exit = exit_bar
    
    trade_records.append({
        't': t, 'act': act, 'win': is_win,
        'cp': cp, 'atr': cur_atr,
        'diff_20_50': (e20 - e50) / cur_atr,
        'dist_ema200': (cp - e200) / cur_atr,
        'rel_pos_60': rp,
        'bull_score': bs, 'bear_score': rs, 'margin': bs - rs if act == 1 else rs - bs,
        'prm': prm, 'conf': conf,
    })

tdf = pd.DataFrame(trade_records)
print(f"Total trades: {len(tdf)} | Wins: {tdf['win'].sum()} | Losses: {(~tdf['win']).sum()} | WR: {tdf['win'].mean()*100:.2f}%")

winners = tdf[tdf['win']]
losers = tdf[~tdf['win']]

print("\n--- METRIC COMPARISON (WINNERS vs LOSERS) ---")
for col in ['margin', 'prm', 'conf', 'rel_pos_60', 'dist_ema200', 'diff_20_50', 'atr']:
    print(f"{col:<15} | Win Mean: {winners[col].mean():8.4f} | Loss Mean: {losers[col].mean():8.4f}")

# Test filtering by high conviction
print("\n--- PERFORMANCE BY MARGIN THRESHOLDS ---")
for m_th in [0.0, 0.05, 0.10, 0.15, 0.20]:
    for p_th in [0.30, 0.50, 0.70, 0.85]:
        sub = tdf[(tdf['margin'] >= m_th) & (tdf['prm'] >= p_th)]
        if len(sub) >= 50:
            sub_wr = sub['win'].mean() * 100.0
            n_w = sub['win'].sum()
            n_l = len(sub) - n_w
            sub_pf = (n_w * tp_atr) / (n_l * sl_atr) if n_l > 0 else 999.0
            print(f"Margin >= {m_th:<4.2f} & PRM >= {p_th:<4.2f} | N: {len(sub):<4} ({len(sub)/7.0:4.1f}/d) | WR: {sub_wr:5.2f}% | PF: {sub_pf:5.2f}")
