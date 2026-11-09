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
dts = (df['<DATE>'] + ' ' + df['<TIME>']).values[-N:] if '<DATE>' in df.columns else np.arange(N)

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

# Check bars corresponding to live MT5 trades
check_times = [
    '2025.02.20 04:35:00', # Trade 7/8 entry
    '2025.02.20 07:08:00', # Trade 17 entry
    '2025.02.20 07:13:00', # Trade 18 entry
    '2025.02.20 09:42:00', # Trade 23 entry
    '2025.02.20 10:32:00', # Trade 27 entry
    '2025.02.20 11:08:00', # Trade 29 entry
]

print("Checking what Step 30562 signals were at MT5 losing trade times:")
b_hist, r_hist = deque(maxlen=120), deque(maxlen=120)
for t in range(120, N):
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
    
    t_str = str(dts[t])
    for ct in check_times:
        if ct[:15] in t_str:
            atr = atrs[t]
            h60, l60 = np.max(highs[max(0, t-60):t]), np.min(lows[max(0, t-60):t])
            rel_pos = (prices[t] - l60) / max(1.0, h60 - l60)
            dist_e20 = (prices[t] - ema20[t]) / atr
            print(f"Time: {t_str} | Price: {prices[t]:.2f} | Act: {act} (bz={bz:.2f}, rz={rz:.2f}) | RelPos60: {rel_pos:.2f} | DistEMA20: {dist_e20:+.2f} ATR")
