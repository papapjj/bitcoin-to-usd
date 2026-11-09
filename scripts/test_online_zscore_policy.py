import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from collections import deque
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

print("=" * 80)
print("TESTING ONLINE ROLLING Z-SCORE STANDARDIZATION & DYNAMIC PULLBACK GEOMETRY")
print("=" * 80)

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
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    cps = verdict.counter_punches.numpy()

# Online Simulation with 120-bar rolling buffer
b_hist = deque(maxlen=120)
r_hist = deque(maxlen=120)

trades = []
b_cnt = 0
s_cnt = 0
last_trade_bar = -999

for t in range(50, N - 30):
    b_s = float(b_scores[t])
    r_s = float(r_scores[t])
    b_hist.append(b_s)
    r_hist.append(r_s)
    
    if len(b_hist) < 20:
        continue
        
    mean_b = np.mean(b_hist)
    std_b = max(0.01, np.std(b_hist))
    mean_r = np.mean(r_hist)
    std_r = max(0.01, np.std(r_hist))
    
    b_z = (b_s - mean_b) / std_b
    r_z = (r_s - mean_r) / std_r
    
    # Symmetrical Decision
    net_dir = 0
    if cps[t]:
        # Counter punch
        net_dir = int(verdict.actions[t])
    elif b_z >= 0.20 and (b_z - r_z) >= 0.15:
        net_dir = 1
    elif r_z >= 0.20 and (r_z - b_z) >= 0.15:
        net_dir = -1
        
    if net_dir == 0:
        continue
        
    cur_p = prices[t]
    atr = atrs[t]
    
    # Dynamic Pullback Trend Guard
    ma20 = np.mean(prices[t-20:t])
    ma50 = np.mean(prices[t-50:t])
    if ma20 > ma50:
        if net_dir == -1 and not cps[t]:
            continue
        elif net_dir == 1 and cur_p < ma50 - 0.50 * atr:
            continue
    elif ma20 < ma50:
        if net_dir == 1 and not cps[t]:
            continue
        elif net_dir == -1 and cur_p > ma50 + 0.50 * atr:
            continue
            
    # Sniper Timing Confirmation
    if not cps[t]:
        if net_dir == 1:
            candle_green = closes[t] >= opens[t]
            lower_wick = closes[t] - lows[t]
            if not (candle_green or lower_wick >= 0.20 * atr):
                continue
        elif net_dir == -1:
            candle_red = closes[t] <= opens[t]
            upper_wick = highs[t] - closes[t]
            if not (candle_red or upper_wick >= 0.20 * atr):
                continue
                
    # Spacing
    if t - last_trade_bar < 1:
        continue
        
    last_trade_bar = t
    if net_dir == 1: b_cnt += 1
    else: s_cnt += 1
    
    fwd_h = highs[t+1 : t+31]
    fwd_l = lows[t+1 : t+31]
    fwd_c = closes[t+1 : t+31]
    
    sl_dist = 2.0 * atr
    be_trig = 0.55 * atr
    be_lock = 0.35 * atr
    tp_dist = 2.00 * atr
    trail = 0.60 * atr
    
    cur_sl = sl_dist
    peak_fav = 0.0
    pnl_r = 0.0
    done = False
    
    for k in range(len(fwd_c)):
        fav = (fwd_h[k] - cur_p) if net_dir == 1 else (cur_p - fwd_l[k])
        adv = (cur_p - fwd_l[k]) if net_dir == 1 else (fwd_h[k] - cur_p)
        c_fav = (fwd_c[k] - cur_p) if net_dir == 1 else (cur_p - fwd_c[k])
        
        peak_fav = max(peak_fav, fav)
        if peak_fav >= be_trig:
            cur_sl = -max(be_lock, peak_fav - trail)
            
        if adv >= cur_sl:
            pnl_r = -cur_sl / sl_dist
            done = True
            break
        elif fav >= tp_dist:
            pnl_r = tp_dist / sl_dist
            done = True
            break
            
    if not done:
        pnl_r = c_fav / sl_dist
        
    trades.append({
        'bar': t,
        'act': net_dir,
        'pnl_r': pnl_r,
        'win': pnl_r > 0,
    })

df_res = pd.DataFrame(trades)
n_trades = len(df_res)
wins = df_res[df_res['pnl_r'] > 0]
losses = df_res[df_res['pnl_r'] <= 0]
wr = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0
pf = wins['pnl_r'].sum() / max(0.001, abs(losses['pnl_r'].sum())) if len(losses) > 0 else 999.0
cum = df_res['pnl_r'].cumsum()
dd = (cum.cummax() - cum).max() * 0.25

print("\n" + "=" * 80)
print("  ONLINE ADAPTIVE Z-SCORE SIMULATION RESULTS")
print("=" * 80)
print(f"Total Bars Evaluated:    {N:,}")
print(f"Total Trades:            {n_trades} ({n_trades / 7.0:.1f} trades/day)")
print(f"Trade Breakdown:         BUY: {b_cnt} ({b_cnt/n_trades*100:.1f}%), SELL: {s_cnt} ({s_cnt/n_trades*100:.1f}%)")
print(f"Win Rate:                {wr:.2f}% (Wins: {len(wins)}, Losses: {len(losses)})")
print(f"Profit Factor (PF):      {pf:.2f}")
print(f"Max Drawdown (DD):       {dd:.2f}%")
print("=" * 80)
