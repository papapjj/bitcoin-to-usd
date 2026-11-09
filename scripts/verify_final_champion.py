import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

print("=" * 80)
print("AUDITING NEW CHAMPION OVER ALL 11,181 BARS (7 FULL DAYS OOS)")
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
    actions = verdict.actions.numpy()
    prms = verdict.prm_scores.numpy()
    confs = verdict.confidences.numpy()
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    cps = verdict.counter_punches.numpy()

print(f"Raw Actions: BUY={np.sum(actions == 1)}, SELL={np.sum(actions == -1)}, HOLD={np.sum(actions == 0)}")
print(f"Bull scores mean: {np.mean(b_scores):.4f} | Bear scores mean: {np.mean(r_scores):.4f}")

# Simulate Realistic MT5 Execution Geometry
trades = []
buys_cnt = 0
sells_cnt = 0
bottom_shorts = 0
top_buys = 0
last_trade_bar = -999

for t in range(50, N - 30):
    act = actions[t]
    if act == 0 or (t - last_trade_bar < 2):
        continue
        
    cur_p = prices[t]
    atr = atrs[t]
    
    # 1. Anti-Extreme Exhaustion Guards
    rec_l_60 = np.min(lows[max(0, t - 60) : t])
    rec_h_60 = np.max(highs[max(0, t - 60) : t])
    rng_60 = max(1.0, rec_h_60 - rec_l_60)
    rel_pos = (cur_p - rec_l_60) / rng_60
    if act == -1 and rel_pos <= 0.15:
        bottom_shorts += 1
        continue
    if act == 1 and rel_pos >= 0.85:
        top_buys += 1
        continue
        
    # 2. Universal Trend Alignment Guard
    ma20 = np.mean(prices[t-20:t])
    ma50 = np.mean(prices[t-50:t])
    if ma20 > ma50 and cur_p > ma50 and act == -1 and not cps[t]:
        continue
    if ma20 < ma50 and cur_p < ma50 and act == 1 and not cps[t]:
        continue
        
    # 3. Sniper Timing Confirmation (Anti-MAE)
    if not cps[t]:
        if act == 1:
            candle_green = closes[t] >= opens[t]
            lower_wick = closes[t] - lows[t]
            if not (candle_green or lower_wick >= 0.20 * atr):
                continue
        elif act == -1:
            candle_red = closes[t] <= opens[t]
            upper_wick = highs[t] - closes[t]
            if not (candle_red or upper_wick >= 0.20 * atr):
                continue
                
    last_trade_bar = t
    if act == 1: buys_cnt += 1
    else: sells_cnt += 1
    
    fwd_h = highs[t+1 : t+31]
    fwd_l = lows[t+1 : t+31]
    fwd_c = closes[t+1 : t+31]
    
    sl_dist = 2.0 * atr
    be_trig = 0.80 * atr
    be_lock = 0.35 * atr
    tp_dist = 2.20 * atr
    trail = 0.80 * atr
    
    cur_sl = sl_dist
    peak_fav = 0.0
    pnl_r = 0.0
    done = False
    
    for k in range(len(fwd_c)):
        fav = (fwd_h[k] - cur_p) if act == 1 else (cur_p - fwd_l[k])
        adv = (cur_p - fwd_l[k]) if act == 1 else (fwd_h[k] - cur_p)
        c_fav = (fwd_c[k] - cur_p) if act == 1 else (cur_p - fwd_c[k])
        
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
        'act': act,
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
print("  FINAL 7-DAY VERIFICATION REPORT")
print("=" * 80)
print(f"Total Bars Evaluated:    {N:,}")
print(f"Total Trades:            {n_trades} ({n_trades / 7.0:.1f} trades/day)")
print(f"Trade Breakdown:         BUY: {buys_cnt} ({buys_cnt/n_trades*100:.1f}%), SELL: {sells_cnt} ({sells_cnt/n_trades*100:.1f}%)")
print(f"Win Rate:                {wr:.2f}% (Wins: {len(wins)}, Losses: {len(losses)})")
print(f"Profit Factor (PF):      {pf:.2f}")
print(f"Max Drawdown (DD):       {dd:.2f}%")
print(f"Bottom Shorts Vetoed:    {bottom_shorts}")
print(f"Top Buys Vetoed:         {top_buys}")
print("=" * 80)
