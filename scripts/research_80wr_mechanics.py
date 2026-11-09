import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

print("Loading data...")
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
    b_s = verdict.bull_scores.numpy()
    r_s = verdict.bear_scores.numpy()
    prms = verdict.prm_scores.numpy()

# Calculate M5 and M15 trends from M1 buffer
# 5 bars = M5, 15 bars = M15, 60 bars = H1
ema20 = pd.Series(closes).ewm(span=20).mean().values
ema50 = pd.Series(closes).ewm(span=50).mean().values
ema200 = pd.Series(closes).ewm(span=200).mean().values

# RSI 14
delta = pd.Series(closes).diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
rsi = 100 - (100 / (1 + rs)).values

print("\n" + "=" * 90)
print(f"{'TP_atr':<7} {'BE_trig':<8} {'BE_lock':<8} {'SL_atr':<7} | {'Trades':<7} {'Buys':<6} {'Sells':<6} {'WR%':<7} {'PF':<6} {'DD%':<6}")
print("=" * 90)

configs = []
for tp_atr in [1.0, 1.2, 1.5, 1.8, 2.0]:
    for be_trig_atr in [0.50, 0.60, 0.70, 0.80]:
        for be_lock_atr in [0.25, 0.35, 0.40]:
            for sl_atr in [1.5, 2.0, 2.5]:
                trades = []
                b_cnt = 0
                s_cnt = 0
                last_t = -999
                
                for t in range(200, N - 30):
                    if t - last_t < 2:
                        continue
                        
                    cur_p = prices[t]
                    atr = atrs[t]
                    
                    # HTF Confluence: EMA20 vs EMA50 and EMA200
                    h1_up = cur_p > ema200[t] and ema20[t] > ema50[t]
                    h1_down = cur_p < ema200[t] and ema20[t] < ema50[t]
                    
                    # Champion CoT signals
                    act = 0
                    if b_s[t] >= 0.42 and (b_s[t] - r_s[t]) >= 0.08 and h1_up:
                        act = 1
                    elif r_s[t] >= 0.42 and (r_s[t] - b_s[t]) >= 0.08 and h1_down:
                        act = -1
                    if act == 0:
                        continue
                        
                    # RSI confirmation: don't buy overbought (>70), don't short oversold (<30)
                    if act == 1 and rsi[t] >= 68:
                        continue
                    if act == -1 and rsi[t] <= 32:
                        continue
                        
                    # Sniper candle confirmation
                    if act == 1:
                        candle_green = closes[t] >= opens[t]
                        lower_wick = closes[t] - lows[t]
                        if not (candle_green or lower_wick >= 0.20 * atr):
                            continue
                    else:
                        candle_red = closes[t] <= opens[t]
                        upper_wick = highs[t] - closes[t]
                        if not (candle_red or upper_wick >= 0.20 * atr):
                            continue
                            
                    last_t = t
                    if act == 1: b_cnt += 1
                    else: s_cnt += 1
                    
                    fwd_h = highs[t+1 : t+31]
                    fwd_l = lows[t+1 : t+31]
                    fwd_c = closes[t+1 : t+31]
                    
                    sl_dist = sl_atr * atr
                    be_trig = be_trig_atr * atr
                    be_lock = be_lock_atr * atr
                    tp_dist = tp_atr * atr
                    trail = 0.60 * atr
                    
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
                        
                    trades.append(pnl_r)
                    
                if len(trades) >= 100:
                    tr = np.array(trades)
                    wins = tr[tr > 0]
                    losses = tr[tr <= 0]
                    wr = len(wins) / len(tr) * 100.0
                    pf = wins.sum() / max(0.001, abs(losses.sum()))
                    cum = np.cumsum(tr)
                    dd = (np.maximum.accumulate(cum) - cum).max() * 0.25
                    configs.append((wr, pf, len(tr), b_cnt, s_cnt, dd, tp_atr, be_trig_atr, be_lock_atr, sl_atr))

configs.sort(key=lambda x: (x[0] >= 80.0, x[1], x[0]), reverse=True)
for wr, pf, n, b, s, dd, tp, bet, bel, sl in configs[:20]:
    print(f"{tp:<7.2f} {bet:<8.2f} {bel:<8.2f} {sl:<7.2f} | {n:<7} {b:<6} {s:<6} {wr:<7.2f} {pf:<6.2f} {dd:<6.2f}")
