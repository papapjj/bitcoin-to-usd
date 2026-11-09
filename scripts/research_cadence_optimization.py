import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

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

# EMAs
ema20 = pd.Series(closes).ewm(span=20).mean().values
ema50 = pd.Series(closes).ewm(span=50).mean().values
ema100 = pd.Series(closes).ewm(span=100).mean().values

# RSI 14
delta = pd.Series(closes).diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
rsi = 100 - (100 / (1 + rs)).values

print("\n" + "=" * 90)
print(f"{'Min_Spacing':<12} {'EMA_Mode':<10} {'Conv_Thresh':<12} | {'Trades':<7} {'Buys':<6} {'Sells':<6} {'WR%':<7} {'PF':<6} {'DD%':<6}")
print("=" * 90)

for spacing in [1, 2]:
    for ema_mode in ['m20_m50', 'm50_only', 'dynamic_pullback']:
        for conv_th in [0.38, 0.40, 0.42]:
            trades = []
            b_cnt = 0
            s_cnt = 0
            last_t = -999
            
            for t in range(100, N - 30):
                if t - last_t < spacing:
                    continue
                    
                cur_p = prices[t]
                atr = atrs[t]
                
                # Trend logic
                if ema_mode == 'm20_m50':
                    up_trend = cur_p > ema50[t] and ema20[t] > ema50[t]
                    down_trend = cur_p < ema50[t] and ema20[t] < ema50[t]
                elif ema_mode == 'm50_only':
                    up_trend = cur_p > ema50[t]
                    down_trend = cur_p < ema50[t]
                elif ema_mode == 'dynamic_pullback':
                    # In uptrend, buy pullbacks near EMA20
                    up_trend = ema20[t] > ema50[t] and cur_p >= ema50[t] - 0.5 * atr
                    down_trend = ema20[t] < ema50[t] and cur_p <= ema50[t] + 0.5 * atr
                    
                # Champion CoT signals
                act = 0
                if b_s[t] >= conv_th and (b_s[t] - r_s[t]) >= 0.05 and up_trend:
                    act = 1
                elif r_s[t] >= conv_th and (r_s[t] - b_s[t]) >= 0.05 and down_trend:
                    act = -1
                if act == 0:
                    continue
                    
                # RSI filter
                if act == 1 and rsi[t] >= 72:
                    continue
                if act == -1 and rsi[t] <= 28:
                    continue
                    
                # Sniper confirmation
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
                
                sl_dist = 2.0 * atr
                be_trig = 0.55 * atr
                be_lock = 0.35 * atr
                tp_dist = 2.0 * atr
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
                
            if len(trades) >= 50:
                tr = np.array(trades)
                wins = tr[tr > 0]
                losses = tr[tr <= 0]
                wr = len(wins) / len(tr) * 100.0
                pf = wins.sum() / max(0.001, abs(losses.sum()))
                cum = np.cumsum(tr)
                dd = (np.maximum.accumulate(cum) - cum).max() * 0.25
                print(f"{spacing:<12} {ema_mode:<10} {conv_th:<12.2f} | {len(tr):<7} {b_cnt:<6} {s_cnt:<6} {wr:<7.2f} {pf:<6.2f} {dd:<6.2f}")
