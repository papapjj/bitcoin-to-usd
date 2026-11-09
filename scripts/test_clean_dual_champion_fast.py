import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np, torch
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

print("Loading 7-day dataset and Bilateral Dual-CoT Champion...", flush=True)
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
    cps = verdict.counter_punches.numpy()

s_prices = pd.Series(prices)
ema20 = s_prices.ewm(span=20).mean().values
ema50 = s_prices.ewm(span=50).mean().values

# Precompute rel_pos_60
s_high = pd.Series(highs)
s_low = pd.Series(lows)
rolling_h60 = s_high.rolling(window=60, min_periods=1).max().values
rolling_l60 = s_low.rolling(window=60, min_periods=1).min().values
rng60 = np.maximum(1.0, rolling_h60 - rolling_l60)
rel_pos_60 = (prices - rolling_l60) / rng60

print(f"Precalculated indicators across {N} bars. Precalculating signal candidates...", flush=True)

# Precalculate signals for different margin/conviction configs
def get_signals(min_margin=0.02, min_conv=0.46, val_zone=0.50):
    signals = []
    for t in range(60, N - 35):
        cur_atr = atrs[t]
        cp = prices[t]
        bs = b_s[t]
        rs = r_s[t]
        prm = prms[t]
        cp_sig = cps[t]
        rel_pos = rel_pos_60[t]
        
        up = cp > ema50[t] and ema20[t] > ema50[t]
        down = cp < ema50[t] and ema20[t] < ema50[t]
        
        act = 0
        if cp_sig:
            act = int(verdict.actions[t])
        else:
            diff = bs - rs
            if diff >= min_margin and bs >= min_conv and prm >= 0.32 and up:
                if cp <= ema20[t] + val_zone * cur_atr and rel_pos <= 0.85:
                    if closes[t] >= opens[t] or (closes[t] - lows[t]) >= 0.20 * cur_atr:
                        act = 1
            elif -diff >= min_margin and rs >= min_conv and prm >= 0.32 and down:
                if cp >= ema20[t] - val_zone * cur_atr and rel_pos >= 0.15:
                    if closes[t] <= opens[t] or (highs[t] - closes[t]) >= 0.20 * cur_atr:
                        act = -1
        if act != 0:
            signals.append((t, act, cp, cur_atr))
    return signals

sig_cache = {}
for m in [0.01, 0.02, 0.03]:
    for c in [0.46, 0.47, 0.48]:
        for vz in [0.35, 0.50, 0.70]:
            sig_cache[(m, c, vz)] = get_signals(m, c, vz)

print("Precalculated signal cache. Scanning execution geometry grid...", flush=True)

results = []
for (m, c, vz), sigs in sig_cache.items():
    if len(sigs) < 400: continue
    for sl_m in [1.5, 2.0, 2.5, 3.0]:
        for tp_m in [1.0, 1.2, 1.5, 1.8, 2.0]:
            for bet_m in [0.50, 0.60, 0.70]:
                for bel_m in [0.30, 0.40, 0.50]:
                    if bel_m >= bet_m: continue
                    trades = []
                    last_exit_t = -999
                    last_entry_t = -999
                    b_cnt, s_cnt = 0, 0
                    
                    for t, act, cp, cur_atr in sigs:
                        if t - last_entry_t < 2: continue
                        if t - last_exit_t < 3: continue
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
                            for k in range(t + 1, min(t + 35, N)):
                                if not be_active and highs[k] >= cp + bet_d:
                                    be_active = True
                                    cur_sl = max(cur_sl, cp + bel_d)
                                if lows[k] <= cur_sl:
                                    pnl = cur_sl - cp
                                    reason = 'BE' if be_active else 'SL'
                                    last_exit_t = k
                                    break
                                elif highs[k] >= tp_p:
                                    pnl = tp_d
                                    reason = 'TP'
                                    last_exit_t = k
                                    break
                            else:
                                pnl = closes[min(t + 34, N - 1)] - cp
                                reason = 'TIME'
                                last_exit_t = min(t + 34, N - 1)
                        else:
                            cur_sl = cp + sl_d
                            tp_p = cp - tp_d
                            be_active = False
                            for k in range(t + 1, min(t + 35, N)):
                                if not be_active and lows[k] <= cp - bet_d:
                                    be_active = True
                                    cur_sl = min(cur_sl, cp - bel_d)
                                if highs[k] >= cur_sl:
                                    pnl = cp - cur_sl
                                    reason = 'BE' if be_active else 'SL'
                                    last_exit_t = k
                                    break
                                elif lows[k] <= tp_p:
                                    pnl = tp_d
                                    reason = 'TP'
                                    last_exit_t = k
                                    break
                            else:
                                pnl = cp - closes[min(t + 34, N - 1)]
                                reason = 'TIME'
                                last_exit_t = min(t + 34, N - 1)
                                
                        trades.append((pnl, reason, cur_atr))
                        
                    if len(trades) < 200: continue
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
                    
                    results.append((wr, pf, len(trades), b_cnt, s_cnt, dd, tp_cnt, be_cnt, sl_cnt, m, c, vz, sl_m, tp_m, bet_m, bel_m))

results.sort(key=lambda x: (x[0] >= 80.0, x[1] >= 2.0, x[1], x[0]), reverse=True)

print(f"\nTop 25 Configurations:")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/BE/SL':<16} {'DD':<6} {'Margin':<7} {'Conv':<6} {'VZ':<5} {'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6}")
print("-" * 110)
for wr, pf, n, b, s, dd, tp, be, sl, m, c, vz, sl_m, tp_m, bet_m, bel_m in results[:25]:
    bs = f"{b}/{s}"
    hits = f"{tp}/{be}/{sl}"
    print(f"{wr:5.2f}%   {pf:5.2f}  {n:<8} {n/7.0:<8.1f} {bs:<12} {hits:<16} {dd:4.1f}% {m:<7.2f} {c:<6.2f} {vz:<5.2f} {sl_m:<5.1f} {tp_m:<5.1f} {bet_m:<6.2f} {bel_m:<6.2f}")
