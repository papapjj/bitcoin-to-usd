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

print("Running simulation of Golden Master Champion...", flush=True)

def run_test(
    min_margin=0.03,
    min_conviction=0.47,
    prm_thresh=0.32,
    val_zone=0.50,
    sl_m=2.0,
    tp_m=1.2,
    be_trig_m=0.70,
    be_lock_m=0.35,
    cooldown=3,
    spacing=2,
):
    trades = []
    last_exit_t = -999
    last_entry_t = -999
    b_cnt, s_cnt = 0, 0
    
    for t in range(60, N - 35):
        if t - last_entry_t < spacing: continue
        if t - last_exit_t < cooldown: continue
        
        cur_atr = atrs[t]
        cp = prices[t]
        bs = b_s[t]
        rs = r_s[t]
        prm = prms[t]
        cp_sig = cps[t]
        
        h60 = np.max(highs[max(0, t-60):t])
        l60 = np.min(lows[max(0, t-60):t])
        rng60 = max(1.0, h60 - l60)
        rel_pos = (cp - l60) / rng60
        
        up = cp > ema50[t] and ema20[t] > ema50[t]
        down = cp < ema50[t] and ema20[t] < ema50[t]
        
        act = 0
        if cp_sig:
            act = int(verdict.actions[t])
        else:
            diff = bs - rs
            if diff >= min_margin and bs >= min_conviction and prm >= prm_thresh and up:
                # Value Zone guard: price must not be extended > val_zone ATR above EMA20
                if cp <= ema20[t] + val_zone * cur_atr and rel_pos <= 0.85:
                    # Candle confirmation
                    if closes[t] >= opens[t] or (closes[t] - lows[t]) >= 0.20 * cur_atr:
                        act = 1
            elif -diff >= min_margin and rs >= min_conviction and prm >= prm_thresh and down:
                # Value Zone guard: price must not be extended > val_zone ATR below EMA20
                if cp >= ema20[t] - val_zone * cur_atr and rel_pos >= 0.15:
                    # Candle confirmation
                    if closes[t] <= opens[t] or (highs[t] - closes[t]) >= 0.20 * cur_atr:
                        act = -1
                        
        if act == 0: continue
        last_entry_t = t
        if act == 1: b_cnt += 1
        else: s_cnt += 1
        
        sl_d = sl_m * cur_atr
        tp_d = tp_m * cur_atr
        bet_d = be_trig_m * cur_atr
        bel_d = be_lock_m * cur_atr
        
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
                
        trades.append((pnl, reason, sl_d))
        
    if len(trades) == 0: return None
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
    
    return {
        'wr': wr, 'pf': pf, 'n': len(trades), 'tr_d': len(trades) / 7.0,
        'b_cnt': b_cnt, 's_cnt': s_cnt, 'dd': dd,
        'tp_cnt': tp_cnt, 'be_cnt': be_cnt, 'sl_cnt': sl_cnt,
        'margin': min_margin, 'conv': min_conviction, 'prm': prm_thresh,
        'sl_m': sl_m, 'tp_m': tp_m, 'bet_m': be_trig_m, 'bel_m': be_lock_m,
    }

results = []
for margin in [0.01, 0.02, 0.03, 0.04]:
    for conv in [0.46, 0.47, 0.48]:
        for sl in [1.5, 2.0, 2.5]:
            for tp in [1.0, 1.2, 1.5]:
                for bet in [0.60, 0.70]:
                    for bel in [0.35, 0.45]:
                        if bel >= bet: continue
                        r = run_test(
                            min_margin=margin, min_conviction=conv,
                            sl_m=sl, tp_m=tp, be_trig_m=bet, be_lock_m=bel
                        )
                        if r is not None and r['n'] >= 200:
                            results.append(r)

results.sort(key=lambda x: (x['wr'] >= 80.0, x['pf'] >= 3.0, x['pf'], x['wr']), reverse=True)

print(f"\nTop 20 Configurations:")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/BE/SL':<16} {'DD':<6} {'Margin':<8} {'Conv':<6} {'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6}")
print("-" * 105)
for r in results[:20]:
    bs = f"{r['b_cnt']}/{r['s_cnt']}"
    hits = f"{r['tp_cnt']}/{r['be_cnt']}/{r['sl_cnt']}"
    print(f"{r['wr']:5.2f}%   {r['pf']:5.2f}  {r['n']:<8} {r['tr_d']:<8.1f} {bs:<12} {hits:<16} {r['dd']:4.1f}% {r['margin']:<8.2f} {r['conv']:<6.2f} {r['sl_m']:<5.1f} {r['tp_m']:<5.1f} {r['bet_m']:<6.2f} {r['bel_m']:<6.2f}")
