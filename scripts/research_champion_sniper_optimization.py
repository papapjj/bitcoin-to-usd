import sys, os
sys.path.insert(0, os.getcwd())
"""
Research Script: Sniper Timing, Signal Filtering, and Two-Tier Geometry Optimization
Tests parameters to reach:
- Win Rate >= 80%
- Profit Factor > 3.0
- Trades >= 700 (over 7 days = >100/day)
- Max Drawdown < 10%
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_7day_bilateral_dual_cot_selfplay import extract_features_and_targets_from_m1

print("1. Loading 7-day data and extracting features...")
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
ingress, targets, traps, prices, atrs = extract_features_and_targets_from_m1(df)

config = BilateralDualCoTConfig()
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint('outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt')
engine.eval()

print("2. Computing raw logits, PRM, and convictions across all bars...")
N = len(ingress)
with torch.no_grad():
    x_t = torch.from_numpy(ingress).float()
    z = engine.state_adapter(x_t)
    
    # Process in chunks of 512
    all_b_scores = []
    all_r_scores = []
    all_arb_logits = []
    all_prm_scores = []
    all_trap_logits = []
    
    for start in range(0, N, 512):
        end = min(N, start + 512)
        z_chunk = z[start:end]
        
        bull_out = engine.bull_cot(z_chunk)
        bear_out = engine.bear_cot(z_chunk)
        
        b_s = bull_out.bull_score.numpy()
        r_s = bear_out.bear_score.numpy()
        
        p_thought = bull_out.cot_hops[:, -1, :]
        a_thought = bear_out.cot_hops[:, -1, :]
        
        final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought, a_thought, turns=3)
        synthesis = torch.cat([final_p, final_a], dim=-1)
        
        arb_l = engine.arbiter.arbiter_net(synthesis).numpy()
        prm_s = engine.arbiter.prm_evaluator(synthesis).squeeze(-1).numpy()
        
        joint_state = torch.cat([a_thought, p_thought], dim=-1)
        trap_l = engine.trap_detector(joint_state).numpy()
        
        all_b_scores.append(b_s)
        all_r_scores.append(r_s)
        all_arb_logits.append(arb_l)
        all_prm_scores.append(prm_s)
        all_trap_logits.append(trap_l)
        
b_scores = np.concatenate(all_b_scores, axis=0)
r_scores = np.concatenate(all_r_scores, axis=0)
arb_logits = np.concatenate(all_arb_logits, axis=0)
prm_scores = np.concatenate(all_prm_scores, axis=0)
trap_logits = np.concatenate(all_trap_logits, axis=0)

# Softmax for arbiter probabilities
exp_arb = np.exp(arb_logits - np.max(arb_logits, axis=-1, keepdims=True))
arb_probs = exp_arb / np.sum(exp_arb, axis=-1, keepdims=True)
p_hold = arb_probs[:, 0]
p_buy = arb_probs[:, 1]
p_sell = arb_probs[:, 2]

# High/Low/Open/Close for timing confirmation
opens = df['<OPEN>'].values[-N:] if '<OPEN>' in df.columns else df['open'].values[-N:]
highs = df['<HIGH>'].values[-N:] if '<HIGH>' in df.columns else df['high'].values[-N:]
lows = df['<LOW>'].values[-N:] if '<LOW>' in df.columns else df['low'].values[-N:]
closes = df['<CLOSE>'].values[-N:] if '<CLOSE>' in df.columns else df['close'].values[-N:]

print("Data prepared. Starting simulation grid search...")

def run_simulation(
    prob_threshold=0.45,
    margin_threshold=0.10,
    prm_threshold=0.35,
    require_sniper_confirmation=True,
    sl_atr=2.0,
    be_trig_atr=0.8,
    be_lock_atr=0.3,
    tp_atr=2.2,
    trail_dist_atr=1.0,
    min_spacing=3,
):
    trades = []
    last_entry_t = -999
    
    for t in range(50, N - 30):
        # Spacing
        if t - last_entry_t < min_spacing:
            continue
            
        cur_p = prices[t]
        atr = atrs[t]
        
        # Trend guard (MA20 vs MA50)
        ma20 = np.mean(prices[t-20:t])
        ma50 = np.mean(prices[t-50:t])
        
        # Decision logic
        b_prob = p_buy[t]
        s_prob = p_sell[t]
        h_prob = p_hold[t]
        prm = prm_scores[t]
        b_s = b_scores[t]
        r_s = r_scores[t]
        diff = b_s - r_s
        
        action = 0
        if prm >= prm_threshold:
            if (b_prob >= prob_threshold and b_prob > s_prob and b_prob > h_prob) or (diff >= margin_threshold and b_s >= 0.50):
                action = 1
            elif (s_prob >= prob_threshold and s_prob > b_prob and s_prob > h_prob) or (-diff >= margin_threshold and r_s >= 0.50):
                action = -1
                
        # Trend alignment guard
        if ma20 > ma50 and cur_p > ma50 and action == -1:
            action = 0
        if ma20 < ma50 and cur_p < ma50 and action == 1:
            action = 0
            
        # Sniper Timing Confirmation (Anti-MAE: Don't buy in a free fall, don't short a pump)
        if require_sniper_confirmation and action != 0:
            if action == 1:
                # BUY: requires candle close > open (green bar) OR absorption wick (close - low >= 0.20 ATR)
                candle_green = closes[t] >= opens[t]
                lower_wick = closes[t] - lows[t]
                wick_absorption = lower_wick >= 0.20 * atr
                if not (candle_green or wick_absorption):
                    action = 0
            elif action == -1:
                # SELL: requires candle close < open (red bar) OR rejection wick (high - close >= 0.20 ATR)
                candle_red = closes[t] <= opens[t]
                upper_wick = highs[t] - closes[t]
                wick_rejection = upper_wick >= 0.20 * atr
                if not (candle_red or wick_rejection):
                    action = 0
                    
        if action == 0:
            continue
            
        # Execute trade simulation across next 30 bars
        last_entry_t = t
        entry_price = cur_p
        fwd_highs = highs[t+1 : t+31]
        fwd_lows = lows[t+1 : t+31]
        fwd_closes = closes[t+1 : t+31]
        
        sl_dist = sl_atr * atr
        be_trig = be_trig_atr * atr
        be_lock = be_lock_atr * atr
        tp_dist = tp_atr * atr
        trail = trail_dist_atr * atr
        
        cur_sl = sl_dist
        peak_fav = 0.0
        pnl_r = 0.0
        hit_tp = False
        hit_sl = False
        bars_held = 0
        
        for k in range(len(fwd_closes)):
            bars_held += 1
            if action == 1:
                f_fav = fwd_highs[k] - entry_price
                f_adv = entry_price - fwd_lows[k]
                c_fav = fwd_closes[k] - entry_price
            else:
                f_fav = entry_price - fwd_lows[k]
                f_adv = fwd_highs[k] - entry_price
                c_fav = entry_price - fwd_closes[k]
                
            peak_fav = max(peak_fav, f_fav)
            
            # Trailing stop update
            if peak_fav >= be_trig:
                trail_sl = peak_fav - trail
                cur_sl = -max(be_lock, trail_sl)  # negative means in profit
                
            # Check SL hit
            if f_adv >= cur_sl:
                hit_sl = True
                pnl_r = -cur_sl / sl_dist  # could be positive if cur_sl is locked in profit
                break
                
            # Check TP hit
            if f_fav >= tp_dist:
                hit_tp = True
                pnl_r = tp_dist / sl_dist
                break
                
        if not (hit_sl or hit_tp):
            # Time exit at bar 30
            pnl_r = c_fav / sl_dist
            
        trades.append({
            'bar': t,
            'action': action,
            'pnl_r': pnl_r,
            'win': pnl_r > 0,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'bars_held': bars_held,
        })
        
    if len(trades) == 0:
        return 0, 0, 0, 0, 0, 0
        
    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    wins = df_trades[df_trades['pnl_r'] > 0]
    losses = df_trades[df_trades['pnl_r'] <= 0]
    win_rate = len(wins) / total_trades * 100.0
    
    total_win_r = wins['pnl_r'].sum() if len(wins) > 0 else 0.0
    total_loss_r = abs(losses['pnl_r'].sum()) if len(losses) > 0 else 0.001
    pf = total_win_r / total_loss_r
    
    cum_r = df_trades['pnl_r'].cumsum()
    peak = cum_r.cummax()
    dd = (peak - cum_r).max()
    # DD in percentage assuming 0.25% risk per 1.0 R:
    dd_pct = dd * 0.25
    
    buys = len(df_trades[df_trades['action'] == 1])
    sells = len(df_trades[df_trades['action'] == -1])
    
    return total_trades, buys, sells, win_rate, pf, dd_pct

# Run parameter sweep
print(f"{'Prob':<5} {'Sniper':<7} {'SL':<4} {'BE_trig':<7} {'BE_lock':<7} {'TP':<4} {'Trail':<5} | {'Trades':<6} {'Buys':<5} {'Sells':<5} {'WR%':<6} {'PF':<5} {'DD%':<5}")
print("-" * 85)

results = []
for sniper in [True, False]:
    for prob in [0.40, 0.45, 0.50]:
        for sl in [1.5, 2.0]:
            for be_trig in [0.7, 0.8, 1.0]:
                for be_lock in [0.25, 0.35]:
                    for tp in [2.0, 2.5]:
                        for trail in [0.8, 1.0, 1.2]:
                            n, b, s, wr, pf, dd = run_simulation(
                                prob_threshold=prob,
                                require_sniper_confirmation=sniper,
                                sl_atr=sl,
                                be_trig_atr=be_trig,
                                be_lock_atr=be_lock,
                                tp_atr=tp,
                                trail_dist_atr=trail,
                            )
                            if n >= 500:  # Candidates with healthy trade counts
                                results.append((wr, pf, n, b, s, dd, prob, sniper, sl, be_trig, be_lock, tp, trail))

results.sort(key=lambda x: (x[0] >= 80.0, x[1]), reverse=True)
print("\nTOP 15 PARAMETER CONFIGURATIONS:")
print(f"{'Rank':<4} {'WR%':<6} {'PF':<6} {'Trades':<7} {'Buys':<6} {'Sells':<6} {'DD%':<6} | {'Prob':<5} {'Sniper':<7} {'SL':<4} {'BE_trig':<7} {'BE_lock':<7} {'TP':<4} {'Trail':<5}")
print("-" * 85)
for idx, res in enumerate(results[:15], 1):
    wr, pf, n, b, s, dd, prob, sniper, sl, be_trig, be_lock, tp, trail = res
    print(f"{idx:<4} {wr:<6.2f} {pf:<6.2f} {n:<7} {b:<6} {s:<6} {dd:<6.2f} | {prob:<5.2f} {str(sniper):<7} {sl:<4.1f} {be_trig:<7.2f} {be_lock:<7.2f} {tp:<4.1f} {trail:<5.2f}")
