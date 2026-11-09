import torch
import numpy as np
import pandas as pd
from pathlib import Path
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)

ROOT = Path('.')
champ_pt = ROOT / 'outputs' / 'he_chuyen_gia' / 'v193_runs' / 'mythos_sniper_anti_mae_champion.pt'

config = BilateralDualCoTConfig()
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint(champ_pt)
engine.eval()

# Load clean 7-day M1 data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
print(f"Loaded {len(df)} M1 bars.")

closes = df['<CLOSE>'].values.astype(np.float64)
highs = df['<HIGH>'].values.astype(np.float64)
lows = df['<LOW>'].values.astype(np.float64)
opens = df['<OPEN>'].values.astype(np.float64)
vols = df['<TICKVOL>'].values.astype(np.float64)

# Calculate ATR14
tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
atr = np.zeros(len(df))
atr[:14] = np.mean(tr[:14])
for i in range(14, len(df)):
    atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14.0

# Pre-extract M1 buffer for each bar
results = []
bars_matrix = np.column_stack([opens, highs, lows, closes, vols])

print("Evaluating 2,000 M1 bars (Feb 20-22)...")

for i in range(60, min(2060, len(df))):
    cur_p = closes[i]
    cur_atr = atr[i]
    m1_buf = bars_matrix[i-59 : i+1]
    
    # Simple base_obs_137 with candle metrics
    base_obs = np.zeros(137, dtype=np.float32)
    base_obs[0] = (closes[i] - closes[i-1]) / max(1.0, cur_atr)
    base_obs[7] = (closes[i] - opens[i]) / max(1.0, cur_atr)
    base_obs[8] = (highs[i] - max(opens[i], closes[i])) / max(1.0, cur_atr)
    base_obs[9] = (min(opens[i], closes[i]) - lows[i]) / max(1.0, cur_atr)
    base_obs[10] = (highs[i] - lows[i]) / max(1.0, cur_atr)
    
    ingress_233 = extract_canonical_bilateral_ingress_233(
        base_obs_137=base_obs,
        m1_bars_60=m1_buf,
        cur_p=cur_p,
        cur_atr=cur_atr,
    )
    
    with torch.no_grad():
        t_in = torch.from_numpy(ingress_233).float()
        # Adapt state
        z = engine.state_adapter(t_in.unsqueeze(0))
        bull_out = engine.bull_cot(z)
        bear_out = engine.bear_cot(z)
        b_sc = float(bull_out.bull_score[0].item())
        r_sc = float(bear_out.bear_score[0].item())
        
        p_thought = bull_out.cot_hops[:, -1, :]
        a_thought = bear_out.cot_hops[:, -1, :]
        joint = torch.cat([a_thought, p_thought], dim=-1)
        trap_logits = engine.trap_detector(joint)
        trap_probs = torch.softmax(trap_logits, dim=-1)[0]
        
        final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought, a_thought, turns=config.num_debate_turns)
        conv = torch.tensor([[b_sc, r_sc]], dtype=torch.float32)
        synthesis = torch.cat([final_p, final_a], dim=-1)
        prm = float(engine.arbiter.prm_evaluator(synthesis).item())
        arb_logits = engine.arbiter.arbiter_net(synthesis)
        arb_probs = torch.softmax(arb_logits, dim=-1)[0]
        
    # Future outcome over next 30 bars:
    # Check if +1.8 ATR is hit before -1.5 ATR
    future_high = np.max(highs[i+1 : min(i+31, len(df))])
    future_low = np.min(lows[i+1 : min(i+31, len(df))])
    
    buy_mae = (cur_p - future_low) / cur_atr
    buy_mfe = (future_high - cur_p) / cur_atr
    
    sell_mae = (future_high - cur_p) / cur_atr
    sell_mfe = (cur_p - future_low) / cur_atr
    
    results.append({
        'bar': i,
        'b_sc': b_sc,
        'r_sc': r_sc,
        'prm': prm,
        'p_buy': float(arb_probs[1].item()),
        'p_sell': float(arb_probs[2].item()),
        'p_hold': float(arb_probs[0].item()),
        'buy_mae': buy_mae,
        'buy_mfe': buy_mfe,
        'sell_mae': sell_mae,
        'sell_mfe': sell_mfe,
    })

res_df = pd.DataFrame(results)

# Test different entry thresholds:
print(f"\n{'Threshold':<15} {'Total Trades':<14} {'Wins (MFE>1.8)':<16} {'Loss (MAE>1.5)':<16} {'WinRate (%)':<12}")
print("-" * 75)

for thresh in [0.20, 0.35, 0.50, 0.65, 0.75, 0.85]:
    # BUY entries: p_buy >= thresh & p_buy > p_sell
    buy_trades = res_df[(res_df['p_buy'] >= thresh) & (res_df['p_buy'] > res_df['p_sell'])]
    buy_wins = (buy_trades['buy_mfe'] >= 1.8) & (buy_trades['buy_mae'] < 1.5)
    buy_losses = (buy_trades['buy_mae'] >= 1.5)
    
    # SELL entries: p_sell >= thresh & p_sell > p_buy
    sell_trades = res_df[(res_df['p_sell'] >= thresh) & (res_df['p_sell'] > res_df['p_buy'])]
    sell_wins = (sell_trades['sell_mfe'] >= 1.8) & (sell_trades['sell_mae'] < 1.5)
    sell_losses = (sell_trades['sell_mae'] >= 1.5)
    
    total = len(buy_trades) + len(sell_trades)
    wins = buy_wins.sum() + sell_wins.sum()
    losses = buy_losses.sum() + sell_losses.sum()
    wr = (wins / max(1, (wins + losses))) * 100
    
    print(f"p >= {thresh:<10.2f} {total:<14} {wins:<16} {losses:<16} {wr:<12.1f}%")
