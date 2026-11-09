import torch
import numpy as np
import pandas as pd
from pathlib import Path
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_pure_anti_mae_engine import build_pure_anti_mae_dataset

data = build_pure_anti_mae_dataset()
cfg = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, prm_threshold=0.35)
engine = BilateralDualCoTEngine(cfg)

ckpt_path = Path('outputs/he_chuyen_gia/v193_runs/mythos_symmetric_anti_mae_champion.pt')
engine.load_checkpoint(ckpt_path)
engine.eval()

ingress = torch.from_numpy(data['ingress']).float()
with torch.no_grad():
    v = engine.forward_batch(ingress, batch_size=512)

acts = v.actions.cpu().numpy()
prms = v.prm_scores.cpu().numpy()

closes = data['closes']
highs = data['highs']
lows = data['lows']
atrs = data['atrs']
lookback = data['lookback_bars']
horizon = data['horizon_bars']
N = len(acts)

print("=" * 85)
print("  FULL 7-DAY CANONICAL EVALUATION: ZERO MANUAL FILTERS / HEURISTIC GATES")
print(f"  Checkpoint: {ckpt_path.name}")
print("=" * 85)

for prm_th in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for sl_mult, tp_mult in [(1.2, 1.8), (1.5, 2.25)]:
        balance = 10000.0
        equity = 10000.0
        peak_equity = 10000.0
        max_dd_usd = 0.0
        
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        trade_maes = []
        trade_mfes = []
        
        # Position tracking (at most 1 active position at a time for strict execution)
        in_pos = False
        pos_dir = 0
        pos_entry_p = 0.0
        pos_sl_p = 0.0
        pos_tp_p = 0.0
        pos_atr = 0.0
        pos_vol = 0.01  # Fixed 0.01 BTC lot
        worst_adv = 0.0
        best_fav = 0.0
        
        for i in range(N):
            t = lookback + i
            cur_p = closes[t]
            cur_atr = atrs[i]
            cur_h = highs[t]
            cur_l = lows[t]
            
            # 1. Manage open position if any
            if in_pos:
                if pos_dir == 1:
                    adv = (pos_entry_p - cur_l) / pos_atr
                    fav = (cur_h - pos_entry_p) / pos_atr
                    if adv > worst_adv: worst_adv = adv
                    if fav > best_fav: best_fav = fav
                    
                    # Check SL
                    if cur_l <= pos_sl_p:
                        pnl = (pos_sl_p - pos_entry_p) * pos_vol
                        balance += pnl
                        gross_loss += abs(pnl)
                        losses += 1
                        trade_maes.append(worst_adv)
                        trade_mfes.append(best_fav)
                        in_pos = False
                    # Check TP
                    elif cur_h >= pos_tp_p:
                        pnl = (pos_tp_p - pos_entry_p) * pos_vol
                        balance += pnl
                        gross_profit += pnl
                        wins += 1
                        trade_maes.append(worst_adv)
                        trade_mfes.append(best_fav)
                        in_pos = False
                elif pos_dir == -1:
                    adv = (cur_h - pos_entry_p) / pos_atr
                    fav = (pos_entry_p - cur_l) / pos_atr
                    if adv > worst_adv: worst_adv = adv
                    if fav > best_fav: best_fav = fav
                    
                    # Check SL
                    if cur_h >= pos_sl_p:
                        pnl = (pos_entry_p - pos_sl_p) * pos_vol
                        balance += pnl
                        gross_loss += abs(pnl)
                        losses += 1
                        trade_maes.append(worst_adv)
                        trade_mfes.append(best_fav)
                        in_pos = False
                    # Check TP
                    elif cur_l <= pos_tp_p:
                        pnl = (pos_entry_p - pos_tp_p) * pos_vol
                        balance += pnl
                        gross_profit += pnl
                        wins += 1
                        trade_maes.append(worst_adv)
                        trade_mfes.append(best_fav)
                        in_pos = False
                        
                # Update Max Drawdown
                if balance > peak_equity:
                    peak_equity = balance
                dd = peak_equity - balance
                if dd > max_dd_usd:
                    max_dd_usd = dd
                    
            # 2. Check for new entry if not in position
            if not in_pos:
                act = acts[i]
                prm = prms[i]
                if prm >= prm_th and act != 0:
                    in_pos = True
                    pos_dir = act
                    pos_entry_p = cur_p
                    pos_atr = cur_atr
                    worst_adv = 0.0
                    best_fav = 0.0
                    if act == 1:
                        pos_sl_p = cur_p - sl_mult * cur_atr
                        pos_tp_p = cur_p + tp_mult * cur_atr
                    else:
                        pos_sl_p = cur_p + sl_mult * cur_atr
                        pos_tp_p = cur_p - tp_mult * cur_atr
                        
        tot_trades = wins + losses
        if tot_trades > 0:
            wr = (wins / tot_trades) * 100.0
            pf = (gross_profit / max(1e-4, gross_loss))
            net_pnl = balance - 10000.0
            avg_mae = np.mean(trade_maes) if trade_maes else 0.0
            avg_mfe = np.mean(trade_mfes) if trade_mfes else 0.0
            print(f"PRM >= {prm_th:.2f} | SL={sl_mult}x ATR, TP={tp_mult}x ATR | Trades={tot_trades:3d} | WR={wr:5.1f}% | PF={pf:4.2f} | Net PnL=+${net_pnl:6.2f} | Max DD=${max_dd_usd:5.2f} | Avg MAE={avg_mae:4.2f} ATR")
