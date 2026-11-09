import torch
import numpy as np
from pathlib import Path
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_pure_anti_mae_engine import build_pure_anti_mae_dataset

data = build_pure_anti_mae_dataset()
cfg = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, prm_threshold=0.35)
engine = BilateralDualCoTEngine(cfg)
ckpt_path = Path('outputs/he_chuyen_gia/v193_runs/mythos_sniper_anti_mae_champion.pt')
engine.load_checkpoint(ckpt_path)
engine.eval()

dates = data['dates']
test_mask = ~np.isin(dates, ['2025.02.20', '2025.02.21', '2025.02.22', '2025.02.23', '2025.02.24'])
test_indices = np.where(test_mask)[0]

lookback = data["lookback_bars"]
horizon = data["horizon_bars"]
closes = data["closes"]
highs = data["highs"]
lows = data["lows"]
atrs = data["atrs"]
ingress = data["ingress"]

with torch.no_grad():
    x = torch.from_numpy(ingress[test_indices]).float()
    z = engine.state_adapter(x)
    bull_out = engine.bull_cot(z)
    bear_out = engine.bear_cot(z)
    p_th = bull_out.cot_hops[:, -1, :]
    a_th = bear_out.cot_hops[:, -1, :]
    f_p, f_a, _, _, _, _ = engine.cross_examination(p_th, a_th, turns=3)
    synth = torch.cat([f_p, f_a], dim=-1)
    logits = engine.arbiter.arbiter_net(synth)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    prms = engine.arbiter.prm_evaluator(synth).squeeze(-1).cpu().numpy()

p_buy = probs[:, 1]
p_sell = probs[:, 2]

print("=== Pure Neural Directional Contrast & PRM Gating (OOS Test Split) ===")
print("Testing SL=1.2 ATR, TP=1.8 ATR (Tight Anti-MAE Sniper)")
for prm_th in [0.45, 0.50, 0.55, 0.60, 0.65]:
    for diff_th in [0.05, 0.08, 0.12]:
        buy_wins, buy_losses = 0, 0
        sell_wins, sell_losses = 0, 0
        buy_maes, sell_maes = [], []
        
        for local_i, global_i in enumerate(test_indices):
            prm = prms[local_i]
            if prm < prm_th:
                continue
            
            pb = p_buy[local_i]
            ps = p_sell[local_i]
            
            act = 0
            if pb - ps >= diff_th:
                act = 1
            elif ps - pb >= diff_th:
                act = -1
            else:
                continue
                
            t = lookback + global_i
            cur_p = closes[t]
            cur_atr = atrs[global_i]
            f_h = highs[t+1 : t+1+horizon]
            f_l = lows[t+1 : t+1+horizon]
            
            if act == 1:
                sl_p = cur_p - 1.2 * cur_atr
                tp_p = cur_p + 1.8 * cur_atr
                win = False
                worst_adverse = 0.0
                for bar_k in range(len(f_h)):
                    adv = (cur_p - f_l[bar_k]) / cur_atr
                    if adv > worst_adverse:
                        worst_adverse = adv
                    if f_l[bar_k] <= sl_p:
                        break
                    if f_h[bar_k] >= tp_p:
                        win = True
                        break
                buy_maes.append(worst_adverse)
                if win:
                    buy_wins += 1
                else:
                    buy_losses += 1
            elif act == -1:
                sl_p = cur_p + 1.2 * cur_atr
                tp_p = cur_p - 1.8 * cur_atr
                win = False
                worst_adverse = 0.0
                for bar_k in range(len(f_h)):
                    adv = (f_h[bar_k] - cur_p) / cur_atr
                    if adv > worst_adverse:
                        worst_adverse = adv
                    if f_h[bar_k] >= sl_p:
                        break
                    if f_l[bar_k] <= tp_p:
                        win = True
                        break
                sell_maes.append(worst_adverse)
                if win:
                    sell_wins += 1
                else:
                    sell_losses += 1
                    
        tot = buy_wins + buy_losses + sell_wins + sell_losses
        if tot >= 20:
            wr = ((buy_wins + sell_wins) / tot) * 100.0
            b_wr = (buy_wins / max(1, buy_wins + buy_losses)) * 100.0
            s_wr = (sell_wins / max(1, sell_wins + sell_losses)) * 100.0
            mae = float(np.mean(buy_maes + sell_maes))
            print(f"PRM >= {prm_th:.2f} | Diff >= {diff_th:.2f} | Trades={tot:3d} | WR={wr:5.1f}% (BUY: {b_wr:5.1f}% [{buy_wins+buy_losses}], SELL: {s_wr:5.1f}% [{sell_wins+sell_losses}]) | MAE={mae:4.2f} ATR")
