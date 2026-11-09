import torch
import numpy as np
from pathlib import Path
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.train_pure_anti_mae_engine import build_pure_anti_mae_dataset, evaluate_split

data = build_pure_anti_mae_dataset()
dates = data['dates']
test_mask = ~np.isin(dates, ['2025.02.20', '2025.02.21', '2025.02.22', '2025.02.23', '2025.02.24'])
test_indices = np.where(test_mask)[0]
print(f'Evaluating on OOS split: {len(test_indices)} bars')

cfg = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, prm_threshold=0.35)
engine = BilateralDualCoTEngine(cfg)
ckpt_path = Path('outputs/he_chuyen_gia/v193_runs/mythos_symmetric_anti_mae_champion.pt')
engine.load_checkpoint(ckpt_path)
engine.eval()

print('\n=== Symmetrical Anti-MAE Champion OOS Evaluation (SL=1.2 ATR, TP=1.8 ATR) ===')
for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    res = evaluate_split(engine, data['ingress'], test_indices, data, prm_threshold=thresh, sl_mult=1.2, tp_mult=1.8)
    tr = res['trades']
    wr = res['wr']
    b_wr = res['buy_wr']
    b_tr = res['buy_trades']
    s_wr = res['sell_wr']
    s_tr = res['sell_trades']
    mae = res['avg_mae']
    print(f'PRM >= {thresh:.2f} | Trades={tr:3d} | WR={wr:5.1f}% (BUY: {b_wr:5.1f}% [{b_tr:2d}], SELL: {s_wr:5.1f}% [{s_tr:2d}]) | Avg MAE={mae:4.2f} ATR')

print('\n=== Symmetrical Anti-MAE Champion OOS Evaluation (SL=1.5 ATR, TP=2.25 ATR) ===')
for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    res = evaluate_split(engine, data['ingress'], test_indices, data, prm_threshold=thresh, sl_mult=1.5, tp_mult=2.25)
    tr = res['trades']
    wr = res['wr']
    b_wr = res['buy_wr']
    b_tr = res['buy_trades']
    s_wr = res['sell_wr']
    s_tr = res['sell_trades']
    mae = res['avg_mae']
    print(f'PRM >= {thresh:.2f} | Trades={tr:3d} | WR={wr:5.1f}% (BUY: {b_wr:5.1f}% [{b_tr:2d}], SELL: {s_wr:5.1f}% [{s_tr:2d}]) | Avg MAE={mae:4.2f} ATR')
