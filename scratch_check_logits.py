from pathlib import Path
import numpy as np
import torch
from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

config = load_config('configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml')
base_checkpoint = 'outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl'
base_policy = tai_he_chuyen_gia_mot_context_mot_policy(base_checkpoint, config, seed=2026)
policy = MythosMT5Policy(base_policy=base_policy)

# Load triad champion
triad_pt = Path('outputs/he_chuyen_gia/v193_runs/mythos_selfplay_triad_champion.pt')
if triad_pt.exists():
    ckpt = torch.load(triad_pt, map_location='cpu', weights_only=False)
    if 'reasoning_engine_state_dict' in ckpt:
        policy.pipeline.cot_engine.load_state_dict(ckpt['reasoning_engine_state_dict'], strict=True)

audit_path = Path('mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv')
with open(audit_path, 'r') as f:
    header = f.readline().split('\t')
    lines = [l for l in f if l.strip()]

bulls, bears, margins, prms = [], [], [], []
for i, line in enumerate(lines):
    parts = line.split('\t')
    obs = np.array([float(x) for x in parts[4].split('|')], dtype=np.float32)
    res = policy.pipeline.process_tick(
        symbol='BTCUSDm',
        base_obs_137=obs,
        m1_bars_60=policy.m1_buffer,
        macro_quotes={
            'DXY': policy.macro_buffer[0],
            'US10Y': policy.macro_buffer[1],
            'SPX': policy.macro_buffer[2],
            'XAUUSD': policy.macro_buffer[3],
        },
        current_price=float(parts[19]),
    )
    meta = policy.pipeline.xlstm_adapter.metacognitive_engine
    z = policy.pipeline.last_governor_latent
    out = meta(z.unsqueeze(0))
    bull = float(out.bull_score[0].item())
    bear = float(out.bear_score[0].item())
    prm = float(out.prm_score[0].item())
    margin = float(out.adversarial_margin[0].item())
    bulls.append(bull)
    bears.append(bear)
    margins.append(margin)
    prms.append(prm)

print('OVERALL STATS ACROSS 150 BARS:')
print(f'Bull: min={min(bulls):.3f}, mean={np.mean(bulls):.3f}, max={max(bulls):.3f}')
print(f'Bear: min={min(bears):.3f}, mean={np.mean(bears):.3f}, max={max(bears):.3f}')
print(f'Margin: min={min(margins):.3f}, mean={np.mean(margins):.3f}, max={max(margins):.3f}')
print(f'PRM: min={min(prms):.3f}, mean={np.mean(prms):.3f}, max={max(prms):.3f}')
