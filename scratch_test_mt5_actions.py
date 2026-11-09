import torch
import numpy as np
import pandas as pd
from pathlib import Path
from self_growing_trader.config import load_config
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.cppi_risk_engine import CPPIConfig
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

ROOT = Path('.')
bear_path = ROOT / 'outputs' / 'he_chuyen_gia' / 'v193_runs' / 'best_promoted_checkpoint_phase2.pkl'
bull_path = ROOT / 'outputs' / 'he_chuyen_gia' / 'v193_runs' / 'multi_round_phase3' / 'round_01' / 'reasoning_rl_v1_active_candidate.pkl'
regime_model_path = ROOT / 'outputs' / 'regime_models' / 'champion_regime_detector.pkl'
regime_detector = RegimeDetector.load(regime_model_path)

cppi_cfg = CPPIConfig(max_allowed_dd=0.08, pumping_slope=2.2)
base_router = load_dual_champion_router(
    bear_path=bear_path,
    bull_path=bull_path,
    detector_path=regime_detector,
    bull_threshold=0.35,
    bear_threshold=0.35,
    chop_threshold=0.38,
    circuit_breaker_enabled=False,
    micro_scalp_enabled=False,
    cppi_config=cppi_cfg,
)

policy = MythosMT5Policy(base_policy=base_router)
policy.direct_action = True
policy.prm_threshold = 0.35

champ_pt = ROOT / 'outputs' / 'he_chuyen_gia' / 'v193_runs' / 'mythos_sniper_anti_mae_champion.pt'
if champ_pt.exists():
    raw_dict = torch.load(champ_pt, map_location='cpu', weights_only=False)
    if 'bull_cot_state_dict' in raw_dict and hasattr(policy.pipeline, 'bilateral_engine'):
        policy.pipeline.bilateral_engine.load_checkpoint(champ_pt)
        policy.pipeline.bilateral_engine.config.prm_threshold = 0.35
        if hasattr(policy.pipeline.bilateral_engine, 'arbiter'):
            policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.35

df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')

class MockReq:
    def __init__(self, t):
        self.time = t
        self.equity = 10000.0
        self.positions = 0
        self.day_trades = 0

buy_count = 0
sell_count = 0
hold_count = 0

print(f"{'Bar':<4} {'Time':<16} {'Price':<8} {'a0':<3} {'a1':<3}")
print("-" * 45)

for i in range(120):
    row = df.iloc[i]
    t = str(row['<DATE>']) + ' ' + str(row['<TIME>'])[:5]
    req = MockReq(t)
    obs = np.zeros(137, dtype=np.float32)
    c_val = float(row['<CLOSE>'])
    o_val = float(row['<OPEN>'])
    obs[0] = (c_val - o_val) / max(1.0, o_val)
    
    traces = policy.act_nam_tac_nhan(obs, goal=None, request=req)
    a0 = policy._extract_action_idx(traces[0])
    a1 = policy._extract_action_idx(traces[1])
    
    if a0 in (1, 3, 5):
        buy_count += 1
        print(f"{i:<4} {t:<16} {c_val:<8.1f} BUY(a0={a0})")
    elif a1 in (2, 4, 6):
        sell_count += 1
        print(f"{i:<4} {t:<16} {c_val:<8.1f} SELL(a1={a1})")
    else:
        hold_count += 1
        if i % 10 == 0:
            print(f"{i:<4} {t:<16} {c_val:<8.1f} HOLD")

print(f"\nSummary (first 120 bars): BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
