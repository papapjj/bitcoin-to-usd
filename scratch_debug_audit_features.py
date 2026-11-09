import torch
import numpy as np
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

# Read audit file
path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
rows = []
with open(path, 'r') as f:
    f.readline() # header
    for i in range(15):
        line = f.readline()
        if not line: break
        parts = line.strip().split('\t')
        seq_str = parts[1]
        t_str = parts[2]
        feat_str = parts[4]
        feats = np.array([float(x) for x in feat_str.split('|')], dtype=np.float32)
        rows.append((seq_str, t_str, feats))

print(f"Loaded {len(rows)} audit feature vectors.")

class MockReq:
    def __init__(self, t):
        self.time = t
        self.equity = 10000.0
        self.positions = 0
        self.day_trades = 0

for seq_str, t_str, feats in rows:
    req = MockReq(t_str)
    
    # 1. Base router act
    base_traces = base_router.act_nam_tac_nhan(feats, None, request=req)
    base_a0 = policy._extract_action_idx(base_traces[0])
    base_a1 = policy._extract_action_idx(base_traces[1])
    
    # 2. Mythos act
    traces = policy.act_nam_tac_nhan(feats, None, request=req)
    m_a0 = policy._extract_action_idx(traces[0])
    m_a1 = policy._extract_action_idx(traces[1])
    
    # 3. Check tick result from pipeline
    # We can inspect pipeline tick result
    b_eng = policy.pipeline.bilateral_engine
    canon = policy.pipeline.extract_canonical_ingress(feats, policy.m1_buffer, policy.current_price, policy.current_atr) if hasattr(policy.pipeline, "extract_canonical_ingress") else None
    
    print(f"Seq {seq_str} ({t_str}): Base[a0={base_a0}, a1={base_a1}] -> Mythos[a0={m_a0}, a1={m_a1}]")
