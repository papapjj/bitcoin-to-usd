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
raw_dict = torch.load(champ_pt, map_location='cpu', weights_only=False)
policy.pipeline.bilateral_engine.load_checkpoint(champ_pt)
policy.pipeline.bilateral_engine.config.prm_threshold = 0.35
policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.35

# Read audit file row for seq 3
audit_path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
with open(audit_path, 'r') as f:
    f.readline()
    f.readline() # seq 1
    f.readline() # seq 2
    l_seq3 = f.readline() # seq 3

parts = l_seq3.strip().split('\t')
ev_id = parts[1]
t_str = parts[2]
feats = np.array([float(x) for x in parts[4].split('|')], dtype=np.float32)

print(f"Testing ev_id={ev_id}, t_str={t_str}")

class MockReq:
    def __init__(self, t):
        self.time = t
        self.equity = 10000.0
        self.positions = 0
        self.day_trades = 0

req = MockReq(t_str)

# Run act_nam_tac_nhan
traces = policy.act_nam_tac_nhan(feats, None, request=req)

print(f"Traces count: {len(traces)}")
for idx, tr in enumerate(traces):
    act = policy._extract_action_idx(tr)
    details = getattr(tr, 'action_details', {})
    print(f"Agent {idx}: action={act}, five_action_index={details.get('five_action_index')}, wire_action={details.get('wire_action')}, reason={details.get('mythos_verdict')}")
