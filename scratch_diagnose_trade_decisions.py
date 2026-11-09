import torch
import numpy as np
import pandas as pd
from pathlib import Path
from self_growing_trader.config import load_config
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.cppi_risk_engine import CPPIConfig
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.bilateral_dual_cot import extract_canonical_bilateral_ingress_233

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

b_eng = policy.pipeline.bilateral_engine

# Read audit file
audit_path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
audit_data = {}
with open(audit_path, 'r') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            seq = parts[1]
            t = parts[2]
            feats = np.array([float(x) for x in parts[4].split('|')], dtype=np.float32)
            audit_data[seq] = (t, feats)

print(f"Loaded {len(audit_data)} audit steps.")

# Read execution deals that lost
exec_path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df_ex = pd.read_csv(exec_path, sep='\t')
losing_requests = df_ex[(df_ex['requested_action'] == 2) & (df_ex['status'] == 'ACCEPTED')]
print(f"Found {len(losing_requests)} ACCEPTED SELL requests.")

print(f"\n{'Seq':<12} {'Time':<16} {'b_sc':<6} {'r_sc':<6} {'TrapProbs [No, Bull, Bear]':<28} {'ArbAct':<7} {'PRM':<6} {'CP'}")
print("-" * 95)

seen_events = set()
for _, row in losing_requests.iterrows():
    ev_id = str(row['event_id'])
    if ev_id in seen_events:
        continue
    seen_events.add(ev_id)
    if ev_id not in audit_data:
        continue
    
    t_str, feats = audit_data[ev_id]
    
    # Run bilateral engine forward step-by-step
    with torch.no_grad():
        # Get canonical ingress
        cur_p = float(row.get('entry_price', 96500.0))
        cur_atr = float(policy.current_atr)
        
        # Test 1: Using canon ingress from feats
        canon_in = extract_canonical_bilateral_ingress_233(
            base_obs_137=feats,
            m1_bars_60=policy.m1_buffer,
            cur_p=cur_p,
            cur_atr=cur_atr,
        )
        z_in = torch.from_numpy(canon_in).float().unsqueeze(0)
        
        # Adapt z
        z = b_eng.state_adapter(z_in)
        
        # Bull CoT & Bear CoT
        bull_out = b_eng.bull_cot(z)
        bear_out = b_eng.bear_cot(z)
        b_score = float(bull_out.bull_score[0].item())
        r_score = float(bear_out.bear_score[0].item())
        
        p_thought = bull_out.cot_hops[:, -1, :]
        a_thought = bear_out.cot_hops[:, -1, :]
        
        # Trap detector
        joint = torch.cat([a_thought, p_thought], dim=-1)
        trap_logits = b_eng.trap_detector(joint)
        trap_probs = torch.softmax(trap_logits, dim=-1)[0]
        
        # Cross exam
        final_p, final_a, _, _, _, _ = b_eng.cross_examination(p_thought, a_thought, turns=b_eng.config.num_debate_turns)
        
        # Arbiter
        convictions = torch.tensor([[b_score, r_score]], dtype=torch.float32)
        actions, prms, confs, cps = b_eng.arbiter(final_p, final_a, trap_probs.unsqueeze(0), convictions)
        
        synthesis = torch.cat([final_p, final_a], dim=-1)
        arb_logits = b_eng.arbiter.arbiter_net(synthesis)
        arb_probs = torch.softmax(arb_logits, dim=-1)[0]
        
        tp_str = f"[{trap_probs[0]:.2f}, {trap_probs[1]:.2f}, {trap_probs[2]:.2f}]"
        ap_str = f"ArbProbs[H={arb_probs[0]:.2f}, B={arb_probs[1]:.2f}, S={arb_probs[2]:.2f}]"
        print(f"{ev_id:<12} {t_str:<16} {b_score:<6.2f} {r_score:<6.2f} {tp_str:<28} {actions[0].item():<7} {prms[0].item():<6.2f} cp={cps[0].item()} | {ap_str}")
