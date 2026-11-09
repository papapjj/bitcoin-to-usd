import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import pandas as pd, numpy as np, torch

from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig, extract_canonical_bilateral_ingress_233

audit_path = '.mt5_runners/visual_7day_champion_3096_20261031_192533_123620/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
df = pd.read_csv(audit_path, sep='\t')
print(f'Total audit rows: {len(df)}')

config = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.33,
    trap_threshold=0.60,
)
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint('outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt')
engine.eval()

# Let's inspect the ingress and engine predictions on all audit rows
ingress_matrix = np.zeros((len(df), 233), dtype=np.float32)
for i in range(len(df)):
    inp_str = df['inputs'].iloc[i]
    vals = np.array([float(x) for x in inp_str.split('|') if x.strip()], dtype=np.float32)
    # in live MT5: extract_canonical_bilateral_ingress_233 was called with m1_buffer
    # Here base_obs_137 is vals[:137]
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(vals[:137])

with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=256)
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()
    actions = verdict.actions.numpy()

print(f'Actions distribution: 0={np.sum(actions == 0)}, 1={np.sum(actions == 1)}, 2={np.sum(actions == 2)}')
print(f'b_scores: min={np.min(b_scores):.3f}, max={np.max(b_scores):.3f}, mean={np.mean(b_scores):.3f}')
print(f'r_scores: min={np.min(r_scores):.3f}, max={np.max(r_scores):.3f}, mean={np.mean(r_scores):.3f}')
print(f'prm_scores: min={np.min(prm_scores):.3f}, max={np.max(prm_scores):.3f}, mean={np.mean(prm_scores):.3f}')

# Check how many bars satisfy the policy condition:
# if diff >= min_margin and b_s >= 0.48 and prm >= prm_gate and (arb_act == 1 or diff >= 0.08):
buy_cands = []
sell_cands = []

for i in range(len(df)):
    bs = b_scores[i]
    rs = r_scores[i]
    prm = prm_scores[i]
    diff = bs - rs
    act = actions[i] # 0: HOLD, 1: BUY, 2: SELL
    arb_dir = 1 if act == 1 else (-1 if act == 2 else 0)
    
    is_buy = (diff >= 0.05 and bs >= 0.48 and prm >= 0.33 and (arb_dir == 1 or diff >= 0.08))
    is_sell = (-diff >= 0.05 and rs >= 0.48 and prm >= 0.33 and (arb_dir == -1 or -diff >= 0.08))
    
    if is_buy: buy_cands.append(i)
    if is_sell: sell_cands.append(i)

print(f'Neural BUY candidates: {len(buy_cands)}, Neural SELL candidates: {len(sell_cands)}')
