import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import pandas as pd, numpy as np, torch

audit_path = '.mt5_runners/visual_7day_champion_3096_20261031_192533_123620/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
df = pd.read_csv(audit_path, sep='\t')

from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig, extract_canonical_bilateral_ingress_233

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

ingress_matrix = np.zeros((len(df), 233), dtype=np.float32)
for i in range(len(df)):
    inp_str = df['inputs'].iloc[i]
    vals = np.array([float(x) for x in inp_str.split('|') if x.strip()], dtype=np.float32)
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(vals[:137])

with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=256)
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()
    actions = verdict.actions.numpy()

# Simulate the exact policy logic on each bar:
bar_actions = []
filter_breakdown = {}

for i in range(len(df)):
    inp_str = df['inputs'].iloc[i]
    vals = np.array([float(x) for x in inp_str.split('|') if x.strip()], dtype=np.float32)
    logged_at = df['logged_at'].iloc[i]
    
    bs = b_scores[i]
    rs = r_scores[i]
    prm = prm_scores[i]
    diff = bs - rs
    act = actions[i]
    arb_dir = 1 if act == 1 else (-1 if act == 2 else 0)
    
    # 1. Neural condition
    min_margin = 0.05
    prm_gate = 0.33
    net_dir = 0
    if diff >= min_margin and bs >= 0.48 and prm >= prm_gate and (arb_dir == 1 or diff >= 0.08):
        net_dir = 1
    elif -diff >= min_margin and rs >= 0.48 and prm >= prm_gate and (arb_dir == -1 or -diff >= 0.08):
        net_dir = -1
        
    if net_dir == 0:
        filter_breakdown['NEURAL_HOLD'] = filter_breakdown.get('NEURAL_HOLD', 0) + 1
        continue
        
    # 2. Guard condition
    ema_diff_val = float(vals[11]) if vals.size > 11 else 0.0
    p_vs_e20_val = float(vals[26]) if vals.size > 26 else 0.0
    p_vs_e50_val = float(vals[27]) if vals.size > 27 else 0.0
    rel_pos_60_val = float(vals[13]) if vals.size > 13 else 0.50
    body_val = float(vals[7]) if vals.size > 7 else 0.0
    uw_val = float(vals[8]) if vals.size > 8 else 0.0
    lw_val = float(vals[9]) if vals.size > 9 else 0.0
    
    post_guard_dir = net_dir
    reason = ''
    if ema_diff_val > 0.0 and p_vs_e50_val > 0.0:
        if post_guard_dir == -1:
            post_guard_dir = 0
            reason = 'UPTREND_FORBID_SELL'
        elif post_guard_dir == 1:
            in_value_zone = p_vs_e20_val <= 0.40
            not_at_top = rel_pos_60_val <= 0.85
            rejection_confirmed = body_val >= 0.0 or lw_val >= 0.15
            if not in_value_zone:
                post_guard_dir = 0
                reason = 'BUY_NOT_IN_VALUE_ZONE'
            elif not not_at_top:
                post_guard_dir = 0
                reason = 'BUY_AT_TOP'
            elif not rejection_confirmed:
                post_guard_dir = 0
                reason = 'BUY_NO_REJECTION'
    elif ema_diff_val < 0.0 and p_vs_e50_val < 0.0:
        if post_guard_dir == 1:
            post_guard_dir = 0
            reason = 'DOWNTREND_FORBID_BUY'
        elif post_guard_dir == -1:
            in_value_zone = p_vs_e20_val >= -0.40
            not_at_bottom = rel_pos_60_val >= 0.15
            rejection_confirmed = body_val <= 0.0 or uw_val >= 0.15
            if not in_value_zone:
                post_guard_dir = 0
                reason = 'SELL_NOT_IN_VALUE_ZONE'
            elif not not_at_bottom:
                post_guard_dir = 0
                reason = 'SELL_AT_BOTTOM'
            elif not rejection_confirmed:
                post_guard_dir = 0
                reason = 'SELL_NO_REJECTION'
    else:
        if abs(diff) < 0.08:
            post_guard_dir = 0
            reason = 'MIXED_REGIME_LOW_MARGIN'
            
    if post_guard_dir == 0:
        filter_breakdown[reason] = filter_breakdown.get(reason, 0) + 1
    else:
        bar_actions.append((i, logged_at, post_guard_dir, bs, rs, diff, prm, p_vs_e20_val, rel_pos_60_val))

print(f'Total bars with post_guard_dir != 0: {len(bar_actions)}')
print('Filter breakdown:')
for k, v in sorted(filter_breakdown.items(), key=lambda x: -x[1]):
    print(f'  {k:30s}: {v:5d} ({v/len(df)*100:.1f}%)')

print('\nFirst 20 valid entries after guards:')
for b in bar_actions[:20]:
    print(f'Bar {b[0]:3d} ({b[1]}) Dir={b[2]:+d} | BS={b[3]:.2f} RS={b[4]:.2f} diff={b[5]:+.2f} | p_e20={b[7]:+.2f} rp60={b[8]:.2f}')
