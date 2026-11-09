import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import pandas as pd
import numpy as np
import torch

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)

df = pd.read_csv(r'.mt5_runners\visual_7day_champion_6488_20261031_162132_907165\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_feature_audit.csv', sep='\t')
print("Total rows:", len(df))

config = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256, num_hops=4, num_debate_turns=3,
    prm_threshold=0.33, trap_threshold=0.60
)
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
engine.eval()

# Let's see what m1_buffer is doing:
# In mythos_mt5_policy.py:
# self.m1_buffer was reconstructed from base_obs_137:
# ret1 = vals[0]
# body = vals[7]
# upper_wick = vals[8]
# lower_wick = vals[9]
# range_val = vals[10]

m1_buffer = np.zeros((60, 5), dtype=np.float32)
current_price = 97000.0
current_atr = 40.0

model_buys = 0
model_sells = 0

passed_guard_buys = 0
passed_guard_sells = 0

filter_reasons = {
    'neural_hold': 0,
    'margin_too_low': 0,
    'not_in_trend': 0,
    'not_in_value_zone': 0,
    'at_extreme_top_bottom': 0,
    'no_rejection_candle': 0,
    'conf_too_low': 0,
}

for i in range(len(df)):
    inp_str = df['inputs'].iloc[i]
    vals = np.array([float(x) for x in inp_str.split('|') if x.strip()], dtype=np.float32)
    if len(vals) < 137:
        padded = np.zeros(137, dtype=np.float32)
        padded[:len(vals)] = vals
        base_obs_137 = padded
    else:
        base_obs_137 = vals[:137]
        
    ret1 = float(base_obs_137[0])
    body_val = float(base_obs_137[7])
    uw_val = max(0.0, float(base_obs_137[8]))
    lw_val = max(0.0, float(base_obs_137[9]))
    range_val = max(0.0, float(base_obs_137[10]))
    
    current_price += (ret1 * current_atr)
    o = current_price - (body_val * current_atr)
    h = max(o, current_price) + (uw_val * current_atr)
    l = min(o, current_price) - (lw_val * current_atr)
    v = 150.0
    
    m1_buffer[:-1] = m1_buffer[1:]
    m1_buffer[-1] = [o, h, l, current_price, v]
    
    if i < 60: continue
    
    ingress = extract_canonical_bilateral_ingress_233(base_obs_137, m1_buffer, current_price, current_atr)
    with torch.no_grad():
        v_out = engine(torch.from_numpy(ingress).unsqueeze(0).float())
        arb_act = int(v_out.action) if hasattr(v_out.action, "item") else int(v_out.action)
        bs = float(v_out.bull_score.item()) if hasattr(v_out.bull_score, "item") else float(v_out.bull_score)
        rs = float(v_out.bear_score.item()) if hasattr(v_out.bear_score, "item") else float(v_out.bear_score)
        prm = float(v_out.prm_score.item()) if hasattr(v_out.prm_score, "item") else float(v_out.prm_score)
        
    diff = bs - rs
    min_margin = 0.05
    prm_gate = 0.30
    
    net_dir = 0
    if diff >= min_margin and bs >= 0.48 and prm >= prm_gate and (arb_act == 1 or diff >= 0.08):
        net_dir = 1
        model_buys += 1
    elif -diff >= min_margin and rs >= 0.48 and prm >= prm_gate and (arb_act == -1 or -diff >= 0.08):
        net_dir = -1
        model_sells += 1
    else:
        filter_reasons['neural_hold'] += 1
        continue
        
    # Check trend & guards
    ema_diff_val = float(base_obs_137[11])
    p_vs_e20_val = float(base_obs_137[26])
    p_vs_e50_val = float(base_obs_137[27])
    rel_pos_60_val = float(base_obs_137[13])
    
    if ema_diff_val > 0.0 and p_vs_e50_val > 0.0:
        if net_dir == -1:
            filter_reasons['not_in_trend'] += 1
            net_dir = 0
        elif net_dir == 1:
            in_value_zone = p_vs_e20_val <= 0.40
            not_at_top = rel_pos_60_val <= 0.68
            rejection_confirmed = body_val >= 0.0 or lw_val >= 0.20
            if not in_value_zone: filter_reasons['not_in_value_zone'] += 1
            elif not not_at_top: filter_reasons['at_extreme_top_bottom'] += 1
            elif not rejection_confirmed: filter_reasons['no_rejection_candle'] += 1
            else:
                passed_guard_buys += 1
    elif ema_diff_val < 0.0 and p_vs_e50_val < 0.0:
        if net_dir == 1:
            filter_reasons['not_in_trend'] += 1
            net_dir = 0
        elif net_dir == -1:
            in_value_zone = p_vs_e20_val >= -0.40
            not_at_bottom = rel_pos_60_val >= 0.32
            rejection_confirmed = body_val <= 0.0 or uw_val >= 0.20
            if not in_value_zone: filter_reasons['not_in_value_zone'] += 1
            elif not not_at_bottom: filter_reasons['at_extreme_top_bottom'] += 1
            elif not rejection_confirmed: filter_reasons['no_rejection_candle'] += 1
            else:
                passed_guard_sells += 1
    else:
        filter_reasons['not_in_trend'] += 1
        net_dir = 0

print(f"Model BUY signals: {model_buys} | Model SELL signals: {model_sells}")
print(f"Passed BUYs: {passed_guard_buys} | Passed SELLs: {passed_guard_sells}")
print("Filter reasons breakdown:", filter_reasons)
