import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import pandas as pd, numpy as np, torch

audit_path = '.mt5_runners/visual_7day_champion_3096_20261031_192533_123620/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
df = pd.read_csv(audit_path, sep='\t')
print(f'Total audit rows: {len(df)}')

# Let's inspect the inputs and conditions
rejection_reasons = {}

for i in range(len(df)):
    inp_str = df['inputs'].iloc[i]
    vals = np.array([float(x) for x in inp_str.split('|') if x.strip()], dtype=np.float32)
    
    # Extract features exactly as in policy
    ema_diff_val = float(vals[11]) if vals.size > 11 else 0.0
    p_vs_e20_val = float(vals[26]) if vals.size > 26 else 0.0
    p_vs_e50_val = float(vals[27]) if vals.size > 27 else 0.0
    rel_pos_60_val = float(vals[13]) if vals.size > 13 else 0.50
    body_val = float(vals[7]) if vals.size > 7 else 0.0
    uw_val = float(vals[8]) if vals.size > 8 else 0.0
    lw_val = float(vals[9]) if vals.size > 9 else 0.0
    
    # BUY Guard Check:
    if ema_diff_val > 0.0 and p_vs_e50_val > 0.0:
        in_value_zone = p_vs_e20_val <= 0.40
        not_at_top = rel_pos_60_val <= 0.85
        rejection_confirmed = body_val >= 0.0 or lw_val >= 0.15
        if not in_value_zone:
            rejection_reasons['BUY_OUT_OF_VALUE_ZONE'] = rejection_reasons.get('BUY_OUT_OF_VALUE_ZONE', 0) + 1
        elif not not_at_top:
            rejection_reasons['BUY_AT_TOP'] = rejection_reasons.get('BUY_AT_TOP', 0) + 1
        elif not rejection_confirmed:
            rejection_reasons['BUY_NO_REJECTION'] = rejection_reasons.get('BUY_NO_REJECTION', 0) + 1
        else:
            rejection_reasons['BUY_GUARD_PASSED'] = rejection_reasons.get('BUY_GUARD_PASSED', 0) + 1
    elif ema_diff_val < 0.0 and p_vs_e50_val < 0.0:
        in_value_zone = p_vs_e20_val >= -0.40
        not_at_bottom = rel_pos_60_val >= 0.15
        rejection_confirmed = body_val <= 0.0 or uw_val >= 0.15
        if not in_value_zone:
            rejection_reasons['SELL_OUT_OF_VALUE_ZONE'] = rejection_reasons.get('SELL_OUT_OF_VALUE_ZONE', 0) + 1
        elif not not_at_bottom:
            rejection_reasons['SELL_AT_BOTTOM'] = rejection_reasons.get('SELL_AT_BOTTOM', 0) + 1
        elif not rejection_confirmed:
            rejection_reasons['SELL_NO_REJECTION'] = rejection_reasons.get('SELL_NO_REJECTION', 0) + 1
        else:
            rejection_reasons['SELL_GUARD_PASSED'] = rejection_reasons.get('SELL_GUARD_PASSED', 0) + 1
    else:
        rejection_reasons['MIXED_REGIME'] = rejection_reasons.get('MIXED_REGIME', 0) + 1

print('Guard Statistics across audit bars:')
for k, v in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
    print(f'  {k:30s}: {v:5d} ({v/len(df)*100:.1f}%)')
