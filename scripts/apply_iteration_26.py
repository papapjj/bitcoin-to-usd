import re
from pathlib import Path

# 1. Update SelfGrowing_MT5_Trainer_EA.mq5
ea_path = Path('mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5')
with open(ea_path, 'r', encoding='utf-8') as f:
    ea_text = f.read()

# Replace inputs
ea_text = re.sub(r'input double FixedSlAtr\s*=\s*[\d\.]+;', 'input double FixedSlAtr = 3.00;', ea_text)
ea_text = re.sub(r'input double FixedTpAtr\s*=\s*[\d\.]+;', 'input double FixedTpAtr = 2.60;', ea_text)
ea_text = re.sub(r'input double TrailActivationAtr\s*=\s*[\d\.]+;', 'input double TrailActivationAtr = 1.15;', ea_text)
ea_text = re.sub(r'input double TrailLockAtr\s*=\s*[\d\.]+;', 'input double TrailLockAtr = 0.65;', ea_text)

with open(ea_path, 'w', encoding='utf-8') as f:
    f.write(ea_text)
print('Successfully updated SelfGrowing_MT5_Trainer_EA.mq5')

# 2. Update run_mythos_mt5_live.py
py_path = Path('scripts/run_mythos_mt5_live.py')
with open(py_path, 'r', encoding='utf-8') as f:
    py_text = f.read()

py_text = re.sub(r'mythos_policy\.sniper_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.sniper_sl_atr = 3.00', py_text)
py_text = re.sub(r'mythos_policy\.sniper_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.sniper_tp_rr = 0.8667', py_text)
py_text = re.sub(r'mythos_policy\.scalp_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.scalp_sl_atr = 3.00', py_text)
py_text = re.sub(r'mythos_policy\.scalp_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.scalp_tp_rr = 0.8667', py_text)
py_text = re.sub(r'mythos_policy\.trailing_atr\s*=\s*[\d\.]+', 'mythos_policy.trailing_atr = 0.90', py_text)
py_text = re.sub(r'mythos_policy\.min_spacing_bars\s*=\s*\d+', 'mythos_policy.min_spacing_bars = 3', py_text)
py_text = re.sub(r'mythos_policy\.prm_threshold\s*=\s*[\d\.]+', 'mythos_policy.prm_threshold = 0.22', py_text)
py_text = re.sub(r'mythos_policy\.loss_cooldown_bars\s*=\s*\d+', 'mythos_policy.loss_cooldown_bars = 20', py_text)

py_text = re.sub(r'"fixed_sl_atr":\s*[\d\.]+', '"fixed_sl_atr": 3.00', py_text)
py_text = re.sub(r'"fixed_tp_atr":\s*[\d\.]+', '"fixed_tp_atr": 2.60', py_text)
py_text = re.sub(r'"min_rr":\s*[\d\.]+', '"min_rr": 0.80', py_text)
py_text = re.sub(r'"technical_min_rr":\s*[\d\.]+', '"technical_min_rr": 0.80', py_text)
py_text = re.sub(r'"trail_activation_atr":\s*[\d\.]+', '"trail_activation_atr": 1.15', py_text)
py_text = re.sub(r'"trail_lock_atr":\s*[\d\.]+', '"trail_lock_atr": 0.65', py_text)

with open(py_path, 'w', encoding='utf-8') as f:
    f.write(py_text)
print('Successfully applied Iteration 26 parameters to run_mythos_mt5_live.py')
