import re
from pathlib import Path

# 1. Update research_reasoning_world_class/src/mythos_mt5_policy.py
policy_path = Path('research_reasoning_world_class/src/mythos_mt5_policy.py')
with open(policy_path, 'r', encoding='utf-8') as f:
    pol_text = f.read()

target = '''        # Macro Trend Alignment Guard (Strict Trend-Following: never fight the macro trend)
        if net_dir == -1 and (ema_diff_val > 0.0 or p_vs_e50_val > 0.0):
            net_dir = 0  # Strictly forbid shorting when trend is bullish (EMA20 > EMA60 or Price > EMA50)
        elif net_dir == 1 and p_vs_e50_val < -1.80:
            net_dir = 0  # Only veto extreme dumps, allowing agile trade frequency >= 40/d'''

replacement = '''        # Macro Trend Alignment Guard (Strict Trend-Following: never fight the macro trend)
        if net_dir == -1 and (ema_diff_val > -0.05 or p_vs_e50_val > 0.10):
            net_dir = 0  # Strictly forbid shorting when trend is bullish (Bull Specialist owns market)
        elif net_dir == 1 and (ema_diff_val < 0.05 or p_vs_e50_val < -0.10):
            net_dir = 0  # Strictly forbid buying when trend is bearish (Bear Specialist owns market)'''

assert target in pol_text, "Target not found in mythos_mt5_policy.py"
pol_text = pol_text.replace(target, replacement)
with open(policy_path, 'w', encoding='utf-8') as f:
    f.write(pol_text)
print("Successfully updated mythos_mt5_policy.py with Symmetric Trend-Following Guard")

# 2. Update SelfGrowing_MT5_Trainer_EA.mq5
ea_path = Path('mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5')
with open(ea_path, 'r', encoding='utf-8') as f:
    ea_text = f.read()

ea_text = re.sub(r'input int MaxPositions\s*=\s*\d+;', 'input int MaxPositions = 2;', ea_text)
ea_text = re.sub(r'input double FixedSlAtr\s*=\s*[\d\.]+;', 'input double FixedSlAtr = 2.40;', ea_text)
ea_text = re.sub(r'input double FixedTpAtr\s*=\s*[\d\.]+;', 'input double FixedTpAtr = 2.60;', ea_text)
ea_text = re.sub(r'input double TrailActivationAtr\s*=\s*[\d\.]+;', 'input double TrailActivationAtr = 1.10;', ea_text)
ea_text = re.sub(r'input double TrailLockAtr\s*=\s*[\d\.]+;', 'input double TrailLockAtr = 0.70;', ea_text)

with open(ea_path, 'w', encoding='utf-8') as f:
    f.write(ea_text)
print('Successfully updated SelfGrowing_MT5_Trainer_EA.mq5 for Iteration 29')

# 3. Update run_mythos_mt5_live.py
py_path = Path('scripts/run_mythos_mt5_live.py')
with open(py_path, 'r', encoding='utf-8') as f:
    py_text = f.read()

# Unleash Champion SELL!
py_text = re.sub(r'mythos_policy\.long_only\s*=\s*True', 'mythos_policy.long_only = False', py_text)
py_text = re.sub(r'mythos_policy\.sniper_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.sniper_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.sniper_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.sniper_tp_rr = 1.0833', py_text)
py_text = re.sub(r'mythos_policy\.scalp_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.scalp_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.scalp_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.scalp_tp_rr = 1.0833', py_text)
py_text = re.sub(r'mythos_policy\.trailing_atr\s*=\s*[\d\.]+', 'mythos_policy.trailing_atr = 0.75', py_text)
py_text = re.sub(r'mythos_policy\.min_spacing_bars\s*=\s*\d+', 'mythos_policy.min_spacing_bars = 1', py_text)
py_text = re.sub(r'mythos_policy\.prm_threshold\s*=\s*[\d\.]+', 'mythos_policy.prm_threshold = 0.15', py_text)
py_text = re.sub(r'mythos_policy\.loss_cooldown_bars\s*=\s*\d+', 'mythos_policy.loss_cooldown_bars = 6', py_text)
py_text = re.sub(r'mythos_policy\.max_positions\s*=\s*\d+', 'mythos_policy.max_positions = 2', py_text)

py_text = re.sub(r'"max_positions":\s*\d+', '"max_positions": 2', py_text)
py_text = re.sub(r'"fixed_sl_atr":\s*[\d\.]+', '"fixed_sl_atr": 2.40', py_text)
py_text = re.sub(r'"fixed_tp_atr":\s*[\d\.]+', '"fixed_tp_atr": 2.60', py_text)
py_text = re.sub(r'"min_rr":\s*[\d\.]+', '"min_rr": 0.90', py_text)
py_text = re.sub(r'"technical_min_rr":\s*[\d\.]+', '"technical_min_rr": 0.90', py_text)
py_text = re.sub(r'"trail_activation_atr":\s*[\d\.]+', '"trail_activation_atr": 1.10', py_text)
py_text = re.sub(r'"trail_lock_atr":\s*[\d\.]+', '"trail_lock_atr": 0.70', py_text)

with open(py_path, 'w', encoding='utf-8') as f:
    f.write(py_text)
print('Successfully applied Iteration 29 parameters to run_mythos_mt5_live.py (Champion SELL UNLEASHED!)')
