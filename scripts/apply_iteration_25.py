with open('scripts/run_mythos_mt5_live.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Check policy settings
assert 'mythos_policy.sniper_sl_atr = 3.20' in text, "sniper_sl_atr not found"
text = text.replace('mythos_policy.sniper_sl_atr = 3.20', 'mythos_policy.sniper_sl_atr = 2.80')
text = text.replace('mythos_policy.sniper_tp_rr = 0.75', 'mythos_policy.sniper_tp_rr = 1.071')
text = text.replace('mythos_policy.scalp_sl_atr = 3.20', 'mythos_policy.scalp_sl_atr = 2.80')
text = text.replace('mythos_policy.scalp_tp_rr = 0.75', 'mythos_policy.scalp_tp_rr = 1.071')
text = text.replace('mythos_policy.trailing_atr = 1.20', 'mythos_policy.trailing_atr = 0.85')
text = text.replace('mythos_policy.min_spacing_bars = 1', 'mythos_policy.min_spacing_bars = 3')
text = text.replace('mythos_policy.prm_threshold = 0.16', 'mythos_policy.prm_threshold = 0.18')

# Check MT5BacktestConfig kwargs
assert '"fixed_sl_atr": 3.20' in text, "fixed_sl_atr not found"
text = text.replace('"fixed_sl_atr": 3.20', '"fixed_sl_atr": 2.80')
text = text.replace('"fixed_tp_atr": 2.40', '"fixed_tp_atr": 3.00')
text = text.replace('"min_rr": 0.70', '"min_rr": 0.95')
text = text.replace('"technical_min_rr": 0.70', '"technical_min_rr": 0.95')
text = text.replace('"trail_activation_atr": 1.30', '"trail_activation_atr": 1.25')
text = text.replace('"trail_lock_atr": 0.50', '"trail_lock_atr": 0.80')

with open('scripts/run_mythos_mt5_live.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully applied Iteration 25 parameters to run_mythos_mt5_live.py')
