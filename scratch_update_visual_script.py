with open('scripts/launch_visual_mt5_strategy_tester.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_candidate = 'ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_sniper_anti_mae_champion.pt",'
new_candidate = 'ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt",\n        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_sniper_anti_mae_champion.pt",'

text = text.replace(old_candidate, new_candidate)
text = text.replace('policy.scalp_sl_atr = 1.8', 'policy.scalp_sl_atr = 1.5')
text = text.replace('policy.scalp_tp_rr = 1.25', 'policy.scalp_tp_rr = 1.50')
text = text.replace('policy.sniper_sl_atr = 1.8', 'policy.sniper_sl_atr = 1.5')
text = text.replace('policy.sniper_tp_rr = 1.25', 'policy.sniper_tp_rr = 1.50')

with open('scripts/launch_visual_mt5_strategy_tester.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated launch_visual_mt5_strategy_tester.py successfully!')
