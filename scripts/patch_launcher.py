with open('scripts/launch_7day_visual_master_champion.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = """    two_tier = TwoTierConfig(
        initial_sl_atr=2.0,
        breakeven_trigger_atr=1.00,
        breakeven_lock_atr=0.25,
        partial_tp_atr=2.00,
        trailing_atr=1.20,
    )"""

replacement = """    two_tier = TwoTierConfig(
        initial_sl_atr=2.0,
        breakeven_trigger_atr=0.80,
        breakeven_lock_atr=0.35,
        partial_tp_atr=2.20,
        trailing_atr=0.80,
    )"""

code = code.replace(target, replacement)

target_policy = """    policy.scalp_sl_atr = 2.0
    policy.sniper_sl_atr = 2.0
    policy.scalp_tp_rr = 1.00
    policy.sniper_tp_rr = 1.25
    policy.trailing_atr = 1.20"""

replacement_policy = """    policy.scalp_sl_atr = 2.0
    policy.sniper_sl_atr = 2.0
    policy.scalp_tp_rr = 1.10
    policy.sniper_tp_rr = 1.10
    policy.trailing_atr = 0.80"""

code = code.replace(target_policy, replacement_policy)

code = code.replace("policy.prm_threshold = 0.30", "policy.prm_threshold = 0.35")
code = code.replace("policy.min_margin_threshold = 0.10", "policy.min_margin_threshold = 0.08")

with open('scripts/launch_7day_visual_master_champion.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("launch_7day_visual_master_champion.py updated successfully!")
