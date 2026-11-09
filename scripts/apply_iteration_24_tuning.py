from pathlib import Path

# 1. Update SelfGrowing_MT5_Trainer_EA.mq5
p_ea = Path("mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5")
ea_text = p_ea.read_text(encoding="utf-8")

old_ea = """input int MaxPositions = 4;
input int MagicNumber = 123456;
input double FixedSlAtr = 3.50;
input double FixedTpAtr = 3.20;
input double TrailActivationAtr = 1.35;
input double TrailLockAtr = 0.45;"""

new_ea = """input int MaxPositions = 2;
input int MagicNumber = 123456;
input double FixedSlAtr = 3.20;
input double FixedTpAtr = 2.40;
input double TrailActivationAtr = 1.30;
input double TrailLockAtr = 0.50;"""

assert old_ea in ea_text, "old_ea not found"
ea_text = ea_text.replace(old_ea, new_ea, 1)
p_ea.write_text(ea_text, encoding="utf-8")
print("SelfGrowing_MT5_Trainer_EA.mq5 updated.")

# 2. Update scripts/run_mythos_mt5_live.py
p_run = Path("scripts/run_mythos_mt5_live.py")
run_text = p_run.read_text(encoding="utf-8")

old_run1 = """    mythos_policy.sniper_sl_atr = 3.50
    mythos_policy.sniper_tp_rr = 0.914
    mythos_policy.scalp_sl_atr = 3.50
    mythos_policy.scalp_tp_rr = 0.914
    mythos_policy.trailing_atr = 1.20
    mythos_policy.long_only = True
    mythos_policy.max_positions = 4"""

new_run1 = """    mythos_policy.sniper_sl_atr = 3.20
    mythos_policy.sniper_tp_rr = 0.75
    mythos_policy.scalp_sl_atr = 3.20
    mythos_policy.scalp_tp_rr = 0.75
    mythos_policy.trailing_atr = 1.20
    mythos_policy.long_only = True
    mythos_policy.max_positions = 2"""

if old_run1 not in run_text:
    # Check lines around max_positions
    old_run1_alt = """    mythos_policy.sniper_sl_atr = 3.50
    mythos_policy.sniper_tp_rr = 0.914
    mythos_policy.scalp_sl_atr = 3.50
    mythos_policy.scalp_tp_rr = 0.914
    mythos_policy.trailing_atr = 1.20
    mythos_policy.long_only = True"""
    assert old_run1_alt in run_text, "old_run1_alt not found"
    new_run1_alt = """    mythos_policy.sniper_sl_atr = 3.20
    mythos_policy.sniper_tp_rr = 0.75
    mythos_policy.scalp_sl_atr = 3.20
    mythos_policy.scalp_tp_rr = 0.75
    mythos_policy.trailing_atr = 1.20
    mythos_policy.long_only = True"""
    run_text = run_text.replace(old_run1_alt, new_run1_alt, 1)
    run_text = run_text.replace("mythos_policy.max_positions = 4", "mythos_policy.max_positions = 2", 1)
else:
    run_text = run_text.replace(old_run1, new_run1, 1)

old_run2 = """"max_positions": 4,
        "fixed_sl_atr": 3.50,
        "fixed_tp_atr": 3.20,
        "min_rr": 0.85,
        "technical_min_rr": 0.85,
        "max_spread_cost_r": 0.28,
        "trail_activation_atr": 1.35,
        "trail_lock_atr": 0.45,"""

new_run2 = """"max_positions": 2,
        "fixed_sl_atr": 3.20,
        "fixed_tp_atr": 2.40,
        "min_rr": 0.70,
        "technical_min_rr": 0.70,
        "max_spread_cost_r": 0.28,
        "trail_activation_atr": 1.30,
        "trail_lock_atr": 0.50,"""

assert old_run2 in run_text, "old_run2 not found"
run_text = run_text.replace(old_run2, new_run2, 1)

run_text = run_text.replace("mythos_policy.prm_threshold = 0.22", "mythos_policy.prm_threshold = 0.16", 1)
p_run.write_text(run_text, encoding="utf-8")
print("scripts/run_mythos_mt5_live.py updated.")

# 3. Update research_reasoning_world_class/src/mythos_mt5_policy.py
p_pol = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
pol_text = p_pol.read_text(encoding="utf-8")

old_buy = """                if diff >= min_margin and b_s >= (entry_thresh - 0.08) and prm >= max(0.20, prm_gate - 0.02):"""
new_buy = """                if diff >= min_margin and b_s >= (entry_thresh - 0.12) and prm >= max(0.15, prm_gate - 0.02):"""

assert old_buy in pol_text, "old_buy not found"
pol_text = pol_text.replace(old_buy, new_buy, 1)
p_pol.write_text(pol_text, encoding="utf-8")
print("research_reasoning_world_class/src/mythos_mt5_policy.py updated.")
