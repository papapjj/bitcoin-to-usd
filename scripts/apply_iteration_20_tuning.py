from pathlib import Path

# 1. Update scripts/run_mythos_mt5_live.py
p1 = Path("scripts/run_mythos_mt5_live.py")
text1 = p1.read_text(encoding="utf-8")

old1 = """    mythos_policy.sniper_sl_atr = 3.80
    mythos_policy.sniper_tp_rr = 0.737
    mythos_policy.scalp_sl_atr = 3.80
    mythos_policy.scalp_tp_rr = 0.737
    mythos_policy.trailing_atr = 1.80"""

new1 = """    mythos_policy.sniper_sl_atr = 2.80
    mythos_policy.sniper_tp_rr = 1.143
    mythos_policy.scalp_sl_atr = 2.80
    mythos_policy.scalp_tp_rr = 1.143
    mythos_policy.trailing_atr = 1.20"""

assert old1 in text1, "old1 not found"
text1 = text1.replace(old1, new1, 1)

old2 = """"fixed_sl_atr": 3.80,
        "fixed_tp_atr": 2.80,
        "min_rr": 0.70,
        "technical_min_rr": 0.70,
        "max_spread_cost_r": 0.28,
        "trail_activation_atr": 1.30,
        "trail_lock_atr": 0.20,"""

new2 = """"fixed_sl_atr": 2.80,
        "fixed_tp_atr": 3.20,
        "min_rr": 1.10,
        "technical_min_rr": 1.10,
        "max_spread_cost_r": 0.28,
        "trail_activation_atr": 2.00,
        "trail_lock_atr": 1.00,"""

assert old2 in text1, "old2 not found"
text1 = text1.replace(old2, new2, 1)
p1.write_text(text1, encoding="utf-8")
print("scripts/run_mythos_mt5_live.py updated successfully.")

# 2. Update research_reasoning_world_class/src/mythos_mt5_policy.py
p2 = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
text2 = p2.read_text(encoding="utf-8")

old_sell = """                if diff >= min_margin and b_s >= entry_thresh and prm >= prm_gate:
                    net_dir = 1
                elif -diff >= (min_margin + 0.20) and r_s >= entry_thresh and prm >= prm_gate:
                    net_dir = -1"""

new_sell = """                if diff >= min_margin and b_s >= entry_thresh and prm >= prm_gate:
                    net_dir = 1
                elif -diff >= (min_margin + 0.45) and r_s >= (entry_thresh + 0.03) and prm >= (prm_gate + 0.05):
                    net_dir = -1"""

assert old_sell in text2, "old_sell not found"
text2 = text2.replace(old_sell, new_sell, 1)
p2.write_text(text2, encoding="utf-8")
print("research_reasoning_world_class/src/mythos_mt5_policy.py updated successfully.")
