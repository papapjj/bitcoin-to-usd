from pathlib import Path

p = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
content = p.read_text(encoding="utf-8")

old_code = """            if is_cp:
                net_dir = arb_act
                conf = max(conf, 0.88)
            # Valid BUY: Bull conviction beats Bear by >= margin, b_s >= min_conf, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif diff >= min_margin and b_s >= min_conf and prm >= prm_gate and (arb_act == 1 or diff >= 0.08):
                net_dir = 1
                conf = max(conf, b_s)
            # Valid SELL: Bear conviction beats Bull by >= margin, r_s >= min_conf, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif -diff >= min_margin and r_s >= min_conf and prm >= prm_gate and (arb_act == -1 or -diff >= 0.08):
                net_dir = -1
                conf = max(conf, r_s)
            else:
                net_dir = 0"""

new_code = """            bear_thresh = getattr(self, "bear_entry_threshold", min_conf)
            sell_margin = getattr(self, "bear_min_margin", min_margin * 0.7)

            if is_cp:
                net_dir = arb_act
                conf = max(conf, 0.88)
            # Valid BUY: Bull conviction beats Bear by >= margin, b_s >= min_conf, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif diff >= min_margin and b_s >= min_conf and prm >= prm_gate and (arb_act == 1 or diff >= 0.08):
                net_dir = 1
                conf = max(conf, b_s)
            # Valid SELL: Bear conviction beats Bull by >= sell_margin, r_s >= bear_thresh, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif -diff >= sell_margin and r_s >= bear_thresh and prm >= prm_gate and (arb_act == -1 or -diff >= 0.08):
                net_dir = -1
                conf = max(conf, r_s)
            else:
                net_dir = 0"""

old_norm = old_code.replace("\r\n", "\n")
c_norm = content.replace("\r\n", "\n")
assert old_norm in c_norm, "Old bilateral decision code not found!"
c_norm = c_norm.replace(old_norm, new_code.replace("\r\n", "\n"), 1)

p.write_text(c_norm, encoding="utf-8")
print("Successfully patched bilateral sell decision in mythos_mt5_policy.py!")
