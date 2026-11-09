with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add to deepcopy
t1 = '"structural_filter_enabled", "swing_dist_max", "min_margin_threshold", "max_margin_threshold", "min_confidence"'
r1 = '"structural_filter_enabled", "swing_dist_max", "min_margin_threshold", "max_margin_threshold", "min_confidence", "min_atr_threshold"'

# 2. Add to volatility filter
t2 = 'if conf < min_conf or atr_pct > max_atr_pct:'
r2 = '''min_atr = float(getattr(self, "min_atr_threshold", 0.0))
        if conf < min_conf or atr_pct > max_atr_pct or self.current_atr < min_atr:'''

if t1 in text and t2 in text:
    text = text.replace(t1, r1, 1)
    text = text.replace(t2, r2, 1)
    with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("SUCCESS: Updated mythos_mt5_policy.py with min_atr_threshold support.")
else:
    print(f"ERROR: Targets not found. t1 in text: {t1 in text}, t2 in text: {t2 in text}")
