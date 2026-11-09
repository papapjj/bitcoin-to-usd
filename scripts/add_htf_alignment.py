with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """        # Macro & Regime Directional Alignment Guard
        if getattr(self, "long_only", False) and net_dir == -1:
            net_dir = 0
        if getattr(self, "short_only", False) and net_dir == 1:
            net_dir = 0"""

replacement = """        # Macro & Regime Directional Alignment Guard
        if getattr(self, "long_only", False) and net_dir == -1:
            net_dir = 0
        if getattr(self, "short_only", False) and net_dir == 1:
            net_dir = 0

        # Higher-Timeframe Macro Trend Alignment Guard:
        # Strictly forbids shorting in an H1 Bull Trend and buying into an H1 Bear Dump
        if base_obs_137.size > 88:
            h1_diff = float(base_obs_137[88])
            if h1_diff > 0.05 and net_dir == -1:
                net_dir = 0
            elif h1_diff < -0.05 and net_dir == 1:
                net_dir = 0"""

text_norm = text.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')
replacement_norm = replacement.replace('\r\n', '\n')

if target_norm in text_norm:
    text_norm = text_norm.replace(target_norm, replacement_norm, 1)
    with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'w', encoding='utf-8') as f:
        f.write(text_norm)
    print("SUCCESS: Added HTF Macro Trend Alignment Guard")
else:
    print("ERROR: target_norm not found")
