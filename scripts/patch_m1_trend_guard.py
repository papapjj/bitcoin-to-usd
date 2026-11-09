with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """        # Higher-Timeframe Macro Trend Alignment Guard:
        # Strictly forbids shorting in an H1 Bull Trend and buying into an H1 Bear Dump
        if base_obs_137.size > 88:
            h1_diff = float(base_obs_137[88])
            if h1_diff > 0.05 and net_dir == -1:
                net_dir = 0
            elif h1_diff < -0.05 and net_dir == 1:
                net_dir = 0"""

replacement = """        # Authentic M1 Multi-Bar Trend Alignment Guard (from 60-bar rolling memory):
        # In Uptrend (current_price > ma50 and ma20 > ma50), strictly forbid counter-trend shorts!
        # In Downtrend (current_price < ma50 and ma20 < ma50), strictly forbid catching falling knives!
        closes = self.m1_buffer[:, 3]
        valid_closes = closes[closes > 0]
        if len(valid_closes) >= 30:
            ma20 = float(np.mean(valid_closes[-20:]))
            ma50 = float(np.mean(valid_closes[-min(len(valid_closes), 50):]))
            if ma20 > ma50 and self.current_price > ma50 and net_dir == -1:
                net_dir = 0  # Forbid shorting in an established Uptrend!
            elif ma20 < ma50 and self.current_price < ma50 and net_dir == 1:
                net_dir = 0  # Forbid catching a falling knife in an established Downtrend!"""

text_norm = text.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')
replacement_norm = replacement.replace('\r\n', '\n')

if target_norm in text_norm:
    text_norm = text_norm.replace(target_norm, replacement_norm, 1)
    with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'w', encoding='utf-8') as f:
        f.write(text_norm)
    print("SUCCESS: Replaced with authentic M1 Multi-Bar Trend Alignment Guard.")
else:
    print("ERROR: target_norm not found")
