with open(r'C:\Users\Admin\.gemini\antigravity\brain\1b3a582e-943b-4eaf-83e5-12bdf0d1bbe7\.system_generated\tasks\task-12984.log', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if any(sig in l for sig in ['COUNTER_TRAP', 'SNIPER', 'SCALP']):
        if any(w in l for w in ['BUY [', 'SELL [']):
            # Print tick number if nearby
            tick_str = ""
            for k in range(max(0, i-2), i):
                if '[MT5 LIVE] Tick' in lines[k]:
                    tick_str = lines[k].strip()
            print(f"{tick_str} -> {l.strip()}")
