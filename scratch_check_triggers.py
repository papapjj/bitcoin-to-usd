with open(r'C:\Users\Admin\.gemini\antigravity\brain\1b3a582e-943b-4eaf-83e5-12bdf0d1bbe7\.system_generated\tasks\task-12984.log', 'r', encoding='utf-8') as f:
    lines = f.readlines()

entered_ticks = []
for i, l in enumerate(lines):
    if '[BridgeFive] served=' in l and 'packet=[a0=' in l:
        # Check if a0 != 0 or a1 != 0
        if 'a0=1' in l or 'a0=2' in l or 'a1=1' in l or 'a1=2' in l:
            # find previous reasoning line
            reasoning = "N/A"
            for k in range(max(0, i-5), i):
                if '[Reasoning]:' in lines[k]:
                    reasoning = lines[k].strip()
            entered_ticks.append((l.strip(), reasoning))

print(f"Total entries found in bridge: {len(entered_ticks)}")
for b, r in entered_ticks:
    print(f"{b} | {r}")
