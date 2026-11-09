import json

with open(r'C:\Users\Admin\.gemini\antigravity\brain\1b3a582e-943b-4eaf-83e5-12bdf0d1bbe7\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        idx = obj.get('step_index', 0)
        if 30555 <= idx <= 30568:
            print(f"Step {idx}: {obj.get('source')} - {obj.get('type')}")
            for c in obj.get('tool_calls', []):
                print('  Call:', c.get('name'), c.get('args'))
