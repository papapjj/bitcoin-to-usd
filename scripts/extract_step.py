import json

with open(r'C:\Users\Admin\.gemini\antigravity\brain\1b3a582e-943b-4eaf-83e5-12bdf0d1bbe7\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get('step_index') == 30562:
            cmd = obj['tool_calls'][0]['args']['CommandLine']
            with open('scripts/step30562_code.py', 'w', encoding='utf-8') as out_f:
                out_f.write(cmd)
            print("Successfully extracted to scripts/step30562_code.py!")
            break
