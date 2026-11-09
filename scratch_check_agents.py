import json

with open("mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_action.csv.metadata.jsonl", "r") as f:
    lines = [json.loads(line) for line in f.readlines()[-5:]]

for item in lines:
    print(f"Role: {item.get('decision_role')} | Wire Action: {item.get('wire_action')} | Policy Action: {item.get('policy_action')} | Allowed: {item.get('effective_allowed_actions')} | Veto: {item.get('veto_reason')}")
