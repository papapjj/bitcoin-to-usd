import json

found = 0
path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_action.csv.metadata.jsonl'
with open(path, 'r') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        if data.get('action_name') != 'HOLD' or data.get('policy_action') != 0 or data.get('emitted_action') != 0:
            print(f"Line {idx}: seq={data.get('sequence')} role={data.get('decision_role')} act={data.get('policy_action')} emit={data.get('emitted_action')} name={data.get('action_name')} p_bull={data.get('p_bull')} p_bear={data.get('p_bear')} regime={data.get('regime_name')}")
            found += 1
            if found >= 30:
                break
print(f"Total found in sample: {found}")
