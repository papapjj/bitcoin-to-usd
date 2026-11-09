import json

path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_action.csv.metadata.jsonl'
with open(path, 'r') as f:
    for i in range(20):
        line = f.readline()
        if not line: break
        data = json.loads(line)
        seq = data.get('sequence')
        role = data.get('decision_role')
        eff = data.get('effective_allowed_actions')
        act = data.get('policy_action')
        emit = data.get('emitted_action')
        reg_veto = data.get('regime_veto_applied')
        veto_reason = data.get('regime_veto_reason')
        print(f"Row {i:02d} Seq {seq} Role={role:<18} Eff={eff} Act={act} Emit={emit} Veto={reg_veto} Reason={veto_reason}")
