path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
with open(path, 'r') as f:
    header = f.readline().strip().split('\t') if '\t' in f.readline() else f.readline().strip().split(',')
    print(f"Header len: {len(header)}")
    print(f"Header first 15: {header[:15]}")
    for i in range(5):
        line = f.readline()
        if not line: break
        parts = line.strip().split('\t') if '\t' in line else line.strip().split(',')
        print(f"Row {i}: len={len(parts)}, start={parts[:6]}")
