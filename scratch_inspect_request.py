path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_request.csv'
with open(path, 'r') as f:
    for i in range(10):
        line = f.readline()
        if not line: break
        parts = line.strip().split('\t') if '\t' in line else line.strip().split(',')
        print(f"Line {i}: len={len(parts)}, start={parts[:5]}")
