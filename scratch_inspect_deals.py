from pathlib import Path

p = Path('mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv')
if not p.exists():
    p = Path('mt5_v2_runner/MQL5/Files/self_growing_execution.csv')
lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
deals = [l.split('\t') for l in lines if 'DEAL_ADD' in l and 'CLOSED' in l]
print(f'Total closed deals: {len(deals)}')
for i, d in enumerate(deals):
    p_val = d[37] if len(d) > 37 else 'N/A'
    comm = d[38] if len(d) > 38 else 'N/A'
    ex = d[34] if len(d) > 34 else 'N/A'
    trace = d[4] if len(d) > 4 else ''
    print(f'{i+1:2d}: profit={p_val:>7s} comm={comm:>5s} exit={ex:>10s} trace={trace}')
