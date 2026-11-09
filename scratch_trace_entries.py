from pathlib import Path

p = Path('mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv')
if not p.exists():
    p = Path('mt5_v2_runner/MQL5/Files/self_growing_execution.csv')
lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
deals = [l.split('\t') for l in lines if 'DEAL_ADD' in l and 'CLOSED' in l]
for i, d in enumerate(deals):
    pos_id = d[18] if len(d) > 18 else ''
    open_lines = [l.split('\t') for l in lines if 'ORDER_ADD' in l and len(l.split('\t')) > 18 and l.split('\t')[18] == pos_id]
    entry_dir = open_lines[0][7] if open_lines and len(open_lines[0]) > 7 else 'unknown'
    price_open = open_lines[0][20] if open_lines and len(open_lines[0]) > 20 else 'unknown'
    price_close = d[20] if len(d) > 20 else 'unknown'
    print(f'Deal {i+1:2d}: dir={entry_dir:>4s} open={price_open:>8s} close={price_close:>8s} profit={float(d[37]):+6.2f} exit={d[34]}')
