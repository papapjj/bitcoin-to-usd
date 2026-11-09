import json
import pandas as pd

# 1. Read execution ledger
csv_file = r'.mt5_runners\opt_Iter_46_2Day_14240_20261029_202154_339863\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv'
df = pd.read_csv(csv_file, sep='\t')
closed = df[df['position_final_close'] == 1].copy()

# 2. Read reasoning trace
buy_traces = []
with open('outputs/mt5_optimization_loop/reasoning_trace.jsonl', 'r') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get('action') == 'SCALP_BUY':
                buy_traces.append(rec)
        except Exception:
            pass

print(f"Total Closed Deals: {len(closed)}")
print(f"Total SCALP_BUY actions logged in trace: {len(buy_traces)}")

# Match trades with trace by order
print("\n=== ALL CLOSED DEALS WITH NEURAL TELEMETRY ===")
for idx, (_, row) in enumerate(closed.iterrows()):
    p = row['profit']
    status = "WIN " if p > 0 else "LOSS"
    trace_info = ""
    if idx < len(buy_traces):
        bt = buy_traces[idx]
        sc = bt.get('scenarios', {})
        sc_str = f"exp={sc.get('expansion',0):.2f}, rev={sc.get('reversal',0):.2f}, shk={sc.get('shakeout',0):.2f}, chp={sc.get('chop',0):.2f}"
        trace_info = f" | Bull={bt['bull_score']:.3f}, Bear={bt['bear_score']:.3f}, Margin={bt['margin']:.3f}, PRM={bt['prm_score']:.3f} | {sc_str}"
    print(f"Deal #{idx+1:02d} [{status}] Ticket={row['deal_ticket']} Profit={p:6.2f} Entry={row['entry_price']:.2f} Reason={row['close_reason']}/{row['exit_kind']} Time={row['logged_at']}{trace_info}")
