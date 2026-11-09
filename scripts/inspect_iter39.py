import pandas as pd
from pathlib import Path

cand = [
    Path('.mt5_runners/opt_Iter_39_2Day_32_20261029_151354_129504/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'),
    Path('.mt5_runners/opt_Iter_39_2Day_32_20261029_151354_129504/MQL5/Files/self_growing_execution.csv'),
]
target = None
for c in cand:
    if c.exists():
        target = c
        break

print("Target:", target)
df = pd.read_csv(target, sep='\t')
deals = df[(df['transaction_type'] == 'DEAL_ADD') & (df['status'] == 'CLOSED')].copy()
print(f"Total deals: {len(deals)}")
deals['profit'] = pd.to_numeric(deals['profit'], errors='coerce')
wins = deals[deals['profit'] > 0]
losses = deals[deals['profit'] < 0]
win_sum = float(wins['profit'].sum())
loss_sum = float(abs(losses['profit'].sum()))
print(f"Wins: {len(wins)}, Losses: {len(losses)}")
print(f"Win Rate: {len(wins)/len(deals)*100:.2f}%")
print(f"Gross Profit: {win_sum:.2f}, Gross Loss: {loss_sum:.2f}, Net: {win_sum - loss_sum:.2f}")
if loss_sum > 0:
    print(f"Profit Factor: {win_sum/loss_sum:.2f}")

print("\nClose reasons:")
if 'close_reason' in deals.columns:
    print(deals['close_reason'].value_counts())

print("\nProfit stats for wins:")
print(wins['profit'].describe())

print("\nLoss stats for losses:")
print(losses['profit'].describe())


print("\n" + "="*50)
print("COMPARISON ITER 37 vs ITER 39:")
for it, p in [('Iter 37', '.mt5_runners/opt_Iter_37_2Day_32_20261029_132049_656515/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'),
             ('Iter 39', '.mt5_runners/opt_Iter_39_2Day_32_20261029_151354_129504/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv')]:
    target_p = Path(p)
    if target_p.exists():
        df_cmp = pd.read_csv(target_p, sep='\t')
        d_cmp = df_cmp[(df_cmp['transaction_type'] == 'DEAL_ADD') & (df_cmp['status'] == 'CLOSED')].copy()
        d_cmp['profit'] = pd.to_numeric(d_cmp['profit'], errors='coerce')
        w = d_cmp[d_cmp['profit'] > 0]
        l = d_cmp[d_cmp['profit'] < 0]
        be = d_cmp[(d_cmp['profit'] >= 0.5) & (d_cmp['profit'] <= 2.5)]
        big_w = d_cmp[d_cmp['profit'] > 5.0]
        net = d_cmp['profit'].sum()
        pf = w['profit'].sum() / abs(l['profit'].sum()) if abs(l['profit'].sum()) > 0 else 0
        print(f"{it}: Total={len(d_cmp)}, Wins={len(w)} (BE_small={len(be)}, Big_wins={len(big_w)}), Losses={len(l)}, WR={len(w)/len(d_cmp)*100:.2f}%, PF={pf:.2f}, Net={net:.2f}")
