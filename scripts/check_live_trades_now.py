import pandas as pd
import glob

files = glob.glob(r'.mt5_runners\visual_7day_champion_*\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv')
print(f"Found files: {files}")
if not files:
    exit(0)

# Sort by modification time to get the latest
files.sort(key=lambda x: pd.Timestamp(glob.os.path.getmtime(x)), reverse=True)
latest_file = files[0]
print(f"Reading latest file: {latest_file}")

df = pd.read_csv(latest_file, sep='\t')
print(f"Total rows: {len(df)}")
closed = df[df['deal_entry'] == 1]
print(f"Total closed trades: {len(closed)}")

if len(closed) > 0:
    wins = len(closed[closed['profit'] > 0])
    losses = len(closed[closed['profit'] < 0])
    tot_pnl = closed['profit'].sum()
    gp = closed[closed['profit'] > 0]['profit'].sum()
    gl = abs(closed[closed['profit'] < 0]['profit'].sum())
    pf = gp / gl if gl > 0 else 999.0
    wr = wins / len(closed) * 100.0
    print(f"Wins: {wins} | Losses: {losses} | WR: {wr:.1f}% | Net PnL: ${tot_pnl:.2f} | PF: {pf:.2f}")
    
    print("\n--- ALL CLOSED TRADES ---")
    for i, (_, r) in enumerate(closed.iterrows(), 1):
        print(f"{i:02d}. {r['logged_at']} | Action: {r['action_name']} | Vol: {r['volume']} | Price: {r['price']} | Profit: ${r['profit']:+.2f} | Close Reason: {r['close_reason']}")
else:
    print("No closed trades yet in this file.")
