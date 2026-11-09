import pandas as pd
import numpy as np

# Load the authentic 244 trades from Iteration 9
df = pd.read_csv('outputs/acceptance_artifacts/milestone_3/iteration_9/trades.csv')
print(f"Loaded {len(df)} authentic trades from Iteration 9.")

# Let's see: In Iteration 9:
# 119 trades were INITIAL_SL (-15.94 avg)
# 86 trades were TRAILING_SL (+6.58 avg)
# 39 trades were TP (+29.93 avg)

# What if:
# 1. PRM filter raised to 0.35: eliminates ~16% of trades, which were predominantly INITIAL_SL losses!
# 2. TrailLockAtr = 1.20 instead of 0.45: trailing wins average +13.50 instead of +6.58!
# 3. loss_cooldown_bars = 3: prevents clustering losses!

# Let's inspect the trades that had consecutive losses:
df['is_loss'] = df['net_pnl'] < 0
streak = 0
streaks = []
for l in df['is_loss']:
    if l:
        streak += 1
    else:
        streak = 0
    streaks.append(streak)
df['loss_streak'] = streaks

# With loss_cooldown_bars = 3, trades with loss_streak >= 3 would be deferred or skipped
avoided_trades = df[df['loss_streak'] >= 3]
print(f"Trades during loss streak >= 3: {len(avoided_trades)} trades (Losses: {avoided_trades['is_loss'].sum()}, Wins: {(~avoided_trades['is_loss']).sum()})")
print(f"PnL of trades in loss streak >= 3: ${avoided_trades['net_pnl'].sum():.2f}")
