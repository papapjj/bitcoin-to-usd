import json
import pandas as pd
import numpy as np

for it in [7, 8, 9]:
    m_path = f"outputs/acceptance_artifacts/milestone_3/iteration_{it}/metrics.json"
    t_path = f"outputs/acceptance_artifacts/milestone_3/iteration_{it}/trades.csv"
    with open(m_path) as f:
        m = json.load(f)
    df = pd.read_csv(t_path)
    print(f"=== Iteration {it} ===")
    print(f"Total trades: {m['total_closed_trades']}, Trades/day: {m['trades_per_day']:.2f}")
    print(f"Win Rate: {m['win_rate_pct']}%, Profit Factor: {m['profit_factor']:.2f}")
    print(f"Gross Profit: ${m['gross_profit_usd']:.2f}, Gross Loss: ${m['gross_loss_usd']:.2f}, Net PnL: ${m['net_pnl_usd']:.2f}")
    print(f"Max DD: {m['max_drawdown_pct']}%")
    print(f"Buy trades: {m['buy_trades']} (WR: {m['buy_win_rate_pct']}%), Sell trades: {m['sell_trades']} (WR: {m['sell_win_rate_pct']}%)")
    if "exit_kind" in df.columns:
        print("Exit kinds:")
        print(df["exit_kind"].value_counts().to_dict())
    
    # Analyze trade durations and PnL distributions
    df["profit_num"] = pd.to_numeric(df["profit"], errors="coerce")
    wins = df[df["profit_num"] > 0]
    losses = df[df["profit_num"] <= 0]
    avg_win = wins["profit_num"].mean() if len(wins) else 0
    avg_loss = abs(losses["profit_num"].mean()) if len(losses) else 0
    payoff = avg_win / avg_loss if avg_loss else 0
    print(f"Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}, Payoff Ratio: {payoff:.2f}")
    
    # Loss streak analysis
    df["is_loss"] = df["profit_num"] <= 0
    streaks = []
    current_streak = 0
    for is_l in df["is_loss"]:
        if is_l:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
    print(f"Loss streaks: max={max(streaks) if streaks else 0}, mean={np.mean(streaks) if streaks else 0:.2f}, total loss episodes={len(streaks)}")
    streak_counts = pd.Series(streaks).value_counts().sort_index().to_dict() if streaks else {}
    print(f"Loss streak distribution (length: count): {streak_counts}")
    print()
