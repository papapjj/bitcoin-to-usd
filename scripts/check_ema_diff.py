import pandas as pd
import numpy as np

feat_path = 'mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv'
exec_path = 'outputs/acceptance_artifacts/milestone_3/iteration_26/self_growing_execution.csv'

exec_df = pd.read_csv(exec_path, sep='\t')
feat_df = pd.read_csv(feat_path, sep='\t')

# Parse feature 11 (ema_diff) and feature 27 (p_vs_e50) for each feature row
feat_df['features'] = feat_df['inputs'].apply(lambda x: [float(v) for v in x.split('|')])
feat_df['ema_diff'] = feat_df['features'].apply(lambda f: f[11] if len(f) > 11 else 0.0)
feat_df['p_vs_e50'] = feat_df['features'].apply(lambda f: f[27] if len(f) > 27 else 0.0)
feat_df['p_vs_e20'] = feat_df['features'].apply(lambda f: f[26] if len(f) > 26 else 0.0)

feat_df['dt'] = pd.to_datetime(feat_df['logged_at'])
closed = exec_df[exec_df['closed'] == 1].copy()
closed['dt'] = pd.to_datetime(closed['logged_at'])

merged = pd.merge_asof(closed.sort_values('dt'), feat_df[['dt', 'ema_diff', 'p_vs_e50', 'p_vs_e20']].sort_values('dt'), on='dt', direction='backward')

print("Merged closed deals:", len(merged))
wins = merged[merged['profit'] > 0]
losses = merged[merged['profit'] < 0]

print(f"\nEMA Diff (EMA20 - EMA60):")
print(f"Wins: mean={wins['ema_diff'].mean():.4f}, median={wins['ema_diff'].median():.4f}, pct > 0: {(wins['ema_diff'] > 0).mean()*100:.1f}%")
print(f"Losses: mean={losses['ema_diff'].mean():.4f}, median={losses['ema_diff'].median():.4f}, pct > 0: {(losses['ema_diff'] > 0).mean()*100:.1f}%")

print(f"\nPrice vs EMA50:")
print(f"Wins: mean={wins['p_vs_e50'].mean():.4f}, pct > 0: {(wins['p_vs_e50'] > 0).mean()*100:.1f}%")
print(f"Losses: mean={losses['p_vs_e50'].mean():.4f}, pct > 0: {(losses['p_vs_e50'] > 0).mean()*100:.1f}%")

for th in [-0.05, 0.0, 0.05, 0.10]:
    filtered = merged[merged['ema_diff'] >= th]
    f_wins = filtered[filtered['profit'] > 0]
    f_losses = filtered[filtered['profit'] < 0]
    f_gp = f_wins['profit'].sum()
    f_gl = abs(f_losses['profit'].sum())
    f_wr = len(f_wins) / len(filtered) * 100 if len(filtered) > 0 else 0
    f_pf = f_gp / f_gl if f_gl > 0 else 0
    print(f"Threshold ema_diff >= {th:+.2f} -> Deals: {len(filtered)} | WR: {f_wr:.1f}% | PF: {f_pf:.2f} | Net: ${f_gp - f_gl:.2f}")
