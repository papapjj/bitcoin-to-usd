import os
import json
from self_growing_trader.mt5_runner import parse_mt5_report

reports = [
    (1, '2024.10.01', '2024.10.07', 'Early October Market Opening', 'he_chuyen_gia_mt5_round_0000_evaluation_19301.htm'),
    (2, '2024.10.07', '2024.10.14', 'Mid-October Accumulation', 'he_chuyen_gia_mt5_round_0000_evaluation_19303.htm'),
    (3, '2024.10.14', '2024.10.21', 'Pre-Election Positioning', 'he_chuyen_gia_mt5_round_0000_evaluation_19304.htm'),
    (4, '2024.10.21', '2024.10.28', 'Volatility Compression', 'he_chuyen_gia_mt5_round_0000_evaluation_19305.htm'),
    (5, '2024.10.28', '2024.11.04', 'Pre-Election Tension', 'he_chuyen_gia_mt5_round_0000_evaluation_19306.htm'),
    (6, '2024.11.04', '2024.11.11', 'US Presidential Election Regime', 'he_chuyen_gia_mt5_round_0000_evaluation_19307.htm'),
    (7, '2024.11.11', '2024.11.18', 'Post-Election Bull Surge', 'he_chuyen_gia_mt5_round_0000_evaluation_19308.htm'),
    (8, '2024.11.18', '2024.11.25', 'Euphoria Continuation', 'he_chuyen_gia_mt5_round_0000_evaluation_19309.htm'),
    (9, '2024.11.25', '2024.12.02', 'Thanksgiving Liquidity Sweep', 'he_chuyen_gia_mt5_round_0000_evaluation_19310.htm'),
    (10, '2024.12.02', '2024.12.09', 'Bitcoin Historic ATH Test', 'he_chuyen_gia_mt5_round_0000_evaluation_19311.htm'),
    (11, '2024.12.09', '2024.12.16', 'High-Altitude Volatility', 'he_chuyen_gia_mt5_round_0000_evaluation_19312.htm'),
    (12, '2024.12.16', '2024.12.23', 'Year-End Institutional Rebalance', 'he_chuyen_gia_mt5_round_0000_evaluation_19313.htm'),
]

master_records = []
total_trades = 0
total_pnl = 0.0

for wk, fd, td, lbl, rpt in reports:
    path = os.path.join('mt5_v2_runner', 'Reports', rpt)
    m = parse_mt5_report(path)
    total_trades += m.total_trades
    total_pnl += m.net_profit
    master_records.append({
        'week': wk,
        'from_date': fd,
        'to_date': td,
        'label': lbl,
        'report': rpt,
        'trades': m.total_trades,
        'net_pnl': m.net_profit,
        'win_rate': m.win_rate,
        'profit_factor': m.profit_factor,
        'equity_drawdown_pct': m.equity_drawdown_pct,
        'avg_win': m.avg_win,
        'avg_loss': m.avg_loss
    })

avg_wr = round(sum(r['win_rate'] * r['trades'] for r in master_records) / max(total_trades, 1), 2)

consolidated = {
    'title': '12-Week Rolling Walk-Forward Benchmark (2024.10.01 -> 2024.12.23)',
    'total_weeks': 12,
    'total_trades': total_trades,
    'total_net_pnl': round(total_pnl, 2),
    'avg_trades_per_week': round(total_trades / 12, 1),
    'avg_win_rate': avg_wr,
    'weeks': master_records
}

out_path = 'outputs/walk_forward_multi_asset_12w/walk_forward_12weeks_consolidated_master_manifest.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(consolidated, f, indent=2)

print('Consolidated 12 weeks saved successfully!')
print('Total trades:', total_trades)
print('Total Net PnL:', round(total_pnl, 2), 'USD')
print('Weighted Average Win Rate:', avg_wr, '%')
