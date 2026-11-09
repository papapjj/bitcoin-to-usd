from pathlib import Path

p = Path('scripts/train_7day_bilateral_dual_cot_selfplay.py')
text = p.read_text(encoding='utf-8')

old_s = '''    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    
    metrics = {
        "total_bars_evaluated": N,
        "total_trades": total_trades,
        "buys": buys,
        "sells": sells,
        "buy_ratio_pct": round(buy_ratio, 2),
        "sell_ratio_pct": round(100.0 - buy_ratio, 2),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "bottom_shorts": bottom_shorts,
        "counter_punch_trades": len(counter_punch_trades),
        "counter_punch_win_rate_pct": round(cp_win_rate, 2),
        "median_latency_ms": round(median_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "mad_memory_count": len(mad_memory),
        "training_history": training_history,
    }
    
    logger.info("=" * 60)
    logger.info("7-DAY BILATERAL DUAL-CoT BENCHMARK REPORT:")
    logger.info("Total Trades: %d (BUY: %d [%.1f%%], SELL: %d [%.1f%%])", total_trades, buys, buy_ratio, sells, 100.0 - buy_ratio)
    logger.info("Win Rate: %.2f%% (Wins: %d, Losses: %d)", win_rate, wins, losses)
    logger.info("Profit Factor: %.2f", profit_factor)
    logger.info("Bottom Shorts: %d", bottom_shorts)'''

new_s = '''    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    
    # Calculate simulated compounding equity and maximum drawdown
    equity = 10000.0
    peak_equity = equity
    max_dd = 0.0
    for t in trades:
        trade_pnl = t["pnl_r"] * 25.0
        equity += trade_pnl
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / max(1.0, peak_equity) * 100.0
        if dd > max_dd:
            max_dd = dd
            
    metrics = {
        "total_bars_evaluated": N,
        "total_trades": total_trades,
        "buys": buys,
        "sells": sells,
        "buy_ratio_pct": round(buy_ratio, 2),
        "sell_ratio_pct": round(100.0 - buy_ratio, 2),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "bottom_shorts": bottom_shorts,
        "counter_punch_trades": len(counter_punch_trades),
        "counter_punch_win_rate_pct": round(cp_win_rate, 2),
        "median_latency_ms": round(median_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "mad_memory_count": len(mad_memory),
        "training_history": training_history,
    }
    
    logger.info("=" * 60)
    logger.info("7-DAY BILATERAL DUAL-CoT BENCHMARK REPORT:")
    logger.info("Total Trades: %d (BUY: %d [%.1f%%], SELL: %d [%.1f%%])", total_trades, buys, buy_ratio, sells, 100.0 - buy_ratio)
    logger.info("Win Rate: %.2f%% (Wins: %d, Losses: %d)", win_rate, wins, losses)
    logger.info("Profit Factor: %.2f", profit_factor)
    logger.info("Max Drawdown: %.2f%%", max_dd)
    logger.info("Bottom Shorts: %d", bottom_shorts)'''

assert old_s in text, "old_s not found"
p.write_text(text.replace(old_s, new_s), encoding='utf-8')
print("Metrics updated successfully!")
