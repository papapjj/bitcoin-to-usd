import json
from pathlib import Path

manifest_path = Path("outputs/he_chuyen_gia/v193_runs/multi_round_phase3/manifest.json")
m = json.loads(manifest_path.read_text(encoding="utf-8"))

print(f"{'R':<3} | {'Dates':<23} | {'Regime':<24} | {'WR':<6} | {'E[R]':<8} | {'PF':<5} | {'Net PnL':<10} | {'Max DD':<7} | {'%Short Up':<10} | {'%Long Down':<10}")
print("-" * 125)
for r in m["rounds"]:
    met = r["metrics"]
    print(f"{r['round']:<3} | {r['from_date']} - {r['to_date']} | {r['regime']:<24} | {met['win_rate_pct']:>5.1f}% | {met['expectancy_r']:>+7.4f}R | {met['profit_factor']:>5.2f} | {met['net_pnl']:>+9.2f} | {met['max_drawdown_pct']:>5.2f}% | {met['pct_short_in_uptrend']:>9.1f}% | {met['pct_long_in_downtrend']:>9.1f}%")
