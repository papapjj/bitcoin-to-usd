import json
import sys
from pathlib import Path

def analyze_oos(json_path: Path):
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return
    
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"\n=======================================================")
    print(f"FROZEN OOS REPORT: {json_path.name}")
    print(f"Candidate: {data.get('candidate')}")
    print(f"Candidate Bit-for-Bit Frozen: {data.get('candidate_unchanged')}")
    print(f"=======================================================\n")
    print(f"{'Win':<4} | {'Dates':<23} | {'Trades':<7} | {'WR':<7} | {'PF':<6} | {'Net PnL ($)':<12} | {'Max DD':<7} | {'Exp LCB (R)':<12}")
    print("-" * 95)
    
    for w in data.get("windows", []):
        idx = w.get("window")
        dates = f"{w.get('oos', {}).get('from')} - {w.get('oos', {}).get('to')}"
        met = w.get("metrics", {})
        trades = met.get("total_trades", 0)
        wr = met.get("win_rate", 0.0)
        pf = met.get("profit_factor", 0.0)
        pnl = met.get("net_profit", 0.0)
        dd = met.get("equity_drawdown_pct", 0.0)
        exec_sum = met.get("execution_summary", {})
        lcb = exec_sum.get("expectancy_lcb", 0.0)
        
        print(f"{idx:<4} | {dates:<23} | {trades:<7} | {wr:>5.1f}% | {pf:>5.2f} | {pnl:>+11.2f} | {dd:>5.2f}% | {lcb:>+10.4f}R")
    print("-" * 95)

if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/he_chuyen_gia/v193_runs/multi_round_phase3/frozen_oos_round_07.json")
    analyze_oos(p)
