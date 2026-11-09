"""
tests/test_mt5_empirical_audit.py

Automated Acceptance Criteria Test Suite for Milestone M4:
Multi-Cycle MT5 Execution & Ground-Truth Empirical Ledger Verification.

Validates the 5 Strict Acceptance Criteria across all 3 canonical market regimes:
1. Window 1: Bull Run ($96k -> $98k)
2. Window 2: Bear Selloff ($98k -> $93k)
3. Window 3: Chop / Consolidation & Bottom Demand Zone Exhaustion

Acceptance Thresholds:
- Win Rate >= 80.0% (across all cycles and per-cycle)
- Profit Factor > 3.0 (for all 3 windows)
- Max Drawdown <= 10.0% (eliminating historical 33% drawdown)
- Bottom Shorts in Demand Zone == 0 (100% intercepted)
- Total Closed Trades >= 15 per window and overall
- Agent 4 Governor Safety Invariant: 0 violations (ACTION_HOLD = 0, POLICY_CLOSE_ALL == 0.0)
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "outputs/empirical_ledger_audit_summary.json"


@pytest.fixture(scope="module")
def audit_summary():
    """Loads pre-generated empirical audit summary JSON or executes audit if missing."""
    if not SUMMARY_PATH.exists():
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from audit_mt5_empirical_ledger import run_full_empirical_audit
        return run_full_empirical_audit(output_summary_path=SUMMARY_PATH)
    
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_audit_summary_exists_and_valid(audit_summary):
    assert audit_summary is not None
    if isinstance(audit_summary, dict):
        assert "all_criteria_passed" in audit_summary
        assert "window_breakdowns" in audit_summary
    else:
        assert hasattr(audit_summary, "all_criteria_passed")
        assert hasattr(audit_summary, "window_breakdowns")


def test_window_1_bull_run_metrics(audit_summary):
    breakdowns = audit_summary["window_breakdowns"] if isinstance(audit_summary, dict) else [
        w.__dict__ if hasattr(w, "__dict__") else w for w in audit_summary.window_breakdowns
    ]
    w1 = next(w for w in breakdowns if "01" in w["window_name"])
    
    assert w1["total_trades"] >= 15, f"Window 1 trades {w1['total_trades']} < 15"
    assert w1["win_rate_pct"] >= 80.0, f"Window 1 WR {w1['win_rate_pct']}% < 80.0%"
    assert w1["profit_factor"] > 3.0, f"Window 1 PF {w1['profit_factor']} <= 3.0"
    assert w1["max_drawdown_pct"] <= 10.0, f"Window 1 MDD {w1['max_drawdown_pct']}% > 10.0%"
    assert w1["bottom_shorts_in_demand_executed"] == 0, f"Window 1 bottom shorts {w1['bottom_shorts_in_demand_executed']} != 0"
    assert w1["agent4_governor_violations"] == 0, f"Window 1 agent4 violations {w1['agent4_governor_violations']} != 0"
    assert w1["passed_all_criteria"] is True


def test_window_2_bear_selloff_metrics(audit_summary):
    breakdowns = audit_summary["window_breakdowns"] if isinstance(audit_summary, dict) else [
        w.__dict__ if hasattr(w, "__dict__") else w for w in audit_summary.window_breakdowns
    ]
    w2 = next(w for w in breakdowns if "02" in w["window_name"])
    
    assert w2["total_trades"] >= 15, f"Window 2 trades {w2['total_trades']} < 15"
    assert w2["win_rate_pct"] >= 80.0, f"Window 2 WR {w2['win_rate_pct']}% < 80.0%"
    assert w2["profit_factor"] > 3.0, f"Window 2 PF {w2['profit_factor']} <= 3.0"
    assert w2["max_drawdown_pct"] <= 10.0, f"Window 2 MDD {w2['max_drawdown_pct']}% > 10.0%"
    assert w2["bottom_shorts_in_demand_executed"] == 0, f"Window 2 bottom shorts {w2['bottom_shorts_in_demand_executed']} != 0"
    assert w2["agent4_governor_violations"] == 0, f"Window 2 agent4 violations {w2['agent4_governor_violations']} != 0"
    assert w2["passed_all_criteria"] is True


def test_window_3_chop_bottom_metrics(audit_summary):
    breakdowns = audit_summary["window_breakdowns"] if isinstance(audit_summary, dict) else [
        w.__dict__ if hasattr(w, "__dict__") else w for w in audit_summary.window_breakdowns
    ]
    w3 = next(w for w in breakdowns if "03" in w["window_name"])
    
    assert w3["total_trades"] >= 15, f"Window 3 trades {w3['total_trades']} < 15"
    assert w3["win_rate_pct"] >= 80.0, f"Window 3 WR {w3['win_rate_pct']}% < 80.0%"
    assert w3["profit_factor"] > 3.0, f"Window 3 PF {w3['profit_factor']} <= 3.0"
    assert w3["max_drawdown_pct"] <= 10.0, f"Window 3 MDD {w3['max_drawdown_pct']}% > 10.0%"
    assert w3["bottom_shorts_in_demand_executed"] == 0, f"Window 3 bottom shorts {w3['bottom_shorts_in_demand_executed']} != 0"
    assert w3["agent4_governor_violations"] == 0, f"Window 3 agent4 violations {w3['agent4_governor_violations']} != 0"
    assert w3["passed_all_criteria"] is True


def test_aggregate_all_windows_metrics(audit_summary):
    summary_dict = audit_summary if isinstance(audit_summary, dict) else audit_summary.__dict__
    
    assert summary_dict["total_trades"] >= 15, f"Total trades {summary_dict['total_trades']} < 15"
    assert summary_dict["overall_win_rate_pct"] >= 80.0, f"Overall WR {summary_dict['overall_win_rate_pct']}% < 80.0%"
    assert summary_dict["overall_profit_factor"] > 3.0, f"Overall PF {summary_dict['overall_profit_factor']} <= 3.0"
    assert summary_dict["worst_max_drawdown_pct"] <= 10.0, f"Worst MDD {summary_dict['worst_max_drawdown_pct']}% > 10.0%"
    assert summary_dict["total_bottom_shorts_in_demand"] == 0, f"Total bottom shorts {summary_dict['total_bottom_shorts_in_demand']} != 0"
    assert summary_dict["total_agent4_violations"] == 0, f"Total agent4 violations {summary_dict['total_agent4_violations']} != 0"
    assert summary_dict["all_criteria_passed"] is True, "Not all acceptance criteria passed"


def test_agent4_governor_strict_hold_invariant(audit_summary):
    summary_dict = audit_summary if isinstance(audit_summary, dict) else audit_summary.__dict__
    assert summary_dict["total_agent4_violations"] == 0
    checks = summary_dict["criteria_checks"]
    assert checks["ac_6_agent4_governor_hold_invariant"] is True
