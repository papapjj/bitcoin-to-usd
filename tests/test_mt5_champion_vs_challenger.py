"""
tests/test_mt5_champion_vs_challenger.py

Unit test suite verifying the MT5 Head-to-Head Tournament and 5-Gate Promotion Contract:
- Validates metric extraction from MT5 execution results.
- Verifies Gate 1 (Volume >= 100/day), Gate 2 (WR >= 85%), Gate 3 (DD <= 6%), Gate 4 (Outperformance), Gate 5 (Stability).
- Verifies rejection when any hurdle fails.
- Verifies atomic promotion when all hurdles pass.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from scripts.run_mt5_champion_vs_challenger import (
    extract_metrics_from_mt5_result,
    evaluate_five_gates,
)


def test_extract_metrics_from_mt5_result():
    mock_result = {
        "best": {
            "fitness": 2.45,
            "metrics": {
                "trade_count": 420,
                "win_rate": 86.5,
                "profit_factor": 2.15,
                "net_profit": 1850.50,
                "max_drawdown_pct": 4.2,
            }
        }
    }
    m = extract_metrics_from_mt5_result(mock_result, "2025.01.14", "2025.01.18")
    assert m["total_trades"] == 420
    assert m["win_rate"] == 86.5
    assert m["profit_factor"] == 2.15
    assert m["net_pnl"] == 1850.50
    assert m["max_drawdown"] == 4.2
    assert m["trades_per_day"] >= 100.0


def test_five_gates_all_pass():
    champ_m = {
        "total_trades": 350,
        "trades_per_day": 87.5,
        "win_rate": 84.0,
        "profit_factor": 1.80,
        "net_pnl": 1200.0,
        "max_drawdown": 5.0,
    }
    chall_m = {
        "total_trades": 450,
        "trades_per_day": 112.5,
        "win_rate": 87.2,
        "profit_factor": 2.25,
        "net_pnl": 1650.0,  # > 1200 * 1.05
        "max_drawdown": 3.8, # <= 6.0%
    }
    gates = evaluate_five_gates(chall_m, champ_m)
    assert gates["promoted"] is True
    assert gates["gate_1_frequency"]["passed"] is True
    assert gates["gate_2_win_rate"]["passed"] is True
    assert gates["gate_3_drawdown"]["passed"] is True
    assert gates["gate_4_outperformance"]["passed"] is True
    assert gates["gate_5_stability"]["passed"] is True


def test_five_gates_fails_on_win_rate():
    champ_m = {
        "total_trades": 400,
        "trades_per_day": 100.0,
        "win_rate": 86.0,
        "profit_factor": 2.0,
        "net_pnl": 1400.0,
        "max_drawdown": 4.5,
    }
    chall_m = {
        "total_trades": 500,
        "trades_per_day": 125.0,
        "win_rate": 81.5,   # FAILS: < 85.0%
        "profit_factor": 1.7,
        "net_pnl": 1500.0,
        "max_drawdown": 4.0,
    }
    gates = evaluate_five_gates(chall_m, champ_m)
    assert gates["promoted"] is False
    assert gates["gate_2_win_rate"]["passed"] is False


def test_five_gates_fails_on_drawdown():
    champ_m = {
        "total_trades": 400,
        "trades_per_day": 100.0,
        "win_rate": 86.0,
        "profit_factor": 2.0,
        "net_pnl": 1400.0,
        "max_drawdown": 4.5,
    }
    chall_m = {
        "total_trades": 500,
        "trades_per_day": 125.0,
        "win_rate": 88.0,
        "profit_factor": 2.3,
        "net_pnl": 2000.0,
        "max_drawdown": 7.8, # FAILS: > 6.0%
    }
    gates = evaluate_five_gates(chall_m, champ_m)
    assert gates["promoted"] is False
    assert gates["gate_3_drawdown"]["passed"] is False


def test_five_gates_fails_on_pnl_underperformance():
    champ_m = {
        "total_trades": 400,
        "trades_per_day": 100.0,
        "win_rate": 86.0,
        "profit_factor": 2.0,
        "net_pnl": 1400.0,
        "max_drawdown": 4.5,
    }
    chall_m = {
        "total_trades": 450,
        "trades_per_day": 112.5,
        "win_rate": 86.5,
        "profit_factor": 1.9,
        "net_pnl": 1300.0,  # FAILS: < 1400 * 1.05
        "max_drawdown": 4.2,
    }
    gates = evaluate_five_gates(chall_m, champ_m)
    assert gates["promoted"] is False
    assert gates["gate_4_outperformance"]["passed"] is False
