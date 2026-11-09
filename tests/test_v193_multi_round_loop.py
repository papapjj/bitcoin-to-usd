from __future__ import annotations

import json
from pathlib import Path
import sys
import yaml
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_v193_multi_round_loop import (
    evaluate_promotion,
    extract_round_metrics,
    Q1_2025_SCHEDULE,
)
from scripts.analyze_v193_multi_round_results import parse_round_data, format_markdown_report


def test_candidate_profile_all_mandatory_constraints() -> None:
    profile_path = Path("configs/experiments/reasoning_rl_v1_active_candidate.yaml")
    assert profile_path.exists(), "Profile YAML missing"
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    learning = data.get("learning_overrides", {})
    # 1. five_lock_non_entry_to_hold: true
    assert learning.get("five_lock_non_entry_to_hold") is True
    # 2. five_fixed_geometry_enabled: true
    assert learning.get("five_fixed_geometry_enabled") is True
    # 3. context_direction_entry_gate_enabled: false (STRICTLY FALSE - NO HARD DIRECTION GATES!)
    assert learning.get("context_direction_entry_gate_enabled") is False
    # 4. five_action_ranking_bidirectional_enabled: true
    assert learning.get("five_action_ranking_bidirectional_enabled") is True
    # 5. five_expectancy_action_veto_enabled: true
    assert learning.get("five_expectancy_action_veto_enabled") is True
    # 6. five_live_action_veto_enabled: true
    assert learning.get("five_live_action_veto_enabled") is True
    # 7. five_expectancy_action_veto_min_samples: 6
    assert learning.get("five_expectancy_action_veto_min_samples") == 6
    # 8. five_live_action_veto_min_samples: 6
    assert learning.get("five_live_action_veto_min_samples") == 6

    # Fixed barriers
    orders = data.get("order_overrides", {})
    assert orders.get("fixed_sl_atr") == 5.0
    assert orders.get("fixed_tp_atr") == 10.0
    assert orders.get("max_holding_bars") == 120


def test_evaluate_promotion_pass_and_fail_cases() -> None:
    passing_metrics = {
        "expectancy_r": 0.08,
        "win_rate_pct": 45.0,
        "max_drawdown_pct": 5.2,
        "payoff_ratio": 1.75,
        "premature_closes": 0,
    }
    promoted, reasons = evaluate_promotion(passing_metrics)
    assert promoted is True
    assert len(reasons) == 0

    # Failing E[R]
    fail_exp = dict(passing_metrics, expectancy_r=0.02)
    promoted, reasons = evaluate_promotion(fail_exp)
    assert promoted is False
    assert any("E[R]" in r for r in reasons)

    # Failing Win Rate
    fail_wr = dict(passing_metrics, win_rate_pct=35.0)
    promoted, reasons = evaluate_promotion(fail_wr)
    assert promoted is False
    assert any("Win Rate" in r for r in reasons)

    # Failing Max DD
    fail_dd = dict(passing_metrics, max_drawdown_pct=12.5)
    promoted, reasons = evaluate_promotion(fail_dd)
    assert promoted is False
    assert any("Max DD" in r for r in reasons)

    # Failing Payoff Ratio
    fail_payoff = dict(passing_metrics, payoff_ratio=1.2)
    promoted, reasons = evaluate_promotion(fail_payoff)
    assert promoted is False
    assert any("Payoff Ratio" in r for r in reasons)

    # Failing Premature Close (Governor violation)
    fail_close = dict(passing_metrics, premature_closes=1)
    promoted, reasons = evaluate_promotion(fail_close)
    assert promoted is False
    assert any("Premature closes" in r for r in reasons)


def test_q1_schedule_completeness() -> None:
    assert len(Q1_2025_SCHEDULE) >= 10
    rounds = [s["round"] for s in Q1_2025_SCHEDULE]
    assert rounds[:10] == list(range(1, 11))
    assert Q1_2025_SCHEDULE[0]["from_date"] == "2025.01.06"


def test_format_markdown_report() -> None:
    summary = {
        "schema_version": 2,
        "total_rounds": 1,
        "zero_premature_closes_all_rounds": True,
        "trend_learning": {
            "round_1_pct_short_in_uptrend": 60.0,
            "latest_round_pct_short_in_uptrend": 25.0,
            "reduction_pct_vs_round_1": 58.33,
            "trend_learning_passed": True,
        },
        "promoted_rounds": [1],
        "rounds": [
            {
                "round": 1,
                "from_date": "2025.01.06",
                "to_date": "2025.01.08",
                "total_trades": 25,
                "win_rate_pct": 52.0,
                "expectancy_r": 0.12,
                "payoff_ratio": 1.85,
                "max_drawdown_pct": 3.4,
                "net_pnl": 150.0,
                "close_reasons": {"TP": 13, "SL": 12},
                "pct_short_in_uptrend": 25.0,
                "uptrend_short_entries": 3,
                "uptrend_entries_total": 12,
                "promoted": True,
            }
        ],
    }
    report = format_markdown_report(summary)
    assert "POLICY_CLOSE_ALL == 0" in report
    assert "PASSED" in report
    assert "58.33%" in report


def test_multi_dimensional_trend_learning_evaluation() -> None:
    from scripts.analyze_v193_multi_round_results import analyze_multi_round_loop

    loop_dir = Path("outputs/he_chuyen_gia/v193_runs/multi_round_v193")
    if not loop_dir.exists():
        pytest.skip("multi_round_v193 directory not found")

    summary = analyze_multi_round_loop(loop_dir)
    tl = summary.get("trend_learning", {})

    # Invariant: Zero premature closes
    assert summary.get("zero_premature_closes_all_rounds") is True

    # 1. Promoted Champion Checkpoint (Round 6)
    champ_red = tl.get("promoted_champion_short_in_uptrend_reduction_pct", 0.0)
    assert champ_red >= 50.0, f"Champion reduction {champ_red}% < 50%"
    assert abs(champ_red - 91.07) < 0.2, f"Champion reduction expected ~91.07%, got {champ_red}%"

    # 2. Second-Half Multi-Round Aggregate (Rounds 6-10)
    late_red = tl.get("post_convergence_aggregate_reduction_pct", 0.0)
    assert late_red >= 50.0, f"Post-convergence aggregate reduction {late_red}% < 50%"
    assert abs(late_red - 73.37) < 0.5, f"Post-convergence reduction expected ~73.37%, got {late_red}%"

    # 3. Endpoint (Round 10)
    end_red = tl.get("final_endpoint_reduction_pct", 0.0)
    assert abs(end_red - 48.06) < 0.2, f"Endpoint reduction expected ~48.06%, got {end_red}%"

    # 4. Overall Trend Learning Gate Passed
    assert tl.get("trend_learning_passed") is True

    # 5. Markdown report format verification
    report = format_markdown_report(summary)
    assert "POLICY_CLOSE_ALL == 0" in report
    assert "PASSED" in report
    assert "91.07%" in report
    assert "73.37%" in report

