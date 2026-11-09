"""Tests for V193 Phase 2 Downtrend Multi-Round Training Loop."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from scripts.run_v193_phase2_downtrend_loop import (
    PHASE2_SCHEDULE,
    evaluate_phase2_promotion,
    extract_phase2_metrics,
)


def test_phase2_schedule_completeness() -> None:
    """Verify that Phase 2 schedule has at least 10 rounds covering late Jan to mid Feb 2025."""
    assert len(PHASE2_SCHEDULE) >= 10
    for r in PHASE2_SCHEDULE:
        assert "round" in r
        assert "from_date" in r
        assert "to_date" in r
        assert r["from_date"].startswith("2025.")
        assert r["to_date"].startswith("2025.")


def test_evaluate_phase2_promotion() -> None:
    """Verify Phase 2 promotion evaluation handles both uptrend short suppression and downtrend long suppression."""
    pass_metrics = {
        "expectancy_r": 0.08,
        "win_rate_pct": 45.0,
        "max_drawdown_pct": 5.0,
        "payoff_ratio": 1.7,
        "premature_closes": 0,
        "pct_long_in_downtrend": 20.0,
        "pct_short_in_uptrend": 5.0,
    }
    promoted, reasons = evaluate_phase2_promotion(pass_metrics)
    assert promoted is True
    assert len(reasons) == 0

    fail_metrics = {
        "expectancy_r": -0.1,
        "win_rate_pct": 30.0,
        "max_drawdown_pct": 12.0,
        "payoff_ratio": 1.1,
        "premature_closes": 1,
        "pct_long_in_downtrend": 70.0,
        "pct_short_in_uptrend": 50.0,
    }
    promoted_f, reasons_f = evaluate_phase2_promotion(fail_metrics)
    assert promoted_f is False
    assert len(reasons_f) >= 4
