"""Tests for V193 Phase 3 Interleaved Curriculum Multi-Round Loop."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from scripts.run_v193_phase3_interleaved_loop import (
    PHASE3_INTERLEAVED_SCHEDULE,
    evaluate_phase3_promotion,
    extract_phase3_metrics,
)


def test_phase3_schedule_interleaving() -> None:
    """Verify that Phase 3 schedule has at least 8 rounds and actively interleaves regimes."""
    assert len(PHASE3_INTERLEAVED_SCHEDULE) >= 8
    
    types = [r["type"] for r in PHASE3_INTERLEAVED_SCHEDULE]
    # Verify alternation: should not have 3 identical consecutive regimes
    for i in range(len(types) - 2):
        assert not (types[i] == types[i+1] == types[i+2] == "uptrend"), f"Monotonous uptrend at round {i+1}"
        assert not (types[i] == types[i+1] == types[i+2] == "downtrend"), f"Monotonous downtrend at round {i+1}"
        
    for r in PHASE3_INTERLEAVED_SCHEDULE:
        assert "round" in r
        assert "from_date" in r
        assert "to_date" in r
        assert "regime" in r
        assert "type" in r
        assert r["from_date"].startswith("2025.")
        assert r["to_date"].startswith("2025.")


def test_evaluate_phase3_promotion_hurdles() -> None:
    """Verify Phase 3 promotion evaluation enforces Pareto standards and zero premature closes."""
    pass_metrics = {
        "expectancy_r": 0.075,
        "win_rate_pct": 46.2,
        "max_drawdown_pct": 4.8,
        "payoff_ratio": 1.65,
        "premature_closes": 0,
        "pct_long_in_downtrend": 25.0,
        "pct_short_in_uptrend": 10.0,
    }
    promoted, reasons = evaluate_phase3_promotion(pass_metrics)
    assert promoted is True
    assert len(reasons) == 0

    fail_metrics = {
        "expectancy_r": -0.05,
        "win_rate_pct": 32.0,
        "max_drawdown_pct": 11.5,
        "payoff_ratio": 1.2,
        "premature_closes": 1,
        "pct_long_in_downtrend": 65.0,
        "pct_short_in_uptrend": 45.0,
    }
    promoted_f, reasons_f = evaluate_phase3_promotion(fail_metrics)
    assert promoted_f is False
    assert len(reasons_f) >= 4
    assert any("Premature closes" in r for r in reasons_f)
    assert any("E[R]" in r for r in reasons_f)
    assert any("WR" in r for r in reasons_f)
    assert any("Max DD" in r for r in reasons_f)


def test_phase3_active_candidate_config_invariants() -> None:
    """Verify that configuration invariants are maintained: no hard gates, locked governor, fixed geometry."""
    profile_path = Path("configs/experiments/reasoning_rl_v1_active_candidate.yaml")
    assert profile_path.exists()
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    learning = data.get("learning_overrides", {})

    # No hard gates
    assert learning.get("context_direction_entry_gate_enabled") is False
    # Governor locked to HOLD
    assert learning.get("five_lock_non_entry_to_hold") is True
    # Fixed geometry enabled
    assert learning.get("five_fixed_geometry_enabled") is True
    # Bidirectional ranking & RL vetoes active
    assert learning.get("five_action_ranking_bidirectional_enabled") is True
    assert learning.get("five_expectancy_action_veto_enabled") is True
    assert learning.get("five_live_action_veto_enabled") is True

    orders = data.get("order_overrides", {})
    assert orders.get("fixed_sl_atr") == 5.0
    assert orders.get("fixed_tp_atr") == 10.0
    assert orders.get("max_holding_bars") == 120
