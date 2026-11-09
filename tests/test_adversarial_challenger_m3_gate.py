"""Empirical Adversarial Stress Test Suite for Milestone 3: Multi-Regime Gate Evaluator.

Author: Challenger 1 (m3_challenger_1)
Role: critic, specialist (Empirical Challenger)
Integrity mode: development (empirical evidence before claims)

Adversarially tests:
1. Zero trades in a regime (all regimes empty vs single regime zero-trade bypass)
2. All wins / zero losses (canonical fallback payoff == 2.0, zero division safety)
3. All losses / zero wins (payoff == 0.0, zero division safety)
4. Zero realized R trades (handling realized_net_r == 0.0)
5. Sub-decimal borderline hurdle boundaries:
   - Win Rate: 44.99% (FAIL) vs 45.00% (PASS)
   - Payoff Ratio: 1.499R (FAIL) vs 1.500R (PASS)
   - Expectancy: +0.0499R (FAIL) vs +0.0500R (PASS)
   - Counter-Trend: 30.01% (FAIL) vs 30.00% (PASS)
   - Timeout Close Rate: 15.00% (FAIL) vs 14.99% (PASS)
   - Premature Closes: 1 (FAIL) vs 0 (PASS)
6. Malformed and corrupted telemetry records (unhandled exceptions vs fail-closed)
7. Non-finite metric values (NaN, Inf)
8. Weight immutability enforcement during frozen OOS evaluation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from self_growing_trader.multi_regime_gate import (
    DEFAULT_GATE_HURDLES,
    GateHurdles,
    MultiRegimeGateEvaluator,
    MultiRegimeResult,
    PolicyImmutabilityError,
    RegimeMetrics,
    evaluate_multi_regime_gate,
    extract_regime_metrics_from_telemetry,
)


def _build_passing_regimes(base_n: int = 30) -> dict[str, RegimeMetrics]:
    """Helper creating a baseline passing set of 3 regimes (Bull, Bear, Consolidation)."""
    return {
        "bull": RegimeMetrics(
            regime_id="bull",
            from_date="2025.01.14",
            to_date="2025.01.18",
            market_type="uptrend",
            total_trades=base_n,
            wins=int(base_n * 0.60),
            losses=base_n - int(base_n * 0.60),
            win_rate_pct=60.0,
            expectancy_r=0.25,
            payoff_ratio=2.0,
            profit_factor=2.1,
            max_drawdown_pct=3.5,
            timeout_closes=1,
            timeout_close_rate_pct=3.33,
            counter_trend_entries=3,
            total_directional_entries=base_n,
            counter_trend_rate_pct=10.0,
            premature_closes=0,
        ),
        "bear": RegimeMetrics(
            regime_id="bear",
            from_date="2025.01.06",
            to_date="2025.01.10",
            market_type="downtrend",
            total_trades=base_n,
            wins=int(base_n * 0.55),
            losses=base_n - int(base_n * 0.55),
            win_rate_pct=55.0,
            expectancy_r=0.20,
            payoff_ratio=1.85,
            profit_factor=1.9,
            max_drawdown_pct=4.0,
            timeout_closes=2,
            timeout_close_rate_pct=6.67,
            counter_trend_entries=4,
            total_directional_entries=base_n,
            counter_trend_rate_pct=13.33,
            premature_closes=0,
        ),
        "consolidation": RegimeMetrics(
            regime_id="consolidation",
            from_date="2025.01.24",
            to_date="2025.01.28",
            market_type="sideway",
            total_trades=base_n,
            wins=int(base_n * 0.50),
            losses=base_n - int(base_n * 0.50),
            win_rate_pct=50.0,
            expectancy_r=0.10,
            payoff_ratio=1.60,
            profit_factor=1.5,
            max_drawdown_pct=2.8,
            timeout_closes=2,
            timeout_close_rate_pct=6.67,
            counter_trend_entries=0,
            total_directional_entries=base_n,
            counter_trend_rate_pct=0.0,
            premature_closes=0,
        ),
    }


# ==============================================================================
# CHALLENGE 1: ZERO TRADES & SAMPLE SUFFICIENCY
# ==============================================================================

def test_challenger_all_regimes_zero_trades_fails_closed(tmp_path: Path) -> None:
    """Verifies that 0 total trades fails closed with explicit reasons and no ZeroDivisionError."""
    empty_dir = tmp_path / "empty_telemetry"
    empty_dir.mkdir()

    metrics = extract_regime_metrics_from_telemetry(empty_dir, "bull", {"market_type": "uptrend"})
    assert metrics.total_trades == 0
    assert metrics.wins == 0
    assert metrics.losses == 0
    assert metrics.win_rate_pct == 0.0
    assert metrics.expectancy_r == 0.0
    assert metrics.payoff_ratio == 0.0
    assert metrics.timeout_close_rate_pct == 0.0
    assert metrics.counter_trend_rate_pct == 0.0

    regimes = {
        "bull": metrics,
        "bear": RegimeMetrics("bear", market_type="downtrend"),
        "consolidation": RegimeMetrics("consolidation", market_type="sideway"),
    }
    passed, reasons, agg = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("total_trades_insufficient" in r for r in reasons)
    assert any("win_rate_below_hurdle" in r for r in reasons)
    assert any("expectancy_below_hurdle" in r for r in reasons)
    assert any("payoff_ratio_below_hurdle" in r for r in reasons)


def test_challenger_single_regime_zero_trades_bypass_vulnerability() -> None:
    """Adversarial vulnerability probe: A candidate with 0 trades in Bear market.

    VULNERABILITY DEMONSTRATION:
    `GateHurdles` declares `min_trades_per_regime: int = 1`.
    However, `evaluate_multi_regime_gate()` NEVER evaluates `min_trades_per_regime`.
    If Bull and Consolidation have high win rate and trades >= 30, a candidate that
    took ZERO trades in Bear market (e.g. completely froze or failed to execute)
    will bypass evaluation and be promoted!
    """
    regimes = {
        "bull": RegimeMetrics(
            "bull",
            market_type="uptrend",
            total_trades=40,
            wins=35,
            losses=5,
            win_rate_pct=87.5,
            expectancy_r=1.5,
            payoff_ratio=3.0,
            counter_trend_rate_pct=5.0,
            timeout_close_rate_pct=2.5,
            premature_closes=0,
        ),
        "bear": RegimeMetrics(
            "bear",
            market_type="downtrend",
            total_trades=0,  # Zero trades taken in Bear!
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            expectancy_r=0.0,
            payoff_ratio=0.0,
            counter_trend_rate_pct=0.0,
            timeout_close_rate_pct=0.0,
            premature_closes=0,
        ),
        "consolidation": RegimeMetrics(
            "consolidation",
            market_type="sideway",
            total_trades=40,
            wins=35,
            losses=5,
            win_rate_pct=87.5,
            expectancy_r=1.5,
            payoff_ratio=3.0,
            counter_trend_rate_pct=0.0,
            timeout_close_rate_pct=2.5,
            premature_closes=0,
        ),
    }

    # Empirical check: Does the gate check min_trades_per_regime?
    passed, reasons, agg = evaluate_multi_regime_gate(regimes)

    # REMEDIATION VERIFICATION:
    # aggregate trades = 80 >= 30, but bear has 0 trades (< min_trades_per_regime = 1).
    # Gate must fail closed and reject candidate.
    assert passed is False, "Evaluator must reject candidate with 0 trades in bear regime!"
    assert any("regime_trades_insufficient:bear" in r for r in reasons)


# ==============================================================================
# CHALLENGE 2: ALL WINS / ZERO LOSSES & DIVISION BY ZERO SAFETY
# ==============================================================================

def test_challenger_all_wins_zero_losses_canonical_fallback(tmp_path: Path) -> None:
    """Verifies that 100% wins and 0 losses evaluates Payoff Ratio to 2.0 without ZeroDivisionError."""
    run_dir = tmp_path / "all_wins"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    with open(events_file, "w", encoding="utf-8") as f:
        for _ in range(35):
            rec = {
                "event_type": "lifecycle",
                "payload": {
                    "realized_net_r": 2.5,
                    "close_reason": "TP",
                    "duration_bars": 35.0,
                    "growth_events": [
                        {"type": "context_action_expectancy_credit", "action_index": 1}
                    ],
                },
            }
            f.write(json.dumps(rec) + "\n")

    m = extract_regime_metrics_from_telemetry(run_dir, "bull", {"market_type": "uptrend"})
    assert m.total_trades == 35
    assert m.wins == 35
    assert m.losses == 0
    assert m.win_rate_pct == 100.0
    assert m.payoff_ratio == 2.0  # Canonical repo standard when no losses occur
    assert m.expectancy_r == 2.5


def test_challenger_all_losses_zero_wins(tmp_path: Path) -> None:
    """Verifies that 0 wins and 100% losses evaluates Payoff Ratio to 0.0 without ZeroDivisionError."""
    run_dir = tmp_path / "all_losses"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    with open(events_file, "w", encoding="utf-8") as f:
        for _ in range(35):
            rec = {
                "event_type": "lifecycle",
                "payload": {
                    "realized_net_r": -1.2,
                    "close_reason": "SL",
                    "duration_bars": 20.0,
                    "growth_events": [
                        {"type": "context_action_expectancy_credit", "action_index": 1}
                    ],
                },
            }
            f.write(json.dumps(rec) + "\n")

    m = extract_regime_metrics_from_telemetry(run_dir, "bull", {"market_type": "uptrend"})
    assert m.total_trades == 35
    assert m.wins == 0
    assert m.losses == 35
    assert m.win_rate_pct == 0.0
    assert m.payoff_ratio == 0.0
    assert m.expectancy_r == -1.2


def test_challenger_zero_realized_r_trades(tmp_path: Path) -> None:
    """Verifies that breakeven trades (realized_net_r == 0.0) do not cause divide-by-zero."""
    run_dir = tmp_path / "zero_r"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    with open(events_file, "w", encoding="utf-8") as f:
        for _ in range(20):
            rec = {
                "event_type": "lifecycle",
                "payload": {
                    "realized_net_r": 0.0,
                    "close_reason": "BE",
                    "duration_bars": 10.0,
                    "growth_events": [],
                },
            }
            f.write(json.dumps(rec) + "\n")

    m = extract_regime_metrics_from_telemetry(run_dir, "bull", {"market_type": "uptrend"})
    assert m.total_trades == 20
    assert m.wins == 0
    assert m.losses == 20
    assert m.avg_loss_r == 0.0
    assert m.payoff_ratio == 0.0


# ==============================================================================
# CHALLENGE 3: SUB-DECIMAL BORDERLINE BOUNDARY CONDITIONS
# ==============================================================================

def test_challenger_borderline_win_rate_boundary() -> None:
    """Tests exact win rate threshold: 44.99% (FAIL) vs 45.00% (PASS)."""
    regimes = _build_passing_regimes(base_n=10000)
    # 4499 wins out of 10000 trades -> 44.99%
    for r_id, m in regimes.items():
        m.total_trades = 3333
        m.wins = 1500
        m.losses = 1833
    regimes["bull"].total_trades = 3334
    regimes["bull"].wins = 1500
    regimes["bear"].wins = 1500
    regimes["consolidation"].wins = 1499  # Total wins: 4499 / 10000 = 44.99%

    passed_fail, reasons_fail, agg_fail = evaluate_multi_regime_gate(regimes)
    assert passed_fail is False
    assert any("win_rate_below_hurdle" in r for r in reasons_fail)
    assert agg_fail["win_rate_pct"] == 44.99

    # Add exactly 1 win -> 4500 wins / 10000 = 45.00%
    regimes["consolidation"].wins = 1500
    passed_pass, reasons_pass, agg_pass = evaluate_multi_regime_gate(regimes)
    assert passed_pass is True
    assert agg_pass["win_rate_pct"] == 45.00


def test_challenger_borderline_payoff_ratio_boundary() -> None:
    """Tests exact payoff ratio threshold: 1.499R (FAIL) vs 1.500R (PASS)."""
    regimes = _build_passing_regimes()
    for m in regimes.values():
        m.payoff_ratio = 1.499

    passed_fail, reasons_fail, _ = evaluate_multi_regime_gate(regimes)
    assert passed_fail is False
    assert any("payoff_ratio_below_hurdle" in r for r in reasons_fail)

    for m in regimes.values():
        m.payoff_ratio = 1.500

    passed_pass, reasons_pass, _ = evaluate_multi_regime_gate(regimes)
    assert passed_pass is True


def test_challenger_borderline_expectancy_boundary() -> None:
    """Tests exact expectancy threshold: +0.0499R (FAIL) vs +0.0500R (PASS)."""
    regimes = _build_passing_regimes()
    for m in regimes.values():
        m.expectancy_r = 0.0499

    passed_fail, reasons_fail, _ = evaluate_multi_regime_gate(regimes)
    assert passed_fail is False
    assert any("expectancy_below_hurdle" in r for r in reasons_fail)

    for m in regimes.values():
        m.expectancy_r = 0.0500

    passed_pass, reasons_pass, _ = evaluate_multi_regime_gate(regimes)
    assert passed_pass is True


def test_challenger_borderline_counter_trend_boundary() -> None:
    """Tests exact counter-trend threshold: 30.01% (FAIL) vs 30.00% (PASS)."""
    regimes = _build_passing_regimes()
    regimes["bull"].counter_trend_rate_pct = 30.01

    passed_fail, reasons_fail, _ = evaluate_multi_regime_gate(regimes)
    assert passed_fail is False
    assert any("counter_trend_rate_uptrend_exceeded" in r for r in reasons_fail)

    regimes["bull"].counter_trend_rate_pct = 30.00
    passed_pass, reasons_pass, _ = evaluate_multi_regime_gate(regimes)
    assert passed_pass is True

    # Same check for bear
    regimes["bear"].counter_trend_rate_pct = 30.01
    passed_bear_fail, reasons_bear_fail, _ = evaluate_multi_regime_gate(regimes)
    assert passed_bear_fail is False
    assert any("counter_trend_rate_downtrend_exceeded" in r for r in reasons_bear_fail)

    regimes["bear"].counter_trend_rate_pct = 30.00
    passed_bear_pass, _, _ = evaluate_multi_regime_gate(regimes)
    assert passed_bear_pass is True


def test_challenger_borderline_timeout_boundary() -> None:
    """Tests timeout close threshold: 15.00% (FAIL) vs 14.99% (PASS).

    Requirement in ORIGINAL_REQUEST: "giảm xuống dưới 15%" (strictly < 15.0%).
    """
    regimes = _build_passing_regimes()
    regimes["consolidation"].timeout_close_rate_pct = 15.00

    passed_fail, reasons_fail, _ = evaluate_multi_regime_gate(regimes)
    assert passed_fail is False
    assert any("timeout_close_rate_exceeded:consolidation" in r for r in reasons_fail)

    regimes["consolidation"].timeout_close_rate_pct = 14.99
    passed_pass, reasons_pass, _ = evaluate_multi_regime_gate(regimes)
    assert passed_pass is True


def test_challenger_single_policy_close_all_rejection() -> None:
    """Tests Governor invariant: Exactly 1 POLICY_CLOSE_ALL triggers immediate rejection."""
    regimes = _build_passing_regimes()
    regimes["bear"].premature_closes = 1

    passed_fail, reasons_fail, _ = evaluate_multi_regime_gate(regimes)
    assert passed_fail is False
    assert any("premature_closes_violation" in r for r in reasons_fail)

    regimes["bear"].premature_closes = 0
    passed_pass, _, _ = evaluate_multi_regime_gate(regimes)
    assert passed_pass is True


# ==============================================================================
# CHALLENGE 4: CORRUPTED & MALFORMED TELEMETRY
# ==============================================================================

def test_challenger_corrupted_telemetry_unhandled_crash_vulnerability(tmp_path: Path) -> None:
    """Verifies that malformed payload values inside events.jsonl are handled gracefully without crashing."""
    # Test Case A: non-numeric realized_net_r -> discarded, yields 0 trades
    dir_a = tmp_path / "corrupt_r"
    dir_a.mkdir()
    (dir_a / "events.jsonl").write_text(
        json.dumps({"event_type": "lifecycle", "payload": {"realized_net_r": "invalid_string"}}) + "\n",
        encoding="utf-8",
    )
    metrics_a = extract_regime_metrics_from_telemetry(dir_a, "bull", {"market_type": "uptrend"})
    assert metrics_a.total_trades == 0
    assert metrics_a.win_rate_pct == 0.0

    # Test Case B: non-dict item in growth_events -> corrupt growth skipped, valid trade preserved
    dir_b = tmp_path / "corrupt_growth"
    dir_b.mkdir()
    (dir_b / "events.jsonl").write_text(
        json.dumps({
            "event_type": "lifecycle",
            "payload": {
                "realized_net_r": 1.5,
                "growth_events": ["corrupt_string_element"],
            },
        }) + "\n",
        encoding="utf-8",
    )
    metrics_b = extract_regime_metrics_from_telemetry(dir_b, "bull", {"market_type": "uptrend"})
    assert metrics_b.total_trades == 1
    assert metrics_b.wins == 1
    assert metrics_b.expectancy_r == 1.5
    assert metrics_b.counter_trend_rate_pct == 0.0

    # Test Case C: duration_bars is None -> defaults to 0.0 duration, valid trade preserved
    dir_c = tmp_path / "corrupt_duration"
    dir_c.mkdir()
    (dir_c / "events.jsonl").write_text(
        json.dumps({
            "event_type": "lifecycle",
            "payload": {
                "realized_net_r": 2.0,
                "duration_bars": None,
                "close_reason": "TP",
            },
        }) + "\n",
        encoding="utf-8",
    )
    metrics_c = extract_regime_metrics_from_telemetry(dir_c, "bull", {"market_type": "uptrend"})
    assert metrics_c.total_trades == 1
    assert metrics_c.wins == 1
    assert metrics_c.timeout_closes == 0


def test_challenger_corrupted_diagnostics_fails_gracefully(tmp_path: Path) -> None:
    """Verifies that corrupted mt5_diagnostics.json does not crash extraction."""
    dir_c = tmp_path / "corrupt_diag"
    dir_c.mkdir()
    (dir_c / "events.jsonl").write_text(
        json.dumps({
            "event_type": "lifecycle",
            "payload": {"realized_net_r": 1.0, "duration_bars": 20.0, "close_reason": "TP"},
        }) + "\n",
        encoding="utf-8",
    )
    (dir_c / "mt5_diagnostics.json").write_text("NOT_VALID_JSON{:::}", encoding="utf-8")

    # Should not crash because mt5_diagnostics.json is guarded by try-except
    metrics = extract_regime_metrics_from_telemetry(dir_c, "bull", {"market_type": "uptrend"})
    assert metrics.total_trades == 1
    assert metrics.wins == 1
    assert metrics.profit_factor == 0.0


# ==============================================================================
# CHALLENGE 5: NON-FINITE METRICS & WEIGHT IMMUTABILITY
# ==============================================================================

def test_challenger_nan_and_inf_metric_handling() -> None:
    """Verifies that NaN and Inf metrics fail closed with invalid_metric_value."""
    regimes = _build_passing_regimes()
    regimes["bull"].payoff_ratio = float("nan")

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("invalid_metric_value:bull:payoff=nan" in r for r in reasons)

    regimes["bull"].payoff_ratio = 2.0
    regimes["bear"].expectancy_r = float("-inf")
    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("invalid_metric_value:bear:expectancy=-inf" in r for r in reasons)


def test_challenger_weight_immutability_enforcement(tmp_path: Path) -> None:
    """Verifies that candidate policy mutation during evaluation raises PolicyImmutabilityError."""
    cand_file = tmp_path / "candidate_policy.pkl"
    cand_file.write_bytes(b"initial_frozen_weights_v193")

    evaluator = MultiRegimeGateEvaluator()

    # Evaluator function that maliciously mutates the candidate checkpoint file
    def mutating_evaluator(r_id: str, cfg: dict[str, str], out_dir: Path) -> Path:
        cand_file.write_bytes(b"MUTATED_MODIFIED_WEIGHTS_BREACH")
        b = _build_passing_regimes()[r_id]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("")
        return out_dir

    with pytest.raises(PolicyImmutabilityError, match="Policy hash mutated during evaluation"):
        evaluator.evaluate_candidate(
            candidate_path=cand_file,
            output_dir=tmp_path / "eval_out",
            evaluator_fn=mutating_evaluator,
        )
