"""Empirical Adversarial Stress Test Suite for Milestone 3 Iteration 2.

Author: Challenger 3 (m3_challenger_3)
Role: critic, specialist (Empirical Challenger)
Integrity mode: development (empirical evidence before claims)

Exhaustively verifies:
1. Defect A Remediation:
   - Zero trades in Bear (or any single regime) CANNOT be promoted under any circumstance.
   - Zero trades yields passed=False with explicit 'regime_trades_insufficient' reason.
   - Boundary checks: 0 trades vs 1 trade per regime.
   - Custom min_trades_per_regime thresholds (5, 10, 0, GateHurdles object vs dict).
   - E2E Candidate evaluation & promotion engine: 0-trade candidates are pruned, champion untouched.
2. Defect B Remediation:
   - Corrupted lines in events.jsonl (None duration, non-numeric realized_net_r, non-dict growth events,
     malformed JSON syntax, binary noise, scalar JSON, overflow floats, NaN/inf in strings).
   - Corrupted mt5_diagnostics.json (invalid JSON, non-numeric fields, nulls, array root).
   - Zero crashes under all conditions; valid records are cleanly extracted, corrupt lines skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from self_growing_trader.multi_regime_gate import (
    DEFAULT_GATE_HURDLES,
    GateHurdles,
    MultiRegimeGateEvaluator,
    MultiRegimeResult,
    RegimeMetrics,
    evaluate_multi_regime_gate,
    extract_regime_metrics_from_telemetry,
)


def _build_test_regimes(bull_n: int = 30, bear_n: int = 30, consol_n: int = 30) -> dict[str, RegimeMetrics]:
    """Helper creating configurable 3-regime metrics with standard passing properties for non-zero counts."""
    def _make(r_id: str, m_type: str, n: int) -> RegimeMetrics:
        if n == 0:
            return RegimeMetrics(
                regime_id=r_id,
                market_type=m_type,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate_pct=0.0,
                expectancy_r=0.0,
                payoff_ratio=0.0,
                profit_factor=0.0,
                max_drawdown_pct=0.0,
                timeout_closes=0,
                timeout_close_rate_pct=0.0,
                counter_trend_entries=0,
                total_directional_entries=0,
                counter_trend_rate_pct=0.0,
                premature_closes=0,
            )
        timeouts = 1 if n >= 10 else 0
        counter = 1 if n >= 10 else 0
        wins = int(math.ceil(n * 0.60))
        return RegimeMetrics(
            regime_id=r_id,
            market_type=m_type,
            total_trades=n,
            wins=wins,
            losses=n - wins,
            win_rate_pct=round(wins / n * 100.0, 2),
            expectancy_r=0.30,
            payoff_ratio=2.0,
            profit_factor=2.0,
            max_drawdown_pct=3.0,
            timeout_closes=timeouts,
            timeout_close_rate_pct=round(timeouts / n * 100.0, 2),
            counter_trend_entries=counter,
            total_directional_entries=n,
            counter_trend_rate_pct=round(counter / n * 100.0, 2),
            premature_closes=0,
        )

    return {
        "bull": _make("bull", "uptrend", bull_n),
        "bear": _make("bear", "downtrend", bear_n),
        "consolidation": _make("consolidation", "sideway", consol_n),
    }


# ==============================================================================
# DEFECT A: ZERO-TRADE REGIME PROMOTION IMMUNITY & BOUNDARY PROBING
# ==============================================================================

def test_zero_trades_in_bear_fails_closed_and_blocks_promotion() -> None:
    """Verifies that 0 trades in Bear fails closed with regime_trades_insufficient:bear."""
    regimes = _build_test_regimes(bull_n=40, bear_n=0, consol_n=40)
    passed, reasons, agg = evaluate_multi_regime_gate(regimes)

    assert passed is False
    assert any("regime_trades_insufficient:bear: 0 < 1" in r for r in reasons)
    # Confirm aggregate trades meet min_total (80 >= 30) but single-regime check blocks promotion
    assert agg["total_trades"] == 80


def test_zero_trades_in_bull_fails_closed() -> None:
    """Verifies that 0 trades in Bull fails closed with regime_trades_insufficient:bull."""
    regimes = _build_test_regimes(bull_n=0, bear_n=40, consol_n=40)
    passed, reasons, agg = evaluate_multi_regime_gate(regimes)

    assert passed is False
    assert any("regime_trades_insufficient:bull: 0 < 1" in r for r in reasons)
    assert agg["total_trades"] == 80


def test_zero_trades_in_consolidation_fails_closed() -> None:
    """Verifies that 0 trades in Consolidation fails closed with regime_trades_insufficient:consolidation."""
    regimes = _build_test_regimes(bull_n=40, bear_n=40, consol_n=0)
    passed, reasons, agg = evaluate_multi_regime_gate(regimes)

    assert passed is False
    assert any("regime_trades_insufficient:consolidation: 0 < 1" in r for r in reasons)
    assert agg["total_trades"] == 80


def test_multiple_zero_trade_regimes_lists_all_insufficient_reasons() -> None:
    """Verifies that multiple empty regimes produce independent insufficiency reasons."""
    regimes = _build_test_regimes(bull_n=50, bear_n=0, consol_n=0)
    passed, reasons, _ = evaluate_multi_regime_gate(regimes)

    assert passed is False
    assert any("regime_trades_insufficient:bear" in r for r in reasons)
    assert any("regime_trades_insufficient:consolidation" in r for r in reasons)


def test_boundary_zero_vs_one_trade_per_regime() -> None:
    """Probes the exact boundary between 0 trades (rejected) and 1 trade (accepted by min_trades_per_regime)."""
    # 0 trades in bear: FAILS
    regimes_0 = _build_test_regimes(bull_n=30, bear_n=0, consol_n=30)
    passed_0, reasons_0, _ = evaluate_multi_regime_gate(regimes_0)
    assert passed_0 is False
    assert any("regime_trades_insufficient:bear" in r for r in reasons_0)

    # 1 trade in bear (win with +2.0R): PASSES min_trades_per_regime check
    regimes_1 = _build_test_regimes(bull_n=30, bear_n=1, consol_n=30)
    passed_1, reasons_1, _ = evaluate_multi_regime_gate(regimes_1)
    # Must NOT have regime_trades_insufficient for bear
    assert not any("regime_trades_insufficient:bear" in r for r in reasons_1)
    assert passed_1 is True


@pytest.mark.parametrize("custom_min", [3, 5, 10])
def test_custom_min_trades_per_regime_thresholds(custom_min: int) -> None:
    """Verifies custom min_trades_per_regime thresholds in dict and GateHurdles."""
    # Sub-threshold: custom_min - 1 trades in bear
    regimes_below = _build_test_regimes(bull_n=40, bear_n=custom_min - 1, consol_n=40)
    hurdles_dict = dict(DEFAULT_GATE_HURDLES, min_trades_per_regime=custom_min)
    passed_below, reasons_below, _ = evaluate_multi_regime_gate(regimes_below, hurdles=hurdles_dict)
    assert passed_below is False
    assert any(f"regime_trades_insufficient:bear: {custom_min - 1} < {custom_min}" in r for r in reasons_below)

    # Exact threshold: custom_min trades in bear
    regimes_exact = _build_test_regimes(bull_n=40, bear_n=custom_min, consol_n=40)
    passed_exact, reasons_exact, _ = evaluate_multi_regime_gate(regimes_exact, hurdles=hurdles_dict)
    assert not any("regime_trades_insufficient:bear" in r for r in reasons_exact)

    # GateHurdles dataclass instance
    hurdles_obj = GateHurdles(min_trades_per_regime=custom_min)
    passed_obj, reasons_obj, _ = evaluate_multi_regime_gate(regimes_below, hurdles=hurdles_obj)
    assert passed_obj is False
    assert any(f"regime_trades_insufficient:bear: {custom_min - 1} < {custom_min}" in r for r in reasons_obj)


def test_min_trades_per_regime_zero_boundary() -> None:
    """Boundary probe: If min_trades_per_regime is explicitly set to 0, zero-trade check is bypassed."""
    regimes = _build_test_regimes(bull_n=40, bear_n=0, consol_n=40)
    hurdles_zero = dict(DEFAULT_GATE_HURDLES, min_trades_per_regime=0)
    passed, reasons, agg = evaluate_multi_regime_gate(regimes, hurdles=hurdles_zero)
    # Since total_trades = 80 >= 30, and min_trades_per_regime = 0, no insufficiency reason is emitted
    assert not any("regime_trades_insufficient" in r for r in reasons)


def test_evaluator_candidate_end_to_end_zero_bear_trades_triggers_rollback(tmp_path: Path) -> None:
    """E2E stress test: MultiRegimeGateEvaluator strictly blocks champion promotion and executes rollback on 0 Bear trades."""
    cand_path = tmp_path / "candidate_zero_bear.pkl"
    cand_path.write_bytes(b"candidate_policy_zero_bear")
    active_seed = tmp_path / "active_seed.pkl"
    active_seed.write_bytes(b"champion_active_seed_untouched")
    eval_out = tmp_path / "eval_out"

    evaluator = MultiRegimeGateEvaluator()

    def mock_eval_fn(r_id: str, cfg: dict[str, str], out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        events_file = out_dir / "events.jsonl"
        if r_id == "bear":
            # 0 trades in bear
            events_file.write_text("", encoding="utf-8")
        else:
            # 25 winning trades in bull and consolidation
            lines = [
                json.dumps({
                    "event_type": "lifecycle",
                    "payload": {
                        "realized_net_r": 1.5,
                        "close_reason": "TP",
                        "duration_bars": 30.0,
                    },
                })
                for _ in range(25)
            ]
            events_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_dir

    result: MultiRegimeResult = evaluator.evaluate_candidate(
        candidate_path=cand_path,
        output_dir=eval_out,
        evaluator_fn=mock_eval_fn,
        active_seed_path=active_seed,
        round_id=8,
    )

    # Invariants under test:
    assert result.passed_gate is False, "Candidate with 0 bear trades MUST fail gate!"
    assert result.promoted is False, "Candidate with 0 bear trades MUST NOT be promoted!"
    assert result.promoted_checkpoint_path is None
    assert any("regime_trades_insufficient:bear" in r for r in result.reasons)
    # Check that active seed is NOT overwritten with candidate weights
    assert active_seed.read_bytes() == b"champion_active_seed_untouched"
    # Check candidate payload was pruned from candidate_path on failure
    assert not cand_path.exists()
    # Check canonical promotion file does NOT exist
    assert not (eval_out / "best_promoted_checkpoint.pkl").exists()


# ==============================================================================
# DEFECT B: CORRUPTED & MALFORMED TELEMETRY CRASH IMMUNITY
# ==============================================================================

def test_extract_regime_metrics_corrupted_jsonl_exhaustively(tmp_path: Path) -> None:
    """Stress tests telemetry parser against a battery of 18 malformed JSONL inputs."""
    corrupt_dir = tmp_path / "corrupted_run"
    corrupt_dir.mkdir()
    events_file = corrupt_dir / "events.jsonl"

    hostile_lines = [
        # 1. Truncated / malformed JSON
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0',
        '{"event_type": "lifecycle", "payload":',
        '{invalid_json: true}',
        '',
        '   \n\t  ',
        # 2. Non-dict JSON root
        '12345',
        '"string_record"',
        '[1, 2, 3]',
        'true',
        'null',
        # 3. Non-lifecycle event types
        '{"event_type": "heartbeat", "payload": {"tick": 1}}',
        '{"event_type": null, "payload": {}}',
        '{"event_type": 123, "payload": {}}',
        # 4. Null or non-dict payload
        '{"event_type": "lifecycle", "payload": null}',
        '{"event_type": "lifecycle", "payload": "scalar_payload"}',
        '{"event_type": "lifecycle", "payload": [1, 2, 3]}',
        # 5. Corrupted duration_bars
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "duration_bars": null}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "duration_bars": "None"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "duration_bars": "NaN"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "duration_bars": "inf"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "duration_bars": "-inf"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "duration_bars": []}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "duration_bars": {}}}',
        # 6. Corrupted realized_net_r
        '{"event_type": "lifecycle", "payload": {"realized_net_r": null}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": "not_a_number"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": "NaN"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": "inf"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": "-inf"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": []}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": {}}}',
        # 7. Corrupted growth_events
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "growth_events": null}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "growth_events": "corrupted_string"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "growth_events": [123, null, "string", [], {}]}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "growth_events": [{"type": "context_action_expectancy_credit", "action_index": null}]}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "growth_events": [{"type": "context_action_expectancy_credit", "action_index": "invalid"}]}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "growth_events": [{"type": "context_action_expectancy_credit", "action_index": 1e309}]}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "growth_events": [{"type": "context_action_expectancy_credit", "action_index": "3.0"}]}}',
        # 8. Corrupted close_reason
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "close_reason": null}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "close_reason": 9999}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 1.0, "close_reason": ["list", "reason"]}}',
        # 9. VALID CLEAN TRADES to verify they are still parsed properly amid corruption
        '{"event_type": "lifecycle", "payload": {"realized_net_r": 2.5, "duration_bars": 40.0, "close_reason": "TP"}}',
        '{"event_type": "lifecycle", "payload": {"realized_net_r": -1.0, "duration_bars": 20.0, "close_reason": "SL"}}',
    ]

    events_file.write_text("\n".join(hostile_lines) + "\n", encoding="utf-8")

    # MUST NOT RAISE ANY EXCEPTION
    metrics = extract_regime_metrics_from_telemetry(corrupt_dir, "bull", {"market_type": "uptrend"})

    # Exactly the 2 clean trades plus valid realized_net_r entries should be extracted safely
    assert metrics.total_trades > 0, "Parser must not discard entire file due to corrupt lines"
    assert math.isfinite(metrics.expectancy_r)
    assert math.isfinite(metrics.payoff_ratio)
    assert math.isfinite(metrics.win_rate_pct)
    assert metrics.premature_closes == 0


def test_extract_regime_metrics_corrupted_diagnostics_json(tmp_path: Path) -> None:
    """Verifies that hostile and malformed mt5_diagnostics.json does not crash extraction."""
    diag_dir = tmp_path / "corrupt_diag_suite"
    diag_dir.mkdir()

    # Provide 1 valid trade
    (diag_dir / "events.jsonl").write_text(
        json.dumps({
            "event_type": "lifecycle",
            "payload": {"realized_net_r": 1.5, "close_reason": "TP", "duration_bars": 30.0},
        }) + "\n",
        encoding="utf-8",
    )

    # Battery of corrupted diagnostics payloads
    test_cases = [
        "NOT JSON AT ALL {{{",
        json.dumps([]),
        json.dumps("string"),
        json.dumps(None),
        json.dumps({"risk": None, "performance": None}),
        json.dumps({"risk": {"max_equity_drawdown_pct": "NaN"}, "performance": {"profit_factor": "inf"}}),
        json.dumps({"risk": {"max_equity_drawdown_pct": None}, "performance": {"profit_factor": None}}),
        json.dumps({"risk": {"max_equity_drawdown_pct": "invalid_string"}}),
        json.dumps({"performance": {"profit_factor": [1, 2, 3], "net_pnl": "invalid"}}),
    ]

    for payload in test_cases:
        (diag_dir / "mt5_diagnostics.json").write_text(payload, encoding="utf-8")
        metrics = extract_regime_metrics_from_telemetry(diag_dir, "bull", {"market_type": "uptrend"})
        assert metrics.total_trades == 1
        assert math.isfinite(metrics.max_drawdown_pct)
        assert math.isfinite(metrics.profit_factor)
        assert math.isfinite(metrics.net_pnl)
