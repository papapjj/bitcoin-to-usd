"""Proposed Test Suite & Mock Harnesses for Milestone 3: Automated Multi-Regime Gate Evaluation.

Covers:
- Area 1: Telemetry & Metrics Extraction Contract (events.jsonl & mt5_diagnostics.json parsing)
- Area 2: The 7 Gate Hurdle Criteria (WR, Payoff, E[R], Counter-Trend Bull/Bear, Timeout, Governor, No Hard Gate)
- Area 3: Multi-Regime Cross-Validation & Catastrophic Forgetting Elimination
- Area 4: Atomic Champion Promotion & Rollback Mechanics (crash safety, payload pruning, seed binding)
- Area 5: Boundary Conditions & Robustness (exact threshold margins, non-finite fail-closed, corrupted logs)

This test suite is deterministic, fast (<2s for full suite), self-contained, and requires NO MT5 GUI execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest

# ==============================================================================
# CANONICAL INTERFACES & DUAL-TRACK IMPORT ADAPTER
# ==============================================================================

from self_growing_trader.multi_regime_gate import (
    CANONICAL_BENCHMARK_WINDOWS,
    BenchmarkWindow,
    CheckpointPromotionEngine,
    DEFAULT_BENCHMARK_REGIMES,
    DEFAULT_GATE_HURDLES,
    GateHurdles,
    IsolatedMT5RunnerManager,
    MultiRegimeGateEvaluator,
    MultiRegimeResult,
    PolicyImmutabilityError,
    PromotionResult,
    RegimeMetrics,
    compute_policy_state_hash,
    evaluate_multi_regime_gate,
    extract_regime_metrics_from_telemetry,
    promote_champion_atomic,
    rollback_candidate_atomic,
)




# ==============================================================================
# DETERMINISTIC MOCK TELEMETRY HARNESS
# ==============================================================================

class MockTelemetryBuilder:
    """Ultra-fast, deterministic generator for events.jsonl and mt5_diagnostics.json."""

    def __init__(self, regime_type: str = "uptrend"):
        self.regime_type = regime_type
        self.trades: list[dict[str, Any]] = []
        self.policy_close_all_count = 0
        self.max_dd_pct = 4.2
        self.profit_factor = 1.65

    def add_trade(
        self,
        realized_net_r: float,
        close_reason: str = "TP",
        duration_bars: float = 40.0,
        action_index: int = 1,
    ) -> MockTelemetryBuilder:
        trade_event = {
            "event_type": "lifecycle",
            "payload": {
                "realized_net_r": float(realized_net_r),
                "close_reason": str(close_reason),
                "duration_bars": float(duration_bars),
                "growth_events": [
                    {
                        "type": "context_action_expectancy_credit",
                        "action_index": int(action_index),
                        "conditioned_entry_expectancy": {
                            "condition": {
                                "trend": "up" if self.regime_type == "uptrend" else ("down" if self.regime_type == "downtrend" else "neutral")
                            }
                        },
                    }
                ],
            },
        }
        self.trades.append(trade_event)
        if close_reason == "POLICY_CLOSE_ALL":
            self.policy_close_all_count += 1
        return self

    def add_synthetic_batch(
        self,
        wins: int,
        losses: int,
        avg_win_r: float = 2.0,
        avg_loss_r: float = 1.0,
        counter_trend_count: int = 0,
        timeout_count: int = 0,
        policy_close_all_count: int = 0,
    ) -> MockTelemetryBuilder:
        total = wins + losses
        # Assign entry actions
        # For uptrend: Long is 1, Short is 2 (counter-trend)
        # For downtrend: Short is 2, Long is 1 (counter-trend)
        actions = []
        for _ in range(counter_trend_count):
            actions.append(2 if self.regime_type == "uptrend" else 1)
        for _ in range(total - counter_trend_count):
            actions.append(1 if self.regime_type == "uptrend" else 2)

        # Durations & close reasons
        reasons = []
        durations = []
        for _ in range(policy_close_all_count):
            reasons.append("POLICY_CLOSE_ALL")
            durations.append(50.0)
        for _ in range(timeout_count):
            reasons.append("EXPERT")
            durations.append(120.0)

        # Wins
        for i in range(wins):
            reason = reasons.pop() if reasons else "TP"
            dur = durations.pop() if durations else 45.0
            act = actions.pop() if actions else 1
            self.add_trade(realized_net_r=avg_win_r, close_reason=reason, duration_bars=dur, action_index=act)

        # Losses
        for i in range(losses):
            reason = reasons.pop() if reasons else "SL"
            dur = durations.pop() if durations else 30.0
            act = actions.pop() if actions else 2
            self.add_trade(realized_net_r=-avg_loss_r, close_reason=reason, duration_bars=dur, action_index=act)

        return self

    def write_to_directory(self, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        events_file = target_dir / "events.jsonl"
        with open(events_file, "w", encoding="utf-8") as f:
            for t in self.trades:
                f.write(json.dumps(t) + "\n")

        # Compute diagnostics summary
        wins = [t["payload"]["realized_net_r"] for t in self.trades if t["payload"]["realized_net_r"] > 0]
        losses = [t["payload"]["realized_net_r"] for t in self.trades if t["payload"]["realized_net_r"] <= 0]
        total = len(self.trades)

        diag_content = {
            "status": {"execution": "finished"},
            "performance": {
                "closed_trades": total,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / max(1, total),
                "profit_factor": self.profit_factor,
                "payoff_ratio": (np.mean(wins) / abs(np.mean(losses))) if wins and losses else 2.0,
                "expectancy": float(np.mean([t["payload"]["realized_net_r"] for t in self.trades])) if total > 0 else 0.0,
                "close_reasons": {
                    "TP": sum(1 for t in self.trades if t["payload"]["close_reason"] == "TP"),
                    "SL": sum(1 for t in self.trades if t["payload"]["close_reason"] == "SL"),
                    "EXPERT": sum(1 for t in self.trades if t["payload"]["close_reason"] == "EXPERT"),
                    "POLICY_CLOSE_ALL": self.policy_close_all_count,
                },
            },
            "risk": {"max_equity_drawdown_pct": self.max_dd_pct},
            "policy": {"hold_share": 0.72, "normalized_action_entropy": 0.44},
        }

        diag_file = target_dir / "mt5_diagnostics.json"
        diag_file.write_text(json.dumps(diag_content, indent=2), encoding="utf-8")
        return target_dir


# ==============================================================================
# AREA 1: TELEMETRY & METRICS EXTRACTION CONTRACT
# ==============================================================================

def test_extract_metrics_from_mock_events_jsonl(tmp_path: Path) -> None:
    """Verifies that extract_regime_metrics_from_telemetry accurately computes WR, E[R], and Payoff."""
    builder = MockTelemetryBuilder("uptrend")
    builder.add_synthetic_batch(
        wins=15,
        losses=10,
        avg_win_r=2.0,
        avg_loss_r=1.0,
        counter_trend_count=3,
        timeout_count=2,
    )
    r_dir = builder.write_to_directory(tmp_path / "uptrend_test")

    cfg = {"market_type": "uptrend", "from_date": "2025.01.14", "to_date": "2025.01.18"}
    metrics = extract_regime_metrics_from_telemetry(r_dir, "bull", cfg)

    assert metrics.regime_id == "bull"
    assert metrics.total_trades == 25
    assert metrics.wins == 15
    assert metrics.losses == 10
    assert metrics.win_rate_pct == pytest.approx(60.0)
    # E[R] = (15*2.0 - 10*1.0) / 25 = 20/25 = +0.80R
    assert metrics.expectancy_r == pytest.approx(0.80)
    assert metrics.payoff_ratio == pytest.approx(2.0)
    assert metrics.counter_trend_entries == 3
    assert metrics.counter_trend_rate_pct == pytest.approx(12.0)  # 3/25
    assert metrics.timeout_closes == 2
    assert metrics.timeout_close_rate_pct == pytest.approx(8.0)  # 2/25
    assert metrics.premature_closes == 0


def test_counter_trend_rate_calculation_bull_and_bear(tmp_path: Path) -> None:
    """Verifies counter-trend trade identification in Bull (Short) and Bear (Long)."""
    # 1. Bull: 6 short entries out of 20 = 30.0%
    bull_dir = tmp_path / "bull"
    b_builder = MockTelemetryBuilder("uptrend")
    b_builder.add_synthetic_batch(wins=12, losses=8, counter_trend_count=6)
    b_builder.write_to_directory(bull_dir)

    bull_metrics = extract_regime_metrics_from_telemetry(
        bull_dir, "bull", {"market_type": "uptrend"}
    )
    assert bull_metrics.counter_trend_rate_pct == pytest.approx(30.0)

    # 2. Bear: 5 long entries out of 25 = 20.0%
    bear_dir = tmp_path / "bear"
    d_builder = MockTelemetryBuilder("downtrend")
    d_builder.add_synthetic_batch(wins=15, losses=10, counter_trend_count=5)
    d_builder.write_to_directory(bear_dir)

    bear_metrics = extract_regime_metrics_from_telemetry(
        bear_dir, "bear", {"market_type": "downtrend"}
    )
    assert bear_metrics.counter_trend_rate_pct == pytest.approx(20.0)


def test_timeout_close_rate_calculation(tmp_path: Path) -> None:
    """Verifies timeout close rate calculation for duration_bars >= 120 or close_reason EXPERT."""
    t_dir = tmp_path / "timeout_test"
    builder = MockTelemetryBuilder("sideway")
    # 3 timeout closes out of 30 trades = 10.0%
    builder.add_synthetic_batch(wins=15, losses=15, timeout_count=3)
    builder.write_to_directory(t_dir)

    metrics = extract_regime_metrics_from_telemetry(
        t_dir, "consolidation", {"market_type": "sideway"}
    )
    assert metrics.timeout_closes == 3
    assert metrics.timeout_close_rate_pct == pytest.approx(10.0)


# ==============================================================================
# AREA 2: THE 7 GATE HURDLE CRITERIA & REJECTION SEMANTICS
# ==============================================================================

def _make_passing_regimes() -> dict[str, RegimeMetrics]:
    """Helper returning a standard candidate meeting all 7 hurdles across 3 regimes."""
    return {
        "bull": RegimeMetrics(
            regime_id="bull",
            from_date="2025.01.14",
            to_date="2025.01.18",
            market_type="uptrend",
            total_trades=20,
            wins=11,
            losses=9,
            win_rate_pct=55.0,
            expectancy_r=0.25,
            payoff_ratio=1.70,
            max_drawdown_pct=3.5,
            timeout_closes=1,
            timeout_close_rate_pct=5.0,
            counter_trend_entries=3,
            counter_trend_rate_pct=15.0,
            premature_closes=0,
        ),
        "bear": RegimeMetrics(
            regime_id="bear",
            from_date="2025.01.06",
            to_date="2025.01.10",
            market_type="downtrend",
            total_trades=20,
            wins=10,
            losses=10,
            win_rate_pct=50.0,
            expectancy_r=0.15,
            payoff_ratio=1.65,
            max_drawdown_pct=4.2,
            timeout_closes=2,
            timeout_close_rate_pct=10.0,
            counter_trend_entries=4,
            counter_trend_rate_pct=20.0,
            premature_closes=0,
        ),
        "consolidation": RegimeMetrics(
            regime_id="consolidation",
            from_date="2025.01.24",
            to_date="2025.01.28",
            market_type="sideway",
            total_trades=20,
            wins=10,
            losses=10,
            win_rate_pct=50.0,
            expectancy_r=0.08,
            payoff_ratio=1.55,
            max_drawdown_pct=2.8,
            timeout_closes=2,
            timeout_close_rate_pct=10.0,
            counter_trend_entries=0,
            counter_trend_rate_pct=0.0,
            premature_closes=0,
        ),
    }


def test_gate_passes_when_all_hurdles_met() -> None:
    """Verifies that candidate meeting all 7 hurdles passes with zero rejection reasons."""
    regimes = _make_passing_regimes()
    passed, reasons, agg = evaluate_multi_regime_gate(regimes)

    assert passed is True
    assert reasons == []
    assert agg["win_rate_pct"] == pytest.approx(51.67, abs=0.1)
    assert agg["expectancy_r"] >= 0.05
    assert agg["payoff_ratio"] >= 1.50
    assert agg["total_premature_closes"] == 0


def test_gate_fails_on_counter_trend_in_bull() -> None:
    """Hurdle Failure: Short trades in Uptrend > 30% (e.g. 35.0%) -> Rejected."""
    regimes = _make_passing_regimes()
    regimes["bull"].counter_trend_rate_pct = 35.0

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("counter_trend_rate_uptrend_exceeded" in r for r in reasons)


def test_gate_fails_on_counter_trend_in_bear() -> None:
    """Hurdle Failure: Long trades in Downtrend > 30% (e.g. 32.5%) -> Rejected."""
    regimes = _make_passing_regimes()
    regimes["bear"].counter_trend_rate_pct = 32.5

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("counter_trend_rate_downtrend_exceeded" in r for r in reasons)


def test_gate_fails_on_timeout_close_rate() -> None:
    """Hurdle Failure: Timeout closes >= 15% (e.g. 18.0% in consolidation) -> Rejected."""
    regimes = _make_passing_regimes()
    regimes["consolidation"].timeout_close_rate_pct = 18.0

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("timeout_close_rate_exceeded" in r for r in reasons)


def test_gate_fails_on_low_win_rate() -> None:
    """Hurdle Failure: Aggregate Win Rate < 45.0% (e.g. 42.0%) -> Rejected."""
    regimes = _make_passing_regimes()
    # Reduce wins: 8 wins out of 20 in each regime -> 24/60 = 40.0%
    for r in regimes.values():
        r.wins = 8
        r.losses = 12
        r.win_rate_pct = 40.0

    passed, reasons, agg = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("win_rate_below_hurdle" in r for r in reasons)


def test_gate_fails_on_low_payoff_ratio() -> None:
    """Hurdle Failure: Aggregate Payoff Ratio < 1.50R (e.g. 1.30R) -> Rejected."""
    regimes = _make_passing_regimes()
    for r in regimes.values():
        r.payoff_ratio = 1.30

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("payoff_ratio_below_hurdle" in r for r in reasons)


def test_gate_fails_on_low_expectancy() -> None:
    """Hurdle Failure: Aggregate Expectancy < +0.05R (e.g. +0.02R) -> Rejected."""
    regimes = _make_passing_regimes()
    for r in regimes.values():
        r.expectancy_r = 0.02

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("expectancy_below_hurdle" in r for r in reasons)


def test_gate_fails_on_governor_policy_close_all() -> None:
    """Hurdle Failure: Even 1 premature close by POLICY_CLOSE_ALL -> Immediate Rejection."""
    regimes = _make_passing_regimes()
    regimes["consolidation"].premature_closes = 1

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("premature_closes_violation" in r for r in reasons)


def test_gate_fails_on_hard_direction_gate_flag() -> None:
    """Hurdle Failure: context_direction_entry_gate_enabled is True -> Forbidden."""
    regimes = _make_passing_regimes()
    bad_cfg = {"learning": {"context_direction_entry_gate_enabled": True}}

    passed, reasons, _ = evaluate_multi_regime_gate(regimes, candidate_config=bad_cfg)
    assert passed is False
    assert any("hard_direction_gate_forbidden" in r for r in reasons)


def test_gate_fails_on_insufficient_sample_count() -> None:
    """Hurdle Failure: Total trades < 30 -> Rejected."""
    regimes = {
        "bull": RegimeMetrics("bull", "", "", "uptrend", total_trades=5, wins=3, losses=2),
        "bear": RegimeMetrics("bear", "", "", "downtrend", total_trades=4, wins=2, losses=2),
    }
    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("total_trades_insufficient" in r for r in reasons)


def test_gate_fails_on_single_regime_insufficient_trades() -> None:
    """Verifies that candidate with insufficient trades in one regime fails closed."""
    regimes = _make_passing_regimes()
    regimes["bear"].total_trades = 0
    regimes["bear"].wins = 0
    regimes["bear"].losses = 0

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("regime_trades_insufficient:bear: 0 < 1" in r for r in reasons)


# ==============================================================================
# AREA 3: MULTI-REGIME CROSS-VALIDATION & CATASTROPHIC FORGETTING
# ==============================================================================

def test_catastrophic_forgetting_detection() -> None:
    """Empirical proof of Catastrophic Forgetting detection.

    A candidate trained on a strong uptrend achieves spectacular numbers in Bull,
    but collapses in Bear. Single-window evaluation would promote it; Multi-Regime Gate
    strictly catches the collapse and REJECTS promotion.
    """
    asymmetric_regimes = {
        "bull": RegimeMetrics(
            regime_id="bull",
            from_date="2025.01.14",
            to_date="2025.01.18",
            market_type="uptrend",
            total_trades=30,
            wins=24,
            losses=6,
            win_rate_pct=80.0,
            expectancy_r=0.92,
            payoff_ratio=2.50,
            max_drawdown_pct=2.1,
            timeout_closes=1,
            timeout_close_rate_pct=3.33,
            counter_trend_entries=2,
            counter_trend_rate_pct=6.67,  # <30%
            premature_closes=0,
        ),
        "bear": RegimeMetrics(
            regime_id="bear",
            from_date="2025.01.06",
            to_date="2025.01.10",
            market_type="downtrend",
            total_trades=30,
            wins=6,
            losses=24,
            win_rate_pct=20.0,  # collapsed
            expectancy_r=-0.45,  # collapsed
            payoff_ratio=0.85,  # collapsed
            max_drawdown_pct=9.5,
            timeout_closes=2,
            timeout_close_rate_pct=6.67,
            counter_trend_entries=21,
            counter_trend_rate_pct=70.0,  # forgot how to trade bear!
            premature_closes=0,
        ),
        "consolidation": RegimeMetrics(
            regime_id="consolidation",
            from_date="2025.01.24",
            to_date="2025.01.28",
            market_type="sideway",
            total_trades=20,
            wins=9,
            losses=11,
            win_rate_pct=45.0,
            expectancy_r=0.06,
            payoff_ratio=1.52,
            max_drawdown_pct=4.0,
            timeout_closes=2,
            timeout_close_rate_pct=10.0,
            counter_trend_entries=0,
            counter_trend_rate_pct=0.0,
            premature_closes=0,
        ),
    }

    passed, reasons, _ = evaluate_multi_regime_gate(asymmetric_regimes)
    assert passed is False
    assert any("counter_trend_rate_downtrend_exceeded" in r for r in reasons)


def test_frozen_policy_state_hash_immutability(tmp_path: Path) -> None:
    """Verifies that MultiRegimeGateEvaluator strictly enforces policy frozen immutability."""
    cand_path = tmp_path / "candidate.pkl"
    cand_path.write_bytes(b"initial_policy_weights_v193_seed_42")

    evaluator = MultiRegimeGateEvaluator()

    # Define a malicious evaluator that mutates the checkpoint during evaluation
    def mutating_evaluator(r_id: str, cfg: dict[str, str], out_dir: Path) -> Path:
        cand_path.write_bytes(b"mutated_weights_violation!")
        b = MockTelemetryBuilder("uptrend")
        b.add_synthetic_batch(wins=15, losses=5)
        return b.write_to_directory(out_dir)

    with pytest.raises(PolicyImmutabilityError, match="mutated during evaluation"):
        evaluator.evaluate_candidate(
            candidate_path=cand_path,
            output_dir=tmp_path / "output",
            evaluator_fn=mutating_evaluator,
        )


# ==============================================================================
# AREA 4: ATOMIC CHAMPION PROMOTION & ROLLBACK MECHANICS
# ==============================================================================

def test_atomic_promotion_updates_champion_and_active_seed(tmp_path: Path) -> None:
    """Verifies that passing candidate is atomically promoted and bound to active_seed."""
    cand_path = tmp_path / "candidate.pkl"
    cand_data = b"promoted_champion_weights_payload_12345"
    cand_path.write_bytes(cand_data)

    reg_path = tmp_path / "candidate.registry.json"
    reg_path.write_text(json.dumps({"round": 5, "hash": "abc"}), encoding="utf-8")

    out_dir = tmp_path / "promotions"
    active_seed = tmp_path / "active_seed.pkl"
    active_seed.write_bytes(b"old_baseline_seed")

    promoted, seed = promote_champion_atomic(
        candidate_path=cand_path,
        output_dir=out_dir,
        round_id=5,
        active_seed_path=active_seed,
    )

    assert promoted.exists()
    assert promoted.read_bytes() == cand_data
    # Check canonical best_promoted_checkpoint.pkl
    assert (out_dir / "best_promoted_checkpoint.pkl").read_bytes() == cand_data
    # Check round-specific champion artifact
    assert (out_dir / "champion_phase3_round_05.pkl").read_bytes() == cand_data
    # Check sidecar registry
    assert (out_dir / "best_promoted_checkpoint.registry.json").exists()
    # Check active_seed is updated
    assert active_seed.read_bytes() == cand_data


def test_rollback_preserves_champion_on_gate_rejection(tmp_path: Path) -> None:
    """Verifies that failing candidate is pruned and active_seed rolls back to prior champion."""
    champion_seed = tmp_path / "champion_phase3_round_02.pkl"
    champ_data = b"untouched_champion_seed_payload"
    champion_seed.write_bytes(champ_data)

    cand_path = tmp_path / "candidate_round_03.pkl"
    cand_path.write_bytes(b"failed_divergent_candidate")

    # Rollback
    restored = rollback_candidate_atomic(
        candidate_path=cand_path,
        champion_seed_path=champion_seed,
        prune_candidate_payload=True,
    )

    assert restored == champion_seed
    assert champion_seed.exists()
    assert champion_seed.read_bytes() == champ_data
    # Candidate binary payload must be pruned
    assert not cand_path.exists()


def test_prune_unpromoted_candidate_payload_preserves_telemetry(tmp_path: Path) -> None:
    """Verifies that pruning unpromoted candidate removes .pkl/.tmp but preserves JSON telemetry."""
    round_dir = tmp_path / "round_07"
    round_dir.mkdir(parents=True)

    cand_pkl = round_dir / "candidate.pkl"
    cand_pkl.write_bytes(b"large_model_binary_80mb")
    cand_tmp = round_dir / "candidate.tmp"
    cand_tmp.write_bytes(b"scratch")

    events_file = round_dir / "events.jsonl"
    events_file.write_text('{"event_type": "lifecycle"}\n', encoding="utf-8")
    diag_file = round_dir / "mt5_diagnostics.json"
    diag_file.write_text('{"status": "finished"}', encoding="utf-8")
    manifest_file = round_dir / "manifest.json"
    manifest_file.write_text('{"round": 7}', encoding="utf-8")

    from self_growing_trader.auto_rl_engine import prune_unpromoted_checkpoints

    pruned = prune_unpromoted_checkpoints(round_dir)
    assert cand_pkl in pruned
    assert not cand_pkl.exists()
    assert not cand_tmp.exists()

    # Telemetry manifests must remain intact!
    assert events_file.exists()
    assert diag_file.exists()
    assert manifest_file.exists()


def test_atomic_rename_crash_safety_and_file_integrity(tmp_path: Path) -> None:
    """Verifies that atomic write via temporary file prevents file corruption/truncation."""
    dest = tmp_path / "atomic_target.pkl"
    dest.write_bytes(b"original_valid_data")

    # Simulate an aborted atomic write where .tmp exists
    stale_tmp = dest.with_suffix(".tmp")
    stale_tmp.write_bytes(b"half_written_corrupted_data")

    # Original target remains intact
    assert dest.read_bytes() == b"original_valid_data"

    # New atomic promotion replaces safely
    new_cand = tmp_path / "new_candidate.pkl"
    new_cand.write_bytes(b"new_valid_champion_payload")
    promote_champion_atomic(new_cand, tmp_path, champion_name="atomic_target.pkl")

    assert dest.read_bytes() == b"new_valid_champion_payload"
    assert not stale_tmp.exists()


# ==============================================================================
# AREA 5: BOUNDARY CONDITIONS & ROBUSTNESS
# ==============================================================================

@pytest.mark.parametrize(
    "wr_bull,wr_bear,wr_sideway,expected_pass",
    [
        (45.0, 45.0, 45.0, True),   # Exact boundary at 45.0% -> PASS
        (44.9, 45.0, 45.0, False),  # 44.96% aggregate < 45.0% -> FAIL
    ],
)
def test_exact_win_rate_boundary(wr_bull: float, wr_bear: float, wr_sideway: float, expected_pass: bool) -> None:
    regimes = _make_passing_regimes()
    regimes["bull"].win_rate_pct = wr_bull
    regimes["bull"].wins = int(wr_bull * 0.20)
    regimes["bull"].losses = 20 - regimes["bull"].wins

    regimes["bear"].win_rate_pct = wr_bear
    regimes["bear"].wins = int(wr_bear * 0.20)
    regimes["bear"].losses = 20 - regimes["bear"].wins

    regimes["consolidation"].win_rate_pct = wr_sideway
    regimes["consolidation"].wins = int(wr_sideway * 0.20)
    regimes["consolidation"].losses = 20 - regimes["consolidation"].wins

    passed, _, _ = evaluate_multi_regime_gate(regimes)
    assert passed is expected_pass


@pytest.mark.parametrize(
    "counter_pct,expected_pass",
    [
        (30.0, True),   # Exact boundary at 30.0% -> PASS
        (30.1, False),  # 30.1% > 30.0% -> FAIL
    ],
)
def test_exact_counter_trend_boundary(counter_pct: float, expected_pass: bool) -> None:
    regimes = _make_passing_regimes()
    regimes["bull"].counter_trend_rate_pct = counter_pct
    passed, _, _ = evaluate_multi_regime_gate(regimes)
    assert passed is expected_pass


@pytest.mark.parametrize(
    "timeout_pct,expected_pass",
    [
        (14.9, True),   # < 15.0% -> PASS
        (15.0, False),  # >= 15.0% -> FAIL
    ],
)
def test_exact_timeout_boundary(timeout_pct: float, expected_pass: bool) -> None:
    regimes = _make_passing_regimes()
    regimes["consolidation"].timeout_close_rate_pct = timeout_pct
    passed, _, _ = evaluate_multi_regime_gate(regimes)
    assert passed is expected_pass


def test_nan_and_inf_metric_fail_closed() -> None:
    """Verifies that non-finite metrics (NaN, Inf) fail closed without unhandled exceptions."""
    regimes = _make_passing_regimes()
    regimes["bull"].expectancy_r = float("nan")

    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("invalid_metric_value" in r for r in reasons)

    regimes["bull"].expectancy_r = 0.25
    regimes["bear"].max_drawdown_pct = float("inf")
    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("invalid_metric_value" in r for r in reasons)


def test_missing_events_or_empty_diagnostics_fails_closed(tmp_path: Path) -> None:
    """Verifies that an empty directory or corrupted JSON fails gracefully with 0 trades."""
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()

    metrics = extract_regime_metrics_from_telemetry(
        empty_dir, "bull", {"market_type": "uptrend"}
    )
    assert metrics.total_trades == 0
    assert metrics.win_rate_pct == 0.0
    assert metrics.expectancy_r == 0.0

    regimes = {"bull": metrics}
    passed, reasons, _ = evaluate_multi_regime_gate(regimes)
    assert passed is False
    assert any("total_trades_insufficient" in r for r in reasons)


def test_full_pipeline_multi_regime_evaluator_end_to_end(tmp_path: Path) -> None:
    """E2E simulation of MultiRegimeGateEvaluator across Bull, Bear, and Consolidation windows."""
    cand_path = tmp_path / "candidate_round_06.pkl"
    cand_path.write_bytes(b"mock_v193_model_checkpoint_weights")
    active_seed = tmp_path / "active_seed.pkl"
    active_seed.write_bytes(b"baseline_seed")

    evaluator = MultiRegimeGateEvaluator()

    # Deterministic mock evaluator function
    def synthetic_evaluator(r_id: str, cfg: dict[str, str], out_dir: Path) -> Path:
        m_type = cfg.get("market_type", "sideway")
        b = MockTelemetryBuilder(m_type)
        if m_type == "uptrend":
            # 16 wins, 8 losses (WR 66.7%), 3 counter-trend shorts (12.5%), 1 timeout (4.2%)
            b.add_synthetic_batch(wins=16, losses=8, avg_win_r=2.0, avg_loss_r=1.0, counter_trend_count=3, timeout_count=1)
        elif m_type == "downtrend":
            # 12 wins, 12 losses (WR 50.0%), 4 counter-trend longs (16.7%), 2 timeouts (8.3%)
            b.add_synthetic_batch(wins=12, losses=12, avg_win_r=2.0, avg_loss_r=1.0, counter_trend_count=4, timeout_count=2)
        else:
            # 11 wins, 11 losses (WR 50.0%), 0 counter-trend, 2 timeouts (9.1%)
            b.add_synthetic_batch(wins=11, losses=11, avg_win_r=1.5, avg_loss_r=1.0, counter_trend_count=0, timeout_count=2)
        return b.write_to_directory(out_dir)

    result = evaluator.evaluate_candidate(
        candidate_path=cand_path,
        output_dir=tmp_path / "multi_regime_output",
        evaluator_fn=synthetic_evaluator,
        active_seed_path=active_seed,
        round_id=6,
    )

    assert result.passed_gate is True
    assert result.promoted is True
    assert result.reasons == []
    assert len(result.regimes) == 3
    assert result.aggregate_metrics["total_trades"] == 70
    assert result.aggregate_metrics["win_rate_pct"] >= 45.0
    assert result.aggregate_metrics["expectancy_r"] >= 0.05
    assert result.aggregate_metrics["payoff_ratio"] >= 1.50
    assert result.promoted_checkpoint_path is not None
    assert Path(result.promoted_checkpoint_path).exists()
    assert active_seed.read_bytes() == b"mock_v193_model_checkpoint_weights"


def test_canonical_benchmark_windows_and_gate_hurdles_contracts() -> None:
    """Verifies that CANONICAL_BENCHMARK_WINDOWS and GateHurdles adhere to project specifications."""
    assert len(CANONICAL_BENCHMARK_WINDOWS) == 3
    names = [w.name for w in CANONICAL_BENCHMARK_WINDOWS]
    assert names == ["bull", "bear", "consolidation"]

    bull = CANONICAL_BENCHMARK_WINDOWS[0]
    assert bull.from_date == "2025.01.14" and bull.to_date == "2025.01.18"
    assert bull.regime == "uptrend" and bull.market_type == "uptrend"

    bear = CANONICAL_BENCHMARK_WINDOWS[1]
    assert bear.from_date == "2025.01.06" and bear.to_date == "2025.01.10"
    assert bear.regime == "downtrend" and bear.market_type == "downtrend"

    consolidation = CANONICAL_BENCHMARK_WINDOWS[2]
    assert consolidation.from_date == "2025.01.24" and consolidation.to_date == "2025.01.28"
    assert consolidation.regime == "sideway" and consolidation.market_type == "sideway"

    hurdles = GateHurdles()
    assert hurdles.min_win_rate == 45.0
    assert hurdles.min_payoff_ratio == 1.50
    assert hurdles.min_expectancy == 0.05
    assert hurdles.max_counter_trend_rate == 30.0
    assert hurdles.max_timeout_close_rate == 15.0
    assert hurdles.max_premature_closes == 0
    assert hurdles.require_natural_rl is True


def test_checkpoint_promotion_engine_atomic_and_rollback(tmp_path: Path) -> None:
    """Verifies CheckpointPromotionEngine handles weight immutability, atomic promotion, and rollback."""
    loop_dir = tmp_path / "phase3_loop"
    engine = CheckpointPromotionEngine(checkpoint_dir=loop_dir, max_rollback_backups=2)

    # 1. Test weight immutability helper
    candidate_data = b"immutable_policy_state_data_seed"
    cand_file = tmp_path / "candidate_eval.pkl"
    cand_file.write_bytes(candidate_data)

    def dummy_eval_fn(policy_clone: Any) -> str:
        return "eval_complete"

    res, state_hash = engine.assert_weight_immutability(cand_file, dummy_eval_fn)
    assert res == "eval_complete"
    assert state_hash == compute_policy_state_hash(cand_file)

    # 2. Test atomic promotion
    gate_summary = {"passed": True, "win_rate": 55.0, "expectancy": 0.25}
    promo_res = engine.atomic_promote(
        candidate_source=cand_file,
        round_index=1,
        gate_summary=gate_summary,
        candidate_hash=state_hash,
    )
    assert promo_res.promoted is True
    assert promo_res.champion_path.exists()
    assert promo_res.champion_path.read_bytes() == candidate_data
    assert engine.canonical_champion_path.exists()
    assert engine.canonical_champion_path.read_bytes() == candidate_data
    assert (engine.canonical_champion_path.with_suffix(".registry.json")).exists()

    # 3. Test rollback on failure
    bad_cand = tmp_path / "candidate_failed.pkl"
    bad_cand.write_bytes(b"bad_weights")
    round_dir = tmp_path / "round_02"
    round_dir.mkdir(parents=True)
    bad_round_cand = round_dir / "candidate.pkl"
    bad_round_cand.write_bytes(b"bad_weights_in_round")

    rollback_res = engine.rollback_on_failure(
        candidate_path=bad_round_cand,
        active_seed_path=engine.canonical_champion_path,
        round_dir=round_dir,
        failure_reasons=["win_rate_below_hurdle: 40.0% < 45.0%"],
    )
    assert rollback_res.promoted is False
    assert rollback_res.champion_path == engine.canonical_champion_path
    assert not bad_round_cand.exists()  # Pruned


def test_isolated_mt5_runner_manager_lifecycle(tmp_path: Path) -> None:
    """Verifies IsolatedMT5RunnerManager creates collision-free directories and cleans up."""
    base_runner = tmp_path / "base_mt5_runner"
    base_runner.mkdir()
    (base_runner / "terminal64.exe").write_bytes(b"dummy_exe")
    (base_runner / "MQL5" / "Experts").mkdir(parents=True)
    (base_runner / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").write_text("// EA source", encoding="utf-8")
    (base_runner / "Config").mkdir()
    (base_runner / "Config" / "common.ini").write_text("[Common]", encoding="utf-8")

    clone_dir = IsolatedMT5RunnerManager.create_isolated_runner(base_runner, "bull")
    assert clone_dir.exists()
    assert "runner_gate_bull_" in clone_dir.name
    assert (clone_dir / "terminal64.exe").exists()
    assert (clone_dir / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").exists()
    assert (clone_dir / "Config" / "common.ini").exists()

    IsolatedMT5RunnerManager.cleanup_isolated_runner(clone_dir)
    assert not clone_dir.exists()


def test_generate_report_markdown(tmp_path: Path) -> None:
    """Verifies Markdown report generation from MultiRegimeResult."""
    regimes = {
        "bull": RegimeMetrics("bull", "2025.01.14", "2025.01.18", "uptrend", total_trades=20, wins=12, losses=8, win_rate_pct=60.0, expectancy_r=0.40, payoff_ratio=1.8),
    }
    res = MultiRegimeResult(
        passed_gate=True,
        promoted=True,
        regimes=regimes,
        aggregate_metrics={"total_trades": 20, "win_rate_pct": 60.0, "payoff_ratio": 1.8, "expectancy_r": 0.40, "total_premature_closes": 0},
        candidate_path="candidate.pkl",
        evaluated_at="2026-10-19T22:00:00",
        candidate_hash_before="abc12345",
        candidate_hash_after="abc12345",
    )
    evaluator = MultiRegimeGateEvaluator()
    report_file = tmp_path / "report.md"
    report_md = evaluator.generate_report(res, output_path=report_file)

    assert "Multi-Regime Gate Evaluation Report" in report_md
    assert "PASSED & PROMOTED" in report_md
    assert "bull" in report_md
    assert report_file.exists()

