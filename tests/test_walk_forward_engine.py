"""Unit tests for Walk-Forward Multi-Cycle Orchestration Engine & Adaptive Tuning Module.

Verifies:
1. CLI Argument Parsing and Defaults.
2. Market Cycle Definitions and Date Boundaries (Cycles 1, 2, 3).
3. Pure Headless MT5 INI Generation and Flag Verification (Visual=0, timeout=3600, max_wall=0).
4. Telemetry Ingestion and Metric Extraction (mt5_diagnostics.json, events.jsonl).
5. Acceptance Criteria Evaluation (Net Profit > 0, PF >= 1.15, Max DD < 15%, Trades >= 20, zero Simplex).
6. Adaptive Tuning on Loss Engine (R2): Root-cause diagnosis, dynamic knob transitions, profile generation.
7. Sequential Checkpoint Lineage & Rollback Mechanics on Loss.
8. Numerical Integrity and Simplex Safety Defense.
9. Candidate Artifact Manifest Generation and Multi-Cycle Certification.
10. Full Engine Orchestration with simulated runner.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pytest
import yaml

from self_growing_trader.walk_forward_engine import (
    DEFAULT_BASE_PROFILE,
    DEFAULT_CYCLES,
    DEFAULT_INITIAL_SEED,
    DEFAULT_MAX_ADAPTIVE_RETRIES,
    DEFAULT_MAX_WALL_SEC,
    DEFAULT_RUNNER_SOURCE,
    DEFAULT_SOURCE_CONFIG,
    DEFAULT_TIMEOUT_SEC,
    AcceptanceVerdict,
    AdaptiveTuningDiagnosis,
    CycleMetrics,
    WalkForwardCycleConfig,
    WalkForwardEngine,
    build_parser,

    build_tester_ini_content,
    compute_adaptive_tuning_overrides,
    diagnose_failure,
    evaluate_acceptance_criteria,
    extract_cycle_metrics,
    generate_adapted_profile_yaml,
    generate_walk_forward_manifest,
    sha256_file,
    verify_headless_configuration,
)


# ---------------------------------------------------------------------------
# 1. Argument Parsing & CLI Validation Tests
# ---------------------------------------------------------------------------

def test_argument_parsing_single_cycle():
    """Verify CLI parser parses single cycle flags correctly."""
    parser = build_parser()
    args = parser.parse_args(["--cycle", "1", "--timeout-sec", "3600", "--max-wall-sec", "0"])
    assert args.cycle == 1
    assert not args.all_cycles
    assert args.timeout_sec == 3600
    assert args.max_wall_sec == 0
    assert args.seed_checkpoint == DEFAULT_INITIAL_SEED
    assert args.output_dir == Path("outputs/walk_forward_multi_cycle")


def test_argument_parsing_all_cycles():
    """Verify CLI parser parses --all-cycles flag."""
    parser = build_parser()
    args = parser.parse_args(["--all-cycles", "--dry-run"])
    assert args.all_cycles
    assert args.cycle is None
    assert args.dry_run


def test_argument_parsing_requires_mode():
    """Verify CLI parser requires either --cycle or --all-cycles."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_argument_parsing_rejects_invalid_cycle():
    """Verify CLI parser rejects cycles outside {1, 2, 3}."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--cycle", "4"])


# ---------------------------------------------------------------------------
# 2. Cycle Definitions and Date Boundaries
# ---------------------------------------------------------------------------

def test_cycle_definitions_and_date_boundaries():
    """Verify that all 3 walk-forward cycles have exact specified date boundaries."""
    assert len(DEFAULT_CYCLES) == 3

    # Cycle 1: Q1/2024
    c1 = DEFAULT_CYCLES[1]
    assert c1.cycle_id == 1
    assert c1.from_date == "2024.01.08"
    assert c1.to_date == "2024.03.29"
    assert "Q1/2024" in c1.name

    # Cycle 2: Q3/2024
    c2 = DEFAULT_CYCLES[2]
    assert c2.cycle_id == 2
    assert c2.from_date == "2024.07.01"
    assert c2.to_date == "2024.09.30"
    assert "Q3/2024" in c2.name

    # Cycle 3: Q1/2025
    c3 = DEFAULT_CYCLES[3]
    assert c3.cycle_id == 3
    assert c3.from_date == "2025.01.06"
    assert c3.to_date == "2025.03.31"
    assert "Q1/2025" in c3.name


# ---------------------------------------------------------------------------
# 3. Headless MT5 Configuration & INI Generation Tests
# ---------------------------------------------------------------------------

def test_tester_ini_generation_pure_headless():
    """Verify that build_tester_ini_content generates strict Visual=0 and ShutdownTerminal=1."""
    ini_content = build_tester_ini_content(
        from_date="2024.01.08",
        to_date="2024.03.29",
        symbol="BTCUSDm",
        visual=False,
        shutdown_terminal=True,
    )

    assert "Visual=0" in ini_content
    assert "ShutdownTerminal=1" in ini_content
    assert "FromDate=2024.01.08" in ini_content
    assert "ToDate=2024.03.29" in ini_content
    assert "Symbol=BTCUSDm" in ini_content


def test_verify_headless_configuration_valid():
    """Verify that verify_headless_configuration accepts compliant command args."""
    valid_cmd = [
        "python", "candidate.py",
        "--timeout-sec", "3600",
        "--max-wall-sec", "0",
    ]
    ok, violations = verify_headless_configuration(valid_cmd)
    assert ok
    assert len(violations) == 0


def test_verify_headless_configuration_detects_violations():
    """Verify that verify_headless_configuration flags violations like --mt5-visual."""
    invalid_cmd = [
        "python", "candidate.py",
        "--mt5-visual",
        "--timeout-sec", "300",  # too short
        "--max-wall-sec", "120",  # premature kill
    ]
    ok, violations = verify_headless_configuration(invalid_cmd)
    assert not ok
    assert any("--mt5-visual" in v for v in violations)
    assert any("--timeout-sec" in v for v in violations)
    assert any("--max-wall-sec" in v for v in violations)


# ---------------------------------------------------------------------------
# 4. Telemetry Ingestion and Metric Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_cycle_metrics_from_diagnostics_and_events(tmp_path: Path):
    """Verify parsing and extraction of metrics from mt5_diagnostics.json and events.jsonl."""
    # Write sample mt5_diagnostics.json
    diag_data = {
        "schema_version": 1,
        "performance": {
            "net_pnl": 2450.75,
            "profit_factor": 1.62,
            "closed_trades": 35,
            "win_rate": 0.60,
            "payoff_ratio": 1.75,
            "expectancy_r": 0.18,
            "close_reasons": {"TAKE_PROFIT": 21, "STOP_LOSS": 14},
        },
        "risk": {
            "max_equity_drawdown_pct": 7.8,
        },
        "policy": {
            "hold_share": 0.75,
        },
    }
    (tmp_path / "mt5_diagnostics.json").write_text(json.dumps(diag_data), encoding="utf-8")

    # Write sample events.jsonl
    with open(tmp_path / "events.jsonl", "w", encoding="utf-8") as f:
        for i in range(35):
            f.write(json.dumps({
                "event_type": "lifecycle",
                "payload": {
                    "realized_net_r": 0.30 if i % 2 == 0 else -0.15,
                    "close_reason": "TAKE_PROFIT" if i % 2 == 0 else "STOP_LOSS",
                },
            }) + "\n")

    metrics = extract_cycle_metrics(tmp_path, cycle_id=1)
    assert metrics.cycle_id == 1
    assert metrics.net_pnl == 2450.75
    assert metrics.profit_factor == 1.62
    assert metrics.max_drawdown_pct == 7.8
    assert metrics.closed_trades == 35
    assert metrics.win_rate_pct == 60.0
    assert metrics.hold_share == 0.75
    assert metrics.simplex_errors == 0
    assert metrics.nan_inf_errors == 0
    assert metrics.premature_closes == 0


def test_extract_cycle_metrics_from_events_only(tmp_path: Path):
    """Verify fallback extraction from events.jsonl when diagnostics is absent."""
    with open(tmp_path / "events.jsonl", "w", encoding="utf-8") as f:
        for i in range(25):
            f.write(json.dumps({
                "event_type": "lifecycle",
                "payload": {
                    "realized_net_r": 1.0 if i < 15 else -0.5,
                    "close_reason": "TAKE_PROFIT" if i < 15 else "STOP_LOSS",
                },
            }) + "\n")

    metrics = extract_cycle_metrics(tmp_path, cycle_id=2)
    assert metrics.cycle_id == 2
    assert metrics.closed_trades == 25
    assert metrics.net_pnl == 15 * 1.0 - 10 * 0.5  # 10.0
    assert metrics.profit_factor == (15.0 / 5.0)  # 3.0
    assert metrics.win_rate_pct == (15 / 25 * 100.0)  # 60.0%


# ---------------------------------------------------------------------------
# 5. Acceptance Criteria Evaluation Tests
# ---------------------------------------------------------------------------

def test_evaluate_acceptance_criteria_passing():
    """Verify acceptance criteria evaluation when all conditions are fulfilled."""
    metrics = CycleMetrics(
        cycle_id=1,
        net_pnl=500.0,
        profit_factor=1.35,
        max_drawdown_pct=9.5,
        closed_trades=28,
        hold_share=0.80,
        simplex_errors=0,
        nan_inf_errors=0,
    )
    verdict = evaluate_acceptance_criteria(metrics)
    assert verdict.passed
    assert len(verdict.reasons) == 0
    assert verdict.net_pnl_ok
    assert verdict.profit_factor_ok
    assert verdict.drawdown_ok
    assert verdict.trade_frequency_ok
    assert verdict.simplex_safe


def test_evaluate_acceptance_criteria_failures():
    """Verify individual acceptance criteria failures are properly identified."""
    # AC-1 Failure: Negative Net Profit
    m_loss = CycleMetrics(cycle_id=1, net_pnl=-50.0, profit_factor=1.20, max_drawdown_pct=8.0, closed_trades=30)
    v_loss = evaluate_acceptance_criteria(m_loss)
    assert not v_loss.passed
    assert not v_loss.net_pnl_ok
    assert any("Net Profit <= 0" in r for r in v_loss.reasons)

    # AC-2 Failure: Profit Factor < 1.15
    m_pf = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.05, max_drawdown_pct=8.0, closed_trades=30)
    v_pf = evaluate_acceptance_criteria(m_pf)
    assert not v_pf.passed
    assert not v_pf.profit_factor_ok
    assert any("Profit Factor < 1.15" in r for r in v_pf.reasons)

    # AC-3 Failure: Max Drawdown >= 15.0%
    m_dd = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.25, max_drawdown_pct=16.5, closed_trades=30)
    v_dd = evaluate_acceptance_criteria(m_dd)
    assert not v_dd.passed
    assert not v_dd.drawdown_ok
    assert any("Max Drawdown >= 15.0%" in r for r in v_dd.reasons)

    # AC-4 Failure: Closed trades < 20
    m_trades = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.25, max_drawdown_pct=5.0, closed_trades=12)
    v_trades = evaluate_acceptance_criteria(m_trades)
    assert not v_trades.passed
    assert not v_trades.trade_frequency_ok
    assert any("Closed trades < 20" in r for r in v_trades.reasons)

    # AC-4 Failure: 100% HOLD trap
    m_hold = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.25, max_drawdown_pct=5.0, closed_trades=22, hold_share=0.995)
    v_hold = evaluate_acceptance_criteria(m_hold)
    assert not v_hold.passed
    assert not v_hold.trade_frequency_ok
    assert any("HOLD trap" in r for r in v_hold.reasons)

    # AC-5 Failure: Simplex / NaN divergence
    m_num = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.25, max_drawdown_pct=5.0, closed_trades=25, simplex_errors=2)
    v_num = evaluate_acceptance_criteria(m_num)
    assert not v_num.passed
    assert not v_num.simplex_safe
    assert any("Numerical instability" in r for r in v_num.reasons)


# ---------------------------------------------------------------------------
# 6. Adaptive Tuning on Loss Engine (R2) Tests
# ---------------------------------------------------------------------------

def test_diagnose_failure_identifies_causes():
    """Verify root-cause diagnosis accurately flags failure mechanisms."""
    metrics = CycleMetrics(
        cycle_id=2,
        net_pnl=-120.0,
        profit_factor=0.88,
        max_drawdown_pct=16.2,
        closed_trades=14,
        hold_share=0.96,
        simplex_errors=1,
    )
    reasons = ["Net Profit <= 0", "Profit Factor < 1.15", "Max Drawdown >= 15.0%", "Closed trades < 20", "Simplex error"]
    diag = diagnose_failure(metrics, reasons)

    assert diag.is_loss
    assert diag.is_low_pf
    assert diag.is_high_dd
    assert diag.is_under_trading
    assert diag.is_numerical_divergence
    assert len(diag.suggested_actions) >= 3


def test_compute_adaptive_tuning_transitions():
    """Verify progressive hyperparameter tightening across retries 1, 2, 3."""
    # Retry 1: LCB -0.05, mix 0.01, pattern 0.15
    ov1 = compute_adaptive_tuning_overrides(retry_idx=1)
    assert ov1["five_expectancy_action_veto_min_lcb_r"] == -0.05
    assert ov1["five_behavior_uniform_mix"] == 0.01
    assert ov1["five_pattern_action_quality_reward_scale_r"] == 0.15
    assert ov1["five_pattern_tree_direction_loss_weight"] == 1.25
    assert ov1["five_lock_non_entry_to_hold"] is True

    # Retry 2: LCB 0.0, mix 0.005, pattern 0.20, direction loss 1.50
    ov2 = compute_adaptive_tuning_overrides(retry_idx=2)
    assert ov2["five_expectancy_action_veto_min_lcb_r"] == 0.00
    assert ov2["five_behavior_uniform_mix"] == 0.005
    assert ov2["five_pattern_action_quality_reward_scale_r"] == 0.20
    assert ov2["five_pattern_tree_direction_loss_weight"] == 1.50

    # Retry 3: LCB 0.05, mix 0.000 (pure exploitation), pattern 0.25
    ov3 = compute_adaptive_tuning_overrides(retry_idx=3)
    assert ov3["five_expectancy_action_veto_min_lcb_r"] == 0.05
    assert ov3["five_behavior_uniform_mix"] == 0.000
    assert ov3["five_pattern_action_quality_reward_scale_r"] == 0.25


def test_generate_adapted_profile_yaml(tmp_path: Path):
    """Verify that adapted profile YAML is generated correctly with overridden parameters."""
    base_profile = tmp_path / "base_profile.yaml"
    base_profile.write_text(yaml.safe_dump({
        "experiment": {"id": "sgt_compact_tier2_candidate"},
        "learning_overrides": {
            "tier2_compact_core_enabled": True,
            "five_expectancy_action_veto_min_lcb_r": -0.15,
            "five_behavior_uniform_mix": 0.02,
        },
    }), encoding="utf-8")

    adapted_profile = tmp_path / "adapted_profile.yaml"
    tuning_ov = {
        "five_expectancy_action_veto_min_lcb_r": -0.05,
        "five_behavior_uniform_mix": 0.01,
        "five_pattern_action_quality_reward_scale_r": 0.15,
    }

    out_path = generate_adapted_profile_yaml(
        base_profile_path=base_profile,
        output_profile_path=adapted_profile,
        tuning_overrides=tuning_ov,
        experiment_id="sgt_compact_cycle_1_retry_1",
    )

    assert out_path.is_file()
    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert loaded["experiment"]["id"] == "sgt_compact_cycle_1_retry_1"
    lo = loaded["learning_overrides"]
    assert lo["tier2_compact_core_enabled"] is True
    assert lo["five_expectancy_action_veto_min_lcb_r"] == -0.05
    assert lo["five_behavior_uniform_mix"] == 0.01
    assert lo["five_pattern_action_quality_reward_scale_r"] == 0.15


# ---------------------------------------------------------------------------
# 7. Candidate Artifact Manifest & Certification Tests
# ---------------------------------------------------------------------------

def test_generate_walk_forward_manifest(tmp_path: Path):
    """Verify Candidate Artifact Manifest generation and schema verification."""
    cand_path = tmp_path / "final_candidate.pkl"
    cand_path.write_bytes(b"AUTHENTIC_CANDIDATE_DATA")
    initial_seed = tmp_path / "initial_seed.pkl"
    initial_seed.write_bytes(b"INITIAL_SEED_DATA")

    cycle_results = {
        1: {
            "cycle_id": 1,
            "name": "Cycle 1 (Q1/2024)",
            "promoted": True,
            "metrics": {"net_pnl": 520.0, "profit_factor": 1.30, "max_drawdown_pct": 6.5, "closed_trades": 28},
            "verdict": {"passed": True},
        },
        2: {
            "cycle_id": 2,
            "name": "Cycle 2 (Q3/2024)",
            "promoted": True,
            "metrics": {"net_pnl": 410.0, "profit_factor": 1.25, "max_drawdown_pct": 8.2, "closed_trades": 24},
            "verdict": {"passed": True},
        },
        3: {
            "cycle_id": 3,
            "name": "Cycle 3 (Q1/2025)",
            "promoted": True,
            "metrics": {"net_pnl": 680.0, "profit_factor": 1.45, "max_drawdown_pct": 5.4, "closed_trades": 32},
            "verdict": {"passed": True},
        },
    }

    manifest = generate_walk_forward_manifest(
        candidate_checkpoint_path=cand_path,
        cycle_results=cycle_results,
        output_dir=tmp_path,
        initial_seed_path=initial_seed,
        target_cycles=[1, 2, 3],
    )

    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.is_file()

    assert manifest["multi_cycle_certified"] is True
    assert manifest["candidate"]["sha256"] == sha256_file(cand_path)
    assert manifest["candidate"]["exists"] is True
    assert manifest["initial_seed"]["sha256"] == sha256_file(initial_seed)

    agg = manifest["aggregate_metrics"]
    assert agg["total_trades"] == 28 + 24 + 32  # 84
    assert agg["total_net_pnl"] == 520.0 + 410.0 + 680.0  # 1610.0
    assert agg["worst_max_drawdown_pct"] == 8.2
    assert agg["all_cycles_passed"] is True


# ---------------------------------------------------------------------------
# 8. Full WalkForwardEngine Execution & Checkpoint Chaining Tests
# ---------------------------------------------------------------------------

def test_walk_forward_engine_runs_cycle_with_adaptive_tuning_retry(tmp_path: Path):
    """Verify that WalkForwardEngine triggers adaptive retry and rollback upon initial loss."""
    seed_file = tmp_path / "test_seed.pkl"
    seed_file.write_bytes(b"INITIAL_SEED_BYTES")

    attempt_counter = 0

    def mock_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
        nonlocal attempt_counter
        attempt_counter += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create dummy candidate checkpoint
        (output_dir / "candidate.pkl").write_bytes(f"CANDIDATE_ATTEMPT_{attempt_counter}".encode())

        if attempt_counter == 1:
            # First attempt fails with Net Profit <= 0
            diag = {
                "performance": {"net_pnl": -50.0, "profit_factor": 0.85, "closed_trades": 22},
                "risk": {"max_equity_drawdown_pct": 12.0},
                "policy": {"hold_share": 0.80},
            }
        else:
            # Retry attempt succeeds with Net Profit > 0, PF >= 1.15
            diag = {
                "performance": {"net_pnl": 450.0, "profit_factor": 1.40, "closed_trades": 30},
                "risk": {"max_equity_drawdown_pct": 7.5},
                "policy": {"hold_share": 0.78},
            }

        (output_dir / "mt5_diagnostics.json").write_text(json.dumps(diag), encoding="utf-8")
        return True

    engine = WalkForwardEngine(
        workspace_dir=tmp_path,
        config_path=Path("dummy_config.yaml"),
        base_profile_path=DEFAULT_BASE_PROFILE,
        runner_fn=mock_runner,
        max_adaptive_retries=3,
    )

    out_base = tmp_path / "cycle_run"
    result = engine.run_cycle(cycle_id=1, seed_checkpoint=seed_file, output_base_dir=out_base)

    assert result["promoted"] is True
    assert result["retries_used"] == 1
    assert attempt_counter == 2
    assert (out_base / "checkpoint_cycle_1.pkl").is_file()
    assert (out_base / "mt5_diagnostics_cycle_1.json").is_file()


def test_walk_forward_engine_full_sequential_chain(tmp_path: Path):
    """Verify full sequential execution (Cycle 1 -> Cycle 2 -> Cycle 3) with checkpoint inheritance."""
    seed_file = tmp_path / "seed_sgt_compact_tier2.pkl"
    seed_file.write_bytes(b"INITIAL_SEED_CORE_SGT_COMPACT")

    received_seeds: list[str] = []

    def mock_sequential_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
        received_seeds.append(seed_artifact.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create output candidate
        (output_dir / "candidate.pkl").write_bytes(f"CANDIDATE_FROM_{seed_artifact.name}".encode())

        # Successful diagnostics
        diag = {
            "performance": {"net_pnl": 500.0, "profit_factor": 1.35, "closed_trades": 25},
            "risk": {"max_equity_drawdown_pct": 6.0},
            "policy": {"hold_share": 0.70},
        }
        (output_dir / "mt5_diagnostics.json").write_text(json.dumps(diag), encoding="utf-8")
        return True

    engine = WalkForwardEngine(
        workspace_dir=tmp_path,
        config_path=Path("dummy_config.yaml"),
        base_profile_path=DEFAULT_BASE_PROFILE,
        runner_fn=mock_sequential_runner,
    )

    out_base = tmp_path / "all_cycles_output"
    manifest = engine.run_all_cycles(initial_seed=seed_file, output_base_dir=out_base)

    assert manifest["multi_cycle_certified"] is True
    assert (out_base / "manifest.json").is_file()
    assert (out_base / "checkpoint_cycle_1.pkl").is_file()
    assert (out_base / "checkpoint_cycle_2.pkl").is_file()
    assert (out_base / "checkpoint_cycle_3.pkl").is_file()
    assert (out_base / "final_promoted_checkpoint.pkl").is_file()

    # Verify sequential inheritance:
    # Cycle 1 received seed_sgt_compact_tier2.pkl
    # Cycle 2 received checkpoint_cycle_1.pkl
    # Cycle 3 received checkpoint_cycle_2.pkl
    assert received_seeds == [
        "seed_sgt_compact_tier2.pkl",
        "checkpoint_cycle_1.pkl",
        "checkpoint_cycle_2.pkl",
    ]


def test_simplex_safety_blocks_promotion(tmp_path: Path):
    """Verify that any Simplex/NaN error in event logs blocks checkpoint promotion."""
    seed_file = tmp_path / "seed.pkl"
    seed_file.write_bytes(b"SEED_DATA")

    def mock_runner_with_simplex_error(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "candidate.pkl").write_bytes(b"DIVERGED_CANDIDATE")

        # Diagnostics looks good on paper...
        diag = {
            "performance": {"net_pnl": 300.0, "profit_factor": 1.25, "closed_trades": 25},
            "risk": {"max_equity_drawdown_pct": 5.0},
        }
        (output_dir / "mt5_diagnostics.json").write_text(json.dumps(diag), encoding="utf-8")

        # But events.jsonl recorded a Simplex distribution error!
        with open(output_dir / "events.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "event_type": "error",
                "message": "ValueError: Expected parameter probs (Tensor of shape (5, 12)) to be within the simplex",
            }) + "\n")
        return True

    engine = WalkForwardEngine(
        workspace_dir=tmp_path,
        config_path=Path("dummy_config.yaml"),
        base_profile_path=DEFAULT_BASE_PROFILE,
        runner_fn=mock_runner_with_simplex_error,
        max_adaptive_retries=0,  # no retries to inspect immediate rejection
    )

    out_base = tmp_path / "simplex_test"
    res = engine.run_cycle(cycle_id=1, seed_checkpoint=seed_file, output_base_dir=out_base)

    assert not res["promoted"]
    assert not res["verdict"]["simplex_safe"]
    assert any("Numerical instability" in r for r in res["verdict"]["reasons"])
    assert not (out_base / "checkpoint_cycle_1.pkl").exists()
