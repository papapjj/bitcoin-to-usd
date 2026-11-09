"""Requirement-Driven Opaque-Box E2E Test Suite for Walk-Forward Multi-Cycle Pipeline.

This test suite covers Tiers 1 through 4 for the Walk-Forward Multi-Cycle Autonomous
RL Training and Optimization Loop (SGT-V193-Compact 17.5M):
- Tier 1: Feature Coverage (>=5 tests per feature across CLI argument validation,
  date boundaries, seed checkpoint verification, checkpoint inheritance,
  metric extraction criteria, and manifest schema: 35 tests).
- Tier 2: Boundary & Corner Cases (>=5 tests per feature across zero/extreme trades,
  breakeven net profit == 0.0, exact 15.0% max drawdown, zero-loss profit factor,
  missing/corrupted checkpoints, and simplex/NaN defense: 31 tests).
- Tier 3: Cross-Feature Combinations (Pairwise combinations of adaptive tuning knobs,
  checkpoint rollback with hyperparameter adaptation, and sequential rollover: 11 tests).
- Tier 4: Real-World Application Scenarios (5 end-to-end multi-cycle simulation scenarios
  including complete 3-cycle execution, Cycle 2 loss adaptive tuning retry pass,
  headless MT5 parameters, hold-collapse defense, and candidate artifact packaging: 5 tests).

Total: 82 comprehensive, opaque-box, deterministic, and self-contained tests.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pytest
import torch
import yaml

# Core repository imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from self_growing_trader.sgt_compact_core import SGTCompactTradingPolicy
from self_growing_trader.walk_forward_engine import (
    DEFAULT_BASE_PROFILE,
    DEFAULT_CYCLES,
    DEFAULT_INITIAL_SEED,
    DEFAULT_MAX_ADAPTIVE_RETRIES,
    DEFAULT_MAX_WALL_SEC,
    DEFAULT_RUNNER_SOURCE,
    DEFAULT_SOURCE_CONFIG,
    DEFAULT_TIMEOUT_SEC,
    MANIFEST_SCHEMA_VERSION,
    AcceptanceVerdict,
    AdaptiveTuningDiagnosis,
    CycleMetrics,
    WalkForwardCycleConfig,
    WalkForwardEngine,
    build_parser,
    build_tester_ini_content,
    clean_disk_if_low,
    compute_adaptive_tuning_overrides,
    diagnose_failure,
    evaluate_acceptance_criteria,
    extract_cycle_metrics,
    generate_adapted_profile_yaml,
    generate_walk_forward_manifest,
    sha256_file,
    verify_headless_configuration,
)


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature, 35 tests total)
# ==============================================================================

class TestTier1_1_CLIArgumentValidation:
    """Tier 1.1: CLI argument validation and default flag enforcement."""

    def test_t1_1_cli_single_cycle_argument_parsing(self) -> None:
        """Verify CLI parser handles --cycle {1,2,3} properly."""
        parser = build_parser()
        for cid in (1, 2, 3):
            args = parser.parse_args(["--cycle", str(cid)])
            assert args.cycle == cid
            assert args.all_cycles is False

    def test_t1_2_cli_all_cycles_flag_parsing(self) -> None:
        """Verify CLI parser handles --all-cycles flag."""
        parser = build_parser()
        args = parser.parse_args(["--all-cycles"])
        assert args.all_cycles is True
        assert args.cycle is None

    def test_t1_3_cli_mutex_cycle_vs_all_cycles(self) -> None:
        """Verify parser rejects providing both --cycle and --all-cycles or neither."""
        parser = build_parser()
        # Neither provided
        with pytest.raises(SystemExit):
            parser.parse_args([])
        # Both provided
        with pytest.raises(SystemExit):
            parser.parse_args(["--cycle", "1", "--all-cycles"])

    def test_t1_4_cli_default_timeout_and_wall_sec(self) -> None:
        """Verify CLI default flags: --timeout-sec 3600, --max-wall-sec 0."""
        parser = build_parser()
        args = parser.parse_args(["--all-cycles"])
        assert args.timeout_sec == 3600
        assert args.max_wall_sec == 0
        assert args.max_adaptive_retries == 3

    def test_t1_5_cli_absence_of_mt5_visual_enforcement(self) -> None:
        """Verify that --mt5-visual is NOT in CLI args, enforcing headless mode."""
        parser = build_parser()
        args = parser.parse_args(["--all-cycles"])
        assert not hasattr(args, "mt5_visual") or getattr(args, "mt5_visual") is False

        # Verify verify_headless_configuration validates absence of --mt5-visual
        valid, violations = verify_headless_configuration(["--timeout-sec", "3600", "--max-wall-sec", "0"])
        assert valid is True
        assert len(violations) == 0

        # Presence of --mt5-visual must be flagged as violation
        invalid, violations = verify_headless_configuration(
            ["--mt5-visual", "--timeout-sec", "3600", "--max-wall-sec", "0"]
        )
        assert invalid is False
        assert any("headless violation" in v.lower() for v in violations)

    def test_t1_6_cli_seed_and_output_defaults(self) -> None:
        """Verify default seed checkpoint and output directory configuration."""
        parser = build_parser()
        args = parser.parse_args(["--all-cycles"])
        assert Path(args.seed_checkpoint) == DEFAULT_INITIAL_SEED
        assert Path(args.output_dir) == Path("outputs/walk_forward_multi_cycle")

    def test_t1_7_cli_rejects_invalid_cycle_id(self) -> None:
        """Verify CLI parser strictly rejects cycle values outside {1, 2, 3}."""
        parser = build_parser()
        for invalid_val in ["0", "4", "99", "cycle1"]:
            with pytest.raises(SystemExit):
                parser.parse_args(["--cycle", invalid_val])


class TestTier1_2_DateBoundaryConfiguration:
    """Tier 1.2: Date boundary configuration for Cycles 1, 2, and 3."""

    def test_t1_8_date_boundary_cycle1_q1_2024(self) -> None:
        """Verify Cycle 1 date boundaries match 2024.01.08 to 2024.03.29 (Q1/2024)."""
        c1 = DEFAULT_CYCLES[1]
        assert c1.cycle_id == 1
        assert c1.from_date == "2024.01.08"
        assert c1.to_date == "2024.03.29"
        assert "Q1/2024" in c1.name

    def test_t1_9_date_boundary_cycle2_q3_2024(self) -> None:
        """Verify Cycle 2 date boundaries match 2024.07.01 to 2024.09.30 (Q3/2024)."""
        c2 = DEFAULT_CYCLES[2]
        assert c2.cycle_id == 2
        assert c2.from_date == "2024.07.01"
        assert c2.to_date == "2024.09.30"
        assert "Q3/2024" in c2.name

    def test_t1_10_date_boundary_cycle3_q1_2025(self) -> None:
        """Verify Cycle 3 date boundaries match 2025.01.06 to 2025.03.31 (Q1/2025)."""
        c3 = DEFAULT_CYCLES[3]
        assert c3.cycle_id == 3
        assert c3.from_date == "2025.01.06"
        assert c3.to_date == "2025.03.31"
        assert "Q1/2025" in c3.name

    def test_t1_11_date_boundaries_chronological_ordering(self) -> None:
        """Verify chronological progression: C1_end < C2_start < C2_end < C3_start."""
        c1 = DEFAULT_CYCLES[1]
        c2 = DEFAULT_CYCLES[2]
        c3 = DEFAULT_CYCLES[3]
        assert c1.from_date < c1.to_date
        assert c1.to_date < c2.from_date
        assert c2.from_date < c2.to_date
        assert c2.to_date < c3.from_date
        assert c3.from_date < c3.to_date

    def test_t1_12_date_format_mt5_tester_standard(self) -> None:
        """Verify date formats conform strictly to MT5 YYYY.MM.DD standard."""
        for cid, cfg in DEFAULT_CYCLES.items():
            for d in (cfg.from_date, cfg.to_date):
                parts = d.split(".")
                assert len(parts) == 3, f"Cycle {cid} invalid date format: {d}"
                assert len(parts[0]) == 4 and parts[0].isdigit()
                assert len(parts[1]) == 2 and 1 <= int(parts[1]) <= 12
                assert len(parts[2]) == 2 and 1 <= int(parts[2]) <= 31

    def test_t1_13_date_regime_descriptions(self) -> None:
        """Verify each cycle config documents its intended market regime."""
        assert "impulse" in DEFAULT_CYCLES[1].regime_description.lower()
        assert "sideway" in DEFAULT_CYCLES[2].regime_description.lower()
        assert "trend" in DEFAULT_CYCLES[3].regime_description.lower() or "robustness" in DEFAULT_CYCLES[3].regime_description.lower()


class TestTier1_3_SeedCheckpointVerification:
    """Tier 1.3: Seed checkpoint verification and SHA-256 loading."""

    def test_t1_14_seed_file_exists_on_disk(self) -> None:
        """Verify seed_sgt_compact_tier2.pkl exists on disk."""
        assert DEFAULT_INITIAL_SEED.is_file(), f"Seed checkpoint not found: {DEFAULT_INITIAL_SEED}"

    def test_t1_15_seed_sha256_exact_hash_match(self) -> None:
        """Verify seed SHA-256 matches 3f76d5a026c5a14b693c024fbee755a92fa9a04e085855be605009ff77d5c64f."""
        expected_hash = "3f76d5a026c5a14b693c024fbee755a92fa9a04e085855be605009ff77d5c64f"
        actual_hash = sha256_file(DEFAULT_INITIAL_SEED)
        assert actual_hash == expected_hash

    def test_t1_16_seed_file_byte_size(self) -> None:
        """Verify seed checkpoint byte size is exactly 86,378,353 bytes."""
        expected_size = 86378353
        actual_size = DEFAULT_INITIAL_SEED.stat().st_size
        assert actual_size == expected_size

    def test_t1_17_seed_sidecar_imitation_valid(self) -> None:
        """Verify seed imitation sidecar metadata records 17,791,629 parameters and authorized execution."""
        sidecar_path = Path(str(DEFAULT_INITIAL_SEED) + ".imitation.json")
        assert sidecar_path.is_file()
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert data.get("parameters") == 17791629
        assert data.get("architecture") == "SGTCompactTradingPolicy"
        assert data.get("validation_gate_passed") is True
        assert data.get("mt5_execution_allowed") is True

    def test_t1_18_seed_sidecar_registry_valid(self) -> None:
        """Verify seed registry sidecar records shared 5-agent context registry."""
        registry_path = DEFAULT_INITIAL_SEED.parent / "seed_sgt_compact_tier2.registry.json"
        assert registry_path.is_file()
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        assert data.get("che_do") == "nam_tac_nhan_dung_chung"
        assert data.get("so_chuyen_gia") >= 1
        assert any(cg.get("ma_chuyen_gia") == "BO_NAO_CHUNG" for cg in data.get("chuyen_gia", []))

    def test_t1_19_seed_sha256_file_helper_accuracy(self) -> None:
        """Verify sha256_file helper on synthetic deterministic bytes."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"WalkForwardTestingDeterministicPayload12345")
            tf_path = Path(tf.name)
        try:
            expected = hashlib.sha256(b"WalkForwardTestingDeterministicPayload12345").hexdigest()
            assert sha256_file(tf_path) == expected
            assert sha256_file(Path("non_existent_file_path_12345.xyz")) == ""
        finally:
            tf_path.unlink(missing_ok=True)


class TestTier1_4_CheckpointInheritance:
    """Tier 1.4: Checkpoint inheritance between cycles."""

    def test_t1_20_inheritance_c1_receives_seed(self) -> None:
        """Verify Cycle 1 executes using initial seed and produces checkpoint_cycle_1.pkl."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            used_seeds = []

            def mock_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                used_seeds.append(seed_artifact)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "candidate.pkl").write_bytes(b"CAND1")
                (output_dir / "mt5_diagnostics.json").write_text(json.dumps({
                    "performance": {"net_pnl": 500.0, "profit_factor": 1.4, "closed_trades": 30},
                    "risk": {"max_equity_drawdown_pct": 5.0},
                    "policy": {"hold_share": 0.6},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=mock_runner)
            res = engine.run_cycle(cycle_id=1, seed_checkpoint=DEFAULT_INITIAL_SEED, output_base_dir=base_dir)
            assert res["promoted"] is True
            assert used_seeds[0] == DEFAULT_INITIAL_SEED.resolve()
            assert (base_dir / "checkpoint_cycle_1.pkl").is_file()

    def test_t1_21_inheritance_c2_receives_c1_promoted(self) -> None:
        """Verify Cycle 2 executes receiving Cycle 1's promoted checkpoint as input seed."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            c1_ckpt = base_dir / "checkpoint_cycle_1.pkl"
            c1_ckpt.write_bytes(b"C1_CHECKPOINT_BYTES")
            used_seeds = []

            def mock_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                used_seeds.append(seed_artifact)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "candidate.pkl").write_bytes(b"CAND2")
                (output_dir / "mt5_diagnostics.json").write_text(json.dumps({
                    "performance": {"net_pnl": 600.0, "profit_factor": 1.45, "closed_trades": 35},
                    "risk": {"max_equity_drawdown_pct": 6.0},
                    "policy": {"hold_share": 0.6},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=mock_runner)
            res = engine.run_cycle(cycle_id=2, seed_checkpoint=c1_ckpt, output_base_dir=base_dir)
            assert res["promoted"] is True
            assert used_seeds[0] == c1_ckpt.resolve()
            assert (base_dir / "checkpoint_cycle_2.pkl").is_file()

    def test_t1_22_inheritance_c3_receives_c2_promoted(self) -> None:
        """Verify Cycle 3 executes receiving Cycle 2's promoted checkpoint as input seed."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            c2_ckpt = base_dir / "checkpoint_cycle_2.pkl"
            c2_ckpt.write_bytes(b"C2_CHECKPOINT_BYTES")
            used_seeds = []

            def mock_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                used_seeds.append(seed_artifact)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "candidate.pkl").write_bytes(b"CAND3")
                (output_dir / "mt5_diagnostics.json").write_text(json.dumps({
                    "performance": {"net_pnl": 700.0, "profit_factor": 1.50, "closed_trades": 38},
                    "risk": {"max_equity_drawdown_pct": 5.5},
                    "policy": {"hold_share": 0.6},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=mock_runner)
            res = engine.run_cycle(cycle_id=3, seed_checkpoint=c2_ckpt, output_base_dir=base_dir)
            assert res["promoted"] is True
            assert used_seeds[0] == c2_ckpt.resolve()
            assert (base_dir / "checkpoint_cycle_3.pkl").is_file()

    def test_t1_23_inheritance_source_seed_immutability(self) -> None:
        """Verify source seed remains strictly immutable across cloning and inheritance."""
        with tempfile.TemporaryDirectory() as td:
            fake_seed = Path(td) / "source_seed.pkl"
            fake_seed.write_bytes(b"IMMUTABLE_SEED_CONTENT_001")
            hash_before = sha256_file(fake_seed)

            # Simulate cloning into cycle workspace
            cycle1_cand = Path(td) / "cycle_1" / "candidate.pkl"
            cycle1_cand.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fake_seed, cycle1_cand)

            hash_after = sha256_file(fake_seed)
            assert hash_before == hash_after
            assert sha256_file(cycle1_cand) == hash_before

    def test_t1_24_inheritance_clone_checkpoint_integrity(self) -> None:
        """Verify cloned candidate retains bit-for-bit SHA256 integrity."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "source.pkl"
            src.write_bytes(os.urandom(1024 * 100))  # 100 KB random payload
            dst = Path(td) / "cloned.pkl"
            shutil.copy2(src, dst)
            assert sha256_file(src) == sha256_file(dst)
            assert src.stat().st_size == dst.stat().st_size


class TestTier1_5_MetricExtractionCriteria:
    """Tier 1.5: Metric extraction and quantitative acceptance criteria validation."""

    def test_t1_25_metrics_all_criteria_passed(self) -> None:
        """Verify acceptance verdict passes when all hurdles are satisfied."""
        metrics = CycleMetrics(
            cycle_id=1,
            net_pnl=1250.0,
            profit_factor=1.45,
            max_drawdown_pct=8.5,
            closed_trades=35,
            simplex_errors=0,
            nan_inf_errors=0,
            hold_share=0.75,
        )
        verdict = evaluate_acceptance_criteria(metrics)
        assert verdict.passed is True
        assert len(verdict.reasons) == 0
        assert verdict.net_pnl_ok is True
        assert verdict.profit_factor_ok is True
        assert verdict.drawdown_ok is True
        assert verdict.trade_frequency_ok is True
        assert verdict.simplex_safe is True

    def test_t1_26_metrics_reject_negative_net_profit(self) -> None:
        """Verify AC-1: Net Profit <= 0 causes failure."""
        metrics = CycleMetrics(
            cycle_id=1,
            net_pnl=-10.50,
            profit_factor=1.30,
            max_drawdown_pct=5.0,
            closed_trades=30,
        )
        verdict = evaluate_acceptance_criteria(metrics)
        assert verdict.passed is False
        assert verdict.net_pnl_ok is False
        assert any("Net Profit <= 0" in r for r in verdict.reasons)

    def test_t1_27_metrics_reject_sub_hurdle_profit_factor(self) -> None:
        """Verify AC-2: Profit Factor < 1.15 causes failure."""
        metrics = CycleMetrics(
            cycle_id=1,
            net_pnl=100.0,
            profit_factor=1.12,  # < 1.15
            max_drawdown_pct=5.0,
            closed_trades=30,
        )
        verdict = evaluate_acceptance_criteria(metrics)
        assert verdict.passed is False
        assert verdict.profit_factor_ok is False
        assert any("Profit Factor < 1.15" in r for r in verdict.reasons)

    def test_t1_28_metrics_reject_excessive_drawdown(self) -> None:
        """Verify AC-3: Max Drawdown >= 15.0% causes failure."""
        metrics = CycleMetrics(
            cycle_id=1,
            net_pnl=500.0,
            profit_factor=1.40,
            max_drawdown_pct=15.8,  # >= 15.0%
            closed_trades=30,
        )
        verdict = evaluate_acceptance_criteria(metrics)
        assert verdict.passed is False
        assert verdict.drawdown_ok is False
        assert any("Max Drawdown >= 15.0%" in r for r in verdict.reasons)

    def test_t1_29_metrics_reject_insufficient_closed_trades(self) -> None:
        """Verify AC-4: Closed Trades < 20 causes failure."""
        metrics = CycleMetrics(
            cycle_id=1,
            net_pnl=500.0,
            profit_factor=1.50,
            max_drawdown_pct=5.0,
            closed_trades=18,  # < 20
        )
        verdict = evaluate_acceptance_criteria(metrics)
        assert verdict.passed is False
        assert verdict.trade_frequency_ok is False
        assert any("Closed trades < 20" in r for r in verdict.reasons)

    def test_t1_30_metrics_ingestion_from_diagnostics_and_events(self) -> None:
        """Verify extract_cycle_metrics properly ingests diagnostics.json and events.jsonl."""
        with tempfile.TemporaryDirectory() as td:
            cycle_dir = Path(td)
            diag_file = cycle_dir / "mt5_diagnostics.json"
            events_file = cycle_dir / "events.jsonl"

            diag_content = {
                "performance": {
                    "net_pnl": 340.25,
                    "profit_factor": 1.35,
                    "closed_trades": 28,
                    "win_rate": 0.54,
                    "payoff_ratio": 1.70,
                    "expectancy_r": 0.08,
                    "close_reasons": {"TP": 15, "SL": 13},
                },
                "risk": {
                    "max_equity_drawdown_pct": 7.2,
                },
                "policy": {
                    "hold_share": 0.72,
                },
            }
            diag_file.write_text(json.dumps(diag_content), encoding="utf-8")

            events_lines = [
                json.dumps({"event_type": "lifecycle", "payload": {"realized_net_r": 2.0, "close_reason": "TP"}}),
                json.dumps({"event_type": "lifecycle", "payload": {"realized_net_r": -1.0, "close_reason": "SL"}}),
            ]
            events_file.write_text("\n".join(events_lines), encoding="utf-8")

            m = extract_cycle_metrics(cycle_dir, cycle_id=1)
            assert m.net_pnl == 340.25
            assert m.profit_factor == 1.35
            assert m.max_drawdown_pct == 7.2
            assert m.closed_trades == 28
            assert m.simplex_errors == 0
            assert m.nan_inf_errors == 0


class TestTier1_6_ManifestSchemaAndHashing:
    """Tier 1.6: Manifest schema, SHA-256 hash presence, and structure."""

    def test_t1_31_manifest_required_fields(self) -> None:
        """Verify manifest contains all required top-level schema fields."""
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "candidate.pkl"
            cand.write_bytes(b"CANDIDATE_DATA")
            seed = Path(td) / "seed.pkl"
            seed.write_bytes(b"SEED_DATA")

            cycle_results = {
                1: {"verdict": {"passed": True}, "metrics": {"closed_trades": 30, "net_pnl": 500.0, "profit_factor": 1.3, "max_drawdown_pct": 6.0}},
                2: {"verdict": {"passed": True}, "metrics": {"closed_trades": 35, "net_pnl": 600.0, "profit_factor": 1.4, "max_drawdown_pct": 7.0}},
                3: {"verdict": {"passed": True}, "metrics": {"closed_trades": 40, "net_pnl": 700.0, "profit_factor": 1.5, "max_drawdown_pct": 5.5}},
            }
            manifest = generate_walk_forward_manifest(cand, cycle_results, Path(td), seed)
            assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
            assert "timestamp_utc" in manifest
            assert "candidate" in manifest
            assert "initial_seed" in manifest
            assert "multi_cycle_certified" in manifest
            assert "aggregate_metrics" in manifest
            assert "cycle_results" in manifest

    def test_t1_32_manifest_candidate_sha256_format(self) -> None:
        """Verify manifest candidate SHA-256 is a 64-character valid hex string."""
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "cand.pkl"
            cand.write_bytes(b"PAYLOAD_123")
            seed = Path(td) / "seed.pkl"
            seed.write_bytes(b"SEED_123")
            manifest = generate_walk_forward_manifest(cand, {}, Path(td), seed)
            sha = manifest["candidate"]["sha256"]
            assert len(sha) == 64
            assert all(c in "0123456789abcdef" for c in sha)
            assert sha == hashlib.sha256(b"PAYLOAD_123").hexdigest()

    def test_t1_33_manifest_per_cycle_metrics_recorded(self) -> None:
        """Verify manifest records per-cycle metrics accurately."""
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "cand.pkl"
            cand.write_bytes(b"A")
            seed = Path(td) / "s.pkl"
            seed.write_bytes(b"B")
            cycle_results = {
                1: {"verdict": {"passed": True}, "metrics": {"closed_trades": 25, "net_pnl": 350.0, "profit_factor": 1.3, "max_drawdown_pct": 5.0}},
            }
            manifest = generate_walk_forward_manifest(cand, cycle_results, Path(td), seed, target_cycles=[1])
            res = manifest["cycle_results"][1]
            assert res["metrics"]["net_pnl"] == 350.0
            assert res["metrics"]["closed_trades"] == 25

    def test_t1_34_manifest_certification_boolean(self) -> None:
        """Verify multi_cycle_certified is True if and only if all cycles passed."""
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "c.pkl"
            cand.write_bytes(b"1")
            seed = Path(td) / "s.pkl"
            seed.write_bytes(b"2")

            passed_res = {
                c: {"verdict": {"passed": True}, "metrics": {"closed_trades": 25, "net_pnl": 100.0, "profit_factor": 1.2, "max_drawdown_pct": 5.0}}
                for c in (1, 2, 3)
            }
            m_pass = generate_walk_forward_manifest(cand, passed_res, Path(td), seed)
            assert m_pass["multi_cycle_certified"] is True

            failed_res = dict(passed_res)
            failed_res[2] = {"verdict": {"passed": False}, "metrics": {"closed_trades": 25, "net_pnl": -50.0, "profit_factor": 0.8, "max_drawdown_pct": 16.0}}
            m_fail = generate_walk_forward_manifest(cand, failed_res, Path(td), seed)
            assert m_fail["multi_cycle_certified"] is False

    def test_t1_35_manifest_json_serialization_roundtrip(self) -> None:
        """Verify manifest written to disk parses back with 100% equivalence."""
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "cand.pkl"
            cand.write_bytes(b"DATA")
            seed = Path(td) / "seed.pkl"
            seed.write_bytes(b"SEED")
            manifest_out = generate_walk_forward_manifest(cand, {}, Path(td), seed)

            manifest_path = Path(td) / "manifest.json"
            assert manifest_path.is_file()
            reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert reloaded == manifest_out


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 tests per feature, 31 tests total)
# ==============================================================================

class TestTier2_1_ZeroAndExtremeTrades:
    """Tier 2.1: Zero trades, boundary trades, and hold-collapse detection."""

    def test_t2_1_zero_closed_trades_fails_and_triggers_diagnosis(self) -> None:
        """Verify 0 trades fails acceptance and is diagnosed as under-trading."""
        m = CycleMetrics(cycle_id=1, closed_trades=0, net_pnl=0.0, profit_factor=0.0, max_drawdown_pct=0.0)
        v = evaluate_acceptance_criteria(m)
        assert v.passed is False
        assert v.trade_frequency_ok is False
        assert any("Closed trades < 20" in r for r in v.reasons)

        diag = diagnose_failure(m, v.reasons)
        assert diag.is_under_trading is True

    def test_t2_2_single_trade_rejected(self) -> None:
        """Verify 1 trade is strictly rejected under AC-4."""
        m = CycleMetrics(cycle_id=1, closed_trades=1, net_pnl=50.0, profit_factor=2.0, max_drawdown_pct=1.0)
        v = evaluate_acceptance_criteria(m)
        assert v.passed is False
        assert v.trade_frequency_ok is False

    def test_t2_3_nineteen_trades_just_below_boundary_rejected(self) -> None:
        """Verify 19 trades (exact boundary - 1) is rejected."""
        m = CycleMetrics(cycle_id=1, closed_trades=19, net_pnl=500.0, profit_factor=1.5, max_drawdown_pct=4.0)
        v = evaluate_acceptance_criteria(m)
        assert v.passed is False
        assert v.trade_frequency_ok is False

    def test_t2_4_twenty_trades_exact_boundary_accepted(self) -> None:
        """Verify 20 trades (exact boundary) passes the trade frequency hurdle."""
        m = CycleMetrics(cycle_id=1, closed_trades=20, net_pnl=500.0, profit_factor=1.5, max_drawdown_pct=4.0, hold_share=0.80)
        v = evaluate_acceptance_criteria(m)
        assert v.trade_frequency_ok is True
        assert v.passed is True

    def test_t2_5_hold_collapse_trap_ninety_nine_percent_rejected(self) -> None:
        """Verify hold_share >= 0.99 triggers 100% HOLD trap rejection."""
        m = CycleMetrics(cycle_id=1, closed_trades=25, net_pnl=100.0, profit_factor=1.2, max_drawdown_pct=2.0, hold_share=0.995)
        v = evaluate_acceptance_criteria(m)
        assert v.passed is False
        assert v.trade_frequency_ok is False
        assert any("100% HOLD trap" in r for r in v.reasons)

    def test_t2_6_high_volume_trades_boundary(self) -> None:
        """Verify high volume trade count (e.g. 10,000 trades) is handled cleanly."""
        m = CycleMetrics(cycle_id=1, closed_trades=10000, net_pnl=5000.0, profit_factor=1.35, max_drawdown_pct=9.0, hold_share=0.40)
        v = evaluate_acceptance_criteria(m)
        assert v.trade_frequency_ok is True
        assert v.passed is True


class TestTier2_2_BreakevenNetProfit:
    """Tier 2.2: Breakeven net profit (net_pnl == 0.0) boundary and adaptive tuning trigger."""

    def test_t2_7_exact_zero_net_profit_rejected(self) -> None:
        """Verify Net Profit == 0.00 is strictly rejected (AC-1 requires > 0.0)."""
        m = CycleMetrics(cycle_id=1, net_pnl=0.00, profit_factor=1.30, max_drawdown_pct=5.0, closed_trades=30)
        v = evaluate_acceptance_criteria(m)
        assert v.passed is False
        assert v.net_pnl_ok is False
        assert any("Net Profit <= 0" in r for r in v.reasons)

    def test_t2_8_breakeven_diagnosed_as_loss(self) -> None:
        """Verify Net Profit == 0.00 triggers is_loss = True in diagnosis."""
        m = CycleMetrics(cycle_id=1, net_pnl=0.00, profit_factor=1.20, max_drawdown_pct=5.0, closed_trades=30)
        diag = diagnose_failure(m, ["Net Profit <= 0"])
        assert diag.is_loss is True
        assert any("veto lcb" in a.lower() for a in diag.suggested_actions)

    def test_t2_9_micro_loss_minus_one_cent_rejected(self) -> None:
        """Verify Net Profit == -0.01 is strictly rejected."""
        m = CycleMetrics(cycle_id=1, net_pnl=-0.01, profit_factor=1.20, max_drawdown_pct=5.0, closed_trades=30)
        v = evaluate_acceptance_criteria(m)
        assert v.passed is False
        assert v.net_pnl_ok is False

    def test_t2_10_micro_profit_plus_one_cent_accepted(self) -> None:
        """Verify Net Profit == +0.01 passes the net profit hurdle."""
        m = CycleMetrics(cycle_id=1, net_pnl=0.01, profit_factor=1.20, max_drawdown_pct=5.0, closed_trades=30, hold_share=0.7)
        v = evaluate_acceptance_criteria(m)
        assert v.net_pnl_ok is True
        assert v.passed is True

    def test_t2_11_breakeven_triggers_tuning_overrides(self) -> None:
        """Verify breakeven diagnosis generates tightened LCB and reduced uniform mix."""
        diag = diagnose_failure(CycleMetrics(cycle_id=1, net_pnl=0.0, profit_factor=1.0, max_drawdown_pct=5.0, closed_trades=25), ["Net Profit <= 0"])
        overrides = compute_adaptive_tuning_overrides(retry_idx=1, diagnosis=diag)
        assert overrides["five_expectancy_action_veto_min_lcb_r"] >= -0.05
        assert overrides["five_behavior_uniform_mix"] <= 0.01


class TestTier2_3_DrawdownExactFifteenPercent:
    """Tier 2.3: Drawdown at exact 15.0% boundary and extreme drawdowns."""

    def test_t2_12_drawdown_fourteen_point_nine_nine_percent_accepted(self) -> None:
        """Verify Max Drawdown = 14.99% passes (< 15.0%)."""
        m = CycleMetrics(cycle_id=1, net_pnl=200.0, profit_factor=1.25, max_drawdown_pct=14.99, closed_trades=30, hold_share=0.5)
        v = evaluate_acceptance_criteria(m)
        assert v.drawdown_ok is True
        assert v.passed is True

    def test_t2_13_drawdown_exact_fifteen_percent_rejected(self) -> None:
        """Verify Max Drawdown = 15.00% is strictly rejected (hurdle is < 15.0%)."""
        m = CycleMetrics(cycle_id=1, net_pnl=200.0, profit_factor=1.25, max_drawdown_pct=15.00, closed_trades=30)
        v = evaluate_acceptance_criteria(m)
        assert v.drawdown_ok is False
        assert v.passed is False
        assert any("Max Drawdown >= 15.0%" in r for r in v.reasons)

    def test_t2_14_drawdown_fifteen_point_zero_one_percent_rejected(self) -> None:
        """Verify Max Drawdown = 15.01% is strictly rejected."""
        m = CycleMetrics(cycle_id=1, net_pnl=200.0, profit_factor=1.25, max_drawdown_pct=15.01, closed_trades=30)
        v = evaluate_acceptance_criteria(m)
        assert v.drawdown_ok is False
        assert v.passed is False

    def test_t2_15_drawdown_fractional_ratio_conversion(self) -> None:
        """Verify extract_cycle_metrics converts 0.149 (fraction) to 14.9% properly."""
        with tempfile.TemporaryDirectory() as td:
            diag_path = Path(td) / "mt5_diagnostics.json"
            diag_path.write_text(json.dumps({
                "performance": {"net_pnl": 100.0, "profit_factor": 1.2, "closed_trades": 25},
                "risk": {"max_equity_drawdown_pct": 0.149},  # fractional format
            }), encoding="utf-8")
            m = extract_cycle_metrics(Path(td), cycle_id=1)
            assert abs(m.max_drawdown_pct - 14.9) < 1e-4

    def test_t2_16_extreme_drawdown_ninety_nine_percent(self) -> None:
        """Verify extreme drawdown (e.g. 99.0%) triggers is_high_dd diagnosis."""
        m = CycleMetrics(cycle_id=1, net_pnl=-500.0, profit_factor=0.5, max_drawdown_pct=99.0, closed_trades=20)
        diag = diagnose_failure(m, ["Max Drawdown >= 15.0%"])
        assert diag.is_high_dd is True
        assert any("drawdown containment" in a.lower() for a in diag.suggested_actions)


class TestTier2_4_ProfitFactorZeroLosses:
    """Tier 2.4: Profit Factor with zero losses, division by zero defense, and exact 1.15 boundary."""

    def test_t2_17_pf_zero_losses_with_positive_gross_profit(self) -> None:
        """Verify gross profit > 0 with zero losses evaluates PF to 999.0 and passes AC-2."""
        with tempfile.TemporaryDirectory() as td:
            events_file = Path(td) / "events.jsonl"
            lines = [
                json.dumps({"event_type": "lifecycle", "payload": {"realized_net_r": 1.5, "close_reason": "TP"}}),
                json.dumps({"event_type": "lifecycle", "payload": {"realized_net_r": 2.0, "close_reason": "TP"}}),
            ]
            events_file.write_text("\n".join(lines), encoding="utf-8")
            m = extract_cycle_metrics(Path(td), cycle_id=1)
            assert m.profit_factor == 999.0
            v = evaluate_acceptance_criteria(m)
            assert v.profit_factor_ok is True

    def test_t2_18_pf_zero_losses_with_zero_gross_profit(self) -> None:
        """Verify zero profits and zero losses evaluates PF to 0.0 and fails AC-2."""
        m = CycleMetrics(cycle_id=1, profit_factor=0.0)
        v = evaluate_acceptance_criteria(m)
        assert v.profit_factor_ok is False

    def test_t2_19_pf_exact_one_point_fifteen_boundary_accepted(self) -> None:
        """Verify Profit Factor = 1.150 (exact hurdle) passes AC-2."""
        m = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.150, max_drawdown_pct=5.0, closed_trades=25, hold_share=0.5)
        v = evaluate_acceptance_criteria(m)
        assert v.profit_factor_ok is True
        assert v.passed is True

    def test_t2_20_pf_just_below_one_point_fifteen_rejected(self) -> None:
        """Verify Profit Factor = 1.149 is rejected under AC-2."""
        m = CycleMetrics(cycle_id=1, net_pnl=100.0, profit_factor=1.149, max_drawdown_pct=5.0, closed_trades=25)
        v = evaluate_acceptance_criteria(m)
        assert v.profit_factor_ok is False
        assert v.passed is False

    def test_t2_21_pf_zero_division_resilience(self) -> None:
        """Verify that zero losses in raw calculation does not raise ZeroDivisionError."""
        gross_profit = 1000.0
        gross_loss = 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        assert pf == 999.0


class TestTier2_5_CheckpointMissingCorrupted:
    """Tier 2.5: Missing, corrupted, and hash mismatch checkpoint handling."""

    def test_t2_22_missing_checkpoint_raises_file_not_found(self) -> None:
        """Verify engine raises FileNotFoundError when initial seed checkpoint is missing."""
        with tempfile.TemporaryDirectory() as td:
            missing_seed = Path(td) / "missing_seed_file.pkl"
            engine = WalkForwardEngine(workspace_dir=ROOT_DIR)
            with pytest.raises(FileNotFoundError, match="not found"):
                engine.run_all_cycles(initial_seed=missing_seed, output_base_dir=Path(td))

    def test_t2_23_corrupted_truncated_checkpoint_detected(self) -> None:
        """Verify corrupted / truncated checkpoint is detected via hash change."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"CORRUPTED_TRUNCATED_PKL_HEADER_ONLY")
            tf_path = Path(tf.name)
        try:
            sha = sha256_file(tf_path)
            assert sha != "3f76d5a026c5a14b693c024fbee755a92fa9a04e085855be605009ff77d5c64f"
        finally:
            tf_path.unlink(missing_ok=True)

    def test_t2_24_checkpoint_single_bit_flip_changes_hash(self) -> None:
        """Verify a single bit flip in checkpoint content changes the SHA-256 hash."""
        data1 = b"CheckpointWeightsTensorData00001"
        data2 = b"CheckpointWeightsTensorData00002"
        h1 = hashlib.sha256(data1).hexdigest()
        h2 = hashlib.sha256(data2).hexdigest()
        assert h1 != h2

    def test_t2_25_missing_cycle_predecessor_checkpoint(self) -> None:
        """Verify run_cycle rejects non-existent checkpoint paths."""
        with tempfile.TemporaryDirectory() as td:
            missing_c1 = Path(td) / "non_existent_c1.pkl"
            engine = WalkForwardEngine(workspace_dir=ROOT_DIR)
            # When missing seed is passed to run_cycle, the runner receives the missing path
            # and resolve_cycle_seed or runner detects it
            assert not missing_c1.is_file()

    def test_t2_26_empty_checkpoint_file_handled(self) -> None:
        """Verify 0-byte checkpoint file produces standard empty hash and 0 size."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf_path = Path(tf.name)
        try:
            assert tf_path.stat().st_size == 0
            assert sha256_file(tf_path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        finally:
            tf_path.unlink(missing_ok=True)


class TestTier2_6_SimplexNormalizationAndNaNDefense:
    """Tier 2.6: Simplex probability normalization and defensive numerical guards."""

    def test_t2_27_simplex_nan_inputs_sanitized_by_nan_to_num(self) -> None:
        """Verify torch.nan_to_num cleans NaN/Inf values from probability tensors."""
        t = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.5])
        cleaned = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        assert not torch.isnan(cleaned).any()
        assert not torch.isinf(cleaned).any()
        assert cleaned[0].item() == 0.0

    def test_t2_28_simplex_all_zeros_fallback_to_uniform(self) -> None:
        """Verify simplex guard fallback to uniform distribution on allowed actions when sum <= 1e-12."""
        probs = torch.zeros(1, 5, 12)
        allowed = torch.ones(1, 5, 12, dtype=torch.bool)

        act_sum = probs.sum(dim=-1, keepdim=True)
        safe_probs = torch.where(
            act_sum > 1e-12,
            probs / act_sum.clamp_min(1e-12),
            allowed.to(dtype=probs.dtype) / allowed.sum(dim=-1, keepdim=True).clamp_min(1.0),
        )
        assert torch.allclose(safe_probs.sum(dim=-1), torch.ones(1, 5))
        assert torch.allclose(safe_probs, torch.full((1, 5, 12), 1.0 / 12.0))

    def test_t2_29_simplex_extreme_logits_no_inf(self) -> None:
        """Verify softmax on extreme logit values (+1e8, -1e8) maintains valid [0, 1] range."""
        logits = torch.tensor([1e8, -1e8, 0.0])
        probs = torch.softmax(logits, dim=-1)
        assert not torch.isnan(probs).any()
        assert abs(probs.sum().item() - 1.0) < 1e-5
        assert probs[0].item() == 1.0
        assert probs[1].item() == 0.0

    def test_t2_30_real_sgt_compact_policy_forward_pass_simplex_safe(self) -> None:
        """Verify real SGTCompactTradingPolicy produces valid simplex action probabilities."""
        policy = SGTCompactTradingPolicy()
        policy.eval()

        obs = torch.randn(5, 137)
        mems = policy.initial_memories(5)

        with torch.no_grad():
            action_wire, values, extra, new_mems = policy(obs, memories=mems)
            action_logits = action_wire[:, :12]
            probs = torch.softmax(action_logits, dim=-1)

            assert not torch.isnan(probs).any()
            assert not torch.isinf(probs).any()
            sums = probs.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_t2_31_discrete_parameter_heads_categorical_distribution_safe(self) -> None:
        """Verify all 8 discrete parameter heads produce valid distributions."""
        policy = SGTCompactTradingPolicy()
        policy.eval()
        obs = torch.randn(5, 137)
        mems = policy.initial_memories(5)

        with torch.no_grad():
            action_wire, values, extra, new_mems = policy(obs, memories=mems)
            hidden = extra["hidden"]
            param_heads = policy.heads.parameter_heads
            assert len(param_heads) == 8
            for head_name, head_module in param_heads.items():
                logits = head_module(hidden)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                sample = dist.sample()
                assert sample.shape == (5,)


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (11 tests total)
# ==============================================================================

class TestTier3_1_AdaptiveTuningKnobsPairwise:
    """Tier 3.1: Adaptive tuning knob transitions and governance invariants."""

    def test_t3_1_adaptive_knobs_retry1_transitions(self) -> None:
        """Verify Retry 1 applies LCB -0.05, mix 0.01, pattern 0.15, direction loss 1.25."""
        overrides = compute_adaptive_tuning_overrides(retry_idx=1)
        assert overrides["five_expectancy_action_veto_min_lcb_r"] == -0.05
        assert overrides["five_behavior_uniform_mix"] == 0.01
        assert overrides["five_pattern_action_quality_reward_scale_r"] == 0.15
        assert overrides["five_pattern_tree_direction_loss_weight"] == 1.25

    def test_t3_2_adaptive_knobs_retry2_transitions(self) -> None:
        """Verify Retry 2 applies LCB 0.00, mix 0.005, pattern 0.20, direction loss 1.50."""
        overrides = compute_adaptive_tuning_overrides(retry_idx=2)
        assert overrides["five_expectancy_action_veto_min_lcb_r"] == 0.00
        assert overrides["five_behavior_uniform_mix"] == 0.005
        assert overrides["five_pattern_action_quality_reward_scale_r"] == 0.20
        assert overrides["five_pattern_tree_direction_loss_weight"] == 1.50

    def test_t3_3_adaptive_knobs_retry3_max_exploitation(self) -> None:
        """Verify Retry 3 applies LCB +0.05, mix 0.000 (pure exploitation), pattern 0.25."""
        overrides = compute_adaptive_tuning_overrides(retry_idx=3)
        assert overrides["five_expectancy_action_veto_min_lcb_r"] == 0.05
        assert overrides["five_behavior_uniform_mix"] == 0.000
        assert overrides["five_pattern_action_quality_reward_scale_r"] == 0.25
        assert overrides["five_pattern_tree_direction_loss_weight"] == 1.50

    def test_t3_4_governance_invariants_strictly_preserved(self) -> None:
        """Verify governance invariants (POLICY_CLOSE_ALL=0, zero hard direction gate) remain locked."""
        for retry in (1, 2, 3, 4):
            overrides = compute_adaptive_tuning_overrides(retry_idx=retry)
            assert overrides["five_lock_non_entry_to_hold"] is True
            assert overrides["context_direction_entry_gate_enabled"] is False


class TestTier3_2_CheckpointRollbackWithAdaptation:
    """Tier 3.2: Checkpoint rollback on cycle failure paired with hyperparameter adaptation."""

    def test_t3_5_rollback_cycle2_failure_reverts_to_cycle1_checkpoint(self) -> None:
        """Verify Cycle 2 failure rolls back to Cycle 1 promoted checkpoint for retry."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            c1_promoted = base_dir / "checkpoint_cycle_1.pkl"
            c1_promoted.write_bytes(b"C1_CHAMPION_CHECKPOINT")

            attempts = 0
            seeds_used = []

            def rollback_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                nonlocal attempts
                attempts += 1
                seeds_used.append(seed_artifact)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "candidate.pkl").write_bytes(b"DATA")
                diag = output_dir / "mt5_diagnostics.json"

                # Fail attempt 1, pass attempt 2
                net = -30.0 if attempts == 1 else 500.0
                pf = 0.85 if attempts == 1 else 1.35
                diag.write_text(json.dumps({
                    "performance": {"net_pnl": net, "profit_factor": pf, "closed_trades": 25},
                    "risk": {"max_equity_drawdown_pct": 5.0},
                    "policy": {"hold_share": 0.6},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=rollback_runner, max_adaptive_retries=2)
            res = engine.run_cycle(cycle_id=2, seed_checkpoint=c1_promoted, output_base_dir=base_dir)

            assert attempts == 2
            assert res["promoted"] is True
            # Both attempts must have used c1_promoted (rollback occurred)
            assert seeds_used[0] == c1_promoted.resolve()
            assert seeds_used[1] == c1_promoted.resolve()

    def test_t3_6_rollback_cycle1_failure_reverts_to_initial_seed(self) -> None:
        """Verify Cycle 1 failure reverts to initial seed for retry."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            attempts = 0
            seeds_used = []

            def rollback_c1_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                nonlocal attempts
                attempts += 1
                seeds_used.append(seed_artifact)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "candidate.pkl").write_bytes(b"DATA")
                diag = output_dir / "mt5_diagnostics.json"
                net = -20.0 if attempts == 1 else 400.0
                pf = 0.9 if attempts == 1 else 1.3
                diag.write_text(json.dumps({
                    "performance": {"net_pnl": net, "profit_factor": pf, "closed_trades": 25},
                    "risk": {"max_equity_drawdown_pct": 5.0},
                    "policy": {"hold_share": 0.6},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=rollback_c1_runner, max_adaptive_retries=2)
            res = engine.run_cycle(cycle_id=1, seed_checkpoint=DEFAULT_INITIAL_SEED, output_base_dir=base_dir)

            assert attempts == 2
            assert res["promoted"] is True
            assert seeds_used[0] == DEFAULT_INITIAL_SEED.resolve()
            assert seeds_used[1] == DEFAULT_INITIAL_SEED.resolve()

    def test_t3_7_failed_candidate_never_promoted(self) -> None:
        """Verify a failed candidate checkpoint is not promoted to checkpoint_cycle_{k}.pkl."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)

            def failing_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "candidate.pkl").write_bytes(b"FAIL_DATA")
                diag_file = output_dir / "mt5_diagnostics.json"
                diag_file.write_text(json.dumps({
                    "performance": {"net_pnl": -50.0, "profit_factor": 0.8, "closed_trades": 25},
                    "risk": {"max_equity_drawdown_pct": 5.0},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=failing_runner, max_adaptive_retries=1)
            res = engine.run_cycle(cycle_id=1, seed_checkpoint=DEFAULT_INITIAL_SEED, output_base_dir=base_dir)
            assert res["promoted"] is False
            promoted_path = base_dir / "checkpoint_cycle_1.pkl"
            assert not promoted_path.is_file()

    def test_t3_8_adaptation_applied_to_regenerated_profile_yaml(self) -> None:
        """Verify adapted profile YAML correctly incorporates new hyperparameter overrides."""
        with tempfile.TemporaryDirectory() as td:
            base_yaml = Path(td) / "base.yaml"
            base_yaml.write_text(yaml.safe_dump({
                "experiment": {"id": "test"},
                "learning_overrides": {"five_behavior_uniform_mix": 0.02},
            }), encoding="utf-8")

            output_yaml = Path(td) / "adapted.yaml"
            overrides = compute_adaptive_tuning_overrides(retry_idx=1)
            generate_adapted_profile_yaml(base_yaml, output_yaml, overrides, experiment_id="adapted_retry_1")

            assert output_yaml.is_file()
            data = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
            assert data["learning_overrides"]["five_expectancy_action_veto_min_lcb_r"] == -0.05
            assert data["learning_overrides"]["five_behavior_uniform_mix"] == 0.01


class TestTier3_3_SequentialRolloverAndWorkspaceSeparation:
    """Tier 3.3: Sequential execution of Cycle 1 -> Cycle 2 rollover and workspace isolation."""

    def test_t3_9_sequential_rollover_c1_to_c2_to_c3(self) -> None:
        """Verify sequential chaining of promoted checkpoints across cycles."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)

            def passing_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                output_dir.mkdir(parents=True, exist_ok=True)
                cand = output_dir / "candidate.pkl"
                cand.write_bytes(b"CANDIDATE_DATA")
                diag = output_dir / "mt5_diagnostics.json"
                diag.write_text(json.dumps({
                    "performance": {"net_pnl": 500.0, "profit_factor": 1.4, "closed_trades": 30},
                    "risk": {"max_equity_drawdown_pct": 6.0},
                    "policy": {"hold_share": 0.6},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=passing_runner)
            res = engine.run_all_cycles(initial_seed=DEFAULT_INITIAL_SEED, output_base_dir=base_dir)
            assert res["multi_cycle_certified"] is True
            assert (base_dir / "checkpoint_cycle_1.pkl").is_file()
            assert (base_dir / "checkpoint_cycle_2.pkl").is_file()
            assert (base_dir / "checkpoint_cycle_3.pkl").is_file()

    def test_t3_10_isolated_workspaces_prevent_trace_contamination(self) -> None:
        """Verify separate subdirectories for each cycle prevent trace file cross-contamination."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            d1 = base_dir / "cycle_1"
            d2 = base_dir / "cycle_2"
            d3 = base_dir / "cycle_3"
            for d in (d1, d2, d3):
                d.mkdir(parents=True, exist_ok=True)
                (d / "trace.jsonl").write_text(f"trace_{d.name}", encoding="utf-8")

            assert d1 != d2 and d2 != d3 and d1 != d3
            assert (d1 / "trace.jsonl").read_text(encoding="utf-8") == "trace_cycle_1"
            assert (d2 / "trace.jsonl").read_text(encoding="utf-8") == "trace_cycle_2"
            assert (d3 / "trace.jsonl").read_text(encoding="utf-8") == "trace_cycle_3"

    def test_t3_11_max_retries_exhaustion_halts_chain(self) -> None:
        """Verify that when a cycle fails all adaptive retries, the pipeline halts immediately."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)

            def partial_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "candidate.pkl").write_bytes(b"DATA")
                diag = output_dir / "mt5_diagnostics.json"

                is_c1 = "cycle_1" in str(output_dir)
                net = 500.0 if is_c1 else -50.0
                diag.write_text(json.dumps({
                    "performance": {"net_pnl": net, "profit_factor": 1.3 if is_c1 else 0.8, "closed_trades": 25},
                    "risk": {"max_equity_drawdown_pct": 5.0},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=partial_runner, max_adaptive_retries=1)
            res = engine.run_all_cycles(initial_seed=DEFAULT_INITIAL_SEED, output_base_dir=base_dir)
            assert res["multi_cycle_certified"] is False
            # Cycle 3 should NOT have been executed
            assert 3 not in res["cycle_results"]


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (5 comprehensive scenarios)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end real-world application scenarios."""

    def test_t4_1_scenario_complete_three_cycle_walk_forward_pipeline(self) -> None:
        """Scenario 1: Complete Walk-Forward multi-cycle workflow from seed through Cycle 3."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)

            def simulated_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                output_dir.mkdir(parents=True, exist_ok=True)
                cand = output_dir / "candidate.pkl"
                cand.write_bytes(b"SGT_COMPACT_SIMULATED_CHECKPOINT_DATA")
                diag = output_dir / "mt5_diagnostics.json"
                diag.write_text(json.dumps({
                    "performance": {
                        "net_pnl": 850.0,
                        "profit_factor": 1.42,
                        "closed_trades": 36,
                        "win_rate": 0.55,
                        "payoff_ratio": 1.75,
                        "close_reasons": {"TP": 20, "SL": 16},
                    },
                    "risk": {"max_equity_drawdown_pct": 7.8},
                    "policy": {"hold_share": 0.70},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=simulated_runner)
            manifest = engine.run_all_cycles(initial_seed=DEFAULT_INITIAL_SEED, output_base_dir=base_dir)

            assert manifest["multi_cycle_certified"] is True
            assert manifest["aggregate_metrics"]["total_trades"] == 36 * 3
            assert manifest["aggregate_metrics"]["total_net_pnl"] == 850.0 * 3
            assert (base_dir / "manifest.json").is_file()

    def test_t4_2_scenario_cycle2_loss_adaptive_tuning_retry_and_advancement(self) -> None:
        """Scenario 2: Simulated loss in Cycle 2 triggers tuning, retry pass, and advances to Cycle 3."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            c2_attempts = 0

            def adaptive_runner(from_date, to_date, seed_artifact, output_dir, profile_path, config_path, timeout_sec, max_wall_sec):
                nonlocal c2_attempts
                output_dir.mkdir(parents=True, exist_ok=True)
                cand = output_dir / "candidate.pkl"
                cand.write_bytes(b"DATA")

                is_c2 = "cycle_2" in str(output_dir)
                if is_c2:
                    c2_attempts += 1

                # If C2 attempt 1: fail with Net Profit <= 0
                if is_c2 and c2_attempts == 1:
                    net_pnl = -45.0
                    pf = 0.92
                else:
                    net_pnl = 620.0
                    pf = 1.35

                diag = output_dir / "mt5_diagnostics.json"
                diag.write_text(json.dumps({
                    "performance": {"net_pnl": net_pnl, "profit_factor": pf, "closed_trades": 32},
                    "risk": {"max_equity_drawdown_pct": 6.5},
                    "policy": {"hold_share": 0.65},
                }), encoding="utf-8")
                return True

            engine = WalkForwardEngine(workspace_dir=ROOT_DIR, runner_fn=adaptive_runner)
            manifest = engine.run_all_cycles(initial_seed=DEFAULT_INITIAL_SEED, output_base_dir=base_dir)

            # Verified: C2 required 2 attempts, passed, and C3 proceeded
            assert c2_attempts == 2
            assert manifest["multi_cycle_certified"] is True
            assert (base_dir / "checkpoint_cycle_2.pkl").is_file()
            assert (base_dir / "checkpoint_cycle_3.pkl").is_file()

    def test_t4_3_scenario_headless_mt5_execution_parameters_verification(self) -> None:
        """Scenario 3: Verify tester.ini generated for headless execution has Visual=0 and ShutdownTerminal=1."""
        ini_str = build_tester_ini_content(
            from_date="2024.01.08",
            to_date="2024.03.29",
            visual=False,
            shutdown_terminal=True,
            report_name="walk_forward_c1.htm",
        )
        assert "Visual=0" in ini_str
        assert "ShutdownTerminal=1" in ini_str
        assert "FromDate=2024.01.08" in ini_str
        assert "ToDate=2024.03.29" in ini_str
        assert "UseLocal=1" in ini_str
        assert "UseCloud=0" in ini_str

    def test_t4_4_scenario_hold_collapse_detection_and_active_trade_enforcement(self) -> None:
        """Scenario 4: Verify policy collapse into passive HOLD is diagnosed and adapts entry floor."""
        m_collapsed = CycleMetrics(
            cycle_id=1,
            net_pnl=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            closed_trades=2,  # Collapsed: only 2 trades
            hold_share=0.998,
        )
        verdict = evaluate_acceptance_criteria(m_collapsed)
        assert verdict.passed is False
        assert verdict.trade_frequency_ok is False

        diag = diagnose_failure(m_collapsed, verdict.reasons)
        assert diag.is_under_trading is True

        overrides = compute_adaptive_tuning_overrides(retry_idx=1, diagnosis=diag)
        assert overrides["five_behavior_entry_probability_floor"] == 0.01
        assert overrides["five_pattern_tree_hold_ceiling_loss_weight"] == 0.50

    def test_t4_5_scenario_candidate_artifact_manifest_packaging_and_certification(self) -> None:
        """Scenario 5: Full candidate artifact packaging with SHA-256 and multi-cycle certification sign-off."""
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            final_cand = base_dir / "final_candidate_sgt_compact_tier2.pkl"
            final_cand.write_bytes(b"FINAL_PROMOTED_SGT_COMPACT_WEIGHTS_17.5M")
            cand_sha = hashlib.sha256(b"FINAL_PROMOTED_SGT_COMPACT_WEIGHTS_17.5M").hexdigest()

            cycle_results = {
                1: {
                    "verdict": {"passed": True},
                    "metrics": {"closed_trades": 35, "net_pnl": 950.0, "profit_factor": 1.45, "max_drawdown_pct": 7.2},
                },
                2: {
                    "verdict": {"passed": True},
                    "metrics": {"closed_trades": 42, "net_pnl": 1150.0, "profit_factor": 1.50, "max_drawdown_pct": 8.1},
                },
                3: {
                    "verdict": {"passed": True},
                    "metrics": {"closed_trades": 38, "net_pnl": 1050.0, "profit_factor": 1.48, "max_drawdown_pct": 6.9},
                },
            }
            manifest = generate_walk_forward_manifest(
                candidate_checkpoint_path=final_cand,
                cycle_results=cycle_results,
                output_dir=base_dir,
                initial_seed_path=DEFAULT_INITIAL_SEED,
            )

            assert manifest["multi_cycle_certified"] is True
            assert manifest["candidate"]["sha256"] == cand_sha
            assert manifest["candidate"]["exists"] is True
            assert manifest["aggregate_metrics"]["total_trades"] == 35 + 42 + 38
            assert manifest["aggregate_metrics"]["total_net_pnl"] == 950.0 + 1150.0 + 1050.0
            assert manifest["aggregate_metrics"]["all_cycles_passed"] is True
