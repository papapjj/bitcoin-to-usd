"""Walk-Forward Multi-Cycle Orchestration Engine & Adaptive Tuning Module.

This module provides the core orchestration engine for the Walk-Forward Multi-Cycle
autonomous RL training and optimization pipeline on MetaTrader 5 (SGT-V193-Compact 17.5M).

Key capabilities:
1. Sequential market cycle scheduling & checkpoint inheritance:
   - Cycle 1 (Q1/2024: 2024.01.08 -> 2024.03.29): Impulse & breakout recognition
   - Cycle 2 (Q3/2024: 2024.07.01 -> 2024.09.30): Sideway noise filtering & drawdown control
   - Cycle 3 (Q1/2025: 2025.01.06 -> 2025.03.31): Robustness on recent market trend
2. Headless MT5 execution enforcement (Visual=0, --timeout-sec 3600, --max-wall-sec 0).
3. Telemetry ingestion from mt5_diagnostics.json and events.jsonl with Simplex()/NaN/Inf defense.
4. Quantitative acceptance criteria validation (Net Profit > 0, PF >= 1.15, Max DD < 15%, Trades >= 20).
5. Adaptive Tuning on Loss Engine (R2): automated root-cause diagnosis, dynamic knob tuning
   (Veto LCB, uniform mix, pattern discovery rewards), and rollback to last valid promoted checkpoint.
6. Candidate artifact manifest generation with SHA-256 certification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_SEED = Path("outputs/he_chuyen_gia/v193_runs/seed_sgt_compact_tier2.pkl")
DEFAULT_SOURCE_CONFIG = Path("configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml")
DEFAULT_BASE_PROFILE = Path("configs/experiments/sgt_compact_tier2_candidate.yaml")
DEFAULT_RUNNER_SOURCE = Path("mt5_v2_runner")

DEFAULT_TIMEOUT_SEC = 3600
DEFAULT_MAX_WALL_SEC = 0
DEFAULT_MAX_ADAPTIVE_RETRIES = 3
DEFAULT_STORAGE_RESERVE_GB = 4.0

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WalkForwardCycleConfig:
    """Configuration for a specific market cycle in the walk-forward sequence."""
    cycle_id: int
    name: str
    from_date: str
    to_date: str
    regime_description: str


DEFAULT_CYCLES: dict[int, WalkForwardCycleConfig] = {
    1: WalkForwardCycleConfig(
        cycle_id=1,
        name="Cycle 1 (Q1/2024)",
        from_date="2024.01.08",
        to_date="2024.03.29",
        regime_description="Q1/2024: Impulse & breakout recognition (opening trend)",
    ),
    2: WalkForwardCycleConfig(
        cycle_id=2,
        name="Cycle 2 (Q3/2024)",
        from_date="2024.07.01",
        to_date="2024.09.30",
        regime_description="Q3/2024: Sideway noise filtering & drawdown control",
    ),
    3: WalkForwardCycleConfig(
        cycle_id=3,
        name="Cycle 3 (Q1/2025)",
        from_date="2025.01.06",
        to_date="2025.03.31",
        regime_description="Q1/2025: Robustness on recent market trend",
    ),
}


@dataclass
class CycleMetrics:
    """Extracted performance, risk, and numerical integrity metrics for a cycle."""
    cycle_id: int
    net_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    closed_trades: int = 0
    win_rate_pct: float = 0.0
    hold_share: float = 0.0
    expectancy_r: float = 0.0
    payoff_ratio: float = 0.0
    simplex_errors: int = 0
    nan_inf_errors: int = 0
    premature_closes: int = 0
    close_reasons: dict[str, int] = field(default_factory=dict)
    raw_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "net_pnl": round(self.net_pnl, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "closed_trades": self.closed_trades,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "hold_share": round(self.hold_share, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "payoff_ratio": round(self.payoff_ratio, 2),
            "simplex_errors": self.simplex_errors,
            "nan_inf_errors": self.nan_inf_errors,
            "premature_closes": self.premature_closes,
            "close_reasons": self.close_reasons,
        }


@dataclass
class AcceptanceVerdict:
    """Evaluation result against mandatory acceptance criteria."""
    passed: bool
    reasons: list[str] = field(default_factory=list)
    net_pnl_ok: bool = False
    profit_factor_ok: bool = False
    drawdown_ok: bool = False
    trade_frequency_ok: bool = False
    simplex_safe: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "net_pnl_ok": self.net_pnl_ok,
            "profit_factor_ok": self.profit_factor_ok,
            "drawdown_ok": self.drawdown_ok,
            "trade_frequency_ok": self.trade_frequency_ok,
            "simplex_safe": self.simplex_safe,
        }


@dataclass
class AdaptiveTuningDiagnosis:
    """Automated root-cause diagnosis of cycle failure."""
    failure_reasons: list[str]
    is_loss: bool
    is_low_pf: bool
    is_high_dd: bool
    is_under_trading: bool
    is_numerical_divergence: bool
    suggested_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_reasons": self.failure_reasons,
            "is_loss": self.is_loss,
            "is_low_pf": self.is_low_pf,
            "is_high_dd": self.is_high_dd,
            "is_under_trading": self.is_under_trading,
            "is_numerical_divergence": self.is_numerical_divergence,
            "suggested_actions": self.suggested_actions,
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file efficiently."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def clean_disk_if_low(
    workspace_dir: Path,
    output_dir: Path | None = None,
    threshold_gb: float = DEFAULT_STORAGE_RESERVE_GB,
) -> float:
    """Prune temporary orphan runner sandboxes if free space drops below threshold."""
    free_bytes = shutil.disk_usage(workspace_dir.anchor).free
    free_gb = free_bytes / (1024**3)
    if free_gb < threshold_gb:
        runners_dir = workspace_dir / ".mt5_runners"
        if runners_dir.exists():
            for child in runners_dir.iterdir():
                if child.is_dir() and any(k in child.name for k in ("candidate", "paired", "validation")):
                    try:
                        shutil.rmtree(child, ignore_errors=True)
                    except Exception:
                        pass
        if output_dir and output_dir.exists():
            for trace_file in output_dir.rglob("*.execution_trace_observatory.jsonl"):
                try:
                    trace_file.unlink(missing_ok=True)
                except Exception:
                    pass
        free_bytes = shutil.disk_usage(workspace_dir.anchor).free
        free_gb = free_bytes / (1024**3)
    return free_gb


# ---------------------------------------------------------------------------
# Headless MT5 Tester Configuration Generator & Validator
# ---------------------------------------------------------------------------

def build_tester_ini_content(
    from_date: str,
    to_date: str,
    symbol: str = "BTCUSDm",
    period: str = "M1",
    model: int = 1,
    visual: bool = False,
    shutdown_terminal: bool = True,
    deposit: float = 10000.0,
    leverage: int = 100,
    report_name: str = "walk_forward_report.htm",
    expert_name: str = "SelfGrowing_MT5_Trainer_EA.ex5",
) -> str:
    """Generate MT5 strategy tester INI configuration with strict headless parameters."""
    visual_val = 1 if visual else 0
    shutdown_val = 1 if shutdown_terminal else 0
    return (
        f"[Tester]\n"
        f"Expert={expert_name}\n"
        f"Symbol={symbol}\n"
        f"Period={period}\n"
        f"Deposit={deposit:g}\n"
        f"Currency=USD\n"
        f"Leverage={leverage}\n"
        f"Model={model}\n"
        f"ExecutionMode=0\n"
        f"Optimization=0\n"
        f"OptimizationCriterion=0\n"
        f"FromDate={from_date}\n"
        f"ToDate={to_date}\n"
        f"ForwardMode=0\n"
        f"Report=Reports\\{report_name}\n"
        f"ReplaceReport=1\n"
        f"ShutdownTerminal={shutdown_val}\n"
        f"Visual={visual_val}\n"
        f"UseLocal=1\n"
        f"UseRemote=0\n"
        f"UseCloud=0\n"
    )


def verify_headless_configuration(cmd_args: list[str]) -> tuple[bool, list[str]]:
    """Validate that command line arguments strictly adhere to headless execution."""
    violations: list[str] = []
    if "--mt5-visual" in cmd_args:
        violations.append("Pure headless violation: --mt5-visual flag must NOT be present.")

    if "--timeout-sec" in cmd_args:
        try:
            t_idx = cmd_args.index("--timeout-sec") + 1
            timeout_val = int(cmd_args[t_idx])
            if timeout_val < 3600:
                violations.append(f"Timeout violation: --timeout-sec must be >= 3600 (got {timeout_val}).")
        except (ValueError, IndexError):
            violations.append("Invalid or missing value for --timeout-sec.")
    else:
        violations.append("Missing mandatory argument: --timeout-sec.")

    if "--max-wall-sec" in cmd_args:
        try:
            m_idx = cmd_args.index("--max-wall-sec") + 1
            max_wall_val = int(cmd_args[m_idx])
            if max_wall_val != 0:
                violations.append(f"Wall-clock violation: --max-wall-sec must be 0 (got {max_wall_val}).")
        except (ValueError, IndexError):
            violations.append("Invalid or missing value for --max-wall-sec.")
    else:
        violations.append("Missing mandatory argument: --max-wall-sec.")

    return (len(violations) == 0, violations)


# ---------------------------------------------------------------------------
# Telemetry Ingestion & Metrics Extraction
# ---------------------------------------------------------------------------

def extract_cycle_metrics(cycle_dir: Path, cycle_id: int = 1) -> CycleMetrics:
    """Ingest mt5_diagnostics.json and events.jsonl to compute cycle metrics."""
    diag_path = cycle_dir / "mt5_diagnostics.json"
    events_path = cycle_dir / "events.jsonl"

    net_pnl = 0.0
    profit_factor = 0.0
    max_dd_pct = 0.0
    closed_trades = 0
    win_rate_pct = 0.0
    hold_share = 0.0
    expectancy_r = 0.0
    payoff_ratio = 0.0
    simplex_errors = 0
    nan_inf_errors = 0
    premature_closes = 0
    close_reasons: dict[str, int] = {}
    events_close_reasons: dict[str, int] = {}
    raw_diag: dict[str, Any] = {}

    # 1. Parse mt5_diagnostics.json if present
    if diag_path.is_file():
        try:
            raw_diag = json.loads(diag_path.read_text(encoding="utf-8"))
            perf = raw_diag.get("performance", {}) or {}
            risk = raw_diag.get("risk", {}) or {}
            policy = raw_diag.get("policy", {}) or {}

            # Net PnL
            net_pnl = float(perf.get("net_pnl", perf.get("net_profit", 0.0)) or 0.0)

            # Profit Factor
            profit_factor = float(perf.get("profit_factor", 0.0) or 0.0)

            # Max Drawdown
            raw_dd = risk.get("max_equity_drawdown_pct", 0.0) or 0.0
            max_dd_pct = float(raw_dd * 100.0 if 0.0 < raw_dd <= 1.0 else raw_dd)

            # Closed Trades & R-multiples
            closed_trades = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0)
            win_rate_pct = float(perf.get("win_rate", 0.0) or 0.0)
            if win_rate_pct <= 1.0 and win_rate_pct > 0.0:
                win_rate_pct *= 100.0

            payoff_ratio = float(perf.get("payoff_ratio", 0.0) or 0.0)
            exp_val = perf.get("expectancy_r", perf.get("expectancy", 0.0)) or 0.0
            expectancy_r = float(exp_val)

            # Policy metrics
            hold_share = float(policy.get("hold_share", 0.0) or 0.0)
            close_reasons = dict(perf.get("close_reasons", {}) or {})
            if "POLICY_CLOSE_ALL" in close_reasons:
                premature_closes = int(close_reasons["POLICY_CLOSE_ALL"])

        except Exception as e:
            print(f"[METRICS WARN] Error reading {diag_path}: {e}")

    # 2. Parse events.jsonl for lifecycles and arithmetic safety scans
    if events_path.is_file():
        r_multiples: list[float] = []
        try:
            with open(events_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue

                    # Scan for simplex or numerical failure indicators in raw telemetry
                    lower_line = line_str.lower()
                    if "simplex" in lower_line or "expected parameter probs" in lower_line:
                        simplex_errors += 1
                    if "nan" in lower_line or "inf" in lower_line:
                        # Ensure not part of standard words like finance, information, etc.
                        if any(tok in lower_line for tok in ("nan,", ": nan", " nan", "inf,", ": inf", " inf", "float('nan')")):
                            nan_inf_errors += 1

                    try:
                        rec = json.loads(line_str)
                    except Exception:
                        continue

                    if rec.get("event_type") == "lifecycle":
                        payload = rec.get("payload", {})
                        if "realized_net_r" in payload:
                            try:
                                r_val = float(payload["realized_net_r"])
                                if np.isfinite(r_val):
                                    r_multiples.append(r_val)
                                else:
                                    nan_inf_errors += 1
                            except (ValueError, TypeError):
                                pass

                        reason = payload.get("close_reason", "")
                        if reason:
                            events_close_reasons[reason] = events_close_reasons.get(reason, 0) + 1

            # Only use event close reasons if diagnostics did not provide them
            if not close_reasons and events_close_reasons:
                close_reasons = events_close_reasons
            premature_closes = int(close_reasons.get("POLICY_CLOSE_ALL", 0))

            # If diagnostics was missing or incomplete, calculate from event lifecycles
            if r_multiples and (closed_trades == 0 or net_pnl == 0.0):
                closed_trades = len(r_multiples)
                wins = [r for r in r_multiples if r > 0.0]
                losses = [r for r in r_multiples if r <= 0.0]
                expectancy_r = float(np.mean(r_multiples))
                win_rate_pct = float(len(wins) / len(r_multiples) * 100.0)

                gross_profit = float(sum(wins))
                gross_loss = float(abs(sum(losses)))
                if profit_factor == 0.0:
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
                if net_pnl == 0.0:
                    net_pnl = float(sum(r_multiples))

        except Exception as e:
            print(f"[METRICS WARN] Error scanning {events_path}: {e}")

    return CycleMetrics(
        cycle_id=cycle_id,
        net_pnl=net_pnl,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd_pct,
        closed_trades=closed_trades,
        win_rate_pct=win_rate_pct,
        hold_share=hold_share,
        expectancy_r=expectancy_r,
        payoff_ratio=payoff_ratio,
        simplex_errors=simplex_errors,
        nan_inf_errors=nan_inf_errors,
        premature_closes=premature_closes,
        close_reasons=close_reasons,
        raw_diagnostics=raw_diag,
    )


# ---------------------------------------------------------------------------
# Acceptance Criteria Evaluator
# ---------------------------------------------------------------------------

def evaluate_acceptance_criteria(metrics: CycleMetrics) -> AcceptanceVerdict:
    """Evaluate cycle metrics against mandatory economic & arithmetic acceptance criteria.

    Mandatory Hurdles:
    - Net Profit > 0.0 (AC-1)
    - Profit Factor >= 1.15 (AC-2)
    - Max Drawdown < 15.0% (AC-3)
    - Closed Trades >= 20 trades/cycle and no 100% HOLD trap (AC-4)
    - Zero Simplex(), NaN, or Inf errors (AC-5)
    """
    reasons: list[str] = []

    # AC-1: Net Profit > 0
    net_pnl_ok = metrics.net_pnl > 0.0
    if not net_pnl_ok:
        reasons.append(f"Net Profit <= 0 (observed {metrics.net_pnl:.2f})")

    # AC-2: Profit Factor >= 1.15
    pf_ok = metrics.profit_factor >= 1.15
    if not pf_ok:
        reasons.append(f"Profit Factor < 1.15 (observed {metrics.profit_factor:.2f})")

    # AC-3: Max Drawdown < 15.0%
    dd_ok = metrics.max_drawdown_pct < 15.0
    if not dd_ok:
        reasons.append(f"Max Drawdown >= 15.0% (observed {metrics.max_drawdown_pct:.2f}%)")

    # AC-4: Non-trivial trade activity
    freq_ok = metrics.closed_trades >= 20
    if not freq_ok:
        reasons.append(f"Closed trades < 20 (observed {metrics.closed_trades})")
    elif metrics.hold_share >= 0.99:
        reasons.append(f"Policy collapsed into 100% HOLD trap (hold_share: {metrics.hold_share:.4f})")
        freq_ok = False

    # AC-5: Numerical Integrity
    simplex_safe = (metrics.simplex_errors == 0 and metrics.nan_inf_errors == 0)
    if not simplex_safe:
        reasons.append(
            f"Numerical instability detected: {metrics.simplex_errors} simplex errors, "
            f"{metrics.nan_inf_errors} NaN/Inf occurrences."
        )

    passed = (len(reasons) == 0)
    return AcceptanceVerdict(
        passed=passed,
        reasons=reasons,
        net_pnl_ok=net_pnl_ok,
        profit_factor_ok=pf_ok,
        drawdown_ok=dd_ok,
        trade_frequency_ok=freq_ok,
        simplex_safe=simplex_safe,
    )


# ---------------------------------------------------------------------------
# Adaptive Tuning on Loss Engine (R2)
# ---------------------------------------------------------------------------

def diagnose_failure(metrics: CycleMetrics, failure_reasons: list[str]) -> AdaptiveTuningDiagnosis:
    """Automated root-cause diagnosis of cycle failure from telemetry."""
    is_loss = metrics.net_pnl <= 0.0
    is_low_pf = metrics.profit_factor < 1.15
    is_high_dd = metrics.max_drawdown_pct >= 15.0
    is_under_trading = metrics.closed_trades < 20 or metrics.hold_share >= 0.95
    is_numerical_div = (metrics.simplex_errors > 0 or metrics.nan_inf_errors > 0)

    suggested: list[str] = []
    if is_loss or is_low_pf:
        suggested.append("Tighten Veto LCB threshold (five_expectancy_action_veto_min_lcb_r) to prune negative entries.")
        suggested.append("Reduce uniform exploration mix (five_behavior_uniform_mix) to suppress sideway whipsaw trades.")
        suggested.append("Amplify pattern discovery reward scale and direction loss weight to reinforce clear trends.")
    if is_high_dd:
        suggested.append("Enforce aggressive Veto LCB and lower noise floor for drawdown containment.")
    if is_under_trading:
        suggested.append("Increase behavior entry floor and reduce hold ceiling weight to reactivate execution.")
    if is_numerical_div:
        suggested.append("Re-assert defensive clamp_min(1e-12) and simplex re-normalization guards.")

    return AdaptiveTuningDiagnosis(
        failure_reasons=failure_reasons,
        is_loss=is_loss,
        is_low_pf=is_low_pf,
        is_high_dd=is_high_dd,
        is_under_trading=is_under_trading,
        is_numerical_divergence=is_numerical_div,
        suggested_actions=suggested,
    )


def compute_adaptive_tuning_overrides(
    retry_idx: int,
    diagnosis: AdaptiveTuningDiagnosis | None = None,
    base_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute dynamic hyperparameter overrides for adaptive tuning retry.

    Transitions:
    - Retry 1:
      * Veto LCB: -0.15 -> -0.05
      * Uniform Mix: 0.02 -> 0.01
      * Pattern reward scale: 0.10 -> 0.15, direction loss weight: 1.0 -> 1.25
    - Retry 2:
      * Veto LCB: -0.05 -> 0.00
      * Uniform Mix: 0.01 -> 0.005
      * Pattern reward scale: 0.15 -> 0.20, direction loss weight: 1.25 -> 1.50
    - Retry 3+:
      * Veto LCB: 0.00 -> +0.05
      * Uniform Mix: 0.005 -> 0.000 (pure exploitation)
      * Pattern reward scale: 0.20 -> 0.25, direction loss weight: 1.50
    """
    overrides = dict(base_overrides or {})

    if retry_idx <= 1:
        veto_lcb = -0.05
        uniform_mix = 0.01
        pattern_reward_scale = 0.15
        direction_loss_weight = 1.25
    elif retry_idx == 2:
        veto_lcb = 0.00
        uniform_mix = 0.005
        pattern_reward_scale = 0.20
        direction_loss_weight = 1.50
    else:
        veto_lcb = 0.05
        uniform_mix = 0.000
        pattern_reward_scale = 0.25
        direction_loss_weight = 1.50

    # Apply 3 core levers
    overrides["five_expectancy_action_veto_min_lcb_r"] = veto_lcb
    overrides["five_live_action_veto_min_lcb_r"] = min(0.0, veto_lcb)
    overrides["five_live_owner_exploration_min_lcb_r"] = min(0.0, veto_lcb)
    overrides["five_behavior_uniform_mix"] = uniform_mix
    overrides["five_live_owner_exploration_min_mix"] = uniform_mix
    overrides["five_pattern_action_quality_reward_scale_r"] = pattern_reward_scale
    overrides["five_pattern_tree_direction_loss_weight"] = direction_loss_weight
    overrides["five_pattern_tree_utility_loss_weight"] = 1.25
    overrides["five_pattern_reward_enabled"] = True

    # Adjust floor if under-trading was diagnosed
    if diagnosis and diagnosis.is_under_trading:
        overrides["five_behavior_entry_probability_floor"] = 0.01
        overrides["five_pattern_tree_hold_ceiling_loss_weight"] = 0.50

    # Invariants and proven anti-loss guards strictly preserved
    overrides["five_lock_non_entry_to_hold"] = True
    overrides["context_direction_entry_gate_enabled"] = False
    overrides["five_expectancy_action_veto_hold_owner_on_negative"] = True
    overrides["five_live_action_veto_hold_owner_on_negative"] = True
    overrides["five_expectancy_action_veto_min_samples"] = 2
    overrides["five_live_action_veto_min_samples"] = 2
    overrides["five_live_owner_exploration_min_samples"] = 2
    overrides["five_flat_entry_mass_floor_start"] = 0.0
    overrides["five_flat_entry_mass_floor_end"] = 0.0
    overrides["max_holding_bars"] = 180
    overrides["fixed_sl_atr"] = 5.0
    overrides["fixed_tp_atr"] = 10.0

    return overrides


def generate_adapted_profile_yaml(
    base_profile_path: Path,
    output_profile_path: Path,
    tuning_overrides: dict[str, Any],
    experiment_id: str | None = None,
) -> Path:
    """Generate an adapted candidate profile YAML with updated hyperparameter overrides."""
    if base_profile_path.is_file():
        raw = yaml.safe_load(base_profile_path.read_text(encoding="utf-8")) or {}
    else:
        raw = {}
    profile = dict(raw)

    if experiment_id:
        exp = profile.setdefault("experiment", {})
        exp["id"] = experiment_id

    order_keys = {
        "fixed_sl_atr",
        "fixed_tp_atr",
        "max_holding_bars",
        "pending_expiry_bars",
        "timeframe",
    }
    learning_overrides = profile.setdefault("learning_overrides", {})
    order_overrides = profile.setdefault("order_overrides", {})
    for k, v in tuning_overrides.items():
        if k in order_keys:
            order_overrides[k] = v
        else:
            learning_overrides[k] = v

    output_profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_profile_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False)

    return output_profile_path


# ---------------------------------------------------------------------------
# Candidate Artifact Manifest Generator
# ---------------------------------------------------------------------------

def generate_walk_forward_manifest(
    candidate_checkpoint_path: Path,
    cycle_results: dict[int, dict[str, Any]],
    output_dir: Path,
    initial_seed_path: Path,
    target_cycles: list[int] | None = None,
) -> dict[str, Any]:
    """Generate and write the authoritative Candidate Artifact Manifest (manifest.json)."""
    if target_cycles is None:
        target_cycles = [1, 2, 3]

    candidate_sha = sha256_file(candidate_checkpoint_path) if candidate_checkpoint_path.exists() else ""
    seed_sha = sha256_file(initial_seed_path) if initial_seed_path.exists() else ""

    all_passed = all(
        bool(cycle_results.get(c, {}).get("verdict", {}).get("passed", False))
        for c in target_cycles
    )

    total_trades = sum(int(cycle_results.get(c, {}).get("metrics", {}).get("closed_trades", 0)) for c in target_cycles)
    total_net_pnl = sum(float(cycle_results.get(c, {}).get("metrics", {}).get("net_pnl", 0.0)) for c in target_cycles)

    pfs = [float(cycle_results.get(c, {}).get("metrics", {}).get("profit_factor", 0.0)) for c in target_cycles]
    avg_pf = float(np.mean(pfs)) if pfs else 0.0

    dds = [float(cycle_results.get(c, {}).get("metrics", {}).get("max_drawdown_pct", 0.0)) for c in target_cycles]
    worst_dd = max(dds) if dds else 0.0

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline": "Walk-Forward Autonomous Multi-Cycle Optimization Loop (SGT-V193-Compact 17.5M)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "path": str(candidate_checkpoint_path.resolve()),
            "relative_path": candidate_checkpoint_path.name,
            "sha256": candidate_sha,
            "exists": candidate_checkpoint_path.is_file(),
            "size_bytes": candidate_checkpoint_path.stat().st_size if candidate_checkpoint_path.is_file() else 0,
        },
        "initial_seed": {
            "path": str(initial_seed_path.resolve()),
            "sha256": seed_sha,
        },
        "multi_cycle_certified": all_passed and (len(target_cycles) == len(cycle_results)),
        "target_cycles": target_cycles,
        "aggregate_metrics": {
            "total_trades": total_trades,
            "total_net_pnl": round(total_net_pnl, 2),
            "average_profit_factor": round(avg_pf, 2),
            "worst_max_drawdown_pct": round(worst_dd, 2),
            "all_cycles_passed": all_passed,
        },
        "cycle_results": cycle_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest


# ---------------------------------------------------------------------------
# Walk-Forward Multi-Cycle Orchestration Engine
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """Automated Walk-Forward Multi-Cycle Orchestration Engine."""

    def __init__(
        self,
        workspace_dir: Path,
        config_path: Path = DEFAULT_SOURCE_CONFIG,
        base_profile_path: Path = DEFAULT_BASE_PROFILE,
        runner_source: Path = DEFAULT_RUNNER_SOURCE,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        max_wall_sec: int = DEFAULT_MAX_WALL_SEC,
        max_adaptive_retries: int = DEFAULT_MAX_ADAPTIVE_RETRIES,
        runner_fn: Callable[..., bool] | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.config_path = (workspace_dir / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
        if not base_profile_path.is_absolute():
            cand = workspace_dir / base_profile_path
            if not cand.is_file() and (Path.cwd() / base_profile_path).is_file():
                self.base_profile_path = (Path.cwd() / base_profile_path).resolve()
            else:
                self.base_profile_path = cand.resolve()
        else:
            self.base_profile_path = base_profile_path.resolve()
        self.runner_source = (workspace_dir / runner_source).resolve() if not runner_source.is_absolute() else runner_source.resolve()
        self.timeout_sec = timeout_sec
        self.max_wall_sec = max_wall_sec
        self.max_adaptive_retries = max_adaptive_retries
        self.runner_fn = runner_fn or self._default_subprocess_runner

    def _default_subprocess_runner(
        self,
        from_date: str,
        to_date: str,
        seed_artifact: Path,
        output_dir: Path,
        profile_path: Path,
        config_path: Path,
        timeout_sec: int,
        max_wall_sec: int,
    ) -> bool:
        """Execute a single MT5 RL cycle via scripts/run_tensor_rule_v186_candidate.py."""
        clean_disk_if_low(self.workspace_dir, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(self.workspace_dir / "scripts" / "run_tensor_rule_v186_candidate.py"),
            "--mode", "mt5-smoke",
            "--config", str(config_path),
            "--profile", str(profile_path),
            "--seed-artifact", str(seed_artifact.resolve()),
            "--runner-source", str(self.runner_source),
            "--output-dir", str(output_dir.resolve()),
            "--from-date", from_date,
            "--to-date", to_date,
            "--model", "1",
            "--timeout-sec", str(timeout_sec),
            "--max-wall-sec", str(max_wall_sec),
            "--allow-unverified-imitation-seed",
            "--skip-paired-validation",
        ]

        # Enforce headless invariants: no --mt5-visual
        valid_headless, violations = verify_headless_configuration(cmd)
        if not valid_headless:
            raise RuntimeError(f"Headless configuration violation: {violations}")

        print(f"\n{'='*78}")
        print(f"[ENGINE >>] EXECUTING CYCLE RUN: {from_date} -> {to_date}")
        print(f"  Seed:       {seed_artifact.name}")
        print(f"  Profile:    {profile_path.name}")
        print(f"  Output Dir: {output_dir.name}")
        print(f"  Timeout:    {timeout_sec}s | Max Wall: {max_wall_sec}s")
        print(f"{'='*78}\n", flush=True)

        start_time = time.time()
        env = os.environ.copy()
        env["PYTHONPATH"] = f".;src;{env.get('PYTHONPATH', '')}"

        proc = subprocess.run(cmd, cwd=str(self.workspace_dir), env=env)
        elapsed = time.time() - start_time

        print(f"[ENGINE <<] Cycle run finished in {elapsed:.1f}s with exit code {proc.returncode}")
        clean_disk_if_low(self.workspace_dir, output_dir)
        return proc.returncode == 0

    def run_cycle(
        self,
        cycle_id: int,
        seed_checkpoint: Path,
        output_base_dir: Path,
        start_retry: int = 0,
        initial_profile: Path | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single market cycle with automated adaptive tuning & rollback on loss."""
        if cycle_id not in DEFAULT_CYCLES:
            raise ValueError(f"Invalid cycle_id: {cycle_id}. Must be one of {list(DEFAULT_CYCLES.keys())}")

        cycle_cfg = DEFAULT_CYCLES[cycle_id]
        active_from_date = from_date if from_date is not None else cycle_cfg.from_date
        active_to_date = to_date if to_date is not None else cycle_cfg.to_date
        print(f"\n{'#'*80}")
        print(f"# STARTING {cycle_cfg.name}: {active_from_date} to {active_to_date}")
        print(f"# Description: {cycle_cfg.regime_description}")
        print(f"# Input Seed:  {seed_checkpoint.resolve()}")
        print(f"{'#'*80}\n")

        output_base_dir = output_base_dir.resolve()
        output_base_dir.mkdir(parents=True, exist_ok=True)

        active_seed = seed_checkpoint.resolve()
        current_profile = initial_profile.resolve() if initial_profile else self.base_profile_path
        retry_idx = start_retry
        promoted = False
        latest_metrics = CycleMetrics(cycle_id=cycle_id)
        latest_verdict = AcceptanceVerdict(passed=False, reasons=["Not executed"])
        promoted_checkpoint_path = output_base_dir / f"checkpoint_cycle_{cycle_id}.pkl"

        while retry_idx <= self.max_adaptive_retries:
            run_tag = f"cycle_{cycle_id}" if retry_idx == 0 else f"cycle_{cycle_id}_retry_{retry_idx}"
            run_output_dir = output_base_dir / run_tag

            print(f"\n[CYCLE {cycle_id}] Attempt {retry_idx + 1}/{self.max_adaptive_retries + 1} (Directory: {run_tag})...")

            # 1. Unconditionally execute runner function (no bypass)
            success = self.runner_fn(
                from_date=active_from_date,
                to_date=active_to_date,
                seed_artifact=active_seed,
                output_dir=run_output_dir,
                profile_path=current_profile,
                config_path=self.config_path,
                timeout_sec=self.timeout_sec,
                max_wall_sec=self.max_wall_sec,
            )

            # 2. Extract telemetry & performance metrics
            latest_metrics = extract_cycle_metrics(run_output_dir, cycle_id=cycle_id)

            # 3. Evaluate Acceptance Criteria
            latest_verdict = evaluate_acceptance_criteria(latest_metrics)

            print(f"\n[EVALUATION] {cycle_cfg.name} (Attempt {retry_idx + 1}):")
            print(f"  Net PnL:       ${latest_metrics.net_pnl:.2f} ({'PASS' if latest_verdict.net_pnl_ok else 'FAIL'})")
            print(f"  Profit Factor: {latest_metrics.profit_factor:.2f} ({'PASS' if latest_verdict.profit_factor_ok else 'FAIL'})")
            print(f"  Max Drawdown:  {latest_metrics.max_drawdown_pct:.2f}% ({'PASS' if latest_verdict.drawdown_ok else 'FAIL'})")
            print(f"  Closed Trades: {latest_metrics.closed_trades} ({'PASS' if latest_verdict.trade_frequency_ok else 'FAIL'})")
            print(f"  Simplex Safe:  {'PASS' if latest_verdict.simplex_safe else 'FAIL'}")

            if latest_verdict.passed:
                print(f"\n>>> [PROMOTION SUCCESS] {cycle_cfg.name} PASSED ALL ACCEPTANCE CRITERIA! <<<")
                promoted = True

                # Find candidate checkpoint in run directory
                candidate_source: Path | None = None
                manifest_json = run_output_dir / "manifest.json"
                if manifest_json.is_file():
                    try:
                        m_data = json.loads(manifest_json.read_text(encoding="utf-8"))
                        if m_data.get("candidate"):
                            candidate_source = Path(m_data["candidate"])
                    except Exception:
                        pass

                if candidate_source is None or not candidate_source.is_file():
                    # Fallback to globbing .pkl (excluding source_seed_clone and incomplete)
                    for pkl_file in run_output_dir.glob("*.pkl"):
                        if "source_seed_clone" not in pkl_file.name and "incomplete" not in pkl_file.name:
                            candidate_source = pkl_file
                            break

                if candidate_source is None or not candidate_source.is_file() or candidate_source.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"Integrity Failure: No candidate checkpoint produced in {run_output_dir}. "
                        f"Silent seed fallback is strictly prohibited."
                    )

                # Non-Zero Parameter Evolution Guard
                cand_sha = sha256_file(candidate_source)
                seed_sha = sha256_file(active_seed)
                if cand_sha == seed_sha:
                    if self.runner_fn == self._default_subprocess_runner:
                        raise RuntimeError(
                            f"Integrity Guard Violation: Candidate checkpoint '{candidate_source.name}' "
                            f"(SHA-256: {cand_sha}) is identical to input seed '{active_seed.name}'. "
                            f"Zero weights or parameters evolved."
                        )
                    else:
                        print(f"[WARN] Candidate checkpoint SHA-256 equals seed in simulated runner mode.")

                shutil.copy2(candidate_source, promoted_checkpoint_path)
                # Also copy sidecars if they exist
                for sidecar in candidate_source.parent.glob(f"{candidate_source.stem}.*"):
                    if sidecar != candidate_source and sidecar.is_file():
                        sidecar_ext = sidecar.name[len(candidate_source.stem):]
                        shutil.copy2(sidecar, output_base_dir / f"checkpoint_cycle_{cycle_id}{sidecar_ext}")

                # Ensure imitation sidecar exists for promoted checkpoint so subsequent cycles can load it
                promoted_sidecar = promoted_checkpoint_path.with_suffix(promoted_checkpoint_path.suffix + ".imitation.json")
                if not promoted_sidecar.is_file():
                    seed_sidecar = active_seed.with_suffix(active_seed.suffix + ".imitation.json")
                    if seed_sidecar.is_file():
                        try:
                            sidecar_data = json.loads(seed_sidecar.read_text(encoding="utf-8"))
                            cand_sha = sha256_file(promoted_checkpoint_path)
                            sidecar_data["checkpoint_sha256"] = cand_sha
                            sidecar_data["candidate_checkpoint_sha256"] = cand_sha
                            sidecar_data["reason"] = f"promoted_cycle_{cycle_id}_checkpoint"
                            promoted_sidecar.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
                        except Exception:
                            pass

                # Copy diagnostics & events to base output dir for standardized contract
                src_diag = run_output_dir / "mt5_diagnostics.json"
                if src_diag.is_file():
                    shutil.copy2(src_diag, output_base_dir / f"mt5_diagnostics_cycle_{cycle_id}.json")

                src_events = run_output_dir / "events.jsonl"
                if src_events.is_file():
                    shutil.copy2(src_events, output_base_dir / f"events_cycle_{cycle_id}.jsonl")

                break

            # 4. If failed, trigger Adaptive Tuning on Loss (R2)
            print(f"\n[ADAPTIVE TUNING R2 TRIGGERED] Reasons: {latest_verdict.reasons}")
            diagnosis = diagnose_failure(latest_metrics, latest_verdict.reasons)
            print(f"  Diagnosis suggested actions: {diagnosis.suggested_actions}")

            if retry_idx >= self.max_adaptive_retries:
                print(f"[RETRY LIMIT REACHED] Cycle {cycle_id} failed after {self.max_adaptive_retries} retries.")
                break

            retry_idx += 1
            print(f"\n[ROLLBACK] Rolling back seed to original promoted champion: {seed_checkpoint.name}")
            active_seed = seed_checkpoint.resolve()

            # Compute tuned overrides
            tuning_overrides = compute_adaptive_tuning_overrides(
                retry_idx=retry_idx,
                diagnosis=diagnosis,
            )
            print(f"[TUNED KNOBS] Adjusting: {tuning_overrides}")

            # Generate adapted profile
            adapted_profile_path = output_base_dir / f"profile_cycle_{cycle_id}_retry_{retry_idx}.yaml"
            current_profile = generate_adapted_profile_yaml(
                base_profile_path=self.base_profile_path,
                output_profile_path=adapted_profile_path,
                tuning_overrides=tuning_overrides,
                experiment_id=f"sgt_compact_cycle_{cycle_id}_retry_{retry_idx}",
            )

        return {
            "cycle_id": cycle_id,
            "name": cycle_cfg.name,
            "from_date": active_from_date,
            "to_date": active_to_date,
            "promoted": promoted,
            "retries_used": retry_idx,
            "metrics": latest_metrics.to_dict(),
            "verdict": latest_verdict.to_dict(),
            "promoted_checkpoint": str(promoted_checkpoint_path) if promoted else None,
            "promoted_checkpoint_sha256": sha256_file(promoted_checkpoint_path) if (promoted and promoted_checkpoint_path.is_file()) else "",
        }

    def run_all_cycles(
        self,
        initial_seed: Path = DEFAULT_INITIAL_SEED,
        output_base_dir: Path = Path("outputs/walk_forward_multi_cycle"),
    ) -> dict[str, Any]:
        """Execute full Walk-Forward sequence: Cycle 1 -> Cycle 2 -> Cycle 3 with checkpoint inheritance."""
        initial_seed = (self.workspace_dir / initial_seed).resolve() if not initial_seed.is_absolute() else initial_seed.resolve()
        output_base_dir = (self.workspace_dir / output_base_dir).resolve() if not output_base_dir.is_absolute() else output_base_dir.resolve()
        output_base_dir.mkdir(parents=True, exist_ok=True)

        if not initial_seed.is_file():
            raise FileNotFoundError(f"Initial seed checkpoint not found: {initial_seed}")

        print(f"\n{'='*80}")
        print("STARTING WALK-FORWARD MULTI-CYCLE CHAIN (CYCLES 1 -> 2 -> 3)")
        print(f"Base Output:  {output_base_dir}")
        print(f"Initial Seed: {initial_seed}")
        print(f"{'='*80}\n")

        cycle_results: dict[int, dict[str, Any]] = {}
        current_seed = initial_seed

        for cycle_id in (1, 2, 3):
            cycle_res = self.run_cycle(
                cycle_id=cycle_id,
                seed_checkpoint=current_seed,
                output_base_dir=output_base_dir,
            )
            cycle_results[cycle_id] = cycle_res

            if not cycle_res["promoted"]:
                print(f"\n[PIPELINE ABORTED] Cycle {cycle_id} did not pass acceptance criteria. Halting sequence.")
                break

            # Inherit promoted checkpoint for next cycle
            promoted_path = Path(cycle_res["promoted_checkpoint"])
            current_seed = promoted_path
            print(f"\n[INHERITANCE] Cycle {cycle_id} promoted checkpoint passed as seed to Cycle {cycle_id + 1}: {current_seed.name}")

        # Final packaging if all cycles completed
        final_checkpoint_path = output_base_dir / "final_promoted_checkpoint.pkl"
        if 3 in cycle_results and cycle_results[3]["promoted"]:
            shutil.copy2(current_seed, final_checkpoint_path)
            # Copy sidecars
            for sidecar in current_seed.parent.glob(f"{current_seed.stem}.*"):
                if sidecar != current_seed:
                    sidecar_ext = sidecar.name[len(current_seed.stem):]
                    shutil.copy2(sidecar, output_base_dir / f"final_promoted_checkpoint{sidecar_ext}")

            final_sidecar = final_checkpoint_path.with_suffix(final_checkpoint_path.suffix + ".imitation.json")
            if not final_sidecar.is_file():
                seed_sidecar = current_seed.with_suffix(current_seed.suffix + ".imitation.json")
                if seed_sidecar.is_file():
                    try:
                        sidecar_data = json.loads(seed_sidecar.read_text(encoding="utf-8"))
                        final_sha = sha256_file(final_checkpoint_path)
                        sidecar_data["checkpoint_sha256"] = final_sha
                        sidecar_data["candidate_checkpoint_sha256"] = final_sha
                        sidecar_data["reason"] = "final_promoted_checkpoint"
                        final_sidecar.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
                    except Exception:
                        pass

        # Generate manifest
        manifest = generate_walk_forward_manifest(
            candidate_checkpoint_path=final_checkpoint_path,
            cycle_results=cycle_results,
            output_dir=output_base_dir,
            initial_seed_path=initial_seed,
            target_cycles=[1, 2, 3],
        )

        print(f"\n{'='*80}")
        print("WALK-FORWARD MULTI-CYCLE PIPELINE EXECUTION COMPLETE")
        print(f"Multi-cycle Certified: {manifest['multi_cycle_certified']}")
        print(f"Manifest Path:         {output_base_dir / 'manifest.json'}")
        print(f"{'='*80}\n")

        return manifest


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for walk-forward multi-cycle orchestrator."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Multi-Cycle RL Training & Adaptive Tuning Engine (SGT-V193-Compact 17.5M)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--cycle",
        type=int,
        choices=[1, 2, 3],
        help="Run a single walk-forward cycle (1: Q1/2024, 2: Q3/2024, 3: Q1/2025).",
    )
    mode_group.add_argument(
        "--all-cycles",
        action="store_true",
        help="Run all cycles sequentially (Cycle 1 -> Cycle 2 -> Cycle 3) with checkpoint inheritance.",
    )
    mode_group.add_argument(
        "--package-only",
        action="store_true",
        help="Package final promoted checkpoint, imitation sidecars, and generate manifest.json from existing cycle checkpoints.",
    )

    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=DEFAULT_INITIAL_SEED,
        help="Path to initial seed checkpoint for the cycle run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/walk_forward_multi_cycle"),
        help="Base directory for cycle execution artifacts, checkpoints, and manifest.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Timeout in seconds for MT5 Strategy Tester bridge communication (must be >= 3600).",
    )
    parser.add_argument(
        "--max-wall-sec",
        type=int,
        default=DEFAULT_MAX_WALL_SEC,
        help="Max wall-clock seconds for MT5 execution (0 = infinite / no premature termination).",
    )
    parser.add_argument(
        "--max-adaptive-retries",
        type=int,
        default=DEFAULT_MAX_ADAPTIVE_RETRIES,
        help="Maximum number of adaptive tuning retries per cycle upon loss or criteria failure.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SOURCE_CONFIG,
        help="Path to base candidate configuration YAML.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_BASE_PROFILE,
        help="Path to active candidate experiment profile YAML.",
    )
    parser.add_argument(
        "--runner-source",
        type=Path,
        default=DEFAULT_RUNNER_SOURCE,
        help="Path to master MT5 portable runner directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute simulated cycle passes for pipeline & contract verification without MT5 terminal.",
    )
    parser.add_argument(
        "--start-retry",
        type=int,
        default=0,
        help="Starting retry index (e.g. 2 to start at Attempt 3 using adapted profile).",
    )
    parser.add_argument(
        "--initial-profile",
        type=Path,
        default=None,
        help="Optional initial profile path for starting retry.",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Optional start date override (YYYY.MM.DD) for cycle backtest.",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="Optional end date override (YYYY.MM.DD) for cycle backtest.",
    )

    return parser

