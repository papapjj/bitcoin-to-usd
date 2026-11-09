"""Run paired frozen real-tick OOS arms on disposable MT5 portable clones.

The login terminal is never used by this script.  Each candidate/baseline arm
gets a fresh portable directory with hard-linked history and private mutable
folders; the directory is removed only after reports and bridge artifacts have
been copied into the audit directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from self_growing_trader.chuyen_gia_policy import (  # noqa: E402
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _mt5_best_metrics,
    evaluate_policy_mt5_oos,
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config  # noqa: E402
from self_growing_trader.learning_integrity import (  # noqa: E402
    frozen_oos_promotion_gate,
    paired_oos_non_regression_gate,
    sha256_file,
)
from self_growing_trader.contextual_expectancy import robust_expectancy  # noqa: E402
from self_growing_trader.mt5_runner import parse_mt5_execution_ledger  # noqa: E402
from self_growing_trader.mt5_runner import MT5_MODEL_REAL_TICKS  # noqa: E402


SEED = ROOT / os.environ.get(
    "V13_SEED_SYSTEM",
    "outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl",
)
CANDIDATE = ROOT / os.environ.get(
    "V13_V118_CANDIDATE",
    "outputs/he_chuyen_gia/candidates/v118_seed_bounded_scale0001_realticks.pkl",
)
OUT = ROOT / os.environ.get(
    "V13_V118_OOS_OUT",
    "outputs/he_chuyen_gia/audit/v118_seed_bounded_scale0001_realticks_oos",
)
SOURCE_RUNNER = ROOT / os.environ.get("V13_SOURCE_RUNNER", "mt5_v2_runner")
MODEL = int(os.environ.get("V13_MODEL", str(MT5_MODEL_REAL_TICKS)))
WINDOWS = tuple(
    tuple(item.split(":", 1))
    for item in os.environ.get(
        "V13_V118_OOS_WINDOWS",
        "2025.05.29:2025.05.30,2025.05.30:2025.05.31,2025.05.31:2025.06.01",
    ).split(",")
    if ":" in item
)


def _clone_runner(target: Path) -> None:
    """Create a minimal isolated runner without copying the multi-GB history."""
    if target.exists():
        raise FileExistsError(target)
    source = SOURCE_RUNNER.resolve()
    target.mkdir(parents=True)
    for folder in ("Common/Files", "MQL5/Files", "Reports", "Tester", "logs"):
        (target / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        os.link(source / name, target / name)

    experts = target / "MQL5" / "Experts"
    experts.mkdir(parents=True, exist_ok=True)
    for name in ("SelfGrowing_MT5_Trainer_EA.ex5", "SelfGrowing_MT5_Trainer_EA.mq5"):
        shutil.copy2(source / "MQL5" / "Experts" / name, experts / name)

    for root, _dirs, files in os.walk(source / "Bases"):
        relative = Path(root).relative_to(source / "Bases")
        destination = target / "Bases" / relative
        destination.mkdir(parents=True, exist_ok=True)
        for filename in files:
            os.link(Path(root) / filename, destination / filename)

    shutil.copytree(source / "Config", target / "Config")
    terminal_ini = target / "Config" / "terminal.ini"
    text = terminal_ini.read_text(encoding="utf-16")
    for key in ("LastBuildDataPath", "LastBuild", "LastBuildIDE", "LastBuildTester"):
        text = re.sub(rf"{key}=\d+", f"{key}=5833", text)
    terminal_ini.write_text(text, encoding="utf-16")


def _safe_remove_clone(clone: Path) -> None:
    root = (OUT / "mt5_arms").resolve()
    resolved = clone.resolve()
    if resolved.parent != root:
        raise RuntimeError(f"refuse to remove runner outside arm root: {resolved}")
    if clone.exists():
        shutil.rmtree(clone)


def _copy_artifacts(metrics: dict[str, object], destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, source_value in (
        ("report", metrics.get("report_path")),
        ("execution", metrics.get("execution_ledger_path")),
        ("bridge", metrics.get("bridge_path")),
        ("feature_audit", metrics.get("feature_audit_path")),
    ):
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.is_file():
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        copied[key] = str(target.resolve())
    return copied


def _config():
    config = load_config(ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    # OOS is inference-only; no telemetry is needed in the disposable arm.
    return replace(config, observability=replace(config.observability, enabled=False))


def _lifecycle_day(lifecycle: object, *, window: int) -> str:
    """Return a deterministic UTC day key for a completed lifecycle.

    MT5 lifecycle samples are grouped by exit day so correlated trades from
    the same market day cannot masquerade as independently strong evidence.
    The parser accepts the native ``datetime`` values used by the MT5 ledger,
    timezone-aware ISO strings and epoch seconds.  Naive datetimes keep their
    ledger date because the legacy parser intentionally returns broker-local
    datetimes without timezone metadata.  Old ledgers may not carry an exit
    timestamp; those samples stay tied to their frozen OOS window instead of
    being silently discarded.
    """
    raw_exit_time = getattr(lifecycle, "exit_time", None)
    if isinstance(raw_exit_time, datetime):
        if raw_exit_time.tzinfo is not None:
            return raw_exit_time.astimezone(timezone.utc).date().isoformat()
        return raw_exit_time.date().isoformat()
    if isinstance(raw_exit_time, date):
        return raw_exit_time.isoformat()

    parsed_exit_time: datetime | None = None
    if isinstance(raw_exit_time, str):
        text = raw_exit_time.strip()
        if text:
            try:
                parsed_exit_time = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    raw_exit_time = float(text)
                except ValueError:
                    raw_exit_time = None
    if parsed_exit_time is not None:
        if parsed_exit_time.tzinfo is not None:
            return parsed_exit_time.astimezone(timezone.utc).date().isoformat()
        return parsed_exit_time.date().isoformat()

    try:
        exit_timestamp = float(raw_exit_time or 0.0)
    except (TypeError, ValueError):
        exit_timestamp = 0.0
    if np.isfinite(exit_timestamp) and exit_timestamp > 0.0:
        try:
            return datetime.fromtimestamp(exit_timestamp, tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    return f"window-{int(window):04d}"


def _block_bootstrap_lcb(
    blocks: dict[str, list[float]],
    *,
    min_blocks: int = 3,
    draws: int = 4096,
    seed: int = 20260922,
) -> dict[str, object]:
    """Estimate a conservative lower bound over independent day blocks.

    Each block contributes one mean net-R, then blocks are resampled with
    replacement.  The reported lower bound is the stricter of the bootstrap
    5th percentile and the existing prior-shrunk robust LCB.  This prevents a
    high trade count inside one volatile day from narrowing uncertainty as if
    every trade were independent.
    """
    clean_blocks: dict[str, list[float]] = {}
    for key, raw_values in blocks.items():
        values = np.asarray(raw_values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            values = np.clip(values, -12.0, 12.0)
            clean_blocks[str(key)] = [float(value) for value in values]
    block_means = np.asarray(
        [np.mean(values) for values in clean_blocks.values()],
        dtype=float,
    )
    result: dict[str, object] = {
        "eligible": False,
        "reasons": [],
        "block_count": int(block_means.size),
        "block_sample_counts": {
            key: len(values) for key, values in sorted(clean_blocks.items())
        },
        "block_mean_r": float(np.mean(block_means)) if block_means.size else None,
        "block_lcb_r": None,
        "bootstrap_draws": 0,
        "confidence": 0.95,
    }
    if block_means.size < int(min_blocks):
        result["reasons"] = [
            f"block_samples_insufficient:{int(block_means.size)}<{int(min_blocks)}"
        ]
        return result
    if int(draws) < 100:
        result["reasons"] = ["bootstrap_draws_insufficient"]
        return result

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        int(block_means.size),
        size=(int(draws), int(block_means.size)),
    )
    bootstrap_means = np.mean(block_means[indices], axis=1)
    bootstrap_lcb = float(np.quantile(bootstrap_means, 0.05))
    robust = robust_expectancy(
        block_means,
        min_samples=int(min_blocks),
        policy_origin_ratio=1.0,
    )
    result.update({
        "eligible": True,
        "bootstrap_draws": int(draws),
        "block_lcb_r": float(min(bootstrap_lcb, robust.expectancy_lcb)),
        "robust_block_lcb_r": float(robust.expectancy_lcb),
        "bootstrap_percentile_lcb_r": bootstrap_lcb,
    })
    return result


def _pooled_policy_evidence(
    rows: list[dict[str, object]],
    *,
    arm: str,
    config: object,
    min_samples: int = 30,
    min_lcb: float | None = 0.0,
    min_blocks: int = 3,
    min_block_lcb: float | None = None,
) -> dict[str, object]:
    """Aggregate unique policy-owned position R across frozen OOS windows.

    Per-window LCB is necessary but insufficient: a candidate can pass several
    small windows while its pooled policy edge remains negative.  Read-only
    ledgers are the source of samples so deals cannot inflate the count, and a
    duplicate position ticket fails closed instead of silently narrowing the
    uncertainty interval.
    """
    risk = getattr(config, "risk", None)
    initial_equity = float(getattr(risk, "initial_equity", 10000.0))
    values: list[float] = []
    # MT5 Strategy Tester tickets are only unique inside one isolated
    # portable/window ledger.  Separate windows can restart ticket numbering,
    # so the pooled identity must be scoped by the frozen OOS window.
    tickets: set[tuple[int, str]] = set()
    duplicate_tickets: set[str] = set()
    reasons: list[str] = []
    window_checks: list[dict[str, object]] = []
    day_blocks: dict[str, list[float]] = {}

    for row in rows:
        window = int(row.get("window", len(window_checks) + 1))
        arm_result = row.get(arm)
        metrics = dict(arm_result.get("metrics", {}) or {}) if isinstance(arm_result, dict) else {}
        check_reasons: list[str] = []
        reconciliation = metrics.get("reconciliation")
        if not isinstance(reconciliation, dict):
            reconciliation = {}
        reconciled = metrics.get("fitness_reconciled", reconciliation.get("reconciled"))
        if reconciled is not True:
            check_reasons.append("reconciliation_missing_or_failed")
        terminal_open = metrics.get(
            "fitness_terminal_open_positions",
            reconciliation.get("terminal_open_positions"),
        )
        try:
            terminal_open_count = int(terminal_open)
        except (TypeError, ValueError):
            terminal_open_count = -1
        if terminal_open_count != 0:
            check_reasons.append(f"terminal_open_positions:{terminal_open_count if terminal_open_count >= 0 else 'missing'}")
        feature_ok = metrics.get("feature_audit_valid", metrics.get("feature_parity_numerical_valid"))
        if feature_ok is not True:
            check_reasons.append("feature_parity_missing_or_invalid")
        real_tick_ok = metrics.get("real_tick_evidence_verified")
        if real_tick_ok is not True:
            check_reasons.append("real_tick_evidence_missing_or_invalid")

        ledger_path = metrics.get("execution_ledger_path")
        summary = parse_mt5_execution_ledger(
            ledger_path,
            initial_equity=initial_equity,
        )
        window_values = 0
        for lifecycle in tuple(getattr(summary, "lifecycles", ()) or ()):
            if not bool(getattr(lifecycle, "final_close_confirmed", False)):
                continue
            metadata = dict(getattr(lifecycle, "metadata", {}) or {})
            owner_trace = str(
                metadata.get("position_owner_trace_id", "")
                or getattr(lifecycle, "trace_id", "")
                or ""
            ).strip()
            if not owner_trace:
                continue
            ticket = str(getattr(lifecycle, "position_ticket", "") or "").strip()
            if not ticket:
                check_reasons.append("policy_lifecycle_ticket_missing")
                continue
            ticket_key = (window, ticket)
            if ticket_key in tickets:
                scoped_ticket = f"window-{window:04d}:{ticket}"
                duplicate_tickets.add(scoped_ticket)
                check_reasons.append(f"duplicate_position_ticket:{scoped_ticket}")
                continue
            try:
                realized_r = float(getattr(lifecycle, "realized_r", float("nan")))
            except (TypeError, ValueError):
                realized_r = float("nan")
            if not np.isfinite(realized_r):
                check_reasons.append(f"nonfinite_policy_realized_r:{ticket}")
                continue
            tickets.add(ticket_key)
            values.append(realized_r)
            window_values += 1
            day_blocks.setdefault(
                _lifecycle_day(lifecycle, window=window),
                [],
            ).append(realized_r)
        window_checks.append({
            "window": window,
            "arm": arm,
            "policy_lifecycle_samples": window_values,
            "reasons": check_reasons,
        })
        reasons.extend(f"window_{window}_{arm}:{reason}" for reason in check_reasons)

    estimate = robust_expectancy(
        values,
        min_samples=min_samples,
        policy_origin_ratio=1.0,
    )
    pooled_lcb = float(estimate.expectancy_lcb) if values else None
    pooled_mean = float(estimate.mean_net_r) if values else None
    if len(values) < int(min_samples):
        reasons.append(f"pooled_policy_samples_insufficient:{len(values)}<{int(min_samples)}")
    if min_lcb is not None and (pooled_lcb is None or pooled_lcb < float(min_lcb)):
        shown = "missing" if pooled_lcb is None else f"{pooled_lcb:.6g}"
        reasons.append(f"pooled_policy_lcb_below_limit:{shown}<{float(min_lcb):g}")
    day_bootstrap = _block_bootstrap_lcb(
        day_blocks,
        min_blocks=min_blocks,
    )
    if not day_bootstrap["eligible"]:
        reasons.extend(
            f"day_bootstrap:{reason}"
            for reason in day_bootstrap.get("reasons", [])
        )
    elif (
        min_block_lcb is not None
        and float(day_bootstrap["block_lcb_r"]) < float(min_block_lcb)
    ):
        reasons.append(
            "day_bootstrap_lcb_below_limit:"
            f"{float(day_bootstrap['block_lcb_r']):.6g}<{float(min_block_lcb):g}"
        )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "arm": arm,
        "windows": len(rows),
        "policy_lifecycle_samples": len(values),
        "unique_position_tickets": len(tickets),
        "duplicate_position_tickets": sorted(duplicate_tickets),
        "policy_expectancy_mean_r": pooled_mean,
        "policy_expectancy_lcb_r": pooled_lcb,
        "policy_expectancy": estimate.as_dict(),
        "day_bootstrap": day_bootstrap,
        "window_checks": window_checks,
    }


def _pooled_paired_gate(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    max_lcb_regression: float = 0.02,
) -> dict[str, object]:
    """Compare pooled candidate/baseline LCB without hiding missing evidence."""
    reasons: list[str] = []
    candidate_lcb = candidate.get("policy_expectancy_lcb_r")
    baseline_lcb = baseline.get("policy_expectancy_lcb_r")
    candidate_mean = candidate.get("policy_expectancy_mean_r")
    candidate_day = candidate.get("day_bootstrap")
    baseline_day = baseline.get("day_bootstrap")
    if not isinstance(candidate_lcb, (int, float)) or not np.isfinite(float(candidate_lcb)):
        reasons.append("candidate_pooled_lcb_missing")
    if not isinstance(baseline_lcb, (int, float)) or not np.isfinite(float(baseline_lcb)):
        reasons.append("baseline_pooled_lcb_missing")
    if not isinstance(candidate_mean, (int, float)) or not np.isfinite(float(candidate_mean)) or float(candidate_mean) < 0.0:
        reasons.append("candidate_pooled_expectancy_below_zero")
    candidate_day_lcb = (
        candidate_day.get("block_lcb_r")
        if isinstance(candidate_day, dict)
        else None
    )
    baseline_day_lcb = (
        baseline_day.get("block_lcb_r")
        if isinstance(baseline_day, dict)
        else None
    )
    if not isinstance(candidate_day_lcb, (int, float)) or not np.isfinite(float(candidate_day_lcb)):
        reasons.append("candidate_day_bootstrap_lcb_missing")
    if not isinstance(baseline_day_lcb, (int, float)) or not np.isfinite(float(baseline_day_lcb)):
        reasons.append("baseline_day_bootstrap_lcb_missing")
    if (
        isinstance(candidate_lcb, (int, float))
        and isinstance(baseline_lcb, (int, float))
        and np.isfinite(float(candidate_lcb))
        and np.isfinite(float(baseline_lcb))
        and float(candidate_lcb) < float(baseline_lcb) - float(max_lcb_regression)
    ):
        reasons.append(
            f"pooled_expectancy_lcb_regression:{float(candidate_lcb):.6g}"
            f"<{float(baseline_lcb) - float(max_lcb_regression):.6g}"
        )
    if (
        isinstance(candidate_day_lcb, (int, float))
        and isinstance(baseline_day_lcb, (int, float))
        and np.isfinite(float(candidate_day_lcb))
        and np.isfinite(float(baseline_day_lcb))
        and float(candidate_day_lcb) < float(baseline_day_lcb) - float(max_lcb_regression)
    ):
        reasons.append(
            "pooled_day_bootstrap_lcb_regression:"
            f"{float(candidate_day_lcb):.6g}"
            f"<{float(baseline_day_lcb) - float(max_lcb_regression):.6g}"
        )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "candidate_pooled_lcb_r": candidate_lcb,
        "baseline_pooled_lcb_r": baseline_lcb,
        "candidate_day_bootstrap_lcb_r": candidate_day_lcb,
        "baseline_day_bootstrap_lcb_r": baseline_day_lcb,
        "max_lcb_regression": float(max_lcb_regression),
    }


def _run_arm(policy_path: Path, config, *, window: int, arm: str, from_date: str, to_date: str) -> dict[str, object]:
    clone = (OUT / "mt5_arms" / f"window_{window:02d}_{arm}").resolve()
    _clone_runner(clone)
    try:
        with policy_path.open("rb") as handle:
            import pickle

            policy = pickle.load(handle)
        _cau_hinh_policy_chuyen_gia_cho_mt5(policy, config)
        result = evaluate_policy_mt5_oos(
            policy,
            config,
            runner_dir=clone,
            timeout_sec=180,
            visual=False,
            shutdown_terminal=True,
            from_date=from_date,
            to_date=to_date,
            model=MODEL,
        )
        metrics = _mt5_best_metrics(result)
        artifacts = _copy_artifacts(metrics, OUT / "oos_artifacts" / f"window_{window:02d}" / arm)
        return {
            "metrics": metrics,
            "fitness": result.get("best", {}).get("fitness"),
            "artifacts": artifacts,
            "source_policy_state_hash_before": result.get("source_policy_state_hash_before"),
            "source_policy_state_hash_after": result.get("source_policy_state_hash_after"),
            "frozen": bool(result.get("frozen", False)),
        }
    finally:
        _safe_remove_clone(clone)


def main() -> None:
    if not SEED.is_file():
        raise FileNotFoundError(SEED)
    if not CANDIDATE.is_file():
        raise FileNotFoundError(CANDIDATE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mt5_arms").mkdir(parents=True, exist_ok=True)
    seed_hash = sha256_file(SEED)
    candidate_hash = sha256_file(CANDIDATE)
    config = _config()
    rows: list[dict[str, object]] = []
    for index, (from_date, to_date) in enumerate(WINDOWS, start=1):
        print(json.dumps({"phase": "oos", "window": index, "from": from_date, "to": to_date}, ensure_ascii=False), flush=True)
        candidate = _run_arm(CANDIDATE, config, window=index, arm="candidate", from_date=from_date, to_date=to_date)
        baseline = _run_arm(SEED, config, window=index, arm="baseline", from_date=from_date, to_date=to_date)
        paired = paired_oos_non_regression_gate(
            candidate["metrics"],
            baseline["metrics"],
            min_candidate_expectancy=0.0,
            max_profit_regression=0.0,
            max_expectancy_regression=0.02,
            max_drawdown_increase_pct=1.0,
        )
        strict_candidate = frozen_oos_promotion_gate(
            candidate["metrics"],
            require_fitness_eligible=True,
            require_reconciled=True,
            min_trades=30,
            max_drawdown_pct=10.0,
            min_policy_expectancy_lcb=0.0,
            max_negative_context_scopes=0,
            require_feature_parity=True,
            require_real_tick_evidence=True,
        )
        strict_baseline = frozen_oos_promotion_gate(
            baseline["metrics"],
            require_fitness_eligible=False,
            require_reconciled=True,
            min_trades=30,
            max_drawdown_pct=10.0,
            min_policy_expectancy_lcb=None,
            max_negative_context_scopes=None,
            require_feature_parity=True,
            require_real_tick_evidence=True,
        )
        reasons = list(paired.get("reasons", []) or [])
        for label, strict in (("candidate", strict_candidate), ("baseline", strict_baseline)):
            if not strict.get("eligible", False):
                paired["eligible"] = False
                reasons.extend(f"{label}_strict_gate:{reason}" for reason in strict.get("reasons", []))
        paired["reasons"] = reasons
        row = {
            "window": index,
            "oos": {"from": from_date, "to": to_date},
            "candidate": candidate,
            "baseline": baseline,
            "strict_candidate": strict_candidate,
            "strict_baseline": strict_baseline,
            "paired": paired,
        }
        rows.append(row)
        (OUT / f"window_{index:02d}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"window": index, "eligible": bool(paired.get("eligible", False)), "reasons": paired.get("reasons", [])}, ensure_ascii=False), flush=True)
    summary = {
        "seed": str(SEED),
        "seed_sha256_before": seed_hash,
        "seed_sha256_after": sha256_file(SEED),
        "candidate": str(CANDIDATE),
        "candidate_sha256": candidate_hash,
        "model": MODEL,
        "windows": rows,
        "all_windows_eligible": bool(rows) and all(bool(row["paired"].get("eligible", False)) for row in rows),
        "promoted": False,
    }
    pooled_candidate = _pooled_policy_evidence(
        rows,
        arm="candidate",
        config=config,
        min_samples=30,
        min_lcb=0.0,
        min_blocks=3,
        min_block_lcb=0.0,
    )
    pooled_baseline = _pooled_policy_evidence(
        rows,
        arm="baseline",
        config=config,
        min_samples=30,
        min_lcb=None,
        min_blocks=3,
        min_block_lcb=None,
    )
    pooled_paired = _pooled_paired_gate(pooled_candidate, pooled_baseline)
    summary["pooled_candidate"] = pooled_candidate
    summary["pooled_baseline"] = pooled_baseline
    summary["pooled_paired"] = pooled_paired
    summary["all_windows_eligible"] = bool(
        summary["all_windows_eligible"]
        and pooled_candidate["eligible"]
        and pooled_baseline["eligible"]
        and pooled_paired["eligible"]
    )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": str((OUT / "summary.json").resolve()), "all_windows_eligible": summary["all_windows_eligible"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
