#!/usr/bin/env python3
"""Create and optionally MT5-smoke an isolated edge-discovery candidate.

The source config and seed artifact are read-only. The script always creates
an explicit candidate seed clone and a portable MT5 runner clone, enforces the
configured mutable-output quota plus reserve floor, and never touches an
existing Strategy Tester process outside its own runner directory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
import yaml

CANDIDATE_MANIFEST_SCHEMA_VERSION = 5
STORAGE_GATE_SCHEMA_VERSION = 2
CANDIDATE_DATE_FORMAT = "%Y.%m.%d"
CURRENT_IMITATION_REPORT_SCHEMA_VERSION = 11
CURRENT_IMITATION_REPRESENTATION_SCOPE = (
    "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_growing_trader.chuyen_gia_policy import (  # noqa: E402
    TENSOR_RULE_ARTIFACT_VERSION,
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    huan_luyen_he_chuyen_gia_mt5,
    tai_he_chuyen_gia_mot_context_mot_policy,
    tao_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config  # noqa: E402
from self_growing_trader.mt5_live_diagnostics import Mt5DiagnosticsMonitor  # noqa: E402
from self_growing_trader.mt5_runner import (  # noqa: E402
    DEFAULT_SYMBOL,
    MT5_MODEL_REAL_TICKS,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT  # noqa: E402
from self_growing_trader.seven_choice_teacher import (  # noqa: E402
    seven_choice_runtime_contract,
)
from self_growing_trader.tensor_rule_bank import TensorRuleBankState  # noqa: E402
from self_growing_trader.state_alias_memory import (  # noqa: E402
    STATE_ALIAS_MEMORY_ARTIFACT_VERSION,
    StateAliasingMemoryState,
)


def _ensure_utf8_stdio() -> None:
    """Keep structured candidate diagnostics printable on Windows consoles.

    The candidate runner emits Vietnamese status/error text with
    ``ensure_ascii=False``.  A PowerShell-launched process can still inherit
    the legacy ``cp1252`` streams, which would abort the run before MT5 is
    opened.  Replacement is limited to the human-facing stream boundary;
    receipt/status files continue to be written as UTF-8 and are unaffected.
    """
    for stream in (sys.stdout, sys.stderr):
        configure = getattr(stream, "reconfigure", None)
        if not callable(configure):
            continue
        try:
            configure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Some embedded/test streams expose reconfigure but reject one of
            # the options.  File-backed artifacts remain the authoritative log.
            continue


def _install_graceful_shutdown_handlers():
    """Turn external stop signals into the trainer's persist-and-stop path.

    ``huan_luyen_he_chuyen_gia_mt5`` already persists an incomplete checkpoint
    when it catches ``KeyboardInterrupt``.  Windows process termination can
    otherwise bypass that path because the default ``SIGTERM`` handler exits
    immediately.  The runner is the process boundary that owns the MT5 call,
    so translate both supported console/termination signals here and restore
    the caller's handlers when the CLI returns (important for in-process tests).
    Installing a handler is best-effort because ``signal.signal`` is only
    legal on the interpreter's main thread.
    """

    previous: dict[int, object] = {}

    def _handler(signum: int, _frame: object) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        raise KeyboardInterrupt(f"received {name}")

    for signal_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, signal_name, None)
        if sig is None:
            continue
        try:
            prior = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except (OSError, RuntimeError, ValueError):
            continue
        previous[sig] = prior

    def _restore() -> None:
        for sig, prior in previous.items():
            try:
                signal.signal(sig, prior)
            except (OSError, RuntimeError, ValueError):
                # A caller may restore after the interpreter has moved off the
                # main thread or begun shutdown.  The process is already
                # exiting then, so there is nothing left to persist here.
                continue

    return _restore


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CODEX_STATUS_PHASE_KEYS = (
    "enabled",
    "submitted",
    "collected",
    "graded",
    "hypotheses",
    "hypotheses_confirmed",
    "hypothesis_rejections",
    "updates",
    "codex_optimizer_updates",
    "generic_optimizer_updates",
    "interventions_applied",
    "failures",
    "grade_failures",
    "queue_failures",
    "pending",
    "rejected_updates",
    "skips",
    "account_impact_records",
    "account_impact_confirmed",
    "account_impact_partial",
    "account_impact_reward_sum",
    "deferred",
    "deferred_released",
    "deferred_pending",
    "deferred_forced_partial",
    "counterfactual_validated",
    "counterfactual_unavailable",
    "rlhf_parameter_changed",
    "rlhf_internal_policy_state_changed",
    "rlhf_probability_shift_count",
    "rlhf_max_probability_shift",
    "codex_causal_links_observed",
    "codex_causal_updates_linked",
    "codex_causal_hash_matches",
    "codex_causal_hash_unknown",
    "last_codex_causal_link",
    "report_ledger_reconciled",
    "live_owner_outcome_records",
    "live_owner_outcome_samples_by_action",
    "live_owner_outcome_negative_count",
    "live_owner_outcome_last",
    "online_wait_sec",
    "online_barrier_waits",
    "online_barrier_passed",
    "online_barrier_timeouts",
    "online_barrier_waited_sec",
    "recovery_snapshot_updates",
    "recovery_snapshot_path",
    "recovery_snapshot_error",
    "updated_at",
)


def _latest_codex_teacher_status(output: Path) -> tuple[dict[str, object], str]:
    """Read the newest Codex status sidecar left by an interrupted run.

    The training function intentionally raises after persisting an incomplete
    checkpoint, so the candidate runner cannot rely on a returned ``result``
    to update the runtime contract.  The append-only receipt status is the
    authoritative recovery source in that path.  Restrict the search to the
    candidate output directory and prefer the newest primary/fallback sidecar.
    """
    candidates: list[Path] = []
    try:
        candidates = [
            path
            for path in output.rglob("*codex_teacher_receipts.jsonl.status.json*")
            if path.is_file()
        ]
    except OSError:
        return {}, ""
    candidates.sort(
        key=lambda path: (
            path.stat().st_mtime_ns if path.exists() else 0,
            str(path),
        )
    )
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload, str(path)
    return {}, ""


def recover_interrupted_codex_feedback(
    feedback_contract: dict[str, object], output: Path,
) -> dict[str, object]:
    """Merge receipt/status telemetry into a contract after a hard interruption.

    This is observability recovery only.  It never fabricates a receipt,
    hypothesis, account impact, or optimizer update: every copied value comes
    from the status sidecar written by the Codex teacher.  The returned summary
    is suitable for attaching to a candidate manifest.
    """
    status, status_path = _latest_codex_teacher_status(output)
    if not status:
        return {}
    preflight = status.get("preflight")
    if not isinstance(preflight, dict):
        preflight = {}
    preflight_passed = str(preflight.get("status", "") or "").lower() == "passed"
    cli_found = bool(
        str(status.get("executable_path", "") or "").strip()
        and str(status.get("cli_version", "") or "").strip()
    )
    effective = bool(status.get("enabled", False)) and preflight_passed
    phase = {
        key: status[key]
        for key in _CODEX_STATUS_PHASE_KEYS
        if key in status
    }
    phase.update({
        "status_path": status_path,
        "preflight": dict(preflight),
        "model": str(status.get("model", "") or ""),
        "reasoning_effort": str(status.get("reasoning_effort", "") or ""),
        "executable_path": str(status.get("executable_path", "") or ""),
        "cli_version": str(status.get("cli_version", "") or ""),
        "recovered_after_exception": True,
    })
    feedback_contract["codex_teacher_phase"] = phase
    feedback_contract["codex_preflight"] = dict(preflight)
    feedback_contract["codex_auto_connection"] = {
        "requested": bool(status.get("enabled", False)),
        "connected": effective,
        "effective": effective,
        "cli_found": cli_found,
        "preflight_passed": preflight_passed,
        "model": str(status.get("model", "") or ""),
        "reasoning_effort": str(status.get("reasoning_effort", "") or ""),
        "executable_path": str(status.get("executable_path", "") or ""),
        "cli_version": str(status.get("cli_version", "") or ""),
        "preflight": dict(preflight),
        "recovered_after_exception": True,
    }
    effective_contract = feedback_contract.setdefault("effective", {})
    if not isinstance(effective_contract, dict):
        effective_contract = {}
        feedback_contract["effective"] = effective_contract
    effective_contract.update({
        "five_codex_teacher_effective": effective,
        "codex_cli_found": cli_found,
        "codex_preflight_passed": preflight_passed,
        "codex_receipts_valid": int(status.get("graded", 0) or 0) > 0,
        "codex_hypotheses_confirmed": int(
            status.get("hypotheses_confirmed", 0) or 0
        ),
        "codex_optimizer_updates": int(
            status.get("updates", status.get("codex_optimizer_updates", 0)) or 0
        ),
    })
    for status_key, contract_key in (
        ("model", "five_codex_teacher_model"),
        ("online_wait_sec", "five_codex_teacher_online_wait_sec"),
    ):
        if status_key in status:
            effective_contract[contract_key] = status[status_key]
    runtime_feedback = feedback_contract.setdefault("runtime_feedback_contract", {})
    if not isinstance(runtime_feedback, dict):
        runtime_feedback = {}
        feedback_contract["runtime_feedback_contract"] = runtime_feedback
    runtime_feedback.update({
        "five_codex_teacher_effective": effective,
        "codex_cli_found": cli_found,
        "codex_preflight_passed": preflight_passed,
        "codex_receipts_valid": int(status.get("graded", 0) or 0) > 0,
        "codex_hypotheses_confirmed": int(
            status.get("hypotheses_confirmed", 0) or 0
        ),
        "codex_optimizer_updates": int(
            status.get("updates", status.get("codex_optimizer_updates", 0)) or 0
        ),
        "codex_account_impact_records": int(
            status.get("account_impact_records", 0) or 0
        ),
        "codex_account_impact_confirmed": int(
            status.get("account_impact_confirmed", 0) or 0
        ),
        "codex_account_impact_partial": int(
            status.get("account_impact_partial", 0) or 0
        ),
        "codex_account_impact_reward_sum": float(
            status.get("account_impact_reward_sum", 0.0) or 0.0
        ),
        "codex_rlhf_parameter_changed": int(
            status.get("rlhf_parameter_changed", 0) or 0
        ),
        "codex_internal_policy_state_changed": int(
            status.get("rlhf_internal_policy_state_changed", 0) or 0
        ),
        "codex_rlhf_probability_shift_count": int(
            status.get("rlhf_probability_shift_count", 0) or 0
        ),
        "codex_rlhf_max_probability_shift": float(
            status.get("rlhf_max_probability_shift", 0.0) or 0.0
        ),
        "codex_teacher_model": str(status.get("model", "") or ""),
        "codex_teacher_executable": str(status.get("executable_path", "") or ""),
        "codex_teacher_cli_version": str(status.get("cli_version", "") or ""),
        "codex_teacher_preflight": dict(preflight),
    })
    feedback_contract["codex_teacher_status_path"] = status_path
    feedback_contract["codex_teacher_status_recovered"] = True
    return {
        "status_path": status_path,
        "submitted": int(status.get("submitted", 0) or 0),
        "graded": int(status.get("graded", 0) or 0),
        "hypotheses_confirmed": int(status.get("hypotheses_confirmed", 0) or 0),
        "updates": int(status.get("updates", 0) or 0),
        "account_impact_confirmed": int(
            status.get("account_impact_confirmed", 0) or 0
        ),
        "rlhf_parameter_changed": int(status.get("rlhf_parameter_changed", 0) or 0),
        "recovered_after_exception": True,
    }


def _sync_runtime_recovery_snapshot_interval(
    feedback_contract: dict[str, object],
    runtime_contract: Mapping[str, object],
) -> None:
    """Keep the recovery cadence visible in the top-level effective contract.

    ``runtime_feedback_contract`` uses the Codex-prefixed field name because it
    is the evaluator-facing contract.  The candidate contract also exposes the
    profile-facing ``five_rlhf_*`` name; keeping both values synchronized avoids
    a stale UI reading the initial profile value after the trainer finishes.
    Missing runtime telemetry is deliberately non-destructive so interrupted
    runs retain the profile value already written before MT5 started.
    """
    if not isinstance(runtime_contract, Mapping):
        return
    raw_interval = runtime_contract.get(
        "codex_teacher_recovery_snapshot_interval_updates"
    )
    if raw_interval is None:
        return
    try:
        interval = max(0, int(raw_interval or 0))
    except (TypeError, ValueError):
        return
    effective = feedback_contract.setdefault("effective", {})
    if not isinstance(effective, dict):
        effective = {}
        feedback_contract["effective"] = effective
    effective["five_rlhf_recovery_snapshot_interval_updates"] = interval


def _ensure_mt5_runtime_uncompressed(paths: tuple[Path, ...]) -> dict[str, object]:
    """Remove inherited NTFS compression from MT5's mutable runtime trees.

    Candidate output folders can inherit the Windows ``Compressed`` attribute
    from a disk-saving workspace.  MetaTester may then fail to package or load
    an EA/tester-file dependency even though the files exist.  Keep the
    operation scoped to the newly-created candidate clone; source artifacts
    and unrelated terminals are never modified.
    """

    if os.name != "nt":
        return {"attempted": False, "ok": True, "paths": []}
    results: list[dict[str, object]] = []
    ok = True
    for path in paths:
        if not path.exists():
            continue
        try:
            completed = subprocess.run(
                ["compact.exe", "/U", "/Q", "/F", f"/S:{path}"],
                capture_output=True,
                text=True,
                check=False,
            )
            returncode = int(completed.returncode)
        except OSError as exc:
            returncode = -1
            results.append({"path": str(path), "returncode": returncode, "error": str(exc)})
        else:
            results.append({"path": str(path), "returncode": returncode})
        ok = ok and returncode == 0
    return {"attempted": True, "ok": bool(ok), "paths": results}


def inspect_seven_choice_runtime_contract(
    checkpoint: Path, config: Any,
) -> dict[str, object]:
    """Inspect a seven-choice sidecar before allowing an MT5 candidate run."""

    report_path = checkpoint.with_suffix(checkpoint.suffix + ".seven_choice.json")
    if not report_path.is_file():
        return {"declared": False, "reason": "no_seven_choice_sidecar"}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "declared": True,
            "match": False,
            "reason": "seven_choice_sidecar_unreadable",
            "report_path": str(report_path),
            "error": str(exc),
        }
    manifest = report.get("dataset_manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        return {
            "declared": True,
            "match": False,
            "reason": "seven_choice_manifest_missing",
            "report_path": str(report_path),
        }
    contract = seven_choice_runtime_contract(manifest, config)
    return {
        "declared": True,
        "report_path": str(report_path),
        "checkpoint": str(checkpoint),
        **contract,
    }


def validate_date_window(from_date: str, to_date: str) -> tuple[datetime, datetime]:
    """Từ chối ngày lịch không tồn tại thay vì để MT5 tự chuẩn hóa ngầm."""

    try:
        start = datetime.strptime(str(from_date).strip(), CANDIDATE_DATE_FORMAT)
        end = datetime.strptime(str(to_date).strip(), CANDIDATE_DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(
            "from-date/to-date phai la ngay ton tai theo dinh dang YYYY.MM.DD"
        ) from exc
    if start >= end:
        raise ValueError("from-date phai som hon to-date")
    return start, end


def projected_candidate_payload_bytes(
    seed_bytes: int,
    telemetry_budget: int,
    *,
    metrics_only: bool = False,
) -> int:
    """Estimate mutable bytes needed by the candidate runner.

    A metrics-only MT5 run never serializes a candidate checkpoint.  Its seed
    is protected separately and is normally an NTFS hardlink, so reserving
    ``1.25 * seed`` would overstate the mutable payload and reject a safe
    diagnostic run.  Keep the telemetry budget as the only projected payload
    in that mode; the post-clone free-space check still rejects a fallback
    physical copy that would violate the reserve floor.
    """
    if bool(metrics_only):
        return max(0, int(telemetry_budget))
    return max(int(int(seed_bytes) * 1.25), int(seed_bytes) + int(telemetry_budget))


def effective_storage_reserve_floor(
    configured_reserve: int,
    runner_bytes: int = 0,
    *,
    metrics_only: bool = False,
) -> int:
    """Account for the temporary MT5 runner in metrics-only mode.

    A metrics-only run never persists a candidate checkpoint.  Its isolated
    runner is temporary and is removed at the terminal boundary, so the live
    free-space floor can reserve the configured floor minus that runner's
    footprint.  Candidate-persisting runs retain the configured floor exactly.
    """
    reserve = max(0, int(configured_reserve))
    if not bool(metrics_only):
        return reserve
    temporary_runner = max(0, int(runner_bytes))
    minimum_floor = 512 * 1024**2
    return max(minimum_floor, reserve - temporary_runner)


def directory_size(path: Path, *, exclude: tuple[Path, ...] = ()) -> int:
    excluded = {item.resolve() for item in exclude}
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                candidate = (Path(root) / name).resolve()
                if candidate in excluded:
                    continue
                total += candidate.stat().st_size
            except OSError:
                pass
    return total


def write_storage_preflight_failure(
    output: Path,
    *,
    mode: str,
    free_bytes: int,
    reserve_floor_bytes: int,
    required_free_bytes: int,
    projected_payload_bytes: int = 0,
    reason: str,
) -> dict[str, object]:
    """Persist a fail-closed storage decision before MT5 is opened.

    The early candidate gate runs before the full manifest/feedback contract
    exists.  Without a small durable record, a rejected launch looks like a
    missing run to the UI and the operator cannot distinguish it from a
    crashed MT5 process.  This record contains no learner state and is safe to
    write even when the requested candidate payload cannot fit.
    """
    record: dict[str, object] = {
        "schema_version": STORAGE_GATE_SCHEMA_VERSION,
        "type": "storage_preflight_failure",
        "mode": str(mode),
        "reason": str(reason),
        "free_bytes": int(free_bytes),
        "reserve_floor_bytes": int(reserve_floor_bytes),
        "required_free_bytes": int(required_free_bytes),
        "projected_candidate_payload_bytes": int(projected_payload_bytes),
        "timestamp": time.time(),
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        (output / "storage_gate_failure.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "manifest.json").write_text(
            json.dumps({
                "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
                "mode": str(mode),
                "promotable": False,
                "promotion_block_reason": "storage_preflight_failed",
                "storage_gate_failure": record,
                "run_exception": {
                    "type": "StoragePreflightError",
                    "message": (
                        f"storage preflight failed: free={free_bytes} "
                        f"< required={required_free_bytes}"
                    ),
                },
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        # The caller still fails closed.  A second disk error must not turn a
        # clear storage rejection into an unrelated traceback.
        pass
    return record


def inspect_action_collapse(status_path: Path) -> dict[str, object]:
    """Classify a terminal checkpoint that ended in a one-action policy.

    A profitable window can still finish after the policy has collapsed to
    HOLD. Persisting that terminal state makes the next independent window
    unable to discover new lifecycle evidence. The run's ledger and metrics
    remain valid; this guard only decides whether its weights are safe as a
    continuation seed.
    """
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "collapsed": False, "reason": type(exc).__name__}
    policy = dict(payload.get("policy", {}) or {})
    wire_hold_share = float(
        policy.get("wire_hold_share", policy.get("hold_share", 0.0)) or 0.0
    )
    wire_effective_actions = float(
        policy.get(
            "wire_effective_action_count",
            policy.get("effective_action_count", 0.0),
        ) or 0.0
    )
    wire_normalized_entropy = float(
        policy.get(
            "wire_normalized_action_entropy",
            policy.get("normalized_action_entropy", 0.0),
        ) or 0.0
    )
    wire_collapse_detected = bool(
        wire_hold_share >= 0.95
        or wire_effective_actions <= 1.25
        or wire_normalized_entropy <= 0.05
    )
    selected = policy
    scope = str(policy.get("scope", "wire") or "wire")
    economic_scope = policy.get("economic_actor")
    if isinstance(economic_scope, dict):
        try:
            economic_total = int(economic_scope.get("total", 0) or 0)
        except (TypeError, ValueError):
            economic_total = 0
        if economic_total > 0:
            selected = economic_scope
            scope = "economic_actor"
    hold_share = float(selected.get("hold_share", 0.0) or 0.0)
    effective_actions = float(selected.get("effective_action_count", 0.0) or 0.0)
    normalized_entropy = float(selected.get("normalized_action_entropy", 0.0) or 0.0)
    economic_collapse_detected = bool(
        hold_share >= 0.95
        or effective_actions <= 1.25
        or normalized_entropy <= 0.05
    )
    status = str(dict(payload.get("status", {}) or {}).get("policy", "") or "")
    tail = policy.get("tail", {})
    if not isinstance(tail, dict):
        tail = {}
    tail_collapse_detected = bool(
        policy.get("tail_collapse_detected", tail.get("collapse_detected", False))
    )
    collapsed = bool(
        (scope == "wire" and status == "collapsed")
        or hold_share >= 0.99
        or effective_actions <= 1.10
        or normalized_entropy <= 0.01
        or tail_collapse_detected
    )
    return {
        "available": True,
        "collapsed": collapsed,
        "status": status,
        "collapse_scope": scope,
        "wire_collapse_detected": wire_collapse_detected,
        "economic_actor_collapse_detected": economic_collapse_detected,
        "wire_collapse_warning": bool(wire_collapse_detected and scope != "wire"),
        "hold_share": hold_share,
        "effective_action_count": effective_actions,
        "normalized_action_entropy": normalized_entropy,
        "wire_hold_share": wire_hold_share,
        "wire_effective_action_count": wire_effective_actions,
        "wire_normalized_action_entropy": wire_normalized_entropy,
        "tail_window_size": int(policy.get("tail_window_size", tail.get("total", 0)) or 0),
        "tail_collapse_detected": tail_collapse_detected,
        "tail": tail,
    }


def finalize_checkpoint_persistence(
    manifest: dict[str, object],
    *,
    metrics_only: bool,
    selected_champion: bool,
    collapsed: bool,
    candidate_exists: bool,
) -> dict[str, object]:
    """Record checkpoint disposition without discarding learned weights.

    Action collapse makes a checkpoint unsafe for automatic continuation, but
    it does not invalidate the receipt-backed optimizer evidence in that run.
    Keep the candidate snapshot for audit/manual rollback and block only
    promotion/continuation.
    """

    if metrics_only or not candidate_exists:
        return manifest
    if selected_champion:
        manifest["checkpoint_persistence_decision"] = "champion_snapshot_selected"
        manifest["checkpoint_reuse_reason"] = (
            "terminal action distribution collapsed, but the same-window "
            "reconciled champion snapshot passed frozen validation"
        )
        return manifest
    if collapsed:
        manifest["checkpoint_persistence_decision"] = (
            "candidate_snapshot_saved_research_only_due_action_collapse"
        )
        manifest["checkpoint_reuse_method"] = "none"
        manifest["checkpoint_reuse_reason"] = (
            "terminal action distribution is collapsed; candidate weights are "
            "saved for research/audit but blocked from continuation and promotion"
        )
        manifest["promotable"] = False
        manifest["promotion_block_reason"] = "action_collapse_research_only"
    return manifest


def annotate_continuation_economic_verdict(manifest: dict[str, object]) -> dict[str, object]:
    """Compare a completed candidate with its explicit parent checkpoint.

    Research checkpoints remain available even when they regress economically;
    this function only records a deterministic next-seed recommendation.  The
    recommendation prevents an orchestration/UI layer from treating every
    receipt-backed checkpoint as a better continuation while preserving the
    user's ability to run an explicitly selected negative branch for research.
    """

    source_seed = str(manifest.get("source_seed", "") or "")
    seed_evidence = dict(manifest.get("source_seed_imitation", {}) or {})
    mt5 = dict(manifest.get("mt5", {}) or {})
    metrics = dict(mt5.get("metrics", {}) or {})
    paired_validation = dict(metrics.get("paired_seed_validation", {}) or {})
    paired_performed = bool(paired_validation.get("performed", False))
    parent_raw = seed_evidence.get("previous_net_profit")
    if paired_performed and paired_validation.get("candidate_net_profit") is not None:
        current_raw = paired_validation.get("candidate_net_profit")
        candidate_metric_source = "paired_seed_validation.candidate_net_profit"
        candidate_fitness_eligible = bool(
            paired_validation.get(
                "candidate_fitness_eligible", metrics.get("fitness_eligible", False)
            )
        )
    else:
        current_raw = metrics.get("net_profit")
        candidate_metric_source = "mt5.metrics.net_profit"
        candidate_fitness_eligible = bool(metrics.get("fitness_eligible", False))
    try:
        parent_profit = float(parent_raw)
        current_profit = float(current_raw)
    except (TypeError, ValueError):
        manifest["continuation_economic_verdict"] = {
            "available": False,
            "reason": "parent_or_candidate_net_profit_missing",
        }
        return manifest
    if not math.isfinite(parent_profit) or not math.isfinite(current_profit):
        manifest["continuation_economic_verdict"] = {
            "available": False,
            "reason": "parent_or_candidate_net_profit_non_finite",
        }
        return manifest

    parent_fitness_eligible = bool(seed_evidence.get("previous_fitness_eligible", False))
    delta = current_profit - parent_profit
    regression = bool(delta < -1.0e-9)
    if regression:
        economic_comparison = "regressed"
    elif not candidate_fitness_eligible:
        economic_comparison = "inconclusive_ineligible"
    else:
        economic_comparison = "eligible_non_regression"
    if candidate_fitness_eligible and not regression:
        recommendation = "candidate_preferred"
        preferred_seed = str(manifest.get("candidate", "") or "")
    elif source_seed:
        recommendation = "reuse_parent_seed_until_fitness_eligible"
        preferred_seed = source_seed
    else:
        recommendation = "no_preferred_continuation_seed"
        preferred_seed = ""
    manifest["continuation_economic_verdict"] = {
        "available": True,
        "parent_net_profit": parent_profit,
        "candidate_net_profit": current_profit,
        "net_profit_delta": delta,
        "economic_regression": regression,
        "parent_fitness_eligible": parent_fitness_eligible,
        "candidate_fitness_eligible": candidate_fitness_eligible,
        "candidate_metric_source": candidate_metric_source,
        "paired_validation_performed": paired_performed,
        "economic_comparison": economic_comparison,
        "recommendation": recommendation,
        "preferred_continuation_seed": preferred_seed,
    }
    manifest["preferred_continuation_seed"] = preferred_seed
    manifest["checkpoint_reuse_recommended"] = recommendation == "candidate_preferred"
    return manifest


def clone_seed_artifact(source: Path, target: Path) -> str:
    """Create an immutable seed path without duplicating multi-GiB bytes.

    Same-volume NTFS hard links keep independent directory entries and survive
    deletion of either name. The candidate workflow only reads this path and
    writes its learned state to a different checkpoint. Cross-volume or
    unsupported filesystems fall back to a physical copy.
    """
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy2"


def _inspect_mt5_continuation_seed(
    source: Path, *, checkpoint_sha256: str,
) -> dict[str, object] | None:
    """Validate an online candidate through the manifest that produced it.

    An MT5-trained checkpoint no longer has the exact bytes certified by the
    original imitation sidecar. It may still seed another isolated research
    run when its producer proves an unbroken, reconciled lineage.
    """

    manifest_path = source.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"MT5 continuation manifest khong doc duoc: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"MT5 continuation manifest khong phai JSON object: {manifest_path}")

    declared_hash = str(manifest.get("candidate_sha256", "") or "")
    declared_candidate = str(manifest.get("candidate", "") or "")
    mt5 = dict(manifest.get("mt5", {}) or {})
    metrics = dict(mt5.get("metrics", {}) or {})
    reconciliation = dict(metrics.get("reconciliation", {}) or {})
    replay = dict(metrics.get("mt5_failure_replay", {}) or {})
    replay_validation = dict(replay.get("post_replay_validation", {}) or {})
    paired_validation = dict(metrics.get("paired_seed_validation", {}) or {})
    upstream = dict(manifest.get("source_seed_imitation", {}) or {})

    manifest_schema = int(manifest.get("schema_version", 0) or 0)
    candidate_matches = bool(declared_candidate) and Path(declared_candidate).resolve() == source.resolve()
    hash_matches = bool(declared_hash) and declared_hash == str(checkpoint_sha256)
    inputs_immutable = bool(manifest.get("source_config_unchanged", False)) and bool(
        manifest.get("source_seed_unchanged", False)
    )
    execution_reconciled = bool(reconciliation.get("reconciled", False))
    producer_complete = (
        manifest_schema >= CANDIDATE_MANIFEST_SCHEMA_VERSION
        and str(manifest.get("mode", "")) == "mt5-smoke"
        and not bool(mt5.get("interrupted", False))
        and not mt5.get("training_error")
        and not manifest.get("run_exception")
    )
    paired_performed = bool(paired_validation.get("performed", False))
    replay_performed = bool(replay_validation.get("performed", False))
    if paired_performed:
        saved_weights_verified = bool(
            paired_validation.get("metrics_match_saved_weights", False)
        )
    elif replay_performed:
        saved_weights_verified = bool(
            replay_validation.get("metrics_match_saved_weights", False)
        )
    else:
        saved_weights_verified = False
    upstream_effective = bool(upstream.get("effective_for_mt5", False))
    upstream_autonomous_certified = bool(
        upstream.get("autonomous_validation_certified", False)
    )
    collapse_guard = dict(manifest.get("checkpoint_collapse_guard", {}) or {})
    checkpoint_not_collapsed = not bool(collapse_guard.get("collapsed", False))
    # These fields prove that the saved checkpoint came from the declared,
    # complete and reconciled producer.  They are sufficient to continue a
    # bounded research run, but deliberately not sufficient for promotion.
    research_continuation_allowed = bool(
        hash_matches
        and candidate_matches
        and inputs_immutable
        and execution_reconciled
        and producer_complete
        and checkpoint_not_collapsed
    )
    promotion_lineage_verified = bool(
        research_continuation_allowed
        and saved_weights_verified
        and upstream_effective
        and upstream_autonomous_certified
    )
    return {
        "declared": True,
        # Keep this strict flag for promotion/autonomous MT5 eligibility.
        "effective_for_mt5": promotion_lineage_verified,
        "research_continuation_allowed": research_continuation_allowed,
        "promotion_lineage_verified": promotion_lineage_verified,
        "reason": (
            "mt5_continuation_lineage" if promotion_lineage_verified
            else "mt5_research_continuation" if research_continuation_allowed
            else "invalid_mt5_continuation_lineage"
        ),
        "report_path": str(manifest_path),
        "lineage_type": "mt5_online_candidate_v1",
        "checkpoint_sha256_matches": hash_matches,
        "candidate_path_matches": candidate_matches,
        "source_inputs_unchanged": inputs_immutable,
        "execution_reconciled": execution_reconciled,
        "producer_complete": producer_complete,
        "checkpoint_not_collapsed": checkpoint_not_collapsed,
        "checkpoint_collapse_status": str(collapse_guard.get("status", "") or ""),
        "saved_weights_verified": saved_weights_verified,
        "paired_seed_validation_performed": paired_performed,
        "paired_seed_selected": str(paired_validation.get("selected", "") or ""),
        "upstream_effective_for_mt5": upstream_effective,
        "upstream_autonomous_validation_certified": upstream_autonomous_certified,
        "autonomous_validation_certified": bool(
            promotion_lineage_verified and upstream_autonomous_certified
        ),
        "upstream_report_path": str(upstream.get("report_path", "") or ""),
        "previous_fitness_eligible": bool(metrics.get("fitness_eligible", False)),
        "previous_net_profit": float(metrics.get("net_profit", 0.0) or 0.0),
    }


def inspect_imitation_seed(source: Path, *, checkpoint_sha256: str) -> dict[str, object]:
    """Read the small imitation sidecar without unpickling a large checkpoint.

    A sidecar is evidence only when it names the exact checkpoint hash. Legacy
    schema-2 candidates are reported explicitly as MARKET-only so an MT5 run
    cannot be mistaken for the new all-entry prior.
    """

    report_path = source.with_suffix(source.suffix + ".imitation.json")
    resume_path = source.with_suffix(source.suffix + ".resume.pkl")
    if resume_path.is_file():
        return {
            "declared": True,
            "effective_for_mt5": False,
            "reason": "imitation_paused",
            "report_path": str(report_path),
            "resume_path": str(resume_path),
            "checkpoint_sha256_matches": True,
            "validation_gate_enforced": False,
            "validation_gate_passed": False,
        }
    if not report_path.is_file():
        continuation = _inspect_mt5_continuation_seed(
            source, checkpoint_sha256=checkpoint_sha256,
        )
        if continuation is not None:
            return continuation
        return {
            "declared": False,
            "effective_for_mt5": False,
            "reason": "no_imitation_sidecar",
            "report_path": str(report_path),
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"imitation sidecar khong doc duoc: {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise RuntimeError(f"imitation sidecar khong phai JSON object: {report_path}")
    declared_hash = str(report.get("candidate_checkpoint_sha256", "") or "")
    if not declared_hash or declared_hash != str(checkpoint_sha256):
        raise RuntimeError(
            "imitation sidecar khong khop checkpoint: "
            f"declared={declared_hash or '<missing>'}, actual={checkpoint_sha256}"
        )
    schema_version = int(report.get("schema_version", 0) or 0)
    integration = dict(report.get("mt5_integration", {}) or {})
    entry_scope = str(integration.get("entry_scope", "market_only_v1") or "market_only_v1")
    representation_scope = str(
        integration.get("representation_scope", "frozen_recurrent_legacy")
        or "frozen_recurrent_legacy"
    )
    retention_required = bool(integration.get("requires_runtime_teacher_retention", False))
    geometry_scope = str(integration.get("geometry_scope", "") or "")
    geometry_heads = tuple(str(item) for item in (
        integration.get("geometry_heads", ()) or ()
    ))
    geometry_objective = str(integration.get("geometry_objective", "") or "")
    execution_contract_version = int(integration.get("execution_contract_version", 0) or 0)
    reload_passed = bool(dict(report.get("verification", {}) or {}).get(
        "candidate_reload_passed", False
    ))
    training = dict(report.get("training", {}) or {})
    validation_gate = dict(training.get("validation_gate", {}) or {})
    autonomous_gate = dict(training.get("autonomous_validation_gate", {}) or {})
    current_gate_required = True
    validation_gate_enforced = bool(validation_gate.get("enforced", False))
    validation_gate_passed = bool(validation_gate.get("passed", False))
    current_projection = representation_scope == CURRENT_IMITATION_REPRESENTATION_SCOPE
    autonomous_gate_required = True
    autonomous_gate_declared = bool(integration.get(
        "requires_autonomous_continuous_validation", False
    ))
    autonomous_gate_enforced = bool(autonomous_gate.get("enforced", False))
    autonomous_gate_passed = bool(autonomous_gate.get("passed", False))
    autonomous_validation_certified = bool(
        schema_version >= CURRENT_IMITATION_REPORT_SCHEMA_VERSION
        and current_projection
        and autonomous_gate_declared
        and autonomous_gate_enforced
        and autonomous_gate_passed
    )
    effective = bool(
        schema_version >= CURRENT_IMITATION_REPORT_SCHEMA_VERSION
        and entry_scope == "all_entry_mass_v2"
        and current_projection
        and retention_required
        and reload_passed
        and validation_gate_enforced
        and validation_gate_passed
        and autonomous_validation_certified
    )
    verification = dict(report.get("verification", {}) or {})
    geometry_reload_passed = bool(verification.get("geometry_checkpoint_reload_passed", False))
    geometry_effective = bool(
        effective
        and schema_version >= 6
        and geometry_scope in {
            "accepted_pivot_entries_v1",
            "accepted_pivot_entries_sl_only_v1",
        }
        and geometry_objective == "ordinal_log_distance_v2"
        and execution_contract_version >= 7
        and geometry_reload_passed
    )
    after = dict(training.get("after", {}) or {})
    test_metrics = dict(after.get("test", {}) or {})
    offline_test_metrics_available = all(
        name in test_metrics
        for name in ("balanced_accuracy", "entry_f0_5")
    )
    dataset_manifest = dict(report.get("dataset_manifest", {}) or {})
    geometry_training = dict(report.get("geometry_training", {}) or {})
    geometry_after = dict(geometry_training.get("after", {}) or {})
    geometry_test = dict(geometry_after.get("test", {}) or {})
    return {
        "declared": True,
        "effective_for_mt5": effective,
        "reason": (
            (
                "action_and_sl_geometry_with_runtime_retention"
                if geometry_scope == "accepted_pivot_entries_sl_only_v1"
                else "action_and_geometry_with_runtime_retention"
            ) if geometry_effective
            else "all_entry_mass_with_runtime_retention" if effective
            else "autonomous_validation_missing_or_failed" if not autonomous_validation_certified
            else "legacy_or_incomplete_imitation"
        ),
        "report_path": str(report_path),
        "report_schema_version": schema_version,
        "entry_scope": entry_scope,
        "representation_scope": representation_scope,
        "geometry_scope": geometry_scope,
        "geometry_heads": list(geometry_heads),
        "geometry_objective": geometry_objective,
        "execution_contract_version": execution_contract_version,
        "geometry_effective_for_mt5": geometry_effective,
        "geometry_checkpoint_reload_passed": geometry_reload_passed,
        "requires_runtime_teacher_retention": retention_required,
        "candidate_reload_passed": reload_passed,
        "validation_gate_required": current_gate_required,
        "validation_gate_enforced": validation_gate_enforced,
        "validation_gate_passed": validation_gate_passed,
        "autonomous_validation_gate_required": autonomous_gate_required,
        "autonomous_validation_gate_declared": autonomous_gate_declared,
        "autonomous_validation_gate_enforced": autonomous_gate_enforced,
        "autonomous_validation_gate_passed": autonomous_gate_passed,
        "autonomous_validation_certified": autonomous_validation_certified,
        "checkpoint_sha256_matches": True,
        "dataset_sample_count": int(dataset_manifest.get("sample_count", 0) or 0),
        "offline_test_metrics_available": offline_test_metrics_available,
        "test_balanced_accuracy": float(test_metrics.get("balanced_accuracy", 0.0) or 0.0),
        "test_entry_f0_5": float(test_metrics.get("entry_f0_5", 0.0) or 0.0),
        "geometry_test_sl_accuracy": float(geometry_test.get("sl_accuracy", 0.0) or 0.0),
        "geometry_test_tp_accuracy": float(geometry_test.get("tp_accuracy", 0.0) or 0.0),
        "geometry_test_joint_accuracy": float(geometry_test.get("joint_accuracy", 0.0) or 0.0),
    }


def apply_imitation_seed_gate_policy(
    evidence: dict[str, object], *, allow_unverified: bool,
) -> dict[str, object]:
    """Keep failed evidence visible while allowing an explicit research run."""

    result = dict(evidence)
    effective = bool(result.get("effective_for_mt5", False))
    research_continuation_allowed = bool(
        result.get("research_continuation_allowed", False)
    )
    bypassed = bool(allow_unverified and not effective)
    result["gate_bypass_requested"] = bool(allow_unverified)
    result["gate_bypassed"] = bypassed
    result["mt5_execution_allowed"] = bool(
        effective or research_continuation_allowed or bypassed
    )
    if research_continuation_allowed and not effective:
        result["execution_scope"] = "mt5_smoke_research_continuation"
        result["promotable"] = False
        result["promotion_block_reason"] = "promotion_lineage_unverified"
    if bypassed:
        result["bypass_scope"] = "mt5_smoke_research_only"
        result["promotable"] = False
    return result


def detect_runner_server(source: Path) -> str:
    """Read the exact cached server selected by this portable terminal."""

    common_ini = source / "Config" / "common.ini"
    last_error: Exception | None = None
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            text = common_ini.read_text(encoding=encoding)
        except (OSError, UnicodeError) as exc:
            last_error = exc
            continue
        match = re.search(r"(?mi)^Server\s*=\s*([^\r\n]+?)\s*$", text)
        if match:
            return match.group(1).strip()
    raise RuntimeError(f"khong doc duoc Server trong {common_ini}: {last_error}")


def clone_runner_candidate(
    source: Path,
    target: Path,
    *,
    required_symbol: str | None = None,
    required_server: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, object]:
    """Clone mutable tester state with a matched terminal/tester build.

    MetaTester services may keep the runner's legacy ``metatester64.exe``
    locked.  A signed, verified update can therefore live in
    ``runtime_overrides`` until those services are restarted.  Candidate
    clones must take both executables from the same override set; mixing MT5
    builds makes Terminal launch LiveUpdate between evaluation phases.
    """
    if target.exists():
        raise FileExistsError(f"runner clone already exists: {target}")
    target.mkdir(parents=True)
    for folder in (
        "Common/Files",
        "MQL5/Experts",
        "MQL5/Files",
        "MQL5/Profiles/Tester",
        "Reports",
        "Tester",
        "Tester/bases",
        "logs",
    ):
        (target / folder).mkdir(parents=True, exist_ok=True)
    uncompressed = _ensure_mt5_runtime_uncompressed(
        (
            target,
            target / "Common" / "Files",
            target / "MQL5" / "Experts",
            target / "MQL5" / "Files",
            target / "MQL5" / "Profiles",
            target / "Tester",
        )
    )
    if not bool(uncompressed.get("ok", False)):
        raise RuntimeError(
            "khong the bo NTFS compression khoi MT5 runner clone: "
            + json.dumps(uncompressed, ensure_ascii=False)
        )
    override_dir = source / "runtime_overrides"
    executable_source = source
    runtime_build = ""
    if override_dir.is_dir():
        required = tuple(override_dir / name for name in ("terminal64.exe", "metatester64.exe"))
        missing = [str(path) for path in required if not path.is_file()]
        build_file = override_dir / "build.txt"
        if missing or not build_file.is_file():
            raise RuntimeError(
                "runner runtime_overrides khong hoan chinh: "
                f"missing={missing}, build_file={build_file.is_file()}"
            )
        runtime_build = build_file.read_text(encoding="utf-8").strip()
        if not runtime_build.isdigit() or int(runtime_build) <= 0:
            raise RuntimeError(f"runner override build khong hop le: {runtime_build!r}")
        executable_source = override_dir
    for name in ("terminal64.exe", "metatester64.exe", "MetaEditor64.exe"):
        if (executable_source / name).is_file():
            shutil.copy2(executable_source / name, target / name)
    # Keep the build marker beside the matched runtime binaries.  The
    # terminal config is patched below, but MetaTrader's portable bootstrap
    # also consults the root-level marker when deciding whether the local
    # tester agent matches terminal64.exe.  Omitting it leaves a clone with
    # internally inconsistent runtime metadata and can stall before the EA
    # creates its first bridge request.
    if runtime_build:
        shutil.copy2(override_dir / "build.txt", target / "build.txt")
    experts = target / "MQL5" / "Experts"
    experts.mkdir(parents=True, exist_ok=True)
    for name in ("SelfGrowing_MT5_Trainer_EA.ex5", "SelfGrowing_MT5_Trainer_EA.mq5"):
        shutil.copy2(source / "MQL5" / "Experts" / name, experts / name)
    # Keep the EA's source dependency boundary complete.  MetaTrader can
    # recompile an expert when the terminal/compiler build or source timestamp
    # changes; a clone containing only .mq5/.ex5 then fails with the opaque
    # "cannot load expert" tester error because its Include tree is absent.
    include_source = source / "MQL5" / "Include"
    include_target = target / "MQL5" / "Include"
    if include_source.is_dir():
        shutil.copytree(include_source, include_target, copy_function=shutil.copy2)
    base_hardlinks = 0
    base_copies = 0
    isolated_market_data_copies = 0
    isolated_market_data_bytes = 0
    skipped_volatile_base_files: list[str] = []
    excluded_symbol_selection_files: list[str] = []
    deferred_locked_tick_files: list[str] = []
    excluded_symbol_trees: list[str] = []
    excluded_non_tester_trees: list[str] = []
    excluded_server_trees: list[str] = []
    excluded_regenerable_cache_trees: list[str] = []
    excluded_out_of_window_market_files: list[str] = []
    tester_market_data_copies = 0
    tester_market_data_bytes = 0
    non_tester_directory_names = {"mail", "news", "subscriptions", "signals", "trades"}
    required_symbol_key = str(required_symbol or "").strip().casefold()
    required_server_key = str(required_server or "").strip().casefold()
    window_start: datetime | None = None
    window_end: datetime | None = None
    if from_date and to_date:
        window_start, window_end = validate_date_window(from_date, to_date)
    required_tick_months: set[int] = set()
    required_history_years: set[int] = set()
    if window_start is not None and window_end is not None:
        # Include prior year so technical indicators (EMA 200, H1 62, ATR) have warmup bars immediately at window_start
        required_history_years.update(range(window_start.year - 1, window_end.year + 1))
        month_cursor = datetime(window_start.year, window_start.month, 1)
        end_month = datetime(window_end.year, window_end.month, 1)
        while month_cursor <= end_month:
            required_tick_months.add(month_cursor.year * 100 + month_cursor.month)
            if month_cursor.month == 12:
                month_cursor = datetime(month_cursor.year + 1, 1, 1)
            else:
                month_cursor = datetime(month_cursor.year, month_cursor.month + 1, 1)
    for root, dirs, files in os.walk(source / "Bases"):
        relative = Path(root).relative_to(source / "Bases")
        if required_server_key and relative == Path("."):
            selected_dirs: list[str] = []
            for dirname in dirs:
                if dirname.casefold() == required_server_key:
                    selected_dirs.append(dirname)
                else:
                    excluded_server_trees.append(dirname)
            dirs[:] = selected_dirs
        retained_after_cache: list[str] = []
        for dirname in dirs:
            if dirname.casefold() == "cache":
                excluded_regenerable_cache_trees.append((relative / dirname).as_posix())
            else:
                retained_after_cache.append(dirname)
        dirs[:] = retained_after_cache
        kept_dirs: list[str] = []
        for dirname in dirs:
            if dirname.casefold() in non_tester_directory_names:
                excluded_non_tester_trees.append((relative / dirname).as_posix())
            else:
                kept_dirs.append(dirname)
        dirs[:] = kept_dirs
        # The tester only needs the requested market's price history and ticks.
        # Keep every server's metadata, but never touch unrelated symbol trees:
        # another live tester may legitimately hold those files exclusively.
        if required_symbol_key and relative.parts and relative.parts[-1].casefold() in {
            "history", "ticks",
        }:
            retained_dirs: list[str] = []
            for dirname in dirs:
                if dirname.casefold() == required_symbol_key:
                    retained_dirs.append(dirname)
                else:
                    excluded_symbol_trees.append((relative / dirname).as_posix())
            dirs[:] = retained_dirs
        destination = target / "Bases" / relative
        destination.mkdir(parents=True, exist_ok=True)
        for filename in files:
            source_file = Path(root) / filename
            relative_file = source_file.relative_to(source / "Bases")
            relative_parts = tuple(part.casefold() for part in relative.parts)
            is_history_file = "history" in relative_parts and source_file.suffix.casefold() == ".hcc"
            is_tick_file = "ticks" in relative_parts and source_file.suffix.casefold() == ".tkc"
            if window_start is not None and window_end is not None:
                if is_tick_file and filename[:6].isdigit():
                    if int(filename[:6]) not in required_tick_months:
                        excluded_out_of_window_market_files.append(relative_file.as_posix())
                        continue
                if is_history_file and filename[:4].isdigit():
                    if int(filename[:4]) not in required_history_years:
                        excluded_out_of_window_market_files.append(relative_file.as_posix())
                        continue
            if (
                relative.parts
                and relative.parts[-1].casefold() == "symbols"
                and filename.casefold().startswith("selected-")
                and source_file.suffix.casefold() == ".dat"
            ):
                excluded_symbol_selection_files.append(relative_file.as_posix())
                continue
            destination_file = destination / filename
            # History and real-tick files are read-only inputs for the
            # candidate, but MetaTester may update their local indexes/cache
            # while a run is active.  Hard-linking them makes two concurrent
            # candidate runners share the same inode and can leave the second
            # tester with an incomplete market window.  Copy the retained
            # window-specific market files so parallel MT5 runs have a real
            # data boundary; keep cheap hardlinks for unrelated immutable base
            # metadata.
            market_data_file = bool(
                window_start is not None
                and window_end is not None
                and (is_history_file or is_tick_file)
            )
            if market_data_file:
                try:
                    shutil.copy2(source_file, destination_file)
                    isolated_market_data_copies += 1
                    isolated_market_data_bytes += int(source_file.stat().st_size)
                except PermissionError:
                    time.sleep(1.0)
                    try:
                        shutil.copy2(source_file, destination_file)
                        isolated_market_data_copies += 1
                        isolated_market_data_bytes += int(source_file.stat().st_size)
                        continue
                    except PermissionError:
                        pass
                    required_tick_file = bool(
                        required_symbol_key
                        and is_tick_file
                        and relative_file.suffix.casefold() == ".tkc"
                    )
                    if required_tick_file or is_history_file:
                        # A resident MetaTester service or terminal can exclusively lock
                        # its market cache. Leave the clone's file absent so MT5
                        # synchronizes or generates that exact segment safely.
                        deferred_locked_tick_files.append(relative_file.as_posix())
                        continue
                    raise
                continue
            try:
                os.link(source_file, destination_file)
                base_hardlinks += 1
            except OSError:
                # MT5 can hold small mutable Base files such as alerts.dat in
                # a mode that rejects hard-link creation while normal reads
                # still work.  Copy only those files; retained market data
                # already took the isolated-copy path above.
                try:
                    shutil.copy2(source_file, destination_file)
                    base_copies += 1
                except PermissionError:
                    relative_parts = tuple(part.casefold() for part in relative_file.parts)
                    required_tick_file = bool(
                        required_symbol_key
                        and "ticks" in relative_parts
                        and required_symbol_key in relative_parts
                        and relative_file.suffix.casefold() == ".tkc"
                    )
                    if required_tick_file:
                        # A resident MetaTester service can exclusively lock its
                        # terminal tick cache.  Leave the clone's file absent so
                        # MT5 downloads that exact real-tick month into the
                        # isolated runner.  Model/history quality is verified in
                        # the retained Strategy Tester report after the run.
                        deferred_locked_tick_files.append(relative_file.as_posix())
                        continue
                    if not (
                        relative_file.parent == Path(".")
                        and relative_file.suffix.lower() == ".dat"
                    ):
                        raise
                    # Root-level .dat files contain terminal-local alerts,
                    # books, options or UI state. They are unrelated to the
                    # server history below Bases/<server> and may be locked.
                    skipped_volatile_base_files.append(relative_file.as_posix())
    # MetaTester keeps an additional, tester-owned history cache under
    # Tester/bases.  The terminal Bases tree alone is insufficient for a fresh
    # portable clone: without this cache MT5 may redownload M1 data and hit the
    # short preliminary-history timeout.  Copy it per worker (never hard-link)
    # so parallel workers cannot mutate the same cache inode.
    tester_bases_source = source / "Tester" / "bases"
    if tester_bases_source.is_dir():
        for root, dirs, files in os.walk(tester_bases_source):
            relative = Path(root).relative_to(tester_bases_source)
            if required_server_key and relative.parts:
                if relative.parts[0].casefold() != required_server_key:
                    dirs[:] = []
                    continue
            if required_symbol_key and relative.parts and relative.parts[-1].casefold() in {
                "history", "ticks",
            }:
                dirs[:] = [
                    dirname
                    for dirname in dirs
                    if dirname.casefold() == required_symbol_key
                ]
            destination = target / "Tester" / "bases" / relative
            destination.mkdir(parents=True, exist_ok=True)
            for filename in files:
                source_file = Path(root) / filename
                relative_parts = tuple(part.casefold() for part in relative.parts)
                is_tick_file = "ticks" in relative_parts and source_file.suffix.casefold() == ".tkc"
                if window_start is not None and window_end is not None:
                    if is_tick_file and filename[:6].isdigit():
                        if int(filename[:6]) not in required_tick_months:
                            continue
                destination_file = destination / filename
                shutil.copy2(source_file, destination_file)
                tester_market_data_copies += 1
                tester_market_data_bytes += int(source_file.stat().st_size)
    shutil.copytree(source / "Config", target / "Config")
    if runtime_build:
        terminal_ini = target / "Config" / "terminal.ini"
        try:
            text = terminal_ini.read_text(encoding="utf-16")
            used_enc = "utf-16"
        except Exception:
            text = terminal_ini.read_text(encoding="utf-8", errors="replace")
            used_enc = "utf-8"
        for key in (
            "LastBuildDataPath", "LastBuild", "LastBuildIDE",
            "LastBuildTester", "ProgramBuild",
        ):
            text = re.sub(rf"{key}=\d+", f"{key}={runtime_build}", text)
        terminal_ini.write_text(text, encoding=used_enc)
    return {
        "runtime_override": bool(runtime_build),
        "runtime_build": runtime_build,
        "mt5_runtime_uncompressed": uncompressed,
        "terminal_sha256": sha256_file(target / "terminal64.exe"),
        "metatester_sha256": sha256_file(target / "metatester64.exe"),
        "base_hardlinks": base_hardlinks,
        "base_copies": base_copies,
        "isolated_market_data_copies": isolated_market_data_copies,
        "isolated_market_data_bytes": isolated_market_data_bytes,
        "skipped_volatile_base_files": skipped_volatile_base_files,
        "excluded_symbol_selection_files": sorted(excluded_symbol_selection_files),
        "deferred_locked_tick_files": sorted(deferred_locked_tick_files),
        "required_symbol": str(required_symbol or ""),
        "required_server": str(required_server or ""),
        "excluded_symbol_trees": sorted(excluded_symbol_trees),
        "excluded_non_tester_trees": sorted(excluded_non_tester_trees),
        "excluded_server_trees": sorted(excluded_server_trees),
        "excluded_regenerable_cache_trees": sorted(excluded_regenerable_cache_trees),
        "excluded_out_of_window_market_files": sorted(excluded_out_of_window_market_files),
        "tester_market_data_copies": tester_market_data_copies,
        "tester_market_data_bytes": tester_market_data_bytes,
        "market_data_window": [str(from_date or ""), str(to_date or "")],
    }


def candidate_config(base, profile: dict[str, Any], output: Path):
    learning_overrides = dict(profile.get("learning_overrides", {}) or {})
    app_overrides = dict(profile.get("app_overrides", {}) or {})
    expectancy_overrides = dict(profile.get("expectancy_overrides", {}) or {})
    order_overrides = dict(profile.get("order_overrides", {}) or {})
    learning = replace(base.learning, **learning_overrides)
    expectancy = replace(base.expectancy, **expectancy_overrides)
    orders = replace(base.orders, **order_overrides) if order_overrides else base.orders
    isolation = dict(profile.get("isolation", {}) or {})
    observability = replace(
        base.observability,
        enabled=True,
        event_path=str(output / "events.jsonl"),
        decision_profile_path=str(output / "decision_profiles.jsonl"),
        structural_event_path=str(output / "structural_events.jsonl"),
        network_development_path=str(output / "network_development.jsonl"),
        max_event_bytes=int(isolation.get("event_jsonl_max_bytes", 32 * 1024 * 1024)),
    )
    return replace(
        base,
        expectancy=expectancy,
        learning=learning,
        observability=observability,
        orders=orders,
        **app_overrides,
    )


def build_candidate_feedback_contract(
    config: Any,
    profile: dict[str, Any],
    *,
    source_config: str = "",
    profile_path: str = "",
) -> dict[str, object]:
    """Describe the feedback/optimizer contract that will reach MT5.

    A candidate profile is the authority for whether an MT5 run is expected
    to learn.  Keeping this decision explicit prevents a stale UI/profile
    combination from silently launching a telemetry-only tester while its
    label says ``reasoning`` or ``Codex teacher``.
    """

    experiment = dict(profile.get("experiment", {}) or {})
    experiment_id = str(experiment.get("id", "") or "").strip()
    learning = getattr(config, "learning", None)
    expert_config = getattr(config, "he_chuyen_gia", None)
    decision_mode = str(getattr(
        expert_config, "che_do_quyet_dinh", ""
    ) or "")
    direct_rnn_full_rl_enabled = bool(
        decision_mode == "mot_context_mot_chuyen_gia"
        and getattr(learning, "direct_rnn_full_rl_enabled", False)
    )
    codex_teacher_requested = bool(getattr(
        learning, "five_codex_teacher_enabled", False
    ))
    effective = {
        "mt5_learning_enabled": bool(getattr(config, "mt5_learning_enabled", False)),
        "five_reasoning_rl_enabled": bool(getattr(
            learning, "five_reasoning_rl_enabled", False
        )),
        "five_reasoning_action_scale": float(getattr(
            learning, "five_reasoning_action_scale", 0.0
        ) or 0.0),
        "five_pattern_reward_enabled": bool(getattr(
            learning, "five_pattern_reward_enabled", False
        )),
        "five_codex_teacher_enabled": bool(getattr(
            learning, "five_codex_teacher_enabled", False
        )),
        "five_codex_teacher_auto_connect": bool(getattr(
            learning, "five_codex_teacher_auto_connect", False
        )),
        "five_codex_teacher_hypothesis_required": bool(getattr(
            learning, "five_codex_teacher_hypothesis_required", True
        )),
        "five_codex_teacher_effective": False,
        "codex_cli_found": False,
        "codex_preflight_passed": False,
        "codex_receipts_valid": False,
        "codex_hypotheses_confirmed": 0,
        "codex_optimizer_updates": 0,
        "optimizer_state_stripped": False,
        "optimizer_state_reinitialized_for_research": False,
        "optimizer_state_resume_mode": "continuation",
        # Keep the crash-recovery cadence visible in the candidate contract.
        # Recovery snapshots remain research-only, but without this field a
        # run could silently fall back to final-only persistence.
        "five_rlhf_recovery_snapshot_interval_updates": int(getattr(
            learning,
            "five_rlhf_recovery_snapshot_interval_updates",
            0,
        ) or 0),
        "five_rlhf_allow_optimizer_state_reinitialization_for_research": bool(
            getattr(
                learning,
                "five_rlhf_allow_optimizer_state_reinitialization_for_research",
                False,
            )
        ),
        "five_codex_teacher_model": str(getattr(
            learning, "five_codex_teacher_model", "gpt-5.6-luna"
        ) or "gpt-5.6-luna"),
        "codex_teacher_mode": decision_mode,
        "codex_direct_rnn_full_rl_enabled": direct_rnn_full_rl_enabled,
        "codex_parameter_scope": (
            "shared_five_action_head"
            if decision_mode == "nam_tac_nhan_dung_chung"
            else (
                "direct_rnn_full_rl_adamw_v1"
                if direct_rnn_full_rl_enabled
                else "legacy_direct_rnn_action_bias_adamw"
            )
        ),
        "five_codex_teacher_online_wait_sec": float(getattr(
            learning, "five_codex_teacher_online_wait_sec", 30.0
        ) or 0.0),
        "five_codex_teacher_max_workers": int(getattr(
            learning, "five_codex_teacher_max_workers", 2
        ) or 2),
        # Codex candidate runs must expose the same evidence-gated action
        # veto that the policy applies at decision time.  These are masks,
        # not logit rewrites: they require fresh lifecycle evidence and keep
        # HOLD available, so the contract can prove the selector is not still
        # sampling a known-bad entry merely because the profile omitted an
        # override.
        "five_expectancy_action_veto_enabled": bool(getattr(
            learning, "five_expectancy_action_veto_enabled", False
        )) or codex_teacher_requested,
        "five_expectancy_action_veto_min_samples": int(getattr(
            learning, "five_expectancy_action_veto_min_samples", 12
        ) or 12),
        "five_expectancy_action_veto_min_confidence": float(getattr(
            learning, "five_expectancy_action_veto_min_confidence", 0.20
        ) or 0.0),
        "five_expectancy_action_veto_min_lcb_r": float(getattr(
            learning, "five_expectancy_action_veto_min_lcb_r", -0.15
        )),
        "five_expectancy_action_veto_hold_owner_on_negative": bool(getattr(
            learning,
            "five_expectancy_action_veto_hold_owner_on_negative",
            codex_teacher_requested,
        )),
        "five_live_action_veto_enabled": bool(getattr(
            learning, "five_live_action_veto_enabled", False
        )) or codex_teacher_requested,
        "five_live_action_veto_min_samples": int(getattr(
            learning, "five_live_action_veto_min_samples", 12
        ) or 12),
        "five_live_action_veto_min_confidence": float(getattr(
            learning, "five_live_action_veto_min_confidence", 0.20
        ) or 0.0),
        "five_live_action_veto_min_lcb_r": float(getattr(
            learning, "five_live_action_veto_min_lcb_r", -0.15
        )),
        "five_live_action_veto_hold_owner_on_negative": bool(getattr(
            learning,
            "five_live_action_veto_hold_owner_on_negative",
            True,
        )),
        "five_live_action_veto_probe_enabled": bool(getattr(
            learning, "five_live_action_veto_probe_enabled", False
        )) or codex_teacher_requested,
        "five_live_action_veto_probe_on_negative_transition": bool(getattr(
            learning, "five_live_action_veto_probe_on_negative_transition", False
        )) or codex_teacher_requested,
        "five_live_action_veto_probe_interval_decisions": int(getattr(
            learning, "five_live_action_veto_probe_interval_decisions", 64
        ) or 64),
        "five_live_owner_exploration_enabled": bool(getattr(
            learning, "five_live_owner_exploration_enabled", False
        )) or codex_teacher_requested,
        "five_live_owner_exploration_min_samples": int(getattr(
            learning, "five_live_owner_exploration_min_samples", 12
        ) or 12),
        "five_live_owner_exploration_min_confidence": float(getattr(
            learning, "five_live_owner_exploration_min_confidence", 0.20
        ) or 0.0),
        "five_live_owner_exploration_min_lcb_r": float(getattr(
            learning, "five_live_owner_exploration_min_lcb_r", -0.15
        )),
        "five_live_owner_exploration_min_mix": float(getattr(
            learning, "five_live_owner_exploration_min_mix", 0.02
        ) or 0.0),
        "five_live_owner_bootstrap_probe_enabled": bool(getattr(
            learning, "five_live_owner_bootstrap_probe_enabled", False
        )),
        "five_live_owner_bootstrap_probe_interval_decisions": int(getattr(
            learning, "five_live_owner_bootstrap_probe_interval_decisions", 16
        ) or 16),
        "five_live_owner_bootstrap_probe_max_per_run": int(getattr(
            learning, "five_live_owner_bootstrap_probe_max_per_run", 2
        ) or 0),
        "five_live_owner_bootstrap_probe_force_market": bool(getattr(
            learning, "five_live_owner_bootstrap_probe_force_market", False
        )),
    }
    learning_profiles = {
        "reasoning_rl_v1_active_candidate",
        "codex_teacher_every_close_candidate",
    }
    expected_learning = bool(
        experiment_id in learning_profiles
        or (
            effective["five_codex_teacher_auto_connect"]
            and effective["mt5_learning_enabled"]
        )
    )
    contract = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "source_config": str(source_config or ""),
        "profile": str(profile_path or ""),
        "expected_feedback_to_learner": bool(expected_learning),
        "effective": effective,
        "negative_close_route": (
            (
                "codex_teacher_receipt_to_five_adam"
                if decision_mode == "nam_tac_nhan_dung_chung"
                else (
                    "codex_teacher_receipt_to_direct_rnn_full_rl_adamw"
                    if direct_rnn_full_rl_enabled
                    else "codex_teacher_receipt_to_legacy_direct_rnn_adamw"
                )
            )
            if (
                effective["five_codex_teacher_enabled"]
                or effective["five_codex_teacher_auto_connect"]
            )
            else (
                "owner_lifecycle_reward_to_direct_rnn_full_rl_adamw"
                if direct_rnn_full_rl_enabled
                else "owner_lifecycle_reward_to_five_adam"
            )
        ),
        "codex_preflight": {"status": "pending", "reason": "runtime_preflight_not_run"},
        "optimizer": (
            "five_ppo_adam"
            if decision_mode == "nam_tac_nhan_dung_chung"
            else (
                "direct_rnn_full_rl_adamw"
                if direct_rnn_full_rl_enabled
                else "legacy_direct_rnn_adamw"
            )
        ),
        "violations": [],
    }
    violations: list[str] = []
    if expected_learning and not effective["mt5_learning_enabled"]:
        violations.append("profile_requires_mt5_learning_enabled=true")
    if experiment_id == "reasoning_rl_v1_active_candidate":
        if not effective["five_reasoning_rl_enabled"]:
            violations.append("reasoning profile requires five_reasoning_rl_enabled=true")
        if effective["five_reasoning_action_scale"] < 0.0:
            violations.append("reasoning profile requires five_reasoning_action_scale>=0")
        if not effective["five_pattern_reward_enabled"]:
            violations.append("reasoning profile requires five_pattern_reward_enabled=true")
    if experiment_id == "codex_teacher_every_close_candidate":
        if not effective["five_codex_teacher_enabled"]:
            violations.append("Codex profile requires five_codex_teacher_enabled=true")
    contract["violations"] = violations
    return contract


def validate_candidate_feedback_contract(contract: dict[str, object]) -> None:
    """Fail before MT5 starts when a learning candidate is not learnable."""

    violations = [str(item) for item in (contract.get("violations", []) or [])]
    if violations:
        raise RuntimeError(
            "candidate feedback contract rejected; MT5 would not update the learner: "
            + "; ".join(violations)
        )


def configure_candidate(system: object, config: object) -> None:
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    requested_authority = str(
        getattr(getattr(config, "learning", None), "tensor_rule_authority", "telemetry")
        or "telemetry"
    ).strip().lower()
    if requested_authority not in {"telemetry", "shadow", "residual", "primary"}:
        raise ValueError(f"invalid candidate TensorRule authority: {requested_authority}")
    for expert in getattr(system, "experts", {}).values():
        local = expert.policy
        local.tensor_rule_authority = requested_authority
        # Public Five owns the 137/20 decision surface directly and deliberately
        # has no legacy TensorRule module list. Candidate tooling must still
        # configure Five in python-debug mode without indexing modules[0].
        if not getattr(local, "modules", None):
            continue
        bank = local.modules[0].tensor_rule_bank
        if isinstance(bank, TensorRuleBankState):
            bank.authority = requested_authority


def validate_candidate(system: object, *, expected_authority: str = "telemetry") -> dict[str, object]:
    rows: list[dict[str, object]] = []
    bank_ids: set[int] = set()
    bank_fingerprints: set[str] = set()
    parameter_fingerprints: set[str] = set()
    memory_ids: set[int] = set()
    memory_parameter_fingerprints: set[str] = set()
    experts = getattr(system, "experts", {})
    first_policy = next(iter(experts.values())).policy if experts else None
    if first_policy is not None and (not getattr(first_policy, "modules", None) or len(first_policy.modules) == 0):
        shared_net = getattr(first_policy, "five_shared_network", None)
        if shared_net is not None:
            return {
                "schema_version": 2,
                "architecture": getattr(shared_net, "policy_architecture_version", 2),
                "model_type": type(shared_net).__name__,
                "parameter_count": int(shared_net.parameter_count()),
                "five_shared": True,
                "experts": [str(owner) for owner in sorted(experts.keys())],
            }
    for owner, expert in sorted(getattr(system, "experts", {}).items()):
        local = expert.policy
        module = local.modules[0]
        bank = getattr(module, "tensor_rule_bank", None)
        if not isinstance(bank, TensorRuleBankState):
            raise RuntimeError(f"expert {owner} has no TensorRuleBank")
        if str(bank.authority) != str(expected_authority):
            raise RuntimeError(
                f"expert {owner} authority={bank.authority!r}, expected {expected_authority!r}"
            )
        if int(bank.schema_version) < 2:
            raise RuntimeError(f"expert {owner} has an obsolete tensor-rule schema")
        if float(bank.lifecycle_encoder_credit_scale) != 0.0:
            raise RuntimeError(
                f"expert {owner} allows realized lifecycle PnL to update the shared encoder"
            )
        if bank.action_adapter_w.shape != (
            6, bank.component_size * 5, bank.action_latent_size
        ):
            raise RuntimeError(f"expert {owner} has invalid action-conditioned adapters")
        if id(bank) in bank_ids:
            raise RuntimeError("TensorRuleBank was shared across experts")
        bank_ids.add(id(bank))
        bank_fingerprints.add(str(bank.seed_fingerprint))
        parameter_fingerprint = hashlib.sha256(
            np.ascontiguousarray(np.asarray(bank.input_w, dtype=np.float32)).tobytes()
            + np.ascontiguousarray(np.asarray(bank.action_adapter_w, dtype=np.float32)).tobytes()
        ).hexdigest()
        parameter_fingerprints.add(parameter_fingerprint)
        memory = getattr(module, "state_alias_memory", None)
        if not isinstance(memory, StateAliasingMemoryState) or not memory.enabled:
            raise RuntimeError(f"expert {owner} has no enabled state-alias memory")
        if id(memory) in memory_ids:
            raise RuntimeError("StateAliasingMemory was shared across experts")
        memory_ids.add(id(memory))
        memory_parameter_fingerprint = hashlib.sha256(
            np.ascontiguousarray(np.asarray(memory.query_w, dtype=np.float32)).tobytes()
        ).hexdigest()
        memory_parameter_fingerprints.add(memory_parameter_fingerprint)
        local.reset_runtime_state()
        enabled = local.copy()
        gate_disabled = local.copy()
        gate_disabled.tensor_rule_entry_lcb_gate_enabled = False
        tensor_disabled = local.copy()
        tensor_disabled.tensor_rule_enabled = False
        observation = np.linspace(-0.4, 0.6, INPUT_COUNT, dtype=float)
        enabled_trace = None
        gate_disabled_trace = None
        tensor_disabled_trace = None
        warmup_steps = max(1, int(memory.min_history) + 1)
        for _ in range(warmup_steps):
            enabled_trace = enabled.act(observation)
            gate_disabled_trace = gate_disabled.act(observation)
            tensor_disabled_trace = tensor_disabled.act(observation)
        assert enabled_trace is not None
        assert gate_disabled_trace is not None and tensor_disabled_trace is not None
        # Legacy compatibility is defined by the explicit gate-off candidate:
        # UNKNOWN residual TensorRule must remain action-neutral there.
        np.testing.assert_array_equal(
            gate_disabled_trace.selected_action,
            tensor_disabled_trace.selected_action,
        )
        gate_reason = str(enabled_trace.action_details.get(
            "tensor_rule_entry_lcb_gate_reason", ""
        ))
        if bool(getattr(enabled, "tensor_rule_entry_lcb_gate_enabled", False)):
            if gate_reason == "unknown_hold":
                if int(np.argmax(enabled_trace.selected_action[:7])) != 0:
                    raise RuntimeError(f"expert {owner} cold gate did not force HOLD")
            elif gate_reason not in {"exploration_entry", "exploration_hold"}:
                raise RuntimeError(
                    f"expert {owner} has unexpected cold gate reason {gate_reason!r}"
                )
        if not bool(enabled_trace.action_details.get("direct_rnn_state_memory_ready", False)):
            raise RuntimeError(f"expert {owner} state-alias memory did not become ready")
        rows.append({
            "owner": str(owner),
            "artifact_version": int(local.artifact_version),
            "parameter_count": int(bank.parameter_count()),
            "seed_fingerprint": str(bank.seed_fingerprint),
            "initial_parameter_fingerprint": parameter_fingerprint,
            "state_memory_parameter_count": int(memory.summary()["parameter_count"]),
            "state_memory_initial_parameter_fingerprint": memory_parameter_fingerprint,
            "state_memory_ready": bool(enabled_trace.action_details.get(
                "direct_rnn_state_memory_ready", False
            )),
            "state_memory_warmup_steps": int(warmup_steps),
            "cold_unknown_gate_reason": gate_reason,
            "legacy_gate_off_action_unchanged": True,
            "tensor_rule_authority": str(bank.authority),
            "unknown_reason": str(enabled_trace.tensor_rule_trace.get("unknown_reason", "")),
            "entry_exploration_configured": bool(enabled_trace.action_details.get(
                "direct_rnn_rl_exploration_configured", False
            )),
            "entry_exploration_active": bool(enabled_trace.action_details.get(
                "direct_rnn_rl_exploration_active", False
            )),
            "entry_exploration_rate": float(getattr(
                local, "direct_rnn_rl_exploration_rate", 0.0
            ) or 0.0),
            "entry_exploration_behavior_probability": float(
                enabled_trace.action_details.get("direct_rnn_rl_behavior_probability", 0.0) or 0.0
            ),
        })
    if len(bank_fingerprints) != len(rows):
        raise RuntimeError("seed fingerprints are not independent across experts")
    if len(parameter_fingerprints) != len(rows):
        raise RuntimeError("tensor parameters were identically seeded across experts")
    if len(memory_ids) != len(rows):
        raise RuntimeError("state-alias memory ownership is not independent")
    if len(memory_parameter_fingerprints) != len(rows):
        raise RuntimeError("state-alias memories were identically seeded across experts")
    return {
        "experts": rows,
        "expert_count": len(rows),
        "independent_banks": True,
        "independent_state_memories": True,
        "entry_exploration_enabled": all(
            bool(row["entry_exploration_configured"]) for row in rows
        ),
        "legacy_train_entry_probe_enabled": False,
        "artifact_version_at_least": max(
            TENSOR_RULE_ARTIFACT_VERSION, STATE_ALIAS_MEMORY_ARTIFACT_VERSION
        ),
        "bridge_shape": [INPUT_COUNT, OUTPUT_COUNT],
    }


def atomic_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def incomplete_checkpoint_manifest(error: BaseException) -> dict[str, object] | None:
    checkpoint = Path(str(getattr(error, "checkpoint_path", "") or ""))
    metadata = Path(str(getattr(error, "metadata_path", "") or ""))
    if not checkpoint.is_file():
        return None
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "metadata": str(metadata) if metadata.is_file() else "",
        "promotable": False,
        "original_error_type": str(getattr(error, "original_error_type", "") or ""),
    }


def incomplete_checkpoint_manifest_from_result(
    result: object,
) -> dict[str, object] | None:
    """Build the same artifact record when MT5 returns an interrupted result.

    Wall-clock and save-and-stop boundaries are returned as a structured MT5
    result rather than raised exceptions.  Keep that path equivalent to the
    exception path so the runner cannot lose a valid partial checkpoint merely
    because the interruption was reported in-band.
    """

    if not isinstance(result, dict):
        return None
    checkpoint_meta = dict(result.get("checkpoint", {}) or {})
    checkpoint = Path(str(checkpoint_meta.get("incomplete_artifact", "") or ""))
    metadata = Path(str(checkpoint_meta.get("incomplete_metadata", "") or ""))
    if not checkpoint.is_file():
        return None
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "metadata": str(metadata) if metadata.is_file() else "",
        "promotable": False,
        "original_error_type": str(
            dict(result.get("metrics", {}) or {}).get("interrupted_reason", "") or ""
        ),
    }


def expose_incomplete_checkpoint(
    manifest: dict[str, object],
    *,
    incomplete: dict[str, object] | None,
    candidate_path: Path,
    metrics_only: bool,
) -> dict[str, object]:
    """Expose a persisted partial checkpoint without making it a continuation seed.

    ``huan_luyen_he_chuyen_gia_mt5`` writes ``*.incomplete.pkl`` before raising
    or returning an interrupted result on a wall-time/stop boundary.  The old
    runner only recorded that path under ``incomplete_checkpoint``; the UI and
    downstream artifact discovery then
    reported no candidate at all even though learned weights were on disk.  An
    incomplete snapshot is useful for receipt/audit inspection, so publish it
    as the run's research candidate while keeping continuation and promotion
    fail-closed.
    """

    if metrics_only or not incomplete or candidate_path.is_file():
        return manifest
    checkpoint = Path(str(incomplete.get("path", "") or ""))
    if not checkpoint.is_file():
        return manifest
    manifest["candidate"] = str(checkpoint)
    manifest["candidate_sha256"] = str(
        incomplete.get("sha256", "") or sha256_file(checkpoint)
    )
    manifest["candidate_bytes"] = int(
        incomplete.get("bytes", checkpoint.stat().st_size) or checkpoint.stat().st_size
    )
    manifest["checkpoint_persistence_decision"] = (
        "incomplete_snapshot_saved_research_only"
    )
    manifest["checkpoint_reuse_method"] = "none"
    manifest["checkpoint_reuse_reason"] = (
        "MT5 stopped before a complete candidate was finalized; incomplete "
        "weights are saved for research/audit but blocked from continuation "
        "and promotion"
    )
    manifest["checkpoint_incomplete"] = True
    manifest["promotable"] = False
    manifest["promotion_block_reason"] = "incomplete_run_research_only"
    return manifest


class StorageMonitor:
    def __init__(
        self,
        output: Path,
        runner: Path,
        quota: int,
        reserve: int,
        *,
        excluded_paths: tuple[Path, ...] = (),
    ) -> None:
        self.output = output.resolve()
        self.runner = runner.resolve()
        self.quota = int(quota)
        self.reserve = int(reserve)
        self.excluded_paths = tuple(item.resolve() for item in excluded_paths)
        self.stop = threading.Event()
        self.failure: dict[str, object] = {}
        self.thread = threading.Thread(target=self._run, name="candidate-storage-gate", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=10.0)

    def _terminate_candidate_tester(self) -> None:
        escaped = str(self.runner).replace("'", "''")
        command = (
            f"$root=[IO.Path]::GetFullPath('{escaped}'); "
            "Get-CimInstance Win32_Process | Where-Object { "
            "$_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath).StartsWith($root,[StringComparison]::OrdinalIgnoreCase) "
            "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            check=False, capture_output=True, text=True, timeout=30,
        )

    def _run(self) -> None:
        while not self.stop.wait(5.0):
            free = shutil.disk_usage(self.output.anchor).free
            total_size = directory_size(self.output)
            payload_size = directory_size(self.output, exclude=self.excluded_paths)
            if free >= self.reserve and payload_size <= self.quota:
                continue
            self.failure = {
                "schema_version": STORAGE_GATE_SCHEMA_VERSION,
                "type": "storage_gate_failure",
                "free_bytes": int(free),
                "reserve_floor_bytes": int(self.reserve),
                "candidate_total_bytes": int(total_size),
                "candidate_payload_bytes": int(payload_size),
                "candidate_payload_quota_bytes": int(self.quota),
                "excluded_protected_bytes": int(max(0, total_size - payload_size)),
                "timestamp": time.time(),
            }
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "storage_gate_failure.json").write_text(
                json.dumps(self.failure, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._terminate_candidate_tester()
            return


def metrics_only_result_complete_for_storage_pressure(
    manifest: dict[str, object],
    *,
    candidate_payload_bytes: int,
    candidate_payload_quota: int,
) -> bool:
    """Return whether metrics-only output is complete enough to warn, not fail.

    The storage monitor can cross its floor while the final MT5 report and
    receipt files are already complete. Such a pressure event must remain
    visible in the manifest, but it should not turn an otherwise reconciled
    metrics-only research run into a misleading training failure. Persistent
    candidate runs never use this helper and remain fail-closed.
    """

    if int(candidate_payload_bytes) > int(candidate_payload_quota):
        return False
    result = manifest.get("mt5")
    if not isinstance(result, dict):
        return False
    if bool(result.get("interrupted", False)):
        return False
    if str(result.get("training_error", "") or "").strip():
        return False
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return False
    reconciliation = metrics.get("reconciliation")
    if not isinstance(reconciliation, dict) or not bool(
        reconciliation.get("reconciled", False)
    ):
        return False
    feedback = manifest.get("feedback_contract")
    phase = feedback.get("codex_teacher_phase") if isinstance(feedback, dict) else None
    if isinstance(phase, dict) and int(phase.get("pending", 0) or 0) != 0:
        return False
    return True


def safe_remove_runner(runner: Path, parent: Path) -> dict[str, object]:
    resolved = runner.resolve()
    resolved_parent = parent.resolve()
    if resolved.parent != resolved_parent or not resolved.name.startswith("mt5_v186_"):
        raise RuntimeError(f"refusing to remove unexpected runner path: {resolved}")
    if not resolved.exists():
        return {"removed": True, "path": str(resolved), "attempts": 0}
    last_error = ""
    for attempt in range(1, 4):
        try:
            shutil.rmtree(resolved)
            return {"removed": True, "path": str(resolved), "attempts": attempt}
        except PermissionError as exc:
            last_error = str(exc)
            time.sleep(1.0)
    return {
        "removed": False, "path": str(resolved), "attempts": 3,
        "reason": "locked_file_cleanup_deferred", "error": last_error,
    }


def allocate_candidate_runner(output: Path) -> tuple[Path, Path]:
    """Allocate a short runner namespace unique per candidate process.

    MT5's Windows agent has a substantially shorter effective path limit than
    modern Python/NTFS. Deep candidate output names can therefore make an
    otherwise valid ``MQL5/Experts/*.ex5`` unreadable. Keep the mutable MT5
    tree below a short workspace-local root while preserving a unique process
    namespace; diagnostics and manifests still point back to the candidate
    output directory.
    """

    output = output.resolve()
    runner_parent = (Path.cwd() / ".mt5_runners").resolve()
    runner_parent.mkdir(parents=True, exist_ok=True)
    runner = runner_parent / f"mt5_v186_{os.getpid()}_{time.time_ns()}"
    return runner_parent, runner


def preserve_runner_diagnostics(
    runner: Path,
    output: Path,
    *,
    max_bytes: int = 128 * 1024**2,
) -> dict[str, object]:
    """Giữ log/report nhỏ của runner lỗi trước khi xóa runner tạm."""

    runner = runner.resolve()
    destination = (output / "mt5_failure_diagnostics").resolve()
    allowed_suffixes = {".log", ".ini", ".set", ".htm", ".html", ".txt"}
    copied: list[str] = []
    copied_bytes = 0
    for folder_name in ("Reports", "logs", "Tester", "MQL5/Logs", "MQL5/Experts"):
        folder = runner / folder_name
        if not folder.is_dir():
            continue
        for source in sorted(folder.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in allowed_suffixes:
                continue
            try:
                size = int(source.stat().st_size)
            except OSError:
                continue
            if copied_bytes + size > int(max_bytes):
                continue
            relative = source.relative_to(runner)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(relative))
            copied_bytes += size
    return {
        "path": str(destination),
        "files": copied,
        "file_count": len(copied),
        "bytes": int(copied_bytes),
        "max_bytes": int(max_bytes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("python-debug", "mt5-smoke"), default="python-debug")
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "configs/experiments/edge_discovery_v193_candidate.yaml",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    parser.add_argument("--seed-artifact", type=Path)
    parser.add_argument("--runner-source", type=Path, default=ROOT / "mt5_v2_runner")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/he_chuyen_gia/experiments/tensor_rule_v186")
    parser.add_argument("--from-date", default="2025.01.01")
    parser.add_argument("--to-date", default="2025.01.02")
    parser.add_argument(
        "--model",
        type=int,
        choices=(0, 1, 2, MT5_MODEL_REAL_TICKS),
        default=1,
        help="MT5 tester model; 4 = Every tick based on real ticks.",
    )
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-wall-sec", type=int, default=120)
    parser.add_argument(
        "--stop-request-file",
        type=Path,
        default=None,
        help=(
            "Workspace-side cooperative save-and-stop request. The bridge mirrors "
            "it into MT5 and waits for the authoritative report before persisting."
        ),
    )
    parser.add_argument(
        "--max-decision-packets", type=int, default=0,
        help="Ask MT5 to finalize normally after N completed bridge packets; 0 disables it.",
    )
    parser.add_argument(
        "--mt5-visual",
        action="store_true",
        help="Open MT5 Strategy Tester Visual mode for the mt5-smoke run.",
    )
    parser.add_argument(
        "--allow-unverified-imitation-seed",
        action="store_true",
        help=(
            "Run MT5 smoke with a loadable seed even when imitation provenance/gates "
            "fail; manifest remains research-only and non-promotable."
        ),
    )
    parser.add_argument("--keep-runner", action="store_true")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help=(
            "Chi chay MT5/reward/ledger research va khong serialize learner; "
            "khong tao candidate deployable, phu hop reward ablation tren o dia nho."
        ),
    )
    parser.add_argument(
        "--skip-paired-validation",
        action="store_true",
        help=(
            "Chi bo qua paired candidate/baseline pass trong phep do reward-only; "
            "khong thay doi promotion gate va khong dung cho deployment."
        ),
    )
    return parser


def main() -> int:
    """Run the candidate CLI with a persist-and-stop signal boundary."""

    restore_shutdown_handlers = _install_graceful_shutdown_handlers()
    try:
        return _main_impl()
    finally:
        restore_shutdown_handlers()


def _main_impl() -> int:
    _ensure_utf8_stdio()
    args = build_parser().parse_args()
    validate_date_window(args.from_date, args.to_date)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop_request_path = args.stop_request_file.resolve() if args.stop_request_file else None
    raw_profile = yaml.safe_load(args.profile.read_text(encoding="utf-8")) or {}
    profile = dict(raw_profile)
    isolation = dict(profile.get("isolation", {}) or {})
    quota = int(isolation.get("candidate_output_quota_bytes", 3 * 1024**3))
    reserve = int(isolation.get("reserve_floor_bytes", int(2.5 * 1024**3)))
    # Metrics-only runs do not persist a candidate checkpoint.  Account for
    # the runner clone that will be created later when applying the pre-clone
    # gate as well as in the live monitor.  Previously the first gate used the
    # configured reserve verbatim, so a safe metrics-only run could be rejected
    # even though the effective floor (and the post-clone monitor) already
    # allowed the temporary runner footprint.
    storage_gate_runner_bytes = 0
    if args.mode == "mt5-smoke" and bool(args.metrics_only):
        storage_gate_runner_bytes = directory_size(args.runner_source.resolve())
    storage_gate_reserve = effective_storage_reserve_floor(
        reserve,
        storage_gate_runner_bytes,
        metrics_only=bool(args.metrics_only),
    )
    free_before = shutil.disk_usage(output.anchor).free
    if free_before < storage_gate_reserve:
        write_storage_preflight_failure(
            output,
            mode=args.mode,
            free_bytes=free_before,
            reserve_floor_bytes=storage_gate_reserve,
            required_free_bytes=storage_gate_reserve,
            reason="free_below_reserve_floor",
        )
        raise RuntimeError(
            f"storage gate: free={free_before} < reserve={storage_gate_reserve}"
        )
    config_hash_before = sha256_file(args.config.resolve())
    base = load_config(args.config)
    config = candidate_config(base, profile, output)
    feedback_contract = build_candidate_feedback_contract(
        config,
        profile,
        source_config=str(args.config.resolve()),
        profile_path=str(args.profile.resolve()),
    )
    feedback_contract_path = output / "runtime_learning_contract.json"
    try:
        feedback_contract_path.write_text(
            json.dumps(feedback_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        validate_candidate_feedback_contract(feedback_contract)
    except Exception as exc:
        feedback_contract["status"] = "rejected"
        feedback_contract["error"] = f"{type(exc).__name__}: {exc}"
        feedback_contract_path.write_text(
            json.dumps(feedback_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "manifest.json").write_text(
            json.dumps({
                "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
                "mode": args.mode,
                "source_config": str(args.config.resolve()),
                "profile": str(args.profile.resolve()),
                "runtime_learning_contract": str(feedback_contract_path),
                "feedback_contract": feedback_contract,
                "run_exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    feedback_contract["status"] = "accepted"
    feedback_contract_path.write_text(
        json.dumps(feedback_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "candidate_feedback_contract",
        "path": str(feedback_contract_path),
        **feedback_contract,
    }, ensure_ascii=False), flush=True)
    seven_choice_contract: dict[str, object] = {"declared": False}
    if args.seed_artifact is not None:
        seven_choice_contract = inspect_seven_choice_runtime_contract(
            args.seed_artifact.resolve(), config
        )
        if args.mode == "mt5-smoke" and bool(seven_choice_contract.get("declared")) and not bool(
            seven_choice_contract.get("match", False)
        ):
            raise RuntimeError(
                "checkpoint bay lua chon va cau hinh MT5 khong cung hop dong: "
                + json.dumps(seven_choice_contract, ensure_ascii=False)
            )
    seed_hash_before = ""
    seed_imitation: dict[str, object] = {
        "declared": False,
        "effective_for_mt5": False,
        "reason": "no_seed_checkpoint",
    }
    seed_clone: Path | None = None
    seed_clone_method = ""
    if args.seed_artifact is not None:
        source_seed = args.seed_artifact.resolve()
        if not source_seed.is_file():
            raise FileNotFoundError(source_seed)
        source_seed_bytes = source_seed.stat().st_size
        telemetry_budget = int(isolation.get("event_jsonl_max_bytes", 32 * 1024**2)) + int(
            isolation.get("execution_trace_max_bytes", 64 * 1024**2)
        )
        projected_payload = projected_candidate_payload_bytes(
            source_seed_bytes,
            telemetry_budget,
            metrics_only=bool(args.metrics_only),
        )
        if projected_payload > quota:
            raise RuntimeError(
                "candidate output quota qua nho cho checkpoint: "
                f"projected_payload={projected_payload} > quota={quota}; "
                f"seed_bytes={source_seed_bytes}, telemetry_budget={telemetry_budget}"
            )
        required_free = storage_gate_reserve + projected_payload
        if free_before < required_free:
            write_storage_preflight_failure(
                output,
                mode=args.mode,
                free_bytes=free_before,
                reserve_floor_bytes=storage_gate_reserve,
                required_free_bytes=required_free,
                projected_payload_bytes=projected_payload,
                reason="free_below_reserve_plus_projected_payload",
            )
            raise RuntimeError(
                "storage gate before clone: "
                f"free={free_before} < required={required_free} "
                f"(reserve={storage_gate_reserve}, projected_payload={projected_payload}, "
                f"candidate_payload_quota={quota})"
            )
        seed_hash_before = sha256_file(source_seed)
        seed_imitation = inspect_imitation_seed(
            source_seed, checkpoint_sha256=seed_hash_before,
        )
        seed_imitation = apply_imitation_seed_gate_policy(
            seed_imitation,
            allow_unverified=bool(args.allow_unverified_imitation_seed),
        )
        print(json.dumps({
            "event": "seed_imitation_preflight",
            "checkpoint": str(source_seed),
            **seed_imitation,
        }, ensure_ascii=False), flush=True)
        if args.mode == "mt5-smoke" and not bool(
            seed_imitation.get("mt5_execution_allowed", False)
        ):
            raise RuntimeError(
                "checkpoint khong du dieu kien lam MT5 seed: "
                f"reason={seed_imitation.get('reason', '')}; "
                "can co sidecar dung hash, train hoan tat, reload pass va "
                "kiem dinh toan bo tai ranh gio 0 cung kiem dinh tu van hanh lien tuc dat"
            )
        if bool(seed_imitation.get("gate_bypassed", False)):
            print(json.dumps({
                "event": "seed_imitation_gate_bypassed",
                "checkpoint": str(source_seed),
                "failed_reason": str(seed_imitation.get("reason", "")),
                "scope": "mt5_smoke_research_only",
                "promotable": False,
            }, ensure_ascii=False), flush=True)
        seed_clone = output / f"source_seed_clone{source_seed.suffix}"
        seed_clone_method = clone_seed_artifact(source_seed, seed_clone)
        free_after_clone = shutil.disk_usage(output.anchor).free
        if free_after_clone < required_free:
            seed_clone.unlink(missing_ok=True)
            raise RuntimeError(
                "storage gate after seed clone: "
                f"free={free_after_clone} < required={required_free}; "
                f"clone_method={seed_clone_method}"
            )

    manifest: dict[str, object] = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "mode": args.mode,
        "mt5_visual": bool(args.mt5_visual),
        "source_config": str(args.config.resolve()),
        "source_config_sha256_before": config_hash_before,
        "source_seed": str(args.seed_artifact.resolve()) if args.seed_artifact else "",
        "source_seed_sha256_before": seed_hash_before,
        "source_seed_imitation": seed_imitation,
        "source_seed_clone_method": seed_clone_method,
        "free_bytes_before": int(free_before),
        "quota_bytes": quota,
        "quota_scope": "mutable_candidate_payload_excluding_protected_seed_clone",
        "reserve_floor_bytes": reserve,
        "storage_gate_reserve_floor_bytes": storage_gate_reserve,
        "storage_monitor_reserve_floor_bytes": reserve,
        "effective_mt5_learning_enabled": bool(config.mt5_learning_enabled),
        "runtime_learning_contract": str(feedback_contract_path),
        "feedback_contract": feedback_contract,
        "sampling_seed": int(args.seed),
        "max_decision_packets": int(args.max_decision_packets),
        "save_and_stop": {
            "requested": False,
            "path": str(stop_request_path) if stop_request_path else "",
            "boundary": "next_ea_decision_boundary",
        },
        "checkpoint_persistence": "disabled_metrics_only" if args.metrics_only else "enabled",
        "paired_validation": (
            "skipped_reward_only" if args.skip_paired_validation else "enabled_when_learning"
        ),
        "profile_app_overrides": dict(profile.get("app_overrides", {}) or {}),
        "profile_expectancy_overrides": dict(profile.get("expectancy_overrides", {}) or {}),
        "profile_learning_overrides": dict(profile.get("learning_overrides", {}) or {}),
        "seven_choice_runtime_contract": seven_choice_contract,
    }
    if bool(seed_imitation.get("research_continuation_allowed", False)) and not bool(
        seed_imitation.get("effective_for_mt5", False)
    ):
        manifest["run_scope"] = "research_only_unverified_continuation_lineage"
        manifest["promotable"] = False
        manifest["promotion_block_reason"] = "promotion_lineage_unverified"
        print(json.dumps({
            "event": "seed_research_continuation_allowed",
            "checkpoint": str(source_seed),
            "promotion_lineage_verified": False,
            "promotable": False,
        }, ensure_ascii=False), flush=True)
    elif bool(seed_imitation.get("gate_bypassed", False)):
        manifest["run_scope"] = "research_only_unverified_imitation_seed"
        manifest["promotable"] = False
        manifest["promotion_block_reason"] = "unverified_imitation_seed_gate_bypassed"
    experiment_id = str(dict(profile.get("experiment", {}) or {}).get(
        "id", "edge_discovery_v193_candidate"
    ) or "edge_discovery_v193_candidate")
    candidate_path = output / f"{experiment_id}.pkl"
    runner_parent, runner = allocate_candidate_runner(output)
    monitor: StorageMonitor | None = None
    diagnostics_monitor: Mt5DiagnosticsMonitor | None = None
    try:
        if args.mode == "python-debug":
            if seed_clone is not None:
                system = tai_he_chuyen_gia_mot_context_mot_policy(seed_clone, config, seed=args.seed)
            else:
                system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=args.seed)
            configure_candidate(system, config)
            manifest["validation"] = validate_candidate(
                system,
                expected_authority=str(config.learning.tensor_rule_authority),
            )
            atomic_pickle(system, candidate_path)
            manifest["candidate"] = str(candidate_path)
            manifest["candidate_sha256"] = sha256_file(candidate_path)
        else:
            if seed_clone is None:
                raise ValueError("--seed-artifact is required for mt5-smoke")
            manifest["runner_clone"] = clone_runner_candidate(
                args.runner_source.resolve(), runner,
                required_symbol=DEFAULT_SYMBOL,
                required_server=detect_runner_server(args.runner_source.resolve()),
                from_date=args.from_date,
                to_date=args.to_date,
            )
            monitor_reserve = effective_storage_reserve_floor(
                reserve,
                directory_size(runner),
                metrics_only=bool(args.metrics_only),
            )
            manifest["storage_monitor_reserve_floor_bytes"] = int(monitor_reserve)
            diagnostics_monitor = Mt5DiagnosticsMonitor(output, runner)
            diagnostics_monitor.start()
            manifest["mt5_diagnostics"] = {
                "schema_version": 1,
                "path": str(diagnostics_monitor.status_path),
                "runtime_dir": str(runner),
                "read_only": True,
            }
            monitor = StorageMonitor(
                output,
                runner,
                quota,
                monitor_reserve,
                excluded_paths=((seed_clone,) if seed_clone is not None else ()),
            )
            monitor.start()
            result = huan_luyen_he_chuyen_gia_mt5(
                config,
                output_path=candidate_path,
                seed_system_path=seed_clone,
                runner_dir=runner,
                timeout_sec=int(args.timeout_sec),
                max_wall_sec=int(args.max_wall_sec),
                max_decision_packets=int(args.max_decision_packets),
                visual=bool(args.mt5_visual),
                shutdown_terminal=True,
                from_date=args.from_date,
                to_date=args.to_date,
                model=int(args.model),
                seed=int(args.seed),
                resume_existing=False,
                paired_seed_validation=bool(
                    config.mt5_learning_enabled and not args.skip_paired_validation
                ),
                persist_checkpoint=not bool(args.metrics_only),
                stop_request_file=stop_request_path,
            )
            # Freeze the storage verdict before reading it or finalising the
            # manifest; otherwise a 5-second polling tick can race the checks.
            monitor.close()
            manifest["mt5"] = result
            if isinstance(result, dict) and result.get("research_candidate"):
                # Paired validation may intentionally select the source seed
                # for safe continuation. Keep the actually trained Codex
                # candidate discoverable as a separate non-promotable artifact
                # instead of making the UI look as if learning never occurred.
                manifest["research_candidate"] = dict(
                    result.get("research_candidate", {}) or {}
                )
            runtime_contract = dict(result.get("runtime_feedback_contract", {}) or {})
            codex_metrics = dict(
                dict(result.get("metrics", {}) or {}).get("codex_auto_connection", {}) or {}
            )
            feedback_contract["runtime_feedback_contract"] = runtime_contract
            feedback_contract["codex_auto_connection"] = codex_metrics
            feedback_contract["runtime_status"] = (
                "interrupted"
                if bool(result.get("interrupted", False))
                else "completed"
            )
            feedback_contract["updated_at"] = time.time()
            feedback_contract["effective"]["five_codex_teacher_effective"] = bool(
                runtime_contract.get("five_codex_teacher_effective", False)
            )
            for field in (
                "codex_cli_found", "codex_preflight_passed", "codex_receipts_valid",
                "codex_hypotheses_confirmed", "codex_optimizer_updates",
                "optimizer_state_stripped",
                "optimizer_state_reinitialized_for_research",
                "optimizer_state_resume_mode",
            ):
                if field in runtime_contract:
                    feedback_contract["effective"][field] = runtime_contract[field]
            if "codex_teacher_online_wait_sec" in runtime_contract:
                feedback_contract["effective"]["five_codex_teacher_online_wait_sec"] = (
                    runtime_contract["codex_teacher_online_wait_sec"]
                )
            _sync_runtime_recovery_snapshot_interval(
                feedback_contract, runtime_contract
            )
            for field in (
                "codex_teacher_model", "codex_teacher_executable",
                "codex_teacher_cli_version", "codex_teacher_preflight",
            ):
                if field in runtime_contract:
                    feedback_contract[field] = runtime_contract[field]
            codex_phase = dict(
                dict(result.get("metrics", {}) or {}).get("codex_teacher_phase", {}) or {}
            )
            # Keep the final contract compact and receipt-backed. Detailed
            # rationale stays in the append-only receipt/status files; the
            # contract only needs the lifecycle/update evidence and barrier
            # state that UI and automation use for a final verdict.
            feedback_contract["codex_teacher_phase"] = {
                key: codex_phase.get(key)
                for key in (
                    "enabled", "submitted", "graded", "hypotheses",
                    "hypotheses_confirmed", "updates", "interventions_applied",
                    "failures", "pending", "rejected_updates",
                    "hypothesis_rejections", "skips",
                    "account_impact_records", "account_impact_confirmed",
                    "account_impact_partial", "account_impact_reward_sum",
                    "counterfactual_validated", "counterfactual_unavailable",
                    "live_owner_outcome_records",
                    "live_owner_outcome_samples_by_action",
                    "live_owner_outcome_negative_count",
                    "live_owner_outcome_last",
                    "online_wait_sec", "online_barrier_waits",
                    "online_barrier_passed", "online_barrier_timeouts",
                    "online_barrier_waited_sec", "report_ledger_reconciled",
                    "codex_causal_links_observed", "codex_causal_updates_linked",
                    "codex_causal_hash_matches", "codex_causal_hash_unknown",
                    "rlhf_parameter_changed", "rlhf_internal_policy_state_changed",
                    "rlhf_probability_shift_count", "rlhf_max_probability_shift",
                    "generic_optimizer_updates", "codex_optimizer_updates",
                    "last_codex_causal_link",
                    "recovery_snapshot_updates", "recovery_snapshot_path",
                    "recovery_snapshot_error",
                    "recovery_snapshot_optimizer_state_stripped",
                    "recovery_snapshot_optimizer_state_stripped_count",
                    "recovery_snapshot_optimizer_resume_mode",
                )
                if key in codex_phase
            }
            feedback_contract["codex_preflight"] = dict(
                codex_metrics.get("preflight", {}) or {}
            )
            feedback_contract_path.write_text(
                json.dumps(feedback_contract, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest["save_and_stop"] = dict(result.get("save_and_stop", {}) or {})
            if bool(manifest["save_and_stop"].get("requested", False)):
                manifest["promotable"] = False
                manifest["promotion_block_reason"] = "save_and_stop_requested"
            collapse_guard = inspect_action_collapse(diagnostics_monitor.status_path)
            manifest["checkpoint_collapse_guard"] = collapse_guard
            # The tester can finish a profitable window after the action head
            # has already collapsed to HOLD. Such terminal weights are not a
            # safe continuation seed: carrying them forward creates a
            # no-trade feedback loop and hides whether the edge generalizes.
            # That is a continuation/promotion decision, not a persistence
            # decision: the learned candidate must still be saved so Codex
            # updates are not silently discarded and can be inspected or
            # explicitly rolled back by the caller.
            checkpoint_meta = {}
            if isinstance(result, dict):
                checkpoint_meta = dict(result.get("checkpoint", {}) or {})
            selected_champion = str(
                checkpoint_meta.get("continuation_policy", "") or ""
            ).strip().lower() == "champion"
            collapse_guard = {
                **collapse_guard,
                "selected_champion": selected_champion,
                "continuation_safe": selected_champion,
            }
            manifest["checkpoint_collapse_guard"] = collapse_guard
            finalize_checkpoint_persistence(
                manifest,
                metrics_only=bool(args.metrics_only),
                selected_champion=selected_champion,
                collapsed=bool(collapse_guard.get("collapsed", False)),
                candidate_exists=candidate_path.is_file(),
            )
            if monitor.failure:
                manifest["storage_gate_failure"] = monitor.failure
            if candidate_path.is_file():
                manifest["candidate"] = str(candidate_path)
                manifest["candidate_sha256"] = sha256_file(candidate_path)
                annotate_continuation_economic_verdict(manifest)
            else:
                incomplete = incomplete_checkpoint_manifest_from_result(result)
                if incomplete is not None:
                    manifest["incomplete_checkpoint"] = incomplete
                    expose_incomplete_checkpoint(
                        manifest,
                        incomplete=incomplete,
                        candidate_path=candidate_path,
                        metrics_only=bool(args.metrics_only),
                    )
        config_hash_after = sha256_file(args.config.resolve())
        seed_hash_after = sha256_file(args.seed_artifact.resolve()) if args.seed_artifact else ""
        manifest["source_config_sha256_after"] = config_hash_after
        manifest["source_config_unchanged"] = config_hash_before == config_hash_after
        manifest["source_seed_sha256_after"] = seed_hash_after
        manifest["source_seed_unchanged"] = seed_hash_before == seed_hash_after
        manifest["candidate_bytes"] = directory_size(output)
        manifest["candidate_payload_bytes"] = directory_size(
            output,
            exclude=((seed_clone,) if seed_clone is not None else ()),
        )
        manifest["protected_seed_clone_bytes"] = (
            seed_clone.stat().st_size if seed_clone is not None and seed_clone.is_file() else 0
        )
        manifest["free_bytes_after"] = shutil.disk_usage(output.anchor).free
        if not manifest["source_config_unchanged"] or not manifest["source_seed_unchanged"]:
            raise RuntimeError("protected source hash changed")
        if monitor is not None and monitor.failure:
            failure = monitor.failure
            completed_metrics_only = bool(args.metrics_only) and (
                metrics_only_result_complete_for_storage_pressure(
                    manifest,
                    candidate_payload_bytes=int(manifest["candidate_payload_bytes"]),
                    candidate_payload_quota=quota,
                )
            )
            if completed_metrics_only:
                manifest["storage_gate_pressure_after_completion"] = dict(failure)
                manifest["storage_gate_completion_policy"] = (
                    "completed_metrics_only_warning"
                )
            else:
                raise RuntimeError(
                    "candidate storage gate triggered: "
                    f"payload={failure.get('candidate_payload_bytes', 0)} "
                    f"> quota={failure.get('candidate_payload_quota_bytes', quota)} "
                    f"or free={failure.get('free_bytes', 0)} "
                    f"< reserve={failure.get('reserve_floor_bytes', reserve)}"
                )
        if int(manifest["candidate_payload_bytes"]) > quota:
            raise RuntimeError(
                "candidate payload quota exceeded: "
                f"payload={manifest['candidate_payload_bytes']} > quota={quota}; "
                "protected seed clone is excluded"
            )
    finally:
        active_exception = sys.exc_info()[1]
        if active_exception is not None:
            manifest["run_exception"] = {
                "type": type(active_exception).__name__, "message": str(active_exception)
            }
            incomplete = incomplete_checkpoint_manifest(active_exception)
            if incomplete is not None:
                manifest["incomplete_checkpoint"] = incomplete
                expose_incomplete_checkpoint(
                    manifest,
                    incomplete=incomplete,
                    candidate_path=candidate_path,
                    metrics_only=bool(args.metrics_only),
                )
            # Keep the contract truthful even when the training function
            # raises after MT5 has already persisted an incomplete checkpoint.
            # Without this final write, a live contract remains stuck at
            # ``runtime_status=running`` and the UI cannot distinguish an
            # interrupted run from one still learning.
            try:
                contract = json.loads(
                    feedback_contract_path.read_text(encoding="utf-8")
                )
                if isinstance(contract, dict):
                    contract["runtime_status"] = "interrupted"
                    contract["updated_at"] = time.time()
                    contract["run_exception"] = {
                        "type": type(active_exception).__name__,
                        "message": str(active_exception)[:2000],
                    }
                    recovered_codex = recover_interrupted_codex_feedback(
                        contract, output
                    )
                    if recovered_codex:
                        manifest["codex_teacher_feedback_recovered"] = recovered_codex
                        manifest["codex_teacher_phase"] = dict(
                            contract.get("codex_teacher_phase", {}) or {}
                        )
                    feedback_contract_path.write_text(
                        json.dumps(contract, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    # ``manifest`` was initialized with a copy of the accepted
                    # contract before MT5 started.  Keep its embedded contract
                    # synchronized with the recovered sidecar, otherwise the
                    # UI reads stale ``updates=0`` even though receipt status
                    # proves Codex updates occurred before the interruption.
                    manifest["feedback_contract"] = contract
            except Exception as contract_error:
                manifest["runtime_learning_contract_error"] = {
                    "type": type(contract_error).__name__,
                    "message": str(contract_error),
                }
        if monitor is not None:
            monitor.close()
        if diagnostics_monitor is not None:
            try:
                diagnostics_monitor.close()
            except Exception as diagnostics_error:
                manifest["mt5_diagnostics_error"] = {
                    "type": type(diagnostics_error).__name__,
                    "message": str(diagnostics_error),
                    "training_affected": False,
                }
        if active_exception is not None and runner.exists():
            manifest["runner_diagnostics"] = preserve_runner_diagnostics(runner, output)
        if runner.exists() and not args.keep_runner:
            manifest["runner_cleanup"] = safe_remove_runner(runner, runner_parent)
        elif runner.exists():
            manifest["runner_cleanup"] = {"removed": False, "kept_by_request": True, "path": str(runner)}
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    print(json.dumps({"manifest": str(output / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
