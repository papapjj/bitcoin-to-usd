"""Transactional champion-shadow training for the context-expert policy.

The deployed/champion artifact is immutable.  Independently seeded shadows are
trained from it, and every shadow is frozen-evaluated on the same future OOS
windows against that champion.  Promotion is fail-closed and writes a distinct
artifact only after the complete matrix passes.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .chuyen_gia_policy import (
    _mt5_best_fitness,
    _mt5_best_metrics,
    evaluate_policy_mt5_oos,
    huan_luyen_he_chuyen_gia_mt5,
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from .learning_integrity import (
    champion_candidate_matrix_gate,
    frozen_oos_promotion_gate,
    paired_oos_non_regression_gate,
    sha256_file,
)
from .mt5_runner import MT5_MODEL_REAL_TICKS


TrainCallback = Callable[..., dict[str, object]]
EvaluateCallback = Callable[..., dict[str, object]]


def _rlhf_matrix_storage_preflight(
    config: Any,
    *,
    storage_path: str | Path,
) -> dict[str, object]:
    """Check configured free-space floor before an RLHF matrix can write MT5 artifacts."""
    learning = getattr(config, "learning", None)
    enabled = bool(getattr(learning, "five_rlhf_policy_update_enabled", False))
    required = max(0, int(
        getattr(learning, "five_rlhf_min_free_disk_bytes", 0) or 0
    ))
    anchor = Path(storage_path).resolve()
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if not enabled or required <= 0:
        return {
            "enabled": bool(enabled),
            "passed": True,
            "required_free_bytes": int(required),
            "free_bytes": None,
            "storage_path": str(anchor),
            "reason": "disabled_or_no_floor",
        }
    usage = shutil.disk_usage(anchor)
    free = int(usage.free)
    passed = bool(free >= required)
    return {
        "enabled": True,
        "passed": passed,
        "required_free_bytes": int(required),
        "free_bytes": free,
        "storage_path": str(anchor),
        "reason": "ok" if passed else "insufficient_free_disk",
    }


def build_consecutive_validation_windows(
    train_to: str,
    *,
    count: int = 3,
    window_days: int = 1,
) -> list[dict[str, str]]:
    """Build non-overlapping frozen windows strictly after the train segment."""
    try:
        cursor = datetime.strptime(str(train_to), "%Y.%m.%d")
    except ValueError as exc:
        raise ValueError("--train-to phai co dang YYYY.MM.DD") from exc
    count = int(count)
    window_days = int(window_days)
    if count < 1:
        raise ValueError("--validation-count phai >= 1")
    if window_days < 1:
        raise ValueError("--validation-days phai >= 1")
    windows: list[dict[str, str]] = []
    for index in range(count):
        start = cursor + timedelta(days=index * window_days)
        end = start + timedelta(days=window_days)
        windows.append({
            "id": f"{start:%Y.%m.%d}:{end:%Y.%m.%d}",
            "from": start.strftime("%Y.%m.%d"),
            "to": end.strftime("%Y.%m.%d"),
        })
    return windows


def _edge_metrics(metrics: dict[str, object]) -> dict[str, float | int | None]:
    summary = dict(metrics.get("execution_summary", {}) or {})

    def _number(*values: object) -> float | None:
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return None

    return {
        "net_profit": _number(metrics.get("net_profit"), metrics.get("profit")),
        "profit_factor": _number(metrics.get("profit_factor"), summary.get("profit_factor")),
        "expectancy": _number(summary.get("expectancy"), metrics.get("expectancy")),
        "policy_expectancy_lcb": _number(
            metrics.get("fitness_policy_expectancy_lcb"),
            metrics.get("policy_expectancy_lcb"),
            summary.get("expectancy_lcb"),
        ),
        "drawdown_pct": _number(metrics.get("equity_drawdown_pct"), metrics.get("balance_drawdown_pct")),
        "trades": int(_number(metrics.get("total_trades"), summary.get("lifecycle_count")) or 0),
    }


def _snapshot_oos_artifacts(
    metrics: dict[str, object],
    *,
    audit_dir: Path,
    candidate_seed: int,
    window_index: int,
    arm: str,
) -> dict[str, str]:
    target = audit_dir / "oos_artifacts" / f"seed_{candidate_seed}" / f"window_{window_index:02d}" / arm
    target.mkdir(parents=True, exist_ok=True)
    sources = {
        "report": metrics.get("report_path"),
        "bridge": metrics.get("bridge_path"),
        "execution": metrics.get("execution_ledger_path"),
        "feature_audit": metrics.get("feature_audit_path"),
    }
    copied: dict[str, str] = {}
    for label, raw in sources.items():
        if not str(raw or "").strip():
            continue
        source = Path(str(raw))
        if not source.is_file():
            continue
        destination = target / source.name
        shutil.copy2(source, destination)
        copied[label] = str(destination.resolve())
    (target / "artifact_manifest.json").write_text(
        json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return copied


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def run_context_expert_champion_training_mt5(
    config: Any,
    *,
    champion_path: str | Path,
    output_path: str | Path,
    audit_dir: str | Path,
    train_from: str,
    train_to: str,
    validation_count: int | None = None,
    validation_days: int = 1,
    candidate_seed_count: int | None = None,
    base_seed: int = 0,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    max_wall_sec: int = 0,
    visual: bool = False,
    shutdown_terminal: bool = False,
    model: int = MT5_MODEL_REAL_TICKS,
    paired_min_expectancy: float = 0.0,
    paired_max_profit_regression: float = 0.0,
    paired_max_expectancy_lcb_regression: float = 0.02,
    paired_max_drawdown_increase_pct: float = 1.0,
    trainer: TrainCallback | None = None,
    evaluator: EvaluateCallback | None = None,
) -> dict[str, object]:
    """Train shadows and promote one only after a complete frozen OOS matrix."""
    champion = Path(champion_path).resolve()
    output = Path(output_path).resolve()
    audit = Path(audit_dir).resolve()
    if not champion.is_file():
        raise FileNotFoundError(f"khong tim thay champion: {champion}")
    if champion == output:
        raise ValueError("--output phai khac --champion; champion la artifact bat bien")
    if output.exists():
        raise FileExistsError(f"output da ton tai: {output}")
    if audit.exists() and any(audit.iterdir()):
        raise FileExistsError(f"audit-dir phai rong de khong tron bang chung: {audit}")
    storage_preflight = _rlhf_matrix_storage_preflight(
        config,
        storage_path=audit.parent,
    )
    if not bool(storage_preflight.get("passed", False)):
        raise RuntimeError(
            "RLHF matrix storage preflight failed: "
            f"free={storage_preflight.get('free_bytes')} "
            f"required={storage_preflight.get('required_free_bytes')} "
            f"path={storage_preflight.get('storage_path')}"
        )
    audit.mkdir(parents=True, exist_ok=True)
    try:
        train_start = datetime.strptime(str(train_from), "%Y.%m.%d")
        train_end = datetime.strptime(str(train_to), "%Y.%m.%d")
    except ValueError as exc:
        raise ValueError("--train-from va --train-to phai co dang YYYY.MM.DD") from exc
    if train_start >= train_end:
        raise ValueError("--train-from phai truoc --train-to")

    learning = getattr(config, "learning", None)
    production_required_windows = max(1, int(
        getattr(learning, "promotion_min_oos_windows", 3)
    ))
    production_required_seeds = max(1, int(
        getattr(learning, "promotion_min_seeds", 3)
    ))
    evaluation_window_count = max(1, int(
        validation_count if validation_count is not None
        else production_required_windows
    ))
    candidate_seed_runs = max(1, int(
        candidate_seed_count if candidate_seed_count is not None
        else production_required_seeds
    ))
    windows = build_consecutive_validation_windows(
        str(train_to), count=evaluation_window_count, window_days=int(validation_days)
    )
    champion_hash = sha256_file(champion)
    # The RLHF reference and reward receipts must be rooted in the exact
    # immutable champion used to seed this matrix.  Without this check a
    # copied YAML hash could silently describe a different source checkpoint,
    # making the PPO/reference attribution unverifiable.
    rlhf_learning = getattr(config, "learning", None)
    if bool(getattr(rlhf_learning, "five_rlhf_policy_update_enabled", False)):
        configured_source_hash = str(
            getattr(rlhf_learning, "five_rlhf_source_checkpoint_sha256", "") or ""
        ).strip().lower()
        if configured_source_hash != str(champion_hash).strip().lower():
            raise ValueError(
                "RLHF source checkpoint hash must match the immutable champion: "
                f"configured={configured_source_hash or '<empty>'} champion={champion_hash}"
            )
    train_fn = trainer or huan_luyen_he_chuyen_gia_mt5
    evaluate_fn = evaluator or evaluate_policy_mt5_oos

    candidate_gate_kwargs = {
        "require_fitness_eligible": bool(getattr(learning, "promotion_require_fitness_eligible", True)),
        "require_reconciled": bool(getattr(learning, "promotion_require_reconciled", True)),
        "min_trades": int(getattr(learning, "promotion_min_oos_trades", 30) or 0),
        "max_drawdown_pct": float(getattr(learning, "promotion_max_oos_drawdown_pct", 10.0)),
        "min_policy_expectancy_lcb": float(getattr(learning, "promotion_min_policy_expectancy_lcb", 0.0)),
        "max_negative_context_scopes": int(getattr(learning, "promotion_max_negative_context_scopes", 0)),
        "require_feature_parity": bool(getattr(learning, "promotion_require_feature_parity", True)),
        "require_real_tick_evidence": bool(getattr(learning, "promotion_require_real_tick_evidence", True)),
    }
    baseline_gate_kwargs = dict(candidate_gate_kwargs)
    baseline_gate_kwargs.update({
        "require_fitness_eligible": False,
        "min_policy_expectancy_lcb": None,
        "max_negative_context_scopes": None,
    })

    def evaluate_artifact(path: Path, window: dict[str, str]) -> dict[str, object]:
        policy = tai_he_chuyen_gia_mot_context_mot_policy(path, config, seed=int(base_seed))
        result = evaluate_fn(
            policy,
            config,
            runner_dir=runner_dir,
            timeout_sec=int(timeout_sec),
            visual=bool(visual),
            shutdown_terminal=bool(shutdown_terminal),
            from_date=window["from"],
            to_date=window["to"],
            model=int(model),
        )
        metrics = dict(_mt5_best_metrics(result) or {})
        return {
            "fitness": _mt5_best_fitness(result),
            "metrics": metrics,
            "edge": _edge_metrics(metrics),
        }

    # Champion is immutable and deterministic for a frozen window. Evaluate it
    # lazily after the first usable shadow exists, then reuse the exact metrics
    # for every independently trained candidate.
    baseline_by_window: dict[str, dict[str, object]] = {}

    rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    shadow_by_identity: dict[tuple[str, str], Path] = {}
    for seed_index in range(candidate_seed_runs):
        candidate_seed = int(base_seed) + seed_index
        shadow = audit / f"shadow_seed_{candidate_seed}.pkl"
        print(json.dumps({
            "phase": "train_shadow", "candidate_seed": candidate_seed,
            "from": str(train_from), "to": str(train_to),
        }, ensure_ascii=False), flush=True)
        train_result = train_fn(
            config,
            output_path=shadow,
            seed_system_path=champion,
            runner_dir=runner_dir,
            timeout_sec=int(timeout_sec),
            max_wall_sec=max(0, int(max_wall_sec)),
            visual=bool(visual),
            shutdown_terminal=bool(shutdown_terminal),
            from_date=str(train_from),
            to_date=str(train_to),
            model=int(model),
            seed=candidate_seed,
            resume_existing=False,
        )
        interrupted = bool(train_result.get("interrupted", False))
        shadow_hash = sha256_file(shadow) or ""
        candidate_record = {
            "seed": candidate_seed,
            "shadow": str(shadow),
            "candidate_sha256": shadow_hash,
            "interrupted": interrupted,
            "train_result": train_result,
        }
        candidates.append(candidate_record)
        if interrupted or not shadow_hash:
            reason = "interrupted_or_missing_shadow"
            for window in windows:
                rows.append({
                    "candidate_seed": candidate_seed,
                    "candidate_sha256": shadow_hash,
                    "validation_window": window["id"],
                    "eligible": False,
                    "policy_expectancy_lcb": None,
                    "reasons": [reason],
                })
            continue
        shadow_by_identity[(str(candidate_seed), shadow_hash)] = shadow
        for window_index, window in enumerate(windows, start=1):
            if window["id"] not in baseline_by_window:
                print(json.dumps({
                    "phase": "frozen_oos_champion", "window": window["id"],
                    "champion_sha256": champion_hash,
                }, ensure_ascii=False), flush=True)
                baseline = evaluate_artifact(champion, window)
                baseline["artifacts"] = _snapshot_oos_artifacts(
                    dict(baseline["metrics"]), audit_dir=audit,
                    candidate_seed=-1, window_index=window_index, arm="champion",
                )
                baseline_by_window[window["id"]] = baseline
            print(json.dumps({
                "phase": "frozen_oos_candidate", "candidate_seed": candidate_seed,
                "candidate_sha256": shadow_hash, "window": window["id"],
            }, ensure_ascii=False), flush=True)
            candidate = evaluate_artifact(shadow, window)
            candidate["artifacts"] = _snapshot_oos_artifacts(
                dict(candidate["metrics"]), audit_dir=audit,
                candidate_seed=candidate_seed, window_index=window_index, arm="candidate",
            )
            baseline = baseline_by_window[window["id"]]
            strict_candidate = frozen_oos_promotion_gate(
                dict(candidate["metrics"]), **candidate_gate_kwargs
            )
            strict_baseline = frozen_oos_promotion_gate(
                dict(baseline["metrics"]), **baseline_gate_kwargs
            )
            paired = paired_oos_non_regression_gate(
                dict(candidate["metrics"]),
                dict(baseline["metrics"]),
                min_candidate_expectancy=float(paired_min_expectancy),
                max_profit_regression=float(paired_max_profit_regression),
                max_expectancy_regression=float(paired_max_expectancy_lcb_regression),
                max_drawdown_increase_pct=float(paired_max_drawdown_increase_pct),
            )
            reasons = [
                *(f"candidate:{reason}" for reason in strict_candidate.get("reasons", []) or []),
                *(f"champion:{reason}" for reason in strict_baseline.get("reasons", []) or []),
                *(f"paired:{reason}" for reason in paired.get("reasons", []) or []),
            ]
            edge = dict(candidate["edge"])
            row = {
                "candidate_seed": candidate_seed,
                "candidate_sha256": shadow_hash,
                "validation_window": window["id"],
                "candidate": candidate,
                "champion": baseline,
                "strict_candidate_gate": strict_candidate,
                "strict_champion_gate": strict_baseline,
                "paired_gate": paired,
                "eligible": not reasons,
                "policy_expectancy_lcb": edge.get("policy_expectancy_lcb"),
                "reasons": reasons,
            }
            rows.append(row)
            (audit / f"seed_{candidate_seed}_window_{window_index:02d}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        if sha256_file(shadow) != shadow_hash:
            raise RuntimeError(f"frozen OOS da lam thay doi shadow artifact seed={candidate_seed}")

    matrix_gate = champion_candidate_matrix_gate(
        rows,
        min_windows=production_required_windows,
        min_candidate_seeds=production_required_seeds,
    )
    selected_seed = matrix_gate.get("selected_seed")
    selected_hash = str(matrix_gate.get("selected_artifact_sha256") or "")
    selected_shadow = shadow_by_identity.get((str(selected_seed), selected_hash))
    promoted = bool(matrix_gate.get("eligible", False) and selected_shadow is not None)
    registry_output = ""
    if promoted and selected_shadow is not None:
        _atomic_copy(selected_shadow, output)
        source_registry = selected_shadow.with_suffix(".registry.json")
        if source_registry.is_file():
            destination_registry = output.with_suffix(".registry.json")
            _atomic_copy(source_registry, destination_registry)
            registry_output = str(destination_registry)
        if sha256_file(output) != selected_hash:
            raise RuntimeError("hash promoted artifact khong khop selected shadow")
    if sha256_file(champion) != champion_hash:
        raise RuntimeError("protected champion SHA256 changed during training/evaluation")

    summary: dict[str, object] = {
        "mode": "context_expert_champion_shadow_matrix_v1",
        "champion": str(champion),
        "champion_sha256": champion_hash,
        "champion_immutable_verified": True,
        "output": str(output) if promoted else "",
        "output_registry": registry_output,
        "promoted": promoted,
        "train": {"from": str(train_from), "to": str(train_to)},
        "validation_windows": windows,
        "production_required_windows": production_required_windows,
        "production_required_candidate_seeds": production_required_seeds,
        "evaluated_windows": evaluation_window_count,
        "candidate_seeds": candidate_seed_runs,
        "storage_preflight": storage_preflight,
        "candidates": candidates,
        "matrix_gate": matrix_gate,
        "evaluations": rows,
    }
    summary_path = audit / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    print(json.dumps({
        "phase": "champion_matrix_complete", "promoted": promoted,
        "summary": str(summary_path), "output": str(output) if promoted else "",
    }, ensure_ascii=False), flush=True)
    return summary
