"""Small, deterministic helpers for frozen evaluation and reproducibility."""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


def clone_policy(policy: Any):
    """Return an isolated policy without invoking a potentially mutating copy().

    Several legacy policy ``copy`` implementations initialize graph/runtime
    fields as a side effect. Frozen OOS must never call a method that can touch
    the champion. ``deepcopy`` is the first choice; bespoke copy is fallback
    only for objects that cannot be deep-copied.
    """
    try:
        return copy.deepcopy(policy)
    except Exception:
        copier = getattr(policy, "copy", None)
        if callable(copier):
            return copier()
        raise


def policy_state_hash(policy: Any) -> str:
    """Hash model state without trusting a display repr as a proof of freezing."""
    try:
        payload = pickle.dumps(policy, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        try:
            raw = asdict(policy) if is_dataclass(policy) else vars(policy)
        except Exception:
            raw = {"type": type(policy).__qualname__, "repr": repr(policy)}
        payload = json.dumps(raw, sort_keys=True, default=repr, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path, *, suffixes: tuple[str, ...] = (".py", ".mq5", ".mqh", ".yaml", ".yml")) -> str | None:
    """Content hash of source/config files, stable across timestamps and OS paths."""
    root = Path(path)
    if not root.exists():
        return None
    digest = sha256()
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in suffixes
    )
    for item in files:
        rel = item.name if root.is_file() else str(item.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stable_object_hash(value: Any) -> str:
    """Hash effective dataclass/dict configuration deterministically."""
    if is_dataclass(value):
        value = asdict(value)
    raw = json.dumps(value, sort_keys=True, default=repr, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(raw).hexdigest()


def run_manifest(*, seed: int, config_path: str | Path | None = None, data_path: str | Path | None = None, protocol_version: int = 7, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "protocol_version": int(protocol_version),
        "seed": int(seed),
        "config_path": str(config_path) if config_path else "",
        "config_sha256": sha256_file(config_path) if config_path else None,
        "data_path": str(data_path) if data_path else "",
        "data_sha256": sha256_file(data_path) if data_path else None,
        "extra": dict(extra or {}),
    }


def promotion_stability_report(
    evaluations: list[dict[str, Any]],
    *,
    min_windows: int = 3,
    min_seeds: int = 3,
    score_key: str = "val_fitness",
) -> dict[str, Any]:
    """Summarise whether independent OOS evidence is sufficient for promotion.

    A single successful OOS segment, or the last chronologically trained model,
    is only a *candidate*.  Promotion needs a matrix of frozen evaluations that
    spans independent validation windows and independently seeded candidates.
    The caller decides how to run that matrix; this helper makes the evidence
    criterion explicit, serialisable and deterministic.
    """
    min_windows = max(1, int(min_windows))
    min_seeds = max(1, int(min_seeds))
    valid: list[dict[str, Any]] = []
    for row in evaluations:
        try:
            score = float(row.get(score_key, float("nan")))
        except (TypeError, ValueError):
            continue
        if score != score or score in (float("inf"), -float("inf")):
            continue
        valid.append(dict(row))

    def _identifier(row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value) != "":
                return str(value)
        return ""

    windows = sorted({_identifier(row, "validation_window", "window", "val_from", "validation_from") for row in valid} - {""})
    seeds = sorted({_identifier(row, "seed", "training_seed", "evaluation_seed") for row in valid} - {""})
    pairs = {
        (
            _identifier(row, "validation_window", "window", "val_from", "validation_from"),
            _identifier(row, "seed", "training_seed", "evaluation_seed"),
        )
        for row in valid
    }
    pairs.discard(("", ""))
    required_pairs = min_windows * min_seeds
    scores = [float(row[score_key]) for row in valid]
    reasons: list[str] = []
    if len(windows) < min_windows:
        reasons.append(f"oos_windows_insufficient:{len(windows)}<{min_windows}")
    if len(seeds) < min_seeds:
        reasons.append(f"seed_runs_insufficient:{len(seeds)}<{min_seeds}")
    if len(pairs) < required_pairs:
        reasons.append(f"oos_seed_matrix_incomplete:{len(pairs)}<{required_pairs}")
    if not valid:
        reasons.append("no_finite_oos_evaluations")

    values = sorted(scores)
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "required_windows": min_windows,
        "required_seeds": min_seeds,
        "required_pairs": required_pairs,
        "valid_evaluations": len(valid),
        "distinct_windows": len(windows),
        "distinct_seeds": len(seeds),
        "distinct_pairs": len(pairs),
        "mean_score": (sum(scores) / len(scores)) if scores else -float("inf"),
        "median_score": (values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2.0) if values else -float("inf"),
        "worst_score": min(scores) if scores else -float("inf"),
        "best_score": max(scores) if scores else -float("inf"),
    }


def champion_candidate_matrix_gate(
    evaluations: list[dict[str, Any]],
    *,
    min_windows: int = 3,
    min_candidate_seeds: int = 3,
    artifact_hash_key: str = "candidate_sha256",
    seed_key: str = "candidate_seed",
    window_key: str = "validation_window",
    eligible_key: str = "eligible",
    score_key: str = "policy_expectancy_lcb",
) -> dict[str, Any]:
    """Fail closed unless every candidate is tested on the same OOS matrix.

    ``promotion_stability_report`` describes coverage of an experiment.  It
    cannot by itself prove that the artifact selected for deployment was the
    artifact evaluated in every cell: older walk-forward callers could mix one
    different candidate per window and then copy only the best row.  This gate
    binds every row to an immutable artifact hash, requires each independently
    trained candidate seed to pass every validation window, and selects only
    among complete candidates by maximin (worst-window) score.
    """
    required_windows = max(1, int(min_windows))
    required_seeds = max(1, int(min_candidate_seeds))
    rows = [dict(row) for row in evaluations]

    def _text(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        return str(value).strip() if value is not None else ""

    def _score(row: dict[str, Any]) -> float:
        try:
            value = float(row.get(score_key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    windows = sorted({_text(row, window_key) for row in rows} - {""})
    seeds = sorted({_text(row, seed_key) for row in rows} - {""})
    reasons: list[str] = []
    if len(windows) < required_windows:
        reasons.append(f"oos_windows_insufficient:{len(windows)}<{required_windows}")
    if len(seeds) < required_seeds:
        reasons.append(f"candidate_seeds_insufficient:{len(seeds)}<{required_seeds}")

    candidates: list[dict[str, Any]] = []
    stable_candidates: list[dict[str, Any]] = []
    hashes_by_seed: dict[str, set[str]] = {}
    for row in rows:
        seed = _text(row, seed_key)
        artifact_hash = _text(row, artifact_hash_key)
        if seed and artifact_hash:
            hashes_by_seed.setdefault(seed, set()).add(artifact_hash)
    for seed, hashes in sorted(hashes_by_seed.items()):
        if len(hashes) != 1:
            candidates.append({
                "seed": seed,
                "artifact_sha256": "",
                "eligible": False,
                "reasons": [f"candidate_hash_not_unique:{len(hashes)}"],
                "windows": 0,
                "worst_score": None,
                "mean_score": None,
            })
            continue
        artifact_hash = next(iter(hashes))
        candidate_rows = [
            row for row in rows
            if _text(row, seed_key) == seed and _text(row, artifact_hash_key) == artifact_hash
        ]
        rows_by_window: dict[str, list[dict[str, Any]]] = {}
        for row in candidate_rows:
            window = _text(row, window_key)
            if window:
                rows_by_window.setdefault(window, []).append(row)
        candidate_reasons: list[str] = []
        missing = [window for window in windows if window not in rows_by_window]
        if missing:
            candidate_reasons.append(f"candidate_windows_missing:{len(missing)}")
        duplicates = [window for window, items in rows_by_window.items() if len(items) != 1]
        if duplicates:
            candidate_reasons.append(f"candidate_window_rows_not_unique:{len(duplicates)}")
        failed_windows = [
            window for window, items in rows_by_window.items()
            if len(items) == 1 and not bool(items[0].get(eligible_key, False))
        ]
        if failed_windows:
            candidate_reasons.append(f"candidate_window_gate_failed:{len(failed_windows)}")
        scores = [_score(items[0]) for items in rows_by_window.values() if len(items) == 1]
        finite_scores = [score for score in scores if np.isfinite(score)]
        if len(finite_scores) != len(windows):
            candidate_reasons.append(f"candidate_scores_incomplete:{len(finite_scores)}<{len(windows)}")
        candidate = {
            "seed": seed,
            "artifact_sha256": artifact_hash,
            "eligible": not candidate_reasons and len(rows_by_window) >= required_windows,
            "reasons": candidate_reasons,
            "windows": len(rows_by_window),
            "worst_score": min(finite_scores) if finite_scores else None,
            "mean_score": (sum(finite_scores) / len(finite_scores)) if finite_scores else None,
        }
        candidates.append(candidate)
        if candidate["eligible"]:
            stable_candidates.append(candidate)

    if len(stable_candidates) < required_seeds:
        reasons.append(f"stable_candidate_seeds_insufficient:{len(stable_candidates)}<{required_seeds}")
    selected = max(
        stable_candidates,
        key=lambda item: (float(item["worst_score"]), float(item["mean_score"]), item["artifact_sha256"]),
        default=None,
    )
    return {
        "eligible": not reasons and selected is not None,
        "reasons": reasons,
        "required_windows": required_windows,
        "required_candidate_seeds": required_seeds,
        "distinct_windows": len(windows),
        "distinct_candidate_seeds": len(seeds),
        "stable_candidate_seeds": len(stable_candidates),
        "selected_seed": selected["seed"] if selected is not None else None,
        "selected_artifact_sha256": selected["artifact_sha256"] if selected is not None else None,
        "selection_rule": "maximin_policy_expectancy_lcb_then_mean",
        "candidates": candidates,
    }


def frozen_oos_promotion_gate(
    metrics: dict[str, Any] | None,
    *,
    require_fitness_eligible: bool = True,
    require_reconciled: bool = True,
    min_trades: int = 30,
    max_drawdown_pct: float | None = 10.0,
    min_policy_expectancy_lcb: float | None = 0.0,
    max_negative_context_scopes: int | None = 0,
    require_feature_parity: bool = True,
    require_real_tick_evidence: bool = True,
) -> dict[str, Any]:
    """Fail-closed gate for a single frozen OOS evaluation.

    A finite MT5 fitness value is not sufficient evidence: the evaluator uses
    ``-100`` for several ineligible outcomes, which is still a finite float.
    This gate keeps learning unrestricted while making promotion require an
    eligible, reconciled, sufficiently sampled and data-faithful OOS result.
    Missing production evidence fails closed; callers may disable individual
    checks explicitly for lightweight unit fixtures.
    """
    raw = dict(metrics or {})
    reasons: list[str] = []

    def _value(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in raw and raw.get(key) is not None:
                return raw.get(key)
        return default

    def _number(*keys: str, default: float = float("nan")) -> float:
        value = _value(*keys, default=default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _bool(*keys: str, default: bool | None = None) -> bool | None:
        value = _value(*keys, default=default)
        if value is None:
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y"}:
                return True
            if lowered in {"0", "false", "no", "n"}:
                return False
        return bool(value)

    if require_fitness_eligible:
        eligible = _bool("fitness_eligible", "eligible")
        if eligible is None:
            reasons.append("fitness_eligibility_missing")
        elif not eligible:
            reasons.append("fitness_ineligible")

    if require_reconciled:
        reconciled = _bool("fitness_reconciled", "reconciled")
        if reconciled is None:
            reconciliation = raw.get("reconciliation")
            if isinstance(reconciliation, dict):
                reconciled = _bool_from_mapping(reconciliation, "reconciled")
        if reconciled is None:
            reasons.append("reconciliation_missing")
        elif not reconciled:
            reasons.append("reconciliation_failed")

    lifecycle_count = _number(
        "total_trades",
        "fitness_sample_count",
        "fitness_lifecycle_count",
        default=float("nan"),
    )
    if not np.isfinite(lifecycle_count):
        summary = raw.get("execution_summary")
        if isinstance(summary, dict):
            lifecycle_count = _number_from_mapping(summary, "lifecycle_count", default=float("nan"))
    if int(min_trades) > 0 and (not np.isfinite(lifecycle_count) or lifecycle_count < int(min_trades)):
        reasons.append(f"oos_trades_insufficient:{int(lifecycle_count) if np.isfinite(lifecycle_count) else 'missing'}<{int(min_trades)}")

    drawdowns = [
        _number("balance_drawdown_pct", default=float("nan")),
        _number("equity_drawdown_pct", default=float("nan")),
    ]
    drawdowns = [value for value in drawdowns if np.isfinite(value)]
    max_observed_drawdown = max(drawdowns) if drawdowns else float("nan")
    if max_drawdown_pct is not None and (
        not np.isfinite(max_observed_drawdown) or max_observed_drawdown > float(max_drawdown_pct)
    ):
        shown = f"{max_observed_drawdown:.4g}" if np.isfinite(max_observed_drawdown) else "missing"
        reasons.append(f"oos_drawdown_above_limit:{shown}>{float(max_drawdown_pct):g}")

    policy_lcb = _number(
        "fitness_policy_expectancy_lcb",
        "policy_expectancy_lcb",
        default=float("nan"),
    )
    if not np.isfinite(policy_lcb):
        expectancy = raw.get("expectancy_audit")
        if isinstance(expectancy, dict):
            policy = expectancy.get("policy_expectancy")
            if isinstance(policy, dict):
                policy_lcb = _number_from_mapping(policy, "expectancy_lcb", default=float("nan"))
    if min_policy_expectancy_lcb is not None and (
        not np.isfinite(policy_lcb) or policy_lcb < float(min_policy_expectancy_lcb)
    ):
        shown = f"{policy_lcb:.4g}" if np.isfinite(policy_lcb) else "missing"
        reasons.append(f"policy_expectancy_lcb_below_limit:{shown}<{float(min_policy_expectancy_lcb):g}")

    negative_scopes = _number(
        "fitness_context_negative_critical",
        "negative_critical_scope_count",
        default=float("nan"),
    )
    if not np.isfinite(negative_scopes):
        expectancy = raw.get("expectancy_audit")
        if isinstance(expectancy, dict):
            negative_scopes = _number_from_mapping(
                expectancy,
                "negative_critical_scope_count",
                default=float("nan"),
            )
    if max_negative_context_scopes is not None and (
        not np.isfinite(negative_scopes) or negative_scopes > int(max_negative_context_scopes)
    ):
        shown = str(int(negative_scopes)) if np.isfinite(negative_scopes) else "missing"
        reasons.append(f"negative_context_scopes_above_limit:{shown}>{int(max_negative_context_scopes)}")

    if require_feature_parity:
        feature_valid = _bool("feature_audit_valid", "feature_parity_numerical_valid")
        if feature_valid is None:
            feature = raw.get("feature_audit")
            if isinstance(feature, dict):
                feature_valid = _bool_from_mapping(feature, "valid")
        if feature_valid is None:
            reasons.append("feature_parity_evidence_missing")
        elif not feature_valid:
            reasons.append("feature_parity_invalid")

    if require_real_tick_evidence:
        real_ticks = _bool("real_tick_evidence_verified")
        if real_ticks is None:
            evidence = raw.get("real_tick_evidence")
            if isinstance(evidence, dict):
                real_ticks = _bool_from_mapping(evidence, "verified")
        if real_ticks is None:
            reasons.append("real_tick_evidence_missing")
        elif not real_ticks:
            reasons.append("real_tick_evidence_invalid")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "sample_count": int(lifecycle_count) if np.isfinite(lifecycle_count) else None,
        "max_drawdown_pct": float(max_observed_drawdown) if np.isfinite(max_observed_drawdown) else None,
        "policy_expectancy_lcb": float(policy_lcb) if np.isfinite(policy_lcb) else None,
        "negative_context_scopes": int(negative_scopes) if np.isfinite(negative_scopes) else None,
    }


def paired_oos_non_regression_gate(
    candidate_metrics: dict[str, Any] | None,
    baseline_metrics: dict[str, Any] | None,
    *,
    min_candidate_expectancy: float = 0.0,
    max_profit_regression: float = 0.0,
    max_expectancy_regression: float = 0.02,
    max_drawdown_increase_pct: float = 1.0,
) -> dict[str, Any]:
    """Require a frozen candidate to keep the seed's paired OOS behaviour.

    A candidate that beats its own training fitness but loses to the protected
    seed on the same OOS window is overfit by definition.  This gate is
    deliberately small and metric based: it does not replace
    :func:`frozen_oos_promotion_gate`, it adds a paired non-regression check
    before a walk-forward row can be promoted.
    """
    candidate = dict(candidate_metrics or {})
    baseline = dict(baseline_metrics or {})
    reasons: list[str] = []

    def _number(mapping: dict[str, Any], *keys: str) -> float:
        # Paired OOS compares the current evaluation window.  Prefer its
        # execution summary over flattened checkpoint-level audit fields,
        # which may include training and failure-replay history.
        summary = mapping.get("execution_summary")
        if isinstance(summary, dict):
            for key in keys:
                if key in summary and summary.get(key) is not None:
                    try:
                        value = float(summary[key])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        return value
        for key in keys:
            if key in mapping and mapping.get(key) is not None:
                try:
                    value = float(mapping[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
        return float("nan")

    candidate_profit = _number(candidate, "net_profit", "profit")
    baseline_profit = _number(baseline, "net_profit", "profit")
    candidate_expectancy = _number(candidate, "edge", "expectancy")
    baseline_expectancy = _number(baseline, "edge", "expectancy")
    candidate_lcb = _number(candidate, "fitness_policy_expectancy_lcb", "policy_expectancy_lcb", "expectancy_lcb")
    baseline_lcb = _number(baseline, "fitness_policy_expectancy_lcb", "policy_expectancy_lcb", "expectancy_lcb")
    candidate_dd = _number(candidate, "equity_drawdown_pct", "balance_drawdown_pct")
    baseline_dd = _number(baseline, "equity_drawdown_pct", "balance_drawdown_pct")

    if not np.isfinite(candidate_expectancy) or candidate_expectancy < float(min_candidate_expectancy):
        shown = f"{candidate_expectancy:.6g}" if np.isfinite(candidate_expectancy) else "missing"
        reasons.append(f"candidate_expectancy_below_floor:{shown}<{float(min_candidate_expectancy):g}")
    if not np.isfinite(candidate_profit) or not np.isfinite(baseline_profit):
        reasons.append("paired_profit_missing")
    elif candidate_profit < baseline_profit - float(max_profit_regression):
        reasons.append(
            f"paired_profit_regression:{candidate_profit:.6g}<{baseline_profit - float(max_profit_regression):.6g}"
        )
    if np.isfinite(candidate_lcb) and np.isfinite(baseline_lcb):
        if candidate_lcb < baseline_lcb - float(max_expectancy_regression):
            reasons.append(
                f"paired_expectancy_lcb_regression:{candidate_lcb:.6g}"
                f"<{baseline_lcb - float(max_expectancy_regression):.6g}"
            )
    elif not np.isfinite(candidate_lcb) or not np.isfinite(baseline_lcb):
        if not np.isfinite(candidate_lcb):
            reasons.append("candidate_expectancy_lcb_missing")
        if not np.isfinite(baseline_lcb):
            reasons.append("baseline_expectancy_lcb_missing")
    if np.isfinite(candidate_dd) and np.isfinite(baseline_dd):
        if candidate_dd > baseline_dd + float(max_drawdown_increase_pct):
            reasons.append(
                f"paired_drawdown_increase:{candidate_dd:.6g}>{baseline_dd + float(max_drawdown_increase_pct):.6g}"
            )
    elif not np.isfinite(candidate_dd) or not np.isfinite(baseline_dd):
        if not np.isfinite(candidate_dd):
            reasons.append("candidate_drawdown_missing")
        if not np.isfinite(baseline_dd):
            reasons.append("baseline_drawdown_missing")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "candidate_profit": float(candidate_profit) if np.isfinite(candidate_profit) else None,
        "baseline_profit": float(baseline_profit) if np.isfinite(baseline_profit) else None,
        "candidate_expectancy": float(candidate_expectancy) if np.isfinite(candidate_expectancy) else None,
        "baseline_expectancy": float(baseline_expectancy) if np.isfinite(baseline_expectancy) else None,
        "candidate_expectancy_lcb": float(candidate_lcb) if np.isfinite(candidate_lcb) else None,
        "baseline_expectancy_lcb": float(baseline_lcb) if np.isfinite(baseline_lcb) else None,
        "candidate_drawdown_pct": float(candidate_dd) if np.isfinite(candidate_dd) else None,
        "baseline_drawdown_pct": float(baseline_dd) if np.isfinite(baseline_dd) else None,
    }


def _number_from_mapping(mapping: dict[str, Any], key: str, *, default: float = float("nan")) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _bool_from_mapping(mapping: dict[str, Any], key: str) -> bool | None:
    if key not in mapping or mapping.get(key) is None:
        return None
    value = mapping.get(key)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)
