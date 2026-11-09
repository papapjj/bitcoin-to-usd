from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .graph_metrics import graph_metrics
from .graph_plasticity import GraphCreditSignal
from .module_need import ModuleErrorCluster, ModuleErrorTrace, ModuleNeedSignal, cluster_module_errors


@dataclass(frozen=True)
class CandidateValidationResult:
    candidate_id: str
    replay_passed: bool
    mt5_passed: bool
    promoted: bool
    reject_reasons: tuple[str, ...] = ()
    replay_score: float | None = None
    mt5_score: float | None = None
    telemetry_complete: bool = False
    created_at: str = ""


@dataclass(frozen=True)
class SupernetworkTrialRecord:
    candidate_id: str
    parent_policy_id: str
    structural_diff: dict[str, object]
    module_need_reason: str
    validation: CandidateValidationResult
    graph_metrics: dict[str, float]
    created_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_policy_id": self.parent_policy_id,
            "structural_diff": self.structural_diff,
            "module_need_reason": self.module_need_reason,
            "validation": {
                "replay_passed": self.validation.replay_passed,
                "mt5_passed": self.validation.mt5_passed,
                "promoted": self.validation.promoted,
                "reject_reasons": list(self.validation.reject_reasons),
                "replay_score": self.validation.replay_score,
                "mt5_score": self.validation.mt5_score,
                "telemetry_complete": self.validation.telemetry_complete,
            },
            "graph_metrics": self.graph_metrics,
            "created_at": self.created_at,
        }


class CandidatePromotionGate:
    """Replay + MT5 promotion contract with explicit reject reasons."""

    def validate(
        self,
        *,
        candidate_id: str,
        candidate: object,
        parent_score: float | None,
        replay_validator: Callable[[object], float | tuple[bool, float] | None] | None,
        mt5_validator: Callable[[object], float | tuple[bool, float, bool] | None] | None,
    ) -> CandidateValidationResult:
        reasons: list[str] = []
        replay_passed, replay_score = self._run_replay(candidate, replay_validator)
        if not replay_passed:
            reasons.append("replay_regression")
            return CandidateValidationResult(candidate_id, False, False, False, tuple(reasons), replay_score=replay_score)
        mt5_passed, mt5_score, telemetry_complete = self._run_mt5(candidate, mt5_validator)
        if not telemetry_complete:
            reasons.append("missing_telemetry")
        if not mt5_passed:
            reasons.append("mt5_validation_failed")
        if parent_score is not None and mt5_score is not None and mt5_score <= parent_score:
            reasons.append("fitness_regression")
        promoted = replay_passed and mt5_passed and telemetry_complete and not reasons
        return CandidateValidationResult(
            candidate_id=candidate_id,
            replay_passed=replay_passed,
            mt5_passed=mt5_passed,
            promoted=promoted,
            reject_reasons=tuple(reasons),
            replay_score=replay_score,
            mt5_score=mt5_score,
            telemetry_complete=telemetry_complete,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _run_replay(candidate: object, validator: Callable[[object], float | tuple[bool, float] | None] | None) -> tuple[bool, float | None]:
        if validator is None:
            # No replay evaluator means a candidate is not eligible for MT5.
            return False, None
        value = validator(candidate)
        if value is None:
            return False, None
        if isinstance(value, tuple):
            return bool(value[0]), float(value[1]) if len(value) > 1 and value[1] is not None else None
        return bool(np.isfinite(float(value))), float(value)

    @staticmethod
    def _run_mt5(candidate: object, validator: Callable[[object], float | tuple[bool, float, bool] | None] | None) -> tuple[bool, float | None, bool]:
        if validator is None:
            return False, None, False
        value = validator(candidate)
        if value is None:
            return False, None, False
        if isinstance(value, tuple):
            passed = bool(value[0])
            score = float(value[1]) if len(value) > 1 and value[1] is not None else None
            telemetry = bool(value[2]) if len(value) > 2 else False
            return passed, score, telemetry
        return bool(np.isfinite(float(value))), float(value), False


def load_valid_journal_transitions(
    path: str | Path,
    *,
    max_items: int = 200,
    memory_schema: int = 6,
) -> list[dict[str, object]]:
    """Read bounded, replay-safe experience without copying parent memory state."""
    file_path = Path(path)
    if not file_path.exists() or max_items <= 0:
        return []
    rows: list[dict[str, object]] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            trace = row.get("trace", {}) if isinstance(row.get("trace", {}), dict) else {}
            schema = int(row.get("memory_key_schema", trace.get("memory_key_schema", 0)) or 0)
            execution = dict(row.get("execution_details", trace.get("execution_details", {})) or {})
            if schema != memory_schema or not bool(execution.get("memory_eligible", False)):
                continue
            if float(row.get("td_delta", 0.0) or 0.0) <= 0.0 and float(row.get("reward", 0.0) or 0.0) <= 0.0:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (float(row.get("td_delta", 0.0) or 0.0), float(row.get("reward", 0.0) or 0.0)), reverse=True)
    return rows[:max_items]


def error_clusters_from_journal(rows: Iterable[dict[str, object]]) -> list[ModuleErrorCluster]:
    traces: list[ModuleErrorTrace] = []
    for row in rows:
        trace = row.get("trace", {}) if isinstance(row.get("trace", {}), dict) else {}
        tags = tuple(str(item) for item in trace.get("lifecycle_tags", row.get("lifecycle_tags", [])) or [])
        context = dict(trace.get("account_context", row.get("account_context", {})) or {})
        traces.append(ModuleErrorTrace(
            trace_id=str(row.get("trace_id", "journal")),
            regime=str(row.get("regime", "journal")),
            lifecycle_tags=tags,
            prediction_error=float(row.get("prediction_error", trace.get("prediction_error", 0.0)) or 0.0),
            novelty=float(trace.get("novelty", 0.0) or 0.0),
            conflict=float(trace.get("conflict", 0.0) or 0.0),
            memory_match_quality=float(trace.get("memory_match_quality", 0.0) or 0.0),
            reward=float(row.get("reward", 0.0) or 0.0),
            td_delta=float(row.get("td_delta", 0.0) or 0.0),
            account_stress=float(context.get("current_equity_drawdown", 0.0) or 0.0),
            drawdown_entry_loss=float(context.get("open_risk_ratio", 0.0) or 0.0),
        ))
    return cluster_module_errors(traces)


def write_trial_history(path: str | Path, record: SupernetworkTrialRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")


def module_need_from_cluster(cluster: ModuleErrorCluster | None) -> ModuleNeedSignal:
    return cluster.signal if cluster is not None else ModuleNeedSignal()


def candidate_graph_metrics(candidate: object) -> dict[str, float]:
    graph = getattr(candidate, "graph", None)
    return graph_metrics(graph) if graph is not None else {}
