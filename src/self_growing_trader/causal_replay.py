"""Deterministic counterfactual replay protocol for Causal Ledger V19.

The module never fabricates a replay result.  It creates bounded requests from
an already-settled causal report, accepts results from a deterministic market
replay/Strategy-Tester worker, and marks a hypothesis validated only when the
worker returns an explicitly verified result with provenance.

The Python project deliberately keeps the executor transport-independent:
MT5/EA can append results to JSONL, an offline historical simulator can call
``submit_results`` directly, and tests can use a small callback.  This keeps
causal learning correct even when the canonical EA is unavailable in an
archive.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .causal_ledger import (
    CAUSAL_SCHEMA_VERSION,
    CauseHypothesis,
    CounterfactualPlan,
    CausalOutcomeReport,
    CausalLedger,
    build_credit_plan,
)


REPLAY_SCHEMA_VERSION = 19


@dataclass(frozen=True)
class CounterfactualReplayRequest:
    request_id: str
    trace_id: str
    cause_id: str
    plan_name: str
    family: str
    parameters: dict[str, float | str]
    action_name: str
    context_frame_id: str
    realized_r: float
    created_at: float
    requires_execution_model: bool = True
    schema_version: int = REPLAY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualReplayResult:
    request_id: str
    trace_id: str
    plan_name: str
    result_r: float | None
    verified: bool
    execution_model: str = ""
    data_provenance: str = ""
    notes: str = ""
    completed_at: float = 0.0
    schema_version: int = REPLAY_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CounterfactualReplayResult":
        raw = values.get("result_r")
        try:
            result_r = None if raw is None or str(raw).strip() == "" else float(raw)
        except (TypeError, ValueError):
            result_r = None
        return cls(
            request_id=str(values.get("request_id", "") or ""),
            trace_id=str(values.get("trace_id", "") or ""),
            plan_name=str(values.get("plan_name", "") or ""),
            result_r=result_r,
            verified=bool(values.get("verified", False)),
            execution_model=str(values.get("execution_model", "") or ""),
            data_provenance=str(values.get("data_provenance", "") or ""),
            notes=str(values.get("notes", "") or ""),
            completed_at=float(values.get("completed_at", 0.0) or 0.0),
        )


def replay_requests(report: CausalOutcomeReport) -> tuple[CounterfactualReplayRequest, ...]:
    """Turn unvalidated hypothesis plans into exact, deduplicated requests."""
    if report.snapshot is None:
        return ()
    requests: list[CounterfactualReplayRequest] = []
    seen: set[tuple[str, str]] = set()
    for hypothesis in report.hypotheses:
        for plan in hypothesis.counterfactuals:
            if plan.status not in {"needs_replay", "shadow", "evidence_only"}:
                continue
            key = (hypothesis.cause_id, plan.name)
            if key in seen:
                continue
            seen.add(key)
            request_id = f"{report.trace_id}:{hypothesis.cause_id}:{plan.name}"
            requests.append(CounterfactualReplayRequest(
                request_id=request_id,
                trace_id=report.trace_id,
                cause_id=hypothesis.cause_id,
                plan_name=plan.name,
                family=plan.family,
                parameters=dict(plan.parameters),
                action_name=report.snapshot.action_name,
                context_frame_id=report.snapshot.context_frame_id,
                realized_r=float(report.outcome.realized_r),
                created_at=time.time(),
            ))
    return tuple(requests)


def _validates(cause_id: str, plan_name: str, result_r: float, actual_r: float) -> bool:
    """Strict cause-specific replay validation; no result means no validation."""
    if cause_id == "stop_loss_too_tight":
        return plan_name.startswith("sl_") and result_r > max(0.0, actual_r + 0.20)
    if cause_id == "wrong_direction_or_regime":
        return plan_name == "opposite_direction" and result_r > max(0.15, actual_r + 0.30)
    if cause_id == "entry_too_early":
        return plan_name.startswith("delay_") and result_r > max(0.10, actual_r + 0.25)
    if cause_id in {"management_failure", "take_profit_too_far", "take_profit_too_close"}:
        return result_r > actual_r + 0.20
    if cause_id in {"volatility_shock", "spread_or_slippage_damage"}:
        return plan_name.startswith(("skip_", "limit_")) and result_r >= 0.0
    if cause_id in {"memory_mismatch", "reasoning_override_error"}:
        return plan_name.endswith("_off") and result_r > actual_r + 0.20
    return False


def apply_replay_results(
    report: CausalOutcomeReport,
    results: Iterable[CounterfactualReplayResult | Mapping[str, Any]],
) -> CausalOutcomeReport:
    """Return an immutable report enriched only by verified replay results."""
    result_index: dict[str, CounterfactualReplayResult] = {}
    for raw in results:
        result = raw if isinstance(raw, CounterfactualReplayResult) else CounterfactualReplayResult.from_mapping(raw)
        if result.trace_id == report.trace_id and result.plan_name:
            result_index[result.plan_name] = result

    changed = False
    hypotheses: list[CauseHypothesis] = []
    any_validated = False
    for hypothesis in report.hypotheses:
        plans: list[CounterfactualPlan] = []
        validated = False
        replayed = False
        for plan in hypothesis.counterfactuals:
            result = result_index.get(plan.name)
            if result is None:
                plans.append(plan)
                continue
            replayed = True
            verified = bool(result.verified and result.result_r is not None and result.execution_model and result.data_provenance)
            status = "replay_verified" if verified else "replay_unverified"
            evidence = (result.execution_model + " | " + result.data_provenance).strip(" |")
            plan_new = replace(plan, status=status, result_r=result.result_r, evidence_source=evidence)
            plans.append(plan_new)
            changed = True
            if verified and result.result_r is not None and _validates(hypothesis.cause_id, plan.name, result.result_r, report.outcome.realized_r):
                validated = True
        status = hypothesis.status
        confidence = hypothesis.confidence
        if validated:
            status = "replay_validated"
            confidence = min(0.98, confidence + 0.15)
            any_validated = True
        elif replayed and status != "replay_validated":
            status = "replayed_inconclusive"
        hypotheses.append(replace(hypothesis, counterfactuals=tuple(plans), status=status, confidence=confidence))
    if not changed:
        return report
    coverage = dict(report.coverage)
    coverage["counterfactual_replay_validated"] = bool(any_validated)
    coverage["counterfactual_replay_results"] = len(result_index)
    refreshed_credit = build_credit_plan(report.snapshot, report.outcome, hypotheses)
    return replace(report, hypotheses=tuple(hypotheses), coverage=coverage, credit_plan=refreshed_credit)


class CounterfactualReplayJournal:
    """Append-only JSONL request/result journal used by MT5 or offline replay."""

    def __init__(self, request_path: str | Path, result_path: str | Path | None = None) -> None:
        self.request_path = Path(request_path)
        self.result_path = Path(result_path) if result_path else self.request_path.with_name("causal_replay_results.jsonl")

    def enqueue(self, report: CausalOutcomeReport) -> list[CounterfactualReplayRequest]:
        requests = list(replay_requests(report))
        if not requests:
            return requests
        self.request_path.parent.mkdir(parents=True, exist_ok=True)
        known: set[str] = set()
        if self.request_path.exists():
            for line in self.request_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    item = json.loads(line)
                    known.add(str(item.get("request_id", "")))
                except Exception:
                    continue
        with self.request_path.open("a", encoding="utf-8") as handle:
            for request in requests:
                if request.request_id in known:
                    continue
                handle.write(json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return requests

    def read_results(self, trace_id: str) -> list[CounterfactualReplayResult]:
        if not self.result_path.exists():
            return []
        output: list[CounterfactualReplayResult] = []
        for line in self.result_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                item = json.loads(line)
            except Exception:
                continue
            result = CounterfactualReplayResult.from_mapping(item)
            if result.trace_id == str(trace_id):
                output.append(result)
        return output

    def apply_to_ledger(self, ledger: CausalLedger, trace_id: str) -> CausalOutcomeReport | None:
        report = ledger.report_for(trace_id)
        if report is None:
            return None
        updated = apply_replay_results(report, self.read_results(trace_id))
        ledger.reports[str(trace_id)] = updated
        return updated
