"""MT5-native failure-context curriculum for the public Five policy.

The replay contract is deliberately on-policy:

* a settled losing entry contributes only a market/time anchor to the bank;
* Strategy Tester is relaunched on the historical tick slice around that anchor;
* the CURRENT policy generates fresh actions and behavior likelihoods;
* only those fresh MT5 transitions reach PPO/lifecycle training;
* stale rewards, stale actions and stale PPO ratios are never replayed.

MT5 cannot seek backwards inside an active tester pass, therefore each replay is
an independent bounded Strategy Tester session.  The EA gates bridge requests
to the intraday replay window and calls ``TesterStop`` after the horizon.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping

from .five_failure_replay import FiveFailureReplayBank, FiveFailureReplayCase
from .mt5_runner import MT5BacktestConfig, MT5StrategyTesterEvaluator


_MT5_TIME_FORMATS = (
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
)


def parse_mt5_failure_anchor(value: str) -> datetime:
    text = str(value or "").strip()
    for fmt in _MT5_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"invalid MT5 failure replay anchor: {text!r}")


@dataclass(frozen=True)
class MT5FailureReplayWindow:
    case_id: str
    anchor: datetime
    start: datetime
    end: datetime
    from_date: str
    to_date: str


def build_mt5_failure_replay_window(
    case: FiveFailureReplayCase,
    *,
    warmup_minutes: int,
    horizon_minutes: int,
) -> MT5FailureReplayWindow:
    anchor = parse_mt5_failure_anchor(str(getattr(case, "market_time", "") or ""))
    start = anchor - timedelta(minutes=max(1, int(warmup_minutes)))
    end = anchor + timedelta(minutes=max(1, int(horizon_minutes)))
    # MT5 Tester FromDate/ToDate are day-granularity. The EA performs the exact
    # intraday gate; make ToDate safely include the replay end day.
    tester_to = end.date() + timedelta(days=1)
    return MT5FailureReplayWindow(
        case_id=str(case.case_id),
        anchor=anchor,
        start=start,
        end=end,
        from_date=start.strftime("%Y.%m.%d"),
        to_date=tester_to.strftime("%Y.%m.%d"),
    )


@dataclass
class MT5FailureReplayRunSummary:
    enabled: bool = False
    requested_sessions: int = 0
    completed_sessions: int = 0
    failed_sessions: int = 0
    skipped_cases: int = 0
    replayed_case_ids: list[str] = field(default_factory=list)
    resolved_case_ids: list[str] = field(default_factory=list)
    reopened_case_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_bank: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "requested_sessions": int(self.requested_sessions),
            "completed_sessions": int(self.completed_sessions),
            "failed_sessions": int(self.failed_sessions),
            "skipped_cases": int(self.skipped_cases),
            "replayed_case_ids": list(self.replayed_case_ids),
            "resolved_case_ids": list(self.resolved_case_ids),
            "reopened_case_ids": list(self.reopened_case_ids),
            "errors": list(self.errors),
            "final_bank": dict(self.final_bank),
        }


class MT5FailureReplayCoordinator:
    """Run unresolved failure anchors through fresh MT5 Strategy Tester passes."""

    def __init__(
        self,
        base_config: MT5BacktestConfig,
        *,
        evaluator_factory: Callable[[MT5BacktestConfig], MT5StrategyTesterEvaluator] = MT5StrategyTesterEvaluator,
        journal_path: str | Path | None = None,
    ) -> None:
        self.base_config = base_config
        self.evaluator_factory = evaluator_factory
        self.journal_path = Path(journal_path) if journal_path else None

    @staticmethod
    def _owner_policy(policy: object) -> object | None:
        hook = getattr(policy, "five_failure_replay_owner_policy", None)
        if callable(hook):
            return hook()
        if isinstance(getattr(policy, "five_failure_replay_bank", None), FiveFailureReplayBank):
            return policy
        return None

    @staticmethod
    def _reset_runtime(policy: object) -> None:
        hook = getattr(policy, "reset_runtime_state", None)
        if callable(hook):
            hook()

    @staticmethod
    def _audit(policy: object, case_ids: list[str]) -> dict[str, object]:
        if not case_ids:
            return {"audited": 0, "probabilities": {}, "bank": {}}
        hook = getattr(policy, "audit_five_failure_replay_cases", None)
        if callable(hook):
            result = hook(case_ids)
            return dict(result) if isinstance(result, Mapping) else {"audited": 0}
        owner = MT5FailureReplayCoordinator._owner_policy(policy)
        owner_hook = getattr(owner, "audit_five_failure_replay_cases", None)
        if callable(owner_hook):
            result = owner_hook(case_ids)
            return dict(result) if isinstance(result, Mapping) else {"audited": 0}
        return {"audited": 0, "probabilities": {}, "bank": {}}

    def _append_journal(self, payload: Mapping[str, object]) -> None:
        if self.journal_path is None:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")

    def _replay_config(
        self,
        case: FiveFailureReplayCase,
        window: MT5FailureReplayWindow,
        *,
        warmup_minutes: int,
        horizon_minutes: int,
    ) -> MT5BacktestConfig:
        symbol = str(getattr(case, "symbol", "") or self.base_config.symbol)
        period = str(getattr(case, "period", "") or self.base_config.period)
        return replace(
            self.base_config,
            symbol=symbol,
            period=period,
            from_date=window.from_date,
            to_date=window.to_date,
            visual=False,
            # The parent training transaction may deliberately keep one
            # portable terminal alive across replay sessions so mandatory
            # LiveUpdate cannot erase the runner between launches.
            shutdown_terminal=bool(self.base_config.shutdown_terminal),
            # Replay is training, but it must not be aborted because a known
            # failure window is intentionally poor. Safety/execution limits in
            # the EA remain active.
            learning_enabled=True,
            runtime_fitness_guard_enabled=False,
            reset_execution_ledger_on_init=True,
            mt5_failure_replay_mode=True,
            mt5_failure_replay_case_id=str(case.case_id),
            mt5_failure_replay_anchor_time=window.anchor.strftime("%Y.%m.%d %H:%M"),
            mt5_failure_replay_warmup_minutes=max(1, int(warmup_minutes)),
            mt5_failure_replay_horizon_minutes=max(1, int(horizon_minutes)),
        )

    def run(
        self,
        policy: object,
        *,
        sessions: int,
        max_attempts_per_case: int,
        warmup_minutes: int,
        horizon_minutes: int,
        resolved_audit_items: int = 16,
        round_index: int = 0,
        policy_id_base: int = 800_000,
    ) -> MT5FailureReplayRunSummary:
        summary = MT5FailureReplayRunSummary(
            enabled=True,
            requested_sessions=max(0, int(sessions)),
        )
        owner = self._owner_policy(policy)
        bank = getattr(owner, "five_failure_replay_bank", None)
        if not isinstance(bank, FiveFailureReplayBank) or int(sessions) <= 0:
            summary.enabled = bool(isinstance(bank, FiveFailureReplayBank))
            summary.final_bank = bank.stats() if isinstance(bank, FiveFailureReplayBank) else {}
            return summary
        bank.ensure_compatible()

        # Dormant failures remain in memory. Audit a bounded set for regression;
        # if current-policy probability has risen, the bank reopens them and the
        # next selection below puts them back through MT5.
        resolved_due = [
            row.case_id for row in bank.audit_candidates(
                audit_interval=max(1, int(getattr(owner, "five_failure_replay_audit_interval", 8) or 8)),
                max_items=max(0, int(resolved_audit_items)),
            )
            if bool(getattr(row, "resolved", False))
        ]
        if resolved_due:
            audit = self._audit(policy, resolved_due)
            if int(audit.get("newly_reopened", 0) or 0) > 0:
                summary.reopened_case_ids.extend(resolved_due)

        local_attempts: dict[str, int] = {}
        max_per_case = max(1, int(max_attempts_per_case))
        for session_index in range(max(0, int(sessions))):
            # Select again after every optimizer-bearing MT5 pass. If a case is
            # still unresolved, it remains eligible and therefore returns to the
            # training distribution rather than disappearing after one replay.
            pool = bank.mt5_replay_candidates(count=max(1, len(bank.cases)))
            pool = [
                row for row in pool
                if local_attempts.get(str(row.case_id), 0) < max_per_case
            ]
            if not pool:
                break
            # Preserve the bank's severity/bucket ordering, but never let one
            # unresolved context monopolize the replay budget. Every eligible
            # case gets its next attempt before a case with more local attempts
            # is selected again. With a single unresolved case this naturally
            # repeats the same historical test until it is repaired.
            case = min(
                enumerate(pool),
                key=lambda item: (
                    local_attempts.get(str(item[1].case_id), 0),
                    int(item[0]),
                ),
            )[1]
            case_id = str(case.case_id)
            local_attempts[case_id] = local_attempts.get(case_id, 0) + 1
            try:
                window = build_mt5_failure_replay_window(
                    case,
                    warmup_minutes=warmup_minutes,
                    horizon_minutes=horizon_minutes,
                )
            except ValueError as exc:
                summary.skipped_cases += 1
                summary.errors.append(f"{case_id}:{exc}")
                # An invalid old anchor must not spin forever in this cycle.
                local_attempts[case_id] = max_per_case
                continue

            replay_config = self._replay_config(
                case,
                window,
                warmup_minutes=warmup_minutes,
                horizon_minutes=horizon_minutes,
            )
            evaluator = self.evaluator_factory(replay_config)
            # A second /config command forwarded to an idle portable terminal
            # is not reliable: MT5 may accept the launcher process but never
            # start a tester or write a fresh bridge request.  Close the
            # runner-scoped terminal before every replay, then the finally
            # block below closes the new one after its report is archived.
            close_before_launch = getattr(evaluator, "close_runner_processes", None)
            if callable(close_before_launch):
                close_before_launch()
            # Keep admission enabled inside MT5 replay. If the current policy
            # merely swaps the parent failure for a different losing action,
            # that alternative mistake becomes a deduplicated child case. The
            # policy-side admission path reinforces the parent when the exact
            # same mistake repeats, preventing recursive trace explosion.
            previous_suppressed = bool(getattr(owner, "five_failure_replay_admission_suppressed", False))
            setattr(owner, "five_failure_replay_admission_suppressed", False)
            self._reset_runtime(policy)
            record: dict[str, object] = {
                "schema_version": 1,
                "type": "mt5_failure_replay_session",
                "session_index": int(session_index),
                "case_id": case_id,
                "condition_key": str(case.condition_key),
                "failed_action_index": int(case.failed_action_index),
                "original_realized_r": float(case.realized_r),
                "anchor_time": window.anchor.strftime("%Y.%m.%d %H:%M"),
                "from_date": window.from_date,
                "to_date": window.to_date,
            }
            try:
                fitness, metrics = evaluator.evaluate_policy(
                    int(policy_id_base + session_index), policy, int(round_index)
                )
                anchor_seen = bool(getattr(evaluator, "_mt5_failure_replay_anchor_seen", False))
                if not anchor_seen:
                    raise RuntimeError(
                        f"MT5 replay session never reached anchor {window.anchor:%Y.%m.%d %H:%M}"
                    )
                bank.note_mt5_replay(
                    case_id,
                    fitness=float(fitness),
                    net_profit=float(getattr(metrics, "net_profit", 0.0) or 0.0),
                    successful_session=True,
                    replayed_at=datetime.now().isoformat(),
                )
                audit = self._audit(policy, [case_id])
                summary.completed_sessions += 1
                summary.replayed_case_ids.append(case_id)
                if int(audit.get("newly_resolved", 0) or 0) > 0:
                    summary.resolved_case_ids.append(case_id)
                if int(audit.get("newly_reopened", 0) or 0) > 0:
                    summary.reopened_case_ids.append(case_id)
                record.update({
                    "status": "completed",
                    "fitness": float(fitness),
                    "net_profit": float(getattr(metrics, "net_profit", 0.0) or 0.0),
                    "total_trades": int(getattr(metrics, "total_trades", 0) or 0),
                    "anchor_seen": True,
                    "audit": audit,
                    "bank": bank.stats(),
                })
            except Exception as exc:
                summary.failed_sessions += 1
                summary.errors.append(f"{case_id}:{type(exc).__name__}:{exc}")
                record.update({
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "anchor_seen": bool(getattr(evaluator, "_mt5_failure_replay_anchor_seen", False)),
                    "bank": bank.stats(),
                })
            finally:
                setattr(owner, "five_failure_replay_admission_suppressed", previous_suppressed)
                close = getattr(evaluator, "close_runner_processes", None)
                if callable(close):
                    close()
                self._append_journal(record)

        summary.final_bank = bank.stats()
        return summary
