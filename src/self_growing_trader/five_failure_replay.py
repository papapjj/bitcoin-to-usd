"""Persistent failure-context replay for the public Five policy.

The bank does **not** replay stale PPO trajectories.  It stores immutable
settled failure anchors (observation + recurrent state + failed entry), asks the
*current* policy to re-evaluate those anchors, and only returns anchors that the
current policy has not demonstrably repaired yet.  Repaired anchors remain in
an audit pool so regression re-opens them automatically.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


FIVE_FAILURE_REPLAY_SCHEMA_VERSION = 2


def _array(value: object, *, dtype: np.dtype | type) -> np.ndarray:
    return np.asarray(value, dtype=dtype).copy()


@dataclass
class FiveFailureReplayCase:
    case_id: str
    trace_id: str
    condition_key: str
    failed_agent_index: int
    failed_action_index: int
    realized_r: float
    severity: float
    observations: np.ndarray
    memories_hidden: np.ndarray
    memories_cell: np.ndarray
    memories_temporal_tokens: np.ndarray | None
    memories_temporal_valid: np.ndarray | None
    previous_actions: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    delay_indices: np.ndarray
    risk_indices: np.ndarray
    entry_offset_indices: np.ndarray
    sl_indices: np.ndarray
    tp_indices: np.ndarray
    trailing_indices: np.ndarray
    manage_mode_indices: np.ndarray
    close_fraction_indices: np.ndarray
    allowed_action_masks: np.ndarray
    # MT5 replay anchor.  This is deliberately market/time metadata, not an
    # old PPO trajectory: the Strategy Tester will revisit the historical
    # market slice and the current policy will generate fresh actions/log-probs.
    market_time: str = ""
    symbol: str = ""
    period: str = "M1"
    source_event_id: str = ""
    source_sequence: int = 0
    mt5_replay_count: int = 0
    mt5_replay_success_count: int = 0
    last_mt5_replay_fitness: float = 0.0
    last_mt5_replay_net_profit: float = 0.0
    last_mt5_replay_at: str = ""
    origin: str = "normal"
    parent_case_id: str = ""
    admitted_batch: int = 0
    replay_count: int = 0
    audit_count: int = 0
    reopen_count: int = 0
    repair_streak: int = 0
    resolved: bool = False
    last_failed_action_probability: float = float("nan")
    last_audit_batch: int = -1

    def copy(self) -> "FiveFailureReplayCase":
        return copy.deepcopy(self)

    @property
    def bucket(self) -> tuple[str, int]:
        return (str(self.condition_key or "UNKNOWN"), int(self.failed_action_index))

    def training_weight(self) -> float:
        # Bounded severity: a -3R event should matter more than -0.3R, but
        # replay must not overwhelm fresh lifecycle evidence.
        return float(np.clip(0.5 + 0.5 * float(self.severity), 0.50, 1.50))


@dataclass
class FiveFailureReplayBank:
    schema_version: int = FIVE_FAILURE_REPLAY_SCHEMA_VERSION
    capacity: int = 256
    cases: dict[str, FiveFailureReplayCase] = field(default_factory=dict, repr=False)
    order: list[str] = field(default_factory=list, repr=False)
    batch_index: int = 0
    admitted_count: int = 0
    replayed_count: int = 0
    resolved_count: int = 0
    reopened_count: int = 0
    discarded_count: int = 0

    def copy(self) -> "FiveFailureReplayBank":
        return copy.deepcopy(self)

    def ensure_compatible(self, *, capacity: int | None = None) -> None:
        previous_version = int(getattr(self, "schema_version", 0) or 0)
        self.schema_version = FIVE_FAILURE_REPLAY_SCHEMA_VERSION
        self.capacity = max(8, int(self.capacity if capacity is None else capacity))
        if previous_version != FIVE_FAILURE_REPLAY_SCHEMA_VERSION:
            self.cases = {}
            self.order = []
            self.batch_index = 0
            self.admitted_count = 0
            self.replayed_count = 0
            self.resolved_count = 0
            self.reopened_count = 0
            self.discarded_count = 0
            return
        if not isinstance(getattr(self, "cases", None), dict):
            self.cases = {}
        self.cases = {
            str(key): value for key, value in self.cases.items()
            if isinstance(value, FiveFailureReplayCase)
        }
        raw_order = getattr(self, "order", [])
        self.order = [str(item) for item in raw_order if str(item) in self.cases]
        for case_id in self.cases:
            if case_id not in self.order:
                self.order.append(case_id)
        for name in (
            "batch_index", "admitted_count", "replayed_count", "resolved_count",
            "reopened_count", "discarded_count",
        ):
            setattr(self, name, max(0, int(getattr(self, name, 0) or 0)))
        for case in self.cases.values():
            for name, default in (
                ("market_time", ""), ("symbol", ""), ("period", "M1"),
                ("source_event_id", ""), ("source_sequence", 0),
                ("mt5_replay_count", 0), ("mt5_replay_success_count", 0),
                ("last_mt5_replay_fitness", 0.0),
                ("last_mt5_replay_net_profit", 0.0),
                ("last_mt5_replay_at", ""),
                ("origin", "normal"), ("parent_case_id", ""),
            ):
                if not hasattr(case, name):
                    setattr(case, name, default)
        self._trim()

    @staticmethod
    def build_case_id(
        *, trace_id: str, condition_key: str, failed_action_index: int,
        observations: object,
        dedupe_key: str = "",
    ) -> str:
        obs = np.asarray(observations, dtype=np.float32).reshape(-1)
        # Trace identity is primary.  The short observation fingerprint makes
        # synthetic/offline anchors deterministic even when no trace id exists.
        fingerprint = hashlib.sha256(obs[:64].tobytes()).hexdigest()[:12]
        raw = (
            f"dedupe|{dedupe_key}"
            if str(dedupe_key or "")
            else f"{trace_id}|{condition_key}|{int(failed_action_index)}|{fingerprint}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def add(
        self,
        *,
        trace_id: str,
        condition_key: str,
        failed_agent_index: int,
        failed_action_index: int,
        realized_r: float,
        observations: object,
        memories_hidden: object,
        memories_cell: object,
        memories_temporal_tokens: object | None = None,
        memories_temporal_valid: object | None = None,
        previous_actions: object,
        actions: object,
        targets: object,
        delay_indices: object,
        risk_indices: object,
        entry_offset_indices: object,
        sl_indices: object,
        tp_indices: object,
        trailing_indices: object,
        manage_mode_indices: object,
        close_fraction_indices: object,
        allowed_action_masks: object,
        market_time: str = "",
        symbol: str = "",
        period: str = "M1",
        source_event_id: str = "",
        source_sequence: int = 0,
        origin: str = "normal",
        parent_case_id: str = "",
        dedupe_key: str = "",
    ) -> str:
        self.ensure_compatible()
        case_id = self.build_case_id(
            trace_id=str(trace_id or ""),
            condition_key=str(condition_key or "UNKNOWN"),
            failed_action_index=int(failed_action_index),
            observations=observations,
            dedupe_key=str(dedupe_key or ""),
        )
        severity = float(np.clip(max(0.0, -float(realized_r)), 0.05, 3.0))
        if case_id in self.cases:
            case = self.cases[case_id]
            case.realized_r = min(float(case.realized_r), float(realized_r))
            case.severity = max(float(case.severity), severity)
            # Backfill replay metadata when an old compatible case is seen
            # again under the new MT5 curriculum.
            if not str(getattr(case, "market_time", "") or "") and market_time:
                case.market_time = str(market_time)
            if not str(getattr(case, "source_event_id", "") or "") and source_event_id:
                case.source_event_id = str(source_event_id)
            if not str(getattr(case, "parent_case_id", "") or "") and parent_case_id:
                case.parent_case_id = str(parent_case_id)
            if str(getattr(case, "origin", "normal") or "normal") == "normal" and origin:
                case.origin = str(origin)
            return case_id
        self.cases[case_id] = FiveFailureReplayCase(
            case_id=case_id,
            trace_id=str(trace_id or ""),
            condition_key=str(condition_key or "UNKNOWN"),
            failed_agent_index=int(failed_agent_index),
            failed_action_index=int(failed_action_index),
            realized_r=float(realized_r),
            severity=severity,
            observations=_array(observations, dtype=np.float32),
            memories_hidden=_array(memories_hidden, dtype=np.float32),
            memories_cell=_array(memories_cell, dtype=np.float32),
            memories_temporal_tokens=(
                None if memories_temporal_tokens is None
                else _array(memories_temporal_tokens, dtype=np.float32)
            ),
            memories_temporal_valid=(
                None if memories_temporal_valid is None
                else _array(memories_temporal_valid, dtype=bool)
            ),
            previous_actions=_array(previous_actions, dtype=np.int64),
            actions=_array(actions, dtype=np.int64),
            targets=_array(targets, dtype=np.int64),
            delay_indices=_array(delay_indices, dtype=np.int64),
            risk_indices=_array(risk_indices, dtype=np.int64),
            entry_offset_indices=_array(entry_offset_indices, dtype=np.int64),
            sl_indices=_array(sl_indices, dtype=np.int64),
            tp_indices=_array(tp_indices, dtype=np.int64),
            trailing_indices=_array(trailing_indices, dtype=np.int64),
            manage_mode_indices=_array(manage_mode_indices, dtype=np.int64),
            close_fraction_indices=_array(close_fraction_indices, dtype=np.int64),
            allowed_action_masks=_array(allowed_action_masks, dtype=bool),
            market_time=str(market_time or ""),
            symbol=str(symbol or ""),
            period=str(period or "M1"),
            source_event_id=str(source_event_id or ""),
            source_sequence=max(0, int(source_sequence or 0)),
            origin=str(origin or "normal"),
            parent_case_id=str(parent_case_id or ""),
            admitted_batch=int(self.batch_index),
        )
        self.order.append(case_id)
        self.admitted_count += 1
        self._trim()
        return case_id

    def _trim(self) -> None:
        while len(self.order) > self.capacity:
            # Prefer evicting old resolved anchors. If none are resolved, evict
            # the oldest low-severity/high-replay anchor rather than a fresh hard
            # failure. This keeps the bank bounded without turning it into FIFO.
            candidates = [
                self.cases[item] for item in self.order if item in self.cases
            ]
            if not candidates:
                self.order = []
                break
            resolved = [row for row in candidates if bool(row.resolved)]
            pool = resolved or candidates
            victim = min(
                pool,
                key=lambda row: (
                    0 if row.resolved else 1,
                    float(row.severity),
                    -int(row.replay_count),
                    int(row.admitted_batch),
                ),
            )
            self.cases.pop(victim.case_id, None)
            self.order = [item for item in self.order if item != victim.case_id]
            self.discarded_count += 1

    @staticmethod
    def _balanced(cases: Sequence[FiveFailureReplayCase], count: int) -> list[FiveFailureReplayCase]:
        buckets: dict[tuple[str, int], list[FiveFailureReplayCase]] = {}
        for case in cases:
            buckets.setdefault(case.bucket, []).append(case)
        for rows in buckets.values():
            rows.sort(key=lambda row: (
                -float(np.nan_to_num(row.last_failed_action_probability, nan=1.0)),
                -float(row.severity), int(row.replay_count), int(row.admitted_batch), row.case_id,
            ))
        selected: list[FiveFailureReplayCase] = []
        keys = sorted(buckets, key=lambda value: (str(value[0]), int(value[1])))
        while keys and len(selected) < max(0, int(count)):
            next_keys: list[tuple[str, int]] = []
            for key in keys:
                rows = buckets[key]
                if rows and len(selected) < count:
                    selected.append(rows.pop(0))
                if rows:
                    next_keys.append(key)
            keys = next_keys
        return selected

    def audit_candidates(self, *, audit_interval: int, max_items: int) -> list[FiveFailureReplayCase]:
        self.ensure_compatible()
        interval = max(1, int(audit_interval))
        unresolved = [
            row for row in self.cases.values()
            if not bool(row.resolved) and int(row.admitted_batch) < int(self.batch_index)
        ]
        due_resolved = [
            row for row in self.cases.values()
            if bool(row.resolved)
            and int(row.admitted_batch) < int(self.batch_index)
            and (int(self.batch_index) - int(row.last_audit_batch)) >= interval
        ]
        # Unresolved anchors are checked every lifecycle optimizer batch. A
        # bounded sample of dormant anchors is periodically audited for relapse.
        pool = unresolved + due_resolved
        return [row.copy() for row in self._balanced(pool, max_items)]

    def observe_probabilities(
        self,
        probabilities: Mapping[str, float],
        *, repair_probability: float,
        reopen_probability: float,
        repair_streak_required: int,
    ) -> None:
        repair = float(np.clip(repair_probability, 0.0, 1.0))
        reopen = float(np.clip(max(reopen_probability, repair), 0.0, 1.0))
        required = max(1, int(repair_streak_required))
        for case_id, value in probabilities.items():
            case = self.cases.get(str(case_id))
            if case is None:
                continue
            probability = float(np.clip(float(value), 0.0, 1.0))
            case.last_failed_action_probability = probability
            case.last_audit_batch = int(self.batch_index)
            case.audit_count += 1
            if case.resolved:
                if probability >= reopen:
                    case.resolved = False
                    case.repair_streak = 0
                    case.reopen_count += 1
                    self.reopened_count += 1
                continue
            if probability <= repair:
                case.repair_streak += 1
                if case.repair_streak >= required:
                    case.resolved = True
                    self.resolved_count += 1
            else:
                case.repair_streak = 0

    def active_replay(
        self, *, count: int, repair_probability: float, min_bucket_support: int = 1,
    ) -> list[FiveFailureReplayCase]:
        repair = float(np.clip(repair_probability, 0.0, 1.0))
        support = max(1, int(min_bucket_support))
        active = [
            row for row in self.cases.values()
            if not bool(row.resolved)
            and int(row.admitted_batch) < int(self.batch_index)
            and (
                not np.isfinite(float(row.last_failed_action_probability))
                or float(row.last_failed_action_probability) > repair
            )
        ]
        if support > 1:
            bucket_counts: dict[tuple[str, int], int] = {}
            for row in active:
                bucket_counts[row.bucket] = bucket_counts.get(row.bucket, 0) + 1
            active = [row for row in active if bucket_counts.get(row.bucket, 0) >= support]
        return [row.copy() for row in self._balanced(active, max(0, int(count)))]

    def note_replayed(self, items: Sequence[FiveFailureReplayCase]) -> None:
        for item in items:
            case = self.cases.get(str(item.case_id))
            if case is not None:
                case.replay_count += 1
                self.replayed_count += 1
        self.batch_index += 1

    def note_batch_without_replay(self) -> None:
        self.batch_index += 1

    def mt5_replay_candidates(
        self, *, count: int, include_resolved: bool = False, max_attempts_per_case: int = 0,
    ) -> list[FiveFailureReplayCase]:
        """Return balanced failure anchors that can be replayed by Strategy Tester."""
        self.ensure_compatible()
        limit = max(0, int(max_attempts_per_case or 0))
        candidates = [
            row for row in self.cases.values()
            if bool(str(getattr(row, "market_time", "") or "").strip())
            and (include_resolved or not bool(row.resolved))
            and (limit <= 0 or int(getattr(row, "mt5_replay_count", 0) or 0) < limit)
        ]
        return [row.copy() for row in self._balanced(candidates, max(0, int(count)))]

    def note_mt5_replay(
        self, case_id: str, *, fitness: float = 0.0, net_profit: float = 0.0,
        successful_session: bool = True, replayed_at: str = "",
    ) -> None:
        case = self.cases.get(str(case_id))
        if case is None:
            return
        case.mt5_replay_count = max(0, int(getattr(case, "mt5_replay_count", 0) or 0)) + 1
        if successful_session:
            case.mt5_replay_success_count = max(0, int(
                getattr(case, "mt5_replay_success_count", 0) or 0
            )) + 1
        case.last_mt5_replay_fitness = float(fitness)
        case.last_mt5_replay_net_profit = float(net_profit)
        case.last_mt5_replay_at = str(replayed_at or "")
        # Keep the generic replay counter meaningful for dashboards while
        # distinguishing MT5 attempts with the dedicated counter above.
        case.replay_count += 1
        self.replayed_count += 1
        self.batch_index += 1

    def stats(self) -> dict[str, object]:
        unresolved = [row for row in self.cases.values() if not bool(row.resolved)]
        finite = [
            float(row.last_failed_action_probability) for row in self.cases.values()
            if np.isfinite(float(row.last_failed_action_probability))
        ]
        return {
            "schema_version": int(self.schema_version),
            "cases": int(len(self.cases)),
            "unresolved": int(len(unresolved)),
            "resolved": int(len(self.cases) - len(unresolved)),
            "batch_index": int(self.batch_index),
            "admitted_count": int(self.admitted_count),
            "replayed_count": int(self.replayed_count),
            "resolved_count": int(self.resolved_count),
            "reopened_count": int(self.reopened_count),
            "discarded_count": int(self.discarded_count),
            "mt5_replay_count": int(sum(int(getattr(row, "mt5_replay_count", 0) or 0) for row in self.cases.values())),
            "mt5_replay_success_count": int(sum(int(getattr(row, "mt5_replay_success_count", 0) or 0) for row in self.cases.values())),
            "mt5_schedulable_cases": int(sum(bool(str(getattr(row, "market_time", "") or "").strip()) for row in self.cases.values())),
            "mean_failed_action_probability": float(np.mean(finite)) if finite else 0.0,
            "max_failed_action_probability": float(np.max(finite)) if finite else 0.0,
        }
