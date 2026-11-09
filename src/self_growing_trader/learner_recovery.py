"""Action-recovery learner.

The learner deliberately separates *evidence about an action* from a global
penalty on entry or HOLD. It continuously creates causal, no-money shadow
labels for market BUY/SELL while the live policy is flat, while final broker
lifecycle evidence remains scoped to the exact entry action that created the
position. A SELL_STOP loss must not be silently merged into SELL_LIMIT or
MARKET_SELL evidence.

It is intentionally conservative:
- a loss never adds a global HOLD bias;
- a negative entry record does not lower a different entry action;
- no synthetic pending-order fill is invented;
- replacement can choose only an already legal candidate emitted by M1;
- shadow labels update market-direction evidence, not broker execution heads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# V12 fixes the second half of the lifecycle accounting bug: intraday reset
# used to erase lifecycle dedup markers before MT5 settled the final ledger.
# V10/V11 evidence is retained with a conservative discount because it still
# contains useful direction/context information, but its support was inflated
# by the duplicate close path and must not retain full confidence.
LEARNER_RECOVERY_SCHEMA_VERSION = 12
MARKET_BUY = 1
MARKET_SELL = 2
ENTRY_ACTIONS = frozenset({1, 2, 3, 4, 5, 6})
LONG_ACTIONS = frozenset({1, 3, 5})
SHORT_ACTIONS = frozenset({2, 4, 6})


def _finite_vector(value: object, size: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = np.zeros(0, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    if size is None:
        return array
    out = np.zeros(max(1, int(size)), dtype=float)
    out[: min(out.size, array.size)] = array[:out.size]
    return out


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    count = min(left.size, right.size)
    if count <= 0:
        return 0.0
    left = left[:count]
    right = right[:count]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 1.0 if float(np.linalg.norm(left) + np.linalg.norm(right)) <= 1e-12 else 0.0
    return float(np.clip(float(left @ right) / denom, -1.0, 1.0))


def action_direction(action: int) -> int:
    action = int(action)
    if action in LONG_ACTIONS:
        return 1
    if action in SHORT_ACTIONS:
        return -1
    return 0


def primary_action(action_vector: object, *, action_count: int = 12) -> int:
    vector = _finite_vector(action_vector)
    if vector.size <= 0:
        return 0
    count = max(1, min(int(action_count), vector.size))
    return int(np.argmax(vector[:count]))


def trace_context(trace: object, *, fallback_size: int = 46) -> np.ndarray:
    context = _finite_vector(getattr(trace, "canonical_context_key", np.zeros(0)))
    # Legacy / bridge traces may carry an allocated but all-zero canonical key.
    # Treating every such trace as the same context would recreate a global
    # action penalty.  Fall back to meaningful raw-market/account/execution
    # partitions unless the canonical key contains actual state.
    if context.size and float(np.linalg.norm(context)) > 1e-10:
        return context
    raw = _finite_vector(getattr(trace, "raw_obs", np.zeros(0)))
    if raw.size:
        # Preserve broad market/account/execution partitions without relying on
        # a raw ticket/order identifier.  This is a safe fallback for legacy
        # traces that predate ContextFrame.
        parts = [raw[:34], raw[74:82], raw[108:119]]
        return np.concatenate([part for part in parts if part.size])
    return np.zeros(max(1, int(fallback_size)), dtype=float)


@dataclass
class ActionEvidenceCell:
    """Bounded sufficient statistics for one action in one context cluster."""

    prototype: np.ndarray
    action: int
    support: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    shadow_support: float = 0.0
    real_support: float = 0.0
    last_sequence: int = 0
    last_trace_id: str = ""

    def ensure_compatible(self, context_size: int) -> None:
        self.prototype = _finite_vector(self.prototype, context_size)
        self.action = int(self.action)
        self.support = float(max(0.0, self.support))
        self.mean = float(np.clip(self.mean, -5.0, 5.0)) if np.isfinite(self.mean) else 0.0
        self.m2 = float(max(0.0, self.m2)) if np.isfinite(self.m2) else 0.0
        self.shadow_support = float(max(0.0, self.shadow_support))
        self.real_support = float(max(0.0, self.real_support))
        self.last_sequence = max(0, int(self.last_sequence or 0))
        self.last_trace_id = str(self.last_trace_id or "")

    @property
    def variance(self) -> float:
        if self.support <= 1.0:
            return 0.50
        return float(np.clip(self.m2 / max(self.support - 1.0, 1e-9), 1e-4, 25.0))

    def update(
        self,
        context: np.ndarray,
        outcome: float,
        *,
        weight: float,
        sequence: int,
        trace_id: str,
        is_real: bool,
    ) -> None:
        """Weighted Welford update with a slowly moving context prototype."""
        self.ensure_compatible(context.size)
        weight = float(np.clip(weight, 0.01, 4.0))
        value = float(np.clip(outcome, -5.0, 5.0))
        old_support = float(self.support)
        new_support = old_support + weight
        delta = value - self.mean
        self.mean += (weight / max(new_support, 1e-9)) * delta
        self.m2 = float(max(0.0, self.m2 + weight * delta * (value - self.mean)))
        # Keep a current local prototype but do not make one market bar erase
        # an older cluster.  Actual lifecycle evidence can move it slightly
        # more than a shadow observation.
        # V9 cold-start: increased shadow context rate so prototypes diverge
        # from zero fast enough for cosine similarity matching.
        context_rate = 0.035 if is_real else 0.15
        self.prototype = np.nan_to_num(
            (1.0 - context_rate) * self.prototype + context_rate * _finite_vector(context, context.size),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        self.support = new_support
        if is_real:
            self.real_support += weight
        else:
            self.shadow_support += weight
        self.last_sequence = max(int(sequence), int(self.last_sequence))
        self.last_trace_id = str(trace_id)

    def snapshot(self, similarity: float, *, min_support: float, lcb_z: float) -> dict[str, float]:
        support = float(max(0.0, self.support))
        stderr = float(np.sqrt(self.variance / max(support, 1.0)))
        lcb = float(self.mean - float(lcb_z) * stderr)
        confidence = float(np.clip(similarity, 0.0, 1.0)) * float(np.clip(support / max(min_support, 1e-9), 0.0, 1.0))
        return {
            "action": float(self.action),
            "similarity": float(np.clip(similarity, -1.0, 1.0)),
            "support": support,
            "shadow_support": float(self.shadow_support),
            "real_support": float(self.real_support),
            "mean_net_r": float(self.mean),
            "variance": float(self.variance),
            "stderr": stderr,
            "lcb": lcb,
            "confidence": confidence,
            "last_sequence": float(self.last_sequence),
        }


@dataclass
class ActionRecoveryLearner:
    """Context-local replacement evidence with shadow learning during HOLD.

    ``cells`` is deliberately action-indexed.  The system can decide that SELL
    is poor in a context, but it never writes that conclusion into a global
    HOLD or entry penalty.  A live replacement is permitted only when the
    positive action itself has enough local evidence.
    """

    context_size: int = 46
    merge_similarity: float = 0.90
    min_replacement_support: float = 3.0
    min_real_support_for_strong: float = 1.0
    min_replacement_lcb: float = 0.08
    negative_lcb_threshold: float = -0.04
    lcb_z: float = 1.28
    shadow_weight: float = 0.50
    real_weight: float = 1.0
    max_cells_per_action: int = 128
    max_seen_traces: int = 8192
    negative_veto_enabled: bool = False
    negative_veto_min_real_support: float = 3.0
    negative_veto_lcb: float = -0.04
    schema_version: int = LEARNER_RECOVERY_SCHEMA_VERSION
    cells: dict[int, list[ActionEvidenceCell]] = field(default_factory=dict)
    seen_shadow_trace_ids: set[str] = field(default_factory=set, repr=False)
    seen_lifecycle_signatures: set[str] = field(default_factory=set, repr=False)
    seen_lifecycle_ids: set[str] = field(default_factory=set, repr=False)
    decision_count: int = 0
    shadow_updates: int = 0
    lifecycle_updates: int = 0
    last_summary: dict[str, object] = field(default_factory=dict)
    _cells_consolidated_similarity: float = field(default=-1.0, repr=False)

    def ensure_compatible(self, *, context_size: int | None = None) -> None:
        # ``pickle`` can expose a newly added dataclass default as a class
        # attribute even when the field is absent from an old instance. Read
        # the instance dictionary so legacy artifacts are actually detected.
        stored_schema = int(dict(getattr(self, "__dict__", {}) or {}).get("schema_version", 0) or 0)
        if stored_schema != int(LEARNER_RECOVERY_SCHEMA_VERSION):
            # V10/V11 mixed shadow samples and duplicate real lifecycle
            # samples. Keep the learned mean/prototype, but halve sufficient
            # statistics so support/confidence cannot claim twice the evidence.
            # Unknown pre-V10 layouts are too ambiguous and are reset.
            if stored_schema in {10, 11}:
                for raw_cells in dict(getattr(self, "cells", {}) or {}).values():
                    for cell in list(raw_cells or []):
                        if isinstance(cell, ActionEvidenceCell):
                            cell.support *= 0.50
                            cell.shadow_support *= 0.50
                            cell.real_support *= 0.50
                            cell.m2 *= 0.50
                self.shadow_updates = int(round(float(getattr(self, "shadow_updates", 0) or 0) * 0.50))
                self.lifecycle_updates = int(round(float(getattr(self, "lifecycle_updates", 0) or 0) * 0.50))
            else:
                self.cells = {}
                self.shadow_updates = 0
                self.lifecycle_updates = 0
            self.seen_shadow_trace_ids = set()
            self.seen_lifecycle_signatures = set()
            self.seen_lifecycle_ids = set()
            self.decision_count = 0
            self.last_summary = {}
            self._cells_consolidated_similarity = -1.0
            self.schema_version = int(LEARNER_RECOVERY_SCHEMA_VERSION)
        if context_size is not None:
            self.context_size = max(8, int(context_size))
        self.context_size = max(8, int(getattr(self, "context_size", 46) or 46))
        self.merge_similarity = float(np.clip(float(getattr(self, "merge_similarity", 0.90) or 0.90), 0.50, 0.999))
        self.min_replacement_support = float(np.clip(float(getattr(self, "min_replacement_support", 3.0) or 3.0), 1.0, 256.0))
        self.min_real_support_for_strong = float(np.clip(float(getattr(self, "min_real_support_for_strong", 1.0) or 1.0), 0.0, 64.0))
        self.min_replacement_lcb = float(np.clip(float(getattr(self, "min_replacement_lcb", 0.08) or 0.08), -1.0, 2.0))
        raw_negative_lcb_threshold = getattr(self, "negative_lcb_threshold", -0.04)
        if raw_negative_lcb_threshold is None:
            raw_negative_lcb_threshold = -0.04
        self.negative_lcb_threshold = float(np.clip(float(raw_negative_lcb_threshold), -2.0, 0.0))
        self.lcb_z = float(np.clip(float(getattr(self, "lcb_z", 1.28) or 1.28), 0.0, 3.0))
        self.shadow_weight = float(np.clip(float(getattr(self, "shadow_weight", 0.50) or 0.50), 0.01, 1.0))
        self.real_weight = float(np.clip(float(getattr(self, "real_weight", 1.0) or 1.0), 0.10, 4.0))
        self.max_cells_per_action = max(4, int(getattr(self, "max_cells_per_action", 128) or 128))
        self.max_seen_traces = max(128, int(getattr(self, "max_seen_traces", 8192) or 8192))
        self.negative_veto_enabled = bool(getattr(self, "negative_veto_enabled", False))
        self.negative_veto_min_real_support = float(np.clip(
            float(getattr(self, "negative_veto_min_real_support", 3.0) or 3.0), 1.0, 256.0,
        ))
        raw_negative_veto_lcb = getattr(self, "negative_veto_lcb", -0.04)
        if raw_negative_veto_lcb is None:
            raw_negative_veto_lcb = -0.04
        self.negative_veto_lcb = float(np.clip(float(raw_negative_veto_lcb), -2.0, 0.0))
        clean: dict[int, list[ActionEvidenceCell]] = {}
        for raw_action, raw_cells in dict(getattr(self, "cells", {}) or {}).items():
            try:
                action = int(raw_action)
            except (TypeError, ValueError):
                continue
            parsed: list[ActionEvidenceCell] = []
            for raw_cell in list(raw_cells or []):
                if isinstance(raw_cell, ActionEvidenceCell):
                    cell = raw_cell
                elif isinstance(raw_cell, Mapping):
                    cell = ActionEvidenceCell(
                        prototype=_finite_vector(raw_cell.get("prototype", np.zeros(self.context_size)), self.context_size),
                        action=action,
                        support=float(raw_cell.get("support", 0.0) or 0.0),
                        mean=float(raw_cell.get("mean", 0.0) or 0.0),
                        m2=float(raw_cell.get("m2", 0.0) or 0.0),
                        shadow_support=float(raw_cell.get("shadow_support", 0.0) or 0.0),
                        real_support=float(raw_cell.get("real_support", 0.0) or 0.0),
                        last_sequence=int(raw_cell.get("last_sequence", 0) or 0),
                        last_trace_id=str(raw_cell.get("last_trace_id", "") or ""),
                    )
                else:
                    continue
                cell.ensure_compatible(self.context_size)
                parsed.append(cell)
            if parsed:
                clean[action] = parsed[-self.max_cells_per_action:]
        self.cells = clean
        if abs(float(getattr(self, "_cells_consolidated_similarity", -1.0)) - self.merge_similarity) > 1e-9:
            self._consolidate_cells()
            self._cells_consolidated_similarity = float(self.merge_similarity)
        self.seen_shadow_trace_ids = set(str(item) for item in list(getattr(self, "seen_shadow_trace_ids", set()) or set())[-self.max_seen_traces:])
        self.seen_lifecycle_signatures = set(str(item) for item in list(getattr(self, "seen_lifecycle_signatures", set()) or set())[-self.max_seen_traces:])
        self.seen_lifecycle_ids = set(str(item) for item in list(getattr(self, "seen_lifecycle_ids", set()) or set())[-self.max_seen_traces:])
        self.schema_version = int(LEARNER_RECOVERY_SCHEMA_VERSION)
        self.decision_count = max(0, int(getattr(self, "decision_count", 0) or 0))
        self.shadow_updates = max(0, int(getattr(self, "shadow_updates", 0) or 0))
        self.lifecycle_updates = max(0, int(getattr(self, "lifecycle_updates", 0) or 0))
        self.last_summary = dict(getattr(self, "last_summary", {}) or {})

    def copy(self) -> "ActionRecoveryLearner":
        self.ensure_compatible()
        copied = ActionRecoveryLearner(
            context_size=self.context_size,
            merge_similarity=self.merge_similarity,
            min_replacement_support=self.min_replacement_support,
            min_real_support_for_strong=self.min_real_support_for_strong,
            min_replacement_lcb=self.min_replacement_lcb,
            negative_lcb_threshold=self.negative_lcb_threshold,
            lcb_z=self.lcb_z,
            shadow_weight=self.shadow_weight,
            real_weight=self.real_weight,
            max_cells_per_action=self.max_cells_per_action,
            max_seen_traces=self.max_seen_traces,
            negative_veto_enabled=self.negative_veto_enabled,
            negative_veto_min_real_support=self.negative_veto_min_real_support,
            negative_veto_lcb=self.negative_veto_lcb,
            schema_version=LEARNER_RECOVERY_SCHEMA_VERSION,
            cells={
                int(action): [
                    ActionEvidenceCell(
                        prototype=cell.prototype.copy(),
                        action=int(cell.action),
                        support=float(cell.support),
                        mean=float(cell.mean),
                        m2=float(cell.m2),
                        shadow_support=float(cell.shadow_support),
                        real_support=float(cell.real_support),
                        last_sequence=int(cell.last_sequence),
                        last_trace_id=str(cell.last_trace_id),
                    )
                    for cell in items
                ]
                for action, items in self.cells.items()
            },
            seen_shadow_trace_ids=set(self.seen_shadow_trace_ids),
            seen_lifecycle_signatures=set(self.seen_lifecycle_signatures),
            seen_lifecycle_ids=set(self.seen_lifecycle_ids),
            decision_count=self.decision_count,
            shadow_updates=self.shadow_updates,
            lifecycle_updates=self.lifecycle_updates,
            last_summary=dict(self.last_summary),
        )
        copied.ensure_compatible()
        return copied

    def reset_runtime_state(self) -> None:
        """Forget idempotency markers, retain accumulated evidence."""
        self.ensure_compatible()
        self.seen_shadow_trace_ids.clear()
        self.seen_lifecycle_signatures.clear()
        self.seen_lifecycle_ids.clear()
        self.last_summary = {}

    def reset_intraday_state(self) -> None:
        """Reset per-day shadow state without losing final-ledger dedup keys.

        MT5 can report a position close and settle its final execution ledger
        after a trading-day boundary. Lifecycle markers must therefore live for
        the whole trial; only shadow transition idempotency is intraday state.
        """
        self.ensure_compatible()
        self.seen_shadow_trace_ids.clear()
        self.last_summary = {}

    @staticmethod
    def _merge_cell(target: ActionEvidenceCell, incoming: ActionEvidenceCell) -> None:
        """Merge sufficient statistics without replaying a fake outcome."""
        left_support = float(max(0.0, target.support))
        right_support = float(max(0.0, incoming.support))
        total = left_support + right_support
        if total <= 1e-12:
            return
        delta = float(incoming.mean - target.mean)
        target.m2 = float(max(0.0, target.m2 + incoming.m2 + delta * delta * left_support * right_support / total))
        target.mean = float((target.mean * left_support + incoming.mean * right_support) / total)
        target.support = total
        target.shadow_support = float(target.shadow_support + incoming.shadow_support)
        target.real_support = float(target.real_support + incoming.real_support)
        target.prototype = np.nan_to_num(
            (target.prototype * left_support + incoming.prototype * right_support) / total,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        if int(incoming.last_sequence) >= int(target.last_sequence):
            target.last_sequence = int(incoming.last_sequence)
            target.last_trace_id = str(incoming.last_trace_id)

    def _consolidate_cells(self) -> None:
        """Reduce sparse fragmentation after loading older checkpoints."""
        for action, raw_cells in list(self.cells.items()):
            merged: list[ActionEvidenceCell] = []
            ordered = sorted(raw_cells, key=lambda cell: (float(cell.support), float(cell.real_support)), reverse=True)
            for cell in ordered:
                best: ActionEvidenceCell | None = None
                best_similarity = -1.0
                for candidate in merged:
                    similarity = _cosine(cell.prototype, candidate.prototype)
                    if similarity > best_similarity:
                        best = candidate
                        best_similarity = similarity
                if best is not None and best_similarity >= self.merge_similarity:
                    self._merge_cell(best, cell)
                else:
                    merged.append(cell)
            self.cells[action] = merged[-self.max_cells_per_action:]

    def _prune_seen(self, target: set[str]) -> None:
        if len(target) <= self.max_seen_traces:
            return
        # Sets are not ordered.  The markers are only duplicate guards, so
        # retaining an arbitrary bounded subset is safe and avoids a growing
        # session-state object.
        kept = sorted(target)[-self.max_seen_traces:]
        target.clear()
        target.update(kept)

    def _find_or_create(self, action: int, context: np.ndarray) -> tuple[ActionEvidenceCell, float, bool]:
        self.ensure_compatible(context_size=context.size)
        items = self.cells.setdefault(int(action), [])
        best: ActionEvidenceCell | None = None
        best_similarity = -1.0
        for cell in items:
            similarity = _cosine(context, cell.prototype)
            if similarity > best_similarity:
                best = cell
                best_similarity = similarity
        if best is not None and best_similarity >= self.merge_similarity:
            return best, float(best_similarity), False
        if len(items) >= self.max_cells_per_action:
            # Evict weak, old context before discarding an evidence-rich cell.
            items.sort(key=lambda cell: (float(cell.support) + 2.0 * float(cell.real_support), int(cell.last_sequence)))
            items.pop(0)
        created = ActionEvidenceCell(prototype=context.copy(), action=int(action))
        items.append(created)
        self._cells_consolidated_similarity = -1.0
        return created, 1.0, True

    def update(
        self,
        action: int,
        context: object,
        outcome: float,
        *,
        weight: float,
        sequence: int,
        trace_id: str,
        is_real: bool,
    ) -> dict[str, object]:
        self.ensure_compatible()
        action = int(action)
        if action not in ENTRY_ACTIONS:
            return {"updated": False, "reason": "unsupported_action", "action": action}
        vector = _finite_vector(context, self.context_size)
        cell, similarity, created = self._find_or_create(action, vector)
        cell.update(
            vector,
            float(outcome),
            weight=float(weight),
            sequence=int(sequence),
            trace_id=str(trace_id),
            is_real=bool(is_real),
        )
        if is_real:
            self.lifecycle_updates += 1
        else:
            self.shadow_updates += 1
        return {
            "updated": True,
            "action": action,
            "created_cell": bool(created),
            "matched_similarity": float(similarity),
            "cell": cell.snapshot(max(float(similarity), 0.0), min_support=self.min_replacement_support, lcb_z=self.lcb_z),
        }

    def query(self, context: object, *, allowed_actions: Iterable[int] | None = None) -> dict[int, dict[str, float]]:
        self.ensure_compatible()
        vector = _finite_vector(context, self.context_size)
        allowed = {int(item) for item in allowed_actions} if allowed_actions is not None else set(ENTRY_ACTIONS)
        actions = sorted(ENTRY_ACTIONS & allowed)
        result: dict[int, dict[str, float]] = {}
        for action in actions:
            best: ActionEvidenceCell | None = None
            best_similarity = -1.0
            for cell in self.cells.get(action, []):
                similarity = _cosine(vector, cell.prototype)
                if similarity > best_similarity:
                    best = cell
                    best_similarity = similarity
            if best is None or best_similarity < self.merge_similarity:
                result[action] = {
                    "action": float(action), "available": 0.0, "similarity": max(0.0, float(best_similarity)),
                    "support": 0.0, "shadow_support": 0.0, "real_support": 0.0,
                    "mean_net_r": 0.0, "variance": 0.50, "stderr": 1.0,
                    "lcb": -1.0, "confidence": 0.0, "last_sequence": 0.0,
                }
            else:
                snap = best.snapshot(best_similarity, min_support=self.min_replacement_support, lcb_z=self.lcb_z)
                snap["available"] = 1.0
                result[action] = snap
        return result

    def _eligible_positive(self, item: Mapping[str, float]) -> bool:
        return bool(
            float(item.get("available", 0.0)) > 0.5
            and float(item.get("support", 0.0)) >= self.min_replacement_support
            and float(item.get("lcb", -1.0)) >= self.min_replacement_lcb
            and float(item.get("confidence", 0.0)) >= 0.45
        )

    def _eligible_shadow_bootstrap(self, item: Mapping[str, float]) -> bool:
        """V9 cold-start: allow replacement from shadow evidence alone.

        When real_support is 0 and shadow support is sufficient, use mean
        direction instead of LCB to bootstrap BUY/SELL evidence.
        Requires 2x the normal support to avoid premature replacement.
        """
        real = float(item.get("real_support", 0.0))
        shadow = float(item.get("shadow_support", 0.0))
        if real > 0.0:
            return False
        if shadow < 2.0 * self.min_replacement_support:
            return False
        mean = float(item.get("mean_net_r", 0.0))
        return mean > 0.001

    def rerank(
        self,
        trace: object,
        *,
        allowed_actions: Iterable[int] | None = None,
    ) -> dict[str, object]:
        """Find an evidence-backed replacement without applying a penalty.

        The base graph's candidate scores remain intact. A replacement only
        swaps to an already legal entry candidate when positive local evidence
        is sufficient. The selected action is never hard-blocked merely because
        the original action has a loss history.
        """
        self.ensure_compatible()
        self.decision_count += 1
        candidates = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
        scores = _finite_vector(getattr(trace, "candidate_scores", np.zeros(0)))
        base_idx = int(getattr(trace, "selected_idx", 0) or 0)
        base_action = primary_action(getattr(trace, "selected_action", np.zeros(0)))
        allowed = {int(item) for item in allowed_actions} if allowed_actions is not None else set()
        if not allowed:
            mask = tuple(bool(item) for item in tuple(getattr(trace, "allowed_action_mask", ()) or ()))
            allowed = {idx for idx, value in enumerate(mask) if value}
        if not allowed:
            allowed = set(ENTRY_ACTIONS)
        context = trace_context(trace, fallback_size=self.context_size)
        values = self.query(context, allowed_actions=allowed)
        positive = [action for action, item in values.items() if self._eligible_positive(item)]
        # V9 cold-start: also check shadow-only bootstrap eligibility
        if not positive:
            shadow_bootstrap = [action for action, item in values.items() if self._eligible_shadow_bootstrap(item)]
            if shadow_bootstrap:
                shadow_bootstrap.sort(key=lambda action: float(values[action].get("mean_net_r", 0.0)), reverse=True)
                positive = shadow_bootstrap
        positive.sort(key=lambda action: (float(values[action]["lcb"]), float(values[action]["confidence"])), reverse=True)
        best_action = positive[0] if positive else None

        candidate_primary: list[int] = []
        if candidates.ndim == 2 and candidates.shape[0] > 0:
            candidate_primary = [primary_action(row) for row in candidates]
        candidate_index = None
        if best_action is not None and candidate_primary:
            matching = [idx for idx, action in enumerate(candidate_primary) if int(action) == int(best_action)]
            if matching:
                candidate_index = max(matching, key=lambda idx: float(scores[idx]) if idx < scores.size else -1e9)
            elif best_action in {MARKET_BUY, MARKET_SELL} and self._eligible_shadow_bootstrap(values.get(best_action, {})):
                # V9 cold-start: no neural candidate proposes the replacement
                # action.  Create a synthetic candidate so the replacement trace
                # can select it.  The candidate action vector has +1 at the
                # replacement action index and -1 elsewhere.
                width = int(candidates.shape[1]) if candidates.ndim == 2 else 0
                if width > 0:
                    synth = np.full(width, -1.0, dtype=float)
                    synth[int(best_action)] = 1.0
                    candidates = np.vstack([candidates, synth.reshape(1, -1)])
                    candidate_primary.append(int(best_action))
                    # Score the synthetic candidate slightly above the worst
                    # existing candidate so it is selectable.
                    worst_score = float(np.min(scores)) if scores.size else -1.0
                    scores = np.append(scores, worst_score + 0.01)
                    candidate_index = candidates.shape[0] - 1

        base_value = values.get(base_action, {})
        base_support = float(base_value.get("support", 0.0))
        base_real_support = float(base_value.get("real_support", 0.0))
        base_lcb = float(base_value.get("lcb", 0.0))
        negative_support_threshold = float(self.min_replacement_support)
        if self.negative_veto_enabled:
            negative_support_threshold = min(
                negative_support_threshold,
                float(self.negative_veto_min_real_support),
            )
        base_negative = bool(
            base_action in ENTRY_ACTIONS
            and (
                (
                    base_support >= negative_support_threshold
                    and base_lcb <= self.negative_lcb_threshold
                )
                or (
                    base_real_support >= 1.0
                    and base_support >= max(1.5, 0.5 * self.min_replacement_support)
                    and base_lcb <= min(-0.75, self.negative_lcb_threshold)
                )
            )
        )
        soft_action_penalty = 0.0
        if base_negative:
            # Keep entry available, but let repeated local losses lower the
            # candidate's score. This is deliberately a soft penalty: it never
            # edits the context action mask and never creates a global HOLD bias.
            soft_action_penalty = float(np.clip(
                0.35 * max(0.0, -base_lcb)
                + 0.15 * max(0.0, -float(base_value.get("mean_net_r", 0.0))),
                0.05,
                0.75,
            ))
        changed = bool(candidate_index is not None and best_action is not None and int(best_action) != int(base_action) and (base_action == 0 or base_negative))
        veto_to_hold = bool(
            self.negative_veto_enabled
            and base_negative
            and not changed
            and 0 in allowed
            and base_real_support >= self.negative_veto_min_real_support
            and base_lcb <= self.negative_veto_lcb
        )
        hold_reason = ""
        if base_action == 0 and not changed:
            if not (ENTRY_ACTIONS & allowed):
                hold_reason = "HOLD_NO_LEGAL_ENTRY"
            elif any(float(item.get("support", 0.0)) > 0.0 for item in values.values()):
                if all(float(item.get("lcb", -1.0)) <= 0.0 for item in values.values()):
                    hold_reason = "HOLD_ALL_ENTRY_EVIDENCE_NONPOSITIVE"
                else:
                    hold_reason = "HOLD_WAITING_FOR_LOCAL_SUPPORT"
            else:
                hold_reason = "HOLD_SHADOW_EVIDENCE_BUILDING"
        elif base_action == 0 and changed:
            hold_reason = "REPLACED_HOLD_WITH_VERIFIED_ACTION"
        elif veto_to_hold:
            hold_reason = "HOLD_LOCAL_REAL_NEGATIVE_VETO"
        elif base_negative and not changed:
            hold_reason = "NO_VERIFIED_ENTRY_REPLACEMENT"
        else:
            hold_reason = "NOT_HOLD"


        payload = {
            "learner_recovery_schema_version": LEARNER_RECOVERY_SCHEMA_VERSION,
            "base_action": int(base_action),
            "base_candidate_index": int(base_idx),
            "replacement_action": None if best_action is None else int(best_action),
            "replacement_candidate_index": None if candidate_index is None else int(candidate_index),
            "replacement_applied": bool(changed),
            "negative_veto_to_hold": bool(veto_to_hold),
            "negative_veto_min_real_support": float(self.negative_veto_min_real_support),
            "negative_veto_lcb": float(self.negative_veto_lcb),
            "negative_support_threshold": float(negative_support_threshold),
            "base_action_local_negative": bool(base_negative),
            "soft_action_penalty": float(soft_action_penalty),
            "hold_reason": str(hold_reason),
            "action_values": {str(action): {key: float(value) for key, value in item.items()} for action, item in values.items()},
            "shadow_learning_active": True,
            "shadow_market_actions": [MARKET_BUY, MARKET_SELL],
            "real_lifecycle_entry_actions": sorted(ENTRY_ACTIONS),
        }
        # V9 cold-start: if rerank appended a synthetic candidate, pass the
        # modified arrays so _v9_replacement_trace can select it.
        if candidates.ndim == 2 and candidates.shape[0] > 5:
            payload["_replacement_candidates"] = candidates.copy()
            payload["_replacement_scores"] = scores.copy()
        self.last_summary = dict(payload)
        return payload

    def observe_shadow_transition(
        self,
        trace: object,
        next_observation: object,
        *,
        sequence: int = 0,
    ) -> dict[str, object]:
        """Learn long/short counterfactual direction without opening a trade."""
        self.ensure_compatible()
        trace_id = str(getattr(trace, "trace_id", "") or "")
        signature = trace_id or f"shadow:{int(sequence)}:{self.shadow_updates}"
        if signature in self.seen_shadow_trace_ids:
            return {"updated": False, "reason": "duplicate_shadow_transition", "trace_id": trace_id}
        self.seen_shadow_trace_ids.add(signature)
        self._prune_seen(self.seen_shadow_trace_ids)
        context = trace_context(trace, fallback_size=self.context_size)
        future = _finite_vector(next_observation, 119)
        # Feature 0 is the canonical next closed-bar return in ATR units;
        # feature 15 is the canonical spread proxy used by the world model.
        # Shadow learning uses a deliberately small estimated cost and never
        # claims a broker fill, TP, SL, MAE or MFE outcome.
        forward_return = float(np.clip(future[0] if future.size else 0.0, -3.0, 3.0))
        spread_proxy = float(abs(future[15])) if future.size > 15 else 0.0
        estimated_cost = float(np.clip(0.02 * spread_proxy, 0.0, 0.15))
        long_outcome = float(np.clip(forward_return - estimated_cost, -3.0, 3.0))
        short_outcome = float(np.clip(-forward_return - estimated_cost, -3.0, 3.0))
        updates = {
            "buy": self.update(MARKET_BUY, context, long_outcome, weight=self.shadow_weight, sequence=int(sequence), trace_id=trace_id, is_real=False),
            "sell": self.update(MARKET_SELL, context, short_outcome, weight=self.shadow_weight, sequence=int(sequence), trace_id=trace_id, is_real=False),
        }
        result = {
            "updated": True,
            "trace_id": trace_id,
            "sequence": int(sequence),
            "forward_return_atr": forward_return,
            "estimated_cost_r": estimated_cost,
            "shadow_buy_net_r": long_outcome,
            "shadow_sell_net_r": short_outcome,
            "updates": updates,
        }
        self.last_summary = {"last_shadow": result, **dict(self.last_summary)}
        return result

    def observe_lifecycle(
        self,
        trace: object,
        realized_r: float,
        *,
        close_reason: str = "",
        sequence: int = 0,
        signature: str = "",
        override_action: int | None = None,
        lifecycle_identity: str = "",
    ) -> dict[str, object]:
        """Add actual lifecycle evidence to the exact executed entry action."""
        self.ensure_compatible()
        details = dict(getattr(trace, "action_details", {}) or {})
        execution_details = dict(getattr(trace, "execution_details", {}) or {})
        for key, value in execution_details.items():
            details.setdefault(str(key), value)
        action = int(override_action) if override_action is not None and override_action > 0 else int(details.get("emitted_action", primary_action(getattr(trace, "selected_action", np.zeros(0)))) or 0)
        if action not in ENTRY_ACTIONS:
            return {"updated": False, "reason": "non_directional_lifecycle", "action": action}
        trace_id = str(getattr(trace, "trace_id", "") or "")
        event_signature = signature or f"lifecycle:{trace_id}:{action}:{round(float(realized_r), 8)}:{close_reason}:{int(sequence)}"
        if event_signature in self.seen_lifecycle_signatures:
            return {"updated": False, "reason": "duplicate_lifecycle", "trace_id": trace_id, "action": action}
        details_identity = str(
            lifecycle_identity
            or details.get("position_ticket", "")
            or details.get("position_owner_ticket", "")
            or details.get("deal_ticket", "")
            or details.get("order_ticket", "")
            or ""
        ).strip()
        canonical_identity = f"position:{details_identity}" if details_identity else ""
        if canonical_identity and canonical_identity in self.seen_lifecycle_ids:
            return {
                "updated": False,
                "reason": "duplicate_lifecycle_identity",
                "trace_id": trace_id,
                "action": action,
                "lifecycle_identity": details_identity,
            }
        self.seen_lifecycle_signatures.add(event_signature)
        if canonical_identity:
            self.seen_lifecycle_ids.add(canonical_identity)
        self._prune_seen(self.seen_lifecycle_signatures)
        self._prune_seen(self.seen_lifecycle_ids)
        # V9 timing: SL ngay = entry sai timing, penalty nặng
        duration = getattr(trace, "duration_bars", 0)
        if close_reason == "SL" and duration is not None and float(duration) < 4.0:
            realized_r = float(realized_r) * 1.5  # penalty 1.5x cho SL sớm
        result = self.update(
            action,
            trace_context(trace, fallback_size=self.context_size),
            float(realized_r),
            weight=self.real_weight,
            sequence=int(sequence),
            trace_id=trace_id,
            is_real=True,
        )
        result.update({
            "close_reason": str(close_reason),
            "recorded_action": int(action),
            "original_action": int(action),
            "global_hold_penalty": 0.0,
            "negative_action_suppression": 0.0,
            "lifecycle_identity": details_identity,
        })
        self.last_summary = {"last_lifecycle": result, **dict(self.last_summary)}
        return result

    def summary(self) -> dict[str, object]:
        self.ensure_compatible()
        counts = {
            str(action): {
                "cells": len(items),
                "support": float(sum(float(cell.support) for cell in items)),
                "shadow_support": float(sum(float(cell.shadow_support) for cell in items)),
                "real_support": float(sum(float(cell.real_support) for cell in items)),
            }
            for action, items in self.cells.items()
        }
        return {
            "schema_version": LEARNER_RECOVERY_SCHEMA_VERSION,
            "decision_count": int(self.decision_count),
            "shadow_updates": int(self.shadow_updates),
            "lifecycle_updates": int(self.lifecycle_updates),
            "schema_version": LEARNER_RECOVERY_SCHEMA_VERSION,
            "action_cells": counts,
            "last": dict(self.last_summary),
            "replacement_policy": "positive_local_lcb_or_real_negative_local_hold_veto",
            "global_hold_penalty": 0.0,
            "negative_veto_enabled": bool(self.negative_veto_enabled),
            "negative_veto_min_real_support": float(self.negative_veto_min_real_support),
            "negative_veto_lcb": float(self.negative_veto_lcb),
        }
