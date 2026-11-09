"""Persistent, replay-safe evidence retention for Direct-RNN Full-RL.

The store owns no weights and performs no optimization.  It only preserves
deep-copied, already-settled training items and returns deep copies to the
single Full-RL AdamW batch.  This keeps lifecycle provenance immutable while
making useful and hard-negative evidence survive beyond the short FIFO queue.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Mapping


RETENTION_SCHEMA_VERSION = 6
RETENTION_AUDIT_SPLIT_MODULUS = 5


@dataclass
class RetentionEpisode:
    episode_id: str
    bucket: tuple[str, ...]
    payload: dict[str, object]
    admitted_batch: int
    anchor: bool
    behavior_anchor: bool
    hard_negative: bool
    preference_anchor: bool
    audit_only: bool
    replay_count: int = 0


@dataclass
class ExpertRetentionStore:
    """Bounded recent/diverse/anchor indexes over immutable episode payloads."""

    schema_version: int = RETENTION_SCHEMA_VERSION
    recent_capacity: int = 64
    diverse_capacity: int = 128
    anchor_capacity: int = 64
    episodes: dict[str, RetentionEpisode] = field(default_factory=dict, repr=False)
    recent_ids: list[str] = field(default_factory=list, repr=False)
    diverse_ids: list[str] = field(default_factory=list, repr=False)
    anchor_ids: list[str] = field(default_factory=list, repr=False)
    batch_index: int = 0
    admitted_count: int = 0
    replayed_count: int = 0
    discarded_count: int = 0

    def copy(self) -> "ExpertRetentionStore":
        return copy.deepcopy(self)

    def ensure_compatible(
        self,
        *,
        recent_capacity: int | None = None,
        diverse_capacity: int | None = None,
        anchor_capacity: int | None = None,
    ) -> None:
        previous_schema_version = int(getattr(self, "schema_version", 0) or 0)
        self.schema_version = RETENTION_SCHEMA_VERSION
        self.recent_capacity = max(1, int(
            self.recent_capacity if recent_capacity is None else recent_capacity
        ))
        self.diverse_capacity = max(1, int(
            self.diverse_capacity if diverse_capacity is None else diverse_capacity
        ))
        self.anchor_capacity = max(1, int(
            self.anchor_capacity if anchor_capacity is None else anchor_capacity
        ))
        if not isinstance(getattr(self, "episodes", None), dict):
            self.episodes = {}
        for name in ("recent_ids", "diverse_ids", "anchor_ids"):
            values = getattr(self, name, None)
            setattr(self, name, list(values) if isinstance(values, (list, tuple)) else [])
        self.batch_index = max(0, int(getattr(self, "batch_index", 0) or 0))
        self.admitted_count = max(0, int(getattr(self, "admitted_count", 0) or 0))
        self.replayed_count = max(0, int(getattr(self, "replayed_count", 0) or 0))
        self.discarded_count = max(0, int(getattr(self, "discarded_count", 0) or 0))
        valid = {
            str(key): value for key, value in self.episodes.items()
            if isinstance(value, RetentionEpisode)
        }
        for episode in valid.values():
            # Schema-v2 artifacts do not have ``preference_anchor``. Derive it
            # from the immutable target so a resumed model immediately gains
            # the stronger retention contract instead of silently discarding
            # its already-collected HOLD/hard-negative evidence.
            episode.preference_anchor = bool(getattr(
                episode,
                "preference_anchor",
                self._preference_anchor_from_item(episode.payload),
            ))
            episode.anchor = bool(
                getattr(episode, "behavior_anchor", False)
                or getattr(episode, "hard_negative", False)
                or episode.preference_anchor
            )
            migrated_audit_only = self._audit_only_from_item(
                episode.payload, episode.episode_id
            )
            episode.audit_only = bool(
                episode.anchor
                and (
                    migrated_audit_only
                    if previous_schema_version < RETENTION_SCHEMA_VERSION
                    else getattr(episode, "audit_only", migrated_audit_only)
                )
            )
            episode.payload["entry_preference_retention_anchor"] = bool(
                episode.preference_anchor
            )
            episode.payload["retention_audit_only"] = bool(episode.audit_only)
        self.episodes = valid
        self.recent_ids = [item for item in self.recent_ids if item in valid][-self.recent_capacity:]
        self.diverse_ids = [item for item in self.diverse_ids if item in valid]
        self.anchor_ids = [
            item for item in self.anchor_ids
            if item in valid and bool(valid[item].anchor)
        ]
        for episode_id, episode in valid.items():
            if episode.anchor and episode_id not in self.anchor_ids:
                self.anchor_ids.append(episode_id)
        self._trim_diverse()
        self._trim_anchors()
        self._cleanup()

    def discard_incompatible_preference_targets(
        self,
        valid_target_modes: set[str],
    ) -> int:
        """Remove replay labels whose learning contract has been replaced."""
        valid_modes = {str(mode) for mode in valid_target_modes}
        stale = {
            episode_id for episode_id, episode in self.episodes.items()
            if bool(episode.payload.get("entry_preference_only", False))
            and str(episode.payload.get("entry_preference_target_mode", "") or "")
            not in valid_modes
        }
        if not stale:
            return 0
        self.recent_ids = [item for item in self.recent_ids if item not in stale]
        self.diverse_ids = [item for item in self.diverse_ids if item not in stale]
        self.anchor_ids = [item for item in self.anchor_ids if item not in stale]
        for episode_id in stale:
            self.episodes.pop(episode_id, None)
        self.discarded_count += len(stale)
        self._cleanup()
        return len(stale)

    @staticmethod
    def _trace_metadata(item: Mapping[str, object]) -> tuple[str, int, str]:
        trace = item.get("trace")
        details = dict(getattr(trace, "action_details", {}) or {})
        trace_id = str(getattr(trace, "trace_id", "") or "")
        try:
            action = int(details.get(
                "direct_rnn_rl_policy_action_index",
                getattr(trace, "selected_idx", -1),
            ))
        except (TypeError, ValueError):
            action = -1
        context = str(details.get(
            "market_context",
            details.get("context", details.get("decision_policy_context", "unknown")),
        ) or "unknown")
        return trace_id, action, context

    @classmethod
    def _identity(cls, item: Mapping[str, object]) -> str:
        trace_id, action, _context = cls._trace_metadata(item)
        kind = str(item.get("credit_kind", "lifecycle") or "lifecycle")
        scope = str(item.get("entry_preference_sample_scope", "") or "")
        raw = f"{trace_id}|{kind}|{scope}|{action}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _audit_only_from_id(episode_id: str) -> bool:
        """Stable 20% replay holdout; independent of process RNG and resume."""
        try:
            return int(str(episode_id)[:8], 16) % RETENTION_AUDIT_SPLIT_MODULUS == 0
        except (TypeError, ValueError):
            return False

    @classmethod
    def _audit_only_from_item(
        cls,
        item: Mapping[str, object],
        episode_id: str,
    ) -> bool:
        trace = item.get("trace")
        details = dict(getattr(trace, "action_details", {}) or {})
        deterministic_frame = str(details.get("context_frame_id", "") or "")
        trace_id, action, context = cls._trace_metadata(item)
        source_id = deterministic_frame or trace_id or str(episode_id)
        kind = str(item.get("credit_kind", "lifecycle") or "lifecycle")
        scope = str(item.get("entry_preference_sample_scope", "") or "")
        digest = hashlib.sha256(
            f"{source_id}|{kind}|{scope}|{action}|{context}".encode("utf-8")
        ).hexdigest()
        return cls._audit_only_from_id(digest)

    @classmethod
    def _bucket(cls, item: Mapping[str, object]) -> tuple[str, ...]:
        _trace_id, action, context = cls._trace_metadata(item)
        trace = item.get("trace")
        rule_trace = dict(getattr(trace, "tensor_rule_trace", {}) or {})
        rule_suffix: tuple[str, ...] = ()
        if rule_trace and int(rule_trace.get("schema_version", 0) or 0) >= 1:
            try:
                rule_id = int(rule_trace.get("rule_id", -1))
            except (TypeError, ValueError):
                rule_id = -1
            try:
                raw_fold = item.get("tensor_rule_temporal_fold", None)
                if raw_fold is None:
                    frame_id = str(getattr(trace, "context_frame_id", "") or getattr(trace, "trace_id", ""))
                    temporal_fold = int.from_bytes(
                        hashlib.sha256(frame_id.encode("utf-8")).digest()[:2], "little"
                    ) % 3
                else:
                    temporal_fold = int(raw_fold) % 3
            except (TypeError, ValueError):
                temporal_fold = 0
            rule_suffix = (
                f"rule:{rule_id if rule_id >= 0 else 'unknown'}",
                f"fold:{temporal_fold % 3}",
            )
        if bool(item.get("entry_preference_only", False)):
            if bool(item.get("entry_preference_hard_negative", False)):
                evidence = "hard"
            elif bool(item.get("entry_preference_retention_anchor", False)):
                evidence = "control_anchor"
            else:
                evidence = "ordinary"
            return (
                "preference", context, str(action), evidence,
            ) + rule_suffix
        outcome = float(item.get("outcome", 0.0) or 0.0)
        sign = "win" if outcome > 0.05 else "loss" if outcome < -0.05 else "flat"
        return ("lifecycle", context, str(action), sign) + rule_suffix

    @classmethod
    def _preference_anchor_from_item(cls, item: Mapping[str, object]) -> bool:
        """Identify immutable preference controls worth guarding with KL.

        Every hard negative remains an anchor. A neural HOLD becomes an anchor
        only when its delayed target independently agrees strongly enough with
        HOLD. The default threshold mirrors the inference trade threshold, and
        is stored on new items so artifact migration remains deterministic.
        """
        if not bool(item.get("entry_preference_only", False)):
            return False
        if bool(item.get("entry_preference_hard_negative", False)):
            return True
        if bool(item.get("entry_preference_retention_anchor", False)):
            return True
        _trace_id, selected_action, _context = cls._trace_metadata(item)
        if selected_action != 0:
            return False
        try:
            values = [max(0.0, float(value)) for value in item.get(
                "entry_preference_target", ()
            )]
        except (TypeError, ValueError):
            return False
        total = float(sum(values))
        if not values or total <= 0.0:
            return False
        try:
            threshold = float(item.get(
                "entry_preference_retention_hold_threshold", 0.55
            ) or 0.55)
        except (TypeError, ValueError):
            threshold = 0.55
        threshold = min(1.0, max(0.5, threshold))
        return bool(values[0] / total >= threshold)

    def add(self, item: Mapping[str, object], *, anchor_min_r: float) -> str:
        """Admit one settled item once; callers never receive the stored object."""
        episode_id = self._identity(item)
        if episode_id in self.episodes:
            return episode_id
        preference = bool(item.get("entry_preference_only", False))
        hard_negative = bool(item.get("entry_preference_hard_negative", False))
        outcome = float(item.get("outcome", 0.0) or 0.0)
        behavior_anchor = bool(not preference and outcome >= float(anchor_min_r))
        preference_anchor = self._preference_anchor_from_item(item)
        anchor = bool(behavior_anchor or hard_negative or preference_anchor)
        payload = copy.deepcopy(dict(item))
        payload["retention_episode_id"] = episode_id
        payload["entry_preference_retention_anchor"] = bool(preference_anchor)
        audit_only = bool(
            anchor and self._audit_only_from_item(payload, episode_id)
        )
        payload["retention_audit_only"] = audit_only
        self.episodes[episode_id] = RetentionEpisode(
            episode_id=episode_id,
            bucket=self._bucket(payload),
            payload=payload,
            admitted_batch=int(self.batch_index),
            anchor=anchor,
            behavior_anchor=behavior_anchor,
            hard_negative=hard_negative,
            preference_anchor=preference_anchor,
            audit_only=audit_only,
        )
        self.recent_ids.append(episode_id)
        self.diverse_ids.append(episode_id)
        if anchor:
            self.anchor_ids.append(episode_id)
        self.admitted_count += 1
        self.recent_ids = self.recent_ids[-self.recent_capacity:]
        self._trim_diverse()
        self._trim_anchors()
        self._cleanup()
        return episode_id

    def _trim_ids_by_bucket(self, ids: list[str], capacity: int) -> None:
        """Evict the oldest row from the most crowded evidence bucket."""
        while len(ids) > capacity:
            counts: dict[tuple[str, ...], int] = {}
            for episode_id in ids:
                episode = self.episodes.get(episode_id)
                if episode is not None:
                    counts[episode.bucket] = counts.get(episode.bucket, 0) + 1
            if not counts:
                del ids[:-capacity]
                break
            largest = max(counts.values())
            crowded = sorted(bucket for bucket, count in counts.items() if count == largest)
            target = crowded[0]
            remove_at = next(
                index for index, episode_id in enumerate(ids)
                if self.episodes.get(episode_id) is not None
                and self.episodes[episode_id].bucket == target
            )
            ids.pop(remove_at)

    def _trim_diverse(self) -> None:
        self._trim_ids_by_bucket(self.diverse_ids, self.diverse_capacity)

    def _trim_anchors(self) -> None:
        # A stream of HOLD controls must not evict every profitable lifecycle
        # anchor (and vice versa). Bucket-balanced eviction preserves both.
        self._trim_ids_by_bucket(self.anchor_ids, self.anchor_capacity)

    def _cleanup(self) -> None:
        referenced = set(self.recent_ids) | set(self.diverse_ids) | set(self.anchor_ids)
        self.episodes = {
            key: value for key, value in self.episodes.items() if key in referenced
        }

    @staticmethod
    def _balanced_ids(episodes: list[RetentionEpisode], count: int) -> list[str]:
        buckets: dict[tuple[str, ...], list[RetentionEpisode]] = {}
        for episode in episodes:
            buckets.setdefault(episode.bucket, []).append(episode)
        for values in buckets.values():
            values.sort(key=lambda row: (row.replay_count, row.admitted_batch, row.episode_id))
        selected: list[str] = []
        keys = sorted(buckets)
        while keys and len(selected) < count:
            remaining: list[tuple[str, ...]] = []
            for key in keys:
                values = buckets[key]
                if values and len(selected) < count:
                    selected.append(values.pop(0).episode_id)
                if values:
                    remaining.append(key)
            keys = remaining
        return selected

    def sample(
        self,
        count: int,
        *,
        replay_weight: float,
        min_lifecycle_fraction: float = 0.50,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        """Return deterministic balanced copies from batches strictly in the past."""
        count = max(0, int(count))
        if count <= 0:
            return []
        excluded = set(exclude_ids or ())
        ordered_ids: list[str] = []
        # Anchors lead, diverse evidence fills gaps, recent evidence supplies
        # genuinely new regimes.  Deduplication keeps one gradient per item.
        for episode_id in self.anchor_ids + self.diverse_ids + list(reversed(self.recent_ids)):
            if episode_id not in ordered_ids:
                ordered_ids.append(episode_id)
        eligible = [
            self.episodes[episode_id] for episode_id in ordered_ids
            if episode_id in self.episodes
            and episode_id not in excluded
            and self.episodes[episode_id].admitted_batch < self.batch_index
            and not bool(getattr(self.episodes[episode_id], "audit_only", False))
        ]
        lifecycle = [row for row in eligible if not bool(row.payload.get("entry_preference_only", False))]
        preference = [row for row in eligible if bool(row.payload.get("entry_preference_only", False))]
        lifecycle_quota = min(count, max(0, int(round(count * float(min_lifecycle_fraction)))))
        selected_ids = self._balanced_ids(lifecycle, lifecycle_quota)
        selected_ids += self._balanced_ids(
            [row for row in preference if row.episode_id not in selected_ids],
            count - len(selected_ids),
        )
        if len(selected_ids) < count:
            selected_set = set(selected_ids)
            selected_ids += self._balanced_ids(
                [row for row in eligible if row.episode_id not in selected_set],
                count - len(selected_ids),
            )
        result: list[dict[str, object]] = []
        for episode_id in selected_ids[:count]:
            episode = self.episodes[episode_id]
            payload = copy.deepcopy(episode.payload)
            payload["retention_replay"] = True
            payload["retention_anchor"] = bool(episode.anchor)
            payload["retention_behavior_anchor"] = bool(episode.behavior_anchor)
            payload["retention_hard_negative"] = bool(episode.hard_negative)
            payload["retention_preference_anchor"] = bool(episode.preference_anchor)
            payload["retention_source_bucket"] = episode.bucket
            payload["update_scale"] = float(payload.get("update_scale", 1.0) or 1.0) * float(replay_weight)
            result.append(payload)
        return result

    def audit(
        self,
        count: int,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        """Return replay-held-out anchors for acceptance testing only.

        These payloads never receive ``retention_replay`` or an update scale,
        so the optimizer cannot train on the same examples used to decide
        whether its candidate step is safe.
        """
        count = max(0, int(count))
        if count <= 0:
            return []
        excluded = set(exclude_ids or ())
        eligible = [
            self.episodes[episode_id]
            for episode_id in self.anchor_ids
            if episode_id in self.episodes
            and episode_id not in excluded
            and self.episodes[episode_id].admitted_batch < self.batch_index
            and bool(getattr(self.episodes[episode_id], "audit_only", False))
        ]
        selected_ids = self._balanced_ids(eligible, count)
        result: list[dict[str, object]] = []
        for episode_id in selected_ids:
            episode = self.episodes[episode_id]
            payload = copy.deepcopy(episode.payload)
            payload["retention_audit_only"] = True
            payload["retention_anchor"] = True
            payload["retention_behavior_anchor"] = bool(episode.behavior_anchor)
            payload["retention_hard_negative"] = bool(episode.hard_negative)
            payload["retention_preference_anchor"] = bool(episode.preference_anchor)
            payload["retention_source_bucket"] = episode.bucket
            result.append(payload)
        return result

    def note_batch(self, replay_items: list[Mapping[str, object]]) -> None:
        for item in replay_items:
            episode_id = str(item.get("retention_episode_id", "") or "")
            episode = self.episodes.get(episode_id)
            if episode is not None:
                episode.replay_count += 1
                self.replayed_count += 1
        self.batch_index += 1

    def stats(self) -> dict[str, object]:
        lifecycle = sum(
            1 for row in self.episodes.values()
            if not bool(row.payload.get("entry_preference_only", False))
        )
        behavior_anchors = sum(
            bool(getattr(row, "behavior_anchor", False)) for row in self.episodes.values()
        )
        preference_anchors = sum(
            bool(getattr(row, "preference_anchor", False)) for row in self.episodes.values()
        )
        audit_only_anchors = sum(
            bool(row.anchor and getattr(row, "audit_only", False))
            for row in self.episodes.values()
        )
        return {
            "schema_version": int(self.schema_version),
            "episodes": int(len(self.episodes)),
            "lifecycle_episodes": int(lifecycle),
            "preference_episodes": int(len(self.episodes) - lifecycle),
            "recent": int(len(self.recent_ids)),
            "diverse": int(len(self.diverse_ids)),
            "anchors": int(len(self.anchor_ids)),
            "behavior_anchors": int(behavior_anchors),
            "preference_anchors": int(preference_anchors),
            "hard_negative_anchors": int(sum(
                bool(row.hard_negative) for row in self.episodes.values()
            )),
            "audit_only_anchors": int(audit_only_anchors),
            "batch_index": int(self.batch_index),
            "admitted_count": int(self.admitted_count),
            "replayed_count": int(self.replayed_count),
            "discarded_count": int(self.discarded_count),
        }
