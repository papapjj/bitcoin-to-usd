"""Context- and action-scoped robust expectancy learning.

This module provides the normative value signal used by the self-growing
trader.  It intentionally treats expectancy as *net realised R* from finalised
lifecycles, then estimates a conservative lower confidence bound (LCB) within
one compatible context and action scope.

Design invariants
-----------------
* Action scopes are exact: BUY evidence never biases SELL/exit/manage.
* Bootstrap/mask/risk-override and derived memory outcomes are stored for audit
  but never create policy action bias.
* Small samples are shrunk hard toward zero and carry a negative/low LCB.
* Retrieval is similarity-weighted in the canonical ContextFrame coordinate
  system; raw features and latent M1 activations are never mixed.
* The ledger is checkpoint-friendly, deterministic, and uses no salted Python
  ``hash`` identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .context_frame import MEMORY_KEY_SIZE, blockwise_cosine
from .replacement_learning import action_scope, scope_matches, stable_identifier

EXPECTANCY_SCHEMA_VERSION = 14
ACTION_COUNT = 12
POLICY_ORIGINS = {"policy"}
# A management/exit decision does not have a causal value from the final
# position R by itself.  Without a validated counterfactual path, keep that
# evidence visible but prevent it from suppressing the entry policy.
MANAGEMENT_SHADOW_ORIGINS = {"management_shadow"}
NON_POLICY_ORIGINS = {
    "bootstrap", "mask", "risk_override", "forced_exit", "memory_extrema",
    "management_shadow", "unknown",
}
DERIVED_EVIDENCE_ORIGINS = {"memory_extrema"}


def _vector(values: object, size: int = MEMORY_KEY_SIZE) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raw = np.zeros(0, dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=2.0, neginf=-2.0)
    out = np.zeros(max(0, int(size)), dtype=float)
    out[: min(out.size, raw.size)] = raw[: out.size]
    return np.clip(out, -2.0, 2.0)


def _norm_origin(origin: object) -> str:
    value = str(origin or "policy").strip().lower()
    return value if value else "policy"


def _scope_fingerprint(scope: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(scope.get("action_index", -1) or -1),
        str(scope.get("action_family", "") or ""),
        str(scope.get("side", "") or ""),
        str(scope.get("lifecycle", "") or ""),
        str(scope.get("position_state", "") or ""),
        int(scope.get("target_slot", -1) or -1),
        str(scope.get("management_mode", "") or ""),
    )


def context_bucket(context_key: object) -> tuple[int, ...]:
    """A stable coarse key only for record identity, never for retrieval.

    Similarity still controls recall.  The bucket merely prevents every one-tick
    floating-point perturbation from spawning a separate statistical record.
    """
    key = _vector(context_key)
    # Pool 46 semantic dimensions into 10 coarse bins, then quantise in 0.5
    # increments.  This preserves market/account/position distinctions while
    # allowing nearby contexts to accumulate an evidence sample.
    edges = np.linspace(0, key.size, 11)
    pooled: list[float] = []
    for index in range(10):
        start = int(np.floor(edges[index]))
        stop = min(key.size, max(start + 1, int(np.ceil(edges[index + 1]))))
        pooled.append(float(np.mean(key[start:stop])) if start < stop else 0.0)
    return tuple(int(np.clip(np.rint(item / 0.50), -4, 4)) for item in pooled)


@dataclass(frozen=True)
class ExpectancyEstimate:
    sample_count: int = 0
    mean_net_r: float = 0.0
    std_net_r: float = 0.0
    expectancy_lcb: float = 0.0
    cvar_5pct: float = 0.0
    win_rate: float = 0.0
    confidence: float = 0.0
    reliability: float = 0.0
    policy_origin_ratio: float = 0.0
    eligible: bool = False

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "sample_count": int(self.sample_count),
            "mean_net_r": float(self.mean_net_r),
            "std_net_r": float(self.std_net_r),
            "expectancy_lcb": float(self.expectancy_lcb),
            "cvar_5pct": float(self.cvar_5pct),
            "win_rate": float(self.win_rate),
            "confidence": float(self.confidence),
            "reliability": float(self.reliability),
            "policy_origin_ratio": float(self.policy_origin_ratio),
            "eligible": bool(self.eligible),
        }


def robust_expectancy(
    values: object,
    *,
    min_samples: int = 6,
    prior_strength: float = 4.0,
    z_score: float = 1.645,
    policy_origin_ratio: float = 1.0,
) -> ExpectancyEstimate:
    """Compute a conservative robust expectancy estimate in net R.

    The mean is shrunk to zero for small samples and the LCB includes a prior
    variance floor.  Consequently a single +3R trade cannot appear as a
    reliable positive edge.
    """
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        arr = np.zeros(0, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, -12.0, 12.0)
    n = int(arr.size)
    if n <= 0:
        return ExpectancyEstimate(policy_origin_ratio=float(np.clip(policy_origin_ratio, 0.0, 1.0)))
    mean = float(np.mean(arr))
    sample_std = float(np.std(arr, ddof=1)) if n > 1 else max(1.0, abs(mean))
    # Prior scale is deliberately 1R: no small group of trades earns a narrow
    # confidence interval merely because its observed variance happened to be
    # near zero.
    prior_var = 1.0
    shrink = float(n / (n + max(0.0, prior_strength)))
    shrunk_mean = shrink * mean
    posterior_var = ((max(0, n - 1) * sample_std * sample_std) + prior_strength * prior_var) / max(1.0, (n - 1) + prior_strength)
    standard_error = float(np.sqrt(max(1e-12, posterior_var) / max(1.0, n)))
    lcb = float(shrunk_mean - max(0.0, z_score) * standard_error)
    tail_count = max(1, int(np.ceil(0.05 * n)))
    cvar = float(max(0.0, -float(np.mean(np.sort(arr)[:tail_count]))))
    win_rate = float(np.mean(arr > 0.0))
    coverage = float(np.sqrt(min(1.0, n / max(1.0, float(min_samples)))))
    origin_ratio = float(np.clip(policy_origin_ratio, 0.0, 1.0))
    # Confidence captures both sample evidence and the fraction actually
    # generated by policy rather than bridge bootstrap/masking.
    confidence = float(np.clip(coverage * origin_ratio * (1.0 / (1.0 + standard_error)), 0.0, 1.0))
    reliability = float(np.clip(coverage * origin_ratio, 0.0, 1.0))
    return ExpectancyEstimate(
        sample_count=n,
        mean_net_r=shrunk_mean,
        std_net_r=float(np.sqrt(max(0.0, posterior_var))),
        expectancy_lcb=lcb,
        cvar_5pct=cvar,
        win_rate=win_rate,
        confidence=confidence,
        reliability=reliability,
        policy_origin_ratio=origin_ratio,
        eligible=bool(n >= int(min_samples) and origin_ratio >= 0.50),
    )


@dataclass
class ContextActionExpectancyRecord:
    record_id: str
    context_prototype: np.ndarray
    scope: dict[str, object]
    values: list[float] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)
    lifecycle_tags: list[tuple[str, ...]] = field(default_factory=list)
    age: int = 0

    def copy(self) -> "ContextActionExpectancyRecord":
        return ContextActionExpectancyRecord(
            record_id=str(self.record_id),
            context_prototype=_vector(self.context_prototype),
            scope=dict(self.scope),
            values=[float(item) for item in self.values],
            origins=[str(item) for item in self.origins],
            trace_ids=[str(item) for item in self.trace_ids],
            frame_ids=[str(item) for item in self.frame_ids],
            lifecycle_tags=[tuple(str(tag) for tag in item) for item in self.lifecycle_tags],
            age=int(self.age),
        )

    def observe(
        self,
        context_key: object,
        net_r: float,
        *,
        origin: str,
        trace_id: str = "",
        frame_id: str = "",
        lifecycle_tags: object = (),
        max_samples: int = 256,
    ) -> None:
        context = _vector(context_key)
        lr = 1.0 / float(min(max(1, self.age + 1), 32))
        self.context_prototype = (1.0 - lr) * _vector(self.context_prototype) + lr * context
        self.values.append(float(np.clip(float(net_r), -12.0, 12.0)))
        self.origins.append(_norm_origin(origin))
        self.trace_ids.append(str(trace_id or ""))
        self.frame_ids.append(str(frame_id or ""))
        tags = tuple(str(item) for item in (lifecycle_tags or ()) if str(item))
        self.lifecycle_tags.append(tags)
        cap = max(8, int(max_samples))
        if len(self.values) > cap:
            self.values = self.values[-cap:]
            self.origins = self.origins[-cap:]
            self.trace_ids = self.trace_ids[-cap:]
            self.frame_ids = self.frame_ids[-cap:]
            self.lifecycle_tags = self.lifecycle_tags[-cap:]
        self.age += 1

    @property
    def policy_origin_ratio(self) -> float:
        if not self.origins:
            return 0.0
        return float(np.mean([_norm_origin(origin) in POLICY_ORIGINS for origin in self.origins]))

    def estimate(self, *, policy_only: bool, min_samples: int) -> ExpectancyEstimate:
        if policy_only:
            values = [value for value, origin in zip(self.values, self.origins) if _norm_origin(origin) in POLICY_ORIGINS]
            ratio = self.policy_origin_ratio
        else:
            values = self.values
            ratio = self.policy_origin_ratio
        return robust_expectancy(values, min_samples=min_samples, policy_origin_ratio=ratio if policy_only else 1.0)


@dataclass
class ContextualExpectancyLedger:
    """Persistent non-symbolic evidence ledger for net-R expectancy."""

    records: dict[str, ContextActionExpectancyRecord] = field(default_factory=dict)
    schema_version: int = EXPECTANCY_SCHEMA_VERSION
    min_samples: int = 6
    merge_similarity: float = 0.90
    # Query matching is intentionally looser than record consolidation.  A
    # frozen OOS state may move within the same regime without being close
    # enough to the training prototype to contribute evidence.  Keep the
    # historical default, while allowing an isolated candidate to widen the
    # read-only evidence radius without changing how new records merge.
    query_min_similarity: float = 0.35
    max_samples_per_record: int = 256

    def copy(self) -> "ContextualExpectancyLedger":
        return ContextualExpectancyLedger(
            records={str(key): value.copy() for key, value in self.records.items()},
            schema_version=int(self.schema_version),
            min_samples=max(1, int(self.min_samples)),
            merge_similarity=float(self.merge_similarity),
            query_min_similarity=float(self.query_min_similarity),
            max_samples_per_record=max(8, int(self.max_samples_per_record)),
        )

    def ensure_compatible(self) -> None:
        stored_schema = int(getattr(self, "schema_version", 0) or 0)
        if stored_schema != EXPECTANCY_SCHEMA_VERSION:
            # V13 records may contain duplicate management traces and derived
            # extrema evidence under lifecycle origins. They are audit history,
            # not trustworthy decision-time samples after the V14 contract.
            # Reset only this ledger; neural/network memory remains intact.
            self.records = {}
            self._compatibility_signature = None
            self.schema_version = EXPECTANCY_SCHEMA_VERSION
        raw_records = getattr(self, "records", None)
        signature = (
            EXPECTANCY_SCHEMA_VERSION,
            len(raw_records) if isinstance(raw_records, dict) else -1,
            int(getattr(self, "min_samples", 6) or 6),
            float(getattr(self, "merge_similarity", 0.90) or 0.90),
            float(getattr(self, "query_min_similarity", 0.35) or 0.35),
            int(getattr(self, "max_samples_per_record", 256) or 256),
        )
        if getattr(self, "_compatibility_signature", None) == signature:
            return
        if not isinstance(getattr(self, "records", None), dict):
            self.records = {}
        converted: dict[str, ContextActionExpectancyRecord] = {}
        for key, value in self.records.items():
            if isinstance(value, ContextActionExpectancyRecord):
                value.context_prototype = _vector(value.context_prototype)
                value.scope = dict(value.scope or {})
                value.values = [float(item) for item in list(value.values or []) if np.isfinite(float(item))]
                value.origins = [_norm_origin(item) for item in list(value.origins or [])]
                converted[str(key)] = value
        self.records = converted
        self.schema_version = EXPECTANCY_SCHEMA_VERSION
        self.min_samples = max(1, int(getattr(self, "min_samples", 6) or 6))
        self.merge_similarity = float(np.clip(getattr(self, "merge_similarity", 0.90) or 0.90, 0.50, 0.999))
        self.query_min_similarity = float(np.clip(
            getattr(self, "query_min_similarity", 0.35) or 0.35,
            0.05,
            0.95,
        ))
        self.max_samples_per_record = max(8, int(getattr(self, "max_samples_per_record", 256) or 256))
        self._compatibility_signature = (
            EXPECTANCY_SCHEMA_VERSION,
            len(self.records),
            int(self.min_samples),
            float(self.merge_similarity),
            float(self.query_min_similarity),
            int(self.max_samples_per_record),
        )

    def _find_merge_target(self, context_key: np.ndarray, scope: Mapping[str, object]) -> ContextActionExpectancyRecord | None:
        best: ContextActionExpectancyRecord | None = None
        best_similarity = float(self.merge_similarity)
        for record in self.records.values():
            if not scope_matches(record.scope, scope):
                continue
            sim = max(0.0, blockwise_cosine(context_key, record.context_prototype))
            if sim >= best_similarity:
                best = record
                best_similarity = sim
        return best

    def observe(
        self,
        context_key: object,
        scope: Mapping[str, object],
        net_r: float,
        *,
        origin: object = "policy",
        trace_id: str = "",
        frame_id: str = "",
        lifecycle_tags: object = (),
    ) -> ContextActionExpectancyRecord:
        self.ensure_compatible()
        context = _vector(context_key)
        normalized_scope = dict(scope or {})
        target = self._find_merge_target(context, normalized_scope)
        if target is None:
            identifier = stable_identifier(
                "context_expectancy",
                _scope_fingerprint(normalized_scope),
                context_bucket(context),
                length=32,
            )
            target = self.records.get(identifier)
            if target is None:
                target = ContextActionExpectancyRecord(
                    record_id=identifier,
                    context_prototype=context.copy(),
                    scope=normalized_scope,
                )
                self.records[identifier] = target
        target.observe(
            context,
            float(net_r),
            origin=_norm_origin(origin),
            trace_id=trace_id,
            frame_id=frame_id,
            lifecycle_tags=lifecycle_tags,
            max_samples=self.max_samples_per_record,
        )
        return target

    def query(
        self,
        context_key: object,
        *,
        output_size: int,
        positions_pipe: str = "",
        allowed_action_mask: object | None = None,
    ) -> dict[str, object]:
        self.ensure_compatible()
        query_min_similarity = float(np.clip(
            getattr(self, "query_min_similarity", 0.35) or 0.35,
            0.05,
            0.95,
        ))
        key = _vector(context_key)
        size = max(1, int(output_size))
        mask = np.ones(size, dtype=bool)
        if allowed_action_mask is not None:
            raw = np.asarray(allowed_action_mask, dtype=bool).reshape(-1)
            if raw.size:
                mask[:] = False
                mask[: min(size, raw.size)] = raw[:size]
        if not np.any(mask):
            mask[0] = True

        means = np.zeros(size, dtype=float)
        lcbs = np.zeros(size, dtype=float)
        samples = np.zeros(size, dtype=float)
        confidence = np.zeros(size, dtype=float)
        cvar = np.zeros(size, dtype=float)
        evidence_weight = np.zeros(size, dtype=float)
        contributors: dict[str, list[dict[str, object]]] = {str(index): [] for index in range(size)}
        for index in range(min(size, ACTION_COUNT)):
            scope = action_scope(index, positions_pipe=positions_pipe)
            pieces: list[tuple[float, ContextActionExpectancyRecord, ExpectancyEstimate]] = []
            for record in self.records.values():
                if not scope_matches(record.scope, scope):
                    continue
                estimate = record.estimate(policy_only=True, min_samples=self.min_samples)
                if estimate.sample_count <= 0:
                    continue
                similarity = max(0.0, blockwise_cosine(key, record.context_prototype))
                if similarity < query_min_similarity:
                    continue
                weight = similarity * max(0.05, estimate.reliability)
                if weight <= 1e-9:
                    continue
                pieces.append((weight, record, estimate))
            if not pieces:
                continue
            pieces.sort(key=lambda item: item[0], reverse=True)
            pieces = pieces[:5]
            total = float(sum(item[0] for item in pieces))
            if total <= 1e-12:
                continue
            means[index] = float(sum(weight * est.mean_net_r for weight, _, est in pieces) / total)
            lcbs[index] = float(sum(weight * est.expectancy_lcb for weight, _, est in pieces) / total)
            samples[index] = float(sum(weight * est.sample_count for weight, _, est in pieces) / total)
            confidence[index] = float(np.clip(sum(weight * est.confidence for weight, _, est in pieces) / total, 0.0, 1.0))
            cvar[index] = float(sum(weight * est.cvar_5pct for weight, _, est in pieces) / total)
            evidence_weight[index] = float(np.clip(total / max(1.0, len(pieces)), 0.0, 1.0))
            contributors[str(index)] = [
                {
                    "record_id": rec.record_id,
                    "similarity": float(weight / max(0.05, est.reliability)),
                    "weight": float(weight),
                    "scope": dict(rec.scope),
                    **est.as_dict(),
                }
                for weight, rec, est in pieces
            ]
        # Positive LCB turns into bounded action evidence.  Reliable negative
        # evidence remains *action-local*: it is carried forward with its
        # support/confidence so M1 can penalise only candidates whose primary
        # action matches the demonstrated bad action in this context.
        positive_lcb = np.maximum(0.0, lcbs)
        action_bias = np.tanh(positive_lcb * np.clip(confidence, 0.0, 1.0))
        action_bias[~mask] = 0.0
        reliable_negative = (lcbs < 0.0) & (samples >= float(self.min_samples)) & (confidence >= 0.20)
        reliable_negative[~mask] = False
        negative_inhibition = np.maximum(0.0, -lcbs) * np.clip(confidence, 0.0, 1.0) * np.clip(evidence_weight, 0.0, 1.0)
        negative_inhibition[~reliable_negative] = 0.0
        entry_indices = np.array([index for index in range(min(size, 7)) if index in {1, 2, 3, 4, 5, 6}], dtype=int)
        legal_entry = entry_indices[mask[entry_indices]] if entry_indices.size else np.zeros(0, dtype=int)
        if legal_entry.size:
            # Choose one entry action by lower-bound evidence.  Confidence and
            # support must refer to the same action, never maxima from different
            # rows, otherwise a weak BUY LCB can borrow SELL evidence.
            best_entry_index = int(legal_entry[int(np.argmax(lcbs[legal_entry]))])
            best_entry_lcb = float(lcbs[best_entry_index])
            best_entry_confidence = float(confidence[best_entry_index])
            best_entry_support = float(samples[best_entry_index])
            best_entry_evidence_weight = float(evidence_weight[best_entry_index])
        else:
            best_entry_index = 0
            best_entry_lcb = 0.0
            best_entry_confidence = 0.0
            best_entry_support = 0.0
            best_entry_evidence_weight = 0.0
        entry_evidence_verified = bool(
            best_entry_lcb > 0.0
            and best_entry_support >= float(self.min_samples)
            and best_entry_confidence >= 0.20
            and best_entry_evidence_weight > 0.0
        )
        # This is a compact diagnostic only.  It must be zero until the entire
        # evidence tuple above is verified; downstream curiosity must not use
        # uncertainty/no-trade pressure as a substitute.
        market_opportunity = float(
            np.clip(max(0.0, best_entry_lcb) * best_entry_confidence, 0.0, 1.0)
            if entry_evidence_verified
            else 0.0
        )
        selected_idx = int(np.argmax(action_bias)) if action_bias.size else 0
        selected_scope = action_scope(selected_idx, positions_pipe=positions_pipe)
        selected_lcb = float(lcbs[selected_idx]) if selected_idx < lcbs.size else 0.0
        best_lcb = float(np.max(lcbs[mask])) if np.any(mask) else 0.0
        return {
            "expectancy_schema_version": EXPECTANCY_SCHEMA_VERSION,
            "expectancy_action_mean": means,
            "expectancy_action_lcb": lcbs,
            "expectancy_action_samples": samples,
            "expectancy_action_confidence": confidence,
            "expectancy_action_cvar": cvar,
            "expectancy_action_bias": action_bias,
            "expectancy_reliable_negative_mask": reliable_negative,
            "expectancy_negative_inhibition": negative_inhibition,
            "expectancy_market_opportunity": market_opportunity,
            "expectancy_entry_lcb": float(best_entry_lcb),
            "expectancy_entry_confidence": float(best_entry_confidence),
            "expectancy_entry_support": float(best_entry_support),
            "expectancy_entry_evidence_weight": float(best_entry_evidence_weight),
            "expectancy_entry_evidence_verified": bool(entry_evidence_verified),
            "expectancy_verified_entry_index": int(best_entry_index),
            "expectancy_best_action": int(selected_idx),
            "expectancy_best_lcb": best_lcb,
            "expectancy_selected_scope": selected_scope,
            "expectancy_regret": float(max(0.0, best_lcb - selected_lcb)),
            "expectancy_contributors": contributors,
            "expectancy_record_count": int(len(self.records)),
        }

    def aggregate_action_expectancy(
        self,
        action_index: int,
        *,
        positions_pipe: str = "",
    ) -> ExpectancyEstimate:
        """Return owner-local evidence across context prototypes for one action.

        Exact prototype retrieval remains the primary path.  This fallback is
        deliberately action/side/lifecycle scoped and never mixes BUY with
        SELL or entry with management.  It exists for candidate hardening when
        a frozen OOS state moves outside the prototype radius of an otherwise
        reliable owner-local record.
        """
        try:
            index = int(action_index)
        except (TypeError, ValueError):
            return ExpectancyEstimate()
        if index < 0 or index >= ACTION_COUNT:
            return ExpectancyEstimate()
        target = action_scope(index, positions_pipe=positions_pipe)
        values: list[float] = []
        for record in self.records.values():
            scope = dict(getattr(record, "scope", {}) or {})
            # Ignore only prototype/position-slot identity, but keep the
            # semantic action and account-state contract exact. A flat BUY
            # cannot borrow SELL/open or management evidence through fallback.
            if int(scope.get("action_index", -1) or -1) != index:
                continue
            if any(
                str(scope.get(field, "") or "") != str(target.get(field, "") or "")
                for field in ("action_family", "side", "lifecycle", "position_state")
            ):
                continue
            values.extend(
                float(value)
                for value, origin in zip(
                    list(getattr(record, "values", ()) or ()),
                    list(getattr(record, "origins", ()) or ()),
                )
                if _norm_origin(origin) in POLICY_ORIGINS
            )
        return robust_expectancy(values, min_samples=int(self.min_samples), policy_origin_ratio=1.0)

    def report(self) -> dict[str, object]:
        self.ensure_compatible()
        policy_values: list[float] = []
        lifecycle_values: list[float] = []
        management_shadow_values: list[float] = []
        derived_evidence_count = 0
        derived_evidence_origins: dict[str, int] = {}
        origin_counts: dict[str, int] = {}
        negative_critical = 0
        positive_scopes = 0
        scopes: list[dict[str, object]] = []
        for record in self.records.values():
            for value, origin in zip(record.values, record.origins):
                normalized_origin = _norm_origin(origin)
                if normalized_origin in DERIVED_EVIDENCE_ORIGINS:
                    derived_evidence_count += 1
                    derived_evidence_origins[normalized_origin] = derived_evidence_origins.get(normalized_origin, 0) + 1
                elif normalized_origin in MANAGEMENT_SHADOW_ORIGINS:
                    management_shadow_values.append(float(value))
                else:
                    lifecycle_values.append(float(value))
            selected = [
                float(value)
                for value, origin in zip(record.values, record.origins)
                if _norm_origin(origin) in POLICY_ORIGINS
            ]
            policy_values.extend(selected)
            for origin in record.origins:
                origin_counts[_norm_origin(origin)] = origin_counts.get(_norm_origin(origin), 0) + 1
            estimate = record.estimate(policy_only=True, min_samples=self.min_samples)
            if estimate.eligible and estimate.expectancy_lcb < 0.0:
                negative_critical += 1
            if estimate.eligible and estimate.expectancy_lcb > 0.0:
                positive_scopes += 1
            scopes.append({"record_id": record.record_id, "scope": dict(record.scope), **estimate.as_dict()})
        all_values = lifecycle_values + management_shadow_values
        total = len(all_values)
        policy_count = len(policy_values)
        policy_ratio = float(policy_count / max(1, total)) if total else 0.0
        global_policy = robust_expectancy(policy_values, min_samples=self.min_samples, policy_origin_ratio=policy_ratio)
        global_all = robust_expectancy(all_values, min_samples=self.min_samples, policy_origin_ratio=1.0)
        management_shadow = robust_expectancy(
            management_shadow_values,
            min_samples=self.min_samples,
            policy_origin_ratio=0.0,
        )
        scopes.sort(key=lambda item: (float(item.get("expectancy_lcb", 0.0)), int(item.get("sample_count", 0))), reverse=True)
        return {
            "schema_version": EXPECTANCY_SCHEMA_VERSION,
            "record_count": int(len(self.records)),
            "policy_origin_lifecycle_count": int(policy_count),
            "all_origin_lifecycle_count": int(total),
            "policy_origin_ratio": policy_ratio,
            "management_shadow_lifecycle_count": int(len(management_shadow_values)),
            "management_shadow_expectancy": management_shadow.as_dict(),
            "derived_evidence_count": int(derived_evidence_count),
            "derived_evidence_origins": derived_evidence_origins,
            "policy_expectancy": global_policy.as_dict(),
            "all_expectancy": global_all.as_dict(),
            "negative_critical_scope_count": int(negative_critical),
            "positive_eligible_scope_count": int(positive_scopes),
            "origin_counts": origin_counts,
            "scope_preview": scopes[:24],
        }

    def action_regret(
        self,
        context_key: object,
        *,
        selected_action_index: int,
        positions_pipe: str = "",
        allowed_action_mask: object | None = None,
    ) -> float:
        query = self.query(
            context_key,
            output_size=ACTION_COUNT,
            positions_pipe=positions_pipe,
            allowed_action_mask=allowed_action_mask,
        )
        lcbs = np.asarray(query.get("expectancy_action_lcb", np.zeros(ACTION_COUNT)), dtype=float)
        selected = int(np.clip(int(selected_action_index), 0, max(0, lcbs.size - 1))) if lcbs.size else 0
        mask = np.ones(lcbs.size, dtype=bool)
        if allowed_action_mask is not None:
            raw = np.asarray(allowed_action_mask, dtype=bool).reshape(-1)
            if raw.size:
                mask[:] = False
                mask[: min(mask.size, raw.size)] = raw[:mask.size]
        best = float(np.max(lcbs[mask])) if lcbs.size and np.any(mask) else 0.0
        current = float(lcbs[selected]) if lcbs.size else 0.0
        return float(max(0.0, best - current))
