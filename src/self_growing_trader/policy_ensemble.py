from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyCandidateScore:
    policy_id: str
    confidence: float
    recent_utility: float
    context_key: str = ""
    evidence_group: str = ""
    memory_match: float = 0.0
    proposal_margin: float = 0.0
    role_fit: float = 0.0
    account_stress: float = 0.0
    drawdown: float = 0.0
    open_positions: int = 0
    pending_outcomes: int = 0
    correlated_exposure: int = 0
    valid: bool = True
    reason: str = ""
    # True-lifecycle/evidence metadata.  These fields are optional so older
    # pickles and direct callers remain loadable; the facade fills them for
    # context-role candidates.
    evidence_support: int = 0
    lifecycle_support: int = 0
    readiness: float = 0.0
    cold_start: bool = False


@dataclass(frozen=True)
class PolicySelectionTrace:
    selected_policy_id: str
    fallback_policy_id: str
    scores: dict[str, float]
    reason: str
    used_fallback: bool = False


@dataclass(frozen=True)
class CorrelatedExposureState:
    """One direction/context cluster, used for selection and evidence telemetry."""

    market_context: str
    direction: int
    exposure_count: int
    open_position_count: int
    pending_trace_count: int
    mean_unrealized_r: float
    stage: str
    utility_context: str


@dataclass(frozen=True)
class RoleReadinessState:
    """True-lifecycle readiness for one role and one entry direction."""

    evidence_group: str
    support: int
    learned_utility: float
    readiness: float
    risk_multiplier: float


class RoleReadinessSupervisor:
    """Measure cold-role readiness without changing entry or volume.

    A context-role specialist must still trade to obtain causal evidence. The
    supervisor emits telemetry for hierarchical evidence; it never invalidates
    an entry and never rewrites the policy risk head.
    """

    def __init__(
        self,
        *,
        min_risk_multiplier: float = 0.25,
        support_scale: float = 4.0,
        negative_utility_weight: float = 0.50,
    ) -> None:
        self.min_risk_multiplier = float(np.clip(float(min_risk_multiplier), 0.05, 1.0))
        self.support_scale = max(1.0, float(support_scale))
        self.negative_utility_weight = float(np.clip(float(negative_utility_weight), 0.0, 1.0))

    def assess(
        self,
        *,
        evidence_group: str,
        support: int,
        learned_utility: float,
    ) -> RoleReadinessState:
        support = max(0, int(support))
        learned_utility = float(np.clip(float(learned_utility), -1.0, 1.0))
        evidence_progress = float(1.0 - np.exp(-support / self.support_scale))
        readiness = float(
            self.min_risk_multiplier
            + (1.0 - self.min_risk_multiplier) * evidence_progress
        )
        if learned_utility < 0.0:
            readiness *= float(max(
                self.min_risk_multiplier,
                1.0 + self.negative_utility_weight * learned_utility,
            ))
        risk_multiplier = float(np.clip(
            readiness,
            self.min_risk_multiplier,
            1.0,
        ))
        return RoleReadinessState(
            evidence_group=str(evidence_group or ""),
            support=support,
            learned_utility=learned_utility,
            readiness=float(readiness),
            risk_multiplier=risk_multiplier,
        )

class CorrelatedExposureSupervisor:
    """Specialise selector evidence by exposure stage without blocking entry.

    The supervisor never changes action logits or the risk head. It gives
    PRIMARY, profitable add-on and drawdown add-on decisions separate utility
    histories; selector penalties remain the exposure response.
    """

    def __init__(self, *, risk_decay: float = 0.50, min_risk_pct: float = 0.001) -> None:
        self.risk_decay = float(np.clip(float(risk_decay), 0.05, 1.0))
        self.min_risk_pct = max(1e-9, float(min_risk_pct))

    def assess(
        self,
        *,
        market_context: str,
        direction: int,
        positions: list[Mapping[str, object]],
        pending_entries: list[Mapping[str, object]],
    ) -> CorrelatedExposureState:
        market_context = str(market_context or "")
        direction = 1 if int(direction) > 0 else (-1 if int(direction) < 0 else 0)
        matched_positions = [
            item for item in positions
            if str(item.get("owner_context", "") or "") == market_context
            and int(item.get("side", 0) or 0) == direction
        ]
        exposure_tokens: set[str] = set()
        for item in matched_positions:
            trace_id = str(item.get("owner_trace", "") or "")
            ticket = str(item.get("ticket", "") or "")
            exposure_tokens.add(f"trace:{trace_id}" if trace_id else f"position:{ticket}")
        matched_pending = [
            item for item in pending_entries
            if str(item.get("context", "") or "") == market_context
            and int(item.get("direction", 0) or 0) == direction
        ]
        for item in matched_pending:
            trace_id = str(item.get("trace_id", "") or "")
            if trace_id:
                exposure_tokens.add(f"trace:{trace_id}")
        unrealized = [float(item.get("unrealized_r", 0.0) or 0.0) for item in matched_positions]
        mean_unrealized = float(np.mean(unrealized)) if unrealized else 0.0
        exposure_count = len(exposure_tokens) if direction else 0
        if exposure_count <= 0:
            stage = "PRIMARY"
            utility_context = market_context
        elif matched_positions and mean_unrealized > 0.0:
            stage = "ADDON_PROFIT"
            utility_context = f"{market_context}|{stage}|SIDE_{direction}"
        elif matched_positions:
            stage = "ADDON_DRAWDOWN"
            utility_context = f"{market_context}|{stage}|SIDE_{direction}"
        else:
            stage = "ADDON_PENDING"
            utility_context = f"{market_context}|{stage}|SIDE_{direction}"
        return CorrelatedExposureState(
            market_context=market_context,
            direction=direction,
            exposure_count=int(exposure_count),
            open_position_count=len(matched_positions),
            pending_trace_count=len(matched_pending),
            mean_unrealized_r=mean_unrealized,
            stage=stage,
            utility_context=utility_context,
        )

class PolicyEnsembleSelector:
    def __init__(
        self,
        *,
        fallback_policy_id: str = "m1",
        stress_penalty: float = 0.4,
        drawdown_penalty: float = 0.3,
        min_score: float = -1.0,
        utility_alpha: float = 0.35,
        exploration_weight: float = 0.25,
        inflight_penalty: float = 0.20,
        exposure_penalty: float = 0.20,
        win_signal_alpha: float = 0.35,
        loss_streak_penalty: float = 0.12,
        policy_group_loss_streak_penalty: float = 0.0,
        cold_start_penalty: float = 0.60,
        cold_start_support: int = 8,
    ) -> None:
        self.fallback_policy_id = fallback_policy_id
        self.stress_penalty = float(stress_penalty)
        self.drawdown_penalty = float(drawdown_penalty)
        self.min_score = float(min_score)
        self.utility_alpha = float(np.clip(float(utility_alpha), 0.001, 1.0))
        self.exploration_weight = float(np.clip(float(exploration_weight), 0.0, 1.0))
        self.inflight_penalty = float(np.clip(float(inflight_penalty), 0.0, 1.0))
        self.exposure_penalty = float(np.clip(float(exposure_penalty), 0.0, 1.0))
        self.win_signal_alpha = float(np.clip(float(win_signal_alpha), 0.001, 1.0))
        self.loss_streak_penalty = float(np.clip(float(loss_streak_penalty), 0.0, 1.0))
        # Optional continuation-only backoff: aggregate a policy's losses
        # across context/microstate keys while keeping role direction in the
        # key.  The protected v13 default is disabled.
        self.policy_group_loss_streak_penalty = float(
            np.clip(float(policy_group_loss_streak_penalty), 0.0, 1.0)
        )
        self.cold_start_penalty = float(np.clip(float(cold_start_penalty), 0.0, 2.0))
        self.cold_start_support = max(1, int(cold_start_support))
        self.learned_utility: dict[str, float] = {}
        self.outcome_count: dict[str, int] = {}
        self.win_signal: dict[str, float] = {}
        self.loss_streak: dict[str, int] = {}
        # Leaf evidence remains microstate-specific.  These family maps are a
        # hierarchical back-off shared by the same role and entry direction,
        # so a STOP loss cannot be forgotten merely by moving to a new context
        # or transition microstate.
        self.group_learned_utility: dict[str, float] = {}
        self.group_outcome_count: dict[str, int] = {}
        self.group_win_signal: dict[str, float] = {}
        self.group_loss_streak: dict[str, int] = {}
        self.policy_group_loss_streak: dict[str, int] = {}
        self.last_trace: PolicySelectionTrace | None = None

    def select(
        self,
        candidates: list[PolicyCandidateScore],
        *,
        require_valid_selection: bool = False,
    ) -> PolicySelectionTrace:
        if not candidates:
            trace = PolicySelectionTrace(
                selected_policy_id=self.fallback_policy_id,
                fallback_policy_id=self.fallback_policy_id,
                scores={},
                reason="no_candidates",
                used_fallback=True,
            )
            self.last_trace = trace
            return trace
        # Cold-start specialists are deliberately not invalidated: if every
        # valid candidate is cold, one can still be selected to collect its
        # first causal lifecycle.  Once a non-cold candidate is available,
        # however, an unseen specialist must pay a bounded exploration cost so
        # proposal confidence cannot turn it into a permanent overfit route.
        self._selection_has_non_cold_candidate = any(
            not bool(getattr(candidate, "cold_start", False))
            for candidate in candidates
            if bool(getattr(candidate, "valid", True))
        )
        scores: dict[str, float] = {}
        for candidate in candidates:
            scores[candidate.policy_id] = self._score(candidate)
        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        if not valid_candidates:
            return self._fallback(scores, "no_valid_candidates")
        best = max(valid_candidates, key=lambda item: scores[item.policy_id])
        if scores[best.policy_id] < self.min_score and not require_valid_selection:
            return self._fallback(scores, "score_below_min")
        trace = PolicySelectionTrace(
            selected_policy_id=best.policy_id,
            fallback_policy_id=self.fallback_policy_id,
            scores=scores,
            reason=("best_valid_required" if scores[best.policy_id] < self.min_score else (best.reason or "best_score")),
            used_fallback=False,
        )
        self.last_trace = trace
        return trace

    def _score(self, candidate: PolicyCandidateScore) -> float:
        if not candidate.valid:
            return -float("inf")
        utility_key = self._utility_key(candidate.policy_id, candidate.context_key)
        utilities = getattr(self, "learned_utility", {})
        win_signals = getattr(self, "win_signal", {})
        learned = float(utilities.get(utility_key, 0.0))
        count = int(getattr(self, "outcome_count", {}).get(utility_key, 0))
        win_signal = float(win_signals.get(utility_key, learned))
        loss_streak = int(getattr(self, "loss_streak", {}).get(utility_key, 0))
        has_performance_evidence = utility_key in utilities or utility_key in win_signals
        leaf_reliability = float(
            1.0 - np.exp(-max(0, count) / 3.0)
            if has_performance_evidence
            else 0.0
        )
        group_key = self._group_key(candidate.evidence_group)
        group_utilities = getattr(self, "group_learned_utility", {})
        group_win_signals = getattr(self, "group_win_signal", {})
        group_count = int(getattr(self, "group_outcome_count", {}).get(group_key, 0)) if group_key else 0
        group_learned = float(group_utilities.get(group_key, 0.0)) if group_key else 0.0
        group_win_signal = float(group_win_signals.get(group_key, group_learned)) if group_key else 0.0
        group_loss_streak = int(getattr(self, "group_loss_streak", {}).get(group_key, 0)) if group_key else 0
        policy_group_key = self._policy_group_key(candidate.policy_id, candidate.evidence_group)
        policy_group_loss_streak = int(
            getattr(self, "policy_group_loss_streak", {}).get(policy_group_key, 0)
        ) if policy_group_key else 0
        has_group_evidence = bool(
            group_key and (group_key in group_utilities or group_key in group_win_signals)
        )
        group_reliability = float(
            1.0 - np.exp(-max(0, group_count) / 3.0)
            if has_group_evidence
            else 0.0
        )
        evidence_reliability = float(
            1.0 - (1.0 - leaf_reliability) * (1.0 - group_reliability)
        )
        model_weight = float(1.0 - 0.65 * evidence_reliability)
        evidence_weight = float(0.20 + 0.55 * evidence_reliability)
        model_score = float(
            0.25 * _clamp(candidate.confidence, -1.0, 1.0)
            + 0.15 * _clamp(candidate.recent_utility, -1.0, 1.0)
            + 0.15 * _clamp(candidate.memory_match, -1.0, 1.0)
            + 0.15 * _clamp(candidate.proposal_margin, 0.0, 1.0)
            + 0.10 * _clamp(candidate.role_fit, -1.0, 1.0)
        )
        leaf_performance = float(
            0.75 * _clamp(learned, -1.0, 1.0)
            + 0.25 * _clamp(win_signal, -1.0, 1.0)
        )
        group_performance = float(
            0.75 * _clamp(group_learned, -1.0, 1.0)
            + 0.25 * _clamp(group_win_signal, -1.0, 1.0)
        )
        # A supported leaf owns the decision.  A sparse or unseen leaf backs
        # off continuously to its role-direction family instead of resetting
        # to zero evidence at every context/microstate boundary.
        leaf_weight = float(leaf_reliability)
        group_weight = float((1.0 - leaf_reliability) * group_reliability)
        evidence_weight_total = leaf_weight + group_weight
        performance_signal = float(
            (
                leaf_weight * leaf_performance
                + group_weight * group_performance
            ) / evidence_weight_total
            if evidence_weight_total > 1e-12
            else 0.0
        )
        effective_count = float(max(0, count) + 0.75 * max(0, group_count))
        exploration = float(getattr(self, "exploration_weight", 0.25) or 0.0) / np.sqrt(1.0 + effective_count)
        inflight_penalty = float(getattr(self, "inflight_penalty", 0.20) or 0.0)
        exposure_penalty = float(getattr(self, "exposure_penalty", 0.20) or 0.0)
        streak_penalty = float(getattr(self, "loss_streak_penalty", 0.12) or 0.0)
        effective_loss_streak = max(loss_streak, group_loss_streak)
        policy_group_streak_penalty = float(
            getattr(self, "policy_group_loss_streak_penalty", 0.0) or 0.0
        )
        cold_start_penalty = float(getattr(self, "cold_start_penalty", 0.60) or 0.0)
        cold_start_support = max(1, int(getattr(self, "cold_start_support", 8) or 8))
        candidate_support = max(
            0,
            int(getattr(candidate, "evidence_support", 0) or 0),
            int(getattr(candidate, "lifecycle_support", 0) or 0),
            int(group_count),
        )
        cold_start_cost = 0.0
        if (
            bool(getattr(self, "_selection_has_non_cold_candidate", False))
            and bool(getattr(candidate, "cold_start", False))
            and candidate_support < cold_start_support
        ):
            coldness = 1.0 - (float(candidate_support) / float(cold_start_support))
            cold_start_cost = cold_start_penalty * float(np.clip(coldness, 0.0, 1.0))
        return float(
            model_weight * model_score
            + evidence_weight * performance_signal
            + exploration
            - self.stress_penalty * _clamp(candidate.account_stress, 0.0, 1.0)
            - self.drawdown_penalty * _clamp(candidate.drawdown, 0.0, 1.0)
            - 0.03 * max(0, int(candidate.open_positions))
            # Do not saturate at three unresolved/correlated entries.  The old
            # cap made the 4th and 19th same-cluster proposal equally costly,
            # which allowed a winning streak to accumulate a large latent SL
            # cluster.  This remains a ranking term: the sole legal entry is
            # still selected by ``require_valid_selection`` at any score.
            - inflight_penalty * max(0, int(candidate.pending_outcomes))
            - exposure_penalty * max(0, int(candidate.correlated_exposure))
            - streak_penalty * min(3, max(0, effective_loss_streak))
            - policy_group_streak_penalty * min(3, max(0, policy_group_loss_streak))
            - cold_start_cost
        )

    @staticmethod
    def _utility_key(policy_id: str, context_key: str = "") -> str:
        policy_id = str(policy_id or "")
        context_key = str(context_key or "")
        return f"{context_key}::{policy_id}" if context_key else policy_id

    @staticmethod
    def _group_key(evidence_group: str = "") -> str:
        evidence_group = str(evidence_group or "").strip()
        return f"ROLE_GROUP::{evidence_group}" if evidence_group else ""

    @classmethod
    def _policy_group_key(cls, policy_id: str, evidence_group: str = "") -> str:
        policy_id = str(policy_id or "").strip()
        group_key = cls._group_key(evidence_group)
        return f"POLICY_GROUP::{policy_id}::{group_key}" if policy_id and group_key else ""

    def group_evidence(self, evidence_group: str) -> tuple[int, float, float, int]:
        """Return support, utility, win signal and loss streak for audit/risk."""
        key = self._group_key(evidence_group)
        if not key:
            return 0, 0.0, 0.0, 0
        return (
            int(getattr(self, "group_outcome_count", {}).get(key, 0)),
            float(getattr(self, "group_learned_utility", {}).get(key, 0.0)),
            float(getattr(self, "group_win_signal", {}).get(key, 0.0)),
            int(getattr(self, "group_loss_streak", {}).get(key, 0)),
        )

    def observe_outcome(
        self,
        policy_id: str,
        realized_utility: float,
        *,
        context_key: str = "",
        evidence_group: str = "",
    ) -> float:
        """Learn selector utility only for the proposal that MT5 executed.

        This changes future proposal ranking; it never masks an action or
        rewrites the selected policy output.
        """
        policy_id = str(policy_id or "")
        if not policy_id:
            return 0.0
        utility_key = self._utility_key(policy_id, context_key)
        utilities = getattr(self, "learned_utility", None)
        if not isinstance(utilities, dict):
            utilities = {}
            self.learned_utility = utilities
        counts = getattr(self, "outcome_count", None)
        if not isinstance(counts, dict):
            counts = {}
            self.outcome_count = counts
        win_signals = getattr(self, "win_signal", None)
        if not isinstance(win_signals, dict):
            win_signals = {}
            self.win_signal = win_signals
        loss_streaks = getattr(self, "loss_streak", None)
        if not isinstance(loss_streaks, dict):
            loss_streaks = {}
            self.loss_streak = loss_streaks
        alpha = float(np.clip(float(getattr(self, "utility_alpha", 0.35) or 0.35), 0.001, 1.0))
        win_alpha = float(np.clip(float(getattr(self, "win_signal_alpha", 0.35) or 0.35), 0.001, 1.0))
        value = _clamp(realized_utility, -2.0, 2.0)
        previous = float(utilities.get(utility_key, 0.0))
        updated = (1.0 - alpha) * previous + alpha * value
        previous_win_signal = float(win_signals.get(utility_key, previous))
        binary_win_signal = 1.0 if value > 0.0 else -1.0
        updated_win_signal = (1.0 - win_alpha) * previous_win_signal + win_alpha * binary_win_signal
        utilities[utility_key] = float(updated)
        win_signals[utility_key] = float(updated_win_signal)
        loss_streaks[utility_key] = 0 if value > 0.0 else int(loss_streaks.get(utility_key, 0)) + 1
        counts[utility_key] = int(counts.get(utility_key, 0)) + 1
        group_key = self._group_key(evidence_group)
        policy_group_key = self._policy_group_key(policy_id, evidence_group)
        if policy_group_key:
            policy_group_streaks = getattr(self, "policy_group_loss_streak", None)
            if not isinstance(policy_group_streaks, dict):
                policy_group_streaks = {}
                self.policy_group_loss_streak = policy_group_streaks
            policy_group_streaks[policy_group_key] = (
                0 if value > 0.0 else int(policy_group_streaks.get(policy_group_key, 0)) + 1
            )
        if group_key:
            group_utilities = getattr(self, "group_learned_utility", None)
            if not isinstance(group_utilities, dict):
                group_utilities = {}
                self.group_learned_utility = group_utilities
            group_counts = getattr(self, "group_outcome_count", None)
            if not isinstance(group_counts, dict):
                group_counts = {}
                self.group_outcome_count = group_counts
            group_win_signals = getattr(self, "group_win_signal", None)
            if not isinstance(group_win_signals, dict):
                group_win_signals = {}
                self.group_win_signal = group_win_signals
            group_loss_streaks = getattr(self, "group_loss_streak", None)
            if not isinstance(group_loss_streaks, dict):
                group_loss_streaks = {}
                self.group_loss_streak = group_loss_streaks
            group_previous = float(group_utilities.get(group_key, 0.0))
            group_updated = (1.0 - alpha) * group_previous + alpha * value
            group_previous_win = float(group_win_signals.get(group_key, group_previous))
            group_updated_win = (
                (1.0 - win_alpha) * group_previous_win
                + win_alpha * binary_win_signal
            )
            group_utilities[group_key] = float(group_updated)
            group_win_signals[group_key] = float(group_updated_win)
            group_loss_streaks[group_key] = (
                0 if value > 0.0 else int(group_loss_streaks.get(group_key, 0)) + 1
            )
            group_counts[group_key] = int(group_counts.get(group_key, 0)) + 1
        return float(updated)

    def _fallback(self, scores: dict[str, float], reason: str) -> PolicySelectionTrace:
        trace = PolicySelectionTrace(
            selected_policy_id=self.fallback_policy_id,
            fallback_policy_id=self.fallback_policy_id,
            scores=scores,
            reason=reason,
            used_fallback=True,
        )
        self.last_trace = trace
        return trace


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), lo, hi))
