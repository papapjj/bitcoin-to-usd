"""Fail-closed curriculum control for context-expert learning.

The curriculum separates three independent permissions:

* a target may be calculated and logged;
* its gradient may update the shared Direct-RNN AdamW graph;
* a trained component may affect an MT5 decision.

No stage advances from training loss.  Promotion requires immutable,
chronological OOS evidence and is explicit unless ``auto_promote`` is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np


CURRICULUM_SCHEMA_VERSION = 1
CURRICULUM_ARTIFACT_VERSION = 19
CURRICULUM_STAGES = (
    "representation",
    "timing_shadow",
    "timing_veto",
    "trade_hold",
    "action_ranking",
    "execution_quality",
    "lifecycle_actor",
    "tensor_rule",
    "risk_sizing",
)


def normalize_curriculum_stage(stage: object) -> str:
    value = str(stage or "representation").strip().lower().replace("-", "_")
    aliases = {
        "stage0": "representation", "stage_0": "representation",
        "stage1": "timing_shadow", "stage_1": "timing_shadow", "timing": "timing_shadow",
        "stage2": "timing_veto", "stage_2": "timing_veto", "veto": "timing_veto",
        "stage3": "trade_hold", "stage_3": "trade_hold",
        "stage4": "action_ranking", "stage_4": "action_ranking",
        "stage5": "execution_quality", "stage_5": "execution_quality",
        "stage6": "lifecycle_actor", "stage_6": "lifecycle_actor",
        "stage7": "tensor_rule", "stage_7": "tensor_rule",
        "stage8": "risk_sizing", "stage_8": "risk_sizing",
    }
    value = aliases.get(value, value)
    if value not in CURRICULUM_STAGES:
        raise ValueError(f"unknown curriculum stage: {stage!r}")
    return value


def curriculum_stage_index(stage: object) -> int:
    return CURRICULUM_STAGES.index(normalize_curriculum_stage(stage))


@dataclass(frozen=True)
class CurriculumPermissions:
    stage: str
    auxiliary: bool
    reconstruction: bool
    critic: bool
    timing_loss: bool
    timing_runtime: str
    trade_hold_loss: bool
    action_ranking_loss: bool
    execution_quality_loss: bool
    lifecycle_actor_scale: float
    tensor_rule_authority: str
    risk_sizing_learning: bool

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def curriculum_permissions(
    stage: object,
    *,
    lifecycle_actor_scale: float = 0.05,
    tensor_rule_authority: str = "residual",
) -> CurriculumPermissions:
    """Return the only authoritative gradient/runtime permission table."""
    name = normalize_curriculum_stage(stage)
    index = curriculum_stage_index(name)
    requested_rule_authority = str(tensor_rule_authority or "residual").strip().lower()
    if requested_rule_authority not in {"telemetry", "shadow", "residual", "primary"}:
        raise ValueError("invalid curriculum TensorRule authority")
    return CurriculumPermissions(
        stage=name,
        auxiliary=True,
        reconstruction=True,
        critic=index >= curriculum_stage_index("timing_shadow"),
        timing_loss=index >= curriculum_stage_index("timing_shadow"),
        timing_runtime=(
            "shadow" if index < curriculum_stage_index("timing_veto") else "veto_only"
        ),
        trade_hold_loss=index >= curriculum_stage_index("trade_hold"),
        action_ranking_loss=index >= curriculum_stage_index("action_ranking"),
        execution_quality_loss=index >= curriculum_stage_index("execution_quality"),
        lifecycle_actor_scale=(
            float(np.clip(lifecycle_actor_scale, 0.0, 1.0))
            if index >= curriculum_stage_index("lifecycle_actor") else 0.0
        ),
        tensor_rule_authority=(
            requested_rule_authority
            if index >= curriculum_stage_index("tensor_rule") else "shadow"
        ),
        risk_sizing_learning=index >= curriculum_stage_index("risk_sizing"),
    )


@dataclass(frozen=True)
class CurriculumEvidence:
    """Immutable summary of a paired chronological evaluation."""
    stage: str
    source_artifact_hash: str
    candidate_artifact_hash: str
    feature_fingerprint: str
    seed: int
    window_ids: tuple[str, ...]
    metrics: Mapping[str, float]
    real_ticks: bool = False
    reconciled: bool = False
    feature_parity: bool = False

    def normalized(self) -> "CurriculumEvidence":
        return CurriculumEvidence(
            stage=normalize_curriculum_stage(self.stage),
            source_artifact_hash=str(self.source_artifact_hash or ""),
            candidate_artifact_hash=str(self.candidate_artifact_hash or ""),
            feature_fingerprint=str(self.feature_fingerprint or ""),
            seed=int(self.seed),
            window_ids=tuple(str(value) for value in self.window_ids),
            metrics={str(key): float(value) for key, value in dict(self.metrics).items()},
            real_ticks=bool(self.real_ticks), reconciled=bool(self.reconciled),
            feature_parity=bool(self.feature_parity),
        )

    def fingerprint(self) -> str:
        item = self.normalized()
        payload = repr((
            item.stage, item.source_artifact_hash, item.candidate_artifact_hash,
            item.feature_fingerprint, item.seed, item.window_ids,
            tuple(sorted(item.metrics.items())), item.real_ticks,
            item.reconciled, item.feature_parity,
        )).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CurriculumGateResult:
    stage: str
    passed: bool
    failures: tuple[str, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage, "passed": self.passed,
            "failures": list(self.failures), "evidence_fingerprint": self.fingerprint,
        }


def _metric(metrics: Mapping[str, float], name: str, default: float = float("nan")) -> float:
    try:
        value = float(metrics.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def evaluate_curriculum_promotion(
    evidence: CurriculumEvidence,
    *,
    required_source_hash: str = "",
    min_chronological_blocks: int = 3,
) -> CurriculumGateResult:
    """Evaluate the current stage without consulting mutable train metrics."""
    item = evidence.normalized()
    metrics = item.metrics
    failures: list[str] = []
    if required_source_hash and item.source_artifact_hash != str(required_source_hash):
        failures.append("source_artifact_hash_mismatch")
    if not item.source_artifact_hash or not item.candidate_artifact_hash:
        failures.append("artifact_hash_missing")
    if item.source_artifact_hash == item.candidate_artifact_hash:
        failures.append("candidate_not_independent")
    if not item.feature_fingerprint:
        failures.append("feature_fingerprint_missing")
    if not item.reconciled:
        failures.append("execution_reconciliation_failed")
    if not item.feature_parity:
        failures.append("feature_parity_failed")
    blocks = max(len(set(item.window_ids)), int(_metric(metrics, "chronological_blocks", 0.0)))
    if blocks < int(min_chronological_blocks):
        failures.append("insufficient_chronological_blocks")
    if _metric(metrics, "nonfinite_count", 0.0) > 0.0:
        failures.append("nonfinite_training_state")

    stage = item.stage
    if stage == "representation":
        if _metric(metrics, "aux_validation_improvement", -1.0) <= 0.0:
            failures.append("aux_validation_not_improved")
        if _metric(metrics, "latent_variance", 0.0) < 0.02:
            failures.append("latent_collapse")
        if _metric(metrics, "train_validation_gap", 1.0) > 0.20:
            failures.append("representation_generalization_gap")
    elif stage == "timing_shadow":
        if _metric(metrics, "timing_independent_support", 0.0) < 90.0:
            failures.append("timing_support_below_90")
        if _metric(metrics, "timing_uplift_lcb_r", -9.0) < 0.0:
            failures.append("timing_uplift_lcb_negative")
        if _metric(metrics, "timing_top_bottom_delta_r", -9.0) < 0.10:
            failures.append("timing_ranking_delta_below_0_10r")
        if _metric(metrics, "timing_ece", 1.0) > 0.10:
            failures.append("timing_not_calibrated")
        mae_delta = _metric(metrics, "timing_mae_delta_fraction", 1.0)
        ttp_delta = _metric(metrics, "timing_ttp_delta_fraction", 1.0)
        if mae_delta > -0.10 and ttp_delta > -0.15:
            failures.append("timing_path_quality_not_improved")
    elif stage == "timing_veto":
        if _metric(metrics, "policy_origin_trades_per_block", 0.0) < 30.0:
            failures.append("veto_trade_support_below_30")
        if _metric(metrics, "timing_veto_net_uplift_lcb_r", -9.0) < 0.0:
            failures.append("veto_net_uplift_lcb_negative")
        if _metric(metrics, "avoided_loss_r", -9.0) <= _metric(metrics, "missed_opportunity_r", 9.0):
            failures.append("veto_cost_exceeds_avoided_loss")
        if _metric(metrics, "retained_trade_fraction", 0.0) < 0.50:
            failures.append("veto_trade_collapse")
    elif stage == "trade_hold":
        if _metric(metrics, "policy_expectancy_lcb_r", -9.0) < 0.0:
            failures.append("trade_hold_lcb_negative")
        if _metric(metrics, "trade_calibration_ece", 1.0) > 0.10:
            failures.append("trade_hold_not_calibrated")
        trade_fraction = _metric(metrics, "trade_fraction", -1.0)
        if not 0.05 <= trade_fraction <= 0.80:
            failures.append("trade_hold_collapse")
    elif stage == "action_ranking":
        if _metric(metrics, "action_scope_min_lcb_r", -9.0) < 0.0:
            failures.append("negative_action_scope")
        if _metric(metrics, "action_ranking_inversion_rate", 1.0) > 0.20:
            failures.append("action_ranking_unstable")
        if _metric(metrics, "max_action_fraction", 1.0) > 0.70:
            failures.append("action_collapse")
        if _metric(metrics, "action_independent_support", 0.0) < 180.0:
            failures.append("action_support_below_180")
    elif stage == "execution_quality":
        if _metric(metrics, "execution_expected_r_delta_lcb", -9.0) < 0.0:
            failures.append("execution_uplift_lcb_negative")
        if _metric(metrics, "fill_quality_delta", -9.0) < 0.0:
            failures.append("fill_quality_regressed")
        if _metric(metrics, "mae_delta_fraction", 1.0) > 0.0:
            failures.append("mae_regressed")
    elif stage == "lifecycle_actor":
        if not item.real_ticks:
            failures.append("real_tick_evidence_missing")
        if _metric(metrics, "paired_delta_lcb_r", -9.0) < 0.01:
            failures.append("actor_delta_lcb_below_0_01r")
        if _metric(metrics, "policy_expectancy_lcb_r", -9.0) < 0.0:
            failures.append("actor_policy_lcb_negative")
        if _metric(metrics, "retention_delta_lcb_r", -9.0) < -0.01:
            failures.append("timing_or_policy_retention_regressed")
        if _metric(metrics, "max_drawdown_pct", 100.0) > 10.0:
            failures.append("drawdown_above_10pct")
    elif stage == "tensor_rule":
        if not item.real_ticks:
            failures.append("real_tick_evidence_missing")
        if _metric(metrics, "rule_scope_min_lcb_r", -9.0) < 0.0:
            failures.append("negative_rule_scope")
        if _metric(metrics, "stable_rule_temporal_folds", 0.0) < 3.0:
            failures.append("rule_not_stable_in_three_folds")
        if _metric(metrics, "rule_ranking_sign_inversions", 9.0) > 0.0:
            failures.append("rule_ranking_sign_inversion")
        if _metric(metrics, "max_drawdown_pct", 100.0) > 10.0:
            failures.append("drawdown_above_10pct")
    else:
        failures.append("risk_sizing_is_terminal_stage")
    return CurriculumGateResult(stage, not failures, tuple(failures), item.fingerprint())


@dataclass
class CurriculumState:
    """Per-expert persistent stage and auditable promotion history."""
    schema_version: int = CURRICULUM_SCHEMA_VERSION
    enabled: bool = False
    stage: str = "representation"
    auto_promote: bool = False
    required_source_hash: str = ""
    promotion_history: list[dict[str, object]] = field(default_factory=list)
    rollback_history: list[dict[str, object]] = field(default_factory=list)
    timing_support: int = 0
    timing_calibration_error_ema: float = 1.0

    def ensure_compatible(self) -> None:
        self.schema_version = CURRICULUM_SCHEMA_VERSION
        self.enabled = bool(getattr(self, "enabled", False))
        self.stage = normalize_curriculum_stage(getattr(self, "stage", "representation"))
        self.auto_promote = bool(getattr(self, "auto_promote", False))
        self.required_source_hash = str(getattr(self, "required_source_hash", "") or "")
        self.promotion_history = list(getattr(self, "promotion_history", []) or [])[-128:]
        self.rollback_history = list(getattr(self, "rollback_history", []) or [])[-128:]
        self.timing_support = max(0, int(getattr(self, "timing_support", 0) or 0))
        self.timing_calibration_error_ema = float(np.clip(
            getattr(self, "timing_calibration_error_ema", 1.0), 0.0, 1.0
        ))

    def copy(self) -> "CurriculumState":
        return copy.deepcopy(self)

    def permissions(self, **kwargs: object) -> CurriculumPermissions:
        self.ensure_compatible()
        return curriculum_permissions(self.stage, **kwargs)

    def observe_timing(self, predicted: float, target: float) -> None:
        self.ensure_compatible()
        error = abs(float(np.clip(predicted, 0.0, 1.0)) - float(np.clip(target, 0.0, 1.0)))
        self.timing_support += 1
        decay = 0.95
        self.timing_calibration_error_ema = float(
            decay * self.timing_calibration_error_ema + (1.0 - decay) * error
        )

    def promote(self, evidence: CurriculumEvidence) -> CurriculumGateResult:
        self.ensure_compatible()
        if normalize_curriculum_stage(evidence.stage) != self.stage:
            return CurriculumGateResult(
                self.stage, False, ("evidence_stage_mismatch",), evidence.fingerprint()
            )
        result = evaluate_curriculum_promotion(
            evidence, required_source_hash=self.required_source_hash
        )
        record = result.as_dict()
        record["from_stage"] = self.stage
        if result.passed:
            index = curriculum_stage_index(self.stage)
            if index < len(CURRICULUM_STAGES) - 1:
                self.stage = CURRICULUM_STAGES[index + 1]
            record["to_stage"] = self.stage
        self.promotion_history.append(record)
        self.promotion_history = self.promotion_history[-128:]
        return result

    def rollback(self, stage: object, *, reason: str) -> None:
        self.ensure_compatible()
        target = normalize_curriculum_stage(stage)
        if curriculum_stage_index(target) > curriculum_stage_index(self.stage):
            raise ValueError("curriculum rollback cannot move forward")
        self.rollback_history.append({
            "from_stage": self.stage, "to_stage": target, "reason": str(reason or "manual")
        })
        self.rollback_history = self.rollback_history[-128:]
        self.stage = target

    def summary(self) -> dict[str, object]:
        self.ensure_compatible()
        return {
            "schema_version": self.schema_version, "enabled": self.enabled,
            "stage": self.stage, "stage_index": curriculum_stage_index(self.stage),
            "auto_promote": self.auto_promote,
            "timing_support": self.timing_support,
            "timing_calibration_error_ema": self.timing_calibration_error_ema,
            "promotion_count": sum(bool(row.get("passed", False)) for row in self.promotion_history),
            "last_promotion": copy.deepcopy(self.promotion_history[-1]) if self.promotion_history else None,
            "last_rollback": copy.deepcopy(self.rollback_history[-1]) if self.rollback_history else None,
        }


def timing_row_category(row: Mapping[str, object]) -> str:
    if bool(row.get("entry_preference_hard_negative", False)):
        return "hard_negative"
    components = tuple(row.get("entry_preference_components", ()) or ())
    if components and not any(bool(dict(item).get("filled", False)) for item in components):
        return "no_fill"
    target = float(np.clip(row.get("entry_timing_enter_now_target", 0.5) or 0.0, 0.0, 1.0))
    if target >= 0.65:
        return "enter_now"
    if target <= 0.35:
        return "wait"
    return "neutral"


TIMING_BATCH_FRACTIONS = {
    "enter_now": 0.25,
    "wait": 0.25,
    "neutral": 0.25,
    "no_fill": 0.15,
    "hard_negative": 0.10,
}


def select_timing_curriculum_rows(
    rows: Sequence[dict[str, object]], count: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Deterministically balance timing evidence, then backfill FIFO."""
    requested = max(0, int(count))
    buckets = {name: [] for name in TIMING_BATCH_FRACTIONS}
    for row in rows:
        buckets[timing_row_category(row)].append(row)
    quotas = {
        name: int(np.floor(requested * fraction))
        for name, fraction in TIMING_BATCH_FRACTIONS.items()
    }
    remainder = requested - sum(quotas.values())
    for name in TIMING_BATCH_FRACTIONS:
        if remainder <= 0:
            break
        quotas[name] += 1
        remainder -= 1
    selected: list[dict[str, object]] = []
    for name in TIMING_BATCH_FRACTIONS:
        take = min(quotas[name], len(buckets[name]))
        selected.extend(buckets[name][:take])
        buckets[name] = buckets[name][take:]
    if len(selected) < requested:
        selected_ids = {id(row) for row in selected}
        selected.extend(
            row for row in rows if id(row) not in selected_ids
        )
        selected = selected[:requested]
    selected_ids = {id(row) for row in selected}
    return selected, [row for row in rows if id(row) not in selected_ids]


def curriculum_auxiliary_mask(names: Iterable[str], stage: object) -> np.ndarray:
    """Route representation, timing and execution labels by stage."""
    stage_index = curriculum_stage_index(stage)
    mask = []
    for raw_name in names:
        name = str(raw_name)
        is_global_representation = name.startswith((
            "future_direction_", "return_quantile_", "realized_volatility_",
        ))
        is_timing = any(token in name for token in (
            "enter_now_probability", "timing_advantage_r", "optimal_delay_",
        ))
        is_execution = any(token in name for token in (
            "fill_probability", "time_to_positive", "expected_r", "sl_probability",
            "tp_probability", "timing_quality", "mfe_r", "mae_r",
        ))
        allowed = is_global_representation
        if stage_index >= curriculum_stage_index("timing_shadow"):
            allowed = allowed or is_timing
        if stage_index >= curriculum_stage_index("execution_quality"):
            allowed = allowed or is_execution
        mask.append(bool(allowed))
    return np.asarray(mask, dtype=bool)


def deterministic_reconstruction_mask(
    size: int, *, trace_id: str, fraction: float = 0.15
) -> np.ndarray:
    """Select a stable 10-20% masked-feature objective without RNG state."""
    width = max(0, int(size))
    result = np.zeros(width, dtype=bool)
    if width <= 0:
        return result
    count = max(1, min(width, int(round(width * float(np.clip(fraction, 0.10, 0.20))))))
    scored = []
    for index in range(width):
        digest = hashlib.sha256(f"{trace_id}|{index}".encode("utf-8")).digest()
        scored.append((digest, index))
    for _digest, index in sorted(scored)[:count]:
        result[index] = True
    return result


def timing_veto_decision(
    *,
    proposed_action: int,
    enter_probability: float,
    timing_advantage_r: float,
    support: int,
    calibration_error: float,
    probability_threshold: float = 0.65,
    advantage_lcb_threshold_r: float = 0.05,
    uncertainty_z: float = 1.28,
) -> tuple[int, dict[str, object]]:
    """Veto an entry to HOLD; never create or replace an entry action."""
    action = int(proposed_action)
    if action == 0 or not 1 <= action <= 6:
        return action, {"applied": False, "reason": "not_entry", "veto": False}
    support_value = max(0, int(support))
    uncertainty = float(np.clip(
        1.0 / np.sqrt(max(1.0, float(support_value))) + float(calibration_error),
        0.0, 2.0,
    ))
    lcb = float(timing_advantage_r) - float(uncertainty_z) * uncertainty
    enter = bool(
        support_value >= 30
        and float(enter_probability) >= float(probability_threshold)
        and lcb > float(advantage_lcb_threshold_r)
    )
    return (action if enter else 0), {
        "applied": True, "reason": "enter" if enter else "timing_veto",
        "veto": not enter, "proposed_action": action,
        "enter_probability": float(enter_probability),
        "timing_advantage_r": float(timing_advantage_r),
        "timing_advantage_lcb_r": lcb, "uncertainty": uncertainty,
        "support": support_value,
    }


__all__ = [
    "CURRICULUM_ARTIFACT_VERSION", "CURRICULUM_SCHEMA_VERSION", "CURRICULUM_STAGES",
    "CurriculumEvidence", "CurriculumGateResult", "CurriculumPermissions", "CurriculumState",
    "curriculum_auxiliary_mask", "curriculum_permissions", "curriculum_stage_index",
    "deterministic_reconstruction_mask", "evaluate_curriculum_promotion",
    "normalize_curriculum_stage", "select_timing_curriculum_rows", "timing_row_category",
    "timing_veto_decision",
]
