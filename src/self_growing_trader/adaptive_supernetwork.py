from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .graph_development import GraphDevelopmentController, GraphStructuralAction
from .graph_metrics import graph_metrics
from .graph_plasticity import GraphCreditSignal, GraphPlasticityEngine
from .lineage import LineageRegistry, NetworkLineage
from .module_need import ModuleNeedSignal, ModuleRolePrior, module_spec_for_role
from .neural_graph import NeuralGraph
from .supernetwork_blueprint import (
    SuperNetworkBlueprint,
    blueprint_from_module_spec,
    reject_if_clone_like,
    spawn_supernetwork_from_blueprint,
)
from .supernetwork_runtime import (
    CandidatePromotionGate,
    CandidateValidationResult,
    SupernetworkTrialRecord,
    load_valid_journal_transitions,
    write_trial_history,
)


@dataclass
class AdaptiveCandidate:
    candidate_id: str
    parent_network_id: str
    graph: NeuralGraph
    lineage: NetworkLineage
    blueprint: SuperNetworkBlueprint | None = None
    parent_fitness: float | None = None
    structural_diff: dict[str, object] = field(default_factory=dict)
    module_need_reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted: bool = False
    reject_reasons: list[str] = field(default_factory=list)
    validation: CandidateValidationResult | None = None


class AdaptiveSuperNetwork:
    """A standalone child policy with candidate-only topology development."""

    def __init__(
        self,
        *,
        network_id: str,
        graph: NeuralGraph,
        lineage: NetworkLineage,
        role_prior: ModuleRolePrior | None = None,
        memory_graph: NeuralGraph | None = None,
        development_controller: GraphDevelopmentController | None = None,
        plasticity_engine: GraphPlasticityEngine | None = None,
        lineage_registry: LineageRegistry | None = None,
    ) -> None:
        self.network_id = network_id
        self.graph = graph
        self.memory_graph = memory_graph if memory_graph is not None else NeuralGraph()
        self.lineage = lineage
        self.role_prior = role_prior or ModuleRolePrior()
        self.development_controller = development_controller or GraphDevelopmentController(role_prior=self.role_prior)
        self.plasticity_engine = plasticity_engine or GraphPlasticityEngine()
        self.lineage_registry = lineage_registry
        self.utility_history: list[float] = []
        self.promotion_history: list[dict[str, object]] = []
        self.trial_history: list[SupernetworkTrialRecord] = []
        self._candidate: AdaptiveCandidate | None = None

    def act(self, observation: np.ndarray, *, steps: int = 2) -> np.ndarray:
        return self.graph.activate(observation, steps=steps)

    def observe_transition(
        self,
        *,
        reward: float,
        cost: float,
        prediction_error: float,
        active_edge_ids: list[str] | None = None,
        active_neuron_ids: list[str] | None = None,
        lifecycle_tags: tuple[str, ...] = (),
        account_stress: float = 0.0,
        candidate_mode: bool = True,
        update_mode: str = "candidate",
    ) -> dict[str, float]:
        report = self.plasticity_engine.apply_credit(
            self.graph,
            GraphCreditSignal(
                reward=reward,
                cost=cost,
                prediction_error=prediction_error,
                account_stress=account_stress,
                lifecycle_tags=lifecycle_tags,
            ),
            active_edge_ids=active_edge_ids or (),
            active_neuron_ids=active_neuron_ids or (),
            candidate_mode=candidate_mode,
            update_mode=update_mode,
        )
        utility = report.credit
        self.utility_history.append(utility)
        return {
            "utility": utility,
            "prediction_error": float(prediction_error),
            "updated_edges": float(report.updated_edges),
            "updated_neurons": float(report.updated_neurons),
            "disabled_edges": float(report.disabled_edges),
            "credit_attribution_edges": float(len(report.active_edge_ids)),
            "credit_attribution_neurons": float(len(report.active_neuron_ids)),
        }

    def develop(self, signal: ModuleNeedSignal, *, cluster_id: str | None = None) -> dict[str, object]:
        scores = signal.scores()
        action = self.development_controller.choose_action(signal, cluster_id=cluster_id)
        role = action.role or self.role_prior.choose_role(scores, cluster_id=cluster_id)
        spec = module_spec_for_role(role, reason=cluster_id or role)
        return {
            "role": role,
            "scores": scores,
            "module_spec": spec,
            "structural_action": action,
            "module_need_reason": cluster_id or role,
        }

    def propose_development_candidate(
        self,
        signal: ModuleNeedSignal,
        *,
        candidate_id: str,
        cluster_id: str | None = None,
        parent_fitness: float | None = None,
    ) -> AdaptiveCandidate:
        """Apply structural work only to an isolated candidate graph."""
        proposal = self.develop(signal, cluster_id=cluster_id)
        candidate = self.copy_candidate(
            candidate_id=candidate_id,
            parent_fitness=parent_fitness,
            module_need_reason=str(proposal["module_need_reason"]),
        )
        before = graph_metrics(candidate.graph)
        event = self.development_controller.apply_action(candidate.graph, proposal["structural_action"])
        after = graph_metrics(candidate.graph)
        candidate.structural_diff = {
            "action": event.action,
            "target": event.target,
            "reason": event.reason,
            "details": event.details,
            "before": before,
            "after": after,
        }
        return candidate

    def copy_candidate(
        self,
        *,
        candidate_id: str,
        blueprint: SuperNetworkBlueprint | None = None,
        parent_fitness: float | None = None,
        module_need_reason: str = "",
    ) -> AdaptiveCandidate:
        candidate = AdaptiveCandidate(
            candidate_id=candidate_id,
            parent_network_id=self.network_id,
            graph=self.graph.copy(),
            lineage=copy.deepcopy(self.lineage),
            blueprint=blueprint,
            parent_fitness=parent_fitness,
            module_need_reason=module_need_reason,
        )
        self._candidate = candidate
        return candidate

    def validate_candidate(
        self,
        candidate: AdaptiveCandidate,
        *,
        replay_validator: Callable[[NeuralGraph], float | tuple[bool, float] | None] | None,
        mt5_validator: Callable[[NeuralGraph], float | tuple[bool, float, bool] | None] | None,
        trial_history_path: str | Path | None = None,
    ) -> CandidateValidationResult:
        if candidate.parent_network_id != self.network_id:
            raise ValueError("candidate does not belong to this network")
        candidate.graph.assert_invariants()
        gate = CandidatePromotionGate()
        result = gate.validate(
            candidate_id=candidate.candidate_id,
            candidate=candidate.graph,
            parent_score=candidate.parent_fitness,
            replay_validator=replay_validator,
            mt5_validator=mt5_validator,
        )
        candidate.validation = result
        record = SupernetworkTrialRecord(
            candidate_id=candidate.candidate_id,
            parent_policy_id=self.network_id,
            structural_diff=dict(candidate.structural_diff),
            module_need_reason=candidate.module_need_reason,
            validation=result,
            graph_metrics=graph_metrics(candidate.graph),
            created_at=candidate.created_at,
        )
        self.trial_history.append(record)
        if trial_history_path:
            write_trial_history(trial_history_path, record)
        if result.promoted:
            self.promote_candidate(candidate, score=result.mt5_score)
        else:
            self.rollback_candidate(candidate, reason=",".join(result.reject_reasons) or "candidate_not_promoted")
        return result

    def promote_candidate(self, candidate: AdaptiveCandidate, *, score: float | None = None) -> None:
        if candidate.parent_network_id != self.network_id:
            raise ValueError("candidate does not belong to this network")
        self.graph = candidate.graph
        self.lineage.record_promotion()
        self.plasticity_engine.consolidate_high_utility(self.graph, promotions=self.lineage.promotion_count)
        candidate.promoted = True
        self.promotion_history.append({
            "candidate_id": candidate.candidate_id,
            "score": score,
            "blueprint_id": candidate.blueprint.blueprint_id if candidate.blueprint else None,
            "structural_diff": dict(candidate.structural_diff),
        })
        self._candidate = None

    def rollback_candidate(self, candidate: AdaptiveCandidate, *, reason: str) -> None:
        if candidate.parent_network_id != self.network_id:
            raise ValueError("candidate does not belong to this network")
        candidate.reject_reasons.append(reason)
        self.lineage.record_failure()
        self.lineage.archive_if_stale()
        self._candidate = None

    def warmup_from_journal(self, journal_path: str | Path, *, max_items: int = 200) -> int:
        """Warm a child through filtered teaching signals, never copied memory."""
        rows = load_valid_journal_transitions(journal_path, max_items=max_items)
        for row in rows:
            trace = row.get("trace", {}) if isinstance(row.get("trace", {}), dict) else {}
            self.observe_transition(
                reward=float(row.get("reward", 0.0) or 0.0),
                cost=float(row.get("cost", 0.0) or 0.0),
                prediction_error=float(row.get("prediction_error", trace.get("prediction_error", 0.0)) or 0.0),
                active_edge_ids=[str(item) for item in trace.get("active_edge_ids", []) or []],
                active_neuron_ids=[str(item) for item in trace.get("active_neuron_ids", []) or []],
                lifecycle_tags=tuple(str(item) for item in trace.get("lifecycle_tags", []) or []),
                candidate_mode=True,
                update_mode="replay",
            )
        return len(rows)

    def spawn_child(
        self,
        *,
        network_id: str,
        blueprint_id: str,
        role: str,
        input_size: int,
        output_size: int,
        seed: int,
        hidden_size: int = 12,
        min_diversity: float = 0.05,
        max_generation: int = 3,
        max_children_per_network: int = 3,
    ) -> "AdaptiveSuperNetwork":
        child_lineage = self.lineage.make_child(
            network_id=network_id,
            blueprint_id=blueprint_id,
            role=role,
            max_generation=max_generation,
            max_children_per_network=max_children_per_network,
        )
        if self.lineage_registry is not None:
            self.lineage_registry.add(child_lineage)
        spec = module_spec_for_role(role, reason=f"spawned_by:{self.network_id}")
        blueprint = blueprint_from_module_spec(
            spec,
            blueprint_id=blueprint_id,
            parent_id=self.network_id,
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
        )
        child_graph = spawn_supernetwork_from_blueprint(blueprint, seed=seed)
        reject_if_clone_like(self.graph, child_graph, min_diversity=min_diversity)
        return AdaptiveSuperNetwork(
            network_id=network_id,
            graph=child_graph,
            lineage=child_lineage,
            role_prior=copy.deepcopy(self.role_prior),
            development_controller=GraphDevelopmentController(role_prior=copy.deepcopy(self.role_prior)),
            plasticity_engine=GraphPlasticityEngine(),
            lineage_registry=self.lineage_registry,
        )

    def metrics(self) -> dict[str, float]:
        result = graph_metrics(self.graph)
        result.update({
            "utility_history_size": float(len(self.utility_history)),
            "promotion_count": float(self.lineage.promotion_count),
            "failure_count": float(self.lineage.failure_count),
            "trial_count": float(len(self.trial_history)),
        })
        return result

    def copy(self) -> "AdaptiveSuperNetwork":
        copied = AdaptiveSuperNetwork(
            network_id=self.network_id,
            graph=self.graph.copy(),
            lineage=copy.deepcopy(self.lineage),
            role_prior=copy.deepcopy(self.role_prior),
            memory_graph=self.memory_graph.copy(),
            development_controller=copy.deepcopy(self.development_controller),
            plasticity_engine=copy.deepcopy(self.plasticity_engine),
            lineage_registry=self.lineage_registry,
        )
        copied.utility_history = list(self.utility_history)
        copied.promotion_history = copy.deepcopy(self.promotion_history)
        copied.trial_history = copy.deepcopy(self.trial_history)
        return copied
