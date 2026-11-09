from __future__ import annotations

"""Lineage-governed child supernetwork lifecycle.

Child networks are deliberately *not* routed into live action selection while
in ``candidate`` state.  They become selectable only after the supplied replay
and MT5 validators both pass through ``AdaptiveSuperNetwork.validate_candidate``.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
import copy

import numpy as np

from .adaptive_supernetwork import AdaptiveCandidate, AdaptiveSuperNetwork
from .graph_metrics import graph_metrics
from .lineage import LineageRegistry, NetworkLineage
from .module_need import ModuleErrorCluster, ModuleNeedSignal, ModuleRolePrior
from .supernetwork_blueprint import (
    BlueprintGenerator,
    SuperNetworkBlueprint,
    reject_if_clone_like,
    spawn_supernetwork_from_blueprint,
)


@dataclass(frozen=True)
class ChildSelectionTrace:
    """Trace nội bộ; child candidate không bao giờ được route live.

    ``scores`` remains a compact audit field for callers that display selector
    telemetry. It never denotes a routing competition between independent
    policies; the sole selectable identity is the owner M1 graph.
    """
    selected_policy_id: str
    fallback_policy_id: str
    reason: str
    used_fallback: bool
    scores: dict[str, float] = field(default_factory=dict)



@dataclass
class ChildNetworkRecord:
    network_id: str
    blueprint: SuperNetworkBlueprint
    network: AdaptiveSuperNetwork
    status: str = "candidate"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warmup_rows: int = 0
    mt5_promoted: bool = False
    reject_reasons: list[str] = field(default_factory=list)

    @property
    def selectable(self) -> bool:
        return self.status == "promoted" and self.mt5_promoted and not self.network.lineage.archived


@dataclass
class ChildSupernetworkManager:
    root_network_id: str
    root_lineage: NetworkLineage
    registry: LineageRegistry
    role_prior: ModuleRolePrior = field(default_factory=ModuleRolePrior)
    children: dict[str, ChildNetworkRecord] = field(default_factory=dict)
    max_children: int | None = None
    max_generation: int | None = None
    max_children_per_network: int | None = None
    diversity_threshold: float = 0.05

    @classmethod
    def root(cls, *, network_id: str, max_alive_networks: int | None = None, max_children: int | None = None, max_generation: int | None = None, max_children_per_network: int | None = None, diversity_threshold: float = 0.05) -> "ChildSupernetworkManager":
        lineage = NetworkLineage(
            network_id=network_id,
            parent_id=None,
            root_id=network_id,
            generation=0,
            blueprint_id=None,
            role="m1_graph",
        )
        registry = LineageRegistry(max_alive_networks=None if max_alive_networks is None or int(max_alive_networks) <= 0 else int(max_alive_networks))
        registry.add(lineage)
        return cls(
            root_network_id=network_id,
            root_lineage=lineage,
            registry=registry,
            max_children=None if max_children is None or int(max_children) <= 0 else int(max_children),
            max_generation=None if max_generation is None or int(max_generation) <= 0 else int(max_generation),
            max_children_per_network=None if max_children_per_network is None or int(max_children_per_network) <= 0 else int(max_children_per_network),
            diversity_threshold=float(np.clip(diversity_threshold, 0.0, 1.0)),
        )

    def copy(self) -> "ChildSupernetworkManager":
        return copy.deepcopy(self)

    def on_root_promotion(self) -> None:
        self.root_lineage.record_promotion()

    def on_root_rejection(self) -> None:
        self.root_lineage.record_failure()

    def spawn_candidate(
        self,
        *,
        parent_graph,
        input_size: int,
        output_size: int,
        graph_summary: dict[str, float] | None = None,
        memory_summary: dict[str, float] | None = None,
        failure_clusters: Iterable[ModuleErrorCluster] = (),
        winning_causal_patterns: Iterable[dict[str, object]] = (),
        seed: int | None = None,
        journal_path: str | Path | None = None,
        warmup_max_items: int = 200,
    ) -> ChildNetworkRecord | None:
        """Create a graph+memory+controller isolated candidate child.

        It returns ``None`` when lineage/budget/diversity gates say no.  This
        keeps child generation bounded rather than turning every error cluster
        into a new network.
        """
        if (self.max_children is not None and len(self.children) >= self.max_children) or not self.root_lineage.can_spawn_child(
            max_generation=self.max_generation,
            max_children_per_network=self.max_children_per_network,
        ):
            return None
        clusters = list(failure_clusters)
        generator = BlueprintGenerator(role_prior=self.role_prior)
        blueprint_id = f"bp-{self.root_network_id}-{len(self.children) + 1}"
        blueprint = generator.generate(
            blueprint_id=blueprint_id,
            parent_id=self.root_network_id,
            input_size=input_size,
            output_size=output_size,
            graph_summary=graph_summary or graph_metrics(parent_graph),
            memory_summary=memory_summary or {},
            failure_clusters=clusters,
            winning_causal_patterns=winning_causal_patterns,
        )
        network_id = f"child-{self.root_network_id}-{len(self.children) + 1}"
        try:
            child_lineage = self.root_lineage.make_child(
                network_id=network_id,
                blueprint_id=blueprint.blueprint_id,
                role=blueprint.role,
                max_generation=self.max_generation,
                max_children_per_network=self.max_children_per_network,
            )
            self.registry.add(child_lineage)
        except ValueError:
            return None
        child_graph = spawn_supernetwork_from_blueprint(
            blueprint,
            seed=int(seed if seed is not None else np.random.default_rng().integers(1, 2**31 - 1)),
        )
        try:
            reject_if_clone_like(parent_graph, child_graph, min_diversity=self.diversity_threshold)
        except ValueError:
            child_lineage.archived = True
            return None
        child = AdaptiveSuperNetwork(
            network_id=network_id,
            graph=child_graph,
            lineage=child_lineage,
            role_prior=copy.deepcopy(self.role_prior),
            lineage_registry=self.registry,
        )
        record = ChildNetworkRecord(network_id=network_id, blueprint=blueprint, network=child)
        if journal_path:
            record.warmup_rows = child.warmup_from_journal(journal_path, max_items=warmup_max_items)
        self.children[network_id] = record
        return record

    def validate_child(
        self,
        network_id: str,
        *,
        replay_validator: Callable[[object], float | tuple[bool, float] | None] | None,
        mt5_validator: Callable[[object], float | tuple[bool, float, bool] | None] | None,
        trial_history_path: str | Path | None = None,
    ) -> bool:
        record = self.children[network_id]
        candidate: AdaptiveCandidate = record.network.copy_candidate(
            candidate_id=f"{network_id}:validation",
            blueprint=record.blueprint,
            parent_fitness=None,
            module_need_reason=record.blueprint.objective,
        )
        result = record.network.validate_candidate(
            candidate,
            replay_validator=replay_validator,
            mt5_validator=mt5_validator,
            trial_history_path=trial_history_path,
        )
        if result.promoted:
            record.status = "promoted"
            record.mt5_promoted = True
            self.role_prior.record_outcome(record.blueprint.role, promoted=True, cluster_id=record.blueprint.objective)
            return True
        record.status = "rejected"
        record.reject_reasons.extend(result.reject_reasons)
        self.role_prior.record_outcome(record.blueprint.role, promoted=False, cluster_id=record.blueprint.objective)
        return False

    def select(
        self,
        *,
        memory_match: float,
        account_stress: float,
        drawdown: float,
        open_positions: int,
        baseline_confidence: float,
        baseline_utility: float,
        child_scores: dict[str, tuple[float, float]] | None = None,
    ) -> ChildSelectionTrace:
        """Không chọn child như một policy khác.

        Child network chỉ là candidate cấu trúc bên trong *cùng một expert*;
        nó phải được xác nhận rồi hợp nhất vào graph của owner. Vì vậy mọi
        request live vẫn có đúng một policy owner.
        """
        del memory_match, account_stress, drawdown, open_positions, child_scores
        # ``m1`` is the one graph owned by this expert. Children remain
        # candidate structures until validation and merge; they are never a
        # parallel live policy. The score is telemetry only.
        owner_score = float(baseline_utility) + 0.01 * float(baseline_confidence)
        return ChildSelectionTrace(
            selected_policy_id="m1",
            fallback_policy_id="m1",
            reason="child_candidate_khong_duoc_route_live",
            used_fallback=False,
            scores={"m1": owner_score},
        )

    def metrics(self) -> dict[str, float]:
        total = len(self.children)
        promoted = sum(1 for child in self.children.values() if child.selectable)
        candidate = sum(1 for child in self.children.values() if child.status == "candidate")
        return {
            "child_spawned": float(total),
            "child_promoted": float(promoted),
            "child_candidate": float(candidate),
            "child_survival_rate": float(promoted / max(1, total)),
            "alive_networks": float(self.registry.alive_count()),
        }
