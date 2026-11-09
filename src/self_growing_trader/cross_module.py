from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .module_need import (
    ROLE_CADENCE_EXPLORATION_SPECIALIST,
    ROLE_CRITIC_SPECIALIST,
    ROLE_ENTRY_QUALITY_SPECIALIST,
    ROLE_EXIT_QUALITY_SPECIALIST,
    ROLE_MEMORY_CRITIC_SPECIALIST,
    ROLE_MEMORY_SPECIALIST,
    ROLE_POSITION_MANAGEMENT_SPECIALIST,
    ROLE_PREDICTOR_SPECIALIST,
    ROLE_RISK_GATE_SPECIALIST,
    ROLE_SELECTOR_SPECIALIST,
)
from .neural_graph import (
    EDGE_CRITIC,
    EDGE_FORWARD,
    EDGE_GATING,
    EDGE_LATERAL,
    EDGE_MEMORY,
    EDGE_META,
    EDGE_PLASTIC,
    EDGE_PREDICTIVE,
    EDGE_RECURRENT,
    EDGE_SELECTOR,
    REGION_CRITIC,
    REGION_ENCODER,
    REGION_HYPOTHESIS,
    REGION_MEMORY,
    REGION_META,
    REGION_OUTPUT,
    REGION_PREDICTOR,
    REGION_SELECTOR,
    NeuralGraph,
)


FLEXIBLE_CONNECTABLE_REGIONS: tuple[str, ...] = (
    REGION_ENCODER,
    REGION_MEMORY,
    REGION_HYPOTHESIS,
    REGION_PREDICTOR,
    REGION_CRITIC,
    REGION_SELECTOR,
    REGION_META,
    REGION_OUTPUT,
)


@dataclass(frozen=True)
class ConnectionIntent:
    src_region: str
    dst_region: str
    kind: str
    weight: float = 0.05
    gate: float = -2.5


ROLE_CONNECTION_GUIDE: dict[str, tuple[ConnectionIntent, ...]] = {
    ROLE_RISK_GATE_SPECIALIST: (
        ConnectionIntent(REGION_ENCODER, REGION_CRITIC, EDGE_FORWARD),
        ConnectionIntent(REGION_CRITIC, REGION_SELECTOR, EDGE_GATING),
        ConnectionIntent(REGION_CRITIC, REGION_OUTPUT, EDGE_GATING, weight=0.05, gate=-3.0),
    ),
    ROLE_CADENCE_EXPLORATION_SPECIALIST: (
        ConnectionIntent(REGION_ENCODER, REGION_META, EDGE_FORWARD),
        ConnectionIntent(REGION_META, REGION_SELECTOR, EDGE_GATING),
        # Exploration may only arrive at the action head as a weak candidate
        # gate/bias; it never bypasses the risk gate or action schema.
        ConnectionIntent(REGION_META, REGION_OUTPUT, EDGE_GATING, weight=0.04, gate=-3.2),
    ),
    ROLE_POSITION_MANAGEMENT_SPECIALIST: (
        ConnectionIntent(REGION_MEMORY, REGION_SELECTOR, EDGE_MEMORY),
        ConnectionIntent(REGION_SELECTOR, REGION_CRITIC, EDGE_CRITIC),
        ConnectionIntent(REGION_SELECTOR, REGION_OUTPUT, EDGE_GATING, weight=0.05, gate=-3.0),
        ConnectionIntent(REGION_SELECTOR, REGION_SELECTOR, EDGE_RECURRENT, weight=0.04, gate=-2.5),
    ),
    ROLE_MEMORY_SPECIALIST: (
        ConnectionIntent(REGION_ENCODER, REGION_MEMORY, EDGE_MEMORY),
        ConnectionIntent(REGION_MEMORY, REGION_PREDICTOR, EDGE_MEMORY),
        ConnectionIntent(REGION_MEMORY, REGION_CRITIC, EDGE_MEMORY),
        ConnectionIntent(REGION_MEMORY, REGION_SELECTOR, EDGE_GATING),
    ),
    ROLE_PREDICTOR_SPECIALIST: (
        ConnectionIntent(REGION_ENCODER, REGION_PREDICTOR, EDGE_PREDICTIVE),
        ConnectionIntent(REGION_PREDICTOR, REGION_PREDICTOR, EDGE_RECURRENT),
        ConnectionIntent(REGION_PREDICTOR, REGION_CRITIC, EDGE_CRITIC),
    ),
    ROLE_CRITIC_SPECIALIST: (
        ConnectionIntent(REGION_PREDICTOR, REGION_CRITIC, EDGE_CRITIC),
        ConnectionIntent(REGION_CRITIC, REGION_SELECTOR, EDGE_SELECTOR),
        ConnectionIntent(REGION_CRITIC, REGION_OUTPUT, EDGE_GATING, weight=0.05, gate=-3.0),
    ),
    ROLE_SELECTOR_SPECIALIST: (
        ConnectionIntent(REGION_SELECTOR, REGION_SELECTOR, EDGE_LATERAL),
        ConnectionIntent(REGION_SELECTOR, REGION_OUTPUT, EDGE_SELECTOR, weight=0.05, gate=-3.0),
    ),
    ROLE_ENTRY_QUALITY_SPECIALIST: (
        ConnectionIntent(REGION_ENCODER, REGION_SELECTOR, EDGE_FORWARD),
        ConnectionIntent(REGION_SELECTOR, REGION_CRITIC, EDGE_CRITIC),
    ),
    ROLE_EXIT_QUALITY_SPECIALIST: (
        ConnectionIntent(REGION_MEMORY, REGION_SELECTOR, EDGE_GATING),
        ConnectionIntent(REGION_SELECTOR, REGION_OUTPUT, EDGE_GATING, weight=0.05, gate=-3.0),
    ),
    ROLE_MEMORY_CRITIC_SPECIALIST: (
        ConnectionIntent(REGION_MEMORY, REGION_CRITIC, EDGE_CRITIC),
        ConnectionIntent(REGION_CRITIC, REGION_MEMORY, EDGE_GATING),
    ),
}


class CrossModuleConnector:
    """Whitelisted, weak candidate wiring between modules and M1 regions.

    The connector carries its own role/kind allow-list so deployment config can
    reduce connectivity without touching the EA bridge or graph primitive.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        allowed_edge_kinds: Iterable[str] | None = None,
        allowed_roles: Iterable[str] | None = None,
        flexible_connectivity: bool = True,
        flexible_max_edges: int = 12,
        flexible_peer_max_edges: int = 4,
    ) -> None:
        self.enabled = bool(enabled)
        self.allowed_edge_kinds = frozenset(allowed_edge_kinds or ())
        self.allowed_roles = frozenset(allowed_roles or ())
        self.flexible_connectivity = bool(flexible_connectivity)
        self.flexible_max_edges = max(0, int(flexible_max_edges))
        self.flexible_peer_max_edges = max(0, int(flexible_peer_max_edges))

    def wire_specialist(
        self,
        graph: NeuralGraph,
        *,
        module_id: str,
        role: str,
        candidate_id: str,
        need_reason: str,
        rng: np.random.Generator | None = None,
        max_edges: int = 8,
    ) -> list[str]:
        """Wire every role guide through the specialist anchor itself.

        An earlier foundation implementation selected both endpoints from M1
        regions whenever a matching region existed.  That made a specialist
        node decorative: metadata said a module had been wired while its node
        never participated in the graph.  Here an intent whose source or
        destination region matches the specialist anchor routes through that
        anchor; all other endpoints are M1 regions.  Every returned edge
        therefore connects the candidate specialist to M1 (or is its bounded
        recurrent self-loop), preserving whitelists and graph budgets.
        """
        rng = rng or np.random.default_rng(0)
        if not self.enabled:
            return []
        if self.allowed_roles and role not in self.allowed_roles:
            return []
        anchor = self._anchor(graph, module_id)
        if anchor is None:
            return []
        anchor_region = graph.neurons[anchor].region
        edge_ids: list[str] = []
        seen = {
            (edge.src, edge.dst, edge.kind)
            for edge in graph.edges.values()
            if edge.enabled
        }
        for intent in ROLE_CONNECTION_GUIDE.get(role, ()):
            if self.allowed_edge_kinds and intent.kind not in self.allowed_edge_kinds:
                continue
            src = anchor if intent.src_region == anchor_region else self._pick(graph, intent.src_region, module_id="m1", fallback_module=module_id)
            dst = anchor if intent.dst_region == anchor_region else self._pick(graph, intent.dst_region, module_id="m1", fallback_module=module_id)
            if src is None or dst is None:
                continue
            # A specialist must not create unrelated M1→M1 wiring.  A bounded
            # recurrent self edge is allowed for role state only.
            if src != anchor and dst != anchor:
                continue
            edge_id = self._add_candidate_edge(
                graph,
                src=src,
                dst=dst,
                kind=intent.kind,
                weight=intent.weight,
                gate=intent.gate,
                rng=rng,
                need_reason=need_reason,
                candidate_id=candidate_id,
                seen=seen,
            )
            if edge_id is not None:
                edge_ids.append(edge_id)
            if len(edge_ids) >= max_edges:
                break
        if self.flexible_connectivity and len(edge_ids) < max_edges:
            flexible_limit = min(max_edges, len(edge_ids) + self.flexible_max_edges)
            for intent in self._flexible_intents(anchor_region):
                if self.allowed_edge_kinds and intent.kind not in self.allowed_edge_kinds:
                    continue
                src = anchor if intent.src_region == anchor_region else self._pick(graph, intent.src_region, module_id="m1", fallback_module=module_id)
                dst = anchor if intent.dst_region == anchor_region else self._pick(graph, intent.dst_region, module_id="m1", fallback_module=module_id)
                if src is None or dst is None:
                    continue
                if src != anchor and dst != anchor:
                    continue
                edge_id = self._add_candidate_edge(
                    graph,
                    src=src,
                    dst=dst,
                    kind=intent.kind,
                    weight=intent.weight,
                    gate=intent.gate,
                    rng=rng,
                    need_reason=need_reason,
                    candidate_id=candidate_id,
                    seen=seen,
                )
                if edge_id is not None:
                    edge_ids.append(edge_id)
                if len(edge_ids) >= flexible_limit:
                    break
        if self.flexible_connectivity and self.flexible_peer_max_edges > 0 and len(edge_ids) < max_edges:
            peer_limit = min(max_edges, len(edge_ids) + self.flexible_peer_max_edges)
            for peer in self._peer_candidates(graph, anchor, module_id):
                for src, dst in ((anchor, peer), (peer, anchor)):
                    kind = _edge_kind_for(graph.neurons[src].region, graph.neurons[dst].region)
                    if self.allowed_edge_kinds and kind not in self.allowed_edge_kinds:
                        continue
                    edge_id = self._add_candidate_edge(
                        graph,
                        src=src,
                        dst=dst,
                        kind=kind,
                        weight=0.03,
                        gate=-3.25,
                        rng=rng,
                        need_reason=need_reason,
                        candidate_id=candidate_id,
                        seen=seen,
                    )
                    if edge_id is not None:
                        edge_ids.append(edge_id)
                    if len(edge_ids) >= peer_limit:
                        break
                if len(edge_ids) >= peer_limit:
                    break
        graph.assert_invariants()
        return edge_ids

    @staticmethod
    def _anchor(graph: NeuralGraph, module_id: str) -> str | None:
        candidates = [node.id for node in graph.neurons.values() if node.module_id == module_id]
        if not candidates:
            return None
        return max(candidates, key=lambda node_id: (graph.neurons[node_id].utility, graph.neurons[node_id].activation))

    @staticmethod
    def _pick(graph: NeuralGraph, region: str, *, module_id: str, fallback_module: str) -> str | None:
        candidates = [node.id for node in graph.neurons.values() if node.region == region and node.module_id == module_id]
        if not candidates:
            candidates = [node.id for node in graph.neurons.values() if node.region == region and node.module_id == fallback_module]
        if not candidates:
            return None
        return max(candidates, key=lambda node_id: graph.neurons[node_id].utility)

    @staticmethod
    def _peer_candidates(graph: NeuralGraph, anchor: str, module_id: str) -> list[str]:
        anchor_region = graph.neurons[anchor].region
        candidates = [
            node.id
            for node in graph.neurons.values()
            if node.id != anchor
            and node.module_id not in {None, "m1", module_id}
            and node.region in FLEXIBLE_CONNECTABLE_REGIONS
        ]
        return sorted(
            candidates,
            key=lambda node_id: (
                graph.neurons[node_id].region == anchor_region,
                graph.neurons[node_id].utility,
                graph.neurons[node_id].activation,
            ),
            reverse=True,
        )

    def _add_candidate_edge(
        self,
        graph: NeuralGraph,
        *,
        src: str,
        dst: str,
        kind: str,
        weight: float,
        gate: float,
        rng: np.random.Generator,
        need_reason: str,
        candidate_id: str,
        seen: set[tuple[str, str, str]],
    ) -> str | None:
        key = (src, dst, kind)
        if key in seen:
            return None
        try:
            edge_id = graph.add_connection(
                src,
                dst,
                weight=float(np.clip(rng.normal(0.0, weight / 2.0), -weight, weight)),
                gate=float(min(gate, -0.25)),
                kind=kind,
                created_by="graph_development",
                need_reason=need_reason,
                candidate_id=candidate_id,
                lifecycle="candidate",
            )
        except ValueError:
            return None
        seen.add(key)
        return edge_id

    @staticmethod
    def _flexible_intents(anchor_region: str) -> tuple[ConnectionIntent, ...]:
        intents: list[ConnectionIntent] = [
            ConnectionIntent(anchor_region, anchor_region, EDGE_RECURRENT, weight=0.03, gate=-3.0),
        ]
        for region in FLEXIBLE_CONNECTABLE_REGIONS:
            if region != anchor_region:
                intents.append(ConnectionIntent(region, anchor_region, _edge_kind_for(region, anchor_region), weight=0.035, gate=-3.0))
                intents.append(ConnectionIntent(anchor_region, region, _edge_kind_for(anchor_region, region), weight=0.035, gate=-3.0))
        return tuple(intents)


def _edge_kind_for(src_region: str, dst_region: str) -> str:
    if src_region == dst_region:
        return EDGE_RECURRENT
    if dst_region == REGION_OUTPUT:
        return EDGE_GATING
    if dst_region == REGION_MEMORY or src_region == REGION_MEMORY:
        return EDGE_MEMORY
    if dst_region == REGION_PREDICTOR or src_region == REGION_PREDICTOR:
        return EDGE_PREDICTIVE
    if dst_region == REGION_CRITIC:
        return EDGE_CRITIC
    if dst_region == REGION_SELECTOR:
        return EDGE_SELECTOR
    if dst_region == REGION_META or src_region == REGION_META:
        return EDGE_META
    return EDGE_PLASTIC
