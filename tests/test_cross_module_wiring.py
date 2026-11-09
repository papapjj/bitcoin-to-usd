from __future__ import annotations

import numpy as np

from self_growing_trader.cross_module import CrossModuleConnector
from self_growing_trader.module_need import ROLE_MEMORY_SPECIALIST, ROLE_RISK_GATE_SPECIALIST
from self_growing_trader.neural_graph import EDGE_RECURRENT, REGION_CRITIC, REGION_MEMORY
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def _graph():
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=20, hidden_size=5, seed=515)
    return policy.modules[0].m1_component.graph.copy()


def test_specialist_wiring_always_connects_specialist_anchor() -> None:
    graph = _graph()
    module_id = "specialist:memory_specialist:c1"
    anchor = graph.add_neuron(REGION_MEMORY, module_id=module_id, lifecycle="candidate")
    edges = CrossModuleConnector().wire_specialist(
        graph,
        module_id=module_id,
        role=ROLE_MEMORY_SPECIALIST,
        candidate_id="c1",
        need_reason="low_memory_match",
        rng=np.random.default_rng(1),
    )
    assert edges
    for edge_id in edges:
        edge = graph.edges[edge_id]
        assert anchor in {edge.src, edge.dst}
        assert edge.candidate_id == "c1"
        assert edge.need_reason == "low_memory_match"
    graph.assert_invariants(output_size=20)


def test_risk_specialist_has_encoder_input_and_bounded_output_gate() -> None:
    graph = _graph()
    module_id = "specialist:risk_gate_specialist:c2"
    anchor = graph.add_neuron(REGION_CRITIC, module_id=module_id, lifecycle="candidate")
    edges = CrossModuleConnector().wire_specialist(
        graph,
        module_id=module_id,
        role=ROLE_RISK_GATE_SPECIALIST,
        candidate_id="c2",
        need_reason="stress_entry_loss",
        rng=np.random.default_rng(2),
    )
    assert edges
    attached = [graph.edges[edge_id] for edge_id in edges]
    assert any(edge.dst == anchor for edge in attached)
    assert any(edge.src == anchor for edge in attached)
    assert all(abs(edge.weight) <= graph.max_action_bias_weight for edge in attached if edge.dst_module_id == "m1" and graph.neurons[edge.dst].region == "output")


def test_every_documented_specialist_role_has_safe_anchor_wiring() -> None:
    from self_growing_trader.cross_module import ROLE_CONNECTION_GUIDE
    from self_growing_trader.graph_supernetwork_runtime import ROLE_REGION

    graph = _graph()
    connector = CrossModuleConnector()
    for index, (role, region) in enumerate(ROLE_REGION.items()):
        module_id = f"specialist:{role}:all-{index}"
        anchor = graph.add_neuron(region, module_id=module_id, lifecycle="candidate")
        edge_ids = connector.wire_specialist(
            graph,
            module_id=module_id,
            role=role,
            candidate_id=f"all-{index}",
            need_reason="test_role_wiring",
            rng=np.random.default_rng(index + 10),
        )
        # Every role is represented in the connection guide and has at least
        # one endpoint touching its specialist anchor.
        assert role in ROLE_CONNECTION_GUIDE
        assert edge_ids
        assert all(anchor in {graph.edges[edge_id].src, graph.edges[edge_id].dst} for edge_id in edge_ids)
    graph.assert_invariants(output_size=20)


def test_flexible_wiring_fans_new_specialist_into_learned_m1_regions() -> None:
    graph = _graph()
    module_id = "specialist:memory_specialist:flex"
    anchor = graph.add_neuron(REGION_MEMORY, module_id=module_id, lifecycle="candidate")
    edges = CrossModuleConnector(flexible_connectivity=True, flexible_max_edges=12).wire_specialist(
        graph,
        module_id=module_id,
        role=ROLE_MEMORY_SPECIALIST,
        candidate_id="flex",
        need_reason="low_memory_match",
        rng=np.random.default_rng(7),
        max_edges=12,
    )
    attached = [graph.edges[edge_id] for edge_id in edges]
    touched_regions = {
        graph.neurons[edge.src].region if edge.src != anchor else graph.neurons[edge.dst].region
        for edge in attached
    }
    assert len(attached) > len([edge for edge in attached if edge.kind == "memory"])
    assert {REGION_MEMORY, "encoder", "predictor", "critic", "selector"}.issubset(touched_regions)
    assert any(edge.src == anchor and edge.dst == anchor and edge.kind == EDGE_RECURRENT for edge in attached)
    graph.assert_invariants(output_size=20)


def test_flexible_wiring_connects_new_specialist_to_existing_peer_modules() -> None:
    graph = _graph()
    peer_module = "specialist:critic_specialist:peer"
    peer = graph.add_neuron(REGION_CRITIC, module_id=peer_module, lifecycle="candidate")
    module_id = "specialist:memory_specialist:peer-link"
    anchor = graph.add_neuron(REGION_MEMORY, module_id=module_id, lifecycle="candidate")

    edges = CrossModuleConnector(
        flexible_connectivity=True,
        flexible_max_edges=4,
        flexible_peer_max_edges=4,
    ).wire_specialist(
        graph,
        module_id=module_id,
        role=ROLE_MEMORY_SPECIALIST,
        candidate_id="peer-link",
        need_reason="low_memory_match",
        rng=np.random.default_rng(8),
        max_edges=12,
    )

    attached = [graph.edges[edge_id] for edge_id in edges]
    assert any({edge.src, edge.dst} == {anchor, peer} for edge in attached)
    assert all(edge.src_module_id != edge.dst_module_id for edge in attached if peer in {edge.src, edge.dst})
    graph.assert_invariants(output_size=20)
