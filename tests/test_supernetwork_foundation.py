from __future__ import annotations

import numpy as np
import pytest

from self_growing_trader.adaptive_supernetwork import AdaptiveSuperNetwork
from self_growing_trader.lineage import LineageRegistry, NetworkLineage
from self_growing_trader.module_need import (
    ROLE_MEMORY_SPECIALIST,
    ROLE_POSITION_MANAGEMENT_SPECIALIST,
    ModuleNeedSignal,
    module_spec_for_role,
)
from self_growing_trader.neural_graph import EDGE_FORWARD, REGION_INPUT, REGION_OUTPUT, NeuralGraph
from self_growing_trader.supernetwork_blueprint import (
    blueprint_from_module_spec,
    diversity_score,
    reject_if_clone_like,
    spawn_supernetwork_from_blueprint,
)


def _small_parent_graph() -> NeuralGraph:
    graph = NeuralGraph()
    i0 = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    out = graph.add_neuron(REGION_OUTPUT, module_id="m1", lifecycle="active")
    graph.add_connection(i0, out, weight=0.2, gate=-1.0, kind=EDGE_FORWARD)
    return graph


def test_blueprint_spawn_uses_seed_and_does_not_copy_parent_state() -> None:
    spec = module_spec_for_role(ROLE_POSITION_MANAGEMENT_SPECIALIST, reason="near_tp_missed_cluster")
    blueprint = blueprint_from_module_spec(
        spec,
        blueprint_id="bp-position",
        parent_id="parent-1",
        input_size=3,
        output_size=2,
        hidden_size=8,
    )

    blueprint.assert_no_parent_state()
    graph_a = spawn_supernetwork_from_blueprint(blueprint, seed=11)
    graph_b = spawn_supernetwork_from_blueprint(blueprint, seed=12)

    assert graph_a is not graph_b
    assert len(graph_a.neurons) == len(graph_b.neurons)
    assert len(graph_a.edges) > 0
    assert len(graph_b.edges) > 0
    assert [edge.weight for edge in graph_a.edges.values()] != [edge.weight for edge in graph_b.edges.values()]
    graph_a.assert_invariants(output_size=2)


def test_diversity_gate_rejects_clone_like_graphs() -> None:
    parent = _small_parent_graph()
    clone = parent.copy()

    assert diversity_score(parent, clone) == 0.0
    with pytest.raises(ValueError, match="diversity too low"):
        reject_if_clone_like(parent, clone)


def test_lineage_controls_generation_and_cycles() -> None:
    root = NetworkLineage(
        network_id="root",
        parent_id=None,
        root_id="root",
        generation=0,
        blueprint_id=None,
        role="root",
        promotion_count=1,
    )
    registry = LineageRegistry(max_alive_networks=3)
    registry.add(root)

    child = root.make_child(network_id="child", blueprint_id="bp-1", role=ROLE_MEMORY_SPECIALIST)
    registry.add(child)

    assert child.parent_id == "root"
    assert child.generation == 1
    assert registry.would_create_cycle("root", "child") is True

    with pytest.raises(ValueError, match="duplicate"):
        registry.add(child)


def test_adaptive_supernetwork_candidate_promote_and_rollback_are_isolated() -> None:
    graph = _small_parent_graph()
    lineage = NetworkLineage(
        network_id="root",
        parent_id=None,
        root_id="root",
        generation=0,
        blueprint_id=None,
        role="root",
    )
    network = AdaptiveSuperNetwork(network_id="root", graph=graph, lineage=lineage)
    candidate = network.copy_candidate(candidate_id="cand-1")
    edge_id = next(iter(candidate.graph.edges))
    candidate.graph.edges[edge_id].weight = 9.0

    assert network.graph.edges[edge_id].weight == 0.2

    network.rollback_candidate(candidate, reason="fitness_regression")
    assert network.graph.edges[edge_id].weight == 0.2
    assert network.lineage.failure_count == 1

    candidate_2 = network.copy_candidate(candidate_id="cand-2")
    candidate_2.graph.edges[edge_id].weight = 3.0
    network.promote_candidate(candidate_2, score=1.25)

    assert network.graph.edges[edge_id].weight == 3.0
    assert network.lineage.promotion_count == 1
    assert network.promotion_history[-1]["candidate_id"] == "cand-2"


def test_adaptive_supernetwork_has_own_development_and_plasticity() -> None:
    graph = _small_parent_graph()
    edge_id = next(iter(graph.edges))
    lineage = NetworkLineage(
        network_id="root",
        parent_id=None,
        root_id="root",
        generation=0,
        blueprint_id=None,
        role="root",
    )
    network = AdaptiveSuperNetwork(network_id="root", graph=graph, lineage=lineage)

    signal = network.observe_transition(
        reward=1.0,
        cost=0.0,
        prediction_error=0.1,
        active_edge_ids=[edge_id],
        active_neuron_ids=list(graph.neurons)[:1],
    )
    development = network.develop(ModuleNeedSignal(novelty=1.0, memory_match_quality=0.0), cluster_id="unknown_regime")

    assert signal["updated_edges"] == 1.0
    assert graph.edges[edge_id].utility > 0.0
    assert development["structural_action"].role == ROLE_MEMORY_SPECIALIST
    assert network.copy().development_controller is not network.development_controller


def test_adaptive_supernetwork_develop_and_spawn_child() -> None:
    parent = AdaptiveSuperNetwork(
        network_id="root",
        graph=_small_parent_graph(),
        lineage=NetworkLineage(
            network_id="root",
            parent_id=None,
            root_id="root",
            generation=0,
            blueprint_id=None,
            role="root",
            promotion_count=1,
        ),
    )

    development = parent.develop(ModuleNeedSignal(novelty=1.0, memory_match_quality=0.0), cluster_id="unknown_regime")
    assert development["role"] == ROLE_MEMORY_SPECIALIST
    assert development["structural_action"].role == ROLE_MEMORY_SPECIALIST

    child = parent.spawn_child(
        network_id="child-memory",
        blueprint_id="bp-child-memory",
        role=ROLE_MEMORY_SPECIALIST,
        input_size=2,
        output_size=1,
        seed=21,
        hidden_size=8,
    )
    output = child.act(np.array([0.2, -0.1]))

    assert child.lineage.parent_id == "root"
    assert child.lineage.generation == 1
    assert child.development_controller is not parent.development_controller
    assert child.plasticity_engine is not parent.plasticity_engine
    assert output.shape == (1,)
    assert np.isfinite(output).all()
