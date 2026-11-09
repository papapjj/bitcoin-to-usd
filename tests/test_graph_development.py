from __future__ import annotations

import numpy as np

from self_growing_trader.graph_development import (
    ACTION_CONSOLIDATE,
    ACTION_GROW_NEURON,
    ACTION_PRUNE_EDGE,
    GraphDevelopmentController,
    GraphStructuralAction,
)
from self_growing_trader.module_need import ROLE_MEMORY_SPECIALIST, ROLE_PREDICTOR_SPECIALIST, ModuleNeedSignal
from self_growing_trader.neural_graph import REGION_PREDICTOR, NeuralGraphBudget
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def _graph():
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    return policy.modules[0].m1_component.graph.copy()


def test_graph_development_grows_neuron_from_need_signal() -> None:
    graph = _graph()
    controller = GraphDevelopmentController(rng=np.random.default_rng(1))
    before_neurons = len(graph.neurons)

    event = controller.propose_and_apply(
        graph,
        ModuleNeedSignal(novelty=1.0, memory_match_quality=0.0, repeated_unknown_regime=1.0),
        cluster_id="unknown_regime",
    )

    assert event.action == ACTION_GROW_NEURON
    assert event.details["role"] == ROLE_MEMORY_SPECIALIST
    assert len(graph.neurons) == before_neurons + 1
    assert event.after_edges >= event.before_edges
    graph.assert_invariants(output_size=3)


def test_graph_development_holds_when_signal_is_weak() -> None:
    graph = _graph()
    controller = GraphDevelopmentController(grow_threshold=0.9, rng=np.random.default_rng(2))

    event = controller.propose_and_apply(graph, ModuleNeedSignal(prediction_error=0.1), cluster_id="weak")

    assert event.action == "hold"
    assert len(controller.events) == 1


def test_graph_development_can_connect_m1_to_new_predictor_growth_edge() -> None:
    graph = _graph()
    controller = GraphDevelopmentController(rng=np.random.default_rng(3))

    event = controller.apply_action(
        graph,
        GraphStructuralAction(
            ACTION_GROW_NEURON,
            role=ROLE_PREDICTOR_SPECIALIST,
            region=REGION_PREDICTOR,
            reason="prediction_error",
        ),
    )

    assert event.action == ACTION_GROW_NEURON
    assert graph.neurons[event.target].region == REGION_PREDICTOR
    assert any(
        graph.neurons[edge.src].module_id == "m1"
        and graph.neurons[edge.dst].region == REGION_PREDICTOR
        and graph.neurons[edge.dst].module_id != "m1"
        for edge in graph.edges.values()
    )
    graph.assert_invariants(output_size=3)


def test_graph_development_ignores_legacy_total_connection_budget() -> None:
    graph = _graph()
    controller = GraphDevelopmentController(rng=np.random.default_rng(4))
    before_neurons = len(graph.neurons)
    before_edges = len(graph.edges)

    event = controller.apply_action(
        graph,
        GraphStructuralAction(
            ACTION_GROW_NEURON,
            role=ROLE_PREDICTOR_SPECIALIST,
            region=REGION_PREDICTOR,
            reason="prediction_error",
        ),
    )

    assert event.action == ACTION_GROW_NEURON
    assert len(graph.neurons) == before_neurons + 1
    assert len(graph.edges) > before_edges
    graph.assert_invariants(output_size=3)


def test_graph_development_ignores_retired_fixed_module_edge_budget() -> None:
    graph = _graph()
    graph.budget = NeuralGraphBudget(max_m1_module_edges=graph._count_m1_module_edges())
    controller = GraphDevelopmentController(rng=np.random.default_rng(44))

    event = controller.apply_action(
        graph,
        GraphStructuralAction(
            ACTION_GROW_NEURON,
            role=ROLE_PREDICTOR_SPECIALIST,
            region=REGION_PREDICTOR,
            reason="prediction_error",
        ),
    )

    assert event.action == ACTION_GROW_NEURON
    assert not event.details["skipped_edges"]
    graph.assert_invariants(output_size=3)


def test_graph_development_prune_and_consolidate() -> None:
    graph = _graph()
    controller = GraphDevelopmentController()
    edge_id = next(iter(graph.edges))
    graph.edges[edge_id].utility = -2.0

    prune_event = controller.apply_action(graph, GraphStructuralAction(ACTION_PRUNE_EDGE, reason="low_utility"))

    assert prune_event.target == edge_id
    assert graph.edges[edge_id].enabled is False

    active_edge = next(edge for edge in graph.edges.values() if edge.enabled)
    active_edge.utility = 1.0
    before = active_edge.plasticity
    consolidate_event = controller.apply_action(graph, GraphStructuralAction(ACTION_CONSOLIDATE, reason="promoted"))

    assert consolidate_event.details["consolidated_edges"] >= 1
    assert active_edge.plasticity < before
