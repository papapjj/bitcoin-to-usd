from __future__ import annotations

import pickle

import numpy as np

from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_graph_policy_proposal_is_candidate_only_and_trace_safe():
    baseline = SelfGrowingPolicy.seed(input_size=6, output_size=4, hidden_size=6, seed=101)
    baseline.convert_m1_to_graph_backend()
    source_graph = baseline.modules[0].m1_component.graph
    baseline_nodes = len(source_graph.neurons)
    baseline_edges = len(source_graph.edges)

    candidate = baseline.propose(np.random.default_rng(12), mutation_sigma=0.02, max_modules=3)
    proposal = next(event for event in reversed(candidate.growth_events) if event.get("type") == "graph_candidate_proposal")

    assert len(source_graph.neurons) == baseline_nodes
    assert len(source_graph.edges) == baseline_edges
    assert proposal["candidate_id"]
    assert proposal["role"]
    assert proposal["reason"]
    candidate.modules[0].m1_component.graph.assert_invariants()

    trace = candidate.act(np.linspace(-0.3, 0.4, 6))
    assert trace.active_neuron_ids
    assert trace.active_edge_ids
    assert np.all(np.isfinite(trace.selected_action))


def test_graph_candidate_feedback_updates_prior_without_importing_rejected_graph():
    baseline = SelfGrowingPolicy.seed(input_size=5, output_size=3, hidden_size=5, seed=102)
    baseline.convert_m1_to_graph_backend()
    candidate = baseline.propose(np.random.default_rng(7), mutation_sigma=0.02, max_modules=3)
    proposal = next(event for event in reversed(candidate.growth_events) if event.get("type") == "graph_candidate_proposal")
    role = proposal["role"]
    before = baseline.graph_supernetwork.role_prior.prior(role)

    baseline.record_graph_candidate_outcome(promoted=False, source_policy=candidate)
    assert baseline.graph_supernetwork.role_prior.prior(role) < before
    assert baseline.graph_supernetwork.role_prior.cooldown(role, cluster_id=proposal["reason"]) >= 1

    candidate.record_graph_candidate_outcome(promoted=True)
    assert candidate.graph_supernetwork.role_prior.prior(role) >= before


def test_graph_candidate_pickle_preserves_runtime_and_no_aliases():
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=3, hidden_size=5, seed=103)
    policy.convert_m1_to_graph_backend()
    proposal = policy.propose(np.random.default_rng(3), mutation_sigma=0.02, max_modules=3)
    loaded = pickle.loads(pickle.dumps(proposal))

    assert loaded.m1_backend == "m1_graph"
    assert loaded.graph_supernetwork is not None
    assert loaded.modules[0].m1_component.graph is not proposal.modules[0].m1_component.graph
    loaded.modules[0].m1_component.graph.assert_invariants()


def test_graph_proposal_can_grow_past_legacy_total_connection_budget():
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=3, hidden_size=5, seed=104)
    policy.convert_m1_to_graph_backend()
    graph = policy.modules[0].m1_component.graph
    before_neurons = len(graph.neurons)

    proposal = policy.graph_supernetwork.propose(
        graph,
        candidate_id="connection-unbounded",
        trace={"trace_id": "budget-full", "prediction_error": 1.0, "memory_match_quality": 0.0},
        learning_signal={"prediction_error": 1.0, "reward": -1.0, "td_delta": -1.0},
        starvation={"cadence_deficit": 1.0, "no_trade_streak": 20.0},
        rng=np.random.default_rng(5),
    )

    assert proposal.action in {"grow_specialist_module", "grow_neuron", "grow_edge", "rewire_edge", "split_region", "merge_region"}
    assert proposal.structural_action.reason != "budget_exceeded"
    assert len(graph.neurons) >= before_neurons
    if proposal.action == "grow_specialist_module":
        assert proposal.specialist_module_id
        assert proposal.specialist_module_summary["node_count"] >= 9
        assert proposal.specialist_module_id in graph.specialist_modules
