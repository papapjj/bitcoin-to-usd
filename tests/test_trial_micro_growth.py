from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from self_growing_trader.cross_module import CrossModuleConnector
from self_growing_trader.self_growing_runner import SelfGrowingPolicy
from self_growing_trader.trial_micro_growth import TrialMicroGrowthController


def _graph():
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=3, hidden_size=5, seed=771, m1_backend="graph")
    return policy.modules[0].m1_component.graph.copy()


def _trace(*, selected_idx: int = 0, active_nodes=(), active_edges=(), risk: float = 0.1):
    return SimpleNamespace(
        selected_idx=selected_idx,
        risk=risk,
        novelty=0.2,
        conflict=0.1,
        positions_pipe="",
        lifecycle_tags=(),
        account_context={"current_equity_drawdown": 0.0},
        active_neuron_ids=tuple(active_nodes),
        active_edge_ids=tuple(active_edges),
    )


def _signal(reward: float = 0.8, td: float = 0.8, **extra):
    return {"reward": reward, "cost": 0.0, "td_delta": td, "prediction_error": 0.2, **extra}


def _created(events):
    return next(event for event in events if event.kind == "specialist_module_candidate_created")


def _set_module_utility(graph, event, value: float = 0.10) -> None:
    for node_id in event.details["node_ids"]:
        graph.neurons[node_id].utility = value
    for edge_id in event.details["edge_ids"]:
        graph.edges[edge_id].utility = value


def test_micro_growth_creates_a_real_bounded_candidate_circuit():
    graph = _graph()
    controller = TrialMicroGrowthController(
        learning_threshold=0.01,
        max_new_modules_per_trial=1,
        module_input_adapters=3,
        module_core_neurons=4,
        module_output_adapters=2,
    )
    before_nodes = len(graph.neurons)
    before_edges = len(graph.edges)

    event = _created(controller.on_outcome(
        graph,
        trial_id="trial-a",
        role="memory_specialist",
        region="memory",
        reason="unknown_regime",
        trace=_trace(),
        learning_signal=_signal(),
        connector=CrossModuleConnector(),
        rng=np.random.default_rng(7),
    ))

    assert event.module_id and event.module_id.startswith("specialist:memory_specialist:trial-a")
    assert len(event.details["input_adapter_ids"]) == 3
    assert len(event.details["core_ids"]) == 4
    assert len(event.details["output_adapter_ids"]) == 2
    assert len(event.details["node_ids"]) == 9
    assert len(graph.neurons) == before_nodes + 9
    assert len(graph.edges) > before_edges
    record = graph.specialist_modules[event.module_id]
    assert record["status"] == "candidate"
    assert set(record["node_ids"]) == set(event.details["node_ids"])
    assert all(graph.neurons[node_id].module_id == event.module_id for node_id in record["node_ids"])
    assert {graph.neurons[node_id].module_stage for node_id in record["node_ids"]} == {
        "input_adapter", "core", "output_adapter"
    }
    assert all(graph.edges[edge_id].gate <= controller.module_initial_gate + 0.40 for edge_id in record["edge_ids"])
    # Canonical decoder outputs remain fixed; specialist output adapters route
    # *into* them instead of extending the action schema.
    assert len(graph.regions["output"].neuron_ids) == 3
    graph.assert_invariants(output_size=3)


def test_specialist_module_activates_only_after_repeated_nearby_causal_support():
    graph = _graph()
    controller = TrialMicroGrowthController(
        learning_threshold=0.01,
        max_new_modules_per_trial=1,
        survival_similar_outcomes=2,
        survival_positive_supports=2,
        survival_utility=0.01,
        consolidate_utility=0.50,
        context_similarity_threshold=0.80,
        min_causal_confidence_to_activate=0.55,
    )
    event = _created(controller.on_outcome(
        graph, trial_id="trial-b", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(risk=0.10), learning_signal=_signal(causal_confidence=0.72),
        connector=CrossModuleConnector(), rng=np.random.default_rng(8),
    ))
    _set_module_utility(graph, event)
    node_ids = tuple(event.details["node_ids"])
    edge_ids = tuple(event.details["edge_ids"])

    controller.on_outcome(
        graph, trial_id="trial-b", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(selected_idx=0, active_nodes=node_ids, active_edges=edge_ids, risk=0.11),
        learning_signal=_signal(causal_confidence=0.72), connector=CrossModuleConnector(), rng=np.random.default_rng(9),
    )
    events = controller.on_outcome(
        graph, trial_id="trial-b", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(selected_idx=2, active_nodes=node_ids, active_edges=edge_ids, risk=0.13),
        learning_signal=_signal(causal_confidence=0.72), connector=CrossModuleConnector(), rng=np.random.default_rng(10),
    )

    assert any(item.kind == "specialist_module_activated" for item in events)
    record = graph.specialist_modules[event.module_id]
    assert record["status"] == "active"
    assert record["positive_supports"] >= 2
    assert all(graph.neurons[node_id].lifecycle == "active" for node_id in node_ids)


def test_specialist_module_does_not_consolidate_without_verified_causal_replay():
    graph = _graph()
    controller = TrialMicroGrowthController(
        learning_threshold=0.01,
        max_new_modules_per_trial=1,
        survival_similar_outcomes=1,
        survival_positive_supports=1,
        survival_utility=0.01,
        consolidate_utility=0.01,
        require_causal_validation_to_consolidate=True,
    )
    event = _created(controller.on_outcome(
        graph, trial_id="trial-c", role="critic_specialist", region="critic", reason="sl_too_tight",
        trace=_trace(), learning_signal=_signal(causal_confidence=0.95),
        connector=CrossModuleConnector(), rng=np.random.default_rng(11),
    ))
    _set_module_utility(graph, event)
    node_ids = tuple(event.details["node_ids"])
    edge_ids = tuple(event.details["edge_ids"])

    unvalidated = controller.on_outcome(
        graph, trial_id="trial-c", role="critic_specialist", region="critic", reason="sl_too_tight",
        trace=_trace(active_nodes=node_ids, active_edges=edge_ids), learning_signal=_signal(causal_confidence=0.95),
        connector=CrossModuleConnector(), rng=np.random.default_rng(12),
    )
    assert not any(row.kind == "specialist_module_consolidated" for row in unvalidated)
    assert graph.specialist_modules[event.module_id]["status"] in {"active", "candidate"}

    validated = controller.on_outcome(
        graph, trial_id="trial-c", role="critic_specialist", region="critic", reason="sl_too_tight",
        trace=_trace(active_nodes=node_ids, active_edges=edge_ids),
        learning_signal=_signal(causal_confidence=0.95, causal_validated=True, causal_pattern_key="sl_too_tight"),
        connector=CrossModuleConnector(), rng=np.random.default_rng(13),
    )
    assert any(row.kind == "specialist_module_consolidated" for row in validated)
    assert graph.specialist_modules[event.module_id]["status"] == "consolidated"


def test_unrelated_context_is_neutral_not_negative_survival_evidence():
    graph = _graph()
    controller = TrialMicroGrowthController(
        learning_threshold=0.01,
        max_new_modules_per_trial=1,
        grace_outcomes=0,
        context_similarity_threshold=0.90,
    )
    event = _created(controller.on_outcome(
        graph, trial_id="trial-near", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(risk=0.05), learning_signal=_signal(), connector=CrossModuleConnector(), rng=np.random.default_rng(81),
    ))
    record = graph.specialist_modules[event.module_id]
    fade_before = record["fade"]
    near_before = record["similar_context_observations"]

    events = controller.on_outcome(
        graph, trial_id="trial-near", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(selected_idx=2, active_nodes=event.details["node_ids"], active_edges=event.details["edge_ids"], risk=0.95),
        learning_signal={"reward": 0.0, "cost": 0.0, "td_delta": 0.0, "prediction_error": 0.0},
        connector=CrossModuleConnector(), rng=np.random.default_rng(82),
    )

    assert not any(row.kind in {"specialist_module_fading", "specialist_module_pruned"} for row in events)
    assert record["fade"] == fade_before
    assert record["similar_context_observations"] == near_before


def test_unsupported_specialist_module_fades_then_prunes_gradually():
    graph = _graph()
    controller = TrialMicroGrowthController(
        learning_threshold=0.01,
        max_new_modules_per_trial=1,
        fade_rate=0.50,
        prune_fade_threshold=0.30,
        grace_outcomes=0,
    )
    event = _created(controller.on_outcome(
        graph, trial_id="trial-d", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(), learning_signal=_signal(), connector=CrossModuleConnector(), rng=np.random.default_rng(14),
    ))
    record = graph.specialist_modules[event.module_id]
    edge_ids = tuple(record["edge_ids"])
    initial_gates = {edge_id: graph.edges[edge_id].gate for edge_id in edge_ids}

    controller.on_outcome(
        graph, trial_id="trial-d", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(active_nodes=record["node_ids"], active_edges=edge_ids),
        learning_signal={"reward": 0.0, "cost": 0.0, "td_delta": 0.0, "prediction_error": 0.0},
        connector=CrossModuleConnector(), rng=np.random.default_rng(15),
    )
    events = controller.on_outcome(
        graph, trial_id="trial-d", role="memory_specialist", region="memory", reason="unknown",
        trace=_trace(active_nodes=record["node_ids"], active_edges=edge_ids),
        learning_signal={"reward": 0.0, "cost": 0.0, "td_delta": 0.0, "prediction_error": 0.0},
        connector=CrossModuleConnector(), rng=np.random.default_rng(16),
    )

    assert any(row.kind == "specialist_module_pruned" for row in events)
    assert record["status"] == "pruning"
    assert all(graph.edges[edge_id].gate < initial_gates[edge_id] for edge_id in edge_ids)
    assert all(not graph.edges[edge_id].enabled for edge_id in edge_ids)


def test_policy_candidate_grows_real_circuit_without_mutating_baseline():
    baseline = SelfGrowingPolicy.seed(input_size=5, output_size=3, hidden_size=5, seed=772, m1_backend="graph")
    baseline.graph_supernetwork.trial_growth.learning_threshold = 0.01
    baseline.graph_supernetwork.trial_growth.max_new_modules_per_trial = 1
    baseline_nodes = len(baseline.modules[0].m1_component.graph.neurons)
    trial = baseline.propose(np.random.default_rng(17), mutation_sigma=0.01, max_modules=3)
    before = len(trial.modules[0].m1_component.graph.neurons)
    trace = trial.act(np.linspace(-0.2, 0.3, 5))
    realised = replace(trace, execution_details={
        "bridge_observed": True,
        "memory_eligible": True,
        "action_feedback_eligible": True,
        "closed": True,
    })

    trial.observe_transition(realised, np.linspace(0.1, -0.1, 5), reward=0.8, cost=0.0, done=False)

    assert trial.graph_trial_mode is True
    assert trial.graph_trial_id
    graph = trial.modules[0].m1_component.graph
    assert len(graph.neurons) >= before + 9
    assert len(baseline.modules[0].m1_component.graph.neurons) == baseline_nodes
    assert graph.specialist_modules
    assert any(event.get("kind") == "specialist_module_candidate_created" for event in trial.growth_events)
