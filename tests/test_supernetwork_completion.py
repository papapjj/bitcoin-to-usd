from __future__ import annotations

import json

import numpy as np

from self_growing_trader.adaptive_supernetwork import AdaptiveSuperNetwork
from self_growing_trader.cross_module import CrossModuleConnector
from self_growing_trader.lineage import NetworkLineage
from self_growing_trader.module_need import ROLE_MEMORY_SPECIALIST, ModuleNeedSignal, module_spec_for_role
from self_growing_trader.neural_graph import EDGE_GATING, REGION_CRITIC, REGION_MEMORY, REGION_OUTPUT
from self_growing_trader.self_growing_runner import SelfGrowingPolicy
from self_growing_trader.supernetwork_blueprint import BlueprintGenerator
from self_growing_trader.supernetwork_runtime import load_valid_journal_transitions


def _network() -> AdaptiveSuperNetwork:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=31)
    graph = policy.modules[0].m1_component.graph.copy()
    lineage = NetworkLineage("root", None, "root", 0, None, "root")
    return AdaptiveSuperNetwork(network_id="root", graph=graph, lineage=lineage)


def test_graph_policy_records_trace_and_observes_transition() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=32)
    policy.convert_m1_to_graph_backend()
    trace = policy.act(np.array([0.3, -0.2, 0.1, 0.5]))
    signal = policy.observe_transition(trace, np.array([0.2, -0.1, 0.0, 0.4]), reward=-0.2, cost=0.1, done=False)

    assert trace.active_neuron_ids
    assert trace.active_edge_ids
    assert trace.graph_region_activity
    assert "prediction_error" in signal
    assert "graph_credit_attribution" in {event.get("type") for event in policy.growth_events}


def test_cross_module_output_is_bounded_bias_gate() -> None:
    network = _network()
    graph = network.graph
    specialist = graph.add_neuron(REGION_CRITIC, module_id="risk_gate_specialist", lifecycle="candidate")
    m1_output = graph.regions[REGION_OUTPUT].neuron_ids[0]
    edge = graph.add_connection(
        specialist,
        m1_output,
        weight=0.05,
        gate=-3.0,
        kind=EDGE_GATING,
        created_by="graph_development",
        candidate_id="c1",
        need_reason="stress_entry_loss",
    )
    assert graph.edges[edge].weight <= graph.max_action_bias_weight
    assert graph.edges[edge].gate <= graph.max_action_bias_gate


def test_candidate_needs_replay_and_mt5_for_promotion() -> None:
    network = _network()
    baseline_weight = next(iter(network.graph.edges.values())).weight
    candidate = network.propose_development_candidate(
        ModuleNeedSignal(novelty=1.0, memory_match_quality=0.0),
        candidate_id="cand-reject",
        cluster_id="unknown_regime",
        parent_fitness=1.0,
    )
    rejected = network.validate_candidate(
        candidate,
        replay_validator=lambda _graph: 1.1,
        mt5_validator=lambda _graph: (True, 1.2, False),
    )
    assert rejected.promoted is False
    assert "missing_telemetry" in rejected.reject_reasons
    assert next(iter(network.graph.edges.values())).weight == baseline_weight

    candidate = network.copy_candidate(candidate_id="cand-promote", parent_fitness=1.0)
    promoted = network.validate_candidate(
        candidate,
        replay_validator=lambda _graph: (True, 1.2),
        mt5_validator=lambda _graph: (True, 1.3, True),
    )
    assert promoted.promoted is True
    assert network.lineage.promotion_count == 1


def test_blueprint_generator_and_filtered_warmup(tmp_path) -> None:
    network = _network()
    generator = BlueprintGenerator()
    blueprint = generator.generate(
        blueprint_id="bp-generated",
        parent_id="root",
        input_size=4,
        output_size=3,
        graph_summary=network.metrics(),
    )
    blueprint.assert_no_parent_state()

    journal = tmp_path / "journal.jsonl"
    journal.write_text(json.dumps({
        "memory_key_schema": 6,
        "reward": 1.0,
        "td_delta": 0.5,
        "execution_details": {"memory_eligible": True},
        "trace": {"memory_key_schema": 6, "active_edge_ids": [], "active_neuron_ids": []},
    }) + "\n", encoding="utf-8")
    assert len(load_valid_journal_transitions(journal)) == 1
    assert network.warmup_from_journal(journal) == 1
