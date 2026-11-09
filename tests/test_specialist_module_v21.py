from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from self_growing_trader.live_observatory import graph_topology
from self_growing_trader.self_growing_runner import SelfGrowingPolicy
from self_growing_trader.specialist_module_factory import SpecialistModuleFactory


def test_specialist_module_uses_real_circuit_and_preserves_action_output_count() -> None:
    policy = SelfGrowingPolicy.seed(input_size=8, output_size=4, hidden_size=6, seed=2101)
    _, m1 = policy._graph_m1_modules()[0]
    graph = m1.graph
    output_before = len(graph.regions["output"].neuron_ids)

    built = SpecialistModuleFactory().create(
        graph,
        module_id="specialist:entry_quality_specialist:v21",
        role="entry_quality_specialist",
        reason="entry_timing_replacement",
        candidate_id="candidate-v21",
        trial_id="trial-v21",
        context_key="ctx-v21",
        context_vector=np.linspace(-1.0, 1.0, 24),
        rng=np.random.default_rng(2102),
        causal_confidence=0.72,
    )

    record = graph.specialist_modules[built.module_id]
    assert len(built.node_ids) == 9
    assert len(built.edge_ids) >= 30
    assert len(built.core_ids) == 4
    assert len(built.interface_edge_ids) >= 8
    assert len(graph.regions["output"].neuron_ids) == output_before
    assert record["evidence_tier"] == "causal_supported"
    assert all(graph.neurons[node_id].module_id == built.module_id for node_id in built.node_ids)
    assert {graph.neurons[node_id].module_stage for node_id in built.node_ids} == {
        "input_adapter", "core", "output_adapter"
    }
    graph.assert_invariants(output_size=4)


def test_live_topology_exposes_specialist_module_telemetry_and_namespaced_nodes() -> None:
    policy = SelfGrowingPolicy.seed(input_size=8, output_size=4, hidden_size=6, seed=2103)
    _, m1 = policy._graph_m1_modules()[0]
    built = SpecialistModuleFactory().create(
        m1.graph,
        module_id="specialist:risk_gate_specialist:v21",
        role="risk_gate_specialist",
        reason="drawdown_risk",
        candidate_id="candidate-v21",
        trial_id="trial-v21",
        context_key="ctx-v21",
        context_vector=np.ones(16),
        rng=np.random.default_rng(2104),
        causal_confidence=0.66,
    )

    topology = graph_topology(policy)
    modules = topology["specialist_modules"]
    key = f"m1:m1_thinking::{built.module_id}"
    assert key in modules
    assert modules[key]["node_count"] == len(built.node_ids)
    rows = [item for item in topology["nodes"] if item["module_id"] == built.module_id]
    assert len(rows) == len(built.node_ids)
    assert all(item["id"].startswith("m1:m1_thinking::") for item in rows)
    assert {item["module_stage"] for item in rows} == {"input_adapter", "core", "output_adapter"}
