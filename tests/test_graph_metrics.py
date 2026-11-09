from __future__ import annotations

from self_growing_trader.graph_metrics import graph_metrics, module_need_metrics
from self_growing_trader.module_need import ROLE_MEMORY_SPECIALIST, ModuleNeedSignal
from self_growing_trader.neural_graph import EDGE_FORWARD, REGION_CRITIC
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_graph_metrics_reports_size_edges_and_health() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    graph = policy.modules[0].m1_component.graph.copy()
    specialist = graph.add_neuron(REGION_CRITIC, module_id="risk_gate_specialist", lifecycle="candidate")
    src = graph.regions[REGION_CRITIC].neuron_ids[0]
    graph.add_connection(src, specialist, weight=0.1, gate=-2.0, kind=EDGE_FORWARD, created_by="test")

    metrics = graph_metrics(graph)

    assert metrics["graph_neurons"] == float(len(graph.neurons))
    assert metrics["graph_edges"] == float(len(graph.edges))
    assert metrics["cross_module_edges"] >= 1.0
    assert metrics["nan_or_inf_count"] == 0.0
    assert "region_critic_neurons" in metrics


def test_module_need_metrics_prefixes_role_scores() -> None:
    metrics = module_need_metrics(ModuleNeedSignal(novelty=1.0, memory_match_quality=0.0))

    assert metrics[f"module_need_{ROLE_MEMORY_SPECIALIST}"] > 0.0
