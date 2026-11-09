import numpy as np

from self_growing_trader.self_growing_runner import ExperienceSuperNetwork, MemoryNeuron, MemoryCluster, MemoryConcept, StrategyMemory, MetaStrategyMemory


def test_all_memory_layers_are_projected_to_neural_cells_and_synapses():
    mem = ExperienceSuperNetwork()
    node = MemoryNeuron(
        node_id="n1",
        trace_id="t1",
        obs_embedding=np.ones(32),
        action_embedding=np.ones(20),
        next_embedding=np.ones(32) * 0.5,
        reward_ema=1.0,
        td_ema=0.5,
        prediction_error_ema=0.1,
        utility=0.7,
        evidence_for=1.0,
        action_origin="policy",
    )
    mem.neurons[node.node_id] = node
    cluster = MemoryCluster("c1", ["n1"], np.ones(32), np.ones(20), utility=0.6, evidence_for=1.0, policy_support=3.0)
    mem.clusters[cluster.cluster_id] = cluster
    concept = MemoryConcept(
        concept_id="k1",
        name="policy_positive",
        cluster_ids=["c1"],
        context_proto=np.ones(32),
        action_proto=np.ones(20),
        utility=0.5,
        confidence=0.8,
        policy_support=4.0,
        promotion_state="candidate",
        related_neurons=["n1"],
    )
    mem.concepts[concept.concept_id] = concept
    strategy = StrategyMemory("s1", ["c1"], np.ones(32), np.ones(20), utility=0.4, confidence=0.7)
    mem.strategies[strategy.strategy_id] = strategy
    mem.meta_strategies["m1"] = MetaStrategyMemory("m1", ["s1"], np.ones(32), utility=0.3, trust=0.6)
    mem.causal_nodes["cause1"] = {"kind": "outcome", "utility": -0.2, "payload": {"drawdown": 1.0}}

    mem.ensure_compatible()
    summary = mem.summary()

    assert summary["neural_substrate_version"] == 8
    assert summary["neural_memory_cells"] >= 6
    assert summary["neural_memory_synapses"] >= 5
    assert "episodic_memory" in summary["neural_memory_roles"]
    assert "semantic_concept" in summary["neural_memory_roles"]
    assert "cluster_to_concept" in summary["neural_synapse_kinds"]
    assert "concept_to_strategy" in summary["neural_synapse_kinds"] or "cluster_to_strategy" in summary["neural_synapse_kinds"]
