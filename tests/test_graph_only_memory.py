from __future__ import annotations

import numpy as np

from self_growing_trader.config import load_config
from self_growing_trader.m1_graph_module import M1GraphModule
from self_growing_trader.neural_graph import NeuralGraphBudget
from self_growing_trader.self_growing_runner import ExperienceSuperNetwork, SelfGrowingPolicy


def test_default_config_is_direct_rnn_and_bounded() -> None:
    config = load_config("configs/default.yaml")
    assert config.supernetwork.m1_backend == "direct_rnn"
    assert config.supernetwork.max_child_networks == 4
    assert config.supernetwork.micro_growth_max_neurons_per_trial == 8
    assert config.supernetwork.max_graph_neurons == 512
    assert config.supernetwork.max_graph_connections == 16384


def test_policy_can_seed_graph_explicitly_without_legacy_adapter() -> None:
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, hidden_size=12, seed=7, m1_backend="m1_graph")
    policy.ensure_compatible(target_input_size=119)
    assert policy.m1_backend == "m1_graph"
    assert isinstance(policy.modules[0].m1_component, M1GraphModule)
    assert policy.modules[0].m1_component.output_size == 20


def test_open_graph_budget_accepts_typed_edges() -> None:
    budget = NeuralGraphBudget()
    assert budget.max_neurons == 0
    assert budget.max_connections == 0
    assert budget.max_recurrent_edges == 0
    assert budget.max_lateral_edges == 0


def test_memory_returns_critic_bias_uncertainty_and_contradiction() -> None:
    memory = ExperienceSuperNetwork()
    memory.ensure_compatible()
    key = np.linspace(-0.5, 0.5, memory.memory_key_size)
    state, delta, short, regime = memory._observe_context(key)
    from self_growing_trader.self_growing_runner import MemoryNeuron
    node = MemoryNeuron(
        node_id="mem-test",
        trace_id="trace-test",
        obs_embedding=state,
        action_embedding=np.ones(16),
        next_embedding=state,
        reward_ema=-1.0,
        td_ema=-1.0,
        prediction_error_ema=0.2,
        utility=-0.8,
        context_delta=delta,
        short_context=short,
        regime_context=regime,
        evidence_for=0.0,
        evidence_against=8.0,
        uncertainty=0.1,
    )
    memory.neurons[node.node_id] = node
    result = memory.query(key, output_size=20)
    assert result["critic_bias"].shape == (32,)
    assert result["uncertainty"] >= 0.0
    assert result["contradiction_strength"] >= 0.0
    assert result["action_bias"].shape == (20,)


def test_memory_retention_and_blueprint_budget_are_open_ended() -> None:
    from self_growing_trader.mistake_memory import MistakeEpisode, MistakeEpisodeStore
    from self_growing_trader.module_need import module_spec_for_role
    from self_growing_trader.supernetwork_blueprint import blueprint_from_module_spec, spawn_supernetwork_from_blueprint

    store = MistakeEpisodeStore()
    assert store.max_episodes is None
    for index in range(12):
        store.add(MistakeEpisode(episode_id=f"episode-{index}", source="test", action_index=index, severity=0.1))
    assert len(store.episodes) == 12

    blueprint = blueprint_from_module_spec(
        module_spec_for_role("predictor_specialist", reason="open-capacity-test"),
        blueprint_id="open-capacity",
        parent_id=None,
        input_size=4,
        output_size=3,
        hidden_size=5,
    )
    graph = spawn_supernetwork_from_blueprint(blueprint, seed=13)
    assert graph.budget.max_neurons == 0
    assert graph.budget.max_connections == 0


def test_memory_v3_builds_semantic_concept_from_policy_evidence() -> None:
    from self_growing_trader.self_growing_runner import MemoryCluster, MemoryNeuron

    memory = ExperienceSuperNetwork()
    memory.ensure_compatible()
    key = np.linspace(-0.4, 0.4, memory.memory_key_size)
    state, delta, short, regime = memory._observe_context(key)
    member_ids = []
    for index in range(4):
        node = MemoryNeuron(
            node_id=f"mem-policy-{index}",
            trace_id=f"trace-{index}",
            obs_embedding=state + index * 0.001,
            action_embedding=np.eye(1, 16, 1).reshape(-1),
            next_embedding=state,
            reward_ema=1.0,
            td_ema=0.5,
            prediction_error_ema=0.1,
            utility=0.6,
            raw_utility=0.6,
            credit_weight=1.0,
            action_origin="policy",
            support_policy_count=1.0,
            context_delta=delta,
            short_context=short,
            regime_context=regime,
            evidence_for=0.6,
            evidence_against=0.0,
            uncertainty=0.1,
            causal_lesson={"suspected_causes": ["validated_positive_outcome"]},
            counterfactual_action_values={"1": 0.7, "0": -0.2},
        )
        memory.neurons[node.node_id] = node
        member_ids.append(node.node_id)
    cluster = MemoryCluster(
        cluster_id="cluster-policy",
        member_ids=member_ids,
        centroid=state,
        action_proto=np.eye(1, 16, 1).reshape(-1),
        utility=0.6,
        evidence_for=2.4,
        evidence_against=0.0,
        uncertainty=0.1,
        policy_support=4.0,
        bootstrap_support=0.0,
        context_delta_proto=delta,
        short_context_proto=short,
        regime_context_proto=regime,
    )
    memory.clusters[cluster.cluster_id] = cluster
    concept = memory._update_or_create_concept(cluster)
    assert concept is not None
    assert concept.policy_support >= 4.0
    assert concept.promotion_state in {"candidate", "consolidated"}
    assert concept.causal_profile.get("validated_positive_outcome", 0.0) > 0.0
    result = memory.query(key, output_size=20)
    assert result["matches"]
    summary = memory.summary()
    assert summary["concepts"] >= 1
    assert summary["depth"] >= 3
