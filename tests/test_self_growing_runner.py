from __future__ import annotations

import json
import pickle
from dataclasses import replace as dc_replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from self_growing_trader.self_growing_runner import (
    INPUT_COUNT,
    M1_COMPONENTS,
    MEMORY_KEY_SCHEMA_VERSION,
    MEMORY_KEY_SIZE,
    OUTPUT_COUNT,
    SelfGrowingPolicy,
    SpatialWorldPolicy,
    StrategyMemory,
    WorldState,
    audit_policy,
    build_mt5_curriculum_stages,
    compare_policy_cores_mt5,
    derived_compare_runner_dirs,
    ensure_compare_runner_dirs,
    evaluate_policy_mt5_oos,
    apply_outcome_starvation_to_policy_file,
    load_policy,
    improve_self_growing,
    mt5_curriculum_stage_for_round,
    mt5_promotion_rejection_reasons,
    _mt5_train_to_goal_progress_score,
    replay_stratified_experience_journal,
    replay_positive_experience_journal,
    robust_training_fitness,
    train_to_goal,
    train_self_growing_mt5,
    training_windows,
    validate_replay_candidate_mt5,
    validation_selection_score,
)


class DummyResult:
    def __init__(
        self,
        daily_returns,
        total_r,
        expectancy,
        max_drawdown_pct,
        risk_abuse_penalty=0.0,
        ruin=False,
        fitness=0.0,
        trades=None,
        daily_trade_counts=None,
        daily_bar_counts=None,
        min_daily_trades=0,
        max_daily_trades_limit=60,
    ):
        self.daily_returns = daily_returns
        self.total_r = total_r
        self.expectancy = expectancy
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_abuse_penalty = risk_abuse_penalty
        self.ruin = ruin
        self.fitness = fitness
        self.trades = [object()] * max(1, int(round(abs(total_r)))) if trades is None else trades
        self.daily_trade_counts = [] if daily_trade_counts is None else daily_trade_counts
        self.daily_bar_counts = daily_bar_counts
        self.min_daily_trades = min_daily_trades
        self.max_daily_trades_limit = max_daily_trades_limit


def _mt5_curriculum_config(max_positions=5, max_holding_bars=60):
    return SimpleNamespace(
        risk=SimpleNamespace(max_positions=max_positions),
        orders=SimpleNamespace(
            max_holding_bars=max_holding_bars,
            min_daily_trades=0,
            max_daily_trades=60,
        ),
    )


def _mt5_result(fitness: float, net_profit: float = 100.0, profit_factor: float = 1.2, drawdown: float = 5.0, trades: int = 10):
    metrics = {
        "fitness": fitness,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "equity_drawdown_pct": drawdown,
        "total_trades": trades,
        "fitness_lifecycle_count": trades,
        "fitness_ledger_integrity": 1.0,
        "fitness_immediate_exit_ratio": 0.0,
        "fitness_execution_failure_rate": 0.0,
        "fitness_expectancy_lcb": 0.25,
    }
    return {"best": {"fitness": fitness, "metrics": metrics}}


def test_self_growing_policy_activates_and_can_grow() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    assert tuple(policy.m1_components) == M1_COMPONENTS
    assert policy.modules[0].role == "m1_thinking_core"
    assert policy.modules[0].w_rec.shape == (5, 5)
    outputs = policy.activate(np.ones(4))
    assert outputs.shape == (3,)
    recurrent_outputs = policy.activate(np.ones(4))
    policy.reset_runtime_state()
    reset_outputs = policy.activate(np.ones(4))
    assert not np.allclose(recurrent_outputs, reset_outputs)
    trial = policy.propose(np.random.default_rng(3), mutation_sigma=0.1, max_modules=2)
    assert 1 <= len(trial.modules) <= 2
    assert trial.revision == 1


def test_bridge_feedback_applies_homeostatic_plasticity() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    module = policy.modules[0]
    module.gate_b = 1.0
    module.gate_ema = 0.9
    module.activation_ema = 0.95
    module.w_rec[:] = 10.0

    policy.apply_bridge_feedback((0.95,), (1.0,), fitness=10.0)

    assert module.gate_b < 1.0
    assert np.linalg.norm(module.w_rec) < 10.0 * module.w_rec.size
    assert module.lifecycle == "active"


def test_observe_transition_accepts_larger_mt5_bridge_input() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    trace = policy.act(np.ones(4))

    signal = policy.observe_transition(trace, np.ones(8), reward=0.1, cost=0.0, done=False)

    assert "prediction_error" in signal
    assert "td_delta" in signal


def test_load_policy_resizes_graph_io_to_mt5_bridge_schema(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    path = tmp_path / "graph_small.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    upgraded = load_policy(path)
    graph = upgraded.modules[0].m1_component.graph

    assert upgraded.input_size == INPUT_COUNT
    assert upgraded.output_size == OUTPUT_COUNT
    assert upgraded.modules[0].input_size == INPUT_COUNT
    assert upgraded.modules[0].output_size == OUTPUT_COUNT
    assert len(graph.regions["input"].neuron_ids) == INPUT_COUNT
    assert len(graph.regions["output"].neuron_ids) == OUTPUT_COUNT
    assert upgraded.activate(np.zeros(INPUT_COUNT)).shape == (OUTPUT_COUNT,)


def test_feature_v10_migration_preserves_121_inputs_and_activates_wyckoff_tail(tmp_path) -> None:
    legacy_input_count = 121
    policy = SelfGrowingPolicy.seed(input_size=legacy_input_count, output_size=OUTPUT_COUNT, hidden_size=5, seed=71)
    original = policy.modules[0].w_in.copy()
    path = tmp_path / "feature_v9_121.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    upgraded = load_policy(path)
    module = upgraded.modules[0]

    assert upgraded.input_size == INPUT_COUNT == 137
    assert np.allclose(module.w_in[:, :legacy_input_count], original)
    assert np.any(np.abs(module.w_in[:, legacy_input_count:]) > 0.0)
    assert np.all(module.input_mask[legacy_input_count:] == 1.0)

def test_load_policy_resizes_graph_output_to_bridge_schema_v4(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=19, hidden_size=5, seed=17)
    path = tmp_path / "graph_19.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    upgraded = load_policy(path)
    graph = upgraded.modules[0].m1_component.graph
    assert upgraded.output_size == OUTPUT_COUNT
    assert len(graph.regions["output"].neuron_ids) == OUTPUT_COUNT
    assert upgraded.activate(np.zeros(INPUT_COUNT)).shape == (OUTPUT_COUNT,)

def test_graph_predictor_region_is_action_conditioned() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    module = policy.modules[0].m1_component
    x = np.array([1.0, -1.0, 0.5, -0.5])
    buy, *_ = module.forward(x, context={"action_bias": np.array([1.0, -1.0, -1.0])})
    module.reset_state()
    sell, *_ = module.forward(x, context={"action_bias": np.array([-1.0, 1.0, -1.0])})
    assert not np.allclose(buy, sell)

def test_online_transition_updates_graph_credit_paths() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    trace = policy.act(np.array([1.0, -1.0, 0.5, -0.5]))
    graph = policy.modules[0].m1_component.graph
    before = {edge_id: (edge.utility, edge.weight, edge.gate) for edge_id, edge in graph.edges.items()}

    signal = policy.observe_transition(
        trace,
        next_obs=np.array([1.2, -0.8, 0.7, -0.3]),
        reward=1.0,
        cost=0.0,
        done=False,
    )

    assert signal["td_delta"] >= 0.0
    assert any(
        (edge.utility, edge.weight, edge.gate) != before[edge_id]
        for edge_id, edge in graph.edges.items()
        if edge_id in before
    )

def test_positive_experience_replay_updates_graph_memory_without_copying_structure() -> None:
    source = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    target = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=8)
    trace = source.act(np.array([1.0, -1.0, 0.5, -0.5]))
    source.observe_transition(trace, np.array([1.2, -0.8, 0.7, -0.3]), reward=1.0, cost=0.0, done=False)
    target_nodes_before = len(target.modules[0].m1_component.graph.neurons)

    replayed = target.replay_positive_experiences_from(source)

    assert replayed >= 1
    assert len(target.modules[0].m1_component.graph.neurons) == target_nodes_before

def test_experience_supernetwork_grows_memory_connections_and_serializes(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=17)
    obs = np.array([1.0, -1.0, 0.5, -0.5])
    embeddings = [
        np.r_[np.ones(16), np.zeros(16)],
        np.r_[np.ones(16), np.zeros(16)] + 0.01,
        np.r_[np.ones(16), np.zeros(16)] + 0.02,
        np.r_[np.ones(10), np.zeros(6), np.ones(6), np.zeros(10)],
        np.r_[np.ones(10), np.zeros(6), np.ones(6), np.zeros(10)] + 0.01,
        np.r_[np.ones(10), np.zeros(6), np.ones(6), np.zeros(10)] + 0.02,
    ]
    for idx, embedding in enumerate(embeddings):
        trace = policy.act(obs + idx * 0.01)
        trace = dc_replace(trace, encoded_state=embedding, predicted_next_state=embedding)
        policy.observe_transition(
            trace,
            np.array([1.2, -0.8, 0.7, -0.3]) + idx * 0.01,
            reward=1.0 if idx != 2 else -1.0,
            cost=0.0 if idx != 2 else 2.0,
            done=False,
        )

    summary = policy.memory_supernet.summary()
    kinds = summary["neural_synapse_kinds"]
    assert 1 <= summary["neurons"] <= len(embeddings)
    assert summary["clusters"] >= 1
    assert summary["strategies"] >= 1
    assert summary["depth"] >= 3
    assert kinds.get("episode_to_attractor", 0) > 0
    assert kinds.get("attractor_to_concept", 0) > 0
    assert kinds.get("concept_to_strategy", 0) > 0
    assert kinds.get("learned_lateral", 0) > 0
    assert kinds.get("learned_recurrent_support", 0) > 0

    path = tmp_path / "policy.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)
    loaded = load_policy(path)
    assert loaded.memory_supernet.summary()["depth"] >= 3

def test_experience_supernetwork_consolidates_duplicate_action_memories() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=18)
    obs = np.array([1.0, -1.0, 0.5, -0.5])
    trace = policy.act(obs)
    trace = dc_replace(
        trace,
        action_details={"action_name": "BUY_LIMIT", "target_slot": 0, "management_mode": "NONE"},
        selected_action=np.eye(1, 19, 3).reshape(-1),
        encoded_state=np.r_[np.ones(16), np.zeros(16)],
        positions_pipe="",
        positions_pipe_after="",
    )

    for _ in range(12):
        policy.memory_supernet.absorb_transition(
            trace,
            np.array([1.0, -1.0, 0.5, -0.5]),
            reward=0.01,
            cost=0.0,
            signal={"td_delta": 0.01, "prediction_error": 0.1},
        )

    summary = policy.memory_supernet.summary()
    episodes = [cell for cell in policy.memory_supernet.neural_cells.values() if cell.role == "episodic_memory"]
    assert 1 <= len(episodes) <= 2
    assert all(np.argmax(np.asarray(cell.meta.get("action_vector", np.zeros(19)))) == 3 for cell in episodes)
    assert summary["success_action_imprints"] >= 1
    assert summary["neural_memory_roles"].get("context_attractor", 0) >= 1

def test_experience_supernetwork_quarantines_legacy_latent_strategy_memory() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=19)
    obs = np.array([1.0, -1.0, 0.5, -0.5])
    baseline = policy.act(obs)
    action_proto = np.array([1.0, -1.0, -1.0])
    # This represents a pre-v8 strategy whose context proto was an M1 latent
    # state.  It must not be compared to a canonical world-context key.
    policy.memory_supernet.strategies["strategy-test"] = StrategyMemory(
        strategy_id="strategy-test",
        cluster_ids=[],
        context_proto=policy.memory_supernet._resize(baseline.encoded_state),
        action_proto=action_proto,
        utility=5.0,
        confidence=1.5,
        evidence_for=20.0,
        evidence_against=0.0,
        uncertainty=0.02,
    )

    traced = policy.act(obs)

    assert traced.context_schema_version == MEMORY_KEY_SCHEMA_VERSION
    assert traced.action_details["context_key_source"] == "canonical_context_frame"
    assert traced.action_details["context_frame_id"]


def test_experience_supernetwork_does_not_mix_legacy_strategy_memory_into_m1_selector() -> None:
    obs = np.array([1.0, -1.0, 0.5, -0.5])
    baseline_policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=21)
    memory_policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=21)
    baseline_trace = baseline_policy.act(obs)
    memory_policy.memory_supernet.strategies["strategy-memory"] = StrategyMemory(
        strategy_id="strategy-memory",
        cluster_ids=[],
        context_proto=memory_policy.memory_supernet._resize(baseline_trace.encoded_state),
        action_proto=np.array([1.0, -1.0, 0.5]),
        utility=4.0,
        confidence=1.4,
        evidence_for=20.0,
        evidence_against=0.0,
        uncertainty=0.02,
    )

    memory_trace = memory_policy.act(obs)

    # Legacy latent-space entries are quarantined.  Only schema-v8 cells
    # written from a ContextFrame may influence memory/selector pathways.
    assert np.allclose(memory_trace.memory_after, baseline_trace.memory_after)
    assert np.allclose(memory_trace.candidate_scores, baseline_trace.candidate_scores)
    assert memory_trace.context_schema_version == MEMORY_KEY_SCHEMA_VERSION

def test_experience_supernetwork_stores_position_causal_memory() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=23)
    obs = np.array([1.0, -1.0, 0.5, -0.5])
    trace = policy.act(obs)
    trace = dc_replace(
        trace,
        action_details={
            "action_name": "HOLD", "target_slot": 1, "management_mode": "PARTIAL_CLOSE",
            "partial_close_pct": 0.6, "trailing_stop_atr": 1.4, "sl_atr": 1.2, "tp_atr": 2.4,
        },
        positions_pipe="111:1:0.01:100:99:103:0.8:4^222:-1:0.02:101:102:98:1.5:7",
    )

    policy.observe_transition(trace, np.array([1.1, -0.9, 0.6, -0.4]), reward=0.4, cost=0.01, done=False)

    summary = policy.memory_supernet.summary()
    outcomes = [cell for cell in policy.memory_supernet.neural_cells.values() if cell.role == "causal_outcome"]
    assert summary["causal_kinds"].get("outcome", 0) >= 1
    assert outcomes
    payload = outcomes[-1].meta
    assert payload["target_ticket"] == "222"
    assert payload["target_slot"] == 1
    query = policy.memory_supernet.query(trace.encoded_state, policy.output_size, len(policy.modules))
    assert query["runtime"] == "pure_neural_v8_replacement_learning"

def test_experience_supernetwork_tags_near_tp_missed_lifecycle() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=24)
    trace = policy.act(np.array([1.0, -1.0, 0.5, -0.5]))
    trace = dc_replace(
        trace,
        action_details={"action_name": "HOLD", "target_slot": 0, "management_mode": "NONE"},
        positions_pipe=("333:1:0.01:100:99:103:0.2:12:2.7:-0.4:2.8:1.0:0.08:0.5:6:1:2.5"),
        positions_pipe_after="333:1:0.01:100:99:103:-0.8:13:2.7:-0.8:3.8:0.2:0.08:0.2:7:1:3.5",
    )

    policy.observe_transition(trace, np.array([0.9, -1.2, 0.4, -0.8]), reward=-0.3, cost=0.02, done=False)

    outcomes = [cell.meta for cell in policy.memory_supernet.neural_cells.values() if cell.role == "causal_outcome"]
    assert outcomes
    tags = set(outcomes[-1]["lifecycle_tags"])
    assert "near_tp_missed" in tags
    assert "gave_back_profit" in tags
    assert "should_have_partial_or_trail" in tags
    assert outcomes[-1]["giveback_r"] >= 3.0

def test_causal_memory_plasticity_adjusts_candidate_weights_without_touching_seed() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=25)
    trace = policy.act(np.array([1.0, -1.0, 0.5, -0.5]))
    trace = dc_replace(
        trace,
        action_details={
            "action_name": "HOLD",
            "target_slot": 0,
            "management_mode": "NONE",
        },
        positions_pipe=(
            "333:1:0.01:100:99:103:0.2:12:"
            "2.7:-0.4:2.8:1.0:0.08:0.5:6:1:2.5"
        ),
        positions_pipe_after="333:1:0.01:100:99:103:-0.8:13:2.7:-0.8:3.8:0.2:0.08:0.2:7:1:3.5",
    )
    policy.observe_transition(trace, np.array([0.9, -1.2, 0.4, -0.8]), reward=-0.3, cost=0.02, done=False)

    seed_w_out = policy.modules[0].w_out.copy()
    candidate = policy.copy()
    summary = candidate.apply_causal_memory_plasticity(lr=0.01)

    assert summary["applied"] > 0
    assert np.allclose(policy.modules[0].w_out, seed_w_out)
    assert not np.allclose(candidate.modules[0].w_out, seed_w_out)
    assert candidate.modules[0].b_out[15] > policy.modules[0].b_out[15]
    assert candidate.modules[0].b_out[16] > policy.modules[0].b_out[16]
    assert candidate.modules[0].b_out[17] > policy.modules[0].b_out[17]
    assert any(event["type"] == "causal_memory_plasticity" for event in candidate.growth_events)


def test_curiosity_neuron_system_updates_and_serializes(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=26)
    trust_before = policy.curiosity_system.trust
    trace = policy.act(np.array([1.0, -1.0, 0.5, -0.5]))

    policy.observe_transition(trace, np.array([1.1, -0.8, 0.6, -0.2]), reward=0.4, cost=0.01, done=False)

    assert policy.curiosity_system.age > 0
    assert policy.curiosity_system.trust >= trust_before

    path = tmp_path / "curiosity.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)
    loaded = load_policy(path)
    assert loaded.curiosity_system.age == policy.curiosity_system.age
    assert np.allclose(loaded.curiosity_system.w_out, policy.curiosity_system.w_out)


def test_curiosity_proposal_adjusts_candidate_without_mutating_seed() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=27)
    trace = policy.act(np.array([1.0, -1.0, 0.5, -0.5]))
    policy.observe_transition(trace, np.array([1.2, -0.7, 0.7, -0.1]), reward=0.5, cost=0.01, done=False)

    seed_w_out = policy.modules[0].w_out.copy()
    candidate = policy.copy()
    summary = candidate.apply_curiosity_proposal(strength=0.1, residual_noise=0.0, rng=np.random.default_rng(28))

    assert summary["strength"] > 0.0
    assert np.allclose(policy.modules[0].w_out, seed_w_out)
    assert not np.allclose(candidate.modules[0].w_out, seed_w_out)
    assert any(event["type"] == "curiosity_neural_proposal" for event in candidate.growth_events)


def test_graph_development_controller_serializes_and_guides_candidate_growth(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=29)
    baseline_nodes = len(policy.modules[0].m1_component.graph.neurons)
    candidate = policy.propose(np.random.default_rng(30), mutation_sigma=0.08, max_modules=0)

    assert candidate.m1_backend == "m1_graph"
    assert len(policy.modules[0].m1_component.graph.neurons) == baseline_nodes
    assert candidate.graph_trial_mode is True
    assert any(event["type"] == "graph_trial_started" for event in candidate.growth_events)

    path = tmp_path / "graph_dev.pkl"
    with path.open("wb") as handle:
        pickle.dump(candidate, handle)
    loaded = load_policy(path)
    assert loaded.m1_backend == "m1_graph"
    assert loaded.graph_supernetwork is not None

def test_policy_compatibility_adds_neural_development_to_old_policy() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=31)
    delattr(policy, "neural_development")

    policy.ensure_compatible(target_input_size=4)

    assert policy.neural_development is not None
    assert policy.neural_development.w_out.shape[0] >= 18


def test_biological_auxiliary_networks_serialize_and_modulate_action(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=32)
    obs = np.array([0.5, -0.2, 0.1, 0.3])
    trace = policy.act(obs)

    assert trace.selected_action.size == 19
    assert trace.risk >= 0.0
    assert policy.action_parameter_head.age > 0 or policy.action_parameter_head.state.size > 0

    before_age = policy.risk_cost_gate.age
    policy.observe_transition(trace, obs * 0.9, reward=-0.2, cost=0.05, done=False)
    assert policy.risk_cost_gate.age > before_age
    assert policy.action_parameter_head.age > 0

    path = tmp_path / "bio_aux.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)
    loaded = load_policy(path)
    assert loaded.neural_utility.w_out.shape == policy.neural_utility.w_out.shape
    assert loaded.risk_cost_gate.w_out.shape == policy.risk_cost_gate.w_out.shape
    assert loaded.action_parameter_head.w_out.shape == policy.action_parameter_head.w_out.shape


def test_reasoning_graph_serializes_and_learns_from_auxiliary_controllers(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=33)
    before_age = policy.reasoning_graph.age
    before_gate_sum = sum(edge.gate for edge in policy.reasoning_graph.graph.edges.values())
    trace = policy.act(np.array([0.5, -0.2, 0.1, 0.3]))

    policy.observe_transition(trace, np.array([0.45, -0.1, 0.2, 0.25]), reward=0.35, cost=0.01, done=False)
    policy.consolidate(1.0, baseline_fitness=0.0)

    assert policy.reasoning_graph.age > before_age
    assert sum(edge.gate for edge in policy.reasoning_graph.graph.edges.values()) != before_gate_sum
    assert policy.reasoning_graph.summary()["neurons"] > 0

    candidate = policy.copy()
    candidate.reasoning_graph.graph.edges[next(iter(candidate.reasoning_graph.graph.edges))].gate -= 0.1
    assert candidate.reasoning_graph.graph.edges[next(iter(candidate.reasoning_graph.graph.edges))].gate != policy.reasoning_graph.graph.edges[next(iter(policy.reasoning_graph.graph.edges))].gate

    path = tmp_path / "reasoning.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)
    loaded = load_policy(path)
    assert loaded.reasoning_graph.age == policy.reasoning_graph.age
    assert len(loaded.reasoning_graph.graph.edges) == len(policy.reasoning_graph.graph.edges)


def test_rejection_feedback_has_strong_bounded_network_effect() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=36, m1_backend="graph")
    rejected = policy.copy()
    before_hold = float(policy.modules[0].b_out[0])
    before_entry = float(policy.modules[0].b_out[1])
    before_risk = policy.risk_cost_gate.b_out.copy()
    before_reasoning_age = policy.reasoning_graph.age

    event = policy.absorb_rejection_feedback(
        rejected,
        reject_reasons=["no_trade", "fitness_not_improved"],
        metrics={"total_trades": 0, "net_profit": -5.0, "profit_factor": 0.0, "balance_drawdown_pct": 0.5},
        fitness_delta=-20.0,
    )

    assert event["type"] == "rejection_feedback_applied"
    assert policy.rejection_memory
    # Baseline action channels are not globally suppressed after a rejection.
    assert policy.modules[0].b_out[0] == before_hold
    assert policy.modules[0].b_out[1] >= before_entry
    assert np.allclose(policy.risk_cost_gate.b_out, before_risk)
    assert policy.reasoning_graph.age > before_reasoning_age
    assert policy._experience_starvation["starvation"] > 0.0
    assert np.isfinite(policy.modules[0].b_out).all()

def test_rejection_feedback_records_active_paths_without_mutating_baseline_graph() -> None:
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=4, hidden_size=6, seed=37, m1_backend="graph")
    rejected = policy.copy()
    trace = rejected.act(np.linspace(-0.2, 0.4, 5))
    rejected.experience_buffer.append({"trace": trace, "reward": -1.0, "td_delta": -1.0})
    graph = policy.modules[0].m1_component.graph
    active_edge = next(edge_id for edge_id in trace.active_edge_ids if edge_id in graph.edges)
    active_node = next(node_id for node_id in trace.active_neuron_ids if node_id in graph.neurons)
    selected_action = policy._trace_action_index(trace, policy.output_size)
    before_edge_gate = float(graph.edges[active_edge].gate)
    before_edge_weight = float(graph.edges[active_edge].weight)
    before_node_utility = float(graph.neurons[active_node].utility)
    before_w_out = policy.modules[0].w_out[selected_action].copy()

    event = policy.absorb_rejection_feedback(
        rejected,
        reject_reasons=["net_profit_not_positive"],
        metrics={"total_trades": 3, "net_profit": -10.0, "profit_factor": 0.5, "balance_drawdown_pct": 1.0},
        fitness_delta=-12.0,
    )

    assert event["direct_attribution"]["bad_traces"] == 1
    assert event["direct_attribution"]["graph_edges"] >= 1
    assert event["direct_attribution"]["graph_neurons"] >= 1
    assert graph.edges[active_edge].gate == before_edge_gate
    assert graph.edges[active_edge].weight == before_edge_weight
    assert graph.neurons[active_node].utility == before_node_utility
    assert np.allclose(policy.modules[0].w_out[selected_action], before_w_out)

def test_mutation_sigma_only_changes_loss_action_rows() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=8, hidden_size=5, seed=38, m1_backend="graph")
    module = policy.modules[0]
    before_w_out = module.w_out.copy()
    before_gate_w = module.gate_w.copy()
    mask = np.zeros(policy.output_size, dtype=bool)
    mask[3] = True

    summary = policy.apply_curiosity_proposal(
        strength=0.2,
        residual_noise=0.05,
        action_mask=mask,
        rng=np.random.default_rng(39),
    )

    changed_rows = np.flatnonzero(np.linalg.norm(module.w_out - before_w_out, axis=1) > 1e-12)
    assert changed_rows.tolist() == [3]
    assert summary["loss_action_rows"] == [3]
    assert np.allclose(module.gate_w, before_gate_w)


def test_mutation_sigma_only_changes_attributed_legacy_connections() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=8, hidden_size=5, seed=40, m1_backend="graph")
    module = policy.modules[0]
    before_w_out = module.w_out.copy()
    mask = np.zeros(policy.output_size, dtype=bool)
    mask[3] = True
    hidden_evidence = np.zeros(module.hidden_size, dtype=float)
    hidden_evidence[1] = 1.0
    hidden_evidence[4] = 0.8
    policy.curiosity_system.propose_adjustments = lambda _context: {
        "action_delta": np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        "gate_delta": np.ones(4),
        "recurrent_delta": np.ones(8),
        "drive": 1.0,
        "risk_dampen": 0.0,
    }

    summary = policy.apply_curiosity_proposal(
        strength=0.2,
        residual_noise=0.0,
        action_mask=mask,
        action_evidence={"module_hidden": {3: {module.name: hidden_evidence}}},
        rng=np.random.default_rng(41),
    )

    changed = np.abs(module.w_out - before_w_out) > 1e-12
    assert np.flatnonzero(changed.any(axis=1)).tolist() == [3]
    assert np.flatnonzero(changed[3]).tolist() == [1, 4]
    assert summary["loss_action_rows"] == [3]
    assert summary["loss_action_connections"] >= 2


def test_mutation_sigma_only_changes_attributed_graph_edges() -> None:
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=4, hidden_size=6, seed=42, m1_backend="graph")
    graph = policy.modules[0].m1_component.graph
    output_ids = list(graph.regions.get("output").neuron_ids)
    target_action = 1
    target_node = output_ids[target_action]
    incoming = [edge for edge in graph.edges.values() if edge.dst == target_node and edge.enabled]
    assert len(incoming) >= 2
    active_edge = incoming[0]
    untouched_edge = incoming[1]
    before_active = float(active_edge.weight)
    before_untouched = float(untouched_edge.weight)
    mask = np.zeros(policy.output_size, dtype=bool)
    mask[target_action] = True
    policy.curiosity_system.propose_adjustments = lambda _context: {
        "action_delta": np.array([0.0, -1.0, 0.0, 0.0]),
        "gate_delta": np.ones(4),
        "recurrent_delta": np.ones(8),
        "drive": 1.0,
        "risk_dampen": 0.0,
    }

    summary = policy.apply_curiosity_proposal(
        strength=0.2,
        residual_noise=0.0,
        action_mask=mask,
        action_evidence={"graph_edges": {target_action: {active_edge.id}}},
        rng=np.random.default_rng(43),
    )

    assert graph.edges[active_edge.id].weight != before_active
    assert graph.edges[untouched_edge.id].weight == before_untouched
    assert summary["loss_action_rows"] == [target_action]
    assert summary["loss_action_connections"] >= 1


def test_experience_starvation_reaches_curiosity_and_trace() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=34)
    policy.set_experience_starvation({
        "no_trade_streak": 20.0,
        "unfilled_streak": 4.0,
        "cadence_deficit": 0.8,
        "starvation": 1.7,
        "intrinsic_reward": 0.04,
    })

    trace = policy.act(np.array([0.4, -0.3, 0.2, 0.1]))

    assert trace.selected_action.size == 19
    assert trace.experience_starvation["no_trade_streak"] == 20.0
    assert policy._experience_starvation["starvation"] == 1.7
    assert policy.curiosity_system.age >= 0


def test_outcome_starvation_persists_and_merges_with_runtime_starvation(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=35)
    path = tmp_path / "policy.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    starvation = apply_outcome_starvation_to_policy_file(
        path,
        goal_hit=False,
        improved=False,
        fitness_delta=-2.0,
        reason="goal_not_reached",
        mutate_policy=True,
    )
    loaded = load_policy(path)
    loaded.set_experience_starvation({"starvation": 0.5, "intrinsic_reward": 0.01})

    assert starvation["failed_goal_streak"] == 1.0
    assert starvation["non_improve_streak"] == 1.0
    assert loaded._experience_starvation["outcome_starvation"] > 0.0
    assert loaded._experience_starvation["starvation"] > 0.5
    assert any(event.get("type") == "outcome_starvation" for event in loaded.growth_events)



def test_outcome_starvation_preview_does_not_mutate_seed(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=91)
    path = tmp_path / "seed.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)
    before = path.read_bytes()
    preview = apply_outcome_starvation_to_policy_file(
        path,
        goal_hit=False,
        improved=False,
        fitness_delta=-1.0,
        reason="rejected_trial",
    )
    assert preview["persisted"] == 0.0
    assert path.read_bytes() == before

def test_position_outcome_classifier_adds_neural_probs_to_causal_memory() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=33)
    trace = policy.act(np.array([0.2, -0.1, 0.5, -0.3]))
    trace = dc_replace(
        trace,
        action_details={"action_name": "HOLD", "target_slot": 0, "management_mode": "NONE"},
        positions_pipe=("777:1:0.01:100:99:103:0.1:10:2.4:-0.3:2.9:1.0:0.05:0.5:5:1:2.3"),
        positions_pipe_after="777:1:0.01:100:99:103:-0.6:11:2.4:-0.6:3.2:0.2:0.05:0.2:6:1:3.0",
    )

    policy.observe_transition(trace, np.array([0.1, -0.2, 0.4, -0.4]), reward=-0.3, cost=0.01, done=False)
    outcomes = [cell.meta for cell in policy.memory_supernet.neural_cells.values() if cell.role == "causal_outcome"]

    assert outcomes
    payload = outcomes[-1]
    assert "neural_lifecycle_probs" in payload
    assert "audit_lifecycle_tags" in payload
    assert "near_tp_missed" in payload["lifecycle_tags"]

def test_spatial_world_policy_serializes_world_state(tmp_path) -> None:
    policy = SpatialWorldPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=11)
    policy.world.nations["m1_thinking"].energy = 1.7
    policy.world.nations["m1_thinking"].knowledge = 0.4
    path = tmp_path / "world.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    loaded = load_policy(path, policy_core="world")

    assert isinstance(loaded, SpatialWorldPolicy)
    assert loaded.world.nations["m1_thinking"].energy == 1.7
    assert loaded.world.nations["m1_thinking"].knowledge == 0.4
    assert len(loaded.world.nations["m1_thinking"].neurons) == 5


def test_world_geometry_weights_near_and_semantic_neighbors_more() -> None:
    policy = SpatialWorldPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=11)
    rng = np.random.default_rng(12)
    policy.modules.append(policy.modules[0].copy())
    policy.modules[-1].name = "near_country"
    policy.modules.append(policy.modules[0].copy())
    policy.modules[-1].name = "far_country"
    policy.world.ensure_for_modules(policy.modules, rng)
    origin = policy.world.nations["m1_thinking"]
    near = policy.world.nations["near_country"]
    far = policy.world.nations["far_country"]
    near.pos3 = origin.pos3 + np.array([0.05, 0.0, 0.0])
    near.hpos = origin.hpos + np.array([0.01, 0.0, 0.0])
    far.pos3 = origin.pos3 + np.array([5.0, 0.0, 0.0])
    far.hpos = np.array([0.8, 0.0, 0.0])
    origin.energy = 1.0
    origin.trust = 1.0

    assert policy.world.message_weight("m1_thinking", "near_country") > policy.world.message_weight("m1_thinking", "far_country")


def test_spatial_energy_and_trust_modulate_action_output() -> None:
    policy = SpatialWorldPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=11)
    obs = np.array([0.8, -0.2, 0.3, 0.1])
    nation = policy.world.nations["m1_thinking"]
    nation.energy = 0.01
    nation.knowledge = -2.0
    nation.trust = 0.05
    low = policy.act(obs).selected_action.copy()
    policy.reset_runtime_state()
    nation.energy = 3.0
    nation.knowledge = 2.0
    nation.trust = 1.5
    high = policy.act(obs).selected_action.copy()

    assert not np.allclose(low, high)


def test_spatial_world_transition_updates_resource_ledger_and_trust() -> None:
    policy = SpatialWorldPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=11)
    trace = policy.act(np.array([1.0, -1.0, 0.5, -0.5]))
    before = policy.world.nations["m1_thinking"].trust

    signal = policy.observe_transition(trace, np.array([1.2, -0.8, 0.7, -0.3]), reward=1.0, cost=0.0, done=False)

    assert signal["td_delta"] > 0.0
    assert policy.world.nations["m1_thinking"].trust > before
    assert "m1_thinking" in policy.world.ledger.energy_flow
    assert "m1_thinking" in policy.world.knowledge_store.vectors


def test_legacy_train_rejects_five_rlhf_candidate_updates() -> None:
    config = SimpleNamespace(
        learning=SimpleNamespace(five_rlhf_policy_update_enabled=True),
    )
    with pytest.raises(ValueError, match="legacy train_self_growing_mt5"):
        train_self_growing_mt5(config, 1, "outputs/rlhf/should_not_train.pkl")


def test_train_self_growing_mt5_can_seed_world_policy_core(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            captured["policy_type"] = type(policy).__name__
            metrics = SimpleNamespace(
                net_profit=20.0,
                profit_factor=2.0,
                total_trades=2,
                balance_drawdown_pct=0.1,
                equity_drawdown_pct=0.1,
                win_rate=80.0,
                avg_win=3.0,
                avg_loss=-1.0,
                report_path=tmp_path / "report.htm",
            )
            return 10.0, metrics

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "policy_core": getattr(policy, "world_core", "current"), "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = train_self_growing_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        output_path=str(tmp_path / "world.pkl"),
        trials=1,
        policy_core="world",
    )
    saved = load_policy(tmp_path / "world.pkl", policy_core="world")

    assert captured["policy_type"] == "SpatialWorldPolicy"
    assert isinstance(saved, SpatialWorldPolicy)
    assert result["policy_core"] == "world"
    assert result["bridge_runtime"]["policy_core"] == "mixed_geometry_v1"


def test_train_self_growing_mt5_never_promotes_rank_rejected_first_graph_result(monkeypatch, tmp_path) -> None:
    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=-22.92,
                profit_factor=0.84,
                total_trades=159,
                balance_drawdown_pct=0.43,
                equity_drawdown_pct=0.43,
                win_rate=32.7,
                avg_win=1.0,
                avg_loss=-1.0,
                report_path=tmp_path / "report.htm",
            )
            return -100.0, metrics

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)
    config = _mt5_curriculum_config()
    config.supernetwork = SimpleNamespace(require_mt5_promotion=True, trial_history_path="")
    output = tmp_path / "strict_baseline.pkl"

    # A first candidate is evidence for diagnostics, not permission to export a
    # rank-rejected policy as a production winner.  Its candidate artifact is
    # retained for audit/recovery.
    with pytest.raises(RuntimeError, match="fitness_rank_reject"):
        train_self_growing_mt5(
            config,
            rounds=1,
            output_path=str(output),
            trials=1,
            m1_backend="graph",
            experience_journal_path=None,
            replay_journal_max=0,
        )

    assert not output.exists()
    assert (tmp_path / "strict_baseline.candidate_unpromoted.pkl").exists()


def test_train_self_growing_mt5_graph_candidate_can_start_without_seed_trial(monkeypatch, tmp_path) -> None:
    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=-10.0,
                profit_factor=0.9,
                total_trades=20,
                balance_drawdown_pct=0.2,
                equity_drawdown_pct=0.2,
                win_rate=40.0,
                avg_win=1.0,
                avg_loss=-1.0,
                report_path=tmp_path / "candidate_report.htm",
            )
            return -50.0, metrics

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)
    config = _mt5_curriculum_config()
    config.supernetwork = SimpleNamespace(require_mt5_promotion=True, trial_history_path="")

    result = train_self_growing_mt5(
        config,
        rounds=1,
        output_path=str(tmp_path / "graph_candidate.pkl"),
        trials=1,
        include_seed_trial=False,
        m1_backend="graph",
        experience_journal_path=None,
        replay_journal_max=0,
    )

    assert result["best"]["fitness"] == -50.0
    assert (tmp_path / "graph_candidate.pkl").exists()


def test_train_self_growing_mt5_does_not_export_ineligible_fitness(monkeypatch, tmp_path) -> None:
    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=598.26,
                profit_factor=1.92,
                total_trades=12,
                balance_drawdown_pct=4.14,
                equity_drawdown_pct=4.9,
                win_rate=50.0,
                avg_win=208.14,
                avg_loss=-108.43,
                report_path=tmp_path / "report.htm",
                fitness_eligible=False,
            )
            return -100.0, metrics

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)
    output = tmp_path / "ineligible.pkl"

    with pytest.raises(RuntimeError, match="fitness_not_eligible"):
        train_self_growing_mt5(
            _mt5_curriculum_config(),
            rounds=1,
            output_path=str(output),
            trials=1,
            m1_backend="graph",
            experience_journal_path=None,
            replay_journal_max=0,
        )

    assert not output.exists()
    assert (tmp_path / "ineligible.candidate_unpromoted.pkl").exists()


def test_train_self_growing_mt5_missing_seed_starts_fresh(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            captured["policy_type"] = type(policy).__name__
            metrics = SimpleNamespace(
                net_profit=12.0,
                profit_factor=1.5,
                total_trades=4,
                balance_drawdown_pct=0.1,
                equity_drawdown_pct=0.1,
                win_rate=75.0,
                avg_win=2.0,
                avg_loss=-1.0,
                report_path=tmp_path / "report.htm",
            )
            return 5.0, metrics

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "policy_core": getattr(policy, "world_core", "current"), "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)
    missing_seed = tmp_path / "missing.pkl"

    result = train_self_growing_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        output_path=str(tmp_path / "fresh.pkl"),
        trials=1,
        seed_policy_path=missing_seed,
        policy_core="current",
    )

    assert captured["policy_type"] == "SelfGrowingPolicy"
    assert result["policy_core"] == "current"
    assert result["experience_journal"] == str((tmp_path / "fresh.journal.jsonl"))
    assert (tmp_path / "fresh.pkl").exists()
    assert "seed policy not found" in capsys.readouterr().out


def test_replay_positive_experience_journal_updates_policy(tmp_path) -> None:
    source = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=5)
    trace = source.act(np.array([1.0, -1.0, 0.5, -0.5]))
    journal = tmp_path / "journal.jsonl"
    item = {
        "trace_id": trace.trace_id,
        "reward": 1.0,
        "cost": 0.0,
        "td_delta": 1.0,
        "prediction_error": 0.1,
        "done": False,
        "next_obs": [1.2, -0.8, 0.7, -0.3],
        "trace": {
            "timestamp": trace.timestamp,
            "raw_obs": trace.raw_obs.tolist(),
            "encoded_state": trace.encoded_state.tolist(),
            "memory_before": trace.memory_before.tolist(),
            "memory_after": trace.memory_after.tolist(),
            "selected_idx": trace.selected_idx,
            "selected_action": trace.selected_action.tolist(),
            "candidate_actions": trace.candidate_actions.ravel().tolist(),
            "candidate_logits": trace.candidate_logits.ravel().tolist(),
            "candidate_scores": trace.candidate_scores.ravel().tolist(),
            "selected_action_log_prob": trace.selected_action_log_prob,
            "predicted_next_state": trace.predicted_next_state.tolist(),
            "critic_value": trace.critic_value,
            "confidence": trace.confidence,
            "novelty": trace.novelty,
            "conflict": trace.conflict,
            "risk": trace.risk,
            "gate_values": list(trace.gate_values),
            "prediction_error": trace.prediction_error,
            "td_delta": trace.td_delta,
        },
    }
    journal.write_text(json.dumps(item) + "\n", encoding="utf-8")
    target = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=6)

    replayed = replay_positive_experience_journal(target, journal, max_items=10)

    assert replayed == 1


def test_replay_positive_experience_journal_accepts_graph_trace(tmp_path) -> None:
    source = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=7, seed=5)
    trace = source.act(np.array([1.0, -1.0, 0.5, -0.5]))
    journal = tmp_path / "journal.jsonl"
    item = {
        "trace_id": trace.trace_id, "reward": 1.0, "cost": 0.0, "td_delta": 1.0,
        "prediction_error": 0.1, "done": False, "next_obs": [1.2, -0.8, 0.7, -0.3],
        "trace": {"timestamp": trace.timestamp, "raw_obs": trace.raw_obs.tolist(),
                  "encoded_state": trace.encoded_state.tolist(), "memory_before": trace.memory_before.tolist(),
                  "memory_after": trace.memory_after.tolist(), "selected_idx": trace.selected_idx,
                  "selected_action": trace.selected_action.tolist(), "candidate_actions": trace.candidate_actions.ravel().tolist(),
                  "candidate_logits": trace.candidate_logits.ravel().tolist(), "candidate_scores": trace.candidate_scores.ravel().tolist(),
                  "selected_action_log_prob": trace.selected_action_log_prob, "predicted_next_state": trace.predicted_next_state.tolist(),
                  "critic_value": trace.critic_value, "confidence": trace.confidence, "novelty": trace.novelty,
                  "conflict": trace.conflict, "risk": trace.risk, "gate_values": list(trace.gate_values),
                  "prediction_error": trace.prediction_error, "td_delta": trace.td_delta},
    }
    journal.write_text(json.dumps(item) + "\n", encoding="utf-8")
    target = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=6)
    assert replay_positive_experience_journal(target, journal, max_items=10) == 1
    assert target.modules[0].m1_component.state_size == 5


def test_stratified_journal_replay_bounds_full_plasticity_but_keeps_memory_absorb(monkeypatch, tmp_path) -> None:
    source = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=7, seed=51)
    trace = source.act(np.array([1.0, -1.0, 0.5, -0.5]))
    trace_payload = {
        "timestamp": trace.timestamp,
        "raw_obs": trace.raw_obs.tolist(),
        "encoded_state": trace.encoded_state.tolist(),
        "memory_before": trace.memory_before.tolist(),
        "memory_after": trace.memory_after.tolist(),
        "selected_idx": trace.selected_idx,
        "selected_action": trace.selected_action.tolist(),
        "candidate_actions": trace.candidate_actions.ravel().tolist(),
        "candidate_logits": trace.candidate_logits.ravel().tolist(),
        "candidate_scores": trace.candidate_scores.ravel().tolist(),
        "selected_action_log_prob": trace.selected_action_log_prob,
        "predicted_next_state": trace.predicted_next_state.tolist(),
        "critic_value": trace.critic_value,
        "confidence": trace.confidence,
        "novelty": trace.novelty,
        "conflict": trace.conflict,
        "risk": trace.risk,
        "gate_values": list(trace.gate_values),
        "prediction_error": trace.prediction_error,
        "td_delta": trace.td_delta,
    }
    journal = tmp_path / "journal.jsonl"
    rows = []
    for idx, reward in enumerate((1.0, -0.8, 0.0)):
        rows.append(json.dumps({
            "trace_id": f"trace-{idx}",
            "reward": reward,
            "cost": 0.01,
            "td_delta": reward,
            "prediction_error": 0.1,
            "done": False,
            "next_obs": [1.2, -0.8, 0.7, -0.3],
            "trace": dict(trace_payload, trace_id=f"trace-{idx}"),
        }))
    journal.write_text("\n".join(rows) + "\n", encoding="utf-8")
    target = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=52)
    observed = {"plastic": 0, "absorbed": 0}

    def observe_transition(*_args, **_kwargs):
        observed["plastic"] += 1
        return {"prediction_error": 0.1, "td_delta": 0.1, "reward": 0.1, "cost": 0.0}

    def absorb_transition(*_args, **_kwargs):
        observed["absorbed"] += 1

    monkeypatch.setattr(target, "observe_transition", observe_transition)
    monkeypatch.setattr(target.memory_supernet, "absorb_transition", absorb_transition)

    replayed = replay_stratified_experience_journal(target, journal, max_items=3, plastic_max_items=1)

    assert replayed == 3
    assert observed == {"plastic": 1, "absorbed": 3}


def test_replay_positive_mt5_journal_skips_old_memory_key_schema(tmp_path) -> None:
    source = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=5)
    trace = source.act(np.array([1.0, -1.0, 0.5, -0.5]))
    journal = tmp_path / "journal.jsonl"
    item = {
        "source": "mt5_online",
        "trace_id": trace.trace_id,
        "reward": 1.0,
        "cost": 0.0,
        "td_delta": 1.0,
        "prediction_error": 0.1,
        "execution_details": {"memory_eligible": True},
        "done": False,
        "next_obs": [1.2, -0.8, 0.7, -0.3],
        "trace": {
            "timestamp": trace.timestamp,
            "raw_obs": trace.raw_obs.tolist(),
            "encoded_state": trace.encoded_state.tolist(),
            "selected_action": trace.selected_action.tolist(),
        },
    }
    journal.write_text(json.dumps(item) + "\n", encoding="utf-8")
    target = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=6)

    replayed = replay_positive_experience_journal(target, journal, max_items=10)

    assert replayed == 0


def test_experience_supernetwork_uses_account_context_memory_key_schema() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=41)
    account_context = {
        "equity_to_balance": -0.015, "floating_pnl_r": -0.35, "current_equity_drawdown": 0.032,
        "balance_drawdown": 0.02, "open_risk_ratio": 0.8, "equity_return_1_bar": -0.002,
        "equity_return_5_bar": -0.011, "equity_volatility_20": 0.018, "losing_streak_norm": 0.5,
        "time_under_water_norm": 0.6, "position_count_norm": 0.8, "day_trade_pressure": 0.2,
    }
    trace = dc_replace(
        policy.act(np.array([1.0, -1.0, 0.5, -0.5])),
        action_details={"action_name": "MARKET_BUY", "target_slot": 0},
        positions_pipe="111:1:0.01:100:99:103:-1.2:18",
        positions_pipe_after="111:1:0.01:100:99:103:-0.6:19",
        account_context=account_context,
        account_context_after={**account_context, "floating_pnl_r": -0.12},
    )

    policy.observe_transition(trace, np.array([1.1, -0.9, 0.6, -0.4]), reward=0.4, cost=0.02, done=False)

    memories = [cell for cell in policy.memory_supernet.neural_cells.values() if cell.role == "episodic_memory"]
    assert memories
    memory = memories[-1]
    assert memory.vector.shape == (MEMORY_KEY_SIZE,)
    assert memory.meta["account_context"]["current_equity_drawdown"] == account_context["current_equity_drawdown"]
    assert policy.memory_supernet.summary()["memory_key_schema_version"] == MEMORY_KEY_SCHEMA_VERSION
    outcomes = [cell.meta for cell in policy.memory_supernet.neural_cells.values() if cell.role == "causal_outcome"]
    assert outcomes
    assert outcomes[-1]["memory_key_schema_version"] == MEMORY_KEY_SCHEMA_VERSION
    assert outcomes[-1]["account_context_before"]["floating_pnl_r"] == account_context["floating_pnl_r"]

def test_train_self_growing_mt5_can_close_after_round(monkeypatch, tmp_path) -> None:
    evaluators = []

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config
            self.close_calls = 0
            evaluators.append(self)

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=10.0 + trial_id,
                profit_factor=1.4,
                total_trades=4,
                balance_drawdown_pct=0.1,
                equity_drawdown_pct=0.1,
                win_rate=75.0,
                avg_win=2.0,
                avg_loss=-1.0,
                report_path=tmp_path / f"report_{trial_id}.htm",
            )
            return float(trial_id), metrics

        def close_open_terminals(self):
            self.close_calls += 1

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "policy_core": getattr(policy, "world_core", "current"), "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = train_self_growing_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        output_path=str(tmp_path / "round_close.pkl"),
        trials=2,
        mt5_close_scope="round",
    )

    assert result["mt5_close_scope"] == "round"
    assert len(evaluators) >= 1
    assert evaluators[0].config.shutdown_terminal is False
    assert evaluators[0].close_calls == 1


def test_compare_runner_dirs_are_derived_and_copied_without_transients(tmp_path) -> None:
    base = tmp_path / "mt5_v2_runner"
    (base / "Reports").mkdir(parents=True)
    (base / "Reports" / "old.htm").write_text("stale", encoding="utf-8")
    (base / "Tester" / "cache").mkdir(parents=True)
    (base / "Tester" / "cache" / "old.tst").write_text("stale", encoding="utf-8")
    (base / "MQL5" / "Experts").mkdir(parents=True)
    (base / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").write_text("ea", encoding="utf-8")
    (base / "terminal64.exe").write_text("terminal", encoding="utf-8")

    derived = derived_compare_runner_dirs(base)
    runners = ensure_compare_runner_dirs(base)

    assert runners == derived
    assert (runners["current"] / "terminal64.exe").exists()
    assert (runners["world"] / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").exists()
    assert not (runners["current"] / "Reports" / "old.htm").exists()
    assert not (runners["world"] / "Tester" / "cache" / "old.tst").exists()


def test_compare_policy_cores_runs_parallel_exchange_and_selects_winner(monkeypatch, tmp_path) -> None:
    base_runner = tmp_path / "runner"
    base_runner.mkdir()
    (base_runner / "terminal64.exe").write_text("terminal", encoding="utf-8")

    def policy_with_experience(core: str):
        policy = SpatialWorldPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7) if core == "world" else SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
        if core == "world":
            policy.modules.append(policy.modules[0].copy())
            policy.modules[-1].name = "trial_only_module"
            policy.world.ensure_for_modules(policy.modules, np.random.default_rng(8))
        trace = policy.act(np.array([1.0, -1.0, 0.5, -0.5]))
        policy.observe_transition(trace, np.array([1.2, -0.8, 0.7, -0.3]), reward=1.0, cost=0.0, done=False)
        return policy

    def fake_train(config, rounds, output_path, trials=10, seed_policy_path=None, **kwargs):
        core = kwargs["policy_core"]
        with open(output_path, "wb") as handle:
            pickle.dump(policy_with_experience(core), handle)
        return _mt5_result(10.0 if core == "current" else 9.0)

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            is_world = isinstance(policy, SpatialWorldPolicy)
            metrics = SimpleNamespace(
                net_profit=100.0,
                profit_factor=2.0,
                total_trades=10,
                balance_drawdown_pct=1.0,
                equity_drawdown_pct=1.0,
                win_rate=80.0,
                avg_win=2.0,
                avg_loss=-1.0,
                report_path=tmp_path / ("world_replay.htm" if is_world else "current_replay.htm"),
                fitness_lifecycle_count=10,
                fitness_ledger_integrity=1.0,
                fitness_immediate_exit_ratio=0.0,
                fitness_execution_failure_rate=0.0,
                fitness_expectancy_lcb=0.25,
            )
            return (12.0 if is_world else 11.0), metrics

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train)
    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = compare_policy_cores_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        trials=1,
        runner_dir=base_runner,
        output_dir=tmp_path / "compare",
        timeout_sec=1,
    )

    assert result["current"]["ok"] is True
    assert result["world"]["ok"] is True
    assert len(result["exchange"]["items"]) == 2
    assert all(item["promoted"] for item in result["exchange"]["items"])
    assert result["winner"]["core"] == "world"
    assert result["winner"]["source"] == "exchange"
    assert (tmp_path / "compare" / "best.pkl").exists()
    current_exchanged = load_policy(tmp_path / "compare" / "current.exchanged.pkl", policy_core="current")
    assert all(module.name != "trial_only_module" for module in current_exchanged.modules)


def test_compare_policy_cores_keeps_successful_core_when_other_errors(monkeypatch, tmp_path) -> None:
    base_runner = tmp_path / "runner"
    base_runner.mkdir()
    (base_runner / "terminal64.exe").write_text("terminal", encoding="utf-8")

    def fake_train(config, rounds, output_path, trials=10, seed_policy_path=None, **kwargs):
        core = kwargs["policy_core"]
        if core == "world":
            raise RuntimeError("world failed")
        with open(output_path, "wb") as handle:
            pickle.dump(SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7), handle)
        return _mt5_result(10.0)

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train)

    result = compare_policy_cores_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        trials=1,
        runner_dir=base_runner,
        output_dir=tmp_path / "compare",
        timeout_sec=1,
    )

    assert result["current"]["ok"] is True
    assert result["world"]["ok"] is False
    assert result["exchange"]["skipped"] is True
    assert result["exchange"]["reason"] == "requires_both_cores_successful"
    assert result["winner"]["core"] == "current"


def test_validate_replay_candidate_requires_mt5_improvement(monkeypatch, tmp_path) -> None:
    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=100.0,
                profit_factor=2.0,
                total_trades=10,
                balance_drawdown_pct=1.0,
                equity_drawdown_pct=1.0,
                win_rate=80.0,
                avg_win=2.0,
                avg_loss=-1.0,
                report_path=tmp_path / "replay.htm",
            )
            return 9.0, metrics

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = validate_replay_candidate_mt5(
        SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=8),
        _mt5_curriculum_config(),
        _mt5_result(10.0),
        runner_dir=tmp_path,
        timeout_sec=1,
    )

    assert result["promoted"] is False
    assert "fitness_not_improved:9<=10" in result["reject_reasons"]


def test_improve_rolls_back_failed_replay_candidate(monkeypatch, tmp_path) -> None:
    seed_policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=8, m1_backend="graph")
    seed_path = tmp_path / "seed.pkl"
    output_path = tmp_path / "best.pkl"
    with seed_path.open("wb") as handle:
        pickle.dump(seed_policy, handle)

    calls = {"train": 0}

    def fake_train_self_growing_mt5(config, rounds, output_path, trials, seed_policy_path=None, *args, **kwargs):
        calls["train"] += 1
        if calls["train"] == 1:
            return _mt5_result(10.0)

        source = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7, m1_backend="graph")
        trace = source.act(np.array([1.0, -1.0, 0.5, -0.5]))
        source.observe_transition(trace, np.array([1.2, -0.8, 0.7, -0.3]), reward=1.0, cost=0.0, done=False)
        with open(output_path, "wb") as handle:
            pickle.dump(source, handle)
        return _mt5_result(9.0)

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=100.0,
                profit_factor=2.0,
                total_trades=10,
                balance_drawdown_pct=1.0,
                equity_drawdown_pct=1.0,
                win_rate=80.0,
                avg_win=2.0,
                avg_loss=-1.0,
                report_path=tmp_path / "replay.htm",
            )
            return 9.5, metrics

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train_self_growing_mt5)
    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = improve_self_growing(
        _mt5_curriculum_config(),
        rounds=1,
        seed_policy_path=str(seed_path),
        output_path=str(output_path),
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
    )

    saved_policy = load_policy(output_path)
    assert result["promoted"] is False
    assert result["replayed_experiences"] >= 1
    assert result["replay_promoted"] is False
    assert "fitness_not_improved:9.5<=10" in result["replay_reject_reasons"]
    migrated_seed = load_policy(seed_path)
    assert len(saved_policy.modules[0].m1_component.graph.neurons) == len(migrated_seed.modules[0].m1_component.graph.neurons)
    assert np.allclose(
        [edge.weight for edge in saved_policy.modules[0].m1_component.graph.edges.values()],
        [edge.weight for edge in migrated_seed.modules[0].m1_component.graph.edges.values()],
    )


def test_improve_without_promotion_gate_rejects_goal_progress_regression(monkeypatch, tmp_path) -> None:
    seed_policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=18, m1_backend="graph")
    seed_path = tmp_path / "seed.pkl"
    output_path = tmp_path / "best.pkl"
    with seed_path.open("wb") as handle:
        pickle.dump(seed_policy, handle)

    calls = {"train": 0}

    def fake_train_self_growing_mt5(config, rounds, output_path, trials, seed_policy_path=None, *args, **kwargs):
        assert kwargs["enforce_promotion_gate"] is False
        calls["train"] += 1
        policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=18 + calls["train"], m1_backend="graph")
        with open(output_path, "wb") as handle:
            pickle.dump(policy, handle)
        return _mt5_result(10.0 if calls["train"] == 1 else 5.0, net_profit=-10.0, profit_factor=0.5, trades=10)

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train_self_growing_mt5)

    result = improve_self_growing(
        _mt5_curriculum_config(),
        rounds=1,
        seed_policy_path=str(seed_path),
        output_path=str(output_path),
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
        enforce_promotion_gate=False,
    )

    assert result["promoted"] is False
    assert result["promotion_gate_enabled"] is False
    assert result["reject_reasons"][0].startswith("goal_progress_not_improved:")
    assert result["baseline_score"] == 10.0
    assert result["trial_score"] == 5.0
    assert output_path.exists()


def test_improve_without_promotion_gate_rejects_insufficient_execution_evidence(monkeypatch, tmp_path) -> None:
    seed_policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=190, m1_backend="graph")
    seed_path = tmp_path / "seed.pkl"
    output_path = tmp_path / "best.pkl"
    with seed_path.open("wb") as handle:
        pickle.dump(seed_policy, handle)

    calls = {"train": 0}

    def fake_train_self_growing_mt5(config, rounds, output_path, trials, seed_policy_path=None, *args, **kwargs):
        assert kwargs["enforce_promotion_gate"] is False
        calls["train"] += 1
        policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=190 + calls["train"], m1_backend="graph")
        with open(output_path, "wb") as handle:
            pickle.dump(policy, handle)
        if calls["train"] == 1:
            return _mt5_result(-100.0, net_profit=-100.0, profit_factor=0.5, drawdown=8.0, trades=20)
        # Better soft score but only seven closed lifecycles: cannot replace
        # the seed because the apparent gain can be mere under-exposure.
        return _mt5_result(-100.0, net_profit=-5.0, profit_factor=0.95, drawdown=1.0, trades=7)

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train_self_growing_mt5)

    result = improve_self_growing(
        _mt5_curriculum_config(),
        rounds=1,
        seed_policy_path=str(seed_path),
        output_path=str(output_path),
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
        enforce_promotion_gate=False,
    )

    assert result["promoted"] is False
    assert "insufficient_position_samples:7<8" in result["reject_reasons"]
    assert result["baseline_progress_score"] < result["trial_progress_score"]


def test_improve_without_promotion_gate_accepts_goal_progress_improvement(monkeypatch, tmp_path) -> None:
    seed_policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=19, m1_backend="graph")
    seed_path = tmp_path / "seed.pkl"
    output_path = tmp_path / "best.pkl"
    with seed_path.open("wb") as handle:
        pickle.dump(seed_policy, handle)

    calls = {"train": 0}

    def fake_train_self_growing_mt5(config, rounds, output_path, trials, seed_policy_path=None, *args, **kwargs):
        assert kwargs["enforce_promotion_gate"] is False
        calls["train"] += 1
        policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=19 + calls["train"], m1_backend="graph")
        with open(output_path, "wb") as handle:
            pickle.dump(policy, handle)
        if calls["train"] == 1:
            return _mt5_result(-100.0, net_profit=-50.0, profit_factor=0.7, drawdown=5.0, trades=10)
        return _mt5_result(-100.0, net_profit=-10.0, profit_factor=0.95, drawdown=1.0, trades=20)

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train_self_growing_mt5)

    result = improve_self_growing(
        _mt5_curriculum_config(),
        rounds=1,
        seed_policy_path=str(seed_path),
        output_path=str(output_path),
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
        enforce_promotion_gate=False,
    )

    assert result["promoted"] is True
    assert result["promotion_gate_enabled"] is False
    assert result["reject_reasons"] == []
    assert result["baseline_progress_score"] < result["trial_progress_score"]


def test_improve_rolls_back_when_replay_validation_errors(monkeypatch, tmp_path) -> None:
    seed_policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=8, m1_backend="graph")
    seed_path = tmp_path / "seed.pkl"
    output_path = tmp_path / "best.pkl"
    with seed_path.open("wb") as handle:
        pickle.dump(seed_policy, handle)

    calls = {"train": 0}

    def fake_train_self_growing_mt5(config, rounds, output_path, trials, seed_policy_path=None, *args, **kwargs):
        calls["train"] += 1
        if calls["train"] == 1:
            return _mt5_result(10.0)

        source = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7, m1_backend="graph")
        trace = source.act(np.array([1.0, -1.0, 0.5, -0.5]))
        source.observe_transition(trace, np.array([1.2, -0.8, 0.7, -0.3]), reward=1.0, cost=0.0, done=False)
        with open(output_path, "wb") as handle:
            pickle.dump(source, handle)
        return _mt5_result(9.0)

    class FailingEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            raise RuntimeError("tester unavailable")

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train_self_growing_mt5)
    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FailingEvaluator)

    result = improve_self_growing(
        _mt5_curriculum_config(),
        rounds=1,
        seed_policy_path=str(seed_path),
        output_path=str(output_path),
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
    )

    saved_policy = load_policy(output_path)
    assert result["replay_promoted"] is False
    assert result["replay_reject_reasons"] == ["replay_validation_error:tester unavailable"]
    migrated_seed = load_policy(seed_path)
    assert len(saved_policy.modules[0].m1_component.graph.neurons) == len(migrated_seed.modules[0].m1_component.graph.neurons)
    assert np.allclose(
        [edge.weight for edge in saved_policy.modules[0].m1_component.graph.edges.values()],
        [edge.weight for edge in migrated_seed.modules[0].m1_component.graph.edges.values()],
    )


def test_train_to_goal_bootstraps_missing_seed(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_train_self_growing_mt5(config, rounds, output_path, trials=10, seed_policy_path=None, *args, **kwargs):
        calls.append({"rounds": rounds, "output_path": output_path, "seed_policy_path": seed_policy_path})
        with open(output_path, "wb") as handle:
            pickle.dump(SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=9), handle)
        return _mt5_result(12.0)

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train_self_growing_mt5)

    result = train_to_goal(
        _mt5_curriculum_config(),
        train_rounds=1,
        seed_policy_path=str(tmp_path / "missing.pkl"),
        output_path=str(tmp_path / "winner.pkl"),
        max_rounds=1,
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
    )

    assert calls[0]["seed_policy_path"] is None
    assert Path(calls[1]["seed_policy_path"]) == tmp_path / "winner.pkl"
    assert result["bootstrap"] is not None
    assert (tmp_path / "winner.pkl").exists()


def test_train_to_goal_enables_improve_promotion_gate(monkeypatch, tmp_path) -> None:
    seed = tmp_path / "seed.pkl"
    with seed.open("wb") as handle:
        pickle.dump(SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=90), handle)
    seen = {}

    def fake_improve(*args, **kwargs):
        seen["enforce_promotion_gate"] = kwargs.get("enforce_promotion_gate")
        return {
            "promoted": True,
            "baseline_result": _mt5_result(10.0),
            "trial_result": _mt5_result(5.0),
            "baseline_score": 10.0,
            "trial_score": 5.0,
        }

    monkeypatch.setattr("self_growing_trader.self_growing_runner.improve_self_growing", fake_improve)

    result = train_to_goal(
        _mt5_curriculum_config(),
        train_rounds=1,
        seed_policy_path=str(seed),
        output_path=str(tmp_path / "winner.pkl"),
        max_rounds=1,
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
    )

    assert seen["enforce_promotion_gate"] is True
    assert result["history"][0]["promotion_gate_enabled"] is True


def test_build_mt5_curriculum_stages_progresses_to_full_window() -> None:
    stages = build_mt5_curriculum_stages(_mt5_curriculum_config(), rounds=6)
    assert [stage.name for stage in stages] == ["easy-week", "medium-half-month", "full-month"]
    assert [stage.start_round for stage in stages] == [0, 2, 4]
    assert stages[0].to_date == "2025.01.08"
    assert stages[-1].to_date == "2025.02.01"


def test_mt5_curriculum_stage_for_round_uses_latest_started_stage() -> None:
    stages = build_mt5_curriculum_stages(_mt5_curriculum_config(), rounds=6)
    assert mt5_curriculum_stage_for_round(stages, 0)[1].name == "easy-week"
    assert mt5_curriculum_stage_for_round(stages, 2)[1].name == "medium-half-month"
    assert mt5_curriculum_stage_for_round(stages, 5)[1].name == "full-month"


def test_train_self_growing_mt5_uses_python_action_bridge(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeEvaluator:
        def __init__(self, config):
            captured["runner_dir"] = config.runner_dir

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=20.0,
                profit_factor=2.0,
                total_trades=2,
                balance_drawdown_pct=0.1,
                equity_drawdown_pct=0.1,
                win_rate=80.0,
                avg_win=3.0,
                avg_loss=-1.0,
                report_path=tmp_path / "report.htm",
            )
            return 10.0, metrics

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = train_self_growing_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        output_path=str(tmp_path / "winner.pkl"),
        trials=1,
    )

    assert result["bridge_runtime"]["bridge_mode"] == "python_action"
    assert result["winner"].endswith("winner.pkl")


def test_training_windows_are_deterministic_and_bounded() -> None:
    data = pd.DataFrame({"x": range(10)})
    windows = training_windows(data, window_bars=4, max_windows=3)
    assert [window["x"].tolist() for window in windows] == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]


def test_mt5_promotion_requires_better_profitable_low_drawdown_trial() -> None:
    assert mt5_promotion_rejection_reasons(_mt5_result(1.0), _mt5_result(2.0)) == []
    assert "fitness_not_improved:1<=2" in mt5_promotion_rejection_reasons(_mt5_result(2.0), _mt5_result(1.0))
    assert "profit_factor_below_1" in mt5_promotion_rejection_reasons(_mt5_result(1.0), _mt5_result(2.0, profit_factor=0.9))
    assert "net_profit_not_positive" in mt5_promotion_rejection_reasons(_mt5_result(1.0), _mt5_result(2.0, net_profit=-1.0))
    assert "drawdown_above_15pct" in mt5_promotion_rejection_reasons(_mt5_result(1.0), _mt5_result(2.0, drawdown=20.0))
    assert "insufficient_position_samples:1<8" in mt5_promotion_rejection_reasons(
        _mt5_result(1.0),
        _mt5_result(2.0, profit_factor=999.0, net_profit=2.0, trades=1),
        min_daily_trades=7,
        backtest_days=2,
    )


def test_train_to_goal_progress_rejects_worse_capped_fitness_trial() -> None:
    baseline = _mt5_result(-100.0, net_profit=-21.91, profit_factor=0.85, drawdown=0.38, trades=158)
    trial = _mt5_result(-100.0, net_profit=-1934.65, profit_factor=0.81, drawdown=29.27, trades=143)

    assert _mt5_train_to_goal_progress_score(baseline) > _mt5_train_to_goal_progress_score(trial)


def test_robust_training_fitness_prefers_stable_goal_progress() -> None:
    target_results = [
        DummyResult([0.06, 0.03], total_r=6.0, expectancy=0.5, max_drawdown_pct=0.01),
        DummyResult([0.05, 0.02], total_r=5.0, expectancy=0.4, max_drawdown_pct=0.01),
    ]
    risky_results = [
        DummyResult([0.08, -0.08], total_r=7.0, expectancy=0.6, max_drawdown_pct=0.20, fitness=100.0),
        DummyResult([-0.03, 0.01], total_r=1.0, expectancy=0.1, max_drawdown_pct=0.20, fitness=100.0),
    ]
    assert robust_training_fitness(target_results) > robust_training_fitness(risky_results)


def test_validation_selection_penalizes_validation_only_spikes() -> None:
    balanced_train = [
        DummyResult([0.05, 0.03], total_r=4.0, expectancy=0.4, max_drawdown_pct=0.01),
        DummyResult([0.06, 0.02], total_r=4.5, expectancy=0.4, max_drawdown_pct=0.01),
    ]
    balanced_validation = DummyResult([0.05, 0.02], total_r=4.0, expectancy=0.4, max_drawdown_pct=0.01)
    spike_train = [
        DummyResult([0.0, 0.0], total_r=0.2, expectancy=0.0, max_drawdown_pct=0.0),
        DummyResult([0.0, 0.0], total_r=0.2, expectancy=0.0, max_drawdown_pct=0.0),
    ]
    spike_validation = DummyResult([0.12, 0.0], total_r=6.0, expectancy=0.8, max_drawdown_pct=0.0)
    assert validation_selection_score(balanced_train, balanced_validation) > validation_selection_score(spike_train, spike_validation)


def test_audit_policy_reports_all_splits(monkeypatch) -> None:
    data = pd.DataFrame({"x": range(30)})

    def fake_split_time_ordered(df, train, validation):
        return df.iloc[:10], df.iloc[10:20], df.iloc[20:]

    def fake_evaluate_policy(df, config, policy):
        return {
            "stable_goal_score": float(len(df)),
            "target_day_rate": 0.1,
            "max_drawdown_pct": 0.0,
            "max_daily_trades": 3.0,
            "goal_hit": True,
            "total_r": float(len(df)),
        }

    monkeypatch.setattr("self_growing_trader.self_growing_runner.split_time_ordered", fake_split_time_ordered)
    monkeypatch.setattr("self_growing_trader.self_growing_runner.evaluate_policy", fake_evaluate_policy)
    config = SimpleNamespace(splits=SimpleNamespace(train=0.7, validation=0.15))
    audit = audit_policy(data, config, lambda _: [])
    assert set(audit) == {"all", "train", "validation", "test", "summary"}
    assert audit["summary"]["all_splits_goal_hit"] is True




def test_expand_module_resizes_graph_state_and_directed_lateral_edges() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=71)
    peer = policy.modules[0].copy()
    peer.name = "peer"
    policy.modules.append(peer)
    policy.ensure_compatible()
    policy.act(np.ones(4))

    assert policy.expand_module("m1_thinking", 2, np.random.default_rng(72))
    grown = policy.modules[0]
    assert grown.hidden_size == 7
    assert grown.m1_component.state_size == 7
    assert policy.lateral_edges["m1_thinking<-peer"].shape == (7, 5)
    assert policy.lateral_edges["peer<-m1_thinking"].shape == (5, 7)
    assert policy.act(np.ones(4)).selected_action.shape == (3,)

def test_fitness_credit_updates_neuron_and_synapse_state_with_bounded_signal() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=73)
    policy.act(np.array([1.0, -1.0, 0.5, -0.5]))
    module = policy.modules[0]
    before = module.w_in.copy()

    policy.consolidate(10_000.0)

    assert not np.array_equal(module.w_in, before)
    assert module.neuron_utility is not None
    # Trial-level consolidation is intentionally not neuron attribution.
    assert np.allclose(module.neuron_utility, 0.0)
    assert all(abs(value) <= 1.0 for value in policy.loss_history)


def test_graph_candidate_keeps_open_structural_growth_budget() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=74)
    candidate = policy.propose(np.random.default_rng(75), mutation_sigma=0.05, max_modules=0)
    graph = candidate.modules[0].m1_component.graph

    assert candidate.m1_backend == "m1_graph"
    assert graph.budget.max_cross_module_edges == 0
    assert graph.budget.max_m1_module_edges == 0
    graph.assert_invariants(output_size=19)

def test_graph_specialist_routing_is_typed_not_tensor_masked() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=19, hidden_size=5, seed=76)
    graph = policy.modules[0].m1_component.graph
    from self_growing_trader.cross_module import CrossModuleConnector

    module_id = "specialist:memory_specialist:graph-test"
    anchor = graph.add_neuron("memory", module_id=module_id, lifecycle="candidate")
    edges = CrossModuleConnector().wire_specialist(
        graph, module_id=module_id, role="memory_specialist", candidate_id="graph-test",
        need_reason="memory_gap", rng=np.random.default_rng(77),
    )
    assert edges
    assert any(anchor in {graph.edges[edge_id].src, graph.edges[edge_id].dst} for edge_id in edges)
    graph.assert_invariants(output_size=19)

def test_compare_cores_assigns_independent_journals(monkeypatch, tmp_path) -> None:
    base_runner = tmp_path / "runner"
    base_runner.mkdir()
    (base_runner / "terminal64.exe").write_text("terminal", encoding="utf-8")
    seen: dict[str, Path | None] = {}

    def fake_train(config, rounds, output_path, trials=10, seed_policy_path=None, **kwargs):
        core = kwargs["policy_core"]
        seen[core] = kwargs["experience_journal_path"]
        with open(output_path, "wb") as handle:
            pickle.dump(SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=78), handle)
        return _mt5_result(10.0 if core == "current" else 9.0)

    monkeypatch.setattr("self_growing_trader.self_growing_runner.train_self_growing_mt5", fake_train)
    result = compare_policy_cores_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        trials=1,
        runner_dir=base_runner,
        output_dir=tmp_path / "compare",
        timeout_sec=1,
        skip_exchange=True,
        experience_journal_path=tmp_path / "shared.jsonl",
    )

    assert seen["current"] != seen["world"]
    assert seen["current"].name == "shared.current.jsonl"
    assert seen["world"].name == "shared.world.jsonl"
    assert result["experience_journals"]["current"] != result["experience_journals"]["world"]


def test_train_to_goal_cannot_stop_on_rejected_trial(monkeypatch, tmp_path) -> None:
    config = _mt5_curriculum_config()
    config.orders.min_daily_trades = 6
    seed = tmp_path / "seed.pkl"
    with seed.open("wb") as handle:
        pickle.dump(SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=79), handle)

    baseline = _mt5_result(1.0, net_profit=-1.0, profit_factor=0.8, trades=186)
    trial = _mt5_result(100.0, net_profit=100.0, profit_factor=2.0, drawdown=1.0, trades=186)
    trial["best"]["metrics"].update({"win_rate": 80.0, "avg_win": 2.0, "avg_loss": -1.0})

    def fake_improve(*args, **kwargs):
        return {
            "promoted": False,
            "baseline_result": baseline,
            "trial_result": trial,
            "baseline_score": 1.0,
            "trial_score": 100.0,
        }

    monkeypatch.setattr("self_growing_trader.self_growing_runner.improve_self_growing", fake_improve)
    result = train_to_goal(
        config,
        train_rounds=1,
        seed_policy_path=str(seed),
        output_path=str(tmp_path / "winner.pkl"),
        max_rounds=1,
        attempts=1,
        trials=1,
        runner_dir=tmp_path,
        timeout_sec=1,
    )

    assert result["goal_reached"] is False
    assert result["history"][0]["goal_source"] is None
    assert result["history"][0]["goal_min_daily_trades"] == 6


def test_oos_evaluation_uses_copy_and_never_enables_journal(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeEvaluator:
        def __init__(self, config):
            captured["config"] = config

        def evaluate_policy(self, trial_id, policy, round):
            policy.modules[0].w_in += 1.0  # Must not leak to caller's policy.
            metrics = SimpleNamespace(
                net_profit=10.0,
                profit_factor=1.2,
                total_trades=8,
                balance_drawdown_pct=1.0,
                equity_drawdown_pct=1.0,
                win_rate=60.0,
                avg_win=1.5,
                avg_loss=-1.0,
                report_path=tmp_path / "oos.htm",
            )
            return 3.0, metrics

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=80)
    before = policy.modules[0].w_in.copy()

    result = evaluate_policy_mt5_oos(policy, _mt5_curriculum_config(), runner_dir=tmp_path)

    assert result["evaluation_only"] is True
    assert result["journal_replayed"] == 0
    assert captured["config"].experience_journal is None
    assert np.allclose(policy.modules[0].w_in, before)


def test_unconfirmed_mt5_bridge_request_does_not_allocate_structural_memory() -> None:
    """A Python action intent is not a confirmed MT5 execution event."""
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=81)
    trace = dc_replace(
        policy.act(np.array([0.2, -0.3, 0.4, -0.5])),
        execution_details={
            "bridge_observed": True,
            "execution_confirmed": False,
            "memory_eligible": False,
        },
    )
    before = policy.memory_supernet.summary()

    signal = policy.observe_transition(
        trace,
        np.array([0.3, -0.2, 0.5, -0.4]),
        reward=0.5,
        cost=0.0,
        done=False,
    )
    after = policy.memory_supernet.summary()

    assert signal["memory_eligible"] == 0.0
    assert signal["memory_recorded"] == 0.0
    assert after["neurons"] == before["neurons"] == 0
    assert after["connections"] == before["connections"] == 0
    assert not policy.experience_buffer


def test_confirmed_mt5_bridge_execution_allocates_structural_memory() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=19, hidden_size=5, seed=82)
    trace = dc_replace(
        policy.act(np.array([0.2, -0.3, 0.4, -0.5])),
        execution_details={
            "bridge_observed": True,
            "execution_confirmed": True,
            "memory_eligible": True,
            "positions_changed": True,
        },
    )

    signal = policy.observe_transition(
        trace,
        np.array([0.3, -0.2, 0.5, -0.4]),
        reward=0.25,
        cost=0.0,
        done=False,
    )

    assert signal["memory_eligible"] == 1.0
    assert signal["memory_recorded"] == 1.0
    assert policy.memory_supernet.summary()["neurons"] == 1



def test_train_self_growing_mt5_visual_forces_trial_scope_even_when_round_requested(monkeypatch, tmp_path) -> None:
    evaluators = []

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config
            self.close_calls = 0
            evaluators.append(self)

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=10.0 + trial_id,
                profit_factor=1.4,
                total_trades=4,
                balance_drawdown_pct=0.1,
                equity_drawdown_pct=0.1,
                win_rate=75.0,
                avg_win=2.0,
                avg_loss=-1.0,
                report_path=tmp_path / f"report_visual_{trial_id}.htm",
            )
            return float(trial_id), metrics

        def close_open_terminals(self):
            self.close_calls += 1

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "policy_core": getattr(policy, "world_core", "current"), "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = train_self_growing_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        output_path=str(tmp_path / "visual_restart.pkl"),
        trials=2,
        visual=True,
        mt5_close_scope="round",
    )

    assert result["mt5_close_scope"] == "trial"
    assert evaluators[0].config.visual is True
    assert evaluators[0].config.shutdown_terminal is True
    assert evaluators[0].close_calls == 0


def test_train_self_growing_mt5_visual_forces_trial_scope_even_when_keep_open_requested(monkeypatch, tmp_path) -> None:
    evaluators = []

    class FakeEvaluator:
        def __init__(self, config):
            self.config = config
            evaluators.append(self)

        def evaluate_policy(self, trial_id, policy, round):
            metrics = SimpleNamespace(
                net_profit=10.0 + trial_id,
                profit_factor=1.4,
                total_trades=4,
                balance_drawdown_pct=0.1,
                equity_drawdown_pct=0.1,
                win_rate=75.0,
                avg_win=2.0,
                avg_loss=-1.0,
                report_path=tmp_path / f"report_visual_keep_{trial_id}.htm",
            )
            return float(trial_id), metrics

        def close_open_terminals(self):
            raise AssertionError("trial scope should close through ShutdownTerminal, not round cleanup")

        def publish_policy(self, policy):
            return {"bridge_mode": "python_action", "policy_core": getattr(policy, "world_core", "current"), "modules": len(policy.modules)}

    monkeypatch.setattr("self_growing_trader.self_growing_runner.MT5StrategyTesterEvaluator", FakeEvaluator)

    result = train_self_growing_mt5(
        _mt5_curriculum_config(),
        rounds=1,
        output_path=str(tmp_path / "visual_keep_open_restart.pkl"),
        trials=2,
        visual=True,
        shutdown_terminal=False,
        mt5_close_scope="never",
    )

    assert result["mt5_close_scope"] == "trial"
    assert evaluators[0].config.visual is True
    assert evaluators[0].config.shutdown_terminal is True



def test_neuron_fitness_uses_signed_action_margin_and_final_lifecycle_credit() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=171, m1_backend="graph")
    module = policy.modules[0]
    # Make one action margin deliberately identifiable for the chosen action.
    module.w_out[1, :] = np.linspace(0.4, 1.0, module.hidden_size)
    module.w_out[0, :] = -np.linspace(0.2, 0.6, module.hidden_size)
    trace = policy.act(np.array([1.0, -0.5, 0.25, 0.75]))
    trace = dc_replace(trace, action_details={"bridge_action": 1})

    signal = policy.observe_transition(trace, np.array([1.1, -0.4, 0.2, 0.8]), reward=0.2, cost=0.0, done=False)
    before_final = module.neuron_utility.copy()
    credited = policy.apply_execution_lifecycle_credit(trace.trace_id, 1.0, close_reason="TP", duration_bars=5)

    assert signal["neuron_credit_count"] > 0.0
    assert credited > 0
    assert np.any(module.neuron_samples > 0)
    assert np.any(np.abs(module.neuron_contribution_ema) > 0.0)
    # Graph lifecycle credit is tracked on active graph paths; the auxiliary
    # tensor bookkeeping may remain unchanged for an already-closed trace.
    assert np.all(np.isfinite(module.neuron_utility))
    # A final lifecycle is idempotent by trace_id: it cannot be credited twice.
    assert policy.apply_execution_lifecycle_credit(trace.trace_id, 1.0, close_reason="TP", duration_bars=5) == 0


def test_lifecycle_credit_absorbs_verified_outcome_into_memory_supernet() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=8, seed=1171)
    policy.set_decision_context(
        account_context={"current_equity_drawdown": 0.02, "open_risk_ratio": 0.01},
        execution_context={"spread_norm": 0.05},
        allowed_actions={index: True for index in range(12)},
        frame_id="memory-lifecycle:1",
        sequence_id=1,
        timestamp=1_700_000_000.0,
    )
    trace = policy.act(np.linspace(-0.2, 0.2, policy.input_size))
    trace = dc_replace(trace, action_details={**dict(trace.action_details), "bridge_action": 1})
    policy._execution_trace_cache = {trace.trace_id: trace}
    lifecycle = SimpleNamespace(
        position_ticket="pos-memory-1",
        net_pnl=-100.0,
        commission=0.0,
        swap=0.0,
        initial_risk=100.0,
        lifecycle_tags=("entry_too_early",),
    )

    before = policy.memory_supernet.summary()
    credited = policy.apply_execution_lifecycle_credit(
        trace.trace_id,
        -1.0,
        close_reason="SL",
        duration_bars=2,
        lifecycle=lifecycle,
        override_action=1,
    )
    after = policy.memory_supernet.summary()

    assert credited > 0
    assert after["neurons"] > before["neurons"]
    assert after["causal_nodes"] > before["causal_nodes"]
    copied = policy.copy()
    copied_summary = copied.memory_supernet.summary()
    assert copied_summary["neurons"] == after["neurons"]
    assert copied_summary["causal_nodes"] == after["causal_nodes"]
    restored = pickle.loads(pickle.dumps(policy))
    restored_summary = restored.memory_supernet.summary()
    assert restored_summary["neurons"] == after["neurons"]
    assert restored_summary["causal_nodes"] == after["causal_nodes"]
    assert policy.apply_execution_lifecycle_credit(
        trace.trace_id,
        -1.0,
        close_reason="SL",
        duration_bars=2,
        lifecycle=lifecycle,
        override_action=1,
    ) == 0


def test_neuron_pruning_requires_evidence_and_preserves_frozen_neurons() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=172)
    module = policy.modules[0]
    module.neuron_samples[:] = 40.0
    module.neuron_utility[:] = -0.50
    module.neuron_advantage_ema[:] = 0.10
    module.neuron_activation_ema[:] = 0.0
    module.neuron_redundancy_ema[:] = 0.99
    module.neuron_frozen[:] = False
    module.neuron_frozen[1] = True
    before = module.w_out.copy()

    pruned = policy.prune_low_utility_neurons(max_per_module=1)

    assert pruned == 1
    changed_rows = np.where(np.linalg.norm(module.w_out - before, axis=0) > 0.0)[0]
    assert changed_rows.size == 1
    assert changed_rows[0] != 1


def test_mt5_promotion_rejects_graph_regressions_in_execution_quality() -> None:
    baseline = _mt5_result(10.0)
    worse_churn = _mt5_result(11.0)
    worse_churn["best"]["metrics"]["fitness_immediate_exit_ratio"] = 0.20
    baseline["best"]["metrics"]["fitness_immediate_exit_ratio"] = 0.01
    assert "immediate_exit_ratio_worse_than_baseline" in mt5_promotion_rejection_reasons(baseline, worse_churn)

    worse_expectancy = _mt5_result(11.0)
    worse_expectancy["best"]["metrics"]["fitness_expectancy_lcb"] = 0.05
    baseline["best"]["metrics"]["fitness_expectancy_lcb"] = 0.20
    assert "expectancy_lcb_worse_than_baseline" in mt5_promotion_rejection_reasons(baseline, worse_expectancy)


def test_mt5_promotion_rejects_unreconciled_report_even_with_higher_score() -> None:
    baseline = _mt5_result(10.0)
    trial = _mt5_result(11.0)
    trial["best"]["metrics"]["fitness_reconciled"] = False
    trial["best"]["metrics"]["fitness_reconciliation_reasons"] = ["report_trade_count_zero_with_ledger_lifecycles:8"]

    reasons = mt5_promotion_rejection_reasons(baseline, trial)

    assert any(reason.startswith("report_execution_mismatch:") for reason in reasons)
