from __future__ import annotations

import numpy as np

from self_growing_trader.context_frame import (
    ACCOUNT_SLICE,
    EXECUTION_SLICE,
    MARKET_EXTRA_SLICE,
    POSITION_SLOTS_SLICE,
    TIMING_SLICE,
    WYCKOFF_SLICE,
    ContextFrame,
)
from self_growing_trader.m1_graph_module import M1GraphModule
from self_growing_trader.self_growing_runner import (
    INPUT_COUNT,
    MEMORY_KEY_SCHEMA_VERSION,
    SelfGrowingPolicy,
    _component_input_mask,
)


def test_context_frame_keeps_every_semantic_bridge_group() -> None:
    observation = np.zeros(INPUT_COUNT, dtype=float)
    observation[0] = 0.11
    observation[MARKET_EXTRA_SLICE.start] = 0.22
    observation[POSITION_SLOTS_SLICE.start] = 0.33
    observation[ACCOUNT_SLICE.start] = 0.44
    observation[EXECUTION_SLICE.start] = 0.55
    observation[TIMING_SLICE.start] = 0.66
    observation[WYCKOFF_SLICE.start] = 0.77

    frame = ContextFrame.from_observation(
        observation,
        frame_id="schema:all-groups",
        sequence_id=1,
        timestamp=1.0,
        execution_context={"spread_points": 20.0, "position_capacity_remaining": 0.5},
    )

    assert frame.market_vector[0] == 0.11
    assert frame.market_vector[29] == 0.22
    assert frame.position_vector[0] == 0.33
    assert frame.account_vector[0] == 0.44
    assert frame.execution_vector[0] == 0.55
    assert frame.market_vector[58] == 0.66
    assert frame.market_vector[62] == 0.77
    assert np.linalg.norm(frame.canonical_memory_key[:32]) > 0.0
    assert np.linalg.norm(frame.canonical_memory_key[32:40]) > 0.0
    assert np.linalg.norm(frame.canonical_memory_key[40:]) > 0.0


def test_component_masks_are_semantic_not_percentage_ranges() -> None:
    encoder = _component_input_mask(INPUT_COUNT, ("state_encoder",))
    memory = _component_input_mask(INPUT_COUNT, ("working_memory",))
    predictor = _component_input_mask(INPUT_COUNT, ("predictor",))
    critic = _component_input_mask(INPUT_COUNT, ("critic",))

    assert encoder[0] == 1.0
    assert encoder[MARKET_EXTRA_SLICE.start] == 1.0
    assert encoder[EXECUTION_SLICE.start] == 1.0
    assert encoder[TIMING_SLICE.start] == 1.0
    assert encoder[WYCKOFF_SLICE.start] == 1.0
    assert encoder[POSITION_SLOTS_SLICE.start] == 0.0
    assert encoder[ACCOUNT_SLICE.start] == 0.0

    assert memory[POSITION_SLOTS_SLICE.start] == 1.0
    assert memory[ACCOUNT_SLICE.start] == 1.0
    assert memory[EXECUTION_SLICE.start] == 0.0

    assert predictor[POSITION_SLOTS_SLICE.start] == 1.0
    assert predictor[EXECUTION_SLICE.start] == 1.0
    assert critic[0] == 1.0 and critic[-1] == 1.0


def test_context_clock_advances_once_per_frame_and_not_on_outcome() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=5, seed=51)
    frame = ContextFrame.from_observation(
        np.linspace(-0.5, 0.5, INPUT_COUNT),
        frame_id="clock:one",
        sequence_id=1,
        timestamp=1.0,
    )

    first = policy.memory_supernet.advance_context_frame(frame)
    step_after_first = int(policy.memory_supernet.context_step)
    second = policy.memory_supernet.advance_context_frame(frame)
    policy.memory_supernet._observe_context(first.canonical_memory_key, advance=True)

    assert step_after_first == 1
    assert int(policy.memory_supernet.context_step) == 1
    assert np.allclose(first.temporal_delta, second.temporal_delta)
    assert first.temporal_ready and second.temporal_ready


def test_memory_queries_use_active_canonical_frame_not_raw_or_latent_space() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=5, seed=52)
    policy.set_decision_context(
        allowed_actions={0: True, 1: True, 2: True},
        frame_id="canonical:one",
        sequence_id=1,
        goal=np.array([0.2, 0.4]),
    )
    trace = policy.act(np.linspace(-0.25, 0.25, INPUT_COUNT))

    from_raw = policy._query_memory(np.zeros(INPUT_COUNT))
    from_latent = policy._query_memory(np.ones(20))

    assert trace.context_schema_version == MEMORY_KEY_SCHEMA_VERSION
    assert from_raw["context_key_source"] == "canonical_context_frame"
    assert from_latent["context_key_source"] == "canonical_context_frame"
    assert from_raw["context_frame_id"] == from_latent["context_frame_id"] == "canonical:one"
    assert np.allclose(from_raw["canonical_memory_key"], from_latent["canonical_memory_key"])
    assert np.allclose(from_raw["action_bias"], from_latent["action_bias"])


def test_goal_and_portfolio_change_graph_selection_before_post_gate() -> None:
    observation = np.linspace(-0.3, 0.4, 12)

    no_goal = M1GraphModule.seed(input_size=12, output_size=20, state_size=6, K=4, seed=53)
    with_goal = M1GraphModule.seed(input_size=12, output_size=20, state_size=6, K=4, seed=53)
    no_goal.forward(observation, goal=np.zeros(5), context={"portfolio_context": np.zeros(8)})
    with_goal.forward(observation, goal=np.ones(5), context={"portfolio_context": np.zeros(8)})
    assert not np.allclose(no_goal.last_candidate_scores, with_goal.last_candidate_scores)

    calm = M1GraphModule.seed(input_size=12, output_size=20, state_size=6, K=4, seed=54)
    stressed = M1GraphModule.seed(input_size=12, output_size=20, state_size=6, K=4, seed=54)
    calm.forward(observation, context={"portfolio_context": np.zeros(8)})
    stressed.forward(observation, context={"portfolio_context": np.ones(8)})
    assert not np.allclose(calm.last_candidate_scores, stressed.last_candidate_scores)


def test_legality_mask_is_visible_before_policy_returns_action() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=5, seed=55)
    policy.set_decision_context(
        allowed_actions={0: True, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False,
                         7: False, 8: False, 9: False, 10: False, 11: False},
        frame_id="legal:hold-only",
        sequence_id=1,
    )
    trace = policy.act(np.linspace(-0.1, 0.2, INPUT_COUNT))

    assert int(np.argmax(trace.selected_action[:12])) == 0
    assert trace.context_audit["allowed_actions"] == [0]


def test_learning_drive_is_audited_without_becoming_market_entry_evidence() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=5, seed=56)
    policy.set_experience_starvation({
        "no_trade_streak": 48.0,
        "unfilled_streak": 10.0,
        "cadence_deficit": 1.0,
        "runtime_starvation": 3.0,
    })
    policy.set_decision_context(frame_id="drive:not-edge", sequence_id=1)
    trace = policy.act(np.zeros(INPUT_COUNT))

    assert np.max(trace.learning_drive_context) > 0.0
    assert trace.context_audit["learning_drive_norm"] > 0.0
    assert float(trace.action_details.get("market_opportunity", 0.0)) == 0.0


def test_only_current_schema_canonical_cells_can_bias_context_retrieval() -> None:
    from self_growing_trader.self_growing_runner import NeuralMemoryCell

    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=5, seed=57)
    policy.set_decision_context(frame_id="v10:retrieve", sequence_id=1)
    trace = policy.act(np.linspace(-0.15, 0.15, INPUT_COUNT))
    key = trace.canonical_context_key
    action = np.zeros(20, dtype=float)
    action[2] = 1.0
    policy.memory_supernet.neural_cells["v8-success"] = NeuralMemoryCell(
        cell_id="v8-success",
        role="success_action_imprint",
        vector=key.copy(),
        activation=1.0,
        utility=1.5,
        evidence_for=8.0,
        evidence_against=0.0,
        uncertainty=0.01,
        meta={
            "context_schema_version": MEMORY_KEY_SCHEMA_VERSION,
            "memory_key_schema_version": MEMORY_KEY_SCHEMA_VERSION,
            "action_vector": action.tolist(),
            "context_delta": np.zeros_like(key).tolist(),
            "short_context": key.tolist(),
            "regime_context": key.tolist(),
        },
    )

    memory = policy._query_memory(np.ones(3))

    assert memory["success_imprint_matches"]
    assert memory["action_bias"][2] > 0.0
