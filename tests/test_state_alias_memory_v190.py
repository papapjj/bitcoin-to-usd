from __future__ import annotations

import pickle

import numpy as np
import pytest

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy, ModuleState
from self_growing_trader.config import LearningIntegrityConfig
from self_growing_trader.state_alias_memory import (
    STATE_ALIAS_MEMORY_ARTIFACT_VERSION,
    STATE_ALIAS_MEMORY_PARAMETER_PREFIX,
    StateAliasingMemoryState,
)


def _memory(seed: int = 7) -> StateAliasingMemoryState:
    memory = StateAliasingMemoryState.seed(
        input_size=5,
        hidden_size=4,
        memory_size=4,
        capacity=4,
        min_history=2,
        temperature=0.8,
        injection_scale=0.75,
        enabled=True,
        seed=seed,
    )
    memory.gate_b[:] = 0.25
    memory.remember(np.asarray([-0.8, 0.2, 0.5, -0.1, 0.3]))
    memory.remember(np.asarray([0.7, -0.4, 0.1, 0.8, -0.2]))
    return memory


def _module(name: str = "memory_module") -> ModuleState:
    return ModuleState.create(name, 5, 4, 3, np.random.default_rng(17))


def _memory_config(*, enabled: bool = True) -> dict[str, object]:
    return {
        "direct_rnn_adamw_enabled": True,
        "direct_rnn_full_rl_enabled": True,
        "direct_rnn_state_memory_enabled": enabled,
        "direct_rnn_state_memory_capacity": 4,
        "direct_rnn_state_memory_size": 4,
        "direct_rnn_state_memory_min_history": 2,
        "direct_rnn_state_memory_temperature": 0.8,
        "direct_rnn_state_memory_injection_scale": 0.75,
    }


def test_config_is_opt_in_and_validates_memory_geometry() -> None:
    config = LearningIntegrityConfig()
    assert config.direct_rnn_state_memory_enabled is False
    assert config.direct_rnn_state_memory_capacity == 32
    with pytest.raises(ValueError, match="min_history"):
        LearningIntegrityConfig(
            direct_rnn_state_memory_capacity=8,
            direct_rnn_state_memory_min_history=9,
        )


def test_old_policy_remains_memory_free_and_behavior_identical() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=5, output_size=3, hidden_size=4, seed=3)
    module = policy.modules[0]
    clone = module.copy()
    module.configure_rnn_adamw_cosine(_memory_config(enabled=False))
    clone.configure_rnn_adamw_cosine(_memory_config(enabled=False))
    assert module.state_alias_memory is None
    observation = np.linspace(-0.2, 0.2, 5)
    left = module.activate(observation, np.zeros(4))
    right = clone.activate(observation, np.zeros(4))
    for expected, observed in zip(left, right):
        np.testing.assert_array_equal(expected, observed)


def test_same_current_observation_is_disambiguated_by_ordered_history() -> None:
    first = _module("shared_name")
    second = first.copy()
    first.configure_rnn_adamw_cosine(_memory_config())
    second.configure_rnn_adamw_cosine(_memory_config())
    assert isinstance(first.state_alias_memory, StateAliasingMemoryState)
    assert isinstance(second.state_alias_memory, StateAliasingMemoryState)
    history = (
        np.asarray([-1.0, 0.0, 0.4, 0.0, 0.2]),
        np.asarray([1.0, 0.0, -0.4, 0.0, -0.2]),
    )
    for item in history:
        first.state_alias_memory.remember(item)
    for item in reversed(history):
        second.state_alias_memory.remember(item)
    current = np.asarray([0.25, -0.1, 0.2, 0.0, 0.5])
    first_output = first.activate(current, np.zeros(4))[2]
    second_output = second.activate(current, np.zeros(4))[2]
    first_trace = first.last_forward_trace.state_alias_memory_forward
    second_trace = second.last_forward_trace.state_alias_memory_forward
    assert first_trace is not None and first_trace.ready
    assert second_trace is not None and second_trace.ready
    assert first_trace.history_fingerprint != second_trace.history_fingerprint
    assert np.linalg.norm(first_trace.injected - second_trace.injected) > 1e-10
    assert np.linalg.norm(first_output - second_output) > 1e-10


def test_realistic_memory_seed_is_path_selective_and_has_material_injection() -> None:
    memory = StateAliasingMemoryState.seed(
        input_size=137,
        hidden_size=64,
        memory_size=32,
        capacity=32,
        min_history=4,
        temperature=0.35,
        injection_scale=0.50,
        path_delta_scale=0.75,
        enabled=True,
        seed=70,
    )
    rng = np.random.default_rng(71)
    current = np.zeros(137, dtype=float)
    for _ in range(32):
        current = 0.85 * current + rng.normal(0.0, 0.30, size=137)
        memory.remember(current)
    result = memory.read(
        0.85 * current + rng.normal(0.0, 0.30, size=137), count=False
    )
    payload = result.payload()

    assert payload["attention_entropy_normalized"] < 0.99
    assert payload["top_attention_weight"] > 1.25 / 32.0
    assert payload["injection_norm"] > 0.01
    assert payload["path_delta_norm"] > 0.0


def test_read_is_causal_and_reset_removes_cross_run_history() -> None:
    memory = _memory()
    current = np.linspace(-0.3, 0.3, 5)
    before = memory.read(current, count=False)
    memory.remember(np.full(5, 99.0))
    # A previously materialized decision trace is immutable.
    np.testing.assert_array_equal(before.tokens, np.asarray([
        [-0.8, 0.2, 0.5, -0.1, 0.3],
        [0.7, -0.4, 0.1, 0.8, -0.2],
    ]))
    assert before.history_length == 2
    after = memory.read(current, count=False)
    assert after.history_length == 3
    memory.reset_runtime()
    cold = memory.read(current, count=False)
    assert cold.ready is False
    assert cold.history_length == 0
    np.testing.assert_array_equal(cold.injected, np.zeros(4))


@pytest.mark.parametrize(
    ("name", "index"),
    (
        ("query_w", (0, 0)),
        ("key_w", (1, 2)),
        ("value_w", (2, 1)),
        ("output_w", (3, 2)),
        ("relative_position", (2, 1)),
        ("recency_bias", (2,)),
        ("gate_w", (3,)),
    ),
)
def test_memory_backward_matches_finite_difference(name: str, index: tuple[int, ...]) -> None:
    memory = _memory()
    current = np.asarray([0.1, -0.3, 0.2, 0.6, -0.5])
    upstream = np.asarray([0.4, -0.2, 0.3, 0.1])
    forward = memory.read(current, count=False)
    gradients = memory.backward(forward, upstream)
    analytical = float(gradients[f"{STATE_ALIAS_MEMORY_PARAMETER_PREFIX}{name}"][index])
    parameter = np.asarray(getattr(memory, name), dtype=float).copy()
    epsilon = 1e-5

    plus_parameter = parameter.copy()
    plus_parameter[index] += epsilon
    setattr(memory, name, plus_parameter)
    plus = float(memory.read(current, count=False).injected @ upstream)

    minus_parameter = parameter.copy()
    minus_parameter[index] -= epsilon
    setattr(memory, name, minus_parameter)
    minus = float(memory.read(current, count=False).injected @ upstream)
    setattr(memory, name, parameter)

    numerical = (plus - minus) / (2.0 * epsilon)
    assert analytical == pytest.approx(numerical, rel=2e-3, abs=2e-6)


def test_memory_parameters_use_the_full_rl_adamw_sole_writer() -> None:
    module = _module()
    module.configure_rnn_adamw_cosine(_memory_config())
    memory = module.state_alias_memory
    assert isinstance(memory, StateAliasingMemoryState)
    memory.gate_b[:] = 0.25
    memory.remember(np.linspace(-0.4, 0.4, 5))
    memory.remember(np.linspace(0.5, -0.5, 5))
    module.activate(np.linspace(-0.2, 0.3, 5), np.zeros(4))
    forward = module.last_forward_trace.state_alias_memory_forward
    directions = memory.backward(forward, np.ones(4))
    assert directions
    before = memory.output_w.copy()
    rejected = module.apply_rnn_adamw_cosine(directions, source="legacy_credit")
    assert rejected["updated"] == 0.0
    np.testing.assert_array_equal(before, memory.output_w)
    accepted = module.apply_rnn_adamw_cosine(directions, source="full_lifecycle_rl")
    assert accepted["updated"] == 1.0
    assert not np.array_equal(before, memory.output_w)
    assert "state_alias_temporal_memory" in accepted["direct_rnn_blocks"]["blocks"]


def test_per_expert_memory_is_independent_copyable_and_pickle_stable() -> None:
    settings = LearningIntegrityConfig(
        direct_rnn_adamw_enabled=True,
        direct_rnn_full_rl_enabled=True,
        direct_rnn_state_memory_enabled=True,
        direct_rnn_state_memory_capacity=4,
        direct_rnn_state_memory_size=4,
        direct_rnn_state_memory_min_history=2,
    )
    left = ChuyenGiaPolicy.seed(input_size=5, output_size=3, hidden_size=4, seed=5)
    right = ChuyenGiaPolicy.seed(input_size=5, output_size=3, hidden_size=4, seed=5)
    left.configure_learning_integrity(settings)
    right.configure_learning_integrity(settings)
    left_memory = left.modules[0].state_alias_memory
    right_memory = right.modules[0].state_alias_memory
    assert isinstance(left_memory, StateAliasingMemoryState)
    assert isinstance(right_memory, StateAliasingMemoryState)
    assert left_memory is not right_memory
    left_memory.query_w[0, 0] += 1.0
    assert left_memory.query_w[0, 0] != right_memory.query_w[0, 0]
    assert left.artifact_version >= STATE_ALIAS_MEMORY_ARTIFACT_VERSION

    restored = pickle.loads(pickle.dumps(left, protocol=pickle.HIGHEST_PROTOCOL))
    restored.ensure_compatible()
    restored_memory = restored.modules[0].state_alias_memory
    assert isinstance(restored_memory, StateAliasingMemoryState)
    np.testing.assert_array_equal(left_memory.query_w, restored_memory.query_w)
    assert restored.artifact_version >= STATE_ALIAS_MEMORY_ARTIFACT_VERSION
