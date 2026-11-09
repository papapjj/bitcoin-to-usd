from __future__ import annotations

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy, ModuleState


def test_every_hidden_block_has_a_neutral_attention_layer() -> None:
    module = ModuleState.create(
        "attention_stack", 9, 7, 5, np.random.default_rng(801), hidden_layers=4
    )
    observation = np.linspace(-1.0, 1.0, module.input_size)

    attended, weights = module._feature_attention(
        observation, module.input_attention_q, module.input_attention_b, enabled=True
    )
    module.activate(observation)

    assert module.attention_enabled is True
    assert len(module.deep_attention_q) == len(module.deep_w) == 3
    assert len(module.deep_attention_b) == len(module.deep_w) == 3
    assert np.array_equal(weights, np.ones(module.input_size))
    assert np.array_equal(attended, observation)
    assert len(module.last_forward_trace.deep_attention_weights) == 3


def test_feature_attention_vjp_matches_finite_difference() -> None:
    values = np.array([-0.8, -0.1, 0.4, 1.2], dtype=float)
    query = np.array([0.3, -0.2, 0.1, 0.4], dtype=float)
    bias = np.array([0.05, -0.15, 0.2, -0.1], dtype=float)
    upstream = np.array([0.7, -0.5, 0.2, 0.9], dtype=float)
    _, weights = ModuleState._feature_attention(values, query, bias, enabled=True)
    input_grad, query_grad, bias_grad = ChuyenGiaPolicy._feature_attention_vjp(
        values, weights, query, upstream
    )

    def objective(values: np.ndarray, query: np.ndarray, bias: np.ndarray) -> float:
        attended, _ = ModuleState._feature_attention(values, query, bias, enabled=True)
        return float(attended @ upstream)

    epsilon = 1e-6
    for actual, source, slot in (
        (input_grad, values, "values"),
        (query_grad, query, "query"),
        (bias_grad, bias, "bias"),
    ):
        numerical = np.zeros_like(source)
        for index in range(source.size):
            plus = source.copy()
            minus = source.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            args_plus = {"values": values, "query": query, "bias": bias}
            args_minus = dict(args_plus)
            args_plus[slot] = plus
            args_minus[slot] = minus
            numerical[index] = (
                objective(**args_plus) - objective(**args_minus)
            ) / (2.0 * epsilon)
        assert np.allclose(actual, numerical, atol=1e-6, rtol=1e-5)


def test_depth_growth_always_adds_matching_attention() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=8, output_size=5, hidden_size=10, hidden_layers=2, seed=802
    )
    policy.graph_supernetwork_config["max_hidden_layers_per_module"] = 5

    assert policy.expand_module_depth(
        "m1_thinking", 2, rng=np.random.default_rng(803)
    ) is True
    module = policy.modules[0]

    assert 1 + len(module.deep_w) == 4
    assert 1 + len(module.deep_attention_q) == 4
    assert 1 + len(module.deep_attention_b) == 4
    assert all(np.array_equal(item, np.zeros(module.hidden_size)) for item in module.deep_attention_q[-2:])


def test_full_rl_optimizer_owns_every_attention_parameter() -> None:
    module = ModuleState.create(
        "attention_optimizer", 11, 6, 5, np.random.default_rng(804), hidden_layers=3
    )
    parameters = module._rnn_optimizer_parameters()

    assert parameters["input_attention_q"].shape == (11,)
    assert parameters["input_attention_b"].shape == (11,)
    for index in range(2):
        assert parameters[f"deep_attention_q_{index}"].shape == (6,)
        assert parameters[f"deep_attention_b_{index}"].shape == (6,)


def test_input_schema_growth_resizes_base_attention_neutrally() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=8, output_size=5, hidden_size=6, hidden_layers=3, seed=805
    )

    policy.ensure_compatible(target_input_size=13)
    module = policy.modules[0]

    assert module.input_size == 13
    assert module.input_attention_q.shape == (13,)
    assert module.input_attention_b.shape == (13,)
    assert np.array_equal(module.input_attention_q[8:], np.zeros(5))
    assert np.array_equal(module.input_attention_b[8:], np.zeros(5))
