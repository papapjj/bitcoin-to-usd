import pickle

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy, ModuleState
from self_growing_trader.config import INPUT_COUNT, OUTPUT_COUNT, ModelConfig, SupernetworkConfig
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_chuyen_gia_policy_supports_deep_hidden_stack() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=8,
        output_size=5,
        hidden_size=12,
        hidden_layers=4,
        seed=123,
    )

    module = policy.modules[0]
    assert module.hidden_size == 12
    assert len(module.deep_w) == 3
    assert all(layer.shape == (12, 12) for layer in module.deep_w)

    gate, hidden, output = module.activate(np.linspace(-1.0, 1.0, 8))

    assert 0.0 <= gate <= 1.0
    assert hidden.shape == (12,)
    assert output.shape == (5,)
    assert len(module.deep_state) == 3


def test_self_growing_policy_supports_deep_hidden_stack() -> None:
    policy = SelfGrowingPolicy.seed(
        input_size=8,
        output_size=5,
        hidden_size=10,
        hidden_layers=3,
        seed=124,
    )

    module = policy.modules[0]
    assert len(module.deep_w) == 2
    assert all(layer.shape == (10, 10) for layer in module.deep_w)
    assert module.activate(np.ones(8))[2].shape == (5,)


def test_self_growing_policy_can_grow_hidden_depth_under_representation_need() -> None:
    policy = SelfGrowingPolicy.seed(
        input_size=8,
        output_size=5,
        hidden_size=10,
        hidden_layers=2,
        seed=126,
    )
    policy.graph_supernetwork_config.update({
        "max_hidden_layers_per_module": 4,
        "hidden_depth_growth_threshold": 0.0,
    })

    summary = policy._maybe_expand_hidden_depth(rng=np.random.default_rng(127))
    module = policy.modules[0]

    assert summary["expanded"] == 1
    assert 1 + len(module.deep_w) == 3
    assert module.deep_w[-1].shape == (10, 10)
    assert any(event.get("type") == "expand_module_depth" for event in policy.growth_events)


def test_self_growing_policy_can_grow_hidden_width_under_representation_need() -> None:
    policy = SelfGrowingPolicy.seed(
        input_size=8,
        output_size=5,
        hidden_size=16,
        hidden_layers=3,
        seed=130,
    )
    policy.graph_supernetwork_config.update({
        "max_hidden_neurons_per_module": 24,
        "hidden_depth_growth_threshold": 0.0,
    })

    summary = policy._maybe_expand_hidden_width(rng=np.random.default_rng(131))
    module = policy.modules[0]

    assert summary["expanded"] == 1
    assert module.hidden_size > 16
    assert module.hidden_size <= 24
    assert module.w_in.shape == (module.hidden_size, 8)
    assert module.w_out.shape == (5, module.hidden_size)
    assert all(layer.shape == (module.hidden_size, module.hidden_size) for layer in module.deep_w)
    assert any(event.get("type") == "expand_module" for event in policy.growth_events)


def test_hidden_depth_growth_respects_layer_cap() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=8,
        output_size=5,
        hidden_size=10,
        hidden_layers=3,
        seed=128,
    )
    policy.graph_supernetwork_config.update({"max_hidden_layers_per_module": 3})

    changed = policy.expand_module_depth("m1_thinking", 1, rng=np.random.default_rng(129))

    assert changed is False
    assert 1 + len(policy.modules[0].deep_w) == 3


def test_legacy_single_layer_module_pickles_still_activate() -> None:
    rng = np.random.default_rng(125)
    module = ModuleState.create("legacy", 6, 7, 4, rng)
    legacy_payload = pickle.loads(pickle.dumps(module))
    del legacy_payload.deep_w
    del legacy_payload.deep_rec
    del legacy_payload.deep_b
    del legacy_payload.deep_state

    gate, hidden, output = legacy_payload.activate(np.ones(6))
    copied = legacy_payload.copy()

    assert 0.0 <= gate <= 1.0
    assert hidden.shape == (7,)
    assert output.shape == (4,)
    assert copied.deep_w == ()


def test_model_config_defaults_to_deeper_capacity() -> None:
    config = ModelConfig(input_count=INPUT_COUNT, output_count=OUTPUT_COUNT)
    supernetwork = SupernetworkConfig()

    assert config.hidden_size == 128
    assert config.hidden_layers == 6
    assert supernetwork.max_hidden_neurons_per_module == 512
    assert supernetwork.max_hidden_layers_per_module == 24
