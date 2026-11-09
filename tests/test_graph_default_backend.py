from __future__ import annotations

from self_growing_trader.config import SupernetworkConfig
from self_growing_trader.m1_graph_module import M1GraphModule
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_fresh_policy_defaults_to_direct_rnn() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=5, seed=101)
    assert policy.m1_backend == "direct_rnn"
    assert policy.modules[0].m1_component is None
    assert policy.artifact_version >= 9


def test_unknown_backend_is_rejected() -> None:
    try:
        SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=102, m1_backend="legacy")
    except ValueError as exc:
        assert "direct_rnn" in str(exc)
    else:
        raise AssertionError("unknown decision backends must be rejected")


def test_config_defaults_to_direct_rnn_without_fixed_topology_caps() -> None:
    config = SupernetworkConfig()
    assert config.m1_backend == "direct_rnn"
    # Numerical ceilings are deliberately disabled (0). Structural growth is
    # instead gated by trace-owned lifecycle evidence, replay validation and
    # non-interference checks.
    assert config.max_child_networks == 0
    assert config.max_generation == 0
    assert config.max_cross_module_edges == 0
    assert config.max_m1_module_edges == 0
    assert config.recurrent_loop_limit == 0
    assert config.max_graph_neurons == 0
    assert config.max_graph_connections == 0


def test_direct_compatible_upgrade_does_not_reinitialize_non_graph_component() -> None:
    policy = SelfGrowingPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=5, seed=103)
    policy.m1_backend = "retired"
    policy.modules[0].m1_component = object()
    policy.graph_supernetwork = None
    policy.ensure_compatible()
    assert policy.m1_backend == "direct_rnn"
    assert not isinstance(policy.modules[0].m1_component, M1GraphModule)
    assert policy.graph_supernetwork is None
