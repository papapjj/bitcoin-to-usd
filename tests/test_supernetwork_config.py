from __future__ import annotations

from pathlib import Path

import pytest

from self_growing_trader.config import EnvironmentConfig, GRAPH_CONNECTABLE_REGIONS, SupernetworkConfig, load_config
from self_growing_trader.cross_module import CrossModuleConnector
from self_growing_trader.neural_graph import EDGE_MEMORY
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_supernetwork_config_parses_connectivity_lists(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
supernetwork:
  m1_backend: graph
  cross_module_enabled: false
  allowed_cross_module_edge_kinds: [memory, critic]
  allowed_module_to_m1_targets: [critic]
  allowed_m1_to_module_targets: [memory]
  allowed_specialist_roles: [memory_specialist]
  flexible_cross_module_connectivity: false
  flexible_connection_fanout: 4
  flexible_peer_module_fanout: 2
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.supernetwork.m1_backend == "m1_graph"
    assert config.supernetwork.cross_module_enabled is False
    assert config.supernetwork.allowed_cross_module_edge_kinds == ("memory", "critic")
    assert config.supernetwork.allowed_module_to_m1_targets == ("critic",)
    assert config.supernetwork.flexible_cross_module_connectivity is False
    assert config.supernetwork.flexible_connection_fanout == 4
    assert config.supernetwork.flexible_peer_module_fanout == 2


def test_environment_backend_is_optional_and_mt5_remains_default() -> None:
    assert EnvironmentConfig().backend == "mt5"
    with pytest.raises(ValueError, match="model1_tape_path"):
        EnvironmentConfig(backend="python_model1")
    replay = EnvironmentConfig(backend="python_model1", model1_tape_path="oracle.npz")
    assert replay.strict_model1_parity is False


def test_supernetwork_defaults_allow_flexible_learned_region_connectivity_without_fixed_caps() -> None:
    config = SupernetworkConfig()
    assert config.allowed_module_to_m1_targets == GRAPH_CONNECTABLE_REGIONS
    assert config.allowed_m1_to_module_targets == GRAPH_CONNECTABLE_REGIONS
    # V7+ uses zero as “no fixed numerical cap”.  Growth is constrained by
    # trace-owned lifecycle/replay evidence, not a preallocated edge quota.
    assert config.max_cross_module_edges == 0
    assert config.max_m1_module_edges == 0
    assert config.flexible_cross_module_connectivity is True
    assert config.flexible_connection_fanout == 0
    assert config.flexible_peer_module_fanout == 0


def test_graph_policy_applies_connectivity_guardrails() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=301)
    policy.convert_m1_to_graph_backend()
    settings = type(
        "Cfg",
        (),
        {
            "cross_module_enabled": False,
            "allowed_cross_module_edge_kinds": (EDGE_MEMORY,),
            "allowed_specialist_roles": ("memory_specialist",),
            "allowed_module_to_m1_targets": ("critic",),
            "allowed_m1_to_module_targets": ("memory",),
            "flexible_cross_module_connectivity": False,
            "flexible_connection_fanout": 4,
            "flexible_peer_module_fanout": 2,
            "max_cross_module_edges": 7,
            "max_m1_module_edges": 5,
            "max_cross_module_recurrent_edges": 2,
            "max_cross_module_lateral_edges": 1,
            "recurrent_loop_limit": 2,
            "module_role_budget": 1,
            "prediction_need_threshold": 0.3,
            "memory_need_threshold": 0.3,
            "risk_need_threshold": 0.3,
            "position_need_threshold": 0.3,
            "cadence_need_threshold": 0.3,
        },
    )()
    policy.configure_graph_supernetwork(settings)
    graph = policy.modules[0].m1_component.graph
    assert policy.graph_supernetwork.connector.enabled is False
    assert policy.graph_supernetwork.connector.allowed_edge_kinds == frozenset({EDGE_MEMORY})
    assert policy.graph_supernetwork.connector.flexible_connectivity is False
    assert policy.graph_supernetwork.connector.flexible_max_edges == 4
    assert policy.graph_supernetwork.connector.flexible_peer_max_edges == 2
    assert graph.budget.max_cross_module_edges == 0
    assert graph.allowed_module_m1_targets == frozenset({"critic"})
    assert graph.allowed_m1_module_targets == frozenset({"memory"})


def test_cross_module_connector_honours_enabled_and_kind_allowlist() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=302, m1_backend="m1_graph")
    graph = policy.modules[0].m1_component.graph.copy()
    specialist = graph.add_neuron("memory", module_id="specialist:memory_specialist:one", lifecycle="candidate")
    del specialist
    blocked = CrossModuleConnector(enabled=False)
    assert blocked.wire_specialist(
        graph,
        module_id="specialist:memory_specialist:one",
        role="memory_specialist",
        candidate_id="c1",
        need_reason="low_memory_match",
    ) == []
    limited = CrossModuleConnector(allowed_edge_kinds=(EDGE_MEMORY,))
    edge_ids = limited.wire_specialist(
        graph,
        module_id="specialist:memory_specialist:one",
        role="memory_specialist",
        candidate_id="c2",
        need_reason="low_memory_match",
    )
    assert edge_ids
    assert {graph.edges[edge_id].kind for edge_id in edge_ids} == {EDGE_MEMORY}
