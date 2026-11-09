from __future__ import annotations

import numpy as np

from self_growing_trader.config import AppConfig, load_config
from self_growing_trader.live_observatory import LiveEventReader, LiveObservabilityConfig, LiveTelemetryWriter
from self_growing_trader.self_growing_runner import SelfGrowingPolicy, SpatialWorldPolicy


def _decision_context(policy: SelfGrowingPolicy, frame_id: str) -> None:
    policy.set_decision_context(
        account_context={"current_equity_drawdown": 0.01, "open_risk_ratio": 0.0},
        positions_pipe="",
        allowed_actions={0: True, 1: True, 2: True},
        execution_context={"spread_norm": 0.1, "margin_available_ratio": 1.0},
        goal=np.asarray([0.1, 0.2, 0.0, 0.0], dtype=float),
        sequence_id=1,
        frame_id=frame_id,
        timestamp=1_700_000_000.0,
    )


def test_live_observatory_is_read_only_for_decision(tmp_path) -> None:
    base = SelfGrowingPolicy.seed(seed=17)
    observed = base.copy()
    event_path = tmp_path / "neural_live.jsonl"
    observed.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="test-read-only",
    )

    _decision_context(base, "base:1")
    _decision_context(observed, "observed:1")
    observation = np.linspace(-0.5, 0.5, base.input_size)
    baseline_trace = base.act(observation)
    observed_trace = observed.act(observation)

    np.testing.assert_allclose(baseline_trace.selected_action, observed_trace.selected_action, atol=1e-12)
    np.testing.assert_allclose(baseline_trace.candidate_scores, observed_trace.candidate_scores, atol=1e-12)

    state = LiveEventReader(event_path).state()
    types = [item["event_type"] for item in state["events"]]
    assert "run_started" in types
    assert "topology" in types
    assert "decision" in types
    assert state["topology"]["node_count"] > 0
    assert state["latest"]["decision"]["payload"]["trace_id"] == observed_trace.trace_id


def test_live_observatory_records_feedback_and_trial_without_structural_development(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(seed=23)
    event_path = tmp_path / "neural_live.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="test-feedback",
    )
    _decision_context(policy, "test:1")
    trace = policy.act(np.zeros(policy.input_size, dtype=float))
    policy.observe_transition(trace, np.zeros(policy.input_size), reward=0.2, cost=0.01, done=False)
    policy.live_observatory_trial_finished(policy_id=3, round_index=2, fitness=1.5, metrics={"total_trades": 4})

    state = LiveEventReader(event_path).state()
    types = [item["event_type"] for item in state["events"]]
    assert "feedback" in types
    assert "development" not in types
    assert "trial_finished" in types
    feedback = state["latest"]["feedback"]["payload"]
    assert feedback["trace_id"] == trace.trace_id
    assert "activity" in feedback
    assert state["latest"]["decision"]["payload"]["topology_summary"]["node_count"] == state["topology"]["node_count"]
    assert feedback["topology_summary"]["edge_count"] == state["topology"]["edge_count"]
    assert state["latest"]["trial_finished"]["payload"]["topology_summary"]["edge_count"] == state["topology"]["edge_count"]


def test_live_observatory_rehydrates_enabled_writer_from_config(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(seed=24)
    event_path = tmp_path / "neural_live_rehydrated.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="test-rehydrate",
    )
    policy._live_observer = None
    policy.ensure_compatible()
    _decision_context(policy, "test:rehydrate")
    trace = policy.act(np.zeros(policy.input_size, dtype=float))

    state = LiveEventReader(event_path).state()
    types = [item["event_type"] for item in state["events"]]
    assert "decision" in types
    assert state["latest"]["decision"]["payload"]["trace_id"] == trace.trace_id


def test_live_observatory_survives_candidate_copy_and_v9_final_decision(tmp_path) -> None:
    base = SelfGrowingPolicy.seed(seed=25)
    event_path = tmp_path / "neural_live_candidate.jsonl"
    base.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="test-candidate-live",
        source="mt5",
    )
    candidate = base.propose(np.random.default_rng(5), mutation_sigma=0.02, max_modules=3)
    candidate.set_live_observatory_trial(policy_id=1, round_index=0, phase="mt5")
    _decision_context(candidate, "test:candidate-copy")
    trace = candidate.act(np.zeros(candidate.input_size, dtype=float))

    state = LiveEventReader(event_path).state()
    types = [item["event_type"] for item in state["events"]]
    assert "trial_started" in types
    assert "decision" in types
    assert state["latest"]["decision"]["payload"]["trace_id"] == trace.trace_id
    assert state["writer_status"]["last_event_type"] == "decision"


def test_live_observatory_survives_world_policy_conversion_and_propose(tmp_path) -> None:
    base = SpatialWorldPolicy.seed(seed=26)
    event_path = tmp_path / "neural_live_world_candidate.jsonl"
    base.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="test-world-candidate-live",
        source="mt5",
    )
    candidate = base.propose(np.random.default_rng(6), mutation_sigma=0.02, max_modules=3)
    candidate.set_live_observatory_trial(policy_id=1, round_index=0, phase="mt5")
    _decision_context(candidate, "test:world-candidate")
    trace = candidate.act(np.zeros(candidate.input_size, dtype=float))

    state = LiveEventReader(event_path).state()
    types = [item["event_type"] for item in state["events"]]
    assert "trial_started" in types
    assert "decision" in types
    assert state["latest"]["decision"]["payload"]["trace_id"] == trace.trace_id
    assert state["writer_status"]["last_event_type"] == "decision"


def test_live_decision_reports_policy_legal_and_emitted_actions(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(seed=24)
    event_path = tmp_path / "neural_live.jsonl"
    writer = LiveTelemetryWriter(path=str(event_path), enabled=True, run_id="test-actions", source="mt5")
    writer.start(policy)
    _decision_context(policy, "test:actions")
    trace = policy.act(np.zeros(policy.input_size, dtype=float))
    trace.action_details.update({
        "action_name": "HOLD",
        # V9 contract chỉ hiển thị semantic/emitted/wire; legacy policy/legal
        # không tham gia UI/log quyết định.
        "semantic_action": 2,
        "emitted_action": 2,
        "wire_action": 2,
        "wire_matches_semantic_action": True,
        "execution_bootstrap": True,
        "five_reasoning_rl_enabled": True,
        "five_reasoning_action_scale": 0.0,
        "five_reasoning_trace": {
            "schema_version": 1,
            "steps": [1, 3, 2],
            "final_hypothesis": 2,
            "action_scale": 0.0,
            "action_residual_clip": 0.25,
            "training_labels_visible": False,
        },
    })

    writer.decision(policy, trace)

    state = LiveEventReader(event_path).state()
    action = state["latest"]["decision"]["payload"]["action"]
    # V9 không phát action policy/legal legacy. UI chỉ được xem semantic/wire.
    assert action["name"] == "MARKET_SELL"
    assert "policy_name" not in action
    assert "legal_name" not in action
    assert action["semantic_name"] == "MARKET_SELL"
    assert action["wire_name"] == "MARKET_SELL"
    assert action["semantic_action"] == 2
    assert action["emitted_action"] == 2
    assert action["wire_action"] == 2
    assert action["action_contract_complete"] is True
    assert action["execution_bootstrap"] is True
    reasoning = state["latest"]["decision"]["payload"]["five_reasoning"]
    assert reasoning == {
        "enabled": True,
        "schema_version": 1,
        "steps": [1, 3, 2],
        "final_hypothesis": 2,
        "action_scale": 0.0,
        "action_residual_clip": 0.25,
        "training_labels_visible": False,
    }


def test_config_parses_observability_section(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
observability:
  enabled: true
  event_path: outputs/example_live.jsonl
  decision_profile_path: outputs/ho_so_quyet_dinh.jsonl
  structural_event_path: outputs/nhat_ky_cau_truc.jsonl
  network_development_path: outputs/theo_doi_phat_trien_mang.jsonl
  decision_sample_every: 3
  network_development_feedback_sample_every: 5
  max_event_bytes: 2097152
  max_active_nodes: 64
  max_active_edges: 96
  max_state_nodes: 128
  max_state_edges: 256
  max_responsible_neurons: 21
  max_responsible_edges: 34
  state_change_epsilon: 0.0001
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert isinstance(config, AppConfig)
    assert config.observability.enabled is True
    assert config.observability.event_path.endswith("example_live.jsonl")
    assert config.observability.decision_sample_every == 3
    assert config.observability.network_development_feedback_sample_every == 5
    assert config.observability.max_state_nodes == 128
    assert config.observability.max_state_edges == 256
    assert config.observability.decision_profile_path.endswith("ho_so_quyet_dinh.jsonl")
    assert config.observability.structural_event_path.endswith("nhat_ky_cau_truc.jsonl")
    assert config.observability.network_development_path.endswith("theo_doi_phat_trien_mang.jsonl")
    assert config.observability.max_responsible_neurons == 21
    assert config.observability.max_responsible_edges == 34
    assert config.observability.state_change_epsilon == 0.0001


def test_feedback_development_snapshot_is_sampled_but_feedback_events_remain_complete(tmp_path) -> None:
    import json

    policy = SelfGrowingPolicy.seed(seed=49)
    event_path = tmp_path / "events.jsonl"
    network_path = tmp_path / "network.jsonl"
    writer = LiveTelemetryWriter(
        path=str(event_path),
        network_development_path=str(network_path),
        enabled=True,
        network_development_feedback_sample_every=2,
        run_id="sampled-feedback-network",
    )
    writer.start(policy)
    trace = policy.act(np.zeros(policy.input_size, dtype=float))
    writer.feedback(policy, trace, next_obs=np.zeros(policy.input_size), reward=0.0, cost=0.0, done=False, signal={})
    writer.feedback(policy, trace, next_obs=np.zeros(policy.input_size), reward=0.0, cost=0.0, done=False, signal={})

    event_rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    network_rows = [json.loads(line) for line in network_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert sum(1 for row in event_rows if row.get("event_type") == "feedback") == 2
    assert [row["giai_doan"] for row in network_rows] == ["khoi_tao", "feedback"]


def test_active_edges_default_to_full_trace_for_live_ui() -> None:
    config = LiveObservabilityConfig(enabled=True)
    writer = LiveTelemetryWriter.from_config(config, run_id="test-full-active-edges")

    assert config.max_active_edges == 0
    assert writer.max_active_edges == 0


def test_auxiliary_observability_logs_respect_event_byte_quota(tmp_path) -> None:
    import json

    aux_path = tmp_path / "network.jsonl"
    writer = LiveTelemetryWriter(
        enabled=True,
        path=str(tmp_path / "neural_live.jsonl"),
        network_development_path=str(aux_path),
        max_event_bytes=512,
        run_id="test-aux-quota",
    )

    for index in range(40):
        assert writer._append_aux_event(aux_path, {"index": index, "payload": "x" * 24})

    assert aux_path.stat().st_size <= 512
    rows = [json.loads(line) for line in aux_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert rows[-1]["index"] == 39


def test_execution_lifecycle_runs_micro_growth_before_live_lifecycle_event(tmp_path) -> None:
    """A settled MT5-style trace must grow in the candidate, then emit it live."""
    policy = SelfGrowingPolicy.seed(seed=31, m1_backend="m1_graph")
    policy.graph_trial_mode = True
    policy.graph_trial_id = "test-execution-lifecycle-growth"
    assert policy.graph_supernetwork is not None
    policy.graph_supernetwork.begin_trial(policy.graph_trial_id)
    event_path = tmp_path / "neural_live.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="test-execution-lifecycle-growth",
    )
    _decision_context(policy, "test:lifecycle")
    trace = policy.act(np.zeros(policy.input_size, dtype=float))
    # MT5 stores a filled-entry trace and later settles it from the execution
    # ledger.  Simulate exactly that hand-off rather than using a synthetic
    # observe_transition path.
    policy._cache_execution_trace(trace)
    policy.apply_execution_lifecycle_credit(
        trace.trace_id,
        -1.0,
        close_reason="SL",
        duration_bars=5,
        min_holding_bars=3,
    )

    state = LiveEventReader(event_path).state()
    lifecycle = state["latest"]["lifecycle"]["payload"]
    growth_types = {str(item.get("type", "")) for item in lifecycle["growth_events"]}
    assert "specialist_module_candidate_created" in growth_types
    assert "graph_lifecycle_micro_growth_evaluated" in growth_types
    synapse = lifecycle["synapse_state"]
    assert synapse["node_change_count"] > 0
    assert synapse["edge_change_count"] > 0
    assert state["topology"]["node_count"] > 0


def test_topology_event_keeps_full_seed_graph_instead_of_generic_256_item_cap(tmp_path) -> None:
    """The 3D viewer must receive regions after the dense input fan-in.

    A seeded policy has more than 256 nodes and edges.  The generic JSON-safe
    list limit used to truncate the initial topology event at 256, leaving the
    UI with only input->encoder wiring.
    """
    from self_growing_trader.live_observatory import graph_topology

    policy = SelfGrowingPolicy.seed(seed=47, m1_backend="m1_graph")
    canonical = graph_topology(policy)
    assert canonical["node_count"] > 256
    assert canonical["edge_count"] > 256

    event_path = tmp_path / "full_topology.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="test-full-topology",
    )

    state = LiveEventReader(event_path).state()
    topology = state["topology"]
    assert len(topology["nodes"]) == canonical["node_count"]
    assert len(topology["edges"]) == canonical["edge_count"]
    regions = {str(node.get("region", "")) for node in topology["nodes"]}
    assert {"input", "encoder", "memory", "predictor", "critic", "selector", "output"}.issubset(regions)


def test_network_development_stream_keeps_full_topology_activity_and_state(tmp_path) -> None:
    """V9 writes a separate unbounded network-development audit stream."""
    import json

    policy = SelfGrowingPolicy.seed(seed=48)
    event_path = tmp_path / "neural_live.jsonl"
    network_path = tmp_path / "theo_doi_phat_trien_mang.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(
            enabled=True,
            event_path=str(event_path),
            network_development_path=str(network_path),
            max_event_bytes=0,
            max_active_nodes=0,
            max_active_edges=0,
            max_state_nodes=0,
            max_state_edges=0,
            reset_on_process_start=True,
        ),
        run_id="test-network-development-v9",
    )
    _decision_context(policy, "test:network-development")
    policy.act(np.zeros(policy.input_size, dtype=float))

    rows = [json.loads(line) for line in network_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    row = rows[-1]
    assert row["loai_ban_ghi"] == "THEO_DOI_PHAT_TRIEN_MANG"
    topology = row["topology_day_du"]
    assert len(topology["nodes"]) == topology["node_count"]
    assert len(topology["edges"]) == topology["edge_count"]
    assert row["hoat_dong_day_du"]["active_node_count"] > 0
    assert row["hoat_dong_day_du"]["active_edge_count"] > 0
