from __future__ import annotations

from dataclasses import replace
import pickle

import numpy as np

from self_growing_trader.chuyen_gia_policy import (
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _compact_context_expert_artifact_state,
    tao_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.self_growing_runner import (
    _cau_hinh_policy_chuyen_gia_cho_mt5 as _runner_cau_hinh_policy_chuyen_gia_cho_mt5,
)
from self_growing_trader.config import load_config
from self_growing_trader.replacement_learning import action_scope
from self_growing_trader.nhat_ky_giai_thich import xay_ho_so_quyet_dinh


def _config():
    return load_config("configs/he_chuyen_gia.yaml")


def _stable_system(seed: int = 19):
    config = _config()
    config = replace(config, he_chuyen_gia=replace(config.he_chuyen_gia, regime_timeframe="M15"))
    return config, tao_he_chuyen_gia_mot_context_mot_policy(config, seed=seed)


def test_v7_profile_has_no_fixed_numeric_topology_or_telemetry_caps():
    config = _config()
    assert config.model.max_modules == 0
    assert config.supernetwork.max_graph_neurons == 0
    assert config.supernetwork.max_graph_connections == 0
    assert config.supernetwork.max_hidden_neurons_per_module == 0
    assert config.observability.max_active_nodes == 0
    assert config.observability.max_active_edges == 0
    assert config.observability.max_responsible_neurons == 0
    assert config.observability.max_responsible_edges == 0


def test_owner_policy_emits_full_m1_thinking_output_and_profile():
    _config_value, system = _stable_system()
    trace = system.act(np.zeros(119, dtype=float))
    audit = dict(getattr(trace, "context_audit", {}) or {})
    expert = next(item for item in system.experts.values() if item.ma_chuyen_gia == audit["ma_chuyen_gia"])
    policy = expert.policy
    m1 = policy.modules[0].m1_component
    if m1 is None:
        module = policy.modules[0]
        assert module.hidden_size == 128
        assert module.w_in.shape == (128, 137)
        assert module.w_rec.shape == (128, 128)
        return
    thinking = m1.thinking_snapshot()

    assert thinking["schema"] == "M1_THINKING_V8"
    assert len(thinking["candidate_scores"]) >= 1
    assert len(thinking["candidate_probabilities"]) >= 1
    assert thinking["trace"]["active_edges"] > 0

    profile = xay_ho_so_quyet_dinh(policy, trace, max_neurons=0, max_edges=0)
    assert profile["dau_ra_tu_duy"]
    # 0 means all attribution rows: a seeded graph should exceed the former 40/64 UI limits.
    assert len(profile["neuron_chiu_trach_nhiem"]) > 40
    assert len(profile["lien_ket_chiu_trach_nhiem"]) > 64


def test_mt5_owner_setup_keeps_growth_local_and_graph_budget_open(tmp_path):
    config, system = _stable_system(seed=23)
    config = replace(config, observability=replace(config.observability, enabled=False))
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    for expert in system.experts.values():
        policy = expert.policy
        assert policy.graph_trial_mode is True
        assert policy.graph_trial_id.startswith("owner-" + expert.ma_chuyen_gia)
        assert policy.owner_local_mode is True
        for module in policy.modules:
            if module.m1_component is None:
                assert module.hidden_size == 128
                continue
            graph = module.m1_component.graph
            assert graph.budget.max_neurons == 0
            assert graph.budget.max_connections == 0


def test_mt5_owner_setup_applies_learning_integrity_to_direct_rnn() -> None:
    config, system = _stable_system(seed=29)
    learning = replace(
        config.learning,
        action_recovery_min_support=2.0,
        action_recovery_merge_similarity=0.73,
    )
    config = replace(config, learning=learning)

    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)

    for expert in system.experts.values():
        learner = expert.policy.action_recovery_learner
        assert learner.min_replacement_support == 2.0
        assert learner.merge_similarity == 0.73


def test_runner_mt5_owner_setup_applies_learning_integrity_to_direct_rnn() -> None:
    config, system = _stable_system(seed=31)
    learning = replace(
        config.learning,
        action_recovery_enabled=False,
        action_recovery_negative_veto_enabled=True,
        action_recovery_negative_veto_lcb=-0.12,
    )
    config = replace(
        config,
        learning=learning,
        he_chuyen_gia=replace(config.he_chuyen_gia, context_direction_prior_strength=0.07),
    )

    _runner_cau_hinh_policy_chuyen_gia_cho_mt5(system, config, policy_core="current")

    for expert in system.experts.values():
        policy = expert.policy
        assert policy.action_recovery_enabled is False
        assert policy.action_recovery_learning_enabled is True
        assert policy.action_recovery_learner.negative_veto_enabled is True
        assert policy.action_recovery_learner.negative_veto_lcb == -0.12
    assert system.cau_hinh.context_direction_prior_strength == 0.07


def test_context_owner_aggregates_policy_expectancy_for_fitness() -> None:
    _config_value, system = _stable_system(seed=43)
    expert = system.experts["CTX_VOLATILE"]
    expert.policy.ensure_compatible()
    ledger = expert.policy.memory_supernet.contextual_expectancy
    for index in range(8):
        ledger.observe(
            np.zeros(46, dtype=float),
            action_scope(1),
            0.50 + 0.02 * index,
            origin="policy",
        )

    report = system.contextual_expectancy_report()

    assert report["policy_origin_lifecycle_count"] == 8
    assert report["policy_expectancy"]["sample_count"] == 8
    assert report["policy_expectancy"]["expectancy_lcb"] > 0.0


def test_context_owner_aggregates_full_rl_audit_without_reading_decision_memory() -> None:
    _config_value, system = _stable_system(seed=44)
    expert = system.experts["CTX_VOLATILE"]
    policy = expert.policy
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_full_rl_enabled = True
    policy.ensure_compatible()
    audit = policy.full_rl_expectancy_audit
    for index in range(8):
        audit.observe(
            np.zeros(46, dtype=float),
            action_scope(1),
            0.50 + 0.02 * index,
            origin="policy",
        )

    report = system.contextual_expectancy_report()

    assert report["policy_origin_lifecycle_count"] == 8
    assert report["policy_expectancy"]["sample_count"] == 8
    assert report["policy_expectancy"]["expectancy_lcb"] > 0.0
    assert not policy.memory_supernet.contextual_expectancy.records


def test_context_owner_reports_sparse_conditioned_samples_separately_from_eligibility() -> None:
    _config_value, system = _stable_system(seed=45)
    policy = system.experts["CTX_VOLATILE"].policy
    policy.ensure_compatible()
    condition = {
        "context": "CTX_VOLATILE",
        "action": "MARKET_BUY",
        "action_index": 1,
        "trend": "up",
        "location": "far",
        "volatility": "expanding",
    }
    for index in range(3):
        policy.conditioned_entry_expectancy.observe(condition, -0.5, trace_id=f"sparse-{index}")

    conditioned = system.contextual_expectancy_report()["conditioned_entry_expectancy"]

    assert conditioned["sample_count"] == 3
    assert conditioned["eligible_record_count"] == 0
    assert conditioned["eligible_sample_count"] == 0
    assert conditioned["max_record_sample_count"] == 3
    assert conditioned["min_samples"] == 12
    assert conditioned["samples_to_first_eligible"] == 9
    assert conditioned["telemetry_only"] is True


def test_context_owner_compacts_duplicate_trace_cache_before_persisting() -> None:
    _config_value, system = _stable_system(seed=47)
    for expert in system.experts.values():
        expert.policy._causal_trace_archive = {str(index): {"trace": index} for index in range(300)}
        expert.policy._execution_trace_cache = {"duplicate": object()}
        expert.policy._entry_preference_trace_cache = {"closed-tester": object()}
    system._trace_to_expert = {"closed-tester": "CTX_RANGING"}
    system._trace_to_timing_network_trace = {"closed-tester": object()}
    system._bridge_finalized_trace_ids = {"closed-tester": True}

    _compact_context_expert_artifact_state(system)

    for expert in system.experts.values():
        assert len(expert.policy._causal_trace_archive) == 256
        assert expert.policy._execution_trace_cache == {}
        assert expert.policy._entry_preference_trace_cache == {}
    assert system._trace_to_expert == {}
    assert system._trace_to_timing_network_trace == {}
    assert system._bridge_finalized_trace_ids == {}


def test_context_owner_bounds_archived_graph_telemetry_before_pickle() -> None:
    _config_value, system = _stable_system(seed=48)
    trace = system.act(np.zeros(119, dtype=float))
    bulky = {
        "graph_id": "reasoning_graph",
        "active_neuron_ids": tuple(f"n-{index}" for index in range(300)),
        "active_edge_ids": tuple(f"e-{index}" for index in range(400)),
        "controllers": {
            "action_head": {
                "active_neuron_ids": tuple(f"n-{index}" for index in range(300)),
                "active_edge_ids": tuple(f"e-{index}" for index in range(400)),
                "node_values": {f"n-{index}": {"activation": 0.1} for index in range(300)},
                "edge_contributions": {f"e-{index}": 0.1 for index in range(400)},
                "raw_inputs": {f"n-{index}": 0.1 for index in range(300)},
            }
        },
    }
    archived = replace(trace, graph_traces={"reasoning_graph": bulky})
    expert = next(iter(system.experts.values()))
    expert.policy._causal_trace_archive = {"trace": archived}

    _compact_context_expert_artifact_state(system)

    compacted = expert.policy._causal_trace_archive["trace"]
    record = compacted.graph_traces["reasoning_graph"]
    controller = record["controllers"]["action_head"]
    assert len(record["active_neuron_ids"]) == 128
    assert len(record["active_edge_ids"]) == 192
    assert len(controller["node_values"]) == 128
    assert len(controller["edge_contributions"]) == 192
    assert "raw_inputs" not in controller
    pickle.dumps(system, protocol=pickle.HIGHEST_PROTOCOL)


def test_volatile_entry_gate_requires_direction_but_keeps_management_actions() -> None:
    config, system = _stable_system(seed=49)
    config = replace(
        config,
        learning=replace(config.learning, context_direction_entry_gate_enabled=True),
    )
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    policy = next(iter(system.experts.values())).policy
    policy._context_expert_owner_metadata = {
        "context_task": "VOLATILE_EXPANSION",
        "volatile_timing_direction_prior_bat": True,
        "timing_direction": 0,
        "timing_quality": 0.0,
    }
    mask, audit = policy._context_direction_entry_gate_mask(12)
    assert not any(mask[1:7])
    assert mask[0] and all(mask[7:])
    assert audit["context_direction_entry_gate_applied"] is True

    policy._context_expert_owner_metadata["timing_direction"] = 1
    policy._context_expert_owner_metadata["timing_quality"] = 0.20
    confirmed, confirmed_audit = policy._context_direction_entry_gate_mask(12)
    assert all(confirmed)
    assert confirmed_audit["context_direction_entry_gate_applied"] is False


def test_direction_gate_also_blocks_unconfirmed_trend_entries() -> None:
    config, system = _stable_system(seed=491)
    config = replace(
        config,
        learning=replace(config.learning, context_direction_entry_gate_enabled=True),
    )
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    policy = next(iter(system.experts.values())).policy
    policy._context_expert_owner_metadata = {
        "context_task": "GIAM_YEU",
        "timing_direction": 0,
        "timing_quality": 0.0,
    }
    mask, audit = policy._context_direction_entry_gate_mask(12)
    assert not any(mask[1:7])
    assert mask[0] and all(mask[7:])
    assert audit["context_direction_entry_gate_reason"] == "direction_unconfirmed"

    policy._context_expert_owner_metadata.update({"timing_direction": -1, "timing_quality": 0.30})
    confirmed, confirmed_audit = policy._context_direction_entry_gate_mask(12)
    assert all(confirmed)
    assert confirmed_audit["context_direction_entry_gate_applied"] is False


def test_direction_gate_blocks_unconfirmed_ranging_entries() -> None:
    config, system = _stable_system(seed=492)
    config = replace(
        config,
        learning=replace(config.learning, context_direction_entry_gate_enabled=True),
    )
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    policy = next(iter(system.experts.values())).policy
    policy._context_expert_owner_metadata = {
        "context_task": "RANGING",
        "timing_direction": 0,
        "timing_quality": 0.0,
    }
    mask, audit = policy._context_direction_entry_gate_mask(12)
    assert not any(mask[1:7])
    assert mask[0] and all(mask[7:])
    assert audit["context_direction_entry_gate_ranging"] is True


def test_expectancy_entry_gate_requires_reliable_negative_scope() -> None:
    config, system = _stable_system(seed=50)
    config = replace(
        config,
        learning=replace(
            config.learning,
            context_expectancy_entry_gate_enabled=True,
            context_expectancy_entry_gate_min_support=12.0,
            context_expectancy_entry_gate_max_lcb=-0.15,
        ),
    )
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    policy = next(iter(system.experts.values())).policy
    policy._context_expert_owner_metadata = {"context_task": "GIAM_MANH"}
    mask, audit = policy._context_expectancy_entry_gate_mask(
        12,
        {
            "expectancy_action_lcb": np.array([0.0, -0.20, -0.10, 0.0, 0.0, 0.0, 0.0]),
            "expectancy_action_samples": np.array([0.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
            "expectancy_action_confidence": np.array([0.0, 0.80, 0.80, 0.0, 0.0, 0.0, 0.0]),
            "expectancy_reliable_negative_mask": np.array([False, True, True, False, False, False, False]),
        },
    )
    assert not mask[1]
    assert mask[2]
    assert mask[0] and all(mask[7:])
    assert audit["context_expectancy_entry_gate_applied"] is True


def test_expectancy_entry_gate_allowlist_leaves_other_contexts_unmasked() -> None:
    config, system = _stable_system(seed=501)
    config = replace(
        config,
        learning=replace(
            config.learning,
            context_expectancy_entry_gate_enabled=True,
            context_expectancy_entry_gate_allowed_contexts=("CTX_VOLATILE_CHOP",),
            context_expectancy_entry_gate_min_support=12.0,
            context_expectancy_entry_gate_max_lcb=-0.15,
        ),
    )
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    policy = next(iter(system.experts.values())).policy
    negative_sell = {
        "expectancy_action_lcb": np.array([0.0, 0.0, -0.20, 0.0, 0.0, 0.0, 0.0]),
        "expectancy_action_samples": np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
        "expectancy_action_confidence": np.array([0.0, 0.0, 0.80, 0.0, 0.0, 0.0, 0.0]),
        "expectancy_reliable_negative_mask": np.array([False, False, True, False, False, False, False]),
    }

    policy._context_expert_owner_metadata = {"context_task": "GIAM_MANH"}
    mask, audit = policy._context_expectancy_entry_gate_mask(12, negative_sell)
    assert all(mask)
    assert audit["context_expectancy_entry_gate_reason"] == "context_not_allowlisted"
    assert audit["context_expectancy_entry_gate_allowed_contexts"] == ["VOLATILE_CHOP"]

    policy._context_expert_owner_metadata = {"context_task": "VOLATILE_CHOP"}
    mask, audit = policy._context_expectancy_entry_gate_mask(12, negative_sell)
    assert not mask[2]
    assert audit["context_expectancy_entry_gate_applied"] is True


def test_direction_side_gate_blocks_only_opposite_entries() -> None:
    config, system = _stable_system(seed=51)
    config = replace(
        config,
        learning=replace(config.learning, context_direction_side_entry_gate_enabled=True),
    )
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    policy = next(iter(system.experts.values())).policy
    policy._context_expert_owner_metadata = {
        "context_task": "GIAM_MANH",
        "context_on_dinh": True,
        "timing_direction": 0,
        "timing_quality": 0.0,
        "do_tin_cay_context": 0.90,
    }
    mask, audit = policy._context_direction_side_entry_gate_mask(12)
    assert not any(mask[index] for index in (1, 3, 5))
    assert all(mask[index] for index in (2, 4, 6))
    assert mask[0] and all(mask[7:])
    assert audit["context_direction_side_entry_gate_applied"] is True


def test_context_owner_defers_live_event_until_bridge_contract_is_final(tmp_path):
    """The owner facade must not publish the pre-bridge, incomplete trace."""
    from self_growing_trader.live_observatory import LiveEventReader, LiveObservabilityConfig
    from self_growing_trader.mt5_runner import enrich_step_trace

    _config_value, system = _stable_system(seed=31)
    event_path = tmp_path / "context_owner_live.jsonl"
    system.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="context-owner-bridge-final",
        source="test",
    )
    system.set_decision_context(
        account_context={"current_equity_drawdown": 0.01, "open_risk_ratio": 0.0},
        positions_pipe="",
        allowed_actions={0: True, 1: True, 2: True},
        execution_context={"spread_norm": 0.1, "margin_available_ratio": 1.0},
        goal=np.asarray([0.1, 0.2, 0.0, 0.0], dtype=float),
        sequence_id=1,
        frame_id="test:bridge-final",
        timestamp=1_700_000_000.0,
        defer_live_decision_until_bridge=True,
    )
    trace = system.act(np.zeros(119, dtype=float))

    initial = LiveEventReader(event_path).state()
    assert "decision" not in [item["event_type"] for item in initial["events"]]

    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({
        "semantic_action": 0,
        "emitted_action": 0,
        "wire_action": 0,
        "wire_matches_semantic_action": True,
        "context_action_contract_valid": True,
        "execution_bootstrap": False,
    })
    final_trace = enrich_step_trace(
        trace,
        details,
        positions_pipe="",
        account_context={"current_equity_drawdown": 0.01, "open_risk_ratio": 0.0},
    )
    system.finalize_bridge_decision(final_trace)
    system.finalize_bridge_decision(final_trace)  # sidecar retry must be idempotent

    state = LiveEventReader(event_path).state()
    decisions = [item for item in state["events"] if item["event_type"] == "decision"]
    assert len(decisions) == 1
    action = decisions[0]["payload"]["action"]
    assert action["action_contract_complete"] is True
    assert action["action_contract_state"] == "COMMAND"
    assert action["semantic_action"] == 0
    assert action["emitted_action"] == 0
    assert action["wire_action"] == 0
    causal_snapshot = decisions[0]["payload"]["causal"]["entry_snapshot"]
    assert causal_snapshot["action_contract_complete"] is True
    assert causal_snapshot["action_contract_state"] == "COMMAND"
    assert causal_snapshot["semantic_action_index"] == 0
    assert causal_snapshot["emitted_action_index"] == 0
    assert causal_snapshot["wire_action_index"] == 0


def test_live_telemetry_marks_no_command_wire_hold_as_complete_transport_contract(tmp_path):
    from self_growing_trader.live_observatory import LiveEventReader, LiveObservabilityConfig
    from self_growing_trader.mt5_runner import enrich_step_trace

    _config_value, system = _stable_system(seed=37)
    event_path = tmp_path / "context_no_command_live.jsonl"
    system.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(event_path), reset_on_process_start=True),
        run_id="context-owner-no-command",
        source="test",
    )
    system.set_decision_context(
        allowed_actions={0: True, 1: True, 2: True},
        execution_context={"spread_norm": 0.1, "margin_available_ratio": 1.0},
        frame_id="test:no-command",
        defer_live_decision_until_bridge=True,
    )
    trace = system.act(np.zeros(119, dtype=float))
    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({
        "semantic_action": -1,
        "emitted_action": -1,
        "wire_action": 0,
        "wire_noop": True,
        "no_neural_command": True,
        "wire_matches_semantic_action": True,
        "context_action_contract_valid": False,
        "execution_bootstrap": False,
    })
    final_trace = enrich_step_trace(trace, details, positions_pipe="")
    system.finalize_bridge_decision(final_trace)

    state = LiveEventReader(event_path).state()
    decision = next(item for item in state["events"] if item["event_type"] == "decision")
    action = decision["payload"]["action"]
    assert action["action_contract_complete"] is True
    assert action["action_contract_state"] == "NO_COMMAND_WIRE_NOOP"
    assert action["semantic_action"] is None
    assert action["emitted_action"] is None
    assert action["wire_action"] == 0
    assert action["wire_noop"] is True


def test_legacy_context_expert_pickle_rehydrates_bridge_finalization_state():
    _config_value, system = _stable_system(seed=41)
    del system._defer_live_decision_until_bridge
    del system._bridge_finalized_trace_ids

    restored = pickle.loads(pickle.dumps(system, protocol=pickle.HIGHEST_PROTOCOL))

    assert restored._defer_live_decision_until_bridge is False
    assert restored._bridge_finalized_trace_ids == {}
    trace = restored.act(np.zeros(119, dtype=float))
    assert str(getattr(trace, "trace_id", ""))
