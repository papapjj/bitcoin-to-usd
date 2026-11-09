from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from self_growing_trader.m1_graph_module import _neural_action_generator_context
from self_growing_trader.mt5_runner import assess_mt5_real_tick_evidence
from self_growing_trader.partitioned_memory import PartitionedWorkingMemory
from self_growing_trader.self_growing_runner import SelfGrowingPolicy
from self_growing_trader.trading_world_model import TARGET_NAMES, TradingWorldModel


def test_v8_world_model_updates_delayed_market_heads_and_keeps_execution_evidence_separate() -> None:
    model = TradingWorldModel.seed(observation_size=119, action_size=20, seed=8)
    model.min_samples_for_evidence = 4
    action = np.zeros(20, dtype=float)
    for index in range(65):
        observation = np.zeros(119, dtype=float)
        next_observation = np.zeros(119, dtype=float)
        observation[0] = index / 100.0
        next_observation[0] = (index + 1) / 100.0
        next_observation[2] = (index + 2) / 100.0
        next_observation[3] = (index + 3) / 100.0
        next_observation[5] = (index + 4) / 100.0
        next_observation[10] = 0.2
        next_observation[15] = 0.05
        model.observe_transition(observation, action, next_observation)

    summary = model.summary()
    samples = summary["samples_by_head"]
    assert all(samples[f"return_{horizon}"] > 0 for horizon in (1, 5, 15, 60))
    assert samples["execution_cost_r"] == 0
    assert "market_opportunity" not in model.context(np.zeros(119), action)

    model.observe_transition(
        np.zeros(119),
        action,
        np.zeros(119),
        reward=1.0,
        cost=0.2,
        execution_details={"filled": 1, "mae_r": -0.4, "mfe_r": 1.3},
        lifecycle_tags=("tp_hit",),
        execution_feedback_eligible=True,
    )
    samples = model.summary()["samples_by_head"]
    assert samples["execution_cost_r"] > 0
    assert samples["net_r"] > 0
    assert samples["fill_probability"] > 0
    assert samples["tp_before_sl"] > 0
    assert set(TARGET_NAMES) == set(model.summary()["heads"])


def test_v8_partitioned_memory_resets_position_without_erasing_market_or_execution() -> None:
    memory = PartitionedWorkingMemory()
    active = np.zeros(119, dtype=float)
    active[1] = 0.7
    active[34] = 0.9
    active[108] = 0.6
    first = memory.step(active)
    assert first["partitioned_memory_position_active"] == 1.0
    assert first["market_memory_norm"] > 0.0
    assert first["position_lifecycle_memory_norm"] > 0.0
    assert first["execution_memory_norm"] > 0.0

    flat = np.zeros(119, dtype=float)
    flat[1] = 0.7
    second = memory.step(flat)
    assert second["partitioned_memory_position_active"] == 0.0
    assert second["position_lifecycle_memory_norm"] == 0.0
    assert second["market_memory_norm"] > 0.0
    assert second["execution_memory_norm"] < first["execution_memory_norm"]


def test_v8_policy_world_model_is_shadow_only_and_observes_without_hidden_exception() -> None:
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, hidden_size=6, seed=18)
    trace = policy.act(np.zeros(119, dtype=float))
    detail = trace.action_details
    assert detail["mechanism_schema_version"] == 8
    assert detail["curiosity_live_applied"] is False
    assert detail["curiosity"]["live_action_delta_norm"] == 0.0
    policy.observe_transition(trace, np.ones(119, dtype=float) * 0.01, reward=0.0, cost=0.0, done=False)
    assert "error" not in getattr(policy, "_v8_last_world_update", {})
    assert policy.mechanism_summary()["schema_version"] == 8


def test_v8_legacy_opportunity_cost_never_generates_entry_evidence() -> None:
    generator = _neural_action_generator_context(
        {"opportunity_cost": 1.0, "uncertainty": 1.0, "runtime_starvation": 3.0},
        size=20,
    )
    assert generator["opportunity"] == 0.0
    assert generator["verified_entry_evidence"] is False
    assert np.allclose(generator["probe"], 0.0)


def test_v8_real_tick_evidence_requires_model_fresh_report_and_marker(tmp_path: Path) -> None:
    report = tmp_path / "report.htm"
    started = time.time_ns()
    report.write_text("Every tick based on real ticks", encoding="utf-16")
    verified = assess_mt5_real_tick_evidence(report, requested_model=4, started_at_ns=started)
    assert verified["verified"] is True

    missing_marker = tmp_path / "no_marker.htm"
    missing_marker.write_text("Every tick", encoding="utf-16")
    rejected = assess_mt5_real_tick_evidence(missing_marker, requested_model=4)
    assert rejected["verified"] is False
    assert rejected["reason"] == "real_tick_marker_missing"
