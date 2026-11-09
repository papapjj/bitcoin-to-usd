from __future__ import annotations

import pickle
from dataclasses import replace as dc_replace

import numpy as np

from self_growing_trader.mistake_memory import (
    MISTAKE_EPISODE_SCHEMA_VERSION,
    MistakeEpisode,
    MistakeEpisodeStore,
    episode_from_trace,
)
from self_growing_trader.self_growing_runner import SelfGrowingPolicy, load_policy


def _loss_trace(policy: SelfGrowingPolicy):
    trace = policy.act(np.array([1.0, -0.5, 0.25, 0.75]))
    positions_pipe = "101:1:0.1:1.0:0.9:1.2:0.2:12:1.5:-0.2:0.1:1.0:0.1:1.0:4:1:1.0"
    return dc_replace(
        trace,
        action_details={
            "action_index": 1,
            "action_name": "MARKET_BUY",
            "risk_pct": 2.5,
            "sl_atr": 0.5,
            "tp_atr": 3.0,
            "trailing_stop_atr": 0.0,
            "partial_close_pct": 0.0,
            "target_slot": 0,
            "spread_points": 30.0,
        },
        positions_pipe=positions_pipe,
        account_context={
            "current_equity_drawdown": 0.05,
            "balance_drawdown": 0.04,
            "open_risk_ratio": 0.03,
            "floating_pnl_r": -0.7,
            "day_trade_pressure": 0.9,
        },
        active_neuron_ids=("m1:output:1", "m1:hidden:3"),
        active_edge_ids=("edge:entry:1",),
        graph_edge_gates={"edge:entry:1": 0.7},
        graph_region_activity={"output": 1.0},
        lifecycle_tags=("near_tp_giveback", "should_have_partial_or_trail"),
        execution_details={"bridge_observed": True, "memory_eligible": True, "spread_points": 30.0},
    )


def test_mistake_episode_jsonl_round_trip_and_indexes(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=11, m1_backend="graph")
    trace = _loss_trace(policy)
    episode = episode_from_trace(
        trace,
        np.array([0.8, -0.4, 0.2, 0.6]),
        reward=-0.35,
        cost=0.03,
        signal={"td_delta": -0.45, "prediction_error": 0.2},
        source="mt5_ledger",
    )

    assert episode.schema_version == MISTAKE_EPISODE_SCHEMA_VERSION
    assert "wrong_direction" in episode.mistake_tags
    assert "stress_entry" in episode.mistake_tags
    assert "high_spread_entry" in episode.mistake_tags
    assert "near_tp_giveback" in episode.mistake_tags
    assert episode.active_edge_ids == ["edge:entry:1"]

    store = MistakeEpisodeStore(max_episodes=8)
    stored = store.add(episode)
    store.add(MistakeEpisode.from_dict(episode.to_dict()))
    assert stored.support_count == 2
    assert store.summary()["count"] == 1
    assert store.query_by_action(1, limit=1)[0].episode_id == episode.episode_id
    assert store.query_by_tag("high_spread_entry", limit=1)[0].episode_id == episode.episode_id

    path = tmp_path / "mistakes.jsonl"
    store.to_jsonl(path)
    loaded = MistakeEpisodeStore.load_jsonl(path)
    assert loaded.summary()["count"] == 1
    assert loaded.query_by_tag("near_tp_giveback")[0].support_count == 2


def test_policy_records_negative_transition_as_mistake_episode() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=12, m1_backend="graph")
    trace = _loss_trace(policy)

    signal = policy.observe_transition(
        trace,
        np.array([0.8, -0.4, 0.2, 0.6]),
        reward=-0.35,
        cost=0.03,
        done=False,
    )

    assert signal["mistake_episode_recorded"] == 1.0
    summary = policy.mistake_episode_store.summary()
    assert summary["count"] == 1
    assert summary["top_tags"]["high_spread_entry"] == 1


def test_mt5_mistake_episode_uses_execution_requested_action_when_action_details_lack_index() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=15, m1_backend="graph")
    trace = _loss_trace(policy)
    selected_action = np.zeros(20, dtype=float)
    selected_action[0] = 1.0
    trace = dc_replace(
        trace,
        selected_action=selected_action,
        action_details={
            "action_name": "MARKET_SELL",
            "spread_points": 30.0,
        },
        execution_details={
            "bridge_observed": True,
            "memory_eligible": True,
            "requested_action": 2,
            "spread_points": 30.0,
        },
    )

    episode = episode_from_trace(
        trace,
        np.array([0.8, -0.4, 0.2, 0.6]),
        reward=-0.35,
        cost=0.03,
        signal={"td_delta": -0.45, "prediction_error": 0.2},
        source="mt5_ledger",
    )

    assert episode.action_index == 2
    assert episode.action_name == "MARKET_SELL"
    assert "wrong_direction" in episode.mistake_tags


def test_policy_load_migrates_missing_mistake_episode_store(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=13, m1_backend="graph")
    delattr(policy, "mistake_episode_store")
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    loaded = load_policy(path)

    assert isinstance(loaded.mistake_episode_store, MistakeEpisodeStore)
    assert loaded.mistake_episode_store.summary()["count"] == 0


def test_rejection_feedback_creates_mistake_episode() -> None:
    baseline = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=14, m1_backend="graph")
    rejected = baseline.copy()

    event = baseline.absorb_rejection_feedback(
        rejected,
        reject_reasons=["no_trade", "execution_reconciliation_failed"],
        metrics={
            "total_trades": 0,
            "net_profit": -10.0,
            "profit_factor": 0.5,
            "balance_drawdown_pct": 1.2,
        },
        fitness_delta=-5.0,
    )

    assert event["mistake_episode_id"]
    summary = baseline.mistake_episode_store.summary()
    assert summary["count"] == 1
    assert summary["top_tags"]["no_trade_starvation"] == 1
    assert summary["top_tags"]["execution_reject"] == 1
