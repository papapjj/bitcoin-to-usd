from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from self_growing_trader.mistake_memory import MistakeEpisode, MistakeEpisodeStore
from self_growing_trader.mistake_replay import MistakeReplayEngine
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


class _FakeTrace:
    def __init__(self, action: int, output_size: int = 20) -> None:
        self.action_details = {"action_index": action}
        self.selected_action = np.zeros(output_size, dtype=float)
        self.selected_action[action] = 1.0


class _FakePolicy:
    input_size = 4
    output_size = 20

    def __init__(self, action: int) -> None:
        self.action = action
        self._account_context = {}
        self._positions_pipe_context = ""
        self.module = SimpleNamespace(
            name="m1_thinking",
            b_out=np.zeros(self.output_size, dtype=float),
            w_out=np.zeros((self.output_size, 4), dtype=float),
            neuron_activation_ema=np.ones(4, dtype=float),
        )
        self.modules = [self.module]

    def act(self, _obs):
        return _FakeTrace(self.action, self.output_size)

    def reset_runtime_state(self):
        return None

    def set_account_context(self, context, positions_pipe=""):
        self._account_context = dict(context or {})
        self._positions_pipe_context = str(positions_pipe or "")


def _episode() -> MistakeEpisode:
    return MistakeEpisode(
        trace_id="t1",
        source="mt5_ledger",
        raw_obs=[1.0, -0.5, 0.2, 0.1],
        market_key=[0.1, 0.2, 0.3],
        account_key=[1.0, -0.2, 0.05, 0.03],
        position_key=[0.0] * 6,
        action_index=1,
        action_name="MARKET_BUY",
        risk_pct=2.5,
        module_action_traces=[{"module_name": "m1_thinking", "hidden": [1.0, 0.5, 0.25, 0.0]}],
        active_neuron_ids=["m1:output:1"],
        active_edge_ids=["edge:entry:1"],
        reward=-0.4,
        td_delta=-0.3,
        mistake_tags=["wrong_direction", "high_spread_entry"],
        severity=2.0,
    )


def test_mistake_replay_generates_replacement_without_suppressing_old_action() -> None:
    policy = _FakePolicy(action=1)
    before_bias = float(policy.module.b_out[1])

    summary = MistakeReplayEngine(max_items=4).replay(policy, [_episode()], mutate=True)

    assert summary["episodes"] == 1
    assert summary["repeated_mistakes"] == 1
    assert summary["mutated_module_links"] > 0
    assert policy.module.b_out[1] == before_bias
    assert policy.module.b_out[2] > 0.0


def test_mistake_replay_rewards_safer_alternative_lightly() -> None:
    policy = _FakePolicy(action=0)
    before_bias = float(policy.module.b_out[0])

    summary = MistakeReplayEngine(max_items=4).replay(policy, [_episode()], mutate=True)

    assert summary["episodes"] == 1
    assert summary["repeated_mistakes"] == 0
    assert summary["safer_alternatives"] == 1
    assert policy.module.b_out[0] > before_bias


def test_policy_propose_runs_mistake_replay_on_candidate_only() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=101, m1_backend="graph")
    policy.mistake_episode_store = MistakeEpisodeStore()
    policy.mistake_episode_store.add(_episode())
    baseline_events = len(policy.growth_events)

    trial = policy.propose(np.random.default_rng(123), mutation_sigma=0.01, max_modules=1)
    replay_events = [event for event in trial.growth_events if event.get("type") == "mistake_replay_candidate"]

    assert replay_events
    assert replay_events[-1]["episodes"] == 1
    assert len(policy.growth_events) == baseline_events
    assert not [event for event in policy.growth_events if event.get("type") == "mistake_replay_candidate"]
