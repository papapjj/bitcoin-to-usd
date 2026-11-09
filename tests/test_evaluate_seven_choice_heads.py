from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_seven_choice_heads import _pseudo, _self_fed_replay
from self_growing_trader.five_agent_authority import five_agent_authority_mask
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy


def _pseudo_fixture() -> SimpleNamespace:
    rows = 4
    return SimpleNamespace(
        path=Path("fixture.npz"),
        observations=np.zeros((rows, 137), dtype=np.float32),
        best_actions=np.asarray([0, 3, 0, 4], dtype=np.int64),
        outcome_best_actions=np.asarray([1, 0, 6, 0], dtype=np.int64),
        net_r=np.zeros((rows, 7), dtype=np.float32),
        agent_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        sample_times=np.asarray(["t0", "t1", "t0", "t1"]),
        splits=np.asarray(["train", "train", "test", "test"]),
        manifest={},
    )


def test_pseudo_replay_uses_causal_actions_and_future_labels_only_as_targets() -> None:
    data = _pseudo_fixture()
    pseudo = _pseudo(data)

    np.testing.assert_array_equal(pseudo.actions, data.best_actions)
    np.testing.assert_array_equal(pseudo.candidate_actions, data.outcome_best_actions)

    changed = _pseudo_fixture()
    changed.outcome_best_actions = np.asarray([6, 6, 6, 6], dtype=np.int64)
    changed_pseudo = _pseudo(changed)
    np.testing.assert_array_equal(changed_pseudo.actions, pseudo.actions)
    np.testing.assert_array_equal(changed_pseudo.candidate_actions, changed.outcome_best_actions)


def _replay_fixture(steps: int = 7) -> SimpleNamespace:
    rng = np.random.default_rng(7)
    observations = rng.normal(0.0, 0.1, (steps, 137)).astype(np.float32)
    return SimpleNamespace(
        observations=np.repeat(observations, 2, axis=0),
        agent_ids=np.tile([0, 1], steps),
        sample_times=np.repeat([f"t{step:03}" for step in range(steps)], 2),
        splits=np.asarray(["validation"] * (steps * 2)),
        best_actions=np.zeros(steps * 2, dtype=np.int64),
        outcome_best_actions=np.tile([1, 2], steps),
        net_r=np.zeros((steps * 2, 7), dtype=np.float32),
    )


def _replay_network(mode: str) -> SharedFiveTradingPolicy:
    torch.manual_seed(7)
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    network.heads.seven_choice_action_decision_mode = mode
    network.heads.seven_choice_expected_value_decision_enabled = mode == "expected_value"
    with torch.no_grad():
        if mode == "expected_value":
            head = network.observation_encoder.pattern_tree.action_path_head
            head.weight.zero_()
            head.bias.zero_()
            head.bias[2] = torch.atanh(torch.tensor(0.4))
            head.bias[7] = torch.atanh(torch.tensor(0.4))
        else:
            network.heads.direct_action_head[-1].bias[1:3] = 20.0
    return network.eval()


@pytest.mark.parametrize("mode", ["expected_value", "direct"])
def test_self_fed_replay_hidden_and_decisions_ignore_all_offline_labels(mode: str) -> None:
    network = _replay_network(mode)
    data = _replay_fixture()
    hidden, decisions, audit = _self_fed_replay(
        network, data, "validation", sequence_length=3, window_batch_size=2,
    )
    # Both causal audit labels and future labels are scoring data in this path.
    # Changing either must leave every hidden state and emitted action intact.
    changed = SimpleNamespace(**vars(data))
    changed.best_actions = np.tile([3, 4], 7)
    changed.outcome_best_actions = np.tile([5, 6], 7)
    changed.net_r = np.full_like(data.net_r, 999.0)
    relabelled_hidden, relabelled_decisions, relabelled_audit = _self_fed_replay(
        network, changed, "validation", sequence_length=3, window_batch_size=2,
    )
    np.testing.assert_array_equal(relabelled_hidden, hidden)
    np.testing.assert_array_equal(relabelled_decisions, decisions)
    assert audit["action_sequence_sha256"] == relabelled_audit["action_sequence_sha256"]
    assert audit["teacher_action_feedback"] is False
    assert np.any(decisions > 0)


@pytest.mark.parametrize("mode", ["expected_value", "direct"])
def test_batched_self_fed_replay_matches_public_forward_with_same_resets(mode: str) -> None:
    network = _replay_network(mode)
    data = _replay_fixture()
    hidden, decisions, audit = _self_fed_replay(
        network, data, "validation", sequence_length=3, window_batch_size=2,
    )
    allowed = five_agent_authority_mask()
    allowed[2:, :] = False
    allowed[2:, 0] = True
    expected_hidden = np.zeros_like(hidden)
    expected_decisions = np.zeros_like(decisions)
    with torch.no_grad():
        for step in range(7):
            if step % 3 == 0:
                memories = network.initial_memories()
            observation = torch.from_numpy(data.observations[step * 2]).expand(5, -1)
            emitted, memories, tensors = network.forward_five(
                observation, memories,
                allowed_action_masks=torch.as_tensor(allowed),
                deterministic=True, behavior_entry_probability_floor=0.0,
                behavior_uniform_mix=0.0,
            )
            expected_hidden[step * 2:step * 2 + 2] = (
                tensors["policy_hidden"][:2].cpu().numpy().astype(np.float16)
            )
            expected_decisions[step * 2:step * 2 + 2] = [
                item.action_index for item in emitted[:2]
            ]
    np.testing.assert_allclose(hidden, expected_hidden, atol=2e-3, rtol=0.0)
    np.testing.assert_array_equal(decisions, expected_decisions)
    assert audit["state_reset_count"] == 3
