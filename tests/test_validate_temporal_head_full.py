import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch

from scripts import validate_temporal_head_full as runner
from self_growing_trader.five_shared_policy import (
    FactorizedActionHeads, PostLstmCrossAgentTransformer, SharedFiveTradingPolicy,
)
from self_growing_trader.five_teacher_imitation import TeacherImitationData


def dataset():
    times = np.repeat(np.arange(70), 2)
    agents = np.tile([0, 1], 70)
    actions = np.where(times % 3 == 0, agents + 1, 0)
    return TeacherImitationData(
        path=Path("synthetic.npz"), observations=np.repeat(times[:, None], 137, axis=1).astype(np.float32),
        actions=actions, candidate_actions=actions.copy(), teacher_trade_accepted=actions != 0,
        teacher_net_r=(actions != 0).astype(np.float32), agent_ids=agents,
        quality_weights=np.ones(len(times), dtype=np.float32), sample_times=times,
        label_finalized_times=times + 1, splits=np.full(len(times), "validation"), manifest={"synthetic": True},
    )


def test_subset_preserves_metadata_and_slices_every_row_field():
    source = dataset()
    selected = runner.subset(source, np.array([0, 3, 5]))
    assert selected.path == source.path
    assert selected.manifest == source.manifest
    np.testing.assert_array_equal(selected.agent_ids, [0, 1, 1])
    assert selected.observations.shape == (3, 137)
    assert len(source.actions) == 140


def test_chunked_replay_matches_native_full_replay_at_64_step_boundaries():
    torch.manual_seed(12)
    network = SharedFiveTradingPolicy(pre_lstm_size=8, hidden_size=8).eval()
    network.post_lstm_transformer = PostLstmCrossAgentTransformer(8, model_width=8).eval()
    data = dataset()
    full, audit = runner.extract_owner_hidden_sequences(network, data, batch_size=2, sequence_length=64)
    pieces = [runner.extract_owner_hidden_sequences(network, runner.subset(data, selection),
        batch_size=1, sequence_length=64)[0] for selection in (slice(0, 128), slice(128, None))]
    torch.testing.assert_close(torch.cat(pieces), full, atol=1e-6, rtol=1e-5)
    assert audit["state_reset_count"] == 2
    assert audit["unique_time_steps"] == 70


@pytest.mark.parametrize("failure", ["coverage", "shape", "nan", "sequence"])
def test_resume_rejects_incomplete_or_invalid_scores(failure):
    chunk = {"start": 0, "stop": 2, "candidate": np.zeros(2), "baseline": np.zeros(2),
             "audit": {"sequence_length": 64}}
    if failure == "coverage":
        chunk["stop"] = 3
    elif failure == "shape":
        chunk["baseline"] = np.zeros((2, 1))
    elif failure == "nan":
        chunk["candidate"][0] = np.nan
    else:
        chunk["audit"]["sequence_length"] = 32
    with pytest.raises(ValueError):
        runner.validate_chunk(chunk, 0, 2)


def test_full_run_resume_preserves_scores_and_rejects_changed_contract(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("candidate", "baseline", "config", "dataset")}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["validate", "--checkpoint", str(paths["candidate"]),
        "--baseline", str(paths["baseline"]), "--config", str(paths["config"]),
        "--dataset", str(paths["dataset"]), "--output-dir", str(output), "--batch-windows", "1"])
    torch.manual_seed(5)
    network = torch.nn.Module()
    network.hidden_size = 8
    network.heads = FactorizedActionHeads(8)
    network.register_buffer("frozen_backbone", torch.ones(1))
    monkeypatch.setattr(runner, "load_config", lambda _: None)
    monkeypatch.setattr(runner, "load_teacher_imitation_data", lambda _: dataset())
    monkeypatch.setattr(runner, "tai_he_chuyen_gia_mot_context_mot_policy", lambda *a, **kw: copy.deepcopy(network))
    monkeypatch.setattr(runner, "shared_five_network", lambda system: system)
    calls = []

    def extract(net, segment, **kwargs):
        calls.append(segment.sample_times.copy())
        assert kwargs["sequence_length"] == 64
        return torch.as_tensor(segment.observations[:, :8] / 70), {"sequence_length": 64}

    monkeypatch.setattr(runner, "extract_owner_hidden_sequences", extract)
    assert runner.main() == 2  # Identical heads cannot pass the improvement gate.
    assert [len(values) for values in calls] == [128, 12]
    assert calls[0][-1] == 63 and calls[1][0] == 64
    first = json.loads((output / "report.json").read_text())
    assert first["checks"]["all_validation_rows"]
    assert first["checks"]["dataset_unchanged"]
    assert first["metrics"]["candidate"] == first["metrics"]["baseline"]
    assert first["metrics"]["candidate"]["overall"]["samples"] == 140
    assert not first["contract"]["test_read"]
    assert not first["promotable"]
    calls.clear()
    assert runner.main() == 2
    assert calls == []
    assert json.loads((output / "report.json").read_text())["metrics"] == first["metrics"]
    paths["config"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="contract mismatch"):
        runner.main()
    paths["config"].write_bytes(b"config")
    chunk_path = output / "chunk_0000000.pkl"
    with chunk_path.open("rb") as handle:
        chunk = pickle.load(handle)
    chunk["candidate"] = chunk["candidate"][:-1]
    with chunk_path.open("wb") as handle:
        pickle.dump(chunk, handle)
    with pytest.raises(ValueError, match="chunk scores"):
        runner.main()
