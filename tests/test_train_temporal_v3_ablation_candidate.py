from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch
import numpy as np

from scripts.train_temporal_v3_ablation_candidate import (
    _metrics,
    _forward_window_batch,
    _role_conditioned_loss,
    _select_train_boundary,
    _selection_key,
    _temporal_parameter,
    _trainable_parameter,
)
from self_growing_trader.five_shared_policy import (
    FactorizedActionHeads,
    PostLstmCrossAgentTransformer,
)


def test_temporal_ablation_surface_excludes_v2_backbone() -> None:
    assert _temporal_parameter("post_lstm_transformer.temporal_scale")
    assert _temporal_parameter(
        "post_lstm_transformer.temporal_encoder_layer.self_attn.in_proj_weight"
    )
    assert not _temporal_parameter("post_lstm_transformer.residual_scale")
    assert not _temporal_parameter("post_lstm_transformer.encoder_layer.linear1.weight")
    assert _trainable_parameter(
        "heads.entry_role_calibrator.role_calibrators.0.output.weight"
    )
    assert _trainable_parameter("heads.action_query.weight")
    assert not _trainable_parameter("heads.action_embedding.weight")


def test_temporal_validation_selection_uses_worst_role_first() -> None:
    metrics = {
        "overall": {"balanced_accuracy": 0.75},
        "roles": {
            "buy": {
                "balanced_accuracy": 0.70, "mcc": 0.30, "entry_recall": 0.50,
                "entry_precision": 0.40, "predicted_entry_rate": 0.10,
                "label_entry_rate": 0.10,
            },
            "sell": {
                "balanced_accuracy": 0.60, "mcc": 0.20, "entry_recall": 0.40,
                "entry_precision": 0.35, "predicted_entry_rate": 0.12,
                "label_entry_rate": 0.10,
            },
        },
    }
    metrics["roles"]["buy"]["roc_auc"] = 0.72
    metrics["roles"]["sell"]["roc_auc"] = 0.66
    assert _selection_key(metrics) == (2.0, 0.66, 0.20, 0.60, 0.35, 0.40, 0.0, 0.75)


def test_role_conditioned_loss_penalizes_collapsed_entry_rate() -> None:
    labels = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.long)
    roles = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    collapsed = torch.tensor([[0.0, 5.0]] * 8, requires_grad=True)
    calibrated = torch.tensor(
        [[1.0, -1.0]] * 3 + [[-1.0, 1.0]]
        + [[1.0, -1.0]] * 3 + [[-1.0, 1.0]],
        requires_grad=True,
    )
    collapsed_loss, collapsed_parts = _role_conditioned_loss(
        collapsed, labels, roles, false_entry_penalty=0.0,
        ranking_margin=2.0, ranking_loss_weight=0.5, rate_calibration_weight=10.0,
    )
    calibrated_loss, calibrated_parts = _role_conditioned_loss(
        calibrated, labels, roles, false_entry_penalty=0.0,
        ranking_margin=2.0, ranking_loss_weight=0.5, rate_calibration_weight=10.0,
    )
    assert calibrated_loss < collapsed_loss
    assert calibrated_parts["rate_calibration"] < collapsed_parts["rate_calibration"]
    calibrated_loss.backward()
    assert calibrated.grad is not None


def test_train_boundary_is_role_local_and_folds_to_zero() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([-2.0, -1.0, -0.4, 0.1, 0.8, 1.5], dtype=np.float64)
    threshold, metrics = _select_train_boundary(labels, scores)
    folded_predictions = ((scores - threshold) > 0.0).astype(np.int64)
    assert metrics["mcc"] == 1.0
    assert metrics["eligible"] is True
    assert np.array_equal(folded_predictions, labels)
    assert -0.4 <= threshold < 0.8


def test_train_boundary_rejects_constant_uninformative_scores() -> None:
    _, metrics = _select_train_boundary(np.array([0] * 9 + [1]), np.zeros(10))
    assert metrics["eligible"] is False


def test_exact_train_sweep_matches_exhaustive_tied_score_search() -> None:
    rng = np.random.default_rng(44)
    scores = np.round(rng.normal(size=90), 1)
    labels = (rng.random(90) < .25).astype(int)
    threshold, actual = _select_train_boundary(labels, scores)
    rows = []
    for boundary in np.r_[scores.min() - 1e-7, np.unique(scores)]:
        row = _metrics(labels, (scores > boundary).astype(int), scores)
        eligible = (row["balanced_accuracy"] >= .52 and row["mcc"] >= .05
                    and row["entry_recall"] >= .20 and row["entry_precision"] >= .30
                    and row["predicted_entry_rate"] <= 1.2 * row["label_entry_rate"])
        key = (eligible, row["mcc"], row["balanced_accuracy"], row["entry_precision"],
               row["entry_recall"], -max(0, row["predicted_entry_rate"] - 1.2 * row["label_entry_rate"]),
               -abs(row["predicted_entry_rate"] - row["label_entry_rate"]))
        rows.append((key, boundary))
    expected = max(rows, key=lambda row: row[0])[1]
    np.testing.assert_array_equal(scores > threshold, scores > expected)


def test_burn_in_preserves_past_memory_without_scoring_prefix() -> None:
    torch.manual_seed(12)
    network = torch.nn.Module()
    network.hidden_size = 8
    network.post_lstm_transformer = PostLstmCrossAgentTransformer(8, model_width=8)
    network.heads = FactorizedActionHeads(8)
    network.eval()
    hidden = torch.randn(1, 4, 5, 8)
    sampled = {
        "actions": np.zeros((1, 4, 5), dtype=np.int64),
        "owner_masks": np.ones((1, 4, 5), dtype=bool),
    }
    sampled["actions"][0, 2:, 0] = 1
    calls = []
    hook = network.post_lstm_transformer.register_forward_hook(
        lambda *args: calls.append(1)
    )
    with torch.inference_mode():
        full = _forward_window_batch(network, hidden, sampled, np.array([0]))
        assert len(calls) == 4
        calls.clear()
        warmed = _forward_window_batch(
            network, hidden, sampled, np.array([0]), burn_in=2
        )
    hook.remove()
    assert len(calls) == 2
    for actual, reference in zip(warmed, full):
        torch.testing.assert_close(actual, reference[4:])


def test_role_loss_is_finite_when_batch_has_only_one_owner() -> None:
    logits = torch.tensor([[0.0, -1.0], [0.0, 1.0]], requires_grad=True)
    loss, _ = _role_conditioned_loss(
        logits, torch.tensor([0, 1]), torch.tensor([0, 0]),
        false_entry_penalty=0.1, ranking_margin=2.0,
        ranking_loss_weight=1.0, rate_calibration_weight=10.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_temporal_ablation_trainer_direct_cli_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_temporal_v3_ablation_candidate.py"), "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
