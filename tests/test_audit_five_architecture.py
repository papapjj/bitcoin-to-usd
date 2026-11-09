from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.audit_five_architecture import (
    _class_geometry,
    _owner_action_labels,
    _owner_agent_ids,
    _owner_entry_labels,
    _paired_balanced_accuracy_delta_ci,
    _pca_probe_saturation_dimension,
    _rank_metrics,
    _temporal_role_rows,
    _threshold_sweep,
    _transplant_selected_lstm_units,
)
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy
from scripts.audit_stage13_followup import (
    AUTHORIZED_PREFIXES,
    _class_weights,
    _constrained_threshold_sweep,
    _load_and_apply_action_state,
    _validation_key,
)


def test_rank_metrics_detects_low_rank_hidden_space() -> None:
    rng = np.random.default_rng(7)
    factors = rng.normal(size=(240, 3)).astype(np.float32)
    basis = rng.normal(size=(3, 32)).astype(np.float32)
    values = factors @ basis

    metrics = _rank_metrics(values, components=31)

    assert metrics["rank_95pct"] <= 3
    assert metrics["rank_99pct"] <= 3
    assert metrics["variance_captured_by_svd"] > 0.999


def test_owner_entry_labels_matches_replay_step_major_order() -> None:
    sampled = {
        "actions": np.asarray([
            [[0, 1, 0, 0, 0], [2, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 3, 0, 0, 0]],
        ]),
        "owner_masks": np.asarray([
            [[True, True, False, False, False], [True, False, False, False, False]],
            [[True, False, False, False, False], [False, True, False, False, False]],
        ]),
    }

    labels = _owner_entry_labels(sampled)

    assert labels.tolist() == [0, 1, 0, 1, 1]


def test_owner_action_and_agent_labels_match_replay_step_major_order() -> None:
    sampled = {
        "actions": np.asarray([
            [[0, 1, 0, 0, 0], [2, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 3, 0, 0, 0]],
        ]),
        "owner_masks": np.asarray([
            [[True, True, False, False, False], [True, False, False, False, False]],
            [[True, False, False, False, False], [False, True, False, False, False]],
        ]),
    }

    assert _owner_action_labels(sampled).tolist() == [0, 1, 0, 2, 3]
    assert _owner_agent_ids(sampled).tolist() == [0, 1, 0, 0, 1]


def test_class_geometry_detects_separated_clusters_and_test_generalization() -> None:
    rng = np.random.default_rng(11)
    train_hold = rng.normal(loc=0.0, scale=0.2, size=(200, 8)).astype(np.float32)
    train_buy = rng.normal(loc=2.0, scale=0.2, size=(80, 8)).astype(np.float32)
    test_hold = rng.normal(loc=0.0, scale=0.2, size=(100, 8)).astype(np.float32)
    test_buy = rng.normal(loc=2.0, scale=0.2, size=(40, 8)).astype(np.float32)

    result = _class_geometry(
        np.concatenate((train_hold, train_buy)),
        np.asarray([0] * len(train_hold) + [1] * len(train_buy)),
        np.concatenate((test_hold, test_buy)),
        np.asarray([0] * len(test_hold) + [1] * len(test_buy)),
        class_ids=(0, 1),
        class_names={0: "HOLD", 1: "BUY"},
        bootstraps=50,
    )

    pair = result["pairwise"]["HOLD__vs__BUY"]
    nearest = result["nearest_train_centroid_test"]
    assert pair["train_separation_over_pooled_radius"] > 5.0
    assert nearest["balanced_accuracy"] > 0.99
    assert nearest["balanced_accuracy_bootstrap_95pct_ci"][0] > 0.99


def test_paired_balanced_accuracy_delta_ci_uses_same_stratified_samples() -> None:
    labels = np.asarray([0] * 50 + [1] * 50)
    weaker = np.asarray([0] * 50 + [0] * 20 + [1] * 30)
    stronger = labels.copy()

    interval = _paired_balanced_accuracy_delta_ci(
        labels, weaker, stronger, class_ids=(0, 1), bootstraps=200
    )

    assert interval[0] > 0.10
    assert interval[1] < 0.30


def test_pca_probe_saturation_dimension_uses_validation_gain_only() -> None:
    rows = [
        {"components": 1, "validation": {"balanced_accuracy": 0.55}},
        {"components": 2, "validation": {"balanced_accuracy": 0.59}},
        {"components": 4, "validation": {"balanced_accuracy": 0.60}},
        {"components": 8, "validation": {"balanced_accuracy": 0.605}},
    ]

    assert _pca_probe_saturation_dimension(rows) == 4


def test_temporal_role_rows_aligns_current_hidden_with_past_same_role_action() -> None:
    hidden = np.zeros((1, 4, 2, 3), dtype=np.float32)
    for step in range(4):
        hidden[0, step, 0] = step
    actions = np.asarray([[[0, 0], [1, 0], [0, 0], [1, 0]]])
    masks = np.ones((1, 4, 2), dtype=bool)

    features, labels = _temporal_role_rows(
        hidden, actions, masks, agent_id=0, lag=2
    )

    assert features[:, 0].tolist() == [2.0, 3.0]
    assert labels.tolist() == [0, 1]


def test_structured_lstm_transplant_builds_real_narrower_network() -> None:
    source = SharedFiveTradingPolicy(hidden_size=8)
    selected = np.asarray([1, 3, 5, 7])

    target = _transplant_selected_lstm_units(source, selected)

    assert target.hidden_size == 4
    assert target.lstm.weight_ih.shape == (16, source.pre_lstm_size)
    assert target.lstm.weight_hh.shape == (16, 4)
    assert target.heads.action_query.weight.shape[1] == 4
    assert target.post_lstm_transformer.output_projection.weight.shape[0] == 4


def test_threshold_sweep_uses_validation_margin_and_finds_separating_cut() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    margins = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])

    result = _threshold_sweep(labels, margins, threshold_count=21)

    assert -1.0 < result["selected_threshold"] <= 1.0
    assert result["selected_validation_metrics"]["balanced_accuracy"] == 1.0


def test_constrained_threshold_sweep_respects_validation_rate_and_precision() -> None:
    labels = np.asarray([0] * 90 + [1] * 10)
    margins = np.asarray(list(np.linspace(-3.0, -0.2, 90)) + list(np.linspace(0.2, 2.0, 10)))

    result = _constrained_threshold_sweep(labels, margins)
    selected = result["selected_validation_metrics"]

    assert selected["eligible"] is True
    assert selected["predicted_entry_rate"] <= selected["rate_ceiling"]
    assert selected["entry_precision"] >= selected["precision_floor"]
    assert selected["balanced_accuracy"] == 1.0


def test_class_weight_modes_are_ordered_and_normalized() -> None:
    labels = torch.tensor([0] * 90 + [1] * 10)
    unweighted = _class_weights(labels, "unweighted")
    sqrt_inverse = _class_weights(labels, "sqrt_inverse")
    capped = _class_weights(labels, "capped_inverse_3")
    balanced = _class_weights(labels, "balanced_inverse")

    assert torch.allclose(unweighted, torch.ones(2))
    assert all(torch.isclose(value.mean(), torch.tensor(1.0)) for value in (sqrt_inverse, capped, balanced))
    assert 1.0 < float(sqrt_inverse[1] / sqrt_inverse[0]) <= float(capped[1] / capped[0])
    assert float(capped[1] / capped[0]) == pytest.approx(3.0)
    assert float(balanced[1] / balanced[0]) == pytest.approx(9.0)


def test_action_state_loader_rejects_non_authorized_key(tmp_path) -> None:
    checkpoint = tmp_path / "source.pkl"
    checkpoint.write_bytes(b"source")
    network = SharedFiveTradingPolicy(hidden_size=8)
    state = {
        name: value.detach().clone()
        for name, value in network.state_dict().items()
        if name.startswith(AUTHORIZED_PREFIXES)
    }
    state["heads.value.bias"] = network.state_dict()["heads.value.bias"].detach().clone()
    artifact = tmp_path / "bad.pt"
    torch.save({
        "state_dict": state,
        "teacher_entry_scope": "all_entry_mass_v2",
        "source_checkpoint_sha256": __import__("hashlib").sha256(b"source").hexdigest(),
        "candidate_only": True,
    }, artifact)

    with pytest.raises(ValueError, match="authorization fail"):
        _load_and_apply_action_state(network, checkpoint, artifact)


def test_validation_key_breaks_operating_point_tie_with_ranking_quality() -> None:
    def sweeps(roc_auc: float, average_precision: float) -> dict[str, object]:
        selected = {
            "eligible": True,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "mcc": 0.1,
            "balanced_accuracy": 0.55,
            "entry_precision": 0.35,
        }
        return {
            "agent_0_buy_role": {"selected_validation_metrics": selected},
            "agent_1_sell_role": {"selected_validation_metrics": selected},
        }

    assert _validation_key(sweeps(0.62, 0.20), 0.8) > _validation_key(sweeps(0.55, 0.15), 0.2)
