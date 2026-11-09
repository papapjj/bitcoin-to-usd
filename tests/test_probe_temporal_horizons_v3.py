from __future__ import annotations

import numpy as np

from scripts.probe_temporal_horizons_v3 import _fit_probe, _horizon_masks


def test_horizon_masks_drop_exactly_one_requested_token() -> None:
    masks = _horizon_masks()
    assert masks["all"].tolist() == [True, True, True, True]
    assert masks["drop_2"].tolist() == [False, True, True, True]
    assert masks["drop_8"].tolist() == [True, False, True, True]
    assert masks["drop_32"].tolist() == [True, True, False, True]
    assert masks["drop_128"].tolist() == [True, True, True, False]


def test_probe_selects_hyperparameter_on_validation_and_reports_test() -> None:
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(160, 6))
    validation_x = rng.normal(size=(80, 6))
    test_x = rng.normal(size=(80, 6))
    train_y = (train_x[:, 0] > 0).astype(np.int64)
    validation_y = (validation_x[:, 0] > 0).astype(np.int64)
    test_y = (test_x[:, 0] > 0).astype(np.int64)
    result = _fit_probe(train_x, train_y, validation_x, validation_y, test_x, test_y)
    assert result["selected_c_on_validation"] in {0.01, 0.1, 1.0}
    assert result["validation"]["roc_auc"] > 0.95
    assert result["test"]["roc_auc"] > 0.95
