from __future__ import annotations

import numpy as np

from self_growing_trader.auxiliary_objectives import (
    AUXILIARY_OUTPUT_COUNT,
    AUXILIARY_TARGET_NAMES,
    AuxiliaryTargets,
    auxiliary_loss_and_ascent,
    build_auxiliary_targets,
)
from self_growing_trader.entry_preference import EntryPathBar, EntryPreferenceRecord


def _record() -> EntryPreferenceRecord:
    return EntryPreferenceRecord(
        trace_id="aux", sequence=1, decision_time="2025.01.01 00:00",
        selected_action=1, bid=100.0, ask=100.2, atr=1.0, sl_atr=1.0, rr=2.0,
        entry_prices=(100.2, 100.0, 99.8, 100.4, 100.5, 99.7),
        bars=tuple(
            EntryPathBar(
                high=100.0 + 0.12 * index + 0.4,
                low=100.0 + 0.12 * index - 0.3,
                close=100.0 + 0.12 * index,
                spread=0.2,
            )
            for index in range(1, 49)
        ),
    )


def test_one_path_emits_dozen_scale_masked_targets() -> None:
    targets = build_auxiliary_targets(_record())

    assert AUXILIARY_OUTPUT_COUNT == 127
    assert len(AUXILIARY_TARGET_NAMES) == 127
    assert targets.values.shape == targets.mask.shape == (127,)
    assert np.count_nonzero(targets.mask) >= 13
    # The 48-minute path can mature H1/3 only.  Longer horizons must remain
    # masked instead of silently borrowing the last available close.
    assert targets.mask[0]
    assert not targets.mask[1:5].any()
    assert targets.mask[5:8].all()
    assert not targets.mask[8:20].any()
    assert np.isfinite(targets.values).all()


def test_sparse_long_horizon_fields_mature_all_requested_return_targets() -> None:
    record = _record()
    record = EntryPreferenceRecord(
        **{
            **record.__dict__,
            "future_closes": (100.5, 100.7, 101.0, 100.4, 102.0),
            "future_volatilities": (0.01, 0.02, 0.03, 0.04, 0.05),
            "observed_bars": 600,
        }
    )
    targets = build_auxiliary_targets(record)
    assert targets.mask[:25].all()


def test_auxiliary_loss_gradient_matches_direction_for_bce_quantile_and_huber() -> None:
    targets = build_auxiliary_targets(_record())
    logits = np.linspace(-0.25, 0.35, AUXILIARY_OUTPUT_COUNT)
    loss, ascent, counts = auxiliary_loss_and_ascent(logits, targets)

    assert loss > 0.0
    assert np.isfinite(ascent).all()
    assert np.count_nonzero(ascent) > 10
    assert counts["bce"] > 0
    assert counts["quantile"] == 3
    assert counts["huber"] > 0
    # An infinitesimal step along the returned ascent direction must reduce
    # the loss because the policy optimizer consumes negative-loss gradients.
    improved, _, _ = auxiliary_loss_and_ascent(
        logits + 1e-4 * ascent,
        AuxiliaryTargets(targets.values, targets.mask),
    )
    assert improved < loss
