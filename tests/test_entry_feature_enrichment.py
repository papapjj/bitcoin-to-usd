from __future__ import annotations

import numpy as np

from self_growing_trader.entry_feature_enrichment import (
    ENTRY_INTERACTION_NAMES,
    append_entry_quality_interactions,
    apply_entry_feature_quarantine,
)
from self_growing_trader.protocol_contract import INPUT_COUNT
from self_growing_trader.semantic_input_routing import build_semantic_input_route


def _allowed(*indexes: int) -> dict[int, bool]:
    active = set(indexes)
    return {index: index in active for index in range(12)}


def test_quality_interactions_keep_wire_immutable_and_orient_upper_wick() -> None:
    wire = np.zeros(INPUT_COUNT, dtype=float)
    wire[8] = 0.6
    wire[11] = 0.8
    wire[84] = 0.5
    wire[88] = 0.4
    wire[117] = 0.7
    wire[118] = 0.3
    wire[119] = 0.2

    augmented, audit = append_entry_quality_interactions(
        wire, mode="quality_interactions_v1"
    )

    assert np.array_equal(augmented[:INPUT_COUNT], wire)
    assert augmented.size == INPUT_COUNT + len(ENTRY_INTERACTION_NAMES)
    assert audit["entry_feature_interaction_count"] == len(ENTRY_INTERACTION_NAMES)
    interactions = audit["entry_feature_interactions"]
    assert interactions["upper_wick_directional_buy"] < 0.0
    assert interactions["upper_wick_directional_sell"] > 0.0


def test_no_quarantined_mode_removes_only_their_interaction_paths() -> None:
    wire = np.zeros(INPUT_COUNT, dtype=float)
    wire[8] = 0.6
    wire[11] = 0.8
    wire[84] = 0.5
    wire[88] = 0.4
    wire[117] = 0.7
    wire[118] = 0.3
    wire[119] = 0.2
    wire[3] = 0.8
    wire[5] = -0.7

    augmented, audit = append_entry_quality_interactions(
        wire, mode="quality_interactions_no_quarantined_v1"
    )

    interactions = audit["entry_feature_interactions"]
    assert interactions["long_trend_location"] > 0.0
    assert interactions["upper_wick_directional_buy"] < 0.0
    assert interactions["long_mtf_wyckoff_alignment"] == 0.0
    assert interactions["short_mtf_wyckoff_alignment"] == 0.0
    assert interactions["long_location_volatility_warning"] == 0.0
    assert interactions["short_location_volatility_warning"] == 0.0
    assert augmented.size == INPUT_COUNT + len(ENTRY_INTERACTION_NAMES)


def test_structural_sl_fields_are_causal_directional_and_bounded() -> None:
    wire = np.zeros(INPUT_COUNT, dtype=float)
    # A nearby defended low and a distant high should expose different anchor
    # evidence; neither value is a hard stop-loss override.
    wire[118] = 3.0
    wire[119] = 1.1
    wire[120] = 0.8
    wire[126] = 0.5
    wire[129] = 0.9
    wire[131] = 0.7
    wire[133] = 0.8
    wire[3] = 0.3
    wire[5] = 0.2
    wire[10] = 0.4
    wire[15] = 0.1

    augmented, audit = append_entry_quality_interactions(
        wire, mode="quality_interactions_v1"
    )

    assert np.array_equal(augmented[:INPUT_COUNT], wire)
    fields = audit["entry_feature_interactions"]
    assert -1.0 <= fields["long_structural_sl_anchor"] <= 1.0
    assert -1.0 <= fields["short_structural_sl_anchor"] <= 1.0
    assert fields["long_structural_sl_anchor"] < fields["short_structural_sl_anchor"]
    assert fields["long_structural_anchor_quality"] > fields["short_structural_anchor_quality"]
    assert -1.0 <= fields["sl_execution_buffer"] <= 1.0


def test_quarantine_changes_only_raw_entry_drive_not_local_interactions() -> None:
    vector = np.ones(INPUT_COUNT + len(ENTRY_INTERACTION_NAMES), dtype=float)
    routed, audit = apply_entry_feature_quarantine(
        vector,
        mode="quality_interactions_v1",
        quarantine_scale=0.2,
        h1_direct_scale=0.35,
    )

    assert routed[17] == 0.2
    assert routed[78] == 0.2
    assert routed[88] == 0.35
    assert np.all(routed[INPUT_COUNT:] == 1.0)
    assert audit["entry_feature_quarantine_active"] is True


def test_local_interactions_are_entry_only_while_context_history_stays_shared() -> None:
    local_count = len(ENTRY_INTERACTION_NAMES)
    size = INPUT_COUNT + 2 + local_count
    observation = np.linspace(-0.8, 0.8, size)
    entry = build_semantic_input_route(
        observation,
        mode="role_isolated_v1",
        allowed_actions=_allowed(0, 1, 2, 3, 4, 5, 6),
        input_size=size,
        entry_local_tail_count=local_count,
    )
    management = build_semantic_input_route(
        observation,
        mode="role_isolated_v1",
        allowed_actions=_allowed(0, 7, 8, 9, 10, 11),
        positions_pipe="17:1:0.1:1:0:0:0:4",
        input_size=size,
        entry_local_tail_count=local_count,
    )

    assert np.array_equal(entry.observation[INPUT_COUNT:], observation[INPUT_COUNT:])
    assert np.array_equal(
        management.observation[INPUT_COUNT:INPUT_COUNT + 2],
        observation[INPUT_COUNT:INPUT_COUNT + 2],
    )
    assert np.all(management.observation[INPUT_COUNT + 2:] == 0.0)
    assert entry.audit()["semantic_route_entry_interaction_active"] == local_count
    assert management.audit()["semantic_route_entry_interaction_active"] == 0


def test_interaction_only_policy_does_not_expose_first_two_interactions_to_management() -> None:
    """No causal-history tail means index 137 is already an interaction."""

    local_count = len(ENTRY_INTERACTION_NAMES)
    size = INPUT_COUNT + local_count
    observation = np.linspace(-0.8, 0.8, size)
    management = build_semantic_input_route(
        observation,
        mode="role_isolated_v1",
        allowed_actions=_allowed(0, 7, 8, 9, 10, 11),
        positions_pipe="17:1:0.1:1:0:0:0:4",
        input_size=size,
        context_history_tail_count=0,
        entry_local_tail_count=local_count,
    )

    assert np.all(management.observation[INPUT_COUNT:] == 0.0)
    audit = management.audit()
    assert audit["semantic_route_context_history_active"] == 0
    assert audit["semantic_route_entry_interaction_active"] == 0
