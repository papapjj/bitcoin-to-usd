"""Comprehensive unit tests for RLHF Bilinear State-Action reward model,
dataset balance gate, and counterfactual candidate triad generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from self_growing_trader.rlhf_preference import (
    RLHFPreferencePair,
    parse_rlhf_preference_row,
    validate_rlhf_dataset,
    rlhf_dataset_gate,
    rlhf_dataset_balance_gate,
    rlhf_dataset_balance_validation_gate,
)
from self_growing_trader.rlhf_reward_model import (
    RLHFRewardModel,
    RLHFBilinearRewardModel,
    RLHFRewardPair,
    fit_rlhf_reward_model,
    fit_rlhf_bilinear_reward_model,
    load_rlhf_reward_model,
    score_rlhf_reward_pairs,
)
from self_growing_trader.rlhf_annotation import (
    TRIAD_ACTIONS,
    TRIAD_ACTION_NAMES,
    generate_counterfactual_triad_candidates,
    build_counterfactual_triad_tasks,
    build_rlhf_annotation_tasks,
)
from self_growing_trader.rlhf_policy_loop import (
    RLHFPolicyRollout,
    causal_policy_feature_vector,
    prepare_rlhf_rollout_batch,
)


def _make_preference_pair(
    pair_id: str,
    chosen_action: int,
    rejected_action: int,
    *,
    context_id: str = "ctx-1",
    state_hash: str = "state-hash-1",
    confidence: float = 0.9,
    split: str = "train",
) -> RLHFPreferencePair:
    row = {
        "pair_id": pair_id,
        "context_id": context_id,
        "chosen_trace_id": f"trace-chosen-{pair_id}",
        "rejected_trace_id": f"trace-rejected-{pair_id}",
        "chosen_action": int(chosen_action),
        "rejected_action": int(rejected_action),
        "decision_time": "2026-10-01T08:00:00+00:00",
        "feedback_time": "2026-10-01T08:01:00+00:00",
        "pre_decision_state_hash": state_hash,
        "risk_geometry_hash": "geometry-hash-1",
        "label_source": "human",
        "rater_id": "rater-1",
        "rater_version": "v1",
        "confidence": float(confidence),
        "split": split,
    }
    return RLHFPreferencePair(**row)


def _make_intra_state_reward_pair(
    pair_id: str,
    state: np.ndarray,
    chosen_action: int,
    rejected_action: int,
    *,
    confidence: float = 0.9,
    split: str = "train",
) -> RLHFRewardPair:
    """Build an intra-state reward pair where chosen and rejected share the exact same state s."""
    c_features = causal_policy_feature_vector(state, chosen_action)
    r_features = causal_policy_feature_vector(state, rejected_action)
    pref = _make_preference_pair(
        pair_id, chosen_action, rejected_action, confidence=confidence, split=split
    )
    return RLHFRewardPair(
        pair=pref,
        chosen_features=tuple(float(x) for x in c_features),
        rejected_features=tuple(float(x) for x in r_features),
        feature_schema="mt5_raw_obs137_action_onehot_v1",
    )


# =========================================================================
# 1. Bilinear Reward Model Architecture & Contract Tests
# =========================================================================

def test_bilinear_reward_model_initialization_and_properties() -> None:
    mean = tuple(0.0 for _ in range(137))
    scale = tuple(1.0 for _ in range(137))
    W = np.ones((7, 137), dtype=np.float64) * 0.1
    b = tuple(0.05 * a for a in range(7))

    model = RLHFBilinearRewardModel(
        feature_schema="mt5_raw_obs137_action_onehot_v1",
        mean=mean,
        scale=scale,
        weights=tuple(tuple(float(x) for x in row) for row in W),
        bias=b,
    )

    assert model.model_version == "rlhf_reward_bilinear_v2"
    assert model.feature_count == 137
    assert model.weights_array.shape == (7, 137)
    assert len(model.flattened_weights) == 7 * 137
    assert isinstance(model, RLHFRewardModel)


def test_bilinear_reward_model_rejects_invalid_dimensions() -> None:
    valid_mean = tuple(0.0 for _ in range(137))
    valid_scale = tuple(1.0 for _ in range(137))
    valid_W = tuple(tuple(0.0 for _ in range(137)) for _ in range(7))
    valid_b = tuple(0.0 for _ in range(7))

    # Mean has fewer than 137 features
    with pytest.raises(ValueError, match="at least 137 state features"):
        RLHFBilinearRewardModel(
            feature_schema="s", mean=tuple(0.0 for _ in range(10)), scale=valid_scale, weights=valid_W, bias=valid_b
        )

    # Scale contains non-positive values
    bad_scale = tuple(0.0 for _ in range(137))
    with pytest.raises(ValueError, match="scale must be positive"):
        RLHFBilinearRewardModel(
            feature_schema="s", mean=valid_mean, scale=bad_scale, weights=valid_W, bias=valid_b
        )

    # Weights have wrong number of actions (e.g. 5 instead of 7)
    bad_W = tuple(tuple(0.0 for _ in range(137)) for _ in range(5))
    with pytest.raises(ValueError, match="shape \\(7, 137\\)"):
        RLHFBilinearRewardModel(
            feature_schema="s", mean=valid_mean, scale=valid_scale, weights=bad_W, bias=valid_b
        )

    # Bias has wrong shape
    with pytest.raises(ValueError, match="7 finite elements"):
        RLHFBilinearRewardModel(
            feature_schema="s", mean=valid_mean, scale=valid_scale, weights=valid_W, bias=(0.0, 1.0)
        )


def test_score_action_and_score_contracts() -> None:
    mean = tuple(0.0 for _ in range(137))
    scale = tuple(1.0 for _ in range(137))
    W = np.zeros((7, 137), dtype=np.float64)
    # Set known weights for Action 1 (BUY) and Action 2 (SELL)
    W[1, 0] = 2.0
    W[1, 1] = 3.0
    W[2, 0] = -1.0
    b = tuple(0.1 * a for a in range(7))

    model = RLHFBilinearRewardModel(
        feature_schema="mt5_raw_obs137_action_onehot_v1",
        mean=mean,
        scale=scale,
        weights=tuple(tuple(float(x) for x in row) for row in W),
        bias=b,
    )

    state = np.zeros(137, dtype=np.float64)
    state[0] = 4.0
    state[1] = 5.0

    # Score action 1: W[1] . state + b[1] = (2*4 + 3*5) + 0.1 = 8 + 15 + 0.1 = 23.1
    expected_1 = float(2.0 * 4.0 + 3.0 * 5.0 + 0.1)
    assert np.isclose(model.score_action(state, 1), expected_1)

    # Score action 2: W[2] . state + b[2] = (-1*4) + 0.2 = -4 + 0.2 = -3.8
    expected_2 = float(-1.0 * 4.0 + 0.2)
    assert np.isclose(model.score_action(state, 2), expected_2)

    # Score action 0 (HOLD): 0 + 0.0 = 0.0
    assert np.isclose(model.score_action(state, 0), 0.0)

    # score(features) with 144-dimensional vector
    c_features = causal_policy_feature_vector(state, 1)
    assert np.isclose(model.score(c_features), expected_1)

    r_features = causal_policy_feature_vector(state, 2)
    assert np.isclose(model.score(r_features), expected_2)

    # score_pair
    pair = _make_intra_state_reward_pair("pair-1", state, chosen_action=1, rejected_action=2)
    chosen_score, rejected_score = model.score_pair(pair)
    assert np.isclose(chosen_score, expected_1)
    assert np.isclose(rejected_score, expected_2)
    assert chosen_score > rejected_score


def test_serialization_and_sha256_verification(tmp_path: Path) -> None:
    mean = tuple(float(i * 0.01) for i in range(137))
    scale = tuple(float(1.0 + i * 0.01) for i in range(137))
    W = np.random.RandomState(42).randn(7, 137) * 0.05
    b = tuple(0.01 * a for a in range(7))

    model = RLHFBilinearRewardModel(
        feature_schema="mt5_raw_obs137_action_onehot_v1",
        mean=mean,
        scale=scale,
        weights=tuple(tuple(float(x) for x in row) for row in W),
        bias=b,
    )

    row = model.to_row()
    assert row["model_version"] == "rlhf_reward_bilinear_v2"
    assert row["sha256"] == model.sha256

    # Round-trip via from_row
    reconstructed = RLHFBilinearRewardModel.from_row(row)
    assert reconstructed.sha256 == model.sha256
    assert np.allclose(reconstructed.weights_array, model.weights_array)
    assert reconstructed.bias == model.bias

    # Polymorphic loader from RLHFRewardModel.from_row
    poly_model = RLHFRewardModel.from_row(row)
    assert isinstance(poly_model, RLHFBilinearRewardModel)
    assert poly_model.sha256 == model.sha256

    # Save to disk and test load_rlhf_reward_model
    artifact_path = tmp_path / "bilinear_model.json"
    artifact_path.write_text(json.dumps(row), encoding="utf-8")
    loaded = load_rlhf_reward_model(artifact_path)
    assert isinstance(loaded, RLHFBilinearRewardModel)
    assert loaded.sha256 == model.sha256

    # Tampered sha256 fails closed
    tampered_row = dict(row)
    tampered_row["sha256"] = "badhash" * 8
    with pytest.raises(ValueError, match="sha256 mismatch"):
        RLHFBilinearRewardModel.from_row(tampered_row)


# =========================================================================
# 2. Intra-State Candidate Pair Training & Gradient Verification
# =========================================================================

def test_intra_state_candidate_pairs_update_market_weights() -> None:
    """Mathematical verification: intra-state candidate pairs (s_c = s_r = s)
    must produce non-zero gradients on W, updating market features away from 0.0.
    In contrast, the legacy linear model collapses to 0.0 on the exact same data.
    """
    rng = np.random.RandomState(1337)
    pairs: list[RLHFRewardPair] = []

    # Create 30 intra-state preference pairs:
    # 10 pairs: BUY (1) preferred over HOLD (0) in strong uptrend states
    # 10 pairs: SELL (2) preferred over HOLD (0) in strong downtrend states
    # 10 pairs: HOLD (0) preferred over BUY (1) in noisy states
    for i in range(10):
        # Uptrend state: positive momentum
        s = rng.randn(137) * 0.5
        s[0:10] += 2.0  # directional momentum
        pairs.append(_make_intra_state_reward_pair(f"up-{i}", s, chosen_action=1, rejected_action=0))

    for i in range(10):
        # Downtrend state: negative momentum
        s = rng.randn(137) * 0.5
        s[0:10] -= 2.0
        pairs.append(_make_intra_state_reward_pair(f"down-{i}", s, chosen_action=2, rejected_action=0))

    for i in range(10):
        # Choppy state: high volatility, no trend -> HOLD preferred
        s = rng.randn(137) * 1.5
        pairs.append(_make_intra_state_reward_pair(f"chop-{i}", s, chosen_action=0, rejected_action=1))

    # --- Fit Bilinear Model ---
    bilinear_model, fit_stats = fit_rlhf_bilinear_reward_model(
        pairs, epochs=60, learning_rate=0.1, l2=1e-4
    )

    W = bilinear_model.weights_array  # shape (7, 137)
    assert W.shape == (7, 137)

    # 1. Verify market feature weights are NOT all zero
    assert fit_stats["market_weights_non_zero_count"] > 0
    total_market_weights = 7 * 137
    non_zero_count = int(np.count_nonzero(W != 0.0))
    # All actions that participated (0, 1, 2) must have active market feature weights
    assert non_zero_count >= 3 * 137

    # 2. Specifically verify Action 1 (BUY) and Action 2 (SELL) and Action 0 (HOLD) weights changed
    norm_w_hold = float(np.linalg.norm(W[0]))
    norm_w_buy = float(np.linalg.norm(W[1]))
    norm_w_sell = float(np.linalg.norm(W[2]))

    assert norm_w_hold > 0.05, f"HOLD weights remained near zero: {norm_w_hold}"
    assert norm_w_buy > 0.05, f"BUY weights remained near zero: {norm_w_buy}"
    assert norm_w_sell > 0.05, f"SELL weights remained near zero: {norm_w_sell}"

    # 3. Verify optimization progressed: final loss is strictly lower than initial loss
    assert fit_stats["final_loss"] < fit_stats["initial_loss"]

    # 4. CONTRAST: Run legacy linear model on the exact same pairs
    linear_model, linear_stats = fit_rlhf_reward_model(
        pairs, epochs=60, learning_rate=0.1, l2=1e-4
    )
    linear_weights = np.asarray(linear_model.weights)
    # In the linear model, market features are the first 137 weights
    linear_market_weights = linear_weights[:137]
    assert np.all(linear_market_weights == 0.0), (
        "Legacy linear model must have collapsed market weights to 0.0 on intra-state pairs"
    )
    # Only action indicator weights (137:144) moved in the legacy model
    assert np.any(linear_weights[137:] != 0.0)


def test_fit_rlhf_bilinear_with_tasks_input() -> None:
    """Verify fit_rlhf_bilinear_reward_model accepts task dictionaries directly."""
    rng = np.random.RandomState(42)
    state = rng.randn(137)
    c0 = causal_policy_feature_vector(state, 0)
    c1 = causal_policy_feature_vector(state, 1)

    task = {
        "task_id": "task-test-1",
        "context_id": "ctx-1",
        "candidates": [
            {"trace_id": "t-hold", "action": 0, "features": list(c0)},
            {"trace_id": "t-buy", "action": 1, "features": list(c1)},
        ],
        "annotation": {
            "chosen_trace_id": "t-buy",
            "rejected_trace_id": "t-hold",
            "confidence": 0.95,
        },
    }

    model, stats = fit_rlhf_bilinear_reward_model(tasks=[task], epochs=20, learning_rate=0.05)
    assert stats["pair_count"] == 1
    assert stats["market_weights_non_zero_count"] > 0
    assert np.linalg.norm(model.weights[1]) > 0.0


# =========================================================================
# 3. Dataset Balance Gate Verification
# =========================================================================

def test_dataset_balance_gate_passes_on_balanced_dataset() -> None:
    """Balanced dataset with BUY >= 20%, SELL >= 20%, HOLD >= 20% must pass."""
    pairs: list[RLHFPreferencePair] = []
    # 30 BUY (action 1), 30 SELL (action 2), 40 HOLD (action 0)
    for i in range(30):
        pairs.append(_make_preference_pair(f"b-{i}", chosen_action=1, rejected_action=0))
    for i in range(30):
        pairs.append(_make_preference_pair(f"s-{i}", chosen_action=2, rejected_action=0))
    for i in range(40):
        pairs.append(_make_preference_pair(f"h-{i}", chosen_action=0, rejected_action=1))

    gate = rlhf_dataset_balance_gate(pairs, min_action_probability=0.20)
    assert gate["passed"] is True
    assert gate["ready_for_training"] is True
    assert gate["reasons"] == []
    assert gate["action_probabilities"]["BUY"] == 0.30
    assert gate["action_probabilities"]["SELL"] == 0.30
    assert gate["action_probabilities"]["HOLD"] == 0.40
    assert gate["min_action_probability"] == 0.30
    assert gate["min_action_probability"] >= 0.20


def test_dataset_balance_gate_rejects_100_percent_buy_dataset() -> None:
    """100% BUY dataset (0% SELL, 0% HOLD) must fail with 'action_distribution_asymmetric'."""
    pairs = [
        _make_preference_pair(f"pair-{i}", chosen_action=1, rejected_action=0)
        for i in range(50)
    ]

    gate = rlhf_dataset_balance_gate(pairs, min_action_probability=0.20)
    assert gate["passed"] is False
    assert gate["ready_for_training"] is False
    assert "action_distribution_asymmetric" in gate["reasons"]
    assert gate["action_probabilities"]["BUY"] == 1.0
    assert gate["action_probabilities"]["SELL"] == 0.0
    assert gate["action_probabilities"]["HOLD"] == 0.0
    assert gate["min_action_probability"] == 0.0


def test_dataset_balance_gate_rejects_partial_asymmetry() -> None:
    """Dataset with 70% BUY, 20% HOLD, 10% SELL violates 20% floor for SELL."""
    pairs: list[RLHFPreferencePair] = []
    for i in range(70):
        pairs.append(_make_preference_pair(f"b-{i}", chosen_action=1, rejected_action=0))
    for i in range(20):
        pairs.append(_make_preference_pair(f"h-{i}", chosen_action=0, rejected_action=1))
    for i in range(10):
        pairs.append(_make_preference_pair(f"s-{i}", chosen_action=2, rejected_action=0))

    gate = rlhf_dataset_balance_gate(pairs, min_action_probability=0.20)
    assert gate["passed"] is False
    assert "action_distribution_asymmetric" in gate["reasons"]
    assert gate["min_action_probability"] == 0.10


def test_dataset_balance_gate_exact_boundary() -> None:
    """Dataset with exactly 20% BUY, 20% SELL, 60% HOLD meets the >= 0.20 threshold."""
    pairs: list[RLHFPreferencePair] = []
    for i in range(20):
        pairs.append(_make_preference_pair(f"b-{i}", chosen_action=1, rejected_action=0))
    for i in range(20):
        pairs.append(_make_preference_pair(f"s-{i}", chosen_action=2, rejected_action=0))
    for i in range(60):
        pairs.append(_make_preference_pair(f"h-{i}", chosen_action=0, rejected_action=1))

    gate = rlhf_dataset_balance_gate(pairs, min_action_probability=0.20)
    assert gate["passed"] is True
    assert gate["min_action_probability"] == 0.20
    assert gate["reasons"] == []

    # Slightly higher requirement (0.21) rejects it
    gate_strict = rlhf_dataset_balance_gate(pairs, min_action_probability=0.21)
    assert gate_strict["passed"] is False
    assert "action_distribution_asymmetric" in gate_strict["reasons"]


def test_dataset_gate_with_require_action_balance() -> None:
    """Verify rlhf_dataset_gate integrates action balance enforcement."""
    # 100% BUY dataset across all splits
    pairs: list[RLHFPreferencePair] = []
    for split in ("train", "validation", "test"):
        for i in range(15):
            pairs.append(_make_preference_pair(
                f"{split}-{i}", chosen_action=1, rejected_action=0,
                context_id=f"ctx-{split}-{i}", split=split,
            ))

    summary = validate_rlhf_dataset(pairs)
    assert summary["valid"] is True
    assert summary["action_distribution"]["BUY"] == 1.0

    # With require_action_balance=True, rejects asymmetric distribution
    gate = rlhf_dataset_gate(
        summary, min_pairs=10, min_raters=1, require_action_balance=True
    )
    assert gate["ready_for_training"] is False
    assert "action_distribution_asymmetric" in gate["reasons"]

    # Without require_action_balance, passes basic structural checks
    gate_legacy = rlhf_dataset_gate(
        summary, min_pairs=10, min_raters=1, require_action_balance=False
    )
    assert gate_legacy["ready_for_training"] is True
    assert "action_distribution_asymmetric" not in gate_legacy["reasons"]


# =========================================================================
# 4. Counterfactual Candidate Triad Generation Tests
# =========================================================================

def test_generate_counterfactual_triad_candidates() -> None:
    state = np.linspace(-1.0, 1.0, 137, dtype=np.float32)
    feat_0 = causal_policy_feature_vector(state, 0)

    base_candidate = {
        "trace_id": "trace-root-100",
        "action": 0,
        "features": list(feat_0),
        "state_hash": "sha-state-abc",
        "geometry_hash": "sha-geom-xyz",
        "agent_index": 2,
        "action_name": "HOLD",
        "decision_time": "2026-10-01T12:00:00+00:00",
        "source_run_id": "run-42",
        "request_event_id": "evt-77",
        "request_sequence": 10,
        "rater_view": {"sl_atr": 3.0, "tp_atr": 6.0},
    }

    triad = generate_counterfactual_triad_candidates(base_candidate)
    assert len(triad) == 3
    c_hold, c_buy, c_sell = triad

    # 1. Action indices and names
    assert c_hold["action"] == 0 and c_hold["action_name"] == "HOLD"
    assert c_buy["action"] == 1 and c_buy["action_name"] == "MARKET_BUY"
    assert c_sell["action"] == 2 and c_sell["action_name"] == "MARKET_SELL"

    # 2. Immutable contexts must be identical
    for cand in (c_hold, c_buy, c_sell):
        assert cand["state_hash"] == "sha-state-abc"
        assert cand["geometry_hash"] == "sha-geom-xyz"
        assert cand["decision_time"] == "2026-10-01T12:00:00+00:00"
        assert cand["agent_index"] == 2
        # State features (0:137) must be identical
        cand_f = np.asarray(cand["features"], dtype=np.float32)
        assert len(cand_f) == 144
        assert np.allclose(cand_f[:137], state)

    # 3. Action one-hot features (137:144) must differ according to action
    assert np.isclose(c_hold["features"][137], 1.0)
    assert np.isclose(c_buy["features"][138], 1.0)
    assert np.isclose(c_sell["features"][139], 1.0)

    # 4. Trace IDs distinct and prefixed
    assert c_hold["trace_id"] == "trace-root-100:cf_hold"
    assert c_buy["trace_id"] == "trace-root-100:cf_market_buy"
    assert c_sell["trace_id"] == "trace-root-100:cf_market_sell"


def test_build_counterfactual_triad_tasks() -> None:
    state = np.random.RandomState(99).randn(137).astype(np.float32)
    feat = causal_policy_feature_vector(state, 1)

    mock_row = {
        "trace": {
            "trace_id": "tr-1",
            "raw_obs": list(state),
            "selected_idx": 1,
            "action_details": {
                "five_action_index": 1,
                "five_decision_state_sha256": "state-hash-99",
                "five_risk_geometry_sha256": "geom-hash-99",
                "action_name": "MARKET_BUY",
            },
        },
        "request": {
            "event_id": "evt-1",
            "sequence": 5,
            "time": "2026-10-01T10:00:00+00:00",
        },
        "run_id": "run-test",
    }

    tasks, stats = build_counterfactual_triad_tasks([mock_row])
    assert stats["contexts_with_triads"] == 1
    assert stats["tasks_emitted"] == 3
    assert len(tasks) == 3

    # Pairs emitted: (BUY, HOLD), (SELL, HOLD), (BUY, SELL)
    pair_actions = {(task["candidates"][0]["action"], task["candidates"][1]["action"]) for task in tasks}
    assert (1, 0) in pair_actions  # BUY vs HOLD
    assert (2, 0) in pair_actions  # SELL vs HOLD
    assert (1, 2) in pair_actions  # BUY vs SELL


def test_build_rlhf_annotation_tasks_with_triad_expansion() -> None:
    state = np.ones(137, dtype=np.float32)
    mock_row = {
        "trace": {
            "trace_id": "tr-2",
            "raw_obs": list(state),
            "selected_idx": 1,
            "action_details": {
                "five_action_index": 1,
                "five_decision_state_sha256": "state-hash-2",
                "five_risk_geometry_sha256": "geom-hash-2",
                "action_name": "MARKET_BUY",
            },
        },
        "request": {
            "event_id": "evt-2",
            "sequence": 1,
            "time": "2026-10-01T10:00:00+00:00",
        },
        "run_id": "run-2",
    }

    # Without triads: single observatory row produces 0 pairs
    _, stats_no_triad = build_rlhf_annotation_tasks([mock_row], generate_counterfactual_triads=False)
    assert stats_no_triad["tasks_emitted"] == 0

    # With triads: single observatory row expands into triad {0, 1, 2} producing 3 pairs
    tasks, stats_triad = build_rlhf_annotation_tasks([mock_row], generate_counterfactual_triads=True)
    assert stats_triad["tasks_emitted"] == 3
    assert len(tasks) == 3


# =========================================================================
# 5. Integration with Policy Loop & Pair Scoring
# =========================================================================

def test_prepare_rlhf_rollout_batch_with_bilinear_model() -> None:
    mean = tuple(0.0 for _ in range(137))
    scale = tuple(1.0 for _ in range(137))
    W = np.random.RandomState(42).randn(7, 137) * 0.05
    b = tuple(0.0 for _ in range(7))

    model = RLHFBilinearRewardModel(
        feature_schema="mt5_raw_obs137_action_onehot_v1",
        mean=mean,
        scale=scale,
        weights=tuple(tuple(float(x) for x in row) for row in W),
        bias=b,
    )

    state = np.random.RandomState(7).randn(137).astype(np.float32)
    rollouts = [
        RLHFPolicyRollout(
            rollout_id=f"rollout-{i}",
            pair_id=f"pair-{i}",
            state_hash="sh",
            geometry_hash="gh",
            feature_schema="mt5_raw_obs137_action_onehot_v1",
            features=tuple(float(x) for x in causal_policy_feature_vector(state, i % 7)),
            action=i % 7,
            old_logits=tuple(0.0 for _ in range(7)),
            reference_logits=tuple(0.0 for _ in range(7)),
            allowed_action_mask=tuple(True for _ in range(7)),
            sample_mask=True,
            value=0.0,
        )
        for i in range(4)
    ]

    batch = prepare_rlhf_rollout_batch(model, rollouts)
    assert "raw_rewards" in batch
    assert len(batch["raw_rewards"]) == 4
    assert np.all(np.isfinite(batch["raw_rewards"]))
    assert batch["reward_model_sha256"] == model.sha256


def test_score_rlhf_reward_pairs_with_bilinear_model() -> None:
    mean = tuple(0.0 for _ in range(137))
    scale = tuple(1.0 for _ in range(137))
    W = np.zeros((7, 137), dtype=np.float64)
    # Prefer BUY over SELL
    W[1, 0] = 5.0
    W[2, 0] = -5.0
    b = tuple(0.0 for _ in range(7))

    model = RLHFBilinearRewardModel(
        feature_schema="mt5_raw_obs137_action_onehot_v1",
        mean=mean,
        scale=scale,
        weights=tuple(tuple(float(x) for x in row) for row in W),
        bias=b,
    )

    state = np.zeros(137)
    state[0] = 1.0  # signal triggers W[1] > W[2]
    pairs = [
        _make_intra_state_reward_pair(f"p-{i}", state, chosen_action=1, rejected_action=2)
        for i in range(8)
    ]

    metrics = score_rlhf_reward_pairs(model, pairs)
    assert metrics["pair_count"] == 8
    assert metrics["accuracy"] == 1.0
    assert metrics["weighted_accuracy"] == 1.0
    assert metrics["model_sha256"] == model.sha256
