from __future__ import annotations

import numpy as np

from self_growing_trader.timing_network import TimingNetwork
from self_growing_trader.timingnet_candidate import (
    CausalTimingFeatureAdapter,
    TimingSample,
    apply_probability_calibrator,
    build_causal_m15_tensors,
    evaluate_timingnet,
    fit_isotonic_calibrator,
    timingnet_requirements,
    train_timingnet,
    _transition_target,
)


def _samples(count: int = 96) -> list[TimingSample]:
    rows: list[TimingSample] = []
    for index in range(count):
        observation = np.zeros(78, dtype=float)
        observation[0] = float(np.sin(index / 7.0))
        observation[36] = 0.35 if (index // 4) % 2 else -0.35
        rows.append(TimingSample(
            timestamp=f"2025-01-01T{index:04d}",
            observation=observation,
            regime="CTX_TANG_MANH" if (index // 4) % 2 else "CTX_RANGING",
            trend_score=0.35 if index % 6 < 3 else 0.05,
        ))
    return rows


def test_isolated_evaluation_is_frozen_and_keeps_policy_out_of_scope() -> None:
    network = TimingNetwork.seed(hidden_size=6, embedding_size=4, seed=71)
    network.configure_deceleration({
        "timing_network_deceleration_enabled": True,
        "timing_network_deceleration_horizon": 3,
        "timing_network_deceleration_min_trend_strength": 0.18,
        "timing_network_deceleration_min_drop": 0.10,
    })
    network.configure_regime_transition({
        "timing_network_regime_transition_enabled": True,
        "timing_network_regime_transition_horizons": (3, 6),
    })
    samples = _samples()
    train_timingnet(network, samples[:72], epochs=1)
    before = network.w_in.copy()

    report = evaluate_timingnet(
        network,
        samples[72:],
        warmup=samples[48:72],
        deceleration_horizon=3,
        deceleration_min_trend_strength=0.18,
        deceleration_min_drop=0.10,
        transition_horizons=(3, 6),
    )

    assert np.array_equal(network.w_in, before)
    assert report["deceleration"]["count"] > 0
    assert report["transition"]["3"]["count"] > 0


def test_requirements_use_only_the_deployable_transition_head() -> None:
    passing = {
        "count": 200,
        "positives": 12,
        "brier_improvement": 0.08,
        "auc": 0.65,
        "support": 10,
        "precision": 0.60,
    }
    report = timingnet_requirements({
        "deceleration": passing,
        "transition": {"3": passing},
    })
    assert report["passed"] is True

    report = timingnet_requirements({
        "deceleration": {**passing, "auc": 0.55},
        "transition": {"3": passing},
    })
    assert report["passed"] is True
    assert report["deceleration_observational"] is False

    report = timingnet_requirements({
        "deceleration": passing,
        "transition": {"3": {**passing, "auc": 0.55}},
    })
    assert report["passed"] is False


def test_isotonic_calibration_is_monotone_and_serializable() -> None:
    calibrator = fit_isotonic_calibrator([0.2, 0.3, 0.8, 0.9], [0, 0, 1, 1])
    calibrated = apply_probability_calibrator(np.asarray([0.1, 0.5, 0.95]), calibrator)

    assert calibrator["kind"] == "isotonic_pav_v1"
    assert np.all(np.diff(calibrated) >= 0.0)
    assert calibrated[0] <= calibrated[-1]


def test_temporal_adapter_is_causal_and_uses_a_custom_timingnet_input() -> None:
    samples = _samples(8)
    adapter = CausalTimingFeatureAdapter(mtf_scaling_mode="native_tf_atr_v2")
    first = adapter.transform_one(samples[0])
    second = adapter.transform_one(samples[1])
    network = TimingNetwork.seed(
        input_size=adapter.input_size,
        hidden_size=6,
        embedding_size=4,
        mtf_scaling_mode="native_tf_atr_v2",
        seed=73,
    )

    output = network.forward(second, regime_label=samples[1].regime)

    assert first.size == adapter.input_size
    assert second.size == adapter.input_size
    assert output.features.size == adapter.input_size


def test_candidate_tensor_is_causal_oldest_to_newest_and_excludes_warmup_padding() -> None:
    rows = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=float)
    tensors, valid = build_causal_m15_tensors(rows, sequence_length=3)

    assert tensors.shape == (4, 3, 1)
    assert valid.tolist() == [1, 2, 3, 3]
    assert tensors[0, :, 0].tolist() == [0.0, 0.0, 1.0]
    assert tensors[3, :, 0].tolist() == [2.0, 3.0, 4.0]


def test_transition_targets_distinguish_deterioration_from_improvement() -> None:
    assert _transition_target(
        "CTX_TANG_MANH", "CTX_TANG_YEU", target_mode="trend_support_deterioration_v1",
    ) == 1
    assert _transition_target(
        "CTX_TANG_YEU", "CTX_TANG_MANH", target_mode="trend_support_deterioration_v1",
    ) == 0
    assert _transition_target(
        "CTX_TANG_MANH", "CTX_GIAM_YEU", target_mode="trend_support_deterioration_v1",
    ) == 1
    assert _transition_target(
        "CTX_RANGING", "CTX_TANG_MANH", target_mode="trend_support_deterioration_v1",
    ) is None
    assert _transition_target(
        "CTX_TANG_MANH", "CTX_TANG_YEU", target_mode="context_change_v1",
    ) == 1
