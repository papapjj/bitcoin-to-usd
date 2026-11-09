from __future__ import annotations

import numpy as np

from self_growing_trader.feature_contract import feature_schema_fingerprint
from self_growing_trader.input_normalization import (
    INPUT_COUNT,
    INPUT_NORMALIZATION_CLIP,
    INPUT_NORMALIZATION_GROUPWISE_MTF_V3,
    INPUT_NORMALIZATION_GROUPWISE_V2,
    INPUT_NORMALIZATION_RAW,
    INPUT_NORMALIZATION_SOFT_CLIP_V1,
    INPUT_NORMALIZATION_STRUCTURE_AWARE_V4,
    INPUT_NORMALIZATION_TANH_V1,
    input_normalization_summary,
    normalize_input_vector,
)


def test_raw_mode_is_legacy_identity_and_tanh_mode_is_bounded() -> None:
    raw = np.linspace(-80.0, 80.0, INPUT_COUNT)
    assert np.array_equal(normalize_input_vector(raw, mode=INPUT_NORMALIZATION_RAW), raw)

    bounded = normalize_input_vector(raw, mode=INPUT_NORMALIZATION_TANH_V1)
    assert np.all(np.isfinite(bounded))
    assert float(np.max(np.abs(bounded))) <= 1.0
    assert np.allclose(bounded, -normalize_input_vector(-raw, mode=INPUT_NORMALIZATION_TANH_V1))
    assert np.isclose(bounded[0], -np.tanh(INPUT_NORMALIZATION_CLIP))
    assert np.isclose(bounded[-1], np.tanh(INPUT_NORMALIZATION_CLIP))


def test_soft_clip_preserves_normal_region_and_compresses_extreme_tail() -> None:
    raw = np.asarray([-60.0, -1.0, -0.5, 0.0, 0.5, 1.0, 60.0], dtype=float)
    clipped = normalize_input_vector(raw, mode=INPUT_NORMALIZATION_SOFT_CLIP_V1, size=raw.size)
    assert np.all(np.isfinite(clipped))
    assert np.allclose(clipped[1:6], raw[1:6])
    assert np.allclose(clipped, -normalize_input_vector(-raw, mode=INPUT_NORMALIZATION_SOFT_CLIP_V1, size=raw.size))
    assert float(np.max(np.abs(clipped))) < 2.0
    assert clipped[0] < -1.9 and clipped[-1] > 1.9


def test_normalization_summary_exposes_saturation_without_leaking_rows() -> None:
    raw = np.zeros(INPUT_COUNT, dtype=float)
    raw[5] = 60.0
    summary = input_normalization_summary(raw, mode=INPUT_NORMALIZATION_TANH_V1)
    assert summary["mode"] == INPUT_NORMALIZATION_TANH_V1
    assert summary["raw_max_abs"] == 60.0
    assert summary["normalized_max_abs"] <= 1.0
    assert summary["saturated_count"] == 1


def test_normalized_feature_fingerprint_is_distinct_but_raw_hash_is_stable() -> None:
    from self_growing_trader.config import FeatureConfig, SymbolConfig

    raw_hash = feature_schema_fingerprint(FeatureConfig(), SymbolConfig())
    explicit_raw_hash = feature_schema_fingerprint(
        FeatureConfig(), SymbolConfig(), input_normalization_mode=INPUT_NORMALIZATION_RAW
    )
    normalized_hash = feature_schema_fingerprint(
        FeatureConfig(), SymbolConfig(), input_normalization_mode=INPUT_NORMALIZATION_TANH_V1
    )
    assert raw_hash == explicit_raw_hash
    assert normalized_hash != raw_hash
    soft_clip_hash = feature_schema_fingerprint(
        FeatureConfig(), SymbolConfig(), input_normalization_mode=INPUT_NORMALIZATION_SOFT_CLIP_V1
    )
    assert soft_clip_hash not in {raw_hash, normalized_hash}
    groupwise_hash = feature_schema_fingerprint(
        FeatureConfig(), SymbolConfig(), input_normalization_mode=INPUT_NORMALIZATION_GROUPWISE_V2
    )
    assert groupwise_hash not in {raw_hash, normalized_hash, soft_clip_hash}
    mtf_hash = feature_schema_fingerprint(
        FeatureConfig(), SymbolConfig(), input_normalization_mode=INPUT_NORMALIZATION_GROUPWISE_MTF_V3
    )
    assert mtf_hash not in {raw_hash, normalized_hash, soft_clip_hash, groupwise_hash}
    structural_hash = feature_schema_fingerprint(
        FeatureConfig(), SymbolConfig(), input_normalization_mode=INPUT_NORMALIZATION_STRUCTURE_AWARE_V4
    )
    assert structural_hash not in {raw_hash, normalized_hash, soft_clip_hash, groupwise_hash, mtf_hash}


def test_groupwise_v2_compresses_stateful_tails_without_distorting_local_market_signal() -> None:
    raw = np.zeros(INPUT_COUNT, dtype=float)
    raw[0] = 0.75
    raw[29] = 20.0
    raw[69] = -20.0
    raw[106] = 20.0
    raw[93] = 75.0  # chop index is [0, 100]
    raw[135] = 60.0
    normalized = normalize_input_vector(raw, mode=INPUT_NORMALIZATION_GROUPWISE_V2)
    assert normalized[0] == raw[0]
    assert 0.0 < normalized[29] <= 1.0
    assert -1.0 <= normalized[69] < 0.0
    assert np.isclose(normalized[93], 0.5)
    assert np.isclose(normalized[135], 1.0)


def test_groupwise_mtf_v3_keeps_native_h1_variation_off_the_generic_plateau() -> None:
    raw = np.zeros(INPUT_COUNT, dtype=float)
    # Native H1 EMA values would hit the V2 generic soft-clip tail at these
    # magnitudes. V3 stays smooth and bounded near +/-1.2 instead.
    raw[88] = -2.0
    raw[90] = 0.82
    normalized = normalize_input_vector(raw, mode=INPUT_NORMALIZATION_GROUPWISE_MTF_V3)
    assert -1.2 < normalized[88] < -1.1
    assert 0.7 < normalized[90] < 0.9
    assert not np.isclose(normalized[88], normalize_input_vector(raw, mode=INPUT_NORMALIZATION_GROUPWISE_V2)[88])


def test_structure_aware_v4_preserves_structural_distance_and_corrects_unit_chop_contract() -> None:
    low = np.zeros(INPUT_COUNT, dtype=float)
    high = np.zeros(INPUT_COUNT, dtype=float)
    low[93], high[93] = 0.40, 0.60  # EA emits chop as [0, 1], not [0, 100].
    low[119], high[119] = 4.0, 12.0
    low[90], high[90] = 0.85, 1.20  # H1 volatility is a ratio around one.
    transformed_low = normalize_input_vector(low, mode=INPUT_NORMALIZATION_STRUCTURE_AWARE_V4)
    transformed_high = normalize_input_vector(high, mode=INPUT_NORMALIZATION_STRUCTURE_AWARE_V4)
    assert np.isclose(transformed_low[93], -0.20)
    assert np.isclose(transformed_high[93], 0.20)
    assert 0.0 < transformed_low[119] < transformed_high[119] < 1.0
    assert transformed_low[90] < 0.0 < transformed_high[90]
    # V3 inherits auto-scaling from V2, correctly mapping unit-interval chop 0.40 to -0.20.
    assert np.isclose(normalize_input_vector(low, mode=INPUT_NORMALIZATION_GROUPWISE_MTF_V3)[93], -0.20)


def test_position_snapshot_parser_keeps_append_only_immutable_risk_distance() -> None:
    from self_growing_trader.feature_contract import _parse_positions_snapshot

    # Owner fields retain offsets 17..20; V155 appends the immutable initial
    # risk distance at index 21 so trailing SL cannot rewrite unrealized-R.
    parsed = _parse_positions_snapshot(
        "42:1:0.10:100:99:103:0.5:4:0:0:0:0:0:0:0:0:0:"
        "CTX:EXPERT:POLICY:trace-42:2.5"
    )

    assert parsed[0]["unrealized_r"] == 0.5
    assert parsed[0]["initial_risk_distance_price"] == 2.5


def test_stp_input_clip_limits_tail_amplification_without_changing_default() -> None:
    from self_growing_trader.chuyen_gia_policy import ModuleState

    observation = np.full(4, 20.0, dtype=float)
    legacy = ModuleState.create("legacy", 4, 4, 2, np.random.default_rng(7))
    bounded = legacy.copy()
    bounded.stp_input_clip = 0.5
    legacy.activate(observation)
    bounded.activate(observation)
    assert float(np.max(legacy.p_stp)) > 2.0
    assert float(np.max(bounded.p_stp)) < 1.1


def test_output_rms_normalization_only_dampens_action_block() -> None:
    from self_growing_trader.output_normalization import normalize_policy_output

    values = np.asarray([2.0] * 12 + [4.0] * 8, dtype=float)
    normalized = normalize_policy_output(values, mode="rms_v1", target_rms=0.5)
    assert np.isclose(np.sqrt(np.mean(np.square(normalized[:12]))), 0.5)
    assert np.array_equal(normalized[12:], values[12:])
    assert np.array_equal(normalize_policy_output(values, mode="raw"), values)


def test_output_layernorm_is_per_forward_pass_and_leaves_parameters_untouched() -> None:
    from self_growing_trader.output_normalization import normalize_policy_output

    values = np.asarray([-8.0, -2.0, 0.0, 1.0, 3.0] + [0.5] * 7 + [4.0] * 8, dtype=float)
    normalized = normalize_policy_output(values, mode="layernorm_v1", target_rms=0.5)
    action = normalized[:12]
    assert np.all(np.isfinite(normalized))
    assert abs(float(np.mean(action))) < 1e-6
    assert np.isclose(np.sqrt(np.mean(np.square(action))), 0.5, atol=1e-5)
    assert np.array_equal(normalized[12:], values[12:])


def test_output_rms_vjp_matches_finite_difference_on_live_action_block() -> None:
    from self_growing_trader.output_normalization import (
        normalize_policy_output,
        policy_output_normalization_vjp,
    )

    raw = np.asarray([-2.0, -1.2, -0.6, -0.2, 0.3, 0.7, 1.1, 1.8], dtype=float)
    upstream = np.asarray([0.4, -0.3, 0.8, -0.6, 0.2, 0.5, -0.2, 0.1], dtype=float)
    vjp = policy_output_normalization_vjp(raw, upstream, mode="rms_v1", target_rms=0.5)
    epsilon = 1.0e-6
    numeric = np.zeros_like(raw)
    for index in range(raw.size):
        plus = raw.copy()
        minus = raw.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (
            float(np.dot(normalize_policy_output(plus, mode="rms_v1", target_rms=0.5), upstream))
            - float(np.dot(normalize_policy_output(minus, mode="rms_v1", target_rms=0.5), upstream))
        ) / (2.0 * epsilon)
    assert np.allclose(vjp, numeric, atol=2.0e-5)


def test_chop_index_auto_scaling_dynamic_variance_and_legacy_compatibility() -> None:
    """Verify dynamic variance across unit inputs and backward compatibility for legacy percentages."""
    unit_inputs = [0.0, 0.25, 0.5, 0.75, 1.0]
    expected_outputs = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for mode in (INPUT_NORMALIZATION_GROUPWISE_V2, INPUT_NORMALIZATION_GROUPWISE_MTF_V3):
        normalized_outputs = []
        for raw_val, expected_val in zip(unit_inputs, expected_outputs):
            vec = np.zeros(INPUT_COUNT, dtype=float)
            vec[93] = raw_val
            norm = normalize_input_vector(vec, mode=mode)
            val = float(norm[93])
            normalized_outputs.append(val)
            assert np.isclose(val, expected_val, atol=1e-6), (
                f"Mode {mode} failed for chop input {raw_val}: got {val}, expected {expected_val}"
            )
            # Ensure non-zero inputs do not collapse near -0.99
            if raw_val >= 0.25:
                assert val > -0.99, f"Mode {mode} collapsed chop value {raw_val} to {val} (near -0.99)"

        # Verify dynamic variance across unit interval inputs
        variance = float(np.var(normalized_outputs))
        assert variance >= 0.45, f"Mode {mode} dynamic variance too low: {variance}"

    # Verify backward compatibility for legacy inputs [0, 100]
    legacy_cases = [
        (0.0, -1.0),
        (25.0, -0.5),
        (50.0, 0.0),
        (75.0, 0.5),
        (100.0, 1.0),
    ]
    for mode in (INPUT_NORMALIZATION_GROUPWISE_V2, INPUT_NORMALIZATION_GROUPWISE_MTF_V3):
        for raw_legacy, expected_val in legacy_cases:
            vec = np.zeros(INPUT_COUNT, dtype=float)
            vec[93] = raw_legacy
            norm = normalize_input_vector(vec, mode=mode)
            val = float(norm[93])
            assert np.isclose(val, expected_val, atol=1e-6), (
                f"Mode {mode} failed for legacy chop input {raw_legacy}: got {val}, expected {expected_val}"
            )

