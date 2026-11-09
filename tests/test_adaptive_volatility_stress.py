# ==============================================================================
# tests/test_adaptive_volatility_stress.py
#
# EMPIRICAL ADVERSARIAL STRESS TEST SUITE: Milestone 2 Adaptive Volatility Targets
#
# Exhaustively stress-tests:
# 1. Malformed / extreme / NaN / Inf observation features.
# 2. Truncated observation vectors across shapes 1..136.
# 3. Extreme boundary transitions with epsilon jitter across all regime boundaries.
# 4. pack_geometry_choices, unpack_geometry_choices, and MQL5 DecodeEntryGeometry
#    under extreme floats, infinities, out-of-bounds indices, and float32 rounding.
# 5. Governor HOLD lock and Zero Premature Close invariants under adversarial noise.
# 6. M1 spread cost ratio safety bounds.
# ==============================================================================

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest
import torch

from self_growing_trader.config import OrderConfig, RiskConfig
from self_growing_trader.execution_contract import (
    DEFAULT_SL_BIN_INDEX,
    DEFAULT_TP_BIN_INDEX,
    DEFAULT_TRAILING_BIN_INDEX,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_VALUE,
    FIXED_GEOMETRY_MODE,
    GEOMETRY_PACKED_BIN_COUNT,
    GEOMETRY_PACKED_OUTPUT_INDEX,
    LEARNED_GEOMETRY_MARKER_THRESHOLD,
    LEARNED_GEOMETRY_MODE,
    LEGACY_GEOMETRY_PACKED_BIN_COUNT,
    LEGACY_LEARNED_GEOMETRY_MARKER_THRESHOLD,
    REGIME_ADAPTIVE_GEOMETRY,
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    classify_regime_for_geometry,
    decode_execution_geometry,
    get_regime_adaptive_geometry,
    pack_geometry_choices,
    resolve_adaptive_geometry_indices,
    unpack_geometry_choices,
)
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy
from self_growing_trader.orders import Action, DecodedDecision, decode_outputs
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


# ------------------------------------------------------------------------------
# MQL5 Exact Replicas for Parity Testing
# ------------------------------------------------------------------------------

def mql5_sg_sigmoid(x: float) -> float:
    """Exact replica of double SgSigmoid(double x) from MQL5."""
    if math.isnan(x):
        return float("nan")
    if x > 60.0:
        return 1.0
    if x < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def mql5_sl_atr_from_bin(index: int) -> float:
    """Exact replica of SlAtrFromBin(int index) from MQL5."""
    if index <= 0: return 0.75
    if index == 1: return 1.0
    if index == 2: return 1.5
    if index == 3: return 2.0
    if index == 4: return 2.5
    if index == 5: return 3.0
    if index == 6: return 4.0
    return 5.0


def mql5_tp_rr_from_bin(index: int) -> float:
    """Exact replica of TpRrFromBin(int index) from MQL5."""
    if index <= 0: return 0.25
    if index == 1: return 0.50
    if index == 2: return 0.75
    if index == 3: return 1.0
    if index == 4: return 1.5
    if index == 5: return 2.0
    if index == 6: return 2.5
    if index == 7: return 3.0
    if index == 8: return 4.0
    return 5.0


def mql5_trailing_atr_from_bin(index: int) -> float:
    """Exact replica of TrailingAtrFromBin(int index) from MQL5."""
    if index <= 0: return 0.0
    if index == 1: return 0.75
    if index == 2: return 1.0
    if index == 3: return 1.5
    if index == 4: return 2.0
    if index == 5: return 2.5
    if index == 6: return 3.0
    return 4.0


def mql5_decode_entry_geometry(
    outputs: np.ndarray,
    fixed_sl_atr: float = 5.0,
    fixed_tp_atr: float = 10.0,
) -> tuple[float, float, float]:
    """Exact replica of void DecodeEntryGeometry(...) from SelfGrowing_MT5_Trainer_EA.mq5."""
    values = np.asarray(outputs, dtype=np.float64).reshape(-1)
    sl_atr = float(fixed_sl_atr)
    tp_atr = float(fixed_tp_atr)
    trailing_atr = 0.0

    if values.size <= GEOMETRY_PACKED_OUTPUT_INDEX:
        return sl_atr, tp_atr, trailing_atr

    marker = values[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX]
    # In MQL5: if(marker < SG_LEGACY_LEARNED_GEOMETRY_MARKER_THRESHOLD) return;
    # Note: in IEEE 754, if marker is NaN, (NaN < 5.5) evaluates to False in C/C++/MQL5!
    if not math.isnan(marker) and marker < LEGACY_LEARNED_GEOMETRY_MARKER_THRESHOLD:
        return sl_atr, tp_atr, trailing_atr
    if math.isnan(marker):
        # MQL5 evaluates NaN < 5.5 as false, so it falls through. But let's track standard behavior.
        pass

    legacy = (not math.isnan(marker)) and (marker < LEARNED_GEOMETRY_MARKER_THRESHOLD)
    packed_bin_count = LEGACY_GEOMETRY_PACKED_BIN_COUNT if legacy else GEOMETRY_PACKED_BIN_COUNT
    sl_bin_count = 7 if legacy else 8
    tp_bin_count = 5 if legacy else 10

    raw_val = values[GEOMETRY_PACKED_OUTPUT_INDEX]
    sig_val = mql5_sg_sigmoid(raw_val)
    if math.isnan(sig_val):
        packed = 0
    else:
        packed = int(math.floor(sig_val * float(packed_bin_count)))

    packed = max(0, min(packed_bin_count - 1, packed))
    sl_index = packed % sl_bin_count
    packed //= sl_bin_count
    tp_index = packed % tp_bin_count
    trailing_index = packed // tp_bin_count

    sl_atr = mql5_sl_atr_from_bin(sl_index)
    tp_atr = sl_atr * mql5_tp_rr_from_bin(tp_index)
    trailing_atr = mql5_trailing_atr_from_bin(trailing_index)
    return sl_atr, tp_atr, trailing_atr


# ==============================================================================
# 1. Malformed / Extreme / NaN Observation Features Stress Test
# ==============================================================================

class TestMalformedObservationStress:
    """Stress tests classifier and dynamic geometry resolution against adversarial feature values."""

    @pytest.mark.parametrize("nan_val", [float("nan"), np.nan])
    def test_all_nan_features_fallback(self, nan_val: float) -> None:
        """When all input features are NaN, classifier must not crash and must resolve to a valid regime."""
        regime = classify_regime_for_geometry(
            chop_index=nan_val,
            m15_direction=nan_val,
            trend_strength=nan_val,
            volatility=nan_val,
        )
        assert regime in REGIME_ADAPTIVE_GEOMETRY
        # Under NaN comparisons, all conditions evaluate False, falling through to WEAK_TREND
        assert regime == "WEAK_TREND"

        sl, tp, tr, reg = resolve_adaptive_geometry_indices(
            chop_index=nan_val,
            m15_direction=nan_val,
            trend_strength=nan_val,
            volatility=nan_val,
        )
        assert reg == "WEAK_TREND"
        assert sl == 7 and tp == 4 and tr == 0
        assert SL_ATR_BINS[sl] == 5.0
        assert TP_RR_BINS[tp] == 1.5

    @pytest.mark.parametrize("inf_val", [float("inf"), float("-inf"), 1e35, -1e35])
    def test_extreme_infinities_and_magnitudes(self, inf_val: float) -> None:
        """Extreme infinite and huge values must resolve deterministically without numerical overflow."""
        # Extreme volatility chop
        regime = classify_regime_for_geometry(
            chop_index=0.5,
            m15_direction=0.0,
            trend_strength=0.0,
            volatility=inf_val if inf_val > 0 else 0.0,
        )
        if inf_val > 0:
            assert regime == "VOLATILE"

        # Extreme positive direction -> Strong trend
        regime_dir_pos = classify_regime_for_geometry(
            chop_index=0.5,
            m15_direction=1e30,
            trend_strength=1e30,
            volatility=0.5,
        )
        assert regime_dir_pos == "STRONG_TREND"

        # Extreme negative direction -> Strong trend
        regime_dir_neg = classify_regime_for_geometry(
            chop_index=0.5,
            m15_direction=-1e30,
            trend_strength=1e30,
            volatility=0.5,
        )
        assert regime_dir_neg == "STRONG_TREND"

        # Extreme chop -> Ranging
        regime_chop = classify_regime_for_geometry(
            chop_index=1e30,
            m15_direction=0.0,
            trend_strength=0.0,
            volatility=0.5,
        )
        assert regime_chop == "RANGING"

    def test_randomized_adversarial_feature_fuzzing(self) -> None:
        """Fuzzes 2,000 randomized tuples including NaNs, Infs, denormals, negatives, and huge floats."""
        rng = np.random.default_rng(1337)
        extreme_pool = [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -0.0,
            1e-30,
            -1e-30,
            1e30,
            -1e30,
            0.199999999999999,
            0.200000000000001,
            0.449999999999999,
            0.450000000000001,
            0.649999999999999,
            0.650000000000001,
            0.849999999999999,
            0.850000000000001,
        ]

        for _ in range(2000):
            chop = rng.choice(extreme_pool) if rng.random() < 0.3 else float(rng.uniform(-2.0, 2.0))
            m15_dir = rng.choice(extreme_pool) if rng.random() < 0.3 else float(rng.uniform(-2.0, 2.0))
            vol = rng.choice(extreme_pool) if rng.random() < 0.3 else float(rng.uniform(-2.0, 2.0))
            strength = abs(m15_dir)

            sl, tp, tr, reg_name = resolve_adaptive_geometry_indices(
                chop_index=chop,
                m15_direction=m15_dir,
                trend_strength=strength,
                volatility=vol,
            )
            assert reg_name in ("VOLATILE", "RANGING", "STRONG_TREND", "WEAK_TREND")
            assert 0 <= sl < len(SL_ATR_BINS)
            assert 0 <= tp < len(TP_RR_BINS)
            assert 0 <= tr < len(TRAILING_ATR_BINS)
            assert sl == 7  # Strict M1 spread safety: SL ATR must always be 5.0 ATR
            assert SL_ATR_BINS[sl] == 5.0
            assert tr == 0  # Trailing ATR is 0.0

            # Verify packing is always safe and finite
            packed = pack_geometry_choices(sl, tp, tr)
            assert math.isfinite(packed)
            assert -7.2 < packed < 7.2


# ==============================================================================
# 2. Truncated Observation Vector Shapes (1..136) Stress Test
# ==============================================================================

class TestTruncatedObservationStress:
    """Stress tests behavior when observations have truncated shapes or missing features."""

    def test_policy_feature_extraction_under_all_truncated_shapes(self) -> None:
        """Simulates the dual-safe resolution logic across all shapes 1..136.

        Verifies that no IndexError is ever raised regardless of whether shape[-1]
        is < 29, 36, 44, 82, 84, 93, 137.
        """
        rng = np.random.default_rng(42)

        for width in range(1, 140):
            obs_slice = rng.standard_normal((5, width)).astype(np.float32)

            for i in range(5):
                chop_val = 0.5
                m15_dir = 0.0
                vol_val = 0.5

                if obs_slice.shape[-1] > 93 and (
                    abs(float(obs_slice[i, 93])) > 1e-5
                    or abs(float(obs_slice[i, 84])) > 1e-5
                    or abs(float(obs_slice[i, 82])) > 1e-5
                ):
                    chop_val = float(obs_slice[i, 93])
                    m15_dir = float(obs_slice[i, 84])
                    vol_val = float(obs_slice[i, 82])
                elif obs_slice.shape[-1] > 44 and (
                    abs(float(obs_slice[i, 44])) > 1e-5
                    or (obs_slice.shape[-1] > 36 and abs(float(obs_slice[i, 36])) > 1e-5)
                    or (obs_slice.shape[-1] > 29 and abs(float(obs_slice[i, 29])) > 1e-5)
                ):
                    chop_val = float(obs_slice[i, 44])
                    m15_dir = float(obs_slice[i, 36]) if obs_slice.shape[-1] > 36 else 0.0
                    vol_val = float(obs_slice[i, 29]) if obs_slice.shape[-1] > 29 else 0.5
                elif obs_slice.shape[-1] > 93:
                    chop_val = float(obs_slice[i, 93])
                    m15_dir = float(obs_slice[i, 84])
                    vol_val = float(obs_slice[i, 82])

                # Must resolve cleanly without error
                sl, tp, tr, regime = resolve_adaptive_geometry_indices(
                    chop_index=chop_val,
                    m15_direction=m15_dir,
                    trend_strength=abs(m15_dir),
                    volatility=vol_val,
                )
                assert regime in REGIME_ADAPTIVE_GEOMETRY
                assert sl == 7

    def test_forward_five_rejects_non_137_shapes_safely(self) -> None:
        """Verifies forward_five enforces strict 137-dim matrix contract with clear ValueError."""
        network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
        network.eval()
        network.heads.seven_choice_fixed_geometry_enabled = True
        network.heads.adaptive_volatility_targets_enabled = True

        for invalid_width in [1, 20, 58, 93, 100, 136, 138, 200]:
            bad_obs = torch.zeros((5, invalid_width), dtype=torch.float32)
            with pytest.raises(ValueError, match="expected observation matrix"):
                network.forward_five(bad_obs, network.initial_memories())


# ==============================================================================
# 3. Extreme Boundary Transitions Across Regimes
# ==============================================================================

class TestExtremeBoundaryTransitions:
    """Stress tests infinitesimally close values around each regime boundary."""

    EPS = 1e-9

    def test_boundary_volatility_threshold(self) -> None:
        """Threshold: vol >= 0.85 and abs_dir < 0.20 -> VOLATILE else RANGING."""
        # Just below 0.85 -> RANGING
        r_below = classify_regime_for_geometry(chop_index=0.5, m15_direction=0.10, volatility=0.85 - self.EPS)
        assert r_below == "RANGING"

        # Exactly 0.85 -> VOLATILE
        r_exact = classify_regime_for_geometry(chop_index=0.5, m15_direction=0.10, volatility=0.85)
        assert r_exact == "VOLATILE"

        # Just above 0.85 -> VOLATILE
        r_above = classify_regime_for_geometry(chop_index=0.5, m15_direction=0.10, volatility=0.85 + self.EPS)
        assert r_above == "VOLATILE"

    def test_boundary_volatile_vs_weak_trend_direction(self) -> None:
        """Threshold: at vol = 0.90, abs_dir < 0.20 is VOLATILE, abs_dir >= 0.20 escapes VOLATILE."""
        # abs_dir just below 0.20 with chop=0.50
        r_below = classify_regime_for_geometry(chop_index=0.50, m15_direction=0.20 - self.EPS, volatility=0.90)
        assert r_below == "VOLATILE"

        # abs_dir exactly 0.20 with chop=0.50, strength=0.20:
        # escapes volatile (abs_dir < 0.20 False), escapes ranging (chop < 0.65, abs_dir not < 0.20, strength not < 0.30 if >= 0.30)
        r_exact = classify_regime_for_geometry(chop_index=0.50, m15_direction=0.20, trend_strength=0.35, volatility=0.90)
        assert r_exact == "WEAK_TREND"

    def test_boundary_ranging_chop_threshold(self) -> None:
        """Threshold: chop >= 0.65 -> RANGING."""
        # Just below 0.65 with moderate direction
        r_below = classify_regime_for_geometry(chop_index=0.65 - self.EPS, m15_direction=0.30, trend_strength=0.30)
        assert r_below == "WEAK_TREND"

        # Exactly 0.65 -> RANGING
        r_exact = classify_regime_for_geometry(chop_index=0.65, m15_direction=0.30, trend_strength=0.30)
        assert r_exact == "RANGING"

        # Just above 0.65 -> RANGING
        r_above = classify_regime_for_geometry(chop_index=0.65 + self.EPS, m15_direction=0.30, trend_strength=0.30)
        assert r_above == "RANGING"

    def test_boundary_strong_trend_direction_threshold(self) -> None:
        """Threshold: abs_dir >= 0.45 -> STRONG_TREND."""
        # Just below 0.45 with low chop
        r_below = classify_regime_for_geometry(chop_index=0.40, m15_direction=0.45 - self.EPS, trend_strength=0.40)
        assert r_below == "WEAK_TREND"

        # Exactly 0.45 -> STRONG_TREND
        r_exact = classify_regime_for_geometry(chop_index=0.40, m15_direction=0.45, trend_strength=0.40)
        assert r_exact == "STRONG_TREND"

        # Just above 0.45 -> STRONG_TREND
        r_above = classify_regime_for_geometry(chop_index=0.40, m15_direction=0.45 + self.EPS, trend_strength=0.40)
        assert r_above == "STRONG_TREND"

    def test_boundary_strong_trend_strength_threshold(self) -> None:
        """Threshold: strength >= 0.70 -> STRONG_TREND."""
        # Just below 0.70 with abs_dir = 0.30
        r_below = classify_regime_for_geometry(chop_index=0.40, m15_direction=0.30, trend_strength=0.70 - self.EPS)
        assert r_below == "WEAK_TREND"

        # Exactly 0.70 -> STRONG_TREND
        r_exact = classify_regime_for_geometry(chop_index=0.40, m15_direction=0.30, trend_strength=0.70)
        assert r_exact == "STRONG_TREND"

    def test_rapid_alternating_oscillations(self) -> None:
        """Simulates 1,000 rapid back-and-forth switches between regimes."""
        for step in range(1000):
            if step % 4 == 0:
                chop, dir_val, vol, exp = 0.75, 0.05, 0.5, "RANGING"
            elif step % 4 == 1:
                chop, dir_val, vol, exp = 0.30, 0.60, 0.5, "STRONG_TREND"
            elif step % 4 == 2:
                chop, dir_val, vol, exp = 0.35, 0.30, 0.5, "WEAK_TREND"
            else:
                chop, dir_val, vol, exp = 0.70, 0.05, 0.95, "VOLATILE"

            sl, tp, tr, reg = resolve_adaptive_geometry_indices(
                chop_index=chop,
                m15_direction=dir_val,
                trend_strength=abs(dir_val),
                volatility=vol,
            )
            assert reg == exp
            assert sl == 7
            assert tr == 0


# ==============================================================================
# 4. Geometry Packing & MQL5 Decoding Adversarial Stress
# ==============================================================================

class TestGeometryPackingAndMql5DecodingStress:
    """Stress tests codebook bijection, extreme floats, out-of-bounds indices, and MQL5 parity."""

    def test_exhaustive_640_bin_float32_bijectivity(self) -> None:
        """Every single combination of 8 SL x 10 TP x 8 Trailing = 640 combinations

        must roundtrip perfectly under float32 downcasting in both Python and MQL5 decoders.
        """
        for sl in range(len(SL_ATR_BINS)):
            for tp in range(len(TP_RR_BINS)):
                for tr in range(len(TRAILING_ATR_BINS)):
                    packed = pack_geometry_choices(sl, tp, tr)
                    # Simulate transmission over 32-bit floating point wire
                    packed_f32 = float(np.float32(packed))

                    # Python decode
                    u_sl, u_tp, u_tr = unpack_geometry_choices(packed_f32)
                    assert (u_sl, u_tp, u_tr) == (sl, tp, tr), f"Failed Python bijection for ({sl}, {tp}, {tr})"

                    # MQL5 decode
                    outputs = np.zeros(20, dtype=np.float64)
                    outputs[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] = FACTORIZED_ACTION_MARKER_VALUE  # 7.0
                    outputs[GEOMETRY_PACKED_OUTPUT_INDEX] = packed_f32
                    m_sl, m_tp, m_tr = mql5_decode_entry_geometry(outputs)

                    exp_sl = SL_ATR_BINS[sl]
                    exp_tp = exp_sl * TP_RR_BINS[tp]
                    exp_tr = TRAILING_ATR_BINS[tr]

                    assert m_sl == pytest.approx(exp_sl, rel=1e-5), f"MQL5 SL mismatch at ({sl}, {tp}, {tr})"
                    assert m_tp == pytest.approx(exp_tp, rel=1e-5), f"MQL5 TP mismatch at ({sl}, {tp}, {tr})"
                    assert m_tr == pytest.approx(exp_tr, rel=1e-5), f"MQL5 Trailing mismatch at ({sl}, {tp}, {tr})"

    @pytest.mark.parametrize("bad_sl, bad_tp, bad_tr", [
        (-1, 0, 0),
        (-999, -999, -999),
        (8, 0, 0),
        (0, 10, 0),
        (0, 0, 8),
        (999, 999, 999),
        (100, -50, 200),
    ])
    def test_pack_geometry_out_of_bounds_integer_clamping(self, bad_sl: int, bad_tp: int, bad_tr: int) -> None:
        """pack_geometry_choices must safely clamp out-of-bounds integer inputs without error."""
        packed = pack_geometry_choices(bad_sl, bad_tp, bad_tr)
        assert math.isfinite(packed)
        u_sl, u_tp, u_tr = unpack_geometry_choices(packed)
        assert 0 <= u_sl < len(SL_ATR_BINS)
        assert 0 <= u_tp < len(TP_RR_BINS)
        assert 0 <= u_tr < len(TRAILING_ATR_BINS)

    @pytest.mark.parametrize("extreme_logit", [
        -1e12, -1000.0, -100.0, -60.0, -60.0001, -59.9999,
        0.0, 59.9999, 60.0, 60.0001, 100.0, 1000.0, 1e12,
        float("inf"), float("-inf"),
    ])
    def test_unpack_and_mql5_under_extreme_logits(self, extreme_logit: float) -> None:
        """Unpacking extreme logits must never overflow or index out of bounds."""
        u_sl, u_tp, u_tr = unpack_geometry_choices(extreme_logit)
        assert 0 <= u_sl < len(SL_ATR_BINS)
        assert 0 <= u_tp < len(TP_RR_BINS)
        assert 0 <= u_tr < len(TRAILING_ATR_BINS)

        outputs = np.zeros(20, dtype=np.float64)
        outputs[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] = FACTORIZED_ACTION_MARKER_VALUE
        outputs[GEOMETRY_PACKED_OUTPUT_INDEX] = extreme_logit
        m_sl, m_tp, m_tr = mql5_decode_entry_geometry(outputs)
        assert m_sl in SL_ATR_BINS
        assert m_tr in TRAILING_ATR_BINS


# ==============================================================================
# 5. Governor HOLD Lock & Zero Premature Close Adversarial Invariant
# ==============================================================================

class TestGovernorHoldLockAdversarial:
    """Stress tests that Governor and Management agents are strictly locked to HOLD under adversarial conditions."""

    def test_governor_lock_under_extreme_adversarial_logits(self) -> None:
        """Even with noisy/extreme observations and unconstrained action masks,

        Agents 2, 3, 4 MUST be strictly forced to HOLD (action=0) when lock_non_entry_to_hold=True.
        """
        torch.manual_seed(999)
        network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
        network.eval()
        network.heads.seven_choice_fixed_geometry_enabled = True
        network.heads.lock_non_entry_to_hold = True
        network.heads.adaptive_volatility_targets_enabled = True

        for trial in range(50):
            # Create adversarial observation matrix with extreme values
            obs = torch.randn((5, INPUT_COUNT), dtype=torch.float32) * 50.0
            # Inject extreme chop / momentum / volatility
            obs[:, 93] = 0.99
            obs[:, 84] = 0.99
            obs[:, 82] = 0.99

            # Provide permissive mask allowing ALL actions (including Action 6: POLICY_CLOSE_ALL)
            allowed = torch.ones((5, 12), dtype=torch.bool)

            with torch.no_grad():
                results, _, _ = network.forward_five(
                    obs, network.initial_memories(), allowed_action_masks=allowed, deterministic=False
                )

            # Agents 2, 3, 4 MUST be Action 0 (HOLD)
            for agent_idx in (2, 3, 4):
                assert results[agent_idx].action_index == 0, f"Trial {trial}: Agent {agent_idx} emitted non-HOLD {results[agent_idx].action_index}"
                # Must never be Action.CLOSE_ALL (telemetry: POLICY_CLOSE_ALL)
                assert results[agent_idx].action_index != int(Action.CLOSE_ALL)


# ==============================================================================
# 6. M1 Spread Safety Preservation
# ==============================================================================

def test_spread_cost_safety_invariant_all_regimes() -> None:
    """Requirement: spread cost ratio on M1 must never exceed max_spread_cost_r (0.150)."""
    empirical_spread_atr = 0.649373
    max_allowed_spread_cost_r = 0.150

    for regime_name, cfg in REGIME_ADAPTIVE_GEOMETRY.items():
        sl_atr = float(cfg["sl_atr"])
        spread_ratio = empirical_spread_atr / sl_atr
        assert spread_ratio <= max_allowed_spread_cost_r, (
            f"Regime {regime_name} SL ATR {sl_atr} violates spread cost bound: {spread_ratio:.5f} > {max_allowed_spread_cost_r}"
        )
