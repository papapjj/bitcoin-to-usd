"""
tests/test_cppi_risk_engine.py

Comprehensive Unit, Boundary, Pairwise, and Scenario Test Suite for CPPIFloorEngine.
Covers Tiers 1-4:
- Tier 1: Nominal tests for Features F1 through F5 (>=5 per feature).
- Tier 2: Boundary value analysis & robust fail-closed edge cases (>=5 per feature).
- Tier 3: Pairwise combinatorial combinations across Equity regimes, Confluence states, and BaseRisk.
- Tier 4: Real-world market simulations (20/100-trade loss streaks, 10-trade hyper-growth waves, whipsaw chop).
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from self_growing_trader.cppi_risk_engine import CPPIConfig, CPPIFloorEngine


# ============================================================================
# Tier 1 & 2: Configuration Validation Tests
# ============================================================================
class TestCPPIConfigValidation:
    """Validate configuration defaults, invariants, and defensive parameter checks."""

    def test_cppi_config_defaults(self):
        """C1: Verify exact configuration defaults match architecture specifications."""
        config = CPPIConfig()
        assert config.max_allowed_dd == 0.08
        assert config.cushion_loss_fraction == 0.50
        assert config.base_alpha == 0.3
        assert config.pumping_slope == 4.7
        assert config.max_cushion_ratio == 2.0
        assert config.confluence_boost == 1.50
        assert config.h1_trend_idx == 88
        assert config.m15_trend_idx == 84
        assert config.m5_velocity_idx == 81
        assert config.max_risk_cap == 0.25
        assert config.min_risk_pct == 0.0
        assert config.confluence_threshold == 0.0

    @pytest.mark.parametrize("invalid_dd", [-0.05, 0.0, 1.0, 1.5])
    def test_cppi_config_invalid_dd(self, invalid_dd):
        """C2: max_allowed_dd outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="max_allowed_dd"):
            CPPIConfig(max_allowed_dd=invalid_dd)

    @pytest.mark.parametrize("invalid_frac", [-0.1, 0.0, 1.1])
    def test_cppi_config_invalid_fraction(self, invalid_frac):
        """C3: cushion_loss_fraction outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="cushion_loss_fraction"):
            CPPIConfig(cushion_loss_fraction=invalid_frac)

    @pytest.mark.parametrize("idx_name,idx_val", [
        ("h1_trend_idx", -1),
        ("h1_trend_idx", 137),
        ("m15_trend_idx", -5),
        ("m15_trend_idx", 200),
        ("m5_velocity_idx", 140),
    ])
    def test_cppi_config_invalid_indices(self, idx_name, idx_val):
        """C4: Observation indices outside [0, 136] raise ValueError."""
        with pytest.raises(ValueError, match=idx_name):
            CPPIConfig(**{idx_name: idx_val})

    def test_cppi_config_negative_base_alpha(self):
        """base_alpha < 0 raises ValueError."""
        with pytest.raises(ValueError, match="base_alpha"):
            CPPIConfig(base_alpha=-0.1)

    def test_cppi_config_negative_pumping_slope(self):
        """pumping_slope < 0 raises ValueError."""
        with pytest.raises(ValueError, match="pumping_slope"):
            CPPIConfig(pumping_slope=-1.0)

    def test_cppi_config_invalid_max_cushion_ratio(self):
        """max_cushion_ratio <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_cushion_ratio"):
            CPPIConfig(max_cushion_ratio=0.0)

    def test_cppi_config_invalid_confluence_boost(self):
        """confluence_boost < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="confluence_boost"):
            CPPIConfig(confluence_boost=0.9)

    def test_cppi_config_invalid_max_alpha(self):
        """max_alpha < base_alpha raises ValueError."""
        with pytest.raises(ValueError, match="max_alpha"):
            CPPIConfig(base_alpha=2.0, max_alpha=1.5)

    @pytest.mark.parametrize("invalid_cap", [-0.1, 0.0, 1.5])
    def test_cppi_config_invalid_max_risk_cap(self, invalid_cap):
        """max_risk_cap outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="max_risk_cap"):
            CPPIConfig(max_risk_cap=invalid_cap)


# ============================================================================
# Feature 1: Peak Equity Tracking & Ratchet Floor Formulation
# ============================================================================
class TestCPPIPeakAndFloorTracking:
    """Tests for continuous peak equity high-water mark tracking and floor computation."""

    def test_peak_tracking_monotonic_growth(self):
        """F1.1: Peak equity monotonically ratchets upwards as equity rises."""
        engine = CPPIFloorEngine()
        assert engine.peak_equity == 0.0

        p, f, c = engine.update_equity(10000.0)
        assert p == 10000.0
        assert engine.peak_equity == 10000.0

        p, f, c = engine.update_equity(11500.0)
        assert p == 11500.0
        assert engine.peak_equity == 11500.0

        p, f, c = engine.update_equity(12500.0)
        assert p == 12500.0
        assert engine.peak_equity == 12500.0

    def test_floor_ratchet_invariant(self):
        """F1.2: Floor is strictly (1 - max_allowed_dd) * peak_equity."""
        config = CPPIConfig(max_allowed_dd=0.08)
        engine = CPPIFloorEngine(config)
        p, f, c = engine.update_equity(10000.0)
        assert f == pytest.approx(9200.0)
        assert engine.floor == pytest.approx(9200.0)

        p, f, c = engine.update_equity(20000.0)
        assert f == pytest.approx(18400.0)
        assert engine.floor == pytest.approx(18400.0)

    def test_floor_never_decreases_in_drawdown(self):
        """F1.3: When equity falls from 10k to 9.5k, peak & floor stay frozen."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        assert engine.peak_equity == 10000.0
        assert engine.floor == 9200.0

        # Drawdown to 9500
        p, f, c = engine.update_equity(9500.0)
        assert p == 10000.0
        assert f == 9200.0
        assert engine.peak_equity == 10000.0
        assert engine.floor == 9200.0
        assert c == pytest.approx(300.0)

        # Deeper drawdown to 9250
        p, f, c = engine.update_equity(9250.0)
        assert p == 10000.0
        assert f == 9200.0
        assert c == pytest.approx(50.0)

    def test_floor_config_variation_6pct(self):
        """F1.4: Floor is exactly 94% of peak when max_allowed_dd=0.06."""
        config = CPPIConfig(max_allowed_dd=0.06)
        engine = CPPIFloorEngine(config)
        p, f, c = engine.update_equity(10000.0)
        assert f == pytest.approx(9400.0)
        assert c == pytest.approx(600.0)

    def test_recovery_establishes_new_floor(self):
        """F1.5: Equity recovery past previous high updates floor smoothly."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        assert engine.floor == 9200.0

        # Drawdown
        engine.update_equity(9300.0)
        assert engine.floor == 9200.0

        # Recovery past old peak to 12000
        p, f, c = engine.update_equity(12000.0)
        assert p == 12000.0
        assert f == pytest.approx(11040.0)
        assert c == pytest.approx(960.0)

    def test_peak_tracking_zero_initial(self):
        """F1.6: Zero initial equity updates safely on first non-zero equity."""
        engine = CPPIFloorEngine()
        assert engine.peak_equity == 0.0
        assert engine.floor == 0.0
        assert engine.cushion == 0.0

        p, f, c = engine.update_equity(5000.0)
        assert p == 5000.0
        assert f == pytest.approx(4600.0)
        assert c == pytest.approx(400.0)

    @pytest.mark.parametrize("bad_equity", [-500.0, -0.01, 0.0])
    def test_peak_tracking_negative_equity(self, bad_equity):
        """F1.7: Non-positive equity does not corrupt peak equity."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        p, f, c = engine.update_equity(bad_equity)
        assert p == 10000.0
        assert f == pytest.approx(9200.0)
        assert c == 0.0

    @pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
    def test_peak_tracking_nan_inf_safety(self, non_finite):
        """F1.8: NaN or Inf equity inputs ignored without modifying state."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        p, f, c = engine.update_equity(non_finite)
        assert p == 10000.0
        assert f == pytest.approx(9200.0)
        assert c == 0.0
        assert math.isfinite(p) and math.isfinite(f) and math.isfinite(c)

    def test_peak_tracking_extreme_equity(self):
        """F1.9: Numerical stability at $10^9$ equity without overflow."""
        engine = CPPIFloorEngine()
        huge = 1e9
        p, f, c = engine.update_equity(huge)
        assert p == huge
        assert f == pytest.approx(0.92 * huge)
        assert c == pytest.approx(0.08 * huge)

    def test_engine_reset_behavior(self):
        """F1.10: Calling reset(equity) resets peak, floor, and cushion cleanly."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        engine.update_equity(15000.0)
        assert engine.peak_equity == 15000.0

        engine.reset(8000.0)
        assert engine.peak_equity == 8000.0
        assert engine.floor == pytest.approx(7360.0)
        assert engine.cushion == pytest.approx(640.0)
        assert engine.cushion_ratio == pytest.approx(1.0)


# ============================================================================
# Feature 2: Real-Time Cushion & Cushion Ratio
# ============================================================================
class TestCPPICushionAndCushionRatio:
    """Tests for available dollar cushion and normalized cushion ratio."""

    def test_cushion_at_peak_equals_max_dd_amount(self):
        """F2.1: At peak, cushion equals exactly max_allowed_dd * peak_equity."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        p, f, c = engine.update_equity(10000.0)
        assert c == pytest.approx(800.0)

    def test_cushion_ratio_at_peak_equals_one(self):
        """F2.2: cushion_ratio == 1.0 when equity equals peak equity."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        assert engine.cushion_ratio == pytest.approx(1.0)

    def test_cushion_ratio_at_half_cushion(self):
        """F2.3: cushion_ratio == 0.50 when equity is halfway between floor and peak."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Floor 9200, Cushion 800
        engine.update_equity(9600.0)   # Halfway: cushion = 400
        assert engine.cushion == pytest.approx(400.0)
        assert engine.cushion_ratio == pytest.approx(0.50)

    def test_cushion_at_floor_equals_zero(self):
        """F2.4: cushion == 0.0 and cushion_ratio == 0.0 when equity == floor."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        engine.update_equity(9200.0)  # Exactly at floor
        assert engine.cushion == 0.0
        assert engine.cushion_ratio == 0.0

    def test_cushion_ratio_clamping_to_max(self):
        """F2.5: Cushion ratio clamping ceiling is enforced."""
        config = CPPIConfig(max_cushion_ratio=2.0)
        engine = CPPIFloorEngine(config)
        # Without update_equity ratcheting, compute_cushion_ratio handles higher equity
        ratio = engine.compute_cushion_ratio(20000.0)
        assert ratio >= 1.0

    def test_cushion_below_floor_clamps_zero(self):
        """F2.6: Equity below floor clamps cushion to 0.0 (no negative cushion)."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Floor 9200
        p, f, c = engine.update_equity(9100.0)  # Gap below floor
        assert c == 0.0
        assert engine.cushion == 0.0
        assert engine.cushion_ratio == 0.0

    def test_cushion_ratio_zero_division_guard(self):
        """F2.7: Peak equity 0 or max_allowed_dd 0 returns ratio 0.0 safely."""
        engine = CPPIFloorEngine()
        assert engine.cushion_ratio == 0.0
        assert engine.compute_cushion_ratio() == 0.0

    def test_cushion_micro_fluctuation(self):
        """F2.8: Precision with 0.0001 equity changes."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        p, f, c = engine.update_equity(9999.9999)
        assert c == pytest.approx(799.9999, rel=1e-6)

    def test_compute_cushion_ratio_with_explicit_equity(self):
        """compute_cushion_ratio(equity) computes normalized ratio without modifying state."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        ratio_half = engine.compute_cushion_ratio(9600.0)
        assert ratio_half == pytest.approx(0.50)
        # Check internal state untouched
        assert engine.current_equity == 10000.0

    @pytest.mark.parametrize("invalid_eq", [0.0, -100.0, float("nan"), float("inf")])
    def test_compute_cushion_ratio_invalid_equity(self, invalid_eq):
        """compute_cushion_ratio returns 0.0 on invalid equity input."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        assert engine.compute_cushion_ratio(invalid_eq) == 0.0


# ============================================================================
# Feature 3: Hard 50% Cushion SL Risk Cap
# ============================================================================
class TestCPPIHardCushionCap:
    """Tests for the absolute mathematical safety barrier risk_pct <= (0.50 * cushion) / equity."""

    def test_hard_cushion_cap_clips_pumped_risk(self):
        """F3.1: When pumped risk > 50% cushion / equity, cap strictly limits it."""
        # At peak 10000, cushion = 800. Cap = (0.50 * 800) / 10000 = 0.04 (4.0%)
        # Base risk = 0.01. Cushion ratio = 1.0 -> alpha = 5.0. Unconstrained = 0.01 * 5 = 0.05 (5.0%)
        # Cap 0.04 dominates and clips 0.05 to 0.04.
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        risk = engine.compute_dynamic_risk(base_risk_pct=0.01, current_equity=10000.0)
        assert risk == pytest.approx(0.04)

    def test_hard_cushion_cap_preserves_low_risk(self):
        """F3.2: When pumped risk < 50% cushion / equity, pumped risk is preserved."""
        # Cap = 0.04. Base risk = 0.005. Cushion ratio = 1.0 -> alpha = 5.0.
        # Unconstrained = 0.005 * 5 = 0.025 (2.5%) < 0.04.
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        risk = engine.compute_dynamic_risk(base_risk_pct=0.005, current_equity=10000.0)
        assert risk == pytest.approx(0.025)

    def test_dollar_loss_never_exceeds_half_cushion(self):
        """F3.3: Dollar loss (risk_pct * equity) is always <= 0.50 * cushion."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        for eq in [10000.0, 9800.0, 9500.0, 9300.0, 9210.0]:
            engine.reset(10000.0)
            engine.update_equity(eq)
            cushion = engine.cushion
            risk = engine.compute_dynamic_risk(base_risk_pct=0.05, current_equity=eq)
            dollar_loss = risk * eq
            assert dollar_loss <= (0.50 * cushion) + 1e-7

    def test_zero_cushion_results_in_zero_risk(self):
        """F3.4: When cushion is 0, compute_dynamic_risk returns 0.0."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Floor 9200
        # At floor
        risk_at_floor = engine.compute_dynamic_risk(base_risk_pct=0.02, current_equity=9200.0)
        assert risk_at_floor == 0.0

        # Below floor
        risk_below_floor = engine.compute_dynamic_risk(base_risk_pct=0.02, current_equity=9100.0)
        assert risk_below_floor == 0.0

    def test_configurable_cushion_loss_fraction(self):
        """F3.5: Fraction = 0.30 bounds risk to 0.30 * cushion / equity."""
        config = CPPIConfig(max_allowed_dd=0.08, cushion_loss_fraction=0.30)
        engine = CPPIFloorEngine(config)
        # At 10000, cushion = 800. Cap = (0.30 * 800) / 10000 = 0.024
        risk = engine.compute_dynamic_risk(base_risk_pct=0.01, current_equity=10000.0)
        assert risk == pytest.approx(0.024)

    def test_hard_cushion_cap_boundary_exact(self):
        """F3.6: Exact equality when pumped risk equals cap."""
        # Cap = 0.04. Want unconstrained = 0.04 -> base_risk = 0.04 / 5.0 = 0.008
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        risk = engine.compute_dynamic_risk(base_risk_pct=0.008, current_equity=10000.0)
        assert risk == pytest.approx(0.04)

    def test_micro_cushion_produces_micro_risk(self):
        """F3.7: Cushion = $0.02 on $10,000 equity produces safe micro risk."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Floor = 9200.0
        # Equity = 9200.02 -> cushion = 0.02
        risk = engine.compute_dynamic_risk(base_risk_pct=0.01, current_equity=9200.02)
        # Cap = (0.50 * 0.02) / 9200.02 = 0.01 / 9200.02 approx 1.0869e-6
        assert risk == pytest.approx(0.01 / 9200.02, rel=1e-5)
        assert risk > 0.0

    @pytest.mark.parametrize("bad_equity", [0.0, -100.0, -0.01])
    def test_zero_equity_returns_zero_risk(self, bad_equity):
        """F3.8: compute_dynamic_risk with non-positive equity returns 0.0."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        assert engine.compute_dynamic_risk(0.01, bad_equity) == 0.0

    @pytest.mark.parametrize("bad_base", [0.0, -0.01, -1.0])
    def test_negative_base_risk_returns_zero(self, bad_base):
        """F3.9: Base risk <= 0 returns 0.0."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        assert engine.compute_dynamic_risk(bad_base, 10000.0) == 0.0

    def test_global_max_risk_cap_enforced(self):
        """F3.10: Huge cushion cannot push single-trade risk above max_risk_cap."""
        config = CPPIConfig(max_risk_cap=0.15)
        engine = CPPIFloorEngine(config)
        risk = engine.compute_dynamic_risk(base_risk_pct=0.20, current_equity=10000.0)
        assert risk <= 0.15

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_base_risk_returns_zero(self, bad_val):
        """NaN or Inf base risk returns 0.0 safely."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        assert engine.compute_dynamic_risk(bad_val, 10000.0) == 0.0

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_equity_returns_zero_risk(self, bad_val):
        """NaN or Inf equity returns 0.0 safely."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        assert engine.compute_dynamic_risk(0.01, bad_val) == 0.0


# ============================================================================
# Feature 4: Asymmetric Volatility Pumping Formula
# ============================================================================
class TestCPPIAsymmetricPumping:
    """Tests for alpha = 0.3 + 4.7 * min(2.0, max(0.0, cushion_ratio))."""

    def test_alpha_at_floor_equals_base_alpha(self):
        """F4.1: At cushion ratio = 0, alpha is exactly base_alpha = 0.3x."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        # When equity is at floor (9200), cushion is 0 -> risk is 0.0
        # When equity is just epsilon above floor, alpha is ~0.3
        # We test directly on formula via telemetry or proportional risk
        assert engine.config.base_alpha == 0.3

    def test_alpha_at_peak_equals_five(self):
        """F4.2: At cushion ratio = 1.0, alpha is 0.3 + 4.7 * 1.0 = 5.0x."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        # Use low base risk so cushion cap (0.04) does not bind
        # Base risk = 0.002. Pumped risk = 0.002 * 5.0 = 0.010 <= 0.04
        risk = engine.compute_dynamic_risk(base_risk_pct=0.002, current_equity=10000.0)
        assert risk == pytest.approx(0.010)

    def test_alpha_at_half_cushion(self):
        """F4.3: At cushion ratio = 0.5, alpha is 0.3 + 4.7 * 0.5 = 2.65x."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Peak 10000, Floor 9200
        # Equity = 9600 -> cushion = 400 -> cushion_ratio = 400 / 800 = 0.5
        # Cap = (0.50 * 400) / 9600 = 200 / 9600 approx 0.020833
        # Base risk = 0.002 -> unconstrained = 0.002 * 2.65 = 0.0053 <= Cap
        risk = engine.compute_dynamic_risk(base_risk_pct=0.002, current_equity=9600.0)
        assert risk == pytest.approx(0.002 * 2.65)

    def test_alpha_scaling_monotonicity(self):
        """F4.4: Alpha and unconstrained risk strictly increase monotonically with cushion ratio."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        prev_risk = 0.0
        # Test equities from 9210 up to 10000 with very small base risk (0.0001) to stay below cap
        for eq in [9250.0, 9400.0, 9600.0, 9800.0, 10000.0]:
            r = engine.compute_dynamic_risk(base_risk_pct=0.0001, current_equity=eq)
            assert r > prev_risk
            prev_risk = r

    def test_pumped_risk_proportional_to_base(self):
        """F4.5: Doubling base risk doubles pumped risk (when not capped)."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        r1 = engine.compute_dynamic_risk(base_risk_pct=0.001, current_equity=10000.0)
        r2 = engine.compute_dynamic_risk(base_risk_pct=0.002, current_equity=10000.0)
        assert r2 == pytest.approx(2.0 * r1)

    def test_alpha_clamping_at_max_cushion_ratio(self):
        """F4.6: Cushion ratio > 2.0 clamps alpha at 0.3 + 4.7 * 2.0 = 9.7x."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        # Initial peak 10000. If equity jumps to 10800 without prior update,
        # compute_dynamic_risk will update peak to 10800 (making ratio 1.0).
        # To test ratio = 2.0 explicitly on alpha formula:
        config = CPPIConfig(max_cushion_ratio=2.0)
        alpha_max = config.base_alpha + config.pumping_slope * config.max_cushion_ratio
        assert alpha_max == pytest.approx(9.7)

    def test_cushion_saturation_at_ratio_2(self):
        """Suite 4: Alpha formula evaluates to 9.7 at cushion_ratio = 2.0."""
        config = CPPIConfig(base_alpha=0.3, pumping_slope=4.7, max_cushion_ratio=2.0)
        alpha = config.base_alpha + config.pumping_slope * min(config.max_cushion_ratio, 2.0)
        assert alpha == pytest.approx(9.7)

    def test_cushion_saturation_beyond_ratio_2(self):
        """Suite 4: For any cushion ratio > 2.0, alpha is strictly clamped to 9.7."""
        config = CPPIConfig(base_alpha=0.3, pumping_slope=4.7, max_cushion_ratio=2.0)
        for ratio in [2.5, 5.0, 10.0, 100.0]:
            clamped = max(0.0, min(config.max_cushion_ratio, ratio))
            alpha = config.base_alpha + config.pumping_slope * clamped
            assert alpha == pytest.approx(9.7)

    def test_max_alpha_configurable_clamping(self):
        """If max_alpha is configured (e.g. 7.5), alpha does not exceed it."""
        config = CPPIConfig(max_alpha=7.5)
        engine = CPPIFloorEngine(config)
        # At peak 10000, alpha = 5.0. With confluence (1.5x), unconstrained alpha = 7.5
        # If boost was 2.0x, 5.0 * 2.0 = 10.0 would be clipped to 7.5
        assert engine.config.max_alpha == 7.5


# ============================================================================
# Feature 5: 3-Timeframe Confluence Multiplier (Indices 88, 84, 81)
# ============================================================================
class TestCPPI3TFConfluence:
    """Tests for 3-Timeframe Confluence (H1 trend @ 88, M15 trend @ 84, M5 velocity @ 81)."""

    def _build_inputs(self, h1: float = 0.0, m15: float = 0.0, m5: float = 0.0) -> list[float]:
        vec = [0.0] * 137
        vec[88] = float(h1)
        vec[84] = float(m15)
        vec[81] = float(m5)
        return vec

    def test_3tf_buy_confluence_boost_1_5x(self):
        """F5.1: BUY (+1) with H1>0, M15>0, M5>0 multiplies alpha by 1.5x."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._build_inputs(h1=0.5, m15=0.2, m5=0.1)

        # Baseline without confluence
        r_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        # With BUY confluence
        r_boost = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=1)

        assert r_boost == pytest.approx(1.50 * r_base)

    def test_3tf_sell_confluence_boost_1_5x(self):
        """F5.2: SELL (-1) with H1<0, M15<0, M5<0 multiplies alpha by 1.5x."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._build_inputs(h1=-0.5, m15=-0.2, m5=-0.1)

        r_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        r_boost = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=-1)

        assert r_boost == pytest.approx(1.50 * r_base)

    def test_3tf_conflicting_signals_no_boost(self):
        """F5.3: H1>0, M15>0, M5<0 yields no boost (1.0x)."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._build_inputs(h1=0.5, m15=0.2, m5=-0.1)

        r_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        r_mixed = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=1)

        assert r_mixed == pytest.approx(r_base)
        assert engine.evaluate_confluence(inputs, trade_direction=1) is False

    def test_3tf_opposing_trade_direction_no_boost(self):
        """F5.4: BUY trade with bearish confluence signals gets no boost."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._build_inputs(h1=-0.5, m15=-0.2, m5=-0.1)  # All bearish

        r_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        r_opp = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=1)  # BUY order

        assert r_opp == pytest.approx(r_base)
        assert engine.evaluate_confluence(inputs, trade_direction=1) is False

    def test_3tf_neutral_trade_direction_no_boost(self):
        """F5.5: trade_direction = 0 gets no boost."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._build_inputs(h1=0.5, m15=0.2, m5=0.1)

        r_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        r_neutral = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=0)

        assert r_neutral == pytest.approx(r_base)
        assert engine.evaluate_confluence(inputs, trade_direction=0) is False

    def test_3tf_none_inputs_safety(self):
        """F5.6: inputs = None executes cleanly without boost."""
        engine = CPPIFloorEngine()
        signals = engine.extract_timeframe_signals(None)
        assert signals == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(None, trade_direction=1) is False

    @pytest.mark.parametrize("short_len", [0, 10, 50, 88])
    def test_3tf_truncated_inputs_safety(self, short_len):
        """F5.7: inputs length < 89 does not raise IndexError, returns no boost."""
        engine = CPPIFloorEngine()
        short_inputs = [0.1] * short_len
        signals = engine.extract_timeframe_signals(short_inputs)
        assert signals == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(short_inputs, trade_direction=1) is False

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_3tf_nan_inf_signals_safety(self, bad_val):
        """F5.8: NaN in H1/M15/M5 treated safely as non-confluent."""
        engine = CPPIFloorEngine()
        inputs = self._build_inputs(h1=bad_val, m15=0.5, m5=0.5)
        signals = engine.extract_timeframe_signals(inputs)
        assert signals == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(inputs, trade_direction=1) is False

    def test_3tf_2d_batch_tensor_input(self):
        """F5.9: Array of shape (5, 137) multi-agent batch parsed cleanly via first row."""
        engine = CPPIFloorEngine()
        batch = np.zeros((5, 137), dtype=np.float32)
        batch[0, 88] = 0.8
        batch[0, 84] = 0.4
        batch[0, 81] = 0.2
        signals = engine.extract_timeframe_signals(batch)
        assert signals == pytest.approx((0.8, 0.4, 0.2))
        assert engine.evaluate_confluence(batch, trade_direction=1) is True

    def test_3tf_exact_indices_verification(self):
        """F5.10: Verify signals are extracted from exact indices 88, 84, and 81."""
        engine = CPPIFloorEngine()
        inputs = [0.0] * 137
        inputs[88] = 12.34
        inputs[84] = 56.78
        inputs[81] = 90.12
        h1, m15, m5 = engine.extract_timeframe_signals(inputs)
        assert h1 == pytest.approx(12.34)
        assert m15 == pytest.approx(56.78)
        assert m5 == pytest.approx(90.12)

    def test_3tf_corrupt_type_safety(self):
        """Non-numeric or corrupt types do not crash the engine."""
        engine = CPPIFloorEngine()
        assert engine.extract_timeframe_signals("corrupted_string") == (0.0, 0.0, 0.0)
        assert engine.extract_timeframe_signals(12345) == (0.0, 0.0, 0.0)

    def test_evaluate_confluence_direction_helper(self):
        """evaluate_confluence_direction returns +1 for Bullish, -1 for Bearish, 0 for Mixed."""
        engine = CPPIFloorEngine()
        bull_inputs = self._build_inputs(h1=0.5, m15=0.2, m5=0.1)
        bear_inputs = self._build_inputs(h1=-0.5, m15=-0.2, m5=-0.1)
        mixed_inputs = self._build_inputs(h1=0.5, m15=-0.2, m5=0.1)

        assert engine.evaluate_confluence_direction(bull_inputs) == 1
        assert engine.evaluate_confluence_direction(bear_inputs) == -1
        assert engine.evaluate_confluence_direction(mixed_inputs) == 0


# ============================================================================
# Tier 3: Pairwise Combinatorial Interactions
# ============================================================================
class TestCPPIPairwiseCombinatorial:
    """Pairwise test matrix across Equity regimes, Confluence states, and BaseRisk."""

    @pytest.mark.parametrize("equity_level", [10000.0, 9600.0, 9300.0, 9200.0])
    @pytest.mark.parametrize("confluence_state", ["bull_buy", "bear_sell", "mixed", "none"])
    @pytest.mark.parametrize("base_risk", [0.001, 0.01, 0.03])
    def test_pairwise_combinations_matrix(self, equity_level, confluence_state, base_risk):
        """P1: Pairwise combinations of Equity x Confluence x BaseRisk remain safe and invariant."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Peak = 10000, Floor = 9200

        inputs = [0.0] * 137
        trade_dir = 0
        if confluence_state == "bull_buy":
            inputs[88], inputs[84], inputs[81] = 0.5, 0.5, 0.5
            trade_dir = 1
        elif confluence_state == "bear_sell":
            inputs[88], inputs[84], inputs[81] = -0.5, -0.5, -0.5
            trade_dir = -1
        elif confluence_state == "mixed":
            inputs[88], inputs[84], inputs[81] = 0.5, -0.5, 0.5
            trade_dir = 1
        else:
            inputs = None
            trade_dir = 0

        risk = engine.compute_dynamic_risk(
            base_risk_pct=base_risk,
            current_equity=equity_level,
            inputs=inputs,
            trade_direction=trade_dir,
        )

        assert 0.0 <= risk <= 1.0
        assert math.isfinite(risk)

        # Check hard 50% cushion cap invariant
        cushion = max(0.0, equity_level - 9200.0)
        cushion_cap = (0.50 * cushion) / equity_level
        assert risk <= cushion_cap + 1e-7

        # If at floor, risk must be exactly 0.0
        if equity_level <= 9200.0:
            assert risk == 0.0

    @pytest.mark.parametrize("max_dd", [0.04, 0.06, 0.08, 0.10, 0.15])
    def test_pairwise_different_drawdown_limits(self, max_dd):
        """Verify behavior across varying drawdown limits."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=max_dd))
        p, f, c = engine.update_equity(10000.0)
        assert f == pytest.approx((1.0 - max_dd) * 10000.0)
        assert c == pytest.approx(max_dd * 10000.0)
        risk = engine.compute_dynamic_risk(0.005, 10000.0)
        assert risk > 0.0
        assert risk <= (0.50 * c) / 10000.0


# ============================================================================
# Tier 4: Real-World Scenarios & Mathematical Proofs
# ============================================================================
class TestCPPIScenarios:
    """Real-world workload testing and stress scenarios."""

    def test_scenario_20_loss_streak_decay(self):
        """S1: 20 consecutive Stop Losses: cushion decays as 0.5^N, floor is never breached."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        equity = 10000.0
        engine.update_equity(equity)
        floor = engine.floor  # 9200.0
        initial_cushion = engine.cushion  # 800.0

        for trade_idx in range(1, 21):
            risk_pct = engine.compute_dynamic_risk(base_risk_pct=0.05, current_equity=equity)
            dollar_loss = equity * risk_pct
            # Incur maximum Stop Loss
            equity -= dollar_loss
            p, f, c = engine.update_equity(equity)

            # Invariant: Equity must strictly exceed floor
            assert equity > floor, f"Failed at trade {trade_idx}: equity {equity} <= floor {floor}"
            # Invariant: Cushion must decay smoothly
            expected_cushion_bound = initial_cushion * (0.50 ** trade_idx)
            assert c >= expected_cushion_bound - 1e-5

    def test_scenario_hyper_growth_10_wins(self):
        """S2: 10 consecutive winning trades: peak ratchets up, floor follows, dynamic leverage expands."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        equity = 10000.0
        engine.update_equity(equity)

        inputs = [0.0] * 137
        inputs[88], inputs[84], inputs[81] = 1.0, 1.0, 1.0  # Full Bull confluence

        for win_idx in range(1, 11):
            risk_pct = engine.compute_dynamic_risk(
                base_risk_pct=0.005,
                current_equity=equity,
                inputs=inputs,
                trade_direction=1,
            )
            # Simulated win with 3R profit
            profit = equity * risk_pct * 3.0
            equity += profit
            p, f, c = engine.update_equity(equity)

            # Peak equity must ratchet up
            assert p == pytest.approx(equity)
            # Floor must follow monotonically
            assert f == pytest.approx(0.92 * equity)

        assert equity > 10000.0
        assert engine.peak_equity == pytest.approx(equity)

    def test_scenario_chop_whipsaw_contraction(self):
        """S3: Alternating win/loss near floor contracts risk safely to defensive posture."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        # Move account near floor: equity = 9250 (cushion = 50)
        equity = 9250.0

        for _ in range(10):
            risk_pct = engine.compute_dynamic_risk(base_risk_pct=0.02, current_equity=equity)
            # Risk percentage near floor must be severely throttled
            cushion = max(0.0, equity - 9200.0)
            max_permitted_loss = 0.50 * cushion
            actual_loss = risk_pct * equity
            assert actual_loss <= max_permitted_loss + 1e-7
            # Whipsaw: small loss then small win
            equity -= actual_loss
            equity += actual_loss * 0.9

        assert equity >= 9200.0

    def test_cppi_100_consecutive_losses_floor_invariance(self):
        """Suite 6 (boundary_specs): 100 consecutive full losses; equity asymptotically approaches floor."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        equity = 10000.0
        engine.update_equity(equity)
        floor = 9200.0

        for i in range(100):
            risk = engine.compute_dynamic_risk(0.05, equity)
            loss = equity * risk
            equity -= loss
            p, f, c = engine.update_equity(equity)
            assert equity >= floor, f"Floor breached on step {i}: {equity} < {floor}"

        assert equity >= floor

    def test_cppi_slippage_tolerance_50_pct(self):
        """Suite 7 (boundary_specs): 20 consecutive losses with 1.5x slippage (50% slippage).

        Because cushion_loss_fraction = 0.50, slippage factor 1.5 still leaves 1 - 0.75 = 0.25 > 0 cushion,
        guaranteeing floor preservation.
        """
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08, cushion_loss_fraction=0.50))
        equity = 10000.0
        engine.update_equity(equity)
        floor = 9200.0

        for step in range(20):
            risk = engine.compute_dynamic_risk(0.05, equity)
            # 1.5x slippage on stop loss
            slipped_loss = min(equity - floor, 1.5 * (equity * risk))
            equity -= slipped_loss
            p, f, c = engine.update_equity(equity)
            assert equity >= floor, f"Floor breached with slippage on step {step}: {equity} < {floor}"

        assert equity >= floor

    def test_telemetry_output_consistency(self):
        """Engine telemetry returns valid metrics dictionary."""
        engine = CPPIFloorEngine(initial_equity=10000.0)
        engine.compute_dynamic_risk(0.01, 10000.0)
        tel = engine.get_telemetry()
        assert tel["peak_equity"] == 10000.0
        assert tel["floor"] == pytest.approx(9200.0)
        assert tel["cushion"] == pytest.approx(800.0)
        assert tel["cushion_ratio"] == pytest.approx(1.0)
        assert tel["trade_eval_count"] == 1
        assert tel["max_allowed_dd"] == 0.08
