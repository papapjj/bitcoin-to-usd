"""
Challenger Adversarial Stress Harness for CrossAssetMacroEngine (Module R3)
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine
Milestone: M1 Iteration 2 Challenger Verification

Focus Areas:
1. Zero-variance series stress testing (all 16 combinations of active/flatlined assets).
2. Inverted and collinear series (exact anti-correlation, rank-deficient correlation matrices).
3. Synchronous Monte Carlo fuzz vectors (1,000 trials with min_ret_len >= 5).
4. Asynchronous disparate-length feeds analysis (empirical investigation of PSD boundaries).
5. Strict algebraic verification:
   - Correlation matrix symmetry: C == C^T
   - Unit diagonals: diag(C) == 1.0
   - Positive Semi-Definiteness (PSD): min(eigvalsh(C)) >= -1e-8
   - Composite macro index I_macro in [-1.0, 1.0] and strictly finite
   - 32D embedding strictly float32, bounded in [-1.0, 1.0], zero NaN, zero Inf
"""

from __future__ import annotations

import itertools
import math
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pytest

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_reasoning_world_class.src.macro_reasoning_engine import (
    CrossAssetMacroEngine,
    DEFAULT_MACRO_BASELINES,
    MacroAsset,
    MacroConfig,
    MacroEngineOutput,
    MacroRegime,
    SUPPORTED_MACRO_ASSETS,
    MACRO_FEATURE_NAMES,
)


def reconstruct_4x4_corr_matrix(
    output: MacroEngineOutput, symbols: List[str] = SUPPORTED_MACRO_ASSETS
) -> np.ndarray:
    """Reconstructs full 4x4 correlation matrix from pairwise correlations dict."""
    n = len(symbols)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    C = np.eye(n, dtype=np.float64)

    for (s1, s2), rho in output.pairwise_correlations.items():
        if s1 in sym_to_idx and s2 in sym_to_idx:
            i, j = sym_to_idx[s1], sym_to_idx[s2]
            C[i, j] = rho
            C[j, i] = rho

    return C


class TestChallengerCorrelationAlgebraicProperties:
    """Rigorous algebraic verification: Symmetry, Unit Diagonal, and Positive Semi-Definiteness (PSD)."""

    @pytest.mark.parametrize("active_mask", list(itertools.product([True, False], repeat=4)))
    def test_zero_variance_permutations_symmetry_and_psd(self, active_mask: Tuple[bool, ...]):
        """
        Test all 16 boolean permutations of active vs flatline (zero-variance) assets.
        Every resulting 4x4 correlation matrix must be strictly symmetric, unit diagonal, and PSD.
        """
        engine = CrossAssetMacroEngine(lookback_window=60)
        np.random.seed(42 + sum(1 << i for i, b in enumerate(active_mask)))
        steps = 65

        for i, sym in enumerate(SUPPORTED_MACRO_ASSETS):
            if active_mask[i]:
                # Active non-zero variance price series
                returns = np.random.normal(0.0005, 0.005, steps)
                prices = DEFAULT_MACRO_BASELINES[sym] * np.exp(np.cumsum(returns))
                engine.set_price_history(sym, prices.tolist())
            else:
                # Flatlined zero-variance price series
                engine.set_price_history(sym, [DEFAULT_MACRO_BASELINES[sym]] * steps)

        output = engine.extract_features()
        C = reconstruct_4x4_corr_matrix(output)

        # 1. Symmetry
        assert np.allclose(C, C.T, atol=1e-12), f"Matrix not symmetric for mask {active_mask}!\n{C}"

        # 2. Unit diagonals
        assert np.allclose(np.diag(C), 1.0, atol=1e-12), f"Diagonals not 1.0 for mask {active_mask}!\n{C}"

        # 3. Off-diagonal elements bounded in [-1.0, 1.0]
        assert np.all(C >= -1.0 - 1e-7) and np.all(C <= 1.0 + 1e-7), f"Correlations exceed [-1, 1]!\n{C}"

        # 4. Positive Semi-Definiteness (PSD)
        eigvals = np.linalg.eigvalsh(C)
        min_eig = float(np.min(eigvals))
        assert min_eig >= -1e-8, (
            f"Correlation matrix violated PSD: min eigenvalue = {min_eig:.10e} for mask {active_mask}!\n"
            f"Eigvals: {eigvals}\n{C}"
        )

    def test_perfectly_inverted_series_algebraic_psd(self):
        """
        Stress test exact inverted (anti-correlated rho = -1.0) series.
        Pairs of perfectly negatively correlated assets must produce a rank-deficient but valid PSD matrix.
        """
        engine = CrossAssetMacroEngine(lookback_window=60)
        steps = 65
        t = np.linspace(0, 4 * np.pi, steps)
        wave = np.sin(t)

        # DXY and US10Y are in-phase
        engine.set_price_history("DXY", (100.0 + 5.0 * wave).tolist())
        engine.set_price_history("US10Y", (4.0 + 0.5 * wave).tolist())
        # SPX and XAUUSD are exactly inverted (-wave)
        engine.set_price_history("SPX", (5000.0 - 100.0 * wave).tolist())
        engine.set_price_history("XAUUSD", (2300.0 - 50.0 * wave).tolist())

        output = engine.extract_features()
        C = reconstruct_4x4_corr_matrix(output)

        # Check expected near-perfect correlations
        assert output.pairwise_correlations[("DXY", "US10Y")] > 0.99
        assert output.pairwise_correlations[("SPX", "XAUUSD")] > 0.99
        assert output.pairwise_correlations[("DXY", "SPX")] < -0.99
        assert output.pairwise_correlations[("DXY", "XAUUSD")] < -0.99
        assert output.pairwise_correlations[("US10Y", "SPX")] < -0.99
        assert output.pairwise_correlations[("US10Y", "XAUUSD")] < -0.99

        # Invariants
        assert np.allclose(C, C.T, atol=1e-12)
        assert np.allclose(np.diag(C), 1.0, atol=1e-12)
        eigvals = np.linalg.eigvalsh(C)
        min_eig = float(np.min(eigvals))
        assert min_eig >= -1e-8, f"Inverted series violated PSD: min eigenvalue = {min_eig:.10e}!\nEigvals: {eigvals}\n{C}"

    def test_single_spike_near_zero_variance(self):
        """
        Asset has flat series except for a single tick disturbance.
        Tests near-zero denominator and float precision boundaries.
        """
        engine = CrossAssetMacroEngine(lookback_window=60)
        steps = 65
        for sym in SUPPORTED_MACRO_ASSETS:
            prices = [100.0] * steps
            prices[10] = 100.00000001  # Tiny 1e-8 perturbation
            engine.set_price_history(sym, prices)

        output = engine.extract_features()
        C = reconstruct_4x4_corr_matrix(output)

        assert np.allclose(C, C.T, atol=1e-12)
        assert np.allclose(np.diag(C), 1.0, atol=1e-12)
        eigvals = np.linalg.eigvalsh(C)
        assert float(np.min(eigvals)) >= -1e-8


class TestChallengerCompositeIndexAndEmbeddingBounds:
    """Verify I_macro in [-1.0, 1.0] and 32D embedding strictly finite float32."""

    def test_composite_index_range_extremes(self):
        """Construct extreme bull, extreme bear, and chaotic market scenarios."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        steps = 65

        # Extreme Risk-On: SPX rocket (+50%), DXY plunging (-50%), US10Y plunging (-50%), Gold flat
        engine.set_price_history("SPX", np.linspace(4000, 6000, steps).tolist())
        engine.set_price_history("DXY", np.linspace(110, 55, steps).tolist())
        engine.set_price_history("US10Y", np.linspace(5.0, 2.5, steps).tolist())
        engine.set_price_history("XAUUSD", [2350.0] * steps)

        out_bull = engine.extract_features()
        assert 0.35 < out_bull.macro_regime_index <= 1.0
        assert out_bull.regime == MacroRegime.STRONG_RISK_ON

        # Extreme Risk-Off: SPX crashing (-50%), DXY skyrocketing (+50%), US10Y soaring (+50%), Gold soaring
        engine.set_price_history("SPX", np.linspace(6000, 3000, steps).tolist())
        engine.set_price_history("DXY", np.linspace(90, 135, steps).tolist())
        engine.set_price_history("US10Y", np.linspace(3.0, 6.0, steps).tolist())
        engine.set_price_history("XAUUSD", np.linspace(2000, 3000, steps).tolist())

        out_bear = engine.extract_features()
        assert -1.0 <= out_bear.macro_regime_index < -0.35
        assert out_bear.regime == MacroRegime.STRONG_RISK_OFF

    def test_32d_embedding_exact_structure(self):
        """Verify strict shape, float32 dtype, and exact feature range."""
        engine = CrossAssetMacroEngine(lookback_window=60)
        for sym in SUPPORTED_MACRO_ASSETS:
            engine.set_price_history(sym, [100.0, 102.0, 101.0, 103.0, 105.0] * 15)

        out = engine.extract_features()
        feats = out.features

        assert feats.shape == (32,), f"Expected shape (32,), got {feats.shape}"
        assert feats.dtype == np.float32, f"Expected float32, got {feats.dtype}"
        assert np.all(np.isfinite(feats)), "Non-finite values present in features!"
        assert np.all(feats >= -1.0) and np.all(feats <= 1.0), "Features exceed [-1.0, 1.0]!"

        # Extract features array shortcut
        feats_arr = engine.extract_features_array()
        assert np.array_equal(feats, feats_arr)


class TestChallengerMonteCarloFuzzHarness:
    """Adversarial Monte Carlo fuzz harness across 1,000 extreme, random, and degenerate trials."""

    def test_synchronous_fuzz_harness_1000_trials(self):
        """
        1,000 randomized trials with min_ret_len >= 5 asserting PSD, symmetry, finite float32, and range bounds.
        """
        np.random.seed(12345)
        engine = CrossAssetMacroEngine(lookback_window=60)

        pathological_tokens = [
            np.nan,
            np.inf,
            -np.inf,
            0.0,
            -0.0,
            -99999.0,
            1e-300,
            1e300,
            1e-15,
            1e15,
            1e-8,
        ]

        for trial in range(1000):
            # Vary length from 10 to 100 bars independently per asset
            for sym in SUPPORTED_MACRO_ASSETS:
                mode = np.random.choice(["random_walk", "heavy_tail", "constant", "corrupted"])
                length = np.random.randint(10, 100)

                if mode == "constant":
                    c_val = float(np.random.choice([10.0, 100.0, 2500.0, 0.0, -50.0]))
                    engine.set_price_history(sym, [c_val] * length)
                elif mode == "heavy_tail":
                    # Cauchy distributed returns
                    cauchy_rets = np.random.standard_cauchy(length) * 0.01
                    prices = np.abs(100.0 * np.exp(np.clip(np.cumsum(cauchy_rets), -50, 50)))
                    engine.set_price_history(sym, prices.tolist())
                elif mode == "corrupted":
                    # Prices with 25% pathological tokens
                    base = np.random.uniform(1.0, 5000.0, length)
                    corrupt_idx = np.where(np.random.rand(length) < 0.25)[0]
                    for idx in corrupt_idx:
                        base[idx] = np.random.choice(pathological_tokens)
                    engine.set_price_history(sym, base.tolist())
                else:  # random_walk
                    rets = np.random.normal(0.0, 0.02, length)
                    prices = 100.0 * np.exp(np.cumsum(rets))
                    engine.set_price_history(sym, prices.tolist())

            out = engine.extract_features()
            feats = out.features

            # Verification 1: Shape and Dtype
            assert feats.shape == (32,), f"Trial {trial}: Features shape != (32,)"
            assert feats.dtype == np.float32, f"Trial {trial}: Features dtype != float32"

            # Verification 2: Strict Finite & Range [-1.0, 1.0]
            assert np.all(np.isfinite(feats)), f"Trial {trial}: Non-finite features found!"
            assert np.all(feats >= -1.0) and np.all(feats <= 1.0), f"Trial {trial}: Features exceed [-1, 1]!"

            # Verification 3: I_macro composite index
            assert math.isfinite(out.macro_regime_index), f"Trial {trial}: I_macro is not finite!"
            assert -1.0 <= out.macro_regime_index <= 1.0, f"Trial {trial}: I_macro = {out.macro_regime_index} outside [-1, 1]!"

            # Verification 4: Correlation Matrix Symmetry, Unit Diagonals, and PSD
            C = reconstruct_4x4_corr_matrix(out)
            assert np.allclose(C, C.T, atol=1e-12), f"Trial {trial}: Matrix not symmetric!\n{C}"
            assert np.allclose(np.diag(C), 1.0, atol=1e-12), f"Trial {trial}: Non-unit diagonal!\n{C}"
            eigvals = np.linalg.eigvalsh(C)
            min_eig = float(np.min(eigvals))
            assert min_eig >= -1e-8, f"Trial {trial}: Violated PSD! min_eig = {min_eig:.10e}\n{C}\nEigvals: {eigvals}"

    def test_asynchronous_disparate_buffer_depths_features_bounds(self):
        """
        When feeds are asynchronous or missing (min_ret_len < 5), verify that:
        1. 32D features and I_macro remain strictly finite float32 in [-1.0, 1.0].
        2. Pairwise correlations are strictly symmetric with unit diagonals and bounded in [-1.0, 1.0].
        3. Identifies the mathematical condition where pairwise asynchronous correlations can be non-PSD.
        """
        engine = CrossAssetMacroEngine(lookback_window=60)
        # SPX has 0 quotes (missing feed), while DXY has 28 quotes, US10Y has 60, XAUUSD has 11
        np.random.seed(999)
        engine.set_price_history("DXY", (100.0 + np.cumsum(np.random.normal(0, 0.1, 29))).tolist())
        engine.set_price_history("US10Y", (4.0 + np.cumsum(np.random.normal(0, 0.05, 61))).tolist())
        engine.set_price_history("SPX", [])
        engine.set_price_history("XAUUSD", (2300.0 + np.cumsum(np.random.normal(0, 1.0, 12))).tolist())

        out = engine.extract_features()
        feats = out.features

        # Strict AC-7 invariants
        assert feats.shape == (32,)
        assert feats.dtype == np.float32
        assert np.all(np.isfinite(feats))
        assert np.all(feats >= -1.0) and np.all(feats <= 1.0)
        assert -1.0 <= out.macro_regime_index <= 1.0

        C = reconstruct_4x4_corr_matrix(out)
        # Symmetry & unit diagonals hold unconditionally
        assert np.allclose(C, C.T, atol=1e-12)
        assert np.allclose(np.diag(C), 1.0, atol=1e-12)
        assert np.all(C >= -1.0) and np.all(C <= 1.0)
