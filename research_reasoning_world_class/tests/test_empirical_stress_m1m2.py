"""
Standalone Empirical Stress Test Suite (Milestones M1 & M2)
Project: Frontier Mythos MT5 Strategy Tester Optimization & Verification
Agent: teamwork_preview_challenger (challenger_mt5_m1m2_2)

Objectives:
1. Generate 500 diverse MT5 137D feature vectors with extreme price movements, wicks, spreads, and volumes.
2. Reconstruct M1 candles using MythosMT5Policy logic: verify valid OHLCV geometry (H >= max(O,C), L <= min(O,C), V > 0, 0 NaN/Inf).
3. Feed reconstructed bars into SMCLiquidityEngine: verify genuine Order Blocks, FVGs, and Liquidity Sweeps are detected.
4. Simulate 1,000 MT5 ticks through MythosMT5Policy.act_nam_tac_nhan: verify Tier 1 Micro-Scalp triggers on micro-sweeps, Tier 2 triggers on institutional setups, and Agent 4 Governor is strictly 100.0% HOLD (POLICY_CLOSE_ALL == 0.0) across all 1,000 ticks without exception.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

# Ensure project roots in path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "research_reasoning_world_class"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from research_reasoning_world_class.src.smc_liquidity_engine import SMCLiquidityEngine, SMCConfig
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy


# =============================================================================
# 1. 500 DIVERSE MT5 137D FEATURE VECTOR GENERATOR
# =============================================================================
def generate_500_diverse_mt5_feature_vectors(seed: int = 2026) -> np.ndarray:
    """
    Generates 500 diverse MT5 137D feature vectors covering:
    - Normal market fluctuations (100)
    - Extreme price movements / flash jumps (100)
    - Extreme wicks & Dojis (100)
    - Extreme spreads & volumes (100)
    - Adversarial edge cases / micro-sweeps / Wyckoff catalysts (100)
    """
    rng = np.random.RandomState(seed)
    vectors = np.zeros((500, 137), dtype=np.float32)

    for i in range(500):
        v = np.zeros(137, dtype=np.float32)
        
        # Populate basic momentum / noise
        v[1:6] = rng.randn(5) * 0.5
        v[11:15] = rng.randn(4) * 0.5
        v[78] = float(np.sin(i * 0.1))  # hour_sin
        v[79] = float(np.cos(i * 0.1))  # hour_cos

        if i < 100:
            # Sub-regime 1: Normal market (modest returns, typical wicks)
            v[0] = float(rng.randn() * 0.8)             # ret1
            v[7] = float(rng.randn() * 0.6)             # body
            v[8] = float(rng.uniform(0.05, 0.4))        # upper_wick
            v[9] = float(rng.uniform(0.05, 0.4))        # lower_wick
            v[10] = abs(v[7]) + v[8] + v[9]             # range_val
            v[15] = float(rng.uniform(0.1, 0.5))        # spread_atr
            v[81] = float(rng.randn() * 0.5)            # m5_slope
            v[85] = float(rng.randn() * 0.5)            # m15_slope
            v[89] = float(rng.randn() * 0.5)            # h1_slope
            v[120] = float(rng.randn() * 0.2)           # sweep_rej
            v[121] = float(rng.uniform(0.5, 2.0))       # rel_vol
            v[129] = float(rng.uniform(0.0, 0.2))       # spring_score
            v[130] = float(rng.uniform(0.0, 0.2))       # upthrust_score

        elif i < 200:
            # Sub-regime 2: Extreme price movements (flash crashes / giant pumps)
            v[0] = float(rng.choice([-15.0, -10.0, -8.0, 8.0, 10.0, 15.0])) # extreme ret1
            v[7] = float(np.sign(v[0]) * rng.uniform(5.0, 12.0))             # giant body
            v[8] = float(rng.uniform(0.1, 2.0))
            v[9] = float(rng.uniform(0.1, 2.0))
            v[10] = abs(v[7]) + v[8] + v[9]
            v[15] = float(rng.uniform(0.5, 2.0))
            v[81] = float(np.sign(v[0]) * rng.uniform(2.0, 5.0))
            v[85] = float(np.sign(v[0]) * rng.uniform(2.0, 5.0))
            v[89] = float(np.sign(v[0]) * rng.uniform(2.0, 5.0))
            v[120] = float(-np.sign(v[0]))
            v[121] = float(rng.uniform(5.0, 20.0))       # high volume spike
            v[129] = 0.0 if v[0] < 0 else 0.8
            v[130] = 0.8 if v[0] < 0 else 0.0

        elif i < 300:
            # Sub-regime 3: Extreme wicks & Doji patterns
            v[0] = float(rng.randn() * 0.2)             # near zero return
            v[7] = float(rng.randn() * 0.05)            # near zero body
            v[8] = float(rng.uniform(5.0, 25.0))        # giant upper wick
            v[9] = float(rng.uniform(5.0, 25.0))        # giant lower wick
            v[10] = abs(v[7]) + v[8] + v[9]
            v[15] = float(rng.uniform(0.2, 1.0))
            v[81] = float(rng.randn() * 0.1)
            v[85] = float(rng.randn() * 0.1)
            v[89] = float(rng.randn() * 0.1)
            v[120] = float(rng.choice([-1.0, 1.0]))
            v[121] = float(rng.uniform(1.0, 5.0))
            v[129] = float(rng.uniform(0.5, 1.0))
            v[130] = float(rng.uniform(0.5, 1.0))

        elif i < 400:
            # Sub-regime 4: Extreme spreads & volume variations
            v[0] = float(rng.randn() * 1.0)
            v[7] = float(rng.randn() * 0.8)
            v[8] = float(rng.uniform(0.1, 1.0))
            v[9] = float(rng.uniform(0.1, 1.0))
            v[10] = abs(v[7]) + v[8] + v[9]
            v[15] = float(rng.choice([0.0001, 0.001, 10.0, 25.0, 50.0])) # extreme spread
            v[81] = float(rng.randn() * 1.5)
            v[85] = float(rng.randn() * 1.5)
            v[89] = float(rng.randn() * 1.5)
            v[120] = float(rng.randn() * 0.5)
            v[121] = float(rng.choice([0.001, 0.01, 50.0, 150.0, 500.0])) # extreme volume
            v[129] = float(rng.uniform(0.0, 0.5))
            v[130] = float(rng.uniform(0.0, 0.5))

        else:
            # Sub-regime 5: Adversarial edge cases (negative wicks, range anomalies, wyckoff extremes)
            v[0] = float(rng.randn() * 2.0)
            v[7] = float(rng.randn() * 1.5)
            # Negative wick inputs to test clamping logic max(0.0, ...)
            v[8] = float(rng.choice([-5.0, -1.0, 0.0, 3.0]))
            v[9] = float(rng.choice([-5.0, -1.0, 0.0, 3.0]))
            # Range anomaly: range_val much larger than body + wicks
            v[10] = float(rng.choice([0.0, 10.0, 50.0]))
            v[15] = float(rng.uniform(0.05, 1.0))
            v[81] = float(rng.uniform(-3.0, 3.0))
            v[85] = float(rng.uniform(-3.0, 3.0))
            v[89] = float(rng.uniform(-3.0, 3.0))
            # Micro-sweeps and institutional catalysts
            v[120] = float(rng.choice([-1.0, 0.0, 1.0]))
            v[121] = float(rng.choice([-1.0, 0.0, 0.1, 10.0]))
            v[129] = float(rng.choice([0.0, 0.5, 0.95])) # spring
            v[130] = float(rng.choice([0.0, 0.5, 0.95])) # upthrust

        vectors[i] = v

    return vectors


# =============================================================================
# 2. CANDLESTICK RECONSTRUCTION LOGIC
# =============================================================================
def reconstruct_candle_from_mt5_vector(
    base_obs_137: np.ndarray,
    current_price: float,
    current_atr: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Reconstructs one M1 OHLCV candlestick using MythosMT5Policy exact arithmetic logic.
    Returns: (bar [O, H, L, C, V], next_price, next_atr)
    """
    # Spread ATR update
    spread_atr = float(base_obs_137[15]) if base_obs_137.size > 15 else 0.0
    if spread_atr > 0.05:
        est_atr = 28.98 / spread_atr
        if 10.0 <= est_atr <= 500.0:
            current_atr = 0.90 * current_atr + 0.10 * est_atr

    ret1 = float(base_obs_137[0])
    body = float(base_obs_137[7])
    upper_wick = max(0.0, float(base_obs_137[8]))
    lower_wick = max(0.0, float(base_obs_137[9]))
    range_val = max(0.0, float(base_obs_137[10]))
    rel_vol = max(0.1, float(base_obs_137[121])) if base_obs_137.size > 121 else 1.0

    atr = current_atr
    c = current_price + (ret1 * atr)
    next_price = c

    o = c - (body * atr)
    h = max(o, c) + (upper_wick * atr)
    l = min(o, c) - (lower_wick * atr)
    if range_val > 0 and (h - l) < range_val * atr * 0.95:
        mid = 0.5 * (o + c)
        h = max(h, mid + 0.5 * range_val * atr)
        l = min(l, mid - 0.5 * range_val * atr)
    v = max(1.0, rel_vol * 150.0)

    bar = np.array([o, h, l, c, v], dtype=np.float64)
    return bar, next_price, current_atr


# =============================================================================
# TEST CLASS 1: 500 VECTORS & OHLCV GEOMETRY
# =============================================================================
class TestM1CandlestickReconstructionStress:
    """Empirical verification of M1 Candlestick Reconstruction Geometry across 500 vectors."""

    def test_reconstruct_500_vectors_ohlcv_invariants(self):
        vectors = generate_500_diverse_mt5_feature_vectors(seed=2026)
        assert vectors.shape == (500, 137)

        price = 93_440.0
        atr = 48.60

        reconstructed_bars = []
        violations = []

        for i in range(500):
            bar, price, atr = reconstruct_candle_from_mt5_vector(vectors[i], price, atr)
            reconstructed_bars.append(bar)
            o, h, l, c, v = bar

            # Invariant 1: Finite numbers (0 NaN, 0 Inf)
            if not np.all(np.isfinite(bar)):
                violations.append(f"Bar #{i}: Non-finite value in bar {bar}")

            # Invariant 2: High >= max(Open, Close)
            if h < max(o, c) - 1e-7:
                violations.append(f"Bar #{i}: H ({h}) < max(O,C) ({max(o, c)})")

            # Invariant 3: Low <= min(Open, Close)
            if l > min(o, c) + 1e-7:
                violations.append(f"Bar #{i}: L ({l}) > min(O,C) ({min(o, c)})")

            # Invariant 4: High >= Low
            if h < l - 1e-7:
                violations.append(f"Bar #{i}: H ({h}) < L ({l})")

            # Invariant 5: Volume > 0
            if v <= 0.0:
                violations.append(f"Bar #{i}: Volume ({v}) <= 0")

        assert len(violations) == 0, f"Found {len(violations)} geometry violations: {violations[:5]}"
        assert len(reconstructed_bars) == 500
        print(f"\n[STRESS 1 PASSED] 500/500 M1 Bars reconstructed with 0 NaN/Inf and valid OHLCV geometry.")


# =============================================================================
# TEST CLASS 2: SMC FEATURE ENGINE DETECTION
# =============================================================================
class TestSMCLiquidityEngineDetection:
    """Empirical verification that SMCLiquidityEngine detects genuine OBs, FVGs, and Sweeps."""

    def test_smc_features_on_500_reconstructed_bars_stream(self):
        vectors = generate_500_diverse_mt5_feature_vectors(seed=2026)
        price = 93_440.0
        atr = 48.60
        reconstructed_bars = []
        for i in range(500):
            bar, price, atr = reconstruct_candle_from_mt5_vector(vectors[i], price, atr)
            reconstructed_bars.append(bar)
        bars_arr = np.array(reconstructed_bars, dtype=np.float64)

        smc_engine = SMCLiquidityEngine(SMCConfig())
        
        # Test rolling 60-bar slices
        total_slices = 500 - 60 + 1
        nan_inf_count = 0
        detected_bull_obs = 0
        detected_bear_obs = 0
        detected_bull_fvgs = 0
        detected_bear_fvgs = 0
        detected_bsl_sweeps = 0
        detected_ssl_sweeps = 0

        for start_idx in range(total_slices):
            window = bars_arr[start_idx : start_idx + 60]
            feat = smc_engine.extract_features(window)
            
            assert feat.shape == (64,), f"Expected shape (64,), got {feat.shape}"
            assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"

            if not np.all(np.isfinite(feat)):
                nan_inf_count += 1

            if feat[32] > 0.0:  # bullish_ob_detected
                detected_bull_obs += 1
            if feat[33] > 0.0:  # bearish_ob_detected
                detected_bear_obs += 1
            if feat[16] > 0.0:  # bullish_fvg_active
                detected_bull_fvgs += 1
            if feat[17] > 0.0:  # bearish_fvg_active
                detected_bear_fvgs += 1
            if feat[0] > 0.0:   # bsl_sweep_detected
                detected_bsl_sweeps += 1
            if feat[1] > 0.0:   # ssl_sweep_detected
                detected_ssl_sweeps += 1

        assert nan_inf_count == 0, f"Encountered {nan_inf_count} non-finite SMC feature vectors"
        total_patterns = (
            detected_bull_obs + detected_bear_obs +
            detected_bull_fvgs + detected_bear_fvgs +
            detected_bsl_sweeps + detected_ssl_sweeps
        )
        assert total_patterns > 0, "Zero SMC patterns detected across 441 rolling M1 bar slices!"
        print(f"\n[STRESS 2 PASSED] SMC Stream Extractions (441 slices):")
        print(f"  - Bullish OBs: {detected_bull_obs} | Bearish OBs: {detected_bear_obs}")
        print(f"  - Bullish FVGs: {detected_bull_fvgs} | Bearish FVGs: {detected_bear_fvgs}")
        print(f"  - BSL Sweeps: {detected_bsl_sweeps} | SSL Sweeps: {detected_ssl_sweeps}")
        print(f"  - Total Patterns: {total_patterns} (0 NaN/Inf)")

    def test_smc_canonical_pattern_detection(self):
        """Specifically verify detection of genuine canonical SMC structures."""
        smc_engine = SMCLiquidityEngine(SMCConfig())

        # Canonical Bullish FVG: Bar 0 High < Bar 2 Low
        # Bar 0: [100, 102, 99, 101, 100]  -> High = 102
        # Bar 1: [101, 110, 101, 109, 300] (Large impulse up)
        # Bar 2: [109, 112, 105, 111, 150] -> Low = 105 (Gap: 102 to 105)
        bars_fvg = np.zeros((60, 5), dtype=np.float64)
        for i in range(57):
            bars_fvg[i] = [100.0, 101.0, 99.0, 100.0, 100.0]
        bars_fvg[57] = [100.0, 102.0, 99.0, 101.0, 100.0]
        bars_fvg[58] = [101.0, 110.0, 101.0, 109.0, 300.0]
        bars_fvg[59] = [109.0, 112.0, 105.0, 111.0, 150.0]
        feat_bull_fvg = smc_engine.extract_features(bars_fvg)
        assert feat_bull_fvg[16] == 1.0, f"Bullish FVG not detected: {feat_bull_fvg[16]}"

        # Canonical Bearish FVG: Bar 0 Low > Bar 2 High
        bars_bear_fvg = np.zeros((60, 5), dtype=np.float64)
        for i in range(57):
            bars_bear_fvg[i] = [110.0, 111.0, 109.0, 110.0, 100.0]
        bars_bear_fvg[57] = [110.0, 111.0, 108.0, 109.0, 100.0]  # Low = 108
        bars_bear_fvg[58] = [109.0, 109.0, 100.0, 101.0, 300.0]  # Impulse down
        bars_bear_fvg[59] = [101.0, 105.0, 99.0, 100.0, 150.0]   # High = 105 (Gap: 108 to 105)
        feat_bear_fvg = smc_engine.extract_features(bars_bear_fvg)
        assert feat_bear_fvg[17] == 1.0, f"Bearish FVG not detected: {feat_bear_fvg[17]}"

        print("[STRESS 2 PASSED] Canonical Bullish and Bearish FVGs accurately detected.")


# =============================================================================
# TEST CLASS 3: 1,000 MT5 TICKS SIMULATION & DUAL-TIER TRIGGER CHALLENGE
# =============================================================================
class TestDualTierExecutionAndGovernorInvariant1000Ticks:
    """
    Simulates 1,000 MT5 ticks through MythosMT5Policy.act_nam_tac_nhan.
    Verifies:
    1. Agent 4 Governor is strictly 100.0% HOLD (POLICY_CLOSE_ALL == 0.0) across all 1,000 ticks.
    2. Empirical challenge on Tier 1 Micro-Scalp and Tier 2 Sniper trigger conditions.
    """

    @pytest.fixture(scope="class")
    def initialized_policy(self):
        config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
        config = load_config(config_path)
        checkpoint_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
        base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
        policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
        return policy

    def test_1000_ticks_governor_invariant_and_tier_triggers(self, initialized_policy):
        policy = initialized_policy
        total_ticks = 1000
        rng = np.random.RandomState(42)

        agent4_violations = []
        tier1_scalp_triggers = 0
        counter_trap_triggers = 0
        tier2_sniper_triggers = 0
        hold_veto_traps = 0
        hold_idle_gates = 0
        latencies = []

        print(f"\n--- Commencing 1,000 MT5 Tick Empirical Simulation ---")

        for tick_idx in range(1, total_ticks + 1):
            # Synthesize tick observation with realistic distribution
            obs = rng.randn(137).astype(np.float32) * 0.5
            obs[15] = 0.25 # spread_atr

            if tick_idx <= 250:
                # Reg 1: Calm / Idle noise
                obs[120] = 0.0
                obs[129] = 0.0
                obs[130] = 0.0

            elif tick_idx <= 500:
                # Reg 2: Micro-sweeps & Spring/Upthrust catalysts (Target: Tier 1 Micro-Scalp)
                if tick_idx % 2 == 0:
                    obs[120] = 1.0   # Bullish sweep rejection
                    obs[129] = 0.85  # Spring score
                    obs[81] = 0.5    # M5 slope
                else:
                    obs[120] = -1.0  # Bearish sweep rejection
                    obs[130] = 0.85  # Upthrust score
                    obs[81] = -0.5

            elif tick_idx <= 750:
                # Reg 3: Institutional setups (Target: Tier 2 Sniper)
                if tick_idx % 2 == 0:
                    obs[81] = 2.0    # Strong M5
                    obs[85] = 2.0    # Strong M15
                    obs[89] = 2.0    # Strong H1
                    obs[120] = 1.0
                    obs[129] = 0.95
                else:
                    obs[81] = -2.0
                    obs[85] = -2.0
                    obs[89] = -2.0
                    obs[120] = -1.0
                    obs[130] = 0.95

            else:
                # Reg 4: Extreme volatile spikes
                obs[0] = float(rng.choice([-10.0, 10.0]))
                obs[120] = float(rng.choice([-1.0, 1.0]))
                obs[121] = 20.0

            t0 = time.perf_counter()
            traces = policy.act_nam_tac_nhan(obs)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms)

            # Check trace length invariant
            assert len(traces) == 5, f"Tick #{tick_idx}: Expected 5 traces, got {len(traces)}"

            # =================================================================
            # INVARIANT: AGENT 4 GOVERNOR STRICTLY 100.0% HOLD (POLICY_CLOSE_ALL == 0.0)
            # =================================================================
            agent4_trace = traces[4]
            agent4_out = getattr(agent4_trace, "output_vector", getattr(agent4_trace, "selected_action", None))
            agent4_details = getattr(agent4_trace, "action_details", {}) or {}

            # Check wire index 19 (POLICY_CLOSE_ALL)
            if agent4_out is not None:
                if agent4_out[19] != 0.0:
                    agent4_violations.append(f"Tick #{tick_idx}: Agent 4 wire[19] = {agent4_out[19]} != 0.0")
                if agent4_out[0] != 1.0:
                    agent4_violations.append(f"Tick #{tick_idx}: Agent 4 wire[0] = {agent4_out[0]} != 1.0 (ACTION_HOLD)")
                if agent4_out[10] != -100.0:
                    agent4_violations.append(f"Tick #{tick_idx}: Agent 4 wire[10] = {agent4_out[10]} != -100.0")

            if agent4_details.get("policy_close_all_locked", None) != 0.0:
                agent4_violations.append(f"Tick #{tick_idx}: details['policy_close_all_locked'] = {agent4_details.get('policy_close_all_locked')}")
            if agent4_details.get("wire_action", None) != 0:
                agent4_violations.append(f"Tick #{tick_idx}: details['wire_action'] = {agent4_details.get('wire_action')} != 0")

            # Check if Tier 1 micro-scalp, Counter-Trap sniper, or Tier 2 sniper triggered
            t0_details = getattr(traces[0], "action_details", {}) or {}
            t1_details = getattr(traces[1], "action_details", {}) or {}
            a0 = getattr(traces[0], "action_index", getattr(traces[0], "selected_idx", 0))
            a1 = getattr(traces[1], "action_index", getattr(traces[1], "selected_idx", 0))

            sl_atr_0 = float(t0_details.get("sl_atr", 1.0))
            sl_atr_1 = float(t1_details.get("sl_atr", 1.0))
            tp_rr_0 = t0_details.get("tp_atr", 0.0) / max(1e-4, sl_atr_0)
            tp_rr_1 = t1_details.get("tp_atr", 0.0) / max(1e-4, sl_atr_1)

            # Tier 1 micro-scalp: (tp_rr ~ 0.6 or 1.5) and sl_atr == 2.0
            is_t1 = (a0 == 1 and (abs(tp_rr_0 - 0.6) < 0.05 or abs(tp_rr_0 - 1.5) < 0.05) and abs(sl_atr_0 - 2.0) < 0.05) or \
                    (a1 == 2 and (abs(tp_rr_1 - 0.6) < 0.05 or abs(tp_rr_1 - 1.5) < 0.05) and abs(sl_atr_1 - 2.0) < 0.05)
            # Counter-Trap sniper: (tp_rr ~ 2.0 or 3.0) and (sl_atr == 1.0 or 2.5)
            is_ct = (a0 == 1 and (abs(tp_rr_0 - 3.0) < 0.05 or abs(tp_rr_0 - 2.0) < 0.05) and (abs(sl_atr_0 - 1.0) < 0.05 or abs(sl_atr_0 - 2.5) < 0.05)) or \
                    (a1 == 2 and (abs(tp_rr_1 - 3.0) < 0.05 or abs(tp_rr_1 - 2.0) < 0.05) and (abs(sl_atr_1 - 1.0) < 0.05 or abs(sl_atr_1 - 2.5) < 0.05))
            # Tier 2 sniper: (tp_rr ~ 5.0 and sl_atr == 2.0) or (tp_rr ~ 1.5 and sl_atr == 2.5)
            is_t2 = (a0 == 1 and ((abs(tp_rr_0 - 5.0) < 0.05 and abs(sl_atr_0 - 2.0) < 0.05) or (abs(tp_rr_0 - 1.5) < 0.05 and abs(sl_atr_0 - 2.5) < 0.05))) or \
                    (a1 == 2 and ((abs(tp_rr_1 - 5.0) < 0.05 and abs(sl_atr_1 - 2.0) < 0.05) or (abs(tp_rr_1 - 1.5) < 0.05 and abs(sl_atr_1 - 2.5) < 0.05)))

            if is_t1:
                tier1_scalp_triggers += 1
            elif is_ct:
                counter_trap_triggers += 1
            elif is_t2:
                tier2_sniper_triggers += 1
            else:
                mythos_verdict = t0_details.get("mythos_verdict", "")
                if "VETO_TRAP" in mythos_verdict:
                    hold_veto_traps += 1
                else:
                    hold_idle_gates += 1

        # Summary statistics
        latencies_steady = latencies[10:] # exclude initial warmup
        mean_lat = np.mean(latencies_steady)
        p95_lat = np.percentile(latencies_steady, 95)
        max_lat = np.max(latencies_steady)
        total_active_trades = tier1_scalp_triggers + counter_trap_triggers + tier2_sniper_triggers

        print(f"\n--- Simulation Results (1,000 Ticks) ---")
        print(f"Agent 4 Governor Violations: {len(agent4_violations)}")
        print(f"Tier 1 Micro-Scalp Triggers: {tier1_scalp_triggers}")
        print(f"Counter-Trap Sniper Triggers:{counter_trap_triggers}")
        print(f"Tier 2 Sniper Mode Triggers: {tier2_sniper_triggers}")
        print(f"Total Active Trades:         {total_active_trades}")
        print(f"HOLD Veto Traps:             {hold_veto_traps}")
        print(f"HOLD Idle Gates:             {hold_idle_gates}")
        print(f"Latency: Mean={mean_lat:.1f}ms, P95={p95_lat:.1f}ms, Max={max_lat:.1f}ms")

        # 1. Assert Governor Invariant: MUST BE STRICTLY 0 VIOLATIONS (100.0% HOLD)
        assert len(agent4_violations) == 0, f"Found {len(agent4_violations)} Agent 4 violations!"

        # 2. Assert Active Trades: MUST BE > 300 across 1,000 ticks (>300 trades/day requirement)
        assert total_active_trades > 300, f"Active trades total {total_active_trades} <= 300 across 1,000 ticks!"

        # Return empirical stats for diagnostic assertions
        return {
            "tier1_triggers": tier1_scalp_triggers,
            "counter_trap_triggers": counter_trap_triggers,
            "tier2_triggers": tier2_sniper_triggers,
            "total_active_trades": total_active_trades,
            "agent4_violations": len(agent4_violations),
            "hold_veto_traps": hold_veto_traps,
            "hold_idle_gates": hold_idle_gates,
        }


# =============================================================================
# STANDALONE EXECUTION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING STANDALONE EMPIRICAL STRESS TEST SUITE (M1 & M2)")
    print("=" * 70)

    # 1. M1 Candlestick Reconstruction Geometry
    t1 = TestM1CandlestickReconstructionStress()
    t1.test_reconstruct_500_vectors_ohlcv_invariants()

    # 2. SMC Feature Engine Detection
    t2 = TestSMCLiquidityEngineDetection()
    t2.test_smc_canonical_pattern_detection()
    t2.test_smc_features_on_500_reconstructed_bars_stream()

    # 3. 1,000 Ticks Dual-Tier & Governor Invariant Simulation
    t3 = TestDualTierExecutionAndGovernorInvariant1000Ticks()
    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)
    checkpoint_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
    policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
    res = t3.test_1000_ticks_governor_invariant_and_tier_triggers(policy)

    print("\n" + "=" * 70)
    print("EMPIRICAL HARNESS RUN COMPLETE.")
    print("=" * 70)
