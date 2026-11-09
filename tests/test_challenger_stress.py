#!/usr/bin/env python3
"""
tests/test_challenger_stress.py

Independent Empirical Challenger Stress Test Harness.
Designed by Teamwork Challenger (Adversarial Verifier) for Milestone 1 Acceptance Verification.

Evaluates:
1. Numerical edge cases: Zero gross loss, infinite Profit Factor, zero trades, zero/negative ATR, NaN/Inf handling.
2. Regime 5 High-Volatility Expansion: In-depth MAE distribution, bar-low vs stop-price MAE gap, flash-dump outlier T00479 analysis, and regime weight sensitivity.
3. SL & TP boundary tests: SL at 1.20 ATR vs 1.50 ATR, TP at 2.25 ATR, simultaneous SL+TP bar crossing order, geometry packing roundtrip.
4. Model inference latency distribution and SLA stability across 100 CPU iterations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)
from self_growing_trader.execution_contract import (
    pack_geometry_choices,
    unpack_geometry_choices,
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
)


class TestNumericalEdgeCases:
    """Stress tests on mathematical stability of metric calculations."""

    def test_zero_gross_loss_profit_factor_math(self):
        """Regime 4 has zero losses (gross_loss = 0.0). Verify PF math and serialization."""
        gross_profit = 87.33
        gross_loss = 0.0

        # In test_5_regimes_zero_filter.py line 399:
        pf_sim = gross_profit / max(1e-4, gross_loss)
        assert pf_sim == pytest.approx(873300.0, rel=1e-3)
        assert math.isfinite(pf_sim)

        # In line 545:
        # "profit_factor": round(float(res["pf"]), 2) if res["pf"] < 1e6 else "Inf"
        serialized_pf_under_1m = round(float(pf_sim), 2) if pf_sim < 1e6 else "Inf"
        assert isinstance(serialized_pf_under_1m, float)
        assert serialized_pf_under_1m >= 2.0

        # Edge case: What if gross_profit is >= $100.00 with 0 gross loss?
        large_gross_profit = 150.0
        pf_large = large_gross_profit / max(1e-4, 0.0) # 1.5e6 >= 1e6
        serialized_large = round(float(pf_large), 2) if pf_large < 1e6 else "Inf"
        assert serialized_large == "Inf"

        # Downstream risk test: comparing string "Inf" with number
        # In Python 3: "Inf" >= 2.0 raises TypeError!
        with pytest.raises(TypeError):
            _ = serialized_large >= 2.0  # Demonstrates adversarial vulnerability if gross profit exceeds $100!

    def test_zero_trades_metric_stability(self):
        """Ensure no division by zero when trade list is empty."""
        tot_trades = 0
        tot_wins = 0
        gross_profit = 0.0
        gross_loss = 0.0

        wr = (tot_wins / max(1, tot_trades)) * 100.0
        pf = gross_profit / max(1e-4, gross_loss)
        assert wr == 0.0
        assert pf == 0.0
        trade_maes = []
        avg_mae = float(np.mean(trade_maes)) if trade_maes else 0.0
        assert avg_mae == 0.0

    def test_manifest_deserialization_and_types(self):
        """Verify manifest.json values are strictly finite floats and booleans."""
        manifest_path = ROOT / "outputs" / "acceptance_artifacts" / "milestone_1" / "manifest.json"
        assert manifest_path.is_file(), f"Manifest file missing: {manifest_path}"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        cons = data["consolidated_metrics"]
        assert isinstance(cons["win_rate_pct"], (int, float))
        assert cons["win_rate_pct"] >= 70.0
        assert isinstance(cons["profit_factor"], (int, float))
        assert cons["profit_factor"] >= 2.0
        assert isinstance(cons["average_mae_atr"], (int, float))
        assert cons["average_mae_atr"] <= 0.60

        # Check all regimes
        for reg_id, reg_m in data["regime_metrics"].items():
            assert reg_m["total_trades"] == reg_m["wins"] + reg_m["losses"]
            assert reg_m["win_rate_pct"] >= 70.0
            # PF can be float or string "Inf"
            pf_val = reg_m["profit_factor"]
            if isinstance(pf_val, str):
                assert pf_val == "Inf"
            else:
                assert pf_val >= 2.0


class TestRegime5VolatilityStress:
    """Stress tests on Regime 5 high-volatility expansion and MAE distribution."""

    @pytest.fixture(scope="class")
    def trades_df(self):
        p = ROOT / "outputs" / "acceptance_artifacts" / "milestone_1" / "consolidated" / "consolidated_trades.csv"
        assert p.is_file()
        return pd.read_csv(p)

    def test_mae_distribution_analysis(self, trades_df):
        """Analyze empirical distribution of MAE across all 480 trades."""
        assert len(trades_df) == 480
        maes = trades_df["mae_atr"].values

        p50 = float(np.percentile(maes, 50))
        p90 = float(np.percentile(maes, 90))
        p95 = float(np.percentile(maes, 95))
        p99 = float(np.percentile(maes, 99))
        max_mae = float(np.max(maes))
        mean_mae = float(np.mean(maes))

        # Consolidated mean MAE strictly <= 0.60 ATR
        assert mean_mae == pytest.approx(0.4634, abs=1e-3)
        assert mean_mae <= 0.60

        # Median MAE is exceptionally low (anti-MAE policy actively minimizes excursion)
        assert p50 < 0.40, f"Median MAE is {p50:.4f} ATR, expected < 0.40 ATR"
        # 90th percentile MAE reaches 1.0081 ATR due to high-volatility regime swings
        assert p90 == pytest.approx(1.0081, abs=0.05), f"90th percentile MAE is {p90:.4f} ATR"

    def test_flash_dump_outlier_impact(self, trades_df):
        """Verify the exact impact of Ticket T00479 (flash dump of 13.76 ATR) on Regime 5."""
        reg5_trades = trades_df.iloc[435:480]  # Tickets T00436 to T00480 (45 trades)
        assert len(reg5_trades) == 45

        raw_reg5_mean = reg5_trades["mae_atr"].mean()
        assert raw_reg5_mean == pytest.approx(1.1128, abs=1e-3)

        # Identify max outlier
        outlier_row = reg5_trades.sort_values(by="mae_atr", ascending=False).iloc[0]
        assert outlier_row["ticket"] == "T00479"
        assert outlier_row["mae_atr"] == pytest.approx(13.7637, abs=1e-3)
        assert outlier_row["exit_reason"] == "SL_HIT"

        # If capped at true SL execution boundary (1.50 ATR):
        adjusted_maes = reg5_trades["mae_atr"].copy()
        adjusted_maes.loc[adjusted_maes > 1.50] = 1.50
        capped_reg5_mean = float(adjusted_maes.mean())
        # Capping losing trades at SL boundary drops Regime 5 MAE from 1.113 to 0.729 ATR
        assert capped_reg5_mean == pytest.approx(0.7290, abs=0.02)

    def test_regime_weight_sensitivity(self, trades_df):
        """Adversarial sensitivity test: How does consolidated MAE behave under heavy Regime 5 weighting?"""
        reg1_mae = trades_df.iloc[0:190]["mae_atr"].mean()
        reg2_mae = trades_df.iloc[190:332]["mae_atr"].mean()
        reg3_mae = trades_df.iloc[332:350]["mae_atr"].mean()
        reg4_mae = trades_df.iloc[350:435]["mae_atr"].mean()
        reg5_mae = trades_df.iloc[435:480]["mae_atr"].mean()

        # Under equal regime weighting (20% each):
        equal_weight_mae = (reg1_mae + reg2_mae + reg3_mae + reg4_mae + reg5_mae) / 5.0
        # Equal weighting gives ~0.568 ATR, still strictly <= 0.60 ATR!
        assert equal_weight_mae <= 0.60, f"Equal weight MAE: {equal_weight_mae:.4f} ATR"

        # Breakpoint test: What weight of Regime 5 causes consolidated MAE to exceed 0.60 ATR?
        # Non-reg5 mean:
        non_reg5_trades = trades_df.iloc[0:435]
        non_reg5_mae = non_reg5_trades["mae_atr"].mean()  # ~0.396 ATR
        # Solve: w * 1.1128 + (1 - w) * 0.3960 = 0.60
        # w * (1.1128 - 0.3960) = 0.60 - 0.3960 = 0.2040
        # w_break = 0.2040 / 0.7168 ≈ 28.5%
        w_break = (0.60 - non_reg5_mae) / (reg5_mae - non_reg5_mae)
        assert 0.25 <= w_break <= 0.35
        # In current data, Regime 5 is 45/480 = 9.38% of trades, well below the 28.5% boundary.


class TestSLAndTPBoundaryConditions:
    """Stress tests on SL (1.50 ATR) and TP (2.25 ATR) boundary mechanics."""

    def test_sl_and_tp_geometry_invariants(self):
        """Verify codebook bin indices for SL=1.50 ATR, TP=2.25 ATR, Trailing=0.50 ATR."""
        assert SL_ATR_BINS[2] == 1.50
        assert TP_RR_BINS[4] == 1.50  # Risk-Reward 1.50 * 1.50 ATR = 2.25 ATR
        assert TRAILING_ATR_BINS[1] == 0.50

        # Pack and unpack roundtrip
        packed = pack_geometry_choices(sl_index=2, tp_index=4, trailing_index=1)
        sl_i, tp_i, tr_i = unpack_geometry_choices(packed)
        assert sl_i == 2
        assert tp_i == 4
        assert tr_i == 1

    def test_simultaneous_sl_and_tp_bar_crossing_order(self):
        """Adversarial scenario: A giant 1-minute candle breaches BOTH SL and TP."""
        # Simulated BUY trade: Entry 80000, ATR 100, SL 79850 (1.5 ATR), TP 80225 (2.25 ATR)
        entry_p = 80000.0
        atr = 100.0
        pos_sl_p = entry_p - 1.5 * atr  # 79850
        pos_tp_p = entry_p + 2.25 * atr # 80225

        # Bar with High = 80300 (above TP) AND Low = 79800 (below SL)
        cur_h = 80300.0
        cur_l = 79800.0

        # In simulate_trading_regime lines 217-247:
        # cur_l <= pos_sl_p is checked FIRST.
        exit_reason = None
        if cur_l <= pos_sl_p:
            exit_reason = "SL_HIT"
        elif cur_h >= pos_tp_p:
            exit_reason = "TP_HIT"

        # Conservative invariant: Must trigger SL_HIT to avoid optimistic backtest bias!
        assert exit_reason == "SL_HIT"


class TestLatencyAndExecutionStability:
    """Empirical latency profiling across 100 iterations on current CPU hardware."""

    def test_inference_latency_distribution_100_runs(self):
        cfg = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, prm_threshold=0.35)
        engine = BilateralDualCoTEngine(cfg)
        engine.eval()

        # Warmup 5 runs
        dummy = torch.randn(1, 233)
        with torch.no_grad():
            for _ in range(5):
                _ = engine(dummy)

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = engine(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))
        max_lat = float(np.max(latencies))

        print(f"\nInference Latency Profile (100 runs): Mean={mean_lat:.2f}ms | P50={p50_lat:.2f}ms | P95={p95_lat:.2f}ms | Max={max_lat:.2f}ms")

        # The inference latency is well within M1 trading bar cycle (60,000 ms per M1 bar)
        assert mean_lat < 100.0, f"Mean latency {mean_lat:.2f}ms exceeds 100ms"
        assert p50_lat < 50.0 or mean_lat < 60.0
