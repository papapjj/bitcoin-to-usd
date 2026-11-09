"""Adversarial Challenge & Stress Suite — Challenger 1 (Execution & Indicators).

Rigorous empirical verification covering:
1. R1 Microstructure Friction & Timeframe:
   - Invalid geometry construction (SL=1.0 on M1, SL=-5.0, SL=0.0, extreme spread ratios > 0.15, invalid timeframes, RR violations).
   - Valid geometry verification on M5 and M15 (spread/SL < 15%).
2. R2 MT5 Offline Preflight & Synchronization Bypass:
   - Startup failure detection with corrupted logs, empty logs, multiple auth failures followed by testing started, and fatal crashes.
   - Verification that offline bypass ignores auth failures and synchronization errors but retains fatal crash detection.
   - Local cache inspection on real repository cache and missing/invalid cache paths.
3. R5 Input Normalization:
   - Direct and wrapped stress testing of _groupwise_v2 / _groupwise_mtf_v3 with NaN, Inf, extreme values (-1000, +1000),
     short vectors (0 to 137 elements), exact boundaries (0.0, 0.5, 1.0), legacy percentage (75.0),
     anti-collapse check (no -0.99 pinning), and dynamic variance >= 0.45.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import pytest

from src.self_growing_trader.config import OrderConfig
from src.self_growing_trader.input_normalization import (
    INPUT_NORMALIZATION_GROUPWISE_V2,
    INPUT_NORMALIZATION_GROUPWISE_MTF_V3,
    INPUT_NORMALIZATION_STRUCTURE_AWARE_V4,
    _CHOP_INDEX,
    _groupwise_mtf_v3,
    _groupwise_v2,
    normalize_input_vector,
)
from src.self_growing_trader.mt5_runner import (
    _detect_mt5_startup_failure,
    verify_offline_cache,
)


# ==============================================================================
# SECTION 1: R1 MICROSTRUCTURE FRICTION & TIMEFRAME ADVERSARIAL STRESS
# ==============================================================================


class TestAdversarialR1MicrostructureTimeframe:
    """Adversarial stress testing of OrderConfig geometry and timeframe contracts."""

    def test_r1_sl_one_on_m1_strictly_rejected(self) -> None:
        """Attempt SL = 1.0 on M1: empirical spread is ~0.65 ATR, giving 65% spread cost."""
        with pytest.raises(ValueError, match=r"spread/SL|fixed_sl_atr|violates|ratio"):
            OrderConfig(timeframe="M1", fixed_sl_atr=1.0, fixed_tp_atr=2.5)

    def test_r1_sl_two_point_five_on_m1_strictly_rejected(self) -> None:
        """Historical setting SL = 2.5 on M1 (prior trap) must be strictly rejected."""
        with pytest.raises(ValueError, match=r"spread/SL|fixed_sl_atr|violates|ratio"):
            OrderConfig(timeframe="M1", fixed_sl_atr=2.5, fixed_tp_atr=5.0)

    def test_r1_sl_sub_threshold_boundary_on_m1(self) -> None:
        """SL values just below 4.33 ATR threshold (e.g. 4.32, 4.329) are rejected."""
        for sl_val in (4.32, 4.329):
            with pytest.raises(ValueError, match=r"spread/SL|fixed_sl_atr|violates|ratio"):
                OrderConfig(timeframe="M1", fixed_sl_atr=sl_val, fixed_tp_atr=10.0)

    def test_r1_sl_exact_threshold_boundary_on_m1(self) -> None:
        """SL values >= 4.33 ATR (e.g. 4.34, 5.0) pass with default max_spread_cost_r=0.15."""
        cfg_boundary = OrderConfig(timeframe="M1", fixed_sl_atr=4.34, fixed_tp_atr=10.0)
        assert cfg_boundary.fixed_sl_atr == 4.34
        assert cfg_boundary.timeframe == "M1"

        cfg_robust = OrderConfig(timeframe="M1", fixed_sl_atr=5.0, fixed_tp_atr=10.0)
        assert cfg_robust.fixed_sl_atr == 5.0

    def test_r1_negative_or_zero_sl_strictly_rejected(self) -> None:
        """Negative and zero SL must raise ValueError across any timeframe."""
        for sl in (-5.0, -1.0, -0.001, 0.0):
            with pytest.raises(ValueError, match="order risk values are invalid"):
                OrderConfig(fixed_sl_atr=sl, fixed_tp_atr=10.0)
            with pytest.raises(ValueError, match="order risk values are invalid"):
                OrderConfig(timeframe="M15", fixed_sl_atr=sl, fixed_tp_atr=10.0)

    def test_r1_extreme_spread_ratios_rejected(self) -> None:
        """When max_spread_cost_r is tight (e.g. 0.10), SL=5.0 gives 0.65/5 = 13% > 10% and fails."""
        with pytest.raises(ValueError, match=r"spread/SL|max_spread_cost_r|violates"):
            OrderConfig(
                timeframe="M1",
                fixed_sl_atr=5.0,
                fixed_tp_atr=10.0,
                max_spread_cost_r=0.10,
            )

        # Invalid max_spread_cost_r <= 0.0
        with pytest.raises(ValueError, match="order risk values are invalid"):
            OrderConfig(max_spread_cost_r=0.0)
        with pytest.raises(ValueError, match="order risk values are invalid"):
            OrderConfig(max_spread_cost_r=-0.15)

    def test_r1_invalid_timeframe_strings_strictly_rejected(self) -> None:
        """OrderConfig must reject non-whitelisted timeframe strings."""
        invalid_tfs = ["M2", "TICK", "H2", "D1", "W1", "MN1", "", "   ", "M0", "SEC"]
        for tf in invalid_tfs:
            with pytest.raises(ValueError, match="timeframe .* is invalid"):
                OrderConfig(timeframe=tf)

    def test_r1_risk_reward_ratio_strictly_enforced(self) -> None:
        """TP / SL < technical_min_rr (default 1.2) must be strictly rejected."""
        # SL = 5.0, TP = 5.0 (RR = 1.0 < 1.2)
        with pytest.raises(ValueError, match="order risk values are invalid"):
            OrderConfig(timeframe="M1", fixed_sl_atr=5.0, fixed_tp_atr=5.0)

        # Zero or negative TP
        with pytest.raises(ValueError, match="order risk values are invalid"):
            OrderConfig(timeframe="M1", fixed_sl_atr=5.0, fixed_tp_atr=0.0)
        with pytest.raises(ValueError, match="order risk values are invalid"):
            OrderConfig(timeframe="M1", fixed_sl_atr=5.0, fixed_tp_atr=-10.0)

    def test_r1_m5_geometries_pass_spread_ratio_check(self) -> None:
        """M5 timeframe has ~0.30 ATR spread. Geometries with SL=2.5..5.0 pass < 15% ratio."""
        for sl, tp in ((2.5, 5.0), (3.0, 6.0), (5.0, 10.0)):
            cfg = OrderConfig(timeframe="M5", fixed_sl_atr=sl, fixed_tp_atr=tp)
            assert cfg.timeframe == "M5"
            assert cfg.fixed_sl_atr == sl
            # Theoretical M5 spread ratio = 0.30 / sl <= 0.30 / 2.5 = 12.0% < 15%
            theoretical_m5_ratio = 0.30 / sl
            assert theoretical_m5_ratio <= 0.15

    def test_r1_m15_geometries_pass_spread_ratio_check(self) -> None:
        """M15 timeframe has ~0.168 ATR spread. Geometries with SL=2.0..3.0 pass < 15% ratio."""
        for sl, tp in ((2.0, 4.0), (2.5, 5.0), (3.0, 6.0)):
            cfg = OrderConfig(timeframe="M15", fixed_sl_atr=sl, fixed_tp_atr=tp)
            assert cfg.timeframe == "M15"
            assert cfg.fixed_sl_atr == sl
            # Theoretical M15 spread ratio = 0.168 / sl <= 0.168 / 2.0 = 8.4% < 15%
            theoretical_m15_ratio = 0.168 / sl
            assert theoretical_m15_ratio <= 0.15

    def test_r1_timeframe_case_and_whitespace_normalization(self) -> None:
        """Case insensitivity and whitespace trimming for timeframes."""
        assert OrderConfig(timeframe="m5").timeframe == "M5"
        assert OrderConfig(timeframe="  m15  ").timeframe == "M15"
        assert OrderConfig(timeframe="H1").timeframe == "H1"


# ==============================================================================
# SECTION 2: R2 MT5 OFFLINE PREFLIGHT & SYNCHRONIZATION BYPASS ADVERSARIAL STRESS
# ==============================================================================


class TestAdversarialR2MT5OfflinePreflight:
    """Adversarial stress testing of MT5 startup failure detection and cache verification."""

    def test_r2_detect_mt5_startup_failure_corrupted_binary_logs(
        self, tmp_path: Path
    ) -> None:
        """Startup failure detector handles random/corrupted binary bytes without throwing."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)

        # Corrupted non-UTF8/non-UTF16 byte sequences
        (log_dir / "corrupted_1.log").write_bytes(b"\x00\xff\xfe\xaa\xbb\xcc\xdd\xee\xff")
        (log_dir / "corrupted_2.log").write_bytes(b"\x80\x81\x82\x83\x84\x85\x86\x87")
        (log_dir / "null_bytes.log").write_bytes(b"\x00" * 1024)

        result_online = _detect_mt5_startup_failure(tmp_path, offline_preflight=False)
        assert result_online is None

        result_offline = _detect_mt5_startup_failure(tmp_path, offline_preflight=True)
        assert result_offline is None

    def test_r2_detect_mt5_startup_failure_empty_or_whitespace_logs(
        self, tmp_path: Path
    ) -> None:
        """Zero-byte logs or whitespace-only logs return None cleanly."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "empty1.log").write_text("", encoding="utf-8")
        (log_dir / "empty2.log").write_text("   \n\n   \t\n", encoding="utf-8")

        assert _detect_mt5_startup_failure(tmp_path, offline_preflight=False) is None
        assert _detect_mt5_startup_failure(tmp_path, offline_preflight=True) is None

    def test_r2_multiple_auth_failures_followed_by_testing_started(
        self, tmp_path: Path
    ) -> None:
        """Auth failures followed by automatic testing started are marked successful."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        lines = [
            f"2026.10.09 10:00:{i:02d} Network authorization on Exness-MT5Real8 failed (Invalid account)\n"
            for i in range(50)
        ]
        # Followed by success marker
        lines.append("2026.10.09 10:01:00 Tester automatic testing started\n")
        lines.append("2026.10.09 10:01:01 Tester testing started on 2026.01.01\n")

        (log_dir / "terminal.log").write_text("".join(lines), encoding="utf-8")

        # In both modes, newer success line overrides preceding auth failure
        assert _detect_mt5_startup_failure(tmp_path, offline_preflight=False) is None
        assert _detect_mt5_startup_failure(tmp_path, offline_preflight=True) is None

    def test_r2_auth_failure_offline_bypass_vs_online_detection(
        self, tmp_path: Path
    ) -> None:
        """Terminal with only auth failures: online detects failure; offline bypasses."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "20261009.log").write_text(
            "2026.10.09 12:00:00 Network authorization on Exness-MT5Real8 failed (Invalid account)\n",
            encoding="utf-8",
        )

        # Online mode: catches auth failure
        res_online = _detect_mt5_startup_failure(tmp_path, offline_preflight=False)
        assert res_online is not None
        assert res_online["failure_type"] == "authorization_failed"
        assert res_online["reason"] == "Invalid account"

        # Offline mode: ignores auth failure cleanly
        res_offline = _detect_mt5_startup_failure(tmp_path, offline_preflight=True)
        assert res_offline is None

    def test_r2_unsynchronized_offline_bypass_vs_online_detection(
        self, tmp_path: Path
    ) -> None:
        """Terminal unsynchronized message: online detects failure; offline bypasses."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "tester.log").write_text(
            "2026.10.09 12:00:00 Tester terminal is not synchronized with the trade server before automatic testing\n",
            encoding="utf-8",
        )

        res_online = _detect_mt5_startup_failure(tmp_path, offline_preflight=False)
        assert res_online is not None
        assert res_online["failure_type"] == "tester_not_synchronized"

        res_offline = _detect_mt5_startup_failure(tmp_path, offline_preflight=True)
        assert res_offline is None

    def test_r2_offline_preflight_retains_fatal_crash_detection(
        self, tmp_path: Path
    ) -> None:
        """Fatal crash (terminal crashed, memory access violation) is NEVER masked by offline mode."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "crash.log").write_text(
            "2026.10.09 12:00:00 Network authorization on Exness-MT5Real8 failed (Invalid account)\n"
            "2026.10.09 12:00:01 Terminal crashed: memory access violation at 0x00007FFDEADBEEF\n",
            encoding="utf-8",
        )

        # Online mode detects crash
        res_online = _detect_mt5_startup_failure(tmp_path, offline_preflight=False)
        assert res_online is not None
        assert res_online["failure_type"] == "terminal_crashed"

        # Offline mode ALSO detects crash (safety critical invariant!)
        res_offline = _detect_mt5_startup_failure(tmp_path, offline_preflight=True)
        assert res_offline is not None
        assert res_offline["failure_type"] == "terminal_crashed"
        assert "memory access violation" in str(res_offline["reason"]).lower()

    def test_r2_verify_offline_cache_real_repository_data(self) -> None:
        """Empirically verify cache detection against real mt5_v2_runner/Bases in repository."""
        repo_runner = Path("mt5_v2_runner")
        if not repo_runner.is_dir():
            pytest.skip("mt5_v2_runner directory not present in current working directory")

        # Test exact broker symbol BTCUSDm
        cache_btcusdm = verify_offline_cache(repo_runner, "BTCUSDm")
        assert cache_btcusdm["available"] is True
        assert cache_btcusdm["reason"] == "cache_verified"
        assert int(cache_btcusdm["hcc_files"]) > 0
        assert int(cache_btcusdm["tkc_files"]) > 0

        # Test canonical symbol BTCUSD (without broker 'm' postfix)
        cache_btcusd = verify_offline_cache(repo_runner, "BTCUSD")
        assert cache_btcusd["available"] is True
        assert cache_btcusd["reason"] == "cache_verified"
        assert int(cache_btcusd["hcc_files"]) > 0
        assert int(cache_btcusd["tkc_files"]) > 0

    def test_r2_verify_offline_cache_missing_directory(self) -> None:
        """Non-existent runner directory returns available=False with missing_bases."""
        fake_path = Path("non_existent_runner_dir_ad982734")
        res = verify_offline_cache(fake_path, "BTCUSDm")
        assert res["available"] is False
        assert res["reason"] == "missing_bases"
        assert res["hcc_files"] == 0
        assert res["tkc_files"] == 0

    def test_r2_verify_offline_cache_empty_and_mismatched_symbols(
        self, tmp_path: Path
    ) -> None:
        """Synthetic cache directory testing: empty bases, wrong symbol, and partial files."""
        bases = tmp_path / "Bases"
        srv = bases / "TestServer"
        hist = srv / "history" / "ETHUSDm"
        hist.mkdir(parents=True)
        (hist / "2025.hcc").write_bytes(b"dummy_hcc_data")

        # Querying BTCUSDm when only ETHUSDm exists
        res_mismatch = verify_offline_cache(tmp_path, "BTCUSDm")
        assert res_mismatch["available"] is False
        assert res_mismatch["reason"] == "no_cache_files_found"

        # Querying ETHUSDm when ETHUSDm exists
        res_match = verify_offline_cache(tmp_path, "ETHUSDm")
        assert res_match["available"] is True
        assert res_match["reason"] == "cache_verified"
        assert res_match["hcc_files"] == 1
        assert res_match["tkc_files"] == 0


# ==============================================================================
# SECTION 3: R5 INPUT NORMALIZATION ADVERSARIAL STRESS
# ==============================================================================


class TestAdversarialR5InputNormalization:
    """Adversarial stress testing of _groupwise_v2, _groupwise_mtf_v3, and chop_index."""

    def test_r5_exact_boundaries_unit_interval(self) -> None:
        """Exact boundaries (0.0, 0.5, 1.0) on unit chop mapping."""
        vec = np.zeros(137, dtype=np.float64)

        vec[_CHOP_INDEX] = 0.0
        res = _groupwise_v2(vec.copy())
        assert np.isclose(res[_CHOP_INDEX], -1.0, atol=1e-6)

        vec[_CHOP_INDEX] = 0.5
        res = _groupwise_v2(vec.copy())
        assert np.isclose(res[_CHOP_INDEX], 0.0, atol=1e-6)

        vec[_CHOP_INDEX] = 1.0
        res = _groupwise_v2(vec.copy())
        assert np.isclose(res[_CHOP_INDEX], 1.0, atol=1e-6)

    def test_r5_legacy_percentage_scaling(self) -> None:
        """Legacy percentage values (0..100) auto-scale without collapsing."""
        vec = np.zeros(137, dtype=np.float64)

        # 75.0 % -> 0.75 -> 2*0.75 - 1 = +0.50
        vec[_CHOP_INDEX] = 75.0
        res = _groupwise_v2(vec.copy())
        assert np.isclose(res[_CHOP_INDEX], 0.50, atol=1e-6)

        # 50.0 % -> 0.50 -> 2*0.50 - 1 = 0.0
        vec[_CHOP_INDEX] = 50.0
        res = _groupwise_v2(vec.copy())
        assert np.isclose(res[_CHOP_INDEX], 0.0, atol=1e-6)

        # 100.0 % -> 1.00 -> +1.0
        vec[_CHOP_INDEX] = 100.0
        res = _groupwise_v2(vec.copy())
        assert np.isclose(res[_CHOP_INDEX], 1.0, atol=1e-6)

        # 25.0 % -> 0.25 -> -0.50
        vec[_CHOP_INDEX] = 25.0
        res = _groupwise_v2(vec.copy())
        assert np.isclose(res[_CHOP_INDEX], -0.50, atol=1e-6)

    def test_r5_no_minus_099_collapse(self) -> None:
        """Values across [0.05, 0.95] do NOT collapse to -0.99."""
        vec = np.zeros(137, dtype=np.float64)
        for raw in np.linspace(0.05, 0.95, 50):
            vec[_CHOP_INDEX] = raw
            res = _groupwise_v2(vec.copy())
            expected = 2.0 * raw - 1.0
            assert res[_CHOP_INDEX] > -0.95, f"Raw {raw} collapsed to {res[_CHOP_INDEX]}"
            assert np.isclose(res[_CHOP_INDEX], expected, atol=1e-4)

    def test_r5_extreme_values_clipping(self) -> None:
        """Extreme values (-1000.0, +1000.0) clip cleanly to [-1.0, 1.0] without overflow."""
        vec = np.zeros(137, dtype=np.float64)

        vec[_CHOP_INDEX] = 1000.0
        res_high = _groupwise_v2(vec.copy())
        assert res_high[_CHOP_INDEX] == 1.0

        vec[_CHOP_INDEX] = -1000.0
        res_low = _groupwise_v2(vec.copy())
        assert res_low[_CHOP_INDEX] == -1.0

        # Entire vector with +/-1000.0
        extreme_vec = np.full(137, 1000.0, dtype=np.float64)
        res_extreme = _groupwise_v2(extreme_vec)
        assert np.all(np.isfinite(res_extreme))
        assert np.all(res_extreme <= 20.0)

    def test_r5_nan_and_inf_handling(self) -> None:
        """NaN and Inf handling: normalize_input_vector rejects; _groupwise_v2 does not crash."""
        vec_nan = np.zeros(137, dtype=np.float64)
        vec_nan[_CHOP_INDEX] = np.nan

        # Public contract: normalize_input_vector strictly validates finite values
        with pytest.raises(ValueError, match="non-finite"):
            normalize_input_vector(vec_nan, mode=INPUT_NORMALIZATION_GROUPWISE_V2)

        # Low-level transform _groupwise_v2 handles NaN without raising unhandled exception
        res_nan = _groupwise_v2(vec_nan.copy())
        assert np.isnan(res_nan[_CHOP_INDEX])

        # Inf handling: normalize_input_vector rejects
        vec_inf = np.zeros(137, dtype=np.float64)
        vec_inf[_CHOP_INDEX] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            normalize_input_vector(vec_inf, mode=INPUT_NORMALIZATION_GROUPWISE_V2)

        # Low-level transform clips inf safely
        res_inf = _groupwise_v2(vec_inf.copy())
        assert np.isfinite(res_inf[_CHOP_INDEX])
        assert res_inf[_CHOP_INDEX] == 1.0

    def test_r5_short_vectors_no_crash(self) -> None:
        """Vectors shorter than _CHOP_INDEX (0 to 136 elements) normalize safely."""
        short_sizes = [0, 1, 10, 28, 29, 50, 68, 69, 76, 77, 92, 93, 94, 136]
        for sz in short_sizes:
            vec = np.ones(sz, dtype=np.float64)
            res_v2 = _groupwise_v2(vec.copy())
            assert res_v2.shape == (sz,)
            assert np.all(np.isfinite(res_v2))

            res_mtf = _groupwise_mtf_v3(vec.copy())
            assert res_mtf.shape == (sz,)
            assert np.all(np.isfinite(res_mtf))

    def test_r5_dynamic_variance_ge_045(self) -> None:
        """Chop normalization produces dynamic variance >= 0.45 across diverse inputs."""
        # Test five representative regime points: [0.0, 0.25, 0.50, 0.75, 1.00]
        # Mapping to: [-1.0, -0.5, 0.0, +0.5, +1.0]
        # Variance of this sequence: (-1)^2 + (-0.5)^2 + 0^2 + 0.5^2 + 1^2 = 2.5 / 5 = 0.50 >= 0.45
        sample_inputs = [0.0, 0.25, 0.50, 0.75, 1.00]
        normalized_outputs = []
        vec = np.zeros(137, dtype=np.float64)
        for raw in sample_inputs:
            vec[_CHOP_INDEX] = raw
            norm = _groupwise_v2(vec.copy())
            normalized_outputs.append(norm[_CHOP_INDEX])

        variance = float(np.var(normalized_outputs))
        std_dev = float(np.std(normalized_outputs))
        assert variance >= 0.45, f"Chop variance must be >= 0.45, got {variance:.4f}"
        assert std_dev >= math.sqrt(0.45), f"Chop std dev must be >= {math.sqrt(0.45):.4f}, got {std_dev:.4f}"
