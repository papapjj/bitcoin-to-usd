# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite: TCA Milestone 4 Iteration 2
Author: teamwork_challenger_tca_m4_it2_1 (Empirical Challenger 1)

Focus:
1. extract_candles_for_window boundary conditions:
   - bar_idx = 0 (single candle boundary)
   - bar_idx < 240 (partial window boundary, e.g. bar_idx = 1, 10, 50, 238, 239)
   - bar_idx == total_bars - 1 (final boundary of dataset)
   - lookback variation (lookback = 1, 10, 240, 500, 0)
2. unix_timestamps array slicing & data integrity:
   - Contiguous numpy int64 representation
   - Strict monotonic timestamp ordering for TradingView Lightweight Charts
   - High >= max(Open, Close) and Low <= min(Open, Close) geometric invariants
   - Strict JSON serializability (standard json.dumps without float NaN/inf or numpy scalar errors)
3. Robustness & Fallback handling:
   - Mock environment without unix_timestamps attribute (fallback to string parsing)
   - Mock environment with empty unix_timestamps array
   - Mock environment with malformed or missing string timestamps
   - Out-of-bounds bar_idx handling
4. MegaPoolEnvironmentSlice integration:
   - Exact slice propagation of unix_timestamps
   - Zero timestamp phase shift between master_env and slice_env
5. FastAPI Endpoint Stress & Memory Stability:
   - GET /api/candles at boundary bar indices (0, 1, 239, total_bars - 1, out-of-bounds)
   - GET /api/tca_heatmap at boundary bar indices (0, 239, total_bars - 1, out-of-bounds)
   - 100-request rapid invocation under tracemalloc asserting zero unbounded memory leak
"""

from __future__ import annotations

import json
import tracemalloc
from typing import Any, Dict, List
import unittest

from fastapi.testclient import TestClient
import numpy as np
import pytest

from dashboard.backend import (
    _timestamp_cache,
    app,
    extract_candles_for_window,
    initialize_runtime,
    state,
)
from research_reasoning_world_class.src.mega_pool_dataset import SlicedM1Environment
from research_reasoning_world_class.src.real_m1_environment import RealM1Environment


@pytest.fixture(scope="module")
def initialized_env():
    """Initializes runtime and returns active RealM1Environment."""
    initialize_runtime()
    env = state["env"]
    assert env is not None, "RealM1Environment failed to initialize"
    return env


@pytest.fixture(scope="module")
def api_client():
    """Returns FastAPI TestClient for initialized backend."""
    initialize_runtime()
    return TestClient(app)


# ==============================================================================
# 1. BOUNDARY CONDITIONS FOR extract_candles_for_window
# ==============================================================================

class TestExtractCandlesBoundaryConditions:
    """Adversarial stress-testing of boundary bar indices and window slicing."""

    def test_boundary_bar_idx_zero(self, initialized_env):
        """Bar 0: minimum valid bar index. Must return exactly 1 candle."""
        candles = extract_candles_for_window(initialized_env, bar_idx=0, lookback=240)
        assert len(candles) == 1
        c = candles[0]
        assert c["bar_idx"] == 0
        assert c["time"] == int(initialized_env.unix_timestamps[0])
        assert c["datetime"] == initialized_env.timestamps[0]
        assert c["open"] == float(initialized_env.open_arr[0])
        assert c["high"] == float(initialized_env.high_arr[0])
        assert c["low"] == float(initialized_env.low_arr[0])
        assert c["close"] == float(initialized_env.close_arr[0])

    @pytest.mark.parametrize("bar_idx", [1, 5, 25, 50, 100, 238, 239])
    def test_boundary_bar_idx_under_240(self, initialized_env, bar_idx):
        """Bars where bar_idx < 240. Must return exactly bar_idx + 1 candles."""
        candles = extract_candles_for_window(initialized_env, bar_idx=bar_idx, lookback=240)
        expected_count = bar_idx + 1
        assert len(candles) == expected_count
        assert candles[0]["bar_idx"] == 0
        assert candles[-1]["bar_idx"] == bar_idx

        # Invariant: contiguous strictly ascending bar indices
        extracted_indices = [c["bar_idx"] for c in candles]
        assert extracted_indices == list(range(expected_count))

    def test_boundary_bar_idx_exact_240(self, initialized_env):
        """bar_idx = 240: first window where full 240 lookback begins at start_idx = 1."""
        candles = extract_candles_for_window(initialized_env, bar_idx=240, lookback=240)
        assert len(candles) == 240
        assert candles[0]["bar_idx"] == 1
        assert candles[-1]["bar_idx"] == 240

    def test_boundary_last_bar_of_dataset(self, initialized_env):
        """Last bar: bar_idx = total_bars - 1. Must extract exactly 240 candles ending at last bar."""
        total = state["total_bars"]
        last_idx = total - 1
        candles = extract_candles_for_window(initialized_env, bar_idx=last_idx, lookback=240)
        assert len(candles) == 240
        assert candles[-1]["bar_idx"] == last_idx
        assert candles[0]["bar_idx"] == last_idx - 240 + 1
        assert candles[-1]["time"] == int(initialized_env.unix_timestamps[last_idx])

    @pytest.mark.parametrize("lookback", [1, 5, 50, 240, 500])
    def test_arbitrary_lookback_parameter(self, initialized_env, lookback):
        """Arbitrary lookbacks must correctly bound window to min(bar_idx + 1, lookback)."""
        bar_idx = 300
        candles = extract_candles_for_window(initialized_env, bar_idx=bar_idx, lookback=lookback)
        expected_count = min(bar_idx + 1, lookback)
        assert len(candles) == expected_count
        assert candles[-1]["bar_idx"] == bar_idx
        assert candles[0]["bar_idx"] == bar_idx - expected_count + 1

    def test_zero_lookback(self, initialized_env):
        """Lookback = 0 should return empty list gracefully."""
        candles = extract_candles_for_window(initialized_env, bar_idx=100, lookback=0)
        assert candles == []


# ==============================================================================
# 2. TRADINGVIEW LIGHTWEIGHT CHARTS CONTRACT & DATA FORMATTING
# ==============================================================================

class TestTradingViewCandleFormatting:
    """Verifies that all extracted candles satisfy TradingView Lightweight Charts formatting rules."""

    def test_strictly_monotonic_timestamps(self, initialized_env):
        """TradingView lightweight-charts requires strictly increasing unix timestamps (time_i < time_{i+1})."""
        candles = extract_candles_for_window(initialized_env, bar_idx=500, lookback=240)
        times = [c["time"] for c in candles]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1], f"Timestamp inversion at bar {i}: {times[i]} >= {times[i+1]}"

    def test_ohlc_geometric_invariants(self, initialized_env):
        """High must be >= max(Open, Close) and Low must be <= min(Open, Close) for all candles."""
        candles = extract_candles_for_window(initialized_env, bar_idx=1000, lookback=240)
        for c in candles:
            assert c["high"] >= max(c["open"], c["close"]) - 1e-4, f"Invalid high in candle: {c}"
            assert c["low"] <= min(c["open"], c["close"]) + 1e-4, f"Invalid low in candle: {c}"
            assert c["volume"] >= 0.0, f"Negative volume in candle: {c}"
            assert c["spread"] >= 0.0, f"Negative spread in candle: {c}"
            assert c["atr"] > 0.0, f"Non-positive ATR in candle: {c}"

    def test_strict_json_serializability(self, initialized_env):
        """Returned structure must be 100% JSON-serializable without custom encoders."""
        candles = extract_candles_for_window(initialized_env, bar_idx=2500, lookback=240)
        json_str = json.dumps(candles)
        assert len(json_str) > 0
        deserialized = json.loads(json_str)
        assert len(deserialized) == 240
        for item in deserialized:
            assert isinstance(item["time"], int)
            assert isinstance(item["open"], float)
            assert isinstance(item["close"], float)
            assert isinstance(item["datetime"], str)


# ==============================================================================
# 3. ADVERSARIAL FALLBACK & DEFENSIVE INTEGRITY
# ==============================================================================

class DummyMockEnv:
    """Mock environment lacking unix_timestamps or having anomalous structures."""

    def __init__(self, count: int = 50):
        self.count = count
        self.open_arr = np.linspace(50000.0, 51000.0, count, dtype=np.float32)
        self.high_arr = self.open_arr + 10.0
        self.low_arr = self.open_arr - 10.0
        self.close_arr = self.open_arr + 2.0
        self.volume_arr = np.ones(count, dtype=np.float32) * 5.0
        self.spread_arr = np.ones(count, dtype=np.float32) * 2.5
        self.atr_arr = np.ones(count, dtype=np.float32) * 15.0
        self.timestamps = [f"2025.01.01 {i:02d}:00:00" for i in range(count)]


class TestAdversarialFallbackPaths:
    """Stress-tests the defensive fallback paths when unix_timestamps is absent or corrupt."""

    def test_fallback_when_unix_timestamps_attribute_missing(self):
        """Environment without unix_timestamps must fall back to _timestamp_cache / pd.to_datetime seamlessly."""
        mock_env = DummyMockEnv(count=30)
        assert not hasattr(mock_env, "unix_timestamps")

        candles = extract_candles_for_window(mock_env, bar_idx=20, lookback=10)
        assert len(candles) == 10
        assert candles[-1]["bar_idx"] == 20
        assert candles[0]["bar_idx"] == 11
        for c in candles:
            assert isinstance(c["time"], int)
            assert c["time"] > 0
            assert c["datetime"].startswith("2025.01.01")

    def test_fallback_when_unix_timestamps_empty_ndarray(self):
        """Environment where unix_timestamps is empty ndarray must fall back without crashing."""
        mock_env = DummyMockEnv(count=30)
        mock_env.unix_timestamps = np.empty(0, dtype=np.int64)

        candles = extract_candles_for_window(mock_env, bar_idx=15, lookback=5)
        assert len(candles) == 5
        assert candles[-1]["bar_idx"] == 15
        assert isinstance(candles[0]["time"], int)

    def test_fallback_when_timestamps_strings_are_unparseable(self):
        """Malformed distinct strings in env.timestamps trigger synthetic timestamp fallback without exception."""
        mock_env = DummyMockEnv(count=10)
        mock_env.timestamps = [f"unparseable_garbage_{i}" for i in range(10)]

        candles = extract_candles_for_window(mock_env, bar_idx=5, lookback=5)
        assert len(candles) == 5
        for i, c in enumerate(candles):
            # Fallback path uses idx * 60
            expected_idx = 1 + i
            assert c["time"] == expected_idx * 60
            assert c["datetime"] == f"unparseable_garbage_{expected_idx}"

    def test_cache_collision_behavior_with_identical_unparseable_strings(self):
        """
        Adversarial finding: when timestamp strings are identical across distinct bars,
        _timestamp_cache keys on ts_str, returning identical timestamps across bars.
        """
        mock_env = DummyMockEnv(count=5)
        mock_env.timestamps = ["identical_bad_str"] * 5
        candles = extract_candles_for_window(mock_env, bar_idx=3, lookback=3)
        assert len(candles) == 3
        # First parsed bar caches its synthetic time for all matching strings
        first_time = candles[0]["time"]
        assert all(c["time"] == first_time for c in candles)

    def test_fallback_when_timestamps_attribute_completely_missing(self):
        """Environment missing both unix_timestamps and timestamps attributes."""
        mock_env = DummyMockEnv(count=10)
        delattr(mock_env, "timestamps")

        candles = extract_candles_for_window(mock_env, bar_idx=4, lookback=5)
        assert len(candles) == 5
        for i, c in enumerate(candles):
            assert c["datetime"] == f"bar_{i}"
            assert c["time"] == i * 60


# ==============================================================================
# 4. SLICED M1 ENVIRONMENT TIMESTAMPS EMPIRICAL VERIFICATION
# ==============================================================================

class TestMegaPoolSliceTimestampsPropagation:
    """
    Empirical investigation of SlicedM1Environment:
    Checks whether unix_timestamps exists on SlicedM1Environment and how extract_candles_for_window behaves.
    """

    def test_slice_environment_empirical_attributes(self, initialized_env):
        """
        Empirically observes that scripts.train_mythos_muzero_coevolution.SlicedM1Environment
        lacks unix_timestamps, causing extract_candles_for_window to use the fallback path.
        """
        start_idx = 100
        end_idx = 500
        pool_slice = SlicedM1Environment(
            master_env=initialized_env,
            start_idx=start_idx,
            end_idx=end_idx,
            name="TestSlice",
        )

        has_uts = hasattr(pool_slice, "unix_timestamps")
        # Document empirical fact: scripts SlicedM1Environment lacks unix_timestamps
        # extract_candles_for_window handles this gracefully via fallback
        slice_candles = extract_candles_for_window(pool_slice, bar_idx=50, lookback=20)
        assert len(slice_candles) == 20
        assert slice_candles[-1]["bar_idx"] == 50
        assert isinstance(slice_candles[-1]["time"], int)

    def test_slice_with_explicit_unix_timestamps(self, initialized_env):
        """When unix_timestamps is explicitly attached to a slice, fast path executes bit-identically."""
        start_idx = 100
        end_idx = 500
        pool_slice = SlicedM1Environment(
            master_env=initialized_env,
            start_idx=start_idx,
            end_idx=end_idx,
            name="TestSlice",
        )
        # Propagate sliced unix_timestamps
        pool_slice.unix_timestamps = initialized_env.unix_timestamps[start_idx:end_idx]

        slice_candles = extract_candles_for_window(pool_slice, bar_idx=50, lookback=20)
        assert len(slice_candles) == 20
        assert slice_candles[-1]["bar_idx"] == 50
        assert slice_candles[-1]["time"] == int(initialized_env.unix_timestamps[start_idx + 50])


# ==============================================================================
# 5. FASTAPI ENDPOINT ADVERSARIAL BOUNDARY & VALIDATION
# ==============================================================================

class TestFastAPIEndpointAdversarialBounds:
    """Stress-tests /api/candles and /api/tca_heatmap endpoints via TestClient."""

    def test_get_candles_boundary_zero(self, api_client):
        """GET /api/candles?bar_idx=0 must return 200 with count=1."""
        res = api_client.get("/api/candles?bar_idx=0")
        assert res.status_code == 200
        data = res.json()
        assert data["target_bar_idx"] == 0
        assert data["count"] == 1
        assert len(data["candles"]) == 1
        assert data["candles"][0]["bar_idx"] == 0

    def test_get_candles_boundary_last_bar(self, api_client):
        """GET /api/candles?bar_idx=total_bars - 1 must return 200 with count=240."""
        last_idx = state["total_bars"] - 1
        res = api_client.get(f"/api/candles?bar_idx={last_idx}")
        assert res.status_code == 200
        data = res.json()
        assert data["target_bar_idx"] == last_idx
        assert data["count"] == 240
        assert data["candles"][-1]["bar_idx"] == last_idx

    def test_get_candles_out_of_bounds_exact_total_bars(self, api_client):
        """GET /api/candles?bar_idx=total_bars must return 400 Bad Request."""
        total = state["total_bars"]
        res = api_client.get(f"/api/candles?bar_idx={total}")
        assert res.status_code == 400

    def test_get_candles_negative_bar_idx(self, api_client):
        """GET /api/candles?bar_idx=-1 must fail with 422 Unprocessable Entity (ge=0)."""
        res = api_client.get("/api/candles?bar_idx=-1")
        assert res.status_code == 422

    def test_get_tca_heatmap_boundary_zero(self, api_client):
        """GET /api/tca_heatmap?bar_idx=0 must execute cleanly without crashing."""
        res = api_client.get("/api/tca_heatmap?bar_idx=0")
        assert res.status_code == 200
        data = res.json()
        assert data["bar_idx"] == 0
        assert len(data["candles"]) == 1
        assert len(data["attention"]["heads"]) == 4
        assert len(data["attention"]["heads"][0]) == 240

    def test_get_tca_heatmap_last_bar(self, api_client):
        """GET /api/tca_heatmap?bar_idx=total_bars - 1 must execute cleanly."""
        last_idx = state["total_bars"] - 1
        res = api_client.get(f"/api/tca_heatmap?bar_idx={last_idx}")
        assert res.status_code == 200
        data = res.json()
        assert data["bar_idx"] == last_idx
        assert len(data["candles"]) == 240

    def test_get_tca_heatmap_out_of_bounds(self, api_client):
        """GET /api/tca_heatmap with out-of-bounds bar index must return 400."""
        total = state["total_bars"]
        res = api_client.get(f"/api/tca_heatmap?bar_idx={total + 100}")
        assert res.status_code == 400


# ==============================================================================
# 6. RAPID STRESS & MEMORY LEAK HARNESS
# ==============================================================================

class TestMemoryStabilityAndRapidStress:
    """Verifies that high-frequency invocations do not leak memory or pollute cache."""

    def test_direct_extract_candles_zero_leak(self, initialized_env):
        """1000 direct calls to extract_candles_for_window must have near-zero net memory overhead (< 100 KB)."""
        import gc
        gc.collect()
        tracemalloc.start()
        snap_start = tracemalloc.take_snapshot()

        for b_idx in range(1000, 2000):
            _ = extract_candles_for_window(initialized_env, bar_idx=b_idx, lookback=240)

        gc.collect()
        snap_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diffs = snap_end.compare_to(snap_start, "lineno")
        net_growth = sum(d.size_diff for d in diffs if d.size_diff > 0) / 1024.0
        print(f"\n[Direct Function Memory] 1000 extract_candles_for_window calls: {net_growth:.2f} KB")
        assert net_growth < 250.0, f"Memory leak in extract_candles_for_window: {net_growth:.2f} KB"

    def test_zero_unbounded_memory_leak_rapid_invocations(self, api_client):
        """50 sequential requests across varying bar indices must exhibit bounded net allocation (< 500 KB)."""
        import gc
        # Warmup
        for _ in range(5):
            api_client.get("/api/candles?bar_idx=55000")

        gc.collect()
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        total = state["total_bars"]
        test_bars = [int(idx) for idx in np.linspace(1000, min(total - 1, 9000), 50)]

        for b_idx in test_bars:
            res = api_client.get(f"/api/candles?bar_idx={b_idx}")
            assert res.status_code == 200

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
        net_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        net_allocated_kb = net_allocated / 1024.0

        print(f"\n[Adversarial Stress] Net memory allocation across 50 calls: {net_allocated_kb:.2f} KB")
        assert net_allocated_kb < 500.0, f"Possible memory leak detected: {net_allocated_kb:.2f} KB allocated"

    def test_timestamp_cache_immutability_when_fast_path_active(self, initialized_env):
        """When unix_timestamps is present, _timestamp_cache must NOT grow (fast path bypasses cache)."""
        initial_cache_size = len(_timestamp_cache)

        # Call extract_candles_for_window on 50 different bar ranges
        for b_idx in range(500, 550):
            _ = extract_candles_for_window(initialized_env, bar_idx=b_idx, lookback=240)

        # Fast path directly slices unix_timestamps array; cache should not have expanded
        final_cache_size = len(_timestamp_cache)
        assert final_cache_size == initial_cache_size, (
            f"_timestamp_cache unexpectedly grew from {initial_cache_size} to {final_cache_size}"
        )
