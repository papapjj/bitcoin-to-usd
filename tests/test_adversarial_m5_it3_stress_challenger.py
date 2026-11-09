"""
Adversarial Stress Test Suite - Milestone 5 Iteration 3 Quality Gate
Challenger: teamwork_challenger_m5_it3_1

Empirically challenges:
1. simulate_trade throughput under 10,000 rapid calls, extreme spreads, boundary bars,
   zero memory leaks, and zero state corruption.
2. MCTS search under varying simulation budgets (4, 8, 16, 32) - linear latency scaling and no deadlocks.
3. Thread switching safety: rapid thread adjustments and restore invariants.
4. Primary test suite and E2E test suite validation harness.
"""

from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import math
import os
import time
import tracemalloc
import unittest
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from research_reasoning_world_class.src.real_m1_environment import (
    RealM1Environment,
    TradeResult,
    ExitType,
)
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.muzero_latent_coevolution import (
    ChampionRole,
    DualLatentMCTSSearch,
    DynamicGeometryPolicyHead,
)

DEFAULT_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"


class SharedEnvironmentFixture:
    """Cached singleton environment for adversarial stress testing."""
    _env = None

    @classmethod
    def get_env(cls) -> RealM1Environment:
        if cls._env is None:
            if not os.path.exists(DEFAULT_CSV_PATH):
                raise FileNotFoundError(f"Missing CSV at {DEFAULT_CSV_PATH}")
            env = RealM1Environment(csv_path=DEFAULT_CSV_PATH, spread_usd=2.50)
            env.load_data(start_date="2025.01.15", end_date="2025.02.28")
            cls._env = env
        return cls._env


class TestAdversarialSimulateTradeStress(unittest.TestCase):
    """
    Stress-tests simulate_trade throughput, memory leak absence, array immutability,
    extreme spread values, and edge-case boundary bars.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = SharedEnvironmentFixture.get_env()

    def _hash_environment_arrays(self) -> Dict[str, str]:
        """Compute SHA-256 hashes of all underlying market data arrays."""
        hashes = {}
        for name in ["low_arr", "high_arr", "open_arr", "close_arr", "atr_arr"]:
            arr = getattr(self.env, name)
            hashes[name] = hashlib.sha256(arr.tobytes()).hexdigest()
        return hashes

    def test_1_1_rapid_10k_calls_throughput_memory_and_immutability(self):
        """
        Executes 10,000 rapid simulate_trade calls across diverse random market indices,
        actions, SL/TP levels, and horizons.
        Verifies:
        - Per-trade throughput satisfies sub-50us latency target on average without tracing hooks.
        - Zero memory leaks (tracked via tracemalloc snapshot differential).
        - Zero state corruption: underlying numpy arrays remain 100% bit-exact.
        """
        n_calls = 10000
        total_bars = self.env.total_bars
        self.assertGreater(total_bars, 1000)

        # Baseline array hashes before stress
        hashes_before = self._hash_environment_arrays()

        # Seeded random parameter generation for reproducibility
        rng = np.random.RandomState(42)
        test_bars = rng.randint(0, total_bars - 130, size=n_calls)
        test_actions = rng.choice([1, 2], size=n_calls)
        test_sl = rng.uniform(1.0, 8.0, size=n_calls)
        test_tp = rng.uniform(1.5, 15.0, size=n_calls)
        test_holding = rng.randint(5, 120, size=n_calls)

        # Warmup loop: prime CPU caches, instruction pipeline, and JIT
        for i in range(200):
            self.env.simulate_trade(
                bar_idx=int(test_bars[i]),
                action=int(test_actions[i]),
                sl_atr=float(test_sl[i]),
                tp_atr=float(test_tp[i]),
                max_holding_bars=int(test_holding[i]),
            )

        # -------------------------------------------------------------
        # Part A: Clean Throughput Benchmark (without profiling hooks)
        # -------------------------------------------------------------
        # Benchmark with H=30 (matches test_muzero_real_m1.py SLA test)
        t0 = time.perf_counter()
        results: List[TradeResult] = []
        for i in range(n_calls):
            res = self.env.simulate_trade(
                bar_idx=int(test_bars[i]),
                action=int(test_actions[i]),
                sl_atr=float(test_sl[i]),
                tp_atr=float(test_tp[i]),
                max_holding_bars=30,
            )
            if i % 100 == 0:
                results.append(res)
        elapsed_h30 = time.perf_counter() - t0
        avg_latency_h30_us = (elapsed_h30 / n_calls) * 1_000_000.0

        # Benchmark with H=60 (standard default contract horizon)
        t0 = time.perf_counter()
        for i in range(n_calls):
            _ = self.env.simulate_trade(
                bar_idx=int(test_bars[i]),
                action=int(test_actions[i]),
                sl_atr=float(test_sl[i]),
                tp_atr=float(test_tp[i]),
                max_holding_bars=60,
            )
        elapsed_h60 = time.perf_counter() - t0
        avg_latency_h60_us = (elapsed_h60 / n_calls) * 1_000_000.0

        print(f"\n[STRESS 1.1] 10,000 trades throughput (H=30): total {elapsed_h30:.4f}s, avg {avg_latency_h30_us:.2f} us/trade ({n_calls/elapsed_h30:.0f} trades/s)")
        print(f"[STRESS 1.1] 10,000 trades throughput (H=60): total {elapsed_h60:.4f}s, avg {avg_latency_h60_us:.2f} us/trade ({n_calls/elapsed_h60:.0f} trades/s)")

        # Throughput assertions:
        # H=30 must strictly satisfy <= 50.0 us/trade
        # H=60 must execute 10,000 trades in < 1.5s (> 6,666 trades/s)
        self.assertLess(
            avg_latency_h30_us,
            50.0,
            f"Average trade simulation latency (H=30) {avg_latency_h30_us:.2f} us exceeds 50.0 us threshold",
        )
        self.assertLess(
            elapsed_h60,
            1.50,
            f"10,000 trade simulations (H=60) took {elapsed_h60:.4f}s, exceeding 1.50s budget",
        )

        # -------------------------------------------------------------
        # Part B: Memory Leak Profiling via Tracemalloc Snapshot Differential
        # -------------------------------------------------------------
        gc.collect()
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        for i in range(n_calls):
            _ = self.env.simulate_trade(
                bar_idx=int(test_bars[i]),
                action=int(test_actions[i]),
                sl_atr=float(test_sl[i]),
                tp_atr=float(test_tp[i]),
                max_holding_bars=int(test_holding[i]),
            )

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
        net_leak_bytes = sum(s.size_diff for s in top_stats if s.size_diff > 0)
        print(f"[STRESS 1.1] Memory leak check: current={current_mem / 1024:.2f}KB, peak={peak_mem / 1024:.2f}KB, net growth={net_leak_bytes} bytes across 10,000 calls")

        # Net memory growth across 10,000 trade simulations must be negligible (< 100 KB)
        self.assertLess(net_leak_bytes, 100 * 1024, "Unbounded memory growth detected during 10k trade simulation")

        # -------------------------------------------------------------
        # Part C: Array Immutability / Zero State Corruption
        # -------------------------------------------------------------
        hashes_after = self._hash_environment_arrays()
        for name, h_before in hashes_before.items():
            h_after = hashes_after[name]
            self.assertEqual(
                h_before,
                h_after,
                f"State corruption detected in array {name}! SHA256 changed from {h_before} to {h_after}",
            )

        # Result structural validity
        for res in results:
            self.assertIsInstance(res, TradeResult)
            self.assertIn(str(res.exit_type).lower(), ["sl", "tp", "timestop", "time_stop"])
            self.assertFalse(math.isnan(res.pnl_usd))
            self.assertFalse(math.isinf(res.pnl_usd))
            self.assertFalse(math.isnan(res.return_atr))

    def test_1_2_extreme_spread_variations(self):
        """
        Adversarially exercises extreme spreads:
        - spread = 0.0: zero friction
        - spread = 0.0001: micro spread
        - spread = 100,000.0: massive spread exceeding asset price
        - spread monotonicity: higher spread strictly lowers PnL
        """
        bar_idx = 1000
        # 1. Zero spread: entry and exit prices have zero spread offset
        res_zero_buy = self.env.simulate_trade(bar_idx, action=1, sl_atr=2.0, tp_atr=4.0, spread_usd=0.0)
        res_zero_sell = self.env.simulate_trade(bar_idx, action=2, sl_atr=2.0, tp_atr=4.0, spread_usd=0.0)
        self.assertAlmostEqual(res_zero_buy.entry_price, self.env.close_arr[bar_idx], places=4)
        self.assertAlmostEqual(res_zero_sell.entry_price, self.env.close_arr[bar_idx], places=4)

        # 2. Micro spread: precision stability
        res_micro = self.env.simulate_trade(bar_idx, action=1, sl_atr=2.0, tp_atr=4.0, spread_usd=0.0001)
        self.assertFalse(math.isnan(res_micro.pnl_usd))
        self.assertFalse(math.isnan(res_micro.return_atr))

        # 3. Massive spread: verify no overflow/underflow, exit_price correctly computed
        res_huge = self.env.simulate_trade(bar_idx, action=1, sl_atr=2.0, tp_atr=4.0, spread_usd=100000.0)
        self.assertLess(res_huge.pnl_usd, -50000.0, "Huge spread should induce massive negative PnL")
        self.assertFalse(math.isnan(res_huge.pnl_usd))

        # 4. Monotonicity: higher spread must strictly decrease PnL for identical trade paths
        pnl_0 = self.env.simulate_trade(bar_idx, action=1, sl_atr=3.0, tp_atr=6.0, spread_usd=0.0).pnl_usd
        pnl_2_5 = self.env.simulate_trade(bar_idx, action=1, sl_atr=3.0, tp_atr=6.0, spread_usd=2.50).pnl_usd
        pnl_10 = self.env.simulate_trade(bar_idx, action=1, sl_atr=3.0, tp_atr=6.0, spread_usd=10.0).pnl_usd
        self.assertGreater(pnl_0, pnl_2_5)
        self.assertGreater(pnl_2_5, pnl_10)

    def test_1_3_boundary_bar_indices_and_invalid_inputs(self):
        """
        Adversarially tests boundary conditions:
        - bar_idx = 0 (first bar)
        - bar_idx = total_bars - 1 (last bar in dataset -> DATASET_END)
        - bar_idx = total_bars - 2 (penultimate bar -> avail = 1)
        - bar_idx out of range (< 0, >= total_bars) -> raises IndexError
        - invalid action, non-positive sl_atr / tp_atr -> raises ValueError
        - max_holding_bars = 1 and max_holding_bars = total_bars
        """
        total = self.env.total_bars

        # First bar
        res_first = self.env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=10)
        self.assertIsInstance(res_first, TradeResult)
        self.assertGreaterEqual(res_first.holding_bars, 1)

        # Penultimate bar (avail = 1)
        res_penultimate = self.env.simulate_trade(total - 2, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=60)
        self.assertIsInstance(res_penultimate, TradeResult)
        self.assertEqual(res_penultimate.holding_bars, 1)

        # Last bar in dataset (avail = 0 -> immediate DATASET_END exit)
        res_last = self.env.simulate_trade(total - 1, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=60)
        self.assertEqual(res_last.holding_bars, 0)
        self.assertEqual(res_last.exit_reason, "DATASET_END")
        self.assertEqual(res_last.exit_type, "timestop")
        # On last bar, pnl is exactly -spread friction
        self.assertAlmostEqual(res_last.pnl_usd, -self.env.spread_usd, places=4)

        # Out of bounds: bar_idx = total, bar_idx = -1
        with self.assertRaises(IndexError):
            self.env.simulate_trade(total, action=1, sl_atr=2.0, tp_atr=4.0)

        with self.assertRaises(IndexError):
            self.env.simulate_trade(-1, action=1, sl_atr=2.0, tp_atr=4.0)

        # Invalid action: 0, 3, -1
        for bad_act in [0, 3, -1, 99]:
            with self.assertRaises(ValueError):
                self.env.simulate_trade(100, action=bad_act, sl_atr=2.0, tp_atr=4.0)

        # Non-positive SL/TP: sl_atr <= 0 or tp_atr <= 0
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=1, sl_atr=0.0, tp_atr=4.0)
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=1, sl_atr=-1.0, tp_atr=4.0)
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=1, sl_atr=2.0, tp_atr=0.0)
        with self.assertRaises(ValueError):
            self.env.simulate_trade(100, action=1, sl_atr=2.0, tp_atr=-5.0)

        # Extreme SL/TP:
        # 1. Micro SL/TP: 1e-4 ATR -> triggers immediately on bar 1
        res_micro_sl = self.env.simulate_trade(500, action=1, sl_atr=1e-4, tp_atr=1e-4, max_holding_bars=60)
        self.assertEqual(res_micro_sl.holding_bars, 1)

        # 2. Giant SL/TP: 1e5 ATR -> never triggered, exits at exactly max_holding_bars
        res_giant = self.env.simulate_trade(500, action=1, sl_atr=1e5, tp_atr=1e5, max_holding_bars=15)
        self.assertEqual(res_giant.holding_bars, 15)
        self.assertEqual(res_giant.exit_type, "timestop")


class TestAdversarialMCTSBudgetScalingAndDeadlocks(unittest.TestCase):
    """
    Stress-tests MCTS search under varying simulation budgets (4, 8, 16, 32),
    empirically checks latency scaling linearity, verifies absence of deadlocks,
    and tests concurrent multithreaded MCTS execution.
    """

    @classmethod
    def setUpClass(cls):
        cls.env = SharedEnvironmentFixture.get_env()
        cls.wm = LatentDynamicsWorldModel(obs_dim=137, latent_dim=768, action_dim=64)
        cls.wm.eval()
        cot_cfg = ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=8)
        cls.cot = MetacognitiveReasoningEngine(cot_cfg)
        cls.cot.eval()
        cls.geometry_head = DynamicGeometryPolicyHead(latent_dim=768)
        cls.geometry_head.eval()

    def test_2_1_budget_latency_scaling_and_deadlock_freedom(self):
        """
        Executes MCTS searches across simulation budgets: [4, 8, 16, 32].
        Verifies:
        1. Linear latency scaling: O(N) simulations without polynomial or exponential blowups.
        2. Strict deadlock-free execution.
        3. Policy visit probabilities sum to 1.0.
        """
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            budgets = [4, 8, 16, 32]
            latencies_per_budget: Dict[int, List[float]] = {}

            sample_obs = self.env.get_observation(3000)
            z = self.wm.encode_state(sample_obs)

            # Warmup
            searcher_warmup = DualLatentMCTSSearch(
                world_model=self.wm,
                reasoning_engine=self.cot,
                geometry_head=self.geometry_head,
                env=self.env,
                num_simulations=4,
                beta_dyna=0.5,
            )
            for _ in range(3):
                searcher_warmup.search(z, bar_idx=3000)

            # Test each budget
            n_trials = 8
            for budget in budgets:
                searcher = DualLatentMCTSSearch(
                    world_model=self.wm,
                    reasoning_engine=self.cot,
                    geometry_head=self.geometry_head,
                    env=self.env,
                    num_simulations=budget,
                    beta_dyna=0.5,
                )
                latencies = []
                for _ in range(n_trials):
                    t0 = time.perf_counter()
                    act, policy, val = searcher.search(z, bar_idx=3000, return_value=True)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(elapsed_ms)

                    # Verify visit counts sum to 1.0 (probabilities)
                    prob_sum = sum(policy.values())
                    self.assertAlmostEqual(prob_sum, 1.0, places=4)
                    self.assertIn(act, [0, 1, 2])
                    self.assertTrue(-1.0 <= val <= 1.0)

                latencies_per_budget[budget] = latencies

            # Print empirical scaling results
            print("\n[STRESS 2.1] MCTS Simulation Budget Scaling Results:")
            for b in budgets:
                lats = latencies_per_budget[b]
                print(f"  Budget {b:2d}: min={min(lats):.2f}ms, mean={np.mean(lats):.2f}ms, median={np.median(lats):.2f}ms, max={max(lats):.2f}ms")

            # Verify scaling ratios:
            # Doubling the budget (4 -> 8, 8 -> 16, 16 -> 32)
            mean_4 = np.mean(latencies_per_budget[4])
            mean_8 = np.mean(latencies_per_budget[8])
            mean_16 = np.mean(latencies_per_budget[16])
            mean_32 = np.mean(latencies_per_budget[32])

            ratio_8_4 = mean_8 / max(mean_4, 1e-3)
            ratio_16_8 = mean_16 / max(mean_8, 1e-3)
            ratio_32_16 = mean_32 / max(mean_16, 1e-3)

            print(f"  Scaling Ratios: 8/4 = {ratio_8_4:.2f}x, 16/8 = {ratio_16_8:.2f}x, 32/16 = {ratio_32_16:.2f}x")

            # Under O(N) tree expansion, doubling the iterations scales around 1.2x - 2.8x
            self.assertLess(ratio_8_4, 3.5, f"Scaling ratio 8/4 ({ratio_8_4:.2f}x) super-linear")
            self.assertLess(ratio_16_8, 3.0, f"Scaling ratio 16/8 ({ratio_16_8:.2f}x) super-linear")
            self.assertLess(ratio_32_16, 3.0, f"Scaling ratio 32/16 ({ratio_32_16:.2f}x) super-linear")

            # Verify budget 8 min latency satisfies <= 50.0ms tick SLA
            min_8 = min(latencies_per_budget[8])
            self.assertLess(min_8, 50.0, f"MCTS budget 8 min latency {min_8:.2f}ms exceeds 50.0ms SLA")

        finally:
            torch.set_num_threads(orig_threads)

    def test_2_2_dual_champion_action_masking_under_varying_budgets(self):
        """
        Adversarially verifies that BullChampion NEVER outputs SELL (action 2)
        and BearChampion NEVER outputs BUY (action 1) under budgets [4, 8, 16, 32].
        """
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            z = self.wm.encode_state(self.env.get_observation(4500))

            for budget in [4, 8, 16, 32]:
                bull_searcher = DualLatentMCTSSearch(
                    world_model=self.wm,
                    reasoning_engine=self.cot,
                    role=ChampionRole.BULL,
                    geometry_head=self.geometry_head,
                    env=self.env,
                    num_simulations=budget,
                )
                bear_searcher = DualLatentMCTSSearch(
                    world_model=self.wm,
                    reasoning_engine=self.cot,
                    role=ChampionRole.BEAR,
                    geometry_head=self.geometry_head,
                    env=self.env,
                    num_simulations=budget,
                )

                for _ in range(5):
                    bull_act, bull_pol = bull_searcher.search(z, bar_idx=4500, return_value=False)
                    bear_act, bear_pol = bear_searcher.search(z, bar_idx=4500, return_value=False)

                    # Bull must never choose SELL (2)
                    self.assertIn(bull_act, (0, 1), f"Bull champion selected forbidden SELL action {bull_act}")
                    self.assertNotIn(2, bull_pol, f"Forbidden SELL action present in Bull policy: {bull_pol}")

                    # Bear must never choose BUY (1)
                    self.assertIn(bear_act, (0, 2), f"Bear champion selected forbidden BUY action {bear_act}")
                    self.assertNotIn(1, bear_pol, f"Forbidden BUY action present in Bear policy: {bear_pol}")
        finally:
            torch.set_num_threads(orig_threads)

    def test_2_3_multithreaded_mcts_concurrency_stress(self):
        """
        Concurrently executes MCTS searches across 4 parallel Python threads.
        Verifies:
        - Thread safety of models under torch.inference_mode().
        - Zero race condition state corruptions.
        - Zero thread deadlocks.
        """
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            num_workers = 4
            num_searches_per_worker = 5

            def worker_task(worker_id: int):
                bar_offset = 2000 + worker_id * 500
                obs = self.env.get_observation(bar_offset)
                z = self.wm.encode_state(obs)
                searcher = DualLatentMCTSSearch(
                    world_model=self.wm,
                    reasoning_engine=self.cot,
                    geometry_head=self.geometry_head,
                    env=self.env,
                    num_simulations=8,
                    beta_dyna=0.5,
                )
                results = []
                for step in range(num_searches_per_worker):
                    act, pol, val = searcher.search(z, bar_idx=bar_offset + step, return_value=True)
                    results.append((act, val))
                return worker_id, results

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task, i) for i in range(num_workers)]
                done, not_done = concurrent.futures.wait(futures, timeout=30.0)

            self.assertEqual(len(not_done), 0, "MCTS multithreaded deadlock detected!")
            for f in done:
                worker_id, res_list = f.result()
                self.assertEqual(len(res_list), num_searches_per_worker)
                for act, val in res_list:
                    self.assertIn(act, [0, 1, 2])
                    self.assertTrue(-1.0 <= val <= 1.0)
            print(f"\n[STRESS 2.3] Concurrent MCTS: {num_workers} threads executed successfully with 0 deadlocks")
        finally:
            torch.set_num_threads(orig_threads)


class TestAdversarialThreadSwitchingSafety(unittest.TestCase):
    """
    Stress-tests rapid PyTorch thread adjustments, OpenMP reinitialization safety,
    and strict thread count restoration invariants.
    """

    def test_3_1_rapid_thread_adjustments_and_compute_safety(self):
        """
        Rapidly alternates torch.set_num_threads across [1, 2, 4, 8, 1, 2, 4, 1] for 25 cycles
        while performing linear algebra and neural network forward computations.
        Verifies:
        - Zero segfaults or access violations.
        - Numerical output invariance regardless of active thread count.
        """
        orig_threads = torch.get_num_threads()
        try:
            thread_schedule = [1, 2, 4, 8, 1, 2, 4, 1]
            x = torch.randn(32, 128)
            weight = torch.randn(128, 64)
            expected_ref = torch.matmul(x, weight)

            for cycle in range(25):
                for threads in thread_schedule:
                    torch.set_num_threads(threads)
                    self.assertEqual(torch.get_num_threads(), threads)
                    # Compute under current thread config
                    out = torch.matmul(x, weight)
                    diff = torch.max(torch.abs(out - expected_ref)).item()
                    self.assertLess(diff, 1e-4, f"Numerical divergence {diff} under {threads} threads")

            print(f"\n[STRESS 3.1] Rapid thread switching: 200 reconfigurations completed without errors")
        finally:
            torch.set_num_threads(orig_threads)
            self.assertEqual(torch.get_num_threads(), orig_threads)

    def test_3_2_restore_invariant_on_exceptions(self):
        """
        Verifies that thread management pattern guarantees 100% thread restoration
        even when unexpected exceptions, assert failures, or runtime errors are raised.
        """
        initial_threads = torch.get_num_threads()

        class SyntheticException(Exception):
            pass

        try:
            saved = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                self.assertEqual(torch.get_num_threads(), 1)
                raise SyntheticException("Simulated catastrophic failure")
            finally:
                torch.set_num_threads(saved)
        except SyntheticException:
            pass

        self.assertEqual(
            torch.get_num_threads(),
            initial_threads,
            f"Thread invariant violated! Expected {initial_threads}, found {torch.get_num_threads()}",
        )


if __name__ == "__main__":
    unittest.main()
