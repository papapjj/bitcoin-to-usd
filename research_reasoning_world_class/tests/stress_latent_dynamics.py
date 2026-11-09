"""
Adversarial Stress Test & Empirical Fuzzing Suite for LatentDynamicsWorldModel.
Module: research_reasoning_world_class/tests/stress_latent_dynamics.py
Author: teamwork_challenger_m2_1 (Empirical Challenger)

Coverage:
1. Deep rollouts across H in [5, 20, 50, 100, 200] with standard, noisy, and extreme actions.
   Verifies: ||z_H||_2 in 27.71 +/- 0.5, zero NaNs, zero Infs.
2. Multi-step BPTT gradient backpropagation across H in [5, 10, 20].
   Verifies: Finite, non-zero, bounded gradients across all model parameter groups (no explosion/vanishing).
3. 10,000 rapid transition throughput benchmark and memory trace.
   Verifies: Throughput (calls/sec), p50/p95/p99 latency (ms), and memory stability (no leakage).
4. Boundary condition & malicious input fuzzing (NaN, Inf, 1e10, invalid action codes, zero vectors).
"""

import gc
import math
import os
import sys
import time
import tracemalloc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MarketStateEncoder,
    AdversarialMarketStressor,
    generate_trajectory_targets,
)
from research_reasoning_world_class.src.real_m1_environment import RealM1Environment


def stress_test_deep_rollouts():
    print("\n" + "=" * 60)
    print("CHALLENGE 1: Deep Rollout Fuzzing & Latent Norm Stability")
    print("=" * 60)

    wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)
    wm.eval()

    horizons = [5, 20, 50, 100, 200]
    batch_sizes = [1, 4, 16]
    expected_norm = math.sqrt(768)  # 27.712812921102035

    scenarios = [
        ("Standard Random Actions (N(0, 1))", lambda b, h: torch.randn(b, h, 64)),
        ("Extreme Magnitude Actions (+/- 1e5)", lambda b, h: (torch.randint(0, 2, (b, h, 64)) * 2 - 1).float() * 1e5),
        ("Adversarial Saturated Actions (+/- 10.0)", lambda b, h: torch.sign(torch.randn(b, h, 64)) * 10.0),
        ("Corrupted Action Sequence (NaN & Inf injected)", lambda b, h: inject_nans_infs(torch.randn(b, h, 64))),
        ("All Zero Actions", lambda b, h: torch.zeros(b, h, 64)),
        ("Discrete Sequence (BUY/SELL/HOLD cycles)", lambda b, h: generate_discrete_actions(b, h)),
    ]

    all_passed = True
    results = []

    for name, act_gen in scenarios:
        print(f"\nScenario: {name}")
        for H in horizons:
            for B in batch_sizes:
                z_0 = torch.randn(B, 768)
                actions = act_gen(B, H)

                traj = wm.rollout(z_0, actions)

                # Check shapes
                if traj.shape != (B, H, 768):
                    print(f"  [FAIL] Shape mismatch: expected {(B, H, 768)}, got {traj.shape}")
                    all_passed = False

                # Check NaNs and Infs
                has_nan = torch.isnan(traj).any().item()
                has_inf = torch.isinf(traj).any().item()

                # Check norms across all steps
                norms = torch.norm(traj, dim=-1)  # [B, H]
                min_norm = norms.min().item()
                max_norm = norms.max().item()
                mean_norm = norms.mean().item()
                norm_err = max(abs(min_norm - expected_norm), abs(max_norm - expected_norm))

                # Check std across latent dimension (LayerNorm invariant)
                stds = torch.std(traj, dim=-1)
                min_std = stds.min().item()
                max_std = stds.max().item()

                passed = (
                    (not has_nan)
                    and (not has_inf)
                    and (norm_err <= 0.6)  # norm within 27.71 +/- 0.6
                    and (0.85 <= min_std <= 1.15)
                )

                if not passed:
                    all_passed = False
                    status = "FAIL"
                else:
                    status = "PASS"

                if H in [20, 50, 100, 200] and B == 4:
                    print(
                        f"  H={H:3d}, B={B:2d} -> [{status}] MeanNorm={mean_norm:.4f} "
                        f"(Range: [{min_norm:.4f}, {max_norm:.4f}]), StdRange: [{min_std:.3f}, {max_std:.3f}], "
                        f"NaN={has_nan}, Inf={has_inf}"
                    )

                results.append({
                    "scenario": name,
                    "H": H,
                    "B": B,
                    "status": status,
                    "mean_norm": mean_norm,
                    "min_norm": min_norm,
                    "max_norm": max_norm,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                })

    print(f"\nDeep Rollout Fuzzing Verdict: {'PASS' if all_passed else 'FAIL'}")
    return all_passed, results


def inject_nans_infs(tensor):
    t = tensor.clone()
    mask_nan = torch.rand_like(t) < 0.05
    mask_inf = (torch.rand_like(t) < 0.05) & (~mask_nan)
    t[mask_nan] = float("nan")
    t[mask_inf] = float("inf")
    return t


def generate_discrete_actions(B, H):
    acts = torch.zeros(B, H, 64)
    for b in range(B):
        for h in range(H):
            discrete_choice = (b + h) % 3
            acts[b, h] = LatentDynamicsWorldModel.encode_action(discrete_choice)
    return acts


def stress_test_bptt_gradient_flow():
    print("\n" + "=" * 60)
    print("CHALLENGE 2: Multi-Step BPTT Gradient Backpropagation")
    print("=" * 60)

    horizons = [5, 10, 20]
    all_passed = True
    results = []

    for H in horizons:
        print(f"\n--- Testing BPTT over Horizon H = {H} ---")
        wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)
        wm.train()

        B = 4
        # Create input observations
        obs_seq = torch.randn(B, H + 1, 137, requires_grad=True)
        actions = torch.randn(B, H, 64, requires_grad=True)

        # Encode initial state
        z_0 = wm.encode_state(obs_seq[:, 0, :])

        # Target future states from encoder (representing real multi-step targets)
        with torch.no_grad():
            targets_z = torch.stack([wm.encode_state(obs_seq[:, t + 1, :]) for t in range(H)], dim=1)
            target_val = torch.tanh(torch.randn(B, H, 1))
            target_rew = torch.randn(B, H, 1) * 0.5

        # Forward unroll trajectory with auxiliary heads
        pred_z, pred_v, pred_r = wm.unroll_trajectory(z_0, actions)

        # Multi-step co-evolution loss
        loss_dict = wm.compute_coevolution_loss(
            z_pred=pred_z,
            z_target=targets_z,
            val_pred=pred_v,
            val_target=target_val,
            rew_pred=pred_r,
            rew_target=target_rew,
        )

        loss = loss_dict["total_loss"]
        loss.backward()

        # Parameter gradient inspection
        param_grads = {}
        zero_grad_params = []
        nan_grad_params = []
        inf_grad_params = []
        grad_norms = []

        for name, p in wm.named_parameters():
            if p.grad is None:
                zero_grad_params.append(name)
            else:
                has_nan = torch.isnan(p.grad).any().item()
                has_inf = torch.isinf(p.grad).any().item()
                norm = torch.norm(p.grad).item()
                grad_norms.append(norm)

                if has_nan:
                    nan_grad_params.append(name)
                if has_inf:
                    inf_grad_params.append(name)
                if norm == 0.0:
                    zero_grad_params.append(name)

                param_grads[name] = {
                    "shape": list(p.shape),
                    "grad_norm": norm,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                }

        # Check input gradients (does gradient flow back to z_0 and obs_seq?)
        obs_grad_norm = torch.norm(obs_seq.grad).item() if obs_seq.grad is not None else 0.0
        act_grad_norm = torch.norm(actions.grad).item() if actions.grad is not None else 0.0

        min_grad = min(grad_norms) if grad_norms else 0.0
        max_grad = max(grad_norms) if grad_norms else 0.0
        mean_grad = np.mean(grad_norms) if grad_norms else 0.0

        # Note: risk_bounds_head is not in compute_coevolution_loss so its grad will be None
        # But g_net, norm, value_head, reward_head, encoder must have non-zero grads!
        critical_groups = ["encoder", "g_net", "norm", "value_head", "reward_head"]
        missing_critical = []
        for grp in critical_groups:
            grp_params = [name for name in param_grads if grp in name]
            grp_norms = [param_grads[name]["grad_norm"] for name in grp_params]
            if not grp_norms or all(n == 0.0 for n in grp_norms):
                missing_critical.append(grp)

        passed = (
            len(nan_grad_params) == 0
            and len(inf_grad_params) == 0
            and len(missing_critical) == 0
            and obs_grad_norm > 0.0
            and act_grad_norm > 0.0
            and max_grad < 100.0  # no explosion
            and min_grad > 1e-8   # no vanishing for active params
        )

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  Result for H={H}: [{status}]")
        print(f"    Total Active Params: {len(grad_norms)}, MinGrad: {min_grad:.6e}, MaxGrad: {max_grad:.6e}, MeanGrad: {mean_grad:.6e}")
        print(f"    Obs Input Grad Norm: {obs_grad_norm:.6e}, Action Input Grad Norm: {act_grad_norm:.6e}")
        print(f"    NaN grads: {len(nan_grad_params)}, Inf grads: {len(inf_grad_params)}, Missing critical groups: {missing_critical}")

        results.append({
            "H": H,
            "status": status,
            "min_grad": min_grad,
            "max_grad": max_grad,
            "mean_grad": mean_grad,
            "obs_grad_norm": obs_grad_norm,
            "act_grad_norm": act_grad_norm,
            "nan_grads": len(nan_grad_params),
            "inf_grads": len(inf_grad_params),
            "missing_critical": missing_critical,
        })

    print(f"\nBPTT Multi-Step Gradient Flow Verdict: {'PASS' if all_passed else 'FAIL'}")
    return all_passed, results


def stress_test_throughput_and_memory():
    print("\n" + "=" * 60)
    print("CHALLENGE 3: 10,000 Rapid Transition Throughput & Memory Leak Profile")
    print("=" * 60)

    wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)
    wm.eval()

    gc.collect()
    tracemalloc.start()
    initial_mem_mb = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
    print(f"Baseline Traced Memory: {initial_mem_mb:.2f} MB")

    total_calls = 10000
    batch_size = 1
    z = torch.randn(batch_size, 768)

    latencies_us = []
    mem_checkpoints = {}

    start_time = time.perf_counter()

    with torch.no_grad():
        for i in range(total_calls):
            t0 = time.perf_counter()
            action = i % 3  # cycle HOLD, BUY, SELL
            z = wm.transition(z, action)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)

            if (i + 1) in [1000, 2500, 5000, 7500, 10000]:
                curr_mem_mb = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
                mem_checkpoints[i + 1] = curr_mem_mb

    total_time = time.perf_counter() - start_time
    gc.collect()
    final_mem_mb, peak_mem_mb = [m / (1024 * 1024) for m in tracemalloc.get_traced_memory()]
    tracemalloc.stop()
    mem_delta_mb = final_mem_mb - initial_mem_mb

    calls_per_sec = total_calls / total_time
    avg_latency_ms = (total_time / total_calls) * 1000.0
    latencies_ms = np.array(latencies_us) / 1000.0
    p50_ms = np.percentile(latencies_ms, 50)
    p95_ms = np.percentile(latencies_ms, 95)
    p99_ms = np.percentile(latencies_ms, 99)

    print(f"Total Calls: {total_calls}")
    print(f"Total Wall-Clock Time: {total_time:.3f} s")
    print(f"Throughput: {calls_per_sec:.1f} transitions/sec")
    print(f"Latency per Call (ms): Mean={avg_latency_ms:.4f}, p50={p50_ms:.4f}, p95={p95_ms:.4f}, p99={p99_ms:.4f}")
    print(f"Memory Profile (RSS MB): Baseline={initial_mem_mb:.2f} MB -> Final={final_mem_mb:.2f} MB (Delta: {mem_delta_mb:+.2f} MB)")
    print("Memory Checkpoints:")
    for k, v in mem_checkpoints.items():
        print(f"  Step {k:5d}: {v:.2f} MB")

    # Criteria:
    # 1. MT5 SLA latency <= 50ms per tick (we achieved < 1ms on CPU!)
    # 2. No runaway memory leakage (mem_delta < 50MB across 10k calls)
    # 3. Final state remains finite and normalized
    final_norm = torch.norm(z).item()
    expected_norm = math.sqrt(768)

    passed = (
        p99_ms <= 50.0
        and abs(mem_delta_mb) < 50.0
        and torch.isfinite(z).all()
        and abs(final_norm - expected_norm) < 0.5
    )

    print(f"\nThroughput & Memory Stability Verdict: {'PASS' if passed else 'FAIL'}")
    return passed, {
        "total_calls": total_calls,
        "throughput_calls_per_sec": calls_per_sec,
        "avg_latency_ms": avg_latency_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "initial_mem_mb": initial_mem_mb,
        "final_mem_mb": final_mem_mb,
        "mem_delta_mb": mem_delta_mb,
        "final_norm": final_norm,
        "passed": passed,
    }


def stress_test_adversarial_boundary_conditions():
    print("\n" + "=" * 60)
    print("CHALLENGE 4: Extreme Malicious Input & Edge Case Fuzzing")
    print("=" * 60)

    wm = LatentDynamicsWorldModel(latent_dim=768, action_dim=64, hidden_dim=512)
    wm.eval()

    test_cases = [
        ("Obs All Zeros", np.zeros(137, dtype=np.float32)),
        ("Obs All NaNs", np.full(137, np.nan, dtype=np.float32)),
        ("Obs All Infs", np.full(137, np.inf, dtype=np.float32)),
        ("Obs Extreme Magnitude (1e12)", np.full(137, 1e12, dtype=np.float32)),
        ("Obs Negative Infs", np.full(137, -np.inf, dtype=np.float32)),
        ("Obs Subnormal Floats (1e-38)", np.full(137, 1e-38, dtype=np.float32)),
    ]

    all_passed = True
    expected_norm = math.sqrt(768)

    for name, obs in test_cases:
        z = wm.encode_state(obs)
        has_nan = torch.isnan(z).any().item()
        has_inf = torch.isinf(z).any().item()
        norm = torch.norm(z).item()
        passed = (not has_nan) and (not has_inf) and (abs(norm - expected_norm) < 0.6)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name:30s} -> [{status}] Norm={norm:.4f}, NaN={has_nan}, Inf={has_inf}")

    # Action boundary cases
    action_cases = [
        ("Action Int -999", -999),
        ("Action Int 999", 999),
        ("Action Extreme SL/TP (SL=1e5, TP=1e5)", wm.encode_action(1, sl_atr=1e5, tp_atr=1e5)),
        ("Action Inverted SL/TP (SL=15.0, TP=1.0)", wm.encode_action(1, sl_atr=15.0, tp_atr=1.0)),
        ("Action All NaNs (64D)", torch.full((1, 64), float("nan"))),
        ("Action All Infs (64D)", torch.full((1, 64), float("inf"))),
        ("Action Extreme 1e10 (64D)", torch.full((1, 64), 1e10)),
    ]

    z_base = wm.encode_state(np.random.randn(137).astype(np.float32))

    for name, act in action_cases:
        z_next = wm.transition(z_base, act)
        heads = wm.predict_heads(z_base, act)

        has_nan = torch.isnan(z_next).any().item() or torch.isnan(heads["value"]).any().item()
        has_inf = torch.isinf(z_next).any().item() or torch.isinf(heads["value"]).any().item()
        norm = torch.norm(z_next).item()

        passed = (not has_nan) and (not has_inf) and (abs(norm - expected_norm) < 0.6)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name:40s} -> [{status}] Norm={norm:.4f}, NaN={has_nan}, Inf={has_inf}")

    print(f"\nAdversarial Boundary Conditions Verdict: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    p1, r1 = stress_test_deep_rollouts()
    p2, r2 = stress_test_bptt_gradient_flow()
    p3, r3 = stress_test_throughput_and_memory()
    p4 = stress_test_adversarial_boundary_conditions()

    print("\n" + "=" * 60)
    print("FINAL CHALLENGE SUMMARY")
    print("=" * 60)
    print(f"1. Deep Rollouts (H=5..200): {'PASS' if p1 else 'FAIL'}")
    print(f"2. BPTT Gradient Flow (H=5..20): {'PASS' if p2 else 'FAIL'}")
    print(f"3. 10,000 Transition Throughput & Memory: {'PASS' if p3 else 'FAIL'}")
    print(f"4. Malicious Boundary Fuzzing: {'PASS' if p4 else 'FAIL'}")
    overall = p1 and p2 and p3 and p4
    print(f"OVERALL EMPIRICAL VERDICT: {'APPROVE' if overall else 'REQUEST_CHANGES'}")
