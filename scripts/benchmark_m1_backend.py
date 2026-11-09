#!/usr/bin/env python3
"""Measure decision-path cost and bounded graph trace behaviour.

Run from the repository root:
    PYTHONPATH=src python scripts/benchmark_m1_backend.py --iterations 100

This is a diagnostic benchmark, not an MT5 performance claim. It uses the same
119 input / 20 output contract as the action bridge and prints JSON for CI or
manual A/B records.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def measure(policy: SelfGrowingPolicy, obs: np.ndarray, iterations: int) -> dict[str, float | int | str]:
    for _ in range(5):
        policy.act(obs)
    started = time.perf_counter()
    trace = None
    for _ in range(iterations):
        trace = policy.act(obs)
    elapsed = time.perf_counter() - started
    report: dict[str, float | int | str] = {
        "backend": policy.m1_backend,
        "iterations": iterations,
        "mean_ms_per_act": 1000.0 * elapsed / max(1, iterations),
    }
    if policy.m1_backend == "graph" and trace is not None:
        m1 = policy.modules[0].m1_component
        report.update({
            "neurons": len(m1.graph.neurons),
            "edges": len(m1.graph.edges),
            "trace_edges_recorded": len(trace.active_edge_ids),
            "trace_edge_budget": int(m1.graph.trace_max_edges),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    obs = np.linspace(-1.0, 1.0, 119, dtype=float)
    legacy = SelfGrowingPolicy.seed(119, 20, 24, args.seed, m1_backend="legacy")
    graph = SelfGrowingPolicy.seed(119, 20, 24, args.seed, m1_backend="graph")
    results = [measure(legacy, obs, max(1, args.iterations)), measure(graph, obs, max(1, args.iterations))]
    results[1]["graph_to_legacy_ratio"] = float(results[1]["mean_ms_per_act"]) / max(1e-12, float(results[0]["mean_ms_per_act"]))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
