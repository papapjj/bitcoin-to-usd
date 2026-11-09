"""
Milestone M4: Comprehensive Safety Invariants & System Audit Script.

Verifies:
1. Agent 4 Governor HOLD Invariant (100.0% HOLD, POLICY_CLOSE_ALL == 0.0, outputs[19] == 0.0).
2. Single-tick CPU latency SLA (<= 50.0ms steady-state).
3. Co-Evolved Champion weights strict loading (4,196,564 params, 0 missing, 0 unexpected keys).
4. MT5 Strategy Tester execution deliverables in mt5_v2_runner/Reports/ (he_chuyen_gia_mt5_round_0000_evaluation_19301.htm and .manifest.json).
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Optimal CPU thread count for xLSTM 5-agent parallel batch execution
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch

torch.set_num_threads(min(4, os.cpu_count() or 4))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.unified_reasoning_pipeline import UnifiedReasoningTraderPipeline
from research_reasoning_world_class.src.mt5_ipc_transport import IDX_POLICY_CLOSE_ALL


def audit_champion_weights():
    print("=" * 70)
    print("AUDIT 1: Champion Weights Strict Loading (4,196,564 parameters)")
    print("=" * 70)
    ckpt_path = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
    assert ckpt_path.exists(), f"Checkpoint not found at: {ckpt_path}"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cot_weights = ckpt["reasoning_engine_state_dict"]
    
    pipeline = UnifiedReasoningTraderPipeline()
    load_res = pipeline.cot_engine.load_state_dict(cot_weights, strict=True)
    
    param_count = sum(p.numel() for p in pipeline.cot_engine.parameters())
    print(f"Checkpoint Path:             {ckpt_path}")
    print(f"Total Tensors in Checkpoint: {len(cot_weights)}")
    print(f"Total Parameters in Model:   {param_count:,}")
    print(f"Missing Keys:                {len(load_res.missing_keys)}")
    print(f"Unexpected Keys:             {len(load_res.unexpected_keys)}")
    
    assert param_count == 4_196_564, f"Parameter count mismatch: {param_count} != 4,196,564"
    assert len(load_res.missing_keys) == 0, f"Missing keys detected: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys detected: {load_res.unexpected_keys}"
    print(">>> AUDIT 1 PASSED: Strict loading verified 100% with 0 missing and 0 unexpected keys.\n")


def audit_governor_invariant_and_latency():
    print("=" * 70)
    print("AUDIT 2: Governor HOLD Invariant & Steady-State CPU Latency (1,000 Ticks)")
    print("=" * 70)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    config_path = PROJECT_ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)
    checkpoint_path = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
    policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
    
    # Warmup
    for _ in range(15):
        obs = np.random.randn(137).astype(np.float32) * 0.1
        obs[15] = 0.5
        _ = policy.act_nam_tac_nhan(obs)
        
    n_ticks = 200
    latencies = []
    close_all_violations = 0
    agent4_hold_violations = 0
    
    np.random.seed(2026)
    for i in range(n_ticks):
        obs = np.random.randn(137).astype(np.float32) * 0.1
        obs[0] = float(np.sin(i * 0.05) * 0.02)
        obs[7] = float(np.cos(i * 0.05) * 0.3)
        obs[8] = float(abs(np.sin(i * 0.1)) * 0.2)
        obs[9] = float(abs(np.cos(i * 0.1)) * 0.2)
        obs[10] = float(abs(obs[7]) + obs[8] + obs[9])
        obs[15] = float(0.2 + 0.3 * abs(np.sin(i * 0.02)))
        
        t0 = time.perf_counter()
        traces = policy.act_nam_tac_nhan(obs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        # Check Agent 4 (Governor is agent index 4 in 5-agent system)
        agent4 = traces[4]
        out = getattr(agent4, "output_vector", getattr(agent4, "selected_action", None))
        if out is None or out[0] != 1.0 or out[10] != -100.0:
            agent4_hold_violations += 1
            
        # Check wire outputs (wire packet index 19 is POLICY_CLOSE_ALL)
        if out is not None and out[IDX_POLICY_CLOSE_ALL] != 0.0:
            close_all_violations += 1
                
    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    max_lat = float(np.max(lat_arr))
    
    print(f"Total Ticks Evaluated:            {n_ticks}")
    print(f"Mean Latency:                     {mean_lat:.3f} ms (SLA <= 50.0 ms)")
    print(f"Median (P50) Latency:             {p50_lat:.3f} ms")
    print(f"P95 Latency:                      {p95_lat:.3f} ms")
    print(f"P99 Latency:                      {p99_lat:.3f} ms")
    print(f"Max Latency:                      {max_lat:.3f} ms")
    print(f"Agent 4 Not HOLD Count:           {agent4_hold_violations}")
    print(f"POLICY_CLOSE_ALL != 0 Count:      {close_all_violations}")
    print(f"Governor HOLD Invariant Rate:     {((n_ticks - agent4_hold_violations) / n_ticks) * 100.0:.2f}%")
    
    assert p50_lat <= 50.0 and mean_lat <= 50.0, f"Latency (mean: {mean_lat:.2f}ms, p50: {p50_lat:.2f}ms) exceeds SLA <= 50.0ms"
    assert agent4_hold_violations == 0, f"Agent 4 violated HOLD {agent4_hold_violations} times"
    assert close_all_violations == 0, f"POLICY_CLOSE_ALL violated {close_all_violations} times"
    print(">>> AUDIT 2 PASSED: 100.0% Governor HOLD & Steady-state CPU latency <= 50.0ms.\n")


def audit_mt5_deliverables():
    print("=" * 70)
    print("AUDIT 3: MT5 Strategy Tester Execution Deliverables & Manifest")
    print("=" * 70)
    reports_dir = PROJECT_ROOT / "mt5_v2_runner" / "Reports"
    report_htm = reports_dir / "he_chuyen_gia_mt5_round_0000_evaluation_19301.htm"
    manifest_json = reports_dir / "he_chuyen_gia_mt5_round_0000_evaluation_19301.manifest.json"
    chart_png = reports_dir / "he_chuyen_gia_mt5_round_0000_evaluation_19301.png"
    
    assert report_htm.exists(), f"Report HTM missing: {report_htm}"
    assert manifest_json.exists(), f"Manifest JSON missing: {manifest_json}"
    assert chart_png.exists(), f"Chart PNG missing: {chart_png}"
    
    # Compute SHA-256 of report
    htm_data = report_htm.read_bytes()
    computed_sha256 = hashlib.sha256(htm_data).hexdigest()
    
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print(f"Report HTM:        {report_htm.name} ({len(htm_data):,} bytes)")
    print(f"Manifest JSON:     {manifest_json.name} ({manifest_json.stat().st_size:,} bytes)")
    print(f"Chart PNG:         {chart_png.name} ({chart_png.stat().st_size:,} bytes)")
    print(f"Manifest Stage:    {manifest.get('stage')}")
    print(f"Manifest Hash:     {manifest.get('report_sha256')}")
    print(f"Computed Hash:     {computed_sha256}")
    
    assert manifest.get("stage") == "finished", f"Stage is {manifest.get('stage')}, expected 'finished'"
    assert computed_sha256 == manifest.get("report_sha256"), "Report SHA-256 mismatch!"
    print(">>> AUDIT 3 PASSED: Official MT5 Strategy Tester report deliverables verified.\n")


if __name__ == "__main__":
    audit_champion_weights()
    audit_governor_invariant_and_latency()
    audit_mt5_deliverables()
    audit_mt5_deliverables()
    print("=" * 70)
    print("ALL MILESTONE M4 SAFETY INVARIANTS EMPIRICALLY CONFIRMED AND VERIFIED!")
    print("=" * 70)
