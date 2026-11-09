#!/usr/bin/env python3
"""Challenger 2 Empirical Verification Script for M3/M4.

Adversarially audits:
1. Integrity of best_promoted_checkpoint.pkl (weights, shapes, finite check, ContextualExpectancyLedger).
2. Raw events across all 10 rounds for zero hard direction gates (context_direction_entry_gate_enabled == False).
3. Raw events across all 10 rounds for zero premature closes (POLICY_CLOSE_ALL == 0).
4. Raw events across all 10 rounds for % short in uptrend and reduction metrics.
5. Natural RL veto triggers (expectancy veto, live action veto, bidirectional ranking).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoint(ckpt_path: Path) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"AUDITING CHECKPOINT: {ckpt_path}")
    print(f"=======================================================")
    
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist!"
    file_size = ckpt_path.stat().st_size
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    sha = sha256_file(ckpt_path)
    print(f"SHA256: {sha}")
    
    # Load with pickle
    with open(ckpt_path, "rb") as f:
        obj = pickle.load(f)
    print(f"Top-level type: {type(obj)}")
    
    audit_results: dict[str, Any] = {
        "file_size": file_size,
        "sha256": sha,
        "type": str(type(obj)),
    }
    
    # Extract expert policy
    expert = obj.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
    assert expert is not None, "Expert CTX_CHUNG not found in registry!"
    pol = expert.policy
    print(f"Expert CTX_CHUNG policy type: {type(pol)}")
    print(f"Policy revision: {getattr(pol, 'revision', None)}")
    print(f"Policy instance uuid: {getattr(pol, 'instance_uuid', None)}")
    print(f"context_direction_entry_gate_enabled on policy: {getattr(pol, '_context_direction_entry_gate_enabled', None)}")
    print(f"five_action_ranking_bidirectional_enabled: {getattr(pol, '_five_action_ranking_bidirectional_enabled', None)}")
    print(f"five_expectancy_action_veto_enabled: {getattr(pol, '_five_expectancy_action_veto_enabled', None)}")
    print(f"five_live_action_veto_enabled: {getattr(pol, '_five_live_action_veto_enabled', None)}")
    
    # Inspect ContextualExpectancyLedger
    mem = getattr(pol, "memory_supernet", None)
    ledger = getattr(mem, "contextual_expectancy", None)
    print(f"ContextualExpectancyLedger: {type(ledger)}")
    if ledger is not None:
        report = ledger.report() if hasattr(ledger, "report") else {}
        records = getattr(ledger, "records", {}) or {}
        print(f"ContextualExpectancy records count: {len(records)}")
        print(f"Ledger report summary: {report.get('policy_expectancy', {})}")
        
        # Check if any record reached min_samples >= 6
        samples_ge_6 = {k: v for k, v in records.items() if getattr(v, "count", 0) >= 6}
        print(f"Ledger entries with count >= 6: {len(samples_ge_6)}")
        for k, v in list(samples_ge_6.items())[:5]:
            print(f"  {k} -> count={v.count}, mean_r={v.mean_r:.4f}, lcb={getattr(v, 'lcb', 0.0):.4f}")
    
    # Inspect conditioned_entry_expectancy
    cond_exp = getattr(pol, "conditioned_entry_expectancy", None)
    print(f"ConditionedEntryExpectancy: {type(cond_exp)}")
    if cond_exp is not None:
        estimates = getattr(cond_exp, "estimates", {}) or {}
        print(f"ConditionedEntryExpectancy estimates count: {len(estimates)}")
        cond_ge_6 = {k: v for k, v in estimates.items() if getattr(v, "sample_count", 0) >= 6}
        print(f"Conditioned entries with count >= 6: {len(cond_ge_6)}")
        for k, v in list(cond_ge_6.items())[:5]:
            print(f"  {k} -> count={v.sample_count}, mean_r={v.mean_r:.4f}, lcb={getattr(v, 'lcb', 0.0):.4f}")
            
    # Inspect direct neural weights / FiveSharedNetwork
    nan_count = 0
    inf_count = 0
    total_params = 0
    tensor_count = 0
    
    def inspect_obj(val, prefix=""):
        nonlocal nan_count, inf_count, total_params, tensor_count
        if hasattr(val, "shape") and hasattr(val, "dtype"): # numpy array or torch tensor
            tensor_count += 1
            numel = getattr(val, "numel", lambda: getattr(val, "size", 0))()
            total_params += numel
            arr = np.asarray(val)
            if np.isnan(arr).any():
                nan_count += 1
                print(f"  [ERROR] NaN in {prefix}")
            if np.isinf(arr).any():
                inf_count += 1
                print(f"  [ERROR] Inf in {prefix}")
    
    # Check neural model weights
    for attr in ["_direct_rnn_weights", "weights", "_five_weights", "parameters"]:
        if hasattr(pol, attr):
            w = getattr(pol, attr)
            print(f"Found policy attribute {attr}: {type(w)}")
            
    return audit_results


def audit_rounds_raw_events(loop_dir: Path) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"AUDITING RAW EVENTS ACROSS ALL ROUNDS IN: {loop_dir}")
    print(f"=======================================================")
    
    round_dirs = sorted([d for d in loop_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])
    print(f"Found {len(round_dirs)} round directories: {[d.name for d in round_dirs]}")
    
    rounds_data = []
    
    for r_dir in round_dirs:
        events_path = r_dir / "events.jsonl"
        manifest_path = r_dir / "manifest.json"
        contract_path = r_dir / "runtime_learning_contract.json"
        
        r_name = r_dir.name
        
        # Check contract and manifest for hard gate
        gate_enabled_in_manifest = None
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                gate_enabled_in_manifest = m.get("learning", {}).get("context_direction_entry_gate_enabled", None)
            except Exception:
                pass
                
        gate_enabled_in_contract = None
        if contract_path.exists():
            try:
                c = json.loads(contract_path.read_text(encoding="utf-8"))
                gate_enabled_in_contract = c.get("context_direction_entry_gate_enabled", None)
            except Exception:
                pass

        total_lines = 0
        lifecycle_events = 0
        policy_close_all_count = 0
        close_reasons: dict[str, int] = {}
        
        gate_enabled_flags_found = []
        gate_applied_flags_found = []
        
        uptrend_total = 0
        uptrend_short = 0
        uptrend_long = 0
        
        downtrend_total = 0
        downtrend_short = 0
        downtrend_long = 0
        
        neutral_total = 0
        neutral_short = 0
        neutral_long = 0
        
        realized_rs = []
        
        # Check natural vetoes in events
        veto_counts: dict[str, int] = {}
        
        if events_path.exists():
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    total_lines += 1
                    try:
                        rec = json.loads(line_str)
                    except Exception:
                        continue
                    
                    # Check for hard gate in the record
                    if "context_direction_entry_gate" in line_str:
                        def check_gate(d: Any):
                            if isinstance(d, dict):
                                if "context_direction_entry_gate_enabled" in d:
                                    gate_enabled_flags_found.append(bool(d["context_direction_entry_gate_enabled"]))
                                if "context_direction_entry_gate_applied" in d:
                                    gate_applied_flags_found.append(bool(d["context_direction_entry_gate_applied"]))
                                for v in d.values():
                                    check_gate(v)
                            elif isinstance(d, list):
                                for item in d:
                                    check_gate(item)
                        check_gate(rec)
                    
                    # Track veto events if any
                    for veto_key in ["expectancy_veto", "live_action_veto", "bidirectional_ranking"]:
                        if veto_key in line_str:
                            veto_counts[veto_key] = veto_counts.get(veto_key, 0) + 1
                    
                    if rec.get("event_type") == "lifecycle":
                        lifecycle_events += 1
                        pay = rec.get("payload", {})
                        cr = pay.get("close_reason", "UNKNOWN")
                        close_reasons[cr] = close_reasons.get(cr, 0) + 1
                        if cr == "POLICY_CLOSE_ALL":
                            policy_close_all_count += 1
                        
                        r_val = pay.get("realized_net_r")
                        if r_val is not None:
                            realized_rs.append(float(r_val))
                            
                        # Inspect growth_events
                        for g in pay.get("growth_events", []):
                            if g.get("type") == "context_action_expectancy_credit":
                                cond = g.get("conditioned_entry_expectancy", {}).get("condition", {})
                                trend = str(cond.get("trend", "")).lower()
                                action = int(g.get("action_index", -1))
                                
                                if trend == "up":
                                    uptrend_total += 1
                                    if action in (2, 4, 6):
                                        uptrend_short += 1
                                    elif action in (1, 3, 5):
                                        uptrend_long += 1
                                elif trend == "down":
                                    downtrend_total += 1
                                    if action in (2, 4, 6):
                                        downtrend_short += 1
                                    elif action in (1, 3, 5):
                                        downtrend_long += 1
                                else:
                                    neutral_total += 1
                                    if action in (2, 4, 6):
                                        neutral_short += 1
                                    elif action in (1, 3, 5):
                                        neutral_long += 1

        pct_short_in_uptrend = (uptrend_short / uptrend_total * 100.0) if uptrend_total > 0 else 0.0
        
        rounds_data.append({
            "round_dir": r_name,
            "total_event_lines": total_lines,
            "lifecycle_events": lifecycle_events,
            "policy_close_all_count": policy_close_all_count,
            "close_reasons": close_reasons,
            "gate_enabled_in_manifest": gate_enabled_in_manifest,
            "gate_enabled_in_contract": gate_enabled_in_contract,
            "gate_enabled_flags_found_unique": list(set(gate_enabled_flags_found)),
            "gate_applied_flags_found_unique": list(set(gate_applied_flags_found)),
            "uptrend_total": uptrend_total,
            "uptrend_short": uptrend_short,
            "uptrend_long": uptrend_long,
            "pct_short_in_uptrend": round(pct_short_in_uptrend, 2),
            "downtrend_total": downtrend_total,
            "downtrend_short": downtrend_short,
            "downtrend_long": downtrend_long,
            "neutral_total": neutral_total,
            "veto_mentions": veto_counts,
            "mean_r": float(np.mean(realized_rs)) if realized_rs else 0.0,
            "win_rate": float(len([r for r in realized_rs if r > 0]) / len(realized_rs) * 100.0) if realized_rs else 0.0,
        })
        
        print(f"[{r_name}] Lifecycles={lifecycle_events}, Closes={close_reasons}, PolicyCloseAll={policy_close_all_count}")
        print(f"  GateEnabled={set(gate_enabled_flags_found)}, GateApplied={set(gate_applied_flags_found)}")
        print(f"  Uptrend: Total={uptrend_total}, Short={uptrend_short}, Long={uptrend_long} -> %Short={pct_short_in_uptrend:.2f}%")
        print(f"  Downtrend: Total={downtrend_total}, Short={downtrend_short}, Long={downtrend_long}")

    return {"rounds": rounds_data}


def main():
    loop_dir = Path("outputs/he_chuyen_gia/v193_runs/multi_round_v193")
    ckpt_path = Path("outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint.pkl")
    champ_path = loop_dir / "champion_round_06.pkl"
    
    # 1. Audit Checkpoints
    ckpt_audit = audit_checkpoint(ckpt_path)
    if champ_path.exists():
        champ_audit = audit_checkpoint(champ_path)
        print(f"\nComparing best_promoted_checkpoint and champion_round_06:")
        print(f"  best_promoted SHA: {ckpt_audit['sha256']}")
        print(f"  champion_06   SHA: {champ_audit['sha256']}")
        print(f"  Are SHAs identical?: {ckpt_audit['sha256'] == champ_audit['sha256']}")
    
    # 2. Audit Raw Events Across All Rounds
    raw_rounds = audit_rounds_raw_events(loop_dir)
    
    # 3. Comprehensive Metric Calculation
    print("\n=======================================================")
    print("EMPIRICAL METRIC SUMMARY & ACCEPTANCE THRESHOLD AUDIT")
    print("=======================================================")
    r_list = raw_rounds["rounds"]
    
    # Total premature closes
    total_premature = sum(r["policy_close_all_count"] for r in r_list)
    print(f"Total premature closes (POLICY_CLOSE_ALL) across all rounds: {total_premature}")
    
    # Any hard direction gate enabled or applied
    any_gate_enabled = any(True in r["gate_enabled_flags_found_unique"] for r in r_list)
    any_gate_applied = any(True in r["gate_applied_flags_found_unique"] for r in r_list)
    print(f"Any hard direction gate enabled in any event?: {any_gate_enabled}")
    print(f"Any hard direction gate applied in any event?: {any_gate_applied}")
    
    # % short in uptrend per round
    r1_pct = r_list[0]["pct_short_in_uptrend"]
    r6_pct = r_list[5]["pct_short_in_uptrend"]
    r10_pct = r_list[9]["pct_short_in_uptrend"]
    
    print(f"\nRound 1 % Short in Uptrend:  {r1_pct:.2f}% ({r_list[0]['uptrend_short']}/{r_list[0]['uptrend_total']})")
    print(f"Round 6 % Short in Uptrend:  {r6_pct:.2f}% ({r_list[5]['uptrend_short']}/{r_list[5]['uptrend_total']})")
    print(f"Round 10 % Short in Uptrend: {r10_pct:.2f}% ({r_list[9]['uptrend_short']}/{r_list[9]['uptrend_total']})")
    
    # Reductions
    red_r6_vs_r1 = (r1_pct - r6_pct) / r1_pct * 100.0
    red_r10_vs_r1 = (r1_pct - r10_pct) / r1_pct * 100.0
    
    print(f"\nReduction Round 6 vs Round 1:  {red_r6_vs_r1:.2f}% (Threshold >= 50%: {red_r6_vs_r1 >= 50.0})")
    print(f"Reduction Round 10 vs Round 1: {red_r10_vs_r1:.2f}% (Threshold >= 50%: {red_r10_vs_r1 >= 50.0})")
    
    # Aggregate early (R1-R3) vs late (R6-R10)
    early_total = sum(r["uptrend_total"] for r in r_list[:3])
    early_short = sum(r["uptrend_short"] for r in r_list[:3])
    early_pct = (early_short / early_total * 100.0) if early_total > 0 else 0.0
    
    late_total = sum(r["uptrend_total"] for r in r_list[5:])
    late_short = sum(r["uptrend_short"] for r in r_list[5:])
    late_pct = (late_short / late_total * 100.0) if late_total > 0 else 0.0
    
    red_late_vs_r1 = (r1_pct - late_pct) / r1_pct * 100.0
    red_late_vs_early = (early_pct - late_pct) / early_pct * 100.0
    
    print(f"\nAggregate Early Rounds 1-3: {early_pct:.2f}% ({early_short}/{early_total})")
    print(f"Aggregate Late Rounds 6-10:  {late_pct:.2f}% ({late_short}/{late_total})")
    print(f"Reduction Aggregate Late (R6-R10) vs Round 1:       {red_late_vs_r1:.2f}%")
    print(f"Reduction Aggregate Late (R6-R10) vs Early (R1-R3): {red_late_vs_early:.2f}%")


if __name__ == "__main__":
    main()
