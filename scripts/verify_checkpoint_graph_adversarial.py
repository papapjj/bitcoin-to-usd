#!/usr/bin/env python3
"""Adversarial Verification of Checkpoint Model Graphs & Weight Divergence.

Performs:
1. Deserialization of best_promoted_checkpoint_phase2.pkl and best_promoted_checkpoint.pkl
2. PyTorch model graph inspection and submodule validation:
   - FiveObservationEncoder
   - LSTMCell
   - PostLstmCrossAgentTransformer
   - FactorizedActionHeads
   - FiveReasoningHead (reasoning_head & reasoning_reference_head)
3. Exact parameter counting: total, trainable, frozen reference head
4. Tensor-level parameter comparison between Phase 1 and Phase 2 to verify weight divergence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from self_growing_trader.legacy_pickle_migration import load_pickle_with_removed_context_structural_migration
from self_growing_trader.five_teacher_imitation import shared_five_network
from self_growing_trader.five_shared_policy import (
    FiveObservationEncoder,
    PostLstmCrossAgentTransformer,
    FactorizedActionHeads,
    FiveReasoningHead,
)


def main() -> int:
    phase2_path = REPO_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    phase1_path = REPO_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint.pkl"

    print("=" * 80)
    print("STEP 1: DESERIALIZING PHASE 2 CANONICAL CHECKPOINT")
    print("=" * 80)
    with open(phase2_path, "rb") as f:
        system2 = load_pickle_with_removed_context_structural_migration(f)
    net2 = shared_five_network(system2)
    print(f"Phase 2 Network Loaded: {type(net2).__name__}")

    print("\n" + "=" * 80)
    print("STEP 2: SUBMODULE STRUCTURE VERIFICATION (PHASE 2)")
    print("=" * 80)

    # Required submodules check
    expected_submodules = {
        "observation_encoder": FiveObservationEncoder,
        "lstm": torch.nn.LSTMCell,
        "post_lstm_transformer": PostLstmCrossAgentTransformer,
        "heads": FactorizedActionHeads,
        "reasoning_head": FiveReasoningHead,
        "reasoning_reference_head": FiveReasoningHead,
    }

    submodule_ok = True
    for attr_name, expected_cls in expected_submodules.items():
        actual_obj = getattr(net2, attr_name, None)
        if actual_obj is None:
            print(f"  [FAIL] Missing attribute: {attr_name}")
            submodule_ok = False
        elif not isinstance(actual_obj, expected_cls):
            print(f"  [FAIL] Submodule {attr_name}: expected {expected_cls.__name__}, got {type(actual_obj).__name__}")
            submodule_ok = False
        else:
            p_count = sum(p.numel() for p in actual_obj.parameters())
            tr_count = sum(p.numel() for p in actual_obj.parameters() if p.requires_grad)
            fr_count = sum(p.numel() for p in actual_obj.parameters() if not p.requires_grad)
            print(f"  [PASS] {attr_name:25s} -> {expected_cls.__name__:30s} | params: {p_count:10,d} (trainable: {tr_count:10,d}, frozen: {fr_count:10,d})")

    assert submodule_ok, "Submodule verification failed!"

    print("\n" + "=" * 80)
    print("STEP 3: PARAMETER COUNTS & FROZEN REFERENCE VALIDATION")
    print("=" * 80)
    total_params = sum(p.numel() for p in net2.parameters())
    trainable_params = sum(p.numel() for p in net2.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in net2.parameters() if not p.requires_grad)

    print(f"Total Parameters:     {total_params:12,d}")
    print(f"Trainable Parameters: {trainable_params:12,d}")
    print(f"Frozen Parameters:    {frozen_params:12,d}")

    # Specific assertions
    assert total_params == 117350081, f"Expected 117,350,081 total parameters, got {total_params}"
    assert trainable_params == 117062010, f"Expected 117,062,010 trainable parameters, got {trainable_params}"
    assert frozen_params == 288071, f"Expected 288,071 frozen parameters, got {frozen_params}"

    # Verify frozen reference head specifically
    ref_head_params = sum(p.numel() for p in net2.reasoning_reference_head.parameters())
    ref_head_frozen = sum(p.numel() for p in net2.reasoning_reference_head.parameters() if not p.requires_grad)
    assert ref_head_params == 288071, f"Reasoning reference head parameter count mismatch: {ref_head_params}"
    assert ref_head_frozen == 288071, f"Reasoning reference head not fully frozen: {ref_head_frozen} / {ref_head_params}"
    print(f"  [PASS] reasoning_reference_head has exactly {ref_head_params:,} parameters, 100% frozen (requires_grad=False).")

    # Verify reasoning head is trainable
    act_head_params = sum(p.numel() for p in net2.reasoning_head.parameters())
    act_head_trainable = sum(p.numel() for p in net2.reasoning_head.parameters() if p.requires_grad)
    assert act_head_params == 288071, f"Reasoning active head parameter count mismatch: {act_head_params}"
    assert act_head_trainable == 288071, f"Reasoning active head not fully trainable: {act_head_trainable} / {act_head_params}"
    print(f"  [PASS] reasoning_head has exactly {act_head_params:,} parameters, 100% trainable (requires_grad=True).")

    print("\n" + "=" * 80)
    print("STEP 4: DESERIALIZING PHASE 1 SEED CHECKPOINT")
    print("=" * 80)
    with open(phase1_path, "rb") as f:
        system1 = load_pickle_with_removed_context_structural_migration(f)
    net1 = shared_five_network(system1)
    p1_total = sum(p.numel() for p in net1.parameters())
    p1_trainable = sum(p.numel() for p in net1.parameters() if p.requires_grad)
    p1_frozen = sum(p.numel() for p in net1.parameters() if not p.requires_grad)
    print(f"Phase 1 Network Loaded: {type(net1).__name__}")
    print(f"Phase 1 Total Parameters:     {p1_total:12,d}")
    print(f"Phase 1 Trainable Parameters: {p1_trainable:12,d}")
    print(f"Phase 1 Frozen Parameters:    {p1_frozen:12,d}")

    print("\n" + "=" * 80)
    print("STEP 5: PARAMETER DIVERGENCE ANALYSIS (PHASE 1 vs PHASE 2)")
    print("=" * 80)

    params1 = dict(net1.named_parameters())
    params2 = dict(net2.named_parameters())

    common_keys = set(params1.keys()).intersection(set(params2.keys()))
    added_in_p2 = set(params2.keys()) - set(params1.keys())
    removed_in_p2 = set(params1.keys()) - set(params2.keys())

    print(f"Parameter tensors in Phase 1: {len(params1)}")
    print(f"Parameter tensors in Phase 2: {len(params2)}")
    print(f"Common parameter tensors:     {len(common_keys)}")
    print(f"Added tensors in Phase 2:     {len(added_in_p2)}")
    print(f"Removed tensors in Phase 2:   {len(removed_in_p2)}")

    diverged_count = 0
    identical_count = 0
    module_stats: dict[str, dict[str, float]] = {}

    for k in sorted(common_keys):
        t1 = params1[k].data
        t2 = params2[k].data

        mod = k.split(".")[0]
        if mod not in module_stats:
            module_stats[mod] = {"diverged": 0, "identical": 0, "max_diff": 0.0, "total_diff": 0.0, "elem_count": 0}

        if t1.shape != t2.shape:
            print(f"  [SHAPE MISMATCH] {k}: {t1.shape} vs {t2.shape}")
            continue

        is_equal = torch.equal(t1, t2)
        diff = (t1.float() - t2.float()).abs()
        max_d = float(diff.max())
        sum_d = float(diff.sum())
        n_elem = t1.numel()

        if is_equal:
            identical_count += 1
            module_stats[mod]["identical"] += 1
        else:
            diverged_count += 1
            module_stats[mod]["diverged"] += 1
            module_stats[mod]["max_diff"] = max(module_stats[mod]["max_diff"], max_d)
            module_stats[mod]["total_diff"] += sum_d
            module_stats[mod]["elem_count"] += n_elem

    print("\nDivergence breakdown by top-level module:")
    print(f"{'Module':25s} | {'Diverged Tensors':16s} | {'Identical Tensors':17s} | {'Max Abs Diff':14s} | {'Mean Abs Diff':14s}")
    print("-" * 95)
    for mod, st in sorted(module_stats.items()):
        d_cnt = int(st["diverged"])
        i_cnt = int(st["identical"])
        mx = st["max_diff"]
        mean_d = (st["total_diff"] / st["elem_count"]) if st["elem_count"] > 0 else 0.0
        print(f"{mod:25s} | {d_cnt:16d} | {i_cnt:17d} | {mx:14.6e} | {mean_d:14.6e}")

    print("-" * 95)
    print(f"Total Common Tensors Diverged:  {diverged_count} / {len(common_keys)} ({diverged_count / len(common_keys) * 100:.1f}%)")
    print(f"Total Common Tensors Identical: {identical_count} / {len(common_keys)} ({identical_count / len(common_keys) * 100:.1f}%)")

    if added_in_p2:
        print("\nNew Tensors in Phase 2 (Reasoning Architecture):")
        for k in sorted(added_in_p2):
            t = params2[k]
            print(f"  {k:50s} | shape: {str(list(t.shape)):20s} | requires_grad={t.requires_grad}")

    # Check reasoning head vs reference head in Phase 2
    print("\n" + "=" * 80)
    print("STEP 6: REASONING ACTIVE HEAD vs FROZEN REFERENCE HEAD DIVERGENCE (PHASE 2)")
    print("=" * 80)
    ref_params = dict(net2.reasoning_reference_head.named_parameters())
    act_params = dict(net2.reasoning_head.named_parameters())
    ref_diff_count = 0
    ref_identical_count = 0
    for k in sorted(ref_params.keys()):
        r_t = ref_params[k].data
        a_t = act_params[k].data
        if torch.equal(r_t, a_t):
            ref_identical_count += 1
        else:
            ref_diff_count += 1
    print(f"Reasoning Head vs Reference Head: {ref_diff_count} diverged, {ref_identical_count} identical across {len(ref_params)} tensors.")

    print("\n" + "=" * 80)
    print("ALL EMPIRICAL CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
