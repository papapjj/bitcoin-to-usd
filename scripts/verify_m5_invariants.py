"""
Milestone 5 Invariants & Checkpoints Verification Script.
Author: teamwork_worker_tca_m5_1
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RRWC_PATH = REPO_ROOT / "research_reasoning_world_class"
if str(RRWC_PATH) not in sys.path:
    sys.path.insert(0, str(RRWC_PATH))

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    MultiHorizonProjectionHead,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from scripts.train_tca_multi_horizon import hash_reasoning_engine


def main():
    print("=" * 70)
    print("MILESTONE 5 INVARIANTS & CHECKPOINTS VERIFICATION AUDIT")
    print("=" * 70)

    ckpt_dir = REPO_ROOT / "outputs" / "checkpoints"
    print(f"\n[1] Checking directory: {ckpt_dir}")
    assert ckpt_dir.exists(), f"Directory {ckpt_dir} does not exist"

    files = sorted(list(ckpt_dir.iterdir()))
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name:35s} {size_kb:10.2f} KB ({f.stat().st_size} bytes)")

    required_artifacts = [
        "tca_phase1_pretrained.pt",
        "tca_phase2_finetuned.pt",
        "training_logs.json",
        "oos_crash_evaluation_results.json",
    ]
    for req in required_artifacts:
        req_path = ckpt_dir / req
        assert req_path.exists(), f"Required artifact {req} is missing!"
        print(f"  [PASS] Required artifact verified: {req}")

    print("\n[2] Reasoning Engine Invariant Verification:")
    champ_path = REPO_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_specialized_challenger.pt"
    assert champ_path.exists(), f"Challenger baseline missing: {champ_path}"
    champ_ckpt = torch.load(champ_path, map_location="cpu", weights_only=False)
    champ_re_sd = champ_ckpt.get("reasoning_engine_state_dict", champ_ckpt.get("bull_cot_state_dict", None))

    cfg = ReasoningConfig(input_dim=768)
    engine_ref = MetacognitiveReasoningEngine(cfg)
    load_re_res = engine_ref.load_state_dict(champ_re_sd, strict=True)
    assert len(load_re_res.missing_keys) == 0 and len(load_re_res.unexpected_keys) == 0

    ref_sd = engine_ref.state_dict()
    ref_keys = len(ref_sd)
    ref_weights = sum(p.numel() for p in engine_ref.parameters())
    ref_sha256 = hash_reasoning_engine(engine_ref)

    print(f"  Reference Keys:       {ref_keys} (expected 191)")
    print(f"  Reference Parameters: {ref_weights} (expected 4,196,564)")
    print(f"  Reference SHA-256:    {ref_sha256}")
    assert ref_keys == 191, f"Expected 191 keys, got {ref_keys}"
    assert ref_weights == 4196564, f"Expected 4196564 weights, got {ref_weights}"
    assert ref_sha256 == "b69d48bd5993f9f989d466247a0242b5b317e96dae32769b84f1fa73b3b8a929", f"Unexpected reference SHA-256: {ref_sha256}"

    for ckpt_name in ["tca_phase1_pretrained.pt", "tca_phase2_finetuned.pt"]:
        print(f"\n  Checking checkpoint: {ckpt_name}")
        ckpt = torch.load(ckpt_dir / ckpt_name, map_location="cpu", weights_only=False)
        assert "reasoning_engine_state_dict" in ckpt, f"reasoning_engine_state_dict missing in {ckpt_name}"
        re_sd = ckpt["reasoning_engine_state_dict"]
        re_keys = len(re_sd)

        # Test strict load into engine
        engine_test = MetacognitiveReasoningEngine(cfg)
        load_res = engine_test.load_state_dict(re_sd, strict=True)
        assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0
        re_weights = sum(p.numel() for p in engine_test.parameters())
        re_sha256 = hash_reasoning_engine(re_sd)

        max_drift = 0.0
        for k in sorted(re_sd.keys()):
            diff = (re_sd[k] - ref_sd[k]).abs().max().item()
            if diff > max_drift:
                max_drift = diff

        print(f"    Keys:                 {re_keys:10d} (expected 191)       -> {'PASS' if re_keys == 191 else 'FAIL'}")
        print(f"    Unique Parameters:    {re_weights:10d} (expected 4196564)   -> {'PASS' if re_weights == 4196564 else 'FAIL'}")
        print(f"    Max Absolute Drift:   {max_drift:10.8f} (expected 0.0)       -> {'PASS' if max_drift == 0.0 else 'FAIL'}")
        print(f"    SHA-256 Digest:       {re_sha256}")
        print(f"    SHA-256 Bit Match:    {'MATCH' if re_sha256 == ref_sha256 else 'MISMATCH'}")
        assert re_keys == 191
        assert re_weights == 4196564
        assert max_drift == 0.0
        assert re_sha256 == ref_sha256

    print("\n[3] Baseline World Model 30-Key Checkpoint Invariant:")
    wm_base = LatentDynamicsWorldModel()
    base_sd = wm_base.state_dict()
    base_keys = len(base_sd)
    base_weights = sum(p.numel() for p in wm_base.parameters())
    print(f"  Default LatentDynamicsWorldModel Keys:    {base_keys} (expected 30)")
    print(f"  Default LatentDynamicsWorldModel Weights: {base_weights} (expected 2,160,900)")
    assert base_keys == 30
    assert base_weights == 2160900

    for ckpt_name in ["tca_phase1_pretrained.pt", "tca_phase2_finetuned.pt"]:
        print(f"\n  Checking checkpoint: {ckpt_name}")
        ckpt = torch.load(ckpt_dir / ckpt_name, map_location="cpu", weights_only=False)
        assert "world_model_state_dict" in ckpt
        assert "temporal_encoder_state_dict" in ckpt

        wm_sd = ckpt["world_model_state_dict"]
        tca_sd = ckpt["temporal_encoder_state_dict"]

        print(f"    Baseline World Model Keys in ckpt:  {len(wm_sd)} (expected strictly 30)")
        print(f"    Temporal Encoder Keys in ckpt:      {len(tca_sd)} (expected strictly 41)")
        assert len(wm_sd) == 30, f"Expected 30 keys in world_model_state_dict, got {len(wm_sd)}"
        assert len(tca_sd) == 41, f"Expected 41 keys in temporal_encoder_state_dict, got {len(tca_sd)}"

        # Strict load into default baseline WM
        load_wm_res = wm_base.load_state_dict(wm_sd, strict=True)
        print(f"    Strict Load Result into Baseline WM: missing={len(load_wm_res.missing_keys)}, unexpected={len(load_wm_res.unexpected_keys)}")
        assert len(load_wm_res.missing_keys) == 0
        assert len(load_wm_res.unexpected_keys) == 0

        # Strict load into TCA encoder
        tca_encoder = TemporalCrossAttentionEncoder(candle_len=240, latent_dim=768)
        load_tca_res = tca_encoder.load_state_dict(tca_sd, strict=True)
        print(f"    Strict Load Result into TCA Module: missing={len(load_tca_res.missing_keys)}, unexpected={len(load_tca_res.unexpected_keys)}")
        assert len(load_tca_res.missing_keys) == 0
        assert len(load_tca_res.unexpected_keys) == 0

    print("\n[4] Projection Head Checkpoint Invariant:")
    for ckpt_name in ["tca_phase1_pretrained.pt", "tca_phase2_finetuned.pt"]:
        print(f"\n  Checking checkpoint: {ckpt_name}")
        ckpt = torch.load(ckpt_dir / ckpt_name, map_location="cpu", weights_only=False)
        assert "projection_head_state_dict" in ckpt
        head_sd = ckpt["projection_head_state_dict"]
        head_keys = len(head_sd)
        head_weights = sum(t.numel() for t in head_sd.values())
        print(f"    Projection Head Keys:    {head_keys} (expected 8)")
        print(f"    Projection Head Weights: {head_weights} (expected 230,659)")
        assert head_keys == 8
        assert head_weights == 230659

        head = MultiHorizonProjectionHead()
        load_head_res = head.load_state_dict(head_sd, strict=True)
        print(f"    Strict Load Result: missing={len(load_head_res.missing_keys)}, unexpected={len(load_head_res.unexpected_keys)}")
        assert len(load_head_res.missing_keys) == 0
        assert len(load_head_res.unexpected_keys) == 0

    print("\n[5] Training Logs and OOS Results Validation:")
    with open(ckpt_dir / "training_logs.json", "r", encoding="utf-8") as f:
        tlogs = json.load(f)
    print(f"  Training logs status:            {tlogs.get('status')}")
    print(f"  Training logs re_drift_max:      {tlogs.get('re_drift_max')}")
    print(f"  Training logs re_initial_sha256: {tlogs.get('re_initial_sha256')}")
    print(f"  Training logs re_final_sha256:   {tlogs.get('re_final_sha256')}")
    assert tlogs.get("status") == "COMPLETED"
    assert tlogs.get("re_drift_max") == 0.0
    assert tlogs.get("re_initial_sha256") == tlogs.get("re_final_sha256") == ref_sha256

    with open(ckpt_dir / "oos_crash_evaluation_results.json", "r", encoding="utf-8") as f:
        oos = json.load(f)
    print(f"  OOS evaluation status: {oos.get('status')}")
    print(f"  OOS slice range:       {oos.get('dataset', {}).get('range')} (total bars: {oos.get('dataset', {}).get('total_bars')})")
    print(f"  OOS models evaluated:  {list(oos.get('models', {}).keys())}")
    assert oos.get("status") == "COMPLETED"
    assert oos.get("dataset", {}).get("total_bars") == 9785
    assert set(oos.get("models", {}).keys()) == {"baseline", "phase1_pretrained", "phase2_finetuned"}

    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY — ZERO DEFECTS FOUND")
    print("=" * 70)


if __name__ == "__main__":
    main()
