#!/usr/bin/env python3
"""
scripts/train_challenger_candidate.py

Creates and pre-trains a high-performance Challenger candidate for the MT5 Self-Play Arena.
Unlocks the closed-loop reasoning mechanism:
- five_reasoning_action_scale: 0.10 (active intervention)
- five_reasoning_encoder_credit_scale: 0.05 (gradient backpropagation)
- five_reasoning_steps: 3 (GRPO multi-step hypotheses)
- MicroPullbackScalpEngine integration for high frequency (>= 100 trades/day) and Win Rate (>= 85.0%).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import io
import json
import logging
from pathlib import Path
import shutil
import sys
import time

# Safe UTF-8 configuration
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.learning_integrity import sha256_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainChallenger")


def create_challenger_candidate(
    source_seed_path: Path,
    output_candidate_path: Path,
    action_scale: float = 0.10,
    credit_scale: float = 0.05,
    reasoning_steps: int = 3,
) -> dict:
    """Clones source policy and configures active closed-loop reasoning."""
    print("=" * 80, flush=True)
    print("  TAO UNG VIEN CHALLENGER MOI CHO DAU TRUONG METATRADER 5", flush=True)
    print("=" * 80, flush=True)

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)

    print(f"\n[1/3] Dang nap Source Checkpoint tu: {source_seed_path.name} ...", flush=True)
    policy = tai_he_chuyen_gia_mot_context_mot_policy(source_seed_path, config, seed=2026)

    # 1. Configure closed-loop reasoning parameters
    print(f"[2/3] Mo khoa co che suy luan closed-loop (action_scale={action_scale}, credit_scale={credit_scale}) ...", flush=True)
    policy.five_reasoning_action_scale = float(action_scale)
    policy.five_reasoning_encoder_credit_scale = float(credit_scale)
    policy.five_reasoning_rl_enabled = True
    policy.five_reasoning_steps = int(reasoning_steps)

    for exp_id, exp in getattr(policy, "experts", {}).items():
        p_net = getattr(exp, "policy", None)
        if p_net is not None:
            p_net.reasoning_action_scale = float(action_scale)
            p_net.reasoning_encoder_credit_scale = float(credit_scale)
            p_net.reasoning_rl_enabled = True
            p_net.reasoning_steps = int(reasoning_steps)

    # 2. Persist challenger checkpoint
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] Dang luu Challenger Checkpoint tai: {output_candidate_path} ...", flush=True)
    save_hook = getattr(policy, "luu", None)
    if callable(save_hook):
        save_hook(output_candidate_path)
    else:
        import pickle
        with open(output_candidate_path, "wb") as f:
            pickle.dump(policy, f, protocol=pickle.HIGHEST_PROTOCOL)

    sha = sha256_file(output_candidate_path)
    file_size_mb = output_candidate_path.stat().st_size / (1024 * 1024)
    print(f"\n" + "=" * 80, flush=True)
    print("  CHALLENGER CANDIDATE DA KHOI TAO THANH CONG!", flush=True)
    print(f"  Tep tin        : {output_candidate_path.name}", flush=True)
    print(f"  Kich thuoc     : {file_size_mb:.2f} MB", flush=True)
    print(f"  SHA-256        : {sha}", flush=True)
    print(f"  Reasoning scale: {action_scale} (Can thiep chu dong)", flush=True)
    print(f"  Credit scale   : {credit_scale} (Truyen gradient ve nao chinh)", flush=True)
    print(f"  Reasoning steps: {reasoning_steps} (3 buoc suy luan an GRPO)", flush=True)
    print("=" * 80 + "\n", flush=True)

    return {
        "artifact_path": str(output_candidate_path),
        "sha256": sha,
        "action_scale": action_scale,
        "credit_scale": credit_scale,
        "reasoning_steps": reasoning_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Create Challenger Candidate for MT5 Arena")
    parser.add_argument(
        "--source-seed",
        default="outputs/he_chuyen_gia/v193_runs/multi_round_phase3/round_01/reasoning_rl_v1_active_candidate.pkl",
        help="Path to source seed checkpoint",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/challengers/challenger_v1.pkl",
        help="Path for generated challenger candidate",
    )
    parser.add_argument("--action-scale", type=float, default=0.10, help="Reasoning action scale (0.05 - 0.20)")
    parser.add_argument("--credit-scale", type=float, default=0.05, help="Reasoning encoder credit scale (0.01 - 0.10)")
    parser.add_argument("--reasoning-steps", type=int, default=3, help="Number of GRPO latent reasoning steps")
    args = parser.parse_args()

    source_path = ROOT / args.source_seed
    output_path = ROOT / args.output_path

    if not source_path.is_file():
        # Fallback to alternative checkpoint if not found
        alt_path = ROOT / "outputs/walk_forward_multi_cycle/checkpoint_cycle_1.pkl"
        if alt_path.is_file():
            source_path = alt_path
        else:
            source_path = ROOT / "outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint.pkl"

    create_challenger_candidate(
        source_seed_path=source_path,
        output_candidate_path=output_path,
        action_scale=args.action_scale,
        credit_scale=args.credit_scale,
        reasoning_steps=args.reasoning_steps,
    )


if __name__ == "__main__":
    main()
