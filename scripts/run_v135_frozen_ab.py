#!/usr/bin/env python3
"""Paired frozen real-tick OOS for v13 seed versus the v135 candidate."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from self_growing_trader.chuyen_gia_policy import (
    evaluate_policy_mt5_oos,
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl"
CANDIDATE = ROOT / "outputs/he_chuyen_gia/candidates/v135_layernorm_selector_penalty_realticks.pkl"


def _config(base, *, candidate: bool):
    learning = replace(
        base.learning,
        input_normalization_mode="raw",
        stp_input_clip=0.25,
        output_normalization_mode="layernorm_v1" if candidate else "raw",
        output_normalization_target_rms=0.70,
        output_normalization_clip=3.0,
    )
    experts = replace(
        base.he_chuyen_gia,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        selector_loss_streak_penalty=0.12 if candidate else 0.0,
        selector_policy_group_loss_streak_penalty=0.08 if candidate else 0.0,
    )
    return replace(base, mt5_learning_enabled=False, learning=learning, he_chuyen_gia=experts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    parser.add_argument("--seed", type=Path, default=SEED)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--candidate-runner", default="mt5_v2_oos_arm_train_v122_layernorm_selector")
    parser.add_argument("--baseline-runner", default="mt5_v2_oos_arm_train_v123_layernorm_selector_gate")
    parser.add_argument("--from-date", default="2025.05.16")
    parser.add_argument("--to-date", default="2025.05.17")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/he_chuyen_gia/audit/v135_frozen_ab.json")
    args = parser.parse_args()

    base = load_config(args.config)
    candidate_config = _config(base, candidate=True)
    baseline_config = _config(base, candidate=False)
    result: dict[str, object] = {
        "experiment": "v135_layernorm_selector_penalty_frozen_ab",
        "from_date": args.from_date,
        "to_date": args.to_date,
        "candidate": {"path": str(args.candidate.resolve()), "normalization": "layernorm_v1"},
        "baseline": {"path": str(args.seed.resolve()), "normalization": "raw"},
    }
    try:
        candidate = tai_he_chuyen_gia_mot_context_mot_policy(args.candidate, candidate_config, seed=20260724)
        result["candidate"].update(evaluate_policy_mt5_oos(
            candidate,
            candidate_config,
            runner_dir=args.candidate_runner,
            timeout_sec=int(args.timeout_sec),
            shutdown_terminal=True,
            from_date=args.from_date,
            to_date=args.to_date,
            model=4,
        ))
    except Exception as exc:
        result["candidate"].update({"completed": False, "error_type": type(exc).__name__, "error": str(exc)})
    try:
        baseline = tai_he_chuyen_gia_mot_context_mot_policy(args.seed, baseline_config, seed=20260724)
        result["baseline"].update(evaluate_policy_mt5_oos(
            baseline,
            baseline_config,
            runner_dir=args.baseline_runner,
            timeout_sec=int(args.timeout_sec),
            shutdown_terminal=True,
            from_date=args.from_date,
            to_date=args.to_date,
            model=4,
        ))
    except Exception as exc:
        result["baseline"].update({"completed": False, "error_type": type(exc).__name__, "error": str(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if all(bool(isinstance(result.get(key), dict) and result[key].get("best")) for key in ("candidate", "baseline")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
