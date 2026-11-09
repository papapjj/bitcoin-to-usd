#!/usr/bin/env python3
"""Paired frozen MT5 OOS check for the v124 normalized candidate.

Both arms use the same dates and real-tick model, but isolated MT5 runner
directories so reports, bridge files, and manifests cannot overwrite each
other. The candidate uses ``tanh_v1`` while the protected v13 baseline remains
raw. Neither arm gets a journal or learning updates; this is an inference-only
comparison.
"""
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
DEFAULT_SEED = ROOT / "outputs" / "he_chuyen_gia" / "seeds" / "v13_original_7experts_locked_for_continuation.pkl"
DEFAULT_CANDIDATE = ROOT / "outputs" / "he_chuyen_gia" / "candidates" / "v124_seed_input_tanh_v1_realticks.pkl"
DEFAULT_OUTPUT = ROOT / "outputs" / "he_chuyen_gia" / "audit" / "v124_input_tanh_v1" / "frozen_ab.json"


def normalized_config(config):
    learning = replace(config.learning, input_normalization_mode="tanh_v1")
    experts = replace(
        config.he_chuyen_gia,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
    )
    return replace(config, learning=learning, he_chuyen_gia=experts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-runner", default="mt5_v2_oos_arm_train_v119")
    parser.add_argument("--baseline-runner", default="mt5_v2_runner")
    parser.add_argument("--from-date", default="2025.05.16")
    parser.add_argument("--to-date", default="2025.05.17")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_config = load_config(args.config)
    candidate_config = normalized_config(base_config)
    result: dict[str, object] = {
        "experiment": "v124_input_tanh_v1_frozen_ab",
        "from_date": args.from_date,
        "to_date": args.to_date,
        "candidate_path": str(args.candidate.resolve()),
        "baseline_path": str(args.seed.resolve()),
        "candidate_runner": str(Path(args.candidate_runner).resolve()),
        "baseline_runner": str(Path(args.baseline_runner).resolve()),
        "candidate_normalization": "tanh_v1",
        "baseline_normalization": "raw",
        "candidate": {},
        "baseline": {},
    }
    try:
        candidate = tai_he_chuyen_gia_mot_context_mot_policy(args.candidate, candidate_config, seed=20260724)
        result["candidate"] = evaluate_policy_mt5_oos(
            candidate,
            candidate_config,
            runner_dir=args.candidate_runner,
            timeout_sec=int(args.timeout_sec),
            shutdown_terminal=True,
            from_date=args.from_date,
            to_date=args.to_date,
            model=4,
            policy_id=9102,
        )
    except Exception as exc:
        result["candidate"] = {"completed": False, "error_type": type(exc).__name__, "error": str(exc)}
    try:
        baseline = tai_he_chuyen_gia_mot_context_mot_policy(args.seed, base_config, seed=20260724)
        result["baseline"] = evaluate_policy_mt5_oos(
            baseline,
            base_config,
            runner_dir=args.baseline_runner,
            timeout_sec=int(args.timeout_sec),
            shutdown_terminal=True,
            from_date=args.from_date,
            to_date=args.to_date,
            model=4,
            policy_id=9103,
        )
    except Exception as exc:
        result["baseline"] = {"completed": False, "error_type": type(exc).__name__, "error": str(exc)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if all(bool(isinstance(result.get(key), dict) and result[key].get("best")) for key in ("candidate", "baseline")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
