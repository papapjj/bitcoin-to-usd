#!/usr/bin/env python3
"""Audit and optionally run an isolated input-conditioning experiment.

The default action is read-only: it compares the protected v13 seed's raw and
the selected conditioning mode on the latest MT5 feature audit. ``--run-mt5`` then
starts one separate candidate run with structural growth/specialist fan-out
disabled, low online update scale and the protected seed as an immutable
source.  No command in this script can overwrite the seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from self_growing_trader.config import load_config
from self_growing_trader.feature_contract import feature_schema_fingerprint
from self_growing_trader.input_normalization import (
    INPUT_COUNT,
    INPUT_NORMALIZATION_MODES,
    input_normalization_payload,
    normalize_input_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "mt5_v2_oos_arm_train_v119" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_feature_audit.csv"
DEFAULT_SEED = ROOT / "outputs" / "he_chuyen_gia" / "seeds" / "v13_original_7experts_locked_for_continuation.pkl"
DEFAULT_REPORT = ROOT / "outputs" / "he_chuyen_gia" / "audit" / "v124_input_tanh_v1" / "normalization_report.json"
DEFAULT_CANDIDATE = ROOT / "outputs" / "he_chuyen_gia" / "candidates" / "v124_seed_input_tanh_v1_realticks.pkl"


def _read_audit(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig"), delimiter="\t"))
    values = np.asarray([[float(item) for item in row["inputs"].split("|")] for row in rows], dtype=float)
    if values.ndim != 2 or values.shape[1] != INPUT_COUNT:
        raise ValueError(f"audit must contain (N, {INPUT_COUNT}) inputs; got {values.shape}")
    return values


def _matrix_summary(values: np.ndarray) -> dict[str, object]:
    flat = np.abs(values)
    return {
        "rows": int(values.shape[0]),
        "width": int(values.shape[1]),
        "finite": bool(np.isfinite(values).all()),
        "max_abs": float(flat.max(initial=0.0)),
        "p99_abs": float(np.quantile(flat, 0.99)),
        "mean_abs": float(flat.mean()),
        "global_abs_gt_1": float(np.mean(flat > 1.0)),
        "global_abs_gt_5": float(np.mean(flat > 5.0)),
        "global_abs_ge_0_995": float(np.mean(flat >= 0.995)),
        "row_any_abs_gt_5": float(np.mean(np.any(flat > 5.0, axis=1))),
        "row_any_abs_ge_0_995": float(np.mean(np.any(flat >= 0.995, axis=1))),
    }


def _hidden_saturation(seed_path: Path, raw: np.ndarray, normalized_mode: str, stp_input_clip: float = 50.0) -> dict[str, object]:
    with seed_path.open("rb") as handle:
        system = pickle.load(handle)
    output: dict[str, object] = {}
    for mode in ("raw", normalized_mode):
        observations = normalize_input_matrix(raw, mode=mode)
        ratios: list[float] = []
        for expert_name, expert in getattr(system, "experts", {}).items():
            modules = list(getattr(getattr(expert, "policy", None), "modules", ()) or ())
            if not modules:
                continue
            module = modules[0]
            weight = np.asarray(module.w_in, dtype=float)
            stp = np.asarray(module.p_stp, dtype=float).copy()
            bias = np.asarray(module.b_hidden, dtype=float)
            row_ratios = []
            for observation in observations:
                stp_drive = np.minimum(np.abs(observation), float(np.clip(stp_input_clip, 0.1, 50.0)))
                stp = np.clip(0.95 * stp + 0.1 * stp_drive.reshape(1, -1), 0.01, 5.0)
                preactivation = (weight * stp) @ observation + bias
                row_ratios.append(float(np.mean(np.abs(preactivation) > 3.0)))
            ratios.append(float(np.mean(row_ratios)))
        output[mode] = {
            "expert_count": len(ratios),
            "mean_hidden_abs_z_gt_3": float(np.mean(ratios)) if ratios else None,
            "min_hidden_abs_z_gt_3": float(np.min(ratios)) if ratios else None,
            "max_hidden_abs_z_gt_3": float(np.max(ratios)) if ratios else None,
        }
    return output


def _normalized_config(
    config,
    normalized_mode: str,
    online_update_scale: float,
    stp_input_clip: float,
    hidden_normalization_mode: str = "layernorm_v1",
    hidden_normalization_target_rms: float = 1.0,
    hidden_normalization_clip: float = 3.0,
    hidden_normalization_eps: float = 1e-5,
    context_expectancy_entry_gate: bool = False,
    context_direction_entry_gate: bool = False,
    context_direction_side_entry_gate: bool = False,
    context_direction_min_timing_quality: float = 0.25,
    output_normalization_mode: str = "raw",
    output_normalization_target_rms: float = 0.70,
    output_normalization_clip: float = 3.0,
    selector_loss_streak_penalty: float = 0.0,
    selector_policy_group_loss_streak_penalty: float = 0.0,
    timing_network_enabled: bool = True,
    timing_network_action_scale: float = 0.10,
    timing_network_update_scale: float = 0.25,
    timing_network_learning_rate: float = 0.0005,
    timing_network_hidden_size: int = 32,
    timing_network_embedding_size: int = 16,
    timing_network_bptt_window: int = 24,
    timing_network_eligibility_decay: float = 0.90,
    timing_network_gradient_clip: float = 1.0,
    negative_context_credit_multiplier: float = 8.0,
    negative_entry_replacement_gain: float = 3.0,
):
    learning = replace(
        config.learning,
        input_normalization_mode=normalized_mode,
        stp_input_clip=float(stp_input_clip),
        hidden_normalization_mode=str(hidden_normalization_mode).strip().lower(),
        hidden_normalization_target_rms=float(hidden_normalization_target_rms),
        hidden_normalization_clip=float(hidden_normalization_clip),
        hidden_normalization_eps=float(hidden_normalization_eps),
        mt5_online_update_scale=float(online_update_scale),
        runtime_fitness_guard_enabled=True,
        context_expectancy_entry_gate_enabled=bool(context_expectancy_entry_gate),
        context_direction_entry_gate_enabled=bool(context_direction_entry_gate),
        context_direction_entry_gate_min_timing_quality=float(np.clip(context_direction_min_timing_quality, 0.0, 1.0)),
        context_direction_side_entry_gate_enabled=bool(context_direction_side_entry_gate),
        output_normalization_mode=str(output_normalization_mode).strip().lower(),
        output_normalization_target_rms=float(output_normalization_target_rms),
        output_normalization_clip=float(output_normalization_clip),
        timing_network_enabled=bool(timing_network_enabled),
        timing_network_action_scale=float(timing_network_action_scale),
        timing_network_update_scale=float(timing_network_update_scale),
        timing_network_learning_rate=float(timing_network_learning_rate),
        timing_network_hidden_size=int(timing_network_hidden_size),
        timing_network_embedding_size=int(timing_network_embedding_size),
        timing_network_bptt_window=int(timing_network_bptt_window),
        timing_network_eligibility_decay=float(timing_network_eligibility_decay),
        timing_network_gradient_clip=float(timing_network_gradient_clip),
        negative_context_credit_multiplier=float(negative_context_credit_multiplier),
        negative_entry_replacement_gain=float(negative_entry_replacement_gain),
    )
    experts = replace(
        config.he_chuyen_gia,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        selector_loss_streak_penalty=float(np.clip(selector_loss_streak_penalty, 0.0, 1.0)),
        selector_policy_group_loss_streak_penalty=float(
            np.clip(selector_policy_group_loss_streak_penalty, 0.0, 1.0)
        ),
    )
    return replace(config, learning=learning, he_chuyen_gia=experts)


def main() -> int:
    # Windows PowerShell may expose cp1252 stdout; the report contains
    # Vietnamese lifecycle/config keys, so a successful MT5 run must not be
    # turned into a false failure by the final diagnostic print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--runner", default="mt5_v2_oos_arm_train_v119")
    parser.add_argument("--normalization-mode", choices=sorted(INPUT_NORMALIZATION_MODES), default="tanh_v1")
    parser.add_argument("--online-update-scale", type=float, default=0.05)
    parser.add_argument("--stp-input-clip", type=float, default=50.0)
    parser.add_argument("--hidden-normalization-mode", choices=("rms_v1", "layernorm_v1"), default="layernorm_v1")
    parser.add_argument("--hidden-normalization-target-rms", type=float, default=1.0)
    parser.add_argument("--hidden-normalization-clip", type=float, default=3.0)
    parser.add_argument("--hidden-normalization-eps", type=float, default=1e-5)
    parser.add_argument("--context-expectancy-entry-gate", action="store_true")
    parser.add_argument("--context-direction-entry-gate", action="store_true")
    parser.add_argument("--context-direction-side-entry-gate", action="store_true")
    parser.add_argument("--context-direction-min-timing-quality", type=float, default=0.25)
    parser.add_argument(
        "--output-normalization-mode",
        choices=("raw", "rms_v1", "layernorm_v1"),
        default="raw",
    )
    parser.add_argument("--output-normalization-target-rms", type=float, default=0.70)
    parser.add_argument("--output-normalization-clip", type=float, default=3.0)
    parser.add_argument("--selector-loss-streak-penalty", type=float, default=0.0)
    parser.add_argument("--selector-policy-group-loss-streak-penalty", type=float, default=0.0)
    parser.add_argument(
        "--timing-network",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the canonical TimingNet signal (use --no-timing-network for a control)",
    )
    parser.add_argument("--timing-network-action-scale", type=float, default=0.10)
    parser.add_argument("--timing-network-update-scale", type=float, default=0.25)
    parser.add_argument("--timing-network-learning-rate", type=float, default=0.0005)
    parser.add_argument("--timing-network-hidden-size", type=int, default=32)
    parser.add_argument("--timing-network-embedding-size", type=int, default=16)
    parser.add_argument("--timing-network-bptt-window", type=int, default=24)
    parser.add_argument("--timing-network-eligibility-decay", type=float, default=0.90)
    parser.add_argument("--timing-network-gradient-clip", type=float, default=1.0)
    parser.add_argument("--negative-context-credit-multiplier", type=float, default=8.0)
    parser.add_argument("--negative-entry-replacement-gain", type=float, default=3.0)
    parser.add_argument("--from-date", default="2025.05.15")
    parser.add_argument("--to-date", default="2025.05.16")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-wall-sec", type=int, default=900)
    parser.add_argument("--run-mt5", action="store_true")
    args = parser.parse_args()

    # MT5 can temporarily change the process working directory while it
    # launches a portable tester. Resolve filesystem arguments before entering
    # that call so atomic candidate persistence never depends on a stale cwd.
    args.audit = args.audit if args.audit.is_absolute() else (ROOT / args.audit)
    args.seed = args.seed if args.seed.is_absolute() else (ROOT / args.seed)
    args.report = args.report if args.report.is_absolute() else (ROOT / args.report)
    args.candidate = args.candidate if args.candidate.is_absolute() else (ROOT / args.candidate)
    runner_path = Path(args.runner)
    args.runner = str(runner_path if runner_path.is_absolute() else (ROOT / runner_path))
    for name in ("audit", "seed", "report", "candidate"):
        setattr(args, name, Path(getattr(args, name)).resolve())

    raw = _read_audit(args.audit)
    config = load_config(args.config)
    normalized_mode = str(args.normalization_mode).strip().lower()
    normalized_payload = input_normalization_payload(normalized_mode)
    stp_input_clip = float(np.clip(args.stp_input_clip, 0.1, 50.0))
    result: dict[str, object] = {
        "experiment": f"input_{normalized_mode}_stp_{stp_input_clip:g}",
        "seed": str(args.seed.resolve()),
        "audit": str(args.audit.resolve()),
        "raw": _matrix_summary(raw),
        normalized_mode: _matrix_summary(normalize_input_matrix(raw, mode=normalized_mode)),
        "schema_hash_raw": feature_schema_fingerprint(config.features, config.symbol, input_normalization_mode="raw"),
        f"schema_hash_{normalized_mode}": feature_schema_fingerprint(config.features, config.symbol, input_normalization_mode=normalized_mode),
        "normalization_contract": {
            **normalized_payload,
            "leakage_free": True,
        },
    }
    if args.seed.is_file():
        result["hidden_saturation"] = _hidden_saturation(args.seed, raw, normalized_mode, stp_input_clip)
    result["stp_contract"] = {
        "stp_input_clip": stp_input_clip,
        "legacy_equivalent_clip": 50.0,
        "leakage_free": True,
    }
    result["hidden_normalization_contract"] = {
        "mode": str(args.hidden_normalization_mode).strip().lower(),
        "version": 1,
        "target_rms": float(np.clip(args.hidden_normalization_target_rms, 1e-6, 10.0)),
        "clip": float(np.clip(args.hidden_normalization_clip, 1e-6, 20.0)),
        "eps": float(np.clip(args.hidden_normalization_eps, 1e-9, 1.0)),
        "per_forward_pass": True,
        "running_statistics": False,
        "leakage_free": True,
    }
    result["entry_gate_contract"] = {
        "context_expectancy_entry_gate_enabled": bool(args.context_expectancy_entry_gate),
        "context_direction_entry_gate_enabled": bool(args.context_direction_entry_gate),
        "context_direction_side_entry_gate_enabled": bool(args.context_direction_side_entry_gate),
        "context_direction_entry_gate_min_timing_quality": float(np.clip(args.context_direction_min_timing_quality, 0.0, 1.0)),
        "min_support": 12.0,
        "max_lcb": -0.15,
        "leakage_free": True,
    }
    result["output_contract"] = {
        "mode": str(args.output_normalization_mode).strip().lower(),
        "target_rms": float(args.output_normalization_target_rms),
        "clip": float(args.output_normalization_clip),
        "action_count": 12,
        "parameter_rows_untouched": True,
        "leakage_free": True,
    }
    result["selector_contract"] = {
        "loss_streak_penalty": float(np.clip(args.selector_loss_streak_penalty, 0.0, 1.0)),
        "policy_group_loss_streak_penalty": float(
            np.clip(args.selector_policy_group_loss_streak_penalty, 0.0, 1.0)
        ),
        "leakage_free": True,
    }
    result["timing_network_contract"] = {
        "enabled": bool(args.timing_network),
        "action_scale": float(args.timing_network_action_scale),
        "update_scale": float(args.timing_network_update_scale),
        "learning_rate": float(args.timing_network_learning_rate),
        "hidden_size": int(args.timing_network_hidden_size),
        "embedding_size": int(args.timing_network_embedding_size),
        "bptt_window": int(args.timing_network_bptt_window),
        "eligibility_decay": float(args.timing_network_eligibility_decay),
        "gradient_clip": float(args.timing_network_gradient_clip),
        "input_groups": "market+timing+wyckoff_only",
        "lifecycle_credit": "sigmoid_tanh_chain_rule_with_online_truncated_bptt",
        "leakage_free": True,
    }
    result["negative_credit_contract"] = {
        "negative_context_credit_multiplier": float(np.clip(args.negative_context_credit_multiplier, 0.0, 16.0)),
        "negative_entry_replacement_gain": float(np.clip(args.negative_entry_replacement_gain, 0.0, 8.0)),
        "baseline_defaults_preserved": True,
        "leakage_free": True,
    }

    if args.run_mt5:
        if not args.seed.is_file():
            raise FileNotFoundError(f"protected seed missing: {args.seed}")
        args.candidate.parent.mkdir(parents=True, exist_ok=True)
        normalized_config = _normalized_config(
            config,
            normalized_mode,
            args.online_update_scale,
            stp_input_clip,
            args.hidden_normalization_mode,
            args.hidden_normalization_target_rms,
            args.hidden_normalization_clip,
            args.hidden_normalization_eps,
            args.context_expectancy_entry_gate,
            args.context_direction_entry_gate,
            args.context_direction_side_entry_gate,
            args.context_direction_min_timing_quality,
            args.output_normalization_mode,
            args.output_normalization_target_rms,
            args.output_normalization_clip,
            args.selector_loss_streak_penalty,
            args.selector_policy_group_loss_streak_penalty,
            args.timing_network,
            args.timing_network_action_scale,
            args.timing_network_update_scale,
            args.timing_network_learning_rate,
            args.timing_network_hidden_size,
            args.timing_network_embedding_size,
            args.timing_network_bptt_window,
            args.timing_network_eligibility_decay,
            args.timing_network_gradient_clip,
            args.negative_context_credit_multiplier,
            args.negative_entry_replacement_gain,
        )
        from self_growing_trader.chuyen_gia_policy import huan_luyen_he_chuyen_gia_mt5

        try:
            training = huan_luyen_he_chuyen_gia_mt5(
                normalized_config,
                output_path=args.candidate,
                seed_system_path=args.seed,
                runner_dir=args.runner,
                timeout_sec=int(args.timeout_sec),
                max_wall_sec=int(args.max_wall_sec),
                shutdown_terminal=True,
                from_date=args.from_date,
                to_date=args.to_date,
                model=4,
                resume_existing=False,
            )
            result["mt5"] = training
        except Exception as exc:
            result["mt5"] = {
                "completed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "candidate_exists": args.candidate.exists(),
            }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
