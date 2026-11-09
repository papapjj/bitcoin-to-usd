"""Train an isolated TimingNet candidate, then freeze it on MT5 audit data.

The command is deliberately independent of ``huan-luyen-he-chuyen-gia-mt5``:
it cannot load, write or update a Direct-RNN policy artifact.  The only MT5
input is a completed/append-safe FeatureAudit CSV used as the final holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from self_growing_trader.config import load_config
from self_growing_trader.timing_network import TimingNetwork
from self_growing_trader.timingnet_candidate import (
    apply_probability_calibrator,
    CausalTimingFeatureAdapter,
    choose_validation_threshold,
    evaluate_timingnet,
    fit_isotonic_calibrator,
    load_mt5_feature_audit_samples,
    load_reference_csv_samples,
    save_json_report,
    save_timingnet_candidate,
    source_sha256,
    TIMINGNET_TRANSITION_TARGET,
    TIMINGNET_TRANSITION_TARGETS,
    timingnet_requirements,
    train_timingnet,
)

def _rates(raw: dict[str, object], horizons: tuple[int, ...]) -> dict[str, float]:
    rates = {
        "deceleration": float(np.mean(raw["deceleration_target"])) if raw["deceleration_target"] else 0.0,
    }
    transition_target = dict(raw["transition_target"])
    for horizon in horizons:
        values = transition_target[int(horizon)]
        rates[f"transition_{int(horizon)}"] = float(np.mean(values)) if values else 0.0
    return rates


def _thresholds(raw: dict[str, object], horizons: tuple[int, ...]) -> dict[str, float]:
    thresholds = {
        "deceleration": choose_validation_threshold(
            raw["deceleration_probability"], raw["deceleration_target"],
        ),
    }
    probability = dict(raw["transition_probability"])
    target = dict(raw["transition_target"])
    for horizon in horizons:
        thresholds[f"transition_{int(horizon)}"] = choose_validation_threshold(
            probability[int(horizon)], target[int(horizon)],
        )
    return thresholds


def _calibrators(raw: dict[str, object], horizons: tuple[int, ...]) -> dict[str, object]:
    calibrators: dict[str, object] = {
        "deceleration": fit_isotonic_calibrator(
            raw["deceleration_probability"], raw["deceleration_target"],
        ),
    }
    probability = dict(raw["transition_probability"])
    target = dict(raw["transition_target"])
    for horizon in horizons:
        calibrators[f"transition_{int(horizon)}"] = fit_isotonic_calibrator(
            probability[int(horizon)], target[int(horizon)],
        )
    return calibrators


def _apply_calibrators_to_raw(
    raw: dict[str, object],
    horizons: tuple[int, ...],
    calibrators: dict[str, object],
) -> dict[str, object]:
    return {
        "deceleration_probability": apply_probability_calibrator(
            raw["deceleration_probability"], calibrators.get("deceleration"),
        ).tolist(),
        "deceleration_target": list(raw["deceleration_target"]),
        "transition_probability": {
            int(horizon): apply_probability_calibrator(
                dict(raw["transition_probability"])[int(horizon)],
                calibrators.get(f"transition_{int(horizon)}"),
            ).tolist()
            for horizon in horizons
        },
        "transition_target": {
            int(horizon): list(dict(raw["transition_target"])[int(horizon)])
            for horizon in horizons
        },
    }


def _validation_score(metrics: dict[str, object]) -> float:
    deceleration = dict(metrics.get("deceleration", {}) or {})
    transition = dict(dict(metrics.get("transition", {}) or {}).get("3", {}) or {})
    # Selection happens strictly on the reference validation slice.  The MT5
    # audit is read only once for the final frozen report.
    return float(
        2.0 * float(transition.get("average_precision", 0.0))
        + float(transition.get("auc", 0.5))
        + float(transition.get("brier_improvement", 0.0))
        + float(deceleration.get("average_precision", 0.0))
        + 0.5 * float(deceleration.get("auc", 0.5))
        + 0.5 * float(deceleration.get("brier_improvement", 0.0))
    )


def _new_network(
    *,
    seed: int,
    mtf_scaling_mode: str,
    deceleration_horizon: int,
    horizons: tuple[int, ...],
    hidden_size: int,
    embedding_size: int,
    input_size: int,
    bptt_window: int,
    sequence_length: int,
    memory_decay: float,
    transition_learning_rate: float,
) -> TimingNetwork:
    network = TimingNetwork.seed(
        input_size=input_size,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        learning_rate=0.001,
        bptt_window=bptt_window,
        sequence_length=sequence_length,
        eligibility_decay=0.90,
        gradient_clip=1.0,
        mtf_scaling_mode=mtf_scaling_mode,
        seed=seed,
    )
    network.memory_decay = float(np.clip(memory_decay, 0.50, 0.99))
    # No lifecycle/PnL actor loss is configured.  AdamW is the one auxiliary
    # writer; SWATS remains in Adam because this isolated run has no realised
    # R signal from which a PID/SGD hand-off could be justified.
    network.configure_optimizer({
        "timing_network_swats_enabled": True,
        "timing_network_pid_enabled": False,
        "timing_network_adamw_enabled": True,
        "timing_network_adamw_weight_decay": 1.0e-4,
        "timing_network_cosine_annealing_enabled": True,
        "timing_network_cosine_min_factor": 0.40,
        "timing_network_cosine_cycle_steps": 512,
        "timing_network_min_projection_steps": 4096,
    })
    network.configure_deceleration({
        "timing_network_deceleration_enabled": True,
        "timing_network_deceleration_horizon": deceleration_horizon,
        "timing_network_deceleration_min_trend_strength": 0.18,
        "timing_network_deceleration_min_drop": 0.10,
        "timing_network_deceleration_update_scale": 1.0,
        "timing_network_deceleration_encoder_scale": 0.05,
        "timing_network_deceleration_l2": 1.0e-4,
    })
    network.configure_regime_transition({
        "timing_network_regime_transition_enabled": True,
        "timing_network_regime_transition_horizons": horizons,
        "timing_network_regime_transition_update_scale": 1.0,
        "timing_network_regime_transition_learning_rate": transition_learning_rate,
        "timing_network_regime_transition_encoder_scale": 0.04,
        "timing_network_regime_transition_l2": 1.0e-4,
    })
    return network


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "he_chuyen_gia_mot_context_mot_policy.yaml"))
    parser.add_argument("--reference-csv", default=r"C:\Users\Admin\Desktop\BTCUSDm.csv")
    parser.add_argument("--reference-from", default="2024-09-01")
    # The live MT5 audit begins on 2024-12-31; keep this causal reference
    # window strictly earlier so no M15 state can leak into frozen scoring.
    parser.add_argument("--reference-to", default="2024-12-30")
    parser.add_argument("--mt5-audit", default=str(ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_feature_audit.csv"))
    parser.add_argument(
        "--mt5-audit-snapshot",
        default=None,
        help="immutable copy made before frozen MT5 holdout scoring; defaults beside --report",
    )
    parser.add_argument("--output", default=str(ROOT / "outputs" / "he_chuyen_gia" / "candidates" / "timingnet_isolated_m15_candidate.pkl"))
    parser.add_argument("--report", default=str(ROOT / "outputs" / "he_chuyen_gia" / "analysis" / "timingnet_isolated_m15_report.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=16)
    parser.add_argument("--bptt-window", type=int, default=24)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="number of closed M15 rows in each causal TimingNet tensor",
    )
    parser.add_argument("--memory-decay", type=float, default=0.85)
    parser.add_argument(
        "--temporal-adapter",
        action="store_true",
        help="append 12 causal M15 dynamics to the isolated candidate only",
    )
    parser.add_argument("--transition-learning-rate", type=float, default=0.003)
    parser.add_argument("--deceleration-horizon", type=int, default=3)
    parser.add_argument("--transition-horizons", default="3,6,12")
    parser.add_argument(
        "--transition-target",
        choices=tuple(sorted(TIMINGNET_TRANSITION_TARGETS)),
        default=TIMINGNET_TRANSITION_TARGET,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    mode = str(config.features.mtf_scaling_mode)
    horizons = tuple(sorted({int(value.strip()) for value in str(args.transition_horizons).split(",") if value.strip()}))
    if not horizons or min(horizons) < 1:
        raise ValueError("transition horizons must be positive")
    if int(args.hidden_size) < 4 or int(args.embedding_size) < 4:
        raise ValueError("hidden and embedding size must be at least 4")
    if int(args.batch_size) < 16:
        raise ValueError("batch size must be at least 16")
    if not 1 <= int(args.bptt_window) <= 128:
        raise ValueError("bptt window must be within [1, 128]")
    if not 1 <= int(args.sequence_length) <= 128:
        raise ValueError("sequence length must be within [1, 128]")
    if not 0.50 <= float(args.memory_decay) <= 0.99:
        raise ValueError("memory decay must be within [0.50, 0.99]")
    adapter = CausalTimingFeatureAdapter(mtf_scaling_mode=mode) if bool(args.temporal_adapter) else None
    if not 1e-5 <= float(args.transition_learning_rate) <= 0.05:
        raise ValueError("transition learning rate must be within [1e-5, 0.05]")

    reference = load_reference_csv_samples(
        args.reference_csv,
        feature_config=config.features,
        symbol_config=config.symbol,
        start=args.reference_from,
        end=args.reference_to,
        mtf_scaling_mode=mode,
    )
    train_end = int(len(reference) * 0.70)
    calibration_end = int(len(reference) * 0.85)
    train_rows = reference[:train_end]
    calibration_rows = reference[train_end:calibration_end]
    validation_rows = reference[calibration_end:]
    if len(calibration_rows) < 180 or len(validation_rows) < 180:
        raise ValueError("reference calibration/validation split is too small")
    candidates: list[dict[str, object]] = []
    selected: TimingNetwork | None = None
    selected_report: dict[str, object] | None = None
    best_score = float("-inf")
    for offset in range(max(1, int(args.candidates))):
        network = _new_network(
            seed=int(args.seed) + offset,
            mtf_scaling_mode=mode,
            deceleration_horizon=int(args.deceleration_horizon),
            horizons=horizons,
            hidden_size=int(args.hidden_size),
            embedding_size=int(args.embedding_size),
            input_size=adapter.input_size if adapter is not None else 78,
            bptt_window=int(args.bptt_window),
            sequence_length=int(args.sequence_length),
            memory_decay=float(args.memory_decay),
            transition_learning_rate=float(args.transition_learning_rate),
        )
        train_timingnet(
            network,
            train_rows,
            epochs=max(1, int(args.epochs)),
            batch_size=int(args.batch_size),
            transition_target=str(args.transition_target),
            feature_adapter=adapter,
        )
        calibration_raw = evaluate_timingnet(
            network,
            calibration_rows,
            warmup=train_rows[-24:],
            deceleration_horizon=int(args.deceleration_horizon),
            deceleration_min_trend_strength=network.deceleration_min_trend_strength,
            deceleration_min_drop=network.deceleration_min_drop,
            transition_horizons=horizons,
            transition_target_mode=str(args.transition_target),
            feature_adapter=adapter,
            include_raw=True,
        )
        calibration_raw_values = dict(calibration_raw.pop("_raw"))
        calibrators = _calibrators(calibration_raw_values, horizons)
        calibrated_raw = _apply_calibrators_to_raw(calibration_raw_values, horizons, calibrators)
        rates = _rates(calibrated_raw, horizons)
        thresholds = _thresholds(calibrated_raw, horizons)
        calibration = evaluate_timingnet(
            network,
            calibration_rows,
            warmup=train_rows[-24:],
            deceleration_horizon=int(args.deceleration_horizon),
            deceleration_min_trend_strength=network.deceleration_min_trend_strength,
            deceleration_min_drop=network.deceleration_min_drop,
            transition_horizons=horizons,
            transition_target_mode=str(args.transition_target),
            feature_adapter=adapter,
            thresholds=thresholds,
            baseline_rates=rates,
            calibrators=calibrators,
        )
        validation = evaluate_timingnet(
            network,
            validation_rows,
            warmup=(train_rows + calibration_rows)[-24:],
            deceleration_horizon=int(args.deceleration_horizon),
            deceleration_min_trend_strength=network.deceleration_min_trend_strength,
            deceleration_min_drop=network.deceleration_min_drop,
            transition_horizons=horizons,
            transition_target_mode=str(args.transition_target),
            feature_adapter=adapter,
            thresholds=thresholds,
            baseline_rates=rates,
            calibrators=calibrators,
        )
        score = _validation_score(validation)
        candidate_summary = {
            "seed": int(args.seed) + offset,
            "validation_score": score,
            "thresholds": thresholds,
            "baseline_rates": rates,
            "calibrators": calibrators,
            "calibration": calibration,
            "validation": validation,
        }
        candidates.append(candidate_summary)
        if score > best_score:
            best_score = score
            selected = network
            selected_report = candidate_summary
    if selected is None or selected_report is None:
        raise RuntimeError("no TimingNet candidate was trained")

    # The active tester can still append its audit while this command runs.
    # Capture exactly one immutable file before scoring so the report's hash,
    # sample count and frozen result always identify the same MT5 evidence.
    audit_source = Path(args.mt5_audit)
    audit_snapshot = Path(args.mt5_audit_snapshot) if args.mt5_audit_snapshot else (
        Path(args.report).with_suffix(".mt5_feature_audit.csv")
    )
    audit_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(audit_source, audit_snapshot)

    # This actual MT5 bridge snapshot is never used during candidate
    # selection, tuning or threshold fitting.
    mt5_samples = load_mt5_feature_audit_samples(audit_snapshot, mtf_scaling_mode=mode)
    final_metrics = evaluate_timingnet(
        selected,
        mt5_samples,
        warmup=(),
        deceleration_horizon=int(args.deceleration_horizon),
        deceleration_min_trend_strength=selected.deceleration_min_trend_strength,
        deceleration_min_drop=selected.deceleration_min_drop,
        transition_horizons=horizons,
        transition_target_mode=str(args.transition_target),
        feature_adapter=adapter,
        thresholds=dict(selected_report["thresholds"]),
        baseline_rates=dict(selected_report["baseline_rates"]),
        calibrators=dict(selected_report["calibrators"]),
    )
    final_raw_metrics = evaluate_timingnet(
        selected,
        mt5_samples,
        warmup=(),
        deceleration_horizon=int(args.deceleration_horizon),
        deceleration_min_trend_strength=selected.deceleration_min_trend_strength,
        deceleration_min_drop=selected.deceleration_min_drop,
        transition_horizons=horizons,
        transition_target_mode=str(args.transition_target),
        feature_adapter=adapter,
        thresholds=dict(selected_report["thresholds"]),
        baseline_rates=dict(selected_report["baseline_rates"]),
    )
    report = {
        "schema_version": 1,
        "kind": "isolated_timingnet_m15_train",
        "policy_touched": False,
        "lifecycle_pnl_used": False,
        "mtf_scaling_mode": mode,
        "reference": {
            "path": str(Path(args.reference_csv).resolve()),
            "sha256": source_sha256(args.reference_csv),
            "from": str(args.reference_from),
            "to": str(args.reference_to),
            "samples": len(reference),
            "train_samples": len(train_rows),
            "calibration_samples": len(calibration_rows),
            "validation_samples": len(validation_rows),
        },
        "mt5_holdout": {
            "source_path": str(audit_source.resolve()),
            "snapshot_path": str(audit_snapshot.resolve()),
            "sha256": source_sha256(audit_snapshot),
            "samples": len(mt5_samples),
            "first_timestamp": mt5_samples[0].timestamp,
            "last_timestamp": mt5_samples[-1].timestamp,
        },
        "training": {
            "epochs": int(args.epochs),
            "candidate_count": int(args.candidates),
            "batch_size": int(args.batch_size),
            "deceleration_horizon_m15": int(args.deceleration_horizon),
            "transition_horizons_m15": list(horizons),
            "transition_target": str(args.transition_target),
            "hidden_size": int(args.hidden_size),
            "embedding_size": int(args.embedding_size),
            "bptt_window": int(args.bptt_window),
            "sequence_length_m15": int(selected.sequence_length),
            "memory_decay": float(args.memory_decay),
            "temporal_adapter": bool(adapter is not None),
            "timing_input_size": int(selected.input_size),
            "transition_learning_rate": float(args.transition_learning_rate),
            "selected": selected_report,
            "all_candidates": candidates,
            "optimizer_phase": selected.optimizer_phase,
            "shadow_adam_step": selected.shadow_adam_step,
            "deceleration_updates": selected.deceleration_update_count,
            "regime_transition_updates": selected.regime_transition_update_count,
        },
        "mt5_frozen_holdout": final_metrics,
        "mt5_frozen_holdout_raw": final_raw_metrics,
    }
    report["requirements"] = timingnet_requirements(final_metrics)
    report["requirements"]["target"] = str(args.transition_target)
    save_timingnet_candidate(
        args.output,
        network=selected,
        report=report,
        feature_adapter=adapter,
    )
    save_json_report(args.report, report)
    print(json.dumps({
        "candidate": str(Path(args.output).resolve()),
        "report": str(Path(args.report).resolve()),
        "requirements": report["requirements"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
