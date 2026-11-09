"""Train one conservative V13 continuation, then run paired frozen MT5 OOS.

The protected seed is never overwritten.  This is intentionally a single
candidate artifact followed by several OOS windows: it limits disk pressure
while making it impossible to select a different candidate per lucky day.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import atexit
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from self_growing_trader.chuyen_gia_policy import (
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _mt5_best_fitness,
    _mt5_best_metrics,
    evaluate_policy_mt5_oos,
    huan_luyen_he_chuyen_gia_mt5,
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config
from self_growing_trader.learning_integrity import (
    frozen_oos_promotion_gate,
    paired_oos_non_regression_gate,
    sha256_file,
)
from self_growing_trader.mt5_runner import MT5_MODEL_REAL_TICKS


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / os.environ.get(
    "V13_SEED_SYSTEM",
    "outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl",
)
CANDIDATE = ROOT / os.environ.get(
    "V13_CONSERVATIVE_CANDIDATE",
    "outputs/he_chuyen_gia/candidates/v13_conservative_candidate.pkl",
)
OUT = ROOT / os.environ.get(
    "V13_CONSERVATIVE_OUT",
    "outputs/he_chuyen_gia/audit/v13_conservative_matrix",
)
RUNNER = ROOT / os.environ.get("V13_RUNNER_DIR", "mt5_v2_runner")
TRAIN_FROM = os.environ.get("V13_TRAIN_FROM", "2025.05.15")
TRAIN_TO = os.environ.get("V13_TRAIN_TO", "2025.05.29")
MODEL = int(os.environ.get("V13_MODEL", str(MT5_MODEL_REAL_TICKS)))
KEEP_SPECIALISTS = str(os.environ.get("V13_KEEP_SPECIALISTS", "0")).strip().lower() in {"1", "true", "yes", "on"}
# A portable clone must close its own terminal after each arm, otherwise the
# next /config launch can be forwarded to an idle terminal and never start a
# fresh Strategy Tester. The login terminal remains untouched because callers
# opt into this only for the isolated runner.
SHUTDOWN_TERMINAL = str(os.environ.get("V13_CONSERVATIVE_SHUTDOWN_TERMINAL", "0")).strip().lower() in {"1", "true", "yes", "on"}
# Direction evidence is observable/learnable but is not a hard one-sided
# entry veto in the default continuation profile.  Explicit gate experiments
# can still opt in with V13_DIRECTION_GATE=1.
DIRECTION_GATE = str(os.environ.get("V13_DIRECTION_GATE", "0")).strip().lower() in {"1", "true", "yes", "on"}
SIDE_GATE = str(os.environ.get("V13_SIDE_GATE", "0")).strip().lower() in {"1", "true", "yes", "on"}
EXPECTANCY_GATE = str(os.environ.get("V13_EXPECTANCY_GATE", "0")).strip().lower() in {"1", "true", "yes", "on"}
EXPECTANCY_MIN_SUPPORT = float(os.environ.get("V13_EXPECTANCY_MIN_SUPPORT", "12"))
EXPECTANCY_MAX_LCB = float(os.environ.get("V13_EXPECTANCY_MAX_LCB", "-0.15"))
UPDATE_SCALE = float(os.environ.get("V13_UPDATE_SCALE", "0.05"))
# Hidden regularization is a system invariant for every arm.  The seed file
# remains immutable, but its runtime copy is migrated before evaluation so a
# paired matrix never compares a regularized candidate with an unregularized
# baseline.
CANDIDATE_HIDDEN_NORMALIZATION_MODE = "layernorm_v1"
CANDIDATE_OUTPUT_NORMALIZATION_MODE = os.environ.get(
    "V13_CANDIDATE_OUTPUT_NORMALIZATION_MODE", "rms_v1"
).strip().lower()
CANDIDATE_HIDDEN_NORMALIZATION_TARGET_RMS = float(
    os.environ.get("V13_CANDIDATE_HIDDEN_NORMALIZATION_TARGET_RMS", "1.0")
)
CANDIDATE_OUTPUT_NORMALIZATION_TARGET_RMS = float(
    os.environ.get("V13_CANDIDATE_OUTPUT_NORMALIZATION_TARGET_RMS", "0.70")
)
DISABLE_STRUCTURAL_GROWTH = str(os.environ.get("V13_DISABLE_STRUCTURAL_GROWTH", "1")).strip().lower() in {"1", "true", "yes", "on"}
REQUIRE_REAL_TICKS = str(os.environ.get("V13_REQUIRE_REAL_TICKS", "1")).strip().lower() in {"1", "true", "yes", "on"}
ENABLE_LIVE_WATCH = str(os.environ.get("V13_ENABLE_LIVE_WATCH", "1")).strip().lower() in {"1", "true", "yes", "on"}
ENABLE_LEARNING_TELEMETRY = str(os.environ.get("V13_ENABLE_LEARNING_TELEMETRY", "1")).strip().lower() in {"1", "true", "yes", "on"}
# Allow a persisted candidate to resume only the paired OOS matrix after an
# audit/gate bug.  The protected seed is still hash-checked and never reused
# as a writable output.
RESUME_CANDIDATE = str(os.environ.get("V13_CONSERVATIVE_RESUME_CANDIDATE", "0")).strip().lower() in {"1", "true", "yes", "on"}

OOS_WINDOWS = tuple(
    tuple(item.split(":", 1))
    for item in os.environ.get(
        "V13_OOS_WINDOWS",
        "2025.05.29:2025.05.30,2025.05.30:2025.05.31,2025.05.31:2025.06.01",
    ).split(",")
    if ":" in item
)

_WATCHER: subprocess.Popen[bytes] | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


CANDIDATE_HIDDEN_NORMALIZATION_ENABLED = True
CANDIDATE_HIDDEN_NORMALIZATION_LAYERS = (True, True, True, True, True, True)
CANDIDATE_HIDDEN_DROPOUT_ENABLED = True
CANDIDATE_HIDDEN_DROPOUT_RATES = (0.05, 0.10, 0.10, 0.05, 0.10, 0.10)
CANDIDATE_HIDDEN_DROPOUT_SEED = int(os.environ.get("V13_CANDIDATE_HIDDEN_DROPOUT_SEED", "0"))
CANDIDATE_HIDDEN_DROPOUT_TRAIN_ONLY = _env_bool("V13_CANDIDATE_HIDDEN_DROPOUT_TRAIN_ONLY", True)
# TimingNet starts with zeroed auxiliary heads so the legacy action surface is
# preserved.  A short MT5 chunk therefore produces only microscopic head
# movement at the legacy 5e-4 learning rate.  Candidate experiments may raise
# these values explicitly, but the protected baseline never receives them.
CANDIDATE_TIMING_NETWORK_ACTION_SCALE = float(
    os.environ.get("V13_CANDIDATE_TIMING_NETWORK_ACTION_SCALE", "0.0")
)
CANDIDATE_TIMING_NETWORK_UPDATE_SCALE = float(
    os.environ.get("V13_CANDIDATE_TIMING_NETWORK_UPDATE_SCALE", "0.75")
)
CANDIDATE_TIMING_NETWORK_LIFECYCLE_SCALE = float(
    os.environ.get("V13_CANDIDATE_TIMING_NETWORK_LIFECYCLE_SCALE", "0.50")
)
CANDIDATE_TIMING_NETWORK_LEARNING_RATE = float(
    os.environ.get("V13_CANDIDATE_TIMING_NETWORK_LEARNING_RATE", "0.003")
)
CANDIDATE_TIMING_NETWORK_BPTT_WINDOW = int(
    os.environ.get("V13_CANDIDATE_TIMING_NETWORK_BPTT_WINDOW", "24")
)
CANDIDATE_TIMING_NETWORK_SWATS_ENABLED = _env_bool("V13_CANDIDATE_TIMING_NETWORK_SWATS_ENABLED", True)
CANDIDATE_TIMING_NETWORK_PID_ENABLED = _env_bool("V13_CANDIDATE_TIMING_NETWORK_PID_ENABLED", True)


def _stop_live_watcher() -> None:
    global _WATCHER
    if _WATCHER is None or _WATCHER.poll() is not None:
        return
    _WATCHER.terminate()
    try:
        _WATCHER.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _WATCHER.kill()
        _WATCHER.wait(timeout=5)
    _WATCHER = None


def _start_live_watcher() -> None:
    global _WATCHER
    if not ENABLE_LIVE_WATCH:
        return
    env = os.environ.copy()
    env["V13_CONSERVATIVE_OUT"] = str(OUT)
    env["V13_RUNNER_DIR"] = str(RUNNER)
    env["V13_WATCH_RUN_ID"] = CANDIDATE.stem
    env["V13_LEARNING_EVENTS"] = str((OUT / "learning_events.jsonl").resolve())
    watcher_log = OUT / "live_learning_watcher.log"
    handle = watcher_log.open("a", encoding="utf-8")
    _WATCHER = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/watch_v13_learning_live.py")],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    atexit.register(_stop_live_watcher)


def _config():
    app = load_config(ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    he = replace(
        app.he_chuyen_gia,
        # The first conservative profile proved too restrictive in train.
        # Keep the proven role lattice by default while blocking only
        # structural mutation; env overrides allow a bounded ablation.
        entry_style_specialists_bat=KEEP_SPECIALISTS,
        position_management_specialist_bat=KEEP_SPECIALISTS,
        volatile_subexperts_bat=KEEP_SPECIALISTS,
    )
    learning = replace(
        app.learning,
        context_direction_entry_gate_enabled=DIRECTION_GATE,
        context_direction_side_entry_gate_enabled=SIDE_GATE,
        context_expectancy_entry_gate_enabled=EXPECTANCY_GATE,
        context_expectancy_entry_gate_min_support=EXPECTANCY_MIN_SUPPORT,
        context_expectancy_entry_gate_max_lcb=EXPECTANCY_MAX_LCB,
        action_recovery_enabled=False,
        action_recovery_negative_veto_enabled=False,
        runtime_fitness_guard_enabled=True,
        mt5_online_update_scale=UPDATE_SCALE,
        require_real_ticks_for_promotion=True,
        require_real_tick_evidence_for_promotion=True,
        hidden_normalization_mode="layernorm_v1",
        hidden_normalization_enabled=True,
        hidden_normalization_layers=(True,) * 6,
        output_normalization_mode="raw",
        hidden_dropout_enabled=True,
        hidden_dropout_rates=CANDIDATE_HIDDEN_DROPOUT_RATES,
        direct_rnn_full_rl_enabled=False,
        timing_network_enabled=False,
        timing_network_action_scale=0.0,
        timing_network_swats_enabled=False,
        timing_network_pid_enabled=False,
        timing_network_adamw_enabled=False,
        timing_network_cosine_annealing_enabled=False,
    )
    observability = replace(
        app.observability,
        enabled=ENABLE_LEARNING_TELEMETRY,
        event_path=str((OUT / "learning_events.jsonl").resolve()),
        decision_profile_path=str((OUT / "decision_profiles.jsonl").resolve()),
        structural_event_path=str((OUT / "structural_events.jsonl").resolve()),
        network_development_path=str((OUT / "network_development.jsonl").resolve()),
        decision_sample_every=max(1, int(os.environ.get("V13_TELEMETRY_DECISION_EVERY", "25"))),
        # Keep the proof stream readable during a 15-day run.  Lifecycle and
        # feedback counters remain complete; large topology arrays are capped
        # because they are not needed to prove policy updates.
        max_active_nodes=max(0, int(os.environ.get("V13_TELEMETRY_MAX_NODES", "20"))),
        max_active_edges=max(0, int(os.environ.get("V13_TELEMETRY_MAX_EDGES", "20"))),
        max_state_nodes=max(0, int(os.environ.get("V13_TELEMETRY_MAX_STATE_NODES", "20"))),
        max_state_edges=max(0, int(os.environ.get("V13_TELEMETRY_MAX_STATE_EDGES", "20"))),
        max_responsible_neurons=max(0, int(os.environ.get("V13_TELEMETRY_MAX_RESPONSIBLE_NEURONS", "20"))),
        max_responsible_edges=max(0, int(os.environ.get("V13_TELEMETRY_MAX_RESPONSIBLE_EDGES", "20"))),
        reset_on_process_start=True,
    )
    # Baseline arm: restore the seed's historical risk envelope and keep the
    # expectancy gate off. _candidate_config then inherits the profile's
    # isolated candidate risk/gate settings without mutating the seed.
    baseline_risk = replace(
        app.risk,
        max_risk_pct=0.02,
        max_open_risk_pct=0.05,
        enforce_entry_limits=False,
    )
    learning = replace(
        learning,
        context_expectancy_entry_gate_enabled=False,
        context_expectancy_entry_gate_min_support=12.0,
        context_expectancy_entry_gate_max_lcb=-0.15,
        context_expectancy_entry_gate_scope_fallback_enabled=False,
    )
    return replace(app, risk=baseline_risk, he_chuyen_gia=he, learning=learning, observability=observability)


def _candidate_config(base):
    """Apply candidate-only output/learning changes on the common hidden contract."""

    learning = replace(
        base.learning,
        hidden_normalization_mode=CANDIDATE_HIDDEN_NORMALIZATION_MODE,
        hidden_normalization_target_rms=CANDIDATE_HIDDEN_NORMALIZATION_TARGET_RMS,
        hidden_normalization_enabled=CANDIDATE_HIDDEN_NORMALIZATION_ENABLED,
        hidden_normalization_layers=CANDIDATE_HIDDEN_NORMALIZATION_LAYERS,
        output_normalization_mode=CANDIDATE_OUTPUT_NORMALIZATION_MODE,
        output_normalization_target_rms=CANDIDATE_OUTPUT_NORMALIZATION_TARGET_RMS,
        hidden_dropout_enabled=CANDIDATE_HIDDEN_DROPOUT_ENABLED,
        hidden_dropout_rates=CANDIDATE_HIDDEN_DROPOUT_RATES,
        hidden_dropout_seed=CANDIDATE_HIDDEN_DROPOUT_SEED,
        hidden_dropout_train_only=CANDIDATE_HIDDEN_DROPOUT_TRAIN_ONLY,
        # Both arms share hidden regularization; this candidate changes only
        # the explicitly listed output and learning controls.
        context_expectancy_entry_gate_enabled=True,
        context_expectancy_entry_gate_min_support=12.0,
        context_expectancy_entry_gate_max_lcb=-0.15,
        context_expectancy_entry_gate_scope_fallback_enabled=True,
        timing_network_action_scale=float(np.clip(CANDIDATE_TIMING_NETWORK_ACTION_SCALE, 0.0, 0.50)),
        timing_network_enabled=True,
        timing_network_update_scale=float(np.clip(CANDIDATE_TIMING_NETWORK_UPDATE_SCALE, 0.0, 1.0)),
        timing_network_lifecycle_scale=float(np.clip(CANDIDATE_TIMING_NETWORK_LIFECYCLE_SCALE, 0.0, 1.0)),
        timing_network_learning_rate=float(np.clip(CANDIDATE_TIMING_NETWORK_LEARNING_RATE, 1e-6, 0.02)),
        timing_network_bptt_window=int(np.clip(CANDIDATE_TIMING_NETWORK_BPTT_WINDOW, 1, 128)),
        timing_network_swats_enabled=CANDIDATE_TIMING_NETWORK_SWATS_ENABLED,
        timing_network_pid_enabled=CANDIDATE_TIMING_NETWORK_PID_ENABLED,
        timing_network_adamw_enabled=True,
        timing_network_cosine_annealing_enabled=True,
        direct_rnn_full_rl_enabled=True,
        direct_rnn_rl_entropy_weight=0.0,
        direct_rnn_rl_temporal_bptt_steps=8,
    )
    # Keep the seed's historical risk envelope on the baseline arm, while
    # bounding the candidate's exposure so a bad update cannot compound into a
    # large account drawdown before the frozen OOS gate can reject it.
    candidate_risk = replace(
        base.risk,
        max_risk_pct=0.005,
        max_open_risk_pct=0.015,
        enforce_entry_limits=True,
    )
    return replace(base, risk=candidate_risk, learning=learning)


def _copy_runtime(metrics: dict[str, object], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("report_path", "report.htm"),
        ("execution_ledger_path", "execution.csv"),
        ("bridge_path", "bridge.csv"),
        ("feature_audit_path", "feature_audit.csv"),
    ):
        value = metrics.get(key)
        if not value:
            continue
        source = Path(str(value))
        if source.is_file():
            shutil.copy2(source, target / name)


def main() -> None:
    if not SEED.is_file():
        raise FileNotFoundError(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    config = _config()
    candidate_config = _candidate_config(config)
    seed_hash_before = sha256_file(SEED)
    if CANDIDATE.exists() and not RESUME_CANDIDATE:
        raise FileExistsError(f"candidate da ton tai, khong ghi de: {CANDIDATE}")
    if RESUME_CANDIDATE and not CANDIDATE.is_file():
        raise FileNotFoundError(f"resume candidate khong ton tai: {CANDIDATE}")
    _start_live_watcher()

    print(json.dumps({
        "phase": "resume_candidate_oos" if RESUME_CANDIDATE else "train",
        "seed": str(SEED),
        "seed_sha256": seed_hash_before,
        "candidate": str(CANDIDATE),
        "train": [TRAIN_FROM, TRAIN_TO],
        "model": MODEL,
        "profile": (
            f"v13_bounded_update_scale_{UPDATE_SCALE:g}_roles_{int(KEEP_SPECIALISTS)}"
            f"_direction_gate_{int(DIRECTION_GATE)}_side_gate_{int(SIDE_GATE)}"
            f"_expectancy_gate_{int(bool(candidate_config.learning.context_expectancy_entry_gate_enabled))}"
            f"_hidden_{CANDIDATE_HIDDEN_NORMALIZATION_MODE}"
            f"_output_{CANDIDATE_OUTPUT_NORMALIZATION_MODE}"
            f"_dropout_{int(CANDIDATE_HIDDEN_DROPOUT_ENABLED)}"
            f"_timing_lifecycle_{CANDIDATE_TIMING_NETWORK_LIFECYCLE_SCALE:g}"
        ),
    }, ensure_ascii=False), flush=True)
    if RESUME_CANDIDATE:
        train_result = {
            "mode": "resume_existing_candidate",
            "policy": str(CANDIDATE),
            "interrupted": False,
            "date_range": {"from": TRAIN_FROM, "to": TRAIN_TO},
        }
    else:
        train_result = huan_luyen_he_chuyen_gia_mt5(
            candidate_config,
            output_path=CANDIDATE,
            seed_system_path=SEED,
            runner_dir=RUNNER,
            timeout_sec=180,
            max_wall_sec=0,
            visual=False,
            shutdown_terminal=SHUTDOWN_TERMINAL,
            from_date=TRAIN_FROM,
            to_date=TRAIN_TO,
            model=MODEL,
            seed=20260717,
            resume_existing=False,
        )
    candidate_hash = sha256_file(CANDIDATE)
    if sha256_file(SEED) != seed_hash_before:
        raise RuntimeError("protected seed SHA256 changed during candidate train")

    with CANDIDATE.open("rb") as handle:
        candidate = pickle.load(handle)
    _cau_hinh_policy_chuyen_gia_cho_mt5(candidate, candidate_config)

    windows: list[dict[str, object]] = []
    for index, (from_date, to_date) in enumerate(OOS_WINDOWS, start=1):
        print(json.dumps({"phase": "oos", "window": index, "from": from_date, "to": to_date}, ensure_ascii=False), flush=True)
        candidate_result = evaluate_policy_mt5_oos(
            candidate,
            candidate_config,
            runner_dir=RUNNER,
            timeout_sec=180,
            visual=False,
            shutdown_terminal=SHUTDOWN_TERMINAL,
            from_date=from_date,
            to_date=to_date,
            model=MODEL,
        )
        candidate_metrics = _mt5_best_metrics(candidate_result)
        # The candidate and baseline intentionally share one portable runner
        # directory. Snapshot candidate files before baseline starts, or the
        # second arm overwrites the paths stored in candidate_metrics and the
        # audit loses its paired execution proof.
        _copy_runtime(candidate_metrics, OUT / f"window_{index:02d}_candidate")
        baseline = tai_he_chuyen_gia_mot_context_mot_policy(SEED, config, seed=20260717 + index)
        _cau_hinh_policy_chuyen_gia_cho_mt5(baseline, config)
        baseline_result = evaluate_policy_mt5_oos(
            baseline,
            config,
            runner_dir=RUNNER,
            timeout_sec=180,
            visual=False,
            shutdown_terminal=SHUTDOWN_TERMINAL,
            from_date=from_date,
            to_date=to_date,
            model=MODEL,
        )
        baseline_metrics = _mt5_best_metrics(baseline_result)
        strict_gate = frozen_oos_promotion_gate(
            candidate_metrics,
            require_fitness_eligible=True,
            require_reconciled=True,
            min_trades=30,
            max_drawdown_pct=10.0,
            min_policy_expectancy_lcb=0.0,
            max_negative_context_scopes=0,
            require_feature_parity=True,
        require_real_tick_evidence=REQUIRE_REAL_TICKS,
        )
        paired_gate = paired_oos_non_regression_gate(
            candidate_metrics,
            baseline_metrics,
            min_candidate_expectancy=0.0,
            max_profit_regression=0.0,
            max_expectancy_regression=0.02,
            max_drawdown_increase_pct=1.0,
        )
        row = {
            "window": index,
            "from_date": from_date,
            "to_date": to_date,
            "candidate_fitness": _mt5_best_fitness(candidate_result),
            "candidate_metrics": candidate_metrics,
            "baseline_fitness": _mt5_best_fitness(baseline_result),
            "baseline_metrics": baseline_metrics,
            "strict_gate": strict_gate,
            "paired_gate": paired_gate,
            "eligible": bool(strict_gate.get("eligible", False) and paired_gate.get("eligible", False)),
        }
        _copy_runtime(baseline_metrics, OUT / f"window_{index:02d}_baseline")
        windows.append(row)
        (OUT / f"window_{index:02d}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"window": index, "eligible": row["eligible"], "candidate_profit": candidate_metrics.get("net_profit"), "candidate_expectancy": (candidate_metrics.get("execution_summary") or {}).get("expectancy")}, ensure_ascii=False), flush=True)

    all_eligible = bool(windows) and all(bool(row["eligible"]) for row in windows)
    promoted = ""
    if all_eligible:
        promoted_path = OUT / f"{CANDIDATE.stem}_promoted.pkl"
        shutil.copy2(CANDIDATE, promoted_path)
        promoted = str(promoted_path)
    summary = {
        "seed": str(SEED),
        "seed_sha256_before": seed_hash_before,
        "seed_sha256_after": sha256_file(SEED),
        "candidate": str(CANDIDATE),
        "candidate_sha256": candidate_hash,
        "train_result": train_result,
        "windows": windows,
        "all_windows_eligible": all_eligible,
        "promoted": promoted,
        "profile": (
            f"v13_bounded_update_scale_{UPDATE_SCALE:g}_roles_{int(KEEP_SPECIALISTS)}"
            f"_direction_gate_{int(DIRECTION_GATE)}_side_gate_{int(SIDE_GATE)}"
            f"_expectancy_gate_{int(bool(candidate_config.learning.context_expectancy_entry_gate_enabled))}"
            f"_hidden_{CANDIDATE_HIDDEN_NORMALIZATION_MODE}"
            f"_output_{CANDIDATE_OUTPUT_NORMALIZATION_MODE}"
            f"_dropout_{int(CANDIDATE_HIDDEN_DROPOUT_ENABLED)}"
        ),
        "normalization": {
            "baseline": {
                "hidden": str(config.learning.hidden_normalization_mode),
                "hidden_enabled": bool(config.learning.hidden_normalization_enabled),
                "hidden_layers": list(config.learning.hidden_normalization_layers),
                "output": str(config.learning.output_normalization_mode),
                "dropout_enabled": bool(config.learning.hidden_dropout_enabled),
                "dropout_rates": list(config.learning.hidden_dropout_rates),
            },
            "candidate": {
                "hidden": str(candidate_config.learning.hidden_normalization_mode),
                "hidden_enabled": bool(candidate_config.learning.hidden_normalization_enabled),
                "hidden_layers": list(candidate_config.learning.hidden_normalization_layers),
                "output": str(candidate_config.learning.output_normalization_mode),
                "dropout_enabled": bool(candidate_config.learning.hidden_dropout_enabled),
                "dropout_rates": list(candidate_config.learning.hidden_dropout_rates),
                "timing_network_enabled": bool(candidate_config.learning.timing_network_enabled),
                "timing_network_action_scale": float(candidate_config.learning.timing_network_action_scale),
                "timing_network_update_scale": float(candidate_config.learning.timing_network_update_scale),
                "timing_network_lifecycle_scale": float(candidate_config.learning.timing_network_lifecycle_scale),
                "timing_network_learning_rate": float(candidate_config.learning.timing_network_learning_rate),
                "timing_network_swats_enabled": bool(candidate_config.learning.timing_network_swats_enabled),
                "timing_network_pid_enabled": bool(candidate_config.learning.timing_network_pid_enabled),
            },
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": str(OUT / "summary.json"), "all_windows_eligible": all_eligible, "promoted": promoted}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
