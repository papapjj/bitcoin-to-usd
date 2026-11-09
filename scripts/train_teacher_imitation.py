#!/usr/bin/env python3
"""Train a protected full checkpoint from one Bottom/Top teacher dataset."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import (
    TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH,
    ImitationMentorConfig,
    TeacherGeometryImitationConfig,
    TeacherImitationConfig,
    TeacherImitationPaused,
    geometry_invariant_sha256,
    geometry_trainable_sha256,
    load_teacher_imitation_data,
    load_teacher_geometry_targets,
    network_invariant_sha256,
    recurrent_imitation_frozen_sha256,
    recurrent_imitation_trainable_sha256,
    sha256_file,
    shared_five_network,
    teacher_trainable_sha256,
    train_recurrent_geometry_heads,
    train_recurrent_entry_calibrator,
)
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.teacher_imitation_dataset import (
    CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION,
)


CONFIRMED_ACTION_DATASET = (
    ROOT
    / "outputs"
    / "bottom_top_teacher_confirmed_geometry_v9_sl150_20260901"
    / "teacher_confirmed_geometry_v9_sl150.npz"
).resolve()
RETIRED_UI_ACTION_DATASETS = frozenset({
    (
        ROOT / "outputs" / "bottom_top_teacher_confirmed_action_v8_20260830"
        / "teacher_confirmed_action_v8.npz"
    ).resolve(),
    (
        ROOT / "outputs" / "bottom_top_teacher_recurrent_v5_max_sl075_20260827"
        / "teacher_max_sl075_recurrent.npz"
    ).resolve(),
    (
        ROOT / "outputs" / "bottom_top_teacher_recurrent_v6_max_sl075_20260828"
        / "teacher_max_sl075_geometry_v6.npz"
    ).resolve(),
    (
        ROOT / "outputs" / "bottom_top_teacher_recurrent_v7_max_sl075_20260828"
        / "teacher_max_sl075_geometry_v7.npz"
    ).resolve(),
})
CURRENT_IMITATION_REPORT_SCHEMA_VERSION = 12
CURRENT_IMITATION_REPRESENTATION_SCOPE = (
    "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
)


def _atomic_pickle(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _action_resume_contract(
    *, source: Path, source_hash: str, source_bytes: int,
    dataset: Path, dataset_hash: str, config_path: Path, config_hash: str,
    output: Path, trainer_config: TeacherImitationConfig,
    mentor_config: ImitationMentorConfig | None = None,
) -> dict[str, object]:
    contract: dict[str, object] = {
        "source": str(source),
        "source_sha256": source_hash,
        "source_bytes": int(source_bytes),
        "dataset": str(dataset),
        "dataset_sha256": dataset_hash,
        "config": str(config_path),
        "config_sha256": config_hash,
        "output": str(output),
        "trainer_config": dict(trainer_config.__dict__),
        "mode": "action_only_full_recurrent_spatiotemporal_transformer_role_calibrator_v3",
    }
    if mentor_config is not None and mentor_config.enabled:
        contract["mentor_config"] = dict(mentor_config.__dict__)
    return contract


def _load_action_resume(path: Path, expected_contract: dict[str, object]) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 2:
        raise ValueError("resume imitation khong dung schema 2 temporal/role calibrator")
    if payload.get("status") not in {"paused", "checkpoint"} or payload.get("contract") != expected_contract:
        raise ValueError("resume imitation khong khop source/dataset/config/output")
    if "system" not in payload or not isinstance(payload.get("training_state"), dict):
        raise ValueError("resume imitation thieu system/training state")
    return payload


def _reset_five_runtime_after_imitation(system: object) -> int:
    """Drop stale PPO/rollout state that predates the teacher calibration."""
    registry = getattr(system, "registry", None)
    mapping = getattr(registry, "chuyen_gia_theo_context", {}) if registry is not None else {}
    reset = 0
    seen: set[int] = set()
    for expert in mapping.values():
        policy = getattr(expert, "policy", None)
        if policy is None or id(policy) in seen:
            continue
        seen.add(id(policy))
        if hasattr(policy, "five_ppo_trainer"):
            policy.five_ppo_trainer = None
            reset += 1
        for name in (
            "five_rollout_steps", "five_pattern_steps",
            "five_pending_timeout_steps", "five_lifecycle_steps",
        ):
            if hasattr(policy, name):
                setattr(policy, name, [])
        if hasattr(policy, "five_last_training_stats"):
            policy.five_last_training_stats = {}
    return reset


def _preflight_space(
    source: Path, output: Path, *, action_resume: bool = False,
    resume_file: Path | None = None,
) -> dict[str, int]:
    free = int(shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free)
    # Full recurrent pause stores the system, Adam moments, best weights and
    # exact batch cursor. Reserve enough room before a many-hour run starts.
    multiplier = 6.0 if action_resume else 1.25
    required = int(source.stat().st_size * multiplier + 512 * 1024 * 1024)
    reusable = int(resume_file.stat().st_size) if resume_file is not None and resume_file.is_file() else 0
    effective_free = free + reusable
    if effective_free < required:
        raise RuntimeError(
            f"khong du dung luong: can {required} bytes, con hieu dung {effective_free} bytes"
        )
    return {
        "free_bytes": free,
        "resume_reusable_bytes": reusable,
        "effective_free_bytes": effective_free,
        "required_bytes": required,
    }


def _verification_space(output: Path, resume_file: Path | None = None) -> dict[str, int]:
    """Report disk state without reserving capacity for a training run.

    ``--verify-only`` must reach the dataset/hash contract checks even on a
    nearly full disk.  It does not create checkpoints, resume artifacts or
    hidden caches, so applying the action-training multiplier here would hide
    the actual provenance error behind an unrelated storage error.
    """

    location = output.parent if output.parent.exists() else ROOT
    free = int(shutil.disk_usage(location).free)
    reusable = (
        int(resume_file.stat().st_size)
        if resume_file is not None and resume_file.is_file()
        else 0
    )
    return {
        "free_bytes": free,
        "resume_reusable_bytes": reusable,
        "effective_free_bytes": free + reusable,
        "required_bytes": 0,
    }


def _geometry_should_reuse_hidden_cache(
    hidden_cache_path: Path | None, *, geometry_only: bool, explicit_reuse: bool
) -> bool:
    """Reuse a cache created this run or explicitly requested after a probe."""

    return bool(hidden_cache_path is not None and (explicit_reuse or not geometry_only))


def _effective_imitation_modes(
    *, action_only: bool, geometry_only: bool,
) -> tuple[bool, bool]:
    if action_only and geometry_only:
        raise ValueError("--action-only va --geometry-only loai tru nhau")
    return (not geometry_only, bool(geometry_only))


def _current_action_imitation_report_passed(report: object) -> bool:
    """Accept geometry-only input only with the current closed-loop proof."""

    if not isinstance(report, dict):
        return False
    integration = dict(report.get("mt5_integration", {}) or {})
    verification = dict(report.get("verification", {}) or {})
    training = dict(report.get("training", {}) or {})
    validation_gate = dict(training.get("validation_gate", {}) or {})
    autonomous_gate = dict(training.get("autonomous_validation_gate", {}) or {})
    return bool(
        int(report.get("schema_version", 0) or 0)
        >= CURRENT_IMITATION_REPORT_SCHEMA_VERSION
        and integration.get("entry_scope") == "all_entry_mass_v2"
        and integration.get("representation_scope")
        == CURRENT_IMITATION_REPRESENTATION_SCOPE
        and bool(integration.get("requires_runtime_teacher_retention", False))
        and bool(integration.get("requires_autonomous_continuous_validation", False))
        and bool(verification.get("candidate_reload_passed", False))
        and bool(validation_gate.get("enforced", False))
        and bool(validation_gate.get("passed", False))
        and bool(autonomous_gate.get("enforced", False))
        and bool(autonomous_gate.get("passed", False))
    )


def _resolve_imitation_dataset(
    requested: str | Path, *, action_only: bool,
) -> tuple[Path, bool]:
    requested_path = Path(requested).resolve()
    if action_only and requested_path in RETIRED_UI_ACTION_DATASETS:
        return CONFIRMED_ACTION_DATASET, True
    return requested_path, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Train imitation BUY day / SELL dinh tren candidate rieng")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--recurrent-learning-rate", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument("--false-entry-penalty", type=float, default=0.50)
    parser.add_argument(
        "--ranking-loss-weight", type=float, default=0.50,
        help="Trong so pairwise ranking ENTRY dung de hoc mau hiem ma khong lam lech prior.",
    )
    parser.add_argument(
        "--checkpoint-every-windows", type=int, default=0,
        help="Ghi resume checkpoint giua epoch moi N recurrent windows; 0 chi ghi sau validation epoch.",
    )
    parser.add_argument(
        "--mentor-guided", action=argparse.BooleanOptionalAction, default=True,
        help="Mentor tu dong dieu chinh penalty, can bang BUY/SELL va learning rate tu validation; dung --no-mentor-guided de chay baseline tinh.",
    )
    parser.add_argument(
        "--mentor-min-epoch", type=int, default=2,
        help="Epoch toi thieu truoc khi mentor bat dau dieu chinh.",
    )
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument(
        "--action-only",
        action="store_true",
        help="Hoc nonlinear mapping HOLD/BUY/SELL; bo stage geometry chua qua MT5/OOS",
    )
    parser.add_argument("--geometry-epochs", type=int, default=60)
    parser.add_argument("--geometry-learning-rate", type=float, default=0.002)
    parser.add_argument("--geometry-patience", type=int, default=10)
    parser.add_argument(
        "--hidden-cache", default="",
        help="Chi geometry-only: memmap float16 frozen hidden; action BPTT khong dung cache",
    )
    parser.add_argument("--keep-hidden-cache", action="store_true")
    parser.add_argument(
        "--reuse-hidden-cache", action="store_true",
        help="Chi geometry-only: tai su dung cache sau replay spot-check",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--resume-file", default="",
        help="Checkpoint pause exact; mac dinh <output>.resume.pkl va tu dong nap neu ton tai",
    )
    parser.add_argument(
        "--stop-request-file", default="",
        help="UI tao file nay de yeu cau luu va dung tai ranh gioi batch BPTT",
    )
    args = parser.parse_args()
    # Action-only is the production default. Geometry can run only through the
    # explicit research flag; an older packaged UI that omits --action-only
    # must never silently revive the rejected combined workflow.
    args.action_only, args.geometry_only = _effective_imitation_modes(
        action_only=bool(args.action_only), geometry_only=bool(args.geometry_only)
    )

    source = Path(args.checkpoint).resolve()
    requested_dataset_path = Path(args.dataset).resolve()
    dataset_path, dataset_migrated = _resolve_imitation_dataset(
        requested_dataset_path, action_only=bool(args.action_only)
    )
    output = Path(args.output).resolve()
    resume_file = (
        Path(args.resume_file).resolve()
        if args.resume_file else output.with_suffix(output.suffix + ".resume.pkl")
    )
    resume_report = resume_file.with_suffix(resume_file.suffix + ".json")
    stop_request_file = (
        Path(args.stop_request_file).resolve()
        if args.stop_request_file else output.with_suffix(output.suffix + ".stop-request")
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source or output.exists():
        raise ValueError("output phai la file moi va khac checkpoint goc")
    if resume_file in {source, output} or stop_request_file in {source, output, resume_file}:
        raise ValueError("resume/stop-request phai la path rieng")
    if not args.verify_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        space = _preflight_space(
            source, output, action_resume=bool(args.action_only), resume_file=resume_file
        )
    else:
        space = _verification_space(output, resume_file)
    source_stat = source.stat()
    started = time.time()
    source_hash_before = sha256_file(source)
    data = load_teacher_imitation_data(dataset_path)
    dataset_hash = sha256_file(dataset_path)
    if args.action_only and (
        int(data.manifest.get("schema_version", 0) or 0)
        != CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION
        or not bool(dict(data.manifest.get("confirmation_audit", {}) or {}).get("enabled", False))
    ):
        raise ValueError(
            "action imitation yeu cau dataset schema 9 da doi nhan sang bar xac nhan; "
            f"got schema={data.manifest.get('schema_version')} path={dataset_path}"
        )
    geometry_targets = None if args.action_only else load_teacher_geometry_targets(data)
    source_imitation_report: dict[str, object] | None = None
    source_imitation_report_path = source.with_suffix(source.suffix + ".imitation.json")
    if args.geometry_only:
        if not source_imitation_report_path.is_file():
            raise RuntimeError("--geometry-only yeu cau sidecar action imitation cua checkpoint source")
        source_imitation_report = json.loads(
            source_imitation_report_path.read_text(encoding="utf-8")
        )
        if (
            not isinstance(source_imitation_report, dict)
            or str(source_imitation_report.get("candidate_checkpoint_sha256", "")) != source_hash_before
            or not _current_action_imitation_report_passed(source_imitation_report)
        ):
            raise RuntimeError(
                "sidecar source khong chung minh action imitation hien tai da qua "
                "kiem dinh lien tuc tu dua hanh dong va reload"
            )
    print(json.dumps({
        "event": "preflight_ok", "samples": len(data.actions),
        "action_only": bool(args.action_only),
        "mentor_guided": bool(args.mentor_guided),
        "dataset_requested": str(requested_dataset_path),
        "dataset_effective": str(dataset_path),
        "dataset_migrated": dataset_migrated,
        "actions": {str(value): int((data.actions == value).sum()) for value in (0, 1, 2)},
        "geometry_targets": (
            0 if geometry_targets is None else int(geometry_targets.valid.sum())
        ),
        "splits": {name: int((data.splits == name).sum()) for name in ("train", "validation", "test")},
        **space,
    }, ensure_ascii=False), flush=True)
    if args.verify_only:
        return 0

    config_path = Path(args.config).resolve()
    config_hash = sha256_file(config_path)
    config = load_config(config_path)
    action_trainer_config = TeacherImitationConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size,
        learning_rate=args.learning_rate,
        recurrent_learning_rate=args.recurrent_learning_rate,
        patience=args.patience, seed=args.seed,
        sequence_length=args.sequence_length,
        false_entry_penalty=args.false_entry_penalty,
        ranking_loss_weight=args.ranking_loss_weight,
    )
    mentor_config = ImitationMentorConfig(
        enabled=bool(args.mentor_guided),
        min_epoch=args.mentor_min_epoch,
    )
    mentor_config.validate()
    resume_contract = _action_resume_contract(
        source=source,
        source_hash=source_hash_before,
        source_bytes=source_stat.st_size,
        dataset=dataset_path,
        dataset_hash=dataset_hash,
        config_path=config_path,
        config_hash=config_hash,
        output=output,
        trainer_config=action_trainer_config,
        mentor_config=mentor_config,
    )
    resume_payload: dict[str, object] | None = None
    if args.action_only and resume_file.is_file():
        resume_payload = _load_action_resume(resume_file, resume_contract)
        system = resume_payload["system"]
        print(json.dumps({
            "event": "action_resume_artifact_loaded",
            "path": str(resume_file),
            "bytes": resume_file.stat().st_size,
            "sha256": sha256_file(resume_file),
        }, ensure_ascii=False), flush=True)
    else:
        system = tai_he_chuyen_gia_mot_context_mot_policy(source, config, seed=args.seed)
    # A request left by an old/aborted UI session must not pause a new launch.
    stop_request_file.unlink(missing_ok=True)
    network = shared_five_network(system)
    invariant_before = network_invariant_sha256(network)
    if resume_payload is None:
        recurrent_frozen_before = recurrent_imitation_frozen_sha256(network)
        recurrent_trainable_before = recurrent_imitation_trainable_sha256(network)
    else:
        saved_training_state = dict(resume_payload["training_state"])
        recurrent_frozen_before = str(saved_training_state["frozen_before"])
        recurrent_trainable_before = str(saved_training_state["trainable_before"])
    cache_width = int(network.hidden_size)
    cache_representation = "hidden"
    hidden_ram_bytes = int(len(data.actions) * cache_width * 4)
    hidden_cache_path: Path | None = None
    if args.action_only and (args.hidden_cache or args.reuse_hidden_cache):
        raise ValueError(
            "action BPTT recompute representation moi moi epoch; hidden cache frozen da bi loai bo"
        )
    if not args.action_only and (args.hidden_cache or hidden_ram_bytes > 2 * 1024**3):
        hidden_cache_path = (
            Path(args.hidden_cache).resolve() if args.hidden_cache
            else output.with_suffix(output.suffix + ".hidden.f16.mmap")
        )
        hidden_cache_path.parent.mkdir(parents=True, exist_ok=True)
        hidden_cache_bytes = int(len(data.actions) * cache_width * 2)
        free = int(shutil.disk_usage(hidden_cache_path.parent).free)
        reusable_cache_bytes = int(hidden_cache_path.stat().st_size) if hidden_cache_path.is_file() else 0
        effective_free = free + reusable_cache_bytes
        required = hidden_cache_bytes + int(source.stat().st_size * 1.25) + 512 * 1024**2
        if effective_free < required:
            raise RuntimeError(
                "khong du dung luong cho hidden memmap: "
                f"can {required} bytes, con hieu dung {effective_free} bytes"
            )
        print(json.dumps({
            "event": "hidden_cache_enabled",
            "path": str(hidden_cache_path),
            "dtype": "float16",
            "bytes": hidden_cache_bytes,
            "avoided_ram_bytes": hidden_ram_bytes,
            "representation": cache_representation,
            "representation_width": cache_width,
            "free_bytes": free,
            "reusable_cache_bytes": reusable_cache_bytes,
            "effective_free_bytes": effective_free,
        }, ensure_ascii=False), flush=True)
    if args.reuse_hidden_cache and hidden_cache_path is None:
        raise ValueError("--reuse-hidden-cache yeu cau --hidden-cache")
    if args.geometry_only:
        calibrator = network.heads.entry_role_calibrator
        if not bool(getattr(calibrator, "teacher_imitation_trained", False)):
            raise RuntimeError("--geometry-only yeu cau checkpoint da qua action imitation")
        result = {
            "skipped": True,
            "reason": "geometry_only_on_existing_action_imitation",
            "network_invariant_sha256": invariant_before,
            "teacher_trainable_after_sha256": teacher_trainable_sha256(network),
        }
    else:
        def persist_action_resume(
            state: dict[str, object], *, status: str, event: str,
        ) -> dict[str, object]:
            if sha256_file(source) != source_hash_before or source.stat().st_size != source_stat.st_size:
                raise RuntimeError("checkpoint goc da bi thay doi truoc khi luu resume")
            if recurrent_imitation_frozen_sha256(network) != recurrent_frozen_before:
                raise RuntimeError("frozen-head hash thay doi truoc khi luu resume")
            _atomic_pickle({
                "schema_version": 2,
                "status": status,
                "promotable": False,
                "contract": resume_contract,
                "system": system,
                "training_state": state,
                "saved_at_unix": time.time(),
            }, resume_file)
            resume_hash = sha256_file(resume_file)
            resume_metadata = {
                "schema_version": 2,
                "status": status,
                "promotable": False,
                "resume_file": str(resume_file),
                "resume_sha256": resume_hash,
                "resume_bytes": resume_file.stat().st_size,
                "resume_epoch": state["epoch"],
                "windows_done": state["next_batch_start"],
                "windows_total": len(state["permutation"]),
                "completed_epochs": len(state["history"]),
                "contract": resume_contract,
            }
            _atomic_json(resume_metadata, resume_report)
            print(json.dumps({
                "event": event,
                **resume_metadata,
            }, ensure_ascii=False), flush=True)
            return resume_metadata

        def persist_epoch_checkpoint(state: dict[str, object]) -> None:
            checkpoint_kind = str(state.get("checkpoint_kind", "epoch"))
            event = (
                "action_batch_checkpoint_saved"
                if checkpoint_kind == "batch"
                else "action_epoch_checkpoint_saved"
            )
            persist_action_resume(
                state,
                status="checkpoint",
                event=event,
            )

        try:
            result = train_recurrent_entry_calibrator(
                network,
                data,
                action_trainer_config,
                progress=lambda row: print(json.dumps({
                    "event": str(row.get("event", "action_epoch")),
                    **{key: value for key, value in row.items() if key != "event"},
                }), flush=True),
                stop_requested=stop_request_file.is_file,
                resume_state=(
                    None if resume_payload is None
                    else dict(resume_payload["training_state"])
                ),
                checkpoint=persist_epoch_checkpoint,
                mentor_config=mentor_config,
                checkpoint_every_windows=args.checkpoint_every_windows,
            )
        except TeacherImitationPaused as paused:
            state = paused.resume_state
            print(json.dumps({
                "event": "action_checkpoint_saving",
                "path": str(resume_file),
                "epoch": state["epoch"],
                "windows_done": state["next_batch_start"],
                "windows_total": len(state["permutation"]),
            }, ensure_ascii=False), flush=True)
            persist_action_resume(
                state,
                status="paused",
                event="action_checkpoint_saved",
            )
            stop_request_file.unlink(missing_ok=True)
            return 0
        if result["recurrent_imitation_frozen_sha256"] != recurrent_frozen_before:
            raise RuntimeError("head frozen thay doi trong recurrent imitation")
    action_trainable_hash = teacher_trainable_sha256(network)
    if args.action_only:
        reset_trainer_count = _reset_five_runtime_after_imitation(system)
        _atomic_pickle(system, output)
        output_hash = sha256_file(output)
        reloaded = tai_he_chuyen_gia_mot_context_mot_policy(
            output, config, seed=args.seed
        )
        reloaded_network = shared_five_network(reloaded)
        if recurrent_imitation_frozen_sha256(reloaded_network) != recurrent_frozen_before:
            raise RuntimeError("checkpoint action-only reload lam sai frozen head weights")
        if (
            recurrent_imitation_trainable_sha256(reloaded_network)
            != result["recurrent_imitation_trainable_after_sha256"]
        ):
            raise RuntimeError("checkpoint action-only reload lam sai recurrent representation")
        source_hash_after = sha256_file(source)
        if source_hash_after != source_hash_before or source.stat().st_size != source_stat.st_size:
            raise RuntimeError("checkpoint goc da bi thay doi trong luc train")
        report = {
            "schema_version": CURRENT_IMITATION_REPORT_SCHEMA_VERSION,
            "status": "candidate_only",
            "promotable": False,
            "training_scope": (
                "full BPTT HOLD/BUY/SELL mapping through observation encoder, "
                "LSTM, cross-agent plus temporal-token Transformer, action-query and "
                "independent BUY/SELL calibrators; independent "
                "action-style/value/geometry head weights frozen; final validation also "
                "runs continuous deterministic self-fed recurrent state"
            ),
            "mt5_integration": {
                "entry_scope": "all_entry_mass_v2",
                "representation_scope": (
                    "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
                ),
                "ppo_runtime_reset_count": reset_trainer_count,
                "requires_runtime_teacher_retention": True,
                "requires_autonomous_continuous_validation": True,
                "autonomous_validation_mode": (
                    "continuous_self_fed_deterministic_entry_v1"
                ),
                "geometry_scope": "disabled_after_negative_mt5_ab",
                "geometry_heads": [],
            },
            "base_checkpoint": str(source),
            "base_checkpoint_sha256": source_hash_before,
            "base_checkpoint_bytes": source_stat.st_size,
            "dataset_requested": str(requested_dataset_path),
            "dataset_migrated_from_retired_ui_path": dataset_migrated,
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_manifest": data.manifest,
            "candidate_checkpoint": str(output),
            "candidate_checkpoint_sha256": output_hash,
            "candidate_checkpoint_bytes": output.stat().st_size,
            "elapsed_seconds": time.time() - started,
            "training": result,
            "verification": {
                "source_hash_unchanged": True,
                "frozen_head_weights_unchanged": True,
                "recurrent_representation_updated": (
                    result["recurrent_imitation_trainable_after_sha256"]
                    != recurrent_trainable_before
                ),
                "shared_downstream_logits_may_change": True,
                "candidate_reload_passed": True,
                "future_fields_used_as_inputs": False,
                "test_split_used_for_selection": False,
                "geometry_stage_executed": False,
                "mt5_oos_proven": False,
            },
        }
        report_path = output.with_suffix(output.suffix + ".imitation.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        resume_file.unlink(missing_ok=True)
        resume_report.unlink(missing_ok=True)
        stop_request_file.unlink(missing_ok=True)
        if hidden_cache_path is not None and not args.keep_hidden_cache:
            gc.collect()
            hidden_cache_path.unlink(missing_ok=True)
        print(json.dumps({
            "event": "training_complete",
            "output": str(output),
            "report": str(report_path),
            "action_only": True,
            "action_test_before": result["before"]["test"],
            "action_test_after": result["after"]["test"],
        }, ensure_ascii=False), flush=True)
        return 0
    geometry_result = train_recurrent_geometry_heads(
        network,
        data,
        TeacherGeometryImitationConfig(
            epochs=args.geometry_epochs, batch_size=args.batch_size,
            feature_batch_size=args.feature_batch_size,
            learning_rate=args.geometry_learning_rate,
            patience=args.geometry_patience, seed=args.seed,
            sequence_length=args.sequence_length,
        ),
        progress=lambda row: print(json.dumps({"event": "geometry_epoch", **row}), flush=True),
        hidden_cache_path=hidden_cache_path,
        # A full action->geometry run has already materialized this cache in
        # the action stage. Geometry-only is the first consumer and must build
        # it instead of pretending an empty path is reusable.
        reuse_hidden_cache=_geometry_should_reuse_hidden_cache(
            hidden_cache_path,
            geometry_only=bool(args.geometry_only),
            explicit_reuse=bool(args.reuse_hidden_cache),
        ),
    )
    reset_trainer_count = _reset_five_runtime_after_imitation(system)
    _atomic_pickle(system, output)
    output_hash = sha256_file(output)
    reloaded = tai_he_chuyen_gia_mot_context_mot_policy(output, config, seed=args.seed)
    reloaded_network = shared_five_network(reloaded)
    if geometry_invariant_sha256(reloaded_network) != geometry_result["geometry_invariant_sha256"]:
        raise RuntimeError("checkpoint reload lam sai parameter frozen ngoai geometry")
    if teacher_trainable_sha256(reloaded_network) != action_trainable_hash:
        raise RuntimeError("checkpoint reload lam sai teacher action calibration")
    if geometry_trainable_sha256(reloaded_network) != geometry_result["geometry_trainable_after_sha256"]:
        raise RuntimeError("checkpoint reload lam sai teacher geometry")
    source_hash_after = sha256_file(source)
    if source_hash_after != source_hash_before or source.stat().st_size != source_stat.st_size:
        raise RuntimeError("checkpoint goc da bi thay doi trong luc train")
    inherited_action_training = dict(
        source_imitation_report.get("training", {}) if source_imitation_report is not None else result
    )
    report = {
        "schema_version": CURRENT_IMITATION_REPORT_SCHEMA_VERSION,
        "status": "candidate_only",
        "promotable": False,
        "training_scope": (
            "continuous self-fed recurrent action validation inherited from the exact "
            "source plus delay/SL/opposite-pivot TP/trailing geometry; base Five "
            "encoder/LSTM/action style/value heads frozen"
        ),
        "mt5_integration": {
            "entry_scope": "all_entry_mass_v2",
            "representation_scope": CURRENT_IMITATION_REPRESENTATION_SCOPE,
            "ppo_runtime_reset_count": reset_trainer_count,
            "requires_runtime_teacher_retention": True,
            "requires_autonomous_continuous_validation": True,
            "autonomous_validation_mode": "continuous_self_fed_deterministic_entry_v1",
            "geometry_scope": "accepted_pivot_entries_v1",
            "geometry_heads": ["delay", "sl", "tp", "trailing"],
            "geometry_objective": "ordinal_log_distance_v2",
            "execution_contract_version": 7,
        },
        "base_checkpoint": str(source),
        "base_checkpoint_sha256": source_hash_before,
        "base_checkpoint_bytes": source_stat.st_size,
        "parent_imitation_report": (
            str(source_imitation_report_path) if source_imitation_report is not None else ""
        ),
        "parent_imitation_report_sha256": (
            sha256_file(source_imitation_report_path) if source_imitation_report is not None else ""
        ),
        "inherited_action_imitation": (
            {
                "report_schema_version": int(source_imitation_report.get("schema_version", 0) or 0),
                "training_scope": source_imitation_report.get("training_scope", ""),
                "mt5_integration": source_imitation_report.get("mt5_integration", {}),
                "verification": source_imitation_report.get("verification", {}),
            }
            if source_imitation_report is not None else {}
        ),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest": data.manifest,
        "candidate_checkpoint": str(output),
        "candidate_checkpoint_sha256": output_hash,
        "candidate_checkpoint_bytes": output.stat().st_size,
        "elapsed_seconds": time.time() - started,
        "training": inherited_action_training,
        "action_training_stage": result,
        "geometry_training": geometry_result,
        "verification": {
            "source_hash_unchanged": True,
            "frozen_network_hash_unchanged": True,
            "protected_action_logits_unchanged": True,
            "action_calibration_preserved_during_geometry_train": True,
            "geometry_checkpoint_reload_passed": True,
            "non_entry_roles_unchanged": True,
            "candidate_reload_passed": True,
            "autonomous_validation_inherited": bool(source_imitation_report is not None),
            "future_fields_used_as_inputs": False,
            "mt5_oos_proven": False,
        },
    }
    report_path = output.with_suffix(output.suffix + ".imitation.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if hidden_cache_path is not None and not args.keep_hidden_cache:
        gc.collect()
        hidden_cache_path.unlink(missing_ok=True)
    print(json.dumps({
        "event": "training_complete", "output": str(output), "report": str(report_path),
        "action_training_skipped": bool(result.get("skipped", False)),
        "geometry_test_before": geometry_result["before"]["test"],
        "geometry_test_after": geometry_result["after"]["test"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
