"""Sequence-aware offline imitation for historical Bottom/Top teacher labels.

The causal observations are replayed through bounded recurrent windows whose
length is configured independently from the online Five PPO unroll. Action
imitation performs real BPTT through the observation encoder, shared LSTM,
post-LSTM Transformer and action-query projection. Independent action-style,
value, target and geometry head weights remain frozen.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from .execution_contract import ACTION_COUNT, SL_ATR_BINS, TP_RR_BINS
from .five_agent_authority import five_agent_authority_mask
from .five_shared_policy import (
    ACTION_EMBED_SIZE,
    FIVE_AGENT_COUNT,
    FIVE_CROSS_AGENT_POOL_FRACTION,
    EntryRoleCalibrator,
    SharedFiveTradingPolicy,
)
from .orders import Action
from .teacher_imitation_dataset import (
    SUPPORTED_TEACHER_DATASET_SCHEMAS,
    TEACHER_DATASET_SCHEMA_VERSION,
    TEACHER_FEATURE_COUNT,
)


ALLOWED_ACTION_ROWS = (int(Action.HOLD), int(Action.MARKET_BUY), int(Action.MARKET_SELL))
TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH = 64
TEACHER_IMITATION_MAX_SEQUENCE_LENGTH = 64


class TeacherImitationPaused(RuntimeError):
    """Cooperative pause raised only after a complete optimizer batch."""

    def __init__(self, resume_state: dict[str, Any]) -> None:
        super().__init__("teacher imitation da dung an toan sau batch BPTT")
        self.resume_state = resume_state


@dataclass(frozen=True)
class TeacherImitationData:
    path: Path
    observations: np.ndarray
    actions: np.ndarray
    candidate_actions: np.ndarray
    teacher_trade_accepted: np.ndarray
    teacher_net_r: np.ndarray
    agent_ids: np.ndarray
    quality_weights: np.ndarray
    sample_times: np.ndarray
    label_finalized_times: np.ndarray
    splits: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class TeacherImitationConfig:
    epochs: int = 80
    batch_size: int = 2
    feature_batch_size: int = 16
    learning_rate: float = 0.003
    recurrent_learning_rate: float = 0.00001
    patience: int = 12
    seed: int = 0
    sequence_length: int = TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH
    false_entry_penalty: float = 0.50
    ranking_loss_weight: float = 0.50

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.feature_batch_size < 1:
            raise ValueError("epochs va batch size phai >= 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate phai > 0")
        if (
            not math.isfinite(self.recurrent_learning_rate)
            or self.recurrent_learning_rate <= 0.0
        ):
            raise ValueError("recurrent_learning_rate phai > 0")
        if self.patience < 1:
            raise ValueError("patience phai >= 1")
        if self.seed < 0:
            raise ValueError("seed phai >= 0")
        if not 2 <= self.sequence_length <= TEACHER_IMITATION_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "sequence_length phai nam trong "
                f"[2,{TEACHER_IMITATION_MAX_SEQUENCE_LENGTH}]"
            )
        if not math.isfinite(self.false_entry_penalty) or self.false_entry_penalty < 0.0:
            raise ValueError("false_entry_penalty phai huu han va >= 0")
        if not math.isfinite(self.ranking_loss_weight) or self.ranking_loss_weight < 0.0:
            raise ValueError("ranking_loss_weight phai huu han va >= 0")


@dataclass(frozen=True)
class ImitationMentorConfig:
    """Bounded, validation-driven guidance for the action imitation loop.

    The mentor never creates labels and never reads the test split.  It only
    adjusts the next epoch's false-entry penalty, role balancing and learning
    rate from causal train/validation telemetry.  Keeping this separate from
    ``TeacherImitationConfig`` preserves exact resume compatibility for the
    older static trainer.
    """

    enabled: bool = True
    min_epoch: int = 2
    false_entry_penalty_min: float = 0.10
    false_entry_penalty_max: float = 2.00
    false_entry_penalty_step: float = 0.10
    role_scale_min: float = 0.75
    role_scale_max: float = 1.50
    role_scale_step: float = 0.05
    learning_rate_multiplier_min: float = 0.50
    learning_rate_multiplier_max: float = 1.25
    learning_rate_step: float = 0.10
    teacher_confidence_weight_min: float = 0.05
    teacher_confidence_weight_max: float = 0.50
    teacher_confidence_weight_step: float = 0.05
    hard_negative_multiplier_min: float = 1.00
    hard_negative_multiplier_max: float = 2.00
    hard_negative_multiplier_step: float = 0.10
    ranking_loss_weight_min: float = 0.25
    ranking_loss_weight_max: float = 1.00
    ranking_loss_weight_step: float = 0.05
    over_entry_ratio: float = 1.15
    under_entry_ratio: float = 0.70
    role_gap_trigger: float = 0.08

    def validate(self) -> None:
        if self.min_epoch < 1:
            raise ValueError("mentor min_epoch phai >= 1")
        if (
            not math.isfinite(self.false_entry_penalty_min)
            or not math.isfinite(self.false_entry_penalty_max)
            or not 0.0 <= self.false_entry_penalty_min <= self.false_entry_penalty_max
        ):
            raise ValueError("mentor false-entry penalty range khong hop le")
        if not math.isfinite(self.false_entry_penalty_step) or self.false_entry_penalty_step <= 0.0:
            raise ValueError("mentor false-entry penalty step phai > 0")
        if (
            not math.isfinite(self.role_scale_min)
            or not math.isfinite(self.role_scale_max)
            or not 0.0 < self.role_scale_min <= self.role_scale_max
        ):
            raise ValueError("mentor role scale range khong hop le")
        if not math.isfinite(self.role_scale_step) or self.role_scale_step <= 0.0:
            raise ValueError("mentor role scale step phai > 0")
        if (
            not math.isfinite(self.learning_rate_multiplier_min)
            or not math.isfinite(self.learning_rate_multiplier_max)
            or not 0.0 < self.learning_rate_multiplier_min <= self.learning_rate_multiplier_max
        ):
            raise ValueError("mentor learning-rate range khong hop le")
        if not math.isfinite(self.learning_rate_step) or self.learning_rate_step <= 0.0:
            raise ValueError("mentor learning-rate step phai > 0")
        if (
            not math.isfinite(self.teacher_confidence_weight_min)
            or not math.isfinite(self.teacher_confidence_weight_max)
            or not 0.0 <= self.teacher_confidence_weight_min <= self.teacher_confidence_weight_max
        ):
            raise ValueError("mentor teacher-confidence range khong hop le")
        if not math.isfinite(self.teacher_confidence_weight_step) or self.teacher_confidence_weight_step <= 0.0:
            raise ValueError("mentor teacher-confidence step phai > 0")
        if (
            not math.isfinite(self.hard_negative_multiplier_min)
            or not math.isfinite(self.hard_negative_multiplier_max)
            or not 0.0 < self.hard_negative_multiplier_min <= self.hard_negative_multiplier_max
        ):
            raise ValueError("mentor hard-negative range khong hop le")
        if not math.isfinite(self.hard_negative_multiplier_step) or self.hard_negative_multiplier_step <= 0.0:
            raise ValueError("mentor hard-negative step phai > 0")
        if (
            not math.isfinite(self.ranking_loss_weight_min)
            or not math.isfinite(self.ranking_loss_weight_max)
            or not 0.0 <= self.ranking_loss_weight_min <= self.ranking_loss_weight_max
        ):
            raise ValueError("mentor ranking range khong hop le")
        if not math.isfinite(self.ranking_loss_weight_step) or self.ranking_loss_weight_step <= 0.0:
            raise ValueError("mentor ranking step phai > 0")
        if not math.isfinite(self.over_entry_ratio) or self.over_entry_ratio <= 1.0:
            raise ValueError("mentor over_entry_ratio phai > 1")
        if not math.isfinite(self.under_entry_ratio) or not 0.0 < self.under_entry_ratio < 1.0:
            raise ValueError("mentor under_entry_ratio phai nam trong (0,1)")
        if not math.isfinite(self.role_gap_trigger) or self.role_gap_trigger <= 0.0:
            raise ValueError("mentor role_gap_trigger phai > 0")


@dataclass(frozen=True)
class TeacherGeometryTargets:
    valid: np.ndarray
    delay_indices: np.ndarray
    sl_indices: np.ndarray
    tp_indices: np.ndarray
    trailing_indices: np.ndarray
    sl_atr: np.ndarray
    tp_rr: np.ndarray


@dataclass(frozen=True)
class TeacherGeometryImitationConfig:
    epochs: int = 60
    batch_size: int = 512
    feature_batch_size: int = 32
    learning_rate: float = 0.002
    patience: int = 10
    seed: int = 0
    sequence_length: int = TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.feature_batch_size < 1:
            raise ValueError("epochs va batch size geometry phai >= 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate geometry phai > 0")
        if self.patience < 1 or self.seed < 0:
            raise ValueError("patience phai >= 1 va seed khong am")
        if not 2 <= self.sequence_length <= TEACHER_IMITATION_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "sequence_length phai nam trong "
                f"[2,{TEACHER_IMITATION_MAX_SEQUENCE_LENGTH}]"
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher_imitation_data(path: str | Path) -> TeacherImitationData:
    dataset_path = Path(path).resolve()
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".json")
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"thieu dataset hoac manifest: {dataset_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {
            "schema_version", "feature_count", "observations", "actions",
            "candidate_actions", "teacher_trade_accepted", "teacher_net_r",
            "agent_ids", "quality_weights", "sample_times", "label_finalized_times",
            "splits",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"dataset imitation thieu mang: {sorted(missing)}")
        schema = int(np.asarray(data["schema_version"]).reshape(-1)[0])
        feature_count = int(np.asarray(data["feature_count"]).reshape(-1)[0])
        if schema not in SUPPORTED_TEACHER_DATASET_SCHEMAS or feature_count != TEACHER_FEATURE_COUNT:
            raise ValueError(f"dataset contract sai: schema={schema}, features={feature_count}")
        arrays = {name: np.array(data[name], copy=True) for name in required - {"schema_version", "feature_count"}}

    observations = arrays["observations"].astype(np.float32, copy=False)
    actions = arrays["actions"].astype(np.int64, copy=False)
    candidate_actions = arrays["candidate_actions"].astype(np.int64, copy=False)
    teacher_trade_accepted = arrays["teacher_trade_accepted"].astype(np.bool_, copy=False)
    teacher_net_r = arrays["teacher_net_r"].astype(np.float32, copy=False)
    agent_ids = arrays["agent_ids"].astype(np.int64, copy=False)
    weights = arrays["quality_weights"].astype(np.float32, copy=False)
    sample_times = arrays["sample_times"].astype("U19", copy=False)
    finalized = arrays["label_finalized_times"].astype("U19", copy=False)
    splits = arrays["splits"].astype("U10", copy=False)
    count = int(actions.size)
    if observations.shape != (count, TEACHER_FEATURE_COUNT):
        raise ValueError(f"observations phai co shape [N,{TEACHER_FEATURE_COUNT}]")
    for name, values in (
        ("candidate_actions", candidate_actions),
        ("teacher_trade_accepted", teacher_trade_accepted),
        ("teacher_net_r", teacher_net_r), ("agent_ids", agent_ids),
        ("quality_weights", weights), ("sample_times", sample_times),
        ("label_finalized_times", finalized), ("splits", splits),
    ):
        if values.shape != (count,):
            raise ValueError(f"{name} phai co shape [N]")
    if (
        count < 3 or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(teacher_net_r))
    ):
        raise ValueError("dataset rong/qua nho hoac chua NaN/Inf")
    if np.any(weights <= 0.0):
        raise ValueError("quality_weights phai duong")
    if not set(np.unique(actions)).issubset(ALLOWED_ACTION_ROWS):
        raise ValueError("dataset chua action ngoai HOLD/MARKET_BUY/MARKET_SELL")
    if not set(np.unique(candidate_actions)).issubset(ALLOWED_ACTION_ROWS):
        raise ValueError("candidate action ngoai contract")
    authority_ok = ((agent_ids == 0) & np.isin(actions, [0, 1])) | ((agent_ids == 1) & np.isin(actions, [0, 2]))
    if not np.all(authority_ok):
        raise ValueError("dataset vi pham Five agent authority cho BUY/SELL")
    if np.any(finalized <= sample_times):
        raise ValueError("label khong duoc finalized sau sample_time")
    if set(np.unique(splits)) != {"train", "validation", "test"}:
        raise ValueError("dataset phai co du train/validation/test")
    if np.any(sample_times[1:] < sample_times[:-1]):
        raise ValueError("dataset phai duoc sap xep theo sample_time")
    split_rank = np.asarray([{"train": 0, "validation": 1, "test": 2}[value] for value in splits])
    order = np.argsort(sample_times, kind="stable")
    if np.any(np.diff(split_rank[order]) < 0):
        raise ValueError("split khong theo thu tu thoi gian")
    for split in ("train", "validation", "test"):
        mask = splits == split
        for agent, entry_action in ((0, 1), (1, 2)):
            role_actions = actions[mask & (agent_ids == agent)]
            if role_actions.size == 0 or not {0, entry_action}.issubset(set(role_actions.tolist())):
                raise ValueError(f"split {split} thieu HOLD/entry cua agent {agent}")
    expected_labels_hash = str(manifest.get("historical_label_sha256", ""))
    labels_path = Path(str(manifest.get("historical_label_csv", "")))
    if not expected_labels_hash or not labels_path.is_file():
        raise ValueError("manifest thieu historical label source hop le")
    if sha256_file(labels_path) != expected_labels_hash:
        raise ValueError("historical label CSV da thay doi sau khi build dataset")
    if int(manifest.get("sample_count", -1)) != count:
        raise ValueError("manifest sample_count khong khop NPZ")
    return TeacherImitationData(
        path=dataset_path, observations=observations, actions=actions,
        candidate_actions=candidate_actions,
        teacher_trade_accepted=teacher_trade_accepted,
        teacher_net_r=teacher_net_r, agent_ids=agent_ids,
        quality_weights=weights, sample_times=sample_times,
        label_finalized_times=finalized, splits=splits, manifest=manifest,
    )


def load_teacher_geometry_targets(data: TeacherImitationData) -> TeacherGeometryTargets:
    """Load schema-v6 geometry targets aligned to the causal action dataset."""

    required = {
        "schema_version", "geometry_target_valid", "teacher_delay_indices",
        "teacher_sl_indices", "teacher_tp_indices", "teacher_trailing_indices",
        "teacher_sl_atr", "teacher_tp_rr",
    }
    with np.load(data.path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "dataset chua co geometry target v6; can rebuild historical labels: "
                + str(sorted(missing))
            )
        schema = int(np.asarray(archive["schema_version"]).reshape(-1)[0])
        if schema != TEACHER_DATASET_SCHEMA_VERSION:
            raise ValueError(f"geometry imitation can dataset schema {TEACHER_DATASET_SCHEMA_VERSION}, got {schema}")
        arrays = {name: np.array(archive[name], copy=True) for name in required - {"schema_version"}}
    count = len(data.actions)
    for name, values in arrays.items():
        if values.shape != (count,):
            raise ValueError(f"geometry target {name} phai co shape [N]")
    valid = arrays["geometry_target_valid"].astype(np.bool_, copy=False)
    expected_valid = data.teacher_trade_accepted & np.isin(data.actions, [1, 2])
    if not np.array_equal(valid, expected_valid) or not np.any(valid):
        raise ValueError("geometry_target_valid khong khop accepted BUY/SELL")
    indices = {
        "delay": arrays["teacher_delay_indices"].astype(np.int64, copy=False),
        "sl": arrays["teacher_sl_indices"].astype(np.int64, copy=False),
        "tp": arrays["teacher_tp_indices"].astype(np.int64, copy=False),
        "trailing": arrays["teacher_trailing_indices"].astype(np.int64, copy=False),
    }
    from .five_shared_policy import PARAMETER_BINS
    for name, values in indices.items():
        if np.any(values[valid] < 0) or np.any(values[valid] >= int(PARAMETER_BINS[name])):
            raise ValueError(f"geometry target index {name} ngoai codebook")
    sl_atr = arrays["teacher_sl_atr"].astype(np.float32, copy=False)
    tp_rr = arrays["teacher_tp_rr"].astype(np.float32, copy=False)
    if not np.all(np.isfinite(sl_atr[valid])) or not np.all(np.isfinite(tp_rr[valid])):
        raise ValueError("geometry target chua NaN/Inf")
    if np.any(sl_atr[valid] <= 0.0) or np.any(tp_rr[valid] <= 0.0):
        raise ValueError("geometry target SL/TP phai duong")
    return TeacherGeometryTargets(
        valid=valid, delay_indices=indices["delay"], sl_indices=indices["sl"],
        tp_indices=indices["tp"], trailing_indices=indices["trailing"],
        sl_atr=sl_atr, tp_rr=tp_rr,
    )


def shared_five_network(system: object) -> SharedFiveTradingPolicy:
    registry = getattr(system, "registry", None)
    mapping = getattr(registry, "chuyen_gia_theo_context", {}) if registry is not None else {}
    policies = []
    for expert in mapping.values():
        policy = getattr(expert, "policy", None)
        network = getattr(policy, "five_shared_network", None)
        if isinstance(network, SharedFiveTradingPolicy):
            policies.append((policy, network))
    unique = {id(network): (policy, network) for policy, network in policies}
    if len(unique) != 1:
        raise ValueError(f"checkpoint phai co dung mot shared Five network, tim thay {len(unique)}")
    return next(iter(unique.values()))[1]


def network_invariant_sha256(network: SharedFiveTradingPolicy) -> str:
    """Hash tensors outside the teacher-authorized action calibration surface."""
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        if name.startswith("heads.entry_role_calibrator."):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def teacher_trainable_sha256(network: SharedFiveTradingPolicy) -> str:
    digest = hashlib.sha256()
    calibrator = network.heads.entry_role_calibrator
    for _, tensor in sorted(calibrator.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


RECURRENT_IMITATION_PREFIXES = (
    "observation_encoder.",
    "lstm.",
    "post_lstm_transformer.",
    "heads.action_query.",
    "heads.entry_role_calibrator.role_calibrators.",
)


def recurrent_imitation_trainable_sha256(network: SharedFiveTradingPolicy) -> str:
    """Hash the complete representation/calibration surface owned by imitation."""

    digest = hashlib.sha256()
    matched = 0
    for name, value in sorted(network.state_dict().items()):
        if not name.startswith(RECURRENT_IMITATION_PREFIXES):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
        matched += 1
    if matched == 0:
        raise RuntimeError("khong tim thay recurrent imitation surface")
    return digest.hexdigest()


def recurrent_imitation_frozen_sha256(network: SharedFiveTradingPolicy) -> str:
    """Hash head weights that full recurrent imitation is not allowed to update."""

    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        if name.startswith(RECURRENT_IMITATION_PREFIXES):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


GEOMETRY_TEACHER_HEADS = ("delay", "sl", "tp", "trailing")


def geometry_invariant_sha256(network: SharedFiveTradingPolicy) -> str:
    """Hash every tensor except the four explicitly teacher-owned geometry heads."""

    prefixes = tuple(f"heads.parameter_heads.{name}." for name in GEOMETRY_TEACHER_HEADS)
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        if name.startswith(prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def geometry_trainable_sha256(network: SharedFiveTradingPolicy) -> str:
    digest = hashlib.sha256()
    for name in GEOMETRY_TEACHER_HEADS:
        for key, tensor in sorted(network.heads.parameter_heads[name].state_dict().items()):
            digest.update(f"{name}.{key}".encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def extract_owner_hidden_sequences(
    network: SharedFiveTradingPolicy,
    data: TeacherImitationData,
    *,
    batch_size: int = 32,
    sequence_length: int = TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH,
    cache_path: str | Path | None = None,
    representation: str = "hidden",
    allow_extended_sequence: bool = False,
    output_dtype: torch.dtype | None = None,
) -> tuple[Tensor | np.memmap, dict[str, Any]]:
    """Replay chronological teacher-forced windows through the frozen LSTM.

    A window is reset only at a split boundary or the configured public Five
    unroll boundary. Rows sharing one timestamp use one five-agent observation
    packet and must contain identical causal observations.
    """
    maximum_sequence_length = (
        8192 if allow_extended_sequence else TEACHER_IMITATION_MAX_SEQUENCE_LENGTH
    )
    if batch_size < 1 or not 2 <= sequence_length <= maximum_sequence_length:
        raise ValueError("batch_size/sequence_length recurrent khong hop le")
    if representation not in {"hidden", "action_query"}:
        raise ValueError("representation phai la hidden hoac action_query")
    device = next(network.parameters()).device
    network.eval()
    count = len(data.observations)
    output_width = network.hidden_size if representation == "hidden" else ACTION_EMBED_SIZE
    resolved_cache = Path(cache_path).resolve() if cache_path is not None else None
    if output_dtype is not None and output_dtype not in (torch.float16, torch.float32):
        raise ValueError("output_dtype phai la float16 hoac float32")
    storage_dtype = output_dtype or torch.float32
    if resolved_cache is None:
        hidden_out: Tensor | np.memmap = torch.zeros(
            (count, output_width), dtype=storage_dtype
        )
        hidden_storage = f"ram_{str(storage_dtype).split('.')[-1]}"
    else:
        resolved_cache.parent.mkdir(parents=True, exist_ok=True)
        hidden_out = np.memmap(
            resolved_cache, mode="w+", dtype=np.float16,
            shape=(count, output_width),
        )
        hidden_storage = "disk_memmap_float16"
    groups: list[tuple[str, np.ndarray]] = []
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and data.sample_times[stop] == data.sample_times[start]:
            stop += 1
        indices = np.arange(start, stop, dtype=np.int64)
        if len(set(data.splits[indices].tolist())) != 1:
            raise ValueError("mot sample_time khong duoc cat qua nhieu split")
        if len(set(data.agent_ids[indices].tolist())) != len(indices):
            raise ValueError("trung agent_id tai cung sample_time")
        if not np.all(data.observations[indices] == data.observations[start]):
            raise ValueError("cac agent cung sample_time phai co observation giong nhau")
        groups.append((str(data.splits[start]), indices))
        start = stop

    windows: list[list[np.ndarray]] = []
    for split in ("train", "validation", "test"):
        split_groups = [indices for group_split, indices in groups if group_split == split]
        windows.extend(
            split_groups[offset:offset + sequence_length]
            for offset in range(0, len(split_groups), sequence_length)
        )

    agent_template = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
    with torch.no_grad():
        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start:batch_start + batch_size]
            window_count = len(batch_windows)
            hidden = torch.zeros(
                (window_count, FIVE_AGENT_COUNT, network.hidden_size), device=device
            )
            cell = torch.zeros_like(hidden)
            previous = torch.zeros(
                (window_count, FIVE_AGENT_COUNT), dtype=torch.long, device=device
            )
            temporal_tokens, temporal_valid = network.post_lstm_transformer.empty_temporal_memory(
                window_count, device=device, dtype=hidden.dtype
            )
            for step in range(max(len(window) for window in batch_windows)):
                active = [index for index, window in enumerate(batch_windows) if step < len(window)]
                active_tensor = torch.as_tensor(active, dtype=torch.long, device=device)
                step_groups = [batch_windows[index][step] for index in active]
                obs = torch.as_tensor(
                    np.stack([data.observations[indices[0]] for indices in step_groups]),
                    dtype=torch.float32, device=device,
                )
                active_count = len(active)
                obs5 = obs[:, None, :].expand(
                    active_count, FIVE_AGENT_COUNT, -1
                ).reshape(-1, obs.shape[-1])
                ids = agent_template.repeat(active_count)
                previous_flat = previous[active_tensor].reshape(-1)
                components = network.observation_encoder.encode_components(
                    obs5, ids, previous_flat
                )
                flattened = torch.cat((
                    components["market"], components["account"], components["execution"],
                    components["positions"], components["agent"],
                    components["previous_action"],
                ), dim=-1)
                encoded = torch.tanh(
                    network.observation_encoder.pre_lstm(flattened)
                    + network.observation_encoder.pattern_residual(components["pattern_root"])
                ).reshape(active_count, FIVE_AGENT_COUNT, -1)
                width = max(1, int(round(encoded.shape[-1] * FIVE_CROSS_AGENT_POOL_FRACTION)))
                prefix = encoded[:, :, :width].max(dim=1, keepdim=True).values
                encoded = torch.cat((
                    prefix.expand(-1, FIVE_AGENT_COUNT, -1), encoded[:, :, width:]
                ), dim=-1)
                next_hidden, next_cell = network.lstm(
                    encoded.reshape(active_count * FIVE_AGENT_COUNT, -1),
                    (
                        hidden[active_tensor].reshape(active_count * FIVE_AGENT_COUNT, -1),
                        cell[active_tensor].reshape(active_count * FIVE_AGENT_COUNT, -1),
                    ),
                )
                next_hidden = next_hidden.reshape(active_count, FIVE_AGENT_COUNT, -1)
                next_cell = next_cell.reshape(active_count, FIVE_AGENT_COUNT, -1)
                active_temporal = temporal_tokens[active_tensor]
                active_temporal_valid = temporal_valid[active_tensor]
                policy_hidden = network.post_lstm_transformer(
                    next_hidden, active_temporal, active_temporal_valid
                )
                updated_temporal, updated_valid = (
                    network.post_lstm_transformer.update_temporal_memory(
                        next_hidden, active_temporal, active_temporal_valid
                    )
                )
                hidden[active_tensor] = next_hidden
                cell[active_tensor] = next_cell
                temporal_tokens[active_tensor] = updated_temporal
                temporal_valid[active_tensor] = updated_valid
                next_previous = torch.zeros(
                    (active_count, FIVE_AGENT_COUNT), dtype=torch.long, device=device
                )
                for local_index, indices in enumerate(step_groups):
                    owners = torch.as_tensor(
                        data.agent_ids[indices], dtype=torch.long, device=device
                    )
                    owner_hidden = policy_hidden[local_index, owners].detach().cpu()
                    if representation == "action_query":
                        owner_hidden = network.heads.action_query(
                            policy_hidden[local_index, owners]
                        ).detach().cpu()
                    if isinstance(hidden_out, Tensor):
                        hidden_out[indices] = owner_hidden.to(dtype=hidden_out.dtype)
                    else:
                        hidden_out[indices] = owner_hidden.numpy().astype(np.float16)
                    next_previous[local_index, owners] = torch.as_tensor(
                        data.actions[indices], dtype=torch.long, device=device
                    )
                previous[active_tensor] = next_previous
    if tuple(hidden_out.shape) != (count, output_width):
        raise RuntimeError("owner hidden extraction khong hop le")
    if isinstance(hidden_out, Tensor):
        finite = bool(torch.isfinite(hidden_out).all())
    else:
        hidden_out.flush()
        finite = all(
            bool(np.isfinite(np.asarray(hidden_out[start:start + 4096])).all())
            for start in range(0, count, 4096)
        )
    if not finite:
        raise RuntimeError("owner hidden extraction chua NaN/Inf")
    return hidden_out, {
        "sequence_length": int(sequence_length),
        "allow_extended_sequence": bool(allow_extended_sequence),
        "sequence_windows": len(windows),
        "unique_time_steps": len(groups),
        "state_reset_count": len(windows),
        "hidden_storage": hidden_storage,
        "hidden_dtype": str(hidden_out.dtype),
        "representation": representation,
        "representation_width": int(output_width),
        "hidden_cache_path": str(resolved_cache) if resolved_cache is not None else "",
        "hidden_cache_bytes": int(
            count * output_width
            * (hidden_out.element_size() if isinstance(hidden_out, Tensor) else hidden_out.dtype.itemsize)
        ),
    }


def reuse_verified_owner_hidden_cache(
    network: SharedFiveTradingPolicy,
    data: TeacherImitationData,
    *,
    cache_path: str | Path,
    sequence_length: int = TEACHER_IMITATION_DEFAULT_SEQUENCE_LENGTH,
    representation: str = "hidden",
) -> tuple[np.memmap, dict[str, Any]]:
    """Reuse a completed float16 cache only after semantic spot verification.

    The verifier replays deterministic first/middle/last recurrent windows of
    every chronological split through the current frozen network and compares
    them with the corresponding cache rows. This is stronger than trusting a
    filename or file size after a late training failure.
    """

    resolved = Path(cache_path).resolve()
    if representation not in {"hidden", "action_query"}:
        raise ValueError("representation cache khong hop le")
    output_width = network.hidden_size if representation == "hidden" else ACTION_EMBED_SIZE
    expected_bytes = int(len(data.observations) * output_width * 2)
    if not resolved.is_file() or resolved.stat().st_size != expected_bytes:
        raise ValueError(
            f"hidden cache khong du/kich thuoc sai: expected={expected_bytes}, path={resolved}"
        )
    hidden = np.memmap(
        resolved, mode="r+", dtype=np.float16,
        shape=(len(data.observations), output_width),
    )
    if not all(
        bool(np.isfinite(np.asarray(hidden[start:start + 4096])).all())
        for start in range(0, len(hidden), 4096)
    ):
        raise ValueError("hidden cache reuse chua NaN/Inf")

    groups_by_split: dict[str, list[np.ndarray]] = {
        "train": [], "validation": [], "test": [],
    }
    start = 0
    while start < len(data.observations):
        stop = start + 1
        while stop < len(data.observations) and data.sample_times[stop] == data.sample_times[start]:
            stop += 1
        split = str(data.splits[start])
        if split not in groups_by_split or not np.all(data.splits[start:stop] == split):
            raise ValueError("hidden cache verify gap split khong hop le")
        groups_by_split[split].append(np.arange(start, stop, dtype=np.int64))
        start = stop

    sampled_groups: list[np.ndarray] = []
    sampled_windows = 0
    for split in ("train", "validation", "test"):
        groups = groups_by_split[split]
        window_starts = list(range(0, len(groups), sequence_length))
        if not window_starts:
            raise ValueError(f"hidden cache verify thieu split {split}")
        selected_starts = sorted({
            window_starts[0],
            window_starts[len(window_starts) // 2],
            window_starts[-1],
        })
        for window_start in selected_starts:
            sampled_groups.extend(groups[window_start:window_start + sequence_length])
            sampled_windows += 1
    sampled_indices = np.concatenate(sampled_groups)
    subset = TeacherImitationData(
        path=data.path,
        observations=data.observations[sampled_indices],
        actions=data.actions[sampled_indices],
        candidate_actions=data.candidate_actions[sampled_indices],
        teacher_trade_accepted=data.teacher_trade_accepted[sampled_indices],
        teacher_net_r=data.teacher_net_r[sampled_indices],
        agent_ids=data.agent_ids[sampled_indices],
        quality_weights=data.quality_weights[sampled_indices],
        sample_times=data.sample_times[sampled_indices],
        label_finalized_times=data.label_finalized_times[sampled_indices],
        splits=data.splits[sampled_indices],
        manifest=data.manifest,
    )
    fresh, fresh_audit = extract_owner_hidden_sequences(
        network, subset, batch_size=max(1, sampled_windows),
        sequence_length=sequence_length, representation=representation,
    )
    fresh_np = fresh.detach().cpu().numpy() if isinstance(fresh, Tensor) else np.asarray(fresh)
    cached_np = np.asarray(hidden[sampled_indices], dtype=np.float32)
    max_abs_drift = float(np.max(np.abs(fresh_np.astype(np.float32) - cached_np)))
    if not np.allclose(fresh_np, cached_np, rtol=2e-3, atol=2e-3):
        raise RuntimeError(
            f"hidden cache khong khop network/dataset hien tai: max_abs_drift={max_abs_drift}"
        )
    return hidden, {
        "sequence_length": int(sequence_length),
        "sequence_windows": 0,
        "unique_time_steps": 0,
        "state_reset_count": 0,
        "hidden_storage": "disk_memmap_float16_reused_verified",
        "hidden_dtype": str(hidden.dtype),
        "hidden_cache_path": str(resolved),
        "hidden_cache_bytes": expected_bytes,
        "representation": representation,
        "representation_width": int(output_width),
        "reuse_semantic_probe_windows": int(sampled_windows),
        "reuse_semantic_probe_rows": int(len(sampled_indices)),
        "reuse_semantic_probe_max_abs_drift": max_abs_drift,
        "reuse_probe_sequence_audit": fresh_audit,
    }


def _full_action_logits(
    hidden: Tensor, agent_ids: Tensor, network: SharedFiveTradingPolicy
) -> Tensor:
    if int(hidden.shape[-1]) == int(network.hidden_size):
        return network.heads.primary_action_logits(hidden, agent_ids)
    elif int(hidden.shape[-1]) == ACTION_EMBED_SIZE:
        calibrator = network.heads.entry_role_calibrator
        base_logits = torch.matmul(
            hidden, network.heads.action_embedding.weight.t()
        ) / math.sqrt(float(ACTION_EMBED_SIZE))
        adjustment = calibrator.forward_projected(hidden, agent_ids)
        role_buy = (agent_ids == 0).to(dtype=base_logits.dtype)
        role_sell = (agent_ids == 1).to(dtype=base_logits.dtype)
        residual = torch.zeros_like(base_logits)
        residual[:, 0] = adjustment[:, 0] * (role_buy + role_sell)
        entry_adjustment = adjustment[:, 1]
        for action in (1, 3, 5):
            residual[:, action] = entry_adjustment * role_buy
        for action in (2, 4, 6):
            residual[:, action] = entry_adjustment * role_sell
        return base_logits + residual
    else:
        raise ValueError(
            f"query width khong hop le: {hidden.shape[-1]} (can {network.hidden_size} hoac {ACTION_EMBED_SIZE})"
        )


def _role_logits(hidden: Tensor, agent_ids: Tensor, network: SharedFiveTradingPolicy) -> Tensor:
    logits = _full_action_logits(hidden, agent_ids, network)
    hold = logits[:, 0]
    # Runtime chooses one concrete MARKET/LIMIT/STOP action by argmax. Using
    # logsumexp here adds an artificial ln(3) entry offset and trains a boundary
    # that does not match MT5 inference. The maximum style logit is the exact
    # binary HOLD-vs-any-owned-entry decision surface.
    long_entry = torch.amax(logits[:, [1, 3, 5]], dim=-1)
    short_entry = torch.amax(logits[:, [2, 4, 6]], dim=-1)
    entry = torch.where(agent_ids == 0, long_entry, short_entry)
    return torch.stack((hold, entry), dim=-1)


def _hidden_rows(hidden: Tensor | np.memmap, indices: Tensor) -> Tensor:
    if isinstance(hidden, Tensor):
        return hidden[indices]
    numpy_indices = indices.detach().cpu().numpy()
    # Advanced indexing creates a bounded batch copy. Explicit float32 keeps
    # CPU Linear dtype-compatible while the persistent cache remains float16.
    rows = np.asarray(hidden[numpy_indices], dtype=np.float32)
    return torch.from_numpy(np.array(rows, copy=True))


def _classification_statistics(targets: Tensor, predictions: Tensor) -> dict[str, float]:
    tp = float(((targets == 1) & (predictions == 1)).sum().item())
    tn = float(((targets == 0) & (predictions == 0)).sum().item())
    fp = float(((targets == 0) & (predictions == 1)).sum().item())
    fn = float(((targets == 1) & (predictions == 0)).sum().item())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp, tn, fp, fn = (counts[name] for name in ("tp", "tn", "fp", "fn"))
    total = max(1.0, tp + tn + fp + fn)
    entry_recall = tp / max(1.0, tp + fn)
    hold_recall = tn / max(1.0, tn + fp)
    precision = tp / max(1.0, tp + fp)
    fpr = fp / max(1.0, fp + tn)
    denom = math.sqrt(max(1.0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom
    f_beta = 1.25 * precision * entry_recall / max(1e-12, 0.25 * precision + entry_recall)
    return {
        "accuracy": (tp + tn) / total,
        "hold_recall": hold_recall,
        "entry_recall": entry_recall,
        "balanced_accuracy": 0.5 * (hold_recall + entry_recall),
        "entry_precision": precision,
        "entry_false_positive_rate": fpr,
        "entry_f0_5": f_beta,
        "mcc": mcc,
        "label_entry_rate": (tp + fn) / total,
        "predicted_entry_rate": (tp + fp) / total,
    }


def _empty_epoch_diagnostics() -> dict[str, float]:
    return {
        "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0,
        "entry_probability_sum": 0.0,
        "entry_probability_count": 0.0,
        "hold_entry_probability_sum": 0.0,
        "hold_entry_probability_count": 0.0,
        "teacher_confidence_loss_sum": 0.0,
        "teacher_confidence_weight_sum": 0.0,
        "teacher_hard_negative_count": 0.0,
    }


def _fit_diagnostic(
    *, epoch: int, train_metrics: dict[str, float], validation: dict[str, float],
    train_loss: float, history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Heuristic, auditable fit indicators; never a promotion decision."""

    train_ba = float(train_metrics["balanced_accuracy"])
    train_mcc = float(train_metrics["mcc"])
    validation_ba = float(validation["balanced_accuracy"])
    gap = train_ba - validation_ba
    if epoch < 3:
        underfit = "insufficient_data"
        overfit = "insufficient_data"
    else:
        if train_ba < 0.58 or train_mcc < 0.10:
            underfit = "high"
        elif train_ba < 0.65 or train_mcc < 0.20:
            underfit = "watch"
        else:
            underfit = "low"

        previous = history[-1] if history else {}
        train_improving = train_loss < float(previous.get("train_weighted_loss", math.inf))
        validation_declining = (
            validation_ba < float(previous.get("validation_balanced_accuracy", validation_ba))
            and float(validation["loss"]) > float(previous.get("validation_loss", validation["loss"]))
        )
        validation_peak = max(
            [validation_ba]
            + [float(row.get("validation_balanced_accuracy", validation_ba)) for row in history]
        )
        validation_drop = validation_peak - validation_ba
        if gap >= 0.12 and validation_drop >= 0.02:
            overfit = "high"
        elif gap >= 0.08 or (train_improving and validation_declining and validation_drop >= 0.02):
            overfit = "watch"
        else:
            overfit = "low"
    return {
        "underfit_status": underfit,
        "overfit_status": overfit,
        "train_validation_ba_gap": gap,
    }


def _initial_imitation_mentor_state(
    config: TeacherImitationConfig,
    mentor_config: ImitationMentorConfig,
) -> dict[str, Any]:
    return {
        "active_false_entry_penalty": float(
            np.clip(
                config.false_entry_penalty,
                mentor_config.false_entry_penalty_min,
                mentor_config.false_entry_penalty_max,
            )
        ),
        "role_scales": {"buy": 1.0, "sell": 1.0},
        "learning_rate_multiplier": 1.0,
        "teacher_confidence_weight": (
            mentor_config.teacher_confidence_weight_min
            if mentor_config.enabled else 0.0
        ),
        "hard_negative_multiplier": (
            mentor_config.hard_negative_multiplier_min
            if mentor_config.enabled else 1.0
        ),
        "active_ranking_loss_weight": float(config.ranking_loss_weight),
        "stage": "warmup" if mentor_config.enabled else "static",
        "last_action": "warmup" if mentor_config.enabled else "disabled",
        "decision_count": 0,
    }


def _imitation_mentor_step(
    record: dict[str, Any],
    history: list[dict[str, Any]],
    state: dict[str, Any],
    mentor_config: ImitationMentorConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose bounded next-epoch guidance from one validation record.

    This is deliberately a deterministic controller rather than a second
    learned model.  It makes the feedback auditable and replayable: the same
    record and saved state always produce the same decision on resume.
    """

    updated = {
        "active_false_entry_penalty": float(state["active_false_entry_penalty"]),
        "role_scales": {
            "buy": float(dict(state["role_scales"])["buy"]),
            "sell": float(dict(state["role_scales"])["sell"]),
        },
        "learning_rate_multiplier": float(state["learning_rate_multiplier"]),
        "teacher_confidence_weight": float(state.get("teacher_confidence_weight", 0.0)),
        "hard_negative_multiplier": float(state.get("hard_negative_multiplier", 1.0)),
        "active_ranking_loss_weight": float(
            state.get("active_ranking_loss_weight", 0.0)
        ),
        "stage": str(state.get("stage", "warmup")),
        "last_action": str(state.get("last_action", "warmup")),
        "decision_count": int(state.get("decision_count", 0)),
    }
    epoch = int(record["epoch"])
    validation_rate = float(record["validation_predicted_entry_rate"])
    label_rate = float(record.get("validation_label_entry_rate", 0.0))
    if not mentor_config.enabled:
        decision = {
            "enabled": False,
            "epoch": epoch,
            "action": "disabled",
            "signals": {},
        }
        return updated, decision
    if epoch < mentor_config.min_epoch:
        decision = {
            "enabled": True,
            "epoch": epoch,
            "action": "warmup",
            "signals": {"label_entry_rate": label_rate, "predicted_entry_rate": validation_rate},
        }
        return updated, decision

    actions: list[str] = []
    signals: dict[str, Any] = {
        "label_entry_rate": label_rate,
        "predicted_entry_rate": validation_rate,
        "entry_rate_ratio": validation_rate / max(label_rate, 1e-12),
        "overfit_status": str(record.get("overfit_status", "unknown")),
        "underfit_status": str(record.get("underfit_status", "unknown")),
    }
    over_entry = (
        label_rate > 0.0
        and validation_rate > label_rate * mentor_config.over_entry_ratio
    )
    under_entry = (
        label_rate > 0.0
        and validation_rate < label_rate * mentor_config.under_entry_ratio
    )
    low_precision = float(record.get("validation_entry_precision", 0.0)) < max(
        0.30, 1.20 * label_rate
    )
    low_recall = float(record.get("validation_entry_recall", 0.0)) < 0.20
    # Absolute floors alone missed the failure observed in the long teacher
    # run: entry rate was still above 70% of prevalence and recall was above
    # 0.20, but both BA and recall had materially regressed from the previous
    # validation epoch.  Treat that pattern as a bounded recovery signal.  It
    # changes only the next epoch's lesson weights; it never creates labels or
    # bypasses the economic/OOS gates.
    validation_ba = float(record.get("validation_balanced_accuracy", 0.0))
    validation_recall = float(record.get("validation_entry_recall", 0.0))
    previous = history[-1] if history else {}
    previous_ba = float(previous.get("validation_balanced_accuracy", validation_ba))
    previous_recall = float(previous.get("validation_entry_recall", validation_recall))
    validation_ba_regression = bool(
        history and validation_ba < previous_ba - 0.002
    )
    validation_recall_regression = bool(
        history and validation_recall < previous_recall - 0.010
    )
    recovery_entry_ratio = max(float(mentor_config.under_entry_ratio), 0.95)
    entry_mass_recovery = bool(
        label_rate > 0.0
        and validation_rate < label_rate * recovery_entry_ratio
    )
    # A severe entry collapse often also produces low precision because only a
    # handful of predictions survive the threshold.  Treating low precision
    # as an unconditional veto would keep raising the false-entry penalty and
    # make the collapse self-reinforcing.  Recovery is allowed whenever the
    # mass is clearly below the operating range, while over-entry still wins.
    validation_recovery = bool(
        not over_entry
        and (
            entry_mass_recovery
            or (
                not low_precision
                and (validation_ba_regression or validation_recall_regression)
            )
        )
    )
    role_precision_floor = max(0.30, 1.20 * label_rate)
    role_metrics = {
        "buy": {
            "precision": float(record.get("validation_buy_agent_entry_precision", 0.0)),
            "recall": float(record.get("validation_buy_agent_entry_recall", 0.0)),
            "eligible": bool(float(record.get("validation_threshold_buy_eligible", 1.0))),
        },
        "sell": {
            "precision": float(record.get("validation_sell_agent_entry_precision", 0.0)),
            "recall": float(record.get("validation_sell_agent_entry_recall", 0.0)),
            "eligible": bool(float(record.get("validation_threshold_sell_eligible", 1.0))),
        },
    }
    weak_role = min(
        role_metrics,
        key=lambda role: (
            role_metrics[role]["eligible"],
            role_metrics[role]["precision"],
            role_metrics[role]["recall"],
        ),
    )
    role_operating_point_failure = (
        not role_metrics["buy"]["eligible"]
        or not role_metrics["sell"]["eligible"]
        or any(
            float(role_metrics[role]["precision"]) < role_precision_floor
            for role in role_metrics
        )
    )
    signals.update({
        "over_entry": over_entry,
        "under_entry": under_entry,
        "low_precision": low_precision,
        "low_recall": low_recall,
        "validation_ba_regression": validation_ba_regression,
        "validation_recall_regression": validation_recall_regression,
        "entry_mass_recovery": entry_mass_recovery,
        "validation_recovery": validation_recovery,
        "previous_validation_ba": previous_ba,
        "previous_validation_recall": previous_recall,
        "recovery_entry_ratio": recovery_entry_ratio,
        "role_precision_floor": role_precision_floor,
        "buy_entry_precision": role_metrics["buy"]["precision"],
        "sell_entry_precision": role_metrics["sell"]["precision"],
        "buy_entry_recall": role_metrics["buy"]["recall"],
        "sell_entry_recall": role_metrics["sell"]["recall"],
        "buy_operating_point_eligible": role_metrics["buy"]["eligible"],
        "sell_operating_point_eligible": role_metrics["sell"]["eligible"],
        "weak_role": weak_role,
        "role_operating_point_failure": role_operating_point_failure,
    })
    if validation_recovery:
        updated["active_false_entry_penalty"] = max(
            mentor_config.false_entry_penalty_min,
            updated["active_false_entry_penalty"] - mentor_config.false_entry_penalty_step,
        )
        actions.append("lower_false_entry_penalty_for_validation_recovery")
    elif under_entry and low_recall:
        updated["active_false_entry_penalty"] = max(
            mentor_config.false_entry_penalty_min,
            updated["active_false_entry_penalty"] - mentor_config.false_entry_penalty_step,
        )
        actions.append("lower_false_entry_penalty")
    elif over_entry or low_precision:
        updated["active_false_entry_penalty"] = min(
            mentor_config.false_entry_penalty_max,
            updated["active_false_entry_penalty"] + mentor_config.false_entry_penalty_step,
        )
        actions.append("raise_false_entry_penalty")

    if validation_recovery:
        updated["hard_negative_multiplier"] = max(
            mentor_config.hard_negative_multiplier_min,
            updated["hard_negative_multiplier"] - mentor_config.hard_negative_multiplier_step,
        )
        actions.append("relax_hard_negative_for_validation_recovery")
    elif under_entry and low_recall:
        updated["hard_negative_multiplier"] = max(
            mentor_config.hard_negative_multiplier_min,
            updated["hard_negative_multiplier"] - mentor_config.hard_negative_multiplier_step,
        )
        actions.append("relax_hard_negative_curriculum")
    elif over_entry or low_precision:
        updated["hard_negative_multiplier"] = min(
            mentor_config.hard_negative_multiplier_max,
            updated["hard_negative_multiplier"] + mentor_config.hard_negative_multiplier_step,
        )
        actions.append("teach_rejected_hard_negatives")

    buy_ba = float(record.get("validation_buy_agent_balanced_accuracy", 0.0))
    sell_ba = float(record.get("validation_sell_agent_balanced_accuracy", 0.0))
    role_gap = buy_ba - sell_ba
    signals["role_balanced_accuracy_gap_buy_minus_sell"] = role_gap
    role_scales = updated["role_scales"]
    if role_gap <= -mentor_config.role_gap_trigger:
        role_scales["buy"] = min(
            mentor_config.role_scale_max,
            role_scales["buy"] + mentor_config.role_scale_step,
        )
        role_scales["sell"] = max(
            mentor_config.role_scale_min,
            role_scales["sell"] - 0.5 * mentor_config.role_scale_step,
        )
        actions.append("rebalance_toward_buy")
    elif role_gap >= mentor_config.role_gap_trigger:
        role_scales["sell"] = min(
            mentor_config.role_scale_max,
            role_scales["sell"] + mentor_config.role_scale_step,
        )
        role_scales["buy"] = max(
            mentor_config.role_scale_min,
            role_scales["buy"] - 0.5 * mentor_config.role_scale_step,
        )
        actions.append("rebalance_toward_sell")

    # The previous controller only looked at the aggregate precision/recall
    # and a role BA gap that was not populated in the epoch record.  A weak
    # SELL operating point could therefore remain ineligible while the mentor
    # reported hold_guidance.  Focus the bounded role scale on the weakest
    # operating point so the next epoch supplies a real corrective signal.
    if role_operating_point_failure:
        role_scales[weak_role] = min(
            mentor_config.role_scale_max,
            role_scales[weak_role] + mentor_config.role_scale_step,
        )
        actions.append(f"focus_{weak_role}_operating_point")

    if str(record.get("overfit_status")) == "high":
        updated["learning_rate_multiplier"] = max(
            mentor_config.learning_rate_multiplier_min,
            updated["learning_rate_multiplier"] - mentor_config.learning_rate_step,
        )
        actions.append("reduce_learning_rate_overfit")
        updated["teacher_confidence_weight"] = max(
            mentor_config.teacher_confidence_weight_min,
            updated["teacher_confidence_weight"] - mentor_config.teacher_confidence_weight_step,
        )
        updated["active_ranking_loss_weight"] = max(
            mentor_config.ranking_loss_weight_min,
            updated["active_ranking_loss_weight"] - mentor_config.ranking_loss_weight_step,
        )
    elif str(record.get("underfit_status")) == "high":
        updated["learning_rate_multiplier"] = min(
            mentor_config.learning_rate_multiplier_max,
            updated["learning_rate_multiplier"] + mentor_config.learning_rate_step,
        )
        actions.append("increase_learning_rate_underfit")

    if str(record.get("overfit_status")) != "high":
        updated["teacher_confidence_weight"] = min(
            mentor_config.teacher_confidence_weight_max,
            updated["teacher_confidence_weight"] + mentor_config.teacher_confidence_weight_step,
        )
    if validation_recovery or (low_recall and not under_entry):
        updated["active_ranking_loss_weight"] = min(
            mentor_config.ranking_loss_weight_max,
            updated["active_ranking_loss_weight"] + mentor_config.ranking_loss_weight_step,
        )
        actions.append(
            "teach_entry_recovery" if validation_recovery else "teach_entry_ranking"
        )

    if validation_recovery:
        updated["stage"] = "recover_entry_recall"
    elif role_operating_point_failure and not (over_entry or low_precision):
        updated["stage"] = "focus_role_operating_point"
    elif over_entry or low_precision:
        updated["stage"] = "suppress_false_entries"
    elif under_entry and low_recall:
        updated["stage"] = "recover_entry_recall"
    elif abs(role_gap) >= mentor_config.role_gap_trigger:
        updated["stage"] = "rebalance_roles"
    elif str(record.get("overfit_status")) == "high":
        updated["stage"] = "anti_overfit"
    else:
        updated["stage"] = "balanced"
    if not actions:
        actions.append("hold_guidance")
    updated["last_action"] = "+".join(actions)
    updated["decision_count"] += 1
    decision = {
        "enabled": True,
        "epoch": epoch,
        "action": "+".join(actions),
        "stage": updated["stage"],
        "active_false_entry_penalty": updated["active_false_entry_penalty"],
        "role_scales": dict(updated["role_scales"]),
        "learning_rate_multiplier": updated["learning_rate_multiplier"],
        "teacher_confidence_weight": updated["teacher_confidence_weight"],
        "hard_negative_multiplier": updated["hard_negative_multiplier"],
        "active_ranking_loss_weight": updated["active_ranking_loss_weight"],
        "signals": signals,
    }
    return updated, decision


def imitation_metrics(
    queries: Tensor | np.memmap,
    agent_ids: Tensor,
    actions: Tensor,
    network: SharedFiveTradingPolicy,
    mask: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        selected_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        counts = {
            "": {name: 0.0 for name in ("tp", "tn", "fp", "fn")},
            "buy_agent_": {name: 0.0 for name in ("tp", "tn", "fp", "fn")},
            "sell_agent_": {name: 0.0 for name in ("tp", "tn", "fp", "fn")},
        }
        loss_sum = 0.0
        for start in range(0, len(selected_indices), 4096):
            index = selected_indices[start:start + 4096]
            batch_agents = agent_ids[index]
            batch_targets = (actions[index] != 0).long()
            logits = _role_logits(_hidden_rows(queries, index), batch_agents, network)
            predictions = torch.argmax(logits, dim=-1)
            loss_sum += float(torch.nn.functional.cross_entropy(
                logits, batch_targets, reduction="sum"
            ).item())
            for prefix, selected in (
                ("", torch.ones_like(batch_targets, dtype=torch.bool)),
                ("buy_agent_", batch_agents == 0),
                ("sell_agent_", batch_agents == 1),
            ):
                batch_counts = _classification_statistics(
                    batch_targets[selected], predictions[selected]
                )
                for name, value in batch_counts.items():
                    counts[prefix][name] += value
        sample_count = float(len(selected_indices))
        result: dict[str, float] = {
            "samples": sample_count,
            "loss": loss_sum / max(1.0, sample_count),
        }
        for prefix, values in counts.items():
            result.update({prefix + name: value for name, value in _metrics_from_counts(values).items()})
        return result


def autonomous_entry_validation_metrics(
    network: SharedFiveTradingPolicy,
    data: TeacherImitationData,
    *,
    split: str = "validation",
    progress: Any | None = None,
) -> dict[str, Any]:
    """Evaluate entry decisions with continuous, self-fed recurrent state.

    Labels are used only for scoring.  The previous-action input and every
    recurrent state at the next market step come from the model's own
    deterministic decision, matching the causal direction used by MT5.  The
    historical dataset does not encode lifecycle authority for agents 2..4, so
    they are held at HOLD while the two entry owners retain their normal action
    styles.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split tu van hanh khong hop le: {split}")
    selected = np.flatnonzero(data.splits == split)
    if selected.size == 0:
        raise ValueError(f"split tu van hanh rong: {split}")

    groups: list[np.ndarray] = []
    offset = 0
    while offset < selected.size:
        first = int(selected[offset])
        stop = offset + 1
        while (
            stop < selected.size
            and data.sample_times[int(selected[stop])] == data.sample_times[first]
        ):
            stop += 1
        indices = selected[offset:stop]
        owners = data.agent_ids[indices]
        if len(set(owners.tolist())) != len(indices):
            raise ValueError("trung agent_id tai cung sample_time tu van hanh")
        if not np.all(data.observations[indices] == data.observations[first]):
            raise ValueError("cac agent cung sample_time phai co observation giong nhau")
        groups.append(indices)
        offset = stop

    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    allowed = five_agent_authority_mask()
    allowed[2:, :] = False
    allowed[2:, int(Action.HOLD)] = True
    allowed_tensor = torch.as_tensor(allowed, dtype=torch.bool, device=device)
    memories = network.initial_memories(device=device)
    counts = {
        "": {name: 0.0 for name in ("tp", "tn", "fp", "fn")},
        "buy_agent_": {name: 0.0 for name in ("tp", "tn", "fp", "fn")},
        "sell_agent_": {name: 0.0 for name in ("tp", "tn", "fp", "fn")},
    }
    action_counts: Counter[str] = Counter()
    action_digest = hashlib.sha256()
    binary_loss_sum = 0.0
    sample_count = 0
    missing_owner_steps = {"buy": 0, "sell": 0}
    was_training = network.training
    network.eval()
    progress_interval = max(1, math.ceil(len(groups) / 10))
    started = time.monotonic()
    with torch.no_grad():
        for step, indices in enumerate(groups, start=1):
            observation = torch.as_tensor(
                data.observations[int(indices[0])], dtype=dtype, device=device,
            ).unsqueeze(0).expand(FIVE_AGENT_COUNT, -1)
            decisions, memories, _ = network.forward_five(
                observation,
                memories,
                allowed_action_masks=allowed_tensor,
                deterministic=True,
                behavior_uniform_mix=0.0,
                behavior_entry_probability_floor=0.0,
            )
            emitted = np.asarray(
                [int(item.action_index) for item in decisions], dtype=np.int16,
            )
            action_digest.update(emitted.tobytes(order="C"))
            for action in emitted.tolist():
                action_counts[Action(int(action)).name] += 1

            present = set(int(value) for value in data.agent_ids[indices].tolist())
            missing_owner_steps["buy"] += int(0 not in present)
            missing_owner_steps["sell"] += int(1 not in present)
            for index in indices.tolist():
                owner = int(data.agent_ids[index])
                target = int(data.actions[index] != int(Action.HOLD))
                prediction = int(emitted[owner] != int(Action.HOLD))
                key = (
                    "tp" if target and prediction else
                    "tn" if not target and not prediction else
                    "fp" if not target and prediction else "fn"
                )
                counts[""][key] += 1.0
                role_prefix = "buy_agent_" if owner == 0 else "sell_agent_"
                counts[role_prefix][key] += 1.0
                probabilities = decisions[owner].policy_action_probabilities
                entry_probability = float(1.0 - probabilities[int(Action.HOLD)])
                clipped = float(np.clip(entry_probability, 1e-7, 1.0 - 1e-7))
                binary_loss_sum += -math.log(clipped if target else 1.0 - clipped)
                sample_count += 1
            if progress is not None and (
                step == len(groups) or step % progress_interval == 0
            ):
                progress({
                    "event": "autonomous_validation_progress",
                    "split": split,
                    "steps_done": step,
                    "steps_total": len(groups),
                    "percent": 100.0 * step / max(1, len(groups)),
                    "elapsed_seconds": time.monotonic() - started,
                })
    network.train(was_training)

    result: dict[str, Any] = {
        "samples": float(sample_count),
        "loss": binary_loss_sum / max(1, sample_count),
    }
    for prefix, values in counts.items():
        result.update({prefix + name: value for name, value in values.items()})
        result.update({prefix + name: value for name, value in _metrics_from_counts(values).items()})
    result.update({
        "mode": "continuous_self_fed_deterministic_entry_v1",
        "split": split,
        "time_steps": len(groups),
        "state_reset_count": 1,
        "teacher_action_feedback": False,
        "continuous_recurrent_state": True,
        "manager_agents_forced_hold": True,
        "missing_owner_steps": missing_owner_steps,
        "action_counts_all_agents": dict(action_counts),
        "action_sequence_sha256": action_digest.hexdigest(),
    })
    return result


def autonomous_entry_validation_gate(
    metrics: dict[str, Any], *, minimum_role_samples: int = 50,
) -> dict[str, Any]:
    """Apply the existing entry-quality floors to self-fed validation."""

    role_samples = min(
        float(metrics.get("buy_agent_tp", 0.0))
        + float(metrics.get("buy_agent_tn", 0.0))
        + float(metrics.get("buy_agent_fp", 0.0))
        + float(metrics.get("buy_agent_fn", 0.0)),
        float(metrics.get("sell_agent_tp", 0.0))
        + float(metrics.get("sell_agent_tn", 0.0))
        + float(metrics.get("sell_agent_fp", 0.0))
        + float(metrics.get("sell_agent_fn", 0.0)),
    )
    enforced = role_samples >= float(minimum_role_samples)
    label_entry_rate = float(metrics.get("label_entry_rate", 0.0))
    precision_floor = max(0.30, 1.20 * label_entry_rate)
    rate_ceiling = min(0.50, max(0.03, 1.20 * label_entry_rate))
    checks = {
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)) >= 0.55,
        "role_balanced_accuracy": min(
            float(metrics.get("buy_agent_balanced_accuracy", 0.0)),
            float(metrics.get("sell_agent_balanced_accuracy", 0.0)),
        ) >= 0.52,
        "role_entry_recall": min(
            float(metrics.get("buy_agent_entry_recall", 0.0)),
            float(metrics.get("sell_agent_entry_recall", 0.0)),
        ) >= 0.20,
        "mcc": float(metrics.get("mcc", 0.0)) >= 0.08,
        "entry_precision": float(metrics.get("entry_precision", 0.0)) >= precision_floor,
        "predicted_entry_rate": float(metrics.get("predicted_entry_rate", 1.0)) <= rate_ceiling,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "enforced": enforced,
        "minimum_samples_per_role": int(minimum_role_samples),
        "observed_minimum_role_samples": role_samples,
        "precision_floor": precision_floor,
        "predicted_entry_rate_ceiling": rate_ceiling,
        "checks": checks,
        "failed_checks": failed,
        "passed": bool(not enforced or not failed),
    }


def _audit_role_operating_points(
    hidden: Tensor | np.memmap,
    agent_ids: Tensor,
    actions: Tensor,
    network: SharedFiveTradingPolicy,
    validation_mask: Tensor,
    *,
    rate_ceiling_multiplier: float = 1.10,
    minimum_rate_ceiling: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Measure role-specific operating points without mutating runtime logits."""
    if not math.isfinite(rate_ceiling_multiplier) or rate_ceiling_multiplier <= 0.0:
        raise ValueError("rate_ceiling_multiplier phai huu han va > 0")
    if not math.isfinite(minimum_rate_ceiling) or not 0.0 <= minimum_rate_ceiling < 1.0:
        raise ValueError("minimum_rate_ceiling phai nam trong [0,1)")
    audit: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for role, name in ((0, "buy"), (1, "sell")):
            selected = validation_mask & (agent_ids == role)
            selected_indices = torch.nonzero(selected, as_tuple=False).reshape(-1)
            margin_parts: list[np.ndarray] = []
            target_parts: list[np.ndarray] = []
            for start in range(0, len(selected_indices), 4096):
                index = selected_indices[start:start + 4096]
                logits = _role_logits(_hidden_rows(hidden, index), agent_ids[index], network)
                margin_parts.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
                target_parts.append((actions[index] != 0).detach().cpu().numpy().astype(np.int64))
            margins = np.concatenate(margin_parts)
            targets = np.concatenate(target_parts)
            label_rate = float(targets.mean())
            rate_ceiling = max(
                float(minimum_rate_ceiling),
                float(rate_ceiling_multiplier) * label_rate,
            )
            precision_floor = max(0.30, 1.20 * label_rate)
            order = np.argsort(margins, kind="stable")
            sorted_margins = margins[order]
            sorted_targets = targets[order]
            unique_margins, first_indices, counts = np.unique(
                sorted_margins, return_index=True, return_counts=True
            )
            midpoints = (
                (unique_margins[:-1] + unique_margins[1:]) * 0.5
                if len(unique_margins) > 1 else np.empty(0, dtype=unique_margins.dtype)
            )
            thresholds = np.concatenate((
                np.asarray([float(unique_margins.min() - 1e-6)]),
                midpoints,
                np.asarray([float(unique_margins.max() + 1e-6)]),
            ))
            positive_per_value = np.add.reduceat(sorted_targets, first_indices).astype(np.float64)
            negative_per_value = counts.astype(np.float64) - positive_per_value
            cumulative_positive = np.cumsum(positive_per_value)
            cumulative_negative = np.cumsum(negative_per_value)
            low_positive = np.concatenate((
                np.asarray([0.0]), cumulative_positive[:-1], cumulative_positive[-1:],
            ))
            low_negative = np.concatenate((
                np.asarray([0.0]), cumulative_negative[:-1], cumulative_negative[-1:],
            ))
            total_positive = float(cumulative_positive[-1])
            total_negative = float(cumulative_negative[-1])
            tp = total_positive - low_positive
            fn = low_positive
            fp = total_negative - low_negative
            tn = low_negative
            entry_recall = tp / np.maximum(1.0, tp + fn)
            hold_recall = tn / np.maximum(1.0, tn + fp)
            precision = tp / np.maximum(1.0, tp + fp)
            predicted_rate = (tp + fp) / max(1.0, float(len(targets)))
            denominator = np.sqrt(np.maximum(
                1.0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            ))
            mcc = (tp * tn - fp * fn) / denominator
            f_beta = 1.25 * precision * entry_recall / np.maximum(
                1e-12, 0.25 * precision + entry_recall
            )
            balanced = 0.5 * (entry_recall + hold_recall)
            eligible = (
                (balanced >= 0.52)
                & (precision >= precision_floor)
                & (entry_recall >= 0.20)
                & (predicted_rate <= rate_ceiling)
            )
            scores = mcc + 0.50 * f_beta + 0.10 * balanced
            secondary = -np.abs(predicted_rate - label_rate)
            feasible = predicted_rate <= rate_ceiling
            selection_mask = eligible if bool(np.any(eligible)) else feasible
            constrained_scores = np.where(selection_mask, scores, -np.inf)
            best_index = int(np.lexsort((secondary, constrained_scores))[-1])
            chosen = {
                "threshold": float(thresholds[best_index]),
                "balanced_accuracy": float(balanced[best_index]),
                "entry_precision": float(precision[best_index]),
                "entry_recall": float(entry_recall[best_index]),
                "predicted_entry_rate": float(predicted_rate[best_index]),
                "label_entry_rate": label_rate,
                "rate_ceiling": float(rate_ceiling),
                "rate_ceiling_multiplier": float(rate_ceiling_multiplier),
                "mcc": float(mcc[best_index]),
                "eligible": float(eligible[best_index]),
            }
            audit[name] = chosen
    return audit


def _calibrate_role_zero_boundary_from_train(
    hidden: Tensor | np.memmap,
    agent_ids: Tensor,
    actions: Tensor,
    network: SharedFiveTradingPolicy,
    train_mask: Tensor,
) -> dict[str, dict[str, float]]:
    """Embed the best feasible train-only operating point into each role bias.

    Matching prevalence alone can leave the zero boundary on a low-quality
    block of tied/clustered margins.  Select with the same causal train labels
    used by the loss, subject to the runtime precision/recall/rate constraints.
    Validation still selects epochs and never supplies a threshold.
    """

    calibrator = network.heads.entry_role_calibrator
    operating_points = _audit_role_operating_points(
        hidden,
        agent_ids,
        actions,
        network,
        train_mask,
        # Reserve the validation gate's 1.20x allowance for temporal drift.
        # Spending it on train made a useful ranker fail at runtime when the
        # following split shifted only slightly toward larger margins.
        rate_ceiling_multiplier=1.00,
    )
    audit: dict[str, dict[str, float]] = {}
    for role, name in ((0, "buy"), (1, "sell")):
        selected = train_mask & (agent_ids == role)
        indices = torch.nonzero(selected, as_tuple=False).reshape(-1)
        if len(indices) < 2:
            raise ValueError(f"zero-boundary calibration thieu train rows cho role {name}")
        with torch.no_grad():
            logits = _role_logits(_hidden_rows(hidden, indices), agent_ids[indices], network)
            margins = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        targets = (actions[indices] != 0).detach().cpu().numpy().astype(bool)
        target_count = int(targets.sum())
        if not 0 < target_count < len(indices):
            raise ValueError(f"zero-boundary calibration thieu HOLD/ENTRY cho role {name}")
        threshold = float(operating_points[name]["threshold"])
        with torch.no_grad():
            calibrator.role_calibrators[role].output.bias[1].sub_(threshold)
        shifted = margins - threshold
        audit[name] = {
            "train_samples": float(len(indices)),
            "train_label_entry_rate": float(target_count / len(indices)),
            "train_zero_boundary_entry_rate": float(np.mean(shifted >= 0.0)),
            "embedded_margin_shift": threshold,
            "train_operating_point_balanced_accuracy": float(
                operating_points[name]["balanced_accuracy"]
            ),
            "train_operating_point_entry_precision": float(
                operating_points[name]["entry_precision"]
            ),
            "train_operating_point_entry_recall": float(
                operating_points[name]["entry_recall"]
            ),
            "train_operating_point_mcc": float(operating_points[name]["mcc"]),
            "train_operating_point_eligible": float(
                operating_points[name]["eligible"]
            ),
        }
    calibrator.teacher_calibration_mode = "role_specific_train_operating_point_zero_boundary_v6"
    return audit


def _threshold_selection_score(
    validation: dict[str, float], audit: dict[str, dict[str, float]]
) -> float:
    """Select by the actual zero-boundary runtime, using sweeps as diagnostics."""

    roles = [audit[name] for name in ("buy", "sell")]
    role_mcc = [
        validation["buy_agent_mcc"], validation["sell_agent_mcc"]
    ]
    role_recall = [
        validation["buy_agent_entry_recall"], validation["sell_agent_entry_recall"]
    ]
    role_precision = [
        validation["buy_agent_entry_precision"], validation["sell_agent_entry_precision"]
    ]
    role_ba = [
        validation["buy_agent_balanced_accuracy"],
        validation["sell_agent_balanced_accuracy"],
    ]
    return float(
        1.50 * min(role_mcc)
        + 0.25 * sum(role_mcc)
        + 0.40 * min(role_recall)
        + 0.25 * min(role_precision)
        + 0.20 * min(role_ba)
        + 0.05 * min(item["mcc"] for item in roles)
        - 0.01 * validation["loss"]
    )


def _refine_role_calibrators(
    queries: Tensor | np.memmap,
    agent_ids: Tensor,
    actions: Tensor,
    quality: Tensor,
    masks: dict[str, Tensor],
    network: SharedFiveTradingPolicy,
    config: TeacherImitationConfig,
    *,
    role_entry_scales: dict[str, float] | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Freeze the backbone and fit each role boundary on train rows only.

    Validation chooses the epoch using the actual zero-boundary runtime metrics;
    no validation threshold is copied into the checkpoint. Test remains unseen
    until the final evaluation.
    """

    calibrator = network.heads.entry_role_calibrator
    parameters = list(calibrator.role_calibrators.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=max(1e-6, 0.25 * config.learning_rate), weight_decay=0.0
    )
    train_indices = torch.nonzero(masks["train"], as_tuple=False).reshape(-1)
    # The runtime decision boundary is fixed at logit margin zero. Inverse
    # prevalence weighting moves that boundary toward the rare ENTRY class and
    # made a model with 11% labels emit 83% entries. Keep cross-entropy
    # calibrated to the observed train distribution; the pairwise term below
    # supplies the rare-class ranking signal without changing class priors.
    role_scale = torch.ones(len(actions), dtype=torch.float32)
    if role_entry_scales is not None:
        role_scale[agent_ids == 0] = float(role_entry_scales.get("buy", 1.0))
        role_scale[agent_ids == 1] = float(role_entry_scales.get("sell", 1.0))
    for role in (0, 1):
        role_rows = masks["train"] & (agent_ids == role)
        positive = int((role_rows & (actions != 0)).sum().item())
        negative = int((role_rows & (actions == 0)).sum().item())
        if positive <= 0 or negative <= 0:
            raise ValueError(f"head refinement thieu class cho role {role}")
    initial_validation = imitation_metrics(
        queries, agent_ids, actions, network, masks["validation"]
    )
    initial_audit = _audit_role_operating_points(
        queries, agent_ids, actions, network, masks["validation"]
    )
    best_score = _threshold_selection_score(initial_validation, initial_audit)
    best_epoch = 0
    best_state = {
        name: value.detach().clone() for name, value in calibrator.state_dict().items()
    }
    stale = 0
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(config.seed + 91_337)
    epoch_limit = max(8, min(40, 2 * config.patience))
    for epoch in range(1, epoch_limit + 1):
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(permutation), 4096):
            index = permutation[start:start + 4096]
            logits = _role_logits(
                _hidden_rows(queries, index), agent_ids[index], network
            )
            targets = (actions[index] != 0).long()
            loss_rows = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            entry_probability = torch.softmax(logits, dim=-1)[:, 1]
            loss_rows = loss_rows + config.false_entry_penalty * (
                targets == 0
            ).to(dtype=loss_rows.dtype) * entry_probability
            weights = quality[index] * role_scale[index]
            loss = torch.sum(loss_rows * weights) / torch.sum(weights).clamp_min(1e-12)
            margins = logits[:, 1] - logits[:, 0]
            ranking_terms: list[Tensor] = []
            for role in (0, 1):
                role_rows = agent_ids[index] == role
                positive_margin = margins[role_rows & (targets == 1)]
                negative_margin = margins[role_rows & (targets == 0)]
                if len(positive_margin) and len(negative_margin):
                    ranking_terms.append(torch.nn.functional.softplus(
                        -(positive_margin[:, None] - negative_margin[None, :])
                    ).mean())
            if ranking_terms:
                loss = loss + config.ranking_loss_weight * torch.stack(ranking_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            loss_sum += float(torch.sum(loss_rows.detach() * weights).item())
            weight_sum += float(torch.sum(weights).item())
        zero_boundary_calibration = _calibrate_role_zero_boundary_from_train(
            queries, agent_ids, actions, network, masks["train"]
        )
        validation = imitation_metrics(
            queries, agent_ids, actions, network, masks["validation"]
        )
        operating_points = _audit_role_operating_points(
            queries, agent_ids, actions, network, masks["validation"]
        )
        score = _threshold_selection_score(validation, operating_points)
        record = {
            "event": "role_calibrator_refinement",
            "epoch": epoch,
            "train_weighted_loss": loss_sum / max(weight_sum, 1e-12),
            "validation_balanced_accuracy": validation["balanced_accuracy"],
            "validation_mcc": validation["mcc"],
            "validation_buy_agent_balanced_accuracy": validation["buy_agent_balanced_accuracy"],
            "validation_sell_agent_balanced_accuracy": validation["sell_agent_balanced_accuracy"],
            "validation_buy_agent_entry_precision": validation["buy_agent_entry_precision"],
            "validation_sell_agent_entry_precision": validation["sell_agent_entry_precision"],
            "validation_buy_agent_entry_recall": validation["buy_agent_entry_recall"],
            "validation_sell_agent_entry_recall": validation["sell_agent_entry_recall"],
            "validation_buy_recall": validation["buy_agent_entry_recall"],
            "validation_sell_recall": validation["sell_agent_entry_recall"],
            "validation_buy_predicted_entry_rate": validation["buy_agent_predicted_entry_rate"],
            "validation_sell_predicted_entry_rate": validation["sell_agent_predicted_entry_rate"],
            "train_zero_boundary_buy_entry_rate": zero_boundary_calibration["buy"]["train_zero_boundary_entry_rate"],
            "train_zero_boundary_sell_entry_rate": zero_boundary_calibration["sell"]["train_zero_boundary_entry_rate"],
        }
        history.append(record)
        if progress is not None:
            progress(record)
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in calibrator.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= min(config.patience, 8):
                break
    calibrator.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "selection_uses_validation_zero_boundary": True,
        "test_used_for_selection": False,
        "posthoc_threshold_applied": False,
        "zero_boundary_calibration_source": "train_operating_point_rate_capped_at_prevalence",
        "history": history,
    }


def _recurrent_windows_by_split(
    data: TeacherImitationData, sequence_length: int,
) -> dict[str, list[list[np.ndarray]]]:
    """Build chronological, split-bounded Five windows without future leakage."""

    groups: dict[str, list[np.ndarray]] = {
        "train": [], "validation": [], "test": [],
    }
    start = 0
    count = len(data.actions)
    while start < count:
        stop = start + 1
        while stop < count and data.sample_times[stop] == data.sample_times[start]:
            stop += 1
        indices = np.arange(start, stop, dtype=np.int64)
        split_values = set(data.splits[indices].tolist())
        if len(split_values) != 1:
            raise ValueError("mot sample_time khong duoc cat qua nhieu split")
        split = str(next(iter(split_values)))
        if split not in groups:
            raise ValueError(f"split recurrent khong hop le: {split}")
        if len(set(data.agent_ids[indices].tolist())) != len(indices):
            raise ValueError("trung agent_id tai cung sample_time")
        if not np.all(data.observations[indices] == data.observations[start]):
            raise ValueError("cac agent cung sample_time phai co observation giong nhau")
        groups[split].append(indices)
        start = stop
    return {
        split: [
            split_groups[offset:offset + sequence_length]
            for offset in range(0, len(split_groups), sequence_length)
        ]
        for split, split_groups in groups.items()
    }


def _forward_recurrent_window_batch(
    network: SharedFiveTradingPolicy,
    data: TeacherImitationData,
    windows: list[list[np.ndarray]],
) -> tuple[Tensor, Tensor]:
    """Run one differentiable teacher-forced BPTT batch and return owner queries."""

    if not windows:
        raise ValueError("BPTT batch rong")
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    window_count = len(windows)
    hidden = torch.zeros(
        (window_count, FIVE_AGENT_COUNT, network.hidden_size),
        dtype=dtype, device=device,
    )
    cell = torch.zeros_like(hidden)
    previous = torch.zeros(
        (window_count, FIVE_AGENT_COUNT), dtype=torch.long, device=device,
    )
    temporal_tokens, temporal_valid = network.post_lstm_transformer.empty_temporal_memory(
        window_count, device=device, dtype=hidden.dtype
    )
    agent_template = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
    query_parts: list[Tensor] = []
    index_parts: list[Tensor] = []
    for step in range(max(len(window) for window in windows)):
        active = [step < len(window) for window in windows]
        observations = np.stack([
            data.observations[window[step][0]] if is_active
            else np.zeros(data.observations.shape[1], dtype=np.float32)
            for window, is_active in zip(windows, active)
        ])
        obs = torch.as_tensor(observations, dtype=dtype, device=device)
        obs5 = obs[:, None, :].expand(
            window_count, FIVE_AGENT_COUNT, -1
        ).reshape(-1, obs.shape[-1])
        ids = agent_template.repeat(window_count)
        components = network.observation_encoder.encode_components(
            obs5, ids, previous.reshape(-1)
        )
        flattened = torch.cat((
            components["market"], components["account"], components["execution"],
            components["positions"], components["agent"],
            components["previous_action"],
        ), dim=-1)
        encoded = torch.tanh(
            network.observation_encoder.pre_lstm(flattened)
            + network.observation_encoder.pattern_residual(components["pattern_root"])
        ).reshape(window_count, FIVE_AGENT_COUNT, -1)
        width = max(1, int(round(
            encoded.shape[-1] * FIVE_CROSS_AGENT_POOL_FRACTION
        )))
        prefix = encoded[:, :, :width].max(dim=1, keepdim=True).values
        encoded = torch.cat((
            prefix.expand(-1, FIVE_AGENT_COUNT, -1), encoded[:, :, width:]
        ), dim=-1)
        next_hidden, next_cell = network.lstm(
            encoded.reshape(window_count * FIVE_AGENT_COUNT, -1),
            (
                hidden.reshape(window_count * FIVE_AGENT_COUNT, -1),
                cell.reshape(window_count * FIVE_AGENT_COUNT, -1),
            ),
        )
        next_hidden = next_hidden.reshape(window_count, FIVE_AGENT_COUNT, -1)
        next_cell = next_cell.reshape(window_count, FIVE_AGENT_COUNT, -1)
        policy_hidden = network.post_lstm_transformer(
            next_hidden, temporal_tokens, temporal_valid
        )
        next_temporal, next_temporal_valid = (
            network.post_lstm_transformer.update_temporal_memory(
                next_hidden, temporal_tokens, temporal_valid
            )
        )
        active_mask = torch.as_tensor(
            active, dtype=torch.bool, device=device
        ).reshape(window_count, 1, 1)
        hidden = torch.where(active_mask, next_hidden, hidden)
        cell = torch.where(active_mask, next_cell, cell)
        temporal_tokens = torch.where(
            active_mask.unsqueeze(-1), next_temporal, temporal_tokens
        )
        temporal_valid = torch.where(
            active_mask, next_temporal_valid, temporal_valid
        )
        next_previous = torch.zeros_like(previous)
        for window_index, is_active in enumerate(active):
            if not is_active:
                continue
            indices = windows[window_index][step]
            owners = torch.as_tensor(
                data.agent_ids[indices], dtype=torch.long, device=device
            )
            owner_hidden = policy_hidden[window_index, owners]
            query_parts.append(network.heads.action_query(owner_hidden))
            index_parts.append(torch.as_tensor(indices, dtype=torch.long, device=device))
            next_previous[window_index, owners] = torch.as_tensor(
                data.actions[indices], dtype=torch.long, device=device
            )
        previous = torch.where(active_mask.squeeze(-1), next_previous, previous)
    return torch.cat(query_parts, dim=0), torch.cat(index_parts, dim=0)


def _extract_recurrent_action_queries(
    network: SharedFiveTradingPolicy,
    data: TeacherImitationData,
    windows_by_split: dict[str, list[list[np.ndarray]]],
    *,
    batch_size: int,
    splits: Iterable[str] = ("train", "validation", "test"),
    progress: Any | None = None,
    progress_stage: str = "replay",
) -> tuple[Tensor, dict[str, Any]]:
    """Replay current parameters without detaching until each window completes."""

    queries = torch.zeros((len(data.actions), ACTION_EMBED_SIZE), dtype=torch.float32)
    selected_windows = [
        window for split in splits for window in windows_by_split[split]
    ]
    was_training = network.training
    network.eval()
    replay_started = time.monotonic()
    progress_interval = max(1, math.ceil(len(selected_windows) / 10))
    with torch.no_grad():
        for start in range(0, len(selected_windows), batch_size):
            batch = selected_windows[start:start + batch_size]
            batch_queries, batch_indices = _forward_recurrent_window_batch(
                network, data, batch
            )
            queries[batch_indices.cpu()] = batch_queries.detach().cpu()
            completed = min(start + len(batch), len(selected_windows))
            if progress is not None and (
                completed == len(selected_windows)
                or completed % progress_interval < len(batch)
            ):
                progress({
                    "event": "action_replay_progress",
                    "stage": progress_stage,
                    "windows_done": completed,
                    "windows_total": len(selected_windows),
                    "percent": 100.0 * completed / max(1, len(selected_windows)),
                    "elapsed_seconds": time.monotonic() - replay_started,
                })
    network.train(was_training)
    return queries, {
        "sequence_length": max((len(item) for item in selected_windows), default=0),
        "sequence_windows": len(selected_windows),
        "unique_time_steps": sum(len(item) for item in selected_windows),
        "state_reset_count": len(selected_windows),
        "representation": "live_action_query_post_lstm_spatiotemporal_transformer_bptt_v3",
        "representation_width": ACTION_EMBED_SIZE,
        "hidden_storage": "recomputed_ram_float32",
        "hidden_cache_reused": False,
    }


def _protected_logits_sha256(
    hidden: Tensor | np.memmap,
    agent_ids: Tensor,
    network: SharedFiveTradingPolicy,
) -> str:
    """Hash only logits that are outside the authorized teacher surface.

    The base action head is already covered byte-for-byte by
    ``network_invariant_sha256``.  The teacher adds the same float residual to
    the three owned styles. Re-hashing their final margins/order is both
    redundant and numerically wrong: adding a common float can perturb a
    subtraction or a near-tie by a few ULPs. Here we enforce the independent
    contract: every non-owned logit and every manager-agent logit is exact.
    """
    digest = hashlib.sha256()
    with torch.no_grad():
        for start in range(0, len(agent_ids), 4096):
            index = torch.arange(start, min(start + 4096, len(agent_ids)), dtype=torch.long)
            rows = _hidden_rows(hidden, index)
            owner_ids = agent_ids[index]
            owner_logits = _full_action_logits(rows, owner_ids, network)
            # The teacher may move HOLD versus the three owned entry styles,
            # but it must not modify any action outside that owner-specific
            # surface. Relative execution-style weights are protected by the
            # frozen base-head invariant plus the shared-residual unit test.
            for role, owned in ((0, (1, 3, 5)), (1, (2, 4, 6))):
                selected = owner_ids == role
                if not bool(torch.any(selected)):
                    continue
                role_logits = owner_logits[selected]
                protected = [item for item in range(role_logits.shape[-1]) if item not in {0, *owned}]
                digest.update(role_logits[:, protected].detach().cpu().contiguous().numpy().tobytes(order="C"))
            for role in range(2, FIVE_AGENT_COUNT):
                role_ids = torch.full((len(index),), role, dtype=torch.long)
                role_logits = _full_action_logits(rows, role_ids, network)
                digest.update(role_logits.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def train_recurrent_entry_calibrator(
    network: SharedFiveTradingPolicy,
    data: TeacherImitationData,
    config: TeacherImitationConfig,
    *,
    progress: Any | None = None,
    stop_requested: Any | None = None,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Any | None = None,
    mentor_config: ImitationMentorConfig | None = None,
    checkpoint_every_windows: int = 0,
) -> dict[str, Any]:
    config.validate()
    if checkpoint_every_windows < 0:
        raise ValueError("checkpoint_every_windows phai >= 0")
    mentor_config = mentor_config or ImitationMentorConfig(enabled=False)
    mentor_config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    network.cpu()
    network.require_current_architecture()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    calibrator = network.heads.entry_role_calibrator
    if not isinstance(calibrator, EntryRoleCalibrator):
        raise RuntimeError("checkpoint khong co entry-role calibrator hien tai")
    resuming = resume_state is not None
    if resuming and int(resume_state.get("schema_version", 0)) != 2:
        raise ValueError(
            "resume imitation v2 cu khong tuong thich temporal/role calibrator v3; "
            "hay chay output moi"
        )
    # Reset only for a fresh run. A resumed system already contains the exact
    # recurrent and adapter weights from the last completed optimizer batch.
    if not resuming:
        calibrator.reset_teacher_mapping()
    # Promote the action surface before measuring/training. Older checkpoints
    # remain MARKET-only during ordinary inference and are changed only by an
    # explicit imitation run using this v2 objective.
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    recurrent_named = [
        (name, parameter) for name, parameter in network.named_parameters()
        if name.startswith((
            "observation_encoder.",
            "lstm.",
            "post_lstm_transformer.",
            "heads.action_query.",
        ))
    ]
    adapter_named = [
        (name, parameter) for name, parameter in network.named_parameters()
        if name.startswith((
            "heads.entry_role_calibrator.role_calibrators.",
        ))
    ]
    trainable_named = recurrent_named + adapter_named
    if not recurrent_named or not adapter_named:
        raise RuntimeError("thieu recurrent/calibrator parameter cho imitation")
    for _, parameter in trainable_named:
        parameter.requires_grad_(True)
    trainable = [parameter for _, parameter in trainable_named]
    windows_by_split = _recurrent_windows_by_split(data, config.sequence_length)
    agent_ids = torch.as_tensor(data.agent_ids, dtype=torch.long)
    actions = torch.as_tensor(data.actions, dtype=torch.long)
    candidate_actions = torch.as_tensor(data.candidate_actions, dtype=torch.long)
    teacher_trade_accepted = torch.as_tensor(data.teacher_trade_accepted, dtype=torch.bool)
    teacher_net_r = torch.as_tensor(data.teacher_net_r, dtype=torch.float32)
    quality = torch.as_tensor(data.quality_weights, dtype=torch.float32)
    masks = {name: torch.as_tensor(data.splits == name) for name in ("train", "validation", "test")}
    train_windows = windows_by_split["train"]
    # Preserve probability calibration at the immutable zero boundary. The
    # older inverse-prevalence weights made rare teacher entries behave like a
    # 50/50 prior and caused severe over-entry at inference.
    class_scale = torch.ones(len(data.actions), dtype=torch.float32)
    role_entry_scales: dict[str, float] = {}
    for agent, name in ((0, "buy"), (1, "sell")):
        role = masks["train"] & (agent_ids == agent)
        positives = role & (actions != 0)
        negatives = role & (actions == 0)
        positive_count = int(positives.sum().item())
        negative_count = int(negatives.sum().item())
        if positive_count == 0 or negative_count == 0:
            raise ValueError(f"train split thieu HOLD/ENTRY cua role {name}")
        role_entry_scales[name] = 1.0
    optimizer = torch.optim.Adam([
        {
            "params": [parameter for _, parameter in recurrent_named],
            "lr": config.recurrent_learning_rate,
        },
        {
            "params": [parameter for _, parameter in adapter_named],
            "lr": config.learning_rate,
        },
    ], weight_decay=0.0)
    mentor_state = _initial_imitation_mentor_state(config, mentor_config)
    generator = torch.Generator().manual_seed(config.seed)
    if resuming:
        assert resume_state is not None
        if int(resume_state.get("schema_version", 0)) != 2:
            raise ValueError("resume imitation khong dung schema 2")
        if dict(resume_state.get("config", {})) != config.__dict__:
            raise ValueError("resume imitation khong khop hyperparameter")
        frozen_before = str(resume_state["frozen_before"])
        trainable_before = str(resume_state["trainable_before"])
        if recurrent_imitation_frozen_sha256(network) != frozen_before:
            raise ValueError("resume imitation lam sai frozen-head hash")
        if (
            recurrent_imitation_trainable_sha256(network)
            != str(resume_state["current_trainable_sha256"])
        ):
            raise ValueError("resume imitation khong khop recurrent weights da luu")
        initial_trainable_state = dict(resume_state["initial_trainable_state"])
        sequence_audit = dict(resume_state["initial_sequence_audit"])
        before = dict(resume_state["before"])
        before_thresholds = dict(resume_state["before_thresholds"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        saved_mentor_config = resume_state.get("mentor_config")
        if mentor_config.enabled:
            if dict(saved_mentor_config or {}) != dict(mentor_config.__dict__):
                raise ValueError("resume imitation khong khop mentor config")
            if not isinstance(resume_state.get("mentor_state"), dict):
                raise ValueError("resume imitation thieu mentor state")
            mentor_state = dict(resume_state["mentor_state"])
            mentor_state["role_scales"] = {
                "buy": float(dict(mentor_state["role_scales"])["buy"]),
                "sell": float(dict(mentor_state["role_scales"])["sell"]),
            }
        elif saved_mentor_config not in (None, {}, {"enabled": False}):
            raise ValueError("resume imitation yeu cau bat lai mentor config cu")
        generator.set_state(resume_state["generator_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        np.random.set_state(resume_state["numpy_rng_state"])
        best_state = dict(resume_state["best_state"])
        best_score = float(resume_state["best_score"])
        best_epoch = int(resume_state["best_epoch"])
        stale = int(resume_state["stale"])
        gradient_seen = set(resume_state["gradient_seen"])
        gradient_norm_max = {
            str(name): float(value)
            for name, value in dict(resume_state["gradient_norm_max"]).items()
        }
        history = list(resume_state["history"])
        resume_epoch = int(resume_state["epoch"])
        resume_permutation = [int(value) for value in resume_state["permutation"]]
        resume_batch_start = int(resume_state["next_batch_start"])
        resume_epoch_loss = float(resume_state["epoch_loss"])
        resume_epoch_weight = float(resume_state["epoch_weight"])
        resume_epoch_diagnostics = {
            str(name): float(value)
            for name, value in dict(
                resume_state.get("epoch_diagnostics", _empty_epoch_diagnostics())
            ).items()
        }
        # A checkpoint written after a completed validation epoch deliberately
        # stores an empty permutation and cursor 0: the next epoch must draw a
        # fresh permutation from the restored RNG state.  Only an in-epoch
        # checkpoint carries a permutation that must match the current window
        # count.  The previous unconditional length check rejected every
        # restartable epoch-boundary checkpoint before training could resume.
        if resume_permutation and len(resume_permutation) != len(train_windows):
            raise ValueError("resume imitation khong khop so recurrent window")
        if not resume_permutation and resume_batch_start != 0:
            raise ValueError("resume imitation boundary checkpoint co cursor khong hop le")
        if not 0 <= resume_batch_start <= len(resume_permutation):
            raise ValueError("resume imitation co batch cursor khong hop le")
        if progress is not None:
            progress({
                "event": "action_resume_loaded",
                "epoch": resume_epoch,
                "windows_done": resume_batch_start,
                "windows_total": len(resume_permutation),
            })
            for saved_record in history:
                progress({"event": "action_epoch_restored", **saved_record})
    else:
        frozen_before = recurrent_imitation_frozen_sha256(network)
        trainable_before = recurrent_imitation_trainable_sha256(network)
        initial_trainable_state = {
            name: parameter.detach().clone() for name, parameter in trainable_named
        }
        hidden, sequence_audit = _extract_recurrent_action_queries(
            network, data, windows_by_split,
            batch_size=config.feature_batch_size,
            progress=progress,
            progress_stage="initial_metric_replay",
        )
        before = {
            name: imitation_metrics(hidden, agent_ids, actions, network, mask)
            for name, mask in masks.items()
        }
        before_thresholds = _audit_role_operating_points(
            hidden, agent_ids, actions, network, masks["validation"]
        )
        initial_zero_boundary_calibration = _calibrate_role_zero_boundary_from_train(
            hidden, agent_ids, actions, network, masks["train"]
        )
        calibrated_initial_validation = imitation_metrics(
            hidden, agent_ids, actions, network, masks["validation"]
        )
        calibrated_initial_thresholds = _audit_role_operating_points(
            hidden, agent_ids, actions, network, masks["validation"]
        )
        del hidden
        best_state = {
            name: parameter.detach().clone() for name, parameter in trainable_named
        }
        best_score = _threshold_selection_score(
            calibrated_initial_validation, calibrated_initial_thresholds
        )
        best_epoch = 0
        stale = 0
        gradient_seen: set[str] = set()
        gradient_norm_max: dict[str, float] = {
            "observation_encoder": 0.0,
            "lstm": 0.0,
            "post_lstm_transformer": 0.0,
            "action_query": 0.0,
            "entry_calibrator": 0.0,
        }
        history: list[dict[str, Any]] = []
        resume_epoch = 1
        resume_permutation = []
        resume_batch_start = 0
        resume_epoch_loss = 0.0
        resume_epoch_weight = 0.0
        resume_epoch_diagnostics = _empty_epoch_diagnostics()

    base_learning_rates = [config.recurrent_learning_rate, config.learning_rate]
    if resuming and mentor_config.enabled:
        saved_multiplier = float(mentor_state["learning_rate_multiplier"])
        for group, base_lr in zip(optimizer.param_groups, base_learning_rates):
            # The optimizer state is authoritative on resume; this assignment
            # only makes the saved mentor state visible if an older torch
            # serializer omitted a group's lr field.
            group["lr"] = float(group.get("lr", base_lr * saved_multiplier))

    def apply_mentor_learning_rate() -> None:
        multiplier = float(mentor_state["learning_rate_multiplier"])
        for group, base_lr in zip(optimizer.param_groups, base_learning_rates):
            group["lr"] = float(base_lr * multiplier)

    def pause_state(
        *, epoch: int, permutation: list[int], next_batch_start: int,
        epoch_loss: float, epoch_weight: float, epoch_diagnostics: dict[str, float],
        checkpoint_kind: str = "pause",
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "config": dict(config.__dict__),
            "epoch": int(epoch),
            "permutation": list(permutation),
            "next_batch_start": int(next_batch_start),
            "checkpoint_kind": str(checkpoint_kind),
            "epoch_loss": float(epoch_loss),
            "epoch_weight": float(epoch_weight),
            "epoch_diagnostics": dict(epoch_diagnostics),
            "optimizer_state_dict": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "best_state": best_state,
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "stale": int(stale),
            "gradient_seen": sorted(gradient_seen),
            "gradient_norm_max": dict(gradient_norm_max),
            "history": list(history),
            "initial_trainable_state": initial_trainable_state,
            "initial_sequence_audit": dict(sequence_audit),
            "before": before,
            "before_thresholds": before_thresholds,
            "frozen_before": frozen_before,
            "trainable_before": trainable_before,
            "current_trainable_sha256": recurrent_imitation_trainable_sha256(network),
            "mentor_config": dict(mentor_config.__dict__) if mentor_config.enabled else None,
            "mentor_state": dict(mentor_state),
        }

    for epoch in range(resume_epoch, config.epochs + 1):
        if epoch == resume_epoch and resume_permutation:
            permutation = resume_permutation
            batch_start = resume_batch_start
            epoch_loss = resume_epoch_loss
            epoch_weight = resume_epoch_weight
            epoch_diagnostics = dict(resume_epoch_diagnostics)
        else:
            permutation = torch.randperm(len(train_windows), generator=generator).tolist()
            batch_start = 0
            epoch_loss = 0.0
            epoch_weight = 0.0
            epoch_diagnostics = _empty_epoch_diagnostics()
        epoch_started = time.monotonic()
        active_false_entry_penalty = float(mentor_state["active_false_entry_penalty"])
        mentor_role_scales = {
            "buy": float(dict(mentor_state["role_scales"])["buy"]),
            "sell": float(dict(mentor_state["role_scales"])["sell"]),
        }
        mentor_row_scales = torch.ones(len(data.actions), dtype=torch.float32)
        mentor_row_scales[agent_ids == 0] = mentor_role_scales["buy"]
        mentor_row_scales[agent_ids == 1] = mentor_role_scales["sell"]
        apply_mentor_learning_rate()
        batch_progress_interval = max(1, math.ceil(len(permutation) / 20))
        next_batch_progress = max(
            batch_progress_interval,
            math.ceil(max(1, batch_start) / batch_progress_interval) * batch_progress_interval,
        )
        checkpoint_interval = max(
            config.batch_size,
            int(checkpoint_every_windows),
        ) if checkpoint_every_windows > 0 else 0
        next_batch_checkpoint = (
            ((max(0, batch_start) // checkpoint_interval) + 1) * checkpoint_interval
            if checkpoint_interval > 0 else 0
        )
        network.train()
        for start in range(batch_start, len(permutation), config.batch_size):
            batch_windows = [
                train_windows[index]
                for index in permutation[start:start + config.batch_size]
            ]
            queries, index = _forward_recurrent_window_batch(
                network, data, batch_windows
            )
            index = index.cpu()
            logits = _role_logits(queries, agent_ids[index], network)
            targets = (actions[index] != 0).long()
            loss_rows = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            entry_probability = torch.softmax(logits, dim=-1)[:, 1]
            loss_rows = loss_rows + active_false_entry_penalty * (
                targets == 0
            ).to(dtype=loss_rows.dtype) * entry_probability
            hard_negative = (
                (candidate_actions[index] != int(Action.HOLD))
                & (targets == 0)
                & (~teacher_trade_accepted[index])
            )
            hard_negative_weight = torch.where(
                hard_negative,
                torch.full_like(
                    entry_probability,
                    float(mentor_state["hard_negative_multiplier"]),
                ),
                torch.ones_like(entry_probability),
            )
            row_weight = (
                quality[index]
                * class_scale[index]
                * mentor_row_scales[index]
                * hard_negative_weight
            )
            epoch_diagnostics["teacher_hard_negative_count"] += float(
                hard_negative.sum().item()
            )
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                batch_counts = _classification_statistics(targets, predictions)
                for name, value in batch_counts.items():
                    epoch_diagnostics[name] += value
                entry_rows = targets == 1
                hold_rows = targets == 0
                epoch_diagnostics["entry_probability_sum"] += float(
                    entry_probability[entry_rows].sum().item()
                )
                epoch_diagnostics["entry_probability_count"] += float(entry_rows.sum().item())
                epoch_diagnostics["hold_entry_probability_sum"] += float(
                    entry_probability[hold_rows].sum().item()
                )
                epoch_diagnostics["hold_entry_probability_count"] += float(hold_rows.sum().item())
            loss = torch.sum(loss_rows * row_weight) / torch.sum(row_weight).clamp_min(1e-12)
            margins = logits[:, 1] - logits[:, 0]
            teacher_r = teacher_net_r[index]
            teacher_positive_strength = torch.log1p(
                torch.clamp(teacher_r, min=0.0, max=230.0)
            )
            teacher_negative_strength = torch.log1p(
                torch.clamp(-teacher_r, min=0.0, max=1.0)
            )
            teacher_margin_target = torch.where(
                targets == 1,
                0.35 + 0.12 * teacher_positive_strength,
                -0.35 - 0.12 * teacher_negative_strength,
            )
            teacher_confidence_loss = torch.nn.functional.smooth_l1_loss(
                margins, teacher_margin_target, reduction="none", beta=0.50,
            )
            confidence_weight = float(mentor_state["teacher_confidence_weight"])
            if confidence_weight > 0.0:
                confidence_term = torch.sum(
                    teacher_confidence_loss * row_weight
                ) / torch.sum(row_weight).clamp_min(1e-12)
                loss = loss + confidence_weight * confidence_term
                epoch_diagnostics["teacher_confidence_loss_sum"] += float(
                    torch.sum(teacher_confidence_loss * row_weight).item()
                )
                epoch_diagnostics["teacher_confidence_weight_sum"] += float(
                    torch.sum(row_weight).item()
                )
            ranking_terms: list[Tensor] = []
            for role in (0, 1):
                role_rows = agent_ids[index] == role
                positive = margins[role_rows & (targets == 1)]
                negative = margins[role_rows & (targets == 0)]
                if len(positive) and len(negative):
                    ranking_terms.append(torch.nn.functional.softplus(
                        -(positive[:, None] - negative[None, :])
                    ).mean())
            if ranking_terms:
                loss = loss + float(mentor_state["active_ranking_loss_weight"]) * torch.stack(ranking_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in trainable_named:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(f"gradient imitation NaN/Inf: {name}")
                if bool(torch.any(gradient != 0)):
                    gradient_seen.add(name)
                norm = float(torch.linalg.vector_norm(gradient.detach()).item())
                if name.startswith("observation_encoder."):
                    group = "observation_encoder"
                elif name.startswith("lstm."):
                    group = "lstm"
                elif name.startswith("post_lstm_transformer."):
                    group = "post_lstm_transformer"
                elif name.startswith("heads.action_query."):
                    group = "action_query"
                else:
                    group = "entry_calibrator"
                gradient_norm_max[group] = max(gradient_norm_max[group], norm)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * float(torch.sum(row_weight).item())
            epoch_weight += float(torch.sum(row_weight).item())
            windows_done = min(start + len(batch_windows), len(permutation))
            if progress is not None and (
                windows_done >= next_batch_progress
                or windows_done == len(permutation)
            ):
                progress({
                    "event": "action_batch_progress",
                    "epoch": epoch,
                    "windows_done": windows_done,
                    "windows_total": len(permutation),
                    "percent": 100.0 * windows_done / max(1, len(permutation)),
                    "train_weighted_loss_so_far": (
                        epoch_loss / max(epoch_weight, 1e-12)
                    ),
                    "elapsed_seconds": time.monotonic() - epoch_started,
                })
                while next_batch_progress <= windows_done:
                    next_batch_progress += batch_progress_interval
            if (
                checkpoint is not None
                and checkpoint_interval > 0
                and windows_done >= next_batch_checkpoint
                and windows_done < len(permutation)
            ):
                checkpoint(pause_state(
                    epoch=epoch,
                    permutation=permutation,
                    next_batch_start=windows_done,
                    epoch_loss=epoch_loss,
                    epoch_weight=epoch_weight,
                    epoch_diagnostics=epoch_diagnostics,
                    checkpoint_kind="batch",
                ))
                while next_batch_checkpoint <= windows_done:
                    next_batch_checkpoint += checkpoint_interval
            if stop_requested is not None and bool(stop_requested()):
                if progress is not None:
                    progress({
                        "event": "action_pause_requested",
                        "epoch": epoch,
                        "windows_done": windows_done,
                        "windows_total": len(permutation),
                    })
                raise TeacherImitationPaused(pause_state(
                    epoch=epoch,
                    permutation=permutation,
                    next_batch_start=windows_done,
                    epoch_loss=epoch_loss,
                    epoch_weight=epoch_weight,
                    epoch_diagnostics=epoch_diagnostics,
                    checkpoint_kind="pause",
                ))
        validation_queries, _ = _extract_recurrent_action_queries(
            network, data, windows_by_split,
            batch_size=config.feature_batch_size,
            splits=("train", "validation"),
            progress=progress,
            progress_stage=f"epoch_{epoch}_validation_replay",
        )
        zero_boundary_calibration = _calibrate_role_zero_boundary_from_train(
            validation_queries, agent_ids, actions, network, masks["train"]
        )
        validation = imitation_metrics(
            validation_queries, agent_ids, actions, network, masks["validation"]
        )
        validation_thresholds = _audit_role_operating_points(
            validation_queries, agent_ids, actions, network,
            masks["validation"]
        )
        train_metrics_online = _metrics_from_counts(epoch_diagnostics)
        # Compare like with like: both sides of the generalisation gap are a
        # replay with the same final weights from this epoch.  The in-batch
        # counters remain useful telemetry, but they span changing weights and
        # shuffled windows and therefore cannot form a meaningful BA gap.
        train_metrics_replay = imitation_metrics(
            validation_queries, agent_ids, actions, network, masks["train"]
        )
        reward_signal = (
            epoch_diagnostics["entry_probability_sum"]
            / max(1.0, epoch_diagnostics["entry_probability_count"])
        )
        penalty_signal = active_false_entry_penalty * (
            epoch_diagnostics["hold_entry_probability_sum"]
            / max(1.0, epoch_diagnostics["hold_entry_probability_count"])
        )
        reward_penalty_defined = float(penalty_signal > 1e-12)
        reward_penalty_ratio = (
            reward_signal / penalty_signal if reward_penalty_defined else 0.0
        )
        train_loss = epoch_loss / max(epoch_weight, 1e-12)
        teacher_confidence_loss = (
            epoch_diagnostics["teacher_confidence_loss_sum"]
            / max(epoch_diagnostics["teacher_confidence_weight_sum"], 1e-12)
        )
        fit = _fit_diagnostic(
            epoch=epoch,
            train_metrics=train_metrics_replay,
            validation=validation,
            train_loss=train_loss,
            history=history,
        )
        record = {
            "epoch": float(epoch),
            "train_weighted_loss": train_loss,
            "train_balanced_accuracy_online": train_metrics_online["balanced_accuracy"],
            "train_entry_precision_online": train_metrics_online["entry_precision"],
            "train_entry_recall_online": train_metrics_online["entry_recall"],
            "train_mcc_online": train_metrics_online["mcc"],
            "train_predicted_entry_rate_online": train_metrics_online["predicted_entry_rate"],
            "train_balanced_accuracy_replay": train_metrics_replay["balanced_accuracy"],
            "train_entry_precision_replay": train_metrics_replay["entry_precision"],
            "train_entry_recall_replay": train_metrics_replay["entry_recall"],
            "train_mcc_replay": train_metrics_replay["mcc"],
            "train_predicted_entry_rate_replay": train_metrics_replay["predicted_entry_rate"],
            "validation_loss": validation["loss"],
            "validation_balanced_accuracy": validation["balanced_accuracy"],
            "validation_entry_recall": validation["entry_recall"],
            "validation_entry_precision": validation["entry_precision"],
            "validation_label_entry_rate": validation["label_entry_rate"],
            "validation_predicted_entry_rate": validation["predicted_entry_rate"],
            "validation_mcc": validation["mcc"],
            "validation_buy_agent_balanced_accuracy": validation["buy_agent_balanced_accuracy"],
            "validation_sell_agent_balanced_accuracy": validation["sell_agent_balanced_accuracy"],
            "validation_buy_agent_entry_precision": validation["buy_agent_entry_precision"],
            "validation_sell_agent_entry_precision": validation["sell_agent_entry_precision"],
            "validation_buy_agent_entry_recall": validation["buy_agent_entry_recall"],
            "validation_sell_agent_entry_recall": validation["sell_agent_entry_recall"],
            "imitation_reward_signal": reward_signal,
            "imitation_penalty_signal": penalty_signal,
            "imitation_false_entry_penalty_active": active_false_entry_penalty,
            "imitation_reward_penalty_ratio": reward_penalty_ratio,
            "imitation_reward_penalty_ratio_defined": reward_penalty_defined,
            "teacher_confidence_loss": teacher_confidence_loss,
            "teacher_hard_negative_count": epoch_diagnostics["teacher_hard_negative_count"],
            "teacher_confidence_weight_active": float(mentor_state["teacher_confidence_weight"]),
            "teacher_hard_negative_multiplier_active": float(mentor_state["hard_negative_multiplier"]),
            "teacher_ranking_loss_weight_active": float(mentor_state["active_ranking_loss_weight"]),
            **fit,
            "validation_threshold_buy_precision": validation_thresholds["buy"]["entry_precision"],
            "validation_threshold_sell_precision": validation_thresholds["sell"]["entry_precision"],
            "validation_threshold_buy_mcc": validation_thresholds["buy"]["mcc"],
            "validation_threshold_sell_mcc": validation_thresholds["sell"]["mcc"],
            "validation_threshold_buy_eligible": validation_thresholds["buy"]["eligible"],
            "validation_threshold_sell_eligible": validation_thresholds["sell"]["eligible"],
            "train_zero_boundary_buy_entry_rate": zero_boundary_calibration["buy"]["train_zero_boundary_entry_rate"],
            "train_zero_boundary_sell_entry_rate": zero_boundary_calibration["sell"]["train_zero_boundary_entry_rate"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        mentor_state, mentor_decision = _imitation_mentor_step(
            record, history, mentor_state, mentor_config,
        )
        record.update({
            "mentor_enabled": bool(mentor_config.enabled),
            "mentor_action": mentor_decision["action"],
            "mentor_stage": mentor_decision.get("stage", mentor_state["stage"]),
            "mentor_signals": mentor_decision.get("signals", {}),
            "mentor_next_false_entry_penalty": float(mentor_state["active_false_entry_penalty"]),
            "mentor_next_buy_scale": float(dict(mentor_state["role_scales"])["buy"]),
            "mentor_next_sell_scale": float(dict(mentor_state["role_scales"])["sell"]),
            "mentor_next_learning_rate_multiplier": float(mentor_state["learning_rate_multiplier"]),
            "mentor_next_teacher_confidence_weight": float(
                mentor_state["teacher_confidence_weight"]
            ),
            "mentor_next_hard_negative_multiplier": float(
                mentor_state["hard_negative_multiplier"]
            ),
            "mentor_next_ranking_loss_weight": float(
                mentor_state["active_ranking_loss_weight"]
            ),
        })
        history.append(record)
        if progress is not None:
            progress(record)
        score = _threshold_selection_score(validation, validation_thresholds)
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in trainable_named
            }
            stale = 0
        else:
            stale += 1
        stop_after_epoch = stale >= config.patience
        if checkpoint is not None:
            # Persist a restart point after every completed validation epoch.
            # The next resume starts at a fresh epoch, so it never replays a
            # partially updated batch or silently loses a multi-hour epoch if
            # the owning terminal/UI process disappears.
            checkpoint(pause_state(
                epoch=epoch + 1,
                permutation=[],
                next_batch_start=0,
                epoch_loss=0.0,
                epoch_weight=0.0,
                epoch_diagnostics=_empty_epoch_diagnostics(),
                checkpoint_kind="epoch",
            ))
        if stop_after_epoch:
            break
    with torch.no_grad():
        for name, parameter in trainable_named:
            parameter.copy_(best_state[name])
    hidden, final_sequence_audit = _extract_recurrent_action_queries(
        network, data, windows_by_split,
        batch_size=config.feature_batch_size,
        progress=progress,
        progress_stage="final_metric_replay",
    )
    refinement_config = TeacherImitationConfig(
        **{
            **config.__dict__,
            "false_entry_penalty": float(mentor_state["active_false_entry_penalty"]),
        }
    )
    role_calibrator_refinement = _refine_role_calibrators(
        hidden, agent_ids, actions, quality, masks, network, refinement_config,
        role_entry_scales={
            "buy": float(dict(mentor_state["role_scales"])["buy"]),
            "sell": float(dict(mentor_state["role_scales"])["sell"]),
        },
        progress=progress,
    )
    threshold_calibration = _audit_role_operating_points(
        hidden, agent_ids, actions, network, masks["validation"]
    )
    after = {name: imitation_metrics(hidden, agent_ids, actions, network, mask) for name, mask in masks.items()}
    validation_role_samples = min(
        float((masks["validation"] & (agent_ids == 0)).sum().item()),
        float((masks["validation"] & (agent_ids == 1)).sum().item()),
    )
    if validation_role_samples >= 50.0:
        autonomous_validation = autonomous_entry_validation_metrics(
            network, data, split="validation", progress=progress,
        )
        autonomous_validation["teacher_forced_balanced_accuracy_gap"] = (
            after["validation"]["balanced_accuracy"]
            - float(autonomous_validation["balanced_accuracy"])
        )
        autonomous_gate = autonomous_entry_validation_gate(autonomous_validation)
    else:
        autonomous_validation = {
            "mode": "continuous_self_fed_deterministic_entry_v1",
            "split": "validation",
            "skipped": True,
            "skipped_reason": "insufficient_role_samples",
        }
        autonomous_gate = {
            "enforced": False,
            "minimum_samples_per_role": 50,
            "observed_minimum_role_samples": validation_role_samples,
            "checks": {},
            "failed_checks": [],
            "passed": True,
        }
    frozen_after = recurrent_imitation_frozen_sha256(network)
    if frozen_after != frozen_before:
        raise RuntimeError("head weight ngoai recurrent imitation surface da thay doi")
    trainable_after = recurrent_imitation_trainable_sha256(network)
    if trainable_after == trainable_before:
        raise RuntimeError("optimizer khong cap nhat recurrent representation")
    missing_gradient_groups = [
        name for name, value in gradient_norm_max.items() if value <= 0.0
    ]
    if missing_gradient_groups:
        raise RuntimeError(
            "imitation gradient chua di qua du recurrent surface: "
            + ",".join(missing_gradient_groups)
        )
    validation_gain = (
        after["validation"]["balanced_accuracy"]
        - before["validation"]["balanced_accuracy"]
    )
    validation_role_floor = min(
        after["validation"]["buy_agent_balanced_accuracy"],
        after["validation"]["sell_agent_balanced_accuracy"],
    )
    validation_role_recall_floor = min(
        after["validation"]["buy_agent_entry_recall"],
        after["validation"]["sell_agent_entry_recall"],
    )
    validation_metrics = after["validation"]
    validation_gate_enforced = validation_role_samples >= 50.0
    precision_floor = max(0.30, 1.20 * validation_metrics["label_entry_rate"])
    rate_ceiling = min(0.50, max(0.03, 1.20 * validation_metrics["label_entry_rate"]))
    if validation_gate_enforced and (
        validation_metrics["balanced_accuracy"] < 0.55
        or validation_gain < 0.02
        or validation_role_floor < 0.52
        or validation_role_recall_floor < 0.20
        or validation_metrics["mcc"] < 0.08
        or validation_metrics["entry_precision"] < precision_floor
        or validation_metrics["predicted_entry_rate"] > rate_ceiling
        or not all(bool(item["eligible"]) for item in threshold_calibration.values())
    ):
        raise RuntimeError(
            "validation gate failed: candidate phai tang balanced accuracy, MCC/precision duong, "
            "khong over-entry, recall moi role >= 0.20 va ca BUY/SELL deu vuot random; "
            + json.dumps({
                "before": before["validation"],
                "after": validation_metrics,
                "threshold_calibration": threshold_calibration,
            }, ensure_ascii=False, sort_keys=True)
        )
    if autonomous_gate["enforced"] and not autonomous_gate["passed"]:
        raise RuntimeError(
            "autonomous validation gate failed: candidate phai giu duoc chat luong vao lenh "
            "khi tu dua hanh dong truoc vao tri nho va giu trang thai lien tuc; "
            + json.dumps({
                "metrics": autonomous_validation,
                "gate": autonomous_gate,
            }, ensure_ascii=False, sort_keys=True)
        )
    if not all(math.isfinite(value) for metrics in after.values() for value in metrics.values()):
        raise RuntimeError("metric imitation NaN/Inf")
    calibrator.teacher_imitation_trained = True
    calibrator.teacher_imitation_schema_version = 7
    calibrator.teacher_imitation_projection = (
        "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3"
    )
    changed_parameters = sorted([
        name for name, parameter in trainable_named
        if not torch.equal(parameter.detach(), initial_trainable_state[name])
    ])
    return {
        "config": config.__dict__,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "before": before,
        "after": after,
        "history": history,
        "sequence_audit": {
            **final_sequence_audit,
            "train_sequence_windows": len(train_windows),
            "bptt_window_batch_size": config.batch_size,
            "evaluation_window_batch_size": config.feature_batch_size,
            "initial_replay": sequence_audit,
        },
        "role_entry_scales": role_entry_scales,
        "role_calibrator_refinement": role_calibrator_refinement,
        "projection_contract": {
            "representation": "full_recurrent_bptt_spatiotemporal_transformer_role_calibrator_v3",
            "width": ACTION_EMBED_SIZE,
            "gradient_trainable": [
                "observation_encoder.*", "lstm.*", "post_lstm_transformer.*",
                "heads.action_query.*",
                "heads.entry_role_calibrator.role_calibrators.0.*",
                "heads.entry_role_calibrator.role_calibrators.1.*",
            ],
            "gradient_norm_max_by_group": gradient_norm_max,
            "gradient_nonzero_parameter_count": len(gradient_seen),
            "gradient_nonzero_parameters": sorted(gradient_seen),
            "shared_downstream_logits_expected_to_change": True,
            "validation_calibrated": [],
            "decision_boundary": "learned_role_specific_zero_boundary_no_posthoc_threshold",
        },
        "threshold_calibration": threshold_calibration,
        "autonomous_validation": autonomous_validation,
        "autonomous_validation_gate": autonomous_gate,
        "validation_gate": {
            "enforced": validation_gate_enforced,
            "minimum_samples_per_role": 50,
            "observed_minimum_role_samples": validation_role_samples,
            "passed": True,
        },
        "recurrent_imitation_frozen_sha256": frozen_after,
        "recurrent_imitation_trainable_before_sha256": trainable_before,
        "recurrent_imitation_trainable_after_sha256": trainable_after,
        "changed_parameters": changed_parameters,
        "mentor": {
            "enabled": bool(mentor_config.enabled),
            "config": dict(mentor_config.__dict__),
            "final_state": dict(mentor_state),
            "decision_count": int(mentor_state["decision_count"]),
        },
        "teacher_guidance": {
            "lessons": [
                "causal_hold_entry_timing_bptt",
                "rejected_candidate_hard_negatives",
                "teacher_net_r_confidence_margin",
                "buy_sell_role_balance_and_ranking",
                "continuous_self_fed_state_and_overfit_control",
            ],
            "future_fields_used_as_inputs": False,
            "test_split_used_for_guidance": False,
            "geometry_stage": "separate_teacher_owned_heads_when_enabled",
        },
    }


def geometry_imitation_metrics(
    hidden: Tensor | np.memmap,
    targets: TeacherGeometryTargets,
    network: SharedFiveTradingPolicy,
    mask: Tensor,
) -> dict[str, float]:
    valid = mask & torch.as_tensor(targets.valid, dtype=torch.bool)
    indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        raise ValueError("split khong co accepted geometry target")
    target_map = {
        "delay": torch.as_tensor(targets.delay_indices, dtype=torch.long),
        "sl": torch.as_tensor(targets.sl_indices, dtype=torch.long),
        "tp": torch.as_tensor(targets.tp_indices, dtype=torch.long),
        "trailing": torch.as_tensor(targets.trailing_indices, dtype=torch.long),
    }
    continuous_target_map = {
        "sl": torch.as_tensor(targets.sl_atr, dtype=torch.float32),
        "tp": torch.as_tensor(targets.tp_rr, dtype=torch.float32),
    }
    distance_bins = {
        "sl": torch.as_tensor(SL_ATR_BINS, dtype=torch.float32),
        "tp": torch.as_tensor(TP_RR_BINS, dtype=torch.float32),
    }
    correct: dict[str, int] = {name: 0 for name in GEOMETRY_TEACHER_HEADS}
    within_one_bin = {name: 0 for name in distance_bins}
    absolute_error_sum = {name: 0.0 for name in distance_bins}
    log_ratio_error_sum = {name: 0.0 for name in distance_bins}
    expected_sample_log_ratio_error_sum = {name: 0.0 for name in distance_bins}
    joint = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), 4096):
            batch = indices[start:start + 4096]
            rows = _hidden_rows(hidden, batch)
            batch_joint = torch.ones(len(batch), dtype=torch.bool)
            for name in GEOMETRY_TEACHER_HEADS:
                logits = network.heads.parameter_heads[name](rows)
                expected = target_map[name][batch]
                predicted = (
                    _ordinal_median_choice(logits)
                    if name in distance_bins
                    else logits.argmax(dim=-1)
                )
                matches = predicted == expected
                correct[name] += int(matches.sum().item())
                batch_joint &= matches
                loss_sum += float(torch.nn.functional.cross_entropy(
                    logits, expected, reduction="sum"
                ).item())
                if name in distance_bins:
                    bins = distance_bins[name]
                    continuous = continuous_target_map[name][batch].clamp(
                        min=float(bins[0]), max=float(bins[-1])
                    )
                    predicted_values = bins[predicted]
                    probabilities = torch.softmax(logits, dim=-1)
                    sample_distances = torch.abs(
                        torch.log(bins).unsqueeze(0)
                        - torch.log(continuous).unsqueeze(1)
                    )
                    within_one_bin[name] += int(
                        (torch.abs(predicted - expected) <= 1).sum().item()
                    )
                    absolute_error_sum[name] += float(
                        torch.abs(predicted_values - continuous).sum().item()
                    )
                    log_ratio_error_sum[name] += float(
                        torch.abs(torch.log(predicted_values / continuous)).sum().item()
                    )
                    expected_sample_log_ratio_error_sum[name] += float(
                        (probabilities * sample_distances).sum().item()
                    )
            joint += int(batch_joint.sum().item())
    count = int(len(indices))
    result = {
        "count": float(count),
        "loss": loss_sum / float(count * len(GEOMETRY_TEACHER_HEADS)),
        "joint_accuracy": joint / float(count),
    }
    for name in GEOMETRY_TEACHER_HEADS:
        result[f"{name}_accuracy"] = correct[name] / float(count)
        values = target_map[name][indices]
        bincount = torch.bincount(values)
        majority_index = int(bincount.argmax().item())
        result[f"{name}_majority_baseline_accuracy"] = float(
            bincount.max().item() / float(count)
        )
        if name in distance_bins:
            bins = distance_bins[name]
            continuous = continuous_target_map[name][indices].clamp(
                min=float(bins[0]), max=float(bins[-1])
            )
            majority_values = torch.full_like(continuous, float(bins[majority_index]))
            oracle_values = bins[values]
            result[f"{name}_within_one_bin_accuracy"] = (
                within_one_bin[name] / float(count)
            )
            result[f"{name}_majority_baseline_within_one_bin_accuracy"] = float(
                (torch.abs(values - majority_index) <= 1).float().mean().item()
            )
            result[f"{name}_value_mae"] = absolute_error_sum[name] / float(count)
            result[f"{name}_majority_baseline_value_mae"] = float(
                torch.abs(majority_values - continuous).mean().item()
            )
            result[f"{name}_log_ratio_mae"] = log_ratio_error_sum[name] / float(count)
            result[f"{name}_expected_sample_log_ratio_mae"] = (
                expected_sample_log_ratio_error_sum[name] / float(count)
            )
            result[f"{name}_majority_baseline_log_ratio_mae"] = float(
                torch.abs(torch.log(majority_values / continuous)).mean().item()
            )
            result[f"{name}_oracle_quantization_log_ratio_mae"] = float(
                torch.abs(torch.log(oracle_values / continuous)).mean().item()
            )
            baseline = result[f"{name}_majority_baseline_log_ratio_mae"]
            result[f"{name}_log_ratio_gain_vs_majority"] = (
                1.0 - result[f"{name}_log_ratio_mae"] / max(baseline, 1e-12)
            )
            result[f"{name}_expected_sample_log_ratio_gain_vs_majority"] = (
                1.0
                - result[f"{name}_expected_sample_log_ratio_mae"]
                / max(baseline, 1e-12)
            )
    result["sl_tp_mean_accuracy"] = 0.5 * (
        result["sl_accuracy"] + result["tp_accuracy"]
    )
    result["sl_tp_mean_log_ratio_mae"] = 0.5 * (
        result["sl_log_ratio_mae"] + result["tp_log_ratio_mae"]
    )
    result["sl_tp_expected_sample_mean_log_ratio_mae"] = 0.5 * (
        result["sl_expected_sample_log_ratio_mae"]
        + result["tp_expected_sample_log_ratio_mae"]
    )
    result["sl_tp_majority_baseline_mean_log_ratio_mae"] = 0.5 * (
        result["sl_majority_baseline_log_ratio_mae"]
        + result["tp_majority_baseline_log_ratio_mae"]
    )
    result["sl_tp_log_ratio_gain_vs_majority"] = (
        1.0
        - result["sl_tp_mean_log_ratio_mae"]
        / max(result["sl_tp_majority_baseline_mean_log_ratio_mae"], 1e-12)
    )
    result["sl_tp_geometry_decision_score"] = 0.5 * (
        result["sl_tp_mean_log_ratio_mae"]
        + result["sl_tp_expected_sample_mean_log_ratio_mae"]
    )
    return result


def _ordinal_median_choice(logits: Tensor) -> Tensor:
    """Return the lowest discrete bin whose cumulative mass reaches 50%."""

    probabilities = torch.softmax(logits, dim=-1)
    return (torch.cumsum(probabilities, dim=-1) < 0.5).sum(dim=-1).clamp_max(
        logits.shape[-1] - 1
    )


def _distance_aware_geometry_row_loss(
    logits: Tensor,
    hard_targets: Tensor,
    continuous_targets: Tensor,
    bins: tuple[float, ...],
) -> Tensor:
    """Ordinal classification loss where nearby executable bins cost less.

    The hard term keeps the runtime argmax contract. The soft ordinal term and
    expected-log Huber term distinguish a one-bin miss from an economically
    unrelated SL/TP value without inventing an execution value between bins.
    """

    bin_values = torch.as_tensor(bins, dtype=logits.dtype, device=logits.device)
    log_bins = torch.log(bin_values)
    target_log = torch.log(continuous_targets.to(logits.dtype).clamp(
        min=float(bin_values[0]), max=float(bin_values[-1])
    ))
    hard_ce = torch.nn.functional.cross_entropy(logits, hard_targets, reduction="none")
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    ordinal_targets = torch.softmax(
        -torch.abs(log_bins.unsqueeze(0) - target_log.unsqueeze(1)) / 0.25,
        dim=-1,
    )
    ordinal_ce = -(ordinal_targets * log_probabilities).sum(dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    expected_absolute_distance = (
        probabilities
        * torch.abs(log_bins.unsqueeze(0) - target_log.unsqueeze(1))
    ).sum(dim=-1)
    expected_log_value = (probabilities * log_bins.unsqueeze(0)).sum(dim=-1)
    regression = torch.nn.functional.smooth_l1_loss(
        expected_log_value, target_log, beta=0.10, reduction="none"
    )
    return (
        0.40 * hard_ce
        + 0.40 * ordinal_ce
        + expected_absolute_distance
        + 0.25 * regression
    )


def _calibrate_geometry_logit_scales(
    hidden: Tensor | np.memmap,
    targets: TeacherGeometryTargets,
    network: SharedFiveTradingPolicy,
    validation_mask: Tensor,
) -> dict[str, dict[str, float]]:
    """Sharpen/soften SL and TP distributions using validation rows only."""

    audit: dict[str, dict[str, float]] = {}
    candidate_scales = (0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
    for name in ("sl", "tp"):
        head = network.heads.parameter_heads[name]
        original = {
            key: value.detach().clone() for key, value in head.state_dict().items()
        }
        best_scale = 1.0
        best_score = math.inf
        best_metrics: dict[str, float] = {}
        before_metrics: dict[str, float] = {}
        for scale in candidate_scales:
            head.load_state_dict(original)
            with torch.no_grad():
                for parameter in head.parameters():
                    parameter.mul_(scale)
            metrics = geometry_imitation_metrics(
                hidden, targets, network, validation_mask
            )
            score = 0.5 * (
                metrics[f"{name}_log_ratio_mae"]
                + metrics[f"{name}_expected_sample_log_ratio_mae"]
            )
            if scale == 1.0:
                before_metrics = {
                    "decoded_log_ratio_mae": metrics[f"{name}_log_ratio_mae"],
                    "expected_sample_log_ratio_mae": metrics[
                        f"{name}_expected_sample_log_ratio_mae"
                    ],
                    "within_one_bin_accuracy": metrics[
                        f"{name}_within_one_bin_accuracy"
                    ],
                }
            if score < best_score - 1e-12:
                best_score = score
                best_scale = scale
                best_metrics = {
                    "decoded_log_ratio_mae": metrics[f"{name}_log_ratio_mae"],
                    "expected_sample_log_ratio_mae": metrics[
                        f"{name}_expected_sample_log_ratio_mae"
                    ],
                    "within_one_bin_accuracy": metrics[
                        f"{name}_within_one_bin_accuracy"
                    ],
                }
        head.load_state_dict(original)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.mul_(best_scale)
        audit[name] = {
            "selected_scale": float(best_scale),
            "selection_score": float(best_score),
            **{f"before_{key}": value for key, value in before_metrics.items()},
            **{f"after_{key}": value for key, value in best_metrics.items()},
        }
    return audit


def train_recurrent_geometry_heads(
    network: SharedFiveTradingPolicy,
    data: TeacherImitationData,
    config: TeacherGeometryImitationConfig,
    *,
    progress: Any | None = None,
    hidden_cache_path: str | Path | None = None,
    reuse_hidden_cache: bool = False,
) -> dict[str, Any]:
    """Fit execution geometry to accepted pivot exits while freezing action timing."""

    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    network.cpu()
    network.require_current_architecture()
    targets = load_teacher_geometry_targets(data)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    trainable: list[Tensor] = []
    for name in GEOMETRY_TEACHER_HEADS:
        head = network.heads.parameter_heads[name]
        for parameter in head.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    invariant_before = geometry_invariant_sha256(network)
    trainable_before = geometry_trainable_sha256(network)
    extraction_options: dict[str, Any] = {
        "batch_size": config.feature_batch_size,
        "sequence_length": config.sequence_length,
    }
    if hidden_cache_path is not None:
        extraction_options["cache_path"] = hidden_cache_path
    if reuse_hidden_cache:
        if hidden_cache_path is None:
            raise ValueError("reuse_hidden_cache can --hidden-cache")
        hidden, sequence_audit = reuse_verified_owner_hidden_cache(
            network, data, cache_path=hidden_cache_path,
            sequence_length=config.sequence_length,
        )
    else:
        hidden, sequence_audit = extract_owner_hidden_sequences(network, data, **extraction_options)
    masks = {name: torch.as_tensor(data.splits == name) for name in ("train", "validation", "test")}
    for name, mask in masks.items():
        if not bool((mask & torch.as_tensor(targets.valid)).any()):
            raise ValueError(f"split {name} thieu accepted geometry target")
    before = {
        name: geometry_imitation_metrics(hidden, targets, network, mask)
        for name, mask in masks.items()
    }
    valid_tensor = torch.as_tensor(targets.valid, dtype=torch.bool)
    train_indices = torch.nonzero(masks["train"] & valid_tensor, as_tuple=False).reshape(-1)
    target_map = {
        "delay": torch.as_tensor(targets.delay_indices, dtype=torch.long),
        "sl": torch.as_tensor(targets.sl_indices, dtype=torch.long),
        "tp": torch.as_tensor(targets.tp_indices, dtype=torch.long),
        "trailing": torch.as_tensor(targets.trailing_indices, dtype=torch.long),
    }
    continuous_target_map = {
        "sl": torch.as_tensor(targets.sl_atr, dtype=torch.float32),
        "tp": torch.as_tensor(targets.tp_rr, dtype=torch.float32),
    }
    distance_bins = {"sl": SL_ATR_BINS, "tp": TP_RR_BINS}
    # All accepted pivot trades define one execution-geometry observation.
    # Dataset quality includes a teacher_net_r component and therefore rises
    # with the TP class itself. Reusing it here would reweight the target
    # distribution toward distant pivots (label-dependent sampling bias).
    # Action imitation keeps quality weighting; geometry intentionally does not.
    geometry_weights = torch.ones(len(data.actions), dtype=torch.float32)
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(config.seed)
    best_state = {
        name: {key: value.detach().clone() for key, value in network.heads.parameter_heads[name].state_dict().items()}
        for name in GEOMETRY_TEACHER_HEADS
    }
    best_distance_score = before["validation"]["sl_tp_geometry_decision_score"]
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        epoch_loss_sum = 0.0
        epoch_weight = 0.0
        for start in range(0, len(permutation), config.batch_size):
            index = permutation[start:start + config.batch_size]
            rows = _hidden_rows(hidden, index)
            row_loss = torch.zeros(len(index), dtype=torch.float32)
            for name in GEOMETRY_TEACHER_HEADS:
                logits = network.heads.parameter_heads[name](rows)
                if name in distance_bins:
                    row_loss += _distance_aware_geometry_row_loss(
                        logits,
                        target_map[name][index],
                        continuous_target_map[name][index],
                        distance_bins[name],
                    )
                else:
                    row_loss += torch.nn.functional.cross_entropy(
                        logits, target_map[name][index], reduction="none"
                    )
            row_loss /= float(len(GEOMETRY_TEACHER_HEADS))
            weights = geometry_weights[index]
            loss = torch.sum(row_loss * weights) / torch.sum(weights).clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
                raise RuntimeError("gradient geometry imitation NaN/Inf")
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            epoch_loss_sum += float(loss.item()) * float(weights.sum().item())
            epoch_weight += float(weights.sum().item())
        validation = geometry_imitation_metrics(hidden, targets, network, masks["validation"])
        record = {
            "epoch": float(epoch),
            "train_weighted_loss": epoch_loss_sum / max(epoch_weight, 1e-12),
            "validation_loss": validation["loss"],
            "validation_sl_accuracy": validation["sl_accuracy"],
            "validation_tp_accuracy": validation["tp_accuracy"],
            "validation_joint_accuracy": validation["joint_accuracy"],
            "validation_sl_log_ratio_mae": validation["sl_log_ratio_mae"],
            "validation_tp_log_ratio_mae": validation["tp_log_ratio_mae"],
            "validation_sl_tp_mean_log_ratio_mae": validation["sl_tp_mean_log_ratio_mae"],
            "validation_sl_tp_expected_sample_mean_log_ratio_mae": validation[
                "sl_tp_expected_sample_mean_log_ratio_mae"
            ],
            "validation_geometry_decision_score": validation["sl_tp_geometry_decision_score"],
        }
        history.append(record)
        if progress is not None:
            progress(record)
        validation_distance_score = validation["sl_tp_geometry_decision_score"]
        if validation_distance_score < best_distance_score - 1e-8:
            best_distance_score = validation_distance_score
            best_epoch = epoch
            best_state = {
                name: {key: value.detach().clone() for key, value in network.heads.parameter_heads[name].state_dict().items()}
                for name in GEOMETRY_TEACHER_HEADS
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    for name, state in best_state.items():
        network.heads.parameter_heads[name].load_state_dict(state)
    logit_scale_calibration = _calibrate_geometry_logit_scales(
        hidden, targets, network, masks["validation"]
    )
    after = {
        name: geometry_imitation_metrics(hidden, targets, network, mask)
        for name, mask in masks.items()
    }
    invariant_after = geometry_invariant_sha256(network)
    if invariant_after != invariant_before:
        raise RuntimeError("parameter ngoai geometry teacher surface da thay doi")
    trainable_after = geometry_trainable_sha256(network)
    if trainable_after == trainable_before:
        raise RuntimeError("optimizer khong cap nhat geometry heads")
    validation_exact_gain = (
        after["validation"]["sl_tp_mean_accuracy"]
        - before["validation"]["sl_tp_mean_accuracy"]
    )
    validation_metrics = after["validation"]
    if (
        best_epoch == 0
        or validation_metrics["sl_log_ratio_gain_vs_majority"] < 0.10
        or validation_metrics["tp_log_ratio_gain_vs_majority"] < 0.01
        or validation_metrics["tp_expected_sample_log_ratio_gain_vs_majority"] < 0.01
        or validation_metrics["sl_tp_log_ratio_gain_vs_majority"] < 0.05
        or validation_metrics["tp_within_one_bin_accuracy"]
        <= validation_metrics["tp_majority_baseline_within_one_bin_accuracy"] + 0.005
    ):
        raise RuntimeError(
            "validation geometry gate failed: SL/TP distance phai vuot constant-majority baseline: "
            + json.dumps({"before": before["validation"], "after": after["validation"]}, sort_keys=True)
        )
    setattr(network.heads, "teacher_geometry_imitation_trained", True)
    setattr(network.heads, "teacher_geometry_imitation_schema_version", 2)
    setattr(network.heads, "teacher_geometry_imitation_heads", ("sl", "tp"))
    return {
        "objective": {
            "name": "ordinal_log_distance_v2",
            "selection_metric": "validation_geometry_decision_score",
            "sample_weighting": "uniform_accepted_geometry_rows",
            "exact_accuracy_role": "audit_only",
            "runtime_output": "factorized_discrete_bins",
        },
        "config": config.__dict__,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "validation_exact_accuracy_gain": validation_exact_gain,
        "before": before,
        "after": after,
        "history": history,
        "logit_scale_calibration": logit_scale_calibration,
        "sequence_audit": sequence_audit,
        "geometry_invariant_sha256": invariant_after,
        "geometry_trainable_before_sha256": trainable_before,
        "geometry_trainable_after_sha256": trainable_after,
        "changed_parameters": sorted(
            f"heads.parameter_heads.{name}.{key}"
            for name in GEOMETRY_TEACHER_HEADS
            for key in network.heads.parameter_heads[name].state_dict()
        ),
    }
