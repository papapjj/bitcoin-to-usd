#!/usr/bin/env python3
"""Đánh giá đầu bảy lựa chọn theo đúng đường quyết định đang chạy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import (
    TeacherImitationData,
    extract_owner_hidden_sequences,
)
from self_growing_trader.five_agent_authority import five_agent_authority_mask
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FIVE_CROSS_AGENT_POOL_FRACTION,
    LEGACY_HEURISTIC_CLOSE_ACTIONS,
)
from self_growing_trader.self_growing_runner import (
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.seven_choice_teacher import (
    _head_metrics,
    load_seven_choice_dataset,
    shared_five_network,
    seven_choice_runtime_contract,
)


def _roots(network, data):
    unique = np.flatnonzero(data.agent_ids == 0)
    width = int(network.observation_encoder.pattern_tree.action_path_head.in_features)
    roots = torch.empty((len(unique), width), dtype=torch.float32)
    network.eval()
    with torch.no_grad():
        for start in range(0, len(unique), 4096):
            idx = unique[start:start + 4096]
            obs = torch.as_tensor(data.observations[idx], dtype=torch.float32)
            roots[start:start + len(idx)] = (
                network.observation_encoder.pattern_predictions(obs)["root"].cpu()
            )
    return torch.repeat_interleave(roots, 2, dim=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pseudo(data) -> TeacherImitationData:
    selected = data.outcome_best_actions
    return TeacherImitationData(
        path=data.path,
        observations=data.observations,
        # Recurrent previous-action inputs must come from the causal stream.
        # The outcome labels use future bars and are targets only; feeding them
        # back here would leak the answer into validation/test hidden states.
        actions=data.best_actions,
        candidate_actions=selected,
        teacher_trade_accepted=selected > 0,
        teacher_net_r=data.net_r[
            np.arange(len(selected)), selected
        ],
        agent_ids=data.agent_ids,
        quality_weights=np.ones(len(selected), dtype=np.float32),
        sample_times=data.sample_times,
        label_finalized_times=data.sample_times,
        splits=data.splits,
        manifest=data.manifest,
    )


def _block_metrics(network, data, hidden, roots, split: str, blocks: int, target_actions):
    indices = np.flatnonzero(data.splits == split)
    result = []
    for part in np.array_split(indices, blocks):
        if len(part) == 0:
            continue
        mask = np.zeros(len(data.observations), dtype=bool)
        mask[part] = True
        result.append(
            _head_metrics(
                network,
                data,
                hidden,
                roots,
                mask,
                target_actions=target_actions,
            )
        )
    return result


def _split_groups(data, split: str) -> list[np.ndarray]:
    selected = np.flatnonzero(data.splits == split)
    if selected.size == 0:
        raise ValueError(f"split rong: {split}")
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
            raise ValueError("trung agent_id tai cung sample_time")
        if not np.all(data.observations[indices] == data.observations[first]):
            raise ValueError("hai vai cung thoi diem phai co observation giong nhau")
        groups.append(indices)
        offset = stop
    return groups


def _self_fed_replay(
    network,
    data,
    split: str,
    *,
    sequence_length: int = 8192,
    window_batch_size: int = 16,
    previous_action_mode: str = "self_fed",
):
    """Phat lai tu sinh theo cac doan cuon, van hoi tiep hanh dong thuc te.

    ``forward_five`` is intentionally a one-step public API.  Calling it once
    per timestamp makes a 16-week audit take hours on CPU, so this routine
    mirrors the exact encoder/LSTM/temporal/head path while batching independent
    chronological windows.  A reset occurs only at the public Five unroll
    boundary (or a split boundary), never between rows that belong to one
    window.  No future-derived target is used as ``previous_action``.
    """
    if sequence_length < 2 or window_batch_size < 1:
        raise ValueError("sequence_length/window_batch_size khong hop le")
    if previous_action_mode not in {"self_fed", "hold"}:
        raise ValueError("previous_action_mode khong hop le")
    groups = _split_groups(data, split)
    windows = [
        groups[offset:offset + sequence_length]
        for offset in range(0, len(groups), sequence_length)
    ]
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    allowed = five_agent_authority_mask()
    # Historical snapshots do not carry portfolio state for manager agents;
    # keeping them at HOLD makes the replay explicit and deterministic.
    allowed[2:, :] = False
    allowed[2:, 0] = True
    allowed_tensor = torch.as_tensor(allowed, dtype=torch.bool, device=device)
    agent_template = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
    # Half precision is sufficient for an audit representation and keeps one
    # complete split below 2.7 GB.  The metric routine promotes each minibatch
    # to float32 when applying the heads.
    hidden_out = np.zeros(
        (len(data.observations), int(network.hidden_size)), dtype=np.float16
    )
    predicted = np.full(len(data.observations), -1, dtype=np.int64)
    action_digest = hashlib.sha256()
    network.eval()
    with torch.no_grad():
        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start:batch_start + window_batch_size]
            batch_count = len(batch_windows)
            hidden_state = torch.zeros(
                (batch_count, FIVE_AGENT_COUNT, network.hidden_size),
                dtype=dtype,
                device=device,
            )
            cell_state = torch.zeros_like(hidden_state)
            previous = torch.zeros(
                (batch_count, FIVE_AGENT_COUNT), dtype=torch.long, device=device
            )
            temporal_tokens, temporal_valid = (
                network.post_lstm_transformer.empty_temporal_memory(
                    batch_count, device=device, dtype=dtype
                )
            )
            max_steps = max(len(window) for window in batch_windows)
            for step in range(max_steps):
                active = [
                    index for index, window in enumerate(batch_windows)
                    if step < len(window)
                ]
                active_tensor = torch.as_tensor(active, dtype=torch.long, device=device)
                step_groups = [batch_windows[index][step] for index in active]
                observations = torch.as_tensor(
                    np.stack([data.observations[int(indices[0])] for indices in step_groups]),
                    dtype=torch.float32,
                    device=device,
                )
                active_count = len(active)
                observations5 = observations[:, None, :].expand(
                    active_count, FIVE_AGENT_COUNT, -1
                ).reshape(-1, observations.shape[-1])
                agent_ids = agent_template.repeat(active_count)
                previous_flat = previous[active_tensor].reshape(-1)
                components = network.observation_encoder.encode_components(
                    observations5, agent_ids, previous_flat
                )
                encoded = torch.tanh(
                    network.observation_encoder.pre_lstm(
                        torch.cat((
                            components["market"],
                            components["account"],
                            components["execution"],
                            components["positions"],
                            components["agent"],
                            components["previous_action"],
                        ), dim=-1)
                    )
                    + network.observation_encoder.pattern_residual(
                        components["pattern_root"]
                    )
                ).reshape(active_count, FIVE_AGENT_COUNT, -1)
                width = max(
                    1,
                    int(round(encoded.shape[-1] * FIVE_CROSS_AGENT_POOL_FRACTION)),
                )
                prefix = encoded[:, :, :width].max(dim=1, keepdim=True).values
                encoded = torch.cat(
                    (prefix.expand(-1, FIVE_AGENT_COUNT, -1), encoded[:, :, width:]),
                    dim=-1,
                )
                next_hidden, next_cell = network.lstm(
                    encoded.reshape(active_count * FIVE_AGENT_COUNT, -1),
                    (
                        hidden_state[active_tensor].reshape(
                            active_count * FIVE_AGENT_COUNT, -1
                        ),
                        cell_state[active_tensor].reshape(
                            active_count * FIVE_AGENT_COUNT, -1
                        ),
                    ),
                )
                next_hidden = next_hidden.reshape(
                    active_count, FIVE_AGENT_COUNT, network.hidden_size
                )
                next_cell = next_cell.reshape(
                    active_count, FIVE_AGENT_COUNT, network.hidden_size
                )
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
                hidden_state[active_tensor] = next_hidden
                cell_state[active_tensor] = next_cell
                temporal_tokens[active_tensor] = updated_temporal
                temporal_valid[active_tensor] = updated_valid

                hidden_flat = policy_hidden.reshape(-1, network.hidden_size)
                action_mode = str(
                    getattr(network.heads, "seven_choice_action_decision_mode", "expected_value")
                )
                action_logits = (
                    network.heads.direct_action_logits(
                        hidden_flat,
                        agent_ids,
                        observations5,
                    )
                    if action_mode == "direct"
                    else network.heads.primary_action_logits(hidden_flat, agent_ids)
                )
                action_logits = network._apply_seven_choice_expected_value_decision(
                    action_logits,
                    agent_ids,
                    components["pattern_action_path"],
                    policy_hidden=hidden_flat,
                )
                for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
                    action_logits[:, legacy_action] = -30.0
                allowed_batch = allowed_tensor.repeat(active_count, 1)
                choices, _, _, _ = network._primary_action_distribution(
                    action_logits,
                    allowed=allowed_batch,
                    deterministic=True,
                    entry_probability_floor=0.0,
                    uniform_mix=0.0,
                )
                emitted = choices.reshape(active_count, FIVE_AGENT_COUNT)
                if previous_action_mode == "self_fed":
                    previous[active_tensor] = emitted
                emitted_np = emitted.detach().cpu().numpy().astype(np.int16)
                for local_index, indices in enumerate(step_groups):
                    action_digest.update(emitted_np[local_index].tobytes(order="C"))
                    owners = np.asarray(data.agent_ids[indices], dtype=np.int64)
                    if np.any((owners < 0) | (owners >= FIVE_AGENT_COUNT)):
                        raise ValueError("agent_id ngoai pham vi Five")
                    hidden_rows = policy_hidden[local_index, owners]
                    hidden_out[indices] = hidden_rows.detach().cpu().numpy().astype(
                        np.float16
                    )
                    predicted[indices] = emitted_np[local_index, owners]
    split_indices = np.flatnonzero(data.splits == split)
    if np.any(predicted[split_indices] < 0):
        raise RuntimeError("phat lai tu sinh khong gan du hanh dong")
    return hidden_out, predicted, {
        "mode": "batched_window_self_fed_deterministic_v3",
        "split": split,
        "time_steps": len(groups),
        "sequence_windows": len(windows),
        "window_batch_size": int(window_batch_size),
        "sequence_length": int(sequence_length),
        "state_reset_count": len(windows),
        "teacher_action_feedback": False,
        "previous_action_mode": previous_action_mode,
        "manager_agents_forced_hold": True,
        "action_sequence_sha256": action_digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--sequence-length", type=int, default=8192,
        help="Do dai moi doan cuon; 8192 la bien cong khai cua Five.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--replay-window-batch-size", type=int, default=32,
        help="So doan doc lap chay song song trong phat lai tu sinh.",
    )
    parser.add_argument(
        "--previous-action-mode", choices=("self_fed", "hold"), default="self_fed",
        help=(
            "Cach lap lai truong hanh dong truoc: self_fed dung hanh dong vua phat; "
            "hold giu HOLD de do do on dinh khi dau vao khong phu thuoc lich su hanh dong."
        ),
    )
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument(
        "--decision-mode", choices=("checkpoint", "expected_value", "direct"),
        default="checkpoint",
        help=(
            "Nguon quyet dinh can danh gia. checkpoint giu cau hinh luu trong tep; "
            "expected_value bat mat gia tri; direct chi dung dau ra hanh dong."
        ),
    )
    parser.add_argument(
        "--residual-scales", default=None,
        help="Cac muc phan bu dau hanh dong de do cung mot lan phat lai.",
    )
    parser.add_argument(
        "--thresholds", default="",
        help="Cac nguong R tuy chon de quet; mac dinh dung nguong trong checkpoint.",
    )
    parser.add_argument(
        "--teacher-forced", action="store_true",
        help="Chi dung cho chan doan cu; mac dinh phat lai tu sinh.",
    )
    parser.add_argument(
        "--sweep-self-fed", action="store_true",
        help="Phat lai lai tung cap nguong/phan bu; co the rat lau.",
    )
    parser.add_argument(
        "--allow-contract-mismatch", action="store_true",
        help="Chi cho phep soi ho so cu co hop dong khac bo nhan; bao cao van khong duoc dung de day MT5/PPO.",
    )
    args = parser.parse_args()
    if args.blocks < 1 or args.batch_size < 1 or args.replay_window_batch_size < 1:
        raise ValueError("blocks/batch-size/replay-window-batch-size khong hop le")
    residual_scales = (
        None if args.residual_scales is None
        else [float(item) for item in str(args.residual_scales).split(",") if item.strip()]
    )
    thresholds = [float(item) for item in str(args.thresholds).split(",") if item.strip()]
    data = load_seven_choice_dataset(args.dataset)
    config = load_config(args.config)
    runtime_contract = seven_choice_runtime_contract(data, config)
    if not bool(runtime_contract["match"]) and not args.allow_contract_mismatch:
        raise RuntimeError(
            "bo nhan va cau hinh chay khong cung hop dong; "
            "dung --allow-contract-mismatch chi de soi ho so cu: "
            + json.dumps(runtime_contract["mismatches"], ensure_ascii=False)
        )
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        args.checkpoint, config, seed=0
    )
    network = shared_five_network(system)
    network.require_current_architecture()
    if args.decision_mode == "expected_value":
        network.heads.seven_choice_expected_value_decision_enabled = True
        network.heads.seven_choice_action_decision_mode = "expected_value"
    elif args.decision_mode == "direct":
        network.heads.seven_choice_expected_value_decision_enabled = False
        network.heads.seven_choice_action_decision_mode = "direct"
    network.eval()
    roots = _roots(network, data)
    target_actions = data.outcome_best_actions
    base_threshold = float(getattr(network.heads, "seven_choice_expected_value_threshold_r", 0.0))
    original_scale = float(
        getattr(network.heads, "seven_choice_expected_value_action_residual_scale", 0.0)
    )
    if residual_scales is None:
        residual_scales = [original_scale]
    if not residual_scales or any(
        not np.isfinite(item) or item < 0.0 for item in residual_scales
    ):
        raise ValueError("residual-scales khong hop le")
    thresholds = [float(item) for item in str(args.thresholds).split(",") if item.strip()]
    if not thresholds:
        thresholds = [base_threshold]
    # A self-fed replay is the default.  It has to be repeated for every
    # calibration point because changing a decision changes the next
    # previous-action input and therefore the recurrent state.
    requested_pairs = [
        (float(scale), float(threshold))
        for scale in residual_scales
        for threshold in thresholds
    ]
    has_non_base_pair = any(
        abs(scale - original_scale) >= 1e-12
        or abs(threshold - base_threshold) >= 1e-12
        for scale, threshold in requested_pairs
    )
    if not args.teacher_forced and has_non_base_pair and not args.sweep_self_fed:
        raise ValueError(
            "cap nguong/phan bu khac checkpoint phai dung --sweep-self-fed; "
            "neu khong chi duoc danh gia cap mac dinh"
        )

    # Teacher-forced hidden states depend on the frozen causal action stream,
    # not on the threshold or residual scale being swept.  Keep one verified
    # extraction in memory so a calibration sweep does not replay the same
    # recurrent windows once per threshold.  Self-fed replay must still run
    # separately because each decision changes the next recurrent state.
    teacher_forced_hidden_by_split = None
    teacher_forced_extract_audit = None

    def evaluate_config(scale: float, threshold: float):
        nonlocal teacher_forced_hidden_by_split, teacher_forced_extract_audit
        network.heads.seven_choice_expected_value_action_residual_scale = scale
        network.heads.seven_choice_expected_value_threshold_r = threshold
        if args.teacher_forced:
            if teacher_forced_hidden_by_split is None:
                pseudo = _pseudo(data)
                hidden, audit = extract_owner_hidden_sequences(
                    network,
                    pseudo,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    allow_extended_sequence=bool(args.sequence_length > 64),
                    output_dtype=torch.float16,
                )
                teacher_forced_hidden_by_split = {
                    split: hidden for split in ("train", "validation", "test")
                }
                teacher_forced_extract_audit = audit
            hidden_by_split = teacher_forced_hidden_by_split
            audits = {
                "mode": "teacher_forced_causal_actions_cached",
                "extract": teacher_forced_extract_audit,
                "reused_for_calibration": True,
            }
        else:
            hidden_by_split = {}
            audits = {"mode": "batched_window_self_fed_deterministic_v3", "splits": {}}
            hidden_all = np.zeros(
                (len(data.observations), int(network.hidden_size)), dtype=np.float16
            )
            for split in ("train", "validation", "test"):
                hidden_split, _, audit = _self_fed_replay(
                    network,
                    data,
                    split,
                    sequence_length=args.sequence_length,
                    window_batch_size=args.replay_window_batch_size,
                    previous_action_mode=args.previous_action_mode,
                )
                split_mask = data.splits == split
                hidden_all[split_mask] = hidden_split[split_mask]
                del hidden_split
                audits["splits"][split] = audit
            # All metrics use one merged representation. Keeping three full
            # zero-padded arrays here previously exhausted the pagefile.
            hidden_by_split = {
                split: hidden_all for split in ("train", "validation", "test")
            }
        split_metrics = {
            split: _head_metrics(
                network,
                data,
                hidden_by_split[split],
                roots,
                data.splits == split,
                target_actions=target_actions,
            )
            for split in ("train", "validation", "test")
        }
        block_metrics = {
            split: _block_metrics(
                network,
                data,
                hidden_by_split[split],
                roots,
                split,
                args.blocks,
                target_actions,
            )
            for split in ("validation", "test")
        }
        return split_metrics, block_metrics, audits

    base_splits, base_blocks, sequence_audit = evaluate_config(
        original_scale, base_threshold
    )
    splits = base_splits
    block_splits = base_blocks
    residual_sweep = []
    for scale in residual_scales:
        for threshold in thresholds:
            if (
                not args.teacher_forced
                and args.sweep_self_fed
                and not (abs(scale - original_scale) < 1e-12 and abs(threshold - base_threshold) < 1e-12)
            ):
                current_splits, _, _ = evaluate_config(scale, threshold)
            elif not args.teacher_forced and not args.sweep_self_fed:
                current_splits = base_splits
            else:
                network.heads.seven_choice_expected_value_action_residual_scale = scale
                network.heads.seven_choice_expected_value_threshold_r = threshold
                if args.teacher_forced:
                    current_splits, _, _ = evaluate_config(scale, threshold)
                else:
                    current_splits = base_splits
            network.heads.seven_choice_expected_value_threshold_r = threshold
            residual_sweep.append({
                "residual_scale": scale,
                "threshold_r": threshold,
                "validation": current_splits["validation"],
                "test": current_splits["test"],
            })
    network.heads.seven_choice_expected_value_action_residual_scale = original_scale
    network.heads.seven_choice_expected_value_threshold_r = base_threshold
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "runtime_contract": runtime_contract,
        "sequence_audit": sequence_audit,
        "decision_config": {
            "requested_mode": args.decision_mode,
            "enabled": bool(getattr(network.heads, "seven_choice_expected_value_decision_enabled", False)),
            "threshold_r": float(getattr(network.heads, "seven_choice_expected_value_threshold_r", 0.0)),
            "score_mode": str(getattr(network.heads, "seven_choice_expected_value_score_mode", "value")),
        "action_residual_scale": float(getattr(network.heads, "seven_choice_expected_value_action_residual_scale", 0.0)),
        "hidden_value_residual_enabled": bool(
            getattr(
                network.heads,
                "seven_choice_expected_value_hidden_residual_enabled",
                False,
            )
        ),
        "hidden_value_residual_scale_r": float(
            getattr(
                network.heads,
                "seven_choice_expected_value_hidden_residual_scale_r",
                0.0,
            )
        ),
        "fill_probability_floor": float(getattr(
                network.heads, "seven_choice_expected_value_fill_probability_floor", 0.0
            )),
        },
        "target": "outcome_best_actions",
        "teacher_forcing_actions": "best_actions_causal",
        "metrics_use_live_decision_path": True,
        "self_fed_replay": not args.teacher_forced,
        "previous_action_mode": (
            "teacher_forced_causal_actions" if args.teacher_forced
            else args.previous_action_mode
        ),
        "splits": splits,
        "blocks": block_splits,
        "residual_threshold_sweep": residual_sweep,
        "verification": {
            "future_fields_used_as_inputs": False,
            "mt5_oos_proven": False,
            "ppo_allowed": False,
            "runtime_contract_match": bool(runtime_contract["match"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "test": splits["test"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
