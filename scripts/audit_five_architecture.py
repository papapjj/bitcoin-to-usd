#!/usr/bin/env python3
"""Sequential, reproducible capacity audit for the active Five v2 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torchinfo import summary
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import (
    _role_logits,
    network_invariant_sha256,
    shared_five_network,
    teacher_trainable_sha256,
)
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FIVE_CROSS_AGENT_POOL_FRACTION,
    LEGACY_HEURISTIC_CLOSE_ACTIONS,
    SharedFiveTradingPolicy,
)
from self_growing_trader.self_growing_runner import (
    tai_he_chuyen_gia_mot_context_mot_policy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_network(checkpoint: Path, config_path: Path):
    config = load_config(config_path)
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        checkpoint, config, policy_core="current", seed=0
    )
    network = shared_five_network(system)
    network.require_current_architecture()
    network.eval()
    return network


def _module_parameter_rows(network: torch.nn.Module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, module in network.named_modules():
        direct = list(module.parameters(recurse=False))
        subtree = list(module.parameters(recurse=True))
        rows.append({
            "module": name or "<root>",
            "type": type(module).__name__,
            "direct_parameters": int(sum(value.numel() for value in direct)),
            "subtree_parameters": int(sum(value.numel() for value in subtree)),
            "direct_trainable_parameters": int(
                sum(value.numel() for value in direct if value.requires_grad)
            ),
            "subtree_trainable_parameters": int(
                sum(value.numel() for value in subtree if value.requires_grad)
            ),
        })
    return rows


def run_torchinfo(checkpoint: Path, config_path: Path, output_dir: Path) -> Path:
    network = _load_network(checkpoint, config_path)
    model_summary = summary(
        network,
        depth=5,
        col_names=("num_params", "trainable"),
        verbose=0,
    )
    rows = _module_parameter_rows(network)
    top_level = [row for row in rows if row["module"].count(".") == 0]
    payload = {
        "stage": 1,
        "name": "torchinfo_parameter_audit",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(config_path),
        "torch_version": torch.__version__,
        "torchinfo_version": __import__("torchinfo").__version__,
        "policy_architecture_version": int(network.policy_architecture_version),
        "total_parameters": int(sum(value.numel() for value in network.parameters())),
        "trainable_parameters": int(
            sum(value.numel() for value in network.parameters() if value.requires_grad)
        ),
        "top_level_modules": top_level,
        "all_modules": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "01_torchinfo_parameters.json"
    text_path = output_dir / "01_torchinfo_parameters.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(str(model_summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": str(json_path),
        "total_parameters": payload["total_parameters"],
        "trainable_parameters": payload["trainable_parameters"],
        "top_level_modules": top_level,
    }, ensure_ascii=False, indent=2))
    return json_path


def run_profiler(
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    *,
    iterations: int,
) -> Path:
    network = _load_network(checkpoint, config_path)
    observations = torch.zeros((5, 137), dtype=torch.float32)
    allowed = torch.ones((5, 12), dtype=torch.bool)
    selected_names = {
        "observation_encoder",
        "lstm",
        "post_lstm_transformer",
        "post_lstm_transformer.input_projection",
        "post_lstm_transformer.encoder_layer",
        "post_lstm_transformer.encoder_layer.self_attn",
        "post_lstm_transformer.encoder_layer.linear1",
        "post_lstm_transformer.encoder_layer.linear2",
        "post_lstm_transformer.output_projection",
        "heads.action_query",
        "heads.action_embeddings",
        "heads.value",
    }
    elapsed_ns: dict[str, int] = {name: 0 for name in selected_names}
    calls: dict[str, int] = {name: 0 for name in selected_names}
    starts: dict[str, list[int]] = {name: [] for name in selected_names}
    handles = []

    def before(name: str):
        def hook(_module, _inputs):
            starts[name].append(time.perf_counter_ns())
        return hook

    def after(name: str):
        def hook(_module, _inputs, _output):
            start = starts[name].pop()
            elapsed_ns[name] += time.perf_counter_ns() - start
            calls[name] += 1
        return hook

    for name, module in network.named_modules():
        if name in selected_names:
            handles.append(module.register_forward_pre_hook(before(name)))
            handles.append(module.register_forward_hook(after(name)))

    try:
        with torch.inference_mode():
            for _ in range(2):
                network.forward_five(
                    observations,
                    network.initial_memories(),
                    allowed_action_masks=allowed,
                    deterministic=True,
                )
            for name in selected_names:
                elapsed_ns[name] = 0
                calls[name] = 0
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                for _ in range(iterations):
                    with record_function("five_policy_forward"):
                        network.forward_five(
                            observations,
                            network.initial_memories(),
                            allowed_action_masks=allowed,
                            deterministic=True,
                        )
    finally:
        for handle in handles:
            handle.remove()

    module_rows = []
    for name in sorted(selected_names):
        count = calls[name]
        module_rows.append({
            "module": name,
            "calls": count,
            "total_ms": elapsed_ns[name] / 1_000_000.0,
            "mean_ms_per_call": (
                elapsed_ns[name] / 1_000_000.0 / count if count else None
            ),
            "timing_scope": "inclusive_wall_clock",
        })
    operator_rows = []
    for event in sorted(
        prof.key_averages(), key=lambda item: item.self_cpu_time_total, reverse=True
    )[:80]:
        operator_rows.append({
            "operator": event.key,
            "calls": int(event.count),
            "self_cpu_ms": float(event.self_cpu_time_total / 1000.0),
            "total_cpu_ms": float(event.cpu_time_total / 1000.0),
            "flops": int(event.flops or 0),
            "self_cpu_memory_bytes": int(event.self_cpu_memory_usage),
        })
    payload = {
        "stage": 2,
        "name": "torch_profiler_module_compute_time",
        "checkpoint": str(checkpoint),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "warmup_iterations": 2,
        "profiled_iterations": iterations,
        "input_shape": [5, 137],
        "module_times": module_rows,
        "operator_profile": operator_rows,
        "operator_flops_total": int(sum(row["flops"] for row in operator_rows)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "02_profiler.json"
    trace_path = output_dir / "02_profiler_trace.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    prof.export_chrome_trace(str(trace_path))
    print(json.dumps({
        "artifact": str(json_path),
        "trace": str(trace_path),
        "profiled_iterations": iterations,
        "module_times": module_rows,
        "operator_flops_total": payload["operator_flops_total"],
    }, ensure_ascii=False, indent=2))
    return json_path


def _sample_split_windows(
    dataset: Path,
    *,
    split: str,
    sequence_length: int,
    window_count: int,
) -> dict[str, np.ndarray]:
    with np.load(dataset, allow_pickle=False) as archive:
        sample_times = archive["sample_times"]
        splits = archive["splits"]
        group_starts = np.flatnonzero(np.r_[True, sample_times[1:] != sample_times[:-1]])
        group_ends = np.r_[group_starts[1:], len(sample_times)]
        split_group_positions = np.flatnonzero(splits[group_starts] == split)
        if len(split_group_positions) < sequence_length:
            raise ValueError(f"split {split} khong du {sequence_length} time steps")
        first = int(split_group_positions[0])
        last = int(split_group_positions[-1])
        if not np.array_equal(split_group_positions, np.arange(first, last + 1)):
            raise ValueError(f"split {split} khong lien tuc theo thoi gian")
        possible = len(split_group_positions) - sequence_length
        offsets = np.unique(np.linspace(0, possible, num=window_count, dtype=np.int64))
        observations_all = archive["observations"]
        actions_all = archive["actions"]
        agent_ids_all = archive["agent_ids"]
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        owner_masks: list[np.ndarray] = []
        selected_times: list[str] = []
        for offset in offsets:
            for position in split_group_positions[offset:offset + sequence_length]:
                start = int(group_starts[position])
                stop = int(group_ends[position])
                ids = np.asarray(agent_ids_all[start:stop], dtype=np.int64)
                if len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= FIVE_AGENT_COUNT):
                    raise ValueError("agent owner tai sample_time khong hop le")
                if not np.all(observations_all[start:stop] == observations_all[start]):
                    raise ValueError("observation cua cac owner tai cung time phai giong nhau")
                action_packet = np.zeros(FIVE_AGENT_COUNT, dtype=np.int64)
                owner_mask = np.zeros(FIVE_AGENT_COUNT, dtype=bool)
                action_packet[ids] = np.asarray(actions_all[start:stop], dtype=np.int64)
                owner_mask[ids] = True
                observations.append(np.asarray(observations_all[start], dtype=np.float32))
                actions.append(action_packet)
                owner_masks.append(owner_mask)
                selected_times.append(str(sample_times[start]))
    return {
        "observations": np.stack(observations).reshape(len(offsets), sequence_length, -1),
        "actions": np.stack(actions).reshape(len(offsets), sequence_length, FIVE_AGENT_COUNT),
        "owner_masks": np.stack(owner_masks).reshape(len(offsets), sequence_length, FIVE_AGENT_COUNT),
        "sample_times": np.asarray(selected_times).reshape(len(offsets), sequence_length),
    }


def _replay_hidden(
    network: torch.nn.Module,
    sampled: dict[str, np.ndarray],
    *,
    collect_attention: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    observations = sampled["observations"]
    actions = sampled["actions"]
    owner_masks = sampled["owner_masks"]
    windows, steps, feature_count = observations.shape
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    hidden = torch.zeros((windows, FIVE_AGENT_COUNT, network.hidden_size), device=device, dtype=dtype)
    cell = torch.zeros_like(hidden)
    previous = torch.zeros((windows, FIVE_AGENT_COUNT), device=device, dtype=torch.long)
    ids = torch.arange(FIVE_AGENT_COUNT, device=device, dtype=torch.long).repeat(windows)
    raw_rows: list[np.ndarray] = []
    transformed_rows: list[np.ndarray] = []
    attention_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for step in range(steps):
            obs = torch.as_tensor(observations[:, step], device=device, dtype=dtype)
            obs5 = obs[:, None, :].expand(windows, FIVE_AGENT_COUNT, feature_count).reshape(-1, feature_count)
            components = network.observation_encoder.encode_components(obs5, ids, previous.reshape(-1))
            flattened = torch.cat((
                components["market"], components["account"], components["execution"],
                components["positions"], components["agent"], components["previous_action"],
            ), dim=-1)
            encoded = torch.tanh(
                network.observation_encoder.pre_lstm(flattened)
                + network.observation_encoder.pattern_residual(components["pattern_root"])
            ).reshape(windows, FIVE_AGENT_COUNT, -1)
            width = max(1, int(round(encoded.shape[-1] * FIVE_CROSS_AGENT_POOL_FRACTION)))
            shared = encoded[:, :, :width].max(dim=1, keepdim=True).values
            encoded = torch.cat((shared.expand(-1, FIVE_AGENT_COUNT, -1), encoded[:, :, width:]), dim=-1)
            next_hidden, next_cell = network.lstm(
                encoded.reshape(windows * FIVE_AGENT_COUNT, -1),
                (hidden.reshape(windows * FIVE_AGENT_COUNT, -1), cell.reshape(windows * FIVE_AGENT_COUNT, -1)),
            )
            hidden = next_hidden.reshape(windows, FIVE_AGENT_COUNT, -1)
            cell = next_cell.reshape(windows, FIVE_AGENT_COUNT, -1)
            if collect_attention:
                block = network.post_lstm_transformer
                projected = block.input_projection(hidden)
                normalized = block.encoder_layer.norm1(projected)
                qkv = torch.nn.functional.linear(
                    normalized,
                    block.encoder_layer.self_attn.in_proj_weight,
                    block.encoder_layer.self_attn.in_proj_bias,
                )
                query, key, _ = qkv.chunk(3, dim=-1)
                head_width = block.model_width // block.head_count
                query = query.reshape(windows, FIVE_AGENT_COUNT, block.head_count, head_width).transpose(1, 2)
                key = key.reshape(windows, FIVE_AGENT_COUNT, block.head_count, head_width).transpose(1, 2)
                weights = torch.softmax(
                    torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(float(head_width)),
                    dim=-1,
                )
                attention_rows.append(weights.detach().cpu().numpy())
            transformed = network.post_lstm_transformer(hidden)
            owner_mask = torch.as_tensor(owner_masks[:, step], device=device, dtype=torch.bool)
            raw_rows.append(hidden[owner_mask].detach().cpu().numpy())
            transformed_rows.append(transformed[owner_mask].detach().cpu().numpy())
            previous = torch.as_tensor(actions[:, step], device=device, dtype=torch.long)
    attention = np.concatenate(attention_rows, axis=0) if attention_rows else None
    return (
        np.concatenate(raw_rows, axis=0),
        np.concatenate(transformed_rows, axis=0),
        attention,
    )


def _replay_raw_hidden_cube(
    network: torch.nn.Module,
    sampled: dict[str, np.ndarray],
) -> np.ndarray:
    """Replay causal LSTM state as [window, time, agent, hidden]."""
    observations = sampled["observations"]
    actions = sampled["actions"]
    windows, steps, feature_count = observations.shape
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    hidden = torch.zeros((windows, FIVE_AGENT_COUNT, network.hidden_size), device=device, dtype=dtype)
    cell = torch.zeros_like(hidden)
    previous = torch.zeros((windows, FIVE_AGENT_COUNT), device=device, dtype=torch.long)
    ids = torch.arange(FIVE_AGENT_COUNT, device=device, dtype=torch.long).repeat(windows)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for step in range(steps):
            obs = torch.as_tensor(observations[:, step], device=device, dtype=dtype)
            obs5 = obs[:, None, :].expand(windows, FIVE_AGENT_COUNT, feature_count).reshape(-1, feature_count)
            components = network.observation_encoder.encode_components(obs5, ids, previous.reshape(-1))
            flattened = torch.cat((
                components["market"], components["account"], components["execution"],
                components["positions"], components["agent"], components["previous_action"],
            ), dim=-1)
            encoded = torch.tanh(
                network.observation_encoder.pre_lstm(flattened)
                + network.observation_encoder.pattern_residual(components["pattern_root"])
            ).reshape(windows, FIVE_AGENT_COUNT, -1)
            width = max(1, int(round(encoded.shape[-1] * FIVE_CROSS_AGENT_POOL_FRACTION)))
            shared = encoded[:, :, :width].max(dim=1, keepdim=True).values
            encoded = torch.cat((shared.expand(-1, FIVE_AGENT_COUNT, -1), encoded[:, :, width:]), dim=-1)
            next_hidden, next_cell = network.lstm(
                encoded.reshape(windows * FIVE_AGENT_COUNT, -1),
                (hidden.reshape(windows * FIVE_AGENT_COUNT, -1), cell.reshape(windows * FIVE_AGENT_COUNT, -1)),
            )
            hidden = next_hidden.reshape(windows, FIVE_AGENT_COUNT, -1)
            cell = next_cell.reshape(windows, FIVE_AGENT_COUNT, -1)
            rows.append(hidden.detach().cpu().numpy().astype(np.float32, copy=False))
            previous = torch.as_tensor(actions[:, step], device=device, dtype=torch.long)
    return np.stack(rows, axis=1)


def _temporal_role_rows(
    hidden: np.ndarray,
    actions: np.ndarray,
    owner_masks: np.ndarray,
    *,
    agent_id: int,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair current h_t with the same role's teacher action at t-lag."""
    if lag < 0 or lag >= hidden.shape[1]:
        raise ValueError("lag must be within the replay sequence")
    current_mask = owner_masks[:, lag:, agent_id]
    past_mask = owner_masks[:, : hidden.shape[1] - lag, agent_id]
    valid = current_mask & past_mask
    x = hidden[:, lag:, agent_id, :][valid]
    y = actions[:, : hidden.shape[1] - lag, agent_id][valid].astype(np.int64)
    return x, y


def _rank_metrics(values: np.ndarray, *, components: int) -> dict[str, object]:
    centered = np.asarray(values, dtype=np.float32)
    centered = centered - centered.mean(axis=0)
    sample_count, width = centered.shape
    max_rank = min(sample_count - 1, width)
    component_count = min(max(2, components), max_rank)
    solver = "full" if component_count == max_rank else "randomized"
    pca = PCA(n_components=component_count, svd_solver=solver, random_state=0)
    pca.fit(centered)
    eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64)
    total_variance = float(np.var(centered, axis=0, ddof=1).sum())
    captured = float(eigenvalues.sum())
    residual = max(0.0, total_variance - captured)
    spectrum_min = np.r_[eigenvalues, residual] if residual > 0.0 else eigenvalues
    probabilities = spectrum_min / max(total_variance, 1e-30)
    probabilities = probabilities[probabilities > 0.0]
    effective_rank_lower = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    remaining_rank = max(1, max_rank - component_count)
    if residual > 0.0 and remaining_rank > 1:
        spectrum_max = np.r_[eigenvalues, np.full(remaining_rank, residual / remaining_rank)]
        spread_probabilities = spectrum_max / max(total_variance, 1e-30)
        effective_rank_upper = float(
            np.exp(-(spread_probabilities * np.log(spread_probabilities)).sum())
        )
    else:
        effective_rank_upper = effective_rank_lower
    cumulative = np.cumsum(eigenvalues) / max(total_variance, 1e-30)

    def rank_at(target: float) -> int | str:
        hits = np.flatnonzero(cumulative >= target)
        return int(hits[0] + 1) if len(hits) else f">{component_count}"

    std = np.std(centered, axis=0, ddof=1)
    squared_sum = float(np.square(eigenvalues).sum())
    participation_lower = float(
        total_variance ** 2 / max(squared_sum + residual ** 2, 1e-30)
    )
    participation_upper = float(
        total_variance ** 2
        / max(squared_sum + residual ** 2 / remaining_rank, 1e-30)
    )
    return {
        "samples": sample_count,
        "width": width,
        "sample_limited_max_rank": max_rank,
        "svd_solver": solver,
        "svd_components": component_count,
        "variance_captured_by_svd": captured / max(total_variance, 1e-30),
        "rank_90pct": rank_at(0.90),
        "rank_95pct": rank_at(0.95),
        "rank_99pct": rank_at(0.99),
        "stable_rank": float(total_variance / max(eigenvalues[0], 1e-30)),
        "entropy_effective_rank_lower": effective_rank_lower,
        "entropy_effective_rank_upper": effective_rank_upper,
        "participation_rank_lower": participation_lower,
        "participation_rank_upper": participation_upper,
        "dimensions_std_gt_1e-4": int(np.sum(std > 1e-4)),
        "dimensions_std_gt_1e-3": int(np.sum(std > 1e-3)),
        "dimensions_std_gt_1e-2": int(np.sum(std > 1e-2)),
        "mean_abs_activation": float(np.mean(np.abs(values))),
        "saturation_fraction_abs_gt_0_95": float(np.mean(np.abs(values) > 0.95)),
    }


def run_effective_rank(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    svd_components: int,
) -> Path:
    network = _load_network(checkpoint, config_path)
    sampled = _sample_split_windows(
        dataset, split="test", sequence_length=sequence_length, window_count=windows
    )
    raw_hidden, transformed_hidden, _ = _replay_hidden(network, sampled)
    payload = {
        "stage": 3,
        "name": "lstm_hidden_pca_svd_effective_rank",
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "split": "test",
        "window_count": int(sampled["observations"].shape[0]),
        "sequence_length": sequence_length,
        "first_sample_time": str(sampled["sample_times"].reshape(-1)[0]),
        "last_sample_time": str(sampled["sample_times"].reshape(-1)[-1]),
        "raw_lstm_hidden": _rank_metrics(raw_hidden, components=svd_components),
        "post_transformer_hidden": _rank_metrics(transformed_hidden, components=svd_components),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "03_effective_rank.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "03_hidden_sample.npz",
        raw_lstm_hidden=raw_hidden.astype(np.float16),
        post_transformer_hidden=transformed_hidden.astype(np.float16),
        sample_times=sampled["sample_times"],
    )
    print(json.dumps({"artifact": str(json_path), **payload}, ensure_ascii=False, indent=2))
    return json_path


def run_attention_map(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
) -> Path:
    network = _load_network(checkpoint, config_path)
    sampled = _sample_split_windows(
        dataset, split="test", sequence_length=sequence_length, window_count=windows
    )
    raw_hidden, transformed_hidden, attention = _replay_hidden(
        network, sampled, collect_attention=True
    )
    if attention is None:
        raise RuntimeError("khong thu duoc attention weights")
    mean_by_head = attention.mean(axis=0)
    diagonal = np.diagonal(attention, axis1=-2, axis2=-1)
    entropy = -np.sum(attention * np.log(np.clip(attention, 1e-12, 1.0)), axis=-1)
    normalized_entropy = entropy / np.log(float(FIVE_AGENT_COUNT))
    raw_norm = np.linalg.norm(raw_hidden, axis=1)
    correction_norm = np.linalg.norm(transformed_hidden - raw_hidden, axis=1)
    block = network.post_lstm_transformer
    payload = {
        "stage": 4,
        "name": "post_lstm_transformer_attention_map",
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "split": "test",
        "token_axis": "five_simultaneous_agents",
        "token_labels": [f"agent_{index}" for index in range(FIVE_AGENT_COUNT)],
        "temporal_attention_available": False,
        "max_temporal_attention_lag": 0,
        "history_conclusion": (
            "Transformer khong co truc thoi gian; history xa chi co the di qua recurrent state cua LSTM."
        ),
        "attention_samples": int(attention.shape[0]),
        "heads": int(attention.shape[1]),
        "mean_attention_by_head": mean_by_head.tolist(),
        "mean_diagonal_attention": float(diagonal.mean()),
        "mean_off_diagonal_attention": float((attention.sum(axis=(-2, -1)) - diagonal.sum(axis=-1)).mean() / 20.0),
        "mean_normalized_attention_entropy": float(normalized_entropy.mean()),
        "residual_scale": float(block.residual_scale.detach().cpu()),
        "mean_relative_transformer_correction_norm": float(
            np.mean(correction_norm / np.maximum(raw_norm, 1e-12))
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "04_attention_map.json"
    png_path = output_dir / "04_attention_map.png"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for head, axis in enumerate(axes.reshape(-1)):
        image = axis.imshow(mean_by_head[head], vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(f"Head {head}")
        axis.set_xlabel("Key agent")
        axis.set_ylabel("Query agent")
        axis.set_xticks(range(5))
        axis.set_yticks(range(5))
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    figure.suptitle("Post-LSTM attention: 5 simultaneous agents (not temporal history)")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    print(json.dumps({
        "artifact": str(json_path),
        "heatmap": str(png_path),
        "temporal_attention_available": False,
        "mean_diagonal_attention": payload["mean_diagonal_attention"],
        "mean_normalized_attention_entropy": payload["mean_normalized_attention_entropy"],
        "mean_relative_transformer_correction_norm": payload["mean_relative_transformer_correction_norm"],
    }, ensure_ascii=False, indent=2))
    return json_path


def _owner_entry_labels(sampled: dict[str, np.ndarray]) -> np.ndarray:
    return (_owner_action_labels(sampled) != 0).astype(np.int64)


def _owner_action_labels(sampled: dict[str, np.ndarray]) -> np.ndarray:
    actions = sampled["actions"]
    masks = sampled["owner_masks"]
    labels = [
        actions[:, step][masks[:, step]].astype(np.int64)
        for step in range(actions.shape[1])
    ]
    return np.concatenate(labels)


def _owner_agent_ids(sampled: dict[str, np.ndarray]) -> np.ndarray:
    masks = sampled["owner_masks"]
    ids = np.broadcast_to(
        np.arange(FIVE_AGENT_COUNT, dtype=np.int64), masks[:, 0].shape
    )
    rows = [ids[masks[:, step]] for step in range(masks.shape[1])]
    return np.concatenate(rows)


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "entry_precision": float(precision_score(labels, predictions, zero_division=0)),
        "entry_recall": float(recall_score(labels, predictions, zero_division=0)),
        "entry_f1": float(f1_score(labels, predictions, zero_division=0)),
        "label_entry_rate": float(np.mean(labels)),
        "predicted_entry_rate": float(np.mean(predictions)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    )
    return result


def _threshold_sweep(
    validation_labels: np.ndarray,
    validation_margins: np.ndarray,
    *,
    threshold_count: int = 401,
) -> dict[str, object]:
    labels = np.asarray(validation_labels, dtype=np.int64)
    margins = np.asarray(validation_margins, dtype=np.float64)
    if len(labels) != len(margins) or len(np.unique(labels)) != 2:
        raise ValueError("threshold sweep requires aligned two-class validation data")
    finite = margins[np.isfinite(margins)]
    if len(finite) != len(margins):
        raise ValueError("threshold sweep margins must be finite")
    quantiles = np.linspace(0.0, 1.0, num=max(3, int(threshold_count)))
    thresholds = np.unique(np.r_[0.0, np.quantile(finite, quantiles)])
    rows: list[dict[str, float]] = []
    best: tuple[tuple[float, float, float, float], dict[str, float]] | None = None
    for threshold in thresholds:
        predictions = (margins >= float(threshold)).astype(np.int64)
        metrics = _binary_metrics(labels, predictions, margins)
        row = {"threshold": float(threshold), **metrics}
        rows.append(row)
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["mcc"]),
            float(metrics["entry_f1"]),
            -abs(float(threshold)),
        )
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return {
        "selection_metric": "validation_balanced_accuracy_then_mcc_then_f1",
        "candidate_count": int(len(rows)),
        "selected_threshold": float(best[1]["threshold"]),
        "selected_validation_metrics": best[1],
        "curve": rows,
    }


def _margin_metrics(labels: np.ndarray, margins: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    margins = np.asarray(margins, dtype=np.float64)
    predictions = (margins >= float(threshold)).astype(np.int64)
    result = _binary_metrics(labels, predictions, margins)
    result["average_precision"] = float(average_precision_score(labels, margins))
    result["threshold"] = float(threshold)
    return result


def _fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, object]:
    candidates = (0.001, 0.01, 0.1, 1.0)
    best: tuple[float, float, object] | None = None
    tuning: list[dict[str, float]] = []
    for c_value in candidates:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=0,
            ),
        )
        model.fit(train_x, train_y)
        predictions = model.predict(validation_x)
        scores = model.predict_proba(validation_x)[:, 1]
        metrics = _binary_metrics(validation_y, predictions, scores)
        tuning.append({"C": c_value, **metrics})
        key = (metrics["balanced_accuracy"], metrics["mcc"])
        if best is None or key > best[:2]:
            best = (key[0], key[1], model)
    assert best is not None
    selected_model = best[2]
    selected_c = float(
        selected_model.named_steps["logisticregression"].C
    )
    evaluations: dict[str, dict[str, float]] = {}
    for split, features, labels in (
        ("train", train_x, train_y),
        ("validation", validation_x, validation_y),
        ("test", test_x, test_y),
    ):
        predictions = selected_model.predict(features)
        scores = selected_model.predict_proba(features)[:, 1]
        evaluations[split] = _binary_metrics(labels, predictions, scores)
    return {
        "selected_C": selected_c,
        "selection_metric": "validation_balanced_accuracy_then_mcc",
        "validation_tuning": tuning,
        "metrics": evaluations,
    }


def run_linear_probe(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
) -> Path:
    network = _load_network(checkpoint, config_path)
    feature_sets: dict[str, dict[str, np.ndarray]] = {
        "raw_lstm_hidden": {}, "post_transformer_hidden": {},
    }
    labels_by_split: dict[str, np.ndarray] = {}
    times_by_split: dict[str, tuple[str, str]] = {}
    for split in ("train", "validation", "test"):
        sampled = _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        raw_hidden, transformed_hidden, _ = _replay_hidden(network, sampled)
        feature_sets["raw_lstm_hidden"][split] = raw_hidden
        feature_sets["post_transformer_hidden"][split] = transformed_hidden
        labels_by_split[split] = _owner_entry_labels(sampled)
        flat_times = sampled["sample_times"].reshape(-1)
        times_by_split[split] = (str(flat_times[0]), str(flat_times[-1]))
        if len(labels_by_split[split]) != len(raw_hidden):
            raise RuntimeError("feature/label owner order khong khop")
    probes = {}
    for name, features in feature_sets.items():
        probes[name] = _fit_linear_probe(
            features["train"], labels_by_split["train"],
            features["validation"], labels_by_split["validation"],
            features["test"], labels_by_split["test"],
        )
    raw_test = probes["raw_lstm_hidden"]["metrics"]["test"]
    post_test = probes["post_transformer_hidden"]["metrics"]["test"]
    payload = {
        "stage": 5,
        "name": "linear_probe_transformer_feature_quality",
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "target": "teacher_entry_vs_hold",
        "chronological_split_contract": True,
        "window_count_per_split": windows,
        "sequence_length": sequence_length,
        "sample_counts": {split: int(len(labels)) for split, labels in labels_by_split.items()},
        "time_ranges": times_by_split,
        "probes": probes,
        "test_delta_post_minus_raw": {
            key: float(post_test[key] - raw_test[key])
            for key in ("balanced_accuracy", "mcc", "entry_precision", "entry_recall", "entry_f1", "roc_auc")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "05_linear_probe.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(json_path),
        "sample_counts": payload["sample_counts"],
        "raw_test": raw_test,
        "post_test": post_test,
        "test_delta_post_minus_raw": payload["test_delta_post_minus_raw"],
    }, ensure_ascii=False, indent=2))
    return json_path


def run_ablation_status(
    checkpoint: Path,
    output_dir: Path,
    *,
    active_run_dir: Path | None,
    active_process_id: int | None,
) -> Path:
    existing_runner = ROOT / "scripts" / "run_mt5_ablation_v193.py"
    architecture_runner = ROOT / "scripts" / "run_mt5_transformer_ablation.py"
    runner_text = existing_runner.read_text(encoding="utf-8")
    runner_declares_non_architecture_ablation = '"architecture_ablation": False' in runner_text
    status_path = active_run_dir / "events.jsonl.status.json" if active_run_dir else None
    diagnostics_path = active_run_dir / "mt5_diagnostics.json" if active_run_dir else None
    active_status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path is not None and status_path.is_file() else {}
    )
    diagnostics_available = bool(diagnostics_path is not None and diagnostics_path.is_file())
    tester_busy = active_process_id is not None
    payload = {
        "stage": 6,
        "name": "transformer_reward_generalization_ablation_gate",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "matched_transformer_on_off_ablation_completed": False,
        "reward_generalization_conclusion": "unproven",
        "reason": (
            (
                "MT5 hien dang chay mot arm don tren checkpoint nay; khong chen them tester."
                if tester_busy else
                "Khong co MT5 tester dang chay. Runner Transformer ON/identity da san sang, "
                "nhung chua co paired run hoan tat de ket luan PnL."
            )
        ),
        "tester_busy": tester_busy,
        "environment_ready_for_sequential_architecture_ablation": not tester_busy,
        "existing_runner": str(existing_runner),
        "existing_runner_declares_architecture_ablation_false": (
            runner_declares_non_architecture_ablation
        ),
        "architecture_ablation_runner": str(architecture_runner),
        "architecture_ablation_runner_available": architecture_runner.is_file(),
        "active_process_id": active_process_id,
        "active_run_dir": str(active_run_dir) if active_run_dir else "",
        "active_run_status": active_status,
        "active_run_diagnostics_available": diagnostics_available,
        "required_next_ablation": {
            "arms": ["transformer_on", "transformer_identity_residual_scale_0"],
            "policy_learning": "frozen",
            "same_checkpoint_except_ablation": True,
            "same_mt5_dates_model_seed_spread": True,
            "minimum_outputs": [
                "net_profit", "profit_factor", "expectancy", "win_rate",
                "max_equity_drawdown", "closed_trades",
            ],
            "generalization": "chronological OOS after imitation train window",
            "execution_plan": "one_day_fail_fast_smoke_before_longer_window",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "06_ablation_status.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(json_path),
        "matched_transformer_on_off_ablation_completed": False,
        "reward_generalization_conclusion": "unproven",
        "active_process_id": active_process_id,
        "active_run_diagnostics_available": diagnostics_available,
    }, ensure_ascii=False, indent=2))
    return json_path


def _centroid_radius(values: np.ndarray) -> tuple[np.ndarray, float]:
    values64 = np.asarray(values, dtype=np.float64)
    centroid = values64.mean(axis=0)
    mean_square_norm = float(np.mean(np.einsum("ij,ij->i", values64, values64)))
    radius_square = max(0.0, mean_square_norm - float(np.dot(centroid, centroid)))
    return centroid, float(np.sqrt(radius_square))


def _stratified_balanced_accuracy_ci(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    class_ids: tuple[int, ...],
    bootstraps: int,
    seed: int = 0,
) -> list[float]:
    if bootstraps <= 0:
        score = float(balanced_accuracy_score(labels, predictions))
        return [score, score]
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(labels == class_id) for class_id in class_ids]
    if any(len(row) == 0 for row in indices):
        return [float("nan"), float("nan")]
    scores = np.empty(bootstraps, dtype=np.float64)
    for index in range(bootstraps):
        selected = np.concatenate([
            rng.choice(row, size=len(row), replace=True) for row in indices
        ])
        scores[index] = balanced_accuracy_score(
            labels[selected], predictions[selected]
        )
    return [float(value) for value in np.percentile(scores, (2.5, 97.5))]


def _nearest_centroid_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    class_ids: tuple[int, ...],
) -> np.ndarray:
    centroids = np.stack([
        np.asarray(train_x[train_y == class_id], dtype=np.float64).mean(axis=0)
        for class_id in class_ids
    ])
    test64 = test_x.astype(np.float64, copy=False)
    distance_square = (
        np.square(test64).sum(axis=1, keepdims=True)
        - 2.0 * test64 @ centroids.T
        + np.square(centroids).sum(axis=1)[None, :]
    )
    return np.asarray(class_ids, dtype=np.int64)[np.argmin(distance_square, axis=1)]


def _paired_balanced_accuracy_delta_ci(
    labels: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    *,
    class_ids: tuple[int, ...],
    bootstraps: int,
    seed: int = 0,
) -> list[float]:
    delta = float(
        balanced_accuracy_score(labels, right_predictions)
        - balanced_accuracy_score(labels, left_predictions)
    )
    if bootstraps <= 0:
        return [delta, delta]
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(labels == class_id) for class_id in class_ids]
    deltas = np.empty(bootstraps, dtype=np.float64)
    for index in range(bootstraps):
        selected = np.concatenate([
            rng.choice(row, size=len(row), replace=True) for row in indices
        ])
        deltas[index] = (
            balanced_accuracy_score(labels[selected], right_predictions[selected])
            - balanced_accuracy_score(labels[selected], left_predictions[selected])
        )
    return [float(value) for value in np.percentile(deltas, (2.5, 97.5))]


def _class_geometry(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    class_ids: tuple[int, ...],
    class_names: dict[int, str],
    bootstraps: int,
) -> dict[str, object]:
    missing_train = [class_id for class_id in class_ids if not np.any(train_y == class_id)]
    missing_test = [class_id for class_id in class_ids if not np.any(test_y == class_id)]
    if missing_train or missing_test:
        raise ValueError(
            f"class geometry thieu lop train={missing_train}, test={missing_test}"
        )

    train_centroids: dict[int, np.ndarray] = {}
    test_centroids: dict[int, np.ndarray] = {}
    train_radii: dict[int, float] = {}
    test_radii: dict[int, float] = {}
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id in class_ids:
        train_values = train_x[train_y == class_id]
        test_values = test_x[test_y == class_id]
        train_centroid, train_radius = _centroid_radius(train_values)
        test_centroid, test_radius = _centroid_radius(test_values)
        train_centroids[class_id] = train_centroid
        test_centroids[class_id] = test_centroid
        train_radii[class_id] = train_radius
        test_radii[class_id] = test_radius
        drift = float(np.linalg.norm(test_centroid - train_centroid))
        per_class[class_names[class_id]] = {
            "train_count": int(len(train_values)),
            "test_count": int(len(test_values)),
            "train_rms_radius": train_radius,
            "test_rms_radius": test_radius,
            "train_to_test_centroid_drift": drift,
            "drift_over_train_radius": drift / max(train_radius, 1e-12),
        }

    pairwise: dict[str, dict[str, float]] = {}
    for left_index, left in enumerate(class_ids):
        for right in class_ids[left_index + 1:]:
            name = f"{class_names[left]}__vs__{class_names[right]}"
            train_distance = float(np.linalg.norm(train_centroids[left] - train_centroids[right]))
            test_distance = float(np.linalg.norm(test_centroids[left] - test_centroids[right]))
            train_pooled_radius = float(np.sqrt(
                (train_radii[left] ** 2 + train_radii[right] ** 2) / 2.0
            ))
            test_pooled_radius = float(np.sqrt(
                (test_radii[left] ** 2 + test_radii[right] ** 2) / 2.0
            ))
            pairwise[name] = {
                "train_centroid_distance": train_distance,
                "test_centroid_distance": test_distance,
                "train_separation_over_pooled_radius": (
                    train_distance / max(train_pooled_radius, 1e-12)
                ),
                "test_separation_over_pooled_radius": (
                    test_distance / max(test_pooled_radius, 1e-12)
                ),
                "test_over_train_centroid_distance": (
                    test_distance / max(train_distance, 1e-12)
                ),
            }

    train_counts = np.asarray([np.sum(train_y == value) for value in class_ids], dtype=np.float64)
    weights = train_counts / train_counts.sum()
    global_centroid = np.sum(
        np.stack([train_centroids[value] for value in class_ids]) * weights[:, None], axis=0
    )
    between_weighted = float(np.sum([
        weight * np.square(train_centroids[value] - global_centroid).sum()
        for weight, value in zip(weights, class_ids)
    ]))
    within_weighted = float(np.sum([
        weight * train_radii[value] ** 2 for weight, value in zip(weights, class_ids)
    ]))
    macro_centroid = np.mean(
        np.stack([train_centroids[value] for value in class_ids]), axis=0
    )
    between_macro = float(np.mean([
        np.square(train_centroids[value] - macro_centroid).sum()
        for value in class_ids
    ]))
    within_macro = float(np.mean([train_radii[value] ** 2 for value in class_ids]))

    predictions = _nearest_centroid_predictions(
        train_x, train_y, test_x, class_ids=class_ids
    )
    balanced_accuracy = float(balanced_accuracy_score(test_y, predictions))
    matrix = confusion_matrix(test_y, predictions, labels=list(class_ids))
    per_class_recall = {
        class_names[class_id]: float(
            matrix[index, index] / max(matrix[index].sum(), 1)
        )
        for index, class_id in enumerate(class_ids)
    }
    return {
        "coordinate_system": "native_hidden_coordinates",
        "distance": "euclidean",
        "radius": "root_mean_squared_distance_to_class_centroid",
        "per_class": per_class,
        "pairwise": pairwise,
        "train_scatter": {
            "frequency_weighted_within_trace": within_weighted,
            "frequency_weighted_between_trace": between_weighted,
            "frequency_weighted_fisher_trace_ratio": (
                between_weighted / max(within_weighted, 1e-30)
            ),
            "macro_within_trace": within_macro,
            "macro_between_trace": between_macro,
            "macro_fisher_trace_ratio": between_macro / max(within_macro, 1e-30),
        },
        "nearest_train_centroid_test": {
            "balanced_accuracy": balanced_accuracy,
            "mcc": float(matthews_corrcoef(test_y, predictions)),
            "class_order": [class_names[value] for value in class_ids],
            "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
            "per_class_recall": per_class_recall,
            "balanced_accuracy_bootstrap_95pct_ci": _stratified_balanced_accuracy_ci(
                test_y,
                predictions,
                class_ids=class_ids,
                bootstraps=bootstraps,
            ),
            "bootstrap_method": "stratified_resampling_with_fixed_train_centroids",
            "bootstrap_replicates": int(bootstraps),
        },
    }


def run_class_conditional_geometry(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    bootstraps: int,
) -> Path:
    network = _load_network(checkpoint, config_path)
    features: dict[str, dict[str, np.ndarray]] = {
        "raw_lstm_hidden": {}, "post_transformer_hidden": {},
    }
    labels: dict[str, np.ndarray] = {}
    agents: dict[str, np.ndarray] = {}
    times: dict[str, tuple[str, str]] = {}
    for split in ("train", "test"):
        sampled = _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        raw_hidden, transformed_hidden, _ = _replay_hidden(network, sampled)
        features["raw_lstm_hidden"][split] = raw_hidden
        features["post_transformer_hidden"][split] = transformed_hidden
        labels[split] = _owner_action_labels(sampled)
        agents[split] = _owner_agent_ids(sampled)
        flat_times = sampled["sample_times"].reshape(-1)
        times[split] = (str(flat_times[0]), str(flat_times[-1]))
        if not (len(raw_hidden) == len(labels[split]) == len(agents[split])):
            raise RuntimeError("feature/action/agent owner order khong khop")

    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    class_agent_support = {
        action_names[action]: sorted({
            int(value) for split in ("train", "test")
            for value in agents[split][labels[split] == action]
        })
        for action in (0, 1, 2)
    }
    geometries: dict[str, object] = {}
    for representation, split_features in features.items():
        pooled = _class_geometry(
            split_features["train"], labels["train"],
            split_features["test"], labels["test"],
            class_ids=(0, 1, 2), class_names=action_names, bootstraps=bootstraps,
        )
        by_role = {}
        for agent_id, entry_action in ((0, 1), (1, 2)):
            train_mask = agents["train"] == agent_id
            test_mask = agents["test"] == agent_id
            by_role[f"agent_{agent_id}_{action_names[entry_action].lower()}_role"] = _class_geometry(
                split_features["train"][train_mask], labels["train"][train_mask],
                split_features["test"][test_mask], labels["test"][test_mask],
                class_ids=(0, entry_action), class_names=action_names, bootstraps=bootstraps,
            )
        geometries[representation] = {
            "pooled_three_action_classes": pooled,
            "role_conditioned_binary": by_role,
        }

    raw = geometries["raw_lstm_hidden"]
    post = geometries["post_transformer_hidden"]
    deltas: dict[str, float] = {
        "pooled_macro_fisher_trace_ratio": float(
            post["pooled_three_action_classes"]["train_scatter"]["macro_fisher_trace_ratio"]
            - raw["pooled_three_action_classes"]["train_scatter"]["macro_fisher_trace_ratio"]
        ),
        "pooled_nearest_centroid_test_balanced_accuracy": float(
            post["pooled_three_action_classes"]["nearest_train_centroid_test"]["balanced_accuracy"]
            - raw["pooled_three_action_classes"]["nearest_train_centroid_test"]["balanced_accuracy"]
        ),
    }
    for role_name in ("agent_0_buy_role", "agent_1_sell_role"):
        raw_role = raw["role_conditioned_binary"][role_name]
        post_role = post["role_conditioned_binary"][role_name]
        pair_name = "HOLD__vs__BUY" if role_name == "agent_0_buy_role" else "HOLD__vs__SELL"
        deltas[f"{role_name}_train_separation_over_pooled_radius"] = float(
            post_role["pairwise"][pair_name]["train_separation_over_pooled_radius"]
            - raw_role["pairwise"][pair_name]["train_separation_over_pooled_radius"]
        )
        deltas[f"{role_name}_test_nearest_centroid_balanced_accuracy"] = float(
            post_role["nearest_train_centroid_test"]["balanced_accuracy"]
            - raw_role["nearest_train_centroid_test"]["balanced_accuracy"]
        )

    paired_delta_intervals: dict[str, list[float]] = {}
    for scope_name, class_ids, agent_id in (
        ("pooled_three_action_classes", (0, 1, 2), None),
        ("agent_0_buy_role", (0, 1), 0),
        ("agent_1_sell_role", (0, 2), 1),
    ):
        train_mask = np.ones(len(labels["train"]), dtype=bool) if agent_id is None else agents["train"] == agent_id
        test_mask = np.ones(len(labels["test"]), dtype=bool) if agent_id is None else agents["test"] == agent_id
        raw_predictions = _nearest_centroid_predictions(
            features["raw_lstm_hidden"]["train"][train_mask], labels["train"][train_mask],
            features["raw_lstm_hidden"]["test"][test_mask], class_ids=class_ids,
        )
        post_predictions = _nearest_centroid_predictions(
            features["post_transformer_hidden"]["train"][train_mask], labels["train"][train_mask],
            features["post_transformer_hidden"]["test"][test_mask], class_ids=class_ids,
        )
        paired_delta_intervals[scope_name] = _paired_balanced_accuracy_delta_ci(
            labels["test"][test_mask], raw_predictions, post_predictions,
            class_ids=class_ids, bootstraps=bootstraps,
        )

    payload = {
        "stage": 7,
        "name": "class_conditional_hidden_state_geometry",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "chronological_split_contract": True,
        "centroids_fit_on_train_only": True,
        "test_used_for_evaluation_only": True,
        "window_count_per_split": int(windows),
        "sequence_length": int(sequence_length),
        "time_ranges": times,
        "class_agent_support": class_agent_support,
        "pooled_three_class_role_confounded": (
            class_agent_support["BUY"] != class_agent_support["SELL"]
        ),
        "primary_inference_scope": [
            "agent_0: HOLD_vs_BUY", "agent_1: HOLD_vs_SELL"
        ],
        "representations": geometries,
        "post_transformer_minus_raw_deltas": deltas,
        "paired_bootstrap_95pct_ci_for_test_balanced_accuracy_delta": paired_delta_intervals,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "07_class_conditional_geometry.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {0: "#808080", 1: "#00a65a", 2: "#d64541"}
    for row, representation in enumerate(("raw_lstm_hidden", "post_transformer_hidden")):
        pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
        pca.fit(features[representation]["train"])
        projected = pca.transform(features[representation]["test"])
        for column, (agent_id, entry_action) in enumerate(((0, 1), (1, 2))):
            axis = axes[row, column]
            role_mask = agents["test"] == agent_id
            for action in (0, entry_action):
                mask = role_mask & (labels["test"] == action)
                axis.scatter(
                    projected[mask, 0], projected[mask, 1], s=10,
                    alpha=0.35 if action == 0 else 0.8,
                    color=colors[action], label=action_names[action], rasterized=True,
                )
            variance = 100.0 * float(np.sum(pca.explained_variance_ratio_))
            axis.set_title(f"{representation} | agent {agent_id} | PCA2 {variance:.1f}%")
            axis.set_xlabel("PC1 fitted on train")
            axis.set_ylabel("PC2 fitted on train")
            axis.legend()
    png_path = output_dir / "07_class_conditional_geometry.png"
    figure.suptitle("Test hidden-state geometry conditioned on fixed execution role")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    print(json.dumps({
        "artifact": str(json_path),
        "plot": str(png_path),
        "class_agent_support": class_agent_support,
        "pooled_three_class_role_confounded": payload["pooled_three_class_role_confounded"],
        "post_transformer_minus_raw_deltas": deltas,
    }, ensure_ascii=False, indent=2))
    return json_path


def _pca_probe_saturation_dimension(
    rows: list[dict[str, object]],
    *,
    fraction_of_gain: float = 0.95,
) -> int | None:
    if not rows:
        return None
    best = max(float(row["validation"]["balanced_accuracy"]) for row in rows)
    if best <= 0.5:
        return None
    threshold = 0.5 + fraction_of_gain * (best - 0.5)
    eligible = [
        int(row["components"])
        for row in rows
        if float(row["validation"]["balanced_accuracy"]) >= threshold
    ]
    return min(eligible) if eligible else None


def _balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    weights = {
        int(value): len(labels) / (len(unique) * int(count))
        for value, count in zip(unique, counts)
    }
    return np.asarray([weights[int(value)] for value in labels], dtype=np.float64)


def _classifier_evaluation(
    model: object,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = np.asarray(model.predict(features), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    entry_column = int(np.flatnonzero(classes != 0)[0])
    scores = probabilities[:, entry_column]
    return _binary_metrics((labels != 0).astype(np.int64), (predictions != 0).astype(np.int64), scores), predictions


def _fit_pca_linear_probe_curve(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    component_grid: tuple[int, ...],
) -> tuple[dict[str, object], np.ndarray]:
    rows: list[dict[str, object]] = []
    best_global: tuple[float, float, int, object] | None = None
    c_grid = (0.001, 0.01, 0.1, 1.0)
    for components in component_grid:
        best: tuple[float, float, object, float] | None = None
        tuning = []
        for c_value in c_grid:
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=0,
            )
            model.fit(train_x[:, :components], train_y)
            validation_metrics, _ = _classifier_evaluation(
                model, validation_x[:, :components], validation_y
            )
            tuning.append({"C": c_value, **validation_metrics})
            key = (
                validation_metrics["balanced_accuracy"],
                validation_metrics["mcc"],
            )
            if best is None or key > best[:2]:
                best = (key[0], key[1], model, c_value)
        assert best is not None
        selected = best[2]
        train_metrics, _ = _classifier_evaluation(
            selected, train_x[:, :components], train_y
        )
        validation_metrics, _ = _classifier_evaluation(
            selected, validation_x[:, :components], validation_y
        )
        test_metrics, test_predictions = _classifier_evaluation(
            selected, test_x[:, :components], test_y
        )
        rows.append({
            "components": int(components),
            "selected_C": float(best[3]),
            "validation_tuning": tuning,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        })
        global_key = (
            validation_metrics["balanced_accuracy"],
            validation_metrics["mcc"],
            -int(components),
        )
        if best_global is None or global_key > best_global[:3]:
            best_global = (global_key[0], global_key[1], global_key[2], test_predictions)
    assert best_global is not None
    selected_components = -int(best_global[2])
    selected_row = next(row for row in rows if row["components"] == selected_components)
    return {
        "selection_rule": "validation_balanced_accuracy_then_mcc_then_fewer_components",
        "saturation_rule": "smallest_k_reaching_95pct_of_best_validation_gain_above_chance",
        "saturation_components": _pca_probe_saturation_dimension(rows),
        "selected_components": selected_components,
        "selected_test_metrics": selected_row["test"],
        "curve": rows,
    }, np.asarray(best_global[3], dtype=np.int64)


def _fit_lda_fisher_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    model = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto", priors=np.asarray([0.5, 0.5])
    )
    model.fit(train_x, train_y)
    train_metrics, _ = _classifier_evaluation(model, train_x, train_y)
    validation_metrics, _ = _classifier_evaluation(model, validation_x, validation_y)
    test_metrics, predictions = _classifier_evaluation(model, test_x, test_y)
    direction = np.asarray(model.coef_[0], dtype=np.float64)
    projection = np.asarray(train_x, dtype=np.float64) @ direction
    classes = np.asarray(model.classes_, dtype=np.int64)
    first = projection[train_y == classes[0]]
    second = projection[train_y == classes[1]]
    fisher_ratio = float(
        np.square(first.mean() - second.mean())
        / max(float(first.var(ddof=1) + second.var(ddof=1)), 1e-30)
    )
    return {
        "components": int(train_x.shape[1]),
        "solver": "lsqr",
        "shrinkage": "auto",
        "balanced_priors": True,
        "train_projected_fisher_ratio": fisher_ratio,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }, predictions


def _fit_nonlinear_probes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    probes: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    sample_weight = _balanced_sample_weight(train_y)

    mlp_candidates = (0.0001, 0.001, 0.01)
    best_mlp: tuple[float, float, object, float, int, int] | None = None
    mlp_tuning = []
    for alpha in mlp_candidates:
        model = MLPClassifier(
            hidden_layer_sizes=(32,), activation="relu", solver="adam",
            alpha=alpha, batch_size=64, learning_rate_init=0.001,
            max_iter=1, shuffle=True, random_state=0,
            early_stopping=False, tol=0.0,
        )
        best_epoch_key: tuple[float, float] | None = None
        best_epoch_state: tuple[list[np.ndarray], list[np.ndarray], int] | None = None
        stale_epochs = 0
        epochs_run = 0
        for epoch in range(1, 301):
            model.partial_fit(
                train_x, train_y, classes=np.unique(train_y), sample_weight=sample_weight
            )
            epochs_run = epoch
            validation_metrics, _ = _classifier_evaluation(
                model, validation_x, validation_y
            )
            epoch_key = (
                validation_metrics["balanced_accuracy"], validation_metrics["mcc"]
            )
            if best_epoch_key is None or epoch_key > best_epoch_key:
                best_epoch_key = epoch_key
                best_epoch_state = (
                    [value.copy() for value in model.coefs_],
                    [value.copy() for value in model.intercepts_],
                    epoch,
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 25:
                break
        assert best_epoch_key is not None and best_epoch_state is not None
        model.coefs_ = best_epoch_state[0]
        model.intercepts_ = best_epoch_state[1]
        validation_metrics, _ = _classifier_evaluation(model, validation_x, validation_y)
        mlp_tuning.append({
            "alpha": alpha,
            "best_epoch": int(best_epoch_state[2]),
            "epochs_run": int(epochs_run),
            "stopped_early": bool(epochs_run < 300),
            **validation_metrics,
        })
        key = best_epoch_key
        if best_mlp is None or key > best_mlp[:2]:
            best_mlp = (
                key[0], key[1], model, alpha,
                int(best_epoch_state[2]), int(epochs_run),
            )
    assert best_mlp is not None
    mlp = best_mlp[2]
    mlp_test, predictions["mlp"] = _classifier_evaluation(mlp, test_x, test_y)
    probes["mlp"] = {
        "architecture": [int(train_x.shape[1]), 32, 2],
        "selected_alpha": float(best_mlp[3]),
        "selected_best_epoch": int(best_mlp[4]),
        "selected_epochs_run": int(best_mlp[5]),
        "early_stopping": "validation_balanced_accuracy_then_mcc_patience_25",
        "validation_tuning": mlp_tuning,
        "train": _classifier_evaluation(mlp, train_x, train_y)[0],
        "validation": _classifier_evaluation(mlp, validation_x, validation_y)[0],
        "test": mlp_test,
    }

    k_candidates = tuple(value for value in (5, 15, 31, 63) if value < len(train_y))
    best_knn: tuple[float, float, object, int] | None = None
    knn_tuning = []
    for neighbors in k_candidates:
        model = KNeighborsClassifier(
            n_neighbors=neighbors, weights="distance", metric="euclidean", n_jobs=-1
        )
        model.fit(train_x, train_y)
        validation_metrics, _ = _classifier_evaluation(
            model, validation_x, validation_y
        )
        knn_tuning.append({"neighbors": neighbors, **validation_metrics})
        key = (validation_metrics["balanced_accuracy"], validation_metrics["mcc"])
        if best_knn is None or key > best_knn[:2]:
            best_knn = (key[0], key[1], model, neighbors)
    assert best_knn is not None
    knn = best_knn[2]
    knn_test, predictions["knn"] = _classifier_evaluation(knn, test_x, test_y)
    probes["knn"] = {
        "selected_neighbors": int(best_knn[3]),
        "weights": "distance",
        "validation_tuning": knn_tuning,
        "train": _classifier_evaluation(knn, train_x, train_y)[0],
        "validation": _classifier_evaluation(knn, validation_x, validation_y)[0],
        "test": knn_test,
    }
    return probes, predictions


def run_priority_one_probe_suite(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    max_pca_components: int,
    bootstraps: int,
) -> Path:
    network = _load_network(checkpoint, config_path)
    features: dict[str, dict[str, np.ndarray]] = {
        "raw_lstm_hidden": {}, "post_transformer_hidden": {},
    }
    labels: dict[str, np.ndarray] = {}
    agents: dict[str, np.ndarray] = {}
    times: dict[str, tuple[str, str]] = {}
    for split in ("train", "validation", "test"):
        sampled = _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        raw_hidden, transformed_hidden, _ = _replay_hidden(network, sampled)
        features["raw_lstm_hidden"][split] = raw_hidden
        features["post_transformer_hidden"][split] = transformed_hidden
        labels[split] = _owner_action_labels(sampled)
        agents[split] = _owner_agent_ids(sampled)
        flat_times = sampled["sample_times"].reshape(-1)
        times[split] = (str(flat_times[0]), str(flat_times[-1]))

    payload_representations: dict[str, object] = {}
    plot_rows: list[tuple[str, str, list[dict[str, object]]]] = []
    for representation, split_features in features.items():
        max_components = min(
            int(max_pca_components), len(split_features["train"]) - 1,
            split_features["train"].shape[1],
        )
        pca = PCA(n_components=max_components, svd_solver="randomized", random_state=0)
        pca_train = pca.fit_transform(split_features["train"])
        pca_validation = pca.transform(split_features["validation"])
        pca_test = pca.transform(split_features["test"])
        scaler = StandardScaler()
        standardized = {
            "train": scaler.fit_transform(pca_train),
            "validation": scaler.transform(pca_validation),
            "test": scaler.transform(pca_test),
        }
        component_grid = tuple(
            value for value in (1, 2, 4, 8, 16, 32, 64, 128, 256)
            if value <= max_components
        )
        if max_components not in component_grid:
            component_grid = tuple(sorted(set((*component_grid, max_components))))
        roles: dict[str, object] = {}
        for role_name, agent_id, entry_action in (
            ("agent_0_buy_role", 0, 1),
            ("agent_1_sell_role", 1, 2),
        ):
            masks = {
                split: agents[split] == agent_id
                for split in ("train", "validation", "test")
            }
            x = {
                split: standardized[split][masks[split]]
                for split in ("train", "validation", "test")
            }
            y = {
                split: labels[split][masks[split]]
                for split in ("train", "validation", "test")
            }
            linear_curve, linear_predictions = _fit_pca_linear_probe_curve(
                x["train"], y["train"], x["validation"], y["validation"],
                x["test"], y["test"], component_grid=component_grid,
            )
            lda, _ = _fit_lda_fisher_probe(
                x["train"], y["train"], x["validation"], y["validation"],
                x["test"], y["test"],
            )
            nonlinear, nonlinear_predictions = _fit_nonlinear_probes(
                x["train"], y["train"], x["validation"], y["validation"],
                x["test"], y["test"],
            )
            delta_intervals = {
                name: _paired_balanced_accuracy_delta_ci(
                    y["test"], linear_predictions, prediction,
                    class_ids=(0, entry_action), bootstraps=bootstraps,
                )
                for name, prediction in nonlinear_predictions.items()
            }
            roles[role_name] = {
                "entry_action": int(entry_action),
                "sample_counts": {
                    split: {
                        "total": int(len(y[split])),
                        "hold": int(np.sum(y[split] == 0)),
                        "entry": int(np.sum(y[split] == entry_action)),
                    }
                    for split in ("train", "validation", "test")
                },
                "pca_linear_probe_curve": linear_curve,
                "lda_fisher": lda,
                "nonlinear_probes": nonlinear,
                "nonlinear_minus_selected_linear_test_ba_paired_bootstrap_95pct_ci": delta_intervals,
            }
            plot_rows.append((representation, role_name, linear_curve["curve"]))
        payload_representations[representation] = {
            "pca_fit_scope": "all_train_owner_hidden",
            "pca_components": int(max_components),
            "pca_variance_captured": float(np.sum(pca.explained_variance_ratio_)),
            "pca_cumulative_variance_at_components": {
                str(value): float(np.sum(pca.explained_variance_ratio_[:value]))
                for value in component_grid
            },
            "standardization_fit_scope": "pca_train_only",
            "roles": roles,
        }

    payload = {
        "stage": 8,
        "name": "priority_one_decision_signal_probe_suite",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "chronological_split_contract": True,
        "hyperparameters_selected_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "role_conditioned": True,
        "window_count_per_split": int(windows),
        "sequence_length": int(sequence_length),
        "time_ranges": times,
        "representations": payload_representations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "08_priority_one_probe_suite.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (representation, role_name, rows) in zip(axes.reshape(-1), plot_rows):
        components = [int(row["components"]) for row in rows]
        validation_scores = [float(row["validation"]["balanced_accuracy"]) for row in rows]
        test_scores = [float(row["test"]["balanced_accuracy"]) for row in rows]
        axis.plot(components, validation_scores, marker="o", label="validation")
        axis.plot(components, test_scores, marker="o", label="test OOS")
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
        axis.set_xscale("log", base=2)
        axis.set_xticks(components, [str(value) for value in components])
        axis.set_ylim(0.45, 0.75)
        axis.set_title(f"{representation} | {role_name}")
        axis.set_xlabel("Leading PCA components")
        axis.set_ylabel("Balanced accuracy")
        axis.grid(alpha=0.25)
        axis.legend()
    png_path = output_dir / "08_pca_probe_curves.png"
    figure.suptitle("Decision signal dimensionality (PCA fitted on train only)")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    print(json.dumps({
        "artifact": str(json_path),
        "plot": str(png_path),
        "representations": {
            name: {
                role: {
                    "saturation_components": data["pca_linear_probe_curve"]["saturation_components"],
                    "selected_components": data["pca_linear_probe_curve"]["selected_components"],
                    "linear_test_ba": data["pca_linear_probe_curve"]["selected_test_metrics"]["balanced_accuracy"],
                    "lda_test_ba": data["lda_fisher"]["test"]["balanced_accuracy"],
                    "mlp_test_ba": data["nonlinear_probes"]["mlp"]["test"]["balanced_accuracy"],
                    "knn_test_ba": data["nonlinear_probes"]["knn"]["test"]["balanced_accuracy"],
                }
                for role, data in value["roles"].items()
            }
            for name, value in payload_representations.items()
        },
    }, ensure_ascii=False, indent=2))
    return json_path


def run_temporal_lag_probe(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    max_pca_components: int,
) -> Path:
    """Decode past role actions from current raw LSTM state.

    Lag 0 is the contemporaneous control. Lag 1 includes the explicit
    previous-action input by architecture; lags >=2 are the useful recurrent
    memory test. PCA and standardization are fitted on train only, while C is
    selected on validation only.
    """
    network = _load_network(checkpoint, config_path)
    lag_grid = tuple(value for value in (0, 1, 2, 4, 8, 16, 32) if value < sequence_length)
    sampled_by_split: dict[str, dict[str, np.ndarray]] = {}
    hidden_by_split: dict[str, np.ndarray] = {}
    time_ranges: dict[str, tuple[str, str]] = {}
    for split in ("train", "validation", "test"):
        sampled = _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        sampled_by_split[split] = sampled
        hidden_by_split[split] = _replay_raw_hidden_cube(network, sampled)
        times = sampled["sample_times"].reshape(-1)
        time_ranges[split] = (str(times[0]), str(times[-1]))

    train_owner_rows = []
    for agent_id in (0, 1):
        cube = hidden_by_split["train"][:, :, agent_id, :]
        mask = sampled_by_split["train"]["owner_masks"][:, :, agent_id]
        train_owner_rows.append(cube[mask])
    pca_train = np.concatenate(train_owner_rows, axis=0)
    scaler = StandardScaler()
    standardized_train = scaler.fit_transform(pca_train)
    component_count = min(
        max(2, int(max_pca_components)),
        standardized_train.shape[0] - 1,
        standardized_train.shape[1],
    )
    pca = PCA(n_components=component_count, svd_solver="randomized", random_state=0)
    pca.fit(standardized_train)

    roles: dict[str, object] = {}
    for role_name, agent_id, entry_action in (
        ("agent_0_buy_role", 0, 1),
        ("agent_1_sell_role", 1, 2),
    ):
        curve: list[dict[str, object]] = []
        for lag in lag_grid:
            features: dict[str, np.ndarray] = {}
            labels: dict[str, np.ndarray] = {}
            counts: dict[str, dict[str, int]] = {}
            valid_classes = True
            for split in ("train", "validation", "test"):
                x, y = _temporal_role_rows(
                    hidden_by_split[split],
                    sampled_by_split[split]["actions"],
                    sampled_by_split[split]["owner_masks"],
                    agent_id=agent_id,
                    lag=lag,
                )
                keep = np.isin(y, (0, entry_action))
                x = x[keep]
                y = y[keep]
                features[split] = pca.transform(scaler.transform(x))
                labels[split] = (y == entry_action).astype(np.int64)
                counts[split] = {
                    "total": int(len(y)),
                    "hold": int(np.sum(y == 0)),
                    "entry": int(np.sum(y == entry_action)),
                }
                valid_classes &= len(np.unique(labels[split])) == 2
            if not valid_classes:
                curve.append({
                    "lag_steps": int(lag),
                    "status": "insufficient_two_class_support",
                    "sample_counts": counts,
                })
                continue
            probe = _fit_linear_probe(
                features["train"], labels["train"],
                features["validation"], labels["validation"],
                features["test"], labels["test"],
            )
            curve.append({
                "lag_steps": int(lag),
                "status": "complete",
                "sample_counts": counts,
                **probe,
            })
        valid_rows = [row for row in curve if row["status"] == "complete"]
        recurrent_rows = [row for row in valid_rows if int(row["lag_steps"]) >= 2]
        above_margin = [
            int(row["lag_steps"]) for row in recurrent_rows
            if float(row["metrics"]["validation"]["balanced_accuracy"]) >= 0.52
        ]
        roles[role_name] = {
            "agent_id": int(agent_id),
            "entry_action": int(entry_action),
            "curve": curve,
            "furthest_recurrent_lag_validation_ba_ge_0_52": (
                max(above_margin) if above_margin else None
            ),
        }

    payload = {
        "stage": 9,
        "name": "lstm_temporal_lag_action_decodability",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "representation": "raw_lstm_hidden_before_transformer",
        "target": "same_role_teacher_action_at_t_minus_lag",
        "interpretation": {
            "lag_0": "contemporaneous control",
            "lag_1": "contains explicit previous_action input and is not pure recurrent-memory evidence",
            "lag_ge_2": "recurrent-memory evidence beyond the explicit previous_action input",
        },
        "chronological_split_contract": True,
        "pca_fit_scope": "train_owner_hidden_only",
        "standardization_fit_scope": "train_owner_hidden_only",
        "hyperparameters_selected_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "sequence_length": int(sequence_length),
        "window_count_per_split": int(windows),
        "lag_grid": list(lag_grid),
        "pca_components": int(component_count),
        "pca_variance_captured": float(np.sum(pca.explained_variance_ratio_)),
        "time_ranges": time_ranges,
        "roles": roles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "09_temporal_lag_probe.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, (role_name, role) in zip(axes, roles.items()):
        rows = [row for row in role["curve"] if row["status"] == "complete"]
        lags = [int(row["lag_steps"]) for row in rows]
        validation = [float(row["metrics"]["validation"]["balanced_accuracy"]) for row in rows]
        test = [float(row["metrics"]["test"]["balanced_accuracy"]) for row in rows]
        axis.plot(lags, validation, marker="o", label="validation")
        axis.plot(lags, test, marker="o", label="test OOS")
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
        axis.axvline(1, color="gray", linestyle=":", linewidth=1, label="explicit prev-action boundary")
        axis.set_xticks(lags, [str(value) for value in lags])
        axis.set_ylim(0.4, 1.0)
        axis.set_xlabel("Lag (M1 decision steps)")
        axis.set_ylabel("Balanced accuracy")
        axis.set_title(role_name)
        axis.grid(alpha=0.25)
        axis.legend()
    png_path = output_dir / "09_temporal_lag_probe.png"
    figure.suptitle("Past teacher-action decodability from current raw LSTM state")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    print(json.dumps({
        "artifact": str(json_path),
        "plot": str(png_path),
        "roles": {
            name: {
                "furthest_recurrent_lag_validation_ba_ge_0_52": role[
                    "furthest_recurrent_lag_validation_ba_ge_0_52"
                ],
                "test_balanced_accuracy_by_lag": {
                    str(row["lag_steps"]): row["metrics"]["test"]["balanced_accuracy"]
                    for row in role["curve"] if row["status"] == "complete"
                },
            }
            for name, role in roles.items()
        },
    }, ensure_ascii=False, indent=2))
    return json_path


def _gate_selected_rows(value: torch.Tensor, selected: torch.Tensor, hidden_size: int) -> torch.Tensor:
    return torch.cat(
        [value.narrow(0, gate * hidden_size, hidden_size).index_select(0, selected) for gate in range(4)],
        dim=0,
    )


def _transplant_selected_lstm_units(
    source: SharedFiveTradingPolicy,
    selected_indices: np.ndarray,
) -> SharedFiveTradingPolicy:
    """Build a real narrower LSTM by retaining train-selected source units."""
    selected_np = np.asarray(selected_indices, dtype=np.int64)
    if selected_np.ndim != 1 or len(np.unique(selected_np)) != len(selected_np):
        raise ValueError("selected LSTM indices must be unique 1-D values")
    source_width = int(source.hidden_size)
    target_width = int(len(selected_np))
    if target_width < 2 or np.any(selected_np < 0) or np.any(selected_np >= source_width):
        raise ValueError("selected LSTM indices are out of range")
    target = SharedFiveTradingPolicy(
        pre_lstm_size=int(source.pre_lstm_size), hidden_size=target_width
    )
    source_transformer = source.post_lstm_transformer
    target.post_lstm_transformer = type(source_transformer)(
        target_width,
        model_width=int(source_transformer.model_width),
        head_count=int(source_transformer.head_count),
        ffn_size=int(source_transformer.ffn_size),
    )
    target = target.to(device=next(source.parameters()).device, dtype=next(source.parameters()).dtype)
    selected = torch.as_tensor(selected_np, device=next(source.parameters()).device, dtype=torch.long)
    source_state = source.state_dict()
    target_state = target.state_dict()
    transplanted: dict[str, torch.Tensor] = {}
    for name, target_value in target_state.items():
        source_value = source_state[name]
        if tuple(source_value.shape) == tuple(target_value.shape):
            transplanted[name] = source_value.detach().clone()
            continue
        if name in {"lstm.weight_ih", "lstm.bias_ih", "lstm.bias_hh"}:
            value = _gate_selected_rows(source_value, selected, source_width)
        elif name == "lstm.weight_hh":
            value = _gate_selected_rows(source_value, selected, source_width).index_select(1, selected)
        else:
            value = source_value
            for dimension, (source_dim, target_dim) in enumerate(zip(value.shape, target_value.shape)):
                if source_dim == target_dim:
                    continue
                if source_dim == source_width and target_dim == target_width:
                    value = value.index_select(dimension, selected)
                else:
                    raise RuntimeError(
                        f"cannot transplant {name}: source={tuple(source_value.shape)} "
                        f"target={tuple(target_value.shape)}"
                    )
        if tuple(value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"transplant shape mismatch {name}: got={tuple(value.shape)} "
                f"expected={tuple(target_value.shape)}"
            )
        transplanted[name] = value.detach().clone()
    target.load_state_dict(transplanted, strict=True)
    target.heads.entry_role_calibrator.teacher_entry_scope = str(
        source.heads.entry_role_calibrator.teacher_entry_scope
    )
    target.heads.entry_role_calibrator.teacher_imitation_trained = bool(
        source.heads.entry_role_calibrator.teacher_imitation_trained
    )
    target.eval()
    return target


def _width_behavior_metrics(
    network: SharedFiveTradingPolicy,
    sampled: dict[str, np.ndarray],
) -> dict[str, object]:
    hidden = _replay_raw_hidden_cube(network, sampled)
    windows, steps, _, width = hidden.shape
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    with torch.inference_mode():
        raw = torch.as_tensor(hidden, device=device, dtype=dtype).reshape(
            windows * steps, FIVE_AGENT_COUNT, width
        )
        transformed = network.post_lstm_transformer(raw)
        agent_ids = torch.arange(FIVE_AGENT_COUNT, device=device).expand(windows * steps, -1)
        logits = network.heads.primary_action_logits(transformed, agent_ids)
        for action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
            logits[..., int(action)] = -30.0
        predictions = logits.argmax(dim=-1).reshape(windows, steps, FIVE_AGENT_COUNT).cpu().numpy()
    roles: dict[str, object] = {}
    for role_name, agent_id, entry_action in (
        ("agent_0_buy_role", 0, 1),
        ("agent_1_sell_role", 1, 2),
    ):
        mask = sampled["owner_masks"][:, :, agent_id]
        truth = sampled["actions"][:, :, agent_id][mask]
        predicted = predictions[:, :, agent_id][mask]
        keep = np.isin(truth, (0, entry_action))
        truth_binary = (truth[keep] == entry_action).astype(np.int64)
        predicted_binary = (predicted[keep] == entry_action).astype(np.int64)
        roles[role_name] = {
            "sample_count": int(len(truth_binary)),
            "balanced_accuracy": float(balanced_accuracy_score(truth_binary, predicted_binary)),
            "mcc": float(matthews_corrcoef(truth_binary, predicted_binary)),
            "entry_precision": float(precision_score(truth_binary, predicted_binary, zero_division=0)),
            "entry_recall": float(recall_score(truth_binary, predicted_binary, zero_division=0)),
            "entry_f1": float(f1_score(truth_binary, predicted_binary, zero_division=0)),
            "predicted_entry_rate": float(np.mean(predicted_binary)),
            "teacher_entry_rate": float(np.mean(truth_binary)),
        }
    return {"roles": roles, "hidden": hidden}


def _forward_latency_ms(network: SharedFiveTradingPolicy, *, iterations: int = 10) -> float:
    observations = torch.zeros((FIVE_AGENT_COUNT, 137), dtype=next(network.parameters()).dtype)
    masks = torch.ones((FIVE_AGENT_COUNT, 12), dtype=torch.bool)
    with torch.inference_mode():
        for _ in range(2):
            network.forward_five(
                observations, network.initial_memories(),
                allowed_action_masks=masks, deterministic=True,
            )
        started = time.perf_counter()
        for _ in range(iterations):
            network.forward_five(
                observations, network.initial_memories(),
                allowed_action_masks=masks, deterministic=True,
            )
    return float((time.perf_counter() - started) * 1000.0 / iterations)


def run_lstm_width_ablation(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
) -> Path:
    source = _load_network(checkpoint, config_path)
    sampled = {
        split: _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        for split in ("train", "validation", "test")
    }
    train_hidden = _replay_raw_hidden_cube(source, sampled["train"])
    train_mask = sampled["train"]["owner_masks"]
    train_rows = train_hidden[train_mask]
    variance = np.var(train_rows, axis=0, dtype=np.float64)
    ranking = np.argsort(-variance, kind="stable")
    source_parameter_count = int(sum(value.numel() for value in source.parameters()))
    arms: dict[str, object] = {}
    for width in (4096, 2048, 1024):
        if width > int(source.hidden_size):
            raise ValueError(f"source hidden width {source.hidden_size} cannot provide arm {width}")
        selected = np.arange(width, dtype=np.int64) if width == int(source.hidden_size) else ranking[:width]
        network = source if width == int(source.hidden_size) else _transplant_selected_lstm_units(source, selected)
        split_results = {
            split: _width_behavior_metrics(network, sampled[split])
            for split in ("train", "validation", "test")
        }
        split_metrics = {
            split: {"roles": result["roles"]}
            for split, result in split_results.items()
        }
        probe_train_rows = []
        for agent_id in (0, 1):
            cube = split_results["train"]["hidden"][:, :, agent_id, :]
            mask = sampled["train"]["owner_masks"][:, :, agent_id]
            probe_train_rows.append(cube[mask])
        probe_fit = np.concatenate(probe_train_rows, axis=0)
        probe_scaler = StandardScaler()
        probe_fit_standardized = probe_scaler.fit_transform(probe_fit)
        probe_components = min(64, probe_fit_standardized.shape[0] - 1, int(width))
        probe_pca = PCA(
            n_components=probe_components, svd_solver="randomized", random_state=0
        )
        probe_pca.fit(probe_fit_standardized)
        probes: dict[str, object] = {}
        for role_name, agent_id, entry_action in (
            ("agent_0_buy_role", 0, 1),
            ("agent_1_sell_role", 1, 2),
        ):
            x: dict[str, np.ndarray] = {}
            y: dict[str, np.ndarray] = {}
            for split in ("train", "validation", "test"):
                cube = split_results[split]["hidden"][:, :, agent_id, :]
                mask = sampled[split]["owner_masks"][:, :, agent_id]
                labels = sampled[split]["actions"][:, :, agent_id][mask]
                keep = np.isin(labels, (0, entry_action))
                x[split] = probe_pca.transform(
                    probe_scaler.transform(cube[mask][keep])
                )
                y[split] = (labels[keep] == entry_action).astype(np.int64)
            probes[role_name] = _fit_linear_probe(
                x["train"], y["train"],
                x["validation"], y["validation"],
                x["test"], y["test"],
            )
        parameter_count = int(sum(value.numel() for value in network.parameters()))
        arms[str(width)] = {
            "hidden_size": int(width),
            "selection": (
                "canonical_full_checkpoint" if width == int(source.hidden_size)
                else "top_train_hidden_variance_units_structured_transplant"
            ),
            "selection_fit_scope": "chronological_train_only",
            "retained_train_hidden_variance_fraction": float(variance[selected].sum() / max(variance.sum(), 1e-30)),
            "parameter_count": parameter_count,
            "parameter_fraction_vs_4096": float(parameter_count / source_parameter_count),
            "mean_forward_latency_ms": _forward_latency_ms(network),
            "behavior": split_metrics,
            "linear_probe": {
                "pca_components": int(probe_components),
                "pca_variance_captured": float(np.sum(probe_pca.explained_variance_ratio_)),
                "roles": probes,
            },
        }
        if network is not source:
            del network
    payload = {
        "stage": 10,
        "name": "lstm_width_structured_transplant_ablation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "widths": [4096, 2048, 1024],
        "chronological_split_contract": True,
        "test_used_for_final_evaluation_only": True,
        "candidate_only": True,
        "promotable": False,
        "scope_warning": (
            "2048/1024 are real narrower LSTM modules with consistently sliced recurrent and downstream "
            "weights, but are not retrained. This tests how much learned behavior survives compression, "
            "not the best attainable quality of freshly imitation-trained smaller models."
        ),
        "sequence_length": int(sequence_length),
        "window_count_per_split": int(windows),
        "arms": arms,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "10_lstm_width_ablation.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(json_path),
        "arms": {
            width: {
                "parameter_count": arm["parameter_count"],
                "parameter_fraction_vs_4096": arm["parameter_fraction_vs_4096"],
                "mean_forward_latency_ms": arm["mean_forward_latency_ms"],
                "retained_train_hidden_variance_fraction": arm["retained_train_hidden_variance_fraction"],
                "test_roles": arm["behavior"]["test"]["roles"],
                "linear_probe_test_balanced_accuracy": {
                    role: values["metrics"]["test"]["balanced_accuracy"]
                    for role, values in arm["linear_probe"]["roles"].items()
                },
            }
            for width, arm in arms.items()
        },
    }, ensure_ascii=False, indent=2))
    return json_path


def _hooked_action_logits(
    network: SharedFiveTradingPolicy,
    sampled: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    """Capture final pre-selection logits through a temporary PyTorch hook."""
    hidden = _replay_raw_hidden_cube(network, sampled)
    windows, steps, _, width = hidden.shape
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    query_captures: list[torch.Tensor] = []
    adjustment_captures: list[torch.Tensor] = []
    logit_captures: list[torch.Tensor] = []
    logit_tap = torch.nn.Identity()
    handles = [
        network.heads.action_query.register_forward_hook(
            lambda _module, _inputs, output: query_captures.append(output.detach().cpu())
        ),
        logit_tap.register_forward_hook(
            lambda _module, _inputs, output: logit_captures.append(output.detach().cpu())
        ),
    ]
    handles.extend(
        branch.output.register_forward_hook(
            lambda _module, _inputs, output: adjustment_captures.append(output.detach().cpu())
        )
        for branch in network.heads.entry_role_calibrator.role_calibrators
    )
    try:
        with torch.inference_mode():
            raw = torch.as_tensor(hidden, device=device, dtype=dtype)
            temporal_tokens, temporal_valid = (
                network.post_lstm_transformer.empty_temporal_memory(
                    windows, device=device, dtype=dtype
                )
            )
            transformed_steps: list[torch.Tensor] = []
            for step in range(steps):
                transformed_steps.append(network.post_lstm_transformer(
                    raw[:, step], temporal_tokens, temporal_valid
                ))
                temporal_tokens, temporal_valid = (
                    network.post_lstm_transformer.update_temporal_memory(
                        raw[:, step], temporal_tokens, temporal_valid
                    )
                )
            transformed = torch.stack(transformed_steps, dim=1).reshape(
                windows * steps, FIVE_AGENT_COUNT, width
            )
            agent_ids = torch.arange(FIVE_AGENT_COUNT, device=device).expand(windows * steps, -1)
            logits = logit_tap(network.heads.primary_action_logits(transformed, agent_ids))
    finally:
        for handle in handles:
            handle.remove()
    if not (len(query_captures) == len(logit_captures) == 1 and len(adjustment_captures) == 2):
        raise RuntimeError(
            "logit hook capture count mismatch: "
            f"query={len(query_captures)}, adjustment={len(adjustment_captures)}, "
            f"logits={len(logit_captures)}"
        )
    captured = logit_captures[0]
    if not torch.equal(captured, logits.detach().cpu()):
        raise RuntimeError("hooked final logits differ from pre-selection logits")
    query = query_captures[0].numpy()
    adjustment = torch.stack(adjustment_captures, dim=0).numpy()
    return (
        captured.numpy().reshape(windows, steps, FIVE_AGENT_COUNT, -1),
        {
            "mechanism": "temporary_forward_hooks_on_action_query_entry_calibrator_and_final_identity_tap",
            "capture_count": 1,
            "role_calibrator_capture_count": 2,
            "query_shape": list(query.shape),
            "entry_adjustment_shape": list(adjustment.shape),
            "final_logits_shape": list(captured.shape),
            "query_abs_mean": float(np.mean(np.abs(query))),
            "entry_adjustment_abs_mean": float(np.mean(np.abs(adjustment))),
            "final_logits_finite": bool(np.all(np.isfinite(captured.numpy()))),
            "verified_equal_to_pre_selection_tensor": True,
        },
    )


def _distribution_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
        },
        "max": float(np.max(values)),
    }


def run_action_logit_diagnostics(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
) -> Path:
    network = _load_network(checkpoint, config_path)
    sampled_by_split: dict[str, dict[str, np.ndarray]] = {}
    logits_by_split: dict[str, np.ndarray] = {}
    hooks_by_split: dict[str, object] = {}
    time_ranges: dict[str, tuple[str, str]] = {}
    for split in ("train", "validation", "test"):
        sampled = _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        logits, hook = _hooked_action_logits(network, sampled)
        sampled_by_split[split] = sampled
        logits_by_split[split] = logits
        hooks_by_split[split] = hook
        times = sampled["sample_times"].reshape(-1)
        time_ranges[split] = (str(times[0]), str(times[-1]))

    roles: dict[str, object] = {}
    histogram_rows: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for role_name, agent_id, entry_action in (
        ("agent_0_buy_role", 0, 1),
        ("agent_1_sell_role", 1, 2),
    ):
        labels: dict[str, np.ndarray] = {}
        margins: dict[str, np.ndarray] = {}
        hold_logits: dict[str, np.ndarray] = {}
        entry_logits: dict[str, np.ndarray] = {}
        for split in ("train", "validation", "test"):
            mask = sampled_by_split[split]["owner_masks"][:, :, agent_id]
            actions = sampled_by_split[split]["actions"][:, :, agent_id][mask]
            keep = np.isin(actions, (0, entry_action))
            role_logits = logits_by_split[split][:, :, agent_id, :][mask][keep]
            labels[split] = (actions[keep] == entry_action).astype(np.int64)
            hold_logits[split] = role_logits[:, 0].astype(np.float64)
            entry_logits[split] = role_logits[:, entry_action].astype(np.float64)
            margins[split] = entry_logits[split] - hold_logits[split]
        sweep = _threshold_sweep(labels["validation"], margins["validation"])
        selected_threshold = float(sweep["selected_threshold"])
        evaluations = {
            split: {
                "default_threshold_0": _margin_metrics(labels[split], margins[split], 0.0),
                "validation_selected_threshold": _margin_metrics(
                    labels[split], margins[split], selected_threshold
                ),
                "sample_counts": {
                    "total": int(len(labels[split])),
                    "hold": int(np.sum(labels[split] == 0)),
                    "entry": int(np.sum(labels[split] == 1)),
                },
                "logits_by_teacher_class": {
                    class_name: {
                        "hold_logit": _distribution_summary(hold_logits[split][class_mask]),
                        "entry_logit": _distribution_summary(entry_logits[split][class_mask]),
                        "entry_minus_hold_margin": _distribution_summary(margins[split][class_mask]),
                    }
                    for class_name, class_mask in (
                        ("HOLD", labels[split] == 0),
                        ("ENTRY", labels[split] == 1),
                    )
                },
            }
            for split in ("train", "validation", "test")
        }
        roles[role_name] = {
            "agent_id": int(agent_id),
            "entry_action": int(entry_action),
            "margin": f"action_{entry_action}_logit_minus_action_0_hold_logit",
            "validation_threshold_sweep": sweep,
            "evaluations": evaluations,
            "test_calibration_delta_selected_minus_default": {
                metric: float(
                    evaluations["test"]["validation_selected_threshold"][metric]
                    - evaluations["test"]["default_threshold_0"][metric]
                )
                for metric in (
                    "balanced_accuracy", "mcc", "entry_precision", "entry_recall", "entry_f1"
                )
            },
        }
        histogram_rows.append(
            (role_name, margins["test"], labels["test"], selected_threshold)
        )

    payload = {
        "stage": 11,
        "name": "pre_selection_action_logit_and_threshold_diagnostics",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "chronological_split_contract": True,
        "threshold_selected_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "network_training_performed": False,
        "sequence_length": int(sequence_length),
        "window_count_per_split": int(windows),
        "time_ranges": time_ranges,
        "hook_audit": hooks_by_split,
        "roles": roles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "11_action_logit_diagnostics.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, (role_name, margins, labels, threshold) in zip(axes, histogram_rows):
        axis.hist(margins[labels == 0], bins=40, alpha=0.65, density=True, label="teacher HOLD")
        axis.hist(margins[labels == 1], bins=40, alpha=0.65, density=True, label="teacher ENTRY")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="default threshold 0")
        axis.axvline(threshold, color="red", linestyle=":", linewidth=1.5, label="validation-selected")
        axis.set_title(role_name)
        axis.set_xlabel("ENTRY logit - HOLD logit")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend()
    png_path = output_dir / "11_action_logit_margin_histograms.png"
    figure.suptitle("OOS pre-selection action margins by teacher class")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    print(json.dumps({
        "artifact": str(json_path),
        "plot": str(png_path),
        "roles": {
            name: {
                "selected_threshold": role["validation_threshold_sweep"]["selected_threshold"],
                "test_default": role["evaluations"]["test"]["default_threshold_0"],
                "test_selected": role["evaluations"]["test"]["validation_selected_threshold"],
                "delta": role["test_calibration_delta_selected_minus_default"],
            }
            for name, role in roles.items()
        },
    }, ensure_ascii=False, indent=2))
    return json_path


def _frozen_action_query_rows(
    network: SharedFiveTradingPolicy,
    sampled: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return post-Transformer action queries, owner ids and binary labels."""
    hidden = _replay_raw_hidden_cube(network, sampled)
    windows, steps, _, width = hidden.shape
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    with torch.inference_mode():
        raw = torch.as_tensor(hidden, device=device, dtype=dtype).reshape(
            windows * steps, FIVE_AGENT_COUNT, width
        )
        policy_hidden = network.post_lstm_transformer(raw)
        query = network.heads.action_query(policy_hidden).reshape(
            windows, steps, FIVE_AGENT_COUNT, -1
        )
    query_rows: list[torch.Tensor] = []
    agent_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for agent_id, entry_action in ((0, 1), (1, 2)):
        mask = sampled["owner_masks"][:, :, agent_id]
        actions = sampled["actions"][:, :, agent_id][mask]
        keep = np.isin(actions, (0, entry_action))
        role_query = query[:, :, agent_id, :][
            torch.as_tensor(mask, device=device)
        ][torch.as_tensor(keep, device=device)]
        query_rows.append(role_query.detach())
        agent_rows.append(torch.full(
            (len(actions[keep]),), agent_id, dtype=torch.long, device=device
        ))
        label_rows.append(torch.as_tensor(
            (actions[keep] == entry_action).astype(np.int64),
            dtype=torch.long, device=device,
        ))
    return torch.cat(query_rows), torch.cat(agent_rows), torch.cat(label_rows)


def _head_only_eval(
    network: SharedFiveTradingPolicy,
    query: torch.Tensor,
    agents: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    network.eval()
    with torch.inference_mode():
        logits = _role_logits(query, agents, network)
        probabilities = torch.softmax(logits, dim=-1)[:, 1]
        predictions = torch.argmax(logits, dim=-1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    y = labels.detach().cpu().numpy()
    prediction = predictions.detach().cpu().numpy()
    score = probabilities.detach().cpu().numpy()
    metrics = _binary_metrics(y, prediction, score)
    metrics["average_precision"] = float(average_precision_score(y, score))
    metrics["loss"] = float(loss.item())
    by_role = {}
    for role_name, agent_id in (("agent_0_buy_role", 0), ("agent_1_sell_role", 1)):
        selected = agents.detach().cpu().numpy() == agent_id
        role_metrics = _binary_metrics(y[selected], prediction[selected], score[selected])
        role_metrics["average_precision"] = float(
            average_precision_score(y[selected], score[selected])
        )
        by_role[role_name] = role_metrics
    return {"overall": metrics, "roles": by_role}


def run_head_only_calibrator_experiment(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    epochs: int = 80,
    patience: int = 12,
    learning_rate: float = 0.003,
) -> Path:
    """Train only the entry-role calibrator on frozen action-query features."""
    from torch.utils.tensorboard import SummaryWriter
    torch.manual_seed(0)
    np.random.seed(0)
    network = _load_network(checkpoint, config_path)
    source_hash_before = _sha256(checkpoint)
    invariant_before = network_invariant_sha256(network)
    trainable_before = teacher_trainable_sha256(network)
    sampled = {
        split: _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        for split in ("train", "validation", "test")
    }
    rows = {
        split: _frozen_action_query_rows(network, sampled[split])
        for split in ("train", "validation", "test")
    }
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    calibrator = network.heads.entry_role_calibrator
    calibrator.reset_teacher_mapping()
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    for parameter in calibrator.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=float(learning_rate))
    train_query, train_agents, train_labels = rows["train"]
    counts = torch.bincount(train_labels, minlength=2).to(dtype=train_query.dtype)
    class_weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    writer_dir = output_dir / "12_head_only_tensorboard"
    writer = SummaryWriter(log_dir=str(writer_dir))
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    try:
        for epoch in range(1, max(1, int(epochs)) + 1):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            logits = _role_logits(train_query, train_agents, network)
            loss = torch.nn.functional.cross_entropy(
                logits, train_labels, weight=class_weights
            )
            loss.backward()
            gradient_norms: dict[str, float] = {}
            for name, parameter in calibrator.named_parameters():
                if parameter.grad is None:
                    gradient_norms[name] = 0.0
                else:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"non-finite head-only gradient: {name}")
                    gradient_norms[name] = float(
                        torch.linalg.vector_norm(parameter.grad.detach()).item()
                    )
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=5.0)
            optimizer.step()
            train_eval = _head_only_eval(network, *rows["train"])
            validation_eval = _head_only_eval(network, *rows["validation"])
            role_values = list(validation_eval["roles"].values())
            key = (
                min(float(value["balanced_accuracy"]) for value in role_values),
                min(float(value["mcc"]) for value in role_values),
                min(float(value["entry_f1"]) for value in role_values),
            )
            row = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "gradient_norms": gradient_norms,
                "train": train_eval,
                "validation": validation_eval,
                "selection_key": list(key),
            }
            history.append(row)
            writer.add_scalar("loss/train_weighted", float(loss.item()), epoch)
            for split_name, values in (("train", train_eval), ("validation", validation_eval)):
                for role_name, metrics in values["roles"].items():
                    prefix = f"{split_name}/{role_name}"
                    for metric in (
                        "balanced_accuracy", "mcc", "entry_precision", "entry_recall",
                        "entry_f1", "predicted_entry_rate", "roc_auc", "average_precision",
                    ):
                        writer.add_scalar(f"{prefix}/{metric}", float(metrics[metric]), epoch)
            for name, value in gradient_norms.items():
                writer.add_scalar(f"gradient_norm/{name}", value, epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in calibrator.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= max(1, int(patience)):
                break
    finally:
        writer.close()
    if best_state is None:
        raise RuntimeError("head-only experiment did not produce a best state")
    calibrator.load_state_dict(best_state, strict=True)
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator.teacher_imitation_trained = True
    final = {
        split: _head_only_eval(network, *rows[split])
        for split in ("train", "validation", "test")
    }
    invariant_after = network_invariant_sha256(network)
    source_hash_after = _sha256(checkpoint)
    if invariant_after != invariant_before or source_hash_after != source_hash_before:
        raise RuntimeError("head-only experiment changed frozen backbone or source checkpoint")
    state_path = output_dir / "12_head_only_calibrator_state.pt"
    torch.save({
        "state_dict": best_state,
        "teacher_entry_scope": "all_entry_mass_v2",
        "source_checkpoint_sha256": source_hash_before,
        "candidate_only": True,
    }, state_path)
    payload = {
        "stage": 12,
        "name": "frozen_backbone_entry_calibrator_training",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_hash_before,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "training_surface": "heads.entry_role_calibrator_only",
        "frozen": ["observation_encoder", "lstm", "post_lstm_transformer", "heads.action_query", "heads.action_embedding", "value_and_geometry_heads"],
        "candidate_only": True,
        "promotable": False,
        "chronological_split_contract": True,
        "best_epoch_selected_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "sequence_length": int(sequence_length),
        "window_count_per_split": int(windows),
        "optimizer": {"name": "AdamW", "learning_rate": float(learning_rate)},
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(history)),
        "patience": int(patience),
        "best_epoch": int(best_epoch),
        "class_weights": class_weights.detach().cpu().tolist(),
        "network_invariant_sha256_before": invariant_before,
        "network_invariant_sha256_after": invariant_after,
        "frozen_backbone_unchanged": True,
        "source_checkpoint_unchanged": True,
        "calibrator_sha256_before": trainable_before,
        "calibrator_sha256_after": teacher_trainable_sha256(network),
        "calibrator_updated": teacher_trainable_sha256(network) != trainable_before,
        "tensorboard_log_dir": str(writer_dir),
        "state_artifact": str(state_path),
        "history": history,
        "final": final,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "12_head_only_calibrator_experiment.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(json_path),
        "tensorboard": str(writer_dir),
        "state": str(state_path),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "frozen_backbone_unchanged": True,
        "test": final["test"],
    }, ensure_ascii=False, indent=2))
    return json_path


def _frozen_policy_hidden_rows(
    network: SharedFiveTradingPolicy,
    sampled: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = _replay_raw_hidden_cube(network, sampled)
    windows, steps, _, width = hidden.shape
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    with torch.inference_mode():
        raw = torch.as_tensor(hidden, device=device, dtype=dtype).reshape(
            windows * steps, FIVE_AGENT_COUNT, width
        )
        policy_hidden = network.post_lstm_transformer(raw).reshape(
            windows, steps, FIVE_AGENT_COUNT, width
        )
    hidden_rows: list[torch.Tensor] = []
    agent_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for agent_id, entry_action in ((0, 1), (1, 2)):
        mask = sampled["owner_masks"][:, :, agent_id]
        actions = sampled["actions"][:, :, agent_id][mask]
        keep = np.isin(actions, (0, entry_action))
        role_hidden = policy_hidden[:, :, agent_id, :][
            torch.as_tensor(mask, device=device)
        ][torch.as_tensor(keep, device=device)]
        hidden_rows.append(role_hidden.detach())
        agent_rows.append(torch.full(
            (len(actions[keep]),), agent_id, dtype=torch.long, device=device
        ))
        label_rows.append(torch.as_tensor(
            (actions[keep] == entry_action).astype(np.int64),
            dtype=torch.long, device=device,
        ))
    return torch.cat(hidden_rows), torch.cat(agent_rows), torch.cat(label_rows)


def _state_hash_excluding(network: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        if name.startswith(prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _action_head_eval(
    network: SharedFiveTradingPolicy,
    hidden: torch.Tensor,
    agents: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    network.eval()
    with torch.inference_mode():
        query = network.heads.action_query(hidden)
        logits = _role_logits(query, agents, network)
        margin = logits[:, 1] - logits[:, 0]
        predictions = torch.argmax(logits, dim=-1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    y = labels.detach().cpu().numpy()
    prediction = predictions.detach().cpu().numpy()
    score = margin.detach().cpu().numpy()
    metrics = _binary_metrics(y, prediction, score)
    metrics["average_precision"] = float(average_precision_score(y, score))
    metrics["loss"] = float(loss.item())
    by_role = {}
    agent_values = agents.detach().cpu().numpy()
    for role_name, agent_id in (("agent_0_buy_role", 0), ("agent_1_sell_role", 1)):
        selected = agent_values == agent_id
        role_metrics = _binary_metrics(y[selected], prediction[selected], score[selected])
        role_metrics["average_precision"] = float(
            average_precision_score(y[selected], score[selected])
        )
        by_role[role_name] = role_metrics
    return {"overall": metrics, "roles": by_role}


def run_action_head_only_experiment(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    epochs: int = 80,
    patience: int = 12,
    learning_rate: float = 0.001,
) -> Path:
    """Train action-query plus entry calibrator while freezing the backbone."""
    from torch.utils.tensorboard import SummaryWriter
    torch.manual_seed(0)
    np.random.seed(0)
    network = _load_network(checkpoint, config_path)
    source_hash_before = _sha256(checkpoint)
    authorized = ("heads.action_query.", "heads.entry_role_calibrator.")
    frozen_before = _state_hash_excluding(network, authorized)
    sampled = {
        split: _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        for split in ("train", "validation", "test")
    }
    rows = {
        split: _frozen_policy_hidden_rows(network, sampled[split])
        for split in ("train", "validation", "test")
    }
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    calibrator = network.heads.entry_role_calibrator
    calibrator.reset_teacher_mapping()
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    for parameter in network.heads.action_query.parameters():
        parameter.requires_grad_(True)
    for parameter in calibrator.parameters():
        parameter.requires_grad_(True)
    trainable = [
        parameter for parameter in network.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=1e-4)
    train_hidden, train_agents, train_labels = rows["train"]
    counts = torch.bincount(train_labels, minlength=2).to(dtype=train_hidden.dtype)
    class_weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    writer_dir = output_dir / "13_action_head_only_tensorboard"
    writer = SummaryWriter(log_dir=str(writer_dir))
    initial = {
        split: _action_head_eval(network, *rows[split])
        for split in ("train", "validation", "test")
    }
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
        if name.startswith(authorized)
    }
    initial_roles = list(initial["validation"]["roles"].values())
    best_key = (
        min(float(value["balanced_accuracy"]) for value in initial_roles),
        min(float(value["mcc"]) for value in initial_roles),
        min(float(value["entry_f1"]) for value in initial_roles),
    )
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    try:
        for epoch in range(1, max(1, int(epochs)) + 1):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            query = network.heads.action_query(train_hidden)
            logits = _role_logits(query, train_agents, network)
            loss = torch.nn.functional.cross_entropy(
                logits, train_labels, weight=class_weights
            )
            loss.backward()
            gradient_norms: dict[str, float] = {
                "action_query": 0.0,
                "buy_calibrator": 0.0,
                "sell_calibrator": 0.0,
            }
            for name, parameter in network.named_parameters():
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise RuntimeError(f"frozen parameter received gradient: {name}")
                    continue
                gradient = parameter.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError(f"missing/non-finite action-head gradient: {name}")
                norm = float(torch.linalg.vector_norm(gradient.detach()).item())
                if name.startswith("heads.action_query."):
                    group = "action_query"
                elif ".role_calibrators.0." in name:
                    group = "buy_calibrator"
                elif ".role_calibrators.1." in name:
                    group = "sell_calibrator"
                else:
                    raise RuntimeError(f"unknown authorized action-head parameter: {name}")
                gradient_norms[group] = max(gradient_norms[group], norm)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            train_eval = _action_head_eval(network, *rows["train"])
            validation_eval = _action_head_eval(network, *rows["validation"])
            role_values = list(validation_eval["roles"].values())
            key = (
                min(float(value["balanced_accuracy"]) for value in role_values),
                min(float(value["mcc"]) for value in role_values),
                min(float(value["entry_f1"]) for value in role_values),
            )
            history.append({
                "epoch": int(epoch),
                "loss": float(loss.item()),
                "gradient_norms": gradient_norms,
                "train": train_eval,
                "validation": validation_eval,
                "selection_key": list(key),
            })
            writer.add_scalar("loss/train_weighted", float(loss.item()), epoch)
            for split_name, values in (("train", train_eval), ("validation", validation_eval)):
                for role_name, metrics in values["roles"].items():
                    for metric in (
                        "balanced_accuracy", "mcc", "entry_precision", "entry_recall",
                        "entry_f1", "predicted_entry_rate", "roc_auc", "average_precision",
                    ):
                        writer.add_scalar(
                            f"{split_name}/{role_name}/{metric}", float(metrics[metric]), epoch
                        )
            for name, value in gradient_norms.items():
                writer.add_scalar(f"gradient_norm/{name}", value, epoch)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                    if name.startswith(authorized)
                }
                stale = 0
            else:
                stale += 1
            if stale >= max(1, int(patience)):
                break
    finally:
        writer.close()
    current = network.state_dict()
    current.update(best_state)
    network.load_state_dict(current, strict=True)
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator.teacher_imitation_trained = True
    final = {
        split: _action_head_eval(network, *rows[split])
        for split in ("train", "validation", "test")
    }
    frozen_after = _state_hash_excluding(network, authorized)
    source_hash_after = _sha256(checkpoint)
    if frozen_after != frozen_before or source_hash_after != source_hash_before:
        raise RuntimeError("action-head experiment changed frozen backbone or source")
    state_path = output_dir / "13_action_head_only_state.pt"
    torch.save({
        "state_dict": best_state,
        "teacher_entry_scope": "all_entry_mass_v2",
        "source_checkpoint_sha256": source_hash_before,
        "candidate_only": True,
    }, state_path)
    payload = {
        "stage": 13,
        "name": "frozen_backbone_action_query_and_calibrator_training",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_hash_before,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "training_surface": ["heads.action_query", "heads.entry_role_calibrator"],
        "frozen": ["observation_encoder", "lstm", "post_lstm_transformer", "heads.action_embedding", "value_and_geometry_heads"],
        "candidate_only": True,
        "promotable": False,
        "chronological_split_contract": True,
        "best_epoch_selected_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "sequence_length": int(sequence_length),
        "window_count_per_split": int(windows),
        "optimizer": {"name": "AdamW", "learning_rate": float(learning_rate), "weight_decay": 1e-4},
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(history)),
        "patience": int(patience),
        "best_epoch": int(best_epoch),
        "class_weights": class_weights.detach().cpu().tolist(),
        "frozen_backbone_sha256_before": frozen_before,
        "frozen_backbone_sha256_after": frozen_after,
        "frozen_backbone_unchanged": True,
        "source_checkpoint_unchanged": True,
        "tensorboard_log_dir": str(writer_dir),
        "state_artifact": str(state_path),
        "initial": initial,
        "history": history,
        "final": final,
    }
    json_path = output_dir / "13_action_head_only_experiment.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(json_path),
        "tensorboard": str(writer_dir),
        "state": str(state_path),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "frozen_backbone_unchanged": True,
        "initial_test": initial["test"],
        "final_test": final["test"],
    }, ensure_ascii=False, indent=2))
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--svd-components", type=int, default=512)
    parser.add_argument("--active-run-dir", type=Path)
    parser.add_argument("--active-process-id", type=int)
    parser.add_argument("--bootstraps", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if args.stage == 1:
        run_torchinfo(checkpoint, config_path, output_dir)
    elif args.stage == 2:
        run_profiler(
            checkpoint, config_path, output_dir,
            iterations=max(1, int(args.iterations)),
        )
    elif args.stage == 3:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 3")
        run_effective_rank(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
            svd_components=max(2, int(args.svd_components)),
        )
    elif args.stage == 4:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 4")
        run_attention_map(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
        )
    elif args.stage == 5:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 5")
        run_linear_probe(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
        )
    elif args.stage == 6:
        run_ablation_status(
            checkpoint, output_dir,
            active_run_dir=(
                args.active_run_dir.resolve() if args.active_run_dir is not None else None
            ),
            active_process_id=args.active_process_id,
        )
    elif args.stage == 7:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 7")
        run_class_conditional_geometry(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
            bootstraps=max(0, int(args.bootstraps)),
        )
    elif args.stage == 8:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 8")
        run_priority_one_probe_suite(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
            max_pca_components=max(2, int(args.svd_components)),
            bootstraps=max(0, int(args.bootstraps)),
        )
    elif args.stage == 9:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 9")
        run_temporal_lag_probe(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(3, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
            max_pca_components=max(2, int(args.svd_components)),
        )
    elif args.stage == 10:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 10")
        run_lstm_width_ablation(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
        )
    elif args.stage == 11:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 11")
        run_action_logit_diagnostics(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
        )
    elif args.stage == 12:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 12")
        run_head_only_calibrator_experiment(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
        )
    elif args.stage == 13:
        if args.dataset is None:
            raise SystemExit("--dataset is required for stage 13")
        run_action_head_only_experiment(
            checkpoint, config_path, args.dataset.resolve(), output_dir,
            sequence_length=max(2, int(args.sequence_length)),
            windows=max(1, int(args.windows)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
