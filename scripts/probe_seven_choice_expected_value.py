#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.seven_choice_teacher import load_seven_choice_dataset, shared_five_network


def _selection(
    predicted: np.ndarray,
    realized: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    choices = []
    scores = []
    values = []
    for actions, columns in (
        ((1, 3, 5), (0, 2, 4)),
        ((2, 4, 6), (1, 3, 5)),
    ):
        local = predicted[:, columns]
        local_index = local.argmax(axis=1)
        action = np.asarray(actions, dtype=np.int64)[local_index]
        score = local[np.arange(len(local)), local_index]
        active = score > threshold
        choices.append(action[active])
        scores.append(score[active])
        values.append(realized[np.flatnonzero(active), action[active]])
    selected = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    count = len(selected)
    mean = float(selected.mean()) if count else None
    if count >= 2:
        lower = mean - 1.64 * float(selected.std(ddof=1)) / math.sqrt(count)
    else:
        lower = None
    return {
        "threshold": float(threshold),
        "entries": int(count),
        "mean_net_r": mean,
        "lcb_95_one_sided": lower,
        "win_rate": float((selected > 0.0).mean()) if count else None,
        "buy_entries": int(len(values[0])) if values else 0,
        "sell_entries": int(len(values[1])) if len(values) > 1 else 0,
        "mean_predicted_score": float(np.concatenate(scores).mean()) if count else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Do kha nang du doan ky vong cua sau cach vao lenh")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--minimum-validation-entries", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_seven_choice_dataset(args.dataset)
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        Path(args.source).resolve(), load_config(Path(args.config).resolve()), seed=args.seed
    )
    network = shared_five_network(system)
    network.require_current_architecture()
    device = next(network.parameters()).device
    network.eval()
    unique = np.flatnonzero(data.agent_ids == 0)
    observations = data.observations[unique]
    unique_observations = torch.as_tensor(observations, dtype=torch.float32)
    splits = data.splits[unique]
    realized = data.net_r[unique]
    filled = data.filled[unique, 1:7]
    head = network.observation_encoder.pattern_tree.action_path_head
    residual = getattr(network.observation_encoder.pattern_tree, "action_path_residual", None)
    observation_adapter = getattr(
        network.observation_encoder.pattern_tree,
        "action_path_observation_adapter",
        None,
    )
    train_modules = [head]
    if isinstance(residual, torch.nn.Module):
        train_modules.append(residual)
    if isinstance(observation_adapter, torch.nn.Module):
        train_modules.append(observation_adapter)

    def action_path_raw(
        root: torch.Tensor,
        raw_observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = head(root)
        if isinstance(residual, torch.nn.Module):
            value = value + residual(root)
        if isinstance(observation_adapter, torch.nn.Module) and raw_observations is not None:
            if raw_observations.ndim != 2 or raw_observations.shape[1] != 137:
                raise ValueError("raw_observations phai co dang [N,137]")
            bounded = torch.tanh(torch.nan_to_num(raw_observations))
            value = value + observation_adapter(bounded)
        return value
    roots = torch.empty((len(unique), int(head.in_features)))
    with torch.no_grad():
        for start in range(0, len(unique), 4096):
            obs = torch.as_tensor(observations[start:start + 4096], dtype=torch.float32, device=device)
            root = network.observation_encoder.pattern_predictions(obs)["root"].cpu()
            roots[start:start + len(root)] = root

    for parameter in network.parameters():
        parameter.requires_grad_(False)
    train_parameters = []
    for module in train_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            train_parameters.append(parameter)
    optimizer = torch.optim.Adam(train_parameters, lr=args.learning_rate, weight_decay=1e-4)
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "validation")
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_indices)
        head.train()
        total = 0.0
        seen = 0
        for start in range(0, len(train_indices), args.batch_size):
            idx = train_indices[start:start + args.batch_size]
            root = roots[idx].to(device=device)
            raw = action_path_raw(
                root,
                unique_observations[idx].to(device=device),
            ).reshape(-1, 6, 5)
            fill_target = torch.as_tensor(filled[idx], dtype=torch.float32, device=device)
            net_target = torch.as_tensor(
                np.clip(realized[idx, 1:7] / 5.0, -1.0, 1.0),
                dtype=torch.float32, device=device,
            )
            win_target = torch.as_tensor(
                realized[idx, 1:7] > 0.0, dtype=torch.float32, device=device,
            )
            loss = (
                2.0 * F.smooth_l1_loss(torch.tanh(raw[..., 2]), net_target)
                + 0.25 * F.binary_cross_entropy_with_logits(raw[..., 0], fill_target)
                + 0.25 * F.binary_cross_entropy_with_logits(raw[..., 1], win_target)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        head.eval()
        if isinstance(residual, torch.nn.Module):
            residual.eval()
        if isinstance(observation_adapter, torch.nn.Module):
            observation_adapter.eval()
        with torch.no_grad():
            raw = action_path_raw(
                roots[validation_indices].to(device=device),
                unique_observations[validation_indices].to(device=device),
            ).reshape(-1, 6, 5)
            predicted = torch.tanh(raw[..., 2]).cpu().numpy() * 5.0
        validation_mae = float(np.mean(np.abs(
            predicted - np.clip(realized[validation_indices, 1:7], -5.0, 5.0)
        )))
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "validation_net_r_mae": validation_mae,
        }
        history.append(row)
        print(json.dumps({"event": "expected_value_epoch", **row}), flush=True)
        if validation_mae < best_loss:
            best_loss = validation_mae
            best_state = {}
            for module_name, module in (
                ("head", head),
                ("residual", residual),
                ("observation_adapter", observation_adapter),
            ):
                if not isinstance(module, torch.nn.Module):
                    continue
                for name, value in module.state_dict().items():
                    best_state[f"{module_name}.{name}"] = value.detach().cpu().clone()
    if best_state is None:
        raise RuntimeError("khong co trang thai du doan ky vong")
    head.load_state_dict({
        name[len("head."):]: value for name, value in best_state.items()
        if name.startswith("head.")
    })
    if isinstance(residual, torch.nn.Module):
        residual.load_state_dict({
            name[len("residual."):]: value for name, value in best_state.items()
            if name.startswith("residual.")
        })
    if isinstance(observation_adapter, torch.nn.Module):
        observation_adapter.load_state_dict({
            name[len("observation_adapter."):]: value for name, value in best_state.items()
            if name.startswith("observation_adapter.")
        })
    for module in train_modules:
        module.eval()
    predictions = np.zeros((len(unique), 6), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(unique), 4096):
            raw = action_path_raw(
                roots[start:start + 4096].to(device=device),
                unique_observations[start:start + 4096].to(device=device),
            ).reshape(-1, 6, 5)
            predictions[start:start + len(raw)] = torch.tanh(raw[..., 2]).cpu().numpy() * 5.0

    validation_prediction = predictions[validation_indices]
    best_per_role = np.concatenate((
        validation_prediction[:, [0, 2, 4]].max(axis=1),
        validation_prediction[:, [1, 3, 5]].max(axis=1),
    ))
    quantiles = np.unique(np.quantile(
        best_per_role, np.linspace(0.25, 0.995, 151)
    ))
    candidates = []
    for threshold in quantiles:
        result = _selection(
            validation_prediction, realized[validation_indices], float(threshold)
        )
        if result["entries"] >= args.minimum_validation_entries:
            candidates.append(result)
    if not candidates:
        raise RuntimeError("khong co nguong du so lenh kiem dinh")
    selected = max(
        candidates,
        key=lambda row: float(row["lcb_95_one_sided"] or -1e9),
    )
    threshold = float(selected["threshold"])
    results = {}
    for split in ("train", "validation", "test"):
        indices = np.flatnonzero(splits == split)
        results[split] = _selection(predictions[indices], realized[indices], threshold)
    report = {
        "schema_version": 1,
        "dataset": str(Path(args.dataset).resolve()),
        "source_checkpoint": str(Path(args.source).resolve()),
        "selection_uses": "validation only",
        "test_used_for_selection": False,
        "best_epoch": int(np.argmin([row["validation_net_r_mae"] for row in history]) + 1),
        "history": history,
        "selected_threshold": threshold,
        "results": results,
        "validation_gate_passed": bool(
            (results["validation"]["lcb_95_one_sided"] or -1e9) > 0.0
        ),
        "test_gate_passed": bool(
            (results["test"]["lcb_95_one_sided"] or -1e9) > 0.0
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "expected_value_probe_complete", **report}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
