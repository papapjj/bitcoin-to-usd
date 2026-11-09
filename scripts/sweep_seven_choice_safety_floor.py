#!/usr/bin/env python3
"""Sweep the causal win-probability floor without changing a checkpoint.

This is a research diagnostic.  It selects the threshold on validation only
and then reports the untouched test result for each fixed floor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config  # noqa: E402
from self_growing_trader.self_growing_runner import (  # noqa: E402
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.seven_choice_teacher import (  # noqa: E402
    _entry_selection_metrics,
    load_seven_choice_dataset,
    shared_five_network,
)


def _scores(checkpoint: Path, config_path: Path, data) -> tuple[np.ndarray, np.ndarray]:
    system = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint, load_config(config_path), seed=0)
    network = shared_five_network(system)
    network.require_current_architecture()
    network.eval()
    tree = network.observation_encoder.pattern_tree
    head = tree.action_path_head
    residual = getattr(tree, "action_path_residual", None)
    adapter = getattr(tree, "action_path_observation_adapter", None)
    unique = np.flatnonzero(data.agent_ids == 0)
    roots = torch.empty((len(unique), int(head.in_features)), dtype=torch.float32)
    values: list[np.ndarray] = []
    wins: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(unique), 4096):
            idx = unique[start:start + 4096]
            obs = torch.as_tensor(data.observations[idx], dtype=torch.float32)
            roots[start:start + len(idx)] = network.observation_encoder.pattern_predictions(obs)["root"].cpu()
        for start in range(0, len(unique), 4096):
            stop = min(len(unique), start + 4096)
            root = roots[start:stop]
            raw = head(root)
            if isinstance(residual, torch.nn.Module):
                raw = raw + residual(root)
            if isinstance(adapter, torch.nn.Module):
                bounded = torch.tanh(torch.nan_to_num(
                    torch.as_tensor(data.observations[unique[start:stop]], dtype=torch.float32),
                    nan=0.0, posinf=0.0, neginf=0.0,
                ))
                raw = raw + adapter(bounded)
            raw = raw.reshape(-1, 6, 5)
            values.append(torch.tanh(raw[..., 2]).cpu().numpy() * 5.0)
            wins.append(torch.sigmoid(raw[..., 1]).cpu().numpy())
    return np.concatenate(values), np.concatenate(wins)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--minimum-validation-entries", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_seven_choice_dataset(args.dataset)
    floors = (0.0, 1.0 / 3.0, 0.3833333333, 0.45, 0.50, 0.55, 0.60)
    report: dict[str, object] = {"selection_uses": "validation only", "models": {}}
    for checkpoint in args.checkpoint:
        predicted, predicted_win = _scores(checkpoint, args.config, data)
        duplicated = np.repeat(predicted[:, None, :], 2, axis=1).reshape(-1, 6)
        duplicated_win = np.repeat(predicted_win[:, None, :], 2, axis=1).reshape(-1, 6)
        model_rows: list[dict[str, object]] = []
        for floor in floors:
            scores = np.zeros((len(data.observations), 7), dtype=np.float32)
            scores[:, 1:7] = np.where(duplicated_win >= floor, duplicated, -5.0)
            validation_mask = data.splits == "validation"
            best_score = np.max(
                np.where(data.allowed_action_mask[validation_mask, 1:7], scores[validation_mask, 1:7], -np.inf),
                axis=1,
            )
            thresholds = np.unique(np.concatenate((
                np.quantile(best_score, np.linspace(0.25, 0.995, 151)),
                np.asarray((0.0, 0.10, 0.20, 0.30, 0.40), dtype=np.float32),
            )))
            candidates: list[tuple[float, dict[str, object]]] = []
            for threshold in thresholds:
                _, metrics = _entry_selection_metrics(data, scores, validation_mask, float(threshold))
                if int(metrics["entries"] or 0) >= args.minimum_validation_entries:
                    candidates.append((float(threshold), metrics))
            if not candidates:
                continue
            threshold, validation = max(
                candidates, key=lambda item: float(item[1]["lcb_95_one_sided"] or -1e9)
            )
            _, test = _entry_selection_metrics(data, scores, data.splits == "test", threshold)
            model_rows.append({"floor": floor, "threshold": threshold, "validation": validation, "test": test})
        report["models"][str(checkpoint)] = model_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": list(report["models"].keys())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
