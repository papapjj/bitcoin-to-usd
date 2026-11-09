#!/usr/bin/env python3
"""Evidence gates for causal and long-horizon Five v3 temporal tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.chuyen_gia_policy import (  # noqa: E402
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config  # noqa: E402
from self_growing_trader.five_shared_policy import (  # noqa: E402
    FIVE_AGENT_COUNT,
    FIVE_POLICY_ARCHITECTURE_VERSION,
    FIVE_TEMPORAL_TOKEN_HORIZONS,
    SharedFiveTradingPolicy,
)
from self_growing_trader.five_teacher_imitation import shared_five_network  # noqa: E402
from self_growing_trader.protocol_contract import INPUT_COUNT  # noqa: E402


TEMPORAL_EVAL_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_timeline(
    network: SharedFiveTradingPolicy,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    memories = network.initial_memories(device=observations.device)
    probabilities: list[torch.Tensor] = []
    hidden_states: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for step in range(int(observations.shape[0])):
            results, memories, _ = network.forward_five(
                observations[step], memories, deterministic=True
            )
            probabilities.append(torch.stack([
                torch.as_tensor(row.policy_action_probabilities) for row in results
            ]))
            hidden_states.append([memory.hidden.detach().clone() for memory in memories])
    return torch.stack(probabilities), hidden_states


def causal_intervention_audit(
    network: SharedFiveTradingPolicy,
    *,
    steps: int = 12,
    cutoff: int = 6,
    seed: int = 20260908,
) -> dict[str, object]:
    """Perturb only unavailable future rows and demand an invariant prefix."""

    if not 1 <= cutoff < steps:
        raise ValueError("causal audit requires 1 <= cutoff < steps")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observations = torch.randn(
        (steps, FIVE_AGENT_COUNT, INPUT_COUNT), generator=generator
    )
    intervention = observations.clone()
    intervention[cutoff:] = torch.randn(
        intervention[cutoff:].shape, generator=generator
    ) * 17.0 + 11.0
    was_training = network.training
    network.eval()
    original_probabilities, original_hidden = _policy_timeline(network, observations)
    changed_probabilities, changed_hidden = _policy_timeline(network, intervention)
    network.train(was_training)
    probability_delta = float(torch.max(torch.abs(
        original_probabilities[:cutoff] - changed_probabilities[:cutoff]
    )).item())
    hidden_delta = max(
        float(torch.max(torch.abs(
            original_hidden[step][agent] - changed_hidden[step][agent]
        )).item())
        for step in range(cutoff)
        for agent in range(FIVE_AGENT_COUNT)
    )

    # A done boundary must erase only the ended agent's old temporal/LSTM state.
    boundary_a = torch.randn(
        (2, FIVE_AGENT_COUNT, INPUT_COUNT), generator=generator
    )
    boundary_b = boundary_a.clone()
    boundary_b[0, 0] = boundary_b[0, 0] * -19.0 + 7.0
    dones = torch.zeros((2, FIVE_AGENT_COUNT), dtype=torch.bool)
    dones[0, 0] = True
    memory_a = network.advance_history(boundary_a, dones=dones)
    memory_b = network.advance_history(boundary_b, dones=dones)
    reset_hidden_delta = float(torch.max(torch.abs(
        memory_a[0].hidden - memory_b[0].hidden
    )).item())
    reset_cell_delta = float(torch.max(torch.abs(
        memory_a[0].cell - memory_b[0].cell
    )).item())
    reset_token_delta = float(torch.max(torch.abs(
        memory_a[0].temporal_tokens - memory_b[0].temporal_tokens
    )).item())
    tolerance = 1e-7
    passed = bool(
        probability_delta <= tolerance
        and hidden_delta <= tolerance
        and reset_hidden_delta <= tolerance
        and reset_cell_delta <= tolerance
        and reset_token_delta <= tolerance
    )
    return {
        "passed": passed,
        "tolerance": tolerance,
        "future_intervention_starts_at_step": int(cutoff),
        "compared_prefix_steps": int(cutoff),
        "max_abs_prefix_policy_probability_delta": probability_delta,
        "max_abs_prefix_lstm_hidden_delta": hidden_delta,
        "agent_boundary_reset": {
            "ended_agent": 0,
            "max_abs_hidden_delta_after_reset": reset_hidden_delta,
            "max_abs_cell_delta_after_reset": reset_cell_delta,
            "max_abs_temporal_token_delta_after_reset": reset_token_delta,
        },
        "interpretation": (
            "Only observations after the decision prefix were changed; prefix outputs "
            "and the reset agent state must remain invariant."
        ),
    }


def temporal_impulse_retention_audit(
    network: SharedFiveTradingPolicy,
    *,
    lags: tuple[int, ...] = (32, 128),
) -> dict[str, object]:
    """Measure how much of a controlled past hidden-state impulse remains."""

    block = network.post_lstm_transformer
    device = next(block.parameters()).device
    dtype = next(block.parameters()).dtype
    signal_hidden = torch.zeros(
        (FIVE_AGENT_COUNT, network.hidden_size), device=device, dtype=dtype
    )
    signal_hidden[0, 0] = 1.0
    baseline_hidden = torch.zeros_like(signal_hidden)
    signal_tokens, signal_valid = block.empty_temporal_memory(
        device=device, dtype=dtype
    )
    baseline_tokens, baseline_valid = block.empty_temporal_memory(
        device=device, dtype=dtype
    )
    with torch.inference_mode():
        signal_tokens, signal_valid = block.update_temporal_memory(
            signal_hidden, signal_tokens, signal_valid
        )
        baseline_tokens, baseline_valid = block.update_temporal_memory(
            baseline_hidden, baseline_tokens, baseline_valid
        )
        initial_delta = signal_tokens[0] - baseline_tokens[0]
        initial_norms = torch.linalg.vector_norm(initial_delta, dim=-1)
        rows: dict[str, object] = {}
        maximum_lag = max(int(value) for value in lags)
        for lag in range(1, maximum_lag + 1):
            signal_tokens, signal_valid = block.update_temporal_memory(
                baseline_hidden, signal_tokens, signal_valid
            )
            baseline_tokens, baseline_valid = block.update_temporal_memory(
                baseline_hidden, baseline_tokens, baseline_valid
            )
            if lag not in lags:
                continue
            delta = signal_tokens[0] - baseline_tokens[0]
            norms = torch.linalg.vector_norm(delta, dim=-1)
            ratios = norms / initial_norms.clamp_min(torch.finfo(dtype).eps)
            cosine = torch.nn.functional.cosine_similarity(
                delta, initial_delta, dim=-1
            )
            expected = torch.pow(
                block.temporal_decay.to(dtype=dtype), lag
            )
            absolute_error = torch.abs(ratios - expected)
            rows[str(lag)] = {
                "retention_ratio_by_horizon": {
                    str(horizon): float(ratios[index].item())
                    for index, horizon in enumerate(FIVE_TEMPORAL_TOKEN_HORIZONS)
                },
                "expected_decay_ratio_by_horizon": {
                    str(horizon): float(expected[index].item())
                    for index, horizon in enumerate(FIVE_TEMPORAL_TOKEN_HORIZONS)
                },
                "cosine_to_original_impulse_by_horizon": {
                    str(horizon): float(cosine[index].item())
                    for index, horizon in enumerate(FIVE_TEMPORAL_TOKEN_HORIZONS)
                },
                "max_abs_ratio_error": float(absolute_error.max().item()),
                "other_agent_max_abs_delta": float(
                    torch.max(torch.abs(
                        (signal_tokens - baseline_tokens)[1:]
                    )).item()
                ),
            }
    required = {
        32: float(rows["32"]["retention_ratio_by_horizon"]["32"]),
        128: float(rows["128"]["retention_ratio_by_horizon"]["128"]),
    }
    min_ratio = 0.30
    min_cosine = 0.999
    max_error = 2e-5
    passed = bool(
        all(value >= min_ratio for value in required.values())
        and float(rows["32"]["cosine_to_original_impulse_by_horizon"]["32"]) >= min_cosine
        and float(rows["128"]["cosine_to_original_impulse_by_horizon"]["128"]) >= min_cosine
        and max(float(row["max_abs_ratio_error"]) for row in rows.values()) <= max_error
        and max(float(row["other_agent_max_abs_delta"]) for row in rows.values()) == 0.0
    )
    return {
        "passed": passed,
        "method": "paired impulse-minus-zero intervention on production input projection",
        "impulse_agent": 0,
        "impulse_hidden_dimension": 0,
        "minimum_required_retention_ratio": min_ratio,
        "minimum_required_cosine": min_cosine,
        "maximum_allowed_decay_error": max_error,
        "required_retention": {
            "lag_32_from_horizon_32": required[32],
            "lag_128_from_horizon_128": required[128],
        },
        "measurements": rows,
        "interpretation": (
            "Subtracting an all-zero control removes projection bias. A retained delta "
            "parallel to the original impulse proves recoverable past information."
        ),
    }


def run_evaluation(
    checkpoint: Path,
    config_path: Path,
    output: Path,
) -> dict[str, object]:
    source_hash_before = _sha256(checkpoint)
    config = load_config(config_path)
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        checkpoint, config, seed=0
    )
    network = shared_five_network(system)
    network.require_current_architecture()
    if int(network.policy_architecture_version) != FIVE_POLICY_ARCHITECTURE_VERSION:
        raise RuntimeError("checkpoint did not load as current Five architecture")
    causality = causal_intervention_audit(network)
    retention = temporal_impulse_retention_audit(network)
    source_hash_after = _sha256(checkpoint)
    report = {
        "schema_version": TEMPORAL_EVAL_SCHEMA_VERSION,
        "created_at": time.time(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_hash_before,
        "config": str(config_path),
        "architecture_version_after_load": int(network.policy_architecture_version),
        "hidden_size": int(network.hidden_size),
        "temporal_token_horizons": list(FIVE_TEMPORAL_TOKEN_HORIZONS),
        "causality_gate": causality,
        "retention_gate": retention,
        "source_checkpoint_unchanged": source_hash_before == source_hash_after,
        "all_offline_temporal_gates_passed": bool(
            causality["passed"]
            and retention["passed"]
            and source_hash_before == source_hash_after
        ),
        "mt5_oos_gate": {
            "passed": False,
            "status": "not_run_by_this_evaluator",
            "required_runner": str(ROOT / "scripts" / "run_mt5_transformer_ablation.py"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    report = run_evaluation(checkpoint, config_path, output)
    print(json.dumps({
        "event": "five_temporal_v3_evaluation_complete",
        "output": str(output),
        "causality_passed": report["causality_gate"]["passed"],
        "retention_passed": report["retention_gate"]["passed"],
        "all_offline_temporal_gates_passed": report["all_offline_temporal_gates_passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if bool(report["all_offline_temporal_gates_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
