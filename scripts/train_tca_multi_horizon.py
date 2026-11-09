#!/usr/bin/env python3
"""
Adaptive Training Runner & Zero-Drift Pipeline for 240-Candle TCA Multi-Horizon Returns.
File: scripts/train_tca_multi_horizon.py
Milestone: Milestone 3 (Features 9, 10, 11, 12, 13)

Architecture:
- Phase 1 Pre-training:
  * Base MarketStateEncoder strictly frozen (requires_grad=False, 8 keys, 467,200 params).
  * MetacognitiveReasoningEngine strictly frozen (requires_grad=False, 191 keys, 4,196,564 weights).
  * Trainable: TemporalCrossAttentionEncoder (40 trainable tensors, 1,842,048 weights) +
               MultiHorizonProjectionHead (8 tensors, 230,659 weights) = 2,072,707 weights.
  * Optimizer: AdamW(Group 1 [tca]: lr=1e-4, Group 2 [head]: lr=3e-4, wd=1e-2).
  * Scheduler: SequentialLR([LinearLR(..., start_factor=0.1), CosineAnnealingLR(..., eta_min=1e-6)]).
  * Checkpoint: outputs/checkpoints/tca_phase1_pretrained.pt.

- Phase 2 Fine-tuning:
  * Base MarketStateEncoder unfrozen (requires_grad=True, lr=1e-5).
  * MetacognitiveReasoningEngine remains strictly frozen (requires_grad=False, 0.0 drift).
  * Trainable: MarketStateEncoder (467,200) + TemporalCrossAttentionEncoder (1,842,048) +
               MultiHorizonProjectionHead (230,659) = 2,539,907 weights across 509k bars.
  * Optimizer: AdamW(Group 1 [encoder]: lr=1e-5, Group 2 [tca]: lr=5e-5, Group 3 [head]: lr=1e-4, wd=1e-2).
  * Scheduler: Cosine Annealing with linear warm-up.
  * Checkpoint: outputs/checkpoints/tca_phase2_finetuned.pt.

- Specialized Mega-Pool Routing:
  * Equal regime batch sampling across Bull (243,084 bars), Bear (132,405 bars), and Chop (133,798 bars).
  * Programmatic OOS firewall strictly isolating reserved crash slice [55000:64785] (9,785 bars).

- Loss & Dual Firewall:
  * Smooth L1 Huber loss (beta=1.0) on multi-horizon return targets [t+5, t+15, t+30] normalized by ATR.
  * Firewall 1: Check torch.isfinite(loss) before backward.
  * Firewall 2: Clip gradient norm (max_norm=1.0) and verify math.isfinite(grad_norm) before step.
  * Telemetry: Loss decomposition (L5, L15, L30), ATR-normalized MAE, and directional accuracy (DA%).

- Cryptographic Zero-Drift Validation:
  * SHA-256 invariant hash check on MetacognitiveReasoningEngine state dict before and after training.
  * Hard assertions: all(p.grad is None) and max(|p_final - p_init|) == 0.0.

- Checkpoint Serialization:
  * 4-namespace dictionary structure:
    world_model_state_dict (strictly 30 keys baseline),
    temporal_encoder_state_dict (41 keys),
    projection_head_state_dict (8 keys),
    reasoning_engine_state_dict (191 keys),
    training_metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Ensure repository root and research_reasoning_world_class are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RRWC_PATH = REPO_ROOT / "research_reasoning_world_class"
if str(RRWC_PATH) not in sys.path:
    sys.path.insert(0, str(RRWC_PATH))

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    MultiHorizonProjectionHead,
    compute_multi_horizon_loss,
    create_projection_head,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.mega_pool_dataset import (
    MegaPoolRouter,
    MegaPoolConfig,
    RoutingMode,
    MegaPoolBatch,
    MarketRegime,
)


# =============================================================================
# 1. DETERMINISTIC SEEDING & HARDWARE RESOLUTION
# =============================================================================


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolves target compute device with safe CPU fallback."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn(f"CUDA requested ('{device_arg}') but not available on this machine. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_arg)
    return torch.device("cpu")


# =============================================================================
# 2. CRYPTOGRAPHIC ZERO-DRIFT HASHING
# =============================================================================


def hash_reasoning_engine(model_or_sd: Union[nn.Module, Dict[str, torch.Tensor]]) -> str:
    """
    Computes deterministic SHA-256 cryptographic digest across all 191 keys
    of MetacognitiveReasoningEngine sorted lexicographically.
    Sensitive to 1e-7 float perturbations.
    """
    sd = model_or_sd.state_dict() if isinstance(model_or_sd, nn.Module) else model_or_sd
    hasher = hashlib.sha256()
    for key in sorted(sd.keys()):
        tensor = sd[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


# =============================================================================
# 3. OPTIMIZER & LEARNING RATE SCHEDULER BUILDERS
# =============================================================================


def build_phase1_optimizer(
    temporal_encoder: nn.Module,
    head: nn.Module,
    lr_temporal: float = 1e-4,
    lr_head: float = 3e-4,
    weight_decay: float = 1e-2,
) -> AdamW:
    """
    Builds Phase 1 AdamW optimizer with 2 differential parameter groups:
    - Group 1: temporal_encoder (40 parameter tensors, 1,842,048 weights, lr=1e-4)
    - Group 2: head (8 parameter tensors, 230,659 weights, lr=3e-4)
    Total: 48 parameter tensors, 2,072,707 trainable parameters.
    Base encoder (8 tensors, 467,200 params) and Reasoning Engine (191 keys) are excluded.
    """
    tca_params = [p for p in temporal_encoder.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]

    param_groups = [
        {
            "name": "temporal_encoder",
            "params": tca_params,
            "lr": lr_temporal,
            "weight_decay": weight_decay,
        },
        {
            "name": "head",
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
        },
    ]
    return AdamW(param_groups)


def build_phase2_optimizer(
    encoder: nn.Module,
    temporal_encoder: nn.Module,
    head: nn.Module,
    lr_encoder: float = 1e-5,
    lr_temporal: float = 5e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-2,
) -> AdamW:
    """
    Builds Phase 2 AdamW optimizer with 3 differential parameter groups:
    - Group 1: base encoder (8 parameter tensors, 467,200 weights, lr=1e-5)
    - Group 2: temporal_encoder (40 parameter tensors, 1,842,048 weights, lr=5e-5)
    - Group 3: head (8 parameter tensors, 230,659 weights, lr=1e-4)
    Total: 56 parameter tensors, 2,539,907 trainable parameters.
    Reasoning Engine remains strictly excluded and frozen.
    """
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    tca_params = [p for p in temporal_encoder.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]

    param_groups = [
        {
            "name": "encoder",
            "params": enc_params,
            "lr": lr_encoder,
            "weight_decay": weight_decay,
        },
        {
            "name": "temporal_encoder",
            "params": tca_params,
            "lr": lr_temporal,
            "weight_decay": weight_decay,
        },
        {
            "name": "head",
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
        },
    ]
    return AdamW(param_groups)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    max_warmup_steps: int = 500,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Builds a step-based SequentialLR scheduler:
    - LinearLR warm-up from 10% base LR to 100% over the first 5% of total steps.
    - CosineAnnealingLR decay down to eta_min=1e-6 at final convergence step.
    Handles heterogeneous parameter groups independently.
    """
    total_steps = max(1, total_steps)
    if total_steps <= 2:
        return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=eta_min)

    warmup_steps = min(max_warmup_steps, max(1, int(total_steps * warmup_ratio)))
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )


# =============================================================================
# 4. CHECKPOINT NAMESPACE ISOLATION & ATOMIC SERIALIZATION
# =============================================================================


def extract_4_namespace_checkpoint(
    world_model: LatentDynamicsWorldModel,
    head: MultiHorizonProjectionHead,
    reasoning_engine: MetacognitiveReasoningEngine,
    training_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts explicit 4-namespace dictionary structure conforming to R3 & Feature 13:
    1. world_model_state_dict: exactly 30 baseline keys without temporal_encoder prefix.
    2. temporal_encoder_state_dict: exactly 41 keys from TemporalCrossAttentionEncoder.
    3. projection_head_state_dict: exactly 8 keys (230,659 params).
    4. reasoning_engine_state_dict: exactly 191 keys (4,196,564 weights).
    5. training_metadata: epoch, loss, metrics, sha256, timestamps.
    """
    full_wm_sd = world_model.state_dict()
    clean_wm_sd = {k: v for k, v in full_wm_sd.items() if not k.startswith("temporal_encoder.")}
    if len(clean_wm_sd) != 30:
        raise ValueError(f"Expected 30 baseline world model keys, got {len(clean_wm_sd)}")

    if world_model.temporal_encoder is None:
        raise RuntimeError("world_model.temporal_encoder must be enabled before checkpoint export")

    tca_sd = world_model.temporal_encoder.state_dict()
    if len(tca_sd) != 41:
        raise ValueError(f"Expected 41 temporal encoder keys, got {len(tca_sd)}")

    head_sd = head.state_dict()
    if len(head_sd) != 8:
        raise ValueError(f"Expected 8 projection head keys, got {len(head_sd)}")

    re_sd = reasoning_engine.state_dict()
    if len(re_sd) != 191:
        raise ValueError(f"Expected 191 reasoning engine keys, got {len(re_sd)}")

    return {
        "world_model_state_dict": clean_wm_sd,
        "temporal_encoder_state_dict": tca_sd,
        "projection_head_state_dict": head_sd,
        "reasoning_engine_state_dict": re_sd,
        "training_metadata": training_metadata or {},
    }


def save_atomic_checkpoint(payload: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Atomically serializes checkpoint using a temporary file and atomic replace."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(filepath)


# =============================================================================
# 5. DUAL-FIREWALL TRAINING STEP & METRICS COMPUTATION
# =============================================================================


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_matrix: torch.Tensor,
) -> Dict[str, float]:
    """Computes decomposed multi-horizon losses, MAEs, and Directional Accuracies."""
    with torch.no_grad():
        abs_diff = (preds - targets).abs()
        da_matrix = (torch.sign(preds) == torch.sign(targets)).float()

        return {
            "loss_total": float(loss_matrix.mean().item()),
            "loss_5": float(loss_matrix[:, 0].mean().item()),
            "loss_15": float(loss_matrix[:, 1].mean().item()),
            "loss_30": float(loss_matrix[:, 2].mean().item()),
            "mae_total": float(abs_diff.mean().item()),
            "mae_5": float(abs_diff[:, 0].mean().item()),
            "mae_15": float(abs_diff[:, 1].mean().item()),
            "mae_30": float(abs_diff[:, 2].mean().item()),
            "da_total": float(da_matrix.mean().item() * 100.0),
            "da_5": float(da_matrix[:, 0].mean().item() * 100.0),
            "da_15": float(da_matrix[:, 1].mean().item() * 100.0),
            "da_30": float(da_matrix[:, 2].mean().item() * 100.0),
        }


def execute_training_step(
    batch: Any,
    world_model: LatentDynamicsWorldModel,
    head: MultiHorizonProjectionHead,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    trainable_params: List[nn.Parameter],
    device: torch.device,
    beta: float = 1.0,
    max_grad_norm: float = 1.0,
) -> Tuple[bool, Dict[str, float]]:
    """
    Executes a forward pass, dual-firewall gradient check, optimizer step,
    and returns decomposed loss and metrics.

    Dual Firewall:
    - Firewall 1: Check torch.isfinite(loss) before backward.
    - Firewall 2: Clip gradient norm (max_norm=1.0) and verify math.isfinite(grad_norm) before step.
    """
    obs = batch.obs_137.to(device=device, dtype=torch.float32)
    candles = batch.candle_seq_240.to(device=device, dtype=torch.float32)
    targets = batch.targets_3.to(device=device, dtype=torch.float32)

    # 1. Forward Pass
    z_out = world_model.encode_state(obs, candle_seq=candles)
    preds = head(z_out)

    loss_matrix = F.smooth_l1_loss(preds, targets, beta=beta, reduction="none")
    total_loss = loss_matrix.mean()

    # FIREWALL 1: Pre-Backward Finite Loss Guard
    if not torch.isfinite(total_loss):
        optimizer.zero_grad(set_to_none=True)
        return False, {"loss_total": float("nan"), "skipped": 1.0}

    # 2. Backward Pass
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    # FIREWALL 2: Gradient Norm Clipping & Finite Guard
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    if not math.isfinite(grad_norm_val):
        optimizer.zero_grad(set_to_none=True)
        return False, {"loss_total": total_loss.item(), "grad_norm": grad_norm_val, "skipped": 1.0}

    # 3. Optimization Step
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    clean_pred = torch.nan_to_num(preds, nan=0.0, posinf=50.0, neginf=-50.0)
    clean_tgt = torch.nan_to_num(targets, nan=0.0, posinf=50.0, neginf=-50.0)
    metrics = compute_metrics(clean_pred, clean_tgt, loss_matrix)
    metrics["grad_norm"] = grad_norm_val
    metrics["lr_primary"] = optimizer.param_groups[0]["lr"]
    metrics["skipped"] = 0.0

    return True, metrics


# =============================================================================
# 6. STRUCTURED TELEMETRY & LOGGER
# =============================================================================


class TrainingLogger:
    """Manages line-delimited JSONL metrics logging and console summary tables."""

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "train_metrics.jsonl"
        self.summary_path = self.log_dir / "training_logs.json"

    def log_step(
        self,
        phase: int,
        epoch: int,
        step: int,
        global_step: int,
        metrics: Dict[str, float],
        lrs: Dict[str, float],
        elapsed_sec: float,
    ) -> None:
        """Appends a single step record to train_metrics.jsonl."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "metrics": metrics,
            "learning_rates": lrs,
            "elapsed_sec": round(elapsed_sec, 3),
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def print_epoch_summary(
        self,
        phase: int,
        epoch: int,
        avg_metrics: Dict[str, float],
        lrs: Dict[str, float],
        duration_sec: float,
    ) -> None:
        """Prints formatted console table at epoch completion."""
        print("\n" + "=" * 90)
        print(f"| EPOCH {epoch:02d} SUMMARY [PHASE {phase}] | Duration: {duration_sec:.1f}s")
        print("-" * 90)
        print(
            f"  Loss (Total/5/15/30) : {avg_metrics['loss_total']:.4f} / "
            f"{avg_metrics['loss_5']:.4f} / {avg_metrics['loss_15']:.4f} / {avg_metrics['loss_30']:.4f}"
        )
        print(
            f"  MAE  (Total/5/15/30) : {avg_metrics['mae_total']:.4f} / "
            f"{avg_metrics['mae_5']:.4f} / {avg_metrics['mae_15']:.4f} / {avg_metrics['mae_30']:.4f}"
        )
        print(
            f"  DA%  (Total/5/15/30) : {avg_metrics['da_total']:.2f}% / "
            f"{avg_metrics['da_5']:.2f}% / {avg_metrics['da_15']:.2f}% / {avg_metrics['da_30']:.2f}%"
        )
        lr_str = " | ".join(f"{k}: {v:.2e}" for k, v in lrs.items())
        print(f"  Learning Rates       : {lr_str}")
        print(f"  Mean Grad Norm       : {avg_metrics.get('grad_norm', 0.0):.4f}")
        print("=" * 90)


# =============================================================================
# 7. CLI ARGUMENT PARSER
# =============================================================================


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command line arguments with defaults and overrides."""
    parser = argparse.ArgumentParser(
        description="Adaptive Training Runner for 240-Candle TCA Multi-Horizon Predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Epochs & Batching
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Number of Phase 1 pre-training epochs")
    parser.add_argument("--phase2-epochs", type=int, default=10, help="Number of Phase 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size across regimes")
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=-1,
        help="Number of batches per epoch (-1 for full dataset pass)",
    )

    # Learning Rates & Optimizer
    parser.add_argument("--lr", type=float, default=1e-4, help="Phase 1 temporal encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=3e-4, help="Phase 1 projection head learning rate")
    parser.add_argument("--encoder-lr", type=float, default=1e-5, help="Phase 2 base encoder learning rate")
    parser.add_argument(
        "--phase2-temporal-lr", type=float, default=5e-5, help="Phase 2 temporal encoder learning rate"
    )
    parser.add_argument("--phase2-head-lr", type=float, default=1e-4, help="Phase 2 projection head learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Maximum gradient norm clipping threshold")

    # Execution & Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic execution")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/checkpoints", help="Output directory for checkpoints and logs"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Epoch interval for saving checkpoints")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Fast verification mode (1 epoch, 10 batches per phase)",
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="stratified",
        choices=["stratified", "regime_batch", "round_robin"],
        help="Regime routing mode for Mega-Pool batches",
    )

    # Data & Paths
    parser.add_argument(
        "--data-path",
        "--csv-path",
        dest="data_path",
        type=str,
        default=r"C:\Users\Admin\Desktop\BTCUSDm.csv",
        help="Path to authentic BTCUSDm.csv dataset",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="outputs/cache", help="Directory containing precomputed .npz pool caches"
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt",
        help="Optional initial checkpoint to load pre-trained weights",
    )
    parser.add_argument("--log-interval", type=int, default=50, help="Step interval for live progress logging")

    parsed = parser.parse_args(args)

    # Dry-run parameter adjustments
    if parsed.dry_run:
        parsed.phase1_epochs = 1
        parsed.phase2_epochs = 1
        if parsed.batches_per_epoch <= 0 or parsed.batches_per_epoch > 10:
            parsed.batches_per_epoch = 10

    return parsed


# =============================================================================
# 8. CORE TRAINING PIPELINE RUNNER
# =============================================================================


def run_training(
    args: argparse.Namespace,
    router: Optional[MegaPoolRouter] = None,
) -> Dict[str, Any]:
    """
    Executes the full adaptive training runner across Phase 1 and Phase 2.
    Returns dictionary with training results, metrics, and checkpoint paths.
    """
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(output_dir)

    print(f"\n[TCA Multi-Horizon Runner] Device: {device} | Seed: {args.seed} | Dry Run: {args.dry_run}")
    print(f"[TCA Multi-Horizon Runner] Output Directory: {output_dir}")

    # 1. Initialize MegaPoolRouter & Validate OOS Firewall
    if router is None:
        cfg = MegaPoolConfig(
            csv_path=args.data_path,
            cache_dir=args.cache_dir,
            use_cache=True,
            lookback_candles=240,
            horizons=(5, 15, 30),
            atr_floor=1e-4,
            clip_range=(-10.0, 10.0),
        )
        router = MegaPoolRouter(config=cfg)

    # Assert programmatic OOS firewall: zero data leakage
    router.validate_zero_oos_leakage()
    print("[TCA Multi-Horizon Runner] MegaPoolRouter initialized and OOS Firewall validated 100%.")

    # 2. Instantiate Models
    world_model = LatentDynamicsWorldModel().to(device)
    reasoning_engine = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768)).to(device)

    # Load initial challenger checkpoint if available
    init_ckpt_path = Path(args.init_checkpoint)
    if init_ckpt_path.exists():
        print(f"[TCA Multi-Horizon Runner] Loading initial weights from {init_ckpt_path}...")
        ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
        if "world_model_state_dict" in ckpt:
            world_model.load_state_dict(ckpt["world_model_state_dict"], strict=True)
        re_state = ckpt.get("reasoning_engine_state_dict", ckpt.get("bull_cot_state_dict", None))
        if re_state is not None:
            reasoning_engine.load_state_dict(re_state, strict=True)
        print("[TCA Multi-Horizon Runner] Successfully loaded baseline weights with strict=True.")

    # Enable Temporal Stream on World Model
    world_model.enable_temporal_stream(candle_len=240)
    world_model.to(device)

    # Attach Multi-Horizon Projection Head
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.05).to(device)

    # Warm-start temporal encoder and projection head if present in checkpoint
    if init_ckpt_path.exists() and "ckpt" in locals():
        if "temporal_encoder_state_dict" in ckpt and world_model.temporal_encoder is not None:
            world_model.temporal_encoder.load_state_dict(ckpt["temporal_encoder_state_dict"], strict=True)
            print("[TCA Multi-Horizon Runner] Successfully warm-started TemporalCrossAttentionEncoder weights!")
        if "projection_head_state_dict" in ckpt:
            head.load_state_dict(ckpt["projection_head_state_dict"], strict=True)
            print("[TCA Multi-Horizon Runner] Successfully warm-started MultiHorizonProjectionHead weights!")

    # Freeze Reasoning Engine strictly (191 keys, 4,196,564 weights)
    reasoning_engine.eval()
    for p in reasoning_engine.parameters():
        p.requires_grad = False

    initial_re_hash = hash_reasoning_engine(reasoning_engine)
    initial_re_snapshot = {k: v.clone().detach().cpu() for k, v in reasoning_engine.state_dict().items()}
    print(f"[TCA Multi-Horizon Runner] Reasoning Engine Initial SHA-256: {initial_re_hash}")
    assert len(reasoning_engine.state_dict()) == 191
    assert sum(p.numel() for p in reasoning_engine.parameters()) == 4196564

    # Determine steps per epoch
    dataset_len = len(router.mega_dataset)
    if args.batches_per_epoch > 0:
        steps_per_epoch = args.batches_per_epoch
    else:
        steps_per_epoch = max(1, dataset_len // args.batch_size)

    routing_mode = RoutingMode(args.routing_mode.lower())
    global_step = 0
    epoch_summaries: List[Dict[str, Any]] = []

    # =========================================================================
    # PHASE 1: PRE-TRAINING (Frozen Base Encoder & Frozen Reasoning Engine)
    # =========================================================================
    print(f"\n{'=' * 90}\n>>> STARTING PHASE 1 PRE-TRAINING ({args.phase1_epochs} Epochs)\n{'=' * 90}")

    # Freeze base encoder
    for p in world_model.encoder.parameters():
        p.requires_grad = False
    for p in world_model.temporal_encoder.parameters():
        p.requires_grad = True
    for p in head.parameters():
        p.requires_grad = True

    p1_trainable = [p for p in world_model.temporal_encoder.parameters() if p.requires_grad] + [
        p for p in head.parameters() if p.requires_grad
    ]
    p1_param_count = sum(p.numel() for p in p1_trainable)
    assert p1_param_count == 2072707, f"Expected 2,072,707 Phase 1 params, got {p1_param_count}"
    print(f"[Phase 1] Trainable Parameters: {p1_param_count:,} (temporal_encoder + head)")

    p1_opt = build_phase1_optimizer(
        world_model.temporal_encoder,
        head,
        lr_temporal=args.lr,
        lr_head=args.head_lr,
        weight_decay=args.weight_decay,
    )
    p1_total_steps = args.phase1_epochs * steps_per_epoch
    p1_sched = build_lr_scheduler(p1_opt, total_steps=p1_total_steps, warmup_ratio=0.05, eta_min=1e-6)

    phase1_ckpt_path = output_dir / "tca_phase1_pretrained.pt"
    avg_metrics: Dict[str, float] = {}

    for epoch in range(1, args.phase1_epochs + 1):
        epoch_start = time.time()
        dataloader = router.create_dataloader(
            batch_size=args.batch_size, mode=routing_mode, num_batches=steps_per_epoch
        )
        step_metrics_list: List[Dict[str, float]] = []

        for step, batch in enumerate(dataloader, start=1):
            global_step += 1
            step_start = time.time()

            success, metrics = execute_training_step(
                batch=batch,
                world_model=world_model,
                head=head,
                optimizer=p1_opt,
                scheduler=p1_sched,
                trainable_params=p1_trainable,
                device=device,
                beta=1.0,
                max_grad_norm=args.clip_grad_norm,
            )

            # Runtime Assertion: Verify Base Encoder received NO gradients
            assert all(
                p.grad is None for p in world_model.encoder.parameters()
            ), "Base MarketStateEncoder received gradients in Phase 1!"

            # Runtime Assertion: Verify Reasoning Engine received NO gradients
            assert all(
                p.grad is None for p in reasoning_engine.parameters()
            ), "Reasoning Engine received gradients in Phase 1!"

            if success:
                step_metrics_list.append(metrics)

            lrs = {g["name"]: g["lr"] for g in p1_opt.param_groups}
            logger.log_step(
                phase=1,
                epoch=epoch,
                step=step,
                global_step=global_step,
                metrics=metrics,
                lrs=lrs,
                elapsed_sec=time.time() - step_start,
            )

            if step % args.log_interval == 0 or step == steps_per_epoch:
                current_loss = metrics.get("loss_total", 0.0)
                current_da = metrics.get("da_total", 0.0)
                print(
                    f"[Phase 1] Epoch {epoch:02d}/{args.phase1_epochs:02d} | "
                    f"Step {step:04d}/{steps_per_epoch:04d} | "
                    f"Loss: {current_loss:.4f} | DA: {current_da:.1f}% | "
                    f"LR_tca: {lrs.get('temporal_encoder', 0.0):.2e}",
                    end="\r",
                    flush=True,
                )

        # Epoch Aggregations
        epoch_duration = time.time() - epoch_start
        avg_metrics: Dict[str, float] = {}
        if step_metrics_list:
            for k in step_metrics_list[0].keys():
                avg_metrics[k] = float(np.mean([m[k] for m in step_metrics_list if not math.isnan(m[k])]))

        lrs = {g["name"]: g["lr"] for g in p1_opt.param_groups}
        logger.print_epoch_summary(1, epoch, avg_metrics, lrs, epoch_duration)
        epoch_summaries.append({"phase": 1, "epoch": epoch, "metrics": avg_metrics, "lrs": lrs})

    # Post-Phase 1 Invariant Assertions: Reasoning Engine Zero Drift
    assert all(
        p.grad is None for p in reasoning_engine.parameters()
    ), "Zero Drift Violation: Reasoning Engine grad is not None!"
    max_drift_p1 = max(
        (v.cpu() - initial_re_snapshot[k]).abs().max().item() for k, v in reasoning_engine.state_dict().items()
    )
    assert max_drift_p1 == 0.0, f"Zero Drift Violation: max drift = {max_drift_p1} > 0.0"
    current_re_hash = hash_reasoning_engine(reasoning_engine)
    assert (
        current_re_hash == initial_re_hash
    ), f"Zero Drift SHA-256 mismatch! Init: {initial_re_hash}, Current: {current_re_hash}"
    print(f"[Phase 1 Verified] Reasoning Engine SHA-256 Invariant Preserved: {current_re_hash}")

    if args.phase1_epochs > 0:
        # Serialize Phase 1 Checkpoint
        p1_meta = {
            "phase": "phase1_pretrained",
            "epochs": args.phase1_epochs,
            "metrics": avg_metrics,
            "re_sha256": current_re_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        p1_payload = extract_4_namespace_checkpoint(world_model, head, reasoning_engine, p1_meta)
        save_atomic_checkpoint(p1_payload, phase1_ckpt_path)
        print(f"[Phase 1 Checkpoint] Saved to {phase1_ckpt_path} with 4 isolated namespaces.")

    # =========================================================================
    # PHASE 2: JOINT FINE-TUNING (Unfrozen Base Encoder & Frozen Reasoning Engine)
    # =========================================================================
    print(f"\n{'=' * 90}\n>>> STARTING PHASE 2 FINE-TUNING ({args.phase2_epochs} Epochs)\n{'=' * 90}")

    # Unfreeze base encoder
    for p in world_model.encoder.parameters():
        p.requires_grad = True
    for p in world_model.temporal_encoder.parameters():
        p.requires_grad = True
    for p in head.parameters():
        p.requires_grad = True
    for p in reasoning_engine.parameters():
        p.requires_grad = False

    p2_trainable = (
        [p for p in world_model.encoder.parameters() if p.requires_grad]
        + [p for p in world_model.temporal_encoder.parameters() if p.requires_grad]
        + [p for p in head.parameters() if p.requires_grad]
    )
    p2_param_count = sum(p.numel() for p in p2_trainable)
    assert p2_param_count == 2539907, f"Expected 2,539,907 Phase 2 params, got {p2_param_count}"
    print(f"[Phase 2] Trainable Parameters: {p2_param_count:,} (encoder + temporal_encoder + head)")

    p2_opt = build_phase2_optimizer(
        world_model.encoder,
        world_model.temporal_encoder,
        head,
        lr_encoder=args.encoder_lr,
        lr_temporal=args.phase2_temporal_lr,
        lr_head=args.phase2_head_lr,
        weight_decay=args.weight_decay,
    )
    p2_total_steps = args.phase2_epochs * steps_per_epoch
    p2_sched = build_lr_scheduler(p2_opt, total_steps=p2_total_steps, warmup_ratio=0.05, eta_min=1e-6)

    phase2_ckpt_path = output_dir / "tca_phase2_finetuned.pt"

    for epoch in range(1, args.phase2_epochs + 1):
        epoch_start = time.time()
        dataloader = router.create_dataloader(
            batch_size=args.batch_size, mode=routing_mode, num_batches=steps_per_epoch
        )
        step_metrics_list = []

        for step, batch in enumerate(dataloader, start=1):
            global_step += 1
            step_start = time.time()

            success, metrics = execute_training_step(
                batch=batch,
                world_model=world_model,
                head=head,
                optimizer=p2_opt,
                scheduler=p2_sched,
                trainable_params=p2_trainable,
                device=device,
                beta=1.0,
                max_grad_norm=args.clip_grad_norm,
            )

            # Runtime Assertion: Base Encoder must receive gradients in Phase 2
            assert any(
                p.grad is not None for p in world_model.encoder.parameters()
            ), "Base encoder missed gradients in Phase 2!"

            # Runtime Assertion: Reasoning Engine must NEVER receive gradients
            assert all(
                p.grad is None for p in reasoning_engine.parameters()
            ), "Reasoning Engine received gradients in Phase 2!"

            if success:
                step_metrics_list.append(metrics)

            lrs = {g["name"]: g["lr"] for g in p2_opt.param_groups}
            logger.log_step(
                phase=2,
                epoch=epoch,
                step=step,
                global_step=global_step,
                metrics=metrics,
                lrs=lrs,
                elapsed_sec=time.time() - step_start,
            )

            if step % args.log_interval == 0 or step == steps_per_epoch:
                current_loss = metrics.get("loss_total", 0.0)
                current_da = metrics.get("da_total", 0.0)
                print(
                    f"[Phase 2] Epoch {epoch:02d}/{args.phase2_epochs:02d} | "
                    f"Step {step:04d}/{steps_per_epoch:04d} | "
                    f"Loss: {current_loss:.4f} | DA: {current_da:.1f}% | "
                    f"LR_enc: {lrs.get('encoder', 0.0):.2e} | LR_tca: {lrs.get('temporal_encoder', 0.0):.2e}",
                    end="\r",
                    flush=True,
                )

        # Epoch Aggregations
        epoch_duration = time.time() - epoch_start
        avg_metrics = {}
        if step_metrics_list:
            for k in step_metrics_list[0].keys():
                avg_metrics[k] = float(np.mean([m[k] for m in step_metrics_list if not math.isnan(m[k])]))

        lrs = {g["name"]: g["lr"] for g in p2_opt.param_groups}
        logger.print_epoch_summary(2, epoch, avg_metrics, lrs, epoch_duration)
        epoch_summaries.append({"phase": 2, "epoch": epoch, "metrics": avg_metrics, "lrs": lrs})

    # Post-Phase 2 Invariant Assertions: Reasoning Engine Zero Drift
    assert all(
        p.grad is None for p in reasoning_engine.parameters()
    ), "Zero Drift Violation: Reasoning Engine grad is not None!"
    max_drift_p2 = max(
        (v.cpu() - initial_re_snapshot[k]).abs().max().item() for k, v in reasoning_engine.state_dict().items()
    )
    assert max_drift_p2 == 0.0, f"Zero Drift Violation: max drift = {max_drift_p2} > 0.0"
    current_re_hash = hash_reasoning_engine(reasoning_engine)
    assert (
        current_re_hash == initial_re_hash
    ), f"Zero Drift SHA-256 mismatch! Init: {initial_re_hash}, Current: {current_re_hash}"
    print(f"[Phase 2 Verified] Reasoning Engine SHA-256 Invariant Preserved: {current_re_hash}")

    # Serialize Phase 2 Checkpoint
    p2_meta = {
        "phase": "phase2_finetuned",
        "epochs": args.phase2_epochs,
        "metrics": avg_metrics,
        "re_sha256": current_re_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p2_payload = extract_4_namespace_checkpoint(world_model, head, reasoning_engine, p2_meta)
    save_atomic_checkpoint(p2_payload, phase2_ckpt_path)
    print(f"[Phase 2 Checkpoint] Saved to {phase2_ckpt_path} with 4 isolated namespaces.")

    # Write Complete Training Summary JSON
    training_summary = {
        "status": "COMPLETED",
        "device": str(device),
        "seed": args.seed,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "phase1_checkpoint": str(phase1_ckpt_path),
        "phase2_checkpoint": str(phase2_ckpt_path),
        "re_initial_sha256": initial_re_hash,
        "re_final_sha256": current_re_hash,
        "re_drift_max": max(max_drift_p1, max_drift_p2),
        "history": epoch_summaries,
    }
    summary_path = output_dir / "training_logs.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print(f"[Training Summary] Written to {summary_path}")

    return training_summary


def main() -> None:
    """CLI Entry Point."""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
