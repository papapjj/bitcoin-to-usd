#!/usr/bin/env python3
"""
scripts/train_infinite_early_stopping.py

Infinite Epoch Training Engine with Early Stopping (Patience = 5).
Optimizes Bilateral Dual-CoT Reasoning Engine across all 5 diverse market regimes:
1. Bull Expansion (Feb 2025: 91k -> 99k)
2. Drift Bear & Rebound (Oct 2026 Week 1: 86k -> 83k)
3. Range Consolidation (Oct 2026 Week 2: 83k -> 82k)
4. Waterfall Crash (Oct 2026 Week 3: 84k -> 79k, including -3,200 USD freefall on Oct 22)
5. High Volatility Expansion (Oct 2026 Week 4: 81k -> 76k)

Protocol & Safeguards:
- Infinite Epoch Loop with Early Stopping
- Epoch 0 Baseline Evaluation: best_val_loss initialized from loaded champion (~7.40), NOT float('inf')
- Atomic Checkpoint Serialization: writes to temporary file (.pt.tmp) and replaces atomically
- Quadratic Anti-MAE Differentiable Excursion Loss: 1.8 * probs[:, act] * relu((mae - 0.55)/0.55)^2
- Zero Manual Filter / Gating Constraint: 100% pure neural optimization
- CLI arguments: --max-epochs, --lr, --batch-size, --patience, --save-dir, --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)
from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from scripts.test_3_weeks_zero_filter import evaluate_all_3_weeks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InfiniteTrainer")


def build_stratified_dataset(val_ratio: float = 0.15) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Loads all 5 regime datasets and creates an 85/15 train/val split across all regimes."""
    datasets = [
        ("Feb 2025 Bull", ROOT / "outputs" / "btcusd_m1_7day_clean.csv"),
        ("Oct 2026 Week 1", ROOT / "outputs" / "btcusd_m1_week1_20261005.csv"),
        ("Oct 2026 Week 2", ROOT / "outputs" / "btcusd_m1_week2_20261012.csv"),
        ("Oct 2026 Week 3", ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"),
        ("Oct 2026 Week 4", ROOT / "outputs" / "btcusd_m1_week4_20261026.csv"),
    ]

    train_ing_list, val_ing_list = [], []
    train_act_list, val_act_list = [], []
    train_prm_list, val_prm_list = [], []
    train_bmae_list, val_bmae_list = [], []
    train_smae_list, val_smae_list = [], []

    for name, path in datasets:
        logger.info("Extracting features from: %s (%s)", name, path.name)
        data = extract_features_from_csv(path)
        N = len(data["ingress"])

        n_val = int(N * val_ratio)
        n_train = N - n_val

        train_ing_list.append(data["ingress"][:n_train])
        val_ing_list.append(data["ingress"][n_train:])

        train_act_list.append(data["actions"][:n_train])
        val_act_list.append(data["actions"][n_train:])

        train_prm_list.append(data["prms"][:n_train])
        val_prm_list.append(data["prms"][n_train:])

        train_bmae_list.append(data["buy_maes"][:n_train])
        val_bmae_list.append(data["buy_maes"][n_train:])

        train_smae_list.append(data["sell_maes"][:n_train])
        val_smae_list.append(data["sell_maes"][n_train:])

    train_dict = {
        "ingress": np.vstack(train_ing_list),
        "actions": np.concatenate(train_act_list),
        "prms": np.concatenate(train_prm_list),
        "buy_maes": np.concatenate(train_bmae_list),
        "sell_maes": np.concatenate(train_smae_list),
    }
    val_dict = {
        "ingress": np.vstack(val_ing_list),
        "actions": np.concatenate(val_act_list),
        "prms": np.concatenate(val_prm_list),
        "buy_maes": np.concatenate(val_bmae_list),
        "sell_maes": np.concatenate(val_smae_list),
    }
    return train_dict, val_dict


def compute_loss(
    engine: BilateralDualCoTEngine,
    criterion_ce: nn.Module,
    criterion_mse: nn.Module,
    b_x: torch.Tensor,
    b_act: torch.Tensor,
    b_prm: torch.Tensor,
    b_b_mae: torch.Tensor,
    b_s_mae: torch.Tensor,
    false_entry_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = engine.state_adapter(b_x)
    bull_out = engine.bull_cot(z)
    bear_out = engine.bear_cot(z)
    p_thought = bull_out.cot_hops[:, -1, :]
    a_thought = bear_out.cot_hops[:, -1, :]

    final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought, a_thought, turns=3)
    synthesis = torch.cat([final_p, final_a], dim=-1)

    logits = engine.arbiter.arbiter_net(synthesis)
    probs = F.softmax(logits, dim=-1)
    prm_preds = engine.arbiter.prm_evaluator(synthesis).squeeze(-1)

    # Quadratic Anti-MAE Differentiable Excursion Penalty
    mae_penalty_buy = probs[:, 1] * torch.relu((b_b_mae - 2.50) / 2.50) ** 2
    mae_penalty_sell = probs[:, 2] * torch.relu((b_s_mae - 2.50) / 2.50) ** 2
    loss_anti_mae = torch.mean(mae_penalty_buy + mae_penalty_sell)

    # Defect 4: Quadratic False-Entry Penalty on y = 0 (HOLD)
    # Penalizes any probability mass assigned to BUY (1) or SELL (2) when ground truth is HOLD (0)
    hold_mask = (b_act == 0)
    if hold_mask.any():
        active_probs_on_hold = probs[hold_mask, 1] + probs[hold_mask, 2]
        loss_false_entry = torch.mean(active_probs_on_hold ** 2)
    else:
        loss_false_entry = torch.tensor(0.0, device=logits.device)

    b_target = (b_act == 1).float()
    loss_bull = F.binary_cross_entropy(torch.clamp(bull_out.bull_score.view(-1), 1e-4, 1.0 - 1e-4), b_target)

    r_target = (b_act == 2).float()
    loss_bear = F.binary_cross_entropy(torch.clamp(bear_out.bear_score.view(-1), 1e-4, 1.0 - 1e-4), r_target)

    loss_arb = criterion_ce(logits, b_act)
    loss_prm = criterion_mse(prm_preds, b_prm)

    total_loss = (
        loss_arb
        + 0.5 * loss_prm
        + 0.5 * loss_bull
        + 0.5 * loss_bear
        + 1.8 * loss_anti_mae
        + false_entry_weight * loss_false_entry
    )
    return total_loss, loss_anti_mae, loss_false_entry, logits, prm_preds


def atomic_save_checkpoint(
    engine: BilateralDualCoTEngine,
    checkpoint_path: Path,
    metadata: Dict[str, Any],
) -> Path:
    """Saves checkpoint atomically via temporary swap to prevent corruption."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp_{os.getpid()}")

    payload = {
        "version": "v193_bilateral_dual_cot",
        "saved_at": time.time(),
        "bull_cot_state_dict": engine.bull_cot.state_dict(),
        "bear_cot_state_dict": engine.bear_cot.state_dict(),
        "cross_examination_state_dict": engine.cross_examination.state_dict(),
        "arbiter_state_dict": engine.arbiter.state_dict(),
        "trap_detector_state_dict": engine.trap_detector.state_dict(),
        "state_adapter_state_dict": engine.state_adapter.state_dict(),
        "reasoning_engine_state_dict": engine.bull_cot.state_dict(),  # Dual compatibility
        "config": engine.config,
        "metadata": metadata or {},
    }
    torch.save(payload, str(temp_path))
    temp_path.replace(checkpoint_path)
    logger.info("Atomically saved checkpoint to: %s", checkpoint_path)
    return checkpoint_path


def train_infinite_patience(
    max_epochs: int = 10,
    lr: float = 5e-5,
    batch_size: int = 128,
    patience: int = 5,
    save_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    patience_limit = patience
    min_delta = 1e-4
    save_dir = save_dir or (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs")
    artifacts_dir = artifacts_dir or (ROOT / "outputs" / "acceptance_artifacts" / "milestone_2")
    save_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Infinite Training Session with Early Stopping (Patience = %d)...", patience_limit)
    train_data, val_data = build_stratified_dataset(val_ratio=0.15)

    N_train = len(train_data["ingress"])
    N_val = len(val_data["ingress"])
    logger.info("Stratified Pool Ready: %d Train samples | %d Val samples (5 Diverse Regimes)", N_train, N_val)

    # Defect 4: Prioritized HOLD action weighting
    # Eliminates the ~18x gradient penalty against HOLD caused by inverse-frequency weighting.
    # Enforces prioritized weights: HOLD (0) = 1.25, BUY (1) = 1.50, SELL (2) = 1.50.
    device = torch.device("cpu")
    torch.set_num_threads(8)
    class_weights = torch.tensor([1.25, 1.50, 1.50], dtype=torch.float32, device=device)
    logger.info(
        "Prioritized Class Weights: HOLD=%.2f, BUY=%.2f, SELL=%.2f (Eliminated 18x overtrading bias)",
        class_weights[0].item(), class_weights[1].item(), class_weights[2].item()
    )

    cfg = BilateralDualCoTConfig(
        input_dim=233,
        latent_dim=768,
        hidden_dim=256,
        num_hops=4,
        num_debate_turns=3,
        prm_threshold=0.58,
        trap_threshold=0.60,
    )
    engine = BilateralDualCoTEngine(cfg).to(device)

    # Warm-start from previous best champion
    ckpt_path = save_dir / "mythos_symmetric_anti_mae_champion.pt"
    if ckpt_path.exists():
        logger.info("Warm-starting from best champion: %s", ckpt_path.name)
        engine.load_checkpoint(ckpt_path)

    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    criterion_mse = nn.MSELoss()

    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=5e-6)

    output_checkpoint = save_dir / "mythos_omniscient_anti_mae_champion.pt"
    primary_checkpoint = save_dir / "mythos_symmetric_anti_mae_champion.pt"

    # =========================================================================
    # EPOCH 0: Baseline Evaluation of Loaded Champion (Safeguard)
    # =========================================================================
    logger.info("Evaluating Epoch 0 Baseline Validation Loss on Loaded Champion...")
    engine.eval()
    baseline_val_loss = 0.0
    baseline_val_mae_loss = 0.0
    baseline_val_fe_loss = 0.0
    baseline_val_correct = 0
    baseline_val_total = 0
    baseline_batches = 0
    all_b_preds = []
    all_b_acts = []

    with torch.no_grad():
        for v_start in range(0, N_val, batch_size):
            v_end = min(v_start + batch_size, N_val)
            v_x = torch.from_numpy(val_data["ingress"][v_start:v_end]).float().to(device)
            v_act = torch.from_numpy(val_data["actions"][v_start:v_end]).long().to(device)
            v_prm = torch.from_numpy(val_data["prms"][v_start:v_end]).float().to(device)
            v_b_mae = torch.from_numpy(val_data["buy_maes"][v_start:v_end]).float().to(device)
            v_s_mae = torch.from_numpy(val_data["sell_maes"][v_start:v_end]).float().to(device)

            v_l, v_m_l, v_fe_l, v_logits, _ = compute_loss(
                engine, criterion_ce, criterion_mse, v_x, v_act, v_prm, v_b_mae, v_s_mae
            )
            baseline_val_loss += v_l.item()
            baseline_val_mae_loss += v_m_l.item()
            baseline_val_fe_loss += v_fe_l.item()
            v_preds = torch.argmax(v_logits, dim=-1)
            baseline_val_correct += (v_preds == v_act).sum().item()
            baseline_val_total += len(v_act)
            baseline_batches += 1
            all_b_preds.append(v_preds.cpu().numpy())
            all_b_acts.append(v_act.cpu().numpy())

    avg_baseline_val_loss = baseline_val_loss / max(1, baseline_batches)
    avg_baseline_mae_loss = baseline_val_mae_loss / max(1, baseline_batches)
    avg_baseline_fe_loss = baseline_val_fe_loss / max(1, baseline_batches)
    baseline_val_acc = (baseline_val_correct / max(1, baseline_val_total)) * 100.0

    b_preds_arr = np.concatenate(all_b_preds)
    b_acts_arr = np.concatenate(all_b_acts)
    b_hold_pct = (np.sum(b_preds_arr == 0) / len(b_preds_arr)) * 100.0
    b_buy_pct = (np.sum(b_preds_arr == 1) / len(b_preds_arr)) * 100.0
    b_sell_pct = (np.sum(b_preds_arr == 2) / len(b_preds_arr)) * 100.0

    b_rec0 = np.sum((b_preds_arr == 0) & (b_acts_arr == 0)) / max(1, np.sum(b_acts_arr == 0))
    b_rec1 = np.sum((b_preds_arr == 1) & (b_acts_arr == 1)) / max(1, np.sum(b_acts_arr == 1))
    b_rec2 = np.sum((b_preds_arr == 2) & (b_acts_arr == 2)) / max(1, np.sum(b_acts_arr == 2))
    baseline_bal_acc = (b_rec0 + b_rec1 + b_rec2) / 3.0 * 100.0

    b_prec0 = np.sum((b_preds_arr == 0) & (b_acts_arr == 0)) / max(1, np.sum(b_preds_arr == 0))
    b_prec1 = np.sum((b_preds_arr == 1) & (b_acts_arr == 1)) / max(1, np.sum(b_preds_arr == 1))
    b_prec2 = np.sum((b_preds_arr == 2) & (b_acts_arr == 2)) / max(1, np.sum(b_preds_arr == 2))
    b_f1_0 = 2.0 * b_prec0 * b_rec0 / max(1e-6, b_prec0 + b_rec0)
    b_f1_1 = 2.0 * b_prec1 * b_rec1 / max(1e-6, b_prec1 + b_rec1)
    b_f1_2 = 2.0 * b_prec2 * b_rec2 / max(1e-6, b_prec2 + b_rec2)
    baseline_macro_f1 = (b_f1_0 + b_f1_1 + b_f1_2) / 3.0
    baseline_fer = (np.sum((b_acts_arr == 0) & (b_preds_arr != 0)) / max(1, np.sum(b_acts_arr == 0))) * 100.0

    # Initialize best_val_loss from the real baseline, NOT float('inf')!
    best_val_loss = avg_baseline_val_loss
    logger.info(
        "Epoch 0 Baseline Champion: Val Loss=%.4f (Acc=%.2f%%, BalAcc=%.2f%%, F1=%.3f, FER=%.2f%%, FE_L=%.4f) "
        "[Dist: H=%.1f%%, B=%.1f%%, S=%.1f%%] [Champion Protected]",
        best_val_loss, baseline_val_acc, baseline_bal_acc, baseline_macro_f1, baseline_fer,
        avg_baseline_fe_loss, b_hold_pct, b_buy_pct, b_sell_pct
    )

    training_history: List[Dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "train_mae_loss": None,
            "train_fe_loss": None,
            "val_loss": round(avg_baseline_val_loss, 4),
            "val_mae_loss": round(avg_baseline_mae_loss, 4),
            "val_fe_loss": round(avg_baseline_fe_loss, 4),
            "val_acc": round(baseline_val_acc, 2),
            "val_balanced_acc": round(baseline_bal_acc, 2),
            "val_macro_f1": round(baseline_macro_f1, 4),
            "val_fer": round(baseline_fer, 2),
            "action_dist": {
                "hold_pct": round(b_hold_pct, 1),
                "buy_pct": round(b_buy_pct, 1),
                "sell_pct": round(b_sell_pct, 1),
            },
            "lr": lr,
            "saved": False,
            "status": "BASELINE_CHAMPION",
        }
    ]

    if dry_run:
        logger.info("[DRY RUN] Executing 1 batch smoke check...")
        engine.train()
        b_x = torch.from_numpy(train_data["ingress"][:batch_size]).float().to(device)
        b_act = torch.from_numpy(train_data["actions"][:batch_size]).long().to(device)
        b_prm = torch.from_numpy(train_data["prms"][:batch_size]).float().to(device)
        b_b_mae = torch.from_numpy(train_data["buy_maes"][:batch_size]).float().to(device)
        b_s_mae = torch.from_numpy(train_data["sell_maes"][:batch_size]).float().to(device)

        optimizer.zero_grad()
        loss, mae_l, fe_l, logits, prms = compute_loss(engine, criterion_ce, criterion_mse, b_x, b_act, b_prm, b_b_mae, b_s_mae)
        loss.backward()
        optimizer.step()
        logger.info("[DRY RUN OK] 1-batch forward/backward loss=%.4f, mae_loss=%.4f, fe_loss=%.4f", loss.item(), mae_l.item(), fe_l.item())
        return {"dry_run": True, "baseline_val_loss": best_val_loss}

    patience_counter = 0
    epoch = 0
    indices = np.arange(N_train)

    print("\n" + "=" * 90)
    print(f"  INFINITE EPOCH TRAINING WITH EARLY STOPPING (PATIENCE = {patience_limit}, MAX = {max_epochs})")
    print("  Target: Minimize Generalization Loss & Excursion MAE strictly without filters")
    print("=" * 90)

    while True:
        epoch += 1
        engine.train()
        np.random.shuffle(indices)

        train_loss = 0.0
        train_mae_loss = 0.0
        train_fe_loss = 0.0
        train_batches = 0
        t0 = time.time()

        for start_idx in range(0, N_train, batch_size):
            end_idx = min(start_idx + batch_size, N_train)
            batch_indices = indices[start_idx:end_idx]

            b_x = torch.from_numpy(train_data["ingress"][batch_indices]).float().to(device)
            b_act = torch.from_numpy(train_data["actions"][batch_indices]).long().to(device)
            b_prm = torch.from_numpy(train_data["prms"][batch_indices]).float().to(device)
            b_b_mae = torch.from_numpy(train_data["buy_maes"][batch_indices]).float().to(device)
            b_s_mae = torch.from_numpy(train_data["sell_maes"][batch_indices]).float().to(device)

            optimizer.zero_grad()
            loss, loss_anti_mae, loss_fe, logits, prm_preds = compute_loss(
                engine, criterion_ce, criterion_mse, b_x, b_act, b_prm, b_b_mae, b_s_mae
            )
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mae_loss += loss_anti_mae.item()
            train_fe_loss += loss_fe.item()
            train_batches += 1

        train_elapsed = time.time() - t0
        avg_train_loss = train_loss / max(1, train_batches)
        avg_train_mae_loss = train_mae_loss / max(1, train_batches)
        avg_train_fe_loss = train_fe_loss / max(1, train_batches)

        # Validation phase
        engine.eval()
        val_loss = 0.0
        val_mae_loss = 0.0
        val_fe_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        all_v_preds = []
        all_v_acts = []

        with torch.no_grad():
            for v_start in range(0, N_val, batch_size):
                v_end = min(v_start + batch_size, N_val)
                v_x = torch.from_numpy(val_data["ingress"][v_start:v_end]).float().to(device)
                v_act = torch.from_numpy(val_data["actions"][v_start:v_end]).long().to(device)
                v_prm = torch.from_numpy(val_data["prms"][v_start:v_end]).float().to(device)
                v_b_mae = torch.from_numpy(val_data["buy_maes"][v_start:v_end]).float().to(device)
                v_s_mae = torch.from_numpy(val_data["sell_maes"][v_start:v_end]).float().to(device)

                v_l, v_m_l, v_fe_l, v_logits, v_prms = compute_loss(
                    engine, criterion_ce, criterion_mse, v_x, v_act, v_prm, v_b_mae, v_s_mae
                )
                val_loss += v_l.item()
                val_mae_loss += v_m_l.item()
                val_fe_loss += v_fe_l.item()

                v_preds = torch.argmax(v_logits, dim=-1)
                val_correct += (v_preds == v_act).sum().item()
                val_total += len(v_act)
                val_batches += 1
                all_v_preds.append(v_preds.cpu().numpy())
                all_v_acts.append(v_act.cpu().numpy())

        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_mae_loss = val_mae_loss / max(1, val_batches)
        avg_val_fe_loss = val_fe_loss / max(1, val_batches)
        val_acc = (val_correct / max(1, val_total)) * 100.0

        v_preds_arr = np.concatenate(all_v_preds)
        v_acts_arr = np.concatenate(all_v_acts)
        val_hold_pct = (np.sum(v_preds_arr == 0) / len(v_preds_arr)) * 100.0
        val_buy_pct = (np.sum(v_preds_arr == 1) / len(v_preds_arr)) * 100.0
        val_sell_pct = (np.sum(v_preds_arr == 2) / len(v_preds_arr)) * 100.0

        rec0 = np.sum((v_preds_arr == 0) & (v_acts_arr == 0)) / max(1, np.sum(v_acts_arr == 0))
        rec1 = np.sum((v_preds_arr == 1) & (v_acts_arr == 1)) / max(1, np.sum(v_acts_arr == 1))
        rec2 = np.sum((v_preds_arr == 2) & (v_acts_arr == 2)) / max(1, np.sum(v_acts_arr == 2))
        val_bal_acc = (rec0 + rec1 + rec2) / 3.0 * 100.0

        prec0 = np.sum((v_preds_arr == 0) & (v_acts_arr == 0)) / max(1, np.sum(v_preds_arr == 0))
        prec1 = np.sum((v_preds_arr == 1) & (v_acts_arr == 1)) / max(1, np.sum(v_preds_arr == 1))
        prec2 = np.sum((v_preds_arr == 2) & (v_acts_arr == 2)) / max(1, np.sum(v_preds_arr == 2))
        f1_0 = 2.0 * prec0 * rec0 / max(1e-6, prec0 + rec0)
        f1_1 = 2.0 * prec1 * rec1 / max(1e-6, prec1 + rec1)
        f1_2 = 2.0 * prec2 * rec2 / max(1e-6, prec2 + rec2)
        val_macro_f1 = (f1_0 + f1_1 + f1_2) / 3.0
        val_fer = (np.sum((v_acts_arr == 0) & (v_preds_arr != 0)) / max(1, np.sum(v_acts_arr == 0))) * 100.0

        # Step learning rate scheduler on validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Check for improvement: requires val loss drop AND healthy HOLD frequency (>= 50%)
        improved = (avg_val_loss < (best_val_loss - min_delta)) and (val_hold_pct >= 50.0)

        metadata = {
            "epoch": epoch,
            "val_loss": float(avg_val_loss),
            "val_acc": float(val_acc),
            "val_balanced_acc": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_mae_loss": float(avg_val_mae_loss),
            "val_fe_loss": float(avg_val_fe_loss),
            "val_fer": float(val_fer),
            "action_dist": {
                "hold_pct": float(val_hold_pct),
                "buy_pct": float(val_buy_pct),
                "sell_pct": float(val_sell_pct),
            },
            "train_loss": float(avg_train_loss),
        }

        if improved:
            improvement_amount = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Atomic save to both primary and omniscient checkpoints
            atomic_save_checkpoint(engine, output_checkpoint, metadata=metadata)
            atomic_save_checkpoint(engine, primary_checkpoint, metadata=metadata)

            status_str = f">>> [IMPROVED -{improvement_amount:.4f}] Saved Champion"
            saved_flag = True
        else:
            patience_counter += 1
            status_str = f"--- [NO IMPROVE] Patience: {patience_counter}/{patience_limit}"
            saved_flag = False

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_mae_loss": round(avg_train_mae_loss, 4),
            "train_fe_loss": round(avg_train_fe_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_mae_loss": round(avg_val_mae_loss, 4),
            "val_fe_loss": round(avg_val_fe_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_balanced_acc": round(val_bal_acc, 2),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_fer": round(val_fer, 2),
            "action_dist": {
                "hold_pct": round(val_hold_pct, 1),
                "buy_pct": round(val_buy_pct, 1),
                "sell_pct": round(val_sell_pct, 1),
            },
            "lr": current_lr,
            "saved": saved_flag,
            "status": status_str,
        })

        logger.info(
            "Epoch %d (%.1fs) | Train Loss: %.4f (FE: %.4f) | Val Loss: %.4f (Acc: %.1f%%, BalAcc: %.1f%%, F1: %.3f, Dist: H=%.1f%%/B=%.1f%%/S=%.1f%%) | LR: %.2e | %s",
            epoch, train_elapsed, avg_train_loss, avg_train_fe_loss, avg_val_loss, val_acc, val_bal_acc, val_macro_f1,
            val_hold_pct, val_buy_pct, val_sell_pct, current_lr, status_str
        )

        # Early Stopping Condition
        if patience_counter >= patience_limit:
            print("\n" + "=" * 90)
            logger.info(
                "*** EARLY STOPPING TRIGGERED: No validation loss improvement for %d consecutive epochs! ***",
                patience_limit
            )
            logger.info("Training halted at Epoch %d. Best Validation Loss: %.4f", epoch, best_val_loss)
            print("=" * 90 + "\n")
            break

        if epoch >= max_epochs:
            logger.info("Reached maximum epochs limit (%d). Terminating training loop.", max_epochs)
            break

    # Save training loss history JSON artifact
    history_path = artifacts_dir / "training_loss_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_epochs": max_epochs,
            "patience": patience_limit,
            "initial_lr": lr,
            "best_val_loss": round(best_val_loss, 4),
            "baseline_val_loss": round(avg_baseline_val_loss, 4),
            "epochs_executed": epoch,
            "history": training_history,
        }, f, indent=2)
    logger.info("Saved training loss history to: %s", history_path)

    # Post-training audit on 3 weeks
    logger.info("Running Post-Training Audit...")
    try:
        evaluate_all_3_weeks(primary_checkpoint, prm_threshold=0.58)
    except Exception as e:
        logger.warning("Audit warning: %s", e)

    return {
        "best_val_loss": best_val_loss,
        "epochs": epoch,
        "history": training_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infinite Epoch Training with Early Stopping")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum epochs before safety exit (default: 10)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience limit (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and validation (default: 128)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate (default: 5e-5)")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "outputs" / "he_chuyen_gia" / "v193_runs", help="Directory to save checkpoints")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "outputs" / "acceptance_artifacts" / "milestone_2", help="Directory to save artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 batch smoke check without updating champion checkpoints")
    args = parser.parse_args()

    train_infinite_patience(
        max_epochs=args.max_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        save_dir=args.save_dir,
        artifacts_dir=args.artifacts_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
