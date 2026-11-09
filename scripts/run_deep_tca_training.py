"""
Automated Orchestration Runner for 20-Epoch Deep Multi-Horizon TCA Training.
Executes:
1. Deep training of 20 epochs warm-started from tca_phase2_finetuned.pt across 509k bars.
2. Saves finalized high-convergence checkpoint: outputs/checkpoints/tca_deep_20epochs.pt.
3. Automatically runs OOS evaluation on February 2025 $12k crash slice (9,785 bars) with $2.50 spread.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    print("=" * 80)
    print(">>> LAUNCHING DEEP MULTI-HORIZON TCA TRAINING (20 EPOCHS)")
    print("=" * 80)
    t_start = time.time()

    init_ckpt = "outputs/checkpoints/tca_phase2_finetuned.pt"
    if not os.path.exists(init_ckpt):
        init_ckpt = "outputs/checkpoints/tca_phase1_pretrained.pt"

    cmd_train = [
        sys.executable,
        "scripts/train_tca_multi_horizon.py",
        "--phase1-epochs", "0",            # Skip phase 1 since we warm-start
        "--phase2-epochs", "20",           # 20 Deep Fine-Tuning Epochs
        "--batches-per-epoch", "50",       # 50 diverse multi-regime batches per epoch = 1,000 batches
        "--batch-size", "48",              # Optimized for CPU throughput
        "--encoder-lr", "5e-6",            # Gentle fine-tuning for base encoder
        "--phase2-temporal-lr", "3e-5",    # Stable convergence for TCA
        "--phase2-head-lr", "8e-5",        # Precise horizon projection learning
        "--init-checkpoint", init_ckpt,
        "--log-interval", "10",
    ]

    print(f"\n[Step 1/2] Commencing 20 Epochs Training with warm-start from {init_ckpt}...")
    proc_train = subprocess.run(cmd_train)
    if proc_train.returncode != 0:
        print("[ERROR] Training failed with non-zero exit code!")
        sys.exit(proc_train.returncode)

    # Copy / symlink finalized checkpoint
    src_ckpt = Path("outputs/checkpoints/tca_phase2_finetuned.pt")
    dst_ckpt = Path("outputs/checkpoints/tca_deep_20epochs.pt")
    if src_ckpt.exists():
        import shutil
        shutil.copy2(src_ckpt, dst_ckpt)
        print(f"\n[Checkpoint] Saved finalized 20-epoch checkpoint to {dst_ckpt}")

    print("\n[Step 2/2] Running Automated Out-Of-Sample (OOS) Benchmark...")
    cmd_eval = [
        sys.executable,
        "scripts/evaluate_tca_oos.py",
    ]
    proc_eval = subprocess.run(cmd_eval)

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f">>> DEEP TCA TRAINING & OOS VERIFICATION COMPLETED IN {elapsed/60:.2f} MINUTES")
    print("=" * 80)

if __name__ == "__main__":
    main()
