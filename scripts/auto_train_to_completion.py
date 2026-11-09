#!/usr/bin/env python3
"""Auto Train to Completion Orchestrator.

Sequentially executes:
1. Cycle 2 (Q3/2024: 2024.07.01 -> 2024.07.08) using adapted profile (Retry 2)
   with seed checkpoint_cycle_1.pkl until certified and promoted to checkpoint_cycle_2.pkl.
2. Cycle 3 (Q1/2025: 2025.01.06 -> 2025.03.31) with seed checkpoint_cycle_2.pkl
   until certified and promoted to checkpoint_cycle_3.pkl.
3. Automatically packages final multi-cycle artifacts, signs manifest.json,
   and writes COMPLETION_REPORT.json.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "walk_forward_multi_cycle"
LOG_FILE = OUTPUT_DIR / "auto_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("auto_pipeline")


def run_command(cmd: list[str]) -> int:
    cmd_str = " ".join(cmd)
    logger.info(f"Executing: {cmd_str}")
    env = os.environ.copy()
    env["SELF_GROWING_MT5_GLOBAL_LOCK"] = str(OUTPUT_DIR / ".mt5_headless_training.lock")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in iter(proc.stdout.readline, ""):
        line_clean = line.rstrip()
        if line_clean:
            logger.info(f"[SUBPROCESS] {line_clean}")
    proc.wait()
    logger.info(f"Exit code: {proc.returncode}")
    return proc.returncode


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== AUTO TRAIN TO COMPLETION PIPELINE INITIATED ===")

    seed_cycle_1 = OUTPUT_DIR / "checkpoint_cycle_1.pkl"
    if not seed_cycle_1.is_file():
        logger.error(f"Missing Cycle 1 promoted champion: {seed_cycle_1}")
        return 1

    # Step 1: Cycle 2 Execution
    champ_cycle_2 = OUTPUT_DIR / "checkpoint_cycle_2.pkl"
    if not champ_cycle_2.is_file():
        logger.info("--- STARTING CYCLE 2 (Q3/2024: 2024.07.01 -> 2024.07.08) ---")
        profile_cycle_2_r3 = OUTPUT_DIR / "profile_cycle_2_retry_3.yaml"
        profile_cycle_2_r2 = OUTPUT_DIR / "profile_cycle_2_retry_2.yaml"
        if profile_cycle_2_r3.is_file():
            profile_cycle_2 = profile_cycle_2_r3
            start_retry = "3"
        elif profile_cycle_2_r2.is_file():
            profile_cycle_2 = profile_cycle_2_r2
            start_retry = "2"
        else:
            profile_cycle_2 = ROOT / "configs" / "experiments" / "sgt_compact_tier2_candidate.yaml"
            start_retry = "0"

        logger.info(f"Using initial profile: {profile_cycle_2.name} with start_retry={start_retry}")

        cmd_cycle_2 = [
            sys.executable,
            "scripts/run_walk_forward_multi_cycle.py",
            "--cycle", "2",
            "--seed-checkpoint", str(seed_cycle_1),
            "--initial-profile", str(profile_cycle_2),
            "--start-retry", start_retry,
            "--from-date", "2024.07.01",
            "--to-date", "2024.07.08",
            "--timeout-sec", "7200",
            "--max-wall-sec", "0",
            "--max-adaptive-retries", "5",
        ]
        ret = run_command(cmd_cycle_2)
        if ret != 0 or not champ_cycle_2.is_file():
            logger.error("Cycle 2 failed to achieve promotion. Stopping pipeline.")
            return 1
        logger.info(f"Cycle 2 successfully promoted: {champ_cycle_2}")
    else:
        logger.info(f"Cycle 2 checkpoint already exists: {champ_cycle_2}")

    # Step 2: Cycle 3 Execution
    champ_cycle_3 = OUTPUT_DIR / "checkpoint_cycle_3.pkl"
    if not champ_cycle_3.is_file():
        logger.info("--- STARTING CYCLE 3 (Q1/2025: 2025.01.06 -> 2025.03.31) ---")
        cmd_cycle_3 = [
            sys.executable,
            "scripts/run_walk_forward_multi_cycle.py",
            "--cycle", "3",
            "--seed-checkpoint", str(champ_cycle_2),
            "--from-date", "2025.01.06",
            "--to-date", "2025.03.31",
            "--timeout-sec", "7200",
            "--max-wall-sec", "0",
            "--max-adaptive-retries", "5",
        ]
        ret = run_command(cmd_cycle_3)
        if ret != 0 or not champ_cycle_3.is_file():
            logger.error("Cycle 3 failed to achieve promotion. Stopping pipeline.")
            return 1
        logger.info(f"Cycle 3 successfully promoted: {champ_cycle_3}")
    else:
        logger.info(f"Cycle 3 checkpoint already exists: {champ_cycle_3}")

    # Step 3: Packaging Final Artifacts
    logger.info("--- PACKAGING FINAL MULTI-CYCLE CHAIN ARTIFACTS ---")
    cmd_package = [
        sys.executable,
        "scripts/run_walk_forward_multi_cycle.py",
        "--package-only",
        "--output-dir", str(OUTPUT_DIR),
        "--seed-checkpoint", str(seed_cycle_1),
    ]
    ret = run_command(cmd_package)
    if ret != 0:
        logger.error("Packaging final artifacts failed.")
        return 1

    manifest_file = OUTPUT_DIR / "manifest.json"
    if not manifest_file.is_file():
        logger.error(f"Manifest not generated at: {manifest_file}")
        return 1

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    logger.info(f"Manifest Status: multi_cycle_certified = {manifest_data.get('multi_cycle_certified')}")

    logger.info("=== AUTO TRAIN TO COMPLETION PIPELINE FINISHED SUCCESSFULLY ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
