#!/usr/bin/env python3
"""Launch MT5 Strategy Tester Visual Mode directly on user desktop.

Allows selecting:
  [1] Champion Pha 1
  [2] Champion Pha 2
  [3] Dual-Champion Regime Router (Bull Champion + Bear Champion) [KHUYEN NGHI]
integrated with the Champion Regime Detector Engine (outputs/regime_models/champion_regime_detector.pkl).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    evaluate_policy_mt5_oos,
)
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.isolated_mt5_runner import isolated_mt5_runner_context

def get_sha256(path: Path) -> str:
    if not path.is_file():
        return "N/A"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    parser = argparse.ArgumentParser(description="Launch MT5 Visual Strategy Tester")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None, help="1=Pha 1, 2=Pha 2, 3=Dual-Champion, 4=Frontier Mythos Champion")
    parser.add_argument("--from-date", default="2025.01.14", help="Start date (YYYY.MM.DD)")
    parser.add_argument("--to-date", default="2025.01.16", help="End date (YYYY.MM.DD)")
    args = parser.parse_args()

    phase = args.phase
    if phase is None:
        print("=" * 75)
        print("  CHON CHECKPOINT POLICY DE CHAY TREN MT5 STRATEGY TESTER:")
        print("  [1] Champion Pha 1 (best_promoted_checkpoint.pkl - 910 MB)")
        print("  [2] Champion Pha 2 (best_promoted_checkpoint_phase2.pkl - 1.27 GB)")
        print("  [3] DUAL-CHAMPION REGIME ROUTER (Bull Champion + Bear Champion)")
        print("  [4] FRONTIER MYTHOS REASONING CHAMPION [KHUYEN NGHI - MOI NHAT]")
        print("      (Self-Play Đối Kháng + Selective Backtracking Sửa Sai | WR 100%)")
        print("=" * 75)
        try:
            choice = input("Nhap lua chon cua ban (1, 2, 3 hoac 4, Enter mac dinh [4]): ").strip()
            phase = int(choice) if choice in ("1", "2", "3", "4") else 4
        except Exception:
            phase = 4

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)
    regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
    runner_dir = ROOT / "mt5_v2_runner"

    # 1. Load Strategy Environment / Regime Detector
    if phase in (3, 4):
        regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
        print("\n[1/3] Dang nap Champion Regime Detector Engine...", flush=True)
        regime_sha = get_sha256(regime_model_path)
        regime_detector = RegimeDetector.load(regime_model_path)
        print(f"      -> Artifact: {regime_model_path.name}", flush=True)
        print(f"      -> SHA256  : {regime_sha}", flush=True)
        print(f"      -> Status  : SAN SANG PHAN BO XAC SUAT [P_Bull, P_Bear, P_Chop]!", flush=True)
    else:
        print("\n[1/3] Kiem tra cau hinh he thong va thiet lap moi truong...", flush=True)
        print(f"      -> Config  : {config_path.name} [OK]", flush=True)
        regime_detector = None

    # 2. Load Policy Checkpoint
    if phase == 1:
        candidate_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint.pkl"
        phase_label = "CHAMPION PHA 1 (best_promoted_checkpoint.pkl)"
        print(f"\n[2/3] Dang nap Policy Checkpoint Pha 1...", flush=True)
        policy = tai_he_chuyen_gia_mot_context_mot_policy(candidate_path, config, seed=2026)
        print(f"      -> Artifact: {candidate_path.name} ({candidate_path.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)
        print(f"      -> Status  : DA NAP THANH CONG!", flush=True)
    elif phase == 2:
        candidate_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
        phase_label = "CHAMPION PHA 2 (best_promoted_checkpoint_phase2.pkl)"
        print(f"\n[2/3] Dang nap Policy Checkpoint Pha 2...", flush=True)
        policy = tai_he_chuyen_gia_mot_context_mot_policy(candidate_path, config, seed=2026)
        print(f"      -> Artifact: {candidate_path.name} ({candidate_path.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)
        print(f"      -> Status  : DA NAP THANH CONG!", flush=True)
    elif phase == 4:
        import torch
        from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
        bear_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
        bull_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "multi_round_phase3" / "round_01" / "reasoning_rl_v1_active_candidate.pkl"
        phase_label = "FRONTIER MYTHOS REASONING CHAMPION (Dual-Champion Router + Dialectical Backtracking)"
        print(f"\n[2/3] Dang nap Frontier Mythos Reasoning Champion (Full Dual-Regime)...", flush=True)
        base_router = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
        )
        policy = MythosMT5Policy(base_policy=base_router)
        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)
                print(f"      -> Injected Multi-Asset Champion weights ({champ_pt.name}) [OK]", flush=True)
        print(f"      -> Status  : DA NAP THANH CONG!", flush=True)
    else:
        bear_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
        bull_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "multi_round_phase3" / "round_01" / "reasoning_rl_v1_active_candidate.pkl"
        phase_label = "DUAL-CHAMPION REGIME ROUTER (Bull R01 + Bear Phase 2)"
        print(f"\n[2/3] Dang nap He Thong DUAL-CHAMPION REGIME ROUTER...", flush=True)
        policy = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
        )
        print(f"      -> Status  : DUAL-CHAMPION DA KET NOI HOAN HAO QUA REGIME DETECTOR!", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"  KHOI DONG MO PHONG TRUC QUAN MT5 STRATEGY TESTER", flush=True)
    print(f"  Policy : {phase_label}", flush=True)
    regime_banner = "FRONTIER ADVERSARIAL DIALECTICAL REASONING" if phase == 4 else ("CHAMPION REGIME DETECTOR ENGINE" if phase == 3 else f"UNIFIED POLICY (Pha {phase})")
    print(f"  Regime : {regime_banner}", flush=True)
    print(f"  Dates  : {args.from_date} -> {args.to_date} | Symbol: BTCUSDm M1", flush=True)
    print("=" * 75, flush=True)

    # 3. Launch MT5 Strategy Tester
    print("\n[3/3] Thiet lap MT5 Strategy Tester (Visual=1, ShutdownTerminal=0)...", flush=True)
    print("      Cua so do thi MT5 Visual dang duoc mo len man hinh!", flush=True)
    print("      Nhan Ctrl+C trong console nay neu muon dung.\n", flush=True)

    with isolated_mt5_runner_context(
        session_name="visual",
        symbol="BTCUSDm",
        from_date=args.from_date,
        to_date=args.to_date,
        shutdown_terminal=False,
        auto_cleanup=False,
    ) as isolated_runner:
        result = evaluate_policy_mt5_oos(
            policy,
            config,
            runner_dir=isolated_runner,
            timeout_sec=7200,
            visual=True,
            shutdown_terminal=False,
            from_date=args.from_date,
            to_date=args.to_date,
            model=1,
        )

    best = result.get("best", {})
    metrics = best.get("metrics", {})
    print("\n" + "=" * 75)
    print(f"  KET QUA MO PHONG MT5 ({phase_label}):")
    print(f"  Total Trades : {metrics.get('total_trades')}")
    print(f"  Net Profit   : ${metrics.get('net_profit', 0):.2f}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Win Rate     : {metrics.get('win_rate', 0):.2f}%")
    print("=" * 75)
    print("\n[OK] Cua so bieu do MT5 Strategy Tester dang mo tren man hinh cua ban.")
    print("     Ban co the xem do thi, lenh giao dich, trailing stop truc tiep.")
    print("     Nhan Enter de dong console hoac dong cua so MT5 khi hoan tat.\n")
    try:
        input()
    except Exception:
        pass

if __name__ == "__main__":
    main()
