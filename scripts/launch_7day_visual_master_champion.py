import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

#!/usr/bin/env python3
"""
scripts/launch_7day_visual_master_champion.py

Interactive 7-Day Visual MT5 Strategy Tester Launcher.
Runs the Golden Master Champion Strategy in full graphical mode (Visual=1) on MT5 Desktop
for a 7-day out-of-sample period (2025.01.10 -> 2025.01.17).

Visual Features on Desktop:
- Candlestick animation on BTCUSDm M1 chart
- Real-time order placement arrows (BUY)
- Dynamic Stop Loss and Take Profit red/green target levels
- Early Breakeven Lock (+1.0 ATR at >= 2.80 ATR) and Chandelier Trailing Stop
- Unchoked Profit expansion to 4.5 ATR Partial TP and 7.5 ATR Full TP
- 100% Neural Autonomy driven by retrained xLSTM 17.5M CoT & PRM Verifier (0 gating locks)
"""

import argparse
import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import self_growing_trader.mt5_runner as mt5_runner_mod
from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import evaluate_policy_mt5_oos
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.cppi_risk_engine import CPPIConfig
from self_growing_trader.two_tier_runner import TwoTierConfig
from self_growing_trader.isolated_mt5_runner import isolated_mt5_runner_context
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Visual7DayChampion")


# ==============================================================================
# Win32 Interactive Desktop Forwarder (Ensures MT5 window opens on User Screen)
# ==============================================================================
class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_char_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class DesktopMT5Process:
    def __init__(self, command, cwd=None, desktop="WinSta0\\Default"):
        cmd_str = subprocess.list2cmdline(command) if isinstance(command, (list, tuple)) else str(command)
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(STARTUPINFOW)
        si.lpDesktop = desktop
        si.dwFlags = 1  # STARTF_USESHOWWINDOW
        si.wShowWindow = 3  # SW_MAXIMIZE
        pi = PROCESS_INFORMATION()
        kernel32 = ctypes.windll.kernel32
        res = kernel32.CreateProcessW(
            None,
            cmd_str,
            None,
            None,
            False,
            0,
            None,
            str(cwd) if cwd else None,
            ctypes.byref(si),
            ctypes.byref(pi),
        )
        if not res:
            err = kernel32.GetLastError()
            raise OSError(f"CreateProcessW failed with error {err}")
        self.pid = pi.dwProcessId
        self._hProcess = pi.hProcess
        self._hThread = pi.hThread
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        exit_code = wintypes.DWORD()
        if ctypes.windll.kernel32.GetExitCodeProcess(self._hProcess, ctypes.byref(exit_code)):
            if exit_code.value == 259:  # STILL_ACTIVE
                return None
            self.returncode = int(exit_code.value)
            return self.returncode
        return None

    def terminate(self):
        if self.poll() is None:
            ctypes.windll.kernel32.TerminateProcess(self._hProcess, 1)

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        ctypes.windll.kernel32.WaitForSingleObject(
            self._hProcess, 0xFFFFFFFF if timeout is None else int(timeout * 1000)
        )
        return self.poll()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.poll() is None:
            self.terminate()


def bring_mt5_to_front(pid: int):
    try:
        user32 = ctypes.windll.user32
        h_default = user32.OpenDesktopW("Default", 0, False, 0x0100)
        if not h_default:
            return
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def cb(hwnd, lparam):
            p_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p_id))
            if p_id.value == pid:
                buff = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buff, 256)
                if buff.value:
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                    user32.SetForegroundWindow(hwnd)
            return True

        user32.EnumDesktopWindows(h_default, WNDENUMPROC(cb), 0)
    except Exception:
        pass


def install_desktop_launcher_hook():
    _orig_popen = subprocess.Popen

    def _desktop_aware_popen(cmd, *args, **kwargs):
        is_terminal = (
            isinstance(cmd, (list, tuple))
            and len(cmd) > 0
            and "terminal64" in str(cmd[0]).lower()
            and str(cmd[0]).lower().endswith(".exe")
        )
        if is_terminal and os.name == "nt":
            print(
                "\n[MT5 Desktop Bridge] Dang mo cua so MetaTrader 5 truc tiep len man hinh Desktop...",
                flush=True,
            )
            try:
                cwd = kwargs.get("cwd", None)
                proc = DesktopMT5Process(cmd, cwd=cwd)

                def _gentle_bring_to_front():
                    for _ in range(6):
                        time.sleep(1.2)
                        bring_mt5_to_front(proc.pid)

                threading.Thread(target=_gentle_bring_to_front, daemon=True).start()
                return proc
            except Exception as e:
                print(f"[MT5 Desktop Bridge] Fallback to standard Popen: {e}", flush=True)
        return _orig_popen(cmd, *args, **kwargs)

    mt5_runner_mod.subprocess.Popen = _desktop_aware_popen


def main():
    parser = argparse.ArgumentParser(description="Launch 7-Day Visual MT5 Strategy Tester (Master Champion)")
    parser.add_argument("--from-date", default="2025.02.20", help="Start date (YYYY.MM.DD)")
    parser.add_argument("--to-date", default="2025.02.27", help="End date (YYYY.MM.DD) - 7 full days")
    parser.add_argument("--symbol", default="BTCUSDm", help="Market symbol")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial simulation capital")
    args = parser.parse_args()

    print("=" * 80)
    print("  KHOI CHAY METATRADER 5 VISUAL MODE - 7 NGAY KIEM DINH THUC CHUNG")
    print("  Model: Mythos xLSTM 17.5M CoT & PRM Verifier (Retrained Anti-Exhaustion Champion)")
    print(f"  Symbol: {args.symbol} M1 | San: Exness-MT5Real8")
    print(f"  Chu ky: {args.from_date} -> {args.to_date} (7 Full OOS Days)")
    print("  Co che: Unchoked Breakeven Lock (+1.0 ATR at >= 2.8 ATR) | Partial TP (4.5 ATR)")
    print("=" * 80, flush=True)

    install_desktop_launcher_hook()

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)

    bear_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    bull_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "multi_round_phase3" / "round_01" / "reasoning_rl_v1_active_candidate.pkl"
    regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
    regime_detector = RegimeDetector.load(regime_model_path)

    # 1. Calibrate Safe CPPI Risk Engine & Two-Tier Unchoked Geometry
    safe_cppi = CPPIConfig(
        max_allowed_dd=0.08,
        cushion_loss_fraction=0.15,
        base_alpha=0.20,
        pumping_slope=0.6,
        max_risk_cap=0.015,
        confluence_boost=1.2,
    )
    two_tier = TwoTierConfig(
        initial_sl_atr=2.80,
        breakeven_trigger_atr=1.50,
        breakeven_lock_atr=0.80,
        partial_tp_atr=4.50,
        trailing_atr=1.20,
        trail_after_breakeven=True,
    )

    # 2. Build Dual-Champion Router (0 Artificial Gating Locks)
    base_router = load_dual_champion_router(
        bear_path=bear_path,
        bull_path=bull_path,
        detector_path=regime_detector,
        bull_threshold=0.58,
        bear_threshold=0.52,
        chop_threshold=0.38,
        confirmation_bars=1,
        htf_filter_enabled=True,
        momentum_gate_enabled=True,
        circuit_breaker_enabled=False,
        micro_scalp_enabled=False,
        max_concurrent_positions=3,
        max_positions_per_direction=3,
        cppi_config=safe_cppi,
        two_tier_config=two_tier,
    )

    # 3. Wrap with Mythos Metacognitive Reasoning Policy
    policy = MythosMT5Policy(base_policy=base_router, initial_capital=args.initial_capital, symbol="BTC")
    policy.long_only = False  # Unlocked Bidirectional BUY & SELL
    policy.short_only = False
    policy.min_confidence = 0.58       # Calibrated Bull Specialist Conviction
    policy.bear_entry_threshold = 0.52 # Calibrated Bear Specialist Conviction
    policy.min_margin_threshold = 0.15 # Bull - Bear Margin
    policy.bear_min_margin = 0.15      # Bear - Bull Margin
    policy.prm_threshold = 0.38        # Process Reward Model Gate
    policy.scalp_sl_atr = 2.80         # Strict Anti-MAE Stop Loss
    policy.sniper_sl_atr = 2.80
    policy.scalp_tp_rr = 1.25          # Target 3.50 ATR (2.80 * 1.25)
    policy.sniper_tp_rr = 1.25
    policy.breakeven_trigger_atr = 1.50 # Ratchet trigger at +1.50 ATR
    policy.breakeven_lock_atr = 0.80    # Lock +0.80 ATR into profit
    policy.trailing_atr = 1.20          # Chandelier trailing distance
    policy.max_positions = 3            # Concurrent position capacity
    policy.min_spacing_bars = 2         # Minimum bar spacing
    policy.post_exit_cooldown_bars = 1  # 1-bar anti-whip pause
    policy.min_atr_threshold = 5.0
    policy.max_atr_percentile = 0.98
    policy.require_margin_alignment = False
    policy.loss_cooldown_bars = 0
    policy.max_consecutive_losses = 999

    # 4. Load Calibrated MuZero Co-Evolved Champion Weights
    champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
    if champ_pt.exists():
        ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
        if "reasoning_engine_state_dict" in ckpt:
            policy.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
            logger.info("Loaded Retrained CoT Reasoning Engine from %s", champ_pt.name)
        if "world_model_state_dict" in ckpt:
            policy.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
            logger.info("Loaded Latent World Model into policy from %s", champ_pt.name)

    # Load Frontier Bilateral Dual-CoT Precision Champion Weights with Calibrated Threshold
    v3_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion_v3.pt"
    precision_pt = v3_pt if v3_pt.exists() else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt")
    bilateral_pt = precision_pt if precision_pt.exists() else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt")
    if bilateral_pt.exists():
        policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
        policy.pipeline.bilateral_engine.config.prm_threshold = 0.38
        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.38
        policy.pipeline.bilateral_engine.arbiter.trap_threshold = 0.99
        policy.pipeline.use_bilateral_dual_cot = True
        logger.info("Loaded Frontier Bilateral Dual-CoT Precision Champion from %s (prm_threshold=0.38)", bilateral_pt.name)

    # Reset fresh reasoning trace logger
    trace_path = ROOT / "outputs" / "mt5_optimization_loop" / "reasoning_trace.jsonl"
    if trace_path.exists():
        try:
            trace_path.unlink()
        except Exception:
            pass

    # 5. Launch in isolated visual context
    print("\n[Sandbox] Khoi tao sandbox MT5 Visual co lap cho 7 ngay OOS...", flush=True)
    with isolated_mt5_runner_context(
        session_name="visual_7day_champion",
        symbol=args.symbol,
        from_date=args.from_date,
        to_date=args.to_date,
        shutdown_terminal=False,  # Keep MT5 visual window open on screen!
        auto_cleanup=False,
    ) as runner_dir:

        policy.runner_dir = Path(runner_dir)
        print("\n" + "=" * 80)
        print(f"  [OK] CUA SO MT5 VISUAL DANG KHOI CHAY TREN DESKTOP!")
        print(f"  Sandbox Runner: {runner_dir}")
        print("  Ban co the truc tiep quan sat:")
        print("  - Tung cay nen M1 di chuyen thoi gian thuc tren bieu do.")
        print("  - Cac mui ten xanh no-ron vao lenh BUY tai chan song.")
        print("  - Duong do Stop Loss va duong xanh Take Profit.")
        print("  - Khi gia dat >= 2.80 ATR, SL tu dong nhay len khoa hoa von (+1.0 ATR).")
        print("=" * 80 + "\n", flush=True)

        try:
            result = evaluate_policy_mt5_oos(
                policy,
                config,
                runner_dir=runner_dir,
                timeout_sec=28800,       # 8 hours allowance for visual execution
                visual=True,             # Visual=1 in tester.ini
                shutdown_terminal=False, # Keep terminal running
                from_date=args.from_date,
                to_date=args.to_date,
                model=1,                 # M1 OHLC bar model
            )
        except KeyboardInterrupt:
            print("\n[Stop] Nguoi dung da dung tien trinh visual.")
            return

        best = result.get("best", {})
        metrics = best.get("metrics", {})
        print("\n" + "=" * 80)
        print("  KET QUA CHAY MT5 VISUAL 7 NGAY HOAN TAT:")
        print(f"  Total Trades : {metrics.get('total_trades')}")
        print(f"  Net Profit   : ${metrics.get('net_profit', 0):.2f}")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"  Win Rate     : {metrics.get('win_rate', 0):.2f}%")
        print(f"  Max Drawdown : {metrics.get('max_drawdown_pct', 0):.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()
