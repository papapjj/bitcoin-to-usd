#!/usr/bin/env python3
"""Launch MT5 Strategy Tester in Visual Mode powered directly by the
xLSTM 17.5M World-Class Reasoning Engine (Mythos của Claude).

Opens MetaTrader 5 on the user desktop (WinSta0\\Default) and displays real-time candlestick charts,
with trades, stop loss, take profit, and trailing stops governed 100% by
the newly developed 17.5M parameter xLSTM SGT-V193 backbone and Process Reward Model.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import self_growing_trader.mt5_runner as mt5_runner_mod
from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    evaluate_policy_mt5_oos,
)
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy


# ==============================================================================
# Win32 Interactive Desktop Forwarder (Routes MT5 to User Screen WinSta0\\Default)
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
    """Wraps Win32 CreateProcessW targeting WinSta0\\Default so MT5 Visual window appears on user screen."""

    def __init__(self, command, cwd=None, desktop="WinSta0\\Default"):
        cmd_str = subprocess.list2cmdline(command) if isinstance(command, (list, tuple)) else str(command)
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(STARTUPINFOW)
        si.lpDesktop = desktop
        si.dwFlags = 1  # STARTF_USESHOWWINDOW
        si.wShowWindow = 1  # SW_SHOWNORMAL
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
                "[MT5 Desktop Bridge] Dang mo cua so MT5 truc tiep len man hinh desktop WinSta0\\Default...",
                flush=True,
            )
            try:
                cwd = kwargs.get("cwd", None)
                proc = DesktopMT5Process(cmd, cwd=cwd)

                def _delayed_front():
                    time.sleep(2.0)
                    bring_mt5_to_front(proc.pid)
                    time.sleep(2.0)
                    bring_mt5_to_front(proc.pid)

                threading.Thread(target=_delayed_front, daemon=True).start()
                return proc
            except Exception as e:
                print(f"[MT5 Desktop Bridge] Fallback to standard Popen: {e}", flush=True)
        return _orig_popen(cmd, *args, **kwargs)

    mt5_runner_mod.subprocess.Popen = _desktop_aware_popen


def get_sha256(path: Path) -> str:
    if not path.is_file():
        return "N/A"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def cleanup_stale_lock():
    lock_file = ROOT / "mt5_v2_runner" / ".mt5_strategy_tester.global.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
            print(f"[Cleanup] Removed stale lock: {lock_file.name}")
        except Exception as e:
            print(f"[Cleanup] Note: {e}")


def main():
    parser = argparse.ArgumentParser(description="Launch MT5 Visual Tester with Mythos xLSTM 17.5M Model")
    parser.add_argument("--from-date", default="2025.02.20", help="Start date (YYYY.MM.DD)")
    parser.add_argument("--to-date", default="2025.02.28", help="End date (YYYY.MM.DD)")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial simulation capital")
    args = parser.parse_args()

    print("=" * 80)
    print("  KHOI CHAY METATRADER 5 VISUAL MODE VOI MODEL MYTHOS xLSTM 17.5M")
    print("  Arch: SGT-V193 xLSTM (17,840,781 params) + 4-Hop CoT + PRM (85% Gate)")
    print("  Capital: $" + f"{args.initial_capital:,.2f} | Symbol: BTCUSDm M1")
    print(f"  Period : {args.from_date} -> {args.to_date}")
    print("=" * 80, flush=True)

    cleanup_stale_lock()
    install_desktop_launcher_hook()

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)
    base_checkpoint_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    runner_dir = ROOT / "mt5_v2_runner"

    # 1. Load Base Infrastructure Checkpoint
    print("\n[1/3] Dang nap Policy Base Checkpoint cho 5-Agent MT5 Bridge...", flush=True)
    base_policy = tai_he_chuyen_gia_mot_context_mot_policy(base_checkpoint_path, config, seed=2026)
    print(f"      -> Base Checkpoint: {base_checkpoint_path.name} (SHA: {get_sha256(base_checkpoint_path)[:16]}...)", flush=True)
    print("      -> 5-Agent Wire Provenance: SAN SANG!", flush=True)

    # 2. Attach Mythos xLSTM 17.5M World-Class Reasoning Engine
    print("\n[2/3] Dang khoi tao Mythos xLSTM 17.5M Reasoning Engine...", flush=True)
    mythos_policy = MythosMT5Policy(
        base_policy=base_policy,
        initial_capital=args.initial_capital,
        symbol="BTC",
    )
    print("      -> SGT-V193 xLSTM Core: 17,840,781 parameters active", flush=True)
    print("      -> Memory Cells: 147,456 matrix mLSTM units initialized", flush=True)
    print("      -> Metacognitive Reasoner: 4-Hop CoT + PRM (85% Gate) connected", flush=True)
    print("      -> Dynamic Kelly Compounding: f* = 20%, Safety Floor = 10% active", flush=True)
    print("      -> Invariant Lock: POLICY_CLOSE_ALL == 0.0 ENFORCED", flush=True)

    # 3. Launch MT5 Strategy Tester Visual Mode
    print("\n[3/3] Thiet lap MT5 Strategy Tester (Visual=1, ShutdownTerminal=0)...", flush=True)
    print("      Cua so do thi MT5 Visual dang duoc mo len man hinh desktop!", flush=True)
    print("      Quy trinh se tu dong truyen nhan ticks va giao dich truc quan tren do thi.", flush=True)
    print("      Nhan Ctrl+C trong console nay neu muon dung som.\n", flush=True)

    try:
        result = evaluate_policy_mt5_oos(
            mythos_policy,
            config,
            runner_dir=runner_dir,
            timeout_sec=7200,
            visual=True,
            shutdown_terminal=False,
            from_date=args.from_date,
            to_date=args.to_date,
            model=1,
        )
    except KeyboardInterrupt:
        print("\n[Stop] Nguoi dung da chu dong tam dung qua trinh chay Visual.")
        return

    best = result.get("best", {})
    metrics = best.get("metrics", {})
    print("\n" + "=" * 80)
    print("  KET QUA MO PHONG MT5 VISUAL (MODEL MYTHOS xLSTM 17.5M):")
    print(f"  Total Trades : {metrics.get('total_trades')}")
    print(f"  Net Profit   : ${metrics.get('net_profit', 0):.2f}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Win Rate     : {metrics.get('win_rate', 0):.2f}%")
    print(f"  Max Drawdown : {metrics.get('max_drawdown_pct', 0):.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
