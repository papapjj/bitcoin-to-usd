"""
src/self_growing_trader/isolated_mt5_runner.py

Isolated MT5 Runner Manager for Concurrent Multi-Session Execution.
Eliminates MT5 portable instance lock collisions (WinError 183) by cloning
a lightweight, self-contained sandbox runner for each session into .mt5_runners/.
Enables running multiple MT5 Strategy Tester sessions (Champion, Challenger, Visual) simultaneously.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Generator, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tensor_rule_v186_candidate import (
    clone_runner_candidate,
    detect_runner_server,
)

logger = logging.getLogger("IsolatedMT5Runner")


def get_default_master_runner() -> Path:
    """Return default master MT5 runner directory."""
    return ROOT / "mt5_v2_runner"


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is currently active on the system."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_proc = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if h_proc:
                exit_code = ctypes.c_uint32()
                active = kernel32.GetExitCodeProcess(h_proc, ctypes.byref(exit_code))
                kernel32.CloseHandle(h_proc)
                return bool(active and exit_code.value == 259)  # STILL_ACTIVE
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def prune_stale_runners(runners_base: Optional[Path] = None) -> list[Path]:
    """Scans .mt5_runners/ and removes orphaned sandbox directories whose creator PID is no longer alive."""
    import re
    base = Path(runners_base or (ROOT / ".mt5_runners")).resolve()
    if not base.is_dir():
        return []

    pruned = []
    current_pid = os.getpid()
    for child in base.iterdir():
        if not child.is_dir():
            continue

        creator_pid = None
        pid_file = child / ".runner_pid"
        if pid_file.is_file():
            try:
                creator_pid = int(pid_file.read_text(encoding="utf-8").strip())
            except ValueError:
                pass

        if creator_pid is None:
            m = re.search(r"_(\d+)_(\d{8}_\d{6}_\d{6})$", child.name)
            if m:
                try:
                    creator_pid = int(m.group(1))
                except ValueError:
                    pass

        if creator_pid is not None and creator_pid != current_pid and not is_pid_alive(creator_pid):
            cleanup_isolated_runner(child)
            if not child.exists():
                pruned.append(child)
                logger.info("Pruned stale orphaned runner: %s", child.name)
    return pruned


def create_isolated_runner(
    master_runner: Optional[Path] = None,
    session_name: str = "session",
    symbol: str = "BTCUSDm",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> Path:
    """Creates a fast, isolated MT5 runner clone in .mt5_runners/.

    Clones terminal64, metatester64, EA and market bases in ~2 seconds using hardlinks.
    Guarantees zero process collision with master runner or other concurrent sessions.
    """
    master = Path(master_runner or get_default_master_runner()).resolve()
    if not master.is_dir():
        raise FileNotFoundError(f"Master MT5 runner not found: {master}")

    runners_base = ROOT / ".mt5_runners"
    runners_base.mkdir(parents=True, exist_ok=True)

    # Automatically prune stale orphaned runners from terminated sessions
    prune_stale_runners(runners_base)

    # Minimum free disk space guard (500 MB)
    usage = shutil.disk_usage(runners_base)
    min_free_bytes = 500 * 1024 * 1024
    if usage.free < min_free_bytes:
        free_mb = usage.free / (1024 * 1024)
        raise OSError(
            f"Khong du dung luong o dia cho MT5 Strategy Tester tai {runners_base}: "
            f"{free_mb:.1f} MB con trong, toi thieu 500 MB."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    target_dir = runners_base / f"{session_name}_{pid}_{timestamp}"

    server = detect_runner_server(master)
    print(f"      -> [Sandbox] Khoi tao sandbox MT5 co lap: {target_dir.name} (Server: {server})...", flush=True)
    logger.info(
        "Creating isolated MT5 runner for '%s' at: %s (Server: %s)",
        session_name, target_dir.name, server
    )

    print(f"      -> [Sandbox] Sao chep binaries MT5, EA & thiet lap market bases ({symbol})...", flush=True)
    clone_runner_candidate(
        source=master,
        target=target_dir,
        required_symbol=symbol,
        required_server=server,
        from_date=from_date,
        to_date=to_date,
    )

    # Clean any residual request/action files in MQL5/Files
    files_dir = target_dir / "MQL5" / "Files"
    if files_dir.is_dir():
        for f in files_dir.glob("self_growing_*.csv"):
            try:
                f.unlink()
            except OSError:
                pass

    try:
        (target_dir / ".runner_pid").write_text(str(pid), encoding="utf-8")
    except OSError:
        pass

    print(f"      -> [Sandbox] Sandbox da san sang hoat dong: {target_dir.name} [OK]", flush=True)
    return target_dir


def cleanup_isolated_runner(runner_dir: Path, preserve_on_error: bool = False, has_error: bool = False) -> None:
    """Safely cleans up isolated runner directory without WinError 32 crashes."""
    if not runner_dir.exists():
        return
    if preserve_on_error and has_error:
        logger.warning("Preserving failed isolated runner directory for diagnostics: %s", runner_dir)
        return

    def _force_remove_readonly(func, path, exc_or_info):
        try:
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(runner_dir, onexc=_force_remove_readonly)
        else:
            shutil.rmtree(runner_dir, onerror=_force_remove_readonly)
        if not runner_dir.exists():
            logger.debug("Cleaned up isolated runner: %s", runner_dir.name)
        else:
            logger.debug("Some files in %s remained locked during cleanup", runner_dir.name)
    except (PermissionError, OSError) as exc:
        logger.warning("Safely bypassed locked files in isolated runner %s: %s", runner_dir, exc)
    except Exception as exc:
        logger.warning("Failed to remove isolated runner %s: %s", runner_dir, exc)


@contextmanager
def isolated_mt5_runner_context(
    master_runner: Optional[Path] = None,
    session_name: str = "session",
    symbol: str = "BTCUSDm",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    preserve_on_error: bool = False,
    auto_cleanup: bool = True,
    shutdown_terminal: Optional[bool] = None,
) -> Generator[Path, None, None]:
    """Context manager for temporary isolated MT5 runner session.

    When visual inspection is active (shutdown_terminal=False), automatically
    preserves the runner sandbox so the user can inspect charts without premature deletion.
    """
    effective_auto_cleanup = auto_cleanup if shutdown_terminal is not False else False

    runner_dir = create_isolated_runner(
        master_runner=master_runner,
        session_name=session_name,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
    )
    has_error = False
    try:
        yield runner_dir
    except Exception:
        has_error = True
        raise
    finally:
        if effective_auto_cleanup:
            cleanup_isolated_runner(runner_dir, preserve_on_error=preserve_on_error, has_error=has_error)
        else:
            msg = f"Preserving isolated runner directory for visual chart inspection: {runner_dir.name}"
            logger.info(msg)
            print(f"      -> [Sandbox] Giu nguyen thu muc sandbox cho phep xem truc quan: {runner_dir.name}", flush=True)
