"""
tests/test_isolated_mt5_runner.py

Test suite verifying isolated MT5 runner creation, concurrent separation, and automatic cleanup.
"""

from pathlib import Path
import pytest

from self_growing_trader.isolated_mt5_runner import (
    create_isolated_runner,
    cleanup_isolated_runner,
    isolated_mt5_runner_context,
    get_default_master_runner,
)


def test_isolated_mt5_runner_creation_and_cleanup():
    master = get_default_master_runner()
    assert master.is_dir(), f"Master runner must exist at {master}"

    runner = create_isolated_runner(
        master_runner=master,
        session_name="test_session",
        symbol="BTCUSDm",
        from_date="2025.01.14",
        to_date="2025.01.18",
    )
    try:
        assert runner.is_dir()
        assert (runner / "terminal64.exe").is_file()
        assert (runner / "metatester64.exe").is_file()
        assert (runner / "MQL5/Experts/SelfGrowing_MT5_Trainer_EA.ex5").is_file()
    finally:
        cleanup_isolated_runner(runner)
        assert not runner.exists()


def test_concurrent_isolated_runners_have_distinct_directories():
    master = get_default_master_runner()
    r1 = create_isolated_runner(master_runner=master, session_name="session_A")
    r2 = create_isolated_runner(master_runner=master, session_name="session_B")
    try:
        assert r1.resolve() != r2.resolve()
        assert r1.is_dir() and r2.is_dir()
        assert r1.name != r2.name
    finally:
        cleanup_isolated_runner(r1)
        cleanup_isolated_runner(r2)


def test_isolated_mt5_runner_context_manager():
    master = get_default_master_runner()
    captured_path = None
    with isolated_mt5_runner_context(master_runner=master, session_name="ctx_test") as runner_dir:
        captured_path = runner_dir
        assert runner_dir.is_dir()
        assert (runner_dir / "terminal64.exe").is_file()

    # After exiting context, directory must be automatically cleaned up
    assert not captured_path.exists()


def test_isolated_mt5_runner_context_manager_preserve_when_auto_cleanup_false():
    master = get_default_master_runner()
    captured_path = None
    with isolated_mt5_runner_context(master_runner=master, session_name="ctx_preserve", auto_cleanup=False) as runner_dir:
        captured_path = runner_dir
        assert runner_dir.is_dir()
        assert (runner_dir / "terminal64.exe").is_file()

    try:
        # Directory must remain preserved for visual inspection
        assert captured_path.is_dir()
    finally:
        cleanup_isolated_runner(captured_path)
        assert not captured_path.exists()


def test_isolated_mt5_runner_context_manager_preserve_when_shutdown_terminal_false():
    master = get_default_master_runner()
    captured_path = None
    with isolated_mt5_runner_context(master_runner=master, session_name="ctx_shutdown_false", shutdown_terminal=False) as runner_dir:
        captured_path = runner_dir
        assert runner_dir.is_dir()
        assert (runner_dir / "terminal64.exe").is_file()

    try:
        # Directory must remain preserved for visual inspection when shutdown_terminal=False
        assert captured_path.is_dir()
    finally:
        cleanup_isolated_runner(captured_path)
        assert not captured_path.exists()


def test_cleanup_isolated_runner_with_readonly_files(tmp_path: Path):
    import os
    import stat

    test_sandbox = tmp_path / "sandbox_with_readonly"
    test_sandbox.mkdir()
    ro_file = test_sandbox / "locked_file.txt"
    ro_file.write_text("read only content", encoding="utf-8")
    os.chmod(ro_file, stat.S_IREAD)

    assert ro_file.is_file()
    cleanup_isolated_runner(test_sandbox)
    assert not test_sandbox.exists()


def test_progress_file_reader_stream_methods_and_sha256(tmp_path: Path):
    import io
    import hashlib
    from self_growing_trader.progress_reader import ProgressFileReader

    raw_data = b"Hello MT5 Strategy Tester! Stream progress verification data.\nLine two."
    expected_sha = hashlib.sha256(raw_data).hexdigest().upper()

    stream = io.BytesIO(raw_data)
    reader = ProgressFileReader(stream, len(raw_data), label="TestReader", track_sha256=True)

    # Test readinto
    buf = bytearray(10)
    n = reader.readinto(buf)
    assert n == 10
    assert bytes(buf) == b"Hello MT5 "

    # Test read1 / read
    chunk = reader.read(15)
    assert chunk == b"Strategy Tester"

    # Test readline
    rest_of_line = reader.readline()
    assert rest_of_line == b"! Stream progress verification data.\n"

    # Test remaining read
    last_chunk = reader.read()
    assert last_chunk == b"Line two."

    reader.finish()
    assert reader.sha256_hexdigest == expected_sha


def test_tester_tpl_contains_utf16_bom(tmp_path: Path):
    """Verifies tester.tpl written for visual mode has standard UTF-16 LE BOM."""
    tpl_dir = tmp_path / "Profiles" / "Templates"
    tpl_dir.mkdir(parents=True, exist_ok=True)
    tester_tpl = tpl_dir / "tester.tpl"
    tester_tpl_content = (
        "<chart>\n"
        "symbol=BTCUSDm\n"
        "period_type=1\n"
        "period_size=1\n"
        "digits=2\n"
        "scale=8\n"
        "mode=1\n"
        "</chart>\n"
    )
    tester_tpl.write_text(tester_tpl_content, encoding="utf-16")

    raw_bytes = tester_tpl.read_bytes()
    assert raw_bytes.startswith(b"\xff\xfe"), "tester.tpl must start with UTF-16 LE BOM"
    decoded = raw_bytes.decode("utf-16")
    assert "<chart>" in decoded
    assert "symbol=BTCUSDm" in decoded


def test_compile_mql5_ea_staleness_guard(tmp_path: Path, monkeypatch):
    """Verifies compile_mql5_ea rejects an .ex5 binary that was created prior to compilation."""
    import time
    from self_growing_trader.mt5_runner import compile_mql5_ea

    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "logs").mkdir()
    editor = runner / "MetaEditor64.exe"
    editor.write_bytes(b"dummy")

    source = runner / "TestEA.mq5"
    source.write_text("// Test EA", encoding="utf-8")
    ex5 = runner / "TestEA.ex5"
    ex5.write_bytes(b"stale_binary")

    # Set ex5 mtime to 100 seconds in the past
    stale_time = time.time() - 100.0
    import os
    os.utime(ex5, (stale_time, stale_time))

    # Mock subprocess.run to simulate compilation producing no log or an empty log
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    res = compile_mql5_ea(runner_dir=runner, source_path=source, timeout_sec=5)
    # Since ex5 predates compilation start, it must NOT be falsely marked as compiled
    assert res["compiled"] is False


def test_prune_stale_runners_removes_dead_pids(tmp_path: Path):
    from self_growing_trader.isolated_mt5_runner import prune_stale_runners

    runners_dir = tmp_path / ".mt5_runners"
    runners_dir.mkdir()

    # Orphaned runner from an impossible/dead PID 999999
    dead_runner = runners_dir / "session_999999_20260101_120000_123456"
    dead_runner.mkdir()
    (dead_runner / "terminal64.exe").write_bytes(b"dummy")

    # Runner from current live PID
    import os
    live_runner = runners_dir / f"session_{os.getpid()}_20260101_120000_654321"
    live_runner.mkdir()
    (live_runner / "terminal64.exe").write_bytes(b"dummy")

    pruned = prune_stale_runners(runners_dir)
    assert dead_runner in pruned
    assert not dead_runner.exists()
    assert live_runner.exists()


def test_create_isolated_runner_low_disk_space_guard(tmp_path: Path, monkeypatch):
    import shutil
    from collections import namedtuple
    from self_growing_trader.isolated_mt5_runner import create_isolated_runner

    # Mock disk_usage to report only 100 MB free (< 500 MB required)
    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda _: Usage(100 * 1024**3, 99 * 1024**3, 100 * 1024 * 1024))

    with pytest.raises(OSError) as exc_info:
        create_isolated_runner(master_runner=tmp_path, session_name="low_disk_test")
    assert "Khong du dung luong o dia cho MT5 Strategy Tester" in str(exc_info.value)
    assert "toi thieu 500 MB" in str(exc_info.value)




