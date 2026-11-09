from __future__ import annotations

from pathlib import Path
import json
import re
import time
import numpy as np
import pandas as pd

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1", "M2": "TIMEFRAME_M2", "M3": "TIMEFRAME_M3", "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def _import_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise RuntimeError("MetaTrader5 package is not installed. Run this command on the Windows machine hosting MT5.") from e
    return mt5


def _initialize_mt5(mt5, terminal_path: str | Path | None = None, *, timeout_ms: int = 60_000) -> dict:
    """Khoi tao dung terminal MT5 va xac nhan tai khoan da dang nhap.

    Khi ``terminal_path`` duoc truyen, cong cu dung chinh terminal portable cua
    MT5 runner de tai su dung ho so dang nhap da luu, giong duong huan luyen MT5.
    """
    if terminal_path is not None:
        terminal = Path(terminal_path).expanduser().resolve()
        if not terminal.is_file():
            raise FileNotFoundError(f"MT5 terminal not found: {terminal}")
        ok = mt5.initialize(str(terminal), timeout=int(timeout_ms), portable=True)
    else:
        terminal = None
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    wait_deadline = time.monotonic() + min(30.0, max(1.0, float(timeout_ms) / 1000.0))
    account = None
    terminal_info = None
    while True:
        try:
            account = mt5.account_info()
        except Exception:
            account = None
        try:
            terminal_info = mt5.terminal_info()
        except Exception:
            terminal_info = None
        connected = bool(getattr(terminal_info, "connected", True)) if terminal_info is not None else account is not None
        if account is not None and connected:
            break
        if time.monotonic() >= wait_deadline:
            try:
                mt5.shutdown()
            finally:
                pass
            raise RuntimeError(
                "MT5 da duoc mo nhung khong ket noi duoc tai khoan. "
                + (f"Hay mo va dang nhap mot lan trong terminal portable: {terminal}, chon luu mat khau, roi quet lai." if terminal else "Hay dang nhap terminal MT5 va quet lai.")
                + (f" terminal_info={terminal_info}" if terminal_info is not None else "")
            )
        time.sleep(0.25)
    return {
        "login": int(getattr(account, "login", 0) or 0),
        "server": str(getattr(account, "server", "") or ""),
        "name": str(getattr(account, "name", "") or ""),
        "company": str(getattr(account, "company", "") or ""),
        "terminal_path": str(terminal) if terminal is not None else str(getattr(terminal_info, "path", "") or ""),
        "connected": bool(getattr(terminal_info, "connected", True)) if terminal_info is not None else True,
        "maxbars": int(getattr(terminal_info, "maxbars", 0) or 0) if terminal_info is not None else 0,
    }


def _validate_fetch_args(timeframe: str, bars: int, chunk_size: int) -> None:
    if timeframe.upper() not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe}; choose from {sorted(TIMEFRAMES)}")
    if bars <= 0:
        raise ValueError("bars must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")


def _fetch_mt5_initialized(mt5, symbol: str, timeframe: str, bars: int, *, chunk_size: int) -> pd.DataFrame:
    _validate_fetch_args(timeframe, bars, chunk_size)
    # Một số máy chủ chỉ trả lịch sử ổn định sau khi mã được thêm vào Market Watch.
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Cannot select {symbol} in MT5 Market Watch. MT5 last_error={mt5.last_error()}")
    tf = getattr(mt5, TIMEFRAMES[timeframe.upper()])
    terminal_info = mt5.terminal_info()
    terminal_maxbars = int(getattr(terminal_info, "maxbars", 0) or 0) if terminal_info is not None else 0
    fetch_limit = min(int(bars), terminal_maxbars) if terminal_maxbars > 0 else int(bars)
    # MT5 build 5833 rejects count == maxbars with "Invalid params". Keep one
    # slot below the chart ceiling and fetch the final row in the next call.
    effective_chunk = min(int(chunk_size), max(1, terminal_maxbars - 1)) if terminal_maxbars > 1 else int(chunk_size)
    parts = []
    start = 0
    while start < fetch_limit:
        count = min(effective_chunk, fetch_limit - start)
        rates = mt5.copy_rates_from_pos(symbol, tf, start, count)
        if rates is None or len(rates) == 0:
            break
        parts.append(pd.DataFrame(rates))
        start += len(rates)
        if len(rates) < count:
            break
    if not parts:
        raise RuntimeError(f"No bars returned for {symbol} {timeframe}. MT5 last_error={mt5.last_error()}")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if "tick_volume" in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    result = df[[c for c in ["time", "open", "high", "low", "close", "volume", "spread", "real_volume"] if c in df.columns]]
    result.attrs["terminal_maxbars"] = terminal_maxbars
    result.attrs["limited_by_terminal_maxbars"] = bool(
        terminal_maxbars > 0 and int(bars) > terminal_maxbars and len(result) >= terminal_maxbars
    )
    return result


def fetch_mt5(symbol: str, timeframe: str, bars: int, *, chunk_size: int = 200_000, terminal_path: str | Path | None = None) -> pd.DataFrame:
    """Lấy lịch sử của một mã từ terminal MT5 cục bộ."""
    mt5 = _import_mt5()
    _validate_fetch_args(timeframe, bars, chunk_size)
    _initialize_mt5(mt5, terminal_path)
    try:
        return _fetch_mt5_initialized(mt5, symbol, timeframe, bars, chunk_size=chunk_size)
    finally:
        mt5.shutdown()


def list_mt5_symbols(*, terminal_path: str | Path | None = None, return_session: bool = False):
    """Quét toàn bộ mã mà terminal/tài khoản MT5 hiện tại nhìn thấy trên máy chủ."""
    mt5 = _import_mt5()
    session = _initialize_mt5(mt5, terminal_path)
    try:
        symbols = mt5.symbols_get()
        if symbols is None:
            raise RuntimeError(f"mt5.symbols_get() failed: {mt5.last_error()}")
        rows = []
        for item in symbols:
            rows.append({
                "name": str(getattr(item, "name", "") or ""),
                "description": str(getattr(item, "description", "") or ""),
                "path": str(getattr(item, "path", "") or ""),
                "visible": bool(getattr(item, "visible", False)),
                "select": bool(getattr(item, "select", False)),
                "digits": int(getattr(item, "digits", 0) or 0),
                "currency_base": str(getattr(item, "currency_base", "") or ""),
                "currency_profit": str(getattr(item, "currency_profit", "") or ""),
                "currency_margin": str(getattr(item, "currency_margin", "") or ""),
                "trade_mode": int(getattr(item, "trade_mode", 0) or 0),
            })
        rows.sort(key=lambda row: (row["path"], row["name"]))
        return (rows, session) if return_session else rows
    finally:
        mt5.shutdown()


def safe_symbol_filename(symbol: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(symbol).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "symbol"


def _symbol_contract(mt5, symbol: str) -> dict:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info({symbol}) failed: {mt5.last_error()}")
    return {
        "digits": int(getattr(info, "digits", 0) or 0),
        "point": float(getattr(info, "point", 0.0) or 0.0),
        "trade_contract_size": float(getattr(info, "trade_contract_size", 0.0) or 0.0),
        "trade_tick_size": float(getattr(info, "trade_tick_size", 0.0) or 0.0),
        "trade_tick_value": float(getattr(info, "trade_tick_value", 0.0) or 0.0),
        "trade_calc_mode": int(getattr(info, "trade_calc_mode", 0) or 0),
        "currency_base": str(getattr(info, "currency_base", "") or ""),
        "currency_profit": str(getattr(info, "currency_profit", "") or ""),
        "currency_margin": str(getattr(info, "currency_margin", "") or ""),
    }




def _existing_npz_rows(path: Path) -> int | None:
    """Return row count for an atomic NPZ produced earlier, else None."""
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "time" not in archive.files:
                return None
            time_values = archive["time"]
            if time_values.ndim != 1:
                return None
            return int(time_values.shape[0])
    except Exception:
        return None


def _write_fetch_progress(output_dir: Path, payload: dict) -> None:
    target = output_dir / "manifest.json"
    temporary = target.with_name(target.name + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)

def fetch_mt5_many(
    symbols: list[str], timeframe: str, bars: int, output_dir: str | Path, *, chunk_size: int = 200_000,
    terminal_path: str | Path | None = None, resume: bool = True
) -> dict:
    """Lấy tuần tự các mã đã chọn, lưu riêng từng NPZ và có thể tiếp tục sau khi gián đoạn."""
    _validate_fetch_args(timeframe, bars, chunk_size)
    selected = []
    seen = set()
    for raw in symbols:
        symbol = str(raw).strip()
        if symbol and symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)
    if not selected:
        raise ValueError("symbol list is empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mt5 = _import_mt5()
    session = _initialize_mt5(mt5, terminal_path)
    results = []

    def snapshot() -> dict:
        ok_count = sum(1 for row in results if row.get("ok"))
        return {
            "requested_symbols": len(selected),
            "processed_symbols": len(results),
            "ok_symbols": ok_count,
            "failed_symbols": len(results) - ok_count,
            "pending_symbols": len(selected) - len(results),
            "timeframe": timeframe.upper(),
            "bars_per_symbol": int(bars),
            "chunk_size": int(chunk_size),
            "output_dir": str(output_dir),
            "mt5_session": session,
            "terminal_maxbars": int(session.get("maxbars", 0) or 0),
            "results": results,
        }

    try:
        for index, symbol in enumerate(selected, start=1):
            target = output_dir / f"{safe_symbol_filename(symbol)}_{timeframe.upper()}.npz"
            symbol_contract = _symbol_contract(mt5, symbol)
            existing_rows = _existing_npz_rows(target) if resume else None
            if existing_rows is not None:
                terminal_maxbars = int(session.get("maxbars", 0) or 0)
                limited = bool(terminal_maxbars > 0 and int(bars) > terminal_maxbars and existing_rows >= terminal_maxbars)
                results.append({
                    "symbol": symbol,
                    "ok": True,
                    "skipped_existing": True,
                    "requested_rows": int(bars),
                    "rows": int(existing_rows),
                    "history_exhausted": bool(existing_rows < int(bars)),
                    "limited_by_terminal_maxbars": limited,
                    "history_limit_reason": "terminal_maxbars" if limited else "broker_history_or_existing_file",
                    "terminal_maxbars": terminal_maxbars,
                    "symbol_contract": symbol_contract,
                    "output": str(target),
                })
                print(f"[{index}/{len(selected)}] {symbol}: SKIP ({existing_rows} rows)", flush=True)
                _write_fetch_progress(output_dir, snapshot())
                continue

            try:
                frame = _fetch_mt5_initialized(mt5, symbol, timeframe, bars, chunk_size=chunk_size)
                save_mt5_npz(frame, target)
                terminal_maxbars = int(frame.attrs.get("terminal_maxbars", 0) or 0)
                limited = bool(frame.attrs.get("limited_by_terminal_maxbars", False))
                results.append({
                    "symbol": symbol,
                    "ok": True,
                    "skipped_existing": False,
                    "requested_rows": int(bars),
                    "rows": int(len(frame)),
                    "history_exhausted": bool(len(frame) < int(bars)),
                    "limited_by_terminal_maxbars": limited,
                    "history_limit_reason": "terminal_maxbars" if limited else ("broker_history" if len(frame) < int(bars) else ""),
                    "terminal_maxbars": terminal_maxbars,
                    "symbol_contract": symbol_contract,
                    "output": str(target),
                })
            except Exception as exc:
                results.append({
                    "symbol": symbol,
                    "ok": False,
                    "skipped_existing": False,
                    "requested_rows": int(bars),
                    "rows": 0,
                    "history_exhausted": False,
                    "output": str(target),
                    "error": str(exc),
                })
            # Keep console status ASCII-only as a second guard for legacy Windows shells.
            print(f"[{index}/{len(selected)}] {symbol}: " + ("OK" if results[-1]["ok"] else "ERROR"), flush=True)
            _write_fetch_progress(output_dir, snapshot())
    finally:
        mt5.shutdown()

    return snapshot()


def save_mt5_npz(df: pd.DataFrame, path: str | Path) -> None:
    """Ghi du lieu MT5 an toan, tu tao thu muc cha neu chua ton tai.

    Ghi qua tep tam roi thay the tep dich de tranh de lai mot tep NPZ hong neu
    qua trinh nen bi dung giua chung.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for c in df.columns:
        if c == "time":
            payload[c] = (
                pd.to_datetime(df[c], utc=True)
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64, copy=False)
            )
        else:
            payload[c] = df[c].to_numpy()

    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
