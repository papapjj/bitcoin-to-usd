from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from zigzag_miner.io import load_ohlc, trim_sparse_timeframe_prefix
from zigzag_miner.mt5_source import save_mt5_npz


def test_trim_sparse_m1_prefix_starts_at_first_dense_run():
    sparse = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    dense = pd.date_range("2020-01-05", periods=8, freq="min", tz="UTC")
    index = sparse.append(dense)
    frame = pd.DataFrame({"open": np.arange(len(index), dtype=float)}, index=index)

    trimmed, removed = trim_sparse_timeframe_prefix(
        frame, "M1", minimum_consecutive_bars=5
    )

    assert removed == 4
    assert trimmed.index.equals(dense)


def test_save_mt5_npz_creates_missing_parent(tmp_path: Path):
    target = tmp_path / "outputs" / "datasets" / "raw.npz"
    frame = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-09T00:00:00Z", "2026-08-09T00:01:00Z"], utc=True),
        "open": [1.0, 2.0],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.2, 2.2],
        "volume": [10, 20],
    })

    save_mt5_npz(frame, target)

    assert target.is_file()
    assert not target.with_name(target.name + ".part").exists()
    with np.load(target, allow_pickle=False) as data:
        assert data["close"].tolist() == [1.2, 2.2]
        assert len(data["time"]) == 2
        assert int(data["time"][0]) == 1_786_233_600_000_000_000
    loaded = load_ohlc(target)
    assert str(loaded.index[0]) == "2026-08-09 00:00:00+00:00"


def test_load_ohlc_accepts_legacy_epoch_second_npz(tmp_path: Path):
    target = tmp_path / "legacy_seconds.npz"
    np.savez_compressed(
        target,
        time=np.asarray([1_767_225_600, 1_767_225_660], dtype=np.int64),
        open=np.asarray([1.0, 2.0]), high=np.asarray([1.0, 2.0]),
        low=np.asarray([1.0, 2.0]), close=np.asarray([1.0, 2.0]),
    )
    loaded = load_ohlc(target)
    assert str(loaded.index[0]) == "2026-01-01 00:00:00+00:00"
    assert str(loaded.index[1]) == "2026-01-01 00:01:00+00:00"


def test_list_mt5_symbols_and_fetch_many_without_real_terminal(tmp_path, monkeypatch):
    import types
    from zigzag_miner import mt5_source

    class FakeSymbol:
        def __init__(self, name, path, visible=False, digits=2):
            self.name = name
            self.description = f"Mo ta {name}"
            self.path = path
            self.visible = visible
            self.select = visible
            self.digits = digits
            self.currency_base = "USD"
            self.currency_profit = "USD"
            self.currency_margin = "USD"
            self.trade_mode = 4

    dtype = [
        ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
        ("close", "f8"), ("tick_volume", "i8"), ("spread", "i8"), ("real_volume", "i8"),
    ]

    class FakeMt5:
        TIMEFRAME_M1 = 1
        TIMEFRAME_M2 = 2
        TIMEFRAME_M3 = 3
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1 = 60
        TIMEFRAME_H4 = 240
        TIMEFRAME_D1 = 1440
        def initialize(self, *args, **kwargs): return True
        def shutdown(self): return None
        def last_error(self): return (0, "ok")
        def account_info(self):
            return types.SimpleNamespace(login=123456, server="Demo-Server", name="Tester", company="Broker")
        def terminal_info(self):
            return types.SimpleNamespace(connected=True, path="C:/mt5/terminal64.exe", maxbars=100000)
        def symbol_info(self, symbol):
            return types.SimpleNamespace(digits=2, point=0.01, trade_contract_size=1.0,
                trade_tick_size=0.01, trade_tick_value=0.01, trade_calc_mode=0,
                currency_base="USD", currency_profit="USD", currency_margin="USD")
        def symbols_get(self):
            return [FakeSymbol("BTCUSDm", "Crypto", True, 2), FakeSymbol("XAUUSDm", "Metals", False, 3)]
        def symbol_select(self, symbol, enabled): return True
        def copy_rates_from_pos(self, symbol, timeframe, start, count):
            total = 5 if symbol == "BTCUSDm" else 3
            if start >= total: return np.empty(0, dtype=dtype)
            n = min(count, total - start)
            rows = np.empty(n, dtype=dtype)
            base = 1_700_000_000 - start * 60
            for i in range(n):
                price = 100.0 + start + i
                rows[i] = (base - i * 60, price, price + 1, price - 1, price + 0.5, 10, 2, 5)
            return rows

    fake = FakeMt5()
    monkeypatch.setattr(mt5_source, "_import_mt5", lambda: fake)

    symbols = mt5_source.list_mt5_symbols()
    assert [row["name"] for row in symbols] == ["BTCUSDm", "XAUUSDm"]
    assert symbols[0]["visible"] is True

    summary = mt5_source.fetch_mt5_many(
        ["BTCUSDm", "XAUUSDm"], "M1", 100, tmp_path / "raw", chunk_size=2
    )
    assert summary["requested_symbols"] == 2
    assert summary["ok_symbols"] == 2
    assert (tmp_path / "raw" / "BTCUSDm_M1.npz").is_file()
    assert (tmp_path / "raw" / "XAUUSDm_M1.npz").is_file()
    assert [row["rows"] for row in summary["results"]] == [5, 3]
    assert summary["terminal_maxbars"] == 100000


def test_mt5_uses_requested_portable_terminal(tmp_path, monkeypatch):
    import types
    from zigzag_miner import mt5_source

    terminal = tmp_path / "mt5_v2_runner" / "terminal64.exe"
    terminal.parent.mkdir(parents=True)
    terminal.write_bytes(b"fake")

    calls = []
    class FakeMt5:
        def initialize(self, *args, **kwargs):
            calls.append((args, kwargs))
            return True
        def account_info(self):
            return types.SimpleNamespace(login=987654, server="Broker-Live", name="A", company="B")
        def terminal_info(self):
            return types.SimpleNamespace(connected=True, path=str(terminal), maxbars=100000)
        def symbols_get(self): return []
        def shutdown(self): return None
        def last_error(self): return (0, "ok")

    monkeypatch.setattr(mt5_source, "_import_mt5", lambda: FakeMt5())
    rows, session = mt5_source.list_mt5_symbols(terminal_path=terminal, return_session=True)
    assert rows == []
    assert session["login"] == 987654
    assert session["server"] == "Broker-Live"
    assert calls and calls[0][0][0] == str(terminal.resolve())
    assert calls[0][1]["portable"] is True


def test_fetch_many_resumes_existing_atomic_file_and_status_is_cp1252_safe(tmp_path, monkeypatch):
    import io
    import sys
    import types
    from zigzag_miner import mt5_source

    dtype = [
        ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
        ("close", "f8"), ("tick_volume", "i8"), ("spread", "i8"), ("real_volume", "i8"),
    ]

    class FakeMt5:
        TIMEFRAME_M1 = 1
        def initialize(self, *args, **kwargs): return True
        def shutdown(self): return None
        def last_error(self): return (0, "ok")
        def account_info(self):
            return types.SimpleNamespace(login=1, server="Demo", name="T", company="B")
        def terminal_info(self):
            return types.SimpleNamespace(connected=True, path="C:/mt5/terminal64.exe", maxbars=100000)
        def symbol_info(self, symbol):
            return types.SimpleNamespace(digits=2, point=0.01, trade_contract_size=1.0,
                trade_tick_size=0.01, trade_tick_value=0.01, trade_calc_mode=0,
                currency_base="USD", currency_profit="USD", currency_margin="USD")
        def symbol_select(self, symbol, enabled): return True
        def copy_rates_from_pos(self, symbol, timeframe, start, count):
            if symbol == "BROKEN":
                return np.empty(0, dtype=dtype)
            if start >= 2:
                return np.empty(0, dtype=dtype)
            n = min(count, 2 - start)
            rows = np.empty(n, dtype=dtype)
            for i in range(n):
                price = 100 + start + i
                rows[i] = (1_700_000_000 - (start + i) * 60, price, price + 1, price - 1, price, 1, 1, 1)
            return rows

    fake = FakeMt5()
    monkeypatch.setattr(mt5_source, "_import_mt5", lambda: fake)
    out = tmp_path / "raw"
    out.mkdir()
    existing = out / "EXISTING_M1.npz"
    frame = pd.DataFrame({
        "time": pd.to_datetime(["2026-08-09T00:00:00Z"], utc=True),
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1],
    })
    mt5_source.save_mt5_npz(frame, existing)

    raw = io.BytesIO()
    cp1252_stdout = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    monkeypatch.setattr(sys, "stdout", cp1252_stdout)

    summary = mt5_source.fetch_mt5_many(
        ["EXISTING", "BROKEN"], "M1", 10, out, chunk_size=2
    )
    assert summary["processed_symbols"] == 2
    assert summary["results"][0]["skipped_existing"] is True
    assert summary["results"][1]["ok"] is False
    assert (out / "manifest.json").is_file()
    cp1252_stdout.flush()
    text = raw.getvalue().decode("cp1252")
    assert "SKIP" in text
    assert "ERROR" in text


def test_fetch_caps_chunk_and_rows_to_terminal_maxbars(tmp_path, monkeypatch):
    import types
    from zigzag_miner import mt5_source

    dtype = [
        ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
        ("close", "f8"), ("tick_volume", "i8"), ("spread", "i8"), ("real_volume", "i8"),
    ]
    calls = []

    class FakeMt5:
        TIMEFRAME_M1 = 1
        def initialize(self, *args, **kwargs): return True
        def shutdown(self): return None
        def last_error(self): return (1, "Success")
        def account_info(self): return types.SimpleNamespace(login=1, server="Demo", name="T", company="B")
        def terminal_info(self): return types.SimpleNamespace(connected=True, path="C:/mt5/terminal64.exe", maxbars=5)
        def symbol_info(self, symbol):
            return types.SimpleNamespace(digits=2, point=0.01, trade_contract_size=1.0,
                trade_tick_size=0.01, trade_tick_value=0.01, trade_calc_mode=0,
                currency_base="USD", currency_profit="USD", currency_margin="USD")
        def symbol_select(self, symbol, enabled): return True
        def copy_rates_from_pos(self, symbol, timeframe, start, count):
            calls.append((start, count))
            assert count < 5
            rows = np.empty(count, dtype=dtype)
            for i in range(count):
                price = 100 + start + i
                rows[i] = (1_700_000_000 - (start + i) * 60, price, price, price, price, 1, 1, 1)
            return rows

    monkeypatch.setattr(mt5_source, "_import_mt5", lambda: FakeMt5())
    summary = mt5_source.fetch_mt5_many(["A"], "M1", 20, tmp_path, chunk_size=200000)
    assert calls == [(0, 4), (4, 1)]
    assert summary["results"][0]["rows"] == 5
    assert summary["results"][0]["limited_by_terminal_maxbars"] is True
    assert summary["results"][0]["history_limit_reason"] == "terminal_maxbars"


def test_selected_cli_returns_nonzero_when_any_symbol_failed(tmp_path, monkeypatch):
    from zigzag_miner import cli

    summary = {
        "requested_symbols": 1, "ok_symbols": 0, "failed_symbols": 1,
        "timeframe": "M1", "bars_per_symbol": 10, "terminal_maxbars": 100,
        "mt5_session": {}, "output_dir": str(tmp_path), "results": [],
    }
    monkeypatch.setattr(cli, "fetch_mt5_many", lambda *args, **kwargs: summary)
    args = SimpleNamespace(
        symbols_file=None, symbols="BROKEN", timeframe="M1", bars=10,
        output=str(tmp_path), chunk_size=10, terminal=None, runner_dir=None,
    )
    with pytest.raises(SystemExit) as exc:
        cli.cmd_mt5_selected(args)
    assert exc.value.code == 1
    assert (tmp_path / "manifest.json").is_file()


def test_mine_five_selected_keeps_one_dataset_per_symbol(tmp_path, monkeypatch):
    from zigzag_miner import cli

    raw = tmp_path / "raw"
    raw.mkdir()
    results = []
    for symbol in ("AAA", "BBB"):
        source = raw / f"{symbol}_M1.npz"
        source.write_bytes(b"placeholder")
        results.append({"symbol": symbol, "ok": True, "output": str(source)})
    (raw / "manifest.json").write_text(json.dumps({
        "timeframe": "M1", "results": results,
    }), encoding="utf-8")

    def fake_mine(**kwargs):
        output = Path(kwargs["output_path"])
        (output / "five_market_tape").mkdir(parents=True)
        (output / "five_market_tape" / "manifest.json").write_text("{}", encoding="utf-8")
        return {"samples": 2 if kwargs["symbol"] == "AAA" else 3, "shards": 1}

    monkeypatch.setattr(cli, "_mine_five_dataset", fake_mine)
    output = tmp_path / "collection"
    args = SimpleNamespace(
        input_dir=str(raw), pattern="pattern.yaml", output=str(output),
        project_root=".", config="config.yaml", atr_period=None,
        zigzag_atr_mult=1.0, shard_size=100, default_spread_points=None,
        allow_neutral_volume=False,
    )
    cli.cmd_mine_five_selected(args)
    collection = json.loads((output / "dataset_collection.json").read_text(encoding="utf-8"))
    assert collection["symbols"] == 2
    assert collection["samples"] == 5
    assert [row["path"] for row in collection["datasets"]] == ["symbols/AAA", "symbols/BBB"]
