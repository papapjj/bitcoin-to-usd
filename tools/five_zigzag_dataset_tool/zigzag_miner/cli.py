from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_ohlc, trim_sparse_timeframe_prefix, write_npz_shards, write_indexed_five_shards
from .miner import mine_dataframe
from .mt5_source import fetch_mt5, fetch_mt5_many, list_mt5_symbols, safe_symbol_filename, save_mt5_npz
from .project_five_adapter import ProjectFiveContract
from .structures import load_pattern
from .synthetic import repeated_swing_market




def _terminal_from_args(args):
    terminal = getattr(args, "terminal", None)
    if terminal:
        return str(Path(terminal))
    runner = getattr(args, "runner_dir", None)
    if runner:
        return str(Path(runner) / "terminal64.exe")
    return None


def cmd_mine(args):
    df = load_ohlc(args.input)
    spec = load_pattern(args.pattern)
    samples = mine_dataframe(
        df,
        spec,
        atr_period=args.atr_period,
        zigzag_atr_mult=args.zigzag_atr_mult,
        atr_mode=args.atr_mode,
    )
    summary = write_npz_shards(samples, args.output, shard_size=args.shard_size)
    print(json.dumps(summary, indent=2))


def _mine_five_dataset(
    *, input_path, pattern_path, output_path, project_root, config_path,
    atr_period=None, zigzag_atr_mult=1.0, shard_size=100000,
    default_spread_points=None, allow_neutral_volume=False, symbol="", symbol_contract=None,
    timeframe="",
):
    df = load_ohlc(input_path)
    df, sparse_prefix_rows_removed = trim_sparse_timeframe_prefix(df, timeframe)
    spec = load_pattern(pattern_path)
    contract = ProjectFiveContract.load(project_root, config_path, symbol_contract)
    project_raw = contract.prepare_project_raw(
        df,
        default_spread_points=default_spread_points,
        allow_neutral_volume=bool(allow_neutral_volume),
    )
    feature_frame = contract.build_feature_frame(project_raw)
    tape_manifest = contract.write_tape(feature_frame, output_path)

    resolved_atr_period = int(atr_period or contract.config.features.atr_period)
    samples = mine_dataframe(
        df,
        spec,
        atr_period=resolved_atr_period,
        zigzag_atr_mult=zigzag_atr_mult,
        atr_mode="mt5_sma",
    )
    enriched = contract.enrich_index_samples(samples, df, feature_frame)
    metadata = contract.metadata(tape_manifest=tape_manifest)
    metadata.update({
        "zigzag_atr_period": resolved_atr_period,
        "zigzag_atr_mode": "mt5_sma",
        "pattern": spec.name,
        "history_bars": int(spec.history_bars),
        "future_bars": int(spec.future_bars),
        "sample_contract": "context_excludes_anchor_v1",
        "symbol": str(symbol or ""),
        "symbol_contract": dict(symbol_contract or {}),
        "source": str(input_path),
        "timeframe": str(timeframe or "").upper(),
        "sparse_prefix_rows_removed": int(sparse_prefix_rows_removed),
    })
    summary = write_indexed_five_shards(
        enriched,
        output_path,
        shard_size=shard_size,
        dataset_metadata=metadata,
    )
    return summary


def cmd_mine_five(args):
    summary = _mine_five_dataset(
        input_path=args.input, pattern_path=args.pattern, output_path=args.output,
        project_root=args.project_root, config_path=args.config,
        atr_period=args.atr_period, zigzag_atr_mult=args.zigzag_atr_mult,
        shard_size=args.shard_size, default_spread_points=args.default_spread_points,
        allow_neutral_volume=args.allow_neutral_volume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_mine_five_selected(args):
    raw_dir = Path(args.input_dir)
    raw_manifest_path = raw_dir / "manifest.json"
    if not raw_manifest_path.is_file():
        raise FileNotFoundError(f"raw MT5 manifest not found: {raw_manifest_path}")
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    selected = [row for row in raw_manifest.get("results", []) if bool(row.get("ok"))]
    if not selected:
        raise ValueError("raw MT5 manifest has no successful symbols")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = []
    failures = []
    for index, row in enumerate(selected, start=1):
        symbol = str(row.get("symbol", "") or "").strip()
        safe = safe_symbol_filename(symbol)
        raw_file = Path(str(row.get("output", "") or ""))
        if not raw_file.is_absolute():
            raw_file = Path.cwd() / raw_file
        if not raw_file.is_file():
            candidate = raw_dir / f"{safe}_{str(raw_manifest.get('timeframe', 'M1')).upper()}.npz"
            raw_file = candidate
        dataset_dir = output / "symbols" / safe
        try:
            summary = _mine_five_dataset(
                input_path=raw_file, pattern_path=args.pattern, output_path=dataset_dir,
                project_root=args.project_root, config_path=args.config,
                atr_period=args.atr_period, zigzag_atr_mult=args.zigzag_atr_mult,
                shard_size=args.shard_size, default_spread_points=args.default_spread_points,
                allow_neutral_volume=args.allow_neutral_volume, symbol=symbol,
                symbol_contract=dict(row.get("symbol_contract") or {}),
                timeframe=str(raw_manifest.get("timeframe", "M1") or "M1"),
            )
            datasets.append({
                "symbol": symbol,
                "path": str(dataset_dir.relative_to(output)).replace("\\", "/"),
                "source": str(raw_file),
                "samples": int(summary.get("samples", 0)),
                "shards": int(summary.get("shards", 0)),
            })
            print(f"[{index}/{len(selected)}] {symbol}: OK ({int(summary.get('samples', 0))} samples)", flush=True)
        except Exception as exc:
            failures.append({"symbol": symbol, "source": str(raw_file), "error": str(exc)})
            print(f"[{index}/{len(selected)}] {symbol}: ERROR", flush=True)
    collection = {
        "schema_version": 1,
        "format": "five-dataset-collection-v1",
        "raw_manifest": str(raw_manifest_path),
        "symbols": len(datasets),
        "active_symbols": sum(int(item["samples"]) > 0 for item in datasets),
        "zero_sample_symbols": [item["symbol"] for item in datasets if int(item["samples"]) == 0],
        "failed_symbols": len(failures),
        "samples": sum(int(item["samples"]) for item in datasets),
        "shards": sum(int(item["shards"]) for item in datasets),
        "datasets": datasets,
        "failures": failures,
    }
    (output / "dataset_collection.json").write_text(
        json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "dataset_summary.json").write_text(
        json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(collection, ensure_ascii=False, indent=2))
    if failures or not datasets or int(collection["samples"]) <= 0:
        raise SystemExit(1)



def cmd_check_five(args):
    contract = ProjectFiveContract.load(args.project_root, args.config)
    report = {
        "input_count": contract.input_count,
        "agent_count": contract.agent_count,
        "feature_schema_version": contract.feature_schema_version,
        "feature_schema_hash": contract.feature_schema_hash,
        "input_normalization_mode": str(getattr(contract.config.learning, "input_normalization_mode", "raw") or "raw"),
        "decision_mode": str(getattr(contract.config.he_chuyen_gia, "che_do_quyet_dinh", "") or ""),
        "environment_columns": len(contract.environment_columns),
        "ok": True,
    }
    if args.input:
        df = load_ohlc(args.input)
        project_raw = contract.prepare_project_raw(
            df,
            default_spread_points=args.default_spread_points,
            allow_neutral_volume=bool(args.allow_neutral_volume),
        )
        feature_frame = contract.build_feature_frame(project_raw)
        report["input_rows"] = int(len(df))
        report["feature_rows_after_warmup"] = int(len(feature_frame))
        report["warmup_rows_removed"] = int(len(df) - len(feature_frame))
    print(json.dumps(report, ensure_ascii=False, indent=2))

def cmd_mt5(args):
    df = fetch_mt5(args.symbol, args.timeframe, args.bars, chunk_size=args.chunk_size, terminal_path=_terminal_from_args(args))
    save_mt5_npz(df, args.output)
    print(json.dumps({
        "requested_rows": int(args.bars),
        "rows": int(len(df)),
        "history_exhausted": bool(len(df) < int(args.bars)),
        "limited_by_terminal_maxbars": bool(df.attrs.get("limited_by_terminal_maxbars", False)),
        "terminal_maxbars": int(df.attrs.get("terminal_maxbars", 0) or 0),
        "output": str(Path(args.output)),
    }, indent=2))



def cmd_mt5_symbols(args):
    symbols, session = list_mt5_symbols(terminal_path=_terminal_from_args(args), return_session=True)
    print(json.dumps({
        "count": len(symbols),
        "mt5_session": session,
        "symbols": symbols,
    }, ensure_ascii=False))


def cmd_mt5_selected(args):
    if args.symbols_file:
        symbols = [
            line.strip()
            for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        symbols = [part.strip() for part in str(args.symbols or "").split(";") if part.strip()]
    summary = fetch_mt5_many(
        symbols,
        args.timeframe,
        args.bars,
        args.output,
        chunk_size=args.chunk_size,
        terminal_path=_terminal_from_args(args),
        resume=not bool(getattr(args, "no_resume", False)),
    )
    output_dir = Path(args.output)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "requested_symbols": summary["requested_symbols"],
        "ok_symbols": summary["ok_symbols"],
        "failed_symbols": summary["failed_symbols"],
        "timeframe": summary["timeframe"],
        "bars_per_symbol": summary["bars_per_symbol"],
        "terminal_maxbars": summary.get("terminal_maxbars", 0),
        "mt5_session": summary.get("mt5_session", {}),
        "output_dir": summary["output_dir"],
        "manifest": str(manifest),
    }, ensure_ascii=False, indent=2))
    if int(summary["failed_symbols"]) > 0:
        raise SystemExit(1)

def cmd_smoke(args):
    df = repeated_swing_market(repeats=60)
    spec = load_pattern(args.pattern)
    samples = list(
        mine_dataframe(
            df,
            spec,
            atr_period=args.atr_period,
            zigzag_atr_mult=args.zigzag_atr_mult,
            atr_mode=args.atr_mode,
        )
    )
    print(json.dumps({
        "bars": len(df),
        "samples": len(samples),
        "first_anchor": samples[0]["anchor_index"] if samples else None,
        "last_anchor": samples[-1]["anchor_index"] if samples else None,
    }, indent=2))
    if not samples:
        raise SystemExit("smoke failed: zero pattern matches")


def build_parser():
    p = argparse.ArgumentParser(prog="five-zigzag", description="Cong cu dao cau truc ZigZag nhan qua")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="Dao mau doc lap tu CSV/NPZ/Parquet")
    m.add_argument("--input", required=True)
    m.add_argument("--pattern", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--atr-period", type=int, default=14)
    m.add_argument("--atr-mode", choices=("mt5_sma", "wilder"), default="mt5_sma")
    m.add_argument("--zigzag-atr-mult", type=float, default=1.0)
    m.add_argument("--shard-size", type=int, default=10000)
    m.set_defaults(func=cmd_mine)

    mf = sub.add_parser("mine-five", help="Dao tap du lieu dung giao uoc 137 va TradingEnvironment hien tai")
    mf.add_argument("--input", required=True)
    mf.add_argument("--pattern", required=True)
    mf.add_argument("--output", required=True)
    mf.add_argument("--project-root", required=True)
    mf.add_argument("--config", default="configs/he_chuyen_gia_latest_ui_v193.yaml")
    mf.add_argument("--atr-period", type=int, default=None, help="Mac dinh lay atr_period tu cau hinh du an")
    mf.add_argument("--zigzag-atr-mult", type=float, default=1.0)
    mf.add_argument("--shard-size", type=int, default=100000)
    mf.add_argument("--default-spread-points", type=float, default=None)
    mf.add_argument("--allow-neutral-volume", action="store_true")
    mf.set_defaults(func=cmd_mine_five)

    mm = sub.add_parser("mine-five-selected", help="Dao tap Five cho moi symbol trong raw MT5 manifest")
    mm.add_argument("--input-dir", required=True)
    mm.add_argument("--pattern", required=True)
    mm.add_argument("--output", required=True)
    mm.add_argument("--project-root", required=True)
    mm.add_argument("--config", default="configs/he_chuyen_gia_latest_ui_v193.yaml")
    mm.add_argument("--atr-period", type=int, default=None)
    mm.add_argument("--zigzag-atr-mult", type=float, default=1.0)
    mm.add_argument("--shard-size", type=int, default=100000)
    mm.add_argument("--default-spread-points", type=float, default=None)
    mm.add_argument("--allow-neutral-volume", action="store_true")
    mm.set_defaults(func=cmd_mine_five_selected)

    cf = sub.add_parser("check-five", help="Kiem tra giao uoc du an va nguon du lieu truoc khi dao")
    cf.add_argument("--project-root", required=True)
    cf.add_argument("--config", default="configs/he_chuyen_gia_latest_ui_v193.yaml")
    cf.add_argument("--input", default=None)
    cf.add_argument("--default-spread-points", type=float, default=None)
    cf.add_argument("--allow-neutral-volume", action="store_true")
    cf.set_defaults(func=cmd_check_five)

    f = sub.add_parser("mt5-fetch", help="Lay thanh gia lich su cua mot ma tu MetaTrader 5 cuc bo")
    f.add_argument("--symbol", required=True)
    f.add_argument("--timeframe", default="M1")
    f.add_argument("--bars", type=int, required=True)
    f.add_argument("--chunk-size", type=int, default=200000)
    f.add_argument("--output", required=True)
    f.add_argument("--runner-dir", default=None, help="Thu muc MT5 runner; se dung terminal64.exe portable trong thu muc nay")
    f.add_argument("--terminal", default=None, help="Duong dan terminal64.exe; uu tien hon --runner-dir")
    f.set_defaults(func=cmd_mt5)

    ls = sub.add_parser("mt5-symbols", help="Quet toan bo ma ma terminal MT5 hien tai nhin thay")
    ls.add_argument("--runner-dir", default=None, help="Thu muc MT5 runner; dung cung terminal portable voi huan luyen")
    ls.add_argument("--terminal", default=None, help="Duong dan terminal64.exe; uu tien hon --runner-dir")
    ls.set_defaults(func=cmd_mt5_symbols)

    fm = sub.add_parser("mt5-fetch-selected", help="Lay lich su cho cac ma duoc chon tu danh sach MT5")
    selected_source = fm.add_mutually_exclusive_group(required=True)
    selected_source.add_argument("--symbols", default=None, help="Danh sach ma cach nhau bang dau cham phay")
    selected_source.add_argument("--symbols-file", default=None, help="Tep UTF-8, moi dong la mot ma da chon")
    fm.add_argument("--timeframe", default="M1")
    fm.add_argument("--bars", type=int, required=True)
    fm.add_argument("--chunk-size", type=int, default=200000)
    fm.add_argument("--output", required=True, help="Thu muc dau ra; moi ma duoc luu thanh mot tep rieng")
    fm.add_argument("--no-resume", action="store_true", help="Tai lai va thay the atomic cac tep NPZ da ton tai")
    fm.add_argument("--runner-dir", default=None, help="Thu muc MT5 runner; dung cung terminal portable voi huan luyen")
    fm.add_argument("--terminal", default=None, help="Duong dan terminal64.exe; uu tien hon --runner-dir")
    fm.set_defaults(func=cmd_mt5_selected)

    s = sub.add_parser("smoke", help="Chay kiem tra nhanh bang thi truong gia lap")
    s.add_argument("--pattern", required=True)
    s.add_argument("--atr-period", type=int, default=14)
    s.add_argument("--atr-mode", choices=("mt5_sma", "wilder"), default="mt5_sma")
    s.add_argument("--zigzag-atr-mult", type=float, default=1.0)
    s.set_defaults(func=cmd_smoke)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
