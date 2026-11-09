from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import load_config
from .continuous_session import ContinuousRuntimeConfig
from .champion_retention import run_context_expert_champion_training_mt5
from .live_observatory import serve_live_observatory
from .legacy_pickle_migration import load_pickle_with_removed_context_structural_migration
from .mt5_runner import MT5_MODEL_REAL_TICKS
from .chuyen_gia_policy import (
    backup_context_expert_artifacts,
    huan_luyen_he_chuyen_gia_mt5,
    tao_he_chuyen_gia_mot_context_mot_policy,
    walk_forward_he_chuyen_gia_mt5,
)
# Legacy commands are retained as explicit compatibility aliases. They are not
# used by the one-context owner workflow, but keeping their parser contracts
# prevents old automation from silently invoking the wrong command.
from .self_growing_runner import (
    compare_policy_cores_mt5,
    continuous_train_mt5,
    improve_self_growing,
    train_self_growing_mt5,
)


def _utf8() -> None:
    configure = getattr(sys.stdout, "reconfigure", None)
    if callable(configure):
        try:
            configure(encoding="utf-8")
        except Exception:
            pass


# Historical name kept for automation/tests that call the helper directly.
def _ensure_utf8_stdout() -> None:
    _utf8()


def _reject_unsafe_direct_online_resume(
    config: Any,
    args: argparse.Namespace,
) -> None:
    """Keep an online continuation behind the champion-shadow transaction.

    The low-level trainer remains available to the champion orchestrator and
    to explicit fresh research runs.  The public direct CLI must not resume a
    mutable checkpoint because that path can execute and later persist an
    unqualified candidate without the artifact-bound frozen-OOS matrix.
    """
    if str(getattr(getattr(config, "environment", None), "backend", "mt5")) == "python_model1":
        # Immutable tape replay cannot mutate MT5 or promote an artifact. It
        # always writes the explicitly requested output and is verified
        # against every captured decision before persistence.
        return
    if not bool(getattr(config, "mt5_learning_enabled", True)):
        return
    if bool(getattr(args, "fresh_system", False)):
        if getattr(args, "seed_system", None):
            raise ValueError(
                "--fresh-system khong the dung cung --seed-system; bo seed hoac "
                "dung huan-luyen-champion-he-chuyen-gia-mt5"
            )
        return
    seed_system = getattr(args, "seed_system", None)
    output = Path(str(getattr(args, "output", "") or ""))
    continuation = Path(str(seed_system)) if seed_system else (output if output.is_file() else None)
    if continuation is None:
        return
    raise ValueError(
        "Tu choi online direct-resume tu checkpoint "
        f"'{continuation}'. Hay dung huan-luyen-champion-he-chuyen-gia-mt5 "
        "de train shadow va chi promote sau ma tran frozen OOS. Neu can mot "
        "artifact nghien cuu doc lap, dung --fresh-system voi --output rieng."
    )


def _them_log_giai_thich(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live-observatory", action="store_true")
    parser.add_argument("--live-events", default=None)
    parser.add_argument("--live-sample-every", type=int, default=None)
    parser.add_argument("--live-max-mb", type=int, default=None)
    parser.add_argument("--live-reset", action="store_true")
    parser.add_argument("--live-decision-profiles", default=None)
    parser.add_argument("--live-structural-events", default=None)
    parser.add_argument("--live-network-development", default=None)
    parser.add_argument("--live-max-responsible-neurons", type=int, default=None)
    parser.add_argument("--live-max-responsible-edges", type=int, default=None)


def _apply_observability(config: Any, args: argparse.Namespace):
    obs = getattr(config, "observability", None)
    if obs is None:
        return config
    enabled = bool(getattr(obs, "enabled", False) or getattr(args, "live_observatory", False) or getattr(args, "live_events", None))
    values: dict[str, Any] = {
        "enabled": enabled,
        "event_path": str(getattr(args, "live_events", None) or getattr(obs, "event_path", "outputs/quan_sat_chuyen_gia_live.jsonl")),
        "decision_profile_path": str(getattr(args, "live_decision_profiles", None) or getattr(obs, "decision_profile_path", "outputs/ho_so_quyet_dinh_chi_tiet.jsonl")),
        "structural_event_path": str(getattr(args, "live_structural_events", None) or getattr(obs, "structural_event_path", "outputs/nhat_ky_phat_trien_neuron_lien_ket.jsonl")),
        "network_development_path": str(getattr(args, "live_network_development", None) or getattr(obs, "network_development_path", "outputs/theo_doi_phat_trien_mang.jsonl")),
    }
    if getattr(args, "live_sample_every", None) is not None:
        values["decision_sample_every"] = int(args.live_sample_every)
    if getattr(args, "live_max_mb", None) is not None:
        values["max_event_bytes"] = max(0, int(args.live_max_mb)) * 1024 * 1024
    for name, value in (
        ("max_responsible_neurons", getattr(args, "live_max_responsible_neurons", None)),
        ("max_responsible_edges", getattr(args, "live_max_responsible_edges", None)),
    ):
        if value is not None:
            values[name] = int(value)
    try:
        return replace(config, observability=replace(obs, **values))
    except TypeError:
        return config


def _xoa_log_quan_sat_neu_duoc_yeu_cau(config: Any, args: argparse.Namespace) -> None:
    if not bool(getattr(args, "live_reset", False)):
        return
    obs = getattr(config, "observability", None)
    if obs is None:
        return
    paths = (
        str(getattr(obs, "event_path", "") or ""),
        str(getattr(obs, "decision_profile_path", "") or ""),
        str(getattr(obs, "structural_event_path", "") or ""),
        str(getattr(obs, "network_development_path", "") or ""),
    )
    for raw in paths:
        if not raw.strip():
            continue
        path = Path(raw)
        for candidate in (path, path.with_suffix(path.suffix + ".status.json")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass


def _require_context_expert_config(config: Any) -> None:
    he = getattr(config, "he_chuyen_gia", None)
    if not bool(getattr(he, "bat", False)):
        raise ValueError("Cần dùng config he_chuyen_gia.bat=true.")
    mode = str(getattr(he, "che_do_quyet_dinh", "") or "")
    shared = bool(getattr(he, "cho_phep_policy_chung_ra_lenh", False))
    if mode in {"mot_bo_nao_dung_chung", "nam_tac_nhan_dung_chung"}:
        if not shared:
            raise ValueError("Che do dung chung can cho_phep_policy_chung_ra_lenh=true.")
    elif shared:
        raise ValueError("Che do mot-context-mot-chuyen-gia khong duoc cho policy chung ra lenh.")
    if bool(getattr(he, "cho_phep_tron_policy", True)):
        raise ValueError("Config khong hop le: tron policy da bi cam tuyet doi.")


def _them_mt5_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeframe",
        "--period",
        dest="timeframe",
        default=None,
        choices=["M1", "M5", "M15", "M30", "H1", "m1", "m5", "m15", "m30", "h1"],
        help="Chart/trading timeframe for MT5 Strategy Tester (e.g. M1, M5, M15)",
    )
    parser.add_argument(
        "--offline-preflight",
        action="store_true",
        default=False,
        help="Allow MT5 Strategy Tester to execute offline on local cached history without requiring live broker synchronization",
    )


def _apply_cli_overrides(config: Any, args: argparse.Namespace):
    orders = getattr(config, "orders", None)
    if orders is not None:
        order_kwargs: dict[str, Any] = {}
        min_daily = getattr(args, "min_daily_trades", None)
        max_daily = getattr(args, "max_daily_trades", None)
        if min_daily is not None or max_daily is not None:
            if min_daily is None:
                min_daily = int(getattr(orders, "min_daily_trades", 0) or 0)
            if max_daily is None:
                max_daily = int(getattr(orders, "max_daily_trades", 0) or 0)
            if int(max_daily) > 0 and int(max_daily) < int(min_daily):
                raise ValueError("max-daily-trades must be 0 or >= min-daily-trades")
            order_kwargs["min_daily_trades"] = int(min_daily)
            order_kwargs["max_daily_trades"] = int(max_daily)
        tf = getattr(args, "timeframe", None)
        if tf is not None and str(tf).strip():
            order_kwargs["timeframe"] = str(tf).strip().upper()
        if order_kwargs:
            try:
                config = replace(config, orders=replace(orders, **order_kwargs))
            except TypeError:
                pass
    offline_preflight = bool(getattr(args, "offline_preflight", False))
    if offline_preflight:
        env = getattr(config, "environment", None)
        if env is not None:
            try:
                config = replace(config, environment=replace(env, offline_preflight=True))
            except TypeError:
                pass
    return config


def _apply_order_overrides(config: Any, args: argparse.Namespace):
    return _apply_cli_overrides(config, args)


def _close_scope(args: argparse.Namespace) -> tuple[bool, str]:
    requested = str(getattr(args, "mt5_close_scope", "trial") or "trial")
    if bool(getattr(args, "keep_mt5_open", False)):
        requested = "never"
    if requested not in {"trial", "round", "never"}:
        raise ValueError("mt5-close-scope must be one of: trial, round, never")
    return requested != "never", requested


def _add_legacy_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", default="outputs/winner.pkl")
    parser.add_argument("--seed-policy", default=None)
    parser.add_argument("--runner-dir", default="mt5_v2_runner")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--mt5-visual", action="store_true")
    parser.add_argument("--keep-mt5-open", action="store_true")
    parser.add_argument("--mt5-close-scope", choices=["trial", "round", "never"], default="trial")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--model", type=int, choices=[0, 1, 2, MT5_MODEL_REAL_TICKS], default=1)
    parser.add_argument("--mutation-sigma", type=float, default=0.08)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--m1-backend", choices=["direct_rnn", "m1_graph", "graph"], default="direct_rnn")
    parser.add_argument("--policy-core", choices=["current", "world"], default="current")
    parser.add_argument("--curriculum-learning", action="store_true")
    parser.add_argument("--min-daily-trades", type=int, default=None)
    parser.add_argument("--max-daily-trades", type=int, default=None)
    _them_mt5_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="self_growing_trader", description="He mot context - mot chuyen gia - mot policy")
    parser.add_argument("--config", default="configs/he_chuyen_gia.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("khoi-tao-he-chuyen-gia", help="Tao kho 5 chuyen gia goc doc lap")
    init.add_argument("--output", default="outputs/he_chuyen_gia/kho_chuyen_gia.pkl")
    init.add_argument("--seed", type=int, default=0)

    train = sub.add_parser(
        "huan-luyen-he-chuyen-gia-mt5",
        help="Train artifact nghiên cứu mới; resume online phải dùng champion-shadow",
    )
    train.add_argument("--output", default="outputs/he_chuyen_gia/kho_chuyen_gia.pkl")
    train.add_argument(
        "--seed-system",
        default=None,
        help="Chỉ dùng với --frozen; online continuation phải dùng lệnh champion-shadow",
    )
    train.add_argument(
        "--fresh-system",
        action="store_true",
        help="Bỏ qua output hiện có và tạo artifact nghiên cứu mới",
    )
    train.add_argument(
        "--frozen",
        action="store_true",
        help="Chi inference tren MT5: tat toan bo learning de doi chieu frozen/online cung contract",
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--runner-dir", default="mt5_v2_runner")
    train.add_argument("--timeout-sec", type=int, default=300)
    train.add_argument("--max-wall-sec", type=int, default=0)
    train.add_argument("--mt5-visual", action="store_true")
    train.add_argument("--keep-mt5-open", action="store_true")
    train.add_argument("--from-date", default="2025.01.01")
    train.add_argument("--to-date", default=None)
    train.add_argument("--backtest-days", type=int, default=31)
    train.add_argument("--model", type=int, choices=[0, 1, 2, MT5_MODEL_REAL_TICKS], default=1)
    _them_log_giai_thich(train)
    _them_mt5_options(train)

    champion = sub.add_parser(
        "huan-luyen-champion-he-chuyen-gia-mt5",
        help="Train shadow va chi promote sau ma tran frozen OOS cung champion bat bien",
    )
    champion.add_argument("--champion", required=True)
    champion.add_argument("--output", default="outputs/he_chuyen_gia/candidates/champion_candidate.pkl")
    champion.add_argument("--audit-dir", default="outputs/he_chuyen_gia/audit/champion_shadow_matrix")
    champion.add_argument("--train-from", default="2025.01.01")
    champion.add_argument("--train-to", required=True)
    champion.add_argument("--validation-count", type=int, default=None)
    champion.add_argument("--validation-days", type=int, default=1)
    champion.add_argument("--candidate-seed-count", type=int, default=None)
    champion.add_argument("--seed", type=int, default=0)
    champion.add_argument("--runner-dir", default="mt5_v2_runner")
    champion.add_argument("--timeout-sec", type=int, default=300)
    champion.add_argument("--max-wall-sec", type=int, default=0)
    champion.add_argument("--mt5-visual", action="store_true")
    champion.add_argument("--keep-mt5-open", action="store_true")
    champion.add_argument(
        "--model", type=int, choices=[0, 1, 2, MT5_MODEL_REAL_TICKS],
        default=MT5_MODEL_REAL_TICKS,
        help="Model 4 real ticks la bat buoc de artifact du dieu kien promotion",
    )
    champion.add_argument("--paired-min-expectancy", type=float, default=0.0)
    champion.add_argument("--paired-max-profit-regression", type=float, default=0.0)
    champion.add_argument("--paired-max-expectancy-lcb-regression", type=float, default=0.02)
    champion.add_argument("--paired-max-drawdown-increase-pct", type=float, default=1.0)
    _them_log_giai_thich(champion)
    _them_mt5_options(champion)

    wf = sub.add_parser("walk-forward-he-chuyen-gia-mt5", help="Walk-forward OOS cho kho chuyen gia")
    wf.add_argument("--output", default="outputs/walk_forward_he_chuyen_gia")
    wf.add_argument("--log", default=None)
    wf.add_argument("--seed-system", default=None)
    wf.add_argument("--seed", type=int, default=0)
    wf.add_argument("--seed-count", type=int, default=3)
    wf.add_argument("--runner-dir", default="mt5_v2_runner")
    wf.add_argument("--timeout-sec", type=int, default=300)
    wf.add_argument("--mt5-visual", action="store_true")
    wf.add_argument("--keep-mt5-open", action="store_true")
    wf.add_argument("--from-date", default="2025.01.01")
    wf.add_argument("--to-date", default="2025.03.31")
    wf.add_argument("--train-window", type=int, default=14)
    wf.add_argument("--test-window", type=int, default=7)
    wf.add_argument("--step", type=int, default=7)
    wf.add_argument("--model", type=int, choices=[0, 1, 2, MT5_MODEL_REAL_TICKS], default=1)
    wf.add_argument(
        "--paired-baseline",
        action="store_true",
        help="Chay frozen seed cung cua so OOS va chan candidate hoi quy so voi seed",
    )
    wf.add_argument(
        "--no-rollback",
        action="store_true",
        help="Khong copy protected seed vao rollback path khi OOS gate fail",
    )
    wf.add_argument("--paired-min-expectancy", type=float, default=0.0)
    wf.add_argument("--paired-max-profit-regression", type=float, default=0.0)
    wf.add_argument("--paired-max-expectancy-lcb-regression", type=float, default=0.02)
    wf.add_argument("--paired-max-drawdown-increase-pct", type=float, default=1.0)
    _them_log_giai_thich(wf)
    _them_mt5_options(wf)

    admin = sub.add_parser("quan-tri-he-chuyen-gia", help="Thang cap/ngu dong/luu tru context owner")
    admin.add_argument("--system", "--policy", dest="system", required=True)
    admin.add_argument("--output", default=None)
    admin.add_argument("--context", required=True)
    admin.add_argument("--action", choices=["promote", "dormant", "archive", "merge"], required=True)
    admin.add_argument("--target-context", default="")
    admin.add_argument("--replay-pass", action="store_true")
    admin.add_argument("--oos-pass", action="store_true")
    admin.add_argument("--risk-pass", action="store_true")
    admin.add_argument("--non-interference-pass", action="store_true")
    admin.add_argument("--lifecycle-count", type=int, default=None)
    admin.add_argument("--note", default="")
    admin.add_argument("--audit-log", default=None)

    # Explicit legacy compatibility commands.
    legacy_train = sub.add_parser("train-mt5", help="Compatibility: train legacy self-growing policy")
    _add_legacy_train_arguments(legacy_train)
    legacy_train.add_argument("--seed", type=int, default=None)

    legacy_improve = sub.add_parser("improve", help="Compatibility: improve an existing legacy policy")
    _add_legacy_train_arguments(legacy_improve)
    legacy_improve.add_argument("--attempts", type=int, default=1)
    legacy_improve.add_argument("--trial", default=None)
    legacy_improve.add_argument("--max-dd", type=float, default=15.0)

    compare = sub.add_parser("compare-cores", help="Compatibility: compare legacy current/world cores")
    _add_legacy_train_arguments(compare)
    compare.add_argument("--output-dir", default="outputs/compare_cores")
    compare.add_argument("--runner-dir-current", default=None)
    compare.add_argument("--runner-dir-world", default=None)
    compare.add_argument("--current-seed", default=None)
    compare.add_argument("--world-seed", default=None)
    compare.add_argument("--exchange-replay-max", type=int, default=200)
    compare.add_argument("--skip-exchange", action="store_true")
    compare.add_argument("--max-dd", type=float, default=15.0)

    # Persistent event-driven compatibility command.  Deliberately has no
    # round/trial flags: learning is driven only by final execution lifecycle
    # receipts and is resumed from the serialized policy state.
    continuous = sub.add_parser("continuous-train", help="Persistent event-driven MT5 learning from final execution receipts")
    continuous.add_argument("--output", default="outputs/continuous_policy.pkl")
    continuous.add_argument("--seed-policy", default=None)
    continuous.add_argument("--runner-dir", default="mt5_v2_runner")
    continuous.add_argument("--timeout-sec", type=int, default=300)
    continuous.add_argument("--mt5-visual", action="store_true")
    continuous.add_argument("--keep-mt5-open", action="store_true")
    continuous.add_argument("--from-date", default=None)
    continuous.add_argument("--to-date", default=None)
    continuous.add_argument("--backtest-days", type=int, default=31)
    continuous.add_argument("--model", type=int, choices=[0, 1, 2, MT5_MODEL_REAL_TICKS], default=0)
    continuous.add_argument("--policy-core", choices=["current", "world"], default="current")
    continuous.add_argument("--m1-backend", choices=["direct_rnn", "m1_graph", "graph"], default="direct_rnn")
    continuous.add_argument("--session-id", default=None)
    continuous.add_argument("--continuous-root", default=None)
    continuous.add_argument("--post-exit-bars", type=int, default=None)
    continuous.add_argument("--max-lifecycle-wait-sequences", type=int, default=None)
    continuous.add_argument("--checkpoint-every-closed-positions", type=int, default=None)
    continuous.add_argument("--checkpoint-every-minutes", type=int, default=None)
    continuous.add_argument("--disable-local-inhibition", action="store_true")
    continuous.add_argument("--disable-persistent-specialists", action="store_true")
    _them_log_giai_thich(continuous)
    _them_mt5_options(continuous)

    obs = sub.add_parser("live-observatory", help="Mo UI doc JSONL")
    obs.add_argument("--events", default="outputs/quan_sat_chuyen_gia_live.jsonl")
    obs.add_argument("--host", default="127.0.0.1")
    obs.add_argument("--port", type=int, default=8765)
    obs.add_argument("--no-open-browser", action="store_true")
    return parser


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    _utf8()
    args = build_parser().parse_args(argv)
    if args.command == "live-observatory":
        serve_live_observatory(args.events, host=args.host, port=args.port, open_browser=not args.no_open_browser)
        return

    config = _apply_observability(load_config(args.config), args) if hasattr(args, "live_observatory") else load_config(args.config)
    config = _apply_cli_overrides(config, args)
    if args.command == "continuous-train":
        base = getattr(config, "continuous", ContinuousRuntimeConfig())
        overrides: dict[str, Any] = {"enabled": True}
        for name, value in (
            ("session_root", getattr(args, "continuous_root", None)),
            ("post_exit_bars", getattr(args, "post_exit_bars", None)),
            ("max_lifecycle_wait_sequences", getattr(args, "max_lifecycle_wait_sequences", None)),
            ("checkpoint_every_closed_positions", getattr(args, "checkpoint_every_closed_positions", None)),
            ("checkpoint_every_minutes", getattr(args, "checkpoint_every_minutes", None)),
        ):
            if value is not None:
                overrides[name] = value
        if bool(getattr(args, "disable_local_inhibition", False)):
            overrides["enable_local_inhibition"] = False
        if bool(getattr(args, "disable_persistent_specialists", False)):
            overrides["enable_persistent_specialists"] = False
        config = replace(config, continuous=replace(base, **overrides))
        result = continuous_train_mt5(
            config,
            output_path=str(args.output),
            seed_policy_path=args.seed_policy,
            runner_dir=args.runner_dir,
            timeout_sec=int(args.timeout_sec),
            visual=bool(args.mt5_visual),
            shutdown_terminal=not bool(args.keep_mt5_open),
            from_date=args.from_date,
            to_date=args.to_date,
            backtest_days=int(args.backtest_days),
            model=int(args.model),
            policy_core=str(args.policy_core),
            m1_backend=str(args.m1_backend),
            session_id=args.session_id,
        )
        _emit(result)
        return

    if args.command in {"train-mt5", "improve", "compare-cores"}:
        config = _apply_order_overrides(config, args)
        shutdown_terminal, close_scope = _close_scope(args)
        common = {
            "runner_dir": args.runner_dir,
            "timeout_sec": int(args.timeout_sec),
            "visual": bool(args.mt5_visual),
            "shutdown_terminal": bool(shutdown_terminal),
            "curriculum_learning": bool(args.curriculum_learning),
            "from_date": args.from_date,
            "to_date": args.to_date,
            "model": int(args.model),
            "policy_core": str(args.policy_core),
            "m1_backend": str(args.m1_backend),
            "mt5_close_scope": close_scope,
            "mutation_sigma": float(args.mutation_sigma),
            "max_modules": args.max_modules,
        }
        if args.command == "train-mt5":
            result = train_self_growing_mt5(
                config,
                int(args.rounds),
                str(args.output),
                trials=int(args.trials),
                seed_policy_path=args.seed_policy,
                seed=args.seed,
                **common,
            )
            _emit(result)
            return
        if args.command == "improve":
            result = improve_self_growing(
                config,
                int(args.rounds),
                str(args.seed_policy or "outputs/winner.pkl"),
                str(args.output),
                trial_path=args.trial,
                attempts=int(args.attempts),
                trials=int(args.trials),
                max_drawdown_pct=float(args.max_dd),
                **common,
            )
            _emit(result)
            return
        compare_kwargs = dict(common)
        compare_kwargs.pop("policy_core", None)
        compare_kwargs.pop("m1_backend", None)
        result = compare_policy_cores_mt5(
            config,
            int(args.rounds),
            output_dir=str(args.output_dir),
            trials=int(args.trials),
            runner_dir_current=args.runner_dir_current,
            runner_dir_world=args.runner_dir_world,
            current_seed=args.current_seed,
            world_seed=args.world_seed,
            exchange_replay_max=int(args.exchange_replay_max),
            skip_exchange=bool(args.skip_exchange),
            max_drawdown_pct=float(args.max_dd),
            **compare_kwargs,
        )
        _emit(result)
        return

    _require_context_expert_config(config)
    _xoa_log_quan_sat_neu_duoc_yeu_cau(config, args)
    if args.command == "khoi-tao-he-chuyen-gia":
        system = tao_he_chuyen_gia_mot_context_mot_policy(config, seed=args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        rollback = backup_context_expert_artifacts(output)
        with output.open("wb") as handle:
            pickle.dump(system, handle, pickle.HIGHEST_PROTOCOL)
        registry = system.luu_so_dang_ky(output.with_suffix(".registry.json"))
        _emit({"mode": "mot_context_mot_chuyen_gia", "system": str(output), "registry": str(registry), "rollback": rollback, "experts": len(system.experts)})
        return
    if args.command == "huan-luyen-he-chuyen-gia-mt5":
        # Frozen is an explicit experiment mode, not a hidden config rewrite.
        # Keep the policy/action surface identical while preventing every
        # lifecycle learner from mutating its owner. This makes a paired MT5
        # run capable of attributing a later PnL change to online learning
        # versus the sampled market path.
        if bool(args.frozen):
            config = replace(config, mt5_learning_enabled=False)
        _reject_unsafe_direct_online_resume(config, args)
        result = huan_luyen_he_chuyen_gia_mt5(
            config,
            output_path=args.output,
            seed_system_path=args.seed_system,
            runner_dir=args.runner_dir,
            timeout_sec=args.timeout_sec,
            max_wall_sec=args.max_wall_sec,
            visual=args.mt5_visual,
            shutdown_terminal=not args.keep_mt5_open,
            from_date=args.from_date,
            to_date=args.to_date,
            backtest_days=args.backtest_days,
            model=args.model,
            seed=args.seed,
            resume_existing=not args.fresh_system,
        )
        _emit(result)
        return
    if args.command == "huan-luyen-champion-he-chuyen-gia-mt5":
        result = run_context_expert_champion_training_mt5(
            config,
            champion_path=args.champion,
            output_path=args.output,
            audit_dir=args.audit_dir,
            train_from=args.train_from,
            train_to=args.train_to,
            validation_count=args.validation_count,
            validation_days=args.validation_days,
            candidate_seed_count=args.candidate_seed_count,
            base_seed=args.seed,
            runner_dir=args.runner_dir,
            timeout_sec=args.timeout_sec,
            max_wall_sec=args.max_wall_sec,
            visual=args.mt5_visual,
            shutdown_terminal=not args.keep_mt5_open,
            model=args.model,
            paired_min_expectancy=args.paired_min_expectancy,
            paired_max_profit_regression=args.paired_max_profit_regression,
            paired_max_expectancy_lcb_regression=args.paired_max_expectancy_lcb_regression,
            paired_max_drawdown_increase_pct=args.paired_max_drawdown_increase_pct,
        )
        _emit(result)
        return
    if args.command == "walk-forward-he-chuyen-gia-mt5":
        result = walk_forward_he_chuyen_gia_mt5(
            config,
            output_dir=args.output,
            train_window_days=args.train_window,
            test_window_days=args.test_window,
            step_days=args.step,
            log_path=args.log,
            seed_count=args.seed_count,
            seed_system_path=args.seed_system,
            runner_dir=args.runner_dir,
            timeout_sec=args.timeout_sec,
            visual=args.mt5_visual,
            shutdown_terminal=not args.keep_mt5_open,
            model=args.model,
            from_date=args.from_date,
            to_date=args.to_date,
            seed=args.seed,
            paired_baseline=bool(args.paired_baseline),
            rollback_on_reject=not bool(args.no_rollback),
            paired_min_expectancy=float(args.paired_min_expectancy),
            paired_max_profit_regression=float(args.paired_max_profit_regression),
            paired_max_expectancy_lcb_regression=float(args.paired_max_expectancy_lcb_regression),
            paired_max_drawdown_increase_pct=float(args.paired_max_drawdown_increase_pct),
        )
        _emit(result)
        return
    if args.command == "quan-tri-he-chuyen-gia":
        path = Path(args.system)
        with path.open("rb") as handle:
            system = load_pickle_with_removed_context_structural_migration(handle)
        ensure = getattr(system, "ensure_compatible", None)
        if callable(ensure):
            ensure()
        if args.action == "promote":
            result = system.xet_thang_cap_chuyen_gia(args.context, replay_pass=args.replay_pass, oos_pass=args.oos_pass, risk_pass=args.risk_pass, lifecycle_count=args.lifecycle_count, ghi_chu=args.note)
        elif args.action == "dormant":
            result = system.ngu_dong_chuyen_gia(args.context, ghi_chu=args.note)
        elif args.action == "archive":
            result = system.luu_tru_chuyen_gia(args.context, ghi_chu=args.note)
        else:
            if not args.target_context:
                raise ValueError("merge can --target-context")
            result = system.hop_nhat_context_chuyen_gia(args.context, args.target_context, oos_pass=args.oos_pass, non_interference_pass=args.non_interference_pass, ghi_chu=args.note)
        output = Path(args.output or args.system)
        output.parent.mkdir(parents=True, exist_ok=True)
        rollback = backup_context_expert_artifacts(output)
        with output.open("wb") as handle:
            pickle.dump(system, handle, pickle.HIGHEST_PROTOCOL)
        registry = system.luu_so_dang_ky(output.with_suffix(".registry.json"))
        response = {"system": str(output), "policy": str(output), "registry": str(registry), "result": result, "rollback": rollback}
        if args.audit_log:
            audit_path = Path(args.audit_log)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"action": args.action, "context": args.context, "result": result, "policy": str(output)}, ensure_ascii=False, default=str) + "\n")
        _emit(response)
        return


if __name__ == "__main__":
    main()
