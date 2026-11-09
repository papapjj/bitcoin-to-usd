from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pytest

from self_growing_trader import cli
from self_growing_trader.he_chuyen_gia_ngu_canh import (
    CauHinhHeChuyenGia,
    HeChuyenGiaMotContextMotPolicy,
)


def test_cli_configures_stdout_utf8(monkeypatch) -> None:
    calls = {}

    class FakeStdout:
        def reconfigure(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(cli.sys, "stdout", FakeStdout())

    cli._ensure_utf8_stdout()

    assert calls["encoding"] == "utf-8"


def test_cli_observability_overrides_use_real_config_fields() -> None:
    config = cli.load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    args = argparse.Namespace(
        live_observatory=True,
        live_events="outputs/test_live.jsonl",
        live_sample_every=7,
        live_max_mb=3,
        live_decision_profiles=None,
        live_structural_events=None,
        live_network_development=None,
        live_max_responsible_neurons=11,
        live_max_responsible_edges=13,
    )

    updated = cli._apply_observability(config, args)

    assert updated.observability.enabled is True
    assert updated.observability.event_path == "outputs/test_live.jsonl"
    assert updated.observability.decision_sample_every == 7
    assert updated.observability.max_event_bytes == 3 * 1024 * 1024
    assert updated.observability.max_responsible_neurons == 11
    assert updated.observability.max_responsible_edges == 13


def test_active_candidate_disables_all_learned_entry_masks_and_keeps_risk_envelope() -> None:
    config = cli.load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")

    assert config.learning.action_recovery_enabled is False
    assert config.learning.action_recovery_negative_veto_enabled is False
    assert config.learning.context_direction_entry_gate_enabled is False
    assert config.learning.context_direction_side_entry_gate_enabled is False
    assert config.learning.context_expectancy_entry_gate_enabled is False
    assert config.learning.context_expectancy_entry_gate_allowed_contexts == []
    assert config.learning.context_expectancy_entry_gate_scope_fallback_enabled is False
    assert config.learning.entry_guard_single_position is False
    assert config.learning.entry_loss_guard_enabled is False
    assert config.learning.runtime_fitness_guard_enabled is False
    # Risk envelope is an EA/broker validity contract, not a learned mask.
    assert config.risk.enforce_entry_limits is True
    assert config.risk.max_risk_pct == 0.005
    assert config.risk.max_open_risk_pct == 0.015


def test_continuation_profile_uses_one_shared_five_owner() -> None:
    config = cli.load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")

    assert config.he_chuyen_gia.che_do_quyet_dinh == "nam_tac_nhan_dung_chung"
    assert config.he_chuyen_gia.cho_phep_policy_chung_ra_lenh is True
    assert config.he_chuyen_gia.volatile_subexperts_bat is False


class _CliExpertPolicy:
    def act(self, observation, goal=None):
        raise AssertionError("admin command must not call policy.act")


def test_context_expert_admin_cli_writes_artifact_registry_and_audit(tmp_path, capsys) -> None:
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=_CliExpertPolicy,
    )
    policy = tmp_path / "he.pkl"
    output = tmp_path / "he_admin.pkl"
    audit_log = tmp_path / "admin.jsonl"
    with policy.open("wb") as handle:
        pickle.dump(system, handle, protocol=pickle.HIGHEST_PROTOCOL)
    output.write_bytes(b"old policy artifact")
    output.with_suffix(".registry.json").write_text('{"old":true}', encoding="utf-8")
    context = next(iter(system.experts.values())).ma_context

    cli.main([
        "--config",
        "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
        "quan-tri-he-chuyen-gia",
        "--policy",
        str(policy),
        "--output",
        str(output),
        "--context",
        context,
        "--action",
        "archive",
        "--note",
        "retire-after-oos",
        "--audit-log",
        str(audit_log),
    ])

    assert output.exists()
    assert output.with_suffix(".registry.json").exists()
    row = json.loads(audit_log.read_text(encoding="utf-8").splitlines()[-1])
    assert row["action"] == "archive"
    assert row["result"]["to"] == "LUU_TRU"
    printed = json.loads(capsys.readouterr().out)
    assert printed["policy"] == str(output)
    rollback = printed["rollback"]
    assert all(Path(path).exists() for path in rollback.values())
    assert Path(rollback["policy"]).read_bytes() == b"old policy artifact"
    assert json.loads(Path(rollback["registry"]).read_text(encoding="utf-8")) == {"old": True}
    assert json.loads(output.with_suffix(".registry.json").read_text(encoding="utf-8"))["che_do"] == "mot_context_mot_chuyen_gia"


def test_train_mt5_cli_applies_daily_trade_overrides(monkeypatch, capsys) -> None:
    captured = {}

    def fake_train_self_growing_mt5(config, rounds, output_path, **kwargs):
        captured["min_daily_trades"] = config.orders.min_daily_trades
        captured["max_daily_trades"] = config.orders.max_daily_trades
        captured["rounds"] = rounds
        captured["output_path"] = output_path
        captured.update(kwargs)
        return {"winner": output_path}

    monkeypatch.setattr(cli, "train_self_growing_mt5", fake_train_self_growing_mt5)

    cli.main(
        [
            "train-mt5",
            "--rounds",
            "2",
            "--output",
            "outputs/test_cli.pkl",
            "--min-daily-trades",
            "12",
            "--max-daily-trades",
            "60",
            "--max-modules",
            "3",
            "--m1-backend",
            "graph",
        ]
    )

    assert captured["min_daily_trades"] == 12
    assert captured["max_daily_trades"] == 60
    assert captured["rounds"] == 2
    assert captured["output_path"] == "outputs/test_cli.pkl"
    assert captured["seed_policy_path"] is None
    assert captured["max_modules"] == 3
    assert captured["m1_backend"] == "graph"
    assert captured["mt5_close_scope"] == "trial"
    assert "outputs/test_cli.pkl" in capsys.readouterr().out


def test_context_expert_mt5_cli_backtest_days_is_not_overridden_by_default_to_date(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_huan_luyen_he_chuyen_gia_mt5(config, **kwargs):
        captured.update(kwargs)
        return {
            "mode": "mot_context_mot_chuyen_gia_mt5",
            "date_range": {"from": kwargs["from_date"], "to": kwargs["to_date"]},
        }

    monkeypatch.setattr(cli, "huan_luyen_he_chuyen_gia_mt5", fake_huan_luyen_he_chuyen_gia_mt5)

    cli.main(
        [
            "--config",
            "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
            "huan-luyen-he-chuyen-gia-mt5",
            "--output",
            str(tmp_path / "new_research_candidate.pkl"),
            "--backtest-days",
            "1",
            "--max-wall-sec",
            "45",
        ]
    )

    assert captured["from_date"] == "2025.01.01"
    assert captured["to_date"] is None
    assert captured["backtest_days"] == 1
    assert captured["max_wall_sec"] == 45
    assert captured["resume_existing"] is True


def test_context_expert_cli_rejects_online_resume_that_bypasses_champion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.pkl"
    output.write_bytes(b"checkpoint")
    called = False

    def fake_huan_luyen_he_chuyen_gia_mt5(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "huan_luyen_he_chuyen_gia_mt5", fake_huan_luyen_he_chuyen_gia_mt5)

    with pytest.raises(ValueError, match="Tu choi online direct-resume"):
        cli.main([
            "--config",
            "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
            "huan-luyen-he-chuyen-gia-mt5",
            "--output",
            str(output),
        ])

    assert called is False


def test_context_expert_cli_rejects_online_seed_outside_champion_transaction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seed = tmp_path / "champion.pkl"
    seed.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        cli,
        "huan_luyen_he_chuyen_gia_mt5",
        lambda *args, **kwargs: pytest.fail("direct trainer must not run"),
    )

    with pytest.raises(ValueError, match="huan-luyen-champion-he-chuyen-gia-mt5"):
        cli.main([
            "--config",
            "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
            "huan-luyen-he-chuyen-gia-mt5",
            "--seed-system",
            str(seed),
            "--output",
            str(tmp_path / "candidate.pkl"),
        ])


def test_context_expert_cli_can_request_fresh_system(monkeypatch) -> None:
    captured = {}

    def fake_huan_luyen_he_chuyen_gia_mt5(config, **kwargs):
        captured.update(kwargs)
        return {"mode": "mot_context_mot_chuyen_gia_mt5"}

    monkeypatch.setattr(cli, "huan_luyen_he_chuyen_gia_mt5", fake_huan_luyen_he_chuyen_gia_mt5)
    cli.main([
        "--config",
        "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
        "huan-luyen-he-chuyen-gia-mt5",
        "--fresh-system",
    ])

    assert captured["resume_existing"] is False


def test_context_expert_cli_frozen_mode_disables_mt5_learning(monkeypatch) -> None:
    captured = {}

    def fake_huan_luyen_he_chuyen_gia_mt5(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {"mode": "mot_context_mot_chuyen_gia_mt5"}

    monkeypatch.setattr(cli, "huan_luyen_he_chuyen_gia_mt5", fake_huan_luyen_he_chuyen_gia_mt5)
    cli.main([
        "--config",
        "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
        "huan-luyen-he-chuyen-gia-mt5",
        "--frozen",
    ])

    assert captured["config"].mt5_learning_enabled is False


def test_cli_rejects_daily_trade_max_below_min(monkeypatch) -> None:
    monkeypatch.setattr(cli, "train_self_growing_mt5", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="max-daily-trades"):
        cli.main(["train-mt5", "--min-daily-trades", "12", "--max-daily-trades", "11"])


def test_train_mt5_cli_can_request_visible_terminal(monkeypatch) -> None:
    captured = {}

    def fake_train_self_growing_mt5(config, rounds, output_path, **kwargs):
        captured.update(kwargs)
        return {"winner": output_path}

    monkeypatch.setattr(cli, "train_self_growing_mt5", fake_train_self_growing_mt5)

    cli.main(["train-mt5", "--mt5-visual", "--keep-mt5-open", "--timeout-sec", "600"])

    assert captured["visual"] is True
    assert captured["shutdown_terminal"] is False
    assert captured["timeout_sec"] == 600
    assert captured["mt5_close_scope"] == "never"


def test_train_mt5_cli_can_close_after_round(monkeypatch) -> None:
    captured = {}

    def fake_train_self_growing_mt5(config, rounds, output_path, **kwargs):
        captured.update(kwargs)
        return {"winner": output_path}

    monkeypatch.setattr(cli, "train_self_growing_mt5", fake_train_self_growing_mt5)

    cli.main(["train-mt5", "--mt5-close-scope", "round"])

    assert captured["shutdown_terminal"] is True
    assert captured["mt5_close_scope"] == "round"


def test_train_mt5_cli_passes_world_policy_core(monkeypatch) -> None:
    captured = {}

    def fake_train_self_growing_mt5(config, rounds, output_path, **kwargs):
        captured.update(kwargs)
        return {"winner": output_path}

    monkeypatch.setattr(cli, "train_self_growing_mt5", fake_train_self_growing_mt5)

    cli.main(["train-mt5", "--policy-core", "world"])

    assert captured["policy_core"] == "world"


def test_improve_cli_uses_mt5_runtime_args(monkeypatch) -> None:
    captured = {}

    def fake_improve_self_growing(config, rounds, seed_policy_path, output_path, trial_path=None, **kwargs):
        captured.update(
            {
                "rounds": rounds,
                "seed_policy_path": seed_policy_path,
                "output_path": output_path,
                "trial_path": trial_path,
                **kwargs,
            }
        )
        return {"promoted": False}

    monkeypatch.setattr(cli, "improve_self_growing", fake_improve_self_growing)

    cli.main(
        [
            "improve",
            "--rounds",
            "2",
            "--attempts",
            "3",
            "--trials",
            "4",
            "--runner-dir",
            "runner",
            "--timeout-sec",
            "7",
            "--mt5-visual",
            "--keep-mt5-open",
            "--curriculum-learning",
            "--max-dd",
            "12",
            "--policy-core",
            "world",
            "--m1-backend",
            "graph",
        ]
    )

    assert captured["rounds"] == 2
    assert captured["seed_policy_path"] == "outputs/winner.pkl"
    assert captured["output_path"] == "outputs/winner.pkl"
    assert captured["attempts"] == 3
    assert captured["trials"] == 4
    assert captured["runner_dir"] == "runner"
    assert captured["timeout_sec"] == 7
    assert captured["visual"] is True
    assert captured["shutdown_terminal"] is False
    assert captured["curriculum_learning"] is True
    assert captured["max_drawdown_pct"] == 12.0
    assert captured["policy_core"] == "world"
    assert captured["m1_backend"] == "graph"


def test_compare_cores_cli_passes_parallel_runner_and_exchange_args(monkeypatch) -> None:
    captured = {}

    def fake_compare_policy_cores_mt5(config, rounds, **kwargs):
        captured["rounds"] = rounds
        captured.update(kwargs)
        return {"winner": {"core": "world"}}

    monkeypatch.setattr(cli, "compare_policy_cores_mt5", fake_compare_policy_cores_mt5)

    cli.main(
        [
            "compare-cores",
            "--rounds",
            "2",
            "--trials",
            "3",
            "--runner-dir",
            "base_runner",
            "--runner-dir-current",
            "runner_current",
            "--runner-dir-world",
            "runner_world",
            "--current-seed",
            "outputs/current_seed.pkl",
            "--world-seed",
            "outputs/world_seed.pkl",
            "--output-dir",
            "outputs/compare_test",
            "--exchange-replay-max",
            "50",
            "--skip-exchange",
            "--timeout-sec",
            "9",
            "--max-dd",
            "11",
        ]
    )

    assert captured["rounds"] == 2
    assert captured["trials"] == 3
    assert captured["runner_dir"] == "base_runner"
    assert captured["runner_dir_current"] == "runner_current"
    assert captured["runner_dir_world"] == "runner_world"
    assert captured["current_seed"] == "outputs/current_seed.pkl"
    assert captured["world_seed"] == "outputs/world_seed.pkl"
    assert captured["output_dir"] == "outputs/compare_test"
    assert captured["exchange_replay_max"] == 50
    assert captured["skip_exchange"] is True
    assert captured["timeout_sec"] == 9
    assert captured["max_drawdown_pct"] == 11.0


