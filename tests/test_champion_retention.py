from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from self_growing_trader.champion_retention import (
    build_consecutive_validation_windows,
    _rlhf_matrix_storage_preflight,
    run_context_expert_champion_training_mt5,
)
from self_growing_trader.learning_integrity import champion_candidate_matrix_gate, sha256_file
from self_growing_trader import cli
from self_growing_trader.mt5_runner import MT5_MODEL_REAL_TICKS
from self_growing_trader.chuyen_gia_policy import _tong_ket_walk_forward


def _metrics(*, lcb: float, profit: float = 11.0) -> dict[str, object]:
    return {
        "fitness_eligible": True,
        "fitness_reconciled": True,
        "total_trades": 50,
        "equity_drawdown_pct": 4.0,
        "fitness_policy_expectancy_lcb": lcb,
        "fitness_context_negative_critical": 0,
        "feature_audit_valid": True,
        "real_tick_evidence_verified": True,
        "net_profit": profit,
        "execution_summary": {
            "expectancy": max(lcb, 0.01),
            "expectancy_lcb": lcb,
            "lifecycle_count": 50,
        },
    }


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        learning=SimpleNamespace(
            promotion_min_oos_windows=2,
            promotion_min_seeds=2,
            promotion_require_fitness_eligible=True,
            promotion_require_reconciled=True,
            promotion_min_oos_trades=30,
            promotion_max_oos_drawdown_pct=10.0,
            promotion_min_policy_expectancy_lcb=0.0,
            promotion_max_negative_context_scopes=0,
            promotion_require_feature_parity=True,
            promotion_require_real_tick_evidence=True,
        )
    )


def test_build_consecutive_validation_windows_never_overlaps_train() -> None:
    windows = build_consecutive_validation_windows("2025.01.10", count=3, window_days=2)
    assert [row["id"] for row in windows] == [
        "2025.01.10:2025.01.12",
        "2025.01.12:2025.01.14",
        "2025.01.14:2025.01.16",
    ]


def test_champion_cli_defaults_to_real_ticks_and_dispatches(monkeypatch) -> None:
    captured = {}

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return {"promoted": False}

    monkeypatch.setattr(cli, "run_context_expert_champion_training_mt5", fake_run)
    cli.main([
        "--config", "configs/he_chuyen_gia_mot_context_mot_policy.yaml",
        "huan-luyen-champion-he-chuyen-gia-mt5",
        "--champion", "champion.pkl",
        "--train-to", "2025.01.10",
    ])
    assert captured["model"] == MT5_MODEL_REAL_TICKS
    assert captured["champion_path"] == "champion.pkl"


def test_champion_matrix_rejects_mixed_one_window_artifacts() -> None:
    rows = [
        {
            "candidate_seed": seed,
            "candidate_sha256": f"hash-{seed}",
            "validation_window": f"window-{seed}",
            "eligible": True,
            "policy_expectancy_lcb": 0.1,
        }
        for seed in range(3)
    ]
    gate = champion_candidate_matrix_gate(rows, min_windows=3, min_candidate_seeds=3)
    assert gate["eligible"] is False
    assert gate["stable_candidate_seeds"] == 0
    assert "stable_candidate_seeds_insufficient:0<3" in gate["reasons"]


def test_champion_matrix_selects_best_worst_window_only_after_complete_pass() -> None:
    rows = []
    for seed, scores in ((10, (0.10, 0.11)), (11, (0.20, 0.05))):
        for index, score in enumerate(scores):
            rows.append({
                "candidate_seed": seed,
                "candidate_sha256": f"hash-{seed}",
                "validation_window": f"window-{index}",
                "eligible": True,
                "policy_expectancy_lcb": score,
            })
    gate = champion_candidate_matrix_gate(rows, min_windows=2, min_candidate_seeds=2)
    assert gate["eligible"] is True
    assert gate["selected_seed"] == "10"
    assert gate["selected_artifact_sha256"] == "hash-10"

    rows[-1]["eligible"] = False
    rejected = champion_candidate_matrix_gate(rows, min_windows=2, min_candidate_seeds=2)
    assert rejected["eligible"] is False
    assert rejected["stable_candidate_seeds"] == 1


def test_walk_forward_cannot_promote_by_mixing_different_artifacts(tmp_path) -> None:
    rows = []
    for window in range(3):
        for seed in range(3):
            artifact = tmp_path / f"candidate-{window}-{seed}.pkl"
            artifact.write_bytes(f"{window}-{seed}".encode())
            rows.append({
                "window": window,
                "validation_window": f"window-{window}",
                "seed": seed,
                "val_fitness": 1.0 + window + seed,
                "output": str(artifact),
                "candidate_sha256": sha256_file(artifact),
                "oos_valid": True,
            })
    summary = _tong_ket_walk_forward(
        results=rows,
        out_dir=tmp_path,
        learning_cfg=SimpleNamespace(promotion_min_oos_windows=3, promotion_min_seeds=3),
    )
    assert summary["promotion_eligible"] is False
    assert summary["final_output"] == ""
    assert summary["promotion_evidence"]["stable_candidate_seeds"] == 0


def test_debug_matrix_cannot_lower_production_floor_or_overwrite_champion(tmp_path, monkeypatch) -> None:
    champion = tmp_path / "champion.pkl"
    champion.write_bytes(b"champion-immutable")
    original_hash = sha256_file(champion)
    output = tmp_path / "promoted.pkl"
    audit = tmp_path / "audit"

    def fake_load(path, config, seed=None):
        return SimpleNamespace(payload=Path(path).read_bytes())

    def fake_train(config, **kwargs):
        Path(kwargs["output_path"]).write_bytes(f"shadow-{kwargs['seed']}".encode())
        return {"interrupted": False, "policy": str(kwargs["output_path"])}

    def fake_evaluate(policy, config, **kwargs):
        is_champion = policy.payload == b"champion-immutable"
        lcb = 0.01 if is_champion else 0.10
        profit = 10.0 if is_champion else 11.0
        return {"best": {"fitness": lcb, "metrics": _metrics(lcb=lcb, profit=profit)}}

    monkeypatch.setattr("self_growing_trader.champion_retention.tai_he_chuyen_gia_mot_context_mot_policy", fake_load)
    result = run_context_expert_champion_training_mt5(
        _config(), champion_path=champion, output_path=output, audit_dir=audit,
        train_from="2025.01.01", train_to="2025.01.10",
        validation_count=1, validation_days=1, candidate_seed_count=1,
        trainer=fake_train, evaluator=fake_evaluate,
    )
    assert result["promoted"] is False
    assert not output.exists()
    assert sha256_file(champion) == original_hash
    assert result["champion_immutable_verified"] is True


def test_rlhf_matrix_rejects_source_hash_that_is_not_the_champion(tmp_path) -> None:
    champion = tmp_path / "champion.pkl"
    champion.write_bytes(b"champion-immutable")
    config = _config()
    config.learning.five_rlhf_policy_update_enabled = True
    config.learning.five_rlhf_source_checkpoint_sha256 = "wrong-source"

    with pytest.raises(ValueError, match="must match the immutable champion"):
        run_context_expert_champion_training_mt5(
            config,
            champion_path=champion,
            output_path=tmp_path / "candidate.pkl",
            audit_dir=tmp_path / "audit",
            train_from="2025.01.01",
            train_to="2025.01.10",
            validation_count=1,
            validation_days=1,
            candidate_seed_count=1,
        )


def test_rlhf_matrix_storage_preflight_fails_closed_below_configured_floor(monkeypatch) -> None:
    config = SimpleNamespace(
        learning=SimpleNamespace(
            five_rlhf_policy_update_enabled=True,
            five_rlhf_min_free_disk_bytes=2_000,
        )
    )
    monkeypatch.setattr(
        "self_growing_trader.champion_retention.shutil.disk_usage",
        lambda _: SimpleNamespace(free=1_999),
    )
    result = _rlhf_matrix_storage_preflight(config, storage_path="outputs/rlhf")
    assert result["enabled"] is True
    assert result["passed"] is False
    assert result["reason"] == "insufficient_free_disk"


def test_transaction_promotes_maximin_shadow_without_overwriting_champion(tmp_path, monkeypatch) -> None:
    champion = tmp_path / "champion.pkl"
    champion.write_bytes(b"champion-immutable")
    original_hash = sha256_file(champion)
    output = tmp_path / "promoted.pkl"
    audit = tmp_path / "audit"

    def fake_load(path, config, seed=None):
        return SimpleNamespace(payload=Path(path).read_bytes())

    def fake_train(config, **kwargs):
        seed = int(kwargs["seed"])
        path = Path(kwargs["output_path"])
        path.write_bytes(f"shadow-{seed}".encode())
        path.with_suffix(".registry.json").write_text(f'{{"seed": {seed}}}', encoding="utf-8")
        return {"interrupted": False, "policy": str(path)}

    def fake_evaluate(policy, config, **kwargs):
        if policy.payload == b"champion-immutable":
            lcb, profit = 0.01, 10.0
        elif policy.payload == b"shadow-10":
            lcb, profit = (0.10 if kwargs["from_date"] == "2025.01.10" else 0.11), 12.0
        else:
            lcb, profit = (0.20 if kwargs["from_date"] == "2025.01.10" else 0.05), 12.0
        return {"best": {"fitness": lcb, "metrics": _metrics(lcb=lcb, profit=profit)}}

    monkeypatch.setattr("self_growing_trader.champion_retention.tai_he_chuyen_gia_mot_context_mot_policy", fake_load)
    result = run_context_expert_champion_training_mt5(
        _config(), champion_path=champion, output_path=output, audit_dir=audit,
        train_from="2025.01.01", train_to="2025.01.10",
        validation_count=2, validation_days=1, candidate_seed_count=2, base_seed=10,
        trainer=fake_train, evaluator=fake_evaluate,
    )
    assert result["promoted"] is True
    assert result["matrix_gate"]["selected_seed"] == "10"
    assert output.read_bytes() == b"shadow-10"
    assert output.with_suffix(".registry.json").is_file()
    assert sha256_file(champion) == original_hash
