from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from scripts import run_frozen_temporal_generalization as runner


def test_frozen_oos_defaults_to_real_tick_model() -> None:
    assert runner.DEFAULT_MT5_TESTER_MODEL == 4
    assert runner.DEFAULT_MT5_TESTER_MODEL in runner.SUPPORTED_MT5_TESTER_MODELS


def test_profile_overrides_match_candidate_runtime_contract() -> None:
    config = runner.load_config(
        "configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    )
    merged, payload = runner.apply_experiment_profile(
        config,
        Path("configs/experiments/reasoning_rl_v1_active_candidate.yaml"),
    )

    assert payload["experiment"]["id"] == "reasoning_rl_v1_active_candidate"
    assert merged.mt5_learning_enabled is True
    assert merged.learning.five_reasoning_action_scale == 0.25
    assert merged.learning.five_live_action_veto_hold_owner_on_negative is False
    assert merged.learning.five_live_owner_exploration_enabled is True
    assert merged.learning.five_live_owner_exploration_min_mix == 0.08


def test_evaluate_window_keeps_runner_and_artifacts_isolated(monkeypatch, tmp_path: Path) -> None:
    source_runner = tmp_path / "source"
    source_runner.mkdir()
    candidate = tmp_path / "candidate.pkl"
    candidate.write_bytes(b"read-only-candidate")
    clone_root = tmp_path / "clones"
    artifacts_root = tmp_path / "artifacts"
    config = SimpleNamespace(learning=SimpleNamespace(seed=17))

    def fake_clone_runner(_source: Path, target: Path) -> None:
        target.mkdir(parents=True)

    def fake_load(_candidate: Path, _config: object, *, seed: int) -> dict[str, int]:
        return {"seed": seed}

    def fake_evaluate(
        _policy: object,
        _config: object,
        *,
        runner_dir: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        ledger = runner_dir / "execution_ledger.csv"
        ledger.write_text("window-local-ledger", encoding="utf-8")
        return {
            "best": {
                "fitness": 1.0,
                "metrics": {"execution_ledger_path": str(ledger)},
            },
            "frozen": True,
            "evaluation_only": True,
            "source_policy_state_hash_before": "before",
            "source_policy_state_hash_after": "after",
        }

    monkeypatch.setattr(runner, "clone_runner", fake_clone_runner)
    monkeypatch.setattr(runner, "tai_he_chuyen_gia_mot_context_mot_policy", fake_load)
    monkeypatch.setattr(runner, "evaluate_policy_mt5_oos", fake_evaluate)

    windows = ((1, "2025.01.01", "2025.01.02"), (2, "2025.01.02", "2025.01.03"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                runner.evaluate_window,
                index=index,
                from_date=from_date,
                to_date=to_date,
                candidate=candidate,
                config=config,
                source_runner=source_runner,
                clone_root=clone_root,
                artifacts_root=artifacts_root,
                timeout_sec=1,
                model=1,
            )
            for index, from_date, to_date in windows
        ]
        rows = [future.result() for future in futures]

    assert [row["window"] for row in rows] == [1, 2]
    assert all(row["completed"] for row in rows)
    assert rows[0]["runner_clone"] != rows[1]["runner_clone"]
    assert Path(rows[0]["runner_clone"]).is_dir()
    assert Path(rows[1]["runner_clone"]).is_dir()
    assert (artifacts_root / "window_01" / "execution_ledger.csv").is_file()
    assert (artifacts_root / "window_02" / "execution_ledger.csv").is_file()
    assert candidate.read_bytes() == b"read-only-candidate"
