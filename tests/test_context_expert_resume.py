from __future__ import annotations

from dataclasses import replace
import copy
import pickle
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    _compact_context_expert_artifact_state,
    _ensure_rlhf_optimizer_state_available_for_update,
    _paired_seed_validation_gate,
    _reset_context_expert_runtime_for_new_mt5_session,
    archive_context_expert_runtime_artifacts,
    resolve_context_expert_resume_path,
    tai_he_chuyen_gia_mot_context_mot_policy,
    tao_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config
from self_growing_trader.he_chuyen_gia_ngu_canh import (
    BoNhanDienNguCanh,
    CauHinhHeChuyenGia,
    HeChuyenGiaMotContextMotPolicy,
)


class _ResumePolicy:
    def ensure_compatible(self) -> None:
        return None


class _RecoverySystem:
    def __init__(self) -> None:
        self.learned_marker = 17

    def luu_so_dang_ky(self, path: Path) -> str:
        Path(path).write_text("{}", encoding="utf-8")
        return str(path)


@pytest.mark.parametrize(
    ("requested", "save_and_stop", "persist_checkpoint", "run", "reason"),
    [
        (False, False, True, False, "not_requested"),
        (True, True, True, False, "save_and_stop_requested"),
        (True, False, False, False, "metrics_only_no_checkpoint"),
        (True, False, True, True, "enabled"),
    ],
)
def test_paired_seed_validation_gate_respects_output_contract(
    requested, save_and_stop, persist_checkpoint, run, reason
) -> None:
    result = _paired_seed_validation_gate(
        requested=requested,
        save_and_stop_requested=save_and_stop,
        persist_checkpoint=persist_checkpoint,
    )
    assert result == {"run": run, "reason": reason}


def test_rlhf_oos_compaction_strips_only_explicitly_requested_five_adam_state() -> None:
    class FakeOptimizer:
        def __init__(self) -> None:
            self.state = {"parameter": {"m": object(), "v": object()}}

    class FakeTrainer:
        def __init__(self) -> None:
            self.optimizer = FakeOptimizer()

    class FakePolicy:
        def __init__(self) -> None:
            self.five_ppo_trainer = FakeTrainer()
            self._causal_trace_archive = {}
            self._execution_trace_cache = {}
            self._entry_preference_trace_cache = {}
            self._five_pattern_anchor_cache = {}
            self._five_pattern_settled_sequences = set()

    class FakeExpert:
        def __init__(self) -> None:
            self.policy = FakePolicy()

    class FakeSystem:
        def __init__(self) -> None:
            self.experts = {"five": FakeExpert()}

    retained = FakeSystem()
    _compact_context_expert_artifact_state(
        retained,
        learning_settings=SimpleNamespace(
            five_rlhf_strip_optimizer_state_for_oos=False,
        ),
    )
    assert retained.experts["five"].policy.five_ppo_trainer.optimizer.state
    assert not getattr(retained, "_five_rlhf_optimizer_state_stripped", False)

    compacted = FakeSystem()
    _compact_context_expert_artifact_state(
        compacted,
        learning_settings=SimpleNamespace(
            five_rlhf_strip_optimizer_state_for_oos=True,
        ),
    )
    assert compacted.experts["five"].policy.five_ppo_trainer.optimizer.state == {}
    assert compacted._five_rlhf_optimizer_state_stripped is True
    assert compacted._five_rlhf_optimizer_state_stripped_count == 1


def test_rlhf_update_rejects_a_stripped_oos_artifact() -> None:
    class StrippedArtifact:
        _five_rlhf_optimizer_state_stripped = True

    with pytest.raises(ValueError, match="optimizer state stripped"):
        _ensure_rlhf_optimizer_state_available_for_update(
            StrippedArtifact(),
            SimpleNamespace(five_rlhf_policy_update_enabled=True),
        )

    # Frozen OOS inspection is allowed to load the same artifact because it
    # does not request another policy update.
    _ensure_rlhf_optimizer_state_available_for_update(
        StrippedArtifact(),
        SimpleNamespace(five_rlhf_policy_update_enabled=False),
    )


def test_rlhf_update_can_reinitialize_stripped_state_only_for_research() -> None:
    class FakeOptimizer:
        def __init__(self) -> None:
            self.state = {"old_parameter": {"step": 8}}

    class FakeTrainer:
        def __init__(self) -> None:
            self.optimizer = FakeOptimizer()

    class LocalPolicy:
        def __init__(self) -> None:
            self.five_ppo_trainer = FakeTrainer()

    class Expert:
        def __init__(self) -> None:
            self.policy = LocalPolicy()

    class StrippedArtifact:
        _five_rlhf_optimizer_state_stripped = True

        def __init__(self) -> None:
            self.experts = {"five": Expert()}

    artifact = StrippedArtifact()
    _ensure_rlhf_optimizer_state_available_for_update(
        artifact,
        SimpleNamespace(
            five_rlhf_policy_update_enabled=True,
            five_rlhf_allow_optimizer_state_reinitialization_for_research=True,
        ),
    )
    assert artifact.experts["five"].policy.five_ppo_trainer.optimizer.state == {}
    assert artifact._five_rlhf_optimizer_state_reinitialized is True
    assert artifact._five_rlhf_optimizer_state_reinitialization_count == 1


def test_existing_output_is_the_default_resume_seed(tmp_path: Path) -> None:
    output = tmp_path / "kho_chuyen_gia.pkl"
    output.write_bytes(b"checkpoint")

    assert resolve_context_expert_resume_path(output) == output
    assert resolve_context_expert_resume_path(output, resume_existing=False) is None


def test_explicit_missing_seed_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="khong tim thay seed system"):
        resolve_context_expert_resume_path(
            tmp_path / "output.pkl",
            seed_system_path=tmp_path / "missing.pkl",
        )


def test_runtime_archive_preserves_mutable_mt5_evidence_by_candidate(tmp_path: Path) -> None:
    sources = {
        "bridge": tmp_path / "self_growing_bridge.csv",
        "request": tmp_path / "self_growing_request.csv",
        "action": tmp_path / "self_growing_action.csv",
        "entry_preference": tmp_path / "self_growing_entry_preference.csv",
        "execution_ledger": tmp_path / "self_growing_execution.csv",
        "feature_audit": tmp_path / "self_growing_feature_audit.csv",
        "report": tmp_path / "report.htm",
    }
    for name, path in sources.items():
        path.write_text(f"{name}-evidence", encoding="utf-8")

    class Evaluator:
        config = types.SimpleNamespace(
            bridge_file="self_growing_bridge.csv",
            request_file="self_growing_request.csv",
            action_file="self_growing_action.csv",
            entry_preference_file="self_growing_entry_preference.csv",
        )

        def _latest_runtime_file(self, filename: str) -> Path:
            return tmp_path / filename

    metrics = types.SimpleNamespace(
        report_path=sources["report"],
        execution_ledger_path=sources["execution_ledger"],
        feature_audit_path=sources["feature_audit"],
    )
    archived = archive_context_expert_runtime_artifacts(
        tmp_path / "candidate.pkl", evaluator=Evaluator(), metrics=metrics
    )

    assert set(archived) == set(sources)
    for label, destination in archived.items():
        assert Path(destination).read_text(encoding="utf-8") == f"{label}-evidence"


def test_continuation_resets_only_runtime_state_before_new_mt5_session() -> None:
    class ResumedSystem:
        def __init__(self) -> None:
            self.weights = [1.0, 2.0]
            self.lifecycle_evidence = ["kept"]
            self.runtime_memory = [9.0]
            self.reset_calls = 0

        def reset_runtime_state(self) -> None:
            self.runtime_memory = [0.0]
            self.reset_calls += 1

    system = ResumedSystem()

    assert _reset_context_expert_runtime_for_new_mt5_session(system) is True
    assert system.reset_calls == 1
    assert system.runtime_memory == [0.0]
    assert system.weights == [1.0, 2.0]
    assert system.lifecycle_evidence == ["kept"]


def test_continuation_runtime_reset_fails_closed_when_contract_is_missing() -> None:
    assert _reset_context_expert_runtime_for_new_mt5_session(object()) is False


def test_mt5_runtime_error_luu_checkpoint_incomplete_roi_van_nem_loi(
    tmp_path: Path, monkeypatch
) -> None:
    import json
    import self_growing_trader.chuyen_gia_policy as policy_module

    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    config = replace(
        config,
        learning=replace(
            config.learning,
            five_codex_teacher_auto_connect=False,
        ),
    )
    seed = tmp_path / "seed.pkl"
    seed.write_bytes(b"seed-placeholder")
    output = tmp_path / "candidate.pkl"
    system = _RecoverySystem()
    evaluator_state = {"shutdown_terminal": None, "closed": 0}

    class FailingEvaluator:
        def __init__(self, config):
            self.config = config
            evaluator_state["shutdown_terminal"] = config.shutdown_terminal

        def evaluate_policy(self, trial_id, policy, round):
            policy.learned_marker = 99
            raise RuntimeError("MT5 exited before writing report")

        def close_runner_processes(self):
            evaluator_state["closed"] += 1

    monkeypatch.setattr(policy_module, "MT5StrategyTesterEvaluator", FailingEvaluator)
    monkeypatch.setattr(
        policy_module, "tai_he_chuyen_gia_mot_context_mot_policy",
        lambda *args, **kwargs: system,
    )
    monkeypatch.setattr(policy_module, "_require_context_expert_architecture_match", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_cau_hinh_policy_chuyen_gia_cho_mt5", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_reset_context_expert_runtime_for_new_mt5_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(policy_module, "_compact_context_expert_artifact_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "archive_context_expert_runtime_artifacts", lambda *args, **kwargs: {})

    with pytest.raises(policy_module.MT5TrainingCheckpointError) as caught:
        policy_module.huan_luyen_he_chuyen_gia_mt5(
            config,
            output_path=output,
            seed_system_path=seed,
            runner_dir=tmp_path / "runner",
            from_date="2025.01.01",
            to_date="2025.01.02",
            resume_existing=False,
        )

    checkpoint = tmp_path / "candidate.incomplete.pkl"
    metadata = tmp_path / "candidate.incomplete.json"
    assert Path(caught.value.checkpoint_path) == checkpoint
    assert Path(caught.value.metadata_path) == metadata
    assert caught.value.original_error_type == "RuntimeError"
    assert not output.exists()
    with checkpoint.open("rb") as handle:
        recovered = pickle.load(handle)
    assert recovered.learned_marker == 99
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"
    assert payload["promotable"] is False
    assert payload["reason"] == "mt5_runtime_error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["evidence"]["rlhf_policy_phase"] == {}
    assert payload["evidence"]["date_range"] == {
        "from": "2025.01.01",
        "to": "2025.01.02",
    }
    assert evaluator_state["shutdown_terminal"] is True
    assert evaluator_state["closed"] == 0


def test_keyboard_interrupt_luu_checkpoint_incomplete_roi_dung_ctrl_c(
    tmp_path: Path, monkeypatch
) -> None:
    """Ctrl+C after MT5 updates must still save a forensic learner state."""
    import json
    import self_growing_trader.chuyen_gia_policy as policy_module

    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    config = replace(
        config,
        learning=replace(
            config.learning,
            five_codex_teacher_auto_connect=False,
        ),
    )
    seed = tmp_path / "seed.pkl"
    seed.write_bytes(b"seed-placeholder")
    output = tmp_path / "candidate.pkl"
    system = _RecoverySystem()

    class InterruptingEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_policy(self, trial_id, policy, round):
            policy.learned_marker = 123
            raise KeyboardInterrupt()

        def close_runner_processes(self):
            return None

    monkeypatch.setattr(policy_module, "MT5StrategyTesterEvaluator", InterruptingEvaluator)
    monkeypatch.setattr(
        policy_module,
        "tai_he_chuyen_gia_mot_context_mot_policy",
        lambda *args, **kwargs: system,
    )
    monkeypatch.setattr(
        policy_module,
        "_require_context_expert_architecture_match",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        policy_module,
        "_cau_hinh_policy_chuyen_gia_cho_mt5",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        policy_module,
        "_reset_context_expert_runtime_for_new_mt5_session",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        policy_module,
        "_compact_context_expert_artifact_state",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        policy_module,
        "archive_context_expert_runtime_artifacts",
        lambda *args, **kwargs: {},
    )

    with pytest.raises(policy_module.MT5TrainingCheckpointError) as caught:
        policy_module.huan_luyen_he_chuyen_gia_mt5(
            config,
            output_path=output,
            seed_system_path=seed,
            runner_dir=tmp_path / "runner",
            from_date="2025.01.01",
            to_date="2025.01.02",
            resume_existing=False,
        )

    checkpoint = tmp_path / "candidate.incomplete.pkl"
    metadata = tmp_path / "candidate.incomplete.json"
    assert Path(caught.value.checkpoint_path) == checkpoint
    assert Path(caught.value.metadata_path) == metadata
    assert caught.value.original_error_type == "KeyboardInterrupt"
    assert not output.exists()
    with checkpoint.open("rb") as handle:
        recovered = pickle.load(handle)
    assert recovered.learned_marker == 123
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"
    assert payload["promotable"] is False
    assert payload["reason"] == "keyboard_interrupt"
    assert payload["error_type"] == "KeyboardInterrupt"


def test_main_runtime_is_archived_before_failure_replay_clears_shared_files(
    tmp_path: Path, monkeypatch
) -> None:
    import self_growing_trader.chuyen_gia_policy as policy_module
    import self_growing_trader.mt5_failure_replay as replay_module

    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    config = replace(
        config,
        mt5_learning_enabled=False,
        learning=replace(config.learning, five_mt5_failure_replay_sessions_per_run=2),
    )
    seed = tmp_path / "seed.pkl"
    seed.write_bytes(b"seed-placeholder")
    output = tmp_path / "candidate.pkl"
    system = _RecoverySystem()
    calls: list[str] = []
    evaluator_learning: list[bool] = []

    class Evaluator:
        def __init__(self, evaluator_config):
            self.config = evaluator_config
            evaluator_learning.append(bool(evaluator_config.learning_enabled))

        def evaluate_policy(self, trial_id, policy, round):
            calls.append("main_evaluate")
            return 1.0, types.SimpleNamespace(net_profit=1.0, report_path=tmp_path / "report.htm")

        def close_runner_processes(self):
            calls.append("close")

    class ReplayResult:
        def as_dict(self):
            return {"completed_sessions": 0, "failed_sessions": 2}

    class Coordinator:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            calls.append("replay")
            assert calls.index("archive") < calls.index("replay")
            return ReplayResult()

    monkeypatch.setattr(policy_module, "MT5StrategyTesterEvaluator", Evaluator)
    monkeypatch.setattr(replay_module, "MT5FailureReplayCoordinator", Coordinator)
    monkeypatch.setattr(
        policy_module, "tai_he_chuyen_gia_mot_context_mot_policy",
        lambda *args, **kwargs: system,
    )
    monkeypatch.setattr(policy_module, "_require_context_expert_architecture_match", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_cau_hinh_policy_chuyen_gia_cho_mt5", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_reset_context_expert_runtime_for_new_mt5_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(policy_module, "_compact_context_expert_artifact_state", lambda *args, **kwargs: None)

    def fake_archive(*args, **kwargs):
        calls.append("archive")
        return {"action": str(tmp_path / "candidate.runtime" / "action.csv")}

    monkeypatch.setattr(policy_module, "archive_context_expert_runtime_artifacts", fake_archive)

    result = policy_module.huan_luyen_he_chuyen_gia_mt5(
        config,
        output_path=output,
        seed_system_path=seed,
        runner_dir=tmp_path / "runner",
        from_date="2025.01.01",
        to_date="2025.01.02",
        resume_existing=False,
    )

    assert calls.count("archive") == 1
    assert evaluator_learning == [bool(config.mt5_learning_enabled)]
    assert result["runtime_artifacts"]["action"].endswith("action.csv")
    assert output.is_file()


def test_damaging_post_replay_is_rolled_back_to_main_policy(
    tmp_path: Path, monkeypatch
) -> None:
    import self_growing_trader.chuyen_gia_policy as policy_module
    import self_growing_trader.mt5_failure_replay as replay_module

    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    config = replace(
        config,
        learning=replace(
            config.learning,
            five_mt5_failure_replay_sessions_per_run=2,
            five_codex_teacher_auto_connect=False,
        ),
    )
    seed = tmp_path / "seed.pkl"
    seed.write_bytes(b"seed-placeholder")
    output = tmp_path / "candidate.pkl"
    system = _RecoverySystem()
    archive_calls: list[int] = []

    class Evaluator:
        def __init__(self, evaluator_config):
            self.config = evaluator_config

        def evaluate_policy(self, trial_id, policy, round):
            if trial_id == 830_000:
                return -100.0, types.SimpleNamespace(net_profit=-5.0, report_path=tmp_path / "frozen.htm")
            return 10.0, types.SimpleNamespace(net_profit=12.0, report_path=tmp_path / "main.htm")

        def close_runner_processes(self):
            return None

    class ReplayResult:
        def as_dict(self):
            return {"completed_sessions": 2, "failed_sessions": 0}

    class Coordinator:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, policy, *args, **kwargs):
            policy.learned_marker = 99
            return ReplayResult()

    monkeypatch.setattr(policy_module, "MT5StrategyTesterEvaluator", Evaluator)
    monkeypatch.setattr(replay_module, "MT5FailureReplayCoordinator", Coordinator)
    monkeypatch.setattr(policy_module, "clone_policy", copy.deepcopy)
    monkeypatch.setattr(
        policy_module, "tai_he_chuyen_gia_mot_context_mot_policy",
        lambda *args, **kwargs: system,
    )
    monkeypatch.setattr(policy_module, "_require_context_expert_architecture_match", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_cau_hinh_policy_chuyen_gia_cho_mt5", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_reset_context_expert_runtime_for_new_mt5_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(policy_module, "_compact_context_expert_artifact_state", lambda *args, **kwargs: None)

    def fake_archive(*args, **kwargs):
        archive_calls.append(1)
        return {"report": str(tmp_path / "main.htm")}

    monkeypatch.setattr(policy_module, "archive_context_expert_runtime_artifacts", fake_archive)

    result = policy_module.huan_luyen_he_chuyen_gia_mt5(
        config,
        output_path=output,
        seed_system_path=seed,
        runner_dir=tmp_path / "runner",
        from_date="2025.01.01",
        to_date="2025.01.02",
        resume_existing=False,
    )

    with output.open("rb") as handle:
        saved = pickle.load(handle)
    validation = result["metrics"]["mt5_failure_replay"]["post_replay_validation"]
    assert saved.learned_marker == 17
    assert result["fitness"] == 10.0
    assert result["metrics"]["net_profit"] == 12.0
    assert validation["accepted"] is False
    assert validation["rollback_applied"] is True
    assert validation["metrics_match_saved_weights"] is True
    assert len(archive_calls) == 1


def test_paired_frozen_gate_rolls_back_candidate_to_protected_seed(
    tmp_path: Path, monkeypatch
) -> None:
    import self_growing_trader.chuyen_gia_policy as policy_module

    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    config = replace(
        config,
        learning=replace(
            config.learning,
            five_mt5_failure_replay_sessions_per_run=0,
            five_codex_teacher_auto_connect=False,
        ),
    )
    seed = tmp_path / "seed.pkl"
    seed.write_bytes(b"seed-placeholder")
    output = tmp_path / "candidate.pkl"
    candidate_system = _RecoverySystem()
    baseline_system = _RecoverySystem()
    loaded = iter((candidate_system, baseline_system))

    class Evaluator:
        def __init__(self, evaluator_config):
            self.config = evaluator_config

        def evaluate_policy(self, trial_id, policy, round):
            if trial_id == 1:
                policy.learned_marker = 99
                return 1.0, types.SimpleNamespace(
                    net_profit=1.0, report_path=tmp_path / "main.htm"
                )
            if trial_id == 840_001:
                return -5.0, types.SimpleNamespace(
                    net_profit=-4.0,
                    edge=-0.1,
                    equity_drawdown_pct=3.0,
                    fitness_policy_expectancy_lcb=-0.2,
                    fitness_eligible=False,
                    fitness_reconciled=True,
                    execution_summary={"expectancy": -0.1, "expectancy_lcb": -0.2},
                    report_path=tmp_path / "candidate_frozen.htm",
                )
            assert trial_id == 840_000
            return 4.0, types.SimpleNamespace(
                net_profit=3.0,
                edge=0.05,
                equity_drawdown_pct=2.0,
                fitness_policy_expectancy_lcb=0.01,
                fitness_eligible=True,
                fitness_reconciled=True,
                execution_summary={"expectancy": 0.05, "expectancy_lcb": 0.01},
                report_path=tmp_path / "baseline_frozen.htm",
            )

        def close_runner_processes(self):
            return None

    monkeypatch.setattr(policy_module, "MT5StrategyTesterEvaluator", Evaluator)
    monkeypatch.setattr(policy_module, "clone_policy", copy.deepcopy)
    monkeypatch.setattr(
        policy_module,
        "tai_he_chuyen_gia_mot_context_mot_policy",
        lambda *args, **kwargs: next(loaded),
    )
    monkeypatch.setattr(policy_module, "_require_context_expert_architecture_match", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_cau_hinh_policy_chuyen_gia_cho_mt5", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "_reset_context_expert_runtime_for_new_mt5_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(policy_module, "_compact_context_expert_artifact_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "archive_context_expert_runtime_artifacts", lambda *args, **kwargs: {})

    result = policy_module.huan_luyen_he_chuyen_gia_mt5(
        config,
        output_path=output,
        seed_system_path=seed,
        runner_dir=tmp_path / "runner",
        from_date="2025.01.01",
        to_date="2025.01.02",
        resume_existing=False,
        paired_seed_validation=True,
    )

    with output.open("rb") as handle:
        saved = pickle.load(handle)
    paired = result["metrics"]["paired_seed_validation"]
    assert saved.learned_marker == 17
    assert result["fitness"] == pytest.approx(4.0)
    assert result["metrics"]["net_profit"] == pytest.approx(3.0)
    assert paired["accepted"] is False
    assert paired["selected"] == "source_seed"
    assert paired["rollback_applied"] is True
    assert "candidate_frozen_fitness_ineligible" in paired["paired_gate"]["reasons"]


def test_context_classifier_reset_clears_m15_cache_without_recreating_detector() -> None:
    classifier = BoNhanDienNguCanh(CauHinhHeChuyenGia())
    classifier._ma_hien_tai = "CTX_VOLATILE"
    classifier._last_regime_signature = (1.0, 2.0, 3.0, 4.0)
    classifier._last_regime_candidate = ("CTX_GIAM_MANH", "CTX_GIAM_MANH", object(), "cached")
    detector_before = classifier._classifier

    classifier.reset_runtime_state()

    assert classifier._ma_hien_tai == ""
    assert classifier._last_regime_signature is None
    assert classifier._last_regime_candidate is None
    assert classifier._classifier is detector_before


def test_loader_rebinds_current_profile_before_volatile_owner_migration(tmp_path: Path, monkeypatch) -> None:
    current = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    legacy_expert_config = replace(current.he_chuyen_gia, volatile_subexperts_bat=False)
    legacy = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=legacy_expert_config,
        tao_policy=_ResumePolicy,
    )
    artifact = tmp_path / "legacy_v13.pkl"
    with artifact.open("wb") as handle:
        pickle.dump(legacy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    import self_growing_trader.chuyen_gia_policy as policy_module

    monkeypatch.setattr(policy_module, "tao_policy_chuyen_gia_doc_lap", lambda config, seed=None: _ResumePolicy)
    loaded = tai_he_chuyen_gia_mot_context_mot_policy(artifact, current)

    assert loaded.cau_hinh is current.he_chuyen_gia
    assert loaded.nhan_dien.cau_hinh is current.he_chuyen_gia
    assert loaded.registry.cau_hinh is current.he_chuyen_gia
    # Cấu hình năm tác nhân hiện hành chỉ có một chủ sở hữu trọng số chung;
    # không được tái sinh các chủ sở hữu volatile của kiến trúc cũ khi nạp lại.
    assert set(loaded.experts) == {"CTX_CHUNG"}


def test_loader_migrates_removed_context_structural_controller_from_legacy_pickle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Old v13 artifacts load once, discard the controller, then stay clean."""
    current = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=current.he_chuyen_gia,
        tao_policy=_ResumePolicy,
    )

    module_name = "self_growing_trader.phat_trien_cau_truc_context"
    legacy_module = types.ModuleType(module_name)
    legacy_type = type("BoDieuKhienPhatTrienCauTruc", (), {})
    legacy_type.__module__ = module_name
    setattr(legacy_module, legacy_type.__name__, legacy_type)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = legacy_module
    try:
        controller = legacy_type()
        controller.old_threshold = 0.25
        next(iter(system.experts.values())).policy._context_structural_growth_controller = controller
        artifact = tmp_path / "v13_with_removed_structural_controller.pkl"
        with artifact.open("wb") as handle:
            pickle.dump(system, handle, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    import self_growing_trader.chuyen_gia_policy as policy_module

    monkeypatch.setattr(policy_module, "tao_policy_chuyen_gia_doc_lap", lambda config, seed=None: _ResumePolicy)
    loaded = tai_he_chuyen_gia_mot_context_mot_policy(artifact, current)

    assert all(
        "_context_structural_growth_controller" not in vars(expert.policy)
        for expert in loaded.experts.values()
    )
    # A re-saved artifact no longer depends on the deleted module.
    clean = tmp_path / "migrated.pkl"
    with clean.open("wb") as handle:
        pickle.dump(loaded, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with clean.open("rb") as handle:
        restored = pickle.load(handle)
    assert isinstance(restored, HeChuyenGiaMotContextMotPolicy)


def test_loader_discards_removed_context_nursery_from_legacy_pickle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=current.he_chuyen_gia,
        tao_policy=_ResumePolicy,
    )

    import self_growing_trader.he_chuyen_gia_ngu_canh as facade_module

    legacy_type = type("BoUomNguCanh", (), {})
    legacy_type.__module__ = facade_module.__name__
    setattr(facade_module, legacy_type.__name__, legacy_type)
    try:
        legacy_nursery = legacy_type()
        legacy_nursery.bang_chung = {"old": {"count": 10.0}}
        system.__dict__["uom_context"] = legacy_nursery
        artifact = tmp_path / "legacy_with_context_nursery.pkl"
        with artifact.open("wb") as handle:
            pickle.dump(system, handle, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        delattr(facade_module, legacy_type.__name__)

    import self_growing_trader.chuyen_gia_policy as policy_module

    monkeypatch.setattr(
        policy_module,
        "tao_policy_chuyen_gia_doc_lap",
        lambda config, seed=None: _ResumePolicy,
    )
    loaded = tai_he_chuyen_gia_mot_context_mot_policy(artifact, current)

    assert not hasattr(loaded, "uom_context")
    assert not hasattr(loaded.registry, "context_con_theo_cha")


def test_copy_migrates_legacy_module_without_value_head_before_role_fork() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=7, output_size=4, hidden_size=5, seed=41)
    module = policy.modules[0]
    del module.value_w
    del module.value_b

    copied = policy.copy()

    assert copied.modules[0].value_w.shape == (5,)
    assert (copied.modules[0].value_w == 0.0).all()
    assert copied.modules[0].value_b == 0.0


def test_role_fork_migrates_legacy_parent_before_copying_modules() -> None:
    system = tao_he_chuyen_gia_mot_context_mot_policy(
        load_config("configs/he_chuyen_gia.yaml"),
        seed=51,
    )
    parent = next(iter(system.experts.values()))
    del parent.policy.modules[0].value_w
    del parent.policy.modules[0].value_b

    role_policy = system._fork_role_policy("MANAGEMENT", parent, warm_start=True)

    assert role_policy.modules[0].value_w.shape == (parent.policy.modules[0].hidden_size,)
    assert role_policy.modules[0].value_b == 0.0
