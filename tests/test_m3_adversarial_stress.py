"""Adversarial Stress Test Suite for Milestone 3 (Immutability, Promotion & Rollback Safety).

Adversarially challenges:
1. Weight Mutation Detection: Perturb parameter tensors by minute delta (1e-6) during evaluation;
   verify PolicyImmutabilityError is raised and candidate is blocked from promotion.
2. Interrupted Promotion Simulation: Verify .tmp pattern guarantees no corrupted partially-written
   champion files can exist, even when .tmp is pre-existing or interrupted mid-write.
3. Rollback Invariant: On gate failure, active_seed retains original champion, candidate binary payloads
   are cleanly pruned to prevent disk exhaustion, and telemetry history is completely preserved.
4. Concurrency & Collision Stress: Rapid sequential and concurrent directory creation and atomic
   promotions on Windows NTFS; verify resilience against WinError 183.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from self_growing_trader.auto_rl_engine import prune_unpromoted_checkpoints
from self_growing_trader.multi_regime_gate import (
    CheckpointPromotionEngine,
    IsolatedMT5RunnerManager,
    MultiRegimeGateEvaluator,
    PolicyImmutabilityError,
    compute_policy_state_hash,
    evaluate_multi_regime_gate,
    promote_champion_atomic,
    rollback_candidate_atomic,
)
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_multi_regime_gate import MockTelemetryBuilder, _make_passing_regimes


# ==============================================================================
# CHALLENGE 1: WEIGHT MUTATION DETECTION (PARAMETER PERTURBATION)
# ==============================================================================

class SimplePolicyNetwork(nn.Module):
    """Synthetic PyTorch policy network matching weight structure."""
    def __init__(self, input_dim: int = 137, hidden_dim: int = 128, output_dim: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = torch.relu(self.fc1(x))
        h_next, c_next = self.lstm(feat, (h, c))
        logits = self.head(h_next)
        return logits, h_next, c_next


def test_hash_sensitivity_single_parameter_perturbation():
    """Hypothesis 1.1: Perturbing a single float parameter by 1e-6 MUST change SHA-256 state hash."""
    torch.manual_seed(42)
    model = SimplePolicyNetwork()

    hash_original = compute_policy_state_hash(model)

    # Clone state dict and verify equality
    state = model.state_dict()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    hash_from_buffer = compute_policy_state_hash(model)
    assert hash_original == hash_from_buffer, "Hashing must be deterministic"

    # Perturb exactly ONE float element in fc1.weight by 1e-6
    with torch.no_grad():
        model.fc1.weight[0, 0] += 1e-6

    hash_perturbed = compute_policy_state_hash(model)
    assert hash_perturbed != hash_original, (
        f"Hash collision! Perturbation of 1e-6 failed to alter hash: {hash_original} vs {hash_perturbed}"
    )


def test_hash_sensitivity_numpy_weights_perturbation():
    """Hypothesis 1.2: Perturbing a numpy-backed policy array by 1e-7 MUST change SHA-256 state hash."""
    class MockNumpyPolicy:
        def __init__(self):
            self.weights = np.ones((100, 100), dtype=np.float32)

    pol = MockNumpyPolicy()
    h1 = compute_policy_state_hash(pol)

    pol.weights[42, 42] += 1e-7
    h2 = compute_policy_state_hash(pol)

    assert h1 != h2, "Numpy policy perturbation was not detected by state hash!"


def test_policy_immutability_error_blocks_promotion(tmp_path: Path):
    """Hypothesis 1.3: If candidate checkpoint file is perturbed during evaluation,

    MultiRegimeGateEvaluator MUST raise PolicyImmutabilityError, and NO champion
    checkpoint or active seed may be written or updated.
    """
    cand_path = tmp_path / "candidate_model.pkl"
    original_bytes = b"canonical_checkpoint_v193_unperturbed"
    cand_path.write_bytes(original_bytes)

    out_dir = tmp_path / "promotions"
    active_seed = tmp_path / "active_seed.pkl"
    seed_original = b"initial_seed_unmodified"
    active_seed.write_bytes(seed_original)

    evaluator = MultiRegimeGateEvaluator()

    # Adversarial evaluator: mutates the checkpoint file on disk mid-evaluation
    def malicious_mutating_eval(r_id: str, cfg: dict[str, str], r_out: Path) -> Path:
        # Subtle perturbation: flip 1 byte in the candidate file
        perturbed = bytearray(original_bytes)
        perturbed[-1] = (perturbed[-1] + 1) % 256
        cand_path.write_bytes(bytes(perturbed))

        # Generate passing telemetry so only immutability fails
        b = MockTelemetryBuilder("uptrend")
        b.add_synthetic_batch(wins=25, losses=5)
        return b.write_to_directory(r_out)

    with pytest.raises(PolicyImmutabilityError, match="mutated during evaluation"):
        evaluator.evaluate_candidate(
            candidate_path=cand_path,
            output_dir=out_dir,
            evaluator_fn=malicious_mutating_eval,
            active_seed_path=active_seed,
            round_id=1,
        )

    # Invariant Verification: Champion must NOT be promoted!
    canonical_champion = out_dir / "best_promoted_checkpoint.pkl"
    round_champion = out_dir / "champion_phase3_round_01.pkl"
    assert not canonical_champion.exists(), "Fatal: Corrupted/mutated candidate was promoted to canonical champion!"
    assert not round_champion.exists(), "Fatal: Corrupted/mutated candidate was promoted to round champion!"

    # Active seed must remain 100% untouched
    assert active_seed.read_bytes() == seed_original, "Fatal: Active seed was overwritten by mutated candidate!"


def test_checkpoint_engine_assert_weight_immutability_detects_mutation():
    """Hypothesis 1.4: CheckpointPromotionEngine.assert_weight_immutability catches direct mutation."""
    engine = CheckpointPromotionEngine()
    torch.manual_seed(100)
    model = SimplePolicyNetwork()

    # Evaluator that secretly mutates the model in-place
    def mutating_fn(clone_model: Any) -> str:
        with torch.no_grad():
            model.fc1.bias[0] += 0.05
        return "eval_done"

    with pytest.raises(PolicyImmutabilityError, match="mutated during evaluation"):
        engine.assert_weight_immutability(model, mutating_fn)


# ==============================================================================
# CHALLENGE 2: INTERRUPTED PROMOTION SIMULATION (.TMP FILE SAFETY)
# ==============================================================================

def test_interrupted_promotion_leaves_prior_champion_uncorrupted(tmp_path: Path):
    """Hypothesis 2.1: If promotion is aborted while .tmp is partially written,

    the destination champion remains 100% byte-for-byte valid.
    """
    out_dir = tmp_path / "checkpoint_output"
    out_dir.mkdir(parents=True)
    champion_path = out_dir / "best_promoted_checkpoint.pkl"

    champion_valid_data = b"VALID_CHAMPION_ROUND_01_STABLE_WEIGHTS_" * 1000
    champion_path.write_bytes(champion_valid_data)

    # Simulate an aborted/crashed promotion that wrote a corrupted half-file to .tmp
    corrupted_tmp = out_dir / "best_promoted_checkpoint.tmp"
    corrupted_tmp.write_bytes(b"CORRUPTED_TRUNCATED_GARBAGE_HALF_WRITTEN")

    # Invariant: Prior champion must not be affected by the existence of corrupted .tmp
    assert champion_path.read_bytes() == champion_valid_data

    # Now run an atomic promotion of a new candidate
    new_cand = tmp_path / "candidate_round_02.pkl"
    new_cand_data = b"VALID_NEW_CHAMPION_ROUND_02_WEIGHTS_" * 1000
    new_cand.write_bytes(new_cand_data)

    promoted, _ = promote_champion_atomic(
        candidate_path=new_cand,
        output_dir=out_dir,
        champion_name="best_promoted_checkpoint.pkl",
        round_id=2,
    )

    # Verify:
    # 1. Champion is now new candidate
    assert champion_path.read_bytes() == new_cand_data
    # 2. Corrupted .tmp was replaced/overwritten cleanly
    assert not corrupted_tmp.exists()


def test_checkpoint_promotion_engine_atomic_write_crash_simulation(tmp_path: Path):
    """Hypothesis 2.2: CheckpointPromotionEngine._atomic_write ensures crash-safety

    even if destination directory has stale .tmp or .registry.json.tmp files.
    """
    engine = CheckpointPromotionEngine(checkpoint_dir=tmp_path / "loop")
    dest_path = tmp_path / "loop" / "best_promoted_checkpoint.pkl"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    dest_path.write_bytes(b"EXISTING_SAFE_CHAMPION")
    final_reg = dest_path.with_suffix(".registry.json")
    final_reg.write_text('{"round": 1, "status": "SAFE"}', encoding="utf-8")

    # Plant stale/corrupted temporary files simulating crash of earlier attempt
    stale_tmp_pkl = dest_path.with_suffix(dest_path.suffix + ".tmp")
    stale_tmp_pkl.write_bytes(b"HALF_WRITTEN_CORRUPTED_BYTES")
    stale_tmp_reg = dest_path.with_suffix(".registry.json.tmp")
    stale_tmp_reg.write_text("NOT_VALID_JSON{", encoding="utf-8")

    # Pre-state verification
    assert dest_path.read_bytes() == b"EXISTING_SAFE_CHAMPION"

    # Execute atomic write
    new_source = tmp_path / "new_model.pkl"
    new_source.write_bytes(b"NEW_VALID_MODEL_PAYLOAD")
    new_registry = {"round": 2, "status": "PROMOTED_CHAMPION"}

    engine._atomic_write(new_source, dest_path, new_registry)

    # Post-state verification:
    assert dest_path.read_bytes() == b"NEW_VALID_MODEL_PAYLOAD"
    assert json.loads(final_reg.read_text(encoding="utf-8"))["status"] == "PROMOTED_CHAMPION"
    assert not stale_tmp_pkl.exists()
    assert not stale_tmp_reg.exists()


# ==============================================================================
# CHALLENGE 3: ROLLBACK INVARIANT & DISK PRUNING
# ==============================================================================

def test_rollback_on_gate_rejection_cleans_candidate_preserves_telemetry(tmp_path: Path):
    """Hypothesis 3.1: On gate failure:

    1. active_seed remains bound to previous champion.
    2. candidate binary (.pkl) is unlinked to avoid disk bloat.
    3. Telemetry files (events.jsonl, mt5_diagnostics.json, manifest.json) remain 100% intact.
    """
    work_dir = tmp_path / "run_session"
    work_dir.mkdir()

    # Active champion seed
    active_seed = work_dir / "active_seed.pkl"
    active_seed_data = b"CHAMPION_ROUND_04_WEIGHTS_PERSISTED"
    active_seed.write_bytes(active_seed_data)

    # Failing candidate round
    round_dir = work_dir / "round_05"
    round_dir.mkdir()
    cand_path = round_dir / "candidate.pkl"
    cand_path.write_bytes(b"FAILED_CANDIDATE_LARGE_BINARY_100MB")
    cand_tmp = round_dir / "candidate.tmp"
    cand_tmp.write_bytes(b"TEMPORARY_CACHE_SCRATCH")

    # Telemetry artifacts
    events_file = round_dir / "events.jsonl"
    events_file.write_text('{"event_type": "lifecycle", "payload": {"realized_net_r": -0.8}}\n', encoding="utf-8")
    diag_file = round_dir / "mt5_diagnostics.json"
    diag_file.write_text('{"performance": {"win_rate": 30.0}}', encoding="utf-8")
    manifest_file = round_dir / "manifest.json"
    manifest_file.write_text('{"round": 5, "gate_status": "REJECTED"}', encoding="utf-8")

    from self_growing_trader.multi_regime_gate import RegimeMetrics

    # Evaluate candidate with failing metrics (total trades = 30, WR = 20%)
    failing_metrics = {
        "bull": RegimeMetrics(
            regime_id="bull",
            market_type="uptrend",
            total_trades=10,
            wins=2,
            losses=8,
            win_rate_pct=20.0,
            expectancy_r=-0.20,
            payoff_ratio=0.80,
        ),
        "bear": RegimeMetrics(
            regime_id="bear",
            market_type="downtrend",
            total_trades=10,
            wins=2,
            losses=8,
            win_rate_pct=20.0,
            expectancy_r=-0.20,
            payoff_ratio=0.80,
        ),
        "consolidation": RegimeMetrics(
            regime_id="consolidation",
            market_type="sideway",
            total_trades=10,
            wins=2,
            losses=8,
            win_rate_pct=20.0,
            expectancy_r=-0.20,
            payoff_ratio=0.80,
        ),
    }

    evaluator = MultiRegimeGateEvaluator()
    result = evaluator.evaluate_candidate(
        candidate_path=cand_path,
        output_dir=work_dir / "eval_out",
        precomputed_metrics=failing_metrics,
        active_seed_path=active_seed,
        round_id=5,
    )

    # Invariants:
    assert result.passed_gate is False
    assert result.promoted is False
    assert result.rollback_seed_path == str(active_seed)

    # 1. active_seed must NOT have changed
    assert active_seed.read_bytes() == active_seed_data

    # 2. Candidate binary payload must be pruned
    assert not cand_path.exists(), "Failure: Candidate .pkl was not unlinked on rollback!"

    # 3. Telemetry files MUST survive
    assert events_file.exists(), "Fatal: Telemetry events.jsonl was deleted on rollback!"
    assert diag_file.exists(), "Fatal: Telemetry mt5_diagnostics.json was deleted on rollback!"
    assert manifest_file.exists(), "Fatal: Telemetry manifest.json was deleted on rollback!"


def test_rollback_safety_guard_candidate_equals_active_seed(tmp_path: Path):
    """Hypothesis 3.2: If candidate_path points to active_seed, rollback MUST NOT delete it."""
    active_seed = tmp_path / "champion.pkl"
    seed_content = b"PRECIOUS_CHAMPION_MUST_NOT_BE_DELETED"
    active_seed.write_bytes(seed_content)

    restored = rollback_candidate_atomic(
        candidate_path=active_seed,
        champion_seed_path=active_seed,
        prune_candidate_payload=True,
    )

    assert restored == active_seed
    assert active_seed.exists(), "Fatal: rollback_candidate_atomic unlinked active seed when cand == champ!"
    assert active_seed.read_bytes() == seed_content


# ==============================================================================
# CHALLENGE 4: CONCURRENCY, RAPID ITERATION & WINERROR 183 STRESS
# ==============================================================================

def test_isolated_runner_rapid_sequential_creation(tmp_path: Path):
    """Hypothesis 4.1: Rapid sequential runner creations must never raise WinError 183."""
    base_runner = tmp_path / "base_runner"
    base_runner.mkdir()
    (base_runner / "terminal64.exe").write_bytes(b"TERMINAL_MOCK")
    (base_runner / "MQL5" / "Experts").mkdir(parents=True)
    (base_runner / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").write_text("// EA", encoding="utf-8")

    runners: list[Path] = []
    # Create 50 runners sequentially in a tight loop
    for i in range(50):
        r = IsolatedMT5RunnerManager.create_isolated_runner(base_runner, f"regime_{i%3}")
        assert r.exists()
        runners.append(r)

    # Cleanup all
    for r in runners:
        IsolatedMT5RunnerManager.cleanup_isolated_runner(r)
        assert not r.exists()


def test_isolated_runner_concurrent_creation(tmp_path: Path):
    """Hypothesis 4.2: Concurrent multi-threaded runner creations must not fail with WinError 183."""
    base_runner = tmp_path / "base_runner_concurrent"
    base_runner.mkdir()
    (base_runner / "terminal64.exe").write_bytes(b"TERMINAL_MOCK")
    (base_runner / "MQL5" / "Experts").mkdir(parents=True)
    (base_runner / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").write_text("// EA", encoding="utf-8")

    def _worker(worker_id: int) -> Path:
        return IsolatedMT5RunnerManager.create_isolated_runner(base_runner, f"worker_{worker_id}")

    created: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            res_path = f.result()
            assert res_path.exists()
            created.append(res_path)

    assert len(created) == 20
    # Cleanup all
    for r in created:
        IsolatedMT5RunnerManager.cleanup_isolated_runner(r)
        assert not r.exists()


def test_checkpoint_promotion_engine_rapid_sequential_promotions(tmp_path: Path):
    """Hypothesis 4.3: 30 rapid sequential promotions must succeed without WinError 183

    and properly maintain bounded rollback backups (max_rollback_backups).
    """
    engine = CheckpointPromotionEngine(checkpoint_dir=tmp_path / "rapid_loop", max_rollback_backups=3)

    cand_src = tmp_path / "cand.pkl"
    cand_src.write_bytes(b"INITIAL_CANDIDATE")

    # Run 25 rapid promotions
    for r in range(1, 26):
        cand_src.write_bytes(f"CANDIDATE_ROUND_{r}_BYTES".encode("utf-8"))
        res = engine.atomic_promote(
            candidate_source=cand_src,
            round_index=r,
            gate_summary={"round": r, "passed": True},
        )
        assert res.promoted is True
        assert res.champion_path.exists()

    # Verify canonical champion has the latest round bytes
    assert engine.canonical_champion_path.read_bytes() == b"CANDIDATE_ROUND_25_BYTES"

    # Verify bounded rollback backups: no more than max_rollback_backups
    backups = list(engine.canonical_champion_path.parent.glob(f"{engine.canonical_champion_path.stem}.rollback.*.pkl"))
    assert len(backups) <= 3, f"Backups exceeded quota: found {len(backups)} backups, expected <= 3"


def test_atomic_rename_windows_replace_existing_semantics(tmp_path: Path):
    """Hypothesis 4.4: Path.replace on Windows replaces existing files atomically without WinError 183."""
    target_file = tmp_path / "target_champion.pkl"
    target_file.write_bytes(b"INITIAL_VERSION_1")

    # Rapid replacements
    for i in range(2, 20):
        tmp_file = tmp_path / f"temp_{i}.tmp"
        payload = f"VERSION_{i}_PAYLOAD".encode("utf-8")
        tmp_file.write_bytes(payload)
        tmp_file.replace(target_file)
        assert target_file.read_bytes() == payload


# ==============================================================================
# ADDITIONAL ADVERSARIAL STRESS PROBES
# ==============================================================================

def test_pruning_matrix_exhaustive_patterns(tmp_path: Path):
    """Stress Probe 5: Exhaustive test of file patterns on rollback pruning.

    Verifies binary model payloads are deleted, while all telemetry, logs, charts,
    and analysis files are preserved.
    """
    round_dir = tmp_path / "comprehensive_round"
    round_dir.mkdir()

    # Files that MUST be pruned (heavy binary weights)
    prune_targets = [
        "candidate.pkl",
        "candidate.tmp",
        "candidate_phase3.pkl",
        "checkpoint_round_05.pkl",
        "candidate_step_200.pkl",
        "model_candidate.pkl",
    ]
    for name in prune_targets:
        (round_dir / name).write_bytes(b"LARGE_BINARY_MODEL_DATA_50MB")

    # Files that MUST be preserved (telemetry, reports, logs)
    preserve_targets = [
        "events.jsonl",
        "mt5_diagnostics.json",
        "manifest.json",
        "evolution_chart.json",
        "phase3_analysis_report.md",
        "training.log",
        "equity_curve.csv",
        "drawdown_chart.png",
    ]
    for name in preserve_targets:
        (round_dir / name).write_text("IMPORTANT_TELEMETRY_RECORD", encoding="utf-8")

    pruned = prune_unpromoted_checkpoints(round_dir)
    pruned_names = {p.name for p in pruned}

    for name in prune_targets:
        assert name in pruned_names, f"Expected {name} to be pruned, but was retained!"
        assert not (round_dir / name).exists(), f"File {name} still exists on disk!"

    for name in preserve_targets:
        assert name not in pruned_names, f"Expected {name} to be preserved, but was pruned!"
        assert (round_dir / name).exists(), f"Preserved file {name} was unexpectedly deleted!"


@pytest.mark.xfail(
    reason="Empirical Vulnerability Finding: Unsynchronized concurrent promotions collide on static .tmp filename on Windows NTFS (WinError 5)",
    strict=False,
)
def test_concurrent_multi_threaded_promotions_static_tmp_collision(tmp_path: Path):
    """Stress Probe 6A: Demonstrates that un-synchronized concurrent promotions on Windows

    can clash on the static '.tmp' filename (raising WinError 5 Access is denied).
    Reported as a finding with recommended mitigation (unique PID/time_ns temp suffix).
    """
    loop_dir = tmp_path / "concurrent_loop"
    engine = CheckpointPromotionEngine(checkpoint_dir=loop_dir, max_rollback_backups=3)

    cand_template = tmp_path / "cand_template.pkl"
    cand_template.write_bytes(b"BASE_TEMPLATE_WEIGHTS")

    def _promote_worker(worker_id: int):
        c_path = tmp_path / f"cand_worker_{worker_id}.pkl"
        c_path.write_bytes(f"WORKER_{worker_id}_WEIGHTS".encode("utf-8"))
        res = engine.atomic_promote(
            candidate_source=c_path,
            round_index=worker_id,
            gate_summary={"worker": worker_id, "passed": True},
        )
        return res.promoted

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_promote_worker, i) for i in range(1, 16)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(results)


def test_concurrent_promotions_with_synchronization(tmp_path: Path):
    """Stress Probe 6B: Proves that when promotions are serialized via synchronization lock,

    atomic promotions and backup rotations succeed flawlessly under multi-threaded load.
    """
    import threading
    loop_dir = tmp_path / "concurrent_loop_sync"
    engine = CheckpointPromotionEngine(checkpoint_dir=loop_dir, max_rollback_backups=3)
    promo_lock = threading.Lock()

    def _safe_worker(worker_id: int):
        c_path = tmp_path / f"cand_safe_{worker_id}.pkl"
        c_path.write_bytes(f"SAFE_WORKER_{worker_id}".encode("utf-8"))
        with promo_lock:
            res = engine.atomic_promote(
                candidate_source=c_path,
                round_index=worker_id,
                gate_summary={"worker": worker_id, "passed": True},
            )
            return res.promoted

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_safe_worker, i) for i in range(1, 16)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(results)
    assert engine.canonical_champion_path.exists()
    assert (engine.canonical_champion_path.with_suffix(".registry.json")).exists()
    backups = list(engine.canonical_champion_path.parent.glob(f"{engine.canonical_champion_path.stem}.rollback.*.pkl"))
    assert len(backups) <= 3


def test_hash_sensitivity_numerical_precision_limits():
    """Stress Probe 7: Determine minimum perturbation epsilon detected by state hash."""
    torch.manual_seed(12345)
    model = SimplePolicyNetwork()
    base_hash = compute_policy_state_hash(model)

    # Test float32 machine epsilon level perturbation (~1e-7)
    with torch.no_grad():
        model.fc1.weight[0, 0] += 1e-7
    eps_hash = compute_policy_state_hash(model)
    assert eps_hash != base_hash, "Hash failed to detect float32 machine epsilon (~1e-7) perturbation!"

    # Test even smaller perturbation (1e-6)
    with torch.no_grad():
        model.fc1.weight[0, 0] += 1e-6
    micro_hash = compute_policy_state_hash(model)
    assert micro_hash != eps_hash, "Hash failed to detect 1e-6 perturbation!"
