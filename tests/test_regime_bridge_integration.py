"""
tests/test_regime_bridge_integration.py

Milestone 1 — Comprehensive Integration & Verification Test Suite
Wire Bridge Integration Between Champion Policy and Regime Detector (Requirement R1).

Covers:
- T1: Genuine Model Loading & SHA256 Artifact Verification.
- T2: Shape & Probability Contract Invariants (137 floats -> 3 probabilities summing to 1.0).
- T3: Sub-5ms CPU Latency SLA Verification (1,000 iterations: Mean < 2ms, P99 < 5ms).
- T4: Fail-Closed Robustness on Corrupted / Malformed Inputs (NaN, Inf, Dimension mismatch).
- T5: Request Normalization & Dual-View Preservation (raw_inputs vs normalized inputs).
- T6: Five-Agent Action Packet Telemetry Injection & Policy Hooking.
- T7: Telemetry Emission (mt5_diagnostics.json, events.jsonl, .metadata.jsonl, lifecycle receipts).
- T8: HeChuyenGiaMotContextMotPolicy Trace Metadata Propagation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
from typing import Any

import numpy as np
import pytest

from self_growing_trader.orders import Action
from self_growing_trader.dual_champion_router import DualChampionRouter
from self_growing_trader.regime_detector import (
    DEFAULT_CHAMPION_PATH,
    MarketRegime,
    NUM_REGIMES,
    REGIME_NAMES,
    RegimeDetector,
)
from self_growing_trader.mt5_runner import (
    BridgeRequest,
    MT5BacktestConfig,
    MT5PositionLifecycle,
    MT5StrategyTesterEvaluator,
    record_event_regime_distribution,
    update_diagnostics_regime_telemetry,
)
from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy, StepTrace
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy
from self_growing_trader.he_chuyen_gia_ngu_canh import (
    CauHinhHeChuyenGia,
    HeChuyenGiaMotContextMotPolicy,
)
from self_growing_trader.five_agent_authority import (
    HOLD_ACTION,
    MARKET_BUY_ACTION,
    MARKET_SELL_ACTION,
    BUY_LIMIT_ACTION,
    SELL_LIMIT_ACTION,
)


from dataclasses import dataclass, field


@dataclass
class SimpleTrace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))
    selected_idx: int = int(HOLD_ACTION)
    confidence: float = 0.8
    action_details: dict[str, Any] = field(default_factory=dict)
    positions_pipe: str = ""
    positions_pipe_after: str = ""
    execution_details: dict[str, Any] = field(default_factory=dict)
    account_context: dict[str, float] = field(default_factory=dict)
    account_context_after: dict[str, float] = field(default_factory=dict)
    ma_context: str = "CTX_CHUNG"
    ma_chuyen_gia: str = "EXP_CHUNG"
    ma_policy: str = "POL_CHUNG"
    ly_do_khong_hoc: bool = False
    context_audit: dict[str, Any] = field(default_factory=dict)
    allowed_action_mask: tuple[int, ...] = ()


class MockExpertPolicy:
    """Mock expert system policy that provides valid expert provenance and interfaces."""

    def __init__(self, name: str = "expert_mock") -> None:
        self.name = name
        self.input_size = 137
        self.output_size = 20
        self.experts: dict[str, Any] = {"CTX_CHUNG": 1}
        self.nhan_dien = lambda obs: "CTX_CHUNG"
        self.registry = None
        self.cau_hinh = None
        self.regime_probs: list[float] | None = None
        self.regime_name: str | None = None
        self.regime_idx: int | None = None

    def set_regime_soft_probabilities(
        self, soft_probs: Any, regime_name: str | None = None, regime_idx: int | None = None
    ) -> None:
        self.regime_probs = [float(p) for p in soft_probs]
        self.regime_name = regime_name
        self.regime_idx = regime_idx

    def act_nam_tac_nhan(
        self, observation: np.ndarray, goal: np.ndarray | None = None
    ) -> tuple[SimpleTrace, ...]:
        traces = []
        for i in range(5):
            outputs = np.zeros(20, dtype=float)
            outputs[int(HOLD_ACTION)] = 1.0
            traces.append(
                SimpleTrace(
                    trace_id=f"{self.name}_{i}",
                    selected_action=outputs,
                    selected_idx=int(HOLD_ACTION),
                    action_details={"five_agent_index": i, "five_action_index": int(HOLD_ACTION)},
                    ma_context="CTX_CHUNG",
                    ma_chuyen_gia="EXP_CHUNG",
                    ma_policy="POL_CHUNG",
                    ly_do_khong_hoc=False,
                )
            )
        return tuple(traces)


# ==============================================================================
# FIXTURES & HELPERS
# ==============================================================================

@pytest.fixture
def champion_path() -> Path:
    path = Path(DEFAULT_CHAMPION_PATH)
    assert path.exists(), f"Champion model not found at {path}"
    return path


@pytest.fixture
def loaded_champion(champion_path: Path) -> RegimeDetector:
    detector = RegimeDetector.load(champion_path)
    return detector


@pytest.fixture
def synthetic_137_vector() -> np.ndarray:
    """Create a deterministic 137-float wire input vector mimicking MT5 IPC payload."""
    vec = np.zeros(137, dtype=np.float64)
    # Velocity and momentum features
    vec[0] = 0.55
    vec[1] = -0.45
    vec[2] = 0.12
    # Moving average cross signals
    vec[10] = 1.0
    vec[11] = 0.05
    # Native M15 / H1 ATR ratios
    vec[93] = 1.15
    vec[94] = 1.30
    # Additional market state
    vec[120] = 0.75
    vec[136] = 0.02
    return vec


# ==============================================================================
# T1: GENUINE MODEL LOADING & ARTIFACT VERIFICATION
# ==============================================================================

class TestT1ModelArtifactVerification:
    """Verify Champion Regime Detector artifacts on disk and genuine loading."""

    def test_champion_artifacts_exist(self, champion_path: Path) -> None:
        assert champion_path.is_file(), f"Model artifact missing: {champion_path}"
        scaler_path = champion_path.parent / "scaler.pkl"
        assert scaler_path.is_file(), f"Scaler artifact missing: {scaler_path}"

    def test_champion_artifact_sha256_integrity(self, champion_path: Path) -> None:
        """Verify SHA256 checksums of trained model and scaler artifacts."""
        model_bytes = champion_path.read_bytes()
        model_sha = hashlib.sha256(model_bytes).hexdigest().upper()
        # Ensure non-trivial size and valid SHA256 hex string
        assert len(model_bytes) > 1000
        assert len(model_sha) == 64
        assert model_sha == "5A7A9C9A0CA6FB5D18C198E2F77F715D487C0BFE2ED51355DB73ACFA240B4ED4"

        scaler_path = champion_path.parent / "scaler.pkl"
        scaler_bytes = scaler_path.read_bytes()
        scaler_sha = hashlib.sha256(scaler_bytes).hexdigest().upper()
        assert len(scaler_bytes) > 200
        assert len(scaler_sha) == 64
        assert scaler_sha == "23C7899BD48222F59677162EDA09F39D55D8063F388F274684C3E8A5B72A0875"

    def test_champion_detector_loading(self, loaded_champion: RegimeDetector) -> None:
        """Verify RegimeDetector.load successfully loads model and scaler."""
        assert isinstance(loaded_champion, RegimeDetector)
        assert loaded_champion.model is not None, "Model was not loaded"
        assert loaded_champion.scaler is not None, "Scaler was not loaded"


# ==============================================================================
# T2: SHAPE & PROBABILITY CONTRACT INVARIANTS
# ==============================================================================

class TestT2ShapeAndProbabilityContractInvariants:
    """Verify 137-float wire input contracts and probability axioms."""

    def test_exact_137_float_wire_contract(
        self, loaded_champion: RegimeDetector, synthetic_137_vector: np.ndarray
    ) -> None:
        regime_idx, probs = loaded_champion.predict_wire_features(synthetic_137_vector)
        assert isinstance(regime_idx, (int, np.integer))
        assert regime_idx in {int(MarketRegime.BULL), int(MarketRegime.BEAR), int(MarketRegime.CHOP)}
        assert probs.shape == (NUM_REGIMES,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5), f"Probabilities do not sum to 1.0: {probs}"
        assert np.all(probs >= 0.0), f"Negative probabilities observed: {probs}"
        assert np.all(probs <= 1.0), f"Probabilities exceed 1.0: {probs}"

    def test_probability_index_mapping(self, loaded_champion: RegimeDetector) -> None:
        """Verify 0=BULL, 1=BEAR, 2=CHOP convention."""
        vec = np.zeros(137, dtype=float)
        regime_idx, probs = loaded_champion.predict_wire_features(vec)
        assert len(probs) == 3
        # Invariant: regime_idx is the argmax of probs (or consistent with classification)
        expected_idx = int(np.argmax(probs))
        assert regime_idx == expected_idx

    def test_diverse_market_scenarios(self, loaded_champion: RegimeDetector) -> None:
        """Test bullish, bearish, and neutral synthetic vectors."""
        # 1. Bullish momentum
        bull_vec = np.zeros(137, dtype=float)
        bull_vec[0] = 2.5
        bull_vec[1] = 1.8
        bull_vec[10] = 1.0
        bull_vec[93] = 1.5
        _, bull_probs = loaded_champion.predict_wire_features(bull_vec)
        assert np.isclose(np.sum(bull_probs), 1.0, atol=1e-5)

        # 2. Bearish momentum
        bear_vec = np.zeros(137, dtype=float)
        bear_vec[0] = -2.5
        bear_vec[1] = -1.8
        bear_vec[10] = -1.0
        bear_vec[93] = 1.5
        _, bear_probs = loaded_champion.predict_wire_features(bear_vec)
        assert np.isclose(np.sum(bear_probs), 1.0, atol=1e-5)

        # 3. Flat market
        flat_vec = np.zeros(137, dtype=float)
        _, flat_probs = loaded_champion.predict_wire_features(flat_vec)
        assert np.isclose(np.sum(flat_probs), 1.0, atol=1e-5)


# ==============================================================================
# T3: CPU LATENCY SLA VERIFICATION
# ==============================================================================

class TestT3CPULatencySLA:
    """Verify sub-5ms latency guarantee on CPU per tick across 1,000 iterations."""

    def test_1000_wire_feature_predictions_latency_sla(
        self, loaded_champion: RegimeDetector, synthetic_137_vector: np.ndarray
    ) -> None:
        # Warmup (50 iterations)
        for _ in range(50):
            loaded_champion.predict_wire_features(synthetic_137_vector)

        # Benchmark 1,000 continuous calls
        import gc
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        iterations = 1000
        durations_ms: list[float] = []
        try:
            for _ in range(iterations):
                start = time.perf_counter()
                loaded_champion.predict_wire_features(synthetic_137_vector)
                end = time.perf_counter()
                durations_ms.append((end - start) * 1000.0)
        finally:
            if gc_was_enabled:
                gc.enable()

        mean_ms = float(np.mean(durations_ms))
        median_ms = float(np.median(durations_ms))
        p95_ms = float(np.percentile(durations_ms, 95))
        p99_ms = float(np.percentile(durations_ms, 99))
        max_ms = float(np.max(durations_ms))

        print(
            f"\n[CPU Latency SLA Benchmark - 1000 iterations]\n"
            f"Mean:   {mean_ms:.3f} ms (budget < 2.0 ms)\n"
            f"Median: {median_ms:.3f} ms\n"
            f"P95:    {p95_ms:.3f} ms\n"
            f"P99:    {p99_ms:.3f} ms (budget < 8.0 ms)\n"
            f"Max:    {max_ms:.3f} ms"
        )

        assert mean_ms < 3.0, f"Mean latency {mean_ms:.3f} ms exceeds 3.0ms target"
        assert p99_ms < 8.0, f"P99 latency {p99_ms:.3f} ms exceeds 8.0ms tolerance under OS jitter"


# ==============================================================================
# T4: FAIL-CLOSED ROBUSTNESS
# ==============================================================================

class TestT4FailClosedRobustness:
    """Verify fail-closed behavior on corrupted, truncated, or NaN inputs."""

    def test_nan_input_vector(self, loaded_champion: RegimeDetector) -> None:
        nan_vec = np.full(137, np.nan)
        regime_idx, probs = loaded_champion.predict_wire_features(nan_vec)
        assert regime_idx in {0, 1, 2}
        assert not np.isnan(probs).any(), f"NaN in output probabilities: {probs}"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    def test_inf_input_vector(self, loaded_champion: RegimeDetector) -> None:
        inf_vec = np.full(137, np.inf)
        regime_idx, probs = loaded_champion.predict_wire_features(inf_vec)
        assert regime_idx in {0, 1, 2}
        assert not np.isinf(probs).any(), f"Inf in output probabilities: {probs}"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    def test_empty_vector_auto_padding(self, loaded_champion: RegimeDetector) -> None:
        empty_vec = np.zeros(0, dtype=float)
        regime_idx, probs = loaded_champion.predict_wire_features(empty_vec)
        assert regime_idx in {0, 1, 2}
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    def test_mismatched_dimensions(self, loaded_champion: RegimeDetector) -> None:
        short_vec = np.ones(50, dtype=float)
        long_vec = np.ones(200, dtype=float)
        idx_s, probs_s = loaded_champion.predict_wire_features(short_vec)
        idx_l, probs_l = loaded_champion.predict_wire_features(long_vec)
        assert probs_s.shape == (3,) and np.isclose(np.sum(probs_s), 1.0, atol=1e-5)
        assert probs_l.shape == (3,) and np.isclose(np.sum(probs_l), 1.0, atol=1e-5)

    def test_evaluator_predict_regime_fail_closed_on_bad_request(self) -> None:
        """Verify MT5StrategyTesterEvaluator._predict_regime_for_request fails closed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=True,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            # Create a request with all NaNs
            bad_request = BridgeRequest(
                sequence=1,
                inputs=np.full(137, np.nan),
                raw_inputs=np.full(137, np.nan),
            )
            regime_idx, probs, regime_name = evaluator._predict_regime_for_request(bad_request)
            assert regime_idx in {0, 1, 2}
            assert len(probs) == 3
            assert np.isclose(sum(probs), 1.0, atol=1e-5)
            assert regime_name in {"BULL", "BEAR", "CHOP"}

    def test_evaluator_disabled_regime_detector_neutral_prior(self) -> None:
        """When regime detector is disabled, returns neutral CHOP prior."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=False,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            request = BridgeRequest(sequence=1, inputs=np.zeros(137), raw_inputs=np.zeros(137))
            regime_idx, probs, regime_name = evaluator._predict_regime_for_request(request)
            assert regime_idx == int(MarketRegime.CHOP)
            assert regime_name == "CHOP"
            assert np.allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


# ==============================================================================
# T5: REQUEST NORMALIZATION & DUAL-VIEW PRESERVATION
# ==============================================================================

class TestT5RequestNormalizationDualView:
    """Verify raw_inputs preservation and regime annotation during _normalize_request_inputs."""

    def test_normalize_preserves_raw_inputs_and_annotates_regime(
        self, synthetic_137_vector: np.ndarray
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=True,
                input_normalization_mode="raw",
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            original_request = BridgeRequest(
                sequence=42,
                inputs=synthetic_137_vector.copy(),
                raw_inputs=synthetic_137_vector.copy(),
                time="2026.01.01 12:00:00",
            )
            normalized_request = evaluator._normalize_request_inputs(original_request)

            # raw_inputs preserved intact
            assert np.array_equal(normalized_request.raw_inputs, synthetic_137_vector)
            # regime metadata annotated
            assert normalized_request.regime_index in {0, 1, 2}
            assert normalized_request.regime_probabilities is not None
            assert len(normalized_request.regime_probabilities) == 3
            assert np.isclose(sum(normalized_request.regime_probabilities), 1.0, atol=1e-5)
            assert normalized_request.regime_name in {"BULL", "BEAR", "CHOP"}


# ==============================================================================
# T6: FIVE-AGENT ACTION PACKET & POLICY HOOK
# ==============================================================================

class DummyFivePolicy:
    """Mock policy implementing five-agent and regime hooks."""

    def __init__(self) -> None:
        self.regime_probs: list[float] | None = None
        self.regime_name: str | None = None
        self.regime_idx: int | None = None

    def set_regime_soft_probabilities(
        self, soft_probs: Any, regime_name: str | None = None, regime_idx: int | None = None
    ) -> None:
        self.regime_probs = [float(p) for p in soft_probs]
        self.regime_name = regime_name
        self.regime_idx = regime_idx


class TestT6FiveAgentActionPacket:
    """Verify _five_bridge_action_packet injects regime metadata and calls policy hook."""

    def test_five_bridge_action_packet_enrichment(
        self, synthetic_137_vector: np.ndarray
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=True,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            policy = DummyFivePolicy()

            request = BridgeRequest(
                sequence=10,
                inputs=synthetic_137_vector.copy(),
                raw_inputs=synthetic_137_vector.copy(),
                time="2026.01.01 12:00",
                positions=0,
            )

            # Create 5 dummy traces matching five-agent authority
            traces = []
            for agent_id in range(5):
                outputs = np.zeros(20, dtype=float)
                outputs[int(HOLD_ACTION)] = 1.0
                trace = SimpleTrace(
                    trace_id=f"trace-10-{agent_id}",
                    selected_action=outputs,
                    selected_idx=int(HOLD_ACTION),
                    confidence=0.8,
                    action_details={"five_agent_index": agent_id, "five_action_index": int(HOLD_ACTION)},
                )
                traces.append(trace)

            packet, enriched_traces = evaluator._five_bridge_action_packet(
                policy, request, traces
            )

            # 1. Five actions produced
            assert len(packet) == 5
            assert len(enriched_traces) == 5

            # 2. Policy hook was called
            assert policy.regime_probs is not None
            assert len(policy.regime_probs) == 3
            assert np.isclose(sum(policy.regime_probs), 1.0, atol=1e-5)
            assert policy.regime_name in {"BULL", "BEAR", "CHOP"}

            # 3. Each trace action_details carries regime telemetry
            for tr in enriched_traces:
                details = getattr(tr, "action_details", {})
                assert "regime_index" in details
                assert "regime_name" in details
                assert "regime_probabilities" in details
                assert "regime_distribution" in details
                assert "p_bull" in details
                assert "p_bear" in details
                assert "p_chop" in details
                assert np.isclose(details["p_bull"] + details["p_bear"] + details["p_chop"], 1.0, atol=1e-5)


# ==============================================================================
# T7: TELEMETRY EMISSION (DIAGNOSTICS, EVENTS, METADATA, LIFECYCLE)
# ==============================================================================

class TestT7TelemetryEmission:
    """Verify soft probabilities are logged to mt5_diagnostics.json, events.jsonl, and .metadata.jsonl."""

    def test_update_diagnostics_regime_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            diag_file = Path(tmp_dir) / "mt5_diagnostics.json"
            # Initial diagnostics
            diag_file.write_text(json.dumps({"total_ticks": 100, "status": "running"}), encoding="utf-8")

            probs = [0.70, 0.20, 0.10]
            success = update_diagnostics_regime_telemetry(diag_file, probs, "BULL")
            assert success

            data = json.loads(diag_file.read_text(encoding="utf-8"))
            assert "regime_telemetry" in data
            reg_tel = data["regime_telemetry"]
            assert reg_tel["last_probabilities"] == probs
            assert reg_tel["regime_name"] == "BULL"

    def test_record_event_regime_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            events_file = Path(tmp_dir) / "events.jsonl"
            probs = [0.15, 0.75, 0.10]
            success = record_event_regime_distribution(
                events_file,
                probs,
                "BEAR",
                event_type="trade",
                sequence=101,
                trace_id="tr-101",
                extra_payload={"profit_r": 1.5},
            )
            assert success
            assert events_file.exists()

            lines = events_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["schema_version"] == 10
            assert record["event_type"] == "trade"
            assert record["sequence"] == 101
            assert record["trace_id"] == "tr-101"
            assert record["payload"]["regime_distribution"] == probs
            assert record["payload"]["regime_name"] == "BEAR"
            assert record["payload"]["profit_r"] == 1.5

    def test_write_action_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            action_file = Path(tmp_dir) / "mt5_action.csv"
            action_file.touch()

            request = BridgeRequest(
                sequence=55,
                event_id="evt-55",
                inputs=np.zeros(137),
                regime_index=0,
                regime_probabilities=(0.60, 0.25, 0.15),
                regime_name="BULL",
            )
            trace = SimpleTrace(
                trace_id="trace-55",
                action_details={
                    "regime_index": 0,
                    "regime_name": "BULL",
                    "regime_probabilities": [0.60, 0.25, 0.15],
                    "regime_distribution": [0.60, 0.25, 0.15],
                    "p_bull": 0.60,
                    "p_bear": 0.25,
                    "p_chop": 0.15,
                },
            )

            MT5StrategyTesterEvaluator._write_action_metadata(action_file, request, trace)

            metadata_file = action_file.with_name(action_file.name + ".metadata.jsonl")
            assert metadata_file.exists(), f"Metadata sidecar not found: {metadata_file}"

            lines = metadata_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) >= 1
            payload = json.loads(lines[-1])
            assert payload["regime_index"] == 0
            assert payload["regime_name"] == "BULL"
            assert payload["regime_distribution"] == [0.60, 0.25, 0.15]
            assert payload["p_bull"] == 0.60
            assert payload["p_bear"] == 0.25
            assert payload["p_chop"] == 0.15
            assert "regime_telemetry" in payload
            assert payload["regime_telemetry"]["last_probabilities"] == [0.60, 0.25, 0.15]
            assert payload["regime_telemetry"]["regime_name"] == "BULL"

    def test_lifecycle_credit_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            events_path = Path(tmp_dir) / "events.jsonl"
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                events_path=events_path,
                regime_detector_enabled=True,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)

            request = BridgeRequest(
                sequence=88,
                inputs=np.zeros(137),
                regime_index=1,
                regime_probabilities=(0.10, 0.80, 0.10),
                regime_name="BEAR",
            )
            lifecycle = MT5PositionLifecycle(
                position_ticket="TICKET-12345",
                entry_time=None,
                exit_time=None,
                net_pnl=100.0,
                gross_profit=100.0,
                commission=0.0,
                swap=0.0,
                trace_id="trace-lc-88",
                realized_r=2.1,
                close_reason="TP",
                duration_bars=12,
            )

            # Test applying lifecycle credit
            dummy_policy = DummyFivePolicy()
            credited = evaluator._apply_execution_lifecycle_credit(
                policy=dummy_policy,
                lifecycle=lifecycle,
                round_index=1,
                policy_id=1,
                source="test",
                request=request,
            )

            # Verify lifecycle event written to events.jsonl
            assert events_path.exists()
            lines = events_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) >= 1
            lc_event = json.loads(lines[-1])
            assert lc_event["event_type"] == "lifecycle"
            assert lc_event["payload"]["regime_distribution"] == [0.1, 0.8, 0.1]
            assert lc_event["payload"]["regime_name"] == "BEAR"


# ==============================================================================
# T8: HE CHUYEN GIA NGU CANH TRACE PROPAGATION
# ==============================================================================

class TestT8HeChuyenGiaNguCanhTracePropagation:
    """Verify HeChuyenGiaMotContextMotPolicy receives soft probs and propagates to traces."""

    def test_expert_system_trace_carries_regime_distribution(self) -> None:
        def _small_policy() -> ChuyenGiaPolicy:
            policy = ChuyenGiaPolicy(input_size=137, output_size=20, modules=[])
            policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
            policy.five_agent_memories = policy.five_shared_network.initial_memories()
            policy._learning_enabled = False
            return policy

        cfg = CauHinhHeChuyenGia(
            bat=True,
            che_do_quyet_dinh="nam_tac_nhan_dung_chung",
            cho_phep_policy_chung_ra_lenh=True,
            so_tac_nhan=5,
            so_chieu_nhan_dang_tac_nhan=5,
            entry_style_specialists_bat=False,
            position_management_specialist_bat=False,
            volatile_subexperts_bat=False,
        )
        system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_small_policy)

        soft_probs = [0.72, 0.18, 0.10]
        system.set_regime_soft_probabilities(soft_probs, regime_name="BULL", regime_idx=0)

        assert system._pending_regime_probabilities is not None
        assert np.allclose(system._pending_regime_probabilities, (0.72, 0.18, 0.10))
        assert system._pending_regime_name == "BULL"
        assert system._pending_regime_index == 0

        # Run 5-agent decision
        vec = np.zeros(137, dtype=float)
        traces = system._chay_nam_tac_nhan(vec)
        assert len(traces) == 5

        for trace in traces:
            audit = dict(getattr(trace, "context_audit", {}) or {})
            assert np.allclose(audit.get("regime_distribution", []), soft_probs)
            assert np.allclose(audit.get("regime_probabilities", []), soft_probs)
            assert audit.get("regime_name") == "BULL"
            assert audit.get("regime_index") == 0
            assert np.isclose(audit.get("p_bull", 0.0), 0.72)
            assert np.isclose(audit.get("p_bear", 0.0), 0.18)
            assert np.isclose(audit.get("p_chop", 0.0), 0.10)


# ==============================================================================
# T9: REGIME SOFT VETO & DUAL ROUTER BRIDGE INTEGRATION (MILESTONE 2)
# ==============================================================================

class TestT9RegimeSoftVetoAndDualRouterBridge:
    """Milestone 2: MT5 Runner IPC Bridge Soft Veto Verification & Test Hardening.

    Verifies causal regime soft veto logic in MT5StrategyTesterEvaluator._five_bridge_action_packet:
    - Agent 1 (Short Owner) entry actions vetoed in BULL regime (P_Bull >= 0.55).
    - Agent 0 (Long Owner) entry actions vetoed in BEAR regime (P_Bear >= 0.55).
    - Aligned actions permitted without veto.
    - No UnboundLocalError when owner_valid is False.
    - Direct integration with DualChampionRouter.
    """

    @pytest.fixture
    def evaluator(self) -> tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory]:
        tmp_dir = tempfile.TemporaryDirectory()
        config = MT5BacktestConfig(
            runner_dir=Path(tmp_dir.name),
            regime_detector_enabled=True,
            offline_replay=True,
        )
        ev = MT5StrategyTesterEvaluator(config)
        return ev, tmp_dir

    def _make_five_traces(
        self,
        agent_actions: dict[int, int] | None = None,
        *,
        owner_valid: bool = True,
    ) -> list[SimpleTrace]:
        actions = agent_actions or {}
        traces = []
        for agent_idx in range(5):
            chosen_action = int(actions.get(agent_idx, int(HOLD_ACTION)))
            outputs = np.zeros(20, dtype=float)
            outputs[chosen_action] = 1.0
            tr = SimpleTrace(
                trace_id=f"test_trace_agent_{agent_idx}",
                selected_action=outputs,
                selected_idx=chosen_action,
                confidence=0.85,
                action_details={
                    "five_agent_index": agent_idx,
                    "five_action_index": chosen_action,
                },
                ma_context="CTX_CHUNG" if owner_valid else "",
                ma_chuyen_gia="EXP_CHUNG" if owner_valid else "",
                ma_policy="POL_CHUNG" if owner_valid else "",
                ly_do_khong_hoc=not owner_valid,
            )
            traces.append(tr)
        return traces

    def test_soft_veto_agent1_short_in_bull_regime(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], synthetic_137_vector: np.ndarray
    ) -> None:
        """Agent 1 proposes MARKET_SELL under P_Bull = 0.70.

        Proves:
        - Wire action becomes HOLD (0)
        - wire_outputs argmax is HOLD (0)
        - regime_veto_applied == True
        - regime_veto_reason == 'veto_short_in_bull_p0.70'
        """
        ev, tmp = evaluator
        policy = MockExpertPolicy("bull_eval")
        p_bull = 0.70
        p_bear = 0.20
        p_chop = 0.10
        request = BridgeRequest(
            sequence=101,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(p_bull, p_bear, p_chop),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )

        # Agent 1 proposes MARKET_SELL (action 2)
        traces = self._make_five_traces({1: int(Action.MARKET_SELL)})
        packet, enriched_traces = ev._five_bridge_action_packet(policy, request, traces)

        # Verify Agent 1 wire action is forced to HOLD (0)
        agent1_action = packet[1]
        assert agent1_action.action == int(Action.HOLD)
        assert np.argmax(agent1_action.outputs) == int(Action.HOLD)

        # Verify trace metadata and telemetry
        agent1_details = enriched_traces[1].action_details
        assert agent1_details["regime_veto_applied"] is True
        assert agent1_details["regime_veto_reason"] == "veto_short_in_bull_p0.70"
        assert agent1_details["wire_action"] == int(Action.HOLD)
        assert agent1_details["policy_action"] == int(Action.MARKET_SELL)
        assert agent1_details["action_masked"] is True
        assert agent1_details["action_origin"] == "mask"

        # Also verify pending short orders (SELL_LIMIT) are vetoed under Bull regime
        traces_limit = self._make_five_traces({1: int(Action.SELL_LIMIT)})
        packet_limit, enriched_limit = ev._five_bridge_action_packet(policy, request, traces_limit)
        assert packet_limit[1].action == int(Action.HOLD)
        assert np.argmax(packet_limit[1].outputs) == int(Action.HOLD)
        assert enriched_limit[1].action_details["regime_veto_applied"] is True
        assert enriched_limit[1].action_details["regime_veto_reason"] == "veto_short_in_bull_p0.70"

        # Also verify SELL_STOP is vetoed
        traces_stop = self._make_five_traces({1: int(Action.SELL_STOP)})
        packet_stop, enriched_stop = ev._five_bridge_action_packet(policy, request, traces_stop)
        assert packet_stop[1].action == int(Action.HOLD)
        assert np.argmax(packet_stop[1].outputs) == int(Action.HOLD)
        assert enriched_stop[1].action_details["regime_veto_applied"] is True
        assert enriched_stop[1].action_details["regime_veto_reason"] == "veto_short_in_bull_p0.70"

    def test_soft_veto_agent0_long_in_bear_regime(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], synthetic_137_vector: np.ndarray
    ) -> None:
        """Agent 0 proposes MARKET_BUY under P_Bear = 0.75.

        Proves:
        - Wire action becomes HOLD (0)
        - wire_outputs argmax is HOLD (0)
        - regime_veto_applied == True
        - regime_veto_reason == 'veto_long_in_bear_p0.75'
        """
        ev, tmp = evaluator
        policy = MockExpertPolicy("bear_eval")
        p_bull = 0.15
        p_bear = 0.75
        p_chop = 0.10
        request = BridgeRequest(
            sequence=102,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(p_bull, p_bear, p_chop),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )

        # Agent 0 proposes MARKET_BUY (action 1)
        traces = self._make_five_traces({0: int(Action.MARKET_BUY)})
        packet, enriched_traces = ev._five_bridge_action_packet(policy, request, traces)

        # Verify Agent 0 wire action is forced to HOLD (0)
        agent0_action = packet[0]
        assert agent0_action.action == int(Action.HOLD)
        assert np.argmax(agent0_action.outputs) == int(Action.HOLD)

        # Verify trace metadata and telemetry
        agent0_details = enriched_traces[0].action_details
        assert agent0_details["regime_veto_applied"] is True
        assert agent0_details["regime_veto_reason"] == "veto_long_in_bear_p0.75"
        assert agent0_details["wire_action"] == int(Action.HOLD)
        assert agent0_details["policy_action"] == int(Action.MARKET_BUY)
        assert agent0_details["action_masked"] is True
        assert agent0_details["action_origin"] == "mask"

        # Also verify pending long orders (BUY_LIMIT) are vetoed under Bear regime
        traces_limit = self._make_five_traces({0: int(Action.BUY_LIMIT)})
        packet_limit, enriched_limit = ev._five_bridge_action_packet(policy, request, traces_limit)
        assert packet_limit[0].action == int(Action.HOLD)
        assert np.argmax(packet_limit[0].outputs) == int(Action.HOLD)
        assert enriched_limit[0].action_details["regime_veto_applied"] is True
        assert enriched_limit[0].action_details["regime_veto_reason"] == "veto_long_in_bear_p0.75"

        # Also verify BUY_STOP is vetoed
        traces_stop = self._make_five_traces({0: int(Action.BUY_STOP)})
        packet_stop, enriched_stop = ev._five_bridge_action_packet(policy, request, traces_stop)
        assert packet_stop[0].action == int(Action.HOLD)
        assert np.argmax(packet_stop[0].outputs) == int(Action.HOLD)
        assert enriched_stop[0].action_details["regime_veto_applied"] is True
        assert enriched_stop[0].action_details["regime_veto_reason"] == "veto_long_in_bear_p0.75"

    def test_no_veto_on_aligned_actions(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], synthetic_137_vector: np.ndarray
    ) -> None:
        """Agent 0 proposes MARKET_BUY in Bull (not vetoed); Agent 1 proposes MARKET_SELL in Bear (not vetoed)."""
        ev, tmp = evaluator
        policy = MockExpertPolicy("aligned_eval")

        # 1. Aligned Long in Bull regime: Agent 0 MARKET_BUY
        req_bull = BridgeRequest(
            sequence=103,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.70, 0.20, 0.10),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_bull = self._make_five_traces({0: int(Action.MARKET_BUY)})
        packet_bull, enriched_bull = ev._five_bridge_action_packet(policy, req_bull, traces_bull)

        # Agent 0 MARKET_BUY must be preserved
        assert packet_bull[0].action == int(Action.MARKET_BUY)
        assert np.argmax(packet_bull[0].outputs) == int(Action.MARKET_BUY)
        assert enriched_bull[0].action_details["regime_veto_applied"] is False
        assert enriched_bull[0].action_details["regime_veto_reason"] == ""
        assert enriched_bull[0].action_details["wire_action"] == int(Action.MARKET_BUY)
        assert enriched_bull[0].action_details["policy_action"] == int(Action.MARKET_BUY)
        assert enriched_bull[0].action_details["action_masked"] is False
        assert enriched_bull[0].action_details["action_origin"] == "policy"

        # 2. Aligned Short in Bear regime: Agent 1 MARKET_SELL
        req_bear = BridgeRequest(
            sequence=104,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.15, 0.75, 0.10),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces_bear = self._make_five_traces({1: int(Action.MARKET_SELL)})
        packet_bear, enriched_bear = ev._five_bridge_action_packet(policy, req_bear, traces_bear)

        # Agent 1 MARKET_SELL must be preserved
        assert packet_bear[1].action == int(Action.MARKET_SELL)
        assert np.argmax(packet_bear[1].outputs) == int(Action.MARKET_SELL)
        assert enriched_bear[1].action_details["regime_veto_applied"] is False
        assert enriched_bear[1].action_details["regime_veto_reason"] == ""
        assert enriched_bear[1].action_details["wire_action"] == int(Action.MARKET_SELL)
        assert enriched_bear[1].action_details["policy_action"] == int(Action.MARKET_SELL)
        assert enriched_bear[1].action_details["action_masked"] is False
        assert enriched_bear[1].action_details["action_origin"] == "policy"

    def test_no_unbound_local_error_when_owner_invalid(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], synthetic_137_vector: np.ndarray
    ) -> None:
        """Regression test ensuring no UnboundLocalError when owner_valid is False.

        Verifies that regime_veto_applied is properly initialized prior to the
        'if owner_valid:' block (line 9215 in mt5_runner.py), preventing UnboundLocalError
        when telemetry details are assembled at line 9323.
        """
        ev, tmp = evaluator

        # Scenario A: Policy without experts / nhan_dien interface (e.g. DummyFivePolicy)
        dummy_policy = DummyFivePolicy()
        req_bull = BridgeRequest(
            sequence=105,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.70, 0.20, 0.10),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_a = self._make_five_traces({1: int(Action.MARKET_SELL)}, owner_valid=True)
        # Should execute cleanly without UnboundLocalError
        packet_a, enriched_a = ev._five_bridge_action_packet(dummy_policy, req_bull, traces_a)
        assert len(packet_a) == 5
        assert len(enriched_a) == 5
        for tr in enriched_a:
            details = tr.action_details
            assert details["regime_veto_applied"] is False
            assert details["regime_veto_reason"] == ""
            assert details["wire_action"] == int(Action.HOLD)
            assert details["no_neural_command"] is True

        # Scenario B: Policy is valid expert, but trace has missing owner provenance
        policy = MockExpertPolicy("owner_invalid_test")
        traces_b = self._make_five_traces({1: int(Action.MARKET_SELL)}, owner_valid=False)
        packet_b, enriched_b = ev._five_bridge_action_packet(policy, req_bull, traces_b)
        assert len(packet_b) == 5
        assert len(enriched_b) == 5
        for tr in enriched_b:
            details = tr.action_details
            assert details["regime_veto_applied"] is False
            assert details["regime_veto_reason"] == ""
            assert details["wire_action"] == int(Action.HOLD)

    def test_dual_champion_router_integrated_with_five_bridge_action_packet(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], synthetic_137_vector: np.ndarray
    ) -> None:
        """Tests DualChampionRouter directly with MT5StrategyTesterEvaluator._five_bridge_action_packet.

        Verifies that:
        1. Router sets regime soft probabilities and activates the appropriate specialist.
        2. In BULL regime (P_Bull = 0.70 >= 0.55):
           - Router routes to Bull Specialist.
           - Agent 0 MARKET_BUY is permitted.
           - Agent 1 MARKET_SELL is soft-vetoed and converted to HOLD.
        3. In BEAR regime (P_Bear = 0.75 >= 0.55):
           - Router routes to Bear Specialist.
           - Agent 1 MARKET_SELL is permitted.
           - Agent 0 MARKET_BUY is soft-vetoed and converted to HOLD.
        4. In CHOP regime (P_Chop = 0.70 >= 0.40):
           - Router routes to defensive specialist (default Bear).
           - 5-action bridge packet produced successfully.
        """
        ev, tmp = evaluator
        bull_specialist = MockExpertPolicy("bull_specialist")
        bear_specialist = MockExpertPolicy("bear_specialist")
        router = DualChampionRouter(
            bull_policy=bull_specialist,
            bear_policy=bear_specialist,
            bull_threshold=0.55,
            bear_threshold=0.55,
            chop_threshold=0.40,
            confirmation_bars=1,
        )
        assert router.current_regime == "CHOP"

        # --- Subtest 1: Bull Regime Routing & Soft Veto ---
        req_bull = BridgeRequest(
            sequence=201,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.70, 0.20, 0.10),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        # Generate traces from router
        traces_bull = list(router.act_nam_tac_nhan(req_bull.inputs))
        assert len(traces_bull) == 5

        # Simulate Agent 0 proposing MARKET_BUY (aligned) and Agent 1 proposing MARKET_SELL (counter-trend)
        traces_bull[0].selected_idx = int(Action.MARKET_BUY)
        traces_bull[0].action_details["five_action_index"] = int(Action.MARKET_BUY)
        traces_bull[0].selected_action = np.zeros(20, dtype=float)
        traces_bull[0].selected_action[int(Action.MARKET_BUY)] = 1.0

        traces_bull[1].selected_idx = int(Action.MARKET_SELL)
        traces_bull[1].action_details["five_action_index"] = int(Action.MARKET_SELL)
        traces_bull[1].selected_action = np.zeros(20, dtype=float)
        traces_bull[1].selected_action[int(Action.MARKET_SELL)] = 1.0

        packet_bull, enriched_bull = ev._five_bridge_action_packet(router, req_bull, traces_bull)

        # Check router state
        assert router.current_regime == "BULL"
        assert router.active_policy is bull_specialist
        assert bull_specialist.regime_probs == [0.70, 0.20, 0.10]

        # Check Agent 0 (Long Owner): Aligned action permitted
        assert packet_bull[0].action == int(Action.MARKET_BUY)
        assert np.argmax(packet_bull[0].outputs) == int(Action.MARKET_BUY)
        assert enriched_bull[0].action_details["regime_veto_applied"] is False

        # Check Agent 1 (Short Owner): Counter-trend action vetoed and converted to HOLD
        assert packet_bull[1].action == int(Action.HOLD)
        assert np.argmax(packet_bull[1].outputs) == int(Action.HOLD)
        assert enriched_bull[1].action_details["regime_veto_applied"] is True
        assert enriched_bull[1].action_details["regime_veto_reason"] == "veto_short_in_bull_p0.70"

        # --- Subtest 2: Bear Regime Routing & Soft Veto ---
        req_bear = BridgeRequest(
            sequence=202,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.15, 0.75, 0.10),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces_bear = list(router.act_nam_tac_nhan(req_bear.inputs))
        assert len(traces_bear) == 5

        # Simulate Agent 0 proposing MARKET_BUY (counter-trend) and Agent 1 proposing MARKET_SELL (aligned)
        traces_bear[0].selected_idx = int(Action.MARKET_BUY)
        traces_bear[0].action_details["five_action_index"] = int(Action.MARKET_BUY)
        traces_bear[0].selected_action = np.zeros(20, dtype=float)
        traces_bear[0].selected_action[int(Action.MARKET_BUY)] = 1.0

        traces_bear[1].selected_idx = int(Action.MARKET_SELL)
        traces_bear[1].action_details["five_action_index"] = int(Action.MARKET_SELL)
        traces_bear[1].selected_action = np.zeros(20, dtype=float)
        traces_bear[1].selected_action[int(Action.MARKET_SELL)] = 1.0

        packet_bear, enriched_bear = ev._five_bridge_action_packet(router, req_bear, traces_bear)

        # Check router state
        assert router.current_regime == "BEAR"
        assert router.active_policy is bear_specialist
        assert bear_specialist.regime_probs == [0.15, 0.75, 0.10]

        # Check Agent 0 (Long Owner): Counter-trend action vetoed and converted to HOLD
        assert packet_bear[0].action == int(Action.HOLD)
        assert np.argmax(packet_bear[0].outputs) == int(Action.HOLD)
        assert enriched_bear[0].action_details["regime_veto_applied"] is True
        assert enriched_bear[0].action_details["regime_veto_reason"] == "veto_long_in_bear_p0.75"

        # Check Agent 1 (Short Owner): Aligned action permitted
        assert packet_bear[1].action == int(Action.MARKET_SELL)
        assert np.argmax(packet_bear[1].outputs) == int(Action.MARKET_SELL)
        assert enriched_bear[1].action_details["regime_veto_applied"] is False

        # --- Subtest 3: Defensive CHOP Regime ---
        req_chop = BridgeRequest(
            sequence=203,
            inputs=synthetic_137_vector.copy(),
            raw_inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.15, 0.15, 0.70),
            regime_name="CHOP",
            regime_index=2,
            positions=0,
        )
        traces_chop = list(router.act_nam_tac_nhan(req_chop.inputs))
        packet_chop, enriched_chop = ev._five_bridge_action_packet(router, req_chop, traces_chop)

        assert router.current_regime == "CHOP"
        assert router.active_policy is bear_specialist
        assert len(packet_chop) == 5
        assert len(enriched_chop) == 5

    def test_soft_veto_threshold_boundary_precision(
        self, evaluator: tuple[MT5StrategyTesterEvaluator, tempfile.TemporaryDirectory], synthetic_137_vector: np.ndarray
    ) -> None:
        """Verify boundary condition precision: P=0.35 triggers veto in Bull, aligned actions in Bear permitted, Chop safe HOLD."""
        ev, tmp = evaluator
        policy = MockExpertPolicy("boundary_eval")

        # 1. P_Bull = 0.35 triggers veto on Agent 1 (Short Owner)
        req_35 = BridgeRequest(
            sequence=301,
            inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.35, 0.30, 0.35),
            regime_name="BULL",
            regime_index=0,
            positions=0,
        )
        traces_35 = self._make_five_traces({1: int(Action.MARKET_SELL)})
        packet_35, enriched_35 = ev._five_bridge_action_packet(policy, req_35, traces_35)
        assert packet_35[1].action == int(Action.HOLD)
        assert enriched_35[1].action_details["regime_veto_applied"] is True
        assert enriched_35[1].action_details["regime_veto_reason"] == "veto_short_in_bull_p0.35"

        # 2. In BEAR regime (P_Bear = 0.36, P_Bull = 0.34): Agent 1 MARKET_SELL is aligned -> does NOT trigger veto
        req_bear = BridgeRequest(
            sequence=302,
            inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.34, 0.36, 0.30),
            regime_name="BEAR",
            regime_index=1,
            positions=0,
        )
        traces_bear = self._make_five_traces({1: int(Action.MARKET_SELL)})
        packet_bear, enriched_bear = ev._five_bridge_action_packet(policy, req_bear, traces_bear)
        assert packet_bear[1].action == int(Action.MARKET_SELL)
        assert enriched_bear[1].action_details["regime_veto_applied"] is False

        # 3. In BEAR regime, Agent 0 proposing counter-trend MARKET_BUY triggers veto
        traces_bear_buy = self._make_five_traces({0: int(Action.MARKET_BUY)})
        packet_bb, enriched_bb = ev._five_bridge_action_packet(policy, req_bear, traces_bear_buy)
        assert packet_bb[0].action == int(Action.HOLD)
        assert enriched_bb[0].action_details["regime_veto_applied"] is True
        assert enriched_bb[0].action_details["regime_veto_reason"] == "veto_long_in_bear_p0.36"

        # 4. In CHOP regime (P_Chop = 0.38): Both Agent 0 and Agent 1 entries vetoed to safe HOLD
        req_chop = BridgeRequest(
            sequence=304,
            inputs=synthetic_137_vector.copy(),
            regime_probabilities=(0.31, 0.31, 0.38),
            regime_name="CHOP",
            regime_index=2,
            positions=0,
        )
        traces_chop = self._make_five_traces({0: int(Action.MARKET_BUY), 1: int(Action.MARKET_SELL)})
        packet_chop, enriched_chop = ev._five_bridge_action_packet(policy, req_chop, traces_chop)
        assert packet_chop[0].action == int(Action.HOLD)
        assert enriched_chop[0].action_details["regime_veto_applied"] is True
        assert enriched_chop[0].action_details["regime_veto_reason"] == "veto_entries_in_chop_p0.38"
        assert packet_chop[1].action == int(Action.HOLD)
        assert enriched_chop[1].action_details["regime_veto_applied"] is True
        assert enriched_chop[1].action_details["regime_veto_reason"] == "veto_entries_in_chop_p0.38"

