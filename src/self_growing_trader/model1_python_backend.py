from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import io
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .chuyen_gia_policy import (
    _compact_context_expert_artifact_state,
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _mt5_execution_contract_kwargs,
    _reset_context_expert_runtime_for_new_mt5_session,
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from .config import AppConfig
from .model1_replay import Model1OracleTape, model1_training_state_hash
from .mt5_runner import BridgeRequest, ExperienceStarvationState, MT5BacktestConfig, MT5StrategyTesterEvaluator
from .learning_integrity import policy_state_hash


@dataclass(frozen=True)
class Model1PythonDecision:
    sequence: int
    event_id: str
    raw_inputs: np.ndarray
    normalized_inputs: np.ndarray
    outputs: np.ndarray
    action: int


@dataclass(frozen=True)
class Model1TapeTrainingResult:
    served: int
    decisions: tuple[Model1PythonDecision, ...]
    policy_state_sha256: str


class _NeverExitProcess:
    def poll(self) -> None:
        return None


class Model1ReplayRequestStream:
    """Yield tape requests while exposing only causally available ledger rows."""

    def __init__(self, tape: Model1OracleTape, execution_target: Path):
        self.tape = tape
        self.execution_target = execution_target
        self.index = 0
        self.header = tape.execution_lines[0] if tape.execution_lines else ""
        self.rows = tuple(tape.execution_lines[1:]) if tape.execution_lines else ()
        self._parsed_rows = self._parse_rows()

    def _parse_rows(self) -> tuple[tuple[str, dict[str, str]], ...]:
        if not self.header:
            return ()
        parsed: list[tuple[str, dict[str, str]]] = []
        for line in self.rows:
            reader = csv.DictReader(io.StringIO(self.header + "\n" + line + "\n"), delimiter="\t")
            parsed.append((line, next(reader)))
        return tuple(parsed)

    @staticmethod
    def _causal_row(row: dict[str, str], cutoff_epoch: int) -> bool:
        try:
            logged = datetime.strptime(
                str(row.get("logged_at", "")), "%Y.%m.%d %H:%M:%S"
            )
        except ValueError as exc:
            raise ValueError("model-1 execution row has invalid logged_at") from exc
        return int(logged.replace(tzinfo=timezone.utc).timestamp()) <= int(cutoff_epoch)

    def _write_ledger(self, cutoff_epoch: int | None) -> None:
        self.execution_target.parent.mkdir(parents=True, exist_ok=True)
        if not self.header:
            self.execution_target.write_text("", encoding="utf-8")
            return
        rows = self.rows if cutoff_epoch is None else tuple(
            line for line, row in self._parsed_rows if self._causal_row(row, cutoff_epoch)
        )
        content = self.header + "\n" + ("\n".join(rows) + "\n" if rows else "")
        self.execution_target.write_text(content, encoding="utf-8")

    def __iter__(self) -> "Model1ReplayRequestStream":
        return self

    def __next__(self) -> BridgeRequest:
        if self.index >= len(self.tape):
            self._write_ledger(None)
            raise StopIteration
        sequence = self.index + 1
        raw = self.tape.inputs[self.index]
        metadata = self.tape.metadata[self.index]
        try:
            cutoff_epoch = int(str(metadata["event_id"]).split("-", 1)[0])
        except (KeyError, ValueError) as exc:
            raise ValueError("model-1 event_id must start with tester epoch") from exc
        self._write_ledger(cutoff_epoch)
        positions_pipe = str(metadata.get("positions_pipe", ""))
        request = BridgeRequest(
            sequence=sequence, inputs=raw.copy(), raw_inputs=raw.copy(),
            schema_version=10, event_id=str(metadata["event_id"]),
            time=str(metadata.get("logged_at", "")),
            balance=float(metadata.get("balance", 0.0)),
            equity=float(metadata.get("equity", 0.0)),
            margin=float(metadata.get("margin", 0.0)),
            free_margin=float(metadata.get("free_margin", 0.0)),
            margin_level_pct=float(metadata.get("margin_level_pct", 0.0)),
            leverage=int(metadata.get("leverage", 0)),
            drawdown_pct=float(metadata.get("drawdown_pct", 0.0)),
            spread_points=float(metadata.get("spread_points", 0.0)),
            positions=sum(1 for item in positions_pipe.split("^") if item),
            day_trades=int(metadata.get("day_trades", 0)),
            positions_pipe=positions_pipe,
            orders_pipe=str(metadata.get("orders_pipe", "")),
            execution_pipe=str(metadata.get("execution_pipe", "")),
        )
        self.index += 1
        return request


class Model1PythonBackend:
    """CSV-free model-1 policy boundary using the production MT5 decision path."""

    def __init__(
        self,
        config: AppConfig,
        seed_path: str | Path,
        *,
        runtime_dir: str | Path | None = None,
    ):
        self.config = config
        self.policy = tai_he_chuyen_gia_mot_context_mot_policy(
            seed_path, config, seed=int(config.learning.seed)
        )
        _cau_hinh_policy_chuyen_gia_cho_mt5(self.policy, config)
        _reset_context_expert_runtime_for_new_mt5_session(self.policy)
        selected_runtime = Path(runtime_dir) if runtime_dir is not None else (
            Path(config.environment.model1_tape_path).parent / ".python_model1_runtime"
            if str(config.environment.model1_tape_path).strip()
            else Path("work/model1_parity/.python_model1_runtime")
        )
        self.evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
            runner_dir=selected_runtime,
            offline_replay=True,
            learning_enabled=bool(config.mt5_learning_enabled),
            max_positions=int(config.risk.max_positions),
            max_holding_bars=int(config.orders.max_holding_bars),
            min_daily_trades=int(config.orders.min_daily_trades),
            max_daily_trades=int(config.orders.max_daily_trades),
            **_mt5_execution_contract_kwargs(config),
        ))
        self.starvation = ExperienceStarvationState()
        self.equity_history: list[float] = []
        self.previous_request: BridgeRequest | None = None
        self.previous_action: int = -1

    def decide(self, *, sequence: int, raw_inputs: np.ndarray, metadata: dict[str, Any]) -> Model1PythonDecision:
        positions_pipe = str(metadata.get("positions_pipe", ""))
        request = BridgeRequest(
            sequence=int(sequence), inputs=np.asarray(raw_inputs, dtype=float).copy(),
            raw_inputs=np.asarray(raw_inputs, dtype=float).copy(), schema_version=10,
            event_id=str(metadata["event_id"]), time=str(metadata.get("logged_at", "")),
            balance=float(metadata.get("balance", 0.0)), equity=float(metadata.get("equity", 0.0)),
            drawdown_pct=float(metadata.get("drawdown_pct", 0.0)),
            spread_points=float(metadata.get("spread_points", 0.0)),
            positions=sum(1 for item in positions_pipe.split("^") if item),
            day_trades=int(metadata.get("day_trades", 0)), positions_pipe=positions_pipe,
            orders_pipe=str(metadata.get("orders_pipe", "")), execution_pipe=str(metadata.get("execution_pipe", "")),
        )
        request = self.evaluator._normalize_request_inputs(request)
        if self.previous_request is not None:
            details = self.evaluator._execution_details(
                self.previous_request, request, self.previous_action
            )
            if not bool(details.get("flat_idle", False)):
                self.starvation.confirm_execution(
                    self.previous_request, request, emitted_action=self.previous_action
                )
        allowed = self.evaluator._allowed_actions_for_request(request)
        account = self.evaluator._account_context_for_request(request, self.equity_history)
        goal = self.evaluator._goal_context_for_request(request)
        execution = self.evaluator._execution_context_for_request(request, allowed)
        starvation_preview = self.starvation.preview(
            request, self.evaluator.config.min_daily_trades,
            suppress_flat_idle_feedback=bool(self.evaluator.config.suppress_flat_idle_starvation),
        )
        self.policy.set_experience_starvation(starvation_preview)
        self.policy.set_account_context(account, positions_pipe)
        self.policy.set_regime_raw_inputs(request.raw_inputs)
        self.policy.set_decision_context(
            account_context=account, positions_pipe=positions_pipe,
            allowed_actions={int(item): True for item in allowed}, execution_context=execution,
            goal=goal, sequence_id=int(sequence), frame_id=f"mt5:0:{int(sequence)}",
            timestamp=0.0, defer_live_decision_until_bridge=True,
        )
        trace = self.policy.act(request.inputs, goal=goal)
        apply_bias = getattr(self.policy, "apply_entry_timing_extrema_bias", None)
        if callable(apply_bias):
            trace = apply_bias(trace)
        outputs = np.asarray(trace.selected_action, dtype=float)
        effective = self.evaluator._effective_context_actions(trace, allowed)
        policy_action, _ = self.evaluator._choose_action(outputs)
        legal_action, _ = self.evaluator._choose_action(outputs, effective)
        trace_details = dict(getattr(trace, "action_details", {}) or {})
        policy_vetoed = bool(
            trace_details.get("learner_recovery_negative_veto_applied", False)
            or trace_details.get("learner_recovery_negative_veto_to_hold", False)
        )
        action, _ = self.evaluator._bootstrap_execution_action(
            request, legal_action, starvation_preview, effective,
            policy_vetoed=policy_vetoed,
            policy_vetoed_action=trace_details.get("learner_recovery_negative_veto_from"),
            probe_details={
                **dict(getattr(trace, "context_audit", {}) or {}),
                **trace_details,
            },
        )
        wire_action = 0 if int(action) < 0 else int(action)
        wire_outputs = np.asarray(outputs, dtype=float).copy()
        if int(action) >= 0 and int(action) != int(legal_action):
            wire_outputs = self.evaluator._force_action_output(wire_outputs, int(action))
        elif int(action) < 0:
            wire_outputs = self.evaluator._force_action_output(wire_outputs, 0)
        wire_outputs = self.evaluator._normalize_pending_offset(wire_outputs, wire_action)
        wire_outputs = self.evaluator._normalize_outputs_for_ea(wire_outputs)
        if bool(self.evaluator.config.strict_model1_parity):
            wire_outputs = np.round(
                wire_outputs, int(self.evaluator.config.output_wire_decimals)
            )
        finalize = getattr(self.policy, "finalize_bridge_decision", None)
        if callable(finalize):
            finalize(trace)
        if request.equity > 0.0:
            self.equity_history.append(float(request.equity))
            self.equity_history = self.equity_history[-64:]
        self.previous_request = request
        self.previous_action = int(action)
        return Model1PythonDecision(
            sequence=int(sequence), event_id=request.event_id,
            raw_inputs=np.asarray(request.raw_inputs, dtype=float).copy(),
            normalized_inputs=np.asarray(request.inputs, dtype=float).copy(),
            # Match _write_action_batch's ``.10g`` transport exactly, not a
            # local decimal rounding approximation.
            outputs=np.asarray([float(f"{float(value):.10g}") for value in wire_outputs]),
            action=int(wire_action),
        )

    def run_tape(self, tape: Model1OracleTape) -> tuple[Model1PythonDecision, ...]:
        return tuple(
            self.decide(sequence=index + 1, raw_inputs=tape.inputs[index], metadata=tape.metadata[index])
            for index in range(len(tape))
        )

    def train_tape(self, tape: Model1OracleTape) -> Model1TapeTrainingResult:
        """Train through the production MT5 feedback loop using an in-memory tape."""
        execution_target = self.evaluator.mql5_files / self.evaluator.config.execution_file
        stream = Model1ReplayRequestStream(tape, execution_target)
        captured: list[Model1PythonDecision] = []

        def observe(request: BridgeRequest, action_path: Path) -> None:
            with action_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if not rows:
                raise RuntimeError(f"model-1 replay action file is empty: {action_path}")
            row = rows[-1]
            captured.append(Model1PythonDecision(
                sequence=int(request.sequence), event_id=str(request.event_id),
                raw_inputs=np.asarray(request.raw_inputs, dtype=float).copy(),
                normalized_inputs=np.asarray(request.inputs, dtype=float).copy(),
                outputs=np.asarray([float(item) for item in row["outputs_pipe"].split("|")]),
                action=int(row["action"]),
            ))

        served = self.evaluator._serve_policy_actions(
            _NeverExitProcess(),
            self.evaluator.runner_dir / "Reports" / "python_model1_replay.htm",
            self.policy,
            1,
            0,
            replay_requests=stream,
            replay_action_observer=observe,
        )
        return Model1TapeTrainingResult(
            served=int(served), decisions=tuple(captured),
            policy_state_sha256=policy_state_hash(self.policy),
        )


def train_context_expert_python_model1(
    config: AppConfig,
    *,
    output_path: str | Path,
    seed_system_path: str | Path,
    runtime_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run the optional Python model-1 backend and persist one verified artifact."""
    if str(config.environment.backend) != "python_model1":
        raise ValueError("environment.backend must be python_model1")
    tape = Model1OracleTape(config.environment.model1_tape_path)
    backend = Model1PythonBackend(
        config, seed_system_path, runtime_dir=runtime_dir
    )
    result = backend.train_tape(tape)
    mismatches: list[dict[str, object]] = []
    for index, decision in enumerate(result.decisions):
        failing = np.flatnonzero(decision.outputs != tape.outputs[index])
        if failing.size or decision.action != int(tape.actions[index]):
            mismatches.append({
                "sequence": index + 1,
                "event_id": decision.event_id,
                "expected_action": int(tape.actions[index]),
                "actual_action": int(decision.action),
                "failing_output_indexes": failing.tolist(),
            })
    if mismatches:
        raise RuntimeError(
            f"python_model1 parity failed at {len(mismatches)} decisions: {mismatches[:3]}"
        )
    _compact_context_expert_artifact_state(
        backend.policy,
        drop_trace_archive=True,
        learning_settings=config.learning,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(backend.policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    with target.open("rb") as handle:
        persisted_policy = pickle.load(handle)
    report = {
        "mode": "python_model1",
        "policy": str(target),
        "seed_system": str(seed_system_path),
        "tape": str(Path(config.environment.model1_tape_path)),
        "tape_content_sha256": tape.content_sha256,
        "served": int(result.served),
        "decision_count": len(result.decisions),
        "input_value_count": int(tape.inputs.size),
        "output_value_count": int(tape.outputs.size),
        "execution_row_count": max(0, len(tape.execution_lines) - 1),
        "exact_decision_parity": True,
        "training_state_sha256": model1_training_state_hash(persisted_policy),
        "learning_enabled": bool(config.mt5_learning_enabled),
    }
    report_path = target.with_suffix(target.suffix + ".model1_report.json")
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
