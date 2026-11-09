"""Incremental, read-only diagnostics for a running MT5/Five candidate.

The monitor consumes the append-only execution/bridge ledgers and the Python
execution observatory.  It never writes to MT5 files or model state; its only
output is an atomically replaced ``mt5_diagnostics.json`` beside the candidate
artifacts so the desktop UI can render evidence without scraping console text.
"""
from __future__ import annotations

from collections import Counter, deque
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterable


DIAGNOSTICS_SCHEMA_VERSION = 3
_PUBLISH_RETRY_COUNT = 20
_PUBLISH_RETRY_DELAY_SECONDS = 0.05


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _nested_scalars(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(child, (dict, list)):
                yield str(key), child
            yield from _nested_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_scalars(child)


class _AppendReader:
    """Read only complete appended lines while a producer keeps the file open."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = b""

    def lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            # A fresh tester file replaced the old one.  Callers create a new
            # monitor per runner, so resetting here cannot mix different runs.
            self.offset = 0
            self.pending = b""
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return []
        parts = (self.pending + chunk).split(b"\n")
        self.pending = parts.pop()
        return [part.rstrip(b"\r").decode("utf-8-sig", errors="replace") for part in parts]


class _TsvReader:
    def __init__(self, path: Path) -> None:
        self.reader = _AppendReader(path)
        self.header: list[str] = []

    def rows(self) -> Iterable[dict[str, str]]:
        for line in self.reader.lines():
            if not line:
                continue
            values = line.split("\t")
            if not self.header:
                self.header = values
                continue
            if values and self.header and values[0] == self.header[0]:
                # A producer can truncate/recreate the ledger and write its
                # header again while the monitor stays alive.  Refresh the
                # column contract without treating that header as a data row.
                self.header = values
                continue
            if len(values) < len(self.header):
                continue
            yield dict(zip(self.header, values, strict=False))


def _latest_runner_file(runner: Path, name: str) -> Path:
    candidates = list(runner.glob(f"Tester/Agent-*/MQL5/Files/{name}"))
    if not candidates:
        return runner / "MQL5" / "Files" / name
    return max(candidates, key=lambda item: (item.stat().st_mtime_ns, item.stat().st_size))


class Mt5LiveDiagnostics:
    """Maintain one compact diagnostic state for one isolated runner."""

    def __init__(self, output: Path, runner: Path) -> None:
        self.output = output.resolve()
        self.runner = runner.resolve()
        self.execution = _TsvReader(_latest_runner_file(self.runner, "self_growing_execution.csv"))
        self.bridge = _TsvReader(_latest_runner_file(self.runner, "self_growing_bridge.csv"))
        observatories = sorted(self.output.glob("*.execution_trace_observatory.jsonl"))
        self.observatory = _AppendReader(
            observatories[0] if observatories else self.output / "execution_trace_observatory.jsonl"
        )
        codex_receipts = sorted(self.output.glob("*codex_teacher_receipts.jsonl"))
        self.codex_receipts = _AppendReader(
            codex_receipts[-1] if codex_receipts else self.output / "codex_teacher_receipts.jsonl"
        )
        self.codex_status_path: Path | None = None
        self.codex_status: dict[str, Any] = {}
        self._codex_receipt_ids: set[str] = set()
        self._codex_update_ids: set[str] = set()
        self._codex_submitted = 0
        self._codex_graded = 0
        self._codex_hypotheses = 0
        self._codex_hypotheses_confirmed = 0
        self._codex_updates = 0
        self._codex_interventions = 0
        self._codex_failures = 0
        self._codex_pending = 0
        self._codex_rejected_updates = 0
        self._codex_account_impact_records = 0
        self._codex_account_impact_confirmed = 0
        self._codex_account_impact_partial = 0
        self._codex_account_cashflow_confirmed = 0
        self._codex_account_mark_confirmed = 0
        self._codex_reward_sum = 0.0
        self._codex_parameter_changed = 0
        self._codex_probability_shift_count = 0
        self._codex_max_probability_shift = 0.0
        self._codex_last_reward: float | None = None

        self.close_count = self.final_close_row_count = self.win_count = self.loss_count = 0
        self._final_position_keys: set[tuple[str, str]] = set()
        self._position_pnl: dict[tuple[str, str], float] = {}
        self.gross_profit = self.gross_loss = self.net_pnl = 0.0
        self.close_reasons: Counter[str] = Counter()
        self.close_reason_net_pnl: Counter[str] = Counter()
        self.action_counts: Counter[str] = Counter()
        self.observatory_wire_action_counts: Counter[str] = Counter()
        self.execution_status: Counter[str] = Counter()
        self.request_count = 0
        self.start_balance: float | None = None
        self.balance = self.equity = 0.0
        self.peak_equity = 0.0
        self.max_equity_drawdown = 0.0
        self.last_market_time = ""
        self._bridge_events: set[str] = set()
        self.economic_action_counts: Counter[str] = Counter()
        self.entry_role_action_counts: dict[str, Counter[str]] = {
            "LONG_ENTRY_OWNER": Counter(),
            "SHORT_ENTRY_OWNER": Counter(),
        }
        self._economic_action_trace_ids: set[str] = set()
        # Keep a bounded action tail in bridge/observatory order.  Whole-run
        # entropy can remain diverse after a late HOLD collapse, so the
        # continuation guard needs the final rows as a separate signal.
        self._wire_action_tail: deque[str] = deque(maxlen=25)
        self._economic_action_tail: deque[str] = deque(maxlen=25)

        self.competitive_batches = 0
        self.ppo_updates = 0
        self.pattern_updates = 0
        self.reasoning_optimizer_updates = 0
        self.reasoning_group_eligible = 0.0
        self.reasoning_optimizer_eligible = 0.0
        self.rollout_steps = 0
        self.policy_loss = self.value_loss = self.total_loss = self.policy_entropy = None
        self.actor_eligible_fraction = self.reward_eligible_fraction = None
        self.reward_nonzero = self.reward_slots = 0.0
        self.reward_active_batches = 0
        self.reward_closed_trades = 0.0
        self.reward_mean_last = None
        self.reasoning_action_scale = None
        self.imitation_prior_enabled = None

    def _consume_execution(self) -> None:
        for row in self.execution.rows():
            transaction = row.get("transaction_type", "")
            status = row.get("status", "")
            self.execution_status[status] += 1
            if transaction == "BATCH_RESOLUTION" and row.get("source") == "policy":
                self.action_counts[row.get("action_name", "UNKNOWN")] += 1
            if transaction == "REQUEST":
                self.request_count += 1
            if row.get("closed") != "1":
                continue
            pnl = (
                _number(row.get("profit"))
                + _number(row.get("commission"))
                + _number(row.get("swap"))
            )
            self.close_count += 1
            source = str(self.execution.reader.path.resolve())
            position_ticket = str(row.get("position_ticket", "")).strip()
            position_key = (
                source,
                position_ticket if position_ticket else f"deal:{self.close_count}",
            )
            self._position_pnl[position_key] = self._position_pnl.get(position_key, 0.0) + pnl
            if row.get("position_final_close") == "1":
                self.final_close_row_count += 1
                self._final_position_keys.add(position_key)
                if not position_ticket:
                    # Older ledgers may mark a final close without exposing the
                    # position ticket.  Such a row cannot be safely merged with
                    # another row, so retain it as one independently observed
                    # final position instead of silently dropping it.
                    pass
            self.net_pnl += pnl
            if pnl > 0.0:
                self.win_count += 1
                self.gross_profit += pnl
            elif pnl < 0.0:
                self.loss_count += 1
                self.gross_loss += -pnl
            close_reason = row.get("close_reason", "UNKNOWN") or "UNKNOWN"
            self.close_reasons[close_reason] += 1
            self.close_reason_net_pnl[close_reason] += pnl

    def _consume_bridge(self) -> None:
        for row in self.bridge.rows():
            event_id = row.get("event_id", "")
            if event_id and event_id in self._bridge_events:
                continue
            if event_id:
                self._bridge_events.add(event_id)
            balance = _number(row.get("balance"), self.balance)
            equity = _number(row.get("equity"), self.equity)
            if self.start_balance is None and balance > 0.0:
                self.start_balance = balance
            self.balance = balance
            self.equity = equity
            self.last_market_time = row.get("time", self.last_market_time)
            self.peak_equity = max(self.peak_equity, equity)
            if self.peak_equity > 0.0:
                drawdown = (self.peak_equity - equity) / self.peak_equity
                self.max_equity_drawdown = max(self.max_equity_drawdown, drawdown)

    @staticmethod
    def _codex_record_id(row: dict[str, Any]) -> str:
        packet = str(row.get("packet_sha256", "") or "").strip()
        if packet:
            return packet
        trace = str(row.get("trace_id", "") or "").strip()
        ticket = str(row.get("position_ticket", "") or "").strip()
        action = str(row.get("action", "") or "").strip()
        return f"{trace}|{ticket}|{action}"

    def _refresh_codex_status(self) -> None:
        status_paths = sorted(self.output.glob("*codex_teacher_receipts.jsonl.status.json"))
        if not status_paths:
            return
        path = status_paths[-1]
        self.codex_status_path = path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return
        if isinstance(raw, dict):
            self.codex_status = raw

    def _consume_codex_receipts(self) -> None:
        receipt_paths = sorted(self.output.glob("*codex_teacher_receipts.jsonl"))
        if receipt_paths and self.codex_receipts.path != receipt_paths[-1]:
            self.codex_receipts = _AppendReader(receipt_paths[-1])
        for line in self.codex_receipts.lines():
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            record_type = str(row.get("record_type", "") or "")
            identity = self._codex_record_id(row)
            if record_type == "codex_teacher_failure":
                failure_id = f"failure:{identity}:{row.get('stage', '')}"
                if failure_id in self._codex_receipt_ids:
                    continue
                self._codex_receipt_ids.add(failure_id)
                self._codex_submitted += 1
                self._codex_failures += 1
                continue
            if record_type == "codex_teacher_receipt":
                if identity in self._codex_receipt_ids:
                    continue
                self._codex_receipt_ids.add(identity)
                self._codex_submitted += 1
                self._codex_graded += 1
                if str(row.get("hypothesis", "") or "").strip():
                    self._codex_hypotheses += 1
                if str(row.get("verification_status", "") or "").lower() == "confirmed":
                    self._codex_hypotheses_confirmed += 1
                impact = row.get("account_impact", {})
                if isinstance(impact, dict):
                    impact_status = str(impact.get("status", "") or "").lower()
                    if impact_status in {"confirmed", "partial"}:
                        self._codex_account_impact_records += 1
                    if impact_status == "confirmed":
                        self._codex_account_impact_confirmed += 1
                        scope = str(impact.get("evidence_scope", "") or "")
                        if scope == "immutable_lifecycle_plus_unique_deal_cashflow":
                            self._codex_account_cashflow_confirmed += 1
                        elif scope == "immutable_lifecycle_plus_observed_runtime_account_mark":
                            self._codex_account_mark_confirmed += 1
                    elif impact_status == "partial":
                        self._codex_account_impact_partial += 1
                reward = row.get("teacher_reward")
                if isinstance(reward, (int, float)) and math.isfinite(float(reward)):
                    self._codex_reward_sum += float(reward)
                    self._codex_last_reward = float(reward)
                continue
            if record_type != "codex_teacher_update" or identity in self._codex_update_ids:
                continue
            self._codex_update_ids.add(identity)
            update = row.get("update", {})
            if not isinstance(update, dict):
                update = {}
            updated = bool(row.get("rlhf_ppo_update", update.get("rlhf_ppo_update", False)))
            rejected = bool(row.get("rlhf_update_rejected", update.get("rlhf_update_rejected", False)))
            self._codex_updates += int(updated and not rejected)
            self._codex_rejected_updates += int(rejected)
            self._codex_interventions += int(
                bool(row.get("accepted", False))
                and bool(update.get("rlhf_codex_global_intervention_applied", updated))
            )
            self._codex_parameter_changed += int(
                bool(row.get("rlhf_parameter_changed", update.get("rlhf_parameter_changed", False)))
            )
            shift = update.get(
                "rlhf_packet_max_probability_shift",
                row.get("packet_max_probability_shift", 0.0),
            )
            shift_value = abs(_number(shift))
            self._codex_max_probability_shift = max(self._codex_max_probability_shift, shift_value)
            self._codex_probability_shift_count += int(shift_value > 0.0)

    def _codex_snapshot(self) -> dict[str, Any]:
        status = self.codex_status

        def metric(*names: str, fallback: Any = 0) -> Any:
            for name in names:
                if name in status and status[name] is not None:
                    return status[name]
            return fallback

        submitted = int(metric("submitted", "collected", fallback=self._codex_submitted) or 0)
        graded = int(metric("graded", fallback=self._codex_graded) or 0)
        hypotheses = int(metric("hypotheses", fallback=self._codex_hypotheses) or 0)
        confirmed = int(
            metric("hypotheses_confirmed", fallback=self._codex_hypotheses_confirmed) or 0
        )
        updates = int(
            metric("codex_optimizer_updates", "updates", fallback=self._codex_updates) or 0
        )
        failures = int(metric("failures", "grade_failures", fallback=self._codex_failures) or 0)
        pending = int(metric("pending", "deferred_pending", fallback=self._codex_pending) or 0)
        preflight = status.get("preflight", {})
        if not isinstance(preflight, dict):
            preflight = {}
        enabled = bool(metric("enabled", fallback=False))
        if str(preflight.get("status", "")).lower() == "failed":
            connection_status = "failed"
        elif failures:
            connection_status = "degraded"
        elif pending:
            connection_status = "pending"
        elif updates:
            connection_status = "active"
        elif graded:
            connection_status = "graded_no_update"
        elif enabled:
            connection_status = "waiting"
        else:
            connection_status = "disabled"
        return {
            "enabled": enabled,
            "status": connection_status,
            "submitted": submitted,
            "graded": graded,
            "hypotheses": hypotheses,
            "hypotheses_confirmed": confirmed,
            "updates": updates,
            "interventions_applied": int(
                metric("interventions_applied", fallback=self._codex_interventions) or 0
            ),
            "failures": failures,
            "pending": pending,
            "rejected_updates": int(
                metric("rejected_updates", fallback=self._codex_rejected_updates) or 0
            ),
            "account_impact_records": int(
                metric("account_impact_records", fallback=self._codex_account_impact_records) or 0
            ),
            "account_impact_confirmed": int(
                metric("account_impact_confirmed", fallback=self._codex_account_impact_confirmed) or 0
            ),
            "account_impact_partial": int(
                metric("account_impact_partial", fallback=self._codex_account_impact_partial) or 0
            ),
            "account_cashflow_confirmed": int(
                metric(
                    "account_cashflow_confirmed",
                    fallback=self._codex_account_cashflow_confirmed,
                ) or 0
            ),
            "account_mark_confirmed": int(
                metric(
                    "account_mark_confirmed",
                    fallback=self._codex_account_mark_confirmed,
                ) or 0
            ),
            "account_impact_reward_sum": _number(
                metric("account_impact_reward_sum", fallback=self._codex_reward_sum)
            ),
            "parameter_changed": int(
                metric("rlhf_parameter_changed", fallback=self._codex_parameter_changed) or 0
            ),
            "probability_shift_count": int(
                metric("rlhf_probability_shift_count", fallback=self._codex_probability_shift_count) or 0
            ),
            "max_probability_shift": _number(
                metric("rlhf_max_probability_shift", fallback=self._codex_max_probability_shift)
            ),
            "last_reward": self._codex_last_reward,
            "model": str(metric("model", fallback="") or ""),
            "reasoning_effort": str(metric("reasoning_effort", fallback="") or ""),
            "executable_path": str(
                metric("executable_path", fallback="") or ""
            ),
            "cli_version": str(metric("cli_version", fallback="") or ""),
            "preflight": dict(preflight),
            "status_path": str(self.codex_status_path) if self.codex_status_path else "",
        }

    def _consume_observatory(self) -> None:
        for line in self.observatory.lines():
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            stage = str(row.get("stage", ""))
            scalars = dict(_nested_scalars(row))
            routing = row.get("routing", {})
            if not isinstance(routing, dict):
                routing = {}
            if stage == "five_execution_receipt":
                trace = row.get("trace", {})
                if not isinstance(trace, dict):
                    trace = {}
                details = trace.get("action_details", {})
                if not isinstance(details, dict):
                    details = {}
                action_name = str(details.get("action_name", "") or "").strip()
                if not action_name:
                    action_index = details.get(
                        "policy_action",
                        details.get("five_action_index", details.get("action_index")),
                    )
                    try:
                        action_name = {
                            0: "HOLD",
                            1: "MARKET_BUY",
                            2: "MARKET_SELL",
                            3: "BUY_LIMIT",
                            4: "SELL_LIMIT",
                            5: "BUY_STOP",
                            6: "SELL_STOP",
                            7: "CANCEL_ORDERS",
                            8: "CLOSE_WORST",
                            9: "CLOSE_BEST",
                            10: "CLOSE_ALL",
                            11: "MANAGE_POSITION",
                        }.get(int(action_index), "UNKNOWN")
                    except (TypeError, ValueError):
                        action_name = "UNKNOWN"
                self._wire_action_tail.append(action_name)
                if action_name and action_name != "UNKNOWN":
                    self.observatory_wire_action_counts[action_name] += 1
                role = str(
                    details.get(
                        "five_agent_authority_role",
                        dict(trace.get("context_audit", {}) or {}).get(
                            "decision_role", ""
                        ),
                    )
                    or ""
                ).strip()
                if role in self.entry_role_action_counts and action_name != "UNKNOWN":
                    self.entry_role_action_counts[role][action_name] += 1
                if bool(details.get("five_economic_actor_eligible", False)):
                    trace_id = str(
                        trace.get("trace_id", row.get("trace_id", "")) or ""
                    ).strip()
                    if not trace_id:
                        # A deterministic fallback keeps duplicate observatory
                        # rows from inflating the eligible action denominator.
                        trace_id = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
                    if trace_id not in self._economic_action_trace_ids:
                        self._economic_action_trace_ids.add(trace_id)
                        self.economic_action_counts[action_name] += 1
                        self._economic_action_tail.append(action_name)
            elif stage == "five_competitive_reward":
                self.competitive_batches += 1
                updated = _number(routing.get("updated")) > 0.5
                self.ppo_updates += int(updated)
                self.rollout_steps = int(_number(routing.get("rollout_steps"), self.rollout_steps))
                if "reward_nonzero_count" in routing:
                    self.reward_nonzero += _number(routing.get("reward_nonzero_count"))
                    self.reward_slots += 5.0
                reward_active = any(
                    abs(_number(routing.get(key))) > 1e-12
                    for key in (
                        "reward_profit_delta_mean", "reward_position_improvement_count",
                        "reward_drawdown_increase", "reward_cost_mean", "reward_risk_penalty_mean",
                        "reward_conflict_penalty_mean", "reward_competitive_bonus_abs_mean",
                        "reward_closed_trades", "reward_invalid_position_action_no_position_count",
                        "reward_invalid_cancel_no_pending_count",
                    )
                )
                self.reward_active_batches += int(reward_active)
                self.reward_closed_trades += _number(routing.get("reward_closed_trades"))
                if "reward_mean" in routing:
                    self.reward_mean_last = _number(routing.get("reward_mean"))
                if "reward_sample_eligible_fraction" in routing:
                    self.reward_eligible_fraction = _number(routing.get("reward_sample_eligible_fraction"))
                if updated:
                    self.policy_loss = _number(routing.get("policy_loss"))
                    self.value_loss = _number(routing.get("value_loss"))
                    self.total_loss = _number(routing.get("total_loss"))
                    self.policy_entropy = _number(
                        routing.get("policy_action_entropy", routing.get("entropy"))
                    )
                    self.actor_eligible_fraction = _number(routing.get("actor_eligible_fraction"))
            elif stage == "five_pattern_path_credit_accepted":
                self.pattern_updates += int(_number(routing.get("updates")))
                # Eligibility is a per-update count, not a last-value gauge.
                # The final pattern receipt can legitimately contain zero
                # eligible groups (for example when every sampled hypothesis
                # received the same verifier value).  Overwriting the
                # accumulated count with that final zero made a run with
                # real reasoning updates look telemetry-only in the UI.
                self.reasoning_group_eligible += max(
                    0.0, _number(routing.get("reasoning_group_eligible"))
                )
                self.reasoning_optimizer_eligible += max(
                    0.0, _number(routing.get("reasoning_optimizer_eligible"))
                )
                self.reasoning_optimizer_updates += int(
                    _number(routing.get("reasoning_optimizer_updated")) > 0.5
                )
            if "five_reasoning_action_scale" in scalars:
                self.reasoning_action_scale = _number(scalars.get("five_reasoning_action_scale"))
            if "five_imitation_prior_enabled" in scalars:
                self.imitation_prior_enabled = bool(scalars.get("five_imitation_prior_enabled"))

    def refresh(self) -> dict[str, Any]:
        # File locations appear only after the local Strategy Tester agent has
        # started.  Runner clones can already contain zero-byte placeholder
        # files, so existence alone does not mean that a reader is bound to the
        # real agent ledger.  Keep resolving until a genuine header is read.
        execution_path = _latest_runner_file(self.runner, "self_growing_execution.csv")
        if execution_path != self.execution.reader.path:
            self.execution = _TsvReader(execution_path)
        bridge_path = _latest_runner_file(self.runner, "self_growing_bridge.csv")
        if bridge_path != self.bridge.reader.path:
            self.bridge = _TsvReader(bridge_path)
        if not self.observatory.path.is_file():
            candidates = sorted(self.output.glob("*.execution_trace_observatory.jsonl"))
            if candidates:
                self.observatory = _AppendReader(candidates[0])
        self._consume_execution()
        self._consume_bridge()
        self._consume_observatory()
        self._consume_codex_receipts()
        self._refresh_codex_status()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        decisive = self.win_count + self.loss_count
        win_rate = self.win_count / decisive if decisive else None
        profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0.0 else None
        avg_win = self.gross_profit / self.win_count if self.win_count else None
        avg_loss = self.gross_loss / self.loss_count if self.loss_count else None
        payoff = avg_win / avg_loss if avg_win is not None and avg_loss else None
        breakeven_win_rate = 1.0 / (1.0 + payoff) if payoff is not None else None
        expectancy = self.net_pnl / self.close_count if self.close_count else None
        # A position can have several close deals at the same terminal time and
        # every deal may carry position_final_close=1.  Compute from the final
        # accumulated value so later rows for the same ticket are not lost.
        completed_position_pnls = [
            self._position_pnl[key] for key in self._final_position_keys
        ]
        position_count = len(completed_position_pnls)
        position_expectancy = (
            sum(completed_position_pnls) / position_count if position_count else None
        )
        if position_count >= 2:
            mean = float(position_expectancy)
            variance = sum((value - mean) ** 2 for value in completed_position_pnls) / (position_count - 1)
            position_expectancy_lcb = mean - 1.64 * math.sqrt(variance) / math.sqrt(position_count)
        else:
            position_expectancy_lcb = None
        return_pct = (
            self.balance / self.start_balance - 1.0
            if self.start_balance and self.start_balance > 0.0 else None
        )

        def summarize_action_counts(counts: Counter[str]) -> dict[str, Any]:
            total = sum(counts.values())
            probabilities = [count / total for count in counts.values() if count] if total else []
            entropy = -sum(value * math.log(value) for value in probabilities)
            normalized = (
                entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
            ) if total else None
            top, top_count = counts.most_common(1)[0] if total else ("", 0)
            long_entries = sum(
                int(counts.get(name, 0))
                for name in ("MARKET_BUY", "BUY_LIMIT", "BUY_STOP")
            )
            short_entries = sum(
                int(counts.get(name, 0))
                for name in ("MARKET_SELL", "SELL_LIMIT", "SELL_STOP")
            )
            directional = long_entries + short_entries
            return {
                "available": bool(total),
                "actions": total,
                "total": total,
                "action_counts": dict(counts),
                "normalized_action_entropy": normalized,
                "effective_action_count": math.exp(entropy) if total else None,
                "top_action": top,
                "top_action_share": top_count / total if total else None,
                "hold_share": counts.get("HOLD", 0) / total if total else None,
                "long_entry_actions": long_entries,
                "short_entry_actions": short_entries,
                "entry_direction_imbalance": (
                    (long_entries - short_entries) / directional if directional else None
                ),
            }

        # The execution CSV is authoritative while the tester agent is alive.
        # After runner cleanup the append-only observatory may remain, so use
        # its raw five_execution_receipt actions as a read-only fallback rather
        # than silently reporting zero wire actions.
        wire_counts = (
            self.action_counts
            if self.action_counts
            else self.observatory_wire_action_counts
        )
        wire_policy = summarize_action_counts(wire_counts)
        economic_policy = summarize_action_counts(self.economic_action_counts)
        tail_wire_policy = summarize_action_counts(Counter(self._wire_action_tail))
        tail_economic_policy = summarize_action_counts(Counter(self._economic_action_tail))
        tail_scope = (
            tail_economic_policy
            if tail_economic_policy["total"] > 0
            else tail_wire_policy
        )
        tail_collapse_detected = bool(
            tail_scope["total"] >= 25
            and (
                (tail_scope["hold_share"] or 0.0) >= 0.95
                or (tail_scope["effective_action_count"] or 0.0) <= 1.25
                or (tail_scope["normalized_action_entropy"] or 0.0) <= 0.05
            )
        )
        policy_scope = "economic_actor" if economic_policy["total"] > 0 else "wire"
        wire_collapse_detected = bool(
            wire_policy["total"] >= 25
            and (
                (wire_policy["hold_share"] or 0.0) >= 0.95
                or (wire_policy["effective_action_count"] or 0.0) <= 1.25
                or (wire_policy["normalized_action_entropy"] or 0.0) <= 0.05
            )
        )
        economic_actor_collapse_detected = bool(
            economic_policy["total"] >= 25
            and (
                (economic_policy["hold_share"] or 0.0) >= 0.95
                or (economic_policy["effective_action_count"] or 0.0) <= 1.25
                or (economic_policy["normalized_action_entropy"] or 0.0) <= 0.05
            )
        )
        entry_role_policy = {
            role: summarize_action_counts(counts)
            for role, counts in self.entry_role_action_counts.items()
            if sum(counts.values()) > 0
        }
        entry_role_collapse_detected = bool(
            any(
                int(scope["total"] or 0) >= 8
                and (
                    (scope["hold_share"] or 0.0) >= 0.80
                    or (scope["effective_action_count"] or 0.0) <= 1.85
                    or (scope["normalized_action_entropy"] or 0.0) <= 0.35
                )
                for scope in entry_role_policy.values()
            )
        )
        selected_policy = economic_policy if policy_scope == "economic_actor" else wire_policy
        action_total = int(selected_policy["total"])
        normalized_entropy = selected_policy["normalized_action_entropy"]
        top_action = str(selected_policy["top_action"])
        top_share = selected_policy["top_action_share"]
        effective_actions = selected_policy["effective_action_count"]
        hold_share = selected_policy["hold_share"]
        accepted_count = self.execution_status.get("ACCEPTED", 0)
        rejected_count = self.execution_status.get("REJECTED", 0)
        rejection_rate = rejected_count / (accepted_count + rejected_count) if accepted_count + rejected_count else None
        placed_count = self.execution_status.get("PLACED", 0)
        filled_count = self.execution_status.get("FILLED", 0)
        timeout_count = self.execution_status.get("UNFILLED_TIMEOUT", 0)
        timeout_rate = timeout_count / (timeout_count + filled_count) if timeout_count + filled_count else None
        reward_density = (
            self.reward_nonzero / self.reward_slots if self.reward_slots
            else self.reward_active_batches / self.competitive_batches if self.competitive_batches
            else None
        )
        reward_density_basis = "exact_slots" if self.reward_slots else "active_economic_batches"

        optimizer_update_count = self.ppo_updates + self.reasoning_optimizer_updates
        optimizer_status = "insufficient_updates" if optimizer_update_count < 3 else (
            "warming_up" if optimizer_update_count < 10 else "active"
        )
        risk_status = "critical" if (
            self.max_equity_drawdown >= 0.20
            or (self.close_count >= 100 and profit_factor is not None and profit_factor < 0.75)
        ) else "warning" if (
            self.max_equity_drawdown >= 0.10
            or (self.close_count >= 50 and profit_factor is not None and profit_factor < 1.0)
        ) else "normal"
        edge_status = "negative" if (
            self.close_count >= 100 and profit_factor is not None
            and profit_factor < 0.90 and expectancy is not None and expectancy < 0.0
        ) else "candidate_positive" if (
            self.close_count >= 200 and profit_factor is not None and profit_factor >= 1.10
            and expectancy is not None and expectancy > 0.0 and self.max_equity_drawdown < 0.15
        ) else "unproven"
        reward_status = "sparse" if (
            self.competitive_batches >= 16 and reward_density is not None and reward_density < 0.10
        ) else "watch" if (
            self.competitive_batches >= 8 and reward_density is not None and reward_density < 0.25
        ) else "healthy" if self.competitive_batches else "insufficient_data"
        policy_status = "collapsed" if (
            entry_role_collapse_detected
            or (
            action_total >= 100 and normalized_entropy is not None
            and (normalized_entropy < 0.35 or (top_share or 0.0) > 0.80)
            )
        ) else "watch" if (
            action_total >= 100 and normalized_entropy is not None
            and (normalized_entropy < 0.60 or (top_share or 0.0) > 0.60)
        ) else "diverse" if action_total >= 100 else "insufficient_data"
        execution_quality = "poor" if (
            (rejection_rate or 0.0) > 0.05 or (timeout_rate or 0.0) > 0.50
        ) else "watch" if (
            (rejection_rate or 0.0) > 0.01 or (timeout_rate or 0.0) > 0.20
        ) else "healthy" if self.request_count else "insufficient_data"
        capacity_diagnosis = (
            "insufficient_optimization" if optimizer_update_count < 5
            else "credit_or_objective_bottleneck" if edge_status == "negative" and reward_status in {"sparse", "watch"}
            else "policy_optimization_bottleneck" if edge_status == "negative"
            else "no_architecture_underfit_evidence"
        )

        final_position_closes = len(self._final_position_keys)
        duplicate_final_close_rows = self.final_close_row_count - final_position_closes
        long_entry_actions = int(selected_policy["long_entry_actions"])
        short_entry_actions = int(selected_policy["short_entry_actions"])
        entry_direction_imbalance = selected_policy["entry_direction_imbalance"]
        codex_teacher = self._codex_snapshot()

        return {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "updated_at": time.time(),
            "runtime_dir": str(self.runner),
            "market_time": self.last_market_time,
            "status": {
                "edge": edge_status,
                "risk": risk_status,
                "optimizer": optimizer_status,
                "reward": reward_status,
                "policy": policy_status,
                "execution": execution_quality,
                "codex_teacher": codex_teacher["status"],
                "capacity_diagnosis": capacity_diagnosis,
                "overfit_live": "not_measurable_without_oos",
            },
            "codex_teacher": codex_teacher,
            "performance": {
                "closed_trades": self.close_count,
                "closed_deals": self.close_count,
                "final_position_closes": final_position_closes,
                "final_close_rows": self.final_close_row_count,
                "duplicate_final_close_rows": duplicate_final_close_rows,
                "wins": self.win_count,
                "losses": self.loss_count,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "position_expectancy": position_expectancy,
                "position_expectancy_lcb_95_one_sided": position_expectancy_lcb,
                "net_pnl": self.net_pnl,
                "gross_profit": self.gross_profit,
                "gross_loss": self.gross_loss,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "payoff_ratio": payoff,
                "breakeven_win_rate": breakeven_win_rate,
                "close_reasons": dict(self.close_reasons),
                "close_reason_net_pnl": dict(self.close_reason_net_pnl),
            },
            "risk": {
                "start_balance": self.start_balance,
                "balance": self.balance,
                "equity": self.equity,
                "return_pct": return_pct,
                "max_equity_drawdown_pct": self.max_equity_drawdown,
            },
            "optimizer": {
                "ppo_updates": self.ppo_updates,
                "reasoning_optimizer_updates": self.reasoning_optimizer_updates,
                "optimizer_update_count": optimizer_update_count,
                # Codex teacher updates are deliberately separate from the
                # generic PPO/pattern total above.  Adding them here would
                # make a Codex receipt look like a generic competitive batch.
                "codex_teacher_updates": codex_teacher["updates"],
                "codex_teacher_graded": codex_teacher["graded"],
                "competitive_batches": self.competitive_batches,
                "rollout_steps": self.rollout_steps,
                "policy_loss": self.policy_loss,
                "value_loss": self.value_loss,
                "total_loss": self.total_loss,
                "policy_entropy": self.policy_entropy,
                "actor_eligible_fraction": self.actor_eligible_fraction,
                "pattern_updates": self.pattern_updates,
                "reasoning_group_eligible": self.reasoning_group_eligible,
                "reasoning_optimizer_eligible": self.reasoning_optimizer_eligible,
            },
            "reward": {
                "nonzero_density": reward_density,
                "nonzero_count": self.reward_nonzero,
                "observed_slots": self.reward_slots,
                "active_batches": self.reward_active_batches,
                "observed_batches": self.competitive_batches,
                "density_basis": reward_density_basis,
                "eligible_fraction_last": self.reward_eligible_fraction,
                "closed_trades_credited": self.reward_closed_trades,
                "mean_last": self.reward_mean_last,
            },
            "policy": {
                "scope": policy_scope,
                "actions": action_total,
                "total": action_total,
                "action_counts": dict(selected_policy["action_counts"]),
                "normalized_action_entropy": normalized_entropy,
                "effective_action_count": effective_actions,
                "top_action": top_action,
                "top_action_share": top_share,
                "hold_share": hold_share,
                "long_entry_actions": long_entry_actions,
                "short_entry_actions": short_entry_actions,
                "entry_direction_imbalance": entry_direction_imbalance,
                "reasoning_action_scale": self.reasoning_action_scale,
                "imitation_prior_enabled": self.imitation_prior_enabled,
                "wire_actions": wire_policy["total"],
                "wire_action_counts": wire_policy["action_counts"],
                "wire_hold_share": wire_policy["hold_share"],
                "wire_effective_action_count": wire_policy["effective_action_count"],
                "wire_normalized_action_entropy": wire_policy["normalized_action_entropy"],
                "wire_collapse_detected": wire_collapse_detected,
                "economic_actor_collapse_detected": economic_actor_collapse_detected,
                "entry_role_distributions": entry_role_policy,
                "entry_role_collapse_detected": entry_role_collapse_detected,
                "wire_collapse_warning": bool(
                    wire_collapse_detected and policy_scope != "wire"
                ),
                "economic_actor": economic_policy,
                "tail_window_size": int(tail_scope["total"]),
                "tail": {
                    "wire": tail_wire_policy,
                    "economic_actor": tail_economic_policy,
                    "collapse_scope": (
                        "economic_actor"
                        if tail_economic_policy["total"] > 0
                        else "wire"
                    ),
                    "collapse_detected": tail_collapse_detected,
                    **dict(tail_scope),
                },
                "tail_collapse_detected": tail_collapse_detected,
            },
            "execution": {
                "requests": self.request_count,
                "accepted": accepted_count,
                "rejected": rejected_count,
                "placed": placed_count,
                "filled": filled_count,
                "rejection_rate": rejection_rate,
                "unfilled_timeout_count": timeout_count,
                "unfilled_timeout_rate": timeout_rate,
                "status_counts": dict(self.execution_status),
            },
        }


class Mt5DiagnosticsMonitor:
    def __init__(self, output: Path, runner: Path, interval_seconds: float = 5.0) -> None:
        self.output = output.resolve()
        self.diagnostics = Mt5LiveDiagnostics(output, runner)
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.stop = threading.Event()
        self._publish_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="mt5-live-diagnostics", daemon=True)

    @property
    def status_path(self) -> Path:
        return self.output / "mt5_diagnostics.json"

    def start(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self._publish(self.diagnostics.snapshot())
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=15.0)
        self._publish(self.diagnostics.refresh())

    def _publish(self, snapshot: dict[str, Any]) -> None:
        # ``close()`` performs one final publish while the monitor thread may
        # still be finishing a refresh.  Serialize both writers: sharing the
        # same fixed temporary filename otherwise lets one replace/unlink it
        # while the other is in ``Path.replace`` and raises WinError 5.
        with self._publish_lock:
            temporary = self.status_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Windows readers opened without FILE_SHARE_DELETE can briefly
            # hold the destination while the desktop/UI polls it. Retry the
            # replace so a harmless read does not terminate the live monitor.
            for attempt in range(_PUBLISH_RETRY_COUNT):
                try:
                    temporary.replace(self.status_path)
                    return
                except PermissionError:
                    if attempt + 1 >= _PUBLISH_RETRY_COUNT:
                        raise
                    time.sleep(_PUBLISH_RETRY_DELAY_SECONDS)

    def _run(self) -> None:
        while not self.stop.wait(self.interval_seconds):
            try:
                self._publish(self.diagnostics.refresh())
            except Exception as exc:  # Diagnostics must never stop model training.
                try:
                    self._publish({
                        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                        "updated_at": time.time(),
                        "status": {"monitor": "error"},
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                except PermissionError:
                    # A polling reader may still own the destination. The
                    # next interval can publish a fresh snapshot; monitoring
                    # must remain best-effort and never kill MT5 learning.
                    continue
