from __future__ import annotations







import csv



import html



import io



import json



import math



import os



import re



import shutil



import subprocess



import time



import traceback



from collections import deque



from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence



from concurrent.futures import Future, ThreadPoolExecutor



from dataclasses import dataclass, field, replace



from datetime import datetime



from pathlib import Path







import numpy as np







from .config import FeatureConfig, OrderConfig, RiskConfig, SymbolConfig



from .feature_contract import (



    compare_mt5_feature_audit_against_reference,



    feature_schema_fingerprint,



    summarize_mt5_feature_audit,



)



from .learning_integrity import clone_policy, policy_state_hash, sha256_file, sha256_tree, stable_object_hash



from .input_normalization import normalize_input_vector



from .causal_ledger import classify_exit_mechanism



from .contextual_expectancy import robust_expectancy



from .regime_detector import (



    DEFAULT_CHAMPION_PATH,



    MarketRegime,



    RegimeDetector,



    REGIME_NAMES,



    NUM_REGIMES,



)



from .orders import (



    ACTION_COUNT, OUTPUT_COUNT,



    Action, PositionManagementMode, action_schema_for_outputs, decode_outputs,



    allowed_actions_for_state, MIN_MANUAL_CLOSE_AGE_BARS, EMERGENCY_CLOSE_R,



    FACTORIZED_ACTION_MARKER_THRESHOLD, POSITION_REQUIRED_ACTIONS,



)



from .protocol_contract import (



    BRIDGE_SCHEMA_VERSION,



    EXECUTION_LEDGER_SCHEMA_VERSION,



    INPUT_COUNT,



    ACCOUNT_OFFSET,



    position_count_reference,



    position_limit_enabled,



)



from .execution_contract import FACTORIZED_ACTION_MARKER_OUTPUT_INDEX



from .entry_preference import EntryPreferenceRecord, parse_entry_preference_row



from .five_lifecycle_reward import (



    FiveLifecycleRewardConfig,



    lifecycle_path_dynamics,



    shape_five_entry_lifecycle,



)



from .five_agent_authority import (



    FIVE_AGENT_AUTHORITY_VERSION,



    HOLD_ACTION,



    resolve_opposite_entry_conflict,



    five_agent_authority,



    five_agent_responsibility_active,



)



from .five_shared_policy import FIVE_AGENT_COUNT as FIVE_PUBLIC_AGENT_COUNT, FIVE_LSTM_UNROLL











def enrich_step_trace(



    trace,



    action_details: dict[str, object],



    positions_pipe: str,



    positions_pipe_after: str | None = None,



    execution_details: dict[str, object] | None = None,



    account_context: dict[str, float] | None = None,



    account_context_after: dict[str, float] | None = None,



):



    """Attach bridge-side execution evidence to an immutable trace.







    ``action_details`` describes what the policy *requested*.  It is not proof



    that MT5 accepted, placed, filled, or cancelled anything.  Structural memory



    must use ``execution_details`` instead of the requested action.



    """



    try:



        from .chuyen_gia_policy import StepTrace



    except Exception:



        StepTrace = None



    after = str(positions_pipe_after) if positions_pipe_after is not None else str(getattr(trace, "positions_pipe_after", ""))



    execution = (



        dict(execution_details)



        if execution_details is not None



        else dict(getattr(trace, "execution_details", {}) or {})



    )



    before_context = (



        dict(account_context)



        if account_context is not None



        else dict(getattr(trace, "account_context", {}) or {})



    )



    after_context = (



        dict(account_context_after)



        if account_context_after is not None



        else dict(getattr(trace, "account_context_after", {}) or {})



    )



    # Keep policy-side evidence (recovery, extrema timing, memory queries)



    # while letting the bridge's decoded wire fields take precedence.



    merged_action_details = dict(getattr(trace, "action_details", {}) or {})



    merged_action_details.update(dict(action_details))



    if StepTrace is not None and isinstance(trace, StepTrace):



        return replace(



            trace,



            action_details=merged_action_details,



            positions_pipe=str(positions_pipe),



            positions_pipe_after=after,



            execution_details=execution,



            account_context=before_context,



            account_context_after=after_context,



        )



    try:



        setattr(trace, "action_details", merged_action_details)



        setattr(trace, "positions_pipe", str(positions_pipe))



        setattr(trace, "positions_pipe_after", after)



        setattr(trace, "execution_details", execution)



        setattr(trace, "account_context", before_context)



        setattr(trace, "account_context_after", after_context)



    except Exception:



        return trace



    return trace











def attach_experience_starvation(trace, starvation: dict[str, float]):



    try:



        from .chuyen_gia_policy import StepTrace



    except Exception:



        StepTrace = None



    if StepTrace is not None and isinstance(trace, StepTrace):



        return replace(trace, experience_starvation=dict(starvation))



    try:



        setattr(trace, "experience_starvation", dict(starvation))



    except Exception:



        return trace



    return trace











def update_diagnostics_regime_telemetry(



    diag_path: Path | str,



    soft_probs: Sequence[float] | np.ndarray,



    regime_name: str,



) -> bool:



    """Atomically record soft probabilities in mt5_diagnostics.json.







    Format:



        regime_telemetry: {"last_probabilities": soft_probs.tolist(), "regime_name": str}



    """



    path = Path(diag_path)



    probs_list = [round(float(p), 6) for p in soft_probs]



    data: dict[str, Any] = {}



    if path.is_file():



        try:



            data = json.loads(path.read_text(encoding="utf-8"))



        except Exception:



            data = {}



    if not isinstance(data, dict):



        data = {}



    data["regime_telemetry"] = {



        "last_probabilities": probs_list,



        "regime_name": str(regime_name),



    }



    path.parent.mkdir(parents=True, exist_ok=True)



    tmp_path = path.with_suffix(path.suffix + ".regime.tmp")



    try:



        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")



        for _ in range(50):



            try:



                tmp_path.replace(path)



                return True



            except PermissionError:



                time.sleep(0.01)



        tmp_path.replace(path)



        return True



    except Exception:



        return False











def record_event_regime_distribution(



    events_path: Path | str,



    soft_probs: Sequence[float] | np.ndarray,



    regime_name: str,



    *,



    event_type: str = "lifecycle",



    sequence: int = 0,



    trace_id: str = "",



    extra_payload: Mapping[str, Any] | None = None,



) -> bool:



    """Record an event with regime_distribution into events.jsonl.







    Format:



        {"schema_version": 10, "event_type": ..., "payload": {"regime_distribution": soft_probs.tolist(), ...}}



    """



    path = Path(events_path)



    probs_list = [round(float(p), 6) for p in soft_probs]



    payload: dict[str, Any] = {



        "regime_distribution": probs_list,



        "regime_name": str(regime_name),



    }



    if extra_payload:



        payload.update(dict(extra_payload))



    record = {



        "schema_version": 10,



        "event_type": str(event_type),



        "sequence": int(sequence),



        "trace_id": str(trace_id),



        "created_at": time.time(),



        "payload": payload,



    }



    path.parent.mkdir(parents=True, exist_ok=True)



    try:



        with path.open("a", encoding="utf-8") as f:



            f.write(json.dumps(record, ensure_ascii=False) + "\n")



        return True



    except Exception:



        return False











def _closed_execution_reward(



    details: dict[str, object],



    *,



    initial_equity: float,



    entry_equity: float = 0.0,



    exit_equity: float = 0.0,



    technical_min_rr: float = 1.2,



) -> float:



    """Return net R for the specific MT5 close receipt, not account drift.







    Final lifecycle settlement already learns in R units. Keeping online close



    feedback at ``0.1 * R`` made the direct-RNN margin update nearly vanish



    because that learner also scales its step by the advantage. Use the same



    bounded net-R unit online so an expert can correct a losing entry or exit



    during the current tester session.



    """



    try:



        profit = float(details.get("profit", 0.0) or 0.0)



    except (TypeError, ValueError):



        profit = 0.0



    try:



        commission = float(details.get("commission", 0.0) or 0.0)



    except (TypeError, ValueError):



        commission = 0.0



    try:



        swap = float(details.get("swap", 0.0) or 0.0)



    except (TypeError, ValueError):



        swap = 0.0



    try:



        initial_risk = abs(float(details.get("initial_risk", 0.0) or 0.0))



    except (TypeError, ValueError):



        initial_risk = 0.0



    net_profit = profit + commission + swap



    close_reason = str(details.get("close_reason", "") or "").upper()



    if initial_risk <= 1e-12 and close_reason in {"SL", "STOP_LOSS", "STOP_OUT"} and net_profit < -1e-12:



        initial_risk = abs(net_profit)



    elif initial_risk <= 1e-12 and close_reason in {"TP", "TAKE_PROFIT"} and net_profit > 1e-12:



        # Same-tick pending fills can disappear before the EA snapshots the



        # live position. The fixed configuration validates RR before a run;



        # this is a bounded audit fallback until the history-order risk is



        # present in the execution receipt.



        initial_risk = net_profit / max(1e-9, float(technical_min_rr))



    if initial_risk > 1e-12:



        return float(np.clip(net_profit / initial_risk, -2.0, 2.0))



    if abs(net_profit) > 1e-12:



        return float(net_profit / max(1.0, float(initial_equity)))



    if entry_equity > 0.0 and exit_equity > 0.0:



        return float(np.clip(np.log(exit_equity / entry_equity), -0.02, 0.02) * 10.0)



    return 0.0











def _legacy_ack_sequence_for_close_row(



    row: dict[str, object],



    open_traces: dict[int, dict[str, object]],



) -> int | None:



    """Resolve an automatic close to its original broker-accepted entry.







    MT5 emits SL/TP closes as ``SYSTEM`` events.  Such receipts normally have



    neither an exit ``trace_id`` nor a useful ``ack_sequence``, but the EA



    persists the immutable entry owner in ``position_owner_trace_id``.



    """



    try:



        ack_sequence = int(float(row.get("ack_sequence", "")))



    except (TypeError, ValueError):



        ack_sequence = -1



    if ack_sequence in open_traces:



        return ack_sequence



    position_ticket = str(row.get("position_ticket", "") or "").strip()



    if position_ticket:



        for sequence, cached in open_traces.items():



            if str(cached.get("position_ticket", "") or "").strip() == position_ticket:



                return int(sequence)



    owner_trace_id = str(



        row.get("position_owner_trace_id", "")



        or row.get("owner_trace_id", "")



        or ""



    ).strip()



    if not owner_trace_id:



        return None



    for sequence, cached in open_traces.items():



        cached_trace_id = str(cached.get("trace_id", "") or "").strip()



        if not cached_trace_id:



            cached_trace_id = str(getattr(cached.get("trace"), "trace_id", "") or "").strip()



        if cached_trace_id == owner_trace_id:



            return int(sequence)



    return None











def _register_legacy_open_trace(



    open_traces: dict[int, dict[str, object]],



    *,



    request: object | None,



    trace: object | None,



    action: int,



    execution_details: Mapping[str, object] | None,



    score: float,



    confidence: float,



    equity: float,



    drawdown: float,



    spread_points: float,



) -> int | None:



    """Retain entry-owner routing after an ACCEPTED/PLACED broker receipt.







    The receipt is only routing evidence: no learner is updated until an MT5



    final-close receipt arrives.  Retaining it early lets a delayed pending



    fill followed by a SYSTEM SL/TP recover the original entry trace.



    """



    if int(action) not in {



        int(Action.MARKET_BUY), int(Action.MARKET_SELL),



        int(Action.BUY_LIMIT), int(Action.SELL_LIMIT),



        int(Action.BUY_STOP), int(Action.SELL_STOP),



    }:



        return None



    details = dict(execution_details or {})



    if not any(bool(details.get(key, False)) for key in ("accepted", "placed", "filled")):



        return None



    trace_id = str(getattr(trace, "trace_id", "") or "").strip()



    if not trace_id or request is None:



        return None



    try:



        fallback_sequence = int(getattr(request, "sequence", 0) or 0)



    except (TypeError, ValueError):



        fallback_sequence = 0



    try:



        ack_sequence = int(float(details.get("ack_sequence", fallback_sequence) or fallback_sequence))



    except (TypeError, ValueError):



        ack_sequence = fallback_sequence



    if ack_sequence <= 0:



        return None



    cached = dict(open_traces.get(ack_sequence, {}) or {})



    cached.update({



        "request": request,



        "trace": trace,



        "trace_id": trace_id,



        "action": int(action),



        "score": float(score),



        "confidence": float(confidence),



        "equity": float(equity),



        "drawdown": float(drawdown),



        "spread_points": float(spread_points),



    })



    position_ticket = str(details.get("position_ticket", "") or "").strip()



    if position_ticket:



        cached["position_ticket"] = position_ticket



    open_traces[ack_sequence] = cached



    return ack_sequence











def _pending_transition_feedback_allowed(



    *,



    flat_idle: bool,



    learning_enabled: bool,



    has_observer: bool,



    trace_learning_blocked: bool,



) -> bool:



    """Keep ordinary bridge feedback independent from the FLAT_IDLE firebreak."""



    return bool(



        learning_enabled



        and not flat_idle



        and has_observer



        and not trace_learning_blocked



    )











def _direct_rnn_actor_bridge_audit(



    action_details: Mapping[str, object],



    emitted_action: int,



) -> dict[str, object]:



    """Revoke actor credit when the bridge emits a different action.







    A context-expert facade may replace a Direct-RNN entry proposal with a



    management command after the local expert has already produced its trace.



    The stale local ``policy_eligible`` flag must never make that management



    outcome look like an on-policy exploration sample.



    """



    details = dict(action_details or {})



    if "direct_rnn_rl_policy_eligible" not in details:



        return {}



    try:



        direct_action = int(details.get("direct_rnn_rl_policy_action_index", -1))



    except (TypeError, ValueError):



        direct_action = -1



    local_eligible = bool(details.get("direct_rnn_rl_policy_eligible", False))



    emitted_matches = bool(direct_action >= 0 and int(emitted_action) == direct_action)



    bridge_eligible = bool(local_eligible and emitted_matches)



    if bridge_eligible:



        reason = "emitted_action_match"



    elif local_eligible:



        reason = "emitted_action_changed"



    else:



        reason = "local_policy_ineligible"



    return {



        "direct_rnn_rl_policy_eligible": bridge_eligible,



        "direct_rnn_rl_emitted_action_match": emitted_matches,



        "direct_rnn_rl_policy_eligibility_reason": reason,



    }











def _execution_record_close_details(



    record: dict[str, object],



    *,



    requested_action: int,



    owner_ack_sequence: int,



) -> dict[str, object]:



    """Normalize one final-close record for entry-owner lifecycle feedback."""



    truthy = {"1", "true", "yes", "ok", "success"}



    details = {



        "bridge_observed": True,



        "requested_action": int(requested_action),



        "ack_sequence": str(owner_ack_sequence),



        "ledger_ack_sequence": str(record.get("ack_sequence", "")),



        "intent_event_id": str(record.get("event_id", "")),



        "intent_trace_id": str(record.get("trace_id", "")),



        "position_owner_trace_id": str(record.get("position_owner_trace_id", "")),



        "acknowledged": True,



        "accepted": str(record.get("accepted", "")).strip().lower() in truthy,



        "placed": False,



        "filled": False,



        "cancelled": False,



        "modified": False,



        "closed": True,



        "rejected": False,



        "execution_confirmed": True,



        "outcome_eligible": True,



        "action_feedback_eligible": True,



        "memory_eligible": True,



        "ledger_required": True,



        "retcode": str(record.get("retcode", "")),



        "ticket": str(record.get("position_ticket", record.get("order_ticket", ""))),



        "order_ticket": str(record.get("order_ticket", "")),



        "deal_ticket": str(record.get("deal_ticket", "")),



        "position_ticket": str(record.get("position_ticket", "")),



        "intent_context": str(record.get("intent_context", "")),



        "intent_expert": str(record.get("intent_expert", "")),



        "intent_policy": str(record.get("intent_policy", "")),



        "position_owner_context": str(record.get("position_owner_context", "")),



        "position_owner_expert": str(record.get("position_owner_expert", "")),



        "position_owner_policy": str(record.get("position_owner_policy", "")),



        "transaction_type": str(record.get("transaction_type", "")),



        "execution_status": str(record.get("status", "")),



        "deal_entry": str(record.get("deal_entry", "")),



        "deal_reason": str(record.get("deal_reason", "")),



        "close_reason": str(record.get("close_reason", "")),



        "volume": str(record.get("volume", "")),



        "price": str(record.get("price", "")),



        "profit": str(record.get("profit", "")),



        "commission": str(record.get("commission", "")),



        "swap": str(record.get("swap", "")),



        "initial_risk": str(record.get("initial_risk", "")),



        "source": str(record.get("source", "")),



    }



    # Canonical Full-RL settlement reads the ledger again, but the immediate



    # observability receipt must retain the same immutable exit facts.  Without



    # this, a real BREAKEVEN_SL/TRAILING_SL was reported as a generic ``SL``.



    for key in (



        "entry_price", "initial_sl_price", "initial_tp_price",



        "initial_risk_distance_price", "final_sl_price", "final_tp_price",



        "old_sl_price", "old_tp_price", "new_sl_price", "new_tp_price",



        "stop_controller", "stop_reason", "exit_kind", "exit_controller",



        "exit_trace_id", "exit_requested_action", "post_exit_horizon_bars",



        "post_exit_hold_r", "geometry_mode", "fixed_sl_atr", "fixed_tp_atr", "fixed_rr",



        "geometry_entry_pipe", "geometry_path_pipe",



    ):



        value = record.get(key)



        if value is not None and str(value).strip() != "":



            details[key] = value



    return details











def _legacy_codex_lifecycle_from_close_details(



    details: Mapping[str, object],



    trace: object,



    realized_r: float,



) -> "MT5PositionLifecycle":



    """Create a bounded final-close view for one legacy Codex packet."""



    values = dict(details or {})







    def _number(name: str, default: float = 0.0) -> float:



        try:



            value = float(values.get(name, default) or default)



        except (TypeError, ValueError):



            value = float(default)



        return value if np.isfinite(value) else float(default)







    trace_id = str(getattr(trace, "trace_id", "") or "").strip()



    ticket = str(values.get("position_ticket", values.get("ticket", "")) or "").strip()



    metadata = {



        "position_owner_trace_id": str(



            values.get("position_owner_trace_id", "") or trace_id



        ).strip() or trace_id,



        "codex_teacher_legacy_close": True,



    }



    post_exit_horizon = _number("post_exit_horizon_bars", 0.0)



    post_exit_horizon_bars = (



        int(post_exit_horizon) if post_exit_horizon > 0.0 else None



    )



    post_exit_hold = _number("post_exit_hold_r", 0.0)



    has_post_exit_hold = bool(str(values.get("post_exit_hold_r", "") or "").strip())



    deal_ticket = str(values.get("deal_ticket", "") or "").strip()



    cashflow_delta = (



        _number("profit") + _number("commission") + _number("swap")



    )



    cashflow_metadata = (



        {"account_cashflow_attribution": "unique_final_deal"}



        if deal_ticket not in {"", "0", "0.0"}



        else {}



    )



    return MT5PositionLifecycle(



        position_ticket=ticket,



        entry_time=None,



        exit_time=None,



        net_pnl=_number("profit"),



        gross_profit=_number("profit"),



        commission=_number("commission"),



        swap=_number("swap"),



        initial_risk=abs(_number("initial_risk")),



        realized_r=float(realized_r),



        close_reason=str(values.get("close_reason", "") or ""),



        source="legacy_codex_live_close",



        trace_id=trace_id,



        entry_count=1,



        close_count=1,



        final_close_confirmed=True,



        entry_price=_number("entry_price"),



        initial_sl_price=_number("initial_sl_price"),



        initial_tp_price=_number("initial_tp_price"),



        final_sl_price=_number("final_sl_price"),



        final_tp_price=_number("final_tp_price"),



        mfe_r=_number("mfe_r"),



        mae_r=_number("mae_r"),



        exit_kind=str(values.get("exit_kind", "") or ""),



        exit_controller=str(values.get("exit_controller", "") or ""),



        exit_trace_id=str(values.get("exit_trace_id", "") or ""),



        exit_requested_action=int(_number("exit_requested_action", -1)),



        post_exit_horizon_bars=post_exit_horizon_bars,



        post_exit_hold_r=post_exit_hold if has_post_exit_hold else None,



        pending_fill_bars=(



            int(_number("pending_fill_bars"))



            if str(values.get("pending_fill_bars", "") or "").strip() else None



        ),



        metadata={**metadata, **cashflow_metadata},



        account_cashflow_delta=(cashflow_delta if deal_ticket not in {"", "0", "0.0"} else None),



        account_cashflow_deal_ticket=deal_ticket,



    )











def _merge_final_close_receipt(



    details: dict[str, object],



    ledger_rows: Iterable[dict[str, object]],



) -> dict[str, object]:



    """Fill an early close acknowledgement from the final MT5 deal receipt.







    The bridge can acknowledge ``CLOSE_*`` before ``OnTradeTransaction`` has



    attached ``initial_risk``.  Learning from that early row falls back to



    ``profit / account_equity`` and then marks the entry lifecycle settled,



    so the complete ``DEAL_ADD`` row is ignored.  Merge only rows for the same



    position and prefer a final close with positive initial risk.



    """



    merged = dict(details or {})



    if not bool(merged.get("closed", False)):



        return merged



    try:



        if abs(float(merged.get("initial_risk", 0.0) or 0.0)) > 1e-12:



            return merged



    except (TypeError, ValueError):



        pass



    ticket = str(merged.get("position_ticket", merged.get("ticket", "")) or "").strip()



    if not ticket:



        return merged



    truthy = {"1", "true", "yes", "ok", "success"}



    candidates: list[dict[str, object]] = []



    for row in ledger_rows:



        if str(row.get("position_ticket", "") or "").strip() != ticket:



            continue



        final_close = str(row.get("position_final_close", "") or "").strip().lower() in truthy



        closed = str(row.get("closed", "") or "").strip().lower() in truthy



        try:



            initial_risk = abs(float(row.get("initial_risk", 0.0) or 0.0))



        except (TypeError, ValueError):



            initial_risk = 0.0



        if (final_close or closed) and initial_risk > 1e-12:



            candidates.append(dict(row))



    if not candidates:



        return merged



    receipt = max(



        candidates,



        key=lambda row: (



            str(row.get("position_final_close", "") or "").strip().lower() in truthy,



            str(row.get("transaction_type", "") or "").upper() == "DEAL_ADD",



        ),



    )



    # A bridge ACK can arrive before the final ``DEAL_ADD`` row.  Preserve the



    # complete immutable close contract when it does: Full-RL must see the



    # original geometry and the EA's explicit exit taxonomy, not a lossy



    # ``SL``/``TP`` shorthand reconstructed from PnL.



    for key in (



        "profit", "commission", "swap", "initial_risk", "close_reason",



        "deal_entry", "deal_reason", "position_final_close", "age_bars", "pending_fill_bars",



        "entry_price", "initial_sl_price", "initial_tp_price",



        "initial_risk_distance_price", "final_sl_price", "final_tp_price",



        "old_sl_price", "old_tp_price", "new_sl_price", "new_tp_price",



        "stop_controller", "stop_reason", "exit_kind", "exit_controller",



        "exit_trace_id", "exit_requested_action", "post_exit_horizon_bars",



        "post_exit_hold_r", "geometry_mode", "fixed_sl_atr", "fixed_tp_atr", "fixed_rr",



        "geometry_entry_pipe", "geometry_path_pipe",



    ):



        value = receipt.get(key)



        if value is not None and str(value).strip() != "":



            merged[key] = value



    merged["final_close_receipt_merged"] = True



    return merged











DEFAULT_RUNNER_DIR = Path("mt5_v2_runner")



DEFAULT_BRIDGE_FILE = "self_growing_bridge.csv"



DEFAULT_REQUEST_FILE = "self_growing_request.csv"



DEFAULT_ACTION_FILE = "self_growing_action.csv"



DEFAULT_EXECUTION_FILE = "self_growing_execution.csv"



DEFAULT_FEATURE_AUDIT_FILE = "self_growing_feature_audit.csv"



DEFAULT_ENTRY_PREFERENCE_FILE = "self_growing_entry_preference.csv"



DEFAULT_STOP_REQUEST_MARKER_FILE = "self_growing_stop.request"



# V10 aligns the live MT5 request width with the 137-input context-expert policy.



ACTION_SCHEMA_VERSION = BRIDGE_SCHEMA_VERSION



FITNESS_SCHEMA_VERSION = 7



DEFAULT_SYMBOL = "BTCUSDm"



DEFAULT_PERIOD = "M1"



DEFAULT_FROM_DATE = "2025.01.01"



DEFAULT_TO_DATE = "2025.02.01"



DEFAULT_EXPERT = "SelfGrowing_MT5_Trainer_EA"



# MT5 tester model identifiers from the official [Tester] configuration:



# 0=Every tick, 1=1-minute OHLC, 2=Open prices only, 3=Math calculations,



# 4=Every tick based on real ticks.  Keep this explicit so model and execution



# delay are never conflated.



MT5_MODEL_EVERY_TICK = 0



MT5_MODEL_1MIN_OHLC = 1



MT5_MODEL_OPEN_PRICES = 2



MT5_MODEL_MATH = 3



MT5_MODEL_REAL_TICKS = 4











class MT5StrategyTesterGlobalLock:



    """Cross-process lock for the MT5 Strategy Tester transport namespace.







    The portable runner isolates files, but the bundled MetaTrader tester still



    binds its local agent/MCP endpoints to fixed loopback ports.  Two portable



    terminals can therefore corrupt each other's bridge even when their



    runner directories are different.  Keep the lock outside every candidate



    runner so sibling processes coordinate on one workspace-wide resource.



    """







    def __init__(self, path: str | Path):



        self.path = Path(path).expanduser().resolve()



        self._handle = None



        self._mutex_handle = None



        self._kernel32 = None



        self.waited_seconds = 0.0







    def __enter__(self) -> "MT5StrategyTesterGlobalLock":



        self.path.parent.mkdir(parents=True, exist_ok=True)



        self.path.touch(exist_ok=True)



        self._handle = self.path.open("r+b")



        started = time.monotonic()



        if os.name == "nt":



            # msvcrt.locking is not a reliable inter-process primitive for



            # Python's independently opened handles on this Windows build.



            # A named kernel mutex is owned by the OS and survives that CRT



            # handle difference, so it is the authoritative MT5 gate here.



            import ctypes



            from ctypes import wintypes







            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)



            kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]



            kernel32.CreateMutexW.restype = wintypes.HANDLE



            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]



            kernel32.WaitForSingleObject.restype = wintypes.DWORD



            kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]



            kernel32.ReleaseMutex.restype = wintypes.BOOL



            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]



            kernel32.CloseHandle.restype = wintypes.BOOL



            # Keep one mutex for the host's bundled tester.  Windows command



            # line quoting can present an absolute path with a different



            # number of backslashes to a sibling process; deriving the kernel



            # object name from that spelling would silently split the lock.



            # A single host-level mutex is conservative and still correct



            # when callers provide separate metadata paths.



            mutex_name = "Local\\SelfGrowingTrader_MT5_StrategyTester"



            handle = kernel32.CreateMutexW(None, False, mutex_name)



            if not handle:



                self._handle.close()



                self._handle = None



                raise ctypes.WinError(ctypes.get_last_error())



            self._kernel32 = kernel32



            self._mutex_handle = handle



            wait_result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)



            if wait_result not in (0x00000000, 0x00000080):  # WAIT_OBJECT_0/ABANDONED



                kernel32.CloseHandle(handle)



                self._mutex_handle = None



                self._kernel32 = None



                self._handle.close()



                self._handle = None



                raise RuntimeError(f"MT5 global mutex wait failed: {wait_result}")



        else:



            import fcntl







            while True:



                try:



                    self._handle.seek(0)



                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)



                    break



                except (BlockingIOError, OSError):



                    time.sleep(0.25)



        self.waited_seconds = max(0.0, time.monotonic() - started)



        self._handle.seek(0)



        self._handle.truncate()



        owner = {



            "pid": os.getpid(),



            "acquired_at": time.time(),



        }



        self._handle.write(json.dumps(owner, sort_keys=True).encode("utf-8"))



        self._handle.flush()



        return self







    def __exit__(self, exc_type, exc_value, traceback) -> None:



        if self._handle is None:



            return



        try:



            if os.name == "nt":



                if self._kernel32 is not None and self._mutex_handle is not None:



                    self._kernel32.ReleaseMutex(self._mutex_handle)



                    self._kernel32.CloseHandle(self._mutex_handle)



                self._mutex_handle = None



                self._kernel32 = None



            else:



                import fcntl







                self._handle.seek(0)



                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)



        finally:



            self._handle.close()



            self._handle = None











@dataclass(frozen=True)



class MT5BacktestConfig:



    runner_dir: Path = DEFAULT_RUNNER_DIR



    # None uses the workspace-wide lock.  A custom path is useful when two



    # independent MT5 installations are intentionally run on separate hosts.



    mt5_global_lock_path: str | Path | None = None



    expert: str = DEFAULT_EXPERT



    symbol: str = DEFAULT_SYMBOL



    period: str = DEFAULT_PERIOD



    from_date: str = DEFAULT_FROM_DATE



    to_date: str = DEFAULT_TO_DATE



    deposit: float = 10000.0



    leverage: str = "1:200"



    timeout_sec: int = 300



    max_wall_sec: int = 0



    # Fail fast when the portable terminal is alive but its account/server



    # startup failed before any EA bridge request can be produced.



    mt5_startup_failure_grace_sec: float = 20.0



    # Ask the EA/Strategy Tester to finalize normally after N completed bridge



    # packets. Unlike max_wall_sec, this produces a valid terminal report.



    max_decision_packets: int = 0



    max_positions: int = 5



    magic_number: int = 123456



    # Strategy target/reporting threshold; not an execution entry gate.



    min_rr: float = 2.0



    # Minimal execution reward/risk floor mirrored into the EA.



    technical_min_rr: float = 1.2



    # Fixed M1-ATR geometry, shared exactly with the EA. Rows 13/14 are never



    # read for order construction.



    fixed_sl_atr: float = 5.0



    fixed_tp_atr: float = 10.0



    trail_activation_atr: float = 1.40



    trail_lock_atr: float = 0.45



    max_holding_bars: int = 120



    # Delayed M1 mark used only to score a policy exit against holding the



    # same position. Entry reward remains the final lifecycle R.



    exit_counterfactual_horizon_bars: int = 12



    # Closed-M1 horizon used only to label one cached HOLD/entry decision's



    # seven counterfactual choices. It is never read by inference/execution.



    entry_preference_horizon_bars: int = 48



    entry_abstraction_horizon_bars: int = 48



    # Candidate-only coverage repair. Entry-only collection is retained as



    # the compatibility default for old configs and artifacts.



    entry_preference_include_hold_snapshots: bool = False



    # Zero means no cadence quota/cap. Frequency is not a learning target.



    min_daily_trades: int = 0



    max_daily_trades: int = 0



    # Keep the flat-idle starvation signal configurable for training. It only



    # shapes bounded exploration pressure; it never removes legal actions.



    suppress_flat_idle_starvation: bool = True



    bridge_file: str = DEFAULT_BRIDGE_FILE



    request_file: str = DEFAULT_REQUEST_FILE



    action_file: str = DEFAULT_ACTION_FILE



    execution_file: str = DEFAULT_EXECUTION_FILE



    entry_preference_file: str = DEFAULT_ENTRY_PREFERENCE_FILE



    # UI/CLI writes this workspace-side request file. The bridge mirrors it



    # into the portable runner so the EA can call TesterStop() at the next



    # causal decision boundary and still emit an authoritative report.



    stop_request_file: str = ""



    stop_request_marker_file: str = DEFAULT_STOP_REQUEST_MARKER_FILE



    action_wait_ms: int = 15000



    entry_offset_atr: float = 5.0



    # Complete execution envelope mirrored into MT5. Do not decode outputs



    # with local defaults; source-of-truth remains AppConfig.



    initial_equity: float = 10000.0



    min_risk_pct: float = 0.001



    max_risk_pct: float = 0.02



    max_open_risk_pct: float = 0.05



    ruin_equity_pct: float = 0.50



    # Disabled by default so strategy-level caps cannot stop an active



    # learning run.  Broker validity checks and the five-position bridge cap



    # remain mandatory.



    enforce_entry_limits: bool = False



    fixed_risk_pct: float | None = None



    pending_expiry_bars: int = 60



    max_spread_atr: float = 8.0



    max_spread_cost_r: float = 0.15



    shock_range_atr: float = 12.0



    reset_execution_ledger_on_init: bool = False



    require_hedging_account: bool = True



    # This is a per-run identity recorded in the action/ledger contract. It



    # prevents restart recovery from silently treating a prior run as current.



    run_id: str = ""



    # Full canonical feature configuration sent to the EA.



    feature_atr_period: int = 14



    feature_ema_fast: int = 20



    feature_ema_slow: int = 60



    feature_range_window: int = 60



    feature_regime_window: int = 90



    feature_atr_percentile_window: int = 240



    feature_wyckoff_range_window: int = 120



    feature_wyckoff_volume_short_window: int = 20



    feature_wyckoff_volume_long_window: int = 60



    feature_wyckoff_event_horizon: int = 60



    # Must match ``FeatureConfig.mtf_scaling_mode`` and the Python parity



    # reference.  The v2 mode prevents higher-timeframe inputs from being



    # expressed in M1 ATR units and then saturating at the normalizer tail.



    feature_mtf_scaling_mode: str = "legacy_m1_atr_v1"



    # Explicit server→UTC conversion used for EA session features/audit time.



    feature_server_utc_offset_hours: int = 0



    feature_schema_hash: str = ""



    # Bridge v9+ feature audit is raw by contract. Policy normalization is



    # applied once in ``_normalize_request_inputs`` after raw sidecar capture.



    feature_audit_inputs_normalized: bool = False



    input_normalization_mode: str = "raw"



    # Internal policy contract recorded with the MT5 run.  50 preserves the



    # v13 STP behavior; candidates may lower it to avoid tail amplification.



    stp_input_clip: float = 50.0



    output_normalization_mode: str = "raw"



    output_normalization_target_rms: float = 0.70



    output_normalization_clip: float = 3.0



    feature_audit_file: str = DEFAULT_FEATURE_AUDIT_FILE



    # Python reference data used to prove numerical parity, not just a wire



    # width/hash contract. The evaluator fails promotion when the reference



    # is unavailable or an exported MT5 row deviates beyond tolerance.



    feature_reference_csv: str = ""



    feature_reference_datetime_column: str = "Datetime"



    symbol_point_size: float = 0.01



    symbol_spread_in_points: bool = True



    feature_parity_atol: float = 1e-6



    feature_parity_rtol: float = 1e-5



    visual: bool = False



    shutdown_terminal: bool = True



    # A completed tester report with no action-bridge request means the



    # Python policy was never evaluated (for example a non-visual tester can



    # consume its whole window before the file bridge is serviced).  That is



    # a runtime failure, not a valid zero-trade candidate result.



    require_action_bridge: bool = True



    # Every-tick is the generic diagnostic default. Promotion additionally



    # requires model 4 (Every tick based on real ticks) and fresh report proof.



    model: int = MT5_MODEL_EVERY_TICK



    strict_model1_parity: bool = False



    output_wire_decimals: int = 5



    offline_replay: bool = False



    offline_preflight: bool = False



    learning_enabled: bool = True



    # Candidate-only RLHF policy phase. The worker loads a frozen reward model



    # and updates the in-memory candidate through the policy's sole optimizer;



    # it never trains the reward model or authorizes promotion.



    rlhf_policy_update_enabled: bool = False



    codex_teacher_enabled: bool = False



    codex_teacher_model: str = "gpt-5.6-luna"



    codex_teacher_reasoning_effort: str = "medium"



    codex_teacher_executable: str = ""



    codex_teacher_cli_version: str = ""



    codex_teacher_preflight: dict[str, object] | None = None



    # Zero means every eligible finalized entry lifecycle in this run.



    codex_teacher_max_cases_per_run: int = 0



    codex_teacher_timeout_sec: int = 120



    # Bounded causal barrier between a finalized close and the next sampled



    # action. A positive value waits for the first verified Codex optimizer



    # update; zero keeps the explicit asynchronous/telemetry-only mode.



    codex_teacher_online_wait_sec: float = 120.0



    codex_teacher_max_workers: int = 2



    codex_teacher_reward_scale: float = 0.25



    codex_teacher_min_confidence: float = 0.55



    codex_teacher_max_probability_shift: float = 0.05



    codex_teacher_learning_rate: float = 0.02



    codex_context_similarity_threshold: float = 0.90



    codex_context_support_floor: float = 0.50



    codex_global_action_bias_scale: float = 0.50



    # Require the structured hypothesis -> experiment -> verification



    # contract before a Codex receipt may reach the optimizer.



    codex_teacher_hypothesis_required: bool = False



    rlhf_reward_model_path: str = ""



    rlhf_reward_feedback_mode: str = "human"



    rlhf_teacher_checkpoint_sha256: str = ""



    rlhf_require_independent_teacher: bool = False



    rlhf_source_checkpoint_sha256: str = ""



    rlhf_update_interval_packets: int = 1



    rlhf_max_updates_per_run: int = 0



    # Optional callback owned by the training facade. It is invoked on the



    # bridge thread after an accepted Codex update, so persistence never races



    # the optimizer mutation. The callback must be best-effort and return a



    # JSON-safe status mapping.



    codex_teacher_recovery_snapshot_hook: Callable[[object, int], Mapping[str, object]] | None = None



    # Historical failure-context Strategy Tester session. MT5 remains the



    # environment; Python only supplies the current policy.  The EA gates the



    # tester to an intraday replay window and stops after the horizon.



    mt5_failure_replay_mode: bool = False



    mt5_failure_replay_case_id: str = ""



    mt5_failure_replay_anchor_time: str = ""



    mt5_failure_replay_warmup_minutes: int = 120



    mt5_failure_replay_horizon_minutes: int = 180



    enable_execution_bootstrap: bool = False



    require_real_ticks_for_promotion: bool = True



    # Model=4 alone is an intent, not proof. Require a fresh tester report



    # carrying a real-tick marker before promotion can claim tick fidelity.



    require_real_tick_evidence_for_promotion: bool = True



    # Reconciliation can include terminal mark-to-market PnL. Promotion



    # requires a genuinely flat tester terminal so every credited lifecycle is



    # backed by a real final close, not only a synthetic terminal mark.



    require_terminal_flat_for_promotion: bool = True



    fail_on_stale_ex5: bool = True



    expected_ea_source_sha256: str = ""



    expected_ea_binary_sha256: str = ""



    experience_journal: str | None = "outputs/mt5_experience_journal.jsonl"



    experience_journal_max_bytes: int = 64 * 1024 * 1024



    # Separate intent -> receipt -> lifecycle audit.  This is never replayed:



    # pending/rejected MT5 intents must stay out of training memory.



    execution_trace_observatory: str | None = None



    execution_trace_observatory_max_bytes: int = 64 * 1024 * 1024



    execution_trace_observatory_include_recurrent_state: bool = True



    # During MT5 training, an unseeded policy can otherwise remain trapped in



    # HOLD/close/cancel or unfilled pending actions and receive no execution



    # feedback.  These controls only affect the bridge evaluator, never a



    # standalone policy call or a live trading integration.



    execution_bootstrap_after: int = 8



    execution_bootstrap_interval: int = 4



    train_entry_probe_enabled: bool = False



    train_entry_probe_after_decisions: int = 12



    train_entry_probe_interval_decisions: int = 6



    train_entry_probe_max_per_day: int = 2



    train_entry_probe_max_per_run: int = 6



    train_entry_probe_min_direction_quality: float = 0.75



    pending_cancel_after_decisions: int = 8



    pending_cancel_interval_decisions: int = 4



    entry_guard_single_position: bool = False



    entry_loss_guard_enabled: bool = False



    entry_loss_guard_min_closed: int = 6



    entry_loss_guard_max_closed_loss_r: float = 0.25



    entry_loss_guard_consecutive_losses: int = 4



    runtime_fitness_guard_enabled: bool = True



    runtime_fitness_rolling_window: int = 50



    runtime_fitness_min_closed: int = 20



    runtime_fitness_min_score: float = -0.30



    runtime_fitness_max_immediate_exit_ratio: float = 0.60



    runtime_fitness_max_daily_loss_r: float = 3.0



    runtime_fitness_max_run_loss_r: float = 8.0



    runtime_fitness_max_daily_trades: int = 300



    runtime_fitness_max_drawdown_pct: float = 20.0



    # Five-agent competitive curriculum.  One schedule step is one atomic



    # five-row bridge transition, never one individual agent receipt.



    five_training_schedule_steps: int = 10_000



    # Reward V2 immutable-close statistics.  These thresholds must match the



    # active policy profile so MT5 and the Python environment shape the same



    # finalized trade outcome.



    five_reward_expectancy_floor_r: float = 0.05



    five_reward_tail_loss_threshold_r: float = 1.0



    five_reward_tail_loss_cap_r: float = 3.0



    five_lifecycle_reward_enabled: bool = True



    five_lifecycle_shape_clip_r: float = 0.75



    five_lifecycle_min_sl_atr: float = 1.5



    five_lifecycle_min_tp_atr: float = 3.0



    five_lifecycle_geometry_penalty_r: float = 0.30



    five_lifecycle_geometry_bonus_r: float = 0.05



    five_lifecycle_mfe_penalty_zero_r: float = 0.40



    five_lifecycle_mfe_penalty_010_r: float = 0.30



    five_lifecycle_mfe_penalty_025_r: float = 0.15



    five_lifecycle_mfe_penalty_neutral_at_r: float = 0.50



    five_lifecycle_excursion_bonus_at_r: float = 1.0



    five_lifecycle_excursion_good_mae_r: float = 0.25



    five_lifecycle_excursion_bad_mae_r: float = 1.0



    five_lifecycle_excursion_min_bonus_r: float = 0.15



    five_lifecycle_excursion_max_bonus_r: float = 0.30



    five_lifecycle_trailing_activation_mfe_r: float = 0.50



    five_lifecycle_trailing_capture_min_ratio: float = 0.50



    five_lifecycle_trailing_capture_target_ratio: float = 0.80



    five_lifecycle_trailing_missed_penalty_r: float = 0.20



    five_lifecycle_trailing_small_capture_penalty_r: float = 0.10



    five_lifecycle_trailing_min_bonus_r: float = 0.10



    five_lifecycle_trailing_max_bonus_r: float = 0.25



    five_lifecycle_exit_capture_min_ratio: float = 0.25



    five_lifecycle_exit_capture_target_ratio: float = 0.80



    five_lifecycle_exit_capture_min_bonus_r: float = 0.10



    five_lifecycle_exit_capture_max_bonus_r: float = 0.30



    five_lifecycle_recovery_min_bonus_r: float = 0.05



    five_lifecycle_recovery_max_bonus_r: float = 0.15



    five_lifecycle_pending_fill_max_bars: int = 5



    five_lifecycle_pending_fill_min_bonus_r: float = 0.05



    five_lifecycle_pending_fill_max_bonus_r: float = 0.10



    five_invalid_position_action_no_position_penalty_r: float = 1.0



    five_invalid_cancel_no_pending_penalty_r: float = 2.0



    five_unfilled_pending_timeout_penalty_r: float = 0.25



    # A continuation checkpoint is captured before a one-action collapse can



    # overwrite a useful online policy.  The snapshot is only a research



    # fallback; promotion still requires the normal frozen/OOS gates.



    five_champion_min_lifecycles: int = 16



    five_champion_min_net_r: float = 0.0



    # Prevent a newly filled position from being closed by the policy on the



    # first bridge ticks. EA-side SL/TP and hard risk controls remain active.



    min_manual_close_age_bars: int = 2



    # Reconciliation fails closed when the report and finalised policy ledger



    # materially disagree. Values are intentionally explicit/run-configurable.



    reconciliation_pnl_relative_tolerance: float = 0.15



    reconciliation_pnl_absolute_tolerance: float = 5.0



    # Runtime global cu da bi loai bo. Bridge chi xu ly owner context.



    runtime_context_owner: object | None = None



    continuous_runtime: object | None = None



    # Regime Detector configuration



    regime_detector_path: str | Path | None = None



    regime_detector_enabled: bool = True



    require_regime_detector: bool = False



    regime_bull_threshold: float = 0.35



    regime_bear_threshold: float = 0.35



    regime_chop_threshold: float = 0.38



    diagnostics_path: str | Path | None = None



    events_path: str | Path | None = None







    def __post_init__(self) -> None:



        if int(self.max_positions) != 0 and not 1 <= int(self.max_positions) <= 5:



            raise ValueError(



                f"MT5 bridge v{BRIDGE_SCHEMA_VERSION} requires max_positions=0 "



                f"(unlimited) or 1..5; its {INPUT_COUNT}-input schema keeps five observation slots"



            )



        if int(self.max_wall_sec) < 0:



            raise ValueError("max_wall_sec must be non-negative")



        if int(self.max_decision_packets) < 0:



            raise ValueError("max_decision_packets must be non-negative")



        if not 1 <= int(self.entry_preference_horizon_bars) <= 64:



            raise ValueError("entry_preference_horizon_bars must be within [1, 64]")



        if not int(self.entry_preference_horizon_bars) <= int(self.entry_abstraction_horizon_bars) <= 600:



            raise ValueError(



                "entry_abstraction_horizon_bars must be within [entry_preference_horizon_bars, 600]"



            )



        if int(self.execution_trace_observatory_max_bytes) < 0:



            raise ValueError("execution_trace_observatory_max_bytes must be non-negative")



        if int(self.rlhf_update_interval_packets) < 1:



            raise ValueError("rlhf_update_interval_packets must be positive")



        if int(self.rlhf_max_updates_per_run) < 0:



            raise ValueError("rlhf_max_updates_per_run must be non-negative")



        if str(self.rlhf_reward_feedback_mode or "").strip().lower() not in {



            "human", "rlaif"



        }:



            raise ValueError("rlhf_reward_feedback_mode must be human or rlaif")



        if bool(self.rlhf_policy_update_enabled):



            if not bool(self.learning_enabled):



                raise ValueError("RLHF policy updates require learning_enabled=true")



            if not str(self.rlhf_reward_model_path or "").strip():



                raise ValueError("RLHF policy updates require a reward model path")



            if not str(self.rlhf_source_checkpoint_sha256 or "").strip():



                raise ValueError("RLHF policy updates require a source checkpoint hash")



            if bool(self.rlhf_require_independent_teacher):



                if str(self.rlhf_reward_feedback_mode or "").strip().lower() != "rlaif":



                    raise ValueError("independent RLHF teacher mode requires feedback_mode=rlaif")



                if not str(self.rlhf_teacher_checkpoint_sha256 or "").strip():



                    raise ValueError("independent RLHF teacher mode requires a teacher checkpoint hash")



        if bool(self.codex_teacher_enabled):



            if not bool(self.learning_enabled):



                raise ValueError("Codex teacher requires learning_enabled=true")



            if not str(self.rlhf_source_checkpoint_sha256 or "").strip():



                raise ValueError("Codex teacher requires a source checkpoint hash")



            if int(self.codex_teacher_max_cases_per_run) < 0:



                raise ValueError("Codex teacher case limit must be non-negative")



            if not 0.0 <= float(self.codex_teacher_online_wait_sec) <= 300.0:



                raise ValueError("Codex teacher online wait must be within [0, 300]")



            if not 1 <= int(self.codex_teacher_max_workers) <= 8:



                raise ValueError("Codex teacher max workers must be within [1, 8]")



            if not 0.0 <= float(self.codex_teacher_min_confidence) <= 1.0:



                raise ValueError("Codex teacher min confidence must be in [0,1]")



            if not 0.0 < float(self.codex_teacher_reward_scale) <= 0.5:



                raise ValueError("Codex teacher reward scale must be in (0,0.5]")



        if not 0.0 < float(self.codex_teacher_max_probability_shift) <= 0.25:



            raise ValueError("Codex teacher max probability shift must be in (0,0.25]")



        if not 0.0 < float(self.codex_teacher_learning_rate) <= 0.02:



            raise ValueError("Codex teacher learning rate must be in (0,0.02]")



        if not 0.90 <= float(self.codex_context_similarity_threshold) < 1.0:



            raise ValueError("Codex context similarity threshold must be in [0.90,1)")



        if not 0.0 <= float(self.codex_context_support_floor) <= 1.0:



            raise ValueError("Codex context support floor must be in [0,1]")



        if not 0.0 <= float(self.codex_global_action_bias_scale) <= 0.5:



            raise ValueError("Codex global action bias scale must be in [0,0.5]")



        if not isinstance(self.codex_teacher_hypothesis_required, bool):



            raise ValueError("Codex teacher hypothesis requirement must be boolean")



        if str(self.codex_teacher_reasoning_effort).strip().lower() not in {



            "none", "minimal", "low", "medium", "high", "xhigh", "max"



        }:



            raise ValueError("Codex teacher reasoning effort is invalid")



        feature_periods = (



            self.feature_atr_period, self.feature_ema_fast, self.feature_ema_slow,



            self.feature_range_window, self.feature_regime_window,



            self.feature_atr_percentile_window, self.feature_wyckoff_range_window,



            self.feature_wyckoff_volume_short_window,



            self.feature_wyckoff_volume_long_window, self.feature_wyckoff_event_horizon,



        )



        if min(int(value) for value in feature_periods) < 1:



            raise ValueError("MT5 feature periods must be positive")



        if int(self.feature_wyckoff_volume_short_window) >= int(self.feature_wyckoff_volume_long_window):



            raise ValueError("Wyckoff short volume window must be lower than long volume window")



        if float(self.deposit) <= 0.0 or float(self.initial_equity) <= 0.0:



            raise ValueError("deposit and initial_equity must be positive")



        if not (0.0 < float(self.min_risk_pct) <= float(self.max_risk_pct) <= 1.0):



            raise ValueError("invalid risk bounds")



        if not float(self.max_risk_pct) <= float(self.max_open_risk_pct) <= 1.0:



            raise ValueError("max_open_risk_pct must be within [max_risk_pct,1]")



        if (float(self.min_rr) <= 0.0 or float(self.technical_min_rr) <= 0.0



                or float(self.technical_min_rr) > float(self.min_rr)



                or float(self.fixed_sl_atr) <= 0.0 or float(self.fixed_tp_atr) <= 0.0



                or float(self.fixed_tp_atr) / float(self.fixed_sl_atr) < float(self.technical_min_rr)):



            raise ValueError("min_rr/technical_min_rr are invalid")







        if self.fixed_risk_pct is not None and not (float(self.min_risk_pct) <= float(self.fixed_risk_pct) <= float(self.max_risk_pct)):



            raise ValueError("fixed_risk_pct must be within risk bounds")



        if not 0.0 < float(self.ruin_equity_pct) <= 1.0 or int(self.pending_expiry_bars) < 1:



            raise ValueError("invalid ruin threshold or pending expiry")



        if min(float(self.max_spread_atr), float(self.max_spread_cost_r), float(self.shock_range_atr)) <= 0.0:



            raise ValueError("execution filters must be positive")



        if int(self.max_daily_trades) > 0 and int(self.min_daily_trades) > int(self.max_daily_trades):



            raise ValueError("min_daily_trades cannot exceed max_daily_trades when a cap is enabled")



        if str(self.input_normalization_mode).strip().lower() not in {"raw", "tanh_v1", "soft_clip_v1", "groupwise_v2", "groupwise_mtf_v3", "structure_aware_v4"}:



            raise ValueError("input_normalization_mode must be raw, tanh_v1, soft_clip_v1, groupwise_v2, groupwise_mtf_v3, or structure_aware_v4")



        if str(self.feature_mtf_scaling_mode).strip().lower() not in {"legacy_m1_atr_v1", "native_tf_atr_v2"}:



            raise ValueError("feature_mtf_scaling_mode must be legacy_m1_atr_v1 or native_tf_atr_v2")



        if not 0.1 <= float(self.stp_input_clip) <= 50.0:



            raise ValueError("stp_input_clip must be within [0.1, 50]")



        if str(self.output_normalization_mode).strip().lower() not in {"raw", "rms_v1", "layernorm_v1"}:



            raise ValueError("output_normalization_mode must be raw, rms_v1, or layernorm_v1")



        if not 1e-6 <= float(self.output_normalization_target_rms) <= 10.0:



            raise ValueError("output_normalization_target_rms must be within (0, 10]")



        if not 1e-6 <= float(self.output_normalization_clip) <= 20.0:



            raise ValueError("output_normalization_clip must be within (0, 20]")



        if not 3 <= int(self.output_wire_decimals) <= 10:



            raise ValueError("output_wire_decimals must be within [3, 10]")



        if int(self.entry_loss_guard_min_closed) < 1:



            raise ValueError("entry_loss_guard_min_closed must be at least 1")



        if float(self.entry_loss_guard_max_closed_loss_r) < 0.0:



            raise ValueError("entry_loss_guard_max_closed_loss_r must be non-negative")



        if int(self.entry_loss_guard_consecutive_losses) < 1:



            raise ValueError("entry_loss_guard_consecutive_losses must be at least 1")



        if int(self.pending_cancel_after_decisions) < 1:



            raise ValueError("pending_cancel_after_decisions must be at least 1")



        if int(self.pending_cancel_interval_decisions) < 1:



            raise ValueError("pending_cancel_interval_decisions must be at least 1")



        if int(self.train_entry_probe_after_decisions) < 1:



            raise ValueError("train_entry_probe_after_decisions must be at least 1")



        if int(self.train_entry_probe_interval_decisions) < 1:



            raise ValueError("train_entry_probe_interval_decisions must be at least 1")



        if int(self.train_entry_probe_max_per_day) < 0:



            raise ValueError("train_entry_probe_max_per_day must be non-negative")



        if int(self.train_entry_probe_max_per_run) < 0:



            raise ValueError("train_entry_probe_max_per_run must be non-negative")



        if not 0.0 <= float(self.train_entry_probe_min_direction_quality) <= 1.0:



            raise ValueError("train_entry_probe_min_direction_quality must be within [0, 1]")



        if int(self.runtime_fitness_rolling_window) < 1:



            raise ValueError("runtime_fitness_rolling_window must be at least 1")



        if int(self.runtime_fitness_min_closed) < 1:



            raise ValueError("runtime_fitness_min_closed must be at least 1")



        if not -5.0 <= float(self.runtime_fitness_min_score) <= 5.0:



            raise ValueError("runtime_fitness_min_score must be within [-5, 5]")



        if not 0.0 <= float(self.runtime_fitness_max_immediate_exit_ratio) <= 1.0:



            raise ValueError("runtime_fitness_max_immediate_exit_ratio must be within [0, 1]")



        if float(self.runtime_fitness_max_daily_loss_r) < 0.0:



            raise ValueError("runtime_fitness_max_daily_loss_r must be non-negative")



        if float(self.runtime_fitness_max_run_loss_r) < 0.0:



            raise ValueError("runtime_fitness_max_run_loss_r must be non-negative")



        if int(self.runtime_fitness_max_daily_trades) < 0:



            raise ValueError("runtime_fitness_max_daily_trades must be non-negative")



        if not 0.0 < float(self.runtime_fitness_max_drawdown_pct) <= 100.0:



            raise ValueError("runtime_fitness_max_drawdown_pct must be within (0, 100]")



        if int(self.five_training_schedule_steps) < FIVE_LSTM_UNROLL:



            raise ValueError(



                "five_training_schedule_steps must be at least one "



                f"{FIVE_LSTM_UNROLL}-step rollout"



            )



        if min(



            float(self.five_reward_expectancy_floor_r),



            float(self.five_reward_tail_loss_threshold_r),



            float(self.five_reward_tail_loss_cap_r),



            *(



                float(getattr(self, f"five_lifecycle_{name}"))



                for name in FiveLifecycleRewardConfig().__dataclass_fields__



                if name != "enabled"



            ),



            float(self.five_invalid_position_action_no_position_penalty_r),



            float(self.five_invalid_cancel_no_pending_penalty_r),



            float(self.five_unfilled_pending_timeout_penalty_r),



        ) < 0.0:



            raise ValueError("five reward thresholds must be non-negative; action-contract penalty must also be non-negative")



        lifecycle = FiveLifecycleRewardConfig.from_source(self)



        if lifecycle.excursion_bad_mae_r <= lifecycle.excursion_good_mae_r:



            raise ValueError("five lifecycle bad MAE must exceed good MAE")



        if not 0.0 <= lifecycle.trailing_capture_min_ratio < lifecycle.trailing_capture_target_ratio <= 1.0:



            raise ValueError("five lifecycle trailing ratios are invalid")



        if not 0.0 <= lifecycle.exit_capture_min_ratio < lifecycle.exit_capture_target_ratio <= 1.0:



            raise ValueError("five lifecycle exit-capture ratios are invalid")



        if float(self.five_invalid_position_action_no_position_penalty_r) > 5.0:



            raise ValueError("five invalid position-action penalty must be within [0, 5]R")



        if float(self.five_invalid_cancel_no_pending_penalty_r) > 5.0:



            raise ValueError("five invalid cancel-without-pending penalty must be within [0, 5]R")



        if float(self.five_unfilled_pending_timeout_penalty_r) > 5.0:



            raise ValueError("five unfilled pending-timeout penalty must be within [0, 5]R")



        if int(self.mt5_failure_replay_warmup_minutes) < 1 or int(self.mt5_failure_replay_warmup_minutes) > 1440:



            raise ValueError("mt5_failure_replay_warmup_minutes must be within [1, 1440]")



        if int(self.mt5_failure_replay_horizon_minutes) < 1 or int(self.mt5_failure_replay_horizon_minutes) > 2880:



            raise ValueError("mt5_failure_replay_horizon_minutes must be within [1, 2880]")



        if bool(self.mt5_failure_replay_mode):



            if not str(self.mt5_failure_replay_case_id or "").strip():



                raise ValueError("mt5_failure_replay_case_id is required in replay mode")



            if not str(self.mt5_failure_replay_anchor_time or "").strip():



                raise ValueError("mt5_failure_replay_anchor_time is required in replay mode")



        if int(self.model) not in {0, 1, 2, 3, 4}:



            raise ValueError("unsupported MT5 tester model")



        for field_name in ("expected_ea_source_sha256", "expected_ea_binary_sha256"):



            value = str(getattr(self, field_name, "") or "").strip()



            if value and not re.fullmatch(r"[0-9a-fA-F]{64}", value):



                raise ValueError(f"{field_name} must be a 64-character SHA256 hex digest")



        if min(int(self.feature_atr_period), int(self.feature_ema_fast), int(self.feature_ema_slow), int(self.feature_range_window), int(self.feature_atr_percentile_window)) < 1:



            raise ValueError("feature periods must be positive")



        if int(self.feature_ema_fast) >= int(self.feature_ema_slow):



            raise ValueError("feature_ema_fast must be lower than feature_ema_slow")



        if float(self.symbol_point_size) <= 0.0:



            raise ValueError("symbol_point_size must be positive")



        if not -23 <= int(self.feature_server_utc_offset_hours) <= 23:



            raise ValueError("feature_server_utc_offset_hours must be within [-23, 23]")



        if min(float(self.feature_parity_atol), float(self.feature_parity_rtol)) < 0.0:



            raise ValueError("feature parity tolerances must be non-negative")











@dataclass(frozen=True)



class MT5PositionLifecycle:



    """One finalised position lifecycle reconstructed from the EA ledger.







    Partial closes are aggregated by ``position_ticket``.  A lifecycle is



    eligible for position-level fitness only after the EA marks its final



    close, preventing multiple deals from inflating the statistical sample.



    """







    position_ticket: str



    entry_time: datetime | None



    exit_time: datetime | None



    net_pnl: float



    gross_profit: float



    commission: float



    swap: float



    initial_risk: float = 0.0



    realized_r: float | None = None



    duration_bars: float | None = None



    close_reason: str = ""



    source: str = ""



    trace_id: str = ""



    entry_sequence: int = 0



    entry_count: int = 0



    close_count: int = 0



    final_close_confirmed: bool = False



    # V19 causal contract. These fields are optional because legacy EA



    # ledgers cannot reconstruct a full trajectory. Their absence is explicit



    # in causal coverage instead of being silently filled with fake values.



    exit_mechanism: str = ""



    entry_price: float = 0.0



    exit_price: float = 0.0



    initial_sl_price: float = 0.0



    final_sl_price: float = 0.0



    initial_tp_price: float = 0.0



    final_tp_price: float = 0.0



    partial_close_count: int = 0



    mfe_r: float = 0.0



    mae_r: float = 0.0



    # True only when explicit ledger values or a causal closed-bar OHLC path



    # proves MFE/MAE. A trajectory row alone is not excursion evidence.



    excursion_available: bool = False



    giveback_r: float = 0.0



    near_tp_ratio: float = 0.0



    slippage_r: float = 0.0



    post_exit_mfe_r: float | None = None



    post_exit_mae_r: float | None = None



    returned_to_entry: bool | None = None



    original_tp_hit_after_exit: bool | None = None



    exit_kind: str = ""



    exit_controller: str = ""



    exit_trace_id: str = ""



    exit_requested_action: int = -1



    post_exit_horizon_bars: int | None = None



    post_exit_hold_r: float | None = None



    # V11 continuous entry-geometry evidence. ``trajectory`` remains empty



    # for old ledgers; those lifecycles are intentionally ineligible for the



    # geometry loss instead of receiving an inferred target.



    initial_atr: float = 0.0



    side: int = 0



    proposed_sl_atr: float = 0.0



    proposed_rr: float = 0.0



    # V12 separates requested order geometry from the immutable broker-fill



    # geometry. A fill-price movement alone is not a broker SL/TP clamp.



    requested_entry_price: float = 0.0



    requested_sl_price: float = 0.0



    requested_tp_price: float = 0.0



    requested_atr: float = 0.0



    geometry_mutation_count: int = 0



    trailing_applied: bool = False



    pending_fill_bars: int | None = None



    trajectory: tuple[dict[str, object], ...] = ()



    metadata: dict[str, object] = field(default_factory=dict)



    # Runtime bridge mark captured after the final close was observed.  It is



    # intentionally separate from the immutable execution-ledger PnL so the



    # Codex packet can distinguish an account snapshot from a position result.



    account_context_after_close: dict[str, float] = field(default_factory=dict)



    # A final DEAL_ADD also carries an immutable cashflow for that deal.  This



    # is distinct from a balance/equity snapshot: several close transactions



    # can legitimately observe the same terminal snapshot, while their



    # deal-level cashflows remain separately attributable by deal ticket.



    account_cashflow_delta: float | None = None



    account_cashflow_deal_ticket: str = ""











@dataclass



class PositionContextExtremaState:



    ticket: str



    owner_context: str = ""



    owner_expert: str = ""



    owner_policy: str = ""



    owner_trace_id: str = ""



    side: int = 0



    entry: float = 0.0



    segment_context: str = ""



    segment_start_sequence: int = 0



    segment_start_time: str = ""



    high_r: float = -float("inf")



    high_sequence: int = 0



    high_time: str = ""



    high_observation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float), repr=False)



    high_position: dict[str, object] = field(default_factory=dict)



    high_context_audit: dict[str, object] = field(default_factory=dict)



    low_r: float = float("inf")



    low_sequence: int = 0



    low_time: str = ""



    low_observation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float), repr=False)



    low_position: dict[str, object] = field(default_factory=dict)



    low_context_audit: dict[str, object] = field(default_factory=dict)



    last_r: float = 0.0



    last_sequence: int = 0



    last_time: str = ""



    last_position: dict[str, object] = field(default_factory=dict)











@dataclass(frozen=True)



class MT5ExecutionSummary:



    """Risk-aware fitness inputs derived from ``self_growing_execution.csv``."""







    ledger_path: Path | None = None



    ledger_available: bool = False



    schema_version: int = 0



    total_events: int = 0



    policy_events: int = 0



    policy_intents: int = 0



    rejected_intents: int = 0



    lifecycle_count: int = 0



    complete_lifecycle_ratio: float = 0.0



    risk_coverage: float = 0.0



    trace_coverage: float = 0.0



    ledger_integrity: float = 0.0



    execution_failure_rate: float = 1.0



    immediate_exit_ratio: float = 0.0



    expectancy: float = 0.0



    expectancy_lcb: float = 0.0



    profit_factor_shrunk: float = 1.0



    cvar_5pct: float = 0.0



    net_cost_r: float = 0.0



    sample_unit: str = "none"



    lifecycles: tuple[MT5PositionLifecycle, ...] = ()







    @property



    def has_position_fitness(self) -> bool:



        return self.ledger_available and self.lifecycle_count > 0











@dataclass(frozen=True)



class MT5ReportLedgerReconciliation:



    """Cross-check the Strategy Tester report against finalised EA lifecycles.







    The HTML report remains the source for account-level PnL and drawdown,



    while the ledger is the source for action attribution. A run that contains



    policy lifecycles but a zero-trade report is operationally invalid, not a



    losing strategy. Keeping this distinction prevents a parser/transport



    defect from silently training the policy toward a false -100 reward.



    """







    available: bool = False



    reconciled: bool = True



    report_total_trades: int = 0



    ledger_lifecycle_count: int = 0



    report_net_profit: float = 0.0



    ledger_net_profit: float = 0.0



    terminal_open_positions: int = 0



    terminal_open_pnl: float = 0.0



    ledger_marked_net_profit: float = 0.0



    trade_count_consistent: bool = True



    pnl_consistent: bool = True



    reasons: tuple[str, ...] = ()



    # Optional report-side lifecycle evidence.  These fields distinguish a



    # native DEAL_ADD close from a Strategy Tester report fallback without



    # weakening the boolean reconciliation gate.



    report_lifecycle_count: int = 0



    report_lifecycle_net_profit: float = 0.0



    report_fallback_lifecycle_count: int = 0



    report_fallback_owner_resolved_count: int = 0











@dataclass



class RuntimeFitnessGuard:



    enabled: bool = True



    rolling_window: int = 50



    min_closed: int = 20



    min_score: float = -0.30



    max_immediate_exit_ratio: float = 0.60



    max_daily_loss_r: float = 3.0



    max_run_loss_r: float = 8.0



    max_daily_trades: int = 300



    max_drawdown_pct: float = 20.0



    initial_equity: float = 10000.0



    min_holding_bars: int = 2



    trades: deque[dict[str, float | bool | str]] = field(default_factory=deque)



    processed_keys: set[str] = field(default_factory=set)



    day: str = ""



    day_closed: int = 0



    day_net_r: float = 0.0



    run_closed: int = 0



    run_net_r: float = 0.0



    active_reason: str = ""



    last_score: float = 0.0



    last_immediate_ratio: float = 0.0



    last_trade_r: float = 0.0







    @staticmethod



    def _truthy(value: object) -> bool:



        return str(value or "").strip().lower() in {"1", "true", "yes", "ok", "success"}







    @staticmethod



    def _float(row: Mapping[str, object], *names: str, default: float = 0.0) -> float:



        for name in names:



            try:



                value = row.get(name)



            except AttributeError:



                value = None



            if value in {None, ""}:



                continue



            try:



                return float(value)



            except (TypeError, ValueError):



                continue



        return float(default)







    @staticmethod



    def _row_key(row: Mapping[str, object]) -> str:



        parts = [



            str(row.get("position_ticket", "") or "").strip(),



            str(row.get("deal_ticket", "") or "").strip(),



            str(row.get("ack_sequence", "") or row.get("sequence", "") or "").strip(),



            str(row.get("event_id", "") or "").strip(),



            str(row.get("trace_id", "") or "").strip(),



        ]



        key = "|".join(parts)



        return key if key.strip("|") else str(hash(tuple(sorted((str(k), str(v)) for k, v in row.items()))))







    @staticmethod



    def _row_day(row: Mapping[str, object], fallback_day: str = "") -> str:



        for name in ("close_time", "logged_at", "time", "timestamp"):



            value = str(row.get(name, "") or "").strip()



            if value:



                return value.split(" ", 1)[0].split("T", 1)[0]



        return str(fallback_day or "")







    def _reset_day_if_needed(self, day: str) -> None:



        if day and day != self.day:



            self.day = day



            self.day_closed = 0



            self.day_net_r = 0.0







    def record_execution_row(



        self,



        row: Mapping[str, object],



        *,



        fallback_day: str = "",



        drawdown_pct: float = 0.0,



    ) -> float | None:



        if not bool(self.enabled):



            return None



        if not self._truthy(row.get("closed")) or not self._truthy(row.get("position_final_close")):



            return None



        key = self._row_key(row)



        if key in self.processed_keys:



            return None



        self.processed_keys.add(key)



        day = self._row_day(row, fallback_day=fallback_day)



        self._reset_day_if_needed(day)







        net = self._float(row, "net_profit", "net_pnl", "profit_after_cost", "profit", default=0.0)



        risk = abs(self._float(row, "initial_risk", "risk_money", "risk_amount", default=0.0))



        if risk <= 0.0:



            risk = max(1.0, float(self.initial_equity) * 0.01)



        trade_r = float(np.clip(net / risk, -5.0, 5.0))



        age_bars = int(self._float(row, "age_bars", "holding_bars", default=999.0))



        close_reason = str(row.get("close_reason", "") or "").upper()



        immediate = bool(age_bars < int(self.min_holding_bars) or (close_reason == "EXPERT" and age_bars <= int(self.min_holding_bars)))



        penalty = 0.35 if immediate else 0.0



        if float(drawdown_pct) >= float(self.max_drawdown_pct):



            penalty += 0.50



        score = float(trade_r - penalty)



        self.trades.append({"score": score, "r": trade_r, "immediate": immediate, "day": day})



        while len(self.trades) > int(self.rolling_window):



            self.trades.popleft()



        self.day_closed += 1



        self.day_net_r += trade_r



        self.run_closed += 1



        self.run_net_r += trade_r



        self.last_trade_r = trade_r



        self._refresh(drawdown_pct=drawdown_pct)



        return trade_r







    def _refresh(self, *, drawdown_pct: float = 0.0) -> None:



        if float(drawdown_pct) >= float(self.max_drawdown_pct):



            self.active_reason = f"drawdown_pct:{float(drawdown_pct):.2f}>={float(self.max_drawdown_pct):.2f}"



            if not self.trades:



                self.last_score = 0.0



                self.last_immediate_ratio = 0.0



            return



        if not self.trades:



            self.last_score = 0.0



            self.last_immediate_ratio = 0.0



            self.active_reason = ""



            return



        scores = [float(item["score"]) for item in self.trades]



        immediate_count = sum(1 for item in self.trades if bool(item["immediate"]))



        self.last_score = float(np.mean(scores))



        self.last_immediate_ratio = immediate_count / float(len(self.trades))



        self.active_reason = ""



        if int(self.max_daily_trades) > 0 and int(self.day_closed) >= int(self.max_daily_trades):



            self.active_reason = f"daily_trades:{int(self.day_closed)}>={int(self.max_daily_trades)}"



        elif float(self.day_net_r) <= -float(self.max_daily_loss_r):



            self.active_reason = f"daily_loss_r:{float(self.day_net_r):.3f}<=-{float(self.max_daily_loss_r):.3f}"



        elif float(self.run_net_r) <= -float(self.max_run_loss_r):



            self.active_reason = f"run_loss_r:{float(self.run_net_r):.3f}<=-{float(self.max_run_loss_r):.3f}"



        elif len(self.trades) >= int(self.min_closed) and self.last_score < float(self.min_score):



            self.active_reason = f"rolling_fitness:{self.last_score:.3f}<{float(self.min_score):.3f}"



        elif len(self.trades) >= int(self.min_closed) and self.last_immediate_ratio > float(self.max_immediate_exit_ratio):



            self.active_reason = f"immediate_exit_ratio:{self.last_immediate_ratio:.3f}>{float(self.max_immediate_exit_ratio):.3f}"







    def blocks_entries(self, *, drawdown_pct: float = 0.0) -> bool:



        if not bool(self.enabled):



            self.active_reason = ""



            return False



        self._refresh(drawdown_pct=drawdown_pct)



        return bool(self.active_reason)







    def snapshot(self) -> dict[str, object]:



        return {



            "enabled": bool(self.enabled),



            "active": bool(self.active_reason),



            "reason": str(self.active_reason),



            "day": str(self.day),



            "closed_count": int(self.day_closed),



            "day_net_r": float(self.day_net_r),



            "run_closed": int(self.run_closed),



            "run_net_r": float(self.run_net_r),



            "rolling_count": int(len(self.trades)),



            "rolling_score": float(self.last_score),



            "immediate_exit_ratio": float(self.last_immediate_ratio),



            "last_trade_r": float(self.last_trade_r),



        }











def reconcile_mt5_report_and_ledger(



    metrics: "MT5BacktestMetrics",



    execution_summary: MT5ExecutionSummary | None,



    *,



    pnl_relative_tolerance: float = 0.15,



    pnl_absolute_tolerance: float = 5.0,



    terminal_open_positions: int = 0,



    terminal_open_pnl: float = 0.0,



    report_lifecycle_count: int | None = None,



    report_lifecycle_net_profit: float | None = None,



) -> MT5ReportLedgerReconciliation:



    """Return a conservative report/ledger reconciliation result.







    When parsed report lifecycles are available, they are the comparable unit



    and must agree exactly with the finalised ledger lifecycle count. This is



    stricter than the legacy summary-only path, where a report may expose deal



    counts rather than position counts and only the non-zero/count and PnL



    guards are available. Never let a loose PnL tolerance turn a different



    lifecycle set into a reconciled run.



    """



    summary = execution_summary



    if summary is None or not summary.ledger_available:



        return MT5ReportLedgerReconciliation()



    lifecycle_count = int(summary.lifecycle_count)



    ledger_net = float(sum(float(item.net_pnl) for item in summary.lifecycles))



    open_positions = max(0, int(terminal_open_positions))



    open_pnl = float(terminal_open_pnl) if np.isfinite(float(terminal_open_pnl)) else 0.0



    marked_ledger_net = ledger_net + (open_pnl if open_positions > 0 else 0.0)



    report_trades = int(metrics.total_trades)



    report_net = float(metrics.net_profit)



    parsed_report_count = (



        int(report_lifecycle_count)



        if report_lifecycle_count is not None



        else None



    )



    parsed_report_net = (



        float(report_lifecycle_net_profit)



        if report_lifecycle_net_profit is not None



        and np.isfinite(float(report_lifecycle_net_profit))



        else None



    )



    reasons: list[str] = []



    if parsed_report_count is not None:



        trade_ok = bool(parsed_report_count == lifecycle_count)



        if not trade_ok:



            reasons.append(



                "report_lifecycle_count_mismatch:"



                f"report={parsed_report_count},ledger={lifecycle_count}"



            )



    else:



        trade_ok = not (lifecycle_count > 0 and report_trades <= 0)



        if not trade_ok:



            reasons.append(f"report_trade_count_zero_with_ledger_lifecycles:{lifecycle_count}")



    comparable_report_net = (



        parsed_report_net if parsed_report_net is not None else report_net



    )



    tolerance = max(0.0, float(pnl_absolute_tolerance)) + max(



        0.0, float(pnl_relative_tolerance)



    ) * max(abs(marked_ledger_net), 1.0)



    pnl_ok = abs(comparable_report_net - marked_ledger_net) <= tolerance



    if not pnl_ok:



        reasons.append(



            f"report_ledger_pnl_delta:{comparable_report_net - marked_ledger_net:.6g}>tol:{tolerance:.6g}"



        )



    return MT5ReportLedgerReconciliation(



        available=True,



        reconciled=bool(trade_ok and pnl_ok),



        report_total_trades=report_trades,



        ledger_lifecycle_count=lifecycle_count,



        report_net_profit=report_net,



        ledger_net_profit=ledger_net,



        terminal_open_positions=open_positions,



        terminal_open_pnl=open_pnl if open_positions > 0 else 0.0,



        ledger_marked_net_profit=marked_ledger_net,



        trade_count_consistent=trade_ok,



        pnl_consistent=pnl_ok,



        reasons=tuple(reasons),



    )











@dataclass(frozen=True)



class MT5FitnessBreakdown:



    schema_version: int



    rank_score: float



    structural_j: float



    learning_signal: float



    eligible: bool



    eligibility_reasons: tuple[str, ...]



    sample_count: int



    sample_gate: float



    source: str



    expectancy: float



    cvar_5pct: float



    immediate_exit_ratio: float



    execution_failure_rate: float



    ledger_integrity: float











@dataclass(frozen=True)



class MT5BacktestMetrics:



    net_profit: float



    profit_factor: float



    total_trades: int



    balance_drawdown_pct: float



    equity_drawdown_pct: float



    report_path: Path



    win_rate: float = 0.0



    avg_win: float = 0.0



    avg_loss: float = 0.0



    bridge_path: Path | None = None



    bridge_rows: int = 0



    bridge_terminal_mark: dict[str, object] = field(default_factory=dict)



    execution_ledger_path: Path | None = None



    execution_summary: MT5ExecutionSummary | None = None



    reconciliation: MT5ReportLedgerReconciliation | None = None



    # V13 audit is populated after final lifecycle credit reaches the policy.



    # It contains policy-origin, context × action scoped net-R expectancy; it



    # is deliberately separate from raw report metrics.



    expectancy_audit: dict[str, object] = field(default_factory=dict)



    feature_audit_path: Path | None = None



    feature_audit: dict[str, object] = field(default_factory=dict)



    # Persist evaluator fidelity in the result so a low-fidelity OHLC run



    # cannot accidentally be promoted by a caller that only sees a scalar.



    tester_model: int = 0



    require_real_ticks_for_promotion: bool = True



    require_real_tick_evidence_for_promotion: bool = True



    require_terminal_flat_for_promotion: bool = True



    real_tick_evidence: dict[str, object] = field(default_factory=dict)



    feature_schema_hash: str = ""



    fitness_schema_version: int = FITNESS_SCHEMA_VERSION



    fitness_eligible: bool = True



    fitness_rank_score: float = 0.0



    fitness_sample_count: int = 0



    fitness_source: str = ""



    fitness_eligibility_reasons: tuple[str, ...] = ()











# V26 internal sentinel. The MT5 EA schema cannot send -1; it still receives



# Action.HOLD (0) as a protocol no-op. The sentinel is never a neural action.



NO_NEURAL_COMMAND = -1











@dataclass(frozen=True)



class BridgeRequest:



    sequence: int



    inputs: np.ndarray



    # The bridge is the sole authoritative raw feature source.  Keep this



    # separate from ``inputs`` once the policy boundary normalizes the latter:



    # M15 regime routing operates on native ATR ratios, not policy-scaled ones.



    raw_inputs: np.ndarray | None = None



    schema_version: int = 1



    event_id: str = ""



    time: str = ""



    balance: float = 0.0



    equity: float = 0.0



    margin: float = 0.0



    free_margin: float = 0.0



    margin_level_pct: float = 0.0



    leverage: int = 0



    drawdown_pct: float = 0.0



    spread_points: float = 0.0



    positions: int = 0



    day_trades: int = 0



    positions_pipe: str = ""



    orders_pipe: str = ""



    execution_pipe: str = ""



    regime_index: int | None = None



    regime_probabilities: tuple[float, ...] | None = None



    regime_name: str | None = None











@dataclass(frozen=True)



class BridgeAgentAction:



    """One row in an atomic bridge action packet."""







    agent_id: int



    trace_id: str



    action: int



    score: float



    confidence: float



    outputs: np.ndarray



    trace: object | None = None



    risk_pct: float | None = None











@dataclass



class ExperienceStarvationState:



    no_trade_streak: int = 0



    unfilled_streak: int = 0



    flat_idle_streak: int = 0



    # Counts bridge decisions while an existing pending order remains visible.



    # This is separate from ``unfilled_streak`` because a policy may HOLD while



    # waiting; without it a 60-bar pending order can silently block bootstrap.



    pending_streak: int = 0



    last_day_trades: int = 0







    def confirm_execution(



        self,



        before: BridgeRequest | None,



        after: BridgeRequest | None,



        *,



        emitted_action: int | None = None,



    ) -> bool:



        """Resolve one previously emitted bridge intent exactly once.







        V24 updated starvation both when an intent was emitted and again when



        the next request arrived.  Because the first update compared the same



        request before/after values, every bar looked unfilled.  V25 mutates



        starvation only here, where MT5 has had one request interval to report a



        fill, rejection or a still-live pending order.



        """



        if before is None or after is None:



            return False



        opened_or_filled = (



            int(getattr(after, "positions", 0)) > int(getattr(before, "positions", 0))



            or int(getattr(after, "day_trades", 0)) > int(getattr(before, "day_trades", 0))



        )



        entry_intent = int(emitted_action if emitted_action is not None else -1) in {1, 2, 3, 4, 5, 6}



        if opened_or_filled:



            self.no_trade_streak = 0



            self.unfilled_streak = 0



            self.flat_idle_streak = 0



        elif entry_intent:



            # V27: only a real entry attempt that did not fill may shape the



            # execution-recovery signal.  Repeated FLAT_IDLE/HOLD ticks must



            # not grow neural starvation or feed a synthetic no-trade outcome



            # into M1.



            self.no_trade_streak += 1



            self.unfilled_streak += 1



            self.flat_idle_streak = 0



        if bool(str(getattr(after, "orders_pipe", "") or "").strip()):



            self.pending_streak += 1



        else:



            self.pending_streak = 0



        self.last_day_trades = max(self.last_day_trades, int(getattr(after, "day_trades", 0)))



        return bool(opened_or_filled)







    def observe_flat_idle(self) -> None:



        """Record bridge-local flat idle pressure without economic feedback."""



        self.flat_idle_streak += 1



        self.no_trade_streak += 1



        self.unfilled_streak = 0



        self.pending_streak = 0







    def preview(



        self,



        request: BridgeRequest,



        min_daily_trades: int,



        *,



        suppress_flat_idle_feedback: bool = False,



    ) -> dict[str, float]:



        """Return execution-recovery context, never an MT5 flat-state reward.







        Raw absence of positions/orders is telemetry



        only.  The neural runtime may see recovery pressure only after its own



        entry intent failed to fill; day-cadence deficit remains audit data and



        is zeroed from the neural feature channel.



        """



        account_offset = 56 + 5 * 8



        day_progress = float(np.clip(request.inputs[account_offset + 6], 0.0, 1.0)) if request.inputs.size > account_offset + 6 else 0.0



        expected = max(0.0, float(min_daily_trades) * day_progress)



        telemetry_cadence_deficit = max(0.0, expected - float(request.day_trades)) / max(1.0, float(min_daily_trades))



        cadence_deficit = 0.0 if suppress_flat_idle_feedback else telemetry_cadence_deficit



        starvation = float(np.clip(0.04 * self.no_trade_streak + 0.03 * self.unfilled_streak + cadence_deficit, 0.0, 3.0))



        return {



            "no_trade_streak": float(self.no_trade_streak),



            "unfilled_streak": float(self.unfilled_streak),



            "flat_idle_streak": float(self.flat_idle_streak),



            "pending_streak": float(self.pending_streak),



            "cadence_deficit": float(cadence_deficit),



            "telemetry_cadence_deficit": float(telemetry_cadence_deficit),



            "starvation": starvation,



            # Starvation controls recovery of failed entry intents only.  It is



            # never an economic reward and never represents FLAT_IDLE/HOLD.



            "intrinsic_reward": 0.0,



            "exploration_pressure": starvation,



            "opened_or_filled": 0.0,



            "flat_idle_feedback_suppressed": float(bool(suppress_flat_idle_feedback)),



        }







    def snapshot(self, request: BridgeRequest, action: int, trades_before: int, trades_after: int, min_daily_trades: int) -> dict[str, float]:



        """Return a non-mutating preview for the action being emitted.







        The actual no-fill/pending state is settled by ``confirm_execution`` on



        the next bridge request.  Keep this method for API compatibility with



        the existing bridge and telemetry callers.



        """



        _ = (action, trades_before, trades_after)



        out = self.preview(request, min_daily_trades)



        out["opened_or_filled"] = 0.0



        return out











def _num(value: str) -> float:



    cleaned = value.strip().replace("\xa0", " ").replace(" ", "").replace(",", "")



    if cleaned in {"", "-"}:



        return 0.0



    return float(cleaned)











def _strip_report_html(raw: str) -> str:



    text = re.sub(r"<[^>]+>", " ", raw)



    text = html.unescape(text)



    return re.sub(r"\s+", " ", text)











def assess_mt5_real_tick_evidence(



    report_path: str | Path,



    *,



    requested_model: int,



    started_at_ns: int | None = None,



) -> dict[str, object]:



    """Assess auditable *configuration evidence* for real-tick promotion.







    MT5's Python-facing report parser cannot independently prove which tick



    history the terminal actually downloaded. This function therefore makes a



    deliberately narrow, honest claim: it verifies that model 4 (real ticks)



    was requested,



    the report is fresh for this invocation, and the report identifies a



    real-tick test. Missing textual evidence is a failed promotion gate rather



    than an assumption based only on ``model==0``.



    """



    path = Path(report_path)



    requested = int(requested_model) == MT5_MODEL_REAL_TICKS



    if not path.exists() or not path.is_file():



        return {



            "available": False,



            "requested_model_real_ticks": requested,



            "report_fresh": False,



            "report_real_tick_marker": False,



            "verified": False,



            "reason": "tester_report_missing",



        }



    try:



        raw = path.read_bytes()



        decoded = ""



        for encoding in ("utf-16", "utf-8", "cp1252"):



            try:



                decoded = raw.decode(encoding, errors="ignore")



                if decoded:



                    break



            except UnicodeDecodeError:



                continue



        plain = _strip_report_html(decoded).lower()



        markers = (



            "every tick based on real ticks",



            "real ticks",



            "based on real ticks",



        )



        marker = any(value in plain for value in markers)



        fresh = True



        if started_at_ns is not None:



            # Some filesystems have coarse timestamp precision; allow a 2s



            # tolerance while still rejecting a stale report from a prior run.



            fresh = int(path.stat().st_mtime_ns) >= int(started_at_ns) - 2_000_000_000



        verified = bool(requested and fresh and marker)



        return {



            "available": True,



            "requested_model_real_ticks": requested,



            "report_fresh": bool(fresh),



            "report_real_tick_marker": bool(marker),



            "verified": verified,



            "reason": "" if verified else (



                "tester_model_not_real_ticks" if not requested else



                "tester_report_stale" if not fresh else



                "real_tick_marker_missing"



            ),



        }



    except Exception as exc:



        return {



            "available": True,



            "requested_model_real_ticks": requested,



            "report_fresh": False,



            "report_real_tick_marker": False,



            "verified": False,



            "reason": f"real_tick_evidence_error:{type(exc).__name__}",



        }











def parse_mt5_report(report_path: str | Path) -> MT5BacktestMetrics:



    path = Path(report_path)



    if not path.exists():



        raise FileNotFoundError(f"MT5 report not found: {path}")



    raw = path.read_text(encoding="utf-16", errors="ignore")



    text = _strip_report_html(raw)







    def required(pattern: str, label: str) -> re.Match[str]:



        match = re.search(pattern, text)



        if not match:



            raise ValueError(f"Cannot parse {label} from MT5 report: {path}")



        return match







    net_profit = _num(required(r"Total Net Profit:\s*([-0-9.,\s]+)", "Total Net Profit").group(1))



    profit_factor = _num(required(r"Profit Factor:\s*([-0-9.,\s]+)", "Profit Factor").group(1))



    total_trades = int(_num(required(r"Total Trades:\s*([0-9.,\s]+)", "Total Trades").group(1)))



    balance_dd_match = required(r"Balance Drawdown Maximal:\s*[-0-9.,\s]+\(([-0-9.,\s]+)%\)", "Balance Drawdown Maximal")



    equity_dd_match = required(r"Equity Drawdown Maximal:\s*[-0-9.,\s]+\(([-0-9.,\s]+)%\)", "Equity Drawdown Maximal")



    win_rate = 0.0



    win_match = re.search(r"Profit Trades \(% of total\):\s*[0-9.,\s]+\(\s*([-0-9.,]+)%\)", text)



    if win_match:



        win_rate = _num(win_match.group(1))



    avg_win = 0.0



    avg_win_match = re.search(r"Average profit trade:\s*([-0-9.,\s]+)", text)



    if avg_win_match:



        avg_win = _num(avg_win_match.group(1))



    avg_loss = 0.0



    avg_loss_match = re.search(r"Average loss trade:\s*([-0-9.,\s]+)", text)



    if avg_loss_match:



        avg_loss = _num(avg_loss_match.group(1))



    loss_trades = 0



    loss_trades_match = re.search(r"Loss Trades \(% of total\):\s*([0-9.,\s]+)", text)



    if loss_trades_match:



        loss_trades = int(_num(loss_trades_match.group(1)))



    if total_trades > 0 and loss_trades == 0 and avg_loss == 0.0 and win_rate > 0.0:



        profit_factor = 999.0



    return MT5BacktestMetrics(



        net_profit=net_profit,



        profit_factor=profit_factor,



        total_trades=total_trades,



        balance_drawdown_pct=_num(balance_dd_match.group(1)),



        equity_drawdown_pct=_num(equity_dd_match.group(1)),



        report_path=path,



        win_rate=win_rate,



        avg_win=avg_win,



        avg_loss=avg_loss,



    )











def _read_mt5_report_raw(report_path: str | Path) -> str:



    raw = Path(report_path).read_bytes()



    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252"):



        try:



            text = raw.decode(encoding)



            if text:



                return text



        except UnicodeDecodeError:



            continue



    return raw.decode("utf-8", errors="ignore")











def _html_table_rows(raw: str) -> list[list[str]]:



    rows: list[list[str]] = []



    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", raw, flags=re.IGNORECASE | re.DOTALL):



        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)



        if not cells:



            continue



        cleaned: list[str] = []



        for cell in cells:



            text = re.sub(r"<[^>]+>", " ", cell)



            cleaned.append(re.sub(r"\s+", " ", html.unescape(text)).strip())



        rows.append(cleaned)



    return rows











def _report_intent_sequence(comment: str) -> int | None:



    """Read sequence from current SG5 and legacy SG4 MT5 comments."""



    match = re.search(r"\bSG(?:4|5):(\d+)(?::\d+)?\b", str(comment or ""))



    if not match:



        return None



    return int(match.group(1))











def parse_mt5_report_lifecycles(



    report_path: str | Path,



    *,



    trace_by_sequence: Mapping[int, str] | None = None,



    initial_equity: float = 10000.0,



    bar_seconds: float = 60.0,



) -> tuple[MT5PositionLifecycle, ...]:



    raw = _read_mt5_report_raw(report_path)



    rows = _html_table_rows(raw)



    trace_by_sequence = dict(trace_by_sequence or {})



    order_by_id: dict[str, dict[str, object]] = {}



    in_deals = False



    for row in rows:



        lower = [cell.lower() for cell in row]



        if "deal" in lower and "direction" in lower:



            in_deals = True



            continue



        if "order" in lower and "comment" in lower and not in_deals:



            continue



        if not in_deals and len(row) >= 8:



            seq = _report_intent_sequence(row[-1])



            if seq is not None:



                order_by_id[str(row[1])] = {



                    "sequence": seq,



                    "trace_id": trace_by_sequence.get(seq, ""),



                    "entry_time": _ledger_time(row[0]),



                    "entry_price": _ledger_float(row[4]),



                    "initial_sl_price": _ledger_float(row[5]),



                    "initial_tp_price": _ledger_float(row[6]),



                }



            continue



        if not in_deals or len(row) < 11:



            continue







    lifecycles: list[MT5PositionLifecycle] = []



    open_by_order: dict[str, dict[str, object]] = {}



    last_open: dict[str, object] | None = None



    in_deals = False



    deal_columns: dict[str, int] = {}



    for row in rows:



        lower = [cell.lower() for cell in row]



        if "deal" in lower and "direction" in lower:



            in_deals = True



            # MT5 report builds differ: older reports omit Symbol/Balance,



            # while current reports include both.  Resolve the deal schema



            # from its header instead of assuming fixed offsets.



            deal_columns = {



                str(cell).strip().lower(): index



                for index, cell in enumerate(row)



                if str(cell).strip()



            }



            continue



        if not in_deals or not deal_columns:



            continue



        def deal_cell(name: str, default: str = "") -> str:



            index = deal_columns.get(name)



            if index is None or index >= len(row):



                return default



            return str(row[index] or default)







        direction = deal_cell("direction").strip().lower()



        if direction not in {"in", "out", "in/out"}:



            continue



        order_id = deal_cell("order")



        comment = deal_cell("comment")



        seq = _report_intent_sequence(comment)



        if direction == "in":



            meta = dict(order_by_id.get(order_id, {}))



            if seq is not None:



                meta["sequence"] = seq



                meta["trace_id"] = trace_by_sequence.get(seq, str(meta.get("trace_id", "")))



            open_by_order[order_id] = {



                **meta,



                "entry_time": _ledger_time(deal_cell("time")) or meta.get("entry_time"),



                "entry_price": _ledger_float(



                    deal_cell("price"), default=float(meta.get("entry_price", 0.0) or 0.0)



                ),



            }



            last_open = open_by_order[order_id]



            continue



        if direction != "out":



            continue



        # A close deal commonly carries a new close-order id whose comment



        # sequence belongs to the exit action.  Prefer the active entry



        # lifecycle/most recent open before consulting order metadata so the



        # realized PnL remains owned by the entry trace.  Keep order metadata



        # as a recovery fallback when the report starts after the entry.



        entry = open_by_order.get(order_id) or last_open or order_by_id.get(order_id) or {}



        entry_sequence = int(entry.get("sequence", seq or 0) or 0)



        if entry_sequence <= 0:



            continue



        commission = _ledger_float(deal_cell("commission"))



        swap = _ledger_float(deal_cell("swap"))



        profit = _ledger_float(deal_cell("profit"))



        net_pnl = profit + commission + swap



        entry_time = entry.get("entry_time")



        exit_time = _ledger_time(deal_cell("time"))



        duration_bars = None



        if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):



            duration_bars = max(0.0, (exit_time - entry_time).total_seconds() / max(1.0, float(bar_seconds)))



        close_reason = "TP" if "tp" in comment.lower() else "SL" if "sl" in comment.lower() else "OTHER"



        risk_unit = max(1e-9, float(initial_equity) * 0.01)



        lifecycles.append(MT5PositionLifecycle(



            position_ticket=f"report:{order_id}:{row[1]}",



            entry_time=entry_time if isinstance(entry_time, datetime) else None,



            exit_time=exit_time,



            net_pnl=float(net_pnl),



            gross_profit=float(profit),



            commission=float(commission),



            swap=float(swap),



            initial_risk=0.0,



            realized_r=float(net_pnl / risk_unit),



            duration_bars=duration_bars,



            close_reason=close_reason,



            source="strategy_tester_report_tag_fallback",



            trace_id=str(entry.get("trace_id", trace_by_sequence.get(entry_sequence, "")) or ""),



            entry_sequence=entry_sequence,



            entry_count=1,



            close_count=1,



            final_close_confirmed=True,



            exit_mechanism=classify_exit_mechanism(close_reason, realized_r=float(net_pnl / risk_unit), partial_close_count=0).value,



            entry_price=float(entry.get("entry_price", 0.0) or 0.0),



            exit_price=_ledger_float(deal_cell("price")),



            initial_sl_price=float(entry.get("initial_sl_price", 0.0) or 0.0),



            initial_tp_price=float(entry.get("initial_tp_price", 0.0) or 0.0),



            final_sl_price=float(entry.get("initial_sl_price", 0.0) or 0.0),



            final_tp_price=float(entry.get("initial_tp_price", 0.0) or 0.0),



            metadata={"join_source": "strategy_tester_report_intent_comment"},



        ))



    return tuple(lifecycles)











def lifecycle_credit_candidates(



    native_lifecycles: tuple[MT5PositionLifecycle, ...],



    report_lifecycles: tuple[MT5PositionLifecycle, ...],



) -> tuple[MT5PositionLifecycle, ...]:



    selected: dict[str, MT5PositionLifecycle] = {}



    for lifecycle in report_lifecycles:



        key = str(lifecycle.trace_id or lifecycle.entry_sequence or lifecycle.position_ticket)



        selected[key] = lifecycle



    for lifecycle in native_lifecycles:



        key = str(lifecycle.trace_id or lifecycle.entry_sequence or lifecycle.position_ticket)



        selected[key] = lifecycle



    return tuple(selected.values())















def _ledger_float(value: object, default: float = 0.0) -> float:



    try:



        text = str(value).strip().replace("\xa0", " ").replace(" ", "").replace(",", "")



        if text in {"", "-", "None", "none"}:



            return float(default)



        return float(text)



    except (TypeError, ValueError):



        return float(default)











def _ledger_bool(value: object) -> bool:



    if isinstance(value, bool):



        return value



    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok", "success"}











def _ledger_time(value: object) -> datetime | None:



    text = str(value or "").strip()



    if not text:



        return None



    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):



        try:



            return datetime.strptime(text, fmt)



        except ValueError:



            continue



    return None











def _mt5_replay_anchor_reached(anchor: object, request_time: object) -> bool:



    """Return whether a causal MT5 request has reached or crossed the anchor.







    Real-tick streams are not guaranteed to contain a tick in every wall-clock



    minute.  Requiring an exact minute match incorrectly rejects a valid replay



    when the stream jumps from the minute before the anchor to the minute after



    it.  The EA already bounds requests to the replay window, so the first



    parseable request at or after the anchor is the causal anchor observation.



    """



    anchor_dt = _ledger_time(anchor)



    request_dt = _ledger_time(request_time)



    return bool(anchor_dt is not None and request_dt is not None and request_dt >= anchor_dt)











def _ledger_first_float(row: dict[str, str], *keys: str, default: float = 0.0) -> float:



    for key in keys:



        value = row.get(key)



        if value is not None and str(value).strip() not in {"", "-", "None", "none"}:



            return _ledger_float(value, default)



    return float(default)











def _ledger_first_text(row: dict[str, str], *keys: str, default: str = "") -> str:



    for key in keys:



        value = row.get(key)



        if value is not None and str(value).strip():



            return str(value).strip()



    return str(default)











def _ledger_account_mark(row: Mapping[str, object]) -> tuple[float, float] | None:



    """Read an optional final-deal balance/equity mark from the ledger."""



    geometry_values = str(row.get("geometry_entry_pipe", "") or "").split("|")



    if len(geometry_values) >= 17:



        balance = _ledger_float(geometry_values[15], default=float("nan"))



        equity = _ledger_float(geometry_values[16], default=float("nan"))



        if np.isfinite(balance) and np.isfinite(equity) and balance > 0.0:



            return float(balance), float(equity)



    pipe = str(row.get("account_mark_pipe", "") or "").strip()



    if pipe:



        parts = [part.strip() for part in pipe.split("|", 1)]



        if len(parts) == 2:



            balance = _ledger_float(parts[0], default=float("nan"))



            equity = _ledger_float(parts[1], default=float("nan"))



            if np.isfinite(balance) and np.isfinite(equity):



                return float(balance), float(equity)



    # Accept the early local fixture spelling as a harmless compatibility



    # path; production EA rows use the single pipe field above.



    if _ledger_has_value(row, "account_balance_after") and _ledger_has_value(



        row, "account_equity_after"



    ):



        balance = _ledger_float(row.get("account_balance_after"), default=float("nan"))



        equity = _ledger_float(row.get("account_equity_after"), default=float("nan"))



        if np.isfinite(balance) and np.isfinite(equity):



            return float(balance), float(equity)



    return None











def _ledger_optional_bool(row: dict[str, str], *keys: str) -> bool | None:



    for key in keys:



        if key in row and str(row.get(key, "")).strip() != "":



            return _ledger_bool(row.get(key))



    return None











def _ledger_has_value(row: Mapping[str, object], *keys: str) -> bool:



    return any(



        key in row and str(row.get(key, "")).strip() not in {"", "-", "None", "none"}



        for key in keys



    )











def _causal_trajectory_point(row: dict[str, str], event_time: datetime | None) -> dict[str, object] | None:



    # A V19 EA may emit a row on every bar/management event. Earlier schemas



    # only produce fill/close rows, which remain useful but provide low



    # trajectory coverage. No values are invented.



    event_kind = _ledger_first_text(row, "transaction_type", "event_type").upper()



    has_path = all(



        _ledger_first_float(row, key) > 0.0



        for key in ("path_high_price", "path_low_price", "path_close_price")



    )



    explicit_excursion = _ledger_has_value(row, "unrealized_r", "mfe_r", "mae_r")



    other_markers = (



        "market_price", "last_price", "management_event", "volatility_shock",



        "spread_atr_ratio",



    )



    if not (



        event_kind == "POSITION_TRAJECTORY"



        or has_path



        or explicit_excursion



        or any(str(row.get(key, "")).strip() for key in other_markers)



    ):



        return None



    return {



        "timestamp": event_time.timestamp() if event_time is not None else 0.0,



        "age_bars": _ledger_first_float(row, "age_bars", "bar_index", default=-1.0),



        "price": _ledger_first_float(row, "market_price", "last_price", "price", "bid", "close"),



        "high": _ledger_first_float(row, "path_high_price", "high", "bar_high"),



        "low": _ledger_first_float(row, "path_low_price", "low", "bar_low"),



        "close": _ledger_first_float(row, "path_close_price", "close", "price"),



        "spread": _ledger_first_float(row, "path_spread_price", "spread_price"),



        "unrealized_r": _ledger_first_float(row, "unrealized_r", "floating_r"),



        "mfe_r": _ledger_first_float(row, "mfe_r"),



        "mae_r": _ledger_first_float(row, "mae_r"),



        "distance_to_sl_atr": _ledger_first_float(row, "distance_to_sl_atr", "sl_distance_atr"),



        "distance_to_tp_atr": _ledger_first_float(row, "distance_to_tp_atr", "tp_distance_atr"),



        "spread_atr_ratio": _ledger_first_float(row, "spread_atr_ratio"),



        "atr": _ledger_first_float(row, "path_atr", "atr"),



        "volatility_shock": _ledger_first_float(row, "volatility_shock", "volatility_shock_score"),



        "regime": _ledger_first_text(row, "regime", "market_regime"),



        "management_event": _ledger_first_text(row, "management_event", "event"),



    }











def _expand_geometry_pipes(row: dict[str, str]) -> dict[str, str]:



    """Expand compact legacy and fixed-V13 ledger cells before parsing.







    MQL ``FileWrite`` has a finite variadic-argument limit. Two pipe fields



    retain the geometry, requested-order, and path scalars without splitting a



    CSV record across lines; legacy/direct-column fixtures remain supported.



    """



    expanded = dict(row)



    for field, keys in {



        "geometry_entry_pipe": ("initial_atr", "side", "proposed_sl_atr", "proposed_rr", "proposed_tp_atr"),



        # V16 preserves the five-value V15 prefix and appends three causal



        # position-R facts without adding another MQL FileWrite argument.



        "geometry_path_pipe": (



            "path_high_price", "path_low_price", "path_close_price", "path_atr",



            "path_spread_price", "unrealized_r", "mfe_r", "mae_r",



        ),



    }.items():



        values = str(expanded.get(field, "") or "").split("|")



        # V12 keeps the V11 prefix and appends the four requested-order



        # values to ``geometry_entry_pipe``.  Parse the stable prefix when



        # the pipe has either shape; geometry_path_pipe still has five.



        minimum = 5 if field == "geometry_path_pipe" else len(keys)



        if len(values) < minimum:



            continue



        for key, value in zip(keys, values):



            if key not in expanded or not str(expanded.get(key, "")).strip():



                expanded[key] = value



    # V12 appends four requested-order fields to the existing compact entry



    # pipe instead of creating another FileWrite argument (MQL has a finite



    # variadic argument limit). V11's five-field pipe remains readable.



    requested_values = str(expanded.get("geometry_entry_pipe", "") or "").split("|")



    requested_keys = ("requested_entry_price", "requested_sl_price", "requested_tp_price", "requested_atr")



    if len(requested_values) >= 9:



        for key, value in zip(requested_keys, requested_values[5:]):



            if key not in expanded or not str(expanded.get(key, "")).strip():



                expanded[key] = value



    # V13 appended fixed geometry; V15 keeps those four positions but gives



    # them geometry-neutral names and appends the learned trailing choice.



    if len(requested_values) >= 13:



        for key, value in zip(



            ("geometry_mode", "geometry_sl_atr", "geometry_tp_atr", "geometry_rr"),



            requested_values[9:13],



        ):



            if key not in expanded or not str(expanded.get(key, "")).strip():



                expanded[key] = value



        if str(expanded.get("geometry_mode", "")).strip() == "fixed_atr_v1":



            for old_key, new_key in (



                ("fixed_sl_atr", "geometry_sl_atr"),



                ("fixed_tp_atr", "geometry_tp_atr"),



                ("fixed_rr", "geometry_rr"),



            ):



                expanded.setdefault(old_key, expanded.get(new_key, ""))



    if len(requested_values) >= 14 and not str(expanded.get("proposed_trailing_atr", "")).strip():



        expanded["proposed_trailing_atr"] = requested_values[13]



    # Schema 17 appends the exact pending-order age at fill without adding a



    # FileWrite argument (MetaTrader has a strict variadic-argument limit).



    if len(requested_values) >= 15 and not str(expanded.get("pending_fill_bars", "")).strip():



        expanded["pending_fill_bars"] = requested_values[14]



    return expanded











def _apply_causal_excursion_evidence(



    row: Mapping[str, object],



    state: dict[str, object],



    point: dict[str, object] | None,



) -> None:



    """Use explicit V16+ excursion facts or rebuild them from the V15 path."""



    if point is None:



        return



    row_dict = dict(row)



    if _ledger_has_value(row, "mfe_r") and _ledger_has_value(row, "mae_r"):



        mfe = _ledger_first_float(row_dict, "mfe_r")



        mae = _ledger_first_float(row_dict, "mae_r")



        point["unrealized_r"] = _ledger_first_float(row_dict, "unrealized_r", "floating_r")



        point["mfe_r"] = mfe



        point["mae_r"] = mae



        state["mfe_r"] = max(float(state.get("mfe_r", 0.0)), mfe)



        state["mae_r"] = min(float(state.get("mae_r", 0.0)), mae)



        state["excursion_available"] = True



        state["excursion_source"] = "ledger_v16"



        return







    entry = float(state.get("entry_price", 0.0) or 0.0)



    risk = float(state.get("initial_risk_distance_price", 0.0) or 0.0)



    if risk <= 1e-12:



        risk = abs(entry - float(state.get("initial_sl_price", 0.0) or 0.0))



    side = int(state.get("side", 0) or 0)



    high = float(point.get("high", 0.0) or 0.0)



    low = float(point.get("low", 0.0) or 0.0)



    close = float(point.get("close", 0.0) or 0.0)



    spread = max(0.0, float(point.get("spread", 0.0) or 0.0))



    if entry <= 0.0 or risk <= 1e-12 or side not in {-1, 1}:



        return



    if high <= 0.0 or low <= 0.0 or close <= 0.0 or high + 1e-12 < low:



        return



    entry_time = state.get("entry_time")



    point_timestamp = float(point.get("timestamp", 0.0) or 0.0)



    if isinstance(entry_time, datetime) and point_timestamp > 0.0:



        closed_bar_start = float(int(point_timestamp // 60.0) * 60 - 60)



        if entry_time.timestamp() > closed_bar_start + 1e-6:



            # The first closed M1 bar may contain prices from before a mid-bar



            # fill. Do not attribute its high/low to the position lifecycle.



            return



    if side > 0:



        favorable = (high - entry) / risk



        adverse = (low - entry) / risk



        unrealized = (close - entry) / risk



    else:



        favorable = (entry - (low + spread)) / risk



        adverse = (entry - (high + spread)) / risk



        unrealized = (entry - (close + spread)) / risk



    cumulative_mfe = max(float(state.get("mfe_r", 0.0)), favorable)



    cumulative_mae = min(float(state.get("mae_r", 0.0)), adverse)



    point["unrealized_r"] = float(unrealized)



    point["mfe_r"] = float(cumulative_mfe)



    point["mae_r"] = float(cumulative_mae)



    state["mfe_r"] = float(cumulative_mfe)



    state["mae_r"] = float(cumulative_mae)



    state["excursion_available"] = True



    state["excursion_source"] = "geometry_path_v15_reconstructed"











def _read_execution_rows(path: Path) -> list[dict[str, str]]:



    try:



        raw = path.read_text(encoding="utf-8", errors="ignore")



    except OSError:



        return []



    if not raw.strip():



        return []



    first = raw.splitlines()[0] if raw.splitlines() else ""



    delimiter = "\t" if "\t" in first else ","



    try:



        return [dict(row) for row in csv.DictReader(raw.splitlines(), delimiter=delimiter) if isinstance(row, dict)]



    except (csv.Error, UnicodeError):



        return []











@dataclass



class _IncrementalCsvTail:



    """Read appended MT5 CSV records once without re-scanning a long run."""







    path: Path | None = None



    offset: int = 0



    header: tuple[str, ...] = ()



    pending_text: str = ""







    def read_rows(self, path: Path) -> list[dict[str, str]]:



        try:



            resolved = path.resolve()



            size = resolved.stat().st_size



        except OSError:



            return []



        if self.path != resolved or size < self.offset:



            self.path = resolved



            self.offset = 0



            self.header = ()



            self.pending_text = ""



        try:



            with resolved.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:



                if not self.header:



                    raw_header = handle.readline()



                    if not raw_header:



                        return []



                    self.header = tuple(next(csv.reader([raw_header], delimiter="\t"), ()))



                    self.offset = handle.tell()



                    if not self.header:



                        return []



                else:



                    handle.seek(self.offset)



                raw = self.pending_text + handle.read()



                self.offset = handle.tell()



        except OSError:



            return []



        self.pending_text = ""



        if raw and not raw.endswith(("\n", "\r")):



            complete, separator, trailing = raw.rpartition("\n")



            if not separator:



                self.pending_text = raw



                return []



            raw = complete + separator



            self.pending_text = trailing



        if not raw.strip():



            return []



        try:



            return [



                dict(row) for row in csv.DictReader(



                    io.StringIO(raw), fieldnames=self.header, delimiter="\t"



                ) if isinstance(row, dict)



            ]



        except (csv.Error, UnicodeError):



            return []











def parse_mt5_execution_ledger(



    ledger_path: str | Path | None,



    *,



    initial_equity: float,



    min_holding_bars: int = 3,



    bar_seconds: float = 60.0,



    technical_min_rr: float = 1.2,



) -> MT5ExecutionSummary:



    """Aggregate versioned execution ledgers into final position lifecycles.







    The parser accepts legacy ledgers for audit compatibility, but only rows



    with ``initial_risk`` and ``position_final_close`` have full position-R



    provenance. For same-tick SL/TP rows from older ledgers, risk is recovered



    from the terminal outcome and explicitly tagged as inferred; all remaining



    missing-risk rows are downgraded to an equity-risk proxy.



    """



    path = Path(ledger_path) if ledger_path else None



    if path is None or not path.exists():



        return MT5ExecutionSummary(ledger_path=path)



    rows = _read_execution_rows(path)



    if not rows:



        return MT5ExecutionSummary(ledger_path=path, ledger_available=True)







    versions = [int(_ledger_float(row.get("schema_version"), 0.0)) for row in rows]



    schema_version = max(versions) if versions else 0



    by_position: dict[str, dict[str, object]] = {}



    intent_state: dict[str, dict[str, bool]] = {}



    policy_events = 0



    traced_policy_events = 0



    for raw_row in rows:



        row = _expand_geometry_pipes(raw_row)



        source = str(row.get("source", "") or "").strip().lower()



        ack_sequence = str(row.get("ack_sequence", "") or "").strip()



        is_policy = source == "policy" or (source == "" and ack_sequence not in {"", "0"})



        if is_policy:



            policy_events += 1



            if str(row.get("trace_id", "") or "").strip():



                traced_policy_events += 1



            key = ack_sequence or f"row-{policy_events}"



            state = intent_state.setdefault(key, {"accepted": False, "rejected": False})



            state["accepted"] = bool(state["accepted"] or _ledger_bool(row.get("accepted")) or _ledger_bool(row.get("placed")) or _ledger_bool(row.get("filled")) or _ledger_bool(row.get("closed")) or _ledger_bool(row.get("modified")) or _ledger_bool(row.get("cancelled")))



            state["rejected"] = bool(state["rejected"] or _ledger_bool(row.get("rejected")))







        ticket = str(row.get("position_ticket", "") or "").strip()



        if not ticket or ticket in {"0", "0.0"}:



            continue



        filled = _ledger_bool(row.get("filled"))



        closed = _ledger_bool(row.get("closed"))



        modified = _ledger_bool(row.get("modified"))



        post_exit_observation = str(



            row.get("transaction_type", row.get("event_type", "")) or ""



        ).strip().upper() == "POST_EXIT_OBSERVATION"



        # V19 trajectory rows may describe a live position without being an



        # entry-fill or final-close row. Retain them when they contain an



        # actual trajectory sample; legacy noise rows still remain ignored.



        event_time = _ledger_time(row.get("logged_at"))



        trajectory_point = _causal_trajectory_point(row, event_time)



        if not (filled or closed or modified or post_exit_observation or trajectory_point is not None):



            continue



        state = by_position.setdefault(ticket, {



            "entry_time": None, "exit_time": None, "net_pnl": 0.0,



            "gross_profit": 0.0, "commission": 0.0, "swap": 0.0,



            "initial_risk": 0.0, "duration_bars": None, "close_reason": "",



            "source": source, "trace_id": "", "entry_sequence": 0,



            "entry_count": 0, "close_count": 0,



            "final_close": False, "saw_final_field": False,



            "entry_price": 0.0, "exit_price": 0.0,



            "initial_risk_distance_price": 0.0,



            "initial_sl_price": 0.0, "final_sl_price": 0.0,



            "initial_tp_price": 0.0, "final_tp_price": 0.0,



            "partial_close_count": 0, "mfe_r": 0.0, "mae_r": 0.0,



            "excursion_available": False, "excursion_source": "",



            "giveback_r": 0.0, "near_tp_ratio": 0.0, "slippage_r": 0.0,



            "post_exit_mfe_r": None, "post_exit_mae_r": None,



            "returned_to_entry": None, "original_tp_hit_after_exit": None,



            "exit_kind": "", "exit_controller": "", "exit_trace_id": "",



            "exit_requested_action": -1, "post_exit_horizon_bars": None,



            "post_exit_hold_r": None,



            "geometry_mode": "", "fixed_sl_atr": 0.0, "fixed_tp_atr": 0.0, "fixed_rr": 0.0,



            "geometry_sl_atr": 0.0, "geometry_tp_atr": 0.0, "geometry_rr": 0.0,



            "initial_atr": 0.0, "side": 0, "proposed_sl_atr": 0.0,



            "proposed_rr": 0.0, "proposed_trailing_atr": 0.0,



            "requested_entry_price": 0.0, "requested_sl_price": 0.0,



            "requested_tp_price": 0.0, "requested_atr": 0.0,



            "geometry_mutation_count": 0, "trailing_applied": False,



            "pending_fill_bars": None,



            "trajectory": [], "metadata": {},



            # Optional append-only v17 final-deal account mark.  Older



            # ledgers omit these columns and remain explicitly partial.



            "account_balance_after_close": None,



            "account_equity_after_close": None,



            "account_mark_sequence": 0,



            "account_mark_batch_key": "",



            "account_mark_attribution": "not_available",



            "account_cashflow_delta": None,



            "account_cashflow_deal_ticket": "",



            "account_cashflow_attribution": "not_available",



        })



        # Causal fields are optional V19 EA extensions. Legacy rows retain



        # zero/None and are reported as incomplete coverage downstream.



        for key, aliases in {



            "fixed_sl_atr": ("fixed_sl_atr",),



            "fixed_tp_atr": ("fixed_tp_atr",),



            "fixed_rr": ("fixed_rr",),



            "geometry_sl_atr": ("geometry_sl_atr", "fixed_sl_atr"),



            "geometry_tp_atr": ("geometry_tp_atr", "fixed_tp_atr"),



            "geometry_rr": ("geometry_rr", "fixed_rr"),



            "initial_atr": ("initial_atr",),



            "proposed_sl_atr": ("proposed_sl_atr",),



            "proposed_rr": ("proposed_rr",),



            "proposed_trailing_atr": ("proposed_trailing_atr",),



            "requested_entry_price": ("requested_entry_price",),



            "requested_sl_price": ("requested_sl_price",),



            "requested_tp_price": ("requested_tp_price",),



            "requested_atr": ("requested_atr",),



        }.items():



            value = _ledger_first_float(row, *aliases, default=0.0)



            if value > 1e-12:



                state[key] = value



        side_value = int(_ledger_first_float(row, "side", "direction", default=0.0))



        if side_value in {-1, 1}:



            state["side"] = side_value



        geometry_mode = str(row.get("geometry_mode", "") or "").strip()



        if geometry_mode:



            state["geometry_mode"] = geometry_mode



        if modified and (_ledger_bool(row.get("accepted")) or _ledger_bool(row.get("placed"))):



            state["geometry_mutation_count"] = int(state.get("geometry_mutation_count", 0)) + 1



            stop_controller = str(row.get("stop_controller", "") or "").strip().lower()



            stop_reason = str(row.get("stop_reason", "") or "").strip().lower()



            if stop_controller == "five_trailing_head" or stop_reason == "learned_trailing_atr":



                state["trailing_applied"] = True



        for key, aliases in {



            "entry_price": ("entry_price", "open_price", "fill_price"),



            "exit_price": ("exit_price", "close_price", "deal_price", "price"),



            "initial_sl_price": ("initial_sl_price", "initial_sl", "sl_initial"),



            "final_sl_price": ("final_sl_price", "final_sl", "sl"),



            "initial_tp_price": ("initial_tp_price", "initial_tp", "tp_initial"),



            "initial_risk_distance_price": ("initial_risk_distance_price",),



            "final_tp_price": ("final_tp_price", "final_tp", "tp"),



            "mfe_r": ("mfe_r",),



            "mae_r": ("mae_r",),



            "giveback_r": ("giveback_r",),



            "near_tp_ratio": ("near_tp_ratio",),



            "slippage_r": ("slippage_r",),



        }.items():



            value = _ledger_first_float(row, *aliases, default=0.0)



            if abs(value) > 1e-12:



                if key == "mae_r":



                    state[key] = min(float(state.get(key, 0.0)), value)



                elif key in {"mfe_r", "near_tp_ratio"}:



                    state[key] = max(float(state.get(key, 0.0)), value)



                else:



                    state[key] = value



        _apply_causal_excursion_evidence(row, state, trajectory_point)



        if trajectory_point is not None:



            state["trajectory"].append(trajectory_point)



        for key, aliases in {



            "post_exit_mfe_r": ("post_exit_mfe_r", "post_exit_max_favorable_r"),



            "post_exit_mae_r": ("post_exit_mae_r", "post_exit_max_adverse_r"),



        }.items():



            if any(str(row.get(alias, "")).strip() for alias in aliases):



                state[key] = _ledger_first_float(row, *aliases, default=0.0)



        if str(row.get("post_exit_hold_r", "")).strip():



            state["post_exit_hold_r"] = _ledger_first_float(row, "post_exit_hold_r", default=0.0)



        if str(row.get("post_exit_horizon_bars", "")).strip():



            state["post_exit_horizon_bars"] = int(_ledger_float(row.get("post_exit_horizon_bars"), default=0.0))



        # EA schema 12 wrote ``0`` into the dedicated post-exit horizon field



        # even though the HORIZON receipt itself carries the exact elapsed M1



        # bar count in ``age_bars``.  Treat that zero as missing, never as a



        # zero-bar counterfactual: the row is emitted only after the configured



        # horizon, and the recorded age is therefore causal evidence of the



        # hold window.  This restores manager credit without inventing a future



        # outcome or accepting ordinary lifecycle age as post-exit evidence.



        if post_exit_observation and int(state.get("post_exit_horizon_bars") or 0) <= 0:



            fallback_horizon = int(max(0.0, _ledger_float(row.get("age_bars"), default=0.0)))



            if fallback_horizon > 0:



                state["post_exit_horizon_bars"] = fallback_horizon



        for key in ("exit_kind", "exit_controller", "exit_trace_id"):



            value = str(row.get(key, "") or "").strip()



            if value:



                state[key] = value



        if str(row.get("exit_requested_action", "")).strip():



            state["exit_requested_action"] = int(_ledger_float(row.get("exit_requested_action"), default=-1.0))



        for key, aliases in {



            "returned_to_entry": ("returned_to_entry", "price_returned_to_entry"),



            "original_tp_hit_after_exit": ("original_tp_hit_after_exit", "original_tp_hit"),



        }.items():



            value = _ledger_optional_bool(row, *aliases)



            if value is not None:



                state[key] = value



        state["metadata"].update({



            key: value for key, value in row.items()



            if key in {



                "initial_tp_r", "target_r", "entry_type", "symbol", "side", "direction", "magic_number",



                "intent_context", "intent_expert", "intent_policy", "intent_state", "intent_reason",



                "position_owner_trace_id",



                "position_owner_context", "position_owner_expert", "position_owner_policy",



                "position_owner_state", "position_owner_reason",



                "stop_controller", "stop_reason", "old_sl_price", "old_tp_price",



                "new_sl_price", "new_tp_price", "initial_risk_distance_price",



                "geometry_mode", "fixed_sl_atr", "fixed_tp_atr", "fixed_rr",



                "geometry_sl_atr", "geometry_tp_atr", "geometry_rr",



                "initial_atr", "side", "proposed_sl_atr", "proposed_rr", "proposed_tp_atr",



                "proposed_trailing_atr",



                "requested_entry_price", "requested_sl_price", "requested_tp_price", "requested_atr",



            } and str(value).strip()



        })



        state["metadata"].update({



            "initial_atr": float(state.get("initial_atr", 0.0)),



            "side": int(state.get("side", 0)),



            "geometry_mode": str(state.get("geometry_mode", "") or ""),



            "fixed_sl_atr": float(state.get("fixed_sl_atr", 0.0)),



            "fixed_tp_atr": float(state.get("fixed_tp_atr", 0.0)),



            "fixed_rr": float(state.get("fixed_rr", 0.0)),



            "geometry_sl_atr": float(state.get("geometry_sl_atr", 0.0)),



            "geometry_tp_atr": float(state.get("geometry_tp_atr", 0.0)),



            "geometry_rr": float(state.get("geometry_rr", 0.0)),



            "proposed_sl_atr": float(state.get("proposed_sl_atr", 0.0)),



            "proposed_rr": float(state.get("proposed_rr", 0.0)),



            "proposed_trailing_atr": float(state.get("proposed_trailing_atr", 0.0)),



            "requested_entry_price": float(state.get("requested_entry_price", 0.0)),



            "requested_sl_price": float(state.get("requested_sl_price", 0.0)),



            "requested_tp_price": float(state.get("requested_tp_price", 0.0)),



            "requested_atr": float(state.get("requested_atr", 0.0)),



            "geometry_mutation_count": int(state.get("geometry_mutation_count", 0)),



        })



        row_trace_id = str(row.get("trace_id", "") or "").strip()



        if not row_trace_id:



            row_trace_id = str(row.get("position_owner_trace_id", "") or row.get("owner_trace_id", "") or "").strip()



        row_sequence = int(_ledger_float(row.get("ack_sequence"), 0.0))



        if row_trace_id and (filled or not str(state.get("trace_id", ""))):



            state["trace_id"] = row_trace_id



            if row_sequence > 0:



                state["entry_sequence"] = row_sequence



        if filled:



            state["entry_count"] = int(state["entry_count"]) + 1



            if str(row.get("pending_fill_bars", "")).strip():



                state["pending_fill_bars"] = max(



                    0, int(_ledger_float(row.get("pending_fill_bars"), default=0.0))



                )



            current = state.get("entry_time")



            if current is None or (event_time is not None and event_time < current):



                state["entry_time"] = event_time



        if closed:



            state["close_count"] = int(state["close_count"]) + 1



            profit = _ledger_float(row.get("profit"))



            commission = _ledger_float(row.get("commission"))



            swap = _ledger_float(row.get("swap"))



            state["gross_profit"] = float(state["gross_profit"]) + profit



            state["commission"] = float(state["commission"]) + commission



            state["swap"] = float(state["swap"]) + swap



            state["net_pnl"] = float(state["net_pnl"]) + profit + commission + swap



            if not _ledger_bool(row.get("position_final_close")):



                state["partial_close_count"] = int(state.get("partial_close_count", 0)) + 1



            current = state.get("exit_time")



            if current is None or (event_time is not None and event_time > current):



                state["exit_time"] = event_time



            reason = str(row.get("close_reason", "") or "").strip()



            if reason and reason.upper() != "OTHER":



                state["close_reason"] = reason



        if closed and _ledger_bool(row.get("position_final_close")):



            # The EA captures these values on the final DEAL_ADD transaction,



            # before another close can share a later bridge snapshot.  Keep



            # the mark only when both finite account values are present.  A



            # terminal can still deliver several close transactions in one



            # timestamp batch after the account has already advanced to the



            # final balance.  The post-pass below marks such reused snapshots



            # ambiguous instead of letting one account delta teach several



            # unrelated actions as if it belonged to each one.



            account_mark = _ledger_account_mark(row)



            if account_mark is not None:



                state["account_balance_after_close"] = float(account_mark[0])



                state["account_equity_after_close"] = float(account_mark[1])



                state["account_mark_sequence"] = max(



                    int(state.get("account_mark_sequence", 0) or 0),



                    int(_ledger_float(row.get("ack_sequence"), default=0.0)),



                )



                logged_at = str(row.get("logged_at", "") or "").strip()



                batch_time = (



                    event_time.isoformat()



                    if isinstance(event_time, datetime)



                    else (logged_at or f"ack:{ack_sequence or '0'}")



                )



                state["account_mark_batch_key"] = (



                    f"{batch_time}|{float(account_mark[0]):.8f}|"



                    f"{float(account_mark[1]):.8f}"



                )



                state["account_mark_attribution"] = "unique_final_deal"



            # ``profit + commission + swap`` is the immutable cashflow applied



            # by this exact DEAL_ADD.  Keep it as a separate evidence channel



            # instead of pretending a shared balance/equity snapshot belongs



            # exclusively to this position.



            deal_ticket_text = str(row.get("deal_ticket", "") or "").strip()



            close_cashflow = profit + commission + swap



            if (



                deal_ticket_text not in {"", "0", "0.0"}



                and np.isfinite(close_cashflow)



            ):



                state["account_cashflow_delta"] = float(close_cashflow)



                state["account_cashflow_deal_ticket"] = deal_ticket_text



                state["account_cashflow_attribution"] = "unique_final_deal"



        risk = _ledger_float(row.get("initial_risk"))



        if risk > 1e-12:



            state["initial_risk"] = max(float(state["initial_risk"]), risk)



        if "position_final_close" in row:



            state["saw_final_field"] = True



            state["final_close"] = bool(state["final_close"] or _ledger_bool(row.get("position_final_close")))



        age = _ledger_float(row.get("age_bars"), default=-1.0)



        if age >= 0.0:



            current_age = state.get("duration_bars")



            state["duration_bars"] = max(float(current_age or 0.0), age)







    equity_risk_unit = max(1e-9, float(initial_equity) * 0.01)



    # A shared final-deal snapshot cannot be causally assigned to every



    # position in the batch.  Preserve the numeric mark for audit, but make



    # the account-impact verifier downgrade it to partial evidence.  The



    # immutable position P/L remains fully usable for signed reward learning.



    account_mark_groups: dict[str, list[str]] = {}



    for ticket, state in by_position.items():



        batch_key = str(state.get("account_mark_batch_key", "") or "")



        if batch_key:



            account_mark_groups.setdefault(batch_key, []).append(ticket)



    for tickets in account_mark_groups.values():



        if len(set(tickets)) <= 1:



            continue



        for ticket in set(tickets):



            state = by_position.get(ticket)



            if state is not None:



                state["account_mark_attribution"] = "ambiguous_same_close_batch"







    # A cashflow can be promoted to per-deal account evidence only when the



    # lifecycle consists of exactly one final close.  For partial closes the



    # position-level total is still a valid signed reward, but it is not a



    # single deal cashflow and must remain partial.



    for state in by_position.values():



        cashflow = state.get("account_cashflow_delta")



        deal_ticket = str(state.get("account_cashflow_deal_ticket", "") or "").strip()



        if (



            int(state.get("close_count", 0) or 0) == 1



            and int(state.get("partial_close_count", 0) or 0) == 0



            and deal_ticket not in {"", "0", "0.0"}



            and cashflow is not None



            and np.isfinite(float(cashflow))



        ):



            state["account_cashflow_attribution"] = "unique_final_deal"



        elif cashflow is not None:



            state["account_cashflow_attribution"] = "ambiguous_position_close_batch"







    lifecycles: list[MT5PositionLifecycle] = []



    for ticket, state in by_position.items():



        if int(state["close_count"]) <= 0:



            continue



        entry_time = state.get("entry_time")



        exit_time = state.get("exit_time")



        duration = state.get("duration_bars")



        if duration is None and isinstance(entry_time, datetime) and isinstance(exit_time, datetime):



            duration = max(0.0, (exit_time - entry_time).total_seconds() / max(1.0, bar_seconds))



        final_close = bool(state["final_close"])



        # Legacy v4 cannot mark a final close.  Keep it for diagnostics and



        # proxy ranking; production eligibility below will expose the reduced



        # integrity rather than pretending the sample is exact.



        if not bool(state["saw_final_field"]):



            final_close = True



        initial_risk = float(state["initial_risk"])



        net_pnl = float(state["net_pnl"])



        if initial_risk <= 1e-12:



            close_reason = str(state.get("close_reason", "") or "").strip().upper()



            gross_profit = float(state.get("gross_profit", 0.0) or 0.0)



            risk_basis = abs(gross_profit) if abs(gross_profit) > 1e-12 else abs(net_pnl)



            inference_method = ""



            if close_reason in {"SL", "STOP_LOSS", "STOP_OUT"} and net_pnl < -1e-12:



                initial_risk = risk_basis



                inference_method = "same_tick_sl_terminal_pnl"



            elif close_reason in {"TP", "TAKE_PROFIT"} and net_pnl > 1e-12:



                initial_risk = risk_basis / max(1e-9, float(technical_min_rr))



                inference_method = "same_tick_tp_technical_min_rr"



            if initial_risk > 1e-12:



                metadata = dict(state.get("metadata", {}) or {})



                metadata.update({



                    "initial_risk_inferred": True,



                    "initial_risk_inference_method": inference_method,



                    "technical_min_rr": float(technical_min_rr),



                })



                state["metadata"] = metadata



        realized_r = net_pnl / initial_risk if initial_risk > 1e-12 else net_pnl / equity_risk_unit



        account_context_after_close: dict[str, float] = {}



        account_balance_after_close = state.get("account_balance_after_close")



        account_equity_after_close = state.get("account_equity_after_close")



        if (



            account_balance_after_close is not None



            and account_equity_after_close is not None



            and np.isfinite(float(account_balance_after_close))



            and np.isfinite(float(account_equity_after_close))



        ):



            account_context_after_close = {



                "balance": float(account_balance_after_close),



                "equity": float(account_equity_after_close),



                "sequence": float(state.get("account_mark_sequence", 0) or 0),



            }



        lifecycles.append(MT5PositionLifecycle(



            position_ticket=ticket,



            entry_time=entry_time if isinstance(entry_time, datetime) else None,



            exit_time=exit_time if isinstance(exit_time, datetime) else None,



            net_pnl=net_pnl,



            gross_profit=float(state["gross_profit"]),



            commission=float(state["commission"]),



            swap=float(state["swap"]),



            initial_risk=initial_risk,



            realized_r=float(realized_r),



            duration_bars=float(duration) if duration is not None else None,



            close_reason=str(state["close_reason"]),



            source=str(state["source"]),



            trace_id=str(state.get("trace_id", "")),



            entry_sequence=int(state.get("entry_sequence", 0)),



            entry_count=int(state["entry_count"]),



            close_count=int(state["close_count"]),



            final_close_confirmed=final_close,



            exit_mechanism=classify_exit_mechanism(



                str(state.get("exit_kind") or state["close_reason"]),



                realized_r=float(realized_r),



                partial_close_count=int(state.get("partial_close_count", 0)),



            ).value,



            entry_price=float(state.get("entry_price", 0.0)),



            exit_price=float(state.get("exit_price", 0.0)),



            initial_sl_price=float(state.get("initial_sl_price", 0.0)),



            final_sl_price=float(state.get("final_sl_price", 0.0)),



            initial_tp_price=float(state.get("initial_tp_price", 0.0)),



            final_tp_price=float(state.get("final_tp_price", 0.0)),



            partial_close_count=int(state.get("partial_close_count", 0)),



            mfe_r=float(state.get("mfe_r", 0.0)),



            mae_r=float(state.get("mae_r", 0.0)),



            excursion_available=bool(state.get("excursion_available", False)),



            giveback_r=float(state.get("giveback_r", 0.0)),



            near_tp_ratio=float(state.get("near_tp_ratio", 0.0)),



            slippage_r=float(state.get("slippage_r", 0.0)),



            post_exit_mfe_r=state.get("post_exit_mfe_r"),



            post_exit_mae_r=state.get("post_exit_mae_r"),



            returned_to_entry=state.get("returned_to_entry"),



            original_tp_hit_after_exit=state.get("original_tp_hit_after_exit"),



            exit_kind=str(state.get("exit_kind", "") or ""),



            exit_controller=str(state.get("exit_controller", "") or ""),



            exit_trace_id=str(state.get("exit_trace_id", "") or ""),



            exit_requested_action=(



                int(state["exit_requested_action"])



                if state.get("exit_requested_action") is not None



                else -1



            ),



            post_exit_horizon_bars=state.get("post_exit_horizon_bars"),



            post_exit_hold_r=state.get("post_exit_hold_r"),



            initial_atr=float(state.get("initial_atr", 0.0)),



            side=int(state.get("side", 0)),



            proposed_sl_atr=float(state.get("proposed_sl_atr", 0.0)),



            proposed_rr=float(state.get("proposed_rr", 0.0)),



            requested_entry_price=float(state.get("requested_entry_price", 0.0)),



            requested_sl_price=float(state.get("requested_sl_price", 0.0)),



            requested_tp_price=float(state.get("requested_tp_price", 0.0)),



            requested_atr=float(state.get("requested_atr", 0.0)),



            geometry_mutation_count=int(state.get("geometry_mutation_count", 0)),



            trailing_applied=bool(state.get("trailing_applied", False)),



            pending_fill_bars=state.get("pending_fill_bars"),



            trajectory=tuple(dict(item) for item in list(state.get("trajectory", []) or [])[-512:]),



            metadata={



                **dict(state.get("metadata", {}) or {}),



                "excursion_evidence_source": str(state.get("excursion_source", "") or ""),



                "account_mark_attribution": str(



                    state.get("account_mark_attribution", "not_available") or "not_available"



                ),



                "account_cashflow_attribution": str(



                    state.get("account_cashflow_attribution", "not_available")



                    or "not_available"



                ),



                "account_cashflow_deal_ticket": str(



                    state.get("account_cashflow_deal_ticket", "") or ""



                ),



            },



            account_context_after_close=account_context_after_close,



            account_cashflow_delta=(



                float(state["account_cashflow_delta"])



                if state.get("account_cashflow_delta") is not None



                and np.isfinite(float(state["account_cashflow_delta"]))



                else None



            ),



            account_cashflow_deal_ticket=str(



                state.get("account_cashflow_deal_ticket", "") or ""



            ),



        ))







    finalised = [item for item in lifecycles if item.final_close_confirmed]



    values = np.asarray([float(item.realized_r or 0.0) for item in finalised], dtype=float)



    n = int(values.size)



    risk_coverage = float(np.mean([item.initial_risk > 1e-12 for item in finalised])) if finalised else 0.0



    complete_ratio = float(len(finalised) / max(1, len(lifecycles))) if lifecycles else 0.0



    trace_coverage = float(traced_policy_events / max(1, policy_events)) if policy_events else 0.0



    if n:



        expectancy = float(np.mean(values))



        std = float(np.std(values, ddof=1)) if n > 1 else abs(expectancy)



        expectancy_lcb = float(expectancy - 1.64 * std / max(1.0, np.sqrt(float(n))))



        wins = float(np.sum(values[values > 0.0]))



        losses = float(-np.sum(values[values < 0.0]))



        raw_pf = wins / losses if losses > 1e-12 else 5.0



        pf_shrunk = float(1.0 + (min(5.0, raw_pf) - 1.0) * n / (n + 10.0))



        tail_n = max(1, int(np.ceil(0.05 * n)))



        cvar = float(max(0.0, -np.mean(np.sort(values)[:tail_n])))



        risk_total = float(sum(item.initial_risk for item in finalised if item.initial_risk > 1e-12))



        if risk_total > 1e-12:



            net_cost_r = float(abs(sum(item.commission + item.swap for item in finalised)) / risk_total)



        else:



            net_cost_r = float(abs(sum(item.commission + item.swap for item in finalised)) / equity_risk_unit)



        immediate = [



            item for item in finalised



            if item.duration_bars is not None and item.duration_bars < max(1, int(min_holding_bars))



        ]



        immediate_ratio = float(len(immediate) / n)



    else:



        expectancy = expectancy_lcb = cvar = net_cost_r = immediate_ratio = 0.0



        pf_shrunk = 1.0







    rejected = sum(1 for state in intent_state.values() if bool(state["rejected"]) and not bool(state["accepted"]))



    failure_rate = float(rejected / max(1, len(intent_state))) if intent_state else 1.0



    schema_score = 1.0 if schema_version >= FITNESS_SCHEMA_VERSION else (0.45 if schema_version >= 4 else 0.0)



    integrity = float(np.clip(



        0.30 * schema_score



        + 0.25 * complete_ratio



        + 0.25 * risk_coverage



        + 0.20 * trace_coverage,



        0.0,



        1.0,



    ))



    sample_unit = "r" if risk_coverage >= 0.90 else ("equity_r_proxy" if n else "none")



    return MT5ExecutionSummary(



        ledger_path=path,



        ledger_available=True,



        schema_version=schema_version,



        total_events=len(rows),



        policy_events=policy_events,



        policy_intents=len(intent_state),



        rejected_intents=rejected,



        lifecycle_count=n,



        complete_lifecycle_ratio=complete_ratio,



        risk_coverage=risk_coverage,



        trace_coverage=trace_coverage,



        ledger_integrity=integrity,



        execution_failure_rate=failure_rate,



        immediate_exit_ratio=immediate_ratio,



        expectancy=expectancy,



        expectancy_lcb=expectancy_lcb,



        profit_factor_shrunk=pf_shrunk,



        cvar_5pct=cvar,



        net_cost_r=net_cost_r,



        sample_unit=sample_unit,



        lifecycles=tuple(finalised),



    )











def execution_trace_by_sequence(ledger_path: str | Path | None) -> dict[int, str]:



    """Return the strongest owner trace observed for each bridge sequence.







    Strategy Tester can emit a final report close after the bridge process has



    stopped receiving ``DEAL_ADD`` rows.  The raw ledger still contains the



    entry ``position_owner_trace_id`` (often on a filled or trajectory row),



    which is the only safe owner for a report-side fallback close.



    """



    if ledger_path is None:



        return {}



    path = Path(ledger_path)



    if not path.exists():



        return {}



    result: dict[int, str] = {}



    try:



        rows = _read_execution_rows(path)



    except Exception:



        return result



    for row in rows:



        sequence = int(_ledger_float(row.get("ack_sequence"), default=0.0))



        if sequence <= 0:



            continue



        owner = str(row.get("position_owner_trace_id", "") or row.get("trace_id", "") or "").strip()



        if not owner:



            continue



        ticket = str(row.get("position_ticket", "") or "").strip()



        strong_owner = bool(str(row.get("position_owner_trace_id", "") or "").strip()) or ticket not in {"", "0", "0.0"}



        if strong_owner or sequence not in result:



            result[sequence] = owner



    return result











def merge_report_lifecycles(



    native_lifecycles: Sequence[MT5PositionLifecycle],



    report_lifecycles: Sequence[MT5PositionLifecycle],



    *,



    trace_by_sequence: Mapping[int, str] | None = None,



) -> tuple[MT5PositionLifecycle, ...]:



    """Join report closes to native owner lifecycles without duplicate credit.







    Native execution-ledger rows remain authoritative whenever a report close



    can be matched uniquely by owner trace, close time and realized PnL.  A



    report row that has no native close is retained only as an explicitly



    marked fallback; its owner trace must come from the raw ledger or remain



    empty (and therefore ineligible for owner-routed learning).



    """



    native = tuple(native_lifecycles or ())



    report = tuple(report_lifecycles or ())



    trace_by_sequence = dict(trace_by_sequence or {})



    used_native: set[int] = set()



    merged: list[MT5PositionLifecycle] = []







    def close_time_matches(left: MT5PositionLifecycle, right: MT5PositionLifecycle) -> bool:



        if left.exit_time is None or right.exit_time is None:



            return True



        return abs((left.exit_time - right.exit_time).total_seconds()) <= 2.0







    for report_item in report:



        pnl_candidates: list[int] = []



        for index, native_item in enumerate(native):



            if index in used_native:



                continue



            if not np.isclose(



                float(native_item.net_pnl),



                float(report_item.net_pnl),



                rtol=1e-8,



                atol=1e-4,



            ):



                continue



            if report_item.trace_id and native_item.trace_id and report_item.trace_id != native_item.trace_id:



                # A report comment may identify the close action rather than



                # the entry owner; fall through to PnL/time matching below.



                pass



            pnl_candidates.append(index)



        # Report timestamps can be shifted by a tester/bridge hand-off even



        # when the realized PnL is exact.  Use time only to disambiguate equal



        # PnL rows; a unique PnL is a stronger identity than wall-clock time.



        candidates = pnl_candidates



        if len(pnl_candidates) > 1:



            timed = [index for index in pnl_candidates if close_time_matches(native[index], report_item)]



            if len(timed) == 1:



                candidates = timed



        if len(candidates) == 1:



            index = candidates[0]



            used_native.add(index)



            native_item = native[index]



            metadata = dict(native_item.metadata or {})



            metadata.update({



                "report_lifecycle_join": True,



                "report_lifecycle_pnl": float(report_item.net_pnl),



                "report_lifecycle_position_ticket": str(report_item.position_ticket),



            })



            merged.append(replace(native_item, metadata=metadata))



            continue







        sequence = int(report_item.entry_sequence or 0)



        owner = str(report_item.trace_id or trace_by_sequence.get(sequence, "") or "").strip()



        metadata = dict(report_item.metadata or {})



        metadata.update({



            "report_fallback": True,



            "report_fallback_owner_resolved": bool(owner),



            "report_fallback_owner_source": (



                "report_trace" if report_item.trace_id else "execution_ledger_sequence" if owner else "unresolved"



            ),



        })



        merged.append(replace(report_item, trace_id=owner, metadata=metadata))







    merged.extend(native[index] for index in range(len(native)) if index not in used_native)



    return tuple(merged)











def replace_execution_summary_lifecycles(



    summary: MT5ExecutionSummary,



    lifecycles: Sequence[MT5PositionLifecycle],



    *,



    initial_equity: float,



    min_holding_bars: int,



) -> MT5ExecutionSummary:



    """Recompute lifecycle-derived fitness fields after a verified report join."""



    finalised = tuple(item for item in tuple(lifecycles or ()) if item.final_close_confirmed)



    if not finalised:



        return replace(summary, lifecycle_count=0, lifecycles=())



    values = np.asarray([float(item.realized_r or 0.0) for item in finalised], dtype=float)



    count = int(values.size)



    risk_coverage = float(np.mean([item.initial_risk > 1e-12 for item in finalised]))



    owner_coverage = float(np.mean([bool(str(item.trace_id or "").strip()) for item in finalised]))



    complete_ratio = float(len(finalised) / max(1, len(tuple(lifecycles or ()))))



    expectancy = float(np.mean(values))



    std = float(np.std(values, ddof=1)) if count > 1 else abs(expectancy)



    expectancy_lcb = float(expectancy - 1.64 * std / max(1.0, np.sqrt(float(count))))



    wins = float(np.sum(values[values > 0.0]))



    losses = float(-np.sum(values[values < 0.0]))



    raw_pf = wins / losses if losses > 1e-12 else 5.0



    profit_factor_shrunk = float(1.0 + (min(5.0, raw_pf) - 1.0) * count / (count + 10.0))



    tail_n = max(1, int(np.ceil(0.05 * count)))



    cvar = float(max(0.0, -np.mean(np.sort(values)[:tail_n])))



    risk_unit = max(1e-9, float(initial_equity) * 0.01)



    risk_total = float(sum(item.initial_risk for item in finalised if item.initial_risk > 1e-12))



    net_cost_r = float(



        abs(sum(item.commission + item.swap for item in finalised))



        / (risk_total if risk_total > 1e-12 else risk_unit)



    )



    immediate_count = sum(



        1 for item in finalised



        if item.duration_bars is not None and item.duration_bars < max(1, int(min_holding_bars))



    )



    integrity = min(float(summary.ledger_integrity), owner_coverage)



    return replace(



        summary,



        lifecycle_count=count,



        complete_lifecycle_ratio=complete_ratio,



        risk_coverage=risk_coverage,



        ledger_integrity=integrity,



        immediate_exit_ratio=float(immediate_count / count),



        expectancy=expectancy,



        expectancy_lcb=expectancy_lcb,



        profit_factor_shrunk=profit_factor_shrunk,



        cvar_5pct=cvar,



        net_cost_r=net_cost_r,



        sample_unit="r" if risk_coverage >= 0.90 else "equity_r_proxy",



        lifecycles=finalised,



    )











def _safe_report_expectancy(metrics: MT5BacktestMetrics) -> float:



    """Return a dimensionless fallback when no position ledger is available.







    A no-loss HTML report contains currency ``avg_win`` only; treating that as



    R makes lot size alter fitness.  The fallback deliberately caps this case



    at a 1R win assumption until the v5 ledger provides initial risk.



    """



    win_rate = float(np.clip(metrics.win_rate / 100.0, 0.0, 1.0))



    if metrics.avg_loss < -1e-12:



        rr = float(np.clip(metrics.avg_win / abs(metrics.avg_loss), 0.0, 5.0))



        return float(win_rate * rr - (1.0 - win_rate))



    return float(win_rate)











def _execution_summary_for(metrics: MT5BacktestMetrics, execution_summary: MT5ExecutionSummary | None = None) -> MT5ExecutionSummary | None:



    if execution_summary is not None:



        return execution_summary



    summary = getattr(metrics, "execution_summary", None)



    return summary if isinstance(summary, MT5ExecutionSummary) else None











def execution_ledger_owner_expectancy_audit(



    execution_summary: MT5ExecutionSummary | None,



    *,



    min_samples: int = 6,



) -> dict[str, object]:



    """Audit final R by immutable position owner, without affecting policy.







    MT5 writes the *entry* owner trace into every later position snapshot.  A



    broker stop or take-profit close, on the other hand, naturally has



    ``source=unknown`` because the terminal, not the Python bridge, emitted



    the close transaction.  Treating that close-row source as the trade



    origin therefore erased valid policy lifecycles from the V13 report.







    This helper is deliberately post-run and read-only.  It neither mutates



    :class:`ContextualExpectancyLedger` nor feeds an action bias, LCB mask,



    selector, risk gate, or Full-RL loss.  Its only job is to make fitness and



    observability count the same policy-owned trades that canonical



    settlement already credits.



    """



    finalised = tuple(getattr(execution_summary, "lifecycles", ()) or ())



    all_values: list[float] = []



    policy_values: list[float] = []



    owner_trace_count = 0



    for lifecycle in finalised:



        if not bool(getattr(lifecycle, "final_close_confirmed", False)):



            continue



        try:



            realized_r = float(getattr(lifecycle, "realized_r", 0.0) or 0.0)



        except (TypeError, ValueError):



            continue



        if not np.isfinite(realized_r):



            continue



        all_values.append(realized_r)



        metadata = dict(getattr(lifecycle, "metadata", {}) or {})



        owner_trace_id = str(



            metadata.get("position_owner_trace_id", "")



            or getattr(lifecycle, "trace_id", "")



            or ""



        ).strip()



        if owner_trace_id:



            # The immutable owner trace is the causal proof of a policy



            # entry.  ``source`` is intentionally not used here: final



            # SL/TP events are terminal-originated and usually say unknown.



            owner_trace_count += 1



            policy_values.append(realized_r)







    all_count = len(all_values)



    policy_count = len(policy_values)



    policy_ratio = float(policy_count / max(1, all_count))



    policy = robust_expectancy(



        policy_values,



        min_samples=int(min_samples),



        policy_origin_ratio=policy_ratio,



    )



    all_origin = robust_expectancy(



        all_values,



        min_samples=int(min_samples),



        policy_origin_ratio=1.0,



    )



    return {



        "schema_version": 1,



        "audit_source": "execution_ledger_immutable_owner_v1",



        "read_only": True,



        "policy_origin_lifecycle_count": int(policy_count),



        "all_origin_lifecycle_count": int(all_count),



        "policy_origin_ratio": policy_ratio,



        "owner_trace_lifecycle_count": int(owner_trace_count),



        "policy_expectancy": policy.as_dict(),



        "all_expectancy": all_origin.as_dict(),



        # Full-RL has masks disabled.  A post-run aggregate cannot infer



        # action/context scope safely, so it must not invent a negative-scope



        # penalty from a broad policy-origin sample.



        "negative_critical_scope_count": 0,



    }











def _expectancy_audit_sample_count(audit: object) -> int:



    if not isinstance(audit, dict):



        return 0



    policy = audit.get("policy_expectancy", {})



    if not isinstance(policy, dict):



        policy = {}



    try:



        return max(0, int(audit.get("policy_origin_lifecycle_count", policy.get("sample_count", 0)) or 0))



    except (TypeError, ValueError):



        return 0











def select_mt5_evaluation_expectancy_audit(



    checkpoint_audit: dict[str, object] | None,



    ledger_owner_audit: dict[str, object] | None,



    *,



    learning_enabled: bool,



) -> dict[str, object]:



    """Choose evidence whose sampling window matches the evaluation mode."""



    checkpoint = dict(checkpoint_audit or {})



    current_window = dict(ledger_owner_audit or {})



    if not learning_enabled:



        selected = current_window



        if checkpoint:



            selected["checkpoint_expectancy_audit"] = checkpoint



        return selected



    if _expectancy_audit_sample_count(checkpoint) <= 0:



        return current_window



    selected = checkpoint



    if current_window:



        selected["execution_ledger_owner_audit"] = current_window



    return selected











def _policy_expectancy_audit(metrics: MT5BacktestMetrics) -> tuple[float | None, int, float, int]:



    """Return V13 policy-origin LCB, samples, origin ratio, and negative scopes.







    ``None`` keeps legacy/report-only tests and artifacts backward-compatible.



    A zero is a valid audited value and must not be conflated with an absent



    audit, because zero policy edge should never silently fall back to a



    bootstrap/global edge.



    """



    audit = getattr(metrics, "expectancy_audit", None)



    if not isinstance(audit, dict) or not audit:



        return None, 0, 0.0, 0



    policy = audit.get("policy_expectancy", {})



    if not isinstance(policy, dict):



        policy = {}



    try:



        lcb = float(policy.get("expectancy_lcb", 0.0))



        samples = int(audit.get("policy_origin_lifecycle_count", policy.get("sample_count", 0)) or 0)



        ratio = float(audit.get("policy_origin_ratio", 0.0) or 0.0)



        negative = int(audit.get("negative_critical_scope_count", 0) or 0)



    except (TypeError, ValueError):



        return None, 0, 0.0, 0



    if not np.isfinite(lcb):



        lcb = 0.0



    return float(lcb), max(0, samples), float(np.clip(ratio, 0.0, 1.0)), max(0, negative)







def compute_mt5_expectancy(metrics: MT5BacktestMetrics) -> float:



    w = metrics.win_rate / 100.0



    if metrics.avg_loss == 0:



        return metrics.avg_win * w



    rr = metrics.avg_win / abs(metrics.avg_loss)



    return (w * rr) - (1.0 - w)











def compute_mt5_edge(metrics: MT5BacktestMetrics, cost_r: float = 0.05) -> float:



    return compute_mt5_expectancy(metrics) - cost_r











def _date_span_days(from_date: str, to_date: str) -> int:



    try:



        start = datetime.strptime(from_date, "%Y.%m.%d")



        end = datetime.strptime(to_date, "%Y.%m.%d")



    except ValueError:



        return 31



    return max(1, (end - start).days)











def _dict_expectancy(metrics: dict[str, object]) -> float:



    def _m(key: str, default: float = 0.0) -> float:



        try:



            return float(metrics.get(key, default))



        except (TypeError, ValueError):



            return default







    w = _m("win_rate") / 100.0



    avg_win = _m("avg_win")



    avg_loss = _m("avg_loss")



    if avg_loss == 0:



        return avg_win * w



    rr = avg_win / abs(avg_loss)



    return (w * rr) - (1.0 - w)











def count_bridge_rows(bridge_path: str | Path) -> int:



    path = Path(bridge_path)



    if not path.exists():



        return 0



    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:



        rows = sum(1 for _ in handle)



    return max(0, rows - 1)











def _detect_mt5_startup_failure(



    runner_dir: str | Path,



    offline_preflight: bool = False,



) -> dict[str, object] | None:



    """Detect a terminal startup failure before the EA can emit a bridge row.







    A portable MT5 runner can remain alive while its account is rejected or



    while the terminal is waiting for a trade-server synchronization that can



    never complete. The bridge loop used to report only a generic timeout in



    this state. Read the runner-owned root terminal logs (never the user's



    interactive terminal logs) and return a small, redacted diagnostic once a



    failure is newer than any later success marker.



    """







    root = Path(runner_dir)



    log_dir = root / "logs"



    if not log_dir.is_dir():



        return None



    failure: dict[str, object] | None = None



    authorization_failure: dict[str, object] | None = None



    latest_failure_line = -1



    latest_success_line = -1



    global_line_counter = 0



    for log_path in sorted(log_dir.glob("*.log"), key=lambda item: item.stat().st_mtime_ns):



        try:



            lines = log_path.read_text(encoding="utf-16", errors="replace").splitlines()



        except (OSError, UnicodeError):



            try:



                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()



            except OSError:



                continue



        for line_number, line in enumerate(lines):



            current_line_idx = global_line_counter + line_number



            lowered = line.casefold()



            if (



                "terminal synchronized with" in lowered



                or "metatester 5 started on" in lowered



                or "tester automatic testing started" in lowered



                or "automatic testing started" in lowered



                or "testing started on" in lowered



            ):



                latest_success_line = max(latest_success_line, current_line_idx)



            if "terminal crashed" in lowered or "memory access violation" in lowered:



                latest_failure_line = max(latest_failure_line, current_line_idx)



                failure = {



                    "stage": "terminal_startup",



                    "failure_type": "terminal_crashed",



                    "reason": line.strip(),



                    "log_path": str(log_path),



                }



                continue



            auth_match = re.search(



                r"authorization(?:\s+on\s+(?P<server>.+?))?\s+failed\s*\((?P<reason>[^)]+)\)",



                line,



                flags=re.IGNORECASE,



            )



            if auth_match:



                if not offline_preflight:



                    latest_failure_line = max(latest_failure_line, current_line_idx)



                    server_name = (auth_match.group("server") or "").strip(" '\"")



                    authorization_failure = {



                        "stage": "terminal_startup",



                        "failure_type": "authorization_failed",



                        "server": server_name,



                        "reason": auth_match.group("reason").strip(),



                        "log_path": str(log_path),



                    }



                    failure = dict(authorization_failure)



                continue



            if "terminal is not synchronized with the trade server" in lowered:



                if not offline_preflight:



                    latest_failure_line = max(latest_failure_line, current_line_idx)



                    failure = {



                        "stage": "terminal_startup",



                        "failure_type": "tester_not_synchronized",



                        "server": str((authorization_failure or {}).get("server", "") or ""),



                        "reason": "terminal is not synchronized with the trade server before automatic testing",



                        "log_path": str(log_path),



                    }



                    if authorization_failure is not None:



                        failure["authorization_reason"] = str(



                            authorization_failure.get("reason", "") or ""



                        )



        global_line_counter += len(lines)



    if failure is None or latest_failure_line <= latest_success_line:



        return None



    return failure











def verify_offline_cache(



    runner_dir: str | Path,



    symbol: str,



    from_date: str = "",



    to_date: str = "",



) -> dict[str, object]:



    """Inspect local Bases folder for cached historical bars (.hcc) and ticks (.tkc)."""



    root = Path(runner_dir)



    bases_dir = root / "Bases"



    if not bases_dir.is_dir():



        bases_dir = root / "Tester" / "bases"



    if not bases_dir.is_dir():



        return {



            "available": False,



            "reason": "missing_bases",



            "symbol": symbol,



            "hcc_files": 0,



            "tkc_files": 0,



        }







    # Match symbol without broker postfix if needed (e.g. BTCUSDm -> BTCUSD)



    sym_clean = symbol.replace("m", "").replace("_", "").upper()



    hcc_matches: list[Path] = []



    tkc_matches: list[Path] = []







    for server_dir in bases_dir.iterdir():



        if not server_dir.is_dir():



            continue



        history_dir = server_dir / "history"



        if history_dir.is_dir():



            for sym_dir in history_dir.iterdir():



                if sym_dir.is_dir() and (sym_clean in sym_dir.name.upper() or symbol.upper() in sym_dir.name.upper()):



                    hcc_matches.extend(sym_dir.glob("*.hcc"))



        ticks_dir = server_dir / "ticks"



        if ticks_dir.is_dir():



            for sym_dir in ticks_dir.iterdir():



                if sym_dir.is_dir() and (sym_clean in sym_dir.name.upper() or symbol.upper() in sym_dir.name.upper()):



                    tkc_matches.extend(sym_dir.glob("*.tkc"))







    total_hcc = len(hcc_matches)



    total_tkc = len(tkc_matches)



    available = (total_hcc > 0) or (total_tkc > 0)



    return {



        "available": available,



        "symbol": symbol,



        "from_date": from_date,



        "to_date": to_date,



        "hcc_files": total_hcc,



        "tkc_files": total_tkc,



        "reason": "cache_verified" if available else "no_cache_files_found",



    }











def parse_bridge_terminal_mark(bridge_path: str | Path | None) -> dict[str, object]:



    """Read the final MT5 account mark used when tester closes open positions."""



    if bridge_path is None:



        return {"available": False, "reason": "bridge_missing"}



    path = Path(bridge_path)



    if not path.exists() or not path.is_file() or path.stat().st_size == 0:



        return {"available": False, "reason": "bridge_missing", "path": str(path)}



    try:



        content = path.read_text(encoding="utf-8-sig", errors="replace").strip()



        if not content:



            return {"available": False, "reason": "bridge_empty", "path": str(path)}



        if path.suffix.lower() == ".json" or (content.startswith("{") and content.endswith("}")):



            import json



            data = json.loads(content)



            balance = float(data.get("balance", 0.0) or 0.0)



            equity = float(data.get("equity", balance) or balance)



            return {



                "available": True,



                "path": str(path),



                "balance": balance,



                "equity": equity,



                "status": data.get("status", "ok"),



                "open_positions": int(data.get("open_positions", data.get("deals", 0)) or 0),



            }



        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:



            rows = list(csv.DictReader(handle, delimiter="\t"))



        if not rows:



            return {"available": False, "reason": "bridge_empty", "path": str(path)}



        row = rows[-1]



        balance = float(row.get("balance", 0.0) or 0.0)



        equity = float(row.get("equity", balance) or balance)



        positions = max(0, int(float(row.get("positions", 0) or 0)))



        if not np.isfinite([balance, equity]).all():



            raise ValueError("non-finite bridge terminal mark")



        position_marks: list[dict[str, object]] = []



        for raw_position in str(row.get("positions_pipe", "") or "").split("^"):



            parts = raw_position.split(":")



            if len(parts) < 8 or not str(parts[0]).strip():



                continue



            try:



                position_marks.append({



                    "position_ticket": str(parts[0]),



                    "side": int(float(parts[1])),



                    "volume": float(parts[2]),



                    "entry_price": float(parts[3]),



                    "sl_price": float(parts[4]),



                    "tp_price": float(parts[5]),



                    "unrealized_r": float(parts[6]),



                    "age_bars": float(parts[7]),



                    "mfe_r": float(parts[8]) if len(parts) > 8 else float(parts[6]),



                    "mae_r": float(parts[9]) if len(parts) > 9 else float(parts[6]),



                    "giveback_r": float(parts[16]) if len(parts) > 16 else 0.0,



                    "owner_context": str(parts[17]) if len(parts) > 17 else "",



                    "owner_expert": str(parts[18]) if len(parts) > 18 else "",



                    "owner_policy": str(parts[19]) if len(parts) > 19 else "",



                    "owner_trace_id": str(parts[20]) if len(parts) > 20 else "",



                })



            except (TypeError, ValueError):



                continue



        return {



            "available": True,



            "path": str(path),



            "time": str(row.get("time", "") or ""),



            "balance": balance,



            "equity": equity,



            "open_positions": positions,



            "open_pnl": float(equity - balance) if positions > 0 else 0.0,



            "positions_pipe": str(row.get("positions_pipe", "") or ""),



            "open_position_marks": position_marks,



        }



    except Exception as exc:



        return {



            "available": False,



            "path": str(path),



            "reason": f"bridge_terminal_mark_error:{type(exc).__name__}",



        }











def settle_terminal_open_position_lifecycles(



    execution_summary: MT5ExecutionSummary,



    ledger_path: str | Path | None,



    terminal_mark: Mapping[str, object] | None,



    *,



    initial_equity: float,



    min_holding_bars: int = 3,



) -> MT5ExecutionSummary:



    """Settle tester-end open positions at the final MT5 mark for learning.







    Strategy Tester includes floating PnL in its final report but does not



    always emit a closing deal before EA deinit. Without this settlement the



    report sees the outcome while the decision policy receives no lifecycle



    credit. The bridge position snapshot already carries exact unrealized R



    and owner trace, while the execution ledger supplies initial risk and



    entry provenance.



    """



    mark = dict(terminal_mark or {})



    raw_marks = list(mark.get("open_position_marks", []) or [])



    if not raw_marks:



        return execution_summary



    path = Path(ledger_path) if ledger_path else None



    rows = _read_execution_rows(path) if path is not None and path.exists() else []



    entry_by_ticket: dict[str, dict[str, object]] = {}



    for row in rows:



        ticket = str(row.get("position_ticket", "") or "").strip()



        if not ticket or ticket in {"0", "0.0"}:



            continue



        state = entry_by_ticket.setdefault(ticket, {



            "initial_risk": 0.0,



            "trace_id": "",



            "entry_sequence": 0,



            "entry_time": None,



            "metadata": {},



        })



        risk = _ledger_float(row.get("initial_risk"))



        if risk > 1e-12:



            state["initial_risk"] = max(float(state["initial_risk"]), risk)



        filled = _ledger_bool(row.get("filled"))



        trace_id = str(



            row.get("position_owner_trace_id", "")



            or row.get("trace_id", "")



            or row.get("owner_trace_id", "")



            or ""



        ).strip()



        if trace_id and (filled or not str(state["trace_id"])):



            state["trace_id"] = trace_id



            state["entry_sequence"] = int(_ledger_float(row.get("ack_sequence"), 0.0))



        if filled:



            event_time = _ledger_time(row.get("logged_at"))



            current_time = state.get("entry_time")



            if current_time is None or (



                event_time is not None and isinstance(current_time, datetime) and event_time < current_time



            ):



                state["entry_time"] = event_time



        state["metadata"].update({



            key: value



            for key, value in row.items()



            if key in {



                "intent_context", "intent_expert", "intent_policy",



                "position_owner_context", "position_owner_expert", "position_owner_policy",



                "position_owner_trace_id", "symbol", "side", "direction",



            } and str(value).strip()



        })







    existing_tickets = {str(item.position_ticket) for item in execution_summary.lifecycles}



    terminal_time = _ledger_time(mark.get("time"))



    synthetic: list[MT5PositionLifecycle] = []



    for raw in raw_marks:



        position = dict(raw or {})



        ticket = str(position.get("position_ticket", "") or "").strip()



        if not ticket or ticket in existing_tickets:



            continue



        entry = entry_by_ticket.get(ticket, {})



        initial_risk = float(entry.get("initial_risk", 0.0) or 0.0)



        trace_id = str(position.get("owner_trace_id", "") or entry.get("trace_id", "") or "")



        realized_r = float(position.get("unrealized_r", 0.0) or 0.0)



        if initial_risk <= 1e-12 or not trace_id or not np.isfinite(realized_r):



            continue



        metadata = dict(entry.get("metadata", {}) or {})



        metadata.update({



            "terminal_mark_settlement": True,



            "terminal_mark_time": str(mark.get("time", "") or ""),



            "position_owner_context": str(position.get("owner_context", "") or metadata.get("position_owner_context", "")),



            "position_owner_expert": str(position.get("owner_expert", "") or metadata.get("position_owner_expert", "")),



            "position_owner_policy": str(position.get("owner_policy", "") or metadata.get("position_owner_policy", "")),



            "position_owner_trace_id": trace_id,



        })



        synthetic.append(MT5PositionLifecycle(



            position_ticket=ticket,



            entry_time=entry.get("entry_time") if isinstance(entry.get("entry_time"), datetime) else None,



            exit_time=terminal_time,



            net_pnl=float(realized_r * initial_risk),



            gross_profit=float(realized_r * initial_risk),



            commission=0.0,



            swap=0.0,



            initial_risk=initial_risk,



            realized_r=realized_r,



            duration_bars=float(position.get("age_bars", 0.0) or 0.0),



            close_reason="TESTER_TERMINAL_MARK",



            source="mt5_terminal_mark",



            trace_id=trace_id,



            entry_sequence=int(entry.get("entry_sequence", 0) or 0),



            entry_count=1,



            close_count=1,



            final_close_confirmed=True,



            exit_mechanism="tester_terminal_mark",



            entry_price=float(position.get("entry_price", 0.0) or 0.0),



            exit_price=0.0,



            initial_sl_price=float(position.get("sl_price", 0.0) or 0.0),



            final_sl_price=float(position.get("sl_price", 0.0) or 0.0),



            initial_tp_price=float(position.get("tp_price", 0.0) or 0.0),



            final_tp_price=float(position.get("tp_price", 0.0) or 0.0),



            mfe_r=float(position.get("mfe_r", 0.0) or 0.0),



            mae_r=float(position.get("mae_r", 0.0) or 0.0),



            giveback_r=float(position.get("giveback_r", 0.0) or 0.0),



            metadata=metadata,



        ))



    if not synthetic:



        return execution_summary







    lifecycles = tuple(execution_summary.lifecycles) + tuple(synthetic)



    values = np.asarray([float(item.realized_r or 0.0) for item in lifecycles], dtype=float)



    n = int(values.size)



    expectancy = float(np.mean(values))



    std = float(np.std(values, ddof=1)) if n > 1 else abs(expectancy)



    expectancy_lcb = float(expectancy - 1.64 * std / max(1.0, np.sqrt(float(n))))



    wins = float(np.sum(values[values > 0.0]))



    losses = float(-np.sum(values[values < 0.0]))



    raw_pf = wins / losses if losses > 1e-12 else 5.0



    pf_shrunk = float(1.0 + (min(5.0, raw_pf) - 1.0) * n / (n + 10.0))



    tail_n = max(1, int(np.ceil(0.05 * n)))



    cvar = float(max(0.0, -np.mean(np.sort(values)[:tail_n])))



    risk_total = float(sum(item.initial_risk for item in lifecycles if item.initial_risk > 1e-12))



    equity_risk_unit = max(1e-9, float(initial_equity) * 0.01)



    net_cost_r = float(



        abs(sum(item.commission + item.swap for item in lifecycles))



        / (risk_total if risk_total > 1e-12 else equity_risk_unit)



    )



    immediate = [



        item for item in lifecycles



        if item.duration_bars is not None and item.duration_bars < max(1, int(min_holding_bars))



    ]



    risk_coverage = float(np.mean([item.initial_risk > 1e-12 for item in lifecycles]))



    return replace(



        execution_summary,



        lifecycle_count=n,



        complete_lifecycle_ratio=1.0,



        risk_coverage=risk_coverage,



        ledger_integrity=float(min(1.0, execution_summary.ledger_integrity)),



        immediate_exit_ratio=float(len(immediate) / n),



        expectancy=expectancy,



        expectancy_lcb=expectancy_lcb,



        profit_factor_shrunk=pf_shrunk,



        cvar_5pct=cvar,



        net_cost_r=net_cost_r,



        sample_unit="r" if risk_coverage >= 0.90 else "equity_r_proxy",



        lifecycles=lifecycles,



    )











def mt5_structural_j(



    metrics: MT5BacktestMetrics,



    backtest_days: int = 31,



    min_daily_trades: int = 0,



    max_daily_trades: int = 0,



    complexity: float = 0.0,



    *,



    execution_summary: MT5ExecutionSummary | None = None,



    lambda_dd: float = 2.0,



    lambda_tail: float = 0.20,



    lambda_churn: float = 0.25,



    lambda_execution: float = 0.20,



    lambda_cost: float = 0.10,



    lambda_cmp: float = 0.01,



) -> float:



    """Dimensionless MT5 selection objective.







    v5 prioritises finalised *position* lifecycles from the execution ledger.



    Report-only metrics remain a diagnostic fallback, but they never convert a



    currency-denominated no-loss average into a fake R-multiple.



    """



    days = max(1, int(backtest_days))



    summary = _execution_summary_for(metrics, execution_summary)



    policy_lcb, policy_samples, policy_ratio, negative_context_scopes = _policy_expectancy_audit(metrics)



    if summary is not None and summary.has_position_fitness:



        sample_count = summary.lifecycle_count



        # Policy-origin contextual LCB is the principal objective whenever



        # lifecycle credit supplied it.  Global ledger LCB remains a fallback



        # only for legacy runs that have no action attribution audit.



        expectancy = float(np.clip(policy_lcb if policy_lcb is not None else summary.expectancy_lcb, -3.0, 3.0))



        profit_factor = float(np.clip(summary.profit_factor_shrunk, 0.20, 5.0))



        tail = float(np.clip(summary.cvar_5pct, 0.0, 5.0))



        immediate = float(np.clip(summary.immediate_exit_ratio, 0.0, 1.0))



        execution_failure = float(np.clip(summary.execution_failure_rate, 0.0, 1.0))



        net_cost_r = float(np.clip(summary.net_cost_r, 0.0, 3.0))



        # Cadence is no longer a reward target.  It only guards against a



        # pathological event storm; sample sufficiency is handled separately.



        avg_daily = sample_count / float(days)



        upper_turnover_gap = (



            max(0.0, avg_daily - float(max_daily_trades)) / max(1.0, float(max_daily_trades))



            if int(max_daily_trades) > 0 else 0.0



        )



        source_penalty = max(0.0, 0.90 - summary.risk_coverage) * 0.30



        if policy_lcb is not None:



            # Bootstrap/mask-heavy runs cannot borrow the global edge as if it



            # were policy skill.  This penalty is intentionally smooth; the



            # hard promotion gate of the context-expert system remains the fail-safe.



            source_penalty += max(0.0, 0.60 - policy_ratio) * 0.75



            # Negative scopes remain a hard eligibility failure below, but a



            # linear rank penalty made a large, inspectable audit report clip



            # the whole fitness to -100 and saturated learning feedback. Keep



            # ranking pressure bounded so later trials can still improve.



            source_penalty += 0.20 * float(np.tanh(float(negative_context_scopes) / 8.0))



    else:



        sample_count = int(metrics.total_trades)



        expectancy = float(np.clip(_safe_report_expectancy(metrics), -3.0, 3.0))



        profit_factor = float(np.clip(metrics.profit_factor, 0.20, 5.0))



        loss_rate = float(np.clip(1.0 - metrics.win_rate / 100.0, 0.0, 1.0))



        tail = loss_rate * (1.0 if metrics.avg_loss < 0.0 else 0.0)



        immediate = 0.0



        execution_failure = 0.0



        net_cost_r = 0.0



        avg_daily = sample_count / float(days)



        upper_turnover_gap = (



            max(0.0, avg_daily - float(max_daily_trades)) / max(1.0, float(max_daily_trades))



            if int(max_daily_trades) > 0 else 0.0



        )



        source_penalty = 0.0







    pf_edge = float(np.log(profit_factor))



    drawdown = float(np.clip(metrics.equity_drawdown_pct / 100.0, 0.0, 1.0))



    return float(



        expectancy



        + 0.20 * pf_edge



        - lambda_dd * drawdown



        - lambda_tail * tail



        - lambda_churn * immediate



        - lambda_execution * execution_failure



        - lambda_cost * net_cost_r



        - 0.08 * upper_turnover_gap



        - lambda_cmp * float(np.log1p(max(0.0, complexity)))



        - source_penalty



    )











def mt5_learning_signal(rank_fitness: float, baseline_fitness: float = 0.0, scale: float = 25.0) -> float:



    """Bound *relative* quality before it changes neural plasticity."""



    safe_scale = max(1e-9, float(scale))



    return float(np.tanh((float(rank_fitness) - float(baseline_fitness)) / safe_scale))











def mt5_fitness_breakdown(



    metrics: MT5BacktestMetrics,



    backtest_days: int = 31,



    min_daily_trades: int = 0,



    max_daily_trades: int = 0,



    complexity: float = 0.0,



    *,



    execution_summary: MT5ExecutionSummary | None = None,



    require_execution_ledger: bool = False,



    min_position_samples: int | None = None,



    max_immediate_exit_ratio: float = 0.99,



    max_execution_failure_rate: float = 0.20,



    require_terminal_flat: bool = False,



) -> MT5FitnessBreakdown:



    """Separate MT5 eligibility, ranking score and bounded learning signal."""



    days = max(1, int(backtest_days))



    summary = _execution_summary_for(metrics, execution_summary)



    has_ledger = bool(summary is not None and summary.has_position_fitness)



    sample_count = int(summary.lifecycle_count) if has_ledger else int(metrics.total_trades)



    # A sample floor is deliberately distinct from the old "must trade N/day"



    # pressure.  It can be reached across the window without rewarding churn.



    if min_position_samples is None:



        cadence_hint = max(0, int(min_daily_trades)) * days



        min_position_samples = max(8, min(40, int(np.ceil(0.5 * cadence_hint)))) if cadence_hint else 8



    required = max(1, int(min_position_samples))



    sample_gate = float(np.sqrt(min(1.0, sample_count / float(required)))) if sample_count > 0 else 0.0



    j = mt5_structural_j(



        metrics,



        days,



        min_daily_trades,



        max_daily_trades,



        complexity,



        execution_summary=summary,



    )



    score = 100.0 * j * sample_gate



    reasons: list[str] = []



    if sample_count <= 0:



        reasons.append("no_finalised_position_lifecycles" if has_ledger else "no_trades")



    if sample_count < required:



        reasons.append(f"insufficient_position_samples:{sample_count}<{required}")



        # A small all-win sample is not rankable.  Do not let a report-only



        # PF=999 artefact outrank a validated lifecycle sample.



        score = min(score - 50.0 * (1.0 - sample_gate), -10.0 * (1.0 - sample_gate))



    reconciliation = getattr(metrics, "reconciliation", None)



    terminal_open_positions = int(



        getattr(reconciliation, "terminal_open_positions", 0) or 0



    ) if isinstance(reconciliation, MT5ReportLedgerReconciliation) else 0



    if require_terminal_flat and terminal_open_positions > 0:



        reasons.append(f"terminal_open_positions_nonzero:{terminal_open_positions}")



        # A terminal mark is valid accounting evidence, but not a fully



        # closed validation window. Keep the result visible for diagnosis



        # while making it impossible to promote by scalar fitness alone.



        score = -100.0



    reconciliation_failed = bool(



        isinstance(reconciliation, MT5ReportLedgerReconciliation)



        and reconciliation.available



        and not reconciliation.reconciled



    )



    if reconciliation_failed:



        details = ";".join(reconciliation.reasons) or "unknown"



        reasons.append(f"report_execution_mismatch:{details}")



        # Do not falsely classify a report parser/transport mismatch as a



        # negative PnL strategy. This run is invalid evidence and cannot rank.



        score = -100.0



    elif float(metrics.net_profit) <= 0.0:



        reasons.append("net_profit_not_positive")



        score -= 50.0



    if float(metrics.equity_drawdown_pct) > 10.0:



        reasons.append("drawdown_above_10pct")



        score -= 10.0 * (float(metrics.equity_drawdown_pct) - 10.0)



    if has_ledger:



        assert summary is not None



        if summary.ledger_integrity < 0.70:



            reasons.append(f"ledger_integrity_below_0.70:{summary.ledger_integrity:.3f}")



            score -= 25.0 * (0.70 - summary.ledger_integrity) / 0.70



        if summary.immediate_exit_ratio > max_immediate_exit_ratio:



            reasons.append(f"immediate_exit_ratio_above_{max_immediate_exit_ratio:.2f}:{summary.immediate_exit_ratio:.3f}")



            score -= 25.0 * (summary.immediate_exit_ratio - max_immediate_exit_ratio)



        if summary.execution_failure_rate > max_execution_failure_rate:



            reasons.append(f"execution_failure_rate_above_{max_execution_failure_rate:.2f}:{summary.execution_failure_rate:.3f}")



            score -= 25.0 * (summary.execution_failure_rate - max_execution_failure_rate)



        if summary.risk_coverage < 0.90:



            reasons.append(f"initial_risk_coverage_below_0.90:{summary.risk_coverage:.3f}")



        policy_lcb, policy_samples, policy_ratio, negative_context_scopes = _policy_expectancy_audit(metrics)



        if policy_lcb is not None:



            min_policy_samples = max(3, int(np.ceil(0.60 * required)))



            if policy_samples < min_policy_samples:



                reasons.append(f"policy_origin_samples_insufficient:{policy_samples}<{min_policy_samples}")



            if policy_ratio < 0.60:



                reasons.append(f"policy_origin_ratio_below_0.60:{policy_ratio:.3f}")



            if policy_samples >= min_policy_samples and policy_lcb <= 0.0:



                reasons.append(f"policy_origin_expectancy_lcb_nonpositive:{policy_lcb:.6g}")



            if negative_context_scopes > 0:



                reasons.append(f"negative_context_action_scopes:{negative_context_scopes}")



    elif require_execution_ledger:



        reasons.append("execution_ledger_missing_or_no_finalised_lifecycle")



        # Runtime MT5 selection must never rank report-only results above a



        # reconciled execution ledger.  This is an explicit hard reject.



        score = -100.0







    # Keep a finite range so scalar feedback cannot saturate structural learning.



    rank_score = float(np.clip(score, -100.0, 100.0))



    policy_lcb, policy_samples, policy_ratio, negative_context_scopes = _policy_expectancy_audit(metrics)



    expectancy = float(policy_lcb) if policy_lcb is not None else (float(summary.expectancy_lcb) if has_ledger and summary is not None else _safe_report_expectancy(metrics))



    cvar = float(summary.cvar_5pct) if has_ledger and summary is not None else 0.0



    immediate = float(summary.immediate_exit_ratio) if has_ledger and summary is not None else 0.0



    failure = float(summary.execution_failure_rate) if has_ledger and summary is not None else 0.0



    integrity = float(summary.ledger_integrity) if has_ledger and summary is not None else 0.0



    return MT5FitnessBreakdown(



        schema_version=FITNESS_SCHEMA_VERSION,



        rank_score=rank_score,



        structural_j=float(j),



        learning_signal=mt5_learning_signal(rank_score),



        eligible=not reasons,



        eligibility_reasons=tuple(reasons),



        sample_count=sample_count,



        sample_gate=sample_gate,



        source=("execution_ledger_unreconciled" if reconciliation_failed else "execution_ledger") if has_ledger else "report_fallback",



        expectancy=expectancy,



        cvar_5pct=cvar,



        immediate_exit_ratio=immediate,



        execution_failure_rate=failure,



        ledger_integrity=integrity,



    )











def mt5_fitness(



    metrics: MT5BacktestMetrics,



    backtest_days: int = 31,



    min_daily_trades: int = 0,



    max_daily_trades: int = 0,



    complexity: float = 0.0,



    *,



    execution_summary: MT5ExecutionSummary | None = None,



    require_execution_ledger: bool = False,



    min_position_samples: int | None = None,



    require_terminal_flat: bool = False,



) -> float:



    """Return the bounded rank score.  Inspect ``mt5_fitness_breakdown`` for gates."""



    return mt5_fitness_breakdown(



        metrics,



        backtest_days=backtest_days,



        min_daily_trades=min_daily_trades,



        max_daily_trades=max_daily_trades,



        complexity=complexity,



        execution_summary=execution_summary,



        require_execution_ledger=require_execution_ledger,



        min_position_samples=min_position_samples,



        require_terminal_flat=require_terminal_flat,



    ).rank_score







def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:



    """Publish a complete MT5 launcher/config file in one rename.







    Portable terminals can be spawned immediately after the Python writer



    returns. A direct write leaves a small window where MT5 observes a missing



    or partially materialized config/set file, then falls back to its stale



    profile and loops in the login terminal instead of starting Tester.



    """



    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")



    try:



        with temporary.open("w", encoding=encoding, newline="\n") as handle:



            handle.write(content)



            handle.flush()



            os.fsync(handle.fileno())



        os.replace(temporary, path)



    finally:



        try:



            temporary.unlink()



        except FileNotFoundError:



            pass











def write_tester_ini(config: MT5BacktestConfig, report_name: str, ini_path: str | Path) -> Path:



    ini = Path(ini_path)



    ini.parent.mkdir(parents=True, exist_ok=True)



    # ExecutionMode is the order-execution delay, not the tick-generation



    # model. Keep normal execution (0) unless a caller adds a future explicit



    # delay field.



    login_part = ""



    model_val = config.model



    content = f"""[Common]



ProxyEnable=0



NewsEnable=0







[Experts]



AllowLiveTrading=1



AllowDllImport=0



Enabled=1



Account=0



Profile=0







[Tester]



Expert={config.expert}



ExpertParameters={config.expert}.set



Symbol={config.symbol}



Period={config.period}



{login_part}Deposit={config.deposit:g}



Currency=USD



Leverage={config.leverage}



Model={model_val}



ExecutionMode=0



Optimization=0



OptimizationCriterion=0



FromDate={config.from_date}



ToDate={config.to_date}



ForwardMode=0



Report=Reports\\{report_name}



ReplaceReport=1



ShutdownTerminal={1 if config.shutdown_terminal else 0}



Visual={1 if config.visual else 0}



UseLocal=1



UseRemote=0



UseCloud=0



"""



    _atomic_write_text(ini, content)



    return ini











def write_tester_set(config: MT5BacktestConfig, set_path: str | Path) -> Path:



    """Write only the input contract declared by the canonical MT5 EA.







    The tester rejects an expert when the ``.set`` file still contains input



    names from an older EA revision.  Keep this list in lockstep with the



    ``input`` declarations in ``SelfGrowing_MT5_Trainer_EA.mq5``; Python-only



    risk/learning settings belong in the Python config and must not be sent to



    MT5 as unknown inputs.



    """



    path = Path(set_path)



    path.parent.mkdir(parents=True, exist_ok=True)



    content = f"""BridgeFile={config.bridge_file}



RequestFile={config.request_file}



ActionFile={config.action_file}



ExecutionFile={config.execution_file}



EntryPreferenceFile={config.entry_preference_file}



StopRequestFile={config.stop_request_marker_file}



FeatureAuditFile={config.feature_audit_file}



FeatureSchemaHash={config.feature_schema_hash}



FeatureMtfScalingMode={config.feature_mtf_scaling_mode}



FeatureAtrPeriod={config.feature_atr_period}||{config.feature_atr_period}||1||140||N



FeatureEmaFast={config.feature_ema_fast}||{config.feature_ema_fast}||1||200||N



FeatureEmaSlow={config.feature_ema_slow}||{config.feature_ema_slow}||1||600||N



FeatureRangeWindow={config.feature_range_window}||{config.feature_range_window}||1||600||N



FeatureRegimeWindow={config.feature_regime_window}||{config.feature_regime_window}||1||900||N



FeatureAtrPercentileWindow={config.feature_atr_percentile_window}||{config.feature_atr_percentile_window}||1||2400||N



FeatureWyckoffRangeWindow={config.feature_wyckoff_range_window}||{config.feature_wyckoff_range_window}||1||1200||N



FeatureWyckoffVolumeShortWindow={config.feature_wyckoff_volume_short_window}||{config.feature_wyckoff_volume_short_window}||1||200||N



FeatureWyckoffVolumeLongWindow={config.feature_wyckoff_volume_long_window}||{config.feature_wyckoff_volume_long_window}||1||600||N



FeatureWyckoffEventHorizon={config.feature_wyckoff_event_horizon}||{config.feature_wyckoff_event_horizon}||1||600||N



FeatureServerUtcOffsetHours={config.feature_server_utc_offset_hours}||{config.feature_server_utc_offset_hours}||-23||23||N



EntryOffsetAtr={config.entry_offset_atr:g}||{config.entry_offset_atr:g}||0.000000||20.000000||N



MaxPositions={config.max_positions}||0||0||5||N



MagicNumber={config.magic_number}||{config.magic_number}||1||999999999||N



FixedSlAtr={config.fixed_sl_atr:g}||{config.fixed_sl_atr:g}||0.001000||50.000000||N



FixedTpAtr={config.fixed_tp_atr:g}||{config.fixed_tp_atr:g}||0.001000||100.000000||N



TrailActivationAtr={getattr(config, "trail_activation_atr", 1.40):g}||{getattr(config, "trail_activation_atr", 1.40):g}||0.220000||22.000000||N



TrailLockAtr={getattr(config, "trail_lock_atr", 0.45):g}||{getattr(config, "trail_lock_atr", 0.45):g}||0.120000||12.000000||N



MaxHoldingBars={config.max_holding_bars:g}||{config.max_holding_bars:g}||0.000000||500.000000||N



MinDailyTrades=0||0||0||200||N



MaxDailyTrades=0||0||0||500||N



ActionWaitMs={config.action_wait_ms}||{config.action_wait_ms}||0||60000||N



MaxDecisionPackets={config.max_decision_packets}||{config.max_decision_packets}||0||1000000||N



PendingExpiryBars={config.pending_expiry_bars}||{config.pending_expiry_bars}||1||10000||N



ExitCounterfactualHorizonBars={config.exit_counterfactual_horizon_bars}||{config.exit_counterfactual_horizon_bars}||0||500||N



EntryPreferenceHorizonBars={config.entry_preference_horizon_bars}||{config.entry_preference_horizon_bars}||1||64||N



EntryAbstractionHorizonBars={config.entry_abstraction_horizon_bars}||{config.entry_abstraction_horizon_bars}||1||600||N



EntryPreferenceIncludeHoldSnapshots={str(bool(config.entry_preference_include_hold_snapshots)).lower()}||false||0||1||N



ReplayMode={str(bool(config.mt5_failure_replay_mode)).lower()}||false||0||1||N



ReplayCaseId={str(config.mt5_failure_replay_case_id or "")}



ReplayAnchorTime={str(config.mt5_failure_replay_anchor_time or "")}



ReplayWarmupMinutes={int(config.mt5_failure_replay_warmup_minutes)}||{int(config.mt5_failure_replay_warmup_minutes)}||1||1440||N



ReplayHorizonMinutes={int(config.mt5_failure_replay_horizon_minutes)}||{int(config.mt5_failure_replay_horizon_minutes)}||1||2880||N



"""



    _atomic_write_text(path, content)



    return path











def compile_mql5_ea(



    runner_dir: str | Path,



    source_path: str | Path,



    timeout_sec: int = 60,



) -> dict[str, object]:



    """Compile an MQL5 EA source file into .ex5 using local MetaEditor64.







    Returns dict with {"compiled": bool, "log": str, "editor_path": str, "source_path": str}.



    """



    runner = Path(runner_dir)



    source = Path(source_path).resolve()



    editor = runner / "MetaEditor64.exe"



    if not editor.exists():



        fallback_editor = Path(r"C:\Program Files\MetaTrader\MetaEditor64.exe")



        if fallback_editor.exists():



            editor = fallback_editor



        else:



            return {"compiled": False, "reason": "metaeditor_not_found"}







    log_path = runner / "logs" / f"compile_{source.stem}.log"



    log_path.parent.mkdir(parents=True, exist_ok=True)



    if log_path.exists():



        try:



            log_path.unlink()



        except Exception:



            pass







    compile_started_at = time.time()
    cmd = [str(editor), "/portable", f"/compile:{source}", f"/log:{log_path}"]



    try:



        subprocess.run(cmd, cwd=str(runner), timeout=timeout_sec, check=False)



    except Exception as exc:



        return {"compiled": False, "reason": f"metaeditor_exec_error:{exc}"}







    log_content = ""



    if log_path.exists():



        try:



            log_content = log_path.read_text(encoding="utf-16-le", errors="ignore")



        except Exception:



            try:



                log_content = log_path.read_text(encoding="utf-8", errors="ignore")



            except Exception:



                pass







    success = "0 errors" in log_content
    if not success and (runner / "logs" / "metaeditor.log").exists():
        try:
            m_log = (runner / "logs" / "metaeditor.log").read_text(encoding="utf-16le", errors="ignore")
            if "0 errors" in m_log and str(source.name) in m_log:
                success = True
                log_content = m_log
        except Exception:
            pass
    if not success:
        binary_path = source.with_suffix(".ex5")
        if binary_path.is_file() and binary_path.stat().st_mtime >= max(source.stat().st_mtime, compile_started_at - 2.0):
            success = True

    return {



        "compiled": success,



        "log": log_content,



        "editor_path": str(editor),



        "source_path": str(source),



    }











class MT5StrategyTesterEvaluator:



    def __init__(self, config: MT5BacktestConfig | None = None):



        self.config = config or MT5BacktestConfig()



        self.runner_dir = Path(self.config.runner_dir).resolve()



        configured_lock_path = getattr(self.config, "mt5_global_lock_path", None)



        configured_lock_path = configured_lock_path or os.environ.get("SELF_GROWING_MT5_GLOBAL_LOCK", "")



        self.mt5_global_lock_path = (



            Path(configured_lock_path).expanduser().resolve()



            if configured_lock_path



            else (self.runner_dir / ".mt5_strategy_tester.global.lock" if ".mt5_runners" in str(self.runner_dir) else Path.cwd().resolve() / ".mt5_strategy_tester.global.lock")



        )



        self._last_mt5_global_lock_wait_seconds = 0.0



        self.common_files = self.runner_dir / "Common" / "Files"



        self.mql5_files = self.runner_dir / "MQL5" / "Files"



        self.terminal = self.runner_dir / "terminal64.exe"



        self._launched_processes: list[subprocess.Popen] = []



        self.last_context_expert_summary: dict[str, object] = {}



        self._execution_bootstrap_count = 0



        self._pending_cancel_key = ""



        self._pending_cancel_age = 0



        self._train_entry_probe_flat_count = 0



        self._train_entry_probe_count = 0



        self._train_entry_probe_day = ""



        self._train_entry_probe_day_count = 0



        self._train_entry_probe_last_source = ""



        self._position_context_extrema: dict[str, PositionContextExtremaState] = {}



        self._runtime_file_min_mtime_ns = 0



        self._report_min_mtime_ns = 0



        self._active_tester_ini_name = ""



        self._last_stop_request: dict[str, object] = {



            "requested": False,



            "path": str(self.config.stop_request_file or ""),



            "marker_file": str(self.config.stop_request_marker_file or ""),



        }



        self.regime_detector: RegimeDetector | None = None



        if bool(getattr(self.config, "regime_detector_enabled", True)):



            configured_path = getattr(self.config, "regime_detector_path", None)



            try:



                self.regime_detector = RegimeDetector.load(configured_path)



            except Exception:



                if bool(getattr(self.config, "require_regime_detector", False)):



                    raise



                self.regime_detector = RegimeDetector()



        self._reset_five_competition_state()



        if not self.config.offline_replay and not self.terminal.exists():



            raise FileNotFoundError(f"MT5 terminal not found: {self.terminal}")







    def _reset_five_competition_state(self) -> None:



        """Reset run-local strategy attribution; never persist it in a checkpoint."""



        self._five_agent_by_trace: dict[str, int] = {}



        self._five_action_by_trace: dict[str, int] = {}



        # Keep the semantic action and the economic-eligibility verdict beside



        # the wire action.  The bridge deliberately emits HOLD for inactive



        # manager/governor roles; counting those safety no-ops as policy



        # collapse makes a healthy entry owner look HOLD-only.



        self._five_policy_action_by_trace: dict[str, int] = {}



        self._five_economic_actor_eligible_by_trace: dict[str, bool] = {}



        self._five_gross_realized_r = np.zeros(5, dtype=np.float32)



        self._five_pending_cost_r = np.zeros(5, dtype=np.float32)



        self._five_closed_trade_count = np.zeros(5, dtype=np.float32)



        self._five_winning_trade_count = np.zeros(5, dtype=np.float32)



        self._five_realized_trade_r = np.zeros(5, dtype=np.float32)



        self._five_expectancy_shortfall_r = np.zeros(5, dtype=np.float32)



        self._five_tail_loss_cost_r = np.zeros(5, dtype=np.float32)



        self._five_lifecycle_route_attempts = 0



        self._five_lifecycle_route_accepted = 0



        self._five_lifecycle_route_misses = 0



        self._five_lifecycle_initial_sl_attempts = 0



        self._five_lifecycle_initial_sl_accepted = 0



        self._five_lifecycle_optimizer_updates = 0



        self._five_lifecycle_initial_sl_optimizer_samples = 0.0



        self._five_lifecycle_last_initial_sl_loss = 0.0



        self._five_lifecycle_last_rollout_steps = 0.0



        self._five_lifecycle_v4_credit_sum = 0.0



        self._five_lifecycle_net_r_credit_sum = 0.0



        self._five_lifecycle_combined_credit_sum = 0.0



        # Persistent failure-context replay telemetry. These counters live in



        # the runner snapshot so observatory rotation cannot hide whether old



        # failures are still being audited/replayed/reopened.



        self._five_failure_replay_admitted = 0



        self._five_failure_replay_audited = 0



        self._five_failure_replay_selected = 0



        self._five_failure_replay_newly_resolved = 0



        self._five_failure_replay_newly_reopened = 0



        self._five_failure_replay_bank_cases = 0



        self._five_failure_replay_bank_unresolved = 0



        self._five_failure_replay_bank_resolved = 0



        self._five_failure_replay_replayed_total = 0



        self._five_failure_replay_reopened_total = 0



        self._five_failure_replay_last_loss = 0.0



        self._five_failure_replay_last_weighted_loss = 0.0



        self._mt5_failure_replay_anchor_seen = False



        self._mt5_failure_replay_anchor_sequence = 0



        self._mt5_failure_replay_anchor_time_seen = ""



        # ``_five_settled_lifecycle_ids`` only contains lifecycles that were



        # successfully routed to an owner.  An old final close with missing



        # provenance must not remain "unsettled" forever and block the



        # account mark for a later, uniquely attributable close.  Keep a



        # separate first-observation set for that causal attribution gate;



        # unresolved lifecycles stay eligible for their normal provenance



        # retry, but no longer poison the next account-mark candidate.



        self._five_account_mark_observed_lifecycle_ids: set[str] = set()



        self._five_settled_lifecycle_ids: set[str] = set()



        # Run-local, immutable-close evidence used by the live action-veto



        # sampler. This is separate from the Codex receipt queue and the



        # legacy lifecycle optimizer.



        self._five_live_owner_outcome_trace_ids: set[str] = set()



        previous_executor = getattr(self, "_codex_teacher_executor", None)



        if isinstance(previous_executor, ThreadPoolExecutor):



            previous_executor.shutdown(wait=False, cancel_futures=True)



        self._codex_teacher_executor: ThreadPoolExecutor | None = None



        self._codex_teacher_pending: dict[



            str,



            tuple[dict[str, object], tuple[object, ...], Future[dict[str, object]]],



        ] = {}



        self._codex_teacher_submitted_ids: set[str] = set()



        # A final close is observed before the EA's delayed post-exit horizon



        # row in normal live execution.  Keep the immutable packet/traces here



        # until that row is available, so a hypothesis-required Codex case can



        # grade the realized outcome *and* the hold counterfactual instead of



        # racing the ledger refresh and silently learning from an incomplete



        # account-impact packet.



        self._codex_teacher_deferred_cases: dict[str, dict[str, object]] = {}



        # Accepted Codex updates are linked to the first bridge action sampled



        # after a verified online barrier.  Keeping this queue separate from



        # ``phase`` makes the causal claim fail closed: an update that was



        # applied asynchronously or followed by a timeout is never reported



        # as having changed a subsequent MT5 action.



        self._codex_teacher_pending_next_action_links: list[dict[str, object]] = []



        self._codex_teacher_receipts_path: Path | None = None



        self._last_codex_teacher_phase: dict[str, object] = {



            "enabled": bool(self.config.codex_teacher_enabled),



            "mode": "per_terminal_action_async_with_causal_barrier",



            "case_limit": int(self.config.codex_teacher_max_cases_per_run),



            "online_wait_sec": float(



                getattr(self.config, "codex_teacher_online_wait_sec", 120.0) or 0.0



            ),



            "max_workers": int(getattr(self.config, "codex_teacher_max_workers", 2) or 2),



            "model": str(self.config.codex_teacher_model or ""),



            "reasoning_effort": str(self.config.codex_teacher_reasoning_effort or ""),



            "executable_path": str(self.config.codex_teacher_executable or ""),



            "cli_version": str(self.config.codex_teacher_cli_version or ""),



            "preflight": dict(self.config.codex_teacher_preflight or {}),



            "collected": 0, "submitted": 0, "graded": 0,



            "updates": 0, "skips": 0, "failures": 0,



            "queue_failures": 0, "grade_failures": 0, "pending": 0,



            "deferred": 0, "deferred_released": 0,



            "deferred_forced_partial": 0,



            "counterfactual_validated": 0,



            "counterfactual_unavailable": 0,



            "hypotheses": 0, "hypotheses_confirmed": 0,



            "account_impact_records": 0,



            "account_impact_confirmed": 0, "account_impact_partial": 0,



            "account_cashflow_confirmed": 0, "account_mark_confirmed": 0,



            "account_impact_reward_sum": 0.0,



            "live_owner_outcome_records": 0,



            "live_owner_outcome_samples_by_action": {},



            "live_owner_outcome_negative_count": 0,



            "live_owner_outcome_last": {},



            "rlhf_parameter_changed": 0,



            "rlhf_internal_policy_state_changed": 0,



            "rlhf_probability_shift_count": 0,



            "rlhf_max_probability_shift": 0.0,



            "interventions_applied": 0, "hypothesis_rejections": 0,



            "online_barrier_waits": 0,



            "online_barrier_passed": 0,



            "online_barrier_timeouts": 0,



            "online_barrier_waited_sec": 0.0,



            "codex_causal_links_observed": 0,



            "codex_causal_updates_linked": 0,



            "codex_causal_hash_matches": 0,



            "codex_causal_hash_unknown": 0,



            "update_receipts": [],



            "rejected_updates": 0, "rejection_records": [],



            "recovery_snapshot_updates": 0,



            "recovery_snapshot_path": "",



            "recovery_snapshot_error": "",



            "recovery_snapshot_optimizer_state_stripped": False,



            "recovery_snapshot_optimizer_state_stripped_count": 0,



            "recovery_snapshot_optimizer_resume_mode": "continuation",



            "promotion_allowed": False,



            "evidence_class": "codex_teacher_mt5_candidate_only",



        }



        self._five_champion_seen_lifecycle_ids: set[str] = set()



        self._five_champion_closed_count = 0



        self._five_champion_net_r = 0.0



        self._five_champion_policy = None



        self._five_champion_info: dict[str, object] = {



            "available": False,



            "captured": False,



            "reason": "not_enough_positive_lifecycle_evidence",



        }



        self._five_action_distribution: dict[str, object] = {



            "available": False,



            "counts": {},



            "total": 0,



            "hold_share": 0.0,



            "effective_action_count": 0.0,



            "normalized_action_entropy": 0.0,



            "collapsed": False,



        }



        # Separate from the competition attribution set: this idempotency



        # set guards the live context-policy credit path.  The authoritative



        # ledger can be parsed repeatedly between bridge packets.



        self._five_live_context_settled_lifecycle_ids: set[str] = set()



        self._five_execution_timeout_offset = 0



        self._five_execution_timeout_header: tuple[str, ...] = ()



        self._five_execution_timeout_path = ""



        self._five_settled_pending_timeout_ids: set[str] = set()



        self._five_reward_steps = 0



        self._five_episode_started_at: datetime | None = None



        self._five_last_market_time: datetime | None = None



        self._last_rlhf_policy_updates = 0



        self._last_rlhf_policy_update_skips = 0



        self._last_rlhf_policy_update_failures = 0



        self._last_rlhf_policy_update_manifest: dict[str, object] = {



            "enabled": False,



            "updates": 0,



            "skips": 0,



            "failures": 0,



            "promotion_allowed": False,



            "evidence_class": "rlhf_policy_candidate_only",



        }



        # Keeps the phase receipt observable when the bridge is interrupted



        # before the normal finished-evaluation footer is reached.  The



        # candidate remains ineligible; this is only forensic state.



        self._active_rlhf_policy_update_manifest: dict[str, object] | None = None







    def _snapshot_rlhf_policy_update_manifest(



        self,



        manifest: Mapping[str, object] | None = None,



    ) -> None:



        """Persist the current RLHF phase receipt for complete or interrupted runs."""







        active = manifest if manifest is not None else self._active_rlhf_policy_update_manifest



        if not isinstance(active, Mapping):



            return



        snapshot = dict(active)



        snapshot["updates"] = int(self._last_rlhf_policy_updates)



        snapshot["skips"] = int(self._last_rlhf_policy_update_skips)



        snapshot["failures"] = int(self._last_rlhf_policy_update_failures)



        self._last_rlhf_policy_update_manifest = snapshot







    def _codex_teacher_receipt_file(self, report_path: Path | None = None) -> Path:



        """Return one stable append-only receipt path for the active run."""



        if self._codex_teacher_receipts_path is not None:



            return self._codex_teacher_receipts_path



        configured = str(self.config.execution_trace_observatory or "").strip()



        receipt_base = Path(configured) if configured else (



            Path(report_path) if report_path is not None



            else Path(self.config.runner_dir) / "Reports" / "codex_teacher_live.jsonl"



        )



        run_label = "".join(



            char if char.isalnum() or char in "-_" else "_"



            for char in str(self.config.run_id or "run")[:80]



        )



        self._codex_teacher_receipts_path = receipt_base.with_name(



            f"{receipt_base.stem}.{run_label}.codex_teacher_receipts.jsonl"



        )



        return self._codex_teacher_receipts_path







    def _sync_runtime_feedback_contract(self) -> None:



        """Publish live Codex state into the candidate contract sidecar.







        ``runtime_learning_contract.json`` is created by the candidate



        launcher before MT5 starts.  The launcher cannot know the result of



        the runtime preflight or of asynchronous teacher updates while the



        tester is running, so leaving that initial snapshot untouched makes a



        live learner appear telemetry-only to the UI.  Keep this sync best



        effort: failure to refresh an observability file must never stop MT5



        or change the reward/update gates.



        """



        configured = str(self.config.execution_trace_observatory or "").strip()



        if not configured:



            return



        contract_path = Path(configured).parent / "runtime_learning_contract.json"



        if not contract_path.is_file():



            return



        try:



            raw = json.loads(contract_path.read_text(encoding="utf-8"))



            if not isinstance(raw, dict):



                return



        except (OSError, TypeError, ValueError):



            return







        phase = self._last_codex_teacher_phase



        preflight = phase.get("preflight", {})



        preflight_map = dict(preflight) if isinstance(preflight, Mapping) else {}



        preflight_passed = str(preflight_map.get("status", "") or "").lower() == "passed"



        submitted = int(phase.get("submitted", 0) or 0)



        graded = int(phase.get("graded", 0) or 0)



        hypotheses = int(phase.get("hypotheses", 0) or 0)



        hypotheses_confirmed = int(phase.get("hypotheses_confirmed", 0) or 0)



        updates = int(phase.get("updates", 0) or 0)



        failures = int(phase.get("failures", 0) or 0)



        pending = int(phase.get("pending", 0) or 0)



        effective = dict(raw.get("effective", {}) or {})



        effective.update({



            # Auto-connect is effective only after the actual preflight, not



            # merely because a PATH executable was discovered.



            "five_codex_teacher_effective": bool(



                bool(phase.get("enabled", False)) and preflight_passed



            ),



            "codex_cli_found": bool(



                str(phase.get("executable_path", "") or "").strip()



                or str(preflight_map.get("executable_path", "") or "").strip()



            ),



            "codex_preflight_passed": preflight_passed,



            "codex_receipts_valid": bool(



                graded > 0



                or str(preflight_map.get("receipt_validation", "") or "").lower()



                in {"ok", "passed"}



            ),



            "codex_hypotheses_confirmed": hypotheses_confirmed,



            "codex_optimizer_updates": updates,



            "five_codex_teacher_model": str(



                phase.get("model", "")



                or effective.get("five_codex_teacher_model", "")



                or "gpt-5.6-luna"



            ),



            "five_codex_teacher_reasoning_effort": str(



                phase.get("reasoning_effort", "")



                or effective.get("five_codex_teacher_reasoning_effort", "")



                or "medium"



            ),



            "codex_teacher_executable": str(



                phase.get("executable_path", "")



                or preflight_map.get("executable_path", "")



                or ""



            ),



            "codex_teacher_cli_version": str(



                phase.get("cli_version", "")



                or preflight_map.get("cli_version", "")



                or ""



            ),



            "five_codex_teacher_max_workers": int(



                phase.get("max_workers", getattr(self.config, "codex_teacher_max_workers", 2))



                or 2



            ),



            "five_codex_teacher_online_wait_sec": float(



                phase.get(



                    "online_wait_sec",



                    getattr(self.config, "codex_teacher_online_wait_sec", 120.0),



                )



                or 0.0



            ),



            "five_codex_teacher_learning_rate": float(



                getattr(self.config, "codex_teacher_learning_rate", 0.02) or 0.02



            ),



            "five_codex_context_similarity_threshold": float(



                getattr(self.config, "codex_context_similarity_threshold", 0.90) or 0.90



            ),



            "five_codex_context_support_floor": float(



                getattr(self.config, "codex_context_support_floor", 0.50) or 0.50



            ),



            "five_codex_global_action_bias_scale": float(



                getattr(self.config, "codex_global_action_bias_scale", 0.50)



            ),



        })



        raw["effective"] = effective



        raw["codex_preflight"] = preflight_map



        raw["runtime_status"] = "running"



        raw["updated_at"] = float(time.time())



        raw["live_codex_teacher"] = {



            "submitted": submitted,



            "graded": graded,



            "hypotheses": hypotheses,



            "hypotheses_confirmed": hypotheses_confirmed,



            "account_impact_records": int(



                phase.get("account_impact_records", 0) or 0



            ),



            "account_impact_confirmed": int(



                phase.get("account_impact_confirmed", 0) or 0



            ),



            "account_impact_partial": int(



                phase.get("account_impact_partial", 0) or 0



            ),



            "account_cashflow_confirmed": int(



                phase.get("account_cashflow_confirmed", 0) or 0



            ),



            "account_mark_confirmed": int(



                phase.get("account_mark_confirmed", 0) or 0



            ),



            "account_impact_reward_sum": float(



                phase.get("account_impact_reward_sum", 0.0) or 0.0



            ),



            "live_owner_outcome_records": int(



                phase.get("live_owner_outcome_records", 0) or 0



            ),



            "live_owner_outcome_samples_by_action": dict(



                phase.get("live_owner_outcome_samples_by_action", {}) or {}



            ),



            "live_owner_outcome_negative_count": int(



                phase.get("live_owner_outcome_negative_count", 0) or 0



            ),



            "live_owner_outcome_last": dict(



                phase.get("live_owner_outcome_last", {}) or {}



            ),



            "updates": updates,



            "interventions_applied": int(phase.get("interventions_applied", 0) or 0),



            "deferred": int(phase.get("deferred", 0) or 0),



            "deferred_released": int(phase.get("deferred_released", 0) or 0),



            "deferred_pending": int(len(self._codex_teacher_deferred_cases)),



            "deferred_forced_partial": int(



                phase.get("deferred_forced_partial", 0) or 0



            ),



            "counterfactual_validated": int(



                phase.get("counterfactual_validated", 0) or 0



            ),



            "counterfactual_unavailable": int(



                phase.get("counterfactual_unavailable", 0) or 0



            ),



            "rlhf_parameter_changed": int(



                phase.get("rlhf_parameter_changed", 0) or 0



            ),



            "rlhf_internal_policy_state_changed": int(



                phase.get("rlhf_internal_policy_state_changed", 0) or 0



            ),



            "rlhf_probability_shift_count": int(



                phase.get("rlhf_probability_shift_count", 0) or 0



            ),



            "rlhf_max_probability_shift": float(



                phase.get("rlhf_max_probability_shift", 0.0) or 0.0



            ),



            "failures": failures,



            "pending": pending,



            "online_barrier_waits": int(phase.get("online_barrier_waits", 0) or 0),



            "online_barrier_passed": int(phase.get("online_barrier_passed", 0) or 0),



            "online_barrier_timeouts": int(phase.get("online_barrier_timeouts", 0) or 0),



            "online_barrier_waited_sec": float(



                phase.get("online_barrier_waited_sec", 0.0) or 0.0



            ),



            "online_barrier_last": phase.get("online_barrier_last", {}),



            "codex_causal_links_observed": int(



                phase.get("codex_causal_links_observed", 0) or 0



            ),



            "codex_causal_updates_linked": int(



                phase.get("codex_causal_updates_linked", 0) or 0



            ),



            "codex_causal_hash_matches": int(



                phase.get("codex_causal_hash_matches", 0) or 0



            ),



            "codex_causal_hash_unknown": int(



                phase.get("codex_causal_hash_unknown", 0) or 0



            ),



            "last_codex_causal_link": phase.get("last_codex_causal_link", {}),



            "skips": int(phase.get("skips", 0) or 0),



            "last_error": str(phase.get("last_error", "") or "")[:500],



            "updated_at": float(phase.get("updated_at", time.time()) or time.time()),



        }



        payload = json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False)



        try:



            _atomic_write_text(contract_path, payload)



        except OSError as exc:



            # Match the status writer's Windows-lock fallback.  The primary



            # contract remains untouched, while the latest sidecar still



            # gives the UI a current, parseable snapshot.



            fallback = Path(str(contract_path) + ".latest.json")



            try:



                _atomic_write_text(fallback, payload)



            except OSError:



                phase["feedback_contract_write_error"] = (



                    f"{type(exc).__name__}: {exc}"[:500]



                )







    def _write_codex_teacher_status(self, report_path: Path | None = None) -> None:



        """Publish live queue/update counters without blocking MT5."""



        phase = self._last_codex_teacher_phase



        phase["pending"] = int(len(self._codex_teacher_pending))



        phase["deferred_pending"] = int(len(self._codex_teacher_deferred_cases))



        phase["updated_at"] = float(time.time())



        phase["generic_optimizer_updates"] = int(



            self._five_lifecycle_optimizer_updates + self._last_rlhf_policy_updates



        )



        phase["codex_optimizer_updates"] = int(phase.get("updates", 0) or 0)



        receipt_path = self._codex_teacher_receipt_file(report_path)



        phase["receipts_path"] = str(receipt_path)



        status_path = Path(str(receipt_path) + ".status.json")



        fallback_path = Path(str(status_path) + ".latest.json")



        phase["status_path"] = str(status_path)



        phase["status_fallback_path"] = str(fallback_path)



        phase.pop("status_write_error", None)



        phase["status_written_path"] = str(status_path)



        self._sync_runtime_feedback_contract()



        payload = json.dumps(phase, sort_keys=True, ensure_ascii=False, allow_nan=False)



        last_error: OSError | None = None



        for attempt in range(3):



            try:



                _atomic_write_text(status_path, payload)



                return



            except OSError as exc:



                last_error = exc



                if attempt < 2:



                    time.sleep(0.05 * (attempt + 1))



        phase["status_written_path"] = str(fallback_path)



        phase["status_write_error"] = (



            f"primary_locked:{type(last_error).__name__}: {last_error}"[:500]



            if last_error is not None else "primary_locked"



        )



        payload = json.dumps(phase, sort_keys=True, ensure_ascii=False, allow_nan=False)



        try:



            _atomic_write_text(fallback_path, payload)



        except OSError as exc:



            phase["status_write_error"] = f"{type(exc).__name__}: {exc}"[:500]







    def _append_codex_teacher_failure(



        self,



        packet: Mapping[str, object],



        exc: BaseException,



        *,



        report_path: Path | None = None,



        stage: str | None = None,



    ) -> None:



        """Persist one bounded failure record for every submitted packet."""



        receipt_path = self._codex_teacher_receipt_file(report_path)



        execution = getattr(exc, "execution", {})



        execution_map = dict(execution) if isinstance(execution, Mapping) else {}



        command = execution_map.get("command", [])



        record = {



            "record_type": "codex_teacher_failure",



            "timestamp": float(time.time()),



            "packet_sha256": str(packet.get("packet_sha256", "") or ""),



            "trace_id": str(packet.get("trace_id", "") or ""),



            "position_ticket": str(packet.get("position_ticket", "") or ""),



            # ``0`` is the valid HOLD action.  Do not use ``or -1`` here:



            # failure records must preserve the exact packet identity so a



            # failed HOLD can still be joined to the lifecycle/bridge ledger.



            "action": (



                int(packet["action"])



                if packet.get("action") is not None



                else -1



            ),



            "action_name": str(packet.get("action_name", "") or ""),



            "account_impact": dict(packet.get("account_impact", {}) or {})



            if isinstance(packet.get("account_impact"), Mapping) else {},



            "model": str(self.config.codex_teacher_model or ""),



            "executable_path": str(



                execution_map.get("executable_path")



                or self.config.codex_teacher_executable



                or ""



            ),



            "cli_version": str(



                execution_map.get("cli_version")



                or self.config.codex_teacher_cli_version



                or ""



            ),



            "reasoning_effort": str(



                execution_map.get("reasoning_effort")



                or self.config.codex_teacher_reasoning_effort



                or ""



            ),



            "stage": str(



                execution_map.get("stage")



                or stage



                or "unknown"



            ),



            "returncode": execution_map.get("returncode"),



            "stderr_tail": str(execution_map.get("stderr_tail", "") or "")[-2000:],



            "failure_type": type(exc).__name__,



            "error": str(exc)[:1000],



            # Keep a bounded local traceback for failures raised after a



            # structurally valid Codex receipt (for example in the learner



            # handoff). This is diagnostic evidence only; it never enters the



            # reward packet or the optimizer contract.



            "traceback_tail": traceback.format_exc()[-4000:],



            "command": command if isinstance(command, list) else [],



            "codex_execution": execution_map,



        }



        try:



            receipt_path.parent.mkdir(parents=True, exist_ok=True)



            with receipt_path.open("a", encoding="utf-8") as stream:



                stream.write(



                    json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)



                    + "\n"



                )



        except OSError as write_error:



            phase = self._last_codex_teacher_phase



            phase["failure_log_write_error"] = f"{type(write_error).__name__}: {write_error}"[:500]







    def _append_codex_teacher_update(



        self,



        packet: Mapping[str, object],



        receipt: Mapping[str, object],



        update: Mapping[str, object],



        *,



        report_path: Path | None = None,



        accepted: bool,



        account_impact_reward: float | None = None,



        account_impact_reward_source: str = "none",



    ) -> None:



        """Persist the learner result beside the immutable receipt.







        The receipt is written before the learner handoff so a crash cannot



        lose a valid Codex grade.  The learner result therefore needs its own



        append-only record; keeping it only in the bounded status snapshot



        makes an old case impossible to audit after 256 newer updates.



        """



        receipt_path = self._codex_teacher_receipt_file(report_path)



        packet_impact = packet.get("account_impact", {})



        scalar_update = {



            key: value



            for key, value in dict(update).items()



            if isinstance(value, (str, int, float, bool, type(None)))



        }



        record: dict[str, object] = {



            "record_type": "codex_teacher_update",



            "timestamp": float(time.time()),



            "accepted": bool(accepted),



            "packet_sha256": str(packet.get("packet_sha256", "") or ""),



            "receipt_sha256": str(receipt.get("receipt_sha256", "") or ""),



            "trace_id": str(packet.get("trace_id", "") or ""),



            "position_ticket": str(packet.get("position_ticket", "") or ""),



            # ``0`` is HOLD and is a real packet identity, not a missing value.



            "action": (



                int(packet["action"])



                if packet.get("action") is not None



                else -1



            ),



            "action_name": str(packet.get("action_name", "") or ""),



            "action_family": str(packet.get("action_family", "") or ""),



            "action_surface_count": packet.get("action_surface_count"),



            "model": str(self.config.codex_teacher_model or ""),



            "executable_path": str(self.config.codex_teacher_executable or ""),



            "cli_version": str(self.config.codex_teacher_cli_version or ""),



            "reasoning_effort": str(self.config.codex_teacher_reasoning_effort or ""),



            "teacher_reward": float(receipt.get("teacher_reward", 0.0) or 0.0),



            "combined_reward": scalar_update.get("rlhf_combined_reward"),



            "reward_composition": str(



                scalar_update.get(



                    "rlhf_reward_composition",



                    "teacher_account_impact_50_50_account_negative_floor_v1",



                )



                or "teacher_account_impact_50_50_account_negative_floor_v1"



            ),



            "reward_direction_source": str(



                scalar_update.get("rlhf_reward_direction_source", "teacher_reward")



                or "teacher_reward"



            ),



            "verification_status": str(



                receipt.get("verification_status", "legacy") or "legacy"



            ),



            "account_impact": dict(packet_impact)



            if isinstance(packet_impact, Mapping) else {},



            "account_impact_reward": (



                float(account_impact_reward)



                if account_impact_reward is not None else None



            ),



            "account_impact_reward_source": str(account_impact_reward_source or "none"),



            "account_impact_status": str(



                scalar_update.get("rlhf_account_impact_status", "inconclusive")



                or "inconclusive"



            ),



            "negative_account_floor_applied": bool(



                scalar_update.get("rlhf_negative_account_floor_applied", False)



            ),



            "scaled_reward": scalar_update.get("rlhf_reward_mean"),



            "advantage": scalar_update.get("rlhf_ppo_advantage_mean"),



            "advantage_mean": scalar_update.get("rlhf_ppo_advantage_mean"),



            "rlhf_ppo_update": bool(update.get("rlhf_ppo_update", False)),



            "ppo_update": bool(update.get("rlhf_ppo_update", False)),



            "rlhf_update_rejected": bool(update.get("rlhf_update_rejected", False)),



            "rlhf_rejection_reason": str(



                update.get("rlhf_rejection_reason", "") or ""



            ),



            "rlhf_parameter_scope": str(



                update.get("rlhf_parameter_scope", "") or ""



            ),



            "parameter_scope": str(update.get("rlhf_parameter_scope", "") or ""),



            "rlhf_trainable_parameter_tensors": int(



                update.get("rlhf_trainable_parameter_tensors", 0) or 0



            ),



            "trainable_parameter_tensors": int(



                update.get("rlhf_trainable_parameter_tensors", 0) or 0



            ),



            "rlhf_codex_ppo_gradient_parameter_tensors": int(



                update.get("rlhf_codex_ppo_gradient_parameter_tensors", 0) or 0



            ),



            "rlhf_auxiliary_loss": update.get("rlhf_auxiliary_loss"),



            "rlhf_factorized_geometry_update": bool(



                update.get("rlhf_factorized_geometry_update", False)



            ),



            "rlhf_factorized_geometry_heads": list(



                update.get("rlhf_factorized_geometry_heads", []) or []



            ),



            "rlhf_factorized_geometry_advantage": update.get(



                "rlhf_factorized_geometry_advantage"



            ),



            "rlhf_factorized_geometry_logprob_before": update.get(



                "rlhf_factorized_geometry_logprob_before"



            ),



            "rlhf_factorized_geometry_logprob_after": update.get(



                "rlhf_factorized_geometry_logprob_after"



            ),



            "rlhf_factorized_geometry_ratio": update.get(



                "rlhf_factorized_geometry_ratio"



            ),



            "rlhf_factorized_geometry_loss_scale": update.get(



                "rlhf_factorized_geometry_loss_scale"



            ),



            "rlhf_factorized_geometry_clip_active": bool(



                update.get("rlhf_factorized_geometry_clip_active", False)



            ),



            "rlhf_factorized_geometry_unclipped_correction": update.get(



                "rlhf_factorized_geometry_unclipped_correction"



            ),



            "rlhf_factorized_geometry_unclipped_correction_scale": update.get(



                "rlhf_factorized_geometry_unclipped_correction_scale"



            ),



            "rlhf_factorized_geometry_parameter_hash_before": str(



                update.get("rlhf_factorized_geometry_parameter_hash_before", "") or ""



            ),



            "rlhf_factorized_geometry_parameter_hash_after": str(



                update.get("rlhf_factorized_geometry_parameter_hash_after", "") or ""



            ),



            "rlhf_factorized_geometry_parameter_changed": bool(



                update.get("rlhf_factorized_geometry_parameter_changed", False)



            ),



            "rlhf_factorized_geometry_update_norm": float(



                update.get("rlhf_factorized_geometry_update_norm", 0.0) or 0.0



            ),



            "rlhf_parameter_hash_before": str(



                update.get("rlhf_parameter_hash_before", "") or ""



            ),



            "rlhf_parameter_hash_after": str(



                update.get("rlhf_parameter_hash_after", "") or ""



            ),



            "parameter_hash_before": str(



                update.get("rlhf_parameter_hash_before", "") or ""



            ),



            "parameter_hash_after": str(



                update.get("rlhf_parameter_hash_after", "") or ""



            ),



            "rlhf_parameter_changed": bool(



                update.get("rlhf_parameter_changed", False)



            ),



            "rlhf_internal_policy_state_parameter_changed": bool(



                update.get("rlhf_internal_policy_state_parameter_changed", False)



            ),



            "action_probability_before": update.get(



                "rlhf_target_action_probability_before"



            ),



            "action_probability_after": update.get(



                "rlhf_target_action_probability_after"



            ),



            "action_probability_delta": update.get(



                "rlhf_target_action_probability_delta"



            ),



            "packet_max_probability_shift": update.get(



                "rlhf_packet_max_probability_shift"



            ),



            "rlhf_direction_expected": str(



                update.get("rlhf_direction_expected", "") or ""



            ),



            "rlhf_directional_update": bool(



                update.get("rlhf_directional_update", False)



            ),



            "rlhf_codex_global_bias_gradient_frozen": bool(



                update.get("rlhf_codex_global_bias_gradient_frozen", False)



            ),



            "update": scalar_update,



        }



        try:



            receipt_path.parent.mkdir(parents=True, exist_ok=True)



            with receipt_path.open("a", encoding="utf-8") as stream:



                stream.write(



                    json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)



                    + "\n"



                )



        except (OSError, TypeError, ValueError) as write_error:



            phase = self._last_codex_teacher_phase



            phase["update_log_write_error"] = (



                f"{type(write_error).__name__}: {write_error}"[:500]



            )







    def _submit_codex_teacher_case(



        self,



        identity: str,



        lifecycle: object,



        traces: Sequence[object],



        *,



        target_trace_id: str | None = None,



    ) -> bool:



        """Queue one immutable terminal action packet for non-blocking grading."""



        phase = self._last_codex_teacher_phase



        if not bool(phase.get("enabled", False)) or identity in self._codex_teacher_submitted_ids:



            return False



        limit = int(self.config.codex_teacher_max_cases_per_run)



        if limit > 0 and len(self._codex_teacher_submitted_ids) >= limit:



            phase["limit_reached"] = True



            self._write_codex_teacher_status()



            return False



        from .codex_teacher import build_codex_teaching_packet, grade_with_codex







        packet: dict[str, object] | None = None



        try:



            packet = build_codex_teaching_packet(



                lifecycle,



                traces,



                target_trace_id=target_trace_id,



            )



            if self._codex_teacher_executor is None:



                self._codex_teacher_executor = ThreadPoolExecutor(



                    max_workers=max(



                        1, min(8, int(getattr(self.config, "codex_teacher_max_workers", 2) or 2))



                    ),



                    thread_name_prefix="codex-teacher",



                )



            future = self._codex_teacher_executor.submit(



                grade_with_codex,



                packet,



                model=str(self.config.codex_teacher_model),



                reasoning_effort=str(self.config.codex_teacher_reasoning_effort),



                timeout_sec=int(self.config.codex_teacher_timeout_sec),



                codex_exe=str(self.config.codex_teacher_executable or "") or None,



                cli_version=str(self.config.codex_teacher_cli_version or "") or None,



                require_hypothesis=bool(self.config.codex_teacher_hypothesis_required),



            )



            self._codex_teacher_pending[identity] = (packet, tuple(traces), future)



            self._codex_teacher_submitted_ids.add(identity)



            phase["collected"] = int(phase.get("collected", 0)) + 1



            phase["submitted"] = int(phase.get("submitted", 0)) + 1



            phase.pop("reason", None)



            self._write_codex_teacher_status()



            return True



        except Exception as exc:



            phase["failures"] = int(phase.get("failures", 0)) + 1



            phase["queue_failures"] = int(phase.get("queue_failures", 0)) + 1



            phase["last_error"] = f"{type(exc).__name__}: {exc}"[:500]



            if isinstance(packet, Mapping):



                self._append_codex_teacher_failure(packet, exc, stage="queue")



            self._write_codex_teacher_status()



            print(f"[MT5] codex-teacher queue failed: {phase['last_error']}", flush=True)



            return False







    def _codex_teacher_waits_for_counterfactual(self) -> bool:



        """Return whether this teacher path must wait for the hold horizon.







        The structured hypothesis contract is the explicit opt-in for causal



        teacher grading.  When it is active, submitting immediately at the



        first final-close row races the EA's delayed ``POST_EXIT_OBSERVATION``



        row and makes the receipt claim ``counterfactual_status=not_available``



        even though the ledger will publish the observation a few M1 bars



        later.  Legacy/non-hypothesis Codex modes retain their historical



        immediate realized-R behavior.



        """



        if not bool(self.config.codex_teacher_enabled):



            return False



        if not bool(self.config.codex_teacher_hypothesis_required):



            return False



        try:



            return int(self.config.exit_counterfactual_horizon_bars) > 0



        except (TypeError, ValueError):



            return False







    @staticmethod



    def _codex_lifecycle_identity(lifecycle: object) -> str:



        metadata = dict(getattr(lifecycle, "metadata", {}) or {})



        owner = str(



            metadata.get("position_owner_trace_id", "")



            or getattr(lifecycle, "trace_id", "")



            or ""



        )



        return "|".join((



            str(getattr(lifecycle, "position_ticket", "") or ""),



            owner,



            str(getattr(lifecycle, "close_reason", "") or ""),



            f"{float(getattr(lifecycle, 'realized_r', 0.0) or 0.0):.8f}",



        ))







    @staticmethod



    def _codex_lifecycle_has_counterfactual(lifecycle: object) -> bool:



        hold_r = getattr(lifecycle, "post_exit_hold_r", None)



        horizon = getattr(lifecycle, "post_exit_horizon_bars", None)



        try:



            hold_value = float(hold_r)



            horizon_value = int(float(horizon))



        except (TypeError, ValueError):



            return False



        return bool(np.isfinite(hold_value) and horizon_value > 0)







    def _defer_codex_teacher_case(



        self,



        identity: str,



        lifecycle: object,



        rows: Sequence[object],



        consume_packet: object | None,



        *,



        target_trace_id: str,



    ) -> None:



        """Hold one immutable teacher case until its delayed label is ready."""



        case_id = str(identity or "")



        if not case_id or case_id in self._codex_teacher_submitted_ids:



            return



        if case_id not in self._codex_teacher_deferred_cases:



            self._codex_teacher_deferred_cases[case_id] = {



                "lifecycle": lifecycle,



                "rows": tuple(rows),



                "consume_packet": consume_packet,



                "target_trace_id": str(target_trace_id or ""),



            }



            phase = self._last_codex_teacher_phase



            phase["deferred"] = int(phase.get("deferred", 0) or 0) + 1



            self._write_codex_teacher_status()







    def _flush_deferred_codex_teacher_cases(



        self,



        lifecycles: Iterable[object],



        *,



        force_partial: bool = False,



    ) -> int:



        """Release delayed teacher cases after the hold label is available.







        ``force_partial`` is used only at the terminal report boundary.  A



        tester can end before the configured horizon; in that case the case is



        still graded from immutable realized lifecycle/account evidence, but



        telemetry explicitly records that it was forced without the



        counterfactual instead of pretending the label existed.



        """



        if not self._codex_teacher_deferred_cases:



            return 0



        current_by_identity = {



            self._codex_lifecycle_identity(item): item



            for item in tuple(lifecycles)



            if bool(getattr(item, "final_close_confirmed", False))



        }



        released = 0



        for case_id, item in list(self._codex_teacher_deferred_cases.items()):



            # The deferred key is case-scoped (``<lifecycle>|entry|trace`` or



            # ``<lifecycle>|action|trace``), while the refreshed ledger is



            # indexed by the bare lifecycle identity. Looking up with the



            # composite key silently fell back to the original pre-horizon



            # object, so terminal reconciliation forced a partial receipt even



            # when a POST_EXIT_OBSERVATION row was already present.



            original = item.get("lifecycle")



            if original is not None:



                base_identity = self._codex_lifecycle_identity(original)



            else:



                base_identity = str(case_id)



                for marker in ("|entry|", "|action|", "|legacy|"):



                    if marker in base_identity:



                        base_identity = base_identity.split(marker, 1)[0]



                        break



            current = current_by_identity.get(base_identity, original)



            if current is None:



                continue



            has_counterfactual = self._codex_lifecycle_has_counterfactual(current)



            if not has_counterfactual and not force_partial:



                continue



            original_account = dict(



                getattr(original, "account_context_after_close", {}) or {}



            ) if original is not None else {}



            current_account = dict(



                getattr(current, "account_context_after_close", {}) or {}



            )



            if original_account and not current_account:



                try:



                    current = replace(



                        current,



                        account_context_after_close=original_account,



                    )



                except (TypeError, ValueError):



                    try:



                        setattr(current, "account_context_after_close", original_account)



                    except (AttributeError, TypeError, ValueError):



                        pass



            rows = tuple(item.get("rows", ()) or ())



            queued = self._submit_codex_teacher_case(



                case_id,



                current,



                rows,



                target_trace_id=str(item.get("target_trace_id", "") or ""),



            )



            if not queued:



                continue



            consume_packet = item.get("consume_packet")



            if callable(consume_packet):



                try:



                    consume_packet(str(item.get("target_trace_id", "") or ""))



                except (TypeError, ValueError):



                    pass



            self._codex_teacher_deferred_cases.pop(case_id, None)



            phase = self._last_codex_teacher_phase



            phase["deferred_released"] = int(



                phase.get("deferred_released", 0) or 0



            ) + 1



            if force_partial and not has_counterfactual:



                phase["deferred_forced_partial"] = int(



                    phase.get("deferred_forced_partial", 0) or 0



                ) + 1



            released += 1



        if released:



            self._write_codex_teacher_status()



        return released







    def _drain_codex_teacher_results(



        self,



        policy: object,



        *,



        wait: bool = False,



        report_path: Path | None = None,



    ) -> dict[str, object]:



        """Apply finished teacher receipts on the bridge thread between actions."""



        phase = self._last_codex_teacher_phase



        if not bool(phase.get("enabled", False)):



            return phase



        from .codex_teacher import CodexTeacherRewardModel







        # The facade exposes one mode-neutral route.  Keep the old Five hook



        # as a compatibility fallback for lightweight test doubles and older



        # candidates, but do not let its public-Five-only guard disable legacy



        # Direct-RNN teacher updates.



        optimizer = getattr(policy, "optimize_codex_teacher_trace_packet", None)



        if not callable(optimizer):



            optimizer = getattr(policy, "optimize_five_rlhf_trace_packet", None)



        reference_reader = getattr(policy, "get_codex_teacher_reference_policy_sha256", None)



        if not callable(reference_reader):



            reference_reader = getattr(policy, "get_five_rlhf_reference_policy_sha256", None)



        reference_hash = str(



            reference_reader() if callable(reference_reader)



            else getattr(policy, "five_rlhf_reference_policy_sha256", "") or ""



        )



        if not callable(optimizer) or not reference_hash:



            phase.setdefault("reason", "frozen_reference_or_optimizer_unavailable")



            self._write_codex_teacher_status(report_path)



            return phase







        receipt_path = self._codex_teacher_receipt_file(report_path)



        receipt_path.parent.mkdir(parents=True, exist_ok=True)







        def _receipt_int(mapping: Mapping[str, object], key: str, default: int = -1) -> int:



            """Parse receipt telemetry without treating the valid value 0 as missing."""



            value = mapping.get(key, default)



            if value is None:



                return int(default)



            try:



                return int(value)



            except (TypeError, ValueError):



                return int(default)







        for identity, (packet, traces, future) in list(self._codex_teacher_pending.items()):



            if not wait and not future.done():



                continue



            try:



                receipt = future.result()



                if str(packet["reference_policy_sha256"]) != reference_hash:



                    raise ValueError("case reference differs from installed frozen reference")



                receipt_record = dict(receipt)



                receipt_record["record_type"] = "codex_teacher_receipt"



                receipt_record["executable_path"] = str(



                    self.config.codex_teacher_executable or ""



                )



                receipt_record["cli_version"] = str(



                    self.config.codex_teacher_cli_version or ""



                )



                execution_observability = receipt_record.get("codex_execution")



                if isinstance(execution_observability, Mapping):



                    phase["last_codex_execution"] = dict(execution_observability)



                # Keep the teacher schema/hash untouched while enriching the



                # append-only observatory line with the action identity that



                # the UI and forensic readers need.



                for field in (



                    "action", "action_name", "action_family", "action_surface_count",



                    "target_is_entry_owner",



                ):



                    if field in packet:



                        receipt_record[field] = packet[field]



                if isinstance(packet.get("account_impact"), Mapping):



                    # Packet evidence is immutable and is copied to the



                    # append-only observatory line so the UI can show exactly



                    # what was known before a receipt/update was accepted.



                    receipt_record["account_impact_packet"] = dict(



                        packet["account_impact"]



                    )



                with receipt_path.open("a", encoding="utf-8") as stream:



                    stream.write(



                        json.dumps(receipt_record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"



                    )



                phase["graded"] = int(phase.get("graded", 0)) + 1



                packet_account_impact = packet.get("account_impact", {})



                account_impact_status = str(



                    packet_account_impact.get("status", "inconclusive")



                    if isinstance(packet_account_impact, Mapping) else "inconclusive"



                ).strip().lower()



                if account_impact_status in {"confirmed", "partial"}:



                    phase["account_impact_records"] = int(



                        phase.get("account_impact_records", 0)



                    ) + 1



                if account_impact_status == "confirmed":



                    phase["account_impact_confirmed"] = int(



                        phase.get("account_impact_confirmed", 0)



                    ) + 1



                    evidence_scope = str(



                        packet_account_impact.get("evidence_scope", "")



                        if isinstance(packet_account_impact, Mapping) else ""



                    )



                    if evidence_scope == "immutable_lifecycle_plus_unique_deal_cashflow":



                        phase["account_cashflow_confirmed"] = int(



                            phase.get("account_cashflow_confirmed", 0)



                        ) + 1



                    elif evidence_scope == "immutable_lifecycle_plus_observed_runtime_account_mark":



                        phase["account_mark_confirmed"] = int(



                            phase.get("account_mark_confirmed", 0)



                        ) + 1



                elif account_impact_status == "partial":



                    phase["account_impact_partial"] = int(



                        phase.get("account_impact_partial", 0)



                    ) + 1



                counterfactual_status = str(



                    packet_account_impact.get("counterfactual_status", "not_available")



                    if isinstance(packet_account_impact, Mapping) else "not_available"



                ).strip().lower()



                if counterfactual_status == "validated":



                    phase["counterfactual_validated"] = int(



                        phase.get("counterfactual_validated", 0) or 0



                    ) + 1



                else:



                    phase["counterfactual_unavailable"] = int(



                        phase.get("counterfactual_unavailable", 0) or 0



                    ) + 1



                if bool(self.config.codex_teacher_hypothesis_required):



                    phase["hypotheses"] = int(phase.get("hypotheses", 0)) + 1



                    verification_status = str(



                        receipt.get("verification_status", "") or ""



                    ).strip().lower()



                    if verification_status != "confirmed":



                        phase["hypothesis_rejections"] = int(



                            phase.get("hypothesis_rejections", 0)



                        ) + 1



                        phase["skips"] = int(phase.get("skips", 0)) + 1



                        phase["last_skip_reason"] = (



                            "hypothesis_not_confirmed:" + verification_status



                        )



                        print(



                            f"[MT5] codex-hypothesis trace={packet['trace_id']} "



                            f"status={verification_status or 'missing'} intervention=skipped",



                            flush=True,



                        )



                        continue



                    phase["hypotheses_confirmed"] = int(



                        phase.get("hypotheses_confirmed", 0)



                    ) + 1



                if float(receipt["confidence"]) < float(self.config.codex_teacher_min_confidence):



                    phase["skips"] = int(phase.get("skips", 0)) + 1



                    phase["last_skip_reason"] = "confidence_below_threshold"



                    continue



                reward_model = CodexTeacherRewardModel(



                    packet,



                    receipt,



                    scale=float(self.config.codex_teacher_reward_scale),



                    max_probability_shift=float(self.config.codex_teacher_max_probability_shift),



                )



                update = optimizer(



                    traces,



                    reward_model,



                    source_checkpoint_sha256=str(self.config.rlhf_source_checkpoint_sha256),



                    reference_policy_sha256=reference_hash,



                    config=self.config,



                )



                if not isinstance(update, Mapping):



                    raise ValueError("teacher optimizer did not return an update receipt")



                # A Codex learner result is only an update when the bounded



                # action-head write is observable.  Older drains trusted the



                # generic ``rlhf_ppo_update`` flag, which allowed a no-op (or



                # a wrong-direction write) to be counted as an intervention.



                # The policy-side guard is authoritative for rollback; this



                # bridge-side check keeps status/UI counters fail-closed if a



                # stale policy, malformed plugin, or old checkpoint returns a



                # contradictory receipt.  Test doubles that do not identify



                # themselves as an MT5 trace packet retain their lightweight



                # compatibility contract.



                update = dict(update)



                if (



                    bool(update.get("rlhf_mt5_trace_packet", False))



                    and not bool(update.get("rlhf_update_rejected", False))



                ):



                    contract_reason = ""



                    parameter_hash_before = str(



                        update.get("rlhf_parameter_hash_before", "") or ""



                    )



                    parameter_hash_after = str(



                        update.get("rlhf_parameter_hash_after", "") or ""



                    )



                    parameter_changed = bool(



                        update.get("rlhf_parameter_changed", False)



                    )



                    directional_update = bool(



                        update.get("rlhf_directional_update", False)



                    )



                    try:



                        probability_before = float(



                            update.get("rlhf_target_action_probability_before", 0.0)



                        )



                        probability_after = float(



                            update.get("rlhf_target_action_probability_after", 0.0)



                        )



                        probability_shift = float(



                            update.get("rlhf_packet_max_probability_shift", 0.0)



                        )



                        reward_mean = float(update.get("rlhf_reward_mean", 0.0))



                    except (TypeError, ValueError):



                        probability_before = float("nan")



                        probability_after = float("nan")



                        probability_shift = float("nan")



                        reward_mean = float("nan")



                    expected_direction = str(



                        update.get("rlhf_direction_expected", "") or ""



                    ).strip().lower()



                    if (



                        not parameter_hash_before



                        or not parameter_hash_after



                        or parameter_hash_before == parameter_hash_after



                    ):



                        contract_reason = "update_contract_parameter_hash_unchanged"



                    elif not parameter_changed:



                        contract_reason = "update_contract_parameter_not_changed"



                    elif not directional_update:



                        contract_reason = "update_contract_directional_update_false"



                    elif not all(



                        np.isfinite(value)



                        for value in (



                            probability_before,



                            probability_after,



                            probability_shift,



                            reward_mean,



                        )



                    ):



                        contract_reason = "update_contract_non_finite_telemetry"



                    elif abs(probability_after - probability_before) <= 1e-8:



                        contract_reason = "update_contract_probability_no_change"



                    elif probability_shift <= 1e-8:



                        contract_reason = "update_contract_probability_shift_missing"



                    elif reward_mean < -1e-8 and (



                        expected_direction != "decrease"



                        or not probability_after < probability_before



                    ):



                        contract_reason = "update_contract_negative_reward_wrong_direction"



                    elif reward_mean > 1e-8 and (



                        expected_direction != "increase"



                        or not probability_after > probability_before



                    ):



                        contract_reason = "update_contract_positive_reward_wrong_direction"



                    elif abs(reward_mean) <= 1e-8 or expected_direction == "none":



                        contract_reason = "update_contract_zero_reward_direction"



                    if contract_reason:



                        update["rlhf_update_rejected"] = True



                        update["rlhf_update_contract_validated"] = False



                        update["rlhf_rejection_reason"] = contract_reason



                    else:



                        update["rlhf_update_contract_validated"] = True



                if bool(update.get("rlhf_update_rejected", False)):



                    phase["skips"] = int(phase.get("skips", 0)) + 1



                    phase["last_rejection"] = {



                        key: value for key, value in dict(update).items()



                        if isinstance(value, (str, int, float, bool, type(None)))



                    }



                    phase["rejected_updates"] = int(



                        phase.get("rejected_updates", 0) or 0



                    ) + 1



                    rejected_records = list(phase.get("rejection_records", []))



                    rejected_records.append({



                        "position_ticket": str(packet["position_ticket"]),



                        "trace_id": str(packet["trace_id"]),



                        "action": (



                            int(packet["action"])



                            if packet.get("action") is not None



                            else -1



                        ),



                        "action_name": str(packet.get("action_name", "") or ""),



                        "receipt_sha256": str(receipt.get("receipt_sha256", "") or ""),



                        "teacher_reward": float(receipt.get("teacher_reward", 0.0) or 0.0),



                        "combined_reward": (



                            float(update["rlhf_combined_reward"])



                            if update.get("rlhf_combined_reward") is not None else None



                        ),



                        "account_impact_reward": (



                            float(reward_model.account_impact_reward)



                            if reward_model.account_impact_reward is not None else None



                        ),



                        "reward_direction_source": str(



                            update.get("rlhf_reward_direction_source", "teacher_reward")



                            or "teacher_reward"



                        ),



                        "reward_composition": str(



                            update.get(



                                "rlhf_reward_composition",



                                "teacher_account_impact_50_50_account_negative_floor_v1",



                            )



                            or "teacher_account_impact_50_50_account_negative_floor_v1"



                        ),



                        "reason": str(update.get("rlhf_rejection_reason", "unknown") or "unknown"),



                        "direction_expected": str(update.get("rlhf_direction_expected", "") or ""),



                        "parameter_changed": bool(update.get("rlhf_parameter_changed", False)),



                        "attempt_parameter_changed": bool(



                            update.get("rlhf_update_attempt_parameter_changed", False)



                        ),



                        "action_probability_before": float(



                            update.get("rlhf_target_action_probability_before", 0.0) or 0.0



                        ),



                        "action_probability_after": float(



                            update.get("rlhf_target_action_probability_after", 0.0) or 0.0



                        ),



                        "target_action_probability_delta": float(



                            update.get("rlhf_target_action_probability_delta", 0.0) or 0.0



                        ),



                        "target_action_argmax_before": int(



                            _receipt_int(update, "rlhf_target_action_argmax_before")



                        ),



                        "target_action_argmax_after": int(



                            _receipt_int(update, "rlhf_target_action_argmax_after")



                        ),



                        "target_action_argmax_changed": bool(



                            update.get("rlhf_target_action_argmax_changed", False)



                        ),



                        "target_action_rank_before": int(



                            _receipt_int(update, "rlhf_target_action_rank_before")



                        ),



                        "target_action_rank_after": int(



                            _receipt_int(update, "rlhf_target_action_rank_after")



                        ),



                        "target_action_logit_before": float(



                            update.get("rlhf_target_action_logit_before", 0.0) or 0.0



                        ),



                        "target_action_logit_after": float(



                            update.get("rlhf_target_action_logit_after", 0.0) or 0.0



                        ),



                        "target_action_logit_delta": float(



                            update.get("rlhf_target_action_logit_delta", 0.0) or 0.0



                        ),



                        "target_action_allowed": bool(



                            update.get("rlhf_target_action_allowed", False)



                        ),



                        "global_intervention_applied": bool(



                            update.get("rlhf_codex_global_intervention_applied", False)



                        ),



                        "global_intervention_step": float(



                            update.get("rlhf_codex_global_intervention_step", 0.0) or 0.0



                        ),



                        "global_intervention_residual_fraction": float(



                            update.get(



                                "rlhf_codex_global_intervention_residual_fraction", 1.0



                            )



                            or 0.0



                        ),



                        "global_intervention_attempts": int(



                            update.get("rlhf_codex_global_intervention_attempts", 0) or 0



                        ),



                        "global_intervention_reason": str(



                            update.get("rlhf_codex_global_intervention_reason", "") or ""



                        ),



                        "backtrack_applied": bool(



                            update.get("rlhf_codex_backtrack_applied", False)



                        ),



                        "backtrack_attempts": int(



                            update.get("rlhf_codex_backtrack_attempts", 0) or 0



                        ),



                        "backtrack_reason": str(



                            update.get("rlhf_codex_backtrack_reason", "") or ""



                        ),



                    })



                    phase["rejection_records"] = rejected_records[-256:]



                    self._append_codex_teacher_update(



                        packet,



                        receipt,



                        update,



                        report_path=report_path,



                        accepted=False,



                        account_impact_reward=reward_model.account_impact_reward,



                        account_impact_reward_source=getattr(



                            reward_model, "account_impact_reward_source", "none"



                        ),



                    )



                    continue



                if not bool(update.get("rlhf_ppo_update", False)):



                    raise ValueError("teacher optimizer did not confirm PPO update")



                if bool(update.get("rlhf_parameter_changed", False)):



                    phase["rlhf_parameter_changed"] = int(



                        phase.get("rlhf_parameter_changed", 0)



                    ) + 1



                if bool(update.get("rlhf_internal_policy_state_parameter_changed", False)):



                    phase["rlhf_internal_policy_state_changed"] = int(



                        phase.get("rlhf_internal_policy_state_changed", 0)



                    ) + 1



                probability_shift = float(



                    update.get("rlhf_packet_max_probability_shift", 0.0) or 0.0



                )



                phase["rlhf_max_probability_shift"] = max(



                    float(phase.get("rlhf_max_probability_shift", 0.0) or 0.0),



                    abs(probability_shift),



                )



                if abs(probability_shift) > 1.0e-8:



                    phase["rlhf_probability_shift_count"] = int(



                        phase.get("rlhf_probability_shift_count", 0)



                    ) + 1



                if reward_model.account_impact_reward is not None:



                    phase["account_impact_reward_sum"] = float(



                        phase.get("account_impact_reward_sum", 0.0) or 0.0



                    ) + float(reward_model.account_impact_reward)



                phase["updates"] = int(phase.get("updates", 0)) + 1



                phase["interventions_applied"] = int(



                    phase.get("interventions_applied", 0)



                ) + 1



                phase["last_receipt_sha256"] = str(receipt["receipt_sha256"])



                update_receipts = list(phase.get("update_receipts", []))



                packet_account_impact = packet.get("account_impact", {})



                update_receipts.append({



                    "position_ticket": str(packet["position_ticket"]),



                    "trace_id": str(packet["trace_id"]),



                    "action": (



                        int(packet["action"])



                        if packet.get("action") is not None



                        else -1



                    ),



                    "action_name": str(packet.get("action_name", "") or ""),



                    "action_family": str(packet.get("action_family", "") or ""),



                    "receipt_sha256": str(receipt["receipt_sha256"]),



                    "teacher_reward": float(receipt["teacher_reward"]),



                    "combined_reward": (



                        float(update["rlhf_combined_reward"])



                        if update.get("rlhf_combined_reward") is not None else None



                    ),



                    "reward_composition": str(



                        update.get(



                            "rlhf_reward_composition",



                            "teacher_account_impact_50_50_account_negative_floor_v1",



                        )



                        or "teacher_account_impact_50_50_account_negative_floor_v1"



                    ),



                    "reward_direction_source": str(



                        update.get("rlhf_reward_direction_source", "teacher_reward")



                        or "teacher_reward"



                    ),



                    "confidence": float(receipt["confidence"]),



                    "scaled_reward": float(update.get("rlhf_reward_mean", 0.0)),



                    "advantage_mean": float(update.get("rlhf_ppo_advantage_mean", 0.0)),



                    "advantage": float(update.get("rlhf_ppo_advantage_mean", 0.0)),



                    "hypothesis_status": str(



                        receipt.get("verification_status", "legacy") or "legacy"



                    ),



                    "verification_status": str(



                        receipt.get("verification_status", "legacy") or "legacy"



                    ),



                    "account_impact_status": str(



                        packet_account_impact.get("status", "inconclusive")



                        if isinstance(packet_account_impact, Mapping) else "inconclusive"



                    ),



                    "account_impact_evidence_scope": str(



                        packet_account_impact.get("evidence_scope", "")



                        if isinstance(packet_account_impact, Mapping) else ""



                    ),



                    "counterfactual_status": counterfactual_status,



                    "counterfactual_horizon_bars": (



                        int(packet_account_impact["counterfactual_horizon_bars"])



                        if isinstance(packet_account_impact, Mapping)



                        and packet_account_impact.get("counterfactual_horizon_bars") is not None



                        else None



                    ),



                    "counterfactual_hold_r": (



                        float(packet_account_impact["counterfactual_hold_r"])



                        if isinstance(packet_account_impact, Mapping)



                        and packet_account_impact.get("counterfactual_hold_r") is not None



                        else None



                    ),



                    "counterfactual_delta_r": (



                        float(packet_account_impact["counterfactual_delta_r"])



                        if isinstance(packet_account_impact, Mapping)



                        and packet_account_impact.get("counterfactual_delta_r") is not None



                        else None



                    ),



                    "account_impact_reward": (



                        float(reward_model.account_impact_reward)



                        if reward_model.account_impact_reward is not None else None



                    ),



                    "account_impact_reward_source": str(



                        getattr(reward_model, "account_impact_reward_source", "none")



                        or "none"



                    ),



                    "account_impact_status_runtime": str(



                        update.get("rlhf_account_impact_status", "inconclusive")



                        or "inconclusive"



                    ),



                    "negative_account_floor_applied": bool(



                        update.get("rlhf_negative_account_floor_applied", False)



                    ),



                    # Flatten immutable post-close evidence for the UI/audit



                    # row.  These values come from the validated packet, not



                    # from Codex prose or the derived reward scalar.



                    "account_impact_observed_realized_r": (



                        float(packet_account_impact["observed_realized_r"])



                        if isinstance(packet_account_impact, Mapping)



                        and packet_account_impact.get("observed_realized_r") is not None



                        else None



                    ),



                    "account_impact_observed_net_pnl": (



                        float(packet_account_impact["observed_net_pnl"])



                        if isinstance(packet_account_impact, Mapping)



                        and packet_account_impact.get("observed_net_pnl") is not None



                        else None



                    ),



                    "account_impact_balance_delta": (



                        float(packet_account_impact["balance_delta"])



                        if isinstance(packet_account_impact, Mapping)



                        and packet_account_impact.get("balance_delta") is not None



                        else None



                    ),



                    "account_impact_equity_delta": (



                        float(packet_account_impact["equity_delta"])



                        if isinstance(packet_account_impact, Mapping)



                        and packet_account_impact.get("equity_delta") is not None



                        else None



                    ),



                    "intervention_multiplier": float(



                        receipt.get("intervention_multiplier", 1.0) or 1.0



                    ),



                    "rlhf_codex_learning_rate": float(



                        update.get("rlhf_codex_learning_rate", 0.0) or 0.0



                    ),



                    "rlhf_optimizer_learning_rate_before": float(



                        update.get("rlhf_optimizer_learning_rate_before", 0.0) or 0.0



                    ),



                    "rlhf_optimizer_learning_rate_for_update": float(



                        update.get("rlhf_optimizer_learning_rate_for_update", 0.0) or 0.0



                    ),



                    "rlhf_optimizer_learning_rate_restored": bool(



                        update.get("rlhf_optimizer_learning_rate_restored", False)



                    ),



                    "rlhf_codex_backtrack_applied": bool(



                        update.get("rlhf_codex_backtrack_applied", False)



                    ),



                    "rlhf_codex_backtrack_attempts": int(



                        update.get("rlhf_codex_backtrack_attempts", 0) or 0



                    ),



                    "rlhf_codex_backtrack_reason": str(



                        update.get("rlhf_codex_backtrack_reason", "") or ""



                    ),



                    "rlhf_codex_learning_rate_initial": float(



                        update.get(



                            "rlhf_codex_learning_rate_initial",



                            update.get("rlhf_codex_learning_rate", 0.0),



                        )



                        or 0.0



                    ),



                    "rlhf_codex_context_similarity_threshold": float(



                        update.get("rlhf_codex_context_similarity_threshold", 0.0) or 0.0



                    ),



                    "rlhf_codex_context_support_floor": float(



                        update.get("rlhf_codex_context_support_floor", 0.0) or 0.0



                    ),



                    "rlhf_codex_context_support_target": float(



                        update.get("rlhf_codex_context_support_target", 0.0) or 0.0



                    ),



                    "rlhf_codex_context_support_max": float(



                        update.get("rlhf_codex_context_support_max", 0.0) or 0.0



                    ),



                    "rlhf_codex_context_prototype_captured": bool(



                        update.get("rlhf_codex_context_prototype_captured", False)



                    ),



                    "rlhf_codex_context_prototype_count": int(



                        update.get("rlhf_codex_context_prototype_count", 0) or 0



                    ),



                     "rlhf_codex_global_action_bias_enabled": bool(



                         update.get("rlhf_codex_global_action_bias_enabled", False)



                     ),



                     "rlhf_codex_global_bias_gradient_frozen": bool(



                         update.get("rlhf_codex_global_bias_gradient_frozen", False)



                     ),



                    "rlhf_codex_global_action_bias_scale": float(



                        update.get("rlhf_codex_global_action_bias_scale", 0.0) or 0.0



                    ),



                    "rlhf_codex_global_action_bias_max_abs": float(



                        update.get("rlhf_codex_global_action_bias_max_abs", 0.0) or 0.0



                    ),



                    "rlhf_codex_global_intervention_applied": bool(



                        update.get("rlhf_codex_global_intervention_applied", False)



                    ),



                    "rlhf_codex_global_intervention_step": float(



                        update.get("rlhf_codex_global_intervention_step", 0.0) or 0.0



                    ),



                    "rlhf_codex_global_intervention_residual_fraction": float(



                        update.get(



                            "rlhf_codex_global_intervention_residual_fraction", 1.0



                        )



                        or 0.0



                    ),



                    "rlhf_codex_global_intervention_attempts": int(



                        update.get("rlhf_codex_global_intervention_attempts", 0) or 0



                    ),



                    "rlhf_codex_global_intervention_nonfinite_rejections": int(



                        update.get(



                            "rlhf_codex_global_intervention_nonfinite_rejections", 0



                        )



                        or 0



                    ),



                    "rlhf_codex_global_intervention_shift_rejections": int(



                        update.get(



                            "rlhf_codex_global_intervention_shift_rejections", 0



                        )



                        or 0



                    ),



                    "rlhf_codex_global_intervention_direction_rejections": int(



                        update.get(



                            "rlhf_codex_global_intervention_direction_rejections", 0



                        )



                        or 0



                    ),



                    "rlhf_codex_global_intervention_reason": str(



                        update.get("rlhf_codex_global_intervention_reason", "") or ""



                    ),



                    "rlhf_action_head_update_norm": float(



                        update.get("rlhf_action_head_update_norm", 0.0) or 0.0



                    ),



                    "rlhf_action_head_target_delta": float(



                        update.get("rlhf_action_head_target_delta", 0.0) or 0.0



                    ),



                    "rlhf_action_bias_target_value_before": float(



                        update.get("rlhf_action_bias_target_value_before", 0.0) or 0.0



                    ),



                    "rlhf_action_bias_target_value_after": float(



                        update.get("rlhf_action_bias_target_value_after", 0.0) or 0.0



                    ),



                    "rlhf_action_bias_target_update_delta": float(



                        update.get("rlhf_action_bias_target_update_delta", 0.0) or 0.0



                    ),



                    "rlhf_codex_entropy_weight": float(



                        update.get("rlhf_codex_entropy_weight", 0.0) or 0.0



                    ),



                    "rlhf_codex_kl_beta": float(



                        update.get("rlhf_codex_kl_beta", 0.0) or 0.0



                    ),



                    "rlhf_codex_optimizer_state_reset": bool(



                        update.get("rlhf_codex_optimizer_state_reset", False)



                    ),



                    "rlhf_direction_expected": str(



                        update.get("rlhf_direction_expected", "") or ""



                    ),



                    "rlhf_directional_update": bool(



                        update.get("rlhf_directional_update", False)



                    ),



                    "rlhf_parameter_changed": bool(



                        update.get("rlhf_parameter_changed", False)



                    ),



                    "rlhf_internal_policy_state_trained": bool(



                        update.get("rlhf_internal_policy_state_trained", False)



                    ),



                    "rlhf_internal_policy_state_parameter_changed": bool(



                        update.get(



                            "rlhf_internal_policy_state_parameter_changed", False



                        )



                    ),



                    "rlhf_internal_policy_state_trainable_parameter_tensors": int(



                        update.get(



                            "rlhf_internal_policy_state_trainable_parameter_tensors", 0



                        )



                        or 0



                    ),



                    "rlhf_update_attempt_parameter_changed": bool(



                        update.get("rlhf_update_attempt_parameter_changed", False)



                    ),



                    "parameter_hash_before": str(



                        update.get("rlhf_parameter_hash_before", "") or ""



                    ),



                    "parameter_hash_after": str(



                        update.get("rlhf_parameter_hash_after", "") or ""



                    ),



                    "rlhf_action_head_parameter_hash_before": str(



                        update.get("rlhf_parameter_hash_before", "") or ""



                    ),



                    "rlhf_action_head_parameter_hash_after": str(



                        update.get("rlhf_parameter_hash_after", "") or ""



                    ),



                    "action_probability_before": float(



                        update.get("rlhf_target_action_probability_before", 0.0) or 0.0



                    ),



                    "action_probability_after": float(



                        update.get("rlhf_target_action_probability_after", 0.0) or 0.0



                    ),



                    "target_action_probability_delta": float(



                        update.get("rlhf_target_action_probability_delta", 0.0) or 0.0



                    ),



                    "target_action_argmax_before": int(



                        _receipt_int(update, "rlhf_target_action_argmax_before")



                    ),



                    "target_action_argmax_after": int(



                        _receipt_int(update, "rlhf_target_action_argmax_after")



                    ),



                    "target_action_argmax_changed": bool(



                        update.get("rlhf_target_action_argmax_changed", False)



                    ),



                    "target_action_rank_before": int(



                        _receipt_int(update, "rlhf_target_action_rank_before")



                    ),



                    "target_action_rank_after": int(



                        _receipt_int(update, "rlhf_target_action_rank_after")



                    ),



                    "target_action_logit_before": float(



                        update.get("rlhf_target_action_logit_before", 0.0) or 0.0



                    ),



                    "target_action_logit_after": float(



                        update.get("rlhf_target_action_logit_after", 0.0) or 0.0



                    ),



                    "target_action_logit_delta": float(



                        update.get("rlhf_target_action_logit_delta", 0.0) or 0.0



                    ),



                    "target_action_allowed": bool(



                        update.get("rlhf_target_action_allowed", False)



                    ),



                    "rlhf_target_action_probability_before": float(



                        update.get("rlhf_target_action_probability_before", 0.0) or 0.0



                    ),



                    "rlhf_target_action_probability_after": float(



                        update.get("rlhf_target_action_probability_after", 0.0) or 0.0



                    ),



                    "packet_max_probability_shift": float(



                        update.get("rlhf_packet_max_probability_shift", 0.0) or 0.0



                    ),



                    "parameter_scope": str(



                        update.get("rlhf_parameter_scope", "") or ""



                    ),



                     "trainable_parameter_tensors": int(



                         update.get("rlhf_trainable_parameter_tensors", 0) or 0



                     ),



                    "rlhf_codex_ppo_gradient_parameter_tensors": int(



                        update.get("rlhf_codex_ppo_gradient_parameter_tensors", 0) or 0



                    ),



                    "rlhf_auxiliary_loss": update.get("rlhf_auxiliary_loss"),



                    "rlhf_factorized_geometry_update": bool(



                        update.get("rlhf_factorized_geometry_update", False)



                    ),



                    "rlhf_factorized_geometry_heads": list(



                        update.get("rlhf_factorized_geometry_heads", []) or []



                    ),



                    "rlhf_factorized_geometry_advantage": update.get(



                        "rlhf_factorized_geometry_advantage"



                    ),



                    "rlhf_factorized_geometry_logprob_before": update.get(



                        "rlhf_factorized_geometry_logprob_before"



                    ),



                    "rlhf_factorized_geometry_logprob_after": update.get(



                        "rlhf_factorized_geometry_logprob_after"



                    ),



                    "rlhf_factorized_geometry_ratio": update.get(



                        "rlhf_factorized_geometry_ratio"



                    ),



                    "rlhf_factorized_geometry_loss_scale": update.get(



                        "rlhf_factorized_geometry_loss_scale"



                    ),



                    "rlhf_factorized_geometry_clip_active": bool(



                        update.get("rlhf_factorized_geometry_clip_active", False)



                    ),



                    "rlhf_factorized_geometry_unclipped_correction": update.get(



                        "rlhf_factorized_geometry_unclipped_correction"



                    ),



                    "rlhf_factorized_geometry_unclipped_correction_scale": update.get(



                        "rlhf_factorized_geometry_unclipped_correction_scale"



                    ),



                    "rlhf_factorized_geometry_parameter_hash_before": str(



                        update.get("rlhf_factorized_geometry_parameter_hash_before", "") or ""



                    ),



                    "rlhf_factorized_geometry_parameter_hash_after": str(



                        update.get("rlhf_factorized_geometry_parameter_hash_after", "") or ""



                    ),



                    "rlhf_factorized_geometry_parameter_changed": bool(



                        update.get("rlhf_factorized_geometry_parameter_changed", False)



                    ),



                    "rlhf_factorized_geometry_update_norm": float(



                        update.get("rlhf_factorized_geometry_update_norm", 0.0) or 0.0



                    ),



                    "rlhf_packet_max_probability_shift": float(



                        update.get("rlhf_packet_max_probability_shift", 0.0) or 0.0



                    ),



                    "rlhf_parameter_scope": str(



                        update.get("rlhf_parameter_scope", "") or ""



                    ),



                    "rlhf_trainable_parameter_tensors": int(



                        update.get("rlhf_trainable_parameter_tensors", 0) or 0



                    ),



                    "rlhf_ppo_update": True,



                    "ppo_update": True,



                })



                phase["update_receipts"] = update_receipts[-256:]



                phase["last_update"] = {



                    key: value for key, value in dict(update).items()



                    if isinstance(value, (str, int, float, bool, type(None)))



                }



                self._append_codex_teacher_update(



                    packet,



                    receipt,



                    update,



                    report_path=report_path,



                    accepted=True,



                    account_impact_reward=reward_model.account_impact_reward,



                    account_impact_reward_source=getattr(



                        reward_model, "account_impact_reward_source", "none"



                    ),



                )



                recovery_hook = getattr(



                    self.config, "codex_teacher_recovery_snapshot_hook", None



                )



                if callable(recovery_hook):



                    try:



                        recovery_status = recovery_hook(



                            policy, int(phase.get("updates", 0) or 0)



                        )



                        if isinstance(recovery_status, Mapping):



                            if bool(recovery_status.get("saved", False)):



                                phase["recovery_snapshot_updates"] = int(



                                    phase.get("recovery_snapshot_updates", 0) or 0



                                ) + 1



                                phase["recovery_snapshot_path"] = str(



                                    recovery_status.get("path", "") or ""



                                )



                                phase["recovery_snapshot_optimizer_state_stripped"] = bool(



                                    recovery_status.get("optimizer_state_stripped", False)



                                )



                                phase["recovery_snapshot_optimizer_state_stripped_count"] = int(



                                    recovery_status.get(



                                        "optimizer_state_stripped_count", 0



                                    ) or 0



                                )



                                phase["recovery_snapshot_optimizer_resume_mode"] = str(



                                    "research_reinitialized"



                                    if phase["recovery_snapshot_optimizer_state_stripped"]



                                    else "continuation"



                                )



                            if recovery_status.get("error"):



                                phase["recovery_snapshot_error"] = str(



                                    recovery_status.get("error", "")



                                )[:500]



                    except Exception as snapshot_error:



                        # A forensic snapshot must never turn a verified



                        # learner update into a failed Codex case.



                        phase["recovery_snapshot_error"] = (



                            f"{type(snapshot_error).__name__}: {snapshot_error}"



                        )[:500]



                # Do not infer action causality from a changed parameter hash



                # alone.  The bridge must first pass the online barrier and



                # then observe the next emitted action.  The queued record is



                # deliberately scalar/JSON-safe so it can be copied into the



                # append-only action metadata without exposing hidden model



                # reasoning.



                self._codex_teacher_pending_next_action_links.append({



                    "update_index": int(phase.get("updates", 0) or 0),



                    "position_ticket": str(packet["position_ticket"]),



                    "trace_id": str(packet["trace_id"]),



                    "action": (



                        int(packet["action"])



                        if packet.get("action") is not None



                        else -1



                    ),



                    "action_name": str(packet.get("action_name", "") or ""),



                    "receipt_sha256": str(receipt.get("receipt_sha256", "") or ""),



                    "teacher_reward": float(receipt.get("teacher_reward", 0.0) or 0.0),



                    "scaled_reward": float(update.get("rlhf_reward_mean", 0.0) or 0.0),



                    "parameter_hash_after": str(



                        update.get("rlhf_parameter_hash_after", "") or ""



                    ),



                    "target_action_probability_after": float(



                        update.get("rlhf_target_action_probability_after", 0.0) or 0.0



                    ),



                    "barrier_passed": False,



                    "barrier_reason": "",



                    "barrier_waited_sec": 0.0,



                })



                print(



                    f"[MT5] codex-teacher ticket={packet['position_ticket']} "



                    f"reward={float(receipt['teacher_reward']):+.3f} "



                    f"confidence={float(receipt['confidence']):.2f} "



                    f"hypothesis={str(receipt.get('verification_status', 'legacy'))} "



                    f"multiplier={float(receipt.get('intervention_multiplier', 1.0) or 1.0):.3f} "



                    f"receipt={str(receipt['receipt_sha256'])[:12]}",



                    flush=True,



                )



            except Exception as exc:



                phase["failures"] = int(phase.get("failures", 0)) + 1



                phase["grade_failures"] = int(phase.get("grade_failures", 0)) + 1



                phase["last_error"] = f"{type(exc).__name__}: {exc}"[:500]



                execution_observability = getattr(exc, "execution", None)



                if isinstance(execution_observability, Mapping):



                    phase["last_codex_execution"] = dict(execution_observability)



                self._append_codex_teacher_failure(



                    packet,



                    exc,



                    report_path=report_path,



                    stage="learner_handoff",



                )



                print(f"[MT5] codex-teacher failed: {phase['last_error']}", flush=True)



            finally:



                self._codex_teacher_pending.pop(identity, None)



        self._write_codex_teacher_status(report_path)



        return phase







    def _mark_codex_teacher_barrier_links(



        self,



        *,



        passed: bool,



        reason: str,



        waited_sec: float,



    ) -> None:



        """Mark accepted updates that are now causally before the next action.







        This is intentionally a separate step from accepting an optimizer



        receipt.  A receipt can be valid while the online barrier times out;



        in that case the run still records the update, but must not claim that



        MT5 consumed it before its next decision.



        """



        if not passed:



            return



        for link in self._codex_teacher_pending_next_action_links:



            if bool(link.get("barrier_passed", False)):



                continue



            link["barrier_passed"] = True



            link["barrier_reason"] = str(reason or "finalized_lifecycle")



            link["barrier_waited_sec"] = float(waited_sec or 0.0)







    def _consume_codex_teacher_next_action_links(



        self,



        *,



        policy: object | None = None,



        request: BridgeRequest,



        trace_id: str,



        action: int,



        score: float,



        packet_actions: Iterable[Mapping[str, object]] | None = None,



    ) -> list[dict[str, object]]:



        """Bind barrier-passed updates to the next emitted bridge action."""



        ready = [



            dict(item)



            for item in self._codex_teacher_pending_next_action_links



            if bool(item.get("barrier_passed", False))



        ]



        if not ready:



            return []



        action_rows = [dict(item) for item in (packet_actions or ())]



        observed = {



            "sequence": int(getattr(request, "sequence", 0) or 0),



            "event_id": str(getattr(request, "event_id", "") or ""),



            "trace_id": str(trace_id or ""),



            "action": int(action),



            "score": float(score),



        }



        if action_rows:



            observed["packet_actions"] = action_rows



        live_hash_reader = getattr(



            policy, "get_codex_teacher_live_parameter_hash", None



        )



        live_parameter_hash = ""



        if callable(live_hash_reader):



            try:



                live_parameter_hash = str(live_hash_reader() or "")



            except Exception:



                live_parameter_hash = ""



        observed["policy_parameter_hash_at_next_action"] = live_parameter_hash



        for item in ready:



            update_hash = str(item.get("parameter_hash_after", "") or "")



            item.update({



                "next_request_sequence": observed["sequence"],



                "next_request_event_id": observed["event_id"],



                "next_trace_id": observed["trace_id"],



                "next_action": observed["action"],



                "next_action_score": observed["score"],



                "policy_parameter_hash_at_next_action": live_parameter_hash,



                "policy_parameter_hash_matches_update": bool(



                    bool(update_hash)



                    and bool(live_parameter_hash)



                    and update_hash == live_parameter_hash



                ),



            })



            if action_rows:



                item["next_packet_actions"] = action_rows



        self._codex_teacher_pending_next_action_links = [



            item



            for item in self._codex_teacher_pending_next_action_links



            if not bool(item.get("barrier_passed", False))



        ]



        phase = self._last_codex_teacher_phase



        phase["codex_causal_links_observed"] = int(



            phase.get("codex_causal_links_observed", 0) or 0



        ) + 1



        phase["codex_causal_updates_linked"] = int(



            phase.get("codex_causal_updates_linked", 0) or 0



        ) + len(ready)



        hash_matches = sum(



            bool(item.get("policy_parameter_hash_matches_update", False))



            for item in ready



        )



        hash_unknown = sum(



            not bool(item.get("policy_parameter_hash_at_next_action", ""))



            for item in ready



        )



        phase["codex_causal_hash_matches"] = int(



            phase.get("codex_causal_hash_matches", 0) or 0



        ) + hash_matches



        phase["codex_causal_hash_unknown"] = int(



            phase.get("codex_causal_hash_unknown", 0) or 0



        ) + hash_unknown



        phase["last_codex_causal_link"] = {



            "causal_link": "barrier_to_next_action",



            **observed,



            "updates": ready,



        }



        # Publish the observation immediately; otherwise the UI would only



        # see the link on a later close/drain even though this action already



        # consumed the updated policy state.



        self._write_codex_teacher_status()



        return ready







    @staticmethod



    def _codex_teacher_action_link_details(



        links: Iterable[Mapping[str, object]],



    ) -> dict[str, object]:



        """Return compact action metadata for the UI/observatory."""



        records = [dict(item) for item in links]



        return {



            "codex_causal_link": "barrier_to_next_action",



            "codex_causal_link_count": len(records),



            "codex_causal_updates": records,



        }







    def _wait_for_codex_teacher_action_barrier(



        self,



        policy: object,



        *,



        report_path: Path | None = None,



        reason: str = "finalized_lifecycle",



    ) -> dict[str, object]:



        """Wait for one verified Codex update before the next action is sampled.







        The grader remains in a bounded worker pool, but an entirely



        non-blocking drain lets MT5 produce a full action stream before a



        subprocess receipt can reach the learner.  This barrier is the causal



        boundary: after a finalized close is submitted, the bridge waits for



        the first accepted optimizer update (or a bounded timeout/failure)



        before returning to the policy action loop.  It never waits forever



        and it does not relax receipt, hypothesis, or probability guards.



        """



        phase = self._last_codex_teacher_phase



        wait_sec = max(



            0.0,



            float(getattr(self.config, "codex_teacher_online_wait_sec", 120.0) or 0.0),



        )



        pending_before = int(len(self._codex_teacher_pending))



        updates_before = int(phase.get("updates", 0) or 0)



        barrier = {



            "enabled": bool(phase.get("enabled", False)) and wait_sec > 0.0,



            "reason": str(reason or "finalized_lifecycle"),



            "wait_sec": float(wait_sec),



            "pending_before": pending_before,



            "updates_before": updates_before,



            "waited_sec": 0.0,



            "polls": 0,



            "passed": False,



            "timed_out": False,



        }



        if not barrier["enabled"] or pending_before <= 0:



            barrier["skip_reason"] = (



                "disabled" if not barrier["enabled"] else "no_pending_case"



            )



            phase["online_barrier_last"] = barrier



            return phase







        started = time.monotonic()



        deadline = started + wait_sec



        while self._codex_teacher_pending:



            self._drain_codex_teacher_results(



                policy,



                wait=False,



                report_path=report_path,



            )



            barrier["polls"] = int(barrier["polls"]) + 1



            if int(phase.get("updates", 0) or 0) > updates_before:



                barrier["passed"] = True



                break



            remaining = deadline - time.monotonic()



            if remaining <= 0.0:



                break



            # A short polling interval keeps the action boundary responsive



            # while avoiding a tight loop that would repeatedly rewrite the



            # Windows status/contract sidecars.



            time.sleep(min(0.10, remaining))







        elapsed = max(0.0, time.monotonic() - started)



        barrier["waited_sec"] = float(elapsed)



        barrier["pending_after"] = int(len(self._codex_teacher_pending))



        barrier["updates_after"] = int(phase.get("updates", 0) or 0)



        if (



            not bool(barrier["passed"])



            and barrier["pending_after"] > 0



            and elapsed + 1.0e-6 >= wait_sec



        ):



            barrier["timed_out"] = True



        phase["online_barrier_waits"] = int(



            phase.get("online_barrier_waits", 0) or 0



        ) + 1



        phase["online_barrier_passed"] = int(



            phase.get("online_barrier_passed", 0) or 0



        ) + int(bool(barrier["passed"]))



        phase["online_barrier_timeouts"] = int(



            phase.get("online_barrier_timeouts", 0) or 0



        ) + int(bool(barrier["timed_out"]))



        phase["online_barrier_waited_sec"] = float(



            phase.get("online_barrier_waited_sec", 0.0) or 0.0



        ) + elapsed



        phase["online_barrier_last"] = barrier



        self._mark_codex_teacher_barrier_links(



            passed=bool(barrier["passed"]),



            reason=str(reason or "finalized_lifecycle"),



            waited_sec=float(elapsed),



        )



        self._write_codex_teacher_status(report_path)



        return phase







    def _abort_codex_teacher_phase(



        self,



        reason: str,



        *,



        report_path: Path | None = None,



    ) -> dict[str, object]:



        """Cancel queued teacher work when the MT5 run has already failed.







        The normal completion path drains every receipt so valid feedback can



        reach the learner.  An MT5/report exception has no valid continuation



        boundary, however; waiting for hundreds of Codex subprocesses can



        leave the Python runner alive for hours after an incomplete checkpoint



        has already been persisted.  Record one bounded failure for every



        abandoned packet, cancel queued futures, and let at most the already



        running worker calls finish in the background.



        """



        phase = self._last_codex_teacher_phase



        if not bool(phase.get("enabled", False)):



            return phase



        error = RuntimeError(f"Codex teacher phase aborted: {str(reason)[:500]}")



        pending_items = list(self._codex_teacher_pending.items())



        for identity, (packet, _traces, future) in pending_items:



            cancelled = False



            try:



                cancelled = bool(future.cancel())



            except Exception:



                cancelled = False



            # Whether queued or already running, this packet will not be



            # allowed to update a checkpoint from a failed MT5 session.



            phase["failures"] = int(phase.get("failures", 0) or 0) + 1



            phase["grade_failures"] = int(phase.get("grade_failures", 0) or 0) + 1



            phase["last_error"] = str(error)[:500]



            self._append_codex_teacher_failure(



                packet,



                error,



                report_path=report_path,



                stage="run_abort_cancelled" if cancelled else "run_abort_inflight",



            )



            self._codex_teacher_pending.pop(identity, None)



        executor = self._codex_teacher_executor



        self._codex_teacher_executor = None



        if executor is not None:



            try:



                executor.shutdown(wait=False, cancel_futures=True)



            except TypeError:



                # Python versions without cancel_futures still need the



                # non-blocking shutdown; queued work is already removed from



                # the evaluator's pending map above.



                executor.shutdown(wait=False)



            except Exception as exc:



                phase["executor_shutdown_error"] = f"{type(exc).__name__}: {exc}"[:500]



        phase["pending"] = 0



        phase["aborted"] = True



        phase["abort_reason"] = str(reason)[:500]



        self._write_codex_teacher_status(report_path)



        return phase







    def _run_codex_teacher_phase(



        self, policy: object, *, reconciled: bool, report_path: Path,



    ) -> dict[str, object]:



        """Finish every queued per-close grade and publish the final phase receipt."""



        phase = self._last_codex_teacher_phase



        if not bool(phase.get("enabled", False)):



            return phase



        phase["report_ledger_reconciled"] = bool(reconciled)



        if not reconciled:



            phase["reason"] = "report_ledger_mismatch"



        elif int(phase.get("submitted", 0)) == 0:



            phase["reason"] = "no_owned_final_close_packet"



        self._drain_codex_teacher_results(policy, wait=True, report_path=report_path)



        if self._codex_teacher_executor is not None:



            self._codex_teacher_executor.shutdown(wait=True, cancel_futures=False)



            self._codex_teacher_executor = None



        self._write_codex_teacher_status(report_path)



        return phase







    def _five_action_distribution_snapshot(self) -> dict[str, object]:



        def summarize(actions: Iterable[int]) -> dict[str, object]:



            counts: dict[str, int] = {}



            for action in actions:



                action_id = int(action)



                if not 0 <= action_id < ACTION_COUNT:



                    continue



                key = str(action_id)



                counts[key] = int(counts.get(key, 0)) + 1



            total = int(sum(counts.values()))



            if total <= 0:



                return {



                    "available": False,



                    "counts": counts,



                    "total": 0,



                    "hold_share": 0.0,



                    "effective_action_count": 0.0,



                    "normalized_action_entropy": 0.0,



                    "collapsed": False,



                }



            probabilities = np.asarray(list(counts.values()), dtype=float) / float(total)



            entropy = float(-np.sum(



                probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))



            ))



            effective = float(np.exp(entropy))



            hold_share = float(counts.get(str(int(Action.HOLD)), 0) / float(total))



            normalized = float(entropy / np.log(max(2, len(counts))))



            collapsed = bool(



                hold_share >= 0.95



                or effective <= 1.25



                or normalized <= 0.05



            )



            return {



                "available": True,



                "counts": counts,



                "total": total,



                "hold_share": hold_share,



                "effective_action_count": effective,



                "normalized_action_entropy": normalized,



                "collapsed": collapsed,



            }







        wire_distribution = summarize(self._five_action_by_trace.values())



        economic_actions: list[int] = []



        for trace_id in self._five_action_by_trace:



            if not bool(self._five_economic_actor_eligible_by_trace.get(trace_id, False)):



                continue



            action = self._five_policy_action_by_trace.get(



                trace_id,



                self._five_action_by_trace[trace_id],



            )



            if int(action) >= 0:



                economic_actions.append(int(action))



        economic_distribution = summarize(economic_actions)



        # A shared five-agent action head can look healthy in an aggregate



        # histogram while one entry owner has become deterministic HOLD.  The



        # manager/governor roles are intentionally HOLD-heavy when their



        # responsibility is inactive, so keep a role-scoped view and let the



        # continuation guard inspect only the two entry owners.



        entry_role_names = {



            "LONG_ENTRY_OWNER",



            "SHORT_ENTRY_OWNER",



        }



        role_actions: dict[str, list[int]] = {}



        for trace_id, wire_action in self._five_action_by_trace.items():



            agent_id = getattr(self, "_five_agent_by_trace", {}).get(trace_id)



            if agent_id is None:



                continue



            try:



                role = str(five_agent_authority(int(agent_id)).role)



            except (TypeError, ValueError, KeyError):



                continue



            if role not in entry_role_names:



                continue



            policy_action = self._five_policy_action_by_trace.get(



                trace_id,



                wire_action,



            )



            try:



                action_id = int(policy_action)



            except (TypeError, ValueError):



                continue



            if action_id >= 0:



                role_actions.setdefault(role, []).append(action_id)



        entry_role_distributions = {



            role: summarize(actions)



            for role, actions in sorted(role_actions.items())



        }



        entry_role_collapse_detected = bool(



            any(



                int(distribution.get("total", 0) or 0) >= 8



                and bool(distribution.get("collapsed", False))



                for distribution in entry_role_distributions.values()



            )



        )



        # A whole-window histogram can look diverse even when the policy has



        # collapsed during the final part of the run.  The latest MT5 smoke



        # had 16 non-HOLD packets followed by hundreds of HOLD rows, so the



        # aggregate effective-action count incorrectly reported "diverse".



        # Preserve insertion order (trace registration is bridge order) and



        # classify a bounded tail independently for continuation safety.



        trace_order = list(self._five_action_by_trace)



        tail_window_size = min(len(trace_order), max(25, 5 * 5))



        tail_trace_ids = trace_order[-tail_window_size:] if tail_window_size else []



        tail_wire_distribution = summarize(



            self._five_action_by_trace[trace_id] for trace_id in tail_trace_ids



        )



        tail_economic_actions: list[int] = []



        for trace_id in tail_trace_ids:



            if not bool(self._five_economic_actor_eligible_by_trace.get(trace_id, False)):



                continue



            action = self._five_policy_action_by_trace.get(



                trace_id,



                self._five_action_by_trace[trace_id],



            )



            if int(action) >= 0:



                tail_economic_actions.append(int(action))



        tail_economic_distribution = summarize(tail_economic_actions)



        tail_selected = (



            tail_economic_distribution



            if bool(tail_economic_distribution.get("available", False))



            else tail_wire_distribution



        )



        tail_collapse_detected = bool(



            int(tail_selected.get("total", 0) or 0) >= 25



            and (



                bool(tail_selected.get("collapsed", False))



                or float(tail_selected.get("hold_share", 0.0) or 0.0) >= 0.95



            )



        )



        return {



            **wire_distribution,



            # Raw wire actions remain available for bridge/audit diagnostics.



            "wire": wire_distribution,



            # Collapse decisions must use only actions for which the runtime



            # marked an economic actor eligible.  Inactive manager/governor



            # HOLDs therefore cannot veto continuation by themselves.



            "economic_actor": economic_distribution,



            "collapse_scope": (



                "economic_actor"



                if bool(economic_distribution.get("available", False))



                else "wire"



            ),



            "wire_collapse_detected": bool(wire_distribution.get("collapsed", False)),



            "economic_actor_collapse_detected": bool(



                economic_distribution.get("collapsed", False)



            ),



            "entry_role_distributions": entry_role_distributions,



            "entry_role_collapse_detected": entry_role_collapse_detected,



            "wire_collapse_warning": bool(



                wire_distribution.get("collapsed", False)



                and bool(economic_distribution.get("available", False))



            ),



            "tail_window_size": int(tail_window_size),



            "tail": {



                **dict(tail_wire_distribution),



                "wire": tail_wire_distribution,



                "economic_actor": tail_economic_distribution,



                "collapse_scope": (



                    "economic_actor"



                    if bool(tail_economic_distribution.get("available", False))



                    else "wire"



                ),



                "collapse_detected": bool(tail_collapse_detected),



            },



            "tail_collapse_detected": bool(tail_collapse_detected),



        }







    def _maybe_capture_five_champion(



        self,



        *,



        policy: object,



        lifecycle: object,



    ) -> None:



        """Keep one pre-collapse continuation snapshot from real lifecycles."""



        if not bool(getattr(self.config, "learning_enabled", True)):



            return



        trace_id = str(getattr(lifecycle, "trace_id", "") or "")



        if not trace_id.startswith("five:"):



            return



        identity = "|".join((



            str(getattr(lifecycle, "position_ticket", "") or ""),



            trace_id,



            str(getattr(lifecycle, "close_reason", "") or ""),



            f"{float(getattr(lifecycle, 'realized_r', 0.0) or 0.0):.8f}",



        ))



        if identity in self._five_champion_seen_lifecycle_ids:



            return



        self._five_champion_seen_lifecycle_ids.add(identity)



        self._five_champion_closed_count += 1



        self._five_champion_net_r += float(getattr(lifecycle, "realized_r", 0.0) or 0.0)



        if self._five_champion_policy is not None:



            return



        minimum = max(1, int(getattr(self.config, "five_champion_min_lifecycles", 16) or 16))



        minimum_net = float(getattr(self.config, "five_champion_min_net_r", 0.0) or 0.0)



        if self._five_champion_closed_count < minimum or self._five_champion_net_r <= minimum_net:



            return



        distribution = self._five_action_distribution_snapshot()



        # Capture the first economically positive continuation point even if



        # the action head is already trending toward HOLD. The snapshot is



        # only a validation candidate; chuyen_gia_policy must re-run it on an



        # isolated frozen MT5 runner before it can replace final weights.



        if not bool(distribution.get("available", False)):



            return



        try:



            self._five_champion_policy = clone_policy(policy)



            self._five_champion_info = {



                "available": True,



                "captured": True,



                "closed_count": int(self._five_champion_closed_count),



                "net_r": float(self._five_champion_net_r),



                "action_distribution": distribution,



            }



        except Exception as exc:



            self._five_champion_info = {



                "available": False,



                "captured": False,



                "reason": f"{type(exc).__name__}: {exc}",



                "closed_count": int(self._five_champion_closed_count),



                "net_r": float(self._five_champion_net_r),



            }







    def five_champion_candidate(self) -> tuple[object | None, dict[str, object]]:



        """Return the isolated pre-collapse policy, if one was captured."""



        return self._five_champion_policy, dict(self._five_champion_info)







    def _five_lifecycle_routing_snapshot(self) -> dict[str, object]:



        attempts = int(self._five_lifecycle_route_attempts)



        accepted = int(self._five_lifecycle_route_accepted)



        return {



            "attempts": attempts,



            "accepted": accepted,



            "misses": int(self._five_lifecycle_route_misses),



            "acceptance_ratio": float(accepted / attempts) if attempts else 0.0,



            "initial_sl_attempts": int(self._five_lifecycle_initial_sl_attempts),



            "initial_sl_accepted": int(self._five_lifecycle_initial_sl_accepted),



            "optimizer_updates": int(self._five_lifecycle_optimizer_updates),



            "initial_sl_optimizer_samples": float(



                self._five_lifecycle_initial_sl_optimizer_samples



            ),



            "last_initial_sl_loss": float(self._five_lifecycle_last_initial_sl_loss),



            "last_rollout_steps": float(self._five_lifecycle_last_rollout_steps),



            "v4_credit_sum_r": float(self._five_lifecycle_v4_credit_sum),



            "net_r_credit_sum_r": float(self._five_lifecycle_net_r_credit_sum),



            "combined_credit_sum_r": float(self._five_lifecycle_combined_credit_sum),



        }







    @staticmethod



    def _five_direct_rnn_routing_snapshot(policy: object) -> dict[str, object]:



        """Aggregate the sole Direct-RNN delayed-credit writer for MT5 audit."""



        candidates: list[object] = []



        registry = getattr(policy, "registry", None)



        experts = getattr(registry, "chuyen_gia_theo_context", {})



        if isinstance(experts, Mapping):



            candidates.extend(



                getattr(expert, "policy", None)



                for expert in experts.values()



            )



        candidates.append(policy)



        seen: set[int] = set()



        snapshots: list[dict[str, object]] = []



        for candidate in candidates:



            if candidate is None or id(candidate) in seen:



                continue



            seen.add(id(candidate))



            hook = getattr(candidate, "direct_rnn_rl_runtime_snapshot", None)



            if not callable(hook):



                continue



            try:



                snapshot = hook()



            except Exception as exc:



                snapshots.append({"available": False, "error": type(exc).__name__})



                continue



            if isinstance(snapshot, Mapping):



                snapshots.append(dict(snapshot))



        if not snapshots:



            return {"available": False, "owners": 0}



        total_keys = (



            "pending_total", "pending_lifecycle", "pending_preference",



            "queued_lifecycle", "queued_preference", "lifecycle_optimizer_batches",



            "preference_optimizer_batches", "optimizer_updated_modules", "optimizer_steps",



        )



        result: dict[str, object] = {



            "available": True,



            "owners": int(len(snapshots)),



        }



        for key in total_keys:



            result[key] = int(sum(int(item.get(key, 0) or 0) for item in snapshots))



        result["owners_detail"] = snapshots



        return result







    def _five_failure_replay_snapshot(self) -> dict[str, object]:



        return {



            "admitted": int(self._five_failure_replay_admitted),



            "audited": int(self._five_failure_replay_audited),



            "selected": int(self._five_failure_replay_selected),



            "newly_resolved": int(self._five_failure_replay_newly_resolved),



            "newly_reopened": int(self._five_failure_replay_newly_reopened),



            "bank_cases": int(self._five_failure_replay_bank_cases),



            "bank_unresolved": int(self._five_failure_replay_bank_unresolved),



            "bank_resolved": int(self._five_failure_replay_bank_resolved),



            "replayed_total": int(self._five_failure_replay_replayed_total),



            "reopened_total": int(self._five_failure_replay_reopened_total),



            "last_loss": float(self._five_failure_replay_last_loss),



            "last_weighted_loss": float(self._five_failure_replay_last_weighted_loss),



        }







    def _register_five_competitor_trace(



        self,



        trace_id: str,



        agent_id: int,



        action: int | None = None,



    ) -> None:



        trace_key = str(trace_id or "")



        agent = int(agent_id)



        if not trace_key or not 0 <= agent < 5:



            raise ValueError("five competitor trace requires trace_id and agent_id 0..4")



        previous = self._five_agent_by_trace.get(trace_key)



        if previous is not None and int(previous) != agent:



            raise RuntimeError("one five trace cannot move between competitors")



        self._five_agent_by_trace[trace_key] = agent



        if action is not None:



            action_id = int(action)



            if not 0 <= action_id < ACTION_COUNT:



                raise ValueError("five competitor action must belong to the bridge action schema")



            previous_action = self._five_action_by_trace.get(trace_key)



            if previous_action is not None and int(previous_action) != action_id:



                raise RuntimeError("one five trace cannot change its requested action")



            self._five_action_by_trace[trace_key] = action_id







    def _record_five_live_owner_outcome(



        self,



        policy: object | None,



        *,



        agent_index: int,



        action: int | None,



        net_r: float,



        trace_id: str,



    ) -> bool:



        """Record immutable close evidence and expose it in Codex status.







        The policy hook owns the actual run-local veto pool and remains the



        source of truth for idempotency. The runner only mirrors accepted



        samples into status so the UI can distinguish close evidence from an



        optimizer update.



        """



        hook = getattr(policy, "record_five_live_owner_outcome", None)



        if not callable(hook) or action is None:



            return False



        trace_key = str(trace_id or "").strip()



        if trace_key and trace_key in self._five_live_owner_outcome_trace_ids:



            return False



        accepted = bool(hook(



            agent_index=int(agent_index),



            action=int(action),



            net_r=float(net_r),



            trace_id=trace_key,



        ))



        if not accepted:



            return False



        if trace_key:



            self._five_live_owner_outcome_trace_ids.add(trace_key)



        phase = self._last_codex_teacher_phase



        phase["live_owner_outcome_records"] = int(



            phase.get("live_owner_outcome_records", 0) or 0



        ) + 1



        samples = dict(phase.get("live_owner_outcome_samples_by_action", {}) or {})



        action_key = str(int(action))



        samples[action_key] = int(samples.get(action_key, 0) or 0) + 1



        phase["live_owner_outcome_samples_by_action"] = samples



        if float(net_r) < 0.0:



            phase["live_owner_outcome_negative_count"] = int(



                phase.get("live_owner_outcome_negative_count", 0) or 0



            ) + 1



        phase["live_owner_outcome_last"] = {



            "agent_index": int(agent_index),



            "action": int(action),



            "net_r": float(net_r),



            "trace_id": trace_key,



        }



        return True







    def _record_five_competition_lifecycles(



        self,



        lifecycles: Iterable[object],



        policy: object | None = None,



        *,



        account_context_after_close: Mapping[str, object] | None = None,



    ) -> int:



        """Attribute immutable final MT5 outcomes to the creating competitor."""







        def terminal_packet(



            trace_id: str,



            *,



            entry: bool,



        ) -> tuple[tuple[object, ...], object | None]:



            """Read one immutable packet and return its successful consume hook.







            The lifecycle cache is populated only when the fill is visible in



            the same bridge acknowledgement as the decision.  MT5 commonly



            emits ``SCHEDULED`` first and ``DEAL_ADD/FILLED`` on a later



            acknowledgement, so a finalized entry can legitimately miss that



            narrow cache even though ``register_five_action_packet`` retained



            the exact decision-time packet.  Prefer the lifecycle cache for



            compatibility, then fall back to the all-action cache.  Returning



            the consume hook separately prevents a failed/limited queue from



            destroying the only provenance copy.



            """



            trace_key = str(trace_id or "").strip()



            if not trace_key or policy is None:



                return (), None



            if entry:



                primary_get = getattr(policy, "get_five_entry_lifecycle_packet", None)



                primary_pop = getattr(policy, "pop_five_entry_lifecycle_packet", None)



            else:



                primary_get = getattr(policy, "get_five_action_packet", None)



                primary_pop = getattr(policy, "pop_five_action_packet", None)



            # A lightweight compatibility policy in older tests/checkpoints may



            # expose only ``pop_*``.  Preserve that behavior when no safer read



            # method exists.



            if callable(primary_get):



                rows = tuple(primary_get(trace_key))



                if len(rows) == FIVE_PUBLIC_AGENT_COUNT:



                    return rows, primary_pop if callable(primary_pop) else None



            elif callable(primary_pop):



                rows = tuple(primary_pop(trace_key))



                if len(rows) == FIVE_PUBLIC_AGENT_COUNT:



                    return rows, None



            if entry:



                fallback_get = getattr(policy, "get_five_action_packet", None)



                fallback_pop = getattr(policy, "pop_five_action_packet", None)



                if callable(fallback_get):



                    rows = tuple(fallback_get(trace_key))



                    if len(rows) == FIVE_PUBLIC_AGENT_COUNT:



                        return rows, fallback_pop if callable(fallback_pop) else None



                elif callable(fallback_pop):



                    rows = tuple(fallback_pop(trace_key))



                    if len(rows) == FIVE_PUBLIC_AGENT_COUNT:



                        return rows, None



            return (), None







        lifecycles = tuple(lifecycles)



        final_lifecycles = [



            lifecycle for lifecycle in lifecycles



            if bool(getattr(lifecycle, "final_close_confirmed", False))



        ]







        def lifecycle_identity(lifecycle: object) -> str:



            metadata = dict(getattr(lifecycle, "metadata", {}) or {})



            owner = str(



                metadata.get("position_owner_trace_id", "")



                or getattr(lifecycle, "trace_id", "")



                or ""



            )



            return "|".join((



                str(getattr(lifecycle, "position_ticket", "") or ""),



                owner,



                str(getattr(lifecycle, "close_reason", "") or ""),



                f"{float(getattr(lifecycle, 'realized_r', 0.0) or 0.0):.8f}",



            ))







        final_identities = {



            lifecycle_identity(lifecycle)



            for lifecycle in final_lifecycles



        }



        unseen_final_ids = final_identities.difference(



            self._five_account_mark_observed_lifecycle_ids



        )



        # Mark every final lifecycle in this ledger snapshot as observed,



        # including unknown/unowned rows.  They remain in the settled set



        # only after exact owner routing below; this set solely prevents a



        # stale unresolved row from competing with a newer close for the one



        # bridge-level post-close account mark.



        self._five_account_mark_observed_lifecycle_ids.update(final_identities)



        # One bridge account mark cannot be causally assigned to several



        # closes that arrived in the same poll.  In that case the packet keeps



        # immutable lifecycle evidence but is explicitly partial instead of



        # pretending the shared delta belongs to every action.



        account_mark_for_packet = (



            account_context_after_close



            if account_context_after_close and len(unseen_final_ids) == 1



            else None



        )



        account_mark_identity = (



            next(iter(unseen_final_ids))



            if account_mark_for_packet is not None



            else None



        )







        # A previous refresh may have seen the final close but not yet the EA's



        # delayed POST_EXIT_OBSERVATION row.  Release those cases before the



        # normal settled-lifecycle guard below; the generic lifecycle remains



        # idempotent while the Codex packet receives the now-complete label.



        self._flush_deferred_codex_teacher_cases(final_lifecycles)







        def queue_teacher_case(



            case_id: str,



            teacher_lifecycle: object,



            rows: Sequence[object],



            consume_packet: object | None,



            *,



            target_trace_id: str,



            wait_for_counterfactual: bool = True,



        ) -> bool:



            should_wait = (



                bool(wait_for_counterfactual)



                and



                self._codex_teacher_waits_for_counterfactual()



                # Legacy/context traces may not expose a consume callback.



                # That callback only retires the cached packet after submit;



                # it must never bypass the causal POST_EXIT_OBSERVATION wait.



                and not self._codex_lifecycle_has_counterfactual(teacher_lifecycle)



            )



            if should_wait:



                self._defer_codex_teacher_case(



                    case_id,



                    teacher_lifecycle,



                    rows,



                    consume_packet,



                    target_trace_id=target_trace_id,



                )



                return False



            return self._submit_codex_teacher_case(



                case_id,



                teacher_lifecycle,



                rows,



                target_trace_id=target_trace_id,



            )







        recorded = 0



        for lifecycle in lifecycles:



            if not bool(getattr(lifecycle, "final_close_confirmed", False)):



                continue



            teacher_lifecycle = lifecycle



            lifecycle_identity_value = lifecycle_identity(lifecycle)



            lifecycle_account_mark = dict(



                getattr(lifecycle, "account_context_after_close", {}) or {}



            )



            if (



                account_mark_for_packet is not None



                and lifecycle_identity_value == account_mark_identity



                and not lifecycle_account_mark



            ):



                # ``MT5PositionLifecycle`` is frozen.  Keep the generic



                # lifecycle untouched and attach the bridge's current



                # post-close account mark only to the Codex packet view.



                try:



                    teacher_lifecycle = replace(



                        lifecycle,



                        account_context_after_close={



                            str(key): float(value)



                            for key, value in dict(account_mark_for_packet).items()



                            if value is not None and np.isfinite(float(value))



                        },



                    )



                except (TypeError, ValueError):



                    # Lightweight replay/test lifecycle objects are mutable



                    # namespaces rather than the frozen production dataclass.



                    # Preserve the same packet contract for them without



                    # changing the generic settlement object.



                    try:



                        setattr(



                            lifecycle,



                            "account_context_after_close",



                            {



                                str(key): float(value)



                                for key, value in dict(account_mark_for_packet).items()



                                if value is not None and np.isfinite(float(value))



                            },



                        )



                    except (AttributeError, TypeError, ValueError):



                        pass



                    teacher_lifecycle = lifecycle



            lifecycle_trace_id = str(getattr(lifecycle, "trace_id", "") or "")



            # A final deal normally carries the trace of the closing command



            # (MANAGE/CLOSE_ALL), while the EA also persists the immutable



            # entry owner in ``position_owner_trace_id``.  Reward ownership



            # must follow the latter; using the close trace would either route



            # a lifecycle to a management head or trip the fail-closed guard.



            metadata = dict(getattr(lifecycle, "metadata", {}) or {})



            owner_trace_id = str(



                metadata.get("position_owner_trace_id", "") or ""



            ).strip()



            trace_id = owner_trace_id or lifecycle_trace_id



            agent = self._five_agent_by_trace.get(trace_id)



            if agent is None:



                # A legacy context policy has no Five competitor registry.



                # Recover its immutable cached owner trace at final-report



                # time so delayed/system closes still receive one Codex case.



                if (



                    bool(self.config.codex_teacher_enabled)



                    and bool(self.config.learning_enabled)



                    and trace_id



                    and not trace_id.startswith("five:")



                ):



                    trace_reader = getattr(policy, "get_codex_teacher_trace", None)



                    legacy_trace = trace_reader(trace_id) if callable(trace_reader) else None



                    if legacy_trace is not None:



                        queue_teacher_case(



                            f"legacy|{getattr(lifecycle, 'position_ticket', '')}|{trace_id}|final",



                            teacher_lifecycle,



                            (legacy_trace,),



                            None,



                            target_trace_id=trace_id,



                        )



                continue



            owner_action = self._five_action_by_trace.get(trace_id)



            if owner_action is not None and owner_action not in {



                int(Action.MARKET_BUY), int(Action.MARKET_SELL),



                int(Action.BUY_LIMIT), int(Action.SELL_LIMIT),



                int(Action.BUY_STOP), int(Action.SELL_STOP),



            }:



                raise RuntimeError(



                    "final MT5 lifecycle owner trace is not an entry action; "



                    "refusing corrupt Five reward attribution"



                )



            identity = lifecycle_identity_value



            if identity in self._five_settled_lifecycle_ids:



                continue



            if bool(self.config.codex_teacher_enabled) and bool(self.config.learning_enabled):



                # The entry owner remains eligible for its own terminal



                # outcome.  A separate close/manage trace may also be eligible



                # when the ledger carries an immutable exit_trace_id.  Each



                # case is bound to one exact target trace; no winner-wide or



                # latest-trace credit is allowed.



                if owner_trace_id and owner_action in (1, 2, 3, 4, 5, 6):



                    rows, consume_packet = terminal_packet(owner_trace_id, entry=True)



                    if len(rows) == 5:



                        queued = queue_teacher_case(



                            f"{identity}|entry|{owner_trace_id}",



                            teacher_lifecycle,



                            rows,



                            consume_packet,



                            target_trace_id=owner_trace_id,



                            # Entry ownership now waits for the immutable



                            # post-exit hold horizon as well.  The final



                            # realized outcome alone says whether the trade



                            # lost; the hold path says whether entering was



                            # worse than staying flat.  Keeping the packet



                            # until that row arrives is required for the



                            # learner to receive the action's causal account



                            # impact instead of a premature partial label.



                            wait_for_counterfactual=True,



                        )



                        is_lifecycle_packet = callable(consume_packet) and (



                            consume_packet == getattr(policy, "pop_five_entry_lifecycle_packet", None)



                        )



                        if (



                            queued



                            and callable(consume_packet)



                            and not (



                                is_lifecycle_packet



                                and bool(self.config.learning_enabled)



                                and hasattr(policy, "apply_five_entry_lifecycle_reward")



                            )



                        ):



                            consume_packet(owner_trace_id)



                exit_trace_id = str(getattr(lifecycle, "exit_trace_id", "") or "").strip()



                exit_action = self._five_action_by_trace.get(exit_trace_id)



                if (



                    exit_trace_id



                    and exit_trace_id != owner_trace_id



                    and exit_action in (7, 8, 9, 10, 11)



                ):



                    action_rows, consume_packet = terminal_packet(exit_trace_id, entry=False)



                    if len(action_rows) == 5:



                        queued = queue_teacher_case(



                            f"{identity}|action|{exit_trace_id}",



                            teacher_lifecycle,



                            action_rows,



                            consume_packet,



                            target_trace_id=exit_trace_id,



                            # Management/exit owners require the delayed



                            # hold counterfactual so their reward is the



                            # incremental decision effect, not total trade



                            # P/L.



                            wait_for_counterfactual=True,



                        )



                        if queued and callable(consume_packet):



                            consume_packet(exit_trace_id)



            initial_risk = abs(float(getattr(lifecycle, "initial_risk", 0.0) or 0.0))



            cost_r = 0.0



            if initial_risk > 1e-12:



                commission = float(getattr(lifecycle, "commission", 0.0) or 0.0)



                swap = float(getattr(lifecycle, "swap", 0.0) or 0.0)



                cost_r = max(0.0, -(commission + min(0.0, swap)) / initial_risk)



            net_r = float(getattr(lifecycle, "realized_r", 0.0) or 0.0)



            # Keep a strictly run-local side outcome sample for adaptive



            # behavior exploration.  It is fed only from the immutable,



            # owner-reconciled final lifecycle and is separate from the



            # persistent contextual ledger used for audit/replay.



            # This is run-local diagnostic/veto evidence, not the legacy



            # lifecycle optimizer.  Codex mode must record the same immutable



            # owner outcome so its live action-veto pool can react to a bad



            # BUY/SELL after the position is closed.  The persistent legacy



            # lifecycle hook below remains explicitly disabled for Codex.



            self._record_five_live_owner_outcome(



                policy,



                agent_index=int(agent),



                action=owner_action,



                net_r=net_r,



                trace_id=trace_id,



            )



            # Store gross R and subtract explicit costs once in the shaper.



            self._five_gross_realized_r[int(agent)] += np.float32(net_r + cost_r)



            self._five_pending_cost_r[int(agent)] += np.float32(cost_r)



            # Reward V2 is based only on the immutable net close outcome.  It



            # deliberately does not infer wins or expectancy from account PnL,



            # open marks, order acceptance, or mutable SL/TP events.



            self._five_closed_trade_count[int(agent)] += np.float32(1.0)



            self._five_winning_trade_count[int(agent)] += np.float32(net_r > 0.0)



            self._five_realized_trade_r[int(agent)] += np.float32(net_r)



            expectancy_shortfall = max(



                0.0, float(self.config.five_reward_expectancy_floor_r) - net_r



            )



            tail_excess = min(



                max(0.0, -net_r - float(self.config.five_reward_tail_loss_threshold_r)),



                float(self.config.five_reward_tail_loss_cap_r),



            )



            self._five_expectancy_shortfall_r[int(agent)] += np.float32(expectancy_shortfall)



            self._five_tail_loss_cost_r[int(agent)] += np.float32(tail_excess ** 2)



            entry_atr = float(getattr(lifecycle, "initial_atr", 0.0) or 0.0)



            entry_price = float(getattr(lifecycle, "entry_price", 0.0) or 0.0)



            initial_sl = float(getattr(lifecycle, "initial_sl_price", 0.0) or 0.0)



            initial_tp = float(getattr(lifecycle, "initial_tp_price", 0.0) or 0.0)



            if entry_atr > 1e-12 and entry_price > 0.0 and initial_sl > 0.0:



                sl_atr = abs(entry_price - initial_sl) / entry_atr



            else:



                sl_atr = float(metadata.get("geometry_sl_atr", metadata.get("fixed_sl_atr", 0.0)) or 0.0)



            if entry_atr > 1e-12 and entry_price > 0.0 and initial_tp > 0.0:



                tp_atr = abs(initial_tp - entry_price) / entry_atr



            else:



                tp_atr = float(metadata.get("geometry_tp_atr", metadata.get("fixed_tp_atr", 0.0)) or 0.0)



            risk_distance_price = abs(entry_price - initial_sl)



            side = int(getattr(lifecycle, "side", 0) or metadata.get("side", 0) or 0)



            final_sl = float(getattr(lifecycle, "final_sl_price", 0.0) or 0.0)



            locked_profit_r = 0.0



            if bool(getattr(lifecycle, "trailing_applied", False)) and risk_distance_price > 1e-12:



                if side > 0 and final_sl > 0.0:



                    locked_profit_r = max(0.0, (final_sl - entry_price) / risk_distance_price)



                elif side < 0 and final_sl > 0.0:



                    locked_profit_r = max(0.0, (entry_price - final_sl) / risk_distance_price)



            recovery_fraction, positive_persistence = lifecycle_path_dynamics(



                getattr(lifecycle, "trajectory", ())



            )



            lifecycle_shape = shape_five_entry_lifecycle(



                action_index=int(owner_action or 0),



                sl_atr=sl_atr,



                tp_atr=tp_atr,



                trailing_atr=float(metadata.get("proposed_trailing_atr", 0.0) or 0.0),



                trailing_applied=bool(getattr(lifecycle, "trailing_applied", False)),



                locked_profit_r=float(locked_profit_r),



                mfe_r=float(getattr(lifecycle, "mfe_r", 0.0) or 0.0),



                mae_r=float(getattr(lifecycle, "mae_r", 0.0) or 0.0),



                mfe_available=bool(getattr(lifecycle, "excursion_available", False)),



                realized_r=net_r,



                exit_kind=str(getattr(lifecycle, "exit_kind", "") or getattr(lifecycle, "close_reason", "")),



                recovery_fraction=recovery_fraction,



                positive_persistence=positive_persistence,



                pending_fill_bars=getattr(lifecycle, "pending_fill_bars", None),



                config=FiveLifecycleRewardConfig.from_source(self.config),



            )



            lifecycle_hook = (



                getattr(policy, "apply_five_entry_lifecycle_reward", None)



                if policy is not None



                else None



            )



            self._five_lifecycle_route_attempts += 1



            if bool(lifecycle_shape.initial_sl_failure):



                self._five_lifecycle_initial_sl_attempts += 1



            routed_exactly = False



            routing: Mapping[str, object] = {}



            if callable(lifecycle_hook):



                raw_routing = lifecycle_hook(



                    trace_id,



                    lifecycle_reward_r=float(lifecycle_shape.total_r),



                    realized_r=net_r,



                    components=lifecycle_shape.as_dict(),



                )



                routing = raw_routing if isinstance(raw_routing, Mapping) else {}



                routed_exactly = bool(



                    routing.get("accepted", False)



                )



            if routed_exactly:



                self._five_lifecycle_route_accepted += 1



                self._five_lifecycle_v4_credit_sum += float(



                    routing.get("reward_v4_r", lifecycle_shape.total_r) or 0.0



                )



                self._five_lifecycle_net_r_credit_sum += float(



                    routing.get("net_r_credit_r", 0.0) or 0.0



                )



                self._five_lifecycle_combined_credit_sum += float(



                    routing.get("reward_r", lifecycle_shape.total_r) or 0.0



                )



                if bool(lifecycle_shape.initial_sl_failure):



                    self._five_lifecycle_initial_sl_accepted += 1



                self._five_lifecycle_last_rollout_steps = float(



                    routing.get("lifecycle_rollout_steps", 0.0) or 0.0



                )



                self._five_lifecycle_last_initial_sl_loss = float(



                    routing.get("initial_sl_local_loss_value", 0.0) or 0.0



                )



                self._five_lifecycle_initial_sl_optimizer_samples += float(



                    routing.get("initial_sl_local_samples", 0.0) or 0.0



                )



                self._five_lifecycle_optimizer_updates += int(



                    bool(routing.get("updated", 0))



                )



                if str(routing.get("failure_replay_case_id", "") or ""):



                    self._five_failure_replay_admitted += 1



                self._five_failure_replay_audited += int(float(



                    routing.get("failure_replay_audited", 0.0) or 0.0



                ))



                self._five_failure_replay_selected += int(float(



                    routing.get("failure_replay_selected", 0.0) or 0.0



                ))



                self._five_failure_replay_newly_resolved += int(float(



                    routing.get("failure_replay_newly_resolved", 0.0) or 0.0



                ))



                self._five_failure_replay_newly_reopened += int(float(



                    routing.get("failure_replay_newly_reopened", 0.0) or 0.0



                ))



                self._five_failure_replay_bank_cases = int(float(



                    routing.get("failure_replay_bank_cases", 0.0) or 0.0



                ))



                self._five_failure_replay_bank_unresolved = int(float(



                    routing.get("failure_replay_bank_unresolved", 0.0) or 0.0



                ))



                self._five_failure_replay_bank_resolved = int(float(



                    routing.get("failure_replay_bank_resolved", 0.0) or 0.0



                ))



                self._five_failure_replay_replayed_total = int(float(



                    routing.get("failure_replay_bank_replayed_total", 0.0) or 0.0



                ))



                self._five_failure_replay_reopened_total = int(float(



                    routing.get("failure_replay_bank_reopened_total", 0.0) or 0.0



                ))



                self._five_failure_replay_last_loss = float(



                    routing.get("failure_replay_loss", 0.0) or 0.0



                )



                self._five_failure_replay_last_weighted_loss = float(



                    routing.get("failure_replay_weighted_loss", 0.0) or 0.0



                )



            # INITIAL_SL can have scalar shaping equal to zero. It still must



            # reach the exact owner because its conditional six-entry loss is



            # the mechanism that lowers the failed entry without rewarding



            # HOLD/MANAGE. A miss may therefore never be hidden by total_r=0.



            route_required = bool(



                lifecycle_shape.initial_sl_failure



                or abs(float(lifecycle_shape.total_r)) > 1e-12



            )



            if not routed_exactly and route_required:



                self._five_lifecycle_route_misses += 1



            self._five_settled_lifecycle_ids.add(identity)



            recorded += 1



        return recorded







    def _refresh_five_competition_lifecycles(



        self,



        policy: object | None = None,



        *,



        round_index: int = 0,



        policy_id: int = 0,



        request: BridgeRequest | None = None,



        account_context_after_close: Mapping[str, object] | None = None,



    ) -> int:



        execution_path = self._latest_runtime_file(self.config.execution_file)



        if execution_path is None or not execution_path.exists():



            return 0



        summary = parse_mt5_execution_ledger(



            execution_path,



            initial_equity=float(self.config.deposit),



            min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



            technical_min_rr=float(self.config.technical_min_rr),



        )



        submitted_before = int(self._last_codex_teacher_phase.get("submitted", 0) or 0)



        recorded = self._record_five_competition_lifecycles(



            summary.lifecycles,



            policy=policy,



            account_context_after_close=account_context_after_close,



        )



        if policy is not None and bool(self.config.codex_teacher_enabled):



            # A newly submitted close now gets a bounded causal barrier. The



            # first verified Codex update must land before the next policy



            # action is sampled; unrelated already-pending cases still drain



            # opportunistically without blocking this bridge request.



            submitted_after = int(self._last_codex_teacher_phase.get("submitted", 0) or 0)



            if (



                submitted_after > submitted_before



                or (recorded > 0 and bool(self._codex_teacher_pending))



            ):



                self._wait_for_codex_teacher_action_barrier(



                    policy,



                    reason="five_lifecycle",



                )



            else:



                self._drain_codex_teacher_results(policy, wait=False)



        # The competition shaper above is not the context-policy settlement



        # path.  Public Five needs both: once a final close is durable, write



        # the same lifecycle R into the owner's contextual ledger before the



        # next action is sampled.  Waiting for ``evaluate_policy`` would make



        # the report correct but leave every in-window expectancy query empty.



        if policy is not None and bool(getattr(self.config, "learning_enabled", True)):



            live_settled = getattr(self, "_five_live_context_settled_lifecycle_ids", set())



            if not isinstance(live_settled, set):



                live_settled = set()



                self._five_live_context_settled_lifecycle_ids = live_settled



            for lifecycle in summary.lifecycles:



                trace_id = str(getattr(lifecycle, "trace_id", "") or "")



                if (



                    not trace_id.startswith("five:")



                    or not bool(getattr(lifecycle, "final_close_confirmed", False))



                ):



                    continue



                identity = "|".join((



                    str(getattr(lifecycle, "position_ticket", "") or ""),



                    trace_id,



                    str(getattr(lifecycle, "close_reason", "") or ""),



                    f"{float(getattr(lifecycle, 'realized_r', 0.0) or 0.0):.8f}",



                ))



                if identity in live_settled:



                    continue



                self._last_live_lifecycle_settlements += 1



                # ``_record_five_competition_lifecycles`` normally resolves



                # the owner action from the trace registry.  Automatic SL/TP



                # rows can arrive without that trace registry entry, while



                # the authoritative lifecycle still carries the immutable



                # entry side.  Use that side as a fail-closed fallback for



                # the run-local exploration sample; it cannot alter reward



                # ownership or ledger reconciliation.



                metadata = dict(getattr(lifecycle, "metadata", {}) or {})



                owner_trace_id = str(



                    metadata.get("position_owner_trace_id", "")



                    or getattr(lifecycle, "trace_id", "")



                    or ""



                ).strip()



                owner_agent = self._five_agent_by_trace.get(owner_trace_id)



                side = int(getattr(lifecycle, "side", 0) or metadata.get("side", 0) or 0)



                if owner_agent not in (0, 1) and side != 0:



                    owner_agent = 0 if side > 0 else 1



                owner_action = self._five_action_by_trace.get(owner_trace_id)



                if owner_action not in self._entry_actions():



                    owner_action = int(



                        Action.MARKET_BUY if int(owner_agent or -1) == 0



                        else Action.MARKET_SELL



                    ) if owner_agent in (0, 1) else None



                if owner_agent in (0, 1) and owner_action is not None:



                    self._record_five_live_owner_outcome(



                        policy,



                        agent_index=int(owner_agent),



                        action=int(owner_action),



                        net_r=float(getattr(lifecycle, "realized_r", 0.0) or 0.0),



                        trace_id=owner_trace_id or trace_id,



                    )



                credited = self._apply_execution_lifecycle_credit(



                    policy=policy,



                    lifecycle=lifecycle,



                    round_index=round_index,



                    policy_id=policy_id,



                    source="live_execution_ledger",



                    request=request,



                )



                self._last_live_lifecycle_updates += int(credited)



                live_settled.add(identity)



        return recorded







    def _refresh_five_pending_order_timeouts(self, policy: object) -> dict[str, int]:



        """Route only five-bar pending-order timeouts to the creating trace.







        Deliberate early cancellations, fills, rejected placements and terminal



        cleanup rows do not create behavioral reward credit.



        """



        result = {



            "observed": 0, "accepted": 0, "updated": 0, "cache_miss": 0,



        }



        timeout_penalty = float(self.config.five_unfilled_pending_timeout_penalty_r)



        hook = getattr(policy, "apply_five_unfilled_pending_timeout_penalty", None)



        if timeout_penalty <= 0.0 or not callable(hook):



            return result



        path = self._latest_runtime_file(self.config.execution_file)



        if path is None or not path.exists():



            return result



        path_key = str(path.resolve())



        size = int(path.stat().st_size)



        if path_key != self._five_execution_timeout_path or size < self._five_execution_timeout_offset:



            self._five_execution_timeout_path = path_key



            self._five_execution_timeout_offset = 0



            self._five_execution_timeout_header = ()



            self._five_settled_pending_timeout_ids.clear()



        pending_names = {"BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"}



        with path.open("rb") as handle:



            if self._five_execution_timeout_offset > 0:



                handle.seek(self._five_execution_timeout_offset)



            else:



                header_line = handle.readline()



                if not header_line.endswith(b"\n"):



                    return result



                header_text = header_line.decode("utf-8-sig", errors="replace").rstrip("\r\n")



                self._five_execution_timeout_header = tuple(



                    next(csv.reader([header_text], delimiter="\t"))



                )



            header = self._five_execution_timeout_header



            if not header:



                return result



            while True:



                row_start = handle.tell()



                raw_line = handle.readline()



                if not raw_line:



                    break



                if not raw_line.endswith(b"\n"):



                    handle.seek(row_start)



                    break



                values = next(csv.reader([



                    raw_line.decode("utf-8", errors="replace").rstrip("\r\n")



                ], delimiter="\t"))



                row = dict(zip(header, values))



                status = str(row.get("status", "") or "").strip().upper()



                if (



                    str(row.get("source", "") or "").strip().lower() != "policy"



                    or str(row.get("transaction_type", "") or "").strip().upper() != "ORDER_DELETE"



                    or status != "UNFILLED_TIMEOUT"



                    or str(row.get("action_name", "") or "").strip().upper() not in pending_names



                    or not _ledger_bool(row.get("cancelled"))



                ):



                    continue



                trace_id = str(row.get("trace_id", "") or "").strip()



                order_ticket = str(row.get("order_ticket", "") or "").strip()



                if not trace_id or not order_ticket or order_ticket == "0":



                    continue



                identity = f"{order_ticket}|{trace_id}"



                if identity in self._five_settled_pending_timeout_ids:



                    continue



                result["observed"] += 1



                routed = hook(trace_id, timeout_penalty)



                routed = dict(routed) if isinstance(routed, Mapping) else {}



                result["accepted"] += int(bool(routed.get("accepted", False)))



                result["updated"] += int(routed.get("updated", 0) or 0)



                result["cache_miss"] += int(routed.get("reason") == "trace_cache_miss")



                self._five_settled_pending_timeout_ids.add(identity)



            self._five_execution_timeout_offset = handle.tell()



        return result







    def _five_invalid_position_action_no_position_penalties(



        self,



        request: BridgeRequest,



        pending_actions: Iterable[Mapping[str, object]],



    ) -> tuple[np.ndarray, int]:



        """Phat raw policy quan ly/dong neu snapshot truoc khong co vi the."""



        positions_before = max(



            0,



            int(getattr(request, "positions", 0) or 0),



            len(self._positions_from_pipe(str(getattr(request, "positions_pipe", "") or ""))),



        )



        penalties = np.zeros(5, dtype=np.float32)



        if positions_before > 0:



            return penalties, positions_before



        penalty_r = float(self.config.five_invalid_position_action_no_position_penalty_r)



        for pending_item in pending_actions:



            agent_id = int(pending_item.get("agent_id", -1) or 0)



            if not 0 <= agent_id < 5:



                continue



            trace = pending_item.get("trace")



            details = dict(getattr(trace, "action_details", {}) or {})



            policy_action = int(details.get(



                "policy_action",



                details.get("five_action_index", pending_item.get("action", Action.HOLD)),



            ) or 0)



            if policy_action in POSITION_REQUIRED_ACTIONS:



                penalties[agent_id] = np.float32(penalty_r)



        return penalties, positions_before







    def _five_invalid_cancel_no_pending_penalties(



        self,



        request: BridgeRequest,



        pending_actions: Iterable[Mapping[str, object]],



    ) -> tuple[np.ndarray, int]:



        """Phat raw policy CANCEL_ORDERS neu snapshot truoc khong co pending."""



        pending_before = sum(



            1



            for item in str(getattr(request, "orders_pipe", "") or "").split("^")



            if item.strip()



        )



        penalties = np.zeros(5, dtype=np.float32)



        if pending_before > 0:



            return penalties, pending_before



        penalty_r = float(self.config.five_invalid_cancel_no_pending_penalty_r)



        for pending_item in pending_actions:



            agent_id = int(pending_item.get("agent_id", -1) or 0)



            if not 0 <= agent_id < 5:



                continue



            trace = pending_item.get("trace")



            details = dict(getattr(trace, "action_details", {}) or {})



            policy_action = int(details.get(



                "policy_action",



                details.get("five_action_index", pending_item.get("action", Action.HOLD)),



            ) or 0)



            if policy_action == int(Action.CANCEL_ORDERS):



                penalties[agent_id] = np.float32(penalty_r)



        return penalties, pending_before







    @staticmethod



    def _five_request_datetime(request: BridgeRequest) -> datetime | None:



        raw = str(getattr(request, "time", "") or "").strip()



        for pattern in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):



            try:



                return datetime.strptime(raw[:19], pattern)



            except ValueError:



                continue



        return None







    def _five_economic_inputs(



        self,



        request: BridgeRequest,



        conflict_cost_r: np.ndarray,



        policy_violation_penalty_r: np.ndarray | None = None,



    ) -> dict[str, object]:



        """Build one causal economic state from owner traces and account facts."""



        marked = self._five_gross_realized_r.astype(np.float32).copy()



        risk = np.zeros(5, dtype=np.float32)



        for position in self._positions_from_pipe(str(request.positions_pipe or "")):



            trace_id = str(position.get("owner_trace_id", "") or "")



            agent = self._five_agent_by_trace.get(trace_id)



            if agent is None:



                continue



            marked[int(agent)] += np.float32(float(position.get("unrealized_r", 0.0) or 0.0))



            risk[int(agent)] += np.float32(1.0)



        costs = self._five_pending_cost_r.astype(np.float32).copy()



        self._five_pending_cost_r.fill(0.0)



        risk_unit = max(1e-9, float(self.config.initial_equity) * 0.01)



        team_equity_r = (float(request.equity) - float(self.config.initial_equity)) / risk_unit



        # One percentage point of account drawdown equals one canonical R.



        team_drawdown_r = max(0.0, float(request.drawdown_pct))



        current_time = self._five_request_datetime(request)



        if self._five_episode_started_at is None:



            self._five_episode_started_at = current_time



        elapsed = 0.0



        step_seconds = 60.0



        if current_time is not None and self._five_episode_started_at is not None:



            elapsed = max(0.0, (current_time - self._five_episode_started_at).total_seconds())



        if current_time is not None and self._five_last_market_time is not None:



            step_seconds = max(1.0, (current_time - self._five_last_market_time).total_seconds())



        self._five_last_market_time = current_time



        self._five_reward_steps += 1



        progress = min(



            1.0,



            float(self._five_reward_steps) / max(16.0, float(self.config.five_training_schedule_steps)),



        )



        return {



            "agent_mark_to_market_r": marked,



            "transaction_cost_r": costs,



            "agent_risk_ratio": risk,



            "conflict_cost_r": np.asarray(conflict_cost_r, dtype=np.float32).reshape(5),



            "policy_violation_penalty_r": (



                np.zeros(5, dtype=np.float32)



                if policy_violation_penalty_r is None



                else np.asarray(policy_violation_penalty_r, dtype=np.float32).reshape(5)



            ),



            "agent_closed_trade_count": self._five_closed_trade_count.copy(),



            "agent_winning_trade_count": self._five_winning_trade_count.copy(),



            "agent_expectancy_shortfall_r": self._five_expectancy_shortfall_r.copy(),



            "agent_tail_loss_cost_r": self._five_tail_loss_cost_r.copy(),



            "agent_realized_trade_r": self._five_realized_trade_r.copy(),



            "team_equity_r": float(team_equity_r),



            "team_drawdown": float(team_drawdown_r),



            "elapsed_seconds": float(elapsed),



            "step_seconds": float(step_seconds),



            "training_progress": float(progress),



        }







    def _finish_five_episode(self) -> None:



        self._five_episode_started_at = None



        self._five_last_market_time = None







    def publish_policy(self, policy) -> dict[str, object]:



        """Bridge chỉ nhận hệ một-context-một-chuyên-gia đã xác minh owner."""



        if not (hasattr(policy, "experts") and hasattr(policy, "nhan_dien") and hasattr(policy, "registry")):



            raise TypeError("MT5 bridge chỉ chấp nhận HeChuyenGiaMotContextMotPolicy đã xác minh owner")



        ensure = getattr(policy, "ensure_compatible", None)



        if callable(ensure):



            ensure()



        policy_input_size = int(getattr(policy, "input_size", 0) or 0)



        policy_output_size = int(getattr(policy, "output_size", 0) or 0)



        if policy_input_size != INPUT_COUNT or policy_output_size != OUTPUT_COUNT:



            raise ValueError(f"He chuyen gia phai dung bridge {INPUT_COUNT}x{OUTPUT_COUNT} (got {policy_input_size}x{policy_output_size}).")



        return {"bridge_mode": "python_action", "policy_runtime": "mot_context_mot_chuyen_gia", "so_chuyen_gia": len(getattr(policy, "experts", {}) or {}), "policy_chung_ra_lenh": False, "tron_policy": False}







    def _clear_bridge_files(self) -> None:



        def remove_if_possible(path: Path) -> None:



            if not path.exists():



                return



            for _ in range(50):



                try:



                    path.unlink()



                    return



                except PermissionError:



                    time.sleep(0.05)



            raise PermissionError(f"Cannot clear locked MT5 bridge file: {path}")







        names = tuple(name for name in (



            self.config.bridge_file, self.config.request_file, self.config.action_file,



            self.config.execution_file, self.config.entry_preference_file, self.config.feature_audit_file,



            self.config.stop_request_marker_file,



            f"{self.config.action_file}.metadata.jsonl",



        ) if str(name or "").strip())



        for base in (self.common_files, self.mql5_files):



            for name in names:



                remove_if_possible(base / name)



        tester_dir = self.runner_dir / "Tester"



        if tester_dir.exists():



            for name in names:



                for path in tester_dir.glob(f"Agent-*/MQL5/Files/{name}"):



                    remove_if_possible(path)



        



        # Create initial empty bridge file so EA can find it



        for base in (self.common_files, self.mql5_files):



            base.mkdir(parents=True, exist_ok=True)



            (base / self.config.bridge_file).write_text("", encoding="utf-8")



            (base / self.config.request_file).write_text("", encoding="utf-8")



            (base / self.config.action_file).write_text("", encoding="utf-8")



            (base / self.config.execution_file).write_text("", encoding="utf-8")



            (base / self.config.entry_preference_file).write_text("", encoding="utf-8")



            if str(self.config.feature_audit_file or "").strip():



                (base / self.config.feature_audit_file).write_text("", encoding="utf-8")







    def _clear_tester_cache(self) -> None:



        cache_dir = self.runner_dir / "Tester" / "cache"



        if not cache_dir.exists():



            return



        expert_prefix = self.config.expert



        symbol = self.config.symbol



        for path in cache_dir.glob(f"{expert_prefix}.{symbol}.*.tst"):



            path.unlink()







    def _close_stale_tester_agents(self) -> None:



        """Stop local Strategy Tester workers so a new /config cannot reuse stale state."""



        if os.name != "nt":



            return



        runner = str(self.runner_dir.resolve()).lower().replace("'", "''")



        script = (



            f"$runner='{runner}';"



            "Get-CimInstance Win32_Process | "



            "Where-Object { $_.Name -ieq 'metatester64.exe' -and "



            "((-not $_.ExecutablePath -and -not $_.CommandLine) -or "



            "($_.ExecutablePath -and $_.ExecutablePath.ToLower().Contains($runner)) -or "



            "($_.CommandLine -and $_.CommandLine.ToLower().Contains($runner))) } | "



            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"



        )



        try:



            subprocess.run(



                ["powershell", "-NoProfile", "-Command", script],



                cwd=str(self.runner_dir),



                stdout=subprocess.DEVNULL,



                stderr=subprocess.DEVNULL,



                timeout=10,



                check=False,



            )



        except Exception:



            return







    def _runner_terminal_is_alive(self) -> bool:



        """Check the portable terminal and its runner-scoped LiveUpdate helper."""



        if os.name != "nt":



            return False



        terminal_path = str(self.terminal.resolve()).lower().replace("'", "''")



        runner = str(self.runner_dir.resolve()).lower().replace("'", "''")



        script = (



            f"$path='{terminal_path}';$runner='{runner}';"



            "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'terminal64.exe' -and "



            "(($_.ExecutablePath -and $_.ExecutablePath.ToLower() -eq $path) -or "



            "($_.CommandLine -and $_.CommandLine.ToLower().Contains($runner))) } | "



            "Select-Object -ExpandProperty ProcessId)"



        )



        try:



            result = subprocess.run(



                ["powershell", "-NoProfile", "-Command", script],



                cwd=str(self.runner_dir),



                capture_output=True,



                text=True,



                timeout=5,



                check=False,



            )



            return bool(str(result.stdout or "").strip())



        except Exception:



            return False







    def _runner_tester_is_alive(self) -> bool:



        """Return whether this portable runner still owns a tester worker."""



        if os.name != "nt":



            return False



        runner = str(self.runner_dir.resolve()).lower().replace("'", "''")



        script = (



            f"$runner='{runner}';"



            "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'metatester64.exe' -and "



            "(($_.ExecutablePath -and $_.ExecutablePath.ToLower().Contains($runner)) -or "



            "($_.CommandLine -and $_.CommandLine.ToLower().Contains($runner))) } | "



            "Select-Object -ExpandProperty ProcessId)"



        )



        try:



            result = subprocess.run(



                ["powershell", "-NoProfile", "-Command", script],



                cwd=str(self.runner_dir),



                capture_output=True,



                text=True,



                timeout=10,



                check=False,



            )



            return bool(str(result.stdout or "").strip())



        except Exception:



            return False







    def _tester_report_ready_for_finalize(



        self,



        located_report: Path | None,



        *,



        process_poll_status: int | None,



        now: float,



        last_request_seen: float,



    ) -> bool:



        """Finish a tester pass without requiring the login terminal to exit."""



        if located_report is None:



            return False



        if process_poll_status is not None:



            return True



        if float(now) - float(last_request_seen) <= 2.0:



            return False



        if not bool(getattr(self.config, "shutdown_terminal", True)):
            if not self._runner_tester_is_alive():
                return True
            try:
                parse_mt5_report(located_report)
                return True
            except Exception:
                return False

        return not self._runner_tester_is_alive()







    def _bridge_path(self) -> Path | None:



        trials = [



            self.mql5_files / self.config.bridge_file,



            self.common_files / self.config.bridge_file,



        ]



        tester_dir = self.runner_dir / "Tester"



        if tester_dir.exists():



            trials.extend(tester_dir.glob(f"Agent-*/MQL5/Files/{self.config.bridge_file}"))



        existing = [path for path in trials if path.exists()]



        if existing:



            return max(existing, key=lambda path: path.stat().st_mtime)



        return None







    def _latest_runtime_file(self, filename: str) -> Path | None:



        trials = [



            self.mql5_files / filename,



            self.common_files / filename,



        ]



        tester_dir = self.runner_dir / "Tester"



        if tester_dir.exists():



            trials.extend(tester_dir.glob(f"Agent-*/MQL5/Files/{filename}"))



        min_mtime_ns = int(getattr(self, "_runtime_file_min_mtime_ns", 0) or 0)



        existing = [



            path for path in trials



            if path.exists() and (min_mtime_ns <= 0 or path.stat().st_mtime_ns >= min_mtime_ns)



        ]



        if existing:



            return max(existing, key=lambda path: path.stat().st_mtime)



        return None







    def _report_candidates(self, report_name: str) -> list[Path]:



        names = [f"{report_name}.htm", f"{report_name}.html", report_name]



        roots = [



            self.runner_dir / "Reports",



            self.runner_dir / "reports",



            self.runner_dir / "MQL5" / "Files" / "Reports",



            self.runner_dir / "MQL5" / "Files" / "reports",



            self.runner_dir / "Common" / "Files" / "Reports",



            self.runner_dir / "Common" / "Files" / "reports",



        ]



        candidates: list[Path] = []



        seen: set[str] = set()



        for root in roots:



            for name in names:



                path = root / name



                key = os.path.normcase(os.path.normpath(str(path)))



                if key not in seen:



                    candidates.append(path)



                    seen.add(key)



        if self.runner_dir.exists():



            for pattern in (f"{report_name}.htm", f"{report_name}.html"):



                for path in self.runner_dir.glob(f"**/{pattern}"):



                    key = os.path.normcase(os.path.normpath(str(path)))



                    if key not in seen:



                        candidates.append(path)



                        seen.add(key)



        return candidates







    def _materialize_report(self, report_name: str, expected_path: Path) -> Path | None:



        min_mtime_ns = int(getattr(self, "_report_min_mtime_ns", 0) or 0)



        if (



            expected_path.exists()



            and expected_path.stat().st_size > 0



            and (min_mtime_ns <= 0 or expected_path.stat().st_mtime_ns >= min_mtime_ns)



        ):



            return expected_path



        existing = [



            path for path in self._report_candidates(report_name)



            if (



                path.exists()



                and path.stat().st_size > 0



                and (min_mtime_ns <= 0 or path.stat().st_mtime_ns >= min_mtime_ns)



            )



        ]



        if not existing:



            return None



        source = max(existing, key=lambda path: path.stat().st_mtime)



        expected_path.parent.mkdir(parents=True, exist_ok=True)



        if source.resolve() != expected_path.resolve():



            shutil.copy2(source, expected_path)



        return expected_path







    def _clear_report_artifacts(self, report_name: str) -> None:



        for path in self._report_candidates(report_name):



            try:



                if path.exists() and path.is_file():



                    path.unlink()



            except OSError:



                continue







    def _session_manifest_path(self, report_name: str) -> Path:



        return self.runner_dir / "Reports" / f"{report_name}.manifest.json"







    def compile_mql5_ea(self, source_path: str | Path | None = None, timeout_sec: int = 60) -> dict[str, object]:



        src = Path(source_path).resolve() if source_path else self._expert_source_path()



        return compile_mql5_ea(self.runner_dir, src, timeout_sec=timeout_sec)







    def _expert_source_path(self) -> Path:



        return self.runner_dir / "MQL5" / "Experts" / f"{self.config.expert}.mq5"







    def _expert_binary_path(self) -> Path:



        return self.runner_dir / "MQL5" / "Experts" / f"{self.config.expert}.ex5"







    def _verify_ea_build_integrity(self) -> dict[str, object]:



        """Fail closed when a local EA source/binary pair is stale or hash-pinned incorrectly."""



        source = self._expert_source_path()



        binary = self._expert_binary_path()



        if source.exists() and bool(self.config.fail_on_stale_ex5):



            if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:



                compile_mql5_ea(self.runner_dir, source)







        source_hash = sha256_file(source)



        binary_hash = sha256_file(binary)



        reasons: list[str] = []



        if not source.exists():



            reasons.append("ea_source_missing")



        if not binary.exists():



            reasons.append("ea_binary_missing")



        if str(self.config.expected_ea_source_sha256 or "").strip():



            expected = str(self.config.expected_ea_source_sha256).strip().lower()



            if source_hash != expected:



                reasons.append("ea_source_hash_mismatch")



        if str(self.config.expected_ea_binary_sha256 or "").strip():



            expected = str(self.config.expected_ea_binary_sha256).strip().lower()



            if binary_hash != expected:



                reasons.append("ea_binary_hash_mismatch")



        if source.exists() and bool(self.config.fail_on_stale_ex5):



            if not binary.exists():



                reasons.append("ea_binary_missing_for_source")



            elif binary.stat().st_mtime_ns < source.stat().st_mtime_ns:



                reasons.append("ea_binary_older_than_source")



        if reasons:



            raise RuntimeError(



                "MT5 EA build integrity failed: "



                + ",".join(reasons)



                + f" source={source} ex5={binary}"



            )



        return {



            "source_path": str(source),



            "source_sha256": source_hash,



            "binary_path": str(binary),



            "binary_sha256": binary_hash,



            "fail_on_stale_ex5": bool(self.config.fail_on_stale_ex5),



        }







    def _write_session_manifest(



        self,



        *,



        report_name: str,



        policy_id: int,



        round_index: int,



        ini_path: Path,



        set_path: Path,



        report_path: Path,



        policy: object,



        stage: str,



        extra: Mapping[str, object] | None = None,



    ) -> Path:



        repo_root = Path(__file__).resolve().parents[2]



        expert_source = self._expert_source_path()



        expert_binary = self._expert_binary_path()



        payload: dict[str, object] = {



            "schema_version": 1,



            "stage": str(stage),



            "report_name": str(report_name),



            "policy_id": int(policy_id),



            "round_index": int(round_index),



            "created_at_ns": time.time_ns(),



            "runner_dir": str(self.runner_dir),



            "expert": str(self.config.expert),



            "symbol": str(self.config.symbol),



            "period": str(self.config.period),



            "from_date": str(self.config.from_date),



            "to_date": str(self.config.to_date),



            "model": int(self.config.model),



            "bridge_schema_version": BRIDGE_SCHEMA_VERSION,



            "execution_ledger_schema_version": EXECUTION_LEDGER_SCHEMA_VERSION,



            "feature_schema_hash": str(self.config.feature_schema_hash),



            "policy_state_sha256": policy_state_hash(policy),



            "python_tree_sha256": sha256_tree(repo_root / "src" / "self_growing_trader", suffixes=(".py",)),



            "protocol_contract_sha256": sha256_file(repo_root / "src" / "self_growing_trader" / "protocol_contract.py"),



            "effective_config_sha256": stable_object_hash(self.config),



            "ea_source_path": str(expert_source),



            "ea_source_sha256": sha256_file(expert_source),



            "ea_binary_path": str(expert_binary),



            "ea_binary_sha256": sha256_file(expert_binary),



            "terminal_path": str(self.terminal),



            "terminal_sha256": sha256_file(self.terminal),



            "tester_ini_path": str(ini_path),



            "tester_ini_sha256": sha256_file(ini_path),



            "tester_set_path": str(set_path),



            "tester_set_sha256": sha256_file(set_path),



            "report_path": str(report_path),



            "report_sha256": sha256_file(report_path),



            "mt5_failure_replay_mode": bool(self.config.mt5_failure_replay_mode),



            "mt5_failure_replay_case_id": str(self.config.mt5_failure_replay_case_id or ""),



            "mt5_failure_replay_anchor_time": str(self.config.mt5_failure_replay_anchor_time or ""),



            "extra": dict(extra or {}),



        }



        path = self._session_manifest_path(report_name)



        path.parent.mkdir(parents=True, exist_ok=True)



        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")



        return path







    @staticmethod



    def _read_request(path: Path) -> BridgeRequest | None:



        try:



            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()



        except OSError:



            return None



        if len(lines) < 2:



            return None



        line = lines[-1].strip()



        delimiter = "\t" if "\t" in line else ","



        parts = line.split(delimiter)



        if len(parts) < 3:



            return None



        try:



            if len(parts) >= 11:



                schema_version = int(float(parts[0]))



                event_id = parts[1]



                sequence = int(float(parts[2]))



                request_time = parts[3]



                balance = float(parts[4])



                equity = float(parts[5])



                drawdown_pct = float(parts[6])



                spread_points = float(parts[7])



                positions = int(float(parts[8]))



                day_trades = int(float(parts[9]))



                inputs_text = parts[10]



                positions_pipe = parts[11] if len(parts) > 11 else ""



                orders_pipe = parts[12] if len(parts) > 12 else ""



                execution_pipe = parts[13] if len(parts) > 13 else ""



                margin = float(parts[14]) if len(parts) > 14 and parts[14] else 0.0



                free_margin = float(parts[15]) if len(parts) > 15 and parts[15] else 0.0



                margin_level_pct = float(parts[16]) if len(parts) > 16 and parts[16] else 0.0



                leverage = int(float(parts[17])) if len(parts) > 17 and parts[17] else 0



            elif len(parts) >= 7:



                schema_version = 1



                event_id = str(parts[0])



                sequence = int(parts[0])



                request_time = parts[1]



                balance = float(parts[2])



                equity = float(parts[3])



                drawdown_pct = 0.0



                spread_points = 0.0



                positions = int(float(parts[4]))



                day_trades = int(float(parts[5]))



                inputs_text = parts[6]



                positions_pipe = ""



                orders_pipe = ""



                execution_pipe = ""



                margin = 0.0



                free_margin = 0.0



                margin_level_pct = 0.0



                leverage = 0



            else:



                schema_version = 1



                event_id = str(parts[0])



                balance = 0.0



                equity = 0.0



                drawdown_pct = 0.0



                spread_points = 0.0



                positions = 0



                day_trades = 0



                sequence = int(parts[0])



                request_time = ""



                inputs_text = parts[2]



                positions_pipe = ""



                orders_pipe = ""



                execution_pipe = ""



                margin = 0.0



                free_margin = 0.0



                margin_level_pct = 0.0



                leverage = 0



            values = [float(value) for value in inputs_text.split("|") if value != ""]



        except ValueError:



            return None



        raw_inputs = np.asarray(values, dtype=float)



        return BridgeRequest(



            sequence=sequence,



            inputs=raw_inputs.copy(),



            raw_inputs=raw_inputs.copy(),



            schema_version=schema_version,



            event_id=event_id,



            time=request_time,



            balance=balance,



            equity=equity,



            margin=margin,



            free_margin=free_margin,



            margin_level_pct=margin_level_pct,



            leverage=leverage,



            drawdown_pct=drawdown_pct,



            spread_points=spread_points,



            positions=positions,



            day_trades=day_trades,



            positions_pipe=positions_pipe,



            orders_pipe=orders_pipe,



            execution_pipe=execution_pipe,



        )







    def _predict_regime_for_request(



        self,



        request: BridgeRequest | None,



    ) -> tuple[int, np.ndarray, str]:



        """Causal Regime Detection on raw wire inputs.







        Evaluates soft probability distribution [P_Bull, P_Bear, P_Chop]



        with sub-5ms CPU latency SLA.



        Fails closed to neutral prior (MarketRegime.CHOP, [1/3, 1/3, 1/3], "CHOP")



        if inputs are None, empty, NaN, Inf, or if exception occurs.



        """



        neutral_probs = np.full(NUM_REGIMES, 1.0 / NUM_REGIMES, dtype=np.float64)



        if request is None:



            return int(MarketRegime.CHOP), neutral_probs, REGIME_NAMES[MarketRegime.CHOP]







        if getattr(request, "regime_probabilities", None) is not None:



            try:



                raw_p = np.asarray(request.regime_probabilities, dtype=np.float64).ravel()



                if (



                    raw_p.size >= NUM_REGIMES



                    and not np.isnan(raw_p[:NUM_REGIMES]).any()



                    and not np.isinf(raw_p[:NUM_REGIMES]).any()



                    and not (raw_p[:NUM_REGIMES] < 0.0).any()



                ):



                    p_sum = float(np.sum(raw_p[:NUM_REGIMES]))



                    if p_sum > 0:



                        soft_probs = (



                            raw_p[:NUM_REGIMES]



                            if np.isclose(p_sum, 1.0, atol=1e-6)



                            else (raw_p[:NUM_REGIMES] / p_sum)



                        )



                        regime_idx = int(request.regime_index if request.regime_index is not None else int(np.argmax(soft_probs)))



                        regime_name = str(request.regime_name or REGIME_NAMES.get(regime_idx, "CHOP"))



                        return regime_idx, soft_probs, regime_name



            except Exception:



                pass







        raw_vector = request.raw_inputs if request.raw_inputs is not None else request.inputs



        if raw_vector is None:



            return int(MarketRegime.CHOP), neutral_probs, REGIME_NAMES[MarketRegime.CHOP]







        try:



            raw_arr = np.asarray(raw_vector, dtype=float).ravel()



            if raw_arr.size == 0 or np.isnan(raw_arr).any() or np.isinf(raw_arr).any():



                return int(MarketRegime.CHOP), neutral_probs, REGIME_NAMES[MarketRegime.CHOP]







            # Tier 1: Higher-Timeframe Trend Invariance Filter (H1 EMA diff & slope)
            if getattr(self.config, "regime_detector_enabled", True) and raw_arr.size >= 90:
                h1_diff = float(raw_arr[88])
                h1_slope = float(raw_arr[89])
                if abs(h1_diff) > 1e-5 or abs(h1_slope) > 1e-5:
                    if h1_diff > 0.05 and h1_slope >= 0.0:
                        probs = np.array([0.80, 0.10, 0.10], dtype=np.float64)
                        return int(MarketRegime.BULL), probs, "BULL"
                    elif h1_diff < -0.05 and h1_slope <= 0.0:
                        probs = np.array([0.10, 0.80, 0.10], dtype=np.float64)
                        return int(MarketRegime.BEAR), probs, "BEAR"

            detector = getattr(self, "regime_detector", None)



            if detector is not None and hasattr(detector, "predict_wire_features"):



                regime_idx, probs = detector.predict_wire_features(raw_arr)



                probs = np.asarray(probs, dtype=np.float64).ravel()



                if probs.shape != (NUM_REGIMES,) or np.isnan(probs).any() or np.isinf(probs).any():



                    return int(MarketRegime.CHOP), neutral_probs, REGIME_NAMES[MarketRegime.CHOP]



                prob_sum = float(np.sum(probs))



                if prob_sum > 0:



                    probs = probs / prob_sum



                else:



                    probs = neutral_probs



                regime_name = REGIME_NAMES.get(int(regime_idx), "CHOP")



                return int(regime_idx), probs, regime_name



            return int(MarketRegime.CHOP), neutral_probs, REGIME_NAMES[MarketRegime.CHOP]



        except Exception:



            return int(MarketRegime.CHOP), neutral_probs, REGIME_NAMES[MarketRegime.CHOP]







    def _record_regime_telemetry(



        self,



        request: BridgeRequest,



        soft_probs: Sequence[float],



        regime_name: str,



        regime_idx: int,



        report_path: Path | None = None,



        event_type: str = "step",



    ) -> None:



        """Stream soft probabilities to mt5_diagnostics.json and events.jsonl."""



        probs_list = [round(float(p), 6) for p in soft_probs]



        # 1. Update mt5_diagnostics.json if candidate / report directory exists



        diag_candidates: list[Path] = []



        if report_path is not None:



            diag_candidates.append(Path(report_path).parent / "mt5_diagnostics.json")



        configured_diag = getattr(self.config, "diagnostics_path", None)



        if configured_diag:



            diag_candidates.append(Path(configured_diag))



        diag_candidates.append(self.runner_dir / "mt5_diagnostics.json")







        seq_val = int(getattr(request, "sequence", 0) or 0)



        if seq_val % 20 == 0 or str(event_type) != "step":



            for dp in diag_candidates:



                if dp.exists():



                    update_diagnostics_regime_telemetry(dp, probs_list, regime_name)







        # 2. Append to events.jsonl



        events_candidates: list[Path] = []



        if report_path is not None:



            events_candidates.append(Path(report_path).parent / "events.jsonl")



        configured_events = getattr(self.config, "events_path", None)



        if configured_events:



            events_candidates.append(Path(configured_events))



        events_candidates.append(self.runner_dir / "events.jsonl")







        record = {



            "schema_version": 10,



            "event_type": str(event_type),



            "sequence": int(getattr(request, "sequence", 0) or 0),



            "time": str(getattr(request, "time", "") or ""),



            "created_at": time.time(),



            "payload": {



                "regime_distribution": probs_list,



                "regime_name": str(regime_name),



                "regime_index": int(regime_idx),



                "sequence": int(getattr(request, "sequence", 0) or 0),



                "time": str(getattr(request, "time", "") or ""),



            },



        }



        line = json.dumps(record, ensure_ascii=False) + "\n"



        for ep in events_candidates:



            if ep.exists() or (report_path is not None and ep == Path(report_path).parent / "events.jsonl"):



                try:



                    ep.parent.mkdir(parents=True, exist_ok=True)



                    with ep.open("a", encoding="utf-8") as f:



                        f.write(line)



                except Exception:



                    pass







    def _normalize_request_inputs(self, request: BridgeRequest) -> BridgeRequest:



        """Normalize only the policy view while preserving the raw regime view.







        ``structure_aware_v4`` centers native MTF volatility ratios around



        zero.  That is correct for Direct-RNN representation learning, but a



        regime classifier must not reinterpret those transformed values as raw



        M15/H1 ATR ratios.  Carry both views in the immutable request instead



        of making the classifier bypass policy normalization globally.



        """







        mode = str(self.config.input_normalization_mode or "raw").strip().lower()



        raw_inputs = np.asarray(



            request.raw_inputs if request.raw_inputs is not None else request.inputs,



            dtype=float,



        ).reshape(-1).copy()



        if getattr(request, "regime_probabilities", None) is not None:



            try:



                raw_probs = np.asarray(request.regime_probabilities, dtype=float).ravel()



                if (



                    raw_probs.size >= NUM_REGIMES



                    and not np.isnan(raw_probs[:NUM_REGIMES]).any()



                    and not np.isinf(raw_probs[:NUM_REGIMES]).any()



                    and not (raw_probs[:NUM_REGIMES] < 0.0).any()



                ):



                    p_sum = float(np.sum(raw_probs[:NUM_REGIMES]))



                    if p_sum > 0:



                        soft_probs = (



                            raw_probs[:NUM_REGIMES]



                            if np.isclose(p_sum, 1.0, atol=1e-6)



                            else (raw_probs[:NUM_REGIMES] / p_sum)



                        )



                        regime_idx = int(request.regime_index if request.regime_index is not None else np.argmax(soft_probs))



                        regime_name = str(request.regime_name or REGIME_NAMES.get(regime_idx, "CHOP"))



                    else:



                        regime_idx, soft_probs, regime_name = self._predict_regime_for_request(



                            replace(request, regime_probabilities=None)



                        )



                else:



                    regime_idx, soft_probs, regime_name = self._predict_regime_for_request(



                        replace(request, regime_probabilities=None)



                    )



            except Exception:



                regime_idx, soft_probs, regime_name = self._predict_regime_for_request(



                    replace(request, regime_probabilities=None)



                )



        else:



            regime_idx, soft_probs, regime_name = self._predict_regime_for_request(request)



        regime_probs_tuple = tuple(float(p) for p in soft_probs)



        if mode == "raw":



            # Keep legacy/fake bridge fixtures that intentionally use a short



            # vector compatible; the real EA contract is checked by the



            # feature-audit/schema gate before promotion.



            return replace(



                request,



                inputs=np.asarray(request.inputs, dtype=float).reshape(-1).copy(),



                raw_inputs=raw_inputs,



                regime_index=regime_idx,



                regime_probabilities=regime_probs_tuple,



                regime_name=regime_name,



            )



        normalized = normalize_input_vector(request.inputs, mode=mode, size=INPUT_COUNT)



        return replace(



            request,



            inputs=normalized,



            raw_inputs=raw_inputs,



            regime_index=regime_idx,



            regime_probabilities=regime_probs_tuple,



            regime_name=regime_name,



        )







    @staticmethod



    def _parse_execution_records(value: str) -> list[dict[str, object]]:



        """Parse one or more v4 execution records from ``execution_pipe``.







        EA v4 currently emits one aggregate record for the prior policy action.



        The record separator is reserved for forward-compatible multi-event



        responses; v1-v3 payloads remain accepted as a single record.



        """



        raw = str(value or "").strip()



        if not raw:



            return []



        records: list[dict[str, object]] = []



        for record_text in raw.split("^"):



            parsed: dict[str, object] = {}



            for item in re.split(r"[|;]", record_text):



                item = item.strip()



                if not item:



                    continue



                if "=" in item:



                    key, raw_value = item.split("=", 1)



                elif ":" in item:



                    key, raw_value = item.split(":", 1)



                else:



                    continue



                key = key.strip().lower()



                raw_value = raw_value.strip()



                if not key:



                    continue



                normalized = raw_value.lower()



                if normalized in {"1", "true", "yes", "y", "ok", "success"}:



                    parsed[key] = True



                elif normalized in {"0", "false", "no", "n", "", "fail", "failed"}:



                    parsed[key] = False



                else:



                    parsed[key] = raw_value



            if parsed:



                records.append(parsed)



        return records







    @classmethod



    def _parse_execution_pipe(cls, value: str) -> dict[str, object]:



        """Backward-compatible single-record accessor."""



        records = cls._parse_execution_records(value)



        return records[0] if records else {}







    @staticmethod



    def _is_flat_idle_response(



        before: BridgeRequest | None,



        after: BridgeRequest | None,



        requested_action: int,



        acknowledgement: Mapping[str, object] | None = None,



    ) -> bool:



        """True only for an MT5 flat snapshot with no matching execution.







        This is deliberately narrower than ``positions == 0``: an entry intent



        that has not yet been acknowledged is an *unconfirmed intent*, not a



        flat-idle feedback event.  The firebreak applies to a policy HOLD/no



        command followed by an empty MT5 state.



        """



        if int(requested_action) not in {int(Action.HOLD), NO_NEURAL_COMMAND}:



            return False



        before = before or BridgeRequest(sequence=-1, inputs=np.zeros(0, dtype=float))



        after = after or before



        if acknowledgement:



            return False



        return bool(



            int(getattr(before, "positions", 0)) == 0



            and int(getattr(after, "positions", 0)) == 0



            and not str(getattr(before, "orders_pipe", "") or "").strip()



            and not str(getattr(after, "orders_pipe", "") or "").strip()



        )







    @classmethod



    def _execution_details(



        cls,



        before: BridgeRequest | None,



        after: BridgeRequest | None,



        requested_action: int,



        *,



        expected_trace_id: str = "",



        expected_agent_id: int | None = None,



    ) -> dict[str, object]:



        """Return execution evidence tied to one exact policy sequence.







        Schema v4 makes the transaction ledger authoritative.  In particular, a



        position disappearing after a HOLD must never be inferred as evidence



        that HOLD closed it; only an acknowledgement whose ``ack_sequence``



        matches the pending policy sequence can receive action credit.



        """



        before = before or BridgeRequest(sequence=-1, inputs=np.zeros(0, dtype=float))



        after = after or before



        # An internal V26 NO_NEURAL_COMMAND is serialized to the EA as wire



        # action=0, but it is not a HOLD decision and must not receive action



        # feedback when unrelated SL/TP/server events occur on the next tick.



        if int(requested_action) < 0:



            return {



                "bridge_observed": True,



                "requested_action": int(requested_action),



                "wire_noop": True,



                "no_neural_command": True,



                "acknowledged": False,



                "accepted": False,



                "placed": False,



                "filled": False,



                "cancelled": False,



                "modified": False,



                "closed": False,



                "rejected": False,



                "execution_confirmed": False,



                "outcome_eligible": False,



                "action_feedback_eligible": False,



                "memory_eligible": False,



                "ledger_required": int(getattr(after, "schema_version", 1)) >= 4,



            }



        records = cls._parse_execution_records(getattr(after, "execution_pipe", ""))



        expected_sequence = int(getattr(before, "sequence", -1))



        ack: dict[str, object] = {}



        for candidate in records:



            raw_sequence = candidate.get("ack_sequence", candidate.get("sequence", ""))



            try:



                if int(float(raw_sequence)) != expected_sequence:



                    continue



            except (TypeError, ValueError):



                continue



            if expected_trace_id and str(



                candidate.get("intent_trace_id", candidate.get("trace_id", "")) or ""



            ) != str(expected_trace_id):



                continue



            if expected_agent_id is not None:



                try:



                    if int(float(candidate.get("agent_id", -1))) != int(expected_agent_id):



                        continue



                except (TypeError, ValueError):



                    continue



            ack = candidate



            break







        if cls._is_flat_idle_response(before, after, requested_action, ack):



            return {



                "bridge_observed": True,



                "requested_action": int(requested_action),



                "execution_state": "FLAT_IDLE",



                "flat_idle": True,



                "acknowledged": False,



                "accepted": False,



                "placed": False,



                "filled": False,



                "cancelled": False,



                "modified": False,



                "closed": False,



                "rejected": False,



                "execution_confirmed": False,



                "outcome_eligible": False,



                "action_feedback_eligible": False,



                "memory_eligible": False,



                "ledger_required": int(getattr(after, "schema_version", 1)) >= 4,



                "action_attribution_source": "flat_idle",



            }







        def truthy(*keys: str) -> bool:



            for key in keys:



                value = ack.get(key)



                if isinstance(value, bool) and value:



                    return True



                if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "ok", "success"}:



                    return True



            return False







        positions_changed = (



            int(getattr(before, "positions", 0)) != int(getattr(after, "positions", 0))



            or (



                bool(getattr(before, "positions_pipe", ""))



                and bool(getattr(after, "positions_pipe", ""))



                and str(before.positions_pipe) != str(after.positions_pipe)



            )



        )



        orders_changed = (



            bool(getattr(before, "orders_pipe", ""))



            and bool(getattr(after, "orders_pipe", ""))



            and str(before.orders_pipe) != str(after.orders_pipe)



        )



        day_trades_changed = int(getattr(after, "day_trades", 0)) > int(getattr(before, "day_trades", 0))



        accepted = truthy("accepted", "executed")



        placed = truthy("placed")



        filled = truthy("filled", "opened", "partially_filled", "partial_fill")



        cancelled = truthy("cancelled", "canceled")



        modified = truthy("modified")



        closed = truthy("closed")



        acknowledged = bool(accepted or placed or filled or cancelled or modified or closed)



        rejected = truthy("rejected")







        ledger_required = int(getattr(after, "schema_version", 1)) >= 4



        if ledger_required:



            # No matching acknowledgement is deliberately not backfilled from a



            # changed snapshot. That change may have been a server-side SL/TP,



            # a safety close, or another action's pending-order fill.



            execution_confirmed = bool(ack and acknowledged and not rejected)



            outcome_eligible = bool(execution_confirmed and (filled or closed))



        else:



            execution_confirmed = bool(acknowledged or positions_changed or orders_changed or day_trades_changed)



            outcome_eligible = bool(



                (filled or closed or positions_changed or day_trades_changed) and not rejected



            )



        # A fill is kept only to bind the original action trace to its eventual



        # lifecycle. Immediate policy credit is deliberately delayed until a



        # realised close, which eliminates reward for PLACED/ACCEPTED intents.



        action_feedback_eligible = bool(outcome_eligible and closed)



        memory_eligible = bool(outcome_eligible)







        return {



            "bridge_observed": True,



            "requested_action": int(requested_action),



            "ack_sequence": ack.get("ack_sequence", ack.get("sequence", "")),



            "intent_event_id": ack.get("intent_event_id", ack.get("event_id", "")),



            "intent_trace_id": ack.get("intent_trace_id", ack.get("trace_id", "")),



            "agent_id": (



                int(expected_agent_id)



                if expected_agent_id is not None



                else ack.get("agent_id", "")



            ),



            "positions_changed": bool(positions_changed),



            "orders_changed": bool(orders_changed),



            "day_trades_changed": bool(day_trades_changed),



            "acknowledged": bool(acknowledged),



            "accepted": bool(accepted),



            "placed": bool(placed),



            "filled": bool(filled),



            "cancelled": bool(cancelled),



            "modified": bool(modified),



            "closed": bool(closed),



            "rejected": bool(rejected),



            "execution_confirmed": bool(execution_confirmed),



            "outcome_eligible": bool(outcome_eligible),



            "action_feedback_eligible": bool(action_feedback_eligible),



            "memory_eligible": bool(memory_eligible),



            "ledger_required": bool(ledger_required),



            "retcode": ack.get("retcode", ""),



            "ticket": ack.get("position_ticket", ack.get("order_ticket", ack.get("ticket", ""))),



            "order_ticket": ack.get("order_ticket", ""),



            "deal_ticket": ack.get("deal_ticket", ""),



            "position_ticket": ack.get("position_ticket", ""),



            "transaction_type": ack.get("transaction_type", ""),



            "execution_status": ack.get("status", ""),



            "deal_entry": ack.get("deal_entry", ""),



            "deal_reason": ack.get("deal_reason", ""),



            "close_reason": ack.get("close_reason", ""),



            "volume": ack.get("volume", ""),



            "price": ack.get("price", ""),



            "profit": ack.get("profit", ""),



            "commission": ack.get("commission", ""),



            "swap": ack.get("swap", ""),



            "initial_risk": ack.get("initial_risk", ""),



            "entry_price": ack.get("entry_price", ""),



            "initial_sl_price": ack.get("initial_sl_price", ""),



            "initial_tp_price": ack.get("initial_tp_price", ""),



            "initial_risk_distance_price": ack.get("initial_risk_distance_price", ""),



            "final_sl_price": ack.get("final_sl_price", ""),



            "final_tp_price": ack.get("final_tp_price", ""),



            "old_sl_price": ack.get("old_sl_price", ""),



            "old_tp_price": ack.get("old_tp_price", ""),



            "new_sl_price": ack.get("new_sl_price", ""),



            "new_tp_price": ack.get("new_tp_price", ""),



            "stop_controller": ack.get("stop_controller", ""),



            "stop_reason": ack.get("stop_reason", ""),



            "exit_kind": ack.get("exit_kind", ""),



            "exit_controller": ack.get("exit_controller", ""),



            "exit_trace_id": ack.get("exit_trace_id", ""),



            "exit_requested_action": ack.get("exit_requested_action", ""),



            "post_exit_horizon_bars": ack.get("post_exit_horizon_bars", ""),



            "post_exit_hold_r": ack.get("post_exit_hold_r", ""),



            "geometry_entry_pipe": ack.get("geometry_entry_pipe", ""),



            "geometry_path_pipe": ack.get("geometry_path_pipe", ""),



            "codex_decision_account": {



                "balance": float(getattr(before, "balance", 0.0) or 0.0),



                "equity": float(getattr(before, "equity", 0.0) or 0.0),



                "drawdown_pct": float(getattr(before, "drawdown_pct", 0.0) or 0.0),



                "positions": float(getattr(before, "positions", 0) or 0),



                "sequence": float(getattr(before, "sequence", 0) or 0),



            },



            "source": ack.get("source", ""),



        }







    @staticmethod



    def _choose_action(



        outputs: np.ndarray,



        allowed_actions: set[int] | None = None,



    ) -> tuple[int, float]:



        """Choose an action after applying the MT5 execution-state mask.







        The policy output includes close/cancel primitives.  They are invalid



        before a position or pending order exists, yet an unconstrained argmax



        can select them indefinitely for a fresh policy.  The simulator masks



        these actions already; the MT5 bridge must apply the same invariant.



        """



        values = np.asarray(outputs, dtype=float).reshape(-1)



        if values.size == 0:



            return 0, 0.0



        action_count, _, _ = action_schema_for_outputs(values)



        scores = values[:action_count].copy()



        if allowed_actions is not None:



            allowed = {int(item) for item in allowed_actions if 0 <= int(item) < action_count}



            if not allowed:



                allowed = {0}



            for index in range(action_count):



                if index not in allowed:



                    scores[index] = -np.inf



        action = int(np.argmax(scores))



        return action, float(values[action])







    @staticmethod



    def _choose_non_hold_command(



        outputs: np.ndarray,



        allowed_actions: set[int] | None = None,



        *,



        min_score: float = 0.0,



    ) -> tuple[int, float]:



        """Choose an actionable command while excluding HOLD from neural choice.







        Continuous V26 treats an absence of command as ``NO_NEURAL_COMMAND``.



        The bridge will serialize that as Action.HOLD=0 only because the EA



        needs a no-op integer. The HOLD neuron is never selected, rewarded or



        stored as the action for this decision.



        """



        values = np.asarray(outputs, dtype=float).reshape(-1)



        action_count, _, _ = action_schema_for_outputs(values)



        if action_count <= 1:



            return NO_NEURAL_COMMAND, 0.0



        allowed = set(range(1, action_count))



        if allowed_actions is not None:



            allowed &= {int(item) for item in allowed_actions if 1 <= int(item) < action_count}



        if not allowed:



            return NO_NEURAL_COMMAND, 0.0



        best = max(allowed, key=lambda index: float(values[index]))



        score = float(values[best])



        if not np.isfinite(score) or score < float(min_score):



            return NO_NEURAL_COMMAND, score if np.isfinite(score) else 0.0



        return int(best), score







    def _allowed_actions_for_request(self, request: BridgeRequest) -> set[int]:



        """Canonical bridge mask matching :func:`allowed_actions_for_state`.







        Pending orders consume capacity only when a positive quota is



        configured. ``max_positions=0`` means unlimited software capacity;



        broker validity and free margin remain authoritative.



        """



        positions = max(0, int(getattr(request, "positions", 0)))



        max_positions = int(self.config.max_positions)



        parsed_positions = self._positions_from_pipe(str(getattr(request, "positions_pipe", "") or ""))



        orders_pipe = str(getattr(request, "orders_pipe", "") or "")



        pending_count = sum(1 for item in orders_pipe.split("^") if item.strip())



        ages = [float(item.get("age_bars", 0.0) or 0.0) for item in parsed_positions]



        unrealized = [float(item.get("unrealized_r", 0.0) or 0.0) for item in parsed_positions]



        allowed = {



            int(action)



            for action in allowed_actions_for_state(



                position_count=positions,



                pending_count=pending_count,



                max_positions=max_positions,



                ages_bars=ages,



                unrealized_rs=unrealized,



                min_manual_close_age_bars=max(0, int(getattr(self.config, "min_manual_close_age_bars", MIN_MANUAL_CLOSE_AGE_BARS))),



                emergency_close_r=EMERGENCY_CLOSE_R,



            )



        }



        if positions > 0:



            allowed.add(int(Action.MANAGE_POSITION))



        max_daily = int(getattr(self.config, "max_daily_trades", 0))



        if bool(getattr(self.config, "enforce_entry_limits", False)) and max_daily > 0 and int(getattr(request, "day_trades", 0)) >= max_daily:



            allowed.difference_update(self._entry_actions())



        if (



            bool(getattr(self.config, "entry_guard_single_position", False))



            and (positions > 0 or pending_count > 0)



        ):



            allowed.difference_update(self._entry_actions())



        return allowed







    @staticmethod



    def _entry_actions() -> set[int]:



        return {



            int(Action.MARKET_BUY), int(Action.MARKET_SELL),



            int(Action.BUY_LIMIT), int(Action.SELL_LIMIT),



            int(Action.BUY_STOP), int(Action.SELL_STOP),



        }







    def _entry_loss_guard_blocks_entries(



        self,



        *,



        closed_count: int,



        closed_net_r: float,



        consecutive_losses: int,



    ) -> bool:



        if not (



            bool(getattr(self.config, "enforce_entry_limits", False))



            and bool(getattr(self.config, "entry_loss_guard_enabled", False))



        ):



            return False



        if int(closed_count) >= int(getattr(self.config, "entry_loss_guard_min_closed", 6)) and float(closed_net_r) <= -float(getattr(self.config, "entry_loss_guard_max_closed_loss_r", 0.25)):



            return True



        if int(consecutive_losses) >= int(getattr(self.config, "entry_loss_guard_consecutive_losses", 4)):



            return True



        return False







    @staticmethod



    def _positions_from_pipe(positions_pipe: str) -> list[dict[str, object]]:



        positions: list[dict[str, object]] = []



        for item in str(positions_pipe or "").split("^"):



            if not item:



                continue



            parts = item.split(":")



            if len(parts) < 8:



                continue



            try:



                positions.append({



                    "ticket": str(parts[0]),



                    "side": int(float(parts[1])),



                    "volume": float(parts[2]),



                    "entry": float(parts[3]),



                    "sl": float(parts[4]),



                    "tp": float(parts[5]),



                    "unrealized_r": float(parts[6]),



                    "age_bars": float(parts[7]),



                    "mfe_r": float(parts[8]) if len(parts) > 8 else float(parts[6]),



                    "mae_r": float(parts[9]) if len(parts) > 9 else float(parts[6]),



                    "dist_tp_atr": float(parts[10]) if len(parts) > 10 else 0.0,



                    "dist_sl_atr": float(parts[11]) if len(parts) > 11 else 0.0,



                    "min_dist_tp_atr": float(parts[12]) if len(parts) > 12 else 0.0,



                    "min_dist_sl_atr": float(parts[13]) if len(parts) > 13 else 0.0,



                    "bars_since_mfe": float(parts[14]) if len(parts) > 14 else 0.0,



                    "near_tp": bool(float(parts[15])) if len(parts) > 15 else False,



                    "giveback_r": float(parts[16]) if len(parts) > 16 else 0.0,



                    "position_owner_context": str(parts[17]) if len(parts) > 17 else "",



                    "position_owner_expert": str(parts[18]) if len(parts) > 18 else "",



                    "position_owner_policy": str(parts[19]) if len(parts) > 19 else "",



                    "owner_trace_id": str(parts[20]) if len(parts) > 20 else "",



                })



            except ValueError:



                continue



        return positions







    @staticmethod



    def _trace_context_audit(trace: object | None) -> dict[str, object]:



        audit = dict(getattr(trace, "context_audit", {}) or {}) if trace is not None else {}



        for key in ("ma_context", "ma_chuyen_gia", "ma_policy", "phien_ban_policy", "trang_thai_chuyen_gia"):



            value = getattr(trace, key, None) if trace is not None else None



            if value not in (None, ""):



                audit[key] = value



        return audit







    @staticmethod



    def _context_code_from_audit(audit: Mapping[str, object]) -> str:



        return str(



            audit.get("context_nhan_dien")



            or audit.get("ma_context")



            or audit.get("market_context")



            or ""



        )







    def _emit_position_context_extrema_event(



        self,



        policy: object,



        state: PositionContextExtremaState,



        *,



        request: BridgeRequest,



        current_context: str,



        reason: str,



    ) -> None:



        if not np.isfinite(state.high_r) or not np.isfinite(state.low_r):



            return



        payload = {



            "schema_version": 1,



            "event_type": "position_context_extrema_until_context_change",



            "reason": str(reason),



            "position_ticket": str(state.ticket),



            "position_owner_context": str(state.owner_context),



            "position_owner_expert": str(state.owner_expert),



            "position_owner_policy": str(state.owner_policy),



            "owner_trace_id": str(state.owner_trace_id),



            "side": int(state.side),



            "entry": float(state.entry),



            "segment_context": str(state.segment_context),



            "new_context": str(current_context),



            "segment_start_sequence": int(state.segment_start_sequence),



            "segment_start_time": str(state.segment_start_time),



            "closed_sequence": int(getattr(request, "sequence", 0) or 0),



            "closed_time": str(getattr(request, "time", "") or ""),



            "last_unrealized_r": float(state.last_r),



            "last_sequence": int(state.last_sequence),



            "last_time": str(state.last_time),



            "bars_from_segment_start_to_low": max(0, int(state.low_sequence) - int(state.segment_start_sequence)),



            "bars_from_segment_start_to_high": max(0, int(state.high_sequence) - int(state.segment_start_sequence)),



            "recovery_after_low_r": float(



                max(



                    0.0,



                    (state.high_r - state.low_r)



                    if int(state.high_sequence) >= int(state.low_sequence)



                    else (state.last_r - state.low_r),



                )



            ),



            "giveback_from_high_r": float(max(0.0, state.high_r - state.last_r)),



            "high": {



                "unrealized_r": float(state.high_r),



                "sequence": int(state.high_sequence),



                "time": str(state.high_time),



                "position": dict(state.high_position),



                "context_audit": dict(state.high_context_audit),



                "observation": np.asarray(state.high_observation, dtype=float).tolist(),



            },



            "low": {



                "unrealized_r": float(state.low_r),



                "sequence": int(state.low_sequence),



                "time": str(state.low_time),



                "position": dict(state.low_position),



                "context_audit": dict(state.low_context_audit),



                "observation": np.asarray(state.low_observation, dtype=float).tolist(),



            },



        }



        hook = getattr(policy, "record_position_context_extrema", None)



        if callable(hook):



            try:



                hook(payload)



            except Exception:



                return







    def _update_position_context_extrema(



        self,



        *,



        policy: object,



        request: BridgeRequest,



        trace: object | None,



    ) -> None:



        audit = self._trace_context_audit(trace)



        current_context = self._context_code_from_audit(audit)



        if not current_context:



            return



        positions = self._positions_from_pipe(str(getattr(request, "positions_pipe", "") or ""))



        open_tickets = {str(item.get("ticket", "")) for item in positions if str(item.get("ticket", ""))}



        for ticket in list(self._position_context_extrema):



            if ticket not in open_tickets:



                self._emit_position_context_extrema_event(



                    policy,



                    self._position_context_extrema[ticket],



                    request=request,



                    current_context=current_context,



                    reason="position_closed_or_missing",



                )



                self._position_context_extrema.pop(ticket, None)



        for position in positions:



            ticket = str(position.get("ticket", "") or "")



            if not ticket:



                continue



            unrealized = float(position.get("unrealized_r", 0.0) or 0.0)



            state = self._position_context_extrema.get(ticket)



            if state is None:



                state = PositionContextExtremaState(



                    ticket=ticket,



                    owner_context=str(position.get("position_owner_context", "") or ""),



                    owner_expert=str(position.get("position_owner_expert", "") or ""),



                    owner_policy=str(position.get("position_owner_policy", "") or ""),



                    owner_trace_id=str(position.get("owner_trace_id", "") or ""),



                    side=int(position.get("side", 0) or 0),



                    entry=float(position.get("entry", 0.0) or 0.0),



                    segment_context=current_context,



                    segment_start_sequence=int(getattr(request, "sequence", 0) or 0),



                    segment_start_time=str(getattr(request, "time", "") or ""),



                )



                self._position_context_extrema[ticket] = state



            elif state.segment_context and state.segment_context != current_context:



                self._emit_position_context_extrema_event(



                    policy,



                    state,



                    request=request,



                    current_context=current_context,



                    reason="market_context_changed",



                )



                state.segment_context = current_context



                state.segment_start_sequence = int(getattr(request, "sequence", 0) or 0)



                state.segment_start_time = str(getattr(request, "time", "") or "")



                state.high_r = -float("inf")



                state.low_r = float("inf")



                state.high_observation = np.zeros(0, dtype=float)



                state.low_observation = np.zeros(0, dtype=float)



                state.high_position = {}



                state.low_position = {}



                state.high_context_audit = {}



                state.low_context_audit = {}



            state.owner_context = str(position.get("position_owner_context", state.owner_context) or state.owner_context)



            state.owner_expert = str(position.get("position_owner_expert", state.owner_expert) or state.owner_expert)



            state.owner_policy = str(position.get("position_owner_policy", state.owner_policy) or state.owner_policy)



            state.owner_trace_id = str(position.get("owner_trace_id", state.owner_trace_id) or state.owner_trace_id)



            state.last_r = unrealized



            state.last_sequence = int(getattr(request, "sequence", 0) or 0)



            state.last_time = str(getattr(request, "time", "") or "")



            state.last_position = dict(position)



            if unrealized >= state.high_r:



                state.high_r = unrealized



                state.high_sequence = int(getattr(request, "sequence", 0) or 0)



                state.high_time = str(getattr(request, "time", "") or "")



                state.high_observation = np.asarray(getattr(request, "inputs", np.zeros(0)), dtype=float).copy()



                state.high_position = dict(position)



                state.high_context_audit = dict(audit)



            if unrealized <= state.low_r:



                state.low_r = unrealized



                state.low_sequence = int(getattr(request, "sequence", 0) or 0)



                state.low_time = str(getattr(request, "time", "") or "")



                state.low_observation = np.asarray(getattr(request, "inputs", np.zeros(0)), dtype=float).copy()



                state.low_position = dict(position)



                state.low_context_audit = dict(audit)







    def _account_context_for_request(self, request: BridgeRequest, equity_history: list[float]) -> dict[str, float]:



        balance = float(request.balance)



        equity = float(request.equity)



        positions = self._positions_from_pipe(str(request.positions_pipe or ""))



        floating_pnl_r = float(sum(float(item.get("unrealized_r", 0.0) or 0.0) for item in positions))



        negative_positions = [item for item in positions if float(item.get("unrealized_r", 0.0) or 0.0) < 0.0]



        max_negative_age = max((float(item.get("age_bars", 0.0) or 0.0) for item in negative_positions), default=0.0)



        count_reference = float(position_count_reference(self.config.max_positions))



        exposure_ratio = float(np.clip(float(request.positions) / count_reference, 0.0, 2.0))



        raw_inputs = np.asarray(



            request.raw_inputs if request.raw_inputs is not None else request.inputs,



            dtype=float,



        ).reshape(-1)



        monetary_open_risk_ratio = float(



            np.clip(raw_inputs[ACCOUNT_OFFSET + 1], 0.0, 2.0)



            if raw_inputs.size > ACCOUNT_OFFSET + 1 else 0.0



        )



        history = [float(item) for item in equity_history if float(item) > 0.0]







        def equity_return(lookback: int) -> float:



            if equity <= 0.0 or len(history) <= lookback or history[-1 - lookback] <= 0.0:



                return 0.0



            return float(np.clip(np.log(equity / history[-1 - lookback]), -0.20, 0.20))







        returns: list[float] = []



        recent = history[-21:] + ([equity] if equity > 0.0 else [])



        for prev, curr in zip(recent, recent[1:]):



            if prev > 0.0 and curr > 0.0:



                returns.append(float(np.log(curr / prev)))



        equity_volatility_20 = float(np.clip(np.std(returns[-20:]) if returns else 0.0, 0.0, 0.20))



        equity_to_balance = float(np.clip((equity / balance) - 1.0, -0.50, 0.50)) if balance > 0.0 and equity > 0.0 else 0.0



        drawdown = float(np.clip(request.drawdown_pct / 100.0, 0.0, 1.0))



        return {



            "schema_version": 7.0,



            "equity_to_balance": equity_to_balance,



            "floating_pnl_r": float(np.clip(floating_pnl_r / 5.0, -2.0, 2.0)),



            "current_equity_drawdown": drawdown,



            "balance_drawdown": drawdown,



            "open_risk_ratio": monetary_open_risk_ratio,



            "equity_return_1_bar": equity_return(1),



            "equity_return_5_bar": equity_return(5),



            "equity_volatility_20": equity_volatility_20,



            "losing_streak_norm": float(np.clip(len(negative_positions) / count_reference, 0.0, 1.0)),



            "time_under_water_norm": float(np.clip(max_negative_age / max(1.0, float(self.config.max_holding_bars)), 0.0, 2.0)),



            "position_count_norm": exposure_ratio,



            "day_trade_pressure": 0.0,



        }







    def _execution_context_for_request(self, request: BridgeRequest, allowed_actions: set[int]) -> dict[str, float]:



        """Bridge facts that must be visible before M1 selector chooses."""



        max_positions = int(self.config.max_positions)



        positions = max(0, int(getattr(request, "positions", 0)))



        pending = sum(1 for item in str(getattr(request, "orders_pipe", "") or "").split("^") if item.strip())



        capacity_remaining = (



            1.0



            if not position_limit_enabled(max_positions)



            else float(np.clip((max_positions - positions - pending) / max_positions, 0.0, 1.0))



        )



        return {



            "spread_points": float(getattr(request, "spread_points", 0.0) or 0.0),



            "position_capacity_remaining": capacity_remaining,



            "pending_orders": float(pending),



            "allowed_action_fraction": float(len(allowed_actions) / max(1, ACTION_COUNT)),



            "day_trade_pressure": 0.0,



        }







    def _goal_context_for_request(self, request: BridgeRequest) -> np.ndarray:



        """Bounded objective constraints; deliberately excludes no-trade pressure."""



        positions = self._positions_from_pipe(str(getattr(request, "positions_pipe", "") or ""))



        max_age = max((float(item.get("age_bars", 0.0) or 0.0) for item in positions), default=0.0)



        capacity = (



            1.0



            if not position_limit_enabled(self.config.max_positions)



            else 1.0 - float(np.clip(



                float(getattr(request, "positions", 0)) / float(self.config.max_positions),



                0.0,



                1.0,



            ))



        )



        return np.asarray([



            float(np.clip(float(self.config.min_rr) / 5.0, 0.0, 1.0)),



            float(np.clip(1.0 - float(getattr(request, "drawdown_pct", 0.0) or 0.0) / 100.0, 0.0, 1.0)),



            float(np.clip(capacity, 0.0, 1.0)),



            float(np.clip(1.0 - max_age / max(1.0, float(self.config.max_holding_bars)), 0.0, 1.0)),



            (



                float(np.clip(1.0 - float(getattr(request, "day_trades", 0)) / float(self.config.max_daily_trades), 0.0, 1.0))



                if int(self.config.max_daily_trades) > 0 else 1.0



            ),



        ], dtype=float)







    @staticmethod



    def _force_action_output(outputs: np.ndarray, action: int) -> np.ndarray:



        """Return a vector whose serialized argmax unambiguously matches action."""



        adjusted = np.asarray(outputs, dtype=float).copy().reshape(-1)



        if adjusted.size <= action or action < 0:



            return adjusted



        action_count, _, _ = action_schema_for_outputs(adjusted)



        ceiling = float(np.max(adjusted[:action_count])) if action_count else 0.0



        adjusted[action] = ceiling + 1.0



        return adjusted







    @staticmethod



    def _normalize_pending_offset(outputs: np.ndarray, action: int) -> np.ndarray:



        """Compatibility normalizer for V7's magnitude-only pending offset.







        Direction is *never* encoded in the raw parameter. EA and offline



        execution derive it from the order action, so this function must not



        rewrite logits based on side. It only sanitizes non-finite output.



        """



        adjusted = np.asarray(outputs, dtype=float).copy().reshape(-1)



        adjusted[~np.isfinite(adjusted)] = 0.0



        return adjusted







    @staticmethod



    def _trace_recovery_context_vector(trace: object, request: BridgeRequest) -> np.ndarray:



        canonical = getattr(trace, "canonical_context_key", None)



        if canonical is not None:



            return np.asarray(canonical, dtype=float).reshape(-1)



        frame = getattr(trace, "context_frame", None)



        if frame is not None:



            return np.asarray(frame, dtype=float).reshape(-1)



        return np.asarray(getattr(request, "inputs", np.zeros(0, dtype=float)), dtype=float).reshape(-1)







    @staticmethod



    def _owner_action_recovery_learner(policy: object, trace: object):



        expert_from_trace = getattr(policy, "_expert_from_trace", None)



        if not callable(expert_from_trace):



            return None



        expert = expert_from_trace(trace)



        expert_policy = getattr(expert, "policy", None)



        return getattr(expert_policy, "action_recovery_learner", None)







    @staticmethod



    def _negative_action_cell_decision(learner: object, action: int, context_vector: np.ndarray) -> dict[str, object]:



        cells = getattr(learner, "cells", {}) or {}



        candidates = list(cells.get(int(action), []) or [])



        if not candidates:



            return {"blocked": False}



        vector = np.asarray(context_vector, dtype=float).reshape(-1)



        best = None



        best_distance = float("inf")



        for cell in candidates:



            prototype = np.asarray(getattr(cell, "prototype", np.zeros(0, dtype=float)), dtype=float).reshape(-1)



            width = min(vector.size, prototype.size)



            distance = float(np.linalg.norm(vector[:width] - prototype[:width])) if width else 0.0



            if distance < best_distance:



                best = cell



                best_distance = distance



        if best is None:



            return {"blocked": False}



        lcb = float(getattr(best, "lcb", 0.0) or 0.0)



        support = float(getattr(best, "support", 0.0) or 0.0)



        mean = float(getattr(best, "mean", 0.0) or 0.0)



        return {



            "blocked": bool(support >= 1.0 and lcb < 0.0),



            "support": support,



            "lcb": lcb,



            "mean": mean,



            "distance": best_distance,



        }







    def _directional_training_probe_action(



        self,



        details: Mapping[str, object] | None,



        allowed_actions: set[int],



    ) -> tuple[int | None, str]:



        """Return a direction-confirmed market probe, never an alternating one."""



        payload = dict(details or {})



        min_quality = float(getattr(self.config, "train_entry_probe_min_direction_quality", 0.75))



        signals: list[tuple[int, str]] = []







        timing_direction = int(payload.get("timing_direction", 0) or 0)



        timing_quality = float(payload.get("timing_quality", 0.0) or 0.0)



        if timing_direction in {-1, 1} and timing_quality >= min_quality:



            signals.append((timing_direction, "timing"))







        prior_direction = int(payload.get("context_direction_prior", 0) or 0)



        prior_quality = float(payload.get("context_direction_prior_quality", 0.0) or 0.0)



        prior_applied = bool(payload.get("context_direction_prior_applied", False))



        if prior_applied and prior_direction in {-1, 1} and prior_quality >= min_quality:



            signals.append((prior_direction, "context_prior"))







        if not signals:



            return None, "not_emitted_no_reliable_direction"



        directions = {direction for direction, _ in signals}



        if len(directions) != 1:



            return None, "not_emitted_direction_conflict"



        direction = next(iter(directions))



        action = int(Action.MARKET_BUY if direction > 0 else Action.MARKET_SELL)



        if action not in {int(item) for item in allowed_actions}:



            return None, "not_emitted_direction_action_illegal"



        sources = "+".join(source for _, source in signals)



        return action, f"directional_{'buy' if direction > 0 else 'sell'}:{sources}"







    def _bootstrap_execution_action(



        self,



        request: BridgeRequest,



        proposed_action: int,



        starvation: Mapping[str, object] | None,



        allowed_actions: set[int],



        *,



        policy_vetoed: bool = False,



        policy_vetoed_action: int | None = None,



        probe_details: Mapping[str, object] | None = None,



    ) -> tuple[int, bool]:



        """Apply an explicit train probe without overriding policy safety.







        A learner-recovery veto is final for the rejected action. Training



        exploration may collect evidence for a different entry action, but it



        must never reopen the exact action rejected from real losses. A probe



        also needs a reliable direction; alternating BUY/SELL pollutes the



        owner's local evidence and is intentionally unsupported.



        """



        del starvation



        self._train_entry_probe_last_source = ""



        pending_key = str(getattr(request, "orders_pipe", "") or "").strip()



        if pending_key:



            if pending_key != self._pending_cancel_key:



                self._pending_cancel_key = pending_key



                self._pending_cancel_age = 0



            self._pending_cancel_age += 1



            after = int(getattr(self.config, "pending_cancel_after_decisions", 8))



            interval = int(getattr(self.config, "pending_cancel_interval_decisions", 4))



            age = int(self._pending_cancel_age)



            should_cancel = age >= after and ((age - after) % max(1, interval) == 0)



            if should_cancel and int(Action.CANCEL_ORDERS) in {int(item) for item in allowed_actions}:



                return int(Action.CANCEL_ORDERS), True



        else:



            self._pending_cancel_key = ""



            self._pending_cancel_age = 0



        flat_no_orders = int(getattr(request, "positions", 0) or 0) == 0 and not pending_key



        if not flat_no_orders:



            self._train_entry_probe_flat_count = 0



            self._train_entry_probe_last_source = "not_applicable_position_or_order"



            return int(proposed_action), False



        if int(proposed_action) not in {int(Action.HOLD), NO_NEURAL_COMMAND}:



            self._train_entry_probe_flat_count = 0



            self._train_entry_probe_last_source = "not_applicable_policy_entry"



            return int(proposed_action), False



        vetoed_action = None



        if policy_vetoed_action is not None:



            try:



                vetoed_action = int(policy_vetoed_action)



            except (TypeError, ValueError):



                vetoed_action = None



        if bool(policy_vetoed) and vetoed_action is None:



            self._train_entry_probe_flat_count = 0



            self._train_entry_probe_last_source = "not_emitted_policy_veto_without_action"



            return int(proposed_action), False



        self._train_entry_probe_flat_count += 1



        if bool(getattr(self.config, "train_entry_probe_enabled", False)):



            request_day = str(getattr(request, "time", "") or "").strip()[:10]



            if request_day and request_day != self._train_entry_probe_day:



                self._train_entry_probe_day = request_day



                self._train_entry_probe_day_count = 0



                self._train_entry_probe_flat_count = 1



            max_probes = int(getattr(self.config, "train_entry_probe_max_per_run", 6) or 0)



            if max_probes > 0 and int(self._train_entry_probe_count) >= max_probes:



                return int(proposed_action), False



            max_daily_probes = int(getattr(self.config, "train_entry_probe_max_per_day", 2) or 0)



            if max_daily_probes > 0 and int(self._train_entry_probe_day_count) >= max_daily_probes:



                return int(proposed_action), False



            after = int(getattr(self.config, "train_entry_probe_after_decisions", 12))



            interval = int(getattr(self.config, "train_entry_probe_interval_decisions", 6))



            age = int(self._train_entry_probe_flat_count)



            should_probe = age >= after and ((age - after) % max(1, interval) == 0)



            entry_choices = [int(Action.MARKET_BUY), int(Action.MARKET_SELL)]



            legal_entries = [action for action in entry_choices if action in {int(item) for item in allowed_actions}]



            if should_probe and legal_entries:



                action, source = self._directional_training_probe_action(probe_details, set(legal_entries))



                self._train_entry_probe_last_source = str(source)



                if action is None:



                    return int(proposed_action), False



                if bool(policy_vetoed) and int(action) == int(vetoed_action):



                    self._train_entry_probe_last_source = "not_emitted_same_action_veto"



                    return int(proposed_action), False



                if bool(policy_vetoed):



                    self._train_entry_probe_last_source = f"{source}:alternative_to_veto_{int(vetoed_action)}"



                self._train_entry_probe_count += 1



                self._train_entry_probe_day_count += 1



                return int(action), True



        return int(proposed_action), False







    @staticmethod



    def _normalize_outputs_for_ea(outputs: np.ndarray) -> np.ndarray:



        """Normalize an action vector to the versioned 20-output contract."""



        values = np.asarray(outputs, dtype=float).reshape(-1)



        normalised = np.zeros(OUTPUT_COUNT, dtype=float)



        normalised[: min(values.size, OUTPUT_COUNT)] = values[:OUTPUT_COUNT]



        return normalised







    @staticmethod



    def _action_display_metadata(request: BridgeRequest, trace: object | None) -> dict[str, str]:



        if trace is None:



            return {



                "market_context": "",



                "expert_id": "",



                "policy_id": "",



                "expert_state": "",



                "context_reason": "",



                "no_trade_reason": "",



            }



        audit = dict(getattr(trace, "context_audit", {}) or {})







        def provenance(*keys: str) -> str:



            for key in keys:



                value = getattr(trace, key, None)



                if value is None or value == "":



                    value = audit.get(key, "")



                text = str(value or "")



                if text:



                    return text



            return ""







        return {



            "market_context": provenance("context_nhan_dien", "ma_context"),



            "expert_id": provenance("ma_chuyen_gia", "ma_policy"),



            "policy_id": provenance("ma_policy", "ma_chuyen_gia"),



            "expert_state": provenance("trang_thai_chuyen_gia"),



            "context_reason": provenance("ly_do_context"),



            "no_trade_reason": provenance("ly_do_khong_giao_dich"),



        }







    @staticmethod



    def _tsv_field(value: object) -> str:



        text = "" if value is None else str(value)



        return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()







    def _five_bridge_action_packet(



        self,



        policy: object,



        request: BridgeRequest,



        traces: Sequence[object],



        *,



        account_context: Mapping[str, float] | None = None,



    ) -> tuple[tuple[BridgeAgentAction, ...], tuple[object, ...]]:



        """Freeze five policy decisions into one atomic bridge packet."""



        if len(traces) != 5:



            raise ValueError("five-agent bridge requires exactly five traces")



        raw_probs = getattr(request, "regime_probabilities", None)



        if raw_probs is None:



            regime_idx, soft_probs, regime_name = self._predict_regime_for_request(request)



            request = replace(



                request,



                regime_index=regime_idx,



                regime_probabilities=tuple(float(p) for p in soft_probs),



                regime_name=regime_name,



            )



        else:



            try:



                raw_arr = np.asarray(raw_probs, dtype=float).ravel()



                if (



                    raw_arr.size >= NUM_REGIMES



                    and not np.isnan(raw_arr[:NUM_REGIMES]).any()



                    and not np.isinf(raw_arr[:NUM_REGIMES]).any()



                    and not (raw_arr[:NUM_REGIMES] < 0.0).any()



                ):



                    p_sum = float(np.sum(raw_arr[:NUM_REGIMES]))



                    if p_sum > 0:



                        soft_probs = (



                            raw_arr[:NUM_REGIMES]



                            if np.isclose(p_sum, 1.0, atol=1e-6)



                            else (raw_arr[:NUM_REGIMES] / p_sum)



                        )



                        regime_idx = int(request.regime_index if request.regime_index is not None else int(np.argmax(soft_probs)))



                        regime_name = str(request.regime_name or REGIME_NAMES.get(regime_idx, "CHOP"))



                    else:



                        regime_idx, soft_probs, regime_name = self._predict_regime_for_request(



                            replace(request, regime_probabilities=None)



                        )



                else:



                    regime_idx, soft_probs, regime_name = self._predict_regime_for_request(



                        replace(request, regime_probabilities=None)



                    )



            except Exception:



                regime_idx, soft_probs, regime_name = self._predict_regime_for_request(



                    replace(request, regime_probabilities=None)



                )



            request = replace(



                request,



                regime_index=regime_idx,



                regime_probabilities=tuple(float(p) for p in soft_probs),



                regime_name=regime_name,



            )



        if (



            soft_probs is None



            or len(soft_probs) < NUM_REGIMES



            or np.isnan(soft_probs[:NUM_REGIMES]).any()



            or np.isinf(soft_probs[:NUM_REGIMES]).any()



            or any(p < 0.0 for p in soft_probs[:NUM_REGIMES])



            or float(np.sum(soft_probs[:NUM_REGIMES])) <= 0



        ):



            soft_probs = np.full(NUM_REGIMES, 1.0 / NUM_REGIMES, dtype=np.float64)



            regime_idx = int(MarketRegime.CHOP)



            regime_name = "CHOP"



            request = replace(



                request,



                regime_index=regime_idx,



                regime_probabilities=tuple(float(p) for p in soft_probs),



                regime_name=regime_name,



            )



        if hasattr(policy, "set_regime_soft_probabilities"):



            try:



                policy.set_regime_soft_probabilities(



                    soft_probs, regime_name=regime_name, regime_idx=regime_idx



                )



            except Exception:



                pass



        packet: list[BridgeAgentAction] = []



        enriched_traces: list[object] = []



        for expected_agent, original_trace in enumerate(traces):



            authority = five_agent_authority(expected_agent)



            details = dict(getattr(original_trace, "action_details", {}) or {})



            observed_agent = int(details.get("five_agent_index", -1))



            if observed_agent != expected_agent:



                raise ValueError(



                    f"five-agent trace order mismatch: expected {expected_agent}, got {observed_agent}"



                )



            trace = original_trace



            outputs = np.asarray(getattr(trace, "selected_action", ()), dtype=float).reshape(-1)



            regime_veto_applied = bool(details.get("regime_veto_applied", False))



            regime_veto_reason = str(details.get("regime_veto_reason", "") or "")



            owner_valid, owner_reason = self._context_expert_owner_valid(policy, trace)



            if owner_valid:



                request_allowed = set(self._allowed_actions_for_request(request))



                trace_mask = tuple(getattr(trace, "allowed_action_mask", ()) or ())



                if trace_mask:



                    sampled_allowed = {



                        int(index)



                        for index, enabled in enumerate(trace_mask[:ACTION_COUNT])



                        if bool(enabled)



                    }



                else:



                    sampled_allowed = set(request_allowed)



                effective = (



                    request_allowed



                    & sampled_allowed



                    & set(authority.allowed_actions)



                )



                # Causal Regime Soft Veto: prevent counter-trend entries and force safe HOLD in chop



                if soft_probs is not None and len(soft_probs) >= 3:



                    p_bull_f = float(soft_probs[0])



                    p_bear_f = float(soft_probs[1])



                    p_chop_f = float(soft_probs[2])

                    bull_th = float(getattr(policy, 'bull_threshold', getattr(self.config, 'regime_bull_threshold', 0.35)))

                    bear_th = float(getattr(policy, 'bear_threshold', getattr(self.config, 'regime_bear_threshold', 0.35)))

                    chop_th = float(getattr(policy, 'chop_threshold', getattr(self.config, 'regime_chop_threshold', 0.38)))

                    reg_name = str(getattr(request, 'regime_name', '') or '').upper()



                    is_bull = (reg_name == 'BULL') or (p_bull_f >= bull_th and p_bull_f > p_bear_f)

                    is_bear = (reg_name == 'BEAR') or (p_bear_f >= bear_th and p_bear_f > p_bull_f)

                    is_chop = (reg_name == 'CHOP') or (p_chop_f >= chop_th) or (not is_bull and not is_bear)



                    short_entries = {int(Action.MARKET_SELL), int(Action.SELL_LIMIT), int(Action.SELL_STOP)}

                    long_entries = {int(Action.MARKET_BUY), int(Action.BUY_LIMIT), int(Action.BUY_STOP)}

                    all_entries = short_entries | long_entries



                    allow_all_regimes = True

                    if not allow_all_regimes and (is_bull and not is_bear):

                        if expected_agent == 1 and effective.intersection(short_entries):

                            effective.difference_update(short_entries)

                            regime_veto_applied = True

                            regime_veto_reason = f"veto_short_in_bull_p{p_bull_f:.2f}"

                    elif not allow_all_regimes and (is_bear and not is_bull):

                        if expected_agent == 0 and effective.intersection(long_entries):

                            effective.difference_update(long_entries)

                            regime_veto_applied = True

                            regime_veto_reason = f"veto_long_in_bear_p{p_bear_f:.2f}"

                    elif not allow_all_regimes and is_chop:

                        if expected_agent in (0, 1) and effective.intersection(all_entries):

                            effective.difference_update(all_entries)

                            regime_veto_applied = True

                            regime_veto_reason = f"veto_entries_in_chop_p{p_chop_f:.2f}"

                    # Tier 2: Microstructure Momentum Gate
                    if not allow_all_regimes:
                        raw_vec = getattr(request, "raw_inputs", getattr(request, "inputs", None))
                        if raw_vec is not None:
                            raw_arr_gate = np.asarray(raw_vec, dtype=float).ravel()
                            if raw_arr_gate.size >= 85:
                                m5_diff_val = float(raw_arr_gate[80])
                                m5_slope_val = float(raw_arr_gate[81])
                                if expected_agent == 1:
                                    sell_gate_entries = {int(Action.SELL_LIMIT), int(Action.MARKET_SELL)}
                                    if effective.intersection(sell_gate_entries) and (m5_diff_val > 0.0 and m5_slope_val > 0.0):
                                        effective.difference_update(sell_gate_entries)
                                        regime_veto_applied = True
                                        regime_veto_reason = "veto_sell_against_upward_momentum"
                                elif expected_agent == 0:
                                    buy_gate_entries = {int(Action.BUY_LIMIT), int(Action.MARKET_BUY)}
                                    if effective.intersection(buy_gate_entries) and (m5_diff_val < 0.0 and m5_slope_val < 0.0):
                                        effective.difference_update(buy_gate_entries)
                                        regime_veto_applied = True
                                        regime_veto_reason = "veto_buy_against_downward_momentum"

                    # Tier 3: Consecutive Loss Circuit Breaker
                    cb_inst = getattr(policy, "circuit_breaker", None)
                    if cb_inst is not None and getattr(cb_inst, "enabled", False):
                        req_time_val = getattr(request, "time", None)
                        if expected_agent == 0 and cb_inst.is_in_cooldown("BUY", req_time_val):
                            if effective.intersection(long_entries):
                                effective.difference_update(long_entries)
                                regime_veto_applied = True
                                regime_veto_reason = "circuit_breaker_consecutive_sl_cooldown"
                        elif expected_agent == 1 and cb_inst.is_in_cooldown("SELL", req_time_val):
                            if effective.intersection(short_entries):
                                effective.difference_update(short_entries)
                                regime_veto_applied = True
                                regime_veto_reason = "circuit_breaker_consecutive_sl_cooldown"



                if not effective:



                    effective = {int(Action.HOLD)}



                policy_action = int(details.get(



                    "five_action_index", getattr(trace, "selected_idx", int(Action.HOLD))



                ) or 0)



                wire_action = (



                    policy_action



                    if policy_action in effective



                    else int(Action.HOLD)



                )



                if expected_agent == 4 or int(wire_action) == int(Action.CLOSE_ALL):



                    wire_action = int(Action.HOLD)



            else:



                trace = self._gan_ly_do_chan_owner(trace, owner_reason)



                outputs = np.zeros(OUTPUT_COUNT, dtype=float)



                outputs[int(Action.HOLD)] = 1.0



                policy_action = NO_NEURAL_COMMAND



                wire_action = int(Action.HOLD)



                effective = {int(Action.HOLD)}



            wire_outputs = np.asarray(outputs, dtype=float).copy()



            if int(wire_action) != int(policy_action):



                wire_outputs = self._force_action_output(wire_outputs, int(wire_action))



            wire_outputs = self._normalize_pending_offset(wire_outputs, int(wire_action))



            score = float(outputs[wire_action]) if 0 <= int(wire_action) < outputs.size else 0.0



            bridge_details = self._decoded_action_details(wire_outputs, request.positions)



            bridge_details.update({



                "policy_action": int(policy_action),



                "legal_action": int(wire_action),



                "emitted_action": int(wire_action),



                "bridge_action": int(wire_action),



                "wire_action": int(wire_action),



                "wire_noop": bool(int(wire_action) == int(Action.HOLD)),



                "no_neural_command": bool(int(policy_action) < 0),



                "action_masked": bool(int(policy_action) != int(wire_action)),



                "execution_bootstrap": False,



                "training_entry_probe": False,



                "effective_allowed_actions": sorted(int(item) for item in effective),



                "context_contract_enforced_at_wire": True,



                "ea_receipt_enforced_safety": False,



                "five_agent_index": int(expected_agent),



                "five_agent_bridge_simultaneous": True,



                "five_agent_bridge_serialized": False,



                "five_agent_packet_size": 5,



                "five_agent_authority_version": FIVE_AGENT_AUTHORITY_VERSION,



                "five_agent_authority_role": authority.role,



                "five_agent_authority_actions": sorted(authority.allowed_actions),



                "five_agent_authority_responsibility": authority.responsibility,



                "five_agent_authority_denied": bool(



                    int(policy_action) not in authority.allowed_actions



                ),



                "action_origin": "mask" if int(policy_action) != int(wire_action) else "policy",



                "mt5_request_time": str(getattr(request, "time", "") or ""),



                "mt5_request_event_id": str(getattr(request, "event_id", "") or ""),



                "mt5_request_sequence": int(getattr(request, "sequence", 0) or 0),



                # Keep raw account marks outside the normalized policy feature



                # vector.  Codex uses them for decision-to-close attribution;



                # the policy still receives only ACCOUNT_CONTEXT_FIELDS.



                "codex_decision_account": {



                    "balance": float(getattr(request, "balance", 0.0) or 0.0),



                    "equity": float(getattr(request, "equity", 0.0) or 0.0),



                    "drawdown_pct": float(getattr(request, "drawdown_pct", 0.0) or 0.0),



                    "positions": float(getattr(request, "positions", 0) or 0),



                    "sequence": float(getattr(request, "sequence", 0) or 0),



                },



                "mt5_symbol": str(self.config.symbol),



                "mt5_period": str(self.config.period),



                "mt5_failure_replay_mode": bool(self.config.mt5_failure_replay_mode),



                "mt5_failure_replay_case_id": str(self.config.mt5_failure_replay_case_id or ""),



                "regime_index": int(regime_idx),



                "regime_name": str(regime_name),



                "regime_probabilities": [float(p) for p in soft_probs[:NUM_REGIMES]] if soft_probs is not None and len(soft_probs) >= NUM_REGIMES else [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],



                "regime_distribution": [float(p) for p in soft_probs[:NUM_REGIMES]] if soft_probs is not None and len(soft_probs) >= NUM_REGIMES else [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],



                "p_bull": float(soft_probs[0]) if soft_probs is not None and len(soft_probs) > 0 else (1.0 / 3.0),



                "p_bear": float(soft_probs[1]) if soft_probs is not None and len(soft_probs) > 1 else (1.0 / 3.0),



                "p_chop": float(soft_probs[2]) if soft_probs is not None and len(soft_probs) > 2 else (1.0 / 3.0),



                "regime_veto_applied": bool(regime_veto_applied),



                "regime_veto_reason": str(regime_veto_reason),



            })



            trace = enrich_step_trace(



                trace,



                bridge_details,



                request.positions_pipe,



                account_context=dict(account_context or {}),



            )



            action_risk_pct = None



            if trace is not None:



                trace_details = dict(getattr(trace, "action_details", {}) or {})



                if "dynamic_risk" in trace_details:



                    action_risk_pct = float(trace_details["dynamic_risk"])



                elif "risk_pct" in trace_details:



                    action_risk_pct = float(trace_details["risk_pct"])



                elif getattr(trace, "dynamic_risk", None) is not None:



                    action_risk_pct = float(trace.dynamic_risk)



                elif getattr(trace, "risk_pct", None) is not None:



                    action_risk_pct = float(trace.risk_pct)



            packet.append(BridgeAgentAction(



                agent_id=expected_agent,



                trace_id=str(getattr(trace, "trace_id", f"five-{request.sequence}-{expected_agent}")),



                action=int(wire_action),



                score=score,



                confidence=float(getattr(trace, "confidence", 0.0) or 0.0),



                outputs=wire_outputs,



                trace=trace,



                risk_pct=action_risk_pct,



            ))




            enriched_traces.append(trace)



        winner, blocked = resolve_opposite_entry_conflict(



            [int(item.action) for item in packet],



            [



                float(dict(getattr(item, "action_details", {}) or {}).get(



                    "five_selected_action_probability", getattr(item, "confidence", 0.0)



                ) or 0.0)



                for item in enriched_traces



            ],



        )



        if blocked is not None and winner is not None:



            blocked_row = packet[blocked]



            blocked_trace = enriched_traces[blocked]



            blocked_details = dict(getattr(blocked_trace, "action_details", {}) or {})



            blocked_outputs = self._force_action_output(



                np.asarray(blocked_row.outputs, dtype=float), HOLD_ACTION



            )



            blocked_details.update({



                "legal_action": HOLD_ACTION,



                "emitted_action": HOLD_ACTION,



                "bridge_action": HOLD_ACTION,



                "wire_action": HOLD_ACTION,



                "wire_noop": True,



                "action_masked": True,



                "action_origin": "opposite_entry_arbitration",



                "five_pre_wire_arbitration_blocked": True,



                "five_opposite_entry_conflict": True,



                "five_opposite_entry_winner_agent": int(winner),



                "five_opposite_entry_blocked_agent": int(blocked),



            })



            blocked_trace = enrich_step_trace(



                blocked_trace,



                blocked_details,



                request.positions_pipe,



                account_context=dict(account_context or {}),



            )



            enriched_traces[blocked] = blocked_trace



            packet[blocked] = replace(



                blocked_row,



                action=HOLD_ACTION,



                score=float(blocked_outputs[HOLD_ACTION]),



                outputs=blocked_outputs,



                trace=blocked_trace,



            )



            winner_details = dict(getattr(enriched_traces[winner], "action_details", {}) or {})



            winner_details.update({



                "five_opposite_entry_conflict": True,



                "five_opposite_entry_winner_agent": int(winner),



                "five_opposite_entry_blocked_agent": int(blocked),



            })



            enriched_traces[winner] = enrich_step_trace(



                enriched_traces[winner], winner_details, request.positions_pipe,



                account_context=dict(account_context or {}),



            )



            packet[winner] = replace(packet[winner], trace=enriched_traces[winner])



        return tuple(packet), tuple(enriched_traces)







    @staticmethod



    def _write_action(



        path: Path,



        request: BridgeRequest,



        trace_id: str,



        action: int,



        score: float,



        confidence: float,



        outputs: np.ndarray,



        *,



        risk_config: RiskConfig | None = None,



        order_config: OrderConfig | None = None,



        trace: object | None = None,



    ) -> None:



        MT5StrategyTesterEvaluator._write_action_batch(



            path,



            request,



            [BridgeAgentAction(



                agent_id=0,



                trace_id=str(trace_id),



                action=int(action),



                score=float(score),



                confidence=float(confidence),



                outputs=np.asarray(outputs, dtype=float),



                trace=trace,



            )],



            risk_config=risk_config,



            order_config=order_config,



        )







    @staticmethod



    def _write_action_batch(



        path: Path,



        request: BridgeRequest,



        actions: Sequence[BridgeAgentAction],



        *,



        risk_config: RiskConfig | None = None,



        order_config: OrderConfig | None = None,



    ) -> None:



        """Atomically write one legacy action or a complete five-agent packet."""



        rows = tuple(actions)



        if len(rows) not in {1, 5}:



            raise ValueError("bridge action packet must contain exactly 1 or 5 rows")



        agent_ids = [int(item.agent_id) for item in rows]



        expected_ids = list(range(len(rows)))



        if sorted(agent_ids) != expected_ids or len(set(agent_ids)) != len(rows):



            raise ValueError(f"bridge action packet agent ids must be {expected_ids}")



        path.parent.mkdir(parents=True, exist_ok=True)



        risk_config = risk_config or RiskConfig()



        order_config = order_config or OrderConfig()



        header = (



            "schema_version\tevent_id\ttrace_id\tsequence\tagent_id\tagent_count\t"



            "action\tscore\tconfidence\trisk_pct\toutputs_pipe\tmarket_context\t"



            "expert_id\tpolicy_id\texpert_state\tcontext_reason\tno_trade_reason"



        )



        body: list[str] = []



        for item in sorted(rows, key=lambda value: int(value.agent_id)):



            # Wire clamp Agent 4: strictly lock action to ACTION_HOLD (0), guaranteeing POLICY_CLOSE_ALL == 0



            if int(item.agent_id) == 4 and int(item.action) != int(Action.HOLD):



                item = replace(



                    item,



                    action=int(Action.HOLD),



                    outputs=MT5StrategyTesterEvaluator._force_action_output(



                        item.outputs, int(Action.HOLD)



                    ),



                )



            outputs = MT5StrategyTesterEvaluator._normalize_outputs_for_ea(item.outputs)



            # Check for dynamic risk_pct from CPPI engine attached to item or trace



            dynamic_risk: float | None = None



            if getattr(item, "risk_pct", None) is not None:



                dynamic_risk = float(item.risk_pct)



            elif item.trace is not None:



                trace_details = dict(getattr(item.trace, "action_details", {}) or {})



                if "dynamic_risk" in trace_details:



                    dynamic_risk = float(trace_details["dynamic_risk"])



                elif "risk_pct" in trace_details:



                    dynamic_risk = float(trace_details["risk_pct"])



                elif getattr(item.trace, "dynamic_risk", None) is not None:



                    dynamic_risk = float(item.trace.dynamic_risk)



                elif getattr(item.trace, "risk_pct", None) is not None:



                    dynamic_risk = float(item.trace.risk_pct)



            if dynamic_risk is not None and math.isfinite(dynamic_risk) and dynamic_risk > 0.0:



                risk_pct = float(dynamic_risk)



            else:



                try:



                    decision = decode_outputs(outputs, risk_config, order_config)



                    risk_pct = float(decision.risk_pct)



                except Exception:



                    risk_pct = float(risk_config.min_risk_pct)




            output_text = "|".join(f"{float(value):.10g}" for value in outputs)



            metadata = MT5StrategyTesterEvaluator._action_display_metadata(request, item.trace)



            fields = (



                ACTION_SCHEMA_VERSION,



                request.event_id,



                item.trace_id,



                request.sequence,



                item.agent_id,



                len(rows),



                item.action,



                f"{float(item.score):.10g}",



                f"{float(item.confidence):.10g}",



                f"{risk_pct:.10g}",



                output_text,



                metadata["market_context"],



                metadata["expert_id"],



                metadata["policy_id"],



                metadata["expert_state"],



                metadata["context_reason"],



                metadata["no_trade_reason"],



            )



            body.append("\t".join(MT5StrategyTesterEvaluator._tsv_field(value) for value in fields))



        content = header + "\n" + "\n".join(body) + "\n"



        tmp = path.with_suffix(path.suffix + ".tmp")



        tmp.write_text(content, encoding="utf-8")



        for _ in range(100):



            try:



                tmp.replace(path)



                return



            except PermissionError:



                time.sleep(0.01)



        tmp.replace(path)







    @staticmethod



    def _write_action_metadata(path: Path, request: BridgeRequest, trace: object | None) -> None:



        """Ghi provenance context/expert/policy ra sidecar JSONL.







        Bridge v10 carries the display-safe owner subset in every action row.



        The sidecar retains richer audit-only fields without widening the MQL



        parser again whenever Python observability gains a new attribute.



        """



        if trace is None:



            return



        audit = dict(getattr(trace, "context_audit", {}) or {})



        action_details = dict(getattr(trace, "action_details", {}) or {})



        def provenance(key: str) -> str:



            return str(getattr(trace, key, "") or audit.get(key, ""))



        payload = {



            "schema_version": ACTION_SCHEMA_VERSION,



            "event_id": str(getattr(request, "event_id", "")),



            "sequence": int(getattr(request, "sequence", 0)),



            "trace_id": str(getattr(trace, "trace_id", "")),



            "ma_context": provenance("ma_context"),



            "context_nhan_dien": provenance("context_nhan_dien"),



            "ma_chuyen_gia": provenance("ma_chuyen_gia"),



            "ma_policy": provenance("ma_policy"),



            "phien_ban_policy": provenance("phien_ban_policy"),



            "trang_thai_chuyen_gia": provenance("trang_thai_chuyen_gia"),



            "ly_do_context": provenance("ly_do_context"),



            "khong_co_policy_ra_lenh": bool(getattr(trace, "ly_do_khong_hoc", False) or audit.get("khong_co_policy_ra_lenh", False)),



            "ly_do_khong_giao_dich": str(getattr(trace, "ly_do_khong_giao_dich", "") or audit.get("ly_do_khong_giao_dich", "")),



        }



        for key in (



            "context_task",



            "context_allowed_actions",



            "context_action_contract",



            "trend_exit_guard_active",



            "trend_exit_guard_reason",



            "trend_exit_protected_tickets",



            "trend_exit_ready_tickets",



            "trend_exit_manage_available",



            "management_only",



            "management_owner_expert",



            "management_owner_context",



            "management_target_ticket",



            "management_target_owner_trace",



            "management_reason",



            "decision_policy_id",



            "decision_policy_context",



            "decision_role",



            "decision_policy_identity",



            "selector_selected_policy",



             "selector_reason",



             "selector_used_fallback",



             "selector_proposals",



             "selector_cold_start_probe_count",



             "selector_cold_start_probe_budget",



            "entry_proposal_evaluated",



            "entry_actions_available_in_selector",



            "entry_proposal_not_blocked",



            "correlated_exposure_supervisor_active",



            "correlated_exposure_stage",



            "correlated_exposure_direction",



            "correlated_exposure_count",



            "correlated_exposure_open_positions",



            "correlated_exposure_pending_traces",



            "correlated_exposure_mean_unrealized_r",



            "correlated_exposure_utility_context",



            "correlated_exposure_risk_decay",



            "correlated_exposure_original_risk_pct",



            "correlated_exposure_target_risk_pct",



            "correlated_exposure_risk_scaled",



            "correlated_exposure_entry_blocked",



            "timing_quality",



            "timing_direction",



            "timing_network_deceleration_probability",



            "timing_network_deceleration_update_count",



            "timing_network_trend_score",



            "timing_network_regime_transition_probability",



            "timing_network_regime_transition_probabilities",



            "timing_network_regime_transition_update_count",



            "timing_network_regime_transition_calibration_count",



            "timing_network_regime_transition_brier",



            "timing_network_regime_transition_high_confidence_support",



            "timing_network_regime_transition_high_confidence_precision",



            "candidate_context",



            "context_transition_active",



            "context_transition_progress",



            "selector_microstate",



            "selector_utility_context",



            "volatile_timing_direction_min_quality",



            "context_direction_prior_applied",



            "context_direction_prior",



            "context_direction_prior_score",



            "context_direction_prior_source",



            "context_direction_prior_quality",



            "owner_management_guard_active",



            "owner_management_guard_reason",



            "owner_management_best_unrealized_r",



            "owner_management_worst_unrealized_r",



            "owner_management_all_profitable",



            "owner_management_min_profit_r",



            "owner_management_emergency_r",



        ):



            if key in audit:



                payload[key] = audit[key]



        # Keep wire CSV stable while exposing the exact action provenance in



        # the sidecar. This distinguishes policy HOLD from transport NO_OP,



        # context masking, bootstrap and an expert-generated entry.



        for key in (



            "action_name",



            "policy_action",



            "legal_action",



            "emitted_action",



            "wire_action",



            "action_origin",



            "action_masked",



            "no_neural_command",



            "wire_noop",



            "effective_allowed_actions",



            "execution_bootstrap",



            "training_entry_probe",



            "training_entry_probe_source",



            "flat_idle_non_feedback",



            "flat_idle_starvation_observed",



            "flat_idle_no_trade_streak",



            "entry_timing_extrema_evidence_available",



            "entry_timing_action_override_applied",



            "entry_timing_original_action",



            "entry_timing_preferred_action",



            "entry_timing_memory_score",



            "timing_deceleration_response_enabled",



            "timing_deceleration_transition_active",



            "timing_deceleration_probability",



            "timing_deceleration_threshold",



            "timing_deceleration_update_count",



            "timing_deceleration_min_updates",



            "timing_deceleration_qualified",



            "timing_deceleration_action_before",



            "timing_deceleration_action_after",



            "timing_deceleration_entry_type_changed",



            "timing_deceleration_severity",



            "timing_deceleration_risk_multiplier",



            "timing_deceleration_original_risk_pct",



            "timing_deceleration_target_risk_pct",



            "timing_deceleration_risk_scaled",



            "timing_regime_transition_entry_block_enabled",



            "timing_regime_transition_probability",



            "timing_regime_transition_threshold",



            "timing_regime_transition_update_count",



            "timing_regime_transition_min_updates",



            "timing_regime_transition_calibration_count",



            "timing_regime_transition_min_calibration_samples",



            "timing_regime_transition_brier",



            "timing_regime_transition_max_brier",



            "timing_regime_transition_high_confidence_support",



            "timing_regime_transition_min_high_confidence_samples",



            "timing_regime_transition_high_confidence_precision",



            "timing_regime_transition_min_precision",



            "timing_regime_transition_calibration_ready",



            "timing_regime_transition_entry_blocked",



            "timing_regime_transition_action_before",



            "timing_regime_transition_action_after",



            "expectancy_action_delta_applied",



            "expectancy_action_delta_norm",



            "hold_reason",



            "learner_recovery_schema_version",



            "learner_recovery_selector_mode",



            "learner_recovery_negative_veto_applied",



            "learner_recovery_negative_veto_from",



            "learner_recovery_negative_veto_to_hold",



            "learner_recovery_negative_action_suppression",



            "learner_recovery_base_action",



            "learner_recovery_base_action_local_negative",



            "learner_recovery_base_available",



            "learner_recovery_base_similarity",



            "learner_recovery_base_support",



            "learner_recovery_base_real_support",



            "learner_recovery_base_shadow_support",



            "learner_recovery_base_mean_net_r",



            "learner_recovery_base_lcb",



            "learner_recovery_base_confidence",



            "learner_recovery_negative_support_threshold",



            "regime_index",



            "regime_name",



            "regime_probabilities",



            "regime_distribution",



            "p_bull",



            "p_bear",



            "p_chop",



            "regime_telemetry",



        ):



            if key in action_details:



                payload[key] = action_details[key]



        if getattr(request, "regime_probabilities", None) is not None:



            payload.setdefault("regime_probabilities", list(request.regime_probabilities))



            payload.setdefault("regime_distribution", list(request.regime_probabilities))



        if getattr(request, "regime_name", None) is not None:



            payload.setdefault("regime_name", str(request.regime_name))



        if getattr(request, "regime_index", None) is not None:



            payload.setdefault("regime_index", int(request.regime_index))



        if "regime_distribution" in payload and len(payload["regime_distribution"]) == 3:



            payload.setdefault("p_bull", float(payload["regime_distribution"][0]))



            payload.setdefault("p_bear", float(payload["regime_distribution"][1]))



            payload.setdefault("p_chop", float(payload["regime_distribution"][2]))



            payload.setdefault("regime_telemetry", {



                "last_probabilities": list(payload["regime_distribution"]),



                "regime_name": str(payload.get("regime_name", "CHOP")),



            })



        if not (any(payload.get(key) for key in ("ma_context", "ma_chuyen_gia", "ma_policy")) or "regime_distribution" in payload or "regime_telemetry" in payload):



            return



        metadata_path = path.with_name(path.name + ".metadata.jsonl")



        metadata_path.parent.mkdir(parents=True, exist_ok=True)



        with metadata_path.open("a", encoding="utf-8") as handle:



            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")







    @staticmethod



    def _context_expert_owner_valid(policy: object, trace: object | None) -> tuple[bool, str]:



        """Validate owner provenance without turning debounce into an entry gate.







        A raw neural trace or an old generic artifact must still fail closed.



        ``context_on_dinh`` is intentionally audit-only: candidate observations



        during debounce must reach the owning expert so the router and policy



        can learn from those transitions instead of silently losing entries.



        """



        if trace is None:



            return False, "THIEU_TRACE"



        if not hasattr(policy, "experts") or not hasattr(policy, "nhan_dien"):



            return False, "KHONG_PHAI_HE_CHUYEN_GIA"



        audit = dict(getattr(trace, "context_audit", {}) or {})



        context_id = str(getattr(trace, "ma_context", "") or audit.get("ma_context", ""))



        expert_id = str(getattr(trace, "ma_chuyen_gia", "") or audit.get("ma_chuyen_gia", ""))



        policy_id = str(getattr(trace, "ma_policy", "") or audit.get("ma_policy", ""))



        no_learning = bool(getattr(trace, "ly_do_khong_hoc", False) or audit.get("khong_co_policy_ra_lenh", False))



        if not context_id or not expert_id or not policy_id:



            return False, "THIEU_OWNER_CONTEXT_CHUYEN_GIA_POLICY"



        if no_learning:



            return False, "TRACE_KHONG_DUOC_RA_LENH"



        return True, ""







    @staticmethod



    def _effective_context_actions(trace: object | None, executable_actions: set[int]) -> set[int]:



        """Intersect MT5 executability with the immutable owner action contract."""



        executable = {int(item) for item in executable_actions if 0 <= int(item) < ACTION_COUNT}



        if trace is None:



            return executable or {int(Action.HOLD)}



        audit = dict(getattr(trace, "context_audit", {}) or {})



        if "context_allowed_actions" not in audit:



            return executable or {int(Action.HOLD)}



        raw = audit.get("context_allowed_actions", [])



        if isinstance(raw, Mapping):



            context = {



                int(key) for key, enabled in dict(raw).items()



                if bool(enabled) and str(key).lstrip("-").isdigit() and 0 <= int(key) < ACTION_COUNT



            }



        else:



            context: set[int] = set()



            for item in list(raw or []):



                try:



                    index = int(item)



                except (TypeError, ValueError):



                    continue



                if 0 <= index < ACTION_COUNT:



                    context.add(index)



        effective = executable & context



        return effective or {int(Action.HOLD)}







    @staticmethod



    def _gan_ly_do_chan_owner(trace: object, reason: str):



        audit = dict(getattr(trace, "context_audit", {}) or {})



        audit.update({



            "quyet_dinh_hop_le": False,



            "ly_do_chan_bridge": str(reason),



            "khong_co_policy_ra_lenh": True,



        })



        try:



            trace.context_audit = audit



            trace.ly_do_khong_hoc = True



            trace.ly_do_khong_giao_dich = str(reason)



        except Exception:



            pass



        return trace







    @staticmethod



    def _trace_learning_blocked(trace: object | None) -> bool:



        """True for transport/no-policy traces that must never become learning data."""



        if trace is None:



            return False



        audit = dict(getattr(trace, "context_audit", {}) or {})



        details = dict(getattr(trace, "action_details", {}) or {})



        if bool(getattr(trace, "ly_do_khong_hoc", False)):



            return True



        if bool(audit.get("khong_co_policy_ra_lenh", False)):



            return True



        if bool(details.get("no_neural_command", False)) or bool(details.get("wire_noop", False)):



            return True



        source = str(details.get("action_attribution_source", "") or "").lower()



        return source in {"none", "flat_idle", "hold_no_execution"}







    @staticmethod



    def _json_array(value) -> list[float]:



        return [float(item) for item in np.asarray(value, dtype=float).ravel().tolist()]







    def _risk_config(self) -> RiskConfig:



        return RiskConfig(



            initial_equity=float(self.config.initial_equity),



            min_risk_pct=float(self.config.min_risk_pct),



            max_risk_pct=float(self.config.max_risk_pct),



            max_open_risk_pct=float(self.config.max_open_risk_pct),



            ruin_equity_pct=float(self.config.ruin_equity_pct),



            fixed_risk_pct=0.0 if self.config.fixed_risk_pct is None else float(self.config.fixed_risk_pct),



            max_positions=int(self.config.max_positions),



        )







    def _order_config(self) -> OrderConfig:



        return OrderConfig(



            technical_min_rr=float(self.config.technical_min_rr),



            fixed_sl_atr=float(self.config.fixed_sl_atr),



            fixed_tp_atr=float(self.config.fixed_tp_atr),



            max_holding_bars=int(self.config.max_holding_bars),



            pending_expiry_bars=int(self.config.pending_expiry_bars),



            entry_offset_atr=float(self.config.entry_offset_atr),



            max_spread_cost_r=float(self.config.max_spread_cost_r),



        )







    def _decoded_action_details(self, outputs: np.ndarray, positions: int) -> dict[str, object]:



        try:



            decision = decode_outputs(



                outputs,



                self._risk_config(),



                self._order_config(),



                # EA target decoding spans the positions that actually exist;



                # it is independent of the optional entry quota.



                max_positions=max(1, int(positions)),



            )



        except Exception:



            return {}



        return {



            "action_name": Action(int(decision.action)).name,



            "target_slot": int(decision.target_slot),



            "management_mode": PositionManagementMode(int(decision.management_mode)).name,



            "partial_close_pct": float(decision.partial_close_pct),



            "trailing_stop_atr": float(decision.trailing_stop_atr),



            "sl_atr": float(decision.sl_atr),



            "tp_atr": float(decision.tp_atr),



        }







    def _compact_experience_journal(self, path: Path) -> None:



        """Keep only a complete JSONL tail once a journal exceeds its budget."""



        limit = max(0, int(getattr(self.config, "experience_journal_max_bytes", 64 * 1024 * 1024)))



        if limit <= 0 or not path.exists():



            return



        try:



            if path.stat().st_size <= limit:



                return



            with path.open("rb") as handle:



                handle.seek(0, os.SEEK_END)



                start = max(0, handle.tell() - limit)



                handle.seek(start)



                if start > 0:



                    handle.readline()  # discard the first partial JSON line



                retained = handle.read()



            tmp = path.with_suffix(path.suffix + ".compact.tmp")



            with tmp.open("wb") as handle:



                handle.write(retained)



            os.replace(tmp, path)



        except OSError:



            # Journal persistence is auxiliary; a write failure must not stop



            # a completed MT5 evaluation.



            return







    def _append_experience_journal(



        self,



        *,



        round_index: int,



        policy_id: int,



        request: BridgeRequest,



        trace,



        next_obs: np.ndarray,



        reward: float,



        cost: float,



        done: bool,



        signal: dict[str, float],



        action: int,



        score: float,



        confidence: float,



    ) -> None:



        if not self.config.experience_journal:



            return



        if self._trace_learning_blocked(trace):



            return



        execution_details = dict(getattr(trace, "execution_details", {}) or {})



        # MT5 bridge requests are action *intent* messages.  Do not persist an



        # unconfirmed request because replay would turn ignored/rejected actions



        # into fabricated episodic memory in later trials.



        if bool(execution_details.get("bridge_observed", False)) and not bool(execution_details.get("memory_eligible", False)):



            return



        path = Path(self.config.experience_journal)



        path.parent.mkdir(parents=True, exist_ok=True)



        self._compact_experience_journal(path)



        td_delta = float(signal.get("td_delta", 0.0))



        outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float)



        action_details = dict(getattr(trace, "action_details", {}) or self._decoded_action_details(outputs, request.positions))



        positions_pipe = str(getattr(trace, "positions_pipe", request.positions_pipe))



        positions_pipe_after = str(getattr(trace, "positions_pipe_after", ""))



        account_context = dict(getattr(trace, "account_context", {}) or {})



        account_context_after = dict(getattr(trace, "account_context_after", {}) or {})



        context_audit = dict(getattr(trace, "context_audit", {}) or {})



        item = {



            "schema_version": 3,



            "memory_key_schema": 6,



            "source": "mt5_online",



            "created_at": time.time(),



            "runner_dir": str(self.runner_dir),



            "round": int(round_index),



            "policy_id": int(policy_id),



            "event_id": str(request.event_id),



            "sequence": int(request.sequence),



            "trace_id": str(getattr(trace, "trace_id", "")),



            "context_owner": {



                "ma_context": str(context_audit.get("ma_context", getattr(trace, "ma_context", ""))),



                "ma_context_cha": str(context_audit.get("ma_context_cha", getattr(trace, "ma_context_cha", ""))),



                "ma_chuyen_gia": str(context_audit.get("ma_chuyen_gia", getattr(trace, "ma_chuyen_gia", ""))),



                "ma_policy": str(context_audit.get("ma_policy", getattr(trace, "ma_policy", ""))),



                "phien_ban_policy": str(context_audit.get("phien_ban_policy", getattr(trace, "phien_ban_policy", ""))),



                "context_on_dinh": bool(context_audit.get("context_on_dinh", False)),



                "ly_do_context": str(context_audit.get("ly_do_context", "")),



                "khong_co_policy_ra_lenh": bool(context_audit.get("khong_co_policy_ra_lenh", False)),



            },



            "action": int(action),



            "action_details": action_details,



            "positions_pipe": positions_pipe,



            "positions_pipe_after": positions_pipe_after,



            "execution_details": execution_details,



            "account_context": account_context,



            "account_context_after": account_context_after,



            "experience_starvation": dict(getattr(trace, "experience_starvation", {}) or {}),



            "score": float(score),



            "confidence": float(confidence),



            "reward": float(reward),



            "cost": float(cost),



            "td_delta": td_delta,



            "prediction_error": float(signal.get("prediction_error", 0.0)),



            "done": bool(done),



            "is_positive": bool(td_delta > 0.0 or reward > 0.0),



            "next_obs": self._json_array(next_obs),



            "trace": {



                "memory_key_schema": 6,



                "timestamp": float(getattr(trace, "timestamp", time.time())),



                "raw_obs": self._json_array(getattr(trace, "raw_obs", [])),



                "encoded_state": self._json_array(getattr(trace, "encoded_state", [])),



                "memory_before": self._json_array(getattr(trace, "memory_before", [])),



                "memory_after": self._json_array(getattr(trace, "memory_after", [])),



                "selected_idx": int(getattr(trace, "selected_idx", 0)),



                "selected_action": self._json_array(getattr(trace, "selected_action", [])),



                "candidate_actions": self._json_array(getattr(trace, "candidate_actions", [])),



                "candidate_logits": self._json_array(getattr(trace, "candidate_logits", [])),



                "candidate_scores": self._json_array(getattr(trace, "candidate_scores", [])),



                "allowed_action_mask": [



                    bool(value)



                    for value in tuple(getattr(trace, "allowed_action_mask", ()) or ())



                ],



                "selected_action_log_prob": float(getattr(trace, "selected_action_log_prob", 0.0)),



                "predicted_next_state": self._json_array(getattr(trace, "predicted_next_state", [])),



                "critic_value": float(getattr(trace, "critic_value", 0.0)),



                "confidence": float(getattr(trace, "confidence", confidence)),



                "novelty": float(getattr(trace, "novelty", 0.0)),



                "conflict": float(getattr(trace, "conflict", 0.0)),



                "risk": float(getattr(trace, "risk", 0.0)),



                "gate_values": self._json_array(getattr(trace, "gate_values", [])),



                "prediction_error": float(getattr(trace, "prediction_error", 0.0)),



                "td_delta": float(getattr(trace, "td_delta", 0.0)),



                "action_details": action_details,



                "positions_pipe": positions_pipe,



                "positions_pipe_after": str(getattr(trace, "positions_pipe_after", "")),



                "execution_details": execution_details,



                "account_context": account_context,



                "account_context_after": account_context_after,



                "experience_starvation": dict(getattr(trace, "experience_starvation", {}) or {}),



                "active_neuron_ids": [str(item) for item in tuple(getattr(trace, "active_neuron_ids", ()) or ())],



                "active_edge_ids": [str(item) for item in tuple(getattr(trace, "active_edge_ids", ()) or ())],



                "graph_edge_gates": {str(key): float(value) for key, value in dict(getattr(trace, "graph_edge_gates", {}) or {}).items()},



                "graph_region_activity": {str(key): float(value) for key, value in dict(getattr(trace, "graph_region_activity", {}) or {}).items()},



                "memory_match_quality": float(getattr(trace, "memory_match_quality", 0.0)),



                "lifecycle_tags": [str(item) for item in tuple(getattr(trace, "lifecycle_tags", ()) or ())],



                "module_need_reason": str(getattr(trace, "module_need_reason", "") or ""),



                "context_audit": context_audit,



                "module_action_traces": [



                    {



                        "module_name": str(getattr(item, "module_name", "")),



                        "gate": float(getattr(item, "gate", 0.0)),



                        "hidden": self._json_array(getattr(item, "hidden", [])),



                        "raw_output": self._json_array(getattr(item, "raw_output", [])),



                    }



                    for item in tuple(getattr(trace, "module_action_traces", ()) or ())



                ],



            },



        }



        with path.open("a", encoding="utf-8") as handle:



            # Action metadata may contain exact NumPy snapshots (for example



            # five_cell_before/after).  Journal persistence must preserve



            # their values without allowing a telemetry-only serialization



            # failure to abort the MT5 execution loop.



            serializable_item = self._observatory_value(item)



            handle.write(json.dumps(serializable_item, ensure_ascii=False, separators=(",", ":")) + "\n")



        # Enforce the bound after the write as well; otherwise a journal can



        # remain above its budget until a subsequent MT5 event arrives.



        self._compact_experience_journal(path)







    @staticmethod



    def _observatory_value(value):



        """Convert bridge metadata to JSON without allowing telemetry to fail MT5."""



        if isinstance(value, np.generic):



            return value.item()



        if isinstance(value, np.ndarray):



            return value.tolist()



        if isinstance(value, Mapping):



            return {str(key): MT5StrategyTesterEvaluator._observatory_value(item) for key, item in value.items()}



        if isinstance(value, (list, tuple, set)):



            return [MT5StrategyTesterEvaluator._observatory_value(item) for item in value]



        if isinstance(value, Path):



            return str(value)



        return value







    def _compact_execution_trace_observatory(self, path: Path) -> None:



        limit = max(0, int(getattr(self.config, "execution_trace_observatory_max_bytes", 64 * 1024 * 1024)))



        if limit <= 0 or not path.exists():



            return



        try:



            if path.stat().st_size <= limit:



                return



            # Retaining exactly ``limit`` bytes made the next append exceed the



            # limit again, rewriting the whole 64 MiB stream on every bridge



            # event. Keep 75% so compaction has enough hysteresis for many



            # subsequent records while preserving the newest complete JSONL.



            retain_bytes = max(1, (limit * 3) // 4)



            with path.open("rb") as handle:



                handle.seek(0, os.SEEK_END)



                start = max(0, handle.tell() - retain_bytes)



                handle.seek(start)



                if start > 0:



                    handle.readline()



                retained = handle.read()



            tmp = path.with_suffix(path.suffix + ".compact.tmp")



            with tmp.open("wb") as handle:



                handle.write(retained)



            os.replace(tmp, path)



        except OSError:



            return







    def _execution_trace_snapshot(self, trace) -> dict[str, object]:



        if trace is None:



            return {}



        action_details = dict(getattr(trace, "action_details", {}) or {})



        if not bool(getattr(



            self.config,



            "execution_trace_observatory_include_recurrent_state",



            True,



        )):



            action_details.pop("five_cell_before", None)



            action_details.pop("five_cell_after", None)



            action_details.pop("five_temporal_tokens_before", None)



            action_details.pop("five_temporal_valid_before", None)



        module_traces: list[dict[str, object]] = []



        for item in tuple(getattr(trace, "module_action_traces", ()) or ()):



            snapshot: dict[str, object] = {



                "module_name": str(getattr(item, "module_name", "") or ""),



                "gate": float(getattr(item, "gate", 0.0) or 0.0),



                "hidden": self._json_array(getattr(item, "hidden", [])),



                "raw_output": self._json_array(getattr(item, "raw_output", [])),



            }



            module_traces.append(snapshot)



        return {



            "trace_id": str(getattr(trace, "trace_id", "") or ""),



            "raw_obs": self._json_array(getattr(trace, "raw_obs", [])),



            "memory_before": self._json_array(getattr(trace, "memory_before", [])),



            "selected_idx": int(getattr(trace, "selected_idx", 0) or 0),



            "selected_action": self._json_array(getattr(trace, "selected_action", [])),



            "allowed_action_mask": [



                bool(value)



                for value in tuple(getattr(trace, "allowed_action_mask", ()) or ())



            ],



            "critic_value": float(getattr(trace, "critic_value", 0.0) or 0.0),



            "action_details": self._observatory_value(action_details),



            "execution_details": self._observatory_value(dict(getattr(trace, "execution_details", {}) or {})),



            "context_audit": self._observatory_value(dict(getattr(trace, "context_audit", {}) or {})),



            "module_action_traces": module_traces,



        }







    def _append_execution_trace_observatory(



        self,



        *,



        stage: str,



        round_index: int,



        policy_id: int,



        trace=None,



        trace_id: str = "",



        request: BridgeRequest | None = None,



        receipt: Mapping[str, object] | None = None,



        routing: Mapping[str, object] | None = None,



        lifecycle: object | None = None,



        credit: int | None = None,



        source: str = "",



    ) -> None:



        """Persist the causal MT5 audit chain, never a replay example."""



        raw_path = getattr(self.config, "execution_trace_observatory", None)



        if not raw_path:



            return



        try:



            path = Path(str(raw_path))



            path.parent.mkdir(parents=True, exist_ok=True)



            self._compact_execution_trace_observatory(path)



            effective_trace_id = str(trace_id or getattr(trace, "trace_id", "") or "")



            item: dict[str, object] = {



                "schema_version": 1,



                "source": "mt5_execution_trace_observatory",



                "created_at": time.time(),



                "stage": str(stage),



                "settlement_source": str(source),



                "runner_dir": str(self.runner_dir),



                "run_id": str(getattr(self.config, "run_id", "") or ""),



                "round": int(round_index),



                "policy_id": int(policy_id),



                "trace_id": effective_trace_id,



                "trace": self._execution_trace_snapshot(trace),



                "routing": self._observatory_value(dict(routing or {})),



                "receipt": self._observatory_value(dict(receipt or {})),



                "credit": None if credit is None else int(credit),



            }



            if request is not None:



                item["request"] = {



                    "event_id": str(getattr(request, "event_id", "") or ""),



                    "sequence": int(getattr(request, "sequence", -1) or -1),



                    "time": str(getattr(request, "time", "") or ""),



                    "positions": int(getattr(request, "positions", 0) or 0),



                }



            if lifecycle is not None:



                item["lifecycle"] = {



                    "position_ticket": str(getattr(lifecycle, "position_ticket", "") or ""),



                    "trace_id": str(getattr(lifecycle, "trace_id", "") or ""),



                    "realized_r": float(getattr(lifecycle, "realized_r", 0.0) or 0.0),



                    "close_reason": str(getattr(lifecycle, "close_reason", "") or ""),



                    "duration_bars": getattr(lifecycle, "duration_bars", None),



                    "final_close_confirmed": bool(getattr(lifecycle, "final_close_confirmed", False)),



                }



            with path.open("a", encoding="utf-8") as handle:



                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")



            self._compact_execution_trace_observatory(path)



        except (OSError, TypeError, ValueError):



            # The observability stream must not change execution or credit.



            return







    @staticmethod



    def _register_execution_intent_trace(policy, trace) -> dict[str, object]:



        hook = getattr(policy, "register_execution_intent_trace", None)



        if not callable(hook):



            return {"registered": False, "reason": "hook_missing"}



        try:



            result = hook(trace)



            return dict(result) if isinstance(result, Mapping) else {"registered": bool(result)}



        except Exception as exc:



            return {"registered": False, "reason": type(exc).__name__}







    @staticmethod



    def _register_entry_preference_trace(policy, trace) -> dict[str, object]:



        hook = getattr(policy, "register_entry_preference_trace", None)



        if not callable(hook):



            return {"registered": False, "reason": "hook_missing"}



        try:



            result = hook(trace)



            return dict(result) if isinstance(result, Mapping) else {"registered": bool(result)}



        except Exception as exc:



            return {"registered": False, "reason": type(exc).__name__}







    @staticmethod



    def _entry_preference_snapshot_eligible(



        action: int,



        *,



        include_hold: bool,



    ) -> bool:



        """Keep delayed labels on the entry surface, including optional HOLD."""



        try:



            selected = int(action)



        except (TypeError, ValueError):



            return False



        return 1 <= selected <= 6 or (bool(include_hold) and selected == 0)







    @staticmethod



    def _entry_preference_credit_backend(policy, *, is_five_pattern: bool) -> str:



        """Select the delayed-label backend without conflating Five channels.







        Public Five historically routed every ``five:`` row through the V4



        pattern/oracle hook. A candidate that disables that hook must opt in



        explicitly to the ordinary delayed selector hook; otherwise all



        mature rows are rejected as ``pattern_disabled``.



        """



        if not bool(is_five_pattern):



            return "entry_preference"



        if bool(getattr(policy, "five_entry_preference_supervision_enabled", False)):



            return "entry_preference"



        return "five_pattern_path"







    @staticmethod



    def _bind_execution_trace_receipt(policy, trace_id: str, receipt: Mapping[str, object]) -> dict[str, object]:



        hook = getattr(policy, "bind_execution_receipt", None)



        if not callable(hook):



            return {"bound": False, "reason": "hook_missing", "trace_id": str(trace_id or "")}



        try:



            result = hook(str(trace_id or ""), receipt)



            return dict(result) if isinstance(result, Mapping) else {"bound": bool(result), "trace_id": str(trace_id or "")}



        except Exception as exc:



            return {"bound": False, "reason": type(exc).__name__, "trace_id": str(trace_id or "")}







    def _apply_execution_lifecycle_credit(



        self,



        *,



        policy,



        lifecycle: object,



        round_index: int,



        policy_id: int,



        source: str,



        request: BridgeRequest | None = None,



    ) -> int:



        trace_id = str(getattr(lifecycle, "trace_id", "") or "")



        if not trace_id:



            return 0



        broker_close_reason = str(getattr(lifecycle, "close_reason", "") or "")



        exit_kind = str(getattr(lifecycle, "exit_kind", "") or "").strip().upper()



        # Fixed V13 geometry has one broker stop taxonomy: TP or INITIAL_SL.



        # Preserve the raw broker reason independently for reconciliation.



        canonical_close_reason = exit_kind or broker_close_reason



        metadata = dict(getattr(lifecycle, "metadata", {}) or {})



        receipt = {



            "receipt_stage": "lifecycle_final",



            "bridge_observed": True,



            "filled": True,



            "closed": True,



            "memory_eligible": True,



            "outcome_eligible": True,



            "final_close_confirmed": bool(getattr(lifecycle, "final_close_confirmed", False)),



            "close_reason": canonical_close_reason,



            "broker_close_reason": broker_close_reason,



            "exit_kind": exit_kind,



            "exit_controller": str(getattr(lifecycle, "exit_controller", "") or ""),



            "stop_controller": str(metadata.get("stop_controller", "") or ""),



            "stop_reason": str(metadata.get("stop_reason", "") or ""),



            "duration_bars": getattr(lifecycle, "duration_bars", None),



            "position_ticket": str(getattr(lifecycle, "position_ticket", "") or ""),



            "realized_r": float(getattr(lifecycle, "realized_r", 0.0) or 0.0),



            "entry_price": float(getattr(lifecycle, "entry_price", 0.0) or 0.0),



            "initial_sl_price": float(getattr(lifecycle, "initial_sl_price", 0.0) or 0.0),



            "initial_tp_price": float(getattr(lifecycle, "initial_tp_price", 0.0) or 0.0),



            "initial_risk_distance_price": str(metadata.get("initial_risk_distance_price", "") or ""),



            "final_sl_price": float(getattr(lifecycle, "final_sl_price", 0.0) or 0.0),



            "final_tp_price": float(getattr(lifecycle, "final_tp_price", 0.0) or 0.0),



            "geometry_mutation_count": int(getattr(lifecycle, "geometry_mutation_count", 0) or 0),



            "geometry_mode": str(metadata.get("geometry_mode", "") or ""),



            "fixed_sl_atr": float(metadata.get("fixed_sl_atr", 0.0) or 0.0),



            "fixed_tp_atr": float(metadata.get("fixed_tp_atr", 0.0) or 0.0),



            "fixed_rr": float(metadata.get("fixed_rr", 0.0) or 0.0),



        }



        if request is not None and getattr(request, "regime_probabilities", None) is not None:



            receipt["regime_distribution"] = [round(float(p), 6) for p in request.regime_probabilities]



            receipt["regime_name"] = str(getattr(request, "regime_name", "CHOP") or "CHOP")



            receipt["regime_index"] = int(getattr(request, "regime_index", 2) or 2)



            events_candidates = [self.runner_dir / "events.jsonl"]



            configured_events = getattr(self.config, "events_path", None)



            if configured_events:



                events_candidates.append(Path(configured_events))



            for ep in events_candidates:



                if ep.exists() or configured_events:



                    record_event_regime_distribution(



                        ep,



                        request.regime_probabilities,



                        str(getattr(request, "regime_name", "CHOP") or "CHOP"),



                        event_type="lifecycle",



                        sequence=int(getattr(request, "sequence", 0) or 0),



                        trace_id=trace_id,



                        extra_payload={"realized_r": float(getattr(lifecycle, "realized_r", 0.0) or 0.0)},



                    )



        routing = self._bind_execution_trace_receipt(policy, trace_id, receipt)



        hook = getattr(policy, "apply_execution_lifecycle_credit", None)



        if not callable(hook):



            self._append_execution_trace_observatory(



                stage="lifecycle_credit", round_index=round_index, policy_id=policy_id,



                trace_id=trace_id, request=request, receipt=receipt, routing=routing,



                lifecycle=lifecycle, credit=0, source=source,



            )



            return 0



        try:



            credited = int(hook(



                trace_id, float(getattr(lifecycle, "realized_r", 0.0) or 0.0),



                close_reason=canonical_close_reason,



                duration_bars=getattr(lifecycle, "duration_bars", None),



                min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



                lifecycle=lifecycle,



            ) or 0)



        except TypeError:



            credited = int(hook(



                trace_id, float(getattr(lifecycle, "realized_r", 0.0) or 0.0),



                close_reason=canonical_close_reason,



                duration_bars=getattr(lifecycle, "duration_bars", None),



                min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



            ) or 0)



        routing = dict(routing)



        routing["five_direct_rnn_rl"] = self._five_direct_rnn_routing_snapshot(policy)



        self._append_execution_trace_observatory(



            stage="lifecycle_credit", round_index=round_index, policy_id=policy_id,



            trace_id=trace_id, request=request, receipt=receipt, routing=routing,



            lifecycle=lifecycle, credit=credited, source=source,



        )



        self._maybe_capture_five_champion(policy=policy, lifecycle=lifecycle)



        return credited







    def _apply_execution_exit_credit(



        self,



        *,



        policy,



        lifecycle: MT5PositionLifecycle,



        round_index: int,



        policy_id: int,



        source: str,



        request: BridgeRequest | None = None,



    ) -> int:



        """Route only delayed POLICY_CLOSE evidence to the manager trace."""



        trace_id = str(getattr(lifecycle, "exit_trace_id", "") or "")



        exit_kind = str(getattr(lifecycle, "exit_kind", "") or "").upper()



        hold_r = getattr(lifecycle, "post_exit_hold_r", None)



        horizon = getattr(lifecycle, "post_exit_horizon_bars", None)



        if (



            not trace_id



            or not exit_kind.startswith("POLICY_CLOSE_")



            or hold_r is None



            or horizon is None



            or int(horizon) <= 0



        ):



            return 0



        receipt = {



            "receipt_stage": "exit_counterfactual_horizon",



            "bridge_observed": True,



            "position_ticket": str(getattr(lifecycle, "position_ticket", "") or ""),



            "exit_kind": exit_kind,



            "final_realized_r": float(getattr(lifecycle, "realized_r", 0.0) or 0.0),



            "hold_to_horizon_r": float(hold_r),



            "horizon_bars": int(horizon),



        }



        hook = getattr(policy, "apply_execution_exit_credit", None)



        if not callable(hook):



            return 0



        credited = int(hook(



            trace_id,



            final_realized_r=float(receipt["final_realized_r"]),



            hold_to_horizon_r=float(receipt["hold_to_horizon_r"]),



            position_ticket=str(receipt["position_ticket"]),



            exit_kind=exit_kind,



            horizon_bars=int(horizon),



        ) or 0)



        self._append_execution_trace_observatory(



            stage="exit_counterfactual_credit",



            round_index=round_index,



            policy_id=policy_id,



            trace_id=trace_id,



            request=request,



            receipt=receipt,



            lifecycle=lifecycle,



            credit=credited,



            source=source,



        )



        return credited







    def _prepare_mt5_rlhf_policy_phase(self, policy) -> tuple[object | None, dict[str, object]]:



        """Load a frozen scorer and install a frozen reference for one MT5 run."""







        enabled = bool(getattr(self.config, "rlhf_policy_update_enabled", False))



        manifest: dict[str, object] = {



            "enabled": enabled,



            "updates": 0,



            "skips": 0,



            "failures": 0,



            "promotion_allowed": False,



            "evidence_class": "rlhf_policy_candidate_only",



        }



        if not enabled:



            return None, manifest



        from .rlhf_policy_loop import RLHF_POLICY_FEATURE_SCHEMA



        from .rlhf_reward_model import load_rlhf_reward_model







        model_path = Path(str(self.config.rlhf_reward_model_path)).expanduser()



        reward_model = load_rlhf_reward_model(



            model_path,



            require_test_gate=True,



            expected_feedback_mode=str(



                getattr(self.config, "rlhf_reward_feedback_mode", "human")



            ),



        )



        feedback_mode = str(



            getattr(self.config, "rlhf_reward_feedback_mode", "human")



        ).strip().lower()



        require_independent_teacher = bool(



            getattr(self.config, "rlhf_require_independent_teacher", False)



        )



        configured_teacher_hash = str(



            getattr(self.config, "rlhf_teacher_checkpoint_sha256", "") or ""



        ).strip().lower()



        if feedback_mode == "rlaif":



            try:



                reward_report = json.loads(model_path.read_text(encoding="utf-8"))



            except (OSError, json.JSONDecodeError) as exc:



                raise ValueError(



                    "RLAIF reward model report cannot be read for teacher provenance"



                ) from exc



            teacher_provenance = (



                reward_report.get("teacher_provenance", {})



                if isinstance(reward_report, Mapping)



                else {}



            )



            teacher_hash = str(



                teacher_provenance.get("teacher_checkpoint_sha256", "")



                if isinstance(teacher_provenance, Mapping) else ""



            ).strip().lower()



            if configured_teacher_hash and teacher_hash != configured_teacher_hash:



                raise ValueError(



                    "RLAIF teacher checkpoint hash does not match reward-model provenance: "



                    f"configured={configured_teacher_hash or '<empty>'} report={teacher_hash or '<empty>'}"



                )



            source_hash = str(self.config.rlhf_source_checkpoint_sha256 or "").strip().lower()



            if require_independent_teacher:



                if not teacher_hash:



                    raise ValueError(



                        "independent RLAIF teacher mode requires teacher provenance in reward model"



                    )



                if teacher_hash == source_hash:



                    raise ValueError(



                        "independent RLAIF teacher must differ from the source champion"



                    )



        if reward_model.feature_schema != RLHF_POLICY_FEATURE_SCHEMA:



            raise ValueError(



                "MT5 RLHF reward model schema must be "



                + RLHF_POLICY_FEATURE_SCHEMA



            )



        installer = getattr(policy, "install_five_rlhf_reference", None)



        if not callable(installer):



            raise RuntimeError("MT5 RLHF policy phase requires the public Five reference hook")



        reference = installer()



        if not isinstance(reference, Mapping) or not bool(reference.get("installed", False)):



            reason = reference.get("reason", "unknown") if isinstance(reference, Mapping) else "invalid_result"



            raise RuntimeError(



                "MT5 RLHF policy phase could not install a frozen reference: " + str(reason)



            )



        manifest.update({



            "reward_model_path": str(model_path),



            "reward_model_sha256": str(reward_model.sha256),



            "feedback_mode": str(



                getattr(self.config, "rlhf_reward_feedback_mode", "human")



            ).strip().lower(),



            "teacher_checkpoint_sha256": configured_teacher_hash,



            "independent_teacher_required": require_independent_teacher,



            "feature_schema": str(reward_model.feature_schema),



            "reference_policy_sha256": str(reference.get("reference_policy_sha256", "") or ""),



            "reference_policy_frozen": bool(reference.get("reference_policy_frozen", False)),



            "source_checkpoint_sha256": str(self.config.rlhf_source_checkpoint_sha256),



        })



        if not manifest["reference_policy_frozen"] or not str(manifest["reference_policy_sha256"]):



            raise RuntimeError("MT5 RLHF policy phase received an incomplete frozen reference receipt")



        return reward_model, manifest







    def _serve_policy_actions(



        self,



        process: subprocess.Popen,



        report_path: Path,



        policy,



        policy_id: int,



        round_index: int,



        *,



        replay_requests: Iterable[BridgeRequest] | None = None,



        replay_action_observer: Callable[[BridgeRequest, Path], None] | None = None,



    ) -> int:



        report_name = report_path.stem



        replay_iterator: Iterator[BridgeRequest] | None = (



            iter(replay_requests) if replay_requests is not None else None



        )



        learning_enabled = bool(getattr(self.config, "learning_enabled", True))



        reset = getattr(policy, "reset_runtime_state", None)



        if callable(reset):



            reset()



        self._reset_five_competition_state()



        # These counters belong to one MT5 tester evaluation. Reset them



        # before loading the phase so a later candidate cannot inherit update



        # counts from an earlier round.



        self._last_rlhf_policy_updates = 0



        self._last_rlhf_policy_update_skips = 0



        self._last_rlhf_policy_update_failures = 0



        self._active_rlhf_policy_update_manifest = None



        rlhf_reward_model, rlhf_manifest = self._prepare_mt5_rlhf_policy_phase(policy)



        # Keep the same mutable receipt object while the bridge records update



        # and skip counters. The evaluator catch path snapshots it when a



        # wall-time guard interrupts the run.



        self._active_rlhf_policy_update_manifest = rlhf_manifest



        rlhf_update_interval = max(



            1, int(getattr(self.config, "rlhf_update_interval_packets", 1) or 1)



        )



        rlhf_max_updates = max(



            0, int(getattr(self.config, "rlhf_max_updates_per_run", 0) or 0)



        )



        reset_five_reward = getattr(policy, "reset_five_economic_reward", None)



        if callable(reset_five_reward):



            reset_five_reward()



        if learning_enabled and hasattr(policy, "experience_buffer"):



            policy.experience_buffer = []



        # These counters are only observability.  They are reset for every



        # tester process and are never read by the policy/optimizer.  Keeping



        # them on the evaluator preserves the public ``int`` return contract



        # of ``_serve_policy_actions`` for older callers.



        self._last_live_lifecycle_settlements = 0



        self._last_live_lifecycle_updates = 0



        self._last_wall_timeout_lifecycle_updates = 0



        self._position_context_extrema = {}



        last_request_day = ""



        # Frozen OOS has no event-driven learner, journal, replay or lifecycle



        # credit path. It only serializes actions and reads execution evidence.



        if not (hasattr(policy, "experts") and hasattr(policy, "registry") and hasattr(policy, "nhan_dien")):



            raise TypeError("_serve_policy_actions chi nhan HeChuyenGiaMotContextMotPolicy")



        served = 0



        last_sequence = -1



        pending_trace = None



        pending_equity: float | None = None



        pending_drawdown: float = 0.0



        pending_spread_points: float = 0.0



        pending_request: BridgeRequest | None = None



        pending_action: int = NO_NEURAL_COMMAND



        pending_wire_action: int = int(Action.HOLD)



        pending_score: float = 0.0



        pending_confidence: float = 0.0



        pending_five_actions: tuple[dict[str, object], ...] = ()



        starvation_state = ExperienceStarvationState()



        pending_starvation: dict[str, float] = {}



        last_inputs = np.zeros(0, dtype=float)



        equity_history: list[float] = []



        legacy_open_traces: dict[int, dict[str, object]] = {}



        legacy_settled_sequences: set[int] = set()



        canonical_lifecycle_settled: set[str] = set()



        exit_counterfactual_settled: set[str] = set()



        entry_preference_settled: set[str] = set()



        entry_preference_rejected: set[str] = set()



        entry_preference_pending: dict[str, EntryPreferenceRecord] = {}



        entry_preference_attempts: dict[str, int] = {}



        entry_preference_reader = _IncrementalCsvTail()



        self._last_live_entry_preference_records = 0



        self._last_live_entry_preference_updates = 0



        self._last_live_entry_preference_accepted = 0



        self._last_live_entry_preference_rejected = 0



        self._last_live_entry_preference_deferred = 0



        self._last_live_entry_preference_pending = 0







        def apply_mt5_rlhf_update(



            packet: Sequence[object],



            request: BridgeRequest,



        ) -> dict[str, object]:



            """Apply one bounded frozen-reward update before the next decision."""







            if rlhf_reward_model is None:



                return {}



            packet_index = int(served)



            if packet_index <= 0 or packet_index % rlhf_update_interval != 0:



                self._last_rlhf_policy_update_skips += 1



                rlhf_manifest["skips"] = int(rlhf_manifest.get("skips", 0) or 0) + 1



                return {



                    "rlhf_policy_update": False,



                    "reason": "update_interval",



                    "packet_index": packet_index,



                    "promotion_allowed": False,



                }



            if rlhf_max_updates and self._last_rlhf_policy_updates >= rlhf_max_updates:



                self._last_rlhf_policy_update_skips += 1



                rlhf_manifest["skips"] = int(rlhf_manifest.get("skips", 0) or 0) + 1



                return {



                    "rlhf_policy_update": False,



                    "reason": "max_updates_per_run",



                    "packet_index": packet_index,



                    "promotion_allowed": False,



                }



            updater = getattr(policy, "optimize_five_rlhf_trace_packet", None)



            if not callable(updater):



                raise RuntimeError("MT5 RLHF policy phase requires the Five update hook")



            try:



                result = updater(



                    tuple(packet),



                    rlhf_reward_model,



                    source_checkpoint_sha256=str(self.config.rlhf_source_checkpoint_sha256),



                    reference_policy_sha256=str(rlhf_manifest.get("reference_policy_sha256", "") or ""),



                )



            except ValueError as exc:



                # A management-only packet is valid for the 12-action Five



                # runtime but has no row in RLHF V1's seven-entry schema. Do



                # not convert that expected coverage gap into a failed MT5 run.



                if "no seven-action entry trace" not in str(exc):



                    self._last_rlhf_policy_update_failures += 1



                    rlhf_manifest["failures"] = int(rlhf_manifest.get("failures", 0) or 0) + 1



                    raise



                self._last_rlhf_policy_update_skips += 1



                rlhf_manifest["skips"] = int(rlhf_manifest.get("skips", 0) or 0) + 1



                return {



                    "rlhf_policy_update": False,



                    "reason": "no_entry_surface_trace",



                    "packet_index": packet_index,



                    "promotion_allowed": False,



                }



            if not isinstance(result, Mapping) or not bool(result.get("rlhf_ppo_update", False)):



                self._last_rlhf_policy_update_failures += 1



                rlhf_manifest["failures"] = int(rlhf_manifest.get("failures", 0) or 0) + 1



                raise RuntimeError("MT5 RLHF update did not return a verified PPO receipt")



            self._last_rlhf_policy_updates += 1



            rlhf_manifest["updates"] = int(rlhf_manifest.get("updates", 0) or 0) + 1



            self._append_execution_trace_observatory(



                stage="rlhf_policy_update",



                round_index=round_index,



                policy_id=policy_id,



                trace=packet[0] if packet else None,



                request=request,



                receipt=dict(result),



                routing={



                    "reward_model_sha256": str(rlhf_manifest.get("reward_model_sha256", "") or ""),



                    "reference_policy_sha256": str(rlhf_manifest.get("reference_policy_sha256", "") or ""),



                    "source_checkpoint_sha256": str(self.config.rlhf_source_checkpoint_sha256),



                },



                source="mt5_rlhf_candidate_ppo",



            )



            return dict(result)







        run_started = time.monotonic()



        last_request_seen = run_started



        # Với phiên trực quan, người dùng có thể đóng terminal nhưng tiến



        # trình launcher vẫn còn sống và tester mồ côi vẫn ghi dữ liệu cầu.



        # Nếu không phát hiện việc terminal biến mất, max_wall_sec=0 sẽ làm



        # vòng cầu chạy vô hạn. Chỉ thăm dò định kỳ để tránh gọi PowerShell



        # ở mỗi vòng lặp 50 ms.



        visual_terminal_missing_since: float | None = None



        visual_terminal_probe_at = run_started



        visual_terminal_missing_grace_sec = 5.0



        stop_request_path = Path(str(self.config.stop_request_file or "")).expanduser()



        stop_marker_name = str(self.config.stop_request_marker_file or "").strip()



        stop_request_published = False







        def publish_save_and_stop_request() -> None:



            """Mirror the UI request into MT5's sandbox exactly once."""



            nonlocal stop_request_published



            if stop_request_published or not str(self.config.stop_request_file or "").strip():



                return



            try:



                requested = stop_request_path.is_file()



            except OSError:



                requested = False



            if not requested:



                return



            if not stop_marker_name:



                raise RuntimeError("save-and-stop request requires a non-empty MT5 marker file")



            marker_payload = (



                f"save-and-stop requested at {datetime.now().isoformat()}\n"



                f"source={stop_request_path}\n"



            )



            bases = [self.common_files, self.mql5_files]



            # Strategy Tester runs the EA inside an Agent-* sandbox.  The



            # runner/common paths are enough for portable terminal mode, but



            # a visual tester may resolve a plain FileIsExist() relative to



            # the agent's MQL5/Files directory.  Mirror the cooperative



            # marker there too, otherwise Save/Stop can appear accepted by



            # the bridge while the EA never sees the request.



            tester_dir = self.runner_dir / "Tester"



            if tester_dir.exists():



                bases.extend(tester_dir.glob("Agent-*/MQL5/Files"))



            for base in bases:



                base.mkdir(parents=True, exist_ok=True)



                _atomic_write_text(base / stop_marker_name, marker_payload)



            stop_request_published = True



            self._last_stop_request = {



                "requested": True,



                "path": str(stop_request_path),



                "marker_file": stop_marker_name,



                "published_at": datetime.now().isoformat(),



                "boundary": "next_ea_decision_boundary",



            }



            print(



                f"[Bridge] save-and-stop published marker={stop_marker_name} "



                f"source={stop_request_path}; waiting for authoritative MT5 report",



                flush=True,



            )



        # Real-tick tester startup may spend several minutes synchronizing



        # tick/M1 history before the EA can emit its first bridge request. The



        # normal 60s stale-terminal guard remains strict for generated-tick



        # modes, while model 4 gets a bounded startup grace instead of an



        # unbounded wait for a broken terminal.



        real_tick_startup_grace_sec = (



            max(300.0, float(self.config.timeout_sec))



            if (



                int(getattr(self.config, "model", -1)) == MT5_MODEL_REAL_TICKS



                or bool(getattr(self.config, "offline_preflight", False))



            )



            else 60.0



        )



        initial_bridge_timeout_sec = (



            real_tick_startup_grace_sec



            if (



                int(getattr(self.config, "model", -1)) == MT5_MODEL_REAL_TICKS



                or bool(getattr(self.config, "offline_preflight", False))



            )



            else float(self.config.timeout_sec)



        )



        startup_failure_probe_at = run_started



        startup_failure_grace_sec = max(



            5.0,



            float(getattr(self.config, "mt5_startup_failure_grace_sec", 20.0) or 20.0),



        )



        debug_counter = 0



        guard_day = ""



        guard_closed_count = 0



        guard_closed_net_r = 0.0



        guard_consecutive_losses = 0



        runtime_guard = RuntimeFitnessGuard(



            enabled=bool(getattr(self.config, "runtime_fitness_guard_enabled", True)),



            rolling_window=int(getattr(self.config, "runtime_fitness_rolling_window", 50) or 50),



            min_closed=int(getattr(self.config, "runtime_fitness_min_closed", 20) or 20),



            min_score=float(getattr(self.config, "runtime_fitness_min_score", -0.30)),



            max_immediate_exit_ratio=float(getattr(self.config, "runtime_fitness_max_immediate_exit_ratio", 0.60)),



            max_daily_loss_r=float(getattr(self.config, "runtime_fitness_max_daily_loss_r", 3.0)),



            max_run_loss_r=float(getattr(self.config, "runtime_fitness_max_run_loss_r", 8.0)),



            max_daily_trades=int(getattr(self.config, "runtime_fitness_max_daily_trades", 300) or 0),



            max_drawdown_pct=float(getattr(self.config, "runtime_fitness_max_drawdown_pct", 20.0)),



            initial_equity=float(getattr(self.config, "initial_equity", 10000.0)),



            min_holding_bars=int(getattr(self.config, "min_manual_close_age_bars", 2) or 2),



        )



        def request_day(request: BridgeRequest) -> str:



            return str(getattr(request, "time", "") or "").split(" ", 1)[0]







        def reset_entry_loss_guard_for_day(request: BridgeRequest) -> None:



            nonlocal guard_day, guard_closed_count, guard_closed_net_r, guard_consecutive_losses



            current_day = request_day(request)



            if current_day and current_day != guard_day:



                guard_day = current_day



                guard_closed_count = 0



                guard_closed_net_r = 0.0



                guard_consecutive_losses = 0







        def record_entry_loss_guard_close(reward: float) -> None:



            nonlocal guard_closed_count, guard_closed_net_r, guard_consecutive_losses



            guard_closed_count += 1



            guard_closed_net_r += float(reward)



            if float(reward) < 0.0:



                guard_consecutive_losses += 1



            elif float(reward) > 0.0:



                guard_consecutive_losses = 0







        def entry_loss_guard_active() -> bool:



            return self._entry_loss_guard_blocks_entries(



                closed_count=guard_closed_count,



                closed_net_r=guard_closed_net_r,



                consecutive_losses=guard_consecutive_losses,



            )







        def record_runtime_fitness_guard_rows(request: BridgeRequest) -> None:



            fallback_day = request_day(request)



            for record in self._parse_execution_records(getattr(request, "execution_pipe", "")):



                runtime_guard.record_execution_row(



                    record,



                    fallback_day=fallback_day,



                    drawdown_pct=float(getattr(request, "drawdown_pct", 0.0) or 0.0),



                )







        def complete_final_close_receipt(details: dict[str, object]) -> dict[str, object]:



            """Resolve a bridge close ack against the authoritative deal row."""



            try:



                if not bool(details.get("closed", False)):



                    return details



                if abs(float(details.get("initial_risk", 0.0) or 0.0)) > 1e-12:



                    return details



            except (TypeError, ValueError):



                pass



            execution_path = self._latest_runtime_file(self.config.execution_file)



            if execution_path is None or not execution_path.exists():



                return details



            try:



                with execution_path.open("r", newline="", encoding="utf-8-sig") as handle:



                    rows = list(csv.DictReader(handle, delimiter="\t"))



            except OSError:



                return details



            return _merge_final_close_receipt(details, rows)







        # A bounded training chunk can hit the wall-clock guard while the



        # tester is still alive.  In that case the normal post-run evaluator



        # path is never reached, so final lifecycles would otherwise remain



        # only in the feedback stream and never reach the owner policy.  Settle



        # the closed rows that are already durable in the execution ledger



        # before raising the timeout; open positions are deliberately left



        # untouched and will be reconciled by a later complete run.



        timeout_lifecycle_settled: set[str] = set()







        def settle_timeout_lifecycle_updates() -> int:



            if not learning_enabled:



                return 0



            execution_path = self._latest_runtime_file(self.config.execution_file)



            summary = parse_mt5_execution_ledger(



                execution_path,



                initial_equity=float(self.config.deposit),



                min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



                technical_min_rr=float(self.config.technical_min_rr),



            )



            credits = 0



            for lifecycle in summary.lifecycles:



                trace_id = str(getattr(lifecycle, "trace_id", "") or "")



                if not trace_id or trace_id in timeout_lifecycle_settled:



                    continue



                timeout_lifecycle_settled.add(trace_id)



                credited = self._apply_execution_lifecycle_credit(



                    policy=policy,



                    lifecycle=lifecycle,



                    round_index=round_index,



                    policy_id=policy_id,



                    source="wall_timeout",



                )



                credits += credited



                self._last_wall_timeout_lifecycle_updates += int(credited)



            return credits







        def settle_entry_preference_updates(request: BridgeRequest | None = None) -> int:



            """Consume mature EA counterfactual rows without touching execution.







            The CSV contains only decisions that already observed their full



            fixed M1 horizon.  Deduplication makes repeated bridge polling and



            tester retries idempotent.



            """



            if not learning_enabled:



                return 0



            preference_path = self._latest_runtime_file(self.config.entry_preference_file)



            if preference_path is None or not preference_path.exists():



                return 0



            rows = entry_preference_reader.read_rows(preference_path)



            credited_total = 0



            five_pattern_hook = getattr(policy, "apply_five_pattern_path_credit_detailed", None)



            five_selector_hook = getattr(policy, "apply_five_entry_preference_credit_detailed", None)



            detailed_hook = getattr(policy, "apply_entry_preference_credit_detailed", None)



            legacy_hook = getattr(policy, "apply_entry_preference_credit", None)



            if (



                not callable(five_pattern_hook)



                and not callable(five_selector_hook)



                and not callable(detailed_hook)



                and not callable(legacy_hook)



            ):



                return 0



            for row in rows:



                record = parse_entry_preference_row(row)



                if not isinstance(record, EntryPreferenceRecord):



                    continue



                identity = f"{record.trace_id}|{record.sequence}"



                if (



                    identity in entry_preference_settled



                    or identity in entry_preference_rejected



                    or identity in entry_preference_pending



                ):



                    continue



                entry_preference_pending[identity] = record



                self._last_live_entry_preference_records += 1



            for identity, record in list(entry_preference_pending.items()):



                attempt = int(entry_preference_attempts.get(identity, 0)) + 1



                entry_preference_attempts[identity] = attempt



                is_five_pattern = str(record.trace_id).startswith("five:")



                credit_backend = self._entry_preference_credit_backend(



                    policy, is_five_pattern=is_five_pattern



                )



                try:



                    if credit_backend == "five_pattern_path":



                        if not callable(five_pattern_hook):



                            raw_result = {



                                "accepted": False, "queued": False, "updates": 0,



                                "retryable": False, "reason": "five_pattern_hook_missing",



                            }



                        else:



                            raw_result = five_pattern_hook(record.trace_id, record)



                        result = dict(raw_result) if isinstance(raw_result, Mapping) else {



                            "accepted": True, "queued": True,



                            "updates": int(raw_result or 0), "retryable": False,



                            "reason": "five_pattern_hook_completed",



                        }



                    elif credit_backend == "entry_preference" and is_five_pattern and callable(five_selector_hook):



                        raw_result = five_selector_hook(record.trace_id, record)



                        result = dict(raw_result) if isinstance(raw_result, Mapping) else {



                            "accepted": True, "queued": True,



                            "updates": int(raw_result or 0), "retryable": False,



                            "reason": "five_selector_hook_completed",



                        }



                    elif callable(detailed_hook):



                        raw_result = detailed_hook(record.trace_id, record)



                        result = dict(raw_result) if isinstance(raw_result, Mapping) else {



                            "accepted": True, "queued": True,



                            "updates": int(raw_result or 0), "retryable": False,



                            "reason": "detailed_hook_completed",



                        }



                    else:



                        updates = int(legacy_hook(record.trace_id, record) or 0)



                        result = {



                            "accepted": True, "queued": True,



                            "updates": updates, "retryable": False,



                            "reason": "legacy_hook_completed",



                        }



                except Exception as exc:



                    result = {



                        "accepted": False, "queued": False, "updates": 0,



                        "retryable": True, "reason": type(exc).__name__,



                    }



                accepted = bool(result.get("accepted", False))



                retryable = bool(result.get("retryable", not accepted))



                updates = int(result.get("updates", 0) or 0)



                routing = {



                    **result,



                    "identity": identity,



                    "attempt": int(attempt),



                    "selected_action": int(record.selected_action),



                    "sample_scope": str(record.sample_scope),



                }



                stage_prefix = credit_backend



                routing["credit_backend"] = stage_prefix



                if accepted:



                    entry_preference_settled.add(identity)



                    entry_preference_pending.pop(identity, None)



                    entry_preference_attempts.pop(identity, None)



                    self._last_live_entry_preference_accepted += 1



                    credited_total += updates



                    self._last_live_entry_preference_updates += updates



                    self._append_execution_trace_observatory(



                        stage=f"{stage_prefix}_credit_accepted",



                        round_index=round_index, policy_id=policy_id,



                        trace_id=str(record.trace_id), request=request,



                        routing=routing, source="entry_preference_tail",



                    )



                elif not retryable:



                    entry_preference_rejected.add(identity)



                    entry_preference_pending.pop(identity, None)



                    entry_preference_attempts.pop(identity, None)



                    self._last_live_entry_preference_rejected += 1



                    self._append_execution_trace_observatory(



                        stage=f"{stage_prefix}_credit_rejected",



                        round_index=round_index, policy_id=policy_id,



                        trace_id=str(record.trace_id), request=request,



                        routing=routing, source="entry_preference_tail",



                    )



                else:



                    self._last_live_entry_preference_deferred += 1



                    # Emit the first attempt and logarithmic retries without



                    # flooding the capped observatory during a long MT5 run.



                    if attempt == 1 or (attempt & (attempt - 1)) == 0:



                        self._append_execution_trace_observatory(



                            stage=f"{stage_prefix}_credit_deferred",



                            round_index=round_index, policy_id=policy_id,



                            trace_id=str(record.trace_id), request=request,



                            routing=routing, source="entry_preference_tail",



                        )



            self._last_live_entry_preference_pending = len(entry_preference_pending)



            return credited_total







        def settle_legacy_close(



            *,



            ack_sequence: int,



            request: BridgeRequest,



            account_context: dict[str, float],



            details: dict[str, object],



        ) -> None:



            if ack_sequence in legacy_settled_sequences:



                return



            cached = legacy_open_traces.get(ack_sequence)



            if not cached:



                return



            cached_request = cached.get("request")



            cached_trace = cached.get("trace")



            cached_action = int(cached.get("action", int(Action.HOLD)))



            if not isinstance(cached_request, BridgeRequest) or cached_trace is None:



                return



            close_trace = enrich_step_trace(



                cached_trace,



                dict(getattr(cached_trace, "action_details", {}) or {}),



                str(getattr(cached_trace, "positions_pipe", "")),



                request.positions_pipe,



                execution_details=details,



                account_context_after=account_context,



            )



            close_receipt = {



                **dict(details),



                "receipt_stage": "legacy_execution_close",



            }



            close_routing = self._bind_execution_trace_receipt(



                policy,



                str(getattr(close_trace, "trace_id", "") or ""),



                close_receipt,



            )



            self._append_execution_trace_observatory(



                stage="execution_close",



                round_index=round_index,



                policy_id=policy_id,



                trace=close_trace,



                request=request,



                receipt=close_receipt,



                routing=close_routing,



                source="legacy_execution_ledger",



            )



            entry_equity = float(cached.get("equity", 0.0) or 0.0)



            reward = _closed_execution_reward(



                details,



                initial_equity=float(self.config.initial_equity),



                entry_equity=entry_equity,



                exit_equity=float(request.equity),



                technical_min_rr=float(self.config.technical_min_rr),



            )



            # Legacy context/shared-brain policies do not enter the public



            # Five competition registry.  Queue their immutable one-trace



            # close here, while the owner trace and final close evidence are



            # still available, then let the normal drain apply it between



            # bridge actions.



            close_trace_id = str(getattr(close_trace, "trace_id", "") or "").strip()



            if (



                bool(self.config.codex_teacher_enabled)



                and bool(learning_enabled)



                and close_trace_id



                and not close_trace_id.startswith("five:")



            ):



                legacy_lifecycle = _legacy_codex_lifecycle_from_close_details(



                    details,



                    close_trace,



                    float(reward),



                )



                legacy_lifecycle = replace(



                    legacy_lifecycle,



                    account_context_after_close={



                        **dict(account_context or {}),



                        "balance": float(getattr(request, "balance", 0.0) or 0.0),



                        "equity": float(getattr(request, "equity", 0.0) or 0.0),



                        "drawdown_pct": float(getattr(request, "drawdown_pct", 0.0) or 0.0),



                        "positions": float(getattr(request, "positions", 0) or 0),



                        "sequence": float(getattr(request, "sequence", 0) or 0),



                    },



                )



                legacy_case_id = "|".join((



                    self._codex_lifecycle_identity(legacy_lifecycle),



                    "legacy",



                    close_trace_id,



                ))



                if (



                    self._codex_teacher_waits_for_counterfactual()



                    and not self._codex_lifecycle_has_counterfactual(legacy_lifecycle)



                ):



                    # The final close is immutable, but the post-exit hold



                    # observation is emitted by the EA only after its horizon.



                    # Keep the exact owner trace until that row is visible so



                    # legacy/context policies receive the same causal contract



                    # as the public Five path.



                    self._defer_codex_teacher_case(



                        legacy_case_id,



                        legacy_lifecycle,



                        (close_trace,),



                        None,



                        target_trace_id=close_trace_id,



                    )



                    submitted_before = int(



                        self._last_codex_teacher_phase.get("submitted", 0) or 0



                    )



                else:



                    submitted_before = int(



                        self._last_codex_teacher_phase.get("submitted", 0) or 0



                    )



                    self._submit_codex_teacher_case(



                        legacy_case_id,



                        legacy_lifecycle,



                        (close_trace,),



                        target_trace_id=close_trace_id,



                    )



                submitted_after = int(



                    self._last_codex_teacher_phase.get("submitted", 0) or 0



                )



                if submitted_after > submitted_before:



                    self._wait_for_codex_teacher_action_barrier(



                        policy,



                        reason="legacy_lifecycle",



                    )



                else:



                    self._drain_codex_teacher_results(policy, wait=False)



            # Net MT5 P/L already contains spread, commission, swap and the



            # resulting drawdown. Subtracting them again would turn small



            # winning lifecycles into false negative credit.



            cost = 0.0



            if self._trace_learning_blocked(close_trace):



                return



            signal = policy.observe_transition(



                close_trace,



                request.inputs,



                float(reward),



                float(cost),



                False,



            )



            self._append_execution_trace_observatory(



                stage="lifecycle_credit",



                round_index=round_index,



                policy_id=policy_id,



                trace=close_trace,



                request=request,



                receipt=close_receipt,



                routing=close_routing,



                credit=int(signal.get("lifecycle_credit", 0) or 0) if isinstance(signal, Mapping) else 0,



                source="settle_legacy_close",



            )



            memory_eligible = bool(



                signal.get("memory_eligible", details["memory_eligible"])



                if isinstance(signal, dict) else details["memory_eligible"]



            )



            if memory_eligible:



                self._append_experience_journal(



                    round_index=round_index,



                    policy_id=policy_id,



                    request=cached_request,



                    trace=close_trace,



                    next_obs=request.inputs,



                    reward=float(reward),



                    cost=float(cost),



                    done=False,



                    signal=signal,



                    action=cached_action,



                    score=float(cached.get("score", 0.0) or 0.0),



                    confidence=float(cached.get("confidence", 0.0) or 0.0),



                )



            legacy_settled_sequences.add(ack_sequence)



            legacy_open_traces.pop(ack_sequence, None)







        def settle_legacy_execution_ledger(request: BridgeRequest, account_context: dict[str, float]) -> None:



            execution_path = self._latest_runtime_file(self.config.execution_file)



            if execution_path is None or not execution_path.exists():



                return



            try:



                with execution_path.open("r", newline="", encoding="utf-8-sig") as handle:



                    rows = list(csv.DictReader(handle, delimiter="\t"))



            except OSError:



                return



            for row in rows:



                if str(row.get("closed", "")).strip().lower() not in {"1", "true", "yes", "ok", "success"}:



                    continue



                ack_sequence = _legacy_ack_sequence_for_close_row(row, legacy_open_traces)



                if ack_sequence is None:



                    continue



                if ack_sequence in legacy_settled_sequences:



                    continue



                details = _execution_record_close_details(



                    row,



                    requested_action=int(legacy_open_traces[ack_sequence].get("action", int(Action.HOLD))),



                    owner_ack_sequence=ack_sequence,



                )



                settle_legacy_close(



                    ack_sequence=ack_sequence,



                    request=request,



                    account_context=account_context,



                    details=details,



                )



            if not learning_enabled:



                return



            summary = parse_mt5_execution_ledger(



                execution_path,



                initial_equity=float(self.config.deposit),



                min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



                technical_min_rr=float(self.config.technical_min_rr),



            )



            deferred_released = self._flush_deferred_codex_teacher_cases(



                summary.lifecycles



            )



            if deferred_released:



                self._wait_for_codex_teacher_action_barrier(



                    policy,



                    reason="legacy_counterfactual",



                )



            # Public Five also needs the authoritative ledger lifecycle while



            # the tester is still running. Its close ACK can be provisional,



            # and the legacy trace-matching path does not always resolve a



            # five-agent packet back to the entry owner. Deferring these



            # lifecycles until ``evaluate_policy`` makes the reward visible



            # only after every decision has already been emitted; the



            # action-local expectancy veto then has no chance to affect the



            # next action. Keep the old immediate-close compatibility path for



            # non-Five legacy policies, but settle complete Five lifecycles



            # from the same authoritative execution ledger used at the end.



            experts = list(getattr(



                getattr(policy, "registry", None), "chuyen_gia_theo_context", {}



            ).values())



            canonical_full_rl = any(



                callable(getattr(getattr(item, "policy", None), "_direct_rnn_full_rl_mode_enabled", None))



                and bool(getattr(item.policy, "_direct_rnn_full_rl_mode_enabled")())



                for item in experts



            )



            policy_candidates = [policy]



            policy_candidates.extend(getattr(item, "policy", None) for item in experts)



            public_five_active = any(



                callable(getattr(candidate, "_uses_public_five_policy", None))



                and bool(getattr(candidate, "_uses_public_five_policy")())



                for candidate in policy_candidates



                if candidate is not None



            )



            if canonical_full_rl or public_five_active:



                for lifecycle in summary.lifecycles:



                    trace_id = str(getattr(lifecycle, "trace_id", "") or "")



                    if not trace_id or not bool(getattr(lifecycle, "final_close_confirmed", False)):



                        continue



                    identity = "|".join((



                        str(getattr(lifecycle, "position_ticket", "") or ""),



                        trace_id,



                        str(getattr(lifecycle, "close_reason", "") or ""),



                        f"{float(getattr(lifecycle, 'realized_r', 0.0) or 0.0):.8f}",



                    ))



                    if identity in canonical_lifecycle_settled:



                        continue



                    self._last_live_lifecycle_settlements += 1



                    credited = self._apply_execution_lifecycle_credit(



                        policy=policy,



                        lifecycle=lifecycle,



                        round_index=round_index,



                        policy_id=policy_id,



                        source="live_execution_ledger",



                        request=request,



                    )



                    self._last_live_lifecycle_updates += int(credited)



                    canonical_lifecycle_settled.add(identity)



            for lifecycle in summary.lifecycles:



                trace_id = str(getattr(lifecycle, "exit_trace_id", "") or "")



                horizon = getattr(lifecycle, "post_exit_horizon_bars", None)



                if not trace_id or horizon is None or getattr(lifecycle, "post_exit_hold_r", None) is None:



                    continue



                identity = "|".join((



                    str(getattr(lifecycle, "position_ticket", "") or ""), trace_id,



                    str(horizon), f"{float(getattr(lifecycle, 'post_exit_hold_r', 0.0) or 0.0):.8f}",



                ))



                if identity in exit_counterfactual_settled:



                    continue



                self._apply_execution_exit_credit(



                    policy=policy,



                    lifecycle=lifecycle,



                    round_index=round_index,



                    policy_id=policy_id,



                    source="live_execution_ledger",



                    request=request,



                )



                exit_counterfactual_settled.add(identity)



        while True:



            now = time.monotonic()



            # Poll before report/timeout handling so a request made while the



            # tester is between ticks is still mirrored into the EA sandbox.



            # We deliberately keep serving the bridge until MT5 writes its



            # normal report; killing the process here would lose reconciliation



            # and leave the checkpoint without a trustworthy terminal boundary.



            publish_save_and_stop_request()



            # Non-visual MT5 runs write their final report before the portable



            # terminal exits when ShutdownTerminal=0.  Waiting for the normal



            # stale-bridge timeout here needlessly keeps the in-memory policy



            # (and therefore its final checkpoint) hostage for ``timeout_sec``



            # after a completed Strategy Tester pass.  A report is accepted



            # only once the normal parser can read it, so a partially written



            # HTML file cannot end the bridge early.  Visual runs deliberately



            # retain the old process-exit boundary because their report can be



            # generated while the operator is still inspecting the session.



            if not self.config.visual:



                located_report = self._materialize_report(report_name, report_path)



                if located_report is not None:



                    try:



                        parse_mt5_report(located_report)



                    except (OSError, ValueError):



                        pass



                    else:



                        settle_entry_preference_updates()



                        return served



            elif served > 0 and (now - last_request_seen >= 5.0):



                located_report = self._materialize_report(report_name, report_path)



                if located_report is not None:



                    try:



                        parse_mt5_report(located_report)



                    except (OSError, ValueError):



                        pass



                    else:



                        settle_entry_preference_updates()



                        print("\n" + "=" * 75, flush=True)



                        print("  [MT5 VISUAL] Mo phong MT5 Strategy Tester da hoan tat thanh cong!", flush=True)



                        print("  Bao cao HTML da duoc tao va doc thanh cong.", flush=True)



                        print("=" * 75 + "\n", flush=True)



                        return served



            # A terminal can stay alive indefinitely after an account/server



            # rejection. Do not wait for the generic bridge timeout when its



            # own runner log already proves that no tester agent can start.



            if served == 0 and now >= startup_failure_probe_at:



                startup_failure_probe_at = now + 1.0



                if now - run_started >= startup_failure_grace_sec:



                    startup_failure = _detect_mt5_startup_failure(



                        self.runner_dir,



                        offline_preflight=bool(getattr(self.config, "offline_preflight", False)),



                    )



                    if startup_failure is not None:



                        raise RuntimeError(



                            "MT5 startup preflight failed: "



                            + json.dumps(startup_failure, ensure_ascii=False, sort_keys=True)



                        )



            if int(getattr(self.config, "max_wall_sec", 0) or 0) > 0 and now - run_started >= int(self.config.max_wall_sec):



                settled = settle_timeout_lifecycle_updates()



                if settled:



                    print(



                        f"[Bridge] wall_guard settled_lifecycle_updates={settled} "



                        f"trace_count={len(timeout_lifecycle_settled)}",



                        flush=True,



                    )



                # A bounded diagnostic run deliberately stops the tester



                # without its normal report-finalization pass. Drain every



                # already-mature counterfactual row before raising so the



                # final polling interval cannot silently discard valid



                # entry-preference credit. Rows whose path has not reached



                # its horizon remain correctly unlabelled.



                settled_preference = settle_entry_preference_updates()



                if settled_preference:



                    print(



                        f"[Bridge] wall_guard settled_entry_preference_updates={settled_preference} "



                        f"rows={self._last_live_entry_preference_records}",



                        flush=True,



                    )



                raise TimeoutError(



                    f"MT5 max wall clock exceeded after {now - run_started:.1f}s "



                    f"(limit={int(self.config.max_wall_sec)}s, served={served}, last_sequence={last_sequence})"



                )



            if self.config.visual and served > 0 and now >= visual_terminal_probe_at:



                visual_terminal_probe_at = now + 1.0



                if self._runner_terminal_is_alive():



                    visual_terminal_missing_since = None



                else:



                    if visual_terminal_missing_since is None:



                        visual_terminal_missing_since = now



                    elif now - visual_terminal_missing_since >= visual_terminal_missing_grace_sec:



                        # Cho phép đường báo cáo hợp lệ đi qua nhánh kết thúc



                        # bên dưới; nếu không có báo cáo thì dừng rõ ràng thay



                        # vì tiếp tục ăn dữ liệu từ tester mồ côi.



                        located_report = self._materialize_report(report_name, report_path)



                        if located_report is None:



                            raise RuntimeError(



                                "MT5 trực quan đã đóng nhưng bộ chạy không nhận được báo cáo; "



                                "đã dừng để không tiếp tục xử lý tester mồ côi"



                            )



            timeout_limit = (



                initial_bridge_timeout_sec



                if served == 0 and last_request_seen == run_started



                else float(self.config.timeout_sec)



            )



            if now - last_request_seen >= timeout_limit:



                located_report = self._materialize_report(report_name, report_path)



                poll_status = process.poll()



                if located_report is None and poll_status is not None:



                    located_report = self._materialize_report(report_name, report_path)



                if located_report is not None and (not self.config.visual or poll_status is not None or not self.config.shutdown_terminal):



                    settle_entry_preference_updates()



                    return served



                raise TimeoutError(



                    f"Timed out serving MT5 action bridge after {now - last_request_seen:.1f}s without a new request "



                    f"and without a tester report (served={served}, last_sequence={last_sequence}): {report_path}"



                )



            request_path = self._latest_runtime_file(self.config.request_file)



            replay_exhausted = False



            if replay_iterator is not None:



                if request_path is None:



                    request_path = self.mql5_files / self.config.request_file



                    request_path.parent.mkdir(parents=True, exist_ok=True)



                try:



                    request = next(replay_iterator)



                except StopIteration:



                    request = None



                    replay_exhausted = True



            else:



                request = self._read_request(request_path) if request_path is not None else None



            if replay_exhausted:



                # Match the bounded MT5 wall-stop contract: settle every



                # durable lifecycle already present in the replay ledger, but



                # never invent a terminal transition for the last open intent.



                settle_timeout_lifecycle_updates()



                settle_entry_preference_updates()



                return served



            if request is not None:



                    request = self._normalize_request_inputs(request)



                    sequence = request.sequence



                    if sequence != last_sequence:



                        if bool(self.config.mt5_failure_replay_mode):



                            anchor = str(self.config.mt5_failure_replay_anchor_time or "").strip()



                            request_time = str(getattr(request, "time", "") or "").strip()



                            if _mt5_replay_anchor_reached(anchor, request_time):



                                self._mt5_failure_replay_anchor_seen = True



                                self._mt5_failure_replay_anchor_sequence = int(sequence)



                                self._mt5_failure_replay_anchor_time_seen = request_time



                        reset_entry_loss_guard_for_day(request)



                        record_runtime_fitness_guard_rows(request)



                        account_context = self._account_context_for_request(request, equity_history)



                        last_inputs = request.inputs



                        # Counterfactual path rows are produced independently



                        # by the EA and must not depend on the execution ledger



                        # being present, parseable, or caught up.  Consume them



                        # once at the top of every fresh bridge request.  The



                        # old placement at the end of legacy-ledger settlement



                        # silently starved this reward stream whenever that



                        # settlement returned early.



                        settle_entry_preference_updates(request)



                        if pending_five_actions:



                            # Settle immutable entry outcomes before shaping



                            # this transition.  This gives the creating agent



                            # final net-R even when MT5 closed it by SL/TP



                            # without a new policy close command.



                            self._refresh_five_competition_lifecycles(



                                policy,



                                round_index=round_index,



                                policy_id=policy_id,



                                request=request,



                                account_context_after_close={



                                    **dict(account_context or {}),



                                    "balance": float(getattr(request, "balance", 0.0) or 0.0),



                                    "equity": float(getattr(request, "equity", 0.0) or 0.0),



                                    "drawdown_pct": float(getattr(request, "drawdown_pct", 0.0) or 0.0),



                                    "positions": float(getattr(request, "positions", 0) or 0),



                                    "sequence": float(getattr(request, "sequence", 0) or 0),



                                },



                            )



                            five_feedback_traces: list[object] = []



                            five_conflict_cost = np.zeros(5, dtype=np.float32)



                            five_invalid_position_penalty, positions_before = (



                                self._five_invalid_position_action_no_position_penalties(



                                    pending_request,



                                    pending_five_actions,



                                )



                            )



                            five_invalid_cancel_penalty, pending_orders_before = (



                                self._five_invalid_cancel_no_pending_penalties(



                                    pending_request,



                                    pending_five_actions,



                                )



                            )



                            five_policy_violation_penalty = (



                                five_invalid_position_penalty + five_invalid_cancel_penalty



                            ).astype(np.float32)



                            five_position_owners_before: set[int] = set()



                            for position in self._positions_from_pipe(



                                str(getattr(pending_request, "positions_pipe", "") or "")



                            ):



                                owner_trace_id = str(position.get("owner_trace_id", "") or "")



                                owner_agent = self._five_agent_by_trace.get(owner_trace_id)



                                if owner_agent is not None:



                                    five_position_owners_before.add(int(owner_agent))



                            five_receipts: list[dict[str, object]] = []



                            for pending_item in pending_five_actions:



                                five_trace = pending_item["trace"]



                                five_action = int(pending_item["action"])



                                five_agent_id = int(pending_item["agent_id"])



                                five_authority = five_agent_authority(five_agent_id)



                                five_details = self._execution_details(



                                    pending_request,



                                    request,



                                    five_action,



                                    expected_trace_id=str(getattr(five_trace, "trace_id", "") or ""),



                                    expected_agent_id=five_agent_id,



                                )



                                proposal_details = dict(getattr(five_trace, "action_details", {}) or {})



                                policy_action = int(proposal_details.get(



                                    "policy_action",



                                    proposal_details.get("five_action_index", five_action),



                                ) or 0)



                                invalid_position_action_without_position = bool(



                                    policy_action in POSITION_REQUIRED_ACTIONS



                                    and positions_before == 0



                                )



                                invalid_cancel_without_pending = bool(



                                    policy_action == int(Action.CANCEL_ORDERS)



                                    and pending_orders_before == 0



                                )



                                five_resolution = str(five_details.get("execution_status", "") or "").upper()



                                arbitration_blocked = bool(



                                    proposal_details.get("five_pre_wire_arbitration_blocked", False)



                                    or



                                    five_resolution.startswith("BATCH_")



                                    or five_resolution.startswith("AUTHORITY_")



                                )



                                owns_position_before = five_agent_id in five_position_owners_before



                                authority_responsibility_active = five_agent_responsibility_active(



                                    five_agent_id,



                                    position_count=positions_before,



                                    pending_order_count=pending_orders_before,



                                    owns_position=owns_position_before,



                                )



                                flat_hold = bool(



                                    policy_action == int(Action.HOLD)



                                    and not authority_responsibility_active



                                )



                                economic_actor_eligible = bool(



                                    invalid_position_action_without_position



                                    or (



                                        policy_action != int(Action.HOLD)



                                        and not arbitration_blocked



                                    )



                                    or authority_responsibility_active



                                )



                                five_trace = enrich_step_trace(



                                    five_trace,



                                    {



                                        **proposal_details,



                                        "five_economic_actor_eligible": economic_actor_eligible,



                                        "five_actor_eligible": economic_actor_eligible,



                                        "five_execution_status": str(five_details.get("execution_status", "") or ""),



                                        # Lifecycle credit needs the exact



                                        # decision packet, but only for an



                                        # entry that MT5 actually accepted as



                                        # a fill or a live pending order.  A



                                        # rejected/batch-arbitrated proposal



                                        # must not consume the bounded owner



                                        # packet cache and evict real entries.



                                        "five_execution_filled": bool(



                                            five_details.get("filled", False)



                                        ),



                                        "five_execution_placed": bool(



                                            five_details.get("placed", False)



                                        ),



                                        "five_execution_closed": bool(



                                            five_details.get("closed", False)



                                        ),



                                        "five_arbitration_blocked": arbitration_blocked,



                                        "five_positions_before": int(positions_before),



                                        "five_pending_orders_before": int(pending_orders_before),



                                        "five_owned_position_before": owns_position_before,



                                        "five_agent_authority_version": FIVE_AGENT_AUTHORITY_VERSION,



                                        "five_agent_authority_role": five_authority.role,



                                        "five_agent_authority_actions": sorted(



                                            five_authority.allowed_actions



                                        ),



                                        "five_agent_authority_responsibility": (



                                            five_authority.responsibility



                                        ),



                                        "five_authority_responsibility_active": (



                                            authority_responsibility_active



                                        ),



                                        "five_flat_hold_no_reward": flat_hold,



                                        "five_reward_sample_eligible": bool(



                                            not flat_hold and not arbitration_blocked



                                        ),



                                        "five_invalid_position_action_no_position": (



                                            invalid_position_action_without_position



                                        ),



                                        "five_invalid_cancel_no_pending": (



                                            invalid_cancel_without_pending



                                        ),



                                        "five_policy_violation_penalty_r": float(



                                            five_policy_violation_penalty[five_agent_id]



                                        ),



                                    },



                                    str(getattr(five_trace, "positions_pipe", "") or ""),



                                    request.positions_pipe,



                                    execution_details=five_details,



                                    account_context_after=account_context,



                                )



                                trace_key = str(getattr(five_trace, "trace_id", "") or "")



                                if trace_key:



                                    # Keep the policy distribution separate from



                                    # the wire action.  Inactive manager/governor



                                    # rows are valid HOLD transport decisions but



                                    # must not be counted as economic policy



                                    # collapse for the actor that could trade.



                                    self._five_policy_action_by_trace.setdefault(



                                        trace_key,



                                        int(policy_action),



                                    )



                                    self._five_economic_actor_eligible_by_trace[



                                        trace_key



                                    ] = bool(economic_actor_eligible)



                                receipt = {**dict(five_details), "receipt_stage": "five_bridge_batch_ack"}



                                five_feedback_traces.append(five_trace)



                                five_receipts.append(receipt)



                                resolution = str(five_details.get("execution_status", "") or "").upper()



                                if bool(five_details.get("rejected", False)):



                                    # Batch arbitration means the proposal was



                                    # never executed. It is audit evidence, not



                                    # an economic loss and must not make HOLD



                                    # artificially optimal for the shared actor.



                                    five_conflict_cost[five_agent_id] = np.float32(



                                        0.0



                                        if (



                                            resolution.startswith("BATCH_")



                                            or resolution.startswith("AUTHORITY_")



                                        )



                                        else 0.5



                                    )



                                routing = self._bind_execution_trace_receipt(



                                    policy,



                                    str(getattr(five_trace, "trace_id", "") or ""),



                                    receipt,



                                )



                                self._append_execution_trace_observatory(



                                    stage="five_execution_receipt",



                                    round_index=round_index,



                                    policy_id=policy_id,



                                    trace=five_trace,



                                    request=request,



                                    receipt=receipt,



                                    routing=routing,



                                    source="five_bridge_batch_ack",



                                )



                            pattern_receipt_hook = getattr(policy, "bind_five_pattern_anchor_receipts", None)



                            if learning_enabled and callable(pattern_receipt_hook):



                                pattern_receipt_routing = pattern_receipt_hook(



                                    tuple(five_feedback_traces), int(pending_request.sequence)



                                )



                                self._append_execution_trace_observatory(



                                    stage="five_pattern_receipts_bound",



                                    round_index=round_index,



                                    policy_id=policy_id,



                                    trace_id=str(getattr(five_feedback_traces[0], "trace_id", "") or ""),



                                    request=request,



                                    routing=(



                                        dict(pattern_receipt_routing)



                                        if isinstance(pattern_receipt_routing, Mapping)



                                        else {"bound": bool(pattern_receipt_routing)}



                                    ),



                                    source="five_bridge_batch_ack",



                                )



                            pending_cache_hook = getattr(



                                policy, "register_five_pending_order_packet", None



                            )



                            pending_timeout_stats: dict[str, int] = {}



                            if learning_enabled and callable(pending_cache_hook):



                                pending_cache_hook(tuple(five_feedback_traces))



                                pending_timeout_stats = self._refresh_five_pending_order_timeouts(



                                    policy



                                )



                            # Keep an immutable packet for every emitted



                            # command, not only entries.  Terminal Codex cases



                            # use the lifecycle's exit_trace_id to bind a



                            # close/manage action back to this exact packet.



                            action_packet_hook = getattr(



                                policy, "register_five_action_packet", None



                            )



                            if learning_enabled and callable(action_packet_hook):



                                action_packet_hook(tuple(five_feedback_traces))



                            lifecycle_cache_hook = getattr(



                                policy, "register_five_entry_lifecycle_packet", None



                            )



                            if learning_enabled and callable(lifecycle_cache_hook):



                                lifecycle_cache_hook(tuple(five_feedback_traces))



                            if learning_enabled:



                                economic_hook = getattr(policy, "observe_five_economic_transition", None)



                                if not callable(economic_hook):



                                    raise RuntimeError("five-agent facade missing economic reward batch hook")



                                before_day = str(getattr(pending_request, "time", "") or "").split(" ", 1)[0]



                                after_day = str(getattr(request, "time", "") or "").split(" ", 1)[0]



                                terminal = bool(before_day and after_day and before_day != after_day)



                                economic = self._five_economic_inputs(



                                    request,



                                    five_conflict_cost,



                                    five_policy_violation_penalty,



                                )



                                stats = economic_hook(



                                    five_feedback_traces,



                                    **economic,



                                    terminal=terminal,



                                    dones=np.ones(5, dtype=bool) if terminal else np.zeros(5, dtype=bool),



                                )



                                self._append_execution_trace_observatory(



                                    stage="five_competitive_reward",



                                    round_index=round_index,



                                    policy_id=policy_id,



                                    trace=five_feedback_traces[0] if five_feedback_traces else None,



                                    request=request,



                                    receipt={"receipts": five_receipts},



                                    routing={



                                        **{str(key): value for key, value in dict(stats).items() if np.isscalar(value)},



                                        **{



                                            f"pending_timeout_{key}": int(value)



                                            for key, value in pending_timeout_stats.items()



                                        },



                                    },



                                    source="five_economic_batch",



                                )



                                if terminal:



                                    self._finish_five_episode()



                            if rlhf_reward_model is not None:



                                apply_mt5_rlhf_update(



                                    tuple(five_feedback_traces),



                                    request,



                                )



                            pending_five_actions = ()



                            pending_trace = None



                        if pending_trace is not None:



                            execution_details = complete_final_close_receipt(



                                self._execution_details(pending_request, request, pending_action)



                            )



                            flat_idle = bool(execution_details.get("flat_idle", False))



                            # V10: every bridge interval yields a market-only



                            # counterfactual label, including explicit HOLD /



                            # FLAT_IDLE.  This deliberately bypasses economic



                            # policy credit and therefore cannot double-credit a



                            # final lifecycle.  It is the recovery path that



                            # prevents HOLD periods from starving BUY/SELL



                            # evidence.



                            # V28: FLAT_IDLE is telemetry only. It is excluded



                            # before shadow/recovery hooks so no-position snapshots



                            # cannot manufacture BUY/SELL evidence.



                            if not flat_idle and False:



                                observe_shadow = getattr(policy, "observe_shadow_transition", None)



                                if callable(observe_shadow):



                                    try:



                                        observe_shadow(



                                            pending_trace,



                                            request.inputs,



                                            sequence=int(sequence),



                                            source="mt5_bridge_interval",



                                        )



                                    except Exception as exc:



                                        emit_heartbeat = getattr(policy, "emit_bridge_heartbeat", None)



                                        if callable(emit_heartbeat):



                                            emit_heartbeat(



                                                stage="shadow_feedback_error",



                                                sequence=int(sequence),



                                                request_event_id=str(getattr(request, "event_id", "")),



                                                trace_id=str(getattr(pending_trace, "trace_id", "")),



                                                metadata={"error": type(exc).__name__},



                                            )



                            # V27: a plain MT5 flat snapshot after HOLD/no-op is



                            # not a transition.  It must not be threaded back into



                            # the trace, starvation input, memory, causal ledger or



                            # any learning path.



                            if not flat_idle:



                                starvation_state.confirm_execution(pending_request, request, emitted_action=pending_action)



                                pending_trace = enrich_step_trace(



                                    pending_trace,



                                    dict(getattr(pending_trace, "action_details", {}) or {}),



                                    str(getattr(pending_trace, "positions_pipe", "")),



                                    request.positions_pipe,



                                    execution_details=execution_details,



                                    account_context_after=account_context,



                                )



                                receipt = {



                                    **dict(execution_details),



                                    "receipt_stage": "bridge_ack",



                                }



                                routing = self._bind_execution_trace_receipt(



                                    policy,



                                    str(getattr(pending_trace, "trace_id", "") or ""),



                                    receipt,



                                )



                                self._append_execution_trace_observatory(



                                    stage="execution_receipt",



                                    round_index=round_index,



                                    policy_id=policy_id,



                                    trace=pending_trace,



                                    request=request,



                                    receipt=receipt,



                                    routing=routing,



                                    source="bridge_ack",



                                )



                            if learning_enabled:



                                _register_legacy_open_trace(



                                    legacy_open_traces,



                                    request=pending_request,



                                    trace=pending_trace,



                                    action=int(pending_action),



                                    execution_details=execution_details,



                                    score=float(pending_score),



                                    confidence=float(pending_confidence),



                                    equity=float(pending_equity or 0.0),



                                    drawdown=float(pending_drawdown),



                                    spread_points=float(pending_spread_points),



                                )



                            if _pending_transition_feedback_allowed(



                                flat_idle=flat_idle,



                                learning_enabled=learning_enabled,



                                has_observer=hasattr(policy, "observe_transition"),



                                trace_learning_blocked=self._trace_learning_blocked(pending_trace),



                            ):



                                # Legacy trial/round path retains its original



                                # transition behaviour.



                                realised_close = bool(execution_details.get("closed", False)) and bool(



                                    execution_details.get("action_feedback_eligible", False)



                                )



                                if realised_close:



                                    reward = _closed_execution_reward(



                                        execution_details,



                                        initial_equity=float(self.config.initial_equity),



                                        entry_equity=float(pending_equity or 0.0),



                                        exit_equity=float(request.equity),



                                        technical_min_rr=float(self.config.technical_min_rr),



                                    )



                                    record_entry_loss_guard_close(float(reward))



                                    # The close reward is authoritative net-R;



                                    # do not debit execution costs twice.



                                    cost = 0.0



                                else:



                                    reward = 0.0



                                    cost = 0.0



                                signal = policy.observe_transition(



                                    pending_trace,



                                    request.inputs,



                                    float(reward),



                                    float(cost),



                                    False,



                                )



                                if bool(execution_details.get("closed", False)):



                                    self._append_execution_trace_observatory(



                                        stage="lifecycle_credit",



                                        round_index=round_index,



                                        policy_id=policy_id,



                                        trace=pending_trace,



                                        request=request,



                                        receipt={**dict(execution_details), "receipt_stage": "bridge_transition_close"},



                                        credit=int(signal.get("lifecycle_credit", 0) or 0) if isinstance(signal, Mapping) else 0,



                                        source="bridge_transition",



                                    )



                                memory_eligible = bool(



                                    signal.get("memory_eligible", execution_details["memory_eligible"])



                                    if isinstance(signal, dict) else execution_details["memory_eligible"]



                                )



                                if memory_eligible and pending_request is not None:



                                    self._append_experience_journal(



                                        round_index=round_index,



                                        policy_id=policy_id,



                                        request=pending_request,



                                        trace=pending_trace,



                                        next_obs=request.inputs,



                                        reward=float(reward),



                                        cost=float(cost),



                                        done=False,



                                        signal=signal,



                                        action=pending_action,



                                        score=pending_score,



                                        confidence=pending_confidence,



                                    )







                            if (



                                learning_enabled



                                and hasattr(policy, "observe_transition")



                                and not False



                            ):



                                for record in self._parse_execution_records(getattr(request, "execution_pipe", "")):



                                    closed_value = str(record.get("closed", "")).strip().lower()



                                    if closed_value not in {"1", "true", "yes", "ok", "success"}:



                                        continue



                                    ack_sequence = _legacy_ack_sequence_for_close_row(record, legacy_open_traces)



                                    if ack_sequence is None:



                                        continue



                                    if ack_sequence in legacy_settled_sequences:



                                        continue



                                    cached = legacy_open_traces.get(ack_sequence)



                                    if not cached:



                                        continue



                                    cached_request = cached.get("request")



                                    cached_trace = cached.get("trace")



                                    cached_action = int(cached.get("action", int(Action.HOLD)))



                                    if not isinstance(cached_request, BridgeRequest) or cached_trace is None:



                                        continue



                                    close_details = _execution_record_close_details(



                                        record,



                                        requested_action=cached_action,



                                        owner_ack_sequence=ack_sequence,



                                    )



                                    close_details = complete_final_close_receipt(close_details)



                                    close_trace = enrich_step_trace(



                                        cached_trace,



                                        dict(getattr(cached_trace, "action_details", {}) or {}),



                                        str(getattr(cached_trace, "positions_pipe", "")),



                                        request.positions_pipe,



                                        execution_details=close_details,



                                        account_context_after=account_context,



                                    )



                                    close_receipt = {



                                        **dict(close_details),



                                        "receipt_stage": "execution_close",



                                    }



                                    close_routing = self._bind_execution_trace_receipt(



                                        policy,



                                        str(getattr(close_trace, "trace_id", "") or ""),



                                        close_receipt,



                                    )



                                    self._append_execution_trace_observatory(



                                        stage="execution_close",



                                        round_index=round_index,



                                        policy_id=policy_id,



                                        trace=close_trace,



                                        request=request,



                                        receipt=close_receipt,



                                        routing=close_routing,



                                        source="execution_pipe",



                                    )



                                    if self._trace_learning_blocked(close_trace):



                                        continue



                                    entry_equity = float(cached.get("equity", 0.0) or 0.0)



                                    reward = _closed_execution_reward(



                                        close_details,



                                        initial_equity=float(self.config.initial_equity),



                                        entry_equity=entry_equity,



                                        exit_equity=float(request.equity),



                                        technical_min_rr=float(self.config.technical_min_rr),



                                    )



                                    record_entry_loss_guard_close(float(reward))



                                    # Keep online action credit aligned with



                                    # final lifecycle expectancy (net-R).



                                    cost = 0.0



                                    signal = policy.observe_transition(



                                        close_trace,



                                        request.inputs,



                                        float(reward),



                                        float(cost),



                                        False,



                                    )



                                    self._append_execution_trace_observatory(



                                        stage="lifecycle_credit",



                                        round_index=round_index,



                                        policy_id=policy_id,



                                        trace=close_trace,



                                        request=request,



                                        receipt=close_receipt,



                                        routing=close_routing,



                                        credit=int(signal.get("lifecycle_credit", 0) or 0) if isinstance(signal, Mapping) else 0,



                                        source="legacy_close_transition",



                                    )



                                    memory_eligible = bool(



                                        signal.get("memory_eligible", close_details["memory_eligible"])



                                        if isinstance(signal, dict) else close_details["memory_eligible"]



                                    )



                                    if memory_eligible:



                                        self._append_experience_journal(



                                            round_index=round_index,



                                            policy_id=policy_id,



                                            request=cached_request,



                                            trace=close_trace,



                                            next_obs=request.inputs,



                                            reward=float(reward),



                                            cost=float(cost),



                                            done=False,



                                            signal=signal,



                                            action=cached_action,



                                            score=float(cached.get("score", 0.0) or 0.0),



                                            confidence=float(cached.get("confidence", 0.0) or 0.0),



                                        )



                                    legacy_settled_sequences.add(ack_sequence)



                                    legacy_open_traces.pop(ack_sequence, None)



                                settle_legacy_execution_ledger(request, account_context)







                        request_day_key = str(getattr(request, "time", "") or "").split(" ", 1)[0]



                        if request_day_key and request_day_key != last_request_day:



                            if last_request_day:



                                reset_context = self._context_code_from_audit(self._trace_context_audit(pending_trace))



                                for state in list(self._position_context_extrema.values()):



                                    self._emit_position_context_extrema_event(



                                        policy,



                                        state,



                                        request=request,



                                        current_context=reset_context or str(state.segment_context),



                                        reason="trading_day_changed",



                                    )



                                self._position_context_extrema = {}



                                reset_intraday = getattr(policy, "reset_intraday_state", None)



                                if callable(reset_intraday):



                                    reset_intraday(request_day_key)



                            last_request_day = request_day_key



                        emit_heartbeat = getattr(policy, "emit_bridge_heartbeat", None)



                        if callable(emit_heartbeat):



                            emit_heartbeat(



                                stage="request_seen",



                                sequence=int(sequence),



                                request_event_id=str(getattr(request, "event_id", "")),



                                metadata={



                                    "positions": int(getattr(request, "positions", 0) or 0),



                                    "day_trades": int(getattr(request, "day_trades", 0) or 0),



                                    "spread_points": float(getattr(request, "spread_points", 0.0) or 0.0),



                                    "entry_loss_guard": {



                                        "active": bool(entry_loss_guard_active()),



                                        "day": str(guard_day),



                                        "closed_count": int(guard_closed_count),



                                        "closed_net_r": float(guard_closed_net_r),



                                        "consecutive_losses": int(guard_consecutive_losses),



                                    },



                                    "runtime_fitness_guard": runtime_guard.snapshot(),



                                },



                            )



                        allowed_actions = self._allowed_actions_for_request(request)



                        runtime_guard_active = runtime_guard.blocks_entries(



                            drawdown_pct=float(getattr(request, "drawdown_pct", 0.0) or 0.0)



                        )



                        if entry_loss_guard_active():



                            allowed_actions.difference_update(self._entry_actions())



                        if runtime_guard_active:



                            allowed_actions.difference_update(self._entry_actions())



                        if hasattr(policy, "act"):



                            flat_idle_firebreak = bool(self.config.suppress_flat_idle_starvation)



                            starvation_preview = starvation_state.preview(



                                request,



                                self.config.min_daily_trades,



                                suppress_flat_idle_feedback=flat_idle_firebreak,



                            )



                            set_starvation = getattr(policy, "set_experience_starvation", None)



                            if callable(set_starvation):



                                set_starvation(starvation_preview)



                            set_account_context = getattr(policy, "set_account_context", None)



                            if callable(set_account_context):



                                set_account_context(account_context, request.positions_pipe)



                            set_regime_raw_inputs = getattr(policy, "set_regime_raw_inputs", None)



                            if callable(set_regime_raw_inputs):



                                # Only the context-expert facade consumes this



                                # side channel.  Direct policies remain



                                # backwards-compatible and receive no new



                                # decision-context argument.



                                set_regime_raw_inputs(request.raw_inputs)



                            goal_context = self._goal_context_for_request(request)



                            five_mode = str(



                                getattr(getattr(policy, "cau_hinh", None), "che_do_quyet_dinh", "") or ""



                            ) == "nam_tac_nhan_dung_chung"



                            # The actor must sample only actions that can be



                            # executed in the current account state.  Letting



                            # Five ignore this mask caused repeated



                            # MAX_ACCOUNT_CAPACITY receipts and taught the



                            # behavior policy from impossible commands.  The EA



                            # remains the final safety layer, but capacity and



                            # canonical bridge legality are now enforced before



                            # categorical sampling for every policy mode.



                            decision_allowed_actions = set(allowed_actions)



                            execution_context = self._execution_context_for_request(request, allowed_actions)



                            execution_context["strict_model1_parity"] = float(



                                bool(self.config.strict_model1_parity)



                            )



                            execution_context["runtime_fitness_guard_active"] = float(bool(runtime_guard_active))



                            execution_context["runtime_fitness_rolling_score"] = float(runtime_guard.last_score)



                            execution_context["runtime_fitness_immediate_exit_ratio"] = float(runtime_guard.last_immediate_ratio)



                            set_decision_context = getattr(policy, "set_decision_context", None)



                            if callable(set_decision_context):



                                set_decision_context(



                                    account_context=account_context,



                                    positions_pipe=request.positions_pipe,



                                    allowed_actions={



                                        int(action): True for action in decision_allowed_actions



                                    },



                                    execution_context=execution_context,



                                    goal=goal_context,



                                    sequence_id=int(sequence),



                                    frame_id=f"mt5:{int(round_index)}:{int(sequence)}",



                                    timestamp=(



                                        float(sequence)



                                        if bool(self.config.strict_model1_parity)



                                        else time.time()



                                    ),



                                    # The facade may otherwise emit the live



                                    # decision while ``act`` still contains



                                    # only the pre-bridge proposal.  The



                                    # semantic/emitted/wire contract becomes



                                    # immutable below, after legality and any



                                    # bounded training probe are resolved.



                                    defer_live_decision_until_bridge=True,



                                )



                            if hasattr(policy, "set_regime_soft_probabilities") and getattr(request, "regime_probabilities", None) is not None:



                                try:



                                    policy.set_regime_soft_probabilities(



                                        request.regime_probabilities,



                                        regime_name=getattr(request, "regime_name", None),



                                        regime_idx=getattr(request, "regime_index", None),



                                    )



                                except Exception:



                                    pass



                            five_hook = getattr(policy, "act_nam_tac_nhan", None)



                            if five_mode:



                                if not callable(five_hook):



                                    raise RuntimeError("five-agent facade does not expose act_nam_tac_nhan")



                                raw_five_traces = tuple(five_hook(request.inputs, goal=goal_context, request=request))



                                apply_entry_timing = getattr(policy, "apply_entry_timing_extrema_bias", None)



                                if callable(apply_entry_timing):



                                    raw_five_traces = tuple(apply_entry_timing(item) for item in raw_five_traces)



                                five_packet, five_traces = self._five_bridge_action_packet(



                                    policy,



                                    request,



                                    raw_five_traces,



                                    account_context=account_context,



                                )



                                finalize_bridge_decision = getattr(policy, "finalize_bridge_decision", None)



                                finalized: list[object] = []



                                packet_rows: list[BridgeAgentAction] = []



                                for row, five_trace in zip(five_packet, five_traces):



                                    if callable(finalize_bridge_decision):



                                        five_trace = finalize_bridge_decision(five_trace)



                                    finalized.append(five_trace)



                                    packet_rows.append(replace(row, trace=five_trace))



                                    self._register_five_competitor_trace(



                                        str(row.trace_id), int(row.agent_id), int(row.action),



                                    )



                                    routing = self._register_execution_intent_trace(policy, five_trace)



                                    self._append_execution_trace_observatory(



                                        stage="five_intent_registered",



                                        round_index=round_index,



                                        policy_id=policy_id,



                                        trace=five_trace,



                                        trace_id=str(row.trace_id),



                                        request=request,



                                        routing=routing,



                                        source="five_bridge_pre_write",



                                    )



                                codex_next_action_links = self._consume_codex_teacher_next_action_links(



                                    policy=policy,



                                    request=request,



                                    trace_id=(



                                        str(getattr(finalized[0], "trace_id", "") or "")



                                        if finalized else ""



                                    ),



                                    action=(



                                        int(packet_rows[0].action)



                                        if packet_rows else int(Action.HOLD)



                                    ),



                                    score=(



                                        float(packet_rows[0].score)



                                        if packet_rows else 0.0



                                    ),



                                    packet_actions=(



                                        {



                                            "agent_id": int(row.agent_id),



                                            "trace_id": str(getattr(trace, "trace_id", "") or ""),



                                            "action": int(row.action),



                                            "score": float(row.score),



                                        }



                                        for row, trace in zip(packet_rows, finalized)



                                    ),



                                )



                                if codex_next_action_links:



                                    codex_link_details = self._codex_teacher_action_link_details(



                                        codex_next_action_links



                                    )



                                    finalized = [



                                        enrich_step_trace(



                                            trace,



                                            codex_link_details,



                                            str(getattr(trace, "positions_pipe", "") or ""),



                                            request.positions_pipe,



                                        )



                                        for trace in finalized



                                    ]



                                    packet_rows = [



                                        replace(row, trace=trace)



                                        for row, trace in zip(packet_rows, finalized)



                                    ]



                                pattern_anchor_hook = getattr(policy, "register_five_pattern_anchor", None)



                                if learning_enabled and callable(pattern_anchor_hook):



                                    pattern_anchor_routing = pattern_anchor_hook(



                                        tuple(finalized), int(sequence)



                                    )



                                    self._append_execution_trace_observatory(



                                        stage="five_pattern_anchor_registered",



                                        round_index=round_index,



                                        policy_id=policy_id,



                                        trace_id=str(getattr(finalized[0], "trace_id", "") or ""),



                                        request=request,



                                        routing=(



                                            dict(pattern_anchor_routing)



                                            if isinstance(pattern_anchor_routing, Mapping)



                                            else {"registered": bool(pattern_anchor_routing)}



                                        ),



                                        source="five_bridge_pre_write",



                                    )



                                self._write_action_batch(



                                    request_path.with_name(self.config.action_file),



                                    request,



                                    packet_rows,



                                    risk_config=self._risk_config(),



                                    order_config=self._order_config(),



                                )



                                for five_trace in finalized:



                                    self._write_action_metadata(



                                        request_path.with_name(self.config.action_file),



                                        request,



                                        five_trace,



                                    )



                                if getattr(request, "regime_probabilities", None) is not None:



                                    self._record_regime_telemetry(



                                        request,



                                        request.regime_probabilities,



                                        getattr(request, "regime_name", "CHOP") or "CHOP",



                                        int(getattr(request, "regime_index", 2) or 2),



                                        report_path=report_path,



                                        event_type="step",



                                    )



                                pending_five_actions = tuple({



                                    "agent_id": int(row.agent_id),



                                    "trace": trace,



                                    "action": int(row.action),



                                    "score": float(row.score),



                                    "confidence": float(row.confidence),



                                    "equity": float(request.equity),



                                } for row, trace in zip(packet_rows, finalized))



                                pending_request = request



                                pending_trace = None



                                pending_equity = request.equity



                                pending_drawdown = request.drawdown_pct



                                pending_spread_points = request.spread_points



                                last_sequence = sequence



                                served += 1



                                if request.equity > 0.0:



                                    equity_history.append(float(request.equity))



                                    equity_history = equity_history[-64:]



                                last_request_seen = time.monotonic()



                                if served <= 5 or served % 100 == 0:



                                    summary = ",".join(



                                        f"a{int(row.agent_id)}={int(row.action)}" for row in packet_rows



                                    )



                                    print(



                                        f"[BridgeFive] served={served} seq={sequence} packet=[{summary}] "



                                        f"positions={request.positions} day_trades={request.day_trades}",



                                        flush=True,



                                    )



                                continue



                            trace = policy.act(request.inputs, goal=goal_context)



                            apply_entry_timing = getattr(policy, "apply_entry_timing_extrema_bias", None)



                            if callable(apply_entry_timing):



                                trace = apply_entry_timing(trace)



                            if bool(self.config.strict_model1_parity):



                                deterministic_trace_id = (



                                    f"{str(getattr(trace, 'ma_chuyen_gia', '') or 'policy')}"



                                    f"~model1-{int(sequence)}"



                                )



                                try:



                                    trace = replace(



                                        trace,



                                        trace_id=deterministic_trace_id,



                                        timestamp=float(sequence),



                                    )



                                except TypeError:



                                    setattr(trace, "trace_id", deterministic_trace_id)



                                    setattr(trace, "timestamp", float(sequence))



                            outputs = np.asarray(trace.selected_action, dtype=float)



                            trace_action_details = dict(getattr(trace, "action_details", {}) or {})



                            owner_valid, owner_reason = self._context_expert_owner_valid(policy, trace)



                            if owner_valid:



                                effective_allowed_actions = self._effective_context_actions(trace, allowed_actions)



                                if (



                                    outputs.size > FACTORIZED_ACTION_MARKER_OUTPUT_INDEX



                                    and float(outputs[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX]) >= FACTORIZED_ACTION_MARKER_THRESHOLD



                                    and any(



                                        int(item) in effective_allowed_actions



                                        for item in (Action.CLOSE_WORST, Action.CLOSE_BEST, Action.CLOSE_ALL)



                                    )



                                ):



                                    effective_allowed_actions = set(effective_allowed_actions)



                                    effective_allowed_actions.add(int(Action.MANAGE_POSITION))



                                policy_action, _ = self._choose_action(outputs)



                                legal_action, _ = self._choose_action(outputs, effective_allowed_actions)



                                policy_vetoed = bool(



                                    trace_action_details.get("learner_recovery_negative_veto_applied", False)



                                    or trace_action_details.get("learner_recovery_negative_veto_to_hold", False)



                                )



                                action, execution_bootstrap = self._bootstrap_execution_action(



                                    request,



                                    legal_action,



                                    starvation_preview,



                                    effective_allowed_actions,



                                    policy_vetoed=policy_vetoed,



                                    policy_vetoed_action=trace_action_details.get(



                                        "learner_recovery_negative_veto_from"



                                    ),



                                    probe_details={



                                        **dict(getattr(trace, "context_audit", {}) or {}),



                                        **trace_action_details,



                                    },



                                )



                            else:



                                trace = self._gan_ly_do_chan_owner(trace, owner_reason)



                                outputs = np.zeros(OUTPUT_COUNT, dtype=float)



                                outputs[int(Action.HOLD)] = 1.0



                                policy_action = NO_NEURAL_COMMAND



                                legal_action = NO_NEURAL_COMMAND



                                action = NO_NEURAL_COMMAND



                                execution_bootstrap = False



                                effective_allowed_actions = {int(Action.HOLD)}



                            # `action` is internal and may be -1.  The EA still



                            # needs 0 for a no-op, but the policy trace retains



                            # its raw non-HOLD logits and never gets a HOLD



                            # action inserted as feedback.



                            wire_action = int(Action.HOLD) if int(action) < 0 else int(action)



                            wire_outputs = np.asarray(outputs, dtype=float).copy()



                            if int(action) >= 0 and int(action) != int(legal_action):



                                wire_outputs = self._force_action_output(wire_outputs, int(action))



                            elif int(action) < 0:



                                wire_outputs = self._force_action_output(wire_outputs, int(Action.HOLD))



                            wire_outputs = self._normalize_pending_offset(wire_outputs, wire_action)



                            score = float(outputs[action]) if 0 <= int(action) < outputs.size else 0.0



                            # Do not overwrite trace.selected_action with a



                            # wire-level HOLD vector. Only actual commands



                            # receive a bridge action attribution.



                            action_details = self._decoded_action_details(wire_outputs, request.positions)



                            training_entry_probe = bool(



                                execution_bootstrap and int(action) in self._entry_actions()



                            )



                            action_origin = (



                                "no_command" if int(action) < 0 else



                                "policy_exploration" if training_entry_probe else



                                "bootstrap" if bool(execution_bootstrap) else



                                ("mask" if int(policy_action) != int(legal_action) else "policy")



                            )



                            flat_idle_starvation_observed = bool(



                                int(policy_action) == int(Action.HOLD)



                                and int(getattr(request, "positions", 0) or 0) <= 0



                                and not bool(str(getattr(request, "orders_pipe", "") or "").strip())



                            )



                            if flat_idle_starvation_observed:



                                starvation_state.observe_flat_idle()



                            if int(action) < 0:



                                action_details["action_name"] = "NO_COMMAND"



                            action_details.update({



                                "policy_action": int(policy_action),



                                "legal_action": int(legal_action),



                                "emitted_action": int(action),



                                "bridge_action": int(action),



                                "wire_action": int(wire_action),



                                # A wire 0 is transport only.  Explicit neural HOLD



                                # is retained as a policy decision but is never



                                # eligible for MT5 action/lifecycle attribution when



                                # the account remains flat.



                                "wire_noop": bool(int(action) < 0),



                                "no_neural_command": bool(int(action) < 0),



                                "flat_idle_non_feedback": bool(int(action) in {int(Action.HOLD), NO_NEURAL_COMMAND}),



                                "flat_idle_starvation_observed": flat_idle_starvation_observed,



                                "flat_idle_no_trade_streak": int(starvation_state.no_trade_streak),



                                "action_attribution_source": (



                                    "none" if int(action) < 0 else



                                    "hold_no_execution" if int(action) == int(Action.HOLD) else



                                    "emitted"



                                ),



                                "model_hypothesis_index": int(getattr(trace, "selected_idx", -1)),



                                "action_masked": bool(int(policy_action) >= 0 and int(policy_action) != int(legal_action)),



                                "execution_bootstrap": bool(execution_bootstrap),



                                "training_entry_probe": training_entry_probe,



                                "training_entry_probe_source": str(self._train_entry_probe_last_source),



                                "effective_allowed_actions": sorted(int(item) for item in effective_allowed_actions),



                                "context_contract_enforced_at_wire": True,



                                "pending_recovery_cancel": bool(execution_bootstrap and int(action) == int(Action.CANCEL_ORDERS) and bool(str(getattr(request, "orders_pipe", "") or "").strip())),



                                "action_origin": action_origin,



                                "five_lifecycle_routing": self._five_lifecycle_routing_snapshot(),



                                "five_failure_replay": self._five_failure_replay_snapshot(),



                                **_direct_rnn_actor_bridge_audit(trace_action_details, int(action)),



                                # Keep the raw decision-time account mark



                                # alongside normalized policy features.  The



                                # Codex account-impact verifier needs the



                                # immutable balance/equity pair to compare



                                # against the final-close mark; normalized



                                # fields alone cannot establish a delta.



                                "codex_decision_account": {



                                    "balance": float(getattr(request, "balance", 0.0) or 0.0),



                                    "equity": float(getattr(request, "equity", 0.0) or 0.0),



                                    "drawdown_pct": float(getattr(request, "drawdown_pct", 0.0) or 0.0),



                                    "positions": float(getattr(request, "positions", 0) or 0),



                                    "sequence": float(getattr(request, "sequence", 0) or 0),



                                },



                            })



                            codex_next_action_links = self._consume_codex_teacher_next_action_links(



                                policy=policy,



                                request=request,



                                trace_id=str(getattr(trace, "trace_id", "") or ""),



                                action=int(action),



                                score=float(score),



                            )



                            if codex_next_action_links:



                                action_details.update(



                                    self._codex_teacher_action_link_details(



                                        codex_next_action_links



                                    )



                                )



                            if runtime_guard_active:



                                action_details.update({



                                    "runtime_fitness_guard_active": True,



                                    "runtime_fitness_guard_reason": str(runtime_guard.active_reason),



                                    "runtime_fitness_guard": runtime_guard.snapshot(),



                                })



                            pending_trace = enrich_step_trace(



                                trace,



                                action_details,



                                request.positions_pipe,



                                account_context=account_context,



                            )



                            self._update_position_context_extrema(



                                policy=policy,



                                request=request,



                                trace=pending_trace,



                            )



                            starvation = starvation_state.preview(



                                request,



                                self.config.min_daily_trades,



                                suppress_flat_idle_feedback=flat_idle_firebreak,



                            )



                            starvation["opened_or_filled"] = 0.0



                            pending_trace = attach_experience_starvation(pending_trace, starvation)



                            # Publish exactly the immutable bridge decision.



                            # In particular, never expose a pre-bridge trace



                            # as ``NO_COMMAND`` when an entry is actually



                            # written to MT5; geometry and action telemetry



                            # must describe the same decision surface.



                            finalize_bridge_decision = getattr(policy, "finalize_bridge_decision", None)



                            if callable(finalize_bridge_decision):



                                pending_trace = finalize_bridge_decision(pending_trace)



                            if callable(set_starvation):



                                set_starvation(starvation)



                            trace_id = str(getattr(trace, "trace_id", f"trace-{sequence}"))



                            confidence = float(getattr(trace, "confidence", 0.0))



                        else:



                            raise RuntimeError("Bridge chỉ chấp nhận HeChuyenGiaMotContextMotPolicy đã xác minh owner")



                        preference_routing = {"registered": False, "reason": "not_entry_action_surface"}



                        preference_action = int(



                            wire_action if 'wire_action' in locals() else action



                        )



                        # V187: cache every decision on the direct entry



                        # surface, including HOLD when candidate coverage is



                        # enabled. Otherwise a learned HOLD removes its own



                        # state region from delayed supervision and the shared



                        # representation can drift back toward TRADE. Existing



                        # portfolio state remains outside this immutable path



                        # replay; management actions 7..11 are never included.



                        if self._entry_preference_snapshot_eligible(



                            preference_action,



                            include_hold=bool(



                                self.config.entry_preference_include_hold_snapshots



                            ),



                        ):



                            preference_routing = self._register_entry_preference_trace(



                                policy,



                                pending_trace if 'pending_trace' in locals() else None,



                            )



                        intent_routing = self._register_execution_intent_trace(



                            policy,



                            pending_trace if 'pending_trace' in locals() else None,



                        )



                        self._append_execution_trace_observatory(



                            stage="intent_registered",



                            round_index=round_index,



                            policy_id=policy_id,



                            trace=pending_trace if 'pending_trace' in locals() else None,



                            trace_id=str(trace_id),



                            request=request,



                            routing=intent_routing,



                            source="bridge_pre_write",



                        )



                        if self._entry_preference_snapshot_eligible(



                            preference_action,



                            include_hold=bool(



                                self.config.entry_preference_include_hold_snapshots



                            ),



                        ):



                            # Failed cache admission must be observable. The EA



                            # will still mature the immutable row, but Python



                            # cannot train it later without this exact trace.



                            self._append_execution_trace_observatory(



                                stage=(



                                    "entry_preference_registered"



                                    if bool(preference_routing.get("registered", False))



                                    else "entry_preference_registration_failed"



                                ),



                                round_index=round_index,



                                policy_id=policy_id,



                                trace=pending_trace if 'pending_trace' in locals() else None,



                                trace_id=str(trace_id),



                                request=request,



                                routing=preference_routing,



                                source="bridge_pre_write",



                            )



                        self._write_action(



                            request_path.with_name(self.config.action_file),



                            request,



                            trace_id,



                            wire_action if 'wire_action' in locals() else action,



                            score,



                            confidence,



                            (



                                np.round(



                                    np.asarray(wire_outputs if 'wire_outputs' in locals() else outputs, dtype=float),



                                    int(self.config.output_wire_decimals),



                                )



                                if bool(self.config.strict_model1_parity)



                                else (wire_outputs if 'wire_outputs' in locals() else outputs)



                            ),



                            risk_config=self._risk_config(),



                            order_config=self._order_config(),



                            trace=pending_trace if 'pending_trace' in locals() else None,



                        )



                        if replay_action_observer is not None:



                            action_path = request_path.with_name(self.config.action_file) if request_path is not None else self.mql5_files / self.config.action_file



                            replay_action_observer(request, action_path)



                        self._append_execution_trace_observatory(



                            stage="intent_written",



                            round_index=round_index,



                            policy_id=policy_id,



                            trace=pending_trace if 'pending_trace' in locals() else None,



                            trace_id=str(trace_id),



                            request=request,



                            routing=intent_routing,



                            source="bridge_write",



                        )



                        self._write_action_metadata(



                            request_path.with_name(self.config.action_file),



                            request,



                            pending_trace if 'pending_trace' in locals() else None,



                        )



                        if getattr(request, "regime_probabilities", None) is not None:



                            self._record_regime_telemetry(



                                request,



                                request.regime_probabilities,



                                getattr(request, "regime_name", "CHOP") or "CHOP",



                                int(getattr(request, "regime_index", 2) or 2),



                                report_path=report_path,



                                event_type="step",



                            )



                        emit_heartbeat = getattr(policy, "emit_bridge_heartbeat", None)



                        if callable(emit_heartbeat):



                            detail = dict(getattr(pending_trace, "action_details", {}) or {}) if pending_trace is not None else {}



                            emit_heartbeat(



                                stage="action_written",



                                sequence=int(sequence),



                                request_event_id=str(getattr(request, "event_id", "")),



                                trace_id=str(trace_id),



                                action=int(action),



                                metadata={



                                    "wire_action": int(wire_action if 'wire_action' in locals() else action),



                                    "policy_action": int(detail.get("policy_action", action) or action),



                                    "legal_action": int(detail.get("legal_action", action) or action),



                                    "emitted_action": int(detail.get("emitted_action", action) or action),



                                    "execution_bootstrap": bool(detail.get("execution_bootstrap", False)),



                                    "action_masked": bool(detail.get("action_masked", False)),



                                    "hold_reason": str(detail.get("hold_reason", "") or ""),



                                    "learner_recovery": dict(detail.get("learner_recovery", {}) or {}),



                                    "runtime_fitness_guard": runtime_guard.snapshot(),



                                },



                            )



                        pending_equity = request.equity



                        pending_drawdown = request.drawdown_pct



                        pending_spread_points = request.spread_points



                        pending_request = request



                        pending_action = int(action)



                        pending_wire_action = int(wire_action if 'wire_action' in locals() else action)



                        pending_score = score



                        pending_confidence = confidence



                        pending_starvation = dict(getattr(pending_trace, "experience_starvation", {}) or {}) if pending_trace is not None else {}



                        last_sequence = sequence



                        served += 1



                        if request.equity > 0.0:



                            equity_history.append(float(request.equity))



                            equity_history = equity_history[-64:]



                        last_request_seen = time.monotonic()



                        if served <= 5 or served % 100 == 0:



                            action_name = Action(action).name if action in {int(item) for item in Action} else str(action)



                            detail = dict(getattr(pending_trace, "action_details", {}) or {}) if pending_trace is not None else {}



                            print(



                                f"[Bridge] served={served} seq={sequence} action={action}({action_name}) wire={pending_wire_action} "



                                f"policy_action={detail.get('policy_action', policy_action)} "



                                f"masked={int(bool(detail.get('action_masked', False)))} "



                                f"bootstrap={int(bool(detail.get('execution_bootstrap', False)))} "



                                f"positions={request.positions} day_trades={request.day_trades}",



                                flush=True,



                            )



            else:



                debug_counter += 1



                if debug_counter % 200 == 0:



                    elapsed = time.monotonic() - run_started



                    print(f"[Bridge] waiting for request... served={served} elapsed={elapsed:.1f}s", flush=True)



            located_report = self._materialize_report(report_name, report_path)



            poll_status = process.poll()



            if located_report is None and poll_status is not None:



                located_report = self._materialize_report(report_name, report_path)



            if self._tester_report_ready_for_finalize(



                located_report,



                process_poll_status=poll_status,



                now=time.monotonic(),



                last_request_seen=last_request_seen,



            ):



                if learning_enabled and pending_five_actions and hasattr(policy, "observe_transition"):



                    for pending_item in pending_five_actions:



                        five_trace = pending_item["trace"]



                        if self._trace_learning_blocked(five_trace):



                            continue



                        policy.observe_transition(



                            five_trace,



                            last_inputs,



                            0.0,



                            0.0,



                            True,



                        )



                    pending_five_actions = ()



                if (



                    learning_enabled



                    and pending_trace is not None



                    and hasattr(policy, "observe_transition")



                    and not False



                    and not self._trace_learning_blocked(pending_trace)



                ):



                    execution_details = self._execution_details(pending_request, None, pending_action)



                    if not bool(execution_details.get("flat_idle", False)):



                        pending_trace = enrich_step_trace(



                            pending_trace,



                            dict(getattr(pending_trace, "action_details", {}) or {}),



                            str(getattr(pending_trace, "positions_pipe", "")),



                            str(getattr(pending_trace, "positions_pipe_after", "")),



                            execution_details=execution_details,



                        )



                        signal = policy.observe_transition(pending_trace, last_inputs, 0.0, 0.0, True)



                    else:



                        signal = {"memory_eligible": False}



                    if bool(signal.get("memory_eligible", execution_details["memory_eligible"])) and pending_request is not None:



                        self._append_experience_journal(



                            round_index=round_index,



                            policy_id=policy_id,



                            request=pending_request,



                            trace=pending_trace,



                            next_obs=last_inputs,



                            reward=0.0,



                            cost=0.0,



                            done=True,



                            signal=signal,



                            action=pending_action,



                            score=pending_score,



                            confidence=pending_confidence,



                        )



                settle_entry_preference_updates()



                return served



            if poll_status is not None and located_report is None and not self._runner_terminal_is_alive():



                raise RuntimeError(f"MT5 exited before writing report: {report_path}")



            fresh_request_path = self._latest_runtime_file(self.config.request_file)



            try:



                fresh_request_has_payload = bool(



                    fresh_request_path is not None



                    and fresh_request_path.is_file()



                    and fresh_request_path.stat().st_size > 0



                )



            except OSError:



                fresh_request_has_payload = False



            if (



                served == 0



                and poll_status is not None



                and located_report is None



                and self._runner_terminal_is_alive()



                and time.monotonic() - run_started > real_tick_startup_grace_sec



                # ``_clear_bridge_files`` creates an empty request file before



                # launch. Existence alone therefore cannot prove that the live



                # terminal started a tester session; require a non-empty fresh



                # request payload before treating the runner as active.



                and not fresh_request_has_payload



            ):



                raise RuntimeError(



                    f"MT5 launcher was forwarded to a live runner terminal, but no fresh bridge request arrived after {real_tick_startup_grace_sec:.0f}s; "



                    "the terminal was idle or running an unrelated tester session."



                )



            if (
                located_report is not None
                and time.monotonic() - last_request_seen > 2.0
            ):



                try:



                    parse_mt5_report(report_path)



                    settle_entry_preference_updates()



                    return served



                except Exception:



                    pass



            time.sleep(0.05)







    @staticmethod



    def _close_terminal_after_trial(process: subprocess.Popen) -> None:



        try:



            process.wait(timeout=30)



            return



        except subprocess.TimeoutExpired:



            process.terminate()



        try:



            process.wait(timeout=5)



        except subprocess.TimeoutExpired:



            process.kill()



            process.wait(timeout=5)







    def evaluate_policy(self, policy_id: int, policy, round: int) -> tuple[float, MT5BacktestMetrics]:



        """Evaluate one policy while protecting the fixed MT5 transport ports."""



        if bool(getattr(self.config, "offline_replay", False)):



            self._last_mt5_global_lock_wait_seconds = 0.0



            return self._evaluate_policy_unlocked(policy_id, policy, round)



        with MT5StrategyTesterGlobalLock(self.mt5_global_lock_path) as lock:



            self._last_mt5_global_lock_wait_seconds = float(lock.waited_seconds)



            if lock.waited_seconds >= 0.5:



                print(



                    f"[MT5] waited {lock.waited_seconds:.2f}s for global tester lock "



                    f"path={self.mt5_global_lock_path}",



                    flush=True,



                )



            return self._evaluate_policy_unlocked(policy_id, policy, round)







    def _evaluate_policy_unlocked(self, policy_id: int, policy, round: int) -> tuple[float, MT5BacktestMetrics]:



        # Evaluation chỉ phục vụ kho context-owner; không có session policy chung.



        set_live_trial = getattr(policy, "set_live_observatory_trial", None)



        if callable(set_live_trial) and not False:



            set_live_trial(policy_id=int(policy_id), round_index=int(round), phase="mt5")



        # MT5 portable is effectively single-instance.  If a prior Visual tester



        # window was closed by the operator while its terminal survived, launching



        # another /config command is forwarded to that old terminal and the next



        # pass often runs without Visual.  A Visual pass therefore starts only



        # after every tracked terminal has been closed.



        if self.config.visual or bool(getattr(self.config, "shutdown_terminal", True)):



            self.close_open_terminals()



            self.close_runner_processes()



        self._verify_ea_build_integrity()



        self._close_stale_tester_agents()



        started_at_ns = time.time_ns()



        self._last_stop_request = {



            "requested": False,



            "path": str(self.config.stop_request_file or ""),



            "marker_file": str(self.config.stop_request_marker_file or ""),



        }



        self._report_min_mtime_ns = started_at_ns



        self._clear_bridge_files()



        # Runtime files are removed first and then recreated above. Use the



        # filesystem's own timestamp domain as the freshness floor instead of



        # comparing it directly with time.time_ns(), whose sub-millisecond



        # resolution can be ahead of the filesystem clock on some systems.



        seed_runtime_files = [



            self.common_files / self.config.execution_file,



            self.mql5_files / self.config.execution_file,



            self.common_files / self.config.request_file,



            self.mql5_files / self.config.request_file,



        ]



        seed_mtimes = [item.stat().st_mtime_ns for item in seed_runtime_files if item.exists()]



        self._runtime_file_min_mtime_ns = min(seed_mtimes) if seed_mtimes else 0



        self._clear_tester_cache()



        set_path = write_tester_set(



            self.config,



            self.runner_dir / "MQL5" / "Profiles" / "Tester" / f"{self.config.expert}.set",



        )



        report_name = f"he_chuyen_gia_mt5_round_{round:04d}_evaluation_{policy_id}"



        # Keep the launcher INI outside MQL5.  MT5 can refresh/replace the



        # portable MQL5 tree during startup even when /skipupdate is present;



        # a follow-up failure-replay launch then loses its /config file before



        # the terminal reads it.  The absolute /config path works from a



        # runner-owned directory that LiveUpdate does not rewrite.



        ini_path = (



            self.runner_dir



            / "codex_run_configs"



            / f"round_{round:04d}_evaluation_{policy_id}.ini"



        )



        (self.runner_dir / "Reports").mkdir(parents=True, exist_ok=True)



        self._clear_report_artifacts(report_name)



        write_tester_ini(self.config, report_name, ini_path)



        report_path = self.runner_dir / "Reports" / f"{report_name}.htm"



        # A portable OOS clone may coexist with the operator terminal.  Skip



        # the shared live-update helper so two installations do not contend on



        # the same updater binary before the tester starts.



        command = [str(self.terminal), "/portable", f"/config:{ini_path}", "/skipupdate"]



        run_label = f"round={round} he_chuyen_gia_evaluation={policy_id}"



        manifest_path = self._write_session_manifest(



            report_name=report_name,



            policy_id=int(policy_id),



            round_index=int(round),



            ini_path=ini_path,



            set_path=set_path,



            report_path=report_path,



            policy=policy,



            stage="started",



            extra={



                "command": command,



                "run_label": run_label,



                "save_and_stop": dict(self._last_stop_request),



                "mt5_global_lock_path": str(self.mt5_global_lock_path),



                "mt5_global_lock_wait_seconds": float(self._last_mt5_global_lock_wait_seconds),



            },



        )



        print(



            f"[MT5] {run_label} start visual={int(self.config.visual)} "



            f"shutdown={int(self.config.shutdown_terminal)} report={report_path}",



            flush=True,



        )



        tester_started_at_ns = time.time_ns()



        self._active_tester_ini_name = ini_path.name.lower()



        if bool(self.config.visual):
            try:
                for tpl_dir in [
                    self.runner_dir / "Profiles" / "Templates",
                    self.runner_dir / "MQL5" / "Profiles" / "Templates",
                ]:
                    tpl_dir.mkdir(parents=True, exist_ok=True)
                    tester_tpl = tpl_dir / "tester.tpl"
                    if not tester_tpl.exists():
                        tester_tpl_content = (
                            "<chart>\n"
                            f"symbol={self.config.symbol}\n"
                            "period_type=1\n"
                            "period_size=1\n"
                            "digits=2\n"
                            "scale=8\n"
                            "mode=1\n"
                            "fore=0\n"
                            "grid=1\n"
                            "volume=1\n"
                            "scroll=1\n"
                            "shift=1\n"
                            "trade_history=1\n"
                            "trade_levels=1\n"
                            "background_color=0\n"
                            "foreground_color=16777215\n"
                            "barup_color=65280\n"
                            "bardown_color=255\n"
                            "bullcandle_color=65280\n"
                            "bearcandle_color=255\n"
                            "</chart>\n"
                        )
                        tester_tpl.write_text(tester_tpl_content, encoding="utf-16")
            except Exception:
                pass

        startupinfo = None
        if os.name == "nt" and bool(self.config.visual):
            try:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 3
            except Exception:
                startupinfo = None

        if os.name == "nt" and bool(self.config.visual):
            class _WinDesktopProcess:
                def __init__(self, command, cwd=None, desktop="WinSta0\\Default", show_window=3):
                    import ctypes
                    from ctypes import wintypes
                    kernel32 = ctypes.windll.kernel32

                    if isinstance(command, (list, tuple)):
                        cmd_str = subprocess.list2cmdline([str(c) for c in command])
                    else:
                        cmd_str = str(command)

                    class _STARTUPINFOW(ctypes.Structure):
                        _fields_ = [
                            ("cb", wintypes.DWORD),
                            ("lpReserved", wintypes.LPWSTR),
                            ("lpDesktop", wintypes.LPWSTR),
                            ("lpTitle", wintypes.LPWSTR),
                            ("dwX", wintypes.DWORD),
                            ("dwY", wintypes.DWORD),
                            ("dwXSize", wintypes.DWORD),
                            ("dwYSize", wintypes.DWORD),
                            ("dwXCountChars", wintypes.DWORD),
                            ("dwYCountChars", wintypes.DWORD),
                            ("dwFillAttribute", wintypes.DWORD),
                            ("dwFlags", wintypes.DWORD),
                            ("wShowWindow", wintypes.WORD),
                            ("cbReserved2", wintypes.WORD),
                            ("lpReserved2", ctypes.c_void_p),
                            ("hStdInput", wintypes.HANDLE),
                            ("hStdOutput", wintypes.HANDLE),
                            ("hStdError", wintypes.HANDLE),
                        ]

                    class _PROCESS_INFORMATION(ctypes.Structure):
                        _fields_ = [
                            ("hProcess", wintypes.HANDLE),
                            ("hThread", wintypes.HANDLE),
                            ("dwProcessId", wintypes.DWORD),
                            ("dwThreadId", wintypes.DWORD),
                        ]

                    si = _STARTUPINFOW()
                    si.cb = ctypes.sizeof(_STARTUPINFOW)
                    si.lpDesktop = desktop
                    si.dwFlags = 0x00000001
                    si.wShowWindow = show_window

                    pi = _PROCESS_INFORMATION()
                    cwd_str = str(cwd) if cwd else None

                    success = kernel32.CreateProcessW(
                        None,
                        cmd_str,
                        None,
                        None,
                        False,
                        0,
                        None,
                        cwd_str,
                        ctypes.byref(si),
                        ctypes.byref(pi),
                    )
                    if not success:
                        err = kernel32.GetLastError()
                        raise OSError(f"CreateProcessW failed with error code {err}")

                    self.hProcess = pi.hProcess
                    self.hThread = pi.hThread
                    self.pid = pi.dwProcessId
                    self.returncode = None
                    kernel32.CloseHandle(self.hThread)

                def poll(self):
                    if self.returncode is not None:
                        return self.returncode
                    import ctypes
                    from ctypes import wintypes
                    code = wintypes.DWORD()
                    if ctypes.windll.kernel32.GetExitCodeProcess(self.hProcess, ctypes.byref(code)):
                        if code.value == 259:
                            return None
                        self.returncode = code.value
                        return self.returncode
                    return None

                def wait(self, timeout=None):
                    import subprocess
                    ms = int(timeout * 1000) if timeout is not None else 0xFFFFFFFF
                    import ctypes
                    res = ctypes.windll.kernel32.WaitForSingleObject(self.hProcess, ms)
                    if res == 0x00000102:
                        raise subprocess.TimeoutExpired(self.pid, timeout)
                    return self.poll()

                def terminate(self):
                    import ctypes
                    ctypes.windll.kernel32.TerminateProcess(self.hProcess, 1)
                    self.poll()

                def kill(self):
                    self.terminate()

                def __del__(self):
                    try:
                        if hasattr(self, "hProcess") and self.hProcess:
                            import ctypes
                            ctypes.windll.kernel32.CloseHandle(self.hProcess)
                    except Exception:
                        pass

            try:
                process = _WinDesktopProcess(
                    command,
                    cwd=str(self.runner_dir),
                    desktop="WinSta0\\Default",
                    show_window=3,
                )
            except Exception as _e:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.runner_dir),
                    startupinfo=startupinfo,
                )
        else:
            process = subprocess.Popen(
                command,
                cwd=str(self.runner_dir),
                startupinfo=startupinfo,
            )

        if os.name == "nt" and bool(self.config.visual):
            import threading
            def _bring_to_foreground(pid: int):
                try:
                    import ctypes
                    from ctypes import wintypes
                    user32 = ctypes.windll.user32
                    kernel32 = ctypes.windll.kernel32

                    # Attach calling thread to interactive WinSta0\\Default desktop
                    try:
                        h_default_desk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
                        if h_default_desk:
                            user32.SetThreadDesktop(h_default_desk)
                    except Exception:
                        pass

                    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

                    def _get_family_pids(root_pid: int) -> set[int]:
                        pids = {root_pid}
                        try:
                            h_snap = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
                            if h_snap != -1:
                                class _PE32(ctypes.Structure):
                                    _fields_ = [
                                        ("dwSize", wintypes.DWORD),
                                        ("cntUsage", wintypes.DWORD),
                                        ("th32ProcessID", wintypes.DWORD),
                                        ("th32DefaultHeapID", ctypes.c_size_t),
                                        ("th32ModuleID", wintypes.DWORD),
                                        ("cntThreads", wintypes.DWORD),
                                        ("th32ParentProcessID", wintypes.DWORD),
                                        ("pcPriClassBase", wintypes.LONG),
                                        ("dwFlags", wintypes.DWORD),
                                        ("szExeFile", ctypes.c_char * 260),
                                    ]
                                pe = _PE32()
                                pe.dwSize = ctypes.sizeof(_PE32)
                                if kernel32.Process32First(h_snap, ctypes.byref(pe)):
                                    parent_map = {}
                                    while True:
                                        parent_map[pe.th32ProcessID] = pe.th32ParentProcessID
                                        if not kernel32.Process32Next(h_snap, ctypes.byref(pe)):
                                            break
                                    changed = True
                                    while changed:
                                        changed = False
                                        for c_id, p_id in parent_map.items():
                                            if p_id in pids and c_id not in pids:
                                                pids.add(c_id)
                                                changed = True
                                kernel32.CloseHandle(h_snap)
                        except Exception:
                            pass
                        return pids

                    def _activate_window(h: int, title: str):
                        try:
                            curr_tid = kernel32.GetCurrentThreadId()
                            fg_hwnd = user32.GetForegroundWindow()
                            fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
                            target_tid = user32.GetWindowThreadProcessId(h, None)
                            if fg_tid and fg_tid != curr_tid:
                                user32.AttachThreadInput(curr_tid, fg_tid, True)
                            if target_tid and target_tid != curr_tid:
                                user32.AttachThreadInput(curr_tid, target_tid, True)

                            # Restore & Maximize
                            if user32.IsIconic(h):
                                user32.ShowWindow(h, 9)  # SW_RESTORE
                            user32.ShowWindow(h, 3)  # SW_MAXIMIZE

                            # Bring to top via SetWindowPos HWND_TOPMOST then HWND_NOTOPMOST
                            user32.SetWindowPos(h, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0040)
                            user32.SetWindowPos(h, -2, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0040)

                            # Bypass foreground lock via ALT key simulation
                            user32.keybd_event(0x12, 0, 0, 0)
                            user32.BringWindowToTop(h)
                            user32.SetForegroundWindow(h)
                            user32.keybd_event(0x12, 0, 2, 0)

                            if hasattr(user32, "SwitchToThisWindow"):
                                user32.SwitchToThisWindow(h, True)

                            if target_tid and target_tid != curr_tid:
                                user32.AttachThreadInput(curr_tid, target_tid, False)
                            if fg_tid and fg_tid != curr_tid:
                                user32.AttachThreadInput(curr_tid, fg_tid, False)
                        except Exception:
                            pass

                    visual_activated_count = 0
                    announced = set()

                    # Monitor for up to 90 seconds (180 * 0.5s)
                    for step in range(180):
                        time.sleep(0.5)
                        family_pids = _get_family_pids(pid)
                        found = []

                        def _cb(hwnd, _l):
                            p = wintypes.DWORD()
                            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
                            cls_buf = ctypes.create_unicode_buffer(256)
                            user32.GetClassNameW(hwnd, cls_buf, 256)
                            cls = cls_buf.value
                            length = user32.GetWindowTextLengthW(hwnd)
                            buf = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buf, length + 1)
                            title = buf.value

                            is_target = False
                            if p.value in family_pids:
                                is_target = True
                            elif "MetaQuotes::MetaTester" in cls or "MetaQuotes::MetaTrader" in cls:
                                is_target = True
                            elif "Strategy Tester Visualization" in title or "Tester" in title or "BTCUSDm" in title:
                                is_target = True

                            if is_target and title not in ("Default IME", "MSCTFIME UI") and not title.startswith("GDI+"):
                                found.append((hwnd, title, cls))
                            return True

                        user32.EnumWindows(WNDENUMPROC(_cb), 0)

                        # Also check WinSta0\Default desktop specifically if needed
                        try:
                            h_desk = user32.OpenDesktopW("Default", 0, False, 0x0040 | 0x0001)
                            if h_desk:
                                user32.EnumDesktopWindows(h_desk, WNDENUMPROC(_cb), 0)
                                user32.CloseDesktop(h_desk)
                        except Exception:
                            pass

                        for h, title, cls in found:
                            is_visual = ("MetaTester" in cls or "Visualization" in title or "Tester" in title)
                            _activate_window(h, title)
                            if h not in announced and (is_visual or "MetaTrader" in cls):
                                announced.add(h)
                                print(f"[MT5 GUI] Da kich hoat va dua cua so len man hinh: '{title}' (HWND: {h})", flush=True)
                            if is_visual:
                                visual_activated_count += 1

                        # Once Strategy Tester Visualization has been firmly kept on top for 6 passes (3s), we can stop
                        if visual_activated_count >= 6:
                            break
                except Exception:
                    pass
            threading.Thread(target=_bring_to_foreground, args=(process.pid,), daemon=True).start()



        self._launched_processes.append(process)



        try:



            served_actions = self._serve_policy_actions(process, report_path, policy, policy_id, round)



            self._five_action_distribution = self._five_action_distribution_snapshot()



            if bool(getattr(self.config, "require_action_bridge", True)) and int(served_actions) <= 0:



                raise RuntimeError(



                    "MT5 action bridge produced zero policy actions; "



                    "rejecting no-op tester report"



                )



        except Exception:



            # A timeout or bridge failure can occur before the normal



            # finished-evaluation footer. Preserve the real RLHF phase



            # receipt so the caller does not mistake an interrupted enabled



            # run for a run where RLHF was disabled.



            self._snapshot_rlhf_policy_update_manifest()



            if bool(getattr(self.config, "shutdown_terminal", True)):
                if process.poll() is None and hasattr(process, "terminate"):



                    process.terminate()



                    try:



                        process.wait(timeout=5)



                    except subprocess.TimeoutExpired:



                        if hasattr(process, "kill"):



                            process.kill()



                self._close_stale_tester_agents()



                self.close_runner_processes()



            raise



        if self.config.shutdown_terminal:



            self._close_terminal_after_trial(process)



            # The /config launcher can exit while the portable terminal stays



            # alive.  A second evaluation on the same runner would then be



            # forwarded to that idle terminal and never produce a fresh



            # bridge request.  Close the runner-scoped processes before



            # parsing/returning so sequential frozen validation is isolated.



            # Test doubles use an empty ``terminal64.exe`` placeholder; the



            # real portable terminal is a large executable.  Use the binary



            # marker as a second signal because WMI can briefly miss a live



            # process while MT5 is handing off the /config launch.



            try:



                real_terminal_binary = bool(



                    self.terminal.is_file() and self.terminal.stat().st_size > 1_048_576



                )



            except OSError:



                real_terminal_binary = False



            if real_terminal_binary:



                self.close_runner_processes()



        metrics = parse_mt5_report(report_path)



        bridge_path = self._bridge_path()



        bridge_terminal_mark = parse_bridge_terminal_mark(bridge_path)



        if bridge_path is not None:



            metrics = replace(



                metrics,



                bridge_path=bridge_path,



                bridge_rows=count_bridge_rows(bridge_path),



                bridge_terminal_mark=bridge_terminal_mark,



            )



        execution_path = self._latest_runtime_file(self.config.execution_file)



        execution_summary = parse_mt5_execution_ledger(



            execution_path,



            initial_equity=self.config.deposit,



            min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



            technical_min_rr=float(self.config.technical_min_rr),



        )



        # MT5 can finish a tester pass after the final DEAL_ADD has left the



        # bridge stream.  Parse the fresh HTML report as a second, explicit



        # lifecycle source and join it only when its close rows account for



        # the report's total PnL/trade count.  Native ledger lifecycles remain



        # authoritative whenever a unique PnL/time match exists.



        report_trace_by_sequence = execution_trace_by_sequence(execution_path)



        report_lifecycles = parse_mt5_report_lifecycles(



            report_path,



            trace_by_sequence=report_trace_by_sequence,



            initial_equity=float(self.config.deposit),



            bar_seconds=60.0,



        )



        report_lifecycle_net = float(sum(item.net_pnl for item in report_lifecycles))



        report_lifecycle_tolerance = max(



            0.0,



            float(self.config.reconciliation_pnl_absolute_tolerance),



        ) + max(0.0, float(self.config.reconciliation_pnl_relative_tolerance)) * max(



            abs(float(metrics.net_profit)), abs(report_lifecycle_net), 1.0



        )



        report_lifecycle_join_accepted = bool(



            report_lifecycles



            and len(report_lifecycles) == int(metrics.total_trades)



            and abs(float(metrics.net_profit) - report_lifecycle_net) <= report_lifecycle_tolerance



        )



        effective_lifecycles = tuple(execution_summary.lifecycles)



        if report_lifecycle_join_accepted:



            merged_lifecycles = merge_report_lifecycles(



                execution_summary.lifecycles,



                report_lifecycles,



                trace_by_sequence=report_trace_by_sequence,



            )



            effective_net = float(sum(item.net_pnl for item in merged_lifecycles))



            report_lifecycle_join_accepted = bool(



                len(merged_lifecycles) == len(report_lifecycles)



                and abs(float(metrics.net_profit) - effective_net) <= report_lifecycle_tolerance



            )



            if report_lifecycle_join_accepted:



                effective_lifecycles = merged_lifecycles



                execution_summary = replace_execution_summary_lifecycles(



                    execution_summary,



                    effective_lifecycles,



                    initial_equity=float(self.config.deposit),



                    min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



                )



            else:



                # A failed post-join check must not leak speculative report



                # rows into logs, fallback counts, or learning.



                effective_lifecycles = tuple(execution_summary.lifecycles)



        reconciliation = reconcile_mt5_report_and_ledger(



            metrics,



            execution_summary,



            pnl_relative_tolerance=self.config.reconciliation_pnl_relative_tolerance,



            pnl_absolute_tolerance=self.config.reconciliation_pnl_absolute_tolerance,



            terminal_open_positions=int(bridge_terminal_mark.get("open_positions", 0) or 0),



            terminal_open_pnl=float(bridge_terminal_mark.get("open_pnl", 0.0) or 0.0),



            report_lifecycle_count=len(report_lifecycles),



            report_lifecycle_net_profit=report_lifecycle_net,



        )



        reconciliation = replace(



            reconciliation,



            report_lifecycle_count=int(len(report_lifecycles)),



            report_lifecycle_net_profit=float(report_lifecycle_net),



            report_fallback_lifecycle_count=int(sum(



                bool(dict(item.metadata or {}).get("report_fallback", False))



                for item in effective_lifecycles



            )),



            report_fallback_owner_resolved_count=int(sum(



                bool(dict(item.metadata or {}).get("report_fallback", False))



                and bool(str(item.trace_id or "").strip())



                for item in effective_lifecycles



            )),



        )



        execution_summary = settle_terminal_open_position_lifecycles(



            execution_summary,



            execution_path,



            bridge_terminal_mark,



            initial_equity=float(self.config.deposit),



            min_holding_bars=max(1, min(5, int(self.config.max_holding_bars))),



        )



        if bool(self.config.codex_teacher_enabled):



            self._record_five_competition_lifecycles(execution_summary.lifecycles, policy=policy)



        feature_audit_path = (



            self._latest_runtime_file(self.config.feature_audit_file)



            if str(self.config.feature_audit_file or "").strip() else None



        )



        feature_audit = summarize_mt5_feature_audit(



            feature_audit_path,



            expected_schema_hash=str(self.config.feature_schema_hash),



            expected_point_size=float(self.config.symbol_point_size),



            require_broker_metadata=True,



        )



        if str(self.config.feature_reference_csv or "").strip():



            feature_audit = compare_mt5_feature_audit_against_reference(



                feature_audit_path or Path("__missing_feature_audit__"),



                self.config.feature_reference_csv,



                FeatureConfig(



                    atr_period=int(self.config.feature_atr_period),



                    ema_fast=int(self.config.feature_ema_fast),



                    ema_slow=int(self.config.feature_ema_slow),



                    range_window=int(self.config.feature_range_window),



                    regime_window=int(self.config.feature_regime_window),



                    atr_percentile_window=int(self.config.feature_atr_percentile_window),



                    max_spread_atr=float(self.config.max_spread_atr),



                    shock_range_atr=float(self.config.shock_range_atr),



                    wyckoff_range_window=int(self.config.feature_wyckoff_range_window),



                    wyckoff_volume_short_window=int(self.config.feature_wyckoff_volume_short_window),



                    wyckoff_volume_long_window=int(self.config.feature_wyckoff_volume_long_window),



                    wyckoff_event_horizon=int(self.config.feature_wyckoff_event_horizon),



                    # Numerical parity must rebuild the same MTF feature



                    # contract that was serialized to the EA tester set.



                    mtf_scaling_mode=str(self.config.feature_mtf_scaling_mode),



                ),



                SymbolConfig(



                    point_size=float(self.config.symbol_point_size),



                    spread_in_points=bool(self.config.symbol_spread_in_points),



                ),



                expected_schema_hash=str(self.config.feature_schema_hash),



                datetime_column=str(self.config.feature_reference_datetime_column),



                initial_equity=float(self.config.initial_equity),



                min_daily_trades=int(self.config.min_daily_trades),



                max_daily_trades=int(self.config.max_daily_trades),



                max_holding_bars=int(self.config.max_holding_bars),



                max_positions=int(self.config.max_positions),



                input_normalization_mode=str(self.config.input_normalization_mode or "raw"),



                audit_inputs_normalized=bool(self.config.feature_audit_inputs_normalized),



                atol=float(self.config.feature_parity_atol),



                rtol=float(self.config.feature_parity_rtol),



            )



        real_tick_evidence = assess_mt5_real_tick_evidence(



            report_path,



            requested_model=int(self.config.model),



            started_at_ns=tester_started_at_ns,



        )



        metrics = replace(



            metrics,



            execution_ledger_path=execution_path,



            execution_summary=execution_summary,



            reconciliation=reconciliation,



            feature_audit_path=feature_audit_path,



            feature_audit=feature_audit,



            tester_model=int(self.config.model),



            require_real_ticks_for_promotion=bool(self.config.require_real_ticks_for_promotion),



            require_real_tick_evidence_for_promotion=bool(self.config.require_real_tick_evidence_for_promotion),



            require_terminal_flat_for_promotion=bool(self.config.require_terminal_flat_for_promotion),



            real_tick_evidence=real_tick_evidence,



            feature_schema_hash=str(self.config.feature_schema_hash),



        )



        live_lifecycle_settlements = int(getattr(self, "_last_live_lifecycle_settlements", 0) or 0)



        live_lifecycle_updates = int(getattr(self, "_last_live_lifecycle_updates", 0) or 0)



        live_entry_preference_records = int(getattr(self, "_last_live_entry_preference_records", 0) or 0)



        live_entry_preference_updates = int(getattr(self, "_last_live_entry_preference_updates", 0) or 0)



        live_entry_preference_accepted = int(getattr(self, "_last_live_entry_preference_accepted", 0) or 0)



        live_entry_preference_rejected = int(getattr(self, "_last_live_entry_preference_rejected", 0) or 0)



        live_entry_preference_deferred = int(getattr(self, "_last_live_entry_preference_deferred", 0) or 0)



        live_entry_preference_pending = int(getattr(self, "_last_live_entry_preference_pending", 0) or 0)



        wall_timeout_lifecycle_updates = int(getattr(self, "_last_wall_timeout_lifecycle_updates", 0) or 0)



        post_run_retry_lifecycle_updates = 0



        five_lifecycle_flush: dict[str, object] = {}



        if bool(getattr(self.config, "learning_enabled", True)):



            live_settled = getattr(self, "_five_live_context_settled_lifecycle_ids", set())



            if not isinstance(live_settled, set):



                live_settled = set()



            for lifecycle in execution_summary.lifecycles:



                if not lifecycle.trace_id:



                    continue



                # A live ledger refresh already settled this immutable



                # lifecycle before the tester finished.  The final report is



                # still reconciled from the ledger, but must not retry the



                # popped execution-trace cache and mislabel an idempotent



                # second pass as ``cache_miss``.



                identity = "|".join((



                    str(getattr(lifecycle, "position_ticket", "") or ""),



                    str(getattr(lifecycle, "trace_id", "") or ""),



                    str(getattr(lifecycle, "close_reason", "") or ""),



                    f"{float(getattr(lifecycle, 'realized_r', 0.0) or 0.0):.8f}",



                ))



                if identity in live_settled:



                    continue



                post_run_retry_lifecycle_updates += self._apply_execution_lifecycle_credit(



                    policy=policy,



                    lifecycle=lifecycle,



                    round_index=round,



                    policy_id=policy_id,



                    source="final_evaluation",



                )



            flush_hook = getattr(policy, "flush_five_lifecycle_reward_queue", None)



            if callable(flush_hook):



                try:



                    raw_flush = flush_hook()



                    five_lifecycle_flush = (



                        dict(raw_flush) if isinstance(raw_flush, Mapping)



                        else {"flushed_batches": 0, "flushed_steps": 0, "pending": 0}



                    )



                    self._five_lifecycle_optimizer_updates += int(



                        five_lifecycle_flush.get("flushed_batches", 0) or 0



                    )



                    self._five_lifecycle_last_rollout_steps = float(



                        five_lifecycle_flush.get("pending", 0) or 0



                    )



                except Exception as exc:



                    five_lifecycle_flush = {



                        "flushed_batches": 0,



                        "flushed_steps": 0,



                        "pending": -1,



                        "error": f"{type(exc).__name__}: {exc}",



                    }



        lifecycle_updates = (



            live_lifecycle_updates



            + wall_timeout_lifecycle_updates



            + post_run_retry_lifecycle_updates



        )



        # Contextual expectancy is intentionally disabled as an action writer



        # in Full-RL.  Keep its detailed report when it contains samples, but



        # fall back to a detached ledger-owner audit when it is empty so



        # terminal-origin SL/TP closes do not disappear from fitness evidence.



        ledger_owner_audit = execution_ledger_owner_expectancy_audit(execution_summary)



        expectancy_audit: dict[str, object] = {}



        checkpoint_expectancy_audit: dict[str, object] = {}



        report_expectancy = getattr(policy, "contextual_expectancy_report", None)



        if callable(report_expectancy):



            try:



                candidate_audit = report_expectancy()



                if isinstance(candidate_audit, dict):



                    checkpoint_expectancy_audit = candidate_audit



            except Exception as exc:



                checkpoint_expectancy_audit = {"audit_error": str(exc)}



        # Frozen OOS must be scored only from lifecycles produced inside this



        # evaluation window.  A resumed policy's contextual ledger is



        # cumulative and can contain training/replay samples.



        expectancy_audit = select_mt5_evaluation_expectancy_audit(



            checkpoint_expectancy_audit,



            ledger_owner_audit,



            learning_enabled=bool(getattr(self.config, "learning_enabled", True)),



        )



        metrics = replace(metrics, expectancy_audit=expectancy_audit)



        exp = compute_mt5_expectancy(metrics)



        edge = compute_mt5_edge(metrics)



        print(



            f"[MT5] {run_label} done "



            f"profit={metrics.net_profit:.2f} pf={metrics.profit_factor:.2f} "



            f"wr={metrics.win_rate:.1f}% exp={exp:.4f} edge={edge:.4f} "



            f"trades={metrics.total_trades} dd={metrics.equity_drawdown_pct:.2f}% "



            f"actions={served_actions} lifecycle_updates={lifecycle_updates} "



            f"entry_preference_rows={live_entry_preference_records} "



            f"entry_preference_updates={live_entry_preference_updates} "



            f"entry_preference_accepted={live_entry_preference_accepted} "



            f"entry_preference_rejected={live_entry_preference_rejected} "



            f"entry_preference_pending={live_entry_preference_pending} "



            f"live_settlements={live_lifecycle_settlements} "



            f"reconciled={int(reconciliation.reconciled)}",



            flush=True,



        )



        print(



            "[MT5] report-lifecycle-join "



            f"accepted={int(report_lifecycle_join_accepted)} "



            f"report={len(report_lifecycles)} effective={len(effective_lifecycles)} "



            f"fallback={int(reconciliation.report_fallback_lifecycle_count)} "



            f"owner_resolved={int(reconciliation.report_fallback_owner_resolved_count)}",



            flush=True,



        )



        if not reconciliation.reconciled:



            print(



                "[MT5] report/ledger mismatch: " + ";".join(reconciliation.reasons),



                flush=True,



            )



        if expectancy_audit:



            policy_exp = dict(expectancy_audit.get("policy_expectancy", {}) or {})



            print(



                "[MT5] expectancy-v13 "



                f"policy_samples={int(expectancy_audit.get('policy_origin_lifecycle_count', 0) or 0)} "



                f"policy_ratio={float(expectancy_audit.get('policy_origin_ratio', 0.0) or 0.0):.3f} "



                f"lcb={float(policy_exp.get('expectancy_lcb', 0.0) or 0.0):.4f} "



                f"negative_scopes={int(expectancy_audit.get('negative_critical_scope_count', 0) or 0)}",



                flush=True,



            )



        backtest_days = _date_span_days(self.config.from_date, self.config.to_date)



        complexity_fn = getattr(policy, "structural_complexity", None)



        complexity = float(complexity_fn()) if callable(complexity_fn) else float(len(getattr(policy, "modules", [])))



        breakdown = mt5_fitness_breakdown(



            metrics,



            backtest_days=backtest_days,



            min_daily_trades=self.config.min_daily_trades,



            max_daily_trades=self.config.max_daily_trades,



            complexity=complexity,



            execution_summary=execution_summary,



            require_execution_ledger=True,



            require_terminal_flat=bool(self.config.require_terminal_flat_for_promotion),



        )



        metrics = replace(



            metrics,



            fitness_eligible=bool(breakdown.eligible),



            fitness_rank_score=float(breakdown.rank_score),



            fitness_sample_count=int(breakdown.sample_count),



            fitness_source=str(breakdown.source),



            fitness_eligibility_reasons=tuple(str(item) for item in breakdown.eligibility_reasons),



        )



        # The tester may finish before the configured post-exit horizon for a



        # just-closed position.  At the terminal reconciliation boundary,



        # release any still-deferred Codex case from immutable realized/account



        # evidence rather than silently dropping it; the phase records how many



        # were necessarily forced without the counterfactual label.



        if bool(reconciliation.reconciled):



            self._flush_deferred_codex_teacher_cases(



                execution_summary.lifecycles,



                force_partial=True,



            )



        self._run_codex_teacher_phase(



            policy, reconciled=bool(reconciliation.reconciled), report_path=report_path,



        )



        self._snapshot_rlhf_policy_update_manifest()



        print(



            f"[MT5] fitness-v{FITNESS_SCHEMA_VERSION} score={breakdown.rank_score:.2f} "



            f"eligible={int(breakdown.eligible)} source={breakdown.source} "



            f"samples={breakdown.sample_count} reasons={','.join(breakdown.eligibility_reasons) or 'ok'}",



            flush=True,



        )



        manifest_path = self._write_session_manifest(



            report_name=report_name,



            policy_id=int(policy_id),



            round_index=int(round),



            ini_path=ini_path,



            set_path=set_path,



            report_path=report_path,



            policy=policy,



            stage="finished",



            extra={



                "started_manifest_path": str(manifest_path),



                "run_label": run_label,



                "save_and_stop": dict(self._last_stop_request),



                "mt5_global_lock_path": str(self.mt5_global_lock_path),



                "mt5_global_lock_wait_seconds": float(self._last_mt5_global_lock_wait_seconds),



                "served_actions": int(served_actions),



                "live_lifecycle_settlements": int(live_lifecycle_settlements),



                "live_lifecycle_updates": int(live_lifecycle_updates),



                "live_entry_preference_records": int(live_entry_preference_records),



                "live_entry_preference_updates": int(live_entry_preference_updates),



                "live_entry_preference_accepted": int(live_entry_preference_accepted),



                "live_entry_preference_rejected": int(live_entry_preference_rejected),



                "live_entry_preference_deferred": int(live_entry_preference_deferred),



                "live_entry_preference_pending": int(live_entry_preference_pending),



                "wall_timeout_lifecycle_updates": int(wall_timeout_lifecycle_updates),



                "post_run_retry_lifecycle_updates": int(post_run_retry_lifecycle_updates),



                "five_lifecycle_flush": five_lifecycle_flush,



                "lifecycle_updates": int(lifecycle_updates),



                "five_lifecycle_routing": self._five_lifecycle_routing_snapshot(),



                "five_direct_rnn_rl": self._five_direct_rnn_routing_snapshot(policy),



                "five_action_distribution": dict(self._five_action_distribution),



                "five_champion": dict(self._five_champion_info),



                "rlhf_policy_phase": dict(self._last_rlhf_policy_update_manifest),



                "codex_teacher_phase": dict(self._last_codex_teacher_phase),



                "report_lifecycle_join": {



                    "accepted": bool(report_lifecycle_join_accepted),



                    "report_count": int(len(report_lifecycles)),



                    "report_net_profit": float(report_lifecycle_net),



                    "effective_count": int(len(effective_lifecycles)),



                    "fallback_count": int(reconciliation.report_fallback_lifecycle_count),



                    "fallback_owner_resolved": int(reconciliation.report_fallback_owner_resolved_count),



                },



                "fitness_rank_score": float(breakdown.rank_score),



                "fitness_eligible": bool(breakdown.eligible),



                "fitness_source": str(breakdown.source),



                "fitness_sample_count": int(breakdown.sample_count),



                "fitness_eligibility_reasons": list(breakdown.eligibility_reasons),



                "metrics": {



                    "net_profit": float(metrics.net_profit),



                    "profit_factor": float(metrics.profit_factor),



                    "total_trades": int(metrics.total_trades),



                    "win_rate": float(metrics.win_rate),



                    "equity_drawdown_pct": float(metrics.equity_drawdown_pct),



                    "expectancy": float(exp),



                    "edge": float(edge),



                    "tester_model": int(metrics.tester_model),



                    "bridge_rows": int(metrics.bridge_rows),



                },



                "execution_ledger_path": str(execution_path) if execution_path else None,



                "entry_preference_path": str(self._latest_runtime_file(self.config.entry_preference_file) or ""),



                "execution_summary": {



                    "ledger_available": bool(execution_summary.ledger_available),



                    "schema_version": int(execution_summary.schema_version),



                    "total_events": int(execution_summary.total_events),



                    "policy_events": int(execution_summary.policy_events),



                    "policy_intents": int(execution_summary.policy_intents),



                    "rejected_intents": int(execution_summary.rejected_intents),



                    "lifecycle_count": int(execution_summary.lifecycle_count),



                    "complete_lifecycle_ratio": float(execution_summary.complete_lifecycle_ratio),



                    "execution_failure_rate": float(execution_summary.execution_failure_rate),



                    "trace_coverage": float(execution_summary.trace_coverage),



                    "ledger_path": str(execution_summary.ledger_path) if execution_summary.ledger_path else None,



                },



                "reconciliation": {



                    "reconciled": bool(reconciliation.reconciled),



                    "reasons": list(reconciliation.reasons),



                    "report_lifecycle_count": int(reconciliation.report_lifecycle_count),



                    "report_lifecycle_net_profit": float(reconciliation.report_lifecycle_net_profit),



                    "report_fallback_lifecycle_count": int(reconciliation.report_fallback_lifecycle_count),



                    "report_fallback_owner_resolved_count": int(



                        reconciliation.report_fallback_owner_resolved_count



                    ),



                },



                "feature_audit_path": str(feature_audit_path) if feature_audit_path else None,



                "feature_audit": feature_audit,



                "real_tick_evidence": real_tick_evidence,



            },



        )



        finish_live_trial = getattr(policy, "live_observatory_trial_finished", None)



        if callable(finish_live_trial) and not False:



            finish_live_trial(



                policy_id=int(policy_id),



                round_index=int(round),



                fitness=float(breakdown.rank_score),



                metrics={



                    "net_profit": float(metrics.net_profit),



                    "profit_factor": float(metrics.profit_factor),



                    "total_trades": int(metrics.total_trades),



                    "win_rate": float(metrics.win_rate),



                    "equity_drawdown_pct": float(metrics.equity_drawdown_pct),



                    "expectancy": float(exp),



                    "edge": float(edge),



                    "served_actions": int(served_actions),



                    "live_lifecycle_settlements": int(live_lifecycle_settlements),



                    "live_lifecycle_updates": int(live_lifecycle_updates),



                    "wall_timeout_lifecycle_updates": int(wall_timeout_lifecycle_updates),



                    "post_run_retry_lifecycle_updates": int(post_run_retry_lifecycle_updates),



                    "lifecycle_updates": int(lifecycle_updates),



                    "reconciled": bool(reconciliation.reconciled),



                    "fitness_eligible": bool(breakdown.eligible),



                },



            )



        return breakdown.rank_score, metrics







    def close_open_terminals(self) -> None:



        for process in list(self._launched_processes):



            if process.poll() is None:



                self._close_terminal_after_trial(process)



        self._launched_processes = [process for process in self._launched_processes if process.poll() is None]







    def close_runner_processes(self) -> None:



        if os.name != "nt":



            return



        process_paths = [



            str(self.terminal.resolve()).lower(),



            str((self.runner_dir / "metatester64.exe").resolve()).lower(),



        ]



        quoted_paths = ",".join("'" + path.replace("'", "''") + "'" for path in process_paths)



        runner = str(self.runner_dir.resolve()).lower().replace("'", "''")



        script = (



            f"$paths=@({quoted_paths});$runner='{runner}';"



            "$deadline=(Get-Date).AddSeconds(12);"



            "$findRunnerProcesses={@(Get-CimInstance Win32_Process | Where-Object { "



            "($_.ExecutablePath -and ($paths -contains $_.ExecutablePath.ToLower())) -or "



            "($_.Name -ieq 'terminal64.exe' -and $_.CommandLine -and "



            "$_.CommandLine.ToLower().Contains($runner)) -or "



            "($_.Name -ieq 'metatester64.exe' -and "



            "(($_.ExecutablePath -and $_.ExecutablePath.ToLower().Contains($runner)) -or "



            "($_.CommandLine -and $_.CommandLine.ToLower().Contains($runner)))) })};"



            "$emptyChecks=0;"



            "while((Get-Date) -lt $deadline){"



            "$found=@(&$findRunnerProcesses);"



            "if($found.Count -eq 0){$emptyChecks++;if($emptyChecks -ge 20){exit 0}}"



            "else{$emptyChecks=0;$found | ForEach-Object { "



            "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }};"



            "Start-Sleep -Milliseconds 150};"



            "exit 1"



        )



        try:



            result = subprocess.run(



                ["powershell", "-NoProfile", "-Command", script],



                cwd=str(self.runner_dir),



                stdout=subprocess.DEVNULL,



                stderr=subprocess.DEVNULL,



                timeout=16,



                check=False,



            )



        except Exception as exc:



            raise RuntimeError(



                f"Khong the xac nhan MT5 runner da dung: {self.runner_dir}"



            ) from exc



        if int(getattr(result, "returncode", 0) or 0) != 0:



            raise RuntimeError(



                f"MT5 runner van con terminal/tester sau 12 giay: {self.runner_dir}"



            )







    def _wait_for_report(self, report_path: Path, process: subprocess.Popen) -> None:



        deadline = time.monotonic() + self.config.timeout_sec



        last_error: Exception | None = None



        report_name = report_path.stem



        while time.monotonic() < deadline:



            located_report = self._materialize_report(report_name, report_path)



            if located_report is not None:



                try:



                    parse_mt5_report(report_path)



                    return



                except Exception as exc:



                    last_error = exc



            if process.poll() is not None and located_report is None:



                raise RuntimeError(f"MT5 exited before writing report: {report_path}")



            time.sleep(1.0)



        detail = f" Last parse error: {last_error}" if last_error else ""



        raise TimeoutError(f"Timed out waiting for MT5 report while terminal remains open: {report_path}.{detail}")



