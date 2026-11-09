"""Hệ chuyên gia taxonomy context cố định, entry-style và position management.

Module này không tạo một ``policy chung + expert delta``.  Mỗi context được
sở hữu bởi đúng một chuyên gia; chuyên gia đó giữ policy, graph, bộ nhớ,
fitness và evidence riêng.  Bộ nhận diện context không sinh
BUY/SELL/SL/TP/risk. Context owner giữ bộ não kế thừa làm entry generalist.
Mỗi market context có một owner canonical. Profile có thể bật các role
MARKET/LIMIT/STOP và position-management tường minh, nhưng không có context
branch tự phát hay prototype owner.
Một correlated-exposure supervisor tách utility PRIMARY/ADDON_PROFIT/
ADDON_DRAWDOWN để chấm evidence/utility của add-on cùng hướng; nó không đổi
action logit, risk head, volume và không chặn entry.
Selector chọn đúng một trace, không blend output. Nếu tồn tại entry proposal
hợp lệ thì HOLD/management không được phủ quyết entry; selector chỉ chọn entry
nào sẽ được gửi sang MT5.

Tất cả các policy owner cố định là instance độc lập. Chúng chỉ chia sẻ:
feature contract, bridge MT5, execution ledger gốc, risk governor và định
nghĩa context.  Chúng không chia sẻ weight, edge, memory, recovery evidence
hoặc fitness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, is_dataclass
from enum import Enum
import copy
import hashlib
import json
import math
import pickle
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .contextual_expectancy import (
    DERIVED_EVIDENCE_ORIGINS,
    EXPECTANCY_SCHEMA_VERSION,
    POLICY_ORIGINS,
    robust_expectancy,
)

import numpy as np

from .five_agent_authority import (
    FIVE_AGENT_AUTHORITY_VERSION,
    five_agent_authority,
)

from .protocol_contract import INPUT_COUNT, TIMING_OFFSET
from .context_history import (
    CONTEXT_HISTORY_INPUT_COUNT,
    append_context_history_inputs,
    encode_context_history,
)
from .entry_feature_enrichment import (
    ENTRY_FEATURE_ENRICHMENT_LEGACY,
    append_entry_quality_interactions,
    entry_feature_interaction_width,
    normalize_entry_feature_enrichment_mode,
)
from .market_regime_classifier import VOLATILE_SUBREGIMES
from .timing_network import TimingNetwork
from .policy_ensemble import (
    CorrelatedExposureState,
    CorrelatedExposureSupervisor,
    PolicyCandidateScore,
    PolicyEnsembleSelector,
    PolicySelectionTrace,
    RoleReadinessSupervisor,
)

# Contract bridge cố định: 20 output, chỉ số HOLD = 0. Không import orders ở đây
# để tránh vòng import Config -> HệChuyênGia -> Orders -> Config.
SO_DAU_RA_BRIDGE = 20
CHI_SO_HOLD = 0
CHI_SO_RISK_OUTPUT = 12
CHI_SO_ENTRY_ACTIONS = frozenset({1, 2, 3, 4, 5, 6})
# An entry owner must also be able to cancel its own stale pending orders.
# Without this capability, the bridge's pending-order mask intersects with the
# six-entry proposal and collapses to HOLD, starving the owner after its first
# batch of unfilled orders.
CHI_SO_ENTRY_AND_PENDING_ACTIONS = frozenset({1, 2, 3, 4, 5, 6, 7})
CHI_SO_MANAGEMENT_ACTIONS = frozenset({8, 9, 10, 11})
ENTRY_STYLE_SPECIALISTS: dict[str, frozenset[int]] = {
    "MARKET": frozenset({1, 2}),
    "LIMIT": frozenset({3, 4}),
    "STOP": frozenset({5, 6}),
}
ROLE_EXPERT_CONTEXTS = {
    "MARKET": "ROLE_ENTRY_MARKET",
    "LIMIT": "ROLE_ENTRY_LIMIT",
    "STOP": "ROLE_ENTRY_STOP",
    "MANAGEMENT": "ROLE_POSITION_MANAGEMENT",
}
LEGACY_GLOBAL_ROLE_CONTEXTS = frozenset(ROLE_EXPERT_CONTEXTS.values())
ROLE_CONTEXT_SEPARATOR = "__"


class TrangThaiChuyenGia(str, Enum):
    MAM = "MAM"
    QUAN_SAT_SONG_SONG = "QUAN_SAT_SONG_SONG"
    THAM_DO_RUI_RO_THAP = "THAM_DO_RUI_RO_THAP"
    HOAT_DONG_HAN_CHE = "HOAT_DONG_HAN_CHE"
    TRUONG_THANH = "TRUONG_THANH"
    NGU_DONG = "NGU_DONG"
    LUU_TRU = "LUU_TRU"


class NhiemVuNguCanh(str, Enum):
    # Legacy contexts (kept for backward compatibility)
    VAO_LENH = "VAO_LENH"
    QUAN_LY_MUA = "QUAN_LY_MUA"
    QUAN_LY_BAN = "QUAN_LY_BAN"
    XU_LY_LENH_CHO = "XU_LY_LENH_CHO"
    PHONG_THU_TAI_KHOAN = "PHONG_THU_TAI_KHOAN"
    # Market regime contexts
    TANG_MANH = "TANG_MANH"
    TANG_YEU = "TANG_YEU"
    GIAM_MANH = "GIAM_MANH"
    GIAM_YEU = "GIAM_YEU"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    VOLATILE_SPIKE = "VOLATILE_SPIKE"
    VOLATILE_EXPANSION = "VOLATILE_EXPANSION"
    VOLATILE_CHOP = "VOLATILE_CHOP"


class PolicyCoTheRaQuyetDinh(Protocol):
    def act(self, observation: np.ndarray, goal: np.ndarray | None = None): ...


@dataclass
class QuyetDinhKhongGiaoDich:
    """Trace vận hành khi context chưa ổn định hoặc chưa có owner.

    Đây không phải output của policy chung. Vector HOLD chỉ là mã no-op bắt
    buộc của wire MT5; ``ly_do_khong_hoc`` buộc bridge không xem nó là dữ liệu
    huấn luyện hay bằng chứng action.
    """

    trace_id: str
    selected_action: np.ndarray
    selected_idx: int
    confidence: float = 0.0
    ma_context: str = ""
    ma_chuyen_gia: str = ""
    ma_policy: str = ""
    phien_ban_policy: str = ""
    ly_do_khong_giao_dich: str = ""
    ly_do_khong_hoc: bool = True
    context_audit: dict[str, object] = field(default_factory=dict)
    action_details: dict[str, object] = field(default_factory=dict)
    execution_details: dict[str, object] = field(default_factory=dict)
    positions_pipe: str = ""
    positions_pipe_after: str = ""
    account_context: dict[str, float] = field(default_factory=dict)
    account_context_after: dict[str, float] = field(default_factory=dict)
    experience_starvation: dict[str, float] = field(default_factory=dict)


def _trace_metadata(trace: object) -> dict[str, object]:
    audit = dict(getattr(trace, "context_audit", {}) or {})
    for key in ("ma_context", "ma_context_cha", "ma_chuyen_gia", "ma_policy", "phien_ban_policy", "trang_thai_chuyen_gia"):
        value = getattr(trace, key, None)
        if value not in (None, ""):
            audit[key] = value
    return audit


def _gan_action_details(trace: object, metadata: Mapping[str, object]):
    """Merge observability-only action detail without mutating frozen StepTrace."""
    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({str(k): v for k, v in dict(metadata).items()})
    if is_dataclass(trace):
        try:
            return replace(trace, action_details=details)
        except (TypeError, ValueError):
            pass
    try:
        setattr(trace, "action_details", details)
    except Exception:
        pass
    return trace


def _gan_trace_metadata(trace: object, metadata: Mapping[str, object]):
    """Gắn provenance mà không phá StepTrace frozen của policy gốc."""
    merged = _trace_metadata(trace)
    merged.update({str(k): v for k, v in dict(metadata).items()})
    if is_dataclass(trace):
        try:
            trace = replace(trace, context_audit=merged)
        except (TypeError, ValueError):
            pass
    for key, value in merged.items():
        try:
            setattr(trace, key, value)
        except Exception:
            continue
    try:
        setattr(trace, "context_audit", merged)
    except Exception:
        pass
    return trace



def _policy_identity(policy: object) -> dict[str, object]:
    """Stable provenance proving that context owners do not share mutable brains.

    The identity fields are attached lazily so they survive ordinary runtime use
    and pickling, while the compact hashes change when topology/weights/memory
    change.  They are audit fields only; they never participate in routing.
    """
    instance = str(getattr(policy, "_context_expert_policy_instance_uuid", "") or "")
    if not instance:
        instance = uuid.uuid4().hex
        try:
            setattr(policy, "_context_expert_policy_instance_uuid", instance)
        except Exception:
            pass

    graph_instance = str(getattr(policy, "_context_expert_graph_instance_uuid", "") or "")
    if not graph_instance:
        graph_instance = uuid.uuid4().hex
        try:
            setattr(policy, "_context_expert_graph_instance_uuid", graph_instance)
        except Exception:
            pass

    graph_rows: list[tuple[object, ...]] = []
    modules = list(getattr(policy, "modules", []) or [])
    for module in modules:
        m1 = getattr(module, "m1_component", None)
        graph = getattr(m1, "graph", None)
        if graph is None:
            continue
        neurons = getattr(graph, "neurons", {})
        edges = getattr(graph, "edges", {})
        try:
            edge_weight_sum = round(sum(float(getattr(edge, "weight", 0.0) or 0.0) for edge in edges.values()), 10)
            edge_gate_sum = round(sum(float(getattr(edge, "gate", 0.0) or 0.0) for edge in edges.values()), 10)
        except Exception:
            edge_weight_sum, edge_gate_sum = 0.0, 0.0
        graph_rows.append((
            str(getattr(module, "name", "")),
            len(neurons) if hasattr(neurons, "__len__") else 0,
            len(edges) if hasattr(edges, "__len__") else 0,
            edge_weight_sum,
            edge_gate_sum,
        ))
    if not graph_rows:
        graph_rows.append((
            "fallback",
            len(modules),
            int(getattr(policy, "revision", 0) or 0),
            0.0,
            0.0,
        ))

    memory = getattr(policy, "memory_supernet", None)
    memory_instance = str(getattr(policy, "_context_expert_memory_instance_uuid", "") or "")
    if not memory_instance:
        memory_instance = uuid.uuid4().hex
        try:
            setattr(policy, "_context_expert_memory_instance_uuid", memory_instance)
        except Exception:
            pass
    contextual_expectancy = getattr(memory, "contextual_expectancy", None)
    memory_rows = {
        "neurons": len(getattr(memory, "neurons", {}) or {}),
        "connections": len(getattr(memory, "connections", []) or []),
        "clusters": len(getattr(memory, "clusters", {}) or {}),
        "concepts": len(getattr(memory, "concepts", {}) or {}),
        "strategies": len(getattr(memory, "strategies", {}) or {}),
        "meta_strategies": len(getattr(memory, "meta_strategies", {}) or {}),
        "causal_nodes": len(getattr(memory, "causal_nodes", {}) or {}),
        "neural_cells": len(getattr(memory, "neural_cells", {}) or {}),
        "neural_synapses": len(getattr(memory, "neural_synapses", []) or []),
        "contextual_expectancy_records": len(getattr(contextual_expectancy, "records", {}) or {}),
    }
    memory_entry_count = int(sum(memory_rows.values()))
    weight_hash = hashlib.sha256(repr((int(getattr(policy, "revision", 0) or 0), tuple(graph_rows))).encode("utf-8")).hexdigest()[:20]
    memory_hash = hashlib.sha256(repr((memory_instance, memory_rows)).encode("utf-8")).hexdigest()[:20]
    checkpoint_hash = hashlib.sha256(repr((instance, graph_instance, weight_hash, memory_hash, int(getattr(policy, "revision", 0) or 0))).encode("utf-8")).hexdigest()[:20]
    return {
        "policy_instance_uuid": instance,
        "graph_instance_uuid": graph_instance,
        "memory_instance_uuid": memory_instance,
        "memory_object_identity": hex(id(memory)) if memory is not None else "",
        "memory_entry_count": memory_entry_count,
        "memory_entry_breakdown": dict(memory_rows),
        "weight_hash": weight_hash,
        "memory_hash": memory_hash,
        "policy_checkpoint_hash": checkpoint_hash,
        "policy_revision": int(getattr(policy, "revision", 0) or 0),
    }


def _allowed_actions_for_context_task(task: NhiemVuNguCanh) -> set[int]:
    """Semantic action contract per context task.

    These are permissions, not trading rules: a context policy is still free to
    choose any *allowed* action from its own neural output. Operational tasks
    such as pending-order handling and account defense remain narrow; market
    regime tasks intentionally keep every entry direction available for learning.
    """
    HOLD = 0
    MARKET_BUY, MARKET_SELL = 1, 2
    BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP = 3, 4, 5, 6
    CANCEL_ORDERS, CLOSE_WORST, CLOSE_BEST, CLOSE_ALL, MANAGE_POSITION = 7, 8, 9, 10, 11
    if task == NhiemVuNguCanh.VAO_LENH:
        return {HOLD, MARKET_BUY, MARKET_SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP}
    if task == NhiemVuNguCanh.QUAN_LY_MUA:
        return {HOLD, CLOSE_WORST, CLOSE_BEST, CLOSE_ALL, MANAGE_POSITION}
    if task == NhiemVuNguCanh.QUAN_LY_BAN:
        return {HOLD, CLOSE_WORST, CLOSE_BEST, CLOSE_ALL, MANAGE_POSITION}
    if task == NhiemVuNguCanh.XU_LY_LENH_CHO:
        return {HOLD, CANCEL_ORDERS}
    if task == NhiemVuNguCanh.PHONG_THU_TAI_KHOAN:
        return {HOLD, CLOSE_WORST, CLOSE_BEST, CLOSE_ALL, MANAGE_POSITION}
    # Market-regime experts must see every entry direction.  The classifier is
    # an input to learning, not an irreversible BUY/SELL gate: if its regime
    # label is wrong, the expert still needs counter-direction examples to
    # correct both the router and its own entry timing memory.
    if task in {
        NhiemVuNguCanh.TANG_MANH,
        NhiemVuNguCanh.TANG_YEU,
        NhiemVuNguCanh.GIAM_MANH,
        NhiemVuNguCanh.GIAM_YEU,
        NhiemVuNguCanh.RANGING,
        NhiemVuNguCanh.VOLATILE,
        NhiemVuNguCanh.VOLATILE_SPIKE,
        NhiemVuNguCanh.VOLATILE_EXPANSION,
        NhiemVuNguCanh.VOLATILE_CHOP,
    }:
        return {
            HOLD, MARKET_BUY, MARKET_SELL,
            BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP,
            CANCEL_ORDERS, CLOSE_WORST, CLOSE_BEST, CLOSE_ALL, MANAGE_POSITION,
        }
    return {HOLD}


def _normalise_allowed_actions(value: object) -> set[int]:
    if isinstance(value, Mapping):
        return {
            int(key) for key, enabled in dict(value).items()
            if bool(enabled) and str(key).lstrip("-").isdigit() and 0 <= int(key) < 12
        }
    if isinstance(value, (set, frozenset, tuple, list)):
        rows: set[int] = set()
        for item in value:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= index < 12:
                rows.add(index)
        return rows
    return set(range(12))


def _context_allowed_actions(nhan_dien: NhanDienNguCanh, upstream: object) -> set[int]:
    upstream_set = _normalise_allowed_actions(upstream)
    task_set = _allowed_actions_for_context_task(nhan_dien.nhiem_vu)
    allowed = upstream_set & task_set
    return allowed if allowed else {CHI_SO_HOLD}


def _trend_exit_guard(
    nhan_dien: NhanDienNguCanh,
    allowed: set[int],
    positions_pipe: object,
    cau_hinh: "CauHinhHeChuyenGia",
) -> tuple[set[int], dict[str, object]]:
    """Protect aligned strong-trend positions until exit evidence exists.

    This guard never removes an entry action. It only prevents a stable strong
    trend owner from turning a fresh/aligned position into CLOSE_BEST/CLOSE_ALL
    churn while the position is still making progress. A regime transition,
    emergency loss, near-TP state or confirmed giveback restores close actions.
    """
    audit: dict[str, object] = {
        "trend_exit_guard_active": False,
        "trend_exit_guard_reason": "",
        "trend_exit_protected_tickets": [],
        "trend_exit_ready_tickets": [],
    }
    if not bool(cau_hinh.bao_ve_thoat_som_trend_manh):
        return set(allowed), audit
    if not bool(nhan_dien.on_dinh) or nhan_dien.nhiem_vu not in {
        NhiemVuNguCanh.TANG_MANH,
        NhiemVuNguCanh.GIAM_MANH,
    }:
        return set(allowed), audit

    # Local import avoids Config -> HeChuyenGia -> Orders -> Config at module load.
    from .mistake_memory import parse_positions_pipe

    positions = parse_positions_pipe(str(positions_pipe or ""))
    if not positions:
        return set(allowed), audit

    expected_side = 1 if nhan_dien.nhiem_vu == NhiemVuNguCanh.TANG_MANH else -1
    protected: list[str] = []
    ready: list[str] = []
    has_risk_exit = False
    for position in positions:
        ticket = str(position.get("ticket", "") or "")
        side = int(position.get("side", 0) or 0)
        current_r = float(position.get("unrealized_r", 0.0) or 0.0)
        mfe_r = float(position.get("mfe_r", current_r) or current_r)
        giveback_r = max(
            float(position.get("giveback_r", 0.0) or 0.0),
            max(0.0, mfe_r - current_r),
        )
        bars_since_mfe = float(position.get("bars_since_mfe", 0.0) or 0.0)
        near_tp = bool(position.get("near_tp", False))
        aligned = side == expected_side
        emergency = current_r <= float(cau_hinh.trend_manh_lo_r_cho_phep_thoat)
        confirmed_giveback = (
            giveback_r >= float(cau_hinh.trend_manh_giveback_r_de_thoat)
            and bars_since_mfe >= int(cau_hinh.trend_manh_bars_sau_mfe_de_thoat)
        )
        exit_ready = (not aligned) or emergency or near_tp or confirmed_giveback
        if exit_ready:
            ready.append(ticket)
            has_risk_exit = has_risk_exit or emergency or not aligned
        else:
            protected.append(ticket)

    if not protected:
        audit["trend_exit_ready_tickets"] = ready
        return set(allowed), audit

    guarded = set(allowed)
    guarded.discard(9)   # CLOSE_BEST
    guarded.discard(10)  # CLOSE_ALL
    if not has_risk_exit:
        guarded.discard(8)  # CLOSE_WORST
    if not guarded:
        guarded = {CHI_SO_HOLD}
    audit.update({
        "trend_exit_guard_active": True,
        "trend_exit_guard_reason": "stable_strong_trend_without_exit_evidence",
        "trend_exit_protected_tickets": protected,
        "trend_exit_ready_tickets": ready,
        "trend_exit_manage_available": 11 in guarded,
        "trend_exit_giveback_threshold_r": float(cau_hinh.trend_manh_giveback_r_de_thoat),
        "trend_exit_bars_since_mfe_threshold": int(cau_hinh.trend_manh_bars_sau_mfe_de_thoat),
    })
    return guarded, audit


def _owner_management_guard(
    allowed: set[int],
    positions_pipe: object,
    owner_expert: str,
    cau_hinh: "CauHinhHeChuyenGia",
) -> tuple[set[int], dict[str, object]]:
    """Keep a context transition from becoming a blind losing exit."""
    audit: dict[str, object] = {
        "owner_management_guard_active": False,
        "owner_management_guard_reason": "",
        "owner_management_best_unrealized_r": 0.0,
        "owner_management_worst_unrealized_r": 0.0,
        "owner_management_all_profitable": False,
    }
    if not bool(cau_hinh.bao_ve_quan_ly_khi_doi_context):
        return set(allowed), audit
    positions = [
        item for item in HeChuyenGiaMotContextMotPolicy._positions_with_owner(positions_pipe)
        if str(item.get("owner_expert", "")) == str(owner_expert)
    ]
    if not positions:
        return set(allowed), audit

    values = [float(item.get("unrealized_r", 0.0) or 0.0) for item in positions]
    best_r = max(values)
    worst_r = min(values)
    min_profit_r = float(cau_hinh.quan_ly_doi_context_min_profit_r)
    emergency_r = float(cau_hinh.quan_ly_doi_context_lo_r_khan_cap)
    all_profitable = all(value >= min_profit_r for value in values)
    guarded = set(allowed)
    if best_r < min_profit_r:
        guarded.discard(9)   # CLOSE_BEST
    if not all_profitable:
        guarded.discard(10)  # CLOSE_ALL
    if worst_r > emergency_r:
        guarded.discard(8)   # CLOSE_WORST
    if not guarded:
        guarded = {CHI_SO_HOLD}
    reason = ""
    if best_r < min_profit_r:
        reason = "NO_PROFIT_FOR_CLOSE_BEST"
    elif not all_profitable:
        reason = "MIXED_PROFIT_FOR_CLOSE_ALL"
    elif worst_r > emergency_r:
        reason = "NO_EMERGENCY_FOR_CLOSE_WORST"
    audit.update({
        "owner_management_guard_active": guarded != set(allowed),
        "owner_management_guard_reason": reason,
        "owner_management_best_unrealized_r": best_r,
        "owner_management_worst_unrealized_r": worst_r,
        "owner_management_all_profitable": all_profitable,
        "owner_management_min_profit_r": min_profit_r,
        "owner_management_emergency_r": emergency_r,
    })
    return guarded, audit


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))[:96] or "unknown"


def _logit(value: float) -> float:
    clipped = min(1.0 - 1e-9, max(1e-9, float(value)))
    return math.log(clipped / (1.0 - clipped))


def _risk_head_for_pct(risk_pct: float, *, min_risk_pct: float, max_risk_pct: float) -> float:
    low = float(min_risk_pct)
    high = float(max_risk_pct)
    if high <= low:
        return 0.0
    ratio = (float(risk_pct) - low) / (high - low)
    return _logit(ratio)


def _risk_pct_for_head(raw_head: float, *, min_risk_pct: float, max_risk_pct: float) -> float:
    low = float(min_risk_pct)
    high = float(max_risk_pct)
    if high <= low:
        return low
    raw = float(np.clip(float(raw_head), -60.0, 60.0))
    ratio = 1.0 / (1.0 + math.exp(-raw))
    return float(low + (high - low) * ratio)


def _with_selected_action(trace: object, selected_action: np.ndarray):
    if is_dataclass(trace):
        try:
            return replace(trace, selected_action=np.asarray(selected_action, dtype=float))
        except (TypeError, ValueError):
            pass
    try:
        setattr(trace, "selected_action", np.asarray(selected_action, dtype=float))
    except Exception:
        pass
    return trace


def _with_owner_identity(trace: object, metadata: Mapping[str, object]):
    """Override MT5 position-owner fields while preserving decision-policy audit."""
    owner_fields = {
        key: value for key, value in dict(metadata).items()
        if key in {"ma_context", "ma_context_cha", "ma_chuyen_gia", "ma_policy", "phien_ban_policy", "trang_thai_chuyen_gia"}
    }
    if is_dataclass(trace):
        dataclass_fields = getattr(trace, "__dataclass_fields__", {})
        updates = {key: value for key, value in owner_fields.items() if key in dataclass_fields}
        if updates:
            try:
                trace = replace(trace, **updates)
            except (TypeError, ValueError):
                pass
    else:
        for key, value in owner_fields.items():
            try:
                setattr(trace, key, value)
            except Exception:
                pass
    return _gan_trace_metadata(trace, metadata)


@dataclass(frozen=True)
class CauHinhHeChuyenGia:
    """Cấu hình runtime của kiến trúc một-context-một-policy.

    Các giá trị là guardrail vận hành; chúng không chứa luật BUY/SELL. Regime
    được phát trực tiếp từ nến M15 đã đóng, không có debounce/countdown.
    """

    bat: bool = False
    che_do_quyet_dinh: str = "mot_context_mot_chuyen_gia"
    so_tac_nhan: int = 5
    so_chieu_nhan_dang_tac_nhan: int = 5
    cho_phep_policy_chung_ra_lenh: bool = False
    cho_phep_tron_policy: bool = False
    so_chuyen_gia_moi_context: int = 1
    regime_timeframe: str = "M15"
    # Candidate-only Python-local inputs: M15 context t-1 and t-2. The MT5
    # EA/CSV bridge remains the fixed 137-feature wire contract.
    context_history_inputs: int = 0
    # Candidate-only entry representation.  The interaction tail is generated
    # in Python from the same causal wire and remains invisible to management.
    # These are encoder gains, not fixed BUY/SELL score weights or masks.
    entry_feature_enrichment_mode: str = ENTRY_FEATURE_ENRICHMENT_LEGACY
    entry_feature_quarantine_scale: float = 1.0
    entry_feature_h1_direct_scale: float = 1.0
    bien_chuyen_ngu_canh: float = 0.10
    nguong_vi_the: float = 0.05
    nguong_drawdown_phong_thu: float = 0.65
    so_lifecycle_toi_thieu_de_hoat_dong: int = 40
    so_lifecycle_toi_thieu_de_truong_thanh: int = 120
    max_risk_pct_tham_do: float = 0.005
    max_open_risk_pct_tham_do: float = 0.015
    # Correlated add-ons remain legal entries. Exposure is used only for
    # selector evidence/score; the policy risk head and volume are preserved.
    correlated_exposure_risk_decay: float = 0.50
    correlated_exposure_min_risk_pct: float = 0.001
    # A new role-direction family remains eligible to enter. Readiness is
    # telemetry/evidence only; it never attenuates the policy risk head.
    role_readiness_min_risk_multiplier: float = 0.25
    role_readiness_support_scale: float = 4.0
    role_readiness_negative_utility_weight: float = 0.50
    # Selector anti-overfit controls.  Cold roles remain legal probes when no
    # supported candidate exists, but cannot outrank an evidenced route solely
    # because their neural confidence is high.  Exposure penalties are ranking
    # terms; they never veto an otherwise legal entry.
    selector_cold_start_penalty: float = 0.0
    selector_cold_start_support: int = 8
    selector_cold_start_max_per_run: int = 0
    selector_pending_entry_max_age_decisions: int = 0
    selector_inflight_penalty: float = 0.20
    selector_exposure_penalty: float = 0.20
    # Loss-streak pressure is a ranking term only.  Keep the protected v13
    # default unchanged; continuation profiles may raise it to reduce repeat
    # routing into a role with freshly confirmed losses.
    selector_loss_streak_penalty: float = 0.12
    # Continuation-only policy/role backoff across context microstates.  The
    # protected v13 default remains disabled so loading the seed is bitwise
    # behavior-compatible; candidate profiles may enable it explicitly.
    selector_policy_group_loss_streak_penalty: float = 0.0
    context_chua_so_huu: str = "DUNG_QUAN_SAT"
    # Hierarchical committee: context generalist + three entry-style policies
    # + one position manager. Capability scopes partition responsibility but
    # their union keeps every entry action available to the selector.
    entry_style_specialists_bat: bool = True
    position_management_specialist_bat: bool = True
    # Strong-trend exits require position evidence. Entry actions are never
    # removed by this guard; MANAGE_POSITION remains available for trailing.
    bao_ve_thoat_som_trend_manh: bool = True
    trend_manh_giveback_r_de_thoat: float = 0.35
    trend_manh_bars_sau_mfe_de_thoat: int = 2
    # Allow a loss exit before broker SL; this only opens exit actions and
    # never removes entry actions or rewrites policy risk/volume.
    trend_manh_lo_r_cho_phep_thoat: float = -0.35
    bao_ve_quan_ly_khi_doi_context: bool = True
    quan_ly_doi_context_min_profit_r: float = 0.10
    quan_ly_doi_context_lo_r_khan_cap: float = -0.35
    # Soft context-direction evidence for direct_rnn action ranking. This is
    # a score prior only: it never removes BUY/SELL entry actions.
    context_direction_prior_strength: float = 0.10
    candidate_transition_direction_prior_bat: bool = True
    volatile_timing_direction_prior_bat: bool = True
    stable_context_direction_prior_during_transition_bat: bool = True
    # Candidate-only response to a learned TimingNet deceleration forecast.
    # It is active solely while context confirmation is in progress: a late
    # LIMIT entry becomes a same-side STOP confirmation and the risk head is
    # capped, never blocked or direction-flipped.
    timing_deceleration_response_bat: bool = False
    timing_deceleration_entry_threshold: float = 0.68
    timing_deceleration_min_updates: int = 24
    timing_deceleration_limit_to_stop_bat: bool = True
    timing_deceleration_min_risk_multiplier: float = 0.35
    # A mature shared TimingNet may suppress a new entry before a likely
    # *confirmed* regime change.  It is intentionally disabled for legacy
    # artifacts.  When enabled, all calibration conditions below must pass;
    # a raw neural score is never allowed to manufacture a HOLD by itself.
    timing_regime_transition_entry_block_bat: bool = False
    timing_regime_transition_entry_threshold: float = 0.78
    timing_regime_transition_min_updates: int = 192
    timing_regime_transition_min_calibration_samples: int = 128
    timing_regime_transition_min_high_confidence_samples: int = 12
    timing_regime_transition_min_precision: float = 0.75
    timing_regime_transition_max_brier: float = 0.20
    # Volatile timing direction can be useful before TimingNet reaches high
    # confidence; it remains a bounded score prior, not a gate.
    volatile_timing_direction_min_quality: float = 0.10
    # Volatile is reserved for high volatility with near-neutral direction;
    # weak signed trends are routed to TANG_YEU/GIAM_YEU.
    volatile_max_directional_strength: float = 0.18
    # Split volatile into independent owners. This is a routing choice, not an
    # entry gate; setting it false is only a legacy compatibility fallback.
    volatile_subexperts_bat: bool = True
    volatile_spike_threshold: float = 0.55
    volatile_expansion_threshold: float = 1.15

    def __post_init__(self) -> None:
        if self.che_do_quyet_dinh not in {"mot_context_mot_chuyen_gia", "mot_bo_nao_dung_chung", "nam_tac_nhan_dung_chung"}:
            raise ValueError("che_do_quyet_dinh khong hop le")
        if self.che_do_quyet_dinh == "mot_context_mot_chuyen_gia" and self.cho_phep_policy_chung_ra_lenh:
            raise ValueError("che do cu khong duoc cho phep policy chung ra lenh")
        if self.che_do_quyet_dinh in {"mot_bo_nao_dung_chung", "nam_tac_nhan_dung_chung"} and not self.cho_phep_policy_chung_ra_lenh:
            raise ValueError("che do dung chung can cho_phep_policy_chung_ra_lenh=true")
        if self.che_do_quyet_dinh == "nam_tac_nhan_dung_chung":
            if int(self.so_tac_nhan) != 5:
                raise ValueError("so_tac_nhan phai bang 5")
            if int(self.so_chieu_nhan_dang_tac_nhan) != 5:
                raise ValueError("so_chieu_nhan_dang_tac_nhan phai bang 5")
        if self.cho_phep_tron_policy:
            raise ValueError("khong duoc tron dau ra cua nhieu policy")
        if int(self.so_chuyen_gia_moi_context) != 1:
            raise ValueError("moi context chi co dung mot chuyen gia owner")
        if str(self.regime_timeframe).upper() != "M15":
            raise ValueError("regime_timeframe hien chi ho tro M15")
        if int(self.context_history_inputs) not in {0, CONTEXT_HISTORY_INPUT_COUNT}:
            raise ValueError("context_history_inputs phai la 0 hoac 2")
        normalize_entry_feature_enrichment_mode(self.entry_feature_enrichment_mode)
        if not 0.0 <= float(self.entry_feature_quarantine_scale) <= 1.0:
            raise ValueError("entry_feature_quarantine_scale phai trong [0, 1]")
        if not 0.0 <= float(self.entry_feature_h1_direct_scale) <= 1.0:
            raise ValueError("entry_feature_h1_direct_scale phai trong [0, 1]")
        if not 0.0 <= float(self.bien_chuyen_ngu_canh) <= 1.0:
            raise ValueError("bien_chuyen_ngu_canh phai trong [0, 1]")
        if not 0.0 < float(self.nguong_vi_the) <= 2.0:
            raise ValueError("nguong_vi_the phai trong (0, 2]")
        if not 0.0 <= float(self.nguong_drawdown_phong_thu) <= 2.0:
            raise ValueError("nguong_drawdown_phong_thu phai trong [0, 2]")
        for field_name in (
            "so_lifecycle_toi_thieu_de_hoat_dong",
            "so_lifecycle_toi_thieu_de_truong_thanh",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} phai >= 1")
        if not 0.0 < float(self.max_risk_pct_tham_do) <= 1.0:
            raise ValueError("max_risk_pct_tham_do phai trong (0, 1]")
        if not float(self.max_risk_pct_tham_do) <= float(self.max_open_risk_pct_tham_do) <= 1.0:
            raise ValueError("max_open_risk_pct_tham_do phai >= max_risk_pct_tham_do va <= 1")
        if not 0.05 <= float(self.correlated_exposure_risk_decay) <= 1.0:
            raise ValueError("correlated_exposure_risk_decay phai trong [0.05, 1]")
        if not 0.0 < float(self.correlated_exposure_min_risk_pct) <= 1.0:
            raise ValueError("correlated_exposure_min_risk_pct phai trong (0, 1]")
        if not 0.05 <= float(self.role_readiness_min_risk_multiplier) <= 1.0:
            raise ValueError("role_readiness_min_risk_multiplier phai trong [0.05, 1]")
        if float(self.role_readiness_support_scale) < 1.0:
            raise ValueError("role_readiness_support_scale phai >= 1")
        if not 0.0 <= float(self.selector_cold_start_penalty) <= 2.0:
            raise ValueError("selector_cold_start_penalty phai trong [0, 2]")
        if int(self.selector_cold_start_support) < 1:
            raise ValueError("selector_cold_start_support phai >= 1")
        if int(self.selector_cold_start_max_per_run) < 0:
            raise ValueError("selector_cold_start_max_per_run phai >= 0")
        if int(self.selector_pending_entry_max_age_decisions) < 0:
            raise ValueError("selector_pending_entry_max_age_decisions phai >= 0")
        for field_name in (
            "selector_inflight_penalty",
            "selector_exposure_penalty",
            "selector_loss_streak_penalty",
            "selector_policy_group_loss_streak_penalty",
        ):
            if not 0.0 <= float(getattr(self, field_name)) <= 1.0:
                raise ValueError(f"{field_name} phai trong [0, 1]")
        if not 0.0 <= float(self.role_readiness_negative_utility_weight) <= 1.0:
            raise ValueError("role_readiness_negative_utility_weight phai trong [0, 1]")
        if float(self.trend_manh_giveback_r_de_thoat) < 0.0:
            raise ValueError("trend_manh_giveback_r_de_thoat phai >= 0")
        if int(self.trend_manh_bars_sau_mfe_de_thoat) < 0:
            raise ValueError("trend_manh_bars_sau_mfe_de_thoat phai >= 0")
        if float(self.trend_manh_lo_r_cho_phep_thoat) >= 0.0:
            raise ValueError("trend_manh_lo_r_cho_phep_thoat phai < 0")
        if float(self.quan_ly_doi_context_min_profit_r) < 0.0:
            raise ValueError("quan_ly_doi_context_min_profit_r phai >= 0")
        if float(self.quan_ly_doi_context_lo_r_khan_cap) >= 0.0:
            raise ValueError("quan_ly_doi_context_lo_r_khan_cap phai < 0")
        if not 0.0 <= float(self.context_direction_prior_strength) <= 0.25:
            raise ValueError("context_direction_prior_strength phai trong [0, 0.25]")
        if not 0.50 <= float(self.timing_deceleration_entry_threshold) < 1.0:
            raise ValueError("timing_deceleration_entry_threshold phai trong [0.50, 1)")
        if int(self.timing_deceleration_min_updates) < 1:
            raise ValueError("timing_deceleration_min_updates phai >= 1")
        if not 0.05 <= float(self.timing_deceleration_min_risk_multiplier) <= 1.0:
            raise ValueError("timing_deceleration_min_risk_multiplier phai trong [0.05, 1]")
        if not 0.50 <= float(self.timing_regime_transition_entry_threshold) < 1.0:
            raise ValueError("timing_regime_transition_entry_threshold phai trong [0.50, 1)")
        if int(self.timing_regime_transition_min_updates) < 1:
            raise ValueError("timing_regime_transition_min_updates phai >= 1")
        if int(self.timing_regime_transition_min_calibration_samples) < 1:
            raise ValueError("timing_regime_transition_min_calibration_samples phai >= 1")
        if int(self.timing_regime_transition_min_high_confidence_samples) < 1:
            raise ValueError("timing_regime_transition_min_high_confidence_samples phai >= 1")
        if not 0.50 <= float(self.timing_regime_transition_min_precision) <= 1.0:
            raise ValueError("timing_regime_transition_min_precision phai trong [0.50, 1]")
        if not 0.0 <= float(self.timing_regime_transition_max_brier) <= 1.0:
            raise ValueError("timing_regime_transition_max_brier phai trong [0, 1]")
        if not 0.0 <= float(self.volatile_timing_direction_min_quality) <= 1.0:
            raise ValueError("volatile_timing_direction_min_quality phai trong [0, 1]")
        if not 0.0 <= float(self.volatile_max_directional_strength) <= 1.0:
            raise ValueError("volatile_max_directional_strength phai trong [0, 1]")
        if not 0.0 <= float(self.volatile_spike_threshold) <= 2.0:
            raise ValueError("volatile_spike_threshold phai trong [0, 2]")
        if not 0.0 <= float(self.volatile_expansion_threshold) <= 2.0:
            raise ValueError("volatile_expansion_threshold phai trong [0, 2]")


@dataclass(frozen=True)
class NhanDienNguCanh:
    ma_context: str
    ma_context_cha: str
    nhiem_vu: NhiemVuNguCanh
    toa_do: tuple[float, ...]
    do_tin_cay: float
    on_dinh: bool
    ly_do: str
    timing_quality: float = 1.0
    timing_direction: int = 0
    candidate_context: str = ""
    transition_active: bool = False
    transition_progress: float = 1.0
    timing_network_update_count: int = 0
    timing_source: str = "TimingNet_disabled"
    timing_network_quality: float = 0.0
    timing_network_direction: float = 0.0
    timing_network_entry_probability: float = 0.0
    timing_network_uncertainty: float = 1.0
    timing_network_deceleration_probability: float = 0.0
    timing_network_deceleration_update_count: int = 0
    timing_network_trend_score: float = 0.0
    timing_network_regime_transition_probability: float = 0.0
    timing_network_regime_transition_probabilities: tuple[float, ...] = ()
    timing_network_regime_transition_update_count: int = 0
    timing_network_regime_transition_calibration_count: int = 0
    timing_network_regime_transition_brier: float = 1.0
    timing_network_regime_transition_high_confidence_support: int = 0
    timing_network_regime_transition_high_confidence_precision: float = 0.0
    # Router observability is intentionally separate from the normalized
    # Direct-RNN input.  It proves that M15 routing saw native bridge ratios.
    regime_input_source: str = "policy_input_fallback"
    regime_input_audit: Mapping[str, object] = field(default_factory=dict)


@dataclass
class ChuyenGiaDocLap:
    ma_chuyen_gia: str
    ma_context: str
    ma_context_cha: str
    policy: Any
    trang_thai: TrangThaiChuyenGia = TrangThaiChuyenGia.HOAT_DONG_HAN_CHE
    so_lifecycle_da_dong: int = 0
    so_feedback_transition: int = 0
    so_quyet_dinh: int = 0
    fitness_cuc_bo: float = 0.0
    ma_phien_ban: str = "V001"
    ghi_chu: str = ""
    lich_su: list[dict[str, object]] = field(default_factory=list)

    @property
    def co_quyen_ra_lenh(self) -> bool:
        return self.trang_thai in {
            TrangThaiChuyenGia.THAM_DO_RUI_RO_THAP,
            TrangThaiChuyenGia.HOAT_DONG_HAN_CHE,
            TrangThaiChuyenGia.TRUONG_THANH,
        }

    def ra_quyet_dinh(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *,
        nhan_dien: NhanDienNguCanh | None = None,
    ):
        if not self.co_quyen_ra_lenh:
            raise RuntimeError(f"chuyen gia {self.ma_chuyen_gia} chua co quyen ra lenh")
        # Gắn owner *và contract context ổn định* trước policy.act. Các writer
        # nội bộ phải có thông tin này ngay ở event đầu tiên, không chờ facade
        # gắn metadata sau khi log đã được ghi.
        owner_metadata = {
            "ma_context": self.ma_context,
            "ma_context_cha": self.ma_context_cha,
            "ma_chuyen_gia": self.ma_chuyen_gia,
            "ma_policy": self.ma_chuyen_gia,
            "phien_ban_policy": self.ma_phien_ban,
            "trang_thai_chuyen_gia": self.trang_thai.value,
            "context_on_dinh": bool(nhan_dien.on_dinh) if nhan_dien is not None else True,
            "do_tin_cay_context": float(nhan_dien.do_tin_cay) if nhan_dien is not None else 1.0,
            "ly_do_context": str(nhan_dien.ly_do) if nhan_dien is not None else "context_owner_hoat_dong",
            "context_task": str(nhan_dien.nhiem_vu.value) if nhan_dien is not None else "",
            "candidate_context": str(nhan_dien.candidate_context or nhan_dien.ma_context) if nhan_dien is not None else "",
            "context_transition_active": bool(nhan_dien.transition_active) if nhan_dien is not None else False,
            "context_transition_progress": float(nhan_dien.transition_progress) if nhan_dien is not None else 0.0,
            "timing_quality": float(nhan_dien.timing_quality) if nhan_dien is not None else 1.0,
            "timing_direction": int(nhan_dien.timing_direction) if nhan_dien is not None else 0,
            "timing_source": str(nhan_dien.timing_source) if nhan_dien is not None else "TimingNet_disabled",
            "timing_network_quality": float(nhan_dien.timing_network_quality) if nhan_dien is not None else 0.0,
            "timing_network_direction": float(nhan_dien.timing_network_direction) if nhan_dien is not None else 0.0,
            "timing_network_entry_probability": float(nhan_dien.timing_network_entry_probability) if nhan_dien is not None else 0.0,
            "timing_network_uncertainty": float(nhan_dien.timing_network_uncertainty) if nhan_dien is not None else 1.0,
            "timing_network_deceleration_probability": float(nhan_dien.timing_network_deceleration_probability) if nhan_dien is not None else 0.0,
            "timing_network_deceleration_update_count": int(nhan_dien.timing_network_deceleration_update_count) if nhan_dien is not None else 0,
            "timing_network_trend_score": float(nhan_dien.timing_network_trend_score) if nhan_dien is not None else 0.0,
            "timing_network_update_count": int(nhan_dien.timing_network_update_count) if nhan_dien is not None else 0,
            "management_only": bool(getattr(self.policy, "_context_expert_management_only", False)),
            "context_direction_prior_strength": float(
                getattr(self.policy, "_context_direction_prior_strength", 0.10) or 0.10
            ),
            "candidate_transition_direction_prior_bat": bool(
                getattr(self.policy, "_candidate_transition_direction_prior_bat", True)
            ),
            "volatile_timing_direction_prior_bat": bool(
                getattr(self.policy, "_volatile_timing_direction_prior_bat", True)
            ),
            "stable_context_direction_prior_during_transition_bat": bool(
                getattr(self.policy, "_stable_context_direction_prior_during_transition_bat", True)
            ),
            "volatile_timing_direction_min_quality": float(
                getattr(self.policy, "_volatile_timing_direction_min_quality", 0.10) or 0.10
            ),
            "volatile_max_directional_strength": float(
                getattr(self.policy, "_volatile_max_directional_strength", 0.18) or 0.18
            ),
            "regime_input_source": str(
                getattr(nhan_dien, "regime_input_source", "policy_input_fallback")
            ) if nhan_dien is not None else "policy_input_fallback",
            "regime_input_audit": dict(
                getattr(nhan_dien, "regime_input_audit", {}) or {}
            ) if nhan_dien is not None else {},
            **dict(getattr(self.policy, "_context_expert_runtime_provenance", {}) or {}),
            **_policy_identity(self.policy),
        }
        try:
            setattr(self.policy, "_context_expert_owner_metadata", dict(owner_metadata))
        except Exception:
            pass
        trace = self.policy.act(observation, goal)
        # Bridge metadata is built from context_audit, while the direct policy
        # keeps selector diagnostics in action_details. Promote only the small
        # direction-prior audit fields so MT5 logs can verify the soft bias per
        # request without copying the full output vector into provenance.
        action_details = dict(getattr(trace, "action_details", {}) or {})
        prior_audit = {
            key: action_details[key]
            for key in (
                "context_direction_prior_applied",
                "context_direction_prior",
                "context_direction_prior_score",
                "context_direction_prior_source",
                "context_direction_prior_quality",
            )
            if key in action_details
        }
        if prior_audit:
            trace = _gan_trace_metadata(trace, prior_audit)
        self.so_quyet_dinh += 1
        return _gan_trace_metadata(trace, owner_metadata)

    def ghi_ket_qua(
        self,
        *,
        reward: float,
        cost: float,
        trace_id: str,
        source: str,
        lifecycle_confirmed: bool = False,
    ) -> None:
        """Ghi feedback nhưng không biến mỗi bar thành một lifecycle đã đóng."""
        if bool(lifecycle_confirmed):
            self.so_lifecycle_da_dong += 1
            # Router fitness chỉ dùng outcome đã đối soát, không dùng nhiễu
            # mark-to-market theo bar.
            self.fitness_cuc_bo = 0.95 * float(self.fitness_cuc_bo) + 0.05 * float(reward - cost)
        else:
            self.so_feedback_transition = int(getattr(self, "so_feedback_transition", 0)) + 1
        self.lich_su.append({
            "trace_id": str(trace_id),
            "reward": float(reward),
            "cost": float(cost),
            "source": str(source),
            "lifecycle_confirmed": bool(lifecycle_confirmed),
            "so_lifecycle_da_dong": int(self.so_lifecycle_da_dong),
            "so_feedback_transition": int(getattr(self, "so_feedback_transition", 0)),
        })
        if len(self.lich_su) > 1024:
            del self.lich_su[:-1024]


class BoNhanDienNguCanh:
    """Direct M15 market-regime detector with three volatile sub-regimes.

    Kết quả phân vùng dùng để chọn position owner và context generalist. Entry
    style cùng position management được giao cho các policy vai trò độc lập.
    """

    def __init__(self, cau_hinh: CauHinhHeChuyenGia, timing_network: TimingNetwork | None = None) -> None:
        self.cau_hinh = cau_hinh
        self._ma_hien_tai = ""
        # Import here to avoid circular imports
        from .market_regime_classifier import MarketRegimeClassifier
        self._classifier = MarketRegimeClassifier()
        self._timing_network = timing_network
        self._timing_network_direction_threshold = 0.10
        self._timing_network_learning_enabled = False
        self._last_timing_trace: dict[str, object] = {}
        self._last_regime_signature: tuple[float, float, float, float] | None = None
        self._last_regime_candidate: tuple[str, str, NhiemVuNguCanh, str] | None = None
        # TimingNet is an M15 predictor.  Reusing this immutable output inside
        # a closed-M15 interval prevents M1 decision frequency from becoming
        # a hidden training clock or repeatedly advancing its RNN state.
        self._last_timing_signature: tuple[float, float, float, float] | None = None
        self._last_timing_result: tuple[
            tuple[float, float, float, float],
            tuple[float, int, str, dict[str, object]],
        ] | None = None

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Migrate artifacts to direct M15 routing without confirmation state."""

        self.__dict__.update(dict(state))
        # Removed components are deliberately discarded: a new profile must
        # never resume an M1 candidate countdown or self-labelled MLP state.
        self.__dict__.pop("_timing", None)
        self.__dict__.pop("_ma_ung_vien", None)
        self.__dict__.pop("_dem_ung_vien", None)
        self.__dict__.pop("_classifier_updates", None)
        if not hasattr(self, "_timing_network"):
            self._timing_network = None
        if not hasattr(self, "_timing_network_direction_threshold"):
            self._timing_network_direction_threshold = 0.10
        if not hasattr(self, "_timing_network_learning_enabled"):
            self._timing_network_learning_enabled = False
        if not isinstance(getattr(self, "_last_timing_trace", None), Mapping):
            self._last_timing_trace = {}
        self._last_regime_signature = None
        self._last_regime_candidate = None
        self._last_timing_signature = None
        self._last_timing_result = None
        from .market_regime_classifier import MarketRegimeClassifier
        if not isinstance(getattr(self, "_classifier", None), MarketRegimeClassifier):
            self._classifier = MarketRegimeClassifier()

    def configure_feature_contract(self, feature_cfg: object | None) -> None:
        """Bind M15 classification and TimingNet to one MTF wire encoding."""

        mode = str(getattr(feature_cfg, "mtf_scaling_mode", "legacy_m1_atr_v1") or "legacy_m1_atr_v1").strip().lower()
        from .market_regime_classifier import MarketRegimeClassifier
        self._classifier = MarketRegimeClassifier(mtf_scaling_mode=mode)
        network = getattr(self, "_timing_network", None)
        if isinstance(network, TimingNetwork):
            network.configure_feature_contract(mtf_scaling_mode=mode)
        self._last_regime_signature = None
        self._last_regime_candidate = None
        self._last_timing_signature = None
        self._last_timing_result = None

    @property
    def timing_network(self) -> TimingNetwork | None:
        return getattr(self, "_timing_network", None)

    def configure_timing_network(self, learning_cfg: object | None) -> None:
        """Bind the single context TimingNet from the active learning profile."""

        enabled = bool(getattr(learning_cfg, "timing_network_enabled", False))
        if not enabled:
            self._timing_network = None
            return
        hidden_size = max(4, int(getattr(learning_cfg, "timing_network_hidden_size", 32) or 32))
        embedding_size = max(4, int(getattr(learning_cfg, "timing_network_embedding_size", 16) or 16))
        learning_rate = float(getattr(learning_cfg, "timing_network_learning_rate", 0.0005) or 0.0005)
        bptt_window = int(getattr(learning_cfg, "timing_network_bptt_window", 24) or 24)
        sequence_length = int(getattr(learning_cfg, "timing_network_sequence_length", 32) or 32)
        eligibility_decay = float(getattr(learning_cfg, "timing_network_eligibility_decay", 0.90) or 0.90)
        gradient_clip = float(getattr(learning_cfg, "timing_network_gradient_clip", 1.0) or 1.0)
        network = getattr(self, "_timing_network", None)
        if not isinstance(network, TimingNetwork):
            network = TimingNetwork.seed(
                hidden_size=hidden_size,
                embedding_size=embedding_size,
                learning_rate=learning_rate,
                bptt_window=bptt_window,
                sequence_length=sequence_length,
                eligibility_decay=eligibility_decay,
                gradient_clip=gradient_clip,
            )
        else:
            network.ensure_compatible()
            network.learning_rate = float(np.clip(learning_rate, 1e-6, 0.02))
            network.bptt_window = int(np.clip(bptt_window, 1, 128))
            network.eligibility_decay = float(np.clip(eligibility_decay, 0.5, 0.999))
            network.gradient_clip = float(np.clip(gradient_clip, 0.05, 10.0))
        network.configure_feature_contract(
            mtf_scaling_mode=str(getattr(self._classifier, "mtf_scaling_mode", "legacy_m1_atr_v1"))
        )
        network.configure_sequence({"timing_network_sequence_length": sequence_length})
        # Keep the optimizer contract inside the TimingNet checkpoint.  The
        # surrounding policy has no second optimizer and therefore cannot
        # accidentally move Policy/M1/graph parameters through this path.
        optimizer_names = (
            "timing_network_swats_enabled", "timing_network_pid_enabled",
            "timing_network_adam_beta1", "timing_network_adam_beta2",
            "timing_network_adam_epsilon", "timing_network_adamw_enabled",
            "timing_network_adamw_weight_decay", "timing_network_cosine_annealing_enabled",
            "timing_network_cosine_min_factor", "timing_network_cosine_cycle_steps",
            "timing_network_projection_beta",
            "timing_network_projection_tolerance", "timing_network_switch_consecutive",
            "timing_network_min_projection_steps", "timing_network_target_alignment",
            "timing_network_sgd_eta_min", "timing_network_sgd_eta_max",
            "timing_network_sgd_eta_init_factor", "timing_network_pid_kp",
            "timing_network_pid_ki", "timing_network_pid_kd",
            "timing_network_pid_integral_limit", "timing_network_pid_derivative_ema",
            "timing_network_pid_deadband", "timing_network_pid_r_target",
            "timing_network_pid_r_scale_floor", "timing_network_pid_alignment_weight",
            "timing_network_pid_r_weight", "timing_network_pid_bad_steps_to_freeze",
            "timing_network_pid_recovery_steps", "timing_network_pid_cooldown_steps",
            "timing_network_pid_fallback_to_adam",
        )
        network.configure_optimizer({
            name: value
            for name in optimizer_names
            for value in (getattr(learning_cfg, name, None),)
            if value is not None
        })
        network.configure_deceleration({
            name: value
            for name in (
                "timing_network_deceleration_enabled",
                "timing_network_deceleration_horizon",
                "timing_network_deceleration_min_trend_strength",
                "timing_network_deceleration_min_drop",
                "timing_network_deceleration_update_scale",
                "timing_network_deceleration_learning_rate",
                "timing_network_deceleration_encoder_scale",
                "timing_network_deceleration_l2",
            )
            for value in (getattr(learning_cfg, name, None),)
            if value is not None
        })
        network.configure_regime_transition({
            name: value
            for name in (
                "timing_network_regime_transition_enabled",
                "timing_network_regime_transition_horizons",
                "timing_network_regime_transition_update_scale",
                "timing_network_regime_transition_learning_rate",
                "timing_network_regime_transition_encoder_scale",
                "timing_network_regime_transition_l2",
            )
            for value in (getattr(learning_cfg, name, None),)
            if value is not None
        })
        self._timing_network = network

    def set_timing_network_learning_enabled(self, enabled: bool) -> None:
        """Separate frozen replay from causal online deceleration learning."""

        self._timing_network_learning_enabled = bool(enabled)

    def _timing_signal(
        self,
        observation: object,
        *,
        regime_label: str = "",
    ) -> tuple[float, int, str, dict[str, object]]:
        network = getattr(self, "_timing_network", None)
        if not isinstance(network, TimingNetwork):
            return 0.0, 0, "TimingNet_disabled", {}
        signature = getattr(self, "_last_regime_signature", None)
        cached = getattr(self, "_last_timing_result", None)
        if (
            signature is not None
            and isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] == signature
            and isinstance(cached[1], tuple)
        ):
            return cached[1]
        output = network.forward(
            observation,
            learn_deceleration=bool(getattr(self, "_timing_network_learning_enabled", False)),
            # The transition target is the direct state from the closed M15
            # bar. M1 requests reuse its cached state, so the label means
            # "M15 ownership will really change" without a countdown.
            regime_label=str(regime_label or ""),
            learn_regime_transition=bool(getattr(self, "_timing_network_learning_enabled", False)),
        )
        signal = output.signal_payload(
            direction_threshold=float(getattr(self, "_timing_network_direction_threshold", 0.10) or 0.10),
        )
        calibration = network.regime_transition_calibration(
            float(getattr(self.cau_hinh, "timing_regime_transition_entry_threshold", 0.78) or 0.78)
        )
        audit = {
            **signal,
            "timing_source": "TimingNet",
            "timing_network_enabled": True,
            "timing_network_update_count": int(getattr(network, "update_count", 0) or 0),
            "timing_network_swats_enabled": bool(getattr(network, "swats_enabled", False)),
            "timing_network_pid_enabled": bool(getattr(network, "pid_enabled", False)),
            "timing_network_adamw_enabled": bool(getattr(network, "adamw_enabled", False)),
            "timing_network_adamw_weight_decay": float(getattr(network, "adamw_weight_decay", 0.0) or 0.0),
            "timing_network_cosine_annealing_enabled": bool(getattr(network, "cosine_annealing_enabled", False)),
            "timing_network_cosine_min_factor": float(getattr(network, "cosine_min_factor", 1.0) or 1.0),
            "timing_network_cosine_cycle_steps": int(getattr(network, "cosine_cycle_steps", 0) or 0),
            "timing_network_optimizer_phase": str(getattr(network, "optimizer_phase", "legacy") or "legacy"),
            "timing_network_shadow_adam_step": int(getattr(network, "shadow_adam_step", 0) or 0),
            "timing_network_projection_count": int(getattr(network, "projection_count", 0) or 0),
            "timing_network_switch_count": int(getattr(network, "switch_count", 0) or 0),
            "timing_network_eta_sgd": float(getattr(network, "eta_sgd", 0.0) or 0.0),
            "timing_network_pid_alignment_ema": float(getattr(network, "pid_alignment_ema", 0.0) or 0.0),
            "timing_network_pid_r_ema": float(getattr(network, "pid_r_ema", 0.0) or 0.0),
            "timing_network_deceleration_enabled": bool(getattr(network, "deceleration_enabled", False)),
            "timing_network_deceleration_update_count": int(getattr(network, "deceleration_update_count", 0) or 0),
            "timing_network_deceleration_positive_count": int(getattr(network, "deceleration_positive_count", 0) or 0),
            "timing_network_regime_transition_enabled": bool(getattr(network, "regime_transition_enabled", False)),
            "timing_network_regime_transition_probability": float(output.regime_transition_probability),
            "timing_network_regime_transition_probabilities": [
                float(value) for value in output.regime_transition_probabilities
            ],
            "timing_network_regime_transition_update_count": int(getattr(network, "regime_transition_update_count", 0) or 0),
            "timing_network_regime_transition_calibration_count": int(calibration["total"]),
            "timing_network_regime_transition_brier": float(calibration["brier"]),
            "timing_network_regime_transition_high_confidence_support": int(calibration["high_confidence_support"]),
            "timing_network_regime_transition_high_confidence_precision": float(calibration["high_confidence_precision"]),
            "timing_network_trace": output.trace_payload(),
        }
        result = (
            float(signal["timing_quality"]),
            int(signal["timing_direction"]),
            "TimingNet",
            audit,
        )
        if signature is not None:
            self._last_timing_signature = signature
            self._last_timing_result = (signature, result)
        return result

    def reset_runtime_state(self) -> None:
        """Reset M15 cache and runtime trace state for a new MT5 session."""
        self._ma_hien_tai = ""
        self._last_regime_signature = None
        self._last_regime_candidate = None
        self._last_timing_trace = {}
        self._last_timing_signature = None
        self._last_timing_result = None
        network = getattr(self, "_timing_network", None)
        if isinstance(network, TimingNetwork):
            network.reset_runtime_state()

    @staticmethod
    def _safe(values: object, size: int) -> np.ndarray:
        raw = np.asarray(values, dtype=float).reshape(-1)
        result = np.zeros(size, dtype=float)
        result[: min(size, raw.size)] = raw[:size]
        return np.nan_to_num(result, nan=0.0, posinf=2.0, neginf=-2.0)

    @staticmethod
    def _market_features_from_observation(observation: object) -> np.ndarray:
        raw = np.asarray(observation, dtype=float).reshape(-1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=2.0, neginf=-2.0)
        if raw.size in (117, 121, INPUT_COUNT):
            # Versioned EA layout: 29 base market, 40 position, 8 account,
            # 29 extra market, 11 execution, then timing/Wyckoff tail slots.
            return np.concatenate([raw[:29], raw[77:106], raw[TIMING_OFFSET:raw.size]])
        if raw.size >= 119:
            return np.concatenate([raw[:34], raw[82:108]])
        return raw[: min(raw.size, 58)]

    @staticmethod
    def _pending_order_count(execution_context: Mapping[str, object] | None) -> float:
        if not isinstance(execution_context, Mapping):
            return 0.0
        try:
            pending = float(execution_context.get("pending_orders", 0.0) or 0.0)
        except (TypeError, ValueError):
            pending = 0.0
        orders_pipe = str(execution_context.get("orders_pipe", "") or "")
        if orders_pipe.strip():
            pending = max(pending, float(sum(1 for item in orders_pipe.split("^") if item.strip())))
        return float(max(0.0, pending))

    @staticmethod
    def _timing_signal_from_reason(reason: str) -> tuple[float, int]:
        match = re.search(r"timing_q=([-+]?\d+(?:\.\d+)?)_dir=(-?\d+)", str(reason or ""))
        if match is None:
            # Custom/test classifiers from older artifacts may not emit the
            # timing marker. Keep their legacy permissive behavior.
            return 1.0, 0
        try:
            quality = float(np.clip(float(match.group(1)), 0.0, 1.0))
            direction = int(match.group(2))
        except (TypeError, ValueError):
            return 1.0, 0
        return quality, direction

    def _toa_do(self, observation: object, execution_context: Mapping[str, object] | None = None) -> tuple[float, ...]:
        raw_size = np.asarray(observation, dtype=float).reshape(-1).size
        x = self._safe(observation, max(INPUT_COUNT, raw_size))
        market = self._market_features_from_observation(observation)
        if raw_size in (117, 121, INPUT_COUNT):
            position = x[29:69]
            account = x[69:77]
            execution = x[106:117]
        else:
            position = x[34:74]
            account = x[74:82]
            execution = x[108:119]
        # Mọi chiều đều là mô tả state; không chiều nào mã hóa action.
        mode = str(getattr(self._classifier, "mtf_scaling_mode", "legacy_m1_atr_v1"))
        trend = float(self._classifier.directional_score(market, mtf_scaling_mode=mode))
        volatility = float(self._classifier.volatility_score(market, mtf_scaling_mode=mode))
        position_energy = float(np.clip(np.linalg.norm(position) / max(1.0, np.sqrt(position.size)), 0.0, 2.0))
        account_risk = float(np.clip(np.mean(np.abs(account)), 0.0, 2.0))
        execution_quality = float(np.clip(np.mean(np.abs(execution)), 0.0, 2.0))
        side_signal = float(np.tanh(position[0])) if position.size else 0.0
        pending_pressure = float(np.clip(self._pending_order_count(execution_context), 0.0, 2.0))
        return tuple(round(value, 6) for value in (
            trend, volatility, position_energy, account_risk, execution_quality, side_signal, pending_pressure
        ))

    def _ung_vien(
        self,
        observation: object,
        execution_context: Mapping[str, object] | None = None,
    ) -> tuple[str, str, NhiemVuNguCanh, tuple[float, ...], str]:
        """Resolve current regime from the latest closed M15 bridge tuple."""
        from .market_regime_classifier import REGIME_NAMES, REGIME_VOLATILE
        market = self._market_features_from_observation(observation)
        signature = self._classifier.m15_signature(
            market,
            mtf_scaling_mode=str(getattr(self._classifier, "mtf_scaling_mode", "legacy_m1_atr_v1")),
        )
        cached = getattr(self, "_last_regime_candidate", None)
        if signature != getattr(self, "_last_regime_signature", None) or cached is None:
            regime_id, probs = self._classifier.deterministic_regime(
                market,
                volatile_max_directional_strength=float(self.cau_hinh.volatile_max_directional_strength),
            )
            regime_name = REGIME_NAMES.get(regime_id, "RANGING")
            context_parent = f"CTX_{regime_name}"
            if regime_id == REGIME_VOLATILE and bool(getattr(self.cau_hinh, "volatile_subexperts_bat", True)):
                subtype = self._classifier.volatile_subregime(
                    market,
                    spike_threshold=float(getattr(self.cau_hinh, "volatile_spike_threshold", 0.55) or 0.55),
                    expansion_threshold=float(getattr(self.cau_hinh, "volatile_expansion_threshold", 1.15) or 1.15),
                )
                regime_name = f"VOLATILE_{subtype}"
                ctx_id = f"CTX_{regime_name}"
                context_parent = "CTX_VOLATILE"
            else:
                ctx_id = context_parent
            regime_map = {
                0: NhiemVuNguCanh.TANG_MANH,
                1: NhiemVuNguCanh.TANG_YEU,
                2: NhiemVuNguCanh.GIAM_MANH,
                3: NhiemVuNguCanh.GIAM_YEU,
                4: NhiemVuNguCanh.RANGING,
                5: NhiemVuNguCanh.VOLATILE,
            }
            nhiem_vu = regime_map.get(regime_id, NhiemVuNguCanh.RANGING)
            if regime_id == REGIME_VOLATILE and ctx_id != "CTX_VOLATILE":
                nhiem_vu = self._nhiem_vu_tu_context(ctx_id, NhiemVuNguCanh.VOLATILE)
            confidence = float(probs[regime_id])
            cached = (
                ctx_id,
                context_parent,
                nhiem_vu,
                f"regime_timeframe=M15;regime={regime_name};confidence={confidence:.2f}",
            )
            self._last_regime_signature = signature
            self._last_regime_candidate = cached
        else:
            ctx_id, context_parent, nhiem_vu, _reason = cached
        toa_do = self._toa_do(observation, execution_context)
        return ctx_id, context_parent, nhiem_vu, toa_do, cached[3]

    @staticmethod
    def _nhiem_vu_tu_context(ma_context: str, fallback: NhiemVuNguCanh) -> NhiemVuNguCanh:
        mapping = {
            "CTX_TANG_MANH": NhiemVuNguCanh.TANG_MANH,
            "CTX_TANG_YEU": NhiemVuNguCanh.TANG_YEU,
            "CTX_GIAM_MANH": NhiemVuNguCanh.GIAM_MANH,
            "CTX_GIAM_YEU": NhiemVuNguCanh.GIAM_YEU,
            "CTX_RANGING": NhiemVuNguCanh.RANGING,
            "CTX_VOLATILE": NhiemVuNguCanh.VOLATILE,
            "CTX_VOLATILE_SPIKE": NhiemVuNguCanh.VOLATILE_SPIKE,
            "CTX_VOLATILE_EXPANSION": NhiemVuNguCanh.VOLATILE_EXPANSION,
            "CTX_VOLATILE_CHOP": NhiemVuNguCanh.VOLATILE_CHOP,
        }
        return mapping.get(str(ma_context), fallback)

    def _regime_input_audit(self, observation: object) -> dict[str, float]:
        """Expose exactly the scalar view consumed by the M15 classifier."""

        market = self._market_features_from_observation(observation)
        mode = str(getattr(self._classifier, "mtf_scaling_mode", "legacy_m1_atr_v1"))
        m15_atr = abs(self._classifier._at(market, 38, default=0.0))
        h1_atr = abs(self._classifier._at(market, 42, default=0.0))
        if mode == "native_tf_atr_v2":
            m15_over_h1 = self._classifier._native_tf_ratio(
                self._classifier._at(market, 38, limit=0.999999),
                self._classifier._at(market, 42, limit=0.999999),
            )
        else:
            m15_over_h1 = max(0.10, m15_atr) / max(0.10, h1_atr)
        return {
            "m15_atr_over_m1": float(m15_atr),
            "h1_atr_over_m1": float(h1_atr),
            "m15_over_h1_atr": float(m15_over_h1),
            "directional_score": float(self._classifier.directional_score(market, mtf_scaling_mode=mode)),
            "volatility_score": float(self._classifier.volatility_score(market, mtf_scaling_mode=mode)),
            "chop_score": float(self._classifier.chop_score(market, mtf_scaling_mode=mode)),
        }

    def nhan_dien(
        self,
        observation: object,
        execution_context: Mapping[str, object] | None = None,
        *,
        raw_regime_observation: object | None = None,
    ) -> NhanDienNguCanh:
        regime_observation = observation
        regime_input_source = "policy_input_fallback"
        if raw_regime_observation is not None:
            try:
                raw_candidate = np.asarray(raw_regime_observation, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                raw_candidate = np.zeros(0, dtype=float)
            if raw_candidate.size:
                regime_observation = raw_candidate.copy()
                regime_input_source = "bridge_raw"
        self._last_timing_audit = {}
        ma, cha, nhiem_vu, toa_do, ly_do = self._ung_vien(regime_observation, execution_context)
        # The closed M15 bar is the authority.  A changed M15 label becomes
        # active immediately; no M1 candidate count can postpone it.
        self._ma_hien_tai = ma
        active = ma
        active_task = nhiem_vu
        # TimingNet sees only the causal feature vector.  The attached label
        # is the direct M15 context at this decision, so the delayed head can
        # learn an upcoming M15 change without future-bar leakage.
        timing_quality, timing_direction, timing_source, timing_audit = self._timing_signal(
            regime_observation,
            regime_label=active,
        )
        self._last_timing_audit = timing_audit
        self._last_timing_trace = dict(timing_audit.get("timing_network_trace", {}) or {})
        timing_tag = f"timing_q={timing_quality:.2f}_dir={timing_direction}_src={timing_source}"
        confidence = float(np.clip(
            self._classifier.rule_based_confidence(self._market_features_from_observation(regime_observation)),
            0.0,
            1.0,
        ))
        regime_input_audit = {
            "raw_bridge": self._regime_input_audit(regime_observation),
            "policy_input": self._regime_input_audit(observation),
        }
        reason = (
            f"active={active};regime_timeframe=M15;direct_switch=true;regime_input={regime_input_source};"
            f"{ly_do}_{timing_tag}"
        )
        return NhanDienNguCanh(
            ma_context=active,
            ma_context_cha=("CTX_VOLATILE" if active.startswith("CTX_VOLATILE_") else active),
            nhiem_vu=active_task,
            toa_do=toa_do,
            do_tin_cay=confidence,
            on_dinh=True,
            ly_do=reason,
            timing_quality=timing_quality,
            timing_direction=timing_direction,
            candidate_context=str(active),
            transition_active=False,
            transition_progress=1.0,
            timing_network_update_count=int(timing_audit.get("timing_network_update_count", 0) or 0),
            timing_source=str(timing_audit.get("timing_source", "TimingNet_disabled") or "TimingNet_disabled"),
            timing_network_quality=float(timing_audit.get("timing_network_quality", 0.0) or 0.0),
            timing_network_direction=float(timing_audit.get("timing_network_direction", 0.0) or 0.0),
            timing_network_entry_probability=float(timing_audit.get("timing_network_entry_probability", 0.0) or 0.0),
            timing_network_uncertainty=float(timing_audit.get("timing_network_uncertainty", 1.0) or 1.0),
            timing_network_deceleration_probability=float(timing_audit.get("timing_network_deceleration_probability", 0.0) or 0.0),
            timing_network_deceleration_update_count=int(timing_audit.get("timing_network_deceleration_update_count", 0) or 0),
            timing_network_trend_score=float(timing_audit.get("timing_network_trend_score", 0.0) or 0.0),
            timing_network_regime_transition_probability=float(timing_audit.get("timing_network_regime_transition_probability", 0.0) or 0.0),
            timing_network_regime_transition_probabilities=tuple(
                float(value) for value in (timing_audit.get("timing_network_regime_transition_probabilities", []) or [])
            ),
            timing_network_regime_transition_update_count=int(timing_audit.get("timing_network_regime_transition_update_count", 0) or 0),
            timing_network_regime_transition_calibration_count=int(timing_audit.get("timing_network_regime_transition_calibration_count", 0) or 0),
            timing_network_regime_transition_brier=float(timing_audit.get("timing_network_regime_transition_brier", 1.0) or 1.0),
            timing_network_regime_transition_high_confidence_support=int(timing_audit.get("timing_network_regime_transition_high_confidence_support", 0) or 0),
            timing_network_regime_transition_high_confidence_precision=float(timing_audit.get("timing_network_regime_transition_high_confidence_precision", 0.0) or 0.0),
            regime_input_source=regime_input_source,
            regime_input_audit=regime_input_audit,
        )


@dataclass
class SoDangKyChuyenGia:
    cau_hinh: CauHinhHeChuyenGia
    chuyen_gia_theo_context: dict[str, ChuyenGiaDocLap] = field(default_factory=dict)

    def dang_ky(self, chuyen_gia: ChuyenGiaDocLap) -> None:
        if chuyen_gia.ma_context in self.chuyen_gia_theo_context:
            raise ValueError(f"context da co owner: {chuyen_gia.ma_context}")
        self.chuyen_gia_theo_context[chuyen_gia.ma_context] = chuyen_gia

    def owner(self, nhan_dien: NhanDienNguCanh) -> ChuyenGiaDocLap | None:
        shared = self.chuyen_gia_theo_context.get("CTX_CHUNG")
        if shared is not None:
            return shared
        # Legacy context-owner mode.
        expert = self.chuyen_gia_theo_context.get(nhan_dien.ma_context)
        if expert is not None and expert.co_quyen_ra_lenh:
            return expert
        return self.chuyen_gia_theo_context.get(nhan_dien.ma_context_cha)

class HeChuyenGiaMotContextMotPolicy:
    """Mặt ngoài tương thích; chế độ mới dùng đúng một bộ não chung."""

    _pending_regime_probabilities: tuple[float, ...] | None = None
    _pending_regime_name: str | None = None
    _pending_regime_index: int | None = None
    _pending_regime_raw_inputs: Any = None

    def __init__(
        self,
        *,
        cau_hinh: CauHinhHeChuyenGia,
        tao_policy: Callable[[], Any],
        registry: SoDangKyChuyenGia | None = None,
    ) -> None:
        self.cau_hinh = cau_hinh
        self.tao_policy = tao_policy
        self.nhan_dien = BoNhanDienNguCanh(cau_hinh)
        self.registry = registry or SoDangKyChuyenGia(cau_hinh)
        self._trace_to_expert: dict[str, str] = {}
        self._trace_to_timing_network_trace: dict[str, Mapping[str, object]] = {}
        self._trace_to_selector_policy: dict[str, str] = {}
        self._trace_to_selector_context: dict[str, str] = {}
        self._trace_to_selector_evidence_group: dict[str, str] = {}
        self._selector_pending_entry_traces: dict[str, tuple[str, str, int]] = {}
        self._selector_pending_entry_ages: dict[str, int] = {}
        self._selector_credited_trace_ids: dict[str, None] = {}
        # A cold role receives a bounded causal probe budget.  Zero keeps the
        # legacy unlimited behavior for direct/unit-test configurations;
        # production profiles set a small positive budget.
        self._selector_cold_start_probe_count = 0
        self._learning_enabled_state: bool = True
        self.policy_selector = PolicyEnsembleSelector(
            fallback_policy_id="",
            min_score=-1.0,
            cold_start_penalty=float(getattr(self.cau_hinh, "selector_cold_start_penalty", 0.0)),
            cold_start_support=int(getattr(self.cau_hinh, "selector_cold_start_support", 8)),
            inflight_penalty=float(getattr(self.cau_hinh, "selector_inflight_penalty", 0.20)),
            exposure_penalty=float(getattr(self.cau_hinh, "selector_exposure_penalty", 0.20)),
            loss_streak_penalty=float(getattr(self.cau_hinh, "selector_loss_streak_penalty", 0.12)),
            policy_group_loss_streak_penalty=float(
                getattr(self.cau_hinh, "selector_policy_group_loss_streak_penalty", 0.0)
            ),
        )
        self.exposure_supervisor = CorrelatedExposureSupervisor(
            risk_decay=float(self.cau_hinh.correlated_exposure_risk_decay),
            min_risk_pct=float(self.cau_hinh.correlated_exposure_min_risk_pct),
        )
        self.role_readiness_supervisor = RoleReadinessSupervisor(
            min_risk_multiplier=float(self.cau_hinh.role_readiness_min_risk_multiplier),
            support_scale=float(self.cau_hinh.role_readiness_support_scale),
            negative_utility_weight=float(self.cau_hinh.role_readiness_negative_utility_weight),
        )
        self._last_expert_id = ""
        self._pending_account_context: tuple[dict[str, float], str] | None = None
        self._pending_decision_context: dict[str, object] = {}
        self._pending_regime_probabilities: tuple[float, ...] | None = None
        self._pending_regime_name: str | None = None
        self._pending_regime_index: int | None = None
        self._pending_starvation: Mapping[str, object] | None = None
        # MT5 resolves semantic/emitted/wire action only after ``act`` returns.
        # When this flag is set by the bridge, telemetry is deferred so every
        # decision event is emitted from the final immutable bridge trace.
        self._defer_live_decision_until_bridge = False
        self._bridge_finalized_trace_ids: dict[str, None] = {}
        self._risk_min_pct = 0.001
        self._risk_max_pct = 0.02
        self._no_decision_sequence = 0
        self._root_run_id = ""
        self._expert_run_ids: dict[str, str] = {}
        self._he_chuyen_gia_live_settings: object | None = None
        self._he_chuyen_gia_live_metadata: dict[str, object] = {}
        self._he_chuyen_gia_live_source = "he_chuyen_gia"
        # Facade owns telemetry emission. Inner policy observers are disabled
        # so context/growth metadata is attached BEFORE a decision is logged.
        self._expert_live_writers: dict[str, object] = {}
        # Stored oldest-to-newest. This is runtime sequence state and resets
        # with a new MT5 session, just like recurrent activations.
        self._context_history: list[str] = []
        self._seed_root_experts()
        self._ensure_volatile_subexperts(warm_start=False)
        self._ensure_role_specialists(warm_start=False)
        self._ensure_context_history_policy_inputs()

    def _context_history_width(self) -> int:
        return (
            CONTEXT_HISTORY_INPUT_COUNT
            if int(getattr(self.cau_hinh, "context_history_inputs", 0) or 0)
            == CONTEXT_HISTORY_INPUT_COUNT
            else 0
        )

    def _entry_feature_interaction_width(self) -> int:
        return entry_feature_interaction_width(
            getattr(self.cau_hinh, "entry_feature_enrichment_mode", ENTRY_FEATURE_ENRICHMENT_LEGACY)
        )

    def _context_history_audit(self) -> dict[str, object]:
        width = self._context_history_width()
        values, labels = encode_context_history(
            getattr(self, "_context_history", ()),
            count=width,
        )
        return {
            "context_history_inputs_enabled": bool(width),
            "context_history_input_count": int(width),
            "context_history_previous_contexts": list(labels),
            "context_history_values": [float(value) for value in values.tolist()],
            "context_history_wire_input_count": int(INPUT_COUNT),
            "context_history_policy_input_count": int(
                INPUT_COUNT + width + self._entry_feature_interaction_width()
            ),
            "entry_feature_enrichment_mode": normalize_entry_feature_enrichment_mode(
                getattr(self.cau_hinh, "entry_feature_enrichment_mode", ENTRY_FEATURE_ENRICHMENT_LEGACY)
            ),
            "entry_feature_interaction_count": int(self._entry_feature_interaction_width()),
            "entry_feature_quarantine_scale": float(
                getattr(self.cau_hinh, "entry_feature_quarantine_scale", 1.0)
            ),
            "entry_feature_h1_direct_scale": float(
                getattr(self.cau_hinh, "entry_feature_h1_direct_scale", 1.0)
            ),
        }

    def _policy_observation_with_context_history(self, observation: object) -> tuple[np.ndarray, dict[str, object]]:
        """Append causal local fields while retaining the immutable wire prefix."""

        augmented, _labels = append_context_history_inputs(
            observation,
            getattr(self, "_context_history", ()),
            count=self._context_history_width(),
        )
        return append_entry_quality_interactions(
            augmented,
            mode=getattr(self.cau_hinh, "entry_feature_enrichment_mode", ENTRY_FEATURE_ENRICHMENT_LEGACY),
        )

    def _record_current_context(self, context: object) -> None:
        """Advance history only after every policy has consumed t-1/t-2."""

        if self._context_history_width() <= 0:
            return
        from .context_history import canonical_context_token

        token = canonical_context_token(context)
        if not token:
            return
        history = [str(item) for item in list(getattr(self, "_context_history", ()) or [])]
        history.append(token)
        self._context_history = history[-CONTEXT_HISTORY_INPUT_COUNT:]

    def _ensure_context_history_policy_inputs(self) -> None:
        """Resize only local Direct-RNN encoders; never change the MT5 wire."""

        local_width = self._context_history_width() + self._entry_feature_interaction_width()
        # With the legacy representation there is no local tail to reconcile.
        # Keeping this fast return preserves the behaviour of external/test
        # policy implementations which deliberately are not Direct-RNNs.
        if local_width <= 0:
            return
        # Local import avoids Config -> context facade -> policy cycle at
        # module initialization time.
        from .chuyen_gia_policy import ChuyenGiaPolicy

        for expert in self.registry.chuyen_gia_theo_context.values():
            policy = getattr(expert, "policy", None)
            if isinstance(policy, ChuyenGiaPolicy):
                policy.entry_feature_enrichment_mode = normalize_entry_feature_enrichment_mode(
                    getattr(self.cau_hinh, "entry_feature_enrichment_mode", ENTRY_FEATURE_ENRICHMENT_LEGACY)
                )
                policy.entry_feature_quarantine_scale = float(
                    getattr(self.cau_hinh, "entry_feature_quarantine_scale", 1.0)
                )
                policy.entry_feature_h1_direct_scale = float(
                    getattr(self.cau_hinh, "entry_feature_h1_direct_scale", 1.0)
                )
                policy.ensure_compatible(target_input_size=INPUT_COUNT + local_width)

    def _seed_root_experts(self) -> None:
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) in {"mot_bo_nao_dung_chung", "nam_tac_nhan_dung_chung"}:
            if "CTX_CHUNG" not in self.registry.chuyen_gia_theo_context:
                self.registry.dang_ky(ChuyenGiaDocLap(
                    ma_chuyen_gia="BO_NAO_CHUNG",
                    ma_context="CTX_CHUNG",
                    ma_context_cha="CTX_CHUNG",
                    policy=self.tao_policy(),
                    trang_thai=TrangThaiChuyenGia.HOAT_DONG_HAN_CHE,
                    ghi_chu="mot bo trong so dung chung cho moi trang thai thi truong",
                ))
            return
        roots = (
            ("CTX_TANG_MANH", NhiemVuNguCanh.TANG_MANH),
            ("CTX_TANG_YEU", NhiemVuNguCanh.TANG_YEU),
            ("CTX_GIAM_MANH", NhiemVuNguCanh.GIAM_MANH),
            ("CTX_GIAM_YEU", NhiemVuNguCanh.GIAM_YEU),
            ("CTX_RANGING", NhiemVuNguCanh.RANGING),
            ("CTX_VOLATILE", NhiemVuNguCanh.VOLATILE),
        )
        for ma_context, _ in roots:
            if ma_context in self.registry.chuyen_gia_theo_context:
                continue
            suffix = ma_context.removeprefix("CTX_GOC_")
            self.registry.dang_ky(ChuyenGiaDocLap(
                ma_chuyen_gia=f"CG_GOC_{suffix}",
                ma_context=ma_context,
                ma_context_cha=ma_context,
                policy=self.tao_policy(),
                trang_thai=TrangThaiChuyenGia.HOAT_DONG_HAN_CHE,
                ghi_chu="chuyen gia goc, policy doc lap",
            ))

    def _new_independent_policy(self, parent: ChuyenGiaDocLap | None) -> Any:
        """Create one fresh policy with the loaded system's public shape."""
        factory = getattr(self, "tao_policy", None)
        if callable(factory):
            child_policy = factory()
        else:
            # Pickled facades intentionally do not retain a closure. Rebuild a
            # fresh policy from the loaded parent's public shape instead of
            # falling back to a copied broad-regime policy.
            from .chuyen_gia_policy import ChuyenGiaPolicy

            source_policy = getattr(parent, "policy", None)
            policy_type = type(source_policy) if source_policy is not None else ChuyenGiaPolicy
            kwargs = {
                "input_size": int(getattr(source_policy, "input_size", INPUT_COUNT) or INPUT_COUNT),
                "output_size": int(getattr(source_policy, "output_size", SO_DAU_RA_BRIDGE) or SO_DAU_RA_BRIDGE),
            }
            for field_name in ("m1_backend", "decision_backend"):
                value = getattr(source_policy, field_name, None)
                if value:
                    kwargs[field_name] = value
            try:
                child_policy = policy_type(**kwargs)
            except TypeError:
                child_policy = ChuyenGiaPolicy(**kwargs)
        return child_policy

    def _fork_volatile_policy(self, parent: ChuyenGiaDocLap | None, *, warm_start: bool) -> Any:
        """Create a genuinely independent policy for a volatile subtype.

        The parent is a broad volatile policy. Copying its neural weights looks
        like a warm start, but it also copies an action preference learned on a
        mixture of regimes. Subexperts therefore start fresh and learn only
        from their own lifecycle data.
        """
        child_policy = self._new_independent_policy(parent)
        setattr(child_policy, "_volatile_subexpert_bootstrap", {
            "source_expert": getattr(parent, "ma_chuyen_gia", "") if parent is not None else "",
            "source_context": getattr(parent, "ma_context", "") if parent is not None else "",
            "mode": "fresh_independent_policy",
            "requested_warm_start": bool(warm_start),
        })
        return child_policy

    def _fork_role_policy(self, role: str, parent: ChuyenGiaDocLap | None, *, warm_start: bool) -> Any:
        source_policy = getattr(parent, "policy", None)
        # Role specialists need the v13 decision surface, not a second copy of
        # every historical episode/causal cell. Start with a fresh independent
        # policy and transplant only forward/parameter weights. This prevents
        # MARKET/LIMIT/STOP/MANAGEMENT from carrying identical 8k-cell memory
        # stores that later receive contradictory role-local credit.
        policy = self._new_independent_policy(parent)
        if bool(warm_start) and source_policy is not None:
            # Role forking copies ModuleState objects directly to retain only
            # the decision surface.  Migrate the loaded parent first: old v13
            # modules can lack newer lifecycle/value-head fields, and this
            # direct copy path must not bypass the policy compatibility hook.
            ensure_source = getattr(source_policy, "ensure_compatible", None)
            if callable(ensure_source):
                ensure_source()
            source_modules = getattr(source_policy, "modules", None)
            if isinstance(source_modules, list) and hasattr(policy, "modules"):
                policy.modules = [
                    module.copy() if callable(getattr(module, "copy", None)) else copy.deepcopy(module)
                    for module in source_modules
                ]
            source_edges = getattr(source_policy, "lateral_edges", None)
            if isinstance(source_edges, Mapping) and hasattr(policy, "lateral_edges"):
                policy.lateral_edges = {
                    str(key): np.asarray(value, dtype=float).copy()
                    for key, value in source_edges.items()
                }
            for field_name in (
                "neural_utility",
                "risk_cost_gate",
                "action_parameter_head",
            ):
                source_component = getattr(source_policy, field_name, None)
                if source_component is None or not hasattr(policy, field_name):
                    continue
                copier = getattr(source_component, "copy", None)
                setattr(
                    policy,
                    field_name,
                    copier() if callable(copier) else copy.deepcopy(source_component),
                )
            for field_name in ("m1_backend", "decision_backend", "m1_components", "gating_mode"):
                if hasattr(source_policy, field_name) and hasattr(policy, field_name):
                    setattr(policy, field_name, copy.deepcopy(getattr(source_policy, field_name)))
        # Provenance and execution caches always describe a new role-local
        # brain with fresh memory and no inherited live ownership.
        for field_name in (
            "_context_expert_policy_instance_uuid",
            "_context_expert_graph_instance_uuid",
            "_context_expert_memory_instance_uuid",
            "_context_expert_owner_metadata",
            "_context_expert_runtime_provenance",
        ):
            try:
                delattr(policy, field_name)
            except (AttributeError, TypeError):
                pass
        for cache_name in (
            "_execution_trace_cache",
            "_pending_execution_traces",
            "_causal_trace_archive",
            "_causal_reports",
            "_causal_report_signatures",
        ):
            cache = getattr(policy, cache_name, None)
            if isinstance(cache, dict):
                cache.clear()
        reset = getattr(policy, "reset_runtime_state", None)
        if callable(reset):
            reset()
        setattr(policy, "_role_specialist_bootstrap", {
            "role": str(role),
            "source_expert": getattr(parent, "ma_chuyen_gia", "") if parent is not None else "",
            "source_context": getattr(parent, "ma_context", "") if parent is not None else "",
            "mode": "context_owner_weight_warm_start_fresh_memory" if warm_start else "fresh_independent_policy",
            "requested_warm_start": bool(warm_start),
        })
        return policy

    def _ensure_volatile_subexperts(self, *, warm_start: bool) -> None:
        """Ensure volatile owners only for the legacy multi-owner mode."""
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) in {"mot_bo_nao_dung_chung", "nam_tac_nhan_dung_chung"}:
            return
        if not bool(getattr(self.cau_hinh, "volatile_subexperts_bat", True)):
            return
        parent = self.registry.chuyen_gia_theo_context.get("CTX_VOLATILE")
        for subtype in VOLATILE_SUBREGIMES:
            ma_context = f"CTX_VOLATILE_{subtype}"
            if ma_context in self.registry.chuyen_gia_theo_context:
                # Artifacts created by the previous split copied the broad
                # volatile parent's weights. Replace that one-time migration
                # state; otherwise the bad parent action preference survives
                # every subsequent run even though the memory stores are local.
                existing = self.registry.chuyen_gia_theo_context[ma_context]
                bootstrap = getattr(existing.policy, "_volatile_subexpert_bootstrap", {})
                if (
                    warm_start
                    and isinstance(bootstrap, Mapping)
                    and bootstrap.get("mode") == "warm_start_fork_independent_memory"
                ):
                    existing.policy = self._fork_volatile_policy(parent, warm_start=False)
                continue
            policy = self._fork_volatile_policy(parent, warm_start=warm_start)
            self.registry.dang_ky(ChuyenGiaDocLap(
                ma_chuyen_gia=f"CG_GOC_CTX_VOLATILE_{subtype}",
                ma_context=ma_context,
                ma_context_cha="CTX_VOLATILE",
                policy=policy,
                trang_thai=TrangThaiChuyenGia.HOAT_DONG_HAN_CHE,
                ghi_chu=(
                    "volatile subexpert doc lap: "
                    f"{subtype}; parent=CTX_VOLATILE; warm_start={bool(warm_start)}"
                ),
            ))

    def _ensure_role_specialists(self, *, warm_start: bool) -> None:
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) in {"mot_bo_nao_dung_chung", "nam_tac_nhan_dung_chung"}:
            return
        """Remove the replaced global-role layer.

        Context-role specialists are created lazily from the active context
        owner. This keeps artifacts bounded to contexts actually observed and
        prevents one global specialist from mixing contradictory regimes.
        """
        removed_policy_ids: set[str] = set()
        for role_context in LEGACY_GLOBAL_ROLE_CONTEXTS:
            expert = self.registry.chuyen_gia_theo_context.pop(role_context, None)
            if expert is not None:
                removed_policy_ids.add(str(expert.ma_chuyen_gia))
        if not removed_policy_ids:
            return
        selector = getattr(self, "policy_selector", None)
        for field_name in ("learned_utility", "outcome_count"):
            values = getattr(selector, field_name, None)
            if isinstance(values, dict):
                setattr(selector, field_name, {
                    key: value
                    for key, value in values.items()
                    if not any(f"{policy_id}::" in str(key) for policy_id in removed_policy_ids)
                })

    def _prune_dynamic_context_branches(self) -> None:
        """One-way migration from removed dynamic context branches.

        A fixed M15 taxonomy is essential for Full-RL: every confirmed
        lifecycle must accumulate in its canonical owner instead of being
        siphoned into sparse branch policies. Old artifacts can still load,
        but their branch owners and every selector reference are discarded.
        """
        removed_contexts = {
            str(context)
            for context in self.registry.chuyen_gia_theo_context
            if "_NHANH_" in str(context)
        }
        removed_policy_ids = {
            str(self.registry.chuyen_gia_theo_context[context].ma_chuyen_gia)
            for context in removed_contexts
        }
        for context in removed_contexts:
            self.registry.chuyen_gia_theo_context.pop(context, None)
        selector = getattr(self, "policy_selector", None)
        for field_name in ("learned_utility", "outcome_count"):
            values = getattr(selector, field_name, None)
            if isinstance(values, dict):
                setattr(selector, field_name, {
                    key: value for key, value in values.items()
                    if not any(f"{policy_id}::" in str(key) for policy_id in removed_policy_ids)
                })
        for field_name in (
            "_trace_to_expert", "_trace_to_selector_policy",
            "_trace_to_selector_context", "_trace_to_selector_evidence_group",
        ):
            values = getattr(self, field_name, None)
            if isinstance(values, Mapping):
                setattr(self, field_name, {
                    key: value for key, value in values.items()
                    if str(value) not in removed_policy_ids and str(value) not in removed_contexts
                })
        writers = getattr(self, "_expert_live_writers", None)
        if isinstance(writers, dict):
            self._expert_live_writers = {
                key: value for key, value in writers.items()
                if str(key) not in removed_policy_ids
            }
        # The following are obsolete serialized fields only. They are removed
        # from a resumed artifact so future persistence cannot revive branches.
        for legacy_field in (
            "context_con_theo_cha",
            "mau_context",
            "ban_kinh_context",
        ):
            self.registry.__dict__.pop(legacy_field, None)
        self.__dict__.pop("uom_context", None)
        self.__dict__.pop("_trace_to_nhan_dien", None)

    def _prune_removed_volatile_parent_selector(self) -> None:
        """Discard persisted evidence for the removed subtype-parent route.

        Every fixed volatile subtype has its own canonical policy.  Letting
        the broad parent enter that committee breaks the one-context/one-owner
        contract and sends lifecycle credit to a different policy.  Old
        artifacts may retain selector statistics for that route, but they are
        no longer meaningful once it cannot be proposed.
        """
        removed_route = "VOLATILE_PARENT_ENTRY_GENERALIST"
        selector = getattr(self, "policy_selector", None)
        for field_name in ("learned_utility", "outcome_count", "policy_group_loss_streak"):
            values = getattr(selector, field_name, None)
            if isinstance(values, dict):
                setattr(selector, field_name, {
                    key: value for key, value in values.items()
                    if removed_route not in str(key)
                })
        for field_name in (
            "_trace_to_selector_policy", "_trace_to_selector_context",
            "_trace_to_selector_evidence_group",
        ):
            values = getattr(self, field_name, None)
            if isinstance(values, Mapping):
                setattr(self, field_name, {
                    key: value for key, value in values.items()
                    if removed_route not in str(value)
                })

    def _prune_disabled_role_specialists(self) -> None:
        """Remove persisted role policies when the one-owner mode disables them.

        Disabling a role in YAML used to stop *creation* only.  A resumed
        artifact kept its MARKET/LIMIT/STOP/MANAGEMENT policies and could
        still route orders through them, violating the context-owner contract.
        """
        disabled_prefixes: list[str] = []
        if not bool(getattr(self.cau_hinh, "entry_style_specialists_bat", True)):
            disabled_prefixes.extend(ROLE_EXPERT_CONTEXTS[role] for role in ENTRY_STYLE_SPECIALISTS)
        if not bool(getattr(self.cau_hinh, "position_management_specialist_bat", True)):
            disabled_prefixes.append(ROLE_EXPERT_CONTEXTS["MANAGEMENT"])
        if not disabled_prefixes:
            return
        removed_contexts = {
            str(context)
            for context in self.registry.chuyen_gia_theo_context
            if any(str(context).startswith(f"{prefix}{ROLE_CONTEXT_SEPARATOR}") for prefix in disabled_prefixes)
        }
        if not removed_contexts:
            return
        removed_policy_ids = {
            str(self.registry.chuyen_gia_theo_context[context].ma_chuyen_gia)
            for context in removed_contexts
        }
        for context in removed_contexts:
            self.registry.chuyen_gia_theo_context.pop(context, None)
        selector = getattr(self, "policy_selector", None)
        for field_name in ("learned_utility", "outcome_count"):
            values = getattr(selector, field_name, None)
            if isinstance(values, dict):
                setattr(selector, field_name, {
                    key: value for key, value in values.items()
                    if not any(f"{policy_id}::" in str(key) for policy_id in removed_policy_ids)
                })
        for field_name in (
            "_trace_to_selector_policy", "_trace_to_selector_context",
            "_trace_to_selector_evidence_group",
        ):
            values = getattr(self, field_name, None)
            if isinstance(values, Mapping):
                setattr(self, field_name, {
                    key: value for key, value in values.items()
                    if str(value) not in removed_policy_ids and str(value) not in removed_contexts
                })
        writers = getattr(self, "_expert_live_writers", None)
        if isinstance(writers, dict):
            self._expert_live_writers = {
                key: value for key, value in writers.items() if str(key) not in removed_policy_ids
            }

    @staticmethod
    def _role_context_key(role: str, market_context: str) -> str:
        base = ROLE_EXPERT_CONTEXTS.get(str(role), "")
        market_context = str(market_context or "")
        return f"{base}{ROLE_CONTEXT_SEPARATOR}{market_context}" if base and market_context else ""

    def _role_specialist(
        self,
        role: str,
        market_context: str,
        *,
        parent: ChuyenGiaDocLap | None,
    ) -> ChuyenGiaDocLap | None:
        role = str(role)
        if role == "MANAGEMENT":
            if not bool(getattr(self.cau_hinh, "position_management_specialist_bat", True)):
                return None
        elif role not in ENTRY_STYLE_SPECIALISTS or not bool(
            getattr(self.cau_hinh, "entry_style_specialists_bat", True)
        ):
            return None
        role_context = self._role_context_key(role, market_context)
        if not role_context:
            return None
        expert = self.registry.chuyen_gia_theo_context.get(role_context)
        if expert is None:
            # A frozen/inference-only pass may route through existing owners,
            # but it must never create a new specialist.  Registry growth is
            # a learned structural change and would make checkpoint A mutate
            # even when all outcome updates are disabled.
            if not bool(getattr(self, "_learning_enabled", True)):
                return None
            if parent is None:
                return None
            expert = ChuyenGiaDocLap(
                ma_chuyen_gia=f"CG_{role_context}",
                ma_context=role_context,
                ma_context_cha=role_context,
                policy=self._fork_role_policy(role, parent, warm_start=True),
                trang_thai=TrangThaiChuyenGia.HOAT_DONG_HAN_CHE,
                ghi_chu=(
                    "context-role specialist doc lap: "
                    f"role={role}; source_context={market_context}"
                ),
            )
            self.registry.dang_ky(expert)
        return expert if expert.co_quyen_ra_lenh else None

    @property
    def experts(self) -> Mapping[str, ChuyenGiaDocLap]:
        return dict(self.registry.chuyen_gia_theo_context)

    @property
    def _learning_enabled(self) -> bool:
        return bool(getattr(self, "_learning_enabled_state", True))

    @_learning_enabled.setter
    def _learning_enabled(self, value: bool) -> None:
        flag = bool(value)
        self._learning_enabled_state = flag
        registry = getattr(self, "registry", None)
        if registry is not None:
            for expert in getattr(registry, "chuyen_gia_theo_context", {}).values():
                sub_policy = getattr(expert, "policy", None)
                if sub_policy is not None:
                    try:
                        setattr(sub_policy, "_learning_enabled", flag)
                    except Exception:
                        pass
        if hasattr(self, "experts"):
            try:
                for expert in self.experts.values():
                    sub_policy = getattr(expert, "policy", None)
                    if sub_policy is not None:
                        try:
                            setattr(sub_policy, "_learning_enabled", flag)
                        except Exception:
                            pass
            except Exception:
                pass

    @property
    def learning_enabled(self) -> bool:
        return self._learning_enabled

    @learning_enabled.setter
    def learning_enabled(self, value: bool) -> None:
        self._learning_enabled = value

    def _quyet_dinh_khong_giao_dich(self, nhan_dien: NhanDienNguCanh, ly_do: str) -> QuyetDinhKhongGiaoDich:
        self._no_decision_sequence += 1
        output = np.zeros(SO_DAU_RA_BRIDGE, dtype=float)
        output[int(CHI_SO_HOLD)] = 1.0
        return QuyetDinhKhongGiaoDich(
            trace_id=f"khong-policy-{self._no_decision_sequence}",
            selected_action=output,
            selected_idx=int(CHI_SO_HOLD),
            ma_context=nhan_dien.ma_context,
            ly_do_khong_giao_dich=str(ly_do),
            context_audit={
                "ma_context": nhan_dien.ma_context,
                "ma_context_cha": nhan_dien.ma_context_cha,
                "context_on_dinh": bool(nhan_dien.on_dinh),
                "do_tin_cay_context": float(nhan_dien.do_tin_cay),
                "ly_do_context": nhan_dien.ly_do,
                "ly_do_khong_giao_dich": str(ly_do),
                "khong_co_policy_ra_lenh": True,
                "giai_doan_telemetry": "CONTEXT_CHUA_ON_DINH" if not bool(nhan_dien.on_dinh) else "CHUA_CO_CHUYEN_GIA_PHU_HOP",
                "root_run_id": str(self._root_run_id),
            },
        )

    @staticmethod
    def _positions_with_owner(positions_pipe: object) -> list[dict[str, object]]:
        """Decode owner fields without importing the MT5 runner.

        The bridge position contract appends owner context/expert/policy/trace
        after the lifecycle fields. Older snapshots may not have those fields;
        they are intentionally ignored here and continue through the legacy
        context path.
        """
        positions: list[dict[str, object]] = []
        for raw in str(positions_pipe or "").split("^"):
            parts = raw.split(":")
            if len(parts) < 19:
                continue
            try:
                positions.append({
                    "ticket": str(parts[0]),
                    "side": int(float(parts[1])),
                    "unrealized_r": float(parts[6]),
                    "age_bars": float(parts[7]),
                    "mfe_r": float(parts[8]) if len(parts) > 8 else 0.0,
                    "mae_r": float(parts[9]) if len(parts) > 9 else 0.0,
                    "bars_since_mfe": float(parts[14]) if len(parts) > 14 else 0.0,
                    "near_tp": str(parts[15]).lower() in {"1", "true"} if len(parts) > 15 else False,
                    "giveback_r": float(parts[16]) if len(parts) > 16 else 0.0,
                    "owner_context": str(parts[17] or ""),
                    "owner_expert": str(parts[18] or ""),
                    "owner_policy": str(parts[19] or "") if len(parts) > 19 else "",
                    "owner_trace": str(parts[20] or "") if len(parts) > 20 else "",
                })
            except (TypeError, ValueError):
                continue
        return [item for item in positions if item["ticket"]]

    def _management_owner_plan(
        self,
        nhan_dien: NhanDienNguCanh,
    ) -> tuple[ChuyenGiaDocLap, dict[str, object], bool, str] | None:
        """Choose the independent manager and its most adverse target."""
        if self._pending_account_context is None:
            return None
        _account, positions_pipe = self._pending_account_context
        positions = [
            item for item in self._positions_with_owner(positions_pipe)
            if str(item.get("owner_expert", ""))
        ]
        if not positions:
            return None

        target = min(
            positions,
            key=lambda item: (
                float(item.get("unrealized_r", 0.0) or 0.0),
                -float(item.get("age_bars", 0.0) or 0.0),
            ),
        )
        owner_expert = str(target.get("owner_expert", ""))
        original_owner = next(
            (
                item for item in self.registry.chuyen_gia_theo_context.values()
                if item.ma_chuyen_gia == owner_expert
            ),
            None,
        )
        manager = self._role_specialist(
            "MANAGEMENT",
            str(getattr(original_owner, "ma_context", "") or target.get("owner_context", "")),
            parent=original_owner,
        )
        if manager is not None:
            return manager, target, True, "INDEPENDENT_POSITION_MANAGER"
        if original_owner is None or not original_owner.co_quyen_ra_lenh:
            return None
        # Compatibility fallback when the role specialist is explicitly off.
        return original_owner, target, True, "POSITION_OWNER_MANAGEMENT_FALLBACK"

    def _chuan_bi_policy(
        self,
        expert: ChuyenGiaDocLap,
        nhan_dien: NhanDienNguCanh,
        *,
        management_only: bool = False,
        management_owner_expert: str = "",
        management_owner_context: str = "",
        management_target_ticket: str = "",
        management_reason: str = "",
        capability_actions: set[int] | frozenset[int] | None = None,
    ) -> tuple[set[int], dict[str, object]]:
        """Prepare one owner proposal with an immutable capability scope."""
        policy = expert.policy
        # Apply the current facade config to loaded and newly-created owners.
        # This keeps the prior adjustable without sharing policy weights or
        # evidence between context experts.
        try:
            setattr(
                policy,
                "_context_direction_prior_strength",
                float(self.cau_hinh.context_direction_prior_strength),
            )
            setattr(
                policy,
                "_candidate_transition_direction_prior_bat",
                bool(self.cau_hinh.candidate_transition_direction_prior_bat),
            )
            setattr(
                policy,
                "_volatile_timing_direction_prior_bat",
                bool(self.cau_hinh.volatile_timing_direction_prior_bat),
            )
            setattr(
                policy,
                "_stable_context_direction_prior_during_transition_bat",
                bool(self.cau_hinh.stable_context_direction_prior_during_transition_bat),
            )
            setattr(
                policy,
                "_volatile_timing_direction_min_quality",
                float(self.cau_hinh.volatile_timing_direction_min_quality),
            )
            setattr(
                policy,
                "_volatile_max_directional_strength",
                float(self.cau_hinh.volatile_max_directional_strength),
            )
            setattr(policy, "_context_expert_management_only", bool(management_only))
        except Exception:
            pass
        pending = dict(self._pending_decision_context or {})
        if management_only:
            upstream = _normalise_allowed_actions(pending.get("allowed_actions", None))
            allowed = upstream & (set(CHI_SO_MANAGEMENT_ACTIONS) | {CHI_SO_HOLD})
            if not allowed:
                allowed = {CHI_SO_HOLD}
            trend_exit_audit: dict[str, object] = {}
            positions_pipe = pending.get("positions_pipe", "")
            if not positions_pipe and self._pending_account_context is not None:
                positions_pipe = self._pending_account_context[1]
            allowed, management_guard_audit = _owner_management_guard(
                allowed,
                positions_pipe,
                management_owner_expert,
                self.cau_hinh,
            )
        else:
            allowed = _context_allowed_actions(nhan_dien, pending.get("allowed_actions", None))
            allowed, trend_exit_audit = _trend_exit_guard(
                nhan_dien,
                allowed,
                pending.get("positions_pipe", ""),
                self.cau_hinh,
            )
            # Volatile children must learn entry timing while still using the
            # same unrealized-R safety contract as continuity management. A
            # copied or young policy can otherwise emit CLOSE_WORST repeatedly
            # during a non-emergency drawdown. This only changes management
            # actions; all six entry actions remain available to the policy.
            if nhan_dien.nhiem_vu in {
                NhiemVuNguCanh.VOLATILE_SPIKE,
                NhiemVuNguCanh.VOLATILE_EXPANSION,
                NhiemVuNguCanh.VOLATILE_CHOP,
            }:
                positions_pipe = pending.get("positions_pipe", "")
                if not positions_pipe and self._pending_account_context is not None:
                    positions_pipe = self._pending_account_context[1]
                allowed, management_guard_audit = _owner_management_guard(
                    allowed,
                    positions_pipe,
                    management_owner_expert,
                    self.cau_hinh,
                )
            else:
                management_guard_audit = {}
        if capability_actions is not None:
            capability = {int(index) for index in capability_actions} | {CHI_SO_HOLD}
            allowed &= capability
            if not allowed:
                allowed = {CHI_SO_HOLD}
        else:
            capability = set()
        context_audit = {**trend_exit_audit, **management_guard_audit}
        pending["allowed_actions"] = {int(index): True for index in sorted(allowed)}
        # `SelfGrowingPolicy.set_decision_context` has a strict signature.
        # Keep context-contract fields on the facade and attach them to the
        # returned trace; do not leak observability-only keys into the neural
        # policy method.
        self._last_context_action_contract = {
            "context_allowed_actions": sorted(int(index) for index in allowed),
            "context_task": nhan_dien.nhiem_vu.value,
            "context_action_contract": "POSITION_OWNER_MANAGEMENT_PROPOSAL_V12" if management_only else "MOT_CONTEXT_MOT_OWNER_V10",
            "management_only": bool(management_only),
            "management_owner_expert": str(management_owner_expert),
            "management_owner_context": str(management_owner_context),
            "management_target_ticket": str(management_target_ticket),
            "management_reason": str(management_reason),
            "proposal_capability_actions": sorted(capability) if capability else [],
            **context_audit,
        }
        if self._pending_account_context is not None:
            account_context, positions_pipe = self._pending_account_context
            hook = getattr(policy, "set_account_context", None)
            if callable(hook):
                hook(account_context, positions_pipe)
        if self._pending_starvation is not None:
            hook = getattr(policy, "set_experience_starvation", None)
            if callable(hook):
                hook(self._pending_starvation)
        if pending:
            hook = getattr(policy, "set_decision_context", None)
            if callable(hook):
                hook(**pending)
        return allowed, context_audit

    @staticmethod
    def _ap_dung_quan_ly_trend_manh(trace: object, trend_exit_audit: Mapping[str, object]):
        """Make MANAGE_POSITION trailing-only while the strong trend is protected."""
        if not bool(trend_exit_audit.get("trend_exit_guard_active", False)):
            return trace
        outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        if outputs.size < SO_DAU_RA_BRIDGE or int(np.argmax(outputs[:12])) != 11:
            return _gan_action_details(trace, trend_exit_audit)
        managed = outputs.copy()
        managed[16] = max(float(managed[16]), 1.0)   # trailing gate > 0.70
        managed[18] = min(float(managed[18]), -1.2)  # mode < 0.25: no partial close
        trace = _with_selected_action(trace, managed)
        return _gan_action_details(trace, {
            **dict(trend_exit_audit),
            "trend_management_prior_applied": True,
            "trend_management_mode": "trailing_only",
        })

    def configure_risk_caps(self, risk_config: object) -> None:
        self._risk_min_pct = float(getattr(risk_config, "min_risk_pct", self._risk_min_pct) or self._risk_min_pct)
        self._risk_max_pct = float(getattr(risk_config, "max_risk_pct", self._risk_max_pct) or self._risk_max_pct)

    def _apply_per_expert_risk_cap(self, trace: object, expert: ChuyenGiaDocLap):
        if expert.trang_thai != TrangThaiChuyenGia.THAM_DO_RUI_RO_THAP:
            return trace
        outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        if outputs.size <= CHI_SO_RISK_OUTPUT:
            return trace
        cap = min(float(self.cau_hinh.max_risk_pct_tham_do), float(self._risk_max_pct))
        raw_cap = _risk_head_for_pct(cap, min_risk_pct=self._risk_min_pct, max_risk_pct=self._risk_max_pct)
        original = float(outputs[CHI_SO_RISK_OUTPUT])
        if original <= raw_cap:
            return _gan_trace_metadata(trace, {
                "per_expert_risk_cap_active": True,
                "per_expert_max_risk_pct": cap,
                "per_expert_max_open_risk_pct": float(self.cau_hinh.max_open_risk_pct_tham_do),
                "per_expert_risk_clamped": False,
            })
        clamped = outputs.copy()
        clamped[CHI_SO_RISK_OUTPUT] = raw_cap
        trace = _with_selected_action(trace, clamped)
        return _gan_trace_metadata(trace, {
            "per_expert_risk_cap_active": True,
            "per_expert_max_risk_pct": cap,
            "per_expert_max_open_risk_pct": float(self.cau_hinh.max_open_risk_pct_tham_do),
            "per_expert_risk_clamped": True,
            "per_expert_risk_raw_before": original,
            "per_expert_risk_raw_after": raw_cap,
        })

    def _apply_timing_deceleration_response(
        self,
        trace: object,
        nhan_dien: NhanDienNguCanh,
    ) -> object:
        """Make a high-probability deceleration entry wait for confirmation.

        The guard is deliberately narrow: it operates only in an unconfirmed
        context transition, preserves the selected direction, and never turns
        an entry into HOLD.  A late LIMIT becomes the same-side STOP, which
        waits for continuation; every other entry style keeps its type but has
        its risk head capped continuously toward the configured floor.
        """

        outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        action_before = self._proposal_action_index(trace)
        probability = float(np.clip(
            float(getattr(nhan_dien, "timing_network_deceleration_probability", 0.0) or 0.0),
            0.0,
            1.0,
        ))
        update_count = max(0, int(getattr(nhan_dien, "timing_network_deceleration_update_count", 0) or 0))
        threshold = float(self.cau_hinh.timing_deceleration_entry_threshold)
        min_updates = int(self.cau_hinh.timing_deceleration_min_updates)
        enabled = bool(self.cau_hinh.timing_deceleration_response_bat)
        transition = bool(nhan_dien.transition_active)
        qualified = bool(
            enabled
            and transition
            and action_before in CHI_SO_ENTRY_ACTIONS
            and update_count >= min_updates
            and probability >= threshold
        )
        audit: dict[str, object] = {
            "timing_deceleration_response_enabled": enabled,
            "timing_deceleration_transition_active": transition,
            "timing_deceleration_probability": probability,
            "timing_deceleration_threshold": threshold,
            "timing_deceleration_update_count": update_count,
            "timing_deceleration_min_updates": min_updates,
            "timing_deceleration_qualified": qualified,
            "timing_deceleration_action_before": int(action_before),
            "timing_deceleration_action_after": int(action_before),
            "timing_deceleration_entry_type_changed": False,
            "timing_deceleration_risk_scaled": False,
        }
        if not qualified or outputs.size <= CHI_SO_RISK_OUTPUT:
            return _gan_action_details(_gan_trace_metadata(trace, audit), audit)

        adjusted = outputs.copy()
        action_after = action_before
        limit_to_stop = {3: 5, 4: 6}
        if bool(self.cau_hinh.timing_deceleration_limit_to_stop_bat) and action_before in limit_to_stop:
            action_after = limit_to_stop[action_before]
            action_count = min(12, adjusted.size)
            if action_after < action_count:
                adjusted[action_after] = float(np.max(adjusted[:action_count])) + 1e-6
                trace = _with_selected_action(trace, adjusted)
                audit.update({
                    "timing_deceleration_action_after": int(action_after),
                    "timing_deceleration_entry_type_changed": True,
                    "selected_candidate_action_index": int(action_after),
                    "selected_hypothesis_action_index": int(action_after),
                    "policy_output_action_index": int(action_after),
                    "semantic_action": int(action_after),
                    "semantic_action_source": "timing_deceleration_limit_to_stop",
                    "action_name": {5: "BUY_STOP", 6: "SELL_STOP"}.get(action_after, ""),
                })

        adjusted = np.asarray(getattr(trace, "selected_action", adjusted), dtype=float).reshape(-1).copy()
        original_raw = float(adjusted[CHI_SO_RISK_OUTPUT])
        original_pct = _risk_pct_for_head(
            original_raw,
            min_risk_pct=self._risk_min_pct,
            max_risk_pct=self._risk_max_pct,
        )
        severity = float(np.clip((probability - threshold) / max(1e-9, 1.0 - threshold), 0.0, 1.0))
        risk_multiplier = float(1.0 - severity * (1.0 - float(self.cau_hinh.timing_deceleration_min_risk_multiplier)))
        target_pct = max(float(self._risk_min_pct), original_pct * risk_multiplier)
        target_raw = _risk_head_for_pct(
            target_pct,
            min_risk_pct=self._risk_min_pct,
            max_risk_pct=self._risk_max_pct,
        )
        if target_raw < original_raw:
            adjusted[CHI_SO_RISK_OUTPUT] = target_raw
            trace = _with_selected_action(trace, adjusted)
            risk_scaled = True
        else:
            risk_scaled = False
        audit.update({
            "timing_deceleration_severity": severity,
            "timing_deceleration_risk_multiplier": risk_multiplier,
            "timing_deceleration_original_risk_pct": original_pct,
            "timing_deceleration_target_risk_pct": target_pct,
            "timing_deceleration_risk_scaled": risk_scaled,
        })
        return _gan_action_details(_gan_trace_metadata(trace, audit), audit)

    def _apply_timing_regime_transition_entry_block(
        self,
        trace: object,
        nhan_dien: NhanDienNguCanh,
        *,
        management_only: bool = False,
    ) -> object:
        """Suppress an entry only after a calibrated imminent-transition forecast.

        The shared detector TimingNet predicts whether the *confirmed* context
        will change at its nearest horizon.  Unlike the retired local
        LIMIT-to-STOP overlay, this is a single, auditable circuit after the
        selector.  It never interferes with position management and remains
        inactive until delayed-label evidence proves high-probability calls
        have adequate precision.
        """

        outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        before = self._proposal_action_index(trace)
        threshold = float(getattr(self.cau_hinh, "timing_regime_transition_entry_threshold", 0.78) or 0.78)
        probability = float(np.clip(
            getattr(nhan_dien, "timing_network_regime_transition_probability", 0.0) or 0.0,
            0.0,
            1.0,
        ))
        updates = max(0, int(getattr(nhan_dien, "timing_network_regime_transition_update_count", 0) or 0))
        calibration_count = max(0, int(getattr(nhan_dien, "timing_network_regime_transition_calibration_count", 0) or 0))
        brier = float(np.clip(
            getattr(nhan_dien, "timing_network_regime_transition_brier", 1.0) or 1.0,
            0.0,
            1.0,
        ))
        high_support = max(0, int(getattr(nhan_dien, "timing_network_regime_transition_high_confidence_support", 0) or 0))
        precision = float(np.clip(
            getattr(nhan_dien, "timing_network_regime_transition_high_confidence_precision", 0.0) or 0.0,
            0.0,
            1.0,
        ))
        min_updates = int(getattr(self.cau_hinh, "timing_regime_transition_min_updates", 192) or 192)
        min_calibration = int(getattr(self.cau_hinh, "timing_regime_transition_min_calibration_samples", 128) or 128)
        min_high_support = int(getattr(self.cau_hinh, "timing_regime_transition_min_high_confidence_samples", 12) or 12)
        min_precision = float(getattr(self.cau_hinh, "timing_regime_transition_min_precision", 0.75) or 0.75)
        max_brier = float(getattr(self.cau_hinh, "timing_regime_transition_max_brier", 0.20) or 0.20)
        enabled = bool(getattr(self.cau_hinh, "timing_regime_transition_entry_block_bat", False))
        calibration_ready = bool(
            updates >= min_updates
            and calibration_count >= min_calibration
            and high_support >= min_high_support
            and precision >= min_precision
            and brier <= max_brier
        )
        qualified = bool(
            enabled and not management_only and before in CHI_SO_ENTRY_ACTIONS
            and calibration_ready and probability >= threshold
        )
        audit: dict[str, object] = {
            "timing_regime_transition_entry_block_enabled": enabled,
            "timing_regime_transition_probability": probability,
            "timing_regime_transition_threshold": threshold,
            "timing_regime_transition_update_count": updates,
            "timing_regime_transition_min_updates": min_updates,
            "timing_regime_transition_calibration_count": calibration_count,
            "timing_regime_transition_min_calibration_samples": min_calibration,
            "timing_regime_transition_brier": brier,
            "timing_regime_transition_max_brier": max_brier,
            "timing_regime_transition_high_confidence_support": high_support,
            "timing_regime_transition_min_high_confidence_samples": min_high_support,
            "timing_regime_transition_high_confidence_precision": precision,
            "timing_regime_transition_min_precision": min_precision,
            "timing_regime_transition_calibration_ready": calibration_ready,
            "timing_regime_transition_entry_blocked": False,
            "timing_regime_transition_action_before": int(before),
            "timing_regime_transition_action_after": int(before),
        }
        if not qualified or outputs.size == 0:
            return _gan_action_details(_gan_trace_metadata(trace, audit), audit)

        adjusted = outputs.copy()
        action_count = min(12, adjusted.size)
        adjusted[0] = float(np.max(adjusted[:action_count])) + 1e-6
        trace = _with_selected_action(trace, adjusted)
        audit.update({
            "timing_regime_transition_entry_blocked": True,
            "timing_regime_transition_action_after": CHI_SO_HOLD,
            "selected_candidate_action_index": CHI_SO_HOLD,
            "selected_hypothesis_action_index": CHI_SO_HOLD,
            "policy_output_action_index": CHI_SO_HOLD,
            "semantic_action": CHI_SO_HOLD,
            "semantic_action_source": "timing_regime_transition_calibrated_entry_block",
            "action_name": "HOLD",
        })
        return _gan_action_details(_gan_trace_metadata(trace, audit), audit)

    @staticmethod
    def _proposal_action_index(trace: object) -> int:
        output = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        return int(np.argmax(output[:12])) if output.size else CHI_SO_HOLD

    @staticmethod
    def _proposal_margin(trace: object) -> float:
        output = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)[:12]
        if output.size < 2:
            return 0.0
        top = np.partition(output, -2)[-2:]
        return float(np.clip(float(top[-1] - top[-2]) / 2.0, 0.0, 1.0))

    def _build_expert_proposal(
        self,
        expert: ChuyenGiaDocLap,
        observation: np.ndarray,
        goal: np.ndarray | None,
        nhan_dien: NhanDienNguCanh,
        *,
        role: str,
        management_only: bool,
        management_owner_expert: str = "",
        management_owner_context: str = "",
        management_target_ticket: str = "",
        management_target_owner_trace: str = "",
        management_reason: str = "",
        capability_actions: set[int] | frozenset[int] | None = None,
    ) -> dict[str, object]:
        allowed, guard_audit = self._chuan_bi_policy(
            expert,
            nhan_dien,
            management_only=management_only,
            management_owner_expert=management_owner_expert,
            management_owner_context=management_owner_context,
            management_target_ticket=management_target_ticket,
            management_reason=management_reason,
            capability_actions=capability_actions,
        )
        trace = expert.ra_quyet_dinh(np.asarray(observation, dtype=float), goal, nhan_dien=nhan_dien)
        trace = self._ap_dung_quan_ly_trend_manh(trace, guard_audit)
        trace = self._apply_per_expert_risk_cap(trace, expert)
        trace = self._apply_timing_deceleration_response(trace, nhan_dien)
        # The policy keeps an entry-time snapshot for delayed MT5 lifecycle
        # credit. A facade execution adaptation must replace that snapshot too;
        # otherwise the bridge would execute STOP/risk-capped output while the
        # local TimingNet learns from the stale LIMIT/raw-risk action.
        trace_id = str(getattr(trace, "trace_id", "") or "")
        cache = getattr(expert.policy, "_execution_trace_cache", None)
        if trace_id and isinstance(cache, dict):
            cache[trace_id] = trace
        trace = self._namespace_proposal_trace(expert, trace)
        return {
            "proposal_id": f"{expert.ma_chuyen_gia}::{role}",
            "expert": expert,
            "trace": trace,
            "role": str(role),
            "allowed": set(allowed),
            "guard_audit": dict(guard_audit),
            "management_only": bool(management_only),
            "management_owner_expert": str(management_owner_expert),
            "management_owner_context": str(management_owner_context),
            "management_target_ticket": str(management_target_ticket),
            "management_target_owner_trace": str(management_target_owner_trace),
            "management_reason": str(management_reason),
            "selector_context": str(nhan_dien.ma_context),
            "selector_microstate": self._selector_microstate_key(nhan_dien),
            "candidate_context": str(nhan_dien.candidate_context or nhan_dien.ma_context),
            "transition_active": bool(nhan_dien.transition_active),
            "transition_progress": float(nhan_dien.transition_progress),
        }

    @staticmethod
    def _entry_direction(action_index: int) -> int:
        if int(action_index) in {1, 3, 5}:
            return 1
        if int(action_index) in {2, 4, 6}:
            return -1
        return 0

    @staticmethod
    def _candidate_context_family(context_key: str) -> str:
        text = str(context_key or "").upper()
        if "TANG" in text:
            return "UP"
        if "GIAM" in text:
            return "DOWN"
        if "VOLATILE" in text:
            return "VOLATILE"
        if "RANG" in text:
            return "RANGING"
        return "UNKNOWN"

    @classmethod
    def _selector_microstate_key(cls, nhan_dien: NhanDienNguCanh) -> str:
        quality = float(nhan_dien.timing_quality)
        quality_bucket = "Q0" if quality < 0.10 else ("Q1" if quality < 0.50 else "Q2")
        candidate = str(nhan_dien.candidate_context or nhan_dien.ma_context)
        phase = (
            f"TRANSITION_{cls._candidate_context_family(candidate)}"
            if bool(nhan_dien.transition_active)
            else "STABLE"
        )
        return f"{phase}|TIMING_{int(nhan_dien.timing_direction)}_{quality_bucket}"

    @staticmethod
    def _pending_entry_parts(value: object) -> tuple[str, str, int]:
        if isinstance(value, Mapping):
            return (
                str(value.get("context", "") or ""),
                str(value.get("proposal_id", "") or ""),
                int(value.get("direction", 0) or 0),
            )
        if isinstance(value, (tuple, list)):
            return (
                str(value[0] or "") if len(value) > 0 else "",
                str(value[1] or "") if len(value) > 1 else "",
                int(value[2] or 0) if len(value) > 2 else 0,
            )
        return "", "", 0

    def _pending_entries_for_exposure_supervisor(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for trace_id, value in self._selector_pending_entry_traces.items():
            context, proposal_id, direction = self._pending_entry_parts(value)
            rows.append({
                "trace_id": str(trace_id),
                "context": context,
                "proposal_id": proposal_id,
                "direction": direction,
            })
        return rows

    def _prune_selector_pending_entries(self) -> None:
        """Remove filled/stale pending traces from selector exposure evidence.

        A selected pending entry is not a lifecycle until it fills and closes.
        If the broker cancels it without a lifecycle receipt, retaining its
        trace forever makes every later proposal look increasingly correlated.
        Position owner traces prove a fill; the bounded age handles cancel/
        expiry paths whose order receipt carries no trace id.
        """
        if not self._selector_pending_entry_traces:
            return
        positions_pipe = self._pending_account_context[1] if self._pending_account_context else ""
        active_traces = {
            str(item.get("owner_trace", "") or "")
            for item in self._positions_with_owner(positions_pipe)
            if str(item.get("owner_trace", "") or "")
        }
        max_age = max(
            0,
            int(getattr(self.cau_hinh, "selector_pending_entry_max_age_decisions", 0) or 0),
        )
        for trace_id in list(self._selector_pending_entry_traces):
            trace_id = str(trace_id)
            if trace_id in active_traces:
                self._selector_pending_entry_traces.pop(trace_id, None)
                self._selector_pending_entry_ages.pop(trace_id, None)
                continue
            age = int(self._selector_pending_entry_ages.get(trace_id, 0) or 0) + 1
            self._selector_pending_entry_ages[trace_id] = age
            if max_age > 0 and age >= max_age:
                self._selector_pending_entry_traces.pop(trace_id, None)
                self._selector_pending_entry_ages.pop(trace_id, None)

    def _correlated_exposure_state(self, proposal: Mapping[str, object]) -> CorrelatedExposureState:
        action_index = self._proposal_action_index(proposal["trace"])
        positions = self._positions_with_owner(
            self._pending_account_context[1] if self._pending_account_context is not None else ""
        )
        return self.exposure_supervisor.assess(
            market_context=str(proposal.get("selector_context", "") or ""),
            direction=self._entry_direction(action_index),
            positions=positions,
            pending_entries=self._pending_entries_for_exposure_supervisor(),
        )

    def _annotate_correlated_exposure(
        self,
        trace: object,
        state: CorrelatedExposureState,
    ) -> object:
        outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        action_index = self._proposal_action_index(trace)
        if outputs.size <= CHI_SO_RISK_OUTPUT or action_index not in CHI_SO_ENTRY_ACTIONS:
            return trace
        original_raw = float(outputs[CHI_SO_RISK_OUTPUT])
        original_pct = _risk_pct_for_head(
            original_raw,
            min_risk_pct=self._risk_min_pct,
            max_risk_pct=self._risk_max_pct,
        )
        # Exposure changes selector evidence only. Preserve the exact risk
        # head emitted by the selected policy and its resulting volume.
        target_pct = original_pct
        scaled = False
        return _gan_trace_metadata(trace, {
            "correlated_exposure_supervisor_active": True,
            "correlated_exposure_stage": state.stage,
            "correlated_exposure_direction": int(state.direction),
            "correlated_exposure_count": int(state.exposure_count),
            "correlated_exposure_open_positions": int(state.open_position_count),
            "correlated_exposure_pending_traces": int(state.pending_trace_count),
            "correlated_exposure_mean_unrealized_r": float(state.mean_unrealized_r),
            "correlated_exposure_utility_context": str(state.utility_context),
            "correlated_exposure_risk_decay": float(self.exposure_supervisor.risk_decay),
            "correlated_exposure_original_risk_pct": float(original_pct),
            "correlated_exposure_target_risk_pct": float(target_pct),
            "correlated_exposure_risk_scaled": bool(scaled),
            "correlated_exposure_entry_blocked": False,
        })

    def _annotate_role_readiness(
        self,
        trace: object,
        proposal: Mapping[str, object],
    ) -> object:
        """Annotate cold-specialist readiness without changing entry volume."""
        outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        action_index = self._proposal_action_index(trace)
        role = str(proposal.get("role", ""))
        evidence_group = str(proposal.get("selector_evidence_group", ""))
        is_role_entry = role.startswith("ENTRY_STYLE_") and action_index in CHI_SO_ENTRY_ACTIONS
        if outputs.size <= CHI_SO_RISK_OUTPUT or not is_role_entry or not evidence_group:
            return _gan_trace_metadata(trace, {
                "role_readiness_supervisor_active": False,
                "role_readiness_entry_blocked": False,
            })
        support, learned_utility, win_signal, loss_streak = self.policy_selector.group_evidence(
            evidence_group
        )
        state = self.role_readiness_supervisor.assess(
            evidence_group=evidence_group,
            support=support,
            learned_utility=learned_utility,
        )
        original_raw = float(outputs[CHI_SO_RISK_OUTPUT])
        original_pct = _risk_pct_for_head(
            original_raw,
            min_risk_pct=self._risk_min_pct,
            max_risk_pct=self._risk_max_pct,
        )
        # Readiness is audit evidence only. A cold role keeps the policy's
        # original risk head and is therefore not volume-limited.
        target_pct = original_pct
        scaled = False
        return _gan_trace_metadata(trace, {
            "role_readiness_supervisor_active": True,
            "role_readiness_evidence_group": evidence_group,
            "role_readiness_support": int(support),
            "role_readiness_learned_utility": float(learned_utility),
            "role_readiness_win_signal": float(win_signal),
            "role_readiness_loss_streak": int(loss_streak),
            "role_readiness_score": float(state.readiness),
            "role_readiness_risk_multiplier": float(state.risk_multiplier),
            "role_readiness_original_risk_pct": float(original_pct),
            "role_readiness_target_risk_pct": float(target_pct),
            "role_readiness_risk_scaled": bool(scaled),
            "role_readiness_entry_blocked": False,
        })

    def _selector_pending_outcomes(self, proposal_id: str, context_key: str) -> int:
        proposal_id = str(proposal_id or "")
        context_key = str(context_key or "")
        return sum(
            1
            for pending_context, pending_proposal, _pending_direction in (
                self._pending_entry_parts(value)
                for value in self._selector_pending_entry_traces.values()
            )
            if str(pending_context) == context_key and str(pending_proposal) == proposal_id
        )

    @staticmethod
    def _selector_evidence_group(role: str, direction: int) -> str:
        direction = 1 if int(direction) > 0 else (-1 if int(direction) < 0 else 0)
        return f"{str(role or 'UNKNOWN')}|SIDE_{direction}"

    def _selector_candidate(self, proposal: Mapping[str, object]) -> PolicyCandidateScore:
        expert = proposal["expert"]
        trace = proposal["trace"]
        action_index = self._proposal_action_index(trace)
        role = str(proposal.get("role", ""))
        proposal_id = str(proposal.get("proposal_id", expert.ma_chuyen_gia))
        selector_context = str(proposal.get("selector_context", ""))
        exposure_state = self._correlated_exposure_state(proposal)
        microstate = str(proposal.get("selector_microstate", "STABLE|TIMING_0_Q0") or "STABLE|TIMING_0_Q0")
        utility_context = f"{exposure_state.utility_context}|{microstate}|ACTION_SIDE_{exposure_state.direction}"
        evidence_group = self._selector_evidence_group(role, exposure_state.direction)
        if isinstance(proposal, dict):
            proposal["correlated_exposure_state"] = exposure_state
            proposal["selector_utility_context"] = utility_context
            proposal["selector_evidence_group"] = evidence_group
        allowed = {int(index) for index in proposal.get("allowed", set())}
        if "ENTRY" in role:
            role_fit = 1.0 if action_index in CHI_SO_ENTRY_ACTIONS else (0.20 if action_index == CHI_SO_HOLD else -1.0)
            valid = (action_index in CHI_SO_ENTRY_ACTIONS or action_index == CHI_SO_HOLD) and action_index in allowed
        else:
            role_fit = 1.0 if action_index in CHI_SO_MANAGEMENT_ACTIONS else (0.40 if action_index == CHI_SO_HOLD else -1.0)
            valid = (action_index in CHI_SO_MANAGEMENT_ACTIONS or action_index == CHI_SO_HOLD) and action_index in allowed
        account = self._pending_account_context[0] if self._pending_account_context is not None else {}
        drawdown = float(account.get("drawdown_pct", account.get("drawdown", 0.0)) or 0.0)
        if drawdown > 1.0:
            drawdown *= 0.01
        stress = max(
            float(account.get("account_stress", 0.0) or 0.0),
            float(account.get("open_risk_ratio", 0.0) or 0.0),
            float(np.clip(drawdown, 0.0, 1.0)),
        )
        positions = self._positions_with_owner(self._pending_account_context[1]) if self._pending_account_context else []
        confidence = max(
            float(getattr(trace, "confidence", 0.0) or 0.0),
            self._proposal_margin(trace),
        )
        evidence_support, learned_utility, _win_signal, _loss_streak = self.policy_selector.group_evidence(
            evidence_group
        )
        lifecycle_support = max(0, int(getattr(expert, "so_lifecycle_da_dong", 0) or 0))
        support = max(int(evidence_support), lifecycle_support)
        readiness_state = self.role_readiness_supervisor.assess(
            evidence_group=evidence_group,
            support=support,
            learned_utility=learned_utility,
        )
        cold_start = bool(
            role.startswith("ENTRY_STYLE_")
            and support < int(getattr(self.cau_hinh, "selector_cold_start_support", 8))
        )
        if isinstance(proposal, dict):
            proposal["selector_evidence_support"] = int(evidence_support)
            proposal["selector_lifecycle_support"] = int(lifecycle_support)
            proposal["selector_support"] = int(support)
            proposal["selector_readiness"] = float(readiness_state.readiness)
            proposal["selector_cold_start"] = bool(cold_start)
        return PolicyCandidateScore(
            policy_id=proposal_id,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            recent_utility=float(np.tanh(float(expert.fitness_cuc_bo))),
            context_key=utility_context,
            evidence_group=evidence_group,
            memory_match=float(np.clip(float(getattr(trace, "memory_match_quality", 0.0) or 0.0), 0.0, 1.0)),
            proposal_margin=self._proposal_margin(trace),
            role_fit=role_fit,
            account_stress=float(np.clip(stress, 0.0, 1.0)),
            drawdown=float(np.clip(drawdown, 0.0, 1.0)),
            open_positions=len(positions),
            # Inflight evidence must remain proposal-local. Correlated exposure
            # is already represented by its stage/evidence; copying its count
            # to every candidate subtracts the same penalty from the
            # whole committee and cannot rotate away from the unresolved policy.
            pending_outcomes=self._selector_pending_outcomes(proposal_id, selector_context),
            correlated_exposure=int(exposure_state.exposure_count),
            valid=bool(valid),
            reason=f"{role}:action={action_index}",
            evidence_support=int(evidence_support),
            lifecycle_support=int(lifecycle_support),
            readiness=float(readiness_state.readiness),
            cold_start=cold_start,
        )

    @staticmethod
    def _namespace_proposal_trace(expert: ChuyenGiaDocLap, trace: object) -> object:
        """Make trace ownership globally unique across cloned role policies."""
        original_trace_id = str(getattr(trace, "trace_id", "") or "")
        prefix = f"{expert.ma_chuyen_gia}~"
        if not original_trace_id or original_trace_id.startswith(prefix):
            return trace
        namespaced_trace_id = f"{prefix}{original_trace_id}"
        namespaced_trace = trace
        if is_dataclass(trace):
            try:
                namespaced_trace = replace(trace, trace_id=namespaced_trace_id)
            except (TypeError, ValueError):
                pass
        if namespaced_trace is trace:
            try:
                setattr(namespaced_trace, "trace_id", namespaced_trace_id)
            except Exception:
                return trace
        policy = getattr(expert, "policy", None)
        for cache_name in (
            "_execution_trace_cache",
            "_pending_execution_traces",
            "_causal_trace_archive",
            "_causal_reports",
            "_causal_report_signatures",
        ):
            cache = getattr(policy, cache_name, None)
            if not isinstance(cache, dict) or original_trace_id not in cache:
                continue
            cached_trace = cache.pop(original_trace_id)
            if is_dataclass(cached_trace):
                try:
                    cached_trace = replace(cached_trace, trace_id=namespaced_trace_id)
                except (TypeError, ValueError):
                    pass
            try:
                setattr(cached_trace, "trace_id", namespaced_trace_id)
            except Exception:
                pass
            cache[namespaced_trace_id] = cached_trace
        return namespaced_trace

    @staticmethod
    def _discard_unselected_proposal(proposal: Mapping[str, object]) -> None:
        expert = proposal.get("expert")
        trace = proposal.get("trace")
        trace_id = str(getattr(trace, "trace_id", "") or "")
        policy = getattr(expert, "policy", None)
        if not trace_id or policy is None:
            return
        # A shadow proposal was never sent to MT5 and must not remain eligible
        # for receipt/lifecycle credit.
        for cache_name in ("_execution_trace_cache", "_pending_execution_traces"):
            cache = getattr(policy, cache_name, None)
            if isinstance(cache, dict):
                cache.pop(trace_id, None)

    def _select_proposal(
        self,
        proposals: list[dict[str, object]],
        *,
        fallback_policy_id: str,
    ) -> tuple[dict[str, object], PolicySelectionTrace]:
        self._prune_selector_pending_entries()
        candidates = [self._selector_candidate(item) for item in proposals]
        entry_candidates = [
            candidate for item, candidate in zip(proposals, candidates)
            if "ENTRY" in str(item.get("role", ""))
            and candidate.valid
            and self._proposal_action_index(item["trace"]) in CHI_SO_ENTRY_ACTIONS
        ]
        management_candidates = [
            candidate for item, candidate in zip(proposals, candidates)
            if "MANAGEMENT" in str(item.get("role", ""))
            and candidate.valid
            and self._proposal_action_index(item["trace"]) in CHI_SO_MANAGEMENT_ACTIONS
        ]
        cold_probe_budget = max(
            0,
            int(getattr(self.cau_hinh, "selector_cold_start_max_per_run", 0) or 0),
        )
        cold_probe_exhausted = bool(
            cold_probe_budget > 0
            and int(getattr(self, "_selector_cold_start_probe_count", 0) or 0) >= cold_probe_budget
        )
        if cold_probe_exhausted:
            # Never invalidate a cold role globally: it remains in the
            # proposal audit and can become eligible after evidence arrives.
            # Once this run has consumed its probes, remove only those cold
            # entry proposals from the ranking and fall back to a supported
            # route or HOLD.  This prevents a HOLDing v13 parent from turning
            # every bar into a fresh specialist experiment.
            entry_candidates = [
                candidate for candidate in entry_candidates
                if not bool(getattr(candidate, "cold_start", False))
            ]
        safe_non_cold_candidates = [
            candidate for candidate in candidates
            if candidate.valid and not bool(getattr(candidate, "cold_start", False))
        ]
        self.policy_selector.fallback_policy_id = str(fallback_policy_id)
        ranking_candidates = entry_candidates or management_candidates or (
            safe_non_cold_candidates if cold_probe_exhausted and safe_non_cold_candidates else candidates
        )
        ranked = self.policy_selector.select(
            ranking_candidates,
            require_valid_selection=bool(entry_candidates),
        )
        all_scores = {candidate.policy_id: float(self.policy_selector._score(candidate)) for candidate in candidates}
        if entry_candidates:
            selection = PolicySelectionTrace(
                selected_policy_id=str(ranked.selected_policy_id),
                fallback_policy_id=str(fallback_policy_id),
                scores=all_scores,
                reason="entry_committee_selected_no_block",
                used_fallback=False,
            )
            self.policy_selector.last_trace = selection
        elif management_candidates:
            selection = PolicySelectionTrace(
                selected_policy_id=str(ranked.selected_policy_id),
                fallback_policy_id=str(fallback_policy_id),
                scores=all_scores,
                reason="position_management_selected",
                used_fallback=False,
            )
            self.policy_selector.last_trace = selection
        elif cold_probe_exhausted and safe_non_cold_candidates:
            selection = PolicySelectionTrace(
                selected_policy_id=str(ranked.selected_policy_id),
                fallback_policy_id=str(fallback_policy_id),
                scores=all_scores,
                reason="cold_start_probe_budget_exhausted",
                used_fallback=False,
            )
            self.policy_selector.last_trace = selection
        else:
            selection = PolicySelectionTrace(
                selected_policy_id=str(ranked.selected_policy_id),
                fallback_policy_id=str(fallback_policy_id),
                scores=all_scores,
                reason=str(ranked.reason),
                used_fallback=bool(ranked.used_fallback),
            )
            self.policy_selector.last_trace = selection
        selected = next(
            (item for item in proposals if str(item.get("proposal_id", "")) == selection.selected_policy_id),
            proposals[0],
        )
        for item in proposals:
            if item is not selected:
                self._discard_unselected_proposal(item)
        selected_candidate = next(
            (candidate for candidate in candidates if candidate.policy_id == selection.selected_policy_id),
            None,
        )
        if (
            selected_candidate is not None
            and bool(getattr(selected_candidate, "cold_start", False))
            and self._proposal_action_index(selected["trace"]) in CHI_SO_ENTRY_ACTIONS
        ):
            self._selector_cold_start_probe_count = int(
                getattr(self, "_selector_cold_start_probe_count", 0) or 0
            ) + 1
        return selected, selection

    def _chay_nam_tac_nhan(self, observation: np.ndarray, goal: np.ndarray | None = None) -> tuple[object, ...]:
        self._pending_regime_raw_inputs = None
        active_expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if active_expert is None or not active_expert.co_quyen_ra_lenh:
            raise RuntimeError("khong tim thay bo nao chung co quyen ra lenh")
        policy = active_expert.policy
        if self._pending_account_context is not None:
            account_context, positions_pipe = self._pending_account_context
            hook = getattr(policy, "set_account_context", None)
            if callable(hook):
                hook(account_context, positions_pipe)
        pending = dict(self._pending_decision_context or {})
        if pending:
            hook = getattr(policy, "set_decision_context", None)
            if callable(hook):
                hook(**pending)
        multi = getattr(policy, "act_five_agents", None)
        if not callable(multi):
            raise RuntimeError("policy khong ho tro nam tac nhan dung chung")
        raw_traces = tuple(multi(np.asarray(observation, dtype=float).reshape(-1), goal))
        if len(raw_traces) != 5:
            raise RuntimeError("nam tac nhan phai tra ve dung 5 quyet dinh")
        traces: list[object] = []
        for agent_index, raw_trace in enumerate(raw_traces):
            authority = five_agent_authority(agent_index)
            meta = {
                "ma_context": "CTX_CHUNG",
                "ma_context_cha": "CTX_CHUNG",
                "ma_chuyen_gia": "BO_NAO_CHUNG",
                "ma_policy": "BO_NAO_CHUNG",
                "decision_policy_id": "BO_NAO_CHUNG",
                "decision_policy_context": "CTX_CHUNG",
                "decision_role": authority.role,
                "five_agent_index": int(agent_index),
                "five_agent_count": 5,
                "five_agent_shared_parameters": True,
                "five_agent_authority_version": FIVE_AGENT_AUTHORITY_VERSION,
                "five_agent_authority_role": authority.role,
                "five_agent_authority_actions": sorted(authority.allowed_actions),
                "five_agent_authority_responsibility": authority.responsibility,
                "context_action_contract": "NAM_TAC_NHAN_PHAN_QUYEN_V1",
                "selector_reason": "khong_dung_bo_chon",
                "selector_used_fallback": False,
                "context_history_inputs_enabled": False,
                "context_history_input_count": 0,
                "entry_feature_interaction_count": 0,
            }
            pending_probs = getattr(self, "_pending_regime_probabilities", None)
            if pending_probs is not None:
                meta.update({
                    "regime_probabilities": list(pending_probs),
                    "regime_distribution": list(pending_probs),
                    "regime_name": str(getattr(self, "_pending_regime_name", "") or ""),
                    "regime_index": getattr(self, "_pending_regime_index", None),
                    "p_bull": float(pending_probs[0]),
                    "p_bear": float(pending_probs[1]),
                    "p_chop": float(pending_probs[2]),
                })
            trace = _gan_trace_metadata(raw_trace, meta)
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if trace_id:
                self._trace_to_expert[trace_id] = "BO_NAO_CHUNG"
            traces.append(trace)
        self._last_five_agent_traces = tuple(traces)
        self._last_expert_id = "BO_NAO_CHUNG"
        return tuple(traces)

    def act_nam_tac_nhan(self, observation: np.ndarray, goal: np.ndarray | None = None) -> tuple[object, ...]:
        """Tra ve nam quyet dinh rieng, cung trong so nhung khac trang thai nho."""
        self._ensure_context_history_policy_inputs()
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) != "nam_tac_nhan_dung_chung":
            raise RuntimeError("he thong chua o che do nam tac nhan dung chung")
        return self._chay_nam_tac_nhan(observation, goal)

    def nap_lich_su_nam_tac_nhan(
        self,
        observations: np.ndarray,
        *,
        actions: np.ndarray | None = None,
        dones: np.ndarray | None = None,
        reset_first: bool = True,
    ) -> None:
        """Nap mot chuoi quan sat qua khu theo dung thu tu de dung lai ky uc."""
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) != "nam_tac_nhan_dung_chung":
            raise RuntimeError("he thong chua o che do nam tac nhan dung chung")
        active_expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if active_expert is None:
            raise RuntimeError("khong tim thay bo nao chung")
        hook = getattr(active_expert.policy, "prime_five_history", None)
        if not callable(hook):
            raise RuntimeError("bo nao chung khong ho tro nap lich su")
        hook(observations, actions=actions, dones=dones, reset_first=reset_first)

    def install_five_rlhf_reference(
        self,
        reference_policy_sha256: str | None = None,
    ) -> dict[str, object]:
        """Install a candidate-only frozen reference on the active owner(s)."""
        mode = str(getattr(self.cau_hinh, "che_do_quyet_dinh", "") or "")
        if mode == "nam_tac_nhan_dung_chung":
            active_expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
            if active_expert is None or not bool(active_expert.co_quyen_ra_lenh):
                return {
                    "installed": False,
                    "reason": "public_five_owner_missing",
                    "promotion_allowed": False,
                }
            hook = getattr(active_expert.policy, "install_five_rlhf_reference", None)
            if not callable(hook):
                return {
                    "installed": False,
                    "reason": "policy_reference_hook_missing",
                    "promotion_allowed": False,
                }
            # Public Five owns its own frozen torch-network hash; the source
            # checkpoint hash remains a separate provenance binding.
            result = hook()
            return dict(result) if isinstance(result, Mapping) else {
                "installed": bool(result),
                "promotion_allowed": False,
            }

        # Legacy context/shared-brain modes own one Direct-RNN policy per
        # expert.  Bind every command-capable owner up front; the generic
        # optimizer later routes the receipt to the exact trace owner.
        owners = [
            expert for expert in self.registry.chuyen_gia_theo_context.values()
            if bool(getattr(expert, "co_quyen_ra_lenh", False))
        ]
        if not owners:
            return {
                "installed": False,
                "reason": "legacy_command_owner_missing",
                "promotion_allowed": False,
            }
        installed: list[dict[str, object]] = []
        for expert in owners:
            hook = getattr(expert.policy, "install_codex_teacher_reference", None)
            if not callable(hook):
                hook = getattr(expert.policy, "install_five_rlhf_reference", None)
            if not callable(hook):
                continue
            result = hook(reference_policy_sha256)
            if isinstance(result, Mapping) and bool(result.get("installed", False)):
                installed.append(dict(result))
        if len(installed) != len(owners):
            return {
                "installed": False,
                "reason": "legacy_policy_reference_hook_missing",
                "owners": len(owners),
                "installed_owners": len(installed),
                "promotion_allowed": False,
            }
        reference_hashes = sorted({
            str(item.get("reference_policy_sha256", "") or "")
            for item in installed
        })
        return {
            "installed": True,
            "mode": "legacy_direct_rnn_v1",
            "owners": len(owners),
            "reference_policy_sha256": reference_hashes[0] if len(reference_hashes) == 1 else "",
            "reference_policy_hashes": reference_hashes,
            "reference_policy_frozen": True,
            "promotion_allowed": False,
            "evidence_class": "rlhf_reference_candidate_only",
        }

    def optimize_five_rlhf_trace_packet(
        self,
        traces,
        reward_model,
        **kwargs,
    ) -> dict[str, object]:
        """Route one candidate RLHF packet to the sole public Five owner."""

        mode = str(getattr(self.cau_hinh, "che_do_quyet_dinh", "") or "")
        if mode != "nam_tac_nhan_dung_chung":
            raise RuntimeError("RLHF Five update requires the shared public Five mode")
        active_expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if active_expert is None or not bool(active_expert.co_quyen_ra_lenh):
            raise RuntimeError("RLHF Five owner is missing or has no command authority")
        hook = getattr(active_expert.policy, "optimize_five_rlhf_trace_packet", None)
        if not callable(hook):
            raise RuntimeError("public Five owner does not expose the RLHF update hook")
        result = hook(traces, reward_model, **kwargs)
        if not isinstance(result, Mapping):
            raise RuntimeError("RLHF Five update hook returned a non-auditable result")
        return dict(result)

    def get_codex_teacher_reference_policy_sha256(self) -> str:
        """Return the run-bound frozen reference hash for any decision mode."""
        mode = str(getattr(self.cau_hinh, "che_do_quyet_dinh", "") or "")
        if mode == "nam_tac_nhan_dung_chung":
            active = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
            policy = getattr(active, "policy", None) if active is not None else None
            return str(getattr(policy, "five_rlhf_reference_policy_sha256", "") or "")
        for expert in self.registry.chuyen_gia_theo_context.values():
            policy = getattr(expert, "policy", None)
            value = str(getattr(policy, "codex_teacher_reference_policy_sha256", "") or "")
            if value:
                return value
        return ""

    def get_codex_teacher_live_parameter_hash(self) -> str:
        """Return the live Codex trainable-scope hash from the active owner.

        MT5 receives this facade as its policy object, while the real public
        Five network and Codex optimizer live on the ``CTX_CHUNG`` owner.  A
        blank value here would make a valid receipt look causally unverified
        even though the inner action head was updated.  Keep legacy multi-owner
        mode fail-closed unless exactly one owner exposes a hash; the shared
        Five mode has one canonical owner and can delegate directly.
        """
        mode = str(getattr(self.cau_hinh, "che_do_quyet_dinh", "") or "")
        if mode == "nam_tac_nhan_dung_chung":
            active = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
            policy = getattr(active, "policy", None) if active is not None else None
            reader = getattr(policy, "get_codex_teacher_live_parameter_hash", None)
            if callable(reader):
                try:
                    return str(reader() or "")
                except Exception:
                    return ""
            return ""
        hashes: set[str] = set()
        for expert in self.registry.chuyen_gia_theo_context.values():
            policy = getattr(expert, "policy", None)
            reader = getattr(policy, "get_codex_teacher_live_parameter_hash", None)
            if not callable(reader):
                continue
            try:
                value = str(reader() or "")
            except Exception:
                value = ""
            if value:
                hashes.add(value)
        return next(iter(hashes)) if len(hashes) == 1 else ""

    def get_codex_teacher_trace(self, trace_id: str):
        """Read a legacy trace from its exact context-policy owner."""
        key = str(trace_id or "")
        expert = self._trace_to_expert.get(key)
        candidates = []
        if expert:
            selected = next(
                (item for item in self.registry.chuyen_gia_theo_context.values()
                 if str(getattr(item, "ma_chuyen_gia", "")) == str(expert)),
                None,
            )
            if selected is not None:
                candidates.append(selected)
        candidates.extend(
            item for item in self.registry.chuyen_gia_theo_context.values()
            if item not in candidates
        )
        for item in candidates:
            hook = getattr(getattr(item, "policy", None), "get_codex_teacher_trace", None)
            if not callable(hook):
                continue
            trace = hook(key)
            if trace is not None:
                return trace
        return None

    def optimize_codex_teacher_trace_packet(
        self,
        traces,
        reward_model,
        **kwargs,
    ) -> dict[str, object]:
        """Route a Codex receipt to the exact owner in every decision mode."""
        mode = str(getattr(self.cau_hinh, "che_do_quyet_dinh", "") or "")
        if mode == "nam_tac_nhan_dung_chung":
            return self.optimize_five_rlhf_trace_packet(traces, reward_model, **kwargs)
        if not traces:
            raise ValueError("legacy Codex update requires an owner trace")
        expert = self._expert_from_trace(traces[0])
        if expert is None and mode == "mot_bo_nao_dung_chung":
            expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None or not bool(getattr(expert, "co_quyen_ra_lenh", False)):
            raise RuntimeError("legacy Codex owner is missing or has no command authority")
        hook = getattr(expert.policy, "optimize_codex_teacher_trace_packet", None)
        if not callable(hook):
            raise RuntimeError("legacy policy does not expose the Codex update hook")
        result = hook(traces, reward_model, **kwargs)
        if not isinstance(result, Mapping):
            raise RuntimeError("legacy Codex update hook returned a non-auditable result")
        return dict(result)

    def clear_five_rlhf_reference(self) -> dict[str, object]:
        """Remove the ephemeral candidate reference before checkpoint persist."""
        cleared = 0
        for expert in getattr(self, "experts", {}).values():
            local = getattr(expert, "policy", None)
            hook = getattr(local, "clear_five_rlhf_reference", None)
            if not callable(hook):
                continue
            hook()
            cleared += 1
        return {"cleared": cleared, "promotion_allowed": False}

    def act(self, observation: np.ndarray, goal: np.ndarray | None = None):
        self._ensure_context_history_policy_inputs()
        mode = str(getattr(self.cau_hinh, "che_do_quyet_dinh", ""))
        if mode == "nam_tac_nhan_dung_chung":
            raise RuntimeError(
                "che do nam_tac_nhan_dung_chung requires act_nam_tac_nhan(); "
                "single-action serialization was removed by bridge v10"
            )

        shared_mode = mode == "mot_bo_nao_dung_chung"
        if shared_mode:
            # Đường tương thích một tác nhân cũ.
            self._pending_regime_raw_inputs = None
            active_expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
            if active_expert is None or not active_expert.co_quyen_ra_lenh:
                raise RuntimeError("khong tim thay bo nao chung co quyen ra lenh")
            policy = active_expert.policy
            if self._pending_account_context is not None:
                account_context, positions_pipe = self._pending_account_context
                hook = getattr(policy, "set_account_context", None)
                if callable(hook):
                    hook(account_context, positions_pipe)
            pending = dict(self._pending_decision_context or {})
            if pending:
                hook = getattr(policy, "set_decision_context", None)
                if callable(hook):
                    hook(**pending)
            policy_observation = np.asarray(observation, dtype=float).reshape(-1)
            trace = active_expert.ra_quyet_dinh(policy_observation, goal, nhan_dien=None)
            trace = _gan_trace_metadata(trace, {
                "ma_context": "CTX_CHUNG",
                "ma_context_cha": "CTX_CHUNG",
                "ma_chuyen_gia": "BO_NAO_CHUNG",
                "ma_policy": "BO_NAO_CHUNG",
                "decision_policy_id": "BO_NAO_CHUNG",
                "decision_policy_context": "CTX_CHUNG",
                "decision_role": "BO_NAO_DUNG_CHUNG",
                "context_action_contract": "MOT_BO_NAO_DUNG_CHUNG_V1",
                "selector_reason": "khong_dung_bo_chon",
                "selector_used_fallback": False,
                "context_history_inputs_enabled": False,
                "context_history_input_count": 0,
                "context_history_policy_input_count": int(policy_observation.size),
                "entry_feature_interaction_count": 0,
            })
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if trace_id:
                self._trace_to_expert[trace_id] = "BO_NAO_CHUNG"
            self._last_expert_id = "BO_NAO_CHUNG"
            if not self._defer_live_decision_until_bridge:
                self._log_owner_decision(active_expert, trace)
            return trace
        execution_context = self._pending_decision_context.get("execution_context")
        # MT5 gives the facade both a raw bridge view for regime ownership and
        # a normalized policy view for Direct-RNN.  Consume the raw sidecar at
        # exactly one decision so it cannot leak to a later direct call.
        raw_regime_observation = getattr(self, "_pending_regime_raw_inputs", None)
        self._pending_regime_raw_inputs = None
        nhan_dien = self.nhan_dien.nhan_dien(
            observation,
            execution_context=execution_context if isinstance(execution_context, Mapping) else None,
            raw_regime_observation=raw_regime_observation,
        )
        context_history_audit = self._context_history_audit()
        policy_observation, entry_feature_audit = self._policy_observation_with_context_history(observation)
        context_history_audit = {**context_history_audit, **entry_feature_audit}
        active_expert = self.registry.owner(nhan_dien)
        if active_expert is None or not active_expert.co_quyen_ra_lenh:
            trace = _gan_trace_metadata(
                self._quyet_dinh_khong_giao_dich(nhan_dien, "CHUA_CO_CHUYEN_GIA_PHU_HOP"),
                context_history_audit,
            )
            self._record_current_context(nhan_dien.ma_context)
            if not self._defer_live_decision_until_bridge:
                self._log_no_policy_decision(trace)
            return trace

        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) in {"mot_bo_nao_dung_chung", "nam_tac_nhan_dung_chung"}:
            policy = active_expert.policy
            if self._pending_account_context is not None:
                account_context, positions_pipe = self._pending_account_context
                hook = getattr(policy, "set_account_context", None)
                if callable(hook):
                    hook(account_context, positions_pipe)
            pending = dict(self._pending_decision_context or {})
            if pending:
                hook = getattr(policy, "set_decision_context", None)
                if callable(hook):
                    hook(**pending)
            trace = active_expert.ra_quyet_dinh(
                policy_observation, goal, nhan_dien=nhan_dien
            )
            trace = _gan_trace_metadata(trace, {
                "ma_context": str(nhan_dien.ma_context),
                "ma_context_cha": str(nhan_dien.ma_context_cha),
                "ma_chuyen_gia": "BO_NAO_CHUNG",
                "ma_policy": "BO_NAO_CHUNG",
                "decision_policy_id": "BO_NAO_CHUNG",
                "decision_policy_context": "CTX_CHUNG",
                "decision_role": "BO_NAO_DUNG_CHUNG",
                "context_action_contract": "MOT_BO_NAO_DUNG_CHUNG_V1",
                "selector_reason": "khong_dung_bo_chon",
                "selector_used_fallback": False,
                **context_history_audit,
            })
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if trace_id:
                self._trace_to_expert[trace_id] = "BO_NAO_CHUNG"
            self._last_expert_id = "BO_NAO_CHUNG"
            if not self._defer_live_decision_until_bridge:
                self._log_owner_decision(active_expert, trace)
            self._record_current_context(nhan_dien.ma_context)
            return trace

        management_plan = self._management_owner_plan(nhan_dien)
        # A candidate context is still a real training observation.  Stability
        # remains telemetry for the router, but it must not suppress the first
        # entry examples of a new session or regime transition.
        if management_plan is not None:
            manager_expert, target_position, management_only, management_reason = management_plan
            management_owner_expert = str(target_position.get("owner_expert", "") or active_expert.ma_chuyen_gia)
            management_owner_context = str(target_position.get("owner_context", "") or active_expert.ma_context)
            management_target_ticket = str(target_position.get("ticket", "") or "")
            management_target_owner_trace = str(target_position.get("owner_trace", "") or "")
        else:
            manager_expert = None
            target_position = {}
            management_only = False
            management_reason = ""
            management_owner_expert = ""
            management_owner_context = ""
            management_target_ticket = ""
            management_target_owner_trace = ""

        # Layer 1: the resumed context owner remains the unrestricted entry
        # generalist, so v13 knowledge is never bypassed. Layer 2 adds three
        # independent style challengers. Their capability union covers all six
        # entry actions; the selector cannot turn an entry committee decision
        # into HOLD/management.
        proposals: list[dict[str, object]] = [
            self._build_expert_proposal(
                active_expert,
                policy_observation,
                goal,
                nhan_dien,
                role="CONTEXT_ENTRY_GENERALIST",
                management_only=False,
                capability_actions=CHI_SO_ENTRY_AND_PENDING_ACTIONS,
            )
        ]
        if bool(getattr(self.cau_hinh, "entry_style_specialists_bat", True)):
            for style, capability in ENTRY_STYLE_SPECIALISTS.items():
                style_expert = self._role_specialist(
                    style,
                    str(active_expert.ma_context),
                    parent=active_expert,
                )
                if style_expert is None:
                    continue
                proposals.append(self._build_expert_proposal(
                    style_expert,
                    policy_observation,
                    goal,
                    nhan_dien,
                    role=f"ENTRY_STYLE_{style}",
                    management_only=False,
                    capability_actions=capability,
                ))
        if manager_expert is not None and management_only:
            proposals.append(self._build_expert_proposal(
                manager_expert,
                policy_observation,
                goal,
                nhan_dien,
                role="POSITION_MANAGEMENT_SPECIALIST",
                management_only=True,
                management_owner_expert=management_owner_expert,
                management_owner_context=management_owner_context,
                management_target_ticket=management_target_ticket,
                management_target_owner_trace=management_target_owner_trace,
                management_reason=management_reason,
                capability_actions=CHI_SO_MANAGEMENT_ACTIONS,
            ))
        selected, selection = self._select_proposal(
            proposals,
            fallback_policy_id=str(proposals[0]["proposal_id"]),
        )

        expert = selected["expert"]
        trace = selected["trace"]
        exposure_state = selected.get("correlated_exposure_state")
        if not isinstance(exposure_state, CorrelatedExposureState):
            exposure_state = self._correlated_exposure_state(selected)
        trace = self._annotate_correlated_exposure(trace, exposure_state)
        trace = self._annotate_role_readiness(trace, selected)
        allowed_context_actions = set(selected["allowed"])
        trend_exit_audit: dict[str, object] = {}
        for proposal in proposals:
            trend_exit_audit.update(dict(proposal.get("guard_audit", {}) or {}))
        management_only = bool(selected["management_only"])
        management_owner_expert = str(selected["management_owner_expert"])
        management_owner_context = str(selected["management_owner_context"])
        management_target_ticket = str(selected["management_target_ticket"])
        management_target_owner_trace = str(selected["management_target_owner_trace"])
        management_reason = str(selected["management_reason"])
        decision_role = str(selected["role"])
        decision_proposal_id = str(selected["proposal_id"])
        # This shared TimingNet gate sits after committee selection so no
        # individual expert can route around it.  It only changes a fresh
        # entry into HOLD; management traces and all pre-calibration runs are
        # unchanged.
        trace = self._apply_timing_regime_transition_entry_block(
            trace,
            nhan_dien,
            management_only=management_only,
        )
        trace_id = str(getattr(trace, "trace_id", "") or "")
        cache = getattr(expert.policy, "_execution_trace_cache", None)
        if trace_id and isinstance(cache, dict):
            cache[trace_id] = trace
        proposal_audit = [
            {
                "proposal_id": str(item["proposal_id"]),
                "policy_id": str(item["expert"].ma_chuyen_gia),
                "policy_instance_uuid": str(_policy_identity(item["expert"].policy).get("policy_instance_uuid", "")),
                "role": str(item["role"]),
                "action_index": self._proposal_action_index(item["trace"]),
                "allowed_actions": sorted(int(index) for index in item["allowed"]),
                "pending_outcomes": self._selector_pending_outcomes(
                    str(item["proposal_id"]),
                    str(item.get("selector_context", "")),
                ),
                "correlated_exposure_stage": str(
                    getattr(item.get("correlated_exposure_state"), "stage", "PRIMARY")
                ),
                "correlated_exposure_count": int(
                    getattr(item.get("correlated_exposure_state"), "exposure_count", 0)
                ),
                "selector_utility_context": str(
                    item.get("selector_utility_context", item.get("selector_context", ""))
                ),
                "selector_evidence_group": str(item.get("selector_evidence_group", "")),
                "selector_group_support": int(self.policy_selector.group_evidence(
                    str(item.get("selector_evidence_group", ""))
                )[0]),
                "selector_group_utility": float(self.policy_selector.group_evidence(
                    str(item.get("selector_evidence_group", ""))
                )[1]),
                "selector_group_loss_streak": int(self.policy_selector.group_evidence(
                    str(item.get("selector_evidence_group", ""))
                )[3]),
                "selector_policy_group_key": self.policy_selector._policy_group_key(
                    str(item.get("proposal_id", "")),
                    str(item.get("selector_evidence_group", "")),
                ),
                "selector_policy_group_loss_streak": int(
                    getattr(self.policy_selector, "policy_group_loss_streak", {}).get(
                        self.policy_selector._policy_group_key(
                            str(item.get("proposal_id", "")),
                            str(item.get("selector_evidence_group", "")),
                        ),
                        0,
                    )
                ),
                "selector_evidence_support": int(item.get("selector_evidence_support", 0)),
                "selector_lifecycle_support": int(item.get("selector_lifecycle_support", 0)),
                "selector_support": int(item.get("selector_support", 0)),
                "selector_readiness": float(item.get("selector_readiness", 0.0)),
                "selector_cold_start": bool(item.get("selector_cold_start", False)),
                "selector_microstate": str(item.get("selector_microstate", "")),
                "candidate_context": str(item.get("candidate_context", "")),
                "transition_active": bool(item.get("transition_active", False)),
                "selected": str(item["proposal_id"]) == selection.selected_policy_id,
                "score": float(np.nan_to_num(
                    float(selection.scores.get(str(item["proposal_id"]), -1.0)),
                    nan=-1.0,
                    posinf=1.0,
                    neginf=-1.0,
                )),
            }
            for item in proposals
        ]
        trace = _gan_trace_metadata(trace, {
            "context_allowed_actions": sorted(int(index) for index in allowed_context_actions),
            "context_task": nhan_dien.nhiem_vu.value,
            "context_action_contract": "FIXED_CONTEXT_OWNER_V32",
            "management_only": bool(management_only),
            "management_owner_expert": str(management_owner_expert),
            "management_owner_context": str(management_owner_context),
            "management_target_ticket": str(management_target_ticket),
            "management_target_owner_trace": str(management_target_owner_trace),
            "management_reason": str(management_reason),
            "timing_quality": float(nhan_dien.timing_quality),
            "timing_direction": int(nhan_dien.timing_direction),
            "timing_source": str(nhan_dien.timing_source),
            "timing_network_quality": float(nhan_dien.timing_network_quality),
            "timing_network_direction": float(nhan_dien.timing_network_direction),
            "timing_network_entry_probability": float(nhan_dien.timing_network_entry_probability),
            "timing_network_uncertainty": float(nhan_dien.timing_network_uncertainty),
            "timing_network_update_count": int(nhan_dien.timing_network_update_count),
            "timing_network_regime_transition_probability": float(nhan_dien.timing_network_regime_transition_probability),
            "timing_network_regime_transition_probabilities": [
                float(value) for value in nhan_dien.timing_network_regime_transition_probabilities
            ],
            "timing_network_regime_transition_update_count": int(nhan_dien.timing_network_regime_transition_update_count),
            "timing_network_regime_transition_calibration_count": int(nhan_dien.timing_network_regime_transition_calibration_count),
            "timing_network_regime_transition_brier": float(nhan_dien.timing_network_regime_transition_brier),
            "timing_network_regime_transition_high_confidence_support": int(nhan_dien.timing_network_regime_transition_high_confidence_support),
            "timing_network_regime_transition_high_confidence_precision": float(nhan_dien.timing_network_regime_transition_high_confidence_precision),
            "candidate_context": str(nhan_dien.candidate_context or nhan_dien.ma_context),
            "context_transition_active": bool(nhan_dien.transition_active),
            "context_transition_progress": float(nhan_dien.transition_progress),
            "selector_microstate": str(selected.get("selector_microstate", "")),
            "selector_utility_context": str(selected.get("selector_utility_context", exposure_state.utility_context)),
            "selector_evidence_group": str(selected.get("selector_evidence_group", "")),
            "selector_policy_group_loss_streak": int(
                getattr(self.policy_selector, "policy_group_loss_streak", {}).get(
                    self.policy_selector._policy_group_key(
                        str(selected.get("proposal_id", "")),
                        str(selected.get("selector_evidence_group", "")),
                    ),
                    0,
                )
            ),
            "selector_selected_policy": str(selection.selected_policy_id),
            "decision_policy_id": str(expert.ma_chuyen_gia),
            "decision_policy_context": str(expert.ma_context),
            "decision_role": decision_role,
             "selector_reason": str(selection.reason),
             "selector_used_fallback": bool(selection.used_fallback),
             "selector_cold_start_probe_count": int(getattr(self, "_selector_cold_start_probe_count", 0) or 0),
             "selector_cold_start_probe_budget": int(getattr(self.cau_hinh, "selector_cold_start_max_per_run", 0) or 0),
            "selector_proposals": proposal_audit,
            "entry_proposal_evaluated": any("ENTRY" in item["role"] for item in proposal_audit),
            "entry_actions_available_in_selector": CHI_SO_ENTRY_ACTIONS.issubset(set().union(*(
                set(item["allowed_actions"]) for item in proposal_audit if "ENTRY" in item["role"]
            ))),
            "entry_proposal_not_blocked": str(selection.reason) == "entry_committee_selected_no_block",
            **context_history_audit,
            **trend_exit_audit,
        })
        trace_id = str(getattr(trace, "trace_id", ""))
        if trace_id:
            self._trace_to_expert[trace_id] = expert.ma_chuyen_gia
            if self._proposal_action_index(trace) in CHI_SO_ENTRY_ACTIONS:
                timing_trace = dict(getattr(self.nhan_dien, "_last_timing_trace", {}) or {})
                if timing_trace:
                    self._trace_to_timing_network_trace[trace_id] = timing_trace
            self._trace_to_selector_policy[trace_id] = str(selection.selected_policy_id)
            self._trace_to_selector_context[trace_id] = str(
                selected.get("selector_utility_context", exposure_state.utility_context)
            )
            self._trace_to_selector_evidence_group[trace_id] = str(
                selected.get("selector_evidence_group", "")
            )
            if self._proposal_action_index(trace) in CHI_SO_ENTRY_ACTIONS:
                self._selector_pending_entry_traces[trace_id] = (
                    str(nhan_dien.ma_context),
                    str(selection.selected_policy_id),
                    self._entry_direction(self._proposal_action_index(trace)),
                )
                self._selector_pending_entry_ages[trace_id] = 0
        self._last_expert_id = expert.ma_chuyen_gia
        position_owner = active_expert
        if decision_role == "POSITION_MANAGEMENT_SPECIALIST" and management_owner_expert:
            position_owner = next(
                (
                    item for item in self.registry.chuyen_gia_theo_context.values()
                    if item.ma_chuyen_gia == management_owner_expert
                ),
                active_expert,
            )
        decision_identity = _policy_identity(expert.policy)
        owner_identity = _policy_identity(position_owner.policy)
        trace = _with_owner_identity(trace, {
            "ma_context": str(position_owner.ma_context),
            "ma_context_cha": str(position_owner.ma_context_cha),
            "ma_chuyen_gia": str(position_owner.ma_chuyen_gia),
            "ma_policy": str(position_owner.ma_chuyen_gia),
            "phien_ban_policy": str(position_owner.ma_phien_ban),
            "trang_thai_chuyen_gia": str(position_owner.trang_thai.value),
            **owner_identity,
            "decision_policy_identity": decision_identity,
            "context_nhan_dien": nhan_dien.ma_context,
            "context_on_dinh": nhan_dien.on_dinh,
            "do_tin_cay_context": nhan_dien.do_tin_cay,
            "ly_do_context": nhan_dien.ly_do,
            "khong_co_policy_ra_lenh": False,
        })
        # With a direct/local caller the owner trace is already final. MT5,
        # however, appends semantic/emitted/wire contract evidence after this
        # method returns; in that path defer emission to ``finalize_bridge_decision``.
        if not self._defer_live_decision_until_bridge:
            self._log_owner_decision(expert, trace)
        self._record_current_context(nhan_dien.ma_context)
        return trace

    def _expert_from_trace(self, trace: object) -> ChuyenGiaDocLap | None:
        audit = _trace_metadata(trace)
        trace_id = str(getattr(trace, "trace_id", "") or "")
        # Position ownership written to MT5 is deliberately separate from the
        # specialist policy that produced the selected proposal. Credit must
        # follow the decision-policy mapping first.
        ma = str(self._trace_to_expert.get(trace_id, "") or audit.get("decision_policy_id", "") or audit.get("ma_chuyen_gia", ""))
        return next((item for item in self.registry.chuyen_gia_theo_context.values() if item.ma_chuyen_gia == ma), None)

    @property
    def input_size(self) -> int:
        # MT5 sees the immutable EA wire, not Python-local history/interaction
        # values appended immediately before Direct-RNN inference.
        if self._context_history_width() > 0 or self._entry_feature_interaction_width() > 0:
            return INPUT_COUNT
        return int(getattr(next(iter(self.registry.chuyen_gia_theo_context.values())).policy, "input_size", 119) or 119)

    @property
    def output_size(self) -> int:
        return int(getattr(next(iter(self.registry.chuyen_gia_theo_context.values())).policy, "output_size", SO_DAU_RA_BRIDGE) or SO_DAU_RA_BRIDGE)

    @property
    def experience_buffer(self) -> list[object]:
        # Không có buffer chung. Getter chỉ phục vụ tương thích bridge.
        return []

    @experience_buffer.setter
    def experience_buffer(self, value: object) -> None:
        # Reset bridge chỉ được reset bộ đệm cục bộ từng expert, không tạo memory dùng chung.
        for expert in self.registry.chuyen_gia_theo_context.values():
            if hasattr(expert.policy, "experience_buffer"):
                expert.policy.experience_buffer = list(value) if isinstance(value, list) else []

    def ensure_compatible(self) -> None:
        # A resumed facade owns one canonical configuration, but older
        # artifacts may still carry separate config objects inside the
        # classifier and registry. Rebind those references
        # before migration.  Otherwise changing the resumed facade config can
        # disable volatile subexperts at the registry level while the stale
        # classifier keeps emitting volatile subtype contexts.
        for component_name in ("nhan_dien", "registry"):
            component = getattr(self, component_name, None)
            if component is not None and hasattr(component, "cau_hinh"):
                component.cau_hinh = self.cau_hinh
        if not isinstance(getattr(self, "_trace_to_selector_policy", None), Mapping):
            self._trace_to_selector_policy = {}
        else:
            self._trace_to_selector_policy = dict(self._trace_to_selector_policy)
        if not isinstance(getattr(self, "_trace_to_selector_context", None), Mapping):
            self._trace_to_selector_context = {}
        else:
            self._trace_to_selector_context = dict(self._trace_to_selector_context)
        if not isinstance(getattr(self, "_trace_to_selector_evidence_group", None), Mapping):
            self._trace_to_selector_evidence_group = {}
        else:
            self._trace_to_selector_evidence_group = dict(self._trace_to_selector_evidence_group)
        timing_traces = getattr(self, "_trace_to_timing_network_trace", {})
        self._trace_to_timing_network_trace = (
            dict(timing_traces) if isinstance(timing_traces, Mapping) else {}
        )
        if not isinstance(getattr(self, "_selector_pending_entry_traces", None), Mapping):
            self._selector_pending_entry_traces = {}
        else:
            self._selector_pending_entry_traces = {
                str(trace_id): self._pending_entry_parts(value)
                for trace_id, value in dict(self._selector_pending_entry_traces).items()
            }
        pending_ages = getattr(self, "_selector_pending_entry_ages", {})
        self._selector_pending_entry_ages = (
            {str(trace_id): max(0, int(age or 0)) for trace_id, age in dict(pending_ages).items()}
            if isinstance(pending_ages, Mapping) else {}
        )
        if not isinstance(getattr(self, "_selector_credited_trace_ids", None), Mapping):
            self._selector_credited_trace_ids = {}
        else:
            self._selector_credited_trace_ids = dict(self._selector_credited_trace_ids)
        try:
            self._selector_cold_start_probe_count = max(
                0, int(getattr(self, "_selector_cold_start_probe_count", 0) or 0)
            )
        except (TypeError, ValueError):
            self._selector_cold_start_probe_count = 0
        self._ensure_policy_selector_compatible()
        self._prune_dynamic_context_branches()
        self._prune_removed_volatile_parent_selector()
        # Older artifacts contain only the broad volatile owner. Migrate them
        # once, before policy compatibility hooks, so the new route always has
        # a real owner and does not fall into no-policy HOLD.
        self._ensure_volatile_subexperts(warm_start=True)
        self._ensure_role_specialists(warm_start=True)
        self._prune_disabled_role_specialists()
        for expert in self.registry.chuyen_gia_theo_context.values():
            # ``phat_trien_cau_truc_context`` was removed with the runtime
            # topology-growth path.  The custom artifact loader may have
            # materialized its old controller as an inert compatibility
            # carrier; discard it before the policy can be used or re-saved.
            # This is a one-way artifact migration, not a fallback that
            # restores ``develop()`` or any structural mutation behaviour.
            if hasattr(expert.policy, "__dict__"):
                expert.policy.__dict__.pop("_context_structural_growth_controller", None)
            hook = getattr(expert.policy, "ensure_compatible", None)
            if callable(hook):
                hook()
        for attr in (
            "_pending_regime_probabilities",
            "_pending_regime_name",
            "_pending_regime_index",
            "_pending_regime_raw_inputs",
        ):
            if attr not in self.__dict__:
                setattr(self, attr, None)
        self._ensure_context_history_policy_inputs()

    def reset_runtime_state(self) -> None:
        classifier_reset = getattr(self.nhan_dien, "reset_runtime_state", None)
        if callable(classifier_reset):
            classifier_reset()
        # Selection traces are observability-only routing state.  Persisting
        # the final fallback/trace from a previous tester session is neither
        # training evidence nor part of the current market surface; a resumed
        # Full-RL run must start with the same selector runtime as a frozen
        # replay.  Learned selector evidence is intentionally not touched here.
        selector = getattr(self, "policy_selector", None)
        if isinstance(selector, PolicyEnsembleSelector):
            selector.last_trace = None
            selector.fallback_policy_id = ""
            selector._selection_has_non_cold_candidate = False
        for expert in self.registry.chuyen_gia_theo_context.values():
            hook = getattr(expert.policy, "reset_runtime_state", None)
            if callable(hook):
                hook()
        self._trace_to_expert.clear()
        self._trace_to_timing_network_trace.clear()
        self._trace_to_selector_policy.clear()
        self._trace_to_selector_context.clear()
        self._trace_to_selector_evidence_group.clear()
        self._selector_pending_entry_traces.clear()
        self._selector_pending_entry_ages.clear()
        self._selector_credited_trace_ids.clear()
        self._selector_cold_start_probe_count = 0
        self._bridge_finalized_trace_ids.clear()
        self._pending_account_context = None
        self._pending_decision_context = {}
        self._pending_starvation = None
        self._pending_regime_probabilities = None
        self._pending_regime_name = None
        self._pending_regime_index = None
        self._pending_regime_raw_inputs = None
        self._context_history = []
        self._defer_live_decision_until_bridge = False
        self._last_expert_id = ""
        self._no_decision_sequence = 0

    def reset_intraday_state(self, trading_day: object = "") -> None:
        for expert in self.registry.chuyen_gia_theo_context.values():
            hook = getattr(expert.policy, "reset_intraday_state", None)
            if callable(hook):
                hook(trading_day)
        # Owner routing is lifecycle state, not intraday account state. Keep
        # it across day boundaries because MT5 settles the final position
        # ledger after the whole tester pass; clearing it here silently drops
        # feedback for every position opened on an earlier day.
        max_owner_traces = 16384
        if len(self._trace_to_expert) > max_owner_traces:
            self._trace_to_expert = dict(list(self._trace_to_expert.items())[-max_owner_traces:])
        if len(self._trace_to_timing_network_trace) > max_owner_traces:
            self._trace_to_timing_network_trace = dict(
                list(self._trace_to_timing_network_trace.items())[-max_owner_traces:]
            )
        if len(self._trace_to_selector_policy) > max_owner_traces:
            self._trace_to_selector_policy = dict(list(self._trace_to_selector_policy.items())[-max_owner_traces:])
        if len(self._trace_to_selector_context) > max_owner_traces:
            self._trace_to_selector_context = dict(list(self._trace_to_selector_context.items())[-max_owner_traces:])
        if len(self._trace_to_selector_evidence_group) > max_owner_traces:
            self._trace_to_selector_evidence_group = dict(
                list(self._trace_to_selector_evidence_group.items())[-max_owner_traces:]
            )
        if len(self._selector_pending_entry_traces) > max_owner_traces:
            self._selector_pending_entry_traces = dict(
                list(self._selector_pending_entry_traces.items())[-max_owner_traces:]
            )
        self._bridge_finalized_trace_ids.clear()
        self._defer_live_decision_until_bridge = False
        self._last_expert_id = ""

    def enable_live_observatory(self, settings: object, *, run_id: str = "", source: str = "he_chuyen_gia", metadata: Mapping[str, object] | None = None) -> None:
        """Bật telemetry tại facade sau khi context-owner hoàn tất trace.

        Không gắn writer trực tiếp vào ``policy.act``. Nếu làm vậy telemetry sẽ
        được emit trước khi facade thêm context, owner và growth review; đó là
        nguồn gốc log rỗng/mất ownership ở các bản trước.
        """
        from .live_observatory import LiveTelemetryWriter
        self._he_chuyen_gia_live_settings = settings
        self._he_chuyen_gia_live_metadata = dict(metadata or {})
        self._he_chuyen_gia_live_source = str(source)
        self._root_run_id = str(run_id or "he-chuyen-gia")
        self._he_chuyen_gia_live_writer = LiveTelemetryWriter.from_config(settings, run_id=self._root_run_id, source=source)
        self._expert_live_writers = {}
        for expert in self.registry.chuyen_gia_theo_context.values():
            self._attach_live_writer_for_expert(expert)

    def _attach_live_writer_for_expert(self, expert: ChuyenGiaDocLap):
        """Gắn writer cho cả specialist được tạo lười sau khi live đã bật."""

        existing = dict(getattr(self, "_expert_live_writers", {}) or {}).get(expert.ma_chuyen_gia)
        if existing is not None:
            return existing
        settings = getattr(self, "_he_chuyen_gia_live_settings", None)
        if settings is None:
            return None
        from .live_observatory import LiveTelemetryWriter

        expert_run_id = f"{self._root_run_id}::{_safe_token(expert.ma_chuyen_gia)}"
        self._expert_run_ids[expert.ma_chuyen_gia] = expert_run_id
        provenance = {
            "root_run_id": self._root_run_id,
            "expert_run_id": expert_run_id,
            "ma_context": expert.ma_context,
            "ma_chuyen_gia": expert.ma_chuyen_gia,
            "ma_policy": expert.ma_chuyen_gia,
            "trang_thai_chuyen_gia": expert.trang_thai.value,
            **_policy_identity(expert.policy),
        }
        setattr(expert.policy, "_context_expert_runtime_provenance", provenance)
        setattr(expert.policy, "_live_observer", None)
        setattr(expert.policy, "_live_observer_started", False)
        writer = LiveTelemetryWriter.from_config(
            settings,
            run_id=self._root_run_id,
            source=str(getattr(self, "_he_chuyen_gia_live_source", "he_chuyen_gia")),
        )
        self._expert_live_writers[expert.ma_chuyen_gia] = writer
        if bool(getattr(writer, "enabled", False)):
            writer.start(
                expert.policy,
                metadata={**dict(getattr(self, "_he_chuyen_gia_live_metadata", {}) or {}), **provenance},
            )
        return writer

    def _writer_for_expert(self, expert: ChuyenGiaDocLap | None):
        if expert is None:
            return None
        writer = dict(getattr(self, "_expert_live_writers", {}) or {}).get(expert.ma_chuyen_gia)
        if writer is None:
            writer = self._attach_live_writer_for_expert(expert)
        return writer if writer is not None and bool(getattr(writer, "enabled", False)) else None

    def _log_owner_decision(self, expert: ChuyenGiaDocLap, trace: object) -> None:
        writer = self._writer_for_expert(expert)
        if writer is None:
            return
        try:
            writer.decision(expert.policy, trace)
        except Exception:
            return

    def set_live_observatory_trial(self, **kwargs: object) -> None:
        for expert in self.registry.chuyen_gia_theo_context.values():
            writer = self._writer_for_expert(expert)
            if writer is not None:
                writer.trial(expert.policy, policy_id=int(kwargs.get("policy_id", 0) or 0), round_index=int(kwargs.get("round_index", 0) or 0), phase=str(kwargs.get("phase", "mt5")))

    def live_observatory_trial_finished(self, **kwargs: object) -> None:
        for expert in self.registry.chuyen_gia_theo_context.values():
            writer = self._writer_for_expert(expert)
            if writer is not None:
                writer.trial_finished(expert.policy, policy_id=int(kwargs.get("policy_id", 0) or 0), round_index=int(kwargs.get("round_index", 0) or 0), fitness=float(kwargs.get("fitness", 0.0) or 0.0), metrics=kwargs.get("metrics") if isinstance(kwargs.get("metrics"), Mapping) else None)

    def stop_live_observatory(self, *, reason: str = "completed") -> None:
        for expert in self.registry.chuyen_gia_theo_context.values():
            writer = self._writer_for_expert(expert)
            if writer is not None:
                try:
                    writer.emit("run_finished", {"reason": str(reason), "mode": "mot_context_mot_chuyen_gia", "ma_chuyen_gia": expert.ma_chuyen_gia})
                    writer.enabled = False
                except Exception:
                    pass
        writer = getattr(self, "_he_chuyen_gia_live_writer", None)
        if writer is not None:
            try:
                writer.emit("run_finished", {"reason": str(reason), "mode": "mot_context_mot_chuyen_gia"})
                writer.enabled = False
            except Exception:
                pass

    def _log_no_policy_decision(self, trace: object) -> None:
        writer = getattr(self, "_he_chuyen_gia_live_writer", None)
        if writer is None or not bool(getattr(writer, "enabled", False)):
            return
        try:
            writer.decision(self, trace)
        except Exception:
            # Observability tuyệt đối không được tác động đến route hay MT5.
            return

    def activate(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self.act(observation).selected_action, dtype=float)

    def set_account_context(self, account_context: Mapping[str, float], positions_pipe: str = "") -> None:
        # The request parser may reuse/mutate its mapping on the next MT5 tick.
        # Freeze this facade-side context before it is routed to an owner policy.
        try:
            frozen_account = copy.deepcopy(dict(account_context))
        except Exception:
            frozen_account = dict(account_context)
        self._pending_account_context = (frozen_account, str(positions_pipe))

    def set_regime_raw_inputs(self, raw_inputs: object | None) -> None:
        """Set the immutable raw bridge view for the next M15 routing call.

        This sidecar never reaches an inner policy, memory key or Full-RL
        cache.  It exists solely because the deterministic regime classifier
        needs native M15/H1 ATR ratios while the policy must keep its
        normalized representation.
        """

        if raw_inputs is None:
            self._pending_regime_raw_inputs = None
            return
        try:
            values = np.asarray(raw_inputs, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            self._pending_regime_raw_inputs = None
            return
        self._pending_regime_raw_inputs = values.copy() if values.size else None

    def set_regime_soft_probabilities(
        self,
        soft_probs: Sequence[float] | np.ndarray | None,
        regime_name: str | None = None,
        regime_idx: int | None = None,
    ) -> None:
        """Cache current market regime soft probabilities for policy decision provenance."""
        if soft_probs is None:
            self._pending_regime_probabilities = None
            self._pending_regime_name = None
            self._pending_regime_index = None
            return
        try:
            probs = np.asarray(soft_probs, dtype=float).ravel()
            if probs.size != 3 or np.isnan(probs).any() or np.isinf(probs).any():
                self._pending_regime_probabilities = None
                self._pending_regime_name = None
                self._pending_regime_index = None
            prob_sum = float(np.sum(probs))
            if not np.isclose(prob_sum, 1.0, atol=1e-6) and prob_sum > 0:
                probs = probs / prob_sum
            self._pending_regime_probabilities = tuple(round(float(p), 6) for p in probs)
            self._pending_regime_name = str(regime_name or "")
            self._pending_regime_index = int(regime_idx) if regime_idx is not None else int(np.argmax(probs))
        except Exception:
            self._pending_regime_probabilities = None
            self._pending_regime_name = None
            self._pending_regime_index = None

    def set_decision_context(self, **kwargs: object) -> None:
        # This is an observability-only instruction for the facade; the inner
        # policy has a strict signature and must never receive it.
        self._defer_live_decision_until_bridge = bool(kwargs.pop("defer_live_decision_until_bridge", False))
        try:
            self._pending_decision_context = copy.deepcopy(dict(kwargs))
        except Exception:
            self._pending_decision_context = dict(kwargs)

    def finalize_bridge_decision(self, trace: object) -> object:
        """Emit exactly one live decision event after MT5 contract enrichment.

        This method is called by ``MT5BridgeRunner`` only after the bridge has
        frozen ``semantic_action``, ``emitted_action``, ``wire_action``,
        owner-contract validation and bootstrap state into the trace. It is
        idempotent because sidecar retries must not duplicate an audit event.
        """
        trace_id = str(getattr(trace, "trace_id", "") or "")
        if trace_id and trace_id in self._bridge_finalized_trace_ids:
            return trace
        try:
            expert = self._expert_from_trace(trace)
            if expert is None:
                self._log_no_policy_decision(trace)
            else:
                refresh_causal = getattr(expert.policy, "refresh_bridge_decision", None)
                if callable(refresh_causal):
                    try:
                        refresh_causal(trace)
                    except Exception:
                        # Causal telemetry is best-effort and must never
                        # suppress the authoritative bridge decision record.
                        pass
                self._log_owner_decision(expert, trace)
        except Exception:
            # Telemetry remains strictly non-authoritative for trading.
            return trace
        if trace_id:
            self._bridge_finalized_trace_ids[trace_id] = None
            # Bounded de-duplication registry: trace ids are only needed for
            # the current run, not as persistent state.
            while len(self._bridge_finalized_trace_ids) > 8192:
                self._bridge_finalized_trace_ids.pop(next(iter(self._bridge_finalized_trace_ids)))
        return trace

    def apply_entry_timing_extrema_bias(self, trace: object) -> object:
        """Áp bài học đỉnh/đáy của đúng expert owner vào trace cuối."""
        expert = self._expert_from_trace(trace)
        if expert is None:
            return trace
        hook = getattr(expert.policy, "apply_entry_timing_extrema_bias", None)
        return hook(trace) if callable(hook) else trace

    def record_position_context_extrema(self, event: Mapping[str, object]) -> object:
        """Ghi bài học đỉnh/đáy của position vào đúng expert sở hữu lệnh."""
        payload = copy.deepcopy(dict(event or {}))
        owner_expert = str(payload.get("position_owner_expert", "") or "")
        owner_context = str(payload.get("position_owner_context", "") or "")
        expert = next(
            (
                item
                for item in self.registry.chuyen_gia_theo_context.values()
                if item.ma_chuyen_gia == owner_expert or item.ma_context == owner_context
            ),
            None,
        )
        if expert is None:
            return None
        full_rl_mode = getattr(expert.policy, "_direct_rnn_full_rl_mode_enabled", None)
        if callable(full_rl_mode) and bool(full_rl_mode()):
            # Do not fall through to the compatibility append below.  The
            # Direct-RNN Full-RL contract forbids position-extrema memory from
            # becoming a second outcome writer; the policy hook emits the
            # corresponding audit-only skip marker.
            payload["full_rl_position_extrema_memory_skipped"] = True
            return payload
        hook = getattr(expert.policy, "record_position_context_extrema", None)
        if callable(hook):
            return hook(payload)
        events = list(getattr(expert.policy, "position_context_extrema_memory", []) or [])
        events.append(payload)
        setattr(expert.policy, "position_context_extrema_memory", events[-512:])
        return payload

    def copy(self) -> "HeChuyenGiaMotContextMotPolicy":
        return self.sao_chep()

    def __getstate__(self):
        state = dict(self.__dict__)
        # Factory closure không phải lúc nào pickle được. Policy instances vẫn
        # được serialize đầy đủ; muốn tạo nhánh mới sau load phải gắn factory.
        state["tao_policy"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Backward-compatible rehydration for policy archives written before
        # bridge-final telemetry was introduced. Existing trained systems must
        # remain loadable without requiring a retrain.
        if not isinstance(getattr(self, "_pending_decision_context", None), Mapping):
            self._pending_decision_context = {}
        history = getattr(self, "_context_history", ())
        self._context_history = [
            str(item) for item in list(history or ()) if str(item)
        ][-CONTEXT_HISTORY_INPUT_COUNT:]
        if not hasattr(self, "_pending_starvation"):
            self._pending_starvation = None
        for attr in (
            "_pending_regime_probabilities",
            "_pending_regime_name",
            "_pending_regime_index",
            "_pending_regime_raw_inputs",
        ):
            if attr not in self.__dict__:
                setattr(self, attr, None)
        if not hasattr(self, "_defer_live_decision_until_bridge"):
            self._defer_live_decision_until_bridge = False
        finalized = getattr(self, "_bridge_finalized_trace_ids", {})
        self._bridge_finalized_trace_ids = dict(finalized) if isinstance(finalized, Mapping) else {}
        timing_traces = getattr(self, "_trace_to_timing_network_trace", {})
        self._trace_to_timing_network_trace = (
            dict(timing_traces) if isinstance(timing_traces, Mapping) else {}
        )
        selector_traces = getattr(self, "_trace_to_selector_policy", {})
        self._trace_to_selector_policy = dict(selector_traces) if isinstance(selector_traces, Mapping) else {}
        selector_contexts = getattr(self, "_trace_to_selector_context", {})
        self._trace_to_selector_context = dict(selector_contexts) if isinstance(selector_contexts, Mapping) else {}
        selector_groups = getattr(self, "_trace_to_selector_evidence_group", {})
        self._trace_to_selector_evidence_group = (
            dict(selector_groups) if isinstance(selector_groups, Mapping) else {}
        )
        pending_entries = getattr(self, "_selector_pending_entry_traces", {})
        self._selector_pending_entry_traces = (
            dict(pending_entries) if isinstance(pending_entries, Mapping) else {}
        )
        pending_ages = getattr(self, "_selector_pending_entry_ages", {})
        self._selector_pending_entry_ages = (
            {str(trace_id): max(0, int(age or 0)) for trace_id, age in dict(pending_ages).items()}
            if isinstance(pending_ages, Mapping) else {}
        )
        selector_credits = getattr(self, "_selector_credited_trace_ids", {})
        self._selector_credited_trace_ids = dict(selector_credits) if isinstance(selector_credits, Mapping) else {}
        self._ensure_policy_selector_compatible()

    def _ensure_policy_selector_compatible(self) -> None:
        if not isinstance(getattr(self, "policy_selector", None), PolicyEnsembleSelector):
            self.policy_selector = PolicyEnsembleSelector(fallback_policy_id="", min_score=-1.0)
        if getattr(self.policy_selector, "exploration_weight", None) is None:
            self.policy_selector.exploration_weight = 0.25
        # These are profile controls, not learned state.  Re-apply them after
        # loading an older candidate so a continuation cannot silently retain
        # the weaker selector hyperparameters from its source artifact.
        self.policy_selector.inflight_penalty = float(getattr(self.cau_hinh, "selector_inflight_penalty", 0.20))
        self.policy_selector.exposure_penalty = float(getattr(self.cau_hinh, "selector_exposure_penalty", 0.20))
        self.policy_selector.loss_streak_penalty = float(
            getattr(self.cau_hinh, "selector_loss_streak_penalty", 0.12)
        )
        self.policy_selector.policy_group_loss_streak_penalty = float(
            getattr(self.cau_hinh, "selector_policy_group_loss_streak_penalty", 0.0)
        )
        self.policy_selector.cold_start_penalty = float(getattr(self.cau_hinh, "selector_cold_start_penalty", 0.0))
        self.policy_selector.cold_start_support = int(getattr(self.cau_hinh, "selector_cold_start_support", 8))
        # V15 selector migration: old V14 candidates used alpha=0.10, which
        # allowed proposal confidence to dominate several confirmed MT5 losses.
        self.policy_selector.utility_alpha = max(
            0.35,
            float(getattr(self.policy_selector, "utility_alpha", 0.35) or 0.35),
        )
        if getattr(self.policy_selector, "win_signal_alpha", None) is None:
            self.policy_selector.win_signal_alpha = 0.35
        if getattr(self.policy_selector, "loss_streak_penalty", None) is None:
            self.policy_selector.loss_streak_penalty = float(
                getattr(self.cau_hinh, "selector_loss_streak_penalty", 0.12)
            )
        if not isinstance(getattr(self.policy_selector, "win_signal", None), Mapping):
            self.policy_selector.win_signal = dict(getattr(self.policy_selector, "learned_utility", {}) or {})
        else:
            self.policy_selector.win_signal = dict(self.policy_selector.win_signal)
        if not isinstance(getattr(self.policy_selector, "loss_streak", None), Mapping):
            self.policy_selector.loss_streak = {}
        else:
            self.policy_selector.loss_streak = dict(self.policy_selector.loss_streak)
        for field_name in (
            "group_learned_utility",
            "group_outcome_count",
            "group_win_signal",
            "group_loss_streak",
            "policy_group_loss_streak",
        ):
            values = getattr(self.policy_selector, field_name, {})
            setattr(
                self.policy_selector,
                field_name,
                dict(values) if isinstance(values, Mapping) else {},
            )
        if not isinstance(getattr(self, "exposure_supervisor", None), CorrelatedExposureSupervisor):
            self.exposure_supervisor = CorrelatedExposureSupervisor(
                risk_decay=float(self.cau_hinh.correlated_exposure_risk_decay),
                min_risk_pct=float(self.cau_hinh.correlated_exposure_min_risk_pct),
            )
        else:
            self.exposure_supervisor.risk_decay = float(np.clip(
                float(self.cau_hinh.correlated_exposure_risk_decay), 0.05, 1.0
            ))
            self.exposure_supervisor.min_risk_pct = max(
                1e-9, float(self.cau_hinh.correlated_exposure_min_risk_pct)
            )
        if not isinstance(getattr(self, "role_readiness_supervisor", None), RoleReadinessSupervisor):
            self.role_readiness_supervisor = RoleReadinessSupervisor(
                min_risk_multiplier=float(self.cau_hinh.role_readiness_min_risk_multiplier),
                support_scale=float(self.cau_hinh.role_readiness_support_scale),
                negative_utility_weight=float(self.cau_hinh.role_readiness_negative_utility_weight),
            )
        else:
            self.role_readiness_supervisor.min_risk_multiplier = float(np.clip(
                float(self.cau_hinh.role_readiness_min_risk_multiplier), 0.05, 1.0
            ))
            self.role_readiness_supervisor.support_scale = max(
                1.0, float(self.cau_hinh.role_readiness_support_scale)
            )
            self.role_readiness_supervisor.negative_utility_weight = float(np.clip(
                float(self.cau_hinh.role_readiness_negative_utility_weight), 0.0, 1.0
            ))

    def gan_factory_tao_policy(self, tao_policy: Callable[[], Any]) -> None:
        self.tao_policy = tao_policy

    def _credit_selector_outcome(self, trace_id: str, proposal_id: str, utility: float) -> None:
        trace_id = str(trace_id or "")
        proposal_id = str(proposal_id or "")
        self._selector_pending_entry_traces.pop(trace_id, None)
        self._selector_pending_entry_ages.pop(trace_id, None)
        # The trace maps are only transient routing state.  Retire them once
        # the final receipt is known so an online candidate cannot retain a
        # stale proposal as latent selector state.
        expert_id = str(getattr(self, "_trace_to_expert", {}).get(trace_id, "") or "")
        expert = next(
            (item for item in self.registry.chuyen_gia_theo_context.values()
             if str(getattr(item, "ma_chuyen_gia", "") or "") == expert_id),
            None,
        )
        context_key = self._trace_to_selector_context.get(trace_id, "")
        evidence_group = self._trace_to_selector_evidence_group.get(trace_id, "")
        # A frozen/OOS pass still has to retire transient inflight bookkeeping
        # so its next decision sees the same live-position surface.  It must
        # never turn a realised outcome into selector evidence, however:
        # selector utility changes later proposal scores and therefore makes a
        # purportedly frozen replay self-modifying.
        if not bool(getattr(self, "_learning_enabled", True)):
            return
        full_rl_mode = getattr(getattr(expert, "policy", None), "_direct_rnn_full_rl_mode_enabled", None)
        if callable(full_rl_mode) and bool(full_rl_mode()):
            # Full-RL owns Direct-RNN learning through the batch optimizer.
            # A selector outcome is a legacy ranking writer: even with one
            # static owner it becomes a hidden source of future action drift
            # if role routing is ever re-enabled.  Keep the settled lifecycle
            # telemetry below, but never persist selector utility/counts here.
            writer = self._writer_for_expert(expert)
            if writer is not None:
                try:
                    writer.emit("selector_update_skipped_full_rl", {
                        "trace_id": trace_id,
                        "proposal_id": proposal_id,
                        "policy_id": expert_id,
                        "context_key": context_key,
                        "evidence_group": evidence_group,
                        "realized_utility": float(utility),
                        "reason": "full_rl_exclusive_writer",
                    })
                except Exception:
                    pass
            self._trace_to_selector_context.pop(trace_id, None)
            self._trace_to_selector_evidence_group.pop(trace_id, None)
            self._trace_to_selector_policy.pop(trace_id, None)
            return
        if not trace_id or not proposal_id or trace_id in self._selector_credited_trace_ids:
            return
        scaled_utility = float(utility) * float(np.clip(
            float(getattr(self, "_mt5_online_update_scale", 1.0)),
            0.0,
            1.0,
        ))
        updated_utility = self.policy_selector.observe_outcome(
            proposal_id,
            scaled_utility,
            context_key=context_key,
            evidence_group=evidence_group,
        )
        self._selector_credited_trace_ids[trace_id] = None
        if len(self._selector_credited_trace_ids) > 8192:
            self._selector_credited_trace_ids = dict(list(self._selector_credited_trace_ids.items())[-4096:])
        # Selector learning is a first-class audit event.  Keep it separate
        # from lifecycle so the live watcher can distinguish a real ranking
        # update from a merely closed position; the event is telemetry-only.
        writer = self._writer_for_expert(expert)
        if writer is not None:
            try:
                selector = self.policy_selector
                utility_key = selector._utility_key(proposal_id, context_key)
                group_key = selector._group_key(evidence_group)
                policy_group_key = selector._policy_group_key(proposal_id, evidence_group)
                writer.emit("selector_update", {
                    "trace_id": trace_id,
                    "proposal_id": proposal_id,
                    "policy_id": expert_id,
                    "context_key": context_key,
                    "evidence_group": evidence_group,
                    "realized_utility": float(scaled_utility),
                    "updated_utility": float(updated_utility),
                    "outcome_count": int(getattr(selector, "outcome_count", {}).get(utility_key, 0)),
                    "loss_streak": int(getattr(selector, "loss_streak", {}).get(utility_key, 0)),
                    "group_key": group_key,
                    "group_outcome_count": int(getattr(selector, "group_outcome_count", {}).get(group_key, 0)) if group_key else 0,
                    "group_loss_streak": int(getattr(selector, "group_loss_streak", {}).get(group_key, 0)) if group_key else 0,
                    "policy_group_key": policy_group_key,
                    "policy_group_loss_streak": int(
                        getattr(selector, "policy_group_loss_streak", {}).get(policy_group_key, 0)
                    ) if policy_group_key else 0,
                })
            except Exception:
                pass

    def observe_transition(self, trace: object, observation: np.ndarray, reward: float, cost: float, done: bool = False, **kwargs: Any):
        expert = self._expert_from_trace(trace)
        if expert is None:
            raise RuntimeError("khong tim thay chuyen gia so huu trace")
        learning_enabled = bool(getattr(self, "_learning_enabled", True))
        five_mode = str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) == "nam_tac_nhan_dung_chung"
        if five_mode and learning_enabled:
            audit = dict(getattr(trace, "context_audit", {}) or {})
            details = dict(getattr(trace, "action_details", {}) or {})
            agent_index = int(details.get("five_agent_index", audit.get("five_agent_index", -1)))
            if not 0 <= agent_index < 5:
                role = str(audit.get("decision_role", "") or "")
                if role.startswith("TAC_NHAN_"):
                    try:
                        agent_index = int(role.rsplit("_", 1)[-1]) - 1
                    except ValueError:
                        agent_index = -1
            if not 0 <= agent_index < 5:
                raise RuntimeError("trace nam tac nhan thieu five_agent_index hop le")
            pending = getattr(self, "_five_runtime_feedback", None)
            if not isinstance(pending, dict):
                pending = {}
                self._five_runtime_feedback = pending
            pending[agent_index] = (trace, float(reward) - float(cost), bool(done))
            result = {
                "five_feedback_buffered": 1.0,
                "five_feedback_count": float(len(pending)),
                "five_agent_index": float(agent_index),
                "reward": float(reward),
                "cost": float(cost),
            }
            if len(pending) == 5:
                ordered = [pending[i] for i in range(5)]
                hook = getattr(expert.policy, "observe_five_transition", None)
                if not callable(hook):
                    raise RuntimeError("bo nao chung khong ho tro observe_five_transition")
                five_result = hook(
                    [item[0] for item in ordered],
                    np.asarray([item[1] for item in ordered], dtype=np.float32),
                    dones=np.asarray([item[2] for item in ordered], dtype=bool),
                )
                pending.clear()
                result.update({str(k): float(v) for k, v in dict(five_result).items() if np.isscalar(v)})
                result["five_feedback_batch_submitted"] = 1.0
            return result
        if learning_enabled:
            try:
                result = expert.policy.observe_transition(trace, observation, reward, cost, done, **kwargs)
            except TypeError:
                result = expert.policy.observe_transition(trace, observation, reward, cost, **kwargs)
        else:
            # Do not call arbitrary owner policies in frozen mode.  A legacy
            # owner can still mutate its own evidence inside
            # ``observe_transition`` even when its Direct-RNN optimizer is
            # disabled.  MT5 feedback is emitted below for observability, but
            # no policy/system learner receives it.
            result = {
                "frozen_inference": 1.0,
                "learning_skipped": 1.0,
                "reward": float(reward),
                "cost": float(cost),
            }
        execution_details = dict(getattr(trace, "execution_details", {}) or {})
        if (
            str(getattr(trace, "trace_id", "") or "")
            and not bool(execution_details.get("filled", False))
            and (bool(execution_details.get("rejected", False)) or bool(execution_details.get("cancelled", False)))
        ):
            self._selector_pending_entry_traces.pop(str(getattr(trace, "trace_id", "") or ""), None)
        full_rl_mode = getattr(expert.policy, "_direct_rnn_full_rl_mode_enabled", None)
        full_rl_contract = bool(full_rl_mode()) if callable(full_rl_mode) else False
        if learning_enabled and not full_rl_contract:
            expert.ghi_ket_qua(
                reward=float(reward),
                cost=float(cost),
                trace_id=str(getattr(trace, "trace_id", "")),
                source="transition_feedback",
                lifecycle_confirmed=False,
            )
        # MT5's close path historically stopped after ``observe_transition``.
        # The owner policy therefore learned a local recovery signal, while
        # the facade-level selector and contextual-expectancy ledgers never
        # received the same authoritative lifecycle receipt.  When the trace
        # carries a verified close, settle it immediately through the normal
        # owner routing path.  This keeps selector/expectancy credit causal and
        # idempotent: a later explicit settlement sees the consumed trace and
        # cannot apply the same lifecycle twice.
        lifecycle_confirmed = bool(
            execution_details.get("closed", False)
            and (
                execution_details.get("action_feedback_eligible", False)
                or execution_details.get("memory_eligible", False)
                or execution_details.get("outcome_eligible", False)
            )
        )
        lifecycle_credit = 0
        lifecycle = kwargs.get("lifecycle")
        full_rl_canonical_only = full_rl_contract
        # A bridge ACK can say "closed" before the durable ledger has the
        # immutable fill geometry and final-close proof. Full-RL must not
        # consume that provisional receipt: doing so made geometry credit
        # permanently fail as ``final_close_unconfirmed`` and idempotency then
        # blocked the later canonical lifecycle. Legacy policies retain their
        # historic immediate route for compatibility.
        defer_to_canonical_ledger = bool(
            lifecycle_confirmed
            and full_rl_canonical_only
            and not bool(getattr(lifecycle, "final_close_confirmed", False))
        )
        if lifecycle_confirmed and not defer_to_canonical_ledger:
            apply_lifecycle = getattr(self, "apply_execution_lifecycle_credit", None)
            if callable(apply_lifecycle):
                lifecycle_kwargs: dict[str, object] = {
                    "close_reason": str(execution_details.get("close_reason", "") or ""),
                    "duration_bars": execution_details.get(
                        "duration_bars", execution_details.get("age_bars")
                    ),
                }
                if lifecycle is not None:
                    lifecycle_kwargs["lifecycle"] = lifecycle
                try:
                    lifecycle_credit = int(apply_lifecycle(
                        str(getattr(trace, "trace_id", "")),
                        float(reward),
                        **lifecycle_kwargs,
                    ) or 0)
                except TypeError:
                    # Preserve compatibility with lightweight/legacy policies
                    # whose lifecycle hook accepts only the core arguments.
                    try:
                        lifecycle_credit = int(apply_lifecycle(
                            str(getattr(trace, "trace_id", "")),
                            float(reward),
                            close_reason=lifecycle_kwargs["close_reason"],
                            duration_bars=lifecycle_kwargs["duration_bars"],
                        ) or 0)
                    except Exception:
                        pass
                except Exception:
                    # Lifecycle observability/credit must not break the bridge
                    # action path; the durable ledger remains authoritative for
                    # a later retry/settlement.
                    pass
        writer = self._writer_for_expert(expert)
        if writer is not None:
            try:
                writer.feedback(expert.policy, trace, next_obs=observation, reward=float(reward), cost=float(cost), done=bool(done), signal=kwargs.get("signal") if isinstance(kwargs.get("signal"), Mapping) else None)
            except Exception:
                pass
        if isinstance(result, Mapping):
            result = dict(result)
            result["lifecycle_credit"] = int(lifecycle_credit)
            result["lifecycle_credit_routed"] = bool(lifecycle_confirmed and not defer_to_canonical_ledger)
            result["lifecycle_credit_deferred_to_canonical_ledger"] = bool(defer_to_canonical_ledger)
        return result

    def observe_five_economic_transition(
        self,
        traces: Iterable[object],
        *,
        agent_mark_to_market_r: np.ndarray,
        transaction_cost_r: np.ndarray,
        agent_risk_ratio: np.ndarray,
        conflict_cost_r: np.ndarray,
        team_equity_r: float,
        team_drawdown: float,
        policy_violation_penalty_r: np.ndarray | None = None,
        agent_closed_trade_count: np.ndarray | None = None,
        agent_winning_trade_count: np.ndarray | None = None,
        agent_expectancy_shortfall_r: np.ndarray | None = None,
        agent_tail_loss_cost_r: np.ndarray | None = None,
        agent_realized_trade_r: np.ndarray | None = None,
        terminal: bool = False,
        dones: np.ndarray | None = None,
        elapsed_seconds: float = 0.0,
        step_seconds: float = 60.0,
        training_progress: float | None = None,
    ) -> dict[str, float]:
        """Route one atomic five-agent economic transition to the shared PPO writer.

        The bridge must call this batch API instead of five independent
        ``observe_transition`` calls.  That keeps marked PnL, costs, risk,
        conflict and episode boundaries on one coherent 5 x T timeline.
        """
        rows = tuple(traces)
        if len(rows) != 5:
            raise ValueError("can dung dung 5 trace cho economic transition")
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) != "nam_tac_nhan_dung_chung":
            raise RuntimeError("he thong chua o che do nam tac nhan dung chung")
        if not bool(getattr(self, "_learning_enabled", True)):
            return {"frozen_inference": 1.0, "learning_skipped": 1.0}
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            raise RuntimeError("khong tim thay bo nao chung")
        policy = expert.policy
        if training_progress is not None:
            set_progress = getattr(policy, "set_five_training_progress", None)
            if callable(set_progress):
                set_progress(float(training_progress))
        hook = getattr(policy, "observe_five_economic_transition", None)
        if not callable(hook):
            raise RuntimeError("bo nao chung khong ho tro economic reward nam tac nhan")
        return dict(hook(
            rows,
            agent_mark_to_market_r=np.asarray(agent_mark_to_market_r, dtype=np.float32),
            transaction_cost_r=np.asarray(transaction_cost_r, dtype=np.float32),
            agent_risk_ratio=np.asarray(agent_risk_ratio, dtype=np.float32),
            conflict_cost_r=np.asarray(conflict_cost_r, dtype=np.float32),
            policy_violation_penalty_r=(
                None
                if policy_violation_penalty_r is None
                else np.asarray(policy_violation_penalty_r, dtype=np.float32)
            ),
            team_equity_r=float(team_equity_r),
            team_drawdown=float(team_drawdown),
            agent_closed_trade_count=agent_closed_trade_count,
            agent_winning_trade_count=agent_winning_trade_count,
            agent_expectancy_shortfall_r=agent_expectancy_shortfall_r,
            agent_tail_loss_cost_r=agent_tail_loss_cost_r,
            agent_realized_trade_r=agent_realized_trade_r,
            terminal=bool(terminal),
            dones=dones,
            elapsed_seconds=float(elapsed_seconds),
            step_seconds=float(step_seconds),
        ))

    def observe_five_pattern_transition(
        self,
        traces: Iterable[object],
        pattern_rewards: np.ndarray,
        *,
        auxiliary_targets: Mapping[str, object] | None = None,
    ) -> dict[str, float]:
        """Route a mature dataset/MT5 pattern label to the shared PPO writer."""
        if not bool(getattr(self, "_learning_enabled", True)):
            return {"frozen_inference": 1.0, "learning_skipped": 1.0}
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            raise RuntimeError("khong tim thay bo nao chung")
        hook = getattr(expert.policy, "observe_five_pattern_transition", None)
        if not callable(hook):
            raise RuntimeError("bo nao chung khong ho tro pattern reward")
        return dict(hook(
            tuple(traces),
            np.asarray(pattern_rewards, dtype=np.float32),
            auxiliary_targets=auxiliary_targets,
        ))

    def register_five_pattern_anchor(
        self,
        traces: Iterable[object],
        sequence: int,
    ) -> dict[str, object]:
        """Route one Five packet to the shared owner's delayed-label cache."""
        rows = tuple(traces)
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) != "nam_tac_nhan_dung_chung":
            return {"registered": False, "reason": "not_five_mode", "sequence": int(sequence)}
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return {"registered": False, "reason": "owner_missing", "sequence": int(sequence)}
        for trace in rows:
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if trace_id:
                self._trace_to_expert[trace_id] = str(expert.ma_chuyen_gia)
        hook = getattr(expert.policy, "register_five_pattern_anchor", None)
        if not callable(hook):
            return {"registered": False, "reason": "hook_missing", "sequence": int(sequence)}
        try:
            result = hook(rows, int(sequence))
        except Exception as exc:
            return {"registered": False, "reason": type(exc).__name__, "sequence": int(sequence)}
        return dict(result) if isinstance(result, Mapping) else {
            "registered": bool(result), "sequence": int(sequence)
        }

    def bind_five_pattern_anchor_receipts(
        self,
        traces: Iterable[object],
        sequence: int,
    ) -> dict[str, object]:
        """Attach the verified MT5 batch receipt before delayed actor credit."""
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        hook = getattr(getattr(expert, "policy", None), "bind_five_pattern_anchor_receipts", None)
        if not callable(hook):
            return {"bound": False, "reason": "hook_missing", "sequence": int(sequence)}
        try:
            result = hook(tuple(traces), int(sequence))
        except Exception as exc:
            return {"bound": False, "reason": type(exc).__name__, "sequence": int(sequence)}
        return dict(result) if isinstance(result, Mapping) else {
            "bound": bool(result), "sequence": int(sequence)
        }

    def register_five_pending_order_packet(
        self,
        traces: Iterable[object],
    ) -> dict[str, object]:
        """Cache receipt-enriched pending-entry traces in the shared owner."""
        rows = tuple(traces)
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return {"registered": 0, "reason": "owner_missing"}
        for trace in rows:
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if trace_id:
                self._trace_to_expert[trace_id] = str(expert.ma_chuyen_gia)
        hook = getattr(expert.policy, "register_five_pending_order_packet", None)
        if not callable(hook):
            return {"registered": 0, "reason": "hook_missing"}
        result = hook(rows)
        return dict(result) if isinstance(result, Mapping) else {
            "registered": int(bool(result))
        }

    def apply_five_unfilled_pending_timeout_penalty(
        self,
        trace_id: str,
        penalty_r: float,
    ) -> dict[str, object]:
        """Route a five-bar timeout cost to the original LIMIT/STOP trace."""
        trace_id = str(trace_id or "")
        ma = self._trace_to_expert.get(trace_id, "")
        expert = next((
            item for item in self.registry.chuyen_gia_theo_context.values()
            if item.ma_chuyen_gia == ma
        ), None)
        if expert is None:
            return {"accepted": False, "updated": 0, "reason": "owner_missing"}
        hook = getattr(expert.policy, "apply_five_unfilled_pending_timeout_penalty", None)
        if not callable(hook):
            return {"accepted": False, "updated": 0, "reason": "hook_missing"}
        result = hook(trace_id, float(penalty_r))
        return dict(result) if isinstance(result, Mapping) else {
            "accepted": bool(result), "updated": int(bool(result))
        }

    def register_five_entry_lifecycle_packet(
        self,
        traces: Iterable[object],
    ) -> dict[str, object]:
        """Cache receipt-confirmed entry traces in their shared owner."""
        rows = tuple(traces)
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return {"registered": 0, "reason": "owner_missing"}
        for trace in rows:
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if trace_id:
                self._trace_to_expert[trace_id] = str(expert.ma_chuyen_gia)
        hook = getattr(expert.policy, "register_five_entry_lifecycle_packet", None)
        if not callable(hook):
            return {"registered": 0, "reason": "hook_missing"}
        result = hook(rows)
        return dict(result) if isinstance(result, Mapping) else {
            "registered": int(bool(result))
        }

    def get_five_entry_lifecycle_packet(self, trace_id: str) -> tuple[object, ...]:
        """Expose immutable owner traces to the candidate-only MT5 teacher."""
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return ()
        hook = getattr(expert.policy, "get_five_entry_lifecycle_packet", None)
        return tuple(hook(trace_id)) if callable(hook) else ()

    def pop_five_entry_lifecycle_packet(self, trace_id: str) -> tuple[object, ...]:
        """Consume one entry packet only after its causal teacher case is queued."""
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return ()
        hook = getattr(expert.policy, "pop_five_entry_lifecycle_packet", None)
        return tuple(hook(trace_id)) if callable(hook) else ()

    def register_five_action_packet(self, traces: Iterable[object]) -> dict[str, object]:
        """Cache every exact Five command packet for terminal action grading."""
        rows = tuple(traces)
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return {"registered": 0, "reason": "owner_missing"}
        for trace in rows:
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if trace_id:
                self._trace_to_expert[trace_id] = str(expert.ma_chuyen_gia)
        hook = getattr(expert.policy, "register_five_action_packet", None)
        if not callable(hook):
            return {"registered": 0, "reason": "hook_missing"}
        result = hook(rows)
        return dict(result) if isinstance(result, Mapping) else {"registered": int(bool(result))}

    def get_five_action_packet(self, trace_id: str) -> tuple[object, ...]:
        """Expose one exact command packet to the candidate-only Codex grader."""
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return ()
        hook = getattr(expert.policy, "get_five_action_packet", None)
        return tuple(hook(trace_id)) if callable(hook) else ()

    def pop_five_action_packet(self, trace_id: str) -> tuple[object, ...]:
        """Consume one exact command packet after case submission."""
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return ()
        hook = getattr(expert.policy, "pop_five_action_packet", None)
        return tuple(hook(trace_id)) if callable(hook) else ()

    def get_five_rlhf_reference_policy_sha256(self) -> str:
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            return ""
        return str(getattr(expert.policy, "five_rlhf_reference_policy_sha256", "") or "")

    def five_failure_replay_owner_policy(self):
        """Return the shared Five owner used by MT5 failure replay scheduling."""
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        return getattr(expert, "policy", None) if expert is not None else None

    def audit_five_failure_replay_cases(
        self, case_ids: Sequence[str] | None = None,
    ) -> dict[str, object]:
        owner = self.five_failure_replay_owner_policy()
        hook = getattr(owner, "audit_five_failure_replay_cases", None)
        if not callable(hook):
            return {"audited": 0, "probabilities": {}, "bank": {}}
        result = hook(case_ids)
        return dict(result) if isinstance(result, Mapping) else {"audited": 0}

    def apply_five_entry_lifecycle_reward(
        self,
        trace_id: str,
        **payload: object,
    ) -> dict[str, object]:
        """Route finalized lifecycle shaping to the immutable entry trace."""
        trace_id = str(trace_id or "")
        ma = self._trace_to_expert.get(trace_id, "")
        expert = next((
            item for item in self.registry.chuyen_gia_theo_context.values()
            if item.ma_chuyen_gia == ma
        ), None)
        if expert is None:
            return {"accepted": False, "updated": 0, "reason": "owner_missing"}
        hook = getattr(expert.policy, "apply_five_entry_lifecycle_reward", None)
        if not callable(hook):
            return {"accepted": False, "updated": 0, "reason": "hook_missing"}
        result = hook(trace_id, **{str(key): value for key, value in payload.items()})
        return dict(result) if isinstance(result, Mapping) else {
            "accepted": bool(result), "updated": int(bool(result))
        }

    def flush_five_lifecycle_reward_queue(self) -> dict[str, object]:
        """Drain settled Five lifecycle samples left below the normal batch size."""
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        hook = getattr(getattr(expert, "policy", None), "flush_five_lifecycle_reward_queue", None)
        if not callable(hook):
            return {"flushed_batches": 0, "flushed_steps": 0, "pending": 0, "reason": "hook_missing"}
        result = hook()
        return dict(result) if isinstance(result, Mapping) else {
            "flushed_batches": int(bool(result)), "flushed_steps": 0, "pending": 0
        }

    def apply_five_pattern_path_credit_detailed(
        self,
        trace_id: str,
        record: object,
    ) -> dict[str, object]:
        """Consume an MT5 Five label without falling back to Direct-RNN."""
        trace_id = str(trace_id or "")
        ma = self._trace_to_expert.get(trace_id, "")
        expert = next(
            (item for item in self.registry.chuyen_gia_theo_context.values()
             if item.ma_chuyen_gia == ma),
            None,
        )
        if expert is None:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": "owner_missing", "trace_id": trace_id,
            }
        hook = getattr(expert.policy, "apply_five_pattern_path_credit_detailed", None)
        if not callable(hook):
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": False, "reason": "five_pattern_hook_missing",
                "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
            }
        try:
            result = hook(trace_id, record)
        except Exception as exc:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": type(exc).__name__,
                "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
            }
        payload = dict(result) if isinstance(result, Mapping) else {
            "accepted": True, "queued": True, "updates": int(result or 0),
            "retryable": False, "reason": "five_pattern_hook_completed",
        }
        payload.setdefault("trace_id", trace_id)
        payload.setdefault("owner", str(expert.ma_chuyen_gia))
        return payload

    def apply_five_entry_preference_credit_detailed(
        self,
        trace_id: str,
        record: object,
    ) -> dict[str, object]:
        """Route Five selector-only delayed credit to the immutable owner."""
        trace_id = str(trace_id or "")
        ma = self._trace_to_expert.get(trace_id, "")
        expert = next(
            (item for item in self.registry.chuyen_gia_theo_context.values()
             if item.ma_chuyen_gia == ma),
            None,
        )
        if expert is None:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": "owner_missing", "trace_id": trace_id,
            }
        hook = getattr(expert.policy, "apply_five_entry_preference_credit_detailed", None)
        if not callable(hook):
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": False, "reason": "five_selector_hook_missing",
                "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
            }
        try:
            result = hook(trace_id, record)
        except Exception as exc:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": type(exc).__name__,
                "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
            }
        payload = dict(result) if isinstance(result, Mapping) else {
            "accepted": True, "queued": True, "updates": int(result or 0),
            "retryable": False, "reason": "five_selector_hook_completed",
        }
        payload.setdefault("trace_id", trace_id)
        payload.setdefault("owner", str(expert.ma_chuyen_gia))
        return payload

    def reset_five_economic_reward(self, *, clear_rollout: bool = True) -> None:
        """Reset episode-local reward/memory and optionally discard PPO rollout.

        Dataset samples are independent recurrent episodes, but their terminal
        done masks allow one 64-step PPO batch to span sample boundaries. The
        trainer therefore preserves an incomplete rollout between samples.
        """
        if str(getattr(self.cau_hinh, "che_do_quyet_dinh", "")) != "nam_tac_nhan_dung_chung":
            return
        expert = self.registry.chuyen_gia_theo_context.get("CTX_CHUNG")
        if expert is None:
            raise RuntimeError("khong tim thay bo nao chung")
        reset_shaper = getattr(expert.policy, "reset_five_reward_shaper", None)
        if callable(reset_shaper):
            reset_shaper()
        reset_memories = getattr(expert.policy, "reset_five_agent_memories", None)
        if callable(reset_memories):
            reset_memories(clear_rollout=bool(clear_rollout))

    def register_execution_intent_trace(self, trace: object) -> dict[str, object]:
        """Freeze the exact bridge-bound entry trace at the owner policy.

        ``act`` caches a proposal before the MT5 bridge enriches it with the
        emitted action, order type and risk envelope.  Lifecycle credit must
        never use that stale proposal.  This method is deliberately
        observability/routing-only: it neither learns nor makes an entry legal.
        """
        trace_id = str(getattr(trace, "trace_id", "") or "")
        expert = self._expert_from_trace(trace)
        if not trace_id or expert is None:
            return {"registered": False, "reason": "owner_missing", "trace_id": trace_id}
        details = dict(getattr(trace, "action_details", {}) or {})
        action = int(details.get("emitted_action", details.get("bridge_action", self._proposal_action_index(trace))) or 0)
        if action not in CHI_SO_ENTRY_ACTIONS:
            return {"registered": False, "reason": "not_entry", "trace_id": trace_id}
        # Be defensive for resumed facades whose ephemeral map was trimmed;
        # the decision owner is immutable provenance on the selected trace.
        self._trace_to_expert[trace_id] = expert.ma_chuyen_gia
        trace = _gan_action_details(trace, {
            "execution_intent_registered": True,
            "execution_intent_owner": str(expert.ma_chuyen_gia),
        })
        cache_trace = getattr(expert.policy, "_cache_execution_trace", None)
        if callable(cache_trace):
            cache_trace(trace)
            cache_hit = True
        else:
            cache = getattr(expert.policy, "_execution_trace_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                setattr(expert.policy, "_execution_trace_cache", cache)
            cache[trace_id] = trace
            cache_hit = True
        return {
            "registered": True,
            "trace_id": trace_id,
            "owner": str(expert.ma_chuyen_gia),
            "cache_hit": cache_hit,
        }

    def register_entry_preference_trace(self, trace: object) -> dict[str, object]:
        """Route a flat bridge decision to the owner preference cache only.

        Unlike execution intent registration, HOLD is deliberately retained:
        the delayed counterfactual label must be able to prove that HOLD was
        better than an entry without fabricating an order lifecycle.
        """
        trace_id = str(getattr(trace, "trace_id", "") or "")
        expert = self._expert_from_trace(trace)
        if not trace_id or expert is None:
            return {"registered": False, "reason": "owner_missing", "trace_id": trace_id}
        self._trace_to_expert[trace_id] = expert.ma_chuyen_gia
        hook = getattr(expert.policy, "register_entry_preference_trace", None)
        if not callable(hook):
            return {"registered": False, "reason": "hook_missing", "trace_id": trace_id}
        try:
            result = hook(trace)
        except Exception as exc:
            return {"registered": False, "reason": type(exc).__name__, "trace_id": trace_id}
        payload = dict(result) if isinstance(result, Mapping) else {"registered": bool(result)}
        payload.setdefault("trace_id", trace_id)
        payload.setdefault("owner", str(expert.ma_chuyen_gia))
        return payload

    def apply_entry_preference_credit_detailed(
        self, trace_id: str, record: object
    ) -> dict[str, object]:
        """Route a mature label and preserve the owner's explicit queue ACK."""
        trace_id = str(trace_id or "")
        ma = self._trace_to_expert.get(trace_id, "")
        expert = next(
            (item for item in self.registry.chuyen_gia_theo_context.values()
             if item.ma_chuyen_gia == ma),
            None,
        )
        if expert is None:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": "owner_missing",
                "trace_id": trace_id,
            }
        hook = getattr(expert.policy, "apply_entry_preference_credit_detailed", None)
        legacy_hook = getattr(expert.policy, "apply_entry_preference_credit", None)
        if not callable(hook):
            if not callable(legacy_hook):
                return {
                    "accepted": False, "queued": False, "updates": 0,
                    "retryable": False, "reason": "hook_missing",
                    "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
                }
            try:
                updates = int(legacy_hook(trace_id, record) or 0)
            except Exception as exc:
                return {
                    "accepted": False, "queued": False, "updates": 0,
                    "retryable": True, "reason": type(exc).__name__,
                    "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
                }
            return {
                "accepted": True, "queued": True, "updates": updates,
                "retryable": False, "reason": "legacy_hook_completed",
                "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
            }
        try:
            result = hook(trace_id, record)
        except Exception as exc:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": type(exc).__name__,
                "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia),
            }
        payload = dict(result) if isinstance(result, Mapping) else {
            "accepted": True, "queued": True, "updates": int(result or 0),
            "retryable": False, "reason": "hook_completed",
        }
        payload.setdefault("trace_id", trace_id)
        payload.setdefault("owner", str(expert.ma_chuyen_gia))
        return payload

    def apply_entry_preference_credit(self, trace_id: str, record: object) -> int:
        """Compatibility wrapper returning only the optimizer update count."""
        result = self.apply_entry_preference_credit_detailed(trace_id, record)
        return int(result.get("updates", 0) or 0)

    def apply_rlhf_preference_credit_detailed(
        self, pair: object, *, trace_id: str = ""
    ) -> dict[str, object]:
        """Route one causal chosen/rejected pair to its trace owner."""
        pair_id = str(getattr(pair, "pair_id", "") or "")
        candidate_trace_ids = tuple(
            item for item in (
                str(trace_id or ""),
                str(getattr(pair, "chosen_trace_id", "") or ""),
                str(getattr(pair, "rejected_trace_id", "") or ""),
            ) if item
        )
        ma = next((self._trace_to_expert.get(item, "") for item in candidate_trace_ids if self._trace_to_expert.get(item, "")), "")
        expert = next(
            (item for item in self.registry.chuyen_gia_theo_context.values()
             if item.ma_chuyen_gia == ma),
            None,
        )
        if expert is None:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": "owner_missing", "pair_id": pair_id,
            }
        hook = getattr(expert.policy, "apply_rlhf_preference_credit_detailed", None)
        if not callable(hook):
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": False, "reason": "hook_missing", "pair_id": pair_id,
                "owner": str(expert.ma_chuyen_gia),
            }
        try:
            result = hook(pair, trace_id=trace_id)
        except Exception as exc:
            return {
                "accepted": False, "queued": False, "updates": 0,
                "retryable": True, "reason": type(exc).__name__, "pair_id": pair_id,
                "owner": str(expert.ma_chuyen_gia),
            }
        payload = dict(result) if isinstance(result, Mapping) else {
            "accepted": True, "queued": True, "updates": int(result or 0),
            "retryable": False, "reason": "hook_completed",
        }
        payload.setdefault("pair_id", pair_id)
        payload.setdefault("owner", str(expert.ma_chuyen_gia))
        return payload

    def bind_execution_receipt(self, trace_id: str, receipt: Mapping[str, object]) -> dict[str, object]:
        """Replace the cached entry trace with the latest verified MT5 receipt.

        Pending/placed receipts remain outside replay memory.  They are kept
        only in the in-process lifecycle cache and the separate observability
        stream so a later fill/close can be attributed to the exact emitted
        command without fabricating training examples.
        """
        trace_id = str(trace_id or "")
        ma = self._trace_to_expert.get(trace_id, "")
        expert = next((item for item in self.registry.chuyen_gia_theo_context.values() if item.ma_chuyen_gia == ma), None)
        if not trace_id or expert is None:
            return {"bound": False, "reason": "owner_missing", "trace_id": trace_id}
        cache = getattr(expert.policy, "_execution_trace_cache", None)
        cached_trace = cache.get(trace_id) if isinstance(cache, Mapping) else None
        if cached_trace is None:
            return {"bound": False, "reason": "cache_miss", "trace_id": trace_id, "owner": str(expert.ma_chuyen_gia)}
        existing = dict(getattr(cached_trace, "execution_details", {}) or {})
        existing.update({str(key): value for key, value in dict(receipt).items()})
        updated_trace = cached_trace
        if is_dataclass(updated_trace):
            try:
                updated_trace = replace(updated_trace, execution_details=existing)
            except (TypeError, ValueError):
                pass
        try:
            setattr(updated_trace, "execution_details", existing)
        except Exception:
            pass
        updated_trace = _gan_action_details(updated_trace, {
            "execution_receipt_bound": True,
            "execution_receipt_stage": str(existing.get("receipt_stage", "bridge")),
        })
        cache_trace = getattr(expert.policy, "_cache_execution_trace", None)
        if callable(cache_trace):
            cache_trace(updated_trace)
        elif isinstance(cache, dict):
            cache[trace_id] = updated_trace
        return {
            "bound": True,
            "trace_id": trace_id,
            "owner": str(expert.ma_chuyen_gia),
            "filled": bool(existing.get("filled", False)),
            "closed": bool(existing.get("closed", False)),
        }

    def record_five_live_owner_outcome(
        self,
        *,
        agent_index: int,
        action: int,
        net_r: float,
        trace_id: str = "",
    ) -> bool:
        """Forward run-local Five exploration evidence to the trace owner."""
        ma = self._trace_to_expert.get(str(trace_id or ""), "")
        expert = next(
            (
                item for item in self.registry.chuyen_gia_theo_context.values()
                if item.ma_chuyen_gia == ma
            ),
            None,
        )
        if expert is None:
            return False
        hook = getattr(expert.policy, "record_five_live_owner_outcome", None)
        if not callable(hook):
            return False
        return bool(hook(
            agent_index=int(agent_index),
            action=int(action),
            net_r=float(net_r),
            trace_id=str(trace_id or ""),
        ))

    def apply_execution_lifecycle_credit(self, trace_id: str, *args: Any, **kwargs: Any) -> int:
        """Chuyển lifecycle credit về đúng policy đã tạo trace, không broadcast."""
        ma = self._trace_to_expert.get(str(trace_id), "")
        expert = next((item for item in self.registry.chuyen_gia_theo_context.values() if item.ma_chuyen_gia == ma), None)
        if expert is None:
            return 0
        hook = getattr(expert.policy, "apply_execution_lifecycle_credit", None)
        if not callable(hook):
            return 0
        # Policy gốc có thể pop trace khỏi execution cache khi settle lifecycle.
        # Lấy bản tham chiếu trước để controller context-owner vẫn nhìn được
        # đúng trace đã tạo lệnh và có thể đánh giá/tạo candidate sau outcome thật.
        trace = None
        cache = getattr(expert.policy, "_execution_trace_cache", None)
        if isinstance(cache, Mapping):
            trace = cache.get(str(trace_id))
        settlement_kwargs = dict(kwargs)
        # Canonical lifecycle settlement must receive the immutable entry trace
        # and the exact emitted entry action.  Inferring either value after an
        # older wrapper has popped/replaced cache state was the source of
        # action-local credit drifting into an unrelated scope.
        if trace is not None:
            settlement_kwargs.setdefault("trace", trace)
            details = dict(getattr(trace, "action_details", {}) or {})
            for key in ("emitted_action", "wire_action", "bridge_action", "policy_action", "semantic_action"):
                try:
                    action = int(details.get(key, 0) or 0)
                except (TypeError, ValueError):
                    action = 0
                if action in CHI_SO_ENTRY_ACTIONS:
                    settlement_kwargs.setdefault("override_action", action)
                    break
        learning_enabled = bool(getattr(self, "_learning_enabled", True)) and bool(
            getattr(expert.policy, "_learning_enabled", True)
        )
        full_rl_mode = getattr(expert.policy, "_direct_rnn_full_rl_mode_enabled", None)
        full_rl_contract = bool(full_rl_mode()) if callable(full_rl_mode) else False
        if learning_enabled:
            result = hook(trace_id, *args, **settlement_kwargs)
        else:
            # Retire only transient lifecycle state.  The separate lifecycle
            # event below still records the realised R with ``credited=0``;
            # no optimizer, selector, TimingNet, expert fitness or evidence
            # ledger is allowed to mutate during frozen/OOS evaluation.
            if isinstance(cache, dict):
                cache.pop(str(trace_id), None)
            self._selector_pending_entry_traces.pop(str(trace_id), None)
            self._selector_pending_entry_ages.pop(str(trace_id), None)
            result = 0
        # TimingNet is the shared context signal producer.  Credit it once per
        # realised entry lifecycle, even when the selector chose a role
        # specialist whose local policy is different from the context owner.
        timing_network = getattr(self.nhan_dien, "timing_network", None)
        timing_trace = self._trace_to_timing_network_trace.pop(str(trace_id), {})
        if (
            result > 0
            and isinstance(timing_network, TimingNetwork)
            and bool(getattr(expert.policy, "_learning_enabled", True))
            and self._proposal_action_index(trace) in CHI_SO_ENTRY_ACTIONS
        ):
            if timing_trace:
                lifecycle_scale = float(getattr(expert.policy, "timing_network_lifecycle_scale", 0.0) or 0.0)
                timing_scale = lifecycle_scale if lifecycle_scale > 0.0 else float(
                    getattr(expert.policy, "_mt5_online_update_scale", 1.0)
                )
                timing_network.update_lifecycle_credit(
                    timing_trace,
                    float(args[0]) if args else float(kwargs.get("realized_r", 0.0) or 0.0),
                    selected_action=self._proposal_action_index(trace),
                    duration_bars=kwargs.get("duration_bars"),
                    close_reason=str(kwargs.get("close_reason", "") or ""),
                    update_scale=float(np.clip(
                        timing_scale
                        * float(getattr(expert.policy, "timing_network_update_scale", 0.25) or 0.25),
                        0.0,
                        1.0,
                    )),
                )
        # V9: increment expert lifecycle count so facade audit sees it.
        if result > 0:
            realized_r = float(args[0]) if args else float(kwargs.get("realized_r", 0.0) or 0.0)
            if not full_rl_contract:
                expert.ghi_ket_qua(
                    reward=realized_r,
                    cost=0.0,
                    trace_id=str(trace_id),
                    source="execution_lifecycle",
                    lifecycle_confirmed=True,
                )
            self._credit_selector_outcome(
                str(trace_id),
                self._trace_to_selector_policy.get(str(trace_id), expert.ma_chuyen_gia),
                realized_r,
            )
        reward = float(args[0]) if args else float(kwargs.get("realized_r", 0.0) or 0.0)
        if trace is not None:
            writer = self._writer_for_expert(expert)
            if writer is not None:
                try:
                    writer.lifecycle(expert.policy, trace_id=str(trace_id), realized_r=float(reward), close_reason=str(kwargs.get("close_reason", "") or ""), duration_bars=kwargs.get("duration_bars"), credited=int(result or 0))
                except Exception:
                    pass
        return int(result or 0)

    def apply_execution_exit_credit(
        self,
        trace_id: str,
        *,
        final_realized_r: float,
        hold_to_horizon_r: float,
        position_ticket: str = "",
        exit_kind: str = "",
        horizon_bars: int | None = None,
    ) -> int:
        """Route delayed counterfactual credit to the exact exit controller.

        This never falls back to the entry owner: a manager close is judged by
        its improvement over holding, while the entry has already received its
        immutable final lifecycle R.
        """
        trace_id = str(trace_id or "")
        manager_id = self._trace_to_expert.get(trace_id, "")
        manager = next(
            (item for item in self.registry.chuyen_gia_theo_context.values() if item.ma_chuyen_gia == manager_id),
            None,
        )
        if manager is None:
            return 0
        if not bool(getattr(self, "_learning_enabled", True)) or not bool(
            getattr(manager.policy, "_learning_enabled", True)
        ):
            return 0
        full_rl_mode = getattr(manager.policy, "_direct_rnn_full_rl_mode_enabled", None)
        full_rl_contract = bool(full_rl_mode()) if callable(full_rl_mode) else False
        hook = getattr(manager.policy, "apply_execution_exit_credit", None)
        if not callable(hook):
            return 0
        credited = int(hook(
            trace_id,
            final_realized_r=float(final_realized_r),
            hold_to_horizon_r=float(hold_to_horizon_r),
            position_ticket=str(position_ticket or ""),
            exit_kind=str(exit_kind or ""),
            horizon_bars=horizon_bars,
        ) or 0)
        if not full_rl_contract:
            raw_benefit = float(final_realized_r) - float(hold_to_horizon_r)
            normalized_benefit = float(raw_benefit / (1.0 + abs(raw_benefit)))
            manager.ghi_ket_qua(
                reward=normalized_benefit,
                cost=0.0,
                trace_id=trace_id,
                source="execution_exit_counterfactual",
                lifecycle_confirmed=True,
            )
            self._credit_selector_outcome(
                trace_id,
                self._trace_to_selector_policy.get(trace_id, manager.ma_chuyen_gia),
                normalized_benefit,
            )
        return credited

    def contextual_expectancy_report(self) -> dict[str, object]:
        """Báo cáo local; gộp từ các expert con thành fitness system-level."""
        records: dict[str, object] = {}
        policy_values: list[float] = []
        all_values: list[float] = []
        total_policy_count = 0
        total_all_count = 0
        derived_evidence_count = 0
        derived_evidence_origins: dict[str, int] = {}
        negative_critical = 0
        positive_eligible = 0
        record_count = 0
        # Conditional entry evidence is intentionally high-cardinality.  A
        # system report that exposes only eligible buckets makes real, sparse
        # lifecycle credit look like "0 samples".  Aggregate the raw support
        # separately, without giving it any inference or fitness authority.
        conditioned_sample_count = 0
        conditioned_eligible_record_count = 0
        conditioned_eligible_sample_count = 0
        conditioned_max_record_sample_count = 0
        conditioned_min_samples: int | None = None
        conditioned_samples_to_first_eligible: int | None = None
        for context_id, expert in self.registry.chuyen_gia_theo_context.items():
            report = getattr(expert.policy, "contextual_expectancy_report", None)
            expert_report = report() if callable(report) else {}
            records[context_id] = {
                "ma_chuyen_gia": expert.ma_chuyen_gia,
                "trang_thai": expert.trang_thai.value,
                "so_lifecycle_da_dong": int(expert.so_lifecycle_da_dong),
                "fitness_cuc_bo": float(expert.fitness_cuc_bo),
                "bao_cao_policy": expert_report,
            }
            # Aggregate counts from expert-level report
            total_policy_count += int(expert_report.get("policy_origin_lifecycle_count", 0) or 0)
            total_all_count += int(expert_report.get("all_origin_lifecycle_count", 0) or 0)
            negative_critical += int(expert_report.get("negative_critical_scope_count", 0) or 0)
            positive_eligible += int(expert_report.get("positive_eligible_scope_count", 0) or 0)
            record_count += int(expert_report.get("record_count", 0) or 0)
            conditioned = dict(expert_report.get("conditioned_entry_expectancy", {}) or {})
            if conditioned:
                conditioned_sample_count += int(conditioned.get("sample_count", 0) or 0)
                conditioned_eligible_record_count += int(
                    conditioned.get("eligible_record_count", 0) or 0
                )
                conditioned_eligible_sample_count += int(
                    conditioned.get("eligible_sample_count", 0) or 0
                )
                conditioned_max_record_sample_count = max(
                    conditioned_max_record_sample_count,
                    int(conditioned.get("max_record_sample_count", 0) or 0),
                )
                minimum = max(1, int(conditioned.get("min_samples", 1) or 1))
                conditioned_min_samples = (
                    minimum if conditioned_min_samples is None else min(conditioned_min_samples, minimum)
                )
                missing = max(0, int(conditioned.get("samples_to_first_eligible", minimum) or 0))
                conditioned_samples_to_first_eligible = (
                    missing
                    if conditioned_samples_to_first_eligible is None
                    else min(conditioned_samples_to_first_eligible, missing)
                )
            # Full-RL lifecycle evidence is deliberately audit-only and must
            # never enter decision memory.  Aggregate the same ledger that the
            # expert report selected; otherwise system-level fitness would
            # show 187 policy lifecycles but a misleading zero samples/LCB.
            uses_full_rl_audit = str(expert_report.get("ledger_source", "") or "") == "full_rl_audit"
            ledger = (
                getattr(expert.policy, "full_rl_expectancy_audit", None)
                if uses_full_rl_audit
                else getattr(getattr(expert.policy, "memory_supernet", None), "contextual_expectancy", None)
            )
            for record in dict(getattr(ledger, "records", {}) or {}).values():
                values = [float(value) for value in list(getattr(record, "values", ()) or ())]
                origins = list(getattr(record, "origins", ()) or ())
                for value, origin in zip(values, origins):
                    normalized_origin = str(origin or "policy").strip().lower() or "policy"
                    if normalized_origin in DERIVED_EVIDENCE_ORIGINS:
                        derived_evidence_count += 1
                        derived_evidence_origins[normalized_origin] = derived_evidence_origins.get(normalized_origin, 0) + 1
                        continue
                    all_values.append(value)
                    if normalized_origin in POLICY_ORIGINS:
                        policy_values.append(value)
        policy_ratio = float(total_policy_count / max(1, total_all_count)) if total_all_count else 0.0
        min_samples = 6
        for context_id, expert in self.registry.chuyen_gia_theo_context.items():
            expert_report = dict(records.get(context_id, {}).get("bao_cao_policy", {}) or {})
            ledger = (
                getattr(expert.policy, "full_rl_expectancy_audit", None)
                if str(expert_report.get("ledger_source", "") or "") == "full_rl_audit"
                else getattr(getattr(expert.policy, "memory_supernet", None), "contextual_expectancy", None)
            )
            if ledger is not None:
                min_samples = max(1, int(getattr(ledger, "min_samples", min_samples) or min_samples))
                break
        policy_expectancy = robust_expectancy(
            policy_values,
            min_samples=min_samples,
            policy_origin_ratio=policy_ratio,
        ).as_dict()
        all_expectancy = robust_expectancy(
            all_values,
            min_samples=min_samples,
            policy_origin_ratio=1.0,
        ).as_dict()
        return {
            "che_do": "mot_context_mot_chuyen_gia",
            "schema_version": int(EXPECTANCY_SCHEMA_VERSION),
            "record_count": int(record_count),
            "policy_origin_lifecycle_count": int(total_policy_count),
            "all_origin_lifecycle_count": int(total_all_count),
            "policy_origin_ratio": policy_ratio,
            "derived_evidence_count": int(derived_evidence_count),
            "derived_evidence_origins": derived_evidence_origins,
            "policy_expectancy": policy_expectancy,
            "all_expectancy": all_expectancy,
            "negative_critical_scope_count": int(negative_critical),
            "positive_eligible_scope_count": int(positive_eligible),
            "conditioned_entry_expectancy": {
                "sample_count": int(conditioned_sample_count),
                "eligible_record_count": int(conditioned_eligible_record_count),
                "eligible_sample_count": int(conditioned_eligible_sample_count),
                "max_record_sample_count": int(conditioned_max_record_sample_count),
                "min_samples": int(conditioned_min_samples or 0),
                "samples_to_first_eligible": int(
                    conditioned_samples_to_first_eligible
                    if conditioned_samples_to_first_eligible is not None else 0
                ),
                "telemetry_only": True,
            },
            "chuyen_gia": records,
        }

    def xet_thang_cap_chuyen_gia(
        self,
        ma_context: str,
        *,
        replay_pass: bool,
        oos_pass: bool,
        risk_pass: bool,
        lifecycle_count: int | None = None,
        ghi_chu: str = "",
    ) -> dict[str, object]:
        """Gate thăng cấp explicit cho context owner.

        State của một owner taxonomy chỉ được thăng cấp khi replay, OOS, risk
        và lifecycle thật đều đủ ngưỡng.
        """
        expert = self.registry.chuyen_gia_theo_context.get(str(ma_context))
        if expert is None:
            return {"promoted": False, "reason": "context_khong_ton_tai", "ma_context": str(ma_context)}
        lifecycle = int(expert.so_lifecycle_da_dong if lifecycle_count is None else lifecycle_count)
        gates = {
            "replay_pass": bool(replay_pass),
            "oos_pass": bool(oos_pass),
            "risk_pass": bool(risk_pass),
            "lifecycle_count": lifecycle,
        }
        missing = [name for name in ("replay_pass", "oos_pass", "risk_pass") if not gates[name]]
        if missing:
            return {
                "promoted": False,
                "reason": "gate_chua_dat:" + ",".join(missing),
                "ma_context": expert.ma_context,
                "ma_chuyen_gia": expert.ma_chuyen_gia,
                **gates,
            }
        old = expert.trang_thai
        target = old
        active_min = int(self.cau_hinh.so_lifecycle_toi_thieu_de_hoat_dong)
        mature_min = int(self.cau_hinh.so_lifecycle_toi_thieu_de_truong_thanh)
        if old in {TrangThaiChuyenGia.MAM, TrangThaiChuyenGia.QUAN_SAT_SONG_SONG}:
            if lifecycle < active_min:
                return {
                    "promoted": False,
                    "reason": f"chua_du_lifecycle:{lifecycle}<{active_min}",
                    "ma_context": expert.ma_context,
                    "ma_chuyen_gia": expert.ma_chuyen_gia,
                    **gates,
                }
            target = TrangThaiChuyenGia.THAM_DO_RUI_RO_THAP
        elif old == TrangThaiChuyenGia.THAM_DO_RUI_RO_THAP:
            if lifecycle < active_min:
                return {
                    "promoted": False,
                    "reason": f"chua_du_lifecycle:{lifecycle}<{active_min}",
                    "ma_context": expert.ma_context,
                    "ma_chuyen_gia": expert.ma_chuyen_gia,
                    **gates,
                }
            target = TrangThaiChuyenGia.HOAT_DONG_HAN_CHE
        elif old == TrangThaiChuyenGia.HOAT_DONG_HAN_CHE:
            if lifecycle < mature_min:
                return {
                    "promoted": False,
                    "reason": f"chua_du_lifecycle_truong_thanh:{lifecycle}<{mature_min}",
                    "ma_context": expert.ma_context,
                    "ma_chuyen_gia": expert.ma_chuyen_gia,
                    **gates,
                }
            target = TrangThaiChuyenGia.TRUONG_THANH
        elif old in {TrangThaiChuyenGia.TRUONG_THANH, TrangThaiChuyenGia.NGU_DONG, TrangThaiChuyenGia.LUU_TRU}:
            target = old
        expert.trang_thai = target
        expert.lich_su.append({
            "event": "thang_cap_chuyen_gia",
            "from": old.value,
            "to": target.value,
            "lifecycle_count": lifecycle,
            "replay_pass": bool(replay_pass),
            "oos_pass": bool(oos_pass),
            "risk_pass": bool(risk_pass),
            "ghi_chu": str(ghi_chu),
        })
        return {
            "promoted": target != old,
            "reason": "promoted" if target != old else "khong_can_doi_trang_thai",
            "ma_context": expert.ma_context,
            "ma_chuyen_gia": expert.ma_chuyen_gia,
            "from": old.value,
            "to": target.value,
            **gates,
        }

    def dat_trang_thai_chuyen_gia(
        self,
        ma_context: str,
        trang_thai: TrangThaiChuyenGia | str,
        *,
        ghi_chu: str = "",
        merged_into: str = "",
    ) -> dict[str, object]:
        expert = self.registry.chuyen_gia_theo_context.get(str(ma_context))
        if expert is None:
            return {"changed": False, "reason": "context_khong_ton_tai", "ma_context": str(ma_context)}
        target = trang_thai if isinstance(trang_thai, TrangThaiChuyenGia) else TrangThaiChuyenGia(str(trang_thai))
        old = expert.trang_thai
        expert.trang_thai = target
        event = {
            "event": "dat_trang_thai_chuyen_gia",
            "from": old.value,
            "to": target.value,
            "ghi_chu": str(ghi_chu),
        }
        if merged_into:
            event["merged_into"] = str(merged_into)
        expert.lich_su.append(event)
        return {
            "changed": target != old,
            "reason": "updated" if target != old else "khong_doi",
            "ma_context": expert.ma_context,
            "ma_chuyen_gia": expert.ma_chuyen_gia,
            "from": old.value,
            "to": target.value,
            "merged_into": str(merged_into),
        }

    def luu_tru_chuyen_gia(self, ma_context: str, *, ghi_chu: str = "") -> dict[str, object]:
        return self.dat_trang_thai_chuyen_gia(ma_context, TrangThaiChuyenGia.LUU_TRU, ghi_chu=ghi_chu)

    def ngu_dong_chuyen_gia(self, ma_context: str, *, ghi_chu: str = "") -> dict[str, object]:
        return self.dat_trang_thai_chuyen_gia(ma_context, TrangThaiChuyenGia.NGU_DONG, ghi_chu=ghi_chu)

    def hop_nhat_context_chuyen_gia(
        self,
        ma_context_nguon: str,
        ma_context_dich: str,
        *,
        oos_pass: bool,
        non_interference_pass: bool,
        ghi_chu: str = "",
    ) -> dict[str, object]:
        source = self.registry.chuyen_gia_theo_context.get(str(ma_context_nguon))
        target = self.registry.chuyen_gia_theo_context.get(str(ma_context_dich))
        if source is None:
            return {"merged": False, "reason": "context_nguon_khong_ton_tai", "ma_context": str(ma_context_nguon)}
        if target is None:
            return {"merged": False, "reason": "context_dich_khong_ton_tai", "ma_context": str(ma_context_dich)}
        if source.ma_context == target.ma_context:
            return {"merged": False, "reason": "khong_the_merge_chinh_no", "ma_context": source.ma_context}
        missing = []
        if not bool(oos_pass):
            missing.append("oos_pass")
        if not bool(non_interference_pass):
            missing.append("non_interference_pass")
        if missing:
            return {
                "merged": False,
                "reason": "gate_chua_dat:" + ",".join(missing),
                "ma_context": source.ma_context,
                "target_context": target.ma_context,
            }
        old = source.trang_thai
        source.trang_thai = TrangThaiChuyenGia.LUU_TRU
        source.lich_su.append({
            "event": "hop_nhat_context_chuyen_gia",
            "from": old.value,
            "to": TrangThaiChuyenGia.LUU_TRU.value,
            "target_context": target.ma_context,
            "target_expert": target.ma_chuyen_gia,
            "oos_pass": bool(oos_pass),
            "non_interference_pass": bool(non_interference_pass),
            "ghi_chu": str(ghi_chu),
        })
        target.lich_su.append({
            "event": "nhan_context_hop_nhat",
            "source_context": source.ma_context,
            "source_expert": source.ma_chuyen_gia,
            "ghi_chu": str(ghi_chu),
        })
        return {
            "merged": True,
            "reason": "source_luu_tru_target_giu_owner",
            "ma_context": source.ma_context,
            "ma_chuyen_gia": source.ma_chuyen_gia,
            "target_context": target.ma_context,
            "target_expert": target.ma_chuyen_gia,
            "from": old.value,
            "to": TrangThaiChuyenGia.LUU_TRU.value,
        }

    def structural_complexity(self) -> float:
        total = 0.0
        for expert in self.registry.chuyen_gia_theo_context.values():
            hook = getattr(expert.policy, "structural_complexity", None)
            total += float(hook()) if callable(hook) else float(len(getattr(expert.policy, "modules", []) or []))
        return total

    def set_live_observatory_trial(self, **kwargs: object) -> None:
        for expert in self.registry.chuyen_gia_theo_context.values():
            hook = getattr(expert.policy, "set_live_observatory_trial", None)
            if callable(hook):
                hook(**kwargs)

    def live_observatory_trial_finished(self, **kwargs: object) -> None:
        for expert in self.registry.chuyen_gia_theo_context.values():
            hook = getattr(expert.policy, "live_observatory_trial_finished", None)
            if callable(hook):
                hook(**kwargs)

    def observe_shadow_transition(self, trace: object, observation: np.ndarray, **kwargs: Any):
        expert = self._expert_from_trace(trace)
        if expert is None:
            return None
        hook = getattr(expert.policy, "observe_shadow_transition", None)
        return hook(trace, observation, **kwargs) if callable(hook) else None

    def set_experience_starvation(self, value: Mapping[str, object]) -> None:
        # Cache trước act; chỉ policy owner sẽ nhận ở _chuan_bi_policy.
        self._pending_starvation = dict(value)
        expert = next((item for item in self.registry.chuyen_gia_theo_context.values() if item.ma_chuyen_gia == self._last_expert_id), None)
        hook = getattr(expert.policy, "set_experience_starvation", None) if expert is not None else None
        if callable(hook):
            hook(value)

    def emit_bridge_heartbeat(self, **kwargs: Any) -> None:
        expert = next((item for item in self.registry.chuyen_gia_theo_context.values() if item.ma_chuyen_gia == self._last_expert_id), None)
        hook = getattr(expert.policy, "emit_bridge_heartbeat", None) if expert is not None else None
        if callable(hook):
            hook(**kwargs)

    def sao_chep(self) -> "HeChuyenGiaMotContextMotPolicy":
        copied = copy.deepcopy(self)
        return copied

    def luu_so_dang_ky(self, duong_dan: str | Path) -> Path:
        """Persist only the lightweight index beside the authoritative artifact.

        Older versions also rewrote a global per-expert checkpoint directory.
        Those files were never read and silently mixed candidates, while
        duplicating the entire neural memory once per expert.
        """
        path = Path(duong_dan)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "che_do": self.cau_hinh.che_do_quyet_dinh,
            "so_chuyen_gia": len(self.registry.chuyen_gia_theo_context),
            "chuyen_gia": [
                {
                    "ma_chuyen_gia": expert.ma_chuyen_gia,
                    "ma_context": expert.ma_context,
                    "ma_context_cha": expert.ma_context_cha,
                    "trang_thai": expert.trang_thai.value,
                    "so_lifecycle_da_dong": expert.so_lifecycle_da_dong,
                    "so_feedback_transition": int(getattr(expert, "so_feedback_transition", 0)),
                    "fitness_cuc_bo": expert.fitness_cuc_bo,
                    "phien_ban_policy": expert.ma_phien_ban,
                }
                for expert in self.registry.chuyen_gia_theo_context.values()
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
