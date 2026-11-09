"""Hồ sơ giải thích quyết định và nhật ký phát triển cấu trúc.

Module này chỉ đọc trace, graph và growth event đã có. Nó không thay đổi action,
trọng số, plasticity, growth hay MT5. Mục tiêu là xuất bằng chứng có thể audit:
context nào sở hữu quyết định, policy nào ra lệnh, neuron/cạnh nào hoạt động mạnh,
và vì sao neuron/liên kết/module được tạo hoặc bị thay đổi.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping

import numpy as np

from .entry_feature_enrichment import ENTRY_INTERACTION_NAMES

# Không import bắt buộc `features` lúc module này được nạp: `config` ->
# `live_observatory` -> module này có thể tạo vòng import. Danh sách fallback
# giữ đúng thứ tự feature contract để log vẫn mang tên tiếng Việt khi vòng import
# đang chưa hoàn chỉnh.
_FEATURE_COLUMNS_FALLBACK = (
    "ret1_atr", "ret3_atr", "ret5_atr", "ret15_atr", "ret30_atr", "ret60_atr",
    "gap_atr", "body_atr", "upper_wick_atr", "lower_wick_atr", "range_atr",
    "ema_diff_atr", "ema_slope_atr", "range_position", "atr_percentile",
    "spread_atr", "rsi14", "rsi5", "macd_hist_atr",
    "bb_position", "volatility_ratio", "momentum_5", "momentum_15",
    "candle_body_ratio", "high_low_ratio", "ema50_slope_atr", "price_vs_ema20_atr",
    "price_vs_ema50_atr", "consec_up_atr", "consec_down_atr", "hour_sin", "hour_cos",
    "m5_ema_diff_atr", "m5_ema_slope_atr", "m5_volatility_ratio", "m5_gap_atr",
    "m15_ema_diff_atr", "m15_ema_slope_atr", "m15_volatility_ratio", "m15_gap_atr",
    "h1_ema_diff_atr", "h1_ema_slope_atr", "h1_volatility_ratio", "h1_gap_atr",
    "trend_strength_atr", "chop_index", "range_compression",
    "breakout_pressure_20", "breakout_pressure_60", "shock_range_atr",
    "session_asia", "spread_percentile", "dist_high20_atr", "dist_low20_atr",
    "dist_high60_atr", "dist_low60_atr", "pullback_ema20_atr", "pullback_ema50_atr",
    "swing_range120_position", "dist_swing_high120_atr", "dist_swing_low120_atr",
    "liquidity_sweep_rejection", "relative_volume_20", "relative_volume_60",
    "volume_zscore_60", "volume_trend_20", "effort_result_ratio", "absorption_score",
    "range_width_atr", "range_age_normalized", "spring_score", "upthrust_score",
    "sign_of_strength_score", "sign_of_weakness_score", "spring_test_quality",
    "upthrust_test_quality", "bars_since_spring", "bars_since_upthrust",
)
try:  # Chỉ dùng để gắn nhãn feature; fallback luôn có đủ thứ tự.
    from .features import MARKET_FEATURE_COLUMNS as _MARKET_COLUMNS_IMPORTED
    from .features import TAIL_FEATURE_COLUMNS as _TAIL_COLUMNS_IMPORTED
except Exception:  # pragma: no cover - vòng import lúc khởi động hoặc artifact tối giản
    _MARKET_COLUMNS_IMPORTED = ()
    _TAIL_COLUMNS_IMPORTED = ()
FEATURE_COLUMNS = tuple(_MARKET_COLUMNS_IMPORTED) + tuple(_TAIL_COLUMNS_IMPORTED)
FEATURE_COLUMNS = FEATURE_COLUMNS or _FEATURE_COLUMNS_FALLBACK

MA_HANH_DONG = (
    "HOLD", "MARKET_BUY", "MARKET_SELL", "BUY_LIMIT", "SELL_LIMIT",
    "BUY_STOP", "SELL_STOP", "CANCEL_ORDERS", "CLOSE_WORST",
    "CLOSE_BEST", "CLOSE_ALL", "MANAGE_POSITION",
)
TEN_HANH_DONG_VI = (
    "GIỮ / KHÔNG_VÀO_LỆNH (HOLD)",
    "MUA_THỊ_TRƯỜNG (MARKET_BUY)",
    "BÁN_THỊ_TRƯỜNG (MARKET_SELL)",
    "MUA_GIỚI_HẠN (BUY_LIMIT)",
    "BÁN_GIỚI_HẠN (SELL_LIMIT)",
    "MUA_DỪNG (BUY_STOP)",
    "BÁN_DỪNG (SELL_STOP)",
    "HỦY_LỆNH_CHỜ (CANCEL_ORDERS)",
    "ĐÓNG_LỆNH_XẤU_NHẤT (CLOSE_WORST)",
    "ĐÓNG_LỆNH_TỐT_NHẤT (CLOSE_BEST)",
    "ĐÓNG_TẤT_CẢ_LỆNH (CLOSE_ALL)",
    "QUẢN_LÝ_VỊ_THẾ (MANAGE_POSITION)",
)

# Tên feature dùng trong audit/UI; mã nguồn feature gốc vẫn giữ nguyên để
# không làm sai hợp đồng 137 input với MT5.
TEN_FEATURE_THI_TRUONG_VI = {
    "ret1_atr": "biến_động_giá_1_nến_theo_ATR",
    "ret3_atr": "biến_động_giá_3_nến_theo_ATR",
    "ret5_atr": "biến_động_giá_5_nến_theo_ATR",
    "ret15_atr": "biến_động_giá_15_nến_theo_ATR",
    "ret30_atr": "biến_động_giá_30_nến_theo_ATR",
    "ret60_atr": "biến_động_giá_60_nến_theo_ATR",
    "gap_atr": "khoảng_trống_giá_theo_ATR",
    "body_atr": "thân_nến_theo_ATR",
    "upper_wick_atr": "bóng_nến_trên_theo_ATR",
    "lower_wick_atr": "bóng_nến_dưới_theo_ATR",
    "range_atr": "biên_độ_nến_theo_ATR",
    "ema_diff_atr": "chênh_lệch_EMA_theo_ATR",
    "ema_slope_atr": "độ_dốc_EMA_theo_ATR",
    "range_position": "vị_trí_trong_biên_giá",
    "atr_percentile": "phân_vị_biến_động_ATR",
    "spread_atr": "spread_theo_ATR",
    "rsi14": "RSI_14_đã_chuẩn_hóa",
    "rsi5": "RSI_5_đã_chuẩn_hóa",
    "macd_hist_atr": "histogram_MACD_theo_ATR",
    "bb_position": "vị_trí_trong_dải_Bollinger",
    "volatility_ratio": "tỷ_lệ_biến_động",
    "momentum_5": "động_lượng_5_nến",
    "momentum_15": "động_lượng_15_nến",
    "candle_body_ratio": "tỷ_lệ_thân_nến",
    "high_low_ratio": "tỷ_lệ_đỉnh_đáy",
    "ema50_slope_atr": "độ_dốc_EMA50_theo_ATR",
    "price_vs_ema20_atr": "giá_so_với_EMA20_theo_ATR",
    "price_vs_ema50_atr": "giá_so_với_EMA50_theo_ATR",
    "consec_up_atr": "chuỗi_nến_tăng_theo_ATR",
    "consec_down_atr": "chuỗi_nến_giảm_theo_ATR",
    "hour_sin": "chu_kỳ_giờ_sin",
    "hour_cos": "chu_kỳ_giờ_cos",
    "m5_ema_diff_atr": "chênh_lệch_EMA_khung_M5_theo_ATR",
    "m5_ema_slope_atr": "độ_dốc_EMA_khung_M5_theo_ATR",
    "m5_volatility_ratio": "tỷ_lệ_biến_động_khung_M5",
    "m5_gap_atr": "khoảng_trống_giá_khung_M5_theo_ATR",
    "m15_ema_diff_atr": "chênh_lệch_EMA_khung_M15_theo_ATR",
    "m15_ema_slope_atr": "độ_dốc_EMA_khung_M15_theo_ATR",
    "m15_volatility_ratio": "tỷ_lệ_biến_động_khung_M15",
    "m15_gap_atr": "khoảng_trống_giá_khung_M15_theo_ATR",
    "h1_ema_diff_atr": "chênh_lệch_EMA_khung_H1_theo_ATR",
    "h1_ema_slope_atr": "độ_dốc_EMA_khung_H1_theo_ATR",
    "h1_volatility_ratio": "tỷ_lệ_biến_động_khung_H1",
    "h1_gap_atr": "khoảng_trống_giá_khung_H1_theo_ATR",
    "trend_strength_atr": "sức_mạnh_xu_hướng_theo_ATR",
    "chop_index": "chỉ_số_nhiễu_đi_ngang",
    "range_compression": "mức_co_hẹp_biên_độ",
    "breakout_pressure_20": "áp_lực_phá_vỡ_20_nến",
    "breakout_pressure_60": "áp_lực_phá_vỡ_60_nến",
    "shock_range_atr": "biên_độ_sốc_theo_ATR",
    "session_asia": "phiên_châu_Á",
    "spread_percentile": "phân_vị_spread",
    "dist_high20_atr": "khoảng_cách_đỉnh_20_nến_theo_ATR",
    "dist_low20_atr": "khoảng_cách_đáy_20_nến_theo_ATR",
    "dist_high60_atr": "khoảng_cách_đỉnh_60_nến_theo_ATR",
    "dist_low60_atr": "khoảng_cách_đáy_60_nến_theo_ATR",
    "pullback_ema20_atr": "mức_hồi_về_EMA20_theo_ATR",
    "pullback_ema50_atr": "mức_hồi_về_EMA50_theo_ATR",
    "swing_range120_position": "vị_trí_trong_swing_Wyckoff",
    "dist_swing_high120_atr": "khoảng_cách_tới_đỉnh_swing_theo_ATR",
    "dist_swing_low120_atr": "khoảng_cách_tới_đáy_swing_theo_ATR",
    "liquidity_sweep_rejection": "quét_thanh_khoản_và_từ_chối_giá",
    "relative_volume_20": "khối_lượng_tương_đối_20_nến",
    "relative_volume_60": "khối_lượng_tương_đối_60_nến",
    "volume_zscore_60": "độ_bất_thường_khối_lượng_60_nến",
    "volume_trend_20": "xu_hướng_khối_lượng_20_nến",
    "effort_result_ratio": "tỷ_lệ_nỗ_lực_kết_quả_Wyckoff",
    "absorption_score": "điểm_hấp_thụ_Wyckoff",
    "range_width_atr": "độ_rộng_trading_range_theo_ATR",
    "range_age_normalized": "tuổi_trading_range",
    "spring_score": "điểm_Spring",
    "upthrust_score": "điểm_Upthrust",
    "sign_of_strength_score": "điểm_SOS",
    "sign_of_weakness_score": "điểm_SOW",
    "spring_test_quality": "chất_lượng_retest_Spring",
    "upthrust_test_quality": "chất_lượng_retest_Upthrust",
    "bars_since_spring": "số_nến_từ_Spring",
    "bars_since_upthrust": "số_nến_từ_Upthrust",
}

HANH_DONG_VAO_LENH = frozenset({1, 2, 3, 4, 5, 6})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _input_value(value: object) -> float:
    """Collapse legacy controller-value maps without silently reporting zero."""

    if isinstance(value, Mapping):
        numbers = [_finite(item, float("nan")) for item in value.values()]
        numbers = [item for item in numbers if math.isfinite(item)]
        return float(np.mean(numbers)) if numbers else 0.0
    return _finite(value)


def _action_name(value: object, fallback: str = "") -> str:
    try:
        index = int(float(value))
    except (TypeError, ValueError):
        return str(fallback or "")
    return TEN_HANH_DONG_VI[index] if 0 <= index < len(TEN_HANH_DONG_VI) else str(fallback or index)


def _action_index_from_details(details: Mapping[str, Any], key: str, default: int = -1) -> int:
    """Read only explicit bridge/policy action fields.

    M1 ``selected_idx`` is a hypothesis index, not an MT5 action code.  Audit
    logs must keep those two namespaces separate or the profile can describe a
    BUY_STOP/CLOSE_ALL hypothesis as if it were the order sent to the bridge.
    """
    if key not in details:
        return int(default)
    return int(_finite(details.get(key), float(default)))


def _feature_name(index: int) -> str:
    """Tên tiếng Việt ổn định cho wire và local candidate input."""
    idx = int(index)
    features = list(FEATURE_COLUMNS)
    if 0 <= idx < 29:
        return f"thi_truong.{TEN_FEATURE_THI_TRUONG_VI.get(features[idx], features[idx])}"
    if 29 <= idx < 69:
        return f"vi_the.slot_{idx - 29:02d}"
    if 69 <= idx < 77:
        return f"tai_khoan.slot_{idx - 69:02d}"
    if 77 <= idx < 106:
        feature_idx = 29 + (idx - 77)
        if feature_idx < len(features):
            return f"thi_truong.{TEN_FEATURE_THI_TRUONG_VI.get(features[feature_idx], features[feature_idx])}"
        return f"thi_truong.slot_{feature_idx:02d}"
    if 106 <= idx < 117:
        return f"thuc_thi.slot_{idx - 106:02d}"
    if 117 <= idx < 137:
        feature_idx = 58 + (idx - 117)
        if feature_idx < len(features):
            return f"thi_truong.{TEN_FEATURE_THI_TRUONG_VI.get(features[feature_idx], features[feature_idx])}"
    if idx == 137:
        return "local.context_M15_t-1"
    if idx == 138:
        return "local.context_M15_t-2"
    interaction_idx = idx - 139
    if 0 <= interaction_idx < len(ENTRY_INTERACTION_NAMES):
        return f"tuong_tac_entry.{ENTRY_INTERACTION_NAMES[interaction_idx]}"
    return f"input.slot_{idx:03d}"


def _input_index(node_id: str) -> int | None:
    text = str(node_id or "")
    marker = ":input:"
    if marker not in text:
        return None
    try:
        return int(text.rsplit(marker, 1)[1])
    except (TypeError, ValueError):
        return None


def _iter_graphs(policy: Any) -> dict[str, tuple[str, Any]]:
    """Trả graph theo namespace trùng với graph_traces/live observatory."""
    found: dict[str, tuple[str, Any]] = {}
    for index, module in enumerate(getattr(policy, "modules", []) or []):
        m1 = getattr(module, "m1_component", None)
        graph = getattr(m1, "graph", None)
        if graph is None or not hasattr(graph, "neurons") or not hasattr(graph, "edges"):
            continue
        name = str(getattr(module, "name", f"module_{index}") or f"module_{index}")
        found[f"m1:{name}"] = (name, graph)
    reasoning = getattr(getattr(policy, "reasoning_graph", None), "graph", None)
    if reasoning is not None and hasattr(reasoning, "neurons") and hasattr(reasoning, "edges"):
        found["reasoning_graph"] = ("reasoning_graph", reasoning)
    manager = getattr(policy, "child_supernetwork_manager", None)
    children = getattr(manager, "children", {}) if manager is not None else {}
    if isinstance(children, Mapping):
        for network_id, record in children.items():
            graph = getattr(getattr(record, "network", None), "graph", None)
            if graph is not None and hasattr(graph, "neurons") and hasattr(graph, "edges"):
                found[f"child:{network_id}"] = (f"child:{network_id}", graph)
    return found


def _thinking_outputs(policy: Any) -> list[dict[str, Any]]:
    """Latest cognitive pipeline per M1 graph, retained read-only for audit."""
    rows: list[dict[str, Any]] = []
    for graph_id, (module_name, _graph) in _iter_graphs(policy).items():
        if graph_id == "reasoning_graph":
            continue
        module = next((item for item in (getattr(policy, "modules", []) or []) if str(getattr(item, "name", "")) == str(module_name)), None)
        m1 = getattr(module, "m1_component", None) if module is not None else None
        snapshot = getattr(m1, "thinking_snapshot", None)
        data = snapshot() if callable(snapshot) else getattr(m1, "last_thinking_output", {})
        if isinstance(data, Mapping) and data:
            rows.append({"graph_id": str(graph_id), "module": str(module_name), **dict(data)})
    return rows


def _graph_records(policy: Any, trace: Any) -> dict[str, dict[str, Any]]:
    raw = getattr(trace, "graph_traces", None)
    if isinstance(raw, Mapping) and raw:
        return {str(key): _mapping(value) for key, value in raw.items() if isinstance(value, Mapping)}
    # Artifact cũ: chỉ có trace cấp policy. Lấy graph M1 đầu tiên.
    graphs = _iter_graphs(policy)
    first = next((key for key in graphs if key != "reasoning_graph"), "")
    if not first:
        return {}
    return {
        first: {
            "graph_id": first,
            "active_neuron_ids": tuple(str(x) for x in getattr(trace, "active_neuron_ids", ()) or ()),
            "active_edge_ids": tuple(str(x) for x in getattr(trace, "active_edge_ids", ()) or ()),
            "edge_gates": _mapping(getattr(trace, "graph_edge_gates", {})),
            "edge_contributions": {},
            "node_values": {},
            "raw_inputs": {},
            "region_activity": _mapping(getattr(trace, "graph_region_activity", {})),
        }
    }


def _context_payload(trace: Any) -> dict[str, Any]:
    audit = _mapping(getattr(trace, "context_audit", {}))
    return {
        "ma_context": str(audit.get("ma_context", audit.get("context_nhan_dien", "")) or ""),
        "ma_context_cha": str(audit.get("ma_context_cha", "") or ""),
        "ma_chuyen_gia": str(audit.get("ma_chuyen_gia", "") or ""),
        "ma_policy": str(audit.get("ma_policy", "") or ""),
        "phien_ban_policy": str(audit.get("phien_ban_policy", "") or ""),
        "trang_thai_chuyen_gia": str(audit.get("trang_thai_chuyen_gia", "") or ""),
        "policy_instance_uuid": str(audit.get("policy_instance_uuid", "") or ""),
        "graph_instance_uuid": str(audit.get("graph_instance_uuid", "") or ""),
        "weight_hash": str(audit.get("weight_hash", "") or ""),
        "memory_hash": str(audit.get("memory_hash", "") or ""),
        "policy_revision": int(_finite(audit.get("policy_revision", 0), 0.0)),
        "context_on_dinh": bool(audit.get("context_on_dinh", False)),
        "do_tin_cay_context": _finite(audit.get("do_tin_cay_context", 0.0)),
        "ly_do_context": str(audit.get("ly_do_context", "") or ""),
        "ly_do_khong_giao_dich": str(audit.get("ly_do_khong_giao_dich", getattr(trace, "ly_do_khong_giao_dich", "")) or ""),
        "audit_day_du": audit,
    }


def _candidate_rows(trace: Any, details: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
    scores = np.asarray(getattr(trace, "candidate_scores", np.zeros(0)), dtype=float).reshape(-1)
    logits = np.asarray(getattr(trace, "candidate_logits", np.zeros(0)), dtype=float).reshape(-1)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    selected = int(getattr(trace, "selected_idx", -1) or -1)
    rows: list[dict[str, Any]] = []
    count = min(max(actions.shape[0], scores.size, logits.size), 32)
    for index in range(count):
        vector = actions[index] if index < actions.shape[0] else np.zeros(0, dtype=float)
        primary = int(np.argmax(vector[:12])) if vector.size else -1
        rows.append({
            "ma_ung_vien": index,
            "duoc_chon": index == selected,
            "hanh_dong_chinh": _action_name(primary),
            "chi_so_hanh_dong_chinh": primary,
            "diem": _finite(scores[index]) if index < scores.size else 0.0,
            "logit": _finite(logits[index]) if index < logits.size else 0.0,
            "vector_dau_ra": [float(x) for x in vector[:20].tolist()],
        })
    rows.sort(key=lambda item: float(item["diem"]), reverse=True)
    return rows


def _tham_so_dau_ra(trace: Any, details: Mapping[str, Any]) -> dict[str, Any]:
    """Diễn giải head tham số mà không tự ý coi raw logit là giá trị MT5 đã khớp."""
    values = np.asarray(getattr(trace, "selected_action", np.zeros(0)), dtype=float).reshape(-1)
    labels = (
        (12, "rui_ro_raw", "Head rủi ro thô; giá trị % thật chỉ có sau bridge decode."),
        (13, "stop_loss_ATR_raw", "Head khoảng cách stop-loss theo ATR ở dạng thô."),
        (14, "take_profit_ATR_raw", "Head khoảng cách take-profit theo ATR ở dạng thô."),
        (15, "offset_vao_lenh_ATR_raw", "Head khoảng cách lệnh chờ theo ATR ở dạng thô."),
        (16, "trailing_ATR_raw", "Head trailing theo ATR ở dạng thô."),
        (17, "chot_mot_phan_raw", "Head tỷ lệ chốt một phần ở dạng thô."),
        (18, "che_do_quan_ly_raw", "Head chọn chế độ quản lý vị thế ở dạng thô."),
        (19, "muc_tieu_vi_the_raw", "Head chọn slot/vị thế mục tiêu ở dạng thô."),
    )
    raw = [
        {"chi_so": idx, "ten": name, "gia_tri_raw": _finite(values[idx]) if idx < values.size else 0.0, "dien_giai": note}
        for idx, name, note in labels
    ]
    decoded_keys = (
        "risk_pct", "sl_atr", "tp_atr", "entry_offset_atr", "trailing_stop_atr",
        "partial_close_pct", "management_mode", "target_slot", "position_scale",
    )
    decoded = {key: details[key] for key in decoded_keys if key in details}
    return {
        "head_tho": raw,
        "tham_so_da_giai_ma_neu_da_co": decoded,
        "ghi_chu": "Giá trị đã giải mã chỉ xuất hiện khi bridge/MT5 đã bổ sung action_details. Head thô không phải là lệnh đã khớp.",
    }


def _reason_rows(trace: Any, details: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    ma_context = str(context.get("ma_context", "") or "")
    ma_expert = str(context.get("ma_chuyen_gia", "") or "")
    if ma_context:
        reasons.append({
            "ma": "CONTEXT_OWNER",
            "giai_doan": "dinh_tuyen",
            "muc_do": "thong_tin",
            "gia_tri": _finite(context.get("do_tin_cay_context", 0.0)),
            "noi_dung": f"Context {ma_context} được gán cho chuyên gia {ma_expert or 'chưa có mã'}.",
        })
    selected = int(getattr(trace, "selected_idx", -1) or -1)
    scores = np.asarray(getattr(trace, "candidate_scores", np.zeros(0)), dtype=float).reshape(-1)
    if 0 <= selected < scores.size:
        reasons.append({
            "ma": "CHON_UNG_VIEN_POLICY",
            "giai_doan": "policy",
            "muc_do": "chinh",
            "gia_tri": _finite(scores[selected]),
            "noi_dung": f"Policy chọn ứng viên {selected} vì điểm selector/critic của ứng viên này là {_finite(scores[selected]):.6f}.",
        })
    # Hợp đồng V9 chỉ công bố action nghĩa vụ và action wire. Các giá trị
    # policy_action/legal_action legacy có thể còn tồn tại nội bộ để đối chiếu
    # bridge, nhưng tuyệt đối không đi vào hồ sơ/UI dưới tên action hay sentinel -1.
    semantic_action = _action_index_from_details(
        details, "semantic_action", _action_index_from_details(details, "emitted_action", -1)
    )
    emitted_action = _action_index_from_details(details, "emitted_action", semantic_action)
    wire_action = _action_index_from_details(details, "wire_action", emitted_action)
    if semantic_action >= 0:
        reasons.append({
            "ma": "ACTION_NGHIA_VU",
            "giai_doan": "policy_context_owner",
            "muc_do": "chinh",
            "gia_tri": float(semantic_action),
            "noi_dung": f"Chuyên gia owner chọn hành động nghĩa vụ {_action_name(semantic_action)}.",
        })
    if emitted_action >= 0 and emitted_action != semantic_action:
        reasons.append({
            "ma": "LECH_NGHIA_VU_VA_EMIT",
            "giai_doan": "hop_dong_hanh_dong",
            "muc_do": "loi",
            "gia_tri": float(emitted_action),
            "noi_dung": f"Chặn hành động vì emit {_action_name(emitted_action)} lệch nghĩa vụ {_action_name(semantic_action)}.",
        })
    if wire_action >= 0 and wire_action != semantic_action:
        reasons.append({
            "ma": "LECH_NGHIA_VU_VA_WIRE",
            "giai_doan": "hop_dong_hanh_dong",
            "muc_do": "loi",
            "gia_tri": float(wire_action),
            "noi_dung": f"Chặn hành động vì wire MT5 {_action_name(wire_action)} lệch nghĩa vụ {_action_name(semantic_action)}.",
        })
    if bool(details.get("action_masked", False)):
        reasons.append({
            "ma": "ACTION_MASKED",
            "giai_doan": "kiem_tra_hanh_dong",
            "muc_do": "canh_bao",
            "gia_tri": 1.0,
            "noi_dung": "Action gốc bị mask vì không hợp lệ với trạng thái vị thế/rủi ro hiện tại.",
        })
    hold_reason = str(details.get("hold_reason", context.get("ly_do_khong_giao_dich", "")) or "")
    if hold_reason:
        reasons.append({
            "ma": "LY_DO_KHONG_GIAO_DICH",
            "giai_doan": "quyet_dinh",
            "muc_do": "chinh",
            "gia_tri": 0.0,
            "noi_dung": f"Không mở lệnh hoặc gửi no-op vì: {hold_reason}.",
        })
    if bool(details.get("reasoning_changed_candidate", False)):
        reasons.append({
            "ma": "PHAN_BIEN_DOI_UNG_VIEN",
            "giai_doan": "reasoning",
            "muc_do": "chinh",
            "gia_tri": _finite(details.get("reasoning_action_score_gain", 0.0)),
            "noi_dung": "Mô-đun phản biện đã đổi ứng viên sau khi đánh giá lại candidate hợp lệ.",
        })
    recovery = _mapping(details.get("learner_recovery", {}))
    if bool(recovery.get("replacement_applied", False)):
        reasons.append({
            "ma": "THAY_THE_HANH_DONG",
            "giai_doan": "learner_recovery",
            "muc_do": "chinh",
            "gia_tri": _finite(recovery.get("replacement_confidence", 0.0)),
            "noi_dung": "Learner recovery đã dùng bằng chứng cục bộ để thay phương án hành động/điều chỉnh.",
        })
    expected_lcb = _finite(details.get("expectancy_predicted_lcb", 0.0))
    expected_samples = _finite(details.get("expectancy_predicted_samples", 0.0))
    if expected_samples > 0:
        reasons.append({
            "ma": "EXPECTANCY_CUC_BO",
            "giai_doan": "ky_vong",
            "muc_do": "thong_tin",
            "gia_tri": expected_lcb,
            "noi_dung": f"Expectancy cục bộ có {expected_samples:.0f} mẫu, LCB dự đoán {expected_lcb:.6f}.",
        })
    if bool(context.get("audit_day_du", {}).get("per_expert_risk_cap_active", False)):
        reasons.append({
            "ma": "GIOI_HAN_RUI_RO_CHUYEN_GIA",
            "giai_doan": "an_toan",
            "muc_do": "canh_bao",
            "gia_tri": _finite(context.get("audit_day_du", {}).get("per_expert_max_risk_pct", 0.0)),
            "noi_dung": "Policy của chuyên gia đang bị giới hạn rủi ro riêng theo lifecycle/context.",
        })
    return reasons


def _responsibility(policy: Any, trace: Any, *, max_neurons: int, max_edges: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    graphs = _iter_graphs(policy)
    records = _graph_records(policy, trace)
    node_scores: dict[tuple[str, str], dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    features: dict[int, dict[str, Any]] = {}
    details = _mapping(getattr(trace, "action_details", {}))
    # Direct RNN is the active production backend.  Its old explanation path
    # incorrectly interpreted a graph controller's nested values as scalar
    # MT5 features, yielding ``gia_tri=0`` for every feature.  Prefer the
    # policy-local Jacobian proxy emitted at decision time; it is explicitly
    # labelled as approximate, but it is attached to the real 137 inputs.
    direct_attribution = details.get("direct_input_attribution", [])
    if isinstance(direct_attribution, (list, tuple)):
        for row in direct_attribution:
            item = _mapping(row)
            index = int(_finite(item.get("index", -1), -1.0))
            if index < 0:
                continue
            signed = _finite(item.get("signed_contribution", 0.0))
            features[index] = {
                "chi_so": index, "ten": _feature_name(index),
                "gia_tri": _finite(item.get("value", 0.0)),
                "tong_dong_gop_tuyet_doi": abs(signed), "tong_dong_gop_co_dau": signed,
                "canh": [], "muc_do_quy_trach": "xap_xi_local_w_in_action_margin",
            }

    for graph_id, record in records.items():
        entry = graphs.get(graph_id)
        if entry is None:
            continue
        module_name, graph = entry
        edge_contrib = {str(k): _finite(v) for k, v in _mapping(record.get("edge_contributions", {})).items()}
        edge_gates = {str(k): _finite(v) for k, v in _mapping(record.get("edge_gates", {})).items()}
        node_values = {str(k): _mapping(v) for k, v in _mapping(record.get("node_values", {})).items()}
        raw_inputs = {str(k): _input_value(v) for k, v in _mapping(record.get("raw_inputs", record.get("input_values", {}))).items()}
        active_nodes = [str(x) for x in (record.get("active_neuron_ids", ()) or ())]

        for local_id in active_nodes:
            node = getattr(graph, "neurons", {}).get(local_id)
            if node is None:
                continue
            value = _mapping(node_values.get(local_id, {}))
            key = (graph_id, local_id)
            node_scores.setdefault(key, {
                "id": f"{graph_id}::{local_id}", "local_id": local_id, "graph_id": graph_id,
                "module": str(module_name), "module_id": str(getattr(node, "module_id", "") or ""),
                "region": str(getattr(node, "region", "")), "lifecycle": str(getattr(node, "lifecycle", "")),
                "activation": _finite(value.get("activation", getattr(node, "activation", 0.0))),
                "state": _finite(value.get("state", getattr(node, "state", 0.0))),
                "utility": _finite(getattr(node, "utility", 0.0)),
                "plasticity": _finite(getattr(node, "plasticity", 0.0)),
                "fade": _finite(getattr(node, "fade", 1.0), 1.0),
                "tong_dong_gop_tuyet_doi": 0.0, "tong_dong_gop_co_dau": 0.0,
                "so_canh_dang_hoat_dong": 0, "canh_lien_quan": [],
            })

        for edge_id, contribution in edge_contrib.items():
            edge = getattr(graph, "edges", {}).get(edge_id)
            if edge is None:
                continue
            src = str(getattr(edge, "src", "")); dst = str(getattr(edge, "dst", ""))
            src_node = getattr(graph, "neurons", {}).get(src)
            dst_node = getattr(graph, "neurons", {}).get(dst)
            row = {
                "id": f"{graph_id}::{edge_id}", "local_id": edge_id, "graph_id": graph_id,
                "module": str(module_name), "src": f"{graph_id}::{src}", "dst": f"{graph_id}::{dst}",
                "src_region": str(getattr(src_node, "region", "")), "dst_region": str(getattr(dst_node, "region", "")),
                "src_module_id": str(getattr(src_node, "module_id", "") or ""), "dst_module_id": str(getattr(dst_node, "module_id", "") or ""),
                "kind": str(getattr(edge, "kind", "")), "lifecycle": str(getattr(edge, "lifecycle", "")),
                "contribution": _finite(contribution), "contribution_abs": abs(_finite(contribution)),
                "gate": _finite(edge_gates.get(edge_id, getattr(edge, "gate", 0.0))),
                "weight": _finite(getattr(edge, "weight", 0.0)), "utility": _finite(getattr(edge, "utility", 0.0)),
                "plasticity": _finite(getattr(edge, "plasticity", 0.0)), "fade": _finite(getattr(edge, "fade", 1.0), 1.0),
                "created_by": str(getattr(edge, "created_by", "") or ""), "need_reason": str(getattr(edge, "need_reason", "") or ""),
                "candidate_id": str(getattr(edge, "candidate_id", "") or ""),
                "lien_quan_truc_tiep_dau_ra": bool(str(getattr(dst_node, "region", "")) == "output"),
            }
            edges.append(row)
            for local_id in (src, dst):
                node = getattr(graph, "neurons", {}).get(local_id)
                if node is None:
                    continue
                key = (graph_id, local_id)
                target = node_scores.setdefault(key, {
                    "id": f"{graph_id}::{local_id}", "local_id": local_id, "graph_id": graph_id,
                    "module": str(module_name), "module_id": str(getattr(node, "module_id", "") or ""),
                    "region": str(getattr(node, "region", "")), "lifecycle": str(getattr(node, "lifecycle", "")),
                    "activation": _finite(getattr(node, "activation", 0.0)), "state": _finite(getattr(node, "state", 0.0)),
                    "utility": _finite(getattr(node, "utility", 0.0)), "plasticity": _finite(getattr(node, "plasticity", 0.0)),
                    "fade": _finite(getattr(node, "fade", 1.0), 1.0),
                    "tong_dong_gop_tuyet_doi": 0.0, "tong_dong_gop_co_dau": 0.0,
                    "so_canh_dang_hoat_dong": 0, "canh_lien_quan": [],
                })
                target["tong_dong_gop_tuyet_doi"] += abs(_finite(contribution))
                target["tong_dong_gop_co_dau"] += _finite(contribution)
                target["so_canh_dang_hoat_dong"] += 1
                target["canh_lien_quan"].append(f"{graph_id}::{edge_id}")
            input_idx = _input_index(src)
            # Do not mix controller-local graph dimensions with the 137 MT5
            # feature attribution when direct-RNN supplied its own trace.
            if input_idx is not None and not direct_attribution and input_idx not in features:
                item = features.setdefault(input_idx, {
                    "chi_so": input_idx, "ten": _feature_name(input_idx), "gia_tri": _finite(raw_inputs.get(src, 0.0)),
                    "tong_dong_gop_tuyet_doi": 0.0, "tong_dong_gop_co_dau": 0.0, "canh": [],
                })
                item["tong_dong_gop_tuyet_doi"] += abs(_finite(contribution))
                item["tong_dong_gop_co_dau"] += _finite(contribution)
                item["canh"].append(f"{graph_id}::{edge_id}")

    for item in node_scores.values():
        # Trace là attribution theo activity/cạnh; không tuyên bố đây là causal proof.
        item["muc_do_quy_trach"] = "xap_xi_theo_dong_gop_canh_hoat_dong"
        item["huong_tin_hieu"] = "thuan" if item["tong_dong_gop_co_dau"] > 0 else "nghich" if item["tong_dong_gop_co_dau"] < 0 else "trung_tinh"
    neuron_rows = sorted(node_scores.values(), key=lambda x: (float(x["tong_dong_gop_tuyet_doi"]), abs(float(x["activation"]))), reverse=True)
    if int(max_neurons) > 0:
        neuron_rows = neuron_rows[:int(max_neurons)]
    edge_rows = sorted(edges, key=lambda x: float(x["contribution_abs"]), reverse=True)
    if int(max_edges) > 0:
        edge_rows = edge_rows[:int(max_edges)]
    # 0 giới hạn hiển thị: giữ toàn bộ feature có attribution để UI/audit không mất dữ liệu.
    feature_rows = sorted(features.values(), key=lambda x: float(x["tong_dong_gop_tuyet_doi"]), reverse=True)
    return neuron_rows, edge_rows, feature_rows


def _kiem_tra_hop_dong_owner(trace: Any, context: Mapping[str, Any], details: Mapping[str, Any]) -> tuple[bool, str]:
    """Không để log mô tả lệnh của trace thiếu context-owner là lệnh thật."""
    if bool(getattr(trace, "ly_do_khong_hoc", False)) or bool(context.get("audit_day_du", {}).get("khong_co_policy_ra_lenh", False)):
        return False, str(context.get("ly_do_khong_giao_dich", "TRACE_KHONG_DUOC_RA_LENH") or "TRACE_KHONG_DUOC_RA_LENH")
    for key in ("ma_context", "ma_chuyen_gia", "ma_policy"):
        if not str(context.get(key, "") or ""):
            return False, f"THIEU_{key.upper()}"
    if bool(details.get("wire_noop", False)) or bool(details.get("no_neural_command", False)):
        return False, "NO_COMMAND_TRANSPORT"
    if "wire_matches_semantic_action" in details and not bool(details.get("wire_matches_semantic_action", False)):
        return False, "LECH_HANH_DONG_NGHIAVU_VA_WIRE"
    semantic = _action_index_from_details(details, "semantic_action", _action_index_from_details(details, "emitted_action", -1))
    emitted = _action_index_from_details(details, "emitted_action", semantic)
    wire = _action_index_from_details(details, "wire_action", emitted)
    # Một record thiếu semantic/emitted/wire action không được phép bị hiển thị
    # như action -1. Hợp đồng V9 biểu diễn nó thành KHÔNG_GỬI_LENH fail-closed.
    if semantic < 0 or emitted < 0 or wire < 0:
        return False, "THIEU_HANH_DONG_NGHIAVU_HOP_LE"
    if semantic != emitted:
        return False, "LECH_HANH_DONG_NGHIAVU_VA_EMIT"
    if wire != semantic:
        return False, "LECH_HANH_DONG_NGHIAVU_VA_WIRE"
    return True, ""


def xay_ho_so_quyet_dinh(
    policy: Any,
    trace: Any,
    *,
    max_neurons: int = 0,
    max_edges: int = 0,
) -> dict[str, Any]:
    """Tạo log giải thích quyết định. Không thay đổi policy/trace."""
    details = _mapping(getattr(trace, "action_details", {}))
    context = _context_payload(trace)
    neurons, edges, features = _responsibility(policy, trace, max_neurons=max_neurons, max_edges=max_edges)
    model_hypothesis_index = int(_finite(details.get("model_hypothesis_index", getattr(trace, "selected_idx", -1)), -1.0))
    # V10 không đọc/khôi phục policy_action hay legal_action legacy.
    # Hypothesis index là namespace riêng, action semantic do contract bridge
    # tạo sau khi policy owner chọn vector output hợp lệ.
    semantic_action = _action_index_from_details(details, "semantic_action", -1)
    emitted_action = _action_index_from_details(details, "emitted_action", semantic_action)
    wire_action = _action_index_from_details(details, "wire_action", emitted_action)
    hop_le, ly_do_chan = _kiem_tra_hop_dong_owner(trace, context, details)
    if not hop_le:
        # Trace cũ/generic có thể giữ action raw. Hợp đồng audit bắt buộc biểu
        # diễn nó là NO_COMMAND, không được khiến UI hiểu là order MT5.
        emitted_action = -1
        semantic_action = -1
        wire_action = -1
    reasons = _reason_rows(trace, details, context)
    if not hop_le:
        reasons.insert(0, {"nhom": "hop_dong/chan", "noi_dung": f"Không gửi MT5: {ly_do_chan}.", "gia_tri": 0.0})
    is_entry = hop_le and emitted_action in HANH_DONG_VAO_LENH
    no_trade = str(context.get("ly_do_khong_giao_dich", "") or ly_do_chan)
    if not hop_le:
        summary = f"Không gửi lệnh MT5: {ly_do_chan}. Context={context.get('ma_context') or 'chưa xác định'}."
    elif no_trade:
        summary = f"Không giao dịch: {no_trade}. Context={context.get('ma_context') or 'chưa xác định'}."
    elif is_entry:
        summary = f"Đặt lệnh {_action_name(emitted_action)} bởi {context.get('ma_chuyen_gia')} trong context {context.get('ma_context')}."
    else:
        summary = f"Quyết định {_action_name(emitted_action)} trong context {context.get('ma_context')}."
    context_chua_on_dinh = bool(getattr(trace, "ly_do_khong_hoc", False)) or not bool(context.get("ma_context", ""))
    context_dang_xac_nhan = bool(not context.get("context_on_dinh", False) and hop_le)
    loai_ban_ghi = (
        "CONTEXT_CHUA_ON_DINH" if context_chua_on_dinh
        else "CONTEXT_DANG_XAC_NHAN" if context_dang_xac_nhan
        else "HO_SO_QUYET_DINH"
    )
    modules = tuple(getattr(policy, "modules", ()) or ())
    action_details = dict(getattr(trace, "action_details", {}) or {})
    regularization = {
        "hidden_normalization_mode": str(getattr(policy, "hidden_normalization_mode", "layernorm_v1") or "layernorm_v1"),
        "hidden_normalization_enabled": bool(getattr(policy, "hidden_normalization_enabled", True)),
        "hidden_normalization_layers": [
            bool(item) for item in tuple(getattr(policy, "hidden_normalization_layers", ()) or ())
        ],
        "hidden_dropout_enabled": bool(getattr(policy, "hidden_dropout_enabled", True)),
        "hidden_dropout_rates": [
            float(item) for item in tuple(getattr(policy, "hidden_dropout_rates", ()) or ())
        ],
        "hidden_dropout_train_only": bool(getattr(policy, "hidden_dropout_train_only", True)),
        "hidden_dropout_step": int(max(
            [int(getattr(module, "hidden_dropout_step", 0) or 0) for module in modules] or [0]
        )),
        "hidden_dropout_training_active": bool(
            action_details.get("hidden_dropout_training_active", False)
        ),
        "hidden_dropout_steps_before": dict(
            action_details.get("hidden_dropout_steps_before", {}) or {}
        ),
        "hidden_dropout_steps_advanced": dict(
            action_details.get("hidden_dropout_steps_advanced", {}) or {}
        ),
        "output_normalization_mode": str(getattr(policy, "output_normalization_mode", "raw") or "raw"),
        "module_count": len(modules),
    }
    raw_allowed_value = getattr(trace, "allowed_action_mask", ())
    if raw_allowed_value is None:
        raw_allowed_actions = ()
    else:
        try:
            raw_allowed_actions = tuple(raw_allowed_value)
        except TypeError:
            raw_allowed_actions = ()
    def _audit_sequence(value: object) -> tuple[object, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            return (value,)
        try:
            return tuple(value)  # type: ignore[arg-type]
        except TypeError:
            return (value,)
    allowed_action_indices = [
        int(index) for index, allowed in enumerate(raw_allowed_actions)
        if bool(allowed)
    ]
    live_vetoed_actions = [
        int(value) for value in _audit_sequence(
            details.get("five_live_action_veto_vetoed_actions", ())
        )
        if str(value).lstrip("-").isdigit()
    ]
    expectancy_vetoed_actions = [
        int(value) for value in _audit_sequence(
            details.get("five_expectancy_vetoed_actions", ())
        )
        if str(value).lstrip("-").isdigit()
    ]
    probe_actions = [
        int(value) for value in _audit_sequence(
            details.get("five_expectancy_probe_preserved_actions", ())
        )
        if str(value).lstrip("-").isdigit()
    ]
    context_escape_actions = [
        int(value) for value in _audit_sequence(
            details.get("five_expectancy_context_escape_actions", ())
        )
        if str(value).lstrip("-").isdigit()
    ]
    action_surface_audit = {
        "schema_version": 1,
        "allowed_action_indices": allowed_action_indices,
        "allowed_action_count": len(allowed_action_indices),
        "mask_only_hold": allowed_action_indices == [0],
        "selected_action_index": semantic_action if hop_le else None,
        "selected_action_probability": _finite(
            details.get("five_selected_action_probability", 0.0)
        ),
        "selected_behavior_action_probability": _finite(
            details.get("five_selected_behavior_action_probability", 0.0)
        ),
        "live_vetoed_actions": sorted(set(live_vetoed_actions)),
        "expectancy_vetoed_actions": sorted(set(expectancy_vetoed_actions)),
        "probe_preserved_actions": sorted(set(probe_actions)),
        "context_escape_actions": sorted(set(context_escape_actions)),
        "live_probe_due": bool(details.get("five_live_action_veto_probe_due", False)),
        "selector_reason": str(context.get("selector_reason", "") or ""),
        "action_origin": str(details.get("action_origin", "policy") or "policy"),
        "neural_command": bool(
            hop_le
            and semantic_action >= 0
            and not bool(details.get("action_masked", False))
        ),
    }
    return {
        "loai_ban_ghi": loai_ban_ghi,
        "schema_version": 5,
        "giai_doan": (
            "CONTEXT_CHUA_ON_DINH" if context_chua_on_dinh
            else "CONTEXT_DANG_XAC_NHAN" if context_dang_xac_nhan
            else "QUYET_DINH_OWNER"
        ),
        "tao_luc": _now(),
        "trace_id": str(getattr(trace, "trace_id", "") or ""),
        "frame_id": str(getattr(trace, "context_frame_id", "") or ""),
        "tom_tat_tieng_viet": summary,
        "hop_dong_owner": {"hop_le": bool(hop_le), "ly_do_chan": str(ly_do_chan)},
        "context": context,
        "dau_ra_tu_duy": _thinking_outputs(policy),
        "quyet_dinh": {
            # V9: policy_action/legal_action là sentinel legacy (-1) ở đường
            # context-owner. Không ghi chúng vào hồ sơ vì dễ bị UI hiểu nhầm
            # là action thật. Hồ sơ chỉ công bố action nghĩa vụ và wire cuối.
            "hanh_dong_nghia_vu": "KHÔNG_GỬI_LENH" if not hop_le else _action_name(semantic_action),
            "hanh_dong_wire_mt5": "KHÔNG_GỬI_LENH" if not hop_le else _action_name(wire_action),
            "hanh_dong_gui_mt5": "KHÔNG_GỬI_LENH" if not hop_le else _action_name(emitted_action),
            "semantic_action_index": semantic_action if hop_le else None,
            "emitted_action_index": emitted_action if hop_le else None,
            "wire_action_index": wire_action if hop_le else None,
            "wire_matches_semantic_action": bool(hop_le and details.get("wire_matches_semantic_action", wire_action == semantic_action)),
            "model_hypothesis_index": model_hypothesis_index,
            "confidence": _finite(getattr(trace, "confidence", 0.0)),
            "critic_value": _finite(getattr(trace, "critic_value", 0.0)),
            "la_lenh_vao": is_entry, "action_masked": bool(details.get("action_masked", False)),
            "execution_bootstrap": False, "hold_reason": str(details.get("hold_reason", "") or ""),
            "action_surface_audit": action_surface_audit,
            "ung_vien": _candidate_rows(trace, details), "tham_so_lenh": _tham_so_dau_ra(trace, details),
        },
        # Effective runtime switches are part of the decision profile, not
        # only the YAML, so a live run can prove which regularizer actually
        # participated in the forward/update path.
        "regularization": regularization,
        "ly_do": reasons,
        "dau_vao_anh_huong": ([] if context_chua_on_dinh else features),
        "neuron_chiu_trach_nhiem": ([] if context_chua_on_dinh else neurons),
        "lien_ket_chiu_trach_nhiem": ([] if context_chua_on_dinh else edges),
        "growth_review": _mapping(details.get("growth_review", context.get("growth_review", {}))),
        "gioi_han_dien_giai": {
            "quy_trach_neuron": "Xếp hạng theo độ lớn đóng góp của cạnh đang hoạt động và activation; đây là attribution vận hành, không phải chứng minh nhân quả tuyệt đối.",
            "quy_trach_feature": "Direct-RNN dùng xấp xỉ local w_in/action-margin trên đúng input đã vào policy; graph backend dùng cạnh input đang hoạt động. Cả hai là attribution vận hành, không phải chứng minh nhân quả tuyệt đối.",
            "nguon_su_that_ket_qua": "Kết quả lời/lỗ chỉ được xác nhận sau execution ledger và lifecycle MT5.",
        },
    }


def _event_mapping(event: object) -> dict[str, Any]:
    if is_dataclass(event):
        try:
            return asdict(event)
        except Exception:
            pass
    return _mapping(event)


def _structural_refs(values: object, *, limit: int, kind: str) -> list[dict[str, Any]]:
    """Chuẩn hóa ID/snapshot cấu trúc để JSONL giữ được chi tiết tạo mới.

    `GraphDevelopmentController` trả danh sách ID cho neuron và danh sách dict
    cho edge. Các runtime cũ đôi khi trả object dataclass. Hàm này giữ lại
    snapshot khi có, thay vì ép mọi thứ thành chuỗi khó audit.
    """
    if isinstance(values, (str, bytes)) or values is None:
        raw_values = [] if values is None else [values]
    else:
        try:
            raw_values = list(values)  # type: ignore[arg-type]
        except TypeError:
            raw_values = [values]
    rows: list[dict[str, Any]] = []
    selected_values = raw_values if int(limit) <= 0 else raw_values[:int(limit)]
    for item in selected_values:
        if is_dataclass(item):
            try:
                item = asdict(item)
            except Exception:
                item = str(item)
        if isinstance(item, Mapping):
            row = {str(key): value for key, value in dict(item).items()}
            row.setdefault("loai", kind)
            if not row.get("id"):
                row["id"] = str(row.get("neuron_id", row.get("edge_id", "")) or "")
            rows.append(row)
        else:
            rows.append({"loai": kind, "id": str(item)})
    return rows


def _object_snapshot(policy: Any, target: str) -> dict[str, Any]:
    target = str(target or "")
    if not target:
        return {}
    for graph_id, (module_name, graph) in _iter_graphs(policy).items():
        node = getattr(graph, "neurons", {}).get(target)
        if node is not None:
            return {
                "loai": "neuron", "id": f"{graph_id}::{target}", "graph_id": graph_id, "module": module_name,
                "region": str(getattr(node, "region", "")), "module_id": str(getattr(node, "module_id", "") or ""),
                "lifecycle": str(getattr(node, "lifecycle", "")), "bias": _finite(getattr(node, "bias", 0.0)),
                "utility": _finite(getattr(node, "utility", 0.0)), "plasticity": _finite(getattr(node, "plasticity", 0.0)),
                "fade": _finite(getattr(node, "fade", 1.0), 1.0), "home_context_key": str(getattr(node, "home_context_key", "") or ""),
            }
        edge = getattr(graph, "edges", {}).get(target)
        if edge is not None:
            return {
                "loai": "lien_ket", "id": f"{graph_id}::{target}", "graph_id": graph_id, "module": module_name,
                "src": f"{graph_id}::{getattr(edge, 'src', '')}", "dst": f"{graph_id}::{getattr(edge, 'dst', '')}",
                "kind": str(getattr(edge, "kind", "")), "lifecycle": str(getattr(edge, "lifecycle", "")),
                "weight": _finite(getattr(edge, "weight", 0.0)), "gate": _finite(getattr(edge, "gate", 0.0)),
                "utility": _finite(getattr(edge, "utility", 0.0)), "plasticity": _finite(getattr(edge, "plasticity", 0.0)),
                "fade": _finite(getattr(edge, "fade", 1.0), 1.0), "created_by": str(getattr(edge, "created_by", "") or ""),
                "need_reason": str(getattr(edge, "need_reason", "") or ""), "candidate_id": str(getattr(edge, "candidate_id", "") or ""),
            }
    return {"loai": "khong_tim_thay", "target": target}


def xay_ho_so_phat_trien_cau_truc(policy: Any, trace: Any, event: object, *, giai_doan: str) -> dict[str, Any]:
    """Tạo audit log cho tạo/tắt/nối lại neuron, edge và specialist module."""
    raw = _event_mapping(event)
    details = _mapping(raw.get("details", raw.get("metadata", {})))
    action = str(raw.get("action", raw.get("type", raw.get("kind", "khong_ro"))) or "khong_ro")
    reason = str(raw.get("reason", raw.get("need_reason", raw.get("causal_pattern_key", ""))) or "")
    target = str(raw.get("target", raw.get("module_id", raw.get("specialist_module_id", ""))) or "")
    target_snapshot = _object_snapshot(policy, target)
    created_edges = _structural_refs(
        details.get("created_edges", details.get("interface_edge_ids", raw.get("cross_module_edge_ids", []))),
        limit=0,
        kind="lien_ket",
    )
    created_nodes = _structural_refs(
        details.get("created_neurons", details.get("node_ids", raw.get("specialist_node_ids", []))),
        limit=0,
        kind="neuron",
    )
    action_text = {
        "grow_neuron": "Tạo neuron mới", "grow_edge": "Tạo liên kết mới", "grow_specialist_module": "Tạo module chuyên môn mới",
        "prune_edge": "Tắt/cắt liên kết utility thấp", "rewire_edge": "Nối lại liên kết", "consolidate": "Củng cố liên kết", 
        "split_region": "Tách vùng/module", "merge_region": "Hợp nhất vùng/module", "hold": "Không thay đổi cấu trúc",
        "graph_credit_attribution": "Phân bổ credit lên graph, không tạo cấu trúc mới",
        "context_structural_candidate_created": "Tạo candidate neuron/liên kết cục bộ",
        "context_structural_candidate_kept": "Giữ lại candidate sau lifecycle",
        "context_structural_candidate_faded": "Làm mờ candidate do evidence âm",
    }.get(action, action)
    before_neurons = int(_finite(raw.get("before_neurons", 0), 0.0))
    after_neurons = int(_finite(raw.get("after_neurons", 0), 0.0))
    before_edges = int(_finite(raw.get("before_edges", 0), 0.0))
    after_edges = int(_finite(raw.get("after_edges", 0), 0.0))
    structural_birth = bool(
        created_nodes or created_edges or before_neurons != after_neurons or before_edges != after_edges
        or action in {"context_structural_candidate_created", "context_structural_candidate_kept", "context_structural_candidate_faded"}
    )
    return {
        "loai_ban_ghi": "SU_KIEN_PHAT_TRIEN_CAU_TRUC",
        "schema_version": 1,
        "tao_luc": _now(),
        "giai_doan": str(giai_doan),
        "trace_id": str(getattr(trace, "trace_id", "") or raw.get("trace_id", "") or ""),
        "context": _context_payload(trace),
        "su_kien": {
            "ma_su_kien": str(raw.get("event_id", "") or ""),
            "thoi_diem_su_kien": str(raw.get("created_at", "") or ""),
            "hanh_dong": action, "ten_hanh_dong_tieng_viet": action_text, "ly_do": reason, "target": target,
            "so_neuron_truoc": before_neurons,
            "so_neuron_sau": after_neurons,
            "so_lien_ket_truoc": before_edges,
            "so_lien_ket_sau": after_edges,
            "la_moc_cau_truc": structural_birth,
            "phan_loai_su_kien": "moc_cau_truc" if structural_birth else "credit_khong_moc_cau_truc",
            "target_hien_tai": target_snapshot,
            "neuron_duoc_tao": created_nodes,
            "lien_ket_duoc_tao": created_edges,
            "ma_neuron_duoc_tao": [str(item.get("id", "")) for item in created_nodes],
            "ma_lien_ket_duoc_tao": [str(item.get("id", "")) for item in created_edges],
            "chi_tiet": details,
        },
        "nguon_kich_hoat": {
            "vai_tro": str(raw.get("role", details.get("role", "")) or ""),
            "diem": _finite(raw.get("score", details.get("score", 0.0))),
            "module_need_scores": _mapping(raw.get("module_need_scores", details.get("module_need_scores", {}))),
            "neural_logits": _mapping(details.get("neural_logits", {})),
            "learning_signal": _mapping(raw.get("learning_signal", details.get("learning_signal", {}))),
            "causal_confidence": _finite(details.get("causal_confidence", raw.get("causal_confidence", 0.0))),
            "causal_validated": bool(details.get("causal_validated", raw.get("causal_validated", False))),
        },
        "giai_thich_tieng_viet": (
            f"{action_text}. Lý do kỹ thuật: {reason or 'không có lý do cụ thể trong event'}. "
            + (
                "Đây là event credit/đánh dấu, không phải bằng chứng mọc neuron hay liên kết."
                if not structural_birth
                else "Mọi cấu trúc mới chỉ là candidate cho đến khi có bằng chứng lifecycle/replay/OOS phù hợp."
            )
        ),
        "gioi_han_dien_giai": "Sự kiện ghi lại lý do controller tạo/thay đổi cấu trúc; không đồng nghĩa neuron/module đã được promote hoặc đã chứng minh lợi thế giao dịch.",
    }
