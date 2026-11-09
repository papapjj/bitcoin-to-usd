from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _call(data: dict[str, Any], tool: str, *, path_suffix: str | None = None, action: str | None = None) -> dict[str, Any]:
    for item in data.get("calls", []):
        if item.get("tool") != tool or not item.get("success"):
            continue
        arguments = item.get("arguments") or {}
        if path_suffix is not None and not str(arguments.get("path", "")).replace("\\", "/").endswith(path_suffix):
            continue
        if action is not None and arguments.get("action") != action:
            continue
        return item.get("result") or {}
    raise KeyError(f"Khong tim thay ket qua {tool!r} phu hop")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _temporal_mass(data: dict[str, Any]) -> dict[str, float]:
    result = _call(data, "temporal_attention_distribution", path_suffix="temporal_attention.npy")
    item = next(
        entry
        for entry in data["calls"]
        if entry.get("tool") == "temporal_attention_distribution" and entry.get("success")
    )
    labels = (item.get("arguments") or {}).get("labels") or []
    values = result.get("key_distribution") or []
    grouped: dict[str, float] = defaultdict(float)
    for label, value in zip(labels, values):
        grouped[str(label)] += float(value)
    return dict(grouped)


def _effective_rank(matrix: np.ndarray) -> dict[str, Any]:
    centered = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(centered, compute_uv=False)
    singular_sum = float(singular.sum())
    squared = singular * singular
    squared_sum = float(squared.sum())
    singular_share = singular / singular_sum if singular_sum else np.zeros_like(singular)
    variance_share = squared / squared_sum if squared_sum else np.zeros_like(squared)
    positive_singular = singular_share[singular_share > 0]
    positive_variance = variance_share[variance_share > 0]
    tolerance = singular.max(initial=0.0) * max(centered.shape) * np.finfo(singular.dtype).eps
    return {
        "effective_singular": float(np.exp(-np.sum(positive_singular * np.log(positive_singular)))) if positive_singular.size else 0.0,
        "effective_variance": float(np.exp(-np.sum(positive_variance * np.log(positive_variance)))) if positive_variance.size else 0.0,
        "numerical": int(np.sum(singular > tolerance)),
        "components_95": int(np.searchsorted(np.cumsum(variance_share), 0.95) + 1) if squared_sum else 0,
        "first_8_variance": float(np.cumsum(variance_share)[min(7, len(variance_share) - 1)]) if variance_share.size else 0.0,
    }


def _role_structure(path: Path) -> dict[str, Any]:
    values = np.load(path)
    if values.ndim != 2 or values.shape[0] % 5:
        raise ValueError(f"Can ma tran [so_thoi_diem * 5, so_chieu], nhan {values.shape}")
    cube = values.reshape(-1, 5, values.shape[1]).astype(np.float64, copy=False)
    global_mean = cube.mean(axis=(0, 1))
    role_means = cube.mean(axis=0)
    time_means = cube.mean(axis=1)
    total = float(np.square(cube - global_mean).sum())
    role_sum = float(cube.shape[0] * np.square(role_means - global_mean).sum())
    time_sum = float(cube.shape[1] * np.square(time_means - global_mean).sum())
    role_centered = (cube - role_means[None, :, :]).reshape(values.shape)
    per_role = []
    for index in range(5):
        role_values = cube[:, index, :]
        per_role.append({"role": index, **_effective_rank(role_values - role_values.mean(axis=0))})
    return {
        "time_steps": int(cube.shape[0]),
        "between_role_variance_fraction": role_sum / total if total else 0.0,
        "between_time_variance_fraction": time_sum / total if total else 0.0,
        "role_time_interaction_fraction": max(total - role_sum - time_sum, 0.0) / total if total else 0.0,
        "role_centered": _effective_rank(role_centered),
        "per_role": per_role,
    }


def _attention_details(artifact_dir: Path) -> dict[str, Any]:
    attention = np.asarray(np.load(artifact_dir / "attention.npy"), dtype=np.float64)
    attention = attention / np.clip(attention.sum(axis=-1, keepdims=True), 1e-12, None)
    entropy = -np.sum(np.clip(attention, 1e-12, 1.0) * np.log(np.clip(attention, 1e-12, 1.0)), axis=-1) / math.log(attention.shape[-1])
    head_entropy = entropy.mean(axis=(0, 2))
    head_diagonal = np.asarray(
        [np.diagonal(attention[:, head], axis1=-2, axis2=-1).mean() for head in range(attention.shape[1])]
    )
    return {
        "snapshots": int(attention.shape[0]),
        "head_count": int(attention.shape[1]),
        "head_entropy_min": float(head_entropy.min()),
        "head_entropy_max": float(head_entropy.max()),
        "head_diagonal_min": float(head_diagonal.min()),
        "head_diagonal_max": float(head_diagonal.max()),
        "key_agent_mass": [float(value) for value in attention.mean(axis=(0, 1, 2))],
    }


def _temporal_stability(artifact_dir: Path) -> dict[str, Any]:
    values = np.asarray(np.load(artifact_dir / "temporal_attention.npy"), dtype=np.float64)
    values = values / np.clip(values.sum(axis=-1, keepdims=True), 1e-12, None)
    snapshot_distribution = values.mean(axis=(1, 2))
    overall = snapshot_distribution.mean(axis=0)
    group_distribution = snapshot_distribution.reshape(snapshot_distribution.shape[0], 5, 5).sum(axis=-1)
    return {
        "snapshots": int(values.shape[0]),
        "max_key_std_across_snapshots": float(snapshot_distribution.std(axis=0).max()),
        "mean_l1_from_overall": float(np.abs(snapshot_distribution - overall).sum(axis=1).mean()),
        "group_std_across_snapshots": [float(value) for value in group_distribution.std(axis=0)],
    }


def _tensor_details(artifact_dir: Path) -> dict[str, Any]:
    features = np.load(artifact_dir / "features.npy")
    cell = np.load(artifact_dir / "cell.npy")
    return {
        "feature_always_zero_count": int(np.all(features == 0, axis=0).sum()),
        "feature_unique_rows": int(np.unique(features, axis=0).shape[0]),
        "cell": {
            "shape": list(cell.shape),
            "nonfinite_count": int((~np.isfinite(cell)).sum()),
            "zero_fraction": float(np.mean(cell == 0)),
            "mean": float(np.mean(cell)),
            "std": float(np.std(cell)),
            "min": float(np.min(cell)),
            "max": float(np.max(cell)),
        },
    }


def build_summary(data: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    raw_rank = _call(data, "effective_rank", path_suffix="raw_hidden.npy")
    raw_var = _call(data, "explained_variance", path_suffix="raw_hidden.npy")
    policy_rank = _call(data, "effective_rank", path_suffix="policy_hidden.npy")
    policy_var = _call(data, "explained_variance", path_suffix="policy_hidden.npy")
    attention_entropy = _call(data, "attention_entropy", path_suffix="attention.npy")
    diagonal = _call(data, "diagonal_attention", path_suffix="attention.npy")
    diversity = _call(data, "head_diversity", path_suffix="attention.npy")
    proposals = _call(data, "action_distribution_monitor", path_suffix="proposed_actions.npy")
    features = _call(data, "runtime_feature_monitor", path_suffix="features.npy")
    hidden = _call(data, "hidden_state_monitor", path_suffix="raw_hidden.npy")
    profiler = _call(data, "pytorch_profiler", action="read")

    roles: list[dict[str, Any]] = []
    role_action_ids = {
        0: [0, 1, 3, 5],
        1: [0, 2, 4, 6],
        2: [0, 11],
        3: [0, 7],
        4: [0, 10],
    }
    for index in range(5):
        result = _call(data, "logit_margin_monitor", path_suffix=f"role_{index}_logits.npy")
        counts = [int(v) for v in result["argmax_counts"]]
        logits = np.asarray(np.load(artifact_dir / f"role_{index}_logits.npy"), dtype=np.float64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        argmax = logits.argmax(axis=1)
        sorted_logits = np.sort(logits, axis=1)
        margins = sorted_logits[:, -1] - sorted_logits[:, -2]
        non_hold_decisions = [
            {
                "sample_index": int(row),
                "action": int(role_action_ids[index][argmax[row]]),
                "margin": float(margins[row]),
                "top_probability": float(probabilities[row, argmax[row]]),
            }
            for row in np.flatnonzero(argmax != 0)
        ]
        hold_margins = margins[argmax == 0]
        roles.append(
            {
                "role": index,
                "rows": sum(counts),
                "hold_probability_mean": float(result["mean_probabilities"][0]),
                "non_hold_argmax": sum(counts[1:]),
                "argmax_counts": counts,
                "margin_mean": float(result["margin_mean"]),
                "margin_min": float(result["margin_min"]),
                "margin_quantiles": [float(v) for v in result["margin_quantiles"]],
                "hold_margin_median": float(np.median(hold_margins)) if hold_margins.size else None,
                "non_hold_decisions": non_hold_decisions,
            }
        )

    gradient_norms = {str(k): float(v) for k, v in (data.get("gradient_full_norms") or {}).items()}
    gradient_total = math.sqrt(sum(value * value for value in gradient_norms.values()))
    gradient_top = sorted(gradient_norms.items(), key=lambda item: item[1], reverse=True)[:8]
    operator_top = (profiler.get("operators") or [{}])[0]
    profile_steps = max(int(profiler.get("steps", 0)), 1)
    step_row = next(
        (row for row in profiler.get("operators", []) if row.get("name") == "ProfilerStep*"),
        {},
    )
    tensor_details = _tensor_details(artifact_dir)
    raw_role_structure = _role_structure(artifact_dir / "raw_hidden.npy")
    policy_role_structure = _role_structure(artifact_dir / "policy_hidden.npy")
    profile_total = float(step_row.get("cpu_time_us", 0.0))

    calls = data.get("calls") or []
    return {
        "schema": 1,
        "status": data.get("status"),
        "successful_calls": sum(bool(item.get("success")) for item in calls),
        "total_calls": len(calls),
        "unique_tools": data.get("unique_tools"),
        "checkpoint": data.get("checkpoint"),
        "checkpoint_sha256": data.get("checkpoint_sha256"),
        "checkpoint_unchanged": bool(data.get("checkpoint_unchanged")),
        "sample": {
            "feature_rows": int(data.get("sample_rows", 0)),
            "hidden_rows": int(raw_rank["shape"][0]),
            "feature_count": int(features["shape"][1]),
            "hidden_size": int(raw_rank["shape"][1]),
            "first_bar": data.get("first_bar"),
            "last_bar": data.get("last_bar"),
        },
        "rank": {
            "raw_hidden": {
                "effective_singular": float(raw_rank["effective_rank_singular"]),
                "effective_variance": float(raw_rank["effective_rank_variance"]),
                "numerical": int(raw_rank["numerical_rank"]),
                "components_95": int(raw_var["components_95"]),
                "role_structure": raw_role_structure,
            },
            "policy_hidden": {
                "effective_singular": float(policy_rank["effective_rank_singular"]),
                "effective_variance": float(policy_rank["effective_rank_variance"]),
                "numerical": int(policy_rank["numerical_rank"]),
                "components_95": int(policy_var["components_95"]),
                "role_structure": policy_role_structure,
            },
        },
        "attention": {
            "entropy_normalized": float(attention_entropy["entropy_normalized"]),
            "diagonal_mass": float(diagonal["diagonal_attention"]),
            "uniform_diagonal_reference": 0.2,
            "head_diversity_js_nats": float(diversity["head_diversity_js_nats"]),
            "head_diversity_fraction_ln2": float(diversity["head_diversity_js_nats"]) / math.log(2.0),
            "temporal_mass": _temporal_mass(data),
            "between_agents": _attention_details(artifact_dir),
            "temporal_stability": _temporal_stability(artifact_dir),
        },
        "actions": {
            "proposed_count": int(proposals["count"]),
            "counts": {str(k): int(v) for k, v in proposals["counts"].items()},
            "entropy_nats": float(proposals["entropy_nats"]),
            "hold_fraction": float(proposals["counts"].get("0", 0)) / max(int(proposals["count"]), 1),
            "roles": roles,
        },
        "tensors": {
            "features": {key: features[key] for key in ("shape", "nonfinite_count", "zero_fraction", "mean", "std", "min", "max")},
            "hidden": {key: hidden[key] for key in ("shape", "nonfinite_count", "zero_fraction", "mean", "std", "min", "max")},
            "cell": tensor_details["cell"],
            "feature_always_zero_count": tensor_details["feature_always_zero_count"],
            "feature_unique_rows": tensor_details["feature_unique_rows"],
        },
        "gradient_probe": {
            "objective_value": float(data.get("gradient_probe_loss", 0.0)),
            "tensor_count": len(gradient_norms),
            "zero_tensor_count": sum(value == 0.0 for value in gradient_norms.values()),
            "zero_tensor_names": sorted(name for name, value in gradient_norms.items() if value == 0.0),
            "total_l2": gradient_total,
            "top_l2": [{"name": name, "l2": value} for name, value in gradient_top],
            "meaning": "diagnostic_sensitivity_without_optimizer_step",
        },
        "profile": {
            "steps": int(profiler.get("steps", 0)),
            "instrumented_cpu_ms_per_step": float(step_row.get("cpu_time_us", 0.0)) / 1000.0 / profile_steps,
            "top_operator": operator_top,
            "top_operator_self_cpu_fraction": float(operator_top.get("self_cpu_time_us", 0.0)) / profile_total if profile_total else 0.0,
            "trace_path": profiler.get("trace_path"),
        },
        "tensorboard_http": data.get("tensorboard_http"),
        "limitations": data.get("limitations") or [],
    }


def write_chart(summary: dict[str, Any], path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("V193 vòng 13 — tổng hợp phép soi mạng", fontsize=17, fontweight="bold")

    raw = summary["rank"]["raw_hidden"]
    policy = summary["rank"]["policy_hidden"]
    labels = [
        f"Trạng thái LSTM\nvai trò: {raw['role_structure']['between_role_variance_fraction']:.1%}",
        f"Trạng thái quyết định\nvai trò: {policy['role_structure']['between_role_variance_fraction']:.1%}",
    ]
    x = np.arange(2)
    width = 0.2
    raw_adjusted = raw["role_structure"]["role_centered"]
    policy_adjusted = policy["role_structure"]["role_centered"]
    axes[0, 0].bar(x - 1.5 * width, [raw["effective_singular"], policy["effective_singular"]], width, label="Giá trị kỳ dị, toàn bộ")
    axes[0, 0].bar(x - 0.5 * width, [raw_adjusted["effective_singular"], policy_adjusted["effective_singular"]], width, label="Giá trị kỳ dị, bỏ vai trò")
    axes[0, 0].bar(x + 0.5 * width, [raw["effective_variance"], policy["effective_variance"]], width, label="Phương sai, toàn bộ")
    axes[0, 0].bar(x + 1.5 * width, [raw_adjusted["effective_variance"], policy_adjusted["effective_variance"]], width, label="Phương sai, bỏ vai trò")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Hạng hiệu dụng")
    axes[0, 0].set_title("Không gian biểu diễn và ảnh hưởng của vai trò")
    axes[0, 0].legend(fontsize=8)

    attention = summary["attention"]
    att_labels = ["Độ phân tán\nchuẩn hóa", "Khối lượng\nđường chéo", "Mốc đều\nđường chéo", "Khác biệt\ngiữa nhánh"]
    att_values = [
        attention["entropy_normalized"],
        attention["diagonal_mass"],
        attention["uniform_diagonal_reference"],
        attention["head_diversity_fraction_ln2"],
    ]
    axes[0, 1].bar(att_labels, att_values, color=["#4c78a8", "#f58518", "#9ca3af", "#54a24b"])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Chú ý giữa năm tác nhân")
    axes[0, 1].set_ylabel("Tỷ lệ")
    for index, value in enumerate(att_values):
        axes[0, 1].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=10)

    temporal = attention["temporal_mass"]
    temporal_order = ["2", "8", "32", "128", "current"]
    temporal_labels = ["2", "8", "32", "128", "Hiện tại"]
    temporal_values = [float(temporal.get(key, 0.0)) for key in temporal_order]
    axes[1, 0].bar(temporal_labels, temporal_values, color="#6f4e9c")
    axes[1, 0].set_ylim(0, max(temporal_values) * 1.25)
    axes[1, 0].set_title("Phân bố chú ý theo khoảng thời gian")
    axes[1, 0].set_xlabel("Nhóm khoảng nhìn lại")
    axes[1, 0].set_ylabel("Khối lượng chú ý")
    for index, value in enumerate(temporal_values):
        axes[1, 0].text(index, value + 0.008, f"{value:.3f}", ha="center", fontsize=10)

    roles = summary["actions"]["roles"]
    hold = [item["hold_probability_mean"] for item in roles]
    non_hold = [item["non_hold_argmax"] / max(item["rows"], 1) for item in roles]
    role_x = np.arange(5)
    axes[1, 1].bar(role_x - width / 2, hold, width, label="Xác suất giữ nguyên")
    axes[1, 1].bar(role_x + width / 2, non_hold, width, label="Tỷ lệ chọn hành động khác")
    axes[1, 1].set_xticks(role_x, ["Mua", "Bán", "Quản lý 1", "Quản lý 2", "Quản lý 3"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Quyết định theo vai trò trên 32 thời điểm")
    axes[1, 1].set_ylabel("Tỷ lệ")
    axes[1, 1].legend()

    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path, source_results: Path, chart: Path) -> None:
    raw = summary["rank"]["raw_hidden"]
    policy = summary["rank"]["policy_hidden"]
    attention = summary["attention"]
    actions = summary["actions"]
    features = summary["tensors"]["features"]
    hidden = summary["tensors"]["hidden"]
    gradient = summary["gradient_probe"]
    profile = summary["profile"]
    temporal = attention["temporal_mass"]
    between_agents = attention["between_agents"]
    temporal_stability = attention["temporal_stability"]
    raw_structure = raw["role_structure"]
    policy_structure = policy["role_structure"]
    raw_adjusted = raw_structure["role_centered"]
    policy_adjusted = policy_structure["role_centered"]
    role_rows = "\n".join(
        f"| {item['role']} | {item['rows']} | {item['non_hold_argmax']} | {_fmt(item['hold_probability_mean'])} | {_fmt(item['margin_mean'])} | {_fmt(item['margin_min'], 6)} |"
        for item in actions["roles"]
    )
    top_grad = "\n".join(
        f"| `{item['name']}` | {_fmt(item['l2'])} |" for item in gradient["top_l2"][:5]
    )
    text = f"""# Kết quả toàn bộ phép soi mạng — V193 vòng 13

## Phán đoán chính

Khối mã hóa, LSTM và khối chú ý **không có dấu hiệu hỏng số học hoặc mất hoàn toàn đường truyền học** trên mẫu đã phát lại. Khối chú ý làm thay đổi phổ phương sai và các nhánh chú ý không hoàn toàn giống nhau; phép dò độ dốc đi được từ đầu ra hành động về phần lớn các tầng trước. Các phép đo này chưa chứng minh kiến trúc tạo được biểu diễn có ích cho lợi nhuận.

Điểm nghẽn nhìn thấy rõ hơn nằm ở bề mặt quyết định: {actions['counts'].get('0', 0)}/{actions['proposed_count']} quyết định là giữ nguyên. Chỉ vai trò mua và bán tạo tổng cộng 4 quyết định vào lệnh trên 32 thời điểm được đo; ba vai trò quản lý bị giới hạn quyền nên không được dùng để suy ra xu hướng vào lệnh. Kết quả này phù hợp với chẩn đoán trước đó rằng bài học bắt chước chỉ dạy tốt phần giữ/vào, còn kiểu lệnh và hình học lệnh chưa được dạy đầy đủ.

## Phạm vi

- Điểm lưu: `{summary['checkpoint']}`
- Dấu kiểm tra: `{summary['checkpoint_sha256']}`; tệp không đổi sau khi soi: `{summary['checkpoint_unchanged']}`.
- Dữ liệu: {summary['sample']['feature_rows']} dòng đặc trưng, từ {summary['sample']['first_bar']} đến {summary['sample']['last_bar']}; tạo {summary['sample']['hidden_rows']} trạng thái cho năm tác nhân.
- 32 dòng đầu làm nóng trạng thái, 32 dòng sau dùng để đo; hành động trước được lấy từ chính lần phát lại.
- Đây là phát lại điểm lưu vòng 13, không phải trọng số cuối của lần MT5 bị kết thúc bất thường.
- Tất cả {summary['successful_calls']}/{summary['total_calls']} lượt gọi đạt; có {summary['unique_tools']} tên công cụ khác nhau, gồm một công cụ kiểm kê bổ sung.

## Hạng và phương sai

| Trạng thái | Phương sai do vai trò | Hạng theo phương sai, toàn bộ | Hạng theo phương sai, bỏ vai trò | Số chiều giữ 95%, bỏ vai trò |
|---|---:|---:|---:|---:|
| Sau LSTM | {_fmt(raw_structure['between_role_variance_fraction'])} | {_fmt(raw['effective_variance'])} | {_fmt(raw_adjusted['effective_variance'])} | {raw_adjusted['components_95']} |
| Sau khối chú ý | {_fmt(policy_structure['between_role_variance_fraction'])} | {_fmt(policy['effective_variance'])} | {_fmt(policy_adjusted['effective_variance'])} | {policy_adjusted['components_95']} |

Chênh lệch trung bình giữa năm vai trò chiếm {raw_structure['between_role_variance_fraction']:.2%} phương sai trước và {policy_structure['between_role_variance_fraction']:.2%} sau khối chú ý. Sau khi bỏ trung bình riêng của từng vai trò, hạng hiệu dụng theo phương sai chỉ tăng từ {_fmt(raw_adjusted['effective_variance'])} lên {_fmt(policy_adjusted['effective_variance'])}; cả hai cần 7 chiều để giữ 95% phương sai. Phần lớn cấu trúc toàn bộ vì vậy là mã vai trò, không phải biến thiên thị trường.

Đây vẫn **chưa phải bằng chứng sụp biểu diễn**: chỉ có {raw_structure['time_steps']} thời điểm liên tiếp. Hạng số {raw['numerical']} của ma trận toàn bộ gần bằng đúng mức trần {summary['sample']['hidden_rows'] - 1} sau khi trừ trung bình nên hầu như không có giá trị chẩn đoán. Cần lặp phép đo trên nhiều chế độ thị trường và cửa sổ dài hơn.

## Chú ý giữa tác nhân và theo thời gian

- Độ phân tán chú ý chuẩn hóa: {_fmt(attention['entropy_normalized'])}. Giá trị dưới 1 cho thấy phân bố không đều hoàn toàn.
- Khối lượng trên đường chéo: {_fmt(attention['diagonal_mass'])}; mốc phân bố đều cho năm tác nhân là 0,2000. Mạng không chỉ nhìn chính nó.
- Mức khác biệt Jensen–Shannon giữa các nhánh: {_fmt(attention['head_diversity_js_nats'])} nat. Các nhánh có khác biệt, nhưng chỉ số này không nói khác biệt đó có tạo lợi nhuận.
- Từng nhánh có độ phân tán từ {_fmt(between_agents['head_entropy_min'])} đến {_fmt(between_agents['head_entropy_max'])}, còn khối lượng đường chéo từ {_fmt(between_agents['head_diagonal_min'])} đến {_fmt(between_agents['head_diagonal_max'])}.
- Khối lượng trung bình hướng tới năm tác nhân là {', '.join(_fmt(value) for value in between_agents['key_agent_mass'])}; hai tác nhân vào lệnh nhận phần lớn hơn ba tác nhân quản lý.
- Khối lượng theo khoảng nhìn lại: 2 = {_fmt(temporal.get('2', 0.0))}, 8 = {_fmt(temporal.get('8', 0.0))}, 32 = {_fmt(temporal.get('32', 0.0))}, 128 = {_fmt(temporal.get('128', 0.0))}, hiện tại = {_fmt(temporal.get('current', 0.0))}.

Khoảng 128 nhận nhiều chú ý nhất và thời điểm hiện tại nhận ít nhất. Phân bố này gần như đứng yên qua {temporal_stability['snapshots']} ảnh chụp: độ lệch chuẩn lớn nhất của một khóa chỉ {temporal_stability['max_key_std_across_snapshots']:.2e}, còn độ lệch tuyệt đối tổng trung bình so với phân bố chung là {_fmt(temporal_stability['mean_l1_from_overall'], 6)}. Đây có thể là thiên lệch tĩnh của phần mã hóa khoảng nhìn lại; chưa có bằng chứng nó điều chỉnh theo trạng thái thị trường. Các nhãn 2/8/32/128 đại diện cho nhóm đường tích lũy, không phải độ trễ chính xác.

## Khả năng phát lệnh

| Vai trò | Số thời điểm | Chọn khác giữ nguyên | Xác suất giữ nguyên trung bình | Biên điểm trung bình | Biên nhỏ nhất |
|---|---:|---:|---:|---:|---:|
{role_rows}

Vai trò mua chọn 1 lệnh mua thị trường và 1 lệnh mua giới hạn; vai trò bán chọn 2 lệnh bán dừng. Tổng hợp cả năm vai trò tạo {actions['counts'].get('0', 0)} giữ nguyên, 1 mua thị trường, 1 mua giới hạn và 2 bán dừng. Biên nhỏ nhất của hai vai trò vào lệnh chỉ {_fmt(actions['roles'][0]['margin_min'], 6)} và {_fmt(actions['roles'][1]['margin_min'], 6)}, nên một số thời điểm nằm rất sát ranh giới quyết định dù biên trung bình khá lớn.

Bốn quyết định vào đều yếu: xác suất cao nhất chỉ nằm trong khoảng {min(item['top_probability'] for role in actions['roles'][:2] for item in role['non_hold_decisions']):.3f}–{max(item['top_probability'] for role in actions['roles'][:2] for item in role['non_hold_decisions']):.3f}, và biên điểm nằm trong khoảng {min(item['margin'] for role in actions['roles'][:2] for item in role['non_hold_decisions']):.4f}–{max(item['margin'] for role in actions['roles'][:2] for item in role['non_hold_decisions']):.4f}. Trong khi đó, trung vị biên của các quyết định giữ nguyên là {_fmt(actions['roles'][0]['hold_margin_median'])} ở vai trò mua và {_fmt(actions['roles'][1]['hold_margin_median'])} ở vai trò bán. Điểm lưu có khả năng vào lệnh, nhưng bốn tín hiệu vào trong cửa sổ ngắn này đều mong manh.

## Đặc trưng, trạng thái và độ dốc

- Đầu vào không có giá trị không hữu hạn; tỷ lệ số 0 là {_fmt(features['zero_fraction'])}, trung bình {_fmt(features['mean'])}, độ lệch chuẩn {_fmt(features['std'])}, khoảng [{_fmt(features['min'])}, {_fmt(features['max'])}]. Có {summary['tensors']['feature_always_zero_count']}/137 cột luôn bằng 0 trong đoạn này, dù cả {summary['tensors']['feature_unique_rows']} hàng đều khác nhau. Cần đối chiếu 44 cột với ý nghĩa đặc trưng và trạng thái không có vị thế; số 0 chưa tự chứng minh đặc trưng chết.
- Trạng thái LSTM không có giá trị không hữu hạn; trung bình {_fmt(hidden['mean'], 6)}, độ lệch chuẩn {_fmt(hidden['std'], 6)}, khoảng [{_fmt(hidden['min'])}, {_fmt(hidden['max'])}].
- Trạng thái ô nhớ cũng hữu hạn, khoảng [{_fmt(summary['tensors']['cell']['min'])}, {_fmt(summary['tensors']['cell']['max'])}].
- Phép dò độ dốc chạm {gradient['tensor_count']} khối tham số, {gradient['zero_tensor_count']} khối bằng 0; chuẩn tổng là {_fmt(gradient['total_l2'])}.

| Khối nhận độ dốc mạnh nhất | Chuẩn độ dốc |
|---|---:|
{top_grad}

Đây là độ nhạy của hàm thử độ phân tán hành động tại một dòng, không phải độ dốc PPO lịch sử và không có bước cập nhật trọng số. Chuẩn của một hệ số vô hướng không so trực tiếp được với chuẩn của một ma trận hàng triệu phần tử. Năm khối bằng 0 đều thuộc nhánh trạng thái vị thế và có thể do dòng thử không có vị thế mở. Phép đo chỉ chứng minh đường truyền học tồn tại.

## Thời gian chạy và theo dõi trực tiếp

- Bộ đo ghi {profile['steps']} bước; trung bình khoảng {_fmt(profile['instrumented_cpu_ms_per_step'], 1)} mili giây CPU mỗi bước khi đang gắn bộ thu thập.
- Phép tốn thời gian nhất là `{profile['top_operator'].get('name')}`, tức phép nhân ma trận có cộng, với {_fmt(float(profile['top_operator'].get('self_cpu_time_us', 0.0)) / 1000.0, 1)} mili giây tự thân trên toàn bộ ba bước, khoảng {profile['top_operator_self_cpu_fraction']:.1%} tổng thời gian bước được đo.
- TensorBoard trả mã {summary['tensorboard_http']}, sau đó đã được dừng an toàn.
- Móc truyền xuôi đã thu được LSTM và đầu truy vấn hành động, rồi được tắt; điểm lưu không bị sửa.

## Kết luận có thể hành động

1. Chưa có bằng chứng buộc phải tăng kích thước mạng hoặc thay LSTM/khối chú ý. Khối biểu diễn hoạt động về mặt số học, nhưng chất lượng kiến trúc vẫn chưa được chứng minh.
2. Cần sửa mục tiêu và đầu ra trước: dạy riêng sáu kiểu vào, giá đặt, khả năng khớp, cắt lỗ, chốt lời, thời gian chờ và thoát lệnh.
3. Cần đo riêng vai trò mua/bán; tỷ lệ giữ nguyên toàn hệ thống bị ba vai trò quản lý làm lệch.
4. Chỉ đánh giá khối chú ý bằng thử nghiệm cùng dữ liệu và cùng điều kiện MT5. Độ phân tán, hạng và khác biệt giữa nhánh không thay thế được lợi nhuận sau chi phí.

## Tệp bằng chứng

- Kết quả đầy đủ: [{source_results.name}]({source_results.resolve().as_posix()})
- Biểu đồ: [{chart.name}]({chart.resolve().as_posix()})
- Dấu vết thời gian: `{summary['profile']['trace_path']}`
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tong hop ket qua bo soi mang PyTorch thanh JSON, bao cao va bieu do.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    results_path = args.results.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    summary = build_summary(data, results_path.parent)

    summary_path = output_dir / "summary.json"
    chart_path = output_dir / "analysis_overview.png"
    report_path = output_dir / "ANALYSIS.vi.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chart(summary, chart_path)
    write_report(summary, report_path, results_path, chart_path)
    print(json.dumps({"summary": str(summary_path), "chart": str(chart_path), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
