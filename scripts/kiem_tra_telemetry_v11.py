from __future__ import annotations

"""Kiểm tra resolver telemetry V11 mà không nạp toàn bộ JSONL vào RAM.

Dùng trước khi mở Live 3D hoặc khi UI báo 0 neuron/0 liên kết.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _resolve(workspace: Path, raw: object) -> Path:
    text = str(raw or "").strip().replace("\\", "/")
    if not text:
        return Path()
    path = Path(text)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _read_manifest(workspace: Path) -> tuple[Path | None, dict[str, Any]]:
    for candidate in (workspace / "outputs" / "active_run.json", workspace / "active_run.json"):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return candidate, data
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[LOI] Khong doc duoc {candidate}: {exc}")
            return candidate, {}
    return None, {}


def _first_line(path: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        data = bytearray()
        while len(data) < max_bytes:
            block = handle.read(min(1024 * 1024, max_bytes - len(data)))
            if not block:
                break
            data.extend(block)
            marker = data.find(b"\n")
            if marker >= 0:
                return bytes(data[:marker]).decode("utf-8", errors="replace")
    return ""


def _last_line(path: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        chunk = min(size, 1024 * 1024)
        while chunk <= min(size, max_bytes):
            handle.seek(size - chunk)
            data = handle.read(chunk)
            lines = [line for line in data.splitlines() if line.strip()]
            if lines:
                return lines[-1].decode("utf-8", errors="replace")
            if chunk == size:
                break
            chunk = min(size, chunk * 2)
    return ""


def _describe_json_line(line: str) -> str:
    if not line:
        return "khong co dong JSONL hoan chinh trong 16 MiB dau/cuoi"
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return f"dong JSON khong parse duoc (do dai={len(line):,})"
    if not isinstance(row, dict):
        return f"JSON khong phai object ({type(row).__name__})"
    event = row.get("event_type") or row.get("giai_doan") or row.get("event") or "?"
    topology = row.get("topology") or row.get("topology_day_du") or {}
    if not isinstance(topology, dict):
        topology = {}
    nodes = topology.get("node_count", row.get("node_count", "?"))
    edges = topology.get("edge_count", row.get("edge_count", "?"))
    return f"loai={event}; node={nodes}; edge={edges}; keys={','.join(list(row)[:8])}"


def _show_file(label: str, path: Path, *, required: bool = True) -> bool:
    print(f"\n[{label}]")
    print(f"  path: {path}")
    if not path.is_file():
        print("  TRANG_THAI: KHONG_TIM_THAY" + ("" if required else " (TUY_CHON)"))
        return not required
    stat = path.stat()
    print(f"  TRANG_THAI: TON_TAI; size={stat.st_size:,} bytes; modified={stat.st_mtime:.3f}")
    print(f"  dong_dau: {_describe_json_line(_first_line(path))}")
    print(f"  dong_cuoi: {_describe_json_line(_last_line(path))}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiem tra telemetry V11 theo active_run.json")
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    print("=== KIEM TRA TELEMETRY V11 ===")
    print(f"workspace: {workspace}")
    manifest_path, manifest = _read_manifest(workspace)
    if manifest_path is None:
        print("[LOI] Khong co outputs/active_run.json hoac active_run.json tai workspace.")
        return 2
    if not manifest:
        return 2
    print(f"manifest: {manifest_path}")
    print(f"root_run_id: {manifest.get('root_run_id', '')}")
    print(f"schema_version: {manifest.get('schema_version', '')}")
    paths = {
        "TELEMETRY": _resolve(workspace, manifest.get("event_path")),
        "HO_SO_QUYET_DINH": _resolve(workspace, manifest.get("decision_profile_path")),
        "SU_KIEN_CAU_TRUC": _resolve(workspace, manifest.get("structural_event_path")),
        "THEO_DOI_PHAT_TRIEN": _resolve(workspace, manifest.get("network_development_path")),
    }
    valid = True
    for label, path in paths.items():
        valid = _show_file(label, path, required=(label != "SU_KIEN_CAU_TRUC")) and valid
    print("\nKET_LUAN:")
    if valid:
        print("  OK: active_run resolver co du duong dan. UI V188 phai hien root_run_id va doc tung chunk.")
        print("  Neu Live 3D van hien outputs\\quan_sat_chuyen_gia_live.jsonl va 0 neuron, ban dang mo nham JAR cu, khong phai V11.")
        return 0
    print("  LOI: thieu it nhat mot file telemetry cua run hien tai.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
