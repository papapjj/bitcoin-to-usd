#!/usr/bin/env python3
"""Kiểm tra cấu trúc V6: một context ổn định có đúng một owner policy."""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(text: str, marker: str, label: str) -> None:
    if marker not in text:
        raise AssertionError(f"{label}: thiếu marker {marker!r}")


def main() -> int:
    cfg = (ROOT / "configs" / "he_chuyen_gia.yaml").read_text(encoding="utf-8")
    cli = (ROOT / "src" / "self_growing_trader" / "cli.py").read_text(encoding="utf-8")
    registry = (ROOT / "src" / "self_growing_trader" / "he_chuyen_gia_ngu_canh.py").read_text(encoding="utf-8")
    bridge = (ROOT / "src" / "self_growing_trader" / "mt5_runner.py").read_text(encoding="utf-8")
    for marker in (
        "che_do_quyet_dinh: mot_context_mot_chuyen_gia",
        "cho_phep_policy_chung_ra_lenh: false",
        "cho_phep_tron_policy: false",
    ):
        require(cfg, marker, "config")
    for marker in (
        '"khoi-tao-he-chuyen-gia"',
        '"huan-luyen-he-chuyen-gia-mt5"',
        '"walk-forward-he-chuyen-gia-mt5"',
    ):
        require(cli, marker, "cli")
    for marker in (
        "class HeChuyenGiaMotContextMotPolicy",
        "class ChuyenGiaDocLap",
        "ma_chuyen_gia",
        "ma_policy",
    ):
        require(registry, marker, "registry")
    for marker in (
        "_context_expert_owner_valid",
        "THIEU_OWNER_CONTEXT",
        "CONTEXT_CHUA_ON_DINH",
    ):
        require(bridge, marker, "bridge")
    print(json.dumps({
        "ok": True,
        "architecture": "mot_context_mot_chuyen_gia_mot_policy",
        "source": str(ROOT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"V6 FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
