#!/usr/bin/env python3
"""Read-only inventory for RLHF/MT5 artifacts before a matrix run.

The command never deletes files, stops MT5, or rewrites manifests.  It only
classifies files so a human can explicitly decide which forensic artifacts may
be archived or removed before the next 3-window x 3-seed matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def classify_rlhf_artifact(path: str | Path) -> str:
    """Return a conservative retention class for one RLHF artifact."""
    item = Path(path)
    name = item.name.lower()
    parts = {part.lower() for part in item.parts}
    if name.endswith(".incomplete.pkl"):
        return "incomplete_checkpoint_for_review"
    if "execution_trace_observatory" in name:
        return "observatory_for_review"
    if "runtime" in parts or name in {"bridge.csv", "request.csv", "action.csv"}:
        return "runtime_transport"
    if name == "summary.json" or name.endswith(".manifest.json"):
        return "matrix_manifest"
    if item.suffix.lower() in {".json", ".jsonl", ".htm", ".html", ".csv"}:
        return "evidence_or_report"
    if item.suffix.lower() == ".pkl":
        return "checkpoint_for_review"
    return "other"


def _existing_anchor(path: Path) -> Path:
    anchor = path.resolve()
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    return anchor


def audit_rlhf_storage(
    root: str | Path,
    *,
    required_free_bytes: int = 0,
) -> dict[str, object]:
    """Inventory one artifact root without mutating it."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"RLHF storage root not found: {root_path}")
    entries: list[dict[str, object]] = []
    for item in sorted(root_path.rglob("*")):
        if not item.is_file():
            continue
        try:
            size = int(item.stat().st_size)
        except OSError:
            continue
        entries.append({
            "path": str(item.relative_to(root_path)),
            "bytes": size,
            "category": classify_rlhf_artifact(item),
        })
    by_category: dict[str, dict[str, int]] = {}
    for entry in entries:
        category = str(entry["category"])
        bucket = by_category.setdefault(category, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += int(entry["bytes"])
    anchor = _existing_anchor(root_path)
    free_bytes = int(shutil.disk_usage(anchor).free)
    required = max(0, int(required_free_bytes))
    return {
        "root": str(root_path),
        "total_files": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "categories": by_category,
        "free_bytes": free_bytes,
        "required_free_bytes": required,
        "free_floor_passed": bool(not required or free_bytes >= required),
        "mutations_performed": False,
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/rlhf"))
    parser.add_argument("--required-free-bytes", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = audit_rlhf_storage(
            args.root,
            required_free_bytes=args.required_free_bytes,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "mutations_performed": False}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
