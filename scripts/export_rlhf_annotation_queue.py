"""Export unlabeled causal RLHF annotation tasks from an MT5 observatory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from self_growing_trader.rlhf_annotation import build_rlhf_annotation_tasks


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid observatory JSON at line {line_number}") from exc
            if isinstance(row, dict):
                yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-pairs-per-context", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite only for an intentional new queue")
    tasks, stats = build_rlhf_annotation_tasks(
        _rows(args.input),
        max_pairs_per_context=args.max_pairs_per_context,
        max_tasks=args.max_tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "schema_version": "rlhf_annotation_queue_manifest_v1",
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "stats": stats,
        "promotion_allowed": False,
        "evidence_class": "rlhf_annotation_queue_only",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {"output": str(args.output), "manifest": str(manifest_path), **stats},
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
