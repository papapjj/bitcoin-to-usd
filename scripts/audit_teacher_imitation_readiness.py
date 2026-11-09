from __future__ import annotations

"""Read-only audit for the teacher-imitation deployment prerequisite.

This command deliberately does not create a checkpoint, mutate a seed, or
reserve training storage.  It makes the provenance and sidecar gates visible
before an expensive action-imitation run is started.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_tensor_rule_v186_candidate import (  # noqa: E402
    inspect_imitation_seed,
    sha256_file,
)
from self_growing_trader.five_teacher_imitation import (  # noqa: E402
    load_teacher_imitation_data,
)
from self_growing_trader.teacher_imitation_dataset import (  # noqa: E402
    CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION,
)


def audit_readiness(checkpoint: Path, dataset: Path) -> dict[str, object]:
    checkpoint = checkpoint.resolve()
    dataset = dataset.resolve()
    result: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "free_bytes": int(shutil.disk_usage(dataset.parent if dataset.parent.exists() else ROOT).free),
        "gates": {},
    }
    gates = result["gates"]
    assert isinstance(gates, dict)

    if not checkpoint.is_file():
        gates["checkpoint"] = {"passed": False, "reason": "checkpoint_missing"}
    else:
        digest = sha256_file(checkpoint)
        gates["checkpoint"] = {
            "passed": True,
            "sha256": digest,
            "bytes": checkpoint.stat().st_size,
        }
        try:
            seed_gate = inspect_imitation_seed(checkpoint, checkpoint_sha256=digest)
            gates["imitation_sidecar"] = seed_gate
        except Exception as exc:  # pragma: no cover - defensive audit boundary
            gates["imitation_sidecar"] = {
                "declared": False,
                "effective_for_mt5": False,
                "reason": f"sidecar_audit_error:{type(exc).__name__}",
                "error": str(exc),
            }

    manifest_path = dataset.with_suffix(dataset.suffix + ".json")
    if not dataset.is_file() or not manifest_path.is_file():
        gates["teacher_dataset"] = {
            "passed": False,
            "reason": "dataset_missing",
            "dataset_exists": dataset.is_file(),
            "manifest_exists": manifest_path.is_file(),
        }
    else:
        try:
            data = load_teacher_imitation_data(dataset)
            manifest = dict(data.manifest)
            schema = int(manifest.get("schema_version", 0) or 0)
            confirmation = dict(manifest.get("confirmation_audit", {}) or {})
            passed = bool(
                schema == CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION
                and confirmation.get("enabled", False)
            )
            gates["teacher_dataset"] = {
                "passed": passed,
                "reason": "ok" if passed else "wrong_teacher_contract",
                "schema_version": schema,
                "expected_schema_version": CONFIRMED_TEACHER_DATASET_SCHEMA_VERSION,
                "confirmation_audit_enabled": bool(confirmation.get("enabled", False)),
                "samples": int(len(data.actions)),
                "manifest": str(manifest_path),
            }
        except Exception as exc:
            gates["teacher_dataset"] = {
                "passed": False,
                "reason": "dataset_wrong_contract",
                "error": f"{type(exc).__name__}: {exc}",
                "manifest": str(manifest_path),
            }

    sidecar = dict(gates.get("imitation_sidecar", {}) or {})
    dataset_gate = dict(gates.get("teacher_dataset", {}) or {})
    result["ready_for_action_imitation"] = bool(dataset_gate.get("passed", False))
    result["ready_for_mt5_seed"] = bool(sidecar.get("effective_for_mt5", False))
    result["deployable"] = False
    result["blocking_reasons"] = [
        str(item.get("reason", "unknown"))
        for item in (sidecar, dataset_gate)
        if not bool(item.get("passed", item.get("effective_for_mt5", False)))
        and item.get("reason")
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit teacher imitation provenance without training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    result = audit_readiness(Path(args.checkpoint), Path(args.dataset))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready_for_action_imitation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
