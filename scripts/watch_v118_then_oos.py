"""Wait for the atomic v118 checkpoint, then run isolated paired OOS once."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / os.environ.get(
    "V13_V118_CANDIDATE",
    "outputs/he_chuyen_gia/candidates/v118_seed_bounded_scale0001_realticks.pkl",
)
REGISTRY = CANDIDATE.with_suffix(".registry.json")
OUT = ROOT / os.environ.get(
    "V13_V118_OOS_OUT",
    "outputs/he_chuyen_gia/audit/v118_seed_bounded_scale0001_realticks_oos",
)
STATUS = OUT / "oos_monitor_status.json"
LOCK = OUT / ".oos_monitor.lock"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_status(**values: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        lock_handle = LOCK.open("x", encoding="utf-8")
    except FileExistsError:
        return 0
    with lock_handle:
        _write_status(status="waiting_for_atomic_candidate", candidate=str(CANDIDATE), started_at=time.time())
        previous: tuple[int, int] | None = None
        while True:
            if CANDIDATE.is_file() and REGISTRY.is_file():
                current = (CANDIDATE.stat().st_size, CANDIDATE.stat().st_mtime_ns)
                if current == previous and current[0] > 0:
                    break
                previous = current
            time.sleep(30)
        candidate_hash = _sha256(CANDIDATE)
        _write_status(status="launching_isolated_oos", candidate_sha256=candidate_hash, launched_at=time.time())
        log = OUT / "isolated_oos.stdout.log"
        error = OUT / "isolated_oos.stderr.log"
        env = os.environ.copy()
        env["V13_V118_CANDIDATE"] = str(CANDIDATE)
        env["V13_V118_OOS_OUT"] = str(OUT)
        with log.open("a", encoding="utf-8") as stdout, error.open("a", encoding="utf-8") as stderr:
            result = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts/run_v118_isolated_oos.py")],
                cwd=str(ROOT),
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        _write_status(status="oos_finished", returncode=result.returncode, candidate_sha256=candidate_hash, finished_at=time.time())
        return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
