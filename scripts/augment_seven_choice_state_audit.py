#!/usr/bin/env python3
"""Augment a seven-choice dataset with causal MT5 state snapshots.

The dense counterfactual exporter intentionally starts flat, so its position,
account and execution slots are zero.  This tool reuses only historical state
that was observed at or before the matching decision time.  The immutable
market/extra/timing fields remain from the current snapshot contract; dynamic
position, account and execution fields are rebuilt from the archived bridge
state and the matching snapshot price/ATR.  Rows with an open position or a
pending order are HOLD-only because the seven-choice contract has no
management action.

The input state audit must be an MT5 feature-audit TSV/CSV with
``bar_time``, ``positions_pipe`` and the account fields.  It is a research
augmentation: no future outcome is copied into an observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.protocol_contract import (
    ACCOUNT_OFFSET,
    EXECUTION_OFFSET,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    POSITION_FEATURE_COUNT,
    POSITION_OFFSET,
    TIMING_OFFSET,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_time(value: object) -> str:
    text = str(value or "").strip().replace("T", " ")
    match = re.match(
        r"^(\d{4})[./-](\d{2})[./-](\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?",
        text,
    )
    if not match:
        return ""
    return (
        f"{match.group(1)}-{match.group(2)}-{match.group(3)}T"
        f"{match.group(4)}:{match.group(5)}:{match.group(6) or '00'}"
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        return list(csv.DictReader(handle, delimiter=delimiter))


def _feature_schema_values(rows: list[dict[str, str]]) -> tuple[str, ...]:
    """Return the non-empty feature contract identifiers in an export."""
    return tuple(sorted({
        value
        for row in rows
        for value in (str(row.get("feature_schema_hash", "") or "").strip(),)
        # A trailing empty MQL5 row can be parsed as a field of NUL bytes.
        # It is malformed data, not a second feature contract.
        if value and len(value) <= 256 and value.isprintable()
    }))


def _feature_contract_report(
    snapshot_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    expected_feature_schema_hash: str | None,
) -> dict[str, Any]:
    """Describe whether the dense and state exports carry the same contract.

    Older MQL5 exporters wrote a human-readable label (for example
    ``five_observation_v10_137``), while the Python V193 audit writes the full
    fingerprint.  Missing or mixed identifiers therefore remain ``None`` when
    no expected hash is supplied; callers that need a training-grade artifact
    can require an explicit expected fingerprint.
    """
    snapshot_hashes = _feature_schema_values(snapshot_rows)
    audit_hashes = _feature_schema_values(audit_rows)
    expected = str(expected_feature_schema_hash or "").strip()
    hash_sets = {
        "snapshots": list(snapshot_hashes),
        "state_audit": list(audit_hashes),
    }
    if expected:
        missing = [name for name, values in hash_sets.items() if not values]
        mismatched = [
            f"{name}={value}"
            for name, values in hash_sets.items()
            for value in values
            if value != expected
        ]
        matched: bool | None = not missing and not mismatched
        reason = "matched" if matched else "; ".join(
            item for item in (
                f"missing={','.join(missing)}" if missing else "",
                f"mismatched={','.join(mismatched)}" if mismatched else "",
            ) if item
        )
    elif snapshot_hashes and audit_hashes:
        matched = snapshot_hashes == audit_hashes
        reason = "matched" if matched else "snapshot_and_audit_hashes_differ"
    else:
        matched = None
        reason = "schema_hash_missing_or_unverifiable"
    return {
        "feature_contract_match": matched,
        "feature_contract_match_reason": reason,
        "expected_feature_schema_hash": expected or None,
        "observed_feature_schema_hashes": hash_sets,
    }


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _positions(raw: object) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for item in str(raw or "").split("^"):
        parts = item.split(":")
        if len(parts) < 8 or not parts[0].strip():
            continue
        try:
            side = int(float(parts[1]))
            if side not in {-1, 1}:
                continue
            result.append({
                "side": side,
                "volume": float(parts[2]),
                "entry": float(parts[3]),
                "sl": float(parts[4]),
                "tp": float(parts[5]),
                "age_bars": float(parts[7]),
            })
        except (TypeError, ValueError):
            continue
    return result[:MAX_POSITION_SLOTS]


def _augment_observation(
    observation: np.ndarray,
    *,
    snapshot: dict[str, str],
    audit: dict[str, str],
    max_positions_reference: int,
) -> tuple[np.ndarray, bool, bool]:
    if observation.shape != (INPUT_COUNT,):
        raise ValueError(f"observation phai co {INPUT_COUNT} gia tri")
    result = np.asarray(observation, dtype=np.float32).copy()
    close = _float(snapshot, "close_bid")
    atr = max(_float(snapshot, "atr"), 1e-12)
    spread = max(0.0, _float(snapshot, "spread_price"))
    if not np.isfinite([close, atr, spread]).all() or close <= 0.0:
        raise ValueError("snapshot co close/atr/spread khong hop le")

    positions = _positions(audit.get("positions_pipe", ""))
    pending = any(item.strip() for item in str(audit.get("orders_pipe", "") or "").split("^"))
    result[POSITION_OFFSET:ACCOUNT_OFFSET] = 0.0
    for slot, position in enumerate(positions):
        offset = POSITION_OFFSET + slot * POSITION_FEATURE_COUNT
        side = int(position["side"])
        entry = float(position["entry"])
        sl = float(position["sl"])
        tp = float(position["tp"])
        risk_distance = max(abs(entry - sl), 1e-12)
        current_exit = close if side == 1 else close + spread
        price_move_atr = side * (close - entry) / atr
        age_norm = min(5.0, max(0.0, float(position["age_bars"])))
        sl_distance_atr = abs(entry - sl) / atr
        tp_distance_atr = abs(tp - entry) / atr
        rr_norm = min(tp_distance_atr / max(sl_distance_atr, 1e-12), 5.0) / 5.0
        unrealized_r = side * (current_exit - entry) / risk_distance
        result[offset:offset + POSITION_FEATURE_COUNT] = np.asarray((
            1.0,
            float(side),
            np.clip(price_move_atr, -20.0, 20.0),
            age_norm,
            max(0.0, sl_distance_atr),
            max(0.0, tp_distance_atr),
            max(0.0, rr_norm),
            np.clip(unrealized_r, -5.0, 5.0),
        ), dtype=np.float32)

    account = np.asarray(result[ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET], dtype=np.float32).copy()
    balance = _float(audit, "balance", 0.0)
    equity = _float(audit, "equity", balance)
    drawdown = _float(audit, "drawdown_pct", 0.0)
    if drawdown > 1.0:
        drawdown /= 100.0
    account[0] = len(positions) / max(1, int(max_positions_reference))
    account[1] = max(0.0, _float(audit, "open_risk_ratio", 0.0))
    account[2] = np.clip(drawdown, 0.0, 1.0)
    account[3] = np.clip((equity / balance) - 1.0, -1.0, 1.0) if balance > 0.0 else 0.0
    result[ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET] = account

    result[EXECUTION_OFFSET:TIMING_OFFSET] = 0.0
    result[EXECUTION_OFFSET] = spread / atr
    for slot, position in enumerate(positions):
        offset = EXECUTION_OFFSET + 1 + slot * 2
        entry = float(position["entry"])
        sl = float(position["sl"])
        tp = float(position["tp"])
        side = int(position["side"])
        exit_price = close if side == 1 else close + spread
        dist_sl = (exit_price - sl) / atr if side == 1 else (sl - exit_price) / atr
        dist_tp = (tp - exit_price) / atr if side == 1 else (exit_price - tp) / atr
        result[offset] = np.clip(dist_sl, -20.0, 20.0)
        result[offset + 1] = np.clip(dist_tp, -20.0, 20.0)
    return result, bool(positions), bool(pending)


def augment_dataset(
    dataset_path: Path,
    snapshots_path: Path,
    state_audit_path: Path,
    output_path: Path,
    *,
    max_positions_reference: int = MAX_POSITION_SLOTS,
    expected_feature_schema_hash: str | None = None,
    require_matching_feature_schema: bool = False,
) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    observations = np.asarray(arrays["observations"], dtype=np.float32)
    sample_times = np.asarray(arrays["sample_times"]).astype(str)
    if observations.ndim != 2 or observations.shape[1] != INPUT_COUNT:
        raise ValueError("dataset khong dung hop dong quan sat 137")
    snapshot_source_rows = _read_tsv(snapshots_path)
    audit_source_rows = _read_tsv(state_audit_path)
    contract_report = _feature_contract_report(
        snapshot_source_rows,
        audit_source_rows,
        expected_feature_schema_hash,
    )
    if require_matching_feature_schema and contract_report["feature_contract_match"] is not True:
        raise ValueError(
            "dense snapshot va state audit khong cung hop dong dac trung: "
            f"{contract_report['feature_contract_match_reason']}"
        )
    snapshot_rows = {_normalise_time(row.get("sample_time")): row for row in snapshot_source_rows}
    audit_rows = {_normalise_time(row.get("bar_time")): row for row in audit_source_rows}
    matched = 0
    active = 0
    position_rows = 0
    pending_rows = 0
    unavailable = 0
    seen_times: set[str] = set()
    for time_key in sorted(set(sample_times.tolist())):
        snapshot = snapshot_rows.get(time_key)
        audit = audit_rows.get(time_key)
        if snapshot is None or audit is None:
            unavailable += 1
            continue
        indices = np.flatnonzero(sample_times == time_key)
        if len(indices) != 2:
            raise ValueError(f"sample_time {time_key} phai co dung hai vai")
        augmented, has_position, has_pending = _augment_observation(
            observations[indices[0]],
            snapshot=snapshot,
            audit=audit,
            max_positions_reference=max_positions_reference,
        )
        observations[indices] = augmented
        matched += 1
        seen_times.add(time_key)
        if has_position:
            position_rows += 1
        if has_pending:
            pending_rows += 1
        if has_position or has_pending:
            active += 1
            arrays["allowed_action_mask"][indices] = False
            arrays["allowed_action_mask"][indices, 0] = True
            arrays["best_actions"][indices] = 0
            arrays["outcome_best_actions"][indices] = 0
            arrays["causal_action_scores_r"][indices] = -5.0
            arrays["causal_action_scores_r"][indices, 0] = 0.0
    arrays["observations"] = observations
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["state_augmentation"] = {
        "version": 1,
        "source_dataset": str(dataset_path.resolve()),
        "source_dataset_sha256": _sha256(dataset_path),
        "snapshot_source": str(snapshots_path.resolve()),
        "snapshot_source_sha256": _sha256(snapshots_path),
        "state_audit_source": str(state_audit_path.resolve()),
        "state_audit_source_sha256": _sha256(state_audit_path),
        "matched_unique_times": matched,
        "matched_fraction": matched / max(1, len(set(sample_times.tolist()))),
        "state_position_rows": position_rows,
        "state_pending_rows": pending_rows,
        "state_active_rows_hold_only": active,
        "unavailable_unique_times": unavailable,
        "dynamic_slots_recomputed": True,
        "immutable_market_and_extra_slots_preserved": True,
        "causal_state_source": True,
        "active_state_label_policy": "HOLD_only_until_management_action_contract_exists",
        **contract_report,
        "research_only": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    output_manifest = output_path.with_suffix(output_path.suffix + ".json")
    manifest["output"] = str(output_path.resolve())
    manifest["output_sha256"] = _sha256(output_path)
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest["state_augmentation"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Them trang thai MT5 nhan qua vao bo nhan bay lua chon.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--state-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-positions-reference", type=int, default=MAX_POSITION_SLOTS)
    parser.add_argument(
        "--expected-feature-schema-hash",
        default="",
        help="Ma bam hop dong day du phai co trong ca snapshot va state audit.",
    )
    parser.add_argument(
        "--require-matching-feature-schema",
        action="store_true",
        help="Dung loi neu hai nguon khong xac nhan cung hop dong dac trung.",
    )
    args = parser.parse_args()
    if args.max_positions_reference < 1:
        raise ValueError("max-positions-reference phai duong")
    report = augment_dataset(
        args.dataset.resolve(), args.snapshots.resolve(), args.state_audit.resolve(),
        args.output.resolve(),
        max_positions_reference=args.max_positions_reference,
        expected_feature_schema_hash=args.expected_feature_schema_hash,
        require_matching_feature_schema=args.require_matching_feature_schema,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
