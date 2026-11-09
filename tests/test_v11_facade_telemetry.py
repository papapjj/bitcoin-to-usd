from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia, HeChuyenGiaMotContextMotPolicy
from self_growing_trader.live_observatory import LiveObservabilityConfig


def _factory() -> ChuyenGiaPolicy:
    return ChuyenGiaPolicy.seed(seed=911)


def test_facade_emits_after_owner_metadata_are_attached(tmp_path: Path) -> None:
    cfg = CauHinhHeChuyenGia(bat=True)
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_factory)
    event_anchor = tmp_path / "outputs" / "quan_sat_chuyen_gia_live.jsonl"
    settings = LiveObservabilityConfig(enabled=True, event_path=str(event_anchor), scope_paths_by_run=True)
    system.enable_live_observatory(settings, run_id="root-v11", source="test")

    trace = system.act(np.zeros(119, dtype=float))
    context = dict(getattr(trace, "context_audit", {}) or {})
    assert context["ma_chuyen_gia"]
    assert "growth_review" not in dict(getattr(trace, "action_details", {}) or {})

    stream = tmp_path / "outputs" / "runs" / "root-v11" / "quan_sat_chuyen_gia_live.jsonl"
    decision_log = tmp_path / "outputs" / "runs" / "root-v11" / "quan_sat_chuyen_gia_live.ho_so_quyet_dinh.jsonl"
    assert stream.exists() and decision_log.exists()
    telemetry = [json.loads(line) for line in stream.read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = [row for row in telemetry if row.get("event_type") == "decision"]
    assert decisions
    audit = decisions[-1]["payload"]["context"]["audit"]
    assert audit["ma_chuyen_gia"] == context["ma_chuyen_gia"]
    assert audit["ma_policy"] == context["ma_policy"]

    profiles = [json.loads(line) for line in decision_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    profile = next(item for item in profiles if item.get("loai_ban_ghi") == "HO_SO_QUYET_DINH")
    assert not profile.get("growth_review")


def test_v11_topology_is_self_contained_for_every_decision(tmp_path: Path) -> None:
    cfg = CauHinhHeChuyenGia(bat=True)
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=cfg, tao_policy=_factory)
    settings = LiveObservabilityConfig(enabled=True, event_path=str(tmp_path / "outputs" / "neural.jsonl"), scope_paths_by_run=True)
    system.enable_live_observatory(settings, run_id="root-v11-repeat", source="test")
    for _ in range(4):
        system.act(np.zeros(119, dtype=float))
    development = tmp_path / "outputs" / "runs" / "root-v11-repeat" / "neural.theo_doi_phat_trien_mang.jsonl"
    rows = [json.loads(line) for line in development.read_text(encoding="utf-8").splitlines() if line.strip()]
    full = [row for row in rows if row.get("topology_day_du")]
    decision_full = [row for row in full if row.get("giai_doan") == "quyet_dinh"]
    # Live consumers can attach after initialization.  A decision record must
    # therefore be independently renderable instead of relying on an earlier
    # topology baseline in the same JSONL file.
    assert len(full) == len(rows)
    assert len(decision_full) >= 4
    assert all(row["topology_tham_chieu"]["snapshot_day_du_co_san"] for row in rows)
