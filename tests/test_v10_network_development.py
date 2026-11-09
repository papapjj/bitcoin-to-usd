from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia, HeChuyenGiaMotContextMotPolicy
from self_growing_trader.live_observatory import LiveObservabilityConfig, LiveTelemetryWriter
from self_growing_trader.nhat_ky_giai_thich import xay_ho_so_quyet_dinh


def _policy_factory() -> ChuyenGiaPolicy:
    return ChuyenGiaPolicy.seed(seed=71)


def _owner(system: HeChuyenGiaMotContextMotPolicy, trace):
    expert_id = str(getattr(trace, "context_audit", {}).get("ma_chuyen_gia", ""))
    return next(item for item in system.experts.values() if item.ma_chuyen_gia == expert_id)


def test_direct_m15_context_profile_keeps_real_unmasked_policy_decision() -> None:
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=_policy_factory,
    )
    trace = system.act(np.zeros(119, dtype=float))
    profile = xay_ho_so_quyet_dinh(system, trace, max_neurons=0, max_edges=0)

    assert profile["loai_ban_ghi"] == "HO_SO_QUYET_DINH"
    assert profile["giai_doan"] == "QUYET_DINH_OWNER"
    assert profile["neuron_chiu_trach_nhiem"] == []
    assert profile["lien_ket_chiu_trach_nhiem"] == []
    assert profile["quyet_dinh"]["hanh_dong_gui_mt5"] != "KHÔNG_GỬI_LENH"
    # A regularized unbiased seed may correctly choose HOLD. What matters is
    # that the owner profile exposes the actual unmasked policy decision.
    assert int(np.argmax(trace.selected_action[:12])) == int(trace.action_details["semantic_action"])



def test_owner_identity_includes_separate_memory_and_checkpoint_identity() -> None:
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=_policy_factory,
    )
    trace = system.act(np.zeros(119, dtype=float))
    active_owner = _owner(system, trace)
    other = next(item for item in system.experts.values() if item is not active_owner)
    active_identity = active_owner.policy._context_expert_owner_metadata
    other_identity = other.policy._context_expert_owner_metadata if hasattr(other.policy, "_context_expert_owner_metadata") else None

    # Policy instances and their memory identity must be independent. The other
    # owner gets provenance after its first decision, so compare object identity
    # directly here and full identity after the active owner has run.
    assert id(active_owner.policy) != id(other.policy)
    assert active_identity["memory_instance_uuid"]
    assert active_identity["policy_checkpoint_hash"]
    assert id(getattr(active_owner.policy, "memory_supernet")) != id(getattr(other.policy, "memory_supernet"))
    assert other_identity is None


def test_run_scoped_telemetry_publishes_active_run_pointer(tmp_path: Path) -> None:
    configured = tmp_path / "outputs" / "quan_sat_chuyen_gia_live.jsonl"
    cfg = LiveObservabilityConfig(
        enabled=True,
        event_path=str(configured),
        scope_paths_by_run=True,
        reset_on_process_start=True,
    )
    writer = LiveTelemetryWriter.from_config(cfg, run_id="root run: v10")
    policy = ChuyenGiaPolicy.seed(seed=17)
    writer.start(policy)

    event_path = Path(writer.path)
    assert event_path.parent.name == "root_run__v10"
    pointer = configured.parent / "active_run.json"
    assert pointer.exists()
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert Path(payload["event_path"]).resolve() == event_path.resolve()
    assert payload["root_run_id"] == "root run: v10"


def test_network_development_keeps_each_row_self_contained_with_full_topology(tmp_path: Path) -> None:
    event_path = tmp_path / "neural_live.jsonl"
    development_path = tmp_path / "network.jsonl"
    policy = ChuyenGiaPolicy.seed(seed=21)
    policy._context_expert_owner_metadata = {
        "ma_context": "CTX_TEST",
        "ma_chuyen_gia": "CG_TEST",
        "ma_policy": "POLICY_TEST",
        "context_on_dinh": True,
        "memory_instance_uuid": "MEM_TEST",
        "policy_checkpoint_hash": "CKPT_TEST",
    }
    writer = LiveTelemetryWriter(
        path=str(event_path),
        network_development_path=str(development_path),
        enabled=True,
        run_id="v10-network",
    )
    writer.start(policy)
    trace = policy.act(np.zeros(policy.input_size, dtype=float))
    trace.action_details.update({
        "semantic_action": 0,
        "emitted_action": 0,
        "wire_action": 0,
        "wire_matches_semantic_action": True,
    })
    writer.decision(policy, trace)
    writer.development(policy, trace, actions={"growth_review": {"ket_qua": "CHO_LIFECYCLE_XAC_NHAN"}})
    writer.development(policy, trace, actions={"growth_review": {"ket_qua": "CHO_LIFECYCLE_XAC_NHAN"}})

    rows = [json.loads(line) for line in development_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 2
    full = [row for row in rows if row.get("topology_day_du")]
    refs = [row for row in rows if row.get("topology_tham_chieu")]
    # Live tail/UI clients may begin at any file offset.  Every row therefore
    # carries a self-contained topology; the reference still marks whether the
    # topology is newly introduced or unchanged.
    assert len(full) == len(rows)
    assert refs
    assert all(row["topology_day_du"] for row in rows)
    assert sum(1 for row in refs if row["topology_tham_chieu"].get("topology_moi")) == 1
    assert all("policy_action" not in json.dumps(row, ensure_ascii=False) for row in rows)
    assert all("legal_action" not in json.dumps(row, ensure_ascii=False) for row in rows)
