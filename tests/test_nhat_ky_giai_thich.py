from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from self_growing_trader.graph_development import (
    ACTION_GROW_NEURON,
    GraphDevelopmentController,
    GraphStructuralAction,
)
from self_growing_trader.live_observatory import LiveObservabilityConfig, LiveTelemetryWriter
from self_growing_trader.module_need import ROLE_PREDICTOR_SPECIALIST
from self_growing_trader.neural_graph import EDGE_PREDICTIVE, REGION_PREDICTOR
from self_growing_trader.nhat_ky_giai_thich import xay_ho_so_quyet_dinh
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def _decision_context(policy: SelfGrowingPolicy, frame_id: str) -> None:
    policy.set_decision_context(
        account_context={"current_equity_drawdown": 0.01, "open_risk_ratio": 0.0},
        positions_pipe="",
        allowed_actions={0: True, 1: True, 2: True},
        execution_context={"spread_norm": 0.1, "margin_available_ratio": 1.0},
        goal=np.asarray([0.1, 0.2, 0.0, 0.0], dtype=float),
        sequence_id=1,
        frame_id=frame_id,
        timestamp=1_700_000_000.0,
    )


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_ho_so_quyet_dinh_contains_context_policy_neurons_edges_and_reasons(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(seed=411)
    event_path = tmp_path / "neural_live.jsonl"
    decision_path = tmp_path / "ho_so_quyet_dinh.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(
            enabled=True,
            event_path=str(event_path),
            decision_profile_path=str(decision_path),
            max_responsible_neurons=20,
            max_responsible_edges=30,
            reset_on_process_start=True,
        ),
        run_id="test-ho-so-quyet-dinh",
    )
    policy._context_expert_owner_metadata = {
        "ma_context": "CTX_KIEM_THU_VAO_LENH",
        "ma_context_cha": "CTX_GOC_VAO_LENH_KHONG_CO_LENH",
        "ma_chuyen_gia": "CG_CTX_KIEM_THU",
        "ma_policy": "POLICY_CTX_KIEM_THU_V001",
        "phien_ban_policy": "V001",
        "trang_thai_chuyen_gia": "HOAT_DONG_HAN_CHE",
        "context_on_dinh": True,
        "do_tin_cay_context": 0.91,
        "ly_do_context": "kiem thu ownership truoc telemetry",
    }
    _decision_context(policy, "kiem-thu:quyet-dinh")
    trace = policy.act(np.zeros(policy.input_size, dtype=float))

    rows = _rows(decision_path)
    assert len(rows) == 1
    profile = rows[0]
    assert profile["loai_ban_ghi"] == "HO_SO_QUYET_DINH"
    assert profile["schema_version"] >= 4
    assert profile["trace_id"] == trace.trace_id
    assert profile["context"]["ma_context"] == "CTX_KIEM_THU_VAO_LENH"
    assert profile["context"]["ma_chuyen_gia"] == "CG_CTX_KIEM_THU"
    assert profile["context"]["ma_policy"] == "POLICY_CTX_KIEM_THU_V001"
    assert any(reason["ma"] == "CONTEXT_OWNER" for reason in profile["ly_do"])
    assert profile["neuron_chiu_trach_nhiem"]
    assert profile["lien_ket_chiu_trach_nhiem"]
    assert len(profile["neuron_chiu_trach_nhiem"]) <= 20
    assert len(profile["lien_ket_chiu_trach_nhiem"]) <= 30
    assert {"id", "module", "region", "activation", "tong_dong_gop_tuyet_doi"} <= set(profile["neuron_chiu_trach_nhiem"][0])
    assert {"id", "src", "dst", "contribution", "gate", "weight"} <= set(profile["lien_ket_chiu_trach_nhiem"][0])
    assert profile["dau_vao_anh_huong"]
    assert profile["dau_vao_anh_huong"][0]["ten"].startswith(("thi_truong.", "vi_the.", "tai_khoan.", "thuc_thi."))
    assert {
        "hidden_normalization_mode",
        "hidden_normalization_enabled",
        "hidden_dropout_enabled",
        "hidden_dropout_rates",
        "hidden_dropout_training_active",
        "hidden_dropout_steps_before",
        "hidden_dropout_steps_advanced",
        "output_normalization_mode",
    } <= set(profile["regularization"])


def test_ho_so_quyet_dinh_never_treats_hypothesis_index_as_mt5_action() -> None:
    policy = SelfGrowingPolicy.seed(seed=421)
    trace = SimpleNamespace(
        trace_id="trace-hypothesis-not-action",
        context_frame_id="frame-hypothesis-not-action",
        selected_idx=5,
        selected_action=np.zeros(20, dtype=float),
        action_details={},
        context_audit={
            "ma_context": "CTX_KIEM_THU_ACTION_NAMESPACE",
            "ma_chuyen_gia": "CG_CTX_KIEM_THU_ACTION_NAMESPACE",
            "ma_policy": "POLICY_CTX_KIEM_THU_ACTION_NAMESPACE_V001",
            "context_on_dinh": True,
            "do_tin_cay_context": 0.9,
        },
        active_neuron_ids=(),
        active_edge_ids=(),
        graph_traces={},
        confidence=0.0,
        critic_value=0.0,
    )

    profile = xay_ho_so_quyet_dinh(policy, trace)

    decision = profile["quyet_dinh"]
    assert decision["model_hypothesis_index"] == 5
    # V9 cấm ghi các trường legacy policy/legal với sentinel -1.
    assert "hanh_dong_policy" not in decision
    assert "hanh_dong_hop_le" not in decision
    assert "policy_action_index" not in decision
    assert "legal_action_index" not in decision
    assert decision["semantic_action_index"] is None
    assert decision["emitted_action_index"] is None
    assert decision["wire_action_index"] is None
    assert decision["hanh_dong_nghia_vu"] == "KHÔNG_GỬI_LENH"
    assert decision["hanh_dong_gui_mt5"] == "KHÔNG_GỬI_LENH"


def test_ho_so_quyet_dinh_exposes_action_surface_source_without_changing_policy() -> None:
    policy = SelfGrowingPolicy.seed(seed=423)
    trace = SimpleNamespace(
        trace_id="trace-action-surface-audit",
        context_frame_id="frame-action-surface-audit",
        selected_idx=2,
        selected_action=np.zeros(20, dtype=float),
        # Replay/bridge paths may carry a NumPy mask; audit code must never
        # evaluate its multi-element truthiness.
        allowed_action_mask=np.asarray(
            (True, False, True, False, False, False, False), dtype=bool
        ),
        action_details={
            "semantic_action": 2,
            "emitted_action": 2,
            "wire_action": 2,
            "wire_matches_semantic_action": True,
            "five_selected_action_probability": 0.37,
            "five_selected_behavior_action_probability": 0.21,
            "five_live_action_veto_vetoed_actions": [1, 3],
            "five_expectancy_vetoed_actions": [4, 5, 6],
            "five_expectancy_probe_preserved_actions": [2],
            "five_expectancy_context_escape_actions": [5],
            "five_live_action_veto_probe_due": True,
            "action_origin": "policy",
            "action_masked": False,
        },
        context_audit={
            "ma_context": "CTX_ACTION_SURFACE",
            "ma_chuyen_gia": "CG_ACTION_SURFACE",
            "ma_policy": "POLICY_ACTION_SURFACE",
            "context_on_stable": True,
            "context_on_dinh": True,
            "selector_reason": "khong_dung_bo_chon",
        },
        active_neuron_ids=(),
        active_edge_ids=(),
        graph_traces={},
        confidence=0.8,
        critic_value=0.1,
    )

    profile = xay_ho_so_quyet_dinh(policy, trace)
    audit = profile["quyet_dinh"]["action_surface_audit"]

    assert audit["allowed_action_indices"] == [0, 2]
    assert audit["allowed_action_count"] == 2
    assert audit["mask_only_hold"] is False
    assert audit["selected_action_index"] == 2
    assert audit["selected_action_probability"] == 0.37
    assert audit["live_vetoed_actions"] == [1, 3]
    assert audit["expectancy_vetoed_actions"] == [4, 5, 6]
    assert audit["probe_preserved_actions"] == [2]
    assert audit["context_escape_actions"] == [5]
    assert audit["live_probe_due"] is True
    assert audit["neural_command"] is True


def test_step_trace_selected_action_is_final_policy_output() -> None:
    policy = SelfGrowingPolicy.seed(seed=422)
    _decision_context(policy, "kiem-thu:final-action-output")
    trace = policy.act(np.zeros(policy.input_size, dtype=float))

    final_output = np.asarray(trace.action_details["final_action_output"], dtype=float)
    np.testing.assert_allclose(trace.selected_action, final_output)
    assert trace.action_details["model_hypothesis_index"] == trace.selected_idx
    assert trace.action_details["policy_output_action_index"] == int(np.argmax(final_output[:12]))


def test_nhat_ky_cau_truc_keeps_reason_and_created_neuron_edge_snapshots(tmp_path) -> None:
    # The production default is direct_rnn; this structural telemetry test
    # exercises the opt-in graph backend explicitly.
    policy = SelfGrowingPolicy.seed(seed=412, m1_backend="m1_graph")
    graph = policy.modules[0].m1_component.graph
    controller = GraphDevelopmentController(rng=np.random.default_rng(7))
    event = controller.apply_action(
        graph,
        GraphStructuralAction(
            action=ACTION_GROW_NEURON,
            role=ROLE_PREDICTOR_SPECIALIST,
            region=REGION_PREDICTOR,
            edge_kind=EDGE_PREDICTIVE,
            reason="novelty_cao_va_sai_so_du_doan_lap_lai",
            score=0.83,
            metadata={
                "module_need_scores": {ROLE_PREDICTOR_SPECIALIST: 0.83},
                "neural_logits": {ROLE_PREDICTOR_SPECIALIST: 0.77},
            },
        ),
    )
    structural_path = tmp_path / "nhat_ky_cau_truc.jsonl"
    writer = LiveTelemetryWriter(
        path=str(tmp_path / "neural_live.jsonl"),
        structural_event_path=str(structural_path),
        enabled=True,
    )
    trace = SimpleNamespace(
        trace_id="trace-cau-truc-01",
        context_audit={
            "ma_context": "CTX_KIEM_THU_CAU_TRUC",
            "ma_chuyen_gia": "CG_CTX_KIEM_THU_CAU_TRUC",
            "ma_policy": "POLICY_CTX_KIEM_THU_CAU_TRUC_V001",
            "context_on_dinh": True,
            "do_tin_cay_context": 0.88,
        },
    )
    logs = writer.nhat_ky_cau_truc(policy, trace, [event], giai_doan="kiem_thu")

    assert len(logs) == 1
    rows = _rows(structural_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["loai_ban_ghi"] == "SU_KIEN_PHAT_TRIEN_CAU_TRUC"
    assert row["trace_id"] == "trace-cau-truc-01"
    assert row["context"]["ma_context"] == "CTX_KIEM_THU_CAU_TRUC"
    assert row["su_kien"]["hanh_dong"] == ACTION_GROW_NEURON
    assert row["su_kien"]["ly_do"] == "novelty_cao_va_sai_so_du_doan_lap_lai"
    assert row["su_kien"]["ma_su_kien"] == event.event_id
    assert row["su_kien"]["la_moc_cau_truc"] is True
    assert row["su_kien"]["phan_loai_su_kien"] == "moc_cau_truc"
    assert row["su_kien"]["neuron_duoc_tao"]
    assert row["su_kien"]["neuron_duoc_tao"][0]["id"] == event.target
    assert row["su_kien"]["lien_ket_duoc_tao"]
    assert isinstance(row["su_kien"]["lien_ket_duoc_tao"][0], dict)
    assert row["nguon_kich_hoat"]["module_need_scores"][ROLE_PREDICTOR_SPECIALIST] == 0.83


def test_graph_credit_attribution_empty_is_not_reported_as_growth(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(seed=423)
    structural_path = tmp_path / "nhat_ky_cau_truc.jsonl"
    writer = LiveTelemetryWriter(
        path=str(tmp_path / "neural_live.jsonl"),
        structural_event_path=str(structural_path),
        enabled=True,
    )
    trace = SimpleNamespace(trace_id="trace-credit-only", context_audit={})
    event = {
        "type": "graph_credit_attribution",
        "before_neurons": 0,
        "after_neurons": 0,
        "before_edges": 0,
        "after_edges": 0,
        "details": {"module_need_scores": {}, "causal_confidence": 0},
    }

    writer.nhat_ky_cau_truc(policy, trace, [event], giai_doan="feedback")

    # V9 chỉ ghi structural journal khi topology thực sự đổi. Credit rỗng
    # không được tạo file/event giả như một lần mọc neuron/liên kết.
    assert not structural_path.exists()


def test_context_chua_on_dinh_van_duoc_log_la_entry_hoc(tmp_path) -> None:
    from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia, HeChuyenGiaMotContextMotPolicy

    class GiaPolicyToiGian:
        def __init__(self):
            self.calls = 0

        def act(self, observation, goal=None):
            self.calls += 1
            output = np.zeros(20, dtype=float)
            output[1] = 1.0
            return SimpleNamespace(
                trace_id=f"trace-candidate-{self.calls}",
                selected_action=output,
                selected_idx=1,
                confidence=1.0,
                action_details={},
                context_audit={},
            )

    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=GiaPolicyToiGian,
    )
    decision_path = tmp_path / "ho_so_no_policy.jsonl"
    system.enable_live_observatory(
        LiveObservabilityConfig(
            enabled=True,
            event_path=str(tmp_path / "neural_live.jsonl"),
            decision_profile_path=str(decision_path),
            reset_on_process_start=True,
        ),
        run_id="test-context-no-policy",
    )
    trace = system.act(np.zeros(119, dtype=float))

    assert getattr(trace, "ly_do_khong_giao_dich", "") == ""
    assert getattr(trace, "ly_do_khong_hoc", False) is False
    profile = _rows(decision_path)[0]
    assert profile["context"]["ma_context"]
    assert profile["context"]["ly_do_khong_giao_dich"] == ""
    assert profile["context"]["audit_day_du"]["khong_co_policy_ra_lenh"] is False
    assert profile["hop_dong_owner"]["ly_do_chan"] != "CONTEXT_CHUA_ON_DINH"


def test_ho_so_quyet_dinh_limit_zero_keeps_all_attribution() -> None:
    """V9: quota 0 không được cắt neuron, cạnh hoặc feature attribution."""
    policy = SelfGrowingPolicy.seed(seed=423)
    policy._context_expert_owner_metadata = {
        "ma_context": "CTX_KIEM_THU_DAY_DU",
        "ma_chuyen_gia": "CG_CTX_KIEM_THU_DAY_DU",
        "ma_policy": "POLICY_CTX_KIEM_THU_DAY_DU_V001",
        "context_on_dinh": True,
        "do_tin_cay_context": 0.95,
    }
    _decision_context(policy, "kiem-thu:attribution-day-du")
    trace = policy.act(np.zeros(policy.input_size, dtype=float))
    profile = xay_ho_so_quyet_dinh(policy, trace, max_neurons=0, max_edges=0)

    # Seed graph có nhiều hơn các quota V8 từng dùng; V9 phải giữ toàn bộ trace.
    assert len(profile["neuron_chiu_trach_nhiem"]) > 32
    assert len(profile["lien_ket_chiu_trach_nhiem"]) > 48
    # Hidden regularization changes attribution sparsity but quota zero must
    # still preserve every available feature attribution.
    assert len(profile["dau_vao_anh_huong"]) >= 10
