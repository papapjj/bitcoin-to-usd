from __future__ import annotations

from self_growing_trader.config import load_config
from self_growing_trader.he_chuyen_gia_ngu_canh import HeChuyenGiaMotContextMotPolicy


def test_cau_hinh_san_xuat_chi_con_mot_bo_nao_chung() -> None:
    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    assert config.he_chuyen_gia.che_do_quyet_dinh == "nam_tac_nhan_dung_chung"
    assert config.he_chuyen_gia.so_tac_nhan == 5
    assert config.he_chuyen_gia.so_chieu_nhan_dang_tac_nhan == 5
    assert config.he_chuyen_gia.context_history_inputs == 0
    assert config.he_chuyen_gia.entry_feature_enrichment_mode == "legacy"
    assert config.he_chuyen_gia.entry_style_specialists_bat is False
    assert config.he_chuyen_gia.position_management_specialist_bat is False
    assert config.he_chuyen_gia.volatile_subexperts_bat is False
    assert config.learning.timing_network_enabled is False
    assert config.learning.tensor_rule_enabled is False
    assert config.learning.direct_rnn_state_memory_enabled is False
    assert config.learning.direct_rnn_rl_auxiliary_enabled is False
    assert config.learning.direct_rnn_rl_entry_preference_enabled is False
    assert config.learning.direct_rnn_rl_retention_enabled is False


def test_so_dang_ky_chi_tao_mot_chu_so_huu_trong_so() -> None:
    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=config.he_chuyen_gia,
        tao_policy=lambda: object(),
    )
    assert list(system.experts) == ["CTX_CHUNG"]
    assert system.experts["CTX_CHUNG"].ma_chuyen_gia == "BO_NAO_CHUNG"


def test_so_tham_so_cua_loi_san_xuat() -> None:
    # Một bộ tham số duy nhất: xử lý quan sát + LSTM 4096 + các đầu hành động/giá trị.
    assert 85_153_426 == 236_704 + 83_918_848 + 997_874
