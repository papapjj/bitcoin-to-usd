from types import SimpleNamespace

from self_growing_trader.cli import _require_context_expert_config


def test_cli_chap_nhan_che_do_nam_tac_nhan_dung_chung() -> None:
    config = SimpleNamespace(he_chuyen_gia=SimpleNamespace(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        cho_phep_tron_policy=False,
    ))
    _require_context_expert_config(config)
