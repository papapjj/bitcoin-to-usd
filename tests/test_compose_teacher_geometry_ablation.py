import torch

from scripts.compose_teacher_geometry_ablation import compose_sl_only, _head_sha256
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy


def test_compose_sl_only_preserves_every_other_factorized_head() -> None:
    torch.manual_seed(41)
    base = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    donor = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    before = {name: _head_sha256(base, name) for name in ("delay", "sl", "tp", "trailing")}
    donor_sl = _head_sha256(donor, "sl")

    result = compose_sl_only(base, donor)

    assert _head_sha256(base, "sl") == donor_sl
    assert result["donor_sl_sha256"] == donor_sl
    for name in ("delay", "tp", "trailing"):
        assert _head_sha256(base, name) == before[name]
    assert base.heads.teacher_geometry_imitation_heads == ("sl",)
