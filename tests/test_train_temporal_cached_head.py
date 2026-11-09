import torch
import json
import sys
import pytest
from scripts import train_temporal_cached_head as trainer
from scripts.train_temporal_cached_head import standardize_linear, fold_linear


def test_train_normalization_and_fold_preserve_linear_logits():
    torch.manual_seed(16)
    layer = torch.nn.Linear(24, 8)
    train = torch.randn(80, 24) * .01 + .03
    heldout = torch.randn(32, 24) * .02 - .01
    mean = train.mean(0)
    scale = train.std(0, unbiased=False).clamp_min(1e-3)
    before = layer(heldout).detach()
    standardize_linear(layer, mean, scale)
    torch.testing.assert_close(layer((heldout - mean) / scale), before)
    with torch.no_grad():
        layer.weight.add_(.02)
    learned = layer((heldout - mean) / scale).detach()
    fold_linear(layer, mean, scale)
    torch.testing.assert_close(layer(heldout), learned, atol=1e-5, rtol=1e-5)


def test_head_fit_rejects_config_not_used_for_feature_extraction(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("checkpoint", "config", "feature-cache", "audit-report")}
    for path in paths.values():
        path.write_bytes(b"test")
    paths["audit-report"].write_text(json.dumps({
        "checkpoint_sha256": trainer._sha256(paths["checkpoint"]),
        "feature_cache_sha256": trainer._sha256(paths["feature-cache"]),
        "config_sha256": "different_config_hash",
    }), encoding="utf-8")
    argv = ["head-fit"]
    for name, path in paths.items():
        argv.extend(["--" + name, str(path)])
    argv.extend(["--output", str(tmp_path / "candidate.pkl")])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="config differs"):
        trainer.main()
    assert not (tmp_path / "candidate.pkl").exists()
