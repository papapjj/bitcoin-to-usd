from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from self_growing_trader.cli import build_parser
from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    ArtifactNgoaiHopDongBiCam,
    tai_he_chuyen_gia_mot_context_mot_policy,
    tao_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.he_chuyen_gia_ngu_canh import QuyetDinhKhongGiaoDich
from self_growing_trader.mt5_runner import MT5StrategyTesterEvaluator
from self_growing_trader.nhat_ky_giai_thich import xay_ho_so_quyet_dinh


def _config():
    return load_config("configs/he_chuyen_gia.yaml")


def test_cli_keeps_context_owner_commands_and_explicit_compatibility_aliases():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    commands = set(subparsers.choices)
    assert {
        "khoi-tao-he-chuyen-gia",
        "huan-luyen-he-chuyen-gia-mt5",
        "walk-forward-he-chuyen-gia-mt5",
        "quan-tri-he-chuyen-gia",
        "live-observatory",
    }.issubset(commands)
    # Legacy/continuous commands are retained as explicit compatibility
    # namespaces.  They do not enter the context-owner MT5 evaluator path.
    assert {"train-mt5", "improve", "compare-cores", "continuous-train"}.issubset(commands)
    with pytest.raises(SystemExit):
        parser.parse_args(["lenh-khong-thuoc-contract"])


def test_config_enforces_context_owner_contract_with_owner_safe_lifecycle_config():
    config = _config()
    assert hasattr(config, "continuous")
    assert config.he_chuyen_gia.bat is True
    assert config.he_chuyen_gia.cho_phep_policy_chung_ra_lenh is False
    assert config.he_chuyen_gia.cho_phep_tron_policy is False


def test_context_and_role_specialists_have_distinct_policy_instances():
    system = tao_he_chuyen_gia_mot_context_mot_policy(_config(), seed=41)
    assert len(system.experts) == 9
    policies = [expert.policy for expert in system.experts.values()]
    assert len({id(policy) for policy in policies}) == 9

    system.act(np.zeros(119, dtype=float))
    assert len(system.experts) == 12
    policies = [expert.policy for expert in system.experts.values()]
    assert len({id(policy) for policy in policies}) == 12


def test_direct_m15_context_emits_owner_entry_without_confirmation_delay():
    system = tao_he_chuyen_gia_mot_context_mot_policy(_config(), seed=7)
    obs = np.zeros(119, dtype=float)
    first = system.act(obs)
    assert not isinstance(first, QuyetDinhKhongGiaoDich)
    assert getattr(first, "ly_do_khong_hoc", False) is False
    assert first.context_audit["context_on_dinh"] is True
    assert first.context_audit["entry_actions_available_in_selector"] is True
    proposal_union = set().union(*(
        set(item["allowed_actions"])
        for item in first.context_audit["selector_proposals"]
        if "ENTRY" in item["role"]
    ))
    assert set(range(1, 7)).issubset(proposal_union)
    valid, reason = MT5StrategyTesterEvaluator._context_expert_owner_valid(system, first)
    assert valid is True, reason
    audit = dict(getattr(first, "context_audit", {}) or {})
    assert audit["context_on_dinh"] is True
    assert audit["ma_context"]
    assert audit["ma_chuyen_gia"]
    assert audit["ma_policy"]
    valid, reason = MT5StrategyTesterEvaluator._context_expert_owner_valid(system, first)
    assert valid is True, reason


def test_decision_profile_never_calls_unowned_trace_an_mt5_order():
    trace = QuyetDinhKhongGiaoDich(
        trace_id="khong-co-owner",
        selected_action=np.r_[np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.zeros(15)],
        selected_idx=4,
        context_audit={"context_on_dinh": False},
        action_details={"policy_action": 4, "legal_action": 4, "emitted_action": 4},
    )
    profile = xay_ho_so_quyet_dinh(object(), trace)
    assert profile["hop_dong_owner"]["hop_le"] is False
    assert profile["quyet_dinh"]["hanh_dong_gui_mt5"] == "KHÔNG_GỬI_LENH"
    assert "Không gửi lệnh MT5" in profile["tom_tat_tieng_viet"]


def test_out_of_contract_artifact_is_rejected(tmp_path):
    path = tmp_path / "ngoai_hop_dong.pkl"
    with path.open("wb") as handle:
        pickle.dump({"khong_thuoc_contract": True}, handle)
    with pytest.raises(TypeError):
        tai_he_chuyen_gia_mot_context_mot_policy(path, _config(), seed=3)


def test_module_cli_launcher_works_with_runtime_pythonpath():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    source = str(root / "src")
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = source if not prior else source + os.pathsep + prior
    result = subprocess.run(
        [sys.executable, "-m", "self_growing_trader.cli", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "huan-luyen-he-chuyen-gia-mt5" in result.stdout


def test_context_owner_runtime_uses_local_selector_not_legacy_or_continuous_paths():
    root = Path(__file__).resolve().parents[1]
    # The policy facade/MT5 evaluator must retain ownership routing locally.
    # Compatibility modules can exist in the repository, but no production
    # context-owner path may invoke their generic routing entry points.
    scan_roots = (
        root / "src" / "self_growing_trader" / "chuyen_gia_policy.py",
        root / "src" / "self_growing_trader" / "he_chuyen_gia_ngu_canh.py",
        root / "src" / "self_growing_trader" / "mt5_runner.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in scan_roots)
    assert "continuous_train_mt5(" not in text
    assert "compare_policy_cores_mt5(" not in text
    assert "PolicyEnsembleSelector(" in text
    assert "FIXED_CONTEXT_OWNER_V32" in text


def test_bridge_refuses_any_object_outside_context_owner_contract():
    evaluator = object.__new__(MT5StrategyTesterEvaluator)
    with pytest.raises(TypeError):
        evaluator.publish_policy(object())
