from __future__ import annotations

import json
import pickle
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from self_growing_trader.he_chuyen_gia_ngu_canh import (
    BoNhanDienNguCanh,
    CauHinhHeChuyenGia,
    HeChuyenGiaMotContextMotPolicy,
    NhiemVuNguCanh,
    TrangThaiChuyenGia,
    _policy_identity,
)
from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.input_normalization import normalize_input_vector


@dataclass
class GiaTrace:
    trace_id: str
    confidence: float = 0.5
    selected_action: np.ndarray | None = None
    context_audit: dict[str, object] | None = None


@dataclass(frozen=True)
class FrozenTrace:
    trace_id: str


class GiaPolicy:
    dem = 0

    def __init__(self) -> None:
        self.ten = f"P{GiaPolicy.dem}"
        GiaPolicy.dem += 1
        self.act_calls = 0
        self.transition_calls = 0

    def act(self, observation, goal=None):
        self.act_calls += 1
        outputs = np.zeros(20, dtype=float)
        outputs[1] = 1.0
        outputs[12] = 999.0
        return GiaTrace(trace_id=f"{self.ten}-{self.act_calls}", selected_action=outputs, context_audit={})

    def observe_transition(self, trace, observation, reward, cost, **kwargs):
        self.transition_calls += 1
        return {"reward": reward, "cost": cost}


class MutatingPolicy(GiaPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.bias = 0.0
        self.credit = 0

    def observe_transition(self, trace, observation, reward, cost, **kwargs):
        self.transition_calls += 1
        self.bias += float(reward) - float(cost)
        return {"reward": reward, "cost": cost, "bias": self.bias}

    def apply_execution_lifecycle_credit(self, trace_id, realized_r=0.0, **kwargs):
        self.credit += 1
        self.bias += float(realized_r)
        return 1


class CanonicalOnlyFullRlPolicy(MutatingPolicy):
    def _direct_rnn_full_rl_mode_enabled(self):
        return True

    def apply_execution_exit_credit(self, trace_id, **kwargs):
        self.exit_credit = int(getattr(self, "exit_credit", 0)) + 1
        return 1


class CloseWorstPolicy(GiaPolicy):
    def act(self, observation, goal=None):
        self.act_calls += 1
        outputs = np.zeros(20, dtype=float)
        outputs[8] = 1.0
        outputs[12] = 999.0
        return GiaTrace(trace_id=f"{self.ten}-{self.act_calls}", selected_action=outputs, context_audit={})


class HoldPolicy(GiaPolicy):
    def act(self, observation, goal=None):
        self.act_calls += 1
        outputs = np.zeros(20, dtype=float)
        outputs[0] = 1.0
        outputs[12] = 999.0
        return GiaTrace(trace_id=f"{self.ten}-{self.act_calls}", selected_action=outputs, context_audit={})


class ManagePolicy(GiaPolicy):
    def act(self, observation, goal=None):
        self.act_calls += 1
        outputs = np.zeros(20, dtype=float)
        outputs[11] = 1.0
        outputs[12] = 999.0
        return GiaTrace(trace_id=f"{self.ten}-{self.act_calls}", selected_action=outputs, context_audit={})


class MutatingManagePolicy(MutatingPolicy):
    def act(self, observation, goal=None):
        self.act_calls += 1
        outputs = np.zeros(20, dtype=float)
        outputs[11] = 1.0
        outputs[12] = 999.0
        return GiaTrace(trace_id=f"{self.ten}-{self.act_calls}", selected_action=outputs, context_audit={})


class MutatingHoldPolicy(MutatingPolicy):
    def act(self, observation, goal=None):
        self.act_calls += 1
        outputs = np.zeros(20, dtype=float)
        outputs[0] = 1.0
        outputs[12] = 999.0
        return GiaTrace(trace_id=f"{self.ten}-{self.act_calls}", selected_action=outputs, context_audit={})


def _system():
    GiaPolicy.dem = 0
    return HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=GiaPolicy,
    )


def test_facade_consumes_raw_regime_sidecar_without_leaking_it_to_policy_input() -> None:
    system = _system()
    raw = np.zeros(137, dtype=float)
    # M15 market indexes 36..42 map to EA wire indexes 84..90.
    raw[84], raw[85], raw[86], raw[87], raw[90] = 2.50, 1.00, 3.90, 0.0, 7.80
    normalized = normalize_input_vector(raw, mode="structure_aware_v4", size=137)

    system.set_regime_raw_inputs(raw)
    trace = system.act(normalized)

    assert trace.context_audit["regime_input_source"] == "bridge_raw"
    audit = trace.context_audit["regime_input_audit"]
    assert audit["raw_bridge"]["m15_over_h1_atr"] == pytest.approx(0.5)
    assert audit["policy_input"]["m15_over_h1_atr"] != pytest.approx(0.5)


def test_regime_context_switches_immediately_without_confirmation_count():
    detector = BoNhanDienNguCanh(CauHinhHeChuyenGia(bat=True))

    def candidate(observation, execution_context=None):
        return (
            "CTX_VOLATILE",
            "CTX_VOLATILE",
            NhiemVuNguCanh.VOLATILE,
            (0.0, 1.0, 0.0),
            "regime_timeframe=M15;regime=VOLATILE;confidence=0.90",
        )

    detector._ung_vien = candidate
    result = detector.nhan_dien(np.zeros(121))

    assert result.ma_context == "CTX_VOLATILE"
    assert result.on_dinh is True
    assert result.transition_active is False
    assert result.candidate_context == "CTX_VOLATILE"
    assert "direct_switch=true" in result.ly_do
    assert "confirm=" not in result.ly_do


def test_low_confidence_regime_is_still_applied_without_confirmation_count():
    detector = BoNhanDienNguCanh(CauHinhHeChuyenGia(bat=True))

    def candidate(observation, execution_context=None):
        return (
            "CTX_GIAM_MANH",
            "CTX_GIAM_MANH",
            NhiemVuNguCanh.GIAM_MANH,
            (0.0, 0.0, 0.0),
            "regime_timeframe=M15;regime=GIAM_MANH;confidence=0.10",
        )

    detector._ung_vien = candidate
    result = detector.nhan_dien(np.zeros(121))

    assert result.ma_context == "CTX_GIAM_MANH"
    assert result.on_dinh is True
    assert result.transition_active is False
    assert "confirm=" not in result.ly_do


def test_detector_switch_does_not_create_candidate_transition_microstate() -> None:
    detector = BoNhanDienNguCanh(CauHinhHeChuyenGia(bat=True))
    candidates = iter([
        ("CTX_VOLATILE_EXPANSION", "CTX_VOLATILE", NhiemVuNguCanh.VOLATILE_EXPANSION, (0.0,), "regime_timeframe=M15;regime=VOLATILE_EXPANSION;confidence=0.90"),
        ("CTX_TANG_MANH", "CTX_TANG_MANH", NhiemVuNguCanh.TANG_MANH, (1.0,), "regime_timeframe=M15;regime=TANG_MANH;confidence=0.90"),
    ])
    detector._ung_vien = lambda observation, execution_context=None: next(candidates)

    first = detector.nhan_dien(np.zeros(119))
    second = detector.nhan_dien(np.zeros(119))

    assert first.ma_context == "CTX_VOLATILE_EXPANSION"
    assert second.ma_context == "CTX_TANG_MANH"
    assert second.candidate_context == "CTX_TANG_MANH"
    assert second.transition_active is False
    assert second.transition_progress == 1.0
    assert HeChuyenGiaMotContextMotPolicy._selector_microstate_key(second) == "STABLE|TIMING_0_Q0"


def test_root_experts_are_independent_policies():
    system = _system()
    policies = [expert.policy for expert in system.experts.values()]
    assert len(policies) == 9
    assert len({id(policy) for policy in policies}) == 9
    assert not {
        "ROLE_ENTRY_MARKET",
        "ROLE_ENTRY_LIMIT",
        "ROLE_ENTRY_STOP",
        "ROLE_POSITION_MANAGEMENT",
    }.intersection(system.experts)

    system.act(np.zeros(119))
    assert {
        "ROLE_ENTRY_MARKET__CTX_RANGING",
        "ROLE_ENTRY_LIMIT__CTX_RANGING",
        "ROLE_ENTRY_STOP__CTX_RANGING",
    }.issubset(system.experts)
    assert len({id(expert.policy) for expert in system.experts.values()}) == 12


def test_disabling_role_specialists_prunes_persisted_role_owners() -> None:
    system = _system()
    system.act(np.zeros(119))
    assert any(str(context).startswith("ROLE_ENTRY_") for context in system.experts)
    system.cau_hinh = replace(
        system.cau_hinh,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
    )
    system.ensure_compatible()
    assert not any(str(context).startswith("ROLE_ENTRY_") for context in system.experts)
    assert not any(str(context).startswith("ROLE_POSITION_MANAGEMENT") for context in system.experts)


def test_ensure_compatible_rebinds_canonical_config_before_volatile_migration(monkeypatch):
    config = CauHinhHeChuyenGia(
        bat=True,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=config,
        tao_policy=GiaPolicy,
    )
    stale_config = replace(config, volatile_subexperts_bat=True)
    system.nhan_dien.cau_hinh = stale_config
    system.registry.cau_hinh = stale_config

    system.ensure_compatible()

    assert system.nhan_dien.cau_hinh is system.cau_hinh
    assert system.registry.cau_hinh is system.cau_hinh
    assert not any(context.startswith("CTX_VOLATILE_") for context in system.experts)
    monkeypatch.setattr(
        system.nhan_dien._classifier,
        "deterministic_regime",
        lambda *args, **kwargs: (5, np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
    )
    monkeypatch.setattr(
        system.nhan_dien._classifier,
        "volatile_subregime",
        lambda *args, **kwargs: pytest.fail("disabled volatile split must not be evaluated"),
    )
    candidate = system.nhan_dien._ung_vien(np.zeros(119))
    assert candidate[0] == "CTX_VOLATILE"


def test_volatile_is_split_into_three_live_independent_experts():
    system = _system()
    volatile = {
        context: system.experts[context]
        for context in (
            "CTX_VOLATILE_SPIKE",
            "CTX_VOLATILE_EXPANSION",
            "CTX_VOLATILE_CHOP",
        )
    }

    assert set(volatile) <= set(system.experts)
    assert all(item.ma_context_cha == "CTX_VOLATILE" for item in volatile.values())
    assert all(item.co_quyen_ra_lenh for item in volatile.values())
    assert len({id(item.policy) for item in volatile.values()}) == 3


def test_volatile_children_start_fresh_instead_of_copying_broad_action_bias():
    system = _system()
    volatile = [
        system.experts[context]
        for context in (
            "CTX_VOLATILE_SPIKE",
            "CTX_VOLATILE_EXPANSION",
            "CTX_VOLATILE_CHOP",
        )
    ]

    for expert in volatile:
        bootstrap = getattr(expert.policy, "_volatile_subexpert_bootstrap", {})
        assert bootstrap["mode"] == "fresh_independent_policy"
        assert bootstrap["source_context"] == "CTX_VOLATILE"


def test_old_warm_started_volatile_child_is_migrated_once():
    system = _system()
    child = system.experts["CTX_VOLATILE_EXPANSION"]
    old_policy = child.policy
    child.policy._volatile_subexpert_bootstrap = {
        "mode": "warm_start_fork_independent_memory",
        "source_context": "CTX_VOLATILE",
    }

    system.ensure_compatible()

    assert child.policy is not old_policy
    assert child.policy._volatile_subexpert_bootstrap["mode"] == "fresh_independent_policy"


def test_volatile_owner_guard_keeps_entries_but_blocks_non_emergency_close():
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=CloseWorstPolicy,
    )

    def candidate(observation, execution_context=None):
        return (
            "CTX_VOLATILE_EXPANSION",
            "CTX_VOLATILE",
            NhiemVuNguCanh.VOLATILE_EXPANSION,
            (0.0, 0.9, 1.0),
            "regime=VOLATILE_EXPANSION_conf=0.90_timing_q=0.15_dir=1",
        )

    system.nhan_dien._ung_vien = candidate
    opened = system.act(np.zeros(119))
    owner = system.experts["CTX_VOLATILE_EXPANSION"].ma_chuyen_gia
    positions = (
        "701:1:0.01:100:99:103:-0.2:12:0.0:-0.2:2.0:0.2:2.0:0.0:8:0:0.0:"
        f"CTX_VOLATILE_EXPANSION:{owner}:{owner}:"
        f"{opened.trace_id}"
    )
    system.set_account_context({}, positions)

    managed = system.act(np.zeros(119))
    audit = dict(managed.context_audit or {})

    assert 1 in set(audit["context_allowed_actions"])
    assert 8 not in set(audit["context_allowed_actions"])
    assert audit["owner_management_guard_active"] is True
    assert audit["owner_management_worst_unrealized_r"] == pytest.approx(-0.2)
    assert audit["owner_management_guard_reason"] in {
        "NO_PROFIT_FOR_CLOSE_BEST",
        "MIXED_PROFIT_FOR_CLOSE_ALL",
        "NO_EMERGENCY_FOR_CLOSE_WORST",
    }


def test_entry_committee_evaluates_four_policies_but_selects_one_trace():
    system = _system()
    trace = system.act(np.zeros(119))
    owners = [expert for expert in system.experts.values() if expert.policy.act_calls]
    assert len(owners) == 4
    assert {item.ma_context for item in owners} == {
        "CTX_RANGING",
        "ROLE_ENTRY_MARKET__CTX_RANGING",
        "ROLE_ENTRY_LIMIT__CTX_RANGING",
        "ROLE_ENTRY_STOP__CTX_RANGING",
    }
    audit = dict(trace.context_audit or {})
    assert trace.ma_chuyen_gia == "CG_GOC_CTX_RANGING"
    assert trace.ma_context == "CTX_RANGING"
    assert sum(bool(item["selected"]) for item in audit["selector_proposals"]) == 1
    assert audit["entry_proposal_not_blocked"] is True


def test_entry_committee_tracks_only_unsettled_entry_exposure() -> None:
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )

    first = system.act(np.zeros(119))
    first_audit = dict(first.context_audit or {})
    assert all(item["pending_outcomes"] == 0 for item in first_audit["selector_proposals"])
    assert system._selector_pending_entry_traces == {
        first.trace_id: ("CTX_RANGING", first_audit["selector_selected_policy"], 1)
    }

    decision_expert_id = system._trace_to_expert[first.trace_id]
    decision_expert = next(
        item for item in system.experts.values() if item.ma_chuyen_gia == decision_expert_id
    )
    assert system.apply_execution_lifecycle_credit(first.trace_id, 0.5) == 1
    assert system._selector_pending_entry_traces == {}
    evidence_group = system._trace_to_selector_evidence_group[first.trace_id]
    assert evidence_group.endswith("|SIDE_1")
    assert system.policy_selector.group_evidence(evidence_group)[0] == 1
    assert decision_expert.so_lifecycle_da_dong == 1
    assert decision_expert.so_feedback_transition == 0
    assert decision_expert.fitness_cuc_bo == pytest.approx(0.025)


def test_frozen_facade_keeps_owner_and_selector_evidence_immutable() -> None:
    """Frozen MT5 may log settlement but must not train any facade writer."""

    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )
    system._learning_enabled = False
    trace = system.act(np.zeros(119))
    owner = next(
        item for item in system.experts.values()
        if item.ma_chuyen_gia == system._trace_to_expert[trace.trace_id]
    )
    selector_before = (
        copy.deepcopy(system.policy_selector.learned_utility),
        copy.deepcopy(system.policy_selector.outcome_count),
        copy.deepcopy(system.policy_selector.group_outcome_count),
    )

    credited = system.apply_execution_lifecycle_credit(trace.trace_id, -0.7)

    assert credited == 0
    assert owner.policy.credit == 0
    assert owner.policy.bias == pytest.approx(0.0)
    assert owner.so_lifecycle_da_dong == 0
    assert owner.so_feedback_transition == 0
    assert owner.lich_su == []
    assert system._selector_pending_entry_traces == {}
    assert (
        system.policy_selector.learned_utility,
        system.policy_selector.outcome_count,
        system.policy_selector.group_outcome_count,
    ) == selector_before
def test_online_full_rl_facade_keeps_legacy_owner_and_selector_state_immutable() -> None:
    """Online Full-RL may update its policy batch, never facade evidence."""

    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=CanonicalOnlyFullRlPolicy,
    )
    trace = system.act(np.zeros(119))
    owner = next(
        item for item in system.experts.values()
        if item.ma_chuyen_gia == system._trace_to_expert[trace.trace_id]
    )
    selector_before = (
        copy.deepcopy(system.policy_selector.learned_utility),
        copy.deepcopy(system.policy_selector.outcome_count),
        copy.deepcopy(system.policy_selector.group_outcome_count),
    )

    credited = system.apply_execution_lifecycle_credit(trace.trace_id, 0.6)

    assert credited == 1
    # The dummy Full-RL policy is permitted to receive its own canonical
    # lifecycle; facade history/fitness and selector ranking are not.
    assert owner.policy.credit == 1
    assert owner.so_lifecycle_da_dong == 0
    assert owner.so_feedback_transition == 0
    assert owner.lich_su == []
    assert system._selector_pending_entry_traces == {}
    assert trace.trace_id not in system._trace_to_selector_policy
    assert trace.trace_id not in system._trace_to_selector_context
    assert trace.trace_id not in system._trace_to_selector_evidence_group
    assert (
        system.policy_selector.learned_utility,
        system.policy_selector.outcome_count,
        system.policy_selector.group_outcome_count,
    ) == selector_before
    extrema = system.record_position_context_extrema({
        "position_owner_expert": owner.ma_chuyen_gia,
        "position_owner_context": owner.ma_context,
        "position_ticket": "facade-full-rl-extrema",
    })
    assert extrema["full_rl_position_extrema_memory_skipped"] is True
    assert not hasattr(owner.policy, "position_context_extrema_memory")
    exit_credited = system.apply_execution_exit_credit(
        trace.trace_id,
        final_realized_r=0.4,
        hold_to_horizon_r=-0.2,
        position_ticket="full-rl-exit",
        exit_kind="POLICY_CLOSE_BEST",
        horizon_bars=8,
    )
    assert exit_credited == 1
    assert owner.policy.exit_credit == 1
    assert owner.so_lifecycle_da_dong == 0
    assert owner.lich_su == []
    assert (
        system.policy_selector.learned_utility,
        system.policy_selector.outcome_count,
        system.policy_selector.group_outcome_count,
    ) == selector_before


def test_reset_runtime_state_clears_selector_session_trace_not_learned_evidence() -> None:
    """A resumed tester session cannot inherit its predecessor's fallback."""

    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=GiaPolicy,
    )
    selector = system.policy_selector
    selector.learned_utility["kept"] = 0.42
    selector.outcome_count["kept"] = 3
    system.act(np.zeros(119))
    assert selector.last_trace is not None
    assert selector.fallback_policy_id
    assert selector._selection_has_non_cold_candidate is True

    system.reset_runtime_state()

    assert selector.last_trace is None
    assert selector.fallback_policy_id == ""
    assert selector._selection_has_non_cold_candidate is False
    assert selector.learned_utility == {"kept": 0.42}
    assert selector.outcome_count == {"kept": 3}


def test_execution_receipt_replaces_owner_cached_entry_trace() -> None:
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )
    trace = system.act(np.zeros(119))
    decision_expert = next(
        item for item in system.experts.values()
        if item.ma_chuyen_gia == system._trace_to_expert[trace.trace_id]
    )

    registered = system.register_execution_intent_trace(trace)
    bound = system.bind_execution_receipt(trace.trace_id, {
        "receipt_stage": "bridge_ack",
        "bridge_observed": True,
        "placed": True,
        "filled": True,
        "position_ticket": "T-17",
    })

    cached = decision_expert.policy._execution_trace_cache[trace.trace_id]
    assert registered["registered"] is True
    assert bound["bound"] is True
    assert cached.execution_details["filled"] is True
    assert cached.execution_details["position_ticket"] == "T-17"
    assert cached.action_details["execution_intent_registered"] is True
    assert cached.action_details["execution_receipt_bound"] is True


def test_verified_close_returns_routed_lifecycle_credit_for_observability() -> None:
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )
    trace = system.act(np.zeros(119))
    system.register_execution_intent_trace(trace)
    system.bind_execution_receipt(trace.trace_id, {
        "bridge_observed": True,
        "filled": True,
        "closed": True,
        "action_feedback_eligible": True,
        "memory_eligible": True,
        "outcome_eligible": True,
        "close_reason": "SL",
        "duration_bars": 4,
    })
    expert = next(
        item for item in system.experts.values()
        if item.ma_chuyen_gia == system._trace_to_expert[trace.trace_id]
    )
    cached = expert.policy._execution_trace_cache[trace.trace_id]

    signal = system.observe_transition(cached, np.zeros(119), -0.4, 0.0)

    assert signal["lifecycle_credit_routed"] is True
    assert signal["lifecycle_credit"] == 1
    assert expert.policy.credit == 1


def test_inflight_penalty_is_proposal_local_not_broadcast_to_correlated_committee() -> None:
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )

    first = system.act(np.zeros(119))
    first_policy = dict(first.context_audit or {})["selector_selected_policy"]
    second = system.act(np.zeros(119))
    proposals = list(dict(second.context_audit or {})["selector_proposals"])
    pending = {item["proposal_id"]: item["pending_outcomes"] for item in proposals}
    exposure = {item["proposal_id"]: item["correlated_exposure_count"] for item in proposals}

    assert pending[first_policy] == 1
    assert all(count == 0 for policy_id, count in pending.items() if policy_id != first_policy)
    assert all(count == 1 for count in exposure.values())


def test_correlated_addon_keeps_policy_risk_and_entry_action_open() -> None:
    system = _system()
    system.configure_risk_caps(type("Risk", (), {"min_risk_pct": 0.001, "max_risk_pct": 0.02})())
    first = system.act(np.zeros(119))
    first_action = int(np.argmax(first.selected_action[:12]))
    owner = system.experts["CTX_RANGING"].ma_chuyen_gia
    positions = (
        "701:1:0.01:100:99:103:-0.2:12:0.0:-0.2:2.0:0.2:2.0:0.0:8:0:0.0:"
        f"CTX_RANGING:{owner}:{owner}:{first.trace_id}"
    )
    system.set_account_context({}, positions)

    addon = system.act(np.zeros(119))
    audit = dict(addon.context_audit or {})

    assert first_action == 1
    assert int(np.argmax(addon.selected_action[:12])) == 1
    assert addon.selected_action[12] == pytest.approx(first.selected_action[12])
    assert audit["correlated_exposure_stage"] == "ADDON_DRAWDOWN"
    assert audit["correlated_exposure_count"] == 1
    assert audit["correlated_exposure_risk_scaled"] is False
    assert audit["correlated_exposure_entry_blocked"] is False
    assert audit["entry_proposal_not_blocked"] is True
    assert "ADDON_DRAWDOWN" in system._trace_to_selector_context[addon.trace_id]
    assert "STABLE|TIMING_0_Q0|ACTION_SIDE_1" in system._trace_to_selector_context[addon.trace_id]


def test_role_lattice_warm_starts_from_each_context_without_sharing_policy_identity() -> None:
    system = _system()
    ranging_parent = system.experts["CTX_RANGING"]
    market_ranging = system._role_specialist(
        "MARKET", "CTX_RANGING", parent=ranging_parent
    )
    tang_parent = system.experts["CTX_TANG_MANH"]
    market_tang = system._role_specialist(
        "MARKET", "CTX_TANG_MANH", parent=tang_parent
    )

    assert market_ranging is not None
    assert market_tang is not None
    assert market_ranging.policy is not ranging_parent.policy
    assert market_tang.policy is not tang_parent.policy
    assert market_ranging.policy is not market_tang.policy
    assert market_ranging.policy._role_specialist_bootstrap == {
        "role": "MARKET",
        "source_expert": ranging_parent.ma_chuyen_gia,
        "source_context": "CTX_RANGING",
        "mode": "context_owner_weight_warm_start_fresh_memory",
        "requested_warm_start": True,
    }
    identities = {
        _policy_identity(ranging_parent.policy)["policy_instance_uuid"],
        _policy_identity(market_ranging.policy)["policy_instance_uuid"],
        _policy_identity(market_tang.policy)["policy_instance_uuid"],
    }
    assert len(identities) == 3


def test_volatile_subtype_uses_its_own_policy_without_broad_parent_challenger() -> None:
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True, entry_style_specialists_bat=False),
        tao_policy=GiaPolicy,
    )

    def candidate(observation, execution_context=None):
        return (
            "CTX_VOLATILE_EXPANSION",
            "CTX_VOLATILE",
            NhiemVuNguCanh.VOLATILE,
            (0.0, 1.0, 0.0),
            "regime=VOLATILE_EXPANSION_conf=0.95_timing_q=0.10_dir=0",
        )

    system.nhan_dien._ung_vien = candidate
    trace = system.act(np.zeros(119))
    proposals = list(dict(trace.context_audit or {})["selector_proposals"])
    assert all(item["role"] != "VOLATILE_PARENT_ENTRY_GENERALIST" for item in proposals)
    assert all(item["policy_id"] == "CG_GOC_CTX_VOLATILE_EXPANSION" for item in proposals)
    assert len(proposals) == 1
    assert dict(trace.context_audit or {})["context_action_contract"] == "FIXED_CONTEXT_OWNER_V32"
    assert dict(trace.context_audit or {})["entry_proposal_not_blocked"] is True


def test_ensure_compatible_discards_removed_volatile_parent_selector_evidence() -> None:
    system = _system()
    stale_key = "CG_GOC_CTX_VOLATILE::VOLATILE_PARENT_ENTRY_GENERALIST"
    system.policy_selector.learned_utility[stale_key] = 0.75
    system.policy_selector.outcome_count[stale_key] = 9
    system.policy_selector.policy_group_loss_streak[stale_key] = 3
    system._trace_to_selector_policy["legacy-parent"] = stale_key
    system._trace_to_selector_context["legacy-parent"] = "legacy"
    system._trace_to_selector_evidence_group["legacy-parent"] = stale_key

    system.ensure_compatible()

    assert all("VOLATILE_PARENT_ENTRY_GENERALIST" not in str(key)
               for key in system.policy_selector.learned_utility)
    assert all("VOLATILE_PARENT_ENTRY_GENERALIST" not in str(key)
               for key in system.policy_selector.outcome_count)
    assert "legacy-parent" not in system._trace_to_selector_policy
    assert "legacy-parent" not in system._trace_to_selector_evidence_group


def test_cold_role_specialist_keeps_entry_and_policy_volume() -> None:
    system = _system()
    system.configure_risk_caps(type("Risk", (), {"min_risk_pct": 0.001, "max_risk_pct": 0.02})())
    style = system._role_specialist(
        "MARKET", "CTX_RANGING", parent=system.experts["CTX_RANGING"]
    )
    assert style is not None
    style.fitness_cuc_bo = 10.0

    trace = system.act(np.zeros(119))
    audit = dict(trace.context_audit or {})

    assert audit["decision_role"] == "ENTRY_STYLE_MARKET"
    assert int(np.argmax(trace.selected_action[:12])) == 1
    assert audit["role_readiness_supervisor_active"] is True
    assert audit["role_readiness_support"] == 0
    assert audit["role_readiness_risk_multiplier"] == pytest.approx(0.25)
    assert audit["role_readiness_target_risk_pct"] == pytest.approx(
        audit["role_readiness_original_risk_pct"]
    )
    assert audit["role_readiness_risk_scaled"] is False
    assert audit["role_readiness_entry_blocked"] is False


def test_frozen_inference_does_not_lazy_create_role_specialist() -> None:
    system = _system()
    system._learning_enabled = False
    before = set(system.experts)
    parent = system.experts["CTX_RANGING"]

    role = system._role_specialist("MARKET", "CTX_RANGING", parent=parent)

    assert role is None
    assert set(system.experts) == before


def test_role_specialist_warm_starts_weights_but_not_parent_memory() -> None:
    counter = iter(range(100, 200))
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=lambda: ChuyenGiaPolicy.seed(
            input_size=119,
            output_size=20,
            hidden_size=8,
            seed=next(counter),
        ),
    )
    parent = system.experts["CTX_RANGING"]
    parent.policy.memory_supernet.neural_cells["parent-only-memory"] = object()

    role = system._role_specialist("MARKET", "CTX_RANGING", parent=parent)

    assert role is not None
    assert role.policy.memory_supernet is not parent.policy.memory_supernet
    assert "parent-only-memory" not in role.policy.memory_supernet.neural_cells
    assert role.policy.experience_buffer == []
    assert len(role.policy.modules) == len(parent.policy.modules)
    for source_module, role_module in zip(parent.policy.modules, role.policy.modules):
        assert role_module is not source_module
        assert np.allclose(role_module.w_in, source_module.w_in)
        assert np.allclose(role_module.w_out, source_module.w_out)


def test_role_trace_namespace_rekeys_frozen_execution_cache() -> None:
    cached = FrozenTrace("trace-1")
    policy = SimpleNamespace(
        _execution_trace_cache={"trace-1": cached},
        _causal_trace_archive={"trace-1": cached},
    )
    expert = SimpleNamespace(ma_chuyen_gia="CG_ROLE_ENTRY_STOP__CTX_RANGING", policy=policy)

    renamed = HeChuyenGiaMotContextMotPolicy._namespace_proposal_trace(expert, cached)

    expected = "CG_ROLE_ENTRY_STOP__CTX_RANGING~trace-1"
    assert renamed.trace_id == expected
    assert set(policy._execution_trace_cache) == {expected}
    assert policy._execution_trace_cache[expected].trace_id == expected
    assert set(policy._causal_trace_archive) == {expected}
    assert policy._causal_trace_archive[expected].trace_id == expected


def test_outcome_updates_only_owner_policy():
    system = _system()
    trace = system.act(np.zeros(119))
    owner = next(expert for expert in system.experts.values() if expert.ma_chuyen_gia == trace.ma_chuyen_gia)
    system.observe_transition(trace, np.zeros(119), 0.2, 0.01)
    assert owner.policy.transition_calls == 1
    assert owner.so_lifecycle_da_dong == 0
    assert owner.so_feedback_transition == 1
    assert owner.fitness_cuc_bo == pytest.approx(0.0)
    owner.ghi_ket_qua(
        reward=0.5,
        cost=0.0,
        trace_id=trace.trace_id,
        source="execution_lifecycle",
        lifecycle_confirmed=True,
    )
    assert owner.so_lifecycle_da_dong == 1
    assert owner.fitness_cuc_bo == pytest.approx(0.025)
    assert all(expert.policy.transition_calls == 0 for expert in system.experts.values() if expert is not owner)


def test_close_transition_settles_lifecycle_credit_through_facade():
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )
    trace = system.act(np.zeros(119))
    owner = next(expert for expert in system.experts.values() if expert.ma_chuyen_gia == trace.ma_chuyen_gia)
    trace.execution_details = {
        "closed": True,
        "action_feedback_eligible": True,
        "memory_eligible": True,
        "outcome_eligible": True,
        "close_reason": "SL",
        "duration_bars": 4,
    }

    system.observe_transition(trace, np.zeros(119), -0.8, 0.0)

    assert owner.policy.transition_calls == 1
    assert owner.policy.credit == 1
    assert owner.so_lifecycle_da_dong == 1
    assert owner.fitness_cuc_bo == pytest.approx(-0.04)


def test_full_rl_close_ack_defers_lifecycle_until_canonical_ledger() -> None:
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=CanonicalOnlyFullRlPolicy,
    )
    trace = system.act(np.zeros(119))
    owner = next(expert for expert in system.experts.values() if expert.ma_chuyen_gia == trace.ma_chuyen_gia)
    trace.execution_details = {
        "closed": True,
        "action_feedback_eligible": True,
        "memory_eligible": True,
        "outcome_eligible": True,
        "close_reason": "SL",
        "duration_bars": 4,
    }

    signal = system.observe_transition(trace, np.zeros(119), -0.8, 0.0)

    assert owner.policy.transition_calls == 1
    assert owner.policy.credit == 0
    assert signal["lifecycle_credit_routed"] is False
    assert signal["lifecycle_credit_deferred_to_canonical_ledger"] is True


def test_position_owner_identity_is_separate_from_selected_style_policy_credit():
    system = _system()
    style = system._role_specialist(
        "MARKET", "CTX_RANGING", parent=system.experts["CTX_RANGING"]
    )
    assert style is not None
    style.fitness_cuc_bo = 10.0

    trace = system.act(np.zeros(119))
    audit = dict(trace.context_audit or {})
    assert trace.ma_chuyen_gia == "CG_GOC_CTX_RANGING"
    assert audit["decision_policy_id"] == style.ma_chuyen_gia
    assert audit["decision_role"] == "ENTRY_STYLE_MARKET"

    system.observe_transition(trace, np.zeros(119), 0.3, 0.0)
    assert style.policy.transition_calls == 1
    assert system.experts["CTX_RANGING"].policy.transition_calls == 0


def test_management_trace_does_not_redirect_final_lifecycle_credit_from_entry_owner():
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )
    opened = system.act(np.zeros(119))
    owner = opened.ma_chuyen_gia
    system.experts["CTX_RANGING"].policy = MutatingHoldPolicy()
    system.experts["ROLE_ENTRY_MARKET__CTX_RANGING"].policy = MutatingHoldPolicy()
    system.experts["ROLE_ENTRY_LIMIT__CTX_RANGING"].policy = MutatingHoldPolicy()
    system.experts["ROLE_ENTRY_STOP__CTX_RANGING"].policy = MutatingHoldPolicy()
    manager_policy = MutatingManagePolicy()
    manager = system._role_specialist(
        "MANAGEMENT", "CTX_RANGING", parent=system.experts["CTX_RANGING"]
    )
    assert manager is not None
    manager.policy = manager_policy
    positions = (
        "701:1:0.01:100:99:103:-0.4:12:0.0:-0.4:2.0:0.2:2.0:0.0:8:0:0.0:"
        f"CTX_RANGING:{owner}:{owner}:{opened.trace_id}"
    )
    system.set_account_context({}, positions)

    managed = system.act(np.zeros(119))
    assert managed.context_audit["decision_role"] == "POSITION_MANAGEMENT_SPECIALIST"
    assert system.apply_execution_lifecycle_credit(opened.trace_id, 0.5) == 1
    # The immutable final R belongs to the entry trace.  A management decision
    # receives only its own delayed exit-counterfactual reward, routed by the
    # EA via ``managed.trace_id`` after a POLICY_CLOSE_* event.
    owner_policy = system.experts["CTX_RANGING"].policy
    assert owner_policy.credit == 1
    assert owner_policy.bias == pytest.approx(0.5)
    assert manager_policy.credit == 0
    assert manager_policy.bias == pytest.approx(0.0)


def test_intraday_reset_keeps_owner_routing_until_lifecycle_settlement():
    system = _system()
    trace = system.act(np.zeros(119))
    trace_id = str(trace.trace_id)
    owner_id = system._trace_to_expert[trace_id]

    system.reset_intraday_state("2025-04-11")

    assert system._trace_to_expert[trace_id] == owner_id
    assert not hasattr(system, "_trace_to_nhan_dien")


def test_no_cross_expert_mutation_after_one_thousand_owner_lifecycles():
    GiaPolicy.dem = 0
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=MutatingPolicy,
    )
    trace = system.act(np.zeros(119))
    owner = next(expert for expert in system.experts.values() if expert.ma_chuyen_gia == trace.ma_chuyen_gia)
    others = [expert for expert in system.experts.values() if expert is not owner]
    before = {expert.ma_chuyen_gia: (expert.policy.bias, expert.policy.credit, expert.policy.transition_calls) for expert in others}

    for index in range(1000):
        system.observe_transition(trace, np.zeros(119), 0.01, 0.001, done=True)
        system.apply_execution_lifecycle_credit(trace.trace_id, realized_r=0.002)

    assert owner.policy.transition_calls == 1000
    assert owner.policy.credit == 1000
    assert owner.so_feedback_transition == 1000
    assert owner.so_lifecycle_da_dong == 1000
    assert owner.policy.bias > 0.0
    after = {expert.ma_chuyen_gia: (expert.policy.bias, expert.policy.credit, expert.policy.transition_calls) for expert in others}
    assert after == before


def test_fixed_taxonomy_routes_every_sample_to_a_root_owner():
    system = _system()
    roots = set(system.registry.chuyen_gia_theo_context)
    assert not hasattr(system.registry, "tao_context_con")

    for _ in range(24):
        trace = system.act(np.zeros(119))
        assert trace.ma_context in roots
        assert "_NHANH_" not in trace.ma_context

    assert not any("_NHANH_" in context for context in system.registry.chuyen_gia_theo_context)


def test_ensure_compatible_prunes_persisted_dynamic_context_branches():
    system = _system()
    parent = next(iter(system.registry.chuyen_gia_theo_context.values()))
    branch = copy.deepcopy(parent)
    branch.ma_context = f"{parent.ma_context}_NHANH_TEST"
    branch.ma_chuyen_gia = "CG_DYNAMIC_TEST"
    system.registry.chuyen_gia_theo_context[branch.ma_context] = branch
    system.registry.__dict__["context_con_theo_cha"] = {
        parent.ma_context: [branch.ma_context],
    }
    system.registry.__dict__["mau_context"] = {branch.ma_context: (0.0,)}
    system.registry.__dict__["ban_kinh_context"] = {branch.ma_context: 0.1}
    system.__dict__["uom_context"] = SimpleNamespace()
    system.__dict__["_trace_to_nhan_dien"] = {"legacy": object()}

    system.ensure_compatible()

    assert branch.ma_context not in system.registry.chuyen_gia_theo_context
    assert not any("_NHANH_" in context for context in system.registry.chuyen_gia_theo_context)
    assert not hasattr(system.registry, "context_con_theo_cha")
    assert not hasattr(system, "uom_context")
    assert not hasattr(system, "_trace_to_nhan_dien")


def test_archive_and_merge_context_require_explicit_gates():
    system = _system()
    roots = list(system.experts.values())
    source = roots[0]
    target = roots[1]

    dormant = system.ngu_dong_chuyen_gia(source.ma_context, ghi_chu="pause")
    assert dormant["changed"] is True
    assert source.trang_thai == TrangThaiChuyenGia.NGU_DONG

    rejected = system.hop_nhat_context_chuyen_gia(
        source.ma_context,
        target.ma_context,
        oos_pass=True,
        non_interference_pass=False,
    )
    assert rejected["merged"] is False
    assert source.trang_thai == TrangThaiChuyenGia.NGU_DONG

    merged = system.hop_nhat_context_chuyen_gia(
        source.ma_context,
        target.ma_context,
        oos_pass=True,
        non_interference_pass=True,
        ghi_chu="oos-ok",
    )
    assert merged["merged"] is True
    assert source.trang_thai == TrangThaiChuyenGia.LUU_TRU
    assert source.lich_su[-1]["target_context"] == target.ma_context
    assert target.lich_su[-1]["source_context"] == source.ma_context


def test_probe_expert_risk_cap_clamps_risk_head_before_bridge_decode():
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
            max_risk_pct_tham_do=0.005,
            max_open_risk_pct_tham_do=0.015,
        ),
        tao_policy=GiaPolicy,
    )
    system.configure_risk_caps(type("Risk", (), {"min_risk_pct": 0.001, "max_risk_pct": 0.02})())
    expert = system.experts["CTX_RANGING"]
    expert.trang_thai = TrangThaiChuyenGia.THAM_DO_RUI_RO_THAP

    trace = system.act(np.zeros(119))

    assert trace.selected_action[12] < 999.0
    audit = dict(trace.context_audit or {})
    assert audit["per_expert_risk_cap_active"] is True
    assert audit["per_expert_risk_clamped"] is True
    assert audit["per_expert_max_risk_pct"] == pytest.approx(0.005)


def test_probe_expert_open_risk_cap_clamps_risk_without_blocking_entry():
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
            max_risk_pct_tham_do=0.005,
            max_open_risk_pct_tham_do=0.015,
        ),
        tao_policy=GiaPolicy,
    )
    expert = system.experts["CTX_RANGING"]
    expert.trang_thai = TrangThaiChuyenGia.THAM_DO_RUI_RO_THAP
    system.set_account_context({"open_risk_ratio": 0.02})

    trace = system.act(np.zeros(119))

    assert int(np.argmax(trace.selected_action[:12])) == 1
    assert expert.policy.act_calls == 1
    assert trace.context_audit["ma_chuyen_gia"] == expert.ma_chuyen_gia
    assert trace.context_audit["per_expert_risk_cap_active"] is True
    assert trace.context_audit["per_expert_risk_clamped"] is True
    assert trace.selected_action[12] < 999.0


def test_flat_state_runs_context_generalist_and_three_entry_styles():
    system = _system()
    observation = np.zeros(119)
    observation[34] = 0.8
    trace = system.act(observation)
    assert trace.ma_context == "CTX_RANGING"
    owners = [expert for expert in system.experts.values() if expert.policy.act_calls]
    assert len(owners) == 4


def test_regime_contexts_route_to_distinct_market_expert_owners():
    system = _system()
    forced = iter([
        ("CTX_TANG_MANH", NhiemVuNguCanh.TANG_MANH),
        ("CTX_GIAM_MANH", NhiemVuNguCanh.GIAM_MANH),
        ("CTX_VOLATILE", NhiemVuNguCanh.VOLATILE),
    ])

    def candidate(observation, execution_context=None):
        context_id, task = next(forced)
        return (
            context_id,
            context_id,
            task,
            (0.0, 0.0, 0.0),
            f"regime={context_id.removeprefix('CTX_')}_conf=0.95_timing_q=0.10_dir=0",
        )

    system.nhan_dien._ung_vien = candidate

    first = system.act(np.zeros(119))
    second = system.act(np.zeros(119))
    third = system.act(np.zeros(119))

    assert first.ma_context == "CTX_TANG_MANH"
    assert second.ma_context == "CTX_GIAM_MANH"
    assert third.ma_context == "CTX_VOLATILE"

    called_contexts = {
        expert.ma_context
        for expert in system.experts.values()
        if expert.policy.act_calls > 0
    }
    assert {
        "CTX_TANG_MANH",
        "CTX_GIAM_MANH",
        "CTX_VOLATILE",
    }.issubset(called_contexts)


def test_context_switch_evaluates_active_entry_and_does_not_mask_it_with_position_owner():
    system = _system()
    forced = iter([
        ("CTX_TANG_MANH", NhiemVuNguCanh.TANG_MANH),
        ("CTX_GIAM_MANH", NhiemVuNguCanh.GIAM_MANH),
    ])

    def candidate(observation, execution_context=None):
        context_id, task = next(forced)
        return (
            context_id,
            context_id,
            task,
            (0.0, 0.0, 0.0),
            f"regime={context_id.removeprefix('CTX_')}_conf=0.95_timing_q=0.10_dir=0",
        )

    system.nhan_dien._ung_vien = candidate
    opened = system.act(np.zeros(119))
    owner = opened.ma_chuyen_gia
    owner_expert = next(item for item in system.experts.values() if item.ma_chuyen_gia == owner)

    # Full V10 position layout: owner context/expert/policy/trace are fields
    # 17..20 after the lifecycle features.
    positions = (
        "701:1:0.01:100:99:103:-0.4:12:0.0:-0.4:2.0:0.2:2.0:0.0:8:0:0.0:"
        "CTX_TANG_MANH:CG_GOC_CTX_TANG_MANH:CG_GOC_CTX_TANG_MANH:"
        f"{opened.trace_id}"
    )
    system.set_account_context({}, positions)
    selected = system.act(np.zeros(119))
    audit = dict(selected.context_audit or {})

    assert selected.ma_chuyen_gia == "CG_GOC_CTX_GIAM_MANH"
    assert audit["context_nhan_dien"] == "CTX_GIAM_MANH"
    assert audit["management_only"] is False
    assert audit["context_action_contract"] == "FIXED_CONTEXT_OWNER_V32"
    assert audit["entry_proposal_evaluated"] is True
    assert audit["entry_actions_available_in_selector"] is True
    assert audit["entry_proposal_not_blocked"] is True
    assert set(range(1, 7)).issubset(set(audit["context_allowed_actions"]))
    assert len(audit["selector_proposals"]) == 5
    assert owner_expert.policy.act_calls == 1
    assert system.experts["CTX_GIAM_MANH"].policy.act_calls == 1
    assert system.experts[
        "ROLE_POSITION_MANAGEMENT__CTX_TANG_MANH"
    ].policy.act_calls == 1


def test_context_switch_uses_position_owner_management_when_active_proposes_hold():
    system = _system()
    forced = iter([
        ("CTX_TANG_MANH", NhiemVuNguCanh.TANG_MANH),
        ("CTX_GIAM_MANH", NhiemVuNguCanh.GIAM_MANH),
    ])

    def candidate(observation, execution_context=None):
        context_id, task = next(forced)
        return (
            context_id,
            context_id,
            task,
            (0.0, 0.0, 0.0),
            f"regime={context_id.removeprefix('CTX_')}_conf=0.95_timing_q=0.10_dir=0",
        )

    system.nhan_dien._ung_vien = candidate
    opened = system.act(np.zeros(119))
    owner = opened.ma_chuyen_gia
    system.experts["CTX_GIAM_MANH"].policy = HoldPolicy()
    for role in ("MARKET", "LIMIT", "STOP"):
        expert = system._role_specialist(
            role, "CTX_GIAM_MANH", parent=system.experts["CTX_GIAM_MANH"]
        )
        assert expert is not None
        expert.policy = HoldPolicy()
    manager = system._role_specialist(
        "MANAGEMENT", "CTX_TANG_MANH", parent=system.experts["CTX_TANG_MANH"]
    )
    assert manager is not None
    manager.policy = ManagePolicy()
    positions = (
        "701:1:0.01:100:99:103:-0.4:12:0.0:-0.4:2.0:0.2:2.0:0.0:8:0:0.0:"
        "CTX_TANG_MANH:CG_GOC_CTX_TANG_MANH:CG_GOC_CTX_TANG_MANH:"
        f"{opened.trace_id}"
    )
    system.set_account_context({}, positions)

    selected = system.act(np.zeros(119))
    audit = dict(selected.context_audit or {})

    assert selected.ma_chuyen_gia == owner
    assert audit["management_only"] is True
    assert audit["decision_role"] == "POSITION_MANAGEMENT_SPECIALIST"
    assert audit["management_owner_expert"] == owner
    assert audit["management_target_ticket"] == "701"
    # V32 opens the emergency CLOSE_WORST path at -0.35R during a
    # context switch; it must remain a management capability, not an entry
    # gate or a risk/volume rewrite.
    assert set(audit["context_allowed_actions"]) == {0, 8, 11}
    assert audit["entry_proposal_evaluated"] is True
    assert audit["entry_actions_available_in_selector"] is True


def test_low_timing_keeps_all_entry_actions_available_for_learning():
    system = _system()

    def candidate(observation, execution_context=None):
        return (
            "CTX_RANGING",
            "CTX_RANGING",
            NhiemVuNguCanh.RANGING,
            (0.0, 0.0, 0.0),
            "regime=RANGING_conf=0.95_timing_q=0.00_dir=0",
        )

    system.nhan_dien._ung_vien = candidate
    trace = system.act(np.zeros(119))

    assert int(np.argmax(trace.selected_action[:12])) == 1
    assert set(range(1, 7)).issubset(set(trace.context_audit["context_allowed_actions"]))
    assert trace.context_audit["timing_quality"] == 0.0


def test_invalid_configuration_rejects_global_policy_and_blending():
    with pytest.raises(ValueError):
        CauHinhHeChuyenGia(cho_phep_policy_chung_ra_lenh=True)
    with pytest.raises(ValueError):
        CauHinhHeChuyenGia(cho_phep_tron_policy=True)


def test_direct_m15_context_calls_owner_policy_without_confirmation_delay():
    system = _system()
    trace = system.act(np.zeros(119))
    assert getattr(trace, "ly_do_khong_giao_dich", "") == ""
    assert getattr(trace, "ly_do_khong_hoc", False) is False
    assert trace.context_audit["context_on_dinh"] is True
    assert set(range(1, 7)).issubset(set(trace.context_audit["context_allowed_actions"]))
    assert any(expert.policy.act_calls == 1 for expert in system.experts.values())


def test_two_prior_context_tokens_are_causal_and_resize_only_direct_rnn() -> None:
    """Policy sees t-1/t-2 while the MT5 facade remains a 137-input bridge."""

    config = CauHinhHeChuyenGia(
        bat=True,
        context_history_inputs=2,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=config,
        tao_policy=lambda: ChuyenGiaPolicy.seed(
            input_size=137,
            output_size=20,
            hidden_size=6,
            seed=921,
        ),
    )
    regimes = iter((
        ("CTX_TANG_MANH", NhiemVuNguCanh.TANG_MANH),
        ("CTX_GIAM_YEU", NhiemVuNguCanh.GIAM_YEU),
        ("CTX_RANGING", NhiemVuNguCanh.RANGING),
    ))

    def next_regime(_observation, execution_context=None):
        context, task = next(regimes)
        return context, context, task, (0.0, 0.0, 0.0), "test_context_sequence"

    system.nhan_dien._ung_vien = next_regime
    first = system.act(np.zeros(137, dtype=float))
    second = system.act(np.zeros(137, dtype=float))
    third = system.act(np.zeros(137, dtype=float))

    assert system.input_size == 137
    assert all(expert.policy.input_size == 139 for expert in system.experts.values())
    assert first.raw_obs.shape == (139,)
    assert np.all(first.raw_obs[-2:] == 0.0)
    assert second.context_audit["context_history_previous_contexts"] == ["CTX_TANG_MANH", ""]
    assert third.context_audit["context_history_previous_contexts"] == ["CTX_GIAM_YEU", "CTX_TANG_MANH"]
    assert not np.allclose(second.raw_obs[-2:], 0.0)
    assert not np.allclose(third.raw_obs[-2:], second.raw_obs[-2:])


def test_frozen_like_trace_uses_context_audit_to_keep_owner_provenance():
    system = _system()
    trace = system.act(np.zeros(119))
    audit = dict(getattr(trace, "context_audit", {}) or {})
    assert audit["ma_context"] == "CTX_RANGING"
    assert audit["ma_chuyen_gia"] == trace.ma_chuyen_gia


def test_policy_identity_counts_contextual_expectancy_as_memory():
    memory = type("Memory", (), {})()
    memory.neurons = {}
    memory.connections = []
    memory.clusters = {}
    memory.concepts = {"concept-a": object()}
    memory.strategies = {}
    memory.meta_strategies = {}
    memory.causal_nodes = {}
    memory.neural_cells = {"cell-a": object(), "cell-b": object()}
    memory.neural_synapses = [object()]
    memory.contextual_expectancy = type("Expectancy", (), {"records": {"r1": object(), "r2": object()}})()
    policy = type("Policy", (), {"modules": [], "revision": 0, "memory_supernet": memory})()

    identity = _policy_identity(policy)

    assert identity["memory_entry_count"] == 6
    assert identity["memory_entry_breakdown"]["contextual_expectancy_records"] == 2
    assert identity["memory_entry_breakdown"]["neural_cells"] == 2


def test_registry_is_lightweight_and_does_not_duplicate_expert_checkpoints(tmp_path):
    GiaPolicy.dem = 0
    checkpoint_root = tmp_path / "kho_chuyen_gia"
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
        ),
        tao_policy=GiaPolicy,
    )
    registry_path = system.luu_so_dang_ky(tmp_path / "he_chuyen_gia.registry.json")

    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    assert len(payload["chuyen_gia"]) == 9
    assert all("checkpoint_policy_cuc_bo" not in row for row in payload["chuyen_gia"])
    assert all("so_feedback_transition" in row for row in payload["chuyen_gia"])
    assert checkpoint_root.exists() is False


def test_artifact_restart_preserves_fixed_context_owner_mapping(tmp_path):
    system = _system()
    before_trace = system.act(np.zeros(119))
    before_mapping = {
        context_id: expert.ma_chuyen_gia
        for context_id, expert in system.registry.chuyen_gia_theo_context.items()
    }

    artifact = tmp_path / "he_chuyen_gia.pkl"
    with artifact.open("wb") as handle:
        pickle.dump(system, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with artifact.open("rb") as handle:
        loaded = pickle.load(handle)

    after_mapping = {
        context_id: expert.ma_chuyen_gia
        for context_id, expert in loaded.registry.chuyen_gia_theo_context.items()
    }
    assert after_mapping == before_mapping
    assert loaded.tao_policy is None
    after_trace = loaded.act(np.zeros(119))
    assert after_trace.ma_chuyen_gia == before_trace.ma_chuyen_gia
    assert set(loaded.registry.chuyen_gia_theo_context) == set(before_mapping)
