from __future__ import annotations

import pytest

from self_growing_trader.policy_ensemble import (
    CorrelatedExposureSupervisor,
    PolicyCandidateScore,
    PolicyEnsembleSelector,
    RoleReadinessSupervisor,
)


def test_correlated_exposure_supervisor_deduplicates_open_and_pending_trace() -> None:
    supervisor = CorrelatedExposureSupervisor(risk_decay=0.5, min_risk_pct=0.001)

    state = supervisor.assess(
        market_context="CTX_VOLATILE_EXPANSION",
        direction=1,
        positions=[{
            "ticket": "7", "side": 1, "owner_context": "CTX_VOLATILE_EXPANSION",
            "owner_trace": "trace-open", "unrealized_r": -0.4,
        }],
        pending_entries=[
            {"trace_id": "trace-open", "context": "CTX_VOLATILE_EXPANSION", "direction": 1},
            {"trace_id": "trace-pending", "context": "CTX_VOLATILE_EXPANSION", "direction": 1},
        ],
    )

    assert state.exposure_count == 2
    assert state.open_position_count == 1
    assert state.stage == "ADDON_DRAWDOWN"
    assert state.utility_context == "CTX_VOLATILE_EXPANSION|ADDON_DRAWDOWN|SIDE_1"


def test_policy_ensemble_selects_best_valid_candidate() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="m1")

    trace = selector.select(
        [
            PolicyCandidateScore("m1", confidence=0.4, recent_utility=0.2, memory_match=0.3, reason="baseline"),
            PolicyCandidateScore("child-risk", confidence=0.8, recent_utility=0.7, memory_match=0.6, reason="stress_specialist"),
        ]
    )

    assert trace.selected_policy_id == "child-risk"
    assert trace.used_fallback is False
    assert trace.reason == "stress_specialist"
    assert "m1" in trace.scores
    assert "child-risk" in trace.scores


def test_policy_ensemble_falls_back_when_all_candidates_invalid() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="m1")

    trace = selector.select([PolicyCandidateScore("child", confidence=1.0, recent_utility=1.0, valid=False)])

    assert trace.selected_policy_id == "m1"
    assert trace.used_fallback is True
    assert trace.reason == "no_valid_candidates"


def test_policy_ensemble_falls_back_when_scores_too_low() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="m1", min_score=0.2)

    trace = selector.select(
        [
            PolicyCandidateScore(
                "child",
                confidence=0.1,
                recent_utility=0.1,
                memory_match=0.0,
                account_stress=1.0,
                drawdown=1.0,
                open_positions=4,
            )
        ]
    )

    assert trace.selected_policy_id == "m1"
    assert trace.used_fallback is True
    assert trace.reason == "score_below_min"


def test_policy_ensemble_learns_only_executed_policy_utility() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="entry", utility_alpha=0.5)
    before = selector.select(
        [
            PolicyCandidateScore("entry", confidence=0.5, recent_utility=0.0),
            PolicyCandidateScore("manager", confidence=0.5, recent_utility=0.0),
        ]
    )
    assert before.selected_policy_id == "entry"

    selector.observe_outcome("manager", 1.0)
    after = selector.select(
        [
            PolicyCandidateScore("entry", confidence=0.5, recent_utility=0.0),
            PolicyCandidateScore("manager", confidence=0.5, recent_utility=0.0),
        ]
    )

    assert selector.outcome_count == {"manager": 1}
    assert after.selected_policy_id == "manager"


def test_policy_ensemble_keeps_utility_separate_per_market_context() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="market", utility_alpha=1.0)
    selector.observe_outcome("stop", -1.0, context_key="CTX_VOLATILE_CHOP")
    selector.observe_outcome("stop", 1.0, context_key="CTX_GIAM_MANH")

    volatile = selector.select([
        PolicyCandidateScore("market", confidence=0.5, recent_utility=0.0, context_key="CTX_VOLATILE_CHOP"),
        PolicyCandidateScore("stop", confidence=0.5, recent_utility=0.0, context_key="CTX_VOLATILE_CHOP"),
    ])
    downtrend = selector.select([
        PolicyCandidateScore("market", confidence=0.5, recent_utility=0.0, context_key="CTX_GIAM_MANH"),
        PolicyCandidateScore("stop", confidence=0.5, recent_utility=0.0, context_key="CTX_GIAM_MANH"),
    ])

    assert volatile.selected_policy_id == "market"
    assert downtrend.selected_policy_id == "stop"
    assert selector.outcome_count == {
        "CTX_VOLATILE_CHOP::stop": 1,
        "CTX_GIAM_MANH::stop": 1,
    }


def test_policy_ensemble_gives_untried_context_specialist_a_bounded_exploration_turn() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="generalist", exploration_weight=0.25)
    selector.outcome_count["CTX_RANGING::generalist"] = 9

    selected = selector.select([
        PolicyCandidateScore("generalist", confidence=0.5, recent_utility=0.0, context_key="CTX_RANGING"),
        PolicyCandidateScore("limit", confidence=0.5, recent_utility=0.0, context_key="CTX_RANGING"),
    ])

    assert selected.selected_policy_id == "limit"


def test_supported_route_beats_high_confidence_cold_role() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="parent",
        exploration_weight=0.0,
        cold_start_penalty=0.60,
        cold_start_support=8,
    )

    selected = selector.select(
        [
            PolicyCandidateScore("parent", confidence=0.50, recent_utility=0.10, role_fit=1.0),
            PolicyCandidateScore(
                "cold-style",
                confidence=1.0,
                recent_utility=0.90,
                role_fit=1.0,
                cold_start=True,
            ),
        ],
        require_valid_selection=True,
    )

    assert selected.selected_policy_id == "parent"


def test_cold_role_remains_probeable_when_it_is_the_only_route() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="parent",
        exploration_weight=0.0,
        cold_start_penalty=0.60,
        cold_start_support=8,
    )

    selected = selector.select(
        [PolicyCandidateScore("cold-style", confidence=0.20, recent_utility=0.0, cold_start=True)],
        require_valid_selection=True,
    )

    assert selected.selected_policy_id == "cold-style"
    assert selected.used_fallback is False


def test_policy_ensemble_does_not_stack_unresolved_specialist_trials() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="generalist",
        exploration_weight=0.25,
        inflight_penalty=0.20,
    )
    selector.outcome_count["CTX_RANGING::generalist"] = 9

    selected = selector.select([
        PolicyCandidateScore(
            "generalist",
            confidence=0.5,
            recent_utility=0.0,
            context_key="CTX_RANGING",
        ),
        PolicyCandidateScore(
            "limit",
            confidence=0.5,
            recent_utility=0.0,
            context_key="CTX_RANGING",
            pending_outcomes=1,
        ),
    ])

    assert selected.selected_policy_id == "generalist"


def test_pending_outcome_penalty_never_blocks_the_only_entry_candidate() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="generalist",
        exploration_weight=0.25,
        inflight_penalty=0.20,
    )

    selected = selector.select([
        PolicyCandidateScore(
            "limit",
            confidence=0.5,
            recent_utility=0.0,
            context_key="CTX_RANGING",
            pending_outcomes=3,
        ),
    ])

    assert selected.selected_policy_id == "limit"
    assert selected.used_fallback is False


def test_correlated_direction_exposure_prefers_a_less_crowded_entry_side() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="generalist",
        exploration_weight=0.0,
        inflight_penalty=0.0,
        exposure_penalty=0.20,
    )

    selected = selector.select([
        PolicyCandidateScore(
            "sell_stop",
            confidence=0.8,
            recent_utility=0.0,
            context_key="CTX_VOLATILE",
            correlated_exposure=3,
        ),
        PolicyCandidateScore(
            "buy_limit",
            confidence=0.5,
            recent_utility=0.0,
            context_key="CTX_VOLATILE",
            correlated_exposure=1,
        ),
    ], require_valid_selection=True)

    assert selected.selected_policy_id == "buy_limit"
    assert selected.used_fallback is False


def test_correlated_exposure_penalty_never_blocks_the_only_entry_candidate() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="generalist",
        min_score=0.9,
        exploration_weight=0.0,
        exposure_penalty=0.20,
    )

    selected = selector.select([
        PolicyCandidateScore(
            "sell_stop",
            confidence=0.1,
            recent_utility=-1.0,
            correlated_exposure=10,
        ),
    ], require_valid_selection=True)

    assert selected.selected_policy_id == "sell_stop"
    assert selected.used_fallback is False


def test_inflight_and_exposure_penalties_keep_growing_after_three_entries() -> None:
    selector = PolicyEnsembleSelector(
        exploration_weight=0.0,
        inflight_penalty=0.20,
        exposure_penalty=0.20,
    )
    at_three = PolicyCandidateScore(
        "clustered",
        confidence=0.5,
        recent_utility=0.0,
        pending_outcomes=3,
        correlated_exposure=3,
    )
    at_eight = PolicyCandidateScore(
        "clustered",
        confidence=0.5,
        recent_utility=0.0,
        pending_outcomes=8,
        correlated_exposure=8,
    )

    assert selector._score(at_eight) == pytest.approx(
        selector._score(at_three) - 2.0
    )


def test_required_entry_selection_ignores_min_score_without_masking_action() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="generalist",
        min_score=0.9,
        exploration_weight=0.0,
    )

    selected = selector.select(
        [PolicyCandidateScore("market", confidence=0.1, recent_utility=-1.0)],
        require_valid_selection=True,
    )

    assert selected.selected_policy_id == "market"
    assert selected.used_fallback is False
    assert selected.reason == "best_valid_required"


def test_confirmed_losses_outweigh_network_confidence_and_margin() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="high-confidence",
        utility_alpha=0.35,
        exploration_weight=0.0,
    )
    for _ in range(3):
        selector.observe_outcome("high-confidence", -1.0, context_key="CTX_VOLATILE")

    selected = selector.select([
        PolicyCandidateScore(
            "high-confidence",
            confidence=1.0,
            recent_utility=1.0,
            context_key="CTX_VOLATILE",
            memory_match=1.0,
            proposal_margin=1.0,
            role_fit=1.0,
        ),
        PolicyCandidateScore(
            "alternative",
            confidence=0.4,
            recent_utility=0.0,
            context_key="CTX_VOLATILE",
            role_fit=1.0,
        ),
    ])

    assert selected.selected_policy_id == "alternative"
    assert selector.loss_streak["CTX_VOLATILE::high-confidence"] == 3


def test_policy_role_loss_backoff_aggregates_across_microstates() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="generalist",
        exploration_weight=0.0,
        loss_streak_penalty=0.0,
        policy_group_loss_streak_penalty=0.60,
    )
    role_group = "POSITION_MANAGEMENT_SPECIALIST|SIDE_0"
    for index in range(3):
        selector.observe_outcome(
            "volatile-manager",
            -1.0,
            context_key=f"CTX_{index}|MICRO_{index}",
            evidence_group=role_group,
        )

    key = selector._policy_group_key("volatile-manager", role_group)
    assert selector.policy_group_loss_streak[key] == 3
    selected = selector.select([
        PolicyCandidateScore(
            "volatile-manager",
            confidence=1.0,
            recent_utility=1.0,
            context_key="CTX_NEW|MICRO_NEW",
            evidence_group=role_group,
            role_fit=1.0,
        ),
        PolicyCandidateScore(
            "generalist",
            confidence=0.45,
            recent_utility=0.0,
            context_key="CTX_NEW|MICRO_NEW",
            evidence_group="CONTEXT_ENTRY_GENERALIST|SIDE_0",
            role_fit=1.0,
        ),
    ], require_valid_selection=True)
    assert selected.selected_policy_id == "generalist"


def test_positive_outcome_resets_selector_loss_streak() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="market", utility_alpha=0.35)
    selector.observe_outcome("market", -1.0, context_key="CTX_RANGING")
    selector.observe_outcome("market", -1.0, context_key="CTX_RANGING")
    selector.observe_outcome("market", 1.2, context_key="CTX_RANGING")

    key = "CTX_RANGING::market"
    assert selector.loss_streak[key] == 0
    assert selector.outcome_count[key] == 3
    assert selector.win_signal[key] > -1.0


def test_role_direction_losses_back_off_into_unseen_context_microstate() -> None:
    selector = PolicyEnsembleSelector(
        fallback_policy_id="generalist-new-context",
        utility_alpha=0.35,
        exploration_weight=0.25,
    )
    role_group = "ENTRY_STYLE_STOP|SIDE_1"
    for index in range(4):
        selector.observe_outcome(
            "stop-old-context",
            -1.0,
            context_key=f"CTX_OLD|MICRO_{index}",
            evidence_group=role_group,
        )

    selected = selector.select([
        PolicyCandidateScore(
            "generalist-new-context",
            confidence=0.45,
            recent_utility=0.0,
            context_key="CTX_NEW|UNSEEN_MICROSTATE",
            evidence_group="CONTEXT_ENTRY_GENERALIST|SIDE_1",
            role_fit=1.0,
        ),
        PolicyCandidateScore(
            "stop-new-context",
            confidence=0.95,
            recent_utility=0.0,
            context_key="CTX_NEW|UNSEEN_MICROSTATE",
            evidence_group=role_group,
            role_fit=1.0,
        ),
    ], require_valid_selection=True)

    support, utility, _win_signal, loss_streak = selector.group_evidence(role_group)
    assert support == 4
    assert utility < 0.0
    assert loss_streak == 4
    assert selected.selected_policy_id == "generalist-new-context"


def test_hierarchical_role_evidence_never_blocks_only_entry_candidate() -> None:
    selector = PolicyEnsembleSelector(fallback_policy_id="generalist")
    role_group = "ENTRY_STYLE_MARKET|SIDE_1"
    for _ in range(6):
        selector.observe_outcome(
            "market",
            -1.0,
            context_key="CTX_OLD",
            evidence_group=role_group,
        )

    selected = selector.select([
        PolicyCandidateScore(
            "market-new-context",
            confidence=0.01,
            recent_utility=-1.0,
            context_key="CTX_NEW",
            evidence_group=role_group,
            role_fit=1.0,
        ),
    ], require_valid_selection=True)

    assert selected.selected_policy_id == "market-new-context"
    assert selected.used_fallback is False


def test_role_readiness_is_telemetry_without_volume_scaling() -> None:
    supervisor = RoleReadinessSupervisor(
        min_risk_multiplier=0.25,
        support_scale=4.0,
        negative_utility_weight=0.5,
    )
    cold = supervisor.assess(
        evidence_group="ENTRY_STYLE_STOP|SIDE_1",
        support=0,
        learned_utility=0.0,
    )
    mature = supervisor.assess(
        evidence_group="ENTRY_STYLE_STOP|SIDE_1",
        support=20,
        learned_utility=0.5,
    )

    assert cold.risk_multiplier == 0.25
    assert 0.95 < mature.risk_multiplier <= 1.0
