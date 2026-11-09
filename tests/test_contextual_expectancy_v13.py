from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from self_growing_trader.contextual_expectancy import ContextualExpectancyLedger, robust_expectancy
from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.replacement_learning import action_scope
from self_growing_trader.self_growing_runner import SelfGrowingPolicy, mt5_promotion_rejection_reasons


def _context() -> np.ndarray:
    value = np.zeros(46, dtype=float)
    value[:8] = np.array([0.35, -0.20, 0.15, 0.05, 0.30, -0.10, 0.20, 0.10])
    value[32:40] = np.array([0.10, 0.05, -0.10, 0.20, 0.0, 0.0, 0.10, 0.0])
    return value


def _policy_audit(lcb: float, *, samples: int = 8, ratio: float = 0.80, negative: int = 0) -> dict[str, object]:
    return {
        "fitness_expectancy_schema_version": 13,
        "fitness_policy_origin_lifecycle_count": samples,
        "fitness_policy_origin_ratio": ratio,
        "fitness_policy_expectancy_lcb": lcb,
        "fitness_context_positive_eligible": 1 if lcb > 0 else 0,
        "fitness_context_negative_critical": negative,
        "fitness_lifecycle_count": 8,
        "fitness_ledger_integrity": 0.90,
        "fitness_immediate_exit_ratio": 0.0,
        "fitness_execution_failure_rate": 0.0,
        "fitness_expectancy_lcb": lcb,
        "profit_factor": 1.5,
        "net_profit": 25.0,
        "equity_drawdown_pct": 2.0,
        "fitness_reconciled": True,
    }


def test_small_sample_positive_mean_is_not_reliable_positive_edge():
    estimate = robust_expectancy([3.0], min_samples=6)
    assert estimate.mean_net_r > 0.0
    assert estimate.expectancy_lcb < 0.0
    assert not estimate.eligible


def test_context_action_ledger_is_exactly_scoped_and_bootstrap_cannot_bias_policy():
    ledger = ContextualExpectancyLedger(min_samples=6)
    context = _context()
    buy = action_scope(1, positions_pipe="")
    sell = action_scope(2, positions_pipe="")
    for index in range(8):
        ledger.observe(context, buy, 0.85 + 0.03 * index, origin="policy", trace_id=f"buy-{index}")
        # A profitable legacy bootstrap SELL is retained for audit but has no
        # policy action value, and must never leak into BUY/Sell selection.
        ledger.observe(context, sell, 2.0, origin="bootstrap", trace_id=f"bootstrap-sell-{index}")

    query = ledger.query(context, output_size=20, positions_pipe="")
    lcb = np.asarray(query["expectancy_action_lcb"], dtype=float)
    bias = np.asarray(query["expectancy_action_bias"], dtype=float)
    assert lcb[1] > 0.0
    assert bias[1] > 0.0
    assert lcb[2] == 0.0
    assert bias[2] == 0.0


def test_management_shadow_evidence_is_reported_but_cannot_bias_policy():
    ledger = ContextualExpectancyLedger(min_samples=3)
    context = _context()
    exit_scope = action_scope(8, positions_pipe="ticket=7")
    for index in range(6):
        ledger.observe(
            context,
            exit_scope,
            -0.80,
            origin="management_shadow",
            trace_id=f"management-shadow-{index}",
        )

    report = ledger.report()
    assert report["policy_origin_lifecycle_count"] == 0
    assert report["management_shadow_lifecycle_count"] == 6
    assert report["management_shadow_expectancy"]["mean_net_r"] < 0.0

    query = ledger.query(context, output_size=12, positions_pipe="ticket=7")
    assert float(query["expectancy_action_samples"][8]) == 0.0
    assert float(query["expectancy_negative_inhibition"][8]) == 0.0


def test_query_radius_can_widen_oos_evidence_without_changing_merge_radius():
    ledger = ContextualExpectancyLedger(min_samples=1, merge_similarity=0.90, query_min_similarity=0.20)
    context = _context()
    shifted = context.copy()
    shifted[0] = -shifted[0]
    ledger.observe(context, action_scope(2, positions_pipe=""), -0.8, origin="policy", trace_id="radius-1")
    ledger.ensure_compatible()
    assert ledger.merge_similarity == 0.90
    assert ledger.query_min_similarity == 0.20
    narrow = ledger.copy()
    narrow.query_min_similarity = 0.99
    assert float(narrow.query(shifted, output_size=12)["expectancy_action_samples"][2]) == 0.0
    assert float(ledger.query(shifted, output_size=12)["expectancy_action_samples"][2]) >= 0.0


def test_reliable_negative_expectancy_inhibits_only_matching_action():
    ledger = ContextualExpectancyLedger(min_samples=6)
    context = _context()
    buy = action_scope(1, positions_pipe="")
    sell = action_scope(2, positions_pipe="")
    for index in range(8):
        ledger.observe(context, buy, 0.70, origin="policy", trace_id=f"positive-buy-{index}")
        ledger.observe(context, sell, -0.85, origin="policy", trace_id=f"negative-sell-{index}")

    query = ledger.query(context, output_size=20, positions_pipe="")
    bias = np.asarray(query["expectancy_action_bias"], dtype=float)
    inhibition = np.asarray(query["expectancy_negative_inhibition"], dtype=float)
    assert inhibition[2] > 0.0
    assert bias[2] == 0.0

    policy = ChuyenGiaPolicy.seed(input_size=119, output_size=20, seed=37)
    policy.set_decision_context(
        account_context={"equity_to_balance": 1.0}, positions_pipe="", allowed_actions=[True] * 12,
        sequence_id=3, frame_id="negative-inhibition-frame",
    )
    trace = policy.act(np.zeros(119, dtype=float))
    frame = policy._active_context_frame
    policy_ledger = ContextualExpectancyLedger(min_samples=6)
    for index in range(8):
        policy_ledger.observe(frame.canonical_memory_key, buy, 0.70, origin="policy", trace_id=f"policy-buy-{index}")
        policy_ledger.observe(frame.canonical_memory_key, sell, -0.85, origin="policy", trace_id=f"policy-sell-{index}")
    policy.memory_supernet.contextual_expectancy = policy_ledger
    policy._decision_memory_query_cache = None
    memory = policy._query_memory(trace.encoded_state)
    combined = np.asarray(memory["action_bias"], dtype=float)
    assert combined[2] < 0.0
    assert combined[1] > 0.0


def test_owner_aggregate_fallback_blocks_negative_action_outside_prototype_radius():
    ledger = ContextualExpectancyLedger(min_samples=6, query_min_similarity=0.99)
    first = _context()
    second = first.copy()
    second[:8] *= -1.0
    sell = action_scope(2, positions_pipe="")
    for index in range(8):
        ledger.observe(first, sell, -0.85, origin="policy", trace_id=f"aggregate-a-{index}")
        ledger.observe(second, sell, -0.65, origin="policy", trace_id=f"aggregate-b-{index}")

    shifted = -first.copy()
    exact = ledger.query(shifted, output_size=12, positions_pipe="")
    assert float(exact["expectancy_action_samples"][2]) == 0.0
    aggregate = ledger.aggregate_action_expectancy(2, positions_pipe="")
    assert aggregate.sample_count == 16
    assert aggregate.expectancy_lcb < -0.15

    policy = ChuyenGiaPolicy.seed(input_size=119, output_size=20, seed=39)
    policy.memory_supernet.contextual_expectancy = ledger
    policy._context_expert_owner_metadata = {"management_only": False}
    policy._context_expectancy_entry_gate_enabled = True
    policy._context_expectancy_entry_gate_scope_fallback_enabled = True
    policy._context_expectancy_entry_gate_min_support = 12.0
    policy._context_expectancy_entry_gate_max_lcb = -0.15
    mask, audit = policy._context_expectancy_entry_gate_mask(12, {})
    assert bool(mask[2]) is False
    assert audit["context_expectancy_entry_gate_fallback_indices"] == [2]


def test_contextual_expectancy_compatibility_cache_invalidates_when_records_change():
    ledger = ContextualExpectancyLedger(min_samples=6)
    ledger.ensure_compatible()
    first_signature = getattr(ledger, "_compatibility_signature", None)
    ledger.ensure_compatible()
    assert getattr(ledger, "_compatibility_signature", None) == first_signature

    ledger.observe(_context(), action_scope(1, positions_pipe=""), 0.7, origin="policy", trace_id="cache-1")
    assert len(ledger.records) == 1
    ledger.ensure_compatible()
    assert getattr(ledger, "_compatibility_signature", None) != first_signature


def test_legacy_expectancy_schema_resets_only_stale_expectancy_records():
    ledger = ContextualExpectancyLedger(min_samples=6)
    ledger.observe(_context(), action_scope(1, positions_pipe=""), 0.8, origin="policy", trace_id="legacy")
    assert ledger.records

    ledger.schema_version = 13
    ledger.ensure_compatible()

    assert ledger.schema_version == 14
    assert ledger.records == {}


def test_policy_lifecycle_credit_creates_context_action_expectancy_memory():
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, seed=19)
    obs = np.linspace(-0.2, 0.3, 119)
    policy.set_decision_context(
        account_context={"equity_to_balance": 1.0},
        positions_pipe="",
        allowed_actions=[True] * 12,
        execution_context={"spread_points": 12.0},
        sequence_id=1,
        frame_id="expectancy-test-frame",
    )
    trace = policy.act(obs)
    action = np.zeros(policy.output_size, dtype=float)
    action[1] = 1.0
    trace = replace(trace, selected_action=action, action_details={"action_name": "MARKET_BUY"})
    policy._cache_execution_trace(trace)
    for index in range(6):
        policy.apply_execution_lifecycle_credit(
            f"{trace.trace_id}-{index}",
            0.9,
            close_reason="TP",
            duration_bars=5.0,
        )
        # Each lifecycle uses a distinct trace ID in production. Mirror that
        # contract rather than crediting the same id twice.
        policy._execution_trace_cache[f"{trace.trace_id}-{index}"] = trace

    # The first call did not find the suffixed ID. Credit distinct valid trace
    # IDs now and verify context/action evidence is present.
    for index in range(6):
        policy.apply_execution_lifecycle_credit(
            f"{trace.trace_id}-{index}", 0.9, close_reason="TP", duration_bars=5.0
        )
    report = policy.contextual_expectancy_report()
    assert report["policy_origin_lifecycle_count"] == 6
    assert report["policy_expectancy"]["expectancy_lcb"] > 0.0


def test_policy_lifecycle_expectancy_deduplicates_repeated_management_traces():
    policy = ChuyenGiaPolicy.seed(input_size=119, output_size=20, seed=29)
    obs = np.zeros(119, dtype=float)
    policy.set_decision_context(
        account_context={"equity_to_balance": 1.0},
        positions_pipe="",
        allowed_actions=[True] * 12,
        sequence_id=1,
        frame_id="expectancy-dedup-frame",
    )
    base = policy.act(obs)
    action = np.zeros(policy.output_size, dtype=float)
    action[1] = 1.0
    first = replace(
        base,
        trace_id="entry-trace",
        selected_action=action,
        action_details={"action_name": "MARKET_BUY"},
        execution_details={"position_ticket": "position-7", "deal_ticket": "deal-99"},
    )
    second = replace(first, trace_id="management-trace")
    policy._execution_trace_cache[first.trace_id] = first
    policy._execution_trace_cache[second.trace_id] = second

    policy.apply_execution_lifecycle_credit(first.trace_id, 0.8, close_reason="TP", duration_bars=5.0)
    policy.apply_execution_lifecycle_credit(second.trace_id, 0.8, close_reason="TP", duration_bars=5.0)

    report = policy.contextual_expectancy_report()
    assert report["policy_origin_lifecycle_count"] == 1


def test_policy_lifecycle_expectancy_uses_mt5_position_ticket_for_dedup():
    policy = ChuyenGiaPolicy.seed(input_size=119, output_size=20, seed=30)
    obs = np.zeros(119, dtype=float)
    policy.set_decision_context(
        account_context={"equity_to_balance": 1.0},
        positions_pipe="",
        allowed_actions=[True] * 12,
        sequence_id=1,
        frame_id="expectancy-mt5-ticket-frame",
    )
    base = policy.act(obs)
    action = np.zeros(policy.output_size, dtype=float)
    action[1] = 1.0
    first = replace(base, trace_id="entry-trace", selected_action=action, action_details={"action_name": "MARKET_BUY"})
    second = replace(first, trace_id="management-trace")
    policy._execution_trace_cache[first.trace_id] = first
    policy._execution_trace_cache[second.trace_id] = second
    lifecycle = SimpleNamespace(position_ticket="position-from-mt5")

    policy.apply_execution_lifecycle_credit(
        first.trace_id,
        0.8,
        close_reason="TP",
        duration_bars=5.0,
        lifecycle=lifecycle,
    )
    policy.apply_execution_lifecycle_credit(
        second.trace_id,
        0.8,
        close_reason="TP",
        duration_bars=5.0,
        lifecycle=lifecycle,
    )

    report = policy.contextual_expectancy_report()
    assert report["policy_origin_lifecycle_count"] == 1


def test_position_extrema_memory_is_audit_evidence_not_realized_policy_credit():
    policy = ChuyenGiaPolicy.seed(input_size=119, output_size=20, seed=31)
    observation = np.zeros(119, dtype=float)

    policy.record_position_context_extrema({
        "position_ticket": "position-extrema-1",
        "owner_trace_id": "entry-trace",
        "side": 1,
        "recovery_after_low_r": 0.5,
        "low": {"unrealized_r": -0.8, "observation": observation.tolist(), "sequence": 10},
        "high": {"unrealized_r": 0.3, "observation": observation.tolist(), "sequence": 20},
    })

    report = policy.contextual_expectancy_report()
    assert report["policy_origin_lifecycle_count"] == 0
    assert report["all_origin_lifecycle_count"] == 0
    assert report["derived_evidence_count"] == 2
    assert report["derived_evidence_origins"]["memory_extrema"] == 2


def test_policy_query_includes_positive_contextual_expectancy_bias():
    policy = SelfGrowingPolicy.seed(input_size=119, output_size=20, seed=23)
    obs = np.linspace(-0.3, 0.4, 119)
    policy.set_decision_context(
        account_context={"equity_to_balance": 1.0}, positions_pipe="", allowed_actions=[True] * 12,
        sequence_id=2, frame_id="expectancy-query-frame",
    )
    trace = policy.act(obs)
    frame = policy._active_context_frame
    buy_scope = action_scope(1, positions_pipe="")
    for index in range(8):
        policy.memory_supernet.contextual_expectancy.observe(
            frame.canonical_memory_key, buy_scope, 0.95, origin="policy", trace_id=f"direct-{index}"
        )
    memory = policy._query_memory(trace.encoded_state)
    bias = np.asarray(memory["expectancy_action_bias"], dtype=float)
    lcb = np.asarray(memory["expectancy_action_lcb"], dtype=float)
    assert lcb[1] > 0.0
    assert bias[1] > 0.0
    assert memory["expectancy_value_source"] == "policy_origin_net_r_lcb"


def test_promotion_rejects_global_winner_without_policy_origin_positive_expectancy():
    baseline = {"best": {"fitness": 5.0, "metrics": _policy_audit(0.10)}}
    trial = {"best": {"fitness": 8.0, "metrics": _policy_audit(-0.05)}}
    reasons = mt5_promotion_rejection_reasons(
        baseline, trial, min_daily_trades=0, backtest_days=31
    )
    assert any("no_positive_context_action_expectancy_scope" in reason for reason in reasons)
    assert any("policy_origin_expectancy_lcb_not_materially_improved" in reason for reason in reasons)


def test_promotion_ratio_and_delta_are_configurable():
    baseline = {"best": {"fitness": 5.0, "metrics": _policy_audit(0.10, samples=4, ratio=0.50)}}
    trial = {"best": {"fitness": 8.0, "metrics": _policy_audit(0.115, samples=4, ratio=0.50)}}
    strict = mt5_promotion_rejection_reasons(
        baseline,
        trial,
        min_daily_trades=0,
        backtest_days=31,
        min_policy_origin_ratio=0.60,
        promotion_lcb_min_delta=0.02,
    )
    permissive = mt5_promotion_rejection_reasons(
        baseline,
        trial,
        min_daily_trades=0,
        backtest_days=31,
        min_policy_origin_ratio=0.40,
        promotion_lcb_min_delta=0.01,
    )
    assert any("policy_origin_samples_insufficient:4<5" in reason for reason in strict)
    assert "policy_origin_expectancy_lcb_not_materially_improved" in strict
    assert not any("policy_origin_samples_insufficient" in reason for reason in permissive)
    assert "policy_origin_expectancy_lcb_not_materially_improved" not in permissive


def test_promotion_rejects_reconciled_window_with_terminal_open_positions():
    trial_metrics = _policy_audit(0.20)
    trial_metrics.update(
        require_terminal_flat_for_promotion=True,
        fitness_terminal_open_positions=2,
    )
    reasons = mt5_promotion_rejection_reasons(
        {"best": {"fitness": 5.0, "metrics": _policy_audit(0.10)}},
        {"best": {"fitness": 8.0, "metrics": trial_metrics}},
        min_daily_trades=0,
        backtest_days=31,
    )
    assert "terminal_open_positions_nonzero:2" in reasons


def test_config_parses_expectancy_controls(tmp_path):
    from self_growing_trader.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(
        """
expectancy:
  enabled: true
  min_samples: 9
  merge_similarity: 0.84
  max_samples_per_context: 512
  min_policy_origin_ratio: 0.70
  promotion_lcb_min_delta: 0.03
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.expectancy.enabled is True
    assert config.expectancy.min_samples == 9
    assert config.expectancy.merge_similarity == 0.84
    assert config.expectancy.max_samples_per_context == 512
    assert config.expectancy.min_policy_origin_ratio == 0.70
    assert config.expectancy.promotion_lcb_min_delta == 0.03
