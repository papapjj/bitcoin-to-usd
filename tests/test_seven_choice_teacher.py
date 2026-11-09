from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from self_growing_trader.orders import Action
from self_growing_trader.five_shared_policy import (
    ACCOUNT_EMBED_SIZE,
    EXECUTION_EMBED_SIZE,
    MARKET_STRUCTURE_EMBED_SIZE,
    ObservationPatternTree,
    POSITION_EMBED_SIZE,
    SharedFiveTradingPolicy,
)
from self_growing_trader.protocol_contract import (
    INPUT_COUNT,
)
from self_growing_trader.seven_choice_teacher import (
    SevenChoiceBuildConfig,
    _causal_action_targets,
    _fixed_geometry_ticks,
    _normalized_pending_price,
    _placement_on_ticks,
    build_seven_choice_dataset,
    load_seven_choice_dataset,
    _require_hidden_cache_metadata,
    seven_choice_runtime_contract,
)


def _write_snapshots(path: Path, rows: int = 180) -> None:
    header = [
        "schema_version", "feature_schema_hash", "sample_time", "symbol", "timeframe",
        "open_bid", "high_bid", "low_bid", "close_bid", "atr", "spread_price",
        "point", "pending_min_distance", "digits", "causal_input",
        "decision_bar_time", "tick_count", "tick_time_msc_pipe",
        "tick_bid_pipe", "tick_ask_pipe", "inputs_pipe",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for index in range(rows):
            price = 100.0 + np.sin(index / 8.0) * 2.0 + index * 0.01
            sample_time = (datetime(2025, 1, 1) + timedelta(minutes=index)).strftime(
                "%Y.%m.%d %H:%M:%S"
            )
            decision_time = (datetime(2025, 1, 1) + timedelta(minutes=index + 1)).strftime(
                "%Y.%m.%d %H:%M:%S"
            )
            tick_bid = np.asarray([
                price + 0.10, price + 0.25, price - 0.15,
                price + 0.35, price - 0.30, price + 0.20,
            ])
            tick_ask = tick_bid + 0.05
            inputs = "|".join([f"{index / rows:.8f}"] + ["0"] * 136)
            values = [
                "2", "five_observation_v10_137", sample_time, "TEST", "M1",
                f"{price:.5f}", f"{price + 0.4:.5f}", f"{price - 0.4:.5f}",
                f"{price:.5f}", "1.0", "0.05", "0.01", "0.02", "2", "1",
                decision_time, str(len(tick_bid)),
                "|".join(str(1_000_000 + index * 100 + tick) for tick in range(len(tick_bid))),
                "|".join(f"{value:.5f}" for value in tick_bid),
                "|".join(f"{value:.5f}" for value in tick_ask), inputs,
            ]
            handle.write("\t".join(values) + "\n")


def test_builder_emits_complete_seven_choice_cost_aware_contract(tmp_path) -> None:
    source = tmp_path / "snapshots.tsv"
    output = tmp_path / "teacher.npz"
    _write_snapshots(source)
    manifest = build_seven_choice_dataset(
        source, output,
        config=SevenChoiceBuildConfig(horizon_bars=8, pending_expiry_bars=2),
    )
    data = load_seven_choice_dataset(output)

    assert manifest["choice_count"] == 7
    assert manifest["source_feature_schema_hashes"] == ["five_observation_v10_137"]
    assert data.net_r.shape[1] == 7
    assert data.outcome_best_actions.shape == data.best_actions.shape
    assert data.filled.shape == data.net_r.shape
    assert set(np.unique(data.agent_ids)) == {0, 1}
    assert set(np.unique(data.splits)) == {"train", "validation", "test", "purged"}
    assert np.all(data.allowed_action_mask[data.agent_ids == 0][:, [0, 1, 3, 5]])
    assert np.all(data.allowed_action_mask[data.agent_ids == 1][:, [0, 2, 4, 6]])
    assert np.all(data.allowed_action_mask[np.arange(len(data.best_actions)), data.best_actions])
    assert np.all(data.allowed_action_mask[np.arange(len(data.outcome_best_actions)), data.outcome_best_actions])
    assert manifest["outcome_target_audit"]["used_as_model_input"] is False
    audit = manifest["causal_selection_audit"]
    assert manifest["config"]["fixed_entry_offset_effective_atr"] == pytest.approx(
        audit["fixed_entry_offset_atr"]
    )
    assert audit["placement_search_uses_future_terminal_price"] is False
    assert audit["geometry_search_uses_future_path"] is False
    assert audit["action_search_uses_current_sample_future"] is False
    assert audit["production_eligible"] is True
    assert np.isfinite(data.action_scores_r).all()

    entry = data.filled[:, 1:7]
    assert np.any(entry)
    assert np.all(data.order_prices[:, 1:7][entry] > 0.0)
    assert np.all(data.entry_prices[:, 1:7][entry] > 0.0)
    market_buy = data.filled[:, int(Action.MARKET_BUY)]
    np.testing.assert_allclose(
        data.entry_prices[market_buy, int(Action.MARKET_BUY)],
        data.order_prices[market_buy, int(Action.MARKET_BUY)],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        data.gross_r[:, 1:7][entry] - data.net_r[:, 1:7][entry],
        data.spread_cost_r[:, 1:7][entry],
        atol=1e-5,
    )
    assert np.all(data.fill_wait_bars[:, 1:7][entry] >= 0)
    assert np.all(data.fill_wait_ticks[:, 1:7][entry] >= 0)
    assert np.all(data.exit_bars[:, 1:7][entry] > 0)
    assert np.all(data.mfe_r[:, 1:7][entry] >= 0.0)
    assert np.all(data.mae_r[:, 1:7][entry] <= 0.0)
    assert manifest["choices"][str(int(Action.SELL_STOP))] == "SELL_STOP"


def test_builder_matures_causal_pending_outcomes_after_fill_expiry(tmp_path, monkeypatch) -> None:
    import self_growing_trader.seven_choice_teacher as teacher

    source = tmp_path / "snapshots.tsv"
    output = tmp_path / "teacher_fill_horizon.npz"
    _write_snapshots(source, rows=180)
    observed: dict[str, int] = {}
    original = teacher._causal_action_targets

    def wrapped(net_r, regimes, **kwargs):
        observed["horizon_bars"] = int(kwargs["horizon_bars"])
        return original(net_r, regimes, **kwargs)

    monkeypatch.setattr(teacher, "_causal_action_targets", wrapped)
    manifest = build_seven_choice_dataset(
        source,
        output,
        config=SevenChoiceBuildConfig(
            horizon_bars=8,
            pending_expiry_bars=2,
            holding_horizon_from_fill=True,
        ),
    )

    assert observed["horizon_bars"] == 10
    assert manifest["causal_selection_audit"]["lookahead_bars"] == 10


def test_causal_action_target_cannot_use_current_or_unmatured_future() -> None:
    net = np.zeros((160, 7), dtype=np.float32)
    net[:80, int(Action.MARKET_BUY)] = 1.0
    regimes = np.zeros((160, 4), dtype=np.int8)
    before, scores_before, _ = _causal_action_targets(
        net, regimes, horizon_bars=8, minimum_count=16,
        lower_bound_z=1.0, minimum_lcb_r=0.05,
    )
    changed = net.copy()
    changed[120:, int(Action.MARKET_BUY)] = -100.0
    changed[120:, int(Action.MARKET_SELL)] = 100.0
    after, scores_after, _ = _causal_action_targets(
        changed, regimes, horizon_bars=8, minimum_count=16,
        lower_bound_z=1.0, minimum_lcb_r=0.05,
    )

    # Rows through 127 cannot observe any of the changed outcomes because an
    # eight-bar horizon has not completed yet.
    np.testing.assert_array_equal(before[:256], after[:256])
    np.testing.assert_allclose(scores_before[:256], scores_after[:256])


def test_causal_action_history_stride_skips_overlapping_outcomes() -> None:
    net = np.zeros((40, 7), dtype=np.float32)
    net[0, int(Action.MARKET_BUY)] = 1.0
    net[1, int(Action.MARKET_BUY)] = -1.0
    regimes = np.zeros((40, 4), dtype=np.int8)
    dense, _, _ = _causal_action_targets(
        net, regimes, horizon_bars=1, minimum_count=1,
        lower_bound_z=0.0, minimum_lcb_r=-2.0,
    )
    blocked, _, audit = _causal_action_targets(
        net, regimes, horizon_bars=1, minimum_count=1,
        lower_bound_z=0.0, history_stride_bars=2, minimum_lcb_r=-2.0,
    )
    assert audit["history_stride_bars"] == 2
    # The dense history sees both matured rows; the strided history only
    # admits the first row, so the later negative outcome cannot alter it.
    assert dense[4] == int(Action.HOLD)
    assert blocked[4] == int(Action.MARKET_BUY)


def test_outcome_best_actions_are_dense_role_legal_offline_labels() -> None:
    from self_growing_trader.seven_choice_teacher import _outcome_best_actions

    net = np.zeros((3, 7), dtype=np.float32)
    net[0, int(Action.MARKET_BUY)] = 0.8
    net[0, int(Action.MARKET_SELL)] = 0.9
    net[1, int(Action.SELL_LIMIT)] = 0.7
    net[2, int(Action.MARKET_BUY)] = -0.2
    labels, audit = _outcome_best_actions(net)

    # Each role sees only its own side; HOLD wins when every legal result is
    # non-positive.  The target is explicitly marked as future-derived label
    # data rather than a causal input.
    assert labels[:, 0].tolist() == [int(Action.MARKET_BUY), 0, 0]
    assert labels[:, 1].tolist() == [int(Action.MARKET_SELL), int(Action.SELL_LIMIT), 0]
    assert audit["used_as_model_input"] is False


def test_hidden_cache_requires_exact_provenance(tmp_path) -> None:
    metadata = tmp_path / "hidden.f16.mmap.json"
    expected = {
        "schema_version": 1,
        "dataset_sha256": "abc",
        "rows": 12,
        "hidden_size": 8,
    }
    metadata.write_text(json.dumps(expected, indent=2), encoding="utf-8")

    _require_hidden_cache_metadata(metadata, expected)

    changed = dict(expected, dataset_sha256="def")
    with pytest.raises(RuntimeError, match="khong khop"):
        _require_hidden_cache_metadata(metadata, changed)


def test_tick_delay_and_pending_price_follow_mt5_quote_side() -> None:
    bid = np.asarray([100.0, 100.1, 100.3, 100.5, 100.4, 100.2])
    ask = bid + 0.2
    bars = np.asarray([0, 4, 6])
    offsets = np.asarray([0.45])

    market = _placement_on_ticks(
        int(Action.MARKET_BUY), 3, 0, 100.0, 1.0, 0.02, 2,
        offsets, bid, ask, bars, 1,
    )
    assert market is not None
    assert market[:5] == (3, 100.7, 100.7, 100.5, 0)

    stop = _placement_on_ticks(
        int(Action.BUY_STOP), 0, 0, 100.0, 1.0, 0.02, 2,
        offsets, bid, ask, bars, 1,
    )
    assert stop is not None
    assert stop[0] == 2  # Ask first crosses the 100.45 order price here.
    assert stop[1] == 100.45
    assert stop[2] == 100.5
    assert stop[4] == 2
    assert _normalized_pending_price(
        int(Action.BUY_LIMIT), 101.0, 100.0, 100.2, 0.02, 2,
    ) == 100.18


def test_pending_order_mode_uses_requested_price_like_mt5() -> None:
    bid = np.asarray([100.0, 100.1, 100.3, 100.5, 100.4, 100.2])
    ask = bid + 0.2
    bars = np.asarray([0, 4, 6])
    offsets = np.asarray([0.45])
    stop = _placement_on_ticks(
        int(Action.BUY_STOP), 0, 0, 100.0, 1.0, 0.02, 2,
        offsets, bid, ask, bars, 1, "order",
    )
    assert stop is not None
    assert stop[0] == 2
    assert stop[1] == pytest.approx(100.45)
    assert stop[2] == pytest.approx(100.45)
    assert stop[3] == pytest.approx(100.25)


def test_fill_risk_denominator_matches_gap_anchored_mt5_lifecycle() -> None:
    bid = np.asarray([99.0, 99.5, 99.7], dtype=np.float64)
    ask = bid + 0.1
    legacy = _fixed_geometry_ticks(
        int(Action.BUY_LIMIT), 100.0, 99.0, 98.9, 1.0,
        bid, ask, 2, 5, geometry_price=100.0,
    )
    fill_anchored = _fixed_geometry_ticks(
        int(Action.BUY_LIMIT), 100.0, 99.0, 98.9, 1.0,
        bid, ask, 2, 5, geometry_price=100.0,
        risk_denominator_from_fill=True,
    )
    assert fill_anchored["net_r"] > legacy["net_r"]
    assert fill_anchored["gross_r"] > legacy["gross_r"]


def test_zero_initialized_raw_observation_adapter_preserves_legacy_surface() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    tree = ObservationPatternTree()
    batch = 3
    kwargs = {
        "market": torch.randn(batch, MARKET_STRUCTURE_EMBED_SIZE),
        "positions": torch.randn(batch, POSITION_EMBED_SIZE),
        "account": torch.randn(batch, ACCOUNT_EMBED_SIZE),
        "execution": torch.randn(batch, EXECUTION_EMBED_SIZE),
    }
    raw = torch.randn(batch, INPUT_COUNT)
    without = tree(**kwargs)["pattern_action_path"]
    with_raw = tree(**kwargs, raw_observations=raw)["pattern_action_path"]
    torch.testing.assert_close(with_raw, without)


def test_direct_seven_choice_head_starts_as_legacy_and_leaves_managers_untouched() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.require_current_architecture()
    hidden = torch.randn(5, 8)
    observations = torch.randn(5, INPUT_COUNT)
    agents = torch.arange(5, dtype=torch.long)
    primary = network.heads.primary_action_logits(hidden, agents)
    direct = network.heads.direct_action_logits(hidden, agents, observations)
    torch.testing.assert_close(direct, primary)
    with torch.no_grad():
        network.heads.direct_action_head[-1].bias[1] = 1.0
    changed = network.heads.direct_action_logits(hidden, agents, observations)
    assert changed[0, 1] > direct[0, 1]
    torch.testing.assert_close(changed[2:], primary[2:])


def test_runtime_contract_requires_the_same_finite_lifecycle_geometry() -> None:
    manifest = {
        "config": {
            "horizon_bars": 48,
            "pending_expiry_bars": 5,
            "holding_horizon_from_fill": True,
            "risk_denominator_from_fill": False,
            "pending_fill_price_mode": "order",
            "entry_offset_cap_atr": 5.0,
            "fixed_sl_atr": 1.5,
            "fixed_tp_rr": 2.0,
        }
    }
    matching = SimpleNamespace(
        orders=SimpleNamespace(
            max_holding_bars=48,
            pending_expiry_bars=5,
            entry_offset_atr=5.0,
            fixed_sl_atr=1.5,
            fixed_tp_atr=3.0,
            technical_min_rr=2.0,
        )
    )
    result = seven_choice_runtime_contract(manifest, matching)
    assert result["match"] is True
    assert result["mismatches"] == {}

    mismatching = SimpleNamespace(
        orders=SimpleNamespace(
            max_holding_bars=0,
            pending_expiry_bars=60,
            entry_offset_atr=5.0,
            fixed_sl_atr=2.5,
            fixed_tp_atr=5.0,
            technical_min_rr=2.0,
        )
    )
    result = seven_choice_runtime_contract(manifest, mismatching)
    assert result["match"] is False
    assert set(result["mismatches"]) == {
        "pending_expiry_bars", "fixed_sl_atr", "max_holding_bars",
    }


def test_runtime_contract_rejects_a_dataset_from_another_feature_profile() -> None:
    from self_growing_trader.config import FeatureConfig, SymbolConfig
    from self_growing_trader.feature_contract import feature_schema_fingerprint

    features = FeatureConfig()
    symbol = SymbolConfig()
    expected = feature_schema_fingerprint(
        features, symbol, input_normalization_mode="raw"
    )
    manifest = {
        "source_feature_schema_hashes": ["five_observation_v10_137"],
        "config": {
            "horizon_bars": 48,
            "pending_expiry_bars": 5,
            "holding_horizon_from_fill": True,
            "risk_denominator_from_fill": False,
            "pending_fill_price_mode": "order",
            "entry_offset_cap_atr": 5.0,
            "fixed_sl_atr": 1.5,
            "fixed_tp_rr": 2.0,
        },
    }
    config = SimpleNamespace(
        features=features,
        symbol=symbol,
        learning=SimpleNamespace(input_normalization_mode="raw"),
        orders=SimpleNamespace(
            max_holding_bars=48,
            pending_expiry_bars=5,
            entry_offset_atr=5.0,
            fixed_sl_atr=1.5,
            fixed_tp_atr=3.0,
            technical_min_rr=2.0,
        ),
    )
    result = seven_choice_runtime_contract(manifest, config)
    assert result["checks"]["feature_schema"]["runtime"] == expected
    assert result["checks"]["feature_schema"]["matches"] is False
    assert "feature_schema" in result["mismatches"]


def test_runtime_contract_rejects_legacy_touch_fill_labels() -> None:
    manifest = {
        "config": {
            "horizon_bars": 48,
            "pending_expiry_bars": 5,
            "holding_horizon_from_fill": True,
            "risk_denominator_from_fill": False,
            "pending_fill_price_mode": "touch",
            "entry_offset_cap_atr": 5.0,
            "fixed_sl_atr": 2.5,
            "fixed_tp_rr": 2.0,
        }
    }
    config = SimpleNamespace(
        orders=SimpleNamespace(
            max_holding_bars=48,
            pending_expiry_bars=5,
            entry_offset_atr=5.0,
            fixed_sl_atr=2.5,
            fixed_tp_atr=5.0,
            technical_min_rr=2.0,
        )
    )
    result = seven_choice_runtime_contract(manifest, config)
    assert result["match"] is False
    assert result["mismatches"]["pending_fill_price_mode"]["dataset"] == "touch"


def test_manifest_describes_order_price_pending_fill_mode(tmp_path) -> None:
    source = tmp_path / "snapshots.tsv"
    output = tmp_path / "teacher.npz"
    _write_snapshots(source)
    manifest = build_seven_choice_dataset(
        source,
        output,
        config=SevenChoiceBuildConfig(
            horizon_bars=8,
            pending_expiry_bars=2,
            pending_fill_price_mode="order",
        ),
    )
    assert "dung gia dat nhu MT5" in manifest["label_contract"]["pending_fill"]


def test_v193_counterfactual_profile_matches_the_candidate_feature_contract() -> None:
    from pathlib import Path

    profile = Path(
        "mt5_v2_runner/MQL5/Profiles/Tester/BottomTop_Counterfactual_16w_v193_raw.set"
    ).read_text(encoding="utf-8")
    assert "FeatureAtrPeriod=30||30" in profile
    assert "FeatureEmaFast=50||50" in profile
    assert "FeatureEmaSlow=200||200" in profile
    assert "FeatureMtfScalingMode=native_tf_atr_v2" in profile
    assert "FeatureSchemaHash=d5fd9dec6381657e3d0728baa18353f7fab1cf3651f044f1bea449f030e3a402" in profile


@pytest.mark.parametrize("target_mode", ["causal", "outcome"])
def test_training_replay_preserves_causal_actions_before_relabelling(
    tmp_path, monkeypatch, target_mode: str,
) -> None:
    from dataclasses import replace

    import torch
    import self_growing_trader.seven_choice_teacher as teacher

    source = tmp_path / "snapshots.tsv"
    output = tmp_path / "teacher.npz"
    _write_snapshots(source)
    build_seven_choice_dataset(
        source, output,
        config=SevenChoiceBuildConfig(horizon_bars=8, pending_expiry_bars=2),
    )
    data = load_seven_choice_dataset(output)
    causal_actions = np.tile([1, 2], len(data.observations) // 2)
    outcome_actions = np.tile([3, 4], len(data.observations) // 2)
    value_teacher_actions = np.tile([5, 6], len(data.observations) // 2)
    data = replace(data, best_actions=causal_actions, outcome_best_actions=outcome_actions)

    def fake_value_teacher(network, original, **kwargs):
        del network, kwargs
        return replace(original, best_actions=value_teacher_actions), None, {
            "selected_threshold": 0.0,
            "validation_gate_passed": True,
            "test_gate_passed": True,
        }

    class ExtractionChecked(Exception):
        pass

    def check_extraction(network, pseudo, **kwargs):
        del network, kwargs
        np.testing.assert_array_equal(pseudo.actions, causal_actions)
        np.testing.assert_array_equal(
            pseudo.candidate_actions,
            outcome_actions if target_mode == "outcome" else causal_actions,
        )
        assert not np.shares_memory(pseudo.actions, data.best_actions)
        raise ExtractionChecked

    monkeypatch.setattr(teacher, "_fit_expected_value_teacher", fake_value_teacher)
    monkeypatch.setattr(teacher, "extract_owner_hidden_sequences", check_extraction)
    torch.manual_seed(7)
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    with pytest.raises(ExtractionChecked):
        teacher.train_seven_choice_heads(
            network, data, epochs=1, expected_value_epochs=1,
            action_target_mode=target_mode, allow_research_only=True,
            sequence_length=4,
        )
    np.testing.assert_array_equal(data.best_actions, causal_actions)
