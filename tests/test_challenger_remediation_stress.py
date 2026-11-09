"""Challenger Remediation Stress Test Suite.

Empirical verification and adversarial challenge suite targeting:
1. Bidirectional scale normalization: `_normalize_comparison_pct`, `WinRate`, and `DrawdownPct`
   with boundary values (0.0, 1.0, 100.0, negative, large, NaN, None, mixed types).
2. Fail-closed behavior in `AutoRLEngine._default_train_round` and `_default_oos_eval`
   on crashed runs, zero trades, negative trades, and missing metrics.
3. Precise edge cases near hurdle boundaries in `evaluate_candidate_oos_gate`
   (E[R] = 0.0499 vs 0.0501, WR = 39.9% vs 40.0%, DD = 9.99% vs 10.01%, trades = 29 vs 30).
4. Adversarial discovery & vulnerability verification: NaN metric handling in
   `evaluate_candidate_oos_gate` vs `OOSValidator.evaluate`.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.self_growing_trader.auto_rl_engine import (
    AutoRLEngine,
    AutoRLEngineConfig,
    TimeframeAdaptationStateMachine,
    evaluate_candidate_oos_gate,
)
from src.self_growing_trader.oos_validator import (
    DrawdownPct,
    OOSValidator,
    WinRate,
    _normalize_comparison_pct,
)


# ==============================================================================
# PART 1: _normalize_comparison_pct, WinRate, and DrawdownPct ADVERSARIAL TESTS
# ==============================================================================


class TestNormalizeComparisonPctAdversarial:
    """Boundary and adversarial stress tests for _normalize_comparison_pct."""

    def test_boundary_zero_one_hundred(self) -> None:
        """Verify normalization at 0.0, 1.0, and 100.0."""
        # 0.0 is within [0.0, 1.0], normalized to 0.0 * 100.0 = 0.0
        assert _normalize_comparison_pct(0.0) == 0.0
        # 1.0 is within [0.0, 1.0], normalized to 1.0 * 100.0 = 100.0
        assert _normalize_comparison_pct(1.0) == 100.0
        # 100.0 is > 1.0, left as 100.0
        assert _normalize_comparison_pct(100.0) == 100.0
        # 1.0 and 100.0 normalize to the exact same value
        assert _normalize_comparison_pct(1.0) == _normalize_comparison_pct(100.0)

    def test_negative_and_large_numbers(self) -> None:
        """Verify behavior on negative numbers and numbers > 100."""
        # Negative numbers are < 0.0, so not in [0.0, 1.0] -> returned as-is
        assert _normalize_comparison_pct(-0.01) == -0.01
        assert _normalize_comparison_pct(-1.0) == -1.0
        assert _normalize_comparison_pct(-50.0) == -50.0

        # Numbers > 100.0 are returned as-is
        assert _normalize_comparison_pct(150.0) == 150.0
        assert _normalize_comparison_pct(1000.0) == 1000.0
        assert _normalize_comparison_pct(1e6) == 1e6

    def test_nan_and_infinities(self) -> None:
        """Verify behavior on NaN and Inf."""
        # NaN comparisons in (0.0 <= nan <= 1.0) evaluate to False, returns nan
        res_nan = _normalize_comparison_pct(float("nan"))
        assert math.isnan(res_nan)

        # Inf and -Inf
        assert _normalize_comparison_pct(float("inf")) == float("inf")
        assert _normalize_comparison_pct(float("-inf")) == float("-inf")

    def test_none_and_invalid_types_raise_type_error(self) -> None:
        """Verify None and unparseable types raise TypeError / ValueError."""
        with pytest.raises(TypeError):
            _normalize_comparison_pct(None)

        with pytest.raises(ValueError):
            _normalize_comparison_pct("non_numeric")

        with pytest.raises(TypeError):
            _normalize_comparison_pct([0.4])

        with pytest.raises(TypeError):
            _normalize_comparison_pct({"val": 0.4})

    def test_string_numeric_coercion(self) -> None:
        """Verify string numbers coerce correctly."""
        assert _normalize_comparison_pct("0.40") == 40.0
        assert _normalize_comparison_pct("40.0") == 40.0
        assert _normalize_comparison_pct("1.0") == 100.0
        assert _normalize_comparison_pct("0.0") == 0.0

    def test_boolean_coercion(self) -> None:
        """Verify bool values coerce as 1.0 and 0.0."""
        # True is 1 -> 100.0
        assert _normalize_comparison_pct(True) == 100.0
        # False is 0 -> 0.0
        assert _normalize_comparison_pct(False) == 0.0

    def test_boundary_ambiguity_near_one(self) -> None:
        """Document and verify behavior between 1.0 and 1.0001."""
        # 1.0 is treated as fraction 100%
        assert _normalize_comparison_pct(1.0) == 100.0
        # 1.0001 is > 1.0, so treated as percentage 1.0001%
        assert _normalize_comparison_pct(1.0001) == 1.0001


class TestWinRateAndDrawdownPctAdversarial:
    """Boundary and adversarial stress tests for WinRate and DrawdownPct classes."""

    def test_win_rate_all_comparisons_and_boundaries(self) -> None:
        """Verify WinRate rich comparisons across fractional and percentage scales."""
        wr_frac = WinRate(0.40)
        wr_pct = WinRate(40.0)

        # Equality across scales
        assert wr_frac == wr_pct
        assert wr_frac == 40.0
        assert wr_frac == 0.40
        assert wr_pct == 0.40
        assert wr_pct == 40.0

        # Greater than / Greater equal
        assert wr_frac >= 0.40
        assert wr_frac >= 40.0
        assert wr_frac >= 39.9
        assert not (wr_frac >= 40.1)

        assert wr_frac > 39.9
        assert wr_frac > 0.399
        assert not (wr_frac > 40.0)

        # Less than / Less equal
        assert wr_frac <= 40.0
        assert wr_frac <= 0.40
        assert wr_frac <= 40.1
        assert not (wr_frac <= 39.9)

        assert wr_frac < 40.1
        assert wr_frac < 0.401
        assert not (wr_frac < 40.0)

        # Extreme boundaries
        wr_zero = WinRate(0.0)
        assert wr_zero == 0.0
        assert wr_zero >= 0.0
        assert not (wr_zero > 0.0)

        wr_one = WinRate(1.0)
        assert wr_one == 100.0
        assert wr_one >= 100.0
        assert wr_one >= 1.0

        wr_hundred = WinRate(100.0)
        assert wr_hundred == 1.0
        assert wr_hundred == 100.0

    def test_drawdown_pct_all_comparisons_and_boundaries(self) -> None:
        """Verify DrawdownPct rich comparisons across fractional and percentage scales."""
        dd_frac = DrawdownPct(0.085)
        dd_pct = DrawdownPct(8.5)

        # Equality across scales
        assert dd_frac == dd_pct
        assert dd_frac == 8.5
        assert dd_frac == 0.085
        assert dd_pct == 0.085
        assert dd_pct == 8.5

        # Less than hurdle (10.0%)
        assert dd_pct < 10.0
        assert dd_pct < 0.10
        assert dd_frac < 10.0
        assert dd_frac < 0.10

        # Boundary around 10.0%
        assert DrawdownPct(9.99) < 10.0
        assert DrawdownPct(0.0999) < 10.0
        assert DrawdownPct(0.0999) < 0.10

        assert not (DrawdownPct(10.00) < 10.0)
        assert not (DrawdownPct(0.1000) < 10.0)
        assert not (DrawdownPct(0.1000) < 0.10)

        assert not (DrawdownPct(10.01) < 10.0)
        assert not (DrawdownPct(0.1001) < 10.0)

        # 1.0 is normalized to 100.0% (total account loss)
        assert not (DrawdownPct(1.0) < 10.0)
        assert DrawdownPct(1.0) == 100.0

    def test_nan_comparisons_are_false(self) -> None:
        """Verify comparisons against NaN evaluate to False as per IEEE-754."""
        nan_val = float("nan")
        wr = WinRate(0.40)
        dd = DrawdownPct(5.0)

        assert not (wr >= nan_val)
        assert not (wr > nan_val)
        assert not (wr <= nan_val)
        assert not (wr < nan_val)
        assert not (wr == nan_val)
        assert wr != nan_val

        assert not (dd < nan_val)
        assert not (dd <= nan_val)
        assert not (dd > nan_val)
        assert not (dd >= nan_val)
        assert not (dd == nan_val)
        assert dd != nan_val

        wr_nan = WinRate(nan_val)
        assert not (wr_nan >= 40.0)
        assert not (wr_nan == 40.0)
        assert not (wr_nan == wr_nan)

    def test_invalid_types_raise_type_error(self) -> None:
        """Verify comparisons against None and unsupported types raise TypeError."""
        wr = WinRate(0.40)
        dd = DrawdownPct(5.0)

        with pytest.raises(TypeError):
            _ = wr >= None

        with pytest.raises(TypeError):
            _ = wr < "40.0"

        with pytest.raises(TypeError):
            _ = dd < None

        with pytest.raises(TypeError):
            _ = dd > [10.0]

    def test_unhashable_type_behavior(self) -> None:
        """Verify that overriding __eq__ makes WinRate and DrawdownPct unhashable."""
        wr = WinRate(0.40)
        dd = DrawdownPct(5.0)

        with pytest.raises(TypeError, match="unhashable type"):
            hash(wr)

        with pytest.raises(TypeError, match="unhashable type"):
            hash(dd)


# ==============================================================================
# PART 2: FAIL-CLOSED STRESS TESTS FOR AutoRLEngine
# ==============================================================================


class TestAutoRLEngineFailClosed:
    """Empirically stress-test AutoRLEngine._default_train_round and _default_oos_eval."""

    @pytest.fixture
    def engine(self) -> AutoRLEngine:
        with tempfile.TemporaryDirectory() as td:
            cfg = AutoRLEngineConfig(runs_dir=Path(td), max_rounds=3)
            return AutoRLEngine(cfg)

    def test_train_round_raises_runtime_error_on_trainer_crash(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify _default_train_round fails closed (raises RuntimeError) when trainer crashes."""
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            with patch(
                "self_growing_trader.chuyen_gia_policy.huan_luyen_he_chuyen_gia_mt5",
                side_effect=RuntimeError("MT5 terminal synchronization crashed"),
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    engine._default_train_round(
                        round_num=1,
                        timeframe="M1",
                        seed_path="dummy_seed.pkl",
                        round_dir=round_dir,
                    )
                assert "Multi-round training failed during Round 1 (M1)" in str(exc_info.value)

    def test_train_round_raises_runtime_error_on_zero_trades(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify _default_train_round fails closed when MT5 reports 0 trades."""
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            mock_train_res = {"metrics": {"total_trades": 0, "expectancy": 0.0}}
            with patch(
                "self_growing_trader.chuyen_gia_policy.huan_luyen_he_chuyen_gia_mt5",
                return_value=mock_train_res,
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    engine._default_train_round(
                        round_num=1,
                        timeframe="M1",
                        seed_path="dummy_seed.pkl",
                        round_dir=round_dir,
                    )
                assert "produced 0 trades or invalid metrics" in str(exc_info.value)

    def test_train_round_raises_runtime_error_on_negative_trades(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify _default_train_round fails closed when MT5 reports negative trades."""
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            mock_train_res = {"metrics": {"total_trades": -5, "expectancy": -0.5}}
            with patch(
                "self_growing_trader.chuyen_gia_policy.huan_luyen_he_chuyen_gia_mt5",
                return_value=mock_train_res,
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    engine._default_train_round(
                        round_num=1,
                        timeframe="M1",
                        seed_path="dummy_seed.pkl",
                        round_dir=round_dir,
                    )
                assert "produced 0 trades or invalid metrics" in str(exc_info.value)

    def test_train_round_raises_runtime_error_on_missing_metrics(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify _default_train_round fails closed when MT5 returns empty dict or None metrics."""
        with tempfile.TemporaryDirectory() as td:
            round_dir = Path(td)
            for bad_res in [{}, {"metrics": None}, {"metrics": {}}, None]:
                with patch(
                    "self_growing_trader.chuyen_gia_policy.huan_luyen_he_chuyen_gia_mt5",
                    return_value=bad_res,
                ):
                    with pytest.raises(RuntimeError) as exc_info:
                        engine._default_train_round(
                            round_num=1,
                            timeframe="M1",
                            seed_path="dummy_seed.pkl",
                            round_dir=round_dir,
                        )
                    assert (
                        "Multi-round training failed" in str(exc_info.value)
                        or "produced 0 trades or invalid metrics" in str(exc_info.value)
                    )

    def test_oos_eval_fails_closed_on_missing_candidate_file(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify _default_oos_eval returns fail-closed payload if candidate file does not exist."""
        non_existent_file = Path("non_existent_dir_12345/candidate.pkl")
        res = engine._default_oos_eval(non_existent_file, "M1")

        assert res["passed_gate"] is False
        assert res["oos_executed"] is False
        assert res["total_trades"] == 0
        assert res["expectancy"] == -999.0
        assert "candidate_file_not_found" in res["reasons"]

    def test_oos_eval_fails_closed_on_evaluator_crash(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify _default_oos_eval returns fail-closed payload if MT5 OOS evaluation raises an exception."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            candidate_path = Path(f.name)

        try:
            with patch(
                "self_growing_trader.chuyen_gia_policy.tai_he_chuyen_gia_mot_context_mot_policy",
                return_value=MagicMock(),
            ), patch(
                "self_growing_trader.chuyen_gia_policy.evaluate_policy_mt5_oos",
                side_effect=TimeoutError("Strategy Tester timed out after 300s"),
            ):
                res = engine._default_oos_eval(candidate_path, "M1")
                assert res["passed_gate"] is False
                assert res["oos_executed"] is False
                assert res["total_trades"] == 0
                assert res["expectancy"] == -999.0
                assert any("oos_evaluation_crashed" in r for r in res["reasons"])
        finally:
            if candidate_path.exists():
                candidate_path.unlink()

    def test_oos_eval_fails_closed_on_invalid_return_type(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify _default_oos_eval returns fail-closed payload if evaluator returns non-dict."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            candidate_path = Path(f.name)

        try:
            with patch(
                "self_growing_trader.chuyen_gia_policy.tai_he_chuyen_gia_mot_context_mot_policy",
                return_value=MagicMock(),
            ), patch(
                "self_growing_trader.chuyen_gia_policy.evaluate_policy_mt5_oos",
                return_value="unexpected_string_output",
            ):
                res = engine._default_oos_eval(candidate_path, "M1")
                assert res["passed_gate"] is False
                assert res["oos_executed"] is False
                assert res["reasons"] == ["invalid_oos_return_type"]
        finally:
            if candidate_path.exists():
                candidate_path.unlink()

    def test_step_round_rejects_candidate_when_oos_fails(
        self, engine: AutoRLEngine
    ) -> None:
        """Verify step_round refuses promotion when OOS evaluation fails closed."""
        def mock_executor(round_num: int, tf: str, seed: str, rdir: Path):
            ckpt = rdir / "candidate.pkl"
            ckpt.write_text("dummy")
            return ckpt, {"ppo_loss": 0.5, "total_trades": 50}

        def mock_crashed_oos(candidate: Any, tf: str):
            return {
                "passed_gate": False,
                "oos_executed": False,
                "total_trades": 0,
                "expectancy": -999.0,
                "reasons": ["oos_evaluation_crashed"],
            }

        res = engine.step_round(
            round_executor=mock_executor,
            oos_evaluator=mock_crashed_oos,
        )
        assert res["promoted"] is False
        assert res["gate_passed"] is False


# ==============================================================================
# PART 3: HURDLE BOUNDARY EDGE CASES IN evaluate_candidate_oos_gate
# ==============================================================================


class TestEvaluateCandidateOOSGateHurdleBoundaries:
    """Adversarial boundary stress tests for evaluate_candidate_oos_gate."""

    @pytest.fixture
    def base_passing_metrics(self) -> dict[str, Any]:
        return {
            "total_trades": 30,
            "expectancy": 0.05,
            "win_rate": 40.0,
            "risk_reward_ratio": 2.0,
            "max_equity_drawdown_pct": 9.99,
            "hold_share": 0.799,
            "entry_direction_imbalance": 0.60,
            "entropy": 0.3501,
        }

    def test_baseline_passes_all_criteria(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify baseline boundary metrics satisfy all gate criteria."""
        res = evaluate_candidate_oos_gate(base_passing_metrics)
        assert res.passed_gate is True
        assert len(res.reasons) == 0

    def test_expectancy_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify E[R] boundary: 0.0499 must fail, 0.0500 and 0.0501 must pass."""
        # 0.0499 fails
        res_fail = evaluate_candidate_oos_gate({**base_passing_metrics, "expectancy": 0.0499})
        assert res_fail.passed_gate is False
        assert any("expectancy_below_hurdle:0.0499<0.0500" in r for r in res_fail.reasons)

        # 0.0500 passes
        res_pass_exact = evaluate_candidate_oos_gate({**base_passing_metrics, "expectancy": 0.0500})
        assert res_pass_exact.passed_gate is True

        # 0.0501 passes
        res_pass_above = evaluate_candidate_oos_gate({**base_passing_metrics, "expectancy": 0.0501})
        assert res_pass_above.passed_gate is True

    def test_win_rate_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify WR boundary: 39.9% (0.399) fails, 40.0% (0.400) passes, 40.1% (0.401) passes."""
        # 39.9% fails
        res_fail_pct = evaluate_candidate_oos_gate({**base_passing_metrics, "win_rate": 39.9})
        assert res_fail_pct.passed_gate is False
        assert any("win_rate_below_hurdle:39.90%<40.00%" in r for r in res_fail_pct.reasons)

        res_fail_frac = evaluate_candidate_oos_gate({**base_passing_metrics, "win_rate": 0.399})
        assert res_fail_frac.passed_gate is False

        # 40.0% passes
        res_pass_pct = evaluate_candidate_oos_gate({**base_passing_metrics, "win_rate": 40.0})
        assert res_pass_pct.passed_gate is True

        res_pass_frac = evaluate_candidate_oos_gate({**base_passing_metrics, "win_rate": 0.400})
        assert res_pass_frac.passed_gate is True

        # 40.1% passes
        res_pass_above = evaluate_candidate_oos_gate({**base_passing_metrics, "win_rate": 40.1})
        assert res_pass_above.passed_gate is True

    def test_drawdown_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify DD boundary: 9.99% passes, 10.00% fails (strictly < 10%), 10.01% fails."""
        # 9.99% passes
        res_pass_pct = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "max_equity_drawdown_pct": 9.99}
        )
        assert res_pass_pct.passed_gate is True

        res_pass_frac = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "max_equity_drawdown_pct": 0.0999}
        )
        assert res_pass_frac.passed_gate is True

        # 10.00% fails (criteria is < 10.0%, code checks max_dd >= 10.0)
        res_fail_exact = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "max_equity_drawdown_pct": 10.00}
        )
        assert res_fail_exact.passed_gate is False
        assert any("max_drawdown_exceeded:10.00%>=10.00%" in r for r in res_fail_exact.reasons)

        res_fail_exact_frac = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "max_equity_drawdown_pct": 0.1000}
        )
        assert res_fail_exact_frac.passed_gate is False

        # 10.01% fails
        res_fail_above = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "max_equity_drawdown_pct": 10.01}
        )
        assert res_fail_above.passed_gate is False

    def test_trade_count_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify total trades boundary: 29 fails, 30 passes, 31 passes."""
        res_29 = evaluate_candidate_oos_gate({**base_passing_metrics, "total_trades": 29})
        assert res_29.passed_gate is False
        assert any("oos_trades_insufficient:29<30" in r for r in res_29.reasons)

        res_30 = evaluate_candidate_oos_gate({**base_passing_metrics, "total_trades": 30})
        assert res_30.passed_gate is True

        res_31 = evaluate_candidate_oos_gate({**base_passing_metrics, "total_trades": 31})
        assert res_31.passed_gate is True

    def test_risk_reward_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify RR boundary: 1.99 fails, 2.00 passes, 2.01 passes."""
        res_199 = evaluate_candidate_oos_gate({**base_passing_metrics, "risk_reward_ratio": 1.99})
        assert res_199.passed_gate is False
        assert any("risk_reward_below_hurdle:1.99<2.00" in r for r in res_199.reasons)

        res_200 = evaluate_candidate_oos_gate({**base_passing_metrics, "risk_reward_ratio": 2.00})
        assert res_200.passed_gate is True

        res_201 = evaluate_candidate_oos_gate({**base_passing_metrics, "risk_reward_ratio": 2.01})
        assert res_201.passed_gate is True

    def test_hold_share_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify hold share boundary: 0.799 passes, 0.800 fails, 0.801 fails."""
        res_799 = evaluate_candidate_oos_gate({**base_passing_metrics, "hold_share": 0.799})
        assert res_799.passed_gate is True

        res_800 = evaluate_candidate_oos_gate({**base_passing_metrics, "hold_share": 0.800})
        assert res_800.passed_gate is False
        assert any("hold_collapse_detected:0.800>=0.800" in r for r in res_800.reasons)

        res_801 = evaluate_candidate_oos_gate({**base_passing_metrics, "hold_share": 0.801})
        assert res_801.passed_gate is False

    def test_directional_imbalance_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify directional imbalance: 0.600 passes, 0.601 fails, -0.600 passes, -0.601 fails."""
        res_pos_pass = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "entry_direction_imbalance": 0.600}
        )
        assert res_pos_pass.passed_gate is True

        res_pos_fail = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "entry_direction_imbalance": 0.601}
        )
        assert res_pos_fail.passed_gate is False
        assert any("directional_imbalance_exceeded:0.601>0.600" in r for r in res_pos_fail.reasons)

        res_neg_pass = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "entry_direction_imbalance": -0.600}
        )
        assert res_neg_pass.passed_gate is True

        res_neg_fail = evaluate_candidate_oos_gate(
            {**base_passing_metrics, "entry_direction_imbalance": -0.601}
        )
        assert res_neg_fail.passed_gate is False
        assert any("directional_imbalance_exceeded:0.601>0.600" in r for r in res_neg_fail.reasons)

    def test_entropy_boundary_edges(
        self, base_passing_metrics: dict[str, Any]
    ) -> None:
        """Verify action entropy: 0.350 fails, 0.3501 passes."""
        res_350 = evaluate_candidate_oos_gate({**base_passing_metrics, "entropy": 0.350})
        assert res_350.passed_gate is False
        assert any("entropy_collapse_detected:0.350<=0.350" in r for r in res_350.reasons)

        res_3501 = evaluate_candidate_oos_gate({**base_passing_metrics, "entropy": 0.3501})
        assert res_3501.passed_gate is True


# ==============================================================================
# PART 4: ADVERSARIAL DISCOVERY & EMPIRICAL PROOF OF NaN VULNERABILITY
# ==============================================================================


class TestEvaluateCandidateOOSGateAdversarialVulnerabilities:
    """Empirical proof of the NaN fail-open vulnerability in evaluate_candidate_oos_gate."""

    def test_empty_and_none_metrics_fail_closed(self) -> None:
        """Verify empty dict or None metrics fail closed due to 0 trades and default 0.0 values."""
        res_none = evaluate_candidate_oos_gate(None)
        assert res_none.passed_gate is False
        assert any("oos_trades_insufficient:0<30" in r for r in res_none.reasons)

        res_empty = evaluate_candidate_oos_gate({})
        assert res_empty.passed_gate is False
        assert any("oos_trades_insufficient:0<30" in r for r in res_empty.reasons)

    def test_nan_metrics_fail_closed_remediation(self) -> None:
        """Verify that evaluate_candidate_oos_gate strictly FAILS CLOSED when metrics are NaN!

        Under IEEE-754 semantics, relational comparisons with NaN normally evaluate to False.
        Remediation enforces strict fail-closed checks: invalid non-finite metrics are rejected
        and passed_gate is asserted False with descriptive reasons.
        """
        all_nan_metrics = {
            "total_trades": 35,
            "expectancy": float("nan"),
            "win_rate": float("nan"),
            "risk_reward_ratio": float("nan"),
            "max_equity_drawdown_pct": float("nan"),
            "hold_share": float("nan"),
            "entry_direction_imbalance": float("nan"),
            "entropy": float("nan"),
        }

        res = evaluate_candidate_oos_gate(all_nan_metrics)

        assert res.passed_gate is False
        assert len(res.reasons) >= 7
        assert any("expectancy" in r for r in res.reasons)
        assert any("win_rate" in r for r in res.reasons)
        assert any("risk_reward_ratio" in r for r in res.reasons)
        assert any("max_equity_drawdown_pct" in r for r in res.reasons)
        assert any("hold_share" in r for r in res.reasons)
        assert any("entry_direction_imbalance" in r for r in res.reasons)
        assert any("entropy" in r for r in res.reasons)

    def test_inf_metrics_fail_closed(self) -> None:
        """Verify that evaluate_candidate_oos_gate strictly FAILS CLOSED when metrics are infinite."""
        inf_metrics = {
            "total_trades": 35,
            "expectancy": float("inf"),
            "win_rate": float("inf"),
            "risk_reward_ratio": float("inf"),
            "max_equity_drawdown_pct": float("-inf"),
            "hold_share": float("inf"),
            "entry_direction_imbalance": float("inf"),
            "entropy": float("inf"),
        }

        res = evaluate_candidate_oos_gate(inf_metrics)

        assert res.passed_gate is False
        assert len(res.reasons) >= 7
        assert any("expectancy" in r for r in res.reasons)
        assert any("win_rate" in r for r in res.reasons)
        assert any("risk_reward_ratio" in r for r in res.reasons)

    def test_step_round_rejects_nan_metrics_candidate(self) -> None:
        """Verify that step_round rejects an all-NaN candidate, prevents champion promotion, and rolls back."""
        with tempfile.TemporaryDirectory() as td:
            cfg = AutoRLEngineConfig(runs_dir=Path(td), max_rounds=2)
            engine = AutoRLEngine(cfg)

            nan_metrics = {
                "total_trades": 40,
                "expectancy": float("nan"),
                "win_rate": float("nan"),
                "risk_reward_ratio": float("nan"),
                "max_equity_drawdown_pct": float("nan"),
                "hold_share": float("nan"),
                "entry_direction_imbalance": float("nan"),
                "entropy": float("nan"),
                "ppo_loss": 1.0,  # normal PPO loss
            }

            res = engine.step_round(metric_overrides=nan_metrics)

            assert res["promoted"] is False
            assert res["gate_passed"] is False
            assert engine.champion_checkpoint is None

    def test_state_machine_records_nan_and_inf_expectancy_as_loss(self) -> None:
        """Verify that TimeframeAdaptationStateMachine treats NaN and Inf expectancy as losses, triggering migration."""
        sm = TimeframeAdaptationStateMachine(initial_timeframe="M1", m1_failure_threshold=3)
        sm.record_round_result(round_num=1, expectancy=-0.05, trades=20)
        assert sm.consecutive_m1_losses == 1

        # Round 2: NaN expectancy must NOT reset consecutive losses!
        sm.record_round_result(round_num=2, expectancy=float("nan"), trades=20)
        assert sm.consecutive_m1_losses == 2

        # Round 3: Inf expectancy must also be treated as a loss and trigger migration to M5
        tf = sm.record_round_result(round_num=3, expectancy=float("inf"), trades=20)
        assert tf == "M5"
        assert len(sm.migration_history) == 1
        assert sm.migration_history[0]["reason"] == "3_consecutive_m1_losses"

    def test_contrast_oos_validator_evaluate_blocks_nan_metrics(self) -> None:
        """Contrast: OOSValidator.evaluate properly blocks NaN expectancy, win rate, RR, and DD.

        OOSValidator uses positive compliance checks:
            expectancy_passed = expectancy_val >= self.min_expectancy (nan >= 0.05 -> False)
            if not expectancy_passed: reasons.append(...)
        which correctly fails closed on NaN values.
        """
        v = OOSValidator()
        nan_metrics = {
            "total_trades": 35,
            "expectancy": float("nan"),
            "win_rate": 0.50,
            "risk_reward_ratio": 2.5,
            "max_equity_drawdown_pct": 5.0,
            "hold_share": 0.20,
            "entry_direction_imbalance": 0.10,
            "normalized_action_entropy": 0.80,
        }

        res = v.evaluate(None, metrics=nan_metrics)
        assert res.passed_gate is False
        assert any("gate:expectancy_below_threshold" in r for r in res.reasons)
