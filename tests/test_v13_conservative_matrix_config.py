from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_v13_conservative_matrix.py"
_SPEC = importlib.util.spec_from_file_location("v13_conservative_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)


def test_matrix_arms_share_mandatory_hidden_regularization() -> None:
    baseline = matrix._config()
    candidate = matrix._candidate_config(baseline)

    assert baseline.learning.hidden_normalization_mode == "layernorm_v1"
    assert baseline.learning.hidden_normalization_enabled is True
    assert baseline.learning.hidden_normalization_layers == (True,) * baseline.model.hidden_layers
    assert baseline.learning.output_normalization_mode == "raw"
    assert baseline.learning.hidden_dropout_enabled is True
    assert all(rate > 0.0 for rate in baseline.learning.hidden_dropout_rates)
    assert candidate.learning.hidden_normalization_mode == "layernorm_v1"
    assert candidate.learning.hidden_normalization_enabled is True
    assert candidate.learning.hidden_normalization_layers == (True,) * candidate.model.hidden_layers
    assert candidate.learning.hidden_dropout_enabled is True
    assert all(rate > 0.0 for rate in candidate.learning.hidden_dropout_rates)
    assert candidate.learning.output_normalization_mode == "rms_v1"
    assert candidate.risk.max_risk_pct == 0.005
    assert candidate.risk.max_open_risk_pct == 0.015
    assert candidate.risk.enforce_entry_limits is True
    assert candidate.learning.context_expectancy_entry_gate_enabled is True
    assert candidate.learning.context_expectancy_entry_gate_min_support == 12.0
    assert candidate.learning.context_expectancy_entry_gate_max_lcb == -0.15
    assert candidate.learning.context_expectancy_entry_gate_scope_fallback_enabled is True
    assert candidate.learning.timing_network_lifecycle_scale == 0.50
    assert baseline.learning.context_expectancy_entry_gate_scope_fallback_enabled is False


def test_runtime_snapshot_isolated_before_shared_runner_is_reused(tmp_path) -> None:
    source = tmp_path / "shared" / "execution.csv"
    source.parent.mkdir()
    source.write_text("candidate\n", encoding="utf-8")

    matrix._copy_runtime({"execution_ledger_path": str(source)}, tmp_path / "candidate")
    source.write_text("baseline\n", encoding="utf-8")
    matrix._copy_runtime({"execution_ledger_path": str(source)}, tmp_path / "baseline")

    assert (tmp_path / "candidate" / "execution.csv").read_text(encoding="utf-8") == "candidate\n"
    assert (tmp_path / "baseline" / "execution.csv").read_text(encoding="utf-8") == "baseline\n"
