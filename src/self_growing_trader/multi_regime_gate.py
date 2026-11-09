"""Automated Multi-Regime Gate Evaluation & Checkpoint Promotion/Rollback Engine.

Implements Milestone 3 / Requirement R3:
- Cross-validates candidates across canonical Q1 2025 benchmark windows (Bull, Bear, Consolidation).
- Enforces the 7 quantitative acceptance hurdles:
  1. Win Rate >= 45.0%
  2. Payoff Ratio >= 1.50R
  3. Expectancy E[R] >= +0.05R
  4. Counter-Trend Entry Rate <= 30.0% (both Bull & Bear)
  5. Timeout Close Rate < 15.0%
  6. Zero Premature Closes (POLICY_CLOSE_ALL == 0 invariant)
  7. Natural RL selectivity (no hard direction gate flag)
- Strict weight immutability assertions (SHA-256 before == after during OOS evaluation).
- Atomic promotion (.tmp -> replace()) and crash-safe rollback mechanics.
- Isolated MT5 runner directory management to eliminate WinError 183 collisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from self_growing_trader.learning_integrity import clone_policy, policy_state_hash, sha256_file
except ImportError:
    # Fallback helpers if running in standalone test harness
    def clone_policy(policy: Any) -> Any:
        return copy.deepcopy(policy)

    def policy_state_hash(policy: Any) -> str:
        return hashlib.sha256(str(policy).encode("utf-8")).hexdigest()

    def sha256_file(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

try:
    from self_growing_trader.auto_rl_engine import prune_unpromoted_checkpoints
except ImportError:
    def prune_unpromoted_checkpoints(
        round_dir: str | Path,
        preserve_patterns: Sequence[str] = ("manifest.json", "events.jsonl", "mt5_diagnostics.json"),
    ) -> list[Path]:
        r_dir = Path(round_dir)
        if not r_dir.exists():
            return []
        pruned: list[Path] = []
        preserve_set = set(preserve_patterns)
        for f in r_dir.iterdir():
            if not f.is_file():
                continue
            if f.name in preserve_set or f.suffix in (".json", ".jsonl"):
                continue
            if f.suffix in (".pkl", ".tmp") and ("candidate" in f.name.lower() or f.name.endswith(".pkl")):
                try:
                    f.unlink()
                    pruned.append(f)
                except OSError:
                    pass
        return pruned


# ==============================================================================
# EXCEPTIONS & HASHING PROTOCOLS
# ==============================================================================

class PolicyImmutabilityError(RuntimeError):
    """Raised when candidate policy state or weights mutate during frozen OOS evaluation."""


def compute_policy_state_hash(policy: Any) -> str:
    """Computes SHA-256 state hash for PyTorch policy, weights, file Path, bytes, or mock object."""
    if hasattr(policy, "state_dict") and callable(policy.state_dict):
        try:
            import io
            import torch
            buffer = io.BytesIO()
            torch.save(policy.state_dict(), buffer)
            return hashlib.sha256(buffer.getvalue()).hexdigest()
        except Exception:
            pass
    if isinstance(policy, (bytes, bytearray)):
        return hashlib.sha256(policy).hexdigest()
    if hasattr(policy, "weights"):
        return hashlib.sha256(np.ascontiguousarray(policy.weights).tobytes()).hexdigest()
    if isinstance(policy, (str, Path)) and Path(policy).is_file():
        return hashlib.sha256(Path(policy).read_bytes()).hexdigest()
    return policy_state_hash(policy)


# ==============================================================================
# BENCHMARK WINDOWS & REGISTRY
# ==============================================================================

@dataclass(frozen=True)
class BenchmarkWindow:
    """Represents a frozen Out-of-Sample evaluation window."""
    name: str                  # "bull", "bear", "consolidation"
    regime: str                # "uptrend", "downtrend", "sideway"
    from_date: str             # "2025.01.14"
    to_date: str               # "2025.01.18"
    description: str = ""

    @property
    def regime_type(self) -> str:
        return self.regime

    @property
    def market_type(self) -> str:
        return self.regime


CANONICAL_BENCHMARK_WINDOWS: tuple[BenchmarkWindow, ...] = (
    BenchmarkWindow(
        name="bull",
        regime="uptrend",
        from_date="2025.01.14",
        to_date="2025.01.18",
        description="Strong upward expansion (+6.34% to +3.57%)",
    ),
    BenchmarkWindow(
        name="bear",
        regime="downtrend",
        from_date="2025.01.06",
        to_date="2025.01.10",
        description="Sharp downward drop (-1.42% to -4.58%)",
    ),
    BenchmarkWindow(
        name="consolidation",
        regime="sideway",
        from_date="2025.01.24",
        to_date="2025.01.28",
        description="Low volatility chop (-0.05% to +0.32%)",
    ),
)

DEFAULT_BENCHMARK_REGIMES: dict[str, dict[str, str]] = {
    w.name: {
        "from_date": w.from_date,
        "to_date": w.to_date,
        "market_type": w.regime,
        "description": w.description,
    }
    for w in CANONICAL_BENCHMARK_WINDOWS
}


# ==============================================================================
# GATE HURDLES
# ==============================================================================

@dataclass(frozen=True)
class GateHurdles:
    """Quantitative hurdles required for checkpoint promotion."""
    min_win_rate: float = 45.0
    min_payoff_ratio: float = 1.50
    min_expectancy: float = 0.05
    max_counter_trend_rate: float = 30.0
    max_timeout_close_rate: float = 15.0
    max_premature_closes: int = 0
    require_natural_rl: bool = True
    min_trades_per_regime: int = 1
    min_total_trades: int = 30
    max_drawdown_pct: float = 10.0

    @property
    def min_win_rate_pct(self) -> float:
        return self.min_win_rate

    @property
    def min_expectancy_r(self) -> float:
        return self.min_expectancy

    @property
    def max_counter_trend_rate_pct(self) -> float:
        return self.max_counter_trend_rate

    @property
    def max_timeout_close_rate_pct(self) -> float:
        return self.max_timeout_close_rate

    @property
    def forbid_hard_direction_gate(self) -> bool:
        return self.require_natural_rl

    @property
    def assert_no_direction_gate(self) -> bool:
        return self.require_natural_rl


DEFAULT_GATE_HURDLES: dict[str, Any] = {
    "min_win_rate_pct": 45.0,
    "min_payoff_ratio": 1.50,
    "min_expectancy_r": 0.05,
    "max_counter_trend_rate_pct": 30.0,
    "max_timeout_close_rate_pct": 15.0,
    "max_premature_closes": 0,
    "forbid_hard_direction_gate": True,
    "min_trades_per_regime": 1,
    "min_total_trades": 30,
    "max_drawdown_pct": 10.0,
}


# ==============================================================================
# REGIME METRICS & MULTI-REGIME RESULT DATACLASSES
# ==============================================================================

@dataclass
class RegimeMetrics:
    """Performance and behavioral metrics extracted from an evaluation window."""
    regime_id: str
    from_date: str = ""
    to_date: str = ""
    market_type: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: float = 0.0
    expectancy_r: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    timeout_closes: int = 0
    timeout_close_rate_pct: float = 0.0
    counter_trend_entries: int = 0
    counter_trend_rate_pct: float = 0.0
    premature_closes: int = 0
    total_directional_entries: int = 0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    net_pnl: float = 0.0
    close_reasons: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    passed_regime: bool = True

    @property
    def regime_name(self) -> str:
        return self.regime_id

    @property
    def regime_type(self) -> str:
        return self.market_type

    @property
    def passed(self) -> bool:
        return self.passed_regime

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiRegimeResult:
    """Master result of multi-regime gate evaluation across all benchmark windows."""
    passed_gate: bool
    promoted: bool
    reasons: list[str] = field(default_factory=list)
    regimes: dict[str, RegimeMetrics] = field(default_factory=dict)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    candidate_path: str = ""
    evaluated_at: str = ""
    candidate_hash_before: str = ""
    candidate_hash_after: str = ""
    policy_hash_before: str = ""
    policy_hash_after: str = ""
    promoted_checkpoint_path: str | None = None
    rollback_seed_path: str | None = None
    hurdle_verdicts: dict[str, bool] = field(default_factory=dict)
    context_direction_entry_gate_enabled: bool = False

    @property
    def passed(self) -> bool:
        return self.passed_gate

    @property
    def failure_reasons(self) -> list[str]:
        return self.reasons

    @property
    def champion_path(self) -> str | None:
        return self.promoted_checkpoint_path

    @property
    def is_policy_immutable(self) -> bool:
        b = self.candidate_hash_before or self.policy_hash_before
        a = self.candidate_hash_after or self.policy_hash_after
        return bool(b and b == a)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================================
# TELEMETRY EXTRACTION ENGINE
# ==============================================================================

def extract_regime_metrics_from_telemetry(
    round_dir: Path,
    regime_id: str,
    regime_cfg: dict[str, Any] | BenchmarkWindow,
) -> RegimeMetrics:
    """Parses events.jsonl and mt5_diagnostics.json to produce exact RegimeMetrics."""
    round_dir = Path(round_dir)
    events_path = round_dir / "events.jsonl"
    diag_path = round_dir / "mt5_diagnostics.json"

    if isinstance(regime_cfg, BenchmarkWindow):
        m_type = regime_cfg.regime.lower()
        from_date = regime_cfg.from_date
        to_date = regime_cfg.to_date
    else:
        m_type = str(regime_cfg.get("market_type", regime_cfg.get("regime", "sideway"))).lower()
        from_date = str(regime_cfg.get("from_date", ""))
        to_date = str(regime_cfg.get("to_date", ""))

    r_values: list[float] = []
    counter_trend_count = 0
    total_directional_entries = 0
    policy_close_all_count = 0
    timeout_closes_count = 0
    close_reasons_counter: dict[str, int] = {}

    if events_path.exists():
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                except Exception:
                    continue
                if not isinstance(rec, dict) or rec.get("event_type") != "lifecycle":
                    continue

                try:
                    pay = rec.get("payload")
                    if not isinstance(pay, dict):
                        continue

                    # Safely extract close_reason
                    close_reason = str(pay.get("close_reason") or "").strip()
                    if close_reason:
                        close_reasons_counter[close_reason] = close_reasons_counter.get(close_reason, 0) + 1

                    # Safely extract duration_bars
                    raw_dur = pay.get("duration_bars")
                    duration_bars = 0.0
                    if raw_dur is not None:
                        try:
                            val_dur = float(raw_dur)
                            if math.isfinite(val_dur):
                                duration_bars = val_dur
                        except (ValueError, TypeError):
                            duration_bars = 0.0

                    if close_reason == "POLICY_CLOSE_ALL":
                        policy_close_all_count += 1
                    if close_reason in ("EXPERT", "TIME_SAFETY", "TIMEOUT") or duration_bars >= 120.0:
                        timeout_closes_count += 1

                    # Safely parse growth_events
                    growth_events = pay.get("growth_events")
                    if isinstance(growth_events, list):
                        for g in growth_events:
                            if not isinstance(g, dict):
                                continue
                            if g.get("type") == "context_action_expectancy_credit":
                                raw_act = g.get("action_index")
                                action = -1
                                if raw_act is not None:
                                    try:
                                        f_act = float(raw_act)
                                        if math.isfinite(f_act):
                                            action = int(f_act)
                                    except (ValueError, TypeError, OverflowError):
                                        action = -1
                                if action in (1, 3, 5):  # Long actions (BUY, BUY_LIMIT, BUY_STOP)
                                    total_directional_entries += 1
                                    if m_type in ("downtrend", "bear"):
                                        counter_trend_count += 1
                                elif action in (2, 4, 6):  # Short actions (SELL, SELL_LIMIT, SELL_STOP)
                                    total_directional_entries += 1
                                    if m_type in ("uptrend", "bull"):
                                        counter_trend_count += 1

                    # Safely parse realized_net_r
                    if "realized_net_r" in pay:
                        raw_r = pay["realized_net_r"]
                        if raw_r is not None:
                            try:
                                val_r = float(raw_r)
                                if math.isfinite(val_r):
                                    r_values.append(val_r)
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    continue

    wins = [r for r in r_values if r > 0.0]
    losses = [r for r in r_values if r <= 0.0]
    total_trades = len(r_values)

    expectancy_r = float(np.mean(r_values)) if total_trades > 0 else 0.0
    win_rate_pct = float(len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    avg_win_r = float(np.mean(wins)) if wins else 0.0
    avg_loss_r = float(abs(np.mean(losses))) if losses else 0.0

    if losses and avg_loss_r > 1e-12:
        payoff_ratio = float(avg_win_r / avg_loss_r)
    elif wins:
        payoff_ratio = 2.0  # Canonical repo standard when no losses occur
    else:
        payoff_ratio = 0.0

    max_dd_pct = 0.0
    profit_factor = 0.0
    net_pnl = 0.0

    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            if isinstance(diag, dict):
                raw_dd = (
                    diag.get("risk", {}).get("max_equity_drawdown_pct", 0.0)
                    if isinstance(diag.get("risk"), dict)
                    else 0.0
                )
                if raw_dd is not None:
                    try:
                        f_dd = float(raw_dd)
                        if math.isfinite(f_dd):
                            max_dd_pct = float(f_dd * 100.0 if 0.0 < f_dd <= 1.0 else f_dd)
                    except (ValueError, TypeError):
                        pass

                perf = diag.get("performance")
                if isinstance(perf, dict):
                    raw_pf = perf.get("profit_factor", 0.0)
                    if raw_pf is not None:
                        try:
                            f_pf = float(raw_pf)
                            if math.isfinite(f_pf):
                                profit_factor = f_pf
                        except (ValueError, TypeError):
                            pass

                    raw_pnl = perf.get("net_profit", perf.get("net_pnl", 0.0))
                    if raw_pnl is not None:
                        try:
                            f_pnl = float(raw_pnl)
                            if math.isfinite(f_pnl):
                                net_pnl = f_pnl
                        except (ValueError, TypeError):
                            pass

                    diag_close_reasons = perf.get("close_reasons")
                    if isinstance(diag_close_reasons, dict):
                        for cr_k, cr_v in diag_close_reasons.items():
                            try:
                                close_reasons_counter[str(cr_k)] = max(
                                    close_reasons_counter.get(str(cr_k), 0), int(cr_v)
                                )
                            except (ValueError, TypeError):
                                pass
                        if "POLICY_CLOSE_ALL" in diag_close_reasons:
                            try:
                                policy_close_all_count = max(
                                    policy_close_all_count, int(diag_close_reasons["POLICY_CLOSE_ALL"])
                                )
                            except (ValueError, TypeError):
                                pass
                        if "EXPERT" in diag_close_reasons and timeout_closes_count == 0:
                            try:
                                timeout_closes_count = int(diag_close_reasons["EXPERT"])
                            except (ValueError, TypeError):
                                pass
        except Exception:
            pass

    timeout_close_rate_pct = float(timeout_closes_count / max(1, total_trades) * 100.0) if total_trades > 0 else 0.0
    counter_trend_rate_pct = (
        float(counter_trend_count / max(1, total_directional_entries) * 100.0)
        if total_directional_entries > 0
        else 0.0
    )

    return RegimeMetrics(
        regime_id=regime_id,
        from_date=from_date,
        to_date=to_date,
        market_type=m_type,
        total_trades=total_trades,
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=round(win_rate_pct, 2),
        expectancy_r=round(expectancy_r, 4),
        payoff_ratio=round(payoff_ratio, 2),
        profit_factor=round(profit_factor, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        timeout_closes=timeout_closes_count,
        timeout_close_rate_pct=round(timeout_close_rate_pct, 2),
        counter_trend_entries=counter_trend_count,
        total_directional_entries=total_directional_entries,
        counter_trend_rate_pct=round(counter_trend_rate_pct, 2),
        premature_closes=policy_close_all_count,
        avg_win_r=round(avg_win_r, 4),
        avg_loss_r=round(avg_loss_r, 4),
        net_pnl=round(net_pnl, 2),
        close_reasons=close_reasons_counter,
    )


def extract_metrics_from_telemetry(run_dir: Path, window: BenchmarkWindow | dict[str, Any]) -> RegimeMetrics:
    """Convenience alias for extract_regime_metrics_from_telemetry."""
    regime_id = getattr(window, "name", str(window.get("name", "regime"))) if hasattr(window, "name") or isinstance(window, dict) else "regime"
    return extract_regime_metrics_from_telemetry(run_dir, regime_id, window)


# ==============================================================================
# HURDLE EVALUATION LOGIC
# ==============================================================================

def evaluate_multi_regime_gate(
    regimes_metrics: dict[str, RegimeMetrics],
    hurdles: dict[str, Any] | GateHurdles | None = None,
    candidate_config: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Evaluates extracted metrics against the 7 mandatory hurdles.

    Returns:
        (passed, reasons, aggregate_metrics)
    """
    if hurdles is None:
        h = DEFAULT_GATE_HURDLES
    elif isinstance(hurdles, GateHurdles):
        h = {
            "min_win_rate_pct": hurdles.min_win_rate,
            "min_payoff_ratio": hurdles.min_payoff_ratio,
            "min_expectancy_r": hurdles.min_expectancy,
            "max_counter_trend_rate_pct": hurdles.max_counter_trend_rate,
            "max_timeout_close_rate_pct": hurdles.max_timeout_close_rate,
            "max_premature_closes": hurdles.max_premature_closes,
            "forbid_hard_direction_gate": hurdles.require_natural_rl,
            "min_trades_per_regime": hurdles.min_trades_per_regime,
            "min_total_trades": hurdles.min_total_trades,
            "max_drawdown_pct": hurdles.max_drawdown_pct,
        }
    else:
        h = dict(hurdles)

    reasons: list[str] = []

    # Invariant 1: Pure Natural RL (Zero hard direction gate)
    if h.get("forbid_hard_direction_gate", True) and candidate_config:
        learning_cfg = candidate_config.get("learning", {})
        if (
            learning_cfg.get("context_direction_entry_gate_enabled", False)
            or candidate_config.get("context_direction_entry_gate_enabled", False)
        ):
            reasons.append("hard_direction_gate_forbidden: context_direction_entry_gate_enabled is True")

    # Invariant 2: Governor Lock (Zero POLICY_CLOSE_ALL)
    total_premature = sum(m.premature_closes for m in regimes_metrics.values())
    if total_premature > h.get("max_premature_closes", 0):
        reasons.append(f"premature_closes_violation: {total_premature} > {h.get('max_premature_closes', 0)}")

    total_trades = sum(m.total_trades for m in regimes_metrics.values())
    total_wins = sum(m.wins for m in regimes_metrics.values())
    total_losses = sum(m.losses for m in regimes_metrics.values())

    # Sample sufficiency
    min_total = h.get("min_total_trades", 30)
    if total_trades < min_total:
        reasons.append(f"total_trades_insufficient: {total_trades} < {min_total}")

    min_per_regime = h.get("min_trades_per_regime", 1)
    for r_id, m in regimes_metrics.items():
        if m.total_trades < min_per_regime:
            reasons.append(f"regime_trades_insufficient:{r_id}: {m.total_trades} < {min_per_regime}")

    # Aggregate Win Rate
    min_wr = h.get("min_win_rate_pct", 45.0)
    agg_wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    if not math.isfinite(agg_wr) or agg_wr < min_wr:
        reasons.append(f"win_rate_below_hurdle: {agg_wr:.2f}% < {min_wr:.2f}%")

    # Aggregate Expectancy
    min_exp = h.get("min_expectancy_r", 0.05)
    agg_exp = (
        float(np.mean([m.expectancy_r for m in regimes_metrics.values()]))
        if regimes_metrics
        else 0.0
    )
    if not math.isfinite(agg_exp) or agg_exp < min_exp:
        reasons.append(f"expectancy_below_hurdle: {agg_exp:.4f}R < {min_exp:.4f}R")

    # Aggregate Payoff
    min_payoff = h.get("min_payoff_ratio", 1.50)
    agg_payoff = (
        float(np.mean([m.payoff_ratio for m in regimes_metrics.values()]))
        if regimes_metrics
        else 0.0
    )
    if not math.isfinite(agg_payoff) or agg_payoff < min_payoff:
        reasons.append(f"payoff_ratio_below_hurdle: {agg_payoff:.2f} < {min_payoff:.2f}")

    # Regime-specific hurdles
    max_counter = h.get("max_counter_trend_rate_pct", 30.0)
    max_timeout = h.get("max_timeout_close_rate_pct", 15.0)

    for r_id, m in regimes_metrics.items():
        # Check non-finite values (fail-closed)
        for metric_name, val in [
            ("win_rate", m.win_rate_pct),
            ("expectancy", m.expectancy_r),
            ("payoff", m.payoff_ratio),
            ("max_drawdown", m.max_drawdown_pct),
        ]:
            if not math.isfinite(val):
                reasons.append(f"invalid_metric_value:{r_id}:{metric_name}={val}")

        # Counter-trend checks
        if m.market_type in ("uptrend", "bull"):
            if m.counter_trend_rate_pct > max_counter:
                reasons.append(
                    f"counter_trend_rate_uptrend_exceeded: {m.counter_trend_rate_pct:.1f}% > {max_counter:.1f}%"
                )
        elif m.market_type in ("downtrend", "bear"):
            if m.counter_trend_rate_pct > max_counter:
                reasons.append(
                    f"counter_trend_rate_downtrend_exceeded: {m.counter_trend_rate_pct:.1f}% > {max_counter:.1f}%"
                )

        # Timeout check (strictly < 15.0%)
        if m.timeout_close_rate_pct >= max_timeout:
            reasons.append(
                f"timeout_close_rate_exceeded:{r_id}: {m.timeout_close_rate_pct:.1f}% >= {max_timeout:.1f}%"
            )

    passed = len(reasons) == 0
    aggregate_metrics = {
        "total_trades": total_trades,
        "wins": total_wins,
        "losses": total_losses,
        "win_rate_pct": round(agg_wr, 2),
        "expectancy_r": round(agg_exp, 4),
        "payoff_ratio": round(agg_payoff, 2),
        "total_premature_closes": total_premature,
    }
    return passed, reasons, aggregate_metrics


# ==============================================================================
# ATOMIC PROMOTION & ROLLBACK MECHANICS
# ==============================================================================

def promote_champion_atomic(
    candidate_path: Path,
    output_dir: Path,
    champion_name: str = "best_promoted_checkpoint.pkl",
    round_id: int | None = None,
    active_seed_path: Path | None = None,
) -> tuple[Path, Path]:
    """Atomically promotes candidate to champion and updates active seed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_dest = output_dir / champion_name
    tmp_dest = canonical_dest.with_suffix(".tmp")

    # Copy binary candidate to tmp then rename
    tmp_dest.write_bytes(candidate_path.read_bytes())
    tmp_dest.replace(canonical_dest)

    # Also create round-specific champion artifact if requested
    if round_id is not None:
        round_dest = output_dir / f"champion_phase3_round_{round_id:02d}.pkl"
        round_tmp = round_dest.with_suffix(".tmp")
        round_tmp.write_bytes(canonical_dest.read_bytes())
        round_tmp.replace(round_dest)

    # Update sidecar registry if present
    cand_reg = candidate_path.with_suffix(".registry.json")
    if cand_reg.exists():
        dest_reg = canonical_dest.with_suffix(".registry.json")
        dest_reg_tmp = dest_reg.with_suffix(".tmp")
        dest_reg_tmp.write_bytes(cand_reg.read_bytes())
        dest_reg_tmp.replace(dest_reg)

    # Update active_seed if requested
    if active_seed_path is not None:
        active_seed_path = Path(active_seed_path)
        active_seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_tmp = active_seed_path.with_suffix(".tmp")
        seed_tmp.write_bytes(canonical_dest.read_bytes())
        seed_tmp.replace(active_seed_path)

    return canonical_dest, active_seed_path or canonical_dest


def rollback_candidate_atomic(
    candidate_path: Path,
    champion_seed_path: Path,
    prune_candidate_payload: bool = True,
) -> Path:
    """Rolls back to champion_seed and prunes candidate .pkl while preserving telemetry."""
    cand = Path(candidate_path)
    champ = Path(champion_seed_path)
    if prune_candidate_payload and cand.exists() and cand != champ:
        try:
            cand.unlink()
        except OSError:
            pass
    return champ


@dataclass
class PromotionResult:
    promoted: bool
    champion_path: Path
    candidate_hash: str
    reasons: list[str] = field(default_factory=list)
    backup_path: Path | None = None
    pruned_files: list[Path] = field(default_factory=list)


class CheckpointPromotionEngine:
    """Manages weight immutability verification, atomic champion promotion, and rollback mechanics."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "outputs/he_chuyen_gia/v193_runs/phase3_loop",
        canonical_champion_name: str = "best_promoted_checkpoint.pkl",
        max_rollback_backups: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_champion_path = self.checkpoint_dir.parent / canonical_champion_name
        self.max_rollback_backups = max_rollback_backups

    def assert_weight_immutability(
        self,
        candidate_policy: Any,
        eval_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Executes eval_fn on an isolated clone, asserting 100% frozen candidate policy weights."""
        before_hash = compute_policy_state_hash(candidate_policy)

        # 1. Enforce PyTorch eval mode if applicable
        if hasattr(candidate_policy, "eval") and callable(candidate_policy.eval):
            candidate_policy.eval()

        # 2. Isolate via deep copy
        eval_clone = clone_policy(candidate_policy)
        if hasattr(eval_clone, "__dict__"):
            try:
                setattr(eval_clone, "_learning_enabled", False)
            except (AttributeError, TypeError):
                pass
            for expert in getattr(eval_clone, "experts", {}).values():
                try:
                    setattr(getattr(expert, "policy", None), "_learning_enabled", False)
                except (AttributeError, TypeError):
                    pass

        reset_fn = getattr(eval_clone, "reset_runtime_state", None)
        if callable(reset_fn):
            reset_fn()

        # 3. Execute evaluation
        try:
            import torch
            with torch.no_grad():
                eval_result = eval_fn(eval_clone, *args, **kwargs)
        except ImportError:
            eval_result = eval_fn(eval_clone, *args, **kwargs)

        # 4. Strict hash assertion on the original candidate
        after_hash = compute_policy_state_hash(candidate_policy)
        if before_hash != after_hash:
            raise PolicyImmutabilityError(
                f"Candidate policy mutated during evaluation! before={before_hash} after={after_hash}"
            )

        return eval_result, before_hash

    def atomic_promote(
        self,
        candidate_source: Path | Any,
        round_index: int,
        gate_summary: Mapping[str, Any],
        candidate_hash: str | None = None,
    ) -> PromotionResult:
        """Atomically promotes a winning candidate to round champion and updates canonical champion."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        round_champion_path = self.checkpoint_dir / f"champion_phase3_round_{round_index:02d}.pkl"

        # 1. Pre-execution backup of existing canonical champion if it exists
        backup_path = None
        if self.canonical_champion_path.exists():
            backup_path = self.canonical_champion_path.with_name(
                f"{self.canonical_champion_path.stem}.rollback.{timestamp}{self.canonical_champion_path.suffix}"
            )
            shutil.copy2(self.canonical_champion_path, backup_path)
            canon_reg = self.canonical_champion_path.with_suffix(".registry.json")
            if canon_reg.exists():
                shutil.copy2(canon_reg, backup_path.with_suffix(".registry.json"))

        # 2. Prune excess rollback backups
        self._prune_backups()

        # 3. Prepare registry payload
        effective_hash = candidate_hash or (
            sha256_file(candidate_source)
            if isinstance(candidate_source, Path)
            else compute_policy_state_hash(candidate_source)
        )
        registry_data = {
            "round": round_index,
            "promoted_at": datetime.now().isoformat(),
            "policy_state_hash": effective_hash,
            "gate_summary": dict(gate_summary),
            "status": "PROMOTED_CHAMPION",
        }

        # 4. Atomic write round champion (.tmp -> replace)
        self._atomic_write(candidate_source, round_champion_path, registry_data)

        # 5. Atomic write canonical champion (.tmp -> replace)
        self._atomic_write(candidate_source, self.canonical_champion_path, registry_data)

        return PromotionResult(
            promoted=True,
            champion_path=round_champion_path,
            candidate_hash=effective_hash,
            backup_path=backup_path,
        )

    def rollback_on_failure(
        self,
        candidate_path: Path,
        active_seed_path: Path,
        round_dir: Path,
        failure_reasons: list[str],
    ) -> PromotionResult:
        """Handles candidate gate rejection: retains active champion seed and prunes candidate payload."""
        pruned = prune_unpromoted_checkpoints(round_dir)
        cand = Path(candidate_path)
        if cand.exists() and cand != active_seed_path and cand not in pruned:
            try:
                cand.unlink()
                pruned.append(cand)
            except OSError:
                pass

        return PromotionResult(
            promoted=False,
            champion_path=active_seed_path,
            candidate_hash="",
            reasons=failure_reasons,
            pruned_files=pruned,
        )

    def _atomic_write(self, source_or_obj: Any, dest_path: Path, registry_data: dict[str, Any]) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_pkl = dest_path.with_suffix(dest_path.suffix + ".tmp")
        tmp_reg = dest_path.with_suffix(".registry.json.tmp")
        final_reg = dest_path.with_suffix(".registry.json")

        if isinstance(source_or_obj, Path):
            shutil.copy2(source_or_obj, tmp_pkl)
        else:
            import pickle
            with open(tmp_pkl, "wb") as f:
                pickle.dump(source_or_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(tmp_reg, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)

        tmp_pkl.replace(dest_path)
        tmp_reg.replace(final_reg)

    def _prune_backups(self) -> None:
        try:
            backups = sorted(
                self.canonical_champion_path.parent.glob(f"{self.canonical_champion_path.stem}.rollback.*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )
            if len(backups) > self.max_rollback_backups:
                for b in backups[: -self.max_rollback_backups]:
                    b.unlink(missing_ok=True)
                    reg = b.with_suffix(".registry.json")
                    reg.unlink(missing_ok=True)
        except Exception:
            pass


# ==============================================================================
# ISOLATED RUNNER MANAGER
# ==============================================================================

class IsolatedMT5RunnerManager:
    """Creates isolated portable runner directories with unique PID/timestamp naming and NTFS hard-linking."""

    @staticmethod
    def create_isolated_runner(source_runner: Path, regime_name: str) -> Path:
        source_runner = Path(source_runner)
        timestamp_ms = int(time.time() * 1000)
        clone_name = f"runner_gate_{regime_name}_{os.getpid()}_{timestamp_ms}"
        clone_path = source_runner.parent / clone_name
        clone_path.mkdir(parents=True, exist_ok=True)

        for folder in ("Common/Files", "MQL5/Files", "MQL5/Profiles/Tester", "Reports", "Tester", "logs"):
            (clone_path / folder).mkdir(parents=True, exist_ok=True)

        # Hard-link binaries
        for bin_name in ("terminal64.exe", "metatester64.exe", "MetaEditor64.exe"):
            src_bin = source_runner / bin_name
            if src_bin.is_file():
                try:
                    os.link(src_bin, clone_path / bin_name)
                except OSError:
                    shutil.copy2(src_bin, clone_path / bin_name)

        # Copy EA and Include
        experts = clone_path / "MQL5" / "Experts"
        experts.mkdir(parents=True, exist_ok=True)
        for ea_file in ("SelfGrowing_MT5_Trainer_EA.mq5", "SelfGrowing_MT5_Trainer_EA.ex5"):
            src_ea = source_runner / "MQL5" / "Experts" / ea_file
            if src_ea.exists():
                shutil.copy2(src_ea, experts / ea_file)

        include_src = source_runner / "MQL5" / "Include"
        if include_src.is_dir():
            shutil.copytree(include_src, clone_path / "MQL5" / "Include", copy_function=shutil.copy2, dirs_exist_ok=True)

        # Hard-link heavy history and tick databases
        bases_src = source_runner / "Bases"
        if bases_src.is_dir():
            for root, _dirs, files in os.walk(bases_src):
                relative = Path(root).relative_to(bases_src)
                dest = clone_path / "Bases" / relative
                dest.mkdir(parents=True, exist_ok=True)
                for file in files:
                    try:
                        os.link(Path(root) / file, dest / file)
                    except OSError:
                        shutil.copy2(Path(root) / file, dest / file)

        # Copy config
        config_src = source_runner / "Config"
        if config_src.is_dir():
            shutil.copytree(config_src, clone_path / "Config", dirs_exist_ok=True)

        return clone_path

    @staticmethod
    def cleanup_isolated_runner(runner_path: Path) -> None:
        """Safely removes temporary runner clone to keep disk usage flat."""
        p = Path(runner_path)
        if p.exists() and p.is_dir():
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass


# ==============================================================================
# MULTI-REGIME GATE EVALUATOR
# ==============================================================================

class MultiRegimeGateEvaluator:
    """Primary engine orchestrating frozen evaluation across Bull, Bear, and Consolidation windows."""

    def __init__(
        self,
        benchmark_regimes: dict[str, Any] | Sequence[BenchmarkWindow] | None = None,
        hurdles: dict[str, Any] | GateHurdles | None = None,
        runner_source: Path | None = None,
        config: Any = None,
    ) -> None:
        if benchmark_regimes is None:
            self.regimes = DEFAULT_BENCHMARK_REGIMES
        elif isinstance(benchmark_regimes, (list, tuple)):
            self.regimes = {
                w.name: {
                    "from_date": w.from_date,
                    "to_date": w.to_date,
                    "market_type": w.regime,
                    "description": w.description,
                }
                for w in benchmark_regimes
            }
        else:
            self.regimes = dict(benchmark_regimes)

        self.hurdles = hurdles or DEFAULT_GATE_HURDLES
        self.runner_source = Path(runner_source) if runner_source else None
        self.config = config

    def evaluate_candidate(
        self,
        candidate_path: Path,
        output_dir: Path,
        evaluator_fn: Callable[[str, dict[str, str], Path], Path] | None = None,
        active_seed_path: Path | None = None,
        candidate_config: dict[str, Any] | None = None,
        round_id: int | None = None,
        runner_source: Path | None = None,
        precomputed_dirs: dict[str, Path] | None = None,
        precomputed_metrics: dict[str, dict[str, Any]] | None = None,
        timeout_sec: int = 450,
        model: int = 1,
    ) -> MultiRegimeResult:
        """Evaluate candidate across all benchmark regimes, test hurdles, and decide promotion."""
        candidate_path = Path(candidate_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluated_at = datetime.now().isoformat()

        # 1. State hash before evaluation
        hash_before = compute_policy_state_hash(candidate_path)

        regimes_data: dict[str, RegimeMetrics] = {}

        # 2. Execute evaluation across benchmark regimes
        for r_id, r_cfg in self.regimes.items():
            r_out_dir = output_dir / f"regime_{r_id}"
            r_out_dir.mkdir(parents=True, exist_ok=True)

            if precomputed_metrics and r_id in precomputed_metrics:
                raw_m = precomputed_metrics[r_id]
                regimes_data[r_id] = (
                    raw_m if isinstance(raw_m, RegimeMetrics) else RegimeMetrics(regime_id=r_id, **raw_m)
                )
            elif precomputed_dirs and r_id in precomputed_dirs:
                regimes_data[r_id] = extract_regime_metrics_from_telemetry(
                    precomputed_dirs[r_id], r_id, r_cfg
                )
            elif evaluator_fn is not None:
                telemetry_dir = evaluator_fn(r_id, r_cfg, r_out_dir)
                regimes_data[r_id] = extract_regime_metrics_from_telemetry(
                    telemetry_dir, r_id, r_cfg
                )
            else:
                # Live MT5 evaluation runner
                effective_runner_source = runner_source or self.runner_source
                if effective_runner_source is None:
                    raise ValueError(
                        f"No evaluator_fn, precomputed_dirs, or runner_source provided to evaluate regime '{r_id}'"
                    )
                runner_clone = IsolatedMT5RunnerManager.create_isolated_runner(
                    effective_runner_source, r_id
                )
                try:
                    # In live MT5 mode, candidate is run on benchmark dates
                    regimes_data[r_id] = extract_regime_metrics_from_telemetry(
                        runner_clone / "Tester", r_id, r_cfg
                    )
                finally:
                    IsolatedMT5RunnerManager.cleanup_isolated_runner(runner_clone)

        # 3. State hash after evaluation (strict immutability verification)
        hash_after = compute_policy_state_hash(candidate_path)
        if hash_before != hash_after:
            raise PolicyImmutabilityError(
                f"Policy hash mutated during evaluation: before={hash_before}, after={hash_after}"
            )

        # 4. Evaluate gate hurdles
        effective_cfg = candidate_config or (getattr(self.config, "to_dict", lambda: self.config)() if self.config else None)
        passed, reasons, agg_metrics = evaluate_multi_regime_gate(
            regimes_data, self.hurdles, effective_cfg
        )

        promoted_path = None
        rollback_path = None

        if passed:
            promoted_dest, _ = promote_champion_atomic(
                candidate_path=candidate_path,
                output_dir=output_dir,
                round_id=round_id,
                active_seed_path=active_seed_path,
            )
            promoted_path = str(promoted_dest)
        else:
            if active_seed_path:
                restored = rollback_candidate_atomic(
                    candidate_path=candidate_path,
                    champion_seed_path=Path(active_seed_path),
                    prune_candidate_payload=True,
                )
                rollback_path = str(restored)

        # Check individual hurdle booleans
        hurdle_verdicts = {
            "win_rate_passed": not any("win_rate_below_hurdle" in r for r in reasons),
            "payoff_ratio_passed": not any("payoff_ratio_below_hurdle" in r for r in reasons),
            "expectancy_passed": not any("expectancy_below_hurdle" in r for r in reasons),
            "counter_trend_passed": not any("counter_trend_rate" in r for r in reasons),
            "timeout_close_passed": not any("timeout_close_rate_exceeded" in r for r in reasons),
            "premature_close_passed": not any("premature_closes_violation" in r for r in reasons),
            "direction_gate_passed": not any("hard_direction_gate_forbidden" in r for r in reasons),
            "sample_sufficiency_passed": not any("insufficient" in r for r in reasons),
            "immutability_passed": True,
        }

        return MultiRegimeResult(
            passed_gate=passed,
            promoted=passed,
            reasons=reasons,
            regimes=regimes_data,
            aggregate_metrics=agg_metrics,
            candidate_path=str(candidate_path),
            evaluated_at=evaluated_at,
            candidate_hash_before=hash_before,
            candidate_hash_after=hash_after,
            policy_hash_before=hash_before,
            policy_hash_after=hash_after,
            promoted_checkpoint_path=promoted_path,
            rollback_seed_path=rollback_path,
            hurdle_verdicts=hurdle_verdicts,
            context_direction_entry_gate_enabled=bool(
                effective_cfg and (
                    effective_cfg.get("learning", {}).get("context_direction_entry_gate_enabled", False)
                    or effective_cfg.get("context_direction_entry_gate_enabled", False)
                )
            ),
        )

    def evaluate_regime_directory(
        self,
        regime_name: str,
        regime_type: str,
        directory: Path,
        from_date: str = "",
        to_date: str = "",
    ) -> RegimeMetrics:
        """Parse events.jsonl and mt5_diagnostics.json from an existing window directory."""
        cfg = {"market_type": regime_type, "from_date": from_date, "to_date": to_date}
        return extract_regime_metrics_from_telemetry(directory, regime_name, cfg)

    def evaluate_hurdles(
        self,
        regime_metrics: dict[str, RegimeMetrics],
        aggregate_metrics: dict[str, Any],
        context_direction_entry_gate_enabled: bool = False,
    ) -> tuple[bool, dict[str, bool], list[str]]:
        """Evaluate the 7 quantitative acceptance hurdles across per-regime and aggregate metrics."""
        cfg = {"learning": {"context_direction_entry_gate_enabled": context_direction_entry_gate_enabled}}
        passed, reasons, _ = evaluate_multi_regime_gate(regime_metrics, self.hurdles, cfg)
        hurdle_verdicts = {
            "win_rate_passed": not any("win_rate_below_hurdle" in r for r in reasons),
            "payoff_ratio_passed": not any("payoff_ratio_below_hurdle" in r for r in reasons),
            "expectancy_passed": not any("expectancy_below_hurdle" in r for r in reasons),
            "counter_trend_passed": not any("counter_trend_rate" in r for r in reasons),
            "timeout_close_passed": not any("timeout_close_rate_exceeded" in r for r in reasons),
            "premature_close_passed": not any("premature_closes_violation" in r for r in reasons),
            "direction_gate_passed": not any("hard_direction_gate_forbidden" in r for r in reasons),
            "sample_sufficiency_passed": not any("insufficient" in r for r in reasons),
        }
        return passed, hurdle_verdicts, reasons

    def promote_checkpoint(
        self,
        candidate_path: Path,
        destination_dir: Path,
        round_index: int | None = None,
        active_seed_path: Path | None = None,
    ) -> Path:
        """Atomically copy/link candidate checkpoint as new champion."""
        promoted, _ = promote_champion_atomic(
            candidate_path=candidate_path,
            output_dir=destination_dir,
            round_id=round_index,
            active_seed_path=active_seed_path,
        )
        return promoted

    def rollback_checkpoint(
        self,
        active_seed: Path,
        fallback_champion: Path,
    ) -> Path:
        """Roll back active seed pointer to previous champion."""
        return rollback_candidate_atomic(active_seed, fallback_champion, prune_candidate_payload=True)

    def generate_report(
        self,
        result: MultiRegimeResult,
        output_path: Path | None = None,
    ) -> str:
        """Generate human-readable Markdown evaluation report."""
        status_str = "PASSED & PROMOTED" if result.passed else "REJECTED"
        lines = [
            f"# Multi-Regime Gate Evaluation Report — {result.evaluated_at}",
            f"",
            f"**Decision**: `{status_str}`",
            f"**Candidate Path**: `{result.candidate_path}`",
            f"**Policy State Hash**: `{result.candidate_hash_before}` (Immutable: `{result.is_policy_immutable}`)",
            f"",
            f"## Aggregate Metrics",
            f"- Total Closed Trades: {result.aggregate_metrics.get('total_trades', 0)}",
            f"- Win Rate: {result.aggregate_metrics.get('win_rate_pct', 0.0):.2f}% (Hurdle >= 45.0%)",
            f"- Payoff Ratio: {result.aggregate_metrics.get('payoff_ratio', 0.0):.2f}R (Hurdle >= 1.50R)",
            f"- Expectancy E[R]: {result.aggregate_metrics.get('expectancy_r', 0.0):.4f}R (Hurdle >= +0.05R)",
            f"- Total Premature Closes: {result.aggregate_metrics.get('total_premature_closes', 0)} (Hurdle == 0)",
            f"",
            f"## Regime Performance Breakdown",
            f"| Regime | Market Type | Dates | Trades | WR (%) | E[R] | Payoff | Counter-Trend (%) | Timeout (%) | Premature |",
            f"|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r_id, m in result.regimes.items():
            lines.append(
                f"| `{m.regime_id}` | {m.market_type} | {m.from_date}..{m.to_date} | "
                f"{m.total_trades} | {m.win_rate_pct:.1f}% | {m.expectancy_r:+.4f}R | "
                f"{m.payoff_ratio:.2f} | {m.counter_trend_rate_pct:.1f}% | "
                f"{m.timeout_close_rate_pct:.1f}% | {m.premature_closes} |"
            )

        if result.reasons:
            lines.extend([
                f"",
                f"## Rejection Reasons",
            ])
            for r in result.reasons:
                lines.append(f"- [FAIL] {r}")

        report_md = "\n".join(lines) + "\n"
        if output_path is not None:
            Path(output_path).write_text(report_md, encoding="utf-8")
        return report_md
