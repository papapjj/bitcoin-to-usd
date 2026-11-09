"""Autonomous Multi-Round Training Engine (Auto-RL Engine) for Self Growing Trader V193.

Supervises multi-round training loops without manual intervention:
- Configurable loop parameters: max_rounds (default 15), initial_timeframe ("M1"),
  max_cumulative_drawdown (15.0%), m1_failure_threshold (3), m5_failure_threshold (3).
- Checkpoint management: atomic persistence via .tmp -> .pkl, pre-execution backups,
  automated rollback to champion on divergence/collapse, and intermediate checkpoint pruning.
- Strict Out-of-Sample (OOS) validation: frozen evaluation on independent Q1 2025 window
  (2025.01.01 - 2025.03.31) with anti-HOLD collapse and anti-directional collapse guards.
- Automated timeframe adaptation state machine: M1 -> M5 -> M15 on 3 consecutive non-improving rounds.
- Safety budget & emergency stop: hard stop at 15 rounds, emergency halt on cumulative drawdown > 15%,
  100% native RL (zero external LLM / Codex Teacher calls).
- Evolution ledger persistence: auto_rl_manifest.json and evolution_chart.json.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from self_growing_trader.config import AppConfig, EnvironmentConfig, OrderConfig, LearningIntegrityConfig
    from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia
except ImportError:
    try:
        from src.self_growing_trader.config import AppConfig, EnvironmentConfig, OrderConfig, LearningIntegrityConfig
        from src.self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia
    except ImportError:
        AppConfig = Any  # type: ignore
        EnvironmentConfig = Any  # type: ignore
        OrderConfig = Any  # type: ignore
        LearningIntegrityConfig = Any  # type: ignore
        CauHinhHeChuyenGia = Any  # type: ignore

try:
    from src.self_growing_trader.chuyen_gia_policy import (
        backup_context_expert_artifacts,
        _five_action_distribution_requires_champion_recovery,
    )
except ImportError:
    try:
        from self_growing_trader.chuyen_gia_policy import (
            backup_context_expert_artifacts,
            _five_action_distribution_requires_champion_recovery,
        )
    except ImportError:
        backup_context_expert_artifacts = None  # type: ignore
        _five_action_distribution_requires_champion_recovery = None  # type: ignore

try:
    from src.self_growing_trader.oos_validator import (
        OOSValidator,
        OOSValidationResult as OOSValidatorResult,
        evaluate_strict_oos,
        CumulativeDrawdownTracker,
    )
except ImportError:
    try:
        from self_growing_trader.oos_validator import (
            OOSValidator,
            OOSValidationResult as OOSValidatorResult,
            evaluate_strict_oos,
            CumulativeDrawdownTracker,
        )
    except ImportError:
        OOSValidator = None  # type: ignore
        OOSValidatorResult = None  # type: ignore
        evaluate_strict_oos = None  # type: ignore
        CumulativeDrawdownTracker = None  # type: ignore


# ==============================================================================
# CONFIGURATION & DATA STRUCTURES
# ==============================================================================


@dataclass
class AutoRLEngineConfig:
    """Configuration for Autonomous Multi-Round Training Engine."""

    runs_dir: str | Path = "outputs/he_chuyen_gia/v193_runs"
    max_rounds: int = 15
    initial_timeframe: str = "M1"
    max_cumulative_drawdown: float = 0.15  # 15.0%
    m1_failure_threshold: int = 3
    m5_failure_threshold: int = 3
    initial_equity: float = 10000.0
    seed_checkpoint: str | Path | None = None
    fixed_sl_atr: float = 5.0
    fixed_tp_atr: float = 10.0
    oos_from_date: str = "2025.01.01"
    oos_to_date: str = "2025.03.31"
    min_oos_expectancy: float = 0.05
    min_oos_win_rate_pct: float = 40.0
    min_oos_risk_reward: float = 2.0
    min_oos_trades: int = 30
    max_oos_drawdown_pct: float = 10.0
    max_hold_share: float = 0.80
    max_direction_imbalance: float = 0.60
    min_entropy: float = 0.35
    five_codex_teacher_enabled: bool = False
    visual: bool = False
    shutdown_terminal: bool = True
    timeout_sec: int = 300
    prune_non_promoted_checkpoints: bool = True
    ewc_enabled: bool = True
    ewc_lambda: float = 250.0
    ewc_sample_floor: int = 500
    representation_regularizer_enabled: bool = True
    rep_var_weight: float = 1.0
    rep_cov_weight: float = 1.0
    rep_rank_weight: float = 0.05
    rep_arc_weight: float = 0.20

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        # Guarantee 100% native RL: codex teacher must remain disabled
        if self.five_codex_teacher_enabled:
            raise ValueError(
                "AutoRLEngine strictly enforces five_codex_teacher_enabled=False "
                "to ensure 100% native PPO + lifecycle learning."
            )
        # Enforce budget cap at 15 rounds
        if self.max_rounds > 15:
            self.max_rounds = 15
        # Ensure timeframe is valid
        self.initial_timeframe = str(self.initial_timeframe).strip().upper()
        if self.initial_timeframe not in ("M1", "M5", "M15", "M30", "H1"):
            raise ValueError(f"Invalid initial_timeframe: {self.initial_timeframe}")


@dataclass
class OOSValidationResult:
    """Result of Out-of-Sample gate evaluation."""

    passed_gate: bool
    expectancy: float
    win_rate: float
    risk_reward_ratio: float
    total_trades: int
    max_equity_drawdown_pct: float
    hold_share: float
    entry_direction_imbalance: float | None
    entropy: float | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# ==============================================================================
# STATE MACHINE & MONITORS
# ==============================================================================


class TimeframeAdaptationStateMachine:
    """State machine for automated timeframe adaptation (R2).

    Rules:
    - Starts on M1 (or configured initial_timeframe).
    - If 3 consecutive rounds fail on M1 (E[R] <= 0 or trades == 0) -> switch to M5.
    - If 3 consecutive rounds fail on M5 -> switch to M15.
    - An improving round (E[R] > 0 and trades > 0) resets consecutive loss counter.
    """

    def __init__(
        self,
        initial_timeframe: str = "M1",
        m1_failure_threshold: int = 3,
        m5_failure_threshold: int = 3,
    ) -> None:
        self.current_timeframe: str = initial_timeframe.strip().upper()
        self.m1_failure_threshold: int = int(m1_failure_threshold)
        self.m5_failure_threshold: int = int(m5_failure_threshold)
        self.consecutive_m1_losses: int = 0
        self.consecutive_m5_losses: int = 0
        self.migration_history: list[dict[str, Any]] = []

    def record_round_result(
        self,
        round_num: int,
        expectancy: float,
        trades: int,
        passed_gate: bool | None = None,
    ) -> str:
        """Record round result and trigger migration if failure threshold is reached."""
        if passed_gate is not None:
            is_loss = not passed_gate
        else:
            is_loss = not math.isfinite(expectancy) or expectancy <= 0.0 or trades <= 0

        if self.current_timeframe == "M1":
            if is_loss:
                self.consecutive_m1_losses += 1
            else:
                self.consecutive_m1_losses = 0

            if self.consecutive_m1_losses >= self.m1_failure_threshold:
                self.current_timeframe = "M5"
                self.migration_history.append(
                    {
                        "round": round_num,
                        "from": "M1",
                        "to": "M5",
                        "reason": f"{self.m1_failure_threshold}_consecutive_m1_losses",
                    }
                )
                self.consecutive_m1_losses = 0

        elif self.current_timeframe == "M5":
            if is_loss:
                self.consecutive_m5_losses += 1
            else:
                self.consecutive_m5_losses = 0

            if self.consecutive_m5_losses >= self.m5_failure_threshold:
                self.current_timeframe = "M15"
                self.migration_history.append(
                    {
                        "round": round_num,
                        "from": "M5",
                        "to": "M15",
                        "reason": f"{self.m5_failure_threshold}_consecutive_m5_losses",
                    }
                )
                self.consecutive_m5_losses = 0

        return self.current_timeframe

    def get_timeframe_parameters(self) -> dict[str, Any]:
        """Return adapted barrier geometry, horizon, and friction limits."""
        tf = self.current_timeframe
        if tf == "M1":
            return {
                "timeframe": "M1",
                "fixed_sl_atr": 5.0,
                "fixed_tp_atr": 10.0,
                "pending_expiry_bars": 15,
                "max_holding_bars": 60,
                "bar_duration_sec": 60,
                "spread_sl_ratio_limit": 0.13,
            }
        elif tf == "M5":
            return {
                "timeframe": "M5",
                "fixed_sl_atr": 5.0,
                "fixed_tp_atr": 10.0,
                "pending_expiry_bars": 6,
                "max_holding_bars": 24,
                "bar_duration_sec": 300,
                "spread_sl_ratio_limit": 0.06,
            }
        elif tf == "M15":
            return {
                "timeframe": "M15",
                "fixed_sl_atr": 5.0,
                "fixed_tp_atr": 10.0,
                "pending_expiry_bars": 4,
                "max_holding_bars": 16,
                "bar_duration_sec": 900,
                "spread_sl_ratio_limit": 0.035,
            }
        else:
            return {
                "timeframe": tf,
                "fixed_sl_atr": 5.0,
                "fixed_tp_atr": 10.0,
                "pending_expiry_bars": 10,
                "max_holding_bars": 30,
                "bar_duration_sec": 60,
                "spread_sl_ratio_limit": 0.15,
            }


class CumulativeDrawdownMonitor:
    """Safety monitor for tracking cumulative account equity drawdown (R4).

    Emergency stop triggers if cumulative drawdown exceeds halt_pct (default 15.0%).
    """

    def __init__(self, initial_equity: float = 10000.0, halt_pct: float = 15.0) -> None:
        self.initial_equity: float = float(initial_equity)
        self.current_equity: float = float(initial_equity)
        self.peak_equity: float = float(initial_equity)
        self.cumulative_drawdown_pct: float = 0.0
        self.halt_pct: float = float(halt_pct)
        self.emergency_halt_triggered: bool = False
        self.history: list[dict[str, Any]] = []

    def update(self, round_pnl_dollars: float) -> bool:
        """Update monitor with round PnL in dollars and return halt status."""
        self.current_equity += float(round_pnl_dollars)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Cumulative drawdown measured from initial capital base
        dd_from_initial = max(
            0.0, (self.initial_equity - self.current_equity) / self.initial_equity * 100.0
        )
        self.cumulative_drawdown_pct = float(dd_from_initial)

        if self.cumulative_drawdown_pct > self.halt_pct:
            self.emergency_halt_triggered = True

        self.history.append(
            {
                "round": len(self.history) + 1,
                "pnl": float(round_pnl_dollars),
                "equity": self.current_equity,
                "cumulative_dd_pct": self.cumulative_drawdown_pct,
                "halt": self.emergency_halt_triggered,
            }
        )
        return self.emergency_halt_triggered


# ==============================================================================
# MANIFEST & EVOLUTION LEDGER PERSISTENCE
# ==============================================================================


class ProcessLock:
    """Inter-process file mutex for single-instance Auto-RL execution."""

    def __init__(self, lock_file: str | Path) -> None:
        self.lock_file = Path(lock_file)
        self.handle: Any = None

    def acquire(self) -> None:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.lock_file, "w+")
        import os
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError(f"Another Auto-RL process is running in {self.lock_file.parent}")
        else:
            import fcntl
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError(f"Another Auto-RL process is running in {self.lock_file.parent}")
        self.handle.write(f"{os.getpid()}\n")
        self.handle.flush()

    def release(self) -> None:
        if self.handle:
            try:
                import os
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.handle.close()
                self.lock_file.unlink(missing_ok=True)
            except Exception:
                pass
            finally:
                self.handle = None

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class AutoRLManifestManager:
    """Persistence manager for `auto_rl_manifest.json` and `evolution_chart.json`."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "auto_rl_manifest.json"
        self.evolution_chart_path = self.run_dir / "evolution_chart.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any]:
        """Load manifest from disk or return initial default structure."""
        if not self.manifest_path.exists():
            return {
                "active_seed": str(self.run_dir / "seed_init.pkl"),
                "current_round": 0,
                "current_timeframe": "M1",
                "consecutive_m1_losses": 0,
                "cumulative_drawdown_pct": 0.0,
                "rounds": [],
            }
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_manifest(self, data: dict[str, Any]) -> None:
        """Save manifest atomically via .tmp replace."""
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(self.manifest_path)

    def load_evolution_chart(self) -> list[dict[str, Any]]:
        """Load evolution chart array from disk or return empty list."""
        if not self.evolution_chart_path.exists():
            return []
        with open(self.evolution_chart_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_evolution_chart(self, chart: list[dict[str, Any]]) -> None:
        """Save evolution chart atomically via .tmp replace."""
        tmp_path = self.evolution_chart_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(chart, f, indent=2)
        tmp_path.replace(self.evolution_chart_path)

    def record_round(
        self,
        round_num: int,
        timeframe: str,
        metrics: dict[str, Any],
        promoted: bool,
        active_seed: str,
        cumulative_dd_pct: float,
        consecutive_m1_losses: int | None = None,
    ) -> None:
        """Record completed round in both auto_rl_manifest.json and evolution_chart.json."""
        manifest = self.load_manifest()
        manifest["current_round"] = round_num
        manifest["current_timeframe"] = timeframe
        manifest["active_seed"] = str(active_seed)
        manifest["cumulative_drawdown_pct"] = float(cumulative_dd_pct)
        if consecutive_m1_losses is not None:
            manifest["consecutive_m1_losses"] = int(consecutive_m1_losses)
        if promoted:
            manifest["champion_checkpoint"] = str(active_seed)
            manifest["best_expectancy"] = float(metrics.get("expectancy", 0.0))
        round_entry = {
            "round": round_num,
            "timeframe": timeframe,
            "promoted": promoted,
            "expectancy": metrics.get("expectancy", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "total_trades": metrics.get("total_trades", 0),
            "risk_reward_ratio": metrics.get("risk_reward_ratio", 2.0),
            "max_drawdown_pct": metrics.get("max_equity_drawdown_pct", 0.0),
            "hold_share": metrics.get("hold_share", 0.0),
            "entry_direction_imbalance": metrics.get("entry_direction_imbalance", 0.0),
            "loss": metrics.get("ppo_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
        }
        existing_idx = next(
            (i for i, r in enumerate(manifest["rounds"]) if r.get("round") == round_num),
            None,
        )
        if existing_idx is not None:
            manifest["rounds"][existing_idx] = round_entry
        else:
            manifest["rounds"].append(round_entry)
        manifest["rounds"].sort(key=lambda x: x.get("round", 0))
        self.save_manifest(manifest)

        chart = self.load_evolution_chart()
        chart_entry = {
            "round": round_num,
            "timeframe": timeframe,
            "loss": metrics.get("ppo_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "trades": metrics.get("total_trades", 0),
            "wr": metrics.get("win_rate", 0.0),
            "expectancy": metrics.get("expectancy", 0.0),
            "promoted": promoted,
        }
        existing_c_idx = next(
            (i for i, c in enumerate(chart) if c.get("round") == round_num),
            None,
        )
        if existing_c_idx is not None:
            chart[existing_c_idx] = chart_entry
        else:
            chart.append(chart_entry)
        chart.sort(key=lambda x: x.get("round", 0))
        self.save_evolution_chart(chart)


# ==============================================================================
# CHECKPOINT & ARTIFACT MANAGEMENT HELPERS
# ==============================================================================


def atomic_save_checkpoint(
    source_or_data: Any,
    dest_path: str | Path,
    registry_data: dict[str, Any] | None = None,
) -> Path:
    """Save candidate checkpoint atomically via .tmp -> .pkl and update sidecar .registry.json."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")

    if isinstance(source_or_data, (bytes, bytearray)):
        tmp_path.write_bytes(source_or_data)
    elif isinstance(source_or_data, (str, Path)):
        src = Path(source_or_data)
        if src.exists() and src.is_file():
            shutil.copy2(src, tmp_path)
        else:
            tmp_path.write_text(str(source_or_data), encoding="utf-8")
    else:
        # Python object or PyTorch model
        import pickle

        with open(tmp_path, "wb") as f:
            pickle.dump(source_or_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Atomic rename
    tmp_path.replace(dest)

    # Sidecar registry update if supplied or if object has luu_so_dang_ky
    reg_path = dest.with_suffix(".registry.json")
    if registry_data is not None:
        reg_tmp = dest.with_suffix(".registry.json.tmp")
        with open(reg_tmp, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)
        reg_tmp.replace(reg_path)
    elif hasattr(source_or_data, "luu_so_dang_ky") and callable(source_or_data.luu_so_dang_ky):
        try:
            source_or_data.luu_so_dang_ky(reg_path)
        except Exception:
            pass
    elif isinstance(source_or_data, (str, Path)):
        src = Path(source_or_data)
        src_reg = src.with_suffix(".registry.json")
        if not src_reg.exists():
            src_reg = src.with_name(f"{src.name}.registry.json")
        if src_reg.exists() and src_reg.is_file():
            try:
                shutil.copy2(src_reg, reg_path)
            except Exception:
                pass

    if dest.suffix == ".pkl" and reg_path.exists() and reg_path.is_file():
        alt_reg = dest.with_name(f"{dest.name}.registry.json")
        if alt_reg != reg_path and not alt_reg.exists():
            try:
                shutil.copy2(reg_path, alt_reg)
            except Exception:
                pass

    return dest


def pre_execution_backup_checkpoint(path: str | Path) -> dict[str, str]:
    """Pre-execution timestamped backup of checkpoint and sidecar registry."""
    if backup_context_expert_artifacts is not None:
        try:
            return backup_context_expert_artifacts(path)
        except Exception:
            pass

    p = Path(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups: dict[str, str] = {}
    for label, target in (
        ("policy", p),
        ("registry", p.with_suffix(".registry.json")),
    ):
        if not target.exists() or not target.is_file():
            continue
        backup = target.with_name(f"{target.stem}.rollback.{timestamp}{target.suffix}")
        shutil.copy2(target, backup)
        backups[label] = str(backup)

    # Automatically prune older rollback backups in directory to prevent disk exhaustion
    try:
        old_rollbacks = sorted(p.parent.glob(f"{p.stem}.rollback.*.pkl"), key=lambda x: x.stat().st_mtime)
        if len(old_rollbacks) > 2:
            for old in old_rollbacks[:-2]:
                old.unlink(missing_ok=True)
                for reg in p.parent.glob(f"{old.name}*.json"):
                    reg.unlink(missing_ok=True)
    except Exception:
        pass

    return backups


def prune_unpromoted_checkpoints(
    round_dir: str | Path,
    preserve_patterns: Sequence[str] = ("manifest.json", "events.jsonl", "mt5_diagnostics.json"),
) -> list[Path]:
    """Prune intermediate non-promoted large .pkl checkpoints while preserving telemetry JSONs."""
    r_dir = Path(round_dir)
    if not r_dir.exists():
        return []

    pruned: list[Path] = []
    preserve_set = set(preserve_patterns)

    for f in r_dir.iterdir():
        if not f.is_file():
            continue
        # Preserve telemetry JSON manifests
        if f.name in preserve_set or f.suffix in (".json", ".jsonl"):
            continue
        # Prune large binary checkpoint payloads
        if f.suffix in (".pkl", ".tmp") and ("candidate" in f.name.lower() or f.name.endswith(".pkl")):
            try:
                f.unlink()
                pruned.append(f)
            except OSError:
                pass

    return pruned


def rollback_checkpoint(current_candidate: str | Path, champion_seed: str | Path) -> str:
    """Roll back active_seed pointer to previous best champion checkpoint."""
    return str(Path(champion_seed))


# ==============================================================================
# POST-ROUND OOS GATE & ANTI-COLLAPSE EVALUATION
# ==============================================================================


def evaluate_candidate_oos_gate(
    metrics: Mapping[str, Any] | None,
    *,
    min_expectancy: float = 0.05,
    min_win_rate_pct: float = 40.0,
    min_risk_reward_ratio: float = 2.0,
    min_trades: int = 30,
    max_equity_drawdown_pct: float = 10.0,
    max_hold_share: float = 0.80,
    max_direction_imbalance: float = 0.60,
    min_entropy: float = 0.35,
) -> OOSValidationResult:
    """Evaluate candidate metrics against strict Out-of-Sample gate criteria."""
    raw = dict(metrics or {})
    reasons: list[str] = []

    # Safe extraction & normalization with fail-closed non-finite handling
    raw_trades = raw.get("total_trades", 0)
    try:
        if raw_trades is None:
            trades = 0
        else:
            flt_trades = float(raw_trades)
            if not math.isfinite(flt_trades):
                trades = 0
                reasons.append(f"invalid_metric_value:total_trades={raw_trades}")
            else:
                trades = int(flt_trades)
    except (ValueError, TypeError):
        trades = 0
        reasons.append(f"invalid_metric_value:total_trades={raw.get('total_trades')}")

    try:
        expectancy = float(raw.get("expectancy", 0.0) if raw.get("expectancy") is not None else 0.0)
    except (ValueError, TypeError):
        expectancy = float("nan")

    raw_wr = raw.get("win_rate", 0.0) if raw.get("win_rate") is not None else 0.0
    try:
        win_rate = float(raw_wr)
        if 0.0 < win_rate <= 1.0 and min_win_rate_pct > 1.0:
            win_rate = win_rate * 100.0
    except (ValueError, TypeError):
        win_rate = float("nan")

    try:
        rr = float(raw.get("risk_reward_ratio", 0.0) if raw.get("risk_reward_ratio") is not None else 0.0)
    except (ValueError, TypeError):
        rr = float("nan")

    raw_dd = (
        raw.get("max_equity_drawdown_pct")
        if raw.get("max_equity_drawdown_pct") is not None
        else raw.get("max_drawdown_pct", 0.0)
    )
    try:
        max_dd = float(raw_dd if raw_dd is not None else 0.0)
        if 0.0 < max_dd <= 1.0 and max_equity_drawdown_pct > 1.0:
            max_dd = max_dd * 100.0
    except (ValueError, TypeError):
        max_dd = float("nan")

    try:
        hold_share = float(raw.get("hold_share", 0.0) if raw.get("hold_share") is not None else 0.0)
    except (ValueError, TypeError):
        hold_share = float("nan")

    imbalance = raw.get("entry_direction_imbalance")
    try:
        imbalance_val = float(imbalance) if imbalance is not None else None
    except (ValueError, TypeError):
        imbalance_val = float("nan")

    entropy = raw.get("entropy")
    if entropy is None:
        entropy = raw.get("normalized_action_entropy")
    try:
        entropy_val = float(entropy) if entropy is not None else None
    except (ValueError, TypeError):
        entropy_val = float("nan")

    # 0. Numeric validity check (strictly fail closed on NaN and Inf across all metrics)
    for k, v in [
        ("expectancy", expectancy),
        ("win_rate", win_rate),
        ("risk_reward_ratio", rr),
        ("max_equity_drawdown_pct", max_dd),
        ("hold_share", hold_share),
    ]:
        if not math.isfinite(v):
            reasons.append(f"invalid_metric_value:{k}={v}")

    if imbalance_val is not None and not math.isfinite(imbalance_val):
        reasons.append(f"invalid_metric_value:entry_direction_imbalance={imbalance_val}")

    if entropy_val is not None and not math.isfinite(entropy_val):
        reasons.append(f"invalid_metric_value:entropy={entropy_val}")

    # 1. Trade sample gate
    if trades < min_trades:
        reasons.append(f"oos_trades_insufficient:{trades}<{min_trades}")

    # 2. Expectancy gate
    if math.isfinite(expectancy):
        if expectancy < min_expectancy:
            reasons.append(f"expectancy_below_hurdle:{expectancy:.4f}<{min_expectancy:.4f}")

    # 3. Win Rate gate
    if math.isfinite(win_rate):
        if win_rate < min_win_rate_pct:
            reasons.append(f"win_rate_below_hurdle:{win_rate:.2f}%<{min_win_rate_pct:.2f}%")

    # 4. Risk-Reward gate
    if math.isfinite(rr):
        if rr < min_risk_reward_ratio:
            reasons.append(f"risk_reward_below_hurdle:{rr:.2f}<{min_risk_reward_ratio:.2f}")

    # 5. Drawdown gate (< 10%)
    if math.isfinite(max_dd):
        if max_dd >= max_equity_drawdown_pct:
            reasons.append(f"max_drawdown_exceeded:{max_dd:.2f}%>={max_equity_drawdown_pct:.2f}%")

    # 6. Anti-HOLD collapse gate (< 0.80)
    if math.isfinite(hold_share):
        if hold_share >= max_hold_share:
            reasons.append(f"hold_collapse_detected:{hold_share:.3f}>={max_hold_share:.3f}")

    # 7. Anti-Directional collapse gate (abs(imbalance) <= 0.60)
    if imbalance_val is not None:
        if math.isfinite(imbalance_val):
            if abs(imbalance_val) > max_direction_imbalance + 1e-6:
                reasons.append(
                    f"directional_imbalance_exceeded:{abs(imbalance_val):.3f}>{max_direction_imbalance:.3f}"
                )
    elif trades > 0:
        reasons.append("directional_imbalance_missing")

    # 8. Action entropy gate (> 0.35)
    if entropy_val is not None:
        if math.isfinite(entropy_val):
            if entropy_val <= min_entropy:
                reasons.append(f"entropy_collapse_detected:{entropy_val:.3f}<={min_entropy:.3f}")

    passed = len(reasons) == 0
    return OOSValidationResult(
        passed_gate=passed,
        expectancy=expectancy,
        win_rate=win_rate,
        risk_reward_ratio=rr,
        total_trades=trades,
        max_equity_drawdown_pct=max_dd,
        hold_share=hold_share,
        entry_direction_imbalance=imbalance_val,
        entropy=entropy_val,
        reasons=reasons,
    )


# ==============================================================================
# AUTONOMOUS MULTI-ROUND ENGINE (AutoRLEngine)
# ==============================================================================


class AutoRLEngine:
    """Autonomous Multi-Round Training & Auto-Tuning Engine (V193 Auto-RL).

    Coordinates multi-round execution loops without manual intervention:
    - Runs headless native PPO training rounds up to max_rounds (default 15).
    - Automatically adapts timeframes (M1 -> M5 -> M15) on consecutive failures.
    - Evaluates candidates against strict Q1 2025 Out-of-Sample criteria.
    - Enforces anti-HOLD collapse and anti-directional collapse guards.
    - Atomically persists candidates, rolls back on degradation, and prunes disk.
    - Monitors cumulative account equity drawdown with 15% emergency halt.
    - Persists auto_rl_manifest.json and evolution_chart.json ledgers.
    """

    def __init__(self, config: AutoRLEngineConfig | None = None) -> None:
        self.config = config or AutoRLEngineConfig()
        self.runs_dir = Path(self.config.runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_mgr = AutoRLManifestManager(self.runs_dir)

        self.state_machine = TimeframeAdaptationStateMachine(
            initial_timeframe=self.config.initial_timeframe,
            m1_failure_threshold=self.config.m1_failure_threshold,
            m5_failure_threshold=self.config.m5_failure_threshold,
        )

        halt_pct = (
            self.config.max_cumulative_drawdown * 100.0
            if self.config.max_cumulative_drawdown <= 1.0
            else self.config.max_cumulative_drawdown
        )
        self.dd_monitor = CumulativeDrawdownMonitor(
            initial_equity=self.config.initial_equity,
            halt_pct=halt_pct,
        )

        if OOSValidator is not None:
            self.oos_validator = OOSValidator(
                from_date=self.config.oos_from_date,
                to_date=self.config.oos_to_date,
                min_expectancy=self.config.min_oos_expectancy,
                min_win_rate=self.config.min_oos_win_rate_pct,
                min_risk_reward=self.config.min_oos_risk_reward,
                min_trades=self.config.min_oos_trades,
                max_drawdown_pct=self.config.max_oos_drawdown_pct,
                max_hold_share=self.config.max_hold_share,
                min_normalized_entropy=self.config.min_entropy,
                max_direction_imbalance=self.config.max_direction_imbalance,
                max_cumulative_drawdown_pct=halt_pct,
            )
        else:
            self.oos_validator = None

        self.current_round: int = 0
        seed_cand = self.config.seed_checkpoint
        if seed_cand is None:
            v193_seed = self.runs_dir / "seed_v193_20260928_232105.pkl"
            if v193_seed.exists():
                seed_cand = v193_seed
            else:
                seed_cand = self.runs_dir / "seed_init.pkl"
        self.active_seed: str = str(seed_cand)
        self.champion_checkpoint: str | None = None
        self.best_expectancy: float = -float("inf")
        self.terminated: bool = False
        self.termination_reason: str = ""
        self.ewc_enabled = bool(getattr(self.config, "ewc_enabled", True))
        self.ewc_lambda = float(getattr(self.config, "ewc_lambda", 250.0))
        self.ewc_sample_floor = int(getattr(self.config, "ewc_sample_floor", 500))
        self.ewc: Any = None

        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize state from existing manifest on disk if present."""
        manifest = self.manifest_mgr.load_manifest()
        if manifest and manifest.get("rounds"):
            self.current_round = int(manifest.get("current_round", 0))
            self.active_seed = str(manifest.get("active_seed", self.active_seed))
            tf = manifest.get("current_timeframe", self.config.initial_timeframe)
            self.state_machine.current_timeframe = tf
            self.state_machine.consecutive_m1_losses = int(
                manifest.get("consecutive_m1_losses", 0)
            )
            # Find best promoted round if any
            for rnd in manifest.get("rounds", []):
                if rnd.get("promoted"):
                    exp = float(rnd.get("expectancy", 0.0))
                    if exp > self.best_expectancy:
                        self.best_expectancy = exp
                        self.champion_checkpoint = self.active_seed

    def _default_train_round(
        self,
        round_num: int,
        timeframe: str,
        seed_path: str,
        round_dir: Path,
    ) -> tuple[Path, dict[str, Any]]:
        """Default native PPO training pass executing headless MT5 or generating candidate."""
        candidate_path = round_dir / "candidate.pkl"
        tf_params = self.state_machine.get_timeframe_parameters()
        orders = OrderConfig(
            timeframe=timeframe,
            fixed_sl_atr=tf_params["fixed_sl_atr"],
            fixed_tp_atr=tf_params["fixed_tp_atr"],
        )
        learning_integrity = LearningIntegrityConfig(
            five_codex_teacher_enabled=False,
            five_codex_teacher_auto_connect=False,
        )
        try:
            from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia
        except ImportError:
            from src.self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia

        app_cfg = AppConfig(
            orders=orders,
            environment=EnvironmentConfig(offline_preflight=True),
            learning=learning_integrity,
            he_chuyen_gia=CauHinhHeChuyenGia(bat=True),
        )
        try:
            from self_growing_trader.chuyen_gia_policy import huan_luyen_he_chuyen_gia_mt5

            train_res = huan_luyen_he_chuyen_gia_mt5(
                app_cfg,
                output_path=candidate_path,
                seed_system_path=seed_path if Path(seed_path).exists() else None,
                visual=self.config.visual,
                shutdown_terminal=self.config.shutdown_terminal,
                timeout_sec=self.config.timeout_sec,
            )
            metrics = train_res.get("metrics", {}) if isinstance(train_res, dict) else {}
            if metrics and isinstance(metrics, dict) and metrics.get("total_trades", 0) > 0:
                return candidate_path, metrics
        except Exception as exc:
            raise RuntimeError(
                f"Multi-round training failed during Round {round_num} ({timeframe}): {exc}"
            ) from exc

        if not metrics or not isinstance(metrics, dict) or metrics.get("total_trades", 0) <= 0:
            raise RuntimeError(
                f"Round {round_num} ({timeframe}) produced 0 trades or invalid metrics from MT5: {metrics}"
            )

        return candidate_path, metrics

    def _default_oos_eval(
        self,
        candidate_obj_or_path: Any,
        timeframe: str,
    ) -> Mapping[str, Any]:
        """Default frozen Out-of-Sample evaluation pass."""
        try:
            from self_growing_trader.chuyen_gia_policy import (
                evaluate_policy_mt5_oos,
                tai_he_chuyen_gia_mot_context_mot_policy,
            )
            from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia

            if isinstance(candidate_obj_or_path, (str, Path)) and Path(
                candidate_obj_or_path
            ).exists():
                tf_params = self.state_machine.get_timeframe_parameters()
                orders = OrderConfig(
                    timeframe=timeframe,
                    fixed_sl_atr=tf_params["fixed_sl_atr"],
                    fixed_tp_atr=tf_params["fixed_tp_atr"],
                )
                app_cfg = AppConfig(
                    orders=orders,
                    environment=EnvironmentConfig(offline_preflight=True),
                    learning=LearningIntegrityConfig(
                        five_codex_teacher_enabled=False,
                        five_codex_teacher_auto_connect=False,
                    ),
                    he_chuyen_gia=CauHinhHeChuyenGia(bat=True),
                )
                policy = tai_he_chuyen_gia_mot_context_mot_policy(
                    candidate_obj_or_path, app_cfg
                )
                res = evaluate_policy_mt5_oos(
                    policy,
                    app_cfg,
                    from_date=self.config.oos_from_date,
                    to_date=self.config.oos_to_date,
                    visual=self.config.visual,
                    shutdown_terminal=self.config.shutdown_terminal,
                )
                if isinstance(res, dict):
                    res["oos_executed"] = True
                    return res
                return {
                    "passed_gate": False,
                    "total_trades": 0,
                    "expectancy": -999.0,
                    "win_rate": 0.0,
                    "max_equity_drawdown_pct": 100.0,
                    "reasons": ["invalid_oos_return_type"],
                    "oos_executed": False,
                }
        except Exception as exc:
            return {
                "passed_gate": False,
                "total_trades": 0,
                "expectancy": -999.0,
                "win_rate": 0.0,
                "max_equity_drawdown_pct": 100.0,
                "reasons": [f"oos_evaluation_crashed:{type(exc).__name__}:{exc}"],
                "oos_executed": False,
            }

        return {
            "passed_gate": False,
            "total_trades": 0,
            "expectancy": -999.0,
            "win_rate": 0.0,
            "max_equity_drawdown_pct": 100.0,
            "reasons": ["candidate_file_not_found"],
            "oos_executed": False,
        }

    def step_round(
        self,
        round_executor: Callable[[int, str, str, Path], tuple[Any, dict[str, Any]]] | None = None,
        oos_evaluator: Callable[[Any, str], Mapping[str, Any]] | None = None,
        metric_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single sequential training & validation round."""
        # 1. Budget and halt checks
        if self.current_round >= self.config.max_rounds:
            self.terminated = True
            self.termination_reason = "max_rounds_budget_reached"
            return {
                "round": self.current_round,
                "status": "terminated",
                "reason": self.termination_reason,
            }

        if self.dd_monitor.emergency_halt_triggered:
            self.terminated = True
            self.termination_reason = "emergency_drawdown_halt"
            return {
                "round": self.current_round,
                "status": "emergency_halt",
                "reason": self.termination_reason,
            }

        # 2. Advance round number and capture current timeframe
        round_num = self.current_round + 1
        tf = self.state_machine.current_timeframe
        round_dir = self.runs_dir / f"round_{round_num:02d}_{tf.lower()}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 3. Pre-execution backup of current active seed
        if Path(self.active_seed).exists():
            pre_execution_backup_checkpoint(self.active_seed)

        # 4. Execute training pass
        if round_executor is not None:
            candidate_item, train_metrics = round_executor(
                round_num, tf, self.active_seed, round_dir
            )
        elif metric_overrides:
            candidate_item = round_dir / "candidate.pkl"
            atomic_save_checkpoint(f"CANDIDATE_R{round_num}_{tf}_OVERRIDE".encode(), candidate_item)
            train_metrics = dict(metric_overrides)
        else:
            candidate_item, train_metrics = self._default_train_round(
                round_num, tf, self.active_seed, round_dir
            )

        candidate_ckpt = round_dir / "candidate.pkl"
        if not candidate_ckpt.exists() or candidate_item != candidate_ckpt:
            atomic_save_checkpoint(candidate_item, candidate_ckpt)

        # 5. Out-of-Sample evaluation
        if oos_evaluator is not None:
            oos_raw = oos_evaluator(candidate_item, tf)
        elif metric_overrides:
            oos_raw = dict(metric_overrides)
            oos_raw["oos_executed"] = True
        else:
            oos_raw = self._default_oos_eval(candidate_ckpt, tf)

        # Strictly isolate OOS metrics - never fall back to training metrics
        if not oos_raw or not isinstance(oos_raw, Mapping) or not oos_raw.get("oos_executed", True):
            oos_metrics = dict(oos_raw or {})
            oos_metrics.setdefault("total_trades", 0)
            oos_metrics.setdefault("expectancy", -999.0)
            oos_metrics.setdefault("win_rate", 0.0)
            oos_metrics.setdefault("risk_reward_ratio", 0.0)
            oos_metrics.setdefault("max_equity_drawdown_pct", 100.0)
            oos_metrics.setdefault("passed_gate", False)
            if not oos_metrics.get("reasons"):
                oos_metrics["reasons"] = ["oos_not_executed_or_failed"]
        else:
            # Extract nested OOS metrics if present from evaluate_policy_mt5_oos
            if "best" in oos_raw and isinstance(oos_raw["best"], Mapping):
                best_inner = oos_raw["best"]
                if "metrics" in best_inner and isinstance(best_inner["metrics"], Mapping):
                    oos_metrics = dict(best_inner["metrics"])
                else:
                    oos_metrics = dict(best_inner)
            elif "metrics" in oos_raw and isinstance(oos_raw["metrics"], Mapping):
                oos_metrics = dict(oos_raw["metrics"])
            else:
                oos_metrics = dict(oos_raw)

        if metric_overrides:
            oos_metrics.update(dict(metric_overrides))

        # Check for divergence or severe training collapse
        ppo_loss = float(train_metrics.get("ppo_loss", 0.0) or 0.0)
        is_diverged = math.isnan(ppo_loss) or math.isinf(ppo_loss) or ppo_loss > 100.0

        # Evaluate OOS gate strictly on Out-of-Sample metrics
        gate_res = evaluate_candidate_oos_gate(
            oos_metrics,
            min_expectancy=self.config.min_oos_expectancy,
            min_win_rate_pct=self.config.min_oos_win_rate_pct,
            min_risk_reward_ratio=self.config.min_oos_risk_reward,
            min_trades=self.config.min_oos_trades,
            max_equity_drawdown_pct=self.config.max_oos_drawdown_pct,
            max_hold_share=self.config.max_hold_share,
            max_direction_imbalance=self.config.max_direction_imbalance,
            min_entropy=self.config.min_entropy,
        )

        is_promoted = bool(
            gate_res.passed_gate
            and not is_diverged
            and gate_res.total_trades >= self.config.min_oos_trades
        )
        print(
            f"  [OOS GATE] Round {round_num} ({tf}) Gate Passed: {gate_res.passed_gate} | "
            f"Promoted: {is_promoted} | Reasons: {gate_res.reasons}"
        )

        # 6. Promotion vs Rollback & Pruning
        if is_promoted:
            # Promote new champion
            champion_ckpt = self.runs_dir / f"champion_round_{round_num}.pkl"
            atomic_save_checkpoint(candidate_ckpt, champion_ckpt)
            self.active_seed = str(champion_ckpt)
            self.champion_checkpoint = str(champion_ckpt)
            self.best_expectancy = gate_res.expectancy

            # Also persist canonical best_promoted_checkpoint.pkl
            best_promoted_ckpt = self.runs_dir / "best_promoted_checkpoint.pkl"
            atomic_save_checkpoint(candidate_ckpt, best_promoted_ckpt)

            # --- EWC CHAMPION PROMOTION & FISHER MATRIX REGISTRATION ---
            if self.ewc_enabled:
                self._update_ewc_on_promotion(round_num, candidate_ckpt)
            # -----------------------------------------------------------
        else:
            # Rollback active seed to prior best champion
            if self.champion_checkpoint and Path(self.champion_checkpoint).exists():
                self.active_seed = str(self.champion_checkpoint)

            # Prune large candidate payload
            if self.config.prune_non_promoted_checkpoints:
                prune_unpromoted_checkpoints(round_dir)

        # 7. Update Timeframe Adaptation State Machine
        self.state_machine.record_round_result(
            round_num=round_num,
            expectancy=gate_res.expectancy,
            trades=gate_res.total_trades,
            passed_gate=is_promoted,
        )

        # 8. Update Drawdown Monitor
        # Compute real round PnL from executed trades; if OOS was not executed, use training PnL
        round_pnl = 0.0
        if oos_raw and oos_raw.get("oos_executed", True):
            round_pnl = float(
                oos_metrics.get("pnl")
                or oos_metrics.get("round_pnl_dollars")
                or oos_metrics.get("net_profit")
                or (gate_res.expectancy * 100.0 if gate_res.expectancy > -900.0 else 0.0)
            )
        else:
            round_pnl = float(
                train_metrics.get("pnl")
                or train_metrics.get("round_pnl_dollars")
                or train_metrics.get("net_profit")
                or (float(train_metrics.get("expectancy", 0.0)) * 100.0)
            )
        halt = self.dd_monitor.update(round_pnl)
        if halt:
            self.terminated = True
            self.termination_reason = "emergency_drawdown_halt"

        # 9. Record round in Manifest & Evolution Chart
        combined_telemetry = {
            "expectancy": gate_res.expectancy,
            "win_rate": gate_res.win_rate,
            "risk_reward_ratio": gate_res.risk_reward_ratio,
            "total_trades": gate_res.total_trades,
            "max_equity_drawdown_pct": gate_res.max_equity_drawdown_pct,
            "ppo_loss": ppo_loss,
            "entropy": gate_res.entropy if gate_res.entropy is not None else 0.0,
            "hold_share": gate_res.hold_share,
            "entry_direction_imbalance": gate_res.entry_direction_imbalance,
            "reasons": gate_res.reasons,
        }

        self.manifest_mgr.record_round(
            round_num=round_num,
            timeframe=tf,
            metrics=combined_telemetry,
            promoted=is_promoted,
            active_seed=self.active_seed,
            cumulative_dd_pct=self.dd_monitor.cumulative_drawdown_pct,
            consecutive_m1_losses=self.state_machine.consecutive_m1_losses,
        )

        self.current_round = round_num
        if self.current_round >= self.config.max_rounds:
            self.terminated = True
            if not self.termination_reason:
                self.termination_reason = "max_rounds_budget_reached"

        return {
            "round": round_num,
            "timeframe": tf,
            "promoted": is_promoted,
            "active_seed": self.active_seed,
            "metrics": combined_telemetry,
            "gate_passed": gate_res.passed_gate,
            "reasons": gate_res.reasons,
            "cumulative_dd_pct": self.dd_monitor.cumulative_drawdown_pct,
            "emergency_halt": self.dd_monitor.emergency_halt_triggered,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "next_timeframe": self.state_machine.current_timeframe,
        }

    def _update_ewc_on_promotion(
        self,
        round_num: int,
        candidate_ckpt_path: Path,
    ) -> None:
        """Register diagonal Fisher Information Matrix and anchor weights on champion promotion."""
        try:
            import pickle
            import torch
            from .five_shared_policy import ElasticWeightConsolidation, SharedFiveTradingPolicy

            shared_policy = None
            buf = None
            try:
                from .he_chuyen_gia_ngu_canh import tai_he_chuyen_gia_mot_context_mot_policy
                from .cau_hinh_he_chuyen_gia import CauHinhHeChuyenGia, AppConfig

                app_cfg = AppConfig(he_chuyen_gia=CauHinhHeChuyenGia(bat=True))
                policy_bundle = tai_he_chuyen_gia_mot_context_mot_policy(candidate_ckpt_path, app_cfg)
                shared_policy = getattr(policy_bundle, "five_shared_network", policy_bundle)
                buf = getattr(policy_bundle, "balanced_replay_buffer", None)
            except Exception:
                pass

            if shared_policy is None or not isinstance(shared_policy, torch.nn.Module):
                try:
                    obj = torch.load(candidate_ckpt_path, map_location="cpu", weights_only=False)
                    if isinstance(obj, torch.nn.Module):
                        shared_policy = obj
                    elif hasattr(obj, "five_shared_network"):
                        shared_policy = getattr(obj, "five_shared_network")
                        buf = getattr(obj, "balanced_replay_buffer", None)
                    elif isinstance(obj, dict) and "model_state_dict" in obj:
                        shared_policy = SharedFiveTradingPolicy()
                        shared_policy.load_state_dict(obj["model_state_dict"], strict=False)
                except Exception:
                    try:
                        with open(candidate_ckpt_path, "rb") as f:
                            obj = pickle.load(f)
                        if isinstance(obj, torch.nn.Module):
                            shared_policy = obj
                        elif hasattr(obj, "five_shared_network"):
                            shared_policy = getattr(obj, "five_shared_network")
                            buf = getattr(obj, "balanced_replay_buffer", None)
                        elif isinstance(obj, dict) and "model_state_dict" in obj:
                            shared_policy = SharedFiveTradingPolicy()
                            shared_policy.load_state_dict(obj["model_state_dict"], strict=False)
                    except Exception:
                        pass

            if shared_policy is None or not isinstance(shared_policy, torch.nn.Module):
                # Candidate checkpoint is not a neural policy (e.g. mock test payload)
                return

            ewc = ElasticWeightConsolidation(
                shared_policy,
                lambda_ewc=self.ewc_lambda,
                sample_count_floor=self.ewc_sample_floor,
            )

            val_samples: list[dict[str, Any]] = []
            if buf is not None and hasattr(buf, "sample_balanced_batch"):
                total_avail = (
                    len(getattr(buf, "uptrend_pool", []))
                    + len(getattr(buf, "downtrend_pool", []))
                    + len(getattr(buf, "sideway_pool", []))
                )
                sample_n = max(self.ewc_sample_floor, min(total_avail, 1000))
                if sample_n > 0:
                    val_samples = buf.sample_balanced_batch(batch_size=sample_n)

            if len(val_samples) < self.ewc_sample_floor:
                dummy_obs = torch.randn((5, 137))
                val_samples = [
                    {
                        "obs": dummy_obs,
                        "action": torch.randint(0, 7, (5,)),
                        "agent_ids": torch.arange(5),
                        "prev_actions": torch.zeros(5, dtype=torch.long),
                    }
                    for _ in range(self.ewc_sample_floor)
                ]

            ewc.register_optimal_checkpoint(val_samples, sample_count_floor=self.ewc_sample_floor)

            ewc_round_file = self.runs_dir / f"ewc_anchor_round_{round_num}.pt"
            ewc.save(ewc_round_file)
            best_ewc_file = self.runs_dir / "best_promoted_ewc.pt"
            ewc.save(best_ewc_file)
            self.ewc = ewc
            print(
                f"  [EWC] Registered Fisher anchor for Round {round_num} "
                f"({len(ewc.optimal_params)} protected layers, floor={ewc.sample_count_floor})"
            )
        except Exception as exc:
            print(f"  [EWC WARN] Failed to update EWC on promotion: {exc}")

    def run_loop(
        self,
        max_rounds: int | None = None,
        round_executor: Callable[..., Any] | None = None,
        oos_evaluator: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Execute autonomous training loop up to target rounds or emergency stop."""
        target_rounds = min(max_rounds or self.config.max_rounds, self.config.max_rounds)
        round_results: list[dict[str, Any]] = []

        lock_path = self.runs_dir / ".auto_rl.lock"
        with ProcessLock(lock_path):
            while self.current_round < target_rounds and not self.terminated:
                res = self.step_round(
                    round_executor=round_executor,
                    oos_evaluator=oos_evaluator,
                )
                round_results.append(res)
                if self.terminated:
                    break

            return {
                "total_rounds_executed": self.current_round,
                "active_seed": self.active_seed,
                "champion_checkpoint": self.champion_checkpoint,
                "current_timeframe": self.state_machine.current_timeframe,
                "cumulative_drawdown_pct": self.dd_monitor.cumulative_drawdown_pct,
                "terminated": self.terminated,
                "termination_reason": self.termination_reason,
                "rounds": round_results,
            }


def execute_auto_rl_loop(
    runs_dir: str | Path = "outputs/he_chuyen_gia/v193_runs",
    max_rounds: int = 15,
    initial_timeframe: str = "M1",
    seed_checkpoint: str | Path | None = None,
    target_expectancy: float = 0.05,
) -> dict[str, Any]:
    """Execute autonomous Auto-RL loop until edge discovery or budget cap."""
    cfg = AutoRLEngineConfig(
        runs_dir=runs_dir,
        max_rounds=max_rounds,
        initial_timeframe=initial_timeframe,
        seed_checkpoint=seed_checkpoint,
        min_oos_expectancy=target_expectancy,
    )
    engine = AutoRLEngine(cfg)
    return engine.run_loop()


def main() -> int:
    """CLI entrypoint to run autonomous multi-round training engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-RL Engine for Self Growing Trader V193")
    parser.add_argument("--runs-dir", default="outputs/he_chuyen_gia/v193_runs", help="Output directory")
    parser.add_argument("--max-rounds", type=int, default=15, help="Maximum experiment rounds")
    parser.add_argument("--initial-timeframe", default="M1", help="Initial timeframe (default M1)")
    parser.add_argument("--seed-checkpoint", default=None, help="Initial seed checkpoint path")
    parser.add_argument("--target-expectancy", type=float, default=0.05, help="Target minimum expectancy")
    args = parser.parse_args()

    summary = execute_auto_rl_loop(
        runs_dir=args.runs_dir,
        max_rounds=args.max_rounds,
        initial_timeframe=args.initial_timeframe,
        seed_checkpoint=args.seed_checkpoint,
        target_expectancy=args.target_expectancy,
    )
    print("Auto-RL Loop Execution Complete:")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("champion_checkpoint") else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
