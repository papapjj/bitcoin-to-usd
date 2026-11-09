"""Run a fresh 30-day TimingNet MT5 pass with bounded learning caches.

The full-fidelity trace objects are useful for short audits but are too large
to retain without a cap over a multi-day MT5 run.  This launcher keeps the
TimingNet/lifecycle updates enabled while bounding only replay and trace
retention; MT5 bridge and execution ledgers remain full-fidelity on disk.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from self_growing_trader.chuyen_gia_policy import huan_luyen_he_chuyen_gia_mt5
from self_growing_trader.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs" / "he_chuyen_gia_mot_context_mot_policy.yaml")
    learning = replace(
        config.learning,
        # Keep recent replay/credit evidence, but prevent multi-day trace
        # retention from growing without bound.
        experience_buffer_max=4,
        rejection_memory_max=32,
        execution_trace_cache_max=2,
        memory_neurons_max=256,
        memory_connections_max=512,
        mistake_episodes_max=32,
    )
    # The causal archive stores immutable StepTrace objects separately from
    # the replay cache. Keep a small lifecycle window for online credit while
    # preventing one lazy specialist from retaining thousands of graph traces.
    causal = replace(config.causal, max_decisions=128, max_reports=128)
    observability = replace(config.observability, enabled=False)
    config = replace(config, learning=learning, observability=observability, causal=causal)
    result = huan_luyen_he_chuyen_gia_mt5(
        config,
        output_path=ROOT / "outputs" / "he_chuyen_gia" / "candidates" / "timingnet_from_scratch_30d_20260719_v123.pkl",
        runner_dir=ROOT / "mt5_v2_oos_arm_train_v123_causal128_compact",
        timeout_sec=300,
        max_wall_sec=86400,
        visual=False,
        shutdown_terminal=False,
        from_date="2025.05.15",
        to_date="2025.06.14",
        backtest_days=30,
        model=1,
        seed=20260719,
        resume_existing=False,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
