#!/usr/bin/env python3
"""CLI Entry Point for Walk-Forward Multi-Cycle Orchestration Engine.

Usage Examples:
    # Run full 3-cycle walk-forward pipeline (Cycle 1 -> Cycle 2 -> Cycle 3)
    python scripts/run_walk_forward_multi_cycle.py --all-cycles

    # Run Cycle 1 only from seed
    python scripts/run_walk_forward_multi_cycle.py --cycle 1 --seed-checkpoint outputs/he_chuyen_gia/v193_runs/seed_sgt_compact_tier2.pkl

    # Run Cycle 2 inheriting Cycle 1 promoted checkpoint
    python scripts/run_walk_forward_multi_cycle.py --cycle 2 --seed-checkpoint outputs/walk_forward_multi_cycle/checkpoint_cycle_1.pkl

    # Dry-run validation (verifying pipeline without MT5 terminal)
    python scripts/run_walk_forward_multi_cycle.py --all-cycles --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

# Ensure repository root and src/ are in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.walk_forward_engine import (
    DEFAULT_BASE_PROFILE,
    DEFAULT_CYCLES,
    DEFAULT_INITIAL_SEED,
    DEFAULT_MAX_ADAPTIVE_RETRIES,
    DEFAULT_MAX_WALL_SEC,
    DEFAULT_RUNNER_SOURCE,
    DEFAULT_SOURCE_CONFIG,
    DEFAULT_TIMEOUT_SEC,
    WalkForwardEngine,
    build_parser,
    evaluate_acceptance_criteria,
    extract_cycle_metrics,
    generate_walk_forward_manifest,
    sha256_file,
)



def create_dry_run_runner(output_base_dir: Path) -> callable:
    """Create a simulated runner callable for dry-run verification."""
    def _dry_run(
        from_date: str,
        to_date: str,
        seed_artifact: Path,
        output_dir: Path,
        profile_path: Path,
        config_path: Path,
        timeout_sec: int,
        max_wall_sec: int,
    ) -> bool:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create simulated candidate checkpoint
        cand_path = output_dir / "sgt_compact_tier2_candidate.pkl"
        if seed_artifact.is_file():
            shutil.copy2(seed_artifact, cand_path)
        else:
            cand_path.write_bytes(b"SGT_COMPACT_SIMULATED_CHECKPOINT_DATA")

        # Create simulated mt5_diagnostics.json that passes all acceptance criteria
        simulated_diag = {
            "schema_version": 1,
            "performance": {
                "net_pnl": 1250.50,
                "profit_factor": 1.45,
                "closed_trades": 42,
                "win_rate": 0.58,
                "payoff_ratio": 1.85,
                "expectancy_r": 0.12,
                "close_reasons": {"TAKE_PROFIT": 25, "STOP_LOSS": 17},
            },
            "risk": {
                "max_equity_drawdown_pct": 8.4,
            },
            "policy": {
                "hold_share": 0.82,
                "normalized_action_entropy": 0.65,
            },
        }
        (output_dir / "mt5_diagnostics.json").write_text(json.dumps(simulated_diag, indent=2), encoding="utf-8")

        # Create simulated events.jsonl with zero simplex/NaN errors
        with open(output_dir / "events.jsonl", "w", encoding="utf-8") as f:
            for i in range(42):
                f.write(json.dumps({
                    "event_type": "lifecycle",
                    "step": i * 100,
                    "payload": {
                        "realized_net_r": 0.25 if i % 2 == 0 else -0.15,
                        "close_reason": "TAKE_PROFIT" if i % 2 == 0 else "STOP_LOSS",
                    },
                }) + "\n")

        print(f"[DRY-RUN] Simulated run complete for {from_date} -> {to_date} in {output_dir.name}")
        return True

    return _dry_run


def package_final_chain_artifacts(
    output_base_dir: Path,
    initial_seed: Path,
) -> dict[str, Any]:
    """Package final candidate checkpoint, imitation sidecars, and authoritative manifest.json."""
    output_base_dir = output_base_dir.resolve()
    c3_path = output_base_dir / "checkpoint_cycle_3.pkl"
    if not c3_path.is_file():
        raise FileNotFoundError(f"Cannot package final: {c3_path} does not exist.")

    final_checkpoint_path = output_base_dir / "final_promoted_checkpoint.pkl"
    shutil.copy2(c3_path, final_checkpoint_path)

    # Copy sidecars
    for sidecar in c3_path.parent.glob(f"{c3_path.stem}.*"):
        if sidecar != c3_path:
            sidecar_ext = sidecar.name[len(c3_path.stem):]
            shutil.copy2(sidecar, output_base_dir / f"final_promoted_checkpoint{sidecar_ext}")

    final_sidecar = final_checkpoint_path.with_suffix(final_checkpoint_path.suffix + ".imitation.json")
    if not final_sidecar.is_file():
        seed_sidecar = c3_path.with_suffix(c3_path.suffix + ".imitation.json")
        if seed_sidecar.is_file():
            try:
                sidecar_data = json.loads(seed_sidecar.read_text(encoding="utf-8"))
                final_sha = sha256_file(final_checkpoint_path)
                sidecar_data["checkpoint_sha256"] = final_sha
                sidecar_data["candidate_checkpoint_sha256"] = final_sha
                sidecar_data["reason"] = "final_promoted_checkpoint"
                final_sidecar.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
            except Exception:
                pass

    # Collect cycle results from cycle folders
    cycle_results: dict[int, dict[str, Any]] = {}
    for cid in (1, 2, 3):
        candidate_dirs = sorted(output_base_dir.glob(f"cycle_{cid}*"))
        if candidate_dirs:
            latest_run_dir = candidate_dirs[-1]
            metrics = extract_cycle_metrics(latest_run_dir, cycle_id=cid)
            verdict = evaluate_acceptance_criteria(metrics)
            ckpt_path = output_base_dir / f"checkpoint_cycle_{cid}.pkl"
            cycle_results[cid] = {
                "cycle_id": cid,
                "name": DEFAULT_CYCLES[cid].name,
                "from_date": DEFAULT_CYCLES[cid].from_date,
                "to_date": DEFAULT_CYCLES[cid].to_date,
                "promoted": ckpt_path.is_file() and verdict.passed,
                "metrics": metrics.to_dict(),
                "verdict": verdict.to_dict(),
                "promoted_checkpoint": str(ckpt_path) if ckpt_path.is_file() else None,
                "promoted_checkpoint_sha256": sha256_file(ckpt_path) if ckpt_path.is_file() else "",
            }

    manifest = generate_walk_forward_manifest(
        candidate_checkpoint_path=final_checkpoint_path,
        cycle_results=cycle_results,
        output_dir=output_base_dir,
        initial_seed_path=initial_seed,
        target_cycles=[1, 2, 3],
    )
    print(f"\n[PACKAGING] Final Promoted Checkpoint: {final_checkpoint_path}")
    print(f"[PACKAGING] Authoritative Manifest:   {output_base_dir / 'manifest.json'}")
    print(f"[PACKAGING] Multi-Cycle Certified:    {manifest.get('multi_cycle_certified')}")
    return manifest


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Enforce pure headless defaults
    if args.timeout_sec < 3600:
        print(f"[ERROR] --timeout-sec must be >= 3600 for reliable cycle completion (got {args.timeout_sec})", file=sys.stderr)
        return 1
    if args.max_wall_sec != 0:
        print(f"[ERROR] --max-wall-sec must be 0 to avoid premature timeouts (got {args.max_wall_sec})", file=sys.stderr)
        return 1

    if getattr(args, "package_only", False):
        manifest = package_final_chain_artifacts(
            output_base_dir=args.output_dir,
            initial_seed=args.seed_checkpoint,
        )
        return 0 if manifest.get("multi_cycle_certified", False) else 1

    runner_fn = create_dry_run_runner(args.output_dir) if args.dry_run else None

    engine = WalkForwardEngine(
        workspace_dir=ROOT,
        config_path=args.config,
        base_profile_path=args.profile,
        runner_source=args.runner_source,
        timeout_sec=args.timeout_sec,
        max_wall_sec=args.max_wall_sec,
        max_adaptive_retries=args.max_adaptive_retries,
        runner_fn=runner_fn,
    )

    if args.all_cycles:
        # Run all 3 cycles sequentially
        manifest = engine.run_all_cycles(
            initial_seed=args.seed_checkpoint,
            output_base_dir=args.output_dir,
        )
        if manifest.get("multi_cycle_certified", False):
            print("\n[SUCCESS] Full Walk-Forward Multi-Cycle Chain Certified 100% Passed!")
            return 0
        else:
            print("\n[FAILURE] Walk-Forward Multi-Cycle Chain failed one or more criteria.", file=sys.stderr)
            return 1
    else:
        # Run single cycle
        seed_path = args.seed_checkpoint
        if args.cycle > 1:
            prev_champ = args.output_dir / f"checkpoint_cycle_{args.cycle - 1}.pkl"
            explicit_seed = any(arg == "--seed-checkpoint" or arg.startswith("--seed-checkpoint=") for arg in sys.argv)
            if not explicit_seed and prev_champ.is_file():
                seed_path = prev_champ
                print(f"[INFO] Auto-inheriting previous cycle champion: {seed_path}")
            elif not seed_path.is_file() and prev_champ.is_file():
                seed_path = prev_champ
                print(f"[INFO] Auto-inheriting previous cycle champion: {seed_path}")

        res = engine.run_cycle(
            cycle_id=args.cycle,
            seed_checkpoint=seed_path,
            output_base_dir=args.output_dir,
            start_retry=args.start_retry,
            initial_profile=args.initial_profile,
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
        )
        if res.get("promoted", False):
            print(f"\n[SUCCESS] Cycle {args.cycle} Promoted Successfully: {res.get('promoted_checkpoint')}")
            if args.cycle == 3:
                package_final_chain_artifacts(
                    output_base_dir=args.output_dir,
                    initial_seed=args.seed_checkpoint,
                )
            return 0
        else:
            print(f"\n[FAILURE] Cycle {args.cycle} failed promotion criteria.", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
