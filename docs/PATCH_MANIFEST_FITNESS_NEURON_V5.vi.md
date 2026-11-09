# Patch manifest — P0 Fitness MT5 & Neuron Attribution v5

## Files changed

| File | Nội dung |
|---|---|
| `src/self_growing_trader/mt5_runner.py` | Parser execution ledger v5; lifecycle/R metrics; eligibility/rank/learning split; score bounded; evaluator backfill trace-to-lifecycle; journal persists module action traces. |
| `src/self_growing_trader/self_growing_runner.py` | Signed action-margin neuron attribution; trace cache; final lifecycle credit; evidence-based freeze/prune; relative global learning signal; world-core API alignment. |
| `SelfGrowing_MT5_Trainer_EA.mq5` | Execution ledger schema 5 fields `position_final_close`, `initial_risk`, `age_bars`; static forward declarations for lifecycle helpers. |
| `tests/test_mt5_runner.py` | Lifecycle, partial-close, no-ledger, immediate-exit regression fixtures. |
| `tests/test_self_growing_runner.py` | Neuron final-credit/idempotency/pruning regression checks and v5 promotion fixtures. |
| `docs/FITNESS_V5_NEURON_ATTRIBUTION_CHECKLIST.vi.md` | Completion checklist and MT5 validation matrix. |

## Deliberate semantic changes

- `mt5_fitness()` returns a **bounded rank score** in `[-100, 100]`; raw historical scores from older schemas are not comparable.
- Runtime MT5 evaluator calls fitness with `require_execution_ledger=True`; a report without finalised lifecycle ledger is a hard reject (`-100`).
- Position sample count means **finalised position lifecycle**, not MT5 raw `Total Trades` and not partial-close deal count.
- Global trial fitness can shape module-level structural plasticity but cannot directly become `neuron_utility`.
- Neuron fitness is linked to actual action margin and final realised outcome via `trace_id`.

## Required post-merge commands

```bash
python -m pip install -e '.[test]'
pytest -q
python -m compileall -q src tests scripts
python scripts/preflight_mt5.py --runner-dir mt5_v2_runner
```

The final command remains expected to fail until the missing portable runner and `compute_features.mqh` are supplied.
