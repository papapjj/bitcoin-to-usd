import pandas as pd
import json

# Check the exact parameters of Iteration 7, 8, 9 in live runs
for it in [7, 8, 9]:
    with open(f"outputs/acceptance_artifacts/milestone_3/iteration_{it}/manifest.json") as f:
        mf = json.load(f)
    print(f"Manifest Iteration {it} parameters:", mf.get("parameters"))

# Check runner script settings or differences between runs
# In Iteration 8: policy_id was 19308, in Iteration 9: 19309, in Iteration 7: 19307
# Look at mt5_mythos_run files
for pid in [19307, 19308, 19309]:
    with open(f"outputs/mt5_mythos_run_{pid}.json") as f:
        run = json.load(f)
    print(f"\nRun {pid}:")
    print(f"  Packets served: {run.get('packets_served')}")
    print(f"  Traps vetoed:   {run.get('traps_vetoed')}")
    print(f"  Non-vetoed:     {run.get('packets_served') - run.get('traps_vetoed')}")
    print(f"  Closed deals:   {run.get('closed_deals')}")
    print(f"  Conversion:     {run.get('closed_deals') / (run.get('packets_served') - run.get('traps_vetoed')):.1%}")
    print(f"  WR:             {run.get('win_rate'):.2f}%")
    print(f"  PF:             {run.get('profit_factor'):.2f}")
    print(f"  Gross Profit:   ${run.get('gross_profit'):.2f}")
    print(f"  Gross Loss:     ${run.get('gross_loss'):.2f}")
    print(f"  Avg Win:        ${run.get('gross_profit') / run.get('wins'):.2f}")
    print(f"  Avg Loss:       ${run.get('gross_loss') / run.get('losses'):.2f}")
    print(f"  Payoff:         {(run.get('gross_profit') / run.get('wins')) / (run.get('gross_loss') / run.get('losses')):.2f}")
