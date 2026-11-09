import pandas as pd
import numpy as np

path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df = pd.read_csv(path, sep='\t')
print("Columns:", df.columns.tolist())
print(f"Total rows: {len(df)}")
if 'deal_type' in df.columns or 'type' in df.columns:
    print(df.head(20))
else:
    for i in range(min(20, len(df))):
        print(df.iloc[i].to_dict())
