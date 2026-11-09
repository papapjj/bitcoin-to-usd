import pandas as pd
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df['Time'] = df['<DATE>'] + ' ' + df['<TIME>']

times = [
    ('2025.02.20 05:50:00', 'SELL'),
    ('2025.02.20 10:03:00', 'BUY'),
    ('2025.02.20 12:56:00', 'BUY'),
    ('2025.02.20 15:41:00', 'SELL'),
    ('2025.02.20 18:48:00', 'BUY'),
    ('2025.02.21 01:44:00', 'SELL'),
    ('2025.02.21 06:03:00', 'BUY'),
]

for t, act in times:
    b = df[df['Time'] == t]
    if not b.empty:
        idx = b.index[0]
        c = df.loc[idx, '<CLOSE>']
        fwd30 = df.loc[idx:idx+30]
        fwd60 = df.loc[idx:idx+60]
        min30, max30 = fwd30['<LOW>'].min(), fwd30['<HIGH>'].max()
        min60, max60 = fwd60['<LOW>'].min(), fwd60['<HIGH>'].max()
        print(f"{t} | {act:<4} | Entry: {c:<8.1f} | 30m Low/High: {min30:8.1f} / {max30:8.1f} | 60m Low/High: {min60:8.1f} / {max60:8.1f}")
