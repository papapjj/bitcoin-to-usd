import numpy as np

spread = 28.98 # BTCUSDm spread in USD
atrs = [35.0, 45.0, 60.0]

print(f"{'ATR':<6} {'SL_mult':<8} {'TP_mult':<8} {'Nominal SL':<12} {'Nominal TP':<12} {'Effective SL':<14} {'Effective TP':<14} {'Effective RR':<12} {'Breakeven WR'}")
print("-" * 105)

for atr in atrs:
    for sl_m, tp_m in [(1.5, 1.5), (2.0, 2.0), (2.5, 2.5), (3.5, 3.5), (4.0, 6.0), (5.0, 10.0)]:
        nom_sl = sl_m * atr
        nom_tp = tp_m * atr
        
        # In BUY: Entry at Ask = p + spread/2. SL is at p - nom_sl (triggered by Bid).
        # Distance to SL from initial Bid: nom_sl - spread
        eff_sl = max(1.0, nom_sl - spread)
        # Distance to TP for Bid: nom_tp + spread
        eff_tp = nom_tp + spread
        
        eff_rr = eff_tp / eff_sl
        # Breakeven win rate: WR * eff_tp - (1 - WR) * eff_sl = 0 => WR = eff_sl / (eff_sl + eff_tp)
        # But wait! If eff_sl is small, price hits eff_sl MUCH more often!
        # Under random walk: P(hit eff_tp before eff_sl) = eff_sl / (eff_sl + eff_tp)
        p_win_random = eff_sl / (eff_sl + eff_tp)
        
        print(f"{atr:<6.1f} {sl_m:<8.1f} {tp_m:<8.1f} ${nom_sl:<11.2f} ${nom_tp:<11.2f} ${eff_sl:<13.2f} ${eff_tp:<13.2f} {eff_rr:<12.2f} {p_win_random*100:.1f}%")
