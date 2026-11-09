import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("outputs/checkpoints/phase3_and_pullback_limit_results.json") as f:
    d = json.load(f)

m = d["models"]["tca_phase3_30epochs"]["market_order"]
l = d["models"]["tca_phase3_30epochs"]["pullback_limit_order"]

print("======================================================================")
print("PHÂN TÍCH DRAWDOWN THEO QUY MÔ VỐN VÀ KHỐI LƯỢNG LOT (POSITION SIZING)")
print("======================================================================")
print(f"Kỷ Lục Gốc (Fixed 1.0 BTC trên tài khoản $50,000 USD):")
print(f"  Market: Net PnL = +${m['net_pnl_usd']:.2f} | Max DD = ${m['max_drawdown_usd']:.2f} ({m['max_drawdown_pct']}%)")
print(f"  Limit : Net PnL = +${l['net_pnl_usd']:.2f} | Max DD = ${l['max_drawdown_usd']:.2f} ({l['max_drawdown_pct']}%)")

for lot in [0.20, 0.10, 0.05]:
    net_mkt = m['net_pnl_usd'] * lot
    dd_mkt = m['max_drawdown_usd'] * lot
    dd_pct_mkt = (dd_mkt / 50000.0) * 100.0
    ret_pct_mkt = (net_mkt / 50000.0) * 100.0

    net_lim = l['net_pnl_usd'] * lot
    dd_lim = l['max_drawdown_usd'] * lot
    dd_pct_lim = (dd_lim / 50000.0) * 100.0
    ret_pct_lim = (net_lim / 50000.0) * 100.0

    print(f"\nKhối lượng {lot:.2f} BTC (Lot thực chiến) trên tài khoản $50,000 USD:")
    print(f"  [Market Order]       Lãi Ròng: +${net_mkt:>8.2f} (+{ret_pct_mkt:>5.1f}%) | Max DD: ${dd_mkt:>7.2f} ({dd_pct_mkt:>5.2f}%)")
    print(f"  [Pullback Limit 50%] Lãi Ròng: +${net_lim:>8.2f} (+{ret_pct_lim:>5.1f}%) | Max DD: ${dd_lim:>7.2f} ({dd_pct_lim:>5.2f}%)")
