import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd
from research_reasoning_world_class.src.bilateral_dual_cot import (
    extract_canonical_bilateral_ingress_233,
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)
from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)


def run_benchmark():
    csv_path = ROOT / "outputs" / "btcusd_m1_7day_with_warmup.csv"
    print(f"Loading real M1 data from {csv_path}...")
    df = pd.read_csv(csv_path, sep="\t")
    df_renamed = df.rename(columns={
        "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close",
        "<TICKVOL>": "Volume", "<VOL>": "RealVolume", "<SPREAD>": "Spread"
    })
    df_renamed["Datetime"] = df_renamed["<DATE>"] + " " + df_renamed["<TIME>"]

    feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
    N = len(feat_df)

    m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
    slots = np.zeros((N, 40), dtype=np.float32)
    account = np.zeros((N, 8), dtype=np.float32)
    m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
    execution = np.zeros((N, 11), dtype=np.float32)
    timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
    wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
    base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

    closes = feat_df["Close"].values.astype(np.float64)
    highs = feat_df["High"].values.astype(np.float64)
    lows = feat_df["Low"].values.astype(np.float64)
    opens = feat_df["Open"].values.astype(np.float64)
    atrs = feat_df["atr"].values.astype(np.float64)

    # Load Champion
    v3_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion_v3.pt"
    ckpt = v3_pt if v3_pt.exists() else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt")
    print(f"Loading checkpoint {ckpt.name}...")
    engine = BilateralDualCoTEngine(BilateralDualCoTConfig())
    engine.load_checkpoint(ckpt)
    engine.eval()

    lookback = 60
    horizon = 30
    spread = 28.98 # Real Exness spread in USD

    print("Extracting 233D ingress features on real M1 bars...")
    batch_size = 512
    all_ingress = []
    valid_indices = list(range(lookback, N - horizon))

    for t in valid_indices:
        cur_p = closes[t]
        cur_atr = max(atrs[t], 10.0)
        m1_60 = np.column_stack([
            opens[t-lookback:t],
            highs[t-lookback:t],
            lows[t-lookback:t],
            closes[t-lookback:t],
        ])
        ing = extract_canonical_bilateral_ingress_233(base_137[t], m1_60, cur_p, cur_atr)
        all_ingress.append(ing)

    all_ingress_tensor = torch.from_numpy(np.array(all_ingress, dtype=np.float32))

    print("Running batch neural inference...")
    bull_scores = []
    bear_scores = []
    prm_scores = []

    with torch.no_grad():
        for i in range(0, len(all_ingress_tensor), batch_size):
            b_chunk = all_ingress_tensor[i:i+batch_size]
            out = engine.forward_batch(b_chunk)
            bull_scores.extend(out.bull_scores.cpu().numpy().tolist())
            bear_scores.extend(out.bear_scores.cpu().numpy().tolist())
            prm_scores.extend(out.prm_scores.cpu().numpy().tolist())

    bull_scores = np.array(bull_scores)
    bear_scores = np.array(bear_scores)
    prm_scores = np.array(prm_scores)

    print(f"Bull scores: [{bull_scores.min():.3f}, {bull_scores.max():.3f}], Bear scores: [{bear_scores.min():.3f}, {bear_scores.max():.3f}], PRM: [{prm_scores.min():.3f}, {prm_scores.max():.3f}]")

    print("\n" + "=" * 115)
    print("  KIEM TOAN BENCHMARK OOS 7 NGAY (CONCURRENT ANTI-MAE DUAL-CHAMPION V3)")
    print("=" * 115)
    print(f"{'MaxPos':<7} {'Diff':<6} {'PRM':<6} {'BuConf':<7} {'SL':<5} {'BE_T':<5} {'BE_L':<5} {'TP':<5} | {'WR':<7} {'PF':<6} {'Trades':<8} {'Tr/d':<7} {'B_W/L':<11} {'S_W/L':<11} {'MaxDD':<7} {'Net PnL ($)':<12}")
    print("-" * 115)

    for max_pos in [1, 3, 5]:
        for diff_th in [0.06, 0.08, 0.10]:
            for prm_th in [0.28, 0.30, 0.35]:
                for sl_atr, be_trig, be_lock, tp_atr in [
                    (2.60, 1.60, 0.80, 3.60),
                    (2.80, 1.80, 0.80, 3.80),
                ]:
                    open_pos = []
                    closed_trades = []
                    last_entry_t = -999
                    peak_capital = 10000.0
                    cur_capital = 10000.0
                    max_dd = 0.0

                    b_wins, b_losses = 0, 0
                    s_wins, s_losses = 0, 0

                    for idx, t in enumerate(valid_indices):
                        cur_atr = max(atrs[t], 10.0)
                        cur_h = highs[t]
                        cur_l = lows[t]
                        cur_c = closes[t]

                        # Check exits
                        remaining = []
                        for pos in open_pos:
                            p_type = pos["type"]
                            entry_p = pos["entry_p"]
                            sl_p = pos["sl_p"]
                            be_p = pos["be_p"]
                            be_lock_p = pos["be_lock_p"]
                            tp_p = pos["tp_p"]
                            be_active = pos["be_active"]

                            closed = False
                            pnl = 0.0

                            if p_type == "BUY":
                                # Conservative execution: if low hits SL, close at SL
                                if cur_l <= sl_p:
                                    pnl = (sl_p - entry_p) * 0.01
                                    closed_trades.append({"pnl": pnl, "type": "BUY", "win": pnl > 0})
                                    closed = True
                                    if pnl > 0: b_wins += 1
                                    else: b_losses += 1
                                elif cur_h >= tp_p:
                                    pnl = (tp_p - entry_p) * 0.01
                                    closed_trades.append({"pnl": pnl, "type": "BUY", "win": True})
                                    closed = True
                                    b_wins += 1
                                elif cur_h >= be_p and not be_active:
                                    pos["be_active"] = True
                                    pos["sl_p"] = be_lock_p
                            else:
                                if cur_h >= sl_p:
                                    pnl = (entry_p - sl_p) * 0.01
                                    closed_trades.append({"pnl": pnl, "type": "SELL", "win": pnl > 0})
                                    closed = True
                                    if pnl > 0: s_wins += 1
                                    else: s_losses += 1
                                elif cur_l <= tp_p:
                                    pnl = (entry_p - tp_p) * 0.01
                                    closed_trades.append({"pnl": pnl, "type": "SELL", "win": True})
                                    closed = True
                                    s_wins += 1
                                elif cur_l <= be_p and not be_active:
                                    pos["be_active"] = True
                                    pos["sl_p"] = be_lock_p

                            if closed:
                                cur_capital += pnl
                                if cur_capital > peak_capital:
                                    peak_capital = cur_capital
                                dd = (peak_capital - cur_capital) / peak_capital
                                if dd > max_dd:
                                    max_dd = dd
                            else:
                                if t - pos["entry_t"] >= 30:
                                    exit_p = cur_c + spread if p_type == "SELL" else cur_c - spread
                                    pnl = ((exit_p - entry_p) if p_type == "BUY" else (entry_p - exit_p)) * 0.01
                                    closed_trades.append({"pnl": pnl, "type": p_type, "win": pnl > 0})
                                    if p_type == "BUY":
                                        if pnl > 0: b_wins += 1
                                        else: b_losses += 1
                                    else:
                                        if pnl > 0: s_wins += 1
                                        else: s_losses += 1
                                    cur_capital += pnl
                                    if cur_capital > peak_capital:
                                        peak_capital = cur_capital
                                    dd = (peak_capital - cur_capital) / peak_capital
                                    if dd > max_dd:
                                        max_dd = dd
                                else:
                                    remaining.append(pos)
                        open_pos = remaining

                        # Check entry
                        if len(open_pos) < max_pos and (t - last_entry_t >= 2):
                            bu = bull_scores[idx]
                            be = bear_scores[idx]
                            pr = prm_scores[idx]

                            is_buy = (bu - be >= diff_th) and (pr >= prm_th) and (bu >= 0.35)
                            is_sell = (be - bu >= diff_th * 0.7) and (pr >= prm_th) and (be >= 0.28)

                            if is_buy and not is_sell:
                                entry_p = cur_c + spread
                                sl_p = entry_p - sl_atr * cur_atr
                                be_p = entry_p + be_trig * cur_atr
                                be_lock_p = entry_p + be_lock * cur_atr
                                tp_p = entry_p + tp_atr * cur_atr
                                open_pos.append({
                                    "type": "BUY", "entry_t": t, "entry_p": entry_p,
                                    "sl_p": sl_p, "be_p": be_p, "be_lock_p": be_lock_p,
                                    "tp_p": tp_p, "be_active": False
                                })
                                last_entry_t = t
                            elif is_sell and not is_buy:
                                entry_p = cur_c - spread
                                sl_p = entry_p + sl_atr * cur_atr
                                be_p = entry_p - be_trig * cur_atr
                                be_lock_p = entry_p - be_lock * cur_atr
                                tp_p = entry_p - tp_atr * cur_atr
                                open_pos.append({
                                    "type": "SELL", "entry_t": t, "entry_p": entry_p,
                                    "sl_p": sl_p, "be_p": be_p, "be_lock_p": be_lock_p,
                                    "tp_p": tp_p, "be_active": False
                                })
                                last_entry_t = t

                    if len(closed_trades) > 0:
                        wins_pnl = [tr["pnl"] for tr in closed_trades if tr["win"]]
                        loss_pnl = [abs(tr["pnl"]) for tr in closed_trades if not tr["win"]]
                        wr = len(wins_pnl) / len(closed_trades)
                        pf = sum(wins_pnl) / sum(loss_pnl) if sum(loss_pnl) > 0 else 999.0
                        net_pnl = sum(wins_pnl) - sum(loss_pnl)
                        t_day = len(closed_trades) / 7.0

                        if True:
                            b_str = f"{b_wins}/{b_losses}"
                            s_str = f"{s_wins}/{s_losses}"
                            print(f"{max_pos:<7} {diff_th:<6.2f} {prm_th:<6.2f} {0.35:<7.2f} {sl_atr:<5.1f} {be_trig:<5.1f} {be_lock:<5.1f} {tp_atr:<5.1f} | {wr*100:>5.1f}%  {pf:>5.2f}  {len(closed_trades):<8} {t_day:<7.1f} {b_str:<11} {s_str:<11} {max_dd*100:>5.1f}%  $+{-net_pnl if net_pnl < 0 else net_pnl:,.0f}")


if __name__ == "__main__":
    run_benchmark()
