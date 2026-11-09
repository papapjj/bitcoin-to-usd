import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from self_growing_trader.features import add_features
from self_growing_trader.config import FeatureConfig, SymbolConfig
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)

# Load clean 7-day data
df = pd.read_csv(ROOT / "outputs" / "btcusd_m1_7day_clean.csv", sep="\t")
df_renamed = df.rename(columns={
    "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close",
    "<TICKVOL>": "Volume", "<SPREAD>": "Spread"
})
df_renamed["Datetime"] = df["<DATE>"] + " " + df["<TIME>"]
feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())

from self_growing_trader.features import (
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)

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
atrs = feat_df["atr"].values.astype(np.float64)

start_idx = 60
end_idx = N - 30
valid_count = end_idx - start_idx

ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
sample_prices = np.zeros(valid_count, dtype=np.float64)
sample_atrs = np.zeros(valid_count, dtype=np.float64)

for i, t in enumerate(range(start_idx, end_idx)):
    cur_p = closes[t]
    cur_atr = max(atrs[t], 1.0)
    sample_prices[i] = cur_p
    sample_atrs[i] = cur_atr
    
    recent_h_20 = float(np.max(highs[t-20:t]))
    recent_l_20 = float(np.min(lows[t-20:t]))
    recent_h_60 = float(np.max(highs[t-60:t]))
    recent_l_60 = float(np.min(lows[t-60:t]))
    rng_20 = max(1.0, recent_h_20 - recent_l_20)
    rng_60 = max(1.0, recent_h_60 - recent_l_60)
    
    smc_64 = np.zeros(64, dtype=np.float32)
    smc_64[0] = float((recent_h_20 - cur_p) / cur_atr)
    smc_64[1] = float((cur_p - recent_l_20) / cur_atr)
    smc_64[2] = 1.0 if highs[t] >= recent_h_20 else 0.0
    smc_64[3] = 1.0 if lows[t] <= recent_l_20 else 0.0
    smc_64[4] = float((cur_p - recent_l_20) / rng_20 * 2.0 - 1.0)
    smc_64[5] = float(feat_df["lower_wick_atr"].iloc[t])
    smc_64[6] = float(feat_df["upper_wick_atr"].iloc[t])
    smc_64[7] = float((cur_p - recent_l_60) / rng_60)
    smc_64[8] = float((recent_h_60 - cur_p) / cur_atr)
    smc_64[9] = float((cur_p - recent_l_60) / cur_atr)
    
    macro_32 = np.zeros(32, dtype=np.float32)
    macro_32[0] = 0.5
    macro_32[1] = 0.5
    
    ingress_matrix[i] = np.concatenate([base_137[t], smc_64, macro_32], axis=0)

print(f"Prepared {valid_count} samples. Testing checkpoint...")

device = torch.device("cpu")
config = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.83,
    trap_threshold=0.75,
)
engine = BilateralDualCoTEngine(config).to(device)
ckpt_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
if ckpt_path.exists():
    engine.load_checkpoint(ckpt_path)
    print("Loaded champion checkpoint!")
engine.eval()

with torch.no_grad():
    batch_res = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=512)
    actions = batch_res.actions.cpu().numpy()
    prms = batch_res.prm_scores.cpu().numpy()

for be_mult in [0.40, 0.50, 0.60, 0.70, 0.80, 1.00]:
    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0
    last_trade = -999
    trade_list = []
    
    for i in range(valid_count):
        act = int(actions[i])
        if act == 0 or (i - last_trade < 2):
            continue
            
        cur_p = sample_prices[i]
        cur_atr = sample_atrs[i]
        
        # Anti-bottom shorts
        recent_l_60 = np.min(sample_prices[max(0, i-60):i+1])
        recent_h_60 = np.max(sample_prices[max(0, i-60):i+1])
        rng_60 = max(1.0, recent_h_60 - recent_l_60)
        rel_pos = (cur_p - recent_l_60) / rng_60
        if act == -1 and rel_pos <= 0.18:
            continue
            
        last_trade = i
        fwd_c = sample_prices[i+1 : min(valid_count, i+31)]
        if len(fwd_c) < 5:
            continue
            
        be_trig = be_mult * cur_atr
        tp_dist = 2.00 * cur_atr
        sl_dist = 1.00 * cur_atr
        cur_sl = sl_dist
        pnl = 0.0
        
        for k in range(len(fwd_c)):
            if act == 1:
                fav = fwd_c[k] - cur_p
                adv = cur_p - fwd_c[k]
                if fav >= be_trig:
                    cur_sl = -0.10 * cur_atr
                if adv >= cur_sl:
                    pnl = 0.10 * cur_atr if cur_sl < 0 else -1.0 * sl_dist
                    break
                elif fav >= tp_dist:
                    pnl = tp_dist
                    break
            else:
                fav = cur_p - fwd_c[k]
                adv = fwd_c[k] - cur_p
                if fav >= be_trig:
                    cur_sl = -0.10 * cur_atr
                if adv >= cur_sl:
                    pnl = 0.10 * cur_atr if cur_sl < 0 else -1.0 * sl_dist
                    break
                elif fav >= tp_dist:
                    pnl = tp_dist
                    break
                    
        if pnl == 0.0:
            diff = (fwd_c[-1] - cur_p) if act == 1 else (cur_p - fwd_c[-1])
            pnl = diff
            
        if pnl > 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)
            
        trade_list.append((act, pnl))
        
    tot = wins + losses
    wr = (wins / tot * 100) if tot > 0 else 0
    pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0
    print(f"BE Trigger: {be_mult:.2f} ATR | Trades: {tot} (BUY: {sum(1 for a, _ in trade_list if a == 1)}, SELL: {sum(1 for a, _ in trade_list if a == -1)}) | WR: {wr:.2f}% | PF: {pf:.2f}")
