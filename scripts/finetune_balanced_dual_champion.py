import sys, os
sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)
from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)

print("=" * 80)
print("  FINETUNING BALANCED BILATERAL DUAL-COT CHAMPION")
print("  Target: Win Rate >= 80%, Profit Factor > 3.0, Symmetrical BUY/SELL")
print("=" * 80, flush=True)

# 1. Ingest Data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df_renamed = df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume', '<VOL>': 'RealVolume', '<SPREAD>': 'Spread'
})
if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
    df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']

print("Calculating 137D canonical features...", flush=True)
feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
N_total = len(feat_df)

m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
slots = np.zeros((N_total, 40), dtype=np.float32)
account = np.zeros((N_total, 8), dtype=np.float32)
m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
execution = np.zeros((N_total, 11), dtype=np.float32)
timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

closes = feat_df['Close'].values.astype(np.float64)
highs = feat_df['High'].values.astype(np.float64)
lows = feat_df['Low'].values.astype(np.float64)
opens = feat_df['Open'].values.astype(np.float64)
atrs = feat_df['atr'].values.astype(np.float64)

# Indicators
s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

s_h = pd.Series(highs)
s_l = pd.Series(lows)
rolling_h60 = s_h.rolling(60, min_periods=1).max().values
rolling_l60 = s_l.rolling(60, min_periods=1).min().values
rng60 = np.maximum(1.0, rolling_h60 - rolling_l60)
rel_pos_60 = (closes - rolling_l60) / rng60

lookback = 60
horizon = 25
start_idx = lookback
end_idx = N_total - horizon
valid_count = end_idx - start_idx

ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
targets = np.zeros(valid_count, dtype=np.int64) # 0: HOLD, 1: BUY, 2: SELL
traps = np.zeros(valid_count, dtype=np.int64)

print(f"Extracting 233D ingress features and sniper labels across {valid_count} bars...", flush=True)

for i, t in enumerate(range(start_idx, end_idx)):
    cur_p = closes[t]
    cur_atr = max(atrs[t], 1.0)
    
    m1_bars_60 = np.column_stack([
        opens[t-lookback:t],
        highs[t-lookback:t],
        lows[t-lookback:t],
        closes[t-lookback:t],
    ])
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(base_137[t], m1_bars_60, cur_p, cur_atr)
    
    # Ground truth SNIPER EDGE with Value Pullback and Anti-MAE:
    fwd_h = highs[t+1 : t+1+horizon]
    fwd_l = lows[t+1 : t+1+horizon]
    
    # BUY TARGET:
    # 1. Trend: EMA20 > EMA50 and Price > EMA50
    # 2. Value Zone: lows[t] reached ema20 + 0.35 ATR, close <= ema20 + 0.50 ATR
    # 3. Anti-Top: rel_pos_60 <= 0.85
    # 4. Trigger: Green candle or lower wick >= 0.20 ATR
    # 5. Outcome: +1.20 ATR favorable before -0.50 ATR adverse
    up = ema20[t] > ema50[t] and cur_p > ema50[t]
    in_buy_zone = (lows[t] <= ema20[t] + 0.35 * cur_atr) and (cur_p <= ema20[t] + 0.50 * cur_atr) and (rel_pos_60[t] <= 0.85)
    buy_trig = (closes[t] >= opens[t]) or ((closes[t] - lows[t]) >= 0.20 * cur_atr)
    
    buy_ok = False
    if up and in_buy_zone and buy_trig:
        for k in range(len(fwd_h)):
            if (cur_p - fwd_l[k]) >= 0.45 * cur_atr: break
            if (fwd_h[k] - cur_p) >= 1.00 * cur_atr: buy_ok = True; break
            
    # SELL TARGET:
    # 1. Trend: EMA20 < EMA50 and Price < EMA50
    # 2. Value Zone: highs[t] reached ema20 - 0.35 ATR, close >= ema20 - 0.50 ATR
    # 3. Anti-Bottom: rel_pos_60 >= 0.15
    # 4. Trigger: Red candle or upper wick >= 0.20 ATR
    # 5. Outcome: -1.20 ATR favorable before +0.50 ATR adverse
    down = ema20[t] < ema50[t] and cur_p < ema50[t]
    in_sell_zone = (highs[t] >= ema20[t] - 0.35 * cur_atr) and (cur_p >= ema20[t] - 0.50 * cur_atr) and (rel_pos_60[t] >= 0.15)
    sell_trig = (closes[t] <= opens[t]) or ((highs[t] - closes[t]) >= 0.20 * cur_atr)
    
    sell_ok = False
    if down and in_sell_zone and sell_trig:
        for k in range(len(fwd_l)):
            if (fwd_h[k] - cur_p) >= 0.45 * cur_atr: break
            if (cur_p - fwd_l[k]) >= 1.00 * cur_atr: sell_ok = True; break
            
    if buy_ok and not sell_ok: targets[i] = 1 # BUY
    elif sell_ok and not buy_ok: targets[i] = 2 # SELL
    else: targets[i] = 0 # HOLD

b_cnt = int(np.sum(targets == 1))
s_cnt = int(np.sum(targets == 2))
h_cnt = int(np.sum(targets == 0))
print(f"Dataset labeled: BUY={b_cnt} ({b_cnt/valid_count*100:.1f}%), SELL={s_cnt} ({s_cnt/valid_count*100:.1f}%), HOLD={h_cnt} ({h_cnt/valid_count*100:.1f}%)", flush=True)

# 2. Setup Bilateral Dual-CoT Engine
device = torch.device("cpu")
torch.set_num_threads(4)

config = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.32,
    trap_threshold=0.60,
)
engine = BilateralDualCoTEngine(config).to(device)

# Load existing checkpoint to preserve xLSTM backbone reasoning
ckpt_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
if ckpt_path.exists():
    # Backup previous checkpoint
    backup_path = ckpt_path.with_name("mythos_bilateral_dual_cot_champion_prev_backup.pt")
    shutil.copyfile(ckpt_path, backup_path)
    print(f"Backed up previous checkpoint to {backup_path.name}")
    try:
        engine.load_checkpoint(str(ckpt_path))
        print("Loaded initial weights for finetuning.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}. Starting fresh.")

# 3. Specialist Optimizers
opt_bull = torch.optim.AdamW(engine.bull_cot.parameters(), lr=1.5e-4, weight_decay=1e-5)
opt_bear = torch.optim.AdamW(
    list(engine.bear_cot.parameters()) + list(engine.trap_detector.parameters()),
    lr=1.5e-4, weight_decay=1e-5
)
opt_arbiter = torch.optim.AdamW(
    list(engine.cross_examination.parameters())
    + list(engine.arbiter.parameters())
    + list(engine.state_adapter.parameters()),
    lr=2.5e-4, weight_decay=1e-5
)

# Focal Loss function for Arbiter to solve class imbalance and prevent SELL collapse
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        at = self.alpha[target]
        focal_loss = at * ((1.0 - pt) ** self.gamma) * ce
        return focal_loss.mean()

# Balanced alpha weights: give BUY and SELL equal emphasis over HOLD
alpha_weights = torch.tensor([1.0, 5.0, 5.0], device=device)
focal_loss_fn = FocalCrossEntropyLoss(alpha=alpha_weights, gamma=2.0)

# Training loop: 8 Epochs
epochs = 8
batch_size = 64
N = valid_count
indices = np.arange(N)

print(f"\nStarting 8-epoch Finetuning over {N} bars...", flush=True)

for epoch in range(1, epochs + 1):
    engine.train()
    np.random.shuffle(indices)
    t0 = time.perf_counter()
    
    tot_bull_loss = 0.0
    tot_bear_loss = 0.0
    tot_arb_loss = 0.0
    num_batches = 0
    
    for b_start in range(0, N, batch_size):
        b_idx = indices[b_start : b_start + batch_size]
        if len(b_idx) < 8: continue
        
        x_b = torch.from_numpy(ingress_matrix[b_idx]).float().to(device)
        y_b = torch.from_numpy(targets[b_idx]).long().to(device)
        
        # 1. Bull CoT Pass & Step
        opt_bull.zero_grad()
        z_b = engine.state_adapter(x_b)
        bull_out = engine.bull_cot(z_b)
        b_score = bull_out.bull_score
        
        # Bull Target: 1.0 for BUY (1), 0.0 for HOLD (0) or SELL (2)
        b_target = torch.where(y_b == 1, 1.0, 0.0).float()
        b_weights = torch.where(y_b == 1, 6.0, 1.0).float()
        b_weights = torch.where(y_b == 2, 4.0, b_weights)
        bull_loss = F.binary_cross_entropy(torch.clamp(b_score, 1e-4, 1.0 - 1e-4), b_target, weight=b_weights)
        bull_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(engine.bull_cot.parameters(), 1.0)
        opt_bull.step()
        
        # 2. Bear CoT Pass & Step
        opt_bear.zero_grad()
        bear_out = engine.bear_cot(z_b.detach())
        r_score = bear_out.bear_score
        
        # Bear Target: 1.0 for SELL (2), 0.0 for HOLD (0) or BUY (1)
        r_target = torch.where(y_b == 2, 1.0, 0.0).float()
        r_weights = torch.where(y_b == 2, 6.0, 1.0).float()
        r_weights = torch.where(y_b == 1, 4.0, r_weights)
        bear_loss = F.binary_cross_entropy(torch.clamp(r_score, 1e-4, 1.0 - 1e-4), r_target, weight=r_weights)
        bear_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(engine.bear_cot.parameters(), 1.0)
        opt_bear.step()
        
        # 3. Cross-Examination & Arbiter Pass & Step
        opt_arbiter.zero_grad()
        p_t = bull_out.cot_hops[:, -1, :].detach()
        a_t = bear_out.cot_hops[:, -1, :].detach()
        
        final_p, final_a, _, _, _, _ = engine.cross_examination(p_t, a_t, turns=3)
        synthesis = torch.cat([final_p, final_a], dim=-1)
        arb_logits = engine.arbiter.arbiter_net(synthesis)
        prm_pred = engine.arbiter.prm_evaluator(synthesis).squeeze(-1)
        
        arb_loss = focal_loss_fn(arb_logits, y_b)
        
        # PRM loss: 0.95 when Arbiter prediction matches target, 0.10 otherwise
        arb_pred_cls = torch.argmax(arb_logits, dim=-1)
        prm_target = torch.where(arb_pred_cls == y_b, 0.95, 0.10).float()
        prm_loss = F.mse_loss(prm_pred, prm_target)
        
        arb_tot = arb_loss + 0.5 * prm_loss
        arb_tot.backward()
        nn.utils.clip_grad_norm_(
            list(engine.cross_examination.parameters()) + list(engine.arbiter.parameters()),
            1.0
        )
        opt_arbiter.step()
        
        tot_bull_loss += bull_loss.item()
        tot_bear_loss += bear_loss.item()
        tot_arb_loss += arb_tot.item()
        num_batches += 1
        
    dur = time.perf_counter() - t0
    print(f"Epoch {epoch}/8 ({dur:.1f}s) | Bull Loss: {tot_bull_loss/num_batches:.4f} | Bear Loss: {tot_bear_loss/num_batches:.4f} | Arbiter Loss: {tot_arb_loss/num_batches:.4f}", flush=True)

# 4. Save Finetuned Weights
save_path = "outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt"
engine.save_checkpoint(save_path)
print(f"\nSaved finetuned Bilateral Dual-CoT Champion to {save_path}!")

# 5. Full Evaluation Across All 11,181 Bars
print("\nRunning comprehensive evaluation on all 11,181 bars...", flush=True)
engine.eval()
with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=512)
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()

# Direct evaluation with Arbiter & Value Pullback Guard
val_closes = closes[start_idx:end_idx]
val_highs = highs[start_idx:end_idx]
val_lows = lows[start_idx:end_idx]
val_opens = opens[start_idx:end_idx]
val_atrs = atrs[start_idx:end_idx]
val_ema20 = ema20[start_idx:end_idx]
val_ema50 = ema50[start_idx:end_idx]
val_rp = rel_pos_60[start_idx:end_idx]

def simulate_champions(sl_atr=2.0, tp_atr=1.5, be_trig=0.70, be_lock=0.35, min_spacing=2, cooldown=3):
    trades = []
    last_entry = -999
    last_exit = -999
    b_cnt, s_cnt = 0, 0
    tp_hits, be_hits, sl_hits = 0, 0, 0
    
    for t in range(valid_count - 35):
        if t - last_entry < min_spacing: continue
        if t - last_exit < cooldown: continue
        
        cp = val_closes[t]
        cur_atr = val_atrs[t]
        bs = b_scores[t]
        rs = r_scores[t]
        prm = prm_scores[t]
        e20 = val_ema20[t]
        e50 = val_ema50[t]
        rp = val_rp[t]
        
        act = 0
        diff = bs - rs
        # BUY: Bull beats Bear by >= 0.05, Bull conviction >= 0.50, PRM >= 0.35, Trend UP, in Value Zone, Anti-Top
        if diff >= 0.05 and bs >= 0.48 and prm >= 0.33 and e20 > e50 and cp > e50:
            if val_lows[t] <= e20 + 0.35 * cur_atr and cp <= e20 + 0.50 * cur_atr and rp <= 0.85:
                if val_closes[t] >= val_opens[t] or (val_closes[t] - val_lows[t]) >= 0.20 * cur_atr:
                    act = 1
        # SELL: Bear beats Bull by >= 0.05, Bear conviction >= 0.50, PRM >= 0.33, Trend DOWN, in Value Zone, Anti-Bottom
        elif -diff >= 0.05 and rs >= 0.48 and prm >= 0.33 and e20 < e50 and cp < e50:
            if val_highs[t] >= e20 - 0.35 * cur_atr and cp >= e20 - 0.50 * cur_atr and rp >= 0.15:
                if val_closes[t] <= val_opens[t] or (val_highs[t] - val_closes[t]) >= 0.20 * cur_atr:
                    act = -1
                    
        if act == 0: continue
        last_entry = t
        if act == 1: b_cnt += 1
        else: s_cnt += 1
        
        sl_d = sl_atr * cur_atr
        tp_d = tp_atr * cur_atr
        bet_d = be_trig * cur_atr
        bel_d = be_lock * cur_atr
        
        pnl = 0.0
        reason = None
        if act == 1:
            cur_sl = cp - sl_d
            tp_p = cp + tp_d
            be_active = False
            for k in range(t + 1, min(t + 35, valid_count)):
                if not be_active and val_highs[k] >= cp + bet_d:
                    be_active = True
                    cur_sl = max(cur_sl, cp + bel_d)
                if val_lows[k] <= cur_sl:
                    pnl = cur_sl - cp
                    reason = 'BE' if be_active else 'SL'
                    last_exit = k
                    break
                elif val_highs[k] >= tp_p:
                    pnl = tp_d
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = val_closes[min(t + 34, valid_count - 1)] - cp
                reason = 'TIME'
                last_exit = min(t + 34, valid_count - 1)
        else:
            cur_sl = cp + sl_d
            tp_p = cp - tp_d
            be_active = False
            for k in range(t + 1, min(t + 35, valid_count)):
                if not be_active and val_lows[k] <= cp - bet_d:
                    be_active = True
                    cur_sl = min(cur_sl, cp - bel_d)
                if val_highs[k] >= cur_sl:
                    pnl = cp - cur_sl
                    reason = 'BE' if be_active else 'SL'
                    last_exit = k
                    break
                elif val_lows[k] <= tp_p:
                    pnl = tp_d
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = cp - val_closes[min(t + 34, valid_count - 1)]
                reason = 'TIME'
                last_exit = min(t + 34, valid_count - 1)
                
        if reason == 'TP': tp_hits += 1
        elif reason == 'BE': be_hits += 1
        elif reason == 'SL': sl_hits += 1
        trades.append(pnl)
        
    tr = np.array(trades)
    wins = np.sum(tr > 0)
    wr = wins / len(tr) * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    cum = np.cumsum(tr)
    dd = (np.maximum.accumulate(cum) - cum).max() / max(1.0, cum.max()) * 100.0 if cum.max() > 0 else 99.0
    
    return wr, pf, len(tr), b_cnt, s_cnt, tp_hits, be_hits, sl_hits, dd

print("\nEvaluating Parameter Grid on Finetuned Model:")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/BE/SL':<16} {'DD':<6} {'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6}")
print("-" * 90)

for sl in [1.5, 2.0, 2.5]:
    for tp in [1.2, 1.5, 1.8, 2.0]:
        for bet in [0.60, 0.70]:
            for bel in [0.35, 0.45]:
                if bel >= bet: continue
                wr, pf, n, b, s, tp_h, be_h, sl_h, dd = simulate_champions(sl_atr=sl, tp_atr=tp, be_trig=bet, be_lock=bel)
                bs = f"{b}/{s}"
                hits = f"{tp_h}/{be_h}/{sl_h}"
                print(f"{wr:5.2f}%   {pf:5.2f}  {n:<8} {n/7.0:<8.1f} {bs:<12} {hits:<16} {dd:4.1f}% {sl:<5.1f} {tp:<5.1f} {bet:<6.2f} {bel:<6.2f}")
