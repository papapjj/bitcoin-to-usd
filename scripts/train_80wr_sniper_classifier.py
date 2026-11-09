import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import pandas as pd
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
print("  TRAINING HIGH-PRECISION BILATERAL DUAL-COT CHAMPION")
print("  Target: Realistic Spread-Immune Edge (SL 3.5 ATR, TP 2.8 ATR)")
print("=" * 80, flush=True)

# 1. Ingest Data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df_renamed = df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume', '<VOL>': 'RealVolume', '<SPREAD>': 'Spread'
})
if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
    df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']

feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
N_total = len(feat_df)

closes = feat_df['Close'].values.astype(np.float64)
highs = feat_df['High'].values.astype(np.float64)
lows = feat_df['Low'].values.astype(np.float64)
opens = feat_df['Open'].values.astype(np.float64)
atrs = feat_df['atr'].values.astype(np.float64)

# 137D features
m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
slots = np.zeros((N_total, 40), dtype=np.float32)
account = np.zeros((N_total, 8), dtype=np.float32)
m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
execution = np.zeros((N_total, 11), dtype=np.float32)
timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

lookback = 60
horizon = 60
start_idx = lookback
end_idx = N_total - horizon
valid_count = end_idx - start_idx

REAL_SPREAD = 28.98
SL_ATR = 3.5
TP_ATR = 2.8

ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
targets = np.zeros(valid_count, dtype=np.int64) # 0: HOLD, 1: BUY, 2: SELL

val_ema_diff = base_137[:, 11]
val_p_e20 = base_137[:, 26]
val_p_e50 = base_137[:, 27]
val_rp = base_137[:, 13]
val_lw = base_137[:, 9]
val_uw = base_137[:, 8]

print(f"Extracting 233D ingress features and labeling {valid_count} bars...", flush=True)

for i, t in enumerate(range(start_idx, end_idx)):
    cur_p = closes[t]
    cur_atr = max(atrs[t], 10.0)
    
    m1_bars_60 = np.column_stack([
        opens[t-lookback:t],
        highs[t-lookback:t],
        lows[t-lookback:t],
        closes[t-lookback:t],
    ])
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(base_137[t], m1_bars_60, cur_p, cur_atr)
    
    # Ground truth SNIPER EDGE:
    up = (val_ema_diff[t] > 0) and (val_p_e50[t] > 0)
    in_buy_zone = (val_p_e20[t] <= 0.40) and (val_rp[t] <= 0.85)
    buy_trig = (closes[t] >= opens[t]) or (val_lw[t] >= 0.20)
    
    buy_ok = False
    if up and in_buy_zone and buy_trig:
        sl_p = cur_p - SL_ATR * cur_atr
        tp_p = cur_p + TP_ATR * cur_atr
        for k in range(t + 1, min(t + 1 + horizon, N_total)):
            if lows[k] <= sl_p: break
            if highs[k] >= tp_p: buy_ok = True; break
            
    down = (val_ema_diff[t] < 0) and (val_p_e50[t] < 0)
    in_sell_zone = (val_p_e20[t] >= -0.40) and (val_rp[t] >= 0.15)
    sell_trig = (closes[t] <= opens[t]) or (val_uw[t] >= 0.20)
    
    sell_ok = False
    if down and in_sell_zone and sell_trig:
        sl_p = cur_p + SL_ATR * cur_atr
        tp_p = cur_p - TP_ATR * cur_atr
        for k in range(t + 1, min(t + 1 + horizon, N_total)):
            if highs[k] + REAL_SPREAD >= sl_p: break
            if lows[k] + REAL_SPREAD <= tp_p: sell_ok = True; break
            
    if buy_ok and not sell_ok: targets[i] = 1
    elif sell_ok and not buy_ok: targets[i] = 2
    else: targets[i] = 0

b_cnt = int(np.sum(targets == 1))
s_cnt = int(np.sum(targets == 2))
h_cnt = int(np.sum(targets == 0))
print(f"Target distribution: BUY={b_cnt} ({b_cnt/valid_count*100:.1f}%), SELL={s_cnt} ({s_cnt/valid_count*100:.1f}%), HOLD={h_cnt} ({h_cnt/valid_count*100:.1f}%)", flush=True)

# 2. Setup Bilateral Dual-CoT Engine
device = torch.device("cpu")
torch.set_num_threads(4)

config = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.33,
    trap_threshold=0.60
)
engine = BilateralDualCoTEngine(config).to(device)

pt_path = "outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt"
if os.path.exists(pt_path):
    engine.load_checkpoint(pt_path)
    print(f"Loaded existing champion weights from {pt_path}")

# Separate optimizers for Bull Specialist, Bear Specialist, and Arbiter
opt_bull = torch.optim.AdamW(engine.bull_cot.parameters(), lr=1.5e-4, weight_decay=1e-4)
opt_bear = torch.optim.AdamW(engine.bear_cot.parameters(), lr=1.5e-4, weight_decay=1e-4)
opt_arbiter = torch.optim.AdamW(
    list(engine.cross_examination.parameters()) + list(engine.arbiter.parameters()),
    lr=2e-4, weight_decay=1e-4
)

X_tensor = torch.from_numpy(ingress_matrix).float()
y_tensor = torch.from_numpy(targets).long()

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()

focal_loss_fn = FocalLoss(alpha=torch.tensor([1.0, 5.0, 5.0], dtype=torch.float32, device=device))

num_epochs = 8
print(f"\nStarting {num_epochs} Epochs of Supervised Finetuning...", flush=True)

for epoch in range(1, num_epochs + 1):
    engine.train()
    tot_bull, tot_bear, tot_arb = 0.0, 0.0, 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        z_b = engine.state_adapter(batch_x)
        
        # 1. Bull CoT Pass & Step
        opt_bull.zero_grad()
        bull_out = engine.bull_cot(z_b)
        b_score = bull_out.bull_score
        b_target = torch.where(batch_y == 1, 1.0, 0.0).float()
        b_weights = torch.where(batch_y == 1, 6.0, 1.0).float()
        b_weights = torch.where(batch_y == 2, 4.0, b_weights)
        bull_loss = F.binary_cross_entropy(torch.clamp(b_score, 1e-4, 1.0 - 1e-4), b_target, weight=b_weights)
        bull_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(engine.bull_cot.parameters(), 1.0)
        opt_bull.step()
        
        # 2. Bear CoT Pass & Step
        opt_bear.zero_grad()
        bear_out = engine.bear_cot(z_b.detach())
        r_score = bear_out.bear_score
        r_target = torch.where(batch_y == 2, 1.0, 0.0).float()
        r_weights = torch.where(batch_y == 2, 6.0, 1.0).float()
        r_weights = torch.where(batch_y == 1, 4.0, r_weights)
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
        
        arb_loss = focal_loss_fn(arb_logits, batch_y)
        arb_pred_cls = torch.argmax(arb_logits, dim=-1)
        prm_target = torch.where(arb_pred_cls == batch_y, 0.95, 0.10).float()
        prm_loss = F.mse_loss(prm_pred, prm_target)
        
        arb_tot = arb_loss + 0.5 * prm_loss
        arb_tot.backward()
        nn.utils.clip_grad_norm_(
            list(engine.cross_examination.parameters()) + list(engine.arbiter.parameters()),
            1.0
        )
        opt_arbiter.step()
        
        tot_bull += bull_loss.item()
        tot_bear += bear_loss.item()
        tot_arb += arb_tot.item()
        correct += (arb_pred_cls == batch_y).sum().item()
        total += len(batch_y)
        
    acc = correct / total * 100.0
    print(f"Epoch {epoch:02d}/{num_epochs:02d} | Bull: {tot_bull/len(dataloader):.3f} | Bear: {tot_bear/len(dataloader):.3f} | Arbiter: {tot_arb/len(dataloader):.3f} | Acc: {acc:5.2f}%", flush=True)

# Save Finetuned Champion Checkpoint
engine.save_checkpoint(pt_path)
print(f"\nSaved calibrated Frontier Champion weights to {pt_path}")

# Run Full OOS Simulation
engine.eval()
with torch.no_grad():
    verdict = engine.forward_batch(X_tensor, batch_size=512)
    actions = verdict.actions.numpy()
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()

val_closes = closes[start_idx:end_idx]
val_highs = highs[start_idx:end_idx]
val_lows = lows[start_idx:end_idx]
val_opens = opens[start_idx:end_idx]
val_atrs = atrs[start_idx:end_idx]

trades = []
last_entry = -999
last_exit = -999
consec_sl = 0
pause_until = 0

tp_hits, sl_hits = 0, 0
b_w, b_l, s_w, s_l = 0, 0, 0, 0

for t in range(valid_count - horizon):
    if t < pause_until: continue
    if t - last_entry < 2: continue
    if t - last_exit < 1: continue
    
    act = int(actions[t])
    if act == 0: continue
    
    bs = b_scores[t]
    rs = r_scores[t]
    prm = prm_scores[t]
    diff = bs - rs
    
    # Precision gate
    if act == 1:
        if not (diff >= 0.05 and bs >= 0.48 and prm >= 0.33): continue
        if not (val_ema_diff[t] > 0 and val_p_e50[t] > 0 and val_p_e20[t] <= 0.40 and val_rp[t] <= 0.85): continue
        if not (val_closes[t] >= val_opens[t] or val_lw[t] >= 0.20): continue
    elif act == -1:
        if not (-diff >= 0.05 and rs >= 0.48 and prm >= 0.33): continue
        if not (val_ema_diff[t] < 0 and val_p_e50[t] < 0 and val_p_e20[t] >= -0.40 and val_rp[t] >= 0.15): continue
        if not (val_closes[t] <= val_opens[t] or val_uw[t] >= 0.20): continue
        
    last_entry = t
    cur_atr = max(val_atrs[t], 10.0)
    cp = val_closes[t]
    sl_dist = SL_ATR * cur_atr
    tp_dist = TP_ATR * cur_atr
    pnl = 0.0
    
    if act == 1:
        entry_p = cp + REAL_SPREAD * 0.5
        sl_p = cp - sl_dist
        tp_p = cp + tp_dist
        for k in range(t + 1, min(t + 1 + horizon, valid_count)):
            if val_lows[k] <= sl_p:
                pnl = sl_p - entry_p
                last_exit = k
                break
            elif val_highs[k] >= tp_p:
                pnl = tp_p - entry_p
                last_exit = k
                break
        else:
            pnl = val_closes[min(t + horizon - 1, valid_count - 1)] - entry_p
            last_exit = min(t + horizon - 1, valid_count - 1)
        if pnl > 0: b_w += 1; tp_hits += 1
        else: b_l += 1; sl_hits += 1
    else:
        entry_p = cp - REAL_SPREAD * 0.5
        sl_p = cp + sl_dist
        tp_p = cp - tp_dist
        for k in range(t + 1, min(t + 1 + horizon, valid_count)):
            if val_highs[k] + REAL_SPREAD >= sl_p:
                pnl = entry_p - sl_p
                last_exit = k
                break
            elif val_lows[k] + REAL_SPREAD <= tp_p:
                pnl = entry_p - tp_p
                last_exit = k
                break
        else:
            pnl = entry_p - (val_closes[min(t + horizon - 1, valid_count - 1)] + REAL_SPREAD)
            last_exit = min(t + horizon - 1, valid_count - 1)
        if pnl > 0: s_w += 1; tp_hits += 1
        else: s_l += 1; sl_hits += 1
        
    trades.append(pnl)
    if pnl <= 0:
        consec_sl += 1
        if consec_sl >= 2:
            pause_until = last_exit + 45
            consec_sl = 0
    else:
        consec_sl = 0

tr = np.array(trades)
n = len(tr)
wr = np.sum(tr > 0) / n * 100.0 if n > 0 else 0
gp = np.sum(tr[tr > 0])
gl = abs(np.sum(tr[tr < 0]))
pf = gp / gl if gl > 0 else 999.0
tot_pnl = np.sum(tr)
cum = np.cumsum(tr)
peak = np.maximum.accumulate(cum)
dd = (peak - cum).max() / max(10000.0, peak.max() + 10000.0) * 100.0

print("\n" + "=" * 80)
print("  POST-TRAINING OOS EMPIRICAL VERIFICATION RESULTS")
print("=" * 80)
print(f"Total Trades:     {n} ({n/7.0:.1f} trades/day)")
print(f"Win Rate:         {wr:.2f}% (Wins: {tp_hits}, Losses: {sl_hits})")
print(f"Profit Factor:    {pf:.2f}")
print(f"Net PnL ($):      ${tot_pnl:+,.2f} USD")
print(f"Max Drawdown:     {dd:.2f}%")
print(f"BUY Performance:  {b_w} Wins / {b_l} Losses ({b_w/(b_w+b_l)*100 if b_w+b_l>0 else 0:.1f}% WR)")
print(f"SELL Performance: {s_w} Wins / {s_l} Losses ({s_w/(s_w+s_l)*100 if s_w+s_l>0 else 0:.1f}% WR)")
print("=" * 80)
