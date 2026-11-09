import os
import sys
from pathlib import Path
import time
import math
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    BilateralBatchVerdict,
    extract_canonical_bilateral_ingress_233,
)
from self_growing_trader.canonical_features import (
    MARKET_FEATURE_COLUMNS,
    add_features,
    calculate_atr,
    calculate_ema,
)

print("=" * 80)
print("  TRAINING GENUINE SPECIALIZED DUAL CHAMPIONS (BULL & BEAR) - v2")
print("  Target: Precision Timing, Spread-Immune Pullback Entries, WR >= 80%, PF > 3.0")
print("=" * 80, flush=True)

# 1. Load M1 Data (11,513 bars)
data_path = Path("self_growing_trader/data/canonical_train_m1.csv")
if not data_path.exists():
    data_path = Path("self_growing_trader/data/BTCUSDm_M1_continuous.csv")

df = pd.read_csv(data_path)
if 'Datetime' not in df.columns:
    df['Datetime'] = pd.to_datetime(df['time'] if 'time' in df.columns else df.iloc[:, 0])
else:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

print(f"Calculating 137D canonical features across {len(df):,} bars...", flush=True)
df_feat = add_features(df)

closes = df['close'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
opens = df['open'].values.astype(np.float64)

atrs = calculate_atr(highs, lows, closes, 14)
ema20 = calculate_ema(closes, 20)
ema50 = calculate_ema(closes, 50)
ema200 = calculate_ema(closes, 200)

feat_cols = [c for c in MARKET_FEATURE_COLUMNS if c in df_feat.columns]
raw_features = df_feat[feat_cols].values.astype(np.float32)
if raw_features.shape[1] < 137:
    pad = np.zeros((len(df), 137 - raw_features.shape[1]), dtype=np.float32)
    base_obs_137_all = np.hstack([raw_features, pad])
else:
    base_obs_137_all = raw_features[:, :137]

lookback = 60
horizon = 20
N_total = len(df)
valid_count = N_total - lookback - horizon

print(f"Extracting 233D ingress features and labeling sniper pullback targets...", flush=True)
ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
targets = np.zeros(valid_count, dtype=np.int64) # 0: HOLD, 1: BUY, 2: SELL

rel_pos_60 = np.zeros(N_total, dtype=np.float64)
for t in range(lookback, N_total):
    min_60 = np.min(lows[t-lookback:t+1])
    max_60 = np.max(highs[t-lookback:t+1])
    rng = max(1e-4, max_60 - min_60)
    rel_pos_60[t] = (closes[t] - min_60) / rng

for i in range(valid_count):
    t = i + lookback
    cur_p = closes[t]
    cur_atr = max(atrs[t], 1.0)
    
    # 233D ingress vector
    obs_137 = base_obs_137_all[t]
    bars_60 = df[['open', 'high', 'low', 'close', 'tick_volume']].iloc[t-60+1:t+1].to_dict('records')
    ingress_233 = extract_canonical_bilateral_ingress_233(
        base_obs_137=obs_137,
        m1_bars_60=bars_60,
        cur_p=cur_p,
        cur_atr=cur_atr
    )
    ingress_matrix[i] = ingress_233
    
    fwd_h = highs[t+1 : t+1+horizon]
    fwd_l = lows[t+1 : t+1+horizon]
    
    # BUY TARGET (BULL SPECIALIST):
    # 1. Uptrend: EMA20 > EMA50 and Price > EMA50
    # 2. Value Zone Pullback: lows[t] reached near EMA20 (lows <= ema20 + 0.40 ATR) and Close <= ema20 + 0.60 ATR
    # 3. Anti-Top: rel_pos_60 <= 0.80 (must NOT be near 60-bar high)
    # 4. Candlestick Confirmation: Green candle or lower wick >= 0.15 ATR
    # 5. Real Spread-Compensated Forward Outcome:
    #    Target move: +2.0 ATR before hitting -1.50 ATR adverse excursion.
    up = ema20[t] > ema50[t] and cur_p > ema50[t]
    in_buy_zone = (lows[t] <= ema20[t] + 0.40 * cur_atr) and (cur_p <= ema20[t] + 0.60 * cur_atr) and (rel_pos_60[t] <= 0.80)
    buy_trig = (closes[t] >= opens[t]) or ((closes[t] - lows[t]) >= 0.15 * cur_atr)
    
    buy_ok = False
    if up and in_buy_zone and buy_trig:
        for k in range(len(fwd_h)):
            if (cur_p - fwd_l[k]) >= 1.50 * cur_atr:
                break
            if (fwd_h[k] - cur_p) >= 2.00 * cur_atr:
                buy_ok = True
                break
                
    # SELL TARGET (BEAR SPECIALIST):
    # 1. Downtrend: EMA20 < EMA50 and Price < EMA50
    # 2. Value Zone Rally: highs[t] reached near EMA20 (highs >= ema20 - 0.40 ATR) and Close >= ema20 - 0.60 ATR
    # 3. Anti-Bottom: rel_pos_60 >= 0.20 (must NOT be near 60-bar low)
    # 4. Candlestick Confirmation: Red candle or upper wick >= 0.15 ATR
    # 5. Real Spread-Compensated Forward Outcome:
    #    Target move: -2.0 ATR before rising +1.50 ATR adverse excursion.
    down = ema20[t] < ema50[t] and cur_p < ema50[t]
    in_sell_zone = (highs[t] >= ema20[t] - 0.40 * cur_atr) and (cur_p >= ema20[t] - 0.60 * cur_atr) and (rel_pos_60[t] >= 0.20)
    sell_trig = (closes[t] <= opens[t]) or ((highs[t] - closes[t]) >= 0.15 * cur_atr)
    
    sell_ok = False
    if down and in_sell_zone and sell_trig:
        for k in range(len(fwd_l)):
            if (fwd_h[k] - cur_p) >= 1.50 * cur_atr:
                break
            if (cur_p - fwd_l[k]) >= 2.00 * cur_atr:
                sell_ok = True
                break
                
    if buy_ok and not sell_ok:
        targets[i] = 1 # BUY
    elif sell_ok and not buy_ok:
        targets[i] = 2 # SELL
    else:
        targets[i] = 0 # HOLD

b_cnt = int(np.sum(targets == 1))
s_cnt = int(np.sum(targets == 2))
h_cnt = int(np.sum(targets == 0))
print(f"Dataset labeled: BUY={b_cnt} ({b_cnt/valid_count*100:.1f}%), SELL={s_cnt} ({s_cnt/valid_count*100:.1f}%), HOLD={h_cnt} ({h_cnt/valid_count*100:.1f}%)", flush=True)

# 2. Instantiate Bilateral Dual-CoT Engine
device = torch.device("cpu")
torch.set_num_threads(4)

config = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.33,
    trap_threshold=0.60,
)
engine = BilateralDualCoTEngine(config).to(device)

ckpt_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
backup_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion_prev_backup.pt")
if backup_path.exists():
    try:
        engine.load_checkpoint(str(backup_path))
        print("Loaded initial weights from backup checkpoint.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

# Function to properly initialize specialist module with orthogonal weights & ones for LayerNorms!
def init_specialist_weights(module):
    if isinstance(module, (nn.Linear,)):
        nn.init.orthogonal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

engine.bear_cot.apply(init_specialist_weights)
print("Re-initialized Bear CoT with orthogonal weights & ones-norm to guarantee distinct specialist representations!")

# 3. Unified Joint Optimizer with balanced learning rates
optimizer = torch.optim.AdamW([
    {"params": engine.state_adapter.parameters(), "lr": 1.5e-4},
    {"params": engine.bull_cot.parameters(), "lr": 2.0e-4},
    {"params": engine.bear_cot.parameters(), "lr": 3.0e-4}, # Slightly higher lr for newly initialized bear
    {"params": engine.cross_examination.parameters(), "lr": 2.0e-4},
    {"params": engine.arbiter.parameters(), "lr": 2.5e-4},
    {"params": engine.trap_detector.parameters(), "lr": 1.5e-4},
], weight_decay=1e-5)

class FocalLoss(nn.Module):
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

alpha_weights = torch.tensor([1.0, 6.0, 6.0], device=device)
focal_loss_fn = FocalLoss(alpha=alpha_weights, gamma=2.0)

# 4. Training Loop: 12 Epochs
epochs = 12
batch_size = 64
N = valid_count
indices = np.arange(N)

print(f"\nStarting 12-Epoch True Specialist Training over {N} bars...", flush=True)

for epoch in range(1, epochs + 1):
    engine.train()
    np.random.shuffle(indices)
    t0 = time.perf_counter()
    
    tot_bull_loss = 0.0
    tot_bear_loss = 0.0
    tot_arb_loss = 0.0
    tot_loss_sum = 0.0
    num_batches = 0
    
    for b_start in range(0, N, batch_size):
        b_idx = indices[b_start : b_start + batch_size]
        if len(b_idx) < 8: continue
        
        x_b = torch.from_numpy(ingress_matrix[b_idx]).float().to(device)
        y_b = torch.from_numpy(targets[b_idx]).long().to(device)
        
        # 1. State adapter pass
        z_b = engine.state_adapter(x_b)
        
        # 2. Bull specialist pass
        bull_out = engine.bull_cot(z_b)
        b_score = bull_out.bull_score
        b_target = torch.where(y_b == 1, 1.0, 0.0).float()
        b_weights = torch.where(y_b == 1, 8.0, 1.0).float()
        b_weights = torch.where(y_b == 2, 5.0, b_weights) # Penalize buying during sell setups
        bull_loss = F.binary_cross_entropy(torch.clamp(b_score, 1e-4, 1.0 - 1e-4), b_target, weight=b_weights)

        # 3. Bear specialist pass
        bear_out = engine.bear_cot(z_b)
        r_score = bear_out.bear_score
        r_target = torch.where(y_b == 2, 1.0, 0.0).float()
        r_weights = torch.where(y_b == 2, 8.0, 1.0).float()
        r_weights = torch.where(y_b == 1, 5.0, r_weights) # Penalize shorting during buy setups
        bear_loss = F.binary_cross_entropy(torch.clamp(r_score, 1e-4, 1.0 - 1e-4), r_target, weight=r_weights)
        
        # 4. Cross-Examination & Arbiter Pass
        p_t = bull_out.cot_hops[:, -1, :]
        a_t = bear_out.cot_hops[:, -1, :]
        final_p, final_a, _, _, _, _ = engine.cross_examination(p_t, a_t, turns=3)
        synthesis = torch.cat([final_p, final_a], dim=-1)
        arb_logits = engine.arbiter.arbiter_net(synthesis)
        prm_pred = engine.arbiter.prm_evaluator(synthesis).squeeze(-1)
        
        arb_loss = focal_loss_fn(arb_logits, y_b)
        
        # PRM loss
        arb_pred_cls = torch.argmax(arb_logits, dim=-1)
        prm_target = torch.where(arb_pred_cls == y_b, 0.95, 0.10).float()
        prm_loss = F.mse_loss(prm_pred, prm_target)
        
        # Unified Joint Loss
        total_batch_loss = bull_loss + bear_loss + 1.2 * arb_loss + 0.5 * prm_loss
        
        optimizer.zero_grad()
        total_batch_loss.backward()
        nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        optimizer.step()
        
        tot_bull_loss += bull_loss.item()
        tot_bear_loss += bear_loss.item()
        tot_arb_loss += arb_loss.item()
        tot_loss_sum += total_batch_loss.item()
        num_batches += 1
        
    dur = time.perf_counter() - t0
    print(f"Epoch {epoch:02d}/12 ({dur:.1f}s) | Total Loss: {tot_loss_sum/num_batches:.4f} | Bull: {tot_bull_loss/num_batches:.4f} | Bear: {tot_bear_loss/num_batches:.4f} | Arbiter: {tot_arb_loss/num_batches:.4f}", flush=True)

# 5. Save Newly Trained Champions
save_path = "outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt"
engine.save_checkpoint(save_path)
print(f"\n[OK] Saved genuine specialized Dual-Champion Checkpoint to {save_path}!")

# 6. Evaluation & Verification on Full 11,513 Bars with Real Spread
print("\nEvaluating new Dual Champions across full dataset...", flush=True)
engine.eval()
with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=512)
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()
    acts = verdict.actions.numpy()

print(f"New predictions: SELL={np.sum(acts == -1)}, HOLD={np.sum(acts == 0)}, BUY={np.sum(acts == 1)}")
print(f"Bull Score: mean={np.mean(b_scores):.3f}, max={np.max(b_scores):.3f}, p95={np.percentile(b_scores, 95):.3f}")
print(f"Bear Score: mean={np.mean(r_scores):.3f}, max={np.max(r_scores):.3f}, p95={np.percentile(r_scores, 95):.3f}")
print(f"PRM Score:  mean={np.mean(prm_scores):.3f}, max={np.max(prm_scores):.3f}")

bull_w = sum(p.norm().item() for p in engine.bull_cot.parameters())
bear_w = sum(p.norm().item() for p in engine.bear_cot.parameters())
print(f"Bull Weights Norm: {bull_w:.2f} vs Bear Weights Norm: {bear_w:.2f}")

# Backtest with Real Exness Spread ($28.98)
vol = 0.15
spread_cost = 28.98

def run_backtest(sl_atr=2.2, tp_atr=3.2, be_trig=1.4, be_lock=0.6, min_spacing=2, cooldown=30, prm_th=0.33, margin_th=0.08, score_th=0.55):
    pnl_list = []
    wins = 0
    losses = 0
    last_exit = -999
    consec_loss = 0
    cd_until = -999
    b_trades = 0
    s_trades = 0
    
    for i in range(len(b_scores) - 60):
        t = i + lookback
        if t < cd_until: continue
        if t - last_exit < min_spacing: continue
        
        bs = b_scores[i]
        rs = r_scores[i]
        prm = prm_scores[i]
        diff = bs - rs
        cur_p = closes[t]
        cur_atr = max(atrs[t], 1.0)
        
        act = 0
        if bs >= score_th and diff >= margin_th and prm >= prm_th:
            act = 1
        elif rs >= score_th and -diff >= margin_th and prm >= prm_th:
            act = -1
            
        if act == 0: continue
        
        trade_pnl = 0.0
        exit_r = ''
        
        if act == 1:
            b_trades += 1
            entry_p = cur_p + spread_cost # BUY at Ask
            sl_p = cur_p - sl_atr * cur_atr
            tp_p = cur_p + tp_atr * cur_atr
            be_active = False
            
            for k in range(t + 1, min(t + 120, N_total)):
                h_k = highs[k]
                l_k = lows[k]
                if not be_active and h_k >= cur_p + be_trig * cur_atr:
                    be_active = True
                    sl_p = max(sl_p, cur_p + be_lock * cur_atr)
                if l_k <= sl_p:
                    exit_p = sl_p
                    exit_r = 'BE' if be_active else 'SL'
                    trade_pnl = (exit_p - entry_p) * vol
                    last_exit = k
                    break
                elif h_k >= tp_p:
                    exit_p = tp_p
                    exit_r = 'TP'
                    trade_pnl = (exit_p - entry_p) * vol
                    last_exit = k
                    break
            else:
                exit_p = closes[min(t + 119, N_total - 1)]
                trade_pnl = (exit_p - entry_p) * vol
                last_exit = min(t + 119, N_total - 1)
                exit_r = 'TIME'
        else:
            s_trades += 1
            entry_p = cur_p # SELL at Bid
            sl_p = cur_p + sl_atr * cur_atr + spread_cost
            tp_p = cur_p - tp_atr * cur_atr
            be_active = False
            
            for k in range(t + 1, min(t + 120, N_total)):
                h_k = highs[k] + spread_cost # BUY to cover Ask
                l_k = lows[k] + spread_cost
                if not be_active and lows[k] <= cur_p - be_trig * cur_atr:
                    be_active = True
                    sl_p = min(sl_p, cur_p - be_lock * cur_atr)
                if h_k >= sl_p:
                    exit_p = sl_p
                    exit_r = 'BE' if be_active else 'SL'
                    trade_pnl = (entry_p - exit_p) * vol
                    last_exit = k
                    break
                elif l_k <= tp_p:
                    exit_p = tp_p
                    exit_r = 'TP'
                    trade_pnl = (entry_p - exit_p) * vol
                    last_exit = k
                    break
            else:
                exit_p = closes[min(t + 119, N_total - 1)] + spread_cost
                trade_pnl = (entry_p - exit_p) * vol
                last_exit = min(t + 119, N_total - 1)
                exit_r = 'TIME'
                
        pnl_list.append(trade_pnl)
        if trade_pnl > 0:
            wins += 1
            consec_loss = 0
        else:
            losses += 1
            consec_loss += 1
            if consec_loss >= 2:
                cd_until = last_exit + cooldown
                consec_loss = 0
                
    if not pnl_list:
        print(f'Thresh={score_th:.2f} Diff={margin_th:.2f} | No trades.')
        return
    tr = np.array(pnl_list)
    wr = np.sum(tr > 0) / len(tr) * 100.0
    gp = np.sum(tr[tr > 0])
    gl = np.sum(np.abs(tr[tr < 0]))
    pf = gp / max(1e-4, gl)
    net = np.sum(tr)
    cum = np.cumsum(tr)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / 10000.0 * 100.0
    max_dd = np.max(dd) if len(dd) > 0 else 0.0
    
    print(f'Thresh={score_th:.2f} Diff={margin_th:.2f} SL={sl_atr:.1f} TP={tp_atr:.1f} | Trades={len(tr):3d} (BUY={b_trades}, SELL={s_trades}) | WR={wr:5.1f}% | PF={pf:4.2f} | MaxDD={max_dd:4.1f}% | Net=${net:+7.2f} | GP=${gp:6.2f} | GL=${gl:6.2f}')

print('\nTesting Performance across Geometry & Thresholds with Real Exness Spread ($28.98):')
for (sl, tp, trig, lk) in [(2.2, 3.2, 1.4, 0.6), (2.5, 3.5, 1.5, 0.8), (3.0, 4.0, 1.8, 1.0)]:
    for sc in [0.45, 0.50, 0.55]:
        for mg in [0.05, 0.08]:
            run_backtest(sl_atr=sl, tp_atr=tp, be_trig=trig, be_lock=lk, score_th=sc, margin_th=mg)
