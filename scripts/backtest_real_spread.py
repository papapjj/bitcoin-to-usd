import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import pandas as pd, numpy as np, torch

from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig, extract_canonical_bilateral_ingress_233
from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)

df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df_renamed = df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume', '<VOL>': 'RealVolume', '<SPREAD>': 'Spread'
})
df_renamed['Datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
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

closes = feat_df['Close'].values.astype(np.float64)
highs = feat_df['High'].values.astype(np.float64)
lows = feat_df['Low'].values.astype(np.float64)
opens = feat_df['Open'].values.astype(np.float64)
atrs = feat_df['atr'].values.astype(np.float64)

config = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.33,
    trap_threshold=0.60,
)
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint('outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt')
engine.eval()

lookback = 60
print('Computing all scores across all 11,513 bars...')
with torch.no_grad():
    ingress_matrix = np.zeros((N - lookback, 233), dtype=np.float32)
    for i, t in enumerate(range(lookback, N)):
        cur_p = closes[t]
        cur_atr = max(atrs[t], 1.0)
        m1_bars_60 = np.column_stack([opens[t-lookback:t], highs[t-lookback:t], lows[t-lookback:t], closes[t-lookback:t]])
        ingress_matrix[i] = extract_canonical_bilateral_ingress_233(base_137[t], m1_bars_60, cur_p, cur_atr)
    
    verdict = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=512)
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()

print(f'Computed {len(b_scores)} bar scores.')

# Test with Real Exness Spread ($28.98)
spread_cost = 28.98
vol = 0.15

def run_backtest(sl_atr=4.5, tp_atr=3.5, be_trig=1.8, be_lock=1.0, min_spacing=2, cooldown=45, prm_th=0.33, margin_th=0.08, score_th=0.55):
    pnl_list = []
    wins = 0
    losses = 0
    last_exit = -999
    consec_loss = 0
    cd_until = -999
    
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
            entry_p = cur_p + spread_cost # BUY at Ask
            sl_p = cur_p - sl_atr * cur_atr
            tp_p = cur_p + tp_atr * cur_atr
            be_active = False
            
            for k in range(t + 1, min(t + 120, N)):
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
                exit_p = closes[min(t + 119, N - 1)]
                trade_pnl = (exit_p - entry_p) * vol
                last_exit = min(t + 119, N - 1)
                exit_r = 'TIME'
        else:
            entry_p = cur_p # SELL at Bid
            sl_p = cur_p + sl_atr * cur_atr + spread_cost
            tp_p = cur_p - tp_atr * cur_atr
            be_active = False
            
            for k in range(t + 1, min(t + 120, N)):
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
                exit_p = closes[min(t + 119, N - 1)] + spread_cost
                trade_pnl = (entry_p - exit_p) * vol
                last_exit = min(t + 119, N - 1)
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
        print('No trades.')
        return
    tr = np.array(pnl_list)
    wr = np.sum(tr > 0) / len(tr) * 100.0
    gp = np.sum(tr[tr > 0])
    gl = np.sum(np.abs(tr[tr < 0]))
    pf = gp / max(1e-4, gl)
    net = np.sum(tr)
    print(f'Score>={score_th:.2f} Diff>={margin_th:.2f} | Trades={len(tr):3d} | WR={wr:5.1f}% | PF={pf:4.2f} | Net=${net:+7.2f} | GP=${gp:6.2f} | GL=${gl:6.2f}')

print('\nTesting different thresholds with Real Exness Spread ($28.98):')
for sc in [0.48, 0.52, 0.55, 0.60, 0.65]:
    for mg in [0.05, 0.10, 0.15]:
        run_backtest(score_th=sc, margin_th=mg)
