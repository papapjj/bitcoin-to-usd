import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.frontier_selfplay_triad import (
    SelfPlayFineTuningSPIN,
    MultiAgentDebateAndReflexion,
    SelfTaughtReasonerSTaR,
    extract_action_logits,
)

print('1. Extracting real Governor Latents from audit bars...')
config = load_config('configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml')
base_checkpoint = 'outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl'
base_policy = tai_he_chuyen_gia_mot_context_mot_policy(base_checkpoint, config, seed=2026)
policy = MythosMT5Policy(base_policy=base_policy)

audit_path = Path('mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv')
with open(audit_path, 'r') as f:
    _ = f.readline()
    lines = [l for l in f if l.strip()]

closes, atrs, latents = [], [], []
for l in lines:
    parts = l.split('\t')
    obs = np.array([float(x) for x in parts[4].split('|')], dtype=np.float32)
    px = float(parts[19])
    atr = float(parts[20])
    res = policy.pipeline.process_tick(
        symbol='BTCUSDm',
        base_obs_137=obs,
        m1_bars_60=policy.m1_buffer,
        macro_quotes={
            'DXY': policy.macro_buffer[0],
            'US10Y': policy.macro_buffer[1],
            'SPX': policy.macro_buffer[2],
            'XAUUSD': policy.macro_buffer[3],
        },
        current_price=px,
    )
    z = policy.pipeline.last_governor_latent.detach().clone().cpu()
    closes.append(px)
    atrs.append(atr)
    latents.append(z)

k = 3
samples = []
for t in range(len(closes) - k):
    dp = closes[t+k] - closes[t]
    atr = atrs[t]
    target = 1 if dp > 0.35 * atr else (2 if dp < -0.35 * atr else 0)
    samples.append((latents[t], target, closes[t], atr))

print(f'Extracted {len(samples)} real samples. BUY: {sum(1 for s in samples if s[1]==1)}, SELL: {sum(1 for s in samples if s[1]==2)}, HOLD: {sum(1 for s in samples if s[1]==0)}')

# 2. Initialize Challenger and Adversary
cfg = ReasoningConfig(input_dim=768, hidden_dim=256, num_hops=4, max_hops=8)
challenger = MetacognitiveReasoningEngine(cfg)
adversary = MetacognitiveReasoningEngine(cfg)

champ_p = Path('outputs/he_chuyen_gia/v193_runs/mythos_muzero_coevolved_champion.pt')
ckpt = torch.load(champ_p, map_location='cpu', weights_only=False)
challenger.load_state_dict(ckpt['reasoning_engine_state_dict'], strict=True)
adversary.load_state_dict(ckpt['reasoning_engine_state_dict'], strict=True)

opt_c = torch.optim.AdamW(challenger.parameters(), lr=1e-4, weight_decay=1e-4)
opt_adv = torch.optim.AdamW(adversary.parameters(), lr=1e-4, weight_decay=1e-4)

spin = SelfPlayFineTuningSPIN(beta=0.1, inactivity_penalty=2.0)
mad = MultiAgentDebateAndReflexion(hidden_dim=256, debate_turns=3)
star = SelfTaughtReasonerSTaR(challenger, lr=1e-4)

# 3. Run Bilateral Co-Evolution Loop
print('\nStarting Bilateral Co-Evolution across 15 epochs on real market data...')
for epoch in range(1, 16):
    epoch_c_wins, epoch_adv_wins, epoch_inact = 0, 0, 0
    indices = np.random.permutation(len(samples))
    for idx in indices:
        z_t, target_t, px_t, atr_t = samples[idx]
        with torch.no_grad():
            out_c = challenger(z_t.unsqueeze(0))
            out_adv = adversary(z_t.unsqueeze(0))
        
        logits_c = extract_action_logits(out_c)
        logits_adv = extract_action_logits(out_adv)
        
        c_act = int(torch.argmax(logits_c, dim=-1).item())
        adv_act = int(torch.argmax(logits_adv, dim=-1).item())
        
        # Reflexion veto check for challenger
        vetoed, _ = mad.query_reflection_veto(z_t.unsqueeze(0), proposed_action=c_act)
        if vetoed:
            c_act = 0
            
        pnl_c = 45.0 if (c_act == target_t and c_act != 0) else (-30.0 if c_act != 0 else 0.0)
        pnl_adv = 45.0 if (adv_act == target_t and adv_act != 0) else (-30.0 if adv_act != 0 else 0.0)
        
        if c_act == 0 and adv_act == 0:
            epoch_inact += 1
        elif pnl_c > pnl_adv:
            epoch_c_wins += 1
        elif pnl_adv > pnl_c:
            epoch_adv_wins += 1
            
        if pnl_c < 0.0:
            mad.record_reflection(
                state_embedding=z_t.unsqueeze(0),
                trigger_price=px_t,
                regime='m1_market_flow',
                action_taken=c_act,
                pnl=pnl_c,
                mae=1.5,
            )
            
        spin.collect_bilateral_interaction(
            state_z=z_t,
            action_challenger=c_act,
            pnl_challenger=pnl_c,
            action_adversary=adv_act,
            pnl_adversary=pnl_adv,
            market_ground_truth=target_t,
            inactivity_penalty=2.0,
            prm_c=float(out_c.prm_score[0].item()),
            prm_adv=float(out_adv.prm_score[0].item()),
        )
        
        star.process_tick_trajectory(
            state_z=z_t.unsqueeze(0),
            real_future_pnl=pnl_c,
            real_future_action=target_t,
        )
        
    metrics = spin.bilateral_train_step(
        challenger_model=challenger,
        adversary_model=adversary,
        opt_challenger=opt_c,
        opt_adversary=opt_adv,
        batch_size=16,
        inactivity_weight=0.5,
        prm_weight=0.2,
    )
    star_m = star.train_step(batch_size=16)
    
    c_wr = (epoch_c_wins / max(1, epoch_c_wins + epoch_adv_wins)) * 100.0
    inact_rate = (epoch_inact / len(samples)) * 100.0
    loss_c = metrics['loss_challenger']
    loss_adv = metrics['loss_adversary']
    star_l = star_m['star_loss']
    print(f'Epoch {epoch:02d} | C-Loss: {loss_c:.3f} | Adv-Loss: {loss_adv:.3f} | Inact: {inact_rate:4.1f}% | C-WR: {c_wr:5.1f}% | STaR-Loss: {star_l:.4f}')

# Test final accuracy on all samples
correct, total = 0, 0
for z_t, target_t, _, _ in samples:
    with torch.no_grad():
        out = challenger(z_t.unsqueeze(0))
        logits = extract_action_logits(out)
        pred = int(torch.argmax(logits, dim=-1).item())
        if pred == target_t:
            correct += 1
        total += 1

print(f'\nFinal Ground-Truth Match Rate: {correct}/{total} ({correct/total*100:.1f}%)')
