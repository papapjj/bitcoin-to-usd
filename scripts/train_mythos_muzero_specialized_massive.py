"""
Specialized Dual-Champion MuZero Self-Play on Massive M1 Datasets (>500,000 Bars).
File: scripts/train_mythos_muzero_specialized_massive.py

Architecture & Strategy:
1. Massive Authentic Dataset Pools (>500,000 M1 bars from BTCUSDm.csv):
   - Bull Pool: 2023.10.15 to 2024.03.31 (243,084 bars, +165% mega bull run)
   - Bear Pool: 2022.04.01 to 2022.07.01 (132,405 bars, -58% crypto liquidation winter)
   - Chop Pool: 2023.06.20 to 2023.09.20 (133,798 bars, flat range accumulation)
   - OOS Benchmark: 2025.02.18 to 2025.02.28 (9,785 unseen bars, $12k volatility crash)

2. Specialized Self-Play (Zero Gradient Interference):
   - Bull Champion trains EXCLUSIVELY on:
     * 70% Bull Pool: Masters trend expansion, long runner dynamic TP, breakout execution.
     * 30% Chop Pool: Learns HOLD discipline to avoid fakeouts in sideways markets.
     * NEVER sees Bear crashes (prevents destruction of bullish latent features).
   - Bear Champion trains EXCLUSIVELY on:
     * 70% Bear Pool: Masters liquidation flushes, rapid short momentum, wick protection.
     * 30% Chop Pool: Learns HOLD discipline to avoid shorting bottom support.
     * NEVER sees Bull rallies (prevents destruction of bearish latent features).

3. Surgical Layer Freezing:
   - Metacognitive Reasoning Engine (4,196,564 weights) remains 100% frozen.
   - World Model State Encoder & LayerNorms remain frozen (latent representation z invariant).
   - Dynamics Transitions (g_net), Value/Reward Heads, and Dynamic Geometry Heads train specialized.

4. Dual OOS Telemetry:
   - Reports both Pure Latent Consensus and Macro EMA-100 Filtered Consensus.
   - Enforces strict Gatekeeper: WR >= 50.0%, PF >= 1.50, MaxDD <= 8.0%.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(min(4, os.cpu_count() or 4))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REASONING_DIR = ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from research_reasoning_world_class.src.real_m1_environment import (
    RealM1Environment,
    TradeResult,
)
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    AdversarialMarketStressor,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
    MetacognitiveOutput,
)
from research_reasoning_world_class.src.muzero_latent_coevolution import (
    MuZeroLatentCoEvolutionTrainer,
    SingleChampionPipeline,
    DynamicGeometryPolicyHead,
    ChampionRole,
    LatentMCTSNode,
    DualLatentMCTSSearch,
)
from scripts.train_mythos_muzero_coevolution import (
    SlicedM1Environment,
    OOSEvaluationEngine,
    OOSConfig,
    OOSMetrics,
    OOSTradeRecord,
    export_canonical_champion_checkpoint,
)
from scripts.train_mythos_muzero_volatility_finetune import (
    ActionConditionalDualMCTSSearch,
    compute_rapid_sl_rate,
)


# =============================================================================
# 3-PILLAR REFORM & POLICY ENTROPY REGULARIZATION HELPERS
# =============================================================================
def compute_trade_reward(res: TradeResult, sl_atr: float) -> float:
    """
    Pillar 1 & Pillar 3: Computes R-Multiple reward and applies Wick Fragility penalty.
    - TP (1:2) -> R = +2.0 -> tanh(R / 2.0) = +0.7616 (+0.76)
    - Normal SL -> R = -1.0 -> tanh(R / 2.0) = -0.4621 (-0.46)
    - Wick SL (holding <= 2 M1 bars) -> tanh(R / 2.0) - 0.15 = -0.6121 (-0.61)
    - Breakeven win rate: 37.76% (37.8%) guarantees positive mathematical expectancy.
    """
    sl_atr_safe = max(float(sl_atr), 0.5)
    r_multiple = float(res.return_atr) / sl_atr_safe
    base_reward = math.tanh(r_multiple / 2.0)

    exit_type_str = str(res.exit_type).lower()
    is_sl = exit_type_str in ("sl", "sl_hit") or "sl" in str(getattr(res, "exit_reason", "")).lower()
    if is_sl and res.holding_bars <= 2:
        return base_reward - 0.15
    return base_reward


def compute_hold_reward(
    env: Any,
    bar_idx: int,
    role: ChampionRole,
    lookahead_bars: int = 30,
) -> float:
    """
    Pillar 2: Computes Opportunity Cost & Discipline Reward on HOLD.
    - Missed Opportunity (> 1.5 ATR in champion direction): -0.15 * min(1.0, dir_move / 2.0)
    - Trap Avoidance (market dumps for Bull / pumps for Bear, dir_move < -0.5): +0.15
    - Patience Discipline (sideways chop |delta_atr| < 0.5): +0.05
    - Neutral Drift (0.5 <= dir_move <= 1.5): 0.0
    - Zero lookahead guarantee: ONLY used during training target creation.
    """
    total_bars = getattr(env, "total_bars", len(getattr(env, "close_arr", [])))
    if total_bars == 0 or bar_idx >= total_bars - 1:
        return 0.0

    target_idx = min(bar_idx + lookahead_bars, total_bars - 1)
    close_now = float(env.close_arr[bar_idx])
    close_future = float(env.close_arr[target_idx])
    atr = max(float(env.atr_arr[bar_idx]), 1e-4)

    delta_atr = (close_future - close_now) / atr
    dir_move = delta_atr if role == ChampionRole.BULL else -delta_atr

    if dir_move > 1.5:
        # Missed opportunity in favored direction
        return -0.15 * min(1.0, dir_move / 2.0)
    elif dir_move < -0.5:
        # Avoided adverse trap (Bull dodged dump, Bear dodged pump)
        return 0.15
    elif abs(delta_atr) < 0.5:
        # Disciplined patience in sideways chop
        return 0.05
    else:
        return 0.0


def compute_policy_entropy_loss(
    champ: Any,
    z_0: torch.Tensor,
    sl_val: float,
    tp_val: float,
) -> Tuple[torch.Tensor, float]:
    """
    Computes differentiable soft Q-policy entropy regularization over allowed actions.
    - Bull Champion: allowed actions (0, 1) [HOLD, BUY]
    - Bear Champion: allowed actions (0, 2) [HOLD, SELL]
    - pi_soft = softmax([Q(z0, a) for a in allowed])
    - H(pi) = -sum(pi * ln(pi + 1e-8))
    - L_entropy = 0.01 * (ln(2) - H(pi))
    Returns (entropy_loss, float(H(pi)))
    """
    if z_0.ndim == 1:
        z_root = z_0.unsqueeze(0)
    elif z_0.shape[0] > 1:
        z_root = z_0[0:1]
    else:
        z_root = z_0

    if hasattr(champ, "role"):
        if champ.role == ChampionRole.BULL:
            allowed = (0, 1)
        elif champ.role == ChampionRole.BEAR:
            allowed = (0, 2)
        else:
            allowed = getattr(champ.mcts, "allowed_actions", (0, 1, 2))
    elif hasattr(champ, "mcts") and hasattr(champ.mcts, "allowed_actions"):
        allowed = champ.mcts.allowed_actions
    else:
        allowed = (0, 1)

    q_list = []
    action_dim = champ.world_model.action_dim
    for act in allowed:
        act_emb = champ.world_model.encode_action(
            act, sl_atr=sl_val, tp_atr=tp_val, action_dim=action_dim, device=z_root.device
        )
        heads = champ.world_model.predict_heads(z_root, act_emb)
        next_z = champ.world_model.transition(z_root, act_emb)
        v_next = champ.world_model.value_head(next_z)
        q_val = heads["reward"] + 0.90 * torch.tanh(v_next)
        q_list.append(q_val)

    q_tensor = torch.cat(q_list, dim=-1)  # shape [1, len(allowed)]
    pi_soft = F.softmax(q_tensor, dim=-1)
    entropy = -torch.sum(pi_soft * torch.log(pi_soft + 1e-8), dim=-1).mean()
    entropy_loss = 0.01 * (math.log(float(len(allowed))) - entropy)
    return entropy_loss, float(entropy.item())


class SpecializedMassiveTrainer(MuZeroLatentCoEvolutionTrainer):
    """
    Dual-Champion MuZero Trainer with Specialized Domain Routing across Massive M1 Datasets.
    """

    def __init__(
        self,
        bull_env: RealM1Environment,
        bear_env: RealM1Environment,
        chop_env: RealM1Environment,
        lr: float = 3e-5,
        mcts_simulations: int = 4,
        beta_dyna: float = 0.5,
        sl_penalty_factor: float = 2.0,
        elevated_stress_scale: float = 0.20,
    ) -> None:
        super().__init__(
            env=bull_env,
            lr=lr,
            mcts_simulations=mcts_simulations,
            beta_dyna=beta_dyna,
        )
        self.bull_env = bull_env
        self.bear_env = bear_env
        self.chop_env = chop_env
        self.sl_penalty_factor = float(sl_penalty_factor)
        self.elevated_stress_scale = float(elevated_stress_scale)

        # Replace MCTS with ActionConditionalDualMCTSSearch
        for champ in [self.bull_champion, self.bear_champion]:
            champ.mcts = ActionConditionalDualMCTSSearch(
                world_model=champ.world_model,
                reasoning_engine=champ.reasoning_engine,
                geometry_head=champ.geometry_head,
                c_puct=1.25,
                num_simulations=mcts_simulations,
                role=champ.role,
                env=champ.env,
                beta_dyna=beta_dyna,
            )

        # Encapsulate unconditional freezing of Metacognitive Reasoning Engine (4,196,564 weights)
        if hasattr(self, "reasoning_engine") and self.reasoning_engine is not None:
            for p in self.reasoning_engine.parameters():
                p.requires_grad = False

        for champ in [self.bull_champion, self.bear_champion]:
            if hasattr(champ, "reasoning_engine") and champ.reasoning_engine is not None:
                for p in champ.reasoning_engine.parameters():
                    p.requires_grad = False
            # Ensure optimizer excludes frozen reasoning engine parameters
            trainable_params = [
                p for p in champ.world_model.parameters() if p.requires_grad
            ] + [p for p in champ.geometry_head.parameters() if p.requires_grad]
            champ.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    def _train_replay_step(self, champ: SingleChampionPipeline, batch_size: int = 4) -> None:
        if len(champ.replay_buffer) < batch_size:
            return

        batch = random.sample(champ.replay_buffer, batch_size)
        z0_batch = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=z0_batch.device)
        target_v = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=z0_batch.device).unsqueeze(1)
        target_z = torch.stack([item[3] for item in batch])

        champ.optimizer.zero_grad()
        act_embs = [
            champ.world_model.encode_action(a.item(), sl_atr=3.0, tp_atr=6.0, device=z0_batch.device)
            for a in actions
        ]
        act_tensor = torch.cat(act_embs, dim=0)

        pred_z = champ.world_model.transition(z0_batch, act_tensor)
        pred_v = champ.world_model.value_head(pred_z)

        heads = champ.world_model.predict_heads(z0_batch, act_tensor)
        pred_reward = heads["reward"]

        loss_z = F.mse_loss(pred_z, target_z)
        loss_v = F.mse_loss(pred_v, target_v)

        v0_raw = champ.world_model.value_head(z0_batch)
        v0_baseline = torch.tanh(v0_raw).detach()
        actions_col = actions.view(-1, 1)
        adv = target_v - v0_baseline
        pos_mask = ((actions_col > 0) & (adv > 0.0)).squeeze(1)

        if pos_mask.any():
            loss_r = F.smooth_l1_loss(pred_reward[pos_mask], target_v[pos_mask], beta=0.5)
        else:
            loss_r = torch.tensor(0.0, device=z0_batch.device)

        replay_loss = loss_z + 0.5 * loss_v + 1.0 * loss_r
        replay_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in champ.world_model.parameters() if p.requires_grad]
            + [p for p in champ.geometry_head.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        champ.optimizer.step()

    def run_specialized_self_play(
        self,
        num_rounds: int = 150,
        batch_size: int = 4,
        log_interval: int = 15,
    ) -> Dict[str, Any]:
        """
        Executes specialized self-play across >500k M1 bars:
        - Bull Champion samples from {Bull Pool (70%), Chop Pool (30%)}
        - Bear Champion samples from {Bear Pool (70%), Chop Pool (30%)}
        """
        round_losses: List[float] = []
        bull_entropies: List[float] = []
        bear_entropies: List[float] = []
        t0 = time.perf_counter()

        print(f"\n[Specialized Self-Play] Starting {num_rounds} rounds across >500,000 M1 bars...")
        print(f"  - Bull Champion routing: 70% Bull Mega-Run ({self.bull_env.total_bars:,} bars) + 30% Chop ({self.chop_env.total_bars:,} bars)")
        print(f"  - Bear Champion routing: 70% Bear Liquidation ({self.bear_env.total_bars:,} bars) + 30% Chop ({self.chop_env.total_bars:,} bars)")

        for round_idx in range(1, num_rounds + 1):
            round_loss_sum = 0.0

            # -----------------------------------------------------------------
            # 1. Specialized Bull Champion Step
            # -----------------------------------------------------------------
            is_bull_primary = random.random() < 0.70
            bull_active_env = self.bull_env if is_bull_primary else self.chop_env
            bull_regime_name = "Bull Mega-Run" if is_bull_primary else "Chop Accumulation"

            champ = self.bull_champion
            champ.optimizer.zero_grad()
            champ.env = bull_active_env
            champ.mcts.env = bull_active_env

            max_start = max(10, bull_active_env.total_bars - 150)
            sample_bar = random.randint(10, max_start)
            obs_list = [bull_active_env.get_observation(sample_bar + b) for b in range(batch_size)]
            obs_batch = torch.from_numpy(np.stack(obs_list)).float()

            z_0 = champ.world_model.encode_state(obs_batch)
            sl_t, tp_t, _ = champ.geometry_head(z_0)
            sl_vals = [float(x.item()) for x in sl_t]
            tp_vals = [float(x.item()) for x in tp_t]

            best_action, policy, mcts_value = champ.mcts.search(z_0[0], bar_idx=sample_bar)
            best_action = champ.filter_action(best_action)

            H = 4
            action_dim = champ.world_model.action_dim
            actions_list = []
            for b in range(batch_size):
                b_acts = [
                    champ.world_model.encode_action(
                        best_action,
                        sl_atr=sl_vals[b],
                        tp_atr=tp_vals[b],
                        action_dim=action_dim,
                        device=z_0.device,
                    )
                    for _ in range(H)
                ]
                actions_list.append(torch.cat(b_acts, dim=0))
            actions_tensor = torch.stack(actions_list, dim=0)
            pred_traj = champ.world_model.rollout(z_0, actions_tensor)

            target_val_list = []
            for b in range(batch_size):
                b_idx = sample_bar + b
                if best_action == 1:
                    res = bull_active_env.simulate_trade(
                        bar_idx=b_idx,
                        action=1,
                        sl_atr=sl_vals[b],
                        tp_atr=tp_vals[b],
                        max_holding_bars=120,
                        spread_usd=2.50,
                    )
                    target_ret = compute_trade_reward(res, sl_atr=sl_vals[b])
                else:
                    target_ret = compute_hold_reward(bull_active_env, b_idx, role=ChampionRole.BULL, lookahead_bars=30)
                target_val_list.append(target_ret)
            target_val = torch.tensor(target_val_list, dtype=torch.float32, device=z_0.device).unsqueeze(1)

            target_traj_list = []
            for b in range(batch_size):
                fut_obs = np.stack([bull_active_env.get_observation(sample_bar + b + step + 1) for step in range(H)])
                fut_t = torch.from_numpy(fut_obs).float().to(z_0.device)
                z_target_step = champ.world_model.encode_state(fut_t).detach()
                target_traj_list.append(z_target_step)
            target_z = torch.stack(target_traj_list, dim=0)

            pred_val = champ.world_model.value_head(pred_traj[:, -1])
            heads_0 = champ.world_model.predict_heads(z_0, actions_tensor[:, 0])
            pred_reward = heads_0["reward"]

            reason_out: MetacognitiveOutput = champ.reasoning_engine(pred_traj[:, -1])
            v0_raw = champ.world_model.value_head(z_0)
            v0_baseline = torch.tanh(v0_raw).detach()
            target_adv = float((target_val - v0_baseline).mean().item())

            if best_action == 1 and target_adv > 0.0:
                loss_reward = F.smooth_l1_loss(pred_reward, target_val, beta=0.5)
            else:
                loss_reward = torch.tensor(0.0, device=z_0.device)

            loss_dict = self.loss_fn(
                pred_z=pred_traj,
                target_z=target_z,
                pred_val=pred_val,
                target_val=target_val,
                prm_score=reason_out.prm_score,
                hop_rewards=reason_out.hop_rewards,
                expected_depth=float(reason_out.cot_hops.shape[1]),
            )
            bull_ent_loss, bull_ent_val = compute_policy_entropy_loss(
                champ, z_0, sl_val=sl_vals[0], tp_val=tp_vals[0]
            )
            bull_entropies.append(bull_ent_val)
            bull_loss = loss_dict["total_loss"] + 1.0 * loss_reward + bull_ent_loss
            bull_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in champ.world_model.parameters() if p.requires_grad]
                + [p for p in champ.geometry_head.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            champ.optimizer.step()
            round_loss_sum += float(bull_loss.item())

            champ.add_replay(
                z_0=z_0[0].detach(),
                action=best_action,
                target_val=float(target_val[0].item()),
                target_z=target_z[0, 0].detach(),
            )
            self._train_replay_step(champ, batch_size=batch_size)

            # -----------------------------------------------------------------
            # 2. Specialized Bear Champion Step
            # -----------------------------------------------------------------
            is_bear_primary = random.random() < 0.70
            bear_active_env = self.bear_env if is_bear_primary else self.chop_env
            bear_regime_name = "Bear Liquidation" if is_bear_primary else "Chop Distribution"

            champ = self.bear_champion
            champ.optimizer.zero_grad()
            champ.env = bear_active_env
            champ.mcts.env = bear_active_env

            max_start = max(10, bear_active_env.total_bars - 150)
            sample_bar = random.randint(10, max_start)
            obs_list = [bear_active_env.get_observation(sample_bar + b) for b in range(batch_size)]
            obs_batch = torch.from_numpy(np.stack(obs_list)).float()

            z_0 = champ.world_model.encode_state(obs_batch)
            sl_t, tp_t, _ = champ.geometry_head(z_0)
            sl_vals = [float(x.item()) for x in sl_t]
            tp_vals = [float(x.item()) for x in tp_t]

            best_action, policy, mcts_value = champ.mcts.search(z_0[0], bar_idx=sample_bar)
            best_action = champ.filter_action(best_action)

            actions_list = []
            for b in range(batch_size):
                b_acts = [
                    champ.world_model.encode_action(
                        best_action,
                        sl_atr=sl_vals[b],
                        tp_atr=tp_vals[b],
                        action_dim=action_dim,
                        device=z_0.device,
                    )
                    for _ in range(H)
                ]
                actions_list.append(torch.cat(b_acts, dim=0))
            actions_tensor = torch.stack(actions_list, dim=0)
            pred_traj = champ.world_model.rollout(z_0, actions_tensor)

            target_val_list = []
            for b in range(batch_size):
                b_idx = sample_bar + b
                if best_action == 2:
                    res = bear_active_env.simulate_trade(
                        bar_idx=b_idx,
                        action=2,
                        sl_atr=sl_vals[b],
                        tp_atr=tp_vals[b],
                        max_holding_bars=120,
                        spread_usd=2.50,
                    )
                    target_ret = compute_trade_reward(res, sl_atr=sl_vals[b])
                else:
                    target_ret = compute_hold_reward(bear_active_env, b_idx, role=ChampionRole.BEAR, lookahead_bars=30)
                target_val_list.append(target_ret)
            target_val = torch.tensor(target_val_list, dtype=torch.float32, device=z_0.device).unsqueeze(1)

            target_traj_list = []
            for b in range(batch_size):
                fut_obs = np.stack([bear_active_env.get_observation(sample_bar + b + step + 1) for step in range(H)])
                fut_t = torch.from_numpy(fut_obs).float().to(z_0.device)
                z_target_step = champ.world_model.encode_state(fut_t).detach()
                target_traj_list.append(z_target_step)
            target_z = torch.stack(target_traj_list, dim=0)

            pred_val = champ.world_model.value_head(pred_traj[:, -1])
            heads_0 = champ.world_model.predict_heads(z_0, actions_tensor[:, 0])
            pred_reward = heads_0["reward"]

            reason_out = champ.reasoning_engine(pred_traj[:, -1])
            v0_raw = champ.world_model.value_head(z_0)
            v0_baseline = torch.tanh(v0_raw).detach()
            target_adv = float((target_val - v0_baseline).mean().item())

            if best_action == 2 and target_adv > 0.0:
                loss_reward = F.smooth_l1_loss(pred_reward, target_val, beta=0.5)
            else:
                loss_reward = torch.tensor(0.0, device=z_0.device)

            loss_dict = self.loss_fn(
                pred_z=pred_traj,
                target_z=target_z,
                pred_val=pred_val,
                target_val=target_val,
                prm_score=reason_out.prm_score,
                hop_rewards=reason_out.hop_rewards,
                expected_depth=float(reason_out.cot_hops.shape[1]),
            )
            bear_ent_loss, bear_ent_val = compute_policy_entropy_loss(
                champ, z_0, sl_val=sl_vals[0], tp_val=tp_vals[0]
            )
            bear_entropies.append(bear_ent_val)
            bear_loss = loss_dict["total_loss"] + 1.0 * loss_reward + bear_ent_loss
            bear_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in champ.world_model.parameters() if p.requires_grad]
                + [p for p in champ.geometry_head.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            champ.optimizer.step()
            round_loss_sum += float(bear_loss.item())

            champ.add_replay(
                z_0=z_0[0].detach(),
                action=best_action,
                target_val=float(target_val[0].item()),
                target_z=target_z[0, 0].detach(),
            )
            self._train_replay_step(champ, batch_size=batch_size)

            avg_loss = round_loss_sum / 2.0
            round_losses.append(avg_loss)

            if round_idx % log_interval == 0 or round_idx == num_rounds:
                elapsed = time.perf_counter() - t0
                fps = round_idx / max(1e-4, elapsed)
                b_ent = bull_entropies[-1] if bull_entropies else 0.0
                br_ent = bear_entropies[-1] if bear_entropies else 0.0
                print(
                    f"  [Round {round_idx:3d}/{num_rounds}] "
                    f"Bull: {bull_regime_name[:12]} | Bear: {bear_regime_name[:12]} | "
                    f"Loss: {avg_loss:.4f} | Ent(H): Bull={b_ent:.3f}, Bear={br_ent:.3f} | "
                    f"Speed: {fps:.1f} r/s | Elapsed: {elapsed:.1f}s"
                )

        total_time = time.perf_counter() - t0
        return {
            "num_rounds": num_rounds,
            "final_loss": round_losses[-1] if round_losses else 0.0,
            "avg_loss": float(np.mean(round_losses)) if round_losses else 0.0,
            "total_time_seconds": total_time,
            "rounds_per_second": num_rounds / max(1e-4, total_time),
            "final_bull_entropy": bull_entropies[-1] if bull_entropies else 0.0,
            "final_bear_entropy": bear_entropies[-1] if bear_entropies else 0.0,
            "avg_bull_entropy": float(np.mean(bull_entropies)) if bull_entropies else 0.0,
            "avg_bear_entropy": float(np.mean(bear_entropies)) if bear_entropies else 0.0,
        }


# =============================================================================
# TREND-AWARE OOS EVALUATION RUNNER
# =============================================================================
def evaluate_trend_aware_oos(
    trainer: SpecializedMassiveTrainer,
    env: SlicedM1Environment,
    cycle_name: str,
    use_ema_filter: bool = True,
    ema_period: int = 100,
    min_confidence: float = 0.60,
    spread_usd: float = 2.50,
) -> Tuple[OOSMetrics, List[OOSTradeRecord]]:
    """
    Evaluates OOS performance with optional Macro EMA Trend Filter.
    Zero lookahead guaranteed: EMA is computed causally up to current bar.
    """
    total_env_bars = len(env)
    bar_idx = 100  # Warmup for EMA

    # Precompute causal EMA for speed
    closes = env.close_arr
    ema = np.zeros_like(closes)
    ema[0] = closes[0]
    alpha = 2.0 / (ema_period + 1.0)
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]

    equity = 10000.0
    peak_equity = 10000.0
    max_dd = 0.0
    max_dd_usd = 0.0
    trades: List[OOSTradeRecord] = []
    wins = 0
    losses = 0
    breakeven = 0
    gross_profit = 0.0
    gross_loss = 0.0

    print(f"\n[{cycle_name}] Commencing OOS Evaluation across bars [100 -> {total_env_bars:,}] (EMA Filter: {use_ema_filter})...")

    # Disable Dyna mixing during OOS
    trainer.bull_champion.mcts.beta_dyna = 0.0
    trainer.bear_champion.mcts.beta_dyna = 0.0

    with torch.inference_mode():
        while bar_idx < total_env_bars - 1:
            avail_bars = min(120, total_env_bars - 1 - bar_idx)
            if avail_bars <= 0:
                break

            obs = env.get_observation(bar_idx)
            candle_seq = env.get_candle_sequence(bar_idx, lookback=240) if hasattr(env, "get_candle_sequence") else None

            decision = trainer.resolve_consensus(
                obs_137=obs,
                current_active_positions=0,
                active_position_dir=0,
                min_confidence=min_confidence,
                conflict_margin=0.15,
                bar_idx=None,
                candle_seq=candle_seq,
            )

            raw_action = int(decision["action"])

            # Apply Macro EMA Filter if enabled
            if use_ema_filter and raw_action != 0:
                current_close = float(closes[bar_idx])
                current_ema = float(ema[bar_idx])
                if raw_action == 1 and current_close < current_ema:
                    action = 0  # Block BUY when below EMA
                elif raw_action == 2 and current_close > current_ema:
                    action = 0  # Block SELL when above EMA
                else:
                    action = raw_action
            else:
                action = raw_action

            if action == 0:
                bar_idx += 5
                continue

            sl_atr = float(np.clip(decision.get("sl_atr", 3.0), 1.0, 8.0))
            tp_atr = float(np.clip(decision.get("tp_atr", 6.0), 1.5, 15.0))
            if tp_atr < 1.5 * sl_atr:
                tp_atr = 1.5 * sl_atr

            res = env.simulate_trade(
                bar_idx=bar_idx,
                action=action,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                max_holding_bars=avail_bars,
                spread_usd=spread_usd,
            )

            holding_bars = max(1, res.holding_bars)
            pnl_usd = float(res.pnl_usd)
            equity += pnl_usd
            if equity > peak_equity:
                peak_equity = equity
            dd_usd = peak_equity - equity
            dd_pct = dd_usd / peak_equity if peak_equity > 0 else 0.0
            if dd_pct > max_dd:
                max_dd = dd_pct
                max_dd_usd = dd_usd

            if pnl_usd > 0.0:
                wins += 1
                gross_profit += pnl_usd
            elif pnl_usd < 0.0:
                losses += 1
                gross_loss += abs(pnl_usd)
            else:
                breakeven += 1

            trades.append(
                OOSTradeRecord(
                    trade_id=len(trades) + 1,
                    entry_bar=bar_idx,
                    exit_bar=bar_idx + holding_bars,
                    action=action,
                    champion=str(decision.get("active_champion", "UNKNOWN")),
                    entry_price=float(res.entry_price),
                    exit_price=float(res.exit_price),
                    sl_atr=sl_atr,
                    tp_atr=tp_atr,
                    exit_type=str(res.exit_type),
                    exit_reason=str(res.exit_reason),
                    holding_bars=holding_bars,
                    raw_pnl_pts=abs(float(res.exit_price) - float(res.entry_price)),
                    spread_paid_usd=float(spread_usd),
                    net_pnl_usd=pnl_usd,
                    return_atr=float(res.return_atr),
                    equity_after=equity,
                )
            )

            bar_idx += max(1, holding_bars)

    total_trades = len(trades)
    win_rate = wins / max(1, total_trades)
    profit_factor = gross_profit / max(1e-4, gross_loss)
    net_pnl = gross_profit - gross_loss
    avg_holding = float(np.mean([t.holding_bars for t in trades])) if trades else 0.0

    gate_passed = (
        total_trades >= 50
        and win_rate >= 0.50
        and profit_factor >= 1.50
        and max_dd <= 0.08
    )

    failure_reasons = []
    if total_trades < 50:
        failure_reasons.append(f"Trades {total_trades} < 50")
    if win_rate < 0.50:
        failure_reasons.append(f"Win Rate {win_rate:.2%} < 50.0%")
    if profit_factor < 1.50:
        failure_reasons.append(f"Profit Factor {profit_factor:.2f} < 1.50")
    if max_dd > 0.08:
        failure_reasons.append(f"Max Drawdown {max_dd:.2%} > 8.0%")

    metrics = OOSMetrics(
        cycle_name=cycle_name,
        total_trades=total_trades,
        winning_trades=wins,
        losing_trades=losses,
        breakeven_trades=breakeven,
        win_rate=win_rate,
        profit_factor=profit_factor,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_pnl,
        initial_equity=10000.0,
        final_equity=equity,
        peak_equity=peak_equity,
        max_drawdown=max_dd,
        max_drawdown_usd=max_dd_usd,
        avg_holding_bars=avg_holding,
        gate_passed=gate_passed,
        failure_reasons=failure_reasons,
    )

    status_str = "[PASSED]" if gate_passed else "[FAILED]"
    print(f"\n================================================================================")
    print(f"OOS PERFORMANCE TELEMETRY: {cycle_name} {status_str}")
    print(f"================================================================================")
    print(f"  Closed Trades:        {total_trades:>5d}")
    print(f"  Winning Trades:       {wins:>5d}  ({win_rate:.2%})")
    print(f"  Losing Trades:        {losses:>5d}")
    print(f"  Gross Profit:        ${gross_profit:>10.2f}")
    print(f"  Gross Loss:          ${gross_loss:>10.2f}")
    print(f"  Profit Factor:        {profit_factor:>6.2f}")
    print(f"  Net Profit:          ${net_pnl:>+10.2f}")
    print(f"  Max Drawdown:        {max_dd:>6.2%} (${max_dd_usd:,.2f})")
    print(f"  Avg Holding Bars:    {avg_holding:>6.1f} M1 bars")
    if failure_reasons:
        print(f"  [!] REASONS:")
        for r in failure_reasons:
            print(f"      - {r}")
    print(f"================================================================================")

    return metrics, trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Specialized Dual-Champion MuZero Self-Play on Massive M1 Data")
    parser.add_argument("--rounds", type=int, default=150, help="Self-play rounds (default: 150)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--min_confidence", type=float, default=0.60, help="Consensus confidence threshold (default: 0.60)")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test mode")
    args = parser.parse_args()

    actual_rounds = 5 if args.smoke else args.rounds
    lr = args.lr
    min_confidence = args.min_confidence

    csv_path = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
    ckpt_dir = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    canonical_ckpt = ckpt_dir / "mythos_muzero_coevolved_champion.pt"
    challenger_ckpt = ckpt_dir / "mythos_muzero_specialized_challenger.pt"

    print("=" * 80)
    print("SPECIALIZED DUAL-CHAMPION MUZERO: MASSIVE M1 SELF-PLAY CO-EVOLUTION")
    print("=" * 80)
    print(f"  Configuration:")
    print(f"    - Target Rounds:          {actual_rounds}")
    print(f"    - Learning Rate:          {lr}")
    print(f"    - Min Consensus Conf:     {min_confidence}")
    print(f"    - Master Dataset:         {csv_path}")

    # Step 1: Ingest Massive Datasets (>500,000 bars)
    print("\n[Step 1/5] Ingesting Massive M1 Historical Pools (>500,000 bars)...")
    t0 = time.time()
    env_bull = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    n_bull = env_bull.load_data("2023.10.15", "2024.03.31")
    print(f"  -> Bull Mega-Pool:     {n_bull:,} bars (2023.10 - 2024.03, +165% gain) loaded in {time.time()-t0:.2f}s")

    t0 = time.time()
    env_bear = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    n_bear = env_bear.load_data("2022.04.01", "2022.07.01")
    print(f"  -> Bear Mega-Pool:     {n_bear:,} bars (2022.04 - 2022.07, -58% crash) loaded in {time.time()-t0:.2f}s")

    t0 = time.time()
    env_chop = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    n_chop = env_chop.load_data("2023.06.20", "2023.09.20")
    print(f"  -> Chop Shared-Pool:   {n_chop:,} bars (2023.06 - 2023.09, range flat) loaded in {time.time()-t0:.2f}s")

    total_bars = n_bull + n_bear + n_chop
    print(f"  TOTAL TRAINING POOL:   {total_bars:,} authentic M1 bars")

    # Step 2: Ingest OOS Benchmark
    print("\n[Step 2/5] Ingesting Out-Of-Sample Benchmark Window (Feb 18-28, 2025)...")
    master_oos = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    master_oos.load_data("2025.01.15", "2025.02.28")
    oos_slice = SlicedM1Environment(master_oos, 55000, 64785, name="Final 10 Days OOS [55k-64.8k]")
    print(f"  OOS Benchmark active:  {len(oos_slice):,} bars")

    # Step 3: Initialize Specialized Trainer & Load Canonical Checkpoint
    print(f"\n[Step 3/5] Initializing Specialized Trainer & Loading Canonical Checkpoint...")
    trainer = SpecializedMassiveTrainer(
        bull_env=env_bull,
        bear_env=env_bear,
        chop_env=env_chop,
        lr=lr,
        mcts_simulations=4,
        beta_dyna=0.5,
        sl_penalty_factor=2.0,
        elevated_stress_scale=0.20,
    )

    if canonical_ckpt.exists():
        print(f"  Loading canonical checkpoint: {canonical_ckpt}...")
        prior = torch.load(str(canonical_ckpt), map_location="cpu", weights_only=False)
        if "reasoning_engine_state_dict" in prior:
            trainer.reasoning_engine.load_state_dict(prior["reasoning_engine_state_dict"], strict=True)
            trainer.bull_champion.reasoning_engine.load_state_dict(prior["reasoning_engine_state_dict"], strict=True)
            trainer.bear_champion.reasoning_engine.load_state_dict(prior["reasoning_engine_state_dict"], strict=True)
        if "world_model_state_dict" in prior:
            trainer.world_model.load_state_dict(prior["world_model_state_dict"], strict=True)
        if "bull_world_model_state_dict" in prior:
            trainer.bull_champion.world_model.load_state_dict(prior["bull_world_model_state_dict"], strict=True)
        if "bear_world_model_state_dict" in prior:
            trainer.bear_champion.world_model.load_state_dict(prior["bear_world_model_state_dict"], strict=True)
        if "geometry_head_state_dict" in prior:
            trainer.geometry_head.load_state_dict(prior["geometry_head_state_dict"], strict=True)
        if "bull_geometry_head_state_dict" in prior:
            trainer.bull_champion.geometry_head.load_state_dict(prior["bull_geometry_head_state_dict"], strict=True)
        if "bear_geometry_head_state_dict" in prior:
            trainer.bear_champion.geometry_head.load_state_dict(prior["bear_geometry_head_state_dict"], strict=True)
        print("  Canonical weights loaded with strict=True.")

    # Freeze representations
    for champ in [trainer.bull_champion, trainer.bear_champion]:
        for p in champ.reasoning_engine.parameters():
            p.requires_grad = False
        for name, p in champ.world_model.named_parameters():
            if "encoder" in name or "norm" in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        for p in champ.geometry_head.parameters():
            p.requires_grad = True
        champ.optimizer = torch.optim.AdamW(
            [p for p in champ.world_model.parameters() if p.requires_grad]
            + list(champ.geometry_head.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

    # Step 4: Baseline Pre-Training OOS Evaluation
    print(f"\n[Step 4/5] Evaluating Pre-Training Baselines on OOS Window...")
    base_no_ema, trades_b_no = evaluate_trend_aware_oos(
        trainer, oos_slice, "Baseline OOS (No EMA)", use_ema_filter=False, min_confidence=min_confidence
    )
    base_ema, trades_b_ema = evaluate_trend_aware_oos(
        trainer, oos_slice, "Baseline OOS (With EMA-100)", use_ema_filter=True, min_confidence=min_confidence
    )
    sl_b, tot_b, rate_b = compute_rapid_sl_rate(trades_b_ema, threshold_bars=3)
    print(f"  Baseline (With EMA-100): WR {base_ema.win_rate:.2%} | PF {base_ema.profit_factor:.2f} | NetPnL ${base_ema.net_profit:,.2f} | MaxDD {base_ema.max_drawdown:.2%} | RapidSL {rate_b:.2%}")

    # Step 5: Execute Specialized Massive Self-Play
    print(f"\n[Step 5/5] Executing {actual_rounds} Specialized Self-Play Rounds across >500k Bars...")
    t0_train = time.perf_counter()
    train_res = trainer.run_specialized_self_play(
        num_rounds=actual_rounds,
        batch_size=4,
        log_interval=max(1, actual_rounds // 10),
    )
    train_time = time.perf_counter() - t0_train
    print(f"  Specialized Training finished in {train_time:.1f}s ({train_time/60:.2f}m).")

    # Post-Training OOS Evaluation
    print(f"\nMeasuring Post-Training Performance on OOS Window...")
    post_no_ema, trades_p_no = evaluate_trend_aware_oos(
        trainer, oos_slice, "Post-Train OOS (No EMA)", use_ema_filter=False, min_confidence=min_confidence
    )
    post_ema, trades_p_ema = evaluate_trend_aware_oos(
        trainer, oos_slice, "Post-Train OOS (With EMA-100)", use_ema_filter=True, min_confidence=min_confidence
    )
    sl_p, tot_p, rate_p = compute_rapid_sl_rate(trades_p_ema, threshold_bars=3)

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION: SPECIALIZED SELF-PLAY (WITH EMA-100 FILTER)")
    print("=" * 80)
    print(f"  Metric                     Baseline          Post-Specialized  Change")
    print(f"  -------------------------  ----------------  ----------------  ------------")
    print(f"  Total Trades:              {base_ema.total_trades:>14d}  {post_ema.total_trades:>14d}  {post_ema.total_trades - base_ema.total_trades:>+10d}")
    print(f"  Win Rate:                  {base_ema.win_rate:>14.2%}  {post_ema.win_rate:>14.2%}  {post_ema.win_rate - base_ema.win_rate:>+10.2%}")
    print(f"  Profit Factor:             {base_ema.profit_factor:>14.2f}  {post_ema.profit_factor:>14.2f}  {post_ema.profit_factor - base_ema.profit_factor:>+10.2f}")
    print(f"  Max Drawdown:              {base_ema.max_drawdown:>14.2%}  {post_ema.max_drawdown:>14.2%}  {post_ema.max_drawdown - base_ema.max_drawdown:>+10.2%}")
    print(f"  Net Profit:                ${base_ema.net_profit:>13,.2f}  ${post_ema.net_profit:>13,.2f}  ${post_ema.net_profit - base_ema.net_profit:>+11,.2f}")
    print(f"  Rapid SL Rate (<=3 bars):  {rate_b:>14.2%}  {rate_p:>14.2%}  {rate_p - rate_b:>+10.2%}")
    print("=" * 80)

    # Gatekeeper Decision
    promote = (
        post_ema.gate_passed
        and (post_ema.profit_factor >= base_ema.profit_factor or post_ema.net_profit >= base_ema.net_profit)
        and post_ema.max_drawdown <= base_ema.max_drawdown + 0.01
    )

    if promote:
        target_out = canonical_ckpt
        status_tag = "PROMOTED_CHAMPION"
        print(f"\n[GATE PASSED] Challenger surpassed baseline criteria -> PROMOTED to Canonical Champion!")
    else:
        target_out = challenger_ckpt
        status_tag = "SAVED_CHALLENGER"
        print(f"\n[GATE RETAINED] Baseline remains Champion -> Exporting Challenger to {challenger_ckpt.name}")

    export_canonical_champion_checkpoint(
        trainer=trainer,
        checkpoint_path=target_out,
        oos_stats={
            "win_rate": post_ema.win_rate,
            "profit_factor": post_ema.profit_factor,
            "net_profit": post_ema.net_profit,
            "max_drawdown": post_ema.max_drawdown,
            "total_trades": post_ema.total_trades,
            "status": status_tag,
            "specialized_datasets": f"Bull: {n_bull}, Bear: {n_bear}, Chop: {n_chop}",
        },
    )


if __name__ == "__main__":
    main()
