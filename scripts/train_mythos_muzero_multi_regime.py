"""
Multi-Regime Diverse M1 Self-Play Co-Evolution Runner for Dual-Champion MuZero.
File: scripts/train_mythos_muzero_multi_regime.py

Blueprint:
1. Multi-Regime Ingestion across 7 Years of Authentic M1 Data (BTCUSDm.csv):
   - Regime 1: Explosive Bull Run (2024.11.05 -> 2024.11.25, +37.2% gain, ~30,235 bars)
   - Regime 2: Bear Panic Dump    (2022.06.05 -> 2022.06.25, -28.1% drop, ~30,182 bars)
   - Regime 3: Chop Accumulation  (2023.07.05 -> 2023.07.25,  -5.0% range, ~30,231 bars)
   - Regime 4: Out-Of-Sample OOS  (2025.02.18 -> 2025.02.28, 9,785 bars, $12k volatility crash)
2. Stratified Multi-Regime Self-Play Co-Evolution:
   - Alternating multi-environment rollouts ensuring Bull Champion masters uptrend expansion,
     Bear Champion masters sell-off momentum, and both master HOLD discipline during chop.
3. Action-Conditional Latent MCTS:
   - Evaluates Q(s, a) = R_wm(s, a) + gamma * V_wm(s') with process reward model guidance.
4. Zero-Lookahead OOS Benchmark:
   - Evaluated on authentic 9,785 M1 bars with $2.50 spread and path-dependent extreme checking.
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

# Constrain OpenMP/MKL thread thrashing
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
# 1. MULTI-REGIME DIVERSE MUZERO TRAINER
# =============================================================================
class MultiRegimeMuZeroTrainer(MuZeroLatentCoEvolutionTrainer):
    """
    Extends Dual-Champion Co-Evolution Trainer to sample transitions across
    distinct, contrasting market regimes (Bull Trend, Bear Dump, Sideway Chop).
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
    ):
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

        # Replace default MCTS search with Action-Conditional search
        for champ in [self.bull_champion, self.bear_champion]:
            champ.mcts = ActionConditionalDualMCTSSearch(
                world_model=champ.world_model,
                reasoning_engine=champ.reasoning_engine,
                role=champ.role,
                geometry_head=champ.geometry_head,
                env=champ.env,
                adversary_stressor=champ.adversary,
                num_simulations=mcts_simulations,
                beta_dyna=beta_dyna,
            )

    def run_multi_regime_self_play(
        self,
        num_rounds: int = 150,
        batch_size: int = 4,
        log_interval: int = 15,
    ) -> Dict[str, Any]:
        """
        Executes self-play co-evolution rounds by cycling through:
        1. Explosive Bull Regime (Uptrend mechanics)
        2. Bear Panic Dump Regime (Crash / Downward mechanics)
        3. Low-Volatility Chop Regime (Patience / HOLD mechanics)
        """
        all_regimes = [
            ("Bull Regime", self.bull_env),
            ("Bear Regime", self.bear_env),
            ("Chop Regime", self.chop_env),
        ]

        t0 = time.perf_counter()
        round_losses: List[float] = []

        print(f"\n[Multi-Regime Co-Evolution] Starting {num_rounds} self-play rounds across 3 contrasting regimes...")

        for round_idx in range(1, num_rounds + 1):
            # Rotate regime each round for balanced exposure
            regime_name, active_env = all_regimes[(round_idx - 1) % len(all_regimes)]

            # Choose random sample bar inside active environment
            max_start = max(10, active_env.total_bars - 150)
            sample_bar = random.randint(10, max_start)

            # Sample observation batch
            obs_list = [active_env.get_observation(sample_bar + b) for b in range(batch_size)]
            obs_batch = torch.from_numpy(np.stack(obs_list)).float()

            round_loss_sum = 0.0

            # Alternating updates for Bull and Bear Champions
            for champ_idx, champ in enumerate([self.bull_champion, self.bear_champion]):
                champ.optimizer.zero_grad()
                champ.env = active_env
                champ.mcts.env = active_env

                # 1. Encode root state
                z_0 = champ.world_model.encode_state(obs_batch)

                # 2. Dynamic Geometry SL/TP
                sl_t, tp_t, _ = champ.geometry_head(z_0)
                sl_vals = [float(x.item()) for x in sl_t]
                tp_vals = [float(x.item()) for x in tp_t]

                # 3. Action Selection via MCTS
                best_action, policy, mcts_value = champ.mcts.search(z_0[0], bar_idx=sample_bar)
                best_action = champ.filter_action(best_action)

                # 4. Multi-step latent rollout with H=4 steps
                H = 4
                action_dim = champ.world_model.action_dim
                actions_list = []
                for b in range(batch_size):
                    b_acts = []
                    for step in range(H):
                        emb = champ.world_model.encode_action(
                            best_action,
                            sl_atr=sl_vals[b],
                            tp_atr=tp_vals[b],
                            action_dim=action_dim,
                            device=z_0.device,
                        )
                        b_acts.append(emb)
                    actions_list.append(torch.cat(b_acts, dim=0))
                actions_tensor = torch.stack(actions_list, dim=0)
                pred_traj = champ.world_model.rollout(z_0, actions_tensor)

                # 5. Non-Circular Authentic Targets with Path-Dependent Simulation
                target_val_list = []
                for b in range(batch_size):
                    if best_action in (1, 2):
                        res = active_env.simulate_trade(
                            bar_idx=sample_bar + b,
                            action=best_action,
                            sl_atr=sl_vals[b],
                            tp_atr=tp_vals[b],
                            max_holding_bars=120,
                            spread_usd=2.50,
                        )
                        ret_atr = float(res.return_atr)
                        exit_type_str = str(res.exit_type).lower()
                        is_sl = exit_type_str in ("sl", "sl_hit") or "sl" in str(res.exit_reason).lower()

                        # Asymmetric penalty for rapid SL wick hit
                        if is_sl and res.holding_bars <= 3:
                            penalized_ret = -abs(ret_atr) * self.sl_penalty_factor
                            target_ret = -abs(math.tanh(abs(penalized_ret) / 3.0))
                        else:
                            target_ret = math.tanh(ret_atr / 3.0)
                    else:
                        target_ret = 0.0
                    target_val_list.append(target_ret)

                target_val = torch.tensor(target_val_list, dtype=torch.float32, device=z_0.device).unsqueeze(1)

                # Target future latent trajectory across H steps
                target_traj_list = []
                for b in range(batch_size):
                    fut_obs = np.stack([active_env.get_observation(sample_bar + b + step + 1) for step in range(H)])
                    fut_t = torch.from_numpy(fut_obs).float().to(z_0.device)
                    z_target_step = champ.world_model.encode_state(fut_t).detach()
                    target_traj_list.append(z_target_step)
                target_z = torch.stack(target_traj_list, dim=0)

                # 6. Prediction Heads
                pred_val = champ.world_model.value_head(pred_traj[:, -1])
                heads_0 = champ.world_model.predict_heads(z_0, actions_tensor[:, 0])
                pred_reward = heads_0["reward"]

                # 7. Reasoning Engine PRM
                reason_out: MetacognitiveOutput = champ.reasoning_engine(pred_traj[:, -1])
                prm_val = float(reason_out.prm_score.mean().item())

                # Option B Positive-Advantage Selective Reward Training
                v0_raw = champ.world_model.value_head(z_0)
                v0_baseline = torch.tanh(v0_raw).detach()
                target_adv = float((target_val - v0_baseline).mean().item())

                is_positive = (best_action in (1, 2)) and (target_adv > 0.0)
                if is_positive:
                    loss_reward = F.smooth_l1_loss(pred_reward, target_val, beta=0.5)
                else:
                    loss_reward = torch.tensor(0.0, device=z_0.device)

                # Multi-task loss
                loss_dict = self.loss_fn(
                    pred_z=pred_traj,
                    target_z=target_z,
                    pred_val=pred_val,
                    target_val=target_val,
                    prm_score=reason_out.prm_score,
                    hop_rewards=reason_out.hop_rewards,
                    expected_depth=float(reason_out.cot_hops.shape[1]),
                )

                total_loss = loss_dict["total_loss"] + 1.0 * loss_reward
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in champ.world_model.parameters() if p.requires_grad]
                    + [p for p in champ.geometry_head.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                champ.optimizer.step()

                round_loss_sum += float(total_loss.item())

                # Add to replay buffer
                champ.add_replay(
                    z_0=z_0[0].detach(),
                    action=best_action,
                    target_val=float(target_val[0].item()),
                    target_z=target_z[0, 0].detach(),
                )

                # Sample experience replay
                self._train_replay_step(champ, batch_size=batch_size)

            avg_loss = round_loss_sum / 2.0
            round_losses.append(avg_loss)

            if round_idx % log_interval == 0 or round_idx == num_rounds:
                elapsed = time.perf_counter() - t0
                fps = round_idx / max(1e-4, elapsed)
                print(
                    f"  [Round {round_idx:>3d}/{num_rounds}] "
                    f"Regime: {regime_name:<11s} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Speed: {fps:.1f} r/s | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        return {
            "num_rounds": num_rounds,
            "final_loss": round_losses[-1] if round_losses else 0.0,
            "mean_loss": float(np.mean(round_losses)) if round_losses else 0.0,
            "total_time": time.perf_counter() - t0,
        }

    def _train_replay_step(self, champion: SingleChampionPipeline, batch_size: int = 4) -> None:
        """Samples experience replay buffer and updates latent dynamics."""
        if len(champion.replay_buffer) < max(batch_size, 8):
            return
        sampled = champion.sample_replay_batch(batch_size=batch_size)
        if sampled is None:
            return
        z_0, actions, target_v, target_z = sampled

        act_emb = champion.world_model._prepare_action(actions, z_0)
        pred_z = champion.world_model.transition(z_0, act_emb)

        # Loss dynamics
        loss_dyn = F.mse_loss(pred_z, target_z)

        # Selective positive reward loss
        actions_col = actions.view(-1, 1).to(device=z_0.device)
        v0_baseline = torch.tanh(champion.world_model.value_head(z_0)).detach()
        adv = target_v - v0_baseline  # shape [B, 1]
        pos_mask = ((actions_col > 0) & (adv > 0.0)).squeeze(1)  # shape [B]

        if pos_mask.any():
            heads_0 = champion.world_model.predict_heads(z_0, act_emb)
            loss_r = F.smooth_l1_loss(heads_0["reward"][pos_mask], target_v[pos_mask], beta=0.5)
            loss_total = loss_dyn + 0.5 * loss_r
        else:
            loss_total = loss_dyn

        champion.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in champion.world_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        champion.optimizer.step()


# =============================================================================
# 2. MAIN ORCHESTRATION PIPELINE
# =============================================================================
def run_multi_regime_pipeline(
    csv_path: str = r"C:\Users\Admin\Desktop\BTCUSDm.csv",
    rounds: int = 120,
    lr: float = 3e-5,
    min_confidence: float = 0.60,
    ema_regime_period: int = 100,
    smoke: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrates the entire Multi-Regime Diverse M1 Self-Play training and evaluation.
    """
    actual_rounds = 5 if smoke else rounds
    print("=" * 80)
    print("DUAL-CHAMPION MUZERO: MULTI-REGIME DIVERSE M1 SELF-PLAY CO-EVOLUTION")
    print("=" * 80)
    print(f"  Configuration:")
    print(f"    - Target Rounds:          {actual_rounds}")
    print(f"    - Learning Rate:          {lr}")
    print(f"    - Min Consensus Conf:     {min_confidence:.2f}")
    print(f"    - Macro EMA Period:       {ema_regime_period}")
    print(f"    - Master Dataset:         {csv_path}")

    ckpt_file = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
    backup_file = ckpt_file.with_suffix(".pt.multi_regime_bak")
    challenger_file = ckpt_file.parent / "mythos_muzero_multiregime_challenger.pt"

    # Step 1: Load 3 Diverse Training Regimes
    print(f"\n[Step 1/5] Ingesting 3 Distinct Market Regimes from BTCUSDm.csv...")
    
    print("  -> Loading Regime 1: Explosive Bull (2024.11.05 to 2024.11.25)...")
    env_bull = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    bars_bull = env_bull.load_data("2024.11.05", "2024.11.25")
    print(f"     Regime 1 loaded: {bars_bull:,} bars (Gain: +37.2%)")

    print("  -> Loading Regime 2: Bear Panic Dump (2022.06.05 to 2022.06.25)...")
    env_bear = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    bars_bear = env_bear.load_data("2022.06.05", "2022.06.25")
    print(f"     Regime 2 loaded: {bars_bear:,} bars (Drop: -28.1%)")

    print("  -> Loading Regime 3: Low-Vol Chop Accumulation (2023.07.05 to 2023.07.25)...")
    env_chop = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    bars_chop = env_chop.load_data("2023.07.05", "2023.07.25")
    print(f"     Regime 3 loaded: {bars_chop:,} bars (Range: -5.0%)")

    # Step 2: Load Out-Of-Sample Benchmark (Final 10 Days of Feb 2025)
    print("\n[Step 2/5] Ingesting Out-Of-Sample (OOS) Benchmark Window (Feb 2025)...")
    master_oos_env = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    master_oos_env.load_data("2025.01.15", "2025.02.28")
    oos_slice = SlicedM1Environment(master_oos_env, 55000, 64785, name="Final 10 Days OOS [55k-64.8k]")
    print(f"  OOS Benchmark active: {len(oos_slice):,} bars (Feb 18 to Feb 28, 2025)")

    # Step 3: Initialize Trainer & Load Baseline Checkpoint
    print(f"\n[Step 3/5] Initializing Multi-Regime Trainer & Loading Baseline Checkpoint...")
    trainer = MultiRegimeMuZeroTrainer(
        bull_env=env_bull,
        bear_env=env_bear,
        chop_env=env_chop,
        lr=lr,
        mcts_simulations=4,
        beta_dyna=0.5,
        sl_penalty_factor=2.0,
        elevated_stress_scale=0.20,
    )

    if ckpt_file.exists():
        print(f"  Loading canonical checkpoint: {ckpt_file}...")
        prior_ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        if "reasoning_engine_state_dict" in prior_ckpt:
            trainer.reasoning_engine.load_state_dict(prior_ckpt["reasoning_engine_state_dict"], strict=True)
            trainer.bull_champion.reasoning_engine.load_state_dict(prior_ckpt["reasoning_engine_state_dict"], strict=True)
            trainer.bear_champion.reasoning_engine.load_state_dict(prior_ckpt["reasoning_engine_state_dict"], strict=True)
        if "world_model_state_dict" in prior_ckpt:
            trainer.world_model.load_state_dict(prior_ckpt["world_model_state_dict"], strict=True)
        if "bull_world_model_state_dict" in prior_ckpt:
            trainer.bull_champion.world_model.load_state_dict(prior_ckpt["bull_world_model_state_dict"], strict=True)
        if "bear_world_model_state_dict" in prior_ckpt:
            trainer.bear_champion.world_model.load_state_dict(prior_ckpt["bear_world_model_state_dict"], strict=True)
        if "geometry_head_state_dict" in prior_ckpt:
            trainer.geometry_head.load_state_dict(prior_ckpt["geometry_head_state_dict"], strict=True)
        if "bull_geometry_head_state_dict" in prior_ckpt:
            trainer.bull_champion.geometry_head.load_state_dict(prior_ckpt["bull_geometry_head_state_dict"], strict=True)
        if "bear_geometry_head_state_dict" in prior_ckpt:
            trainer.bear_champion.geometry_head.load_state_dict(prior_ckpt["bear_geometry_head_state_dict"], strict=True)
        print("  Checkpoint loaded successfully with strict=True.")

        if not backup_file.exists():
            shutil.copyfile(str(ckpt_file), str(backup_file))
            print(f"  Created backup file: {backup_file.name}")
        else:
            print(f"  Pre-existing backup preserved: {backup_file.name}")

    # Surgical Freezing: Keep reasoning engine and state encoder intact (preserves representation z),
    # while training dynamics transitions, value/reward heads, and dynamic geometry heads across regimes.
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

    # Pre-train baseline evaluation on OOS
    print(f"\n[Step 4/5] Evaluating Pre-Training Baseline on Unseen OOS Window...")
    oos_cfg = OOSConfig(
        min_confidence=min_confidence,
        eval_stride=5,
        spread_usd=2.50,
        max_holding_bars=120,
        min_trades=20 if smoke else 50,
        min_win_rate=0.50,
        min_profit_factor=1.5,
        max_drawdown=0.08,
    )

    baseline_metrics, baseline_trades = OOSEvaluationEngine.evaluate_window(
        trainer=trainer,
        env=oos_slice,
        cycle_name="Baseline Pre-Training OOS",
        config=oos_cfg,
        max_trades=30 if smoke else None,
    )
    b_sl, b_tot, b_rate = compute_rapid_sl_rate(baseline_trades, threshold_bars=3)
    print(f"  Baseline OOS Results:")
    print(f"    - Trades: {baseline_metrics.total_trades} | WR: {baseline_metrics.win_rate:.2%} | PF: {baseline_metrics.profit_factor:.2f} | NetPnL: ${baseline_metrics.net_profit:,.2f} | MaxDD: {baseline_metrics.max_drawdown:.2%} | RapidSL: {b_sl}/{b_tot} ({b_rate:.2%})")

    # Step 5: Execute Multi-Regime Self-Play
    print(f"\n[Step 5/5] Executing {actual_rounds} Multi-Regime Diverse Self-Play Rounds...")
    t0_train = time.perf_counter()
    train_results = trainer.run_multi_regime_self_play(
        num_rounds=actual_rounds,
        batch_size=4,
        log_interval=max(1, actual_rounds // 10),
    )
    train_time = time.perf_counter() - t0_train
    print(f"  Training finished in {train_time:.1f}s ({train_time/60:.2f}m).")

    # Post-training evaluation on OOS
    print(f"\nMeasuring Post-Training Performance on Unseen OOS Window...")
    post_metrics, post_trades = OOSEvaluationEngine.evaluate_window(
        trainer=trainer,
        env=oos_slice,
        cycle_name="Post Multi-Regime OOS",
        config=oos_cfg,
        max_trades=30 if smoke else None,
    )
    p_sl, p_tot, p_rate = compute_rapid_sl_rate(post_trades, threshold_bars=3)

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION: PRE vs POST MULTI-REGIME SELF-PLAY")
    print("=" * 80)
    print(f"  Metric                     Baseline          Post Multi-Regime Change")
    print(f"  -------------------------  ----------------  ----------------  ------------")
    print(f"  Total Trades:              {baseline_metrics.total_trades:>14d}  {post_metrics.total_trades:>14d}  {post_metrics.total_trades - baseline_metrics.total_trades:>+10d}")
    print(f"  Win Rate:                  {baseline_metrics.win_rate:>14.2%}  {post_metrics.win_rate:>14.2%}  {post_metrics.win_rate - baseline_metrics.win_rate:>+10.2%}")
    print(f"  Profit Factor:             {baseline_metrics.profit_factor:>14.2f}  {post_metrics.profit_factor:>14.2f}  {post_metrics.profit_factor - baseline_metrics.profit_factor:>+10.2f}")
    print(f"  Max Drawdown:              {baseline_metrics.max_drawdown:>14.2%}  {post_metrics.max_drawdown:>14.2%}  {post_metrics.max_drawdown - baseline_metrics.max_drawdown:>+10.2%}")
    print(f"  Net Profit:                ${baseline_metrics.net_profit:>13,.2f}  ${post_metrics.net_profit:>13,.2f}  ${post_metrics.net_profit - baseline_metrics.net_profit:>+10,.2f}")
    print(f"  Rapid SL Rate (<=3 bars):  {b_rate:>14.2%}  {p_rate:>14.2%}  {p_rate - b_rate:>+10.2%}")
    print("=" * 80)

    # Export Checkpoint
    gate_tr = post_metrics.total_trades >= (20 if smoke else 50)
    gate_pf = post_metrics.profit_factor >= baseline_metrics.profit_factor or post_metrics.profit_factor >= 1.50
    gate_dd = post_metrics.max_drawdown <= 0.08

    promote = gate_tr and gate_pf and gate_dd
    target_out = ckpt_file if promote else challenger_file
    print(f"\n[Export] Target checkpoint file: {target_out.name} ({'PROMOTED_CHAMPION' if promote else 'SAVED_CHALLENGER'})")

    export_canonical_champion_checkpoint(
        trainer=trainer,
        checkpoint_path=target_out,
        oos_stats={
            "win_rate": post_metrics.win_rate,
            "profit_factor": post_metrics.profit_factor,
            "max_drawdown": post_metrics.max_drawdown,
            "total_trades": post_metrics.total_trades,
            "rapid_sl_count": p_sl,
            "rapid_sl_rate": p_rate,
        },
    )

    return {
        "baseline_metrics": baseline_metrics,
        "post_metrics": post_metrics,
        "promoted": promote,
        "target_file": str(target_out),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Regime Diverse M1 Self-Play Co-Evolution Runner")
    parser.add_argument("--rounds", type=int, default=120, help="Number of self-play rounds across regimes")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for world model & geometry")
    parser.add_argument("--min_confidence", type=float, default=0.60, help="MCTS min confidence threshold")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke run for verification")
    args = parser.parse_args()

    run_multi_regime_pipeline(
        rounds=args.rounds,
        lr=args.lr,
        min_confidence=args.min_confidence,
        smoke=args.smoke,
    )
