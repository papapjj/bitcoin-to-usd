"""
High Volatility Defense & Anti-Wick Fine-Tuning Runner for Dual-Champion MuZero.
File: scripts/train_mythos_muzero_volatility_finetune.py

Blueprint:
1. Mode: Continuous Fine-Tuning from outputs/he_chuyen_gia/v193_runs/mythos_muzero_coevolved_champion.pt
2. Objective: High Volatility Defense & Anti-Wick Liquidity Sweep Protection
3. Action-Conditional MCTS:
   - Evaluates Q(s, a) = R_wm(s, a) + gamma * V_wm(s') in zero-lookahead OOS
   - Leverages trained reward_head(z, a) for decisive, high-conviction trade execution
4. Asymmetric SL Penalty:
   - 2.0x penalty for rapid stop-outs (holding_bars <= 3)
   - Elevated adversarial stressor (stress_scale = 0.25)
5. Data Partition:
   - Train Window: Bars [30,000 : 55,000] (~25,000 bars from Feb 2025 peak volatility)
   - OOS Window:   Bars [55,000 : 64,785] (~9,785 bars, final 10 days of Feb 2025)
6. Strict Acceptance Gate:
   - Win Rate >= 80%
   - Profit Factor >= 3.0
   - Max Drawdown <= 8%
   - Rapid SL hit reduction >= 50%
7. Safety: Pre-training backup of champion checkpoint; promote only if strict gates pass,
   else save safely to mythos_muzero_volatility_challenger.pt.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Constrain OpenMP/MKL thread thrashing
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
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


# =============================================================================
# 1. ACTION-CONDITIONAL DUAL MCTS SEARCH
# =============================================================================
class ActionConditionalDualMCTSSearch(DualLatentMCTSSearch):
    """
    Enhanced Dual Latent MCTS with Action-Conditional Predicted Return Grounding.
    Eliminates zero-lookahead MUTUAL_HOLD deadlock by computing:
        Q(s, a) = R_wm(s, a) + gamma * V_wm(s')
    where R_wm(s, a) is predicted by the world model's reward_head.
    """

    def search(
        self,
        root_z: torch.Tensor,
        bar_idx: Optional[int] = None,
        return_value: bool = True,
        root_prm_val: Optional[float] = None,
    ) -> Union[Tuple[int, Dict[int, float], float], Tuple[int, Dict[int, float]]]:
        if root_z.ndim == 1:
            root_z = root_z.unsqueeze(0)

        root = LatentMCTSNode(z=root_z, bar_idx=bar_idx)
        K = len(self.allowed_actions)
        child_prior = 1.0 / max(1, K)

        with torch.inference_mode():
            # 1. Root Metacognitive Reasoning Engine
            if root_prm_val is None:
                try:
                    root_reason_out = self.reasoning_engine(root_z)
                    root_prm_val = float(root_reason_out.prm_score.mean().item())
                except Exception:
                    root_prm_val = 0.5

            # 2. Dynamic Geometry SL/TP
            root_sl, root_tp = 3.0, 6.0
            if self.geometry_head is not None:
                sl_t, tp_t, _ = self.geometry_head(root_z)
                root_sl = float(sl_t.mean().item())
                root_tp = float(tp_t.mean().item())
                act_batch = torch.cat([
                    self.world_model.encode_action(act, sl_atr=root_sl, tp_atr=root_tp, device=root_z.device)
                    for act in self.allowed_actions
                ], dim=0)
            else:
                act_batch = self._cached_default_act_batch.to(device=root_z.device)

            root.sl_atr = root_sl
            root.tp_atr = root_tp

            # 3. MCTS Simulation Loop
            for _ in range(self.num_simulations):
                node = root
                search_path = [node]

                # 3.1 Selection
                while node.children:
                    best_action = -1
                    best_ucb = -1e9
                    sqrt_visits = math.sqrt(node.visit_count)
                    for act, child in node.children.items():
                        u = self.c_puct * child.prior * (sqrt_visits / (1 + child.visit_count))
                        ucb = child.value + u
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_action = act
                    node = node.children[best_action]
                    search_path.append(node)

                # 3.2 Expansion & Leaf Evaluation
                if node.visit_count == 0 or not node.children:
                    raw_val = float(self.world_model.value_head(node.z).mean().item())
                    v_wm = math.tanh(raw_val)
                    v_model = max(-1.0, min(1.0, 0.5 * v_wm + 0.5 * root_prm_val))

                    total_bars = getattr(self.env, "total_bars", len(getattr(self.env, "closes", [])))
                    decision_bar = (
                        node.decision_bar_idx
                        if node.decision_bar_idx is not None
                        else (node.parent.bar_idx if node.parent is not None else node.bar_idx)
                    )
                    has_env_future = (
                        self.env is not None
                        and decision_bar is not None
                        and total_bars > 0
                        and decision_bar + 1 < total_bars
                    )

                    if has_env_future:
                        # Dyna ground truth from future candle simulation
                        if node.action_from_parent is not None and node.action_from_parent != 0:
                            res = self.env.simulate_trade(
                                bar_idx=decision_bar,
                                action=node.action_from_parent,
                                sl_atr=node.sl_atr,
                                tp_atr=node.tp_atr,
                                max_holding_bars=120,
                            )
                            v_env = math.tanh(float(res.return_atr) / 3.0)
                        elif node.action_from_parent == 0:
                            v_env = 0.0
                        else:
                            v_env = v_model
                        q_eval = self.beta_dyna * v_env + (1.0 - self.beta_dyna) * v_model
                    else:
                        # Zero-lookahead OOS: Bellman Q(s, a) = R_wm(s, a) + 0.90 * V_wm(s')
                        if node.action_from_parent is not None and node.parent is not None:
                            if node.action_from_parent == 0:
                                r_pred = 0.0
                            else:
                                act_emb = self.encode_action_vector(
                                    node.action_from_parent,
                                    sl_atr=node.sl_atr,
                                    tp_atr=node.tp_atr,
                                    device=node.parent.z.device,
                                )
                                r_hat = self.world_model.predict_heads(node.parent.z, act_emb)["reward"]
                                r_pred = math.tanh(float(r_hat.mean().item()))
                            q_eval = r_pred + 0.90 * v_model
                        elif node.action_from_parent == 0:
                            q_eval = 0.90 * v_model
                        else:
                            q_eval = v_model

                    q_eval = max(-1.0, min(1.0, q_eval))

                    # Vectorized Transition Batch
                    z_batch = node.z.repeat(K, 1)
                    next_zs = self.world_model.transition(z_batch, act_batch)
                    next_bar = (node.bar_idx + 1) if node.bar_idx is not None else None

                    for i, act in enumerate(self.allowed_actions):
                        node.children[act] = LatentMCTSNode(
                            z=next_zs[i : i + 1],
                            bar_idx=next_bar,
                            parent=node,
                            prior=child_prior,
                            action_from_parent=act,
                            sl_atr=root_sl,
                            tp_atr=root_tp,
                            decision_bar_idx=node.bar_idx,
                        )
                else:
                    q_eval = node.value

                # 3.3 Backpropagation
                for n in reversed(search_path):
                    n.visit_count += 1
                    n.total_value += q_eval

        # Final action selection
        total_visits = sum(c.visit_count for c in root.children.values())
        policy = {act: c.visit_count / max(1, total_visits) for act, c in root.children.items()}
        selected_act = max(root.children.keys(), key=lambda a: root.children[a].visit_count)

        if return_value:
            return selected_act, policy, root.value
        return selected_act, policy


# =============================================================================
# 2. VOLATILITY DEFENSE MUZERO TRAINER
# =============================================================================
class VolatilityDefenseMuZeroTrainer(MuZeroLatentCoEvolutionTrainer):
    """
    MuZero Co-Evolution Trainer augmented with:
    - ActionConditionalDualMCTSSearch
    - Asymmetric SL Penalty (2.0x) on rapid stop-outs (holding_bars <= 3)
    - Online reward_head training against trade outcomes
    - Replay buffer reward_head training
    - Elevated Adversarial Market Stressor (stress_scale = 0.25)
    """

    def __init__(
        self,
        *args,
        sl_penalty_factor: float = 2.0,
        elevated_stress_scale: float = 0.25,
        mcts_simulations: int = 4,
        beta_dyna: float = 0.5,
        **kwargs,
    ):
        self.mcts_simulations = int(mcts_simulations)
        self.beta_dyna = float(beta_dyna)
        super().__init__(*args, mcts_simulations=mcts_simulations, beta_dyna=beta_dyna, **kwargs)
        self.sl_penalty_factor = float(sl_penalty_factor)
        self.elevated_stress_scale = float(elevated_stress_scale)

        # Elevate stress scale across adversaries
        if hasattr(self, "adversary") and self.adversary is not None:
            self.adversary.stress_scale = self.elevated_stress_scale
        if hasattr(self.bull_champion, "adversary") and self.bull_champion.adversary is not None:
            self.bull_champion.adversary.stress_scale = self.elevated_stress_scale
        if hasattr(self.bear_champion, "adversary") and self.bear_champion.adversary is not None:
            self.bear_champion.adversary.stress_scale = self.elevated_stress_scale

        # Upgrade Bull and Bear MCTS to ActionConditionalDualMCTSSearch
        self.bull_champion.mcts = ActionConditionalDualMCTSSearch(
            world_model=self.bull_champion.world_model,
            reasoning_engine=self.bull_champion.reasoning_engine,
            role=ChampionRole.BULL,
            geometry_head=self.bull_champion.geometry_head,
            env=self.env,
            adversary_stressor=self.bull_champion.adversary,
            c_puct=1.25,
            num_simulations=self.mcts_simulations,
            beta_dyna=self.beta_dyna,
        )
        self.bear_champion.mcts = ActionConditionalDualMCTSSearch(
            world_model=self.bear_champion.world_model,
            reasoning_engine=self.bear_champion.reasoning_engine,
            role=ChampionRole.BEAR,
            geometry_head=self.bear_champion.geometry_head,
            env=self.env,
            adversary_stressor=self.bear_champion.adversary,
            c_puct=1.25,
            num_simulations=self.mcts_simulations,
            beta_dyna=self.beta_dyna,
        )

    def _step_single_champion(
        self,
        champion: SingleChampionPipeline,
        z_0: torch.Tensor,
        bar_idx: Optional[int],
        use_env: Any,
        H: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes one self-play co-evolution step with Asymmetric SL Penalty
        and trains reward_head on realized trade returns.
        """
        champion.optimizer.zero_grad()
        batch_size = z_0.shape[0]
        action_dim = 64

        # 1. MCTS Guided Action Search
        best_action, policy, mcts_value = champion.mcts.search(z_0[0:1], bar_idx=bar_idx)
        best_action = champion.filter_action(best_action)

        # 2. Dynamic Geometry SL/TP
        sl_tensor, tp_tensor, _ = champion.geometry_head(z_0)
        sl_vals = [float(s.item()) for s in sl_tensor]
        tp_vals = [float(t.item()) for t in tp_tensor]

        # 3. Construct structured action sequence for rollout
        actions_list = []
        for b in range(batch_size):
            b_acts = []
            for step in range(H):
                if step == 0:
                    act_to_encode = best_action
                else:
                    act_to_encode = 0 if np.random.rand() > 0.3 else best_action
                act_to_encode = champion.filter_action(act_to_encode)
                emb = champion.world_model.encode_action(
                    act_to_encode,
                    sl_atr=sl_vals[b],
                    tp_atr=tp_vals[b],
                    action_dim=action_dim,
                    device=z_0.device,
                )
                b_acts.append(emb)
            actions_list.append(torch.cat(b_acts, dim=0))
        actions_tensor = torch.stack(actions_list, dim=0)

        # 4. Multi-step latent rollout
        pred_trajectory = champion.world_model.rollout(z_0, actions_tensor)

        # 5. Non-Circular Authentic Targets with Asymmetric SL Penalty
        has_valid_env = (
            use_env is not None
            and bar_idx is not None
            and getattr(use_env, "total_bars", 0) > bar_idx + batch_size + H + 1
        )
        if has_valid_env:
            target_traj_list = []
            for b in range(batch_size):
                fut_obs = np.stack([use_env.get_observation(bar_idx + b + step + 1) for step in range(H)])
                fut_t = torch.from_numpy(fut_obs).float().to(z_0.device)
                z_target_step = champion.world_model.encode_state(fut_t).detach()
                target_traj_list.append(z_target_step)
            target_trajectory = torch.stack(target_traj_list, dim=0)

            # Realized Value Targets with Asymmetric SL Penalty
            target_val_list = []
            for b in range(batch_size):
                if best_action in (1, 2):
                    res = use_env.simulate_trade(
                        bar_idx=bar_idx + b,
                        action=best_action,
                        sl_atr=sl_vals[b],
                        tp_atr=tp_vals[b],
                        max_holding_bars=60,
                    )
                    ret_atr = float(res.return_atr)
                    exit_type_str = str(res.exit_type).lower()
                    is_sl = exit_type_str in ("sl", "sl_hit") or "sl" in str(res.exit_reason).lower()

                    # Asymmetric penalty for rapid SL hit (<= 3 bars)
                    if is_sl and res.holding_bars <= 3:
                        penalized_ret = -abs(ret_atr) * self.sl_penalty_factor
                        target_ret = -abs(math.tanh(abs(penalized_ret) / 3.0))
                    else:
                        target_ret = math.tanh(ret_atr / 3.0)
                else:
                    target_ret = 0.0
                target_val_list.append(target_ret)
            target_val = torch.tensor(target_val_list, dtype=torch.float32, device=z_0.device).unsqueeze(1)
        else:
            target_trajectory = torch.randn_like(pred_trajectory) * 0.1
            target_val = torch.zeros(batch_size, 1, device=z_0.device)

        # 6. Predict Value & Reward from heads
        pred_val = champion.world_model.value_head(pred_trajectory[:, -1])
        heads_0 = champion.world_model.predict_heads(z_0, actions_tensor[:, 0])
        pred_reward = heads_0["reward"]

        # Baseline V(z_0) and Advantage
        v0_raw = champion.world_model.value_head(z_0)
        v0_baseline = torch.tanh(v0_raw).detach()
        target_adv = target_val - v0_baseline

        # 7. Evaluate with Reasoning Engine
        reason_out: MetacognitiveOutput = champion.reasoning_engine(pred_trajectory[:, -1])
        prm_val = float(reason_out.prm_score.mean().item())

        # OPTION B: Positive-Advantage & High-PRM Selective Filtering
        # Only update reward_head when trade produced positive advantage OR high PRM verification
        # This prevents negative chop/spread updates from pulling down reward_head bias
        adv_val_scalar = float(target_adv.mean().item())
        is_positive_setup = (best_action in (1, 2)) and (adv_val_scalar > 0.0 or prm_val >= 0.85)

        # Check for rapid SL wick hit
        is_rapid_sl = False
        if has_valid_env and best_action in (1, 2):
            for b in range(batch_size):
                if is_sl and res.holding_bars <= 3:
                    is_rapid_sl = True
                    break

        if is_positive_setup:
            loss_reward = F.smooth_l1_loss(pred_reward, target_val, beta=0.5)
        elif is_rapid_sl and float(pred_reward.mean().item()) > 0.0:
            # One-sided penalty for rapid wick stop-outs to suppress false positive conviction
            loss_reward = 0.5 * F.relu(pred_reward).pow(2).mean()
        else:
            loss_reward = torch.tensor(0.0, device=z_0.device)

        # 8. Joint Co-Evolution Loss
        loss_dict = self.loss_fn(
            pred_z=pred_trajectory,
            target_z=target_trajectory,
            pred_val=pred_val,
            target_val=target_val,
            prm_score=reason_out.prm_score,
            hop_rewards=reason_out.hop_rewards,
            expected_depth=float(reason_out.cot_hops.shape[1]),
        )

        total_loss = loss_dict["total_loss"] + 1.0 * loss_reward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(champion.world_model.parameters())
            + list(champion.reasoning_engine.parameters())
            + list(champion.geometry_head.parameters()),
            max_norm=1.0,
        )
        champion.optimizer.step()

        # 9. Record transition into champion's replay buffer
        z_target_step1 = target_trajectory[0, 0] if has_valid_env else z_0[0]
        champion.add_replay(
            z_0=z_0[0],
            action=best_action,
            target_val=float(target_val[0].item()),
            target_z=z_target_step1,
        )

        # 10. Experience Replay Training Step (with Option B filtering)
        replay_loss_val = self._train_champion_replay_with_reward(champion, batch_size=batch_size)

        return {
            "loss_dict": loss_dict,
            "best_action": best_action,
            "policy": policy,
            "mcts_value": mcts_value,
            "replay_loss": replay_loss_val,
            "reward_loss": float(loss_reward.item()),
        }

    def _train_champion_replay_with_reward(
        self,
        champion: SingleChampionPipeline,
        batch_size: int = 4,
    ) -> Optional[float]:
        """Samples replay buffer and updates latent dynamics, value head, and selective reward head."""
        if len(champion.replay_buffer) < max(batch_size, 8):
            return None
        sampled = champion.sample_replay_batch(batch_size=batch_size)
        if sampled is None:
            return None
        z_0, actions, target_v, target_z = sampled

        act_emb = champion.world_model._prepare_action(actions, z_0)
        pred_z = champion.world_model.transition(z_0, act_emb)
        heads_pred = champion.world_model.predict_heads(pred_z, act_emb)
        pred_v = heads_pred["value"]

        # Reward prediction from transition
        heads_0 = champion.world_model.predict_heads(z_0, act_emb)
        pred_r = heads_0["reward"]

        # Option B Selective Replay Filtering:
        # Only update reward head on samples where action was directional AND advantage > 0
        v0_baseline = torch.tanh(champion.world_model.value_head(z_0)).detach()
        adv = target_v - v0_baseline
        mask_positive = (actions > 0) & (adv > 0.0)

        if mask_positive.any():
            l_rew = F.smooth_l1_loss(pred_r[mask_positive], target_v[mask_positive], beta=0.5)
            champion.optimizer.zero_grad()
            l_rew.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in champion.world_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            champion.optimizer.step()
            return float(l_rew.item())
        return 0.0


# =============================================================================
# 3. RAPID SL AUDIT HELPER
# =============================================================================
def compute_rapid_sl_rate(trades: List[Any], threshold_bars: int = 3) -> Tuple[int, int, float]:
    """
    Computes count and percentage of trades that exited via SL in <= threshold_bars.
    Returns: (rapid_sl_count, total_trades, rapid_sl_rate)
    """
    if not trades:
        return 0, 0, 0.0
    total = len(trades)
    rapid_sl = 0
    for t in trades:
        exit_type_str = str(getattr(t, "exit_type", "")).lower()
        holding = int(getattr(t, "holding_bars", 999))
        is_sl = exit_type_str in ("sl", "sl_hit") or "sl" in str(getattr(t, "exit_reason", "")).lower()
        if is_sl and holding <= threshold_bars:
            rapid_sl += 1
    rate = rapid_sl / total
    return rapid_sl, total, rate


# =============================================================================
# 4. FINE-TUNING PIPELINE RUNNER
# =============================================================================
def run_volatility_defense_finetuning(
    csv_path: str = r"C:\Users\Admin\Desktop\BTCUSDm.csv",
    champion_path: Optional[str] = None,
    rounds: int = 1000,
    lr: float = 5e-5,
    sl_penalty_factor: float = 2.0,
    stress_scale: float = 0.25,
    eval_stride: int = 5,
    min_confidence: float = 0.52,
    smoke: bool = False,
) -> Dict[str, Any]:
    """
    Executes deep high-volatility fine-tuning on Bars [30,000 : 55,000],
    evaluates zero-lookahead OOS on Bars [55,000 : 64,785], and applies strict promotion gates.
    """
    actual_rounds = 5 if smoke else rounds
    print("=" * 80)
    print("DUAL-CHAMPION MUZERO: HIGH VOLATILITY DEFENSE & ANTI-WICK FINE-TUNING")
    print("=" * 80)
    print(f"  Configuration:")
    print(f"    - Target Rounds:         {actual_rounds}")
    print(f"    - Learning Rate:         {lr}")
    print(f"    - Asymmetric SL Penalty: {sl_penalty_factor:.1f}x")
    print(f"    - Adversarial Stress:    {stress_scale:.2f}")
    print(f"    - MCTS Action Q-Value:   Action-Conditional Bellman Q(s, a) = R_wm(s, a) + 0.90*V_wm(s') (Option B Filtering)")
    print(f"    - Min Consensus Conf:    {min_confidence:.2f}")
    print(f"    - Train Window (M1):     [30,000 : 55,000] (~25,000 bars)")
    print(f"    - OOS Window (M1):       [55,000 : 64,785] (~9,785 bars)")

    ckpt_file = Path(champion_path) if champion_path else (
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
    )
    backup_file = ckpt_file.with_suffix(".pt.pre_finetune_bak")
    challenger_file = ckpt_file.parent / "mythos_muzero_volatility_challenger.pt"

    # Step 1: Master Environment
    print(f"\n[Step 1/6] Ingesting authentic M1 market data: {csv_path}...")
    master_env = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
    total_bars = master_env.load_data("2025.01.15", "2025.02.28")
    print(f"  Master dataset active: {total_bars:,} bars.")
    assert total_bars == 64785, f"Expected 64785 bars, got {total_bars}"

    # Step 2: High Volatility Train and OOS slices
    print(f"\n[Step 2/6] Partitioning Volatility Slices:")
    train_env = SlicedM1Environment(master_env, 30000, 55000, name="Feb Volatility Peak Train [30k-55k]")
    oos_env = SlicedM1Environment(master_env, 55000, 64785, name="Final 10 Days OOS [55k-64.8k]")
    print(f"  - Train Slice: [{train_env.start_idx:,} : {train_env.end_idx:,}] ({len(train_env):,} bars)")
    print(f"  - OOS Slice:   [{oos_env.start_idx:,} : {oos_env.end_idx:,}] ({len(oos_env):,} bars)")

    # Step 3: Initialize Trainer & Load Champion Checkpoint
    print(f"\n[Step 3/6] Initializing Volatility Defense Trainer & Loading Checkpoint...")
    trainer = VolatilityDefenseMuZeroTrainer(
        env=master_env,
        lr=lr,
        mcts_simulations=4,
        beta_dyna=0.5,
        sl_penalty_factor=sl_penalty_factor,
        elevated_stress_scale=stress_scale,
    )

    prior_reasoning_sd = None
    if ckpt_file.exists():
        print(f"  Loading checkpoint: {ckpt_file}...")
        prior_ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        if "reasoning_engine_state_dict" in prior_ckpt:
            prior_reasoning_sd = prior_ckpt["reasoning_engine_state_dict"]
            trainer.reasoning_engine.load_state_dict(prior_ckpt["reasoning_engine_state_dict"], strict=True)
            trainer.bull_champion.reasoning_engine.load_state_dict(prior_ckpt["reasoning_engine_state_dict"], strict=True)
            trainer.bear_champion.reasoning_engine.load_state_dict(prior_ckpt["reasoning_engine_state_dict"], strict=True)
        if "world_model_state_dict" in prior_ckpt:
            trainer.world_model.load_state_dict(prior_ckpt["world_model_state_dict"], strict=True)
        if "bull_world_model_state_dict" in prior_ckpt:
            trainer.bull_champion.world_model.load_state_dict(prior_ckpt["bull_world_model_state_dict"], strict=True)
        if "bear_world_model_state_dict" in prior_ckpt:
            trainer.bear_champion.world_model.load_state_dict(prior_ckpt["bear_world_model_state_dict"], strict=True)
        if "adversary_state_dict" in prior_ckpt:
            trainer.adversary.load_state_dict(prior_ckpt["adversary_state_dict"], strict=True)
        if "geometry_head_state_dict" in prior_ckpt:
            trainer.geometry_head.load_state_dict(prior_ckpt["geometry_head_state_dict"], strict=True)
        if "bull_geometry_head_state_dict" in prior_ckpt:
            trainer.bull_champion.geometry_head.load_state_dict(prior_ckpt["bull_geometry_head_state_dict"], strict=True)
        if "bear_geometry_head_state_dict" in prior_ckpt:
            trainer.bear_champion.geometry_head.load_state_dict(prior_ckpt["bear_geometry_head_state_dict"], strict=True)
        print(f"  Checkpoint successfully loaded with strict=True.")

        if not backup_file.exists():
            shutil.copyfile(str(ckpt_file), str(backup_file))
            print(f"  Created backup file: {backup_file.name}")
        else:
            print(f"  Pre-existing backup preserved: {backup_file.name}")
    else:
        print(f"  [Notice] Checkpoint {ckpt_file} not found; initializing with baseline weights.")

    # Option B Surgical Optimization:
    # Freeze reasoning engine and latent dynamics to prevent catastrophic forgetting
    # Only train reward_head and geometry_head on selective positive advantage samples
    for champ in [trainer.bull_champion, trainer.bear_champion]:
        for p in champ.reasoning_engine.parameters():
            p.requires_grad = False
        for name, p in champ.world_model.named_parameters():
            if "reward_head" not in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        for p in champ.geometry_head.parameters():
            p.requires_grad = True

        # Re-initialize optimizer exclusively for trainable parameters
        trainable_params = [p for p in champ.world_model.parameters() if p.requires_grad] + list(champ.geometry_head.parameters())
        champ.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Step 4: Baseline Pre-Fine-Tuning OOS Evaluation
    print(f"\n[Step 4/6] Measuring Pre-Fine-Tuning Baseline on OOS Window...")
    oos_cfg = OOSConfig(
        min_confidence=min_confidence,
        eval_stride=eval_stride,
        spread_usd=2.50,
        max_holding_bars=120,
        min_trades=20 if smoke else 50,
        min_win_rate=0.80,
        min_profit_factor=3.0,
        max_drawdown=0.08,
    )

    baseline_metrics, baseline_trades = OOSEvaluationEngine.evaluate_window(
        trainer=trainer,
        env=oos_env,
        cycle_name="Pre-FineTune Baseline OOS [55k-64.8k]",
        config=oos_cfg,
        max_trades=40 if smoke else None,
    )
    b_sl_count, b_total, b_sl_rate = compute_rapid_sl_rate(baseline_trades, threshold_bars=3)
    print(f"  Baseline OOS Metrics:")
    print(f"    - Total Trades:     {baseline_metrics.total_trades}")
    print(f"    - Win Rate:         {baseline_metrics.win_rate:.2%}")
    print(f"    - Profit Factor:    {baseline_metrics.profit_factor:.2f}")
    print(f"    - Max Drawdown:     {baseline_metrics.max_drawdown:.2%}")
    print(f"    - Net Profit:       ${baseline_metrics.net_profit:,.2f}")
    print(f"    - Rapid SL Hits:    {b_sl_count}/{b_total} ({b_sl_rate:.2%})")

    # Step 5: Execute Deep Fine-Tuning
    print(f"\n[Step 5/6] Executing {actual_rounds} Self-Play Co-Evolution Rounds on Volatility Train Slice...")
    t0_train = time.perf_counter()
    train_results = trainer.run_self_play_mcts(
        num_rounds=actual_rounds,
        log_interval=max(1, actual_rounds // 10),
        env=train_env,
    )
    train_time = time.perf_counter() - t0_train
    print(f"  Training finished in {train_time:.1f}s ({train_time/60:.2f}m). Final Replay Loss: {train_results.get('final_loss', 0.0):.4f}")

    # Step 6: Post-Fine-Tuning OOS Verification & Gate Assessment
    print(f"\n[Step 6/6] Measuring Post-Fine-Tuning Performance on OOS Window...")
    post_metrics, post_trades = OOSEvaluationEngine.evaluate_window(
        trainer=trainer,
        env=oos_env,
        cycle_name="Post-FineTune Final OOS [55k-64.8k]",
        config=oos_cfg,
        max_trades=40 if smoke else None,
    )
    p_sl_count, p_total, p_sl_rate = compute_rapid_sl_rate(post_trades, threshold_bars=3)
    sl_reduction = (b_sl_rate - p_sl_rate) / max(1e-4, b_sl_rate) if b_sl_rate > 0 else 0.0

    print(f"\n" + "=" * 80)
    print("COMPARATIVE EVALUATION: PRE vs POST FINE-TUNING")
    print("=" * 80)
    print(f"  Metric                     Baseline          Post-FineTune     Change")
    print(f"  -------------------------  ----------------  ----------------  ------------")
    print(f"  Total Trades:              {baseline_metrics.total_trades:>14d}  {post_metrics.total_trades:>14d}  {post_metrics.total_trades - baseline_metrics.total_trades:>+10d}")
    print(f"  Win Rate:                  {baseline_metrics.win_rate:>14.2%}  {post_metrics.win_rate:>14.2%}  {post_metrics.win_rate - baseline_metrics.win_rate:>+10.2%}")
    print(f"  Profit Factor:             {baseline_metrics.profit_factor:>14.2f}  {post_metrics.profit_factor:>14.2f}  {post_metrics.profit_factor - baseline_metrics.profit_factor:>+10.2f}")
    print(f"  Max Drawdown:              {baseline_metrics.max_drawdown:>14.2%}  {post_metrics.max_drawdown:>14.2%}  {post_metrics.max_drawdown - baseline_metrics.max_drawdown:>+10.2%}")
    print(f"  Net Profit:                ${baseline_metrics.net_profit:>13,.2f}  ${post_metrics.net_profit:>13,.2f}  ${post_metrics.net_profit - baseline_metrics.net_profit:>+10,.2f}")
    print(f"  Rapid SL Rate (<=3 bars):  {b_sl_rate:>14.2%}  {p_sl_rate:>14.2%}  {p_sl_rate - b_sl_rate:>+10.2%} ({sl_reduction:+.1%} reduction)")
    print("=" * 80)

    # Acceptance Criteria Evaluation
    gate_tr = post_metrics.total_trades >= (20 if smoke else 50)
    gate_wr = post_metrics.win_rate >= 0.80
    gate_pf = post_metrics.profit_factor >= 3.0
    gate_dd = post_metrics.max_drawdown <= 0.08
    gate_sl = (p_sl_rate <= b_sl_rate * 0.50) or (p_sl_count == 0)
    all_gates_passed = gate_tr and gate_wr and gate_pf and gate_dd and gate_sl

    print(f"\nStrict Gate Evaluation Status:")
    print(f"  [ {'PASS' if gate_tr else 'FAIL'} ] Total Trades >= {20 if smoke else 50} ({post_metrics.total_trades})")
    print(f"  [ {'PASS' if gate_wr else 'FAIL'} ] Win Rate >= 80% ({post_metrics.win_rate:.2%})")
    print(f"  [ {'PASS' if gate_pf else 'FAIL'} ] Profit Factor >= 3.0 ({post_metrics.profit_factor:.2f})")
    print(f"  [ {'PASS' if gate_dd else 'FAIL'} ] Max Drawdown <= 8% ({post_metrics.max_drawdown:.2%})")
    print(f"  [ {'PASS' if gate_sl else 'FAIL'} ] Rapid SL Reduction >= 50% ({sl_reduction:.1%} reduction, hits: {p_sl_count})")

    stats = {
        "pre_finetune": {
            "win_rate": baseline_metrics.win_rate,
            "profit_factor": baseline_metrics.profit_factor,
            "max_drawdown": baseline_metrics.max_drawdown,
            "net_profit": baseline_metrics.net_profit,
            "total_trades": baseline_metrics.total_trades,
            "rapid_sl_count": b_sl_count,
            "rapid_sl_rate": b_sl_rate,
        },
        "post_finetune": {
            "win_rate": post_metrics.win_rate,
            "profit_factor": post_metrics.profit_factor,
            "max_drawdown": post_metrics.max_drawdown,
            "net_profit": post_metrics.net_profit,
            "total_trades": post_metrics.total_trades,
            "rapid_sl_count": p_sl_count,
            "rapid_sl_rate": p_sl_rate,
            "rapid_sl_reduction": sl_reduction,
        },
        "win_rate": post_metrics.win_rate,
        "profit_factor": post_metrics.profit_factor,
        "max_drawdown": post_metrics.max_drawdown,
        "net_profit": post_metrics.net_profit,
        "total_trades": post_metrics.total_trades,
        "training_samples": len(train_env),
        "epochs": actual_rounds,
        "gates_passed": all_gates_passed,
    }

    if all_gates_passed:
        target_export = ckpt_file
        print(f"\n>>> PROMOTED TO CHAMPION: Overwriting canonical checkpoint at {target_export} <<<")
        export_canonical_champion_checkpoint(
            trainer=trainer,
            checkpoint_path=target_export,
            oos_stats=stats,
            total_bars=total_bars,
            num_cycles=actual_rounds,
            preserved_reasoning_sd=prior_reasoning_sd,
        )
        verdict = "PROMOTED_CHAMPION"
    else:
        target_export = challenger_file
        print(f"\n>>> GATE NOT FULLY MET: Saving safely as Challenger at {target_export} <<<")
        print(f"    Canonical Champion preserved at {ckpt_file}.")
        export_canonical_champion_checkpoint(
            trainer=trainer,
            checkpoint_path=target_export,
            oos_stats=stats,
            total_bars=total_bars,
            num_cycles=actual_rounds,
            preserved_reasoning_sd=prior_reasoning_sd,
        )
        verdict = "SAVED_CHALLENGER"

    return {
        "verdict": verdict,
        "checkpoint_path": str(target_export),
        "backup_path": str(backup_file),
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="MuZero Volatility Defense Fine-Tuning")
    parser.add_argument("--csv_path", type=str, default=r"C:\Users\Admin\Desktop\BTCUSDm.csv")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--sl_penalty", type=float, default=2.0)
    parser.add_argument("--stress_scale", type=float, default=0.25)
    parser.add_argument("--eval_stride", type=int, default=5)
    parser.add_argument("--min_confidence", type=float, default=0.52)
    parser.add_argument("--smoke", action="store_true", help="Run 5 rounds for rapid verification")
    args = parser.parse_args()

    run_volatility_defense_finetuning(
        csv_path=args.csv_path,
        rounds=args.rounds,
        lr=args.lr,
        sl_penalty_factor=args.sl_penalty,
        stress_scale=args.stress_scale,
        eval_stride=args.eval_stride,
        min_confidence=args.min_confidence,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
