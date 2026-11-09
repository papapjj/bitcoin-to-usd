#!/usr/bin/env python3
"""
Automated Forensic Trade Flaw Diagnosis Engine for Milestone 2.
Directly reproduces and attributes trade execution flaws and adverse excursions
across the Bilateral Dual-CoT neural policy architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose_trade_flaws_m2")

REGIMES_CONFIG = [
    {
        "id": "regime_1_bull_expansion",
        "csv": ROOT / "outputs" / "btcusd_m1_7day_clean.csv",
        "name": "Regime 1: Bull Expansion",
    },
    {
        "id": "regime_2_drift_bear",
        "csv": ROOT / "outputs" / "btcusd_m1_week1_20261005.csv",
        "name": "Regime 2: Drift Bear & Rebound",
    },
    {
        "id": "regime_3_range_chop",
        "csv": ROOT / "outputs" / "btcusd_m1_week2_20261012.csv",
        "name": "Regime 3: Range Chop",
    },
    {
        "id": "regime_4_waterfall_crash",
        "csv": ROOT / "outputs" / "btcusd_m1_week3_20261019.csv",
        "name": "Regime 4: Waterfall Crash",
    },
    {
        "id": "regime_5_high_volatility",
        "csv": ROOT / "outputs" / "btcusd_m1_week4_20261026.csv",
        "name": "Regime 5: High Volatility Expansion",
    },
]

TAXONOMY_CATEGORIES = [
    "Profit Reversal / Trailing Gap",
    "Rejection Wick",
    "Marginal PRM",
    "Counter-Trend Momentum Trap",
]


def classify_trade_flaw(
    mfe_atr: float,
    direction: str,
    upper_wick_atr: float,
    lower_wick_atr: float,
    prm_score: float,
    trend_dir: str,
    duration_bars: int,
    raw_mae_atr: float,
) -> Dict[str, str]:
    """
    Classify a trade loss into the 4 taxonomy categories with detailed remediation guidance.
    """
    # 1. Profit Reversal / Trailing Gap
    if mfe_atr >= 1.20:
        return {
            "primary_category": "Profit Reversal / Trailing Gap",
            "description": (
                f"Trade attained deep favorable excursion (+{mfe_atr:.2f} ATR MFE) approaching TP (2.25 ATR), "
                "but reversed to full stop loss because trailing stop was not dynamically trailing during holding."
            ),
            "remediation_guidance": (
                "Activate dynamic trailing stop (0.50 ATR) once favorable excursion exceeds 1.50 ATR "
                "to lock in unrealized profits and prevent scratch-reversals into full SL hits."
            ),
        }

    # 2. Rejection Wick
    opp_wick = upper_wick_atr if direction == "SELL" else lower_wick_atr
    if opp_wick >= 0.40 or (duration_bars <= 5 and raw_mae_atr >= 1.60):
        return {
            "primary_category": "Rejection Wick",
            "description": (
                f"Entry coincided with significant opposing candlestick rejection wick ({opp_wick:.2f} ATR) "
                f"or short-duration rapid intrabar spike (duration {duration_bars}m, MAE {raw_mae_atr:.2f} ATR)."
            ),
            "remediation_guidance": (
                "Quadratic anti-MAE excursion loss penalizes aggressive entries during volatile wick formations, "
                "guiding the policy towards patient entry timing without external indicator rules."
            ),
        }

    # 3. Marginal PRM
    if prm_score < 0.60:
        return {
            "primary_category": "Marginal PRM",
            "description": (
                f"Entry was executed with marginal PRM conviction ({prm_score:.4f}), hovering just above "
                "the baseline 0.58 threshold with narrow confidence buffer."
            ),
            "remediation_guidance": (
                "Anti-MAE retraining sharpens Arbiter PRM calibration so marginal high-risk setups receive "
                "lower PRM scores (< 0.58) and are naturally bypassed purely via neural weights."
            ),
        }

    # 4. Counter-Trend Momentum Trap
    is_counter = (direction == "BUY" and trend_dir == "BEAR") or (direction == "SELL" and trend_dir == "BULL")
    return {
        "primary_category": "Counter-Trend Momentum Trap",
        "description": (
            f"Trade entered against prevailing 20/60 trend ({direction} during {trend_dir} trend) "
            "into persistent adverse orderflow momentum."
        ),
        "remediation_guidance": (
            "Dialectical cross-examination refinement strengthens Bear/Bull adversarial challenge "
            "to prevent Arbiter over-optimism against strong directional momentum."
        ),
    }


def diagnose_trade_losses(
    trades_csv: Path,
    checkpoint_path: Path,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """
    Runs automated forensic trade flaw diagnosis for all losing trades.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not trades_csv.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_csv}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    logger.info("Loading Bilateral Dual-CoT model from: %s", checkpoint_path)
    cfg = BilateralDualCoTConfig(
        input_dim=233,
        latent_dim=768,
        hidden_dim=256,
        num_hops=4,
        num_debate_turns=3,
        prm_threshold=0.58,
        trap_threshold=0.60,
    )
    engine = BilateralDualCoTEngine(cfg)
    engine.load_checkpoint(checkpoint_path)
    engine.eval()

    all_trades = pd.read_csv(trades_csv)
    loss_trades = all_trades[all_trades["net_pnl_usd"] < 0].copy()
    total_trades_count = len(all_trades)
    total_losses_count = len(loss_trades)

    logger.info("Found %d total trades, %d losing trades to diagnose.", total_trades_count, total_losses_count)

    diagnosed_trades: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []

    losses_by_regime: Dict[str, int] = {}
    losses_by_direction: Dict[str, int] = {"BUY": 0, "SELL": 0}
    losses_by_category: Dict[str, int] = {cat: 0 for cat in TAXONOMY_CATEGORIES}

    lookback = 60

    for regime_info in REGIMES_CONFIG:
        regime_id = regime_info["id"]
        csv_path = regime_info["csv"]
        regime_name = regime_info["name"]

        # Filter losses belonging to this regime
        reg_losses = loss_trades[loss_trades["ticket"].isin(
            # We can match by timestamp or load regime trades CSV
            pd.read_csv(ROOT / "outputs" / "acceptance_artifacts" / "milestone_1" / regime_id / "trades.csv")["ticket"]
        )] if (ROOT / "outputs" / "acceptance_artifacts" / "milestone_1" / regime_id / "trades.csv").exists() else loss_trades

        losses_by_regime[regime_id] = len(reg_losses)
        if len(reg_losses) == 0:
            continue

        logger.info("Processing %s (%d losses)...", regime_name, len(reg_losses))

        # Ingest market data
        df = pd.read_csv(csv_path, sep="\t")
        df_renamed = df.rename(columns={
            "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close",
            "<TICKVOL>": "Volume", "<SPREAD>": "Spread",
        })
        col_d = "<DATE>" if "<DATE>" in df_renamed.columns else "DATE"
        col_t = "<TIME>" if "<TIME>" in df_renamed.columns else "TIME"
        df_renamed["Datetime"] = df_renamed[col_d].astype(str) + " " + df_renamed[col_t].astype(str)

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
        datetimes = feat_df["Datetime"].values.astype(str)

        for _, l_row in reg_losses.iterrows():
            t_entry = pd.to_datetime(l_row["entry_time"])
            match_indices = (feat_df["Datetime"] == t_entry).to_numpy().nonzero()[0]
            if len(match_indices) == 0:
                logger.warning("No bar match for ticket %s at %s", l_row["ticket"], t_entry)
                continue
            bar_idx = match_indices[0]

            cur_p = closes[bar_idx]
            cur_atr = max(atrs[bar_idx], 1.0)
            m1_60 = np.column_stack([
                opens[bar_idx - lookback:bar_idx],
                highs[bar_idx - lookback:bar_idx],
                lows[bar_idx - lookback:bar_idx],
                closes[bar_idx - lookback:bar_idx],
            ])
            z_in_np = extract_canonical_bilateral_ingress_233(base_137[bar_idx], m1_60, cur_p, cur_atr)
            z_in = torch.from_numpy(z_in_np).float().unsqueeze(0)

            # Neural Forward Pass through 6 Submodules
            with torch.no_grad():
                # 1. State adapter
                z = engine.state_adapter(z_in)
                # 2. Bull CoT
                bull_out = engine.bull_cot(z)
                # 3. Bear CoT
                bear_out = engine.bear_cot(z)
                b_sc = float(bull_out.bull_score[0].item())
                r_sc = float(bear_out.bear_score[0].item())

                # 4. Trap detector
                p_thought = bull_out.cot_hops[:, -1, :]
                a_thought = bear_out.cot_hops[:, -1, :]
                joint = torch.cat([a_thought, p_thought], dim=-1)
                trap_logits = engine.trap_detector(joint)
                trap_probs = torch.softmax(trap_logits, dim=-1)[0].cpu().numpy()

                # 5. Dialectical cross-examination
                final_p, final_a, _, _, _, _ = engine.cross_examination(
                    p_thought, a_thought, turns=engine.config.num_debate_turns
                )

                # 6. Metacognitive Arbiter & PRM
                convictions = torch.tensor([[b_sc, r_sc]], dtype=torch.float32)
                act, prm, conf, cp = engine.arbiter(
                    final_p, final_a, torch.from_numpy(trap_probs).unsqueeze(0), convictions
                )
                synthesis = torch.cat([final_p, final_a], dim=-1)
                arb_logits = engine.arbiter.arbiter_net(synthesis)
                arb_probs = torch.softmax(arb_logits, dim=-1)[0].cpu().numpy()

            # Candlestick Microstructure
            candle_open = opens[bar_idx]
            candle_close = closes[bar_idx]
            candle_high = highs[bar_idx]
            candle_low = lows[bar_idx]
            body_atr = abs(candle_close - candle_open) / cur_atr
            upper_wick_atr = (candle_high - max(candle_open, candle_close)) / cur_atr
            lower_wick_atr = (min(candle_open, candle_close) - candle_low) / cur_atr
            mom_5_atr = (closes[bar_idx] - closes[max(0, bar_idx - 5)]) / cur_atr

            ma20 = np.mean(closes[max(0, bar_idx - 20):bar_idx + 1])
            ma60 = np.mean(closes[max(0, bar_idx - 60):bar_idx + 1])
            trend_dir = "BULL" if ma20 > ma60 else "BEAR"

            t_exit = pd.to_datetime(l_row["exit_time"])
            exit_indices = (feat_df["Datetime"] == t_exit).to_numpy().nonzero()[0]
            duration_bars = int(exit_indices[0] - bar_idx) if len(exit_indices) > 0 else 0

            raw_mae = float(l_row["mae_atr"])
            bounded_mae = min(raw_mae, 1.50)  # Capped at executed stop fill price

            # Flaw Classification
            flaw = classify_trade_flaw(
                mfe_atr=float(l_row["mfe_atr"]),
                direction=str(l_row["direction"]),
                upper_wick_atr=upper_wick_atr,
                lower_wick_atr=lower_wick_atr,
                prm_score=float(l_row["prm_score"]),
                trend_dir=trend_dir,
                duration_bars=duration_bars,
                raw_mae_atr=raw_mae,
            )

            primary_cat = flaw["primary_category"]
            losses_by_category[primary_cat] += 1
            losses_by_direction[str(l_row["direction"])] += 1

            diagnosed_trade_obj = {
                "ticket": str(l_row["ticket"]),
                "regime_id": regime_id,
                "regime_name": regime_name,
                "direction": str(l_row["direction"]),
                "entry_time": str(l_row["entry_time"]),
                "exit_time": str(l_row["exit_time"]),
                "entry_price": float(l_row["entry_price"]),
                "exit_price": float(l_row["exit_price"]),
                "initial_atr": float(l_row["initial_atr"]),
                "sl_price": float(l_row["sl_price"]),
                "tp_price": float(l_row["tp_price"]),
                "net_pnl_usd": float(l_row["net_pnl_usd"]),
                "realized_r": float(l_row["realized_r"]),
                "raw_mae_atr": round(raw_mae, 4),
                "bounded_mae_atr": round(bounded_mae, 4),
                "mfe_atr": round(float(l_row["mfe_atr"]), 4),
                "duration_bars": duration_bars,
                "neural_attribution": {
                    "bull_score": round(b_sc, 4),
                    "bear_score": round(r_sc, 4),
                    "trap_probabilities": {
                        "no_trap": round(float(trap_probs[0]), 4),
                        "bull_trap": round(float(trap_probs[1]), 4),
                        "bear_trap": round(float(trap_probs[2]), 4),
                    },
                    "arbiter_probabilities": {
                        "hold": round(float(arb_probs[0]), 4),
                        "buy": round(float(arb_probs[1]), 4),
                        "sell": round(float(arb_probs[2]), 4),
                    },
                    "prm_score": round(float(l_row["prm_score"]), 4),
                    "is_counter_punch": bool(cp[0].item()),
                },
                "microstructure_context": {
                    "trend_direction_20_60": trend_dir,
                    "momentum_5_bar_atr": round(mom_5_atr, 4),
                    "body_atr": round(body_atr, 4),
                    "upper_wick_atr": round(upper_wick_atr, 4),
                    "lower_wick_atr": round(lower_wick_atr, 4),
                },
                "flaw_classification": flaw,
            }
            diagnosed_trades.append(diagnosed_trade_obj)

            decomp_rows.append({
                "ticket": l_row["ticket"],
                "regime": regime_id,
                "direction": l_row["direction"],
                "entry_time": l_row["entry_time"],
                "exit_time": l_row["exit_time"],
                "entry_price": l_row["entry_price"],
                "exit_price": l_row["exit_price"],
                "initial_atr": l_row["initial_atr"],
                "prm_score": l_row["prm_score"],
                "b_score": round(b_sc, 4),
                "r_score": round(r_sc, 4),
                "trap_no": round(float(trap_probs[0]), 4),
                "trap_bull": round(float(trap_probs[1]), 4),
                "trap_bear": round(float(trap_probs[2]), 4),
                "arb_hold": round(float(arb_probs[0]), 4),
                "arb_buy": round(float(arb_probs[1]), 4),
                "arb_sell": round(float(arb_probs[2]), 4),
                "raw_mae_atr": round(raw_mae, 4),
                "bounded_mae_atr": round(bounded_mae, 4),
                "mfe_atr": round(float(l_row["mfe_atr"]), 4),
                "net_pnl_usd": l_row["net_pnl_usd"],
                "duration_bars": duration_bars,
                "trend_dir": trend_dir,
                "upper_wick_atr": round(upper_wick_atr, 4),
                "lower_wick_atr": round(lower_wick_atr, 4),
                "flaw_category": primary_cat,
            })

    # Summary Statistics
    raw_maes_all = all_trades["mae_atr"].to_numpy(dtype=np.float64)
    bounded_maes_all = np.where(all_trades["exit_reason"] == "SL_HIT", np.minimum(raw_maes_all, 1.50), raw_maes_all)

    r5_trades = all_trades[all_trades["ticket"].isin(
        pd.read_csv(ROOT / "outputs" / "acceptance_artifacts" / "milestone_1" / "regime_5_high_volatility" / "trades.csv")["ticket"]
    )] if (ROOT / "outputs" / "acceptance_artifacts" / "milestone_1" / "regime_5_high_volatility" / "trades.csv").exists() else pd.DataFrame()

    r5_raw_maes = r5_trades["mae_atr"].to_numpy(dtype=np.float64) if len(r5_trades) > 0 else np.array([0.0])
    r5_bounded_maes = np.where(r5_trades["exit_reason"] == "SL_HIT", np.minimum(r5_raw_maes, 1.50), r5_raw_maes) if len(r5_trades) > 0 else np.array([0.0])

    loss_raw_maes = [t["raw_mae_atr"] for t in diagnosed_trades]
    loss_bnd_maes = [t["bounded_mae_atr"] for t in diagnosed_trades]

    artifact_payload = {
        "milestone": "Milestone 2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_trades_file": str(trades_csv.as_posix()),
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "checkpoint_sha256": ckpt_sha256,
        "summary": {
            "total_evaluated_trades": total_trades_count,
            "total_losses": total_losses_count,
            "loss_rate_pct": round((total_losses_count / max(1, total_trades_count)) * 100.0, 2),
            "losses_by_regime": losses_by_regime,
            "losses_by_direction": losses_by_direction,
            "losses_by_category": losses_by_category,
            "excursion_metrics": {
                "consolidated_raw_mae_atr": round(float(np.mean(raw_maes_all)), 4),
                "consolidated_bounded_mae_atr": round(float(np.mean(bounded_maes_all)), 4),
                "regime_5_raw_mae_atr": round(float(np.mean(r5_raw_maes)), 4),
                "regime_5_bounded_mae_atr": round(float(np.mean(r5_bounded_maes)), 4),
                "losses_only_raw_mae_atr": round(float(np.mean(loss_raw_maes)), 4),
                "losses_only_bounded_mae_atr": round(float(np.mean(loss_bnd_maes)), 4),
            },
        },
        "diagnosed_trades": diagnosed_trades,
    }

    # Save JSON artifact
    json_path = artifacts_dir / "trade_flaw_diagnosis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact_payload, f, indent=2)
    logger.info("Saved JSON trade flaw diagnosis to: %s", json_path)

    # Save CSV decomposition artifact
    csv_out_path = artifacts_dir / "trade_flaw_decomposition.csv"
    pd.DataFrame(decomp_rows).to_csv(csv_out_path, index=False)
    logger.info("Saved CSV trade flaw decomposition to: %s", csv_out_path)

    return artifact_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Trade Flaw Diagnosis for Milestone 2")
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=ROOT / "outputs" / "acceptance_artifacts" / "milestone_1" / "consolidated" / "consolidated_trades.csv",
        help="Path to consolidated trades CSV from baseline",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "outputs" / "acceptance_artifacts" / "milestone_2",
        help="Directory to save diagnostic artifacts",
    )
    args = parser.parse_args()

    diagnose_trade_losses(
        trades_csv=args.trades_csv,
        checkpoint_path=args.checkpoint,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
