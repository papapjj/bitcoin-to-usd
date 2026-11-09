"""
Dashboard Backend for Temporal Cross-Attention (TCA) and CoT Reasoning Visualization (Feature 16).
Provides high-performance APIs for inspecting 240-candle attention heatmaps,
multi-horizon return predictions [t+5, t+15, t+30], CoT reasoning hops,
and authentic M1 market data serving with graceful checkpoint fallback.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    MultiHorizonProjectionHead,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)

app = FastAPI(title="Mythos Dual-Champion Attention Dashboard", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
state: Dict[str, Any] = {
    "env": None,
    "world_model": None,
    "projection_head": None,
    "reasoning_engine": None,
    "checkpoint_loaded": False,
    "checkpoint_path": None,
    "checkpoint_type": "uninitialized",
    "tca_trained": False,
    "head_trained": False,
    "total_bars": 0,
}


def resolve_checkpoint_path() -> Optional[str]:
    """
    Resolves the highest-priority available checkpoint:
    1. Optimal Convergence Frontier checkpoint (Phase 3 30ep, highest PnL & PF)
    2. Phase 2 Fine-tuned checkpoint (all 4 namespaces trained)
    3. Phase 1 Pre-trained checkpoint (TCA + Head trained, base encoder frozen)
    4. Baseline Challenger checkpoint (30-key WM + RE, temporal stream untrained)
    """
    candidates = [
        "outputs/checkpoints/tca_optimal_frontier.pt",
        "outputs/checkpoints/tca_phase3_30epochs.pt",
        "outputs/checkpoints/tca_deep_20epochs.pt",
        "outputs/checkpoints/tca_phase2_finetuned.pt",
        "outputs/checkpoints/tca_phase1_pretrained.pt",
        "outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def initialize_runtime() -> None:
    """Initializes environment and models if not already initialized."""
    if state["env"] is not None and state["world_model"] is not None:
        return

    # 1. Initialize environment with historical data
    csv_path = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
    if os.path.exists(csv_path):
        env = RealM1Environment(csv_path=csv_path)
        try:
            total = env.load_data(start_date="2025.01.15", end_date="2025.02.28")
            state["total_bars"] = total
            state["env"] = env
            if hasattr(env, "timestamps") and (not hasattr(env, "unix_timestamps") or len(env.unix_timestamps) == 0):
                try:
                    dt_idx = pd.to_datetime([s.replace(".", "-") for s in env.timestamps])
                    env.unix_timestamps = np.ascontiguousarray(dt_idx.astype("datetime64[s]").astype(np.int64))
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Failed to load specified window: {e}. Loading available subset...")
            state["total_bars"] = len(env.close_arr) if hasattr(env, "close_arr") else 0
            state["env"] = env

    # 2. Instantiate Architecture
    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream(candle_len=240, temp_dim=128, num_heads=4, init_gate_bias=-2.0)
    re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.0)

    # 3. Checkpoint Auto-Loading with Graceful Fallback
    ckpt_path = resolve_checkpoint_path()
    tca_trained = False
    head_trained = False
    ckpt_type = "uninitialized"

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # A. Load World Model (30 keys)
            wm_state = ckpt.get("world_model_state_dict", {})
            clean_wm = {k: v for k, v in wm_state.items() if not k.startswith("temporal_encoder.")}
            wm.load_state_dict(clean_wm, strict=False)

            # B. Load Temporal Cross-Attention Encoder (41 keys)
            if "temporal_encoder_state_dict" in ckpt and wm.temporal_encoder is not None:
                wm.temporal_encoder.load_state_dict(ckpt["temporal_encoder_state_dict"], strict=True)
                tca_trained = True
            elif wm.temporal_encoder is not None:
                tca_keys = {
                    k[len("temporal_encoder.") :]: v
                    for k, v in wm_state.items()
                    if k.startswith("temporal_encoder.")
                }
                if len(tca_keys) == 41:
                    wm.temporal_encoder.load_state_dict(tca_keys, strict=True)
                    tca_trained = True

            # C. Load Multi-Horizon Projection Head (8 keys)
            if "projection_head_state_dict" in ckpt:
                head.load_state_dict(ckpt["projection_head_state_dict"], strict=True)
                head_trained = True
            else:
                # Fallback to Phase 1 or Phase 2 head if loaded baseline
                fallback_heads = [
                    "outputs/checkpoints/tca_phase2_finetuned.pt",
                    "outputs/checkpoints/tca_phase1_pretrained.pt",
                ]
                for fb in fallback_heads:
                    if os.path.exists(fb):
                        fb_ckpt = torch.load(fb, map_location="cpu", weights_only=False)
                        if "projection_head_state_dict" in fb_ckpt:
                            head.load_state_dict(fb_ckpt["projection_head_state_dict"], strict=True)
                            head_trained = True
                            break

            # D. Load Reasoning Engine (191 keys)
            re_state = ckpt.get("reasoning_engine_state_dict", ckpt.get("bull_cot_state_dict", {}))
            if re_state:
                re.load_state_dict(re_state, strict=False)

            state["checkpoint_loaded"] = True
            state["checkpoint_path"] = ckpt_path
            if "tca_phase2" in ckpt_path:
                ckpt_type = "tca_phase2_finetuned"
            elif "tca_phase1" in ckpt_path:
                ckpt_type = "tca_phase1_pretrained"
            else:
                ckpt_type = "baseline_challenger"

        except Exception as e:
            print(f"[Dashboard Backend] Warning: Error loading checkpoint {ckpt_path}: {e}")

    state["world_model"] = wm.eval()
    state["projection_head"] = head.eval()
    state["reasoning_engine"] = re.eval()
    state["tca_trained"] = tca_trained
    state["head_trained"] = head_trained
    state["checkpoint_type"] = ckpt_type


@app.on_event("startup")
async def startup_event() -> None:
    initialize_runtime()


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns system status, bar count, loaded checkpoint details, and TCA capabilities."""
    initialize_runtime()
    return {
        "status": "ready" if state["env"] is not None else "no_data",
        "total_bars": state["total_bars"],
        "checkpoint_loaded": state["checkpoint_loaded"],
        "checkpoint_path": state["checkpoint_path"],
        "checkpoint_type": state["checkpoint_type"],
        "tca_trained": state["tca_trained"],
        "head_trained": state["head_trained"],
        "multi_horizon_active": state["head_trained"],
        "lookback_candles": 240,
        "device": "cpu",
    }


_timestamp_cache: Dict[str, int] = {}


def extract_candles_for_window(env: Any, bar_idx: int, lookback: int = 240) -> List[Dict[str, Any]]:
    """
    Extracts and formats lookback OHLCV candles ending at bar_idx for TradingView charts.
    Uses pre-computed unix_timestamps for O(1) microsecond lookup, eliminating repetitive pd.to_datetime overhead.
    """
    start_idx = max(0, bar_idx - lookback + 1)
    count = bar_idx - start_idx + 1

    has_unix_ts = hasattr(env, "unix_timestamps") and len(env.unix_timestamps) > bar_idx
    has_ts = hasattr(env, "timestamps") and len(env.timestamps) > bar_idx

    if has_unix_ts and has_ts:
        u_times = env.unix_timestamps[start_idx : bar_idx + 1]
        ts_strs = env.timestamps[start_idx : bar_idx + 1]
        opens = env.open_arr[start_idx : bar_idx + 1]
        highs = env.high_arr[start_idx : bar_idx + 1]
        lows = env.low_arr[start_idx : bar_idx + 1]
        closes = env.close_arr[start_idx : bar_idx + 1]
        volumes = env.volume_arr[start_idx : bar_idx + 1]
        spreads = env.spread_arr[start_idx : bar_idx + 1]
        atrs = env.atr_arr[start_idx : bar_idx + 1]

        return [
            {
                "bar_idx": start_idx + i,
                "time": int(u_times[i]),
                "datetime": ts_strs[i],
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]),
                "spread": float(spreads[i]),
                "atr": float(atrs[i]),
            }
            for i in range(count)
        ]

    # Fallback path for mocked or partial environments
    candles = []
    for idx in range(start_idx, bar_idx + 1):
        ts_str = env.timestamps[idx] if (has_ts and idx < len(env.timestamps)) else f"bar_{idx}"
        if has_unix_ts and idx < len(env.unix_timestamps):
            unix_time = int(env.unix_timestamps[idx])
        elif ts_str in _timestamp_cache:
            unix_time = _timestamp_cache[ts_str]
        else:
            try:
                unix_time = int(pd.to_datetime(ts_str.replace(".", "-")).timestamp())
            except Exception:
                unix_time = idx * 60
            _timestamp_cache[ts_str] = unix_time

        candles.append({
            "bar_idx": idx,
            "time": unix_time,
            "datetime": ts_str,
            "open": float(env.open_arr[idx]),
            "high": float(env.high_arr[idx]),
            "low": float(env.low_arr[idx]),
            "close": float(env.close_arr[idx]),
            "volume": float(env.volume_arr[idx]),
            "spread": float(env.spread_arr[idx]),
            "atr": float(env.atr_arr[idx]),
        })
    return candles


@app.get("/api/candles")
def get_candles(bar_idx: int = Query(..., ge=0)) -> Dict[str, Any]:
    """Returns 240 raw M1 candles ending at bar_idx formatted for TradingView Lightweight Charts."""
    initialize_runtime()
    env = state["env"]
    if env is None or bar_idx >= state["total_bars"]:
        raise HTTPException(status_code=400, detail="Invalid bar index or environment uninitialized")

    candles = extract_candles_for_window(env, bar_idx, lookback=240)

    return {
        "target_bar_idx": bar_idx,
        "count": len(candles),
        "candles": candles,
    }


class AnalysisRequest(BaseModel):
    bar_idx: int


@app.post("/api/analyze")
def analyze_bar(req: AnalysisRequest) -> Dict[str, Any]:
    """
    Runs full forward inference at bar_idx:
    - Extracts 240-candle normalized sequence and 137 features.
    - Computes TCA attention weights across 240 candles and all 4 heads.
    - Evaluates MultiHorizonProjectionHead for returns at t+5, t+15, t+30.
    - Extracts CoT Reasoning Hops and Action/SL/TP recommendation.
    """
    initialize_runtime()
    env = state["env"]
    wm = state["world_model"]
    re = state["reasoning_engine"]
    head = state["projection_head"]

    bar_idx = req.bar_idx
    if env is None or bar_idx >= state["total_bars"]:
        raise HTTPException(status_code=400, detail="Invalid bar index")

    # 1. Extract 137D observation and 240-candle sequence
    obs_137 = env.get_observation(bar_idx)
    candle_seq = env.get_candle_sequence(bar_idx, lookback=240)
    atr = float(env.atr_arr[bar_idx])
    close_price = float(env.close_arr[bar_idx])

    # 2. Run Forward Passes
    with torch.no_grad():
        z_out = wm.encode_state(obs_137, candle_seq=candle_seq)
        attn_summary = wm.get_attention_summary()
        raw_attn = wm.get_attention_weights()
        cot_out = re(z_out)

        # Multi-horizon return predictions [t+5, t+15, t+30]
        pred_returns = [0.0, 0.0, 0.0]
        if head is not None:
            preds_tensor = head(z_out)
            pred_returns = preds_tensor.squeeze(0).cpu().numpy().tolist()

    # 3. Decision Actions
    bull_score = float(cot_out.bull_score.mean().item())
    bear_score = float(cot_out.bear_score.mean().item())
    prm_score = float(cot_out.prm_score.mean().item())
    confidence = float(cot_out.confidence.mean().item())
    raw_action = int(cot_out.action[0].item())

    if raw_action == 1:
        recommended_action = 1
        action_str = "BUY"
    elif raw_action == -1:
        recommended_action = 2
        action_str = "SELL"
    else:
        recommended_action = 0
        action_str = "HOLD"

    sl_atr = 2.5
    tp_atr = 5.0

    # 4. Attention Weights & Invariants
    attn_by_head = []
    if raw_attn is not None:
        attn_np = raw_attn.squeeze(0).cpu().numpy()  # [4, 240]
        mean_attn = np.mean(attn_np, axis=0).tolist()
        for h in range(4):
            attn_by_head.append(attn_np[h].tolist())
        peak_indices = [int(np.argmax(attn_np[h])) for h in range(4)]
    else:
        mean_attn = [1.0 / 240] * 240
        attn_by_head = [[1.0 / 240] * 240 for _ in range(4)]
        peak_indices = [239] * 4

    recency_ratio = (
        float(attn_summary.get("recency_weight_ratio", attn_summary.get("recency_ratio", 0.0)))
        if attn_summary
        else 0.0
    )
    head_entropy = (
        [float(e) for e in attn_summary.get("head_entropy", [0.0] * 4)] if attn_summary else [0.0] * 4
    )
    mean_entropy = float(np.mean(head_entropy)) if head_entropy else 0.0

    # 5. Multi-Horizon Projections Details
    p5, p15, p30 = pred_returns
    projections = {
        "t+5": {
            "horizon_min": 5,
            "atr_return": round(p5, 4),
            "price_delta": round(p5 * atr, 2),
            "target_price": round(close_price + p5 * atr, 2),
            "signal": "BULLISH" if p5 > 0.05 else ("BEARISH" if p5 < -0.05 else "NEUTRAL"),
        },
        "t+15": {
            "horizon_min": 15,
            "atr_return": round(p15, 4),
            "price_delta": round(p15 * atr, 2),
            "target_price": round(close_price + p15 * atr, 2),
            "signal": "BULLISH" if p15 > 0.05 else ("BEARISH" if p15 < -0.05 else "NEUTRAL"),
        },
        "t+30": {
            "horizon_min": 30,
            "atr_return": round(p30, 4),
            "price_delta": round(p30 * atr, 2),
            "target_price": round(close_price + p30 * atr, 2),
            "signal": "BULLISH" if p30 > 0.05 else ("BEARISH" if p30 < -0.05 else "NEUTRAL"),
        },
    }

    consensus_dir = "NEUTRAL"
    if p5 > 0 and p15 > 0:
        consensus_dir = "BULLISH"
    elif p5 < 0 and p15 < 0:
        consensus_dir = "BEARISH"

    feature_groups = {
        "Trend & Momentum": float(np.mean(np.abs(obs_137[0:15]))),
        "Volatility & ATR": float(np.mean(np.abs(obs_137[15:29]))),
        "Position Slots": float(np.mean(np.abs(obs_137[29:69]))),
        "Account Health": float(np.mean(np.abs(obs_137[69:77]))),
        "Wyckoff Accumulation/Distribution": float(np.mean(np.abs(obs_137[121:137]))),
        "Multi-Timeframe": float(np.mean(np.abs(obs_137[77:90]))),
    }

    hops = [
        {"hop": 1, "title": "Regime & Volatility", "focus": "Assessing macro expansion vs chop"},
        {"hop": 2, "title": "Liquidity & Order Flow", "focus": "Sweeps, order blocks & pinbar traps"},
        {"hop": 3, "title": "Directional Momentum", "focus": "Bull vs Bear pressure balance"},
        {"hop": 4, "title": "Metacognitive Synthesis", "focus": f"Formulating {action_str} proposal"},
    ]

    return {
        "bar_idx": bar_idx,
        "action": action_str,
        "action_code": recommended_action,
        "confidence": round(confidence, 4),
        "bull_score": round(bull_score, 4),
        "bear_score": round(bear_score, 4),
        "prm_verifier_score": round(prm_score, 4),
        "suggested_geometry": {
            "sl_atr": sl_atr,
            "tp_atr": tp_atr,
            "sl_price": float(close_price - (sl_atr * atr)) if recommended_action == 1 else float(close_price + (sl_atr * atr)),
            "tp_price": float(close_price + (tp_atr * atr)) if recommended_action == 1 else float(close_price - (tp_atr * atr)),
        },
        "attention": {
            "mean_weights_240": mean_attn,
            "head_weights": attn_by_head,
            "top_5_candles": attn_summary.get("top_k_indices", []) if attn_summary else [],
            "recency_ratio": recency_ratio,
            "recency_weight_ratio": recency_ratio,
            "entropy": mean_entropy,
            "head_entropy": head_entropy,
            "peak_indices": peak_indices,
            "last_candle_weight": float(attn_summary.get("last_candle_weight", mean_attn[-1])) if attn_summary else mean_attn[-1],
        },
        "multi_horizon": {
            "horizons": [5, 15, 30],
            "pred_returns": [round(x, 4) for x in pred_returns],
            "projections": projections,
            "consensus_direction": consensus_dir,
        },
        "feature_groups": feature_groups,
        "cot_hops": hops,
    }


@app.get("/api/tca_heatmap")
@app.get("/api/tca/attention")
def get_tca_heatmap(bar_idx: int = Query(..., ge=0)) -> Dict[str, Any]:
    """
    Dedicated unified endpoint for TCA Heatmap Visualization (Feature 16).
    Returns complete atomic payload:
    - 240 OHLCV candles ending at bar_idx
    - 4 attention heads of length 240
    - Multi-horizon return predictions [t+5, t+15, t+30]
    - Complete attention telemetry metrics
    """
    initialize_runtime()
    env = state["env"]
    wm = state["world_model"]
    head = state["projection_head"]

    if env is None or bar_idx >= state["total_bars"]:
        raise HTTPException(status_code=400, detail="Invalid bar index or environment uninitialized")

    obs_137 = env.get_observation(bar_idx)
    candle_seq = env.get_candle_sequence(bar_idx, lookback=240)
    atr = float(env.atr_arr[bar_idx])
    close_price = float(env.close_arr[bar_idx])

    with torch.no_grad():
        z_out = wm.encode_state(obs_137, candle_seq=candle_seq)
        attn_summary = wm.get_attention_summary()
        raw_attn = wm.get_attention_weights()

        pred_returns = [0.0, 0.0, 0.0]
        if head is not None:
            preds_tensor = head(z_out)
            pred_returns = preds_tensor.squeeze(0).cpu().numpy().tolist()

    # 1. Format 240 OHLCV candles
    candles = extract_candles_for_window(env, bar_idx, lookback=240)

    # 2. Extract 4 Attention Heads of length 240
    if raw_attn is not None:
        attn_np = raw_attn.squeeze(0).cpu().numpy()  # [4, 240]
        mean_attn = np.mean(attn_np, axis=0).tolist()
        heads = [attn_np[h].tolist() for h in range(4)]
        peak_indices = [int(np.argmax(attn_np[h])) for h in range(4)]
    else:
        mean_attn = [1.0 / 240] * 240
        heads = [[1.0 / 240] * 240 for _ in range(4)]
        peak_indices = [239] * 4

    recency_ratio = (
        float(attn_summary.get("recency_weight_ratio", attn_summary.get("recency_ratio", 0.0)))
        if attn_summary
        else 0.0
    )
    head_entropy = (
        [float(e) for e in attn_summary.get("head_entropy", [0.0] * 4)] if attn_summary else [0.0] * 4
    )
    mean_entropy = float(np.mean(head_entropy)) if head_entropy else 0.0

    # 3. Format Multi-Horizon Predictions
    p5, p15, p30 = pred_returns
    details = {
        "t+5": {
            "horizon_min": 5,
            "atr_normalized_return": round(p5, 4),
            "price_delta": round(p5 * atr, 2),
            "projected_price": round(close_price + p5 * atr, 2),
            "signal": "BULLISH" if p5 > 0.05 else ("BEARISH" if p5 < -0.05 else "NEUTRAL"),
        },
        "t+15": {
            "horizon_min": 15,
            "atr_normalized_return": round(p15, 4),
            "price_delta": round(p15 * atr, 2),
            "projected_price": round(close_price + p15 * atr, 2),
            "signal": "BULLISH" if p15 > 0.05 else ("BEARISH" if p15 < -0.05 else "NEUTRAL"),
        },
        "t+30": {
            "horizon_min": 30,
            "atr_normalized_return": round(p30, 4),
            "price_delta": round(p30 * atr, 2),
            "projected_price": round(close_price + p30 * atr, 2),
            "signal": "BULLISH" if p30 > 0.05 else ("BEARISH" if p30 < -0.05 else "NEUTRAL"),
        },
    }

    consensus = "NEUTRAL"
    if p5 > 0 and p15 > 0:
        consensus = "BULLISH"
    elif p5 < 0 and p15 < 0:
        consensus = "BEARISH"

    return {
        "bar_idx": bar_idx,
        "timestamp": candles[-1]["time"],
        "datetime": candles[-1]["datetime"],
        "checkpoint": {
            "path": state["checkpoint_path"],
            "phase": state["checkpoint_type"],
            "loaded": state["checkpoint_loaded"],
            "tca_trained": state["tca_trained"],
            "head_trained": state["head_trained"],
        },
        "candles": candles,
        "attention": {
            "heads": heads,
            "mean_weights_240": mean_attn,
            "top_5_candles": attn_summary.get("top_k_indices", []) if attn_summary else [],
            "telemetry": {
                "entropy_per_head": head_entropy,
                "mean_entropy": mean_entropy,
                "peak_indices": peak_indices,
                "recency_ratio": recency_ratio,
                "recency_weight_ratio": recency_ratio,
                "last_candle_weight": float(attn_summary.get("last_candle_weight", mean_attn[-1])) if attn_summary else mean_attn[-1],
            },
        },
        "multi_horizon_predictions": {
            "horizons": [5, 15, 30],
            "predicted_returns_atr": [round(x, 4) for x in pred_returns],
            "details": details,
            "consensus_direction": consensus,
        },
    }


# Mount static directory for frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index() -> Any:
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Dashboard frontend is being initialized. Please refresh shortly."}
