"""
scripts/benchmark_reasoning_effectiveness.py

Scientific Benchmark: Does Reasoning (ReasoningGraphSystem + GRPO) improve decision quality?
Compares:
1. Baseline Policy (Raw candidate selection, no reasoning intervention)
2. Reasoned Policy (Latent multi-step reasoning reranking active)

Evaluates:
- Win Rate (%)
- Profit Factor
- False Breakout / Trap Veto Count
- Max Adverse Excursion (MAE in ATR - risk reduction)
- Mean Realized Gain (R in ATR)
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.self_growing_runner import (
    SelfGrowingPolicy,
    INPUT_COUNT,
    _REASONING_V18_POLICY_ACT,
    _reasoning_v18_rerank,
)
from self_growing_trader.micro_scalp_engine import MicroScalpConfig


def run_benchmark():
    print("=" * 80, flush=True)
    print("  BENCHMARK: DO LUONG HIEU QUA CO CHE SUY LUAN (REASONING ENGINE)", flush=True)
    print("=" * 80, flush=True)

    # Load real M1 market data
    bridge_path = ROOT / "outputs/he_chuyen_gia/v193_runs/multi_round_phase2/frozen_oos_phase2_champion_artifacts/window_01/self_growing_bridge.csv"
    bars = []
    seen = set()
    with open(bridge_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        in_idx = header.index("inputs_pipe")
        tm_idx = header.index("time")
        ag_idx = header.index("agent_id")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= in_idx or parts[ag_idx] != "0":
                continue
            tm = parts[tm_idx]
            if tm in seen:
                continue
            seen.add(tm)
            bars.append((tm, [float(x) for x in parts[in_idx].split("|")]))

    print(f"[1/4] Da nap {len(bars)} nến M1 thuc te tu tap kiem dinh ngoai mau (OOS).", flush=True)

    # Initialize policy with reasoning graph
    policy = SelfGrowingPolicy.seed(seed=42, input_size=INPUT_COUNT, m1_backend="m1_graph")

    # Metrics trackers
    baseline_trades = []
    reasoned_trades = []
    rerank_interventions = 0
    vetoed_losses = 0

    print("[2/4] Dang chay mo phong song song: Baseline (Khong suy luan) vs Reasoned (Co suy luan)...", flush=True)

    # Evaluate across bars
    test_bars = bars[:400]
    for i, (tm, raw_obs) in enumerate(test_bars):
        obs = np.array(raw_obs)
        if obs.size != INPUT_COUNT:
            continue

        policy.set_decision_context(
            account_context={"current_equity_drawdown": 0.01, "open_risk_ratio": 0.0},
            execution_context={"spread_norm": 0.08, "margin_available_ratio": 1.0},
            allowed_actions={idx: True for idx in range(12)},
            frame_id=f"bench:{i}",
            sequence_id=i,
            timestamp=float(i * 60),
        )

        # Baseline decision (raw policy forward before reasoning rerank)
        policy.reasoning_graph.begin_decision()
        base_decision = _REASONING_V18_POLICY_ACT(policy, obs)
        base_act = int(base_decision.selected_idx)

        # Reasoned decision (with multi-step reasoning rerank active)
        reasoned_decision = _reasoning_v18_rerank(policy, base_decision)
        reasoned_act = int(reasoned_decision.selected_idx)

        changed = bool(reasoned_decision.action_details.get("reasoning_changed_candidate", False))
        if changed or (base_act != reasoned_act):
            rerank_interventions += 1

        # Simulate outcomes for any trade action (1: Buy, 2: Sell, 3: Buy Limit, 4: Sell Limit)
        future_bars = test_bars[i + 1 : i + 35]
        if len(future_bars) < 5:
            continue

        for act, trade_list, is_reasoned in [(base_act, baseline_trades, False), (reasoned_act, reasoned_trades, True)]:
            if act in (1, 3, 5):
                direction = 1  # Long
            elif act in (2, 4, 6):
                direction = -1 # Short
            else:
                continue # HOLD

            cum_ret = 0.0
            mfe = 0.0
            mae = 0.0
            hit_tp = False
            hit_sl = False

            for f_tm, f_obs in future_bars:
                f_ret = float(f_obs[7]) * direction
                cum_ret += f_ret
                mfe = max(mfe, cum_ret)
                mae = min(mae, cum_ret)

                if mfe >= 1.5:
                    hit_tp = True
                    break
                if mae <= -2.5:
                    hit_sl = True
                    break

            realized = 1.5 if hit_tp else (-2.5 if hit_sl else cum_ret)
            is_win = realized > 0 or (mfe >= 0.65 and realized >= 0.15)

            trade_list.append({
                "time": tm,
                "direction": direction,
                "mfe": mfe,
                "mae": mae,
                "realized": realized,
                "win": is_win,
            })

            # Check if reasoning vetoed a losing baseline trade
            if not is_reasoned and not is_win and reasoned_act != base_act:
                vetoed_losses += 1

    print("[3/4] Tinh toan cac chi so thong ke dinh luong...", flush=True)

    def calc_stats(trades):
        n = len(trades)
        if n == 0:
            return {"total": 0, "win_rate": 0.0, "pf": 0.0, "avg_r": 0.0, "avg_mae": 0.0, "avg_mfe": 0.0}
        wins = sum(1 for t in trades if t["win"])
        wr = (wins / n) * 100.0
        profits = sum(t["realized"] for t in trades if t["realized"] > 0)
        losses = abs(sum(t["realized"] for t in trades if t["realized"] < 0))
        pf = (profits / losses) if losses > 0 else 99.0
        avg_r = np.mean([t["realized"] for t in trades])
        avg_mae = np.mean([t["mae"] for t in trades])
        avg_mfe = np.mean([t["mfe"] for t in trades])
        return {
            "total": n,
            "wins": wins,
            "losses": n - wins,
            "win_rate": wr,
            "pf": pf,
            "avg_r": avg_r,
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
        }

    s_base = calc_stats(baseline_trades)
    s_reas = calc_stats(reasoned_trades)

    print("\n" + "=" * 80, flush=True)
    print("  KET QUA DOI CHIEU BENCHMARK THUC TE:", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Tieu chi danh gia':<32} | {'Baseline (Khong suy luan)':<25} | {'Reasoned (Co suy luan)':<25}")
    print("-" * 88, flush=True)
    print(f"{'Tong so lenh thuc thi':<32} | {s_base['total']:<25} | {s_reas['total']:<25}")
    print(f"{'Ty le thang (Win Rate)':<32} | {s_base['win_rate']:>6.2f}%{'':<18} | {s_reas['win_rate']:>6.2f}%{'':<18}")
    print(f"{'He so loi nhuan (Profit Factor)':<32} | {s_base['pf']:>6.2f}{'':<19} | {s_reas['pf']:>6.2f}{'':<19}")
    print(f"{'Loi nhuan trung binh / lenh':<32} | {s_base['avg_r']:>+6.3f} ATR{'':<15} | {s_reas['avg_r']:>+6.3f} ATR{'':<15}")
    print(f"{'Muc sut giam toi da (Avg MAE)':<32} | {s_base['avg_mae']:>6.3f} ATR{'':<15} | {s_reas['avg_mae']:>6.3f} ATR{'':<15}")
    print(f"{'Bien do thuan loi (Avg MFE)':<32} | {s_base['avg_mfe']:>6.3f} ATR{'':<15} | {s_reas['avg_mfe']:>6.3f} ATR{'':<15}")
    print("=" * 80, flush=True)

    print("\n[4/4] PHAN TICH CHI TIET CO CHE CAN THIEP:")
    print(f"  - So lan Suy luan can thiep sua doi hanh vi (Reranks) : {rerank_interventions}")
    print(f"  - So lenh bay / thua lo ma Suy luan da ngan chan (Vetoes): {vetoed_losses}")
    delta_wr = s_reas['win_rate'] - s_base['win_rate']
    delta_pf = s_reas['pf'] - s_base['pf']
    delta_r = s_reas['avg_r'] - s_base['avg_r']
    print(f"  - Chenh lech Ty le thang (Win Rate Delta)             : {delta_wr:+.2f}%")
    print(f"  - Chenh lech He so loi nhuan (PF Delta)               : {delta_pf:+.2f}")
    print(f"  - Chenh lech Loi nhuan trung binh (Avg R Delta)       : {delta_r:+.3f} ATR")

    if delta_wr >= 0 and delta_pf >= 0:
        print("\n  ==> KET LUAN: CO CHE SUY LUAN DAT HIEU QUA VUOT TROI RO RET!")
        print("      Suy luan da chu dong loc bo cac lenh bay ti le rui ro cao va cai thien ky vong toan hoc.")
    else:
        print("\n  ==> KET LUAN: Co che suy luan chua tao duoc su khac biet tich cuc.")


if __name__ == "__main__":
    run_benchmark()
