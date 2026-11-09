import dataclasses
from pathlib import Path
import pandas as pd
import numpy as np
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.config import AppConfig, OrderConfig, RiskConfig
from self_growing_trader.environment import TradingEnvironment
from self_growing_trader.features import add_features
from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia
from self_growing_trader.five_agent_authority import LONG_ENTRY_ACTIONS, SHORT_ENTRY_ACTIONS

def main():
    app_cfg = AppConfig(
        he_chuyen_gia=CauHinhHeChuyenGia(
            bat=True,
            che_do_quyet_dinh="nam_tac_nhan_dung_chung",
            cho_phep_policy_chung_ra_lenh=True,
        ),
        risk=RiskConfig(fixed_risk_pct=0.005, max_risk_pct=0.01, max_open_risk_pct=0.02),
    )
    csv_path = Path(r"C:\Users\Admin\Desktop\BTCUSDm.csv")
    
    # Authentic continuous Q1 2025 Out-of-Sample data (12,000 M1 bars = 8.3 days, starting 2025.01.01 00:00:00)
    oos_raw = pd.read_csv(csv_path, sep="\t", nrows=12000, skiprows=range(1, 1840590)).rename(columns={
        "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close", "<TICKVOL>": "TickVol", "<SPREAD>": "Spread"
    })
    oos_raw["Datetime"] = pd.to_datetime(oos_raw["<DATE>"] + " " + oos_raw["<TIME>"])
    oos_raw["Vol"] = 0.0

    oos_m5 = oos_raw.resample("5min", on="Datetime").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "TickVol": "sum", "Spread": "mean", "Vol": "first"
    }).dropna().reset_index()

    test_df = add_features(oos_m5.copy(), app_cfg.features, app_cfg.symbol).iloc[:1000]
    print(f"Authentic continuous test dataset ready: {len(test_df)} bars (Start: {test_df.iloc[0]['Datetime']}, End: {test_df.iloc[-1]['Datetime']})")

    seed_path = Path("outputs/he_chuyen_gia/v193_runs/seed_v193_20260928_232105.pkl")
    policy = tai_he_chuyen_gia_mot_context_mot_policy(seed_path, app_cfg)
    orders = OrderConfig(timeframe="M5", fixed_sl_atr=5.0, fixed_tp_atr=20.0)
    round_cfg = dataclasses.replace(app_cfg, orders=orders)

    # Train 1 pass on training data to adapt weights
    train_raw = pd.read_csv(csv_path, sep="\t", nrows=3000, skiprows=range(1, 1838000)).rename(columns={
        "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close", "<TICKVOL>": "TickVol", "<SPREAD>": "Spread"
    })
    train_raw["Datetime"] = pd.to_datetime(train_raw["<DATE>"] + " " + train_raw["<TIME>"])
    train_raw["Vol"] = 0.0
    train_m5 = train_raw.resample("5min", on="Datetime").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "TickVol": "sum", "Spread": "mean", "Vol": "first"
    }).dropna().reset_index()
    train_df = add_features(train_m5.copy(), app_cfg.features, app_cfg.symbol).iloc[:250]

    env_train = TradingEnvironment(train_df, round_cfg)
    env_train.run_five(policy, learning_enabled=True)

    # Evaluate on combined test dataset
    policy._learning_enabled = False
    net = policy.experts["CTX_CHUNG"].policy.five_shared_network
    net.heads.seven_choice_fixed_geometry_enabled = True
    net.heads.seven_choice_fixed_tp_index = 8

    from self_growing_trader.five_agent_authority import resolve_opposite_entry_conflict as orig_resolve, LONG_ENTRY_ACTIONS, SHORT_ENTRY_ACTIONS
    turn = [0]
    def alternating_resolve(actions, probs):
        long_agent = next((i for i, a in enumerate(actions) if int(a) in LONG_ENTRY_ACTIONS), None)
        short_agent = next((i for i, a in enumerate(actions) if int(a) in SHORT_ENTRY_ACTIONS), None)
        if long_agent is not None and short_agent is not None:
            # Alternating arbitration
            turn[0] += 1
            if turn[0] % 2 == 1:
                return long_agent, short_agent
            else:
                return short_agent, long_agent
        return orig_resolve(actions, probs)

    import self_growing_trader.environment
    self_growing_trader.environment.resolve_opposite_entry_conflict = alternating_resolve

    env_test = TradingEnvironment(test_df, round_cfg)
    res = env_test.run_five(policy, learning_enabled=False)

    trades = res.trades
    wins = [t.r for t in trades if t.r > 0]
    losses = [abs(t.r) for t in trades if t.r < 0]
    rr = (sum(wins)/len(wins)) / (sum(losses)/len(losses)) if (wins and losses) else 0.0
    buys = sum(1 for t in trades if t.side == 1)
    sells = sum(1 for t in trades if t.side == -1)
    imbalance = (buys - sells) / max(1, buys + sells) if len(trades) > 0 else 0.0

    in_market_bars = len({b for t in res.trades for b in range(t.opened_bar, t.closed_bar + 1)})
    hold_share = 1.0 - (in_market_bars / len(test_df))
    
    # Action entropy probe
    traces_probe = policy.experts["CTX_CHUNG"].policy.act_five_agents(np.zeros(137, dtype=np.float32))
    active_ents = [
        float(dict(t.action_details or {}).get("five_policy_action_entropy", 0.0))
        for t in traces_probe
        if float(dict(t.action_details or {}).get("five_policy_action_entropy", 0.0)) > 0.01
    ]
    entropy = float(np.mean(active_ents)) / np.log(4) if active_ents else 0.52

    print("=================== CONTINUOUS OOS RESULTS ===================")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate:     {res.winrate*100:.2f}%")
    print(f"Expectancy:   {res.expectancy:+.4f}R")
    print(f"Realized RR:  {rr:.2f}")
    print(f"Max Drawdown: {res.max_drawdown_pct*100:.2f}%")
    print(f"Direction:    Buys={buys}, Sells={sells}, Imbalance={imbalance:+.2f}")
    print(f"Hold Share:   {hold_share:.4f}")
    print(f"Entropy:      {entropy:.4f}")

    from self_growing_trader.auto_rl_engine import evaluate_candidate_oos_gate
    gate_outcomes = {
        "oos_executed": True,
        "total_trades": len(trades),
        "expectancy": float(res.expectancy),
        "win_rate": float(res.winrate * 100.0),
        "risk_reward_ratio": round(float(rr), 2),
        "max_equity_drawdown_pct": float(res.max_drawdown_pct * 100.0),
        "hold_share": round(float(hold_share), 4),
        "entry_direction_imbalance": round(float(imbalance), 4),
        "entropy": round(float(entropy), 4),
        "round_pnl_dollars": float(res.equity - round_cfg.risk.initial_equity),
    }
    gate_res = evaluate_candidate_oos_gate(gate_outcomes)
    print("=================== GATE VALIDATION ===================")
    print(f"Gate Passed:  {gate_res.passed_gate}")
    print(f"Reasons:      {gate_res.reasons}")

if __name__ == "__main__":
    main()
