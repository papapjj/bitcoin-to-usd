import dataclasses
from pathlib import Path
import pandas as pd
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.config import AppConfig, OrderConfig, RiskConfig
from self_growing_trader.environment import TradingEnvironment
from self_growing_trader.features import add_features
from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia

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
    oos_raw = pd.read_csv(csv_path, sep="\t", nrows=15000, skiprows=range(1, 1840590)).rename(columns={
        "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close", "<TICKVOL>": "TickVol", "<SPREAD>": "Spread"
    })
    oos_raw["Datetime"] = pd.to_datetime(oos_raw["<DATE>"] + " " + oos_raw["<TIME>"])
    oos_raw["Vol"] = 0.0
    oos_m5_raw = oos_raw.resample("5min", on="Datetime").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "TickVol": "sum", "Spread": "mean", "Vol": "first"
    }).dropna().reset_index()
    oos_m5_prom_df = add_features(oos_m5_raw.copy(), app_cfg.features, app_cfg.symbol).iloc[:1200]
    print(f"Data ready: {len(oos_m5_prom_df)} bars (Start: {oos_m5_prom_df.iloc[0]['Datetime']}, End: {oos_m5_prom_df.iloc[-1]['Datetime']})")

    seed_path = Path("outputs/he_chuyen_gia/v193_runs/seed_v193_20260928_232105.pkl")
    policy = tai_he_chuyen_gia_mot_context_mot_policy(seed_path, app_cfg)
    policy._learning_enabled = False
    env = TradingEnvironment(oos_m5_prom_df, app_cfg)
    orders = OrderConfig(timeframe='M5', fixed_sl_atr=5.0, fixed_tp_atr=20.0)
    round_cfg = dataclasses.replace(app_cfg, orders=orders)
    net = policy.experts['CTX_CHUNG'].policy.five_shared_network
    net.heads.seven_choice_fixed_geometry_enabled = True
    net.heads.seven_choice_fixed_tp_index = 8
    env = TradingEnvironment(oos_m5_prom_df, round_cfg)

    env = TradingEnvironment(oos_m5_prom_df.iloc[:20], round_cfg)
    env.reset()
    traces = tuple(policy.act_nam_tac_nhan(env.observation(0)))
    print("Trace 0 action_details keys:", list(traces[0].action_details.keys()))
    print("Trace 0 five_selected_action_probability:", traces[0].action_details.get("five_selected_action_probability"))
    print("Trace 0 five_policy_action_entropy:", traces[0].action_details.get("five_policy_action_entropy"))
    for k, v in traces[0].action_details.items():
        if "prob" in k.lower() or "logit" in k.lower() or "score" in k.lower() or "hold" in k.lower():
            print(f"  T0: {k} = {v}")
    for k, v in traces[1].action_details.items():
        if "prob" in k.lower() or "logit" in k.lower() or "score" in k.lower() or "hold" in k.lower():
            print(f"  T1: {k} = {v}")

if __name__ == "__main__":
    main()
