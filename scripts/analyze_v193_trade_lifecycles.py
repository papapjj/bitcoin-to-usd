"""Analyze completed MT5 position lifecycles from the execution ledger.

This is a read-only research utility.  It joins entry fills with every closing
deal by position ticket, so partial closes are not mistaken for independent
positions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ENTRY_ACTIONS = {
    "MARKET_BUY", "MARKET_SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP",
}


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _geometry(row: pd.Series) -> dict[str, object]:
    values = str(row.get("geometry_entry_pipe", "") or "").split("|")
    return {
        "atr": _number(values[0]) if len(values) > 0 else 0.0,
        "side": int(_number(values[1])) if len(values) > 1 else 0,
        "proposed_sl_atr": _number(values[2]) if len(values) > 2 else 0.0,
        "proposed_rr": _number(values[3]) if len(values) > 3 else 0.0,
        "proposed_tp_atr": _number(values[4]) if len(values) > 4 else 0.0,
        "mode": values[9] if len(values) > 9 else "",
        "effective_sl_atr": _number(values[10]) if len(values) > 10 else 0.0,
        "effective_tp_atr": _number(values[11]) if len(values) > 11 else 0.0,
        "effective_rr": _number(values[12]) if len(values) > 12 else 0.0,
        "trailing_atr": _number(values[13]) if len(values) > 13 else 0.0,
        "pending_fill_bars": _number(values[14]) if len(values) > 14 else 0.0,
    }


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _ordered_unique_text(values: pd.Series) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _text(value)
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _summary(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {"positions": 0, "net_pnl": 0.0}
    pnl = rows["net_pnl"].astype(float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "positions": int(len(rows)),
        "wins": int((pnl > 0.0).sum()),
        "losses": int((pnl < 0.0).sum()),
        "win_rate": float((pnl > 0.0).mean()),
        "net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "median_age_bars": float(rows["age_bars"].median()),
        "median_sl_atr": float(rows["effective_sl_atr"].median()),
        "median_rr": float(rows["effective_rr"].median()),
    }


def _deal_summary(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {"deals": 0, "net_pnl": 0.0}
    pnl = rows["deal_net_pnl"].astype(float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "deals": int(len(rows)),
        "wins": int((pnl > 0.0).sum()),
        "losses": int((pnl < 0.0).sum()),
        "win_rate": float((pnl > 0.0).mean()),
        "net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
    }


def analyze(path: Path) -> dict[str, object]:
    ledger = pd.read_csv(path, sep="\t", low_memory=False)
    fills = ledger[(ledger["filled"] == 1) & ledger["action_name"].isin(ENTRY_ACTIONS)].copy()
    closes = ledger[ledger["closed"] == 1].copy()
    final = closes[closes["position_final_close"] == 1].copy()
    records: list[dict[str, object]] = []
    for ticket, terminal in final.groupby("position_ticket", sort=False):
        terminal_row = terminal.iloc[-1]
        position_closes = closes[closes["position_ticket"] == ticket]
        position_fills = fills[fills["position_ticket"] == ticket]
        entry = position_fills.iloc[0] if not position_fills.empty else terminal_row
        geometry = _geometry(entry)
        close_reasons = _ordered_unique_text(position_closes["close_reason"])
        final_close_reasons = _ordered_unique_text(terminal["close_reason"])
        pnl = sum(
            _number(row.get("profit")) + _number(row.get("commission")) + _number(row.get("swap"))
            for _, row in position_closes.iterrows()
        )
        entry_time = pd.to_datetime(entry.get("logged_at"), format="%Y.%m.%d %H:%M:%S", errors="coerce")
        records.append({
            "position_ticket": str(ticket),
            "entry_action": str(entry.get("action_name", "SYSTEM")),
            "entry_time": "" if pd.isna(entry_time) else entry_time.isoformat(),
            "entry_hour": -1 if pd.isna(entry_time) else int(entry_time.hour),
            "entry_day": "" if pd.isna(entry_time) else entry_time.date().isoformat(),
            "close_reason": "+".join(close_reasons),
            "close_reasons": close_reasons,
            "final_close_reason": "+".join(final_close_reasons),
            "final_close_reasons": final_close_reasons,
            "exit_kind": str(terminal_row.get("exit_kind", "")),
            "net_pnl": float(pnl),
            "age_bars": _number(terminal_row.get("age_bars")),
            "initial_risk": _number(terminal_row.get("initial_risk")),
            "close_deals": int(len(position_closes)),
            "final_close_rows": int(len(terminal)),
            **geometry,
        })
    positions = pd.DataFrame(records)
    if positions.empty:
        raise RuntimeError("khong co vi the da dong hoan tat")

    def grouped(column: str) -> dict[str, object]:
        return {
            str(key): _summary(group)
            for key, group in positions.groupby(column, dropna=False, sort=True)
        }

    closes["deal_net_pnl"] = closes.apply(
        lambda row: _number(row.get("profit")) + _number(row.get("commission")) + _number(row.get("swap")),
        axis=1,
    )
    closes["close_reason_group"] = closes["close_reason"].map(_text)

    def grouped_deals(column: str) -> dict[str, object]:
        return {
            str(key): _deal_summary(group)
            for key, group in closes.groupby(column, dropna=False, sort=True)
        }

    positions["age_group"] = pd.cut(
        positions["age_bars"], [-np.inf, 2, 5, 10, 20, 40, np.inf],
        labels=["<=2", "2-5", "5-10", "10-20", "20-40", ">40"], right=True,
    ).astype(str)
    positions["sl_atr_group"] = positions["effective_sl_atr"].map(lambda x: f"{x:.2f}")
    positions["rr_group"] = positions["effective_rr"].map(lambda x: f"{x:.2f}")

    ordered = positions.sort_values(["entry_time", "position_ticket"], kind="stable")
    worst_streak = 0
    current_streak = 0
    streak_loss = 0.0
    worst_streak_loss = 0.0
    for pnl in ordered["net_pnl"]:
        if pnl < 0.0:
            current_streak += 1
            streak_loss += float(pnl)
            if current_streak > worst_streak:
                worst_streak = current_streak
            worst_streak_loss = min(worst_streak_loss, streak_loss)
        else:
            current_streak = 0
            streak_loss = 0.0

    closed_deal_rows = int(len(closes))
    final_close_rows = int(len(final))
    unique_completed_positions = int(len(positions))
    duplicate_final_close_rows = final_close_rows - unique_completed_positions
    lifecycle_close_reason = grouped("close_reason")

    return {
        "schema_version": 2,
        "ledger": str(path.resolve()),
        "method": "join_entry_fill_and_all_close_deals_by_position_ticket",
        "closed_deal_rows": closed_deal_rows,
        "final_close_rows": final_close_rows,
        "unique_completed_positions": unique_completed_positions,
        "duplicate_final_close_rows": duplicate_final_close_rows,
        "overall": _summary(positions),
        "closed_deals": _deal_summary(closes),
        "by_entry_action": grouped("entry_action"),
        "by_close_reason": lifecycle_close_reason,
        "by_lifecycle_close_reason": lifecycle_close_reason,
        "by_deal_close_reason": grouped_deals("close_reason_group"),
        "by_sl_atr": grouped("sl_atr_group"),
        "by_rr": grouped("rr_group"),
        "by_age_bars": grouped("age_group"),
        "by_entry_hour": grouped("entry_hour"),
        "by_entry_day": grouped("entry_day"),
        "geometry_modes": positions["mode"].value_counts().to_dict(),
        "trailing_atr_counts": positions["trailing_atr"].map(lambda x: f"{x:.2f}").value_counts().to_dict(),
        "long_short": grouped("side"),
        "worst_consecutive_losses": worst_streak,
        "worst_consecutive_loss_sum": worst_streak_loss,
        "positions": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **result["overall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
