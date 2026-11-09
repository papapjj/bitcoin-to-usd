#!/usr/bin/env python3
"""Fail-fast static contract checks for the current MT5 sources.

This is source-level validation only. It never claims a MetaEditor compile,
Strategy Tester execution, broker acceptance, or numerical feature parity.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EA = ROOT / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
FEATURES = ROOT / "mt5_v2_runner" / "MQL5" / "Include" / "compute_features.mqh"
EXECUTION_CONTRACT = ROOT / "mt5_v2_runner" / "MQL5" / "Include" / "self_growing_execution_contract.mqh"


def require(source: str, marker: str, label: str) -> None:
    if marker not in source:
        raise AssertionError(f"{label}: missing {marker!r}")


def main() -> int:
    ea = EA.read_text(encoding="utf-8")
    features = FEATURES.read_text(encoding="utf-8")
    execution_contract = EXECUTION_CONTRACT.read_text(encoding="utf-8")
    for marker in (
        '#property version   "8.13"',
        "#define BRIDGE_SCHEMA_VERSION 10",
        "#define EXECUTION_LEDGER_SCHEMA_VERSION 17",
        'input string FeatureMtfScalingMode = "legacy_m1_atr_v1";',
        "input int MaxPositions = 0;",
        "MaxPositions > 0 && CountPositions() + CountPolicyOrders() >= MaxPositions",
        "positionCountReference = (MaxPositions > 0) ? MaxPositions : MAX_POSITION_SLOTS",
        "#define POSITION_LIFECYCLE_MAX 4096",
        "SYMBOL_TRADE_TICK_SIZE",
        "SYMBOL_TRADE_TICK_VALUE",
        "FeatureAuditFile",
        "AppendFeatureAudit",
        "V10: 137-input bridge",
        "ExitCounterfactualHorizonBars",
        "LifecycleInitialRiskDistancePrice",
        "POST_EXIT_OBSERVATION",
        "RecordIntentRequestedGeometry",
        "requestedEntryPrice",
        "#include \"../Include/self_growing_execution_contract.mqh\"",
    ):
        require(ea, marker, "EA")
    for marker in (
        "#define FEATURE_COUNT",
        "#define TIMING_FEATURE_COUNT 4",
        "#define WYCKOFF_FEATURE_COUNT 16",
        "g_bbHandle",
        "Feature schema v10 appends four timing and sixteen Wyckoff inputs",
        "SelectedMarketVolumeAt",
        "SpringScoreAt",
        "UpthrustScoreAt",
        "ComputeTailFeatures",
        "g_macdHandle = iMACD",
        "GetBufAt(g_bbHandle, 1, 1)",
        "features[i++] = (ema20 - ema20p) / atr;",
        "features[i++] = (ema20 - ema60) / atr;",
    ):
        require(features, marker, "feature header")
    for marker in (
        "#define SG_EXECUTION_CONTRACT_VERSION 7",
        "#define SG_ACTION_DELAY_OUTPUT_INDEX 13",
        "#define SG_FACTORIZED_ACTION_MARKER_OUTPUT_INDEX 14",
        "#define SG_ENTRY_OFFSET_OUTPUT_INDEX 15",
        "#define SG_GEOMETRY_PACKED_OUTPUT_INDEX 16",
        "#define SG_LEGACY_FACTORIZED_ACTION_MARKER_VALUE 6.0",
        "#define SG_FACTORIZED_ACTION_MARKER_VALUE 7.0",
        "#define SG_LEGACY_SL_BIN_COUNT 7",
        "#define SG_SL_BIN_COUNT 8",
        "#define SG_LEGACY_GEOMETRY_PACKED_BIN_COUNT 280",
        "#define SG_GEOMETRY_PACKED_BIN_COUNT 640",
    ):
        require(execution_contract, marker, "execution contract")
    # Regression: EMA diff must occupy exactly one slot.
    if features.count("features[i++] = (ema20 - ema60) / atr;") != 1:
        raise AssertionError("feature header: EMA diff must occupy exactly one slot")
    print("MT5 static contract checks: OK (static only; no MQL compile performed)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MT5 static contract checks: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
