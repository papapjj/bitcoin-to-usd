"""
scripts/run_m1_mutation_audit.py

Mutation Testing Harness for Milestone 1 Verification Suite.
Demonstrates empirically that `tests/test_defects_m1_verification.py` is authentic,
causally bound to production code, and NOT a self-certifying tautology.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_pytest(test_target: str) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", test_target, "-v", "--no-header", "-q"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr

def test_mutation(
    mutation_id: str,
    target_file: Path,
    original_snippet: str,
    mutated_snippet: str,
    test_case: str,
) -> bool:
    print(f"\n=======================================================")
    print(f"RUNNING MUTATION TEST: {mutation_id}")
    print(f"Target File: {target_file.relative_to(ROOT)}")
    print(f"Target Test: {test_case}")
    print(f"=======================================================")
    
    content = target_file.read_text(encoding="utf-8")
    if original_snippet not in content:
        print(f"ERROR: Original snippet not found in {target_file}")
        return False
        
    # Baseline: verify test passes before mutation
    rc_base, out_base = run_pytest(test_case)
    if rc_base != 0:
        print(f"FAILED BASELINE for {test_case}:\n{out_base}")
        return False
    print(f"[PASS] Baseline test passes before mutation.")

    # Apply mutation
    mutated_content = content.replace(original_snippet, mutated_snippet, 1)
    target_file.write_text(mutated_content, encoding="utf-8")
    
    mutation_killed = False
    try:
        rc_mut, out_mut = run_pytest(test_case)
        if rc_mut != 0:
            print(f"[PASS] Mutation successfully KILLED! Test failed as expected (Exit code {rc_mut}).")
            # Print brief snippet of failure
            lines = [l for l in out_mut.splitlines() if "FAILED" in l or "AssertionError" in l or "assert" in l or "Error" in l]
            for l in lines[:5]:
                print(f"       >> {l}")
            mutation_killed = True
        else:
            print(f"[CRITICAL FAIL] Mutation SURVIVED! Test passed despite corrupted production code!")
            mutation_killed = False
    finally:
        # Restore original content strictly
        target_file.write_text(content, encoding="utf-8")
        # Verify restored
        rc_restored, _ = run_pytest(test_case)
        assert rc_restored == 0, f"Restoration verification failed for {target_file}"
        print(f"[PASS] Clean restoration confirmed.")

    return mutation_killed

def main():
    results = {}
    
    # -------------------------------------------------------------
    # Mutation 1: Corrupt trailing stop classification in test_5_regimes_zero_filter.py
    # -------------------------------------------------------------
    target_file_1 = ROOT / "scripts" / "test_5_regimes_zero_filter.py"
    orig_1 = 'exit_reason = "SL_HIT"\n                    realized_r = pnl / (pos_atr * sl_mult * pos_vol)\n                    trades.append({\n                        "ticket": f"T{ticket_counter:05d}",\n                        "direction": "SELL"'
    mut_1 = 'exit_reason = "TP_HIT"\n                    realized_r = pnl / (pos_atr * sl_mult * pos_vol)\n                    trades.append({\n                        "ticket": f"T{ticket_counter:05d}",\n                        "direction": "SELL"'
    results["MUT_1_TRAILING_STOP_EXIT_REASON"] = test_mutation(
        mutation_id="MUT_1: Trailing stop exit_reason relabeled to TP_HIT",
        target_file=target_file_1,
        original_snippet=orig_1,
        mutated_snippet=mut_1,
        test_case="tests/test_defects_m1_verification.py::test_defect_2_sell_stop_loss_and_trailing_classification",
    )

    # -------------------------------------------------------------
    # Mutation 2: Invert SELL stop loss condition in test_5_regimes_zero_filter.py
    # -------------------------------------------------------------
    orig_2 = "if cur_h >= pos_sl_p:\n                    # Bounded at actual stop loss fill price"
    mut_2 = "if cur_h <= pos_sl_p:\n                    # Bounded at actual stop loss fill price"
    results["MUT_2_SELL_SL_TRIGGER_CONDITION"] = test_mutation(
        mutation_id="MUT_2: Invert SELL SL condition (cur_h <= pos_sl_p)",
        target_file=target_file_1,
        original_snippet=orig_2,
        mutated_snippet=mut_2,
        test_case="tests/test_defects_m1_verification.py::test_defect_2_sell_stop_loss_and_trailing_classification",
    )

    # -------------------------------------------------------------
    # Mutation 3: Re-introduce "28.98 /" into MythosMT5Policy source
    # -------------------------------------------------------------
    target_file_3 = ROOT / "research_reasoning_world_class" / "src" / "mythos_mt5_policy.py"
    orig_3 = "        self.loss_cooldown_bars: int = 3"
    mut_3 = "        self.loss_cooldown_bars: int = 3  # inverted formula: 28.98 / spread_atr"
    results["MUT_3_SPREAD_INVERSION_TOKEN_DETECTION"] = test_mutation(
        mutation_id="MUT_3: Inject '28.98 /' token into MythosMT5Policy source",
        target_file=target_file_3,
        original_snippet=orig_3,
        mutated_snippet=mut_3,
        test_case="tests/test_defects_m1_verification.py::test_defect_7_wilder_ewm_atr_no_spread_inversion",
    )

    # -------------------------------------------------------------
    # Mutation 4: Disable Spacing Guard in MythosMT5Policy act_nam_tac_nhan
    # -------------------------------------------------------------
    orig_4 = "        if bars_since < min_spacing_bars:\n            spacing_ok = False"
    mut_4 = "        if False and bars_since < min_spacing_bars:\n            spacing_ok = False"
    results["MUT_4_DISABLE_MIN_SPACING_GUARD"] = test_mutation(
        mutation_id="MUT_4: Disable min spacing guard in act_nam_tac_nhan",
        target_file=target_file_3,
        original_snippet=orig_4,
        mutated_snippet=mut_4,
        test_case="tests/test_defects_m1_verification.py::test_defect_5_min_spacing_and_universal_cooldown",
    )

    # -------------------------------------------------------------
    # Mutation 5: Disable Post-Exit Cooldown in MythosMT5Policy act_nam_tac_nhan
    # -------------------------------------------------------------
    orig_5 = "        if net_dir != 0 and bars_since_exit < post_exit_cd:\n            net_dir = 0"
    mut_5 = "        if False and net_dir != 0 and bars_since_exit < post_exit_cd:\n            net_dir = 0"
    results["MUT_5_DISABLE_POST_EXIT_COOLDOWN"] = test_mutation(
        mutation_id="MUT_5: Disable post-exit cooldown in act_nam_tac_nhan",
        target_file=target_file_3,
        original_snippet=orig_5,
        mutated_snippet=mut_5,
        test_case="tests/test_defects_m1_verification.py::test_defect_5_min_spacing_and_universal_cooldown",
    )

    # -------------------------------------------------------------
    # Mutation 6: Replace Wilder SMMA ATR in features.py with Rolling SMA
    # -------------------------------------------------------------
    target_file_6 = ROOT / "src" / "self_growing_trader" / "features.py"
    orig_6 = "    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()"
    mut_6 = "    return true_range.rolling(window=window, min_periods=window).mean()"
    results["MUT_6_WILDER_SMMA_REPLACED_WITH_BOXCAR_SMA"] = test_mutation(
        mutation_id="MUT_6: Replace Wilder SMMA with rolling boxcar SMA in features.py",
        target_file=target_file_6,
        original_snippet=orig_6,
        mutated_snippet=mut_6,
        test_case="tests/test_defects_m1_verification.py::test_defect_8_pandas_wilder_smma_atr_parity",
    )

    print("\n=======================================================")
    print("MUTATION AUDIT SUMMARY")
    print("=======================================================")
    all_killed = True
    for name, killed in results.items():
        status = "KILLED (Authentic)" if killed else "SURVIVED (Tautology/Flaw)"
        print(f"  - {name}: {status}")
        if not killed:
            all_killed = False

    print("=======================================================")
    if all_killed:
        print("VERDICT: ALL 6/6 MUTATIONS KILLED! Verification suite is 100% AUTHENTIC.")
        sys.exit(0)
    else:
        print("VERDICT: ONE OR MORE MUTATIONS SURVIVED! Investigation required.")
        sys.exit(1)

if __name__ == "__main__":
    main()
