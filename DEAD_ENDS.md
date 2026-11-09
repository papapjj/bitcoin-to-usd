# Dead Ends Log

| Iteration | Approach Tried | Why It Failed | Files Touched |
|-----------|---------------|---------------|---------------|
| M1 Iter 1 | Implementing only `act_nam_tac_nhan` on `DualChampionRouter` without `act()` | `mt5_runner.py:11948` explicitly checks `if hasattr(policy, "act"):` at the gateway of `_serve_policy_actions`. Omitting `act()` caused runtime dispatch to be skipped and line 12361 raised `RuntimeError: Bridge chỉ chấp nhận HeChuyenGiaMotContextMotPolicy đã xác minh owner`. | `src/self_growing_trader/dual_champion_router.py` |
| M2 Iter 1 | Assuming `request.regime_probabilities` always has length >= 3 when not None; accessing `soft_probs[0..2]` directly in `_five_bridge_action_packet` | When `regime_probabilities` is an empty list `[]` or truncated `[0.0, 0.0]`, direct indexing at lines 9320-9322 in `mt5_runner.py` raises unhandled `IndexError` instead of falling back to neutral prior and safe HOLD action. | `src/self_growing_trader/mt5_runner.py` |

