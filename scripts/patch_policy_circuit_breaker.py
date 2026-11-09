import re
from pathlib import Path

# --- 1. Patch research_reasoning_world_class/src/mythos_mt5_policy.py ---
policy_path = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
content = policy_path.read_text(encoding="utf-8")

# A. Patch __init__ defaults
old_init_guards = """        # Circuit breaker and directional loss guards
        self._consecutive_losses: Dict[int, int] = {1: 0, -1: 0}
        self._direction_cooldown_until: Dict[int, int] = {1: 0, -1: 0}
        self.max_consecutive_losses: int = 2
        self.loss_cooldown_bars: int = 4
        self.prm_threshold: float = 0.65
        self.veto_similarity_threshold: float = 0.92
        if hasattr(self, "mad"): self.mad.veto_similarity_threshold = self.veto_similarity_threshold
        self.sniper_sl_atr: float = 2.0
        self.sniper_tp_rr: float = 2.50
        self.scalp_sl_atr: float = 2.0
        self.scalp_tp_rr: float = 2.50
        self.trailing_atr: float = 0.75"""

new_init_guards = """        # Circuit breaker and global/directional loss guards
        self._consecutive_losses: Dict[int, int] = {1: 0, -1: 0}
        self._global_consecutive_losses: int = 0
        self._global_cooldown_until: int = 0
        self._direction_cooldown_until: Dict[int, int] = {1: 0, -1: 0}
        self.max_consecutive_losses: int = 2
        self.loss_cooldown_bars: int = 45
        self.prm_threshold: float = 0.30
        self.veto_similarity_threshold: float = 0.92
        if hasattr(self, "mad"): self.mad.veto_similarity_threshold = self.veto_similarity_threshold
        self.sniper_sl_atr: float = 4.5
        self.sniper_tp_rr: float = 0.75
        self.scalp_sl_atr: float = 4.5
        self.scalp_tp_rr: float = 0.75
        self.trailing_atr: float = 1.50
        self.breakeven_trigger_atr: float = 1.80
        self.breakeven_lock_atr: float = 1.00"""

if old_init_guards in content:
    content = content.replace(old_init_guards, new_init_guards, 1)
    print("Patched __init__ guards")
else:
    print("Warning: old_init_guards not exact match, checking partial...")
    content = re.sub(
        r"self\._consecutive_losses: Dict\[int, int\] = \{1: 0, -1: 0\}.*?self\.trailing_atr: float = [0-9\.]+",
        new_init_guards,
        content,
        flags=re.DOTALL
    )
    print("Re-subbed __init__ guards")

# B. Patch _sync_closed_deals consecutive loss tracking
old_sync_losses = """                                if deal_dir != 0 and hasattr(self, "_consecutive_losses"):
                                    if profit < 0:
                                        self._consecutive_losses[deal_dir] = self._consecutive_losses.get(deal_dir, 0) + 1
                                        self._consecutive_losses[-deal_dir] = 0
                                        if self._consecutive_losses[deal_dir] >= getattr(self, "max_consecutive_losses", 3):
                                            cd_bars = getattr(self, "loss_cooldown_bars", 8)
                                            self._direction_cooldown_until[deal_dir] = self.seq_counter + cd_bars
                                        if hasattr(self, "mad") and self._last_reasoning_z is not None:
                                            mad_act = 1 if deal_dir == 1 else (2 if deal_dir == -1 else 0)
                                            if mad_act != 0:
                                                self.mad.record_reflection(
                                                    state_embedding=self._last_reasoning_z,
                                                    trigger_price=self.current_price,
                                                    regime="mt5_live_trap",
                                                    action_taken=mad_act,
                                                    pnl=profit,
                                                    mae=mae_val or 1.5,
                                                )
                                    elif profit > 0:
                                        self._consecutive_losses[deal_dir] = 0"""

new_sync_losses = """                                if hasattr(self, "_consecutive_losses"):
                                    if profit < 0:
                                        self._global_consecutive_losses = getattr(self, "_global_consecutive_losses", 0) + 1
                                        if self._global_consecutive_losses >= getattr(self, "max_consecutive_losses", 2):
                                            cd_bars = getattr(self, "loss_cooldown_bars", 45)
                                            self._global_cooldown_until = self.seq_counter + cd_bars
                                        if deal_dir != 0:
                                            self._consecutive_losses[deal_dir] = self._consecutive_losses.get(deal_dir, 0) + 1
                                            if self._consecutive_losses[deal_dir] >= getattr(self, "max_consecutive_losses", 2):
                                                cd_bars = getattr(self, "loss_cooldown_bars", 45)
                                                self._direction_cooldown_until[deal_dir] = self.seq_counter + cd_bars
                                        if hasattr(self, "mad") and self._last_reasoning_z is not None:
                                            mad_act = 1 if deal_dir == 1 else (2 if deal_dir == -1 else 0)
                                            if mad_act != 0:
                                                self.mad.record_reflection(
                                                    state_embedding=self._last_reasoning_z,
                                                    trigger_price=self.current_price,
                                                    regime="mt5_live_trap",
                                                    action_taken=mad_act,
                                                    pnl=profit,
                                                    mae=mae_val or 1.5,
                                                )
                                    elif profit > 0:
                                        self._global_consecutive_losses = 0
                                        if deal_dir != 0:
                                            self._consecutive_losses[deal_dir] = 0"""

assert old_sync_losses in content, "old_sync_losses pattern not found"
content = content.replace(old_sync_losses, new_sync_losses, 1)
print("Patched _sync_closed_deals loss tracking")

# C. Patch act_nam_tac_nhan to preserve Agent 2 management action and check global cooldown
old_agent2_hold = """        traces_list[2] = self._enforce_hold(traces_list[2], reason="AGENT2_HOLD")"""
new_agent2_hold = """        if self._extract_action_idx(raw_traces[2]) == 11:
            traces_list[2] = raw_traces[2]
        else:
            traces_list[2] = self._enforce_hold(traces_list[2], reason="AGENT2_HOLD")"""
assert old_agent2_hold in content, "old_agent2_hold not found"
content = content.replace(old_agent2_hold, new_agent2_hold, 1)
print("Patched Agent 2 management preservation")

# D. Patch global loss cooldown guard
old_dir_cd = """        # Directional Loss Cooldown Guard: after consecutive losses, pause that direction"""
new_dir_cd = """        # Global Loss Cooldown Guard: after consecutive losses, pause all trading
        if net_dir != 0 and hasattr(self, "_global_cooldown_until"):
            if self.seq_counter < getattr(self, "_global_cooldown_until", 0):
                net_dir = 0

        # Directional Loss Cooldown Guard: after consecutive losses, pause that direction"""
assert old_dir_cd in content, "old_dir_cd not found"
content = content.replace(old_dir_cd, new_dir_cd, 1)
print("Patched Global Loss Cooldown Guard")

policy_path.write_text(content, encoding="utf-8")
print("Saved mythos_mt5_policy.py successfully")
