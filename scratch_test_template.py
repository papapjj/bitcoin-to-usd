import torch
import numpy as np
from pathlib import Path
from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy, _cau_hinh_policy_chuyen_gia_cho_mt5
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

cfg = load_config('configs/he_chuyen_gia_mot_context_mot_policy.yaml')
base_checkpoint = Path('outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl')
base_policy = tai_he_chuyen_gia_mot_context_mot_policy(base_checkpoint, cfg, seed=2026)
_cau_hinh_policy_chuyen_gia_cho_mt5(base_policy, cfg)

p = MythosMT5Policy(base_policy=base_policy)
tmpl = p._base_trace_template
for idx, t in enumerate(tmpl):
    act = getattr(t, "action", None)
    act_idx = getattr(t, "action_index", None)
    sel_idx = getattr(t, "selected_idx", None)
    det = getattr(t, "action_details", {})
    wire = det.get("five_action_index", None) if isinstance(det, dict) else None
    print(f"Trace {idx}: action={act}, action_index={act_idx}, selected_idx={sel_idx}, wire={wire}")
