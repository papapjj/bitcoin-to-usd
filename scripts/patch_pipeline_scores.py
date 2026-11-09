from pathlib import Path

path = Path("research_reasoning_world_class/src/unified_reasoning_pipeline.py")
content = path.read_text(encoding="utf-8")

# 1. Add bull_score and bear_score to PipelineTickResult
old_fields = """    total_latency_ms: float
    executed: bool
    is_counter_punch: bool = False"""

new_fields = """    total_latency_ms: float
    executed: bool
    bull_score: float = 0.0
    bear_score: float = 0.0
    is_counter_punch: bool = False"""

assert old_fields in content, "old_fields not found"
content = content.replace(old_fields, new_fields, 1)

# 2. Pass bull_score and bear_score and bilateral action to PipelineTickResult construction
old_ctor = """            confidence=conf_val,
            prm_score=prm_val,
            net_margin=margin_val,"""

new_ctor = """            confidence=conf_val,
            prm_score=prm_val,
            net_margin=margin_val,
            bull_score=bull_val,
            bear_score=bear_val,"""

assert old_ctor in content, "old_ctor not found"
content = content.replace(old_ctor, new_ctor, 1)

old_action = """            symbol=symbol,
            action=final_action,"""

new_action = """            symbol=symbol,
            action=action_val if getattr(self, "use_bilateral_dual_cot", False) else final_action,"""

assert old_action in content, "old_action not found"
content = content.replace(old_action, new_action, 1)

path.write_text(content, encoding="utf-8")
print("Successfully patched unified_reasoning_pipeline.py")
