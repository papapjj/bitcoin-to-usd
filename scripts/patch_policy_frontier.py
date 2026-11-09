import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
policy_path = ROOT / "research_reasoning_world_class" / "src" / "mythos_mt5_policy.py"
content = policy_path.read_text(encoding="utf-8")

old_block = """            prm_gate = getattr(self, "prm_threshold", 0.30)
            min_margin = getattr(self, "min_margin_threshold", 0.08)
            min_conf = getattr(self, "min_confidence", 0.35)
            bear_thresh = getattr(self, "bear_entry_threshold", 0.28)
            sell_margin = getattr(self, "bear_min_margin", 0.05)"""

new_block = """            prm_gate = getattr(self, "prm_threshold", 0.38)
            min_margin = getattr(self, "min_margin_threshold", 0.15)
            min_conf = getattr(self, "min_confidence", 0.58)
            bear_thresh = getattr(self, "bear_entry_threshold", 0.52)
            sell_margin = getattr(self, "bear_min_margin", 0.15)"""

# Handle CRLF / LF uniformly
content_lf = content.replace("\r\n", "\n")
if old_block in content_lf:
    content_lf = content_lf.replace(old_block, new_block)
    policy_path.write_text(content_lf, encoding="utf-8")
    print("Successfully patched mythos_mt5_policy.py with high-conviction thresholds!")
else:
    print("Old block not found directly, checking regex...")
    pattern = r'prm_gate = getattr\(self, "prm_threshold", 0\.\d+\)\s+min_margin = getattr\(self, "min_margin_threshold", 0\.\d+\)\s+min_conf = getattr\(self, "min_confidence", 0\.\d+\)\s+bear_thresh = getattr\(self, "bear_entry_threshold", 0\.\d+\)\s+sell_margin = getattr\(self, "bear_min_margin", 0\.\d+\)'
    new_sub = 'prm_gate = getattr(self, "prm_threshold", 0.38)\n            min_margin = getattr(self, "min_margin_threshold", 0.15)\n            min_conf = getattr(self, "min_confidence", 0.58)\n            bear_thresh = getattr(self, "bear_entry_threshold", 0.52)\n            sell_margin = getattr(self, "bear_min_margin", 0.15)'
    content_lf, n = re.subn(pattern, new_sub, content_lf)
    if n > 0:
        policy_path.write_text(content_lf, encoding="utf-8")
        print(f"Successfully patched {n} occurrence(s) via regex!")
    else:
        print("Failed to match regex in mythos_mt5_policy.py!")
