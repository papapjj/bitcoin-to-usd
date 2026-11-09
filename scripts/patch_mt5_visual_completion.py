from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
runner_path = ROOT / "src" / "self_growing_trader" / "mt5_runner.py"
content = runner_path.read_text(encoding="utf-8")
content_lf = content.replace("\r\n", "\n")

# Look for the visual report completion guard
pattern = r'if \(\s+not self\.config\.visual\s+and located_report is not None\s+and time\.monotonic\(\) - last_request_seen > 2\.0\s+\):'

replacement = """            if (
                located_report is not None
                and time.monotonic() - last_request_seen > 2.0
            ):"""

if re.search(pattern, content_lf):
    content_lf = re.sub(pattern, replacement, content_lf, count=1)
    runner_path.write_text(content_lf, encoding="utf-8")
    print("Successfully patched MT5 runner visual completion guard!")
else:
    print("Pattern not matched directly, searching closely...")
    # Alternative direct search
    target_str = "not self.config.visual\n\n\n\n                and located_report is not None"
    if "not self.config.visual" in content_lf:
        lines = content_lf.split("\n")
        patched = False
        for idx, line in enumerate(lines):
            if "not self.config.visual" in line and "located_report" in "\n".join(lines[idx:idx+10]):
                # Replace that line with True or remove it
                lines[idx] = line.replace("not self.config.visual", "True  # Visual completion enabled")
                patched = True
                break
        if patched:
            runner_path.write_text("\n".join(lines), encoding="utf-8")
            print("Successfully patched line via line search!")
        else:
            print("Could not find exact block in lines.")
