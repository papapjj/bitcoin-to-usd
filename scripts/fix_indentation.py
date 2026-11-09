from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runner_path = ROOT / "src" / "self_growing_trader" / "mt5_runner.py"
content = runner_path.read_text(encoding="utf-8")
content_lf = content.replace("\r\n", "\n")

old_str = """                        if (
                located_report is not None
                and time.monotonic() - last_request_seen > 2.0
            ):"""

new_str = """            if (
                located_report is not None
                and time.monotonic() - last_request_seen > 2.0
            ):"""

if old_str in content_lf:
    content_lf = content_lf.replace(old_str, new_str)
    runner_path.write_text(content_lf, encoding="utf-8")
    print("Fixed indentation successfully!")
else:
    print("old_str not matched directly, checking lines...")
    lines = content_lf.split("\n")
    for i, line in enumerate(lines):
        if "if (" in line and "located_report" in lines[i+1]:
            lines[i] = "            if ("
            lines[i+1] = "                located_report is not None"
            lines[i+2] = "                and time.monotonic() - last_request_seen > 2.0"
            lines[i+3] = "            ):"
            runner_path.write_text("\n".join(lines), encoding="utf-8")
            print("Fixed lines via index search!")
            break
