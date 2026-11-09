import re
from pathlib import Path

path = Path("scripts/run_mythos_mt5_live.py")
content = path.read_text(encoding="utf-8")

# 1. Add visual param to run_mythos_mt5
content = re.sub(
    r'(def run_mythos_mt5\([^)]*symbol:\s*str\s*=\s*[\'"][^\'"]+[\'"])(,\s*\)\s*->\s*dict:)',
    r'\1,\n    visual: bool = False\2',
    content
)

# 2. Update visual and shutdown_terminal kwargs
content = content.replace('"visual": False,', '"visual": visual,')
content = content.replace('"shutdown_terminal": True,', '"shutdown_terminal": not visual,')

# 3. Add --visual to argparse
if '--visual' not in content:
    content = content.replace(
        'parser.add_argument("--symbol", default="BTCUSDm")',
        'parser.add_argument("--symbol", default="BTCUSDm")\n    parser.add_argument("--visual", action="store_true", help="Enable MT5 GUI Visual Mode")'
    )
    content = content.replace(
        'symbol=args.symbol,',
        'symbol=args.symbol,\n        visual=args.visual,'
    )

path.write_text(content, encoding="utf-8")
print("Patch applied successfully!")
