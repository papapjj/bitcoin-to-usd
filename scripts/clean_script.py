import unicodedata

with open('scripts/launch_7day_visual_master_champion.py', 'r', encoding='utf-8') as f:
    text = f.read()

clean_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
header = """import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

"""
clean_text = header + clean_text

with open('scripts/launch_7day_visual_master_champion.py', 'w', encoding='utf-8') as f:
    f.write(clean_text)

print("SUCCESS: cleaned non-ascii characters and added utf-8 reconfigure!")
