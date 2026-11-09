with open('scripts/run_mt5_observation_optimization_loop.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '    logger.info(f"=== Starting Iteration: {iter_name} ({from_date} -> {to_date}) ===")\n    t_start = time.time()'
replacement = '''    logger.info(f"=== Starting Iteration: {iter_name} ({from_date} -> {to_date}) ===")
    t_start = time.time()
    trace_p = ROOT / "outputs" / "mt5_optimization_loop" / "reasoning_trace.jsonl"
    if trace_p.exists():
        try:
            trace_p.unlink()
        except Exception:
            pass'''

if target in text:
    new_text = text.replace(target, replacement, 1)
    with open('scripts/run_mt5_observation_optimization_loop.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("SUCCESS: Added trace reset on iteration start.")
else:
    print("ERROR: Target not found.")
