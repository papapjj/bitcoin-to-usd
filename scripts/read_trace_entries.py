import json

with open('outputs/mt5_optimization_loop/reasoning_trace.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('action') != 'HOLD':
            print(f"Tick {obj.get('tick')}: Action={obj.get('action')} Price={obj.get('price'):.2f} ATR={obj.get('atr'):.2f} Bull={obj.get('bull_score'):.4f} Bear={obj.get('bear_score'):.4f} Margin={obj.get('margin'):+.4f} Verdict={obj.get('verdict')}")
