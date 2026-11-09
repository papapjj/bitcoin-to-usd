p = 'research_reasoning_world_class/src/mythos_mt5_policy.py'
with open(p, 'r', encoding='utf-8') as f:
    text = f.read()

target = '        # Attach verdict to traces for diagnostics and assertions'
replacement = '''        # Structured Metacognitive Reasoning Telemetry Logger
        log_path = getattr(self, "reasoning_log_path", "outputs/mt5_optimization_loop/reasoning_trace.jsonl")
        if log_path:
            try:
                import json
                from pathlib import Path
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                scen_dict = {}
                meta_out = getattr(self.pipeline.cot_engine, "_last_output_fallback", None)
                if meta_out is not None and hasattr(meta_out, "scenario_probs"):
                    sp = meta_out.scenario_probs[0].detach().cpu().numpy()
                    scen_dict = {"expansion": float(sp[0]), "reversal": float(sp[1]), "shakeout": float(sp[2]), "chop": float(sp[3])}
                log_row = {
                    "tick": self.seq_counter,
                    "price": round(float(self.current_price), 2),
                    "atr": round(float(self.current_atr), 2),
                    "action": entry_mode,
                    "bull_score": round(float(b_s), 4) if 'b_s' in locals() else 0.0,
                    "bear_score": round(float(r_s), 4) if 'r_s' in locals() else 0.0,
                    "prm_score": round(float(prm), 4),
                    "margin": round(float(margin), 4),
                    "scenarios": scen_dict,
                    "was_backtracked": bool(getattr(self.pipeline.cot_engine.cot_head, "last_was_backtracked", False)),
                    "active_positions": int(active_positions) if 'active_positions' in locals() else 0,
                    "verdict": verdict_str
                }
                with open(log_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(log_row) + "\\n")
            except Exception:
                pass

        # Attach verdict to traces for diagnostics and assertions'''

if target in text:
    text = text.replace(target, replacement, 1)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully added Reasoning Telemetry Logger to mythos_mt5_policy.py!")
else:
    print("Target not found in mythos_mt5_policy.py")
