from unittest.mock import patch
from tests.test_codex_teacher import *
import torch

torch.manual_seed(705)
p = ChuyenGiaPolicy.seed(
    input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
    hidden_size=12, hidden_layers=6, seed=705,
)
p.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
p._learning_enabled = False
p.install_five_rlhf_reference()
traces = p.act_five_agents(np.linspace(-0.2, 0.4, INPUT_COUNT, dtype=np.float32))
owner = next(
    trace for trace in traces
    if 1 <= int(trace.action_details.get("five_action_index", trace.selected_idx)) <= 6
)
lifecycle = SimpleNamespace(
    final_close_confirmed=True,
    metadata={"position_owner_trace_id": owner.trace_id},
    position_ticket="repeated-negative", realized_r=-0.9, mfe_r=0.1, mae_r=-1.0,
    commission=0.0, swap=0.0, close_reason="SL", side=1, entry_count=1,
    entry_price=100.0, initial_sl_price=99.0, initial_tp_price=102.0,
    initial_risk=1.0, initial_atr=0.4, pending_fill_bars=0,
)
packet = build_codex_teaching_packet(lifecycle, traces)

with patch('numpy.clip', side_effect=lambda a, a_min, a_max: a):
    for i in range(8):
        raw = receipt_for(packet)
        raw.update(teacher_reward=-0.8, confidence=1.0)
        receipt = validate_codex_teacher_receipt(raw, packet, "test-model")
        scorer = CodexTeacherRewardModel(packet, receipt, max_probability_shift=0.05)
        res = p.optimize_five_rlhf_trace_packet(
            traces, scorer, source_checkpoint_sha256="source",
            config=MT5BacktestConfig(codex_teacher_learning_rate=0.005, codex_context_similarity_threshold=0.995, codex_context_support_floor=0.5),
        )
        print(f"iter {i}: update={res.get('rlhf_ppo_update')} reason={res.get('rlhf_rejection_reason')}")
        print(f"   bias delta={res.get('rlhf_action_bias_target_update_delta')} eff_lr={res.get('rlhf_optimizer_learning_rate_for_update')}")
        print(f"   param_ch={res.get('rlhf_parameter_changed')} shift={res.get('rlhf_packet_max_probability_shift')}")
