# Checklist bám sát tài liệu gốc

Nguồn đối chiếu: `C:\Users\Admin\Downloads\So_do_mang_no_ron_tu_phat_trien_bo_sung_tu_phat_trien_module.docx`.

## 1. Kiến trúc tổng quát

- Gate In nhận `x_t`, `goal_t`, `context_t`.
- Tập module `M_k` có gateway, trạng thái nội bộ, cổng ra và utility riêng.
- `M1` là module tư duy gồm `state_encoder`, `working_memory`, `hypothesis_bank`, `predictor`, `critic`, `selector`.
- Gate Out tổng hợp output theo trọng số module.
- Feedback/recurrent giữ trạng thái `m_k(t-1)` qua các bước trong MT5 tester.

## 2. Động học neuron/module

- Mỗi module có `W_in`, `W_rec`, `W_out`, bias, gate và hidden state.
- Activation dùng input hiện tại và state trước đó.
- Module có metadata lifecycle: dormant, active, consolidated, pruning.
- MT5 bridge phải gửi đủ input để Python model chạy cùng kiến trúc.

## 3. Plasticity

- Có gradient-like score từ MT5 fitness.
- Có Hebbian/Oja-like local update hoặc ít nhất thống kê đồng hoạt hóa.
- Có anti-Hebbian/regularization để giảm trùng lặp.
- Có homeostasis để giảm saturation/dead module.
- Có short-term/recurrent state trong runtime.

## 4. Growth và pruning

- Growth không chỉ random; phải dựa trên score gồm error, novelty, conflict, utility và complexity.
- Hỗ trợ expand/module clone khi thiếu capacity.
- Hỗ trợ prune hoặc decay gate khi utility thấp.
- Có consolidation làm giảm plasticity cho module ổn định.
- Có rollback/promote dựa trên MT5 validation.
- Có core optional `SpatialWorldPolicy` biểu diễn module như quốc gia trong mixed geometry world; energy/knowledge/trust tác động gate/action và growth signal.

## 5. Vòng phát triển

- Train loop chạy nhiều round/trial bằng MT5 Strategy Tester.
- Sau mỗi evaluation thu thập fitness, report và bridge telemetry.
- Growth controller chẩn đoán thiếu capacity, dư thừa, hoặc cần module mới.
- Chỉ publish policy tốt nhất sau khi qua promotion gate.
- `compare-cores` có thể chạy current/world song song như hai thế giới, rồi trao đổi positive replay và validate bằng MT5 trước khi giữ tinh hoa.

## 6. Bridge MT5

- EA gửi input/state qua `self_growing_request.csv`.
- Python chạy model và gửi action/output qua `self_growing_action.csv`.
- EA chỉ execute action trong Strategy Tester và ghi bridge telemetry gồm input, output, account/trade state.
- Bridge hiện dùng schema v2 có `event_id` và `trace_id`; EA giữ `trace_id` từ Python action sang telemetry.
- Python đọc bridge để kiểm chứng EA có chạy đủ bar và có dữ liệu cho growth controller.

## Trạng thái hiện tại

- Đã có MT5-backed training, bridge thuần EA-Python, module gated chạy trong Python, clone specialist, và world-core optional bật bằng `--policy-core world`.
- Đã bổ sung trong đợt hiện tại: M1 component metadata, recurrent `W_rec`, action bridge `request/action`, EA không còn model runtime, vòng MT5 `act -> observe_transition -> develop`, `StepTrace.trace_id`, bridge schema v2 và reward online có equity/spread/drawdown cost.
- Đang thiếu hoặc mới một phần: trade-level realized PnL/commission/slippage đầy đủ, credit assignment theo open trade hoặc N-step, sub-module M1 tách thật, Hebbian/anti-Hebbian/STDP đầy đủ, prune/merge/split theo ablation, shadow/OOD validation đầy đủ.
