# Canonical Lifecycle Settlement V1

## V185: bảo vệ chất lượng batch Full-RL trong lúc train

- Hàng đợi AdamW vẫn là writer duy nhất, nhưng batch không còn được phép hình
  thành chỉ từ nhãn `entry_preference`. Batch 8 mẫu phải có ít nhất 4 lifecycle
  MT5 đã settle; nếu có đủ 8 lifecycle thì ưu tiên batch thuần lifecycle.
- Telemetry batch ghi `lifecycle_item_count`, `preference_item_count` và quota
  pending để hậu kiểm đúng thành phần dữ liệu đã thực sự đi qua optimizer.
- `realized_r` vẫn là target thô của critic. Actor entry nhận toàn bộ credit
  cho TP/SL chuẩn, nhận 35% đối với `POLICY_CLOSE_*`, và không nhận actor credit
  từ dấu kết thúc tester cưỡng bức. Mọi owner lifecycle có trace đều có thể
  ghi thêm `POST_EXIT_HORIZON` để quan sát hold-to-horizon; row này là bằng
  chứng counterfactual/audit, không phải actor credit thứ hai. Exit-manager
  credit vẫn chỉ áp dụng cho `POLICY_CLOSE_*` có `exit_trace_id`.
  Ledger/PnL gốc không bị sửa.
- `timing_network_lifecycle_scale: 0` nay khóa tuyệt đối PnL writer của
  TimingNet; không còn fallback ngầm sang `_mt5_online_update_scale`.
- Candidate profile dùng AdamW LR `1e-4` thay cho `5e-4` sau khi telemetry v182
  cho thấy nhiều batch bị gradient clip trên 90%. Checkpoint/seed gốc không bị
  ghi đè.
- Lifecycle thiếu trajectory tiếp tục bị đánh dấu
  `trajectory_insufficient`/`not_available`; không tạo target SL/RR suy diễn.

## Mục đích

Đường credit của context-expert chỉ có một điểm vào:
`ChuyenGiaPolicy.apply_execution_lifecycle_credit()`.

Mỗi lifecycle MT5 đóng phải mang cùng một `trace_id`, ticket vị thế, action
đã phát thực tế, net-R, lý do đóng và thời lượng. Facade context-owner giữ
trace entry bất biến, truyền nó cùng `override_action` tới policy owner.

## Thứ tự settlement

1. Xác thực trace entry và tạo receipt key `trace + ticket + close + R`.
2. Từ chối receipt trùng trước khi bất kỳ learner nào thay đổi state.
3. Credit network/encoder/parameter head đúng một lần.
4. Ghi expectancy vào đúng `action_index` đã thực thi, rồi xóa query cache.
5. Credit reasoning, causal ledger, memory và micro-growth theo cùng trace.
6. Credit ActionRecovery và ghi một `lifecycle_credit_ack` có kết quả từng component.
7. Chỉ sau ack mới phát lifecycle observability.

Không có action mask, side flip hoặc HOLD được tạo bởi settlement. Nếu
expectancy âm, nó chỉ là evidence action-local ở quyết định sau.

## Candidate Direct-RNN full RL

Candidate bật `learning.direct_rnn_full_rl_enabled=true` thay đường credit
cục bộ bằng một batch RL chỉ nhận lifecycle đã settlement. Tại entry, mỗi
module lưu `DirectRnnForwardTrace` bất biến: input đã mask, recurrent state,
preactivation/candidate của base và từng deep block, normalization, dropout
scale, output hidden, output mask và gate. Vì vậy một lệnh đóng sau nhiều bar
không được forward lại bằng hidden/weight hiện tại rồi credit nhầm trạng thái.

Khi batch đạt quota lifecycle thật (ít nhất một nửa batch, hoặc batch thuần
lifecycle), loss gồm policy-gradient trên action đã phát, Huber value loss,
entry-preference/counterfactual có giới hạn và entropy bonus. Gradient đi qua
`w_out/b_out`, value head, toàn bộ `deep_w/deep_rec/deep_b`, rồi
`w_in/w_rec/b_hidden` và gate. Temporal credit dùng
`direct_rnn_rl_temporal_bptt_steps` trong miền 1..32 (candidate hiện tại là 8):
recurrent state trước cửa sổ cache là hằng số, còn base và chiều sâu sáu block
trong cửa sổ được backprop đầy đủ. Dropout và
normalization dùng chính cache forward; lateral edge, TimingNet, memory và
broker/legal overlay là stop-gradient có telemetry rõ ràng.

`ModuleState.apply_rnn_adamw_cosine()` là writer duy nhất cho các trọng số
Direct-RNN trong mode này. Mọi writer credit cũ bị từ chối, kể cả plasticity,
fitness, action-margin và encoder local. AdamW giữ moment bias-corrected,
global clip, decoupled weight decay và cosine LR theo optimizer step.

Nếu overlay ngoài Direct-RNN đổi action winner, mẫu đó chỉ cập nhật value head;
policy gradient bị bỏ để không dạy RNN bắt chước decision của memory/prior/mask.
Ngưỡng evidence dùng mean absolute advantage của batch, không dùng signed mean:
lệnh lãi và lỗ hợp lệ không làm triệt tiêu toàn bộ learning.

Candidate có thể bật `critic_asymmetric_v1`: actor advantage dùng cached
decision-time value prediction làm baseline lai với EMA, shield net win khỏi
negative actor credit và chuẩn hóa residual bằng thống kê causal. Huber value
loss luôn dùng raw net-R. Vì vậy critic cải thiện baseline mà không làm thay
đổi định nghĩa PnL/expectancy của MT5.

### Cadence Full-RL đồng nhất

Mọi owner dùng cùng batch Full-RL cấu hình chung. Không có override batch,
regularizer saturation hay capture-forward riêng theo context, nên số lifecycle
đủ một batch có cùng ý nghĩa đối với mọi chuyên gia. Hidden normalization và
dropout ở mọi tầng đã cấu hình vẫn thuộc forward/backprop chung; execution observatory chỉ
ghi trace action/lifecycle chuẩn và không làm thay đổi settlement, credit hay
checkpoint seed v13.

Telemetry bắt buộc: `direct_rnn_full_rl_queued` và
`direct_rnn_full_rl_batch` với batch size, mean/mean-absolute advantage,
policy/value/entropy loss, số sample policy-eligible, optimizer norm/clip/LR,
`depth_backprop=true`, `temporal_bptt_steps` thực tế và cờ lateral stop-gradient.

## Invariant kiểm tra

- Một receipt trùng trả `0` và lý do `duplicate_receipt`.
- Lệnh BUY đóng lỗ chỉ tạo sample expectancy `action_index=1`; không thể rơi
  sang SELL_STOP hoặc một action khác.
- `lifecycle_credit_ack` phải có trace, ticket, action, R và trạng thái từng
  component; component lỗi phải được ghi rõ, không dùng `except: pass`.
- `self_growing_action.csv`/metadata dừng không được coi là HOLD hay credit
  âm: bridge phải phục vụ receipt trước khi lifecycle mới có thể được settle.

## Phạm vi

Contract này áp dụng cho module CLI `python -m self_growing_trader.cli` và
`HeChuyenGiaMotContextMotPolicy`. `SelfGrowingPolicy` legacy là workflow khác,
không tham gia context-expert MT5 train path.
