# Entry Action Separation V185

## Mục tiêu

V185 không cố ép mọi lệnh thắng/thua thành hai cụm giả tạo. Nó tạo nhãn
action-conditioned: tại cùng một snapshot bất biến, so sánh `HOLD` và sáu
entry action trên cùng path M1 tương lai, nhưng không đưa path đó vào inference.

## Contract dữ liệu

EA/CSV vẫn giữ nguyên 137 feature và schema entry-preference hiện tại. Python
replay mỗi action thành các thành phần:

- `terminal_r`: R tại SL, TP hoặc mark cuối horizon;
- `mfe_r`, `mae_r`: excursion thuận lợi/bất lợi sau khi fill;
- `fill_bar`, `bars_to_positive`: độ trễ fill và xác nhận hướng;
- `direction_r`: follow-through của hướng entry;
- `timing_r`: phạt MAE, xác nhận muộn và pending fill muộn;
- `geometry_r`: phần terminal R chưa được direction giải thích;
- `execution_cost_r`: spread tại decision quy đổi theo initial risk;
- `quality_r`: tổ hợp bounded dùng bởi `path_quality_v1`.

`quality_r` tuyệt đối không chứa account PnL, trailing, policy close hay
management sau entry. `HOLD` luôn là 0R. Pending không fill cũng tương đương
HOLD, tránh tạo lợi nhuận giả.

## Học và batch

Profile candidate bật `direct_rnn_rl_entry_utility_mode=path_quality_v1`.
Artifact/config cũ thiếu field giữ `terminal_r_v1`. Soft target chỉ tác động
logit `[HOLD, BUY/SELL market/limit/stop]` trong chính Full-RL AdamW batch.
Không có head, mask, veto hoặc writer inference mới.

Queue vẫn bắt buộc ít nhất một nửa batch là lifecycle MT5. Trong từng nửa,
mẫu được lấy round-robin:

- lifecycle: `context x executed_action x win/loss/flat`;
- preference: `scope x selected_action x hard_negative`.

Hard-negative là entry thực thi có utility thấp hơn HOLD trong khi tồn tại một
alternative tốt hơn. Đây chỉ là nhãn/telemetry; không tự chặn giao dịch.

## An toàn nhân quả và promotion

- Forward cache luôn là tensor đúng tại decision.
- Path tương lai chỉ xuất hiện sau horizon và chỉ trong train loss.
- Full-RL AdamW tiếp tục là writer duy nhất.
- Thay đổi này mới chứng minh routing/code bằng unit test; chưa phải edge.
- Candidate chỉ được promote sau paired frozen/online nhiều cửa sổ, real ticks,
  reconciliation đầy đủ, expectancy LCB dương và drawdown gate đạt yêu cầu.

## Kiểm thử

`tests/test_entry_preference.py` kiểm tra bảo toàn terminal-R legacy, decomposition,
HOLD ordering, mode fail-closed, delayed queue và sole-writer. Các test Full-RL
kiểm tra lifecycle quota, actor credit, frozen immutability và TimingNet isolation.
