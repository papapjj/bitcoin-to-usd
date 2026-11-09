# V188 — Exploration thật cho Direct-RNN entry actor

## Mục tiêu

V188 thay đường chọn action xác định trong lúc **online learning** bằng một
behavior policy có xác suất nhưng bị giới hạn. Phạm vi duy nhất là bảy lệnh
`HOLD + MARKET_BUY + MARKET_SELL + BUY_LIMIT + SELL_LIMIT + BUY_STOP + SELL_STOP`.
Nhánh quản lý/thoát lệnh, frozen replay và OOS vẫn dùng quyết định xác định.

TensorRule trong candidate V191 ở quyền `residual`, nhưng exploration vẫn thuộc
Direct-RNN actor. Cold-start LCB gate giữ nguyên sampled action cùng provenance;
rule bank không biến action greedy chưa được xác nhận thành trade.

## Behavior policy

Mỗi decision entry có hai nhánh:

1. `1 - epsilon`: giữ action greedy hiện tại, bao gồm logic TRADE/HOLD đang có.
2. `epsilon`: chọn một action hợp lệ khác greedy. Xác suất giữa các alternative
   lấy từ target policy và có floor nhỏ để action hợp lệ không biến mất hoàn toàn.

Candidate hiện dùng:

- `epsilon = 0.10`;
- floor xác suất alternative `0.02`;
- importance ratio clip `2.0`;
- entropy weight `0.002`;
- seed xác định `20260803`.

Mask broker/protocol vẫn được áp dụng trước khi sample. Rows quản lý `7..11`
luôn có xác suất exploration bằng 0. Khi chỉ còn một action hợp lệ, policy giữ
action đó.

Chuỗi draw được tạo xác định từ `seed × expert owner × exploration step`. Hàm
`reset_runtime_state()` đưa step về 0, nên hai candidate clone cùng seed bắt đầu
bằng cùng chuỗi exploration mà không cần lưu RNG state trong artifact.

## Credit và telemetry

`StepTrace.action_details` lưu:

- greedy action và behavior action;
- target probabilities và behavior probabilities;
- behavior probability/log-probability của action đã phát;
- exploration active/sampled/reason/rate/step/draw;
- target/behavior importance weight đã clip.

Khi lifecycle settle, chỉ actor policy-gradient được nhân importance weight.
Value, auxiliary, reconstruction, TensorRule và entry-preference loss không bị
nhân vì chúng không tối ưu `log pi(action)` của behavior sample. AdamW Full-RL
vẫn là writer duy nhất.

Bridge áp thêm firebreak cuối: action Direct-RNN trong trace phải trùng chính
xác action thực sự được facade phát ra. Nếu facade đổi một entry đã sample thành
lệnh quản lý/hủy lệnh, `direct_rnn_rl_policy_eligible=false` với reason
`emitted_action_changed`; outcome đó không thể nhận actor-gradient như một mẫu
exploration entry.

## Isolation và config

Code mặc định `direct_rnn_rl_exploration_enabled=false` để checkpoint/config cũ
không đổi hành vi. Profile V188 cũ đã được loại bỏ khi V191 thay thế đường học.
Profile hiện hành
`configs/experiments/action_conditioned_abstraction_v191_candidate.yaml` bật
exploration 10% nhưng tắt `train_entry_probe_enabled`. Khi TensorRule chưa có
LCB, chỉ sampled behavior action được trade; greedy actor bị ép HOLD. Vì vậy
khám phá tạo dữ liệu có xác suất/importance rõ ràng mà không mở lại probe cũ.

`_learning_enabled=false` luôn trả về greedy action và ghi reason
`frozen_or_oos`. Expert management ghi `management_isolated` và không tăng
exploration step.

## Kiểm thử và trạng thái bằng chứng

Regression bắt buộc kiểm tra:

- không sample action quản lý hoặc action illegal;
- action sampled thực sự trở thành argmax trên output bridge 20 chiều;
- reset tái tạo đúng action/draw;
- frozen/OOS và management giữ deterministic;
- behavior probability hữu hạn và importance weight không vượt clip;
- actor credit bị chặn nếu facade/bridge đổi action sau local expert;
- V186 Python debug tạo candidate clone mà source config không đổi.

Python regression/debug chỉ chứng minh wiring. Nó chưa chứng minh exploration
làm PnL, policy expectancy LCB hoặc độ bền edge tốt hơn. Kết luận đó vẫn cần
paired MT5 real-tick OOS cùng seed/window và tách riêng policy-origin lifecycle.

Smoke MT5 model 1 trên clone ngày 2025-01-01 phục vụ 98 decision trước
`max_wall_sec`: 44 decision đủ điều kiện exploration, 4 action thực sự được
sample, không có management exploration hoặc legacy probe, và importance weight
tối đa đúng clip `2.0`. Hai decision bị facade đổi từ local HOLD sang
`CANCEL_ORDERS` đều ghi `emitted_action_changed` và actor eligibility bằng false.
Source config và seed hash không đổi. Đây là bằng chứng runtime/firebreak, không
phải bằng chứng edge hay promotion.
