# V193 — Reward V2 expectancy-first

## Mục tiêu

Reward V2 tối ưu theo thứ tự:

1. net marked-R sau spread/slippage;
2. giảm drawdown, vượt risk và loss tail;
3. chỉ cải thiện win rate khi realized expectancy đã dương.

Không có reward cho việc mở lệnh, broker nhận lệnh, số lượng lệnh, indicator đồng
thuận hoặc HOLD bị ép. Vì vậy model không thể lấy điểm chỉ bằng spam trade.

`HOLD` chỉ tham gia reward kinh tế khi tác nhân sở hữu vị thế trước quyết định.
Khi đó phần delta marked-R dương của vị thế được ghi riêng là
`position_improvement_reward_r` với hệ số cấu hình mặc định `1.0`; delta âm vẫn
phạt trực tiếp. Đây là cách tách telemetry của cùng một reward kinh tế, không phải
bonus cố định và không cộng lần hai vào tổng reward.
Nếu tác nhân không sở hữu vị thế, mẫu `HOLD` bị loại hoàn toàn khỏi reward, actor,
critic, entropy và bộ chuẩn hóa reward; đây là "không có mẫu", không phải reward 0.

## Công thức

Mỗi transition dùng delta của các đại lượng tích lũy, nên một trade chỉ được tính
một lần:

```text
r = delta_marked_R
  + expectancy_gate * wr_weight * (delta_wins - wr_target * delta_closed)
  - transaction_cost_R
  - drawdown_weight * delta_drawdown
  - risk_weight * max(risk_ratio - 1, 0)^2
  - conflict_weight * conflict_cost
  - shortfall_weight * sum(max(expectancy_floor_R - trade_R, 0))
  - tail_weight * sum(min(max(-trade_R - tail_threshold_R, 0), tail_cap_R)^2)
  - 1.0R * I(policy_action = MANAGE_POSITION va positions_before = 0)
```

`expectancy_gate` chỉ bằng 1 khi agent có ít nhất một trade đóng và
`sum(realized_R) / closed_trades >= expectancy_floor_R`. Vì vậy win rate cao nhưng
expectancy âm không thể nhận WR bonus. Net-R vẫn có hệ số 1 và là mục tiêu chính.

## Cấu hình V193

```yaml
five_reward_expectancy_wr_v2_enabled: true
five_reward_wr_target: 0.55
five_reward_wr_balance_weight: 0.05
five_reward_expectancy_floor_r: 0.05
five_reward_expectancy_shortfall_weight: 0.10
five_reward_tail_loss_threshold_r: 1.0
five_reward_tail_loss_weight: 0.10
five_reward_tail_loss_cap_r: 3.0
five_position_improvement_reward_weight: 1.0
five_invalid_position_action_no_position_penalty_r: 1.0
five_invalid_cancel_no_pending_penalty_r: 2.0
five_unfilled_pending_timeout_penalty_r: 0.25
```

Hủy sớm một lệnh chờ có thật không tạo reward hành vi. Chỉ lệnh giới hạn/dừng
được MT5 xác nhận `UNFILLED_TIMEOUT` sau 5 bar mới phạt `-0.25R` vào đúng trace
đã đặt lệnh. Hành động hủy khi tài khoản không hề có lệnh chờ vẫn là vi phạm
hợp đồng hành động riêng và giữ mức phạt `-2.0R`.

Khoản `1.0R` là hình phạt hợp đồng hành động riêng cho tác nhân vi phạm và nằm
ngoài phép trộn phần thưởng đồng đội. Vì vậy bốn tác nhân còn lại không bị phạt
oan, còn tác nhân đã chọn `CLOSE_WORST`, `CLOSE_BEST`, `CLOSE_ALL` hoặc bất kỳ
mode `MANAGE_POSITION` nào khi snapshot trước quyết định không có vị thế nhận
đủ `-1.0R`. Các mode quản lý gồm đổi SL/TP, trailing, đóng một phần và đóng toàn
phần. Five MT5 không che action trước khi lấy mẫu hoặc đổi action
thành `HOLD` trước EA. EA xử lý đúng action gốc; giới hạn tài khoản/broker tạo
biên nhận từ chối thật để actor học đúng nguyên nhân.

`CANCEL_ORDERS` khi snapshot trước quyết định không có lệnh chờ nhận `-2.0R`
trên đúng tác nhân phát lệnh. Khoản phạt này cũng nằm ngoài phép trộn phần thưởng
đồng đội, nên hành động hủy rỗng không thể hưởng lợi từ PnL do tác nhân khác tạo
ra. Hành động vẫn được gửi nguyên trạng tới EA và biên nhận `NO_EFFECT` vẫn được
giữ để kiểm toán; chỉ nhánh học nhận thêm chi phí vi phạm hợp đồng.

Profile cũ mặc định tắt V2 để checkpoint/config cũ không đổi objective ngoài ý muốn.
Tiến trình Python đã chạy trước khi sửa phải được kết thúc và chạy fresh candidate;
hot-edit source không thay đổi object reward đang nằm trong RAM.

## Contract dữ liệu MT5

Train qua Strategy Tester dùng cùng công thức Reward V2, với nguồn sự thật là
`MT5PositionLifecycle` đã đóng và bất biến:

- `trace_id` ánh xạ trade về đúng một trong năm agent đã tạo lệnh;
- comment lệnh mới có dạng `SG5:<sequence>:<agent_id>`; năm intent trong cùng
  packet không còn dùng chung một khóa `sequence` mơ hồ;
- resolver ưu tiên ticket order/deal/position, sau đó mới dùng đúng cặp
  `(sequence, agent_id)`. Comment `SG4:<sequence>` cũ chỉ được chấp nhận khi
  sequence đó khớp duy nhất một intent; nếu có nhiều intent thì fail closed;
- `realized_r` là net PnL sau commission/swap chia cho initial risk;
- `closed`, `wins`, tổng realized-R, expectancy shortfall và squared tail excess
  được cộng dồn riêng theo owner;
- lifecycle identity được dedupe, nên việc đọc lại execution ledger không cộng
  reward lần hai;
- transaction cost vẫn được tách khỏi gross marked-R rồi trừ đúng một lần;
- ranh giới ngày giữ terminal cumulative state làm baseline của ngày kế tiếp,
  tránh phát lại PnL/WR/shortfall/tail đã học.
- Python lưu cả action gốc theo `trace_id` và dừng fail-fast nếu một lifecycle
  đóng bị gán cho HOLD/CANCEL/CLOSE/MANAGE. Reward sai owner không được phép đi
  vào PPO.

Các threshold `expectancy_floor`, `tail_loss_threshold` và `tail_loss_cap` được
truyền từ cùng `learning` config vào MT5 evaluator. Chúng không được hard-code
khác với môi trường dataset.

Observation Pattern Tree vẫn chạy forward và nhận gradient PPO từ reward kinh tế
trên MT5. Reward đường giá V2 và auxiliary target dùng future được phát trễ:
dataset chờ hết sample, MT5 online train chờ đủ 48 nến M1 đóng trong path ledger.
Chúng bị cấm trong time mode, frozen/OOS và live inference; future không đi vào
observation hoặc action request nên không có look-ahead tại thời điểm quyết định.

## Telemetry bắt buộc

- `reward_profit_delta_mean`
- `reward_position_improvement_mean`, `reward_position_improvement_count`
- `reward_closed_trades`, `reward_wins`
- `reward_expectancy_gate_fraction`
- `reward_wr_balance_mean`
- `reward_expectancy_shortfall_mean`
- `reward_tail_loss_mean`
- `reward_invalid_position_action_no_position_count`
- `reward_invalid_position_action_no_position_penalty_mean`
- `reward_invalid_position_action_no_position_penalty_max`
- `reward_invalid_cancel_no_pending_count`
- `reward_invalid_cancel_no_pending_penalty_mean`
- `reward_invalid_cancel_no_pending_penalty_max`
- `actor_eligible_fraction`, `primary_action_entropy`
- `reward_sample_eligible_fraction`, `reward_flat_hold_excluded_count`
- `pending_timeout_observed`, `pending_timeout_accepted`
- `pending_timeout_updated`, `pending_timeout_cache_miss`

Nhánh kinh tế này được bổ sung bởi reward đường giá delayed-causal mô tả tại
`docs/FIVE_PATTERN_CREDIT_V3.vi.md`. Reward đường giá chỉ học decision anchor
sau khi future kết thúc và không thay thế marked net-R.

Reward vòng đời điểm vào V4 tại `docs/FIVE_ENTRY_LIFECYCLE_REWARD_V4.vi.md` bổ sung
thưởng SL/TP hợp lệ có MFE dương, trailing khóa lợi nhuận, entry đạt MFE cao với
MAE thấp và thoát lệnh giữ được MFE; đồng thời vẫn phạt hình học quá gần, trailing
yếu, MFE bằng 0 và SL ban đầu bị chạm. Nó chỉ quyết toán trên lifecycle cuối cùng,
đúng entry owner và nằm ngoài team-spirit.

## Cổng đánh giá

Không kết luận từ train reward. So sánh fresh V1 và fresh V2 bằng cùng seed, dataset
hash và episode order; sau đó đóng băng checkpoint và chạy cùng walk-forward OOS.
Chỉ giữ V2 nếu OOS net-R và expectancy dương, LCB95 expectancy trên 0, WR đạt mục
tiêu mà profit factor không giảm, drawdown không xấu hơn baseline và mọi execution
receipt/owner attribution đều khớp.
