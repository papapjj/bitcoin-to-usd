# Reward cạnh tranh năm tác nhân v1

## Mục tiêu

Năm tác nhân dùng chung trọng số nhưng có `agent_id`, memory và lịch sử hành động riêng. Reward v1 buộc các bản sao cạnh tranh bằng hiệu quả kinh tế thật của chiến lược do chính chúng tạo ra, đồng thời vẫn tối ưu lợi nhuận và drawdown của cả tài khoản.

Đây là một giải đấu nội bộ trong cùng policy. Nó không giả định năm tài khoản, năm checkpoint độc lập hoặc kết quả phản thực tế chưa được MT5 thực thi.

## Nguồn dữ liệu chuẩn

- `trace_id` được gắn cố định với `agent_id` ngay khi packet năm action được tạo; một trace không thể đổi chủ.
- Lợi nhuận đã chốt, commission, swap và initial risk lấy từ lifecycle cuối cùng trong execution ledger v14.
- Lợi nhuận đang mở lấy từ position snapshot và chỉ cộng cho agent sở hữu entry trace.
- Receipt bị resolver chặn tạo conflict cost riêng; không bị coi là HOLD và không được gán cho agent khác.
- Chi phí giao dịch được tích lũy rồi tiêu thụ đúng một lần ở transition kế tiếp.

Không dùng `selected_idx`, số lần mở lệnh hay việc broker chấp nhận lệnh làm đại diện cho lợi nhuận.

## Công thức mỗi bước

Với agent `i`, utility kinh tế là:

`economic_i = delta_marked_R_i - cost_R_i - 0.30 * delta_drawdown - 0.10 * max(risk_i - 1, 0)^2 - 0.05 * conflict_i`

Phần thi đấu chỉ xếp hạng utility mới do agent tạo ra:

`basis_i = delta_marked_R_i - cost_R_i - conflict_penalty_i`

`competitive_i = 0.25 * (basis_i - mean(basis_0..4))`

Tổng `competitive_i` của năm agent bằng 0. Do đó cạnh tranh chỉ phân phối credit giữa các chiến lược; nó không thể tạo reward danh mục giả. Risk và drawdown vẫn bị phạt trực tiếp nhưng không được dùng để thưởng lặp cho bốn agent đứng ngoài ở các tick không đổi.

Reward thô là `economic_i + competitive_i`. Team-spirit tăng tuyến tính từ 0.30 lên 1.00 chỉ trộn **phần kinh tế hợp tác** với trung bình đội; bonus cạnh tranh được cộng sau phép trộn nên không biến mất khi team-spirit đạt 1.00. Phần shaping không-terminal giảm theo `0.6^(elapsed_seconds/600)`; terminal không bị time-decay.

## Kết thúc episode

Biên ngày giao dịch tạo terminal chung:

`team_terminal = team_equity_R - max_drawdown`

Mỗi agent còn nhận rank terminal có chặn biên:

`rank_i = 0.50 * tanh(marked_R_i - mean(marked_R_0..4))`

Sau terminal, baseline kinh tế và năm memory được reset trước episode mới. Transition cuối không có execution receipt kế tiếp không được tự suy diễn thành kết quả; lifecycle chỉ được học khi ledger xác nhận hoặc đường terminal mark-to-market hiện hữu cung cấp bằng chứng.

## Lịch học và telemetry

`five_training_schedule_steps` đếm packet năm tác nhân, không đếm từng dòng agent. Nó điều khiển learning rate, entropy, GAE horizon và team-spirit. Profile V193 mặc định dùng 10.000 packet.

Sự kiện `five_competitive_reward` báo reward trung bình, độ lớn bonus cạnh tranh, strategy score tốt nhất/xấu nhất/spread và terminal rank. Các số này là telemetry huấn luyện, không phải bằng chứng OOS.

## Giới hạn xác nhận

Code và test chỉ chứng minh attribution, công thức và đường cập nhật. Chưa được gọi là chiến lược tối ưu lợi nhuận cho tới khi các checkpoint được đấu trên cùng cửa sổ frozen real-tick/OOS, cùng spread/commission, rồi so sánh Net Profit, PF, DD, số trade và policy-origin expectancy. Không quảng bá champion chỉ từ reward train.
