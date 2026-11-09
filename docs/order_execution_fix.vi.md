# Sửa deadlock “train không vào lệnh”

## Nguyên nhân đã sửa trong Python bridge

Khi account đang trống, runner cũ lấy argmax trực tiếp trên 11 action. Một policy chưa train có thể chọn `CANCEL_ORDERS`, `CLOSE_WORST`, `CLOSE_BEST` hoặc `CLOSE_ALL`; các action này không hợp lệ vì chưa có pending order hay position. Do đó MT5 không thể tạo order.

Runner cũ cũng để `outputs[14]` quyết định dấu offset cho pending order mà không xét loại lệnh. Ví dụ `BUY_LIMIT` với offset dương hoặc `SELL_STOP` với offset dương có thể tạo price sai phía thị trường và bị MT5 reject.

Sau hotfix memory, action bị reject không được TD credit. Kết quả là deadlock:

```text
no position -> invalid/ignored action -> no execution evidence -> no credit -> policy unchanged -> no position
```

## Bản vá

1. **Action mask theo execution state**
   - Không có position/pending: chỉ `HOLD`, `MARKET_*`, `*_LIMIT`, `*_STOP`.
   - Có pending mới cho phép `CANCEL_ORDERS`.
   - Có position mới cho phép `CLOSE_*`.

2. **Chuẩn hóa hướng pending offset**
   - `BUY_LIMIT`, `SELL_STOP`: offset âm, phía dưới market.
   - `SELL_LIMIT`, `BUY_STOP`: offset dương, phía trên market.

3. **Pending recovery cancel trong Strategy Tester**
   - Bridge không còn tự gửi `MARKET_BUY`/`MARKET_SELL` để tạo feedback.
   - Khi MT5 báo cùng một `orders_pipe` qua đủ số decision, bridge có thể gửi `CANCEL_ORDERS` để giải phóng pending order treo.
   - Mục tiêu là trả account về trạng thái có thể entry lại theo policy owner, không thay thế quyết định entry của policy.

4. **Log chứng minh action thực gửi EA**

Ví dụ:

```text
[Bridge] served=124 seq=124 action=7(CANCEL_ORDERS) policy_action=0 masked=0 bootstrap=1 positions=0 day_trades=12
```

- `policy_action=0`: policy giữ `HOLD`.
- `action=7`: bridge phát recovery cancel vì pending order treo.
- `bootstrap=1`: recovery action của bridge, không phải market-entry bootstrap.
- `day_trades`: chỉ là telemetry; daily min/max không còn khóa entry.

5. **Tách namespace hypothesis và MT5 action**
   - `StepTrace.selected_idx` chỉ là index giả thuyết/candidate M1, không phải mã action MT5.
   - `StepTrace.selected_action` và `action_details.final_action_output` phải là cùng vector action cuối sau context mask, risk/action head và learner recovery.
   - Bridge ưu tiên `final_action_output` khi chọn `policy_action/legal_action/emitted_action`.
   - Hồ sơ quyết định chỉ đọc `policy_action`, `legal_action`, `emitted_action` từ `action_details`; nếu thiếu thì ghi `-1`, không fallback sang `selected_idx`.
   - `model_hypothesis_index` được log riêng để audit “não chọn giả thuyết nào” mà không lẫn với “tay gửi lệnh nào”.

## Khi nào phải kiểm tra EA thay vì Python

Nếu log hiển thị `action=1(MARKET_BUY)` hoặc `action=2(MARKET_SELL)` nhưng MT5 vẫn có `Total Trades: 0`, Python đã gửi một market action hợp lệ. Khi đó lỗi nằm ở EA/terminal broker layer, thường là một trong các retcode:

- `10027`: AutoTrading bị tắt;
- `10016`: invalid stops;
- `10014`: invalid volume;
- `10018`: market closed;
- `10030`: unsupported filling mode;
- `10019`: insufficient funds.

EA cần ghi `OrderSend()` result/retcode vào Experts log, hoặc nâng request schema lên v3 và gửi `execution_pipe` với `ack_sequence`, `accepted`, `retcode`, `ticket`.
