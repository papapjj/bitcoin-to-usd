# Quản lý nhiều vị thế và pending order trong MT5 bridge

## Hành vi hiện tại

Bridge không còn tự tạo `MARKET_BUY` hoặc `MARKET_SELL` để phá trạng thái không giao dịch. Chỉ policy owner của context hiện tại được quyết định entry. V193 đặt `risk.max_positions: 0`, vì vậy không còn mask/quota theo tổng số vị thế và lệnh chờ:

1. Không có position/pending: cho phép `HOLD`, market entry và pending entry.
2. Có pending order: vẫn cho phép entry mới; `CANCEL_ORDERS` chỉ là một hành động riêng của policy.
3. Có position: cho phép quản lý/đóng vị thế khi position đủ tuổi hoặc rơi vào trạng thái emergency.
4. Lệnh entry tiếp tục được gửi cho tới khi broker từ chối vì giá/khối lượng, chế độ tài khoản hoặc thiếu ký quỹ.

Năm position slot trong vector 137 chỉ là cửa sổ quan sát cố định, không phải
giới hạn thực thi. `positions_pipe` mang toàn bộ danh sách vị thế để Python giữ
đúng lifecycle và PnL cho cả lệnh thứ 6 trở đi.

## Pending recovery cancel

Để tránh tình trạng pending order treo làm hệ thống giữ `HOLD` quá lâu, bridge theo dõi chữ ký `orders_pipe`. Nếu cùng một pending snapshot tồn tại qua đủ số decision, bridge có thể phát `CANCEL_ORDERS` bằng cơ chế recovery:

```yaml
continuous:
  pending_cancel_after_decisions: 12
  pending_cancel_interval_decisions: 4
```

Giá trị này được truyền vào `MT5BacktestConfig`. Khi pending biến mất hoặc `orders_pipe` đổi sang lệnh mới, bộ đếm được reset, nên lệnh mới không bị hủy vì tuổi của lệnh cũ.

## Log mong đợi

Khi pending bị treo, bridge có thể ghi:

```text
[Bridge] served=... action=7(CANCEL_ORDERS) ... bootstrap=1 positions=0 day_trades=12
```

`bootstrap=1` ở đây chỉ có nghĩa là recovery action của bridge. Nó không còn là market-entry bootstrap.
`day_trades` chỉ là telemetry; daily min/max không còn được dùng để khóa entry.

## Nếu vẫn không có lệnh mới

Khi balance/equity đứng yên quanh một mức và không có entry mới, kiểm tra:

- `orders_pipe` có còn pending order không;
- action `7(CANCEL_ORDERS)` có được ghi sau ngưỡng `pending_cancel_after_decisions` không;
- EA có trả acknowledgement `cancelled=1` hoặc `ORDER_REMOVED` không;
- `risk.max_positions` có đúng bằng `0`, `pending_expiry_bars`, single-position/loss guard và broker retcode có chặn lệnh mới không.
