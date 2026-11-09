# Bridge v10 — năm tác nhân đồng bộ

## Hợp đồng action

Một request thị trường tạo một action packet duy nhất. Packet hợp lệ có một dòng cho chế độ cũ hoặc đúng năm dòng cho chế độ `nam_tac_nhan_dung_chung`. Năm dòng phải dùng chung `schema_version=10`, `event_id`, `sequence`, khai báo `agent_count=5`, có đủ `agent_id` từ 0 đến 4 và `trace_id` riêng.

Python ghi file tạm rồi replace atomically. EA không thực thi packet thiếu, trùng ID, sai sequence/event hoặc sai phiên bản.

## Phân quyền và xử lý hành động

Python lấy mẫu từ mặt nạ kết hợp giữa trạng thái tài khoản và quyền cố định của
từng tác nhân. EA kiểm tra lại quyền trên chính action wire; action sai quyền
không được đổi âm thầm thành `HOLD` mà nhận receipt `AUTHORITY_DENIED`.

| ID | Trách nhiệm | Hành động ngoài `HOLD` |
|---:|---|---|
| 0 | entry mua | `MARKET_BUY`, `BUY_LIMIT`, `BUY_STOP` |
| 1 | entry bán | `MARKET_SELL`, `SELL_LIMIT`, `SELL_STOP` |
| 2 | quản lý vị thế | `CLOSE_WORST`, `CLOSE_BEST`, `MANAGE_POSITION` |
| 3 | quản lý lệnh chờ | `CANCEL_ORDERS` |
| 4 | quản trị toàn danh mục | `CLOSE_ALL` |

Mỗi action ngoài `HOLD` chỉ có một chủ. Hai chủ entry có thể hoạt động đồng thời,
không qua bộ chọn hướng thắng. `CLOSE_ALL` của tác nhân 4 là lệnh phủ quyết duy
nhất: các action khác trong cùng packet nhận `BATCH_AUTHORITY_GOVERNOR`. Khi
không có phủ quyết, EA xử lý tất cả action hợp quyền; thứ tự packet không trao
quyền ưu tiên chiến lược cho tác nhân có ID nhỏ hơn.

V193 dùng `MaxPositions=0`, nghĩa là không có quota số vị thế/lệnh chờ trong
Python hoặc EA. Năm slot trong 137 đầu vào chỉ là cửa sổ quan sát chi tiết;
`positions_pipe` biến độ dài vẫn xuất toàn bộ vị thế để giữ đúng owner, PnL và
lifecycle. Một giá trị `MaxPositions` dương chỉ còn là chế độ tương thích tùy
chọn cho cấu hình cũ.

Hàng rào an toàn vẫn tồn tại dưới dạng biên nhận thật:

- broker vẫn kiểm tra giá, khối lượng, margin, stop level và trả retcode riêng;
- risk của mọi entry trong cùng packet được chia đều để không phóng đại risk tổng;
- `HOLD` nhận `NOOP_HOLD`; chỉ được đưa vào credit học khi trách nhiệm của vai
  trò đang hoạt động, không được thưởng chỉ vì tài khoản trống.

Lệnh LIMIT/STOP chưa khớp được EA tự hủy sau `PendingExpiryBars`; profile V193
đặt giá trị này bằng 5. Sổ thực thi ghi `UNFILLED_TIMEOUT` và tuổi lệnh để Python
phạt `0.25R` đúng trace tạo lệnh.

Delay 0–3 được lưu riêng cho từng agent, không còn một biến delayed-action toàn cục có thể bị agent khác ghi đè.

## Feedback và học

Execution pipe phát một record cho mỗi `trace_id`, phân cách bằng `^`, kèm `agent_id`. Python lọc đồng thời theo `sequence + trace_id + agent_id`; vì vậy event của hai agent trong cùng tick không thể bị gộp. Quan hệ `trace_id → agent_id` được khóa từ lúc phát action và không thể đổi chủ khi lifecycle đóng.

Khi request kế tiếp đến, năm trace nhận năm receipt riêng. Python đọc lifecycle đã xác nhận, position mark-to-market, chi phí, risk và conflict rồi facade gọi `observe_five_economic_transition` đúng một lần cho batch năm phần tử. Reward gồm PnL delta sau chi phí/phạt rủi ro/drawdown/conflict và bonus cạnh tranh zero-sum so với trung bình năm agent. `HOLD` của chủ entry có vị thế được học delta marked-R của owner; các vai trò quản lý chỉ học `HOLD` khi có vị thế, lệnh chờ hoặc mức phơi nhiễm thuộc trách nhiệm của mình. `HOLD` không có trách nhiệm bị loại khỏi reward, actor, critic và entropy. Đường `act()` luân phiên một agent đã bị loại; chế độ Five bắt buộc dùng `act_nam_tac_nhan()`.

Chi tiết công thức và provenance nằm tại `docs/FIVE_COMPETITIVE_REWARD_V1.vi.md`.

UI Java V193 hiển thị hợp đồng này trong tab đọc-only **Five + Bridge v10**. Banner phân biệt rõ encoder v3 + Pattern Tree, packet `5 × T`, target attention và các head delay/offset; nút chạy bị khóa khi đang đứng ở tab thông tin để không biến sơ đồ thành một workflow giả.

## Giới hạn bằng chứng

Compile MetaEditor và test Python chứng minh contract/code path. Chất lượng giao dịch, race thực tế của broker và edge OOS vẫn phải được xác nhận bằng Strategy Tester với report, bridge CSV và execution ledger v14.
