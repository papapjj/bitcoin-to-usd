# Kiến trúc năm tác nhân dùng chung một bộ trọng số

Cấu hình đang dùng là `configs/he_chuyen_gia_latest_ui_v193.yaml`.

## Phần học được

Hệ thống chỉ có một bộ tham số lõi:

- 78 giá trị cấu trúc thị trường đi qua hai lớp tích chập một chiều rồi gom trung bình và cực đại;
- 8 giá trị tài khoản và 11 giá trị thực thi đi qua hai bộ mã hóa đầy đủ riêng;
- Cửa sổ quan sát 5 vị thế × 8 giá trị đi qua cùng hai lớp rồi lấy cực đại như một tập không thứ tự, đồng thời giữ véc-tơ riêng cho từng vị thế; đây không phải quota thực thi;
- nhận dạng tác nhân và hành động trước dùng véc-tơ mã hóa học được;
- các nhánh được ghép và chiếu thành 1024 chiều trước tầng nhớ;
- Một tầng nhớ dài-ngắn có cổng gồm 4096 phần tử.
- Một Transformer hậu-LSTM dùng năm agent làm năm token, bottleneck 512, 8 head và FFN 2048; output residual trở lại 4096.
- Một `InternalPolicyState` liên tục nằm ngay sau biểu diễn hậu-LSTM. Nó lấy
  trung bình causal của năm hidden để tạo một policy state dùng chung, rồi
  điều chế bounded action-logit residual. State này không có tên chiến lược,
  không có `strategy_id` và không có router bên ngoài.
- Các đầu phân rã cho loại hành động, mục tiêu vị thế, độ trễ, rủi ro, khoảng vào lệnh, quản lý vị thế và giá trị trạng thái.

Năm tác nhân không có năm bộ trọng số. Cả năm cùng dùng một bộ ma trận duy nhất.

## Phần riêng và quyền của từng tác nhân

Mỗi tác nhân có riêng:

- một trạng thái ẩn 4096 chiều;
- một ô nhớ 4096 chiều;
- một lịch sử truyền ngược theo thời gian;
- một chỉ số nhận dạng được biến thành véc-tơ học được 32 chiều.

Vì vậy recurrent hidden/cell của tác nhân 2 không thay recurrent hidden/cell của
tác nhân 1. Transformer chỉ cho các head đọc chéo năm trạng thái hiện thời; output
đã trộn không được lưu làm recurrent memory. Cả năm vẫn cập nhật cùng một bộ tham
số học được.

Quyền hành động được cố định bởi hợp đồng `five-agent-authority-v1`:

| ID | Vai trò | Hành động có quyền |
|---:|---|---|
| 0 | Chủ entry mua | `HOLD`, `MARKET_BUY`, `BUY_LIMIT`, `BUY_STOP` |
| 1 | Chủ entry bán | `HOLD`, `MARKET_SELL`, `SELL_LIMIT`, `SELL_STOP` |
| 2 | Quản lý vị thế | `HOLD`, `CLOSE_WORST`, `CLOSE_BEST`, `MANAGE_POSITION` |
| 3 | Quản lý lệnh chờ | `HOLD`, `CANCEL_ORDERS` |
| 4 | Giám sát danh mục | `HOLD`, `CLOSE_ALL` |

Mỗi hành động khác `HOLD` chỉ có đúng một chủ. Mặt nạ quyền được áp dụng trước
khi lấy mẫu; EA kiểm tra lại cùng hợp đồng và trả receipt `AUTHORITY_DENIED` nếu
packet ngoài hệ thống vi phạm quyền. Véc-tơ nhận dạng tác nhân hiện mang ý nghĩa
vai trò, nhưng kích thước đầu vào, đầu ra và tensor checkpoint không đổi.

## Policy state nội bộ (kiến trúc v4)

`SharedFiveTradingPolicy` không được gán trước các nhãn như trend,
mean-reversion hay breakout. Sau LSTM/Transformer, `InternalPolicyState` tự
học một véc-tơ liên tục từ hidden causal dùng chung của khung quyết định. Véc-tơ
này được đưa thẳng vào action surface qua một residual bị giới hạn; vì vậy
PPO/`FiveAdam` cập nhật cùng nhánh đã tạo ra action, thay vì ghi một nhãn chiến
lược ở ngoài mạng.

Mạng vẫn có thể tự hình thành nhiều decision circuit theo context và reward,
nhưng hệ thống không tuyên bố chúng là các strategy ID riêng. Một regularizer
label-free chỉ giữ variance tối thiểu và giảm sao chép giữa các chiều latent;
reward kinh tế mới quyết định khác biệt nào có ích. Trace runtime ghi
`five_internal_policy_state` để audit causal representation, không dùng nó làm
nhãn tương lai hay chain-of-thought.

## Quan sát

Đường năm tác nhân chấp nhận hai dạng:

- một véc-tơ 137 chiều: cùng quan sát được phát cho cả năm tác nhân;
- một ma trận 5 hàng × 137 cột: mỗi tác nhân nhận quan sát riêng.

Python là chủ sở hữu chuẩn hóa đầu vào đúng một lần trước policy. Encoder không chuẩn hóa lại, chỉ khử `NaN/Inf` để tránh lan truyền giá trị không hữu hạn. Giao ước cầu nối vẫn giữ nguyên 137 đầu vào và 20 đầu ra.

“Không gian” trong bài toán này là cấu trúc giá/chỉ báo có thứ tự, không phải ảnh hay bản đồ Dota. Vì chưa có ma trận thời gian × mức giá thật, mạng dùng tích chập **1D** trên hợp đồng 78 đặc trưng ổn định; không dựng giả một ảnh 2D.

## Số tham số

Tổng số tham số học được của policy mặc định hiện tại là **116.773.939**:
encoder **843.278**, tầng nhớ 1024→4096 **83.918.848**, Transformer
**12.610.050** và các đầu ra **19.401.763** (trong đó
`InternalPolicyState` có **16.918.188**). Con số này không nhân năm vì năm tác
nhân dùng chung một bộ trọng số.

## Phần đã loại khỏi cấu hình hiện tại

Không khởi tạo chín chuyên gia theo trạng thái thị trường, strategy bank có
nhãn cố định, router strategy bên ngoài, mạng thời điểm riêng, ngân hàng luật
học được, đầu dựng lại đầu vào, mạng tò mò, mạng đánh giá ích lợi, mạng đánh
giá chi phí rủi ro, mạng tham số hành động phụ, đồ thị suy luận hoặc bộ nhớ
chèn thêm vào lõi. `InternalPolicyState` là phần học được bên trong policy
được mô tả ở trên, không phải một ngoại lệ routing.

## Cầu nối MT5 năm tác nhân

Bridge v10 ghi atomically năm dòng action cho cùng `event_id/sequence`, mỗi dòng có `agent_id`, `agent_count=5` và `trace_id` riêng. Cả năm dòng ký ức được commit trên cùng quan sát. EA chỉ chấp nhận packet đủ và không trùng ID; packet thiếu bị timeout/fail-closed.

Hai chủ entry mua và bán có thể cùng được thực thi trong một packet; hệ thống
không còn chọn một hướng thắng. Nếu tác nhân giám sát danh mục chọn `CLOSE_ALL`,
lệnh này có quyền phủ quyết và bốn tác nhân còn lại nhận receipt
`BATCH_AUTHORITY_GOVERNOR`. Nếu không có phủ quyết, mọi ý định đúng quyền được
xử lý; với V193, `MaxPositions=0` không đặt quota số vị thế và risk packet được
chia đều giữa các entry được tiếp nhận. HOLD và mỗi proposal đều có receipt
riêng. Execution pipe phân tách receipt bằng `trace_id/agent_id`, sau đó Python
gom đúng năm feedback của cùng transition để gọi bộ học chung.

`HOLD` của quản lý vị thế, quản lý lệnh chờ và giám sát danh mục chỉ tham gia học
khi trách nhiệm tương ứng đang hoạt động. `HOLD` của chủ entry chỉ nhận credit
kinh tế khi tác nhân đó đang sở hữu vị thế; nhờ vậy năm vai trò không học từ các
no-op không liên quan.

## Cạnh tranh chiến lược

Mỗi trace được gắn bất biến với một agent. Lifecycle MT5 và vị thế đang mở cấp `marked_R` cho đúng chủ; commission/swap chỉ bị trừ một lần. Ngoài reward kinh tế chung, mỗi agent nhận bonus theo phần cải thiện mới vượt trung bình năm agent. Bonus này zero-sum nên chỉ tạo áp lực chọn chiến lược tốt hơn, không làm tăng reward danh mục giả.

Cuối ngày, cả đội nhận `team_equity_R - max_drawdown`; agent có `marked_R` tốt hơn trung bình nhận thêm rank bonus có chặn `tanh`. Team-spirit tăng từ 0.30 đến 1.00 theo 10.000 packet để chuyển dần từ khám phá khác biệt sang phối hợp. Xem công thức đầy đủ tại `docs/FIVE_COMPETITIVE_REWARD_V1.vi.md`.

Checkpoint dùng encoder phẳng đời trước bị từ chối rõ ràng. Không resize tại chỗ vì thao tác đó sẽ âm thầm phá trọng số đã học; phải dùng `--fresh-system` hoặc một `--output` mới.
