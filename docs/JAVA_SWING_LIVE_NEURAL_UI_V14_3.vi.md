# Java Swing Live Neural UI V14.3

## Mục tiêu

Nhúng trực tiếp hệ quan sát mạng neuron vào `ui-java`, không tạo desktop UI Python, browser dashboard hoặc HTTP server mới.

## Điều đã triển khai

- Thêm checkbox `Ghi telemetry khi train` vào form Java.
- Thêm `Live events` mặc định `outputs/neural_live.jsonl` và `Sample every`.
- Thêm nút toggle `Bật xem Live` / `Tắt xem Live`.
- Bật toggle sẽ mở tab `Mạng sống` ngay trong `JTabbedPane` Java hiện có.
- Tắt toggle sẽ dừng tail/read/render và ẩn tab; không gửi stop đến Python/MT5.
- Khi telemetry bật, command builder tự thêm:
  ```text
  --live-observatory
  --live-events <path>
  --live-sample-every <N>
  --live-reset
  ```
- Panel Java tail JSONL append-only bằng `FileChannel`; không giữ lock file, không ghi trở lại trainer.
- Nếu event file bị reset/compaction/rotate (size giảm), Java tailer reset offset và dựng snapshot mới.
- Thêm JSON parser tối giản nội bộ để không có dependency third-party.
- Vẽ graph M1 theo topology event thật; trace activation/edge contribution được lấy từ event `decision` hoặc `feedback`.
- Checkbox `Toàn bộ topology` chỉ điều khiển cách vẽ trong Java; trainer phải gửi đủ dữ liệu trước. Với cấu hình hiện tại `observability.max_active_edges=0` nghĩa là telemetry không cắt active edge trước khi UI nhận.
- Render limit bảo vệ UI khi graph có nhiều edge chỉ áp dụng khi tắt `Toàn bộ topology`: active trace luôn được giữ, sau đó ưu tiên contribution/utility cao.
- Click node/edge để xem inspector chi tiết.
- Timeline chứa `run_started`, `topology`, `trial_started`, `decision`, `feedback`, `development`, `lifecycle`, `trial_finished` và các `growth_events` có trong telemetry.
- Cập nhật `build-ui.ps1` để compile `--release 17`.

## Hợp đồng an toàn

Java UI chỉ đọc `outputs/neural_live.jsonl`.

Không có code Java nào:

- ghi vào `.pkl` policy;
- gọi phương thức policy;
- đổi action / action mask;
- đổi weight, graph, memory, EXP, growth hoặc promotion;
- gửi command đến EA/MT5 ngoài process CLI vốn đã tồn tại.

## Kiểm thử cần làm trên Windows

1. Build `ui-java\build-ui.cmd` bằng JDK 17+.
2. Mở EXE/JAR tại workspace project.
3. Bật `Ghi telemetry khi train` và `Bật xem Live`.
4. Chạy Smoke MT5, không bật Visual ở bước đầu.
5. Xác nhận command preview có `--live-observatory` và `--live-events`.
6. Xác nhận file `outputs\neural_live.jsonl` xuất hiện.
7. Khi bridge có request, tab `Mạng sống` phải hiện `decision`, node/edge active và action.
8. Bấm `Tắt xem Live` trong khi train: process Python vẫn chạy, journal vẫn tăng.
9. Bấm bật lại: tab đọc event mới và dựng graph từ topology gần nhất.

## Điều không thể tự chứng minh trong Linux container

Không thể mở Windows Swing UI hoặc chạy MT5 Strategy Tester/EA thực ở đây. Mã Java đã compile chuẩn Java 17; cần chạy smoke trên máy Windows để xác nhận font, DPI, tốc độ repaint và event path đúng với workspace MT5 của bạn.
