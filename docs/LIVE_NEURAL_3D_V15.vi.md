# Live Neural 3D V15 — Java Swing UI

## Mục tiêu

Nâng cửa sổ **Mạng sống** từ sơ đồ 2D bị chồng node/edge thành trình quan sát 3D tương tác ngay trong Java Swing UI. Renderer chỉ đọc telemetry JSONL do trainer ghi; không gửi lệnh ngược vào Python, policy, MT5, memory, EXP hoặc graph growth.

## Không gian 3D

- **Trục X:** vùng chức năng của M1 graph: `input → encoder → memory → hypothesis → predictor → critic → selector → meta → output`.
- **Trục Y/Z:** cụm module trong cùng vùng và các neuron bên trong cụm đó.
- **Độ sâu perspective:** thay đổi theo camera; không thay đổi topology thực.
- **Node/edge phát sáng:** lấy từ active-node/active-edge trace của decision telemetry gần nhất.

Node và edge được chọn để render theo thứ tự:

1. Trace active và hai đầu active-edge: luôn được giữ.
2. Node active có activation lớn.
3. Node/edge có utility hoặc contribution lớn.
4. Topology tĩnh còn lại, tới giới hạn render.

Header vẫn hiển thị tổng topology thật trong telemetry, không phải chỉ số đã cắt để vẽ.

## Điều khiển cửa sổ 3D

| Thao tác | Kết quả |
|---|---|
| Kéo chuột trái | Xoay camera theo yaw/pitch |
| Chuột phải hoặc chuột giữa + kéo | Pan khung hình |
| Lăn chuột | Zoom perspective |
| `Tự xoay 3D` | Quay nhẹ quanh graph; không thay train |
| `Reset góc 3D` | Trả camera về góc mặc định |
| `Chỉ đường hoạt hóa` | Chỉ hiện trace active và hai đầu của edge active |
| Click node / edge | Hiện inspector: activation, weight, gate, utility, plasticity, fade, lifecycle, XYZ semantic |

## Cách dùng

1. Trong UI Java, tick **Ghi telemetry khi train**.
2. Bấm **Bật xem Live**. Một cửa sổ riêng `Mạng sống 3D — Self-Growing Trader` sẽ mở.
3. Bấm `Run` để chạy lượt train mới. Khi EA/bridge tạo decision thật, graph sẽ sáng theo trace thực.
4. Nếu `actions=0`, UI chỉ có topology/run event; đó là dấu hiệu EA ↔ bridge chưa gửi decision request, không phải lỗi renderer.

## Giới hạn an toàn

- Renderer sử dụng software projection Java2D, không cần OpenGL, JavaFX hay server/browser ngoài.
- Camera và auto-rotate không thay đổi policy hoặc kết quả train.
- Khi graph quá lớn, renderer giới hạn 720 node và 3.000 edge tĩnh; active trace vẫn được ưu tiên.
- Chưa xác minh FPS hoặc mức dùng CPU/GPU trên Windows với topology/telemetry MT5 thực tế của máy người dùng.
