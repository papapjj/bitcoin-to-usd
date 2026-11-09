# Live View Toggle UI V14.2

## Mục tiêu

Màn hình **Train MT5** có nút độc lập để bật/tắt việc xem trực tiếp mạng neuron trong UI. Nút này không phải nút train và không thay đổi policy, MT5 bridge, ContextFrame, EXP, plasticity hay module growth.

## Điều khiển mới

Trong khung **Theo dõi mạng neuron**:

- **Bật xem Live**: mở tab `Mạng sống`, bắt đầu đọc và render JSONL telemetry hiện có.
- **Tắt xem Live**: ẩn tab `Mạng sống` và dừng refresh/render ở UI.
- **Ghi telemetry khi train**: cấu hình event stream cho lượt train kế tiếp. Cần bật trước khi nhấn `Train MT5` nếu muốn có trace mới.
- **Xóa event cũ trước run**: chỉ xóa JSONL trước khi UI khởi động lượt train mới.

## Contract vận hành

| Thao tác | Train | Telemetry writer | Render UI |
|---|---:|---:|---:|
| Bật xem Live | Không đổi | Không đổi | Mở/tiếp tục |
| Tắt xem Live khi train đang chạy | Không dừng | Vẫn ghi nếu đã bật lúc start | Dừng/ẩn |
| Bật xem Live giữa lượt train không ghi telemetry | Không đổi | Không thể bật ngược cho lượt hiện tại | Mở nhưng chờ stream; hiệu lực writer từ lượt sau |
| Tắt `Ghi telemetry khi train` trước Start | Không đổi | Không tạo event stream mới | Có thể xem JSONL cũ nếu nạp file đó |

## Lý do tách hai điều khiển

`Ghi telemetry khi train` là một contract của trainer lúc start. Thay đổi nó giữa decision sequence có thể làm trace một phần trial thiếu dữ liệu. Ngược lại, `Bật/Tắt xem Live` chỉ điều khiển UI nên có thể thao tác bất kỳ lúc nào mà không làm lệch kết quả train.

## Kiểm thử

- `test_desktop_observatory_ui.py`
- `test_cli.py`
- `python -m py_compile src/self_growing_trader/desktop_app.py`
