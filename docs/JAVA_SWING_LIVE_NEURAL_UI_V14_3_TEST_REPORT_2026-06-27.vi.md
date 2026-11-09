# Test report — Java Swing Live Neural UI V14.3

## Đã chạy

| Kiểm tra | Kết quả |
|---|---|
| `javac --release 17` toàn bộ `ui-java/src` | Pass |
| Build `ui-java/build/SelfGrowingTraderControl.jar` | Pass |
| `jar --describe-module` | Pass, main class `com.selfgrowingtrader.control.Main` |
| Java JSON telemetry smoke | Pass |
| `pytest -q tests/test_live_observatory.py tests/test_cli.py tests/test_desktop_observatory_ui.py` | 16 passed |

## Hợp đồng đã kiểm tra bằng source/build

- Nút `Bật xem Live` bật telemetry checkbox và nhúng tab `Mạng sống`.
- Nút `Tắt xem Live` gọi `stopViewing()` và remove tab, không gọi `stopCommand()`.
- Khi `Ghi telemetry khi train` được chọn, command builder thêm `--live-observatory`, `--live-events`, `--live-sample-every` và tùy chọn `--live-reset`.
- `LiveNeuralPanel` chỉ dùng `FileChannel` read-only để tail JSONL.
- UI Java không có code gọi policy/MT5 write API.

## Chưa thể kiểm tra trong Linux container

- Mở Swing UI và EXE Windows thật.
- MT5 Strategy Tester + EA bridge thật.
- Rendering/scroll/DPI trên máy Windows của bạn.
- Độ mượt khi telemetry chứa graph/edge volume của run dài thực tế.

Cần chạy Smoke MT5 trong workspace Windows để xác nhận `actions > 0`, file JSONL tăng và tab Java hiển thị activation.
