# Test report — Live Neural 3D V15

## Đã kiểm tra trong môi trường build

- `javac --release 17`: pass.
- JAR Java được build: pass.
- Khởi động Java Swing dưới `xvfb-run` trong 4 giây: không có stdout/stderr hoặc exception khởi tạo UI; tiến trình bị dừng chủ động bởi timeout vì UI là ứng dụng tương tác.
- Python regression liên quan telemetry/UI/CLI: `16 passed`.

Lệnh Python đã chạy:

```text
PYTHONPATH=src pytest -q tests/test_live_observatory.py tests/test_desktop_observatory_ui.py tests/test_cli.py
```

## Chưa thể xác minh tại đây

- Render fps, thao tác drag/zoom và chất lượng depth sorting trên Windows desktop thật.
- Nội dung decision trace từ EA/MT5 Strategy Tester thật.
- Mức CPU/RAM khi một graph lớn có >720 node hoặc >3.000 edge hiện diện trong telemetry.

Các phần này cần được smoke-test trên workspace Windows sau khi rebuild UI.
