# Rủi ro khi sửa

- Không đổi số input/output một phía. Python policy, action bridge và EA phải cùng feature schema v10: `input_count=137`, `output_count=20`.
- Không để lại route/config cũ khi thay model. Nếu đổi schema request/action bridge, xóa hoặc cập nhật test/docs tương ứng.
- MT5 Strategy Tester là nguồn xác thực train chính. Report HTML và tester log đáng tin hơn kết quả Python-only.
- `self_growing_request.csv` và `self_growing_action.csv` là kênh state/action giữa EA và Python. Trong Strategy Tester, file thường nằm dưới `Tester\Agent-*\MQL5\Files`, không chỉ `MQL5\Files` chính.
- `self_growing_bridge.csv` là kênh EA gửi telemetry về Python. Nếu report có nhưng bridge không có dòng, kiểm tra `BridgeFile` trong `.set` và quyền ghi file trong tester sandbox.
- Khi đổi EA input, kiểm tra `.set` trong `MQL5\Profiles\Tester` vì file này có thể override default trong source.
