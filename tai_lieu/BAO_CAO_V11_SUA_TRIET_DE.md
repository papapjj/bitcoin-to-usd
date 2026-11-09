# Báo cáo V11 — sửa triệt để UI telemetry và log phát triển mạng

## Lỗi gốc

Run gần nhất có ba log JSONL tổng khoảng 1.7 GB. V10/V10.2 đặt `0 = không giới hạn`, nhưng UI hiểu sai thành đọc toàn bộ file ngay trong poll đầu. Vì parse không hoàn tất, snapshot không được apply và giao diện hiện `0 neuron`, `0 liên kết` dù telemetry thật đã tồn tại trong `outputs/runs/<root_run_id>/`.

Trainer V10 cũng lặp topology đầy đủ trong đa số decision ở nhật ký phát triển mạng, làm file tăng nhanh và khiến UI càng khó khởi động.

## Bản vá V11

- Resolver `active_run.json` hoàn chỉnh theo workspace/run-scoped path, có fallback scan run mới nhất.
- Ingest streaming theo offset/chunk 8 MiB, không cắt record/dữ liệu/lịch sử.
- Chunk tự tăng đến khi có một dòng JSONL hoàn chỉnh; tránh bị kẹt ở topology snapshot lớn.
- Trạng thái/topology active hiện hành luôn giữ đầy đủ, còn history chỉ giữ index để không nổ heap.
- Full topology được deduplicate theo policy instance + topology id; event sau dùng reference + delta.
- Facade expert emit decision sau khi owner/action/growth review hoàn chỉnh.
- V10/V10.2 launcher bị chuyển hướng sang V11 để không thể mở nhầm JAR cũ.
- V11 có script kiểm tra độc lập đường dẫn telemetry mà không nạp toàn bộ JSONL vào RAM.

## Xác minh đã chạy

- Java compile và đóng gói JAR V11.
- Resolver Java smoke test với manifest dùng path kiểu Windows.
- Live tailer đọc được dữ liệu telemetry lớn theo chunk 8 MiB.
- Python compile pass.
- Nhóm test context-owner/action contract/network development/facade telemetry pass.

## Chưa thể xác minh trong môi trường hiện tại

- Compile EA MQL5.
- MT5 Strategy Tester.
- Execution ledger/ticket/lifecycle MT5 thật.
