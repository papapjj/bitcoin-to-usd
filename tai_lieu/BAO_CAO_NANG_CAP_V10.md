# Báo cáo nâng cấp V10

## Vấn đề được xử lý

Các log trước cho thấy topology giữ nguyên trong toàn run, thiếu lời giải thích vì sao growth không xảy ra, và telemetry đầy đủ được lặp lại trên mỗi decision gây file rất lớn. Một hồ sơ legacy cũng từng mâu thuẫn giữa owner/action trong khi context không ổn định; V10 không cho loại record đó xuất hiện như quyết định mạng hợp lệ.

## Thay đổi code chính

1. `he_chuyen_gia_ngu_canh.py`
   - Ghi identity memory/checkpoint/root run/expert run.
   - Lifecycle không còn tạo/cắt topology; credit chỉ đi vào cơ chế học được cấu hình.

2. `live_observatory.py`
   - Tách path theo root run.
   - Phát `active_run.json`.
   - Full topology snapshot theo thay đổi topology; decision còn lại dùng topology reference + delta.
   - Context chưa ổn định không tạo fake topology telemetry.

3. `nhat_ky_giai_thich.py`, `causal_ledger.py`, `mt5_runner.py`
   - Loại các field action legacy khỏi log phát ra.
   - Chỉ semantic/wire action được dùng cho audit.

4. UI Java V10
   - Thêm dashboard `Theo dõi phát triển mạng` có summary, timeline, chi tiết không quota.
   - Tự resolve `active_run.json`.
   - Vẫn là read-only, không ghi vào policy/trainer/MT5.

## Kiểm chứng đã chạy

- Python compile.
- Java UI compile.
- Test V10:
  - context không ổn định không có topology giả;
- lifecycle không làm thay đổi topology;
  - owner memory/policy identity tách biệt;
  - run-scoped telemetry và active pointer;
  - snapshot full topology không lặp khi topology không đổi.

## Chưa thể xác minh trong môi trường đóng gói

- Compile EA MQL5.
- Strategy Tester.
- Feature parity 119 input.
- Ticket/fill/close lifecycle MT5 thật.

Các mục chưa kiểm chứng không được coi là đã hoàn thành chỉ vì Python/UI pass.
