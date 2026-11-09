# Patch Manifest — Live View Toggle UI V14.2

## Chép đè từ patch vào thư mục dự án

```text
src/self_growing_trader/desktop_app.py
tests/test_desktop_observatory_ui.py
docs/LIVE_VIEW_TOGGLE_UI_V14_2.vi.md
docs/LIVE_VIEW_TOGGLE_UI_V14_2_PATCH_MANIFEST_2026-06-27.vi.md
```

## Giao diện thay đổi

Trong tab `Train MT5`, mục quan sát có:

```text
[Bật xem Live] / [Tắt xem Live]
[Ghi telemetry khi train]
[Xóa event cũ trước run]
```

`Bật/Tắt xem Live` chỉ điều khiển tab và render UI. `Ghi telemetry khi train` phải được chọn trước khi bắt đầu lượt train để tạo trace mới.

## Xác minh sau khi áp dụng

```bat
python -m self_growing_trader.cli desktop-ui
```

1. Nút ban đầu hiển thị `Bật xem Live`.
2. Bấm nút: tab `Mạng sống` xuất hiện; nút đổi thành `Tắt xem Live`.
3. Bấm lại: tab ẩn; tiến trình train đang chạy không bị dừng.
