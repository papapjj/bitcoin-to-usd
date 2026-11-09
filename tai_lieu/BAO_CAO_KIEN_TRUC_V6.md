# Báo cáo kiến trúc V6

## Kiến trúc được phép

```text
context ổn định → một chuyên gia owner → một policy độc lập → bộ an toàn → MT5
```

Router chỉ nhận diện context. Router không tạo action, không sửa SL/TP/risk và không kết hợp đầu ra của nhiều policy.

## Bề mặt triển khai V6

- Một config duy nhất: `configs/he_chuyen_gia.yaml`.
- Năm context gốc, năm instance policy độc lập.
- CLI chỉ có các lệnh khởi tạo, huấn luyện MT5, walk-forward, quản trị owner và quan sát live.
- UI chỉ tạo command cho kiến trúc context-owner.
- Artifact chỉ được nhận khi là `HeChuyenGiaMotContextMotPolicy`.
- Child graph chỉ là candidate nội bộ trong cùng owner; không được route live thành policy thứ hai.

## Invariant bắt buộc

1. Thiếu `context_id`, `expert_id`, `policy_id` hoặc context không ổn định → `KHÔNG_GỬI_LENH`, không bridge order, không học.
2. Một request có đúng một chuỗi `context → chuyên gia → policy`.
3. Không voting, blending, fallback hoặc bootstrap action.
4. Lifecycle chỉ cập nhật memory, fitness, reward/loss và growth của policy owner tạo trace.
5. `FLAT_IDLE`, timeout và lỗi bridge không tạo feedback neural.
6. Attribution chỉ dùng graph trace được phát trực tiếp bởi owner policy.

## Xác minh trong môi trường này

- Python compile: pass.
- Kiểm tra V6 source/config/bridge: pass.
- Test V6: pass.
- Java UI V6 compile: pass.

## Cần xác minh tại máy có MT5

- Compile EA MQL5 thành EX5.
- Strategy Tester hoặc real-tick MT5.
- Parity 119 feature Python ↔ EA.
- Execution ledger thực tế sau một run MT5.
