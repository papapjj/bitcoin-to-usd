# Vá Walk-Forward — 01/07/2026

## Sửa thực hiện

- `--train-window` là số ngày huấn luyện thực.
- `--test-window` là số ngày kiểm tra ngoài mẫu (OOS) thực, tách tuyệt đối khỏi train.
- `--step` là độ dịch cửa sổ; mặc định bằng `--test-window`.
- Thêm `--seed-count` để tạo ma trận OOS × seed độc lập.
- Journal được cô lập theo từng ô window × seed.
- Nếu khoảng ngày không đủ `train + test`, CLI báo lỗi rõ thay vì trả `windows: 0` và `-Infinity`.
- Thêm lệnh `walk-forward-he-chuyen-gia-mt5` cho kiến trúc một context — một chuyên gia — một policy; không gọi đường policy chung cũ.

## Lệnh tối thiểu hợp lệ

```powershell
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml walk-forward-he-chuyen-gia-mt5 --from-date 2025.01.01 --to-date 2025.01.29 --train-window 7 --test-window 7 --step 7 --seed-count 3 --output outputs/walk_forward_he_chuyen_gia --runner-dir mt5_v2_runner --timeout-sec 300 --mt5-visual
```

Khoảng `2025.01.01 → 2025.01.02` chỉ dài 1 ngày, không thể tạo một cửa sổ cần tối thiểu 14 ngày.
