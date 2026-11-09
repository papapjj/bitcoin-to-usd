# Sửa lỗi UI Context-Expert V5

## Lỗi của V4

Bản V4 đã thêm các tab quan sát trong `LiveNeuralPanel.java`, nhưng **không thay đổi `Main.java`**. Vì vậy form điều khiển vẫn chỉ có các pipeline cũ như `train-mt5`, `walk-forward`, `policy-core=current` và có thể sinh lệnh policy chung cũ.

Build lại V4 không thể làm các control mới xuất hiện vì source của form chưa có chúng.

## V5 sửa gì

- Thêm pipeline CLI thật:
  - `khoi-tao-he-chuyen-gia`
  - `huan-luyen-he-chuyen-gia-mt5`
  - `walk-forward-he-chuyen-gia-mt5`
- Thêm các mục đích chạy tương ứng trong UI.
- Tự ép cấu hình `configs/he_chuyen_gia_mot_context_mot_policy.yaml` cho pipeline chuyên gia.
- Chặn pipeline policy chung khi profile chuyên gia đang bật để tránh blend.
- Thêm nhập `seed hệ`, `checkpoint hệ`, `số seed OOS`.
- Thêm trường đường dẫn **Hồ sơ quyết định** và **Sự kiện cấu trúc**.
- Khi bật telemetry, tự thêm các cờ:
  - `--live-decision-profiles`
  - `--live-structural-events`
  - `--live-max-responsible-neurons`
  - `--live-max-responsible-edges`
- Cửa sổ có marker bắt buộc: `Self-Growing Trader Control — V5 Context-Expert`.

## Cách chạy đúng

1. Thoát hoàn toàn toàn bộ UI cũ.
2. Giải nén V5 đè lên `C:\Users\Admin\Desktop\new-chat`.
3. Chạy `KHOI_DONG_UI_CONTEXT_EXPERT_V5.cmd` ở thư mục gốc.
4. Kiểm tra tiêu đề cửa sổ có đúng `V5 Context-Expert`.
5. Trong **Mục đích**, chọn `Huấn luyện hệ chuyên gia MT5` hoặc `Walk-forward hệ chuyên gia`.

Nếu tiêu đề không có V5, Windows đang mở EXE/JAR ở thư mục khác hoặc shortcut cũ; không phải bản vừa build.

## Dấu hiệu command đúng

Huấn luyện hệ chuyên gia phải có:

```text
--config configs/he_chuyen_gia_mot_context_mot_policy.yaml
huan-luyen-he-chuyen-gia-mt5
--live-decision-profiles outputs/neural_live.ho_so_quyet_dinh.jsonl
--live-structural-events outputs/neural_live.phat_trien_cau_truc.jsonl
```

Nó **không được** có `train-mt5` hoặc `--policy-core current` như một policy chung live. `--policy-core` trong command chuyên gia chỉ là topology factory khi sinh owner độc lập; nó không route lệnh live thay expert.
