# Cập nhật UI: giải thích quyết định và phát triển cấu trúc

## Mục tiêu

UI Java Swing phải hiển thị trực tiếp các log giải thích đã được trainer ghi, thay vì yêu cầu mở JSONL thủ công. Mọi thành phần mới là **read-only**: không thay đổi policy, graph, memory, risk governor, bridge hay MT5.

## Luồng dữ liệu

```text
trainer / live_observatory
        ↓ append-only
neural_live.jsonl
neural_live.ho_so_quyet_dinh.jsonl
neural_live.phat_trien_cau_truc.jsonl
        ↓ tail read-only
LiveNeuralPanel.java
        ↓
UI Java: 3D + Giải thích quyết định + Phát triển cấu trúc + Inspector neuron
```

## Các thay đổi đã thực hiện

### 1. Tail hai JSONL mới

`LiveNeuralPanel` tạo hai tailer độc lập:

- `decisionProfileTailer`: đọc `*.ho_so_quyet_dinh.jsonl`;
- `structuralEventTailer`: đọc `*.phat_trien_cau_truc.jsonl`.

Path mặc định được dẫn xuất từ `Live events`:

```text
neural_live.jsonl
→ neural_live.ho_so_quyet_dinh.jsonl
→ neural_live.phat_trien_cau_truc.jsonl
```

Nếu file nằm trong MT5 `Common/Files` hoặc `Tester/Agent-*/MQL5/Files`, UI chọn bản mới nhất theo cùng cơ chế tìm runtime file hiện hữu.

### 2. Tab `Giải thích quyết định`

Hồ sơ quyết định hiển thị:

```text
Context sở hữu:
  mã context, context cha, chuyên gia, policy, phiên bản, lifecycle, độ tin cậy.

Action:
  policy đề xuất → legality/mask → bridge/MT5.

Lý do:
  định tuyến, lựa chọn candidate, reasoning, legality, recovery,
  expectancy và gate an toàn.

Evidence:
  feature ảnh hưởng, neuron attribution, edge attribution.
```

### 3. Tab `Phát triển cấu trúc`

Mỗi event tạo/cắt/nối lại có:

```text
context / expert / policy owner
trace ID và giai đoạn
hành động cấu trúc
lý do
before/after neuron và edge count
target snapshot
neuron mới
edge mới
module need / score / causal evidence
lịch sử event gần nhất
```

### 4. Highlight 3D và inspector neuron

- Neuron/edge thuộc attribution của hồ sơ quyết định gần nhất được ưu tiên render.
- Chúng được vẽ màu cam; neuron được tăng halo/radius nhẹ.
- Legend thêm tổng số `trách nhiệm xN/xE`.
- Tab Neuron có filter `Chịu trách nhiệm quyết định`.
- Inspector neuron hiển thị đóng góp tuyệt đối/có dấu, hướng tín hiệu và edge liên quan.

## Cách chạy

```powershell
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml walk-forward-chuyen-gia `
  --rounds 3 `
  --seed-count 3 `
  --output outputs/walk_forward_chuyen_gia `
  --train-window 7 `
  --test-window 7 `
  --step 7 `
  --runner-dir mt5_v2_runner `
  --timeout-sec 300 `
  --mt5-visual `
  --live-observatory `
  --live-events outputs/neural_live.jsonl `
  --live-decision-profiles `
  --live-structural-events
```

Sau đó mở UI Java, bật **Ghi telemetry khi train** và **Bật xem Live**. Chọn tab `Giải thích quyết định` hoặc `Phát triển cấu trúc`.

## Quy ước đọc kết quả

- `Màu cam` không có nghĩa neuron là nguyên nhân duy nhất; đó là neuron/edge có attribution mạnh nhất trong trace.
- Một event tạo cấu trúc chỉ là bằng chứng rằng controller đã tạo candidate hoặc thay đổi cấu trúc. Nó không chứng minh event đó đã cải thiện performance.
- Outcome thực chỉ được xác nhận bằng `execution ledger` và lifecycle MT5.
- `FLAT_IDLE`, bridge timeout và EA reject không được hiển thị như policy failure hoặc evidence học.

## Kiểm thử đã chạy

```text
javac --release 17: pass
UI explainability smoke: pass
pytest trọng tâm: 35 passed
```
