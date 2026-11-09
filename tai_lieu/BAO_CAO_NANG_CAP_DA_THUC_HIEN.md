# Báo cáo nâng cấp đã thực hiện

## 1. Thay đổi code chính

### Hệ một context — một policy

Tệp mới:

```text
src/self_growing_trader/he_chuyen_gia_ngu_canh.py
```

Đã thêm:

- `CauHinhHeChuyenGia` với các invariant chống policy chung/blending.
- `BoNhanDienNguCanh`: chỉ phân loại context nhiệm vụ/trạng thái; không phát action.
- `SoDangKyChuyenGia`: một owner cho mỗi context.
- `ChuyenGiaDocLap`: policy, fitness, lifecycle, history local.
- `HeChuyenGiaMotContextMotPolicy`: facade dùng đúng một owner policy cho mỗi request.
- `QuyetDinhKhongGiaoDich`: no-op transport khi context chưa ổn định/chưa có owner, có cờ không học.
- `BoUomNguCanh`: sinh candidate context từ lifecycle thật, mặc định chỉ quan sát song song.
- Provenance được nhét vào `StepTrace.context_audit` để không phá StepTrace frozen.

### Factory và huấn luyện

Tệp sửa:

```text
src/self_growing_trader/self_growing_runner.py
src/self_growing_trader/cli.py
```

Đã thêm:

- Factory tạo policy root bằng seed khác nhau, không clone winner chung.
- Load artifact context-expert và gắn lại policy factory sau pickle.
- Lệnh `khoi-tao-he-chuyen-gia`.
- Lệnh `huan-luyen-he-chuyen-gia-mt5`.
- Guard CLI: khi profile context-expert active, lệnh train policy chung bị từ chối.

### MT5 bridge / journal

Tệp sửa:

```text
src/self_growing_trader/mt5_runner.py
mt5_v2_runner_current/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5
src/self_growing_trader/protocol_contract.py
```

Đã thêm/sửa:

- `FLAT_IDLE` bị block trước shadow learning/recovery.
- Sidecar `self_growing_action.csv.metadata.jsonl` giữ provenance mà không làm vỡ EA CSV parser 9 cột.
- Journal thêm `context_owner` và `context_audit`.
- Protocol có version provenance context-expert.
- EA source đặt bridge/ledger version 7, timeout action tối đa 60 giây, thêm expiry pending và entry safety gate.

## 2. Test đã chạy

Lệnh đã chạy trong môi trường này:

```text
PYTHONPATH=src pytest -q \
  tests/test_he_chuyen_gia_ngu_canh.py \
  tests/test_flat_idle_firebreak_v27.py \
  tests/test_mt5_runner.py \
  tests/test_context_frame_v12.py
```

Kết quả tại thời điểm hoàn thành: **74 passed**.

Các test có xác nhận trực tiếp:

- 5 policy root độc lập về object/state.
- Một request chỉ gọi một owner policy.
- Reward/lifecycle update không lan sang expert khác.
- Context child chưa active không được route live.
- Context chưa ổn định không gọi policy và không tạo dữ liệu học.
- `FLAT_IDLE` không vào shadow/recovery path.
- Static contract EA/MT5 runner vẫn pass sau patch source.

## 3. Những việc chưa thể xác nhận trong môi trường hiện tại

Môi trường hiện tại không có terminal MetaTrader/MetaEditor Windows hoạt động để compile và chạy visual tester. Vì vậy các mục sau **chưa được khẳng định hoàn thành**:

- compile EA `.mq5` mới thành `.ex5`;
- test Strategy Tester/real ticks;
- feature parity audit 119 input từ EA;
- ledger/report reconciliation sau build EA mới;
- UI Java đọc sidecar provenance live.

Không được dùng artifact này để live money trước khi toàn bộ Gate mục H trong checklist pass.

---

## Bổ sung: log giải thích quyết định và phát triển cấu trúc

Đã bổ sung các tệp audit chỉ đọc:

```text
src/self_growing_trader/nhat_ky_giai_thich.py
outputs/ho_so_quyet_dinh_chi_tiet.jsonl
outputs/nhat_ky_phat_trien_neuron_lien_ket.jsonl
```

Mỗi quyết định ghi chuỗi `context → chuyên gia → policy → action → neuron/edge/feature đóng góp`. Mỗi event tạo/tắt/nối lại neuron, edge hoặc module ghi lý do kỹ thuật, before/after, snapshot cấu trúc, module-need score, neural logits và trạng thái candidate. Xem `tai_lieu/LOG_GIAI_THICH_QUYET_DINH_VA_PHAT_TRIEN_CAU_TRUC.md`.
