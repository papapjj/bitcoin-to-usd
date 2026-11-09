# V7 — đầu ra tư duy và phát triển cấu trúc theo chuyên gia owner

## Vấn đề đã xác định

1. `khoi-tao-he-chuyen-gia` chỉ sinh năm policy owner gốc. Lệnh này **không gọi `policy.act()`**, vì vậy không có activation, hypothesis, selector, critic hoặc output tư duy để UI hiển thị.
2. V6 giấu topology ở ba tầng: quota UI 720 neuron/1.100–3.000 liên kết, quota telemetry activity/attribution, và quota cấu hình 512 neuron/16.384 liên kết.
3. Runtime MT5 owner-local đã đặt `graph_trial_mode=false`, nên candidate structural growth không thể nhận lifecycle evidence để mọc neuron/liên kết trong policy owner.

## Thay đổi V7

### 1. Output tư duy thật của M1

Mỗi `M1GraphModule.forward()` sau khi tính quyết định ghi snapshot read-only:

- `encoder`;
- `memory`;
- `hypotheses`;
- `predictor`;
- `critic`;
- `selector`;
- `candidate_scores` và `candidate_probabilities`;
- `candidate_actions`;
- hypothesis được chọn, confidence, critic value;
- prediction error, novelty, conflict, entry gate;
- activity theo region và tổng neuron/cạnh active.

Hồ sơ quyết định JSONL có trường `dau_ra_tu_duy`. UI có tab **Đầu ra tư duy**.

### 2. Không quota số học cố định

Trong `configs/he_chuyen_gia.yaml`, các trường sau là `0`:

- số module;
- số neuron / liên kết graph;
- neuron ẩn một module;
- recurrent/lateral/cross-module edge;
- fanout;
- neuron/module micro-growth trong trial;
- activity/state/attribution telemetry.

`0` có nghĩa **không áp trần số lượng cố định**. Nó không có nghĩa bỏ toàn bộ kiểm soát: vẫn giữ context-owner isolation, graph invariant, evidence MT5, causal/replay validation, safety governor và giới hạn tài nguyên CPU/RAM của máy.

### 3. Growth chỉ trong policy owner

Mỗi expert có:

```text
context ổn định
→ expert owner
→ graph_trial_id riêng
→ graph_trial_mode=true
→ lifecycle MT5 đã xác nhận
→ micro-growth / fade / prune bên trong policy owner
```

Không có weight, neuron, edge, memory, fitness hay growth event nào truyền sang expert khác.

### 4. UI full topology

UI V7 có checkbox **Toàn bộ topology**, bật mặc định:

- vẽ toàn bộ neuron/link được telemetry ghi;
- không cắt 720 neuron;
- không cắt 1.100/3.000 edge;
- không cắt 720 edge quanh neuron được focus.

Tắt checkbox này hoặc bật **Chế độ mượt** nếu graph quá lớn làm UI nặng.

## Cách kiểm tra sau khi chạy

Không kiểm tra thinking bằng `khoi-tao-he-chuyen-gia`.

1. Khởi động UI bằng `KHOI_DONG_UI_CONTEXT_EXPERT_V7.cmd`.
2. Chọn **Huấn luyện hệ chuyên gia MT5**.
3. Bật ghi telemetry.
4. Mở Live 3D.
5. Chờ context ổn định và một expert owner được route.
6. Mở tab **Đầu ra tư duy**.

Nếu tab nói `context chưa ổn định`, đó là fail-closed: policy không được gọi và không có thinking output. Nếu context ổn định nhưng tab vẫn rỗng, kiểm tra file `outputs/ho_so_quyet_dinh_chi_tiet.jsonl` phải chứa khóa `dau_ra_tu_duy`.

## Kiểm thử đã chạy

- Python compile pass.
- Java UI compile pass.
- `pytest -q tests`: 12 passed.
- Smoke init kho chuyên gia: 5 expert owner.
- Smoke policy act: profile có `dau_ra_tu_duy`, 189 neuron attribution và 557 edge attribution; vượt các giới hạn V6 cũ 40/64.
