# UI V191 — mọi workflow train dùng kiến trúc mới nhất

> Tài liệu lịch sử. UI hiện hành là V193 tìm edge; xem
> `docs/EDGE_DISCOVERY_V193.vi.md` và `ui-java/README.md`.

## Mục tiêu

V191 loại bỏ khả năng UI âm thầm train bằng profile nền cũ. Chức năng Fresh,
Frozen, Champion/Shadow, Walk-forward, quản trị expert và Live 3D vẫn còn; chỉ
nguồn cấu hình của workflow có optimizer writer được khóa.

## Contract command

Nguồn duy nhất cho cấu hình train là
`configs/he_chuyen_gia_latest_ui_v191.yaml`. File này dùng `extends` để kế thừa
toàn bộ `he_chuyen_gia_mot_context_mot_policy.yaml`, sau đó bật State-Aliasing
Memory, TensorRule action-conditioned và abstraction horizon 600. Nhờ vậy thay
đổi không phải sao chép cả profile production và có test phát hiện merge sai.

| Workflow | Có học | Config thực tế |
|---|---:|---|
| Train V191 candidate | Có | latest V191 + experiment profile V191 |
| Fresh train | Có | latest V191 |
| Champion/Shadow | Có | latest V191 |
| Walk-forward | Có ở train fold | latest V191 |
| Frozen OOS | Không | config nền người dùng chọn |
| Khởi tạo seed | Không | config nền người dùng chọn |
| Quản trị artifact | Không | config nền người dùng chọn |

Builder command và runtime preflight đều kiểm tra contract. Một lệnh có học
thiếu `configs/he_chuyen_gia_latest_ui_v191.yaml` bị từ chối trước khi tạo
process. Candidate còn phải có đúng profile experiment V191.

Latest profile hiện đồng thời bật delayed reconciliation: causal trace được
re-forward bằng trọng số hiện tại, lifecycle cũ qua importance/JS gate và không
được cập nhật encoder chung; counterfactual dùng `counterfactual_consensus_v3`.
Profile cũng bật exploration entry 10% và cold-start LCB gate: greedy entry bị
đổi thành HOLD khi TensorRule `UNKNOWN`, còn sampled exploration giữ nguyên
behavior probability/action thật trên bridge. Management không qua gate này.
Vì UI khóa profile này, mọi workflow có học khởi chạy sau thay đổi đều dùng cơ
chế mới. Process đang chạy không hot-reload và không bị dừng.

## Tạo kho và bảo toàn dữ liệu

Nút **Tạo kho mới** tạo hai tên mới trong `outputs/he_chuyen_gia/v191_runs` và
chuyển tới tab Train V191. Pipeline khởi tạo seed trước, train candidate sau;
bước sau không chạy nếu bước trước lỗi. UI không ghi đè checkpoint nguồn và
không mở lại direct online-resume.

Settings mới nằm ở `.context-expert-v191-ui.properties`. Lần đầu UI sao chép
settings V190 nếu có. Launcher V190 cũ trỏ sang V191 để shortcut cũ luôn mở
bản mới; process V190/MT5 đã chạy không bị dừng khi build V191.

## Build và kiểm tra

```powershell
python -m pytest -q tests/test_config_inheritance.py
powershell -NoProfile -ExecutionPolicy Bypass -File ui-java/build-ui.ps1
java -jar .\ui-java\build\SelfGrowingTraderControlV191.jar --self-test
.\KHOI_DONG_UI_CONTEXT_EXPERT_V191.cmd
```

Self-test phải in `V191_UI_LATEST_TRAINING_CONTRACT_OK`. Đây là bằng chứng về
wiring/config, không phải bằng chứng TensorRule đã có edge OOS.
