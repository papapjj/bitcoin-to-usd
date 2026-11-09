# Kế hoạch nâng cấp: Một ngữ cảnh — Một chuyên gia — Một policy

## 1. Mục tiêu cuối cùng

Hệ thống không còn dùng một policy chung quyết định cho toàn bộ thị trường.

```text
Dữ liệu MT5 hợp lệ
→ nhận diện một ngữ cảnh lá ổn định
→ tìm đúng một chuyên gia sở hữu ngữ cảnh đó
→ gọi đúng một policy độc lập
→ policy đó tự xuất 20 đầu ra
→ bộ an toàn độc lập kiểm tra
→ MT5 thực thi hoặc từ chối
→ kết quả chỉ quay về policy owner.
```

Các ràng buộc kiến trúc:

- Mỗi context lá có đúng một owner.
- Mỗi request chỉ được gọi một policy.
- Không blending action/output giữa nhiều policy.
- Không có policy chung làm fallback để vào lệnh.
- Router chỉ nhận diện context; không được phát BUY, SELL, SL, TP hoặc risk.
- Policy không chia sẻ weight, graph, memory, fitness, reward history hoặc replacement evidence.
- `FLAT_IDLE`, lỗi bridge và EA từ chối không được biến thành dữ liệu học.

## 2. Phân chia trách nhiệm

| Tầng | Quyền | Không được quyền |
|---|---|---|
| Nhận diện context | Phân loại trạng thái, khóa chuyển context | Ra lệnh giao dịch |
| Chuyên gia owner | Tạo policy action 20 đầu ra | Vượt bộ an toàn |
| Bộ an toàn | Từ chối action, giới hạn rủi ro/exposure | Tự phát BUY/SELL thay policy |
| Bridge/EA | Thực thi, xác nhận ledger | Đổi policy/context |
| Lifecycle learner | Trả credit về policy đã tạo trace | Phát tán reward sang expert khác |

## 3. Phân kỳ nâng cấp

### P0 — Tính toàn vẹn Python ↔ EA ↔ MT5

Mục tiêu: không cho một policy học từ action không thực thi hoặc feature không đúng schema.

- Đồng bộ `FEATURE_SCHEMA_VERSION`, `BRIDGE_SCHEMA_VERSION`, `EXECUTION_LEDGER_SCHEMA_VERSION` giữa Python và EA.
- Ghi `mã context`, `mã chuyên gia`, `mã policy`, `phiên bản policy` ở action provenance.
- Không thay cấu trúc CSV bridge cũ khi EA chưa đọc được cột mới; dùng sidecar JSONL.
- Cấm fail-open: schema/hash/build không khớp thì không được deployment.
- Xác minh parity 119 feature từ EA sang Python.

Tiêu chí: có thể truy ngược đầy đủ `request → context → expert → policy → action → MT5 → lifecycle`.

### P1 — Firebreak dữ liệu giả

Mục tiêu: tách action chủ động khỏi telemetry rỗng/lỗi.

- `FLAT_IDLE` dừng trước shadow learning, recovery, memory, fitness và growth.
- `bridge timeout` không bị coi là HOLD hay action sai.
- `EA từ chối` được ghi là lỗi thực thi, không phạt direction policy.
- `context chưa ổn định` tạo no-op transport, gắn cờ không học.
- `chưa có expert owner` không dùng policy chung thay thế.

Tiêu chí: không có lifecycle/reward nào được tạo từ flat snapshot không có lệnh.

### P2 — Cây context và khóa chuyển context

Cây context gốc:

```text
VÀO_LỆNH_KHÔNG_CÓ_LỆNH
QUẢN_LÝ_MUA
QUẢN_LÝ_BÁN
XỬ_LÝ_LỆNH_CHỜ
PHÒNG_THỦ_TÀI_KHOẢN
```

Context con chỉ xuất hiện khi có đủ bằng chứng lifecycle xác nhận và khác biệt kết quả so với parent.

- Context đổi trực tiếp theo nến M15 đã đóng; M1 không được dùng để đếm xác nhận hay đổi owner.
- Chuyển context chỉ theo nhiệm vụ hợp lệ: sau entry không quay về context entry; chuyển sang context quản lý vị thế.
- Context mới bắt đầu là `MẦM` hoặc `QUAN_SÁT_SONG_SONG`, chưa có quyền ra lệnh.
- Không tạo context mới từ một tick lạ, `FLAT_IDLE` hay lỗi bridge.

### P3 — Kho policy độc lập

Mỗi context owner chứa riêng:

```text
policy/graph neuron
memory
expectancy ledger
causal ledger
learner recovery
replacement learning
plasticity/growth state
fitness local
checkpoint local
lịch sử lifecycle.
```

Chỉ chia sẻ:

```text
feature contract
MT5 bridge
execution ledger gốc
risk governor
định nghĩa context
hạ tầng quan sát.
```

Không chia sẻ:

```text
weight
edge
memory
fitness
reward/loss
candidate growth
recovery evidence
replacement evidence.
```

### P4 — Huấn luyện theo lifecycle owner

- Khi policy tạo trace, hệ lưu `trace_id → expert owner → context`.
- Khi MT5 đóng lifecycle, chỉ policy owner nhận `observe_transition`, `develop` và lifecycle credit.
- Reward/loss không broadcast sang expert khác.
- Fitness được tính local theo context, không dùng một `winner.pkl` toàn hệ để thay expert.
- Policy candidate được kiểm tra trong chính vùng context của nó.

### P5 — Ươm, tách, thăng cấp, ngủ đông

Vòng đời:

```text
MẦM
→ QUAN_SÁT_SONG_SONG
→ THĂM_DÒ_RỦI_RO_THẤP
→ HOẠT_ĐỘNG_HẠN_CHẾ
→ TRƯỞNG_THÀNH
→ NGỦ_ĐÔNG
→ LƯU_TRỮ
```

- Tách context chỉ tạo candidate; không tự trao quyền ra lệnh.
- Thăng cấp cần replay, kiểm tra ngoài mẫu, rủi ro hợp lệ và đủ lifecycle thực.
- Context con chưa trưởng thành vẫn để parent owner xử lý.
- Hợp nhất chỉ được thực hiện khi overlap context cao và kiểm tra ngoài mẫu chứng minh không cần tách.

### P6 — MT5 production gate

Trước live/promotion phải đạt:

- EA đã compile từ source mới ra EX5 mới.
- Build hash EA/Python được ghi và đối chiếu.
- Parity 119 feature pass.
- Action sidecar khớp sequence/event/trace với bridge CSV và ledger.
- No-op/context chưa ổn định không tạo lệnh.
- Mỗi lifecycle ledger có owner provenance.
- OOS theo thời gian pass trong từng context.

## 4. Chế độ huấn luyện mới

Không dùng `trial/round/winner` tổng quát cho profile này.

```text
Context xuất hiện
→ owner policy quyết định
→ MT5 xác nhận
→ lifecycle đóng
→ local memory/ledger/fitness cập nhật
→ candidate checkpoint riêng được kiểm tra
→ thay checkpoint của chính context đó nếu pass.
```

Lệnh khởi tạo:

```powershell
$env:PYTHONPATH="src"
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml khoi-tao-he-chuyen-gia --output outputs/he_chuyen_gia.pkl --seed 7
```

Lệnh huấn luyện MT5 context-owner:

```powershell
$env:PYTHONPATH="src"
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml huan-luyen-he-chuyen-gia-mt5 --output outputs/he_chuyen_gia.pkl --runner-dir mt5_v2_runner_current --from-date 2025.01.01 --backtest-days 31 --mt5-visual
```

## 5. Quy tắc an toàn bắt buộc

- `RiskGovernor` luôn đứng ngoài policy owner.
- Nếu governor từ chối action, không được route sang policy khác trong cùng request.
- Không chuyển đề xuất BUY bị từ chối thành SELL tự động.
- Context mới không có owner trưởng thành: quan sát hoặc no-op; không fallback về policy chung.
- Không reward policy khi EA không xác nhận action/lifecycle.
- Không dùng output shadow/replay làm reward mạnh hơn MT5 lifecycle thật.
