# Checklist kỹ thuật đầy đủ

Ký hiệu: `[x]` đã thay đổi trong bản nâng cấp; `[ ]` cần xác minh/hoàn tất tại máy Windows có MT5.

## A. Kiến trúc policy/context

- [x] `CauHinhHeChuyenGia` buộc `mot_context_mot_chuyen_gia`.
- [x] Cấm `policy_chung_ra_lenh`.
- [x] Cấm `tron_policy`.
- [x] Buộc `so_chuyen_gia_moi_context = 1`.
- [x] Tạo 5 context gốc theo nhiệm vụ, không theo chiến lược BUY/SELL viết sẵn.
- [x] Mỗi root tạo instance graph policy độc lập với seed riêng.
- [x] Context owner có graph, memory, reward history và fitness local riêng.
- [x] Router/registry chỉ chọn owner; không có ensemble/bỏ phiếu.
- [x] Không context ổn định thì no-op transport có cờ không học.
- [x] Context con chưa active không cướp ownership của context cha.
- [x] Context child owner chỉ được route khi có quyền giao dịch.
- [x] Có nursery sinh candidate context từ lifecycle thật; candidate không auto-trade.
- [ ] Bổ sung gate replay/OOS độc lập để tự chuyển `QUAN_SÁT_SONG_SONG → THĂM_DÒ_RỦI_RO_THẤP`.
- [ ] Bổ sung risk cap per-expert ở giai đoạn thăm dò từ config/EA, không chỉ risk governor tổng.
- [ ] Thêm merge context sau OOS và kiểm tra non-interference.

## B. Context recognition

- [x] `ContextFrame`/119 input được dùng làm state cho recognizer.
- [x] Tách nhiệm vụ entry, management BUY, management SELL, pending, phòng thủ.
- [x] Router chỉ dùng nến M15 đã đóng và cache state M15 giữa các request M1; không còn countdown xác nhận context.
- [x] Context không ổn định không gọi policy.
- [x] Khi có vị thế, router chuyển sang context quản lý vị thế.
- [ ] Thay heuristic context hiện tại bằng encoder học được nhưng không được phát action.
- [ ] Kiểm thử switching khi vị thế thay đổi, pending khớp/hủy, drawdown cao, spread bất thường.
- [ ] Tạo dashboard phân bố context và tỷ lệ no-owner/no-op.

## C. Integrity Python/MT5

- [x] Python provenance sidecar ghi `context/expert/policy/phiên bản` theo action CSV.
- [x] Journal MT5 ghi `context_owner` và `context_audit`.
- [x] Python protocol có `CONTEXT_EXPERT_PROVENANCE_SCHEMA_VERSION = 1`.
- [x] EA source nâng bridge/ledger version lên 7 và tăng ActionWaitMs cap.
- [x] EA source có PendingExpiryBars và entry safety gate.
- [ ] Compile `.mq5` thành `.ex5` mới bằng MetaEditor trên Windows.
- [ ] Ghi hash source EA, hash EX5 và hash Python vào ledger/session manifest.
- [ ] EA đọc/ghi `context/expert/policy` trực tiếp trong ledger V8; hiện thời dùng sidecar Python để không phá parser V7.
- [ ] Fail-closed khi EA build/schema/hash không khớp.
- [ ] Xác minh 119 feature parity bằng `FeatureAuditFile` trên MT5 thật.
- [ ] Đối chiếu action CSV, metadata JSONL, execution ledger theo `event_id + sequence + trace_id`.

## D. Firebreak và feedback

- [x] `FLAT_IDLE` bị chặn trước shadow learning.
- [x] `FLAT_IDLE` bị chặn trước recovery/memory/lifecycle learning.
- [x] Context chưa ổn định tạo no-op có `ly_do_khong_hoc`.
- [x] Reward/loss local chỉ route về trace owner.
- [x] Lifecycle credit local chỉ route về trace owner.
- [ ] Thêm test integration thật: bridge timeout, EA reject, missing ledger, stale file.
- [ ] Bắt buộc `EA_ACK`, `order accepted/filled`, `final close` trước local credit mạnh.

## E. Cấu hình và CLI

- [x] Tạo `configs/he_chuyen_gia_mot_context_mot_policy.yaml`.
- [x] Khóa lệnh `continuous-train`, `train-mt5`, `improve`, `train-to-goal` khi profile expert active để tránh quay lại policy chung.
- [x] Thêm `khoi-tao-he-chuyen-gia`.
- [x] Thêm `huan-luyen-he-chuyen-gia-mt5`.
- [x] Artifact `pickle` giữ nhiều policy; registry JSON cho audit.
- [x] Sau load, factory được gắn lại để có thể tạo context con tiếp.
- [ ] Tạo lệnh promotion per-context với frozen OOS riêng.
- [ ] Tạo command archive/dormant/merge expert có audit log.

## F. Tests

- [x] Test 5 policy root có object id độc lập.
- [x] Test mỗi request chỉ một policy gọi `act`.
- [x] Test outcome chỉ update expert owner.
- [x] Test child chưa active không lấy ownership.
- [x] Test position state chọn đúng một management owner.
- [x] Test config reject global policy/blending.
- [x] Test context chưa ổn định không gọi policy.
- [x] Test provenance context owner trên trace.
- [x] Test `FLAT_IDLE` firebreak và MT5 static contract.
- [ ] Test MT5 real-tick E2E 1 context → 1 policy → 1 lifecycle.
- [ ] Test không có cross-expert mutation sau 1.000 lifecycle.
- [ ] Test power-loss/restart: artifact load không đổi owner mapping.
- [ ] Test no-op transport không tạo bias HOLD trong learner.

## G. UI và vận hành

- [x] Action sidecar/journal dùng khóa tiếng Việt: `ma_context`, `ma_chuyen_gia`, `ma_policy`.
- [ ] UI Java đọc sidecar provenance và hiển thị context owner hiện hành.
- [ ] UI hiển thị context chưa ổn định, no-owner, EA reject, bridge timeout riêng biệt.
- [ ] UI hiển thị lifecycle local: reward, fitness, số mẫu, trạng thái expert.
- [ ] UI không được phép sửa weight/policy/risk/context qua panel quan sát.

## H. Gate trước khi live

- [ ] EX5 mới compile, không còn file binary cũ.
- [ ] Feature parity pass với ít nhất một đoạn MT5 real tick.
- [ ] Ledger/report reconciliation pass.
- [ ] Không có `FLAT_IDLE` trong learner journal.
- [ ] Mọi lifecycle có context-owner provenance.
- [ ] Không policy nào vượt risk governor.
- [ ] Mỗi context active có sample/LCB/DD local đạt ngưỡng.
- [ ] OOS pass theo thời gian cho ít nhất ba cửa sổ context phù hợp.
- [ ] Rollback artifact/context registry đã được lưu.
