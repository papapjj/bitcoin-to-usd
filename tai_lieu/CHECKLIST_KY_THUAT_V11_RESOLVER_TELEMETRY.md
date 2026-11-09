# Checklist kỹ thuật V11 — resolver telemetry, UI và phát triển mạng

## P0 — Không mở nhầm UI cũ

- [x] Resolver V11 được giữ trong launcher V190 `SelfGrowingTraderControlV190.jar`.
- [x] Build script xóa class cache theo tên JAR trước khi compile.
- [x] Launcher V10 và V10.2 chuyển hướng sang V11.
- [x] UI chính/Live 3D có title V11 để kiểm tra phiên bản trực quan.
- [x] Command preview gọi `chay_cli_context_expert_v11.py`.

## P0 — Resolver run-scoped

- [x] Ưu tiên đọc `outputs/active_run.json`.
- [x] Hỗ trợ `active_run.json` tại workspace root.
- [x] Resolve `event_path`, `decision_profile_path`, `structural_event_path`, `network_development_path` theo workspace root.
- [x] Chuẩn hóa dấu `\` và `/` trong manifest Windows/Linux.
- [x] Fallback quét `outputs/runs/*` và lấy telemetry mới nhất nếu manifest thiếu.
- [x] Hiển thị root run/path thực tế trong trạng thái Live 3D.

## P0 — Không treo UI với JSONL lớn

- [x] Không có quota dữ liệu tổng, lịch sử hoặc topology.
- [x] Ingest đọc dần 8 MiB mỗi poll thay vì parse toàn bộ GB ở poll đầu.
- [x] Offset riêng cho telemetry, decision profile, structural event và network development.
- [x] Auto-expand chunk khi record JSONL lớn chưa có newline.
- [x] Snapshot topology hiện hành giữ đầy đủ; history giữ metadata/index gọn, không nhân bản toàn topology trong RAM.

## P0 — Telemetry trung thực

- [x] Không phát `hanh_dong_policy = -1`, `hanh_dong_hop_le = -1`, `policy_action_index` hoặc `legal_action_index` ra UI/audit.
- [x] Context chưa ổn định được ghi `CONTEXT_CHUA_ON_DINH`, không giả là policy `0 node/0 edge`.
- [x] Đường telemetry owner được facade phát sau khi có context-owner/action contract/growth review.
- [x] Mỗi record owner giữ root run id, expert run id, context/expert/policy/graph/weight identity.

## P1 — Không lặp topology vô ích

- [x] Full topology chỉ ghi baseline đầu của mỗi policy instance + topology id.
- [x] Decision/feedback/lifecycle tiếp theo ghi `topology_tham_chieu`, activity và delta.
- [x] UI nối baseline + delta để vẫn xem toàn bộ mạng không mất dữ liệu.

## P1 — Quan sát growth thật

- [x] Growth review gồm novelty, conflict, prediction error, memory mismatch, evidence, threshold và lý do từ chối.
- [x] Structural journal chỉ ghi topology thật: candidate, thử nghiệm, giữ lại, làm mờ, cắt.
- [x] Không gọi credit attribution là “tạo neuron/liên kết” khi node/edge count không đổi.

## Bài kiểm tra chấp nhận

- [ ] Chạy `KIEM_TRA_TELEMETRY_V11.cmd` và tất cả path run-scoped tồn tại.
- [ ] Live 3D tải topology baseline trong vài poll đầu, không hiển thị 0/0 khi JSONL có baseline.
- [ ] Status Live 3D hiển thị `outputs\runs\<root_run_id>\...`.
- [ ] Khi trainer tăng node/edge thật, journal ghi before/after và UI đánh dấu delta/growth.
- [ ] Không học/promote khi thiếu execution ledger + lifecycle MT5 xác nhận.
