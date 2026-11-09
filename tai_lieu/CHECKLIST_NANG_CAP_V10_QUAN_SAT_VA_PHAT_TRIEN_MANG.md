# Checklist kỹ thuật V10 — quan sát và phát triển mạng

## P0 — Hợp đồng action và context-owner

- [x] Không serialize `hanh_dong_policy`, `hanh_dong_hop_le`, `policy_action_index`, `legal_action_index` trong audit mới.
- [x] Context không ổn định ghi `CONTEXT_CHUA_ON_DINH` và `KHÔNG_GỬI_LENH`.
- [x] Không render neuron/edge `0/0` như một policy đã quyết định.
- [x] Action semantic/wire vẫn là nguồn action duy nhất trong hồ sơ.
- [ ] Xác minh EA MQL5 reject payload thiếu `context_id/expert_id/policy_id` trên Strategy Tester thật.

## P0 — Cách ly policy owner

- [x] Một context-owner gọi đúng một policy.
- [x] Mỗi policy có `policy_instance_uuid`, `graph_instance_uuid`, `memory_instance_uuid`, `memory_object_identity`, `weight_hash`, `memory_hash`, `policy_checkpoint_hash`.
- [x] `root_run_id` và `expert_run_id` được tách riêng.
- [x] Reward/lifecycle/growth controller nằm trong owner policy.
- [ ] Chạy smoke MT5 cho 5 expert gốc và kiểm tra tất cả identity trên execution ledger thật.

## P1 — Growth review ở mọi quyết định

- [x] Ghi novelty, conflict, prediction error chuẩn hóa, memory mismatch.
- [x] Ghi evidence lifecycle, causal evidence, replay score và module need score.
- [x] Ghi `neuron_growth_score`, `edge_growth_score`, ngưỡng, kết quả và lý do.
- [x] Decision không có lifecycle chỉ được `CHO_LIFECYCLE_XAC_NHAN`.
- [ ] Bổ sung replay score thật sau khi causal replay run-scoped có execution outcome.

## P1 — Mutation cấu trúc thật

- [x] Mutation chỉ gọi sau lifecycle owner xác nhận.
- [x] Capture trace từ execution cache trước khi policy settle/pop trace.
- [x] Event tạo candidate ghi before/after neuron/edge và ID candidate.
- [x] Candidate theo lifecycle `CANDIDATE → THU_NGHIEM → GIU_LAI | LAM_MO`.
- [x] Credit attribution rỗng không bị gắn nhãn growth.
- [ ] Bổ sung cắt vật lý `CAT_BO` sau nhiều lifecycle fading và validation replay.

## P2 — Telemetry và lưu trữ run

- [x] `scope_paths_by_run: true` tách JSONL theo root run.
- [x] `active_run.json` trỏ run hiện tại.
- [x] Snapshot topology đầy đủ chỉ ghi khi `topology_id` thay đổi.
- [x] Decision thường ghi activity/delta/growth review/topology reference.
- [x] Không áp quota số node, edge, attribution, event hay history.
- [ ] Kiểm tra tốc độ I/O/heap với run dài >= 1 tháng và graph lớn.

## P2 — UI theo dõi phát triển mạng

- [x] Dashboard summary: snapshot, candidate, keep/fade/prune, owner, checkpoint, topology, activity.
- [x] Timeline không giới hạn, mới nhất trước.
- [x] Chọn một mốc để xem full owner, growth review, state delta, event, activity, contract và topology/reference.
- [x] UI tự theo `active_run.json`.
- [x] 3D vẫn render toàn bộ topology không quota.
- [ ] Thêm export một snapshot/chuỗi event sang HTML hoặc CSV nếu cần chia sẻ audit ngoài UI.

## P3 — Gate an toàn học và trade

- [ ] Cấm entry thật khi `trust=0` hoặc `entry_evidence_verified=0`.
- [ ] Cấm entry khi conflict/prediction error vượt ngưỡng config.
- [ ] Không update policy khi thiếu `ticket → fill → close → realized PnL`.
- [ ] Gắn mọi run chưa có execution ledger là `QUAN_SAT_KHONG_HOC`.

## P4 — Kiểm thử bắt buộc

- [x] Python compile.
- [x] Java UI compile/JAR build.
- [x] Test context no-policy không giả topology.
- [x] Test lifecycle tạo candidate topology thật.
- [x] Test memory instance owner tách biệt.
- [x] Test run-scoped active pointer.
- [x] Test full topology chỉ lặp khi topology đổi.
- [ ] Compile EA MQL5.
- [ ] MT5 Strategy Tester smoke.
- [ ] Feature parity 119 input Python ↔ EA.
- [ ] OOS/walk-forward theo context-owner.
