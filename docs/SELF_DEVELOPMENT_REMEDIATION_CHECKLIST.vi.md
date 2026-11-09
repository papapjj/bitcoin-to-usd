# Checklist khắc phục cơ chế tự phát triển — bản đã triển khai

**Phạm vi:** policy graph-native, memory substrate thuần neural, mistake replay, replacement action/parameter, candidate promotion và MT5-selection.

## P0 — Luồng train, migration và kiểm thử

- [x] **Khôi phục API `ensure_compatible(target_input_size=...)`.** Wrapper M1 V11 giờ nhận và chuyển tiếp `target_input_size`; candidate generation, load checkpoint cũ, schema migration và transition feedback không còn nổ `TypeError`.
- [x] **Cô lập test suite chuẩn.** Thêm `pytest.ini`: chỉ thu thập `tests/`, dùng `src/`, loại `work/`, build/cache và runner snapshot cũ để stale imports không che lỗi lõi.
- [x] **Chuẩn hóa nguồn EA MT5.** Đưa `SelfGrowing_MT5_Trainer_EA.mq5` về root dự án từ bản runner hiện hành để test, đóng gói và deployment dùng cùng nguồn canonical.
- [x] **Regression test cho mọi lỗi P0.** Toàn bộ test hiện chạy qua collection mặc định; có test cho compatibility wrapper, graph-native adapter và replacement flow.

## P1 — Không còn “action sai thì giảm action”

- [x] **Mistake replay giữ action cũ.** Repeated mistake không trừ bias action cũ; nó tạo/ưu tiên action thay thế theo cause. Test đã đổi để khóa đúng contract này.
- [x] **Rejection feedback không phá baseline graph.** Reject ghi evidence, starvation, risk/reasoning context và replay episode; không hạ gate/weight/utility của đường graph baseline trực tiếp.
- [x] **Replacement action được phân biệt với suppression.** Runtime v8 chỉ tổng hợp proposal action dương theo context; negative outcome vẫn được giữ cho critic/cause analysis, không trở thành negative action bias.

## P1 — Replacement parameter: SL/TP/risk/trailing/partial-close

- [x] **Scope bắt buộc theo action.** Mỗi proposal parameter có `action_index`, action family, side, lifecycle, position state, target slot, management mode. BUY không thể rò SL/TP sang SELL, exit hay management.
- [x] **ID bền qua restart.** Thay toàn bộ `hash()` dùng cho proposal/concept/causal/growth/imprint IDs bằng SHA-256 stable identifier.
- [x] **Shadow-first.** Một loss chỉ tạo hypothesis `shadow`, không tự động trở thành quy tắc global.
- [x] **Candidate-only probing.** Shadow parameter chỉ được blend nhẹ trong `graph_trial_mode`; baseline không dùng proposal chưa xác nhận.
- [x] **Paired validation.** Khi proposal được thực thi, outcome được so với `baseline_realized_r`; evidence tăng/giảm, confidence được tính từ số trial và direction.
- [x] **Promotion state.** Proposal chuyển `shadow → validated` chỉ khi đủ sample/confidence; thành `rejected` khi liên tiếp kém baseline. Baseline chỉ đọc proposal `validated`/`consolidated`.
- [x] **Parameter-head isolation.** Lifecycle credit không update global parameter head của baseline. Chỉ candidate được phép học parameter head trước khi qua selection.
- [x] **Regression tests.** Test khẳng định scope BUY/Sell không lẫn, shadow không rò baseline, validation đủ sample mới được reuse, và candidate-only head update.

## P1 — Memory, nhân quả và ngữ cảnh

- [x] **Không xóa semantic/causal migration view.** Typed collections cũ được giữ read-only để migration idempotent; query/develop thực tế vẫn chỉ dùng `neural_cells`/`neural_synapses`.
- [x] **Pure-neural causal signals.** `causal_plasticity_signals()` giờ đọc `causal_outcome` cell thay vì legacy `causal_nodes`; causal plasticity có input thật sau v6/v7/v8 migration.
- [x] **Position lifecycle trong neural causal cell.** Causal outcome có near-TP, giveback, partial/trailing recommendation, neural lifecycle probabilities, target ticket/slot.
- [x] **Account context có thể audit.** Causal outcome giữ memory-key schema, account context trước/sau để replay theo equity/drawdown/position state không mất thông tin.
- [x] **Diagnostic compatibility views.** `connection_kinds` và `causal_kinds` là alias đọc từ neural substrate, không phục hồi symbolic runtime song song.

## P1 — Growth, survival cost và pruning

- [x] **Complexity cost toàn diện.** `structural_complexity()` tính module, hidden unit, dense edge, graph node/edge, disabled edge, memory cell/synapse và structural churn.
- [x] **Selection dùng complexity thật.** `validation_selection_score()` và MT5 evaluator dùng topology-aware cost, không còn chỉ đếm module.
- [x] **Không áp hard cap kiến trúc.** Open-ended growth vẫn được giữ; cấu trúc phải trả chi phí survival thay vì bị cắt bằng limit cố định.
- [x] **Compaction hậu promotion.** Chỉ candidate đã promote mới compact: disabled/pruning edge đủ tuổi, neuron pruning bị cô lập, và parameter proposal bị reject sau đủ lịch sử. Baseline call là no-op.
- [x] **Invariant check sau compact.** Graph kiểm tra I/O và topology invariant sau mỗi compaction.
- [x] **Regression tests.** Test xác nhận baseline không bị compact, candidate compact an toàn và graph vẫn invariant.

## P2 — Dễ bảo trì và quan sát

- [x] **Tách primitive dùng chung.** `replacement_learning.py` chứa stable identifier, action scope, scope matching và proposal confidence; tránh lặp logic ID/scope trong runner.
- [x] **Event journal có state.** Proposal credit/validation/compaction ghi event với ID, scope, state, sample/confidence và delta-R để audit.
- [x] **Test contract graph-only.** Adapter test không còn đòi backend `legacy`; nguồn hiện tại chỉ hỗ trợ graph-native.
- [x] **Test contract pure-neural.** Test memory không còn kiểm tra symbolic containers bị retire; kiểm tra trực tiếp cell role/synapse và metadata.

## Cổng nghiệm thu đã chạy

```text
python -m compileall -q src
pytest

251 passed
```

## Ràng buộc còn lại trước khi đưa tiền thật vào MT5

- [ ] **Chạy MT5 Strategy Tester/OOS thực tế.** Môi trường kiểm tra hiện tại không có MetaTrader 5 terminal/agent, nên không thể xác nhận slippage, commission, fill policy và EA bridge trong terminal thật.
- [ ] **Walk-forward nhiều regime.** Chỉ promote policy khi candidate thắng baseline ở in-sample, validation và OOS theo regime; không dùng một backtest window làm bằng chứng đủ mạnh.
- [ ] **Giới hạn rủi ro tài khoản production.** Cần giữ hard risk guard ngoài policy: max daily loss, max concurrent risk, kill switch, reject duplicate/execution mismatch. Đây là tầng an toàn broker/account, không nên giao hoàn toàn cho self-growing graph.

