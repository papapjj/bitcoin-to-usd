# Checklist Fitness MT5 v5 & Neuron Fitness Attribution

**Ngày cập nhật:** 23-06-2026  
**Patch:** `P0 Fitness/Neuron v5`  
**Trạng thái:** code + Python regression hoàn thành; MT5 compile/Strategy Tester runtime **chưa xác minh** vì bundle vẫn thiếu `compute_features.mqh` và portable runner.

## Mục tiêu

Loại bỏ việc policy được thưởng chỉ vì report HTML đẹp hoặc vì một neuron đang active đúng lúc trial toàn cục có score tốt. Fitness v5 chia rõ ba tầng:

```text
execution ledger (order/deal/position) → final position lifecycle → eligibility + rank score
trace_id + action margin + realized R    → neuron-specific credit → freeze/prune/growth
```

## Contract v5 đã triển khai

| Hạng mục | Trạng thái | Bằng chứng code/test |
|---|---|---|
| Position lifecycle là sample thống kê | [x] | `parse_mt5_execution_ledger()` gom partial close theo `position_ticket`; 8 deal đóng từng phần vẫn chỉ là 1 lifecycle. |
| R theo initial risk | [x] | Ledger schema 5 thêm `initial_risk`, `position_final_close`, `age_bars`; fallback legacy bị gắn `equity_r_proxy`, không giả mạo exact R. |
| Trace provenance | [x] | `MT5PositionLifecycle` giữ `trace_id`, `entry_sequence`; evaluator backfill outcome vào trace cache theo `trace_id`. |
| Eligibility tách rank fitness | [x] | `mt5_fitness_breakdown()` trả `eligible`, reasons, sample count, source, score và bounded learning signal. |
| Small-sample/PF=999 guard | [x] | Sample dưới ngưỡng không có positive rank; no-loss HTML không dùng `avg_win` tiền tệ làm R. |
| Churn/thoát sớm penalty | [x] | `immediate_exit_ratio` dựa trên `age_bars < MinHoldingBars`; có eligibility reason và penalty. |
| Execution quality penalty | [x] | Reject rate, trace coverage, initial-risk coverage, final-close completeness tạo `ledger_integrity`/failure rate. |
| Neuron fitness signed | [x] | Credit dùng `gate × hidden_i × (W_selected,i − W_runnerup,i)`, không dùng `abs(hidden)` làm credit chính. |
| Outcome cuối vị thế | [x] | `apply_execution_lifecycle_credit()` backfill realized R, close reason, duration vào trace entry. |
| Prune/freeze neuron có evidence | [x] | Cần effective samples, lower confidence bound utility, activation/redundancy; frozen neuron không nhận credit/prune. |
| MT5 runtime verification | [ ] | Cần MetaEditor compile và Strategy Tester matrix trên terminal thực. |

## Checklist P0 — cần đóng trước khi dùng score v5 để chọn winner MT5

### P0-F1. Build & wire contract

- [ ] Compile `SelfGrowing_MT5_Trainer_EA.mq5` với `compute_features.mqh` từ checkout sạch.
- [ ] Chạy Strategy Tester smoke test và xác nhận `self_growing_execution.csv` thật có header schema 5.
- [ ] Xác nhận một `MARKET_BUY` tạo chuỗi `accepted/filled → position_ticket → closed` với cùng `ack_sequence`, `event_id`, `trace_id`.
- [ ] Xác nhận lệnh pending fill sau nhiều bar vẫn giữ `trace_id` entry đúng.
- [ ] Xác nhận partial close nhiều deal, final close một deal, và `position_final_close=1` chỉ ở final deal.
- [ ] Xác nhận `initial_risk > 0` cho mỗi lifecycle đủ điều kiện, lấy từ SL thực tế và tick-value/contract size đúng broker.
- [ ] Xác nhận `age_bars` chính xác với M1/M5/period được test; không giả định 60 giây nếu period thay đổi.
- [ ] Lưu `fitness_schema_version=5`, `bridge_schema_version=4`, EA hash, terminal build, symbol, period vào result JSON.

### P0-F2. Validation of MT5 fitness

- [ ] Fixture Strategy Tester: một full close, nhiều partial closes, TP, SL, expert close, pending cancel, reject, modify SL/TP.
- [ ] Đối chiếu net PnL ledger = Account History: profit + commission + swap cho từng `position_ticket`.
- [ ] Kiểm tra `realized_r = net_pnl / initial_risk` trên 10 position có size/SL khác nhau.
- [ ] Cùng strategy, lot size tăng/giảm không làm structural score tăng chỉ vì currency PnL.
- [ ] Một thắng duy nhất/PF 999 không có score dương và không pass eligibility.
- [ ] Cùng PnL/PF: tập có immediate exit nhiều hơn phải score thấp hơn.
- [ ] Cùng mean R: tập có CVaR 5% xấu hơn phải score thấp hơn.
- [ ] Report HTML tốt nhưng ledger thiếu/no final lifecycle phải rank `-100` trong evaluator runtime.
- [ ] Cảnh báo nếu risk coverage < 90%, trace coverage thấp, integrity < 0.70, reject rate > 20%.

### P0-N1. Validation of neuron attribution

- [ ] Thêm snapshot test trace gồm hidden state, gate, selected/runner-up action và action logits cho từng module.
- [ ] Với synthetic trace, neuron làm tăng selected-action margin phải nhận credit cùng dấu với realized advantage.
- [ ] Neuron làm giảm selected-action margin phải nhận credit ngược dấu.
- [ ] Chỉ final close đã reconcile mới nhận final credit; calling backfill lần hai cho cùng `trace_id` phải bằng 0.
- [ ] Kiểm tra action pending không fill, request bị reject và `HOLD` true no-op không tự tạo entry-fitness.
- [ ] Verify trace cache eviction, process restart và late close: orphan lifecycle phải log rõ, không âm thầm mất/cấp sai credit.
- [ ] Chạy ablation: global fitness-only vs signed neuron attribution; so sánh churn, OOS score, neuron utility distribution.

## Checklist P1 — scientific validation và vận hành

### P1-F3. Promotion/OOS

- [ ] Promotion dùng `MT5FitnessBreakdown.eligible`, không chỉ scalar fitness.
- [ ] Frozen OOS cấm ghi trace cache, critic/predictor/memory/growth; checksum policy bất biến trước/sau.
- [ ] Adaptive OOS báo riêng số lifecycle credited, learning signal và uplift so với frozen OOS.
- [ ] Walk-forward mỗi window phải report sample count, R-LCB, PF-shrunk, CVaR, DD, immediate-exit, reject rate, integrity.
- [ ] Báo cáo nhiều seed/symbol/regime; không promote theo một best trial.

### P1-N2. Neuron lifecycle management

- [ ] Per-neuron uncertainty dùng variance thực/online Welford thay vì proxy `abs(advantage_ema)`.
- [ ] Chặn prune neuron duy nhất phục vụ risk gate/action safety path bằng dependency mask.
- [ ] Clone/expand chỉ khi high-utility neuron saturated và residual error theo regime còn cao.
- [ ] Freeze có expiry/revalidation; neuron tốt cũ không được bất tử khi regime đổi.
- [ ] Dashboard: utility LCB, samples, activation, signed contribution, redundancy, frozen/pruned count, trace attribution coverage.
- [ ] Bound total frozen/pruned neurons per window và shadow-validate structural change trước commit.

## Open risks còn lại

1. **MT5 runtime chưa được compile/test.** Code MQL v5 chỉ qua static contract và Python parser tests.
2. **Cross-process delayed reconciliation.** Trace cache là in-memory. Journal đã persist module action trace, nhưng chưa có durable `trace_id → lifecycle` reconciler sau Python restart.
3. **Initial risk semantics cần broker validation.** `initial_risk` phải tính chính xác với digit, tick size/value, contract size, SL và volume.
4. **Fitness still depends on report for drawdown/net-profit.** v5 dùng ledger cho position quality nhưng equity curve/DD chưa được rebuild hoàn toàn từ raw deal/equity snapshots.
5. **P0 execution contract khác vẫn mở.** EA vẫn cần đồng bộ risk sizing, pending offset, target slot và management semantics với Python decoder.

## Regression evidence trong bundle này

```text
pytest -q                              129 passed
python -m compileall -q src tests scripts   passed
```

Các test bổ sung kiểm tra: lifecycle aggregation, partial-close sample inflation, missing ledger hard reject, immediate-exit penalty, signed neuron action margin, final lifecycle backfill idempotent, và evidence-based neuron pruning.
