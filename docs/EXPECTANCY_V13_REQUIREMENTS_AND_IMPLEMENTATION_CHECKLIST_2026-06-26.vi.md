# Checklist nâng cấp V13 — Expectancy ròng theo Context × Action

## Mục tiêu vận hành

V13 thay cơ chế đánh giá “trial có PnL/fitness tốt” bằng câu hỏi chính xác hơn:

> Trong **context đang gặp**, action nào có **expectancy ròng theo R dương, đủ mẫu, bảo thủ và do policy thực sự tạo ra**?

`EXP` trong tài liệu này là expectancy của **lifecycle vị thế đã đóng cuối cùng**, không phải reward một tick, không phải lệnh đã đặt, không phải partial-close count, và không phải PnL thô theo lot.

> **Migration runtime hiện tại:** ledger expectancy đã lên schema V14. Khi load artifact có schema cũ, chỉ các record expectancy/derived evidence không còn đáng tin mới bị reset; policy core và `ExperienceSuperNetwork` vẫn được giữ. `memory_extrema` tiếp tục là derived audit, không được tính vào lifecycle fitness.

---

## Công thức chuẩn

### R ròng của một lifecycle

```text
net_R = final_net_pnl / initial_risk
final_net_pnl = gross_profit + commission + swap
```

`final_net_pnl` lấy từ execution lifecycle ledger sau fill/close thực tế. Slippage đã phản ánh trong giá fill và do đó nằm trong PnL thực tế; không cộng giả định một lần nữa.

### Expectancy bảo thủ

Với `n` lifecycle cùng context và cùng action scope:

```text
mean_shrunk = n / (n + 4) × mean(net_R)
posterior_variance = pooled_sample_variance_with_1R_prior
EXP_LCB = mean_shrunk - 1.645 × sqrt(posterior_variance / n)
```

- Prior variance 1R ngăn một cụm vài lệnh thắng có variance nhỏ giả tạo thành edge chắc chắn.
- Chỉ `EXP_LCB > 0` mới tạo action bias cho M1.
- `EXP_LCB < 0` đủ mẫu chỉ đi vào critic/audit; **không** trừ thẳng bias action cũ để tránh HOLD collapse.

---

# A. Checklist yêu cầu nghiệp vụ

| ID | Yêu cầu | Trạng thái | Bằng chứng triển khai |
|---|---|---:|---|
| EXP-R01 | Dùng net-R của lifecycle cuối cùng, không dùng trade count hoặc entry intent | Hoàn thành | `mt5_runner.py` lifecycle `realized_r` → `apply_execution_lifecycle_credit()` |
| EXP-R02 | Chỉ policy-origin được coi là kỹ năng có thể bias quyết định | Hoàn thành | `ContextualExpectancyLedger.query(policy_only=True)` |
| EXP-R03 | Bootstrap/mask/risk override vẫn lưu audit nhưng không tạo edge policy | Hoàn thành | `POLICY_ORIGINS={policy}`; origin khác chỉ trong audit |
| EXP-R04 | BUY/SELL/close/manage không được dùng chung evidence | Hoàn thành | `replacement_learning.action_scope()` + `scope_matches()` |
| EXP-R05 | Context retrieval phải dùng canonical ContextFrame, không trộn latent M1 state | Hoàn thành | `frame.canonical_memory_key`, `blockwise_cosine()` |
| EXP-R06 | Mẫu nhỏ không được coi là edge | Hoàn thành | shrink-to-zero + LCB + `min_samples` |
| EXP-R07 | Decision-time policy phải nhận EXP trước selector | Hoàn thành | V13 memory query → action bias, M1 selector/critic projection |
| EXP-R08 | M1 chỉ thấy action hợp lệ khi so EXP | Hoàn thành từ V12, được giữ | `allowed_action_mask` đưa vào ledger query và graph |
| EXP-R09 | Candidate không được promote nhờ bootstrap/global PnL | Hoàn thành | policy-origin LCB, ratio và sample hard gates |
| EXP-R10 | Growth chỉ dựa trên regret đã được lifecycle xác nhận | Hoàn thành | `context_expectancy_regret` event |
| EXP-R11 | Có report audit theo origin/context scope | Hoàn thành | `contextual_expectancy_report()` và metric flatten |
| EXP-R12 | Có back-compat cho checkpoint/config cũ | Hoàn thành | `ensure_compatible`, legacy config fallback |
| EXP-R13 | Xác nhận performance bằng MT5 real tick/OOS | Chưa xác minh | Cần chạy Strategy Tester/OOS bên ngoài môi trường này |

---

# B. Checklist kỹ thuật chi tiết

## EXP-T01 — Ledger persistent theo context × action scope

**Yêu cầu**

- Lưu record `context_prototype`, `scope`, `values`, `origins`, `trace_ids`, `frame_ids`, lifecycle tags.
- Identity không dùng Python `hash()` không ổn định qua restart.
- Context gần được merge theo similarity thay vì tick tạo record mới vô hạn.

**Triển khai**

- File: `src/self_growing_trader/contextual_expectancy.py`
- Class: `ContextualExpectancyLedger`, `ContextActionExpectancyRecord`
- Stable record id: `stable_identifier(...)`
- Merge: `blockwise_cosine >= merge_similarity` (mặc định 0.90)
- Giới hạn evidence mỗi record: 256 lifecycle mặc định, có thể cấu hình.

**Acceptance test**

- Context giống gần + cùng scope → cùng record.
- Context khác hoặc action/side/lifecycle khác → record khác.
- Restart/copy policy → record identity và values giữ nguyên.

## EXP-T02 — Scope không rò BUY ↔ SELL ↔ lifecycle management

**Scope key**

```text
action_index
action_family
side
lifecycle
position_state
target_slot
management_mode
```

**Quy tắc**

- BUY entry không thể bias SELL entry.
- Flat entry không thể dùng evidence của `position_state=open` management.
- Action không hợp lệ theo mask không có action bias.

**Acceptance test**

- 8 BUY policy +R không làm SELL LCB dương.
- 8 SELL legacy-bootstrap +R không tạo SELL policy bias.

## EXP-T03 — Robust estimate chống low-sample winner

**Quy tắc**

- Min sample mặc định: 6 lifecycle policy-origin trong một scope.
- Prior variance floor: 1R.
- Confidence gồm coverage sample và policy-origin ratio.

**Acceptance test**

- Một lifecycle +3R: mean dương nhưng `EXP_LCB < 0`, `eligible=false`.
- Sáu lifecycle +R ổn định: LCB dương mới có thể tạo bias.

## EXP-T04 — Truy hồi canonical context

**Quy tắc**

- Query dùng `ContextFrame.canonical_memory_key` 46 chiều.
- Similarity dùng blockwise market/account/position semantics của V12.
- Latent `encoded_state` của M1 không được dùng làm memory coordinate.

**Acceptance test**

- Raw/canonical context và latent state cùng index không được trộn trong record/retrieval.
- `frame_id` xuất hiện trong audit lifecycle.

## EXP-T05 — M1 nhận positive expectancy trước selector

**Luồng**

```text
ContextFrame
→ ExpectancyLedger.query
→ expectancy_action_lcb / confidence / positive action bias
→ M1 critic region + selector scores
→ legal action candidate
```

**Quy tắc**

- Chỉ `max(0, LCB)` tạo action bias.
- Negative LCB đi vào `expectancy_critic_bias` và audit, không thành global suppression.
- `market_opportunity` chỉ tăng từ positive legal entry LCB × confidence.

## EXP-T06 — Lifecycle credit và attribution

**Luồng**

```text
MT5 final close
→ parse lifecycle realized_r
→ trace_id provenance
→ capture immutable trace before cache cleanup
→ ContextualExpectancyLedger.observe(...)
```

**Quy tắc**

- Chỉ final close được credit.
- `trace_id + action + close metadata + realized_r` tạo idempotency identity.
- Duplicate close event không ghi double sample.

## EXP-T07 — Regret phục vụ growth, không phục vụ action suppression

```text
exp_regret = max(0, best_legal_LCB - selected_action_LCB)
```

Growth event chỉ được ghi khi:

```text
origin = policy
realized_R < 0
predicted regret >= 0.10R
```

`growth_events.context_expectancy_regret` chỉ là evidence cho cơ chế growth hiện có; specialist không được mọc chỉ từ một loss đơn lẻ. Nó còn phải qua repeated-context/utility/grace-period controls của graph V12.

## EXP-T08 — Promotion objective

Candidate V13 chỉ được promote khi, ngoài toàn bộ execution/reconciliation gate cũ, đồng thời đạt:

```text
policy-origin samples >= max(3, ceil(min_policy_origin_ratio × required_lifecycles))
policy-origin lifecycle ratio >= configured threshold (mặc định 0.60)
policy-origin EXP_LCB > 0
positive eligible context-action scope >= 1
negative critical context-action scopes không tăng so baseline
candidate policy EXP_LCB > baseline policy EXP_LCB + configured delta (mặc định 0.01R)
```

Global PnL, bootstrap PnL hay fitness scalar không thể vượt qua gate này một mình.

## EXP-T09 — MT5 fitness

Khi evaluator có audit V13:

```text
primary expectancy = policy-origin EXP_LCB
```

và objective vẫn phạt drawdown, CVaR, immediate exit, execution failure, net cost, turnover, complexity, coverage/risk integrity. Legacy/report-only run vẫn dùng fallback cũ để tương thích đọc artifact, nhưng không có bằng chứng V13 thì không được hiểu nhầm là policy EXP đã xác nhận.

## EXP-T10 — Cấu hình

`configs/self_growing.yaml`:

```yaml
expectancy:
  enabled: true
  min_samples: 6
  merge_similarity: 0.90
  max_samples_per_context: 256
  min_policy_origin_ratio: 0.60
  promotion_lcb_min_delta: 0.01
```

- Cấu hình được áp vào seed policy trước journal replay và được copy sang candidate.
- Workflow/test shim thiếu block `expectancy` dùng default V13, không lỗi.

---

# C. Chỉ số cần theo dõi trong mỗi trial

Bắt buộc ghi/đọc:

```text
fitness_policy_expectancy_lcb
fitness_policy_expectancy_mean
fitness_policy_origin_lifecycle_count
fitness_all_origin_lifecycle_count
fitness_policy_origin_ratio
fitness_context_positive_eligible
fitness_context_negative_critical
fitness_all_expectancy_lcb
```

Ngoài ra, từ `contextual_expectancy_report()`:

```text
origin_counts
derived_evidence_count
derived_evidence_origins
scope_preview
record_count
policy_expectancy
all_expectancy
```

Cách đọc đúng:

- `all_expectancy` tốt nhưng `policy_expectancy` yếu: bridge/bootstrap đang tạo phần lớn kết quả, policy chưa học xong.
- `policy_expectancy` dương nhưng `negative_critical_scope_count` tăng: policy có lợi thế tổng nhưng phá một context/action nguy hiểm; không promote.
- `memory_extrema` là derived evidence cho timing/TP, chỉ xuất hiện trong `origin_counts` và không được tính vào `all_origin_lifecycle_count` hoặc policy fitness.
- Policy sample thấp: thiếu bằng chứng, không suy ra edge dù PnL dương.
- LCB dương ở BUY không là lý do để SELL thay đổi; scope separation là cố ý.

---

# D. Quy trình chạy sau nâng cấp

1. Dùng checkpoint cũ: V13 tự tạo ledger rỗng; không resize latent-memory cũ sang EXP ledger.
2. Rebuild expectancy evidence từ lifecycle journal chỉ khi journal có raw canonical context, action provenance, initial risk và final net-R đáng tin.
3. Chạy MT5 M1 OHLC để sàng nhanh.
4. Với candidate qua gate, chạy Every Tick 1-minute / real-tick khi broker có dữ liệu.
5. Chạy walk-forward, chọn policy theo OOS policy-origin `EXP_LCB`, không chỉ fitness train cuối.
6. Kiểm tra `policy-origin ratio`, scope preview và reject reasons trước khi thay seed.
7. Không chạy tiền thật chỉ dựa vào unit test hoặc simulator.

---

# E. Hạn chế còn lại cần kiểm chứng thực tế

- Strategy Tester chưa được chạy trong môi trường này.
- Chưa xác minh mapping `action_origin` của EA trên mọi fill/requote/reconnect.
- Chưa xác minh slippage/tick ordering real-tick, spread spike, partial close, gap và broker stop-level thực tế.
- Chưa có OOS context-stratified statistical CI đầy đủ giữa baseline/candidate; V13 hard gate đã dùng LCB và delta nhưng paired bootstrap CI theo regime vẫn là nâng cấp tiếp theo.
