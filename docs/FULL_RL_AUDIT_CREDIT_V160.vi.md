# Full-RL audit credit V160

## Lỗi đã sửa

Trong chế độ `direct_rnn_full_rl_enabled`, canonical settlement từng bỏ cả
`ContextualExpectancyLedger` để bảo đảm không còn legacy writer. Việc đó đúng
về ranh giới cập nhật trọng số, nhưng sai về quan sát: lifecycle MT5 đã đóng
không còn tạo `policy_origin_lifecycle_count` hay
`conditioned_entry_expectancy`. Vì vậy telemetry có thể báo 0 mẫu điều kiện
dù lệnh thật đã đóng và AdamW đã nhận batch.

## Contract V160

Mỗi lifecycle hợp lệ của Full-RL giờ đi theo hai nhánh tách biệt:

```text
immutable MT5 final R
  ├─> Full-RL batch -> actor / critic / geometry / representation -> AdamW
  └─> full_rl_expectancy_audit + conditioned_entry_expectancy -> telemetry only
```

`full_rl_expectancy_audit` là một `ContextualExpectancyLedger` riêng trên
`ChuyenGiaPolicy`, không nằm trong `memory_supernet`. `act()`, `_query_memory`,
`expected_action_values()` và mọi action/mask/risk route chỉ đọc ledger quyết
định cũ trong `memory_supernet`, nên audit mới không thể đổi logit, action,
risk hoặc state recurrent.

## Codex teacher trong topology specialist

Profile candidate
`configs/he_chuyen_gia_reasoning_rl_v1_specialist_candidate.yaml` không dùng
`nam_tac_nhan_dung_chung` của V193. Factory tạo một policy Direct-RNN riêng
cho từng context (`mot_context_mot_chuyen_gia`), hiện có sáu context owner.

Khi `direct_rnn_full_rl_enabled=true`, một receipt Codex hợp lệ và đúng owner
được đưa vào chính hàng đợi Full-RL AdamW của lifecycle post-close. Không có
writer action-bias thứ hai trong profile này. Profile đặt batch `8`; hệ thống
chỉ cho phép giá trị trong khoảng `2..64` (regression test dùng `2` để kiểm
tra nhanh):

```text
receipt 1..N-1 -> direct_rnn_full_rl_adamw_v1, pending=true, chưa đổi tensor
receipt N -> batch AdamW khi quota đủ, optimizer_steps tăng, hash tensor đổi
```

Receipt/update phải được đọc bằng các trường riêng, không suy ra từ
`ppo_updates` tổng quát:

- `rlhf_parameter_scope=direct_rnn_full_rl_adamw_v1`;
- `rlhf_parameter_hash_before/after` và `rlhf_direct_rnn_block_updates`;
- `rlhf_direct_rnn_queue_pending`, `rlhf_direct_rnn_optimizer_steps` và
  `rlhf_direct_rnn_updated_modules`;
- `rlhf_reward_mean` giữ nguyên dấu reward Codex;
- `rlhf_target_action_probability_after` chỉ có khi AdamW đã ghi tensor và
  được gắn nguồn `frozen_forward_action_head_projection_v1`.

Nếu chưa đủ batch, `pending=true` là trạng thái chờ hợp lệ, không phải
`rejected` và cũng không phải bằng chứng model đã học. Nếu trace không có
forward cache hoặc action không hợp lệ, specialist fail-closed; không quay về
legacy global action-bias. Profile V193 shared và các checkpoint cũ vẫn giữ
đường tương thích cũ cho tới khi được chuyển sang profile specialist bằng một
candidate checkpoint mới.

Contract này chỉ chứng minh đường receipt -> tensor update. Nó không chứng
minh lợi nhuận, giảm drawdown hoặc đủ điều kiện promote; các điều đó vẫn cần
paired/frozen OOS và ledger reconciliation độc lập.

Audit vẫn dùng cùng `min_samples` và cùng công thức LCB với ledger legacy để
promotion/frozen replay/OOS báo đúng số lifecycle policy-origin. Báo cáo có:

- `ledger_source=full_rl_audit`
- `telemetry_only=true`
- `policy_origin_lifecycle_count`
- `conditioned_entry_expectancy.record_count`

Lifecycle acknowledgement thay `expectancy_skipped_full_rl` bằng
`expectancy_audit` và `expectancy_audit_only_full_rl`.

## Đọc đúng hỗ trợ điều kiện (V165)

`conditioned_entry_expectancy` có hai cấp số khác nhau, không được gộp làm
một. `sample_count` là tổng lifecycle entry đã settled và được đưa vào các
bucket; `eligible_record_count` chỉ là số bucket riêng lẻ đã đạt
`min_samples`. Vì một bucket giữ đủ `(context, action, trend, location,
volatility)`, một run ngắn có thể có hàng chục `sample_count` nhưng vẫn có
`eligible_record_count=0` do dữ liệu đang phân tán. Báo cáo cũng ghi
`eligible_sample_count`, `max_record_sample_count` và
`samples_to_first_eligible` để dashboard không diễn giải sai thành “policy
không nhận tín dụng”. Các trường này chỉ quan sát, tuyệt đối không đi vào
logit, mask, selector, risk hoặc optimizer.

## Bất biến bắt buộc

1. `w_in`, recurrent/deep weights, `w_out`, value head và gate chỉ được AdamW
   Full-RL batch sửa.
2. `memory_supernet.contextual_expectancy` giữ byte-identical sau một
   Full-RL lifecycle; audit không được tạo decision-time action bias.
3. Cùng một receipt không thể ghi hai lần nhờ receipt identity canonical.
4. Conditioned record chỉ nhận entry policy-origin với final R, không nhận
   pending/reject, management shadow hay extrema suy diễn.

## Regression

`tests/test_full_rl_exclusive_writer.py` chứng minh một lifecycle Full-RL:

- tăng đúng một policy-origin audit sample và một conditional bucket;
- có acknowledgement audit-only;
- không đổi serialized decision ledger, action parameter head hay risk gate.

Run MT5 mới phải xác nhận các counter này tăng cùng số lifecycle đã settled.
