# Direct-RNN multi-task auxiliary V1

> Cập nhật V186: khi `tensor_rule_enabled=true`, các label trong tài liệu này được chuyển sang encoder TensorRuleBank và legacy snapshot auxiliary/conditioned/counterfactual representation bị tắt để tránh hai head cùng học một mục tiêu. Checkpoint/baseline cũ vẫn dùng đường legacy. Xem `docs/TENSOR_RULE_ABSTRACTION_V186.vi.md`.

## Mục tiêu

Một snapshot quyết định chỉ có reward lifecycle cuối là quá thưa. Candidate
hiện tái sử dụng immutable 48-M1 path sau mọi quyết định HOLD/entry để tạo
nhiều nhãn trễ cùng lúc, nhưng không đưa bất kỳ future value nào vào inference.
Thu cả HOLD là bắt buộc trong profile active để model không tự làm mất nhãn ở
vùng state mà nó vừa học được nên đứng ngoài.

## Contract nhãn

`build_auxiliary_targets()` hiện tạo vector 91 chiều có mask:

- 3 BCE dự báo hướng return ở horizon 3, 12 và 48 M1;
- 3 pinball loss cho quantile 0.10, 0.50 và 0.90 của return horizon dài;
- 1 Huber target realized volatility trên path;
- cho mỗi 6 action entry: fill probability, normalized time-to-positive,
  expected R, xác suất SL, xác suất TP, timing quality, MFE và MAE.
- cho mỗi 6 action entry: `enter_now_probability`, `timing_advantage_r` và
  `optimal_delay_fraction`, được so từ cùng action tại M1 0/1/2/4/8/12.

Target action chỉ hợp lệ sau khi hypothetical order fill; fill label luôn hợp
lệ. Mask ngăn unfilled order tạo nhãn MFE/MAE/R giả.

`action ranking` dùng objective phân cấp hiện tại: BCE TRADE/HOLD và
cross-entropy action-given-TRADE trên sáu entry. Cách này thay objective
softmax phẳng cũ để sáu lớp entry không áp đảo một lớp HOLD chỉ vì số lượng.
`regime transition` tiếp tục do TimingNet học ở các horizon 3/6/12 từ context
label trưởng thành. Reconstruction dùng hidden cuối để tái tạo causal masked
input của chính snapshot, không dùng path tương lai.

## Kiến trúc và writer

Mỗi `ModuleState` có `auxiliary_w/b` kích thước `91 × hidden_size` và
`auxiliary_reconstruction_w/b` kích thước `input_size × hidden_size`. Các head
chỉ chạy trong `_apply_direct_rnn_full_rl_batch()` trên item
`entry_preference_only`; inference không đọc chúng.

Gradient head và encoder đi qua Full-RL AdamW hiện hữu. Không có optimizer,
queue hoặc writer thứ hai. Auxiliary item vẫn không được tự tạo batch: quota
bắt buộc ít nhất một nửa batch là lifecycle MT5 canonical vẫn giữ nguyên.
Retention replay có thể bổ sung item cũ sau khi batch live đã đạt quota; store
không phải writer và không thay đổi cadence đủ điều kiện của batch live.

## Cấu hình

```yaml
learning:
  direct_rnn_rl_auxiliary_enabled: true
  direct_rnn_rl_auxiliary_loss_weight: 0.10
  direct_rnn_rl_reconstruction_loss_weight: 0.02
```

Artifact cũ được bổ sung head bằng seed xác định theo tên module. Vì head không
tham gia inference, migration không đổi action surface. Khi input width hoặc
hidden width thay đổi, các head được resize cùng topology.

## Telemetry

Event `direct_rnn_full_rl_batch` ghi:

- `multi_task_auxiliary_sample_count`;
- `multi_task_auxiliary_eligible_targets`;
- `multi_task_auxiliary_loss`;
- `multi_task_auxiliary_reconstruction_loss`;
- `multi_task_auxiliary_gradient_l1`.

Đây là bằng chứng routing/gradient, không phải bằng chứng edge. Promotion vẫn
cần MT5 train và nhiều cửa sổ frozen OOS.
