# Trial micro-growth theo context lân cận

## Mục đích

Khi graph candidate nhận một outcome đã **được thực thi và đóng lifecycle**,
hệ thống có thể mọc ngay một micro-neuron yếu. Cơ chế này chỉ chạy trong bản
copy candidate của trial; baseline/legacy không đổi topology trong lúc chạy.

## Quy tắc sống còn

Một micro-neuron được sinh ra với `home_context_key` và một `context_prototype`
từ encoded market state cộng điều kiện account/position/risk/memory/lifecycle.
Nó **không cần** chứng minh hiệu quả ở các context khác loại.

- Mỗi outcome sau đó được so cosine similarity với prototype.
- Nếu similarity đạt `micro_growth_context_similarity_threshold`, outcome đó
  thuộc cùng hoặc gần cùng context family.
- Credit dương khi neuron/edge đã active trong context family này tăng
  `similar_context_supports`, utility và fade strength; prototype chỉ trôi chậm
  theo `micro_growth_context_prototype_lr`.
- Khi đủ `micro_growth_survival_similar_outcomes`, positive supports và utility,
  neuron chuyển `candidate → active`.
- Context xa là **neutral evidence**: không giữ, cũng không làm neuron fade.
- Context gần nhưng neuron không được dùng hoặc cho credit âm mới làm gate,
  weight và fade giảm dần; dưới ngưỡng thì `pruning`.

Điều này cho phép một specialist chỉ tồn tại cho một regime địa phương, ví dụ
low-risk trend pullback, thay vì buộc nó phải hữu ích cả trend/range/high-stress.

## Config

```yaml
supernetwork:
  enable_trial_micro_growth: true
  micro_growth_survival_similar_outcomes: 3
  micro_growth_context_similarity_threshold: 0.82
  micro_growth_context_prototype_lr: 0.08
```

`micro_growth_survival_contexts` vẫn được đọc cho artifact/config cũ, nhưng
chỉ là alias tương thích. Cấu hình mới phải dùng
`micro_growth_survival_similar_outcomes`.

## Event audit

`growth_events` và `supernetwork_trial_history` ghi:

- `micro_neuron_grown` cùng home context và threshold;
- `micro_neuron_retained` cùng similarity, số support/observation gần context;
- `micro_neuron_fading` hoặc `micro_neuron_pruned` chỉ khi outcome thuộc context
  gần nhưng không có hiệu quả.

Không mọc/cập nhật từ `PLACED`, `ACCEPTED` hoặc fill mở vị thế chưa có outcome
kinh tế cuối.
