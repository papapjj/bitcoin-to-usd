# Nâng cấp micro-growth: giữ neuron theo context giống hoặc gần giống

## Thay đổi yêu cầu

Micro-neuron không còn phải có credit dương ở nhiều context khác nhau. Một
neuron mới là một specialist địa phương: nó được giữ khi lặp lại hiệu quả trong
context ban đầu hoặc các context có embedding/regime đủ gần.

## Cơ chế

1. Outcome đã đóng lifecycle tạo `context_prototype` cho neuron mới.
2. Mỗi outcome sau được so cosine similarity với prototype.
3. Similarity đạt ngưỡng thì mới là evidence cho neuron đó.
4. Credit dương + neuron/edge active trong context gần tăng support, utility,
   gate survival và cập nhật prototype rất chậm.
5. Context xa là neutral: không tăng support và không làm fade.
6. Context gần nhưng inactive/credit âm làm weight/gate fade dần; qua ngưỡng
   pruning thì disable edge.
7. Candidate cần nhiều support trong cùng family để `candidate -> active`; sau
   đó consolidation vẫn yêu cầu utility cao hơn.

Action index không được đưa vào context similarity. Như vậy neuron vẫn có thể
sống khi market/account state gần giống nhưng graph đã học để chọn action khác.

## Config

```yaml
supernetwork:
  micro_growth_survival_similar_outcomes: 3
  micro_growth_context_similarity_threshold: 0.82
  micro_growth_context_prototype_lr: 0.08
```

`micro_growth_survival_contexts` chỉ giữ để đọc config/artifact cũ. Config mới
nên dùng `micro_growth_survival_similar_outcomes`.

## Validation cục bộ

- Unit test: context gần + credit dương lặp lại sẽ retain neuron.
- Unit test: context xa + outcome trung tính không làm fade neuron.
- Unit test: context gần không được dùng sẽ fade rồi prune dần.
- Toàn bộ suite: `213 passed`.

Điều này chứng minh contract mã cục bộ. MT5 A/B thật vẫn là gate cuối trước khi
promote bất kỳ graph candidate nào.
