# Validation — trial micro-growth

**Ngày:** 2026-06-23  
**Phạm vi:** graph candidate micro-growth, plastic survival, fade/prune, config, serialization compatibility.

## Đã triển khai

- `TrialMicroGrowthController` trong `src/self_growing_trader/trial_micro_growth.py`.
- `SelfGrowingPolicy.propose()` gán `graph_trial_id` và bắt đầu trial scope trên candidate copy.
- `SelfGrowingPolicy.observe_transition()` gọi immediate micro-growth chỉ sau outcome đã `action_feedback_eligible`.
- Newborn neuron/edge có metadata survival: trial birth, context set, support count, observation count, fade.
- Newborn edge được wire bằng whitelist vùng học được, fanout dẻo sang M1, peer fanout sang module khác và hồi tiếp hiện hữu, low gate/weight, candidate lifecycle.
- Cross-context positive support giữ neuron; unsupported node fade gate/weight rồi pruning.
- Config YAML và `SupernetworkConfig` đã có tất cả guardrail micro-growth.

## Điều không thay đổi

- Legacy backend không sinh graph neuron.
- Lệnh chỉ `PLACED`, `ACCEPTED`, hoặc fill mở không làm graph mọc / học topology.
- Baseline policy không bị mutation topology; chỉ candidate copy trong trial thay đổi.
- MT5 promotion/reconciliation gate vẫn là điều kiện để candidate được giữ sau trial.

## Xác minh cục bộ

```text
pytest -q
211 passed in 22.71s

PYTHONPATH=src python -m self_growing_trader.cli --help
CLI_OK
```

Test mới: `tests/test_trial_micro_growth.py`

- Mọc neuron ngay sau learning outcome.
- Edge newborn bị giới hạn gate/weight.
- Retain chỉ sau support dương ở context khác nhau.
- Fade dần rồi prune khi không có support.
- Candidate growth không làm baseline đổi topology.

## Bằng chứng còn cần môi trường ngoài archive

- MT5 Strategy Tester paired A/B graph-vs-legacy.
- EA compile/runtime bridge và report–execution ledger reconciliation.

Các test cục bộ chứng minh lifecycle và guardrail đúng; chúng không chứng minh strategy có lợi nhuận.
