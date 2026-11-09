# Live Neural Observatory V14 — Test Report

## Static validation

```text
python -m compileall -q src tests
PASS
```

## Live smoke test

Đã tạo seed policy graph, bật telemetry, gọi:

```text
run_started
→ topology
→ decision
→ feedback
→ development
→ trial_finished
→ run_finished
```

Kết quả topology trong smoke test:

```text
nodes: 308
edges: 8553
```

`LiveEventReader` đọc được full event sequence; local HTTP endpoint `/api/state` trả topology, events và latest decision.

## Regression tests

| Nhóm | Kết quả |
|---|---:|
| Observatory + CLI + V12 + V13 + graph + MT5 focused | 70 passed |
| Regression `test_live_observatory` + runner + MT5 sau final persistence patch | 123 passed |
| Suite nhóm A | 91 passed |
| Suite nhóm B | 182 passed |
| Tổng các nhóm suite A+B | 273 passed |

## Contract được kiểm tra

- Bật observer không đổi `selected_action`.
- Bật observer không đổi `candidate_scores`.
- Event `run_started`, `topology`, `decision` được ghi.
- Event `feedback`, `development`, `trial_finished` được ghi.
- Topology có node thực từ graph M1.
- YAML parse được section `observability`.
- CLI nhận subcommand `neural-observatory` và cờ live telemetry.
- MT5 runner tương thích với policy có event hooks.

## Giới hạn đã biết

- Chưa có MT5 Strategy Tester thực trong môi trường build; smoke này mô phỏng policy/bridge flow ở Python.
- Nếu EA không gửi request, dashboard chỉ thấy run/trial state, không có decision trace. Đây là chẩn đoán bridge/EA/history, không phải lỗi visualizer.
- Live dashboard là localhost HTTP, không được expose internet và không có remote control.
