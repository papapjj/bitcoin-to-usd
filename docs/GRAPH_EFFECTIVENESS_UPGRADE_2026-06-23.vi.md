# Nâng cấp hiệu năng và hiệu quả Graph M1 — 2026-06-23

## Mục tiêu

Đợt nâng cấp này xử lý ba lỗi đã làm graph M1 vừa chậm vừa không có giá trị trading rõ ràng:

1. activation graph duyệt từng edge Python, trace toàn bộ edge ở mọi quyết định;
2. reward/credit được cấp cho `PLACED`, `ACCEPTED` hoặc fill mở vị thế, tức chưa có PnL lifecycle;
3. report MT5 có thể ghi `0 trades/0 profit` trong khi execution ledger vẫn có lifecycle, làm fitness `-100` với lý do sai.

Không có thay đổi nào được coi là bằng chứng PnL chỉ vì unit test pass. MT5 Strategy Tester với EA thật vẫn là authority cuối cùng.

## Các thay đổi đã triển khai

### 1. Sparse/vectorized graph path

- `NeuralGraph.activate()` dùng topology cache và phép cộng vector hoá thay vì vòng lặp Python theo edge cho phần activation nóng.
- Activation step mặc định của `M1GraphModule` giảm từ 5 xuống 3.
- `GraphActivationTrace` giữ tổng số edge active nhưng chỉ lưu tối đa `trace_max_edges=128` edge attribution; candidate list để xếp hạng trace bị giới hạn `trace_candidate_edges=32`.
- `M1GraphModule._candidate_actions()` dùng topology compiled thay vì quét `dict` edge ở mỗi action.
- Clamp/finite-state dùng xử lý mảng để tránh hàng chục nghìn gọi scalar NumPy.

Microbenchmark cục bộ 119 input -> 20 output, 20 iterations:

| Backend | Trung bình `act()` | Ghi chú |
|---|---:|---|
| Legacy | khoảng 2.40 ms | baseline cục bộ |
| Graph sau tối ưu | khoảng 15.50 ms | 243 neuron, 8,532 edge, trace giữ 128 edge |

Graph vẫn chậm hơn legacy khoảng 6.5 lần trong microbenchmark này, nhưng không còn ở mức hàng trăm mili-giây do full Python-edge/full-trace. Đây là benchmark quyết định cục bộ, **không phải benchmark MT5 và không phải bằng chứng lợi nhuận**.

Chạy lại:

```bash
PYTHONPATH=src python scripts/benchmark_m1_backend.py --iterations 20
```

### 2. Credit chỉ sau outcome có ý nghĩa kinh tế

- `PLACED` và `ACCEPTED` không nhận reward/cost, TD credit hoặc memory outcome.
- Fill mở vị thế chỉ cache trace để khớp với lifecycle cuối; graph utility/gate/weight không đổi tại thời điểm fill.
- Credit cho action chỉ chạy khi lifecycle đóng có PnL/cost thực hiện và evidence bridge hợp lệ.
- `ExperienceStarvationState` chỉ tạo `exploration_pressure`; intrinsic reward bằng 0, không còn biến thiếu trải nghiệm thành “lợi nhuận” giả. Journal mặc định được cô lập theo output (`<output>.journal.jsonl`) để một run không replay ngầm lịch sử stale của run trước.
- Position close/management do model điều khiển cần position đủ tuổi (`min_manual_close_age_bars=2`) trừ emergency loss (`unrealized_r <= -0.75`). Điều này giảm close churn quá sớm mà vẫn giữ EA SL/TP độc lập.

### 3. Reconciliation report MT5 và execution ledger

- Thêm `MT5ReportLedgerReconciliation` cho trade count và realized PnL.
- Khi ledger có lifecycle nhưng report không khớp trade/PnL vượt tolerance, trial fail bằng `report_execution_mismatch`.
- Không còn chuyển mismatch đó thành `net_profit_not_positive`; do đó lỗi telemetry không bị nhầm với lỗi strategy.
- MT5 metrics và trial history lưu reconciliation evidence/reasons để audit.

### 4. Promotion graph theo paired baseline

Khi `m1_backend=graph` và `require_mt5_promotion=true`:

- candidate graph được kiểm tra cùng evaluator với baseline legacy;
- report/ledger phải reconciled;
- candidate không được xấu hơn baseline ở expectancy LCB, immediate-exit ratio và execution-failure rate theo guard band;
- chỉ khi các gate pass, structural diff mới được promote; nếu fail, baseline không bị mutate và audit ghi rejection reason.

Cấu hình/CLI phát hành do đó giữ:

```yaml
supernetwork:
  m1_backend: legacy
  require_mt5_promotion: true
```

Graph là opt-in:

```bash
python -m self_growing_trader.cli train-mt5 --m1-backend graph ...
```

## Protocol MT5 A/B bắt buộc

Dùng cùng runner, EA build, symbol, date window, model tick, spread setup, seed policy và trial count.

```powershell
# 1) Baseline legacy
python -m self_growing_trader.cli train-mt5 `
  --m1-backend legacy --seed-policy outputs\baseline.pkl `
  --rounds 1 --trials 1 --runner-dir <MT5_RUNNER> `
  --from-date <YYYY.MM.DD> --to-date <YYYY.MM.DD> `
  --output outputs\ab_legacy.pkl

# 2) Graph candidate, cùng dữ liệu/điều kiện
python -m self_growing_trader.cli train-mt5 `
  --m1-backend graph --seed-policy outputs\baseline.pkl `
  --rounds 1 --trials 1 --runner-dir <MT5_RUNNER> `
  --from-date <YYYY.MM.DD> --to-date <YYYY.MM.DD> `
  --output outputs\ab_graph.pkl

# 3) Smoke có contract bridge/report/artifact graph
python scripts/run_graph_mt5_smoke.py --runner-dir <MT5_RUNNER>
```

Promote graph chỉ khi: bridge telemetry có dòng, artifact 119x20 load được, report parse được, report/ledger reconciled, và graph vượt/không xấu hơn baseline theo gate đã nêu.

## Những gì chưa thể xác nhận trong archive

- compile EA MQL5 trong terminal MT5 thật;
- Strategy Tester smoke có report thực;
- paired A/B profitability trên cùng window;
- stability ngoài mẫu/walk-forward.

Không đánh dấu các kết quả này là hoàn thành cho đến khi có artifact MT5 tương ứng.
