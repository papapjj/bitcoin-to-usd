# Validation nâng cấp hiệu quả Graph M1 — 2026-06-23

## Xác minh cục bộ đã chạy

- `py_compile` cho toàn bộ `src/` và `scripts/`.
- CLI `train-mt5 --help` hiển thị `--m1-backend {legacy,graph}` cùng các cross-module/lineage guard.
- Toàn bộ suite Python: **207/207 tests pass trong 17.47 giây**. Coverage gồm graph, bridge/evaluator, config/CLI, metrics, child/lineage, module wiring, MT5 smoke preflight và `test_self_growing_runner.py` integration.
- `py_compile` và `train-mt5 --help` cũng pass.
- Microbenchmark graph 119->20 xác nhận trace top-K và vectorized path hoạt động.

## Contract test mới/được tăng cường

- Trace bị giới hạn số edge record nhưng vẫn giữ active-edge count.
- Report/ledger mismatch bị reject rõ ràng, không trả reason `net_profit_not_positive`.
- `PLACED`/`ACCEPTED` không nhận credit; fill chỉ cache trace, lifecycle close mới update graph.
- Close/management bị chặn khi position quá non-young, trừ emergency loss.
- Graph promotion bị reject nếu xấu hơn baseline về immediate-exit ratio, expectancy LCB hoặc execution failure rate.
- Graph promotion bị reject nếu reconciliation fail, ngay cả khi score nội bộ cao.

## Giới hạn validation

MT5 terminal, compiler MQL5, historical data và EA runtime không nằm trong archive. Vì vậy không có claim rằng graph đã có lợi nhuận hoặc đã vượt legacy. Bằng chứng đó phải được sinh theo `GRAPH_EFFECTIVENESS_UPGRADE_2026-06-23.vi.md` và lưu cùng report, ledger, journal và artifact.
