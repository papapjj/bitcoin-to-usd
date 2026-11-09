# Báo cáo kiểm thử V13 Expectancy — 2026-06-26

## Môi trường kiểm thử

- Source: `self_growing_trader_expectancy_v13`
- Python test invocation dùng `PYTHONPATH=src`
- Không gọi MetaTrader 5 Strategy Tester hoặc broker/live execution trong môi trường này.

## Kết quả

```text
PYTHONPATH=src pytest -q
270 passed in 19.21s

PYTHONPATH=src python -m compileall -q src
PASS
```

Focused regression trước full suite:

```text
tests/test_contextual_expectancy_v13.py
tests/test_m1_graph_module.py
tests/test_context_frame_v12.py
tests/test_mt5_runner.py
tests/test_self_growing_runner.py

141 passed in 15.62s
```

## Regression cases V13 mới

| Test | Contract được khóa |
|---|---|
| `test_small_sample_positive_mean_is_not_reliable_positive_edge` | +3R một mẫu không tạo edge tin cậy |
| `test_context_action_ledger_is_exactly_scoped_and_bootstrap_cannot_bias_policy` | Không rò BUY/SSELL; legacy bootstrap không bias policy |
| `test_policy_lifecycle_credit_creates_context_action_expectancy_memory` | Final lifecycle được ghi vào ledger policy-origin |
| `test_policy_query_includes_positive_contextual_expectancy_bias` | Positive LCB được đưa vào decision-time memory bias |
| `test_promotion_rejects_global_winner_without_policy_origin_positive_expectancy` | PnL/fitness toàn cục không vượt gate khi policy EXP LCB âm |
| `test_promotion_ratio_and_delta_are_configurable` | Ratio policy-origin và LCB delta là config, không hard-code độc lập |
| `test_config_parses_expectancy_controls` | YAML config V13 được parse đúng |

## Không được suy diễn từ kết quả test

270 unit/integration test pass chỉ chứng minh contract Python hiện tại nhất quán. Nó **không** chứng minh policy có positive expectancy trên BTCUSDm, không chứng minh fill/swap/slippage MT5 khớp production, và không thay thế OOS/walk-forward hoặc real-tick validation.
