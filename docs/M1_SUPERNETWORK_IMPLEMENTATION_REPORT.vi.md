# Báo cáo triển khai M1 Graph Supernetwork — 2026-06-23

> **Đính chính hiệu quả vận hành — cập nhật 2026-07-10:** Báo cáo nền này mô tả phạm vi mã nguồn graph đã được xây dựng, không phải chứng minh graph có lợi nhuận. Cấu hình và CLI phát hành hiện dùng `m1_backend=direct_rnn`; graph là opt-in (`--m1-backend m1_graph` hoặc alias `graph`) cho đến khi thắng paired MT5 A/B với report/ledger reconciliation. Xem `docs/GRAPH_EFFECTIVENESS_UPGRADE_2026-06-23.vi.md`.

## Trạng thái bàn giao

Bản này hoàn tất phần có thể triển khai và xác minh trong mã nguồn: graph M1 là **backend experimental có candidate path**, candidate/child chỉ thay đổi topology trên bản copy, module specialist có `ModuleSpec` và cross-module wiring thật, còn MT5 vẫn là promotion authority cuối cùng.

Checklist hiện có:

- **263 mục `[x]`**: có mã và kiểm tra cục bộ.
- **13 mục `[~]`**: contract đã có nhưng cần Strategy Tester/EA thật để chứng minh theo dữ liệu lệnh.
- **4 mục `[ ]`**: bốn lần ghi của hai evidence external bắt buộc: MT5 smoke và compile EA/report thật.

Không có assertion nào trong báo cáo này tuyên bố EA hoặc MT5 Strategy Tester đã chạy trong archive, vì môi trường đóng gói không có `terminal64.exe`, MetaEditor hoặc report tester thật.

## Những phần đã hoàn tất trong code

### M1 graph experimental và compatibility

- `SelfGrowingPolicy.seed()` mặc định tạo direct RNN policy; graph chỉ được tạo cho test/candidate engineering khi truyền `m1_backend="m1_graph"` hoặc alias `graph`. `SupernetworkConfig` và CLI phát hành mặc định `m1_backend="direct_rnn"`.
- `legacy` là baseline production hiện tại và cũng là route migration/testing cho artifact cũ.
- `ensure_compatible()` giữ artifact thiếu backend ở legacy, đồng thời repair policy khai báo graph nhưng còn mang `M1ThinkingModule` bằng cách migrate nhất quán sang `M1GraphModule`.
- Giữ nguyên bridge v2, action schema v4, `INPUT_COUNT=119`, `OUTPUT_COUNT=20`.

### Runtime, trace và credit assignment

- `M1GraphModule` nhận context `ExperienceSuperNetwork` vào vùng memory/predictor/critic/selector, action bias bị clamp và risk context chỉ giảm logit mở lệnh.
- `GraphActivationTrace` và `StepTrace` lưu active node/edge, contribution, gate, region activity, memory match, lifecycle tag và module-need reason.
- `GraphPlasticityEngine` dùng primitive Hebbian, anti-Hebbian, Oja, gradient-like và homeostatic từ `plasticity.py`; chỉ update evidence active, có clamp, decay, consolidation, pruning pressure và finite-state guard.

### Candidate, specialist, child và lineage

- Structural action grow/prune/rewire/split/merge/consolidate chỉ chạy trên candidate copy; rollback không alias state về baseline.
- `GraphSupernetworkRuntime` chuyển trace/error cluster thành `ModuleNeedSignal -> ModuleSpec -> structural diff`; priors/cooldown chỉ nhận feedback promote/reject.
- Tất cả 10 specialist role có ModuleSpec rõ input/output/validation goal và `CrossModuleConnector` nối **qua anchor neuron specialist thật**, không còn chỉ ghi metadata M1→M1.
- Cross-module edge có whitelist mở trên vùng học được, fanout dẻo sang M1, peer fanout sang module khác, hồi tiếp cho module mới, budget, low gate/weight, metadata audit, output-bias bound và credit riêng.
- `BlueprintGenerator`, `AdaptiveSuperNetwork`, `ChildSupernetworkManager`, `NetworkLineage` đảm bảo child có graph/memory/plasticity/controller riêng, seed/id khác parent và không copy raw parent memory/weight/topology.

### Vận hành, quan sát và tái lập

- Graph observability: graph size, edge health, credit attribution, need score, growth acceptance, role/cluster outcome, prediction trend, memory usefulness, selector switching và trial-history JSONL.
- `pyproject.toml` giúp `pip install -e '.[test]'` chạy trên máy sạch; root `pytest` chỉ thu thập `tests/`, không bị script nghiên cứu trong `work/` làm hỏng CI.
- CLI/config có backend graph opt-in, child lineage limits, module role budget, need thresholds, cross-module/recurrent/lateral budgets, flexible M1 fanout, peer module fanout và whitelist vùng học được; promotion graph dùng paired A/B với legacy và report/ledger reconciliation.
- `scripts/run_graph_mt5_smoke.py` preflight runner cho backend graph opt-in, chạy smoke `rounds=1,trials=1,m1_backend=m1_graph`, rồi chỉ trả `ok=true` khi bridge, report và artifact graph 137×20 đều hợp lệ.

## Xác thực đã chạy

| Hạng mục | Kết quả |
|---|---|
| `pip install -e '.[test]'` | Pass |
| `pytest -q -o faulthandler_timeout=60` | **200 passed in 71.29s** |
| `python -m compileall -q src scripts tests` | Pass |
| `python -m self_growing_trader.cli train-mt5 --help` | Pass |
| MT5 smoke helper preflight trên runner thiếu | Fail-fast đúng kỳ vọng, exit code 2 |
| Graph local paths: default/migration/trace/credit/candidate/child/role wiring/pickle | Covered by test suite |

## Evidence external còn cần trên máy có MT5

Chạy runbook: [`MT5_GRAPH_SMOKE_RUNBOOK.vi.md`](MT5_GRAPH_SMOKE_RUNBOOK.vi.md).

1. Compile `SelfGrowing_MT5_Trainer_EA.mq5` bằng MetaEditor cùng include cần thiết.
2. Chạy `python scripts/preflight_mt5.py --runner-dir <MT5_RUNNER>`.
3. Chạy `python scripts/run_graph_mt5_smoke.py --runner-dir <MT5_RUNNER>`.
4. Chỉ xem graph/child candidate là deployable khi script trả `ok=true` và MT5 fitness/drawdown/trade-count/telemetry promotion gate đều vượt baseline.
