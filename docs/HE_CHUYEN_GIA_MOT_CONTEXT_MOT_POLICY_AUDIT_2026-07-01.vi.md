# Audit nâng cấp — Một context, một chuyên gia, một policy

## Phạm vi

Đối chiếu dự án với kế hoạch `KE_HOACH_NANG_CAP_MOT_CONTEXT_MOT_CHUYEN_GIA (1).md`.

Mục tiêu kiến trúc:

```text
MT5 request hợp lệ
→ nhận diện context lá ổn định
→ gọi đúng một expert owner
→ đúng một policy độc lập xuất 20 output
→ bridge/risk kiểm tra
→ MT5 thực thi hoặc từ chối
→ lifecycle/reward chỉ quay về policy owner.
```

## Trạng thái sau bản vá 2026-07-01

| Mục | Trạng thái | Bằng chứng hiện tại |
|---|---|---|
| P0 schema Python/EA/MT5 | Đã khóa bằng test static | `test_mt5_ea_schema_versions_match_python_contract` đọc cả 3 runner EA và so với `protocol_contract.py`. |
| P0 session manifest hash | Đã ghi manifest cạnh report MT5 | Mỗi lượt `evaluate_policy()` ghi `Reports\<report_name>.manifest.json` với SHA256 source EA, EX5, terminal, tester INI/SET, report, cây Python, `protocol_contract.py`, config hiệu lực và policy state; khóa bằng `test_strategy_tester_evaluator_writes_session_manifest_hashes`. |
| P0 fail-closed EA build/hash | Đã có preflight + test | `MT5StrategyTesterEvaluator._verify_ea_build_integrity()` dừng trước MT5 nếu `.mq5` mới hơn `.ex5`, thiếu EX5, hoặc hash pin không khớp; khóa bằng `test_strategy_tester_evaluator_fail_closes_on_stale_or_mismatched_ea_build`. |
| P0 provenance sidecar | Đã có test | `test_action_metadata_sidecar_records_context_owner_without_changing_csv` xác minh sidecar `.metadata.jsonl` ghi `ma_context`, `ma_chuyen_gia`, `ma_policy`, `phien_ban_policy` mà không đổi CSV bridge. |
| P1 context chưa ổn định/no-owner không học | Đã có guard + test | `MT5StrategyTesterEvaluator._trace_learning_blocked()` chặn journal và các nhánh `observe_transition`; `test_no_policy_context_trace_is_blocked_from_learning_and_journal`. |
| P2 context root + owner đơn | Đã có test | `test_exactly_one_owner_policy_decides_for_a_request`, `test_position_state_switches_to_a_single_management_owner`, `test_context_switches_for_pending_sell_and_defensive_states`. |
| P2 switching pending/phòng thủ | Đã nối execution context vào recognizer | `BoNhanDienNguCanh` nhận `execution_context` read-only để route pending sang `CTX_GOC_XU_LY_LENH_CHO`; test phủ pending mở, pending clear về entry, SELL management và drawdown/account risk phòng thủ. |
| P3 policy/checkpoint độc lập | Đã bổ sung checkpoint local | `HeChuyenGiaMotContextMotPolicy.luu_so_dang_ky()` ghi một checkpoint policy riêng cho mỗi context owner; `test_registry_writes_local_checkpoint_for_each_context_owner`. |
| P4 lifecycle chỉ về owner | Đã có test đơn vị | `test_outcome_updates_only_owner_policy`; runtime facade route `observe_transition`, `develop`, `apply_execution_lifecycle_credit` theo `trace_id -> expert`. |
| P4 không mutate chéo expert | Đã có test 1.000 lifecycle | `test_no_cross_expert_mutation_after_one_thousand_owner_lifecycles` dùng policy giả có state mutating, chạy 1.000 transition + lifecycle credit cho một owner và xác nhận các expert khác giữ nguyên state. |
| P5 context branch | Đã loại bỏ ở V162 | Context branch làm phân mảnh lifecycle/AdamW. Taxonomy M15 nay cố định; artifact cũ được migrate để bỏ mọi owner branch và selector state liên quan. |
| Restart artifact owner mapping | Đã có test | `test_artifact_restart_preserves_fixed_context_owner_mapping` giữ mapping taxonomy cố định qua pickle/load. |
| Risk cap per-expert | Đã có cap riêng cho giai đoạn thăm dò | `max_risk_pct_tham_do` clamp risk head trước bridge; `max_open_risk_pct_tham_do` chặn policy call và trả no-op không học khi exposure vượt cap; `test_probe_expert_risk_cap_clamps_risk_head_before_bridge_decode`, `test_probe_expert_open_risk_cap_blocks_policy_call_with_no_learning_trace`. |
| CLI quản trị context | Đã có promotion/archive/dormant/merge audit command | `quan-tri-he-chuyen-gia` load artifact, gọi gate per-context, lưu pickle/registry/audit JSONL; merge yêu cầu `--confirm-merge`, `--oos-pass`, `--non-interference-pass`. |
| Rollback artifact/registry | Đã backup trước overwrite | `backup_context_expert_artifacts()` copy artifact và registry hiện có sang file `.rollback.<timestamp>.*` trước khi CLI init/admin hoặc MT5 context-expert train ghi bản mới; test CLI admin xác nhận backup giữ nội dung cũ. |
| UI context owner/execution | Đã đọc sidecar + ledger và hiển thị trong Live tab | `LiveNeuralPanel` tail read-only `self_growing_action.csv.metadata.jsonl`, đọc dòng cuối `self_growing_execution.csv`, merge provenance vào `Context + EXP` và hiển thị owner, trace, sequence, execution status, accepted/filled/rejected, retcode, close reason. |
| P6 production gate MT5 | Đạt smoke artifact, chưa đạt promotion chất lượng | Smoke MT5 thật đã sinh report HTML, model `.pkl`, registry, sidecar, execution ledger và feature audit; chiến lược vẫn 0 trade nên chưa đủ gate production/promote. |

## Thay đổi code chính

- `mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5`
  - Đồng bộ version và schema contract lên v7.
- `mt5_v2_runner_world/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5`
  - Đồng bộ version và schema contract lên v7.
- `src/self_growing_trader/mt5_runner.py`
  - Thêm `_trace_learning_blocked()` để chặn trace no-policy/no-learning trước `observe_transition` và journal.
  - Sidecar provenance giữ nguyên CSV bridge cũ.
  - Chuẩn hóa `Report=Reports\...`, tạo thư mục report trước khi chạy và có fallback tìm/copy report `.htm/.html` trong runner để tránh MT5 chạy xong nhưng Python không thấy artifact.
  - Ghi session manifest cạnh report MT5 ở trạng thái `started` và `finished`; manifest chứa hash source EA, EX5, terminal, tester INI/SET, report, cây Python, protocol contract, config hiệu lực, policy state, metrics, ledger/reconciliation và feature audit.
  - Thêm preflight fail-closed cho build EA: source `.mq5` mới hơn `.ex5`, thiếu binary hoặc hash pin sai đều dừng trước khi chạy Strategy Tester.
  - Các INI runner cũ trong `mt5_v2_runner*` đã được rewrite từ `Report=reports\...` sang `Report=Reports\...`; search không còn contract report cũ.
- `src/self_growing_trader/he_chuyen_gia_ngu_canh.py`
  - Ghi checkpoint local riêng cho từng context owner.
  - V162 xóa context branch/nursery; chỉ còn taxonomy M15 tường minh và migration một chiều để bỏ branch trong artifact cũ.
  - Thêm cap rủi ro per-expert cho `THAM_DO_RUI_RO_THAP`: clamp output risk index 12 và no-op khi open risk vượt cap.
  - Recognizer dùng thêm `execution_context.pending_orders`/`orders_pipe` để chọn đúng context xử lý lệnh chờ; đây vẫn là tín hiệu state read-only, không phát action.
- `src/self_growing_trader/cli.py`
  - Cấu hình stdout UTF-8 để JSON tiếng Việt không làm CLI fail trên console Windows `cp1252`.
  - Thêm `quan-tri-he-chuyen-gia` cho `promote`, `archive`, `dormant`, `merge`; mọi thao tác ghi audit JSONL và registry mới.
  - Backup artifact/registry cũ trước khi ghi đè ở `khoi-tao-he-chuyen-gia` và `quan-tri-he-chuyen-gia`.
- `src/self_growing_trader/he_chuyen_gia_ngu_canh.py`
  - Bổ sung admin methods `dat_trang_thai_chuyen_gia`, `luu_tru_chuyen_gia`, `ngu_dong_chuyen_gia`, `hop_nhat_context_chuyen_gia`.
- `src/self_growing_trader/self_growing_runner.py`
  - Truyền risk bounds tổng vào facade hệ chuyên gia để cap per-expert clamp đúng theo risk config hiện hành.
  - Thêm `backup_context_expert_artifacts()` và dùng trong `huan_luyen_he_chuyen_gia_mt5` để lưu rollback trước khi ghi artifact/registry mới.
- `configs/he_chuyen_gia_mot_context_mot_policy.yaml`
  - Bổ sung `max_risk_pct_tham_do: 0.005` và `max_open_risk_pct_tham_do: 0.015`.
- `tests/test_he_chuyen_gia_ngu_canh.py`
  - Bổ sung test checkpoint local, promotion gate, archive/dormant/merge gates, switching pending/sell/drawdown, 1.000 lifecycle không mutate chéo expert và restart artifact giữ nguyên owner mapping.
- `tests/test_mt5_runner.py`
  - Bổ sung test schema EA/Python, session manifest hash, fail-closed EA build/hash, sidecar provenance và firebreak no-learning.
- `tests/test_cli.py`
  - Bổ sung test command `quan-tri-he-chuyen-gia` ghi artifact, registry, audit log và rollback artifact/registry trước overwrite.
- `ui-java/src/com/selfgrowingtrader/control/LiveNeuralPanel.java`
  - Tab `Context + EXP` hiển thị owner/provenance thành dòng riêng thay vì chỉ nằm trong map audit.
  - Thêm sidecar tailer read-only cho `self_growing_action.csv.metadata.jsonl`; tự tìm sidecar mới nhất trong `mt5_v2_runner_current`, `mt5_v2_runner`, `mt5_v2_runner_world` và merge provenance vào context view khi `neural_live.jsonl` thiếu/chậm.
  - Thêm reader read-only cho `self_growing_execution.csv`; hiển thị `status`, `accepted`, `filled`, `rejected`, `retcode`, `close_reason` và source ledger để phân biệt EA/bridge reject.
- `ui-java/README.md`
  - Cập nhật mô tả Live UI đọc provenance từ telemetry/sidecar và execution status từ ledger read-only.

## Kiểm tra đã chạy

```powershell
python -m pytest tests\test_he_chuyen_gia_ngu_canh.py `
  tests\test_mt5_runner.py::test_mt5_ea_schema_versions_match_python_contract `
  tests\test_mt5_runner.py::test_action_metadata_sidecar_records_context_owner_without_changing_csv `
  tests\test_mt5_runner.py::test_no_policy_context_trace_is_blocked_from_learning_and_journal -q
```

Kết quả: `13 passed`.

```powershell
python -m pytest tests\test_he_chuyen_gia_ngu_canh.py tests\test_mt5_runner.py -q
```

Kết quả trước khi sửa CLI/report fallback: `64 passed`.

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả sau bản vá report/CLI: `73 passed`.

Sau khi rewrite các INI runner cũ, chạy lại cùng lệnh: `73 passed`.

Sau khi thêm UI sidecar/ledger và CLI quản trị context:

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `75 passed`.

Sau khi thêm risk cap per-expert:

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `77 passed`.

Sau khi thêm session manifest hash cho mỗi lượt MT5:

```powershell
python -m pytest tests\test_mt5_runner.py::test_strategy_tester_evaluator_writes_session_manifest_hashes -q
```

Kết quả: `1 passed`.

```powershell
python -m pytest tests\test_mt5_runner.py -q
```

Kết quả: `56 passed`.

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `78 passed`.

Sau khi thêm test restart/power-loss cho mapping context owner:

```powershell
python -m pytest tests\test_he_chuyen_gia_ngu_canh.py::test_artifact_restart_preserves_context_owner_mapping -q
```

Kết quả: `1 passed`.

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `79 passed`.

Sau khi nối pending execution context vào recognizer và bổ sung test switching:

```powershell
python -m pytest tests\test_he_chuyen_gia_ngu_canh.py::test_context_switches_for_pending_sell_and_defensive_states -q
```

Kết quả: `1 passed`.

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `80 passed`.

```powershell
python -m pytest tests\test_context_frame_v12.py -q
```

Kết quả: `8 passed`.

Sau khi thêm test 1.000 lifecycle không mutate chéo expert:

```powershell
python -m pytest tests\test_he_chuyen_gia_ngu_canh.py::test_no_cross_expert_mutation_after_one_thousand_owner_lifecycles -q
```

Kết quả: `1 passed`.

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `81 passed`.

Sau khi thêm fail-closed EA build/hash và compile lại runner current:

```powershell
.\mt5_v2_runner_current\MetaEditor64.exe /portable /compile:...\mt5_v2_runner_current\MQL5\Experts\SelfGrowing_MT5_Trainer_EA.mq5 /log:outputs\compile_mt5_v2_runner_current_fail_closed.log
```

Kết quả: `0 errors, 0 warnings`; `mt5_v2_runner_current\MQL5\Experts\SelfGrowing_MT5_Trainer_EA.ex5` cập nhật lúc `2026-07-01 16:59:45`, 70,652 bytes.

```powershell
python -m pytest tests\test_mt5_runner.py::test_strategy_tester_evaluator_fail_closes_on_stale_or_mismatched_ea_build -q
```

Kết quả: `1 passed`.

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `82 passed`.

Sau khi thêm rollback artifact/registry trước overwrite:

```powershell
python -m pytest tests\test_cli.py::test_context_expert_admin_cli_writes_artifact_registry_and_audit -q
```

Kết quả: `1 passed`.

```powershell
python -m pytest tests\test_cli.py tests\test_mt5_runner.py tests\test_he_chuyen_gia_ngu_canh.py -q
```

Kết quả: `82 passed`.

```powershell
python -c "from self_growing_trader.config import load_config; c=load_config('configs/he_chuyen_gia_mot_context_mot_policy.yaml'); print(c.he_chuyen_gia.max_risk_pct_tham_do, c.he_chuyen_gia.max_open_risk_pct_tham_do)"
```

Kết quả: `0.005 0.015`.

```powershell
python -m compileall -q src\self_growing_trader
```

Kết quả: pass.

Compile EA:

```powershell
.\mt5_v2_runner\MetaEditor64.exe /portable /compile:...\mt5_v2_runner\MQL5\Experts\SelfGrowing_MT5_Trainer_EA.mq5 /log:outputs\compile_mt5_v2_runner_schema_v7.log
.\mt5_v2_runner_world\MetaEditor64.exe /portable /compile:...\mt5_v2_runner_world\MQL5\Experts\SelfGrowing_MT5_Trainer_EA.mq5 /log:outputs\compile_mt5_v2_runner_world_schema_v7.log
```

Kết quả:

- `mt5_v2_runner`: `0 errors, 0 warnings`.
- `mt5_v2_runner_world`: `0 errors, 0 warnings`.
- `.ex5` mới được ghi lúc `2026-07-01 15:49:10`.

Java UI:

```powershell
javac --release 17 -encoding UTF-8 -d ui-java\build\classes @ui-java\build\sources.txt
```

Kết quả: pass.

```powershell
jar --create --file ui-java\build\SelfGrowingTraderControl.jar --main-class com.selfgrowingtrader.control.Main -C ui-java\build\classes .
```

Kết quả sau bổ sung sidecar + ledger reader ban đầu: `ui-java\build\SelfGrowingTraderControl.jar` được cập nhật lúc `2026-07-01 16:38:25`, 99,824 bytes. Lúc đó EXE trong `ui-java\dist` chưa package lại vì đang có 2 process `SelfGrowingTraderControl` mở và khóa thư mục dist.

Sau khi đóng UI và chạy lại:

```powershell
.\ui-java\build-ui.cmd
```

Kết quả:

- `ui-java\build\SelfGrowingTraderControl.jar`: 99,824 bytes, cập nhật lúc `2026-07-01 16:54:25`.
- `ui-java\dist\SelfGrowingTraderControl\SelfGrowingTraderControl.exe`: 442,880 bytes, cập nhật lúc `2026-07-01 16:54:38`.

Smoke MT5 profile expert:

```powershell
$env:PYTHONPATH="src"
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml huan-luyen-he-chuyen-gia-mt5 --output outputs/he_chuyen_gia_smoke_model2.pkl --runner-dir mt5_v2_runner_current --from-date 2025.01.01 --backtest-days 1 --timeout-sec 90 --model 2
```

Kết quả smoke artifact:

- Strategy Tester thật khởi động trên `BTCUSDm,M1` từ `2025.01.01` đến `2025.01.02`.
- Tester log xác nhận EA `SelfGrowing_MT5_Trainer_EA.ex5`, `Inputs=119`, `Outputs=20`, `schema_version=7`, `FeatureSchemaHash=f514180d26e7255172108046812e7153bfff0d59d85d4efd13a43e1be4b084aa`.
- Report HTML: `mt5_v2_runner_current\Reports\self_growing_mt5_train_round_0000_policy_1.htm`, 34,036 bytes, ghi lúc `2026-07-01 16:26:35`.
- Output model: `outputs\he_chuyen_gia_smoke_model2.pkl`, 77,215,578 bytes, ghi lúc `2026-07-01 16:28:05`.
- Registry: `outputs\he_chuyen_gia_smoke_model2.registry.json`, 5 chuyên gia, checkpoint local riêng từng owner.
- Sidecar `self_growing_action.csv.metadata.jsonl`: 718 rows.
- Execution ledger `self_growing_execution.csv`: 714 rows, `schema_version=7`.
- Feature audit `self_growing_feature_audit.csv`: 718 rows, `feature_schema_hash=f514180d26e7255172108046812e7153bfff0d59d85d4efd13a43e1be4b084aa`, `point=0.010000000000`.
- 3 rows đầu là warmup/context chưa ổn định: `khong_co_policy_ra_lenh=true`, chưa có owner. Row cuối có owner `CG_GOC_VAO_LENH_KHONG_CO_LENH`, policy `CG_GOC_VAO_LENH_KHONG_CO_LENH`, version `V001`.
- Metrics smoke: `profit=0.0`, `profit_factor=0.0`, `trades=0`, `equity_drawdown_pct=0.0`.
- Fitness gate đúng khi fail chất lượng: `eligible=0`, lý do `no_trades`, `insufficient_position_samples:0<8`, `net_profit_not_positive`, `execution_ledger_missing_or_no_finalised_lifecycle`.
- Lệnh đầu tiên trả exit code 1 do `UnicodeEncodeError` khi in JSON tiếng Việt trên console `cp1252`; model/registry đã được ghi trước đó. Đã sửa CLI bằng stdout UTF-8 và khóa bằng test `test_cli_configures_stdout_utf8`.

## Việc còn mở trước khi coi là production complete

- Chạy MT5 Strategy Tester thật cho cửa sổ production/OOS dài hơn và có lifecycle trade hoàn chỉnh:

```powershell
$env:PYTHONPATH="src"
python -m self_growing_trader.cli --config configs/he_chuyen_gia_mot_context_mot_policy.yaml huan-luyen-he-chuyen-gia-mt5 --output outputs/he_chuyen_gia.pkl --runner-dir mt5_v2_runner_current --from-date 2025.01.01 --backtest-days 31
```

- Artifact path đã chứng minh ở smoke model=2; việc còn lại là chất lượng giao dịch/lifecycle, không còn là lỗi “không xuất model”.
- Xác minh mỗi lifecycle close có thể truy ngược:

```text
event_id / sequence / trace_id
→ ma_context
→ ma_chuyen_gia
→ ma_policy
→ action wire
→ execution ledger
→ owner observe_transition/develop.
```

- `quan-tri-he-chuyen-gia promote` đã yêu cầu `--replay-pass`, `--oos-pass`, `--risk-pass` và lifecycle đủ ngưỡng, nhưng OOS frozen tự động theo từng context vẫn cần chạy/đính bằng chứng MT5 trước khi cho `TRUONG_THANH` hoặc thay checkpoint production.
- UI Java đã package lại EXE sau khi process cũ không còn chạy; cần mở lại `ui-java\dist\SelfGrowingTraderControl\SelfGrowingTraderControl.exe` để dùng bản mới.
