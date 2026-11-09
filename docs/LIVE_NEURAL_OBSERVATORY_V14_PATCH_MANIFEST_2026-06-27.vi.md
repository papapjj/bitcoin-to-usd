# Live Neural Observatory V14 — Patch Manifest

## Mục đích

Bổ sung giao diện quan sát trực tiếp graph M1 trong lúc train mà không can thiệp vào output policy, ContextFrame, EXP, plasticity, growth, promotion hoặc MT5 execution.

## File mới

| File | Vai trò |
|---|---|
| `src/self_growing_trader/live_observatory.py` | JSONL telemetry writer, incremental reader, local HTTP server và browser dashboard canvas. |
| `tests/test_live_observatory.py` | Regression tests cho tính read-only, event flow và config parsing. |
| `docs/LIVE_NEURAL_OBSERVATORY_V14.vi.md` | Hướng dẫn chạy, cách đọc dashboard và chẩn đoán. |

## File thay đổi

| File | Thay đổi |
|---|---|
| `src/self_growing_trader/config.py` | Thêm `LiveObservabilityConfig` vào `AppConfig` và YAML loader. |
| `configs/default.yaml` | Thêm section `observability` mặc định tắt. |
| `src/self_growing_trader/cli.py` | Thêm cờ train `--live-observatory`, `--live-events`, `--live-sample-every`, `--live-max-mb`, `--live-reset`; thêm lệnh `neural-observatory`. |
| `src/self_growing_trader/self_growing_runner.py` | Hook read-only sau V13: decision, feedback, develop, lifecycle, trial state; enable tại train MT5; tắt writer trước khi save winner. |
| `src/self_growing_trader/mt5_runner.py` | Gửi event `trial_started` và `trial_finished` từ pass MT5. |

## Contract không được phá

```text
Observer không được:
- đổi selected_action / candidate_scores / action mask
- đổi ContextFrame hay canonical context key
- đổi weight, gate, utility, plasticity, topology
- gọi MT5, gửi lệnh, sửa execution ledger
- chặn train nếu dashboard/browser/event file lỗi
```

## Cách áp patch gọn

Giải nén patch vào root dự án hiện tại, ghi đè file trùng tên. Không cần thêm package Python ngoài standard library và NumPy đã có.

## Kiểm tra sau khi áp

```bat
python -m compileall -q src tests
set PYTHONPATH=src
pytest -q tests/test_live_observatory.py tests/test_cli.py tests/test_mt5_runner.py
```

## Lệnh vận hành

Terminal A:

```bat
python -m self_growing_trader.cli neural-observatory --events outputs/neural_live.jsonl --port 8765
```

Terminal B: thêm vào lệnh train:

```bat
--live-observatory --live-events outputs/neural_live.jsonl --live-reset --live-sample-every 1
```
