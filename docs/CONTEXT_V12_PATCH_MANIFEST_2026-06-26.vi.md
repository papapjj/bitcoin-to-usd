# Context V12 — Patch Manifest

## Tệp nguồn thay đổi

- `src/self_growing_trader/context_frame.py` — ContextFrame, canonical key V8, semantic field slices, blockwise similarity, legal mask.
- `src/self_growing_trader/self_growing_runner.py` — semantic masks, StepTrace audit fields, temporal frame clock, canonical memory retrieval, context-aware parameter replacement.
- `src/self_growing_trader/m1_graph_module.py` — goal/portfolio/position/execution projections, legal candidate filtering, learning-drive separation.
- `src/self_growing_trader/mt5_runner.py` — goal/execution/action-legality context transfer before policy action.

## Regression tests thay đổi/thêm

- `tests/test_context_frame_v12.py`
- `tests/test_m1_graph_module.py`
- `tests/test_self_growing_runner.py`
- `tests/test_replacement_learning_v11.py` is protected by the existing suite and now passes with V8 canonical parameter replacement.

## Cách áp dụng patch

1. Sao lưu dự án hiện tại.
2. Giải nén patch vào thư mục gốc dự án và cho phép ghi đè `src/`, `tests/`, `docs/`.
3. Chạy `PYTHONPATH=src pytest -q`.
4. Với checkpoint/journal schema cũ: giữ file để audit nhưng rebuild context memory từ lifecycle journal có raw world state. Không chuyển encoded-state cũ bằng resize/padding.
5. Trước tiền thật: chạy MT5 Strategy Tester/OOS theo các gate chưa tick trong checklist.
