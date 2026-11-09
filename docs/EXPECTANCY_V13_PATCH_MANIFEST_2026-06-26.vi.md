# Patch manifest — Expectancy V13

## File mới

- `src/self_growing_trader/contextual_expectancy.py`
- `tests/test_contextual_expectancy_v13.py`
- `docs/EXPECTANCY_V13_REQUIREMENTS_AND_IMPLEMENTATION_CHECKLIST_2026-06-26.vi.md`
- `docs/EXPECTANCY_V13_TEST_REPORT_2026-06-26.vi.md`
- `docs/EXPECTANCY_V13_PATCH_MANIFEST_2026-06-26.vi.md`

## File cập nhật

- `src/self_growing_trader/self_growing_runner.py`
- `src/self_growing_trader/m1_graph_module.py`
- `src/self_growing_trader/mt5_runner.py`
- `src/self_growing_trader/config.py`
- `configs/self_growing.yaml`

## Thứ tự áp patch

1. Sao lưu checkpoint hiện tại và journal.
2. Chép toàn bộ patch, giữ nguyên cấu trúc relative path.
3. Cài đặt không cần dependency Python mới.
4. Chạy:

```text
set PYTHONPATH=src
pytest -q
python -m compileall -q src
```

5. Chạy một MT5 training window tách biệt với journal mới.
6. Đọc `fitness_policy_*` và `scope_preview` trước khi cho candidate thay seed.

## Migration checkpoint

- V13 không cố resize/diễn giải latent memory cũ như canonical context expectancy.
- Checkpoint cũ load được; ledger V13 bắt đầu rỗng nếu chưa có `contextual_expectancy` hợp lệ.
- Chỉ replay lifecycle journal có đầy đủ canonical context, action provenance, final net-R và risk metadata.

## Rollback

Muốn rollback: dùng lại archive Context V12 trước đây cùng checkpoint đã tạo tại thời điểm V12. Không trộn source V12 với checkpoint V13 sau khi V13 ledger đã tích lũy bằng chứng mới nếu mục tiêu là reproducibility.
