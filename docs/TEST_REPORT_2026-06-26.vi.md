# Báo cáo kiểm thử bản khắc phục tự phát triển

Ngày kiểm tra: 2026-06-26

## Lệnh đã chạy

```text
python -m compileall -q src
pytest -q
```

## Kết quả

```text
251 passed in 17.73s
```

## Phạm vi đã xác minh

- Collection mặc định chỉ dùng test hiện hành trong `tests/`.
- Compatibility/migration của policy graph-native.
- Mistake replay theo hướng replacement, không suppression action cũ.
- Scope action/side/lifecycle cho proposal SL/TP/risk/trailing/partial-close.
- Shadow proposal, candidate-only learning, paired validation và promotion state.
- Pure-neural causal memory, lifecycle/account metadata và causal plasticity.
- Complexity-aware selection và graph compaction chỉ sau promotion.

## Phạm vi chưa thể xác minh trong môi trường này

MT5 Strategy Tester/OOS trong terminal thật, độ trượt giá, commission, fill policy, bridge EA và walk-forward đa-regime. Các phần này được liệt kê trong `SELF_DEVELOPMENT_REMEDIATION_CHECKLIST.vi.md` như cổng bắt buộc trước khi dùng tiền thật.
