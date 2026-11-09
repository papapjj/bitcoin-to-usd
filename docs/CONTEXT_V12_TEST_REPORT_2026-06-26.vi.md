# Báo cáo kiểm thử — Context Partition V12

## Phạm vi đã chạy

```text
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

## Kết quả

- `pytest`: **263/263 pass**. Collection xác nhận 263 test; đã chạy theo 3 nhóm không chồng lấp: `96 + 148 + 19`.
- `compileall`: **pass** cho `src` và `tests`.

## Thay đổi có regression protection

1. `ContextFrame` immutable và canonical V8 memory key.
2. Clock temporal một lần/frame; outcome không advance frame cũ.
3. Feature mask semantic theo 119-feature bridge schema.
4. Goal/account/position/execution đi vào M1 graph trước selector.
5. Legal action mask trước selection và final fail-closed policy mask.
6. Learning drive tách khỏi market opportunity.
7. Legacy latent-space strategy memory bị quarantine.
8. Parameter replacement V8 giữ context key và parameter metadata riêng.

## Giới hạn xác minh

Các test Python không mô phỏng được MetaTrader terminal, real spread/slippage/fill và reconnect behavior. Vì vậy Strategy Tester/OOS vẫn là gate bắt buộc trước deployment.
