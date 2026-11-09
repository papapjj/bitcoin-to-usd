# Validation — 2026-06-23

## Lệnh đã chạy

```text
python -m pip install -e '.[test]'
pytest -q -o faulthandler_timeout=60
python -m compileall -q src scripts tests
python -m self_growing_trader.cli train-mt5 --help
python scripts/run_graph_mt5_smoke.py --runner-dir /tmp/nonexistent_runner
```

## Kết quả

- Packaging install: pass.
- Test suite: **200 passed in 71.29s**.
- Compile: pass.
- CLI exposes graph/child/whitelist flags: pass.
- Graph MT5 smoke helper: preflight fail-fast đúng kỳ vọng khi không có MT5 runner (`exit 2`); không launch một terminal giả.
- Graph backend is opt-in with `--m1-backend graph`; production configs default to `legacy`, and explicit graph migration/artifact repair are covered by tests.

## Không chạy trong archive

Không có executable MT5 terminal, MetaEditor hoặc tester report trong môi trường hiện tại. Do đó validation này **không** khẳng định:

- EA `.mq5` đã compile thành `.ex5`.
- Strategy Tester đã sinh bridge telemetry thật.
- HTML report thực đã parse được trong một run graph.
- Candidate graph/child đã vượt MT5 promotion baseline.

Dùng `scripts/run_graph_mt5_smoke.py --runner-dir <MT5_RUNNER>` trên máy MT5 để sinh evidence bắt buộc.
