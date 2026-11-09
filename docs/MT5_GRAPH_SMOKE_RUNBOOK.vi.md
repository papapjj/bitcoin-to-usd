# Runbook MT5 smoke cho Graph M1

## Mục tiêu

Tạo bằng chứng external cho checklist graph M1 bằng một run nhỏ: `rounds=1`, `trials=1`, graph backend, bridge telemetry, report parse và artifact `.pkl` load lại được.

## Chuẩn bị

1. Có portable runner MT5 chứa `terminal64.exe`, `MQL5/Files`, `MQL5/Profiles/Tester`, `MQL5/Include/compute_features.mqh`.
2. Copy `SelfGrowing_MT5_Trainer_EA.mq5` vào `MQL5/Experts/` và compile bằng MetaEditor thành `.ex5`.
3. Xác nhận EA/bridge dùng schema v4 và không đổi action schema trong run này.

## Chạy

```bash
python scripts/preflight_mt5.py --runner-dir <MT5_RUNNER>
python scripts/run_graph_mt5_smoke.py --runner-dir <MT5_RUNNER>
```

Script thứ hai tự chạy tương đương:

```bash
python -m self_growing_trader.cli \
  --config configs/self_growing.yaml \
  train-mt5 \
  --rounds 1 --trials 1 \
  --m1-backend graph \
  --runner-dir <MT5_RUNNER> \
  --output outputs/graph_mt5_smoke.pkl \
  --experience-journal outputs/graph_mt5_smoke_journal.jsonl
```

## Điều kiện pass

Kết quả JSON phải có `ok: true`, nghĩa là đồng thời:

- MT5 command exit `0`.
- Bridge có hơn một dòng (header + telemetry).
- Report tester tồn tại và không rỗng.
- Artifact output load được, `m1_backend == "graph"`, `input_size == 119`, `output_size == 20`.
- Fitness promotion gate vẫn phải kiểm tra baseline, drawdown, trade count và telemetry đầy đủ trước deploy.

Nếu `ok: false`, giữ candidate ở trạng thái reject/pending; không thay baseline production.
