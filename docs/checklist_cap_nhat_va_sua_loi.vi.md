# Checklist cập nhật và sửa lỗi — Self-Growing Trader

> **Tình trạng tài liệu:** historical changelog. Không dùng như release gate duy nhất. Xem `docs/MASTER_COMPLETION_CHECKLIST.vi.md` để biết trạng thái audit hiện tại, các blocker P0/P1 và tiêu chí hoàn thành.


**Trạng thái kiểm chứng:** `102 passed` (`pytest -q`) và `python -m compileall -q src tests` thành công.

## 1. Fitness MT5 và selection objective

- [x] **Loại bỏ win-rate gate nhân với score có dấu.** Policy có `win_rate = 0%` không còn nhận `-0.0`; nó nhận phạt âm rõ ràng và không thể vượt policy mất ít hơn.
- [x] **Sửa thứ tự PF < 1.** Penalty dùng `max(0, 1 - PF)`; PF `0.9` luôn ít bị phạt hơn PF `0.5` trong các điều kiện còn lại như nhau.
- [x] **Chuẩn hóa objective.** Expectancy, PF log-edge, drawdown, tail proxy, cadence và complexity đều là đại lượng không phụ thuộc lot-size.
- [x] **Không còn trộn `avg_loss` theo tiền với expectancy-R.** Tail proxy chỉ dùng tỷ lệ loss/negative-trade, không dùng USD tuyệt đối.
- [x] **Tách score xếp hạng và tín hiệu học.** `mt5_fitness()` dùng cho ranking; `mt5_learning_signal()` giới hạn tín hiệu neural trong `[-1, 1]` trước khi cập nhật utility, plasticity và developmental controller.
- [x] **Hard penalties rõ ràng.** Không trade, PF < 1, net profit <= 0, DD cao, và thiếu mẫu đều bị phạt trực tiếp; không dùng multiplier có thể đảo thứ hạng.
- [x] **Complexity topology-aware.** Fitness tính cả số module, hidden neuron, local active edge, directed lateral edge và memory node.
- [x] **Promotion cần sample thực.** Caller production truyền `config.orders.min_daily_trades`; số trade tối thiểu được kiểm tra theo số ngày backtest.
- [x] **Goal không thể được hit bởi candidate bị reject.** Trial chỉ có thể đạt goal sau khi đã được promotion.
- [x] **Goal dùng cadence config theo mặc định.** `train_to_goal(..., min_daily_trades=None)` lấy `config.orders.min_daily_trades`, không mặc định 0.

> Ghi chú dữ liệu: CVaR chính xác cần chuỗi closed-trade/equity curve từ MT5. Hiện objective dùng tail-risk proxy chuẩn hóa, không tuyên bố là CVaR chính xác.

## 2. Fitness quay về neuron và connection

- [x] **Module-level utility vẫn giữ lại** để duy trì lifecycle/grow-prune cũ.
- [x] **Neuron credit được lưu trong `neuron_utility`.** Credit dùng bounded learning signal và activation của neuron.
- [x] **Synapse credit có eligibility trace.** `w_in`, `w_rec`, `w_out` được cập nhật từ tín hiệu fitness đã giới hạn, với routing mask được tôn trọng.
- [x] **Directed lateral edge cũng nhận credit.** Mỗi cạnh `(target <- source)` được cập nhật từ activity và credit của hai module.
- [x] **Raw MT5 score không còn làm bùng utility.** Các cấu trúc neural không nhận trực tiếp score scale lớn.

## 3. Journal replay và isolation hai core

- [x] **Journal có giới hạn dung lượng.** `experience_journal_max_bytes` mặc định 64 MiB; compaction chạy cả trước và sau append.
- [x] **Replay không đọc toàn file.** Chỉ đọc tail bounded theo số byte/số dòng, chọn top-k bằng heap; không giữ toàn bộ journal trong RAM.
- [x] **Current/World dùng journal riêng.** Compare mode tạo `journals/<base>.current.jsonl` và `journals/<base>.world.jsonl`.
- [x] **Không còn cross-contamination qua validation.** Validation replay và OOS evaluation mặc định tắt journal.
- [x] **Exchange giữa core vẫn explicit.** Chỉ positive replay đã qua validation mới được chuyển qua cơ chế exchange.

## 4. Growth topology, M1 và lateral connection

- [x] **`expand_module()` được đưa vào developmental path thật.** Khi module bão hòa hoặc module count đã chạm cap, action `grow` có thể mở rộng hidden neuron thay vì chỉ clone module.
- [x] **Resize M1 đồng bộ.** State encoder, working memory, hypothesis bank, predictor, critic, runtime memory và eligibility state được resize khi hidden size tăng.
- [x] **Lateral edge heterogeneous-size.** Dùng ma trận directed per-pair `W[target <- source]` kích thước `(h_target × h_source)`; không còn lỗi matmul khi module có hidden size khác nhau.
- [x] **Lateral integration tác động output ở step hiện tại.** Cross-module state ảnh hưởng residual output và recurrent state tiếp theo.
- [x] **Neural spec là topology thật.** `input_sources` và `output_targets` trở thành input/output mask; weight ngoài vùng cho phép được zero hóa.
- [x] **Grow/split/merge/prune repair lateral topology.** Sau thay đổi module, directed edge map được resize/tái tạo nhất quán.

## 5. Validation, packaging và CI

- [x] **Walk-forward OOS là evaluation-only.** Validation không gọi `train_self_growing_mt5()`, không replay và đánh giá trên policy copy.
- [x] **OOS failure không được thay bằng in-sample score.** Window lỗi nhận `-inf`, tránh làm đẹp robustness giả.
- [x] **Root pytest được cô lập.** `pytest.ini` đặt `testpaths = tests`, không còn collect script legacy trong `work/`.
- [x] **Project đã có packaging tối thiểu.** Thêm `pyproject.toml`, dependencies và entry point `self-growing-trader`.

## 6. Regression tests đã thêm

- [x] Fitness ordering: zero-win < PF 0.5 < PF 0.9 < profitable.
- [x] Lot-scale invariance cho structural objective.
- [x] Bounded, complete JSONL journal compaction.
- [x] M1 resize + lateral edge cho hidden sizes 7 × 5 và 5 × 7.
- [x] Fitness credit làm thay đổi synapse/neuron nhưng tín hiệu vẫn bounded.
- [x] Saturated module gọi expansion neuron thực.
- [x] Specialist routing masks có hiệu lực trên weight thật.
- [x] Compare core nhận journal path khác nhau.
- [x] Trial bị reject không thể kết thúc `train_to_goal`.
- [x] OOS evaluator dùng policy copy và tắt journal.

## Lệnh kiểm chứng

```bash
python -m pip install -e . --no-deps
pytest -q
python -m compileall -q src tests
python -m self_growing_trader.cli --help
```

## 7. Hotfix: action intent không được xem là execution

- [x] **Bridge request không còn tự động là experience.** Mỗi bar vẫn có thể gọi `observe_transition()`, nhưng MT5 action intent không được cấp reward/TD credit khi EA không xác nhận thực thi.
- [x] **Structural memory yêu cầu execution evidence.** `MemoryNeuron`, causal node, cluster, connection và experience buffer chỉ được tạo khi có position/order/trade-count change hoặc EA acknowledgement.
- [x] **Journal không lưu action không được xác nhận.** Replay cũng bỏ qua journal MT5 schema cũ không có `execution_details`, ngăn dữ liệu lỗi cũ quay lại làm phình memory.
- [x] **Không có phát triển topology trong lifecycle.** Bridge chỉ settlement credit đã xác nhận; không gọi structural-development hook.
- [x] **Schema bridge v3 được hỗ trợ.** Có `orders_pipe` và `execution_pipe`; EA v1/v2 vẫn chạy theo cơ chế xác nhận conservative từ position/day-trade snapshot.
- [x] **Regression tests bổ sung.** `106 passed`: unconfirmed bridge request không tăng structural memory hoặc journal; confirmed execution vẫn ghi memory.

Chi tiết protocol cho EA: `docs/bridge_execution_ack.vi.md`.
