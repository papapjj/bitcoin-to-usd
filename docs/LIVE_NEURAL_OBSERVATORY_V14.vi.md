# Live Neural Observatory V14

## Mục tiêu

Quan sát trực tiếp graph M1 trong lúc train mà không tạo callback vào policy, M1, plasticity, growth, MT5 bridge hoặc promotion gate.

Luồng dữ liệu là một chiều:

```text
policy.act / observe_transition / develop / lifecycle credit
→ JSONL append-only telemetry
→ local browser dashboard
```

Dashboard đọc event; nó không thể thay output action, edge weight, gate, module, context hoặc fitness.

## Những gì được hiển thị

| Thành phần | Nguồn dữ liệu thật | Ý nghĩa |
|---|---|---|
| Network topology | `NeuralGraph.neurons`, `NeuralGraph.edges` | Toàn bộ node/edge hiện có của graph M1; topology được cập nhật khi train bắt đầu hoặc sau development/growth. |
| Node sáng | `GraphActivationTrace.active_neuron_ids` | Neuron thật sự nằm trong trace quyết định hiện tại. |
| Đường xanh | `GraphActivationTrace.active_edge_ids` | Synapse thật sự được trace khi M1 chọn hypothesis/action. |
| Activation/state | `NeuronNode.activation`, `NeuronNode.state` | Kích hoạt tức thời và working/recurrent state của node. |
| Candidate cạnh tranh | `candidate_scores`, `candidate_actions`, `selected_idx` | Hypothesis nào đang cạnh tranh, hypothesis nào thắng. |
| Context | `ContextFrame` audit, action mask, account, learning drive | Context bất biến dùng cho đúng decision đang xem. |
| EXP | `expectancy_predicted_lcb`, mean net-R, sample, regret | Giá trị expected net-R bảo thủ đã đi vào M1. |
| Feedback | `reward`, `cost`, `td_delta`, execution acknowledgement | Policy có nhận credit thật hay chỉ là intent/fill chưa kết thúc lifecycle. |
| Growth | `growth_events` | Mọc neuron/edge, plasticity, replay, consolidation, fade/prune, candidate outcome. |
| Trial | MT5 rank fitness, trades, DD, expectancy, reconciliation | Chất lượng candidate sau một pass MT5. |

## Cách chạy

### Terminal 1: mở dashboard

```bat
python -m self_growing_trader.cli neural-observatory --events outputs/neural_live.jsonl --port 8765
```

Dashboard mở tại `http://127.0.0.1:8765/`.

### Terminal 2: train có telemetry

Thêm các cờ sau vào lệnh train hiện tại:

```bat
--live-observatory --live-events outputs/neural_live.jsonl --live-reset --live-sample-every 1
```

Ví dụ với MT5 Visual:

```bat
python -m self_growing_trader.cli --config configs/default.yaml train-to-goal ^
  --train-rounds 1 --attempts 1 --max-rounds 100 ^
  --seed-policy outputs/winner.pkl --output outputs/winner.pkl ^
  --log outputs/train_to_goal.jsonl --backtest-days 31 --trials 1 ^
  --runner-dir mt5_v2_runner --timeout-sec 900 --mt5-visual ^
  --mt5-close-scope trial --curriculum-learning ^
  --experience-journal outputs/mt5_experience_journal.jsonl --replay-journal-max 3 ^
  --max-modules 100 --mutation-sigma 0.3 --model 2 --m1-backend graph ^
  --policy-core current --from-date 2025.01.03 --to-date 2025.02.03 ^
  --min-daily-trades 6 --live-observatory ^
  --live-events outputs/neural_live.jsonl --live-reset --live-sample-every 1
```

`--mt5-visual` và Observatory là hai cửa sổ khác nhau:

- MT5 Visual: giá, lệnh, position, tester.
- Observatory: node/edge, trace, context, EXP, growth, credit, fitness.

## Diễn giải đúng dashboard

### Decision event

Một `decision` được ghi sau khi V12 ContextFrame, legal-action mask và V13 expectancy đã hoàn thành. Vì vậy đây là output cuối mà bridge dùng, không phải action trước mask.

- `active node/edge > 0`: M1 graph đã nhận request và đã chạy.
- `allowed_actions`: action hợp lệ trước selector.
- `candidate_scores`: điểm của các hypothesis cạnh tranh.
- `EXP LCB`: edge expectancy bảo thủ của action được chọn.
- `memory_match`: mức retrieval context-memory.

### Feedback event

`feedback` xuất hiện khi request kế tiếp giúp bridge xác nhận intent/fill/close. Nó chưa luôn là lifecycle credit:

- `action_feedback_eligible=false`: fill/intent chưa đóng, không được TD/plasticity credit ngay.
- `action_feedback_eligible=true`: close đã được xác nhận, credit có thể được áp.

### Growth event

Module/neuron/edge mới không phải mặc định là “đã học được”. Dashboard phải cho thấy chuỗi:

```text
repeated context evidence
→ graph_trial_micro_growth / candidate event
→ candidate activity trong context gần giống
→ lifecycle/EXP evidence
→ consolidation hoặc fade/prune
```

Một growth event đơn lẻ chưa phải bằng chứng module hữu ích. Bằng chứng giữ module là EXP LCB delta và promotion qua MT5/OOS.

## Khi dashboard trống

| Dashboard | Chẩn đoán |
|---|---|
| Chỉ có `run_started`, không có `decision` | Python train chưa tới bridge action loop hoặc MT5 không gửi request. |
| Có `trial_started` nhưng 0 decision | EA không OnInit/OnTick, symbol/history không sẵn sàng hoặc file bridge sai path. |
| Có decision nhưng active node/edge = 0 | Cần kiểm tra M1Graph trace/backend; đây là lỗi graph observability. |
| Có decision nhưng no trade | Có thể policy HOLD/risk gate/action legality; xem candidate scores, allowed actions và `EXP LCB`, không kết luận ngay bridge lỗi. |
| Có growth nhưng topology không đổi | Event có thể là plasticity/replay/consolidation không thêm node/edge. |

## Bandwidth

`--live-sample-every 1` ghi mọi decision. Với M1/backtest dài, event file sẽ lớn; dùng `--live-max-mb N` để compact tail khi vượt N MiB. Giới hạn này áp dụng cho event JSONL chính và các log phụ như hồ sơ quyết định, sự kiện cấu trúc, `theo_doi_phat_trien_mang`. Mặc định `0` nghĩa là giữ audit đầy đủ, không compact; khi chạy smoke MT5 với model sâu nên đặt quota rõ ràng, ví dụ `--live-max-mb 32 --live-sample-every 50`.

Với `huan-luyen-he-chuyen-gia-mt5`, nếu không truyền `--to-date` thì `--backtest-days` tự tính ngày kết thúc từ `--from-date`. Smoke nhanh nên dùng `--backtest-days 1` hoặc truyền rõ cả `--from-date/--to-date`. Nếu chỉ muốn chặn tổng thời gian chạy của smoke, dùng thêm `--max-wall-sec N`; cờ này không thay thế `--timeout-sec` vì `--timeout-sec` vẫn là giới hạn khi bridge không có request mới.

Dùng các mức sau khi cần:

```text
1  : debug trực tiếp từng request
5  : xem xu hướng, giảm I/O
20 : backtest dài, tập trung growth/trial
```

Telemetry không chứa callback xử lý UI và không làm training chờ dashboard. Nếu browser bị đóng hoặc event file không đọc được, train vẫn tiếp tục.
