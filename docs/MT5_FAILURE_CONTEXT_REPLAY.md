# MT5 Failure Context Replay

## Mục tiêu

Đưa failure lịch sử quay lại **MT5 Strategy Tester** bằng current policy cho tới khi policy không còn lặp hành động lỗi. Không feed lại old PPO tuple `(state, action, reward, log_prob)`.

## Luồng dữ liệu

1. Một entry policy-origin settle với `realized_r <= five_failure_replay_admit_threshold_r`.
2. Failure bank lưu causal packet và MT5 anchor: request time, symbol, period, event/sequence, failed action và context key.
3. Replay scheduler chọn unresolved cases theo bucket/context-action, severity và fairness theo số attempt trong run.
4. Mỗi case tạo một Strategy Tester session lịch sử độc lập:
   - tester chạy từ ngày chứa `anchor - warmup`;
   - EA không mở bridge trước replay window;
   - từ `anchor - warmup` đến `anchor + horizon`, **current policy** nhận tick/features MT5 và tạo action/log-prob mới;
   - EA gọi `TesterStop()` sau replay horizon.
5. Fresh MT5 transitions đi qua reward/lifecycle/PPO hiện tại như normal training.
6. Sau session, current policy audit lại failed-action probability trên causal failure packet:
   - <= repair threshold đủ số streak => parent case resolved;
   - >= reopen threshold ở audit sau => reopen.
7. Nếu replay bỏ lỗi A nhưng phát sinh loss bằng action B, B được admit thành child failure dedupe theo `(parent, context, action)`. Nếu A lặp lại đúng, parent được reinforce thay vì tạo duplicate vô hạn.

## Tính on-policy

Replay MT5 không dùng old action, old reward hoặc old behavior log-prob cho PPO. Old failure chỉ cung cấp **historical market/time anchor và curriculum identity**. Action/reward/log-prob dùng để train đều được sinh mới từ current policy trong Strategy Tester session replay.

## Vì sao dùng session MT5 riêng

Strategy Tester không có API an toàn để seek ngược về một tick cũ trong cùng pass. Vì vậy replay là session độc lập. EA dùng intraday gate để chỉ mở bridge trong replay window; Tester FromDate/ToDate vẫn là day-granularity.

## Không phải snapshot account restore tuyệt đối

Bản này replay **historical market slice** trên MT5 với current policy từ đầu warmup. Nó không cố inject lại ticket/position/account snapshot cũ vào Strategy Tester, vì việc đó không phải state restore được MT5 hỗ trợ đáng tin cậy. Account/positions trong replay được current policy tái tạo trong warmup. Đây là curriculum replay, không phải deterministic emulator của exact old account state.

## Replay budget mặc định V193

- `five_mt5_failure_replay_sessions_per_run: 12`
- `five_mt5_failure_replay_max_attempts_per_case: 6`
- `five_mt5_failure_replay_warmup_minutes: 120`
- `five_mt5_failure_replay_horizon_minutes: 180`
- `five_mt5_failure_replay_resolved_audit_items: 16`

Một phần nhỏ replay chạy trước normal pass nếu bank đã tồn tại; phần còn lại chạy sau normal pass để failures mới của chính run hiện tại có thể được thi lại ngay. Nếu post replay cập nhật weights, hệ thống bắt buộc chạy một normal-window frozen validation trên clone để fitness/metrics khớp checkpoint được lưu.

## Telemetry

Replay journal: `<candidate>.mt5_failure_replay.jsonl`.

Schema replay hiện tại là v2. Ngoài observation, hidden/cell và previous action,
mỗi failure anchor còn giữ đủ bốn temporal token 2/8/32/128 cùng validity mask
của từng agent. Bank schema v1 bị xóa khi migrate vì không thể phục dựng chính
xác lịch sử Transformer; dùng state rỗng sẽ làm audit/gradient replay sai ngữ
cảnh quyết định gốc.

Mốc neo được xác nhận tại yêu cầu MT5 đầu tiên có thời gian bằng hoặc lớn hơn
mốc neo trong cửa sổ phát lại. Cách này xử lý đúng luồng tick thật có thể không
có tick ở chính xác phút neo; điều kiện bằng tuyệt đối theo phút đã bị loại bỏ.

Mỗi session ghi:
- case/parent/context/action;
- replay anchor/from/to;
- status, fitness, net profit, trades;
- anchor_seen;
- audit/result bank counters.

Failure bank lưu thêm:
- `mt5_replay_count`;
- `mt5_replay_success_count`;
- `last_mt5_replay_fitness`;
- `last_mt5_replay_net_profit`;
- `origin` và `parent_case_id`.

## Deployment MT5

EA source đã đổi input contract nên phải compile lại:

`mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5`

thành `.ex5` bằng MetaEditor trên máy Windows/MT5 đích. Runner có `fail_on_stale_ex5=True`, vì vậy source mới + binary cũ phải fail closed.

## Safety/integrity invariants

- no stale PPO replay;
- no old reward replay;
- replay exact parent failure được dedupe/reinforce;
- alternative replay failure tạo bounded child case;
- resolved case không bị xóa và có thể reopen;
- delayed EA action không được phép execute ngoài replay window;
- metrics sau post replay được refresh trên frozen clone để khớp saved weights;
- failure bank bounded bởi configured capacity.
