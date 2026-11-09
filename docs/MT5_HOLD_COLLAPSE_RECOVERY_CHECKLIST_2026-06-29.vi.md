# Checklist khôi phục train khi SELL -> SL rồi HOLD kéo dài

> Ghi chú cập nhật 2026-07-13: bootstrap BUY/SELL trong checklist này là hành vi lịch sử của recovery bridge. Bridge recovery vẫn chỉ dùng `CANCEL_ORDERS`; riêng profile train có `train_entry_probe` thì có thể tạo entry thăm dò có nhãn `policy_exploration`. Probe không được tính là policy expectancy, nhưng lifecycle đóng thật được nối về action-memory của đúng expert để học entry.

## Hiện tượng runtime đã xác nhận

- [x] Process train vẫn sống: Python chạy `walk-forward --rounds 3 --trials 5 --policy-core world --m1-backend graph`, MT5 Visual Tester vẫn phát request.
- [x] Bridge không chết: `self_growing_bridge.csv`, `self_growing_request.csv` và `self_growing_action.csv` tiếp tục cập nhật.
- [x] Candidate đầu phiên mở chuỗi `MARKET_SELL`, sau đó 10 lệnh đóng đều `SL`, net khoảng `-959`.
- [x] Sau chuỗi lỗ, 500 bridge event gần nhất đều là `action=0` với balance/equity đứng yên, không tạo lifecycle mới.
- [x] `neural_live.jsonl` không còn phản ánh lượt MT5 đang chạy dù CLI có `--live-observatory`.
- [x] Sau patch bootstrap, run mới đã tạo nhiều dữ liệu hơn: 49 lệnh fill/close, nhưng 200 bridge event gần nhất chuyển sang `response_status=timeout` và không có execution mới sau seq `1463`.

## Nguyên nhân kỹ thuật

- [x] Mục lịch sử: config bootstrap exploration đã bị xóa khỏi runtime bridge.
- [x] `min_daily_trades=0` khiến bootstrap cũ bị chặn ngay cả khi được bật, nên không có probe để thoát trạng thái HOLD rỗng trong MT5 training.
- [x] `ExperienceStarvationState` chỉ tăng `no_trade_streak` sau entry intent không fill; flat `HOLD` bị firebreak đúng cách nhưng cũng làm policy không thấy recovery context nào trong train thường.
- [x] Live writer được copy sang candidate cùng decision counter/topology snapshot, dễ làm candidate hiện tại không emit đúng nhịp hoặc không phát topology đầu trial.
- [x] EA mở CSV không dùng `FILE_SHARE_READ | FILE_SHARE_WRITE`, trong khi Python ghi action bằng temp replace; dưới tải live, `self_growing_action.csv` có thể bị khóa độc quyền và EA timeout trước khi thấy response.
- [x] `ActionWaitMs=1500/5000` quá sát với bridge Python khi policy/observatory/journal đang bận; thực tế 5000 ms vẫn tạo timeout vì Python đọc request muộn sau khi xử lý feedback/develop của request trước.
- [x] Legacy close credit dùng biến động equity toàn tài khoản trước `profit/initial_risk`, nên có thể ghi reward dương cho một lệnh `SL` nếu lệnh khác cùng lúc kéo equity tăng.
- [x] Action recovery context-local ghi nhận SELL loss nhưng khi chưa có BUY dương đủ mạnh thì vẫn để SELL lặp tiếp; cần fallback HOLD cục bộ cho action đã âm rõ.
- [x] Close do `SYSTEM/max_holding` có thể ghi `ack_sequence=0`; nếu không map ngược bằng `position_ticket`, positive BUY/SELL close bị mất credit dù entry policy hợp lệ.
- [x] Mistake memory có thể ghi sai `action_index=0` cho lệnh MT5 thật nếu `action_details` chỉ có `action_name`, còn index thật nằm trong `execution_details.requested_action`; hậu quả là mistake episode phạt HOLD thay vì phạt `MARKET_BUY/SELL`.
- [x] Config train mặc định để `orders.max_daily_trades=0` và `risk.enforce_entry_limits=false`, nên policy có thể spam pending entry (`SELL_STOP`) không giới hạn trước khi learner recovery kịp chặn.
- [x] Pending-order ledger có thể ghi `initial_risk=0` cho một số `SELL_STOP` fill/close; nếu Python dùng equity fallback thì SL risk=0 bị reward sai scale.
- [x] `decision` live event được emit ngay sau `policy.act()` nên chưa có execution context từ MT5 bridge; action thật sau mask/bootstrap nằm ở `bridge_heartbeat`/`feedback`, làm UI dễ đọc nhầm `HOLD/-1`.
- [x] Bootstrap chọn side bằng `request.sequence % 2`; khi cadence MT5 rơi vào sequence chẵn liên tiếp, probe bị lệch thành BUY lặp lại. Run 2026-06-30 16:18 ghi `MARKET_BUY` thua `7/7`, net `-760.96`, trong khi `MARKET_SELL` net `+288.47`.

## Sửa đã áp dụng

- [x] Thêm `flat_idle_streak` và `observe_flat_idle()` trong bridge state.
  - Chỉ là recovery context nội bộ của MT5 Strategy Tester.
  - Không gọi `observe_transition`.
  - Không tạo reward, memory, causal credit hoặc lifecycle giả.
- [x] Khi pending trace gặp `FLAT_IDLE`, bridge tăng recovery context nếu không ở continuous firebreak.
- [x] Mục lịch sử: bootstrap fallback BUY/SELL từng không phụ thuộc `min_daily_trades > 0`. Đã thay thế ngày 2026-07-01; bridge hiện chỉ recovery bằng `CANCEL_ORDERS` cho pending treo.
- [x] Mục lịch sử: bootstrap market-entry đã bị loại khỏi đường bridge hiện tại; workflow train dùng policy owner cho entry và pending recovery cancel để gỡ lệnh chờ treo.
- [x] Live telemetry writer copy reset sampling/topology state cho candidate mới, để trial/candidate hiện tại tự emit telemetry của nó.
- [x] Thêm `action_wait_ms=15000` vào `MT5BacktestConfig` và xuất `ActionWaitMs=15000||15000||0||60000||N` vào tester `.set`; đồng bộ default EA source và cap `MathMin(ActionWaitMs, 60000)`.
- [x] Thêm `FILE_SHARE_READ | FILE_SHARE_WRITE` cho các hàm EA mở CSV read/write/append để Python và MT5 không khóa chéo action/request/bridge file.
- [x] Khi settle close từ execution ledger, reward ưu tiên PnL của chính lệnh: `(profit + commission + swap) / initial_risk`, clip theo R; equity toàn tài khoản chỉ còn fallback.
- [x] Action recovery thay `MARKET_BUY/SELL` có LCB âm cục bộ bằng HOLD nếu chưa có replacement dương đủ mạnh, không tạo global HOLD penalty.
- [x] Action recovery nhận lifecycle thật ngay trong `observe_transition()` cho legacy MT5 close path, không chỉ qua `apply_execution_lifecycle_credit()`.
- [x] Legacy close settlement fallback theo `position_ticket` khi ledger close có `ack_sequence=0`, để close `source=max_holding` vẫn credit đúng entry trace/action.
- [x] Mistake memory ưu tiên `execution_details.bridge_action/requested_action/action` trước khi fallback sang argmax output, để MT5 ledger loss ghi đúng action index.
- [x] Giới hạn pre-launch journal replay: workflow MT5 mặc định chỉ chọn tối đa 64 mẫu stratified, trong đó 8 mẫu ưu tiên cao chạy full `observe_transition`/graph plasticity; phần còn lại absorb vào memory bằng signal trong journal.
- [x] Mục lịch sử: daily entry cap từng đặt YAML MT5/train `min_daily_trades=6`, `max_daily_trades=12`. Đã thay thế ngày 2026-07-01; bridge không còn dùng min/max daily trade để khóa entry.
- [x] Reward fallback cho close `SL` có `initial_risk=0`: nếu profit âm, suy ra risk bằng `abs(net_profit)` để vẫn ghi khoảng `-0.10R` thay vì reward nhỏ theo equity.
- [x] Live telemetry thêm `topology_summary` vào `decision`, `feedback`, `development`, `lifecycle`, `trial_finished` để theo dõi trực tiếp tổng neuron, edge, specialist module, region counts mà không phải parse full topology event.
- [x] Live decision action không còn mơ hồ: payload tách `policy_action`, `legal_action`, `emitted_action`, `policy_name`, `legal_name`, `emitted_name`, `execution_bootstrap`, `action_masked`.
- [x] `bridge_heartbeat` sau `action_written` thêm `action_name`, `wire_action_name`, `policy_action_name`, `legal_action_name`, `emitted_action_name`, `execution_bootstrap`, `action_masked`, `topology_summary` và `execution_action_event=true`; đây là nguồn sự thật sau bridge cho UI/live monitor.
- [x] Mục lịch sử: bootstrap side không còn tồn tại; evaluator hiện chỉ theo dõi tuổi pending snapshot để phát `CANCEL_ORDERS`.
- [x] Thêm train entry guard:
  - `entry_guard_single_position=true` trong `configs/default.yaml` để MT5 train không scale-in nhiều vị thế trước khi lệnh trước có feedback.
  - `entry_loss_guard_enabled=true` với `min_closed=6`, `max_closed_loss_r=0.25`, `consecutive_losses=4` để chặn entry mới sau ngày xấu.
  - Guard chỉ xóa action entry khỏi mask; không chặn HOLD, cancel, manage hoặc close.
- [x] Live heartbeat `request_seen` xuất `entry_loss_guard` để UI thấy ngày, số close, net R và chuỗi loss; đã vá cả nhánh continuous feedback để counter không đứng 0 sau close.
- [x] Thêm runtime fitness guard trong bridge train:
  - Đọc execution rows ngay trong lúc Strategy Tester đang chạy, không chờ hết trial mới tính fitness.
  - Chặn entry mới khi rolling fitness dưới ngưỡng, tỷ lệ đóng quá sớm quá cao, lỗ ngày vượt R, số close/ngày quá lớn hoặc drawdown chạm ngưỡng.
  - Guard chỉ xóa action entry khỏi mask; vẫn cho HOLD, cancel, manage và close để hệ thống thoát vị thế/pending đang có.
  - Heartbeat `request_seen` và `action_written` xuất `runtime_fitness_guard` để debug lý do khóa entry.
- [x] Đã thay countdown xác nhận context bằng router M15 trực tiếp; state M15 đã đóng đổi owner ngay, còn M1 chỉ dùng quyết định/execution/timing.

## Guard không được phá

- [x] Valid policy `MARKET_BUY`/`MARKET_SELL` không bị bridge recovery ghi đè.
- [x] `HOLD` vẫn là no-op với EA; flat idle không trở thành action feedback.
- [x] Continuous mode vẫn có thể suppress flat-idle starvation; recovery bridge không còn dùng cờ bootstrap market-entry cũ.
- [x] Recovery bootstrap chỉ là `CANCEL_ORDERS` để giải phóng pending treo, không phải entry hoặc lợi nhuận giả.
- [x] Action wait vẫn bị clamp tối đa 60000 ms trong EA, không biến bridge thành chờ vô hạn.

## Việc cần xác thực bằng MT5 run mới

- [x] Restart train sau patch, vì process đang chạy đã load code/config cũ.
- [x] Kiểm tra sau restart `ActionWaitMs=15000`: bridge 42 event đầu đều `response_status=ok`, `timeout=0`.
- [x] `self_growing_execution.csv` có lifecycle thật: 16 fill, 15 close, net tạm `+188.02` ở mẫu 2026-06-30 14:58.
- [x] `outputs/neural_live.jsonl` cập nhật `decision`, `feedback`, `development`; feedback có `graph_credit_attribution`, `neuron_action_replacement_credit`, `mistake_episode_recorded`.
- [x] Reward ledger mới đúng dấu: các `SL` mới ghi khoảng `-0.10R`, `TP` ghi dương theo `profit/initial_risk`.
- [x] Positive BUY evidence không còn mất: run 2026-06-30 15:34 ghi seq `46` `MARKET_BUY` TP `+263.26`, journal action `1`, reward `+0.2`, `ledger_ack_sequence=46`.
- [x] Regression test khóa lỗi mistake-index: case `MARKET_SELL` với `requested_action=2` và selected output HOLD vẫn tạo episode `action_index=2`.
- [x] Run cũ sau fix reward nhưng trước fix mistake-index: bridge `150/150 ok`, `13` fill/close, `10 SL / 3 TP`, net `-486.21`; process đã dừng để tránh tiếp tục ghi dữ liệu bằng logic sai.
- [x] Regression test khóa replay-budget: `max_items=3`, `plastic_max_items=1` chỉ gọi full plasticity 1 lần nhưng vẫn absorb memory 3 lần.
- [x] Benchmark journal hiện tại: 64 mẫu memory-only khoảng `12.3s`, 64 mẫu với 8 full-plasticity khoảng `37.9s`; mức cũ 200 mẫu với full plasticity quá chậm cho vòng MT5 live.
- [x] Mục lịch sử: run sau replay-budget từng spam `SELL_STOP` khi không có guard theo feedback. Daily cap đã bị loại bỏ; lỗi này phải xử lý bằng policy/recovery/guard theo evidence, không khóa theo số lệnh/ngày.
- [x] Mục lịch sử: run sau daily cap từng chuyển về `HOLD` khi `day_trades=12`. Hành vi này đã bị loại bỏ vì làm train dừng entry dù bridge còn sống.
- [x] Regression test khóa reward risk=0: pending `SL` profit `-97.93`, `initial_risk=0` trả reward `-0.1`.
- [x] Run sau reward fallback: `179/179` bridge `ok`; action sau cap `HOLD=151`; ledger `12` close, net `+436.07`, `TP=4/SL=6/EXPERT=2`, `zero_risk=0`.
- [x] Live learning evidence: `mistake_episode_recorded` ghi `action=2 MARKET_SELL`, không còn phạt nhầm `action=0 HOLD`; close `SYSTEM/EXPERT` đã map về action gốc qua `ledger_ack_sequence=0` nhưng `ack_sequence` đúng.
- [x] Growth evidence: có `graph_candidate_proposal` memory specialist `+9 neuron/+41 edge` và runtime `graph_trial_micro_growth` memory specialist candidate `+9 neuron/+41 edge`; graph credit cập nhật nhiều đợt `34-51 edge`, `9-18 neuron`.
- [x] Run sau topology-summary patch xác nhận live event có `topology_summary`: `322 neuron`, `8599 edge`, `1` memory specialist module candidate, region counts/graph counts đầy đủ trong `decision`, `feedback`, `development`.
- [x] Regression test khóa action telemetry: decision có `action_name=HOLD` nhưng `emitted_action=2` phải hiển thị `name/emitted_name=MARKET_SELL`, `policy_name=HOLD`.
- [x] Run lịch sử 2026-06-30 16:18 trước patch mới: bridge `146/146 ok`, action `HOLD=134/BUY=8/SELL=4`; ledger `12` close, net `-492.61`, `SL=9/TP=2/EXPERT=1`; lỗi chính là bootstrap BUY lặp sau cap, không phải bridge chết.
- [x] Regression test khóa heartbeat execution telemetry: policy heartbeat `action_written` với `policy_action=HOLD`, `emitted_action=MARKET_BUY` phải giữ đúng `policy_action_name=HOLD`, không được nuốt action `0`.
- [x] Mục lịch sử: regression test bootstrap parity đã bị thay bằng test pending recovery cancel; bridge hiện không còn phát probe `MARKET_BUY`/`MARKET_SELL`.
- [x] Run sau heartbeat patch đã nạp code mới: process `python=20448`, `terminal64=24292`; bridge `97/97 ok`, live status `last_error=""`, heartbeat có `action_name` và tên `wire/policy/legal/emitted`.
- [x] Mục lịch sử: run đã xác nhận bootstrap BUY không còn lặp vô hạn. Hành vi hiện tại đi xa hơn: bridge không còn bootstrap market entry.
- [x] Daily entry cap đã bị gỡ khỏi bridge: `day_trades` không còn làm `allowed_actions` chỉ còn `HOLD`.
- [ ] Lỗi hành vi còn mở: policy SELL vẫn lặp đủ cap trước khi recovery thắng selection; snapshot `12` close, net `-285.14`, `SELL=11` net `-193.32` (`8 SL / 3 TP`), `BUY=1` net `-91.82`.
- [x] Run sau single-position guard: bridge `41/41 ok`, `max_positions_seen=1`, `max_day_trades=6`, close `6`, net `+646.64`, `TP=4/SL=2`; không còn cụm scale-in SELL trước khi có feedback.
- [x] Live growth vẫn chạy sau guard: `feedback=30`, `development=12`, `learner_recovery_lifecycle=6`; topology summary tăng lên `331 neuron`, `8640 edge`, `2` specialist modules.
- [ ] Run mới sau loss-guard counter patch phải xác nhận `entry_loss_guard.closed_count/net_r/consecutive_losses` khác 0 khi có close trong continuous path.
- [x] Lý do không thấy `outputs/mt5_trained.pkl`: train chỉ ghi winner sau khi có `best["fitness"]` hợp lệ. Run hiện tại fail trước export vì mọi candidate bị `eligible=0` do reconciliation gate (`report Total Trades=0` nhưng execution ledger có lifecycle), rồi crash ở `round_0002_policy_500005` vì MT5 không ghi report.
- [x] Dừng stale `python/terminal64` sau crash; bridge không tăng thêm và stdout/stderr không cập nhật.
- [x] Thêm fallback artifact cho lần chạy sau: nếu không có winner hợp lệ, vẫn lưu `outputs/mt5_trained.candidate_unpromoted.pkl` kèm manifest để inspect/debug, nhưng không ghi đè winner chính.
- [x] Audit edge-kind graph: graph đã có tên cạnh `forward/recurrent/lateral/memory/predictive/critic/selector/gating/plastic/meta`, nhưng runtime cũ chỉ xử lý đặc biệt `recurrent/memory`; các loại còn lại gần như đi chung công thức `gate * weight * source`.
- [x] Thêm semantics runtime thật cho edge kind:
  - `lateral` dùng chênh lệch source-destination.
  - `gating` dùng source như van mở/đóng.
  - `predictive`, `critic`, `selector`, `plastic`, `meta` có hiệu ứng riêng theo state, eligibility hoặc context.
- [x] Thêm semantics credit theo edge kind: `gating` học gate mạnh hơn, `plastic` học weight mạnh hơn, `predictive/critic/selector/memory/meta` nhận credit phụ theo vai trò thay vì cập nhật y hệt `forward`.
- [x] Regression test khóa edge-kind semantics: `tests/test_neural_graph.py` và `tests/test_graph_plasticity.py` xác nhận các loại cạnh không còn chỉ là metadata.
- [x] Sửa nguyên nhân cụ thể của report/ledger mismatch trong Visual mode: `_serve_policy_actions()` không còn parse/return khi report HTML vừa xuất hiện nhưng process MT5 chưa kết thúc. Trước đó report placeholder có thể parse được với `Total Trades=0` trong khi EA/agent log vẫn đang trade thật.
- [x] Regression test khóa Visual early-report bug: Visual + process chưa exit + report tồn tại phải tiếp tục chờ, không được gọi `parse_mt5_report()`.
- [x] Visual smoke sau early-report patch không còn kết thúc sớm khi report placeholder xuất hiện: bridge vẫn phục vụ `400-800` request trong khi report final chưa sẵn sàng; các run này được dừng thủ công vì Visual chậm, không phải crash.
- [x] Nonvisual MT5 smoke ngắn sau patch đã hoàn tất và report/ledger khớp: `profit=+598.26`, `PF=1.92`, `WR=50.0%`, `trades=12`, `DD=4.90%`, `actions=530`, `neuron_credits=3881`, `reconciled=1`.
- [x] Phát hiện lỗi export model sau smoke: dù `fitness_eligible=0`, workflow vẫn ghi file winner chính. Đã sửa promotion gate để `fitness_not_eligible` không được promote/export vào output chính.
- [x] Regression test khóa export gate: candidate MT5 có `fitness_eligible=false` phải raise `fitness_not_eligible`, không tạo `output.pkl`, chỉ tạo `output.candidate_unpromoted.pkl` để inspect/debug.
- [x] Sửa live observatory rehydrate: policy có config live nhưng `_live_observer=None` phải tạo lại `LiveTelemetryWriter(enabled=True)`, không tạo writer disabled làm custom `--live-events` chỉ có `run_started/topology`.
- [x] Regression test khóa live candidate-copy path: base policy bật live, candidate sinh qua `propose()`, MT5 gọi `set_live_observatory_trial()`, rồi V9 final `act()` phải ghi `trial_started` và `decision` vào custom event file.
- [x] Sửa live world-core path: `SpatialWorldPolicy.from_policy()/propose()` giữ `live_observatory_config` và `LiveTelemetryWriter`, nên `--policy-core world` không còn làm rơi custom live stream.
- [x] MT5 smoke sau live world-core patch xác nhận custom `--live-events` không còn topology-only: live stream có `trial_started`, `bridge_heartbeat`, `decision`, `feedback`, `development`, `lifecycle`, `learner_recovery_lifecycle` và `trial_finished`.
- [x] Sửa EA shutdown/report reliability: thêm `IsStopped()` guard trong vòng `RequestPythonAction()` trước `Sleep(10)` để Strategy Tester không báo `infinite Sleep loop detected` khi tester đang shutdown/forced close. Đã compile `0 errors, 0 warnings` cho runner chính/current/world.
- [x] MT5 smoke sau EA guard chạy qua training pass và tạo report reconciled: policy `1` có `profit=+1033.10`, `PF=3.31`, `WR=66.7%`, `trades=12`, `DD=3.65%`, `actions=431`, `neuron_credits=4372`, `reconciled=1`.
- [x] Phát hiện lỗ hổng promotion gate lần 2: frozen evaluator policy `500001` cũng `fitness_eligible=0` nhưng metrics thô không mang field này về train loop, làm `reject_reasons=[]` và ghi nhầm winner smoke.
- [x] Sửa promotion metadata: `MT5BacktestMetrics` hiện mang `fitness_eligible`, `fitness_rank_score`, `fitness_sample_count`, `fitness_source`, `fitness_eligibility_reasons`; train loop còn có fallback reject khi rank score `-100`.
- [x] Xóa artifact winner smoke bị ghi sai `outputs/mt5_smoke_livefix3.pkl`; chỉ giữ journal/log/report làm bằng chứng audit.
- [ ] Nếu report/ledger vẫn lệch sau patch, bước tiếp theo là thêm `run_id/policy_id` vào execution ledger và filter ledger theo trial hiện tại để loại cross-trial contamination.
- [ ] Nếu probe tiếp tục thua, promotion gate phải reject candidate bằng ledger/fitness, không promote vì né thị trường.
- [ ] Run mới sau heartbeat/action-telemetry patch phải xác nhận UI/live monitor đọc `bridge_heartbeat.action_name` đúng emitted BUY/SELL khi bootstrap/probe phát lệnh.
- [x] Run MT5 mới sau live-observatory rehydrate patch xác nhận custom `--live-events` nhận `decision/feedback/development`, không chỉ `run_started/topology`.
- [ ] Run MT5 mới sau promotion-metadata patch để xác nhận candidate/frozen evaluator `fitness_eligible=0` chỉ ghi `*.candidate_unpromoted.pkl`, không ghi winner chính.
- [ ] Theo dõi cân bằng action: sau lifecycle recovery, position-ticket credit và mistake-index fix, run mới phải bớt phạt nhầm HOLD và ghi đúng BUY/SELL loss support.
- [ ] Xác nhận pre-launch replay sau replay-budget fix không còn mất 5-6 phút trước khi MT5 mở.
- [ ] Nếu `response_status` timeout quay lại dù 15000 ms, cần đo thời gian từ file request MT5 tới Python `request_seen` và tách action-serving khỏi side effects nặng.
