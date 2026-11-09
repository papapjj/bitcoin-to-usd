# V13: walk-forward paired baseline và rollback

## Mục đích

Artifact champion được bảo vệ không bao giờ bị ghi đè. Luồng deployment hiện
dùng `huan-luyen-champion-he-chuyen-gia-mt5`: nhiều shadow độc lập được train
từ cùng champion và **chính từng shadow hash** phải chạy frozen OOS trên cùng
ma trận cửa sổ tương lai. Walk-forward thường chỉ còn là research/audit; các
row của candidate khác nhau không được gộp thành bằng chứng cho một artifact.

## Gate paired OOS

`paired_oos_non_regression_gate` yêu cầu:

- expectancy OOS của candidate không âm;
- profit candidate không thấp hơn seed quá ngưỡng cho phép;
- expectancy LCB không tụt quá ngưỡng cho phép;
- drawdown không tăng quá ngưỡng cho phép.

Gate này bổ sung cho `frozen_oos_promotion_gate`; fitness train dương không thể
thay thế bằng chứng OOS. `champion_candidate_matrix_gate` còn yêu cầu đủ toàn bộ
cửa sổ cho từng SHA256 và đủ candidate seed độc lập. Khi gate fail, shadow vẫn
giữ ở audit để phân tích nhưng không có output promoted nào được tạo.

## Cách chạy

```powershell
python -m self_growing_trader.cli `
  --config configs/he_chuyen_gia_mot_context_mot_policy.yaml `
  huan-luyen-champion-he-chuyen-gia-mt5 `
  --champion outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl `
  --output outputs/he_chuyen_gia/candidates/v13_champion_candidate.pkl `
  --audit-dir outputs/he_chuyen_gia/audit/v13_champion_matrix `
  --train-from 2025.05.15 --train-to 2025.05.22 `
  --validation-count 3 --validation-days 1 `
  --candidate-seed-count 3 --keep-mt5-open --model 4
```

`summary.json` ghi SHA256 champion/shadow, kết quả từng ô, lý do gate và
candidate maximin được chọn. Chỉ artifact có đủ ma trận OOS độc lập và qua toàn
bộ gate mới được sao chép sang đúng `--output`. Xem contract đầy đủ tại
`docs/CHAMPION_EDGE_RETENTION.vi.md`.

## Bằng chứng model thực sự học

Sau khi có candidate và các thư mục `window_*_candidate`/`window_*_baseline`,
chạy:

```powershell
$env:V13_SEED_SYSTEM = "outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl"
$env:V13_CONSERVATIVE_CANDIDATE = "outputs/he_chuyen_gia/candidates/<candidate>.pkl"
$env:V13_CONSERVATIVE_OUT = "outputs/he_chuyen_gia/audit/<run>"
python scripts/audit_v13_learning_proof.py
```

`learning_proof.json` tách riêng các kiểm tra: lifecycle/expectancy/selector có
update, selection-rate có đổi theo timeline, expert mới có bị giảm chọn sau
loss, SHA checkpoint đầu-cuối có khác, và frozen replay có divergence quyết
định. Bảng `performance_comparison` đối chiếu Mean R, expectancy, LCB và
profit factor giữa candidate với seed trên cùng cửa sổ. Các kiểm tra này chỉ
chứng minh có tác động học; việc giữ candidate vẫn phải qua paired gate và
walk-forward OOS.

Với transaction champion, so sánh nằm trong
`summary.evaluations[*].candidate.metrics` và `champion.metrics`. Selector update
được phát thành event riêng trong `learning_events.jsonl`, nên
`selector_update_events` là số update exact, không phải suy luận từ số lệnh đóng.

Với frozen inference, dùng `learning_state_before_after_equal` làm cờ chính.
`policy_relevant_state_before_after_equal` cũ bao gồm cả trace/lifecycle
bookkeeping nên có thể đổi dù learning đã tắt.

Trong lúc MT5 đang chạy, `scripts/watch_v13_learning_live.py` ghi
`live_learning.jsonl`: lifecycle đóng, lời/lỗ theo expert, selection count và
selection rate cùng delta giữa các snapshot, kèm closed Mean R và Profit
Factor tạm thời. Đây là log diễn tiến để phát hiện
kiểu “đầu kỳ lời, sau loss selector vẫn dồn vào expert cũ” trước khi có
checkpoint cuối.

`scripts/run_v13_conservative_matrix.py` tự khởi động watcher này (đặt
`V13_ENABLE_LIVE_WATCH=0` nếu cần tắt), nên một run mới không phụ thuộc vào
việc mở thêm terminal theo dõi thủ công.

Watcher cũng đọc event stream `learning_events.jsonl` riêng của candidate và
ghi trong mỗi snapshot: số `feedback`, `development`, `lifecycle` đã chạy,
credit lifecycle, số lần expectancy được ghi nhận, revision cuối và growth.
Các trường này là bằng chứng policy đã tiêu thụ outcome; bridge/execution chỉ
là bằng chứng lệnh đã được thực thi. Có thể đổi tần suất decision telemetry
bằng `V13_TELEMETRY_DECISION_EVERY`. Một số bridge chỉ phát `feedback` với
`payload.execution.closed=true` thay vì envelope `event_type=lifecycle`; khi
đó watcher ghi `lifecycle_source=feedback_execution_closed_inferred` và
`expectancy_update_exact=false` để không báo sai số lifecycle bằng 0.

Đường close MT5 hiện gọi lifecycle credit ở facade context-owner ngay sau
`observe_transition` khi receipt có `closed` và cờ eligibility xác thực. Nhờ
đó credit đi đúng expert/selector và contextual-expectancy ledger trong cùng
causal event; lần settle lại từ execution ledger chỉ là retry idempotent. Nếu
không có receipt đủ điều kiện, runner vẫn giữ fallback ledger settlement và
không tự suy diễn lệnh đã đóng.

Frozen OOS đặt `mt5_learning_enabled=false`: không update weight/selector,
không tạo role-specialist lazy và không đổi registry. Vì vậy checkpoint A
chạy inference chỉ được dùng làm control; nếu `learning_state_before_after_equal`
không đúng thì control phải bị sửa trước khi diễn giải một candidate là đã học.

Khi một train pass dài bị hiện tượng “đầu kỳ lời, cuối kỳ mất edge”, không nối
checkpoint theo một cửa sổ thắng đơn lẻ. Dùng transaction champion–shadow ở
trên với train window ngắn hơn. Shadow fail ở bất kỳ cửa sổ nào được giữ trong
audit nhưng champion không đổi; shadow chỉ được promote sau ma trận đầy đủ.

Lifecycle credit cũng phải tuân theo `learning.mt5_online_update_scale`. Trước
đây final-close credit dùng scale `1.0`, khiến một cấu hình scale rất nhỏ vẫn
đột biến trọng số sau mỗi lệnh đóng. Runtime hiện bounded cả feedback và
lifecycle credit; test `test_lifecycle_credit_respects_bounded_online_update_scale`
kiểm tra invariant này. Dùng `--keep-mt5-open` để giữ phiên terminal; bỏ cờ này
khi runner là portable clone tách biệt và cần đóng sau từng arm.

Continuation anti-overfit có thể tăng `selector_loss_streak_penalty` trong
candidate config. Đây chỉ là hạng tử xếp hạng sau loss-streak, không phải veto
entry và không làm mất các action hợp lệ; protected v13 giữ mặc định `0.12`.
Profile candidate có thể bật thêm
`selector_policy_group_loss_streak_penalty`: hạng tử này gom loss của cùng
`policy × role-direction` qua các context/microstate khác nhau, tránh việc
policy thua liên tiếp chỉ vì mỗi microstate có một streak riêng. Mặc định
`0.0`, nên seed v13 không đổi; nó chỉ được xem là có tác dụng khi audit
`selector_update` (có `policy_group_loss_streak`) và frozen OOS chứng minh
selection rate giảm hợp lý.

Nếu portable tester clone thoát trước report với log `LiveUpdate start` rồi
`Terminal exit with code 0`, đó là lỗi môi trường MT5 build/update, không được
ghi nhận như train fail của policy. Giữ terminal đăng nhập chính, lưu
`START_FAILURE.md` trong audit, và chỉ chạy lại khi clone không còn tự chuyển
qua LiveUpdate; không promote candidate từ một run không có report.

Profile continuation mặc định đặt `V13_DIRECTION_GATE=0`: direction prior vẫn
được ghi nhận và học cục bộ nhưng không biến thành veto một chiều trên lệnh
vào. Muốn chạy thí nghiệm gate riêng phải bật tường minh `V13_DIRECTION_GATE=1`.

Chunk train không pickle policy trong callback lifecycle. Policy lớn (hơn
`170 MB`) nếu ghi giữa bridge request sẽ chặn tester và làm mất run. Mặc định
`V13_MAX_WALL_SEC=300` dừng một chunk ở biên thời gian sớm, sau đó chỉ pickle
policy đúng một lần ở cuối chunk; artifact bị ngắt vẫn có cờ
`train_result.interrupted=true` và chỉ được xét bằng paired OOS, không tự động
được promote. Khi wall guard cắt bridge trước khi report hoàn tất, bridge vẫn
settle các lifecycle đã đóng và đã ghi bền trong execution ledger trước khi
raise timeout; vị thế còn mở không bị giả định là đã đóng. Final pickle dùng
file tạm rồi replace atomically; raw
`StepTrace`/causal archive trùng lặp được bỏ khỏi artifact cuối để không làm
đầy ổ đĩa, còn lifecycle ledger, expectancy, memory và JSONL observability
vẫn được giữ.

Đặt `--keep-mt5-open` để chỉ dừng Strategy Tester và giữ terminal đăng nhập.
Gate vẫn loại paired arm nếu report không reconcile với execution ledger, dù
profit/expectancy bề mặt có tốt hơn.

Với `scripts/run_v13_conservative_matrix.py`, biến
`V13_CONSERVATIVE_SHUTDOWN_TERMINAL=0` cũng mặc định giữ terminal dùng chung
đăng nhập mở. Chỉ đặt bằng `1` khi toàn bộ train/OOS chạy trên runner portable
đã tách riêng; khi đó terminal sẽ tự đóng sau từng arm để giải phóng cache tick.
Sau batch cần mở lại runner chính và kiểm tra login trước khi giao dịch hoặc
chạy audit tiếp.

Nếu train đã persist candidate nhưng matrix dừng ở lỗi audit/gate, đặt
`V13_CONSERVATIVE_RESUME_CANDIDATE=1` để bỏ qua train và chạy lại toàn bộ
paired OOS từ candidate hiện có. Chế độ này vẫn hash-check protected seed và
không ghi đè candidate; dùng để tái lập các window bị lỗi, không phải để né
gate.

Feature parity cũng được kiểm tra trên toàn bộ 137 input. Riêng các slot tuổi
vị thế có sai số đo thời gian được định lượng: EA lấy tuổi tại tick hiện tại
(làm tròn 0.1 bar), còn input policy lấy thời điểm đóng bar; audit lưu timestamp
đến phút. Verifier chỉ cho phép tối đa `1.05 / max(1, max_holding_bars)` ở
các index tuổi (32, 40, 48, 56, 64), mọi slot market/account/execution vẫn
so sánh tuyệt đối nghiêm ngặt. Vì vậy mismatch lớn hơn biên này vẫn chặn
promotion, không thể che drift bằng cách nới tolerance toàn vector.

Hai trường `spread_atr` và `spread_percentile` là spread tại tick quyết định,
không phải spread đóng bar trong file OHLC. Verifier thay hai slot này bằng giá
trị runtime đã ghi trong audit, nhưng chỉ khi spread ratio trùng với slot spread
được lặp lại trong execution và percentile nằm trong `[0,1]`; mọi input còn lại
vẫn strict. Kết quả ghi `spread_reference_override_rows` và lý do
`runtime_spread_fields_unreconstructible_from_static_ohlc` để audit phân biệt
giới hạn dữ liệu spread với drift feature thật.

Khi terminal đăng nhập đang mở nhưng cần paired OOS không forward vào phiên
idle, dùng `scripts/run_v118_isolated_oos.py`. Script tạo clone mới cho từng
`window/candidate` và `window/baseline`, hard-link kho tick, copy report/bridge/
execution/feature-audit vào `V13_V118_OOS_OUT`, rồi xóa clone sau khi arm kết
thúc. Mặc định chạy model real-tick (`4`) và đặt `promoted=false`; script chỉ
ghi summary/gate, tuyệt đối không copy candidate thành artifact production.

`scripts/watch_v118_then_oos.py` có thể chạy nền trong lúc train: nó chờ
checkpoint và registry xuất hiện ổn định sau lần persist atomic, sau đó gọi
isolated OOS đúng một lần. Trạng thái nằm ở
`outputs/he_chuyen_gia/audit/v118_seed_bounded_scale0001_realticks_oos/oos_monitor_status.json`;
monitor không chạy nếu đã có lock của một phiên OOS khác.
