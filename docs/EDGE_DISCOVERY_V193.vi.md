# V193 — Học hành động và duy trì đa dạng theo bằng chứng

## Mục tiêu

Profile hiện hành huấn luyện trực tiếp hành động `HOLD`, sáu loại entry và
các hành động quản lý. Curriculum timing-first đã bị loại bỏ: lifecycle PnL
không còn bị chặn khỏi actor entry và timing gate có trọng số bằng 0.

Config UI: `configs/he_chuyen_gia_latest_ui_v193.yaml`.
Profile candidate: `configs/experiments/edge_discovery_v193_candidate.yaml`.
Entrypoint: `scripts/run_edge_discovery_v193_candidate.py`.

## Dừng phiên mô phỏng trực quan

Đóng cửa sổ MT5 không tự động làm launcher Python kết thúc. Vì vậy vòng cầu
trực quan thăm dò terminal định kỳ. Nếu đã phục vụ ít nhất một quyết định,
terminal biến mất liên tục 5 giây và chưa có báo cáo MT5 hợp lệ, runner ném lỗi
có kiểm soát; lớp ngoài sẽ ghi checkpoint `*.incomplete.pkl` rồi dọn tester
mồ côi. `--max-wall-sec 0` chỉ tắt giới hạn thời gian tổng, không còn biến một
phiên đã đóng thành vòng lặp vô hạn.

File launcher `/config` của mỗi pass nằm trong `codex_run_configs/` ở gốc
portable runner. Không đặt file này trong `MQL5/Files`: đồng bộ MQL5 có thể xóa
INI trước khi terminal đọc. Khi bật MT5 failure replay, mỗi main/replay pass
yêu cầu EA tự shutdown terminal để giải phóng sạch launcher mutex. Coordinator
dọn terminal theo đúng runner trước mỗi replay và chờ khoảng ba giây không còn terminal/tester, rồi khởi
động một pass mới bằng INI riêng và dọn lại sau khi archive report. Main pass
archive report, ledger và bridge ngay trước replay để dữ liệu dùng chung trong
`Common\\Files` không bị replay xóa. Không chuyển tiếp `/config` vào một terminal đang idle vì MT5 có thể
nhận launcher nhưng không tạo tester/request mới. Probe/cleanup cũng nhận diện
helper LiveUpdate qua đường dẫn runner trong command line. Session manifest vẫn
ghi hash INI và đường dẫn tuyệt đối để truy vết.

Execution-trace observatory vẫn giữ hidden/cell state để phân tích. Khi chạm
trần dung lượng, runner giữ 75% bản ghi JSONL mới nhất thay vì giữ sát 100%;
khoảng trống này tránh việc phải viết lại toàn bộ file ở từng quyết định tiếp
theo mà không làm mất chuỗi intent/receipt/lifecycle gần nhất.

## Phạt lệnh chờ quá hạn

Nếu một lệnh `BUY_LIMIT`, `SELL_LIMIT`, `BUY_STOP` hoặc `SELL_STOP` qua đủ 5
nến M1 vẫn chưa khớp, EA tự hủy lệnh, ghi `UNFILLED_TIMEOUT` và phạt nhẹ
`0.25R` vào đúng trace đã đặt lệnh. Policy chủ động hủy sớm một lệnh chờ thật
không tạo thưởng hoặc phạt hành vi. Không phạt:

- lệnh đã khớp;
- lệnh bị broker từ chối khi chưa được đặt;
- dọn lệnh khi kết thúc terminal;
- hành động `CANCEL_ORDERS` không gắn được với trace entry gốc.

Profile candidate bật `entry_guard_single_position=true`: khi đã có position
hoặc pending order, sáu action entry bị loại khỏi mặt nạ cho tới khi lifecycle
đó kết thúc. Hợp đồng này khớp nhãn teacher từ một pivot cấu trúc tới pivot đối
diện và ngăn policy học scale-in từ các trạng thái teacher chưa giám sát.

Mỗi cặp `order_ticket + trace_id` chỉ được tính một lần. Telemetry ghi số lần
quan sát, route thành công, cập nhật optimizer và cache miss. Khi deal đóng có
`trace_id` của MANAGE/CLOSE, owner vẫn là `position_owner_trace_id` bất biến
của entry; nếu owner không có trong registry thì route bị bỏ qua, không suy
đoán sang hành động quản lý.

## Hợp đồng hành động và GIỮ

Five trên MT5 lấy mẫu theo mặt nạ trạng thái giao với quyền cố định của từng vai
trò: entry mua, entry bán, quản lý vị thế, quản lý lệnh chờ và giám sát danh mục.
EA kiểm tra lại action wire và trả `AUTHORITY_DENIED` nếu vi phạm; không âm thầm
đổi action sai quyền thành `HOLD`. Nếu BUY-agent và SELL-agent cùng đề xuất
entry trong một packet, bridge chỉ phát hướng có xác suất policy gốc cao hơn;
hướng còn lại trở thành `HOLD` có biên nhận
`BATCH_OPPOSITE_ENTRY_CONFLICT`. Mẫu bị chặn không được nhận reward kinh tế và
không vào PPO. Cùng xác suất thì thứ tự tác nhân ổn định quyết định winner để
replay có thể tái lập. `CLOSE_ALL` của giám sát danh mục tiếp tục phủ quyết các
action khác trong packet. Giới hạn tương thích và kiểm tra hợp lệ của broker vẫn
là hàng rào an toàn, mọi từ chối đều có biên nhận theo đúng trace.

`HOLD` khi tác nhân đang sở hữu vị thế được phép học delta marked-R: vị thế cải
thiện cho reward dương, xấu đi cho reward âm. `HOLD` phẳng không nhận reward kinh
tế và bị loại khỏi trộn reward đội, chuẩn hóa reward kinh tế, actor/critic kinh tế
và entropy. Nhãn path vẫn huấn luyện cây quan sát và trần xác suất HOLD khi có
edge; không cộng thưởng `HOLD` dương cho tài khoản trống.

Profile candidate bắt đầu với sàn tổng entry `0,15` trước khi có ba edge đã xác
minh. Kỳ vọng entry âm làm sàn giảm tuyến tính theo độ mạnh bằng chứng, nên
policy không còn bị ép tiếp tục phát lệnh khi dữ liệu hiện tại chứng minh entry
đang thua. Profile dùng trọng số loss `2,0` để sửa nhanh seed đang nghiêng về
`MANAGE_POSITION`, và tăng riêng phạt MANAGE/CLOSE khi tài khoản trống lên
`2,0R`; cấu hình UI sản xuất vẫn giữ trọng số loss `1,0` và phạt hành động
`1,0R`.

Ứng viên còn đặt sàn xác suất hành vi `0,10` cho từng entry hợp lệ. Đây là
trộn khi lấy mẫu của policy, không phải mask ở EA; HOLD và các hành động quản
lý vẫn hợp lệ. Khi đã có edge, sàn này có thể được hạ theo A/B tiếp theo.

Trọng số xếp hạng đường path của ứng viên là `2,0` (UI gốc `0,5`). Nó chỉ
đẩy entry đã chọn về phía entry có chất lượng counterfactual tốt hơn; không
cộng lại MAE/MFE scalar vào lifecycle reward của lệnh đã khớp. Ngưỡng cặp của
ứng viên là `0,05R` (UI gốc `0,10R`) và hàng đợi path dùng unroll `4` thay vì
`16`; nhờ vậy cửa sổ MT5 ngắn vẫn tạo nhiều bước tối ưu, không để mẫu chín bị
bỏ lại trong hàng đợi. Ứng viên còn bật shaping chất lượng action `±0,10R`
theo `quality_r` của path counterfactual: entry tốt nhận tín hiệu dương, entry
xấu nhận tín hiệu âm khi ít nhất một entry vượt ngưỡng edge. Nếu toàn bộ entry
đều không vượt ngưỡng thì không phạt đồng loạt; đường đi đó thuộc mục tiêu
không có edge. Đây là kênh path riêng, không thay đổi reward kinh tế lifecycle
và UI gốc giữ hệ số `0`.

## Selector HOLD hai chiều trên path bất biến

Audit MT5 cho thấy selector-only trước đây chỉ có một chiều: khi path chứng
minh entry tốt, ranking có thể kéo HOLD về entry; khi path không có edge, entry
đang chọn không nhận target quay lại HOLD. Đồng thời scalar reward của snapshot
HOLD phẳng bị loại khỏi `sample_loss_masks`, nên hai tham số
`five_flat_hold_missed_edge_penalty_r` và
`five_flat_hold_confirmed_no_edge_bonus_r` có thể tồn tại trong oracle nhưng
không tạo gradient ở profile `five_pattern_reward_enabled=false`.

Profile ablation hiện hành bật opt-in
`five_action_ranking_bidirectional_enabled`. Nhãn này chỉ dùng trên
`flat_v1`, chỉ cho agent entry 0/1, và có ba trạng thái: entry tốt đủ ngưỡng
thì ranking về entry tốt nhất; mọi entry hợp lệ đều âm dưới ngưỡng thì ranking
về HOLD; vùng mơ hồ không tạo target. Snapshot HOLD flat cũng nhận đúng
missed-edge/no-edge credit từ path, còn agent quản lý/danh mục không nhận nhãn
entry. Đây là sửa đường truyền tín hiệu, không phải bằng chứng edge; mọi
candidate vẫn phải qua nhiều cửa sổ MT5, lifecycle-ledger reconciliation,
terminal-flat và LCB trước khi được giữ lại.

Khi bật live action veto, có một trường hợp cần tách riêng: bằng chứng âm của
một action có thể khiến nhánh `hold_owner_on_negative` khóa toàn bộ ba entry
của một owner về HOLD. `five_live_owner_exploration_enabled` không tự sửa được
điều này vì nó chỉ trộn trong tập action còn hợp lệ. Candidate probe vì vậy có
hai tầng bảo vệ: `five_live_action_veto_probe_enabled` giữ tối đa một entry chưa
có bằng chứng âm, và `five_live_action_veto_probe_interval_decisions` chỉ mở
mask probe theo nhịp quyết định (mặc định 64 packet). Các packet còn lại vẫn
chỉ HOLD; probe dùng geometry/risk của candidate và lifecycle đóng thật để
ghi evidence, không được xem là edge hay promotion. Observatory phải ghi rõ
`five_live_action_veto_probe_due`, `five_live_action_veto_probe_actions` và
`five_live_action_veto_hold_owner_actions` để phân biệt probe thưa với
overtrade do mở mask liên tục.

## Không cho SL đẩy policy khỏi nhóm entry

Mọi trạng thái tài khoản trống tham gia một loss trực tiếp trên policy gốc:
tổng xác suất của sáu entry phải đạt sàn cấu hình khi chưa đủ ba edge dương.
Trong profile candidate, sàn gốc là `0,15`; nó giảm theo bằng chứng expectancy
âm và giảm tuyến tính từ edge thứ 3 đến edge thứ 8 rồi về 0. Cơ chế vẫn ngăn
policy thoát hoàn toàn sang `HOLD` hoặc `MANAGE_POSITION`, nhưng không ghi đè
bằng chứng lifecycle rằng entry đang gây lỗ.

## GAE theo nhịp quyết định MT5

Horizon theo giây vẫn được giữ để tương thích với dữ liệu dày, nhưng horizon
hiệu lực không được ngắn hơn 16 feedback-step. Với nhịp quyết định 1.200 giây,
`60–266` giây cũ làm gamma gần bằng 0; cấu hình mới dùng tối thiểu 19.200 giây,
nên gamma đầu batch là `0,9375`. Batch không-terminal cần 65 feedback: 64 mẫu
được tối ưu và mẫu thứ 65 chỉ cung cấp bootstrap value rồi được giữ lại cho
batch kế tiếp. Batch terminal có thể tối ưu đủ 64 mẫu với bootstrap bằng 0.
Telemetry phải ghi `gae_raw_horizon_seconds`, `gae_effective_horizon_seconds`,
`gae_min_feedback_steps`, `bootstrap_value_mean`,
`bootstrap_value_nonzero_fraction` và `bootstrap_source` để phân biệt cập nhật
có credit dài hạn với cập nhật myopic.

Khi một entry dính `INITIAL_SL`, scalar phạt SL định hình cũ không còn được
cộng vào PPO. Reward kinh tế vẫn giữ nguyên lỗ thật. Một loss cục bộ tại đúng
observation entry gốc hạ xác suất có điều kiện của entry sai chỉ trong nhóm sáu
entry và chuyển xác suất sang năm entry còn lại; gradient tới mọi hành động
không-entry bằng 0.

Không thành phần thưởng dương nào của reward vòng đời được phép bù cho một
`INITIAL_SL` có net-R không dương. Thưởng hình học `+0.05R` chỉ có khi SL/TP
hợp lệ, lệnh không phải `INITIAL_SL` thua và MFE đạt ít nhất `0.50R`. Phần vị
thế còn lại chạm SL sau khi đã chốt đủ lợi nhuận được đánh dấu
`initial_sl_failure=false`, nên không bị học sai thành outcome âm. Telemetry
`five_lifecycle_routing` phải chứng minh route được nhận và
`initial_sl_optimizer_samples > 0`; chỉ thấy lệnh đóng SL trong ledger chưa đủ
để khẳng định mô hình đã học lỗi đó.

## Lịch entropy nền

PPO Five dùng hệ số entropy riêng, không dùng nhầm khóa entropy Direct-RNN.
Lịch hiện hành:

- giữ `0.06` khi có ít hơn 3 phạm vi context/action đủ mẫu với LCB dương;
- từ edge thứ 3 đến edge thứ 8, giảm tuyến tính `0.06 -> 0.001`;
- từ 8 edge trở lên giữ `0.001`.

Tiến độ thời gian không thể tự làm entropy giảm. `positive_eligible_scope_count`
từ sổ lifecycle bất biến là nguồn duy nhất mở khóa giảm đa dạng.

Kỳ vọng âm toàn chính sách không được giảm entropy, trộn đều hành vi 15% hoặc
sàn tổng xác suất entry. Failure replay chỉ hạ đúng hành động lỗi tại observation
và trạng thái hồi quy nhân quả đã lưu; các vùng lân cận chưa được chứng minh xấu
vẫn được khám phá. Drawdown và mức phơi nhiễm chỉ điều khiển rủi ro/khối lượng,
không điều khiển xác suất khám phá.

Path Five được EA phát ngay tại nến thứ 48, không chờ mốc trừu tượng Direct-RNN
600 nến, để reward entry và nhãn cây quan sát được quyết toán sớm hơn.

Mỗi bản ghi `five_pattern_path_credit_accepted` còn ghi véc-tơ reward của năm
tác nhân (`reward_vector_r`), số phần tử khác 0 và thống kê reward của batch vừa
được tối ưu (`pattern_batch_reward_*`). Vì vậy `reward_mean = 0` không còn đủ để
kết luận reward không chạy; phải kiểm tra véc-tơ, phần tử khác 0 và số lần
`updates` cùng với thống kê batch.

## Checkpoint cứu hộ khi MT5 lỗi

Sau khi policy đã được nạp, mọi lỗi bridge/terminal/report của MT5 đều phải ghi
nguyên tử state trong RAM thành `*.incomplete.pkl` trước khi ném lại lỗi. File
`*.incomplete.json` đi kèm ghi loại lỗi, nội dung lỗi, thời điểm lưu và bắt buộc
`promotable=false`. Timeout chủ động cũng dùng cùng hợp đồng này. Checkpoint hoàn
chỉnh ở đường dẫn output chính không được tạo hoặc ghi đè khi lượt đánh giá lỗi.
Nếu người vận hành nhấn `Ctrl+C` trong lúc evaluator đang chạy hoặc chờ Codex
teacher, Python cũng lưu snapshot forensic theo cùng hợp đồng với
`error_type=KeyboardInterrupt`, rồi mới trả lỗi ra ngoài; chỉ việc cưỡng bức
đóng tiến trình bằng hard-kill/`TerminateProcess` mới nằm ngoài bảo đảm lưu
state. Runner CLI cài handler cho `SIGINT` và `SIGTERM`, chuyển hai tín hiệu
đó thành `KeyboardInterrupt` để đi qua cùng đường persist-and-stop; sau khi
runner kết thúc, handler cũ được khôi phục để không làm bẩn process gọi nó.

Manifest candidate ghi đường dẫn, SHA-256, kích thước và metadata của checkpoint
cứu hộ. Pipeline vẫn kết thúc với exit code khác 0 đối với lỗi MT5 thật; checkpoint
cứu hộ chỉ dùng để tiếp tục/điều tra, không được xem là báo cáo MT5 hợp lệ.

Trước khi chạy, `from-date/to-date` được phân tích nghiêm ngặt theo lịch
`YYYY.MM.DD`; ngày không tồn tại như `2025.02.30` bị từ chối, không được tự sửa
ngầm. Preflight dung lượng lấy giá trị lớn hơn giữa `1,25 × kích thước seed` và
`seed + ngân sách telemetry`; nếu vượt quota thì dừng trước khi clone/chạy MT5.
Profile hiện hành dùng quota payload 12 GiB vì checkpoint Five đã vượt 8 GiB.
Mức trống tối thiểu tuyệt đối là 1,5 GiB; phần dự báo payload đã cộng biên 25%
trên kích thước seed nên không giữ lặp thêm biên 2,5 GiB lần thứ hai.

Khi MT5/bridge/report lỗi, các file nhỏ trong `Reports`, `logs` và `Tester`
được sao chép vào `mt5_failure_diagnostics` trước khi runner tạm bị xóa. Giới
hạn sao chép là 128 MiB; cache/tick corpus và file nhị phân không được giữ lại.

Runner candidate không còn ghi đè metadata về build 5833. Nếu thư mục
`mt5_v2_runner/runtime_overrides` tồn tại thì cả `terminal64.exe` và
`metatester64.exe`, kèm `build.txt`, bắt buộc phải có đủ và được clone như một
bộ thống nhất. Manifest ghi build cùng SHA-256 của hai binary. Cách này tránh
trường hợp dịch vụ MetaTester giữ binary cũ khiến Terminal chạy LiveUpdate ở
giữa hai lượt đánh giá và thoát trước khi tạo báo cáo.

Các file history lớn trong `Bases` vẫn được hard-link. Nếu MT5 đang khóa một
file metadata nhỏ như `alerts.dat` và Windows từ chối tạo hard-link, runner chỉ
sao chép file đó rồi tiếp tục; manifest ghi số hard-link và số file phải copy.

Post-replay là transaction có rollback. Runner clone policy ngay sau main pass,
chạy các failure replay rồi đánh giá lại clone inference-only trên đúng normal
window. Với pipeline candidate hiện hành, main pass có learning không còn được
dùng làm đối chứng cho weights sau replay vì hai bên không cùng policy và không
cùng nguồn expectancy. Kết quả sau replay được chuyển sang gate cặp seed ở bước
kế tiếp. Các caller cũ chưa bật gate cặp vẫn giữ comparator main/post cũ.

Gate cặp nạp lại checkpoint nguồn bất biến rồi chạy source và candidate ở chế
độ frozen trên cùng ngày, MT5 model và số packet. Candidate chỉ được lưu khi
fitness frozen của chính nó hợp lệ, execution đã reconcile, expectancy hiện tại
không âm, PnL không hồi quy, LCB không giảm quá `0,02R`, drawdown không tăng quá
1 điểm phần trăm và fitness không thấp hơn source. Nếu không đạt, output chính
được ghi bằng source seed; manifest ghi `selected=source_seed` và
`rollback_applied=true`.

Trong frozen OOS, `policy_expectancy` dùng lifecycle sinh ra trong chính cửa sổ
đánh giá. Ledger tích lũy trong checkpoint chỉ được giữ dưới
`checkpoint_expectancy_audit` để chẩn đoán, không tham gia fitness/gate. Điều
này ngăn các failure replay có chủ đích làm bẩn điểm OOS. Trong lúc học, ledger
lịch sử vẫn điều khiển entry gate theo đúng scope `context + action`; profile
candidate không fallback sang số liệu gộp của context. Smoke run giới hạn bốn
failure-replay session để dành ngân sách cho hai lượt frozen cặp.

Khi MT5 đóng một vị thế sau khi bridge đã ngừng phát `DEAL_ADD`, evaluator còn
đọc bảng `Deals` trong report HTML để không làm mất lifecycle cuối. Parser phải
đọc theo tên cột (không theo offset cố định); các dòng report được join với
native execution ledger bằng owner trace hoặc cặp PnL/thời điểm duy nhất. Native
lifecycle luôn thắng khi có match. Report-only lifecycle chỉ được đưa vào
effective summary khi số dòng đóng và tổng PnL khớp report; nó được đánh dấu
`report_fallback`, và chỉ được Codex credit khi `trace_id` lấy được từ raw ledger.
Manifest/runtime log ghi `report_lifecycle_join`, `fallback_count` và
`fallback_owner_resolved`; report fallback không được dùng để che mismatch hoặc
biến một run chưa có owner thành bằng chứng promotion.
Khi parser đã tạo được report lifecycle, reconciliation dùng đúng đơn vị
position-lifecycle: số lifecycle report phải bằng số lifecycle ledger. P/L
tolerance chỉ xử lý rounding/swap nhỏ trên cùng tập lifecycle; nó không được
dùng để biến chênh lệch count hoặc hai tập lifecycle khác nhau thành
`reconciled=true`. Mismatch bị ghi `report_lifecycle_count_mismatch` hoặc
`report_ledger_pnl_delta`, chuyển fitness source sang
`execution_ledger_unreconciled` và chặn promotion.

Contract compact mà UI đọc từ `manifest.feedback_contract.codex_teacher_phase`
phải giữ riêng các lớp bằng chứng: `graded/updates`,
`account_impact_records/account_impact_confirmed`,
`counterfactual_validated`, `codex_causal_links_observed`,
`codex_causal_updates_linked`, `codex_causal_hash_matches`, cùng
`rlhf_parameter_changed` và `rlhf_probability_shift_count`. Khi run bị dừng
giữa chừng, cùng các trường này phải được phục hồi từ status sidecar; không
được suy ra causal learning chỉ từ `generic_optimizer_updates` hoặc tổng
`ppo_updates`.

Điểm lưu tự học được phép làm nguồn cho cửa sổ kế tiếp khi `manifest.json` dạng
5 cùng thư mục khớp đúng mã băm và đường dẫn, cấu hình nguồn cùng điểm lưu nguồn
không đổi, sổ thực thi đã đối chiếu, lượt sinh đã hoàn tất và phép kiểm tra trọng
số đã lưu thực sự được chạy rồi đạt. Chuỗi nguồn còn phải mang chứng nhận dạng
11 rằng phần học bắt chước hiện tại đã vượt cả phép kiểm định ranh giới 0 và phép
kiểm định tự vận hành liên tục. Chỉ một cờ `effective_for_mt5=true` từ báo cáo cũ
không còn đủ. `fitness_eligible=false` không được biến thành quyền thăng hạng;
trạng thái này chỉ cho phép tiếp tục nghiên cứu theo thời gian.

Các `manifest.json` dạng 4 trở về trước và báo cáo học bắt chước dạng 10 trở về
trước đã bị loại khỏi đường tiếp nối bình thường. Nếu cần khảo sát chúng, phải
dùng tùy chọn bỏ điều kiện dành riêng cho nghiên cứu và kết quả vẫn không được
thăng hạng.

## Điều kiện đánh giá

Checkpoint không được thăng hạng chỉ vì entropy giảm hoặc PnL một đoạn dương.
Vẫn cần policy-origin expectancy LCB dương, không có phạm vi quan trọng âm,
drawdown qua gate và paired frozen OOS trên cùng cửa sổ.

## Chẩn đoán live và ablation matched

Mỗi candidate MT5 mới có một monitor chỉ đọc, cập nhật nguyên tử
`mt5_diagnostics.json` khoảng 5 giây một lần. File này tách rõ:

Monitor không coi file rỗng có sẵn trong runner clone là ledger thật. Nó luôn
theo file mới nhất dưới `Tester/Agent-*/MQL5/Files`, kể cả khi MT5 đổi tác nhân
thử nghiệm giữa lượt, và bỏ qua dòng tiêu đề được ghi lại sau khi tệp được tạo
lại.

Dạng chẩn đoán 3 giữ `performance.close_reason_net_pnl` và sửa đơn vị đếm vòng
đời. `performance.final_close_rows` là số dòng mang cờ đóng cuối,
`performance.final_position_closes` là số mã vị thế độc lập đã hoàn tất, còn
`performance.duplicate_final_close_rows` cho biết số dòng đóng cuối bị lặp.
Giao diện và công cụ xuất biểu đồ dùng các trường này để không nhầm một vị thế
có nhiều dòng đóng với nhiều giao dịch độc lập.

`policy` trong chẩn đoán hiện chọn `scope=economic_actor` khi observatory có
receipt `five_economic_actor_eligible=true`. Khi đó `actions`, entropy, effective
actions, HOLD và hướng vào lệnh chỉ tính các actor có quyền tạo tác động kinh tế;
HOLD vận chuyển từ manager/governor không hoạt động vẫn được giữ trong
`wire_actions`, `wire_action_counts` và các trường `wire_*` để audit. Nếu chưa có
receipt scope kinh tế, monitor giữ fallback `scope=wire` tương thích với ledger
cũ. Guard continuation/champion cũng dùng scope kinh tế trước, tránh đánh dấu
collapse chỉ vì các agent không có quyền vào lệnh buộc phải gửi HOLD.

Có thể xuất một ảnh chụp đã đối chiếu trực tiếp từ ledger bằng lệnh:

```powershell
python scripts/export_mt5_live_diagnosis.py --output-dir <thu_muc_ket_qua> --runner-dir <thu_muc_runner>
```

Công cụ tạo JSON, báo cáo tiếng Việt và biểu đồ PNG; phần giao dịch, hành động và
vốn được đọc lại từ đầu ledger, còn số cập nhật và tín hiệu học lấy từ trạng thái
bộ theo dõi đang chạy.

Từ dạng chẩn đoán 2, báo cáo còn tách lãi ròng theo nguyên nhân đóng. Nhờ đó có
thể phân biệt khoản lỗ do cắt lỗ với kết quả của tác nhân chủ động đóng và chốt
lời, thay vì chỉ nhìn số lần xảy ra của từng nguyên nhân.

Khi cần phân tích từng vòng đời, dùng `scripts/analyze_v193_trade_lifecycles.py`.
Công cụ ghép lần khớp vào với toàn bộ lần khớp đóng theo mã vị thế, báo riêng số
dòng đóng, dòng mang cờ đóng cuối, vị thế hoàn tất và dòng bị lặp. Nó còn xuất
lãi lỗ theo kiểu vào, tuổi vị thế, cắt lỗ, chốt lời và lý do đóng tổng hợp.

`scripts/export_v193_deep_diagnosis.py` đối chiếu kết quả vòng đời với dữ liệu
dạy, điểm lưu vòng học, và thử nghiệm MT5 cùng điều kiện. Báo cáo số liệu có
khoảng tin cậy lấy mẫu lại theo ngày; biểu đồ cho thấy kiểu vào, tuổi vị thế,
mục tiêu lợi nhuận và ba nhánh thử nghiệm trong cùng một ảnh.

Sau khi bộ soi mạng ghi `results.json`, có thể tạo báo cáo và biểu đồ tóm tắt
bằng lệnh:

```powershell
python scripts/export_torch_diagnostics_summary.py --results <results.json> --output-dir <thu_muc_ket_qua>
```

Công cụ xuất `summary.json`, `ANALYSIS.vi.md` và `analysis_overview.png`. Báo cáo
tách hạng hiệu dụng trước/sau khối chú ý, chú ý giữa tác nhân và theo thời gian,
quyết định theo từng vai trò, độ dốc chẩn đoán, tính hữu hạn của đặc trưng/trạng
thái và thời gian chạy. Độ dốc ở đây chỉ là phép dò độ nhạy, không phải bằng
chứng về cập nhật PPO lịch sử.

- hiệu quả giao dịch: số lần khớp đóng, số vị thế độc lập, số thắng/thua, hệ số lợi nhuận, tổng lãi/lỗ và kỳ vọng mỗi lần khớp đóng,
  payoff ratio, WR hòa vốn và nguyên nhân đóng TP/SL/EXPERT;
- rủi ro: balance, equity, return và max equity drawdown tính từ đỉnh đã thấy;
- learning: số PPO update, rollout progress, policy/value/total loss, entropy,
  actor eligibility và pattern update;
- reward/credit: mật độ batch reward kinh tế hoạt động, reward eligibility và
  số giao dịch đóng được credit;
- policy/execution: entropy action chuẩn hóa, số action hiệu dụng, HOLD/top
  action share, reasoning action scale, imitation prior, reject và pending
  timeout.

Heuristic `insufficient_optimization` được dùng khi chưa đủ 5 PPO update; UI
không được đổi trạng thái này thành “thiếu lớp ẩn”. Overfit không đo được trong
một lượt online in-sample và luôn yêu cầu frozen OOS hoặc ablation matched.

`scripts/run_mt5_ablation_v193.py` chạy tuần tự ba arm cùng checkpoint
SHA-256, ngày, MT5 model và seed: frozen policy, full online và reasoning off.
Report `ablation_summary.json` ghi metric từng arm cùng delta PF, expectancy và
max DD so với full online. Đây là attribution một cửa sổ; muốn kết luận tổng
quát phải lặp nhiều seed/cửa sổ và không được promotion trực tiếp từ report này.

## Dong bo anh chup va gia khop 2026-09-17

EA da luu lai `close_bid` va ATR dung ngay truoc khi bridge xu ly. Lenh cham
tre duoc dung lai anh chup nay khi dat gia, SL va TP; khong lay gia/ATR moi sau
khi Python tra ket qua. Neu khong, dau vao co the tre hon mot den vai nhip so
voi gia dung de khop lenh.

Bo nhan bay lua chon cung da chuyen sang khop lenh cho tai dung gia dat. Hop
dong chay tu choi manifest dung cach tinh gia cham bid/ask cu, vi cach cu tao
nhan lac quan hon MT5.

## Vòng kiểm chứng reward/veto ngày 2026-09-20

Các run dưới đây đều là research-only: seed imitation không có sidecar, tester
dùng model 1 nên chưa có real-tick marker để promotion. Mỗi run MT5 chạy tuần tự
vì dùng chung bridge/terminal/ledger; phần đọc manifest, trace và test được chạy
song song nhưng không khởi động hai train trên cùng runner.

- `run68`, scale online `0.50`, cửa sổ 2025.02.05..06: 94 vị thế hoàn tất,
  net `+790.17`, PF `1.51`, policy LCB `+0.0379R`, DD `7.23%`, reconcile=1.
  Frozen validation có 71 vị thế, net `-59.47`, PF `0.97`, LCB `-0.1288R`,
  DD `16.90%`; champion bị loại.
- `run69`, scale `0.25`, probe mỗi 8 quyết định: probe thật xuất hiện 17 lần
  due và 5 action probe, nhưng mở thành 135 vị thế, DD `12.89%`, policy LCB
  `-0.1556R`, 6 negative scopes; bị loại vì overtrade/risk.
- `run71`, scale `0.25`, probe mỗi 32 quyết định: main 88 vị thế,
  net `+1026.11`, PF `1.74`, LCB `+0.1014R`, DD `6.88%`, reconcile=1; frozen
  validation 93 vị thế, net `-392.90`, PF `0.84`, LCB `-0.2368R`, DD `24.16%`.
  Probe nhanh hơn không duy trì được edge.
- `run72`, scale `0.25`, probe mỗi 64 quyết định trên cửa sổ độc lập
  2025.02.04..05: report có 192 trades nhưng chỉ 191 vị thế hoàn tất, còn 1
  position mở cuối tester; closed-ledger net `+773.74`, mean `+4.05 USD/position`
  nhưng policy LCB `-0.01995R` và 10 negative scopes. Đây không phải cửa sổ
  hợp lệ để kết luận dương.
- `run76` chạy lại đúng profile của `run72` nhưng trên `2025.02.05..06` với
  checkpoint và paired validation bật. Main online có `93` vị thế hoàn tất,
  net `+564.28`, PF `1.35`, DD báo cáo `6.89%`, reconcile=1; tuy nhiên policy
  LCB `-0.02184R`, actor eligible `0`, entropy action gần `0` và HOLD chiếm
  `96.54%`, nên policy đã collapse. Frozen candidate cùng cửa sổ có net
  `+36.56/68` vị thế, PF `1.02`, DD `15.81%`, policy LCB `-0.07401R`; baseline
  nguồn là `-318.13/92`, LCB `-0.36860R`. Candidate có cải thiện PnL tương đối
  nhưng bị loại vì `paired_drawdown_increase` (`15.81% > 13.69%`) và
  `candidate_frozen_fitness_ineligible`. Manifest ghi `accepted=false`,
  `selected=source_seed`, `rollback_applied=true`; file candidate cuối cùng
  dùng hard-link về seed nguồn (`candidate_sha256` không đổi), không được dùng
  làm checkpoint tiếp tục học.
- `run83` giữ nguyên profile behavior-compatible của `run82`, chỉ thêm
  directional entry mass floor nhẹ `0.06 -> 0.02`, weight `0.25`, trên cùng
  cửa sổ `2025.02.05..06`. Main kết thúc `-0.01/136` trades, PF `1.00`, DD
  `9.98%`, policy LCB `-0.1947R`; HOLD cuối khoảng `92.6%` và có 6 negative
  scopes. Champion frozen là `-745.14/118`, PF `0.69`, DD `12.67%`, LCB
  `-0.3632R`. Candidate frozen có PnL `+104.77/72`, PF `1.08`, DD `4.58%`,
  nhưng còn 1 vị thế mở và LCB `-0.1723R`; baseline `-290.02/210`, DD
  `15.77%`, LCB `-0.1385R`. Paired gate reject vì candidate LCB regression
  (`-0.16977 < -0.15852`) và candidate fitness không hợp lệ; rollback về
  `source_seed`. Directional floor đã chạy đúng đường optimizer (unit gradient
  test đạt), nhưng không tạo edge hay giảm LCB trên cửa sổ này.
- `run84` giữ nguyên arm run83 và chỉ bật public Five failure-anchor replay
  trên cửa sổ mới `2025.02.06..07`; MT5 historical-session replay vẫn tắt.
  Replay đã chạy thật: `75` failure anchors được admit, `9` replay batches,
  tổng `61` anchor được chọn replay. Tuy nhiên không anchor nào được đánh dấu
  repaired/resolved; failed-action probability ở các batch vẫn khoảng
  `0.56..0.99`. Main là `-213.99/148` trades, PF `0.90`, DD `13.16%`, policy
  LCB `-0.1747R`; champion frozen `-403.19/138`, LCB `-0.2675R`; candidate
  frozen `-479.71/59`, PF `0.58`, DD `5.68%`, LCB `-0.5519R`, terminal open=1;
  baseline `+194.04/358`, PF `1.03`, DD `17.29%`, LCB `-0.0721R`. Paired gate
  reject vì candidate PnL/LCB regression và frozen fitness không hợp lệ;
  rollback về `source_seed`. Replay đã được route đúng nhưng chỉ củng cố các
  failure anchor chưa sửa được, chưa chứng minh giảm variance hay tạo edge.
- `run85` giữ nguyên arm replay của run84 nhưng yêu cầu tối thiểu `2` failure
  độc lập trong cùng condition/action bucket trước khi tạo replay gradient;
  singleton vẫn được giữ để audit. Trên cửa sổ mới `2025.02.07..08`, main
  `-1866.77/97`, PF `0.20`, DD `22.10%`, policy LCB `-0.7526R`; champion
  frozen `-427.92/75`, PF `0.68`, DD `7.99%`, LCB `-0.4804R`; baseline
  `-1938.12/116`, PF `0.21`, DD `22.58%`, LCB `-0.7309R`. Replay admit
  `82` anchor, chạy `6` batch, chọn `40` anchor và resolve được `2`. Candidate
  cải thiện tương đối so với baseline (PnL và DD, LCB paired khoảng
  `+0.255R`) nhưng LCB vẫn âm, candidate fitness không hợp lệ và paired gate
  vẫn reject; rollback về `source_seed`. Support-2 là tín hiệu giảm variance
  trong một regime xấu, chưa phải edge đa cửa sổ.

Kết luận của vòng này: reward/lifecycle đã về đúng owner và ledger vẫn reconcile,
nhưng selector vẫn collapse về HOLD khoảng 90--97% và không giữ được edge qua
frozen validation/ngày độc lập. `run76` cho thấy chạy train thật bằng cấu hình
`run72` có thể tạo PnL dương ở main và cải thiện so với baseline trên frozen
window, nhưng entropy/actor eligibility sụp và DD frozen vượt gate; đó là cải
thiện tương đối chứ chưa phải edge có thể tiếp tục. Probe interval 8 chứng minh
cơ chế exploration đã chạy thật nhưng trade-off rủi ro quá xấu; interval 32/64
giảm turnover nhưng validation vẫn âm hoặc LCB không dương. Run84 xác nhận
failure replay không bị bypass nhưng không sửa được anchor; run85 cho thấy
lọc singleton có thể cải thiện tương đối trong một regime xấu nhưng vẫn không
đưa LCB qua 0. Không profile nào được promote.

Trong quá trình chạy, run70 bị `WinError 112` khi copy market bases ở frozen
validation vì còn 0.46 GiB trống. Chỉ các thư mục runner tạm dưới
`.mt5_runners` (39 thư mục, không có `active_run.json`) được dọn bằng
`safe_remove_runner`; output, seed và source không bị xóa. Sau cleanup còn
18.23 GiB, run71 được chạy lại đầy đủ.
