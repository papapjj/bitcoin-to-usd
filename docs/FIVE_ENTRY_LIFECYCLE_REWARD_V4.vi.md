# Reward vòng đời Five V4

## Mục tiêu

V4 chấm chất lượng của một entry đã khớp đúng một lần, sau khi lifecycle đóng bất biến. Reward kinh tế vẫn phản ánh toàn bộ lãi/lỗ thật và không bị thay thế bởi reward định hình.

## Hợp đồng loại trừ

- `INITIAL_SL` không còn nhận một scalar định hình âm. Lỗ kinh tế thật vẫn đi
  qua reward kinh tế; lỗi chọn entry được đưa vào loss cục bộ có điều kiện chỉ
  giữa sáu entry, nên không thể làm xác suất chảy sang `HOLD` hoặc
  `MANAGE_POSITION`.
- `INITIAL_SL` không nhận thêm phạt MFE thấp.
- `INITIAL_SL` có net-R không dương không nhận bất kỳ thành phần thưởng dương
  nào từ hình học, trailing hoặc tốc độ khớp lệnh chờ. Loss cục bộ xử lý kết
  quả entry sai; lỗi hình học quá gần vẫn có thể nhận scalar âm riêng dưới
  giới hạn tổng.
- Nếu phần vị thế còn lại chạm SL ban đầu nhưng các lần chốt trước đó làm toàn
  lifecycle có net-R dương, outcome vẫn ghi `initial_sl=true` để audit nhưng
  `initial_sl_failure=false` và không bị loss cục bộ coi là entry sai.
- Hình học SL/TP quá gần là lỗi riêng, nhưng tổng reward vòng đời bị chặn trong `[-0.75R, +0.75R]`.
- Trailing chỉ được đánh giá khi model đã yêu cầu trailing và MFE từng đạt ít nhất `0.50R`.
- Giá chưa đi thuận lợi được quy cho entry, không quy cho trailing.
- MAE/MFE của entry thật chỉ sinh scalar reward tại đây; cây quan sát vẫn dùng chúng làm nhãn dự đoán.

## Thành phần

- MFE mềm: `0R → -0.40R`, `0.10R → -0.30R`, `0.25R → -0.15R`, `0.50R → 0`.
- MFE từ `1R` với MAE thấp: `+0.15R .. +0.30R`.
- Thoát giữ được `25% .. 80%` MFE: `+0.10R .. +0.30R`.
- Hồi phục sau MAE và duy trì có lời: `+0.05R .. +0.15R`.
- Lệnh chờ khớp trong tối đa 5 bar và sau khớp có MFE tốt: `+0.05R .. +0.10R`.
- SL/TP hợp lý và entry không phải `INITIAL_SL` thua, đồng thời MFE đạt ít nhất
  `0.50R`: `+0.05R`.
- Trailing khóa `50% .. 80%` MFE: `+0.10R .. +0.25R`.

Không có thưởng chỉ vì đặt hoặc khớp lệnh.

## Loss cục bộ INITIAL_SL

Với đúng observation và ký ức tại entry gốc, hệ thống chuẩn hóa riêng sáu logit
entry rồi hạ xác suất có điều kiện của entry đã dính `INITIAL_SL`. Gradient của
loss này đối với `HOLD`, `CANCEL`, `CLOSE` và `MANAGE_POSITION` bằng đúng 0.
Phần xác suất bị hạ được phân phối lại cho năm entry còn lại. Trọng số hiện hành:
`five_lifecycle_initial_sl_local_loss_weight: 1.0`.

Hình phạt scalar INITIAL_SL cũ đã bị xóa để không đếm cùng lỗi qua reward kinh
tế, reward vòng đời và loss entry cục bộ.

## Batch học

Lifecycle được gom tối thiểu 32 mẫu và lấy round-robin theo:

- sáu loại entry: mua/bán × thị trường/giới hạn/dừng;
- kết quả dương/âm/trung tính;
- chế độ xu hướng/đi ngang/biến động.

Không lặp mẫu, không sửa nhãn và không ép tỷ lệ 50/50. Nếu chỉ có một nhóm kết quả, hàng đợi chờ tối đa 64 mẫu rồi mới giải phóng batch để tránh treo học.

## Candidate settlement net-R 2026-09-18

Profile nghiên cứu `mt5_trade_settlement_reward_v1.yaml` có thêm một kênh
tuỳ chọn, mặc định tắt: `five_lifecycle_net_r_credit_enabled`. Khi bật, đúng
entry owner nhận thêm `clip(realized_r * weight, ±clip_r)` với dấu nguyên vẹn;
V4 vẫn chỉ là tín hiệu phụ. Kênh này không truyền reward sang bốn agent còn lại
và không thay đổi hợp đồng 137 input/20 output. Profile candidate dùng
`weight=1.0`, `clip=2.5R`, batch `16/16` để một run MT5 ngắn có thể thực sự
đưa settlement vào PPO thay vì để toàn bộ lifecycle nằm chờ trong hàng đợi.
Các tổng `v4_credit_sum_r`, `net_r_credit_sum_r` và `combined_credit_sum_r`
được giữ trong `five_lifecycle_routing` để đối chiếu với ledger.

Smoke profile hiện giới hạn execution observatory ở 32 MiB và event log ở 16 MiB
để không làm cạn ổ đĩa khi chạy nhiều cửa sổ; đây chỉ là giới hạn telemetry,
không thay thế `mt5_diagnostics` hoặc execution ledger trong tiêu chí xác nhận.

## Candidate action decision arm 2026-09-18

Sau khi kiểm chứng MT5 trên cửa sổ tháng 1, expected-value surface của seed
v22 vẫn phát ra edge âm và threshold cao hơn làm kết quả xấu hơn. Candidate
được phép chọn `learning.five_action_decision_mode=direct` để PPO học trực tiếp
entry action logits từ owner-only lifecycle net-R. Runtime cũng mở lại
`direct_action_head`, `direct_observation_head` và critic nếu checkpoint cũ đã
lưu chúng với `requires_grad=false`; mặc định `checkpoint` vẫn giữ nguyên mode
trong checkpoint. Đây là ablation research-only, không thay đổi 137 input/20
output, authority Five, hay promotion gate.

## Independent-window gate arms 2026-09-18

Để so sánh công bằng trên từng cửa sổ ngày độc lập, hai profile mới dùng cùng
settlement net-R, cùng giới hạn telemetry và `five_mt5_failure_replay_sessions_per_run: 0`:

- `configs/experiments/mt5_trade_settlement_reward_v1_expected_gate.yaml` giữ
  `five_action_decision_mode=expected_value`.
- `configs/experiments/mt5_trade_settlement_reward_v1_checkpoint_gate.yaml` giữ
  `five_action_decision_mode=checkpoint`.

Replay phụ bị tắt trong hai profile gate để không biến các phiên replay thành
những cửa sổ đánh giá ẩn. Một cửa sổ chỉ hợp lệ khi run hoàn tất, close cuối có
`routing.bound`, `filled`, `closed`, `final_close_confirmed`, ledger reconcile và
đủ số giao dịch tối thiểu; kết quả âm vẫn là kết quả thất bại, không được loại
bỏ khỏi phép gộp. Các profile vẫn research-only cho tới khi seed offline và
paired frozen OOS cùng qua gate.

Trong `optimize_independent_pattern_steps`, raw observation phải giữ dạng
`[batch, 5, 137]` khi đi vào các head action trực tiếp; ép thành
`[batch*5, 137]` làm lệch với hidden `[batch, 5, hidden]` và khiến direct arm
fail trước khi có gradient. Regression test giữ lại shape contract này.

## Kết quả kiểm chứng sau sửa 2026-09-18

Retry checkpoint-gate trên `2025.01.01..2025.01.07` đã hoàn tất và ledger
reconcile đầy đủ, nhưng không đạt paired frozen gate:

- main learning: `29` trades, mean net-R `+0.1627`, PnL `+157.14`,
  reconcile `1`;
- frozen candidate: `28` trades, mean net-R `-0.4183`, PnL `-263.25`,
  reconcile `1`;
- frozen baseline: `20` trades, mean net-R `-0.7263`, PnL `-330.06`,
  reconcile `1`.

Vì main dương nhưng frozen candidate âm, đây là overfit trong cửa sổ train,
không phải candidate đạt. Seed v22 cũng đã fail offline trước MT5 với filled
mean khoảng `-0.234R` ở validation và `-0.223R` ở test. Hai candidate offline
mới dùng outcome target đã được đo nhưng vẫn âm: v23 expected-value test
filled mean `-0.1783R`, block LCB `-0.2560R`; v24 direct test filled mean
`-0.2100R`, block LCB `-0.2913R`. Chúng không được đưa vào MT5.

Tệp raw tháng 1 ở Agent-3001 được dựng riêng để chẩn đoán temporal, nhưng
fingerprint của nó là `five_observation_v10_137`, khác fingerprint runtime hiện
tại; vì vậy không được dùng làm candidate MT5. Không có kết quả dương hợp lệ
để gộp hay promotion sau các phép đo này.

Đã xuất lại tháng 1 bằng Strategy Tester trên Agent-3000 với đúng profile V193:
44.635 nến, 178.535 tick, fingerprint `d5fd9...e3a402`, mỗi dòng đủ 137
input, geometry SL 2,5 ATR / TP 5 ATR (RR2), pending fill theo `order` và risk
neo theo `requested_at_placement`. Dataset sau purge có 44.582 thời điểm,
`production_eligible=true`, nhưng causal selection mean vẫn `-0,00076R`.
Candidate v25 học riêng từ cửa sổ này khớp runtime contract và reload/frozen
invariant đều pass, nhưng validation block lần lượt `-0,225`, `-0,210`,
`-0,279`, `-0,521R`; test `-0,1725R` trên 5.525 entry, block LCB
`-0,2626R`, nên `promotable=false`. Sweep ngưỡng value tốt nhất theo validation
chỉ đạt test mean `+0,0433R` trên 193 entry nhưng LCB `-0,0817R`, không đủ gate.
Đây là kiểm chứng dữ liệu đúng schema vẫn chưa có edge, không phải lỗi reward
receipt hay lifecycle reconcile.

Quét 444 vùng đơn biến trên cùng dataset có 30 vùng dương ở validation, nhưng
không có vùng nào đồng thời giữ LCB dương ở test; vì vậy không có cơ sở chọn
thêm regime filter để che kết quả âm.

Kiểm chứng độc lập tháng 2 (40.307 nến, 161.228 tick, cùng fingerprint và
contract) cũng cho causal selection test `-0,1336R` trên 146 entry, LCB
`-0,2322R`; cả sáu action đều có mean âm ở train, validation và test. Geometry
grid theo lịch sử chỉ cho validation `-0,3459R` và test `-0,1336R`, nên không
có lý do chạy thêm candidate MT5 từ cửa sổ này.

Đã sửa exporter để parity với `ComputeAllInputs()` của EA live: account
day-progress dùng `TimeCurrent()` và execution spread dùng spread broker tại
quyết định, không dùng spread của bar đóng trước. MetaEditor compile lại với
`0 errors, 0 warnings`; parity test xác nhận chỉ hai feature thay đổi (day
progress và spread). Candidate v26 học từ raw parity tháng 1 vẫn khớp contract,
frozen invariant và reload, nhưng test `-0,1702R`/5.689 entry, block LCB
`-0,2576R`; do đó lỗi exporter đã được loại trừ nhưng candidate vẫn không đạt.
Thử geometry với pending offset 4,7 ATR trên tháng 2 cũng âm: validation
`-0,2020R`, test `-0,0512R`, LCB test `-0,1326R`.

Replay chéo candidate v26, train trên tháng 1 rồi đóng băng và tự phát lại trên
test tháng 2, cho `5.793` entry/close, mean net-R `-0,2106`, LCB IID
`-0,2308` và LCB theo block `-0,3628`. Các audit fill, delay, offset, SL, TP,
trailing đều đạt `1.0` (fill `0,777`), nên đây không phải lỗi decode geometry.

Phát hiện và sửa lỗi semantics của `action_target_mode=causal`: trước đây value
teacher thay `best_actions` bằng nhãn thresholded của chính surface tương lai,
khiến causal mode không thực sự huấn luyện bằng causal historical lower-bound
labels. Sau khi khôi phục `best_actions` và `causal_action_scores_r`, v27 causal
expected-value vẫn âm trên Jan (`-0,1584R`, block LCB `-0,2298R`). v28 causal
direct tránh surface expected-value nhưng phát `0` entry ở test, vì vậy không
đủ mẫu và không được coi là mean dương hợp lệ.

Các kết quả này không phủ định việc sửa reward lifecycle. v26--v28 là phép học
head offline từ counterfactual net-R nên không đi qua PPO settlement reward.
Reward MT5 hiện đã có settlement net-R có dấu, owner-only lifecycle credit,
MTM chỉ ở bước không settlement và telemetry ledger. Muốn đánh giá shaping mới
phải chạy một ablation MT5 riêng trên cùng seed/ngày: settlement-only so với
settlement + MTM/drawdown/risk shaping; không được thay terminal net-R bằng
shaping để che edge âm. Cho tới khi ablation đó có close đủ mẫu và ledger
reconcile trên các cửa sổ độc lập, hệ vẫn `research_only`.

## Đường dữ liệu

MT5 ledger schema 17 ghi `pending_fill_bars` tại receipt khớp. Python route
reward về đúng `position_owner_trace_id` bất biến của entry (ưu tiên trường
này khi `trace_id` của bản ghi đóng là MANAGE/CLOSE); nếu owner không được
đăng ký thì route bị bỏ qua và không suy đoán. Route miss tuyệt đối không đổ
reward sang agent hoặc timestep khác. Mỗi quyết định mới công bố snapshot
`five_lifecycle_routing` gồm tổng lần route, số được nhận, số miss, số
`INITIAL_SL`, số batch optimizer, tổng mẫu INITIAL_SL đã thực sự đi qua
optimizer và giá trị loss gần nhất. Một `INITIAL_SL` có scalar bằng 0 vẫn bắt
buộc được tính là route cần thiết; cache miss không còn bị ẩn.

## Chẩn đoán tín hiệu Equity không được duy trì 2026-09-19

Ảnh Equity tăng trong một đoạn không chứng minh policy đã học được edge bền:

- Đường xanh là lãi/lỗ nổi theo vị thế mở; Balance chỉ thay đổi khi lifecycle được
  đóng. Profile settlement-only đặt `five_reward_mark_to_market_weight=0`, nên
  phần tăng Equity chưa tạo ra reward kinh tế cho PPO. MTM/drawdown/risk shaping
  chỉ là ablation riêng và không được dùng để che terminal net-R âm.
- Trước bản sửa live, `execution_ledger` chỉ được route ở
  `source=final_evaluation`, sau khi chuỗi quyết định kết thúc. Vì vậy query
  expectancy trong cùng cửa sổ có `five_expectancy_query_record_count=0` và
  policy không thể phản ứng với lifecycle vừa đóng.
- `_refresh_five_competition_lifecycles` nay route final close của trace `five:`
  ngay sau khi ledger bền vững, trước quyết định kế tiếp, với idempotency riêng
  và `settlement_source=live_execution_ledger`. Bằng chứng Feb 01 cho thấy
  `live_settlements=125`, query có record và veto có hiệu lực; ledger reconcile
  vẫn đạt. Đây là sửa đường phản hồi, không phải bằng chứng edge đã dương.
- Veto entry không tự tạo edge: khi chỉ MARKET_BUY bị veto, action mask vẫn cho
  BUY_LIMIT/BUY_STOP hoặc entry phía đối diện. Cửa sổ thử ngưỡng 6 vì thế giảm số
  MARKET_BUY nhưng chuyển hành vi sang các entry khác và xấu hơn: PnL `-1529.24`,
  mean net-R khoảng `-0.4976`, LCB khoảng `-0.6782`, 94 trade.

### Bằng chứng MT5 cùng seed, cửa sổ độc lập

Tất cả các run dưới đây là `research_only`, `metrics-only` hoặc seed chưa có
imitation sidecar; không run nào được coi là candidate deployable.

- 2025-02-01..02, live settlement + veto min 12: PnL `-372.58`, mean policy
  net-R `-0.1164`, LCB `-0.2995`, 126 trade, live settlements `125`, reconcile `1`.
- 2025-02-01..02, veto min 6: PnL `-1529.24`, execution mean net-R khoảng
  `-0.4976`, LCB khoảng `-0.6782`, 94 trade, feedback vẫn về đúng ledger nhưng
  performance thất bại.
- 2025-02-02..03, settlement-only min 12: PnL `+1905.30`, PF `1.72`, mean
  policy net-R `+0.3322`, LCB `+0.1636`, 165 trade; bị loại vì equity DD
  `10.81%` và `negative_context_action_scopes=4`.
- 2025-02-02..03, settlement + MTM/risk shaping: PnL `+609.05`, PF `1.21`,
  LCB `-0.0698`, equity DD `17.34%`; shaping không cải thiện và bị loại.
- Checkpoint được persist sau 2025-02-02..03 rồi nạp sang 2025-02-03..04:
  HOLD đạt `98.1%`, chỉ 38 trade, PnL `-28.06`, execution LCB khoảng
  `-0.340`; run không reconcile do report có 38 trade nhưng ledger lifecycle
  chỉ 36 và còn 2 terminal open positions. Checkpoint có tác dụng mang trạng
  thái sang ngày sau, nhưng trạng thái học được là gần như không giao dịch,
  không phải edge được duy trì.

Kết luận hiện tại: lỗi reward receipt live đã được sửa và được xác nhận bằng
ledger/trace; tín hiệu tốt trong ảnh chưa được chứng minh là edge có thể phát
triển. Nút thắt còn lại là cold-start/risk exposure, action distribution bị
collapse vào HOLD sau khi thấy lỗ, và seed offline vẫn âm. Không được nới
promotion gate để biến một đoạn Equity đẹp thành kết quả đạt.

MTM-only ablation (settlement `1.0`, MTM `0.10`, drawdown/risk/conflict `0`)
cũng đã chạy trên cùng seed: 01/02 PnL `-1031.52`, mean `-0.2740R`, LCB
`-0.4261`, DD `19.20%`; 02/02 PnL `+1379.57`, mean `+0.2752R`, LCB `+0.0492`,
DD `14.86%`. Cả hai reconcile, nhưng cửa sổ 01/02 âm và cả hai fail risk/context
gate. Vì vậy MTM là tín hiệu phụ hợp lệ để audit nhưng chưa phải lời giải duy trì
edge; profile chính vẫn settlement-only cho tới khi có candidate qua independent
window gate.

## Kiểm chứng guard cold-start và validation độc lập 2026-09-19

Đã thử hai biện pháp candidate-only để kiểm tra giả thuyết phơi nhiễm đầu phiên:

- `five_expectancy_action_veto_hold_owner_on_negative=true` buộc owner về HOLD
  khi có action âm đủ mẫu mà không có entry thay thế dương đủ mẫu. Feb 01 cho
  PnL `-1240.29`, 85 trade, LCB khoảng `-0.616`; kết quả xấu hơn veto action-local,
  nên profile chính đã tắt.
- `entry_guard_single_position=true` chỉ cho phép một position/pending order.
  Feb 01 cho PnL `-363.72`, 48 trade, DD `6.17%`; Feb 02 cho `+313.46`, 41
  trade, LCB `-0.1549`. Gộp hai cửa sổ vẫn âm `-50.26`, nên guard đã được trả
  về `false` trong profile chính. Giảm drawdown không đồng nghĩa tạo edge.

Đã thêm profile validation learning-disabled
`configs/experiments/mt5_reward_validation_feb_settlement_only_low_disk.yaml`.
Khi nạp checkpoint persist từ 02/02 và chạy riêng 03/02, validation không còn
trộn expectancy tích lũy của ngày train: chỉ có 1 lifecycle, PnL `-28.08`, LCB
`-1.845`, bị loại vì `insufficient_position_samples` và net âm. Đây là bằng
chứng độc lập rằng checkpoint trước đó đã collapse về gần như HOLD, không phải
đã duy trì được tín hiệu.

## Kiểm chứng vòng lặp tự làm mất tín hiệu và direct action 2026-09-19

Audit trace của run direct cho thấy ledger có `118` mẫu policy âm toàn cục nhưng
`negative_critical_scope_count=0`, mỗi context tối đa chỉ `9` mẫu trong khi gate
yêu cầu `12`. Code cũ vẫn đặt `negative_evidence_strength=0.5`, làm giảm
flat-entry curriculum dù chưa có scope âm đủ bằng chứng. Đã sửa để negative
global không thể làm giảm khám phá khi breadth của scope âm bằng 0; suite liên
quan đạt `110 passed`.

Audit cũng phát hiện exact-context veto dùng ngưỡng thấp hơn profile: ledger mặc
định chỉ cần `6` mẫu và LCB âm bất kỳ, dù profile khai báo `12` mẫu và LCB tối
thiểu `-0.15R`. Đã sửa để exact path áp dụng cùng `min_samples`, confidence và
`min_lcb_r`; test thêm xác nhận `8` mẫu âm chưa được veto. Đây là sửa tính đúng
của guard, không phải bằng chứng profitability.

MT5 direct cùng seed, settlement-only, lifecycle net-R, không veto:

- 01/02 trước floor-fix: PnL `-656.02`, expectancy `-0.2035R`, DD `7.22%`,
  live settlements `120`, reconcile `1`.
- 01/02 sau floor-fix: PnL `+61.65`, expectancy `+0.0190R`, DD `3.47%`,
  live settlements `119`, reconcile `1`, nhưng policy LCB `-0.2105` nên vẫn
  fail gate.
- 02/03 sau floor-fix: PnL `-618.93`, expectancy `-0.2169R`, DD `7.83%`,
  live settlements `111`, reconcile `1`.

Gộp hai cửa sổ direct floor-fix vẫn `-557.28`; do đó cơ chế đã giảm vòng lặp
collapse ở một regime nhưng chưa tạo edge xuyên ngày. Thử action-local veto và
owner-HOLD guard sau ngưỡng mới lần lượt cho 01/02 `-690.80` và `-230.61`,
không được promote. Phân rã lifecycle cho thấy BUY âm ở cả hai cửa sổ, còn SELL
thường dương, nhưng từng loại SELL không ổn định; đây là bằng chứng về regime/
direction asymmetry, chưa đủ để hard-code chỉ SELL.

Kết luận: feedback live đã về đúng owner và PPO thực sự có update, nhưng dữ liệu
không chứng minh một edge ổn định. Cơ chế tiếp theo nên là counterfactual
action-ranking hoặc regime/side adaptation có gate độc lập; không nên tiếp tục
giảm reward hoặc giảm exploration toàn cục để làm đẹp một cửa sổ đơn.

## Adaptive exploration theo owner và kiểm chứng hai ngày 2026-09-19

Đã thêm một nhánh candidate-only `five_live_owner_exploration_*`. Nó giữ
thống kê outcome chỉ trong run hiện tại, lấy sau final lifecycle đã reconcile và
định tuyến qua facade về đúng policy của trace owner. Khi một owner có ít nhất
12 lifecycle, confidence `>=0.20` và LCB `<=-0.15R`, chỉ mức random của owner
đó giảm từ `0.15` xuống `0.02`; owner còn lại, policy logits và reward writer
không bị sửa. Evidence được reset ở đầu run và khử trùng theo trace.

Lần chạy đầu `-716.06` đã phát hiện hook mới chưa đi qua facade
`HeChuyenGiaMotContextMotPolicy`; đã bổ sung forwarding và chạy lại. Trace của
lần chạy đúng cho thấy `786` action dùng mix `0.02`, `1443` action giữ mix
`0.15`; cửa sổ 02/02 tương ứng `1111` và `1502`. Cơ chế thực sự được kích
hoạt theo evidence chứ không chỉ nằm trong config.

Kết quả MT5 sau khi sửa forwarding, cùng seed và fixed geometry:

- 2025-02-01..02: `71` unique final positions, PnL `+488.43`, mean
  `+6.88/trade`, PF `1.43`, DD `3.25%`, position LCB `-1.65`, reconcile `1`.
- 2025-02-02..03: `57` unique final positions, PnL `+190.52`, mean
  `+3.34/trade`, PF `1.21`, DD `3.45%`, position LCB `-5.77`, reconcile `1`.
- Gộp hai cửa sổ: `128` unique positions, PnL `+678.95`, mean khoảng
  `+5.30/trade`.

Đây là cải thiện rõ so với direct floor-fix trước đó (`-557.28` PnL gộp),
nhưng chưa đạt promotion: cả hai LCB đều âm, candidate vẫn research-only vì
seed imitation sidecar bị bypass, và chưa chứng minh checkpoint sau online giữ
được edge khi chạy frozen OOS. Không được đổi gate thành đạt chỉ vì mean dương.

## Vì sao Equity đẹp không được duy trì: behavior-policy mismatch 2026-09-19

Audit sâu hơn cho thấy `five_behavior_uniform_mix` không chỉ là cách lấy mẫu.
Trong `evaluate_sequence`, log-probability dùng cho PPO trước đây là của policy
hành vi sau khi trộn uniform exploration. Vì vậy một lệnh random do sampler
chọn vẫn được actor coi như lệnh của policy hiện tại; khi reward settlement đến
muộn và sparse, gradient có thể học theo các lệnh random thay vì giữ action
gốc tạo ra tín hiệu tốt. Trace cuối run xác nhận raw policy gần như 100% HOLD,
còn entry vẫn xuất hiện nhờ behavior mix. Đường Equity trong ảnh vì thế có thể
đẹp ở behavior sampler nhưng không phải edge đã nằm trong policy để phát triển.

Đã sửa `evaluate_sequence` trả riêng `policy_log_probability` và dùng trường
này cho PPO; log-prob behavior vẫn được giữ để làm old rollout likelihood.
Thêm test xác nhận hai phân phối không bị đánh tráo. Suite liên quan sau sửa:
`179 passed`; compile các module Five/runner cũng đạt.

Các ablation xác nhận cơ chế:

- Tăng `five_flat_entry_mass_loss_weight` lên `8.0` không làm raw entry mass
  thoát collapse (`flat_entry_mass_mean` khoảng `0.00263`), nhưng làm cửa sổ
  01/02 âm `-310.79`, PF `0.74`; ép entry không khôi phục được edge.
- Bật trực tiếp expected-value surface của teacher làm 339 position, PnL
  `-3383.51`, DD `39.05%`; value surface hiện tại là hướng dẫn chứ chưa phải
  economic gate, nên không được dùng làm bộ lọc cứng.
- Với raw-policy log-prob mới, cùng seed/profile adaptive owner: 01/02 đạt
  `+604.02`, 71 unique positions, PF `1.56`, mean `+8.51/position`, DD
  `4.41%`, reconcile `1`; 02/03 đạt `+22.26`, 74 positions, PF `1.02`, mean
  `+0.30/position`, DD `6.37%`, reconcile `1`. Gộp là `+626.28` trên 145
  positions, mean khoảng `+4.32/position`.

Đây là bằng chứng tín hiệu được giữ tốt hơn sau khi actor học đúng policy gốc,
nhưng chưa phải promotion: LCB policy vẫn âm (`-0.0404` ở 01/02 và `-0.1712`
ở 02/03), cửa sổ 02/03 edge chỉ vừa dương và còn negative scope. Một veto
context-local trên riêng 02/03 cho kết quả stdout `+364.24`, PF `1.31`, DD
`4.72%`, LCB `-0.0344`, negative scope `0`, nhưng run không hoàn tất bước
persist do hết dung lượng; chưa có cửa sổ ghép tương ứng nên không được dùng
để tuyên bố đạt.

Kết luận vận hành: hình Equity đẹp là kết quả cấp execution/behavior, chưa
phải bằng chứng policy đã học. Muốn phát triển bền, phải cập nhật actor bằng
raw-policy likelihood, ghi settlement theo owner/lifecycle, rồi chỉ giảm đúng
context-action đã âm sau đủ mẫu; không tăng uniform entry floor và không bật
expected-value gate cứng để chữa collapse.

## Counterfactual HOLD ranking và kiểm chứng độc lập sau patch 2026-09-19

Audit path-credit sau raw-policy fix cho thấy còn một chặn quan trọng: HOLD
của agent entry phải được giữ làm anchor đối chứng nếu action mask của chính
agent đó còn ít nhất một entry hợp lệ. HOLD của position manager/pending
manager/portfolio governor thì không được biến thành ranking anchor. Code tại
`apply_five_pattern_path_credit_detailed` đã tách rõ hai việc này: actor
lifecycle credit vẫn chỉ cấp cho action entry đã thực thi; ranking chỉ mở cho
HOLD hoặc entry trên trace có legal entry `1..6`. Test cũng bao phủ cả HOLD
entry và management-only mask; test mục tiêu pass, `py_compile` pass.

Regression liên quan đạt `334 passed`; còn một failure literal cũ trong test
MQL5 về chữ ký `ExecuteAction`, không chạm vào patch reward/ranking.

MT5 metrics-only cùng seed/profile, hai cửa sổ độc lập:

- 2025-02-01..02: `55` unique positions, net PnL `-37.02`, mean
  `-0.673/position`, PF `0.964`, position LCB `-9.60`, DD `3.94%`,
  duplicate final-close `0`, reconcile `1`.
- 2025-02-02..03: `60` unique positions, net PnL `+227.62`, mean
  `+3.794/position`, PF `1.241`, position LCB `-5.10`, DD `2.80%`,
  duplicate final-close `0`, reconcile `1`.
- Gộp: `115` unique positions, net PnL `+190.60`, mean khoảng
  `+1.657/position`.

Patch đã tạo learning signal thật, không chỉ nằm trong config: cửa sổ đầu có
`148` ranking pairs và `192` supervised anchors sau khi batch 64 đầy; cửa sổ
thứ hai có `180` ranking pairs và `192` supervised anchors. Tuy nhiên signal
chỉ bắt đầu có sau các batch đầu, trong khi raw policy vẫn HOLD khoảng
`97.9%` và `97.65%`. Do đó ngày đầu vẫn âm, cả hai LCB đều âm, fitness vẫn
`eligible=0`, và seed imitation vẫn là research-only vì gate sidecar bị
bypass. Kết luận hiện tại là patch đã mở đúng đường phát triển tín hiệu nhưng
chưa chứng minh edge duy trì qua ngày; chưa được promote.

## Kiểm tra song song cadence và biên độ ranking 2026-09-19

Để tránh tối ưu tuần tự theo cảm tính, đã chạy các candidate MT5 độc lập và
đồng thời giao ba nhánh read-only audit sampler, ranking và tổng hợp ledger.
Hai profile giảm `five_pattern_unroll` xuống `16` và `32` đều phản hồi sớm
nhưng overtrade nghiêm trọng trên 01/02: lần lượt `277` trade, PnL
`-2614.52`, PF `0.50`, DD `29.26%`, LCB `-0.4952`; và `266` trade, PnL
`-2634.49`, PF `0.50`, DD `31.77%`, LCB `-0.4833`. Giữ unroll `64` nhưng
giảm `five_action_ranking_loss_weight` xuống `0.05` còn tệ hơn: `280` trade,
PnL `-3750.45`, PF `0.31`, DD `39.19%`, LCB `-0.6476`. Cả ba ledger đều
reconcile nhưng fitness đều `eligible=0`.

Kết luận kỹ thuật: thiếu update không phải nguyên nhân duy nhất. Khi nhánh
entry thoát HOLD, sampler/entry-mass có thể tạo turnover rất lớn trước khi
lifecycle reward đủ mẫu để sửa lại. Ranking teacher hiện có thể khuếch đại
action mass; giảm cadence hoặc chỉ giảm loss weight không an toàn. Giữ các
profile này ở research-only để làm negative controls, không đưa vào profile
chính và không nới gate để hợp thức hóa kết quả.

## Siết teacher ranking sau kiểm thử song song 2026-09-19

Kết quả audit song song chỉ ra `action_q_values_r` của teacher là chất lượng
đường đi hindsight, chưa phải realized lifecycle PnL. Vì vậy ranking không
được phép coi mọi chênh lệch nhỏ so với HOLD là tín hiệu phát triển. Profile
chính nay yêu cầu đồng thời: mask immutable hợp lệ, agent là một trong hai
entry owner, có ít nhất hai action entry hợp lệ, `q_best >= 0.10R`, và
`q_best - q_hold >= 0.10R`. Mask malformed vẫn có thể dùng fallback cho PPO
để không làm hỏng rollout, nhưng fail-closed đối với ranking teacher.

Patch này đã được test với HOLD entry, management-only HOLD và malformed mask;
agent quản trị không còn nhận counterfactual ranking. Đây là guard chống
khuếch đại nhãn teacher sai, không phải bằng chứng edge đã có. Các ngưỡng vẫn
phải được kiểm chứng lại trên hai cửa sổ MT5 độc lập và ledger unique-position
trước khi cân nhắc promotion.

## Legacy-threshold control sau strict teacher 2026-09-19

Strict teacher giữ `q_best >= 0.10R`, chênh lệch HOLD `>=0.10R` và tối thiểu
hai entry đủ điều kiện, nhưng cửa sổ 01/02 vẫn âm `-139.75` trên `64`
position hoàn tất. Để tách lỗi threshold khỏi reward lifecycle, đã chạy một
control giữ nguyên toàn bộ reward `settlement_net_r_v1`, owner exploration,
raw-policy PPO và fixed geometry; chỉ đưa ngưỡng teacher về legacy
`min_q=0`, `min_hold_delta=0`, `qualified_entries=1`. Immutable mask và giới
hạn agent entry vẫn giữ nguyên.

Kết quả hai cửa sổ độc lập, cùng seed gốc và cùng profile:

- 2025-02-01..02: PnL `+63.26`, `70` unique positions, mean
  `+0.904/position`, PF `1.053`, DD `3.06%`, live settlements `70`,
  duplicate close `0`, terminal open `0`, lifecycle ratio `1.0`, reconcile
  `true`.
- 2025-02-02..03: PnL `+531.20`, `84` unique positions, mean
  `+6.324/position`, PF `1.422`, DD `3.17%`, live settlements `84`,
  duplicate close `0`, terminal open `0`, lifecycle ratio `1.0`, reconcile
  `true`.
- Gộp theo `(window_id, position_ticket)`: PnL `+594.46` trên `154`
  position, mean `+3.860/position`.

Đây là lần đầu control thỏa tiêu chí research mean dương ở cả hai cửa sổ và
ledger được reconcile. Strict threshold trước đó đã lọc quá mạnh tín hiệu
teacher trong profile này; không được diễn giải thành bằng chứng teacher
hindsight là nhãn kinh tế đúng. Control vẫn chưa promotion được: cửa sổ đầu
có policy LCB `-0.2553R`, cửa sổ hai có một negative context scope, cả hai
dùng tester model `1` nên real-tick evidence chưa verified, seed imitation bị
bypass và chạy `metrics-only` không tạo checkpoint deployable.

Một lỗi vận hành được cố định trong quá trình này: candidate runner chỉ áp
dụng `profile.learning_overrides`, không tự kế thừa `source_config` nằm trong
profile. Các profile strict sau đó ghi rõ toàn bộ reward/lifecycle/decision
override; nếu không, run sẽ rơi về default và tạo turnover giả, không được
đưa vào bảng so sánh.

## Train rồi frozen validation: phân biệt cache miss thật với retry trùng 2026-09-19

Run train legacy-threshold sau cachefix trên `2025-02-01..02` có report PnL
`+145.31`, PF `1.15`, `57` report trades và `2` vị thế còn mở ở terminal.
Audit unique completed-position trên execution ledger chỉ giữ `55` vị thế,
net `+102.82`, mean `+1.869/position`, PF `1.103`, reconcile `true`. Quan
trọng hơn, `events.jsonl` có `55/55` lifecycle online đều `credited=3`; vì vậy
reward live đã đi vào owner policy.

`execution_trace_observatory` trước đó vẫn ghi `55` dòng `cache_miss` ở
`settlement_source=final_evaluation`. Truy nguyên cùng trace cho thấy đây là
lần retry sau khi lifecycle live đã pop `_execution_trace_cache`, không phải
MT5 thiếu fill/close hay ledger mất `position_owner_trace_id`. Runner nay bỏ
qua final retry nếu identity `(position_ticket, trace_id, close_reason,
realized_r)` đã được settle live; như vậy hậu kiểm không biến double-pass
thành lỗi routing. Đồng thời lifecycle owner packet cache được giữ qua
`clear_rollout=True`, và regression test boundary đã pass.

Validation frozen deterministic với `mt5_learning_enabled=false` và tắt
sampling cho kết quả `0` trade, reconcile `true`. Đây là một failure hợp lệ:
train arm đã dùng behavior sampling/exploration, nhưng frozen arm lại ép
`deterministic=True`, nên không thể dùng nó để chứng minh candidate tổng quát.
Profile behavior-compatible tiếp theo bật
`five_frozen_behavior_sampling_enabled=true` với seed cố định `20260702`, vẫn
tắt learning/update. Trên `2025-02-02..03`, kết quả là report PnL `+220.02`,
PF `1.12`, `115` report trades; unique completed-position ledger có `113`
vị thế, net `+188.09`, mean `+1.665/position`, PF `1.099`, reconcile `true`.
Policy-origin mean là `+0.0921R` nhưng LCB `-0.1150R`, và terminal còn `2`
vị thế. Gộp train + validation theo unique completed position là `168` vị
thế, net `+290.91`, mean khoảng `+1.732/position`, nhưng đây chỉ là research
evidence; chưa đạt promotion vì LCB âm, terminal-open, behavior sampling còn
là một phần của contract, tester model `1` chưa có real-tick marker và seed
imitation gate vẫn bị bypass.

Kết luận: đã chứng minh reward live được credit đúng owner và đã loại bỏ
false-positive cache miss ở final retry, nhưng chưa chứng minh một policy
deterministic frozen có edge bền vững qua ngày. Không được gộp arm `0 trade`
và arm frozen-sampling vào cùng một bảng kết quả; chúng trả lời hai câu hỏi
khác nhau về khả năng tái tạo hành vi.

Telemetry cũng được gia cố: writer của `mt5_diagnostics.json` nay có lock
chung giữa monitor thread và `close()` final publish, tránh race khi hai bên
cùng dùng file tạm trên Windows (`WinError 5`). Writer cũng retry ngắn khi UI
hoặc lệnh polling đang giữ handle đọc không cho `os.replace`; lỗi quan sát tạm
thời không được làm chết monitor hay ảnh hưởng training. Test retry publish đã
pass.

## Cổng terminal phẳng cho validation/promotion 2026-09-19

`reconciled=true` không đồng nghĩa mọi vị thế đã đóng. Reconciliation có thể
đúng khi report net profit bằng `closed-ledger-net + terminal mark-to-market`
trong khi `terminal_open_positions > 0`; lifecycle được settle từ
`positions_pipe` trong trường hợp này chỉ là mark cuối tester, không phải deal
close thật.

Vì vậy `require_terminal_flat_for_promotion=true` nay là một gate riêng:

- training vẫn có thể ghi nhận terminal mark để không mất feedback và để chẩn
  đoán;
- fitness/promotion bị loại với lý do
  `terminal_open_positions_nonzero:<n>` nếu tester kết thúc còn vị thế mở;
- test accounting vẫn giữ kỳ vọng `reconciled=true`, còn test promotion phải
  yêu cầu thêm `terminal_open_positions == 0` và lifecycle close thật.

Điều này ngăn đường equity đẹp do floating PnL hoặc synthetic terminal mark bị
đưa nhầm vào bằng chứng policy đã học ổn định.

## Kiểm chứng chéo cửa sổ và nguyên nhân tín hiệu không được duy trì 2026-09-19

Các run dưới đây dùng cùng checkpoint gốc và cùng hợp đồng settlement-net-R;
chỉ khác selector/teacher. Số `mean/position` được tính trên unique
`position_ticket` đã có close thật trong execution ledger, không lấy số dòng
final-close hoặc terminal mark làm một trade mới.

| Arm | Cửa sổ | Vị thế hoàn tất | Net USD | Mean USD/vị thế | Terminal mở | Reconcile | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|
| legacy teacher control | 01-02/02 | 70 | +63.26 | +0.90 | 0 | true | research-only; policy LCB âm |
| legacy teacher control | 02-03/02 | 84 | +531.20 | +6.32 | 0 | true | research-only; còn negative scope |
| legacy teacher control | 03-04/02 | 79 | -181.34 | -2.30 | 1 | true | fail |
| strict teacher | 03-04/02 | 58 | -83.99 | -1.45 | 0 | true | fail |
| strict teacher + path reward | 03-04/02 | 30 | -606.03 | -20.20 | 0 | true | fail nặng hơn |
| strict teacher + expectancy veto | 03-04/02 | 31 | âm; report -492.86 | âm | 0 | true | fail |
| expected-value surface | 03-04/02 | 157 | +923.03 | +5.88 | 3 | true | bị loại: terminal-open và 6 negative scopes |
| expected-value surface | 01-02/02 | 308 live settlements | âm; report -3307.79 | âm | 7 | true | fail nặng |
| expected-value + veto | 01-02/02 | 93 | âm; report -1224.50 | âm | 0 | true | fail |

Điểm quan trọng là `reconciled=true` chỉ xác nhận report và ledger khớp; nó
không xác nhận edge. Với expected-value surface, 03-04/02 có kết quả đẹp nếu
chỉ nhìn một cửa sổ, nhưng cùng selector trên 01-02/02 bị âm mạnh. Vì vậy
không được gộp 03-04/02 vào bằng chứng deployable và cũng không được chọn
checkpoint theo cửa sổ đẹp nhất.

### Chẩn đoán calibration

Phân tích lifecycle có cùng `action_details` cho thấy value teacher không giữ
được calibration khi regime đổi:

- trên 01-02/02, `MARKET_BUY` có q dự đoán trung bình khoảng `+0.69R` nhưng
  realized mean khoảng `-0.48R` trên `293` lifecycle;
- trên 03-04/02, cùng action có q khoảng `+0.92R` và realized mean khoảng
  `+0.16R` trên `149` lifecycle.

Do đó path reward `0.10R` không sửa được vấn đề: nó lấy q counterfactual của
teacher làm shaping trước khi có close thật, và run 03-04/02 rơi xuống PF
`0.11`, win-rate `6.7%`. Expectancy veto cũng chỉ phản ứng sau khi đủ mẫu;
ngưỡng 12 mẫu không thể bảo vệ 12 lệnh đầu của một regime mới. Chuyển sang
expected-value trực tiếp làm kết quả 03-04/02 tốt hơn, nhưng đồng thời gây
over-trading trên 01-02/02 (`315` trades, PF `0.43`, drawdown `34.99%`).

### Trạng thái xác thực

Reward MT5 hiện đã đi đúng causal route tới owner policy và lifecycle ledger;
failure hiện tại là generalization/calibration của selector, không phải
`cache_miss` hay thất lạc reward. Chưa có candidate nào thỏa đồng thời:

1. mọi cửa sổ độc lập đều có mean net profit trên unique completed position
   dương;
2. mọi cửa sổ đều terminal-flat và reconciled;
3. kết quả gộp dương với policy/context confidence đủ an toàn.

Vì vậy trạng thái vẫn là **chưa đạt, research-only**. Các run tiếp theo phải
được đánh giá theo cửa sổ mới chưa dùng để chọn ngưỡng; không tiếp tục tăng
reward shaping hoặc đổi threshold trên cùng cửa sổ để tạo candidate đẹp.

## Candidate thuần net-R

Các profile trước đây có tên `settlement_net_r_v1` nhưng vẫn để mặc định
`five_lifecycle_reward_enabled: true`. Vì vậy mỗi final close vừa cấp net-R
cho owner, vừa cấp V4 geometry/MFE/trailing/pending shaping. Hai tín hiệu này
khác bản chất và có thể xung đột: một entry lỗ nhưng từng có MFE hoặc fill
pending tốt vẫn nhận scalar phụ dương, làm actor giữ hành vi không có lợi nhuận
cuối cùng.

Profile nghiên cứu mới
`configs/experiments/mt5_reward_ablation_feb_direct_adaptive_owner_pure_net_r_low_disk.yaml`
đã tách mục tiêu này rõ ràng: tắt V4 shaping, tắt pattern reward và failure
replay phụ, tắt MTM/competition/rank/risk penalties, rồi chỉ giữ net-R cuối
được route đến immutable entry owner. Đây mới là arm để kiểm tra giả thuyết
“tín hiệu tốt được duy trì vì outcome kinh tế được thưởng đúng owner”; nó chưa
được coi là candidate đạt cho đến khi qua đủ các cửa sổ MT5 độc lập và ledger
reconciliation.

### Kết quả pure net-R trên hai cửa sổ độc lập 2026-09-19

Hai cửa sổ `2025.02.01..02` và `2025.02.02..03` đã được chạy đồng thời sau khi
cô lập runner theo từng output candidate. Cả hai run đều hoàn tất tự nhiên, có
ledger unique-ticket sạch, owner không thiếu và `reconciled=1`; vì vậy đây là
bằng chứng hợp lệ, nhưng kết quả đều âm:

- `2025.02.01..02`: 57 unique completed positions, ledger net `-333.89 USD`,
  mean `-5.8577 USD/position`, policy LCB `-0.3851R`; còn 3 vị thế mở cuối
  tester nên promotion gate cũng loại.
- `2025.02.02..03`: 58 unique completed positions, ledger net `-146.84 USD`,
  mean `-2.5317 USD/position`, policy LCB `-0.3595R`, terminal flat.
- Gộp theo unique position: 115 positions, net `-480.73 USD`, mean
  `-4.1803 USD/position`. Không được lấy đoạn Equity đẹp hoặc report close-row
  trùng làm lý do loại bỏ các kết quả âm.

Audit route không tìm thấy duplicate ticket hoặc missing owner. Nguyên nhân
không nằm ở việc reward thất lạc: profile pure đã đặt
`five_pattern_reward_enabled=false`, nhưng runtime khi thấy trace `five:` vẫn
chọn `apply_five_pattern_path_credit_detailed`; hook này trả
`pattern_disabled`. Vì vậy khoảng `3.790`/`3.835` mature rows bị reject và có
`pattern_updates=0`, làm policy giữ `HOLD` khoảng `97.99%`/`97.84%`. Tắt V4
đã vô tình tắt luôn delayed action-selection supervision, trong khi signed
net-R thưa không đủ kéo policy ra khỏi HOLD.

Đã tách hai kênh bằng
`five_entry_preference_supervision_enabled` và hook Five-native
`apply_five_entry_preference_credit_detailed`: selector chỉ giữ ranking
HOLD-vs-entry trên path bất biến, reward actor và toàn bộ V4 tree weights đều
đặt bằng 0. Profile thử nghiệm là
`configs/experiments/mt5_reward_ablation_feb_direct_adaptive_owner_net_r_selector_low_disk.yaml`.
Đây là arm mới, chưa được coi là đạt; phải chạy lại đúng hai cửa sổ trên và
chỉ tiếp tục nếu selector rows được accept, không collapse, ledger sạch và
mean unique-position dương trên từng cửa sổ lẫn kết quả gộp.

Lần chạy selector song song đầu tiên đã xác nhận route mới hoạt động nhưng
chưa sửa được edge:

- `2025.02.01..02`: 52 unique completed positions, ledger net `-201.31 USD`,
  mean `-3.8713 USD/position`, policy LCB `-0.2916R`, terminal flat.
- `2025.02.02..03`: 52 unique completed positions, ledger net `+27.33 USD`,
  mean `+0.5256 USD/position`, nhưng report còn 1 vị thế mở và policy LCB
  `-0.2333R`; promotion gate loại cửa sổ này.
- Gộp 104 unique completed positions: net `-173.98 USD`, mean
  `-1.6729 USD/position`.

Tất cả mature selector rows đều được accept (`3215` và `3200`), không reject,
không pending; mỗi run tạo 10 selector optimizer updates. Tuy nhiên policy vẫn
collapse về `HOLD` `98.12%` và `97.96%`. Vì vậy lỗi reject do route cũ đã được
loại trừ, nhưng selector-only ranking quá thưa/yếu để hồi phục một policy đã
collapse; signed net-R cũng chưa tạo được expectancy dương có LCB. Nhánh này
vẫn `research-only`, không promotion và chưa đủ lý do chạy thêm ngày thứ ba
trên cùng arm.

## Chạy song song các cửa sổ OOS mà không làm bẩn ledger 2026-09-19

Các cửa sổ OOS độc lập nay có thể chạy bằng
`scripts/run_frozen_temporal_generalization.py --max-workers N`. Mỗi worker
clone một MT5 portable riêng dưới runtime root ngắn, có namespace riêng cho `Common/Files`,
`MQL5/Files`, `Reports`, bridge, execution ledger và artifact; các file history
và real-tick thuộc đúng cửa sổ được copy riêng để MetaTester không ghi index/cache
lên inode dùng chung, còn metadata bất biến mới được hard-link. Checkpoint được load read-only,
kết quả được gom theo số cửa sổ sau khi các worker hoàn tất. Mặc định vẫn là
`--max-workers 1` để giữ tương thích với workflow cũ.

Clone MT5 cũng bỏ NTFS compression kế thừa khỏi các cây `MQL5/Experts`,
`MQL5/Files`, `MQL5/Profiles` và `Common/Files`; MetaTester có thể báo lỗi
`cannot load expert` hoặc không đóng gói được `tester_file` khi các file bridge
ở trong thư mục bị đánh dấu `Compressed`.

Launcher `.ini` và EA `.set` được ghi atomically rồi mới spawn `terminal64.exe`;
nếu MT5 nhìn thấy file chưa hoàn chỉnh, nó có thể rơi về profile cũ và không
khởi động Strategy Tester.

Chỉ được tăng `N` khi các cửa sổ thật sự độc lập. Các bước có dependency như
`train -> freeze -> validation` của cùng một candidate vẫn phải giữ thứ tự;
chỉ các candidate hoặc cửa sổ đã có artifact độc lập mới được chạy đồng thời.
MT5 không được chạy song song nếu dùng chung runner, bridge, report, port hoặc
ledger, vì khi đó reward/ack/close có thể bị quy nhầm owner và làm hỏng bằng
chứng causal. Trên bundle hiện tại, dù đã tách runner, Strategy Tester vẫn bind
agent ở `127.0.0.1:3000` và MCP ở `127.0.0.1:22346`; vì vậy
`MT5StrategyTesterEvaluator.evaluate_policy` dùng mutex host-level để chỉ cho
một đoạn MT5 độc quyền chạy tại một thời điểm. Các worker vẫn có thể chuẩn bị
checkpoint, audit và tổng hợp song song; worker thứ hai sẽ chờ mutex thay vì
tạo report `zero-action` không hợp lệ. Thời gian chờ và đường lock được ghi
trong manifest ở `mt5_global_lock_wait_seconds` và `mt5_global_lock_path`.

Đối với candidate MT5 online, `scripts/run_tensor_rule_v186_candidate.py` tạo
namespace mutable ngắn dưới `.mt5_runners` ở workspace root, rồi đặt tên runner
bằng PID + nanosecond. Không đặt runner sâu trong output candidate vì đường dẫn
MT5/Windows có giới hạn thực tế ngắn hơn NTFS; output candidate vẫn giữ manifest
và diagnostics để đối chiếu. Namespace được cấp riêng cho từng process, nên hai
candidate khởi động đồng thời không dùng chung bridge/report/ledger. Unit test
kiểm tra hai output nhận hai runner khác nhau dưới cùng short workspace root.
