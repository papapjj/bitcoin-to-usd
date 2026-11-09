# Thử nghiệm cô lập drift Full-RL A/B/C

Ba nhánh dùng cùng một bản sao model, seed, khoảng dữ liệu và chế độ mô phỏng
MT5. Model nguồn chỉ được đọc và được kiểm tra SHA-256 trước/sau.
Khoảng mặc định là 2025-01-07 đến 2025-01-08; mật độ giao dịch của model hiện
tại đủ tạo hàng chục lifecycle trong cửa sổ này.

- A: tắt toàn bộ học trong MT5, chỉ inference.
- B: bật Full-RL batch với learning rate Direct-RNN `1e-6`.
- C: bật Full-RL batch với learning rate hiện tại `5e-4`.

Trước khi load mỗi bản sao, runner đặt lại cùng seed cho Python và NumPy.

Runner: `scripts/run_full_rl_drift_abc.py`.

Với context-expert CLI hiện tại, một nhánh frozen tương đương phải dùng
`huan-luyen-he-chuyen-gia-mt5 --frozen`. Cờ này chỉ thay
`mt5_learning_enabled=false`: action surface, bridge schema, owner routing và
broker execution giữ nguyên; không được dùng frozen để thay đổi mask/gate hay
điều kiện vào lệnh. Vì vậy nó là đối chứng trực tiếp cho một run online cùng
seed/cửa sổ.

Mỗi nhánh lưu report MT5, execution ledger, bridge, chuỗi
equity/balance/drawdown, chuỗi norm/sum của Direct-RNN và `risk_cost_gate` sau
từng lifecycle settlement, cùng trạng thái học trước/sau. Kết quả tổng hợp nằm
trong `outputs/he_chuyen_gia/audit/full_rl_drift_abc_20260723/comparison.json`.

Build MT5 hiện tại không khởi động tester thứ hai khi `/config` được forward
vào terminal portable đang mở. Runner vì vậy restart đúng terminal portable
giữa các nhánh; thông tin đăng nhập lưu sẵn và model nguồn không đổi.

## Contract Full-RL hiện tại (2026-07-24)

Kết quả A/B/C bên dưới là lịch sử trước khi contract được hoàn thiện, không
được dùng để suy ra hiệu quả của bản hiện tại. Full-RL nay có một bề mặt hành
động duy nhất: Direct-RNN sau output normalization rồi hard execution scope
`HOLD + entry`. Các action close/manage không còn là action policy vì không có
credit lifecycle tương ứng; broker SL/TP vẫn quản trị vị thế.

- `risk_cost_gate`, action-parameter head, parameter replacement, context
  direction prior, expectancy delta, learned gate/mask, action-recovery,
  selector utility, expert fitness/history và
  memory/curiosity/reasoning/causal/world-model/partitioned-memory không được
  phép đổi action hoặc ghi state học trong Full-RL. Các trace bridge cần thiết
  cho settlement chỉ là cache tạm thời và bị dọn khi lifecycle kết thúc.
- Hidden dropout ở mọi lớp đã cấu hình chạy theo mask xác định trong mọi MT5 Full-RL
  forward, kể cả nhánh frozen. Vì vậy paired frozen/online có cùng bề mặt
  action. Runtime reset đưa dropout step về 0 cho hai arm cùng seed. STP
  per-tick vẫn bị tắt riêng cho Full-RL để `w_in` mà AdamW cập nhật chính là
  bề mặt đã phát lệnh.
- Frozen không chỉ tắt AdamW: canonical lifecycle settlement không được tạo
  geometry target, advantage EMA, conditional expectancy/audit, TimingNet
  label, selector evidence, expert fitness/history, neuron utility EMA hoặc
  queue Full-RL. Nó chỉ dọn trace runtime đã settle và ghi event telemetry
  `credited=0`. Nếu một field outcome-derived trong artifact frozen khác seed,
  đối chứng bị loại; đây là invariant bắt buộc của V168 trở đi.
- Output normalizer có VJP đúng trong backprop; `rms_v1` không còn chỉ là
  transform phía trước khiến logit thực thi và gradient lệch nhau.
- Trace giữ tối đa `direct_rnn_rl_temporal_bptt_steps` full forward states;
  lifecycle gradient được TBPTT qua base và mọi deep recurrent state, không
  còn mặc định dừng ở TBPTT=1.

Các event `direct_rnn_full_rl_batch` phải có
`policy_eligible_count == batch_size`, `temporal_bptt_steps >= 1` và optimizer
chỉ ghi `full_lifecycle_rl`. Bất kỳ sample nào bị action ngoài contract đổi
winner là lỗi contract, không được âm thầm bỏ policy gradient.

### V169–V170 forensic và contract V171 (2026-07-28)

Probe online V169 đã phục vụ 503 decision, settle 151 lifecycle và tạo 15
batch AdamW. Kết quả thật vẫn âm: `-9.40977R`, WR `47.02%`, PF `0.8609`.
MARKET_SELL và BUY_LIMIT dương riêng lẻ, nhưng MARKET_BUY, SELL_LIMIT và STOP
âm; vì vậy đây **không phải edge**.

V169 cũng chứng minh một lỗi attribution còn lại: ngoài AdamW của Direct-RNN,
facade vẫn ghi `expert.ghi_ket_qua()` và `PolicyEnsembleSelector.observe_outcome()`;
các wrapper V8/V9 còn làm reasoning/curiosity/world-model/memory state drift.
PnL V169 do đó chỉ dùng để tìm lỗi, không được gán cho Full-RL thuần.

Probe V170 tiếp theo phục vụ 666 decision, settle 161 lifecycle và tạo 21
batch AdamW mà không có bridge/build error. Nó xác nhận selector, expert
history/fitness, action-recovery, reasoning, curiosity, partitioned-memory và
world-model không đổi; `selector_update_skipped_full_rl=161`. Tuy nhiên so
sánh artifact còn thấy `ExperienceSuperNetwork.contextual_expectancy` và
`neural_cells` tăng. Nguồn là runner gọi `record_position_context_extrema()`
sau lifecycle; hàm này ghi bài học extrema vào memory ngoài batch. Vì vậy V170
cũng forensic-only, PnL `-34.285567R`, WR `43.48%`, PF `0.5537` không phải
bằng chứng chất lượng policy.

V171 chặn position-extrema writer ở cả policy và facade trong Full-RL. Nó trả
event audit `full_rl_position_extrema_memory_skipped=true` nhưng không append
extrema, contextual-expectancy hay neural cell. TimingNet vẫn được phép học
nhãn timing riêng từ lifecycle vì nó không được đọc như action delta/gate; nó
không có quyền đổi BUY/SELL/HOLD hay logits Direct-RNN. Probe MT5 mới phải
chứng minh đồng thời: tensor Direct-RNN thay đổi qua batch, còn selector,
expert history/fitness, action-recovery, reasoning, curiosity, toàn bộ
ExperienceSuperNetwork, partitioned-memory và world-model giữ nguyên sau khi
đã bỏ cache runtime.

V172 khôi phục một ngoại lệ đúng contract: manager trace phát
`POLICY_CLOSE_*` nhận delayed counterfactual `final_R - hold_R` và được queue
vào `direct_rnn_exit_rl_batches`, sau đó đi qua chính
`apply_rnn_adamw_cosine(source="full_lifecycle_rl")`. Đây không phải legacy
exit credit: facade không ghi expert history/fitness hoặc selector utility,
và không có writer thứ hai. Không được tắt queue này chỉ vì cô lập facade, vì
khi đó các output CLOSE/MANAGE vẫn được phát nhưng không có loss để học.

V172 cũng làm lộ lỗi bridge EA schema 12: `POST_EXIT_OBSERVATION` tính đúng
`hold_R` sau 12 nến M1 nhưng bật cờ `postExitEmitted` **sau** khi
`AddExecutionEvent()` gọi `LifecycleAuditValues()`, nên cột
`post_exit_horizon_bars` trong ledger bị ghi `0`. V173 sửa thứ tự này trong EA
và Python chỉ dùng `age_bars` của chính receipt HORIZON làm fallback tương thích
cho ledger cũ có số `0`; fallback không áp dụng cho lifecycle thường và không
tạo future leak.

V174 là probe MT5 thật từ seed bất biến sau khi MetaEditor build `0 error, 0
warning`: phục vụ 603 decision, settle 176 lifecycle, trong đó 53
`POLICY_CLOSE_*` và 52 receipt đã tới horizon. File execution-trace observatory
ghi đúng **52/52** `exit_counterfactual_credit` theo ticket với horizon 12;
checkpoint cũng chứa đủ receipt. Stream live tổng hợp chỉ hiện 46 ack vì sáu
credit được settle sau callback feedback cuối, không phải credit bị mất. Vì vậy
observatory/ledger là nguồn kiểm chứng routing; trường `credit=0` ở một receipt
nghĩa là sample đã queue nhưng chưa tự nó đủ batch 8, không phải bị từ chối.
Full-RL giữ cache tạm thời cho CLOSE_WORST/BEST/ALL đã broker-confirmed dù chúng
không `memory_eligible`, để trace tồn tại tới horizon; cache này không ghi memory
hay writer legacy. Runtime reset cũng xóa `policy_selector.last_trace`, fallback
và cờ cold-start tạm thời nhưng giữ nguyên learned selector evidence.

V174 không chứng minh edge: net `-51.946021R`, WR `39.20%`, PF `0.435`. Sau
runtime reset, selector learned maps, risk/action head, reasoning graph,
ExperienceSuperNetwork cells/decision expectancy, recovery, world model và
partitioned memory đều bằng seed. Chỉ Direct-RNN/critic/conditioned
representation, TimingNet và hai ledger audit-only thay đổi.

### Advantage actor chống nhiễu (candidate-only)

`learning.direct_rnn_rl_advantage_mode=critic_asymmetric_v1` thay EMA-only
bằng baseline lai chỉ đọc từ trace entry bất biến: `B=(1-w) EMA(R)+w V(s)`.
`w` tăng dần từ 0 đến `direct_rnn_rl_critic_max_weight` trong warm-up lifecycle
để critic non không được phép phạt policy quá sớm. Target của value head vẫn là
canonical raw-R từ MT5; `V(s)` bị stop-gradient trong actor loss.

Actor dùng residual `A=R-B` bất đối xứng: net win trên profit floor chỉ có
`max(A, 0)`, net loss dưới loss floor chỉ có `min(A, 0)`, vùng hòa vốn là 0.
Nhờ vậy critic dự đoán quá cao không biến một lệnh lãi nhỏ thành actor penalty.
Nếu bật normalization, chỉ actor signal được chia bởi EMA độ lệch chuẩn causal
của residual (có scale floor và clip); critic vẫn fit raw-R. Telemetry batch
ghi raw/effective advantage, scale, mode và số lần profit/loss shield để MT5
audit phân biệt learning signal với PnL thực.

## Kết quả paired run 2026-07-23

Khoảng MT5 2025-01-07 đến 2025-01-08, model 1, seed 20260723:

| Nhánh | Net profit | PF | WR | Lệnh | DD report |
|---|---:|---:|---:|---:|---:|
| A freeze | -294.44 | 0.56 | 44.55% | 101 | 4.59% |
| B Full-RL 1e-6 | -225.94 | 0.62 | 44.94% | 89 | 3.14% |
| C Full-RL 5e-4 | -112.12 | 0.67 | 50.79% | 63 | 1.36% |

A có learning state trước/sau bằng nhau. B có 12 optimizer step mới:
Direct-RNN L2 đổi `2.096e-6`, risk gate L2 đổi `0.014338`, gate age tăng 39.
C có 8 optimizer step mới: Direct-RNN L2 đổi `0.00119153`, risk gate L2 đổi
`0.0210371`, gate age tăng 38.

Trong paired run này C tốt hơn A và B ở mọi metric chính, nhưng gate độc lập
khi ấy còn thay đổi; vì vậy run này chỉ là baseline lịch sử. MT5 execution
feedback cũng không hoàn toàn deterministic giữa các lần chạy, nên bản contract
mới phải lặp lại paired frozen-OOS trước khi kết luận thống kê.
