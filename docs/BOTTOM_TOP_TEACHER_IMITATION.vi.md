# Historical Bottom/Top Teacher v4 cho imitation learning

## Mục tiêu

`BottomTop_Teacher_EA` là bộ gắn nhãn offline, không đặt lệnh. Strategy Tester
phát lại giá lịch sử theo thứ tự để EA chụp observation nhân quả 137 chiều và
tạo nhãn entry:

- BUY sau khi swing low đã có đủ ba bar bên phải để xác nhận, thoát tại swing high hợp lệ kế tiếp;
- SELL sau khi swing high đã có đủ ba bar bên phải để xác nhận, thoát tại swing low hợp lệ kế tiếp;
- HOLD nếu không phải pivot hoặc mô phỏng lệnh không có PnL ròng dương.

EA raw vẫn ghi pivot hậu nghiệm để giữ bằng chứng; builder schema 9 dời nhãn tới
observation xác nhận, tính lại giá vào và dựng target geometry tại thời điểm vào
khả thi. Không có `CTrade`, `OrderSend`, position thật hoặc lifecycle chờ tương lai.

## Giá thực thi, spread và SL

OHLC của MT5 được xem là giá Bid:

- BUY: vào tại `Ask = close Bid + spread`; thoát tại Bid của swing high kế tiếp;
- SELL: vào tại close Bid; thoát tại `Ask = low Bid + spread` của swing low kế tiếp.

TP không dùng khoảng cách ATR hoặc tỷ lệ RR cố định. `opposite_exit_price` chính
là giá thoát thực thi ở pivot đối diện kế tiếp trong `OutcomeHorizonBars=48`.

SL là SL cấu trúc:

- BUY: thấp hơn low của pivot vào lệnh `PivotStopBufferAtr=1.50` ATR;
- SELL: cao hơn high Ask của pivot vào lệnh `1.50` ATR.

Khoảng đệm SL hiện hành được tăng gấp đôi từ 0,75 ATR lên 1,50 ATR. Đây là độ xa
tính từ cực trị pivot; giá entry vẫn tính theo Bid/Ask và spread như cũ.

EA quét từng bar từ entry đến pivot đối diện. Nếu SL bị chạm trước, nhãn entry bị
từ chối và trở thành hard-negative HOLD. Nếu bar pivot đối diện đồng thời chạm
SL, SL được ưu tiên theo hướng bảo thủ.

Các đại lượng được ghi riêng:

- `teacher_gross_pnl_price`: PnL theo Bid chưa trừ spread;
- `teacher_spread_cost_price`: spread của lần vào/thoát tương ứng;
- `teacher_net_pnl_price = gross PnL - spread`;
- ba đại lượng tương ứng theo R: `teacher_gross_r`,
  `teacher_spread_cost_r`, `teacher_net_r`;
- giá entry, SL, opposite exit, exit reason, MFE/MAE và thời gian pivot đối diện.

BUY/SELL chỉ được chấp nhận khi tìm thấy pivot đối diện, SL chưa chạm trước và
`teacher_net_pnl_price > 0` sau khi làm tròn theo số chữ số giá của symbol.
Sai số dấu phẩy động nhỏ đến mức khi ghi CSV thành `0.00` phải là HOLD, không
được coi là lệnh teacher dương. Không có ngưỡng RR hay spread-R tùy ý.

## Hợp đồng chống leakage

`inputs_pipe` chỉ chứa 137 feature có tại `sample_time`. Pivot đối diện, spread
thoát, PnL, SL outcome và nhãn là metadata tương lai, chỉ dùng lúc train:

- `causal_input=1`;
- `future_derived_label=1`;
- `label_finalized_time > sample_time`;
- schema hiện hành: v4, source `historical_opposite_pivot_net_pnl_v4`.

## Hiển thị trong Strategy Tester

Profile visual bật `Visual=1`, `EnableChartMarkers=true`. Mỗi lệnh teacher hợp
lệ được vẽ thành một cụm riêng:

- mũi tên xanh kèm `BUY ENTRY`: điểm vào BUY theo Ask;
- mũi tên đỏ kèm `SELL ENTRY`: điểm vào SELL theo Bid;
- đoạn chấm đỏ `SL`: stop-loss cấu trúc từ lúc vào đến lúc thoát;
- đoạn gạch cyan `TP/EXIT`: giá thoát thực thi tại pivot đối diện;
- đường vàng: nối giá entry với giá exit thực tế;
- chấm cyan tại cuối đường: thời điểm và giá thoát;
- dấu xám: pivot bị từ chối thành HOLD do SL hoặc PnL ròng không dương;
- HOLD thường không vẽ.

Góc trái chart có legend cố định. `TP/EXIT` chỉ là cách hiển thị dễ đọc cho
điểm thoát pivot đối diện hậu nghiệm; EA không đặt TP RR hoặc lệnh TP thật.

Tooltip hiển thị entry, SL, opposite exit, exit reason, spread-R và net PnL.
Nhãn xuất hiện sau tối đa 48 bar vì phải tìm và xác nhận pivot đối diện. Nếu cửa
sổ nằm sau terminal, dùng Alt+Tab chọn `Strategy Tester Visualization`.

## File và cách chạy

- EA: `mt5_v2_runner/MQL5/Experts/BottomTop_Teacher_EA.mq5`;
- profile nền: `mt5_v2_runner/MQL5/Profiles/Tester/BottomTop_Historical_Labeler.set`;
- profile visual: `mt5_v2_runner/MQL5/Profiles/Tester/BottomTop_Historical_Labeler_Visual.set`;
- config nền: `mt5_v2_runner/bottom_top_historical_labeler_7d.ini`;
- config visual: `mt5_v2_runner/bottom_top_historical_labeler_visual.ini`;
- builder: `scripts/build_teacher_imitation_dataset.py`;
- validator: `src/self_growing_trader/teacher_imitation_dataset.py`.

Compile và chạy visual:

```powershell
$src = (Resolve-Path 'mt5_v2_runner\MQL5\Experts\BottomTop_Teacher_EA.mq5').Path
$log = (Join-Path (Resolve-Path 'outputs').Path 'compile_bottom_top_historical_opposite_pivot_v4.log')
& .\mt5_v2_runner\MetaEditor64.exe /portable "/compile:$src" "/log:$log"
& .\mt5_v2_runner\terminal64.exe /portable `
  "/config:$((Resolve-Path 'mt5_v2_runner\bottom_top_historical_labeler_visual.ini').Path)"
```

Dựng NPZ:

```powershell
python scripts\build_teacher_imitation_dataset.py `
  --labels <Common Files>\bottom_top_historical_opposite_pivot_v4_16w_sl075.csv `
  --output outputs\he_chuyen_gia\v193_repair_20260910\teacher_16w_confirmed_schema9.npz
```

Builder mặc định tạo schema 9. Nó giữ toàn bộ dòng hợp lệ theo thứ
tự thời gian, không bỏ HOLD trước khi dựng chuỗi recurrent, dời pivot tới
observation xác nhận và tính lại net PnL theo Bid/Ask tại thời điểm có thể vào
lệnh. Dataset 16 tuần hiện có dùng pivot buffer 0,75 ATR; accepted BUY/SELL giữ
target SL cấu trúc tính từ pivot và TP tại
pivot đối diện chính xác. Metadata tương lai chỉ là target, không đi vào 137 input.
Giá TP chính xác nằm trong `opposite_exit_prices`; `teacher_tp_index` chỉ là bin
RR gần nhất của hợp đồng 20-output, nên phát lại theo head phân loại là xấp xỉ.

EA không chờ bằng `Sleep` trong `OnInit`. Với cửa sổ Tester dài, việc chờ đó có
thể làm thời gian mô phỏng chạy hết trước khi `OnTick` kịp thu bar. EA hiện khởi
tạo handle ngay, bỏ qua các bar đầu khi buffer MTF chưa sẵn sàng và chỉ ghi dữ
liệu sau khi `FeatureBuffersReady()` trả về true.

## Bằng chứng rebuild SL 1,50 ATR ngày 2026-09-01

Strategy Tester Model 1 chạy BTCUSDm M1 từ 2021-07-01 đến 2025-04-24 với đúng
`PivotStopBufferAtr=1.50`, test pass trong 9 phút 15 giây và không đặt lệnh thật:

- EA compile 0 error, 0 warning; final balance 10.000 USD;
- CSV schema 4 có 960.120 dòng: 101.928 BUY, 102.000 SELL, 756.192 HOLD và
  59.462 rejected pivot trước bước dời nhãn;
- NPZ schema 9 có 53.020 BUY, 52.895 SELL, 854.205 HOLD; 105.915 entry có
  geometry hợp lệ, không có dòng bị builder bỏ qua;
- sau khi vào tại bar xác nhận, khoảng cách SL thực tế có median 3,101 ATR và
  p90 4,995 ATR, vì SL vẫn nằm thêm 1,50 ATR ngoài cực trị pivot;
- TP liên tục dựng lại từ entry/risk/RR khớp `opposite_exit_prices` với sai số
  lớn nhất 0,000233 đơn vị giá do float32; BUY khớp đúng swing high, SELL dùng
  Ask tại swing low, tức low Bid cộng spread;
- 41,90% target TP có RR nhỏ hơn bin tối thiểu 0,25 và 9,95% target SL xa hơn
  bin tối đa 5 ATR. Vì vậy giá liên tục/pivot là ground truth; head geometry phân
  loại hiện hành chỉ xấp xỉ và UI action-only không huấn luyện head này.

Hash CSV và NPZ nằm trong manifest cạnh NPZ. Preflight trainer đọc lại cả hai
hash thành công. Full regression sau thay đổi: 1.156 test pass; UI self-test pass.

## Bằng chứng tái tạo Common Files ngày 2026-09-21

Batch config `mt5_v2_runner/bottom_top_historical_labeler_7d_batch.ini` chạy
Strategy Tester Model 1 trên BTCUSDm M1 từ 2025-01-01 đến 2025-01-08, test pass
với 40.318 ticks và 10.080 bars. Log EA ghi `rows=4893 buy=535 sell=551
hold=3807 rejected_pivots=241`. Vì `UseCommonFiles=true`, CSV được ghi vào
`C:\Users\Admin\AppData\Roaming\MetaQuotes\Terminal\Common\Files`, không phải
`mt5_v2_runner/Common/Files`.

Artifact 16 tuần `bottom_top_historical_opposite_pivot_v4_16w_sl075.csv` sau đó
được builder dựng thành
`outputs/bottom_top_teacher_confirmed_geometry_v9_sl075_20260921/teacher_16w_confirmed_schema9.npz`:

- 77.295 mẫu, 49.329 timestamp duy nhất; train/validation/test =
  54.076/11.631/11.588;
- 67.655 HOLD, 4.775 BUY, 4.865 SELL; skipped 0;
- `confirmation_audit.enabled=true`, 9.640 geometry target hợp lệ;
- CSV SHA-256 `f4722163821f3177f4ef1948bcd44c6ba094be0d037744d969e982c05c64149b`;
- NPZ SHA-256 `3716df72abc6a450dda1a4102ebb899420ae057791437f2ea2f35bf9c9e02474`;
- verify-only pass với checkpoint v26 và config hiện hành.

Đây mới là bằng chứng dữ liệu/provenance. Seed vẫn chưa có imitation sidecar,
chưa có Model 4/real-tick và chưa có frozen multi-window OOS, nên chưa được dùng
để promotion.

## Bằng chứng MT5 ngày 2026-08-26

Strategy Tester Visual, BTCUSDm M1, 2025-01-01 đến 2025-01-02, Model 1:

- 1.440 bar, 5.760 tick;
- 0 giao dịch thật, final balance 10.000 USD;
- 663 dòng raw: 71 BUY, 70 SELL, 522 HOLD;
- 185 pivot candidate: 147 đến được pivot đối diện, 38 chạm SL trước;
- 141 trade teacher được chấp nhận, mọi net PnL đều dương;
- net PnL theo đơn vị giá: min 3,55; median 105,98; max 585,92;
- mọi accepted row thỏa `gross PnL - spread = net PnL`;
- bằng chứng cũ ghi 663 dòng raw; khi dùng trainer hiện hành phải build lại
  dataset schema 5 và giữ toàn bộ dòng hợp lệ, không dùng NPZ đã cap HOLD;
- MetaEditor 0 error, 0 warning; 16 test liên quan pass.

Đây là bằng chứng hợp đồng tạo nhãn, chưa phải bằng chứng model đã học hoặc có
edge OOS. Behavior cloning phải chạy trên checkpoint candidate clone.

Lượt tuần dùng để train, BTCUSDm M1, 2025-01-01 đến 2025-01-08, Model 1:

- History Quality 100%, Initial Deposit 10.000 USD;
- Total Trades 0, Total Net Profit 0,00: EA không gửi lệnh thật;
- 4.711 dòng raw: 489 BUY, 488 SELL, 3.734 HOLD;
- 1.327 pivot candidate: 977 accepted, 324 SL-first, 2 không có pivot đối diện
  và 24 đến pivot đối diện nhưng net PnL sau spread không dương;
- mọi accepted row có net PnL dương, min 0,69 và max 1.688,90 theo đơn vị giá;
- NPZ schema 5: 4.711 x 137, giữ nguyên 3.734 HOLD và đủ hard-negative;
- split thời gian 3.296 train, 699 validation, 716 test; skipped 0.

## Huấn luyện candidate và UI

UI Java có thẻ **Imitation tối đa hiện có**, gọi
`scripts/train_teacher_imitation.py --action-only`. Dataset mặc định là schema 9:
`outputs/he_chuyen_gia/v193_repair_20260910/teacher_16w_confirmed_schema9.npz`.
Dataset có 77.295 row, 49.329 timestamp duy nhất, 4.775 BUY, 4.865 SELL và
67.655 HOLD trong khoảng 01/01/2025–22/04/2025.
Pipeline xác minh lại hash CSV nguồn, hash NPZ, 137 input, authority BUY/SELL,
split thời gian, thời điểm finalize, NaN/Inf và PnL dương sau spread trước khi
nạp checkpoint.

Khi workflow bắt đầu, UI tự chuyển vùng output sang **Nhật ký** và stream
stdout/stderr. Trainer phát `action_replay_progress` trong initial/final metric
replay, `action_batch_progress` trong từng epoch BPTT và `action_epoch` khi kết
thúc epoch; UI hiển thị phần trăm window, loss hiện thời, BA/precision/MCC và
thời gian thay vì bắt người dùng đọc JSON thô.

Toolbar có hai cách dừng tách biệt:

- **Lưu và dừng** chỉ bật cho action imitation đang train. UI tạo file yêu cầu
  riêng; Python hoàn tất optimizer batch hiện tại rồi ghi atomically
  `<candidate>.resume.pkl` và sidecar `.json`. Artifact pause chứa system đúng
  tại batch boundary, Adam moments, epoch, permutation/cursor, RNG Torch/Numpy,
  best-state, patience và history. Nó có `status=paused`, `promotable=false` và
  tuyệt đối không được coi là candidate hoàn tất.
- Ngoài pause thủ công, trainer ghi một resume checkpoint atomically sau mỗi
  validation epoch với `status=checkpoint`, `promotable=false`, contract/hash
  đầy đủ và cursor ở đầu epoch kế tiếp. Vì vậy mất terminal/UI sau một epoch
  không làm mất toàn bộ nhiều giờ BPTT; lần chạy lại sẽ tự nạp checkpoint này.
- Với CPU run kéo dài, `scripts/run_teacher_imitation_supervisor.ps1` bao quanh
  trainer bằng cơ chế một-worker: mỗi attempt có stdout/stderr riêng, chỉ retry
  sau khi worker đã terminal và luôn nạp cùng resume artifact. Supervisor không
  được dùng để chạy song song hay biến resume giữa chừng thành candidate.
- Resume artifact được ghi sau validation epoch với permutation rỗng và cursor
  bằng 0 theo chủ ý; đây là boundary hợp lệ ở đầu epoch kế tiếp, không phải
  artifact hỏng. Resume giữa batch mới bắt buộc phải có permutation/cursor khớp
  toàn bộ recurrent windows.
- Supervisor CPU đặt thêm checkpoint giữa epoch mỗi `128` recurrent windows;
  artifact giữ optimizer/RNG/permutation/cursor nên terminal mất giữa BPTT chỉ
  mất tối đa một đoạn ngắn, không mất cả epoch. CLI mặc định `0` để giữ hành vi
  cũ khi chạy trực tiếp; đây là checkpoint vận hành, không phải candidate.
- Action imitation hiện có **teacher-guided mode**. Teacher không bịa nhãn và
  không đọc test/OOS: nó dạy theo bài HOLD/ENTRY timing qua BPTT, rejected
  hard-negative, độ tin cậy theo `teacher_net_r`, ranking/cân bằng BUY-SELL và
  self-fed state/overfit. Sau mỗi validation epoch teacher chỉ được điều chỉnh
  có giới hạn penalty, hard-negative multiplier, confidence/ranking weights,
  role weights và learning-rate. Mỗi quyết định cùng tín hiệu và lesson state
  được ghi vào `history` và resume state, nên pause/resume tái tạo đúng teacher.
  Record validation hiện lưu riêng BA/precision/recall và eligibility của BUY/SELL;
  nếu một role có operating point không đạt, mentor tăng bounded role scale cho
  đúng role đó thay vì báo `hold_guidance` do chỉ nhìn metric aggregate.
  Có thể chạy baseline tĩnh bằng `--no-mentor-guided`; mặc định CLI mới bật.
- Candidate report có schema `12` để phân biệt teacher-guided artifact với
  sidecar imitation cũ chỉ có binary action loss.
- **Dừng workflow** màu đỏ vẫn hủy process cưỡng bức. Nó dành cho tình huống
  khẩn cấp và chỉ có thể làm mất phần batch chưa nằm trong resume artifact gần
  nhất.

Khi bấm chạy lại với cùng candidate output, CLI tự phát hiện resume artifact,
đối chiếu tuyệt đối source path/hash/bytes, dataset path/hash, config path/hash,
toàn bộ hyperparameter và output path rồi mới nạp. Frozen-head hash và hash
recurrent weights tại batch đã lưu cũng phải khớp. Resume không chạy lại initial
metric replay và tiếp tục đúng permutation/cursor. Khi candidate cuối cùng đã
ghi, reload và kiểm tra hash thành công, file resume, sidecar và stop-request cũ
được xóa. Đổi output hoặc đổi bất kỳ thành phần hợp đồng nào tạo run mới và
không được phép dùng nhầm resume cũ.

Tab `Train candidate` kiểm tra `<seed>.resume.pkl` trước khi tự khởi tạo seed
hoặc gọi MT5. Nếu imitation còn pause/checkpoint, UI chặn ngay và hướng người dùng quay
lại tab `Imitation tối đa hiện có` để resume đúng output; UI không tạo một
checkpoint rỗng trùng tên với candidate đang train dở.

### Tab Chẩn đoán học

Vùng output của UI có **Chẩn đoán học → Imitation**, cập nhật sau mỗi epoch action
imitation; MT5 online và Ablation nằm ở hai mục riêng nên không trộn metric.
Trainer vẫn gom confusion matrix trong các batch để theo dõi tức thời và ghi rõ
các số này là `online`. Sau mỗi epoch, trainer phát lại cả train và validation
bằng đúng cùng bộ trọng số cuối epoch. Bảng UI dùng BA phát lại của train để tính
BA gap, tránh so một số đo qua nhiều trạng thái trọng số với một snapshot cuối
epoch.

Tab hiển thị train loss/BA, validation loss/BA/precision/recall/MCC, BA gap và
ba chỉ báo:

- underfit chỉ được phân loại từ epoch 3: `high` khi train BA dưới 0,58 hoặc MCC
  dưới 0,10; `watch` khi train BA dưới 0,65 hoặc MCC dưới 0,20; còn lại `low`;
- overfit dùng train-replay/validation BA gap và xu hướng validation. `high` khi gap
  ít nhất 0,12 đồng thời validation BA đã rơi ít nhất 0,02 khỏi đỉnh; `watch`
  khi gap ít nhất 0,08 hoặc train loss còn giảm trong khi validation loss tăng,
  BA giảm và đã rơi ít nhất 0,02; còn lại `low`;
- tỷ lệ khuyến khích/phạt là proxy loss imitation:
  `mean P(ENTRY | target ENTRY) / (false_entry_penalty * mean P(ENTRY | target HOLD))`.
  UI hiện cả tử/mẫu; nếu penalty bằng 0 thì tỷ lệ là `N/A`.

Tỷ lệ này không phải tổng reward RL, profit factor hay reward/risk của giao dịch.
Các trạng thái underfit/overfit là cảnh báo heuristic để đọc xu hướng, không
thay validation gate, candidate reload, MT5 reconciliation hoặc frozen OOS.
Khi resume, trainer phát lại các record epoch đã lưu để tab khôi phục đầy đủ
lịch sử trước khi nhận epoch mới.

Schema 9 giữ sửa lỗi nhãn quan trọng từ schema 8. BUY/SELL không còn được gắn vào
chính bar pivot chỉ có thể biết bằng tương lai. Builder dời pivot sang observation
HOLD đầu tiên có sẵn từ bar xác nhận thứ ba đến tối đa năm phút chờ thêm, rồi
tính lại entry theo Bid/Ask của observation đó. Chỉ lệnh vẫn có net PnL dương
tới pivot đối diện mới giữ BUY/SELL; lệnh không còn dương trở thành HOLD
hard-negative. Lần rebuild SL 1,50 ATR ngày 2026-09-01 giữ 960.120 dòng: 105.915
BUY/SELL còn hợp lệ sau xác nhận (53.020 BUY, 52.895 SELL), 76.891 pivot từng
dương tại giá pivot nhưng không còn dương khi vào tại thời điểm khả thi, và
26.494 pivot không có observation xác nhận trong cửa sổ nên không được dùng làm
entry. Cả 105.915 entry đều có target geometry hợp lệ.

Tab imitation dùng seed policy architecture V3 tại
`outputs/he_chuyen_gia/kho_chuyen_gia_transformer_v3.pkl`. Khi field trống hoặc
còn trỏ tới default V1/V2 đã mất, UI đổi sang đường dẫn V3 và chạy
`khoi-tao-he-chuyen-gia` bằng cấu hình Five khóa cứng trước bước train. Seed được
tạo thành file riêng; checkpoint candidate và checkpoint Candidate MT5 không bị
ghi đè.

Observation 137 được nhóm theo `sample_time`, nhân thành packet năm agent, chạy
đúng cross-agent max-pooling và replay cửa sổ BPTT offline mặc định 64 bước.
UI cho chọn từ 2 đến 64; PPO online của model cũng dùng unroll 64 nhưng hai
đường vẫn có buffer và nút điều khiển riêng. Mức 64 bao phủ horizon pivot 48
nến và giữ thêm 16 bước ngữ cảnh. Trainer reset
prior imitation cũ về trung tính rồi chạy BPTT thật qua observation encoder,
toàn bộ LSTM, Transformer hậu-LSTM `512/8 heads/FFN 2048` và projection
`action_query`. Architecture v3 giữ nhánh attention 5 agent hiện tại, đồng thời
thêm bốn temporal token causal cho mỗi agent với horizon EMA 2/8/32/128 bước.
Mỗi bước chỉ ghi hidden hiện tại vào token sau khi tạo decision representation;
reset episode xóa token đúng agent. Nhờ vậy Transformer trực tiếp đọc history xa
thay vì phụ thuộc hoàn toàn vào hidden/cell của LSTM.

Trên representation 128 chiều, `heads.entry_role_calibrator` không còn share một
adapter/output. Hai nhánh BUY và SELL độc lập, mỗi nhánh dùng
`LayerNorm -> Linear -> SiLU -> Linear` và output riêng. LayerNorm theo role làm
giảm drift scale/offset feature; boundary 0 của từng nhánh được học bằng train
loss và chọn epoch bằng hành vi boundary 0 trên validation. Threshold sweep chỉ
là audit ROC/PR, không còn bị trừ vào logits hoặc lưu vào checkpoint. Projection
sao chép riêng trong calibrator cũ và `role_entry_bias` hậu nghiệm đã bị xóa;
train/runtime cùng dùng đúng `heads.action_query`. Representation
thay đổi sau mỗi optimizer step nên workflow action không còn tạo hoặc tái dùng
hidden/action-query cache đã detach.

Residual sau mapping vẫn được cộng giống nhau vào MARKET/LIMIT/STOP cùng phía; teacher chỉ học
thời điểm/hướng và không có nhãn execution style. Trọng số action embedding,
value, target và geometry head bị khóa. Tuy nhiên representation là shared nên
logits của agent 2/3/4 và các head này có thể đổi; report không còn tuyên bố
`protected logits drift = 0`.
Cross-entropy giữ đúng prevalence quan sát của train; không dùng inverse class
weight vì cách đó từng biến tập có khoảng 11% nhãn entry thành policy phát hơn
80% entry tại boundary 0. Quality weight vẫn spread-aware, phạt false-entry và
pairwise ranking loss giữa ENTRY/HOLD cung cấp tín hiệu xếp hạng cho lớp hiếm mà
không làm đổi class prior.
Binary imitation so HOLD với `max(MARKET,LIMIT,STOP)` cùng phía, đúng argmax của
runtime; cách `logsumexp` cũ đã bị xóa vì tự cộng offset `ln(3)` giả vào entry.
Sau BPTT có pha frozen-backbone refinement cho hai calibrator. Epoch được chọn
bằng MCC, precision, balanced accuracy, recall và predicted entry rate tại chính
boundary runtime. Sau mỗi epoch, trainer chọn operating point chỉ từ train với
ràng buộc precision, recall, balanced accuracy và trần entry rate rồi nhúng độ
dịch vào bias output của từng role. Trần train bằng đúng prevalence quan sát để
chừa biên 1,20x của gate validation cho temporal distribution shift. Nếu train
chưa có điểm đủ chuẩn, lựa chọn
vẫn phải nằm dưới trần entry rate. Validation và test không cung cấp ngưỡng; validation
threshold sweep không mutate model. Baseline epoch 0 cũng phải đi qua cùng phép
calibration trước khi được phép thắng lựa chọn epoch. Gate yêu cầu ít
nhất 50 mẫu mỗi role, recall mỗi role >= 20%, cả BUY/SELL vượt random và không
over-entry. Test chỉ được đọc sau khi mọi lựa chọn bằng validation hoàn tất.

Geometry `delay/sl/tp/trailing` vẫn không thuộc workflow action-only chính. Paired
MT5 ngày 2025-05-01 cho thấy full geometry và SL-only đều làm PnL/DD/policy LCB
xấu hơn checkpoint action-only. Schema 9 khôi phục target geometry hợp lệ để dữ
liệu không mất SL/TP, nhưng UI vẫn chỉ học HOLD/BUY/SELL và không tự chạy stage
geometry. Muốn huấn luyện geometry phải dùng luồng nghiên cứu tách riêng, sau đó
kiểm chứng MT5/OOS; sự tồn tại của target không phải bằng chứng policy có edge.

Để teacher thực sự học cả action và execution geometry, dùng
`scripts/run_teacher_full_research_supervisor.ps1`. Supervisor này chờ action
candidate có sidecar schema hiện hành, reload, validation và continuous self-fed
đều đạt; sau đó mới chạy `--geometry-only` vào output khác. Geometry vẫn là
candidate nghiên cứu cho đến khi paired MT5/OOS chứng minh không làm xấu PnL,
drawdown và policy LCB; không được coi kết quả chain này là deployable tự động.

Sau khi chọn và hiệu chỉnh trọng số tốt nhất, schema 11 chạy thêm validation tự
vận hành liên tục. Mạng khởi tạo trạng thái đúng một lần ở đầu validation, tự đưa
hành động vừa chọn vào đầu vào `previous_action` của bước sau và không đặt lại
trạng thái theo cửa sổ 64. Nhãn giáo viên chỉ dùng để chấm HOLD/ENTRY. Tác nhân
BUY/SELL được chọn giữa HOLD và ba kiểu vào lệnh thuộc quyền; ba tác nhân quản lý
được giữ HOLD vì bộ dữ liệu lịch sử chưa cung cấp nhãn quyền theo vòng đời cho
chúng. Điểm lưu schema 11 chỉ được dùng làm nguồn MT5 khi cả cổng zero-boundary
và cổng tự vận hành này đều được thực thi và đạt. Phép kiểm định này phát hiện
sai lệch tích lũy do tự chọn hành động nhưng không thay thế kiểm chứng MT5/OOS.

Luồng chỉ học hình học cũng phải nhận đúng điểm lưu dạng 11 đã đạt hai phép kiểm
định trên. Báo cáo sau học hình học giữ nguyên bằng chứng hành động của điểm lưu
nguồn và vẫn xuất dạng 11; nó không được hạ về dạng báo cáo cũ rồi đi vòng qua
phép kiểm định tự vận hành. Báo cáo dạng 10 trở về trước chỉ còn dùng để truy vết
nghiên cứu.

Điều kiện lưu candidate:

- metric, gradient và calibrator hữu hạn;
- gradient khác 0 được quan sát ở encoder, LSTM, Transformer hậu-LSTM,
  action-query và calibrator;
- hash recurrent trainable phải đổi, còn hash trọng số các head độc lập bị khóa
  phải giữ nguyên;
- validation tăng balanced accuracy, có MCC/precision dương, không over-entry,
  cả BUY/SELL role đều vượt random;
- checkpoint gốc không đổi hash/kích thước;
- candidate nạp lại được và giữ đúng hash recurrent/frozen-head;
- báo cáo ghi `candidate_only`, `promotable=false`, `geometry_stage_executed=false`
  và `mt5_oos_proven=false` cho đến khi paired MT5/OOS thực sự qua.

Khi candidate action-only được dùng làm seed MT5, Five trainer giữ action prior
trong 32 optimizer update đầu rồi neo tới update 256; geometry không được giữ.
Workflow ghi `source_seed_imitation` và `seed_imitation_preflight` vào manifest.
Chỉ `effective_for_mt5=true`, `entry_scope=all_entry_mass_v2`, hash sidecar khớp
checkpoint và `geometry_effective_for_mt5=false` mới đúng contract hiện hành.

Để chạy thử một checkpoint giữa chừng, `Train candidate` có tùy chọn rõ ràng
`Bỏ qua gate imitation seed (research-only)`, tương ứng CLI
`--allow-unverified-imitation-seed`. Runner vẫn giữ nguyên kết quả preflight
thất bại, ghi `gate_bypassed=true`, `run_scope=research_only_unverified_imitation_seed`
và `promotable=false` vào manifest. Tùy chọn này chỉ mở quyền thực thi MT5 smoke;
nó không biến checkpoint giữa chừng thành candidate imitation hoàn tất và output
của run đó không tạo lineage đủ điều kiện promotion.

Phép A/B ngắn dùng
`configs/experiments/imitation_mt5_ab_v1.yaml` cho cả checkpoint gốc và
candidate, khóa cùng cửa sổ/model/seed. Profile chỉ tắt failure-replay hậu kỳ để
không trộn tác động của seed với một curriculum thứ hai; PPO và lifecycle train
chính vẫn chạy. Kết luận phải ưu tiên execution ledger, policy-origin
expectancy, reconciliation, DD/PF và số lệnh đã xác nhận; action được đề xuất
không được tính là lệnh đã khớp.

## Bằng chứng lịch sử trước full recurrent BPTT

Các artifact và số liệu bên dưới dùng trainer frozen-representation v2 cũ. Chúng
được giữ để truy vết thí nghiệm, không chứng minh implementation full recurrent
v3 hiện tại có edge. Candidate mới phải chạy lại paired MT5 và frozen OOS.

### A/B MT5 entry-mass v2 ngày 2026-08-28

Checkpoint `teacher_imitation_entry_mass_v2_final_20260828_154334.pkl` được
train trên 960.120 mẫu, best epoch 46/58. Test split 143.989 mẫu tăng balanced
accuracy từ 0,5000 lên 0,7160, MCC từ 0 lên 0,3774 và entry precision từ 0,2187
lên 0,4369. Hash checkpoint/sidecar khớp, source/frozen tensor không đổi và
reload pass; preflight trả `effective_for_mt5=true`.

A/B Strategy Tester khóa `2025.01.01–2025.01.02`, model 1, seed 0 và cùng
profile `imitation_mt5_ab_v1`:

| Chỉ số | Seed gốc | Imitation v2 |
|---|---:|---:|
| Net profit | -693,92 | -52,63 |
| Profit factor | 0,49 | 0,91 |
| Win rate | 29,05% | 42,22% |
| Equity DD | 6,98% | 2,22% |
| Report trades | 148 | 45 |
| Ledger lifecycle | 145 | 42 |
| Policy-origin LCB | -0,394585 | -0,342016 |
| Negative context-action scopes | 6 | 3 |

Hai lượt đều `reconciled=true`, execution failure bằng 0, source seed/config
không đổi. Trace của baseline ghi imitation prior/representation đều tắt; trace
candidate ghi cả hai bật. Candidate hoàn tất đúng 32 optimizer update đầu với
calibrator hash không đổi; sau reload, warm-up kết thúc nhưng representation
vẫn bị khóa và anchor còn hoạt động đến update 256. Như vậy imitation đã đi qua
đúng MT5 train path và giảm 92,42% mức lỗ cùng 68,19% equity DD. Tuy vậy PF và
policy-origin LCB vẫn chưa dương, nên checkpoint vẫn là candidate, không được
promote thành policy có edge.

### Candidate recurrent v4 ngày 2026-08-27

Candidate `candidate_recurrent_v4.pkl` dùng 191 cửa sổ x 16 bước trên 3.019
sample time. Adapter nhận hidden 4096 chiều qua bottleneck 128; checkpoint nguồn và mọi tensor
frozen giữ nguyên hash, protected logits drift đúng 0 và candidate reload pass.

So với base trên test thời gian 716 mẫu:

- accuracy: 49,58% -> 67,88%;
- balanced accuracy: 49,02% -> 59,96%;
- MCC: -0,015 -> 0,167;
- entry precision: 17,98% -> 28,25%;
- predicted entry rate: 49,72% -> 31,15%;
- HOLD recall: 49,91% -> 72,56%; ENTRY recall: 48,12% -> 47,37%.

BUY test còn thiên ENTRY (53,37% predicted entry), trong khi SELL thận trọng
(9,17%). Vì chỉ có một tuần dữ liệu và test role còn drift so với validation,
artifact vẫn là candidate-only. Các số này chứng minh trainer sửa đúng contract
và giảm over-entry; chưa chứng minh lợi nhuận hay edge MT5 OOS.

### Dataset và candidate 16 tuần SL 0,25 ngày 2026-08-27

Profile `bottom_top_historical_labeler_16w.ini` chạy Strategy Tester thật trên
BTCUSDm M1, Model 1, từ 2025-01-01 đến 2025-04-23:

- 161.233 bar và 644.925 tick mô phỏng, test hoàn tất trong 47,9 giây;
- 0 giao dịch thật, final balance giữ nguyên 10.000 USD;
- 77.295 dòng: 7.976 BUY, 7.994 SELL và 61.325 HOLD;
- 5.393 pivot bị từ chối thành hard-negative HOLD;
- NPZ schema 5 có 49.329 sample time, split thời gian 54.076 train,
  11.631 validation và 11.588 test; skipped 0.

Candidate `candidate_recurrent_16w.pkl` train từ checkpoint gốc chỉ đọc với
chuỗi 16 bước. Early stop chọn epoch 29 sau 41 epoch. Trên test thời gian:

- accuracy: 50,16% -> 73,25%;
- balanced accuracy: 50,04% -> 65,59%;
- MCC: 0,001 -> 0,281;
- entry precision: 20,63% -> 38,95%;
- predicted entry rate: 49,78% -> 27,80%;
- BUY balanced accuracy 66,47%, SELL balanced accuracy 64,71%.

Checkpoint nguồn giữ nguyên hash; frozen network, protected logits và ba role
không-entry giữ nguyên; candidate reload pass. Artifact vẫn
`candidate_only`, `promotable=false`, `mt5_oos_proven=false`.

### Dataset và candidate 16 tuần SL 0,75 ngày 2026-08-27

Sau khi tăng khoảng đệm SL gấp ba, profile mới dùng
`PivotStopBufferAtr=0.75` và ghi sang artifact riêng, không ghi đè dataset SL
0,25. Strategy Tester thật trên cùng BTCUSDm M1, Model 1, từ 2025-01-01 đến
2025-04-23 cho kết quả:

- 77.295 dòng: 8.527 BUY, 8.570 SELL và 60.198 HOLD;
- 4.266 pivot bị SL loại, giảm 1.127 so với SL 0,25;
- 0 giao dịch thật, final balance giữ nguyên 10.000 USD;
- NPZ schema 5 split 54.076 train, 11.631 validation, 11.588 test;
- skipped 0.

Candidate `candidate_recurrent_16w_sl075.pkl` trên test thời gian:

- accuracy: 50,04% -> 73,91%;
- balanced accuracy: 49,88% -> 68,40%;
- MCC: -0,002 -> 0,333;
- entry precision: 21,97% -> 43,24%;
- entry recall: 49,59% -> 58,55%;
- predicted entry rate: 49,78% -> 29,86%;
- BUY balanced accuracy 68,52%, SELL balanced accuracy 68,29%.

Checkpoint nguồn, frozen network, protected logits và ba role không-entry giữ
nguyên; candidate reload pass. Artifact vẫn candidate-only và chưa phải bằng
chứng lợi nhuận MT5 OOS.
## Đo tác dụng trên MT5

Phép A/B nhân quả dùng `configs/experiments/imitation_mt5_frozen_ab_v2.yaml`.
Profile này đặt `mt5_learning_enabled: false`, nên policy không PPO/update/replay
trong lúc Strategy Tester chạy. Cờ `five_frozen_behavior_sampling_enabled` vẫn
lấy mẫu từ phân bố hành vi bằng seed cố định; nhờ vậy phép thử có entry để đo
SL/TP mà không làm thay đổi trọng số. Baseline và candidate phải giữ nguyên cửa
sổ, tester model và seed. Manifest ghi `effective_mt5_learning_enabled` cùng
toàn bộ override để không nhầm phép thử frozen với một lượt train online.
Sampling dùng generator riêng `five_frozen_behavior_sampling_seed`, không dùng
RNG PyTorch toàn cục; các bước load/khởi tạo khác không được phép làm lệch chuỗi
action giữa baseline và candidate.
Profile chỉ bỏ hai tensor cell 4096 chiều khỏi observatory; action, request,
receipt, lifecycle và các chỉ số SL/TP vẫn được giữ nguyên. Việc này tránh vòng
compact JSONL làm chậm Strategy Tester nhưng không làm mất execution ledger.
Riêng profile causal đặt `execution_trace_observatory_max_bytes: 0` để không
rewrite tail sau mỗi event; quota thư mục của runner vẫn là giới hạn dung lượng.

Kết quả paired MT5 đã xác nhận full geometry và ablation SL-only đều thua
action-only trên cùng ngày/model/seed. Vì vậy UI không còn gọi stage geometry;
`scripts/compose_teacher_geometry_ablation.py` chỉ là công cụ tái lập thí nghiệm
cũ và artifact của nó không được đưa vào workflow chính.
