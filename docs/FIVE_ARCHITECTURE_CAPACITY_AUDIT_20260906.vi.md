# Kiểm tra sức chứa và action head của Five v2 (báo cáo lịch sử)

Ngày đo: 2026-09-06 đến 2026-09-07. Script tái lập:
`scripts/audit_five_architecture.py` và `scripts/audit_stage13_followup.py`.
Artifact: `outputs/he_chuyen_gia/architecture_audit_20260906/`.

## Trạng thái code hiện tại sau audit (Five v3, 2026-09-08)

Các số đo bên dưới thuộc kiến trúc v2 tại ngày 2026-09-06/07; không được coi
là phép đo lại cho v3. Kết quả audit đã dẫn tới các thay đổi candidate sau:

- Transformer giữ nhánh cross-agent 5 token cũ và thêm nhánh temporal causal.
  Mỗi agent có bốn token EMA đa thang với horizon 2/8/32/128; token hiện tại
  chỉ đọc lịch sử đã có trước quyết định rồi mới cập nhật memory.
- Token có embedding horizon và agent riêng. State temporal được reset riêng
  từng agent tại episode boundary và được lưu trong rollout/failure replay.
- Block Transformer production tăng từ 7.351.297 lên 12.610.050 tham số. Chưa
  có profiler/PCA/attention/probe mới, nên các bảng cũ không chứng minh v3 tốt
  hơn.
- `EntryRoleCalibrator` dùng hai nhánh nhỏ độc lập BUY/SELL. Runtime không còn
  chép một absolute threshold từ validation và không còn `role_entry_bias` hậu
  nghiệm; mỗi nhánh học zero-boundary trên representation đã normalize.
- Candidate schema 10 chỉ được dùng làm MT5 seed khi full-data validation gate
  thực sự được enforce/pass và checkpoint reload đúng hash. Resume đang pause
  hoặc candidate imitation chưa đạt gate bị runner từ chối.

Smoke production 4096 ngày 2026-09-08 đã load seed v2 thành architecture v3,
forward 5 agent và giữ nguyên SHA-256 nguồn. Với scale migrate bằng 0,
`temporal_scale` nhận gradient `-0,26210`; sau khi mở scale `0,01` chỉ trong RAM,
temporal encoder/projection có gradient L1 lần lượt `35,9314` và `107,9861`.
Điều này chứng minh đường gradient hoạt động, không chứng minh accuracy hoặc PnL.

## Phạm vi và tính bất biến

- Bước 1-2 dùng seed v2 mà tab imitation đang chọn:
  `outputs/he_chuyen_gia/kho_chuyen_gia_transformer_v2.pkl`, SHA-256
  `d39c4cbf03ed40efde899ed90e0cb4c35c5575d6a8ad843a526e125bb7ed50ab`.
- Bước 3-5 dùng snapshot mới nhất sau imitation:
  `teacher_imitation_candidate_20260902_095951.pkl`, SHA-256
  `98d9ce53e8bf63edd08db65c56344e163a0813e2aebb6534ec512a1c65e3fea5`.
  Đây là snapshot của run đang pause ở epoch 10, hoàn tất 9 epoch; không phải
  checkpoint promotable.
- Dữ liệu activation/probe chỉ lấy các cửa sổ chronological, không trộn tương lai
  vào train. Linear probe chọn C bằng validation rồi mới đọc test.

## 1. Torchinfo

Tổng policy có 93.182.878 tham số trainable:

| Module | Tham số | Tỷ lệ |
|---|---:|---:|
| Observation encoder | 795.026 | 0,85% |
| LSTMCell 1024 -> 4096 | 83.918.848 | 90,06% |
| Transformer hậu-LSTM 4096 -> 512 -> 4096 | 7.351.297 | 7,89% |
| Toàn bộ action/value/parameter heads | 1.117.707 | 1,20% |

## 2. PyTorch Profiler

CPU, 4 thread, 2 warm-up và 5 lượt đo độc lập, mỗi lượt xử lý packet 5 agent
với input 137. Tổng forward trung bình là khoảng 176,89 ms. Timing module là
inclusive wall-clock:

| Module | ms/call |
|---|---:|
| LSTM | 101,49 |
| Transformer hậu-LSTM | 12,58 |
| Observation encoder | 11,78 |
| Self-attention bên trong Transformer | 2,40 |
| Action query | 0,82 |
| Value head | 0,17 |

Profiler ghi khoảng 931,82 triệu FLOP mỗi forward; `aten::addmm` chiếm khoảng
922,25 triệu FLOP mỗi forward. LSTM là nút thắt chính về cả tham số và thời gian.

## 3. PCA/SVD và effective rank

Đo 1.600 hidden owner trên 16 cửa sổ test, mỗi cửa sổ dài 64 bước, trải từ
2024-09-27 đến 2025-04-23. Randomized SVD lấy 512 thành phần và đã giữ hơn
99,9999% tổng phương sai.

| Representation | rank 90% | rank 95% | rank 99% | entropy effective rank |
|---|---:|---:|---:|---:|
| LSTM 4096 trước Transformer | 2 | 4 | 11 | 2,116 |
| Sau Transformer | 3 | 5 | 13 | 2,404 |

Cả 4.096 chiều đều có standard deviation lớn hơn 1e-3, nhưng chúng biến thiên
gần như đồng tuyến. Vì vậy không được diễn giải “mọi neuron đều hoạt động” thành
“model dùng 4.096 chiều thông tin độc lập”. Trên mẫu OOS này, hidden space đã
sụp xuống vài trục chính.

## 4. Attention map

Block v2 tại thời điểm đo không có token thời gian. Trục token là đúng 5 agent tại cùng
timestamp; history 64 bước chỉ đi qua hidden/cell của LSTM. Vì vậy Transformer
không thể trực tiếp attend tới history xa theo thiết kế hiện tại.

Trên 1.024 attention sample x 8 head:

- attention diagonal trung bình: 0,2057, gần uniform 0,2000;
- normalized attention entropy: 0,9902, rất gần uniform 1,0;
- residual scale: 0,1;
- norm correction hậu-Transformer bằng trung bình 72,23% norm hidden gốc.

Transformer đang sửa representation khá mạnh nhưng attention giữa agent còn gần
uniform; map không chứa bất kỳ trục lag thời gian nào.

## 5. Linear probe

Probe nhị phân `teacher entry` so với `HOLD`, class-weight balanced, C chọn trên
validation. Mỗi split có khoảng 1.600 owner sample và giữ thứ tự thời gian.

| Test metric | LSTM raw | Sau Transformer | Delta |
|---|---:|---:|---:|
| Balanced accuracy | 0,60415 | 0,60281 | -0,00135 |
| MCC | 0,14255 | 0,14167 | -0,00088 |
| Entry precision | 0,18786 | 0,18997 | +0,00211 |
| Entry recall | 0,61321 | 0,58962 | -0,02358 |
| F1 | 0,28761 | 0,28736 | -0,00025 |
| ROC-AUC | 0,64733 | 0,63991 | -0,00743 |

Không có bằng chứng Transformer tạo feature tổng quát tốt hơn cho target
imitation này. Nó hơi tăng precision nhưng giảm recall, balanced accuracy, MCC,
F1 và ROC-AUC trên test.

## 6. Ablation reward/generalization

Chưa thể kết luận. Runner `run_mt5_ablation_v193.py` hiện so sánh
`frozen_policy/full_online/reasoning_off` và tự khai báo
`architecture_ablation=false`; nó không phải Transformer ON/OFF.

Khi audit chạy, PID 3028 đang dùng chính snapshot trên cho một Strategy Tester
dài từ 2025-01-01 đến 2025-03-20. Run vẫn phát event và chưa sinh
`mt5_diagnostics.json`, nên không mở tester thứ hai để tránh xung đột terminal.

Ablation bắt buộc tiếp theo sau khi run hiện tại kết thúc:

1. Hai clone chỉ khác `post_lstm_transformer.residual_scale`: weight hiện tại và 0.
2. Khóa learning ở cả hai arm.
3. Cùng MT5 model, seed, spread, symbol và khoảng chronological OOS nằm sau cửa
   sổ imitation.
4. So sánh net profit, PF, expectancy, WR, max equity drawdown và số lệnh đóng.
5. Không promote hoặc xóa Transformer nếu chưa có paired OOS gate này.

## 7. Class-conditional geometry của hidden state

Đo trên 16 cửa sổ train và 16 cửa sổ test, mỗi cửa sổ dài 64 bước. Centroid
chỉ được fit bằng train; test chỉ dùng để đánh giá. Khoảng tin cậy là bootstrap
phân tầng 1.000 lượt và so sánh raw/post theo đúng cùng sample.

### Bẫy role confounding

Dataset hiện tại có support cố định: `HOLD` thuộc agent 0 và 1, `BUY` chỉ thuộc
agent 0, còn `SELL` chỉ thuộc agent 1. Vì vậy geometry gộp ba lớp bị nhiễu bởi
embedding role. Bằng chứng: nearest-centroid ba lớp đạt BA 0,66907 nhưng recall
là `HOLD=0,00720`, `BUY=1,0`, `SELL=1,0`; confusion matrix cho thấy 1.378/1.388
HOLD bị ép thành BUY hoặc SELL. Khoảng cách chuẩn hóa `BUY↔SELL` rất lớn
(5,158 raw; 4,606 post) chủ yếu là khoảng cách hai role, không phải khả năng
nhận diện đỉnh/đáy.

Vì thế kết luận chính dùng geometry **trong cùng role**:

| Test / train metric | LSTM raw | Sau Transformer | Delta post-raw |
|---|---:|---:|---:|
| Agent 0, train separation HOLD↔BUY / pooled radius | 0,18693 | 0,19225 | +0,00533 |
| Agent 0, test separation HOLD↔BUY / pooled radius | 0,32912 | 0,31053 | -0,01859 |
| Agent 0, test nearest-centroid BA | 0,60813 | 0,62926 | +0,02113 |
| Agent 0, test MCC | 0,15708 | 0,18573 | +0,02865 |
| Agent 1, train separation HOLD↔SELL / pooled radius | 0,19443 | 0,19478 | +0,00035 |
| Agent 1, test separation HOLD↔SELL / pooled radius | 0,18652 | 0,17872 | -0,00781 |
| Agent 1, test nearest-centroid BA | 0,55266 | 0,56128 | +0,00862 |
| Agent 1, test MCC | 0,07427 | 0,08545 | +0,01118 |

Khoảng tin cậy ghép cặp 95% cho delta BA là `[+0,00343; +0,04028]` ở BUY-role
và `[+0,00144; +0,01580]` ở SELL-role. Transformer vì vậy có cải thiện nhỏ đối
với phân loại nearest-centroid trên mẫu OOS này. Tuy nhiên đây chưa phải một
hidden geometry tốt:

- separation entry↔HOLD chỉ khoảng 0,18-0,33 lần bán kính nội-lớp, nghĩa là các
  cụm chồng lấn mạnh;
- test centroid separation đều giảm sau Transformer;
- normalized centroid drift train→test tăng ở cả bốn lớp-role;
- pooled macro Fisher ratio giảm từ 1,33224 xuống 1,22702; số gộp này vẫn bị
  role confounding nên chỉ là chẩn đoán phụ.

Kết luận: imitation hiện đã tạo một chút tín hiệu class-conditional có thể đọc
được, rõ hơn ở BUY so với SELL, nhưng representation chưa gom BUY/SELL thành các
cụm ổn định tách khỏi HOLD. Transformer làm nearest-centroid tốt hơn một ít,
song không làm geometry OOS sạch hơn. Đây phù hợp với linear probe bước 5: chưa
có bằng chứng cho một bước nhảy feature quality đủ lớn để giải thích hiệu quả
imitation cao.

Artifact chi tiết: `07_class_conditional_geometry.json`; trực quan PCA fit trên
train rồi chiếu test: `07_class_conditional_geometry.png`.

## 8. Probe suite ưu tiên 1

Suite dùng 16 cửa sổ cho từng split train/validation/test, dài 64 bước. PCA và
standardization chỉ fit trên train; số PC và hyperparameter chọn bằng validation;
test chỉ dùng để báo OOS. Mọi target đều giữ cố định execution role.

### PCA → probe curve

Top PC đầu tiên giữ 82,98% variance raw và 79,51% variance post nhưng linear
probe chỉ bằng chance. Tới 8 PC đã giữ 98,33%/97,75% variance mà BA vẫn chỉ
khoảng 0,55-0,61. Cả bốn role/representation chỉ đạt ngưỡng 95% validation gain
ở **64 PC**, nơi đã giữ 99,987% raw và 99,980% post variance.

| Role / representation | PC chọn bởi validation | Validation BA | Test BA |
|---|---:|---:|---:|
| BUY / LSTM raw | 128 | 0,6754 | 0,6698 |
| BUY / post-Transformer | 64 | 0,6712 | 0,6837 |
| SELL / LSTM raw | 128 | 0,6515 | 0,6550 |
| SELL / post-Transformer | 128 | 0,6479 | 0,6349 |

Kết luận: phần lớn variance hidden không phải tín hiệu quyết định. Tín hiệu
entry nằm trong nhiều hướng variance thấp và chỉ bão hòa quanh 64 PC. Con số
effective rank 2-3 ở bước 3 vì vậy không đồng nghĩa chỉ 2-3 chiều là đủ để ra
quyết định.

### LDA / Fisher

Shrinkage LDA trên 128 PC cho train projected Fisher ratio từ 1,681 đến 1,861,
nhưng test BA chỉ 0,655-0,679. Đây là separability mức vừa, không phải BUY/SELL
tách sạch. Post-Transformer không cải thiện nhất quán: BUY test BA tăng nhẹ từ
0,6642 lên 0,6679, còn SELL giảm từ 0,6792 xuống 0,6555.

### MLP và kNN probe

MLP `128→32→2` dùng class-balanced weights và early stopping theo validation
chronological. kNN chọn k bằng validation.

- BUY: MLP thua linear có ý nghĩa ở cả raw và post; bootstrap delta BA 95% lần
  lượt `[-0,1190; -0,0277]` và `[-0,1179; -0,0207]`.
- SELL raw: MLP hơn theo point estimate nhưng CI `[-0,0071; +0,0780]` chứa 0.
- SELL post: MLP hơn trên test với CI `[+0,0062; +0,0783]`, nhưng validation MLP
  0,6283 vẫn thấp hơn linear 0,6479; đây là tín hiệu khám phá, chưa phải cải
  thiện tổng quát đã xác nhận.
- kNN thua linear ở cả bốn phép đo.

Vì probe phi tuyến không thắng nhất quán qua validation và test, chưa có bằng
chứng ổn định rằng hidden chứa decision boundary phi tuyến mà linear probe bỏ
sót. Artifact: `08_priority_one_probe_suite.json`; đồ thị:
`08_pca_probe_curves.png`.

## 9. Intervention / ablation PnL

Đã thêm runner `run_mt5_transformer_ablation.py`. Preflight trên snapshot thật
xác nhận hai seed reload được và toàn bộ network state chỉ khác đúng tensor
`post_lstm_transformer.residual_scale`: arm ON = 0,1; arm identity = 0,0. Source
checkpoint giữ nguyên hash. Cả hai arm dùng cùng config/profile frozen, Model 1,
seed và ngày test.

Chưa có kết quả paired PnL. Lượt một tháng 2025-05-01→2025-06-01 được dừng theo
yêu cầu rút ngắn. Lượt smoke bảy ngày 2025-05-01→2025-05-08 cũng được dừng ở
arm ON sau 600 bridge packet: runner vẫn chậm và diagnostics tại khoảng 598
competitive batch còn `market_time=""`, `closed_trades=0`, `net_pnl=0`. Arm
identity chưa chạy nên tuyệt đối không được suy ra Transformer tốt/xấu từ lượt
này. Hai summary được đánh dấu `interrupted=true`, `complete=false`.

Runner cũ có thêm một lỗi fail-fast: `huan_luyen_he_chuyen_gia_mt5` chủ động lưu
checkpoint `incomplete` rồi trả về bình thường khi chạm `max_wall_sec`; tiến
trình con vì thế có exit code 0 và outer runner từng gắn nhầm `complete=true`.
Runner mới chỉ công nhận một arm khi đồng thời có exit code 0,
`mt5.interrupted=false`, metrics không có `incomplete=true`, và policy hoàn
chỉnh. Summary một ngày bị gắn nhầm trước đây đã được hạ về `complete=false`.

EA v8.13 thêm `MaxDecisionPackets`. Giá trị dương làm chính EA gọi
`TesterStop()` sau khi hoàn thành đủ packet, nhờ đó Strategy Tester chốt report,
đóng vị thế và reconciliation bình thường; `max_wall_sec` chỉ còn là cầu chì.
Compile: 0 error, 0 warning.

Paired smoke hoàn chỉnh dùng 2025-05-01, Model 1, seed 20260702, policy frozen,
128 packet mỗi arm. Hai checkpoint chỉ khác
`post_lstm_transformer.residual_scale` (ON=0,1; identity=0), source hash giữ
nguyên.

| Metric MT5 | Transformer ON | Identity/OFF | ON - OFF |
|---|---:|---:|---:|
| Net profit | -324,83 | -331,95 | +7,12 |
| Profit factor | 0,24 | 0,22 | +0,02 |
| Win rate | 29,11% | 34,62% | -5,51 điểm % |
| Equity DD | 3,34% | 3,43% | -0,09 điểm % |
| Total trades | 79 | 78 | +1 |
| Ledger expectancy R | -0,27772 | -0,24897 | -0,02876 |

Cả hai report đều reconciliation pass, 128 feature rows có numerical parity,
và source checkpoint bất biến. Transformer có ảnh hưởng đến hành vi nhưng chưa
có bằng chứng cải thiện: lợi nhuận lỗ ít hơn một chút, song win rate và
expectancy R xấu hơn; cả hai đều `fitness_eligible=false`. Đây chỉ là smoke
Model 1 ngắn, không đủ để promote hoặc xóa Transformer. Artifact:
`11_mt5_transformer_pair_128/transformer_ablation_summary.json` và report đã
archive trong runtime của từng arm.

## 10. Temporal-lag probe cho LSTM

Probe dùng raw hidden trước Transformer, 16 cửa sổ mỗi chronological split, dài
64 bước. PCA 64 chiều và standardization chỉ fit trên train; regularization C
chọn bằng validation; test chỉ báo OOS. Target là action của đúng execution role
tại `t-lag`. Lag 1 có đường previous-action tường minh nên chỉ lag từ 2 trở lên
mới là bằng chứng recurrent memory.

| Lag | BUY test BA | SELL test BA |
|---:|---:|---:|
| 0 | 0,6736 | 0,6842 |
| 1 | 1,0000 | 1,0000 |
| 2 | 0,9912 | 0,9788 |
| 4 | 0,8595 | 0,8422 |
| 8 | 0,6069 | 0,6078 |
| 16 | 0,5322 | 0,4809 |
| 32 | 0,5083 | 0,4995 |

Kết luận: hidden giữ action-history hữu ích rõ tới khoảng 8 bước M1. Lag 16
chỉ còn biên yếu/không nhất quán và lag 32 bằng chance trên OOS; validation hơi
trên 0,52 ở lag 32 không generalize. Artifact: `09_temporal_lag_probe.json` và
`09_temporal_lag_probe.png`.

## 11. Ablation LSTM 4096 / 2048 / 1024

Đây là structured-transplant ablation: chọn neuron theo variance raw hidden chỉ
trên train, cắt đồng bộ bốn gate LSTM, recurrent columns, Transformer projection
và mọi downstream head. Vì vậy 2048/1024 là module LSTM nhỏ thật, không phải
mask giả; tuy nhiên chúng chưa được imitation retrain. Phép đo trả lời tín hiệu
đã học sống sót sau nén bao nhiêu, chưa trả lời chất lượng tối đa nếu train lại.

| Width | Parameters | Tỷ lệ | Forward CPU | Variance giữ lại | BUY probe BA | SELL probe BA |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 93.182.878 | 100% | 98,0 ms | 100% | 0,6343 | 0,6409 |
| 2048 | 31.808.414 | 34,14% | 41,5 ms | 86,14% | 0,6349 | 0,6192 |
| 1024 | 13.704.094 | 14,71% | 21,4 ms | 64,41% | 0,6425 | 0,5806 |

Action head trực tiếp của cả ba arm dự đoán entry rate 0% trên tập OOS (BA=0,5),
trong khi linear probe vẫn đọc được tín hiệu. Đây là bằng chứng vấn đề hiện tại
nằm ở việc chuyển representation thành action/threshold của imitation, không
phải hidden hoàn toàn không chứa tín hiệu. 2048 giữ gần nguyên BUY và giảm SELL
nhẹ; 1024 mất SELL rõ hơn. Chưa thay width production: cần imitation-train lại
4096/2048/1024 theo cùng budget rồi paired MT5/OOS trước khi chọn 2048.
Artifact: `10_lstm_width_ablation.json`.

## 12. Hook logits và threshold sweep

Stage 11 dùng temporary PyTorch forward hooks tại `action_query`, output của
entry calibrator và một identity tap nhận tensor logits cuối ngay trước bước
chọn action. Mỗi split đều xác nhận đúng một lần capture, tensor finite và tap
khớp chính xác tensor pre-selection. Threshold chỉ chọn trên validation, test
không tham gia tuning.

Với margin MARKET entry trừ HOLD, threshold mặc định 0 làm BUY và SELL entry
rate đều bằng 0%. Tuy nhiên đây không chỉ là sai offset:

| Role | Threshold validation | Validation BA | Test BA | Test ROC-AUC | Test entry rate |
|---|---:|---:|---:|---:|---:|
| BUY | -0,08585 | 0,5710 | 0,5058 | 0,4464 | 77,76% |
| SELL | -0,06276 | 0,5000 | 0,5035 | 0,4752 | 97,36% |

Trên test BUY, mean margin của nhãn HOLD là -0,08130 còn ENTRY là -0,08323;
thứ tự bị đảo. SELL gần như trùng nhau (-0,05793 và -0,05777). Dời threshold
chỉ đổi collapse từ toàn HOLD sang gần như toàn ENTRY, không tạo discrimination.
Temperature scaling cũng không thể sửa thứ tự ROC vì chỉ là biến đổi đơn điệu.
Artifact: `11_action_logit_diagnostics.json`; histogram:
`11_action_logit_margin_histograms.png`.

## 13. Head-only training và gradient telemetry

Hai can thiệp tăng dần đều chỉ dùng 16 cửa sổ chronological mỗi split và là
candidate diagnostic, không phải checkpoint production:

1. Chỉ train entry calibrator: early stop sau 13 epoch, best epoch 1; test dự
   đoán 100% ENTRY, BA=0,5. Calibrator một mình không sửa được action-query
   ranking. Artifact: `12_head_only_calibrator_experiment.json`.
2. Train `heads.action_query + heads.entry_role_calibrator`, freeze encoder,
   LSTM, Transformer, action embedding, value và geometry heads. Best epoch 30,
   early stop sau 42 epoch. Frozen-backbone hash và source checkpoint hash giữ
   nguyên.

| Role | Initial test BA | Final test BA | Final ROC-AUC | Final MCC | Final entry rate | Label rate |
|---|---:|---:|---:|---:|---:|---:|
| BUY | 0,5000 | 0,6181 | 0,6430 | 0,1646 | 54,04% | 14,04% |
| SELL | 0,5000 | 0,5689 | 0,5990 | 0,0913 | 53,58% | 12,45% |

Gradient không chết: action-query max norm 0,6541; calibrator-output 0,5520;
role-bias 0,1697; query-adapter từ 0 ở epoch đầu tăng lên khác 0 sau khi output
neutral bắt đầu học. TensorBoard lưu loss, BA, MCC, precision/recall/F1,
ROC/PR, entry rate và gradient norm từng nhóm tại
`13_action_head_only_tensorboard/`.

Kết luận: backbone chứa đủ tín hiệu để một action head được huấn luyện lại cải
thiện ranking OOS. Nhưng class-balanced CE hiện đẩy entry rate lên khoảng 54%,
cao gấp 4 lần label rate; candidate này chưa an toàn để ghép vào MT5. Bước tiếp
theo phải calibration/rate-constrained selection trên validation, ưu tiên
precision và giới hạn entry rate, rồi mới test OOS/MT5. Temperature scaling chỉ
hữu ích sau khi ranking đã được action-head training sửa.

## 14. Production-vs-Stage13 parity

State Stage13 chỉ được phép chứa 12 tensor thuộc `heads.action_query` và
`heads.entry_role_calibrator`; loader fail-fast nếu hash checkpoint gốc sai,
thiếu tensor, sai shape hoặc có key ngoài hai prefix này. Parity dùng 16 cửa sổ
validation x 64 bước, reset memory theo cửa sổ và teacher-force previous action.

Đường production `SharedFiveTradingPolicy.forward_five` và đường audit causal
replay khớp tuyệt đối:

- raw LSTM hidden: max drift 0;
- post-Transformer hidden: max drift 0;
- 12 action logits sau calibrator và trước chọn action: max drift 0;
- 5.120 deterministic action: 0 mismatch;
- BUY/SELL entry-mass margin: max drift 0.

Như vậy kết quả Stage13 không phải do audit gọi sai head hoặc bỏ qua đường
production. Artifact: `14_production_stage13_parity.json`.

## 15. Logit margin và threshold sweep của Stage13

Margin đúng objective là
`logsumexp(BUY actions 1,3,5) - HOLD` hoặc
`logsumexp(SELL actions 2,4,6) - HOLD`. Threshold chỉ chọn trên validation,
với trần entry rate bằng 1,10 lần label rate, precision tối thiểu 0,30 và BA
tối thiểu 0,52. Không role nào có threshold đáp ứng đồng thời cả ba điều kiện.

| Role | Validation AUC | Threshold | Validation BA | Precision | Entry rate | Test BA | Test entry rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BUY | 0,6547 | 0,12663 | 0,5581 | 0,2170 | 13,28% | 0,5428 | 22,73% |
| SELL | 0,6020 | 0,21835 | 0,5056 | 1,0000 | 0,12% | 0,5072 | 0,75% |

Threshold sửa được hiện tượng bắn khoảng 54% entry nhưng không tạo được
operating point cân bằng: BUY precision còn thấp, SELL gần như tắt. Artifact:
`15_stage13_logit_margin_threshold_sweep.json`; histogram:
`15_stage13_entry_mass_margin_histograms.png`.

## 16. Class-weight ablation

Bốn arm khởi tạo từ cùng action head trung tính, cùng dữ liệu 16 cửa sổ mỗi
split, seed 0, learning rate 0,001, budget tối đa 40 epoch và patience 8.
Encoder, LSTM, Transformer, action embedding, value và geometry head đều khóa.
Winner chỉ dựa trên validation: số role qua gate, worst-role ROC-AUC/AP rồi mới
tới MCC/BA/precision. Test chỉ được đọc sau khi cố định checkpoint tốt nhất.

| Weight | Best epoch | BUY val AUC/AP | SELL val AUC/AP | BUY val BA | SELL val BA |
|---|---:|---:|---:|---:|---:|
| Unweighted | 40 | 0,6823 / 0,2184 | 0,5588 / 0,1273 | 0,5205 | 0,5049 |
| Sqrt inverse | 13 | 0,6722 / 0,2147 | 0,5524 / 0,1245 | 0,5205 | 0,5049 |
| Capped inverse 3 | 12 | 0,6795 / 0,2184 | 0,5530 / 0,1249 | 0,5205 | 0,5049 |
| Balanced inverse | 21 | 0,6818 / 0,2271 | 0,5822 / 0,1442 | 0,5294 | 0,5049 |

`balanced_inverse` thắng vì cải thiện role yếu SELL trên validation. Tuy nhiên
chỉ BUY có operating point qua gate; SELL recall vẫn 1,11%. Đây là winner để
retrain lần 2, không phải bằng chứng promote. Artifact:
`16_class_weight_ablation.json`.

## 17. Head-only retrain lần 2

Retrain dùng winner `balanced_inverse`, tăng mẫu lên 32 cửa sổ mỗi split, dài
64 bước. Best epoch 67, early stop sau 79 epoch. Hai threshold validation được
ghi trực tiếp vào role bias: BUY 0,45997 và SELL 0,33857. Source checkpoint và
backbone bất biến; state reload và full candidate reload đều khớp chính xác.

| Split/role | BA | MCC | Precision | Recall | Entry rate | Label rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation BUY | 0,5025 | 0,0660 | 1,0000 | 0,0050 | 0,06% | 12,44% | 0,6202 |
| Validation SELL | 0,5666 | 0,1282 | 0,2342 | 0,2562 | 14,01% | 12,81% | 0,6472 |
| Test BUY | 0,5068 | 0,0443 | 0,2778 | 0,0230 | 1,13% | 13,57% | 0,6475 |
| Test SELL | 0,5625 | 0,1045 | 0,2024 | 0,3178 | 20,93% | 13,33% | 0,6499 |

Validation gate thất bại: BUY gần như tắt, SELL precision dưới 0,30. Candidate
được giữ riêng tại `17_head_only_retrain_v2_candidate.pkl`, tuyệt đối không ghi
đè source và `promotable=false`. Chưa được đưa vào workflow/MT5; muốn promote
vẫn cần sửa calibration theo role và chạy paired MT5/OOS. Artifact đầy đủ:
`17_head_only_retrain_v2.json`.

## 18. Gate nhân quả của temporal token v3

Audit can thiệp chạy trực tiếp qua `SharedFiveTradingPolicy.forward_five` trên
checkpoint v2 đã migrate deterministic sang kiến trúc v3. Chỉ observation ở
các bước tương lai, bắt đầu từ bước 6, bị thay đổi; sáu quyết định ở prefix và
raw LSTM hidden tương ứng phải bất biến với tolerance `1e-7`.

- max delta xác suất action trong prefix: `0`;
- max delta raw LSTM hidden trong prefix: `0`;
- sau khi reset riêng agent 0, delta hidden/cell/temporal token do lịch sử cũ
  của agent đó đều bằng `0`;
- SHA-256 checkpoint nguồn trước/sau giữ nguyên:
  `d39c4cbf03ed40efde899ed90e0cb4c35c5575d6a8ad843a526e125bb7ed50ab`.

Gate này đạt. Temporal memory được đọc trước rồi mới cập nhật sau quyết định
hiện tại; can thiệp tương lai không thể đổi output quá khứ. Artifact:
`14_temporal_v3_offline_gates.json`.

## 19. Gate lưu giữ thông tin ở lag 32 và 128

Phép đo dùng paired impulse-minus-zero trên đúng `input_projection` production.
Trừ control zero loại bias projection; ratio còn lại được so với decay EMA lý
thuyết và cosine với impulse gốc.

| Lag / token | Retention đo được | Retention lý thuyết | Cosine | Max lỗi ratio |
|---|---:|---:|---:|---:|
| 32 / horizon 32 | 0,36205536 | 0,36205530 | 1,00000012 | 5,96e-8 |
| 128 / horizon 128 | 0,36643773 | 0,36643770 | 1,00000024 | 3,59e-7 |

Delta sang agent khác bằng `0`. Gate yêu cầu retention tối thiểu 0,30, cosine
tối thiểu 0,999 và lỗi decay tối đa `2e-5`; kết quả đạt toàn bộ. Đây chứng minh
token còn giữ thông tin có thể truy hồi, không tự nó chứng minh thông tin đó tạo
edge giao dịch.

## 20. Thử nghiệm ban đầu: candidate temporal-v3 và paired OOS MT5

Trainer kiểm soát freeze encoder, LSTM 4096, cross-agent Transformer, action
embedding và các head không liên quan; chỉ train nhánh `temporal_*` cùng hai
role calibrator BUY/SELL. Window dài 128, split chronological, epoch chỉ chọn
trên validation và test không tham gia selection. Checkpoint v2 migrate có
`temporal_scale=0`, nên trainer v3 khởi tạo gate ở 0,05 để temporal encoder nhận
gradient ngay từ optimizer step đầu. Source hash và frozen-surface hash bất biến.

Ba thử nghiệm validation đều thất bại ở operating point runtime:

- penalty false-entry 0,50 làm BUY/SELL cùng collapse về 100% HOLD;
- bỏ penalty làm BUY 100% ENTRY và SELL 100% HOLD;
- scale 0,05 giúp toàn temporal branch nhận gradient nhưng best validation vẫn
  BA=0,5 và MCC=0 cho từng role.

Vì vậy candidate là `evaluation_candidate_only`, `promotable=false`. Dù gate
validation không đạt, một paired intervention OOS đã được chạy đúng một lần để
trả lời trực tiếp liệu temporal branch hiện tại có tạo uplift hay không. Hai arm
dùng MT5 Model 1, cùng ngày 2025.05.01--2025.05.02, seed 20260702, 96 decision
packets và learning frozen. Reload audit xác nhận tensor duy nhất khác nhau là
`post_lstm_transformer.temporal_scale` (`0,0498` so với `0`).

| Arm | Net profit | PF | Policy lifecycle | Mean R | LCB R | Equity DD |
|---|---:|---:|---:|---:|---:|---:|
| v3 temporal ON | -461,53 | 0,25 | 96 | -0,51520 | -0,65858 | 4,82% |
| v2 temporal OFF | +128,89 | 5,99 | 4 | +0,62498 | -0,39428 | 0,65% |

Hai execution ledger đều integrity `1,0` và reconcile thành công. Tuy nhiên v2
chỉ có 4 lifecycle, dưới support gate 50, nên không được diễn giải số lãi của v2
như bằng chứng generalization. Quan trọng hơn, v3 không thắng ở bất kỳ tiêu chí
uplift nào: delta v3-v2 là net `-590,42`, PF `-5,74`, mean expectancy
`-1,14018R`, expectancy LCB `-0,26429R`, và drawdown xấu hơn `4,17` điểm phần
trăm. Feature audit của arm v3 còn fail numerical parity ở feature index 69;
cả hai arm dùng Model 1, không phải real ticks.

Kết luận gate tổng: causality **đạt**, retention 32/128 **đạt**, v3 thắng v2
trên OOS MT5 **không đạt**. Chưa có cơ sở giảm LSTM 4096 xuống 2048. Artifact
paired cuối: `16_mt5_temporal_v3_vs_v2_20250501_96/transformer_ablation_summary.json`.

## 21. Feature 69 và temporal OOS probe (2026-09-08/09)

Feature index 69 là số position thực chia cho position-count reference. EA dùng
`MaxPositions` nếu dương, nếu không dùng 5; số position không bị clip theo năm
slot mô tả chi tiết. Verifier Python trước đây dùng `min(count, 5)/5`, vì vậy
6 position tạo 1,0 ở verifier nhưng 1,2 ở EA. `feature_contract.py` nay dùng
`position_count_reference(max_positions)` và runner truyền cấu hình thực.
Kiểm lại artifact v3 cũ: 96 rows, 0 mismatch, numerical parity PASS. Đây là
sửa phép đối chiếu, không phải thay đổi observation hoặc làm PnL cũ tốt hơn.

`17_temporal_horizon_oos_probe.json` chạy 8 window/split, mỗi window 256 bước,
burn-in 128. StandardScaler/logistic probe fit trên train, C và horizon chọn
trên validation. Test ghi nhận:

| Role | Current AUC | Current + horizon 2 AUC | Uplift |
|---|---:|---:|---:|
| BUY | 0,72532 | 0,74319 | +0,01787 |
| SELL | 0,76172 | 0,76676 | +0,00504 |

Cả hai role chọn horizon 2. Token-only AUC tại horizon 2/8/32/128 lần lượt:
BUY 0,67326/0,61578/0,51635/0,50603; SELL
0,63057/0,57495/0,54381/0,50506. Retention dài đạt ở mục 19 nhưng tín hiệu
giao dịch horizon 128 chưa đạt. Uplift probe là kết quả thăm dò trên các
window chronological, chưa có khoảng tin cậy theo block hoặc bằng chứng PnL.
Test này đã được xem để chẩn đoán; cần OOS mới cho kết luận cuối.

Fixed production head với temporal ON phát ENTRY 100% trên cả BUY và SELL;
drop từng horizon vẫn collapse. Điều này hỗ trợ thử lại head/fusion, không
cho phép kết luận temporal Transformer đã học quyết định hữu ích.

## 22. Trainer thăm dò và điều kiện chạy MT5 tiếp theo

`train_temporal_v3_ablation_candidate.py` hiện train temporal branch,
`heads.action_query` và hai calibrator; encoder/LSTM, projection cross-agent,
action embedding và các head khác đóng băng. Mặc định sequence 256, burn-in
128; chỉ tính loss/metric sau burn-in nhưng memory được cập nhật từ đầu.
Loss gồm balanced BCE theo role, pairwise ranking margin và phạt sai entry
rate. Epoch chọn bằng validation; bias BUY/SELL fit riêng trên train một lần
sau khi chọn epoch, runtime tiếp tục so margin với 0. Bias train vẫn có thể
drift trên validation/OOS; quy trình này không bảo đảm PASS.

Trainer lưu `*.best_state.pkl` nguyên tử sau mỗi epoch: trainable tensors tốt
nhất, history, arguments và hash nguồn/dataset. File này chỉ dùng khôi phục
nghiên cứu, không phải seed MT5. Các lượt bị ngắt trước khi xuất report không
được tính là candidate hoàn tất. `*.temporal_ablation.json` là báo cáo subset,
dù có schema_version 10 cũng không thay được `*.imitation.json` của pipeline
full recurrent BPTT và full-data validation.

`run_mt5_transformer_ablation.py` nay kiểm tra full imitation schema >=10,
validation enforced/PASS, candidate reload và checkpoint hash trước khi tạo
arm. Gate kết quả yêu cầu numerical feature parity ở cả hai arm và ít nhất
50 policy lifecycle mỗi arm; truyền ngưỡng thấp hơn không hạ sàn 50. Các
điều kiện reconciliation, ledger, expectancy/PnL/PF/DD vẫn được kiểm riêng.
Chỉ sau gate validation mới chạy paired dài trên cửa sổ OOS mới.

Lượt `temporal_v3_schema10_candidate_lr003_retry_20260908` đã hoàn tất 9 epoch,
best epoch 4. Sau train-bias fold, validation BUY/SELL AUC 0,61770/0,61362,
precision 0,15949/0,16535, recall 0,66316/0,79245, predicted entry rate
0,49874/0,63500. Gate subset FAIL precision và entry rate; chưa có full-data
schema 10 PASS và chưa chạy paired dài. Source/frozen tensors được xác nhận
bất biến trong report. Candidate vẫn chỉ dùng nghiên cứu.

Phát hiện lỗi fallback: selector cũ chọn MCC cao nhất ngay cả khi không tồn
tại train boundary hợp lệ. Train precision khi fold chỉ 0,14727/0,12500,
entry rate 0,52298/0,72549. Selector hiện trả `eligible`; calibrator chỉ fold
bias khi train boundary hợp lệ và gate yêu cầu cả hai role đủ điều kiện.
Thay đổi này không sửa hồi tố candidate đã xuất. Script
`audit_temporal_head_bottleneck.py` đo probe trên raw LSTM, temporal OFF/ON,
action-query và margin để xác định nơi mất tín hiệu; chỉ đọc train/validation,
lưu feature cache và hash để tránh phải chạy recurrent replay nhiều lần.

## 23. Điểm nghẽn action-query và candidate head chuẩn hóa (2026-09-09)

`18_temporal_head_bottleneck.json` đã hoàn tất trên train/validation, không đọc
test. Probe tuyến tính fit StandardScaler trên train và chọn C trên validation:

| Representation / margin | BUY validation AUC | SELL validation AUC |
|---|---:|---:|
| Raw LSTM | 0,74778 | 0,72436 |
| Temporal OFF | 0,74519 | 0,72232 |
| Temporal ON | 0,74518 | 0,72242 |
| Action-query 128 chiều | 0,63452 | 0,65228 |
| Margin production | 0,61770 | 0,61362 |

Phép đo chỉ ra mất tín hiệu có thể đọc tuyến tính tại action-query; không phải
bằng chứng thông tin bị mất hoàn toàn hoặc Transformer gây sụt AUC. Temporal
ON/OFF gần như ngang nhau trong phép đo này. Query không có phần tử abs >3
trên các hàng đã đo, nên chưa có bằng chứng tanh saturation là nguyên nhân.

`train_temporal_cached_head.py` dùng feature cache để train action-query và hai
calibrator hiện hữu bằng cross-entropy theo prior thật. Input được chuẩn hóa
theo thống kê train trong quá trình tối ưu rồi fold affine về weight/bias
`action_query`; runtime giữ nguyên architecture và 137/20. Optimizer full-batch
lr=1e-4 ổn định hơn minibatch lr=1e-3. Best-state, hash và lịch sử được lưu từng
epoch; warm-start ghi rõ optimizer được reset, không phải exact optimizer resume.

Candidate `temporal_v3_normalized_head_extended_20260909.pkl`, best epoch 542,
đã đạt **subset validation gate** sau train-only bias fit hợp lệ:

| Role | AUC | BA | MCC | Precision | Recall | Predicted entry rate |
|---|---:|---:|---:|---:|---:|---:|
| BUY | 0,74510 | 0,61318 | 0,23884 | 0,34524 | 0,30526 | 0,10606 |
| SELL | 0,69246 | 0,60674 | 0,20555 | 0,30172 | 0,33019 | 0,14500 |

Support validation 792 BUY-owner rows (95 entry), 800 SELL-owner rows (106
entry). Đây là tập development đã dùng chọn epoch và thử optimizer; không
được gọi là OOS mới. Checkpoint nguồn/backbone bất biến, fold-parity được kiểm
trên validation. Gate này không thay full schema 10, chưa chứng minh v3 thắng v2.

`validate_temporal_head_full.py` kiểm toàn bộ 143841 validation rows, so candidate
với checkpoint đầu vào cùng frozen backbone và gọi đúng
`extract_owner_hidden_sequences` production. Hợp đồng production hiện reset
window mỗi 64 bước; khác điều kiện subset 256/burn-in128. Full validation phải
đo chênh lệch này thực tế, không tự thay window để làm PASS. Mỗi chunk lưu scores
và provenance để tiếp tục sau ngắt; test không tham gia. Kết quả nằm ở
`19_full_validation_normalized_head/report.json` khi hoàn tất. Báo cáo metric
không tự chứng nhận lineage full-BPTT hay cấp sidecar imitation giả.

Reload candidate chuẩn hóa từ file bằng loader thực đã cho metric validation
khớp trước khi lưu (max drift 0, SHA-256 khớp báo cáo head-fit). Đây là kiểm
tra lưu/nạp, không phải bằng chứng full validation hoặc MT5 PASS.

Runner full validation kiểm tra chunk đúng biên, đủ số score một chiều và
không có NaN/Inf; từ chối resume khi checkpoint/baseline/config/dataset khác
contract. Khi tổng kết kiểm tra lại hash của cả bốn đầu vào. Test native
replay xác nhận chia chunk tại biên 64 bước không làm đổi hidden so với replay
nguyên khối; test resume xác nhận không replay lại phần đã hoàn thành và
không chấp nhận chunk thiếu score. Không đổi ngưỡng gate để làm kết quả PASS.

`audit_temporal_validation_regimes.py` đọc kết quả full validation hoàn tất,
ghép đúng timestamp/owner của feature cache thăm dò (thứ tự step rồi window),
so sánh score/decision trên cùng hàng giữa warm256/burn128 và reset64. Script
cũng báo metric theo bốn phần thời gian liên tiếp của validation, không fit
threshold, không train và không đánh giá test. Mục đích là phân biệt độ lệch
do replay với độ lệch do lấy mẫu; đây vẫn là chẩn đoán development.

## 24. Full validation FAIL và parity replay trên cùng mẫu (2026-09-09)

Full validation đã hoàn tất đủ 143841 hàng. Kiểm tra lại các chunk đã lưu
cho metric không đổi; checkpoint, baseline, config và dataset đều bất biến.

| Role / tổng | AUC | BA | MCC | Precision | Recall | Entry rate |
|---|---:|---:|---:|---:|---:|---:|
| BUY | 0,68135 | 0,57193 | 0,16717 | 0,30005 | 0,21358 | 0,08739 |
| SELL | 0,65749 | 0,56620 | 0,12015 | 0,21341 | 0,26877 | 0,15239 |
| Tổng | 0,66773 | 0,56896 | 0,13897 | 0,24514 | 0,24087 | 0,11976 |

Gate FAIL: precision tổng <0,30, SELL precision <0,30 và rate >1,2 lần
teacher rate; BA tổng giảm từ baseline 0,57438 xuống 0,56896, không đạt
gain >=0,02. BUY chỉ vừa qua precision floor. Cải thiện AUC/precision so với
baseline over-entry không đủ để gọi là imitation thành công hoặc schema10 PASS.

`20_validation_replay_regimes.json` đối chiếu trên đúng hàng thăm dò:

| Role | Warm256 precision → reset64 | Warm256 AUC → reset64 | Decision flip |
|---|---:|---:|---:|
| BUY | 0,34524 → 0,36986 | 0,74510 → 0,71442 | 1,89% |
| SELL | 0,30172 → 0,24051 | 0,69246 → 0,65539 | 6,00% |

SELL predicted entry tăng 0,145 → 0,1975 trên cùng mẫu. Đây là bằng chứng
replay mismatch ảnh hưởng quyết định, không chỉ sampling noise. Tuy nhiên
không cô lập riêng temporal token với LSTM: reset thay cả hai trạng thái.
BUY precision trên toàn bộ dữ liệu còn thấp hơn subset reset64, nên mức đại
diện của subset cũng là vấn đề. SELL precision bốn phần thời gian liên tiếp
0,21067 / 0,17937 / 0,22360 / 0,23886: chưa phần nào đạt sàn 0,30.

Sửa bước thu feature nghiên cứu bằng `build_temporal_production_head_cache.py`:
128 train windows và 64 validation windows rải đều theo thời gian, mỗi window
64 bước khớp biên reset tính từ đầu split; không chọn theo label, không burn-in
khác production, không đánh giá test. Cache ghi hash, timestamp/owner và
window indices. Dùng lại trainer head chuẩn hóa hiện hữu, không đổi 137/20
hay backbone; đây là nhánh sửa parity, không thay thế probe warm-history
cũ vốn dùng cho mục tiêu đo retention. Chưa chạy paired MT5 dài do gate FAIL.

Kiểm tra hồi quy cho các thay đổi parity/probe/runner/cache/head: 51 tests PASS.
Trainer head kiểm config SHA-256 của cache mới trước khi load model và kiểm
lại config/cache/source/frozen tensors trước khi xuất candidate. Report giữ
provenance replay, không gán cache reset64 thành cache warm256.

## 25. Head-fit khớp replay64: kết quả tập con và lượt full tiếp theo

`21_production64_head_cache.npz` đã hoàn tất: train 12869 hàng (6479 BUY,
6390 SELL), validation 6419 hàng (3226 BUY, 3193 SELL). Cache fit trên candidate
chuẩn hóa trước đó; các recurrent/Transformer tensors giữ nguyên.

`temporal_v3_production64_head_20260909.pkl` train tối đa120 epoch, patience30,
lr1e-4, batch16384; dừng sớm ở51, chọn epoch21. Gate metric tập con PASS:

| Role | AUC | BA | MCC | Precision | Recall | Entry rate |
|---|---:|---:|---:|---:|---:|---:|
| BUY | 0,74237 | 0,58457 | 0,21287 | 0,37021 | 0,22137 | 0,07285 |
| SELL | 0,70281 | 0,57384 | 0,17780 | 0,31600 | 0,20844 | 0,07830 |

BUY train-boundary hợp lệ được fold; SELL không có train-boundary hợp lệ nên
không áp bias fallback, giữ output đã học. Do đó subset metric PASS không
có nghĩa cả hai train-calibration gate/full-BPTT lineage đều đạt. Hash candidate
`233977bf0b2233611df402b4c38268a0ebd3487d35fbed7ba271e4d63c69e8bc`;
reload thật cho max metric drift0. Không ghi đè model đang dùng.

Đã khởi chạy `22_full_validation_production64_head`: toàn bộ143841 validation
rows, baseline là checkpoint nguồn trực tiếp `temporal_v3_normalized_head_extended_20260909.pkl`,
cùng frozen backbone và replay64. `progress.json` là trạng thái hiện thời,
chỉ `report.json` hoàn tất mới đủ kết luận metric gate. Chưa có schema10 PASS
và chưa chạy paired MT5 dài cho candidate này. Tất cả các kết quả tập con
vẫn là development validation, không phải fresh OOS.
