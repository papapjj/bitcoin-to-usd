# Kế hoạch cải thiện LCB và triển khai

> **Bản ghi runtime hiện hành (live poll, 2026-09-22 17:04):** nguồn sự thật cho
> tiến trình đang chạy là output lineage
> `outputs/bottom_top_teacher_imitation_v27_sl250_16w_20260922/`. Supervisor
> đã đi tới checkpoint boundary `resume_epoch=7`, tương đương
> `completed_epochs=6`; `stderr.log` rỗng, resume schema 2 còn hợp lệ,
> `promotable=false`, final checkpoint
> và `.imitation.json` chưa tồn tại. Bridge teacher-direct vẫn ở
> `WAIT_TEACHER`, vì vậy chưa được chạy MT5 frozen của lineage này. Các đoạn
> ghi `attempt_05/attempt_06`, epoch 10--14 hoặc lineage khác bên dưới là
> nhật ký lịch sử/diagnostic, không được dùng để suy ra trạng thái live này.
> Ở boundary này metadata reset bộ đếm window về `0/0`, nên không được dùng
> để suy ra số window đã train trong epoch 7. Resume artifact hiện có
> `1,965,807,629` bytes; SHA-256 thực tế khớp metadata
> (`6691790d4b2a56e5...486cf24f`), nên provenance của checkpoint đang được giữ.

## 1. Mục tiêu và bằng chứng đầu vào

Mục tiêu không phải làm cho một báo cáo có LCB dương bằng cách đổi công thức
hoặc nới cổng. Mục tiêu là làm cho cận dưới của kỳ vọng policy-origin dương
nhờ chất lượng trade thật, variance thấp hơn, đủ số vị thế độc lập và policy
không bị collapse.

Bằng chứng gần nhất:

| Vòng | mean/LCB | Vấn đề chính |
|---|---|---|
| run72, 04–05/02 | mean khoảng `+0.140R`, LCB `-0.01995R` | variance khoảng `1.35R`, 10 scope âm, 1 vị thế mở |
| run76 main, 05–06/02 | `+564.28/93`, LCB `-0.02184R` | actor eligible `0`, HOLD `96.54%` |
| run76 frozen candidate | `+36.56/68`, LCB `-0.07401R` | DD `15.81%`, candidate fitness không hợp lệ |
| run82 behavior-compatible | main `-260.02/124`, LCB `-0.3050R` | frozen candidate `-402.06/77`, terminal chưa flat |
| run83 directional nhẹ | main `-0.01/136`, LCB `-0.1947R` | candidate LCB `-0.1723R`, paired regression, reject |
| run84 failure replay | main `-213.99/148`, LCB `-0.1747R` | 75 anchor, 0 repaired, candidate LCB `-0.5519R`, reject |
| run85 replay support-2 | main `-1866.77/97`, LCB `-0.7526R` | candidate LCB `-0.4804R` vs baseline `-0.7309R`; relative gain, still reject |

Đối chiếu thêm với `mt5_diagnostics.json` cho cùng các run cho thấy vấn đề
không chỉ là cách tính LCB. Đây là số liệu raw của runtime (đơn vị
expectancy/LCB raw không thay thế normalized gate ở bảng trên):

| Run | Closed trades | PF | Raw expectancy | Raw one-sided LCB 95% | Net PnL | DD | HOLD |
|---|---:|---:|---:|---:|---:|---:|---:|
| run72 | 191 | `1.258` | `+4.05` | `-0.912` | `+773.74` | `8.24%` | `89.78%` |
| run74 | 144 | `1.040` | `+0.649` | `-4.704` | `+93.52` | `7.98%` | `92.17%` |
| run84 | 147 | `0.899` | `-1.471` | `-6.031` | `-216.25` | `12.95%` | `91.52%` |
| run85 | 97 | `0.203` | `-19.245` | `-23.588` | `-1866.77` | `21.71%` | `95.08%` |

Run72 là ví dụ rõ nhất: mean/PnL dương nhưng cận dưới vẫn âm; vì vậy
không được dùng một đoạn equity đẹp hoặc một run dương để promotion. Run84/85
còn cho thấy việc tiếp tục tối ưu trên policy đã collapse làm edge và LCB xấu
đi nhanh hơn. Arm `live_veto_min12_lcbfix` cũng vẫn âm (`123` trades,
PF `0.611`, raw LCB `-13.821`, net PnL `-1022.44`, HOLD `94.98%`) và manifest
ghi rõ `promotable=false`; nó chỉ là research evidence, không phải baseline
production.

Trace run76 còn ghi `flat_entry_mass_mean` gần `10^-6` trong khi
`flat_entry_mass_loss` vẫn dương. Sàn hiện tại là
`ReLU(floor - mass)^2`; khi HOLD đã bão hòa, gradient qua softmax quá nhỏ để
kéo entry mass trở lại. Đây là nguyên nhân kỹ thuật có thể sửa và kiểm thử
độc lập, không phải suy đoán từ PnL.

## 2. Nguyên tắc không được phá

- Không đổi công thức LCB, không giảm `min_samples`, không bỏ negative-scope,
  terminal-flat, reconciliation, frozen paired hoặc real-tick gate.
- Mọi reward cuối cùng phải đi theo `trace_id -> position_ticket -> owner`;
  chỉ vị thế độc lập đã đóng mới vào mẫu LCB.
- Không dùng PnL main online làm OOS. Main chỉ là tín hiệu train; quyết định
  checkpoint phải dựa trên frozen candidate/baseline cùng cửa sổ và cùng seed.
- Candidate-only trước; production default giữ `log_loss_weight=0`.
- RLHF preference pairwise hiện đã có contract/validator/queue nhưng vẫn
  `disabled/candidate-only`; xem
  [`RLHF_PREFERENCE_LCB_DEPLOYMENT_PLAN.vi.md`](RLHF_PREFERENCE_LCB_DEPLOYMENT_PLAN.vi.md).
  Pairwise accuracy hoặc rater agreement không thay cho frozen MT5/OOS LCB;
  RLHF chỉ được xét promotion sau khi reward attribution và unique-position
  ledger reconcile đầy đủ.
- Reward model Bradley--Terry và PPO/reference-KL primitive đã có ở candidate
  nhưng closed-loop rollout -> reward -> PPO -> checkpoint chưa nối vào MT5;
  `teacher`/`mt5_rater` không được gọi là human RLHF.
- MT5 train/validation dùng runner, bridge, report và ledger riêng. Có thể song
  song các worker OOS chỉ khi chúng clone portable runtime hoàn toàn độc lập;
  không chạy hai job cùng ghi một runner.

### Quy tắc song song hóa

Không phải mọi bước chậm đều nên chạy song song. Phân rã hiện tại:

| Nhóm | Có thể song song? | Điều kiện |
|---|---:|---|
| Đọc manifest, parse report, audit trace/ledger, static test | Có | Chỉ đọc; không ghi chung artifact |
| So sánh các cửa sổ frozen OOS | Có | Mỗi worker có portable MT5 runner, `Common/Files`, bridge, report và ledger riêng; checkpoint chỉ đọc |
| Main train -> champion -> candidate -> baseline của cùng một arm | Không | Có phụ thuộc causal: snapshot/champion phải được tạo trước, sau đó mới paired validation và rollback |
| Hai MT5 job dùng chung terminal/runner/ledger | Không | Global lock và file bridge có thể trộn sự kiện, làm mất giá trị reconciliation |
| Hai arm hoàn toàn isolated | Có giới hạn | Chỉ bật khi kiểm tra đủ disk/RAM/CPU và mỗi worker có namespace riêng; nếu không thì tuần tự để giữ bằng chứng |

Code đã có isolated OOS worker (`run_frozen_temporal_generalization.py`) và
nhánh compare-core dùng `ThreadPoolExecutor`; đây là song song hóa an toàn ở
biên độc lập. Run81 vẫn chạy tuần tự vì cùng host còn artifact tester orphan,
stale lock và đỉnh bộ nhớ từng làm wrapper chết. Tăng số worker trong tình
hình đó chỉ làm nhanh việc tạo ra bằng chứng không tin cậy, không làm LCB tốt
hơn.

## 3. Các đòn bẩy LCB

LCB một phía có thể tăng bởi bốn hướng hợp lệ:

1. Tăng mean net-R: giảm INITIAL_SL/SL sớm, tránh action có action-local LCB
   âm, giữ TP/EXPERT capture tốt hơn.
2. Giảm variance: không cho một vài lệnh thắng lớn che nhiều SL; theo dõi
   close-reason, side, owner, action và block/day LCB.
3. Tăng số mẫu độc lập hợp lệ: chỉ tăng entry khi có tín hiệu học và không
   tạo overtrade; loại pending chưa fill và duplicate close khỏi mẫu.
4. Giữ policy có khả năng khám phá: tránh HOLD collapse, nhưng exploration
   phải có risk cap và không được biến thành ép trade.

Không được dùng cách giả tăng LCB: giảm z-score, gộp duplicate close, dùng
subgroup dương thay aggregate, hoặc giữ checkpoint có DD/LCB OOS không đạt.

## 4. Triển khai đã làm trong nhánh candidate

Đã thêm `five_flat_entry_mass_log_loss_weight`, mặc định `0.0`:

- `config.py`: field và validation non-negative.
- `five_shared_policy.py`: helper dùng chung cho PPO sequence và delayed
  lifecycle/pattern sequence.
- Telemetry mới: `flat_entry_mass_log_loss_weight` và
  `flat_entry_mass_log_loss`.
- Test gradient bảo đảm log barrier còn gradient âm khi entry mass đã ở

Directional floor cũng đã được kiểm tra end-to-end cho delayed lifecycle path:

- config có `five_flat_entry_direction_mass_floor_start/end` và weight;
- `optimize_independent_pattern_steps` dùng đúng owner LONG/SHORT trên flat
  account và cộng loss vào cùng `five_ppo_adam` sole-writer;
- regression test xác nhận loss làm tăng mass của cả hai owner, không chỉ ghi
  telemetry;
- arm `run83` dùng floor nhẹ `0.06 -> 0.02`, weight `0.25` nhưng vẫn bị reject
  bởi frozen LCB/terminal/paired gate. Vì vậy directional floor được giữ như
  candidate-only primitive, không được xem là giải pháp LCB hay bật production.
  `10^-8`.
- Treatment run77 trên `2025.02.05..06` xác nhận barrier `0.10` có tác động
  ban đầu nhưng chưa duy trì được: giữa run HOLD giảm về khoảng `83%` rồi
  tăng lại khoảng `91%`, entropy giảm còn `0.017` và actor-eligible còn
  `0.12`. Run bị `max_wall_sec=900` ở sequence 361 nên chưa có frozen paired
  gate; không được coi là LCB/OOS PASS. Trace cho thấy regularizer trước đó
  đọc phân phối sau legality mask, vì vậy các role flat bị HOLD-only mask
  vẫn có log loss lớn nhưng không kéo được raw entry logits.
- Bản sửa tiếp theo giữ nguyên execution mask và chuyển nguồn regularizer
  sang `raw_unmasked_action_logits`; đây là sửa causal của gradient, không
  phải nới mask hay ép lệnh. Cần chạy lại treatment bằng runtime mới trước
  khi tăng trọng số hoặc bật directional floor.
- Bản sửa retention tiếp theo không chờ cờ `collapsed` cũ của sampler:
  nếu HOLD đạt `0.80`, effective action count `<=1.85` hoặc normalized
  entropy `<=0.35`, hệ thống phải validate champion snapshot trước khi dùng
  final weights. Main sampler có exploration nên có thể báo cờ cũ là false ở
  HOLD `0.94`, trong khi frozen deterministic replay biến thành gần 100%
  HOLD. Champion vẫn phải qua frozen/reconciliation/LCB gate; đây không phải
  đường tắt promotion.
- Run79 dùng retention gate mới nhưng wrapper MT5 bị mất giữa main pass:
  PID điều phối không còn, tester artifact chỉ còn `stage=started`, các file
  bridge/execution không tăng thêm và không có report finished. Run79 được
  đánh dấu incomplete, không nhập vào LCB/OOS sample và không dùng để biện
  minh triển khai.
- Run80 xác nhận retention gate đã được gọi khi main rơi vào HOLD collapse,
  nhưng wrapper chết trong lúc bắt đầu champion validation; vì vậy run80 chỉ
  là operational/incomplete artifact, không được tính là kết quả OOS.
- Run81 chạy lại tuần tự với cùng profile và hoàn tất sạch cả bốn nhánh:
  main, champion, frozen candidate và frozen baseline. Main có `+80.72`,
  `147` policy-origin lifecycle, DD `13.53%`, policy LCB `-0.1346R`,
  HOLD cuối `92.74%`, effective actions `1.4389`, entropy chuẩn hóa
  `0.1870`; collapse guard bật và chọn quay về source seed. Champion snapshot
  được kiểm tra độc lập nhưng bị loại: `-486.53`, `64` trade, PF `0.58`,
  DD `9.82%`, LCB `-0.5417R`. Frozen candidate có `0` trade và không có
  policy-origin lifecycle; baseline paired có `-537.61`, `153` trade,
  DD `21.50%`, LCB `-0.2430R`. Cả hai ledger đều reconcile, nhưng candidate
  không đủ fitness nên paired gate tổng thể reject và rollback giữ source seed.
- Run81 cho thấy việc sửa raw-logit barrier và retention đã làm pipeline quan
  sát/rollback đúng hơn, chưa chứng minh edge được duy trì. Champion dương
  trong main snapshot không tổng quát sang frozen validation; candidate bị
  deterministic replay thành no-trade. Vì vậy không được lấy main PnL hay
  champion snapshot làm bằng chứng triển khai.
- Cụ thể hơn, frozen candidate `840001` có `7,075/7,075` action là HOLD,
  `five_direct_rnn_rl.optimizer_steps=0` và `0` lifecycle. Đây là dấu hiệu
  behavior contract sau reload đang khác main online: main có behavior
  exploration/online learning, còn frozen validation đang deterministic. Cần
  sửa và kiểm thử contract này trước khi kết luận log barrier thất bại về
  mặt kinh tế; không được sửa bằng cách bỏ frozen gate.
- Đã tạo profile research-only
  `mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_log_floor_behavior_compatible_low_disk.yaml`.
  Profile giữ nguyên reward/geometry/log-floor/veto của run81 và chỉ bật
  `five_frozen_behavior_sampling_enabled=true` với seed `20260702` cho frozen
  replay. Profile được viết tự đủ vì runner candidate áp override lên
  `--config` và không merge đệ quy metadata `extends`; test phải kiểm tra trực
  tiếp `learning_overrides` để không tạo cảm giác profile đã kế thừa nhưng
  thực tế chạy default.
- Run82 chạy profile này và hoàn tất sạch đủ bốn nhánh. Main đạt `-260.02`,
  `124` lifecycle, DD `13.13%`, policy LCB `-0.3050R`, `5` negative scopes;
  champion frozen đạt `-993.42`, `130` trade, DD `11.42%`, LCB `-0.3727R`;
  candidate frozen đạt `-402.06`, `77` trade, DD `6.33%`, LCB `-0.3149R`
  nhưng còn `1` terminal position; baseline paired đạt `-501.28`, `235`
  trade, DD `17.66%`, LCB `-0.2080R`. Tất cả report/ledger đều reconcile.
  Candidate vẫn bị loại vì expectancy/LCB âm, terminal chưa flat và paired
  LCB regression; rollback chọn source seed. Behavior-compatible sampling đã
  sửa lỗi candidate `0` trade của run81, nhưng không tạo ra edge.
- Để tránh wrapper chết vì đỉnh bộ nhớ, `evaluate_frozen_clone` chỉ bỏ
  `deepcopy` thứ hai khi nguồn đã là champion snapshot vốn đã isolated;
  candidate/baseline vẫn clone phòng thủ. Đây là tối ưu vận hành, không nới
  bất kỳ reward hoặc promotion gate nào.
- Profile A/B đầu tiên:
  `mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_log_floor_low_disk.yaml`.

Profile mới giữ nguyên action-only control; chỉ bật log barrier `0.10`. Như
vậy kết quả của vòng đầu trả lời đúng một câu hỏi nhân quả: barrier có ngăn
HOLD collapse và cải thiện LCB không.

## 5. Ma trận chạy theo phase

### Phase A — unit/static gate

Chạy trước MT5:

```powershell
pytest -q tests/test_pattern_discovery_reward.py tests/test_nam_tac_nhan_dung_chung.py tests/test_config_inheritance.py tests/test_mt5_runner.py
py -m compileall -q src tests
```

Pass bắt buộc:

- log barrier không tác động khi weight bằng `0`;
- weight âm bị reject;
- square loss cũ vẫn giữ hành vi;
- profile load đúng `137 -> 20`, source config/seed không bị sửa;
- telemetry ghi rõ floor, square loss, log loss và flat samples.

### Phase B — matched MT5 ablation

Chạy control action-only và treatment log-floor trên cùng seed, model, ngày và
timeout. Mỗi candidate phải có main learning, frozen candidate và frozen
baseline. Chạy tuần tự trên một máy nếu còn dùng chung disk/terminal; chỉ
parallelize khi mỗi worker có portable runner + Common/Files + ledger riêng.

Kết quả phải đạt tối thiểu để sang phase C:

- `reconciled=true`, terminal flat ở frozen;
- policy-origin lifecycle `N >= 50` cho một cửa sổ thí điểm;
- `actor_eligible_fraction >= 0.05` hoặc delayed lifecycle actor credit phải
  được chứng minh riêng bằng optimizer telemetry;
- `flat_entry_mass_mean` không còn sát `0` ở các batch flat;
- HOLD không vượt `95%` ở cuối main nếu vẫn có đủ legal entry;
- không tăng DD quá `1pp` so với baseline.

Các điều kiện trên là diagnostic gate, chưa phải promotion.

### Phase C — cải thiện mean và variance theo nguyên nhân

Chỉ nếu Phase B qua anti-collapse mới thử từng arm một:

1. `action_only + log_floor` — arm đang triển khai.
2. Thêm directional mass floor nhỏ (`LONG/SHORT`) để tránh một phía bị xóa;
   không bật cùng lúc với thay đổi reward.
3. Bật transition probe interval `64` để kiểm tra phục hồi sau negative
   transition; so sánh với không probe.
4. Sau khi có đủ lifecycle, thử failure replay giới hạn; đo riêng INITIAL_SL,
   timeout và pending-fill, không gộp thành penalty HOLD.
5. Chỉ sau cùng mới điều chỉnh reward weight/clip. Mỗi arm phải giữ nguyên
   geometry `SL/TP/horizon/fill price`.

### Phase D — nhiều cửa sổ và LCB thật

Mỗi candidate qua ít nhất ba cửa sổ ngày độc lập, gồm regime đã có kết quả
khác nhau. Với mỗi cửa sổ lưu:

- unique completed positions, mean/std net-R, one-sided LCB;
- block/day bootstrap LCB;
- PF, WR, payoff, DD, immediate-exit ratio;
- long/short, action, owner, close reason;
- actor/reward eligibility, PPO update count, entropy, entry mass;
- ledger reconciliation, terminal-flat và feature parity.

Gộp bằng lifecycle unique ticket; không cộng các dòng close. Candidate chỉ
được xem là đạt research gate khi mean pooled dương, pooled LCB dương, không
có block/day LCB âm nghiêm trọng, DD qua gate và không có cửa sổ hợp lệ nào
âm theo policy-origin evidence.

Runner paired OOS `scripts/run_v118_isolated_oos.py` hiện thực hiện thêm gate
pooled sau khi tất cả cửa sổ hoàn tất: đọc ledger chỉ đọc, giữ lifecycle có
owner trace, loại vị thế chưa final-close, phát hiện duplicate
`position_ticket` trong cùng portable/OOS window (ticket MT5 có thể được reset
giữa các window), kiểm tra reconciliation/terminal-flat/feature-parity/real-
tick từng cửa sổ, rồi tính robust pooled policy LCB. Candidate phải có pooled
LCB `>= 0`, baseline phải có đủ evidence, và pooled candidate LCB không được
tụt quá `0.02R` so với baseline. Vì vậy `all_windows_eligible=true` chỉ xuất
hiện khi cả gate từng cửa sổ lẫn gate pooled đều đạt.

Runner hiện cũng tính `day_bootstrap.block_lcb_r`: lifecycle được gom theo
ngày đóng, mỗi ngày đóng góp một mean net-R, sau đó bootstrap các ngày và lấy
cận dưới 5% bảo thủ hơn giữa bootstrap percentile và robust prior-shrunk LCB.
Candidate phải có tối thiểu `3` ngày độc lập và day-bootstrap LCB không âm;
baseline phải có đủ day-bootstrap evidence. Paired gate còn kiểm tra day LCB
của candidate không tụt quá `0.02R` so với baseline. Nếu ledger thiếu
`exit_time`, sample chỉ được gắn vào block cửa sổ tương ứng và có thể bị loại
do không đủ ngày độc lập; không được coi nhiều trade trong một ngày là nhiều
bằng chứng độc lập. Parser hiện đọc `exit_time` dạng `datetime` native của
MT5, ISO string có timezone và epoch seconds; datetime legacy không có timezone
giữ ngày broker-local thay vì rơi nhầm toàn bộ về `window-XXXX`.

Chuỗi này đã có supervisor nối tiếp
`scripts/run_teacher_paired_oos_supervisor.ps1`. Supervisor chờ full-teacher
geometry checkpoint, kiểm tra geometry không còn lock ghi, rồi mới truyền
candidate geometry và baseline frozen vào runner paired. Nó ép `Model 4`, mặc
định ba cửa sổ độc lập, tách log/stdout/stderr và dừng fail-closed nếu output
OOS dở dang thiếu `summary.json`. Gate không đạt vẫn được giữ làm bằng chứng
research; supervisor không tự promotion.

## 6. Tiêu chí rollback và promotion

Rollback ngay về seed nguồn nếu một trong các điều kiện sau xảy ra:

- candidate frozen fitness không hợp lệ;
- LCB candidate giảm quá ngưỡng paired;
- DD tăng quá `1pp` so với baseline;
- HOLD/action collapse hoặc actor credit starvation;
- ledger không reconcile, terminal không flat, feature parity fail;
- main dương nhưng frozen candidate âm hoặc chỉ có vài trade.

Promotion chỉ sau khi seed imitation gate, real-tick evidence, nhiều cửa sổ
OOS và aggregate gate đều đạt. Không dùng profile log-floor trực tiếp cho
production trước khi có evidence đó.

## 7. Thứ tự thực thi kế tiếp

### 7.1. Trạng thái readiness hiện tại

Trước khi chạy thêm MT5, phải giữ đúng provenance của imitation. Seed
`value_candidate_v26_jan2025_parity_v2_sl250.pkl` hiện chưa có
`.imitation.json`. Dataset NPZ trong thư mục seven-choice vẫn là schema `14`
của surface bảy hành động, không có `confirmation_audit.enabled`; nó không
được truyền thẳng vào `train_teacher_imitation.py --action-only`.

Nguồn historical labeler đã được khôi phục bằng MT5 Strategy Tester Model 1
trên 2025-01-01..2025-01-08. Log xác nhận 4.893 dòng (BUY 535, SELL 551,
HOLD 3.807, rejected pivots 241) và file được ghi vào MetaQuotes Common Files.
Từ artifact đó đã dựng được schema 9 smoke test. Artifact 16 tuần có sẵn trong
Common Files cũng đã dựng thành công schema 9 đầy đủ 77.295 dòng, khớp bộ dữ
liệu được mô tả trong tài liệu pipeline. Không được sửa JSON để vượt gate; mọi
manifest phải giữ hash CSV/NPZ và confirmation audit.

Readiness snapshot:

| Gate | Hiện trạng | Hành động bắt buộc |
|---|---|---|
| Checkpoint source/hash | Có, hash ổn định | Giữ nguyên seed, chỉ tạo candidate mới |
| Imitation sidecar | Thiếu | Train action-only từ dataset schema 9, rồi kiểm tra reload/hash |
| Dataset teacher schema 9 | Đã có smoke 7 ngày và full 16 tuần; verify-only pass | Dùng artifact 16 tuần cho action BPTT; giữ hash manifest |
| Storage | Đã dọn 29 runner tạm `mt5_v186_*` (~14,59 GB); hiện còn khoảng `14,55 GB` trống, action preflight cần `2,93 GB` | Giữ riêng các lineage train, không xóa seed/dataset/output |
| Real-tick | Chưa có; các run72–85 là Model 1 | Chạy lại frozen matrix bằng Model 4 và kiểm tra marker mới |
| LCB/OOS | Run85 chỉ có relative improvement, LCB vẫn âm | Lặp ít nhất 3 cửa sổ độc lập, pooled LCB phải dương |

Kiểm tra MT5 gần nhất cũng chưa chứng minh edge: run51 có `458` vị thế đóng,
P/L ledger `-1372.12` và fitness registry `-0.1118`; run52 có `6900` dòng
ledger nhưng tất cả là `NOOP_HOLD`, không có vị thế nào. Vì vậy không dùng
registry hoặc số dòng quan sát thay cho unique closed-position LCB; cả hai run
chỉ là bằng chứng diagnostic/research và không đủ điều kiện triển khai.

Vì vậy trạng thái hiện tại là `not deployable`; việc chạy thêm support-2 MT5
trước khi hoàn thành provenance và sidecar chỉ tạo thêm research artifact,
không tiến gần promotion.

### 7.2. Thứ tự thực thi

1. Giữ run81 ở trạng thái research-only; không deploy source seed, champion
   snapshot hay candidate hash. Imitation sidecar vẫn thiếu và Model 1 chưa có
   real-tick evidence.
2. Giữ run82 và run83 ở research-only. Run82 xác nhận behavior contract nhưng
   LCB âm; run83 xác nhận directional floor đã chạy thật nhưng không cải thiện
   LCB, candidate còn open position và paired gate reject. Không tiếp tục sweep
   floor/log weight trên cùng seed và cùng cửa sổ `2025.02.05..06`.
3. Run84 đã kiểm chứng failure-anchor replay trên cửa sổ mới: 75 anchor được
   admit, 61 lượt replay qua 9 batch nhưng 0 anchor repaired/resolved; candidate
   LCB -0.5519R thấp hơn baseline -0.0721R và paired gate reject. Run85 thêm
   `min_bucket_support=2`: 82 anchor, 40 lượt replay, 2 anchor resolved; frozen
   candidate cải thiện tương đối PnL/DD/LCB so với baseline trong cửa sổ xấu,
   nhưng LCB vẫn -0.4804R và fitness không hợp lệ. Giữ support-2 như treatment
   nghiên cứu, không tăng weight/capacity mù quáng; cần kiểm chứng ít nhất 3 cửa
   sổ độc lập và yêu cầu aggregate LCB dương trước khi giữ lại.
4. Dùng dataset teacher schema 9 16 tuần đã dựng từ historical labels có
    provenance. Chạy preflight dataset trước khi cấp dung lượng cho training;
    sau đó chạy `train_teacher_imitation.py --action-only` trên một output mới.
    Sidecar chỉ hợp lệ khi report schema hiện hành, hash checkpoint khớp,
    candidate reload pass, validation gate và autonomous continuous-validation
    gate đều pass. Artifact hiện tại:

    - CSV nguồn: `C:\Users\Admin\AppData\Roaming\MetaQuotes\Terminal\Common\Files\bottom_top_historical_opposite_pivot_v4_16w_sl075.csv`;
    - NPZ schema 9: `outputs/bottom_top_teacher_confirmed_geometry_v9_sl075_20260921/teacher_16w_confirmed_schema9.npz`;
    - CSV SHA-256: `f4722163821f3177f4ef1948bcd44c6ba094be0d037744d969e982c05c64149b`;
    - NPZ SHA-256: `3716df72abc6a450dda1a4102ebb899420ae057791437f2ea2f35bf9c9e02474`;
    - verify-only: pass; readiness còn đúng một blocker là `no_imitation_sidecar`.

    Nếu cần tái tạo, lệnh dựng dataset phải giữ confirmation contract, không
    được chuyển từ NPZ schema 14 bằng cách sửa JSON:

   ```powershell
   py scripts/build_teacher_imitation_dataset.py `
     --labels <bottom_top_historical_labels.csv> `
      --output outputs/bottom_top_teacher_confirmed_geometry_v9_sl075_20260921/teacher_16w_confirmed_schema9.npz `
     --confirmation-bars 3 `
     --max-confirmation-wait-bars 5
   ```

   Sau đó chạy `--verify-only` trên đúng NPZ schema 9, rồi mới cấp storage cho
    action BPTT. Dataset phải có source-label hash và manifest xác nhận tồn tại.
    Có thể kiểm tra nhanh toàn bộ provenance bằng auditor chỉ-đọc:

   ```powershell
    py scripts/audit_teacher_imitation_readiness.py `
      --checkpoint outputs/seven_choice_teacher_causal_v17_20260917/value_candidate_v26_jan2025_parity_v2_sl250.pkl `
      --dataset outputs/bottom_top_teacher_confirmed_geometry_v9_sl075_20260921/teacher_16w_confirmed_schema9.npz
   ```

    Exit code khác `0` là còn blocker; hiện auditor trả `no_imitation_sidecar`.
    Dataset và checkpoint đã pass contract. Storage gate đã pass sau khi dọn
    đúng 29 runner tạm không có process/active marker. Candidate batch-size 2
    đã được pause an toàn tại epoch 1, batch 30/540 với resume schema 2,
    `promotable=false`. Lần batch-size 8 trước đã đạt validation epoch 6 nhưng
     terminal/UI mất trước khi ghi artifact, làm lộ khoảng trống: trainer cũ chỉ
   lưu khi có stop-request. Khoảng trống này đã được sửa: trainer hiện ghi
   resume schema 2 sau mỗi validation epoch (`status=checkpoint`, contract/hash
   đầy đủ, cursor ở đầu epoch kế tiếp). Supervisor còn ghi checkpoint vận hành
   giữa epoch mỗi `128` recurrent windows, giữ optimizer/RNG/permutation/cursor;
    artifact này vẫn `promotable=false` và không phải candidate. Attempt
    batch-size 8 cũ đã dừng ở 112/540 cửa sổ và để lại resume artifact. Sau khi
    sửa lỗi loader boundary, attempt 05 đã được pause mềm tại boundary an toàn
    sau 6 epoch; resume `status=paused` giữ nguyên optimizer, RNG và cursor.
     Attempt 05 được khởi chạy trước khi sửa lỗi truyền role metrics vào record
     mà mentor đọc nên chỉ là diagnostic lineage. Attempt 06 hiện đã nạp resume
     đó bằng code role-aware mới. Epoch 10 đã hoàn tất train `540/540` và
     validation replay `656/656`; so với epoch 9, validation BA tăng từ
     `0.70843` lên `0.70956`, MCC từ `0.42385` lên `0.42602`, precision từ
     `0.50394` lên `0.50573`, recall từ `0.48483` lên `0.48690`, còn
     train-validation BA gap giảm từ `0.01642` xuống `0.01632`. BUY/SELL
     BA gap là `0.00385`. Mentor vẫn ở `hold_guidance`, tăng
     teacher-confidence weight tiếp lên `0.50`; đây là tín hiệu
     imitation/validation, không phải LCB hay PnL OOS. Checkpoint vận hành sau
     epoch 10 có `resume_epoch=11`, `completed_epochs=10`, `promotable=false`;
     worker đã chuyển sang epoch 11. Epoch 11 đã hoàn tất train/replay nhưng
     giảm còn BA `0.70335`, MCC `0.41678`, precision `0.50146`, recall
     `0.47379`; gap tăng lên `0.02387`, nên epoch 10 vẫn là best imitation
     hiện tại và epoch 11 không được promote. Candidate cuối chỉ được đi tiếp nếu history mới có
     đầy đủ role signals, reload/hash và continuous-validation gate; cho đến
     lúc đó vẫn chưa có candidate, sidecar hợp lệ hay OOS claim.
     Epoch 12 đã hoàn tất train `540/540` và validation replay `656/656`.
     Kết quả validation của epoch 12 là BA `0.70591`, MCC `0.42472`,
     precision `0.51185`, recall `0.47655`, gap `0.02553`; so với epoch 10
     (BA `0.70956`, MCC `0.42602`, precision `0.50573`, recall `0.48690`,
     gap `0.01632`), BA/MCC/recall giảm và gap tăng. Vì vậy epoch 10 vẫn là
     best imitation checkpoint; epoch 12 không được promote dù precision
     tăng. Checkpoint sau epoch 12 có `resume_epoch=13`,
     `completed_epochs=12`, `promotable=false`; hash metadata và file thực
     tế khớp trực tiếp (`hash_match=true`, hash hiện tại
     `cb4a89d5...f60058309`). Worker vẫn còn sống, stderr vẫn rỗng và chưa
     tạo action artifact cuối, imitation sidecar hoặc OOS claim. Epoch 13 đã
     bắt đầu và đã hoàn tất train `540/540` (`100%`), sau đó replay validation
     `656/656` (`100%`). Epoch 13 cho BA `0.70142`, MCC `0.42508`, precision
     `0.52381`, recall `0.46276`, gap `0.02807`; so với epoch 10, BA giảm
     `0.00814`, MCC giảm `0.00094`, recall giảm `0.02414`, còn precision tăng
     `0.01808`. Entry rate dự đoán chỉ còn `0.11014` so với label rate
     `0.12467` (ratio `0.88345`), cho thấy xu hướng under-entry dù chưa chạm
     ngưỡng mentor `0.70`. Vì vậy epoch 13 không được promote; epoch 10 vẫn
     là best imitation checkpoint. Checkpoint sau epoch 13 có
     `resume_epoch=14`, `completed_epochs=13`, `windows_done=0`, hash
     metadata/file khớp (`hash_match=true`, hash hiện tại
     `ab6e3260...e6de9f756a`), nhưng vẫn `promotable=false`; chưa có action
     artifact cuối, imitation sidecar hoặc OOS claim. Worker đã chuyển sang
     epoch 14 và đã ghi checkpoint vận hành tại `128/540` (`23.70%`), hash
     metadata/file khớp trực tiếp (`2a92e581...e646951`). Tại lần kiểm tra mới
     nhất worker vẫn sống và tiến tới `136/540`; chưa có artifact cuối,
     sidecar, geometry hay paired OOS summary.

     Phân tích code và metric cho thấy mentor cũ có blind spot: chỉ bật
     recovery khi predicted entry rate thấp hơn `70%` label rate hoặc recall
     tuyệt đối dưới `0.20`. Vì epoch 13 có ratio `0.88345` và recall `0.46276`,
     nó bị ghi `hold_guidance` dù BA giảm `0.00449` so với epoch 12 và recall
     giảm `0.01379`. Đã thêm tín hiệu regression theo epoch (BA giảm hơn
     `0.002`, recall giảm hơn `0.010`, hoặc entry-rate dưới `95%` prevalence)
     để lesson controller giảm false-entry penalty, nới hard-negative và tăng
     ranking recovery trong epoch kế tiếp; tín hiệu này không tạo nhãn, không
     đọc test/OOS và không bypass economic/promotion gate. Test mô phỏng đúng
     epoch 13 đạt `3/3`, compile pass. Toàn file test không được dùng làm kết
     luận pass vì Windows vẫn từ chối quét thư mục temp/basetemp (`WinError 5`)
     ở các fixture cần `tmp_path`.

     Đây là sửa chống imitation co hẹp, chưa phải sửa LCB kinh tế: loss hiện
     vẫn học teacher action/teacher_net_r margin, không nhận settlement net-R
     theo lifecycle MT5. Vì vậy arm kế tiếp phải tách hai câu hỏi: (a) lineage
     teacher mới có giữ BA/recall qua epoch hay không; (b) một arm actor dùng
     lifecycle settlement net-R có làm pooled/day-bootstrap LCB tăng hay không.
     Không lấy reward signal nội bộ của imitation làm proxy cho câu (b).

     Vì patch được nạp sau khi PID `16000` đã khởi chạy, worker hiện tại vẫn là
     lineage code cũ. Không resume cùng output v26 bằng code mới nếu chưa thêm
     version vào contract; để giữ tính tái lập, sau khi worker terminal phải
     tạo output lineage mới `v27` từ cùng source/dataset/seed, chạy smoke
     teacher tối đa 3 epoch để xác nhận mentor thực sự phát
     `validation_recovery`, rồi mới cấp một run dài. Chỉ sau khi v27 có
     sidecar hợp lệ mới cho geometry supervisor và paired OOS supervisor chạy;
     arm settlement net-R dùng cùng candidate hash và baseline frozen, không
     trộn với v26.

     Runtime hiện là nút thắt cần xử lý sau lineage này: epoch 10 mất khoảng
     `3877s`, trong đó validation replay mất khoảng `808s`. Hàm replay hiện
     chạy các cửa sổ recurrent độc lập và báo `hidden_cache_reused=false`.
     Có thể tối ưu bằng cache/batching có kiểm thử parity, nhưng không được
     giảm số cửa sổ, bỏ replay đồng nhất train/validation hoặc nới bất kỳ LCB
     gate nào. Không dừng worker đang chạy chỉ để áp dụng tối ưu này.

     Kiểm chứng code gate gần nhất (2026-09-22):
     `tests/test_pooled_oos_runner.py` đạt `8/8`; `run_v118_isolated_oos.py`,
     `contextual_expectancy.py` và `learning_integrity.py` đều compile thành
     công. Lần chạy gộp thêm `test_contextual_expectancy_v13.py` đã chạy qua
     các test logic nhưng một fixture `tmp_path` bị `WinError 5` khi pytest
     quét thư mục temp của Windows; đây là lỗi quyền môi trường, không phải
     lỗi LCB. Không dùng lần chạy lỗi môi trường đó để kết luận gate pass/fail.
     Kiểm tra provenance trực tiếp sau đó cũng khớp cả ba hash: resume hiện tại,
     checkpoint nguồn và dataset schema 9; resume vẫn `status=checkpoint`,
     `promotable=false`. Điều này cho phép tiếp tục training mà không làm mất
     tính truy nguyên của mẫu LCB/OOS.
5. Imitation không còn chỉ là chép nhãn tĩnh. Teacher-guided mode chạy một
   lesson controller xác định được: dạy timing HOLD/ENTRY qua BPTT, hard-negative
   bị reject, confidence theo `teacher_net_r`, ranking/cân BUY-SELL và self-fed
   state; sau validation nó tự điều chỉnh có giới hạn penalty, lesson weights,
   role weights và learning-rate. Teacher state, tín hiệu và hành động được
   hash-contract/resume cùng checkpoint; test-only, MT5 future PnL và frozen OOS
   không đi vào quyết định. Chạy arm mới trên output lineage riêng, giữ batch8
   static resume làm baseline đối chiếu bằng `--no-mentor-guided` nếu cần.
   Lệnh lineage teacher hiện hành:

   ```powershell
   py -u scripts/train_teacher_imitation.py `
     --config configs/he_chuyen_gia_mot_context_mot_policy.yaml `
     --checkpoint outputs/seven_choice_teacher_causal_v17_20260917/value_candidate_v26_jan2025_parity_v2_sl250.pkl `
     --dataset outputs/bottom_top_teacher_confirmed_geometry_v9_sl075_20260921/teacher_16w_confirmed_schema9.npz `
     --output outputs/bottom_top_teacher_imitation_v26_sl250_16w_20260921/value_candidate_v26_jan2025_parity_v2_sl250_imitation_16w_teacher_full_v1.pkl `
     --action-only --mentor-guided --mentor-min-epoch 2 `
     --epochs 80 --batch-size 8 --feature-batch-size 32 `
     --learning-rate 0.003 --recurrent-learning-rate 0.00001 `
     --patience 12 --seed 0 --sequence-length 64 `
     --false-entry-penalty 0.50 --ranking-loss-weight 0.50
   ```

    Không chạy trùng output; nếu process mất sau một epoch, chạy lại cùng lệnh
    để nạp `<output>.resume.pkl`. Với run CPU dài, dùng supervisor một-worker để
    tự resume nhưng vẫn giữ log riêng từng attempt:

    ```powershell
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_teacher_imitation_supervisor.ps1 `
      -Checkpoint outputs/seven_choice_teacher_causal_v17_20260917/value_candidate_v26_jan2025_parity_v2_sl250.pkl `
      -Dataset outputs/bottom_top_teacher_confirmed_geometry_v9_sl075_20260921/teacher_16w_confirmed_schema9.npz `
      -Output outputs/bottom_top_teacher_imitation_v26_sl250_16w_20260921/value_candidate_v26_jan2025_parity_v2_sl250_imitation_16w_teacher_full_v1.pkl `
      -MaxAttempts 4
    ```

    Supervisor không chạy song song, không xóa resume; nó chỉ retry khi worker
    đã terminal và dừng ngay khi final checkpoint xuất hiện. Resume ở boundary
    sau validation epoch có permutation rỗng/cursor 0 hợp lệ; chỉ resume giữa
    batch mới cần permutation/cursor khớp toàn bộ recurrent windows. Artifact chỉ
   được gọi là candidate khi report schema 12, reload/hash, validation gate và
   continuous self-fed gate đều pass.

   Geometry supervisor có lock theo thư mục và `owner.json`. Khi process chủ
   lock đã kết thúc nhưng lock còn sót, supervisor mới phải kiểm tra PID trước,
   chỉ tự dọn stale lock khi PID không còn tồn tại; nếu PID còn sống thì phải
   fail-closed để không chạy trùng. Cơ chế này đã được thêm vào
   `scripts/run_teacher_full_research_supervisor.ps1` và đã được kiểm chứng khi
   phục hồi PID cũ `8496`.
6. Chỉ sau khi sidecar hợp lệ mới chạy arm support-2 trên ít nhất ba cửa sổ
   OOS độc lập bằng Model 4/real ticks. Main train, champion, candidate và
   baseline của cùng một arm vẫn chạy tuần tự; chỉ các cửa sổ đã cô lập hoàn
   toàn mới được song song. Có thể để
   `run_teacher_paired_oos_supervisor.ps1` chờ sẵn geometry để bước này tự
   khởi chạy đúng thứ tự:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_teacher_paired_oos_supervisor.ps1 `
     -GeometryCheckpoint outputs/bottom_top_teacher_imitation_v26_sl250_16w_20260921/value_candidate_v26_jan2025_parity_v2_sl250_full_teacher_geometry_v1.pkl `
     -BaselineCheckpoint outputs/seven_choice_teacher_causal_v17_20260917/value_candidate_v26_jan2025_parity_v2_sl250.pkl `
     -Output outputs/bottom_top_teacher_imitation_v26_sl250_16w_20260921/paired_oos_v1
   ```
7. Chỉ khi candidate frozen có đủ lifecycle, LCB dương, DD đạt, terminal-flat,
   reconciliation/feature parity đạt và paired baseline không suy giảm mới
   được xem xét promotion. Chọn checkpoint bằng gate và bằng chứng đa cửa sổ,
   không chọn bằng main PnL cao nhất.

8. Kiểm chứng thực tế sau patch mentor (2026-09-22): v26 đã được pause an toàn
   ở `completed_epochs=13`, epoch 14 tại `windows_done=232/540`; supervisor
   cũ đã dừng và không còn worker tiếp tục tự resume. Smoke v27 dùng cùng
   source checkpoint/seed, nhưng dataset schema9 nhỏ 7 ngày `sl150` chỉ để
   kiểm tra runtime, không dùng làm bằng chứng LCB.

   - Bản smoke trước patch: epoch 2 có `entry_mass_recovery=true` nhưng
     `validation_recovery=false` vì `low_precision=true`, nên controller tiếp
     tục `lower_false_entry_penalty+...suppress_false_entries`; đây là lỗi
     self-reinforcing: tín hiệu đã quá thưa lại bị phạt thêm.
   - Patch đã đổi ưu tiên: `entry_mass_recovery` được phép thắng
     `low_precision` khi không có `over_entry`; stage chuyển thành
     `recover_entry_recall`. Thêm regression test cho trường hợp entry rate
     cực thấp nhưng precision thấp do chỉ còn vài dự đoán.
   - Chạy thật sau patch đến epoch 2 cho cùng dataset/seed: label entry rate
     `0.136929`, predicted `0.006916`, ratio `0.050505`; mentor phát đúng
     `validation_recovery=true`, stage `recover_entry_recall`, giảm penalty
     `0.50 -> 0.40`, giữ hard-negative `1.00`, tăng ranking weight
     `0.50 -> 0.55`. Targeted test đạt `4 passed`, compile đạt.
   - Tuy vậy final validation gate vẫn reject run: BA `0.501845`, MCC
     `0.01531`, entry recall `0.01010`, BUY recall `0.02222`, SELL recall
     `0.0`, cả hai operating point đều ineligible. Không có final candidate,
     không chạy geometry/paired OOS/MT5. Kết luận hiện tại: cơ chế mentor đã
     sửa đúng hướng điều khiển collapse, nhưng hiệu quả giao dịch và LCB vẫn
   chưa được chứng minh; cần tiếp tục sửa cơ chế học/reward hoặc dữ liệu,
   không được nới gate.

   - Full lineage v27 đang chạy riêng trên dataset schema9 16 tuần, source
     hash/config hash cố định và checkpoint schema 2. Sau epoch 2, validation
     BA đạt `0.566989`, entry recall `0.226897`, MCC `0.141665`, predicted
     entry rate `0.109621` so với label rate `0.124667`; so với epoch 1,
     BA và recall đã hồi phục. BUY operating point tạm đủ điều kiện nhưng
     SELL precision chỉ `0.226865` và SELL operating point vẫn ineligible.
     Mentor vì vậy chọn `recover_entry_recall`, giảm false-entry penalty kế
     tiếp `0.50 -> 0.40`, tăng sell scale `1.00 -> 1.05` và ranking weight
     `0.50 -> 0.55`.
   - Đây chỉ là bằng chứng imitation/validation. Tại thời điểm ghi nhận,
   action report, geometry report và paired OOS report chưa tồn tại;
   checkpoint vẫn `promotable=false`, nên chưa có kết luận LCB hay quyền
   triển khai MT5. Replay validation epoch 2 đã hoàn tất `656/656`; worker
   còn sống và đang tiếp tục lineage ở epoch 3.

### 8.1. MT5 settlement gate v26 ngày 2026-09-22

- Lần gọi đầu tiên bị fail-closed ở preflight, chưa chạy tester, vì dùng
  `configs/he_chuyen_gia_mot_context_mot_policy.yaml` với `pending_expiry=60`
  và `max_holding=0`, trong khi sidecar v26 yêu cầu contract `5/48`. Lần chạy
  lại dùng đúng `configs/experiments/edge_discovery_v193_value_oos_fixed_rr2_sl250.yaml`;
  runtime contract khớp đầy đủ: pending `5`, holding `48`, SL `2.5 ATR`,
  TP `2.0R`, feature schema hash khớp.
- Run research-only dùng checkpoint v26 load/reload được, profile
  `mt5_reward_ablation_feb_settlement_only_low_disk.yaml`, Model 1, seed 0,
  cửa sổ `2025.02.05–2025.02.06`, runner clone riêng. Imitation sidecar
  không hiện hữu nên bypass chỉ được ghi là `research_only_unverified_imitation_seed`,
  `promotable=false`; source checkpoint/config không đổi hash.
- Frozen policy đã chạy đủ tester: `451` lifecycle đóng, `reconciled=true`,
  report net `-1607.47`, closed-ledger net `-1729.50`, win rate `35.48%`,
  PF `0.8603`, mean `-3.8348 USD/position`, policy-origin expectancy
  `-0.05364R`, policy-origin LCB `-0.15216R`, DD `37.51%`, terminal còn `7`
  vị thế mở; feature audit hợp lệ, execution failure `0`.
- `ppo_updates=0` đúng vì đây là frozen gate. Do LCB cuối âm, wrapper đã chặn
  cả `full_online` và `reasoning_off`; không tạo candidate MT5 để promote.
  Snapshot giữa run từng dương nhưng không được dùng làm kết luận. Đây là
  bằng chứng trực tiếp rằng tín hiệu tốt ngắn hạn không bền trên toàn cửa sổ.
- Artifact và report: `outputs/mt5_train_v26_settlement_gate_20260922_v2/`.
  Kết luận: cơ chế reward/ledger và gate MT5 đang hoạt động, nhưng checkpoint
  v26 không có edge ổn định trên cửa sổ này; cần chờ v27 action lineage hoàn
  tất rồi lặp lại cùng contract trên nhiều cửa sổ độc lập. Không được bật
  online learner chỉ vì một đoạn equity curve hoặc snapshot LCB dương.
- Đã sửa một blocker triển khai: `scripts/run_tensor_rule_v186_candidate.py`
  trước đây chỉ nhận Model `0/1/2`, trong khi OOS/promotion yêu cầu Model 4
  (Every tick based on real ticks). Parser nay dùng hằng số
  `MT5_MODEL_REAL_TICKS`, nhận Model 4 và có regression test; compile và test
  parser pass. Full test file vẫn vướng quyền truy cập thư mục pytest temp của
  Windows, không phải lỗi logic của patch.

### 8.2. Treatment LCB sau v27, không lặp lại arm đã thất bại

- Không chọn lại directional floor đơn độc: run83 đã xác nhận gradient floor
  chạy đúng nhưng main LCB `-0.1947R`, frozen candidate LCB `-0.1723R` và
  paired gate reject. Vì vậy floor chỉ là regularizer hỗ trợ, không phải cơ
  chế tạo edge.
- Không chọn theo các snapshot dương của probe: run71/run68 từng có LCB dương
  trên một cửa sổ, nhưng cửa sổ frozen/độc lập vẫn âm hoặc chưa terminal-flat;
  các run đó không phải promotion evidence.
- Treatment tiếp theo sau khi v27 hoàn tất là profile
  `configs/experiments/mt5_reward_ablation_feb_expected_value_causal_net_r_live_action_veto_bidir_action_only_stable_low_disk.yaml`,
  chạy bằng Model 4 và cùng checkpoint v27. Profile này giữ settlement-net-R,
  bidirectional ranking, online update scale `0.25`, entry mass floor thấp và
  live veto có provenance; wrapper vẫn chạy frozen trước, chỉ bật online khi
  frozen LCB cuối cửa sổ dương.
- Mỗi treatment phải chạy tối thiểu ba cửa sổ độc lập, ghi riêng frozen,
  online, baseline, ledger và real-tick marker. Tiêu chí giữ candidate là
  pooled policy LCB `>=0`, day-bootstrap LCB `>=0`, terminal-flat, DD trong
  ngưỡng và không regression baseline; thiếu một điều kiện thì rollback về
  seed, không tiếp tục học.

### 8.3. Chế độ teacher trực tiếp trong MT5

- Profile cũ đặt `five_imitation_freeze_updates=0`,
  `five_imitation_anchor_updates=0` và `five_imitation_anchor_weight=0`;
  vì vậy checkpoint teacher nếu được nạp cũng không được giữ quyền trên action
  path khi PPO bắt đầu cập nhật. Đây không phải cơ chế “teacher dạy trực tiếp”.
- Profile mới
  `configs/experiments/mt5_teacher_direct_retention_v1_low_disk.yaml` giữ
  teacher action path cố định tối đa `4096` optimizer updates, neo teacher tới
  update `16384` với anchor weight `2.0`, tắt uniform exploration và hạ
  `mt5_online_update_scale` còn `0.05`. Delayed entry-preference labels vẫn
  bật như lớp kiểm tra phụ, không được dùng để thay thế teacher checkpoint.
- Profile này chỉ có tác dụng khi checkpoint có imitation sidecar hợp lệ;
  checkpoint v26 hiện tại không có sidecar nên phải giữ `research-only`, không
  được giả vờ là đã được teacher huấn luyện. V27 full lineage mới là nguồn
  teacher trực tiếp đầu tiên đủ provenance.
- MT5 vẫn chạy frozen gate trước. Nếu LCB frozen âm thì không cho PPO/online
  arm tiếp tục; “teacher trực tiếp” không đồng nghĩa cho phép học từ tín hiệu
  kinh tế xấu. Kết quả phải ghi rõ teacher retention stats, teacher update
  count, anchor distance và unique-position LCB.
- Quy trình chờ và khởi chạy đã được đóng gói trong
  `scripts/run_mt5_teacher_direct_supervisor.ps1`; script từ chối resume file,
  sidecar thiếu gate hoặc output đã tồn tại, sau đó gọi wrapper MT5 tuần tự.

## 9. Kế hoạch triển khai tiếp theo, có tiêu chí dừng

### Pha A — khóa provenance và chạy lại action lineage

1. Giữ v26 ở trạng thái pause, không resume bằng source code mới. Tạo output
   lineage v27 riêng với cùng source checkpoint, dataset schema9 16 tuần,
   seed, sequence length và learning-rate để phép so sánh chỉ thay đổi mentor
   recovery patch.
2. Chạy một worker duy nhất có checkpoint giữa epoch. Nếu worker terminal trước
   final checkpoint thì supervisor chỉ resume từ artifact cùng contract; không
   để supervisor tự lặp vô hạn sau lỗi validation gate.
3. Chỉ chấp nhận action artifact nếu đồng thời đạt validation gate,
   autonomous self-fed gate, role-aware history, reload/hash và report schema
   hiện hành. Nếu fail bất kỳ mục nào thì dừng ở action, không chạy geometry.

### Pha B — geometry và đánh giá kinh tế độc lập

1. Chỉ sau Pha A mới chạy geometry-only trên output khác; geometry không được
   sửa lại action checkpoint và không được dùng hidden cache cũ nếu parity probe
   không đạt.
2. Paired OOS chạy tối thiểu ba cửa sổ độc lập bằng real ticks/Model 4. Mỗi
   worker có portable MT5, bridge, report và ledger riêng; main train,
   champion, candidate và baseline của cùng arm vẫn tuần tự.
3. Mỗi lifecycle phải có owner/trace/position ticket, đóng terminal, reconcile
   đúng và không duplicate. Tính cả pooled policy LCB và day-bootstrap LCB.

### Pha C — tiêu chí LCB và rollback

Candidate chỉ được chuyển bước khi:

- pooled policy LCB `>= 0R`;
- day-bootstrap LCB `>= 0R` trên ít nhất 3 ngày/cửa sổ độc lập;
- không regress baseline quá `0.02R` ở pooled/day LCB;
- không có negative scope nghiêm trọng, terminal flat, feature parity và
  reconciliation đều đạt;
- sample lifecycle đủ lớn, không tăng LCB nhờ giảm trade về gần HOLD-only;
- candidate reload lại cho cùng decision/action mix và cùng ledger accounting.

Nếu fail: giữ source seed làm champion, lưu nguyên report/reason code và chỉ
đổi đúng một biến trong arm tiếp theo. Không chọn theo main PnL, equity curve,
PF đơn lẻ hoặc subgroup dương.

### Pha D — triển khai MT5 an toàn

1. Candidate qua paired OOS chỉ được đóng gói research/shadow trước; chạy shadow
   trên MT5 với execution thật nhưng không gửi lệnh, theo dõi decision
   divergence, HOLD/entry mass, lifecycle reconciliation và LCB theo ngày.
2. Chỉ chuyển canary khi shadow giữ gate trong cửa sổ mới hoàn toàn; canary có
   risk cap thấp hơn production, rollback về source checkpoint bằng một thao
   tác, và không cho online learner tự ghi đè artifact frozen.
3. Production chỉ dùng artifact có manifest/hash, config hash, real-tick report,
   paired baseline và rollback pointer. Mọi LCB âm hoặc accounting mismatch
   đều tự động veto promotion.

### Các arm được chạy theo thứ tự

| Arm | Mục đích | Nếu fail thì làm gì |
|---|---|---|
| v27 mentor recovery | kiểm tra giữ entry mass/role sau nhiều epoch | không chạy MT5, phân tích history |
| v27 static mentor đối chứng | tách tác động mentor khỏi kiến trúc | giữ arm có validation ổn định hơn |
| settlement-net-R MT5 | đo tác động economic lifecycle thật | reject nếu pooled/day LCB âm |
| shadow/canary | kiểm tra drift sau freeze | rollback source, không promote |

Không chạy đồng thời hai arm ghi cùng runner/ledger. Có thể song song parse/audit
read-only hoặc các OOS window đã clone portable hoàn toàn.

Supervisor action cũng đã được fail-closed thêm: nếu stderr chứa
`validation gate failed` hoặc `autonomous validation gate failed`, nó ghi
`SUPERVISOR_REJECT_GATE` và không retry cùng resume. Chỉ lỗi worker không phải
gate mới được retry/resume; như vậy một lineage bị reject không tiêu tốn lại
toàn bộ CPU mà không tạo thêm bằng chứng.

## 10. Cập nhật vận hành ngày 2026-09-22

### 10.1. Trạng thái v27 hiện tại

- Teacher v27 đang chạy ở `attempt_04`, nạp đúng resume schema 2 cùng source
  hash, dataset hash, config hash và seed đã ghi trong contract. Epoch 5 đã
  hoàn tất; validation BA `0.665369`, precision `0.466233`, recall
  `0.395172`, MCC `0.355410`, predicted-entry rate `0.105666` so với label
  rate `0.124667`; BUY/SELL operating-point đều eligible. Đây chỉ là
  imitation/validation evidence, chưa phải economic LCB.
- Checkpoint vận hành sau epoch 5 ghi `resume_epoch=6`,
  `completed_epochs=5`, `promotable=false`; checkpoint gần nhất có
  `windows_done=128/540`, resume SHA-256 bắt đầu bằng
  `85a3b32262ea0c91...`, còn worker đã tiếp tục tới `136/540` recurrent
  windows và hiện đã tới `328/540` (`60.7407%`). Final checkpoint và
  `.imitation.json` chưa tồn tại,
  nên teacher-direct MT5 promotion bridge vẫn đúng trạng thái `WAIT_TEACHER`.
- Epoch 5 mất khoảng `3046.6s`, cho thấy 80 epoch là lịch chạy dài. Không
  được coi resume checkpoint là candidate cuối và không được nới bridge gate
  chỉ để rút ngắn thời gian. Nếu cần quan sát sớm, dùng smoke research-only
  riêng; promotion vẫn chờ final sidecar.

### 10.2. MT5 smoke tách khỏi promotion

- Đã chạy một smoke giới hạn bằng Model 4 trên checkpoint v26 gốc, profile
  teacher-retention low-disk, cửa sổ `2025.02.05..06`,
  `--allow-unverified-imitation-seed`, `--metrics-only` và giới hạn wall-clock
  180 giây. Artifact ở
  `outputs/mt5_bridge_smoke_baseline_v26_model4_20260922_200packets_v2/`.
- Smoke đã chứng minh đường đi MT5 thật: bridge packet, request/action,
  entry-preference, execution trace, diagnostics và manifest đều được ghi;
  tester tạo 69 lifecycle đóng trước khi run bị cắt bởi
  `max_wall_sec=180` tại `served=96`. Diagnostics có LCB tạm dương trên mẫu
  nhỏ, nhưng đây là baseline chưa có imitation sidecar, chưa đủ 200 vị thế,
  run bị interrupt và `promotable=false`; không được đưa vào pooled LCB hoặc
  promotion.
- Bài kiểm tra smoke phát hiện một blocker cấu hình thật: profile đặt
  `five_behavior_uniform_mix=0.0` nhưng còn min-mix exploration dương. Đã thêm
  `five_live_owner_exploration_min_mix=0.0` vào
  `configs/experiments/mt5_teacher_direct_retention_v1_low_disk.yaml`, thêm
  regression assertion và test profile hiện đạt `1 passed`. Teacher-direct
  bridge sẽ dùng profile đã sửa khi sidecar xuất hiện.
- Regression test profile được chạy lại sau checkpoint v27 mới nhất bằng
  `pytest -q tests/test_mt5_teacher_direct_profile.py` với workspace-local
  basetemp: kết quả `1 passed` (chỉ còn cảnh báo Windows không tạo được cache
  pytest cũ, không phải lỗi logic). Vì vậy blocker min-mix không còn là lý do
  chặn MT5; blocker hiện hành duy nhất vẫn là teacher chưa có final sidecar.

### 10.3. Quy trình triển khai được chốt

1. Chạy teacher lineage và smoke/parse/audit trên namespace khác nhau; smoke
   chỉ xác minh hạ tầng và accounting, không tạo quyền promotion.
2. Khi v27 final sidecar xuất hiện, bridge kiểm tra reload, validation gate,
   autonomous gate và resume đã biến mất; sau đó mới chạy Model 4 full window.
3. MT5 full run phải hoàn thành frozen policy trước. Chỉ khi unique-position
   LCB frozen dương, đủ sample, terminal-flat và reconciliation pass mới cho
   online arm; nếu không thì giữ `ppo_updates=0` và kết luận fail-closed.
4. Sau đó lặp tối thiểu ba cửa sổ độc lập, tính pooled policy LCB và
   day-bootstrap LCB, so với baseline cùng checkpoint/date/model/seed. Chỉ
   candidate đạt tất cả economic/OOS gate mới được shadow; chưa có bằng chứng
   hiện tại nào đủ để chuyển sang canary hoặc production.

### 10.4. Sự cố worker và resume an toàn

- `attempt_03` đã dừng bất thường tại epoch 6, `88/540` windows. `stderr.log`
  rỗng, không có `SUPERVISOR_REJECT_GATE`, final checkpoint và sidecar đều
  thiếu; do đó đây không phải validation rejection và không được coi là pass.
- Resume schema 2 vẫn hợp lệ, giữ source/dataset/config hash, optimizer, RNG
  và cursor ở boundary epoch 6. Supervisor đã khởi động `attempt_04` với
  đúng toàn bộ tham số cũ; preflight và `action_resume_artifact_loaded` pass,
  worker mới đang sống, đã ghi checkpoint tại `128/540` và tiếp tục tới
  `328/540` windows của epoch 6. Không đổi output lineage, không reset
  optimizer và không tạo candidate từ resume.
- Teacher-direct bridge đã được khởi động lại ở trạng thái chờ sidecar. Khi
  worker có sự cố, thứ tự xử lý là kiểm tra PID/lock, stderr, resume hash và
  final/sidecar; chỉ khi resume contract còn đúng mới resume. Nếu sidecar đã
  xuất hiện mà resume còn tồn tại, bridge vẫn từ chối MT5.

### 10.5. Quy tắc health-check khi train kéo dài

- Không xem log đứng yên trong một vài chu kỳ là worker chết. Health-check phải
  kiểm tra đồng thời PID còn tồn tại, `Responding=true`, CPU time còn tăng,
  `stderr` không có lỗi và lock vẫn thuộc đúng supervisor.
- Checkpoint vận hành được đánh giá theo boundary `CheckpointEveryWindows`
  (hiện là `128` windows), không theo một timeout cứng; độ dài một batch có
  thể thay đổi do recurrent window và replay/validation.
- Chỉ được resume khi worker đã terminal hoặc PID không còn, resume file còn
  hash/byte-size khớp metadata và không có `SUPERVISOR_REJECT_GATE`. Nếu gate
  bị reject thì dừng fail-closed, không retry cùng artifact.
- Với trạng thái live hiện tại, worker còn phản hồi và CPU tiếp tục tăng dù
  checkpoint gần nhất vẫn ở `128/540`; vì vậy đây là trạng thái đang tính toán,
  chưa đủ bằng chứng để can thiệp hoặc chuyển sang lineage khác.
