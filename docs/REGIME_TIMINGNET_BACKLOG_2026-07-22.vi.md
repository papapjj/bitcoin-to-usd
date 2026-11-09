# Backlog regime, transition và TimingNet — 2026-07-22

Tài liệu này đóng băng các vấn đề phát hiện từ artifact thật để không bị mất
giữa các lượt train. Không mục nào được coi là đã giải quyết chỉ vì equity của
một đoạn dữ liệu tăng; mỗi mục cần bằng chứng MT5 và kiểm tra frozen replay.

## Nguồn bằng chứng đã lưu

- Telemetry/decision artifact: `outputs/he_chuyen_gia/audit/v13_pending_cancel_fix_20260721_v11`.
- MT5 evaluation manifest: `mt5_v2_oos_arm_train_v122_layernorm_selector/Reports/he_chuyen_gia_mt5_round_0000_evaluation_9102.manifest.json`.
- Feature audit MT5: `mt5_v2_oos_arm_train_v122_layernorm_selector/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_feature_audit.csv`.
- Execution ledger cùng MT5 runner ở thư mục trên.

## Việc còn mở

| Ưu tiên | Trạng thái | Vấn đề | Bằng chứng hiện có | Điều kiện đóng |
|---|---|---|---|---|
| P0 | MỞ | Telemetry ghép nhiều root run trong cùng stream. | `learning_events.jsonl` chứa 4 root run; `scope_paths_by_run` mặc định tắt nên round lồng nhau cùng một run id logic. | Mỗi MT5 trial có root/run id riêng, manifest, decision profile và lifecycle ledger cùng lineage; audit từ chối stream lẫn root. |
| P0 | MỞ | Manifest/report và execution ledger chưa đồng nhất số lifecycle. | Manifest policy 9102 ghi 84 lifecycle nhưng parser ledger hiện thấy 79. | Khóa snapshot artifact trước parse, ghi SHA256/source path/row count và fail gate khi report–ledger lệch. |
| P0 | MỞ | Wall guard có thể ngắt orchestration nhưng để worker Strategy Tester còn sống. | Smoke `v143_timingnet_deceleration_model1_20260722_smoke` bị ngắt sau 1,201.3s; launcher/CLI thoát, `metatester64` cùng runner vẫn CPU-bound, manifest còn `stage=started`, report chưa có SHA. | Khi timeout, dừng worker `metatester64` của đúng runner nhưng không đóng terminal đăng nhập; test tự động phải chứng minh keep-open không để orphan worker. |
| P1 | ĐÃ SỬA CẦN MT5 XÁC NHẬN | `market_regime` YAML chết và MLP tự học nhãn luật của chính nó. | Router live gọi `deterministic_regime`; config `enabled/classifier_hidden/confidence_threshold/learning_rate/regime_window` không được truyền vào classifier. | Đã xóa YAML/MLP/update chết; loader bỏ state cũ. MT5 smoke phải ghi `regime_timeframe=M15` và chứng minh không còn M1 owner churn. |
| P1 | ĐÃ SỬA CẦN MT5 XÁC NHẬN | Context M1 churn quanh threshold, không phải regime shift bền. | 302/683 lần đổi nhãn (44.2%), median dwell 1; 73 chuyển trực tiếp UP↔DOWN, trong khi JSD phân phối theo ngày thấp. | Router chỉ đọc/cache tuple nến M15 đã đóng và đổi trực tiếp, không hysteresis/countdown. Cần báo cáo dwell/churn mới và OOS latency. |
| P1 | ĐANG THỬ NGHIỆM | `CTX_GIAM_MANH → CTX_GIAM_YEU → SELL_LIMIT` là subgroup âm. | n=7, mean R -0.332, one-sided LCB -0.721, PF 0.295; stable strong-bear SELL_LIMIT không cho cùng dấu xấu. | Cần đủ mẫu OOS, so sánh frozen baseline/candidate theo đúng microstate và kiểm tra action divergence. Không tổng quát hóa thành cấm toàn bộ transition. |
| P1 | ĐANG THỬ NGHIỆM | Chứng minh TimingNet thực sự dự báo được transition M15 trước entry. | Head shared router dùng nhãn context M15 trực tiếp ở horizon 3/6/12; không có candidate/countdown làm target. Entry block chỉ mở khi delayed-label calibration đủ support, precision và Brier. | Frozen replay đầu-cuối phải báo calibration theo horizon, precision high-confidence, Brier, số entry block và kết quả những entry bị/không bị block. |
| P2 | MỞ | Volatile subexperts chưa đủ lifecycle để kết luận selector/role có ích. | Trace trước đây chủ yếu route owner trend khi candidate volatile. | Báo cáo support, expectation/LCB, selection rate và counterfactual per volatile subtype; không promote theo aggregate equity. |

## Contract candidate TimingNet hiện tại

Trong `configs/he_chuyen_gia_mot_context_mot_policy.yaml`, toàn bộ direction
prior transition đang tắt. TimingNet deceleration có horizon
3, chỉ label khi |trend score| ban đầu tối thiểu 0.18 và giảm ít nhất 0.10.
Head transition shared dự báo context **M15 trực tiếp** tại horizon 3/6/12. Một
entry mới chỉ thành `HOLD` khi p(transition-3) >= 0.78, có tối thiểu 192 update,
128 calibration mẫu, 12 high-confidence mẫu, precision >=0.75 và Brier <=0.20.
Không có điều kiện nào đạt thì action không đổi; management/close không bị gate.

Đây là candidate-only. Seed v13 giữ cờ tắt và không được ghi đè.

## V152 — representation credit, transition target và observability

Đã triển khai ở code/config, nhưng **chưa có bằng chứng edge MT5**:

- `groupwise_v2` chuẩn hóa input nhân quả, không fit mean/std từ train/OOS.
  Nó nén các tail position/account/execution bằng signed-log, giữ market/timing
  soft-clip và map riêng `chop_index`, `bars_since_*`. v13 vẫn raw.
- Direct-RNN có lifecycle credit vào `w_in` chỉ với entry trace cùng
  `module × selected-action`, đủ batch 6, advantage tuyệt đối tối thiểu 0.15
  và sign coherence tối thiểu 0.55. Batch bất đồng dấu bị discard có audit.
- TimingNet schema V5 thêm head cross-entropy dự báo **context M15 trực tiếp**
  ở horizon 3/6/12; label chỉ settle khi H decision tương lai đã đến.
  Nó **không** dùng PnL/account/position vào input. Gate shared sau selector
  chỉ tạo HOLD cho entry khi p/high-confidence precision/Brier có bằng chứng
  delayed-label; không đảo side và không chặn management.
- Hồ sơ decision direct-RNN dùng xấp xỉ local `w_in × action-margin` trên raw
  input 137 chiều. Không còn hiển thị map controller graph lồng nhau như giá
  trị input bằng 0.

Điều kiện đóng V152: cần MT5 train mới tạo checkpoint candidate riêng, sau đó
so sánh frozen replay seed-v13/candidate về action divergence, transition
calibration theo từng horizon, số `input_lifecycle_encoder_credit`, WR/Exp/LCB
theo expert và drawdown. Không promote chỉ vì số update tăng.

## Candidate v148 — full lifecycle update + AdamW/cosine/PID-SWATS

V146 đã bị dừng trước khi tạo artifact vì audit phát hiện sáu owner gốc của
seed v13 vẫn dùng optimizer legacy: TimingNet của chúng chỉ được khởi tạo ở
compatibility pass sau khi code cũ thử gắn optimizer. V147 là forensic run ở
mức update thấp, đã dừng trước checkpoint. V148 là artifact riêng tiếp theo
từ cùng seed v13, không phải continuation của v145/v146/v147. Trước khi
chạy, runtime bắt buộc migrate rồi gắn cùng contract lên tất cả owner; test
regression và kiểm tra trực tiếp seed xác nhận đủ 10 owner ở pha `adam`.

V148 bật 100% lifecycle credit cho TimingNet và policy/graph, AdamW weight
decay 0.001 và
cosine floor 0.40 trong 32 lifecycle **chỉ ở pha Adam trước switch**. SWATS
chỉ chuyển sau tối thiểu 8 projection hợp lệ và 4 projection ổn định; khi đã
ở SGD, PID là controller learning-rate duy nhất và có fallback về Adam khi
R/alignment xấu liên tiếp. Central facade và toàn bộ owner expert nhận cùng
optimizer contract; head BCE deceleration vẫn tách khỏi AdamW/PID-SWATS.

V148 chưa có kết quả MT5 nên không phải bằng chứng edge, không được promotion
và phải qua frozen OOS/walk-forward giống mọi candidate khác.

## Smoke v143 — không dùng làm candidate

Run `v143_timingnet_deceleration_model1_20260722_smoke` dùng model 1 trên
`2025.01.01–2025.01.07`, seed SHA256
`B52684103C0305A8A4E0A8D066EF2AC63DBADB13050BE675E2F1A3187A426BD4`.
Nó bị wall guard cắt ở 1,201.3 giây sau 395 bridge decision; do đó chỉ tạo
`*.incomplete.pkl` và bị cấm làm input cho OOS/promotion. Lệnh smoke đã ghi
topology đầy đủ (`live_sample_every=1`, `max_responsible_*=0` nghĩa là full),
khiến hơn 360 MB network telemetry và làm test chậm. Đây là lỗi quan sát/
thời lượng, không phải bằng chứng TimingNet xấu hay tốt.

Run dài phải đặt wall guard tắt hoặc lớn hơn thời lượng MT5 đã đo, dùng telemetry
giới hạn (sample >= 25; node/edge/responsible đều cap) và vẫn giữ lifecycle,
feedback, deceleration label/update hoàn chỉnh. `network_development` chỉ sample
snapshot nặng ở feedback; lifecycle và structural change luôn ghi đủ. Seed không
bị thay đổi.

Một attempt 15 ngày kế tiếp đã bị dừng sớm, không được dùng làm candidate: sau
57 update deceleration, xác suất vẫn chỉ dao động quanh 0.50000. Postmortem
cho thấy auxiliary dùng squared-sigmoid với LR hiệu dụng 0.000125; 354 update
của smoke trước chỉ đổi bias khoảng -0.0011. Contract candidate sau đó chuyển
sang BCE logit-gradient, LR head riêng 0.01, encoder scale 0.10 và L2 1e-4.
Đây là sửa tốc độ học của head lạnh, còn frozen/OOS vẫn là bằng chứng bắt buộc
trước khi response LIMIT→STOP được coi là có ích.

## Bằng chứng bắt buộc ở lượt MT5 sau

1. Lưu baseline seed và candidate checkpoint cùng SHA256.
2. Chạy frozen inference trên cùng dữ liệu cho cả hai; ghi decision divergence,
   selection rate, action style và risk pct.
3. Chạy train thật; ghi mỗi lifecycle update, deceleration label/update,
   calibration transition theo horizon, số entry block và R theo microstate.
4. Chỉ đánh giá subgroup sau khi số mẫu đủ; báo expectancy, LCB, profit factor,
   drawdown và confidence interval, không chỉ win rate/equity.
5. Promotion chỉ khi paired OOS không thoái lui và lineage/ledger integrity P0
   đều pass.
