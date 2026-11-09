# V155 — chuẩn hóa cấu trúc và representation có điều kiện

## Phạm vi

Đây là candidate opt-in trong `configs/he_chuyen_gia_mot_context_mot_policy.yaml`.
Artifact seed v13 vẫn dùng `input_normalization_mode: raw`; không có trọng số,
normalizer hay head V155 nào được đưa vào đường inference của seed.

## Contract bắt buộc cho hidden blocks

Mọi đường Direct-RNN (context-expert và `SelfGrowingPolicy`) bắt buộc dùng
`rms_v1` hoặc `layernorm_v1` ở từng hidden block và dropout training-only khác
0 ở từng block. Bộ rate chuẩn sáu lớp là `0.05, 0.10, 0.10, 0.05, 0.10, 0.10`;
config được phép đặt schedule dương khác, và khi tăng độ sâu rate cuối được
lặp lại để lớp mới không vô tình không regularize.

Không profile, biến môi trường, checkpoint cũ hay thay đổi tay thuộc tính có
thể tắt contract này: parser config từ chối `none`, `false` hoặc rate `0`, còn
load/copy/forward tự nâng artifact cũ lên contract. Điều này không sửa file
seed gốc; nó chỉ chuẩn hóa runtime copy trước train hoặc evaluation.

`training=True` luôn lấy dropout mask xác định theo seed/step. Frozen replay
và inference dùng `training=False` nên **không lấy mask ngẫu nhiên**, để quyết
định lặp lại được; đây không phải tắt dropout trong kiến trúc hay trong config.
Khi `direct_rnn_full_rl_enabled` và learning MT5 cùng bật, entry forward cache
**bắt buộc** chạy `training=True`: chính masked cache đó là đối tượng được
backprop ở lifecycle batch. Telemetry phải cho thấy
`hidden_dropout_training_active=true` và `hidden_dropout_steps_advanced > 0` ở
mỗi module. Nếu một training forward không tăng step, runtime dừng bằng lỗi
contract thay vì tiếp tục học từ hidden không dropout.

## Router regime dùng raw bridge, policy dùng normalized input

`structure_aware_v4` được áp dụng tại ranh giới bridge chỉ cho encoder
Direct-RNN. Regime router nhận sidecar `raw_inputs` của cùng request và tính
M15 EMA/ATR, M15/H1 ATR ratio, volatility và chop trên **native bridge values**.
Không được đưa vector đã chuẩn hoá vào `MarketRegimeClassifier`: ratio bình
thường `1` bị map gần `0`, rồi classifier floor thành `0.1/0.1=1` và có thể
fabricate `VOLATILE_EXPANSION` liên tục. Decision telemetry ghi
`regime_input_source` cùng `regime_input_audit.raw_bridge` và
`regime_input_audit.policy_input`; hai view này chỉ phục vụ audit, không đi vào
memory key, action head hoặc Full-RL loss. Nếu sidecar không có (unit test hay
caller legacy), router fallback policy input và ghi rõ `policy_input_fallback`.

## Input `structure_aware_v4`

V4 giữ normalizer theo nhóm của V2, nhưng thay map cho những input đã chứng
minh bão hòa trên telemetry.

- `chop_index` index 93: EA thực tế xuất `[0,1]`, nên V4 dùng `2x - 1`.
  V2/V3 vẫn giữ hành vi cũ để replay artifact cũ không bị thay đổi.
- Khoảng cách ATR dương `100..103, 118, 119`: dùng
  `asinh(x / 2) / asinh(10)` với `x` clip `[0,20]`. Map này đơn điệu,
  không fitted và vẫn phân biệt được 4 ATR với 12 ATR.
- MTF volatility ratio `82, 86, 90`: dùng `tanh(log(ratio) / 0.35)`.
  Ratio bằng 1 là trạng thái bình thường nên map đúng về 0; giãn và co biến
  động có dấu đối xứng.
- Các MTF signed ATR còn lại `80,81,83,84,85,87,88,89,91`: dùng map asinh
  signed thay vì tanh quá sớm.

Python và `SelfGrowing_MT5_Trainer_EA.mq5` có cùng công thức. Không có mean,
variance, percentile hay thống kê rolling nào được fit từ train/OOS, nên
frozen replay không mang trạng thái regime tương lai.

`features.mtf_scaling_mode` cũng là một phần của contract này. Candidate V155
phải truyền nguyên giá trị `native_tf_atr_v2` qua cả ba đường: YAML → tester
set của EA, fingerprint/schema, và Python numerical-parity replay. Không được
rebuild `FeatureConfig` thiếu trường này, vì default legacy sẽ khiến H1/M15 lại
chia theo ATR M1 trong khi config nói dùng ATR khung riêng.

Position snapshot cũng append `initial_risk_distance_price` sau các owner
fields cũ. Input unrealized-R trong EA luôn dùng mẫu số risk ban đầu, vì vậy
Python parity không được suy nó từ SL hiện tại sau trailing/breakeven. Format
này append-only: reader cũ vẫn dùng được các field 0..20.

EA chỉ gửi request cho policy khi buffer indicator M1/M5/M15/H1 đã đủ lookback
và ATR completed-bar dương. Trong warm-up, trailing/SL safety vẫn chạy, nhưng
không có inference, entry hay Full-RL credit từ vector fallback `ATR=epsilon`.

## Conditional representation

Mỗi `ModuleState` có head huấn luyện-only:

```text
output_hidden ──> conditioned_quality_w[context, action, side] ──> quality logit
```

Bucket context là `TANG_MANH`, `TANG_YEU`, `GIAM_MANH`, `GIAM_YEU`,
`RANGING`, `VOLATILE`, hoặc `OTHER`; action là action bridge 0..11; side là
`FLAT/BUY/SELL`. Nhãn chỉ được tạo khi lifecycle entry đã đóng theo canonical
settlement:

- `R >= min_abs_r` là quality tốt;
- `R <= -min_abs_r` là quality xấu;
- vùng gần 0 bị bỏ qua để không biến nhiễu/lệch spread nhỏ thành nhãn.

Loss là weighted binary cross entropy. Head này không là output bridge, không
thay action, không là mask/gate và không chạy lúc inference. Gradient của nó
vào `output_hidden`, sau đó backprop đúng cache entry qua sáu hidden blocks.
`conditioned_quality_w/b` cùng `w_in`, recurrent stack, value head và action
rows chỉ được `apply_rnn_adamw_cosine(..., source="full_lifecycle_rl")` cập
nhật. Full-RL từ chối mọi writer cũ.

Khi auxiliary shared gradient ngược actor gradient, `pcgrad_v1` loại đúng
thành phần chống actor ở encoder. Weight/bias riêng của quality head vẫn nhận
nhãn; không có gradient của head bị đưa vào action decoder trực tiếp.

## Telemetry phải theo dõi

Event `direct_rnn_full_rl_batch` ghi:

- `conditioned_representation_eligible_count`, `positive_count`, `loss`;
- `conditions`: hỗ trợ thực tế theo `context|action|side`;
- shared cosine/dot/norm, cờ `shared_projection_applied`;
- `head_delta_norms` để xác nhận AdamW thật sự cập nhật head.

Một candidate chưa được coi là học chỉ vì head có norm khác 0. Cần frozen
replay checkpoint đầu/cuối trên cùng path, và so sánh decision divergence,
entry PF/expectancy/DD cùng các bucket `MARKET_BUY` theo location/volatility.

## Audit lifecycle và fitness

Close do broker thực hiện tại `INITIAL_SL`, `TRAILING_SL` hay `TP` có thể có
`source=unknown`: đó là nguồn của **event đóng**, không phải nguồn của entry.
Vì vậy audit sau tester phân loại trade policy-origin bằng
`position_owner_trace_id` bất biến đã được ghi từ lúc entry. Audit này chỉ đọc
`execution_summary.lifecycles` đã final-close, tính expectancy/LCB theo R và
gắn `audit_source=execution_ledger_immutable_owner_v1`.

Nó không ghi vào contextual expectancy, không tạo action bias, LCB mask,
selector, risk gate hay loss. Khi contextual-expectancy report không có mẫu
(đúng với Full-RL nơi đường cũ bị tắt), fitness dùng audit read-only này để
không báo sai `0 lifecycle` trong khi canonical settlement thực tế đã có
credit. Đây là sửa observability/fitness, không là writer thứ hai cho policy.

Manifest mới không còn dùng nhãn mơ hồ `lifecycle_credits`. Nó ghi riêng:
`live_lifecycle_settlements` (lifecycle đã được route trong lúc tester chạy),
`live_lifecycle_updates` (hook policy chấp nhận update),
`post_run_retry_lifecycle_updates`, `wall_timeout_lifecycle_updates` và tổng
`lifecycle_updates`. Vì vậy số retry cuối run không còn bị hiểu nhầm là toàn
bộ quá trình học.
