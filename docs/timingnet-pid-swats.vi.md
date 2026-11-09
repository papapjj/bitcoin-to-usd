# TimingNet PID-SWATS (candidate-only)

## Phạm vi

PID-SWATS chỉ sở hữu `TimingNetwork`. Các cập nhật Policy/M1/graph vẫn dùng
đường bounded update hiện tại. `LearningIntegrityConfig` mặc định tắt cả hai
cờ để seed v13 không bị đổi optimizer khi migrate checkpoint cũ.

## Candidate tách riêng M15

`scripts/train_timingnet_isolated.py` là đường train TimingNet độc lập: chỉ
đọc market/timing/Wyckoff từ CSV feature reference và một snapshot
`self_growing_feature_audit.csv` MT5 cuối cùng làm OOS. Nó không nạp, ghi hay
update artifact Direct-RNN; report bắt buộc ghi `policy_touched=false` và
`lifecycle_pnl_used=false`. Reference được chia theo thời gian 70% train,
15% calibration, 15% validation; seed/LR/threshold chỉ chọn bằng validation.
Audit MT5 được copy thành file bất biến chỉ **sau** khi khóa candidate.

Mỗi sample của trainer được materialize thành tensor nhân quả
`[32 M15, 78 feature]`, oldest-to-newest. Chỉ nến M15 đã đóng được ghi vào
tensor; 31 sample warm-up đầu không được dùng làm supervised target để padding
0 không trở thành shortcut. Runtime và frozen evaluator dùng cùng rolling
window 32 bước. Trainer dùng truncated-BPTT batch, một AdamW/cosine update
cho mỗi batch tensor thay vì writer online cho từng tick/horizon. Head h3 học trực tiếp xác suất
nhãn đã chọn (`context_change_v1` hoặc `trend_support_deterioration_v1`) và
calibration isotonic chỉ fit trên split calibration. `deceleration` hiện là
telemetry: không được dùng để đổi action khi chưa có edge OOS. Candidate có
thể thêm adapter 12 feature thời gian nhân quả; khi đó capsule ghi rõ input
90 chiều và adapter đi kèm, nên không thể nạp nhầm vào runtime 78 chiều.

Promotion h3 cần cùng lúc: ít nhất 180 mẫu, 8 positive, Brier cải thiện 5%,
AUC 0.60, support 8 và precision 0.50 trên MT5 frozen holdout. Không hạ các
ngưỡng này để ép TimingNet thành entry gate.

## Nhãn deceleration độc lập

Head `deceleration_probability` là một task tự giám sát nhân quả: tại quyết
định `t`, mạng dự báo liệu trị tuyệt đối của chính `directional_score` router
có giảm tối thiểu `timing_network_deceleration_min_drop` về 0 sau
`timing_network_deceleration_horizon` quyết định sau hay không. Target chỉ
được settle khi các observation sau đã có; frozen replay truyền
`learn_deceleration=false` nên không thay checkpoint. Head dùng binary
cross-entropy trên logit (`target - probability`), không dùng thêm hệ số
`p(1-p)` của squared-sigmoid vì head khởi tạo bằng 0 sẽ gần như kẹt ở 0.5.
Nó có learning rate riêng và L2 shrinkage; encoder chỉ nhận phần gradient
auxiliary đã co để chống overfit. Cập nhật auxiliary đi qua **cùng writer
AdamW/SWATS** với TimingNet (không còn `_apply_flat_update` thủ công), nên
shadow Adam tiếp tục phản ánh gradient hiện tại. Tuy vậy nó không advance
projection, switch hay PID: các controller này chỉ nhận alignment và realized
R của lifecycle thật.

## Chu kỳ lifecycle

## Dự báo chuyển regime và entry block

Head transition đa-horizon dùng cùng feature timing, nhưng nhãn router là
**context của nến M15 đã đóng**. Tại thời điểm `t`, trace lưu state M15 trực tiếp
và xác suất đổi ở từng horizon; khi `t+H` tới, target là state M15 thật tại thời
điểm đó. Router không còn candidate rule/hysteresis, còn cache M15 ngăn M1 tick
nhiễu tạo nhãn chuyển regime.

Chỉ xác suất ở horizon gần nhất là tín hiệu hành động. Sau khi nhãn trễ settle,
TimingNet lưu cặp `(p_t, changed_t+H)` trong cửa sổ 4.096 mẫu và tính Brier,
support cùng precision cho đúng vùng `p >= entry_threshold`. Facade chỉ biến
một entry mới thành `HOLD` khi tất cả điều kiện cấu hình cùng đạt: số update,
số calibration mẫu, support high-confidence, precision tối thiểu và Brier tối
đa. Điều kiện thiếu bất kỳ phần nào thì giữ entry nguyên trạng; position
management không đi qua gate này. Đây là một hard gate candidate-only, không
phải score hay action delta từ expert riêng lẻ.

1. Trace tại thời điểm quyết định được giữ nguyên đến khi lifecycle có `R` thật.
2. Gradient target của TimingNet được clip như trước; gradient chuẩn của SWATS
   là `g = -gradient_update_direction`.
3. Shadow Adam cập nhật `m`, `v` và bias-correction ở mỗi lifecycle hợp lệ.
   Candidate v146 có thể dùng AdamW decoupled weight decay cho các weight
   lifecycle (không decay head BCE deceleration). Cosine annealing chỉ nhân
   `eta_adam` theo **số lifecycle hợp lệ**, không theo MT5 tick.
4. Projection dùng:

   ```text
   p_adam = -eta_adam * m_hat / (sqrt(v_hat) + epsilon)
   gamma  = ||p_adam||² / max(-p_adam · g, epsilon)
   alignment = cos(p_adam, -g)
   ```

   `gamma` được EMA và chỉ chuyển ADAM → SGD khi projection ổn định đủ số
   bước, đồng thời đạt ngưỡng alignment. `eta_sgd` khởi tạo bằng
   `0.90 * gamma_hat`, rồi bị chặn trong `[eta_min, eta_max]`.
5. Sau switch, trọng số thật dùng SGD; shadow Adam vẫn cập nhật m/v để giữ
   quỹ đạo tham chiếu hiện tại. Cosine dừng tại đây: PID là actuator duy nhất
   của `eta_sgd`, tránh hai scheduler điều khiển cùng một learning rate.
6. Nếu PID bật, một actuator duy nhất điều khiển `eta_sgd` trong log-space:
   `e = 0.70 * e_alignment + 0.30 * e_R`. `R` dùng raw risk-normalized
   lifecycle return, EMA và volatility floor `0.25`; R dương không tự tăng
   learning rate. Integral có anti-windup, derivative có EMA, deadband,
   cooldown và hysteresis recovery.
7. Alignment/R xấu kéo dài đưa TimingNet vào `frozen` với `eta_min`; chỉ khi
   có đủ lifecycle tốt mới quay lại SGD. `pid_fallback_to_adam` là lựa chọn
   candidate nếu cần quay về shadow Adam thay vì freeze.

## Checkpoint và telemetry

Checkpoint lưu `shadow_adam_m/v`, step, phase, projection EMA/counters, PID
integral/derivative/EMA, eta, AdamW/cosine fields và safety counters trong
chính `TimingNetwork`, kèm `sequence_length`, rolling tensor M15 và số hàng
hợp lệ. `copy()` và `ensure_compatible()` migrate artifact cũ bằng zero-state
an toàn; bridge EA vẫn giữ 137 cột.

Decision telemetry thêm:

- `timing_network_optimizer_phase`
- `timing_network_shadow_adam_step`
- `timing_network_projection_count`
- `timing_network_switch_count`
- `timing_network_eta_sgd`
- `timing_network_pid_alignment_ema`, `timing_network_pid_r_ema`
- `timing_network_adamw_enabled`, `timing_network_adamw_weight_decay`
- `timing_network_cosine_annealing_enabled`, `timing_network_cosine_min_factor`
- `timing_network_deceleration_probability`
- `timing_network_deceleration_update_count`, `timing_network_trend_score`
- `timing_network_regime_transition_probability`, `..._probabilities`, `..._update_count`
- `timing_network_regime_transition_calibration_count`, `..._brier`,
  `..._high_confidence_support`, `..._high_confidence_precision`

Lifecycle event thêm projection/alignment/gamma/PID fields từ kết quả update.
Auxiliary deceleration/regime event có `optimizer_auxiliary=1`, dùng cùng
moment/weight-decay nhưng không có quyền đổi PID/SWATS phase.

## Giới hạn action trong Full-RL

Trong profile Full-RL hiện tại TimingNet chỉ học các task dự báo M15 riêng;
`timing_network_lifecycle_scale=0` khóa P/L lifecycle,
`timing_network_action_scale=0` và response transition bị tắt. Nó là nguồn
telemetry/candidate research, không phải một overlay có thể
đổi BUY/SELL/HOLD sau khi Direct-RNN đã phát logit. Chỉ khi TimingNet được đưa
vào cùng forward cache và loss Full-RL thì mới được phép tác động action trở
lại.

## Runtime profile và candidate riêng

Profile active đặt `timing_network_lifecycle_scale: 0.0`: lifecycle P/L không
được là writer của TimingNet. TimingNet runtime còn lại chỉ ghi/auxiliary M15
telemetry và `timing_network_action_scale=0`, vì vậy candidate không thể đổi
BUY/SELL/HOLD. AdamW/cosine của isolated candidate chạy không cần P/L; SWATS
không được chuyển sang SGD/PID vì không có realised-R phù hợp để điều khiển.

Khi nạp seed legacy, runtime phải hoàn thành `ensure_compatible()` của từng
owner trước rồi mới gọi `configure_optimizer()`. Đây là điều kiện bắt buộc:
owner gốc có thể chưa có object TimingNet trong pickle, còn owner sinh sau đã
có sẵn. Regression kiểm tra cả hai dạng và yêu cầu mọi owner cùng ở pha
`adam` trước khi MT5 bắt đầu chạy.

Luôn đặt output sang file candidate mới; không ghi đè
`outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl`.
Frozen OOS vẫn tắt learning nên không làm thay đổi m/v hay PID state.
