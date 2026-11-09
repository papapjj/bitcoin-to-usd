# Five Reasoning RL V1 — GRPO verifier-first cho giả thuyết giao dịch

## Trạng thái

Đây là nhánh **candidate-only**. Code, cấu hình và test chứng minh đường gradient
đã tồn tại; chưa có MT5 Strategy Tester hoặc frozen-OOS chứng minh lợi thế giao
dịch. Cấu hình production V193 và seed/checkpoint gốc không bị sửa.

Profile thử nghiệm:

`configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml`

Profile này vẫn là cấu hình telemetry-only khi gọi trực tiếp (action scale 0).
Luồng **Train candidate** của UI dùng profile overlay riêng
`configs/experiments/reasoning_rl_v1_active_candidate.yaml`: bật
`mt5_learning_enabled=true`, đặt `five_reasoning_action_scale=0.25` và giữ
`five_reasoning_encoder_credit_scale=0.0`. Vì vậy suy luận thực sự đi vào bảy
logit HOLD/entry và FiveAdam có thể update sau khi path 48-M1 mature; cả hai
đường vẫn candidate-only, không tự động promote.

Khi dùng nhãn bảy lựa chọn mới, phải dùng hồ sơ cùng thời gian giữ lệnh và
hình học cố định: `configs/he_chuyen_gia_reasoning_rl_v1_candidate_sl500_h12.yaml`,
`..._h24.yaml` hoặc `..._h48.yaml`. Hồ sơ cũ ở trên giữ hình học 2.5/5 ATR và
không cùng bộ nhãn sl500; bộ kiểm tra hợp đồng sẽ từ chối ghép chúng.

Bridge vẫn giữ đúng `137 input / 20 output`. Không thêm trường future vào
observation, recurrent memory, action request hoặc value input.

## Ánh xạ tài liệu RL reasoning sang trader

Một sample reasoning được định nghĩa như sau:

- state: representation causal sau LSTM và Transformer hậu-LSTM tại quyết định;
- reasoning action: một token trong `HOLD + MARKET/LIMIT/STOP BUY/SELL`;
- trajectory: ba bước `đề xuất -> phản biện/sửa -> kế hoạch cuối`;
- verifier: `action_q_values_r` từ path counterfactual 48-M1 đã mature;
- group rollout: tám trajectory cho cùng state và cùng agent;
- advantage: reward trừ mean của group, chia standard deviation của group;
- optimizer: PPO clipped surrogate, KL tới reference head cố định và entropy;
- writer: `FiveAdam` hiện hành, không có optimizer thứ hai.

Reward chỉ chấm hypothesis cuối. Hai bước trước vẫn nhận credit qua tổng
log-probability của trajectory, giống sequence-level outcome reward. Nhóm không
có reward variance bị bỏ qua vì không chứa tín hiệu phân biệt.

## Biên chống leakage

`action_q_values_r` chỉ được tạo sau khi path tương lai hoàn tất và chỉ đi vào
`FivePPOTrainer.reasoning_grpo_loss()`. `forward_five()` không nhận target,
future bar, MAE/MFE hoặc outcome label. Telemetry trace luôn ghi
`training_labels_visible=false`.

Đường dữ liệu:

```text
observation 137 + recurrent state
    -> shared Five LSTM hidden
    -> Transformer 1 lớp trên 5 agent token
    -> reasoning head sinh G trajectory x S bước

48-M1 path mature (training-only)
    -> action_q_values_r verifier
    -> group-relative advantage
    -> PPO clip + KL(reference) + entropy
    -> FiveAdam
```

## Checkpoint và reference policy

Checkpoint policy architecture v2 có thể được gắn reasoning head bằng migration
xác định, không tiêu thụ RNG của caller. Checkpoint v1 thiếu Transformer bị từ
chối, không được migrate. `reasoning_reference_head` là bản đóng băng, không thuộc tập
tham số writer. Reference chỉ được thay bằng lệnh gọi tường minh
`refresh_reasoning_reference()`; optimizer update không tự dịch neo KL.

Khi `five_reasoning_action_scale=0`, việc bật head phải giữ nguyên bit-for-bit
action logits, probability và 20-output bridge. Đây là mặc định của config nền.
Profile active đặt scale 0.25; residual được clip ở 0.25 trước khi cộng vào
bảy logit HOLD/entry, còn mask legality và authority Five vẫn áp dụng sau đó.

## Các giai đoạn thẩm quyền

### Giai đoạn 1 — telemetry/head-only

```yaml
five_reasoning_rl_enabled: true
five_reasoning_encoder_credit_scale: 0.0
five_reasoning_action_scale: 0.0
```

Chỉ reasoning head học. Encoder/LSTM/Transformer và action policy không nhận gradient từ
objective reasoning.

### Giai đoạn 2 — encoder credit có giới hạn

Chỉ cân nhắc tăng `five_reasoning_encoder_credit_scale` sau khi prediction/reward
audit cho thấy nhóm có variance, KL ổn định và deterministic replay lặp lại.

### Giai đoạn 3 — ảnh hưởng action có giới hạn (active candidate)

Profile active hiện chạy ở mức cố định `five_reasoning_action_scale=0.25` để
kiểm tra nhân quả trong candidate MT5. Khi bật, residual chỉ tác động bảy logit
HOLD+entry, bị chặn bởi `five_reasoning_action_residual_clip`; hard legality/
agent authority vẫn áp dụng sau đó. Kết quả vẫn phải qua paired MT5 và frozen
OOS trước khi cân nhắc tăng scale hoặc promote.

## Telemetry bắt buộc

- `reasoning_group_eligible`;
- `reasoning_optimizer_eligible`;
- `reasoning_policy_loss`;
- `reasoning_kl`;
- `reasoning_entropy`;
- `reasoning_reward_mean_r` và `reasoning_reward_std_r`;
- `reasoning_reference_frozen`;
- `reasoning_encoder_credit_scale`;
- reasoning steps và final hypothesis ở decision trace.

Một reward chỉ được coi là tác động model khi group đủ variance, sample nằm
trong optimizer-eligible mask, gradient hữu hạn và `step_from_loss()` thực hiện
optimizer step.

## Promotion gate tối thiểu

1. Verifier audit không leak và tái tạo deterministic từ ledger bất biến.
2. Có đủ support theo action/side/entry type; không báo aggregate thay thế scope.
3. KL, entropy, grad norm và reasoning length ổn định qua nhiều checkpoint.
4. Paired MT5 Model 1 cùng seed/window cho baseline và candidate.
5. Frozen chronological OOS không regress expectancy/PF/DD và policy-origin LCB.
6. Reward-hacking audit: không học parser loophole, future field hoặc shortcut
   theo trace ID/sequence.
7. Rollback ngay khi hai OOS liên tiếp regress hoặc action-scope failure tăng.

## Validation hiện có

Targeted tests kiểm tra:

- action-scale 0 giữ nguyên output;
- GRPO có variance cập nhật reasoning head;
- encoder credit 0 giữ nguyên LSTM và Transformer;
- reference KL không tự thay đổi;
- group reward đồng nhất không tạo optimizer-eligible sample;
- config nền giữ reasoning telemetry-only và bridge 137/20;
- active candidate làm action logits thay đổi ở scale 0.25 và bật đường
  FiveAdam, nhưng encoder credit vẫn bằng 0.

Các test này là bằng chứng triển khai, không phải bằng chứng MT5/OOS.

## UI V193

Tab **Train candidate** mặc định chọn `Reasoning RL V1 · action + optimizer` và
ghép config candidate ở trên với profile active
`configs/experiments/reasoning_rl_v1_active_candidate.yaml`. Người dùng vẫn có
thể chọn chế độ V193 không reasoning hoặc Codex teacher trong chính tab này.
Hợp đồng Five/Bridge và trạng thái `3 bước × nhóm 8`, `encoder credit=0`,
`action scale=0.25`, `FiveAdam bật` nằm ngay dưới form train nên không cần một
tab kiến trúc riêng.

Mỗi lần chạy candidate ghi `runtime_learning_contract.json` ngay khi khởi động.
Profile active bị từ chối trước khi mở Strategy Tester nếu `mt5_learning_enabled`
hoặc action/reward path bị tắt; vì vậy không còn phiên mang nhãn reasoning nhưng
thực tế chỉ ghi telemetry với `ppo_updates=0`. Sau khi policy được nạp, bridge còn
kiểm tra lại cờ học của facade và từng owner. Close âm hợp lệ đi theo
`owner_lifecycle_reward_to_five_adam`, được giữ dấu âm trong queue lifecycle rồi
flush thành batch FiveAdam.

Với mode **Codex teacher**, chỉ receipt có owner trace, packet/reference/checkpoint
hash hợp lệ và `verification_status=confirmed` mới được đưa vào `FiveAdam`. Hợp
đồng runtime phải tách riêng `codex_cli_found`, `codex_preflight_passed`,
`codex_receipts_valid`, `codex_hypotheses_confirmed` và
`codex_optimizer_updates`; các số `ppo_updates` tổng quát không chứng minh Codex
đã học. Mỗi receipt thành công ghi reward, advantage, hash action/residual
trước/sau, probability trước/sau, parameter scope, số tensor của latent
policy-state, `rlhf_internal_policy_state_parameter_changed` và
`rlhf_ppo_update`. `rlhf_action_head_update_norm` là norm thực của delta trên
toàn bộ scoped trainable tensors sau bounded step; `rlhf_action_head_target_delta`
là delta logit của action được chấm. Hai trường này không được suy ra từ
`ppo_updates` tổng quát.

Để tránh reward âm của một entry lan thành thiên lệch HOLD/CLOSE trên mọi
context, Codex PPO không backprop qua `codex_action_bias` toàn cục. Cờ
`rlhf_codex_global_bias_gradient_frozen=true` phải xuất hiện trong update
receipt; bias này chỉ nhận intervention bounded cho đúng action được chấm và
vẫn chịu probability/direction guard. Gradient PPO thực sự đi vào
`codex_action_residual` và `internal_policy_state_net`, tức mạng ẩn vẫn là
đường quyết định neural có điều kiện theo context, không phải một router hay
strategy ID bên ngoài.
`rlhf_codex_ppo_gradient_parameter_tensors` ghi riêng số tensor thực sự nhận
PPO gradient; `rlhf_trainable_parameter_tensors` vẫn mô tả toàn bộ scope có thể
được learner/intervention cập nhật.

Bias toàn cục được thêm vào logits qua `effective_codex_action_bias`. Sáu action
entry (BUY/SELL LIMIT/STOP) được center theo mean sau khi giới hạn `tanh`; HOLD
và các action quản lý không bị center. Nhờ vậy receipt âm cho nhiều loại entry
vẫn làm thay đổi thứ hạng từng action, nhưng không cộng dồn thành một shift
chung chống entry khiến frozen replay rơi vào HOLD-only. Guard probability và
direction vẫn đo trên logits hiệu dụng sau bước center; đây là bảo vệ chống
collapse, không phải bỏ qua reward âm hay tự tạo lệnh. Khi packet còn ít nhất
hai entry hợp lệ, Codex line-search còn giữ `entry_probability_mass` trên một
floor bounded: tối đa cho phép giảm `min(0.0125, 25% * max_probability_shift)`
trong một packet. Nhờ đó một action xấu vẫn bị phạt tương đối, nhưng latent
update không thể xoá cả lớp entry trong vài receipt liên tiếp; trường
`rlhf_codex_entry_probability_mass_before/after/floor` ghi lại phép kiểm này.
Positive receipt có account impact `partial` không được ghi global bias cho
context chưa từng thấy; nó chỉ được phép cập nhật residual/context state. Chỉ
account evidence `confirmed` mới đủ điều kiện phát tán positive intervention
toàn cục, còn negative/legacy path vẫn đi qua probability và direction guard.

`account_impact.account_mark_attribution` phải được giữ nguyên từ packet. Verifier
chỉ canonicalize một alias wording đã biết của `evidence_scope`; status,
`verification_ready`, attribution và mọi delta immutable vẫn phải khớp packet,
không thể dùng alias để nâng bằng chứng `partial` thành `confirmed`.
Với entry owner, `realized_r`/net-PnL và dấu balance/equity là mục tiêu account
tuyệt đối; counterfactual HOLD chỉ là tín hiệu phụ để điều chỉnh độ lớn. Vì vậy
sign guard giữ reward entry cùng dấu với outcome account: một lệnh làm giảm
balance không thể biến thành advantage dương chỉ vì giữ lệnh sẽ lỗ nhiều hơn.
Riêng management/exit owner vẫn dùng `counterfactual_delta_r` vì đó là phần
outcome tăng thêm mà action quản lý thực sự sở hữu.
Update receipt còn flatten `account_impact_observed_realized_r`,
`account_impact_observed_net_pnl`, `account_impact_balance_delta` và
`account_impact_equity_delta` từ packet bất biến để UI đối chiếu trực tiếp
account outcome với `scaled_reward`, advantage và hash/probability update.
Receipt/update cũng giữ riêng `teacher_reward`, `account_impact_reward` và
`combined_reward`. `combined_reward` mới là tín hiệu trước confidence/scale mà
PPO dùng; composition hiện tại là
`teacher_account_impact_50_50_account_negative_floor_v1`: hai nguồn được trộn
50/50, còn account evidence âm áp một sàn âm để rationale lạc quan không thể
đảo lỗ thành thưởng. `reward_direction_source`,
`account_impact_reward_source` và `negative_account_floor_applied` cho phép UI
giải thích tại sao hướng update có thể khác dấu với điểm Codex thô mà không
hiển thị hidden reasoning.

`causal_snapshot` được tạo sớm để giữ bằng chứng suy luận, nhưng action contract
cuối cùng chỉ có sau khi bridge chốt `semantic_action`, `emitted_action` và
`wire_action`. Trước khi ghi decision/lifecycle sau bridge, runtime re-register
cùng `trace_id` vào causal ledger và archive, đồng thời ưu tiên giữ các trường
contract khi serializer cắt telemetry lớn. Vì vậy snapshot mới phải có
`action_contract_state=COMMAND` (hoặc `NO_COMMAND_WIRE_NOOP`) và index action
đúng lệnh đã gửi; bản ghi cũ `INCOMPLETE_OR_MISMATCH` không được dùng để kết luận
model không nhận reward. Đây vẫn chỉ là bằng chứng nối action → outcome/learner,
không phải bằng chứng edge, lợi nhuận hay OOS.

Candidate active ghi `five_rlhf_strip_optimizer_state_for_oos=true` ở biên persist:
mọi update FiveAdam/Codex đã hoàn tất trước khi lưu, nhưng tensor m/v của Adam
được bỏ khỏi artifact OOS để checkpoint Windows không phình nhiều gigabyte. Marker
`_five_rlhf_optimizer_state_stripped` làm artifact này không được dùng tiếp như
checkpoint continuation; lần train kế tiếp phải bắt đầu từ source seed còn giữ
optimizer state. Đây là tối ưu lưu trữ sau học, không phải tắt optimizer trong
lúc MT5 đang chạy.

Sau khi candidate được ghi, runner thêm `continuation_economic_verdict` vào
`manifest.json`. Verdict so sánh `previous_net_profit` của checkpoint cha với
metric của **candidate**, không nhầm với metric của checkpoint cha được paired
gate chọn lại để persist. Khi paired validation đã chạy, nguồn được ghi rõ là
`paired_seed_validation.candidate_net_profit`; nếu chưa có paired pass thì dùng
`mt5.metrics.net_profit`. Verdict cũng xét `fitness_eligible`: candidate chỉ
được đánh dấu `candidate_preferred` khi không regress và đã đạt fitness; nếu
không, `preferred_continuation_seed` trỏ về checkpoint cha với lý do
`reuse_parent_seed_until_fitness_eligible`. `economic_comparison` phân biệt
`regressed`, `inconclusive_ineligible` và `eligible_non_regression`, nên một
candidate có PnL tăng nhưng chưa đủ mẫu không bị hiển thị như đã cải thiện.
Candidate âm vẫn được giữ nguyên để audit hoặc chạy research khi người dùng chỉ
định rõ, nhưng không còn được coi ngầm là seed tốt hơn cho thế hệ kế tiếp. Đây
là guard chọn nhánh, không phải bằng chứng profitability hay promotion.

Probability-shift guard vẫn giữ ở mức cấu hình. Nếu một bước hợp lệ vượt guard,
learner rollback snapshot rồi thử lại với learning-rate giảm một nửa trong giới
hạn hữu hạn; receipt ghi `rlhf_codex_backtrack_applied`, số lần thử và learning
rate ban đầu/thực tế. Vì vậy `rejected_updates` là các lần verifier/guard từ chối
cuối cùng, không phải số Codex đã chấm lỗi; lỗi CLI/Tensor phải xuất hiện trong
failure record theo từng packet và làm tăng `failures`/`grade_failures`. Failure
record giữ nguyên packet identity; `action=0` là `HOLD` hợp lệ (chỉ `None`
mới dùng sentinel `-1`), để UI và execution ledger vẫn join được cả case
HOLD bị lỗi.
Receipt thành công cũng được ghi theo hai record append-only cùng identity:
`codex_teacher_receipt` là kết quả chấm đã validate, còn
`codex_teacher_update` ghi kết quả learner handoff (`scaled_reward`, advantage,
`rlhf_ppo_update`, parameter hash trước/sau, parameter scope, probability trước/sau
và account impact). Record update có thể là `accepted=false` khi verifier/guard
từ chối; không được suy ra update từ `graded` hoặc từ các bộ đếm PPO tổng quát.

Grader vẫn chạy trong worker pool để không gọi subprocess trên thread optimizer,
nhưng close mới không còn chạy qua một luồng async vô hạn. Với Codex hypothesis
mode, cả entry owner và management/exit owner đều giữ packet tới khi ledger có
`POST_EXIT_OBSERVATION` đủ horizon. Entry owner dùng `realized_r` cùng
`post_exit_hold_r` để phân biệt tác động của việc vào lệnh với baseline giữ tiền;
management/exit owner dùng cùng hold path để đo incremental effect của lệnh đóng.
Nếu tester kết thúc trước horizon, packet chỉ được flush partial ở boundary cuối và
status phải ghi rõ `counterfactual_unavailable`, không được coi đó là bằng chứng
đã có đối chứng. Sau khi lifecycle cuối được xác nhận và packet được submit, bridge chờ tối đa
`five_codex_teacher_online_wait_sec` (mặc định `30.0` giây) cho **một** receipt
`confirmed` đi hết verifier và tạo `FiveAdam` update trước khi lấy action kế
tiếp. Hết hạn chỉ ghi timeout và giữ guard fail-closed; packet còn lại được drain
ở barrier/lifecycle sau hoặc ở phase cuối run. `online_barrier_passed`,
`online_barrier_timeouts`, `online_barrier_waited_sec` và `online_barrier_last`
được ghi trong status/manifest/runtime contract. Đặt giá trị `0` chỉ khi cố ý
chọn async telemetry.

Bằng chứng Codex tác động hành vi phải có cả hai phía: update receipt có hash
action-head/probability trước-sau, và `five_intent_registered` sau thời điểm update
phải ghi `five_codex_global_action_bias_enabled=true` cùng bias khác 0 (trong
giới hạn). `runtime_learning_contract.json` cuối run giữ các counter này; UI
hiển thị riêng barrier bên cạnh `graded`, `hypothesis confirmed`, `Codex
optimizer updates` và `failures`.

Live telemetry đưa `five_reasoning` vào decision payload: enabled, chuỗi giả
thuyết, hypothesis cuối, action scale, residual clip và cờ
`training_labels_visible`. Java Live và browser observatory hiển thị dữ liệu này
trong **Quyết định**. Tab **Đầu ra tư duy** của ReasoningGraph cũ đã bị xóa để
không đánh đồng trace M1 lịch sử với GRPO verifier-first hiện tại.
MT5 diagnostics hiển thị riêng `reasoning_optimizer_updates`, tổng update,
`reasoning_group_eligible` và `reasoning_optimizer_eligible`. Hai trường
`*_eligible` là tổng số group/sample đủ điều kiện qua toàn bộ các event
`five_pattern_path_credit_accepted` đã đọc, không phải giá trị của event cuối;
vì vậy một event cuối có variance bằng 0 không thể xoá bằng chứng của các group
đã đủ điều kiện trước đó. Số batch pattern được chấp nhận vẫn không tự động
được hiểu nhầm là head reasoning đã update.

### Guard chống trôi entry và collapse ở cuối run (2026-10-02)

Guard Codex không còn chỉ so entry-mass với policy ngay trước receipt. Mỗi
receipt hợp lệ re-forward frozen reference trên đúng packet/context, ghi
`rlhf_codex_entry_probability_mass_reference` và giữ sàn tích lũy
`reference - budget` trong suốt run. Nếu contextual PPO step tự nó vượt sàn,
runtime trước hết thử một compensation bounded trên residual action-head: với
reward âm và target là entry, phần mass còn lại được phân bổ tạm thời cho các
entry thay thế (không cộng lại vào target), đồng thời vẫn giữ direction guard
và `five_codex_teacher_max_probability_shift`. Chỉ khi không tìm được candidate
hợp lệ runtime mới rollback chính xác về snapshot trước update và đánh dấu
reject. Receipt/update audit ghi
`rlhf_codex_entry_mass_compensation_applied`, step, số lần thử/từ chối và lý do
`entry_mass_compensation_for_frozen_reference_floor`; các counter này không biến
thành một reward mới và không nới floor. Cách này giữ được phạt tương đối cho
action xấu mà không để frozen replay dần đẩy toàn bộ entry về `HOLD`. Status/
update audit vẫn ghi `rlhf_codex_entry_mass_guard_mode=frozen_reference_cumulative`
và số lần bị `rlhf_codex_entry_mass_cumulative_floor_rejections`.

Nếu current packet đã bị các global update trước đó kéo thấp hơn frozen floor một
khoảng lớn hơn `max_probability_shift`, floor gốc không còn khả thi trong một
packet bounded. Runtime chuyển sang recovery fail-safe: giữ
`rlhf_codex_entry_probability_mass_cumulative_floor_original` để audit, đặt
effective floor bằng entry-mass hiện tại và chỉ cho phép update nếu không làm
giảm thêm mass. Cờ `rlhf_codex_entry_mass_floor_recovery_active=true` bắt buộc
hiển thị cùng receipt; nó nghĩa là đang sửa drift, không nghĩa là đã phục hồi
về frozen reference và không phải bằng chứng profitability.

Action diagnostics cũng tách histogram toàn cửa sổ khỏi tail 25 bridge rows.
`tail_collapse_detected` bắt trường hợp đầu run còn BUY/SELL nhưng cuối run đã
HOLD-only; continuation/champion gate phải xét cờ tail này, không được tin rằng
`effective_action_count` toàn cửa sổ còn đa dạng là an toàn. Đây là guard hành
vi và bằng chứng rollback, không phải bằng chứng edge, lợi nhuận hay promotion.
`mt5_diagnostics.json` cũng ghi tail tương tự từ observatory (`tail_window_size`,
`policy.tail`, `tail_collapse_detected`) để UI không phải suy diễn từ chart.

Live diagnostics còn có block `codex_teacher` riêng, đọc từ status/receipt
append-only của teacher. Block này giữ tách biệt `submitted`, `graded`,
`hypotheses_confirmed`, `updates`, `interventions_applied`, `failures`,
`account_impact_confirmed`, hash/probability-change và CLI preflight với bộ đếm
PPO/pattern tổng quát. `optimizer.optimizer_update_count` không cộng các update
Codex; UI phải đọc `codex_teacher.updates` hoặc
`optimizer.codex_teacher_updates` khi hiển thị learner handoff.

Khi runner đã cleanup và execution CSV không còn, whole-run wire histogram
được phục hồi read-only từ các `five_execution_receipt` còn lại trong
observatory. Diagnostics giữ riêng `wire_collapse_detected`,
`economic_actor_collapse_detected` và `wire_collapse_warning`: cảnh báo wire
HOLD-only không tự động thay thế gate economic-actor, nhưng không còn bị mất
chỉ vì chọn scope kinh tế hoặc vì CSV đã bị dọn.

Từ bản role-scoped guard, snapshot còn ghi riêng
`entry_role_distributions` cho `LONG_ENTRY_OWNER` và `SHORT_ENTRY_OWNER` cùng
`entry_role_collapse_detected`. Hai role này được xét cả khi tài khoản phẳng,
vì HOLD của entry-owner vẫn là quyết định có thể ngăn cơ hội vào lệnh; ngược
lại HOLD của `POSITION_MANAGER`, `PENDING_MANAGER` và `PORTFOLIO_GOVERNOR`
không được dùng để kết luận collapse. Frozen continuation/champion gate fail
closed nếu một entry-owner có ít nhất 8 mẫu và hold-share/effective-action/
entropy vượt ngưỡng collapse. Điều này phát hiện mismatch mà run học có
behavior exploration che khuất: ví dụ main có thể còn BUY/SELL nhưng paired
deterministic replay lại cho LONG entry HOLD-only. Đây là bằng chứng kiểm soát
hành vi, chưa phải bằng chứng lợi nhuận hay promotion.

Panel **MT5 diagnostics** của Java UI render trực tiếp block này dưới dòng
`Entry owners`: mỗi role có `n`, `HOLD` share và `effective actions`, kèm cờ
`collapse`. Vì vậy aggregate `POLICY / EXECUTION` không còn là nguồn duy nhất
để kết luận model đang đa dạng hành động; khi role-level và aggregate khác nhau,
role-level của entry owner được ưu tiên cho chẩn đoán cơ hội vào lệnh.

Khi một entry owner chưa có lifecycle đóng lệnh nào trong run hiện tại, profile
candidate có thể bật `five_live_owner_bootstrap_probe_enabled`. Theo cadence và
quota giới hạn, selector chỉ giữ `HOLD` cùng market entry của owner thiếu mẫu
(`MARKET_BUY` hoặc `MARKET_SELL`) trong allowed-mask. Đây là lease thu thập bằng
chứng nhân quả, không phải chiến lược ngoài mạng: trace ghi
`five_live_owner_bootstrap_probe_owner/action/samples/count`, negative-veto vẫn
được tôn trọng, account có vị thế hoặc pending order không probe. Chỉ lifecycle
được khớp và đóng mới tạo account-impact reward; pending-only không được coi là
SHORT đã học. Probe này chỉ giúp tạo mẫu để kiểm chứng và không phải bằng chứng
lợi nhuận, OOS hay promotion.

Một profile nghiên cứu có thể bật thêm
`five_live_owner_bootstrap_probe_force_market=true`. Trong đúng lease zero-sample
đó, `HOLD` bị loại khỏi mask của owner được chọn để bảo đảm có một market
lifecycle thay vì chỉ mở xác suất rồi tiếp tục HOLD; action/mask/log-prob vẫn
được ghi như behavior policy và quota vẫn giới hạn. Cờ này mặc định tắt, không
được dùng cho frozen/OOS/promotion, và không vượt qua negative/context veto.

Trong lease force-market, mask gửi lệnh cho bridge vẫn chỉ cho phép market. Trace
đồng thời giữ `five_rlhf_update_action_mask` là legal owner mask trước khi ép
market để PPO còn phân phối xác suất giữa `HOLD` và market. Vì vậy probe không bị
biến thành packet xác suất `1.0 -> 1.0` không có gradient; đây là mask học bounded
cho đúng trace, không mở thêm action cho MT5 và không thay đổi guard
reward/provenance.
