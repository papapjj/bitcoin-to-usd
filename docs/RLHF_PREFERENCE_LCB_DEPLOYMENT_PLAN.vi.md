# RLHF thật cho policy trading và lộ trình triển khai

> Phân biệt bắt buộc: `teacher`/`mt5_rater` là RLAIF hoặc outcome-audit
> supervision, không phải human RLHF. Chỉ pair có `label_source=human` hoặc
> `expert` mới được tính vào nhánh human-feedback; mọi nhánh vẫn phải qua
> reward-model, policy và MT5 frozen-OOS gate.

## Trạng thái triển khai hiện tại

Đã triển khai các lớp candidate sau, nhưng chưa tuyên bố full RLHF hay
promotion:

- `src/self_growing_trader/rlhf_preference.py` định nghĩa schema
  `rlhf_preference_v1`, parser JSONL, validator chống leakage, Bradley--Terry
  loss, metric pairwise và offline gate.
- `ChuyenGiaPolicy.apply_rlhf_preference_credit_detailed()` nhận một cặp
  `chosen/rejected`, kiểm tra trace cache và action mask, rồi đưa item vào
  đúng queue Full-RL AdamW hiện có.
- `_apply_direct_rnn_full_rl_batch()` là writer Direct-RNN AdamW duy nhất cho
  lifecycle credit; Codex legacy/context dùng cùng writer nhưng đi qua
  micro-batch một packet theo đúng owner, không trộn với lifecycle queue. Sau
  mỗi step, projection dùng đủ decision-time action surface của trace (không
  cắt về teaching surface bảy action), giữ nguyên hierarchical HOLD/trade
  denominator nếu action surface đã bật chế độ đó, reset AdamW moment state của Codex
  packet, rồi backtrack theo learning-rate giảm dần nếu hướng reward sai hoặc
  probability shift vượt guard. Nếu không đạt, tensor, optimizer state và
  runtime counters được restore; không có inference veto hay writer thứ hai.
- `HeChuyenGiaNguCanh.apply_rlhf_preference_credit_detailed()` đã route pair
  về đúng expert owner.
- `scripts/validate_rlhf_preference_dataset.py` kiểm tra JSONL trước khi cho
  phép nạp vào training.
- `src/self_growing_trader/rlhf_annotation.py` và hai script
  `export_rlhf_annotation_queue.py`/`label_rlhf_annotation.py` đã nối bước
  observatory causal -> task chưa gắn nhãn -> lựa chọn `human`/`expert` có
  kiểm tra hash, action surface, feature schema và append-only dataset. Bộ
  này không tự chọn winner từ PnL, nên không biến outcome audit thành RLHF.
- `src/self_growing_trader/rlhf_reward_model.py` có reward model Bradley--Terry
  tuyến tính, chuẩn hoá chỉ trên train split, hash bất biến, load/score và gate
  riêng; feature vector bắt buộc causal và candidate-only.
- `src/self_growing_trader/rlhf_ppo.py` có objective PPO thật với old-policy
  ratio, exact categorical KL tới reference policy đóng băng, value loss và
  entropy; hàm chỉ tạo loss, không tự ghi tham số.
- `src/self_growing_trader/rlhf_policy_loop.py` nối candidate rollout causal
  -> reward-model score -> one-step return/advantage -> `FiveAdam` PPO update,
  đồng thời ghi state/geometry/hash provenance và candidate manifest; có helper
  clone/freeze reference policy và hash state dict trước rollout.
- `ChuyenGiaPolicy.optimize_five_rlhf_trace_packet()` đã nối packet trace Five
  thật vào closed loop: sắp thứ tự 5 agent, tái dựng hidden/cell/temporal
  state, re-forward policy có gradient rồi gọi đúng `FiveAdam`. RLHF V1 chỉ
  dùng bảy action entry; trace management 7..11 được đếm và bỏ qua có chủ ý,
  không bị ép vào one-hot entry sai schema. Worker MT5 cũng đã có hook
  candidate-only opt-in để gọi adapter trước quyết định kế tiếp.
- `src/self_growing_trader/rlhf_feedback.py` và
  `scripts/record_rlhf_feedback.py` là đường capture append-only; mặc định chỉ
  nhận `human`/`expert`, reject duplicate hoặc dataset đã có dòng lỗi và ghi
  hash dataset sau mỗi receipt.
- Public Five trace giờ giữ raw categorical policy logits riêng với 20-output
  execution wire. Khi gọi `install_five_rlhf_reference()`, policy clone và
  freeze checkpoint hiện tại, giữ recurrent memory riêng, chạy reference
  song song với cùng observation/action mask và tự ghi reference logits,
  frozen flag cùng SHA256 vào `StepTrace`. `build_rlhf_policy_rollouts()` có
  thể đọc thẳng dữ liệu này; replay offline vẫn được phép truyền map logits
  nhưng phải kèm reference hash, không được suy ngược từ wire hoặc hindsight.
- `scripts/train_rlhf_reward_model.py` fit/đánh giá reward model theo
  train/validation/test và fail-closed khi thiếu human feedback, rater, split
  hoặc pair. Cờ `--feedback-mode rlaif` tạo report riêng cho `teacher`/
  `mt5_rater`, không cho trộn với nhánh human.
- `src/self_growing_trader/rlhf_rlaif.py` và
  `scripts/build_rlaif_teacher_dataset.py` nối score của teacher vào đúng hai
  candidate đã xuất hiện trong annotation task. Score phải khai báo
  `causal_input=true`, không được mang PnL/future/lifecycle field; margin mơ
  hồ bị bỏ qua. Artifact được gắn `offline_rlaif_teacher_only` và vẫn
  candidate-only.
- `scripts/score_rlaif_tasks_with_frozen_policy.py` đã smoke-test trên một task
  thật: checkpoint hash `72d2bad86d53f087d6f2dbe4c190311f5ba30d7008f37fcf4c45bd7780469038`,
  score row và một RLAIF pair được tạo thành công. Dataset smoke chỉ có một
  pair/train nên reward-model gate từ chối đúng vì thiếu validation/test; đây
  không phải artifact đủ điều kiện để bật MT5.
- Dataset RLAIF chronological đầy đủ từ capture đã được tạo tại
  `outputs/rlhf/rlaif_reward_pairs_v2_chrono.jsonl`: 62 pair, chia train/
  validation/test là 42/8/12. Report
  `outputs/rlhf/rlaif_reward_model_v2_chrono_report.json` qua test gate; tuy
  nhiên đây chỉ chứng minh reward model khớp nhãn của frozen teacher, không
  chứng minh trading edge hay hiệu quả MT5.
- Đã tạo thêm nhánh teacher độc lập tại
  `outputs/rlhf/rlaif_reward_pairs_independent_v2.jsonl` và report
  `outputs/rlhf/rlaif_reward_model_independent_v2_report.json`: 99 pair,
  train/validation/test là 69/12/18, teacher hash
  `89ebdd7fa7eb64ef9f151b72285d017b2ee3c6f7268d7b46fd382b4c58cc5342`, khác
  champion hash `72d2bad86d53f087d6f2dbe4c190311f5ba30d7008f37fcf4c45bd7780469038`.
  Report giữ provenance này và qua test gate; vẫn chưa phải trading/OOS claim.
- Candidate MT5 RLAIF thật đã chạy với config
  `configs/he_chuyen_gia_rlaif_policy_candidate.yaml` với teacher độc lập:
  32 packet, 8 PPO updates, 23 skip do interval/management/no-entry surface,
  0 failure. Run bị
  dừng bởi `max_wall_sec=65`, được lưu là incomplete, `promotion_allowed=false`;
  đây là bằng chứng closed-loop đã được gọi, không phải bằng chứng OOS đạt.
- Có thể audit một summary MT5 bằng:

```powershell
python scripts\verify_rlhf_mt5_shadow.py `
  outputs\rlhf\rlaif_mt5_champion_shadow_audit_v2\summary.json `
  --expected-feedback-mode rlaif `
  --min-updates 1
```

Audit pass chỉ có nghĩa là evidence contract hợp lệ. Thêm
`--require-complete` sẽ fail đối với artifact incomplete; tool không bao giờ
tự chuyển `matrix_gate.eligible=false` thành promotion.

Phần policy PPO đã có adapter candidate từ packet trace và đường gọi opt-in
trong worker MT5. Human/expert feedback thật vẫn chưa đủ để tuyên bố human
RLHF; các flag vẫn `disabled/candidate-only` mặc định và candidate RLAIF vừa
chạy không được promote. Vì vậy hiện trạng chính xác là:

| Thành phần | Trạng thái | Bằng chứng được phép kết luận |
|---|---|---|
| Human preference contract | Đã có | pair causal, provenance, split, rater và dedupe |
| RLAIF teacher contract | Đã có candidate adapter | teacher score -> cặp causal, không trộn human, không tự chọn action ngoài task |
| Reward model | Đã có RLAIF artifact với teacher độc lập qua test gate; human artifact chưa có | chỉ là offline teacher-agreement; không phải trading/OOS claim |
| PPO + reference KL | Đã có candidate objective | loss/gradient unit test |
| Closed-loop rollout -> reward -> PPO -> checkpoint | Đã có helper, packet adapter và MT5 candidate smoke với teacher độc lập; mặc định tắt | smoke thật ghi nhận 8 update/0 failure; candidate incomplete, chưa có checkpoint promote |
| MT5 reference shadow capture | Đã kiểm chứng bằng run frozen candidate; chưa bật mặc định | trace có current/reference logits, recurrent boundary và hash; capture không phải performance claim |
| MT5 frozen OOS/LCB | Chưa đạt | run mới chỉ có 1 window/1 seed, gate yêu cầu 3/3; chưa có performance/promotion claim |

Worker imitation v27 đang chạy độc lập, không bị restart hay trộn artifact với
RLHF.

## Vì sao phần preference cũ chưa phải RLHF

`entry_preference` hiện có là delayed counterfactual supervision: Python replay
bảy action trên path MT5 rồi tự tính utility. Nó hữu ích cho nhãn phản thực tế,
nhưng nhãn được sinh từ outcome/path, không thể hiện một rater chọn giữa hai
hành vi theo rubric.

RLHF mới tách rõ hai nguồn:

1. `entry_preference`: nhãn path MT5, dùng để học chất lượng entry/giữ lệnh.
2. `rlhf_preference`: cặp đánh giá chosen/rejected do human, expert, teacher
   hoặc rater MT5 tạo ra, dùng để học thứ tự ưu tiên hành vi. Chỉ `human` và
   `expert` là human-feedback; `teacher` là RLAIF/teacher-guided preference và
   `mt5_rater` là outcome-audit supervision.

Hai nguồn có thể cùng đi qua Full-RL AdamW nhưng không được ghi nhầm RLHF thành
reward lifecycle hoặc ngược lại.

### Nhánh teacher/RLAIF

Teacher không được ghi nhãn là con người. Có thể chấm queue bằng policy đóng
băng local trước, rồi tạo dataset:

```powershell
python scripts/score_rlaif_tasks_with_frozen_policy.py `
  --checkpoint outputs/<frozen-teacher>.pkl `
  --dataset-config configs/<matching-profile>.yaml `
  --tasks outputs/rlhf/annotation_tasks.jsonl `
  --output outputs/rlhf/teacher_scores.jsonl

python scripts/build_rlaif_teacher_dataset.py `
  --tasks outputs/rlhf/annotation_tasks.jsonl `
  --teacher-scores outputs/rlhf/teacher_scores.jsonl `
  --output outputs/rlhf/rlaif_reward_pairs.jsonl `
  --minimum-score-margin 0.05

python scripts/train_rlhf_reward_model.py `
  --feedback-mode rlaif `
  --input outputs/rlhf/rlaif_reward_pairs.jsonl `
  --output outputs/rlhf/rlaif_reward_model_report.json `
  --min-pairs 50
```

`teacher_scores.jsonl` chỉ chứa score bảy action cho từng `task_id`; nó không
được chứa `net_r`, PnL, close row, ticket, future path hay lifecycle. Nếu
teacher được huấn luyện offline từ outcome lịch sử, điều đó phải ghi ở
`teacher_training_evidence`; đây là provenance của RLAIF, không biến thành
human RLHF. Report RLAIF vẫn cần policy gate và MT5 frozen-OOS/LCB, và không
được bật promotion chỉ vì test pairwise đạt.

## Hợp đồng một preference pair

Mỗi dòng JSONL phải có các trường chính:

| Trường | Ý nghĩa và điều kiện |
|---|---|
| `pair_id` | ID bất biến, duy nhất; chống giao lại feedback |
| `context_id` | Cùng market context cho hai phương án |
| `chosen_trace_id`, `rejected_trace_id` | Hai quyết định thật/được replay trong trace cache |
| `chosen_action`, `rejected_action` | Chỉ thuộc `[HOLD, MARKET_BUY, MARKET_SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP]`; phải khác nhau |
| `pre_decision_state_hash` | Hash của 137-feature/state trước quyết định; phải cùng context |
| `risk_geometry_hash` | Hash của ATR, SL/TP, spread semantics và geometry contract; phải cùng pair |
| `decision_time`, `feedback_time` | ISO timestamp; feedback không được trước quyết định |
| `label_source`, `rater_id`, `rater_version` | Provenance bắt buộc; source hiện cho phép `human`, `expert`, `teacher`, `mt5_rater` |
| `teacher_training_evidence` | Bắt buộc về mặt provenance cho RLAIF; ghi teacher được huấn luyện/đóng băng từ nguồn nào, không thay thế human rater |
| `confidence` | Trọng số `[0,1]`; dưới ngưỡng policy thì không cập nhật |
| `feature_causal` | Bắt buộc `true`; future feature không được vào input hash |
| `candidate_only`, `outcome_audit_only` | Bắt buộc `true`; RLHF không tự promote và không thay reward ledger |

Một pair khác state, khác geometry, cùng action nhưng chỉ khác spread hoặc
khác exposure sẽ bị loại. Không được tạo pair bằng cách lấy hindsight-best
alternative rồi gọi đó là human feedback.

## Objective đang dùng

V1 dùng Bradley--Terry pairwise objective trên logits entry:

```text
margin = (logit(chosen) - logit(rejected)) / temperature
L_pair = confidence * softplus(-margin)
```

Gradient chỉ đẩy logit chosen lên và rejected xuống. Nó không chạm management
logits `7..11`, không tạo volume/risk action và không lấy credit từ close row.
Item được gắn với một immutable `StepTrace`; sau đó gradient đi qua
`module_directions` và vẫn kết thúc ở `apply_rnn_adamw_cosine(source="full_lifecycle_rl")`.

Nhánh preference adapter cũ không dùng reward model và không phải PPO. Hai
primitive candidate nằm ở `rlhf_reward_model.py` và `rlhf_ppo.py`; chúng đã
được gọi trong closed-loop RLAIF candidate của worker MT5, nhưng mặc định vẫn
tắt và chưa được phép promote. Khi nối loop, reference policy phải được snapshot từ
checkpoint trước update, không tự refresh sau mỗi batch; reward model cũng phải
được freeze trong policy phase. Chỉ một existing policy optimizer được ghi
tham số, với rollback nếu KL/action-mix/LCB drift vượt ngưỡng.

## Cách nạp và kiểm tra dataset

```powershell
python scripts\validate_rlhf_preference_dataset.py `
  outputs\rlhf\preference_v1.jsonl `
  --output outputs\rlhf\preference_v1.validation.json
```

`ready_for_training=true` chỉ có nghĩa là dataset đạt contract, đủ pair/rater
và đủ split train/validation/test. Nó không có nghĩa là model tốt hoặc LCB
dương. Với full human RLHF, reward-pair JSONL phải có thêm
`chosen_features`, `rejected_features`, `feature_schema`,
`reward_feature_causal=true`, `reward_candidate_only=true`, rồi chạy:

```powershell
python scripts\train_rlhf_reward_model.py `
  --input outputs\rlhf\reward_pairs_v1.jsonl `
  --output outputs\rlhf\reward_model_v1.json `
  --min-raters 3 `
  --min-inter-rater-agreement 0.60
```

Script này chỉ fit trên `train`, đánh giá riêng `validation/test`, bắt buộc có
human/expert rater, mặc định yêu cầu ít nhất 50 pair cho mỗi split và
inter-rater agreement tối thiểu 0.60 trên các context được nhiều rater chấm;
luôn ghi `promotion_allowed=false`. Smoke test nhỏ phải ghi rõ là smoke,
không được dùng làm bằng chứng chất lượng.

Một receipt human/expert được append bằng:

```powershell
python scripts\record_rlhf_feedback.py `
  --row outputs\rlhf\incoming\pair_0001.json `
  --output outputs\rlhf\reward_pairs_v1.jsonl
```

Writer không nhận `teacher` hoặc `mt5_rater` trong chế độ human RLHF. Hai nguồn
đó chỉ được đưa vào adapter RLAIF/outcome-audit riêng.

Metric offline bắt buộc trên split `validation/test`:

- pairwise accuracy và confidence-weighted accuracy;
- pairwise log loss, Brier score, mean margin;
- phân rã theo `label_source`, rater, regime, action pair và ngày;
- disagreement/inter-rater agreement và coverage;
- kiểm tra HOLD collapse, over-trading và action distribution.

Offline gate chỉ trả `evidence_class=offline_preference_only` và luôn trả
`promotion_allowed=false`. Không được dùng accuracy để thay LCB.

## Cấu hình candidate

Trong `learning` đã thêm các flag, mặc định an toàn:

```yaml
direct_rnn_rl_rlhf_enabled: false
direct_rnn_rl_rlhf_loss_weight: 0.0
direct_rnn_rl_rlhf_temperature: 0.25
direct_rnn_rl_rlhf_min_confidence: 0.70
five_rlhf_reference_capture_enabled: false
five_rlhf_policy_update_enabled: false
five_rlhf_reward_model_path: ""
five_rlhf_reward_feedback_mode: human  # human hoặc rlaif; phải khớp report
five_rlhf_teacher_checkpoint_sha256: ""
five_rlhf_require_independent_teacher: false
five_rlhf_source_checkpoint_sha256: ""
five_rlhf_strip_optimizer_state_for_oos: false
five_rlhf_allow_optimizer_state_reinitialization_for_research: false
five_rlhf_min_free_disk_bytes: 0
five_rlhf_update_interval_packets: 1
five_rlhf_max_updates_per_run: 0
```

`five_rlhf_reference_capture_enabled=true` chỉ bật snapshot/reference logits
trong một phiên MT5 candidate để thu thập rollout causal; nó chưa bật PPO hay
thay đổi action. Chỉ bật sau khi observatory đã được kiểm tra có đủ current
logits, reference logits và hash. `direct_rnn_rl_rlhf_enabled` là nhánh
preference adapter cũ, không nên dùng nó để tuyên bố đã chạy reward-model PPO.
Không đổi các flag này trên champion/prod artifact, không cho RLHF ghi tham số
trực tiếp và không bật inference gate.

`five_rlhf_policy_update_enabled=true` là cách bật một candidate MT5 training
run sau khi đã có đủ artifact. Nó bắt buộc có `five_rlhf_reward_model_path` trỏ tới
artifact reward model đã freeze và `five_rlhf_source_checkpoint_sha256` không
rỗng và trong champion-shadow matrix hash này phải trùng đúng file champion
bất biến. Với RLAIF, có thể bật `five_rlhf_require_independent_teacher=true`
để bắt buộc teacher provenance tồn tại và khác source champion. Worker sẽ tự
snapshot reference sau khi reset phiên, load artifact có
SHA256, rồi cập nhật candidate theo packet Five trước quyết định kế tiếp; mỗi
receipt được ghi vào session manifest/observatory. Flag này không train reward
model, không chạy trong frozen OOS, không cho management action 7..11 đi vào
schema RLHF V1 và không cho promotion. Nếu artifact/hash/reference/optimizer
receipt sai, phiên training phải fail-closed.
`five_rlhf_strip_optimizer_state_for_oos=true` chỉ dành cho artifact
candidate/forensic sau khi MT5 đóng: nó xoá các moment `m/v` của FiveAdam để
giảm kích thước pickle, nhưng giữ policy weights và đánh dấu artifact là đã
strip. Artifact này không còn là continuation checkpoint. Mặc định worker vẫn
fail-closed nếu cố bật policy update trên nó; nếu cần dùng weights đó làm điểm
bắt đầu research mới, phải bật tường minh
`five_rlhf_allow_optimizer_state_reinitialization_for_research=true`. Khi đó
FiveAdam xoá/khởi tạo lazy moment state mới, ghi
`optimizer_state_resume_mode=research_reinitialized` vào manifest/status và
không được coi là tiếp tục cùng quỹ đạo optimizer. Champion và source
checkpoint phải luôn giữ optimizer state nguyên vẹn.
Không bật flag này qua lệnh legacy `train-mt5`: flow đó bị chặn vì promotion
logic cũ không đủ để bảo đảm candidate-only. Dùng flow
`huan-luyen-he-chuyen-gia-mt5` của context Five.

Profile capture dùng riêng cho bước kiểm tra MT5 là
`configs/he_chuyen_gia_rlhf_reference_capture_candidate.yaml`. Kết quả của một
phiên hợp lệ phải có
`metrics.rlhf_reference_capture.installed=true`, một reference SHA256 không
rỗng và file `execution_trace_observatory` chứa cả
`five_policy_action_logits` lẫn `five_reference_policy_action_logits` ở cùng
`trace_id`, cùng `raw_obs`, `memory_before` và `allowed_action_mask` để có thể
re-forward causal về sau. Profile này vẫn không được dùng làm champion.

Probe MT5 ngày 2026-09-22 đã xác nhận đường capture thực tế: phiên frozen
candidate dùng seed riêng trong cửa sổ `2025.02.01`--`2025.02.02`, dừng bởi
`max_wall_sec=180` sau 111 packet nên artifact được đánh dấu
`incomplete=true`, không có claim fitness/OOS. Observatory sau compact có 165
dòng với `raw_obs`, current/reference logits và
`five_reference_policy_frozen=true`; exporter tạo 99 task pair chưa gắn nhãn
tại `outputs/rlhf/reference_capture_candidate.annotation_tasks.jsonl` và
sidecar hash manifest cùng tên `.manifest.json`. PPO
updates trong probe là `0`, đúng vì đây là capture-only.

Sau đó profile RLAIF candidate
`configs/he_chuyen_gia_rlaif_policy_candidate.yaml` đã load report reward
model có `feedback_mode=rlaif` và teacher provenance độc lập, snapshot
reference frozen và gọi closed loop trên MT5. Smoke seed `9723` dừng ở 65.8
giây với 32 packet; manifest giữ đúng `enabled=true`, `updates=8`, `skips=23`,
`failures=0`, teacher hash khác source champion. Artifact bị đánh dấu
incomplete và không promote. Vì
observatory có cơ chế compact theo giới hạn byte, số update chính thức lấy từ
manifest/summary; không suy ra performance từ số update.

## Lộ trình triển khai theo gate

### Giai đoạn A — contract và rater calibration

Thu tối thiểu 3 rater độc lập trên cùng các state đã freeze, blind theo kết quả
PnL khi có thể. Tách theo ngày/regime, không trộn train và test. Ghi rubric
reason code: timing, direction, entry type, risk, hold, execution cost.

Để tạo queue từ một observatory đã bật reference capture:

```powershell
python scripts/export_rlhf_annotation_queue.py `
  --input outputs/<run>/<execution_trace_observatory>.jsonl `
  --output outputs/rlhf/annotation_tasks.jsonl
```

Rater chọn hai `trace_id` trong từng task rồi ghi một row bằng
`label_rlhf_annotation.py`. Chỉ row có `label_source=human|expert` mới được
đưa vào reward-model JSONL; `teacher`/`mt5_rater` phải đi qua nhánh RLAIF hoặc
outcome-audit riêng. Exporter cũng ghi sidecar manifest chứa SHA256 của
observatory input và annotation queue output để không trộn task giữa các run.

Đạt khi dataset không có invalid line/duplicate pair, state/geometry hash đầy
đủ, inter-rater agreement được báo cáo và validation không bị một rater chi phối.

### Giai đoạn B — reward model offline

Fit `r_phi(state, action/path)` bằng cặp human/expert chosen/rejected. Không
đưa PnL/close row/future bar vào feature của rater. Chuẩn hoá chỉ lấy thống kê
train; validation/test chỉ score bằng model đã freeze. Ghi model hash, feature
schema, rater coverage, weighted pairwise accuracy, log loss, Brier và
disagreement theo ngày/regime/action-pair. Model không đạt test gate thì không
được tạo policy update.

Lệnh chuẩn cho human/expert là:

```powershell
python scripts/train_rlhf_reward_model.py `
  --input outputs/rlhf/expert_feedback.jsonl `
  --output outputs/rlhf/reward_model_report.json `
  --min-pairs 50 --min-raters 3
```

Với RLAIF teacher, report phải giữ `teacher_provenance.teacher_checkpoint_sha256`.
Profile MT5 có thể bật `five_rlhf_require_independent_teacher=true`; khi đó
loader bắt buộc teacher hash có trong report, khớp hash cấu hình và khác
`five_rlhf_source_checkpoint_sha256`. Teacher độc lập hiện đã được kiểm chứng
bằng report `outputs/rlhf/rlaif_reward_model_independent_v2_report.json`.

MT5 chỉ nhận report nếu `test_gate.passed=true`; loader cũng kiểm tra SHA256
và `feedback_mode` của report phải khớp với `rlhf_reward_feedback_mode` trong
profile MT5. Nếu dataset thiếu split, thiếu rater,
duplicate pair hoặc test gate fail thì loader từ chối, không có đường tắt bằng
việc đặt `promotion_allowed` thủ công.

### Giai đoạn C — offline preference/policy PPO

Đóng băng reward model và reference policy. `rlhf_policy_loop.py` nhận mỗi
rollout candidate và lưu
`old_log_prob`, `reference_logits`, action mask, state hash và reward-model
score. `ChuyenGiaPolicy.optimize_five_rlhf_trace_packet()` lấy packet này từ
MT5 `StepTrace`, tái dựng đúng recurrent boundary rồi lấy logits mới từ forward
có gradient; nó không dùng detached logits trong observatory để giả gradient.
Tính PPO clipped ratio trên action đã rollout, exact categorical KL trên
surface được khai báo của rollout, value loss và entropy; gọi existing
sole-writer optimizer đúng một lần qua
`FivePPOTrainer.optimize_rlhf_ppo_from_logits()`. Reject nếu
reward model không cùng hash, reference không freeze,
state/geometry không khớp, state/score không hữu hạn, hoặc optimizer event
không chứng minh được gradient khác 0.

Chạy metric trên validation/test, so sánh base policy với RLHF scorer. Bắt buộc
đạt pairwise floor theo config của experiment, không chỉ mean toàn tập; mỗi
regime/action-pair chính phải có coverage. Nếu chỉ nhóm BUY hoặc một ngày đẹp
đạt thì reject.

### Giai đoạn D — MT5 shadow/frozen

Dùng cùng checkpoint, cùng 137/20 contract, cùng SL/TP/normalization và cùng
ba cửa sổ chronological. Adapter packet đã có test Python và worker MT5 đã có
đường opt-in candidate-only: worker load reward model đã freeze, kiểm tra
teacher provenance, snapshot reference sau reset, cập nhật trước quyết định kế
tiếp và ghi receipt vào manifest/observatory. RLAIF candidate smoke với teacher
độc lập đã chứng minh đường gọi thật trên MT5, nhưng chưa có run đủ ba cửa sổ/
ba seed; vì thế chưa có bằng chứng hiệu quả hoặc promotion. Khi chạy đủ MT5,
baseline và candidate phải ở namespace riêng, bridge/ledger/report riêng.

Lệnh matrix chuẩn sau khi đã giải phóng đủ dung lượng và kiểm tra terminal:

```powershell
python -m self_growing_trader.cli `
  --config configs/he_chuyen_gia_rlaif_policy_candidate.yaml `
  huan-luyen-champion-he-chuyen-gia-mt5 `
  --champion outputs/he_chuyen_gia/kho_chuyen_gia.pkl `
  --output outputs/he_chuyen_gia/candidates/rlhf_candidate.pkl `
  --audit-dir outputs/rlhf/rlaif_mt5_matrix_v1 `
  --train-from 2025.02.01 `
  --train-to 2025.02.02 `
  --validation-count 3 `
  --validation-days 1 `
  --candidate-seed-count 3 `
  --seed 9723 `
  --model 4
```

Matrix này cố ý chạy tuần tự trong một process/MT5 writer: mỗi seed được train
từ champion bất biến, ghi vào `shadow_seed_<seed>.pkl`, rồi mới chuyển sang
seed kế tiếp. Không chạy chín MT5 writer song song trên cùng `runner_dir` hoặc
registry. Artifact candidate profile sẽ strip FiveAdam moments sau mỗi run và
ghi `optimizer_state_stripped` vào manifest; bước OOS chỉ đọc artifact, không
được tiếp tục update từ artifact đã strip. Chỉ summary có đủ 3 window × 3 seed,
paired baseline/candidate, unique-position ledger reconcile và pooled/block LCB
mới được đưa vào `verify_rlhf_mt5_shadow.py`; command verifier không có quyền
promote. Trước khi tạo MT5 writer, matrix thực hiện storage preflight theo
`five_rlhf_min_free_disk_bytes` và ghi `storage_preflight` vào summary; nếu
không đủ dung lượng thì fail-closed trước khi tạo candidate hoặc xoá/chạm vào
MT5 runtime.
Để kiểm kê trước khi xin phép xử lý artifact cũ, dùng tool chỉ-đọc:

```powershell
python scripts/audit_rlhf_storage.py `
  --root outputs/rlhf `
  --required-free-bytes 2147483648
```

Tool chỉ phân loại và tính dung lượng; `mutations_performed` luôn là `false`.
Các file `*.incomplete.pkl` và observatory được gắn nhãn `for_review`, không
được tự động coi là có thể xóa vì chúng là evidence forensic.
Đối chiếu:

`decision trace -> intent/ack -> position_ticket -> final unique position -> reward/optimizer receipt`.

RLHF candidate chỉ được xem xét nếu optimizer event cho thấy
`rlhf_preference_eligible_count > 0`, gradient khác 0, không có trace cache miss,
không có duplicate receipt và Full-RL AdamW là writer duy nhất.

### Giai đoạn E — LCB/OOS promotion

Candidate phải vượt baseline theo paired frozen OOS, không negative scope/day
nghiêm trọng hơn, terminal flat, ledger reconcile và không vi phạm DD/risk gate.
Bắt buộc dùng unique completed positions; không dùng duplicate final-close rows.
Pooled LCB và block/day LCB đều phải đạt. Một subgroup positive, equity curve
đẹp hoặc offline preference accuracy cao không đủ.

### Giai đoạn F — shadow rồi canary

Sau OOS pass mới cho shadow live với action không ảnh hưởng execution. Canary
đầu tiên vẫn giữ rollback checkpoint và theo dõi HOLD rate, action mix, PF,
expectancy, LCB, DD, cache/receipt health. Bất kỳ violation nào của causal
attribution hoặc LCB regression đều rollback về source checkpoint.

## Checklist xác nhận sau mỗi run

- `rlhf_preference_dataset` hash, schema version, rater versions được ghi vào manifest.
- Không có `feature_causal=false`, `candidate_only=false` hoặc
  `outcome_audit_only=false` lọt vào queue.
- `rlhf_preference_item_count` khớp số pair được ACK; duplicate pair không tạo update lần hai.
- `rlhf_preference_eligible_count`, loss, gradient và probability chosen được
  ghi trong `direct_rnn_full_rl_batch`.
- Reward model hash, feature schema, reference checkpoint hash, PPO ratio,
  exact KL, value loss, entropy, reward-model score và policy optimizer receipt
  phải cùng nằm trong manifest của candidate.
- Không có update ngoài `full_lifecycle_rl`; không có management-logit credit.
- Baseline/candidate cùng cửa sổ, cùng seed/geometry, unique-position ledger
  reconcile và OOS frozen LCB pass.

## Codex teacher trực tiếp trên lượt MT5 (candidate-only, 2026-09-23)

Profile `configs/he_chuyen_gia_codex_teacher_candidate.yaml` bật frozen
reference capture, tắt reward model RLAIF tuyến tính và tắt failure replay phụ.
Candidate runner phải ghép profile này với
`configs/experiments/codex_teacher_every_close_candidate.yaml`; profile smoke
edge-discovery cũ cố ý đặt `mt5_learning_enabled=false` nên không được dùng cho
Codex teacher. UI tự chọn profile learning-safe này khi ô config trỏ tới Codex
teacher và command-contract self-test khóa mapping đó.
Teacher hiện tại của profile là `gpt-5.6-luna` với `xhigh` reasoning effort.
Runner truyền effort này bằng `model_reasoning_effort=xhigh`; không nhầm với
model ID không tồn tại `gpt-5.5-luna`. Smoke CLI thật đã trả structured receipt
đúng schema và packet hash khớp. Đây mới là kiểm tra tương thích CLI; chưa phải
so sánh chất lượng Luna với Astra trên cùng tập packet.
Nguồn phải là checkpoint `outputs/he_chuyen_gia/kho_chuyen_gia.pkl` có SHA256
khớp `five_rlhf_source_checkpoint_sha256`; checkpoint gốc không bị ghi đè.
Trong một lượt Strategy Tester, runner giữ packet 5 trace của **mọi command** đã
phát (entry, close/manage và HOLD để bảo toàn provenance), rồi dùng
`position_owner_trace_id`/`exit_trace_id` và `position_ticket` để bind đúng trace.
Khi một lifecycle có `final_close_confirmed=true`, entry owner được chấm như
trước; nếu ledger có `exit_trace_id` của một command close/manage khác, command
đó nhận một case Codex riêng. Worker chạy tuần tự để không chiếm 15 giây chờ
action của EA; một lệnh Codex được phép mất tối đa 120 giây. MT5 vẫn tiếp tục
chạy, còn receipt hoàn tất được áp dụng trên luồng bridge trước lần lấy action
kế tiếp. Cuối pass, runner chờ nốt toàn bộ case đã xếp hàng;
report/ledger reconcile vẫn là gate fitness/promotion, không phải điều kiện để
bắt đầu chấm. Profile đặt `five_codex_teacher_max_cases_per_run: 0`, tức không
giới hạn: mọi **action có outcome terminal và trace bất biến** đều được gửi
chấm. HOLD không có outcome kinh tế hoặc CANCEL không tạo lifecycle sẽ
fail-closed, không bịa reward; muốn chấm hai loại này cần một counterfactual
horizon riêng.

Codex nhận state trước quyết định, tên action/action family, market/timing/Wyckoff
features có tên, thông tin request, geometry entry/SL/TP từ ledger, context
`exit_trace_id` và outcome đóng vị thế. Entry giữ feature schema 7 action để
không phá artifact cũ; close/manage dùng schema one-hot đủ 12 action.
Nó trả JSON rubric version cố định: bốn điểm chất lượng, scalar reward,
confidence và rationale dựa trên bằng chứng trước khi cho điểm. Hash packet,
trace, ticket, rubric, phạm vi số và rationale được kiểm tra; receipt ghi file
JSONL append-only riêng cho từng run. File `.status.json` cạnh receipt công bố
live các bộ đếm `submitted`, `pending`, `graded`, `updates`, `skips` và
`failures`. `confidence < 0.55` hoặc thiếu bằng chứng sẽ không cập nhật. Reward
được nhân confidence và scale 0.25, khóa vào đúng state 137-feature + one-hot
action của **target trace**; mọi trace khác bị từ chối. PPO/KL dùng frozen reference và writer
FiveAdam sẵn có, candidate-only. Chế độ mọi lệnh làm số lần gọi Codex tăng theo
số position đóng; backlog `pending` là tín hiệu vận hành phải theo dõi. Nếu
Codex trả `packet_sha256` bị rút gọn hoặc sai ký tự, runner retry tối đa hai
lần với cùng packet bất biến và prompt echo đủ 64 ký tự; mismatch sau retry
cuối vẫn ghi failure append-only và không được canonicalize để vượt owner/hash
guard.

Acceptance live ngày 2026-09-24 dùng output
`outputs/he_chuyen_gia/v193_runs/candidate_codex_every_close_20260924_0230`.
Snapshot đầu tiên ghi `case_limit=0`, `submitted=6`, `graded=1`, `pending=5`,
`updates=1`, `failures=0`. Receipt rubric v2 của ticket `3` khớp packet/trace,
cho reward `-0.62`, confidence `0.97`; reward sau scale là `-0.15035` và
FiveAdam (scope cũ chỉ action-head) xác nhận PPO update đúng 3 tensor với probability shift
`0.01681 < 0.05`. Đây là bằng chứng đường chấm/update đang chạy, không phải
bằng chứng edge hay promotion.

Acceptance action-authority live mới dùng output
`outputs/he_chuyen_gia/v193_runs/candidate_codex_action_authority_20260924_0330`.
Status runtime ghi `mode=per_terminal_action_async`, `case_limit=0`,
`submitted=14`, `graded=3`, `updates=3`, `failures=0`. Hai receipt đầu là
`MARKET_BUY`/`MARKET_SELL` với schema entry 7 action; receipt thứ ba của ticket
`14` là `CLOSE_ALL`, `action_family=management`, schema đủ 12 action,
`teacher_reward=+0.35`, `confidence=0.93`, PPO update=true,
`rlhf_action_surface_count=12`, probability shift `0.00806 < 0.05`. Run được
dừng có chủ đích sau khi đủ bằng chứng wiring; `pending=11` còn lại không được
coi là graded. Đây xác nhận Codex có thể thưởng/phạt đúng trace của action
management, không xác nhận trading edge hay quyền promotion.

Các lượt v1--v3 đã phát hiện hai lỗi làm candidate không đáng tin: critic
baseline có thể đảo dấu punishment, và phạm vi update quá rộng làm policy sụp
về toàn `HOLD`. Đường Codex nay dùng trực tiếp scaled teacher reward làm
advantage có hướng; nhánh RLHF reward model cũ vẫn dùng reward trừ value.
Writer mở một scope bounded gồm `codex_action_residual`,
`codex_action_bias` và `heads.internal_policy_state_net` — latent liên tục nằm
trong chính mạng, không có strategy ID hay router bên ngoài. Snapshot cả tensor
lẫn optimizer state, rồi rollback nếu xác suất action trên owner packet dịch
quá `five_codex_teacher_max_probability_shift`. Receipt ghi scope
`codex_context_action_residual_v3_internal_policy_state_global_bias`, số tensor
latent và cờ latent parameter đã đổi; vì vậy một probability shift đơn thuần
 không còn được gọi là strategy learning.

Line-search của scope này không được phép bỏ latent chỉ vì bước Adam đầy đủ
vượt probability guard. Runtime thử residual fraction giảm dần tới
`0.00048828125` và cả `global_step=0`; nếu một mức dương vẫn nằm trong guard
và giữ đúng hướng reward thì giữ nguyên contextual update, còn chỉ khi tất cả
mức dương bị loại mới fallback về global bias. Receipt/status ghi
`rlhf_codex_global_intervention_residual_fraction`, số lần thử và ba bộ đếm
`shift/direction/nonfinite_rejections` để phân biệt latent update thật với
global-only fallback.

Đối với candidate có Codex teacher, profile active đặt
`five_rlhf_recovery_snapshot_interval_updates: 8`. Candidate/runtime contract
phải công bố lại giá trị này trong
`five_rlhf_recovery_snapshot_interval_updates` và
`codex_teacher_recovery_snapshot_interval_updates`. Snapshot recovery là
artifact nghiên cứu, không promotable và không được dùng làm continuation seed
trừ khi có gate nghiên cứu cho phép; mục đích là không làm mất các weight đã
được update nếu process MT5 kết thúc trước final checkpoint boundary. Sau khi
trainer kết thúc, runner phải đồng bộ alias evaluator vào
`effective.five_rlhf_recovery_snapshot_interval_updates`; nếu runtime cũ thiếu
field này thì giữ nguyên giá trị profile đã ghi trước khi MT5 khởi động.
Recovery serializer chỉ tạm bỏ các Adam moment tensors khi đồng thời có
`five_rlhf_strip_optimizer_state_for_oos=true` và
`five_rlhf_allow_optimizer_state_reinitialization_for_research=true`; state
đang dùng trong process được khôi phục sau khi ghi file. Metadata và phase phải
ghi `optimizer_state_stripped` cùng `optimizer_resume_mode=research_reinitialized`
để artifact này không bị hiểu nhầm là continuation checkpoint đầy đủ.

Đường legacy/context-expert dùng cùng account-impact contract: trước khi ghi
action vào MT5, bridge lưu `codex_decision_account.balance/equity` thô trong
`action_details`, tách khỏi vector account đã chuẩn hoá. Khi final close đến
trước `POST_EXIT_OBSERVATION`, Codex case được giữ trong deferred queue theo
đúng lifecycle owner; runner chỉ submit sau khi parser thấy
`post_exit_hold_r` và `post_exit_horizon_bars`, hoặc ghi rõ `forced_partial` ở
ranh giới terminal. Vì vậy `graded` không còn đồng nghĩa với
`account_impact_confirmed`: chỉ packet có mark được gán duy nhất và
counterfactual hợp lệ mới được mở khoá update teacher; các packet partial vẫn
giữ signed realized-R nhưng không được tuyên bố là account-level confirmation.
Điều kiện chờ này áp dụng cả trace legacy không có callback consume; callback
chỉ dùng để xoá bản sao packet sau khi submit, không được dùng làm điều kiện
bỏ qua `POST_EXIT_OBSERVATION`.

Đối với Direct-RNN specialist, `action_surface_count` trong receipt vẫn giữ
teaching contract legacy (thường là 7) để bảo toàn packet/feature hash, còn
  `direct_rnn_rl_action_probabilities` của trace có thể là 12 output thực thi.
Verifier không được so sánh hai mẫu số khác nhau: teaching surface chỉ dùng
cho owner/scorer, projection surface dùng cho kiểm tra trước/sau update. Nếu
profile bật `direct_rnn_rl_trade_hold_separation_enabled`, cả hai phía phải
dùng cùng `hierarchical_action_probabilities` (không thay bằng softmax phẳng),
nếu không một update rất nhỏ có thể bị đọc thành probability shift giả lớn và
bị rollback. Receipt
phải ghi `rlhf_directional_update`, `rlhf_codex_optimizer_state_reset`, số
trial backtrack và hash parameter trước/sau; chỉ update có hướng đúng và shift
trong guard mới được tính vào `codex_teacher.updates`. Nếu wrapper/observatory
đã loại `DirectRnnForwardTrace` lớn nhưng còn `ModuleActionTrace.hidden`,
learner được phép dựng cache tối thiểu để kiểm chứng action-head; cờ
`rlhf_codex_hidden_fallback=true` phải được ghi lại và chỉ `w_out/b_out/value`
được cập nhật, tuyệt đối không bịa gradient cho recurrent encoder. Nếu cả hai
surface đều thiếu thì vẫn fail-closed như trước.

Candidate v4 được train bằng MT5 Model 1 ngày `2025.02.01`--`2025.02.02`, seed
9723, 32 packet. Report/ledger reconcile; 13 giao dịch; train profit -50.05,
LCB -0.49644R nên fitness không đạt. Codex chấm 2 final-close owner packet,
cho hai punishment -0.20/confidence 0.73 và -0.08/confidence 0.70; cả 2 update
giữ advantage âm (-0.0365 và -0.014). Update cuối có exact KL
3.4578e-05; scope action/residual + latent vẫn bị giới hạn bởi max
probability shift 0.009223 < 0.05. Checkpoint nằm tại
`outputs/rlhf/codex_teacher_mt5_candidate_v4.pkl`; SHA256
`c142a0371669072c18984702d24bb587c45950d46ba97bc2b816745a1ded5561`.

Frozen probe của v4 trên ngày kế tiếp `2025.02.02`--`2025.02.03`, cùng Model 1,
seed 40723 và 32 packet đã chứng minh lỗi collapse được sửa: Codex phase tắt,
optimizer update bằng 0, 32/32 agent-0 action là entry và manifest đánh dấu
`collapsed=false`. Report/ledger reconcile; 29 lifecycle đóng cho net profit
+1600.40, PF 34.03, win rate 90.32%, DD 6.47% và LCB 0.8110R. Tuy nhiên cuối
cửa sổ còn 14 vị thế mở, vì vậy fitness gate đúng cách trả -100/ineligible với
`terminal_open_positions_nonzero:14`. Đây chỉ là bằng chứng retention và tín
hiệu kinh tế sơ bộ, **không** phải OOS đạt chuẩn và không có quyền promote.

Attempt không giới hạn packet trên cùng ngày tiếp tục giữ agent-0 entry đến
packet 376, nhưng chạm `max_wall_sec=600` ở khoảng 20:12 dữ liệu. Manifest còn
`stage=started`, không có report/ledger cuối, nên attempt này bị loại hoàn toàn
khỏi thống kê kinh tế; artifact runtime dở dang đã bị xoá.

Các checkpoint v1--v3 không hợp lệ đã bị xoá; JSONL/manifest forensic vẫn được
giữ để truy vết. Test `test_negative_teacher_reward_never_flips_to_positive_advantage`
khoá lỗi đảo dấu, còn test scope thật khoá phạm vi tensor và probability guard.
File receipt cũ dưới `mt5_v2_runner/Reports/` thuộc hai smoke trước, có thể
trộn run do tên report lặp; không dùng làm bằng chứng promotion.

Trước khi dùng candidate thật: chạy nguồn qua `huan_luyen_he_chuyen_gia_mt5`
với output khác checkpoint nguồn, `persist_checkpoint=True`; kiểm tra
`metrics.codex_teacher_phase` (`submitted == graded + grade_failures + pending`
khi đang chạy; cuối pass phải có `pending == 0`, cùng `updates`, receipt hash và
FiveAdam receipt) và so sánh frozen OOS nhiều ngày với cùng seed/geometry, real ticks,
unique position, expectancy/LCB và drawdown gates. Không dùng profit của lượt
train hay điểm teacher làm bằng chứng edge. Runtime cần bản `codex.exe` của
Codex desktop trên PATH; thiếu CLI hoặc schema thì fail-closed, không fallback
sang reward model khác.

### Codex adaptive hypothesis loop (mọi cấu hình MT5 có khả năng học)

Từ contract `codex_hypothesis_v2`, `five_codex_teacher_auto_connect=true` là
đường mặc định cho các phiên MT5 có checkpoint. Orchestrator tự dò `codex.exe`,
khóa source SHA-256 của checkpoint thực tế và tự cài frozen reference nếu
config cũ chưa khai báo; không có checkpoint, reference hoặc optimizer hook thì
ghi `codex_auto_connection.audit_only=true` thay vì tuyên bố đã học.

Mỗi receipt bắt buộc (khi teacher active) có bốn phần có thể kiểm chứng:

1. `hypothesis`: cơ chế giả định;
2. `prediction`: dấu kỳ vọng của `realized_r`;
3. `experiment`: kiểm thử `outcome_direction_v1`;
4. `verification_status` và `verification_evidence`.

Verifier cục bộ đối chiếu `verification_observed_reward` với outcome immutable
trong packet và chỉ chấp nhận `confirmed` khi dấu dự đoán khớp outcome trong
ngưỡng trung tính ±0.05. `intervention_target_action` phải đúng trace đang
được chấm; `intervention_multiplier` bị giới hạn [0.5, 1.5]. Receipt bị
`rejected`/`inconclusive` không được gọi optimizer. Receipt được xác nhận mới
đi qua FiveAdam scope action/residual + latent policy-state với
probability-shift guard hiện hữu; nếu latent không có gradient thì receipt vẫn
ghi `rlhf_internal_policy_state_parameter_changed=false`, không được giả đổi
thành true.

Contract v2 bổ sung `account_impact`: `realized_r` và `net_pnl` là bằng chứng
immutable của lifecycle. `balance_delta`/`equity_delta` chỉ được gắn khi có
snapshot tài khoản tại quyết định và sau close. Ngoài snapshot, một final
`DEAL_ADD` duy nhất có thể được xác nhận ở scope
`immutable_lifecycle_plus_unique_deal_cashflow` nếu ledger có `deal_ticket`,
không có partial-close và `cashflow_delta = profit + commission + swap` khớp
chính xác với `net_pnl`. Scope này xác nhận cashflow tác động vào tài khoản
của đúng deal, nhưng không tự điền hoặc suy diễn balance/equity snapshot;
receipt phải giữ `cashflow_delta` và `cashflow_deal_ticket`.

Nếu không có snapshot duy nhất hoặc cashflow deal duy nhất thì status là
`partial`, không được suy diễn thành hiểu biết account-level. Khi `partial`
vẫn có `net_pnl` hoặc delta tiền tệ immutable, scorer dùng chúng cùng
`realized_r` như signed lifecycle reward
(`account_impact_reward_source=immutable_lifecycle_partial`); điều này không
nâng packet thành `confirmed` và không cho Codex tự xác nhận account mark.
`post_exit_hold_r` và `post_exit_horizon_bars` tạo
`counterfactual_delta_r = realized_r - hold_r` khi ledger có đủ observation;
scorer đưa tín hiệu account-impact có dấu vào reward, còn receipt sửa delta,
scope hoặc horizon bị verifier reject. Credit được tách theo owner: entry owner
(`target_is_entry_owner=true`) có thể dùng realized lifecycle R kết hợp bounded
context của counterfactual; management/exit owner
(`target_is_entry_owner=false`, `action_family=management`) chỉ dùng
`counterfactual_delta_r` khi counterfactual đã `validated`, và ghi
`account_impact_reward_source=management_counterfactual_delta`. Như vậy
`realized_r` dương của cả vị thế không thể tự biến một lệnh `CLOSE_ALL` thành
update dương nếu đóng lệnh kém hơn giữ thêm. Khi nhiều close xuất hiện cùng một poll,
runtime không gán snapshot chung cho từng packet và hạ bằng chứng về partial.
EA hiện khởi động quan sát hậu đóng cho mọi lifecycle đã có owner trace và
`exit_kind` hợp lệ, gồm cả `INITIAL_SL`, `TRAILING_SL`, `TP`, `TIME_SAFETY`,
`OTHER` và `POLICY_CLOSE_*`. Đây là bằng chứng “nếu tiếp tục giữ lệnh thì giá
đi đâu”, không phải exit-manager action. Route `exit_counterfactual_credit`
vẫn chỉ nhận `POLICY_CLOSE_*` có `exit_trace_id`, còn entry owner vẫn nhận
final realized-R đúng một lần.

Để không phụ thuộc hoàn toàn vào snapshot bridge cấp request, EA V193 ghi thêm
hai giá trị account sau final `DEAL_ADD` vào suffix append-only của
`geometry_entry_pipe` (balance và equity). Python đọc suffix này vào
`MT5PositionLifecycle.account_context_after_close` và ưu tiên nó trước
bridge-level fallback; ledger cũ không có suffix vẫn giữ semantics partial.
Parser còn nhóm các final close theo khóa thời điểm sự kiện cùng với balance và
equity. Nếu từ hai position trở lên dùng cùng khóa này, metadata ghi
`account_mark_attribution=ambiguous_same_close_batch`: snapshot vẫn được giữ để
audit, không được chia snapshot chung thành nhiều delta giả. Trường hợp đó
vẫn có thể được xác nhận ở scope cashflow nói trên, vì mỗi `deal_ticket` có
cashflow immutable riêng; UI tách riêng `account_cashflow_confirmed` và
`account_mark_confirmed` để không nhầm hai loại bằng chứng.
Smoke sau khi compile lại EA (`compile_account_mark_v17.log`, 0 errors/0
warnings) tại
`outputs/he_chuyen_gia/v193_runs/account_mark_ledger_smoke24_20261001_062100`
đã ghi `submitted=5`, `graded=4`, `hypotheses_confirmed=4`, `updates=4`,
`account_impact_records=4`, `account_impact_confirmed=4`,
`account_impact_partial=0`, `rlhf_parameter_changed=4`; case thứ năm bị
Codex timeout và được ghi failure riêng, không được dùng để tuyên bố update.
Run này có `net_pnl=-89.01`, `profit_factor=0`, `win_rate=0` và chỉ có 5
closed trades, nên nó xác nhận per-deal account attribution chứ chưa xác nhận
edge hay cải thiện kinh tế.

Telemetry tách rõ `hypotheses`, `hypotheses_confirmed` và
`interventions_applied`; vì vậy log Codex không còn được coi là bằng chứng
 model đã đổi hành vi nếu không có PPO receipt và xác nhận probability shift.
Smoke fine-guard ngày `2026-10-01`
(`account_impact_latent_fineguard_20261001_042120_16`) đã xác nhận
`submitted=graded=hypotheses_confirmed=updates=interventions=3`, `failures=0`,
`rlhf_internal_policy_state_changed=3`, residual fraction `0.015625` cho cả
ba case, max probability shift `0.03455 < 0.05`, và
`report_ledger_reconciled=true`. Đây là bằng chứng đường receipt → FiveAdam →
latent logits đã chạy; cùng run vẫn có net P/L âm `-48.47` trên 3 giao dịch,
nên chưa phải bằng chứng edge, cải thiện lợi nhuận hay promotion.

### Đồng bộ delayed `POST_EXIT_OBSERVATION` với Codex teacher

EA có thể ghi final close trước, rồi mới ghi `POST_EXIT_OBSERVATION` sau đủ
`exit_counterfactual_horizon_bars`. Vì vậy Codex hypothesis mode không được
submit case ngay tại final close nếu packet còn thiếu hold label. Runner giữ
case và packet owner bất biến trong deferred queue; khi ledger có observation,
nó thay lifecycle cũ bằng bản refresh theo identity position/owner/close/R rồi
mới build receipt. Ở terminal boundary, case chưa đạt horizon vẫn được ghi
`deferred_forced_partial` để phân biệt rõ với counterfactual đã kiểm chứng.
Facade shared cũng phải expose `pop_five_entry_lifecycle_packet`, để entry
packet không bị consume trước khi label hậu đóng xuất hiện. Các trường
`deferred`, `deferred_released`, `deferred_pending` và
`deferred_forced_partial` nằm trong status recovery whitelist nên save-and-stop
không làm mất dấu đường `submitted → POST_EXIT_OBSERVATION → receipt → update`.

Smoke sau sửa identity refresh ngày `2026-10-01`
(`outputs/he_chuyen_gia/v193_runs/codex_counterfactual_validated_smoke_20261001_132500`)
dùng Desktop Codex `codex-cli 0.158.0-alpha.2.1`, model `gpt-5.6-luna`,
reasoning `xhigh`: `submitted=8`, `graded=8`, `hypotheses_confirmed=8`,
`updates=8`, `interventions_applied=8`, `failures=0`, `deferred=8`,
`deferred_released=8`, `deferred_pending=0`, `deferred_forced_partial=1`.
Bảy receipt có `counterfactual_status=validated` và horizon 12; một receipt
cuối chạy tới terminal trước đủ horizon nên giữ `not_available` và được đánh
dấu forced partial. Mọi update đều có `rlhf_ppo_update=true`, parameter hash
trước/sau khác nhau và probability shift tối đa `0.04926`, dưới guard `0.05`.
Status và runtime contract đồng thời ghi `counterfactual_validated`/
`counterfactual_unavailable`; mỗi update receipt giữ `counterfactual_status`,
`counterfactual_horizon_bars`, `counterfactual_hold_r` và
`counterfactual_delta_r`, để phân biệt reward quan sát trực tiếp với reward có
đối chứng hold đã được xác nhận.
Run có `net_pnl=-1.72`, expectancy âm nhẹ và `promotable=false`, do đó chỉ
chứng minh đường causal teacher → action-head đã hoạt động, chưa chứng minh
lợi nhuận hay được phép promote.

Smoke current-code sau khi sửa retry hash ngày `2026-10-01`
(`outputs/he_chuyen_gia/v193_runs/codex_hash_retry_v2_metrics_20261001_133038`)
ghi `submitted=graded=hypotheses_confirmed=updates=interventions=8`,
`failures=0`, `deferred=8`, `deferred_released=8`, `deferred_pending=0`,
`counterfactual_validated=7`, `counterfactual_unavailable=1`,
`account_impact_confirmed=8`, `rlhf_parameter_changed=8` và max probability shift
`0.0499455` dưới guard
`0.05`. Một reward âm có advantage âm và giảm xác suất action; một reward
dương có advantage dương và tăng xác suất action. Run này vẫn có
`net_profit=-169.91`, `profit_factor=0.13`, `fitness_eligible=false` và
`promotable=false`; đây là bằng chứng plumbing đã không còn mất case vì hash,
không phải bằng chứng lợi nhuận.

Smoke sau khi tách credit management ngày `2026-10-01`
(`outputs/he_chuyen_gia/v193_runs/codex_management_credit_v1_20261001_145000`)
đã qua preflight Desktop Codex và ghi `submitted=graded=hypotheses_confirmed=
updates=interventions=7`, `failures=0`, `account_impact_confirmed=7`,
`rlhf_parameter_changed=7`, max probability shift `0.0499663`. Hai lifecycle
có counterfactual validated, còn năm lifecycle tới terminal trước horizon nên
`counterfactual_unavailable=5`; hai receipt `CLOSE_ALL` vì vậy dùng fallback
`post_close_account_mark`, không được tính là bằng chứng đã chạy nhánh
`management_counterfactual_delta`. Run research-only có `profit=13.47`,
`PF=1.20`, nhưng chỉ có 5 position samples và expectancy LCB `-0.4022`, nên
fitness/promotable vẫn false. Regression phase trong
`tests/test_codex_teacher.py` xác nhận khi management packet có
`realized_r=0.75`, `hold_r=0.875`, update receipt ghi
`account_impact_reward=-0.125`, source
`management_counterfactual_delta` và signed learner reward âm.

Một smoke 32 packet để kéo dài horizon đã bị storage gate chặn trước khi clone
(`free=2.083 GB`, yêu cầu `2.139 GB`, reserve `1.5 GB`); không tạo MT5 session,
không nới reserve và không xoá artifact để vượt gate. Cần chạy lại khi dung
lượng trống đủ lớn trước khi dùng runtime smoke để kết luận management
counterfactual đã xuất hiện trong MT5 thật.

Smoke hậu kiểm sau khi nối signed reward cho partial lifecycle:
`outputs/he_chuyen_gia/v193_runs/account_impact_partial_fix_smoke35_20261001_045535`.
Run này có `submitted=graded=hypotheses_confirmed=updates=interventions=11`,
`failures=0`, `pending=0`, `account_impact_records=11`, trong đó
`account_impact_confirmed=5` và `account_impact_partial=6`. Cả sáu partial
case đều có `account_impact_reward_source=immutable_lifecycle_partial`; tổng
signed account reward là `-1.18893051195218`, không còn bị bỏ qua như artifact
fine-guard trước đó. `rlhf_parameter_changed=11`,
`rlhf_probability_shift_count=11`, `rlhf_max_probability_shift=0.04954484`
và `report_ledger_reconciled=true`. Đây là bằng chứng trực tiếp rằng partial
reward đi tới FiveAdam và làm đổi action probability trong cùng run.
Economic audit của smoke vẫn là `closed_trades=10`, `net_pnl=-63.58`,
`profit_factor=0.4267`, `win_rate=0.30`, nên status vẫn là research-only:
plumbing đã xác nhận, còn edge/lợi nhuận/OOS/promotion chưa được xác nhận.

Long smoke hậu kiểm cùng seed, giới hạn `160` decision packets và bỏ paired
validation để cô lập learner:
`outputs/he_chuyen_gia/v193_runs/account_impact_partial_fix_long160_20261001_050608`.
Runtime ghi `submitted=graded=hypotheses_confirmed=updates=interventions=27`,
`failures=0`, `pending=0`, `account_impact_records=27`,
`account_impact_confirmed=21`, `account_impact_partial=6`,
`account_impact_reward_sum=-9.71963830481573`,
`rlhf_parameter_changed=27`, `rlhf_probability_shift_count=27` và
`rlhf_max_probability_shift=0.04968874`. Đây là bằng chứng mạnh hơn smoke 35
packet rằng reward sau close được dùng lặp lại trong lúc train.

Economic diagnostics của long smoke vẫn âm: `closed_trades=24`,
`net_pnl=-217.27`, `profit_factor=0.3903`, `win_rate=0.25`,
`max_equity_drawdown_pct=2.6148`, `hold_share=0.73`. Entry-owner attribution
cho thấy lỗ phân tán trên MARKET_BUY, MARKET_SELL, BUY_STOP, SELL_STOP và
SELL_LIMIT; không có đủ bằng chứng để phạt cứng một hướng BUY/SELL hoặc tắt
exploration. Run này xác nhận closed-loop plumbing, nhưng edge/OOS và
promotion vẫn chưa đạt.

Smoke mới sau khi khóa attribution và telemetry action-head:
`outputs/he_chuyen_gia/v193_runs/codex_account_telemetry_smoke2_20261001_113000`.
Preflight dùng Codex Desktop `0.158.0-alpha.2.1`, model `gpt-5.6-luna`/
`xhigh`; runtime ghi `submitted=graded=hypotheses_confirmed=updates=interventions=4`,
`failures=0`, `account_impact_confirmed=2`, `account_impact_partial=2`,
`rlhf_parameter_changed=4`, `rlhf_probability_shift_count=4` và
`rlhf_max_probability_shift=0.049906`. Cả bốn update đều ghi norm scoped
action-head dương; reward âm làm giảm probability action được chấm, reward dương
ở ticket 4 làm tăng probability. Run này vẫn có `net_profit=-24.09`,
`profit_factor=0.60`, `win_rate=0.25`, `fitness_eligible=false` và
`promotable=false`; đây là bằng chứng đường receipt → verifier → FiveAdam →
logit, không phải bằng chứng edge hay cải thiện kinh tế.

Sau long smoke, causal gate của account mark đã được sửa để tách hai trạng thái
khác nhau: lifecycle đã được nhìn thấy trong snapshot nhưng chưa route được
owner vẫn được giữ để retry provenance, đồng thời không còn được tính lại như
`unsettled` vô hạn để chặn một close mới duy nhất. Vì vậy một account mark chỉ
được gắn vào lifecycle mới nếu snapshot hiện tại có đúng một final identity
chưa từng quan sát; test hồi quy là
`test_stale_unowned_close_does_not_block_new_unique_account_mark`. Nhiều final
close xuất hiện lần đầu trong cùng một poll vẫn giữ `partial` cố ý, vì một
bridge-level mark không đủ thông tin để chia delta cho từng action. Muốn nâng
trường hợp đó lên `confirmed` cần account mark theo transaction/deal trong EA
ledger, không được nới guard bằng cách nhân snapshot chung cho nhiều packet.

Đối chứng A/B để tách Codex khỏi optimizer chung:
`outputs/he_chuyen_gia/v193_runs/account_impact_no_codex_control160_20261001_052541`
dùng cùng seed `0`, cùng source checkpoint, cùng ngày và cùng giới hạn `160`
decision packets như long smoke ở trên, nhưng profile
`reasoning_rl_v1_active_candidate_no_codex.yaml` tắt hẳn
`five_codex_teacher_enabled` và `five_codex_teacher_auto_connect`. Contract cuối
ghi `runtime_status=completed`, `codex_teacher_phase.enabled=false`,
`submitted=graded=hypotheses_confirmed=updates=interventions=0` và không có
Codex receipt; ngược lại learner chung vẫn có `ppo_updates=7`,
`reasoning_optimizer_updates=85`, tổng `optimizer_update_count=92`. Vì vậy
`optimizer active` hoặc số PPO tổng quát không được dùng làm bằng chứng Codex
đã chấm hay đã sửa action head.

Run enabled tương ứng có `submitted=graded=hypotheses_confirmed=updates=interventions=27`,
`rlhf_parameter_changed=27`, `rlhf_probability_shift_count=27` và
`rlhf_max_probability_shift=0.04968874`; receipt ghi rõ `teacher_reward` âm,
`advantage` âm, hash parameter trước/sau, scope trainable và xác suất action
trước/sau. Đây là khác biệt kỹ thuật trực tiếp giữa hai nhánh: Codex reward đã
đi vào learner ở enabled run, còn control không có đường Codex.

Kết quả kinh tế của một cặp A/B này vẫn chưa đạt edge: enabled có
`closed_trades=24`, `net_pnl=-217.27`, `profit_factor=0.3903`, `win_rate=0.25`,
`max_equity_drawdown_pct=2.6148`; control có `closed_trades=38`,
`net_pnl=-569.06`, `profit_factor=0.0638`, `win_rate=0.1316`,
`max_equity_drawdown_pct=5.7772`. Chênh lệch này chỉ là tín hiệu trên một
seed/cửa sổ và số lifecycle khác nhau, không được diễn giải là Codex đã tạo
lợi nhuận hay đã đủ điều kiện promote. Cần lặp lại nhiều seed/cửa sổ frozen
OOS, đồng thời giữ nguyên reconciliation và probability guard.

Candidate hiện dùng `five_codex_context_similarity_threshold=0.90` và
`five_codex_context_support_floor=0.50`. Do support được chuẩn hóa sau
threshold, prototype residual chỉ tái sử dụng khi cosine thực tế tối thiểu
`0.95`; state xa vẫn nhận global bias bounded nhưng không nhận residual theo
context. Smoke `codex_context_reuse_090_long_20261001_200000` ghi `6/6`
receipt hợp lệ, `6` optimizer update và `4/6` update tái sử dụng prototype
(`prototype_count=2`). Run dài
`codex_context_reuse_default_20261001_210000` ghi `20/20` receipt hợp lệ,
`20` update, `11/20` reuse và `prototype_count=9`; max probability shift
`0.04987`. Run bị wall guard, có LCB âm và vẫn `promotable=false`, nên đây
chỉ là bằng chứng context-credit đã có reuse, không phải edge.
Smoke non-metrics `codex_context_reuse_persisted_20261001_220000` đã persist
checkpoint với `5/5` update, `3/5` reuse; reload artifact xác nhận
`prototype_count=2`, threshold `0.90`, action surface bật và bias/residual khác
zero. Smoke vẫn `net_pnl=-58.90`, `profit_factor=0.493`, nên đây là bằng chứng
persistence/causal update, không phải edge hay promotion.
Tiêu chí nối nhân quả runtime được tách riêng khỏi bộ đếm optimizer: status
phải ghi `codex_causal_links_observed` và `codex_causal_updates_linked`, còn
`last_codex_causal_link` phải có `causal_link=barrier_to_next_action` cùng
sequence/trace/action kế tiếp. Runner chỉ tạo link sau khi
`online_barrier_passed`; update hợp lệ nhưng barrier timeout hoặc chạy async
không được tự nhận là action kế tiếp đã dùng state mới. Khi policy facade có
runtime hash, `codex_causal_hash_matches` phải tăng và link phải ghi
`policy_parameter_hash_matches_update=true`; `codex_causal_hash_unknown` chỉ
là thiếu bằng chứng tại boundary, không phải pass.

Smoke MT5 sau bản sửa hierarchical projection
`outputs/he_chuyen_gia/v193_runs/specialist_metrics_only_low_disk_postfix_20261004_050000`
đã xác nhận đường chạy thật: Codex Desktop `codex-cli 0.160.0`, model
`gpt-5.6-luna/xhigh`, preflight structured JSON pass, `submitted=4`,
`graded=4`, `hypotheses_confirmed=4`, `updates=3`, `interventions_applied=3`,
`failures=0`, `parameter_changed=3`, probability shift lớn nhất `0.0036286`
trong guard `0.05`. Ba receipt dùng
`frozen_forward_action_head_projection_v1_hierarchical_trade_hold_v1`; một
receipt bị reject fail-closed vì không đạt hướng giảm của reward âm. Run là
`metrics-only`, không tạo checkpoint và bốn account-impact record vẫn ở mức
`partial`, nên đây là bằng chứng Codex đã grading và cập nhật Direct-RNN trong
MT5 thật, chưa phải bằng chứng profitability, post-close counterfactual hoàn
chỉnh cho mọi case, OOS hay promotion.
