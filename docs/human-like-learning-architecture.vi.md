# Kiến trúc học giống con người cho Self-Growing Trader

Tài liệu này mô tả mục tiêu nâng cấp model từ kiểu `mutate -> MT5 chấm -> promote/reject` sang kiểu học gần con người hơn:

```text
trải nghiệm -> hiểu lỗi -> gán trách nhiệm -> liên tưởng -> tạo giả thuyết -> replay trong trí nhớ -> consolidate -> MT5 xác nhận -> mới ghi winner.pkl
```

Đây không phải mô phỏng não người đầy đủ. Mục tiêu thực tế là tạo một hệ trading học ổn định hơn, biết nhớ lỗi cụ thể, liên tưởng tình huống tương tự, dung hợp bài học thành giả thuyết, luyện lại trước khi chạy MT5 và không ghi đè model tốt bằng trial hồi quy.

## Trạng thái hiện tại

Hệ hiện có một số thành phần gần sinh học:

- `ExperienceSuperNetwork`: nhớ transition, causal memory và query ký ức gần giống theo market/account/position context.
- `StepTrace`: lưu action, output, confidence, account context, position context, `module_action_traces`, `active_neuron_ids`, `active_edge_ids`.
- Legacy recurrent module: có `w_rec`, `previous_hidden` và `memory_decay`, nên có liên kết hồi tiếp ngắn hạn.
- M1 legacy: có `working_memory`, predictor, critic, selector và hypothesis bank.
- Graph M1: có neuron, edge, recurrent edge, activation trace, graph plasticity và candidate-only structural growth.
- `ReasoningGraphSystem`: workspace chung cho `curiosity`, `development`, `risk`, `action_head`, `utility`.
- `absorb_rejection_feedback()`: khi trial bị reject, baseline học bài học hành vi có clamp thay vì copy topology lỗi.
- Targeted mutation: `mutation_sigma` chỉ tác động lên action/connection có bằng chứng gây lỗ từ trace hoặc causal memory xấu.
- `train-to-goal` đã có non-regression guard: tắt promotion gate cứng nhưng không ghi đè `winner.pkl` bằng trial tệ hơn baseline theo `goal_progress_score`.
- `MistakeEpisode`/`MistakeEpisodeStore`: lưu lỗi có schema version, trace/action/market/account/position key, neuron/edge active, lifecycle, tag evidence và support count; transition âm và rejected trial đều ghi episode vào artifact `winner.pkl`.

Các thành phần này tạo nền tảng, nhưng chưa đủ để gọi là học như người.

## Khoảng thiếu chính

### 1. Episode chưa đủ giàu ngữ nghĩa

Hệ đã có `MistakeEpisode` nền tảng để mô tả một lỗi như người nhớ:

```text
khi nào xảy ra
trạng thái thị trường/account/position
action nào được chọn
neuron/edge nào active
lệnh được MT5 chấp nhận hay bị reject
kết quả vòng đời position
lý do nghi ngờ gây lỗi
đề xuất sửa
độ tin cậy của bài học
```

Phần còn thiếu là dùng episode này cho analogy, hypothesis và sleep consolidation đầy đủ. `MistakeReplayEngine` đã có bản nền tảng candidate-only, nhưng chưa aggregate theo hypothesis hoặc reject candidate bằng replay regression threshold.

### 2. Liên tưởng hiện mới là similarity, chưa phải analogy

Hiện memory query chủ yếu dựa trên vector similarity. Điều đó trả lời câu hỏi: "state này gần giống state nào?".

Con người còn dùng analogy: "hai tình huống này khác feature thô nhưng cùng cấu trúc lỗi". Ví dụ:

```text
BUY khi drawdown cao
SELL khi spread cao
giữ lệnh sau khi gần TP rồi quay đầu
```

Ba lỗi này có thể cùng thuộc concept lớn hơn: `stress_or_patience_failure`.

### 3. Dung hợp hiện là cluster, chưa phải concept fusion

`MemoryCluster`, `StrategyMemory` và `MetaStrategyMemory` đã gom ký ức thành pattern. Nhưng chưa có lớp tạo giả thuyết mới từ nhiều bài học:

```text
stress_entry + high_spread + loss_after_entry
=> hypothesis: giảm entry gate khi spread và drawdown cùng cao
```

Dung hợp kiểu người cần biến nhiều episode thành một khái niệm/hypothesis có thể test.

### 4. Mistake replay mới ở mức nền tảng

Journal replay hiện thiên về hấp thụ positive transition hoặc khôi phục kinh nghiệm. `MistakeReplayEngine` đã thêm replay lỗi candidate-only:

```text
lấy episode thua
chạy lại policy trên state đó
xem policy còn chọn action sai không
nếu còn, phạt đúng neuron/edge/action trước khi tốn một MT5 trial
```

Replay hiện đã phạt candidate nếu lặp lại action lỗi và reward nhẹ action an toàn hơn. Phần còn thiếu là replay regression gate, aggregate theo hypothesis và sleep phase dùng replay như tiêu chí trước MT5.

### 5. Chưa có sleep/consolidation phase rõ

Con người không biến mọi bài học mới thành thói quen ngay. Hệ cần một pha offline:

```text
collect episodes
deduplicate
associate concepts
fuse hypotheses
mistake replay
homeostasis clamp
candidate checkpoint
MT5 validation
promote nếu tốt hơn
```

Hiện đã có promotion/rollback và non-regression guard, nhưng chưa có pha "ngủ" riêng trước khi ghi `winner.pkl`.

### 6. Chưa có liên kết trễ nhiều thang thời gian

Hiện có recurrent 1 bước và memory state. Chưa có delay edge rõ ràng:

```text
delay 1 bar
delay 5 bars
delay 20 bars
delay 100 bars
```

Trading cần multi-scale memory vì lỗi có thể xuất hiện sau nhiều bar: vào lệnh đúng nhưng quản lý sai sau 20 bar, hoặc overtrade tích lũy cả ngày.

### 7. Homeostasis chưa đủ mạnh

Đã có clamp, nhưng cần cơ chế cân bằng kiểu hệ thần kinh:

- Không để một episode đơn lẻ làm lệch action quá mạnh.
- Action/connection đang tốt cần được freeze hoặc giảm learning rate.
- Học từ lỗi lặp nhiều lần mạnh hơn lỗi đơn lẻ.
- Drawdown tăng mạnh phải làm giảm plasticity entry.
- Khi confidence thấp, sửa nhẹ và cần replay/MT5 xác nhận.

## Kiến trúc mục tiêu

```text
MT5 lifecycle ledger
  -> MistakeEpisodeStore
  -> AssociativeConceptGraph
  -> HypothesisMemory
  -> MistakeReplayEngine
  -> SleepConsolidationEngine
  -> CandidatePolicy
  -> MT5 validation
  -> winner.pkl nếu non-regression + goal progress tốt hơn
```

## Thành phần đề xuất

### MistakeEpisode

Một episode lỗi hiện là object có version/schema rõ trong `src/self_growing_trader/mistake_memory.py`:

```text
episode_id
schema_version
trace_id
source
raw_obs
market_key
account_key
position_key
action_index
action_name
action_details
selected_output
confidence
risk_pct
sl_atr/tp_atr/trailing_stop_atr/partial_close_pct/management_mode/target_slot
module_action_traces
active_neuron_ids
active_edge_ids
graph_edge_gates
graph_region_activity
positions_before
positions_after
execution_details
mfe_r/mae_r/giveback_r/bars_since_mfe/close_reason
reward
td_delta
prediction_error
realized_r
net_profit_delta
drawdown_delta
spread_points
mistake_tags
tag_evidence
suspected_causes
proposed_repairs
severity
confidence_score
support_count
contradiction_count
created_at
last_seen_at
```

`MistakeEpisode` không thay thế journal. Journal là log sự kiện; episode là bản nén có ngữ nghĩa để học. `MistakeEpisodeStore` index theo `trace_id`, `action_index`, `mistake_tags`, context bucket và pattern key; duplicate sẽ merge support count để lỗi lặp có tác động mạnh hơn lỗi đơn lẻ.

### Mistake tags

Tag lỗi nên tách rõ nhóm:

```text
wrong_direction
late_entry
stress_entry
high_spread_entry
overtrading
bad_risk_pct
sl_too_tight
tp_too_far
near_tp_giveback
missed_partial
missed_trailing
bad_target_slot
held_loser_too_long
closed_winner_too_early
execution_reject
no_trade_starvation
immediate_exit_loop
ledger_integrity_bad
```

Tag phải xuất phát từ evidence MT5/ledger/trace, không chỉ từ phỏng đoán.

### AssociativeConceptGraph

Graph khái niệm nối episode, tag, action, condition và outcome:

```text
episode -> mistake_tag
mistake_tag -> concept
concept -> hypothesis
condition -> concept
action -> concept
concept -> repair
```

Ví dụ:

```text
high_spread_entry + stress_entry -> bad_entry_quality
near_tp_giveback + missed_trailing -> exit_management_failure
overtrading + drawdown_rising -> self_control_failure
```

Graph này giúp liên tưởng theo nguyên nhân thay vì chỉ cosine similarity.

### HypothesisMemory

Mỗi hypothesis là một bài học có thể test:

```text
hypothesis_id
condition
suspected_cause
target_action
target_connections
repair_type
repair_strength
expected_effect
support_episodes
contradiction_episodes
replay_score
mt5_score_delta
status: candidate|replay_passed|mt5_passed|rejected|retired
```

Ví dụ:

```text
IF drawdown > 5% AND spread_percentile > 80 AND action in entry_actions
THEN reduce entry output bias and increase HOLD/CLOSE management bias
```

### MistakeReplayEngine

Replay engine trong `src/self_growing_trader/mistake_replay.py` dùng episode xấu để kiểm tra candidate trước MT5:

```text
for episode in recent_bad_episodes:
    run policy.act(episode.raw_obs/account_context/positions)
    compare new action with old bad action
    score whether risk/action improved
    if repeated mistake:
        apply targeted penalty to active connection/action
```

Replay không được tự deploy. Nó chỉ là filter và training nội bộ trước MT5.

Trạng thái hiện tại:

- `propose()` chạy replay trên candidate copy, không mutate baseline.
- Replay dựng lại raw observation/account/position context từ `MistakeEpisode`.
- Nếu candidate vẫn chọn action lỗi cũ, replay phạt action row, hidden contribution hoặc graph edge/node active có attribution.
- Nếu candidate chọn action an toàn hơn, replay reward nhẹ action mới.
- Kết quả replay được ghi vào `growth_events` và train history dưới `mistake_replay`.
- Chưa có threshold reject candidate trước MT5 và chưa aggregate replay score theo hypothesis.

### SleepConsolidationEngine

Sleep phase gom bài học trước khi ghi model:

```text
deduplicate episodes
update concept graph
fuse hypotheses
run mistake replay
apply homeostasis
create candidate checkpoint
run MT5 validation
promote only if non-regression
```

Sleep phase nên chạy ở:

- Cuối mỗi `train-to-goal` round.
- Sau N rejected trials.
- Trước khi overwrite `winner.pkl`.
- Khi `outcome_starvation` cao.

### Multi-scale delay links

Graph edge nên hỗ trợ:

```text
delay_steps: int
delay_buffer: fixed ring buffer
decay: float
kind: recurrent|memory|predictive|gating
```

Các delay hữu ích:

```text
1 bar: phản xạ ngắn
5 bars: micro trend
20 bars: position management
60 bars: session behavior
240 bars: regime memory
1440 bars: day-level cadence/risk memory
```

Không nên bật tất cả ngay. Bắt đầu bằng graph backend và chỉ cho một số region:

- memory
- predictor
- critic
- risk_gate
- output management

### HomeostasisGuard

Guard cân bằng cần chặn học quá tay:

```text
max_weight_delta_per_action_per_round
max_bias_delta_per_action_per_round
max_graph_edge_delta_per_round
freeze_positive_action_paths
reduce_plasticity_after_large_drawdown
require_repeated_support_for_large_update
rollback_if_replay_regresses
rollback_if_mt5_goal_progress_regresses
```

## Luồng học mục tiêu

### Khi MT5 tạo outcome xấu

```text
1. Parse ledger/lifecycle.
2. Tìm trace_id.
3. Lấy StepTrace gốc.
4. Tạo MistakeEpisode.
5. Gắn mistake_tags.
6. Lưu episode vào ExperienceSuperNetwork và MistakeEpisodeStore.
7. Update AssociativeConceptGraph.
8. Nếu nhiều episode cùng concept, tạo hoặc tăng confidence hypothesis.
```

### Khi propose trial mới

```text
1. Copy baseline thành candidate.
2. Apply causal memory plasticity nhẹ.
3. Query bad episodes theo context hiện tại.
4. Query concept graph để lấy hypothesis liên quan.
5. Apply targeted mutation lên action/connection có evidence.
6. Apply structural graph growth nếu need signal đủ mạnh.
7. Run mistake replay.
8. Apply homeostasis clamp.
9. Chỉ đưa candidate qua MT5 nếu replay không hồi quy.
```

### Khi trial bị reject

```text
1. Không copy topology lỗi vào baseline.
2. Tạo reject lesson.
3. Gắn lesson vào rejection_memory.
4. Phạt trực tiếp active edge/neuron hoặc hidden->action connection.
5. Tạo hypothesis "avoid this pattern".
6. Tăng outcome_starvation có clamp.
7. Giữ baseline tốt hơn.
```

### Khi trial tốt hơn

```text
1. MT5 report + execution ledger phải reconciled.
2. goal_progress_score phải tốt hơn baseline.
3. Graph candidate outcome promoted.
4. Concept/hypothesis support tăng.
5. Consolidate positive action paths.
6. Ghi winner.pkl.
```

## Nguyên tắc an toàn

- MT5 report/ledger là bằng chứng cuối, không dùng replay nội bộ để deploy.
- Không ghi đè `winner.pkl` bằng trial hồi quy.
- Không sửa toàn mạng khi chỉ một action gây lỗi.
- Không copy topology của candidate lỗi vào baseline.
- Không dùng raw balance/equity làm durable memory key.
- Không trộn memory schema cũ với schema mới.
- Không để graph experimental thành production default khi chưa có paired MT5 A/B.
- Không tăng `mutation_sigma` để bù thiếu hiểu lỗi; tăng chất lượng episode/replay/hypothesis trước.

## Thứ tự ưu tiên

1. `MistakeEpisode` và tag lỗi.
2. Replay lỗi nội bộ.
3. HypothesisMemory.
4. Sleep consolidation.
5. HomeostasisGuard.
6. AssociativeConceptGraph.
7. Multi-scale delay links.
8. Paired MT5 A/B validation.

## Nâng cấp liên quan

Các phần dưới đây không phải core learning, nhưng cần đi cùng để hệ học giống con người không bị lệch hoặc khó kiểm chứng.

### MT5 lifecycle và ledger

Human-like learning phụ thuộc mạnh vào dữ liệu vòng đời lệnh. Nếu ledger thiếu trace hoặc lifecycle không đủ chi tiết, hệ sẽ gán lỗi sai neuron/edge. Cần đảm bảo MT5 bridge ghi đủ:

- `trace_id` cho intent/action.
- ticket/slot/side/volume/entry/SL/TP.
- accepted/rejected/fill status.
- MFE/MAE, near TP/SL, giveback, duration.
- close reason và realized R.
- spread, drawdown, exposure tại thời điểm action.
- execution failure và immediate exit ratio.

### Policy artifact schema

Artifact `.pkl` cần thêm các vùng mới nhưng vẫn load được model cũ:

- `mistake_episode_store`.
- `associative_concept_graph`.
- `hypothesis_memory`.
- `sleep_consolidation_state`.
- `homeostasis_state`.
- delay-edge buffers nếu graph bật multi-scale delay.

Mọi field mới phải có migration trong `ensure_compatible()` hoặc load path tương đương.

### Journal và replay schema

Journal hiện là nguồn khôi phục transition. Với human-like learning, journal cần phân biệt:

- raw transition journal.
- MT5 online journal.
- mistake episode journal.
- sleep consolidation audit.
- hypothesis replay result.

Không được trộn schema cũ với schema mới nếu không có version gate.

### Fitness và non-regression

Khi model học kiểu người, cần nhiều điểm kiểm tra hơn fitness scalar:

- `goal_progress_score`.
- net profit.
- profit factor.
- expectancy LCB.
- drawdown.
- sample count.
- ledger integrity.
- execution failure rate.
- immediate exit ratio.
- repeated mistake rate trên replay.

Fitness scalar vẫn hữu ích, nhưng không đủ để quyết định ghi `winner.pkl`.

### UI/CLI

Người dùng cần thấy model đang học gì, không chỉ thấy profit/PF:

- top mistake tags.
- số episode xấu mới.
- hypothesis đang test.
- replay pass/fail.
- homeostasis clamp count.
- reason không promote.
- `goal_progress_score` baseline/trial.
- sleep consolidation summary.

Nếu UI không hiển thị các thứ này, người dùng sẽ cảm giác "nó sửa càng lúc càng tệ" mà không biết nó đang học lỗi nào.

### Observability và audit

Mỗi lần hệ sửa action/connection cần có audit:

```text
source episode/hypothesis
target action
target neuron/edge/connection
old value
new value
repair strength
homeostasis clamp
replay result
MT5 result
```

Audit này giúp phân biệt "model học đúng nhưng thị trường khó" với "model đang tự phá".

### Validation pipeline

Human-like learning cần validation nhiều tầng:

```text
unit test -> replay test -> fake MT5 test -> real MT5 smoke -> paired MT5 A/B -> walk-forward
```

Không được kết luận một nâng cấp tốt chỉ vì unit test pass.

### Cleanup và tài liệu

Khi thêm hệ mới, phải xóa hoặc đánh dấu deprecated các đường cũ nếu không còn dùng:

- random global mutation path.
- replay không schema gate.
- docs nói `train-to-goal` ghi mọi best trial.
- UI text gây hiểu nhầm promotion gate.
- tests cũ kỳ vọng tắt gate là promote luôn.
