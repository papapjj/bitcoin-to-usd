# Direct-RNN Retention V2 — replay-held-out audit (schema 5)

## Vấn đề được sửa

Queue Full-RL cũ chỉ giữ tối đa 64 item chưa học. Khi một item đã vào batch,
nó bị lấy khỏi queue và không còn là neo cho các batch sau. AdamW vì thế có
thể học một edge ngắn ở đoạn đầu rồi ghi đè nó khi context/action thắng gần
nhất đổi. Softmax phẳng trên `HOLD + 6 entry` còn tạo thiên lệch cấu trúc:
với bảy logits bằng nhau, tổng xác suất entry là `6/7` dù bằng chứng TRADE và
HOLD thực tế ngang nhau.

Retention V2 thay thế các contract đó nhưng không đổi schema bridge 20 output,
không đổi topology checkpoint gốc và không thêm writer thứ hai.

## Data contract bất biến

Mỗi lifecycle hoặc entry-path đã settle được deep-copy thành một
`RetentionEpisode` gồm:

- `episode_id`: SHA-256 rút gọn từ trace, loại credit, scope và action;
- `bucket`: loại evidence, context, action và win/loss/flat hoặc hard-negative;
- `payload`: `StepTrace` nhân quả, outcome/advantage hoặc utility target, mask,
  nhãn auxiliary và phân phối hành vi tại thời điểm quyết định;
- `admitted_batch`, `replay_count`;
- cờ `anchor`, `behavior_anchor`, `preference_anchor`, `hard_negative` và
  `audit_only`.

Store không trả object đang lưu. Admission và sampling đều deep-copy, nên
gradient/replay consumer không sửa được bằng chứng gốc. Item chỉ đủ điều kiện
replay từ batch kế tiếp; chính item vừa làm đầy batch không thể bị nhân đôi
trong cùng một optimizer step.

Mỗi policy owner/context có `ExpertRetentionStore` riêng với ba index bounded:

- `recent`: bao phủ regime mới;
- `diverse`: round-robin theo bucket và loại bớt bucket đông nhất trước;
- `anchor`: lifecycle dương vượt `anchor_min_r`, entry hard-negative và HOLD
  mà nhãn hậu nghiệm vẫn cho `P(HOLD) >= trade_threshold`.

Anchor được cắt dung lượng theo bucket thay vì chỉ giữ N item mới nhất. Vì vậy
một dòng HOLD dày đặc không thể đẩy hết lifecycle thắng ra khỏi store, và dòng
lifecycle thắng cũng không thể xóa hết control HOLD.

Payload chỉ lưu một lần trong `episodes`; ba index dùng ID nên không nhân ba
dung lượng checkpoint.

## TRADE/HOLD phân cấp

Bảy output cũ vẫn là `[HOLD, MARKET_BUY, MARKET_SELL, BUY_LIMIT, SELL_LIMIT,
BUY_STOP, SELL_STOP]`. Chúng được diễn giải thành:

```text
trade_logit = logmeanexp(sáu entry logits)
P(TRADE)    = sigmoid(trade_logit - hold_logit)
P(action)   = P(TRADE) * softmax(entry logits)[action]
P(HOLD)     = 1 - P(TRADE)
```

`logmeanexp`, thay vì `logsumexp`, loại lợi thế giả do có sáu lớp entry. Sáu
entry logit bằng HOLD logit cho `P(TRADE)=0.5`. Runtime chỉ chọn entry khi
`P(TRADE) >= direct_rnn_rl_trade_threshold`; sau đó mới chọn entry conditional
tốt nhất.

Counterfactual target cũng tách làm hai phần:

- nếu policy đã chọn entry, TRADE/HOLD dùng utility của **đúng action đã
  chọn** so với HOLD cộng margin;
- nếu policy đã chọn HOLD, TRADE/HOLD dùng mean utility của các entry hợp lệ,
  không dùng action tốt nhất nhìn lại sau 48 nến;
- action-given-TRADE vẫn dùng softmax trên các entry hợp lệ, nhưng conditional
  CE được nhân theo target TRADE nên một hard-negative không thể vừa dạy HOLD
  vừa kéo mạnh sang một alternative oracle.

Coverage V187 còn theo dõi chính các quyết định HOLD. Resolver preference riêng
chấp nhận neural HOLD có `flat_idle_non_feedback`, nhưng vẫn loại transport
no-op và HOLD do safety/mask rewrite. Nếu chỉ thu path khi
policy chọn entry, việc học được HOLD sẽ làm vùng state đó biến mất khỏi tập
nhãn, tạo feedback `ít TRADE -> ít supervision -> drift -> TRADE tăng lại`.
HOLD path mới được deep-copy vào cùng recent/diverse retention với bucket
action `0`; store không đổi optimizer, nhãn hay inference.

Contract cũ `hierarchical_trade_hold_v1` dùng `max` của sáu alternative. Smoke
MT5 ngày 2025-01-01 cho thấy lifecycle advantage âm nhưng target TRADE vẫn
`0.71–0.98`; trên 68 path, mean utility sáu entry là `-0.063R` nhưng max
hindsight trung bình `+0.888R`, làm target TRADE trung bình `0.868`. Contract
`hierarchical_on_policy_trade_v2` loại thiên lệch oracle này. Khi migrate,
preference item V1 trong pending queue và retention store bị xóa; lifecycle và
positive behavior anchor vẫn được giữ nguyên.

Actor lifecycle dùng đạo hàm chính xác của policy phân cấp, không dùng gradient
softmax phẳng giả. Entropy, preference CE và behavior distillation đều có
đạo hàm trên cùng bề mặt này.

## Replay và bảo tồn

Mỗi batch live vẫn giữ quota tối thiểu 50% lifecycle MT5 canonical. Sau khi
chọn batch live, store thêm tối đa `replay_items_per_batch` item cũ, cân bằng
lifecycle/preference và bucket. `replay_weight` hạ ảnh hưởng của bằng chứng cũ.

Replay dài hạn tính lại action head hiện tại trên hidden-state nhân quả đã
cache. Nó không dùng decision-time logits cũ để tạo gradient lỗi thời. Hidden
và temporal trace gốc vẫn là representation causal dùng cho BPTT; không có
future path nào vào inference.

Lifecycle dương đủ mạnh có thêm distillation target từ phân phối hành vi đã
đóng băng. Hard-negative không distill hành vi xấu; nó được giữ bằng target
counterfactual HOLD/alternative tốt hơn. Strong HOLD preference cũng là
`preference_anchor`: audit dùng target hậu nghiệm, không distill phân phối HOLD
có thể đã sai tại decision-time.

Schema 5 chia anchor ổn định theo `context_frame_id`, loại credit, action và
context: 20% là `audit_only`. Item này có thể được học một lần khi còn là bằng
chứng live, nhưng từ các batch sau không bao giờ được đưa vào replay gradient.
Tối đa 16 audit-only anchor cân bằng bucket được dùng làm tập kiểm định độc lập
cho mỗi candidate optimizer step. Assignment không phụ thuộc RNG hoặc suffix
timestamp của trace, nên cùng MT5 frame có cùng split giữa các run.

## Audit và rollback nguyên tử

Trước AdamW step, candidate snapshot toàn bộ tensor Direct-RNN liên quan,
AdamW `m/v`, step counter và optimizer telemetry. Sau step, candidate rollback
toàn bộ snapshot khi gặp một trong các lỗi:

- tensor không hữu hạn;
- update norm vượt `direct_rnn_rl_retention_max_update_norm`;
- mean KL hoặc KL của bất kỳ replay-held-out anchor nào tăng quá tolerance.

V188 chứng minh audit V1 cũ chưa đủ: run 254 quyết định có 64 preference anchor
nhưng chỉ audit 24 lượt preference và các item được audit cũng nằm trong
gradient batch, nên KL sau update gần như luôn giảm. MT5 ledger của run đó vẫn
có 66 lifecycle, `-18.782R`, mean `-0.285R`, LCB95 `-0.521R`, PF `0.555` và
equity DD khoảng `4.60%`. Vì vậy self-audit trên training batch đã bị thay thế,
không được giữ như bằng chứng bảo tồn.

V189 kiểm chứng schema 5 trên MT5 thật cùng seed/window/model: 225 decision,
RANGING có 14 audit-only anchor; 17/21 optimizer step dùng nguồn
`replay_held_out_store_v1`, lặp 132 lượt audit và max per-anchor KL increase chỉ
`0.000221`, thấp hơn tolerance `0.02`, nên không rollback. Điều này chứng minh
candidate step không làm hỏng tập anchor đã giữ, nhưng không chứng minh edge:
ledger có 92 lifecycle, `-16.245R`, mean `-0.177R`, LCB95 `-0.406R`, PF
`0.647`, PnL `-458.07` và equity DD khoảng `7.34%`. Đoạn sequence 151–200 của
RANGING đạt `+108.08` và `+4.973R`; sau khi owner đổi sang `CTX_GIAM_YEU`, năm
lifecycle kế tiếp mất `-58.21`/`-2.078R`. Vì vậy dao động còn lại là edge
action/context cục bộ cộng cold expert khi đổi owner, không có bằng chứng
catastrophic forgetting trên anchor bất biến. Candidate vẫn không đủ gate OOS.

Rollback không chỉ trả weight; optimizer moment và step counter cũng quay lại,
nên batch bị từ chối không để lại momentum ẩn. Store không tự rollback/xóa
evidence vì evidence đã settle vẫn hợp lệ cho batch sau.

## Cấu hình candidate hiện tại

```yaml
learning:
  direct_rnn_rl_entry_preference_include_hold_snapshots: true
  direct_rnn_rl_trade_hold_separation_enabled: true
  direct_rnn_rl_trade_hold_margin_r: 0.10
  direct_rnn_rl_trade_hold_temperature: 0.25
  direct_rnn_rl_trade_threshold: 0.55
  direct_rnn_rl_retention_enabled: true
  direct_rnn_rl_retention_recent_capacity: 64
  direct_rnn_rl_retention_diverse_capacity: 128
  direct_rnn_rl_retention_anchor_capacity: 64
  direct_rnn_rl_retention_replay_items_per_batch: 8
  direct_rnn_rl_retention_replay_weight: 0.50
  direct_rnn_rl_retention_distillation_weight: 0.10
  direct_rnn_rl_retention_anchor_min_r: 0.25
  direct_rnn_rl_retention_audit_min_anchors: 2
  direct_rnn_rl_retention_max_anchor_kl_increase: 0.02
  direct_rnn_rl_retention_max_update_norm: 1.0
```

Mặc định trong `LearningConfig` và artifact migration vẫn là `False`, do đó
checkpoint/seed được bảo vệ không đổi hành vi nếu profile không opt-in.

## Telemetry và kiểm chứng

Event `direct_rnn_full_rl_batch` ghi số replay/anchor, số audit-only, nguồn
audit, distillation loss/gradient, mean KL trước/sau, max per-anchor KL
increase, update norm, lý do rollback, thống kê store và
TRADE/HOLD target/probability. `entry_preference_trade_utility_mean` và
`entry_preference_trade_target_bases` chứng minh group target đến từ action đã
chọn hay mean legal entries. Queue receipt cũng ghi basis/utility và trạng thái
store.

Unit/regression bắt buộc:

- finite-difference chứng minh ascent là âm gradient của loss phân cấp;
- hard-negative dùng utility action đã chọn, không dùng max hindsight;
- HOLD counterfactual dùng mean legal-entry utility;
- HOLD được snapshot khi opt-in, đi vào bucket preference action `0`, còn
  management action `7..11` không được nhận nhãn entry;
- migration loại target contract cũ nhưng không xóa lifecycle anchor;
- logits bằng nhau cho `P(TRADE)=0.5` và threshold 0.55 chọn HOLD;
- immutable store, bounded indexes và pickle round-trip;
- batch sau thực sự replay anchor của batch trước;
- audit-only split tách rời replay gradient và ổn định theo context frame;
- gradient TRADE cố ý làm xấu strong-HOLD audit anchor bị rollback nguyên tử;
- A→B audit hoặc không làm KL A xấu đi, hoặc rollback nguyên tử;
- update-norm guard trả cả weight và optimizer step;
- toàn bộ test Full-RL/entry-preference/sole-writer cũ vẫn qua.

Đây là bằng chứng implementation, không phải bằng chứng edge. Promotion vẫn
cần paired frozen baseline trên cùng cửa sổ, nhiều frozen OOS window, policy-
origin expectancy LCB, drawdown, action-scope failures và real-tick parity.
