# V186/V191 — Tensor Rule Abstraction riêng cho từng chuyên gia

> Tài liệu nền. Profile hiện hành V193 dùng schema 6/artifact 18 và authority
> shadow trong chương trình tìm edge; xem `docs/EDGE_DISCOVERY_V193.vi.md`.

## Trạng thái

V191 thay thế latent dùng chung của V186 bằng latent có cấu trúc và adapter riêng
cho từng action. Cấu hình mặc định trong code vẫn là `tensor_rule_enabled=false`;
profile V191 thử nghiệm riêng bật `authority=residual`. Residual chỉ tác động sáu
entry logits khi bank không `UNKNOWN`, bị chặn cứng ở ±0,25; cold bank vẫn giữ
nguyên hành động actor. Bridge MT5 giữ nguyên 137 input và 20 output.

TensorRule schema hiện là 2 và artifact kết hợp dùng version 14. Checkpoint cũ
vẫn load với rule bank tắt. Khi migrate schema 1, posterior action/rule được giữ,
nhưng auxiliary H3/H12/H48 bị reset vì không cùng nghĩa với horizon mới; adapter
action mới khởi tạo với value row bằng 0 nên không đổi output cũ.

## Kiến trúc mỗi expert

Mỗi context owner có đúng một `TensorRuleBankState` gắn vào module `m1_thinking_core` của chính expert đó. Không có encoder, prototype, statistics, gradient accumulator hay Adam moments dùng chung giữa expert.

```text
chuỗi M15 đã đóng [32, 78]
        │
        ▼
linear input + positional tensor
        │
        ▼
2 × pre-norm Transformer block
  causal multi-head Q/K/V attention, 4 head
  FFN 64 → 128 → 64
        │
        ▼
latent sequence [32, 64]
        │
        ├── structure [16]
        ├── timing [16]
        ├── regime [16]
        └── risk [16]
                 │ + action embedding [16]
                 ▼
        6 action adapters [80 → 32]
                 │
                 ├── action-conditioned expected R
                 └── utility contrastive + anti-collapse
        │
        ▼
soft distance tới P[16, 32, 64]
        │
        ▼
membership nhiều rule hoặc UNKNOWN
        │
        ▼
posterior mean/std/LCB95 cho 6 entry action
```

Cấu hình chuẩn: hidden 64, 2 block, 4 head, FFN 128, `Kmax=16`, hai slot active ban đầu. Mỗi rule có prototype tensor, time importance, feature scale, posterior mean-R/log-variance cho sáu entry. `HOLD` luôn là baseline 0R.

Decoder reconstruction dùng chung transpose của input projection, không có
decoder matrix thứ hai. Cấu hình chuẩn có 133.633 tham số TensorRule mỗi expert.
Đây là true Transformer attention với Q/K/V và causal `QKᵀ`, không phải channel
gate attention của Direct-RNN cũ.

## Contract input và UNKNOWN

Inference chỉ đọc `sequence_tensor[32,78]` từ TimingNet trace. Bank không đọc hoặc chia sẻ bất kỳ trọng số/embedding nào của TimingNet. Tensor chỉ chứa M15 đã đóng tại thời điểm quyết định.

Bank trả `UNKNOWN` khi:

- shape hoặc schema fingerprint sai;
- valid length nhỏ hơn 16;
- không có active rule;
- max membership nhỏ hơn 0,55;
- rule chưa đủ support.

Hash decision chỉ tính trên phần chuỗi hợp lệ, vì vậy thay đổi padding không đổi fingerprint. Causal mask bảo đảm thay đổi các row tương lai không thể đổi latent prefix quá khứ.

## Mục tiêu học và nguồn nhãn

V186 tái sử dụng nguồn label đã được kiểm chứng của immutable 48-M1 entry path:

- hướng return H1/3, H1, H4, H6 và H10 từ nhãn thưa M1 đã xác nhận;
- quantile Q10/Q50/Q90 và realized volatility;
- regime transition H2/H4/H6/H8 theo decision M15, chỉ mature sau horizon;
- từng entry action: fill, time-to-positive, expected R, MFE, MAE, xác suất SL/TP, timing quality;
- listwise ranking giữa HOLD và sáu entry;
- masked reconstruction;
- supervised contrastive và prototype-centre loss.

Positive contrastive phải cùng expert, utility signature tương tự và khác temporal fold. Hard negative có latent gần nhưng action ranking khác. Utility signature chỉ tạo delayed training target cho prototype; inference không chứa `if/else` hoặc lookup rule cứng.

`realized_r` từ lifecycle MT5 chỉ sửa posterior và adapter của action thực sự đã
fill/settle; mặc định encoder credit bằng 0. Counterfactual sáu action được phép
cập nhật encoder với scale 0,25. Auxiliary được route theo họ nghĩa: timing label
chỉ vào timing latent, downside vào regime+risk, opportunity vào
structure+regime, transition vào regime. Future confirmed regime và immutable
future path không đi vào inference. Mỗi delayed sample được forward lại từ
tensor quyết định bất biến bằng trọng số hiện tại.

Khi V186 bật, ba representation objective legacy (`auxiliary`, `conditioned`, `counterfactual representation`) bị tắt cho candidate để không huấn luyện song song hai encoder trên cùng label. Đường legacy vẫn tồn tại để load và chạy baseline/checkpoint cũ.

## Sole-writer và bảo tồn

TensorRuleBank không có optimizer. Gradient được tích lũy riêng theo expert đến effective batch 64, rồi gắn vào đúng lần `full_lifecycle_rl` AdamW kế tiếp. `ModuleState.apply_rnn_adamw_cosine` vẫn là writer duy nhất; rule tensors dùng chung step, clipping, cosine LR, weight decay, moments và rollback với Direct-RNN.

Retention schema 6 phân tầng theo expert-store × rule × action × temporal fold × outcome. Champion/shadow copy là deep copy, nên transformer, prototypes, active mask, statistics, gradient accumulator và Adam moments đều nằm trong snapshot/rollback.

Telemetry mỗi decision có schema/bank version, input và seed fingerprint, membership, entropy, rule id, UNKNOWN reason, action mean/std/LCB. Batch telemetry có loss, contrastive pairs, effective-batch state, update count và sole-writer provenance. Optimizer block telemetry ghi parameter fingerprint/delta. Bank summary có rule support, posterior từng rule, LCB95 theo temporal fold/action và topology proposal cuối.

## Authority

- `telemetry`: tính và ghi trace; actor quyết định nguyên trạng.
- `shadow`: so sánh nhưng không tác động actor.
- `residual`: chỉ khi đủ support; rule cộng residual entry tối đa ±0,25 logit. UNKNOWN giữ actor.
- `primary`: rule dùng LCB95; chỉ entry khi tốt hơn HOLD ít nhất 0,10R. UNKNOWN trả HOLD. Actor chỉ là residual ±0,25. Management/exit 7..11 không đổi.

Promotion `residual` fail-closed nếu chưa có tối thiểu hai rule đủ support, tổng support 90, mỗi ba temporal fold đủ 20 và action scope được boost đủ 30. Giữa round, calibration lấy weighted P95 absolute error từ các cell temporal held-out đủ support để cập nhật uncertainty floor. `primary` còn bắt buộc calibration đã hoàn tất, paired real-tick OOS 3 seed × 3 window, source hash/parity/reconciliation, expectancy và delta LCB, half-window stability, DD ≤10%, zero critical negative scope và ranking không đảo dấu.

## Tiến hóa topology

Topology đóng băng trong MT5 window. `evolve_tensor_rules(..., round_boundary=True)` chỉ thực hiện tối đa một operation giữa round:

- split: parent support ≥128, child dự kiến ≥32, held-out loss tốt hơn ≥5% và LCB không xấu;
- merge: prototype cosine ≥0,95, action posterior JS ≤0,05, held-out loss không xấu quá 1%;
- retire: support <16 hoặc không có validation contribution trong ba round.

Capacity luôn cố định 16 slot. Reset/split chỉ xóa Adam moments của slot liên quan; shape artifact không đổi.

## Cấu hình và chạy candidate

Profile V191 lịch sử: `configs/experiments/action_conditioned_abstraction_v191_candidate.yaml`.
Profile hiện hành: `configs/experiments/edge_discovery_v193_candidate.yaml`.

V191 bật behavior exploration entry 10% và entropy 0,002, đồng thời dùng
cold-start LCB gate. Trước khi value surface có LCB, greedy actor phải HOLD;
chỉ sampled action có provenance được trade để tạo nhãn. Profile kích hoạt
TensorRule ở residual có giới hạn, còn `primary` vẫn bị giữ sau promotion gate
paired OOS. Xem
`docs/DIRECT_RNN_ENTRY_EXPLORATION_V188.vi.md`.

Từ V190, cùng profile bật thêm causal State-Aliasing Memory trước base hidden để
phân biệt current snapshot giống nhau nhưng lịch sử có thứ tự khác nhau. Memory
độc lập với TensorRuleBank nhưng dùng chung sole-writer Full-RL AdamW. Xem
`docs/STATE_ALIAS_MEMORY_V190.vi.md`.

Python debug, không cần MT5:

```powershell
python scripts/run_tensor_rule_v186_candidate.py --mode python-debug --output-dir outputs/he_chuyen_gia/debug/tensor_rule_v186_python_debug
```

Debug từ một seed artifact thật nhưng chỉ qua bản clone:

```powershell
python scripts/run_tensor_rule_v186_candidate.py --mode python-debug `
  --seed-artifact <duong-dan-seed.pkl> `
  --output-dir outputs/he_chuyen_gia/debug/tensor_rule_v186_seed_clone
```

MT5 telemetry smoke tuần tự:

```powershell
python scripts/run_tensor_rule_v186_candidate.py --mode mt5-smoke `
  --seed-artifact <duong-dan-seed.pkl> `
  --from-date 2025.01.01 --to-date 2025.01.02 --model 1 `
  --output-dir outputs/he_chuyen_gia/experiments/tensor_rule_v186_smoke
```

Runner clone giám sát quota payload theo profile và reserve floor 2,5 GiB.
`source_seed_clone.pkl` là đầu vào bất biến được bảo vệ nên không bị cộng nhầm
vào quota payload mới sinh. Trên cùng volume, runner tạo hard-link chỉ đọc theo
workflow tới seed nguồn; hai tên file tồn tại độc lập và xóa một tên không xóa
tên còn lại. Nếu filesystem không hỗ trợ, runner mới fallback sang `copy2`.
Preflight dành reserve cộng payload dự kiến có 25% headroom và ngân sách
telemetry; sau khi clone nó kiểm tra lại dung lượng thực, nhờ đó không để lại
seed dở dang khi ổ đĩa thiếu chỗ. Với `--metrics-only`, pre-clone gate dùng
`effective_storage_reserve_floor()` để trừ footprint runner tạm (vẫn giữ sàn
512 MiB), còn run persistent giữ nguyên reserve cấu hình. Event JSONL giới hạn
32 MiB, execution trace 64 MiB. Nếu vi phạm storage gate, chỉ process có
executable path nằm trong runner clone V186 bị dừng và
`storage_gate_failure.json` ghi riêng tổng bytes, payload bytes, phần seed được
loại trừ và dung lượng trống. Nếu MT5 metrics-only đã ghi report, ledger đã
reconcile, Codex không còn packet pending và payload vẫn dưới quota, áp lực
storage ở bước dọn runner được ghi thành warning trong manifest
(`storage_gate_completion_policy=completed_metrics_only_warning`) thay vì biến
run nghiên cứu hoàn chỉnh thành lỗi; run bị interrupt, thiếu reconcile hoặc
persistent checkpoint vẫn fail-closed. Paired seed validation chỉ được chạy khi
`persist_checkpoint=true`; metrics-only không có candidate để so sánh nên ghi
`paired_seed_validation.reason=metrics_only_no_checkpoint` và không clone thêm
tick data sau khi main MT5 evidence đã hoàn tất. Candidate manifest và
storage-gate contract dùng schema 2; log/model cũ không bị xóa.

## Test

```powershell
python -m pytest tests/test_tensor_rule_bank_v186.py -q
python -m pytest tests/test_full_rl_exclusive_writer.py tests/test_direct_rnn_retention.py tests/test_timing_network.py -q
```

Bộ test bao phủ shape/QKV/causal mask/padding/numerical stability, finite-difference Q gradient, schema fingerprint, effective batch, expert isolation, old-checkpoint safety, deterministic save/load, sole-writer, rollback state, topology round boundary, authority gate, delayed regime target và synthetic abstraction trên temporal holdout.

## Giới hạn bằng chứng

Python debug chứng minh implementation, authority wiring và cold/UNKNOWN không
đổi actor; nó không chứng minh edge. Candidate chỉ được promote `primary` bằng
MT5 execution ledger, policy-origin trade, real-tick paired OOS và toàn bộ hard
gate ở trên. PnL/PF aggregate một window không đủ điều kiện.

Báo cáo test hiện tại: `docs/TENSOR_RULE_ABSTRACTION_V186_TEST_REPORT.vi.md`.

## Sửa latent/rule collapse V192 (artifact 17)

Artifact 17 dùng TensorRule schema 5. Delayed preference không còn phải chờ
batch actor có tối thiểu bốn lifecycle mới được TensorRule nhìn thấy. Mỗi expert
gom preference đã ACK thành microbatch 32; đủ effective batch 64 thì bank tạo
direction và clip toàn bộ direction về norm tối đa 0,50. Direction chỉ được đưa
vào `_tensor_rule_pending_directions`; bước AdamW Full-RL do lifecycle sở hữu
vẫn là writer duy nhất. Item đã pre-accumulate được đánh dấu để không học hai
lần khi actor queue về sau đủ quota.

E2E clone ngày 2026-08-05 phục vụ 771 decision, 33 policy-origin trade và nhận
162/162 preference. CTX_RANGING thực hiện bốn bank update, support hai rule đạt
137,19 và 126,81. Ba microbatch preference 32 mẫu được quan sát; hai direction
ready có raw norm 18,51 và 27,17, được clip về budget 0,50 và giữ bền trong
candidate cho lifecycle kế tiếp. Toàn bộ 902 test pass. Đây là xác nhận pipeline
học, không phải xác nhận edge: net -303,10, PF 0,35, WR 24,2%, DD 4,15%, policy
LCB95 -0,5099R; model 1 không có real-tick proof và storage gate cuối run bị
chạm khi lưu candidate, nên tuyệt đối không promote.
