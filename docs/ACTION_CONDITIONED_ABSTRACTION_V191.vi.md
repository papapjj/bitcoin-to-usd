# V191 — Action-conditioned abstraction

> Tài liệu nền lịch sử. Candidate hiện hành là V193 tìm edge và giữ
> TensorRule ở shadow; xem `docs/EDGE_DISCOVERY_V193.vi.md`.

## Mục tiêu

V191 xử lý state aliasing trong latent: hai input có vector tổng hợp gần nhau
không còn mặc nhiên nhận cùng action từng thắng. Mỗi context expert có Transformer,
prototype, latent factor, action adapter, statistics và Adam moments riêng.

## Luồng dữ liệu

```text
32 x 78 M15 đã đóng
  -> 2 causal Transformer block, hidden 64
  -> structure/timing/regime/risk, mỗi nhánh 16
  -> ghép action embedding riêng
  -> 6 adapter 80 -> 32
  -> expected-R + uncertainty theo action
  -> soft TensorRule membership hoặc UNKNOWN
```

Rule vẫn là prototype tensor mềm. Tên các latent factor chỉ xác định phạm vi
gradient; không có điều kiện `if/else` dựa trên thị trường để chọn action.

## Nhãn và thời điểm mature

- return direction, Q10/Q50/Q90 và volatility: H1/3, H1, H4, H6, H10;
- regime transition: H2, H4, H6, H8;
- sáu entry action: fill, time-to-positive, expected R, SL/TP, timing, MFE/MAE;
- listwise ranking HOLD + 6 entry, reconstruction, supervised contrastive;
- anti-collapse theo variance/covariance của batch latent.

EA chỉ giữ path counterfactual chi tiết 48 nến M1, nhưng giữ thêm close và
volatility thống kê thưa đến phút 600. Horizon chưa đủ được mask; tuyệt đối
không dùng close cuối path 48 thay cho H1/H4/H6/H10. Regime tương lai chỉ là
delayed target.

## Phạm vi gradient

| Nguồn tín hiệu | Action branch | Latent factor | Transformer |
|---|---|---|---|
| Lifecycle settled | action đã chạy | không | không |
| Counterfactual 48-M1 | cả 6 action | có, scale 0,25 | có, scale 0,25 |
| Fill/time/timing | head phụ | timing | có |
| SL/MAE | head phụ | regime + risk | có |
| TP/MFE | head phụ | structure + regime | có |
| Expected R | head phụ | cả bốn | có |
| Regime transition | head phụ | regime | có |

Reconstruction không chạy trên sample chỉ có lifecycle PnL. Vì thế một trade
thua không thể vô tình kéo Q/K/V và làm đổi mọi state/action. TensorRule không
có optimizer riêng; mọi gradient vẫn chờ effective batch 64 và đi qua sole
writer Direct-RNN AdamW.

## Hòa giải nhãn đến trễ và policy cũ

Profile V191 không còn backprop trực tiếp qua activation được cache bởi policy
tại thời điểm đặt lệnh. Khi lifecycle hoặc path counterfactual mature:

1. forward lại đúng observation, recurrent boundary, dropout mask và
   State-Aliasing Memory token bất biến bằng **trọng số hiện tại**;
2. không đọc runtime memory hiện tại, nến mới hoặc future label trong forward;
3. với action đã thực thi, tính `pi_current(a|s) / mu_behavior(a|s)`, clip tại
   1,0, giảm thêm theo Jensen-Shannon drift và bỏ mẫu nếu vượt gate;
4. lifecycle chỉ cập nhật action/value head; encoder, hidden, attention, gate
   và memory Q/K/V có credit scale 0;
5. counterfactual xếp hạng toàn bộ HOLD + 6 entry bằng target
   `counterfactual_consensus_v3`, không lấy action cũ làm nhãn. TRADE cần đồng
   thuận utility của hai entry tốt nhất; encoder credit bị chặn ở 0,25.

Executed lifecycle vẫn cung cấp bằng chứng factual cho action đã chạy, nhưng
không thể dùng quyết định của policy cũ để viết lại vùng latent rộng.
Counterfactual full-information mới là đường được phép học quy luật chung.
Telemetry `direct_rnn_full_rl_batch.delayed_reconciliation` ghi số re-forward,
số lifecycle bị loại, mean effective weight, mean policy JS, action disagreement
và hai encoder-credit scale. AdamW vẫn là writer duy nhất; bridge 137x20 và
nhánh management không đổi.

## Cấu hình và vận hành

Profile: `configs/experiments/action_conditioned_abstraction_v191_candidate.yaml`.
Artifact version 14, TensorRule schema 2, 133.633 tham số TensorRule mỗi expert.
Profile V191 hiện bật `authority=residual` và behavior exploration giới hạn
`epsilon=0,10`. Gate `tensor_rule_entry_lcb_gate_enabled=true` thu hồi quyền
entry greedy khi bank còn `UNKNOWN`; khi đó chỉ action có provenance exploration
mới được phép trade, còn actor greedy trả HOLD. Khi bank đã biết, entry chỉ hợp
lệ nếu LCB của action vượt HOLD thêm `0,10R`; residual vẫn bị clip ±0,25.
HOLD và nhánh management/exit không nhận residual. Code mặc định của checkpoint
cũ vẫn tắt gate và exploration, nên migration không đổi hành vi legacy. Đây là
kích hoạt candidate có giới hạn, chưa phải promotion `primary` hay bằng chứng edge.

```powershell
python scripts/run_tensor_rule_v186_candidate.py --mode python-debug `
  --profile configs/experiments/action_conditioned_abstraction_v191_candidate.yaml `
  --output-dir outputs/he_chuyen_gia/debug/action_conditioned_v191
```

Chỉ promote residual/primary sau paired real-tick OOS theo gate trong tài liệu
V186. Python debug, unit test hoặc compile 0 lỗi không phải bằng chứng có edge.

## Bằng chứng implementation 2026-08-04

- `python -m pytest -q`: 886 passed;
- MetaEditor64: 0 error, 0 warning;
- Java UI latest-training contract: `V191_UI_LATEST_TRAINING_CONTRACT_OK`;
- candidate debug: 9/9 bank độc lập, 133.633 tham số/bank, artifact 14,
  exploration rate 0, bridge 137x20, source config/seed hash không đổi;
- manifest: `outputs/he_chuyen_gia/debug/action_conditioned_v191_final_20260804/manifest.json`.
- residual activation debug: 9/9 bank có `authority=residual`, cold/UNKNOWN giữ
  nguyên action actor; manifest:
  `outputs/he_chuyen_gia/debug/tensor_rule_residual_v191_20260804/manifest.json`.

Các kiểm tra này chỉ chứng minh wiring, migration, isolation và gradient contract.
Chưa có một MT5 train/OOS mới để kết luận edge tốt hơn hoặc ổn định hơn.

## Bằng chứng E2E bổ sung 2026-08-05

- causal current-weight re-forward không đổi runtime recurrent/dropout/memory;
- frozen memory token được đọc bằng Q/K/V hiện tại mà không tăng counter;
- policy drift lớn bị reject và có effective weight bằng 0;
- lifecycle batch vẫn đổi action head nhưng encoder/deep recurrence giữ byte ổn định;
- target counterfactual-consensus không chứa behavior action;
- targeted regression sau sửa: 75 test pass; full suite: 897 test pass;
- MT5 model 1, seed 0, 2025.01.01–2025.01.02: 738 action, 129 snapshot H10
  mature, 129 ACK, 0 reject, 0 pending cuối run;
- queue preference tăng 1→2→3→4 và tạo một batch thật gồm 12 lifecycle +
  4 preference; `entry_preference_eligible_count=4`, loss `0,11716`, gradient
  L1 `0,08262`, `updated_modules=1`;
- TensorRule batch dùng cùng sole writer `full_lifecycle_rl_adamw`; retention
  không rollback, KL increase khoảng `2,32e-7`;
- source config và seed hash không đổi; feature parity 256/256, reconciliation
  trade/PnL đạt.

E2E này xác nhận đường học, không xác nhận edge. Kết quả run vẫn âm: net
`−332,10`, PF `0,49`, WR `40%`, equity DD `4,93%`, policy expectancy LCB95
`−0,4415R`; candidate không đủ điều kiện promotion. Manifest:
`outputs/he_chuyen_gia/v193_e2e/candidate_queue_fix_20260805/manifest.json`.
