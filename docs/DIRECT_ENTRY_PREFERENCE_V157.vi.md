# V157/V175/V187 — Học trực tiếp HOLD hay entry

## Mục tiêu

Lifecycle hiện có chỉ cho policy biết kết quả của lệnh đã đặt. Nó không trả
lời được câu hỏi phản thực tế: ở cùng trạng thái đó, `HOLD` hoặc một kiểu
entry khác có tốt hơn không. V157 thêm nhãn trễ cho đúng câu hỏi này mà không
thêm gate, mask, selector hay writer mới.

## Hợp đồng EA → Python

Khi profile V187 bật `direct_rnn_rl_entry_preference_include_hold_snapshots`, EA
snapshot bất biến tại mọi quyết định thuộc surface entry: `HOLD`, `MARKET`,
`LIMIT` hoặc `STOP`:

- `trace_id`, sequence, action được phát;
- bid/ask, ATR quyết định, SL ATR, RR và sáu giá vào lệnh sau broker clamp;
- 48 nến M1 đã đóng sau quyết định: high, low, close, spread.

EA chỉ ghi `self_growing_entry_preference.csv` sau khi đủ horizon. Snapshot
không dùng trailing, policy close hay SL/TP bị sửa sau entry. Vì vậy nhãn đo
chất lượng entry ban đầu, không lẫn credit của management. Lệnh pending không
fill nhận `0R`, giống `HOLD`. Khi SL và TP cùng bị chạm trong một M1 bar, replay
chọn SL để tránh ưu ái candidate không quan sát được intrabar path.

Schema `v2` ghi thêm `sample_scope`, `positions_before`, `orders_before`:

- `flat_v1`: không position và không pending; giữ ngữ nghĩa V157 ban đầu.
- `entry_marginal_v2`: policy đã đề xuất entry khi portfolio còn exposure.
  Replay vẫn chỉ đo P/L thị trường của **proposal mới** trên path bất biến;
  portfolio có sẵn được giữ ngoài nhãn. Nó không phải account-PnL, không sửa
  risk sizing, không tham gia execution và không có quyền gate/mask inference.

V175 từng chỉ cache action entry thật. Telemetry V2 cho thấy đây là một vòng
phản hồi tự chọn mẫu: khi policy học giảm TRADE, các state vừa chọn HOLD không
còn nhãn path; gradient từ entry cũ/shared representation sau đó có thể kéo
`P(TRADE)` tăng lại. V187 cache cả HOLD nên vùng "nên đứng ngoài" vẫn nhận nhãn
trễ và đi vào retention. Action management `7..11` vẫn bị loại để không trộn
objective entry với exit. Không có future path nào được đọc tại inference.

Python dùng resolver riêng cho surface này: neural HOLD có
`flat_idle_non_feedback/action_attribution_source=hold_no_execution` vẫn hợp lệ,
nhưng transport-only HOLD (`wire_noop/no_neural_command`) và HOLD do safety hoặc
mask rewrite bị loại. Nhờ vậy firebreak lifecycle không còn vô tình cắt mất
nhãn HOLD, trong khi không tạo credit cho action không phải neural decision.

## Replay và loss

Python replay bảy action trên cùng path:

`[HOLD, MARKET_BUY, MARKET_SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP]`.

Với utility R đã replay và tập action hợp lệ của trace, contract candidate
hiện tách target thành `P(TRADE)` theo utility của **action policy đã chọn** so
với HOLD cộng margin. Với sample HOLD chưa có entry thực thi, target dùng mean
utility của các entry hợp lệ. Nó không dùng max hindsight của sáu alternative,
vì cách cũ gần như luôn tìm được một hướng/kiểu lệnh thắng sau khi đã thấy 48
nến và kéo TRADE ngược với lifecycle âm. `P(action | TRADE)` vẫn dùng softmax
trên sáu entry nhưng sức conditional CE đi theo target TRADE. `HOLD` luôn hợp
lệ. Target hiện hành mang mode `hierarchical_on_policy_trade_v2`; replay mode
V1 bị loại khi migrate.
Preference loss chỉ đọc logits `0..6`, không so với management logits `7..11`.
Profile legacy không opt-in vẫn dùng softmax phẳng để giữ checkpoint behavior.

Retention schema 5 biến strong HOLD thành preference anchor khi target hậu
nghiệm cho `P(HOLD) >= direct_rnn_rl_trade_threshold`. Một phần cố định 20% của
anchor là replay-held-out audit-only; chúng kiểm tra candidate step nhưng không
tham gia replay gradient ở các batch sau.

Trong `_apply_direct_rnn_full_rl_batch`:

`L = L_actor + L_critic + L_representation + L_counterfactual_representation
     + λ_pref L_preference + L_auxiliary + L_reconstruction + L_retention`.

Mục preference không có actor advantage, value loss, entropy hay geometry
credit riêng. Gradient là đạo hàm chính xác của BCE TRADE/HOLD cộng conditional
entry CE, đi qua exact output-normalization VJP, rồi backprop qua toàn bộ hidden
stack và gate. Kết quả vẫn gọi duy nhất
`ModuleState.apply_rnn_adamw_cosine(source="full_lifecycle_rl")`.

## Cấu hình

Trong `learning`:

- `direct_rnn_rl_entry_preference_enabled: true`
- `direct_rnn_rl_entry_preference_loss_weight: 0.25`
- `direct_rnn_rl_entry_preference_temperature: 0.50`
- `direct_rnn_rl_entry_preference_horizon_bars: 48`
- `direct_rnn_rl_entry_preference_include_hold_snapshots: true`
- `direct_rnn_rl_trade_hold_separation_enabled: true`
- `direct_rnn_rl_trade_hold_margin_r: 0.10`
- `direct_rnn_rl_trade_hold_temperature: 0.25`
- `direct_rnn_rl_trade_threshold: 0.55`

Horizon và coverage được ghi vào `.set` MT5 qua
`EntryPreferenceHorizonBars` và `EntryPreferenceIncludeHoldSnapshots`, nên EA
và Python không thể âm thầm theo dõi hai tập quyết định khác nhau. Default code
vẫn `false` để checkpoint/profile cũ không đổi; candidate active opt-in `true`.

## Telemetry phải kiểm tra sau run mới

Trong growth event `direct_rnn_full_rl_batch` cần thấy:

- `entry_preference_enabled=true`;
- `entry_preference_eligible_count > 0`;
- `entry_preference_sample_scopes.entry_marginal_v2 > 0` khi run có entry
  chồng exposure;
- receipt/retention phải có bucket action `0` sau khi các HOLD path đủ horizon;
- mỗi receipt queue có `batch_triggered`; item thứ 8 không được biến mất chỉ
  vì nó lập tức mở batch;
- `entry_preference_loss > 0`, `entry_preference_gradient_l1 > 0`;
- `entry_preference_hold_target_mean`, target mean từng action và utility mean;
- `entry_preference_trade_target_mean`,
  `entry_preference_trade_probability_mean`,
  `entry_preference_trade_utility_mean`, `entry_preference_trade_target_bases`
  và `trade_hold_threshold`;
- optimizer vẫn là `full_lifecycle_rl`, với `updated_modules > 0`.

Trong `self_growing_entry_preference.csv`, mỗi dòng phải có path đủ 48 bar,
sáu entry price dương, scope hợp lệ và `trace_id` đã được cache đúng owner. Nếu có CSV row
nhưng eligible count bằng 0, kiểm tra `trace_id` owner/cache trước; không bật
gate để che lỗi routing.

Khi probe bị dừng bằng `max_wall_sec`, runner phải flush toàn bộ row đã đủ
horizon trước khi trả timeout. Row chưa đủ horizon vẫn không có nhãn; điều này
ngăn mất credit chỉ do callback cuối không kịp chạy.

## Giới hạn bằng chứng

V157 chứng minh model nhận được nhãn phản thực tế và thay đổi logits. Nó chưa
chứng minh edge. Cần frozen replay checkpoint đầu/cuối trên cùng dữ liệu và
OOS MT5: đo decision divergence, HOLD rate, action mix, PF, expectancy, LCB,
DD và equity. Entry preference chỉ được giữ nếu OOS tốt hơn baseline.
