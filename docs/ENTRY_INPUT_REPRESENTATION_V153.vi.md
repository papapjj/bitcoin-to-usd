# V153 — sửa biểu diễn input entry theo timeframe và điều kiện

## Mục tiêu và phạm vi

Thay đổi này chỉ áp dụng cho candidate trong
`configs/he_chuyen_gia_mot_context_mot_policy.yaml`. Artifact seed v13 và cấu
hình riêng của seed giữ nguyên. Không có action mask, score cộng tay, prior
hướng hoặc optimizer/writer mới. Full-RL AdamW/cosine vẫn là writer duy nhất
của Direct-RNN.

## H1/M15 không còn đo bằng ATR M1

`features.mtf_scaling_mode: native_tf_atr_v2` thay định nghĩa candidate của
bốn trường M5/M15/H1:

- `ema_diff_atr`, `ema_slope_atr`, `gap_atr` chia cho ATR của đúng timeframe;
- `volatility_ratio = (ATR_tf - ATR_M1) / (ATR_tf + ATR_M1)` và bị chặn trong
  `[-1, 1]`;
- ATR thiếu hoặc bằng 0 có fallback hữu hạn giống nhau ở Python và EA.

Vì vậy EMA H1 không còn bị phóng đại bởi ATR M1 rồi soft-clip thành gần hằng
số `-1.393`; volatility H1 cũng không còn là ratio lớn dương bị ghim ở đỉnh.
Đây là chuẩn hóa theo volatility riêng timeframe, không dùng thống kê future.
`legacy_m1_atr_v1` vẫn tồn tại chỉ để đọc/chạy artifact được bảo vệ.

Candidate dùng thêm `learning.input_normalization_mode: groupwise_mtf_v3`.
Nó giữ groupwise v2 cho mọi nhóm khác, nhưng tại indices M5/M15/H1 `80..91`
dùng `1.2*tanh(x)` sau native-ATR projection. Đây là soft map không có plateau
generic ở `±1.393`; Python và EA áp dụng cùng hàm. Kết quả tái chiếu trên
1.572 input MT5 cũ: H1 volatility saturation giảm `99,94% -> 0%`; H1 EMA
native còn `55,73%` ở generic soft-clip nên v3 được thêm để loại phần còn lại.
Đây là bằng chứng transform, không phải kết quả OOS của candidate mới.

## Biểu diễn entry cục bộ

Sau wire MT5 137 input và hai context token nhân quả `t-1/t-2`, candidate
thêm tám input cục bộ chỉ cho entry:

1. `long_trend_location`, `short_trend_location`;
2. `long_mtf_wyckoff_alignment`, `short_mtf_wyckoff_alignment`;
3. `long_location_volatility_warning`, `short_location_volatility_warning`;
4. `upper_wick_directional_buy`, `upper_wick_directional_sell`.

Upper wick được mã hóa `-wick` cho BUY và `+wick` cho SELL. Volatility chỉ
xuất hiện trong warning có điều kiện location; không có biến volatility nào
tự cộng điểm BUY/SELL. Tất cả tám giá trị chỉ là input bounded vào encoder,
nên trọng số và dấu cuối cùng do backprop Full-RL học từ lifecycle thực.
Management route luôn zero toàn bộ interaction market này. Hai context token
vẫn được management đọc, và code xử lý đúng cả policy chỉ có interaction mà
không có context-history.

## Quarantine tạm thời của raw feature

Trong route entry của candidate, không phải cả bridge, các raw input sau được
nhân `0.20`: volatility 15/60, candle range, RSI5, hour sin, M5 volatility,
M15 EMA diff/slope/volatility và trading-range width. Block H1 raw 88..91
tạm nhân `0.35`, trong khi distribution mới được đo lại. Interaction đã tạo
không bị giảm. Nhóm upper wick, EMA hiện tại, Wyckoff và distance swing không
bị quarantine.

Telemetry `action_details` có `entry_feature_quarantine_scales`, semantic
route audit và interaction values để phân tách BUY/SELL. Điều này không phải
là kết luận causal từ 253 lệnh: nó là vùng an toàn cho candidate cho tới khi
OOS xác nhận.

## Expectancy có điều kiện

`ConditionedEntryExpectancyLedger` ghi audit-only theo:

`(context, action, trend, location, volatility)`.

Nó dùng prior strength và minimum sample trước khi report; không đổi logit,
risk, selector hay learning rate. Mục đích là tránh kết luận sai kiểu
"volatility cao luôn xấu": MARKET_BUY đuổi giá và BUY_STOP breakout có bucket
khác nhau. Ledger này không được dùng làm mask cho tới khi có paired OOS.

## Ablation bắt buộc trước promotion

Chạy cùng seed clone, cùng cửa sổ MT5, cùng model tester và output artifact
khác nhau. Không overwrite v13.

| Arm | `entry_feature_enrichment_mode` | raw quarantine / H1 | Mục đích |
|---|---|---|---|
| A | `legacy` | 1.0 / 1.0 | Baseline sau khi chỉ sửa unit MTF |
| B | `quality_interactions_v1` | 0.20 / 0.35 | Candidate hiện tại |
| C | `quality_interactions_v1` | 0.0 / 0.0 | Nhóm bị nghi ngờ chỉ còn đường interaction có nghĩa |
| D | `quality_interactions_no_quarantined_v1` | 0.0 / 0.0 | Bỏ raw và cả interaction phụ thuộc nhóm quarantine; vẫn giữ interaction chỉ dùng input tốt |

Vì mode `legacy` cố ý không thực hiện quarantine, A là baseline raw sau khi
sửa unit MTF. B/C/D lần lượt đo direct-drive giảm, route chỉ qua interaction,
và loại hẳn nhóm nghi ngờ. Chỉ promote nếu OOS giữ hoặc cải thiện
expectancy/PF/DD, không chỉ vì equity một đoạn in-sample tăng.

## Điều kiện kiểm chứng run kế tiếp

1. Histogram H1 EMA/volatility không còn dính `±1.393`; báo median, IQR,
   min/max và saturation rate theo M5/M15/H1.
2. Attribution của entry tách BUY/SELL, cho biết raw và interaction đóng góp
   riêng; management interaction active phải bằng 0.
3. Báo expectancy shrinkage theo bucket có support thực tế, không report
   bucket dưới minimum sample như edge.
4. So sánh checkpoint đầu/cuối trên frozen replay: decision divergence,
   selection/action mix, H1 distribution và OOS trade metrics.
