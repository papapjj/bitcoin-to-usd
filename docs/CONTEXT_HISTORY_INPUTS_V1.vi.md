# Hai input context trước cho Direct-RNN

## Hợp đồng

`he_chuyen_gia.context_history_inputs: 2` chỉ bật trên artifact candidate.
MT5 EA, CSV, feature parity và `INPUT_COUNT` vẫn giữ nguyên **137**. Ngay trước
forward, facade context-owner nối thêm hai scalar token Python-local. Nếu
`entry_feature_enrichment_mode: legacy`, Direct-RNN nhận **139** giá trị;
profile active `quality_interactions_v1` nối thêm 13 interaction entry sau
hai token nên nhận **152** giá trị. Hai token luôn nằm tại:

| Chỉ số local | Nghĩa |
|---|---|
| 137 | Context M15 đã chốt gần nhất, `t-1` |
| 138 | Context M15 đã chốt trước đó, `t-2` |

Context là biến phân loại; mỗi nhãn có một code ổn định, khác `0`. `0` dành
cho warm-up/unknown. Code không phải điểm trend hay khoảng cách kinh tế giữa
hai context. Telemetry luôn ghi nhãn gốc trong
`context_history_previous_contexts` và giá trị đã đưa vào mạng trong
`context_history_values`.

## Nhân quả và học

Ở thời điểm quyết định `t`, policy chỉ thấy hai nhãn đã có trước `t`. Router
M15 xác định nhãn hiện tại, tất cả proposal của cùng quyết định đọc cùng một
snapshot `t-1/t-2`, rồi facade mới push context hiện tại vào history. Lịch sử
reset khi bắt đầu phiên MT5 mới, cùng lúc reset RNN state; không mang sequence
của test cũ sang cửa sổ mới.

Full-RL lưu đúng forward cache của representation đang bật (139 với legacy,
152 với active enrichment) và AdamW/cosine cập nhật hai cột `w_in` token qua
gradient lifecycle bình thường. Không có optimizer, gate, mask hay
writer riêng cho history. Semantic routing v2 cho phép cả entry và management
đọc đúng hai token này, còn tail feature tương lai khác vẫn bị chặn.

## Tương thích

`context_history_inputs: 0` là mặc định và giữ v13 đúng 137 input. Khi load
candidate config với giá trị `2`, `ensure_compatible()` resize có kiểm soát
`w_in`, `gate_w`, STP/input mask và trạng thái optimizer của Direct-RNN từ
137 sang ít nhất 139 (hoặc 152 khi enrichment active); seed `.pkl` không bị ghi đè. Facade vẫn báo 137 input cho
`MT5BacktestConfig`, vì bridge không hề thay schema.

## Kiểm chứng bắt buộc

- trace đầu tiên có `[0, 0]`;
- trace thứ hai chỉ có `t-1`; trace thứ ba có đúng `[t-1, t-2]`;
- tất cả policy Direct-RNN được resize 139 hoặc 152 theo enrichment, nhưng facade/EA vẫn 137;
- frozen replay/OOS reset history trước cửa sổ mới;
- attribution có thể hiển thị index 137/138 nếu hai token ảnh hưởng margin
  action; điều đó là bằng chứng mạng dùng chúng, không tự chứng minh edge.
