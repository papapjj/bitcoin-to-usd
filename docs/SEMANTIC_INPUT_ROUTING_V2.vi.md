# Semantic input routing V2

## Mục tiêu và wire contract

`role_isolated_v1` vẫn là mode cấu hình, nhưng semantic-route contract đã là
version 2. Nó tách quyền đọc trước Direct-RNN mà không đổi bridge MT5:

- EA/CSV luôn có **137** feature: 58 market, 40 position, 8 account, 11
  execution, 4 timing và 16 Wyckoff;
- candidate `context_history_inputs: 2` chỉ nối nội bộ hai token M15 `t-1/t-2`
  tại index 137/138. Với enrichment legacy policy có 139 chiều; profile active
  nối tiếp 13 interaction chỉ-cho-entry nên policy có 152 chiều, còn bridge vẫn 137;
- protected v13 để history `0`, giữ đúng input cũ.

## Quyền đọc

| Nhánh | Input Direct-RNN được phép đọc | Quyết định |
|---|---|---|
| Entry | 58 market + timing/Wyckoff + (candidate) context `t-1/t-2` + 13 interaction trend/location/volatility/Wyckoff khi enrichment active | `HOLD`, BUY/SELL, market/limit/stop |
| Management | 40 position + 11 execution + (candidate) context `t-1/t-2` | cancel, close, trailing |
| Account/risk | account context xác định | risk/volume, hard safety `HOLD` |

Entry không đọc position/account/execution. Management không đọc
market/timing/Wyckoff/account. Hai token context là metadata thị trường M15
đã chốt trước decision, không phải state position hay account. Mọi tail khác
chưa được phân loại vẫn bị mask.

## State, học và safety

Hidden state, deep recurrent state và history TBPTT tách theo route `entry`,
`management`, `legacy`, vì chỉ mask input không ngăn state management rò sang
entry sau đó. Full-RL AdamW/cosine là writer duy nhất của `w_in`, `w_rec`,
`w_out`, value head, gate và deep stack; routing/history chỉ là transform xác
định trước forward, không có optimizer hay credit riêng.

Hard account-risk và duplicate-order safety chạy sau logits. Nếu nó đổi action
thành `HOLD`, trace bị đánh dấu actor-ineligible, nên không tạo policy gradient
giả. Candidate để `semantic_duplicate_entry_block_enabled: false` vì mask side
cũ có thể làm argmax rơi sang side đối ứng và mở hedge ngoài ý muốn.

## Telemetry và regression

Trace ghi `semantic_route`, số feature active/masked theo group và
`semantic_route_context_history_active`. Facade ghi nhãn raw
`context_history_previous_contexts`, code `context_history_values`, kích thước
wire/policy và direct-input attribution có thể hiện index 137/138; interaction
active chiếm các index 139..151 và không được route sang management.

Regression bắt buộc chứng minh entry không đọc stateful wire field, management
state không rò sang entry, gradient Full-RL bằng 0 ở group bị cấm, hai context
token đi đúng cả hai route, và checkpoint v13 legacy vẫn load được. Điều này
không tự chứng minh edge: vẫn cần frozen MT5 A/B và execution ledger thật.
