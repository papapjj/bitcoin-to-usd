# V190 — State-Aliasing Memory cho Direct-RNN

## Mục tiêu

V190 giải quyết trường hợp hai snapshot hiện tại gần như giống nhau nhưng được tạo ra bởi hai lịch sử khác nhau. Direct-RNN cũ đã có recurrent hidden state, nhưng hidden phải nén toàn bộ quá khứ tuần tự vào một vector và có thể quên hoặc làm hai đường đi khác nhau va vào cùng biểu diễn. V190 bổ sung một bộ nhớ làm việc có thứ tự để policy có thể đọc lại trực tiếp những quan sát quá khứ liên quan.

Đây không phải lookup rule cứng và không phải memory database dùng chung. Mỗi expert sở hữu tensor Q/K/V, thống kê optimizer và runtime buffer riêng.

## Data flow nhân quả

```text
quan sát đã route ở thời điểm t
        │
        ├── query Q(x_t)
        │
        ▼
buffer riêng của expert [x_(t-32), ..., x_(t-1)]
        │
        ├── key K(x_i) + relative_position(i)
        ├── value V(x_i)
        └── scaled dot-product attention + recency bias
        │
        ▼
memory context → projection → sigmoid gate
        │
        ▼
cộng vào base hidden preactivation của Direct-RNN
        │
        ▼
hidden stack → action/value
        │
        └── chỉ sau khi forward xong mới append x_t vào buffer
```

Reader dùng true Q/K/V attention trên tối đa 32 observation 137 chiều. Relative-position tensor và recency bias khiến cùng một tập snapshot nhưng khác thứ tự vẫn tạo context khác. Cùng current observation có thể tạo hidden/action khác khi fingerprint lịch sử khác.

Không có future bar trong buffer. `StateAliasingMemoryForward` đóng băng observation, token, Q/K/V activation, attention weight, gate và injection đúng tại decision để lifecycle đến trễ backprop theo decision cũ, không đọc runtime buffer mới hơn.

## Ownership, reset và persistence

- Mỗi module của mỗi context expert có một `StateAliasingMemoryState` độc lập.
- Khi semantic routing hoạt động, buffer được tách tiếp theo role; entry và management không đọc lịch sử của nhau.
- Trọng số Q/K/V, positional tensor, gate và Adam moments được lưu trong artifact/champion-shadow.
- Runtime buffer chỉ là working state. `reset_runtime_state()` xóa nó ở ranh giới run/OOS để không mang nến từ window trước sang window sau.
- `copy()` và pickle deep-copy memory; update một expert không đổi hash/tensor expert khác.
- Checkpoint cũ không có memory sẽ load với `enabled=false`, không tạo tensor và giữ nguyên output cũ.

## Học

Memory không có optimizer riêng. Gradient từ policy/value/auxiliary lifecycle đi qua hidden base rồi qua output projection, gate, value, softmax attention, Q/K/V, positional tensor và recency bias. Tất cả direction `state_memory.*` chỉ được ghi bởi `ModuleState.apply_rnn_adamw_cosine(..., source="full_lifecycle_rl")`.

Do đó V190 không thêm một reward mới. Nó làm tín hiệu học hiện có có thể gán credit cho câu hỏi: “phần nào của lịch sử khiến cùng trạng thái hiện tại này cần hành động khác?”. Các label đa nhiệm và TensorRuleBank vẫn cung cấp mục tiêu trừu tượng; V190 cung cấp lịch sử phân biệt để các mục tiêu đó sử dụng.

## Cấu hình

```yaml
learning:
  direct_rnn_state_memory_enabled: false
  direct_rnn_state_memory_capacity: 32
  direct_rnn_state_memory_size: 32
  direct_rnn_state_memory_min_history: 4
  direct_rnn_state_memory_temperature: 1.0
  direct_rnn_state_memory_injection_scale: 0.25
```

Profile production giữ `enabled=false`. Candidate hiện hành `configs/experiments/edge_discovery_v193_candidate.yaml` giữ memory tắt và TensorRule schema 6, artifact version 18.

Memory ở trạng thái cold trong bốn quyết định đầu của mỗi route. Khi chưa đủ lịch sử, injection đúng bằng 0. `injection_scale=0` cũng vô hiệu hóa tác động nhưng vẫn không nên dùng thay cho `enabled=false` khi cần baseline tương thích tuyệt đối.

## Telemetry

`action_details` của Direct-RNN ghi:

- enabled/ready và history length;
- history fingerprint;
- attention entropy, top history age và top weight;
- gate, context norm và injection norm;
- attention weights theo thứ tự causal;
- tổng injection norm của policy.

Optimizer telemetry có block `state_alias_temporal_memory`, fingerprint trước/sau, số tham số đổi và delta norm. Những trường này giúp phân biệt “memory tồn tại” với “memory thật sự đang đọc lịch sử và được học”.

## Kiểm thử và bằng chứng hiện tại

```powershell
python -m pytest -q tests/test_state_alias_memory_v190.py
```

Bộ test bao phủ: default-off, hành vi checkpoint cũ, cùng current/khác history, causal append, reset, finite-difference Q/K/V/output/position/recency/gate, sole-writer AdamW, expert isolation và deterministic pickle migration.

Toàn bộ regression workspace sau thay đổi: `874 passed in 179.94s`.

Python candidate debug tại `outputs/he_chuyen_gia/experiments/state_alias_memory_v190_debug_warm/manifest.json` xác nhận 9/9 expert có memory object và parameter fingerprint độc lập, tất cả trở thành ready sau warm-up, artifact v13, source config hash không đổi và bridge vẫn 137×20.

Đây mới là bằng chứng implementation. Nó chưa chứng minh memory tạo edge, giữ edge hoặc cải thiện PnL. Bước tiếp theo phải là MT5 paired run cùng seed/window/model, sau đó frozen OOS nhiều window và đánh giá riêng policy-origin expectancy/LCB, drawdown cùng độ ổn định của history-attention fingerprint.
