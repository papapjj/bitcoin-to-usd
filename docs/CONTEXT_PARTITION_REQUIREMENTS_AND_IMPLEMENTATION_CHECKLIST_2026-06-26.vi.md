# Checklist yêu cầu và kỹ thuật — Context Partition V12

**Phạm vi:** `self_growing_trader` / M1 graph / ExperienceSuperNetwork / MT5 bridge.  
**Mục tiêu:** mọi quyết định phải được tạo từ một bức tranh tình huống thống nhất, có thời gian đúng, action hợp lệ, và credit assignment truy ngược được về đúng context đã sinh hành động.

---

## 1. Contract context chuẩn

### 1.1 Yêu cầu bắt buộc

- [x] **CTX-R01 — Một decision chỉ có một snapshot context bất biến.**
  - Mỗi tick/sequence được gán `frame_id` và `sequence_id`.
  - Một `ContextFrame` không được sửa sau khi tạo; temporal fields được trả bằng bản frame mới qua `with_temporal()`.
  - `StepTrace` lưu `context_frame_id`, schema, canonical key, goal, learning drive, action mask và audit.

- [x] **CTX-R02 — Không dùng cùng vector space cho raw market và latent M1 state.**
  - Memory retrieval của policy, dù được gọi từ raw observation hoặc encoded M1 state, luôn dùng `ContextFrame.canonical_memory_key`.
  - Latent M1 chỉ là trạng thái reasoning nội bộ; không còn là memory key của thế giới thị trường.

- [x] **CTX-R03 — Clock context tăng đúng một lần cho một frame.**
  - `ExperienceSuperNetwork.advance_context_frame(frame)` là đường duy nhất được phép advance temporal state trong quyết định online.
  - `observe_transition()`/absorb outcome không được làm delta/short/regime advance thêm cho frame đang active.

- [x] **CTX-R04 — M1 phải biết action hợp lệ trước selector.**
  - `allowed_action_mask` đi từ MT5 bridge vào `ContextFrame`, rồi vào M1.
  - Candidate có primary action không hợp lệ bị loại trước softmax/selection.
  - Có final fail-closed mask tại policy boundary để bảo vệ khi caller không tuân contract.

- [x] **CTX-R05 — Goal, account, position, execution là context online của M1.**
  - Không còn bỏ `goal` trong `M1GraphModule.forward()`.
  - Goal, portfolio, position, execution và temporal state được chiếu có chủ đích vào graph regions.

- [x] **CTX-R06 — `no_trade_streak` không phải market edge.**
  - Scarcity/cadence/no-trade được chuyển thành `learning_drive`.
  - `learning_drive` có thể tăng nhu cầu thu thập evidence/probe nhẹ, nhưng không được giả tạo `market_opportunity`, `opportunity_cost` hay BUY/SELL evidence.

### 1.2 Cấu trúc `ContextFrame`

```text
ContextFrame
├── identity: frame_id, sequence_id, timestamp
├── world
│   ├── market_vector                 # base market + extra market
│   ├── position_vector               # 5 position slots
│   ├── account_vector                # account/equity/risk state
│   └── execution_vector              # bridge execution features + external broker state
├── control
│   ├── goal_vector                   # return/risk/session objectives
│   ├── learning_drive                # coverage/uncertainty, không phải entry evidence
│   └── allowed_action_mask           # legality trước selector
├── memory
│   └── canonical_memory_key          # v8, 46 dimensions
└── temporal
    ├── current
    ├── delta
    ├── short
    └── regime
```

**File triển khai:** `src/self_growing_trader/context_frame.py`

---

## 2. Phân vùng feature v4 (119 input)

| Vùng | Index bridge | Ý nghĩa | Chủ thể dùng chính |
|---|---:|---|---|
| Base market | `0..33` | giá, trend, momentum, volatility cơ bản | encoder, predictor, critic |
| Position slots | `34..73` | 5 vị thế, side, age, R, PnL, TP/SL state | memory, predictor, critic |
| Account | `74..81` | equity/balance/drawdown/open-risk | memory, critic, selector |
| Extra market | `82..107` | market features mở rộng | encoder, predictor, critic |
| Execution | `108..118` | spread/broker/execution fields | encoder, predictor, critic, selector |

### Checklist mask semantic

- [x] **CTX-T01 — Không cắt vector theo tỷ lệ index.**
  - `_component_input_mask()` dùng slice semantic đúng schema.
- [x] **CTX-T02 — `state_encoder` chỉ nhận market + execution.**
- [x] **CTX-T03 — `working_memory` nhận market + position + account.**
- [x] **CTX-T04 — `predictor` nhận market + position + execution.**
- [x] **CTX-T05 — `critic` và `selector` được quyền thấy toàn bộ context cần cho lựa chọn/rủi ro.**
- [x] **CTX-T06 — Policy input nhỏ trong test không giả làm schema MT5.**
  - Fallback: module nhận toàn input thay vì cắt dải phần trăm tùy tiện.

**File triển khai:** `src/self_growing_trader/self_growing_runner.py:_component_input_mask`

---

## 3. Canonical memory key và similarity

### 3.1 Contract V8

Độ rộng memory key giữ **46** để còn tương thích kích thước checkpoint, nhưng ý nghĩa vector được nâng lên schema V8:

```text
K_v8 = [ World(32), Account(8), Position(6) ]
World(32) = [ pool(Market 60 → 24), pool(Execution 16 → 8) ]
Account(8) = account/equity/risk canonicalization
Position(6) = pool(Position 40 → 6)
```

Trong đó Execution 16 gồm 11 execution feature của observation và 5 broker/execution scalar do bridge cấp.

### Checklist kỹ thuật

- [x] **CTX-M01 — Không truncate extra market/execution khi tạo key.**
  - Mean-pooling có deterministic bin boundaries thay vì cut/pad.
- [x] **CTX-M02 — Similarity không dùng cosine của vector nối chung.**
  - `blockwise_cosine()` giữ trọng số: World `0.50`, Account `0.30`, Position `0.20`.
- [x] **CTX-M03 — Cell V8 chỉ match ContextFrame V8.**
- [x] **CTX-M04 — Legacy latent/raw key không bị coi là cùng hệ tọa độ với V8.**
  - Legacy cells được giữ audit/quarantine; chúng không influence retrieval V8.
- [x] **CTX-M05 — Parameter replacement giữ parameter trong metadata, không ghi đè dimension của context key.**
  - Proposal V8 truy hồi theo canonical world context và chỉ apply khi scope action/lifecycle khớp.

### Migration bắt buộc

- [x] Checkpoint cũ vẫn load được.
- [x] Cũ được đánh dấu bằng `legacy_context_schema_versions` thay vì chuyển đổi giả.
- [ ] **CTX-M06 — Rebuild legacy memory từ journal V8.**
  - Cần replay các lifecycle/journal có raw observation + account/position/execution đủ schema.
  - Không được “convert” một encoded state cũ thành canonical market context bằng padding hoặc resize.

---

## 4. Temporal context và sequence semantics

### Contract

```text
MT5 sequence/tick mới
→ create ContextFrame
→ advance_context_frame đúng một lần
→ query memory/M1 nhiều lần vẫn dùng cùng temporal snapshot
→ select/execution
→ observe outcome chỉ ghi evidence và next-state; không advance clock của frame đã quyết định
```

### Checklist

- [x] **CTX-TP01 — Frame cùng `frame_id` không tăng `context_step` lần hai.**
- [x] **CTX-TP02 — Delta/short/regime mô tả tick/sequence world context, không mô tả số lần nội bộ query.**
- [x] **CTX-TP03 — Outcome không làm temporal state nhảy thêm một lần trong cùng decision.**
- [x] **CTX-TP04 — Trace lưu frame ID để lifecycle outcome tìm đúng decision context.**
- [ ] **CTX-TP05 — Cần xác minh bridge cấp sequence ID monotonic khi MT5 reconnect/restart.**
  - Điều kiện deploy: sequence ID phải reset rõ ràng theo `round/session`; không reuse frame ID giữa hai bar khác nhau.

---

## 5. Cách M1 tương tác với context

| Region | Context được chiếu | Chức năng |
|---|---|---|
| Memory | recalled memory, market, portfolio, position, temporal | truy hồi episode/khái niệm cùng tình huống |
| Predictor | market, position, execution, temporal, goal, recalled memory | dự báo hậu quả action trong điều kiện thực thi hiện tại |
| Critic | market, portfolio, position, execution, goal, learning drive, memory | đánh giá utility/risk/goal trade-off |
| Selector | action bias, critic bias, goal, portfolio, execution, learning drive | chọn hypothesis hợp lệ |

### Checklist

- [x] **CTX-G01 — Goal có ảnh hưởng đo được tới candidate score trước post-gate.**
- [x] **CTX-G02 — Portfolio/account có ảnh hưởng đo được tới candidate score trước post-gate.**
- [x] **CTX-G03 — Position và execution không còn chỉ là risk gate hậu quyết định.**
- [x] **CTX-G04 — Legal action mask được áp trước softmax candidate.**
- [x] **CTX-G05 — Final output vẫn bị mask để bridge không sửa intent muộn.**
- [ ] **CTX-G06 — Cần OOS attribution report theo từng region.**
  - Mục tiêu: log delta candidate score do market/account/position/execution/goal tạo ra; phát hiện region chết hoặc dominant bất thường.

---

## 6. Learning drive và exploration an toàn

### Nguyên tắc

```text
market opportunity = bằng chứng market / prediction / recalled result
learning drive     = thiếu evidence / novelty / no-trade / coverage deficit
```

`learning drive` không được dùng làm lý do trực tiếp để mở BUY/SELL.

### Checklist

- [x] **CTX-L01 — `no_trade_streak`, `cadence_deficit`, starvation đi vào vector drive riêng.**
- [x] **CTX-L02 — Không đè `opportunity_cost` và `missed_opportunity` bằng starvation runtime.**
- [x] **CTX-L03 — Neural probe chỉ được tăng nhẹ khi có uncertainty; entry cần market opportunity + legality + risk budget.**
- [x] **CTX-L04 — Test contract cũ “starvation bắt buộc BUY/SELL” đã được loại bỏ.**
- [ ] **CTX-L05 — Cần đo exploration rate riêng theo market regime.**
  - Nếu no-trade rất dài nhưng market evidence thấp, system phải log `evidence_shortfall`, không được tăng forced-entry rate.

---

## 7. MT5 bridge contract

### Checklist

- [x] **CTX-B01 — Bridge gửi allowed action set trước `policy.act()`.**
- [x] **CTX-B02 — Bridge gửi execution context: spread, capacity, pending order, legal fraction, day-trade pressure.**
- [x] **CTX-B03 — Bridge gửi goal context: min RR, drawdown budget còn lại, position capacity, holding budget, turnover capacity.**
- [x] **CTX-B04 — Goal context không chứa “đủ số lệnh/ngày”, tránh biến quota thành lệnh ép.**
- [x] **CTX-B05 — Có fallback tương thích cho policy cũ không có `set_decision_context()` hoặc `goal=`.**
- [ ] **CTX-B06 — Chạy MT5 Strategy Tester để xác minh request thực có `frame_id` duy nhất và mask khớp action EA thực thi.**

**File triển khai:** `src/self_growing_trader/mt5_runner.py`

---

## 8. Credit assignment và self-development

### Checklist

- [x] **CTX-C01 — Episode/action trace mang canonical context key của decision gốc.**
- [x] **CTX-C02 — Outcome giữ lại `next_vector` theo canonical V8.**
- [x] **CTX-C03 — Replacement proposal phải match action side/lifecycle scope và canonical context.**
- [x] **CTX-C04 — Shadow proposal không được thành global rule ngay sau một loss.**
- [x] **CTX-C05 — Parameter proposal V8 không ghi đè phần position của context vector.**
- [ ] **CTX-C06 — OOS promotion phải báo cáo `context_cluster × action_scope × replacement` trước khi promote.**

---

## 9. Regression test matrix

| Contract | Test |
|---|---|
| Giữ đủ các feature group | `test_context_frame_keeps_every_semantic_bridge_group` |
| Semantic component masks | `test_component_masks_are_semantic_not_percentage_ranges` |
| One clock advance/frame | `test_context_clock_advances_once_per_frame_and_not_on_outcome` |
| Không trộn raw/latent space | `test_memory_queries_use_active_canonical_frame_not_raw_or_latent_space` |
| Goal/account vào selector trước post-gate | `test_goal_and_portfolio_change_graph_selection_before_post_gate` |
| Legal mask trước policy output | `test_legality_mask_is_visible_before_policy_returns_action` |
| Learning drive không là entry evidence | `test_learning_drive_is_audited_without_becoming_market_entry_evidence` |
| V8 memory retrieval hoạt động | `test_only_schema_v8_canonical_cells_can_bias_context_retrieval` |
| Legacy latent strategy quarantine | `test_experience_supernetwork_quarantines_legacy_latent_strategy_memory` |
| Parameter replacement V8 | `test_parameter_replacement_is_shadow_scoped_then_validated` |

---

## 10. Điều kiện nghiệm thu trước khi dùng tiền thật

- [x] Static compile pass.
- [x] Unit/regression suite pass trong môi trường Python.
- [ ] MT5 Strategy Tester: verify `allowed_action_mask` và action executed khớp log cho tối thiểu 3 regime.
- [ ] Walk-forward/OOS: kiểm tra tỉ lệ action rejected-after-policy phải giảm về gần 0.
- [ ] Journal audit: `context_frame_id`, `context_schema_version=8`, `context_key_source=canonical_context_frame` phải xuất hiện mọi policy-origin event.
- [ ] Drift test: restart/resume không reuse frame ID và không dùng legacy latent memory để bias action V8.
- [ ] Live-safety dry run: max DD, spread spike, invalid stop distance, margin shortage, partial-close lifecycle.

---

## Quy tắc bất biến sau nâng cấp

1. **Một decision — một ContextFrame — một temporal advance.**
2. **World context không bao giờ bị thay bằng latent M1 state trong memory key.**
3. **Legality trước selection; bridge chỉ là safety enforcement cuối.**
4. **Goal/risk/account/position ảnh hưởng cách mạng suy nghĩ, không chỉ chặn output sau cùng.**
5. **Thiếu trade không phải bằng chứng entry.**
6. **Memory cũ sai không gian được quarantine, không “migrate giả”.**
7. **Growth/replacement phải được credit theo context V8 và scope action thực thi.**
