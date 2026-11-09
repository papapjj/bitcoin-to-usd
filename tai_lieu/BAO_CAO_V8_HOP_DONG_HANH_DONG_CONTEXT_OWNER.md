# V8 — Hợp đồng hành động context-owner

## Mục tiêu

V8 xử lý lỗi P0 từng xuất hiện trong log: hypothesis/thinking chọn một hành động nhưng action cuối hoặc bridge lại nhận hành động khác.

Luồng bắt buộc mới:

```text
Context ổn định
→ owner policy độc lập
→ M1 chọn một hypothesis hợp lệ theo action scope của context
→ action nghĩa vụ cuối
→ legality/risk gate
→ wire MT5 cùng nghĩa vụ
```

## Những thay đổi thực hiện

1. **Action scope đi vào M1 trước khi selector chọn hypothesis.**
   - Context vào lệnh: chỉ `HOLD`, entry market/limit/stop.
   - Context quản lý BUY/SELL: chỉ `HOLD`, quản lý/đóng/bảo vệ vị thế.
   - Context lệnh chờ: chỉ `HOLD`, xử lý/hủy lệnh chờ.

2. **Fallback duy nhất là hypothesis HOLD hợp lệ.**
   - Chỉ được tạo khi tất cả hypothesis neural hiện tại đều nằm ngoài action scope.
   - Đây không phải policy chung, không tạo BUY/SELL và không thay thế policy owner.

3. **Memory/reasoning chỉ được sửa tham số sau khi action đã chọn.**
   - Không được đổi discrete action head sau selection.
   - Action bias hậu selection chỉ tác động risk/SL/TP/trailing/partial/management parameter heads.

4. **Trace tách rõ ba khái niệm.**
   - `selected_hypothesis_action_index`: action của hypothesis trước safety gate.
   - `semantic_action`: action policy thật sự đưa vào pipeline.
   - `wire_action`: action MT5 transport.
   - `wire_matches_semantic_action`: bắt buộc `true` hoặc execution bị chặn.

5. **Xác minh owner policy độc lập.**
   Mỗi trace có:
   - `policy_instance_uuid`
   - `graph_instance_uuid`
   - `weight_hash`
   - `memory_hash`

6. **Nhật ký structural chỉ chứa topology thay đổi thật.**
   `graph_credit_attribution` vẫn tồn tại trong telemetry nhưng không còn bị ghi như sự kiện mọc neuron/liên kết.

## Trạng thái kiểm thử

- `tests/test_he_chuyen_gia_ngu_canh.py`
- `tests/test_v8_context_action_contract.py`

Kết quả: pass trong môi trường đóng gói.

## Không xác minh được ở đây

- Compile MQL5 thành EX5.
- MT5 Strategy Tester và live tick.
- Feature parity thực từ EA sang Python.
- Lệnh MT5 có ticket thật sau bridge.
