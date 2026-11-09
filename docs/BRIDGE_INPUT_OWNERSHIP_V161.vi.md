# V161 — một owner chuẩn hóa input cho bridge MT5

## Lỗi đã loại bỏ

Trước V161, EA chuẩn hóa 137 feature trong ComputeAllInputs, sau đó Python
lại gọi normalize_input_vector khi đọc cùng request. Vì raw_inputs được chụp
sau bước của EA, sidecar raw cũng đã bị biến đổi. Hai bước này làm các distance
cấu trúc bị nén hai lần; ví dụ distance_to_swing không còn đủ biên độ để phân
biệt near, mid và far.

## Contract V161

    EA ComputeAllInputs  -> raw 137 feature trên CSV bridge/audit
    Python read request  -> lưu raw_inputs bất biến
    Python normalizer    -> policy inputs đúng một lần theo YAML
    MarketRegime         -> chỉ đọc raw_inputs
    Direct-RNN/TimingNet -> chỉ đọc policy inputs

EA không còn có InputNormalizationMode, NormalizeInputVector hoặc helper
chuẩn hóa MQL. File tester set cũng không còn khóa mode này. Mode
learning.input_normalization_mode trong YAML vẫn được giữ vì Python là owner
duy nhất của policy representation.

## Tương thích và kiểm chứng

- Bridge schema tăng lên **v9**. EA phải được biên dịch lại; integrity check sẽ
  từ chối .ex5 cũ hơn source.
- Feature audit bridge v9 trở lên mặc định được so với Python ở miền **raw**.
  scripts/verify_feature_parity.py --audit-inputs-normalized chỉ dành cho audit
  lịch sử v8 trở xuống.
- Regression bảo vệ ba bất biến: request giữ raw sidecar riêng, Python tạo đúng
  một vector chuẩn hóa, và source EA không thể tái xuất hiện normalizer MQL.
- Candidate V160 đã chạy trước V161 chỉ dùng để chứng minh audit-credit; không
  được dùng để đánh giá entry edge vì nó chịu lỗi chuẩn hóa kép.

## Gate MT5 cho candidate kế tiếp

1. Log/build MetaEditor phải không lỗi, .ex5 mới hơn .mq5.
2. Feature-audit v9 phải báo audit_input_view=raw_bridge_v9.
3. Từ raw feature, bucket location phải có phân bố near/mid/far; không được
   kết luận từ policy-normalized vector.
4. Full-RL phải tiếp tục báo ledger_source=full_rl_audit, và audit-only
   expectancy không được trở thành writer của action, risk hay gate.
