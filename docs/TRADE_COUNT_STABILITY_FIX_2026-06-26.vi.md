# Bản vá ổn định số trade — 26/06/2026

> Ghi chú cập nhật 2026-07-01: tài liệu này mô tả lỗi lịch sử khi bridge còn ép BUY/SELL theo starvation. Code hiện tại đã loại bỏ market-entry bootstrap; bridge chỉ recovery bằng `CANCEL_ORDERS` cho pending order treo.

## Sự cố đã xác nhận

Lịch sử `outputs/supernetwork_trial_history.jsonl` cho thấy số lifecycle ở các full test giảm theo chuỗi:

```text
104 → 53 → 53 → 31 → 12 → 8 → 6
```

Journal đi kèm cho thấy các entry thực tế đều mang `action_origin=bootstrap`. Policy gốc chủ yếu phát `HOLD` hoặc một action đóng lệnh không hợp lệ khi chưa có vị thế; bridge sau đó ép BUY/SELL theo starvation. Vì vậy số trade trước đây không phản ánh chính sách M1 tự vào lệnh.

## Nguyên nhân gốc

1. `train-to-goal` truyền `enforce_promotion_gate=False`.
   - Candidate chỉ cần có soft progress score tốt hơn baseline.
   - Candidate 31, 12, 8 hoặc 6 lifecycle có thể được promote, dù evidence quá ít.

2. Runtime starvation chỉ đến bridge **sau** khi M1 chọn action.
   - M1 không biết đang thiếu opportunity/lifecycle.
   - Neural probe không được kích hoạt.
   - Bridge buộc entry sau ngưỡng 8 tick.

3. Bootstrap có thể ghi đè cả một market entry hợp lệ do policy vừa chọn.
   - Entry đó bị ghi nhận là `bootstrap`, không phải `policy`.
   - Policy-origin memory không thể tích luỹ đúng.

4. Outcome starvation của những trial thất bại còn tác động trực tiếp đến curiosity/action gain.
   - Lỗi trial cũ tiếp tục nhiễu quyết định live của policy mới.

## Thay đổi đã áp dụng

### A. Promotion không còn chấp nhận candidate ít dữ liệu

- `train_to_goal()` luôn bật hard promotion gate.
- Với `min_daily_trades=6` và cửa sổ 31 ngày, candidate cần ít nhất **40 lifecycle** đã đóng để có thể được promote.
- Ngay cả khi gọi `improve_self_growing(..., enforce_promotion_gate=False)`, candidate vẫn phải có tối thiểu execution evidence (mặc định ít nhất 8 lifecycle, ledger hợp lệ và reconciled).
- Soft progress score chỉ được dùng sau khi candidate đã qua evidence floor.

Điều này **không ép bot giao dịch nhiều hơn để lấy điểm**. Nó chỉ ngăn hệ thống gọi một candidate ít exposure là “tốt hơn” vì nó đã né thị trường.

### B. M1 nhận starvation trước khi quyết định

Ngay trước `M1GraphModule.forward()`, policy đưa các tín hiệu sau vào graph context:

```text
runtime_starvation
no_trade_streak
cadence_deficit
policy_entry_pressure
opportunity_cost
missed_opportunity
uncertainty
```

Khi entry pressure cao, M1 tự tạo hai candidate neural: market BUY và market SELL. Hướng được chọn bởi selector/hypothesis state của graph và memory/replacement bias, không luân phiên cứng theo sequence của bridge.

### C. Account risk vẫn là cổng chặn bắt buộc

`account_stress`, drawdown và open risk vẫn đi qua `entry_gate`.

- Stress thấp: M1 được phép thử entry candidate nhỏ.
- Stress cao: entry logits bị triệt tiêu; candidate probe không thể bypass risk gate.

### D. Bootstrap chỉ còn là fallback thật sự

Nếu policy đã chọn `MARKET_BUY` hoặc `MARKET_SELL` hợp lệ, bridge giữ nguyên action đó và ghi nhận:

```text
action_origin=policy
execution_bootstrap=false
```

Bootstrap chỉ ép market entry khi policy vẫn HOLD/không hợp lệ sau một no-trade streak kéo dài.

### E. Tách runtime starvation khỏi outcome starvation

- Runtime starvation: chỉ dùng để đề nghị neural exploration ngay trong phiên đang chạy.
- Outcome starvation: chỉ dùng cho replay/structural proposal/candidate selection.

Kết quả: trial thất bại trong quá khứ không còn liên tục đẩy policy hiện tại sang hành vi ngẫu nhiên hoặc HOLD/CLOSE collapse.

## Regression tests mới

- M1 high-opportunity tự chọn entry candidate; stress cao vẫn chặn entry.
- Policy truyền runtime starvation vào M1 trước action selection.
- Bridge không ghi đè market entry hợp lệ của policy.
- `train-to-goal` bật promotion gate.
- Exploratory promotion vẫn reject candidate có 7 lifecycle dù soft score tốt hơn.

Kết quả test cục bộ:

```text
255 passed
compileall: passed
```

## Cách chạy lại

Sao lưu checkpoint hiện tại trước:

```powershell
Copy-Item outputs\winner.pkl outputs\winner.pre_tradefix_backup.pkl
```

Giữ nguyên lệnh `train-to-goal` đang dùng. Không cần thêm cờ mới; hard promotion gate đã được bật trong mã.

## Dấu hiệu phải theo dõi sau lần chạy MT5 đầu tiên

1. `winner*.journal.jsonl`
   - `action_origin=policy` phải bắt đầu xuất hiện.
   - Tỷ lệ `action_origin=bootstrap` cần giảm dần, không còn 100%.

2. `train_to_goal.jsonl`
   - Mọi candidate có `fitness_lifecycle_count < 40` (với 6 trade/ngày, 31 ngày) phải bị reject.
   - Cần thấy `reject_reasons`, không được silently promote chỉ vì progress score.

3. `supernetwork_trial_history.jsonl`
   - Số trade có thể dao động; mục tiêu không phải tăng cưỡng bức.
   - Không được chấp nhận candidate giảm sample xuống dưới evidence floor.

## Giới hạn còn lại

Bản vá đã được compile và unit/regression test, nhưng chưa chạy MT5 Strategy Tester thật trong môi trường này. Cần xác nhận bằng một full-window pass rằng policy-origin entry được EA fill, journal ghi đúng provenance và candidate gate hoạt động với spread/slippage/commission thực.
