# Bảo tồn edge bằng champion–shadow

## Vấn đề được sửa

Retention trong optimizer bảo vệ response trên các anchor held-out, nhưng
không chứng minh expectancy/PnL còn dương ở regime tương lai. Luồng
walk-forward cũ còn có thể gom các row của nhiều artifact khác nhau thành một
ma trận rồi copy artifact có fitness cao nhất, dù artifact đó chỉ được frozen
test trên một cửa sổ. Vì vậy `promotion_stability_report` chỉ là thống kê độ
phủ thí nghiệm, không còn được dùng làm quyền promote artifact.

## Contract hiện tại

Lệnh `huan-luyen-champion-he-chuyen-gia-mt5` thực hiện transaction sau:

1. Hash và khóa logic artifact `--champion`; không ghi vào đường dẫn này.
2. Train nhiều shadow độc lập từ cùng champion, mỗi shadow có training seed
   riêng và `resume_existing=false`.
3. Frozen-test **từng shadow** trên cùng tập cửa sổ tương lai, đồng thời
   frozen-test champion trên đúng từng cửa sổ.
4. Mỗi ô phải qua cả `frozen_oos_promotion_gate` và
   `paired_oos_non_regression_gate`: đủ lifecycle, reconcile report/ledger,
   feature parity, real-tick evidence, DD, policy-origin LCB, scope âm và không
   hồi quy so với champion.
5. `champion_candidate_matrix_gate` bind row bằng SHA256. Mỗi candidate hash
   phải có đúng một row cho mọi cửa sổ và mọi row đều pass. Số candidate seed
   ổn định phải đạt cấu hình `promotion_min_seeds`.
6. Chỉ các candidate hoàn chỉnh mới được xếp hạng. Candidate có LCB tệ nhất
   cao nhất được chọn (`maximin`), sau đó mới dùng mean LCB để phá hòa.
7. Artifact thắng được copy nguyên tử sang `--output` riêng. Nếu bất kỳ gate,
   train hoặc OOS arm nào fail, output không được tạo và champion không đổi.

Mặc định profile production yêu cầu 3 cửa sổ × 3 candidate seed. Có thể giảm
hai con số trên CLI để debug contract và tiết kiệm thời gian, nhưng gate vẫn
giữ floor từ YAML: run thiếu 3×3 chỉ ghi audit, không tạo output promoted.

## Cách chạy

```powershell
python -m self_growing_trader.cli `
  --config configs/he_chuyen_gia_mot_context_mot_policy.yaml `
  huan-luyen-champion-he-chuyen-gia-mt5 `
  --champion outputs/he_chuyen_gia/seeds/champion.pkl `
  --output outputs/he_chuyen_gia/candidates/champion_candidate.pkl `
  --audit-dir outputs/he_chuyen_gia/audit/champion_matrix_001 `
  --train-from 2025.01.01 `
  --train-to 2025.01.15 `
  --validation-count 3 `
  --validation-days 1 `
  --candidate-seed-count 3 `
  --seed 20260803 `
  --model 4 `
  --keep-mt5-open
```

Các cửa sổ validation liên tiếp bắt đầu tại `--train-to`, không giao với đoạn
train. `--audit-dir` phải mới/rỗng để không trộn report cũ. `--output` phải
khác champion và chưa tồn tại.

## Artifact và audit

- `summary.json`: hash champion, từng shadow, toàn bộ gate, candidate được chọn
  và trạng thái `promoted`.
- `seed_*_window_*.json`: bằng chứng chi tiết của từng ô.
- `oos_artifacts/`: report, bridge, execution ledger và feature audit được chụp
  trước khi arm MT5 kế tiếp ghi đè tên runtime.
- `shadow_seed_*.pkl`: candidate bị loại vẫn giữ để forensic; không được runtime
  production đọc tự động.

Lệnh train thường `huan-luyen-he-chuyen-gia-mt5` chỉ được tạo một artifact
nghiên cứu mới. Nếu `--output` đã tồn tại hoặc có `--seed-system`, online CLI
fail-closed trước khi mở MT5 để không thể resume/deploy một candidate chưa qua
gate. Muốn chạy từ đầu phải truyền `--fresh-system` cùng một `--output` nghiên
cứu riêng; `--frozen` vẫn được đọc checkpoint để làm đối chứng bất biến.
Walk-forward thường vẫn tạo candidate/audit, nhưng không thể promote bằng cách
trộn nhiều artifact; mọi online continuation và artifact deployment phải dùng
transaction champion–shadow ở trên.
