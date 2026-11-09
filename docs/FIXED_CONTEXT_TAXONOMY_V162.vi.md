# V162 — taxonomy context cố định cho Full-RL

## Bằng chứng buộc phải sửa

Probe V161 chạy từ seed bất biến đã phục vụ 489 decision và settle 164
lifecycle. Dù không có lỗi bridge hay writer, artifact tạm đã tăng từ 9 lên 14
expert vì sinh các context NHANH. Hệ quả là một expert có 32 lifecycle chỉ
nhận 4 AdamW update, còn expert có 9 hoặc 3 lifecycle nhận 0 update. Tín dụng
không tích lũy đủ trong bất kỳ policy owner nào.

## Contract mới

- Taxonomy M15 là cố định: root trend/range/volatile và ba volatile subtype
  đã định nghĩa. Không còn context branch theo prototype.
- Một request có đúng một owner canonical; tất cả lifecycle của cùng regime
  quay về owner đó để tích đủ batch Full-RL.
- Không thêm mask, gate, prior BUY/SELL hoặc writer mới.
- Role specialist nếu profile bật vẫn là role tường minh; candidate hiện
  tắt chúng. Cơ chế branch tự phát đã bị xóa hoàn toàn.

## Migration

Khi load artifact cũ, mọi context có suffix NHANH bị xóa cùng selector
utility/outcome, trace owner, live writer và state branch cũ. Các field branch
serialized bị xóa trước lần persist kế tiếp. Seed gốc không bị ghi đè.

## Gate runtime

1. Candidate mới phải giữ số expert taxonomy cố định trong suốt MT5 run.
2. Mỗi owner xuất hiện đủ lifecycle phải có số AdamW batch tương ứng, không bị
   mất vào expert branch.
3. Chỉ sau đó mới đánh giá entry/SL/TP theo expectancy, frozen replay và OOS.
