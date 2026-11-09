# Báo cáo nâng cấp V9 — Quan sát đầy đủ mạng context–expert

## Vấn đề gốc

Các bản log cũ từng hiển thị `hanh_dong_policy = -1` và `hanh_dong_hop_le = -1`. Đây là sentinel nội bộ của đường legacy/bridge; khi hiển thị như action, nó gây hiểu nhầm rằng policy đã ra một lệnh không xác định.

Ngoài ra, UI từng có các lớp cắt dữ liệu: giới hạn event byte, active node/edge, state delta, attribution, history và các budget render topology.

## Sửa thực tế

1. **Chuẩn hóa hồ sơ quyết định V9**
   - schema hồ sơ nâng lên 4;
   - chỉ xuất `hanh_dong_nghia_vu`, `hanh_dong_gui_mt5`, `hanh_dong_wire_mt5`;
   - action contract thiếu/không hợp lệ → `KHÔNG_GỬI_LENH`, index `null`, không có `-1` trong hồ sơ;
   - lý do quyết định chỉ mô tả action nghĩa vụ/wire, không còn mô tả policy/legal legacy.

2. **Tách internal contract khỏi log người dùng**
   - bridge vẫn có thể dùng biến nội bộ để kiểm tra action contract;
   - heartbeat/console V9 công bố semantic/wire thay vì `policy_action`/`legal_action`;
   - UI không đọc hoặc render các trường legacy.

3. **Bỏ quota ứng dụng**
   - các quota log/config V9 đều bằng 0;
   - 0 = full, không truncate;
   - Java tailer, snapshot history, canvas static graph và Python reader đồng bộ semantics 0 = unlimited.

4. **Theo dõi phát triển mạng chi tiết**
   - thêm stream `theo_doi_phat_trien_mang.jsonl`;
   - snapshot full topology ở baseline/trial/decision/feedback/development/lifecycle;
   - tab UI mới render sâu toàn bộ map/list: node, edge, gate, weight, utility, plasticity, fade, lifecycle, activity, delta, growth evidence, owner và action contract;
   - nhật ký structural chỉ ghi thay đổi topology thật;
   - 3D đánh dấu tím mọi neuron/cạnh thay đổi gần nhất và Inspector có bộ lọc tương ứng.

## Kết quả kiểm thử

- Python compile pass.
- Java Swing compile pass.
- Bộ test explainability/observatory/action contract: **23 passed**.

## Phần chưa thể xác minh trong môi trường đóng gói

- Compile source EA MQL5 thành EX5.
- Strategy Tester/MT5 live và execution ledger thật.
- Feature parity đủ 119 input giữa Python và EA.
- Hiệu năng thực tế khi topology/log vượt tài nguyên phần cứng của máy.

V9 không giấu dữ liệu để duy trì tốc độ UI. Khi graph lớn, máy có thể chậm hoặc hết heap/disk; đó là giới hạn vật lý được hiển thị trung thực, không phải limit logic bị cài ẩn.
