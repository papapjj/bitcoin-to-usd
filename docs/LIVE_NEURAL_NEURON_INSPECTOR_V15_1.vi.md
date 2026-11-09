# Live Neural V15.1 — Chế độ thông tin neuron

## Mục tiêu

Thêm một chế độ quan sát riêng trong cửa sổ **Mạng sống 3D** để xem thông tin của từng neuron mà không phải cố đọc một cụm node/edge dày đặc. Chế độ này chỉ đọc telemetry JSONL; không ghi ngược vào trainer, M1 graph, MT5, EXP ledger hay cơ chế growth/prune.

## Cách bật

1. Trong Java UI, tick **Ghi telemetry khi train**.
2. Bấm **Bật xem Live** để mở cửa sổ `Mạng sống 3D`.
3. Bấm nút **Thông tin neuron** trên header.
4. Tab **Neuron** ở panel bên phải sẽ được chọn.
5. Click một node trong không gian 3D hoặc chọn neuron trong danh sách.

Khi chế độ bật, nút chuyển thành **Tắt thông tin neuron**. Tắt chế độ này chỉ đổi cách trình bày/selection trong Java UI; train và telemetry tiếp tục bình thường.

## Dữ liệu hiển thị

Mỗi neuron có các phần:

- **Định danh:** ID, module, module ID, region, lifecycle, tuổi, `home_context_key`.
- **Trạng thái live:** active/inactive, activation, state và các trace field được trainer ghi trong event gần nhất.
- **Học và tồn tại:** bias, utility, plasticity, fade, `created_by`, `need_reason`.
- **Kết nối:** số liên kết vào/ra, số active, tổng độ lớn weight, top 8 incoming/outgoing edge theo active contribution rồi weight.
- **Raw topology fields:** field gốc do telemetry xuất ra, giới hạn 20 field để UI không quá nặng.

Danh sách neuron hỗ trợ:

- tìm theo ID, module, region, lifecycle hoặc context home;
- lọc chỉ neuron active;
- lọc neuron có plasticity;
- lọc candidate/mới sinh;
- lọc fade/prune.

## Liên hệ với đồ thị 3D

Khi chọn neuron:

- node được viền vàng và glow vàng;
- toàn bộ edge trực tiếp đi vào/đi ra của node được ưu tiên render, kể cả khi topology tĩnh đang bị giới hạn render;
- edge liên quan được tô vàng, edge active liên quan được tô vàng sáng;
- danh sách panel tự đồng bộ với node đã click trong graph.

Không có dữ liệu nào được giả lập. Nếu telemetry chưa có `decision` hay `activity`, activation/state sẽ là `0` hoặc không có field; đó là trạng thái dữ liệu thực tế, không phải lỗi của mode xem.

## Giới hạn

- Mode chỉ hiển thị trường mà trainer telemetry đã xuất ra. Nó không tự suy đoán gradient, activation lịch sử hoặc hidden state không được ghi.
- `active` là active trace của event gần nhất, không phải tổng số lần neuron đã từng active.
- Danh sách có thể update theo event mới; selection được giữ theo node ID khi topology vẫn còn node đó.
