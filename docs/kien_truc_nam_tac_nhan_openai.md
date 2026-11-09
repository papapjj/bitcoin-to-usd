# Kiến trúc năm tác nhân dùng chung một bộ tham số

Tài liệu này mô tả cấu hình hiện hành của dự án và phân biệt phần bám theo công bố chính thức của OpenAI với phần bắt buộc phải thích nghi cho môi trường giao dịch.

## Đường xử lý hiện hành

Năm tác nhân dùng chung đúng một bộ tham số. Mỗi tác nhân giữ trạng thái ẩn và ô nhớ riêng. Cả năm được tính trong cùng một lượt để phép gom chéo có thể lấy cực đại 25% đầu của véc-tơ trước tầng nhớ.

Quan sát cầu nối vẫn gồm 137 giá trị và được chia theo ngữ nghĩa trước khi ghép:

- 78 giá trị thị trường (giá, lợi suất nhiều chân trời, nến, chỉ báo, đa khung thời gian, timing và Wyckoff) đi qua hai lớp tích chập 1D, sau đó gom trung bình+cực đại và chiếu thành 256 chiều;
- 8 giá trị tài khoản và 11 giá trị trạng thái thực thi đi qua hai MLP riêng, mỗi nhánh 64 chiều;
- khối 5 vị thế × 8 giá trị là một tập không thứ tự: cùng hai lớp xử lý mọi vị thế, giữ véc-tơ 64 chiều của từng vị thế cho đầu chọn mục tiêu và lấy cực đại để tạo trạng thái toàn tập;
- nhận dạng tác nhân và hành động trước được biến thành hai véc-tơ học được 32 chiều.

Các phần trên được ghép thành 512 chiều, chiếu đầy đủ sang 1024 chiều, rồi gom cực đại 25% đầu giữa năm tác nhân. Dữ liệu liên tục đã được Python chuẩn hóa đúng một lần ở biên cầu nối; encoder không chuẩn hóa lần hai.

Lõi thời gian là đúng một tầng nhớ dài-ngắn có cổng 4096 phần tử. Sau LSTM là
một Transformer giao dịch-specific dùng năm hidden hiện thời làm năm token:
bottleneck 512, 8 attention head, FFN 2048 và residual trở lại 4096. Output
Transformer cấp cho các head nhưng không thay hidden/cell hồi tiếp riêng của
từng agent. Không có chồng nhiều tầng hồi tiếp, bộ chọn chuyên gia, mạng thời
điểm riêng, mạng tò mò, mạng luật hoặc mạng rủi ro phụ.

Đầu hành động được phân nhánh. Loại hành động chính được tính bằng phép chiếu
representation 4096 sau Transformer sang véc-tơ truy vấn và tích vô hướng với
các véc-tơ hành động học được. Mục tiêu vị thế được chọn bằng chú ý trên các
véc-tơ vị thế, có mặt nạ học được phụ thuộc vào hành động đã chọn. Các tham số
thời điểm, mức rủi ro và khoảng cách vào lệnh có đầu phân loại riêng. Giá trị
trạng thái là một phép chiếu riêng từ cùng representation 4096.

## Đường học

OpenAI Five công bố mẫu huấn luyện 16 bước; bản trading hiện chủ động dùng 64 bước và truyền ngược xuyên thời gian qua toàn bộ đoạn đó. Hàm cập nhật chính sách dùng giới hạn 0,2. Hệ số của ước lượng lợi thế suy rộng là 0,95. Phần thưởng được chuẩn hóa bằng ước lượng chạy của độ lệch chuẩn. Phần thưởng cá nhân của năm tác nhân có thể được trộn với trung bình đội theo hệ số từ 0,3 đến 1,0. Phần thưởng tạo hình không phải kết quả cuối được giảm theo 0,6 lũy thừa thời gian tính theo mỗi 10 phút.

Bộ tối ưu dùng hai hệ số trung bình mũ 0,9 và 0,999. Trung bình bình phương đạo hàm được cập nhật từ đạo hàm chưa cắt; đạo hàm dùng để cập nhật được chặn theo từng tham số trong biên năm lần căn bậc hai của ước lượng trung bình bình phương. Hệ số khuyến khích phân tán có biên công bố 0,01 đến 0,001. Tốc độ học dùng trong cấu hình nằm trong biên công bố 5e-5 đến 1e-6. Khoảng thời gian gán công trạng có thể tăng từ 60 đến 840 giây.

## Những phần không thể đồng nhất với Dota 2

Dự án giao dịch không có bản đồ ô lưới của Dota. Cấu trúc giá vẫn có tính cục bộ, nên bản thích nghi dùng tích chập 1D trên thứ tự đặc trưng ổn định thay vì giả tạo một ảnh 2D. Khi cầu nối cung cấp ma trận thời gian × mức giá thật, nhánh này mới có đủ dữ liệu để chuyển sang tích chập 2D. Dự án cũng chưa có đội đối phương do policy điều khiển, nên không tuyên bố đã tái tạo self-play của Five.

Ngoài ra tài liệu công khai của OpenAI không công bố mọi độ rộng trung gian, hàm kích hoạt, cách khởi tạo và mã nguồn nội bộ. Các độ rộng trước tầng nhớ, kích thước véc-tơ vị thế, véc-tơ tác nhân và véc-tơ hành động trong dự án là lựa chọn thích nghi, không được coi là thông số đã khôi phục từ OpenAI.

Encoder có phiên bản kiến trúc 4 và policy có phiên bản kiến trúc 2. Checkpoint
policy v1 thiếu Transformer không được resize, migrate hoặc nạp một phần; runtime
yêu cầu tạo checkpoint v2 mới.
