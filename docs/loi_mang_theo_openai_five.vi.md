# Lõi mạng theo phần kiến trúc OpenAI Five đã công bố

Bản sửa này giữ lõi hồi tiếp Five-like là một tầng mạng nhớ dài–ngắn có cổng
duy nhất, rồi thêm một Transformer hậu-LSTM riêng cho bài toán giao dịch.

Phần được đối chiếu trực tiếp với công bố cuối năm 2019:

- một tầng hồi tiếp duy nhất;
- 4096 phần tử nhớ;
- trạng thái quan sát sau xử lý đi vào tầng nhớ;
- trạng thái ẩn của tầng nhớ đi vào đường representation sau LSTM;
- mỗi dòng chạy giữ trạng thái nhớ riêng;
- OpenAI Five công bố cửa sổ 16 bước; cấu hình trading hiện ghi đè rõ ràng thành 64 bước.

Phần thêm sau LSTM không phải thông số OpenAI Five được công bố: năm hidden hiện
thời là năm token của một `TransformerEncoderLayer`, dùng width 512, 8 head,
FFN 2048 rồi chiếu residual về 4096. Action/value/reasoning head đọc output này;
LSTM hidden/cell thô mới được lưu làm recurrent memory. Hợp đồng bridge vẫn là
137 input và 20 output.

Trong đường lõi mới, những cơ chế của lõi cũ sau đây không còn tham gia phép tính tiến:

- chồng nhiều tầng hồi tiếp;
- hệ số giữ trạng thái cố định;
- chuẩn hóa trạng thái ẩn;
- bỏ ngẫu nhiên phần tử ẩn;
- tái trọng số đặc trưng trước từng tầng;
- bộ nhớ truy hồi phụ chèn trực tiếp vào tiền kích hoạt;
- tăng hoặc giảm số tầng, tăng độ rộng, cắt bỏ phần tử nhớ trong quá trình phát triển cấu trúc.

## Phần không thể tuyên bố giống tuyệt đối

OpenAI không công bố mã nguồn đầy đủ của lõi OpenAI Five. Tài liệu công khai không xác định đầy đủ cách khởi tạo từng ma trận, thứ tự lưu bốn nhóm tham số trong bộ nhớ hay mọi chi tiết số học của thư viện dùng khi huấn luyện. Vì vậy bản sửa chỉ có thể khớp chính xác với những đặc điểm kiến trúc đã được công bố, không thể chứng minh giống từng phép tính nội bộ chưa được công bố.

Dự án này vẫn giữ 137 đầu vào của cầu giao dịch và 20 đầu ra giao dịch. Đây là
ngữ nghĩa của bài toán giao dịch, không phải ngữ nghĩa quan sát và hành động của
Dota 2. Vì có Transformer hậu-LSTM, tên chính xác của hệ hiện tại là
Five-like shared LSTM policy có cross-agent Transformer, không phải bản sao
nguyên trạng OpenAI Five.
