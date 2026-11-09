# Kiến trúc một bộ não 4096

Cấu hình hiện tại không còn dùng nhiều chuyên gia độc lập. Mọi quyết định đi qua đúng một bộ trọng số chung.

Đường quyết định đang hoạt động:

1. 137 giá trị đầu vào từ cầu nối giao dịch.
2. Một tầng nhớ dài–ngắn có cổng gồm 4096 phần tử nhớ.
3. Một Transformer hậu-LSTM dùng năm agent làm năm token: 4096→512, 8 head,
   FFN 2048, 512→4096 và residual.
4. Các đầu phân rã tạo đúng 20 giá trị bridge cùng giá trị ước lượng trạng thái.

Không còn tham gia đường quyết định hoặc đường học của cấu hình mới:

- bộ chọn nhiều chuyên gia;
- chuyên gia kiểu vào lệnh;
- chuyên gia quản lý vị thế;
- chuyên gia biến động phụ;
- mạng thời điểm riêng;
- đầu học phụ;
- đầu dựng lại đầu vào;
- mạng tò mò;
- mạng đánh giá ích lợi;
- mạng đánh giá chi phí rủi ro;
- mạng tham số hành động phụ;
- đồ thị suy luận;
- ngân hàng luật học được;
- bộ nhớ trạng thái chèn vào lõi;
- mạng nhớ phụ và bộ phân loại kết quả vị thế.

Bộ nhận diện trạng thái thị trường cũ cũng không còn được gọi trước quyết định trong chế độ một bộ não chung.

## Số tham số học được

Policy mặc định hiện có **93.182.878 tham số học được**. Trong đó LSTM
1024→4096 có **83.918.848**, Transformer hậu-LSTM có **7.351.297**, encoder có
**795.026** và các đầu phân rã có **1.117.707**. Hợp đồng ngoài vẫn là 137 input
và 20 output.
