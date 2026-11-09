# Báo cáo tương thích với bộ não và môi trường hiện tại

## Các lỗi của bản ban đầu

1. Chỉ xuất `[history_bars,5]` gồm OHLCV, không phải 137 đầu vào của mô hình.
2. Không có `spread`, nên không tái tạo được `spread_atr`, phân vị chênh lệch và trạng thái thực thi.
3. Dùng ATR Wilder trong khi dự án dùng trung bình động đơn giản của biên độ thực để khớp nền tảng giao dịch.
4. `history_norm` là chuẩn hóa riêng của công cụ, không phải `structure_aware_v4` của dự án.
5. Bộ chuyển đổi cũ nhận từng mẫu độc lập, không buộc phát lại chuỗi quá khứ để dựng ký ức.
6. Nếu cố ghi sẵn 137 giá trị, các khối vị thế/tài khoản/thực thi sẽ sai sau khi tác nhân bắt đầu hành động.
7. Nhãn MFE/MAE chỉ có nghĩa hướng mua nhưng tên không chỉ rõ hướng.
8. Mỗi mẫu nhân bản cả quá khứ và tương lai, không phù hợp mục tiêu hàng chục/hàng trăm triệu mẫu.
9. Không có dấu vân tay giao ước để phát hiện tập dữ liệu được tạo từ một phiên bản đặc trưng khác.

## Sau khi sửa

- Có lệnh `mine-five` đọc trực tiếp cấu hình và mã đặc trưng của dự án.
- Tạo băng dữ liệu môi trường một lần, mẫu chỉ giữ chỉ số.
- Mỗi tập phim dựng lại đúng `TradingEnvironment`.
- Quan sát tại điểm neo đúng 137 chiều và đi qua được bộ mã hóa năm tác nhân.
- Phần ngữ cảnh có dạng `[T,137]`, kết thúc trước điểm neo.
- Dữ liệu tương lai không được trả trong phần ngữ cảnh.
- Trạng thái động do môi trường xây sau mỗi hành động.
- ATR khớp định nghĩa dự án.
- Có dấu vân tay sơ đồ và từ chối tập dữ liệu không khớp.
- MFE/MAE ghi riêng cho hai hướng.

## Kết quả kiểm tra

Bộ kiểm tra nối trực tiếp với dự án xác nhận:

- ATR của công cụ và ATR tham chiếu của dự án trùng nhau sau vùng khởi động;
- tập phim dựng từ băng tạo chính xác observation 137 chiều của môi trường;
- ma trận năm bản của observation đi qua bộ mã hóa đầu vào năm tác nhân;
- bỏ toàn bộ dữ liệu sau điểm neo không làm thay đổi đặc trưng tại điểm neo.
