# Giao ước nối với bộ não năm tác nhân hiện tại

## Giao ước đã kiểm tra

Dự án hiện tại dùng:

- `FEATURE_SCHEMA_VERSION = 10`;
- `INPUT_COUNT = 137`;
- `MAX_POSITION_SLOTS = 5`;
- `FIVE_AGENT_COUNT = 5`;
- chuỗi ký ức có thể nhận `[T,137]` khi cả năm tác nhân nhìn cùng thị trường;
- bước quyết định dùng ma trận `[5,137]` ở bên trong bộ não.

137 giá trị gồm:

```text
29  thị trường cơ bản
40  5 ô vị thế × 8
8   tài khoản
29  thị trường bổ sung
11  thực thi
4   thời điểm
16  Wyckoff
= 137
```

## Điểm quan trọng

Công cụ **không được tự chế 137 giá trị từ OHLCV bằng một bộ công thức riêng**. Nó nhập trực tiếp `add_features`, `TradingEnvironment`, giao ước sơ đồ và chế độ chuẩn hóa từ dự án được chỉ ra bằng `--project-root`.

Nhờ vậy nếu thứ tự hoặc phiên bản đặc trưng của dự án thay đổi, dấu vân tay sơ đồ trong tập dữ liệu sẽ thay đổi. `ProjectFiveDataset` sẽ từ chối mở tập dữ liệu có dấu vân tay khác cấu hình đang chạy.

## Băng dữ liệu và trạng thái động

Băng dữ liệu chỉ giữ các cột thị trường mà `TradingEnvironment` cần. 40 giá trị vị thế, 8 giá trị tài khoản và 11 giá trị thực thi không được lưu cứng vì chúng phải thay đổi theo hành động của năm tác nhân.

Tại lúc bắt đầu một mẫu:

1. dựng môi trường từ đoạn băng đã chọn;
2. đặt trạng thái tài khoản/vị thế về trạng thái đầu tập phim;
3. phát lại phần quá khứ trước điểm neo để dựng ký ức;
4. quyết định tại điểm neo;
5. từ đó trở đi để môi trường sinh observation mới sau mỗi hành động.

## ATR

ZigZag ở chế độ `mine-five` dùng `mt5_sma`, khớp định nghĩa ATR của phần đặc trưng dự án. Chu kỳ mặc định được lấy trực tiếp từ `config.features.atr_period`.

## Nhãn tương lai

Các giá trị MFE/MAE tương lai không thuộc đầu vào. Chúng chỉ là thông tin hậu nghiệm. Công cụ ghi cả hướng mua và hướng bán để tránh lỗi dùng nhãn hướng mua cho cấu trúc giảm.

## Quy mô rất lớn

Không lưu `history` và `future` lặp lại cho từng mẫu trong chế độ `mine-five`. Với hàng trăm triệu mẫu, chỉ số mẫu nhỏ hơn rất nhiều so với việc nhân bản hàng trăm nến cho mỗi trường hợp.

Băng dữ liệu có thể được mở bằng ánh xạ bộ nhớ; các mảnh chỉ số có thể được đọc tuần tự hoặc lấy mẫu theo khối thời gian.
