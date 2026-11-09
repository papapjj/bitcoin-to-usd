# Tích hợp vào giao diện dự án

Bộ công cụ này được gọi trực tiếp từ thẻ **Công cụ dữ liệu** của giao diện Java.
Tệp vào giao diện là `run_tool.py`, để gói có thể chạy ngay trong cây dự án mà
không cần cài thành gói Python riêng.

Luồng mặc định của mục **Khai thác tập 137** với một tệp:

1. chạy `check-five` với chính cấu hình và nguồn giá đã chọn;
2. chỉ khi bước kiểm tra trả mã thành công mới chạy `mine-five`;
3. đầu ra được ghi vào thư mục mới;
4. giao diện đọc lại `dataset_summary.json` để người dùng kiểm tra số mẫu, số
   mảnh, số chiều, số tác nhân, phiên bản sơ đồ, dấu vân tay và hợp đồng nhân
   quả.

Các thiết lập của thẻ được lưu cùng tệp thiết lập giao diện V193. Mục sửa mẫu
cấu trúc có đường dẫn riêng, nên không làm thay đổi mẫu đang dùng để khai thác
cho tới khi người dùng chọn cùng tệp đó ở mục khai thác.

## Chọn mã MT5 từ giao diện

Mục **Công cụ dữ liệu → Lấy dữ liệu MT5** không yêu cầu nhập tay một mã duy nhất nữa.

1. Bấm **Quét mã MT5** để đọc toàn bộ mã mà terminal/tài khoản hiện tại nhìn thấy qua `symbols_get()`.
2. Dùng ô lọc theo tên, mô tả hoặc nhóm máy chủ.
3. Đánh dấu từng mã ở cột **Chọn**, hoặc dùng:
   - **Chọn danh sách đang lọc**;
   - **Chọn mã đang hiện**;
   - **Bỏ chọn tất cả**.
4. Chọn khung thời gian, số thanh tối đa mỗi mã và kích thước mỗi lượt lấy.
5. Chọn một thư mục đầu ra mới/rỗng rồi chạy.

Giao diện ghi `selected_symbols.txt` để tránh giới hạn độ dài dòng lệnh Windows. Công cụ lưu mỗi mã thành một tệp `<ma>_<khung>.npz` riêng và tạo `manifest.json` tổng hợp số nến thực lấy được, trạng thái hết lịch sử và lỗi theo từng mã.

Mặc định mục lấy dữ liệu bật pipeline một nút. Sau fetch, UI tự chạy khai thác toàn
bộ symbol và kiểm tra exit code/collection. Không cần chuyển tab hoặc tự chép đường
dẫn; nếu một symbol, bước fetch hoặc bước mine lỗi, pipeline dừng ngay và không chạy
bước sau. Pipeline này chỉ tạo dữ liệu, không cập nhật trọng số mô hình.

Sau `mt5-fetch-selected`, UI đọc đúng `manifest.json` trong thư mục raw. Nó không tìm
`dataset_summary.json`. Khi bật khai thác toàn bộ symbol, UI chạy `mine-five-selected`
trên chính thư mục raw và mẫu ZigZag đã chọn. Sau bước khai thác, UI đọc
`dataset_summary.json`;
`check-five` và `smoke` không kích hoạt việc đọc artifact tập dữ liệu.

Mỗi symbol được khai thác thành tape riêng trong collection; timeline giữa các
symbol không bị ghép lại.
