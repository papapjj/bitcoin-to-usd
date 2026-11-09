# Nâng cấp thời điểm vào và thoát lệnh cho hệ năm tác nhân

## Mục tiêu

Bản này chuyển thời điểm vào, chờ, giữ và thoát khỏi các luật chọn sẵn sang các quyết định được học từ hậu quả kinh tế của chuỗi hành động.

Mỗi tác nhân vẫn dùng chung một bộ trọng số. Mỗi tác nhân giữ trạng thái ẩn và ô nhớ riêng. Quyết định hiện tại vì vậy phụ thuộc vào quan sát hiện tại, hành động trước và ký ức tích lũy từ quá khứ.

## Nhánh thời điểm

Nhánh thời điểm có bốn lựa chọn rời rạc: 0, 1, 2, 3.

- 0: thực hiện ở nhịp nhân quả kế tiếp trong bộ mô phỏng; ở phần chạy thật là thực hiện ngay khi cầu nối nhận quyết định.
- 1 đến 3: trì hoãn thêm tương ứng 1 đến 3 nhịp.
- Trong thời gian một quyết định đang chờ, tác nhân không được phát một quyết định mới để ghi đè quyết định đang chờ.

Bộ mô phỏng làm việc theo nến nên đơn vị trì hoãn là nến. Phần chạy thật làm việc theo các nhịp giá đến nên đơn vị là nhịp giá. Hai bên cùng giữ ý nghĩa thứ tự 0 đến 3 nhưng độ phân giải thời gian không giống nhau.

## Thoát vị thế theo quyết định học được

Đường mới không dùng hai hành động đóng “tốt nhất” và “xấu nhất” để chọn hộ mạng. Hai hành động tương thích cũ này bị hạ điểm mạnh trong đường năm tác nhân mới.

Hành động quản lý vị thế được phân rã thành:

1. chọn vị thế đích;
2. chọn không làm gì, đóng một phần hoặc đóng toàn bộ;
3. nếu đóng một phần, chọn 25%, 50%, 75% hoặc 100%.

Nhờ vậy mạng phải tự học vị thế nào cần cắt và cắt bao nhiêu, thay vì để luật ngoài mạng quyết định.

## Phần thưởng trung gian

Phần thưởng trung gian chỉ sử dụng thay đổi kinh tế mới phát sinh. Không có điểm thưởng cho việc mở được lệnh, khớp được lệnh, giao dịch nhiều, giữ lệnh lâu hoặc khớp với một chỉ báo.

Các thành phần gồm:

- thay đổi giá trị đánh dấu theo thị trường;
- chi phí giao dịch thực sự phát sinh;
- phần sụt giảm mới tăng thêm, không phạt lặp lại cùng một mức sụt giảm;
- phần rủi ro vượt ngân sách;
- chi phí do các tác nhân tự triệt tiêu lẫn nhau;
- điểm cuối chuỗi dựa trên kết quả đội và mức sụt giảm lớn nhất.

Điểm riêng của năm tác nhân được trộn dần với điểm trung bình của cả đội. Nhờ vậy giai đoạn đầu vẫn có tín hiệu gán công trạng riêng, còn về sau mục tiêu hội tụ về kết quả chung.

## Tương thích trọng số cũ

Giao ước mới có một dấu riêng trong véc-tơ đầu ra. Chỉ khi dấu này tồn tại thì nhánh trì hoãn và quản lý vị thế mới được thực thi.

Véc-tơ cũ không có dấu này tiếp tục giữ hành vi cũ, vì vậy một trọng số cũ không thể bất ngờ bắt đầu đóng một phần hoặc đóng toàn bộ vị thế chỉ do các hàng trước đây được đổi ý nghĩa.

## Kiểm tra bắt buộc

Bản nâng cấp có kiểm tra cho các điều kiện sau:

- bốn lựa chọn trì hoãn đúng 0 đến 3;
- quyết định trì hoãn không bị quyết định mới ghi đè;
- mô phỏng chỉ thực hiện ở nhịp nhân quả hợp lệ;
- chọn đúng vị thế đích;
- đóng một phần làm giảm đúng khối lượng, số tiền rủi ro và tỷ lệ rủi ro của phần còn lại;
- đóng toàn bộ chỉ đóng vị thế đã chọn;
- véc-tơ tương thích cũ không kích hoạt quản lý vị thế mới;
- phần thưởng trung gian không trả điểm hai lần cho cùng một thay đổi kinh tế;
- phần sụt giảm chỉ bị phạt khi tăng thêm;
- vượt ngân sách rủi ro mới bị phạt.
