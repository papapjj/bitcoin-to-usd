# Ký ức quá khứ của năm tác nhân

Ở chế độ năm tác nhân dùng chung, mỗi tác nhân có hai trạng thái riêng gồm trạng thái ẩn và ô nhớ. Hai trạng thái này được giữ liên tục từ nhịp trước sang nhịp sau và chỉ bị xóa khi tác nhân thật sự đi qua ranh giới kết thúc của một chuỗi độc lập.

Mỗi nhịp chỉ cần cung cấp quan sát hiện tại. Quá khứ không bị nối thành một véc-tơ khổng lồ; nó được nén dần vào trạng thái ẩn và ô nhớ. Vì thế hai chuỗi quá khứ khác nhau có thể tạo quyết định hiện tại khác nhau dù quan sát hiện tại giống hệt nhau.

Khi khởi động giữa một đoạn dữ liệu đã có sẵn, có thể dựng lại ký ức bằng cách phát lại các quan sát cũ theo đúng thứ tự thời gian. Hệ điều phối hỗ trợ nạp một chuỗi có dạng số bước nhân 137, hoặc số bước nhân 5 nhân 137. Nếu có lịch sử hành động thật của năm tác nhân thì có thể nạp kèm; nếu không có thì hành động giữ nguyên được dùng làm giá trị quá khứ mặc định.

Trong lúc học, một lần cập nhật dùng tối đa 64 bước liên tiếp. Trạng thái ở đầu đoạn được lấy đúng từ dấu vết lúc thu thập dữ liệu. Nếu một tác nhân kết thúc ở giữa đoạn, trạng thái ẩn, ô nhớ và hành động trước của riêng tác nhân đó được đặt lại trước bước kế tiếp; các tác nhân chưa kết thúc vẫn giữ ký ức. Nhờ vậy ký ức không chảy từ một chuỗi độc lập sang chuỗi khác.

Thao tác đặt lại trạng thái toàn hệ thống được coi là ranh giới chuỗi thật sự và sẽ xóa ký ức của cả năm tác nhân. Nó không được gọi giữa các nhịp thị trường bình thường.
