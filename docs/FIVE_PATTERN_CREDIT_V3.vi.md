# Credit khám phá mẫu Five V3

## Phân quyền

Path tương lai bất biến có ba nhiệm vụ:

1. tạo nhãn phụ cho cây quan sát V4;
2. xếp hạng chất lượng tổng hợp giữa sáu entry;
3. chấm HOLD khi tài khoản trống.

Nó không còn phát scalar reward MAE/MFE cho entry. MAE/MFE thật của lệnh đã khớp thuộc riêng reward vòng đời V4, tránh tính trùng.

## HOLD khi tài khoản trống

- Có entry tổng hợp vượt `+0.10R`: HOLD bị phạt trơn, tối đa `-0.25R`.
- Cả sáu entry đều không vượt `-0.10R`: HOLD được thưởng rất nhỏ, tối đa `+0.01R`.
- Nằm giữa hai ngưỡng: không thưởng/phạt.
- Chỉ áp dụng cho anchor `flat_v1` có `positions_before=0` và `orders_before=0`.

Phần thưởng né tất cả entry xấu nhỏ hơn 25 lần hình phạt bỏ lỡ edge, nên HOLD không trở thành đường tắt hấp dẫn.

## Khám phá độc lập

Reward chỉ đánh giá các hành động đã thử. Khám phá được bảo đảm bởi ba cơ chế
độc lập:

- `five_behavior_uniform_mix: 0.15` để tiếp tục lấy mẫu nhiều hành động;
- entropy khởi đầu `five_entropy_start: 0.06`;
- loss sàn tổng xác suất sáu entry trên mọi trạng thái tài khoản trống.

Bằng chứng kỳ vọng âm và drawdown không được giảm ba cơ chế trên. Một entry đã
chứng minh xấu chỉ bị hạ xác suất qua failure replay tại đúng observation, bộ
nhớ hồi quy, tác nhân và hành động đã lưu. Nó không tạo mask toàn cục, không ép
sang HOLD và không cấm các trạng thái lân cận chưa được kiểm chứng. Rủi ro tài
khoản được xử lý bằng khối lượng/phơi nhiễm, không bằng cách tắt khám phá.

Khi chưa đủ ba edge dương đã xác minh, policy gốc phải dành tối thiểu `0.75`
tổng xác suất cho sáu entry. Sàn giảm tuyến tính từ edge thứ 3 đến edge thứ 8
và về 0 tại edge thứ 8. Đây là ràng buộc trên policy đã học, không phải sàn giả
chỉ dùng lúc lấy mẫu.
