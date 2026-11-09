# Tạo snapshot có trạng thái nhân quả cho bảy lựa chọn

`BottomTop_Counterfactual_Exporter_EA.mq5` cố ý không gọi lệnh. Vì vậy các ô
vị thế, tài khoản và phần thực thi trong 137 đầu vào luôn ở trạng thái rỗng,
dù sáu kết quả vào lệnh được tính đầy đủ từ đường giá tương lai. Bộ nhãn như
vậy phù hợp để học giá trị đối chiếu, nhưng không đủ để kiểm tra mạng có đọc
được trạng thái danh mục hay không.

## Điểm tích hợp

`scripts/augment_seven_choice_stateful_snapshots.py` là bước tiền xử lý tùy
chọn, nằm giữa tệp snapshot v2 và
`scripts/build_seven_choice_teacher_dataset.py`. Nó giữ nguyên các cột của
snapshot và chỉ thay `inputs_pipe` bằng véc-tơ 137 phần tử được ghép lại từ:

- 29 đặc trưng thị trường đầu, 29 đặc trưng thị trường phụ và 20 đặc trưng
  cuối lấy nguyên từ snapshot;
- 40 ô vị thế, 8 ô tài khoản và 10 ô phần thực thi được dựng từ lịch sử đã
  đi qua;
- ô chênh lệch mua-bán phần thực thi lấy từ nhịp giá đầu tiên của quyết định.

Giá lệch mặc định của lịch là `0,588235 ATR`, tức giá trị hiệu dụng của ô 7
trong bảng mã V16 khi cấu hình yêu cầu `0,75 ATR`; nhờ vậy lệnh tổng hợp dùng
đúng hình học mà bộ nhãn đã thực sự dùng.

Lịch hành vi cố định đi qua `HOLD` rồi sáu lựa chọn vào lệnh theo vòng. Mỗi
lựa chọn chỉ được phát khi danh mục rỗng, sau đó mô phỏng khớp lệnh chờ, cắt
lỗ, chốt lời và đóng ở cuối thời gian giữ. Trạng thái của dòng `i` được ghi
**trước** khi xử lý nhịp giá của dòng `i`; do đó nhịp giá tương lai không thể
đi vào đầu vào hiện tại. Hành động của lịch không dùng nhãn kết quả tương lai.

Ví dụ:

```powershell
python scripts/augment_seven_choice_stateful_snapshots.py `
  --snapshots mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/seven_choice_16w.tsv `
  --output outputs/seven_choice_stateful_schedule_v1.tsv

python scripts/build_seven_choice_teacher_dataset.py `
  --snapshots outputs/seven_choice_stateful_schedule_v1.tsv `
  --output outputs/seven_choice_stateful_schedule_v1.npz `
  --horizon-bars 48 --pending-expiry-bars 5 `
  --holding-horizon-from-fill
```

Tệp `.state.json` ghi số lần phát và số lần khớp của từng lựa chọn, cùng
kiểm tra nhân quả. Tệp này luôn có `production_eligible=false`: trạng thái
được tạo từ lịch hành vi tổng hợp chỉ dùng để chẩn đoán khả năng đọc trạng
thái và làm thí nghiệm đầu giá trị. Nó không thay thế bản ghi trạng thái thật
từ MT5 và không mở khóa tự học.

## Điều kiện để thành dữ liệu dùng cho chạy thật

Muốn bỏ nhãn nghiên cứu, cần xuất lại snapshot bằng cùng cấu hình đặc trưng
với hồ sơ đang chạy (chu kỳ ATR, EMA, cách ghép khung thời gian và mã hợp
đồng), đồng thời ghi trạng thái vị thế trước quyết định trong chính lần chạy
đó. Không được ghép các ô trạng thái từ `self_growing_feature_audit.csv` cũ
nếu mã hợp đồng hoặc các đặc trưng thị trường không trùng. Bộ dựng bảy lựa
chọn hiện tại vẫn giữ nguyên và tiếp tục tính nhãn tương lai riêng, sau khi đã
loại các dòng chưa đủ thời gian trưởng thành.
