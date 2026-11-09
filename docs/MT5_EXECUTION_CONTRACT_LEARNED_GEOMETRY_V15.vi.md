# Contract thực thi MT5 v7 — geometry học được và tương thích marker 6

V15 thay thế contract SL/TP cố định trên đường Five bằng ba head phân loại độc
lập. Mạng vẫn dùng đúng 137 đầu vào và cầu nối 20 đầu ra để checkpoint, băng
Model 1 và EA cũ không bị thay đổi kích thước.

## Head và ánh xạ đầu ra

| Head | Lựa chọn |
| --- | --- |
| `sl` | `0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0 ATR` |
| `tp` | RR `0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0`; khoảng TP bằng `SL ATR × RR` |
| `trailing` | tắt hoặc `0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 ATR` |

Ba lựa chọn được đóng gói thuận nghịch vào row 16. Row 14 của contract mới mang
marker `7.0`; marker `6.0` vẫn dùng đúng codebook năm RR cũ, còn marker thấp hơn
`5.5` tiếp tục dùng `FixedSlAtr/FixedTpAtr` và không trailing. Row 13 vẫn là độ
trễ, row 15 là khoảng entry chờ, row 17 là tỷ lệ
đóng, row 18 là mode quản lý và row 19 là target slot.

Ứng viên bảy lựa chọn có hình học cố định vẫn phát marker `7.0`. Policy thay
các bin của hai tác nhân vào lệnh bằng bin đã dùng tạo nhãn trước khi đóng gói
row 16; bộ giải mã không lấy SL/TP mặc định của hồ sơ để thay các bin này.
Row 13 giữ được độ trễ đã chọn. Ba tác nhân quản lý giữ nguyên marker,
target và mode đóng một phần/toàn phần. Lỗi cũ hạ marker về `0` khi bật
`seven_choice_fixed_geometry_enabled` đã làm mất độ trễ và khiến EA bỏ qua
`MANAGE_POSITION`; kiểm thử `test_seven_choice_execution_parity.py` kiểm tra
vòng phát/giải mã và tính bất biến của ba tác nhân quản lý. Đây là kiểm thử
hợp đồng Python cùng logic giải mã EA hiện hành, chưa phải kết quả MT5 mới.

SL/TP ban đầu được gắn trực tiếp vào `MqlTradeRequest` cho cả market và pending.
Trailing thuộc entry đã tạo vị thế, chạy một lần tại biên M1, dùng ATR bất biến
được chụp lúc khớp entry và chỉ được dời SL theo hướng giảm rủi ro, giữ nguyên
TP. `HOLD` không gọi sửa SL/TP. Mọi sửa SL dùng `TRADE_ACTION_SLTP` và ledger
ghi controller `five_trailing_head`. Python, Model 1 và EA dùng cùng mẫu số ATR
này.

## Tương thích checkpoint

Nếu checkpoint có TP head năm bin v6, loader tạo TP head mười bin v7, sao chép
nguyên từng row RR cũ sang đúng giá trị RR tương ứng và đặt logit năm bin mới ở
`-30`. Vì vậy argmax và geometry của checkpoint cũ không đổi trước khi explicit
geometry imitation train. Nếu checkpoint thiếu geometry head, loader nối head
trung tính và đăng ký tham số với `FiveAdam`:

- SL `2.5 ATR`;
- TP `5.0 ATR` tương ứng RR `2.0`;
- trailing tắt.

Trong policy architecture v2, encoder, cây quan sát, LSTM 4096, Transformer
hậu-LSTM, action/risk/entry/management/value head và
moment của các tham số cũ được giữ nguyên. Checkpoint gốc phải được giữ bất
biến; checkpoint sau di chuyển được lưu sang output candidate mới.

## Học và ghi nhận

Log-probability và entropy của ba head chỉ tham gia PPO khi action là một trong
sáu entry. Final lifecycle R của đúng entry owner vì vậy cập nhật đồng thời
action, risk, offset, SL, TP và trailing. Hành động không vào lệnh không tạo
gradient giả cho ba head.

Execution ledger schema 17 dùng `geometry_mode=learned_factorized_atr_v2` cho
marker 7 và vẫn đọc `learned_factorized_atr_v1` cho marker 6, giữ
requested/final SL/TP, SL/TP/RR đã chọn và `proposed_trailing_atr`. Tiền tố năm
giá trị của `geometry_path_pipe` từ schema 15 được giữ nguyên; schema 16 nối thêm
`unrealized_r`, `mfe_r`, `mae_r`; schema 17 nối thêm `pending_fill_bars` chính
xác trong `geometry_entry_pipe`. Parser vẫn đọc và
tái dựng được ledger schema 15, nhưng không được biến dữ liệu thiếu thành MFE=0.
Python environment áp dụng trailing từ giá
đóng của bar hoàn tất cho bar kế tiếp; Model 1 và EA áp dụng tại biên M1 để
không dùng thứ tự high/low tương lai trong cùng bar.

## Kiểm chứng bắt buộc

- đóng gói/giải mã mọi tổ hợp head phải thuận nghịch;
- marker cũ phải giữ geometry cố định;
- checkpoint thiếu ba head phải di chuyển về trạng thái trung tính;
- PPO chỉ tính ba head trên entry;
- Python, Model 1 và EA phải giải mã cùng SL/TP/trailing;
- MetaEditor phải biên dịch EA với 0 lỗi, 0 cảnh báo;
- bằng chứng hiệu quả giao dịch vẫn cần Strategy Tester/OOS riêng, không được
  suy ra chỉ từ kiểm thử mã.
